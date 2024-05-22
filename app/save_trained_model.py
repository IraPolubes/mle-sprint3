import os

import pandas as pd
import numpy as np
import psycopg
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, QuantileTransformer, FunctionTransformer


load_dotenv()

TABLE_NAME = 'clean_flat_price_predict'
BUCKET_NAME = 's3://s3-student-mle-20240226-83e32c0c96/'

RANDOM_STATE = 42

def read_data():
    connection = {'sslmode': 'require', 'target_session_attrs': 'read-write'}
    postgres_credentials = {
        'host': os.getenv('DB_DESTINATION_HOST'),
        'port': os.getenv('DB_DESTINATION_PORT'),
        'dbname': os.getenv('DB_DESTINATION_NAME'),
        'user': os.getenv('DB_DESTINATION_USER'),
        'password': os.getenv('DB_DESTINATION_PASSWORD'),
    }

    connection.update(postgres_credentials)

    with psycopg.connect(**connection) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {TABLE_NAME}")
            data = cur.fetchall()
            columns = [col[0] for col in cur.description]


    df = pd.DataFrame(data, columns=columns)

    ##################### трансформации из первого проекта #############################################
    ##### Категориальные данные уже были 0 или 1 поэтому категориальных энкодеров мы не применяли
    # более детальная читска в ноутбуке потому что легче с визуализацией понимать что убирать

    df = df.drop_duplicates(subset='flat_id', keep='first')

    df['is_apartment'] = df['is_apartment'].astype(int)
    df['studio'] = df['studio'].astype(int)
    df['has_elevator'] = df['has_elevator'].astype(int)


    # remove duplicates
    feature_cols = df.columns.drop('flat_id').tolist()
    is_duplicated_features = df.duplicated(subset=feature_cols, keep=False)
    df = df[~is_duplicated_features].reset_index(drop=True)


    # fill missing values
    cols_with_nans = df.isnull().sum()
    if cols_with_nans.any():
        cols_with_nans = cols_with_nans[cols_with_nans > 0] # список имен столбцов с пропусками
        cols_with_nans.index.drop('target') if 'target' in cols_with_nans.index else cols_with_nans.index
        for col in cols_with_nans:
            if df[col].dtype in [float, int]:
                fill_value = df[col].mean()
            elif df[col].dtype == 'object':
                fill_value = df[col].mode().iloc[0]

            df[col] = df[col].fillna(fill_value)

    return df

def to_dataframe(X, ct):
    """
    This is a helper function for pipelineto reintroduce columns with feature names
    since some pipeline steps return NumPy while others require DataFrame
    """
    return pd.DataFrame(X, columns=ct.get_feature_names_out())

df = read_data()

df = df.drop_duplicates(subset='flat_id', keep='first')

df['is_apartment'] = df['is_apartment'].astype(int)
df['studio'] = df['studio'].astype(int)
df['has_elevator'] = df['has_elevator'].astype(int)


# remove duplicates

feature_cols = df.columns.drop('flat_id').tolist()
is_duplicated_features = df.duplicated(subset=feature_cols, keep=False)
df = df[~is_duplicated_features].reset_index(drop=True)


# fill missing values

cols_with_nans = df.isnull().sum()
if cols_with_nans.any():
    cols_with_nans = cols_with_nans[cols_with_nans > 0] # names of NaN containing columns
    cols_with_nans.index.drop('target') if 'target' in cols_with_nans.index else cols_with_nans.index
    for col in cols_with_nans:
        if df[col].dtype in [float, int]:
            fill_value = df[col].mean()
        elif df[col].dtype == 'object':
            fill_value = df[col].mode().iloc[0]

        df[col] = df[col].fillna(fill_value)



df.drop(['id', 'building_id', 'flat_id', 'studio', 'is_apartment'], inplace=True, axis = 1)

cat_cols = ['rooms', 'building_type_int', 'has_elevator']
num_cols = df.select_dtypes(['float', 'int']).columns.tolist()
columns_to_remove = cat_cols + ['target']
num_cols = [col for col in num_cols if col not in columns_to_remove]


threshold = 1.5
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    margin = threshold * IQR
    lower = Q1 - margin
    upper = Q3 + margin
    mask = df[col].between(lower, upper)
    df = df[mask]

df.drop(['rooms', 'living_area'], inplace=True, axis = 1)
num_cols.remove('living_area')
cols_to_remove = ('rooms')  # Convert to set for efficiency
cat_cols = [item for item in cat_cols if item not in cols_to_remove]

model = RandomForestRegressor(random_state=RANDOM_STATE)
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


n_bins = 5
degree = 3

min_max_scaler = MinMaxScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('poly_features', PolynomialFeatures(degree = degree), num_cols),
        ('quantile_features', QuantileTransformer(output_distribution='uniform', random_state=RANDOM_STATE), num_cols),  # skewed to more uniform distribution
    ],
    remainder='passthrough',
    verbose_feature_names_out=True
)


to_df_transformer = FunctionTransformer(to_dataframe, kw_args={'ct': preprocessor})


pipeline = Pipeline([
    ('scaler', min_max_scaler),
    ('to_df1', FunctionTransformer(to_dataframe, kw_args={'ct': min_max_scaler})), # return feature names each time to returned numpy
    ('preprocess', preprocessor),
    ('to_df2', FunctionTransformer(to_dataframe, kw_args={'ct': preprocessor})),
])



X_train_pipe = pipeline.fit_transform(X_train)
X_test_pipe = pipeline.transform(X_test)

# save pipeline after fitting
if not os.path.exists('pipelines'):
    os.mkdir('pipelines')
dump(pipeline, '../models/my_pipeline.joblib')

zero_features_names = [col for col in X_train_pipe.columns if ((X_train_pipe[col].min() == X_train_pipe[col].max() == 0)
                                                            and X_train_pipe[col].std() == 0)]

X_train_pipe.drop(zero_features_names, axis = 1, inplace = True)
X_test_pipe.drop(zero_features_names, axis = 1, inplace = True)


# concatenate generated features with original category and numerical scaled
model.fit(X_train_pipe, y_train)
prediction = model.predict(X_test_pipe)
dump(model, '../models/my_model.joblib')

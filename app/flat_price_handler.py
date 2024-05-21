
from joblib import load
import pandas as pd

"""
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'model_pipeline.pkl')

###
required_data = new_data[initial_feature_columns]  # Select only the columns used during training
prepared_features = pipeline.transform(required_data)

"""

def remove_outliers(df, num_cols):  # для входящих тестовых данных
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


class FlatPriceHandler:
    """Класс Handler, который обрабатывает запрос и возвращает предсказание."""
    def __init__(self):
        self.model_path = 'models/my_model.joblib'
        self.pipeline_path = 'models/my_pipeline.joblib'
        self.model = None # Объявление полей класса централизовано и заранее в одном месте для ясности
        self.pipeline = None

        # нам не нужен studio, is_apartment, rooms, living_area, оставляем только те что использованы в обучении
        self.required_model_params = [
            'floor', 'kitchen_area', 'living_area', 'total_area', 'build_year',
            'building_type_int', 'latitude', 'longitude', 'ceiling_height', 'flats_count', 'floors_total',
            'has_elevator'
        ]
        self.num_cols = list(set(self.required_model_params) - set(['has_elevator']))
        self.load_model(self, self.model_path)

    def load_model(self, path):
        """Метод загрузки модели"""
        try:
            self.model = load(self.model_pathpath)
            self.pipeline =

        except Exception as e:
            print(f"Failed to load model: {e}")

    def validate_query_params(self, query_params: dict):
        """Метод проверки есть ли все поля в запросе необходимые для модели"""
        if not isinstance(query_params, self.required_model_params):
            return False

        required_params = set(self.required_model_params)
        given_params = set(query_params.keys())
        if required_params <= given_params: # required_model_params is a subset of query_params
            return True
        else:
            print("Missing parameters: ", given_params - required_params)
            return False

    def keep_training_params(self, user_params):
        """
        Не все признаки из изначального сета использовались в тренировке модели.
        Выберем из нового пользовательского инпута только те что пригодились в тренировке.
        """

        selected_keys = [key for key in self.required_model_params if key in user_params]

        data = {key: [user_params[key]] for key in selected_keys}
        df = pd.DataFrame(data, columns=selected_keys)

        return df

    def handle(self, params):
        try:
            if not self.validate_params(params):
                response = {"Error": "Problem with parameters"}
            else:
                model_params_df = self.keep_training_params(params)
                # Используем те же трансформации на признаках из ввода что и на тренировке модели
                model_params_df[self.num_cols] = remove_outliers(model_params_df, self.num_cols)
                transformed_params_df = self.pipeline.transform(transformed_params_df)
                price_prediction = self.model.predict(transformed_params_df)
                response = {'price prediction': price_prediction}
        except Exception as e:
            return {"Error": "Problem with request"}
        else:
            return response



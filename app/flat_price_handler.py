from joblib import load
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier


def to_dataframe(X, ct):
    """ Эта функция нужна объекту пайплайна для корректной работы """
    return pd.DataFrame(X, columns=ct.get_feature_names_out())


def remove_outliers(df, num_cols):
    threshold = 1.5
    df_num = df[num_cols].copy()  # Create a copy of the numerical columns
    for col in num_cols:
        Q1 = df_num[col].quantile(0.25)
        Q3 = df_num[col].quantile(0.75)
        IQR = Q3 - Q1
        margin = threshold * IQR
        lower = Q1 - margin
        upper = Q3 + margin
        mask = df_num[col].between(lower, upper)
        df_num = df_num[mask]  # Filter only numerical columns
    return df_num



class FlatPriceHandler:
    """Класс Handler, который обрабатывает запрос и возвращает предсказание."""
    def __init__(self):
        self.model_path = '../models/my_model.joblib'
        self.pipeline_path = '../models/my_pipeline.joblib'
        self.model = RandomForestRegressor()  # Объявление полей класса централизовано и заранее в одном месте для ясности
        self.pipeline = None

        # По результатам EDA нам не нужен studio, is_apartment, rooms, living_area, оставляем только те что использованы в обучении
        self.required_model_params = [
            'floor', 'kitchen_area', 'total_area', 'build_year',
            'building_type_int', 'latitude', 'longitude', 'ceiling_height', 'flats_count', 'floors_total',
            'has_elevator'
        ]
        self.num_cols = list(set(self.required_model_params) - set(['has_elevator']))
        self.load_model()

    def load_model(self):
        """Метод загрузки модели"""
        try:
            self.model = RandomForestRegressor()
            self.model = load(self.model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")

        try:
            self.pipeline = load(self.pipeline_path)
            print("Pipeline loaded successfully.")
        except Exception as e:
            print(f"Failed to load pipeline: {e}")

    def validate_query_params(self, query_params: dict):
        """Метод проверки есть ли все поля в запросе необходимые для модели"""
        required_params = set(self.required_model_params)
        given_params = set(query_params.keys())
        if required_params <= given_params:  # required_model_params is a subset of query_params
            return True
        else:
            print("Missing parameters: ", required_params - given_params)
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
            if not self.validate_query_params(params):
                response = {"Error": "Problem with parameters"}
            else:
                model_params_df = self.keep_training_params(params)
                # Use the same transformations on input features as during model training
                model_params_df[self.num_cols] = remove_outliers(model_params_df, self.num_cols)
                transformed_params_df = self.pipeline.transform(model_params_df)  # Corrected this line
                if self.model is None:
                    raise ValueError("Model is not loaded.")

                price_prediction = self.model.predict(transformed_params_df)
                response = {'price prediction': price_prediction[0]}  # Assuming predict returns an array
        except Exception as e:
            response = {"Error": f"Problem with request: {e}"}
        return response



# создаём тестовый запрос
data = {
    'id': 802,
    'flat_id': 0.0,
    'building_id': 6220.0,
    'floor': 9,
    'is_apartment': 0,
    'kitchen_area': 9.9,
    'living_area': 19.9,
    'rooms': 1,
    'studio': 0,
    'total_area': 35.1,
    'build_year': 1965,
    'building_type_int': 6,
    'latitude': 55.717113,
    'longitude': 37.781120,
    'ceiling_height': 2.64,
    'flats_count': 84,
    'floors_total': 12,
    'has_elevator': 1,
    'target': 9500000
}

# создаём обработчик запросов для API
handler = FlatPriceHandler()

# делаем тестовый запрос
response = handler.handle(data)
print(f"Response: {response}")


import joblib

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

class FlatPriceHandler:
    """Класс Handler, который обрабатывает запрос и возвращает предсказание."""
    def __init__(self):
        self.model_path = ''
        self.model = None # до тех пор пока не проинициализирован

        # нам не нужен studio, is_apartment, rooms, living_area, оставляем только те что использованы в обучении
        self.required_model_params = [
            'floor', 'kitchen_area', 'living_area', 'total_area', 'build_year',
            'building_type_int', 'latitude', 'longitude', 'ceiling_height', 'flats_count', 'floors_total',
            'has_elevator'
        ]
        self.load_model(self, self.model_path)

    def load_model(self, path):
        """Метод загрузки модели"""
        try:
            self.model = joblib.load(path)

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
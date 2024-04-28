# импортируем библиотеку для работы со случайными числами
import random

# импортируем класс для создания экземпляра FastAPI приложения
from fastapi import FastAPI, Body
from fast_api_handler import FastApiHandler

# создаём экземпляр FastAPI приложения
app = FastAPI()
app.handler = FastApiHandler()

# обрабатываем запросы к корню приложения
@app.get("/")
def read_root():
    return {"Hello": "World"}

# обработка GET-запросов к URL /service-status
@app.get("/service-status")
def health_check():
    return {"status": "ok"}

# обрабатываем запросы к специальному пути для получения предсказания модели
# предсказание пока что в виде заглушки со случайной генерацией score
@app.get("/api/churn/{user_id}")
def get_prediction_for_item(user_id: str):
    return {"user_id": user_id, "score": random.random()}

# обрабатываем GET-запросы по пути /v1/credit/{client_id} для получения 
# ответ выдать или нет кредит на основании кредитного score клиента
@app.post("/api/credit/")
def is_credit_approved(client_id: str, model_params: dict):
    all_params = {
     "client_id": client_id,
     "model_params": model_params
     }
    user_prediction = app.handler.handle(all_params)
    score = user_prediction["predicted_credit_rating"]
    if score >= 600:
         approved = 1
    else:
         approved = 0

    return {"client_id": client_id, "approved": approved} # ответ 
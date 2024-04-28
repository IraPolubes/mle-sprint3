# импортируем библиотеку для работы со случайными числами
import random

# импортируем класс для создания экземпляра FastAPI приложения
from fastapi import FastAPI

# создаём экземпляр FastAPI приложения
app = FastAPI()

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
@app.get("/api/credit/{client_id}")
def is_credit_approved(client_id: str):
    score = random.random()
    if score > 0.8:
        approved = 1
    else:
        approved = 0
    return {"client_id": client_id, "approved": approved}
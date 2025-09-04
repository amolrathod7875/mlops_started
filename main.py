import joblib
import pandas as pd
import numpy as np 
from pydantic import BaseModel
from fastapi import FastAPI

model = joblib.load("LinearRegression_model.pkl")
scaler = joblib.load('scaler.pkl')

app = FastAPI()

class PredictionRequests(BaseModel):
    hours_studied: float

@app.get("/")
def home():
    return{
        "Message":"Welcome to Amol's Prediction Model"
    }

@app.post("/predict")
def predict(request:PredictionRequests):
    hours = request.hours_studied
    data = pd.DataFrame([[hours]],columns=['Hours_Studied'])
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    return{
        "Predicted Test Score ":prediction[0]
    }
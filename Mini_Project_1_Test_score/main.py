import joblib
import pandas as pd
import numpy as np 
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi import HTTPException
import logging

model = None
scaler = None

app = FastAPI()

@app.on_event("startup")
async def load_modal():
    try:
        global model,scaler
        model = joblib.load('LinearRegression_model.pkl')
        scaler = joblib.load('scaler.pkl')
        logging.info('Model And Scaler have been Loaded Successfully')

    except Exception as e:
        logging.error(f"Error Loading model or scaler:{e}")
        raise HTTPException(status_code = 500, detail="Error loading model")

class PredictionRequests(BaseModel):
    hours_studied: float

@app.get("/")
def home():
    return{
        "Message":"Welcome to Amol's Prediction Model"
    }

@app.post("/predict")
def predict(request:PredictionRequests):
    if model is None or scaler is None:
        logging.info("Model is not loaded")
        raise HTTPException(status_code = 503, detail="Model not loaded, please try again later")
    
    if (request.hours_studied <= 0):
        raise HTTPException(status_code = 400, detail="Hours studied should be positive")
    hours = request.hours_studied
    data = pd.DataFrame([[hours]],columns=['Hours_Studied'])
    scaled_data = scaler.transform(data)
    try:
        prediction = model.predict(scaled_data)
        logging.info(f"Prediction for {hours} hours : {prediction[0]}")
    except Exception as e:
        logging.error(f"Error During Prediction : {e}")
        raise HTTPException(status_code = 500, detail="Error during prediction")
    return{
        "Predicted Test Score ":prediction[0]
    }
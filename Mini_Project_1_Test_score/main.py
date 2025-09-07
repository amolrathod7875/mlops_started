import joblib
import pandas as pd
import numpy as np 
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi import HTTPException
import logging
from dotenv import load_dotenv
import os


model = None
scaler = None

load_dotenv()

app = FastAPI()

@app.on_event("startup")
async def load_modal():
    global model,scaler
    try:
        model_path = os.getenv("MODEL_PATH",'LinearRegression_model.pkl')
        scaler_path = os.getenv('SCALER_PATH','scaler.pkl')

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

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
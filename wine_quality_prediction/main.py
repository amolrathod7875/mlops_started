from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import numpy as np
import pandas as pd

model = joblib.load('Best_Model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI()

class WineFeatures(BaseModel):
    fixed_acidity:float
    volatile_acidity:float
    citric_acid:float
    residual_sugar:float
    chlorides:float
    free_sulfur_dioxide:float
    total_sulfur_dioxide:float
    density:float
    pH:float
    sulphate:float
    alcohol:float

@app.get("/")
def home():
    return{
        "Message":"Welcome to the Wine Quality Prediction API"
    }

@app.post("/predict")
def predict(wine:WineFeatures):
    features = np.array([
    [
        wine.fixed_acidity,
        wine.volatile_acidity,
        wine.citric_acid,
        wine.residual_sugar,
        wine.chlorides,
        wine.free_sulfur_dioxide,
        wine.total_sulfur_dioxide,
        wine.density,
        wine.pH,
        wine.sulphate,
        wine.alcohol
    ]
])
    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    return {"Predicted_quality": str(prediction[0])}


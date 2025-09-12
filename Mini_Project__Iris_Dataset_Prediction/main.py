from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np 
import pandas as pd

model = joblib.load("iris_model.pkl")
app = FastAPI()

class IrisInput(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float

species_mapping = {0: "Iris Setosa ",
                   1: "Iris Versicolor",
                   2: "Iris Virginica"}

@app.get("/")
async def home():
    return {
        "Message":"Welcome to the Iris Species Predictor API "
    }

@app.post("/predict")
async def predict_species(input_data : IrisInput):
    
    feature_name = {"Sepal_Length","Sepal_Width","Petal_Length","Petal_Width"}
    input_df = pd.DataFrame([input_data.dict()])

    prediction = model.predict(input_df) #predict using model 

    species = species_mapping.get(int(prediction[0]),"Unknown") #get species name 

    return {
        "Prediction": species
    }
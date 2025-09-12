from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("iris_model.pkl")
app = FastAPI()

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mapping for species names
species_mapping = {0: "Iris Setosa",
                   1: "Iris Versicolor",
                   2: "Iris Virginica"}

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r") as file:
        return file.read()

@app.post("/predict", response_class=JSONResponse)
async def predict_species(sepal_length: float = Form(...),
                          sepal_width: float = Form(...),
                          petal_length: float = Form(...),
                          petal_width: float = Form(...)):
    
    # Create a NumPy array from the input data in the correct order
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make the prediction
    prediction_array = model.predict(input_data)
    
    # Get the integer value of the prediction
    encoded_prediction = int(prediction_array[0])
    
    # Map the encoded value to the species name
    species = species_mapping.get(encoded_prediction, "Unknown")

    return JSONResponse(content={
        "prediction": species,
        "encoded_value": encoded_prediction
    })


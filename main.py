# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Crop Price Prediction API",
    description="API for predicting wholesale prices of agricultural crops",
    version="1.0.0"
)

# Define input model
class PredictionInput(BaseModel):
    crop_name: str
    month: int
    year: int

# Define output model
class PredictionOutput(BaseModel):
    crop_name: str
    month: int
    year: int
    predicted_price: float

# Load the dataset
df = pd.read_csv('Wholesale-Price-Index-from-2012-to-2024.csv')
df_long = df.melt(id_vars=['Crop'], var_name='Date', value_name='Price')
df_long['Date'] = pd.to_datetime(df_long['Date'], format='%B-%Y')
df_long = df_long.sort_values(by=['Crop', 'Date'])
df_long['Month'] = df_long['Date'].dt.month

# Load the saved model and components
model = tf.keras.models.load_model("my_model.keras")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler_features = joblib.load("scaler_features.pkl")

# Define sequence length (same as used during training)
SEQUENCE_LENGTH = 12

# Function to predict price for a given crop, month, and year
def predict_price(crop_name, month, year):
    # Get the historical data for the crop
    crop_data = df_long[df_long['Crop'] == crop_name]
    
    if crop_data.empty:
        raise ValueError(f"Crop '{crop_name}' not found in the dataset.")
        
    prices = crop_data['Price'].values
    
    # Ensure we have enough historical data
    if len(prices) < SEQUENCE_LENGTH:
        raise ValueError(f"Not enough historical data for {crop_name}. Available: {len(prices)} months, Required: {SEQUENCE_LENGTH} months.")
    
    # Use the most recent sequence_length months of data
    input_sequence = prices[-SEQUENCE_LENGTH:]
    input_sequence = np.array(input_sequence).reshape(1, SEQUENCE_LENGTH, 1)
    
    # Encode the crop name
    try:
        crop_label = label_encoder.transform([crop_name])[0]
    except:
        raise ValueError(f"Crop '{crop_name}' not recognized by the model.")
    
    # Normalize the additional features
    additional_features = scaler_features.transform([[crop_label, year, month]])
    year_normalized = additional_features[0][1]
    month_normalized = additional_features[0][2]
    
    # Repeat additional features for each time step
    additional_features_repeated = np.repeat([[crop_label, year_normalized, month_normalized]], 
                                           SEQUENCE_LENGTH, axis=0).reshape(1, SEQUENCE_LENGTH, 3)
    
    # Combine the input sequence and additional features
    input_combined = np.concatenate([input_sequence, additional_features_repeated], axis=2)
    
    # Make the prediction
    predicted_price_normalized = model.predict(input_combined, verbose=0)
    
    # Inverse transform the predicted price to the original scale
    predicted_price = scaler.inverse_transform(predicted_price_normalized.reshape(-1, 1))[0][0]
    
    return predicted_price

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Price Prediction API"}

@app.get("/crops", response_model=List[str])
def get_crops():
    """Get list of all available crops in the dataset"""
    return df['Crop'].unique().tolist()

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """Predict crop price based on crop name, month, and year"""
    try:
        # Validate inputs
        if input_data.month < 1 or input_data.month > 12:
            raise HTTPException(status_code=400, detail="Invalid month. Please enter a value between 1 and 12.")
            
        # You may want to adjust these year limits based on your model and data
        if input_data.year < 2012 or input_data.year > 2030:
            raise HTTPException(status_code=400, 
                               detail="Invalid year. Please enter a year between 2012 and 2030.")
        
        # Predict the price
        predicted_price = predict_price(
            input_data.crop_name, 
            input_data.month, 
            input_data.year
        )
        
        # Return the prediction
        return PredictionOutput(
            crop_name=input_data.crop_name,
            month=input_data.month,
            year=input_data.year,
            predicted_price=round(float(predicted_price), 2)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Run with: uvicorn main:app --reload
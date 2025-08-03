import os
import pandas as pd
import mlflow
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

# --- Configuration & Setup ---

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
# This is crucial for connecting to your MLflow server
load_dotenv()

# --- Pydantic Model for Input Data Validation ---

# This class defines the structure and data types for the input JSON.
# FastAPI will automatically use this for validation.
# The field names MUST match the column names from your training data.
class CarFeatures(BaseModel):
    fueltype: str
    aspiration: str
    doornumber: str
    carbody: str
    drivewheel: str
    enginelocation: str
    enginetype: str
    cylindernumber: str
    fuelsystem: str
    wheelbase: float
    carlength: float
    carwidth: float
    carheight: float
    curbweight: int
    enginesize: int
    boreratio: float
    stroke: float
    compressionratio: float
    horsepower: int
    peakrpm: int
    citympg: int
    highwaympg: int

    # Example data for easy testing with Swagger UI
    class Config:
        json_schema_extra = {
            "example": {
                "fueltype": "gas",
                "aspiration": "std",
                "doornumber": "two",
                "carbody": "convertible",
                "drivewheel": "rwd",
                "enginelocation": "front",
                "enginetype": "dohc",
                "cylindernumber": "four",
                "fuelsystem": "mpfi",
                "wheelbase": 88.6,
                "carlength": 168.8,
                "carwidth": 64.1,
                "carheight": 48.8,
                "curbweight": 2548,
                "enginesize": 130,
                "boreratio": 3.47,
                "stroke": 2.68,
                "compressionratio": 9.0,
                "horsepower": 111,
                "peakrpm": 5000,
                "citympg": 21,
                "highwaympg": 27
            }
        }

# --- FastAPI Application ---

# Initialize the FastAPI application
app = FastAPI(
    title="Car Price Prediction API",
    description="An API to predict the price of a car based on its features.",
    version="1.0.0"
)

# --- Load the MLflow Model ---

# This section loads the model from the MLflow Model Registry.
# It's placed in the global scope to be loaded only once when the app starts.
try:
    # Configure MLflow Tracking URI
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable not set.")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Define the model URI from the MLflow Model Registry using an alias
    # Format: "models:/<registered_model_name>@<alias>"
    model_name = "CarPriceModel"
    model_alias = "production"  # Use an alias like "production" or "champion"
    model_uri = f"models:/{model_name}@{model_alias}"
    
    logging.info(f"Loading model from: {model_uri}")
    # Load the model as a pyfunc model
    model = mlflow.pyfunc.load_model(model_uri)
    logging.info("Model loaded successfully.")

except Exception as e:
    logging.error(f"Failed to load model: {e}")
    # If the model fails to load, we can set it to None and handle it in the endpoint
    model = None

# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the Car Price Prediction API!"}

@app.post("/predict", tags=["Prediction"])
def predict_price(features: CarFeatures):
    """
    Predicts the car price based on input features.
    
    - **features**: A JSON object with the car's features.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please check the server logs.")

    try:
        # Convert the Pydantic model to a dictionary
        data = features.model_dump()
        # Convert the dictionary to a pandas DataFrame
        # The model expects a DataFrame as input, just like during training
        input_df = pd.DataFrame([data])
        
        # Make a prediction
        prediction = model.predict(input_df)
        
        # The prediction is likely a numpy array, so we extract the first element
        predicted_price = prediction[0]
        
        return {"predicted_price": predicted_price}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during the prediction process.")

# --- Main Block to Run the App ---

# This allows you to run the app directly using `python <filename>.py`
# It's useful for development but for production, it's better to use a process manager.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


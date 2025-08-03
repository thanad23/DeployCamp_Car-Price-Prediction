import os
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
import mlflow
import mlflow.sklearn
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Configure connection to the MLflow Tracking Server
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

#set name and create an MLflow experiment
experiment_name = "deploycamp carpp experiment"
mlflow.set_experiment(experiment_name) #set name in the MLflow UI

def main():
    """Main function to train, evaluate, and log the model."""
    logging.info("Starting the model training process...")

    # Load dataset
    data_path = "F:\Car_Price_Predictions\data\CarPrice_Assignment.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    data = pd.read_csv(data_path)
    
    # Check for missing values
    if data.isnull().values.any():
        warnings.warn("Dataset contains missing values. Consider handling them before training.")

    # Create a column transformer for preprocessing
    catCol = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel','enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']
    numCol = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
              'horsepower', 'peakrpm', 'citympg', 'highwaympg']

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),numCol),
        ('cat', OneHotEncoder(handle_unknown='ignore'), catCol)
    ],
    remainder='passthrough' # Keep other columns (like 'CarName' and 'symboling' if not dropped earlier)
    ) 
    #Create a pipeline with preprocessing and model
    rfr= RandomForestRegressor(random_state=42)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', rfr)
    ])

    # Define hyperparameter space
    hyperparam_space_params = {
            'model__n_estimators': [100, 150, 200], # Reduced number of trees
            'model__max_depth': [5, 10, 15, 20], # Reduced max depth values
            'model__min_samples_split': [2, 5, 10], # Add min_samples_split
            'model__min_samples_leaf': [1, 2, 4], # Add min_samples_leaf
            'model__max_features': ['sqrt', 'log2'], # Add max_features
            'model__bootstrap': [True, False] # Add bootstrap
        }
    
    #Define cross-validation 
    crossval = KFold(n_splits=5, shuffle=True, random_state=42)
                          
    # Split dataset into features and target variable
    X = data.drop(['price', 'symboling', 'CarName'], axis=1)
    y = data['price']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Dataset loaded and split into training and testing sets. Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

    # Start an MLflow run
    with mlflow.start_run() as run:
        run_id=run.info.run_id
        logging.info(f"MLflow run started with run ID: {run_id}")

    #hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=hyperparam_space_params,
                               cv=crossval,
                               scoring='r2',
                               n_jobs=-1) # Use all available cores


    # Train an Random Forest Regressor model
    grid_search.fit(X_train, y_train)
    logging.info("Model training completed.")

    # Make predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    r2= r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Log metrics to MLflow
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    logging.info(f"Model evaluation metrics: mae={mae}, rmse={rmse}, r2={r2}")

    #automatically determine the input and output schema
    signature = mlflow.models.infer_signature(X_train,y_pred)
    logging.info("logging model to MLflow Registry...")
    mlflow.sklearn.log_model(best_model, "model", signature=signature, registered_model_name="CarPriceModel")
    logging.info("Model logged to MLflow Registry successfully.")

if __name__ == "__main__":
    main()
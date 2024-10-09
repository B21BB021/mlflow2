import numpy as np 
import pandas as pd
import requests
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load the Dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

# Fetch the data from the URL
response = requests.get(url)

# Save the content to a temporary file
with open("BostonHousing.csv", "wb") as file:
    file.write(response.content)

data = pd.read_csv("BostonHousing.csv")

X = data.drop(columns=['medv'])
y = data['medv']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear_Regression': LinearRegression(),
    'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# MLflow experiment logging
mlflow.set_experiment("Housing_Price_Models")

best_model_name = None
best_mse = float('inf')

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train model
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Calculate MSE
        mse = mean_squared_error(y_test, predictions)

        # Log parameters, metrics, and model
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name} - MSE: {mse}")

        # Check for the best model
        if mse < best_mse:
            best_mse = mse
            best_model_name = model_name

# Log the best model separately
if best_model_name:
    with mlflow.start_run(run_name="Best_Model"):
        mlflow.log_param("best_model_type", best_model_name)
        mlflow.log_metric("best_model_mse", best_mse)

        # Log the best model
        best_model = models[best_model_name]
        mlflow.sklearn.log_model(best_model, "best_model")

        print(f"Best Model: {best_model_name} with MSE: {best_mse}")


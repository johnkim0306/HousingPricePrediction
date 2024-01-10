# prediction_script.py

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def load_models():
    xg_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                max_depth=5, alpha=10, n_estimators=100)
    rf_model = RandomForestRegressor(n_estimators=1000, max_depth=5, random_state=42)

    xg_model.load_model('housePrediction.model')
    rf_model = joblib.load('housePredictionModel.joblib')

    return xg_model, rf_model

def preprocess_input(input_data):
    # Drop the 'Price' key from the dictionary if it exists
    input_data.pop('Price', None)
    return input_data



def make_predictions(input_data, xg_model, rf_model):
    preprocessed_input = preprocess_input(input_data)
    X_input = preprocessed_input

    # Convert input data to DataFrame for XGBoost
    X_input_df = pd.DataFrame([X_input])

    # Convert DataFrame to DMatrix for XGBoost
    X_input_dmatrix = xgb.DMatrix(data=X_input_df, enable_categorical=True)

    y_pred_xg = xg_model.predict(X_input_dmatrix)
    y_pred_rf = rf_model.predict(X_input)

    return y_pred_xg, y_pred_rf

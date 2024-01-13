import pandas as pd
# import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import joblib

def load_models():
    # xg_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
    #                             max_depth=5, alpha=10, n_estimators=100)
    rf_model = RandomForestRegressor(n_estimators=1000, max_depth=5, random_state=42)

    # Load the XGBoost model using native method
    # xg_model.load_model('housePrediction.model')

    # Load the RandomForest model using joblib
    rf_model = joblib.load('housePredictionModel.joblib')

    return rf_model


def preprocess_input(input_data):
    # Drop the 'Price' key from the dictionary if it exists
    input_data.pop('Price', None)

    print('preprocess_input: ', input_data)
    print(input_data)

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    print('input_df: ', input_df)

    # Map 'Neighborhood' to numeric labels
    neighborhood_mapping = {'Urban': 2, 'Suburb': 1, 'Rural': 0}
    input_df['Neighborhood'] = input_df['Neighborhood'].map(neighborhood_mapping)

    # Extract the values from the DataFrame
    input_df_values = input_df.values
    print('input_df_values: ', input_df_values)

    return input_df_values


def make_predictions(input_data, rf_model):
    preprocessed_input = preprocess_input(input_data)
    print('swag')
    print('swag', preprocessed_input)
    X_input_rf = pd.DataFrame(preprocessed_input)

    # Print features of the XGBoost model
    # print("XGBoost Model Features:", xg_model.get_booster().feature_names)

    # Print features of the RandomForest model
    print("RandomForest Model Features:", X_input_rf.columns.tolist())

    # XGBoost prediction
    # y_pred_xg = xg_model.predict(X_input_rf)

    # Random Forest prediction
    y_pred_rf = rf_model.predict(X_input_rf)

    return y_pred_rf

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
@st.cache
def load_data():
    return pd.read_csv("insurance.csv")

# Preprocess data
def preprocess_data(data):
    data = pd.get_dummies(data, drop_first=True)  # Convert categorical to dummy variables
    return data

# Build model
def build_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# Streamlit app
def main():
    st.title("Insurance Charges Prediction App")
    
    # Load and preprocess data
    data = load_data()
    st.write("### Dataset Preview", data.head())
    data_processed = preprocess_data(data)
    
    # Split data
    X = data_processed.drop("charges", axis=1)
    y = data_processed["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = build_model(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("### Model Performance")
    st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    
    # User inputs
    st.sidebar.header("Input Features")
    user_input = {}
    for col in X.columns:
        if "age" in col:
            user_input[col] = st.sidebar.slider("Age", int(X[col].min()), int(X[col].max()), 30)
        elif "bmi" in col:
            user_input[col] = st.sidebar.slider("BMI", float(X[col].min()), float(X[col].max()), 25.0)
        elif "children" in col:
            user_input[col] = st.sidebar.slider("Children", int(X[col].min()), int(X[col].max()), 0)
        else:
            options = [0, 1]
            user_input[col] = st.sidebar.selectbox(col, options)
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Predict
    prediction = model.predict(input_df)[0]
    st.write("### Prediction Result")
    st.write(f"Predicted Insurance Charges: ${prediction:.2f}")

if __name__ == "__main__":
    main()


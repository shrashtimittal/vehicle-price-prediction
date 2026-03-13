import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Paths
MODEL_PATH = Path("models/LinearRegression.joblib")
FEATURE_DATA_PATH = Path("data/processed/vehicle_features.csv")

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Load reference dataset (for column alignment)
@st.cache_data
def load_reference_data():
    df = pd.read_csv(FEATURE_DATA_PATH)
    return df.drop(columns=["price"])

def prepare_input(form_data, reference_df):
    """Prepare features just like build_features.py"""
    df = pd.DataFrame([form_data])

    # feature engineering
    df["vehicle_age"] = 2025 - df["year"]
    df["engine_size"] = df["engine"].str.extract(r"(\d\.\d+)").astype(float)
    df["engine_type"] = df["engine"].str.extract(r"([A-Za-z]+)$")
    df["trim_feature"] = df["trim"].apply(lambda x: str(x).split()[0] if pd.notna(x) else "Unknown")
    df["fuel_simple"] = df["fuel"].apply(lambda x: "Gasoline" if "Gas" in str(x) else x)
    df["transmission_simple"] = df["transmission"].apply(lambda x: str(x).split()[0])
    df["log_mileage"] = np.log1p(df["mileage"])
    df["mileage_per_year"] = df["mileage"] / df["vehicle_age"].replace(0, 1)
    df["is_high_cylinder"] = (df["cylinders"] >= 6).astype(int)

    # ensure all columns exist
    missing_cols = [col for col in reference_df.columns if col not in df.columns]
    for col in missing_cols:
        df[col] = "Unknown"

    df = df[reference_df.columns]
    return df

# UI
st.title("🚗 Vehicle Price Prediction App")
st.markdown("Fill in the vehicle details and get a predicted market price.")

# Form
with st.form("prediction_form"):
    make = st.text_input("Make", "Jeep")
    model = st.text_input("Model", "Cherokee")
    year = st.number_input("Year", min_value=2000, max_value=2025, value=2023)
    mileage = st.number_input("Mileage", min_value=0, value=12000)
    engine = st.text_input("Engine", "2.0L I4 Turbo")
    cylinders = st.number_input("Cylinders", min_value=2, max_value=12, value=4)
    fuel = st.text_input("Fuel", "Gasoline")
    transmission = st.text_input("Transmission", "8-Speed Automatic")
    trim = st.text_input("Trim", "Limited")
    body = st.text_input("Body Type", "SUV")
    doors = st.number_input("Doors", min_value=2, max_value=5, value=4)
    drivetrain = st.text_input("Drivetrain", "Four-wheel Drive")

    submitted = st.form_submit_button("Predict Price")

if submitted:
    st.info("Loading model...")
    model_obj = load_model()
    ref_df = load_reference_data()

    input_data = {
        "name": f"{year} {make} {model} {trim}",
        "description": "N/A",
        "make": make,
        "model": model,
        "year": year,
        "engine": engine,
        "cylinders": cylinders,
        "fuel": fuel,
        "mileage": mileage,
        "transmission": transmission,
        "trim": trim,
        "body": body,
        "doors": doors,
        "exterior_color": "Unknown",
        "interior_color": "Unknown",
        "drivetrain": drivetrain,
    }

    features = prepare_input(input_data, ref_df)
    prediction = model_obj.predict(features)[0]

    st.success(f"💰 Predicted Price: **${prediction:,.2f}**")

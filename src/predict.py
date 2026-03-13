import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
MODEL_PATH = Path("models/LinearRegression.joblib")
FEATURE_DATA_PATH = Path("data/processed/vehicle_features.csv")

def load_model(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)

def prepare_input(args, reference_df):
    """
    Convert CLI args into a dataframe row consistent with training features.
    """
    data = {
        "name": f"{args.year} {args.make} {args.model} {args.trim}",
        "description": "N/A",  # placeholder
        "make": args.make,
        "model": args.model,
        "year": args.year,
        "engine": args.engine,
        "cylinders": args.cylinders,
        "fuel": args.fuel,
        "mileage": args.mileage,
        "transmission": args.transmission,
        "trim": args.trim,
        "body": args.body,
        "doors": args.doors,
        "exterior_color": "Unknown",
        "interior_color": "Unknown",
        "drivetrain": args.drivetrain,
    }
    df = pd.DataFrame([data])

    # --- Feature engineering (match build_features.py) ---
    df["vehicle_age"] = 2025 - df["year"]

    # engine features
    df["engine_size"] = df["engine"].str.extract(r"(\d\.\d+)").astype(float)
    df["engine_type"] = df["engine"].str.extract(r"([A-Za-z]+)$")

    # trim simplification
    df["trim_feature"] = df["trim"].apply(lambda x: str(x).split()[0] if pd.notna(x) else "Unknown")

    # simplified fuel
    df["fuel_simple"] = df["fuel"].apply(lambda x: "Gasoline" if "Gas" in str(x) else x)

    # simplified transmission
    df["transmission_simple"] = df["transmission"].apply(lambda x: str(x).split()[0])

    # log mileage
    df["log_mileage"] = np.log1p(df["mileage"])

    # mileage per year
    df["mileage_per_year"] = df["mileage"] / df["vehicle_age"].replace(0, 1)

    # high cylinder flag
    df["is_high_cylinder"] = (df["cylinders"] >= 6).astype(int)

    # Ensure same columns as reference
    missing_cols = [col for col in reference_df.columns if col not in df.columns]
    for col in missing_cols:
        df[col] = "Unknown"

    # Align order to reference
    df = df[reference_df.columns]

    return df

def main():
    parser = argparse.ArgumentParser(description="Predict vehicle price")
    parser.add_argument("--make", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--mileage", type=int, required=True)
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--cylinders", type=int, required=True)
    parser.add_argument("--fuel", type=str, required=True)
    parser.add_argument("--transmission", type=str, required=True)
    parser.add_argument("--trim", type=str, required=True)
    parser.add_argument("--body", type=str, required=True)
    parser.add_argument("--doors", type=int, required=True)
    parser.add_argument("--drivetrain", type=str, required=True)

    args = parser.parse_args()

    print("📂 Loading model...")
    model = load_model()

    print("📂 Loading reference dataset for feature structure...")
    reference_df = pd.read_csv(FEATURE_DATA_PATH)
    reference_df = reference_df.drop(columns=["price"])  # model input only

    print("🛠 Preparing input features...")
    input_df = prepare_input(args, reference_df)

    print("🤖 Making prediction...")
    prediction = model.predict(input_df)[0]

    print(f"💰 Predicted Price: ${prediction:,.2f}")

if __name__ == "__main__":
    main()

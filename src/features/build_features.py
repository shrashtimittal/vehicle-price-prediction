"""
Feature Engineering / Extraction with logging
"""

import os
import re
import numpy as np
import pandas as pd

# Paths
INPUT_PATH = "data/processed/vehicle_data_final.csv"
OUTPUT_PATH = "data/processed/vehicle_features.csv"


def extract_engine_size(engine_str):
    if pd.isna(engine_str):
        return np.nan
    match = re.search(r"(\d\.\d)L", str(engine_str))
    return float(match.group(1)) if match else np.nan


def extract_engine_type(engine_str):
    if pd.isna(engine_str):
        return "Unknown"
    s = str(engine_str).lower()
    if "turbo" in s:
        return "Turbo"
    elif "hybrid" in s:
        return "Hybrid"
    elif "electric" in s:
        return "Electric"
    elif "diesel" in s:
        return "Diesel"
    return "Other"


def extract_trim_features(trim_str):
    if pd.isna(trim_str):
        return "Standard"
    s = str(trim_str).lower()
    if "sport" in s:
        return "Sport"
    elif "luxury" in s:
        return "Luxury"
    elif "limited" in s:
        return "Limited"
    elif "premium" in s:
        return "Premium"
    return "Standard"


def simplify_transmission(trans_str):
    if pd.isna(trans_str):
        return "Unknown"
    s = str(trans_str).lower()
    if "auto" in s:
        return "Automatic"
    elif "manual" in s:
        return "Manual"
    return "Other"


def simplify_fuel(fuel_str):
    if pd.isna(fuel_str):
        return "Unknown"
    s = str(fuel_str).lower()
    if "gas" in s or "petrol" in s:
        return "Gasoline"
    elif "diesel" in s:
        return "Diesel"
    elif "hybrid" in s:
        return "Hybrid"
    elif "electric" in s:
        return "Electric"
    return "Other"


def build_features(df):
    df["engine_size"] = df["engine"].apply(extract_engine_size)
    df["engine_type"] = df["engine"].apply(extract_engine_type)
    df["trim_feature"] = df["trim"].apply(extract_trim_features)
    df["fuel_simple"] = df["fuel"].apply(simplify_fuel)
    df["transmission_simple"] = df["transmission"].apply(simplify_transmission)

    df["log_mileage"] = np.log1p(df["mileage"])
    df["mileage_per_year"] = df["mileage"] / df["vehicle_age"].replace(0, 1)
    df["is_high_cylinder"] = (df["cylinders"] >= 6).astype(int)

    return df


def main():
    print(f"📂 Looking for input file: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print("❌ Input file not found!")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"✅ Loaded cleaned dataset: {df.shape}")

    # Apply features
    df = build_features(df)

    # Show new columns
    print("\n🆕 New feature columns added:")
    for col in ["engine_size", "engine_type", "trim_feature", "fuel_simple",
                "transmission_simple", "log_mileage", "mileage_per_year", "is_high_cylinder"]:
        print(f" - {col}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Feature dataset saved: {OUTPUT_PATH}")
    print(f"🔎 Final dataset shape: {df.shape}")
    print(df.head(3))


if __name__ == "__main__":
    main()

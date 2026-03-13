"""
clean_data.py
Step 3: Data Cleaning & Feature Engineering
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
RAW_DATA_PATH = "data/raw/dataset.csv"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path):
    """Load raw dataset"""
    df = pd.read_csv(path)
    print(f"📂 Loaded dataset with shape: {df.shape}")
    return df


def visualize_missing(df, title, filename):
    """Plot missing values heatmap"""
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title(title)
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Missing values heatmap saved: {save_path}")


def handle_missing_values(df):
    """Handle missing values with appropriate strategies"""
    # Price: drop rows with no price (target variable)
    df = df.dropna(subset=["price"])

    # Cylinders: fill with mode
    if df["cylinders"].isnull().sum() > 0:
        df["cylinders"].fillna(df["cylinders"].mode()[0], inplace=True)

    # Mileage: fill with median
    if df["mileage"].isnull().sum() > 0:
        df["mileage"].fillna(df["mileage"].median(), inplace=True)

    # Transmission, fuel, body, doors: fill with mode
    for col in ["transmission", "fuel", "body", "doors"]:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Colors & description: fill with "Unknown"
    for col in ["exterior_color", "interior_color", "description", "engine", "trim"]:
        df[col].fillna("Unknown", inplace=True)

    return df


def feature_engineering(df):
    """Add derived features"""
    current_year = pd.Timestamp.now().year
    df["vehicle_age"] = current_year - df["year"]

    # Simplify drivetrain categories
    df["drivetrain"] = df["drivetrain"].replace(
        {
            "Four-wheel Drive": "4WD",
            "All-wheel Drive": "AWD",
            "Front-wheel Drive": "FWD",
            "Rear-wheel Drive": "RWD",
        }
    )

    return df


def visualize_distributions(df):
    """Visualize cleaned numeric distributions"""
    # Price distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], bins=50, kde=True)
    plt.title("Price Distribution (After Cleaning)")
    plt.xlabel("Price (USD)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, "price_distribution_cleaned.png"))
    plt.close()

    # Mileage distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["mileage"], bins=50, kde=True)
    plt.title("Mileage Distribution (After Cleaning)")
    plt.xlabel("Mileage (miles)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, "mileage_distribution.png"))
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    corr = df[["price", "year", "mileage", "cylinders", "doors", "vehicle_age"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.close()

    print("📈 Distribution & correlation plots saved.")


def save_cleaned(df):
    """Save final cleaned dataset"""
    save_path = os.path.join(OUTPUT_DIR, "vehicle_data_final.csv")
    df.to_csv(save_path, index=False)
    print(f"💾 Cleaned dataset saved: {save_path}")


def main():
    # Load data
    df = load_data(RAW_DATA_PATH)

    # Missing before
    print("\n🔎 Missing values before cleaning:\n", df.isnull().sum())
    visualize_missing(df, "Missing Values (Before Cleaning)", "missing_before.png")

    # Handle missing
    df = handle_missing_values(df)

    # Feature engineering
    df = feature_engineering(df)

    # Missing after
    print("\n🔎 Missing values after cleaning:\n", df.isnull().sum())
    visualize_missing(df, "Missing Values (After Cleaning)", "missing_after.png")

    # Visualizations
    visualize_distributions(df)

    # Save
    save_cleaned(df)


if __name__ == "__main__":
    main()

"""
Load dataset, check basic info, and run simple EDA
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Update dataset path (your dataset path here)
DATA_PATH = r"E:/vehicle_price_prediction/data/raw/dataset.csv"

# 🔹 Where to save processed data
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_dataset(path):
    """Load dataset from CSV"""
    print(f"📂 Loading dataset from: {path}")
    df = pd.read_csv(path)
    print("✅ Dataset loaded successfully!")
    return df


def basic_info(df):
    """Show shape, columns, null counts, sample rows"""
    print("\n🔎 Dataset Info")
    print("-" * 50)
    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nSample rows:\n", df.head(3))


def summary_stats(df):
    """Show summary statistics for numeric columns"""
    print("\n📊 Summary Statistics")
    print("-" * 50)
    print(df.describe(include="all").transpose())


def quick_visuals(df):
    """Generate quick plots for key columns"""
    print("\n📈 Generating quick visuals...")

    # Price distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], bins=50, kde=True)
    plt.title("Distribution of Vehicle Prices")
    plt.xlabel("Price (USD)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, "price_distribution.png"))
    plt.close()

    # Count of vehicles by make (top 15)
    plt.figure(figsize=(10, 5))
    top_makes = df["make"].value_counts().nlargest(15)
    sns.barplot(x=top_makes.index, y=top_makes.values)
    plt.title("Top 15 Vehicle Makes in Dataset")
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    plt.savefig(os.path.join(OUTPUT_DIR, "top_makes.png"))
    plt.close()

    print(f"✅ Visuals saved in {OUTPUT_DIR}")


def save_processed(df):
    """Save cleaned version for later steps"""
    save_path = os.path.join(OUTPUT_DIR, "vehicle_data_clean.csv")
    df.to_csv(save_path, index=False)
    print(f"💾 Processed dataset saved at: {save_path}")


def main():
    df = load_dataset(DATA_PATH)
    basic_info(df)
    summary_stats(df)
    quick_visuals(df)
    save_processed(df)


if __name__ == "__main__":
    main()

"""
Exploratory Data Analysis (EDA) with visual insights
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
CLEAN_DATA_PATH = "data/processed/vehicle_data_final.csv"
OUTPUT_DIR = "data/processed/eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    print(f"📂 Loaded cleaned dataset with shape: {df.shape}")
    return df


def price_vs_year(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="year", y="price", data=df)
    plt.xticks(rotation=45)
    plt.title("Price vs Manufacturing Year")
    plt.savefig(os.path.join(OUTPUT_DIR, "price_vs_year.png"))
    plt.close()


def price_vs_vehicle_age(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="vehicle_age", y="price", data=df, alpha=0.6)
    plt.title("Price vs Vehicle Age")
    plt.xlabel("Vehicle Age (years)")
    plt.ylabel("Price (USD)")
    plt.savefig(os.path.join(OUTPUT_DIR, "price_vs_vehicle_age.png"))
    plt.close()


def price_by_make(df):
    plt.figure(figsize=(12, 6))
    top_makes = df["make"].value_counts().nlargest(10).index
    sns.boxplot(x="make", y="price", data=df[df["make"].isin(top_makes)])
    plt.xticks(rotation=45)
    plt.title("Price Distribution by Top 10 Makes")
    plt.savefig(os.path.join(OUTPUT_DIR, "price_by_make.png"))
    plt.close()


def price_by_body(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="body", y="price", data=df)
    plt.xticks(rotation=45)
    plt.title("Price by Body Type")
    plt.savefig(os.path.join(OUTPUT_DIR, "price_by_body.png"))
    plt.close()


def price_by_fuel(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="fuel", y="price", data=df)
    plt.xticks(rotation=45)
    plt.title("Price by Fuel Type")
    plt.savefig(os.path.join(OUTPUT_DIR, "price_by_fuel.png"))
    plt.close()


def price_by_transmission(df):
    plt.figure(figsize=(12, 6))
    top_trans = df["transmission"].value_counts().nlargest(10).index
    sns.boxplot(x="transmission", y="price", data=df[df["transmission"].isin(top_trans)])
    plt.xticks(rotation=45)
    plt.title("Price by Transmission Type (Top 10)")
    plt.savefig(os.path.join(OUTPUT_DIR, "price_by_transmission.png"))
    plt.close()


def mileage_vs_price(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="mileage", y="price", data=df, alpha=0.6)
    plt.title("Mileage vs Price")
    plt.xlabel("Mileage (miles)")
    plt.ylabel("Price (USD)")
    plt.savefig(os.path.join(OUTPUT_DIR, "mileage_vs_price.png"))
    plt.close()


def main():
    df = load_data(CLEAN_DATA_PATH)

    print("📈 Generating EDA plots...")

    price_vs_year(df)
    price_vs_vehicle_age(df)
    price_by_make(df)
    price_by_body(df)
    price_by_fuel(df)
    price_by_transmission(df)
    mileage_vs_price(df)

    print(f"✅ All plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

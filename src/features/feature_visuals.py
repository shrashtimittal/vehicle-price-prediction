# src/features/feature_visuals.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
input_path = "data/processed/vehicle_features.csv"
output_dir = "reports/figures"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(input_path)

# ===============================
# 1. Histogram of Engine Size
# ===============================
plt.figure(figsize=(8,5))
sns.histplot(df["engine_size"].dropna(), bins=20, kde=True, color="steelblue")
plt.title("Distribution of Engine Size")
plt.xlabel("Engine Size (Litres)")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "engine_size_distribution.png"))
plt.close()

# ===============================
# 2. Transmission Type Distribution
# ===============================
plt.figure(figsize=(6,4))
sns.countplot(x="transmission_simple", data=df, palette="viridis")
plt.title("Transmission Type Distribution (Simplified)")
plt.xlabel("Transmission Type")
plt.ylabel("Count")
plt.savefig(os.path.join(output_dir, "transmission_simple_distribution.png"))
plt.close()

# ===============================
# 3. Price vs High Cylinder Indicator
# ===============================
plt.figure(figsize=(6,4))
sns.boxplot(x="is_high_cylinder", y="price", data=df, palette="Set2")
plt.title("Price vs High Cylinder Indicator")
plt.xlabel("High Cylinder (0 = Standard, 1 = High)")
plt.ylabel("Price ($)")
plt.xticks([0,1], ["Standard (≤4)", "High (≥6)"])
plt.savefig(os.path.join(output_dir, "price_vs_cylinder.png"))
plt.close()

print(f"✅ All feature visualizations saved in {output_dir}")

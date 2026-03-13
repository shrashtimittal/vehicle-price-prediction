"""
Train multiple ML models to predict vehicle prices
Now uses feature-engineered dataset (vehicle_features.csv)
- Prepares features
- Splits data
- Trains baseline models (Linear, RandomForest, XGBoost, LightGBM)
- Evaluates models (MAE, RMSE, R2)
- Saves results & trained models
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# Paths
FEATURE_DATA_PATH = "data/processed/vehicle_features.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    print(f"📂 Loaded feature dataset with shape: {df.shape}")
    return df


from sklearn.impute import SimpleImputer

def prepare_data(df):
    """Split features/target and build preprocessing pipeline"""
    X = df.drop("price", axis=1)
    y = df["price"]

    # Separate categorical & numeric columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"🟦 Numeric features: {numeric_cols}")
    print(f"🟨 Categorical features: {categorical_cols}")

    # Pipelines for preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return X, y, preprocessor

def evaluate_model(model, X_test, y_test, name):
    """Evaluate model performance"""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def plot_results(results):
    """Save barplot of model performances"""
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(8, 5))
    sns.barplot(x="model", y="R2", data=results_df)
    plt.title("Model Performance (R² Score)")
    plt.ylabel("R²")
    plt.savefig(os.path.join(OUTPUT_DIR, "model_performance.png"))
    plt.close()
    print("📊 Model performance plot saved.")


def main():
    # Load & prepare data
    df = load_data(FEATURE_DATA_PATH)
    X, y, preprocessor = prepare_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models to train
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.1, random_state=42),
    }

    results = []

    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(pipeline, X_test, y_test, name)
        results.append(metrics)
        print(
            f"✅ {name} -> MAE: {metrics['MAE']:.2f}, "
            f"RMSE: {metrics['RMSE']:.2f}, R²: {metrics['R2']:.4f}"
        )

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(pipeline, model_path)
        print(f"💾 Saved {name} model to {model_path}")

    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUT_DIR, "model_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n📊 Results saved to {results_path}")

    # Plot performance
    plot_results(results)

    print("\n🎉 Training complete. Check 'models/' and 'reports/' folders for outputs.")


if __name__ == "__main__":
    main()

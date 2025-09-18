# scripts/precompute_model_plots.py

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# --- Configuration ---
CACHE_DIR = "data/eda_cache/model_plots"
MODEL_ASSETS_DIR = "model"
DATASET_PATH = "data/processed/dataset_limpios_con_vd_final.csv"

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_ASSETS_DIR, exist_ok=True)

# --- Data Loading ---
print("Loading data...")
try:
    df_modelado = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATASET_PATH}")
    print("Please ensure the clustering script has been run successfully.")
    exit()

# --- Model & Data Preprocessing ---
print("Loading model and preprocessors...")
try:
    best_model = load_model(os.path.join(MODEL_ASSETS_DIR, "modelo_con_embeddings.h5"))
    with open(os.path.join(MODEL_ASSETS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_ASSETS_DIR, "label_encoders.pkl"), "rb") as f:
        le_dict = pickle.load(f)
except FileNotFoundError:
    print("Error: Model assets not found.")
    print(
        f"Please place 'modelo_con_embeddings.h5', 'scaler.pkl', and 'label_encoders.pkl' in the '{MODEL_ASSETS_DIR}' directory."
    )
    exit()

# Define features and split data
y = df_modelado["deal_score"]
cols_to_drop = [
    "deal_score",
    "price",
    "priceByArea",
    "price_per_room",
    "priceInfo.price.priceDropInfo.priceDropPercentage",
    "priceInfo.price.priceDropInfo.priceDropValue",
    "latitude",
    "longitude",
]
X = df_modelado.drop(columns=cols_to_drop, errors="ignore")

numeric_features = [
    col
    for col in X.select_dtypes(include=np.number).columns
    if col not in ["zona_cluster"]
]
categorical_features = [
    col for col in X.select_dtypes(include=["object", "category"]).columns
] + ["zona_cluster"]

# Split data (using the same random state as training)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Prepare inputs for the model
X_test_inputs = []
for col in categorical_features:
    le = le_dict[col]
    X_test_inputs.append(
        X_test[col]
        .astype(str)
        .apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        .values
    )

X_test_num_scaled = scaler.transform(X_test[numeric_features])
X_test_inputs.append(X_test_num_scaled)

# --- Model Evaluation & Plotting ---
print("Generating predictions and plots...")
y_pred = best_model.predict(X_test_inputs).flatten()

# Plot 1: Predictions vs. Actual Values
plt.figure(figsize=(10, 8))
sns.scatterplot(x=y_test.values, y=y_pred, alpha=0.3)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    color="red",
    label="Perfect Prediction",
)
plt.title("Predicciones vs. Reales (Modelo de Deep Learning)", fontsize=16)
plt.xlabel("Deal Score Real", fontsize=12)
plt.ylabel("Deal Score Predicho", fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(CACHE_DIR, "pred_vs_real.png"))
plt.close()
print("Saved 'pred_vs_real.png'")

# Plot 2: Residual Plot
residuals = y_test.values - y_pred
plt.figure(figsize=(10, 8))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.4)
plt.axhline(0, color="r", linestyle="--")
plt.title("Gráfico de Residuales", fontsize=16)
plt.xlabel("Predicción del Deal Score", fontsize=12)
plt.ylabel("Error (Residuo)", fontsize=12)
plt.grid(True)
plt.savefig(os.path.join(CACHE_DIR, "residuals.png"))
plt.close()
print("Saved 'residuals.png'")

# --- Metrics Summary (SAVE AS CSV) ---
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics_df = pd.DataFrame({"Métrica": ["MAE", "R²"], "Valor": [mae, r2]})

metrics_path = os.path.join(CACHE_DIR, "metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"Saved metrics as table: {metrics_path}")

print("Precomputation complete. All plots and tables saved.")

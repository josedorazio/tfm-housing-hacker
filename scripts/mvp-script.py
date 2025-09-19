import pandas as pd
import numpy as np
import pickle
import os

# Create directories if they don't exist
os.makedirs("data/mvp_assets", exist_ok=True)

# Load the full dataset (simulating a local file)
df_full = pd.read_csv("data/processed/dataset_limpios_con_vd_final.csv")

# --- Define Features and Target ---
y = df_full["deal_score"]
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
X = df_full.drop(columns=cols_to_drop)

# This part is just to get the feature lists for consistency
numeric_features = (
    X.select_dtypes(include=np.number).columns.drop("zona_cluster").tolist()
)
categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist() + ["zona_cluster"]

# Load pre-trained model and preprocessors (assuming they are already saved)
with open("model/label_encoders.pkl", "rb") as f:
    loaded_le_dict = pickle.load(f)
with open("model/scaler.pkl", "rb") as f:
    loaded_scaler = pickle.load(f)
from tensorflow.keras.models import load_model

loaded_model = load_model("model/modelo_con_embeddings.h5")

# --- Pre-computation Logic ---
# This replicates the notebook's core logic to identify undervalued properties
X_processed = X.copy()
for col in categorical_features:
    le = loaded_le_dict[col]
    X_processed[col] = (
        X_processed[col]
        .astype(str)
        .apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    )

X_num_scaled = loaded_scaler.transform(X_processed[numeric_features])
X_inputs = [X_processed[col].values for col in categorical_features] + [X_num_scaled]
y_pred_emb = loaded_model.predict(X_inputs)
y_pred = pd.Series(y_pred_emb.flatten(), index=y.index)

# Combine real and predicted scores with key attributes
results_df = pd.DataFrame({"deal_score_real": y, "deal_score_predicho": y_pred})
results_df["diferencia"] = (
    results_df["deal_score_predicho"] - results_df["deal_score_real"]
)
results_df = results_df.merge(
    df_full[["zona_cluster", "rooms", "size", "price", "latitude", "longitude"]],
    left_index=True,
    right_index=True,
)

# Identify the top 100 most undervalued properties for faster filtering in the app
propiedades_infravaloradas = results_df.sort_values(by="diferencia", ascending=False)
df_undervalued_examples = propiedades_infravaloradas.head(100).reset_index(drop=True)

# Save the precomputed dataframes
with open("data/mvp_assets/df_undervalued_examples.pkl", "wb") as f:
    pickle.dump(df_undervalued_examples, f)

# Also save a simplified map data with key columns
df_map_data = (
    df_full[["latitude", "longitude", "size", "price"]].copy().reset_index(drop=True)
)
with open("data/mvp_assets/df_map_data.pkl", "wb") as f:
    pickle.dump(df_map_data, f)

print("Precomputation complete. Saved assets for MVP.")

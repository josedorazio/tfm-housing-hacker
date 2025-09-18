# scripts/script_clustering.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import MarkerCluster
from folium import LinearColormap

# --- Configuration ---
load_dotenv(dotenv_path=".env.prod")
DATASET_PATH = os.getenv("df_clustering")
CACHE_DIR = "data/eda_cache/clustering_cache"
PLOTS_DIR = os.path.join(CACHE_DIR, "plots")
MAPS_DIR = os.path.join(CACHE_DIR, "maps")

# Create directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MAPS_DIR, exist_ok=True)

# --- Data Loading and Feature Engineering ---
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
df.drop(columns=["text"], inplace=True)

print("Performing feature engineering...")
lat_centro, lon_centro = 40.416775, -3.703790
df["distancia_al_centro"] = (
    np.sqrt((df["latitude"] - lat_centro) ** 2 + (df["longitude"] - lon_centro) ** 2)
    * 111.32
)

df["price_per_room"] = df["price"] / df["rooms"].replace(0, 1)
df["size_per_room"] = df["size"] / df["rooms"].replace(0, 1)
amenities_cols = [
    "exterior",
    "hasLift",
    "parkingSpace.hasParkingSpace",
    "newDevelopment",
    "has360",
    "has3DTour",
    "hasPlan",
    "hasStaging",
    "hasVideo",
    "score_final",
]
df["amenities_score"] = df[amenities_cols].sum(axis=1)

# Features for clustering
geo_features = ["latitude", "longitude", "distancia_al_centro"]
X_geo = df[geo_features]
scaler = StandardScaler()
X_geo_scaled = scaler.fit_transform(X_geo)

# --- Clustering and Precomputation ---
k_values = [6, 15, 30, 50, 200]


def precompute_clustering_results():
    """Performs clustering and saves results for each k value."""
    print("Starting clustering precomputation...")

    # Save the original dataframe with engineered features
    df.to_csv(os.path.join(CACHE_DIR, "df_with_features.csv"), index=False)

    for k in k_values:
        print(f"Processing k = {k}...")

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_column = f"zona_cluster_k{k}"
        df[cluster_column] = kmeans.fit_predict(X_geo_scaled)

        # Calculate deal score
        median_price_col = f"median_price_k{k}"
        df[median_price_col] = df.groupby(cluster_column)["priceByArea"].transform(
            "median"
        )
        deal_score_col = f"deal_score_k{k}"
        df[deal_score_col] = (df["priceByArea"] - df[median_price_col]) / df[
            median_price_col
        ]

        # Save Deal Score distribution plot
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(df[deal_score_col], kde=True, bins=30, ax=ax)
        ax.set_title(f"Deal Score Distribution (k={k})")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"deal_score_k{k}.png"))
        plt.close(fig)

        # Save the Folium map
        map_object = create_folium_map(df, k)
        map_object.save(os.path.join(MAPS_DIR, f"map_k{k}.html"))

    # New plots for k=50
    print("Precomputing additional plots for k=50...")
    df_k50 = df[df[f"zona_cluster_k{50}"].notna()]
    if not df_k50.empty:
        create_boxplot(df_k50)
        create_violinplot(df_k50)
        create_deal_score_map(df_k50)

    # Save the final dataframe with all cluster assignments and deal scores
    df.to_csv(os.path.join(CACHE_DIR, "df_with_clusters.csv"), index=False)
    print("Clustering precomputation complete.")


def create_folium_map(df, k, sample_per_cluster=500):
    """Generates a Folium map for a specific k value."""
    cluster_col = f"zona_cluster_k{k}"
    mapa = folium.Map(
        location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=11
    )
    num_clusters = df[cluster_col].nunique()
    colors = sns.color_palette("bright", n_colors=num_clusters).as_hex()
    marker_cluster = MarkerCluster().add_to(mapa)

    for cluster_id in range(num_clusters):
        df_cluster = df[df[cluster_col] == cluster_id].sample(
            n=min(sample_per_cluster, df[df[cluster_col] == cluster_id].shape[0]),
            random_state=42,
        )
        for _, row in df_cluster.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                color=colors[cluster_id],
                fill=True,
                fill_color=colors[cluster_id],
                fill_opacity=0.6,
                popup=f"Zona k={k}: {cluster_id}<br>Precio: {row['price']:,.0f} €",
            ).add_to(marker_cluster)
    return mapa


def create_boxplot(df_limpios, k=50):
    """Generates and saves the Box Plot for k=50."""
    plt.figure(figsize=(20, 8))
    sns.set(style="whitegrid")
    ax = sns.boxplot(
        x=f"zona_cluster_k{k}", y=f"deal_score_k{k}", data=df_limpios, palette="viridis"
    )
    ax.axhline(0, color="r", linestyle="--")
    plt.title(f'Distribución del "Deal Score" por Zona Geográfica (k={k})', fontsize=16)
    plt.xlabel("Zona Geográfica (Cluster)")
    plt.ylabel("Deal Score (Desviación % de la mediana)")
    if df_limpios[f"zona_cluster_k{k}"].nunique() > 25:
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "boxplot_deal_score_k50.png"))
    plt.close()


def create_violinplot(df_limpios, k=50):
    """Generates and saves the Violin Plot for k=50."""
    plt.figure(figsize=(8, 8))
    sns.set(style="whitegrid")
    ax = sns.violinplot(y=df_limpios[f"deal_score_k{k}"], color="#E0E0E0", inner=None)
    ax.axhspan(
        ymin=df_limpios[f"deal_score_k{k}"].min(), ymax=-0.15, color="green", alpha=0.1
    )
    plt.text(
        0.4,
        -0.4,
        "Buenas Ofertas\n(Más baratas que su zona)",
        horizontalalignment="center",
        color="darkgreen",
        weight="bold",
    )
    ax.axhspan(
        ymin=0.15, ymax=df_limpios[f"deal_score_k{k}"].max(), color="red", alpha=0.1
    )
    plt.text(
        0.4,
        0.6,
        "Sobreprecios\n(Más caras que su zona)",
        horizontalalignment="center",
        color="darkred",
        weight="bold",
    )
    ax.axhline(0, color="black", linestyle="--", label="Precio Mediano de la Zona")
    plt.title(
        "Distribución de las Ofertas del Mercado Inmobiliario", fontsize=16, pad=20
    )
    plt.ylabel("Calidad de la Oferta (Mejor <--> Peor)")
    plt.xlabel("Concentración de Inmuebles")
    plt.xticks([])
    plt.yticks(
        [-0.5, 0, 0.5, 1.0, 1.5], ["-50%", "Precio Justo", "+50%", "+100%", "+150%"]
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "violinplot_deal_score_k50.png"))
    plt.close()


def create_deal_score_map(df_limpios, k=50):
    """Generates and saves the Folium Deal Score Heatmap for k=50."""
    mapa_deal_score_final = folium.Map(
        location=[df_limpios["latitude"].mean(), df_limpios["longitude"].mean()],
        zoom_start=11,
    )
    min_score = -0.50
    max_score = 0.50
    colormap = LinearColormap(
        colors=["green", "yellow", "red"],
        index=[min_score, 0, max_score],
        vmin=min_score,
        vmax=max_score,
    )
    colormap.caption = "Deal Score (Verde = Buena Oferta, Rojo = Sobreprecio)"
    sample_size = min(len(df_limpios), 5000)
    df_sample = df_limpios.sample(n=sample_size, random_state=42)
    for index, row in df_sample.iterrows():
        score = row[f"deal_score_k{k}"]
        score_clamped = max(min_score, min(score, max_score))
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=colormap(score_clamped),
            fill=True,
            fill_color=colormap(score_clamped),
            fill_opacity=0.7,
            popup=f"Deal Score: {score:.2%}<br>Precio/m²: {row['priceByArea']:,.0f} €",
        ).add_to(mapa_deal_score_final)
    mapa_deal_score_final.add_child(colormap)
    mapa_deal_score_final.save(os.path.join(MAPS_DIR, f"deal_score_map_k{k}.html"))


if __name__ == "__main__":
    precompute_clustering_results()

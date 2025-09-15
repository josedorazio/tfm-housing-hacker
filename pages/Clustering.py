import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
import folium

# Initialize Streamlit page config and title
st.set_page_config(page_title="Clustering", layout="wide")
st.title("Clustering")

# Load environment variables
load_dotenv(dotenv_path=".env.prod")
dataset_path = os.getenv("DATASET_CLUSTERING_PATH")

## Access Dataset
url = dataset_path
df_limpios = pd.read_csv(url)
df_limpios.drop(columns=["text"], inplace=True)


# Features Geográficas
lat_centro, lon_centro = 40.416775, -3.703790
df_limpios["distancia_al_centro"] = (
    np.sqrt(
        (df_limpios["latitude"] - lat_centro) ** 2
        + (df_limpios["longitude"] - lon_centro) ** 2
    )
    * 111.32
)

# Features de Valor y Calidad
df_limpios["price_per_room"] = df_limpios["price"] / df_limpios["rooms"].replace(0, 1)
df_limpios["size_per_room"] = df_limpios["size"] / df_limpios["rooms"].replace(0, 1)
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
df_limpios["amenities_score"] = df_limpios[amenities_cols].sum(axis=1)

# Seleccionar y escalar variables para el clustering geográfico
geo_features = ["latitude", "longitude", "distancia_al_centro"]
X_geo = df_limpios[geo_features]
scaler = StandardScaler()
X_geo_scaled = scaler.fit_transform(X_geo)

## Generación de clusters y deal_scores con k=6, 15, 30, 50 y 200
k_values = [6, 15, 30, 50, 200]
# K values Dropbox
option = st.selectbox("Select a Cluster (k) value:", k_values)
k = option

# DISTRIBUTION VISUALIZATION
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_limpios[f"zona_cluster_k{int(k)}"] = kmeans.fit_predict(X_geo_scaled)
median_price_col = f"median_price_k{k}"
df_limpios[median_price_col] = df_limpios.groupby(f"zona_cluster_k{k}")[
    "priceByArea"
].transform("median")
deal_score_col = f"deal_score_k{k}"
df_limpios[deal_score_col] = (
    df_limpios["priceByArea"] - df_limpios[median_price_col]
) / df_limpios[median_price_col]
# Comparación visual de las distribuciones de deal_score
st.subheader(f"Distribución del Deal Score para (k={k})")
fig_hist, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df_limpios[deal_score_col], kde=True, bins=30, ax=ax)
ax.set_title(f"Distribución del Deal Score (k={k})")
st.pyplot(fig_hist)


# # Lista de k y las columnas correspondientes
# deal_score_cols = [f'deal_score_k{k}' for k in k_values]

# for i, col in enumerate(deal_score_cols):
#     sns.histplot(ax=axes[i], data=df_limpios, x=col, kde=True, bins=50)
#     axes[i].axvline(0, color='r', linestyle='--')
#     std_dev = df_limpios[col].std()
#     axes[i].set_title(f'k = {k_values[i]} (Std Dev: {std_dev:.2f})')
#     axes[i].set_xlabel('Deal Score')

# axes[0].set_ylabel('Frecuencia')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# # Display the second figure in Streamlit
# st.pyplot(fig_hist)

# def visualizar_clusters_en_mapa(df, k):
#     """
#     Genera un mapa de Folium visualizando los clusters para un valor k específico.

#     Args:
#         df (pd.DataFrame):
#         k (int): El número de clusters.
#     """
#     # Construye el nombre de la columna dinámicamente a partir de k
#     cluster_col = f'zona_cluster_k{k}'

#     # Mapa base centrado en Madrid
#     mapa = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11)

#     # Paleta de colores
#     num_clusters = df[cluster_col].nunique()
#     colors = sns.color_palette('bright', n_colors=num_clusters).as_hex()

#     # Añadir puntos al mapa
#     sample_size = df.shape[0]
#     df_sample = df.sample(n=sample_size, random_state=42)

#     for index, row in df_sample.iterrows():
#         cluster_id = row[cluster_col]

#         folium.CircleMarker(
#             location=[row['latitude'], row['longitude']],
#             radius=3,
#             color=colors[cluster_id],
#             fill=True,
#             fill_color=colors[cluster_id],
#             fill_opacity=0.6,
#             popup=f"Zona k={k}: {cluster_id}<br>Precio: {row['price']:,.0f} €"
#         ).add_to(mapa)

#     return mapa

# mapa_k50 = visualizar_clusters_en_mapa(df_limpios, 50) #Cambiar el valor a uno de los calculados antes por si quieres comparar
# mapa_k50

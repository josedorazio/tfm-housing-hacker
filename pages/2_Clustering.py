import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components

# --- Configuration ---
CACHE_DIR = "data/eda_cache/clustering_cache"
PLOTS_DIR = os.path.join(CACHE_DIR, "plots")
MAPS_DIR = os.path.join(CACHE_DIR, "maps")

st.set_page_config(page_title="Clustering", layout="wide")
st.title("Clustering")


# --- Load Precomputed Data ---
@st.cache_data
def load_k_values():
    return [6, 15, 30, 50, 200]


k_values = load_k_values()
option = st.selectbox("Select a Cluster (k) value:", k_values)

# --- Display Precomputed Results ---
st.subheader(f"Deal Score Distribution for (k={option})")
plot_path = os.path.join(PLOTS_DIR, f"deal_score_k{option}.png")
if os.path.exists(plot_path):
    st.image(
        plot_path, width="stretch", caption=f"Distribution of Deal Score for k={option}"
    )
else:
    st.error(
        "Deal score distribution plot not found. Please run the precomputation script first."
    )

# Display the precomputed Folium map
st.title("Geographic Visualization from Clusters")
st.subheader(f"Map with Clusters (k={option})")
map_path = os.path.join(MAPS_DIR, f"map_k{option}.html")
if os.path.exists(map_path):
    with open(map_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, width=1000, height=700)
else:
    st.error("Folium map not found. Please run the precomputation script first.")

# --- Display additional plots for k=50 ---
if option == 50:
    st.header("Detailed Analysis for k=50", divider=True)

    # Box Plot
    st.subheader("Deal Score by Geographic Cluster (k=50)")
    boxplot_path = os.path.join(PLOTS_DIR, "boxplot_deal_score_k50.png")
    if os.path.exists(boxplot_path):
        st.image(
            boxplot_path,
            width="stretch",
            caption="Distribution of Deal Score by Geographic Cluster",
        )
    else:
        st.warning(
            "Box plot for k=50 not found. Run the precomputation script to generate it."
        )

    # Violin Plot
    st.subheader("Overall Market Offer Distribution (k=50)")
    violinplot_path = os.path.join(PLOTS_DIR, "violinplot_deal_score_k50.png")
    if os.path.exists(violinplot_path):
        st.image(
            violinplot_path,
            width="stretch",
            caption="Distribution of Real Estate Market Offers",
        )
    else:
        st.warning(
            "Violin plot for k=50 not found. Run the precomputation script to generate it."
        )

    # Deal Score Map
    st.subheader("Deal Score Map (k=50)")
    deal_score_map_path = os.path.join(MAPS_DIR, "deal_score_map_k50.html")
    if os.path.exists(deal_score_map_path):
        with open(deal_score_map_path, "r", encoding="utf-8") as f:
            html_content_deal_score = f.read()
        components.html(html_content_deal_score, width=1000, height=700)
    else:
        st.warning(
            "Deal score map for k=50 not found. Run the precomputation script to generate it."
        )

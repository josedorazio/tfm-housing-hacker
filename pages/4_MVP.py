import streamlit as st
import pandas as pd
import pickle
import os
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

st.set_page_config(page_title="MVP - Undervalued Property Finder", layout="wide")
st.title("Find Your Next Real Estate Deal! 🏠")
st.markdown("---")


# --- Load Precomputed Assets ---
@st.cache_data
def load_precomputed_assets():
    """Loads precomputed dataframes for the MVP."""
    try:
        # Update file paths as needed
        with open("data/mvp_assets/df_undervalued_examples.pkl", "rb") as f:
            undervalued_df = pickle.load(f)
        with open("data/mvp_assets/df_map_data.pkl", "rb") as f:
            map_data_df = pickle.load(f)
        return undervalued_df, map_data_df
    except FileNotFoundError:
        st.error(
            "Precomputed assets not found. Please run the `precompute_assets_for_mvp.py` script first."
        )
        st.stop()


df_undervalued, df_map = load_precomputed_assets()

# --- User Input and Filtering ---
st.header("Search for an Undervalued Property")
st.info(
    "Select a range for the size and price, and we'll show you similar, potentially undervalued properties."
)

with st.form("search_form"):
    col1, col2 = st.columns(2)
    with col1:
        # Use st.slider for a size range
        min_size, max_size = st.slider(
            "Desired Size Range (m²)",
            min_value=20,
            max_value=500,
            value=(80, 120),  # Default range
        )
    with col2:
        # Use st.slider for a price range
        min_price, max_price = st.slider(
            "Desired Price Range (€)",
            min_value=50000,
            max_value=1500000,
            value=(250000, 450000),  # Default range
            step=1000,  # Set a step for better control
        )

    submitted = st.form_submit_button("Search for Deals 🔍")

# --- Display Results ---
if submitted:
    st.markdown("---")
    st.subheader("Your Top Recommendations")

    # Filter the precomputed undervalued dataframe directly using the slider outputs
    filtered_deals = (
        df_undervalued[
            (df_undervalued["size"] >= min_size)
            & (df_undervalued["size"] <= max_size)
            & (df_undervalued["price"] >= min_price)
            & (df_undervalued["price"] <= max_price)
        ]
        .sort_values(by="diferencia", ascending=False)
        .head(5)
    )

    if filtered_deals.empty:
        st.warning(
            "No deals found with the specified criteria. Try adjusting your search."
        )
    else:
        st.success(
            "We found some great deals for you! Here are the top 5 most undervalued properties that match your search. 👇"
        )
        st.dataframe(
            filtered_deals[
                [
                    "price",
                    "size",
                    "rooms",
                    "zona_cluster",
                    "deal_score_predicho",
                    "deal_score_real",
                ]
            ].rename(
                columns={
                    "deal_score_predicho": "Predicted Deal Score",
                    "deal_score_real": "Actual Deal Score",
                }
            )
        )

        # --- Map Visualization ---
        st.markdown("---")
        st.subheader("Map of Properties")

        # Create a comparison map
        m = folium.Map(
            location=[df_map["latitude"].mean(), df_map["longitude"].mean()],
            zoom_start=10,
        )

        # Marker cluster for all properties
        all_properties_cluster = MarkerCluster(name="All Properties").add_to(m)
        for _, row in df_map.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.4,
                tooltip=f"Price: {row['price']:,}€<br>Size: {row['size']}m²",
            ).add_to(all_properties_cluster)

        # Marker cluster for undervalued properties
        undervalued_cluster = MarkerCluster(name="Undervalued Properties").add_to(m)
        for _, row in filtered_deals.iterrows():
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=folium.Icon(color="red", icon="star"),
                tooltip=f"**🔥 Undervalued Deal 🔥**<br>Price: {row['price']:,}€<br>Size: {row['size']}m²",
            ).add_to(undervalued_cluster)

        folium.LayerControl().add_to(m)
        folium_static(m)

# pages/3_Model.py

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Model Evaluation", layout="wide")
st.title("Deep Learning Model Evaluation")
st.markdown("---")

CACHE_DIR = "data/eda_cache/model_plots"


def display_plot(filename, caption):
    """Loads and displays a precomputed plot from the cache."""
    plot_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(plot_path):
        st.image(plot_path, use_container_width=True, caption=caption)
    else:
        st.warning(
            f"Plot not found: {filename}. Please run the precomputation script first."
        )


# --- Display Performance Metrics from CSV ---
st.header("Model Performance Metrics")
st.write("A summary of the key performance metrics on the test set.")
metrics_path = os.path.join(CACHE_DIR, "metrics.csv")
if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path)
    st.table(metrics_df)
else:
    st.warning("Metrics table not found. Please run the precomputation script first.")

# --- Display Precomputed Plots ---
st.header("Prediction vs. Actual Values")
st.write(
    "This scatter plot compares the model's predictions to the actual deal scores."
)
display_plot("pred_vs_real.png", "Predicciones del Modelo vs. Valores Reales")

st.header("Residuals Plot")
st.write(
    "This plot helps to diagnose model performance. Randomly distributed residuals around zero indicate a good fit."
)
display_plot("residuals.png", "Gráfico de Residuales")

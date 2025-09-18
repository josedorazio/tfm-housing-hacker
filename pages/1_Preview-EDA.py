# pages/1_Preview-EDA.py

import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components

# --- Configuration ---
CACHE_DIR = "data/eda_cache"
PLOTS_DIR = os.path.join(CACHE_DIR, "plots")

st.set_page_config(page_title="Preprocessing", layout="wide")
st.title("Data Preview and EDA")

# --- Load Precomputed Data ---
@st.cache_data
def load_precomputed_data():
    """Loads all precomputed data from the cache directory."""
    data = {}
    data["column_overview"] = pd.read_csv(
        os.path.join(CACHE_DIR, "column_overview.csv"), index_col=0
    )
    data["head"] = pd.read_csv(os.path.join(CACHE_DIR, "head.csv"), index_col=0)
    data["tail"] = pd.read_csv(os.path.join(CACHE_DIR, "tail.csv"), index_col=0)
    data["numeric_full"] = pd.read_csv(
        os.path.join(CACHE_DIR, "numeric_full.csv"), index_col=0
    )
    data["categoric_full"] = pd.read_csv(
        os.path.join(CACHE_DIR, "categoric_full.csv"), index_col=0
    )
    data["numeric_summary"] = pd.read_csv(
        os.path.join(CACHE_DIR, "numeric_summary.csv"), index_col=0
    )

    with open(os.path.join(CACHE_DIR, "duplicates.txt"), "r") as f:
        data["duplicates"] = f.read()

    return data


precomputed_data = load_precomputed_data()

# --- Display Precomputed Results ---
st.header("Info at a Glance", divider=True)
st.write(
    f"**Shape:** Data shape not precomputed. Please run the EDA script to get this value."
)

st.subheader("Column Overview", divider=True)
st.dataframe(precomputed_data["column_overview"])

st.header("Preview of Data", divider=True)
col1, col2 = st.columns(2)
with col1:
    st.subheader("First 5 rows")
    st.dataframe(precomputed_data["head"])
with col2:
    st.subheader("Last 5 rows")
    st.dataframe(precomputed_data["tail"])

st.header("Data Quality", divider=True)
st.write(precomputed_data["duplicates"])

# 4. Quick Descriptive Statistics
option = st.selectbox(
    "Select which data to preview:",
    ("Numeric Variables", "Categorical Variables", "Full Dataset"),
)

if option == "Numeric Variables":
    st.subheader("Numeric Variables (first rows)")
    st.dataframe(precomputed_data["numeric_full"])
    st.subheader("Summary Statistics")
    st.dataframe(precomputed_data["numeric_summary"])

    st.subheader("Numeric Variable Distributions", divider=True)

    # Load list of available plots
    numeric_plots = sorted(
        [f for f in os.listdir(PLOTS_DIR) if f.endswith(("_hist.png", "_box.png"))]
    )
    numeric_vars = sorted(list(set(f.split("_")[0] for f in numeric_plots)))

    num_var = st.selectbox("Select numeric variable", numeric_vars)
    num_plot_type = st.radio("Select plot type", ("Histogram", "Boxplot"))

    plot_filename = f"{num_var}_{'hist' if num_plot_type == 'Histogram' else 'box'}.png"
    st.image(os.path.join(PLOTS_DIR, plot_filename))

elif option == "Categorical Variables":
    st.subheader("Categorical Variables (first rows)")
    st.dataframe(precomputed_data["categoric_full"])

    # Load list of available plots
    categorical_plots = sorted(
        [f for f in os.listdir(PLOTS_DIR) if f.endswith("_bar.png")]
    )
    cat_vars = sorted(list(set(f.split("_")[0] for f in categorical_plots)))

    cat_var = st.selectbox("Select a categorical variable", cat_vars)
    st.subheader(f"Distribution of {cat_var}")
    st.image(os.path.join(PLOTS_DIR, f"{cat_var}_bar.png"))

else:
    st.subheader("Full Dataset (first rows)")
    st.write(
        "Full dataset preview is not precomputed for display. Please select an option above."
    )

st.header("Correlation Heatmap")
with open(os.path.join(CACHE_DIR, "corr_heatmap.html"), "r", encoding="utf-8") as f:
    html_content = f.read()
    components.html(html_content, height=800, width=1200)

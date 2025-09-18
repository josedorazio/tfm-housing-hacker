# scripts/script_eda.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import plotly.express as px

# --- Configuration ---
# Load environment variables
load_dotenv(".env.prod")
DATASET_PATH = os.getenv("df_path")
CACHE_DIR = "data/eda_cache"
PLOTS_DIR = os.path.join(CACHE_DIR, "plots")

# Create directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Data Loading ---
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# --- Precomputation Functions ---


def precompute_tables():
    """Computes and saves dataframes to CSV files."""
    print("Precomputing tables...")
    # 1. Column Overview
    column_overview = pd.DataFrame(
        {"dtype": df.dtypes, "missing": df.isna().sum(), "unique_values": df.nunique()}
    )
    column_overview.to_csv(os.path.join(CACHE_DIR, "column_overview.csv"))

    # 2. Head and Tail
    df.head().to_csv(os.path.join(CACHE_DIR, "head.csv"))
    df.tail().to_csv(os.path.join(CACHE_DIR, "tail.csv"))

    # 3. Duplicates
    with open(os.path.join(CACHE_DIR, "duplicates.txt"), "w") as f:
        f.write(f"Number of duplicate rows: {df.duplicated().sum()}")

    # 4. Numeric and Categorical data
    df_numeric = df.select_dtypes(include="number")
    df_categoric = df.select_dtypes(exclude="number")

    # Save a sample of the dataframes
    df_numeric.head().to_csv(os.path.join(CACHE_DIR, "numeric_full.csv"))
    df_categoric.head().to_csv(os.path.join(CACHE_DIR, "categoric_full.csv"))

    # Summary statistics for numeric data
    df_numeric.describe().T.to_csv(os.path.join(CACHE_DIR, "numeric_summary.csv"))


def precompute_plots():
    """Generates and saves plot images."""
    print("Precomputing plots...")
    df_numeric = df.select_dtypes(include="number")
    df_categoric = df.select_dtypes(exclude="number")

    # Numeric plots (Histograms and Boxplots)
    for col in df_numeric.columns:
        # Histogram
        plt.figure(figsize=(8, 4))
        sns.histplot(df_numeric[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{col}_hist.png"))
        plt.close()

        # Boxplot
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df_numeric[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{col}_box.png"))
        plt.close()

    # Categorical plots (Bar charts)
    for col in df_categoric.columns:
        # Drop columns with too many unique values for a bar chart
        if df_categoric[col].nunique() < 50:
            plt.figure(figsize=(10, 6))
            df_categoric[col].value_counts().head(20).plot(kind="bar")
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{col}_bar.png"))
            plt.close()

    # Correlation Heatmap
    print("Generating correlation heatmap...")
    corr = df_numeric.corr()
    fig = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto"
    )
    fig.update_layout(width=1200, height=800)
    fig.write_html(os.path.join(CACHE_DIR, "corr_heatmap.html"))


# --- Main Execution ---
if __name__ == "__main__":
    precompute_tables()
    precompute_plots()
    print("Precomputation complete. Files saved to 'data/eda_cache'.")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

st.set_page_config(page_title="Preprocessing", layout="wide")
st.title("Data Preview and EDA")

# Load environment variables
load_dotenv(".env.prod")
dataset_path = os.getenv("DATASET_PATH")
url = dataset_path
df = pd.read_csv(url)

# 1. Dataset Info at a Glance
st.header("Info at a Glance", divider=True)
st.write(f"**Shape:** ROWS {df.shape[0]}  ×  Columns{df.shape[1]} columns")

st.subheader("Column Overview", divider=True)
st.dataframe(
    pd.DataFrame(
        {"dtype": df.dtypes, "missing": df.isna().sum(), "unique_values": df.nunique()}
    )
)

# 2. Head & Tail
st.header("Preview of Data", divider=True)
col1, col2 = st.columns(2)
with col1:
    st.subheader("First 5 rows")
    st.dataframe(df.head())
with col2:
    st.subheader("Last 5 rows")
    st.dataframe(df.tail())

# 3. Duplicates
st.header("Data Quality", divider=True)
st.write(f"Number of duplicate rows: {df.duplicated().sum()}")

# 4. Quick Descriptive Statistics
df_numeric = df.select_dtypes(include="number")
df_categoric = df.select_dtypes(exclude="number")
cols_to_drop = [
    "url",
    "externalReference",
    "thumbnail",
    "description",
    "suggestedTexts.title",
    "suggestedTexts.subtitle",
    "notes",
    "address",
    "country",
    "province",
    "operation",
    "priceInfo.price.currencySuffix",
    "topNewDevelopment",
]
df_categoric.drop(columns=cols_to_drop, inplace=True)


option = st.selectbox(
    "Select which data to preview:",
    ("Numeric Variables", "Categorical Variables", "Full Dataset"),
)

if option == "Numeric Variables":
    st.subheader("Numeric Variables (first rows)")
    st.dataframe(df_numeric.head())
    st.subheader("Summary Statistics")
    st.dataframe(df_numeric.describe().T)
    st.subheader("Numeric Variable Distributions", divider=True)
    num_var = st.selectbox("Select numeric variable", df_numeric.columns)
    num_plot_type = st.radio("Select plot type", ("Histogram", "Boxplot"))
    fig, ax = plt.subplots(figsize=(8, 4))
    if num_plot_type == "Histogram":
        sns.histplot(df_numeric[num_var], kde=True, ax=ax)
    else:
        sns.boxplot(x=df_numeric[num_var], ax=ax)
    st.pyplot(fig)

elif option == "Categorical Variables":
    st.subheader("Categorical Variables (first rows)")
    st.dataframe(df_categoric.head())

    # Dropdown to select specific categorical variable
    cat_var = st.selectbox("Select a categorical variable", df_categoric.columns)
    # Display value counts as a bar chart
    st.subheader(f"Distribution of {cat_var}")
    st.bar_chart(df[cat_var].value_counts())

else:
    st.subheader("Full Dataset (first rows)")
    st.dataframe(df.head())


st.header("Correlation Heatmap")

# Compute correlation matrix
corr = df_numeric.corr()

import plotly.express as px

# Optionally let user pick numeric columns
cols = st.multiselect(
    "Select numeric columns for correlation",
    df_numeric.columns,
    default=df_numeric.columns.tolist(),
)

# Correlation matrix
corr = df_numeric[cols].corr()

# Interactive heatmap
fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto")

fig.update_layout(
    width=1200,  # width in pixels
    height=800,  # height in pixels
    margin=dict(l=50, r=50, t=50, b=50),
)
st.plotly_chart(fig)

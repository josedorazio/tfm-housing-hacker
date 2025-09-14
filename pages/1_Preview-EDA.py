import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Preprocessing", layout="wide")
st.title("Data Preview and EDA")


## Add This
url = "https://raw.githubusercontent.com/guille1006/TFM/refs/heads/main/data2/total_data.csv"
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

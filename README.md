# 🏡 TFM Housing Hacker

Exploring housing data with Machine Learning and interactive visualization.  
This repository contains the full workflow behind my Master's Thesis (TFM):  
from data cleaning and model training (Jupyter notebooks) to an interactive Streamlit app for exploration and predictions.

## 📌 Overview

The goal of this project is to analyze housing data, build predictive models, and provide an interactive tool that allows users to explore insights and test predictions.

Main components:

- **Data preprocessing & feature engineering** (Jupyter notebooks).
- **Machine Learning models** for housing-related predictions.
- **Streamlit application** for visualization and user interaction.

## 📂 Project Structure

tfm-housing-hacker/

```
├── README.md
├── app.py
├── data
│   ├── _20250910_dataset_features_model.csv
│   ├── _20250910_features_df_1.csv
│   ├── _20250910_features_df_2.csv
│   ├── datos_Idealista_limpios V.2.csv
│   ├── datos_Idealista_limpios.csv
│   ├── datos_Idealista_limpios_V_final.csv
│   ├── df_limpios_con_vd V.2.csv
│   ├── df_limpios_con_vd.csv
│   ├── df_limpios_con_vd_V_final.csv
│   ├── dict.txt
│   ├── dist2024
│   │   ├── dist2024.cpg
│   │   ├── dist2024.dbf
│   │   ├── dist2024.prj
│   │   ├── dist2024.shp
│   │   └── dist2024.shx
│   ├── dist2024.zip
│   ├── distr24.xlsx
│   ├── features_df_1_20250909.csv
│   ├── features_df_1_20250910.csv
│   ├── modelo-features-embeddings
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── training_args.bin
│   │   └── vocab.txt
│   └── vocab.txt
├── models
│   ├── label_encoders.pkl
│   ├── modelo_con_embeddings.h5
│   └── scaler.pkl
├── notebooks
├── pages
├── requirements.txt
├── setup.cfg
└── util
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/tfm-housing-hacker.git
cd tfm-housing-hacker
```

Install dependencies:

```
pip install -r requirements.txt
```

## 🚀 Usage

### Run the Streamlit app

```
streamlit run app.py
```

## Explore the notebooks

Navigate to the `notebooks/` folder and open them with Google Colabs:

- 01_preprocessing.ipynb → Data cleaning and feature engineering.
- 02_embedding.ipynb → Dataset enrichment
- 03_clustering.ipynb → Objetive Variable training and clusterization
- 04_model_DL.ipynb → Model training and evaluation.

## 📊 Results

(to be updated once final results and screenshots are ready)

Model performance metrics.

Example visualizations from the Streamlit app.

## 📄 License

This project is part of a team Master's Thesis.
Feel free to explore and adapt the code for educational purposes.

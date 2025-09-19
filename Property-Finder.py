import streamlit as st

st.set_page_config(page_title="TFM Housing Hacker", page_icon="🏡", layout="wide")

st.title("🏡 Análisis de Precios de Viviendas")

# st.sidebar.title("Navegación")

# --- Planteamiento del Problema ---
st.header("El Desafío: Navegar en un Mercado Competitivo")
st.markdown(
    """
Encontrar una propiedad infravalorada en un mercado inmobiliario competitivo es un gran reto. 
Sin información basada en datos, los inversores y compradores suelen tener dificultades para 
identificar las oportunidades con mayor potencial de retorno de inversión.  
Nuestra aplicación aborda este problema aprovechando un modelo predictivo para puntuar las propiedades según su valor potencial.
"""
)

# --- Nuestra Solución ---
st.header("La Solución: Puntuación Basada en Datos")
st.markdown(
    """
Esta aplicación te ayuda a filtrar el ruido.  
Analiza miles de anuncios de propiedades para identificar aquellas con una diferencia significativa entre su valor predicho y su precio real.  
Este "deal score" te permite localizar rápidamente las oportunidades más prometedoras.
"""
)

# --- Cómo Usar ---
st.subheader("Cómo Utilizar Esta Herramienta")
st.markdown(
    """
1.  **Define tu Búsqueda:** Usa los deslizadores de abajo para establecer el tamaño y rango de precios deseados.  
2.  **Busca Oportunidades:** Haz clic en el botón "Buscar Oportunidades" para filtrar los resultados.  
3.  **Analiza los Datos:** La tabla mostrará las 10 propiedades más infravaloradas que coincidan con tus criterios.  
    Presta especial atención al `Predicted Deal Score` y `Actual Deal Score` para entender el valor potencial de cada propiedad.  
4.  **Visualiza en el Mapa:** El mapa mostrará todas las propiedades en contexto, destacando las oportunidades infravaloradas con marcadores de estrella roja para una visualización sencilla.  
"""
)

# --- Marcador de posición para el resto del código de tu app.py ---
# Aquí iría el resto de la lógica de tu aplicación, incluyendo los deslizadores y la visualización en el mapa.
# Por ejemplo:
# st.header("Buscar una Oportunidad Potencial")
# # ... tus formularios, deslizadores y código del mapa ...

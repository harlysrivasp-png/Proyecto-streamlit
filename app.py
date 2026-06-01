import streamlit as st

from inicio import cargar_archivo
from eda import eda
from entrenamiento import entrenamiento
from preprocesamiento import preprocesamiento
from prediccion import prediccion


# ==========================================================
# OCULTAR ELEMENTOS DE STREAMLIT
# ==========================================================

def hide_elements():
    configuracion = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(configuracion, unsafe_allow_html=True)


# ==========================================================
# FUNCIÓN PRINCIPAL DE LA APP
# ==========================================================

def app():

    hide_elements()

    st.sidebar.title("Menú de navegación")

    nombres_paginas = {
        "Inicio": cargar_archivo,
        "Análisis Exploratorio de Datos": eda,
        "Preprocesamiento de Datos": preprocesamiento,
        "Entrenamiento y Prueba": entrenamiento,
        "Predicción": prediccion
    }

    pagina_seleccionada = st.sidebar.selectbox(
        "Escoja una página",
        list(nombres_paginas.keys())
    )

    st.sidebar.divider()

    nombres_paginas[pagina_seleccionada]()


# ==========================================================
# EJECUTAR APP
# ==========================================================

app()


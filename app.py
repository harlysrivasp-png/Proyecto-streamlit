import streamlit as st

from inicio import cargar_archivo
from eda import eda
from entrenamiento import entrenamiento
from preprocesamiento import preprocesamiento
from prediccion import prediccion


def hide_elements():
    configuracion = """
    <style>
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    header {visibility:hidden;}
    </style>
    """
    st.markdown(configuracion, unsafe_allow_html=True)


def app():
    hide_elements()

    nombresPaginas = {
        "Inicio": cargar_archivo,
        "Análisis Exploratorio de Datos": eda,
        "Preprocesamiento de Datos": preprocesamiento,
        "Entrenamiento y Prueba": entrenamiento,
        "Predicción": prediccion
    }

    nombres_Paginas = st.sidebar.selectbox(
        "Escoja una página",
        list(nombresPaginas.keys())
    )

    st.sidebar.divider()

    nombresPaginas[nombres_Paginas]()
import streamlit as st

from escenario1 import escenario1
from escenario2 import escenario2
from escenario3 import escenario3
from escenario4 import escenario4


def prediccion():

    st.markdown("# 🔮 Predicción de Abandono Estudiantil")

    st.markdown("""
    En este módulo se realizan predicciones de abandono estudiantil utilizando
    modelos de Machine Learning previamente entrenados. Puede realizar predicciones
    individuales o masivas según el escenario seleccionado.
    """)

    opcion = st.sidebar.selectbox(
        "Seleccione el escenario de predicción",
        [
            "Escenario 1: Predicción manual con modelo fijo",
            "Escenario 2: Predicción masiva con modelo fijo",
            "Escenario 3: Cargar modelo y predicción manual",
            "Escenario 4: Cargar modelo y archivo CSV"
        ]
    )

    if opcion == "Escenario 1: Predicción manual con modelo fijo":
        escenario1()

    elif opcion == "Escenario 2: Predicción masiva con modelo fijo":
        escenario2()

    elif opcion == "Escenario 3: Cargar modelo y predicción manual":
        escenario3()

    elif opcion == "Escenario 4: Cargar modelo y archivo CSV":
        escenario4()
prediccion()
import streamlit as st
from escenario1 import escenario_1
from escenario2 import escenario_2
from escenario3 import escenario_3
from escenario4 import escenario_4

def prediccion():
    st.markdown("#:chart_with_upwards_trend: Predicción de Abandono de Clientes")
    st.markdown("""
    En esta sección, contarás con 4 escenarios para realizar predicciones distintas y conocer  si los clientes abandonan el negocio.
    ** Instrucciones:**
    -** Escenario_1:** Realiza predicciones manuales de los clientes que abandonarán a partir de un modelo Predice la probabilidad de abandono de los clientes utilizando un modelo de regresión logística.
    -** Escenario_2:** Utiliza un modelo de árbol de decisión para identificar los factores
    -** Escenario_3:** Aplica un modelo de bosque aleatorio para mejorar la precisión de las predicciones.
    -** Escenario_4:** Implementa un modelo de redes neuronales para capturar relaciones complejas entre las variables y predecir el abandono de los clientes.
    """)
    st.divider()
    opciones = st.radio(
    "Seleccione una Escenario:",
    ["Escenario 1", "Escenario 2", "Escenario 3", "Escenario 4"],horizontal=True)

    if opciones=='Escenario 1':
         escenario_1()
    elif opciones=='Escenario 2':
        escenario_2()
    elif opciones=='Escenario 3':
        escenario_3()
    else:
        escenario_4()    

prediccion()

    
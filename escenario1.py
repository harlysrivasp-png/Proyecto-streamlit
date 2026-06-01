import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def escenario1():
    st.markdown("## Escenario 1: Predicción Manual de Abandono de Estudiantes")

    st.markdown("""
    En este escenario se presenta un formulario para ingresar manualmente los datos de un estudiante.
    A partir de estos datos, el modelo entrenado predice si el estudiante presenta riesgo de abandono.
    """)

    # Cargar modelo entrenado
    with open("modelo_entrenado.pkl", "rb") as archivo:
        carga_modelo = pickle.load(archivo)

    datos_dicc = {}

    # Variables categóricas
    program = st.sidebar.selectbox(
        "¿En qué programa está matriculado?",
        ["ASS", "TRF", "TOC"]
    )
    datos_dicc["program"] = program

    gender = st.sidebar.selectbox(
        "Género",
        ["F", "M"]
    )
    datos_dicc["gender"] = gender

    employed = st.sidebar.selectbox(
        "¿Tiene empleo?",
        ["Si", "No"]
    )
    datos_dicc["employed"] = employed

    # Variables numéricas
    age = st.sidebar.number_input(
        "Edad",
        min_value=15,
        max_value=80,
        value=25
    )
    datos_dicc["age"] = age

    socioeconomic_level = st.sidebar.selectbox(
        "Nivel socioeconómico",
        [1, 2, 3, 4]
    )
    datos_dicc["socioeconomic_level"] = socioeconomic_level

    semester = st.sidebar.selectbox(
        "Semestre",
        [1, 2, 3, 4]
    )
    datos_dicc["semester"] = semester

    login_frequency = st.sidebar.number_input(
        "Frecuencia de ingreso al LMS",
        min_value=0.0,
        max_value=1000.0,
        value=100.0
    )
    datos_dicc["login_frequency"] = login_frequency

    forum_participation = st.sidebar.number_input(
        "Participación en foros",
        min_value=0.0,
        max_value=1000.0,
        value=100.0
    )
    datos_dicc["forum_participation"] = forum_participation

    task_submissions = st.sidebar.number_input(
        "Número de tareas enviadas",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )
    datos_dicc["task_submissions"] = task_submissions

    late_submissions = st.sidebar.number_input(
        "Número de tareas entregadas tarde",
        min_value=0.0,
        max_value=50.0,
        value=2.0
    )
    datos_dicc["late_submissions"] = late_submissions

    connection_time = st.sidebar.number_input(
        "Tiempo de conexión",
        min_value=0.0,
        max_value=5000.0,
        value=500.0
    )
    datos_dicc["connection_time"] = connection_time

    resource_views = st.sidebar.number_input(
        "Visualizaciones de recursos",
        min_value=0.0,
        max_value=5000.0,
        value=500.0
    )
    datos_dicc["resource_views"] = resource_views

    final_grade = st.sidebar.number_input(
        "Calificación final",
        min_value=0.0,
        max_value=5.0,
        value=3.5,
        step=0.1
    )
    datos_dicc["final_grade"] = final_grade

    # Crear dataframe
    dataset_nuevo = pd.DataFrame(datos_dicc, index=[0])

    st.markdown("### Datos ingresados para la predicción")
    st.write(dataset_nuevo)

    # Codificación categórica
    for columna in dataset_nuevo.select_dtypes(include=["object"]).columns:
        encoder = LabelEncoder()
        dataset_nuevo[columna] = encoder.fit_transform(dataset_nuevo[columna])

    # Escalado de variables numéricas
    columnas_numericas = [
        "age",
        "socioeconomic_level",
        "semester",
        "login_frequency",
        "forum_participation",
        "task_submissions",
        "late_submissions",
        "connection_time",
        "resource_views",
        "final_grade"
    ]

    scaler = StandardScaler()
    dataset_nuevo[columnas_numericas] = scaler.fit_transform(
        dataset_nuevo[columnas_numericas]
    )

    st.markdown("### Datos procesados para el modelo")
    st.write(dataset_nuevo)

    # Realizar predicción
    prediccion_modelo = carga_modelo.predict(dataset_nuevo)
    prediction_proba_modelo = carga_modelo.predict_proba(dataset_nuevo)

    prediccion = prediccion_modelo[0]
    probabilidad_no = prediction_proba_modelo[0][0] * 100
    probabilidad_si = prediction_proba_modelo[0][1] * 100

    st.subheader("Resultado de la predicción")

    if prediccion == 1:
        st.error(
            f"El estudiante tiene una probabilidad del {probabilidad_si:.2f}% de abandonar la formación."
        )
    else:
        st.success(
            f"El estudiante tiene una probabilidad del {probabilidad_no:.2f}% de permanecer en la formación."
        )

    st.subheader("Probabilidades del modelo")

    df_resultado = pd.DataFrame({
        "Clase": ["No abandona", "Sí abandona"],
        "Probabilidad (%)": [probabilidad_no, probabilidad_si]
    })

    st.write(df_resultado)



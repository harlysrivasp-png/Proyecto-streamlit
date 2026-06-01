import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder


def escenario3():

    st.markdown("## Escenario 3: Cargar Modelo y Realizar Predicción Manual")

    st.markdown("""
    En este escenario el usuario puede cargar un modelo entrenado en formato `.pkl`
    y luego ingresar manualmente los datos de un estudiante para predecir si presenta
    riesgo de abandono académico.
    """)

    # ======================================================
    # SUBIR MODELO
    # ======================================================

    modelo_entrenado = st.file_uploader(
        "Suba el modelo entrenado en formato .pkl",
        type=["pkl"]
    )

    if modelo_entrenado is None:
        st.info("Por favor, suba un modelo entrenado en formato .pkl para continuar.")
        st.stop()

    # ======================================================
    # CARGAR MODELO
    # ======================================================

    try:
        carga_modelo = pickle.load(modelo_entrenado)
        st.success("Modelo cargado correctamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()

    # ======================================================
    # FORMULARIO
    # ======================================================

    st.sidebar.markdown("## Datos del estudiante")

    datos_dicc = {}

    datos_dicc["program"] = st.sidebar.selectbox(
        "Programa académico",
        ["ASS", "TRF", "TOC"]
    )

    datos_dicc["age"] = st.sidebar.number_input(
        "Edad",
        min_value=15,
        max_value=80,
        value=25
    )

    datos_dicc["gender"] = st.sidebar.selectbox(
        "Género",
        ["F", "M"]
    )

    datos_dicc["socioeconomic_level"] = st.sidebar.selectbox(
        "Nivel socioeconómico",
        [1, 2, 3, 4]
    )

    datos_dicc["employed"] = st.sidebar.selectbox(
        "¿Tiene empleo?",
        ["Si", "No"]
    )

    datos_dicc["semester"] = st.sidebar.selectbox(
        "Semestre",
        [1, 2, 3, 4]
    )

    datos_dicc["login_frequency"] = st.sidebar.number_input(
        "Frecuencia de ingreso al LMS",
        min_value=0.0,
        max_value=1000.0,
        value=100.0
    )

    datos_dicc["forum_participation"] = st.sidebar.number_input(
        "Participación en foros",
        min_value=0.0,
        max_value=1000.0,
        value=100.0
    )

    datos_dicc["task_submissions"] = st.sidebar.number_input(
        "Tareas enviadas",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

    datos_dicc["late_submissions"] = st.sidebar.number_input(
        "Tareas entregadas tarde",
        min_value=0.0,
        max_value=50.0,
        value=2.0
    )

    datos_dicc["connection_time"] = st.sidebar.number_input(
        "Tiempo de conexión",
        min_value=0.0,
        max_value=5000.0,
        value=500.0
    )

    datos_dicc["resource_views"] = st.sidebar.number_input(
        "Visualizaciones de recursos",
        min_value=0.0,
        max_value=5000.0,
        value=500.0
    )

    datos_dicc["final_grade"] = st.sidebar.number_input(
        "Calificación final",
        min_value=0.0,
        max_value=5.0,
        value=3.5,
        step=0.1
    )

    dataset_nuevo = pd.DataFrame(datos_dicc, index=[0])

    st.markdown("### Datos ingresados")
    st.dataframe(dataset_nuevo)

    # ======================================================
    # CODIFICACIÓN CATEGÓRICA
    # ======================================================

    columnas_categoricas = dataset_nuevo.select_dtypes(
        include=["object"]
    ).columns

    for columna in columnas_categoricas:
        encoder = LabelEncoder()
        dataset_nuevo[columna] = encoder.fit_transform(
            dataset_nuevo[columna].astype(str)
        )

    # ======================================================
    # ORDENAR COLUMNAS SEGÚN MODELO
    # ======================================================

    if hasattr(carga_modelo, "feature_names_in_"):

        columnas_modelo = list(carga_modelo.feature_names_in_)

        columnas_faltantes = [
            col for col in columnas_modelo
            if col not in dataset_nuevo.columns
        ]

        columnas_sobrantes = [
            col for col in dataset_nuevo.columns
            if col not in columnas_modelo
        ]

        if len(columnas_faltantes) > 0:
            st.error("Faltan columnas requeridas por el modelo:")
            st.write(columnas_faltantes)
            st.stop()

        if len(columnas_sobrantes) > 0:
            dataset_nuevo = dataset_nuevo.drop(
                columns=columnas_sobrantes
            )

        dataset_nuevo = dataset_nuevo[columnas_modelo]

    st.markdown("### Datos enviados al modelo")
    st.dataframe(dataset_nuevo)

    # ======================================================
    # PREDICCIÓN
    # ======================================================

    try:
        prediccion_modelo = carga_modelo.predict(dataset_nuevo)
        prediction_proba_modelo = carga_modelo.predict_proba(dataset_nuevo)
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        st.stop()

    prediccion = prediccion_modelo[0]

    probabilidad_no = prediction_proba_modelo[0][0] * 100
    probabilidad_si = prediction_proba_modelo[0][1] * 100

    # ======================================================
    # RESULTADOS
    # ======================================================

    st.subheader("Resultado de la predicción")

    if prediccion == 1:
        st.error(
            f"El estudiante tiene una probabilidad del {probabilidad_si:.2f}% de abandonar la formación."
        )
    else:
        st.success(
            f"El estudiante tiene una probabilidad del {probabilidad_no:.2f}% de permanecer en la formación."
        )

    st.subheader("Probabilidades")

    df_resultado = pd.DataFrame({
        "Clase": ["No abandono", "Sí abandono"],
        "Probabilidad (%)": [
            round(probabilidad_no, 2),
            round(probabilidad_si, 2)
        ]
    })

    st.dataframe(df_resultado)
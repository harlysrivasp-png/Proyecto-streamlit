import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ==========================================================
# ESCENARIO 3: CARGAR MODELO Y PREDICCIÓN MANUAL
# ==========================================================

def escenario3():

    st.markdown("## Escenario 3: Cargar Modelo y Realizar Predicción Manual")

    st.markdown("""
    En este escenario, el usuario puede cargar un modelo previamente entrenado en formato `.pkl`.
    Posteriormente, se ingresan manualmente los datos de un estudiante para estimar si presenta
    riesgo de abandono académico.
    """)

    # ======================================================
    # SUBIR MODELO ENTRENADO
    # ======================================================

    st.markdown("### Suba el modelo pre-entrenado")

    modelo_entrenado = st.file_uploader(
        "Suba el modelo entrenado en formato .pkl",
        type=["pkl"]
    )

    if modelo_entrenado is None:
        st.info("Por favor, suba un modelo entrenado en formato .pkl para continuar.")
        return

    # ======================================================
    # CARGAR MODELO
    # ======================================================

    try:
        carga_modelo = pickle.load(modelo_entrenado)
        st.success("Modelo cargado correctamente.")

    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return

    # ======================================================
    # FORMULARIO DE DATOS DEL ESTUDIANTE
    # ======================================================

    st.sidebar.markdown("## Datos del estudiante")

    datos_dicc = {}

    program = st.sidebar.selectbox(
        "Programa académico",
        ["ASS", "TRF", "TOC"],
        key="program_selectbox"
    )
    datos_dicc["program"] = program

    age = st.sidebar.number_input(
        "Edad del estudiante",
        min_value=15,
        max_value=80,
        value=25,
        key="age_input"
    )
    datos_dicc["age"] = age

    gender = st.sidebar.selectbox(
        "Género",
        ["F", "M"],
        key="gender_selectbox"
    )
    datos_dicc["gender"] = gender

    socioeconomic_level = st.sidebar.selectbox(
        "Nivel socioeconómico",
        [1, 2, 3, 4],
        key="socioeconomic_level_selectbox"
    )
    datos_dicc["socioeconomic_level"] = socioeconomic_level

    employed = st.sidebar.selectbox(
        "¿Tiene empleo?",
        ["Si", "No"],
        key="employed_selectbox"
    )
    datos_dicc["employed"] = employed

    semester = st.sidebar.selectbox(
        "Semestre académico",
        [1, 2, 3, 4],
        key="semester_selectbox"
    )
    datos_dicc["semester"] = semester

    login_frequency = st.sidebar.number_input(
        "Frecuencia de ingreso al LMS",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
        key="login_frequency_input"
    )
    datos_dicc["login_frequency"] = login_frequency

    forum_participation = st.sidebar.number_input(
        "Participación en foros",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
        key="forum_participation_input"
    )
    datos_dicc["forum_participation"] = forum_participation

    task_submissions = st.sidebar.number_input(
        "Número de tareas enviadas",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        key="task_submissions_input"
    )
    datos_dicc["task_submissions"] = task_submissions

    late_submissions = st.sidebar.number_input(
        "Número de tareas entregadas tarde",
        min_value=0.0,
        max_value=50.0,
        value=2.0,
        key="late_submissions_input"
    )
    datos_dicc["late_submissions"] = late_submissions

    connection_time = st.sidebar.number_input(
        "Tiempo de conexión en el LMS",
        min_value=0.0,
        max_value=5000.0,
        value=500.0,
        key="connection_time_input"
    )
    datos_dicc["connection_time"] = connection_time

    resource_views = st.sidebar.number_input(
        "Visualizaciones de recursos",
        min_value=0.0,
        max_value=5000.0,
        value=500.0,
        key="resource_views_input"
    )
    datos_dicc["resource_views"] = resource_views

    final_grade = st.sidebar.number_input(
        "Calificación final",
        min_value=0.0,
        max_value=5.0,
        value=3.5,
        step=0.1,
        key="final_grade_input"
    )
    datos_dicc["final_grade"] = final_grade

    # ======================================================
    # CREAR DATAFRAME
    # ======================================================

    dataset_nuevo = pd.DataFrame(datos_dicc, index=[0])

    st.markdown("### Datos ingresados para la predicción")
    st.write(dataset_nuevo)

    # ======================================================
    # PREPROCESAMIENTO
    # ======================================================

    dataset_procesado = dataset_nuevo.copy()

    # Codificar variables categóricas
    columnas_categoricas = dataset_procesado.select_dtypes(
        include=["object"]
    ).columns

    for columna in columnas_categoricas:
        encoder = LabelEncoder()
        dataset_procesado[columna] = encoder.fit_transform(
            dataset_procesado[columna].astype(str)
        )

    # Escalar variables numéricas
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
    dataset_procesado[columnas_numericas] = scaler.fit_transform(
        dataset_procesado[columnas_numericas]
    )

    st.markdown("### Datos procesados para el modelo")

    if st.checkbox("Mostrar datos procesados"):
        st.write(dataset_procesado)

    # ======================================================
    # PREDICCIÓN
    # ======================================================

    try:
        prediccion_modelo = carga_modelo.predict(dataset_procesado)
        prediction_proba_modelo = carga_modelo.predict_proba(dataset_procesado)

    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        st.warning(
            "Verifique que el modelo cargado haya sido entrenado con las mismas variables "
            "y el mismo orden de columnas."
        )
        return

    # ======================================================
    # RESULTADOS
    # ======================================================

    prediccion = prediccion_modelo[0]

    probabilidad_permanencia = prediction_proba_modelo[0][0] * 100
    probabilidad_abandono = prediction_proba_modelo[0][1] * 100

    col_pred, col_prob = st.columns((5, 5))

    with col_pred:
        st.subheader("Predicción del modelo")

        if prediccion == 1:
            st.error("Sí abandono")
        else:
            st.success("No abandono")

    with col_prob:
        st.subheader("Probabilidades")

        df_resultado = pd.DataFrame({
            "Clase": ["No abandono", "Sí abandono"],
            "Probabilidad (%)": [
                round(probabilidad_permanencia, 2),
                round(probabilidad_abandono, 2)
            ]
        })

        st.write(df_resultado)

    # ======================================================
    # MENSAJE INTERPRETATIVO
    # ======================================================

    st.divider()

    if prediccion == 1:
        st.markdown(
            f"### El estudiante tiene una probabilidad del "
            f"{probabilidad_abandono:.2f}% de abandonar la formación."
        )
    else:
        st.markdown(
            f"### El estudiante tiene una probabilidad del "
            f"{probabilidad_permanencia:.2f}% de permanecer en la formación."
        )


# ==========================================================
# EJECUTAR ESCENARIO
# ==========================================================

escenario3()
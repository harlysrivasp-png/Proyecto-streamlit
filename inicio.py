# ==========================================================
# EJECUTAR FUNCIÓN
# ==========================================================

cargar_archivo()

import streamlit as st
import pandas as pd


# ==========================================================
# FUNCIÓN PARA LEER CSV
# ==========================================================

def leer_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=";")

    return df


# ==========================================================
# FUNCIÓN PRINCIPAL
# ==========================================================

def cargar_archivo():

    # ======================================================
    # TÍTULO
    # ======================================================

    st.markdown(
        "# 💻 Aplicación Streamlit para Predecir el Abandono de Estudiantes"
    )

    # ======================================================
    # IMAGEN PRINCIPAL
    # ======================================================

    col_1, col_2, col_3 = st.columns(3, gap="large")

    with col_2:
        st.image(
            image="images/Deporte.png",
            width=300
        )

    # ======================================================
    # DESCRIPCIÓN
    # ======================================================

    st.markdown("""
    ## Aplicación de Predicción de Abandono Estudiantil

    **Elaborado por:** Harlys Rivas Perea

    Esta herramienta permite identificar estudiantes con posible riesgo de abandono académico
    a partir de variables sociodemográficas, académicas y de interacción con plataformas LMS.

    La aplicación utiliza modelos de Machine Learning para analizar patrones en los datos,
    entrenar clasificadores, evaluar resultados y generar predicciones individuales o masivas.

    ### Para comenzar:

    1. Sube el conjunto de datos de estudiantes en formato CSV.
    2. Dirígete al módulo de **Análisis Exploratorio de Datos (EDA)** para revisar la estructura, distribución y comportamiento de las variables.
    3. Usa el módulo de **Preprocesamiento** para limpiar datos, tratar valores nulos, eliminar valores atípicos, codificar variables categóricas y escalar variables numéricas.
    4. Ingresa al módulo de **Entrenamiento** para entrenar modelos de clasificación con diferentes parámetros.
    5. Descarga el modelo entrenado en formato `.pkl`.
    6. Usa los escenarios de predicción para estimar el riesgo de abandono de uno o varios estudiantes.
    7. Revisa las predicciones, probabilidades y gráficos generados por el sistema.
    8. Descarga el reporte final con los resultados de predicción.

    Esta aplicación busca apoyar procesos de alerta temprana, seguimiento académico y toma de decisiones
    para fortalecer la permanencia estudiantil.
    """)

    st.divider()

    # ======================================================
    # SUBIR ARCHIVO
    # ======================================================

    st.markdown("### 📂 Sube el archivo de datos")

    uploaded_file = st.file_uploader(
        "Carga tu archivo CSV",
        type=["csv"],
        accept_multiple_files=False
    )

    # ======================================================
    # SI SE CARGA EL ARCHIVO
    # ======================================================

    if uploaded_file is not None:

        # Leer CSV
        df = leer_csv(uploaded_file)

        # Normalizar nombres de columnas
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        # Correcciones específicas para el dataset
        df = df.rename(
            columns={
                "socioeconomic_level": "socioeconomic_level",
                "socioeconomic_level": "socioeconomic_level",
                "socioeconomic_level": "socioeconomic_level",
                "socioeconomic_level": "socioeconomic_level",
                "task_submissions": "task_submissions",
                "late_submissions": "late_submissions"
            }
        )

        # Guardar en session_state
        st.session_state.uploaded_file = uploaded_file
        st.session_state.df = df

        st.success("Archivo cargado correctamente.")

        # ==================================================
        # INFORMACIÓN DEL DATASET
        # ==================================================

        st.markdown("## 📌 Información general del dataset")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("Duplicados", df.duplicated().sum())
        col4.metric("Valores nulos", df.isnull().sum().sum())

        # ==================================================
        # IMAGEN
        # ==================================================

        c1, c2, c3 = st.columns(3)

        with c2:
            st.image(
                image="images/casa.png",
                width=200
            )

        # ==================================================
        # MOSTRAR DATASET
        # ==================================================

        st.markdown("### 📈 ¿Deseas ver el dataset?")

        mostrar_dataset = st.radio(
            "Escoge una opción",
            ["Ocultar Dataset", "Mostrar Dataset"]
        )

        if mostrar_dataset == "Mostrar Dataset":
            st.dataframe(df)

        # ==================================================
        # MOSTRAR HEAD
        # ==================================================

        st.markdown("### 🔎 ¿Deseas ver las primeras filas del dataset?")

        mostrar_head = st.checkbox("Mostrar primeras filas")

        if mostrar_head:
            st.dataframe(df.head())

        # ==================================================
        # MOSTRAR COLUMNAS
        # ==================================================

        st.markdown("### 🧾 Columnas del dataset")

        if st.checkbox("Mostrar columnas"):
            st.write(df.columns.tolist())

    # ======================================================
    # SI YA EXISTE EN SESSION STATE
    # ======================================================

    elif "uploaded_file" in st.session_state:

        st.markdown("## Acerca de los datos")

        st.markdown(
            f"Archivo previamente subido: **{st.session_state.uploaded_file.name}**"
        )

        df = st.session_state.df

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("Duplicados", df.duplicated().sum())
        col4.metric("Valores nulos", df.isnull().sum().sum())

        st.markdown("### 📈 ¿Deseas ver el dataset?")

        mostrar_dataset = st.radio(
            "Escoge una opción",
            ["Ocultar Dataset", "Mostrar Dataset"]
        )

        if mostrar_dataset == "Mostrar Dataset":
            st.dataframe(df)

        st.markdown("### 🔎 ¿Deseas ver las primeras filas del dataset?")

        mostrar_head = st.checkbox("Mostrar primeras filas")

        if mostrar_head:
            st.dataframe(df.head())


# ==========================================================
# EJECUTAR FUNCIÓN
# ==========================================================

cargar_archivo()
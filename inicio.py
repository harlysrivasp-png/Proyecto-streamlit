import streamlit as st
import pandas as pd


def cargar_archivo():

    st.markdown(
        "# 💻 Aplicación Streamlit para Predecir el Abandono de Estudiantes"
    )

    st.markdown("""
    ## Aplicación de Predicción de Abandono Estudiantil

    Esta herramienta permite cargar un dataset de estudiantes para realizar
    análisis exploratorio, preprocesamiento, entrenamiento y predicción de abandono.
    """)

    uploaded_file = st.file_uploader(
        "Carga tu archivo CSV",
        type=["csv"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=";")

        st.session_state.uploaded_file = uploaded_file
        st.session_state.df = df

        st.success("Archivo cargado correctamente.")

        st.markdown("## Información general del dataset")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("Duplicados", df.duplicated().sum())
        col4.metric("Valores nulos", df.isnull().sum().sum())

        if st.checkbox("Mostrar dataset"):
            st.dataframe(df)

        if st.checkbox("Mostrar primeras filas"):
            st.dataframe(df.head())

    elif "df" in st.session_state:

        st.info("Ya hay un dataset cargado en la sesión.")

        df = st.session_state.df

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Filas", df.shape[0])
        col2.metric("Columnas", df.shape[1])
        col3.metric("Duplicados", df.duplicated().sum())
        col4.metric("Valores nulos", df.isnull().sum().sum())

        if st.checkbox("Mostrar dataset"):
            st.dataframe(df)

        if st.checkbox("Mostrar primeras filas"):
            st.dataframe(df.head())



import streamlit as st
import pandas as pd

# ==========================================================
# FUNCIÓN PRINCIPAL
# ==========================================================

def cargar_archivo():

    # ======================================================
    # TÍTULO
    # ======================================================

    st.markdown(
        "# :computer: Mi Aplicación Streamlit para predecir el Abandono de Clientes"
    )

    # ======================================================
    # COLUMNAS
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

    ### Aplicación de Predicción de Abandono de Clientes 
    Elaborado por Est. ### Harlys Rivas Perea ### siguiendo las instrucciones del profesor ## David Andrés Soler#

    Esta herramienta te ayudará a identificar a los clientes que tienen más probabilidades de abandonar tu negocio.

    La aplicación utiliza un modelo de Machine Learning entrenado para analizar patrones en los datos.

    ### Para comenzar:

    1. Sube tu conjunto de datos de clientes utilizando la opción "subir" que encuentra en la parte de abajo
        Asegúrate de que tu archivo esté en formato CSV.
    2. Puedes dirigirte a la página de Análisis Exploratorio de Datos para analizar los datos que se cargaron.
    3. En la Pagina de Preprocesamiento puedes eliminar columnas o valores vacíos y atípico
    4. Una vez realizado todo ese procedimiento puedes dirigirte a Entrenamiento para Entrenar tu modelo con los 
       parámetros de cada modelo.
    5. Sube el modelo entrenado que utilizarás para realizar las predicciones. Asegúrate de que el modelo está en formato PKL.
    6. Navega por las opciones en a barra lateral para ajustar los parámetros y personalizar tu análisis.
    7. Revisa las predicciones y las probabilidades generadas por el modelo.
    8. Descarga el informe con las predicciones  y las probabilidades para cada cliente.
                
    ¡Buena suerte en tus esfuerzos para mejorar la retención de clientes y mantener a tus clientes satisfechos!            
    

    """)

    # ======================================================
    # SUBIR ARCHIVO
    # ======================================================

    st.markdown(
        "### :open_file_folder: Sube el archivo de datos"
    )

    uploaded_file = st.file_uploader(

        "Carga tu archivo csv",

        type=["csv"],

        accept_multiple_files=False
    )

    # ======================================================
    # SI SE CARGA EL ARCHIVO
    # ======================================================

    if uploaded_file is not None:

        # ==============================================
        # LEER CSV
        # ==============================================

        df = pd.read_csv(uploaded_file)

        # ==============================================
        # GUARDAR EN SESSION STATE
        # ==============================================

        st.session_state.uploaded_file = uploaded_file

        st.session_state.df = df

        # ==============================================
        # IMAGEN
        # ==============================================

        c1, c2, c3 = st.columns(3)

        with c2:

            st.image(

                image="images/casa.png",

                width=200
            )

        # ==============================================
        # MOSTRAR DATASET
        # ==============================================

        st.markdown(
            "### :chart_with_upwards_trend: ¿Deseas ver el dataset?"
        )

        mostrar_dataset = st.radio(

            "Escoge una opción",

            ["Mostrar Dataset", "Ocultar Dataset"]
        )

        if mostrar_dataset == "Mostrar Dataset":

            st.write(df)

        # ==============================================
        # MOSTRAR HEAD
        # ==============================================

        st.markdown(
            "### :mag: ¿Deseas ver el head del dataset?"
        )

        mostrar_head = st.checkbox(

            "Mostrar head del dataset"
        )

        if mostrar_head:

            st.write(df.head())

    # ======================================================
    # SI YA EXISTE EN SESSION STATE
    # ======================================================

    elif "uploaded_file" in st.session_state:

        st.markdown("## Acerca de los datos")

        st.markdown(

            f"Archivo previamente subido: "

            f"**{st.session_state.uploaded_file.name}**"
        )

        # ==============================================
        # MOSTRAR DATASET
        # ==============================================

        st.markdown(
            "### :chart_with_upwards_trend: ¿Deseas ver el dataset?"
        )

        mostrar_dataset = st.radio(

            "Escoge una opción",

            ["Mostrar Dataset", "Ocultar Dataset"]
        )

        if mostrar_dataset == "Mostrar Dataset":

            st.write(st.session_state.df)

        # ==============================================
        # MOSTRAR HEAD
        # ==============================================

        st.markdown(
            "### :mag: ¿Deseas ver el head del dataset?"
        )

        mostrar_head = st.checkbox(

            "Mostrar head del dataset"
        )

        if mostrar_head:

            st.write(st.session_state.df.head())

# ==========================================================
# EJECUTAR FUNCIÓN
# ==========================================================

cargar_archivo()
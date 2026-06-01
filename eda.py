import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import seaborn as sns
from io import StringIO


# ==========================================================
# ANÁLISIS EXPLORATORIO DE DATOS
# ==========================================================

def eda():

    st.markdown("# 📊 Análisis Exploratorio de Datos (EDA)")

    st.write("""
    Este módulo permite explorar el dataset de abandono estudiantil.
    El objetivo es identificar patrones generales, distribución de variables,
    valores faltantes, relaciones entre variables numéricas y posibles factores
    asociados al abandono académico.
    """)

    st.divider()

    # ==========================================================
    # VALIDAR DATASET
    # ==========================================================

    if "df" not in st.session_state:

        st.warning(
            "Debe ingresar el dataset primero. Diríjase a la página principal."
        )

        return

    # ==========================================================
    # DATASET
    # ==========================================================

    data = st.session_state.df.copy()

    # Normalizar nombres de columnas
    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    data = data.rename(
        columns={
            "socioeconomic_level": "socioeconomic_level",
            "socioecoNomic_level": "socioeconomic_level",
            "task_submisSions": "task_submissions",
            "late_submisSions": "late_submissions"
        }
    )

    # ==========================================================
    # VARIABLES
    # ==========================================================

    columnas_categoricas = list(
        data.select_dtypes(include="object").columns
    )

    columnas_numericas = list(
        data.select_dtypes(include="number").columns
    )

    if len(columnas_categoricas) == 0:
        st.warning("No se encontraron variables categóricas en el dataset.")
        seleccion_grafica_categoria = None
    else:
        seleccion_grafica_categoria = st.sidebar.selectbox(
            "Seleccione una variable categórica",
            columnas_categoricas
        )

    if len(columnas_numericas) == 0:
        st.warning("No se encontraron variables numéricas en el dataset.")
        seleccion_grafica_numerica = None
    else:
        seleccion_grafica_numerica = st.sidebar.selectbox(
            "Seleccione una variable numérica",
            columnas_numericas
        )

    # ==========================================================
    # MÉTRICAS
    # ==========================================================

    st.markdown("## Métricas generales del dataset")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Filas", data.shape[0])
    col2.metric("Columnas", data.shape[1])
    col3.metric("Duplicados", data.duplicated().sum())
    col4.metric("Categóricas", data.select_dtypes(include="object").shape[1])
    col5.metric("Numéricas", data.select_dtypes(include="number").shape[1])

    st.divider()

    # ==========================================================
    # VISTA PREVIA DEL DATASET
    # ==========================================================

    st.markdown("## Vista previa del dataset")

    if st.checkbox("Mostrar primeras filas"):
        st.dataframe(data.head())

    st.divider()

    # ==========================================================
    # DISTRIBUCIÓN DE VARIABLE OBJETIVO
    # ==========================================================

    if "dropout" in data.columns:

        st.markdown("## Distribución de la variable objetivo: dropout")

        valores_dropout = data["dropout"].value_counts().reset_index()
        valores_dropout.columns = ["dropout", "count"]

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            fig_dropout_bar = px.bar(
                valores_dropout,
                x="dropout",
                y="count",
                text="count",
                title="Distribución de abandono estudiantil"
            )

            fig_dropout_bar.update_layout(
                xaxis_title="Abandono",
                yaxis_title="Cantidad de estudiantes"
            )

            st.plotly_chart(fig_dropout_bar, use_container_width=True)

        with col_d2:
            fig_dropout_pie = px.pie(
                valores_dropout,
                names="dropout",
                values="count",
                title="Proporción de abandono estudiantil",
                hole=0.3
            )

            st.plotly_chart(fig_dropout_pie, use_container_width=True)

        st.divider()

    # ==========================================================
    # GRÁFICOS GENERALES
    # ==========================================================

    st.markdown("## Gráficos exploratorios")

    if seleccion_grafica_categoria is not None:

        c1, c2 = st.columns(2)

        valores_categoricos = (
            data[seleccion_grafica_categoria]
            .value_counts()
            .reset_index()
        )

        valores_categoricos.columns = [
            seleccion_grafica_categoria,
            "count"
        ]

        with c1:

            st.subheader("Gráfico de barras")

            fig_bar = px.bar(
                valores_categoricos,
                x=seleccion_grafica_categoria,
                y="count",
                text="count",
                title=f"Distribución de {seleccion_grafica_categoria}"
            )

            fig_bar.update_layout(
                xaxis_title=seleccion_grafica_categoria,
                yaxis_title="Frecuencia"
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:

            st.subheader("Gráfico circular")

            fig_pie = px.pie(
                valores_categoricos,
                names=seleccion_grafica_categoria,
                values="count",
                title=f"Proporción de {seleccion_grafica_categoria}",
                hole=0.3
            )

            st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ==========================================================
    # BOXPLOT E HISTOGRAMA
    # ==========================================================

    if seleccion_grafica_categoria is not None and seleccion_grafica_numerica is not None:

        gra1, gra2 = st.columns(2)

        with gra1:

            st.subheader("Boxplot")

            fig_box = px.box(
                data,
                x=seleccion_grafica_categoria,
                y=seleccion_grafica_numerica,
                color=seleccion_grafica_categoria,
                title=f"{seleccion_grafica_numerica} según {seleccion_grafica_categoria}"
            )

            st.plotly_chart(fig_box, use_container_width=True)

        with gra2:

            st.subheader("Histograma")

            def sturges_rule(datos):

                n = len(datos.dropna())

                if n <= 1:
                    return 1

                k = 1 + math.log2(n)

                return int(k)

            k = sturges_rule(data[seleccion_grafica_numerica])

            fig_hist = go.Figure()

            fig_hist.add_trace(
                go.Histogram(
                    x=data[seleccion_grafica_numerica],
                    nbinsx=k
                )
            )

            fig_hist.update_layout(
                title=f"Histograma - {seleccion_grafica_numerica}",
                xaxis_title=seleccion_grafica_numerica,
                yaxis_title="Frecuencia"
            )

            st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

    # ==========================================================
    # GRÁFICOS CONTRA DROPOUT
    # ==========================================================

    if "dropout" in data.columns and len(columnas_numericas) > 0:

        st.markdown("## Relación de variables numéricas con dropout")

        variable_dropout = st.selectbox(
            "Seleccione una variable numérica para comparar con dropout",
            columnas_numericas
        )

        fig_dropout_box = px.box(
            data,
            x="dropout",
            y=variable_dropout,
            color="dropout",
            title=f"{variable_dropout} según dropout"
        )

        st.plotly_chart(fig_dropout_box, use_container_width=True)

        st.divider()

    # ==========================================================
    # MATRIZ DE CORRELACIÓN
    # ==========================================================

    st.subheader("Mapa de calor - Correlación")

    if len(columnas_numericas) > 1:

        data_corr = data.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            data_corr,
            annot=True,
            cmap="YlOrRd",
            fmt=".2f",
            ax=ax
        )

        st.pyplot(fig)

    else:
        st.info("No hay suficientes variables numéricas para calcular correlación.")

    st.divider()

    # ==========================================================
    # VALORES FALTANTES
    # ==========================================================

    st.subheader("Mapa de calor - Valores faltantes")

    binary_df = data.isnull().astype(int)

    fig_null = go.Figure(
        data=go.Heatmap(
            z=binary_df.values,
            x=binary_df.columns,
            y=binary_df.index,
            colorscale="YlOrBr",
            showscale=False
        )
    )

    fig_null.update_layout(
        title="Mapa de valores faltantes",
        xaxis_title="Variables",
        yaxis_title="Registros"
    )

    st.plotly_chart(fig_null, use_container_width=True)

    st.divider()

    # ==========================================================
    # INFORMACIÓN DATASET
    # ==========================================================

    st.markdown("## Información sobre los datos")

    col_resu1, col_resu2 = st.columns(2)

    with col_resu1:

        st.markdown("### Resumen conciso")

        if st.checkbox("Mostrar resumen del DataFrame"):

            info = StringIO()

            data.info(buf=info)

            texto_info = info.getvalue()

            st.code(texto_info)

    with col_resu2:

        st.markdown("### Datos nulos")

        if st.checkbox("Mostrar datos nulos"):

            st.write(
                data.isnull()
                .sum()
                .sort_values(ascending=False)
            )

    st.divider()

    # ==========================================================
    # CORRELACIÓN Y ESTADÍSTICAS
    # ==========================================================

    col_resu3, col_resu4 = st.columns(2)

    with col_resu3:

        st.markdown("### Correlación")

        if st.checkbox("Mostrar matriz de correlación"):

            st.write(data.corr(numeric_only=True))

    with col_resu4:

        st.markdown("### Estadísticas descriptivas")

        if st.checkbox("Mostrar estadísticas descriptivas"):

            st.write(data.describe().round(2))


# ==========================================================
# EJECUTAR
# ==========================================================

eda()
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import seaborn as sns
from io import StringIO


def eda():

    st.markdown("# :bar_chart: Análisis Exploratorio de Datos (EDA)")

    st.write("""
    En este análisis exploratorio de datos, nos enfocaremos en comprender
    las variables y patrones subyacentes que afectan el abandono de clientes.
    """)

    st.divider()

    # ==========================================================
    # VALIDAR DATASET
    # ==========================================================

    if "df" not in st.session_state:

        st.image("images/upload-cloud-data.png", width=300)

        st.warning(
            "Debe ingresar el dataset primero. Diríjase a la página principal."
        )

        return

    # ==========================================================
    # DATASET
    # ==========================================================

    data = st.session_state.df

    # ==========================================================
    # VARIABLES
    # ==========================================================

    columnas_categoricas = list(
        data.select_dtypes(include="object").columns
    )

    columnas_numericas = list(
        data.select_dtypes(include="number").columns
    )

    seleccion_grafica_categoria = st.sidebar.selectbox(
        "Selecciona una variable categórica",
        columnas_categoricas
    )

    seleccion_grafica_numerica = st.sidebar.selectbox(
        "Selecciona una variable numérica",
        columnas_numericas
    )

    # ==========================================================
    # MÉTRICAS
    # ==========================================================

    st.markdown("## Métricas de los Datos")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Número de filas", data.shape[0])

    col2.metric("Número de columnas", data.shape[1])

    col3.metric("Datos duplicados", data.duplicated().sum())

    col4.metric(
        "Variables categóricas",
        data.select_dtypes(include="object").shape[1]
    )

    col5.metric(
        "Variables numéricas",
        data.select_dtypes(include="number").shape[1]
    )

    st.divider()

    # ==========================================================
    # DATOS GRÁFICOS
    # ==========================================================

    st.markdown("## Gráficos")

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

    valores_numericos = data[seleccion_grafica_numerica]

    # ==========================================================
    # BAR CHART
    # ==========================================================

    with c1:

        st.subheader("Gráfico de Barras")

        st.bar_chart(
            data=valores_categoricos,
            x=seleccion_grafica_categoria,
            y="count"
        )

    # ==========================================================
    # PIE CHART
    # ==========================================================

    with c2:

        st.subheader("Gráfico Circular")

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.pie(
            valores_categoricos["count"],
            labels=valores_categoricos[seleccion_grafica_categoria],
            autopct="%1.1f%%",
            startangle=140
        )

        ax.set_title(
            f"Gráfico circular - {seleccion_grafica_categoria}"
        )

        st.pyplot(fig)

    st.divider()

    # ==========================================================
    # BOXPLOT
    # ==========================================================

    gra1, gra2 = st.columns(2)

    with gra1:

        st.subheader("BoxPlot")

        fig_box = px.box(
            data,
            x=seleccion_grafica_categoria,
            y=seleccion_grafica_numerica,
            color=seleccion_grafica_categoria
        )

        st.plotly_chart(fig_box, use_container_width=True)

    # ==========================================================
    # HISTOGRAMA
    # ==========================================================

    with gra2:

        st.subheader("Histograma")

        def sturges_rule(datos):

            n = len(datos)

            k = 1 + math.log2(n)

            return int(k)

        k = sturges_rule(valores_numericos)

        fig_hist = go.Figure()

        fig_hist.add_trace(
            go.Histogram(
                x=valores_numericos,
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
    # MATRIZ DE CORRELACIÓN
    # ==========================================================

    st.subheader("Mapa de Calor - Correlación")

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

    st.divider()

    # ==========================================================
    # VALORES NULOS
    # ==========================================================

    st.subheader("Mapa de Calor - Valores Faltantes")

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

    st.plotly_chart(fig_null, use_container_width=True)

    st.divider()

    # ==========================================================
    # INFORMACIÓN DATASET
    # ==========================================================

    st.markdown("## Información sobre los Datos")

    col_resu1, col_resu2 = st.columns(2)

    # ==========================================================
    # INFO DATAFRAME
    # ==========================================================

    with col_resu1:

        st.markdown("### Resumen Conciso")

        if st.checkbox("Mostrar Resumen"):

            info = StringIO()

            data.info(buf=info)

            texto_info = info.getvalue()

            st.code(texto_info)

    # ==========================================================
    # DATOS NULOS
    # ==========================================================

    with col_resu2:

        st.markdown("### Datos Nulos")

        if st.checkbox("Mostrar Datos Nulos"):

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

        if st.checkbox("Mostrar Correlación"):

            st.write(data.corr(numeric_only=True))

    with col_resu4:

        st.markdown("### Estadísticas Descriptivas")

        if st.checkbox("Mostrar Estadísticas"):

            st.write(data.describe().round(2))


# ==========================================================
# EJECUTAR
# ==========================================================

eda()
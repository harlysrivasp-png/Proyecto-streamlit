import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px


# ==========================================================
# FUNCIÓN PARA EXPORTAR RESULTADOS A EXCEL
# ==========================================================

def to_excel(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predicciones")

    output.seek(0)
    return output


# ==========================================================
# FUNCIÓN PARA OBTENER COLUMNAS DEL MODELO
# ==========================================================

def obtener_columnas_modelo(modelo):
    if hasattr(modelo, "feature_names_in_"):
        return list(modelo.feature_names_in_)
    return None


# ==========================================================
# ESCENARIO 4: CARGAR MODELO Y CSV PARA PREDICCIÓN MASIVA
# ==========================================================

def escenario4():

    st.markdown("## Escenario 4: Predicción Masiva con Modelo Cargado")

    st.markdown("""
    En este escenario, el usuario puede cargar un modelo entrenado en formato `.pkl`
    y un archivo CSV con registros de estudiantes. El sistema procesa los datos,
    realiza la predicción de abandono académico y genera un reporte descargable.
    """)

    # ======================================================
    # CARGA DE ARCHIVOS
    # ======================================================

    datos_nuevos = st.sidebar.file_uploader(
        "Suba el archivo CSV con estudiantes",
        type=["csv"]
    )

    modelo_entrenado = st.file_uploader(
        "Suba el modelo entrenado en formato .pkl",
        type=["pkl"]
    )

    if modelo_entrenado is None or datos_nuevos is None:
        st.info("Por favor, suba el modelo entrenado (.pkl) y el archivo CSV para continuar.")
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
    # CARGAR CSV
    # ======================================================

    try:
        dataset_ingresado = pd.read_csv(datos_nuevos)

    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        return

    if st.checkbox("Mostrar datos ingresados"):
        st.write(dataset_ingresado)

    # ======================================================
    # COLUMNAS REQUERIDAS
    # ======================================================

    columnas_requeridas = [
        "program",
        "age",
        "gender",
        "socioeconomic_level",
        "employed",
        "semester",
        "login_frequency",
        "forum_participation",
        "task_submissions",
        "late_submissions",
        "connection_time",
        "resource_views",
        "final_grade"
    ]

    # Si el modelo trae nombres de columnas, se pueden usar
    columnas_modelo = obtener_columnas_modelo(carga_modelo)

    if columnas_modelo is not None:
        columnas_requeridas = columnas_modelo

    # ======================================================
    # VALIDAR COLUMNAS
    # ======================================================

    columnas_faltantes = [
        col for col in columnas_requeridas
        if col not in dataset_ingresado.columns
    ]

    if len(columnas_faltantes) > 0:
        st.error("El archivo CSV no contiene todas las columnas requeridas por el modelo.")
        st.write("Columnas faltantes:")
        st.write(columnas_faltantes)

        st.write("Columnas requeridas:")
        st.write(columnas_requeridas)
        return

    # ======================================================
    # PREPARAR DATASET PARA PREDICCIÓN
    # ======================================================

    dataset_nuevo = dataset_ingresado[columnas_requeridas].copy()

    st.divider()
    st.markdown("### Datos que ingresan a la predicción")

    if st.checkbox("Mostrar datos para predicción"):
        st.write(dataset_nuevo)

    # ======================================================
    # PREPROCESAMIENTO
    # ======================================================

    dataset_procesado = dataset_nuevo.copy()

    # Codificación de variables categóricas
    columnas_categoricas = dataset_procesado.select_dtypes(
        include=["object"]
    ).columns

    for columna in columnas_categoricas:
        encoder = LabelEncoder()
        dataset_procesado[columna] = encoder.fit_transform(
            dataset_procesado[columna].astype(str)
        )

    # Escalado de variables numéricas reales del dataset
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

    columnas_numericas = [
        col for col in columnas_numericas
        if col in dataset_procesado.columns
    ]

    scaler = StandardScaler()

    if len(columnas_numericas) > 0:
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
            "Verifique que el modelo haya sido entrenado con las mismas variables "
            "y en el mismo orden que el archivo cargado."
        )
        return

    # ======================================================
    # RESULTADOS
    # ======================================================

    df_prediccion = pd.DataFrame(prediccion_modelo, columns=["Abandono"])

    df_prediccion["Abandono"] = df_prediccion["Abandono"].map({
        0: "No",
        1: "Si"
    })

    probabilidad_permanencia = prediction_proba_modelo[:, 0] * 100
    probabilidad_abandono = prediction_proba_modelo[:, 1] * 100

    df_resultados = pd.DataFrame({
        "Abandono": df_prediccion["Abandono"],
        "Probabilidad_Permanencia (%)": probabilidad_permanencia.round(2),
        "Probabilidad_Abandono (%)": probabilidad_abandono.round(2)
    })

    df_unido = pd.concat(
        [
            dataset_ingresado.reset_index(drop=True),
            df_resultados.reset_index(drop=True)
        ],
        axis=1
    )

    csv_data = df_unido.to_csv(index=False).encode("utf-8")

    st.divider()
    st.markdown("### Datos con predicción final")

    if st.checkbox("Mostrar datos con predicción final"):
        st.write(df_unido)

    # ======================================================
    # GRÁFICOS
    # ======================================================

    st.divider()
    st.markdown("### Gráficos de la predicción")

    col_gra1, col_gra2 = st.columns((5, 5))

    valores_abandono = df_unido["Abandono"].value_counts()

    colorscale = px.colors.sequential.YlOrBr
    num_categorias = len(valores_abandono.index)

    if num_categorias > 0:
        step_size = max(1, int(len(colorscale) / num_categorias))
        colores = colorscale[::step_size]
    else:
        colores = colorscale

    with col_gra1:
        fig_bar = go.Figure()

        fig_bar.add_trace(
            go.Bar(
                x=valores_abandono.index,
                y=valores_abandono.values,
                text=valores_abandono.values,
                textposition="auto",
                hovertemplate="Predicción: %{x}<br>Cantidad: %{y}<extra></extra>",
                marker=dict(color=colores)
            )
        )

        fig_bar.update_layout(
            title="Distribución de Predicción de Abandono",
            xaxis_title="Predicción de abandono",
            yaxis_title="Cantidad de estudiantes",
            font=dict(size=12),
            width=500,
            height=500
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    with col_gra2:
        fig_pie = go.Figure()

        fig_pie.add_trace(
            go.Pie(
                labels=valores_abandono.index,
                values=valores_abandono.values,
                textinfo="label+percent",
                insidetextorientation="radial",
                hovertemplate="Predicción: %{label}<br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>",
                showlegend=True,
                marker=dict(colors=colores)
            )
        )

        fig_pie.update_layout(
            title="Porcentaje de Predicción de Abandono",
            font=dict(size=15),
            width=500,
            height=500
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    # ======================================================
    # DESCARGAS
    # ======================================================

    st.divider()
    st.markdown("### Descargar archivo con predicciones")

    st.download_button(
        label="📁 Descargar archivo Excel",
        data=to_excel(df_unido),
        file_name="Reporte_Prediccion_Abandono.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.download_button(
        label="📁 Descargar archivo CSV",
        data=csv_data,
        file_name="Reporte_Prediccion_Abandono.csv",
        mime="text/csv"
    )


# ==========================================================
# EJECUTAR ESCENARIO
# ==========================================================

escenario4()
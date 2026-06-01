import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px


# ==========================================================
# FUNCIÓN PARA EXPORTAR A EXCEL
# ==========================================================

def to_excel(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predicciones")

    output.seek(0)
    return output


# ==========================================================
# FUNCIÓN PARA LEER CSV
# ==========================================================

def leer_csv(archivo):
    try:
        df = pd.read_csv(archivo)
    except Exception:
        archivo.seek(0)
        df = pd.read_csv(archivo, sep=";")

    return df


# ==========================================================
# ESCENARIO 4: CARGAR MODELO Y CSV PARA PREDICCIÓN MASIVA
# ==========================================================

def escenario4():

    st.markdown("## Escenario 4: Cargar Modelo y CSV para Predicción Masiva")

    st.markdown("""
    En este escenario se carga un modelo entrenado en formato `.pkl` y un archivo CSV
    con registros de estudiantes. El sistema genera la predicción de abandono académico
    para cada estudiante y permite descargar el reporte final.
    """)

    # ======================================================
    # CARGAR ARCHIVOS
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
        st.info(
            "Por favor, suba el modelo entrenado (.pkl) y el archivo CSV para continuar."
        )
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
    # LEER DATASET
    # ======================================================

    try:
        dataset_ingresado = leer_csv(datos_nuevos)
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        st.stop()

    # ======================================================
    # NORMALIZAR NOMBRES DE COLUMNAS
    # ======================================================

    dataset_ingresado.columns = (
        dataset_ingresado.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    dataset_ingresado = dataset_ingresado.rename(
        columns={
            "task_submisssions": "task_submissions",
            "late_submisssions": "late_submissions",
            "task_submissions": "task_submissions",
            "late_submissions": "late_submissions",
            "socioeconomic_level": "socioeconomic_level"
        }
    )

    # ======================================================
    # VERIFICAR ARCHIVO CARGADO
    # ======================================================

    st.markdown("### Verificación del archivo cargado")

    st.write("Archivo cargado actualmente:")
    st.write(datos_nuevos.name)

    st.write("Dimensiones del archivo cargado:")
    st.write(dataset_ingresado.shape)

    st.write("Vista previa del archivo cargado:")
    st.dataframe(dataset_ingresado.head())

    if st.checkbox("Mostrar archivo completo cargado"):
        st.dataframe(dataset_ingresado)

    st.divider()

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

    if hasattr(carga_modelo, "feature_names_in_"):
        columnas_requeridas = list(carga_modelo.feature_names_in_)

    # ======================================================
    # VALIDAR COLUMNAS DEL CSV
    # ======================================================

    columnas_faltantes = [
        col for col in columnas_requeridas
        if col not in dataset_ingresado.columns
    ]

    if len(columnas_faltantes) > 0:
        st.error(
            "El archivo CSV no contiene todas las columnas requeridas por el modelo."
        )

        st.write("Columnas faltantes:")
        st.write(columnas_faltantes)

        st.write("Columnas requeridas por el modelo:")
        st.write(columnas_requeridas)

        st.write("Columnas encontradas en el archivo:")
        st.write(dataset_ingresado.columns.tolist())

        st.stop()

    # ======================================================
    # CREAR DATASET PARA PREDICCIÓN
    # ======================================================

    dataset_nuevo = dataset_ingresado[columnas_requeridas].copy()

    st.markdown("### Datos seleccionados para la predicción")

    st.write("Dimensiones antes del preprocesamiento:")
    st.write(dataset_nuevo.shape)

    if st.checkbox("Mostrar datos seleccionados para predicción"):
        st.dataframe(dataset_nuevo)

    st.divider()

    # ======================================================
    # PREPROCESAMIENTO DE VARIABLES CATEGÓRICAS
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
    # ORDENAR COLUMNAS SEGÚN EL MODELO ENTRENADO
    # ======================================================

    if hasattr(carga_modelo, "feature_names_in_"):

        columnas_modelo_entrenado = list(carga_modelo.feature_names_in_)

        columnas_faltantes_modelo = [
            col for col in columnas_modelo_entrenado
            if col not in dataset_nuevo.columns
        ]

        columnas_sobrantes = [
            col for col in dataset_nuevo.columns
            if col not in columnas_modelo_entrenado
        ]

        if len(columnas_faltantes_modelo) > 0:
            st.error(
                "Faltan columnas requeridas por el modelo después del preprocesamiento:"
            )
            st.write(columnas_faltantes_modelo)
            st.stop()

        if len(columnas_sobrantes) > 0:
            dataset_nuevo = dataset_nuevo.drop(columns=columnas_sobrantes)

        dataset_nuevo = dataset_nuevo[columnas_modelo_entrenado]

    # ======================================================
    # VERIFICACIÓN DE DATOS QUE ENTRAN AL MODELO
    # ======================================================

    st.markdown("### Verificación de datos que entran al modelo")

    st.write("Dimensiones de los datos que entran al modelo:")
    st.write(dataset_nuevo.shape)

    st.write("Columnas usadas por el modelo:")
    st.write(dataset_nuevo.columns.tolist())

    st.write("Vista previa de los datos que entran al modelo:")
    st.dataframe(dataset_nuevo.head())

    st.divider()

    # ======================================================
    # REALIZAR PREDICCIÓN
    # ======================================================

    try:
        prediccion_modelo = carga_modelo.predict(dataset_nuevo)
        prediction_proba_modelo = carga_modelo.predict_proba(dataset_nuevo)

    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        st.warning(
            "Verifique que el modelo haya sido entrenado con las mismas columnas "
            "y en el mismo orden que el archivo cargado."
        )
        st.stop()

    # ======================================================
    # RESULTADOS DE PREDICCIÓN
    # ======================================================

    df_prediccion = pd.DataFrame(
        prediccion_modelo,
        columns=["Abandono"]
    )

    df_prediccion["Abandono"] = df_prediccion["Abandono"].map({
        0: "No",
        1: "Si"
    })

    if prediction_proba_modelo.shape[1] >= 2:
        probabilidad_permanencia = prediction_proba_modelo[:, 0] * 100
        probabilidad_abandono = prediction_proba_modelo[:, 1] * 100
    else:
        probabilidad_permanencia = np.zeros(len(prediccion_modelo))
        probabilidad_abandono = np.zeros(len(prediccion_modelo))

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

    # ======================================================
    # MOSTRAR RESULTADOS
    # ======================================================

    st.divider()
    st.markdown("### Datos con predicción final")

    st.write("Dimensiones del archivo final con predicciones:")
    st.write(df_unido.shape)

    st.dataframe(df_unido)

    st.write("Distribución de predicciones:")
    st.write(df_unido["Abandono"].value_counts())

    st.divider()

    # ======================================================
    # GRÁFICOS DE PREDICCIÓN
    # ======================================================

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
            xaxis_title="Predicción",
            yaxis_title="Cantidad de estudiantes",
            font=dict(size=12)
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
                marker=dict(colors=colores),
                hovertemplate="Predicción: %{label}<br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>"
            )
        )

        fig_pie.update_layout(
            title="Porcentaje de Predicción de Abandono",
            font=dict(size=15)
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ======================================================
    # DESCARGAS
    # ======================================================

    st.markdown("### Descargar archivo con predicciones")

    csv_data = df_unido.to_csv(index=False).encode("utf-8")

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
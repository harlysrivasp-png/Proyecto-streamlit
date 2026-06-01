
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
# ESCENARIO 2: PREDICCIÓN MASIVA DESDE CSV
# ==========================================================

def escenario_2():

    st.markdown("## Escenario 2: Predicción Masiva de Abandono de Estudiantes")

    st.markdown("""
    En este escenario se puede cargar un archivo CSV con información de varios estudiantes.
    El sistema procesa los datos, aplica el modelo entrenado y genera una predicción de abandono
    para cada estudiante.
    """)

    # ======================================================
    # CARGAR MODELO ENTRENADO
    # ======================================================

    try:
        with open("modelo_entrenado.pkl", "rb") as archivo:
            carga_modelo = pickle.load(archivo)

        st.success("Modelo cargado correctamente.")

    except FileNotFoundError:
        st.error("No se encontró el archivo 'modelo_entrenado.pkl'. Verifique que esté en la carpeta del proyecto.")
        return

    # ======================================================
    # CARGAR ARCHIVO CSV
    # ======================================================

    datos_nuevos = st.sidebar.file_uploader(
        "Suba un archivo CSV con los estudiantes a predecir",
        type=["csv"]
    )

    st.markdown("### Suba el archivo CSV que contenga los nuevos datos")

    if datos_nuevos is not None:

        try:
            dataset_ingresado = pd.read_csv(datos_nuevos)

        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")
            return

        # ==================================================
        # MOSTRAR DATOS INGRESADOS
        # ==================================================

        if st.checkbox("Mostrar datos ingresados"):
            st.write(dataset_ingresado)

        # ==================================================
        # COLUMNAS ESPERADAS POR EL MODELO
        # ==================================================

        columnas_modelo = [
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

        # ==================================================
        # VALIDAR COLUMNAS
        # ==================================================

        columnas_faltantes = [
            columna for columna in columnas_modelo
            if columna not in dataset_ingresado.columns
        ]

        if len(columnas_faltantes) > 0:
            st.error("El archivo cargado no contiene todas las columnas requeridas.")
            st.write("Columnas faltantes:")
            st.write(columnas_faltantes)

            st.write("Columnas requeridas:")
            st.write(columnas_modelo)
            return

        # ==================================================
        # CREAR DATASET PARA PREDICCIÓN
        # ==================================================

        dataset_nuevo = dataset_ingresado[columnas_modelo].copy()

        st.divider()
        st.markdown("### Datos que ingresan a la predicción")

        if st.checkbox("Mostrar datos para predicción"):
            st.write(dataset_nuevo)

        # ==================================================
        # PREPROCESAMIENTO
        # ==================================================

        # Codificación de variables categóricas
        columnas_categoricas = dataset_nuevo.select_dtypes(include=["object"]).columns

        for columna in columnas_categoricas:
            encoder = LabelEncoder()
            dataset_nuevo[columna] = encoder.fit_transform(dataset_nuevo[columna].astype(str))

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
        dataset_nuevo[columnas_numericas] = scaler.fit_transform(dataset_nuevo[columnas_numericas])

        st.markdown("### Datos procesados para el modelo")

        if st.checkbox("Mostrar datos procesados"):
            st.write(dataset_nuevo)

        # ==================================================
        # PREDICCIÓN
        # ==================================================

        try:
            prediccion_modelo = carga_modelo.predict(dataset_nuevo)
            prediction_proba_modelo = carga_modelo.predict_proba(dataset_nuevo)

        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")
            return

        # ==================================================
        # RESULTADOS
        # ==================================================

        df_prediccion = pd.DataFrame(prediccion_modelo, columns=["Abandono"])
        df_prediccion["Abandono"] = df_prediccion["Abandono"].map({
            0: "No",
            1: "Si"
        })

        probabilidad_abandono = prediction_proba_modelo[:, 1] * 100
        probabilidad_permanencia = prediction_proba_modelo[:, 0] * 100

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

        st.divider()
        st.markdown("### Resultados de la predicción")
        st.write(df_unido)

        # ==================================================
        # GRÁFICOS
        # ==================================================

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

        # Gráfico de barras
        with col_gra1:
            fig_bar = go.Figure()

            fig_bar.add_trace(
                go.Bar(
                    x=valores_abandono.index,
                    y=valores_abandono.values,
                    text=valores_abandono.values,
                    textposition="auto",
                    marker=dict(color=colores),
                    hovertemplate="Predicción: %{x}<br>Cantidad: %{y}<extra></extra>"
                )
            )

            fig_bar.update_layout(
                title="Distribución de Predicción de Abandono",
                xaxis_title="Predicción",
                yaxis_title="Cantidad de estudiantes",
                font=dict(size=12),
                width=500,
                height=500
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        # Gráfico circular
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
                font=dict(size=15),
                width=500,
                height=500
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        # ==================================================
        # DESCARGAS
        # ==================================================

        st.divider()
        st.markdown("### Descargar el archivo con predicciones")

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


# ==========================================================
# EJECUTAR ESCENARIO
# ==========================================================

escenario_2()





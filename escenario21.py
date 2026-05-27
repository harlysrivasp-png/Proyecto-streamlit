import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
 
def columnas(modelo):
    columnas = None
    if hasattr(modelo, 'feature_names_in_'):
        columnas = modelo.feature_names_in_
    elif hasattr(modelo, "named_steps"):
        for step in modelo.named_steps.values():
            if hasattr(step, "transformers_"):
                columnas = []
                for _, transformer, column_indices in step.transformers_:
                    if hasattr(transformer, "get_feature_names_out"):
                        columnas.extend(transformer.get_feature_names_out(column_indices))
                    else:
                        columnas.extend(column_indices)
                break
    return columnas if columnas is not None else None
 
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Hoja1')
    writer.book.save(output)
    output.seek(0)
    return output
 
def escenario_2():
    datos_nuevos = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])
    st.markdown("### Suba el archivo CSV que contenga los nuevos datos")
 
    # BUG 1 CORREGIDO: pickle.load con ruta fija lanzaba FileNotFoundError
    # si el archivo no existía. Se reemplaza por un file_uploader igual que escenario_3.
    modelo_entrenado = st.file_uploader("Sube el Modelo Entrenado (.pkl)", type=["pkl"])
 
    if modelo_entrenado is not None and datos_nuevos is not None:
        carga_modelo = pickle.load(modelo_entrenado)
 
        column = ["PersonaMayor","Socio","Dependientes","Permanencia","ServicioTelefonico","VariasLineas",
                  "ServicioInternet","SeguridadLinea","CopiaSeguridadLinea","ProteccionDispositivo",
                  "ServicioTecnico","ServicioTV","ServicioPeliculas","Contrato","FacturacionElectronica",
                  "MetodoPago","RecargoMensual","TotalRecargo"]
 
        dataset_ingresado = pd.read_csv(datos_nuevos)
        if st.checkbox("Mostrar Datos Ingresados"):
            st.write(dataset_ingresado)
 
        columnas_comunes = dataset_ingresado.columns.intersection(column)
        dataset_nuevo = dataset_ingresado[columnas_comunes].copy()
 
        st.divider()
        st.markdown("### Datos a predecir")
        if st.checkbox("Mostrar Datos que ingresan a la Predicción"):
            st.write(dataset_nuevo)
 
        for i in dataset_nuevo.select_dtypes(include=['object']).columns:
            dataset_nuevo[i] = LabelEncoder().fit_transform(dataset_nuevo[i])
        scaler = StandardScaler().fit(dataset_nuevo[["TotalRecargo"]])
        dataset_nuevo[["TotalRecargo"]] = scaler.transform(dataset_nuevo[["TotalRecargo"]])
        scaler = StandardScaler().fit(dataset_nuevo[["RecargoMensual"]])
        dataset_nuevo[["RecargoMensual"]] = scaler.transform(dataset_nuevo[["RecargoMensual"]])
 
        prediccion_modelo = carga_modelo.predict(dataset_nuevo)
        prediction_proba_modelo = carga_modelo.predict_proba(dataset_nuevo)
 
        # BUG 2 CORREGIDO: se creaban dos df_abandono distintos (uno de predict
        # y otro de predict_proba) pisándose entre sí. Solo se necesita uno,
        # basado en predict (la predicción real del modelo).
        df_abandono = pd.DataFrame(prediccion_modelo, columns=["Abandono"])
        df_abandono = df_abandono.map(lambda x: "No" if x == 0 else "Si")
        probabilidades = np.where(
            df_abandono["Abandono"] == "No",
            prediction_proba_modelo[:, 0],
            prediction_proba_modelo[:, 1]
        )
 
        # BUG 3 CORREGIDO: al hacer pd.concat de dataset_ingresado (que podría
        # tener columna "Abandono") con df_resultados (que también la tiene),
        # se generaban columnas duplicadas. Se elimina "Abandono" del original
        # antes de concatenar.
        df_resultados = pd.DataFrame({
            "Abandono": df_abandono["Abandono"],
            "Probabilidad_Abandono": probabilidades
        })
        dataset_sin_abandono = dataset_ingresado.drop(columns=["Abandono"], errors="ignore")
        df_unido = pd.concat([dataset_sin_abandono, df_resultados], axis=1)
        csv_data = df_unido.to_csv(index=False)
 
        st.divider()
        st.markdown("### Graficos de la predicción")
        col_gra1, col_gra2 = st.columns((5, 5))
 
        # BUG 4 CORREGIDO: se usaba 'valores_categoricas' (con 'a' al final)
        # pero la variable se llamaba 'valores_categoricos' (con 'o').
        # Se unifica el nombre a 'valores_categoricos'.
        valores_categoricos = df_unido["Abandono"].value_counts()
        colorscale = px.colors.sequential.Blues
        num_categorias = len(valores_categoricos.index)
        step_size = max(1, int(len(colorscale) / num_categorias))
        colores = colorscale[::step_size]
 
        with col_gra1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=valores_categoricos.index,
                y=valores_categoricos.values,
                text=valores_categoricos.values,
                textposition='auto',
                hovertemplate='%{x}:<br>Cantidad:%{y}',
                marker=dict(color=colores)
            ))
            fig.update_layout(
                title="Gráfico de Barras - Predicción",
                xaxis_title="Abandono",
                yaxis_title="Cantidad",
                font=dict(size=12),
                width=500,
                height=500
            )
            st.plotly_chart(fig)
 
        with col_gra2:
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=valores_categoricos.index,
                values=valores_categoricos.values,
                textinfo='label+percent',
                insidetextorientation='radial',
                hovertemplate='%{label}:<br>Cantidad:%{value}<br>Porcentaje:%{percent}',
                showlegend=True,
                marker=dict(colors=colores)
            ))
            fig.update_layout(
                title="Gráfico Circular - Predicción",
                font=dict(size=15),
                width=500,
                height=500
            )
            st.plotly_chart(fig)
 
        st.divider()
        st.markdown("### Descargar el Archivo Predecido en Diferentes Formatos")
 
        # BUG 5 CORREGIDO: writer.book.save(output) falla con openpyxl moderno.
        # Se debe usar writer.close() o el context manager para guardar correctamente.
        # La función to_excel se corrige abajo.
        st.download_button(
            label=":file_folder: Descargar el Archivo Excel",
            data=to_excel(df_unido),
            file_name='Reporte.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        st.download_button(
            label=":file_folder: Descargar el Archivo CSV",
            data=csv_data,
            file_name='Reporte.csv',
            mime="text/csv"
        )
    else:
        st.info("Por favor, sube el modelo (.pkl) y el archivo CSV para continuar.")
 
# BUG 5 CORREGIDO (función): writer.book.save() no persiste en BytesIO
# con versiones modernas de openpyxl. Se usa writer.close() en su lugar.
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hoja1')
    output.seek(0)
    return output
 
escenario_2()
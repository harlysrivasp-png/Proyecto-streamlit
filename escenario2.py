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
    columnas=None
    if hasattr(modelo, 'feature_names_in_'):
        columnas = modelo.feature_names_in_
    elif hasattr(modelo, "named_steps"):
        for step in modelo.named_steps.values():
            if hasattr(step, "transformers_"):
                columnas = []
                for _,transformer ,column_indices in step.transformers_:
                    if hasattr(transformer, "get_feature_names_out"):
                        columnas.extend(transformer.get_feature_names_out(column_indices))
                    else:
                        columnas.extend(column_indices)
                break
    if columnas is not None:
        return columnas
    else:
        return None
    
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Hoja1')
    writer.book.save(output)
    output.seek(0)
    return output

def escenario_2():
    datos_nuevos = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])
    carga_modelo=pickle.load(open('modelo_entrenado.pkl', 'rb'))
    st.markdown("### Suba el archivo CSV que contenga los nuevos datos")
    if datos_nuevos is not None:
        column=["PersonaMayor","Socio","Dependientes","Permanencia","ServicioTelefonico","VariasLineas",
                     "ServicioInternet","SeguridadLinea","CopiaSeguridadLinea","ProteccionDispositivo",
                      "ServicioTecnico","ServicioTV","ServicioPeliculas","Contrato","FacturacionElectronica","MetodoPago","RecargoMensual",
                       "TotalRecargo"]
        dataset_ingresado = pd.read_csv(datos_nuevos)
        if st.checkbox("Mostrar Datos Ingresados"):
            st.write(dataset_ingresado)
        columnas_comunes=dataset_ingresado.columns.intersection(column)
        dataset_nuevo = dataset_ingresado[columnas_comunes]
        st.divider()
        st.markdown("### Datos a predecir")
        if st.checkbox("Mostrar Datos que ingresan a la Predicción"):
            st.write(dataset_nuevo)
       
        for i in dataset_nuevo.select_dtypes(include=['object']).columns:
            dataset_nuevo[i] = LabelEncoder().fit_transform(dataset_nuevo[i])
        scaler=StandardScaler().fit(dataset_nuevo[["TotalRecargo"]])
        dataset_nuevo[["TotalRecargo"]]=scaler.transform(dataset_nuevo[["TotalRecargo"]])
        scaler=StandardScaler().fit(dataset_nuevo[["RecargoMensual"]])
        dataset_nuevo["RecargoMensual"]=scaler.transform(dataset_nuevo[["RecargoMensual"]])

        prediccion_modelo=carga_modelo.predict(dataset_nuevo)
        prediction_proba_modelo=carga_modelo.predict_proba(dataset_nuevo)

        df_abandono=pd.DataFrame(prediccion_modelo, columns=["Abandono"])     
        df_abandono=df_abandono.map(lambda x: "No" if x == 0 else "Si")   
        df_abandono=pd.DataFrame(prediction_proba_modelo.argmax(axis=1), columns=["Abandono"]) 
        df_abandono=df_abandono.map(lambda x: "No" if x == 0 else "Si")
        probabilidades=np.where(df_abandono["Abandono"]=="No", prediction_proba_modelo[:,0], prediction_proba_modelo[:,1])

        df_resultados=pd.DataFrame({"Abandono": df_abandono["Abandono"], "Probabilidad_Abandono": probabilidades})
        df_unido=pd.concat([dataset_ingresado, df_resultados], axis=1)
        csv_data=df_unido.to_csv(index=False)

        st.divider()

        st.markdown("### Graficos de la predicción")
        col_gra1, col_gra2 = st.columns((5,5))
        valores_categoricas = df_unido["Abandono"].value_counts()
        colorscale=px.colors.sequential.YlOrBr
        num_categorias = len(valores_categoricas.index)
        step_size=int(len(colorscale) / num_categorias)
        colores = colorscale[::step_size]

        with col_gra1:
            fig=go.Figure()

            fig.add_trace(go.Bar(
                x=valores_categoricas.index,
                y=valores_categoricas,
                text=valores_categoricas,
                textposition='auto',
                hovertemplate='%{x}:<br>valores_categoricas:%{y}',
                marker=dict(color=colores)
            ))
            fig.update_layout(
                title=f"Gráfico de Barras-Predicción",
                xaxis_title="Género",
                yaxis_title="valores_categoricas",
                font=dict(size=12),
                width=500,
                height=500
                
            )
            st.plotly_chart(fig)
        with col_gra2:
            fig=go.Figure()
            fig.add_trace(go.Pie(
                labels=valores_categoricas.index,
                values=valores_categoricas.values,
                textinfo='label+percent',
                insidetextorientation='radial',
                hovertemplate='%{label}:<br>valores_categoricas:%{value}<br>Porcentaje:%{percent})',
                showlegend=True,
                marker=dict(colors=colores)
            ))

            fig.update_layout(
                title=f"Gráfico Circular- Predicción",
                font=dict(size=15),
                width=500,
                height=500
            )

            st.plotly_chart(fig)

        st.divider()

        st.markdown("### Descargar el Archivo Predecido en Diferentes Formatos")   
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

escenario_2()
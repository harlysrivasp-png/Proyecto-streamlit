import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from io import BytesIO

def escenario_3():
    st.markdown("### Suba el modelo pre-entrenado")
    modelo_entrenado=st.file_uploader("Sube el Modelo Entrenado", type=["pkl"])
    if modelo_entrenado is not None:
        carga_modelo=pickle.load(modelo_entrenado)

        column=["Genero","PersonaMayor", "Socio", "Dependientes", "Permanencia", "ServicioTelefonico", "VariasLineas", "ServicioInternet", 
            "SeguridadLinea", "CopiaSeguridadLinea","ProteccionDispositivo", "ServicioTecnico", "ServicioTV", "ServicioPeliculas", 
            "Contrato", "FacturacionElectronica", "MetodoPago", "RecargoMensual", "TotalRecargo"]
        datos_dicc={}
        if "Genero" in column:
            genero=st.sidebar.selectbox('Genero', ("Femenino", "Masculino"),key='genero_selectbox')
            datos_dicc['Genero']=genero
        
        if "PersonaMayor" in column:
            PersonaMayor=st.sidebar.selectbox("Es una Persona Adulta Mayor ?", (0, 1), key='persona_mayor_selectbox')
            datos_dicc["PersonaMayor"]=PersonaMayor
        if "Socio" in column:
            Socio=st.sidebar.selectbox('Es un Socio ?', ("Si","No"), key='socio_selectbox')
            datos_dicc["Socio"]=Socio
        if "Dependientes" in column:
            Dependientes=st.sidebar.selectbox('¿Eres Dependientes ?', ("Si","No"), key='dependientes_selectbox')
            datos_dicc["Dependientes"]=Dependientes
        if "Permanencia" in column:
            Permanencia=st.sidebar.number_input('¿Cuántos meses tiene su contrato?',0,72,29, key='permanencia_number_input')
            datos_dicc["Permanencia"]=Permanencia
        if "ServicioTelefonico" in column:
            ServicioTelefonico=st.sidebar.selectbox('¿Tiene Servicio Telefónico ?', ("Si","No"), key='servicio_telefonico_selectbox')
            datos_dicc["ServicioTelefonico"]=ServicioTelefonico
        if "VariasLineas" in column:
            VariasLineas=st.sidebar.selectbox('¿Tiene Varias Lineas ?', ("Si","No","Sin Servicio Telefónico"), key='varias_lineas_selectbox')
            datos_dicc["VariasLineas"]=VariasLineas
        if "ServicioInternet" in column:
            ServicioInternet=st.sidebar.selectbox('Tiene Servicio de Internet ?', ("DLS","No","Fibra Óptica"), key='servicio_internet_selectbox')
            datos_dicc["ServicioInternet"]=ServicioInternet
        if "SeguridadLinea" in column:
            SeguridadLinea=st.sidebar.selectbox('Tiene Seguridad en la Linea ?', ("Si","No","Sin Servicio"), key='seguridad_linea_selectbox')
            datos_dicc["SeguridadLinea"]=SeguridadLinea
        if "CopiaSeguridadLinea" in column:
            CopiaSeguridadLinea=st.sidebar.selectbox('Tiene Copia de Seguridad en la Linea ?', ("Si","No","Sin Servicio"), key='copia_seguridad_linea_selectbox')
            datos_dicc["CopiaSeguridadLinea"]=CopiaSeguridadLinea
        if "ProteccionDispositivo" in column:
            ProteccionDispositivo=st.sidebar.selectbox('Tiene Proteccion de Dispositivo ?', ("Si","No","Sin Servicio"), key='proteccion_dispositivo_selectbox')
            datos_dicc["ProteccionDispositivo"]=ProteccionDispositivo
        if "ServicioTecnico" in column:
            ServicioTecnico=st.sidebar.selectbox('Tiene Servicio Tecnico ?', ("Si","No","Sin Servicio"), key='servicio_tecnico_selectbox')
            datos_dicc["ServicioTecnico"]=ServicioTecnico
        if "ServicioTV" in column:
            ServicioTv=st.sidebar.selectbox('Tiene Servicio de TV ?', ("Si","No","Sin Servicio"), key='servicio_tv_selectbox')
            datos_dicc["ServicioTV"]=ServicioTv
        if "ServicioPeliculas" in column:
            ServicioPeliculas=st.sidebar.selectbox('Tiene Servicio de Peliculas ?', ("Si","No","Sin Servicio"), key='servicio_peliculas_selectbox')
            datos_dicc["ServicioPeliculas"]=ServicioPeliculas
        if "Contrato" in column:
            Contrato=st.sidebar.selectbox('Tipo de Contrato ?', ("Mes a Mes","Un Año","Dos Años"), key='contrato_selectbox')
            datos_dicc["Contrato"]=Contrato
        if "FacturacionElectronica" in column:
            FacturacionElectronica=st.sidebar.selectbox('¿Tiene Facturación Electrónica ?', ("Si","No"), key='facturacion_electronica_selectbox')
            datos_dicc["FacturacionElectronica"]=FacturacionElectronica
        if "MetodoPago" in column:
            MetodoPago=st.sidebar.selectbox('¿Cuál es su Método de Pago ?', ("Tarjeta de Crédito","Transferencia Bancaria","Giro Postal","Efectivo"), key='metodo_pago_selectbox')
            datos_dicc["MetodoPago"]=MetodoPago
        if "RecargoMensual" in column:
            RecargoMensual=st.sidebar.number_input('¿Cuál es su Recargo Mensual?',0.0,100.0,20.0, key='recargo_mensual_number_input')
            datos_dicc["RecargoMensual"]=RecargoMensual
        if "TotalRecargo" in column:
            TotalRecargo=st.sidebar.number_input('¿Cuál es su Total Recargo?',0.0,1000.0,100.0, key='total_recargo_number_input')
            datos_dicc["TotalRecargo"]=TotalRecargo

        dataset_nuevo=pd.DataFrame([datos_dicc],index=[0])
        st.write('Actualmente usando parámetros de entrada(que se muestran a continuación):')
        st.write(dataset_nuevo)


        for i in dataset_nuevo.select_dtypes(include=['object']).columns:
             dataset_nuevo[i] = LabelEncoder().fit_transform(dataset_nuevo[i])
             scaler=StandardScaler().fit(dataset_nuevo[["TotalRecargo"]])
             dataset_nuevo[["TotalRecargo"]]=scaler.transform(dataset_nuevo[["TotalRecargo"]])
             scaler=StandardScaler().fit(dataset_nuevo[["RecargoMensual"]])
             dataset_nuevo[["RecargoMensual"]]=scaler.transform(dataset_nuevo[["RecargoMensual"]])

    prediccion_modelo=carga_modelo.predict(dataset_nuevo)
    prediction_proba_modelo=carga_modelo.predict_proba(dataset_nuevo)
    col_nada_predi,col_nada_pro=st.columns((5,5))
    with col_nada_predi:
        st.subheader("Predicción del Modelo:")
        df_abandono=pd.DataFrame(prediccion_modelo, columns=["Abandono"])
        df_abandono=df_abandono.applymap(lambda x: "No Abandono" if x == 0 else "Si Abandono")
        st.write(df_abandono)
    with col_nada_pro:
        st.subheader('Probabilidad de predicción')
        df_abandono=pd.DataFrame(prediction_proba_modelo.argmax(axis=1), columns=["Abandono"])
        df_abandono=df_abandono.applymap(lambda x: "No Abandono" if x == 0 else "Si Abandono")
        probabilidades=np.where(df_abandono["Abandono"]=="No Abandono", prediction_proba_modelo[:,0], prediction_proba_modelo[:,1])
        df_resultado=pd.DataFrame({"Abandono": df_abandono["Abandono"], "Probabilidad Abandono": probabilidades})
        st.write(df_resultado)
    for index,row in df_resultado.iterrows():
        abandono=row.iloc[0]
        probabilidad=row.iloc[1]*100
        if abandono== "Si":
            st.markdown(f"### La persona tiene una probabilidad del{probabilidad:.2f}% de que abandone el servicio.")
        else:
            st.markdown(f"### La persona tiene una Probabilidad del {probabilidad:.2f}% de que no abandone el servicio.")


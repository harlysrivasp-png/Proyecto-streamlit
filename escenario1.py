import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def escenario_1():
    st.markdown("## Escenario 1: Predicción Manual de Abandono de Clientes")
    st.markdown("""
    En este escenario, se te presentará un formulario para ingresar manualmente los datos de un cliente. A partir de estos datos, el modelo de regresión logística predecirá la probabilidad de abandono del cliente.
    ** Instrucciones:**
    - Completa el formulario con los datos del cliente.
    - Haz clic en el botón "Predecir" para obtener la probabilidad de abandono.
    """)
    
    # Cargar el modelo entrenado
    carga_modelo=pickle.load(open('modelo_entrenado.pkl', 'rb'))
    column=["PersonaMayor", "Socio", "Dependientes", "Permanencia", "ServicioTelefonico", "VariasLineas", "ServicioInternet", 
            "SeguridadLinea", "CopiaSeguridadLinea","ProteccionDispositivo", "ServicioTecnico", "ServicioTV", "ServicioPeliculas", 
            "Contrato", "FacturacionElectronica", "MetodoPago", "RecargoMensual", "TotalRecargo"]
    datos_dicc={}
    if "Genero" in column:
        Genero=st.sidebar.selectbox("Genero", ("Femenino", "Masculino"),key='genero_selectbox')
        datos_dicc['Genero']=Genero

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
        ServicioTecnico=st.sidebar.selectbox('Tiene Servicio Técnico ?', ("Si","No","Sin Servicio"), key='servicio_tecnico_selectbox')
        datos_dicc["ServicioTecnico"]=ServicioTecnico

    if "ServicioTV" in column:
        ServicioTv=st.sidebar.selectbox('Tiene Servicio de TV ?', ("Si","No","Sin Servicio"), key='servicio_tv_selectbox')
        datos_dicc["ServicioTV"]=ServicioTv

    if "ServicioPeliculas" in column:
        ServicioPeliculas=st.sidebar.selectbox('Tiene Servicio de Peliculas ?', ("Si","No","Sin Servicio"), key='servicio_peliculas_selectbox')
        datos_dicc["ServicioPeliculas"]=ServicioPeliculas

    if "Contrato" in column:
        Contrato=st.sidebar.selectbox("Tipo de Contrato", ("Mes a Mes", "Un Año", "Dos Años"), key='contrato_selectbox')
        datos_dicc["Contrato"]=Contrato

    if "FacturacionElectronica" in column:
        FacturacionElectronica=st.sidebar.selectbox('Tiene Facturación Electrónica ?', ("Si","No"), key='facturacion_electronica_selectbox')
        datos_dicc["FacturacionElectronica"]=FacturacionElectronica

    if "MetodoPago" in column:
        MetodoPago=st.sidebar.selectbox('¿Cuál es el método de Pago?', ("Cheque Electrónico", "Cheque por correo","Transferencia bancaria(automática)", "Tarjeta de Crédito (automática)"), key='metodo_pago_selectbox')
        datos_dicc["MetodoPago"]=MetodoPago

    if "RecargoMensual" in column:
        RecargoMensual=st.sidebar.number_input('Recargo Mensual', 0.00, 200.00,70.35, key='recargo_mensual_number_input')
        datos_dicc["RecargoMensual"]=RecargoMensual

    if "TotalRecargo" in column:
        TotalRecargo=st.sidebar.number_input('Recargo Anual',0.00,10000.00,1000.00,key='total_Recargo_number_input')
        datos_dicc["TotalRecargo"]=TotalRecargo

    dataset_nuevo=pd.DataFrame(datos_dicc, index=[0])
    st.markdown("### Actualmente usando parámetros de entrada(que se muestran a continuación):")
    st.write(dataset_nuevo)

    # Preprocesamiento de Predicciones

    for i in dataset_nuevo.select_dtypes(include=['object']).columns:
        dataset_nuevo[i] = LabelEncoder().fit_transform(dataset_nuevo[i])
        scaler=StandardScaler().fit(dataset_nuevo[["TotalRecargo"]])
        dataset_nuevo[["TotalRecargo"]]=scaler.transform(dataset_nuevo[["TotalRecargo"]])
        scaler=StandardScaler().fit(dataset_nuevo[["RecargoMensual"]])
        dataset_nuevo[["RecargoMensual"]]=scaler.transform(dataset_nuevo[["RecargoMensual"]])
    # Realizar la predicción
    prediccion_modelo=carga_modelo.predict(dataset_nuevo)
    prediction_proba_modelo=carga_modelo.predict_proba(dataset_nuevo)
    col_nada_predi,col_nada_pro=st.columns((5,5))
    with col_nada_predi:
        st.subheader('Predicción')
        df_abandono=pd.DataFrame(prediccion_modelo, columns=['Abandono'])
        df_abandono=df_abandono.map(lambda x: "No" if x== 0 else "Si")
        st.write(df_abandono)
    with col_nada_pro:
        st.subheader('Probabilidad de Predicción')
        df_abandono=pd.DataFrame(
            np.argmax(prediction_proba_modelo,axis=1), columns=["Abandono"])
        df_abandono=df_abandono.map(lambda x: "No" if x== 0 else "Si")
        probabilidades=np.where(df_abandono["Abandono"]=="No", prediction_proba_modelo[:,0], prediction_proba_modelo[:,1])
        df_resultado=pd.DataFrame({"Abandono":df_abandono["Abandono"],"Probabilidad":probabilidades})
        st.write(df_resultado)
    for index,row in df_resultado.iterrows():
        abandono=row.iloc[0]
        probabilidad=row.iloc[1]*100
        if row["Abandono"]=="Si":
            st.markdown(f"### El cliente tiene una probabilidad del {probabilidad:.2f}% de abandonar el servicio.")
        else:
            st.markdown(f"###El cliente tiene una probabilidad del {probabilidad:.2f}% de permanecer con el servicio.")
escenario_1()


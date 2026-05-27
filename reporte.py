import streamlit as st
import pandas as pd

def reporte():
    st.markdown("#: Bar_chart: Reporte de los Datos")
    st.markdown('''
    En esta sección, presentamos un reporte detallado y visual creado en Power Bi. Este proporciona información
    valiosa general del análisis que permite a los usuarios aprovechar las capacidades de ambas herramientas, ofreciendo visualizaciones dinámicas  La integración de Power Bi con Streamlit permite a los usuarrios aprovechar las capacidades de ambas herramientas
       ''' )
    st.markdown('''
                iframe title="Reporte Power BI- Abandono de Clientes" width="1100"height="673.5"scr="https://app.powerbi.com/groups/me/reports/5fbae260-ef74-468e-ade2-b6411''',unsafe_allow_html=True)
reporte()


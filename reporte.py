import streamlit as st
import pandas as pd

def reporte():
    st.markdown("#: Bar_chart: Reporte de los Datos")
    st.markdown('''
   En esta sección, presentamos un reporte detallado y visual creado en Power Bi. Este reporte proporciona información
   valiosa y una visión general del análisis de los datos.
                
   La integración de Power BI con Streamlit permite a los usuarios aprovechar las capacidades de ambas herramientas
   ofreciendo visualizaciones dinámicas y una experiencia de usuario mejorada
                
   A continuación, encontrarás el reporte de Power BI que hemos separado para este análisis. Puede interactuar con las visualizaciones
   y explorar diferentes aspectos de los datos. Si desea obtener aspectos de los datos. si deseas obtener más información
   sobre alguna visualización específica, simplemente coloca el cursor sobre ella y aparecerá información adicional.
                
    Para obtener los mejores resultados, le recomendamos visualizar este reporte en pantalla completa. Para hacerlo haz clic en el
    ícono de pantalla completa en la esquina inferior derecha del reporte.
                
    ¡Disfruta explorando el reporte y descubriendo información útil para tu análisis ''' )
    st.markdown('''
                iframe title="Reporte Power BI- Abandono de Clientes" width="1100"height="673.5"scr="https://app.powerbi.com/groups/me/reports/5fbae260-ef74-468e-ade2-b6411''',unsafe_allow_html=True)
reporte()


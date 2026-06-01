import streamlit as st
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def preprocesamiento():

    st.markdown("# ⚙️ Preprocesamiento de Datos")
    st.markdown(
        """
        En este módulo se realiza la preparación del dataset de abandono estudiantil.
        El usuario puede eliminar columnas, tratar valores nulos, remover valores atípicos,
        codificar variables categóricas y escalar variables numéricas.
        """
    )

    # =====================================================
    # VALIDAR DATASET
    # =====================================================

    if "df" not in st.session_state:

        st.warning("Debe cargar el dataset primero.")
        return

    # =====================================================
    # COPIA ORIGINAL
    # =====================================================

    if "original_df" not in st.session_state:
        st.session_state.original_df = st.session_state.df.copy()

    if "predf" not in st.session_state:
        st.session_state.predf = st.session_state.df.copy()

    data = st.session_state.predf.copy()

    # =====================================================
    # LIMPIEZA DE NOMBRES DE COLUMNAS
    # =====================================================

    st.subheader("Limpieza de nombres de columnas")

    if st.button("Normalizar nombres de columnas"):

        data.columns = (
            data.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        # Corrección de nombres específicos del dataset
        data = data.rename(
            columns={
                "socioeconomic_level": "socioeconomic_level",
                "socioeconomic_level": "socioeconomic_level",
                "socioeconomic_level": "socioeconomic_level",
                "socioecoNomic_level": "socioeconomic_level",
                "task_submisSions": "task_submissions",
                "late_submisSions": "late_submissions"
            }
        )

        st.session_state.predf = data
        st.success("Nombres de columnas normalizados correctamente.")
        st.dataframe(data.head())

    st.divider()

    # =====================================================
    # RESTABLECER DATASET
    # =====================================================

    st.subheader("Restablecer Dataset")

    if st.button("Restablecer Preprocesado"):

        st.session_state.predf = st.session_state.original_df.copy()
        data = st.session_state.predf.copy()

        st.success("Dataset restablecido correctamente.")

    st.divider()

    # =====================================================
    # ELIMINAR COLUMNA ID
    # =====================================================

    st.subheader("Eliminar identificador")

    if "id" in data.columns:

        if st.button("Eliminar columna id"):

            data = data.drop(columns=["id"])
            st.session_state.predf = data

            st.success("Columna id eliminada correctamente.")
            st.dataframe(data.head())

    else:
        st.info("La columna id no se encuentra en el dataset actual.")

    st.divider()

    # =====================================================
    # ELIMINAR COLUMNAS MANUALMENTE
    # =====================================================

    st.subheader("Eliminar Columnas")

    columnas_eliminar = st.multiselect(
        "Seleccione columnas a eliminar",
        data.columns
    )

    if st.button("Eliminar Columnas Seleccionadas"):

        if len(columnas_eliminar) > 0:

            data = data.drop(columns=columnas_eliminar)
            st.session_state.predf = data

            st.success("Columnas eliminadas correctamente.")
            st.dataframe(data.head())

        else:
            st.warning("No seleccionó columnas para eliminar.")

    st.divider()

    # =====================================================
    # VALORES NULOS CATEGÓRICOS
    # =====================================================

    st.subheader("Valores Nulos Categóricos")

    variables_cat = st.multiselect(
        "Seleccione variables categóricas",
        data.select_dtypes(include="object").columns
    )

    metodo_cat = st.selectbox(
        "Método de imputación categórica",
        ["most_frequent", "constant"]
    )

    fill_value = None

    if metodo_cat == "constant":
        fill_value = st.text_input("Valor de reemplazo")

    if st.button("Reemplazar Nulos Categóricos"):

        if len(variables_cat) > 0:

            imputer = SimpleImputer(
                strategy=metodo_cat,
                fill_value=fill_value
            )

            data[variables_cat] = imputer.fit_transform(
                data[variables_cat]
            )

            st.session_state.predf = data

            st.success("Valores categóricos reemplazados correctamente.")

        else:
            st.warning("Seleccione al menos una variable categórica.")

    st.divider()

    # =====================================================
    # VALORES NULOS NUMÉRICOS
    # =====================================================

    st.subheader("Valores Nulos Numéricos")

    variables_num = st.multiselect(
        "Seleccione variables numéricas",
        data.select_dtypes(include=["int64", "float64"]).columns
    )

    metodo_num = st.selectbox(
        "Método de imputación numérica",
        ["SimpleImputer", "KNNImputer"]
    )

    if metodo_num == "SimpleImputer":

        estrategia = st.selectbox(
            "Estrategia",
            ["mean", "median", "most_frequent"]
        )

    else:

        vecinos = st.slider(
            "Número de vecinos",
            1,
            10,
            5
        )

    if st.button("Reemplazar Nulos Numéricos"):

        if len(variables_num) > 0:

            if metodo_num == "SimpleImputer":

                imputer = SimpleImputer(
                    strategy=estrategia
                )

            else:

                imputer = KNNImputer(
                    n_neighbors=vecinos
                )

            data[variables_num] = imputer.fit_transform(
                data[variables_num]
            )

            st.session_state.predf = data

            st.success("Valores numéricos reemplazados correctamente.")

        else:
            st.warning("Seleccione al menos una variable numérica.")

    st.divider()

    # =====================================================
    # ELIMINAR ATÍPICOS
    # =====================================================

    st.subheader("Eliminar Valores Atípicos")

    variables_outliers = st.multiselect(
        "Seleccione variables para detectar outliers",
        data.select_dtypes(include=["int64", "float64"]).columns
    )

    metodo_outlier = st.selectbox(
        "Método de detección de outliers",
        ["Boxplot", "Isolation Forest"]
    )

    if st.button("Eliminar Atípicos"):

        if len(variables_outliers) > 0:

            if metodo_outlier == "Boxplot":

                for col in variables_outliers:

                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1

                    inferior = Q1 - 1.5 * IQR
                    superior = Q3 + 1.5 * IQR

                    data = data[
                        (data[col] >= inferior) &
                        (data[col] <= superior)
                    ]

            else:

                iso = IsolationForest(
                    contamination=0.05,
                    random_state=42
                )

                pred = iso.fit_predict(
                    data[variables_outliers]
                )

                data = data[pred == 1]

            st.session_state.predf = data

            st.success("Valores atípicos eliminados correctamente.")
            st.write("Dimensiones actuales:", data.shape)
            st.dataframe(data.head())

        else:
            st.warning("Seleccione al menos una variable para analizar outliers.")

    st.divider()

    # =====================================================
    # ELIMINAR FILAS VACÍAS
    # =====================================================

    st.subheader("Eliminar Filas Vacías")

    if st.button("Eliminar Filas con NaN"):

        data = data.dropna()
        st.session_state.predf = data

        st.success("Filas con valores NaN eliminadas correctamente.")
        st.write("Dimensiones actuales:", data.shape)

    st.divider()

    # =====================================================
    # CODIFICACIÓN DE VARIABLE OBJETIVO
    # =====================================================

    st.subheader("Codificar Variable Objetivo")

    if "dropout" in data.columns:

        if st.button("Codificar dropout"):

            data["dropout"] = (
                data["dropout"]
                .astype(str)
                .str.strip()
                .replace({
                    "No": 0,
                    "NO": 0,
                    "no": 0,
                    "Si": 1,
                    "SI": 1,
                    "Sí": 1,
                    "sí": 1,
                    "si": 1
                })
            )

            st.session_state.predf = data

            st.success("Variable dropout codificada: No = 0, Si = 1.")
            st.dataframe(data[["dropout"]].head())

    else:
        st.info("No se encontró la variable dropout.")

    st.divider()

    # =====================================================
    # LABEL ENCODER PARA VARIABLES CATEGÓRICAS
    # =====================================================

    st.subheader("Codificación de Variables Categóricas")

    if st.button("Aplicar LabelEncoder"):

        encoders = {}

        columnas_object = data.select_dtypes(include="object").columns

        for col in columnas_object:

            le = LabelEncoder()

            data[col] = le.fit_transform(
                data[col].astype(str)
            )

            encoders[col] = le

        st.session_state.predf = data
        st.session_state.encoders = encoders

        with open("encoders.pkl", "wb") as archivo:
            pickle.dump(encoders, archivo)

        st.success("Variables categóricas codificadas y encoders guardados.")
        st.dataframe(data.head())

    st.divider()

    # =====================================================
    # ESCALADO
    # =====================================================

    st.subheader("Escalado de Variables Numéricas")

    columnas_scaler = st.multiselect(
        "Seleccione variables para StandardScaler",
        [
            col for col in data.select_dtypes(
                include=["int64", "float64"]
            ).columns
            if col != "dropout"
        ]
    )

    if st.button("Aplicar StandardScaler"):

        if len(columnas_scaler) > 0:

            scaler = StandardScaler()

            data[columnas_scaler] = scaler.fit_transform(
                data[columnas_scaler]
            )

            st.session_state.predf = data
            st.session_state.scaler = scaler
            st.session_state.columnas_scaler = columnas_scaler

            with open("scaler.pkl", "wb") as archivo:
                pickle.dump(
                    {
                        "scaler": scaler,
                        "columnas_scaler": columnas_scaler
                    },
                    archivo
                )

            st.success("Escalado realizado correctamente y scaler guardado.")
            st.dataframe(data.head())

        else:
            st.warning("Seleccione al menos una variable para escalar.")

    st.divider()

    # =====================================================
    # MOSTRAR DATASET FINAL
    # =====================================================

    st.subheader("Dataset Procesado")

    st.write("Dimensiones del dataset procesado:")
    st.write(st.session_state.predf.shape)

    st.dataframe(st.session_state.predf.head())

    # =====================================================
    # DESCARGAR DATASET PROCESADO
    # =====================================================

    st.subheader("Descargar Dataset Procesado")

    csv = st.session_state.predf.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Descargar datos_preprocesados.csv",
        data=csv,
        file_name="datos_preprocesados.csv",
        mime="text/csv"
    )


# =========================================================
# EJECUTAR
# =========================================================


import streamlit as st
import pandas as pd

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

    # =====================================================
    # VALIDAR DATASET
    # =====================================================

    if "df" not in st.session_state:

        st.image(
            "images/casa.png",
            width=300
        )

        st.warning(
            "Debe ingresar el dataset primero."
        )

        return

    # =====================================================
    # DATASET ORIGINAL
    # =====================================================

    if "original_df" not in st.session_state:

        st.session_state.original_df = (
            st.session_state.df.copy()
        )

    if "predf" not in st.session_state:

        st.session_state.predf = (
            st.session_state.df.copy()
        )

    data = st.session_state.predf

    # =====================================================
    # RESTABLECER DATASET
    # =====================================================

    st.markdown(
        "### Restablecer Dataset"
    )

    reset_button = st.button(
        "Restablecer Preprocesado"
    )

    if reset_button:

        st.session_state.predf = (
            st.session_state.original_df.copy()
        )

        data = st.session_state.predf

        st.success(
            "Dataset restablecido correctamente"
        )

    st.divider()

    # =====================================================
    # ELIMINAR COLUMNAS
    # =====================================================

    st.subheader("Eliminar Columnas")

    columnas_eliminar = st.multiselect(
        "Seleccione columnas",
        data.columns
    )

    if st.button("Eliminar Columnas"):

        data = data.drop(
            columns=columnas_eliminar
        )

        st.session_state.predf = data

        st.success("Columnas eliminadas")

        st.dataframe(data.head())

    st.divider()

    # =====================================================
    # VALORES NULOS CATEGÓRICOS
    # =====================================================

    st.subheader(
        "Valores Nulos Categóricos"
    )

    variables_cat = st.multiselect(
        "Variables categóricas",
        data.select_dtypes(
            include="object"
        ).columns
    )

    metodo_cat = st.selectbox(
        "Método categórico",
        ["most_frequent", "constant"]
    )

    fill_value = None

    if metodo_cat == "constant":

        fill_value = st.text_input(
            "Valor de reemplazo"
        )

    if st.button("Reemplazar Nulos Categóricos"):

        imputer = SimpleImputer(
            strategy=metodo_cat,
            fill_value=fill_value
        )

        data[variables_cat] = imputer.fit_transform(
            data[variables_cat]
        )

        st.session_state.predf = data

        st.success(
            "Valores categóricos reemplazados"
        )

    st.divider()

    # =====================================================
    # VALORES NULOS NUMÉRICOS
    # =====================================================

    st.subheader(
        "Valores Nulos Numéricos"
    )

    variables_num = st.multiselect(
        "Variables numéricas",
        data.select_dtypes(
            include=["int64", "float64"]
        ).columns
    )

    metodo_num = st.selectbox(
        "Método numérico",
        ["SimpleImputer", "KNNImputer"]
    )

    if metodo_num == "SimpleImputer":

        estrategia = st.selectbox(
            "Estrategia",
            [
                "mean",
                "median",
                "most_frequent"
            ]
        )

    else:

        vecinos = st.slider(
            "Número de vecinos",
            1,
            10,
            5
        )

    if st.button("Reemplazar Nulos Numéricos"):

        if metodo_num == "SimpleImputer":

            imputer = SimpleImputer(
                strategy=estrategia
            )

        else:

            imputer = KNNImputer(
                n_neighbors=vecinos
            )

        data[variables_num] = (
            imputer.fit_transform(
                data[variables_num]
            )
        )

        st.session_state.predf = data

        st.success(
            "Valores numéricos reemplazados"
        )

    st.divider()

    # =====================================================
    # ELIMINAR ATÍPICOS
    # =====================================================

    st.subheader(
        "Eliminar Valores Atípicos"
    )

    variables_outliers = st.multiselect(
        "Variables para outliers",
        data.select_dtypes(
            include=["int64", "float64"]
        ).columns
    )

    metodo_outlier = st.selectbox(
        "Método",
        ["Boxplot", "Isolation Forest"]
    )

    if st.button("Eliminar Atípicos"):

        # -------------------------------------------------
        # BOXPLOT
        # -------------------------------------------------

        if metodo_outlier == "Boxplot":

            for col in variables_outliers:

                Q1 = data[col].quantile(0.25)

                Q3 = data[col].quantile(0.75)

                IQR = Q3 - Q1

                inferior = Q1 - 1.5 * IQR

                superior = Q3 + 1.5 * IQR

                data = data[
                    (data[col] >= inferior)
                    &
                    (data[col] <= superior)
                ]

        # -------------------------------------------------
        # ISOLATION FOREST
        # -------------------------------------------------

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

        st.success(
            "Valores atípicos eliminados"
        )

        st.dataframe(data.head())

    st.divider()

    # =====================================================
    # ELIMINAR FILAS VACÍAS
    # =====================================================

    st.subheader(
        "Eliminar Filas Vacías"
    )

    if st.button("Eliminar Filas con NaN"):

        data = data.dropna()

        st.session_state.predf = data

        st.success(
            "Filas eliminadas correctamente"
        )

    st.divider()

    # =====================================================
    # LABEL ENCODER
    # =====================================================

    st.subheader(
        "Codificación de Variables"
    )

    if st.button("Aplicar LabelEncoder"):

        le = LabelEncoder()

        for col in data.select_dtypes(
            include="object"
        ).columns:

            data[col] = le.fit_transform(
                data[col]
            )

        st.session_state.predf = data

        st.success(
            "Variables codificadas"
        )

    st.divider()

    # =====================================================
    # ESCALADO
    # =====================================================

    st.subheader(
        "Escalado de Variables"
    )

    columnas_scaler = st.multiselect(
        "Variables para StandardScaler",
        data.select_dtypes(
            include=["int64", "float64"]
        ).columns
    )

    if st.button("Aplicar StandardScaler"):

        scaler = StandardScaler()

        data[columnas_scaler] = scaler.fit_transform(
            data[columnas_scaler]
        )

        st.session_state.predf = data

        st.success(
            "Escalado realizado correctamente"
        )

    st.divider()

    # =====================================================
    # MOSTRAR DATASET FINAL
    # =====================================================

    st.subheader(
        "Dataset Procesado"
    )

    st.dataframe(
        st.session_state.predf.head()
    )


# =========================================================
# EJECUTAR
# =========================================================

preprocesamiento()
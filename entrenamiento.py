import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score
)

import plotly.graph_objs as go
import pandas as pd
import pickle
import plotly.figure_factory as ff


# =========================================================
# CONVERTIR MODELO A BYTES
# =========================================================

def convert_model_to_bytes(model):

    model_bytes = pickle.dumps(model)

    return model_bytes


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def entrenamiento():

    st.markdown("# 📊 Entrenamiento de Datos")

    # =====================================================
    # VALIDAR DATASET
    # =====================================================

    if "df" not in st.session_state:

        st.image("images/casa.png", width=300)

        st.warning(
            "Debe ingresar el dataset primero."
        )

        return

    # =====================================================
    # DATASET
    # =====================================================

    data = st.session_state.df.copy()

    # =====================================================
    # CONFIGURACIÓN
    # =====================================================

    st.markdown("### Parámetros del Modelo")

    modelo = st.sidebar.selectbox(
        "Seleccione el modelo",
        (
            "Regresión Logística",
            "KNN",
            "Árbol de Decisión",
            "Bosque Aleatorio"
        )
    )

    randomstate = st.sidebar.number_input(
        "Random State",
        min_value=0,
        max_value=1000,
        value=42
    )

    train_perc = st.sidebar.slider(
        "Porcentaje entrenamiento",
        50,
        90,
        80
    )

    test_perc = 100 - train_perc

    # =====================================================
    # MODELOS
    # =====================================================

    parametros = {}

    # -----------------------------------------------------
    # REGRESIÓN LOGÍSTICA
    # -----------------------------------------------------

    if modelo == "Regresión Logística":

        solver = st.sidebar.selectbox(
            "Solver",
            ("lbfgs", "liblinear", "saga")
        )

        penalty = st.sidebar.selectbox(
            "Penalty",
            ("l2", "l1", None)
        )

        # Validación

        if solver == "lbfgs" and penalty not in ["l2", None]:

            st.warning(
                "lbfgs solo soporta l2 o None"
            )

            penalty = "l2"

        if solver == "liblinear" and penalty not in ["l1", "l2"]:

            st.warning(
                "liblinear solo soporta l1 o l2"
            )

            penalty = "l2"

        C = st.sidebar.number_input(
            "C",
            min_value=0.01,
            max_value=100.0,
            value=1.0
        )

        modelo_entrenar = LogisticRegression(
            solver=solver,
            penalty=penalty,
            C=C
        )

        parametros = {
            "solver": solver,
            "penalty": penalty,
            "C": C
        }

    # -----------------------------------------------------
    # KNN
    # -----------------------------------------------------

    elif modelo == "KNN":

        n_neighbors = st.sidebar.slider(
            "N Neighbors",
            1,
            20,
            5
        )

        weights = st.sidebar.selectbox(
            "Weights",
            ("uniform", "distance")
        )

        modelo_entrenar = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights
        )

        parametros = {
            "n_neighbors": n_neighbors,
            "weights": weights
        }

    # -----------------------------------------------------
    # ÁRBOL DE DECISIÓN
    # -----------------------------------------------------

    elif modelo == "Árbol de Decisión":

        criterion = st.sidebar.selectbox(
            "Criterion",
            ("gini", "entropy")
        )

        max_depth = st.sidebar.slider(
            "Max Depth",
            1,
            20,
            5
        )

        modelo_entrenar = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth
        )

        parametros = {
            "criterion": criterion,
            "max_depth": max_depth
        }

    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    elif modelo == "Bosque Aleatorio":

        n_estimators = st.sidebar.slider(
            "N Estimators",
            10,
            300,
            100
        )

        max_depth = st.sidebar.slider(
            "Max Depth",
            1,
            20,
            5
        )

        modelo_entrenar = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=randomstate
        )

        parametros = {
            "n_estimators": n_estimators,
            "max_depth": max_depth
        }

    # =====================================================
    # MOSTRAR PARÁMETROS
    # =====================================================

    st.subheader("Configuración del Entrenamiento")

    st.write(parametros)

    # =====================================================
    # BOTÓN ENTRENAR
    # =====================================================

    boton_entrenar = st.button(
        "Entrenar Modelo"
    )

    if boton_entrenar:

        model = entrenar_modelo(
            data,
            test_perc,
            randomstate,
            modelo_entrenar
        )

        model_data = convert_model_to_bytes(model)

        st.session_state.model_data = model_data

        st.success(
            "Modelo entrenado correctamente"
        )

    # =====================================================
    # DESCARGAR MODELO
    # =====================================================

    if "model_data" in st.session_state:

        st.download_button(
            label="📥 Descargar Modelo",
            data=st.session_state.model_data,
            file_name="modelo.pkl",
            mime="application/octet-stream"
        )


# =========================================================
# ENTRENAR MODELO
# =========================================================

def entrenar_modelo(
    data,
    test_perc,
    randomstate,
    modelo
):

    X = data.drop("Abandono", axis=1)

    y = data["Abandono"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_perc / 100,
        random_state=randomstate
    )

    model = modelo

    model.fit(X_train, y_train)

    # =====================================================
    # PREDICCIONES
    # =====================================================

    y_pred_test = model.predict(X_test)

    y_pred_train = model.predict(X_train)

    # =====================================================
    # MÉTRICAS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Accuracy Test",
            f"{accuracy_score(y_test, y_pred_test):.2f}"
        )

    with col2:

        st.metric(
            "Accuracy Train",
            f"{accuracy_score(y_train, y_pred_train):.2f}"
        )

    st.divider()

    # =====================================================
    # MATRIZ CONFUSIÓN
    # =====================================================

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Matriz de Confusión")

        cm = confusion_matrix(
            y_test,
            y_pred_test
        )

        fig = ff.create_annotated_heatmap(
            z=cm,
            x=["No", "Sí"],
            y=["No", "Sí"],
            colorscale="YlOrBr"
        )

        st.plotly_chart(fig)

    # =====================================================
    # ROC AUC
    # =====================================================

    with col4:

        st.subheader("Curva ROC")

        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            y_prob
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

        fig_roc = go.Figure()

        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"AUC={roc_auc:.2f}"
            )
        )

        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash"),
                name="Aleatorio"
            )
        )

        st.plotly_chart(fig_roc)

    return model


# =========================================================
# EJECUTAR
# =========================================================

entrenamiento()
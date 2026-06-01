import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import plotly.graph_objects as go
import plotly.figure_factory as ff


# =========================================================
# CONVERTIR MODELO A BYTES
# =========================================================

def convert_model_to_bytes(model):
    return pickle.dumps(model)


# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def entrenamiento():

    st.markdown("# 🤖 Entrenamiento de Modelos de Machine Learning")

    st.markdown("""
    En este módulo se entrenan modelos de clasificación para predecir el abandono estudiantil.
    Se utilizan las variables académicas, sociodemográficas y de interacción con el LMS para
    estimar si un estudiante presenta riesgo de abandono.
    """)

    # =====================================================
    # VALIDAR DATASET
    # =====================================================

    if "predf" in st.session_state:
        data = st.session_state.predf.copy()
    elif "df" in st.session_state:
        data = st.session_state.df.copy()
    else:
        st.warning("Debe ingresar y preprocesar el dataset primero.")
        return

    # =====================================================
    # NORMALIZAR NOMBRES DE COLUMNAS
    # =====================================================

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    data = data.rename(
        columns={
            "socioeconomic_level": "socioeconomic_level",
            "socioeconomic_level": "socioeconomic_level",
            "socioecoNomic_level": "socioeconomic_level",
            "task_submisSions": "task_submissions",
            "late_submisSions": "late_submissions"
        }
    )

    # =====================================================
    # VALIDAR VARIABLE OBJETIVO
    # =====================================================

    if "dropout" not in data.columns:
        st.error("No se encontró la variable objetivo 'dropout' en el dataset.")
        st.write("Columnas disponibles:")
        st.write(data.columns.tolist())
        return

    # =====================================================
    # CODIFICAR DROPOUT SI VIENE COMO TEXTO
    # =====================================================

    if data["dropout"].dtype == "object":

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

    data = data[data["dropout"].isin([0, 1])]

    # =====================================================
    # ELIMINAR ID SI EXISTE
    # =====================================================

    if "id" in data.columns:
        data = data.drop(columns=["id"])

    if "student_id" in data.columns:
        data = data.drop(columns=["student_id"])

    # =====================================================
    # VALIDAR QUE TODO SEA NUMÉRICO
    # =====================================================

    columnas_object = data.select_dtypes(include="object").columns.tolist()

    if len(columnas_object) > 0:
        st.error("Existen columnas categóricas sin codificar.")
        st.write("Debe codificar estas columnas antes de entrenar:")
        st.write(columnas_object)
        return

    # =====================================================
    # CONFIGURACIÓN
    # =====================================================

    st.markdown("### Parámetros del modelo")

    modelo = st.sidebar.selectbox(
        "Seleccione el modelo",
        (
            "Regresión Logística",
            "KNN",
            "Árbol de Decisión",
            "Bosque Aleatorio",
            "XGBoost"
        )
    )

    randomstate = st.sidebar.number_input(
        "Random State",
        min_value=0,
        max_value=1000,
        value=42
    )

    train_perc = st.sidebar.slider(
        "Porcentaje de entrenamiento",
        50,
        90,
        80
    )

    test_perc = 100 - train_perc

    # =====================================================
    # MODELOS
    # =====================================================

    parametros = {}

    if modelo == "Regresión Logística":

        solver = st.sidebar.selectbox(
            "Solver",
            ("lbfgs", "liblinear", "saga")
        )

        penalty = st.sidebar.selectbox(
            "Penalty",
            ("l2", "l1", None)
        )

        if solver == "lbfgs" and penalty not in ["l2", None]:
            st.warning("lbfgs solo soporta l2 o None. Se ajustó a l2.")
            penalty = "l2"

        if solver == "liblinear" and penalty not in ["l1", "l2"]:
            st.warning("liblinear solo soporta l1 o l2. Se ajustó a l2.")
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
            C=C,
            max_iter=1000
        )

        parametros = {
            "solver": solver,
            "penalty": penalty,
            "C": C
        }

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
            max_depth=max_depth,
            random_state=randomstate
        )

        parametros = {
            "criterion": criterion,
            "max_depth": max_depth
        }

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

    elif modelo == "XGBoost":

        n_estimators = st.sidebar.slider(
            "N Estimators",
            10,
            300,
            100
        )

        learning_rate = st.sidebar.number_input(
            "Learning Rate",
            min_value=0.01,
            max_value=1.0,
            value=0.1
        )

        max_depth = st.sidebar.slider(
            "Max Depth",
            1,
            20,
            3
        )

        modelo_entrenar = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            eval_metric="logloss",
            random_state=randomstate
        )

        parametros = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth
        }

    # =====================================================
    # MOSTRAR PARÁMETROS
    # =====================================================

    st.subheader("Configuración del entrenamiento")
    st.write(parametros)

    st.subheader("Vista previa del dataset de entrenamiento")
    st.dataframe(data.head())

    # =====================================================
    # BOTÓN ENTRENAR
    # =====================================================

    if st.button("Entrenar Modelo"):

        model = entrenar_modelo(
            data=data,
            test_perc=test_perc,
            randomstate=randomstate,
            modelo=modelo_entrenar
        )

        model_data = convert_model_to_bytes(model)

        st.session_state.model_data = model_data
        st.session_state.modelo_entrenado = model

        st.success("Modelo entrenado correctamente.")

    # =====================================================
    # DESCARGAR MODELO
    # =====================================================

    if "model_data" in st.session_state:

        st.download_button(
            label="📥 Descargar modelo_entrenado.pkl",
            data=st.session_state.model_data,
            file_name="modelo_entrenado.pkl",
            mime="application/octet-stream"
        )


# =========================================================
# ENTRENAR MODELO
# =========================================================

def entrenar_modelo(data, test_perc, randomstate, modelo):

    X = data.drop("dropout", axis=1)
    y = data["dropout"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_perc / 100,
        random_state=randomstate,
        stratify=y
    )

    model = modelo

    model.fit(X_train, y_train)

    # =====================================================
    # PREDICCIONES
    # =====================================================

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    y_prob_test = model.predict_proba(X_test)[:, 1]

    # =====================================================
    # MÉTRICAS
    # =====================================================

    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob_test)

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    cv_mean = cv_scores.mean()

    st.divider()
    st.subheader("Métricas del modelo")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy Test", f"{accuracy_test:.4f}")
    col2.metric("Accuracy Train", f"{accuracy_train:.4f}")
    col3.metric("Cross Validation", f"{cv_mean:.4f}")

    col4, col5, col6, col7 = st.columns(4)

    col4.metric("Precision", f"{precision:.4f}")
    col5.metric("Recall", f"{recall:.4f}")
    col6.metric("F1-score", f"{f1:.4f}")
    col7.metric("ROC-AUC", f"{roc_auc:.4f}")

    resultados = pd.DataFrame({
        "Métrica": [
            "Accuracy Test",
            "Accuracy Train",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
            "Cross Validation"
        ],
        "Valor": [
            accuracy_test,
            accuracy_train,
            precision,
            recall,
            f1,
            roc_auc,
            cv_mean
        ]
    })

    st.dataframe(resultados)

    st.divider()

    # =====================================================
    # REPORTE DE CLASIFICACIÓN
    # =====================================================

    st.subheader("Reporte de clasificación")

    reporte = classification_report(
        y_test,
        y_pred_test,
        output_dict=True,
        zero_division=0
    )

    reporte_df = pd.DataFrame(reporte).transpose()

    st.dataframe(reporte_df)

    st.divider()

    # =====================================================
    # MATRIZ CONFUSIÓN
    # =====================================================

    col8, col9 = st.columns(2)

    with col8:

        st.subheader("Matriz de confusión")

        cm = confusion_matrix(
            y_test,
            y_pred_test
        )

        fig = ff.create_annotated_heatmap(
            z=cm,
            x=["No abandono", "Sí abandono"],
            y=["No abandono", "Sí abandono"],
            colorscale="YlOrBr"
        )

        fig.update_layout(
            xaxis_title="Predicción",
            yaxis_title="Valor real"
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # ROC AUC
    # =====================================================

    with col9:

        st.subheader("Curva ROC")

        fpr, tpr, _ = roc_curve(
            y_test,
            y_prob_test
        )

        fig_roc = go.Figure()

        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"AUC = {roc_auc:.4f}"
            )
        )

        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash"),
                name="Clasificador aleatorio"
            )
        )

        fig_roc.update_layout(
            title="Curva ROC",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )

        st.plotly_chart(fig_roc, use_container_width=True)

    return model


# =========================================================
# EJECUTAR
# =========================================================

entrenamiento()
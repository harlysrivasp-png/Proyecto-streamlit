import streamlit as st
import streamlit_authenticator as stauth
from app import app


st.set_page_config(
    page_title="Predicción de Abandono Estudiantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# USUARIOS
# Usuario: hrivas
# Contraseña: abc
# IMPORTANTE: reemplace el password por el hash generado
# ==========================================================

usuarios = {
    "usernames": {
        "hrivas": {
            "name": "Harlys Rivas",
            "password": "$2b$12$xTIOYm3OuX/x3RkHTz9eYOFAZwlMV2GxxfmLrMNsnDozbfpxbYSSq"
        }
    }
}


authenticator = stauth.Authenticate(
    usuarios,
    "app_abandono_estudiantil",
    "cookie_signature_key",
    cookie_expiry_days=1
)


# ==========================================================
# LOGIN
# ==========================================================

authenticator.login(
    location="main"
)


# ==========================================================
# CONTROL DE ACCESO
# ==========================================================

if st.session_state.get("authentication_status"):

    authenticator.logout(
        button_name="Cerrar sesión",
        location="sidebar"
    )

    st.sidebar.success(
        f"Bienvenido, {st.session_state.get('name')}"
    )

    app()

elif st.session_state.get("authentication_status") is False:

    st.error("Usuario o contraseña incorrectos.")

else:

    st.warning("Ingrese usuario y contraseña para continuar.")

# theme_toggle.py
import streamlit as st

# Inicializar el estado del tema
def initialize_theme():
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False  # Por defecto, inicia en modo claro

# Función para alternar el tema
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Función para aplicar los estilos CSS según el modo seleccionado
def apply_styles():
    if st.session_state.dark_mode:
        # Modo oscuro (negro y blanco, minimalista)
        st.markdown(
            """
            <style>
            .main {
                background-color: #000000;
                color: #ffffff;
            }
            [data-testid="stSidebar"] {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            h1, h2, h3, .st-expander, .st-expander p, .st-expander div {
                color: #ffffff;
            }
            .stButton>button {
                color: white;
                background-color: #000000;
                border-radius: 8px;
                border: 2px solid #ffffff;
            }
            .stButton>button:hover {
                background-color: #ffffff;
                color: #000000;
            }
            .stRadio div {
                color: #ffffff; /* Estilo del texto de los radio buttons en modo oscuro */
            }
            .stAlert {
                background-color: #333333;
                color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        # Modo claro (blanco y negro, minimalista)
        st.markdown(
            """
            <style>
            .main {
                background-color: #ffffff;
                color: #000000;
            }
            [data-testid="stSidebar"] {
                background-color: #f0f0f0;
                color: #000000;
            }
            h1, h2, h3, .st-expander, .st-expander p, .st-expander div {
                color: #000000;
            }
            .stButton>button {
                color: white;
                background-color: #000000;
                border-radius: 8px;
                border: 2px solid #000000;
            }
            .stButton>button:hover {
                background-color: #000000;
                color: #ffffff;
            }
            .stRadio div {
                color: #000000; /* Estilo del texto de los radio buttons en modo claro */
            }
            .stAlert {
                background-color: #e6e6e6;
                color: #000000;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

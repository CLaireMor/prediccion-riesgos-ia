import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ----------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------
st.set_page_config(
    page_title="Predicción de Riesgos",
    page_icon="🧠"  # si falla, bórralo
)

st.write("✔️ La app se está ejecutando hasta aquí.")


# ----------------------------
# FUNCIÓN DE ENTRENAMIENTO
# ----------------------------
@st.cache_resource
def cargar_y_entrenar():
    try:
        df = pd.read_csv('obs_salud1.csv', sep=None, engine='python', encoding='latin-1')

        st.write("👀 COLUMNAS DETECTADAS:", df.columns.tolist())
        st.write("📋 Vista previa de datos:", df.head(3))

        return df

    except Exception as e:
        st.error(f"💣 Error durante la carga/entrenamiento: {e}")
        return None


# ----------------------------
# INTERFAZ Web
# ----------------------------
def interfaz_web():
    st.title("🧠 Predicción de Riesgos Laborales")

    df = cargar_y_entrenar()

    if df is not None:
        st.success("Modelo cargado correctamente.")


if __name__ == "__main__":
    interfaz_web()

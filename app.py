import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. DEFINIR LA FUNCIÓN DE ENTRENAMIENTO (LA RECETA) ---
@st.cache_resource
st.set_page_config(page_title="Predicción de Riesgos", page_icon=)

st.write("✅ La app se está ejecutando hasta aquí.")

def cargar_y_entrenar():
    try:
        df = pd.read_csv('obs_salud1.csv', sep=None, engine='python', encoding='latin-1')

        # Debug
        st.write("👀 COLUMNAS DETECTADAS:", df.columns.tolist())
        st.write("📋 Vista previa de datos:", df.head(3))

        # Limpiar espacios invisibles
        df.columns = [c.strip() for c in df.columns]

        # Renombrar (ajusta los nombres EXACTOS a los de tu CSV)
        df.rename(columns={
            'Tipo de lesión o Sistema Comprometido': 'Lesion',
            'Agente probablemente asociado': 'Agente',
            'Ocupación': 'Ocupacion',
            'Sexo': 'Sexo',
            'Edad': 'Edad'
        }, inplace=True)

        columnas_clave = ['Lesion', 'Agente', 'Ocupacion', 'Sexo', 'Edad']
        df = df[columnas_clave].dropna()

        # --- LABEL ENCODERS ---
        le_lesion = LabelEncoder()
        df['Lesion_Num'] = le_lesion.fit_transform(df['Lesion'])

        le_sexo = LabelEncoder()
        df['Sexo_Num'] = le_sexo.fit_transform(df['Sexo'])

        le_agente = LabelEncoder()
        df['Agente_Num'] = le_agente.fit_transform(df['Agente'])

        le_ocupacion = LabelEncoder()
        df['Ocupacion_Num'] = le_ocupacion.fit_transform(df['Ocupacion'])

        # --- FEATURES & TARGET ---
        X = df[['Edad', 'Sexo_Num', 'Agente_Num', 'Ocupacion_Num']]
        y = df['Lesion_Num']

        # --- ESCALADO ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- TRAIN / TEST ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # --- MODELO ---
        modelo_nn = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        )
        modelo_nn.fit(X_train, y_train)

        # 👇 ESTO ES LO QUE FALTABA
        return modelo_nn, scaler, le_sexo, le_agente, le_ocupacion, le_lesion

    except FileNotFoundError:
        st.error("❌ No se encontró 'obs_salud1.csv'. Pon el archivo en la misma carpeta del script.")
        # Devolvemos Nones para que la interfaz lo pueda manejar
        return None, None, None, None, None, None
    # --- 4. PUNTO DE ENTRADA ---
    interfaz_web() 

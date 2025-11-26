import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. DEFINIR LA FUNCIÓN DE ENTRENAMIENTO (LA RECETA) ---
@st.cache_resource
def cargar_y_entrenar():
    try:
        # Cargamos los datos
        df = pd.read_csv('obs_salud1.csv', sep=';', encoding='latin-1')
        
        # Limpieza
        columnas_clave = ['Lesion', 'Agente', 'Ocupacion', 'Sexo', 'Edad']
        df.dropna(subset=columnas_clave, inplace=True)
        
        lesion_counts = df['Lesion'].value_counts()
        df = df[~df['Lesion'].isin(lesion_counts[lesion_counts < 2].index)]

        # Codificación
        le_sexo = LabelEncoder()
        le_agente = LabelEncoder()
        le_ocupacion = LabelEncoder()
        le_lesion = LabelEncoder()

        df['Sexo_Num'] = le_sexo.fit_transform(df['Sexo'])
        df['Agente_Num'] = le_agente.fit_transform(df['Agente'])
        df['Ocupacion_Num'] = le_ocupacion.fit_transform(df['Ocupacion'])
        Y = le_lesion.fit_transform(df['Lesion'])

        X = df[['Edad', 'Sexo_Num', 'Agente_Num', 'Ocupacion_Num']]

        # Escalado
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Entrenamiento
        modelo = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', 
                               solver='adam', max_iter=500, random_state=42)
        modelo.fit(X_scaled, Y)

        return modelo, scaler, le_sexo, le_agente, le_ocupacion, le_lesion
    except Exception as e:
        return None, None, None, None, None, None

# --- 2. EJECUTAR LA CARGA (COCINAR EL PASTEL) ---
# ¡Esto debe ir DESPUÉS de definir la función de arriba!
modelo_nn, scaler, le_sexo, le_agente, le_ocupacion, le_lesion = cargar_y_entrenar()

# --- 3. INTERFAZ GRÁFICA (SERVIR EL PASTEL) ---
def interfaz_web():
    st.title("🏥 Sistema de Predicción de Riesgos con IA")
    st.markdown("---")

    # Verificación de seguridad: Si falló la carga, avisar y detener.
    if modelo_nn is None:
        st.error("❌ Error: No se encuentra el archivo 'obs_salud1.csv'. Asegúrate de haberlo subido al repositorio.")
        return

    st.sidebar.header("Datos del Paciente")
    
    # Widgets
    edad = st.sidebar.slider("Edad", 18, 80, 30)
    sexo = st.sidebar.selectbox("Sexo", sorted(le_sexo.classes_))
    ocupacion = st.sidebar.selectbox("Ocupación", sorted(le_ocupacion.classes_))
    agente = st.sidebar.selectbox("Agente", sorted(le_agente.classes_))

    if st.sidebar.button("ANALIZAR RIESGO"):
        # Crear DataFrame con los datos del usuario
        nuevo_dato = pd.DataFrame([{
            'Edad': edad, 
            'Sexo': sexo, 
            'Ocupacion': ocupacion, 
            'Agente': agente
        }])

        # Transformar (Codificar y Escalar)
        nuevo_dato['Sexo_Num'] = le_sexo.transform(nuevo_dato['Sexo'])
        nuevo_dato['Ocupacion_Num'] = le_ocupacion.transform(nuevo_dato['Ocupacion'])
        nuevo_dato['Agente_Num'] = le_agente.transform(nuevo_dato['Agente'])
        
        X_nuevo = nuevo_dato[['Edad', 'Sexo_Num', 'Agente_Num', 'Ocupacion_Num']]
        X_nuevo_scaled = scaler.transform(X_nuevo)

        # Predicción
        probs = modelo_nn.predict_proba(X_nuevo_scaled)[0]
        
        # Resultados
        st.subheader("🔍 Resultados del Análisis")
        top_indices = probs.argsort()[-3:][::-1]
        
        for i in top_indices:
            lesion = le_lesion.inverse_transform([i])[0]
            prob = probs[i]
            st.write(f"**{lesion}**: {prob:.1%}")
            st.progress(int(prob * 100))

# --- 4. PUNTO DE ENTRADA ---
if __name__ == '__main__':
    interfaz_web()
    
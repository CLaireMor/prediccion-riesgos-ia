# 1. Primero importamos la librería mágica
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# (Asumimos que 'modelo_nn', 'scaler', y los 'LabelEncoders' ya están cargados en memoria
#  como hiciste en la Parte 1, 2 y 3 de tu código original)

def interfaz_web():
    # Título de la App (Reemplaza los print de la línea 94-96)
    st.title("🏥 Sistema de Predicción de Riesgos con IA")
    st.markdown("---") # Una línea separadora elegante

    # --- BARRA LATERAL (SIDEBAR) ---
    # En lugar de mostrar todo hacia abajo, ponemos los controles a la izquierda
    st.sidebar.header("Ingresa los datos del paciente")

    # Reemplazo de widgets.IntSlider 
    edad = st.sidebar.slider("Edad", 18, 80, 30)

    # Reemplazo de widgets.Dropdown [cite: 103, 107, 111]
    # Nota: st.selectbox es el equivalente a Dropdown
    sexo = st.sidebar.selectbox("Sexo", sorted(le_sexo.classes_))
    ocupacion = st.sidebar.selectbox("Ocupación", sorted(le_ocupacion.classes_))
    agente = st.sidebar.selectbox("Agente", sorted(le_agente.classes_))

    # --- BOTÓN DE PREDICCIÓN ---
    # Reemplazo de widgets.Button [cite: 115]
    if st.sidebar.button("ANALIZAR RIESGO CON IA"):

    # 1. Crear el DataFrame con los datos (Igual que en tu línea 126)
        nuevo_caso = pd.DataFrame([{
            'Edad': edad,
            'Sexo': sexo,
            'Ocupacion': ocupacion,
            'Agente': agente
        }])

        # 2. Codificamos y Escalamos (CRÍTICO: Igual que líneas 135-141)
        nuevo_caso['Sexo_Num'] = le_sexo.transform(nuevo_caso['Sexo'])
        nuevo_caso['Ocupacion_Num'] = le_ocupacion.transform(nuevo_caso['Ocupacion'])
        nuevo_caso['Agente_Num'] = le_agente.transform(nuevo_caso['Agente'])

        X_nuevo = nuevo_caso[['Edad', 'Sexo_Num', 'Agente_Num', 'Ocupacion_Num']]
        X_nuevo_scaled = scaler.transform(X_nuevo) # ¡No olvidar el scaler!

        # 3. Predicción (Línea 143)
        probs = modelo_nn.predict_proba(X_nuevo_scaled)[0]

        # 4. Mostrar Resultados
        st.subheader(f"🔍 Resultados del Análisis")
        st.write(f"Perfil analizado: **{sexo}, {edad} años**.")

        # --- GRÁFICA ---
        # Usamos tu misma lógica de gráfica, pero para mostrarla usamos st.pyplot()
        # (Aquí iría tu código de creación de gráfica de pastel líneas 145-163...)

        # Ejemplo simplificado de cómo mostrar la gráfica en Streamlit:
        fig, ax = plt.subplots()
        # ... (Aquí pegas tu código de plt.pie) ...
        # plt.pie(sizes, labels=labels...) 
        
        st.pyplot(fig) # <--- ¡Así de fácil se muestra en la web!
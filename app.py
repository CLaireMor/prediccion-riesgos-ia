import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px   # 👈 para gráfico dinámico


# ----------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------
st.set_page_config(
    page_title="Predicción de Riesgos",
    page_icon="🏥",
)


# ----------------------------
# 1. FUNCIÓN DE CARGA Y ENTRENAMIENTO (CACHEADA)
# ----------------------------
@st.cache_resource
def cargar_y_entrenar():
    """Carga los datos, entrena el modelo y devuelve todo lo necesario."""
    df = pd.read_csv(
        "obs_salud1.csv",
        sep=None,
        engine="python",
        encoding="latin-1",
    )

    # Limpiar espacios en nombres de columnas
    df.columns = [c.strip() for c in df.columns]

    # Renombrar columnas del CSV a nombres simples
    df.rename(
        columns={
            "Tipo de lesión o Sistema Comprometido": "Lesion",
            "Agente probablemente asociado": "Agente",
            "Ocupación": "Ocupacion",
            "Sexo": "Sexo",
            "Edad": "Edad",
        },
        inplace=True,
    )

    columnas_clave = ["Lesion", "Agente", "Ocupacion", "Sexo", "Edad"]
    df = df[columnas_clave].dropna()

    # ----- LABEL ENCODING -----
    le_lesion = LabelEncoder()
    df["Lesion_Num"] = le_lesion.fit_transform(df["Lesion"])

    le_sexo = LabelEncoder()
    df["Sexo_Num"] = le_sexo.fit_transform(df["Sexo"])

    le_agente = LabelEncoder()
    df["Agente_Num"] = le_agente.fit_transform(df["Agente"])

    le_ocupacion = LabelEncoder()
    df["Ocupacion_Num"] = le_ocupacion.fit_transform(df["Ocupacion"])

    # ----- FEATURES & TARGET -----
    X = df[["Edad", "Sexo_Num", "Agente_Num", "Ocupacion_Num"]]
    y = df["Lesion_Num"]

    # ----- ESCALADO -----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----- TRAIN / TEST -----
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ----- MODELO (más liviano) -----
    modelo_nn = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=300,
        random_state=42,
    )
    modelo_nn.fit(X_train, y_train)

    return modelo_nn, scaler, le_sexo, le_agente, le_ocupacion, le_lesion


# ----------------------------
# 2. INTERFAZ WEB
# ----------------------------
def interfaz_web():
    st.title("🏥 Sistema de Predicción de Riesgos con IA")
    st.markdown(
        "Completa la información del trabajador y obtén las lesiones más probables."
    )
    st.markdown("---")

    # Entrenar / cargar modelo (solo la primera vez, gracias a @st.cache_resource)
    with st.spinner("Cargando modelo de IA... (solo la primera vez)"):
        (
            modelo_nn,
            scaler,
            le_sexo,
            le_agente,
            le_ocupacion,
            le_lesion,
        ) = cargar_y_entrenar()

    # Layout: formulario y resultados
    col_form, col_result = st.columns([1, 1])

    # -------- FORMULARIO: SOLO DATOS DEL TRABAJADOR --------
    with col_form:
        st.subheader("🧍 Datos del trabajador")

        # Si quieres SOLO listas desplegables, quita el slider de edad
        edad = st.slider("Edad", 18, 80, 30)

        sexo = st.selectbox("Sexo", sorted(le_sexo.classes_))
        ocupacion = st.selectbox("Ocupación", sorted(le_ocupacion.classes_))
        agente = st.selectbox("Agente", sorted(le_agente.classes_))

        analizar = st.button("🔎 Analizar riesgo")

    probs = None
    top_indices = None
    labels_top = []
    values_top = []

    # -------- PREDICCIÓN --------
    if analizar:
        nuevo_dato = pd.DataFrame(
            [
                {
                    "Edad": edad,
                    "Sexo": sexo,
                    "Ocupacion": ocupacion,
                    "Agente": agente,
                }
            ]
        )

        nuevo_dato["Sexo_Num"] = le_sexo.transform(nuevo_dato["Sexo"])
        nuevo_dato["Ocupacion_Num"] = le_ocupacion.transform(
            nuevo_dato["Ocupacion"]
        )
        nuevo_dato["Agente_Num"] = le_agente.transform(nuevo_dato["Agente"])

        X_nuevo = nuevo_dato[["Edad", "Sexo_Num", "Agente_Num", "Ocupacion_Num"]]
        X_nuevo_scaled = scaler.transform(X_nuevo)

        probs = modelo_nn.predict_proba(X_nuevo_scaled)[0]
        top_indices = probs.argsort()[-3:][::-1]

        for i in top_indices:
            lesion = le_lesion.inverse_transform([i])[0]
            prob = probs[i]
            labels_top.append(lesion)
            values_top.append(prob)

    # -------- RESULTADOS --------
    with col_result:
        st.subheader("📊 Resultados")

        if probs is None:
            st.info("Pulsa **Analizar riesgo** para ver la predicción.")
        else:
            st.markdown("### 🔝 3 lesiones más probables")
            for lesion, prob in zip(labels_top, values_top):
                st.write(f"**{lesion}**: {prob:.1%}")
                st.progress(int(prob * 100))

            # Gráfico dinámico con Plotly
            st.markdown("### 🥧 Distribución de riesgo (Top 3)")

            fig = px.pie(
                values=values_top,
                names=labels_top,
                hole=0.3,
                title="Distribución de riesgo",
            )
            st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# 3. PUNTO DE ENTRADA
# ----------------------------
if __name__ == "__main__":
    interfaz_web()

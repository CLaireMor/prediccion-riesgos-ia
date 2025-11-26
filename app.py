import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ----------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------
st.set_page_config(
    page_title="Predicción de Riesgos",
    page_icon="🏥",  # si te da problema, puedes borrar este parámetro
)


# ----------------------------
# 1. FUNCIÓN DE CARGA Y ENTRENAMIENTO
# ----------------------------
def cargar_y_entrenar():
    st.write("🔄 Iniciando carga y entrenamiento del modelo...")

    try:
        # Lee el archivo CSV
        df = pd.read_csv(
            "obs_salud1.csv",
            sep=None,
            engine="python",
            encoding="latin-1",
        )

        # Debug: mostrar columnas y muestra
        st.write("👀 COLUMNAS DETECTADAS:", df.columns.tolist())
        st.write("📋 Vista previa de datos:", df.head(3))

        # Limpiar espacios invisibles en nombres de columnas
        df.columns = [c.strip() for c in df.columns]

        # Renombrar columnas (AJUSTA ESTOS NOMBRES SI NO COINCIDEN EXACTO)
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

        # Comprobar que todas las columnas existen
        faltantes = [c for c in columnas_clave if c not in df.columns]
        if faltantes:
            st.error(f"❌ Faltan estas columnas después del rename: {faltantes}")
            return None, None, None, None, None, None

        # Mantener solo lo necesario y eliminar filas con NaN
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

        # ----- MODELO -----
        modelo_nn = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
        )
        modelo_nn.fit(X_train, y_train)

        st.success("✅ Modelo entrenado correctamente.")

        return modelo_nn, scaler, le_sexo, le_agente, le_ocupacion, le_lesion

    except FileNotFoundError:
        st.error(
            "❌ No se encontró 'obs_salud1.csv'. "
            "Asegúrate de que esté en la misma carpeta que este script."
        )
        return None, None, None, None, None, None

    except Exception as e:
        st.error(f"💣 Error durante la carga/entrenamiento: {e}")
        return None, None, None, None, None, None


# ----------------------------
# 2. INTERFAZ WEB
# ----------------------------
def interfaz_web():
    st.title("🏥 Sistema de Predicción de Riesgos con IA")
    st.markdown(
        "Selecciona los datos del trabajador y analiza las posibles lesiones más probables."
    )
    st.markdown("---")

    # Entrenamos / cargamos el modelo
    with st.spinner("Entrenando modelo..."):
        (
            modelo_nn,
            scaler,
            le_sexo,
            le_agente,
            le_ocupacion,
            le_lesion,
        ) = cargar_y_entrenar()

    # Si hubo error al entrenar / cargar
    if modelo_nn is None:
        st.warning(
            "No se pudo entrenar el modelo. Revisa los mensajes de error mostrados arriba."
        )
        return

    # --- Layout: formulario a la izquierda, resultados a la derecha ---
    col_form, col_result = st.columns([1, 1])

    # ---------------- FORMULARIO ----------------
    with col_form:
        st.subheader("🧍 Datos del trabajador")

        edad = st.slider("Edad", 18, 80, 30)
        sexo = st.selectbox("Sexo", sorted(le_sexo.classes_))
        ocupacion = st.selectbox("Ocupación", sorted(le_ocupacion.classes_))
        agente = st.selectbox("Agente", sorted(le_agente.classes_))

        analizar = st.button("🔎 Analizar riesgo")

    probs = None
    top_indices = None

    # ---------------- PREDICCIÓN ----------------
    if analizar:
        # Crear DataFrame con los datos del usuario
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

        # Transformar (Codificar y Escalar)
        nuevo_dato["Sexo_Num"] = le_sexo.transform(nuevo_dato["Sexo"])
        nuevo_dato["Ocupacion_Num"] = le_ocupacion.transform(
            nuevo_dato["Ocupacion"]
        )
        nuevo_dato["Agente_Num"] = le_agente.transform(nuevo_dato["Agente"])

        X_nuevo = nuevo_dato[["Edad", "Sexo_Num", "Agente_Num", "Ocupacion_Num"]]
        X_nuevo_scaled = scaler.transform(X_nuevo)

        # Predicción de probabilidades para cada tipo de lesión
        probs = modelo_nn.predict_proba(X_nuevo_scaled)[0]

        # Índices de las 3 lesiones con mayor probabilidad
        top_indices = probs.argsort()[-3:][::-1]

    # ---------------- RESULTADOS ----------------
    with col_result:
        st.subheader("📊 Resultados")

        if probs is None:
            st.info(
                "Completa los datos y pulsa **Analizar riesgo** para ver la predicción."
            )
        else:
            # Mostrar top 3 en texto y barras de progreso
            st.markdown("### 🔝 3 lesiones más probables")

            labels_top = []
            values_top = []

            for i in top_indices:
                lesion = le_lesion.inverse_transform([i])[0]
                prob = probs[i]
                labels_top.append(lesion)
                values_top.append(prob)

                st.write(f"**{lesion}**: {prob:.1%}")
                st.progress(int(prob * 100))

            # Gráfico de torta (pie chart)
            st.markdown("### 🥧 Distribución de riesgo (Top 3)")

            fig, ax = plt.subplots()
            ax.pie(
                values_top,
                labels=labels_top,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.axis("equal")  # Para que el círculo sea redondo
            st.pyplot(fig)


# ----------------------------
# 3. PUNTO DE ENTRADA
# ----------------------------
if __name__ == "__main__":
    interfaz_web()

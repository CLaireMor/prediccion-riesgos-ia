import pandas as pd
import plotly.express as px
import streamlit as st

import modelo as m

# ----------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------
st.set_page_config(
    page_title="Predicción de Riesgos",
    page_icon="🏥",
    layout="wide",
)


# ----------------------------
# 1. CARGA Y ENTRENAMIENTO (CACHEADO)
# ----------------------------
@st.cache_resource(show_spinner="Entrenando y validando el modelo... (solo la primera vez)")
def cargar_y_entrenar() -> m.ResultadoEntrenamiento:
    return m.entrenar()


def pct(x: float) -> str:
    return f"{x:.1%}"


# ----------------------------
# 2. PESTAÑAS
# ----------------------------
def tab_prediccion(res: m.ResultadoEntrenamiento):
    df = res.datos
    col_form, col_result = st.columns([1, 1.4])

    with col_form:
        st.subheader("🧍 Datos del trabajador")
        edad = st.slider("Edad", 14, 95, 45)
        sexo = st.selectbox("Sexo", sorted(df["Sexo"].unique()))
        ocupaciones = df["Ocupacion"].value_counts().index.tolist()  # más frecuentes primero
        ocupacion = st.selectbox("Ocupación", ocupaciones)

        solo_registrados = st.checkbox(
            "Mostrar solo agentes registrados para esta ocupación", value=True
        )
        if solo_registrados:
            agentes = df.loc[df["Ocupacion"] == ocupacion, "Agente"].value_counts().index.tolist()
        else:
            agentes = df["Agente"].value_counts().index.tolist()
        agente = st.selectbox("Agente de riesgo", agentes)
        k = st.slider("Número de lesiones a mostrar", 3, 10, 5)

        if edad < 18:
            st.warning(
                "⚠️ Menor de edad: en Colombia el trabajo de adolescentes requiere "
                "autorización y está prohibido en actividades peligrosas. Considera "
                "activar la ruta de protección."
            )
        elif edad >= 65:
            st.info("ℹ️ Adulto mayor trabajando en informalidad: mayor vulnerabilidad.")

    perfil = {"Edad": edad, "Sexo": sexo, "Ocupacion": ocupacion, "Agente": agente}
    top = m.top_lesiones(res, perfil, k=k)
    similares = m.casos_similares(df, ocupacion, agente)

    with col_result:
        st.subheader("📊 Lesiones más probables")
        for _, fila in top.iterrows():
            razon = fila["Razon"]
            etiqueta = (
                f"{razon:.1f}× más frecuente que en el total de casos"
                if razon >= 1
                else f"{razon:.1f}× (menos frecuente que en el total de casos)"
            )
            st.write(f"**{fila['Lesion']}** — {pct(fila['Probabilidad'])}  ·  _{etiqueta}_")
            st.progress(min(1.0, float(fila["Probabilidad"])))

        largo = top.melt(
            id_vars="Lesion",
            value_vars=["Probabilidad", "Prevalencia"],
            var_name="Serie",
            value_name="Valor",
        )
        largo["Serie"] = largo["Serie"].map(
            {"Probabilidad": "Este perfil", "Prevalencia": "Todos los casos"}
        )
        fig = px.bar(
            largo,
            x="Valor",
            y="Lesion",
            color="Serie",
            barmode="group",
            orientation="h",
            labels={"Valor": "Proporción", "Lesion": ""},
        )
        fig.update_layout(
            yaxis={"categoryorder": "array", "categoryarray": top["Lesion"].tolist()[::-1]},
            xaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, width="stretch")

        n = int(similares["Casos"].sum()) if not similares.empty else 0
        st.markdown(f"#### 🗂️ Evidencia: {n} casos reales con la misma ocupación y agente")
        if n == 0:
            st.warning("No hay casos registrados con esta combinación: la predicción es una extrapolación.")
        else:
            if n < 30:
                st.caption("Pocos casos: interpreta la distribución observada con cautela.")
            st.dataframe(
                similares.head(k).style.format({"Porcentaje": "{:.1%}"}),
                hide_index=True,
                width="stretch",
            )

    st.caption(
        "Estas probabilidades describen **qué lesión es más probable si el trabajador llega a "
        "ser un caso notificado**. El conjunto de datos contiene solo casos, por lo que no "
        "estima la probabilidad de lesionarse (incidencia). Es una herramienta de apoyo a la "
        "prevención, no un diagnóstico."
    )


def tab_lotes(res: m.ResultadoEntrenamiento):
    st.subheader("📂 Predicción para muchos trabajadores")
    st.markdown(
        "Sube un CSV con las columnas **Edad, Sexo, Ocupacion, Agente** (mismos textos que en el "
        "observatorio). Útil para tamizar una base de UTI, una brigada o un censo y priorizar visitas."
    )
    plantilla = res.datos[m.FEATURES].sample(5, random_state=1)
    st.download_button(
        "⬇️ Descargar plantilla de ejemplo",
        plantilla.to_csv(index=False).encode("utf-8-sig"),
        "plantilla_trabajadores.csv",
        "text/csv",
    )
    archivo = st.file_uploader("CSV de trabajadores", type="csv")
    if archivo is None:
        return
    try:
        perfiles = pd.read_csv(archivo, sep=None, engine="python", encoding="utf-8-sig")
        perfiles.columns = [c.strip() for c in perfiles.columns]
        probs = m.predecir(res, perfiles)
    except Exception as e:  # mensaje claro para usuarios no técnicos
        st.error(f"No se pudo procesar el archivo: {e}")
        return

    orden = probs.to_numpy().argsort(axis=1)[:, ::-1]
    clases = probs.columns.to_numpy()
    salida = perfiles.copy()
    for i in range(3):
        salida[f"Lesion_{i + 1}"] = clases[orden[:, i]]
        salida[f"Prob_{i + 1}"] = probs.to_numpy()[range(len(probs)), orden[:, i]].round(3)
    conocidas = set(res.datos["Ocupacion"])
    salida["Ocupacion_conocida"] = perfiles["Ocupacion"].astype(str).str.strip().isin(conocidas)

    st.dataframe(salida, width="stretch")
    if not salida["Ocupacion_conocida"].all():
        st.warning(
            f"{(~salida['Ocupacion_conocida']).sum()} filas tienen ocupaciones que no aparecen en "
            "los datos de entrenamiento; su predicción es poco fiable."
        )
    st.download_button(
        "⬇️ Descargar resultados",
        salida.to_csv(index=False).encode("utf-8-sig"),
        "predicciones.csv",
        "text/csv",
    )


def filtros_exploracion(df: pd.DataFrame) -> pd.DataFrame:
    c1, c2, c3 = st.columns(3)
    anios = sorted(df["Anio"].unique())
    rango = c1.select_slider("Años", anios, value=(anios[0], anios[-1]))
    localidades = c2.multiselect("Localidades", sorted(df["Localidad"].dropna().unique()))
    sexos = c3.multiselect("Sexo", sorted(df["Sexo"].unique()))
    sub = df[df["Anio"].between(*rango)]
    if localidades:
        sub = sub[sub["Localidad"].isin(localidades)]
    if sexos:
        sub = sub[sub["Sexo"].isin(sexos)]
    return sub


def tab_exploracion(res: m.ResultadoEntrenamiento):
    st.subheader("📈 Vigilancia epidemiológica")
    sub = filtros_exploracion(res.datos)
    st.metric("Casos en la selección", f"{len(sub):,}".replace(",", "."))
    if sub.empty:
        return

    c1, c2 = st.columns(2)
    top_les = sub["Lesion"].value_counts().head(10).rename_axis("Lesion").reset_index(name="Casos")
    c1.plotly_chart(
        px.bar(top_les, x="Casos", y="Lesion", orientation="h", title="Lesiones más frecuentes")
        .update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title=""),
        width="stretch",
    )
    top_ag = sub["Agente"].value_counts().head(10).rename_axis("Agente").reset_index(name="Casos")
    c2.plotly_chart(
        px.bar(top_ag, x="Casos", y="Agente", orientation="h", title="Agentes más frecuentes")
        .update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title=""),
        width="stretch",
    )

    principales = sub["Lesion"].value_counts().head(6).index
    tendencia = (
        sub[sub["Lesion"].isin(principales)]
        .groupby(["Anio", "Lesion"]).size()
        .div(sub.groupby("Anio").size(), level="Anio")
        .rename("Proporcion").reset_index()
    )
    st.plotly_chart(
        px.line(tendencia, x="Anio", y="Proporcion", color="Lesion", markers=True,
                title="Evolución anual (proporción de los casos de cada año)")
        .update_layout(yaxis_tickformat=".0%", xaxis_title="Año"),
        width="stretch",
    )

    ocup = sub["Ocupacion"].value_counts().head(12).index
    les = sub["Lesion"].value_counts().head(10).index
    matriz = pd.crosstab(
        sub.loc[sub["Ocupacion"].isin(ocup), "Ocupacion"],
        sub.loc[sub["Ocupacion"].isin(ocup), "Lesion"],
        normalize="index",
    ).reindex(columns=les, fill_value=0)
    st.plotly_chart(
        px.imshow(matriz, text_auto=".0%", aspect="auto", color_continuous_scale="Blues",
                  title="Perfil de lesiones por ocupación (% de los casos de cada ocupación)")
        .update_layout(xaxis_title="", yaxis_title="", height=600),
        width="stretch",
    )


def tab_priorizacion(res: m.ResultadoEntrenamiento):
    st.subheader("🎯 Priorización de intervenciones")
    st.markdown(
        "Combina **carga de casos** con **vulnerabilidad social** para decidir dónde intervenir "
        "primero. Un caso en un trabajador sin aseguramiento y con ingresos bajo el mínimo "
        "tiene menos capacidad de recuperarse y de acceder a atención."
    )
    df = res.datos
    minimo = st.slider("Mínimo de casos por ocupación", 10, 300, 50, step=10)
    g = df.groupby("Ocupacion")
    tabla = pd.DataFrame(
        {
            "Casos": g.size(),
            "% No asegurado": g["Regimen"].apply(lambda s: (s == "No Asegurado").mean()),
            "% < 1 SMMLV": g["Ingresos"].apply(lambda s: (s == "Menos de 1 SMMLV").mean()),
            "% pago a destajo": g["FormaPago"].apply(lambda s: (s == "A destajo").mean()),
            "% ≥ 65 años": g["Edad"].apply(lambda s: (s >= 65).mean()),
            "Menores de 18": g["Edad"].apply(lambda s: int((s < 18).sum())),
            "Lesión principal": g["Lesion"].agg(lambda s: s.value_counts().index[0]),
            "Agente principal": g["Agente"].agg(lambda s: s.value_counts().index[0]),
        }
    )
    tabla = tabla[tabla["Casos"] >= minimo]
    tabla["Índice de vulnerabilidad"] = tabla[
        ["% No asegurado", "% < 1 SMMLV", "% ≥ 65 años"]
    ].mean(axis=1)
    tabla = tabla.sort_values("Índice de vulnerabilidad", ascending=False)

    fig = px.scatter(
        tabla.reset_index(), x="Casos", y="Índice de vulnerabilidad", size="Casos",
        hover_name="Ocupacion", log_x=True,
        title="Carga vs. vulnerabilidad (arriba a la derecha = prioridad)",
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    formato = {c: "{:.0%}" for c in tabla.columns if c.startswith("%") or c.startswith("Índice")}
    st.dataframe(tabla.style.format(formato), width="stretch")
    st.caption(
        "Índice de vulnerabilidad = promedio simple de % no asegurado, % con ingresos < 1 SMMLV "
        "y % de 65 años o más. Es una heurística transparente, no un índice validado; ajusta "
        "los pesos según el criterio del equipo de salud pública."
    )

    menores = df[df["Edad"] < 18]
    if not menores.empty:
        with st.expander(f"🚸 {len(menores)} casos en menores de 18 años (posible trabajo infantil/adolescente)"):
            st.dataframe(
                menores[["Anio", "Localidad", "Edad", "Sexo", "Ocupacion", "Agente", "Lesion"]],
                hide_index=True, width="stretch",
            )


def tab_calidad(res: m.ResultadoEntrenamiento):
    st.subheader("🧪 ¿Qué tan confiable es el modelo?")
    ev = res.evaluacion.copy()
    st.dataframe(
        ev.rename(columns={"top1": "Acierto (1ª opción)", "top3": "Acierto en top 3",
                           "log_loss": "Log-loss (menor = mejor)", "n": "Casos de prueba"})
        .style.format({"Acierto (1ª opción)": "{:.1%}", "Acierto en top 3": "{:.1%}",
                       "Log-loss (menor = mejor)": "{:.2f}"}),
        hide_index=True, width="stretch",
    )
    st.markdown(
        f"""
- **Línea base** = predecir siempre las lesiones más comunes, sin mirar al trabajador. El
  modelo solo aporta valor en la medida en que la supera.
- **Validación temporal** (entrena con años anteriores a {m.ANIO_CORTE_TEMPORAL}, prueba con
  los siguientes) es la cifra realista: el acierto cae frente a la validación aleatoria porque
  la forma de registrar los casos cambia entre años (ver abajo).
- El modelo usa **Edad, Sexo, Ocupación y Agente**. El agente de exposición es, con
  diferencia, la variable más informativa.
"""
    )

    df = res.datos
    st.markdown("#### Cambios en el registro a lo largo del tiempo")
    genericas = df["Lesion"].str.startswith("Otr")
    por_anio = genericas.groupby(df["Anio"]).mean().rename("Proporción").reset_index()
    st.plotly_chart(
        px.bar(por_anio, x="Anio", y="Proporción",
               title="Casos codificados con categorías genéricas ('Otros…', 'Otras…')")
        .update_layout(yaxis_tickformat=".0%", xaxis_title="Año"),
        width="stretch",
    )
    st.caption(
        "Picos de categorías genéricas sugieren cambios en la codificación o en el personal que "
        "registra, no necesariamente cambios reales en la salud de los trabajadores. Vale la pena "
        "revisarlo con el equipo del subsistema de vigilancia."
    )

    agrupadas = df[df["Lesion_Modelo"] == m.ETIQUETA_OTRAS]
    with st.expander(f"Registros agrupados como poco frecuentes o múltiples ({len(agrupadas)})"):
        st.dataframe(
            agrupadas["Lesion"].value_counts().rename_axis("Lesión registrada").reset_index(name="Casos"),
            hide_index=True, width="stretch",
        )


# ----------------------------
# 3. INTERFAZ
# ----------------------------
def interfaz_web():
    st.title("🏥 Sistema de Predicción de Riesgos con IA")
    st.markdown(
        "Lesiones más probables en trabajadores informales según su ocupación y exposición, "
        "con datos del observatorio de salud de Bogotá."
    )
    res = cargar_y_entrenar()

    tabs = st.tabs(
        ["🔎 Predicción", "📂 Por lotes", "📈 Vigilancia", "🎯 Priorización", "🧪 Confiabilidad"]
    )
    with tabs[0]:
        tab_prediccion(res)
    with tabs[1]:
        tab_lotes(res)
    with tabs[2]:
        tab_exploracion(res)
    with tabs[3]:
        tab_priorizacion(res)
    with tabs[4]:
        tab_calidad(res)


if __name__ == "__main__":
    interfaz_web()

"""Carga, limpieza, entrenamiento y evaluación del modelo de lesiones.

Se separa de la interfaz para poder probarlo y reutilizarlo (p. ej. en
predicciones por lotes) sin depender de Streamlit.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RUTA_DATOS = Path(__file__).parent / "obs_salud1.csv"

# Nombres del CSV -> nombres cortos usados en el código
COLUMNAS = {
    "Año": "Anio",
    "Localidad de ocurrencia del caso": "Localidad",
    "Sexo": "Sexo",
    "Edad": "Edad",
    "Régimen aseguramiento en salud": "Regimen",
    "Escolaridad": "Escolaridad",
    "Ocupación": "Ocupacion",
    "Tipo UTI": "TipoUTI",
    "Clase UTI": "ClaseUTI",
    "Nivel de ingresos": "Ingresos",
    "Forma de Pago": "FormaPago",
    "Síntoma": "Sintoma",
    "Agente probablemente asociado": "Agente",
    "Tipo de lesión o Sistema Comprometido": "Lesion",
}

# Variables de entrada del modelo. Solo se usan las que describen al
# trabajador y su exposición; ver README (sección "Decisiones de modelado")
# sobre por qué no se incluyen Año ni Localidad.
FEATURES_CAT = ["Sexo", "Ocupacion", "Agente"]
FEATURES_NUM = ["Edad"]
FEATURES = FEATURES_NUM + FEATURES_CAT
OBJETIVO = "Lesion"

# Lesiones con menos casos que esto se agrupan: no hay datos suficientes
# para estimarlas y la mayoría son registros con varias lesiones pegadas.
MIN_CASOS_LESION = 30
ETIQUETA_OTRAS = "Otras lesiones / registros múltiples (poco frecuentes)"

# Categorías de ocupación/agente con menos casos se tratan como "infrecuentes"
MIN_FRECUENCIA_CATEGORIA = 10

# Año desde el que se valida en la evaluación temporal
ANIO_CORTE_TEMPORAL = 2024


def cargar_datos(ruta=RUTA_DATOS) -> pd.DataFrame:
    """Lee el CSV del observatorio y normaliza textos y categorías."""
    df = pd.read_csv(ruta, sep=";", encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=COLUMNAS)

    for col in df.columns:
        if col in ("Anio", "Edad"):
            continue
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        # "0" aparece como valor basura en algunas columnas de texto
        df.loc[df[col].isin(["0", ""]), col] = pd.NA

    # Registros idénticos en las 14 variables (incluidos edad y síntoma) se
    # tratan como notificaciones repetidas del mismo caso.
    df = df.drop_duplicates()
    df = df.dropna(subset=FEATURES + [OBJETIVO]).copy()

    frecuencias = df[OBJETIVO].value_counts()
    raras = frecuencias[frecuencias < MIN_CASOS_LESION].index
    df["Lesion_Modelo"] = df[OBJETIVO].where(
        ~df[OBJETIVO].isin(raras), ETIQUETA_OTRAS
    )
    return df.reset_index(drop=True)


def construir_pipeline() -> Pipeline:
    """One-hot para categóricas (sin orden artificial) + regresión logística.

    La versión anterior usaba LabelEncoder + MLP: eso convierte ocupaciones y
    agentes en números (0, 1, 2...) como si tuvieran orden, y el modelo
    rendía peor que esta alternativa más simple e interpretable.
    """
    preprocesado = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_FRECUENCIA_CATEGORIA,
                ),
                FEATURES_CAT,
            ),
            ("num", StandardScaler(), FEATURES_NUM),
        ]
    )
    return Pipeline(
        [
            ("prep", preprocesado),
            ("clf", LogisticRegression(max_iter=3000, C=0.3)),
        ]
    )


def _metricas(modelo, X, y) -> dict:
    """Métricas en un conjunto de prueba, ignorando clases no vistas."""
    clases = modelo.classes_
    mascara = y.isin(clases).to_numpy()
    X, y = X[mascara], y[mascara]
    P = modelo.predict_proba(X)
    return {
        "n": int(len(y)),
        "top1": accuracy_score(y, clases[P.argmax(1)]),
        "top3": top_k_accuracy_score(y, P, k=3, labels=clases),
        "log_loss": log_loss(y, P, labels=clases),
    }


def _metricas_linea_base(y_train, y_test) -> dict:
    """Predecir siempre la prevalencia general (sin mirar al trabajador)."""
    prior = y_train.value_counts(normalize=True).sort_index()
    clases = prior.index.to_numpy()
    mascara = y_test.isin(clases).to_numpy()
    y_test = y_test[mascara]
    P = np.tile(prior.to_numpy(), (len(y_test), 1))
    return {
        "n": int(len(y_test)),
        "top1": accuracy_score(y_test, clases[P.argmax(1)]),
        "top3": top_k_accuracy_score(y_test, P, k=3, labels=clases),
        "log_loss": log_loss(y_test, P, labels=clases),
    }


@dataclass
class ResultadoEntrenamiento:
    modelo: Pipeline
    prevalencia: pd.Series
    evaluacion: pd.DataFrame
    datos: pd.DataFrame = field(repr=False)


def evaluar(df: pd.DataFrame) -> pd.DataFrame:
    """Compara el modelo contra la línea base con dos esquemas de validación.

    - Aleatoria (80/20 estratificada): optimista, mezcla años.
    - Temporal (entrena < 2024, prueba >= 2024): simula el uso real, predecir
      casos futuros. Es la cifra honesta para reportar.
    """
    y = df["Lesion_Modelo"]
    filas = []

    idx_tr, idx_te = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y
    )
    esquemas = {
        "Aleatoria 80/20": (idx_tr, idx_te),
        f"Temporal (< {ANIO_CORTE_TEMPORAL} → ≥ {ANIO_CORTE_TEMPORAL})": (
            df.index[df["Anio"] < ANIO_CORTE_TEMPORAL],
            df.index[df["Anio"] >= ANIO_CORTE_TEMPORAL],
        ),
    }
    for nombre, (tr, te) in esquemas.items():
        modelo = construir_pipeline().fit(df.loc[tr, FEATURES], y[tr])
        m = _metricas(modelo, df.loc[te, FEATURES], y[te])
        b = _metricas_linea_base(y[tr], y[te])
        filas.append({"Validación": nombre, "Método": "Modelo", **m})
        filas.append({"Validación": nombre, "Método": "Línea base (prevalencia)", **b})
    return pd.DataFrame(filas)


def entrenar(df: pd.DataFrame | None = None, evaluar_modelo=True) -> ResultadoEntrenamiento:
    """Entrena el modelo final con todos los datos y (opcional) lo evalúa."""
    if df is None:
        df = cargar_datos()
    evaluacion = evaluar(df) if evaluar_modelo else pd.DataFrame()
    modelo = construir_pipeline().fit(df[FEATURES], df["Lesion_Modelo"])
    prevalencia = df["Lesion_Modelo"].value_counts(normalize=True)
    return ResultadoEntrenamiento(modelo, prevalencia, evaluacion, df)


def predecir(resultado: ResultadoEntrenamiento, perfiles: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad de cada lesión para cada perfil (filas) + razón vs. prevalencia."""
    faltan = set(FEATURES) - set(perfiles.columns)
    if faltan:
        raise ValueError(f"Faltan columnas: {sorted(faltan)}")
    X = perfiles[FEATURES].copy()
    X["Edad"] = pd.to_numeric(X["Edad"], errors="coerce")
    for c in FEATURES_CAT:
        X[c] = X[c].astype("string").str.strip()
    probs = resultado.modelo.predict_proba(X)
    return pd.DataFrame(probs, columns=resultado.modelo.classes_, index=perfiles.index)


def top_lesiones(resultado: ResultadoEntrenamiento, perfil: dict, k=5) -> pd.DataFrame:
    """Top-k lesiones para un perfil, con la razón frente a la prevalencia general.

    La razón (lift) responde: ¿cuántas veces más frecuente es esta lesión en
    casos con este perfil que en el conjunto de casos? Es más informativa que
    la probabilidad sola porque descuenta las lesiones que son comunes en todos.
    """
    probs = predecir(resultado, pd.DataFrame([perfil])).iloc[0]
    tabla = pd.DataFrame(
        {
            "Lesion": probs.index,
            "Probabilidad": probs.to_numpy(),
            "Prevalencia": resultado.prevalencia.reindex(probs.index).to_numpy(),
        }
    )
    tabla["Razon"] = tabla["Probabilidad"] / tabla["Prevalencia"]
    return tabla.sort_values("Probabilidad", ascending=False).head(k).reset_index(drop=True)


def casos_similares(df: pd.DataFrame, ocupacion: str, agente: str) -> pd.DataFrame:
    """Distribución real (observada) de lesiones para la misma ocupación y agente."""
    sub = df[(df["Ocupacion"] == ocupacion) & (df["Agente"] == agente)]
    if sub.empty:
        return pd.DataFrame(columns=["Lesion", "Casos", "Porcentaje"])
    conteo = sub[OBJETIVO].value_counts()
    return pd.DataFrame(
        {"Lesion": conteo.index, "Casos": conteo.to_numpy(), "Porcentaje": (conteo / conteo.sum()).to_numpy()}
    )

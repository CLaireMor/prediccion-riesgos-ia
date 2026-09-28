"""Depuración y transformación del dataset, con evidencia para el informe.

Genera en informe/:
  - figuras/*.png          gráficos antes/después
  - resultados.json        tablas y cifras que cita el informe
y en salidas/ (no se versiona):
  - obs_salud_limpio.csv       dataset depurado (legible)
  - obs_salud_transformado.csv matriz lista para el modelo (codificada y escalada)

Uso: python preparacion_datos.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

import modelo as m

RAIZ = Path(__file__).parent
FIG = RAIZ / "informe" / "figuras"
SALIDAS = RAIZ / "salidas"

# Orden explícito de las variables con jerarquía
ORDEN_ESCOLARIDAD = [
    "No Fue A La Escuela",
    "Primaria Incompleta",
    "Primaria Completa",
    "Secundaria Incompleta",
    "Secundaria Completa",
    "Técnico Pos Secundaria Incompleto",
    "Técnico Pos Secundaria",  # sin especificar si terminó: se ubica entre incompleto y completo
    "Técnico Pos Secundaria Completo",
    "Universidad Incompleta",
    "Universidad Completa",
    "Posgrado Incompleto",
    "Posgrado Completo",
]
ORDEN_INGRESOS = ["Menos de 1 SMMLV", "1 SMMLV", "Entre 1 y 2 SMMLV", "2 y MÁS SMMLV"]

NOMINALES = ["Sexo", "Localidad", "Regimen", "Ocupacion", "TipoUTI", "ClaseUTI", "FormaPago", "Agente"]
ORDINALES = ["Escolaridad", "Ingresos"]
NUM_Z = ["Edad"]
NUM_MINMAX = ["Anio"]

AZUL, NARANJA, GRIS, TINTA, TINTA2 = "#2a78d6", "#eb6834", "#9aa3a0", "#15201b", "#4b5852"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#c9d1cd",
    "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e3e8e5", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 10.5, "axes.titlecolor": TINTA,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def estructura(df: pd.DataFrame) -> list[dict]:
    """Resumen por variable: tipo, nulos, valores distintos y ejemplo."""
    filas = []
    for c in df.columns:
        s = df[c]
        filas.append({
            "Variable": c,
            "Tipo": "numérica" if pd.api.types.is_numeric_dtype(s) else "categórica",
            "dtype": str(s.dtype),
            "Nulos": int(s.isna().sum()),
            "Distintos": int(s.nunique(dropna=True)),
            "Ejemplo": str(s.dropna().iloc[0])[:40],
        })
    return filas


def depurar(crudo: pd.DataFrame, log: list) -> pd.DataFrame:
    df = crudo.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=m.COLUMNAS)

    # 1. Coherencia de texto: espacios sobrantes crean categorías falsas ("Especial" vs "Especial ")
    antes = {c: df[c].nunique() for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")}
    for c in antes:
        df[c] = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    cambios = {c: (antes[c], int(df[c].nunique())) for c in antes if antes[c] != df[c].nunique()}
    log.append({"paso": "Normalizar espacios en textos", "detalle": {k: f"{a} → {b} categorías" for k, (a, b) in cambios.items()}})

    # 2. Valores incoherentes -> faltantes
    marcadores = {}
    for c in df.columns:
        if c in ("Anio", "Edad"):
            continue
        n = int(df[c].isin(["0", ""]).sum())
        if n:
            marcadores[c] = n
            df.loc[df[c].isin(["0", ""]), c] = pd.NA
    n_nsnr = int((df["Ingresos"] == "NS / NR").sum())
    df.loc[df["Ingresos"] == "NS / NR", "Ingresos"] = pd.NA
    marcadores["Ingresos (NS / NR)"] = n_nsnr
    log.append({"paso": "Valores sin sentido ('0', 'NS / NR') marcados como faltantes", "detalle": marcadores})

    # 3. Rango de las numéricas
    rango = {"Edad": [int(df["Edad"].min()), int(df["Edad"].max())], "Anio": [int(df["Anio"].min()), int(df["Anio"].max())]}
    log.append({"paso": "Rangos numéricos verificados (sin valores imposibles)", "detalle": rango})

    # 4. Duplicados exactos
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log.append({"paso": "Eliminar duplicados exactos (14 variables idénticas)", "detalle": {"filas eliminadas": n_dup}})

    # 5. Faltantes: eliminar si es variable clave u objetivo, moda si es secundaria
    nulos = df.isna().sum()
    nulos = {c: int(v) for c, v in nulos.items() if v}
    clave = [c for c in ["Ocupacion", "Agente", "Sexo", "Edad", "Lesion"] if c in nulos]
    n_antes = len(df)
    df = df.dropna(subset=clave).reset_index(drop=True)
    imputadas = {}
    for c in df.columns:
        if df[c].isna().any():
            moda = df[c].mode().iloc[0]
            imputadas[c] = {"n": int(df[c].isna().sum()), "moda": str(moda)}
            df[c] = df[c].fillna(moda)
    log.append({
        "paso": "Tratar faltantes",
        "detalle": {"nulos detectados": nulos, "filas eliminadas (variable clave u objetivo)": n_antes - len(df),
                    "imputación por moda": imputadas},
    })

    # 6. Objetivo: agrupar lesiones con < 30 casos (en su mayoría registros con varias lesiones pegadas)
    vc = df["Lesion"].value_counts()
    raras = vc[vc < m.MIN_CASOS_LESION].index
    df["Lesion_Modelo"] = df["Lesion"].where(~df["Lesion"].isin(raras), m.ETIQUETA_OTRAS)
    log.append({"paso": "Agrupar clases raras del objetivo", "detalle": {
        "clases antes": int(vc.size), "clases después": int(df["Lesion_Modelo"].nunique()),
        "registros agrupados": int(df["Lesion"].isin(raras).sum())}})

    df["Anio"] = df["Anio"].astype(int)
    df["Edad"] = df["Edad"].astype(int)
    return df


def transformador() -> ColumnTransformer:
    return ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=m.MIN_FRECUENCIA_CATEGORIA,
                                 sparse_output=False), NOMINALES),
        ("ordinal", Pipeline([
            ("codificar", OrdinalEncoder(categories=[ORDEN_ESCOLARIDAD, ORDEN_INGRESOS])),
            ("escalar", MinMaxScaler()),
        ]), ORDINALES),
        ("zscore", StandardScaler(), NUM_Z),
        ("minmax", MinMaxScaler(), NUM_MINMAX),
    ], verbose_feature_names_out=True)


def metricas(y_te, P, clases):
    return {"top1": accuracy_score(y_te, clases[P.argmax(1)]),
            "top3": top_k_accuracy_score(y_te, P, k=3, labels=clases),
            "log_loss": log_loss(y_te, P, labels=clases)}


def impacto_en_modelo(crudo: pd.DataFrame, limpio: pd.DataFrame) -> list[dict]:
    """Compara el tratamiento original (LabelEncoder + MLP) con el nuevo, misma partición."""
    res = []
    # Original: textos sin normalizar, LabelEncoder en todo, 61 clases
    df0 = crudo.copy()
    df0.columns = [c.strip() for c in df0.columns]
    df0 = df0.rename(columns=m.COLUMNAS)[["Lesion", "Agente", "Ocupacion", "Sexo", "Edad"]].dropna()
    X0 = pd.DataFrame({"Edad": df0["Edad"]})
    for c in ["Sexo", "Agente", "Ocupacion"]:
        X0[c] = LabelEncoder().fit_transform(df0[c])
    y0 = df0["Lesion"].str.strip()
    tr, te = train_test_split(df0.index, test_size=0.2, random_state=42)
    p0 = Pipeline([("s", StandardScaler()), ("mlp", MLPClassifier((32, 16), max_iter=300, random_state=42))])
    p0.fit(X0.loc[tr], y0[tr])
    mask = y0[te].isin(p0.classes_)
    res.append({"Tratamiento": "Original: LabelEncoder + MLP (sin depurar)",
                **metricas(y0[te][mask], p0.predict_proba(X0.loc[te][mask]), p0.classes_)})

    # Nuevo: dataset depurado + one-hot + z-score + regresión logística
    y = limpio["Lesion_Modelo"]
    tr, te = train_test_split(limpio.index, test_size=0.2, random_state=42, stratify=y)
    p1 = m.construir_pipeline().fit(limpio.loc[tr, m.FEATURES], y[tr])
    res.append({"Tratamiento": "Depurado: One-Hot + Z-score + Regresión logística",
                **metricas(y[te], p1.predict_proba(limpio.loc[te, m.FEATURES]), p1.classes_)})
    prior = y[tr].value_counts(normalize=True).sort_index()
    res.append({"Tratamiento": "Línea base: predecir la prevalencia",
                **metricas(y[te], np.tile(prior.to_numpy(), (len(te), 1)), prior.index.to_numpy())})
    return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in res]


# ---------------------------- Gráficos ----------------------------
def fig_nulos(crudo_ren, limpio):
    cols = [c for c in crudo_ren.columns if crudo_ren[c].isna().any() or (crudo_ren[c].astype(str).str.strip().isin(["0", "NS / NR"])).any()]
    antes = [int(crudo_ren[c].isna().sum() + crudo_ren[c].astype(str).str.strip().isin(["0", "NS / NR"]).sum()) for c in cols]
    despues = [int(limpio[c].isna().sum()) for c in cols]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    y = np.arange(len(cols))
    ax.barh(y + 0.2, antes, 0.38, color=NARANJA, label="Antes")
    ax.barh(y - 0.2, despues, 0.38, color=AZUL, label="Después")
    for i, v in enumerate(antes):
        ax.text(v + 0.2, i + 0.2, str(v), va="center", fontsize=8.5, color=TINTA)
    for i, v in enumerate(despues):
        ax.text(v + 0.2, i - 0.2, str(v), va="center", fontsize=8.5, color=TINTA)
    ax.set_yticks(y, cols)
    ax.invert_yaxis()
    ax.set_xlabel("Valores faltantes o sin sentido")
    ax.set_title("Faltantes por variable antes y después de la depuración", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG / "01_nulos.png")
    plt.close(fig)


def fig_histogramas(limpio, Xt):
    fig, axs = plt.subplots(1, 3, figsize=(9, 2.9))
    datos = [(limpio["Edad"], "Edad original (años)"), (Xt["zscore__Edad"], "Edad Z-score (media 0, DE 1)"),
             (MinMaxScaler().fit_transform(limpio[["Edad"]]).ravel(), "Edad Min-Max (0 a 1)")]
    for ax, (d, t) in zip(axs, datos):
        ax.hist(d, bins=30, color=AZUL, edgecolor="white", linewidth=0.6)
        ax.set_title(t, loc="left", fontsize=9.5)
        ax.axvline(np.mean(d), color=NARANJA, lw=1.5)
        ax.text(0.98, 0.95, f"media {np.mean(d):.2f}\nDE {np.std(d):.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=TINTA)
    axs[0].set_ylabel("Registros")
    fig.suptitle("Histogramas: el escalado cambia la escala, no la forma de la distribución", x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "02_histogramas_edad.png")
    plt.close(fig)


def fig_boxplots(limpio, Xt):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.1), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axs[0]
    ax.boxplot([limpio["Edad"]], widths=0.5, patch_artist=True,
               boxprops={"facecolor": "#cde2fb", "edgecolor": AZUL}, medianprops={"color": NARANJA, "lw": 2},
               whiskerprops={"color": AZUL}, capprops={"color": AZUL}, flierprops={"marker": "o", "markersize": 3, "markeredgecolor": AZUL})
    ax.set_xticks([1], ["Edad (años)"])
    q1, q3 = limpio["Edad"].quantile([0.25, 0.75])
    lim = q3 + 1.5 * (q3 - q1)
    n_out = int((limpio["Edad"] > lim).sum())
    ax.set_title(f"Edad: {n_out} valores sobre {lim:.0f} años", loc="left", fontsize=9.5)
    ax = axs[1]
    grupos = [Xt["zscore__Edad"], Xt["ordinal__Escolaridad"], Xt["ordinal__Ingresos"], Xt["minmax__Anio"]]
    ax.boxplot(grupos, widths=0.5, patch_artist=True,
               boxprops={"facecolor": "#cde2fb", "edgecolor": AZUL}, medianprops={"color": NARANJA, "lw": 2},
               whiskerprops={"color": AZUL}, capprops={"color": AZUL}, flierprops={"marker": "o", "markersize": 3, "markeredgecolor": AZUL})
    ax.set_xticks([1, 2, 3, 4], ["Edad\n(Z-score)", "Escolaridad\n(ordinal 0–1)", "Ingresos\n(ordinal 0–1)", "Año\n(Min-Max)"])
    ax.set_title("Después: variables numéricas en escalas comparables", loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "03_boxplots.png")
    plt.close(fig)
    return {"limite_superior_iqr": round(float(lim), 1), "atipicos_edad": n_out}


def fig_dispersion(limpio, Xt):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(limpio), 3000, replace=False)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    esc = pd.Categorical(limpio["Escolaridad"], categories=ORDEN_ESCOLARIDAD).codes
    axs[0].scatter(limpio["Edad"].to_numpy()[idx], esc[idx] + rng.uniform(-0.3, 0.3, len(idx)), s=6, alpha=0.35, color=AZUL, edgecolors="none")
    axs[0].set_xlabel("Edad (años)")
    axs[0].set_ylabel("Nivel de escolaridad (código 0–11)")
    axs[0].set_title("Antes: escalas de 15–95 y 0–11", loc="left", fontsize=9.5)
    axs[1].scatter(Xt["zscore__Edad"].to_numpy()[idx], Xt["ordinal__Escolaridad"].to_numpy()[idx] + rng.uniform(-0.03, 0.03, len(idx)),
                   s=6, alpha=0.35, color=AZUL, edgecolors="none")
    axs[1].set_xlabel("Edad (Z-score)")
    axs[1].set_ylabel("Escolaridad (Ordinal + Min-Max)")
    axs[1].set_title("Después: ambas en rangos comparables", loc="left", fontsize=9.5)
    r = np.corrcoef(limpio["Edad"], esc)[0, 1]
    for ax in axs:
        ax.text(0.98, 0.04, f"r de Pearson = {r:.2f}", transform=ax.transAxes, ha="right", fontsize=8, color=TINTA)
    fig.suptitle("Dispersión Edad vs. Escolaridad (muestra de 3.000; la relación se conserva)", x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "04_dispersion.png")
    plt.close(fig)
    return round(float(r), 3)


def fig_objetivo(limpio):
    vc = limpio["Lesion"].value_counts()
    vc2 = limpio["Lesion_Modelo"].value_counts()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.0))
    axs[0].bar(range(len(vc)), vc.to_numpy(), color=[AZUL if v >= m.MIN_CASOS_LESION else NARANJA for v in vc], width=0.85)
    axs[0].set_yscale("log")
    axs[0].axhline(m.MIN_CASOS_LESION, color=TINTA2, lw=1, ls="--")
    axs[0].text(len(vc) - 1, m.MIN_CASOS_LESION * 1.2, "umbral: 30 casos", ha="right", fontsize=8, color=TINTA)
    axs[0].set_title(f"Antes: {len(vc)} clases (naranja = menos de 30 casos)", loc="left", fontsize=9.5)
    axs[0].set_xlabel("Tipo de lesión (ordenado por frecuencia)")
    axs[0].set_ylabel("Registros (escala log)")
    axs[1].bar(range(len(vc2)), vc2.to_numpy(), color=[NARANJA if k == m.ETIQUETA_OTRAS else AZUL for k in vc2.index], width=0.85)
    axs[1].set_yscale("log")
    axs[1].set_title(f"Después: {len(vc2)} clases (naranja = grupo 'poco frecuentes')", loc="left", fontsize=9.5)
    axs[1].set_xlabel("Tipo de lesión (ordenado por frecuencia)")
    for ax in axs:
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(FIG / "05_objetivo.png")
    plt.close(fig)


def fig_impacto(impacto):
    etiquetas = ["Original\nLabelEnc. + MLP", "Depurado\nOne-Hot + RL", "Línea base\nprevalencia"]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 2.9))
    for ax, k, t in [(axs[0], "top1", "Acierto en la 1ª opción"), (axs[1], "top3", "Acierto en el top 3")]:
        v = [r[k] for r in impacto]
        ax.bar(range(3), v, color=[GRIS, AZUL, "#d5ddd8"], width=0.6)
        for i, x in enumerate(v):
            ax.text(i, x + 0.01, f"{x:.1%}".replace(".", ","), ha="center", fontsize=9, color=TINTA)
        ax.set_xticks(range(3), etiquetas, fontsize=8.5)
        ax.set_ylim(0, max(v) * 1.2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(t, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "06_impacto_modelo.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    SALIDAS.mkdir(exist_ok=True)
    crudo = pd.read_csv(m.RUTA_DATOS, sep=";", encoding="latin-1")
    crudo.columns = [c.strip() for c in crudo.columns]
    crudo_ren = crudo.rename(columns=m.COLUMNAS)

    log = []
    limpio = depurar(crudo, log)

    ct = transformador()
    X = ct.fit_transform(limpio[NOMINALES + ORDINALES + NUM_Z + NUM_MINMAX])
    Xt = pd.DataFrame(X, columns=ct.get_feature_names_out())
    y = LabelEncoder().fit_transform(limpio["Lesion_Modelo"])
    Xt["objetivo"] = y

    limpio.to_csv(SALIDAS / "obs_salud_limpio.csv", index=False, encoding="utf-8-sig")
    Xt.to_csv(SALIDAS / "obs_salud_transformado.csv", index=False, float_format="%.5g")

    bloques = {}
    for nombre in ct.get_feature_names_out():
        b = nombre.split("__")[0]
        bloques[b] = bloques.get(b, 0) + 1
    columnas_onehot = {c: int(sum(1 for n in ct.get_feature_names_out() if n.startswith(f"onehot__{c}_"))) for c in NOMINALES}

    fig_nulos(crudo_ren, limpio)
    fig_histogramas(limpio, Xt)
    atip = fig_boxplots(limpio, Xt)
    r = fig_dispersion(limpio, Xt)
    fig_objetivo(limpio)
    impacto = impacto_en_modelo(crudo, limpio)
    fig_impacto(impacto)

    num = lambda s: {"media": round(float(s.mean()), 3), "de": round(float(s.std(ddof=0)), 3), "min": round(float(s.min()), 3), "max": round(float(s.max()), 3)}
    resultados = {
        "dimension_antes": list(crudo.shape),
        "dimension_limpio": list(limpio.drop(columns="Lesion_Modelo").shape),
        "dimension_transformado": list(Xt.shape),
        "estructura_antes": estructura(crudo_ren),
        "estructura_despues": estructura(limpio),
        "pasos": log,
        "bloques_transformados": bloques,
        "columnas_onehot": columnas_onehot,
        "orden_escolaridad": ORDEN_ESCOLARIDAD,
        "orden_ingresos": ORDEN_INGRESOS,
        "escalado": {
            "Edad (original)": num(limpio["Edad"]), "Edad (Z-score)": num(Xt["zscore__Edad"]),
            "Año (original)": num(limpio["Anio"]), "Año (Min-Max)": num(Xt["minmax__Anio"]),
            "Escolaridad (ordinal 0–1)": num(Xt["ordinal__Escolaridad"]), "Ingresos (ordinal 0–1)": num(Xt["ordinal__Ingresos"]),
        },
        "atipicos": atip,
        "correlacion_edad_escolaridad": r,
        "impacto_modelo": impacto,
        "menores_18": int((limpio["Edad"] < 18).sum()),
    }
    (RAIZ / "informe" / "resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: resultados[k] for k in ["dimension_antes", "dimension_limpio", "dimension_transformado", "impacto_modelo"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

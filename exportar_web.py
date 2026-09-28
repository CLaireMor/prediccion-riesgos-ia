"""Genera web/index.html: la versión web autónoma de la app (sin servidor).

Exporta los coeficientes del modelo y tablas agregadas a JSON y los inserta
en web/plantilla.html. En el navegador la predicción se calcula con los
mismos coeficientes (softmax de la regresión logística).

Privacidad: solo se exportan conteos agregados, nunca registros individuales.

Uso: python exportar_web.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import modelo as m

RAIZ = Path(__file__).parent
PLANTILLA = RAIZ / "web" / "plantilla.html"
SALIDA = RAIZ / "web" / "index.html"
MARCADOR = "/*__DATOS__*/null"


def exportar_modelo(res: m.ResultadoEntrenamiento) -> dict:
    """Mapa categoría -> columna de la matriz de coeficientes."""
    prep = res.modelo.named_steps["prep"]
    clf = res.modelo.named_steps["clf"]
    enc = prep.named_transformers_["cat"]
    esc = prep.named_transformers_["num"]

    columnas = {}
    inicio = 0
    for i, feat in enumerate(m.FEATURES_CAT):
        cats = list(enc.categories_[i])
        infrecuentes = enc.infrequent_categories_[i]
        infrecuentes = set() if infrecuentes is None else set(infrecuentes)
        frecuentes = [c for c in cats if c not in infrecuentes]
        mapa = {c: inicio + j for j, c in enumerate(frecuentes)}
        n = len(frecuentes)
        col_infrecuente = None
        if infrecuentes:
            col_infrecuente = inicio + n
            for c in infrecuentes:
                mapa[c] = col_infrecuente
            n += 1
        columnas[feat] = {"mapa": mapa, "infrecuente": col_infrecuente}
        inicio += n
    assert inicio + 1 == clf.coef_.shape[1], "Columnas del one-hot no cuadran"

    return {
        "clases": list(clf.classes_),
        "intercepto": np.round(clf.intercept_, 5).tolist(),
        "coef": np.round(clf.coef_, 5).tolist(),
        "columnas": columnas,
        "edad": {"col": inicio, "media": float(esc.mean_[0]), "escala": float(esc.scale_[0])},
    }


def predecir_desde_json(mod: dict, perfil: dict) -> np.ndarray:
    """Réplica exacta del cálculo que hace el navegador (para verificarlo)."""
    W = np.array(mod["coef"])
    z = np.array(mod["intercepto"], dtype=float).copy()
    for feat in m.FEATURES_CAT:
        info = mod["columnas"][feat]
        col = info["mapa"].get(perfil[feat], info["infrecuente"])
        if col is not None:
            z += W[:, col]
    e = mod["edad"]
    z += W[:, e["col"]] * (perfil["Edad"] - e["media"]) / e["escala"]
    z = np.exp(z - z.max())
    return z / z.sum()


def conteos(df, cols, dic):
    """Conteos agregados; los textos se codifican como índices de `dic`."""
    t = df.groupby(cols, observed=True).size().reset_index(name="n")
    for c in cols:
        if c in dic:
            pos = {v: i for i, v in enumerate(dic[c])}
            t[c] = t[c].map(pos)
    return t.astype(int).values.tolist()


def exportar_datos(res: m.ResultadoEntrenamiento) -> dict:
    df = res.datos
    g = df.groupby("Ocupacion")
    vuln = pd.DataFrame(
        {
            "casos": g.size(),
            "noAseg": g["Regimen"].apply(lambda s: (s == "No Asegurado").mean()),
            "bajo1": g["Ingresos"].apply(lambda s: (s == "Menos de 1 SMMLV").mean()),
            "destajo": g["FormaPago"].apply(lambda s: (s == "A destajo").mean()),
            "mayor65": g["Edad"].apply(lambda s: (s >= 65).mean()),
            "menores": g["Edad"].apply(lambda s: int((s < 18).sum())),
            "lesion": g["Lesion"].agg(lambda s: s.value_counts().index[0]),
            "agente": g["Agente"].agg(lambda s: s.value_counts().index[0]),
        }
    ).round(4)

    dic = {c: sorted(df[c].dropna().unique().tolist()) for c in
           ["Ocupacion", "Agente", "Lesion", "Localidad", "Sexo"]}
    ev = res.evaluacion.round(4).to_dict(orient="records")
    return {
        "n": int(len(df)),
        "anios": [int(df["Anio"].min()), int(df["Anio"].max())],
        "dic": dic,
        "prevalencia": res.prevalencia.round(5).to_dict(),
        # [ocupacion, agente, lesion, n] -> casos similares y agentes por ocupación
        "oal": conteos(df, ["Ocupacion", "Agente", "Lesion"], dic),
        # [anio, localidad, sexo, lesion, n] y análogos (índices en `dic`)
        "cuboLesion": conteos(df, ["Anio", "Localidad", "Sexo", "Lesion"], dic),
        "cuboAgente": conteos(df, ["Anio", "Localidad", "Sexo", "Agente"], dic),
        "cuboOcup": conteos(df, ["Anio", "Localidad", "Sexo", "Ocupacion", "Lesion"], dic),
        "vulnerabilidad": vuln.reset_index().values.tolist(),
        "evaluacion": ev,
        "corteTemporal": m.ANIO_CORTE_TEMPORAL,
        "etiquetaOtras": m.ETIQUETA_OTRAS,
    }


def main():
    res = m.entrenar()
    mod = exportar_modelo(res)

    # Verifica que la réplica coincide con scikit-learn en una muestra
    muestra = res.datos[m.FEATURES].sample(200, random_state=0)
    esperado = m.predecir(res, muestra).to_numpy()
    obtenido = np.array([predecir_desde_json(mod, r) for r in muestra.to_dict("records")])
    error = np.abs(esperado - obtenido).max()
    assert error < 1e-3, f"La réplica difiere de scikit-learn: {error}"

    datos = {"modelo": mod, **exportar_datos(res)}
    carga = json.dumps(datos, ensure_ascii=False, separators=(",", ":"))
    html = PLANTILLA.read_text(encoding="utf-8")
    assert MARCADOR in html, "La plantilla no tiene el marcador de datos"
    SALIDA.write_text(html.replace(MARCADOR, carga), encoding="utf-8")
    print(f"{SALIDA} ({SALIDA.stat().st_size / 1e6:.2f} MB, error máx. réplica {error:.1e})")


if __name__ == "__main__":
    main()

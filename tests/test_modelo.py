import pandas as pd
import pytest

import modelo as m


@pytest.fixture(scope="module")
def res():
    return m.entrenar()


def test_datos_limpios(res):
    df = res.datos
    assert not df[m.FEATURES + ["Lesion"]].isna().any().any()
    assert not df["Lesion"].str.endswith(" ").any()
    assert df["Lesion_Modelo"].value_counts().min() >= m.MIN_CASOS_LESION


def test_modelo_supera_linea_base(res):
    ev = res.evaluacion.set_index(["Validación", "Método"])
    for validacion in ev.index.get_level_values(0).unique():
        modelo = ev.loc[(validacion, "Modelo")]
        base = ev.loc[(validacion, "Línea base (prevalencia)")]
        assert modelo["top3"] > base["top3"]
        assert modelo["log_loss"] < base["log_loss"]


def test_probabilidades_suman_uno(res):
    perfil = res.datos[m.FEATURES].iloc[[0]]
    assert m.predecir(res, perfil).sum(axis=1).iloc[0] == pytest.approx(1.0)


def test_categorias_desconocidas_no_fallan(res):
    perfil = pd.DataFrame([{"Edad": 30, "Sexo": "X", "Ocupacion": "Nueva", "Agente": "Otro"}])
    assert m.predecir(res, perfil).shape == (1, len(res.modelo.classes_))


def test_faltan_columnas(res):
    with pytest.raises(ValueError):
        m.predecir(res, pd.DataFrame([{"Edad": 30}]))


def test_top_lesiones_ordenado(res):
    fila = res.datos.iloc[0]
    top = m.top_lesiones(res, fila[m.FEATURES].to_dict(), k=5)
    assert len(top) == 5
    assert top["Probabilidad"].is_monotonic_decreasing

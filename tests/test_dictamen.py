"""Dictamen del auditor. Todo con coeficientes SINTETICOS y sin red (el LLM se mockea)."""
import types

import pandas as pd

from econometria.modelo_event_study import estimar_event_study_twfe, extraer_tabla_coeficientes
from econometria.preparacion_panel import preparar_panel_event_study
from ia import auditor_econometrico as aud


def _coef(taus, beta=0.02):
    return pd.DataFrame({
        "tau": taus,
        "beta": [0.0 if t == -1 else beta for t in taus],
        "std_error": [0.0 if t == -1 else 0.01 for t in taus],
        "pvalue": [float("nan") if t == -1 else 0.04 for t in taus],
        "efecto_pct": [0.0 if t == -1 else (2.718281828 ** beta - 1) * 100 for t in taus],
    })


def _wald(stat=3.0, pval=0.7, df=5):
    return types.SimpleNamespace(stat=stat, pval=pval, df=df)


def test_ventana_corta_no_lanza_y_no_tiene_mas_menos():
    # ventana post=2: no existen tau=6 ni tau=1 en algunos casos
    df = _coef([-3, -2, -1, 0, 1, 2])
    d = aud.generar_dictamen(df, _wald(df=2), api_key=None)
    assert d.motor == "plantilla"
    assert "+-" not in d.texto


def test_ventana_menor_a_6_sin_wald():
    df = _coef([-1, 0, 1])  # sin leads: no hay Wald
    d = aud.generar_dictamen(df, None, api_key=None)
    assert d.motor == "plantilla" and "+-" not in d.texto


def test_df_del_wald_es_coherente():
    df = _coef([-4, -3, -2, -1, 0, 1])  # 3 leads (tau <= -2)
    d = aud.generar_dictamen(df, _wald(df=3), api_key=None)
    assert "(3)" in d.texto and "(5)" not in d.texto


def test_sin_api_key_usa_plantilla_sin_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    d = aud.generar_dictamen(_coef([-2, -1, 0, 1]), _wald(df=1))
    assert d.motor == "plantilla" and d.error is None


def test_si_el_llm_falla_devuelve_plantilla_y_error(monkeypatch):
    def roto(*a, **k):
        raise ConnectionError("sin red (mock)")
    monkeypatch.setattr(aud, "_dictamen_llm", roto)
    d = aud.generar_dictamen(_coef([-2, -1, 0, 1]), _wald(df=1), api_key="clave-falsa")
    assert d.motor == "plantilla"
    assert d.error and "sin red (mock)" in d.error


def test_llm_ok_mockeado(monkeypatch):
    monkeypatch.setattr(aud, "_dictamen_llm", lambda *a, **k: "texto del llm")
    d = aud.generar_dictamen(_coef([-2, -1, 0, 1]), _wald(df=1), api_key="clave-falsa")
    assert d.motor.startswith("llm:") and d.error is None and "texto del llm" in d.texto


def test_integracion_con_estimacion_real_sobre_panel_sintetico(panel_largo):
    """Pipeline completo: el wald que devuelve econometria debe ser consumible por el auditor."""
    df_panel, covars = preparar_panel_event_study(panel_largo)
    res, ordered, wald = estimar_event_study_twfe(df_panel, covars)
    df_coef = extraer_tabla_coeficientes(res, ordered)
    d = aud.generar_dictamen(df_coef, wald, info_modelo=aud.resumen_modelo(res))
    assert d.motor == "plantilla"
    assert "+-" not in d.texto
    assert "(5)" in d.texto  # 5 leads (tau -6..-2) en la ventana completa


def test_wald_como_dict_de_econometria(panel_largo):
    df_panel, covars = preparar_panel_event_study(panel_largo, ventana_pre=4, ventana_post=3)
    res, ordered, wald = estimar_event_study_twfe(df_panel, covars)
    assert isinstance(wald, dict)
    d = aud.generar_dictamen(extraer_tabla_coeficientes(res, ordered), wald)
    assert f"({wald['df']})" in d.texto and "+-" not in d.texto

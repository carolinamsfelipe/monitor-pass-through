"""
tests/test_pipeline.py
Suite de pruebas unitarias para certificar la reproducibilidad del pipeline econométrico.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Incorporar raíz del proyecto al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datos.api_macro_series import descargar_series_macro
from econometria.modelo_ardl import estimar_erpt_ardl, evaluar_estacionariedad
from datos.simulacion_dispersion import capturar_canasta_precios, calcular_metricas_dispersion


def test_descarga_series_macro():
    """Valida la conexión y estructura de las series macroeconómicas oficiales."""
    df = descargar_series_macro(start_date="2022-01-01")
    assert not df.empty, "El DataFrame no debe estar vacío."
    assert "l_ipc" in df.columns
    assert "l_tc" in df.columns
    assert "l_emae" in df.columns
    assert df["tc_oficial"].iloc[-1] > 0


def test_evaluar_estacionariedad():
    """Verifica que el test ADF/KPSS compute estadísticos válidos."""
    serie = pd.Series(np.cumsum(np.random.randn(60)))
    res = evaluar_estacionariedad(serie, "serie_test")
    assert "adf_pvalue" in res
    assert "kpss_pvalue" in res
    assert 0.0 <= res["adf_pvalue"] <= 1.0


def test_estimacion_ardl_erpt():
    """Valida consistencia del modelo ARDL, UECM y cálculo de elasticidades."""
    df = descargar_series_macro(start_date="2018-01-01")
    res, diag, mult = estimar_erpt_ardl(df)
    
    assert diag["erpt_corto_plazo"] > 0.0, "El ERPT de impacto inmediato debe ser positivo."
    assert diag["erpt_largo_plazo"] > 0.0, "El ERPT acumulado de largo plazo debe ser positivo."
    assert len(mult) == 13, "Deben generarse 13 meses de multiplicadores dinámicos (0 a 12)."
    assert diag["r2"] > 0.90, "El modelo debe tener un R2 elevado por ajuste histórico."


def test_simulacion_dispersion_reproducible():
    """Comprueba que la simulación calibrada genere datos consistentes con semilla fija."""
    df1 = capturar_canasta_precios(simular_variaciones=True)
    df2 = capturar_canasta_precios(simular_variaciones=True)
    pd.testing.assert_frame_equal(df1, df2)
    
    metricas = calcular_metricas_dispersion(df1)
    assert not metricas.empty
    assert "coef_variacion" in metricas.columns

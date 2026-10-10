"""Fixtures SINTETICAS (rotuladas): no usan red ni datos reales del INDEC."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Fechas de la ventana [-6, +6] alrededor de dic-2023
FECHAS = pd.date_range("2023-06-01", "2024-06-01", freq="MS")  # 13 meses
N_TRATADOS = 5


def construir_panel_sintetico(seed: int = 0, n_series: int = 12) -> pd.DataFrame:
    """Panel largo SINTETICO: 12 series x 13 meses, 5 'transables'. No son datos reales."""
    rng = np.random.default_rng(seed)
    filas = []
    for i in range(n_series):
        tratado = int(i < N_TRATADOS)
        nivel = 100.0
        for fecha in FECHAS:
            tau = (fecha.year - 2023) * 12 + fecha.month - 12
            nivel *= 1 + 0.03 + 0.005 * rng.standard_normal() + (0.02 if (tratado and tau >= 0) else 0.0)
            filas.append({"serie_id": f"serie_{i}", "producto": f"Producto {i}", "fecha": fecha,
                          "ipc_indice": nivel, "transable": tratado})
    return pd.DataFrame(filas)


@pytest.fixture
def panel_largo() -> pd.DataFrame:
    return construir_panel_sintetico()

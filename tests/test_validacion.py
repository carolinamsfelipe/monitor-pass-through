import pandas as pd
import pytest

from datos.api_indec import CLASIFICACION_DIVISIONES
from datos.validacion import ErrorValidacionDatos, validar_esquema_raw, validar_panel

COLS = list(CLASIFICACION_DIVISIONES)


def _raw_sintetico(n=13):
    d = {"indice_tiempo": pd.date_range("2023-06-01", periods=n, freq="MS").strftime("%Y-%m-%d")}
    for c in COLS:
        d[c] = [100.0 + i for i in range(n)]
    return pd.DataFrame(d)


def test_esquema_valido_no_lanza():
    validar_esquema_raw(_raw_sintetico(), COLS)


def test_esquema_falta_columna():
    with pytest.raises(ErrorValidacionDatos):
        validar_esquema_raw(_raw_sintetico().drop(columns=[COLS[0]]), COLS)


def test_esquema_nan_y_indices_no_positivos():
    df = _raw_sintetico()
    df.loc[3, COLS[1]] = None
    with pytest.raises(ErrorValidacionDatos):
        validar_esquema_raw(df, COLS)
    df = _raw_sintetico()
    df.loc[3, COLS[1]] = 0.0
    with pytest.raises(ErrorValidacionDatos):
        validar_esquema_raw(df, COLS)


def test_esquema_fechas_duplicadas():
    df = _raw_sintetico()
    df.loc[1, "indice_tiempo"] = df.loc[0, "indice_tiempo"]
    with pytest.raises(ErrorValidacionDatos):
        validar_esquema_raw(df, COLS)


def test_panel_valido_y_balance(panel_largo):
    validar_panel(panel_largo, n_series=12, ventana=13)


def test_panel_desbalanceado(panel_largo):
    with pytest.raises(ErrorValidacionDatos):
        validar_panel(panel_largo.iloc[:-1], n_series=12)


def test_panel_cantidad_series_y_duplicados(panel_largo):
    with pytest.raises(ErrorValidacionDatos):
        validar_panel(panel_largo, n_series=11)
    dup = pd.concat([panel_largo, panel_largo.iloc[[0]]])
    with pytest.raises(ErrorValidacionDatos):
        validar_panel(dup, n_series=12)

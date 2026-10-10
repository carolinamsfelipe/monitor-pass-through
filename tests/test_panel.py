import numpy as np
import pytest

from econometria.preparacion_panel import preparar_panel_event_study


def test_panel_tiene_156_filas(panel_largo):
    df_panel, covars = preparar_panel_event_study(panel_largo)
    assert len(df_panel) == 156  # 12 series x 13 meses
    assert df_panel.index.names == ["producto", "fecha"]
    assert len(covars) == 12  # 13 taus menos la base


def test_normalizacion_base100_en_tau_menos_1(panel_largo):
    df_panel, _ = preparar_panel_event_study(panel_largo)
    base = df_panel[df_panel["tau"] == -1]["precio_normalizado_base100"]
    assert len(base) == 12
    assert np.allclose(base, 100.0)


def test_error_si_falta_tau_menos_1(panel_largo):
    sin_base = panel_largo[panel_largo["fecha"] != "2023-11-01"]
    with pytest.raises(ValueError):
        preparar_panel_event_study(sin_base)


def test_error_si_un_producto_no_tiene_tau_menos_1(panel_largo):
    roto = panel_largo[~((panel_largo["serie_id"] == "serie_0") & (panel_largo["fecha"] == "2023-11-01"))]
    with pytest.raises(ValueError):
        preparar_panel_event_study(roto)


@pytest.mark.parametrize("valor", [0.0, -5.0])
def test_error_con_indices_no_positivos(panel_largo, valor):
    panel_largo.loc[panel_largo.index[40], "ipc_indice"] = valor
    with pytest.raises(ValueError):
        preparar_panel_event_study(panel_largo)


def test_error_con_indice_nan(panel_largo):
    panel_largo.loc[panel_largo.index[40], "ipc_indice"] = np.nan
    with pytest.raises(ValueError):
        preparar_panel_event_study(panel_largo)

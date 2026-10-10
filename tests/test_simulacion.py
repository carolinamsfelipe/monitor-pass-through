import warnings

import pandas as pd

from datos.scraping_precios import generar_canasta_simulada


def _gen(seed=42):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generar_canasta_simulada(seed=seed)


def test_reproducible_con_semilla():
    pd.testing.assert_frame_equal(_gen(7), _gen(7))


def test_semillas_distintas_difieren():
    assert not _gen(1)["precio_ars"].equals(_gen(2)["precio_ars"])


def test_columna_fuente_simulado():
    df = _gen()
    assert "fuente" in df.columns
    assert (df["fuente"] == "SIMULADO").all()
    assert len(df) == 12
    assert (df["precio_ars"] > 0).all()


def test_emite_aviso_de_simulacion():
    import pytest
    with pytest.warns(UserWarning, match="SIMULADOS"):
        generar_canasta_simulada(seed=1)

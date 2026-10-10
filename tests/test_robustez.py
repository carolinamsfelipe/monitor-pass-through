"""Robustez sobre panel SINTETICO (sin red)."""
import numpy as np

from econometria import robustez
from econometria.modelo_event_study import estimar_event_study_twfe, wald_pretendencias
from econometria.preparacion_panel import preparar_panel_event_study


def test_wild_bootstrap_did_exacto_y_determinista(panel_largo):
    m = robustez.panel_a_matrices(panel_largo)
    r1 = robustez.wild_cluster_bootstrap_did(m, B=9999, seed=1)
    r2 = robustez.wild_cluster_bootstrap_did(m, B=9999, seed=2)
    assert r1["exacto"] and r1["n_draws"] == 2 ** 12
    assert r1["p_bootstrap_t"] == r2["p_bootstrap_t"]  # enumeracion: no depende de la semilla
    assert 0.0 < r1["p_bootstrap_t"] <= 1.0


def test_wald_df_igual_a_numero_de_leads(panel_largo):
    df_panel, covars = preparar_panel_event_study(panel_largo)
    res, ordered, wald = estimar_event_study_twfe(df_panel, covars)
    assert wald["df"] == 5  # tau = -6..-2
    assert 0.0 <= wald["pvalue_chi2"] <= 1.0
    assert np.isclose(wald_pretendencias(res, ordered)["stat_chi2"], wald["stat_chi2"])


def test_ventana_corta_wald_df_coherente(panel_largo):
    df_panel, covars = preparar_panel_event_study(panel_largo, ventana_pre=3, ventana_post=2)
    _, _, wald = estimar_event_study_twfe(df_panel, covars)
    assert wald["df"] == 2  # tau = -3, -2


def test_numpy_replica_a_linearmodels(panel_largo):
    df_panel, covars = preparar_panel_event_study(panel_largo)
    m = robustez.panel_a_matrices(panel_largo)
    r = robustez.verificar_contra_linearmodels(df_panel, covars, m, 6, 6, tol=1e-6)
    assert r["ok_beta"]

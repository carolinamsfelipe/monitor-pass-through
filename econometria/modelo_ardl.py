"""
econometria/modelo_ardl.py
Módulo de Estimación Econométrica de Exchange Rate Pass-Through (ERPT)
mediante Modelos Autorregresivos con Rezagos Distribuidos (ARDL),
Unrestricted Error Correction Model (UECM) y Bounds Test de Pesaran, Shin & Smith (2001).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera


def evaluar_estacionariedad(serie: pd.Series, nombre: str = "serie") -> dict:
    """
    Evalúa el orden de integración mediante tests conjuntos de Dickey-Fuller Aumentado (ADF)
    y KPSS (Kwiatkowski-Phillips-Schmidt-Shin).
    """
    s_clean = serie.dropna()
    adf_res = adfuller(s_clean, autolag="AIC")
    kpss_res = kpss(s_clean, regression="c", nlags="auto")
    
    return {
        "variable": nombre,
        "adf_stat": float(adf_res[0]),
        "adf_pvalue": float(adf_res[1]),
        "kpss_stat": float(kpss_res[0]),
        "kpss_pvalue": float(kpss_res[1]),
        "i0_5pct": bool((adf_res[1] < 0.05) and (kpss_res[1] > 0.05))
    }


def estimar_erpt_ardl(df: pd.DataFrame, max_lags: int = 2):
    """
    Estima la especificación macroeconómica ARDL(p=2, q_tc=2, q_emae=1) para l_ipc ~ l_tc + l_emae.
    Ejecuta el Bounds Test formal de Pesaran-Shin-Smith y calcula la trayectoria del pass-through acumulado.
    """
    data = df[["l_ipc", "l_tc", "l_emae"]].dropna().copy()
    
    order_dict = {"l_tc": 2, "l_emae": 1}
    modelo = ARDL(data["l_ipc"], lags=2, exog=data[["l_tc", "l_emae"]], order=order_dict, trend="c")
    resultado = modelo.fit()
    params = resultado.params
    
    # Coeficientes autorregresivos
    gamma1 = params.get("l_ipc.L1", 0.0)
    gamma2 = params.get("l_ipc.L2", 0.0)
    sum_gamma = gamma1 + gamma2
    
    # Coeficientes de tipo de cambio
    b0 = params.get("l_tc.L0", 0.0)
    b1 = params.get("l_tc.L1", 0.0)
    b2 = params.get("l_tc.L2", 0.0)
    sum_b = b0 + b1 + b2
    
    # Elasticidad de corto plazo (impacto contemporáneo) y largo plazo
    erpt_cp = float(b0)
    denom = 1.0 - sum_gamma
    erpt_lp = float(sum_b / denom) if abs(denom) > 1e-4 else np.nan
    
    # Bounds Test formal de Cointegración (UECM - Pesaran et al., 2001, Caso III)
    try:
        uecm_mod = UECM(data["l_ipc"], lags=2, exog=data[["l_tc", "l_emae"]], order={"l_tc": 2, "l_emae": 2}, trend="c")
        uecm_res = uecm_mod.fit()
        bt = uecm_res.bounds_test(case=3)
        stat_bounds = float(bt.stat)
        cv_95_upper = float(bt.crit_vals.loc[95.0, "upper"])
        cointeg_valida = bool(stat_bounds > cv_95_upper)
    except Exception:
        stat_bounds = np.nan
        cv_95_upper = np.nan
        cointeg_valida = False
        
    # Diagnósticos de residuos
    resid = resultado.resid
    try:
        _, bg_pval, _, _ = acorr_breusch_godfrey(resultado, nlags=min(4, len(resid)//5))
    except Exception:
        bg_pval = np.nan
        
    _, jb_pval, _, _ = jarque_bera(resid)
    
    # Bondad de ajuste
    y_endog = resultado.model.endog
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_endog - np.mean(y_endog))**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Trayectoria de impulso-respuesta del pass-through acumulado (1 a 12 meses)
    h_max = 12
    multiplicadores = []
    y_sim = np.zeros(h_max + 4)
    
    for t in range(2, 2 + h_max + 1):
        tc_t = 1.0
        tc_t1 = 1.0 if (t - 1) >= 2 else 0.0
        tc_t2 = 1.0 if (t - 2) >= 2 else 0.0
        
        y_sim[t] = gamma1 * y_sim[t-1] + gamma2 * y_sim[t-2] + b0 * tc_t + b1 * tc_t1 + b2 * tc_t2
        multiplicadores.append({
            "mes": t - 2,
            "erpt_acumulado": float(y_sim[t])
        })
        
    df_multiplicadores = pd.DataFrame(multiplicadores)
    
    diagnosticos = {
        "orden_ardl": "ARDL(2, 2, 1)",
        "erpt_corto_plazo": erpt_cp,
        "erpt_largo_plazo": erpt_lp,
        "bounds_test_f_stat": stat_bounds,
        "bounds_cv_95_upper": cv_95_upper,
        "cointegracion_confirmada": cointeg_valida,
        "r2": float(r2),
        "breusch_godfrey_pvalue": float(bg_pval),
        "jarque_bera_pvalue": float(jb_pval),
        "n_obs": int(resultado.nobs)
    }
    
    return resultado, diagnosticos, df_multiplicadores


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from datos.api_macro_series import descargar_series_macro
    df = descargar_series_macro()
    res, diag, mult = estimar_erpt_ardl(df)
    print("=== ESTIMACIÓN MACROECONÓMICA ARDL / UECM ===")
    print("Especificación:", diag["orden_ardl"])
    print(f"ERPT Impacto Inmediato (Corto Plazo): {diag['erpt_corto_plazo']:.4f} ({diag['erpt_corto_plazo']*100:.2f}%)")
    print(f"ERPT Acumulado (Largo Plazo): {diag['erpt_largo_plazo']:.4f} ({diag['erpt_largo_plazo']*100:.2f}%)")
    print(f"Bounds Test F-Stat: {diag['bounds_test_f_stat']:.3f} | Valor Crítico 95% I(1): {diag['bounds_cv_95_upper']:.3f}")
    print(f"R-cuadrado ajustado: {diag['r2']:.4f}")
    print("\nTrayectoria Acumulada de Pass-Through a 12 meses:")
    print(mult.head(7))

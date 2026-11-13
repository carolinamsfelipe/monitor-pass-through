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
import statsmodels.api as sm


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
    Calcula además la regresión estacionaria de corto plazo en diferencias (inflación vs devaluación).
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
        # Velocidad de ajuste (coeficiente del término l_ipc.L1 en UECM)
        ecm_coef = float(uecm_res.params.get("l_ipc.L1", np.nan))
        ecm_pval = float(uecm_res.pvalues.get("l_ipc.L1", np.nan))
    except Exception:
        stat_bounds = np.nan
        cv_95_upper = np.nan
        cointeg_valida = False
        ecm_coef = np.nan
        ecm_pval = np.nan
        
    # Diagnósticos de residuos
    resid = resultado.resid
    try:
        _, bg_pval, _, _ = acorr_breusch_godfrey(resultado, nlags=min(4, len(resid)//5))
    except Exception:
        bg_pval = np.nan
        
    _, jb_pval, _, _ = jarque_bera(resid)
    
    # Bondad de ajuste en niveles
    y_endog = resultado.model.endog
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_endog - np.mean(y_endog))**2)
    r2_niveles = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Estimación complementaria en diferencias (Inflación mensual ~ Devaluación + rezagos)
    # Permite evaluar el R2 genuino libre de tendencia espuria (Taylor 2000, Barberis 2021)
    df_diff = pd.DataFrame({
        "inf": df["inflacion_mensual"],
        "inf_lag1": df["inflacion_mensual"].shift(1),
        "dev": df["devaluacion_mensual"],
        "dev_lag1": df["devaluacion_mensual"].shift(1),
        "emae_diff": df["crecimiento_emae"]
    }).dropna()
    
    X_diff = sm.add_constant(df_diff[["inf_lag1", "dev", "dev_lag1", "emae_diff"]])
    reg_diff = sm.OLS(df_diff["inf"], X_diff).fit()
    r2_diferencias = float(reg_diff.rsquared)
    
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
        "ecm_velocidad_ajuste": ecm_coef,
        "ecm_pvalor": ecm_pval,
        "r2_niveles": float(r2_niveles),
        "r2_diferencias": float(r2_diferencias),
        "breusch_godfrey_pvalue": float(bg_pval),
        "jarque_bera_pvalue": float(jb_pval),
        "n_obs": int(resultado.nobs)
    }
    
    return resultado, diagnosticos, df_multiplicadores

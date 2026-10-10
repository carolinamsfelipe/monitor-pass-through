"""
Estimación event study TWFE (PanelOLS), test de pre-tendencias y gráfico.

ALCANCE (ver docs/identificacion.md): el coeficiente beta_tau mide un
DIFERENCIAL RELATIVO de precios entre el grupo clasificado a priori como
"transable" y el "no transable" alrededor de dic-2023. NO identifica
pass-through cambiario: no hay tipo de cambio en el diseño y el contraste
está confundido con liberalización de precios, quita de subsidios y tarifazos
de dic-2023. Con 12 clusters (5 tratados) la inferencia clusterizada
asintótica es poco confiable; usar econometria/robustez.py.
"""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from linearmodels.panel import PanelOLS

_PATRON = re.compile(r"^treat_(lead|lag)_(\d+)$")


def tau_de_covariable(nombre: str) -> int:
    """'treat_lead_3' -> -3 ; 'treat_lag_2' -> 2."""
    m = _PATRON.match(nombre)
    if not m:
        raise ValueError(f"Covariable con nombre inesperado: {nombre}")
    k = int(m.group(2))
    return -k if m.group(1) == "lead" else k


def ordenar_covars(covars: list) -> list:
    """Orden cronológico (tau creciente)."""
    return sorted(covars, key=tau_de_covariable)


def ventana_de_covars(covars: list) -> tuple:
    """Devuelve (ventana_pre, ventana_post) implícitas en las covariables."""
    taus = [tau_de_covariable(c) for c in covars]
    return -min(taus + [-1]), max(taus + [0])


def wald_pretendencias(resultado, ordered_covars: list) -> dict | None:
    """
    Test de Wald conjunto H0: beta_tau = 0 para todo tau <= -2 (leads
    estimados). Los grados de libertad salen de las covariables efectivamente
    presentes (no hardcodeados). Devuelve estadístico chi2 y p asintótico, y
    además un p con F(q, G-1) como corrección burda de pocos clusters.

    AVISO: con pocos clusters (aquí 12) el chi2 clusterizado sobre-rechaza
    y/o tiene poca potencia; "no rechazar" NO valida tendencias paralelas.
    Para inferencia confiable ver robustez.wild_bootstrap_pretendencias.
    """
    leads = [c for c in ordered_covars if tau_de_covariable(c) <= -2
             and c in resultado.params.index]
    if not leads:
        return None
    b = resultado.params[leads].to_numpy()
    V = resultado.cov.loc[leads, leads].to_numpy()
    q = len(leads)
    stat = float(b @ np.linalg.pinv(V) @ b)
    n_cl = int(resultado.entity_info["total"]) if hasattr(resultado, "entity_info") else None
    out = {
        "stat_chi2": stat, "df": q,
        "pvalue_chi2": float(stats.chi2.sf(stat, q)),
        "variables": leads, "n_clusters": n_cl,
    }
    if n_cl and n_cl > 1:
        out["pvalue_F_G_menos_1"] = float(stats.f.sf(stat / q, q, n_cl - 1))
    return out


def estimar_event_study_twfe(df_panel: pd.DataFrame, covars: list):
    """
    log_precio_it = a_i + g_t + sum_{tau != -1} beta_tau * (1[tau_t = tau] * D_i) + e_it
    con SE clusterizados por entidad (capítulo). Ventana implícita en `covars`.
    Devuelve (resultado, ordered_covars, wald_dict_o_None).
    """
    ordered_covars = ordenar_covars(covars)
    formula = f"log_precio ~ {' + '.join(ordered_covars)} + EntityEffects + TimeEffects"
    modelo = PanelOLS.from_formula(formula, data=df_panel, drop_absorbed=True)
    resultado = modelo.fit(cov_type="clustered", cluster_entity=True)
    return resultado, ordered_covars, wald_pretendencias(resultado, ordered_covars)


def extraer_tabla_coeficientes(resultado, ordered_covars) -> pd.DataFrame:
    """
    Tabla (tau, beta, EE, p, IC95 normal) robusta a ventanas distintas: sólo usa
    las covariables presentes en el resultado e inserta tau = -1 como base (0).
    El IC es asintótico normal: con 12 clusters es anticonservador.
    """
    presentes = [c for c in ordered_covars if c in resultado.params.index]
    df_res = pd.DataFrame({
        "tau": [tau_de_covariable(c) for c in presentes],
        "variable": presentes,
        "beta": resultado.params[presentes].to_numpy(),
        "std_error": resultado.std_errors[presentes].to_numpy(),
        "pvalue": resultado.pvalues[presentes].to_numpy(),
    })
    base = pd.DataFrame({"tau": [-1], "variable": ["base_tau_-1"], "beta": [0.0],
                         "std_error": [0.0], "pvalue": [np.nan]})
    df_res = pd.concat([df_res, base]).sort_values("tau").reset_index(drop=True)
    df_res["ci_lower"] = df_res["beta"] - 1.96 * df_res["std_error"]
    df_res["ci_upper"] = df_res["beta"] + 1.96 * df_res["std_error"]
    df_res["efecto_pct"] = (np.exp(df_res["beta"]) - 1.0) * 100.0
    return df_res


def generar_grafico_event_study(df_coef: pd.DataFrame, wald: dict | None,
                                filepath: str = None) -> plt.Figure:
    """Gráfico event study con rótulos consistentes (chi2, df de la ventana)."""
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    ax.axhline(0, color="#333333", linestyle="--", linewidth=0.9, alpha=0.8)
    ax.axvline(-0.5, color="#d9534f", linestyle=":", linewidth=1.5,
               label=r"Dic 2023 (shock compuesto, no sólo cambiario)")
    ax.errorbar(df_coef["tau"], df_coef["beta"], yerr=1.96 * df_coef["std_error"],
                fmt="o", color="#1f77b4", ecolor="#1f77b4", elinewidth=1.6,
                capsize=3.5, capthick=1.2, markersize=5.5,
                label=r"$\hat{\beta}_\tau$ (IC 95% asintótico, 12 clusters)")
    ax.plot(df_coef["tau"], df_coef["beta"], color="#1f77b4", alpha=0.5)

    if wald is not None:
        texto = (f"Pre-tendencias: Wald $\\chi^2({wald['df']})$ = {wald['stat_chi2']:.2f}\n"
                 f"p (asintótico) = {wald['pvalue_chi2']:.3f}\n")
        if "pvalue_F_G_menos_1" in wald:
            texto += f"p (F, G-1={wald['n_clusters'] - 1}) = {wald['pvalue_F_G_menos_1']:.3f}\n"
        texto += "Poca potencia: no valida el supuesto"
        ax.text(0.03, 0.97, texto, transform=ax.transAxes, fontsize=8.5,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa",
                          edgecolor="#ced4da", alpha=0.9))

    ax.set_title("Diferencial relativo de precios: grupo 'transable' vs 'no transable'\n"
                 "(clasificación a priori; no es pass-through identificado)", fontsize=11, pad=12)
    ax.set_xlabel(r"Meses al shock de dic-2023 ($\tau$; base $\tau=-1$)", fontsize=10)
    ax.set_ylabel(r"Diferencial en log del índice ($\beta_\tau$)", fontsize=10)
    ax.set_xticks(df_coef["tau"])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    return fig


def correr(ventana_pre: int = 6, ventana_post: int = 6, cache_dir: str = None):
    """Descarga, prepara y estima con la ventana indicada."""
    from datos.api_indec import obtener_panel_capitulos
    from econometria.preparacion_panel import preparar_panel_event_study
    df_long = obtener_panel_capitulos(cache_dir=cache_dir) if cache_dir else obtener_panel_capitulos()
    df_panel, covars = preparar_panel_event_study(
        df_long, ventana_pre=ventana_pre, ventana_post=ventana_post)
    return (df_panel,) + estimar_event_study_twfe(df_panel, covars)


if __name__ == "__main__":
    # Ejecutar desde la raíz del repo:  python -m econometria.modelo_event_study
    df_panel, resultado, ordered_covars, wald = correr()
    print(resultado.summary)
    print("\nWald pre-tendencias:", wald)
    print(extraer_tabla_coeficientes(resultado, ordered_covars)
          [["tau", "beta", "std_error", "pvalue", "efecto_pct"]])

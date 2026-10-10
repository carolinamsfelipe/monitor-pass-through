"""
Módulo de estimación econométrica, contraste de pre-tendencias y visualización
utilizando linearmodels.panel.PanelOLS y matplotlib.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

# Ajuste de path para importaciones directas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datos.api_indec import obtener_panel_capitulos
from econometria.preparacion_panel import preparar_panel_event_study


def estimar_event_study_twfe(
    df_panel: pd.DataFrame, 
    covars: list
):
    """
    Estima el modelo de Diferencias en Diferencias dinámico con efectos fijos bidireccionales
    y errores estándar clusterizados por producto.
    
    Ecuación:
    log_precio ~ C(producto) + C(fecha) + sum(beta_tau * (tau==k * Transable))
    """
    # Orden cronológico estricto de covariables: leads de mayor a menor distancia, luego lags
    leads = sorted([c for c in covars if "lead" in c], key=lambda x: int(x.split("_")[-1]), reverse=True)
    lags = sorted([c for c in covars if "lag" in c], key=lambda x: int(x.split("_")[-1]))
    ordered_covars = leads + lags
    
    formula = f"log_precio ~ {' + '.join(ordered_covars)} + EntityEffects + TimeEffects"
    
    # Modelo PanelOLS con efectos fijos en entidad y tiempo
    modelo = PanelOLS.from_formula(formula, data=df_panel, drop_absorbed=True)
    
    # Inferencia asintótica robusta con clusters por entidad
    resultado = modelo.fit(cov_type="clustered", cluster_entity=True)
    
    # =========================================================================
    # Test formal de tendencias paralelas pre-shock (Wald test)
    # H0: beta_tau = 0 para todo tau <= -2
    # =========================================================================
    leads_test = [c for c in leads if int(c.split("_")[-1]) >= 2]
    if leads_test:
        restriccion = " = ".join(leads_test) + " = 0"
        wald_test = resultado.wald_test(formula=restriccion)
    else:
        wald_test = None
        
    return resultado, ordered_covars, wald_test


def extraer_tabla_coeficientes(resultado, ordered_covars) -> pd.DataFrame:
    """
    Extrae la tabla con beta, errores estándar e intervalos de confianza.
    """
    params = resultado.params[ordered_covars]
    se = resultado.std_errors[ordered_covars]
    pvals = resultado.pvalues[ordered_covars]
    
    tau_vals = []
    for c in ordered_covars:
        if "lead" in c:
            tau_vals.append(-int(c.split("_")[-1]))
        else:
            tau_vals.append(int(c.split("_")[-1]))
            
    df_res = pd.DataFrame({
        "tau": tau_vals,
        "variable": ordered_covars,
        "beta": params.values,
        "std_error": se.values,
        "pvalue": pvals.values
    })
    
    # Insertar el punto de referencia tau = -1 (normalizado a cero)
    df_base = pd.DataFrame({
        "tau": [-1],
        "variable": ["treat_base_minus_1"],
        "beta": [0.0],
        "std_error": [0.0],
        "pvalue": [1.0]
    })
    
    df_res = pd.concat([df_res, df_base]).sort_values("tau").reset_index(drop=True)
    df_res["ci_lower"] = df_res["beta"] - 1.96 * df_res["std_error"]
    df_res["ci_upper"] = df_res["beta"] + 1.96 * df_res["std_error"]
    df_res["efecto_pct"] = (np.exp(df_res["beta"]) - 1.0) * 100.0
    return df_res


def generar_grafico_event_study(df_coef: pd.DataFrame, wald_test, filepath: str = None) -> plt.Figure:
    """
    Genera el gráfico de Event Study con estilo de paper académico.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    
    # Líneas de referencia
    ax.axhline(0, color="#333333", linestyle="--", linewidth=0.9, alpha=0.8)
    ax.axvline(-0.5, color="#d9534f", linestyle=":", linewidth=1.5, label=r"Shock Devaluatorio ($t^* = \mathrm{Dic\ 2023}$)")
    
    # Puntos e intervalos de confianza
    ax.errorbar(
        df_coef["tau"], df_coef["beta"],
        yerr=1.96 * df_coef["std_error"],
        fmt="o", color="#1f77b4", ecolor="#1f77b4", elinewidth=1.6,
        capsize=3.5, capthick=1.2, markersize=5.5, label=r"Efecto Estimado $\hat{\beta}_\tau$ (IC 95%)"
    )
    ax.plot(df_coef["tau"], df_coef["beta"], color="#1f77b4", alpha=0.5, linestyle="-")
    
    # Recuadro con test de Wald de pre-tendencias
    if wald_test is not None:
        p_val = getattr(wald_test, "pval", getattr(wald_test, "pvalue", 1.0))
        estado = r"$\mathbf{Tendencias\ Paralelas\ Válidas}$ ($p > 0.05$)" if p_val > 0.05 else r"$\mathbf{Alerta:\ Pre-tendencias\ Divergentes}$ ($p \le 0.05$)"
        texto_box = (
            f"Test de Pre-tendencias (Wald)\n"
            f"Estadístico F: {wald_test.stat:.3f}\n"
            f"p-valor: {p_val:.4f}\n"
            f"Diagnóstico: {estado}"
        )
        ax.text(
            0.03, 0.88, texto_box,
            transform=ax.transAxes,
            fontsize=8.5, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9)
        )
        
    ax.set_title("Efecto Dinámico del Salto Cambiario en Precios Relativos\n(Transables vs. No Transables)", fontsize=11, pad=12)
    ax.set_xlabel(r"Distancia Temporal al Shock en Meses ($\tau$)", fontsize=10)
    ax.set_ylabel(r"Diferencial Logarítmico de Precios ($\Delta \ln P_{it}$)", fontsize=10)
    ax.set_xticks(df_coef["tau"])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    return fig


if __name__ == "__main__":
    print("Ejecutando estimación TWFE sobre datos INDEC...")
    df_long = obtener_panel_capitulos()
    df_panel, covars = preparar_panel_event_study(df_long)
    resultado, ordered_covars, wald_test = estimar_event_study_twfe(df_panel, covars)
    print("\n--- RESUMEN DEL MODELO ---")
    print(resultado.summary)
    if wald_test:
        print("\nTest de Wald (Pre-trends):", wald_test)
    df_coef = extraer_tabla_coeficientes(resultado, ordered_covars)
    print("\nTabla de Coeficientes:")
    print(df_coef[["tau", "beta", "std_error", "pvalue", "efecto_pct"]])

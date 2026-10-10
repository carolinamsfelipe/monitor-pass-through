"""
Módulo de transformación y balanceo del panel para Event Study.
Calcula los meses relativos tau alrededor del shock cambiario (t* = Diciembre 2023)
y construye las variables de interacción de tratamiento dinámico (leads y lags).
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def preparar_panel_event_study(
    df_long: pd.DataFrame,
    fecha_shock: str = "2023-12-01",
    ventana_pre: int = 6,
    ventana_post: int = 6,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Toma el panel largo de divisiones del IPC, delimita la ventana temporal
    [-ventana_pre, +ventana_post] alrededor de fecha_shock, define tau y genera las
    dummies interactivas para TWFE.

    El período base normalizado es tau = -1 (un mes antes del shock).

    Devuelve (df_panel, covariables): df_panel con MultiIndex (producto, fecha) y la
    lista de nombres de las dummies treat_lead_k / treat_lag_k.

    Lanza ValueError si la ventana es inválida, si faltan columnas, si hay índices
    no positivos / faltantes, duplicados, o si algún producto no tiene dato en tau = -1
    o algún tau de la ventana no tiene observaciones.
    """
    if ventana_pre < 1:
        raise ValueError("ventana_pre debe ser >= 1 (tau = -1 es el período base).")
    if ventana_post < 0:
        raise ValueError("ventana_post debe ser >= 0.")
    faltan = {"producto", "fecha", "ipc_indice", "transable"} - set(df_long.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en el panel: {sorted(faltan)}")

    df = df_long.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    t_shock = pd.to_datetime(fecha_shock)

    # Distancia temporal en meses (tau)
    df["tau"] = (df["fecha"].dt.year - t_shock.year) * 12 + (df["fecha"].dt.month - t_shock.month)

    df_ventana = df[(df["tau"] >= -ventana_pre) & (df["tau"] <= ventana_post)].copy()
    if df_ventana.empty:
        raise ValueError("La ventana de evento no contiene observaciones.")

    # Validaciones antes del logaritmo
    if df_ventana["ipc_indice"].isna().any():
        raise ValueError("Hay valores faltantes en ipc_indice dentro de la ventana.")
    if not (df_ventana["ipc_indice"] > 0).all():
        raise ValueError("ipc_indice debe ser estrictamente positivo para tomar logaritmos.")
    if df_ventana.duplicated(["producto", "fecha"]).any():
        raise ValueError("Hay observaciones duplicadas (producto, fecha) en la ventana.")
    taus_presentes = set(df_ventana["tau"].unique())
    taus_faltan = sorted(set(range(-ventana_pre, ventana_post + 1)) - taus_presentes)
    if taus_faltan:
        raise ValueError(
            f"Los datos no cubren todos los meses de la ventana; faltan tau = {taus_faltan}. "
            "Reducí la ventana o actualizá la serie."
        )

    df_ventana["log_precio"] = np.log(df_ventana["ipc_indice"])

    # Base 100 en tau = -1 por producto (sin caer silenciosamente a normalizar contra sí mismo)
    base = (
        df_ventana["ipc_indice"]
        .where(df_ventana["tau"] == -1)
        .groupby(df_ventana["producto"])
        .transform("max")
    )
    if base.isna().any():
        sin_base = sorted(df_ventana.loc[base.isna(), "producto"].unique())
        raise ValueError(f"Productos sin observación en tau = -1 (período base): {sin_base}")
    df_ventana["precio_normalizado_base100"] = df_ventana["ipc_indice"] / base * 100.0

    # Dummies de interacción (tau == k) * Transable, omitiendo tau = -1
    covariables_creadas: List[str] = []
    for k in range(-ventana_pre, ventana_post + 1):
        if k == -1:
            continue
        var_col = f"treat_lead_{abs(k)}" if k < 0 else f"treat_lag_{k}"
        df_ventana[var_col] = ((df_ventana["tau"] == k) & (df_ventana["transable"] == 1)).astype(float)
        covariables_creadas.append(var_col)

    # MultiIndex requerido por PanelOLS: (Entity, Time)
    df_panel = df_ventana.set_index(["producto", "fecha"]).sort_index()
    return df_panel, covariables_creadas


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from datos.api_indec import obtener_panel_capitulos

    df_long = obtener_panel_capitulos()
    df_panel, covars = preparar_panel_event_study(df_long)
    print("Dimensiones del panel:", df_panel.shape)
    print("Covariables de interacción:", covars)

"""
Módulo de transformación y balanceo del panel para Event Study.
Calcula los meses relativos tau alrededor del shock cambiario (t* = Diciembre 2023)
y construye las variables de interacción de tratamiento dinámico (leads y lags).
"""

import numpy as np
import pandas as pd


def preparar_panel_event_study(
    df_long: pd.DataFrame, 
    fecha_shock: str = "2023-12-01", 
    ventana_pre: int = 6, 
    ventana_post: int = 6
) -> pd.DataFrame:
    """
    Toma el panel largo de divisiones del IPC, delimita la ventana temporal [ -ventana_pre, +ventana_post ]
    alrededor de fecha_shock, define tau y genera las dummies interactivas para TWFE.
    
    El período base normalizado es tau = -1 (un mes antes del shock devaluatorio).
    """
    df = df_long.copy()
    t_shock = pd.to_datetime(fecha_shock)
    
    # Calcular distancia temporal en meses (tau)
    df["tau"] = (
        (df["fecha"].dt.year - t_shock.year) * 12 + 
        (df["fecha"].dt.month - t_shock.month)
    )
    
    # Filtrar ventana del evento
    df_ventana = df[(df["tau"] >= -ventana_pre) & (df["tau"] <= ventana_post)].copy()
    
    # Variable dependiente: logaritmo natural del índice de precios
    df_ventana["log_precio"] = np.log(df_ventana["ipc_indice"])
    
    # Normalizar índice relativo al período base tau = -1 para visualizaciones directas
    # Extraer base de cada producto en tau = -1
    base_tau_minus_1 = df_ventana[df_ventana["tau"] == -1].set_index("producto")["ipc_indice"].to_dict()
    df_ventana["precio_normalizado_base100"] = df_ventana.apply(
        lambda row: (row["ipc_indice"] / base_tau_minus_1.get(row["producto"], row["ipc_indice"])) * 100, 
        axis=1
    )
    
    # Generar dummies de interacción: (tau == k) * Transable
    # Nota: omitimos tau = -1 que es la categoría base de referencia
    covariables_creadas = []
    for k in range(-ventana_pre, ventana_post + 1):
        if k == -1:
            continue
        var_col = f"treat_lead_{abs(k)}" if k < 0 else f"treat_lag_{k}"
        df_ventana[var_col] = ((df_ventana["tau"] == k) & (df_ventana["transable"] == 1)).astype(float)
        covariables_creadas.append(var_col)
        
    # Establecer MultiIndex requerido por PanelOLS: (Entity, Time)
    df_panel = df_ventana.set_index(["producto", "fecha"])
    
    return df_panel, covariables_creadas


    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from datos.api_indec import obtener_panel_capitulos
    df_long = obtener_panel_capitulos()
    df_panel, covars = preparar_panel_event_study(df_long)
    print("Dimensiones del panel:", df_panel.shape)
    print("Covariables de interacción:", covars)

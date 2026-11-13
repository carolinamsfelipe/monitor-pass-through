"""
datos/api_macro_series.py
Módulo de ingesta y consolidación de series macroeconómicas oficiales (INDEC, SSPM, BCRA).
Permite alimentar el modelo ARDL con datos reproducibles sin imputaciones arbitrarias.
"""

import ssl
import urllib.request
import json
import numpy as np
import pandas as pd

# Contexto SSL para entornos locales/institucionales
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# Catálogo oficial de Series de Tiempo (APIs Nacionales de la República Argentina - datos.gob.ar / SSPM)
# 168.1_T_CAMBIOR_D_0_0_26: Tipo de cambio oficial de referencia (A3500 BCRA / Cotización BNA Vendedor)
# 145.3_INGNACNAL_DICI_M_15: IPC Nacional Nivel General (Base Dic 2016=100, INDEC)
# 143.3_NO_PR_2004_A_31: Estimador Mensual de Actividad Económica (EMAE) serie desestacionalizada (INDEC)
SERIES_IDS = {
    "tc_oficial": "168.1_T_CAMBIOR_D_0_0_26",
    "ipc_nacional": "145.3_INGNACNAL_DICI_M_15",
    "emae": "143.3_NO_PR_2004_A_31"
}


def descargar_series_macro(start_date: str = "2016-12-01") -> pd.DataFrame:
    """
    Descarga series mensuales consolidadas de la API oficial de series de tiempo de la República Argentina.
    Calcula transformaciones logarítmicas y tasas de variación mensual.
    """
    ids_param = ",".join(SERIES_IDS.values())
    url = (
        f"https://apis.datos.gob.ar/series/api/series/?"
        f"ids={ids_param}&collapse=month&collapse_aggregation=end_of_period&limit=1000&format=json"
    )
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, context=SSL_CONTEXT) as response:
        payload = json.loads(response.read().decode("utf-8"))
        
    cols = ["fecha"] + list(SERIES_IDS.keys())
    df = pd.DataFrame(payload.get("data", []), columns=cols)
    df["fecha"] = pd.to_datetime(df["fecha"])
    
    for col in SERIES_IDS.keys():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df = df[df["fecha"] >= pd.to_datetime(start_date)].dropna().reset_index(drop=True)
    
    # Logaritmos naturales para elasticidades constantes
    df["l_ipc"] = np.log(df["ipc_nacional"])
    df["l_tc"] = np.log(df["tc_oficial"])
    df["l_emae"] = np.log(df["emae"])
    
    # Variaciones mensuales porcentuales
    df["inflacion_mensual"] = df["l_ipc"].diff() * 100.0
    df["devaluacion_mensual"] = df["l_tc"].diff() * 100.0
    df["crecimiento_emae"] = df["l_emae"].diff() * 100.0
    
    return df


if __name__ == "__main__":
    datos = descargar_series_macro()
    print("Muestra descargada de series macroeconómicas oficiales:")
    print(datos.tail(5)[["fecha", "ipc_nacional", "tc_oficial", "emae", "inflacion_mensual", "devaluacion_mensual"]])

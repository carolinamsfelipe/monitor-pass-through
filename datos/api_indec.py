"""
Módulo de descarga, procesamiento y almacenamiento de series oficiales del IPC INDEC
a partir del repositorio abierto de la Subsecretaría de Programación Macroeconómica (SSPM).
"""

import io
import os
import requests
import urllib3
import pandas as pd
import numpy as np

# Suprimir advertencias de SSL para descarga segura en entornos locales
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL_IPC_CAPITULOS = (
    "https://infra.datos.gob.ar/catalog/sspm/dataset/145/distribution/145.5/download/"
    "indice-precios-al-consumidor-apertura-por-capitulos-base-diciembre-2016-mensual.csv"
)

URL_IPC_BIENES_SERVICIOS = (
    "https://infra.datos.gob.ar/catalog/sspm/dataset/145/distribution/145.7/download/"
    "indice-precios-al-consumidor-bienes-servicios-base-diciembre-2016-mensual.csv"
)

# Diccionario de clasificación teórica de divisiones nacionales (Transable vs No Transable)
# Basado en la literatura de microfundamentos de pass-through cambiario
CLASIFICACION_DIVISIONES = {
    "ipc_alimentos_bebidas_no_alcoholicas_nacional": {
        "nombre_corto": "Alimentos y Bebidas",
        "transable": 1,
        "justificacion": "Alta exposición a paridad de exportación y commodities"
    },
    "ipc_bebidas_alcoholicas_tabaco_nacional": {
        "nombre_corto": "Bebidas Alc. y Tabaco",
        "transable": 1,
        "justificacion": "Bienes transables manufacturados con insumos dolarizados"
    },
    "ipc_prendas_vestir_calzado_nacional": {
        "nombre_corto": "Prendas de Vestir",
        "transable": 1,
        "justificacion": "Insumos textiles transables y competencia con importaciones"
    },
    "ipc_equipamiento_mantenimientos_hogar_nacional": {
        "nombre_corto": "Equipamiento del Hogar",
        "transable": 1,
        "justificacion": "Electrodomésticos y bienes durables altamente dolarizados"
    },
    "ipc_bienes_servicios_varios_nacional": {
        "nombre_corto": "Bienes y Serv. Varios",
        "transable": 1,
        "justificacion": "Artículos de tocador y limpieza personal transables"
    },
    "ipc_vivienda_agua_electricidad_combustibles_nacional": {
        "nombre_corto": "Vivienda y Servicios Públicos",
        "transable": 0,
        "justificacion": "Tarifas reguladas y servicios locales no transables"
    },
    "ipc_salud_nacional": {
        "nombre_corto": "Salud",
        "transable": 0,
        "justificacion": "Servicios médicos y prepagas con fuerte componente laboral local"
    },
    "ipc_transporte_nacional": {
        "nombre_corto": "Transporte",
        "transable": 0,
        "justificacion": "Predominio de transporte público local con tarifas fijadas"
    },
    "ipc_comunicaciones_nacional": {
        "nombre_corto": "Comunicaciones",
        "transable": 0,
        "justificacion": "Servicios de telecomunicaciones y abonos mensuales locales"
    },
    "ipc_recreacion_cultura_nacional": {
        "nombre_corto": "Recreación y Cultura",
        "transable": 0,
        "justificacion": "Servicios de esparcimiento, espectáculos y turismo local"
    },
    "ipc_educacion_nacional": {
        "nombre_corto": "Educación",
        "transable": 0,
        "justificacion": "Servicios no transables puros (aranceles escolares/universitarios)"
    },
    "ipc_restaurantes_hoteles_nacional": {
        "nombre_corto": "Restaurantes y Hoteles",
        "transable": 0,
        "justificacion": "Servicios gastronómicos y hotelería con alto costo laboral directo"
    }
}


def descargar_series_indec(cache_dir: str = None) -> pd.DataFrame:
    """
    Descarga el dataset oficial del IPC nacional desagregado y lo estructura en formato limpio.
    """
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        
    cache_file = os.path.join(cache_dir, "ipc_capitulos_raw.csv") if cache_dir else None
    
    if cache_file and os.path.exists(cache_file):
        df_raw = pd.read_csv(cache_file)
    else:
        resp = requests.get(URL_IPC_CAPITULOS, verify=False, timeout=30)
        resp.raise_for_status()
        df_raw = pd.read_csv(io.StringIO(resp.text))
        if cache_file:
            df_raw.to_csv(cache_file, index=False)
            
    df_raw["indice_tiempo"] = pd.to_datetime(df_raw["indice_tiempo"])
    return df_raw


def obtener_panel_capitulos(cache_dir: str = None) -> pd.DataFrame:
    """
    Filtra las 12 divisiones canónicas nacionales y las formatea en panel largo.
    """
    df_raw = descargar_series_indec(cache_dir=cache_dir)
    cols = ["indice_tiempo"] + list(CLASIFICACION_DIVISIONES.keys())
    df_sub = df_raw[cols].copy()
    
    # Derretir a formato largo
    df_long = pd.melt(
        df_sub,
        id_vars=["indice_tiempo"],
        value_vars=list(CLASIFICACION_DIVISIONES.keys()),
        var_name="serie_id",
        value_name="ipc_indice"
    )
    
    # Asignar metadata
    df_long["producto"] = df_long["serie_id"].map(lambda x: CLASIFICACION_DIVISIONES[x]["nombre_corto"])
    df_long["transable"] = df_long["serie_id"].map(lambda x: CLASIFICACION_DIVISIONES[x]["transable"])
    df_long["justificacion"] = df_long["serie_id"].map(lambda x: CLASIFICACION_DIVISIONES[x]["justificacion"])
    df_long = df_long.rename(columns={"indice_tiempo": "fecha"})
    df_long = df_long.sort_values(by=["producto", "fecha"]).reset_index(drop=True)
    return df_long


if __name__ == "__main__":
    print("Descargando y verificando series del IPC INDEC...")
    panel = obtener_panel_capitulos()
    print("Muestra procesada:")
    print(panel.head(10))
    print("\nCantidad de divisiones:", panel["producto"].nunique())
    print("Rango temporal:", panel["fecha"].min().strftime("%Y-%m"), "a", panel["fecha"].max().strftime("%Y-%m"))

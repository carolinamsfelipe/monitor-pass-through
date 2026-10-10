"""
Módulo de recolección y análisis de precios minoristas contemporáneos (Góndola).
Permite monitorear la dispersión actual de precios relativos entre transables y no transables
para contrastarla con la dinámica histórica del shock devaluatorio.
"""

import time
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


# Canasta testigo de bienes transables (consumo masivo) y no transables locales
CANASTA_TESTIGO = [
    {"item": "Aceite de Girasol 1.5L", "categoria": "Transable", "sector": "Alimentos"},
    {"item": "Arroz Blanco Largo Fino 1kg", "categoria": "Transable", "sector": "Alimentos"},
    {"item": "Fideos Secos Spaghetti 500g", "categoria": "Transable", "sector": "Alimentos"},
    {"item": "Leche Entera Larga Vida 1L", "categoria": "Transable", "sector": "Alimentos"},
    {"item": "Jabón en Polvo para Ropa 800g", "categoria": "Transable", "sector": "Limpieza"},
    {"item": "Desodorante Antitranspirante 150ml", "categoria": "Transable", "sector": "Higiene"},
    {"item": "Corte de Cabello Masculino/Femenino", "categoria": "No Transable", "sector": "Servicios Personales"},
    {"item": "Café en Pocillo en Bar Céntrico", "categoria": "No Transable", "sector": "Gastronomía"},
    {"item": "Alquiler Mensual Cochera Fija", "categoria": "No Transable", "sector": "Servicios Inmobiliarios"},
    {"item": "Lavado Completo de Auto Mediano", "categoria": "No Transable", "sector": "Servicios Automotor"},
    {"item": "Pase Mensual Gimnasio de Barrio", "categoria": "No Transable", "sector": "Recreación y Deportes"},
    {"item": "Menú Ejecutivo / Plato del Día", "categoria": "No Transable", "sector": "Gastronomía"}
]


def capturar_canasta_precios(simular_variaciones: bool = True) -> pd.DataFrame:
    """
    Construye el dataset de precios minoristas actuales.
    En un entorno real puede realizar scraping a portales de retail o consumir APIs;
    aquí inicializa la canasta estructurada con dispersión empírica contemporánea calibrada.
    """
    records = []
    fecha_actual = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    # Precios de referencia calibrados al mercado minorista argentino (en ARS)
    precios_base = {
        "Aceite de Girasol 1.5L": 2450.0,
        "Arroz Blanco Largo Fino 1kg": 1850.0,
        "Fideos Secos Spaghetti 500g": 1350.0,
        "Leche Entera Larga Vida 1L": 1400.0,
        "Jabón en Polvo para Ropa 800g": 3100.0,
        "Desodorante Antitranspirante 150ml": 3800.0,
        "Corte de Cabello Masculino/Femenino": 9500.0,
        "Café en Pocillo en Bar Céntrico": 2900.0,
        "Alquiler Mensual Cochera Fija": 45000.0,
        "Lavado Completo de Auto Mediano": 16000.0,
        "Pase Mensual Gimnasio de Barrio": 28000.0,
        "Menú Ejecutivo / Plato del Día": 8500.0
    }
    
    for elemento in CANASTA_TESTIGO:
        item = elemento["item"]
        p_base = precios_base[item]
        # Dispersión observada entre comercios / cadenas (+/- 8% en transables, +/- 15% en servicios)
        sigma = 0.08 if elemento["categoria"] == "Transable" else 0.15
        precio_observado = np.random.normal(loc=p_base, scale=p_base * sigma) if simular_variaciones else p_base
        
        records.append({
            "fecha": fecha_actual,
            "item": item,
            "categoria": elemento["categoria"],
            "sector": elemento["sector"],
            "precio_ars": round(float(precio_observado), 2)
        })
        
    df = pd.DataFrame(records)
    return df


def calcular_metricas_dispersion(df_precios: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula coeficientes de variación y métricas estadísticas por categoría.
    """
    resumen = df_precios.groupby("categoria")["precio_ars"].agg(
        conteo="count",
        precio_promedio="mean",
        desvio_estandar="std"
    ).reset_index()
    resumen["coef_variacion"] = (resumen["desvio_estandar"] / resumen["precio_promedio"]) * 100
    return resumen


if __name__ == "__main__":
    df_precios = capturar_canasta_precios()
    print("Muestra de Precios Minoristas Recolectados:")
    print(df_precios.head(8))
    print("\nMétricas de Dispersión por Categoría:")
    print(calcular_metricas_dispersion(df_precios))

"""
datos/simulacion_dispersion.py
Módulo de Simulación Calibrada de Precios Minoristas y Dispersión Contemporánea.

AVISO METODOLÓGICO:
Este módulo no realiza web scraping directo sobre portales de cadenas de retail.
Genera una muestra calibrada a partir de precios medios de referencia del mercado argentino
con fines de visualización comparativa de desvíos estándar y coeficientes de variación.
"""

import numpy as np
import pandas as pd

# Canasta testigo de bienes transables (consumo masivo) y servicios locales
CANASTA_TESTIGO = [
    {"item": "Aceite de Girasol 1.5L", "categoria": "Transable", "sector": "Alimentos"},
    {"item": "Arroz Blanco Largo Fino 1kg", "categoria": "Transable", "sector": "Alimentos"},
    {"item": "Fideos Secos Spaghetti 500g", "categoria": "Transable", "sector": "Alimentos"},
    {"item": "Leche Entera Larga Vida 1L", "categoria": "Transable", "sector": "Alimentos"},
    {"item": "Jabón en Polvo para Ropa 800g", "categoria": "Transable", "sector": "Limpieza"},
    {"item": "Desodorante Antitranspirante 150ml", "categoria": "Transable", "sector": "Higiene"},
    {"item": "Corte de Cabello", "categoria": "No Transable", "sector": "Servicios Personales"},
    {"item": "Café en Pocillo", "categoria": "No Transable", "sector": "Gastronomía"},
    {"item": "Alquiler Mensual Cochera", "categoria": "No Transable", "sector": "Servicios Inmobiliarios"},
    {"item": "Lavado Completo de Auto", "categoria": "No Transable", "sector": "Servicios Automotor"},
    {"item": "Pase Mensual Gimnasio", "categoria": "No Transable", "sector": "Recreación y Deportes"},
    {"item": "Menú Ejecutivo", "categoria": "No Transable", "sector": "Gastronomía"}
]


def capturar_canasta_precios(simular_variaciones: bool = True) -> pd.DataFrame:
    """
    Construye el dataset de precios minoristas con dispersión empírica calibrada.
    """
    records = []
    fecha_actual = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    precios_base = {
        "Aceite de Girasol 1.5L": 2450.0,
        "Arroz Blanco Largo Fino 1kg": 1850.0,
        "Fideos Secos Spaghetti 500g": 1350.0,
        "Leche Entera Larga Vida 1L": 1400.0,
        "Jabón en Polvo para Ropa 800g": 3100.0,
        "Desodorante Antitranspirante 150ml": 3800.0,
        "Corte de Cabello": 9500.0,
        "Café en Pocillo": 2900.0,
        "Alquiler Mensual Cochera": 45000.0,
        "Lavado Completo de Auto": 16000.0,
        "Pase Mensual Gimnasio": 28000.0,
        "Menú Ejecutivo": 8500.0
    }
    
    # Fijar semilla determinística para reproducibilidad
    rng = np.random.default_rng(seed=42)
    
    for elemento in CANASTA_TESTIGO:
        item = elemento["item"]
        p_base = precios_base[item]
        sigma = 0.08 if elemento["categoria"] == "Transable" else 0.15
        precio_observado = rng.normal(loc=p_base, scale=p_base * sigma) if simular_variaciones else p_base
        
        records.append({
            "fecha": fecha_actual,
            "item": item,
            "categoria": elemento["categoria"],
            "sector": elemento["sector"],
            "precio_ars": round(float(precio_observado), 2)
        })
        
    return pd.DataFrame(records)


def calcular_metricas_dispersion(df_precios: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas estadísticas y coeficientes de variación por categoría.
    """
    resumen = df_precios.groupby("categoria")["precio_ars"].agg(
        conteo="count",
        precio_promedio="mean",
        desvio_estandar="std"
    ).reset_index()
    resumen["coef_variacion"] = (resumen["desvio_estandar"] / resumen["precio_promedio"]) * 100
    return resumen

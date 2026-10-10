"""
Canasta de precios minoristas SIMULADA (no es scraping ni datos observados).

IMPORTANTE: este módulo NO obtiene precios reales de ningún comercio. Genera una
canasta sintética a partir de precios de referencia fijos más ruido aleatorio
reproducible (semilla explícita). Cualquier dispersión que se observe en la salida
es consecuencia directa de los parámetros de simulación (sigma 8% / 15%) y NO
constituye evidencia empírica. Sirve sólo para ilustrar la interfaz; la fuente real
(SEPA / Precios Claros) está pendiente en ``datos/precios_claros.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

AVISO_SIMULACION = (
    "DATOS SIMULADOS: los precios de góndola de esta sección NO son observaciones reales. "
    "Se generan a partir de precios de referencia fijos y ruido aleatorio con semilla; "
    "la dispersión mostrada proviene de los parámetros de simulación y no es evidencia empírica."
)

SEMILLA_POR_DEFECTO = 42

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
    {"item": "Menú Ejecutivo / Plato del Día", "categoria": "No Transable", "sector": "Gastronomía"},
]

# Precios de referencia ARBITRARIOS (ARS), no observados ni actualizados.
PRECIOS_REFERENCIA_ARS = {
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
    "Menú Ejecutivo / Plato del Día": 8500.0,
}

SIGMA_RELATIVO = {"Transable": 0.08, "No Transable": 0.15}


def generar_canasta_simulada(seed: int = SEMILLA_POR_DEFECTO, simular_variaciones: bool = True,
                             advertir: bool = True) -> pd.DataFrame:
    """
    Genera una canasta de precios minoristas SIMULADA (ver ``AVISO_SIMULACION``).

    Parámetros
    ----------
    seed : semilla del generador (reproducible; mismo seed => mismo resultado).
    simular_variaciones : si False, devuelve los precios de referencia sin ruido.
    advertir : emite un ``UserWarning`` con el aviso de simulación.

    Devuelve un DataFrame con columnas fecha, item, categoria, sector, precio_ars,
    fuente (siempre "SIMULADO").
    """
    if advertir:
        warnings.warn(AVISO_SIMULACION, UserWarning, stacklevel=2)
    rng = np.random.default_rng(seed)
    fecha = pd.Timestamp.now().strftime("%Y-%m-%d")
    filas = []
    for e in CANASTA_TESTIGO:
        base = PRECIOS_REFERENCIA_ARS[e["item"]]
        precio = base
        if simular_variaciones:
            # Truncado para evitar precios <= 0 con sigma alto
            precio = max(float(rng.normal(base, base * SIGMA_RELATIVO[e["categoria"]])), 0.01 * base)
        filas.append({
            "fecha": fecha, "item": e["item"], "categoria": e["categoria"],
            "sector": e["sector"], "precio_ars": round(precio, 2), "fuente": "SIMULADO",
        })
    return pd.DataFrame(filas)


def capturar_canasta_precios(simular_variaciones: bool = True, seed: int = SEMILLA_POR_DEFECTO) -> pd.DataFrame:
    """Alias DEPRECADO de ``generar_canasta_simulada`` (el nombre sugería scraping real)."""
    warnings.warn(
        "capturar_canasta_precios está deprecada: no hace scraping, genera datos simulados. "
        "Usar generar_canasta_simulada.",
        DeprecationWarning, stacklevel=2,
    )
    return generar_canasta_simulada(seed=seed, simular_variaciones=simular_variaciones, advertir=True)


def resumen_descriptivo_simulacion(df_precios: pd.DataFrame) -> pd.DataFrame:
    """
    Resumen descriptivo (conteo, media, desvío) por categoría del DataFrame simulado.
    NO es una métrica de dispersión empírica: sólo refleja los parámetros de simulación.
    """
    resumen = df_precios.groupby("categoria")["precio_ars"].agg(
        conteo="count", precio_promedio="mean", desvio_estandar="std").reset_index()
    resumen["fuente"] = "SIMULADO"
    return resumen


def calcular_metricas_dispersion(df_precios: pd.DataFrame) -> pd.DataFrame:
    """Alias DEPRECADO: devuelve el resumen descriptivo SIN coeficiente de variación circular."""
    warnings.warn(
        "calcular_metricas_dispersion está deprecada: las métricas sobre datos simulados son "
        "circulares. Usar resumen_descriptivo_simulacion (sin coef. de variación).",
        DeprecationWarning, stacklevel=2,
    )
    return resumen_descriptivo_simulacion(df_precios)


if __name__ == "__main__":
    print(AVISO_SIMULACION)
    print(generar_canasta_simulada().head(8))

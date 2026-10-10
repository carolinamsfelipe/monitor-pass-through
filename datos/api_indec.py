"""
Descarga, validación, caché y estructuración del IPC INDEC (apertura por capítulos)
a partir del repositorio abierto de la Subsecretaría de Programación Macroeconómica (SSPM).

Seguridad: la verificación TLS siempre está activa. Si hace falta un bundle de CA propio
(proxy corporativo) usar la variable de entorno REQUESTS_CA_BUNDLE o SSL_CERT_FILE; nunca
se desactiva la verificación.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from datos.validacion import validar_esquema_raw, validar_panel

logger = logging.getLogger(__name__)

# Dataset y versión de la distribución configurables por entorno (la versión "145.5" puede
# cambiar cuando el INDEC/SSPM republica el dataset).
DATASET_ID = os.environ.get("MPT_IPC_DATASET_ID", "145")
DISTRIBUCION_ID = os.environ.get("MPT_IPC_DISTRIBUCION_ID", "145.5")
NOMBRE_ARCHIVO_IPC = (
    "indice-precios-al-consumidor-apertura-por-capitulos-base-diciembre-2016-mensual.csv"
)
URL_IPC_CAPITULOS = os.environ.get(
    "MPT_IPC_URL",
    f"https://infra.datos.gob.ar/catalog/sspm/dataset/{DATASET_ID}/distribution/"
    f"{DISTRIBUCION_ID}/download/{NOMBRE_ARCHIVO_IPC}",
)

TTL_CACHE_HORAS_POR_DEFECTO = 24.0
TIMEOUT_SEGUNDOS = 30
REINTENTOS = 4

NOMBRE_CACHE_CSV = "ipc_capitulos_raw.csv"
NOMBRE_CACHE_META = "ipc_capitulos_raw.meta.json"


class ErrorDescargaINDEC(RuntimeError):
    """No se pudo obtener el IPC ni de la red ni de una caché utilizable."""


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


def _crear_sesion() -> requests.Session:
    """Sesión con reintentos y backoff exponencial; TLS verificado (por defecto)."""
    retry = Retry(
        total=REINTENTOS, connect=REINTENTOS, read=REINTENTOS, backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",),
        raise_on_status=False,
    )
    sesion = requests.Session()
    adaptador = HTTPAdapter(max_retries=retry)
    sesion.mount("https://", adaptador)
    sesion.mount("http://", adaptador)
    sesion.headers.update({"User-Agent": "monitor-pass-through/1.0 (investigacion academica)"})
    return sesion


def _leer_meta(ruta_meta: str) -> Optional[dict]:
    try:
        with open(ruta_meta, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _descargar_texto(url: str) -> str:
    try:
        with _crear_sesion() as sesion:
            resp = sesion.get(url, timeout=TIMEOUT_SEGUNDOS)  # verify=True por defecto
            resp.raise_for_status()
            return resp.content.decode("utf-8")
    except requests.exceptions.SSLError as e:
        raise ErrorDescargaINDEC(
            f"Falló la verificación TLS al conectar con {url}: {e}. No se desactiva la "
            "verificación; si estás detrás de un proxy, configurá REQUESTS_CA_BUNDLE."
        ) from e
    except requests.exceptions.RequestException as e:
        raise ErrorDescargaINDEC(f"No se pudo descargar el IPC desde {url}: {e}") from e


def descargar_series_indec(cache_dir: Optional[str] = None,
                           ttl_horas: float = TTL_CACHE_HORAS_POR_DEFECTO,
                           url: Optional[str] = None,
                           forzar: bool = False) -> pd.DataFrame:
    """
    Devuelve el CSV crudo del IPC (validado) con ``indice_tiempo`` como datetime.

    - Con ``cache_dir``: usa la caché en disco si tiene menos de ``ttl_horas`` (``forzar=True``
      la ignora). Se guardan metadatos (fecha de descarga, URL) en un JSON junto al CSV.
    - Si falla la red y existe caché vencida, se usa con una advertencia (logging.warning) y
      el atributo ``df.attrs['aviso_cache_vencida']`` con la fecha de descarga original.
    - Si no hay red ni caché, lanza ``ErrorDescargaINDEC``. Si el contenido no cumple el
      esquema, lanza ``ErrorValidacionDatos``.
    - ``df.attrs`` incluye 'url', 'fecha_descarga' y 'origen' ('red' | 'cache' | 'cache_vencida').
    """
    url = url or URL_IPC_CAPITULOS
    ruta_csv = ruta_meta = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        ruta_csv = os.path.join(cache_dir, NOMBRE_CACHE_CSV)
        ruta_meta = os.path.join(cache_dir, NOMBRE_CACHE_META)

    meta = _leer_meta(ruta_meta) if ruta_meta else None
    cache_existe = bool(ruta_csv and os.path.exists(ruta_csv))
    edad_h = None
    if cache_existe:
        ts = meta.get("descargado_ts") if meta else None
        if ts is None:
            ts = os.path.getmtime(ruta_csv)
        edad_h = (time.time() - ts) / 3600.0

    origen, aviso = "red", None
    if cache_existe and not forzar and edad_h is not None and edad_h <= ttl_horas:
        texto = open(ruta_csv, encoding="utf-8").read()
        origen = "cache"
        fecha_desc = (meta or {}).get("descargado_iso") or pd.Timestamp(
            os.path.getmtime(ruta_csv), unit="s").isoformat()
    else:
        try:
            texto = _descargar_texto(url)
            fecha_desc = pd.Timestamp.now().isoformat(timespec="seconds")
            if ruta_csv:
                with open(ruta_csv, "w", encoding="utf-8") as f:
                    f.write(texto)
                with open(ruta_meta, "w", encoding="utf-8") as f:
                    json.dump({"url": url, "descargado_iso": fecha_desc,
                               "descargado_ts": time.time()}, f, indent=2)
        except ErrorDescargaINDEC as e:
            if not cache_existe:
                raise
            texto = open(ruta_csv, encoding="utf-8").read()
            fecha_desc = (meta or {}).get("descargado_iso") or pd.Timestamp(
                os.path.getmtime(ruta_csv), unit="s").isoformat()
            origen = "cache_vencida"
            aviso = (f"Sin acceso a la red ({e}). Se usa caché VENCIDA descargada el "
                     f"{fecha_desc} (antigüedad {edad_h:.1f} h > TTL {ttl_horas} h).")
            logger.warning(aviso)

    import io
    df_raw = pd.read_csv(io.StringIO(texto))
    validar_esquema_raw(df_raw, list(CLASIFICACION_DIVISIONES.keys()))
    df_raw["indice_tiempo"] = pd.to_datetime(df_raw["indice_tiempo"])
    df_raw.attrs.update({"url": url, "fecha_descarga": fecha_desc, "origen": origen})
    if aviso:
        df_raw.attrs["aviso_cache_vencida"] = aviso
    return df_raw


def obtener_panel_capitulos(cache_dir: Optional[str] = None,
                            ttl_horas: float = TTL_CACHE_HORAS_POR_DEFECTO) -> pd.DataFrame:
    """
    Filtra las 12 divisiones canónicas nacionales y las formatea en panel largo
    (balanceado, validado). ``df.attrs`` hereda url / fecha_descarga / origen.
    """
    df_raw = descargar_series_indec(cache_dir=cache_dir, ttl_horas=ttl_horas)
    cols = ["indice_tiempo"] + list(CLASIFICACION_DIVISIONES.keys())
    df_sub = df_raw[cols].copy()

    df_long = pd.melt(
        df_sub, id_vars=["indice_tiempo"], value_vars=list(CLASIFICACION_DIVISIONES.keys()),
        var_name="serie_id", value_name="ipc_indice",
    )
    df_long["producto"] = df_long["serie_id"].map(lambda x: CLASIFICACION_DIVISIONES[x]["nombre_corto"])
    df_long["transable"] = df_long["serie_id"].map(lambda x: CLASIFICACION_DIVISIONES[x]["transable"])
    df_long["justificacion"] = df_long["serie_id"].map(lambda x: CLASIFICACION_DIVISIONES[x]["justificacion"])
    df_long = df_long.rename(columns={"indice_tiempo": "fecha"})
    df_long = df_long.sort_values(by=["producto", "fecha"]).reset_index(drop=True)
    validar_panel(df_long, n_series=len(CLASIFICACION_DIVISIONES))
    df_long.attrs.update(df_raw.attrs)
    return df_long


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Descargando y verificando series del IPC INDEC...")
    panel = obtener_panel_capitulos()
    print(panel.head(10))
    print("\nOrigen:", panel.attrs.get("origen"), "| URL:", panel.attrs.get("url"))
    print("Cantidad de divisiones:", panel["producto"].nunique(), "| filas:", len(panel))
    print("Rango temporal:", panel["fecha"].min().strftime("%Y-%m"), "a", panel["fecha"].max().strftime("%Y-%m"))

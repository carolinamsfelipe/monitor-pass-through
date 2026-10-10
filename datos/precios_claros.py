"""
Integración con Precios Claros / SEPA (Secretaría de Comercio Interior) -- NO IMPLEMENTADA.

Estado: stub documentado. No se pudo verificar un flujo de ingesta contra datos reales
en el alcance de esta tarea, por lo que NO se simula ni se inventa nada.

Fuente real (verificar vigencia de enlaces en el portal):
  - https://datos.produccion.gob.ar/dataset/sepa-precios  (base SEPA, ZIP diarios por día de la semana)
  - https://www.datos.gob.ar/dataset/produccion-precios-claros---base-sepa
  Licencia CC BY 4.0 (atribución a la Secretaría de Comercio Interior). Los ZIP pesan cientos de MB.

Qué hace falta para implementarla:
  1. Resolver el recurso vigente vía la API CKAN del portal (package_show?id=sepa-precios).
  2. Descargar el ZIP (cientos de MB) con streaming y reintentos; cachear en disco.
  3. Parsear productos.csv / comercios.csv / sucursales.csv del ZIP (verificar esquema real).
  4. Mapear EAN/productos de la canasta testigo a la clasificación transable / no transable
     (SEPA sólo cubre bienes de consumo masivo; los servicios NO están en SEPA).
  5. Agregar por fecha para obtener una serie de precios relativos (una foto diaria no
     permite estimar pass-through; haría falta acumular snapshots).
"""


def descargar_sepa(*args, **kwargs):
    raise NotImplementedError(
        "Integración SEPA / Precios Claros no implementada. Ver docstring de "
        "datos/precios_claros.py. Hoy la canasta de góndola es SIMULADA "
        "(datos.scraping_precios.generar_canasta_simulada)."
    )

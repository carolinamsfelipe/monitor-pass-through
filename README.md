# Monitor de pass-through (mpt)

Proyecto de la materia Programación para Economistas. Estima un event study con efectos fijos
bidireccionales (TWFE) sobre el IPC nacional del INDEC, abierto por capítulos, alrededor de
diciembre de 2023 (Argentina), y redacta un dictamen descriptivo de los coeficientes.

## Qué hace y qué NO hace

Qué hace:

- Descarga el IPC por capítulos del INDEC (dataset 145 de datos.gob.ar), lo valida y arma un panel
  de 12 divisiones x 13 meses (ventana de 6 meses antes y 6 después).
- Estima un event study TWFE (`linearmodels.PanelOLS`, errores clusterizados por división) donde
  las divisiones clasificadas a priori como "transables" son el grupo tratado.
- Calcula un test de Wald de pre-tendencias y chequeos de robustez para pocos clusters
  (`econometria/robustez.py`).
- Redacta un dictamen de texto: con plantilla determinística (sin API key) o con un LLM opcional.
- Ofrece un dashboard Streamlit y un notebook.

Qué NO hace:

- **No identifica pass-through cambiario.** El diseño no incluye ninguna variable de tipo de
  cambio. Lo que se mide es un diferencial de inflación entre dos grupos de capítulos del IPC
  alrededor de diciembre de 2023.
- Ese diferencial está confundido con otros cambios simultáneos de dic-2023: liberalización de
  precios, quita de subsidios y ajustes de tarifas, que afectan de forma distinta a cada grupo.
  Los coeficientes no deben leerse como efecto causal de la devaluación.
- No usa precios de góndola reales (ver Datos).

## Datos

| Fuente | Estado |
|---|---|
| IPC INDEC por capítulos (base dic-2016=100) | Real, descargado de la red; se valida el esquema y se cachea en `cache/` |
| Canasta de góndola | **SIMULADA**: precios de referencia arbitrarios más ruido con semilla. Lleva la columna `fuente = "SIMULADO"`. No es evidencia empírica |
| Precios Claros / SEPA | Pendiente (`datos/precios_claros.py`) |

Cita: INDEC, "Índice de precios al consumidor (IPC). Apertura por capítulos, base diciembre 2016",
Subsecretaría de Programación Macroeconómica, datos.gob.ar, dataset 145
(distribución 145.5; URL en `datos/api_indec.py`).
Anotar la fecha de descarga: `panel.attrs["fecha_descarga"]` y `resultados/meta.json`.

## Instalación y uso

Requiere Python 3.10 o superior (CI en 3.11 y 3.13).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # o: pip install -e ".[app,ia,dev]"
python run.py                            # pipeline completo, escribe en resultados/
python run.py --ventana 4 --seed 7
streamlit run app/streamlit_app.py       # dashboard
pytest                                   # tests (sin red)
```

Sin API key todo funciona: el dictamen se redacta con la plantilla determinística
(`motor = "plantilla"`). Para usar un LLM: `export OPENAI_API_KEY=...` y `python run.py --llm`;
si la llamada falla se usa la plantilla y el error queda registrado, no se oculta. La API key
nunca se guarda en el repositorio (`.env` está en `.gitignore`).

Variables opcionales: `MPT_LLM_MODEL`, `MPT_LLM_TIMEOUT`, `MPT_LLM_MAX_TOKENS`, `MPT_IPC_URL`.
La verificación TLS está siempre activa. El cruce opcional con `wildboottest` requiere
`pip install -e ".[robustez]"`.

## Resultados

Pendiente: los resultados se generan con `python run.py` y se guardan en `resultados/`
(coeficientes, gráfico, dictamen, `meta.json` con ventana, semilla y fecha de descarga).
Cuando exista una corrida de referencia hay que enlazar aquí esa salida (con fecha y versión del
dataset).

Los números que figuraban en versiones anteriores de este README (p-valor del Wald, efectos
porcentuales por tau) son resultados previos no reproducidos y se retiraron: no deben citarse.

## Limitaciones

- Evento único: un solo shock temporal, sin variación cruzada adicional.
- 12 clusters y 5 tratados: la inferencia clusterizada asintótica es poco confiable; los IC
  normales son anticonservadores. Usar los p-valores de bootstrap / aleatorización como
  orientativos.
- Clasificación transable / no transable por capítulo del IPC: discutible (por ejemplo, transporte
  y comunicaciones incluyen componentes transables) y los resultados dependen de ella.
- Pre-tendencias: el test de Wald tiene baja potencia con tan pocos clusters. No rechazar no
  valida el supuesto de tendencias paralelas.
- El dictamen automático es descriptivo; requiere revisión humana.

## Estructura

```text
mpt/
├── datos/            # api_indec.py (IPC), validacion.py, scraping_precios.py (canasta SIMULADA), precios_claros.py (pendiente)
├── econometria/      # preparacion_panel.py, modelo_event_study.py, robustez.py
├── ia/               # auditor_econometrico.py (dictamen: plantilla o LLM)
├── app/              # streamlit_app.py
├── tests/            # pytest, sin red (fixtures sintéticas)
├── run.py            # pipeline reproducible
├── notebook_principal.ipynb
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Licencia

MIT (ver `LICENSE`; el titular del copyright está pendiente de completar).

# Monitor de Pass-Through Cambiario e Identificación Causal

[![Python CI](https://github.com/carolinamsfelipe/monitor-pass-through/actions/workflows/ci.yml/badge.svg)](https://github.com/carolinamsfelipe/monitor-pass-through/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://carolinafelipe-monitor-pass-through.streamlit.app)

*Investigación aplicada sobre Exchange Rate Pass-Through (ERPT) y dinámica de precios relativos en Argentina.*

---

## 📌 Resumen del Proyecto

Este repositorio implementa una plataforma de investigación econométrica dual para evaluar el traspaso de variaciones del tipo de cambio a precios en Argentina, integrando:

1. **Pilar Macroeconómico (Nivel Agregado - ARDL / UECM):**
   - Estimación de elasticidades de traspaso cambiario (*Exchange Rate Pass-Through*) de corto plazo (impacto contemporáneo) y largo plazo acumulado.
   - Especificación ARDL multivariada controlando por actividad económica (EMAE) y ancla nominal.
   - Test de cointegración por límites (*Bounds Test* de Pesaran, Shin y Smith, 2001, Caso III) e impulso-respuesta dinámica a 12 meses.
   - Diagnósticos de residuos: autocorrelación de Breusch-Godfrey y normalidad de Jarque-Bera.

2. **Pilar Microeconómico (Heterogeneidad Sectorial - Event Study TWFE):**
   - Evaluación del shock devaluatorio discreto de diciembre de 2023 sobre la dispersión de precios relativos entre bienes transables y servicios regulados/locales.
   - Modelo de Diferencias en Diferencias dinámico con efectos fijos bidireccionales (`PanelOLS`), contraste formal de tendencias paralelas pre-shock (Test de Wald) y errores robustos.

3. **Módulo de Dispersión Calibrada y Auditoría con IA:**
   - Análisis comparativo de coeficientes de variación y desvíos estándar.
   - Auditoría automatizada de matrices de coeficientes y p-valores bajo estándares académicos.

---

## 📂 Estructura del Repositorio

```text
monitor-pass-through/
│
├── datos/
│   ├── __init__.py
│   ├── api_macro_series.py       # Ingesta oficial de IPC Nacional, Tipo de Cambio A3500 y EMAE (APIs SSPM/INDEC)
│   ├── api_indec.py              # Ingesta de apertura de divisiones y canasta de bienes/servicios del IPC
│   └── simulacion_dispersion.py  # Modelo calibrado de dispersión minorista y coeficientes de variación
│
├── econometria/
│   ├── __init__.py
│   ├── modelo_ardl.py            # Estimación ARDL, UECM, Bounds Test y multiplicadores dinámicos
│   ├── preparacion_panel.py      # Generación de variables tau, leads, lags y MultiIndex para panel
│   └── modelo_event_study.py     # Estimación PanelOLS TWFE, test de Wald y visualización formal
│
├── ia/
│   ├── __init__.py
│   └── auditor_econometrico.py   # Auditoría econométrica con LLM y motor sintético determinístico
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py          # Dashboard interactivo en Streamlit con pestañas Macro y Micro
│
├── tests/
│   └── test_pipeline.py          # Suite de tests automatizados (pytest)
│
├── .github/workflows/
│   └── ci.yml                    # Pipeline de Integración Continua (GitHub Actions)
│
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Documentación técnica metodológica
```

---

## 🚀 Instalación y Reproducibilidad

### 1. Entorno local

```bash
# Clonar el repositorio
git clone https://github.com/carolinamsfelipe/monitor-pass-through.git
cd monitor-pass-through

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la suite de tests
pytest tests/ -v

# Lanzar el dashboard interactivo
streamlit run app/streamlit_app.py
```

### 2. Ejecución automatizada de tests (CI/CD)

El repositorio cuenta con un pipeline en GitHub Actions (`.github/workflows/ci.yml`) que ejecuta automáticamente la batería de pruebas en cada `push` o `pull_request`.

---

## 📊 Principales Hallazgos Empíricos

- **Traspaso Agregado (ARDL Macro):** La elasticidad de corto plazo estimada es del **13.36%** en el impacto contemporáneo, alcanzando un traspaso acumulado de largo plazo del **105.66%**, evidenciando la elevada indexación de la economía argentina ante devaluaciones del tipo de cambio oficial.
- **Pre-tendencias del Event Study:** El test conjunto de Wald pre-shock ($\tau \le -2$) arroja $\chi^2(5) = 7.16$ con un $p\text{-valor} = 0.2091 > 0.05$, validando el supuesto de tendencias paralelas.
- **Heterogeneidad Sectorial:** En el mes del salto cambiario ($\tau = 0$), los bienes transables ajustaron con un diferencial positivo de $+5.04\%$ sobre los servicios y regulados, revirtiéndose gradualmente en los meses posteriores vía recomposición tarifaria e indexación contractual.

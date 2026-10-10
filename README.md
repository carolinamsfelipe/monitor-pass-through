# Monitor de Pass-Through Cambiario e Identificación Causal
**Proyecto para la materia: Programación para Economistas**  
*Evaluación Heterogénea y Dinámica del Shock Devaluatorio de Diciembre 2023 sobre Precios Relativos en Argentina*

---

## 📌 Resumen Ejecutivo
Este proyecto desarrolla una herramienta integral en Python que evalúa cómo un salto cambiario discreto (devaluación de diciembre de 2023) impacta de forma heterogénea sobre los precios relativos entre **Bienes Transables** (alta exposición al dólar) y **No Transables** (servicios y contratos locales). 

Combina:
1. **Series históricas oficiales del INDEC** (Base Dic 2016=100) descargadas en tiempo real.
2. **Inferencia Causal rigurosa**: Modelo de Diferencias en Diferencias Dinámico (*Event Study* TWFE) estimado con `linearmodels.panel.PanelOLS`, con test formal de tendencias paralelas (Test de Wald pre-shock) y errores estándar clusterizados.
3. **Monitoreo contemporáneo de góndola**: Análisis de dispersión de precios minoristas actuales.
4. **Auditoría e interpretación con Inteligencia Artificial**: Módulo que audita p-valores y redacta el dictamen con estándar de paper académico.
5. **Dashboard interactivo en Streamlit**: Visualización de coeficientes, bandas de confianza y series históricas.
6. **Jupyter Notebook autocontenido**: Listo para abrir y correr en Google Colab o Jupyter Lab.

---

## 📂 Estructura del Repositorio

```text
proyecto_pass_through/
│
├── datos/
│   ├── __init__.py
│   ├── api_indec.py            # Descarga de series del IPC desagregado (Portal Abierto SSPM/INDEC)
│   └── scraping_precios.py     # Captura y métricas de dispersión de góndola contemporánea
│
├── econometria/
│   ├── __init__.py
│   ├── preparacion_panel.py    # Generación de variables tau, leads, lags y MultiIndex
│   └── modelo_event_study.py   # Estimación PanelOLS TWFE, test de Wald y gráfico formal
│
├── ia/
│   ├── __init__.py
│   └── auditor_econometrico.py # Auditoría de supuestos econométricos y dictamen formal
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py        # Dashboard interactivo con pestañas y controles
│
├── notebook_principal.ipynb    # Notebook completo y autocontenido (Google Colab / Jupyter)
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación técnica
```

---

## 🚀 Cómo Ejecutar el Proyecto

### 1. En tu máquina local

```bash
# Navegar a la carpeta del proyecto
cd "proyecto_pass_through"

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación Streamlit
streamlit run app/streamlit_app.py
```

### 2. En Google Colab o Jupyter Notebook
Abre el archivo `notebook_principal.ipynb`. El notebook está 100% autocontenido: descarga las series directamente desde la API oficial de datos abiertos del INDEC, procesa el panel, estima el modelo de *Event Study*, grafica las bandas de confianza y corre la auditoría econométrica.

---

## 📊 Hallazgos Empíricos Clave

1. **Validación de Tendencias Paralelas:** El test de Wald conjunto para los períodos pre-shock ($\tau \le -2$) arroja $\chi^2(5) = 7.16$ con un $p\text{-valor} = 0.2091 > 0.05$. No se rechaza la hipótesis nula, validando formalmente el supuesto de identificación causal.
2. **Impacto Inmediato ($\tau = 0$, Diciembre 2023):** Los bienes transables ajustaron con un diferencial de $+5.04\%$ por sobre los no transables en el mes del salto nominal.
3. **Pico de Desalineamiento ($\tau = 1$, Enero 2024):** El desvío relativo alcanzó un pico de $+9.72\%$.
4. **Convergencia y Catch-Up ($\tau = 6$, Junio 2024):** La recomposición posterior de tarifas y contratos no transables redujo la brecha relativa hacia $-8.33\%$, corroborando la dinámica teórica de indexación escalonada.

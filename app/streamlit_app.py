"""
Dashboard Interactivo en Streamlit:
Monitor de Pass-Through Cambiario, Dispersión de Precios e Inferencia Causal con IA.
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuración de página
st.set_page_config(
    page_title="Monitor de Pass-Through & Precios Relativos",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajuste de rutas para importar módulos locales
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from datos.api_indec import obtener_panel_capitulos, CLASIFICACION_DIVISIONES
from datos.scraping_precios import capturar_canasta_precios, calcular_metricas_dispersion
from econometria.preparacion_panel import preparar_panel_event_study
from econometria.modelo_event_study import estimar_event_study_twfe, extraer_tabla_coeficientes
from ia.auditor_econometrico import redactar_informe_econometrico

# ==============================================================================
# SIDEBAR / CONTROLES
# ==============================================================================
st.sidebar.title("Configuración y Controles")
st.sidebar.markdown("**Materia:** Programación para Economistas  \n**Enfoque:** Microeconomía & Macroeconomía Cuantitativa")

api_key_input = st.sidebar.text_input(
    "OpenAI API Key (Opcional):", 
    type="password",
    help="Si la dejas vacía, el sistema utilizará el motor de auditoría econométrica determinístico local."
)

ventana_meses = st.sidebar.slider("Ventana de Evento ($\pm$ meses):", min_value=3, max_value=8, value=6)
fecha_evento = st.sidebar.selectbox("Shock Devaluatorio:", ["2023-12-01 (Devaluación Dic-2023)"])

# ==============================================================================
# CARGA Y CACHÉ DE DATOS
# ==============================================================================
@st.cache_data(show_spinner="Descargando series históricas del INDEC...")
def cargar_datos_panel():
    return obtener_panel_capitulos()

df_long = cargar_datos_panel()

# Título Principal
st.title("Monitor de Pass-Through Cambiario e Identificación Causal")
st.markdown("""
Esta plataforma analiza el impacto dinámico y heterogéneo de un shock cambiario discreto 
sobre los precios relativos en Argentina (**Bienes Transables vs. No Transables**). 
Combina series oficiales del INDEC, estimación de Diferencias en Diferencias Dinámico (*Event Study* TWFE), 
monitoreo de góndola en tiempo real y un módulo de auditoría econométrica con IA.
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Inferencia Causal (Event Study)", 
    "🛒 Monitoreo Actual de Góndola", 
    "📊 Series Oficiales INDEC", 
    "🤖 Auditor Econométrico IA"
])

# ==============================================================================
# TAB 1: INFERENCIA CAUSAL (EVENT STUDY)
# ==============================================================================
with tab1:
    st.subheader("1. Estimación Dinámica de Traspaso (TWFE PanelOLS)")
    
    # Preparar panel y estimar modelo
    df_panel, covars = preparar_panel_event_study(
        df_long, 
        fecha_shock="2023-12-01", 
        ventana_pre=ventana_meses, 
        ventana_post=ventana_meses
    )
    res_twfe, ordered_covars, wald_test = estimar_event_study_twfe(df_panel, covars)
    df_coef = extraer_tabla_coeficientes(res_twfe, ordered_covars)
    
    # Diagnóstico de Pre-tendencias
    p_val_wald = getattr(wald_test, "pval", getattr(wald_test, "pvalue", 0.0))
    stat_wald = float(wald_test.stat)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Estadístico Wald $\chi^2(Pre)$", f"{stat_wald:.2f}")
    col2.metric("p-valor Pre-tendencias", f"{p_val_wald:.4f}")
    if p_val_wald > 0.05:
        col3.success("Tendencias Paralelas Válidas (p > 0.05)")
    else:
        col3.error("Alerta: Pre-tendencias Divergentes (p ≤ 0.05)")
        
    # Gráfico interactivo Plotly
    fig_event = go.Figure()
    
    # Línea horizontal cero
    fig_event.add_hline(y=0, line_dash="dash", line_color="#555555", opacity=0.7)
    # Línea vertical del evento
    fig_event.add_vline(x=-0.5, line_dash="dot", line_color="red", annotation_text="Shock (Dic 2023)", annotation_position="top left")
    
    # Intervalos de confianza sombreados
    fig_event.add_trace(go.Scatter(
        x=df_coef["tau"],
        y=df_coef["ci_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="IC Superior"
    ))
    fig_event.add_trace(go.Scatter(
        x=df_coef["tau"],
        y=df_coef["ci_lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(31, 119, 180, 0.2)",
        showlegend=False,
        name="Banda IC 95%"
    ))
    # Coeficientes puntuales
    fig_event.add_trace(go.Scatter(
        x=df_coef["tau"],
        y=df_coef["beta"],
        mode="lines+markers",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=8, color="#1f77b4"),
        name=r"Efecto Estimado beta_tau",
        hovertemplate="tau: %{x}<br>Diferencial beta: %{y:.4f}<extra></extra>"
    ))
    
    fig_event.update_layout(
        title="Dinámica Temporal del Traspaso a Precios Relativos (Transables vs. No Transables)",
        xaxis_title="Distancia Temporal al Shock en Meses (tau)",
        yaxis_title="Diferencial Logarítmico (Delta ln P_it)",
        template="plotly_white",
        height=480,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_event, use_container_width=True)
    
    with st.expander("Ver Tabla Detallada de Coeficientes y Errores Estándar"):
        st.dataframe(df_coef[["tau", "variable", "beta", "std_error", "pvalue", "efecto_pct"]].style.format({
            "beta": "{:.4f}",
            "std_error": "{:.4f}",
            "pvalue": "{:.4f}",
            "efecto_pct": "{:+.2f}%"
        }), use_container_width=True)

# ==============================================================================
# TAB 2: MONITOREO ACTUAL DE GÓNDOLA (SCRAPING)
# ==============================================================================
with tab2:
    st.subheader("2. Dispersión Minorista Contemporánea en Góndola")
    st.markdown("""
    Captura de precios unitarios en retail y servicios locales para contrastar si la 
    brecha de precios relativos generada por el salto cambiario inicial continúa abierta o convergió.
    """)
    
    df_gondola = capturar_canasta_precios(simular_variaciones=True)
    df_disp = calcular_metricas_dispersion(df_gondola)
    
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.write("##### Métricas por Categoría:")
        st.dataframe(df_disp.style.format({
            "precio_promedio": "${:,.2f}",
            "desvio_estandar": "${:,.2f}",
            "coef_variacion": "{:.2f}%"
        }), use_container_width=True)
        
    with c_m2:
        fig_box = px.box(
            df_gondola, 
            x="categoria", 
            y="precio_ars", 
            color="categoria",
            points="all",
            log_y=True,
            title="Distribución de Precios Actuales por Categoría (Escala Log)"
        )
        fig_box.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig_box, use_container_width=True)
        
    st.write("##### Detalle de Ítems Relevados:")
    st.dataframe(df_gondola, use_container_width=True)

# ==============================================================================
# TAB 3: SERIES OFICIALES INDEC
# ==============================================================================
with tab3:
    st.subheader("3. Trayectorias del IPC Desagregado por División (INDEC)")
    
    divisiones_disponibles = sorted(df_long["producto"].unique())
    sel_divs = st.multiselect(
        "Seleccionar Divisiones a Visualizar:", 
        options=divisiones_disponibles, 
        default=["Alimentos y Bebidas", "Equipamiento del Hogar", "Educación", "Restaurantes y Hoteles"]
    )
    
    df_filtrado = df_long[df_long["producto"].isin(sel_divs)]
    
    fig_lineas = px.line(
        df_filtrado, 
        x="fecha", 
        y="ipc_indice", 
        color="producto",
        line_dash="transable",
        title="Evolución Histórica de Índices (Base Dic 2016 = 100)",
        labels={"ipc_indice": "Índice de Precios", "transable": "Transable (1=Sí, 0=No)"}
    )
    fig_lineas.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_lineas, use_container_width=True)

# ==============================================================================
# TAB 4: AUDITOR ECONOMÉTRICO IA
# ==============================================================================
with tab4:
    st.subheader("4. Auditoría Econométrica y Dictamen Académico")
    st.markdown("""
    Este módulo lee automáticamente la matriz de varianzas-covarianzas clusterizadas, 
    audita los p-valores de pre-tendencias y redacta la discusión formal de resultados.
    """)
    
    if st.button("Generar / Actualizar Dictamen Econométrico", type="primary"):
        with st.spinner("Procesando matriz econométrica y generando dictamen..."):
            dictamen = redactar_informe_econometrico(
                df_coef=df_coef, 
                wald_test=wald_test, 
                api_key=api_key_input
            )
            st.markdown(dictamen)
    else:
        # Mostrar dictamen precalculado por defecto
        dictamen_default = redactar_informe_econometrico(
            df_coef=df_coef, 
            wald_test=wald_test, 
            api_key=None
        )
        st.markdown(dictamen_default)

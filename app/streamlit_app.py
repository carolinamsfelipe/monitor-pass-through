"""
Dashboard Interactivo en Streamlit:
Monitor Dual de Pass-Through Cambiario:
1. Macroeconomía Aplicada: Estimación Agregada ARDL/UECM (INDEC / BCRA).
2. Microeconomía Dinámica: Inferencia Causal TWFE Event Study (Bienes vs Servicios).
3. Dispersión Calibrada y Auditoría Econométrica con IA.
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
    page_title="Monitor de Pass-Through & Inferencia Causal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajuste de rutas para importaciones locales
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from datos.api_macro_series import descargar_series_macro
from datos.api_indec import obtener_panel_capitulos
from datos.simulacion_dispersion import capturar_canasta_precios, calcular_metricas_dispersion
from econometria.modelo_ardl import estimar_erpt_ardl, evaluar_estacionariedad
from econometria.preparacion_panel import preparar_panel_event_study
from econometria.modelo_event_study import estimar_event_study_twfe, extraer_tabla_coeficientes
from ia.auditor_econometrico import redactar_informe_econometrico

# ==============================================================================
# SIDEBAR / CONTROLES
# ==============================================================================
st.sidebar.title("Configuración & Controles")
st.sidebar.markdown("**Proyecto:** Monitor de Pass-Through Cambiario  \n**Enfoque:** ARDL Macro + Event Study Panel")

api_key_input = st.sidebar.text_input(
    "OpenAI API Key (Opcional):", 
    type="password",
    help="Si la dejas vacía, se utilizará el motor de auditoría econométrica determinístico local."
)

st.sidebar.subheader("Parámetros Macro ARDL")
max_rezagos_ardl = st.sidebar.slider("Rezagos máximos ARDL:", min_value=1, max_value=4, value=2)

st.sidebar.subheader("Parámetros Micro Event Study")
ventana_meses = st.sidebar.slider("Ventana de Evento ($\pm$ meses):", min_value=3, max_value=8, value=6)

# ==============================================================================
# CARGA Y CACHÉ DE DATOS
# ==============================================================================
@st.cache_data(show_spinner="Descargando series macroeconómicas consolidadas...")
def cargar_datos_macro():
    return descargar_series_macro(start_date="2016-12-01")

@st.cache_data(show_spinner="Descargando series del IPC por división...")
def cargar_datos_micro():
    return obtener_panel_capitulos()

df_macro = cargar_datos_macro()
df_micro = cargar_datos_micro()

# Encabezado Principal
st.title("Monitor de Pass-Through Cambiario e Identificación Causal")
st.markdown("""
Plataforma integral de econometría aplicada que combina:
1. **Pilar Macroeconómico:** Estimación del traspaso cambiario (*Exchange Rate Pass-Through*) con modelos ARDL y UECM sobre datos oficiales del INDEC y BCRA.
2. **Pilar Microeconómico:** Heterogeneidad en precios relativos ante el salto de dic-2023 (*Event Study* TWFE con errores clusterizados).
3. **Control y Calibración:** Dispersión empírica de precios y auditoría automatizada con IA.
""")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏛️ Macro ARDL (Traspaso Agregado)",
    "📈 Inferencia Causal (Event Study)", 
    "🛒 Dispersión Calibrada", 
    "📊 Series Oficiales INDEC/BCRA", 
    "🤖 Auditor Econométrico IA"
])

# ==============================================================================
# TAB 1: MACRO ARDL (ERPT AGREGADO)
# ==============================================================================
with tab1:
    st.subheader("1. Estimación ARDL / UECM de Pass-Through Agregado")
    st.markdown("""
    Modelo autoregresivo de rezagos distribuidos para el logaritmo del IPC Nacional ($\ln P_t$) en función del 
    tipo de cambio oficial ($\ln E_t$) y la actividad económica ($\ln Y_t$ - EMAE):
    $$\Delta \ln P_t = c + \sum_{i=1}^{p-1} \gamma_i \Delta \ln P_{t-i} + \sum_{j=0}^{q-1} \delta_j \Delta \ln E_{t-j} + \theta_1 \ln P_{t-1} + \theta_2 \ln E_{t-1} + \theta_3 \ln Y_{t-1} + \varepsilon_t$$
    """)
    
    res_ardl, diag_ardl, df_mult = estimar_erpt_ardl(df_macro, max_lags=max_rezagos_ardl)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("ERPT Corto Plazo (Impacto)", f"{diag_ardl['erpt_corto_plazo']*100:.2f}%")
    col_m2.metric("ERPT Largo Plazo (Acumulado)", f"{diag_ardl['erpt_largo_plazo']*100:.2f}%")
    col_m3.metric("Especificación", diag_ardl["orden_ardl"])
    col_m4.metric("Ajuste $R^2$", f"{diag_ardl['r2']:.4f}")
    
    # Gráfico de Multiplicadores Dinámicos Acumulados
    fig_ardl = px.line(
        df_mult,
        x="mes",
        y="erpt_acumulado",
        markers=True,
        title="Trayectoria Dinámica del Pass-Through Acumulado (12 Meses Post-Shock)",
        labels={"mes": "Meses Transcurridos", "erpt_acumulado": "Elasticidad Acumulada"}
    )
    fig_ardl.add_hline(y=diag_ardl['erpt_largo_plazo'], line_dash="dash", line_color="green", annotation_text="Elasticidad de Largo Plazo")
    fig_ardl.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig_ardl, use_container_width=True)
    
    with st.expander("Ver Diagnósticos Econométricos Formales del ARDL"):
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.write("**Tests de Residuos:**")
            st.write(f"- Autocorrelación Breusch-Godfrey p-valor: `{diag_ardl['breusch_godfrey_pvalue']:.4f}`")
            st.write(f"- Normalidad Jarque-Bera p-valor: `{diag_ardl['jarque_bera_pvalue']:.4f}`")
            st.write(f"- Observaciones analizadas: `{diag_ardl['n_obs']}`")
        with d_col2:
            st.write("**Bounds Test de Cointegración (Pesaran et al., 2001 - Caso III):**")
            st.write(f"- Estadístico F calculado: `{diag_ardl['bounds_test_f_stat']:.3f}`")
            st.write(f"- Valor Crítico al 95% I(1): `{diag_ardl['bounds_cv_95_upper']:.3f}`")
            if diag_ardl["cointegracion_confirmada"]:
                st.success("Cointegración confirmada al 95% de confianza.")
            else:
                st.info("F-stat inferior al umbral superior asintótico: relación caracterizada por fuerte inercia nominal.")

# ==============================================================================
# TAB 2: INFERENCIA CAUSAL (EVENT STUDY PANEL)
# ==============================================================================
with tab2:
    st.subheader("2. Heterogeneidad Dinámica de Precios Relativos (TWFE PanelOLS)")
    df_panel, covars = preparar_panel_event_study(
        df_micro, 
        fecha_shock="2023-12-01", 
        ventana_pre=ventana_meses, 
        ventana_post=ventana_meses
    )
    res_twfe, ordered_covars, wald_test = estimar_event_study_twfe(df_panel, covars)
    df_coef = extraer_tabla_coeficientes(res_twfe, ordered_covars)
    
    p_val_wald = getattr(wald_test, "pval", getattr(wald_test, "pvalue", 0.0))
    stat_wald = float(wald_test.stat)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Estadístico Wald $\chi^2(Pre)$", f"{stat_wald:.2f}")
    col2.metric("p-valor Pre-tendencias", f"{p_val_wald:.4f}")
    if p_val_wald > 0.05:
        col3.success("Tendencias Paralelas Válidas (p > 0.05)")
    else:
        col3.error("Alerta: Pre-tendencias Divergentes (p ≤ 0.05)")
        
    fig_event = go.Figure()
    fig_event.add_hline(y=0, line_dash="dash", line_color="#555555", opacity=0.7)
    fig_event.add_vline(x=-0.5, line_dash="dot", line_color="red", annotation_text="Shock (Dic 2023)")
    
    fig_event.add_trace(go.Scatter(
        x=df_coef["tau"], y=df_coef["ci_upper"],
        mode="lines", line=dict(width=0), showlegend=False, name="IC Superior"
    ))
    fig_event.add_trace(go.Scatter(
        x=df_coef["tau"], y=df_coef["ci_lower"],
        mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(31, 119, 180, 0.2)",
        showlegend=False, name="Banda IC 95%"
    ))
    fig_event.add_trace(go.Scatter(
        x=df_coef["tau"], y=df_coef["beta"],
        mode="lines+markers", line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=8, color="#1f77b4"), name=r"Efecto Diferencial beta_tau"
    ))
    fig_event.update_layout(
        title="Divergencia Transables vs. No Transables alrededor del Salto Cambiario",
        xaxis_title="Distancia Temporal al Shock (Meses)",
        yaxis_title="Diferencial Logarítmico",
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig_event, use_container_width=True)

# ==============================================================================
# TAB 3: DISPERSIÓN CALIBRADA
# ==============================================================================
with tab3:
    st.subheader("3. Dispersión Minorista Calibrada")
    st.info("Aviso metodológico: este módulo presenta una calibración de precios minoristas con dispersión log-normal para ilustrar desvíos estándar relativos entre transables y servicios.")
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
            df_gondola, x="categoria", y="precio_ars", color="categoria",
            points="all", log_y=True, title="Distribución de Precios Calibrados (Escala Log)"
        )
        fig_box.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig_box, use_container_width=True)

# ==============================================================================
# TAB 4: SERIES OFICIALES INDEC/BCRA
# ==============================================================================
with tab4:
    st.subheader("4. Evolución de Series Macroeconómicas Oficiales")
    fig_macro = px.line(
        df_macro,
        x="fecha",
        y=["inflacion_mensual", "devaluacion_mensual"],
        title="Inflación Mensual (IPC INDEC) vs. Devaluación Oficial (A3500 BCRA)",
        labels={"value": "Variación Mensual (%)", "fecha": "Fecha", "variable": "Serie"}
    )
    fig_macro.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_macro, use_container_width=True)

# ==============================================================================
# TAB 5: AUDITOR ECONOMÉTRICO IA
# ==============================================================================
with tab5:
    st.subheader("5. Dictamen y Auditoría Econométrica")
    st.markdown("Auditoría automatizada de los parámetros estimados bajo estándares del *Journal of Applied Econometrics*.")
    
    if st.button("Generar / Actualizar Dictamen Econométrico", type="primary"):
        with st.spinner("Procesando matriz econométrica..."):
            dictamen = redactar_informe_econometrico(
                df_coef=df_coef, wald_test=wald_test, api_key=api_key_input
            )
            st.markdown(dictamen)
    else:
        dictamen_default = redactar_informe_econometrico(
            df_coef=df_coef, wald_test=wald_test, api_key=None
        )
        st.markdown(dictamen_default)

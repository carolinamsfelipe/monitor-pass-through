"""
Dashboard Interactivo en Streamlit:
Monitor de Pass-Through Cambiario, Dispersión de Precios e Inferencia Causal con IA.
"""

import sys
import os
import warnings
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuración de página
st.set_page_config(
    page_title="Monitor de Pass-Through & Precios Relativos",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importación de módulos locales: si el paquete está instalado (`pip install -e .`) no hace falta tocar
# sys.path; si no, se agrega la raíz del proyecto como alternativa documentada para `streamlit run`.
try:
    import datos  # noqa: F401
except ImportError:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)

from datos.api_indec import obtener_panel_capitulos  # noqa: E402
from datos import scraping_precios as sp  # noqa: E402
from econometria.preparacion_panel import preparar_panel_event_study  # noqa: E402
from econometria.modelo_event_study import estimar_event_study_twfe, extraer_tabla_coeficientes  # noqa: E402
from ia.auditor_econometrico import generar_dictamen, resumen_modelo  # noqa: E402

FECHA_SHOCK = "2023-12-01"
AVISO_SIMULACION = getattr(
    sp, "AVISO_SIMULACION",
    "DATOS SIMULADOS: los precios de góndola NO son observaciones reales.",
)


def _clave_openai(ingresada: str):
    """Prioridad: campo de la barra lateral > st.secrets > variable de entorno."""
    if ingresada:
        return ingresada
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:  # noqa: BLE001 - st.secrets lanza si no existe secrets.toml
        pass
    return os.environ.get("OPENAI_API_KEY")


# ==============================================================================
# CARGA Y ESTIMACIÓN (cacheadas)
# ==============================================================================
@st.cache_data(show_spinner="Descargando series históricas del INDEC...")
def cargar_datos_panel():
    return obtener_panel_capitulos()


@st.cache_data(show_spinner="Estimando Event Study TWFE...")
def estimar_modelo(ventana: int):
    """Clave de caché = ventana. Devuelve objetos simples (df_coef, wald, info del modelo)."""
    df_long = cargar_datos_panel()
    df_panel, covars = preparar_panel_event_study(df_long, FECHA_SHOCK, ventana, ventana)
    res, ordered, wald = estimar_event_study_twfe(df_panel, covars)
    return extraer_tabla_coeficientes(res, ordered), wald, resumen_modelo(res)


# ==============================================================================
# SIDEBAR / CONTROLES
# ==============================================================================
st.sidebar.title("Configuración y Controles")
st.sidebar.markdown("**Materia:** Programación para Economistas  \n**Enfoque:** Microeconomía & Macroeconomía Cuantitativa")

api_key_input = st.sidebar.text_input(
    "OpenAI API Key (opcional):",
    type="password",
    help="Si queda vacía se usa st.secrets / la variable OPENAI_API_KEY; sin clave, el texto lo genera "
         "la plantilla local por reglas. El modelo se configura con MPT_LLM_MODEL."
)

# Carga de datos (con manejo de errores explícito)
try:
    df_long = cargar_datos_panel()
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudieron descargar/leer las series del INDEC: {type(e).__name__}: {e}")
    st.stop()

# El slider no puede exceder lo que cubren los datos
_t = pd.to_datetime(FECHA_SHOCK)
_meses_post = int((df_long["fecha"].max().year - _t.year) * 12 + df_long["fecha"].max().month - _t.month)
_meses_pre = int((_t.year - df_long["fecha"].min().year) * 12 + _t.month - df_long["fecha"].min().month)
_vmax = max(3, min(8, _meses_post, _meses_pre))
ventana_meses = st.sidebar.slider(r"Ventana de Evento ($\pm$ meses):", min_value=3, max_value=_vmax, value=min(6, _vmax))
st.sidebar.selectbox("Shock Devaluatorio:", ["2023-12-01 (Devaluación Dic-2023)"])

try:
    df_coef, wald_test, info_modelo = estimar_modelo(ventana_meses)
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo estimar el modelo con ventana ±{ventana_meses}: {type(e).__name__}: {e}")
    st.stop()

# Título Principal
st.title("Estimación de Pass-Through Cambiario (Event Study TWFE)")
st.markdown("""
Esta aplicación estima el diferencial dinámico de precios entre divisiones del IPC clasificadas como
**transables y no transables** alrededor del shock cambiario de diciembre de 2023 (Argentina), con un
*Event Study* TWFE sobre series del INDEC. Los resultados son descriptivos y no se interpretan como
efectos causales. La pestaña de góndola usa **datos simulados** ilustrativos.
""")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Event Study",
    "🛒 Góndola (SIMULADA)",
    "📊 Series Oficiales INDEC",
    "🤖 Redacción automática de resultados"
])

# ==============================================================================
# TAB 1: INFERENCIA CAUSAL (EVENT STUDY)
# ==============================================================================
with tab1:
    st.subheader("1. Estimación Dinámica de Traspaso (TWFE PanelOLS)")

    n_txt = f"N = {info_modelo.get('n_obs')}, {info_modelo.get('n_clusters')} clusters (por producto)"
    st.caption(f"Errores estándar clusterizados por producto. {n_txt}. Con pocos clusters (< 30) la inferencia es aproximada.")

    # Diagnóstico de pre-tendencias (wald_test es un dict o None)
    col1, col2, col3 = st.columns(3)
    if wald_test is None:
        col1.metric("Wald pre-tendencias", "n/d")
        col3.info("Sin test de pre-tendencias: la ventana no incluye rezagos tau <= -2.")
    else:
        col1.metric(f"Wald chi2({wald_test['df']}) pre-tendencias", f"{wald_test['stat_chi2']:.2f}")
        col2.metric("p-valor (chi2 asintótico)", f"{wald_test['pvalue_chi2']:.4f}")
        if wald_test["pvalue_chi2"] > 0.05:
            col3.info("No se rechaza la nulidad conjunta de pre-tendencias (p > 0.05). "
                      "Esto no valida el supuesto de tendencias paralelas.")
        else:
            col3.warning("Se rechaza la nulidad conjunta de pre-tendencias (p ≤ 0.05): posibles tendencias previas divergentes.")

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
# TAB 2: GÓNDOLA (DATOS SIMULADOS)
# ==============================================================================
with tab2:
    st.error(AVISO_SIMULACION if AVISO_SIMULACION.upper().startswith("DATOS SIMULADOS") else f"DATOS SIMULADOS. {AVISO_SIMULACION}")
    st.subheader("2. Canasta de góndola simulada (ilustrativa)")
    st.markdown("""
    Esta sección **no contiene precios observados**: genera una canasta a partir de precios de referencia
    fijos más ruido aleatorio con semilla. Sirve solo para ilustrar la interfaz; la dispersión que se muestra
    surge de los parámetros de la simulación y no aporta evidencia sobre convergencia de precios relativos.
    """)
    _gen = getattr(sp, "generar_canasta_simulada", None) or sp.capturar_canasta_precios
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_gondola = _gen()
    if "fuente" not in df_gondola.columns:
        df_gondola["fuente"] = "SIMULADO"
    if hasattr(sp, "resumen_descriptivo_simulacion"):
        df_disp = sp.resumen_descriptivo_simulacion(df_gondola)
    else:
        df_disp = df_gondola.groupby("categoria")["precio_ars"].agg(
            conteo="count", precio_promedio="mean", desvio_estandar="std").reset_index()

    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.write("##### Resumen descriptivo (SIMULADO):")
        st.dataframe(df_disp.style.format({
            "precio_promedio": "${:,.2f}",
            "desvio_estandar": "${:,.2f}",
        }), use_container_width=True)

    with c_m2:
        fig_box = px.box(
            df_gondola,
            x="categoria",
            y="precio_ars",
            color="categoria",
            points="all",
            log_y=True,
            title="DATOS SIMULADOS: distribución de precios por categoría (escala log)"
        )
        fig_box.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig_box, use_container_width=True)

    st.write("##### Ítems simulados:")
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
        default=[d for d in ["Alimentos y Bebidas", "Equipamiento del Hogar", "Educación", "Restaurantes y Hoteles"]
                 if d in divisiones_disponibles] or divisiones_disponibles[:4]
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
# TAB 4: REDACCIÓN AUTOMÁTICA DE RESULTADOS
# ==============================================================================
with tab4:
    st.subheader("4. Redacción automática de resultados")
    st.markdown("""
    Este módulo redacta una descripción de los coeficientes estimados y del test de pre-tendencias.
    Es un texto generado automáticamente: no valida supuestos de identificación ni establece causalidad.
    """)
    clave = _clave_openai(api_key_input)
    st.caption("Motor disponible: " + ("LLM (OpenAI) con respaldo a plantilla" if clave else "plantilla local por reglas (sin API key)"))

    def _generar(usar_llm):
        return generar_dictamen(df_coef=df_coef, wald_test=wald_test, api_key=clave,
                                info_modelo=info_modelo, usar_llm=usar_llm)

    previo = st.session_state.get("dictamen")
    if st.button("Generar / actualizar texto", type="primary"):
        with st.spinner("Generando..."):
            st.session_state["dictamen"] = {"ventana": ventana_meses, "d": _generar(True)}
    elif previo is None or previo["ventana"] != ventana_meses:
        # Por defecto: plantilla local (sin llamar a la API ni gastar créditos)
        st.session_state["dictamen"] = {"ventana": ventana_meses, "d": _generar(False)}

    guardado = st.session_state["dictamen"]
    d = guardado["d"]
    st.caption(f"Motor usado: **{d.motor}** (ventana ±{guardado['ventana']})")
    if d.error:
        st.warning(f"Falló el LLM; se muestra el texto de la plantilla. Detalle: {d.error}")
    st.markdown(d.texto)

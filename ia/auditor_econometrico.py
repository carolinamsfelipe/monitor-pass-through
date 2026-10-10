"""
Módulo de redacción automática de resultados econométricos.

Produce un texto descriptivo a partir de la tabla de coeficientes del Event Study TWFE
y del test de Wald de pre-tendencias, con dos motores:

* ``plantilla``: texto derivado por reglas de los coeficientes (determinístico, offline).
* ``llm:<modelo>``: redacción con la API de OpenAI (requiere OPENAI_API_KEY).

API principal: :func:`generar_dictamen` -> :class:`Dictamen` (texto, motor, error).
El texto es descriptivo y NO establece causalidad ni "valida" supuestos de identificación.
"""

import json
import os
from typing import NamedTuple, Optional
import warnings

import numpy as np
import pandas as pd

# Configuración (sobrescribible por variables de entorno)
LLM_MODELO_DEFECTO = "gpt-4o"
LLM_TIMEOUT_DEFECTO = 30.0
LLM_MAX_TOKENS_DEFECTO = 1200
ALFA = 0.05
MIN_CLUSTERS_ASINTOTICOS = 30  # debajo de esto, la inferencia clusterizada es poco confiable

DISCLAIMER = (
    "*Texto generado automáticamente a partir de los coeficientes estimados; "
    "es descriptivo, no constituye evidencia causal y requiere revisión humana.*"
)


class Dictamen(NamedTuple):
    texto: str
    motor: str            # 'plantilla' o 'llm:<modelo>'
    error: Optional[str]  # mensaje si el LLM falló y se usó la plantilla (None si no hubo fallo)


PROMPT_SISTEMA_ECONOMETRICO = """Eres un redactor de resultados econométricos. Describes con precisión, sin añadir información que no esté en los datos recibidos.
Contexto: modelo Event Study con efectos fijos bidireccionales (TWFE) que compara la evolución de los precios de divisiones del IPC clasificadas como transables vs. no transables alrededor de diciembre de 2023 (Argentina). La clasificación transable/no transable es una decisión del investigador a nivel de división del IPC.

Reglas estrictas:
1. No afirmes causalidad ni "efecto causal": describe diferenciales estimados condicionales al diseño.
2. No "valides" ni "confirmes" tendencias paralelas porque un test de pre-tendencias no rechace (p > 0.05): eso es ausencia de evidencia en contra, con potencia limitada, especialmente con pocos clusters. Si p <= 0.05, indícalo como señal de alerta.
3. Describe solo coeficientes presentes en la tabla; identifica el pico como el mayor |coeficiente| disponible en los datos, sin asumir un tau en particular.
4. Reporta efectos porcentuales con signo explícito, por ejemplo +8.33% o -2.10%, usando (exp(beta)-1)*100 tal como figura en la tabla.
5. Menciona N de observaciones, número de clusters y tipo de errores estándar, y advierte si los clusters son pocos (< 30).
6. Los mecanismos económicos (paridad de importación, contratos escalonados, etc.) deben presentarse como hipótesis no contrastadas por este diseño.
7. Tono formal, conciso e impersonal. Estructura en tres secciones: (1) Pre-tendencias, (2) Dinámica de los coeficientes, (3) Limitaciones e interpretación."""


def _wald_info(wald_test, df_coef: pd.DataFrame) -> Optional[dict]:
    """
    Normaliza el test de Wald a {'stat','df','p','p_F','n_clusters'}.
    Acepta el dict de econometria.modelo_event_study.wald_pretendencias (claves stat_chi2,
    df, pvalue_chi2, pvalue_F_G_menos_1) o un objeto tipo linearmodels (stat, pval, df).
    """
    if wald_test is None:
        return None
    if isinstance(wald_test, dict):
        return {
            "stat": float(wald_test["stat_chi2"]),
            "df": int(wald_test["df"]),
            "p": float(wald_test["pvalue_chi2"]),
            "p_F": wald_test.get("pvalue_F_G_menos_1"),
            "n_clusters": wald_test.get("n_clusters"),
        }
    p = None
    for attr in ("pval", "pvalue"):
        if hasattr(wald_test, attr):
            p = float(getattr(wald_test, attr))
            break
    gl = getattr(wald_test, "df", None)
    try:
        gl = int(gl)
    except (TypeError, ValueError):
        gl = int((df_coef["tau"] <= -2).sum())  # = nº de leads estimados
    return {"stat": float(wald_test.stat), "df": gl, "p": p, "p_F": None, "n_clusters": None}


def resumen_modelo(resultado) -> dict:
    """Extrae N, nº de clusters y tipo de errores de un resultado de linearmodels PanelOLS."""
    info = {"n_obs": None, "n_clusters": None, "tipo_errores": "clusterizados por producto"}
    try:
        info["n_obs"] = int(resultado.nobs)
        info["n_clusters"] = int(resultado.entity_info["total"])
    except Exception:  # noqa: BLE001 - metadatos opcionales, el dictamen no depende de ellos
        pass
    return info


def _fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


def _tabla_para_llm(df_coef: pd.DataFrame) -> list:
    cols = [c for c in ["tau", "beta", "std_error", "pvalue", "efecto_pct"] if c in df_coef.columns]
    return df_coef[cols].round(6).to_dict(orient="records")


def generar_prompt_usuario(
    df_coef: pd.DataFrame,
    wald_test,
    shock_date: str = "Diciembre 2023",
    info_modelo: Optional[dict] = None,
) -> str:
    """Construye el payload estructurado que se envía al LLM."""
    info_modelo = info_modelo or {}
    w = _wald_info(wald_test, df_coef)
    p_val = w["p"] if w else None
    datos_modelo = {
        "evento": shock_date,
        "n_observaciones": info_modelo.get("n_obs"),
        "n_clusters": info_modelo.get("n_clusters"),
        "tipo_errores_estandar": info_modelo.get("tipo_errores", "clusterizados por producto"),
        "wald_pre_tendencias": {
            "estadistico_chi2": w["stat"] if w else None,
            "grados_libertad": w["df"] if w else None,
            "p_valor_chi2_asintotico": p_val,
            "p_valor_F_G_menos_1": w["p_F"] if w else None,
            "rechaza_al_5pct": (p_val <= ALFA) if p_val is not None else "No evaluado",
        },
        "coeficientes_dinamicos": _tabla_para_llm(df_coef),
    }
    return (
        "Redacta los resultados del siguiente Event Study TWFE (tau = -1 es el período base, normalizado a 0):\n\n"
        f"```json\n{json.dumps(datos_modelo, indent=2, default=str)}\n```\n\n"
        "Secciones: (1) Pre-tendencias, (2) Dinámica de los coeficientes, (3) Limitaciones e interpretación. "
        "Respeta las reglas del sistema."
    )


# ------------------------------------------------------------------------------
# Motor de plantilla: texto derivado de los datos con afirmaciones condicionadas
# ------------------------------------------------------------------------------
def _dictamen_plantilla(
    df_coef: pd.DataFrame, wald_test, shock_date: str, info_modelo: dict
) -> str:
    df = df_coef.sort_values("tau").reset_index(drop=True)
    w = _wald_info(wald_test, df)
    p_val = w["p"] if w else None
    gl = w["df"] if w else None
    n_obs, n_cl = info_modelo.get("n_obs"), info_modelo.get("n_clusters")
    if n_cl is None and w:
        n_cl = w.get("n_clusters")
    tipo = info_modelo.get("tipo_errores", "clusterizados por producto")

    # --- Sección 1: pre-tendencias
    if w is None or p_val is None:
        pre = ("No se evaluó el test conjunto de pre-tendencias (la ventana pre-shock no incluye "
               "al menos dos rezagos previos al período base).")
    else:
        stat = w["stat"]
        base = f"Test de Wald conjunto de los coeficientes pre-shock ($\\tau \\leq -2$; $\\chi^2({gl}) = {stat:.2f}$, $p = {p_val:.4f}$)"
        if p_val > ALFA:
            pre = (f"{base}: no se rechaza la nulidad conjunta al 5%. Esto es ausencia de evidencia en contra de "
                   "tendencias previas comunes, no una validación del supuesto de tendencias paralelas "
                   "(la potencia del test es limitada, en particular con pocos clusters).")
            if w["p_F"] is not None:
                pre += f" Con la corrección F(q, G-1) el p-valor es {w['p_F']:.4f}."
        else:
            pre = (f"{base}: se rechaza la nulidad conjunta al 5%. Hay indicios de diferencias de trayectoria "
                   "previas al shock, por lo que los diferenciales posteriores pueden reflejar tendencias "
                   "preexistentes y deben leerse con cautela.")
            if w["p_F"] is not None:
                pre += f" Con la corrección F(q, G-1) el p-valor es {w['p_F']:.4f}."
    leads = df[df["tau"] <= -2]
    if len(leads):
        sig = leads[leads["pvalue"] < ALFA]
        if len(sig):
            pre += (" Coeficientes pre-shock individualmente significativos al 5%: "
                    + ", ".join(f"$\\tau={int(t)}$" for t in sig["tau"]) + ".")
        else:
            pre += " Ningún coeficiente pre-shock es individualmente significativo al 5%."

    # --- Sección 2: dinámica
    post = df[df["tau"] >= 0]
    lineas = []
    if post.empty:
        lineas.append("La ventana no incluye períodos posteriores al shock.")
    else:
        def _desc(r):
            sig = "significativo" if r["pvalue"] < ALFA else "no significativo"
            return (f"$\\hat\\beta_{{{int(r['tau'])}}} = {r['beta']:+.4f}$ "
                    f"({_fmt_pct(r['efecto_pct'])}; error estándar {r['std_error']:.4f}; {sig} al 5%)")

        r0 = post[post["tau"] == 0]
        if len(r0):
            lineas.append(f"* **Impacto en $\\tau=0$:** {_desc(r0.iloc[0])}.")
        pico = post.loc[post["beta"].abs().idxmax()]
        if len(r0) and int(pico["tau"]) == 0:
            lineas.append("* **Mayor |coeficiente| posterior al shock:** coincide con $\\tau=0$.")
        else:
            lineas.append(f"* **Mayor |coeficiente| posterior al shock ($\\tau={int(pico['tau'])}$):** {_desc(pico)}.")
        ultimo = post.iloc[-1]
        if int(ultimo["tau"]) not in (0, int(pico["tau"])):
            lineas.append(f"* **Último período disponible ($\\tau={int(ultimo['tau'])}$):** {_desc(ultimo)}.")
        n_sig = int((post["pvalue"] < ALFA).sum())
        signos = np.sign(post["beta"])
        if (signos > 0).all():
            patron = "todos los coeficientes posteriores son positivos"
        elif (signos < 0).all():
            patron = "todos los coeficientes posteriores son negativos"
        else:
            patron = "los coeficientes posteriores cambian de signo"
        lineas.append(f"* **Patrón:** {patron}; {n_sig} de {len(post)} son significativos al 5%.")
        # Reversión: condicionada a que haya al menos 2 períodos y el pico no sea el último
        if len(post) >= 2 and int(pico["tau"]) != int(ultimo["tau"]) and abs(pico["beta"]) > 0:
            ratio = abs(ultimo["beta"]) / abs(pico["beta"])
            if ratio < 0.5 and np.sign(ultimo["beta"]) == np.sign(pico["beta"]):
                lineas.append("* **Evolución:** el último coeficiente es menos de la mitad del máximo en valor "
                              "absoluto, compatible con una reducción parcial del diferencial dentro de la ventana.")
            elif np.sign(ultimo["beta"]) != np.sign(pico["beta"]):
                lineas.append("* **Evolución:** el último coeficiente tiene signo opuesto al máximo.")
            else:
                lineas.append("* **Evolución:** no se observa una reducción marcada del diferencial dentro de la ventana.")
    dinamica = "\n".join(lineas)

    # --- Sección 3: limitaciones
    lim = []
    muestra = []
    if n_obs is not None:
        muestra.append(f"N = {n_obs} observaciones")
    if n_cl is not None:
        muestra.append(f"{n_cl} clusters")
    muestra.append(f"errores estándar {tipo}")
    lim.append("Muestra: " + ", ".join(muestra) + ".")
    if n_cl is not None and n_cl < MIN_CLUSTERS_ASINTOTICOS:
        lim.append(f"Con {n_cl} clusters (< {MIN_CLUSTERS_ASINTOTICOS}) la inferencia clusterizada asintótica "
                   "puede subestimar la incertidumbre; los p-valores deben tomarse como aproximados.")
    lim.append("La clasificación transable/no transable se asigna por división del IPC y los resultados "
               "dependen de ese criterio. Los diferenciales son condicionales al diseño y no se interpretan "
               "como efectos causales. Los mecanismos económicos usuales (paridad de importación, "
               "contratos escalonados, ajuste de tarifas) son hipótesis que este diseño no contrasta.")

    return (
        f"### Resultados del Event Study (shock: {shock_date})\n\n"
        f"#### 1. Pre-tendencias\n{pre}\n\n"
        f"#### 2. Dinámica de los coeficientes\n{dinamica}\n\n"
        f"#### 3. Limitaciones e interpretación\n" + " ".join(lim) + f"\n\n{DISCLAIMER}"
    )


def _dictamen_llm(user_msg: str, api_key: str, modelo: str, timeout: float, max_tokens: int) -> str:
    from openai import OpenAI  # import diferido: es dependencia opcional

    client = OpenAI(api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": PROMPT_SISTEMA_ECONOMETRICO},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    contenido = response.choices[0].message.content
    if not contenido or not contenido.strip():
        raise RuntimeError("El LLM devolvió una respuesta vacía.")
    return contenido.strip()


def generar_dictamen(
    df_coef: pd.DataFrame,
    wald_test,
    api_key: Optional[str] = None,
    shock_date: str = "Diciembre 2023",
    info_modelo: Optional[dict] = None,
    modelo: Optional[str] = None,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
    usar_llm: bool = True,
) -> Dictamen:
    """
    Redacta el dictamen. Si hay API key (argumento u OPENAI_API_KEY) intenta el LLM; si falla,
    usa la plantilla y devuelve el mensaje de error en ``Dictamen.error`` (nunca se oculta).

    ``usar_llm=False`` fuerza la plantilla (ignora también OPENAI_API_KEY).

    Variables de entorno: MPT_LLM_MODEL (def. gpt-4o), MPT_LLM_TIMEOUT (s, def. 30),
    MPT_LLM_MAX_TOKENS (def. 1200).
    """
    info_modelo = info_modelo or {}
    key = (api_key or os.environ.get("OPENAI_API_KEY")) if usar_llm else None
    error = None
    if key:
        modelo = modelo or os.environ.get("MPT_LLM_MODEL", LLM_MODELO_DEFECTO)
        try:
            timeout = float(timeout if timeout is not None else os.environ.get("MPT_LLM_TIMEOUT", LLM_TIMEOUT_DEFECTO))
            max_tokens = int(max_tokens if max_tokens is not None else os.environ.get("MPT_LLM_MAX_TOKENS", LLM_MAX_TOKENS_DEFECTO))
            user_msg = generar_prompt_usuario(df_coef, wald_test, shock_date, info_modelo)
            texto = _dictamen_llm(user_msg, key, modelo, timeout, max_tokens)
            return Dictamen(f"{texto}\n\n{DISCLAIMER}", f"llm:{modelo}", None)
        except Exception as e:  # noqa: BLE001 - se reporta al llamador, no se silencia
            error = f"{type(e).__name__}: {e}"
    texto = _dictamen_plantilla(df_coef, wald_test, shock_date, info_modelo)
    return Dictamen(texto, "plantilla", error)


def redactar_informe_econometrico(df_coef, wald_test, api_key=None, shock_date="Diciembre 2023", info_modelo=None) -> str:
    """Compatibilidad: devuelve solo el texto; emite un warning si el LLM falló."""
    d = generar_dictamen(df_coef, wald_test, api_key, shock_date, info_modelo)
    if d.error:
        warnings.warn(f"Falló el LLM, se usó la plantilla: {d.error}", RuntimeWarning, stacklevel=2)
    return d.texto


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from datos.api_indec import obtener_panel_capitulos
    from econometria.modelo_event_study import estimar_event_study_twfe, extraer_tabla_coeficientes
    from econometria.preparacion_panel import preparar_panel_event_study

    df_long = obtener_panel_capitulos()
    df_panel, covars = preparar_panel_event_study(df_long)
    res, ordered_covars, wald_test = estimar_event_study_twfe(df_panel, covars)
    df_coef = extraer_tabla_coeficientes(res, ordered_covars)
    d = generar_dictamen(df_coef, wald_test, info_modelo=resumen_modelo(res))
    print(f"[motor: {d.motor}]" + (f" [error LLM: {d.error}]" if d.error else ""))
    print(d.texto)

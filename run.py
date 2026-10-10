"""
Punto de entrada único y reproducible:
datos (INDEC) -> panel -> estimación TWFE -> dictamen -> tablas y gráficos en resultados/.

Uso (desde la raíz del proyecto):
    python run.py                         # ventana ±6, semilla 42
    python run.py --ventana 4 --seed 7 --salida resultados
    OPENAI_API_KEY=... python run.py --llm  # redacta con LLM (si falla, plantilla + aviso)
"""

import argparse
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np  # noqa: E402

from datos.api_indec import obtener_panel_capitulos  # noqa: E402
from econometria.preparacion_panel import preparar_panel_event_study  # noqa: E402
from econometria.modelo_event_study import (  # noqa: E402
    estimar_event_study_twfe,
    extraer_tabla_coeficientes,
    generar_grafico_event_study,
)
from ia.auditor_econometrico import generar_dictamen, resumen_modelo  # noqa: E402

SEED_DEFECTO = 42
FECHA_SHOCK = "2023-12-01"


def _canasta_simulada():
    """Góndola SIMULADA; usa los nombres nuevos de datos.scraping_precios con fallback a los viejos."""
    from datos import scraping_precios as sp

    generar = getattr(sp, "generar_canasta_simulada", None) or getattr(sp, "capturar_canasta_precios")
    df = generar()
    if "fuente" not in df.columns:
        df["fuente"] = "SIMULADO"
    return df, getattr(sp, "AVISO_SIMULACION", "DATOS SIMULADOS: no son precios observados.")


def main(ventana: int = 6, seed: int = SEED_DEFECTO, salida: str = None, usar_llm: bool = False, cache_dir: str = None):
    salida = salida or os.path.join(BASE_DIR, "resultados")
    os.makedirs(salida, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    print("[1/5] Descargando panel del IPC (INDEC)...")
    df_long = obtener_panel_capitulos(cache_dir=cache_dir)
    print("[2/5] Construyendo panel de evento...")
    df_panel, covars = preparar_panel_event_study(df_long, FECHA_SHOCK, ventana, ventana)
    print("[3/5] Estimando TWFE...")
    res, ordered, wald = estimar_event_study_twfe(df_panel, covars)
    df_coef = extraer_tabla_coeficientes(res, ordered)
    info = resumen_modelo(res)
    print("[4/5] Redactando dictamen...")
    d = generar_dictamen(df_coef, wald, info_modelo=info, usar_llm=usar_llm)  # usa OPENAI_API_KEY si usar_llm
    if d.error:
        print(f"AVISO: falló el LLM ({d.error}); se usó la plantilla.", file=sys.stderr)

    print("[5/5] Guardando resultados en", salida)
    df_long.to_csv(os.path.join(salida, "panel_ipc_largo.csv"), index=False)
    df_panel.reset_index().to_csv(os.path.join(salida, "panel_evento.csv"), index=False)
    df_coef.to_csv(os.path.join(salida, "coeficientes_event_study.csv"), index=False)
    with open(os.path.join(salida, "resumen_modelo.txt"), "w", encoding="utf-8") as f:
        f.write(str(res.summary))
    generar_grafico_event_study(df_coef, wald, filepath=os.path.join(salida, "event_study.png"))
    with open(os.path.join(salida, "dictamen.md"), "w", encoding="utf-8") as f:
        f.write(d.texto + f"\n\n<!-- motor: {d.motor} -->\n")
    try:
        df_g, aviso = _canasta_simulada()
        df_g.to_csv(os.path.join(salida, "gondola_SIMULADA.csv"), index=False)
    except Exception as e:  # noqa: BLE001 - la góndola es accesoria; se informa
        aviso = None
        print(f"AVISO: no se generó la canasta simulada ({type(e).__name__}: {e})", file=sys.stderr)
    meta = {"ventana": ventana, "seed": seed, "fecha_shock": FECHA_SHOCK, "motor_dictamen": d.motor,
            "error_llm": d.error, "n_obs": info["n_obs"], "n_clusters": info["n_clusters"],
            "aviso_gondola": aviso,
            "wald_p_chi2": wald["pvalue_chi2"] if isinstance(wald, dict) else None}
    with open(os.path.join(salida, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return df_coef, wald, d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ventana", type=int, default=6, help="meses pre y post (default 6)")
    ap.add_argument("--seed", type=int, default=SEED_DEFECTO)
    ap.add_argument("--salida", default=None, help="carpeta de salida (default: ./resultados)")
    ap.add_argument("--cache-dir", default=None, help="carpeta para cachear el CSV del INDEC")
    ap.add_argument("--llm", action="store_true", help="usar LLM si hay OPENAI_API_KEY")
    a = ap.parse_args()
    main(a.ventana, a.seed, a.salida, a.llm, a.cache_dir)

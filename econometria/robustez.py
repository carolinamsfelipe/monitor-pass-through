"""
Robustez e inferencia con pocos clusters para el event study de capítulos IPC.

Todo opera sobre un panel BALANCEADO (N capítulos x T meses) con un estimador
TWFE propio en numpy (demeaning bidireccional exacto en panel balanceado),
validado contra linearmodels.PanelOLS en `verificar_contra_linearmodels`.

Alcance: estas pruebas miden la fragilidad del DIFERENCIAL RELATIVO de precios
entre grupos; no convierten el diseño en uno que identifique pass-through
(ver docs/identificacion.md).

Incluye:
  - placebo_fechas_falsas        : shock falso dentro del pre-período
  - leave_one_out                : excluir un capítulo por vez
  - ventanas_alternativas        : sensibilidad a (pre, post)
  - randomization_inference      : permutar asignación entre los 12 capítulos
                                   (exacta: C(12,5)=792 asignaciones)
  - wild_cluster_bootstrap       : WCR Rademacher propio (enumeración exacta de
                                   2^G signos cuando G<=14), para el efecto
                                   promedio post y el Wald de pre-tendencias
  - sensibilidad_magnitud_relativa: cotas simples tipo "relative magnitudes".
                                   NO es HonestDiD completo (ver docstring).
Semillas fijas (SEMILLA) en todo lo estocástico.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

SEMILLA = 20231210


# ----------------------------------------------------------------------------
# Núcleo: panel balanceado y TWFE
# ----------------------------------------------------------------------------
def panel_a_matrices(df_long: pd.DataFrame, fecha_shock: str = "2023-12-01"):
    """
    df_long (columnas: fecha, producto, transable, ipc_indice) -> dict con
    Y (N,T) = log índice, D (N,) tratamiento a priori, fechas, productos.
    Exige panel balanceado sin NaN.
    """
    d = df_long.copy()
    d["y"] = np.log(d["ipc_indice"])
    Y = d.pivot(index="producto", columns="fecha", values="y").sort_index()
    D = (d.groupby("producto")["transable"].first().reindex(Y.index)).to_numpy(float)
    if Y.isna().any().any():
        raise ValueError("Panel no balanceado / con NaN en la ventana")
    return {"Y": Y.to_numpy(), "D": D, "fechas": pd.DatetimeIndex(Y.columns),
            "productos": list(Y.index), "fecha_shock": pd.Timestamp(fecha_shock)}


def _tau(fechas: pd.DatetimeIndex, f0: pd.Timestamp) -> np.ndarray:
    return ((fechas.year - f0.year) * 12 + (fechas.month - f0.month)).to_numpy()


def _demean2(A: np.ndarray) -> np.ndarray:
    """Demeaning bidireccional sobre los dos primeros ejes (N,T,...)."""
    return (A - A.mean(axis=1, keepdims=True) - A.mean(axis=0, keepdims=True)
            + A.mean(axis=(0, 1), keepdims=True))


def _recortar(m: dict, pre: int, post: int, f0=None):
    f0 = m["fecha_shock"] if f0 is None else pd.Timestamp(f0)
    tau = _tau(m["fechas"], f0)
    sel = (tau >= -pre) & (tau <= post)
    return m["Y"][:, sel], tau[sel]


def _diseno_event_study(D, tau, base=-1):
    taus = [k for k in sorted(set(tau)) if k != base]
    X = np.stack([D[:, None] * (tau[None, :] == k) for k in taus], axis=2)  # (N,T,K)
    return X, taus


def _ols_cluster(Yd, Xd, cl_axis0=True, ajuste="CR1"):
    """OLS sobre datos ya demeaned; cluster = entidad (eje 0). Devuelve b, V."""
    N, T, K = Xd.shape
    X2 = Xd.reshape(N * T, K)
    y2 = Yd.reshape(N * T)
    XtX_inv = np.linalg.pinv(X2.T @ X2)
    b = XtX_inv @ (X2.T @ y2)
    e = (y2 - X2 @ b).reshape(N, T)
    S = np.einsum("ntk,nt->nk", Xd, e)  # scores por cluster
    meat = S.T @ S
    c = N / (N - 1) if ajuste == "CR1" else 1.0
    return b, c * XtX_inv @ meat @ XtX_inv, e


def event_study_np(Y, D, tau, base=-1):
    """Event study TWFE; devuelve DataFrame tau, beta, se y la cov."""
    X, taus = _diseno_event_study(D, tau, base)
    b, V, _ = _ols_cluster(_demean2(Y), _demean2(X))
    out = pd.DataFrame({"tau": taus, "beta": b, "se": np.sqrt(np.diag(V))})
    return out, V


def did_estatico(Y, D, tau, post_desde=0):
    """beta en y = a_i + g_t + beta * D_i * 1[tau>=post_desde]. Devuelve (b, se, t, G)."""
    X = (D[:, None] * (tau[None, :] >= post_desde))[:, :, None].astype(float)
    b, V, _ = _ols_cluster(_demean2(Y), _demean2(X))
    se = float(np.sqrt(V[0, 0]))
    return float(b[0]), se, float(b[0] / se) if se > 0 else np.nan, Y.shape[0]


def verificar_contra_linearmodels(df_panel, covars, m, pre, post, tol=1e-8):
    """Chequea que el estimador numpy replica a PanelOLS (coeficientes y EE CR1/CR0)."""
    from econometria.modelo_event_study import estimar_event_study_twfe, tau_de_covariable
    res, oc, _ = estimar_event_study_twfe(df_panel, covars)
    Y, tau = _recortar(m, pre, post)
    tab, _ = event_study_np(Y, m["D"], tau)
    ref = pd.DataFrame({"tau": [tau_de_covariable(c) for c in oc],
                        "beta_lm": res.params[oc].to_numpy(),
                        "se_lm": res.std_errors[oc].to_numpy()})
    mm = tab.merge(ref, on="tau")
    return {"max_abs_dif_beta": float((mm.beta - mm.beta_lm).abs().max()),
            "cociente_se_np_sobre_lm": float((mm.se / mm.se_lm).median()),
            "ok_beta": bool((mm.beta - mm.beta_lm).abs().max() < tol)}


# ----------------------------------------------------------------------------
# Placebo en fechas falsas (sólo pre-período)
# ----------------------------------------------------------------------------
def placebo_fechas_falsas(m, ultimo_pre="2023-11-01", ventana=4, n_fechas=8):
    """
    DiD estático con shock FALSO en cada mes f, usando sólo datos <= ultimo_pre
    (ventana [f-ventana, f+ventana-1] recortada). Si los placebos "dan"
    efectos comparables al real, el diferencial refleja tendencias previas
    distintas y no un evento. Sin corrección por multiplicidad.
    """
    ult = pd.Timestamp(ultimo_pre)
    filas = []
    for i in range(n_fechas):
        f = ult - pd.DateOffset(months=ventana + i)
        tau = _tau(m["fechas"], f)
        sel = (tau >= -ventana) & (tau <= ventana - 1) & (m["fechas"] <= ult)
        if (tau[sel] >= 0).sum() < 2 or (tau[sel] < 0).sum() < 2:
            continue
        b, se, t, G = did_estatico(m["Y"][:, sel], m["D"], tau[sel])
        filas.append({"fecha_falsa": f.strftime("%Y-%m"), "beta": b, "se": se, "t": t,
                      "p_normal": 2 * stats.norm.sf(abs(t)),
                      "p_t_G-1": 2 * stats.t.sf(abs(t), G - 1)})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# Leave-one-out por capítulo
# ----------------------------------------------------------------------------
def leave_one_out(m, pre=6, post=6):
    """Re-estima el DiD estático (post = tau>=0) y el efecto medio post excluyendo cada capítulo."""
    Y, tau = _recortar(m, pre, post)
    filas = []
    for i, nombre in enumerate(m["productos"]):
        keep = np.arange(len(m["productos"])) != i
        D = m["D"][keep]
        if D.sum() == 0 or D.sum() == len(D):
            continue
        b, se, t, G = did_estatico(Y[keep], D, tau)
        filas.append({"excluido": nombre, "transable_excluido": int(m["D"][i]),
                      "beta_did": b, "se": se, "t": t,
                      "signo_igual_al_completo": None})
    df = pd.DataFrame(filas)
    b0 = did_estatico(Y, m["D"], tau)[0]
    df["signo_igual_al_completo"] = np.sign(df.beta_did) == np.sign(b0)
    return df, b0


# ----------------------------------------------------------------------------
# Ventanas alternativas
# ----------------------------------------------------------------------------
def ventanas_alternativas(m, grilla=((3, 3), (6, 3), (6, 6), (9, 6), (12, 6), (6, 9), (6, 12))):
    filas = []
    for pre, post in grilla:
        Y, tau = _recortar(m, pre, post)
        if (tau == -pre).sum() == 0 or (tau == post).sum() == 0:
            filas.append({"pre": pre, "post": post, "nota": "ventana fuera de datos"})
            continue
        b, se, t, G = did_estatico(Y, m["D"], tau)
        filas.append({"pre": pre, "post": post, "beta_did": b, "se": se, "t": t,
                      "p_t_G-1": 2 * stats.t.sf(abs(t), G - 1)})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# Randomization inference (permutar tratamiento entre capítulos)
# ----------------------------------------------------------------------------
def randomization_inference(m, pre=6, post=6, max_perm=20000):
    """
    Distribución de referencia del DiD estático bajo todas las asignaciones de
    n_tratados entre los N capítulos. Con N=12 y 5 tratados hay 792: se enumeran
    todas (exacta). p = proporción de asignaciones con |b| >= |b_obs|.
    Hipótesis nula: sharp (efecto cero para todos los capítulos). No corrige
    por la heterogeneidad preexistente de niveles de inflación entre
    capítulos, y la asignación NO fue aleatoria: es una medida de rareza del
    contraste, no de causalidad.
    """
    Y, tau = _recortar(m, pre, post)
    N = len(m["D"]); k = int(m["D"].sum())
    # DiD estático por MCO es lineal en D: b = (sum_i D_i * s_i) / (sum_i D_i^2 ...)  -> más simple, recalcular
    Yd = _demean2(Y[:, :, None])[:, :, 0]
    post_ind = (tau >= 0).astype(float)
    Pd = post_ind - post_ind.mean()                       # post demeaned en t
    # regresor demeaned: (D_i - Dbar) * (post_t - postbar)
    def beta(D):
        Dd = D - D.mean()
        x = np.outer(Dd, Pd)
        return float((x * Yd).sum() / (x * x).sum())
    obs = beta(m["D"])
    combos = itertools.combinations(range(N), k)
    betas = []
    for n, c in enumerate(combos):
        if n >= max_perm:
            break
        D = np.zeros(N); D[list(c)] = 1
        betas.append(beta(D))
    betas = np.array(betas)
    return {"beta_obs": obs, "n_asignaciones": len(betas), "exacta": len(betas) == math.comb(N, k),
            "p_dos_colas": float(np.mean(np.abs(betas) >= abs(obs) - 1e-15)),
            "rango_distribucion": (float(betas.min()), float(betas.max())),
            "p5_p95": tuple(float(x) for x in np.percentile(betas, [5, 95]))}


# ----------------------------------------------------------------------------
# Wild cluster bootstrap (WCR, Rademacher, nulo impuesto)
# ----------------------------------------------------------------------------
def _signos(G, B, rng):
    """Matriz (R,G) de pesos Rademacher: enumeración exacta si 2^G <= B, si no sorteo."""
    if 2 ** G <= B:
        return np.array(list(itertools.product([-1.0, 1.0], repeat=G)))
    return rng.choice([-1.0, 1.0], size=(B, G))


def wild_cluster_bootstrap_did(m, pre=6, post=6, B=9999, seed=SEMILLA):
    """
    WCR bootstrap-t para el DiD estático (H0: beta = 0). Con G=12 enumera las
    2^12=4096 combinaciones de signos (determinista; la semilla sólo se usa
    si G fuera grande). Con pocos clusters y pocos tratados el WCR sigue
    siendo imperfecto (Webb 2014; MacKinnon-Webb 2017): leer como orientativo.
    """
    rng = np.random.default_rng(seed)
    Y, tau = _recortar(m, pre, post)
    D = m["D"]
    X = (D[:, None] * (tau[None, :] >= 0))[:, :, None].astype(float)
    Yd, Xd = _demean2(Y), _demean2(X)
    b, V, _ = _ols_cluster(Yd, Xd)
    t_obs = float(b[0] / np.sqrt(V[0, 0]))
    # modelo restringido (beta=0): residuo = Yd (sólo EF)
    u = Yd
    W = _signos(Y.shape[0], B, rng)
    ts = np.empty(len(W))
    for r, w in enumerate(W):
        ystar = _demean2(u * w[:, None])  # y* = 0 + u*w, re-demeaned
        bb, VV, _ = _ols_cluster(ystar, Xd)
        ts[r] = bb[0] / np.sqrt(VV[0, 0]) if VV[0, 0] > 0 else 0.0
    return {"beta": float(b[0]), "t_obs": t_obs,
            "p_bootstrap_t": float(np.mean(np.abs(ts) >= abs(t_obs) - 1e-12)),
            "p_t_G-1": float(2 * stats.t.sf(abs(t_obs), Y.shape[0] - 1)),
            "p_normal": float(2 * stats.norm.sf(abs(t_obs))),
            "n_draws": len(W), "exacto": 2 ** Y.shape[0] <= B}


def wild_cluster_bootstrap_pretendencias(m, pre=6, post=6, B=9999, seed=SEMILLA):
    """
    WCR bootstrap del Wald conjunto H0: beta_tau=0, tau<=-2 (q = pre-1 restricciones).
    Estimación restringida: se excluyen los leads de la regresión; se remuestrea
    con signos Rademacher por cluster y se recalcula el Wald sin restringir.
    """
    rng = np.random.default_rng(seed)
    Y, tau = _recortar(m, pre, post)
    X, taus = _diseno_event_study(m["D"], tau)
    Yd, Xd = _demean2(Y), _demean2(X)
    idx_pre = [j for j, k in enumerate(taus) if k <= -2]
    idx_rest = [j for j in range(len(taus)) if j not in idx_pre]
    q = len(idx_pre)
    if q == 0:
        return None

    def wald(yd):
        b, V, _ = _ols_cluster(yd, Xd)
        bp, Vp = b[idx_pre], V[np.ix_(idx_pre, idx_pre)]
        return float(bp @ np.linalg.pinv(Vp) @ bp)

    w_obs = wald(Yd)
    # restringido: sólo regresores post
    br, _, e_r = _ols_cluster(Yd, Xd[:, :, idx_rest])
    fit_r = Xd[:, :, idx_rest] @ br
    W = _signos(Y.shape[0], B, rng)
    ws = np.array([wald(_demean2(fit_r + e_r * w[:, None])) for w in W])
    return {"wald_obs": w_obs, "df": q, "p_chi2": float(stats.chi2.sf(w_obs, q)),
            "p_bootstrap": float(np.mean(ws >= w_obs - 1e-12)), "n_draws": len(W),
            "exacto": 2 ** Y.shape[0] <= B}


def cruce_con_wildboottest(m, pre=6, post=6, B=9999, seed=SEMILLA):
    """
    Si `wildboottest` está instalado, replica el WCR del DiD estático con ese
    paquete (statsmodels OLS con dummies de EF) para contrastar con el propio.
    Devuelve None si no está disponible.
    """
    try:
        from wildboottest.wildboottest import wildboottest
        import statsmodels.api as sm
    except Exception:
        return None
    Y, tau = _recortar(m, pre, post)
    N, T = Y.shape
    df = pd.DataFrame({"y": Y.ravel(), "i": np.repeat(np.arange(N), T), "t": np.tile(np.arange(T), N),
                       "x": (m["D"][:, None] * (tau[None, :] >= 0)).ravel()})
    X = pd.get_dummies(df[["i", "t"]].astype(str), drop_first=True, dtype=float)
    X.insert(0, "x", df["x"])
    X.insert(0, "const", 1.0)
    mod = sm.OLS(df["y"], X)
    return wildboottest(mod, param="x", cluster=df["i"].to_numpy(), B=B, bootstrap_type="31",
                        weights_type="rademacher", impose_null=True, seed=seed,
                        parallel=False, show=False)


# ----------------------------------------------------------------------------
# Sensibilidad simple (NO es HonestDiD)
# ----------------------------------------------------------------------------
def sensibilidad_magnitud_relativa(m, pre=6, post=6, Ms=(0, 0.5, 1, 1.5, 2)):
    """
    Cotas heurísticas inspiradas en "relative magnitudes" de Rambachan-Roth:
    se supone que la violación de tendencias paralelas post no supera M veces
    la mayor desviación pre (max |beta_tau|, tau<=-2, puntual) respecto de la
    base. Se reporta  theta_hat +- (M*max_pre + 1.96*se)  para theta = media de
    beta_tau, tau>=0.

    LIMITACIONES: (i) no implementa el FLCI/ conditional de HonestDiD, así que
    el intervalo NO tiene cobertura garantizada; (ii) max_pre usa estimaciones
    puntuales ruidosas; (iii) ignora que la tendencia diferencial puede ser
    monótona (en cuyo caso se extrapolaría, no se acotaría). HonestDiD completo
    (restricciones de suavidad/SD) queda PENDIENTE: no hay port Python maduro.
    """
    Y, tau = _recortar(m, pre, post)
    tab, V = event_study_np(Y, m["D"], tau)
    j_post = np.where(tab.tau.to_numpy() >= 0)[0]
    j_pre = np.where(tab.tau.to_numpy() <= -2)[0]
    w = np.zeros(len(tab)); w[j_post] = 1 / len(j_post)
    theta = float(w @ tab.beta.to_numpy()); se = float(np.sqrt(w @ V @ w))
    max_pre = float(np.abs(tab.beta.to_numpy()[j_pre]).max())
    filas = [{"M": M, "theta": theta, "se": se, "max_pre_abs": max_pre,
              "lo": theta - M * max_pre - 1.96 * se, "hi": theta + M * max_pre + 1.96 * se}
             for M in Ms]
    df = pd.DataFrame(filas)
    df["incluye_cero"] = (df.lo <= 0) & (df.hi >= 0)
    be = (abs(theta) - 1.96 * se) / max_pre if max_pre > 0 else np.inf
    return df, be  # be = M de quiebre (<=0 => ya incluye 0 con M=0)


# ----------------------------------------------------------------------------
def correr_todo(df_long, pre=6, post=6, B=9999):
    """Ejecuta toda la batería y devuelve un dict de resultados."""
    m = panel_a_matrices(df_long)
    Y, tau = _recortar(m, pre, post)
    b, se, t, G = did_estatico(Y, m["D"], tau)
    out = {"did_estatico": {"beta": b, "se": se, "t": t, "G": G,
                            "n_tratados": int(m["D"].sum())},
           "placebo": placebo_fechas_falsas(m),
           "leave_one_out": leave_one_out(m, pre, post),
           "ventanas": ventanas_alternativas(m),
           "randomization": randomization_inference(m, pre, post),
           "wcr_did": wild_cluster_bootstrap_did(m, pre, post, B),
           "wcr_pretend": wild_cluster_bootstrap_pretendencias(m, pre, post, B),
           "wildboottest": cruce_con_wildboottest(m, pre, post, B),
           "sensibilidad": sensibilidad_magnitud_relativa(m, pre, post)}
    return m, out


if __name__ == "__main__":
    # Ejecutar desde la raíz:  python -m econometria.robustez [cache_dir]
    import sys
    from datos.api_indec import obtener_panel_capitulos
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    df_long = obtener_panel_capitulos(cache_dir=cache) if cache else obtener_panel_capitulos()
    m, r = correr_todo(df_long)
    pd.set_option("display.width", 160)
    for k, v in r.items():
        print(f"\n=== {k} ===")
        print(v)

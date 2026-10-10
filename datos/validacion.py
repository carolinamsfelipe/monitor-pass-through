"""
Validaciones de esquema y de panel para los datos del IPC INDEC.

Todas las funciones lanzan ``ErrorValidacionDatos`` con un mensaje explícito
(no devuelven booleanos silenciosos) para que ningún análisis corra sobre
datos malformados.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


class ErrorValidacionDatos(ValueError):
    """Los datos no cumplen el esquema o las propiedades de panel esperadas."""


def validar_esquema_raw(df: pd.DataFrame, columnas_series: Iterable[str],
                        col_fecha: str = "indice_tiempo") -> None:
    """Valida el CSV crudo: columnas esperadas, fechas únicas, sin NaN, índices > 0."""
    columnas_series = list(columnas_series)
    esperadas = [col_fecha] + columnas_series
    faltantes = [c for c in esperadas if c not in df.columns]
    if faltantes:
        raise ErrorValidacionDatos(
            f"Faltan {len(faltantes)} columnas esperadas en el dataset del INDEC: {faltantes}. "
            f"Columnas presentes: {list(df.columns)}. Es posible que la estructura o la "
            f"versión del dataset haya cambiado (revisar URL_IPC_CAPITULOS en datos/api_indec.py)."
        )
    if len(df) == 0:
        raise ErrorValidacionDatos("El dataset del INDEC está vacío.")

    fechas = pd.to_datetime(df[col_fecha], errors="coerce")
    if fechas.isna().any():
        raise ErrorValidacionDatos(f"Hay {int(fechas.isna().sum())} fechas no parseables en '{col_fecha}'.")
    if fechas.duplicated().any():
        dup = fechas[fechas.duplicated()].dt.strftime("%Y-%m").unique().tolist()
        raise ErrorValidacionDatos(f"Fechas duplicadas en '{col_fecha}': {dup[:10]}")

    sub = df[columnas_series].apply(pd.to_numeric, errors="coerce")
    nan_por_col = sub.isna().sum()
    nan_por_col = nan_por_col[nan_por_col > 0]
    if len(nan_por_col):
        raise ErrorValidacionDatos(
            "Valores faltantes o no numéricos en series del IPC: "
            + ", ".join(f"{c} ({n})" for c, n in nan_por_col.items())
        )
    malas = (sub <= 0).sum()
    malas = malas[malas > 0]
    if len(malas):
        raise ErrorValidacionDatos(
            "Índices de precios <= 0 (imposibles para un índice base 100): "
            + ", ".join(f"{c} ({n})" for c, n in malas.items())
        )


def validar_panel(df: pd.DataFrame, n_series: int = 12, ventana: Optional[int] = None,
                  col_id: str = "serie_id", col_fecha: str = "fecha",
                  col_valor: str = "ipc_indice") -> None:
    """
    Valida un panel largo (serie x fecha): columnas, NaN, duplicados (id, fecha),
    índices > 0 y balance (``n_series`` series con las mismas fechas).
    Si ``ventana`` se indica, exige exactamente ``ventana`` fechas por serie.
    """
    req = [col_id, col_fecha, col_valor]
    faltan = [c for c in req if c not in df.columns]
    if faltan:
        raise ErrorValidacionDatos(f"Faltan columnas en el panel: {faltan}")
    if df[req].isna().any().any():
        raise ErrorValidacionDatos(f"El panel tiene NaN en {req}: {df[req].isna().sum().to_dict()}")
    if df.duplicated([col_id, col_fecha]).any():
        raise ErrorValidacionDatos(
            f"Hay {int(df.duplicated([col_id, col_fecha]).sum())} filas duplicadas por ({col_id}, {col_fecha})."
        )
    if (df[col_valor] <= 0).any():
        raise ErrorValidacionDatos(f"Hay {int((df[col_valor] <= 0).sum())} valores de '{col_valor}' <= 0.")
    n = df[col_id].nunique()
    if n != n_series:
        raise ErrorValidacionDatos(f"Se esperaban {n_series} series y hay {n}.")
    por_serie = df.groupby(col_id)[col_fecha].nunique()
    if por_serie.nunique() != 1:
        raise ErrorValidacionDatos(
            f"Panel desbalanceado: cantidad de fechas por serie distinta: {por_serie.to_dict()}"
        )
    if ventana is not None and int(por_serie.iloc[0]) != ventana:
        raise ErrorValidacionDatos(
            f"Se esperaban {ventana} períodos por serie y hay {int(por_serie.iloc[0])} "
            f"(panel {n_series}x{ventana} = {n_series * ventana} filas; hay {len(df)})."
        )
    fechas_por_serie = df.groupby(col_id)[col_fecha].apply(lambda s: tuple(sorted(s)))
    if fechas_por_serie.nunique() != 1:
        raise ErrorValidacionDatos("Las series no comparten el mismo conjunto de fechas.")

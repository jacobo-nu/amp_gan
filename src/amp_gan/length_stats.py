"""Estadísticas descriptivas de la longitud de péptidos.

Funciones puras (sin E/S) para resumir la distribución de longitudes de un
conjunto de secuencias y para calcular qué fracción queda cubierta por distintos
límites superiores de longitud. No filtran ni recortan datos.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

#: Percentiles incluidos en el resumen, como ``(nombre de columna, porcentaje)``.
PERCENTILES: tuple[tuple[str, float], ...] = (
    ("p1", 1.0),
    ("p5", 5.0),
    ("p25", 25.0),
    ("median", 50.0),
    ("p75", 75.0),
    ("p90", 90.0),
    ("p95", 95.0),
    ("p99", 99.0),
)

#: Cortes de longitud por defecto para la tabla de cobertura.
DEFAULT_CUTOFFS: tuple[int, ...] = (20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100)

#: Columnas del resumen, en orden.
SUMMARY_KEYS: tuple[str, ...] = (
    "n",
    "mean",
    "std",
    "min",
    "p1",
    "p5",
    "p25",
    "median",
    "p75",
    "p90",
    "p95",
    "p99",
    "max",
    "mode",
)


def sequence_lengths(sequences: Iterable[str]) -> np.ndarray:
    """Devuelve las longitudes de una colección de secuencias.

    Args:
        sequences: iterable de cadenas.

    Returns:
        Array de enteros (``int64``) con la longitud de cada secuencia, en orden.
    """
    return np.fromiter((len(s) for s in sequences), dtype=np.int64)


def length_summary(lengths: Sequence[int] | np.ndarray | pd.Series) -> dict[str, float]:
    """Resume la distribución de longitudes.

    Los percentiles usan interpolación lineal (``numpy.percentile`` por defecto).
    La desviación típica es la muestral (``ddof=1``; 0.0 si ``n == 1``). La moda
    es la longitud más frecuente; en caso de empate, la menor.

    Args:
        lengths: longitudes (enteros no negativos); no puede estar vacío.

    Returns:
        Diccionario con las claves de :data:`SUMMARY_KEYS`: ``n``, ``mean``,
        ``std``, ``min``, ``p1``, ``p5``, ``p25``, ``median``, ``p75``, ``p90``,
        ``p95``, ``p99``, ``max`` y ``mode``.

    Raises:
        ValueError: si ``lengths`` está vacío.
    """
    arr = np.asarray(lengths, dtype=np.int64)
    if arr.size == 0:
        raise ValueError("No se puede resumir un conjunto de longitudes vacío.")

    values, counts = np.unique(arr, return_counts=True)
    summary: dict[str, float] = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": int(arr.min()),
    }
    for name, pct in PERCENTILES:
        summary[name] = float(np.percentile(arr, pct))
    summary["max"] = int(arr.max())
    summary["mode"] = int(values[np.argmax(counts)])  # argmax devuelve la primera: la menor
    return {key: summary[key] for key in SUMMARY_KEYS}


def coverage_table(
    lengths: Sequence[int] | np.ndarray | pd.Series,
    cutoffs: Iterable[int] = DEFAULT_CUTOFFS,
) -> pd.DataFrame:
    """Calcula cuántas secuencias tienen longitud menor o igual que cada corte.

    Args:
        lengths: longitudes de las secuencias; no puede estar vacío.
        cutoffs: límites superiores de longitud (inclusivos).

    Returns:
        DataFrame con columnas ``cutoff``, ``n_le`` (nº de secuencias con
        longitud <= corte), ``fraction_le`` (fracción sobre el total, en [0, 1])
        y ``n_total``, una fila por corte en el orden dado.

    Raises:
        ValueError: si ``lengths`` está vacío.
    """
    arr = np.asarray(lengths, dtype=np.int64)
    if arr.size == 0:
        raise ValueError("No se puede calcular la cobertura de un conjunto vacío.")

    cut_list = [int(c) for c in cutoffs]
    n_le = [int((arr <= c).sum()) for c in cut_list]
    return pd.DataFrame(
        {
            "cutoff": cut_list,
            "n_le": n_le,
            "fraction_le": [n / arr.size for n in n_le],
            "n_total": arr.size,
        }
    )

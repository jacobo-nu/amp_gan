"""Condiciones numéricas de la GAN condicional y su evaluación (numpy/pandas; sin Keras).

Este módulo no importa Keras, así que se puede usar antes de elegir el backend
(el modelo y el entrenamiento están en :mod:`amp_gan.conditional_gan`).

Condiciones admitidas (:data:`CONDITION_NAMES`) y su normalización a ``[0, 1]``:

``length``
    longitud de la secuencia; ``(L - 5) / 55`` (rango fijo 5-60 aa del proyecto,
    no depende de los datos).
``hydrophobicity``
    hidrofobicidad media de Kyte-Doolittle (GRAVY); min-max con el mínimo y el
    máximo de las secuencias de ENTRENAMIENTO (``(g - min) / (max - min)``).

Una :class:`ConditionSpec` guarda los nombres y los extremos de cada condición y
convierte valores reales en condiciones normalizadas y al revés.

Evaluación por objetivos de longitud (solo condición ``length``)
----------------------------------------------------------------
Se pide al generador una longitud objetivo ``t`` y se mide si la obtiene:

* ``length_mean``: longitud media de TODAS las secuencias generadas (las vacías
  o más cortas que 5 aa cuentan con su longitud real, 0 para una cadena vacía).
* ``length_mae``: ``mean(|L - t|)`` sobre las mismas secuencias.
* ``hit_rate`` (tasa de acierto): fracción de secuencias con ``|L - t| <= tol``
  (``tol`` = :data:`HIT_TOLERANCE` = 2, fijada de antemano).
* ``valid_fraction`` y ``unique_fraction`` como en :mod:`amp_gan.evaluation`.
* W1 de carga neta e hidrofobicidad de las secuencias válidas frente a los datos
  reales con longitud en ``[t - tol, t + tol]`` (``NaN`` si ese subconjunto de
  referencia tiene menos de :data:`MIN_REFERENCE_N` secuencias).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from amp_gan.data import MAX_LEN, MIN_LEN
from amp_gan.encoding import VOCAB, lengths_from_one_hot
from amp_gan.evaluation import ReferenceProfile, build_reference, is_valid, profile_sequences
from amp_gan.properties import KYTE_DOOLITTLE, hydrophobicity

#: Condiciones numéricas soportadas.
CONDITION_NAMES: tuple[str, ...] = ("length", "hydrophobicity")

#: Percentiles de longitud que definen los objetivos y el rango «habitual» (p10-p90).
TARGET_PERCENTILES: tuple[int, ...] = (10, 30, 50, 70, 90)

#: Tolerancia (aa) de la tasa de acierto: se acierta si ``|L - t| <= HIT_TOLERANCE``.
HIT_TOLERANCE: int = 2

#: Tamaño mínimo del subconjunto real de referencia en la evaluación por objetivos.
MIN_REFERENCE_N: int = 30

#: Columnas de la evaluación por objetivos (tras ``epoch`` y ``mode``).
TARGET_COLUMNS: tuple[str, ...] = (
    "target",
    "n",
    "n_valid",
    "length_mean",
    "length_mae",
    "hit_rate",
    "valid_fraction",
    "unique_fraction",
    "n_reference",
    "net_charge_wasserstein",
    "hydrophobicity_wasserstein",
)

# GRAVY por canal de la codificación one-hot (PAD aporta 0).
_KD_BY_CHANNEL = np.array([KYTE_DOOLITTLE.get(symbol, 0.0) for symbol in VOCAB], dtype="float64")


def check_condition_names(names: Sequence[str]) -> tuple[str, ...]:
    """Valida y devuelve los nombres de condición (ValueError si hay alguno desconocido, repetido o ninguno)."""
    names = tuple(names)
    if not names:
        raise ValueError("Hace falta al menos una condición.")
    unknown = [n for n in names if n not in CONDITION_NAMES]
    if unknown:
        raise ValueError(f"Condiciones desconocidas {unknown}; admitidas: {CONDITION_NAMES}.")
    if len(set(names)) != len(names):
        raise ValueError(f"Condiciones repetidas: {names}.")
    return names


# --------------------------------------------------------------------------- #
# Valores reales de las condiciones
# --------------------------------------------------------------------------- #
def raw_conditions_from_sequences(sequences: Sequence[str], names: Sequence[str]) -> np.ndarray:
    """Valores reales (sin normalizar) de las condiciones, ``(n, k)`` ``float64``, a partir de cadenas."""
    names = check_condition_names(names)
    seqs = list(sequences)
    columns = {
        "length": [len(s) for s in seqs],
        "hydrophobicity": [hydrophobicity(s) for s in seqs] if "hydrophobicity" in names else [],
    }
    return np.column_stack([np.asarray(columns[n], dtype="float64") for n in names]).reshape(len(seqs), len(names))


def raw_conditions_from_encoded(X: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Igual que :func:`raw_conditions_from_sequences` pero desde el one-hot ``(n, seq_len, 21)``.

    La longitud son las posiciones antes del primer PAD y el GRAVY la media de la
    escala de Kyte-Doolittle sobre ellas (el relleno aporta 0, así que basta sumar
    todas las posiciones).
    """
    names = check_condition_names(names)
    X = np.asarray(X)
    lengths = lengths_from_one_hot(X).astype("float64")
    columns = {"length": lengths}
    if "hydrophobicity" in names:
        columns["hydrophobicity"] = (X.astype("float64") @ _KD_BY_CHANNEL).sum(axis=1) / np.maximum(lengths, 1.0)
    return np.column_stack([columns[n] for n in names])


# --------------------------------------------------------------------------- #
# Normalización
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConditionSpec:
    """Condiciones numéricas y sus extremos de normalización (``low``, ``high`` por condición)."""

    names: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]

    def __post_init__(self) -> None:
        check_condition_names(self.names)
        if not (len(self.names) == len(self.low) == len(self.high)):
            raise ValueError("names, low y high deben tener la misma longitud.")
        if any(h <= lo for lo, h in zip(self.low, self.high)):
            raise ValueError(f"Rango de normalización degenerado: low={self.low}, high={self.high}.")

    @property
    def k(self) -> int:
        """Dimensión de la condición."""
        return len(self.names)

    def normalize(self, raw) -> np.ndarray:
        """Valores reales ``(n, k)`` (o ``(k,)``) -> condición en ``[0, 1]`` (``float32``, sin recortar)."""
        raw = np.asarray(raw, dtype="float64")
        low, high = np.asarray(self.low), np.asarray(self.high)
        return ((raw - low) / (high - low)).astype("float32")

    def denormalize(self, normalized) -> np.ndarray:
        """Inversa de :meth:`normalize` (``float64``)."""
        normalized = np.asarray(normalized, dtype="float64")
        low, high = np.asarray(self.low), np.asarray(self.high)
        return normalized * (high - low) + low

    def to_dict(self) -> dict:
        return {"names": list(self.names), "low": list(self.low), "high": list(self.high)}


def fit_condition_spec(names: Sequence[str], raw_train: np.ndarray) -> ConditionSpec:
    """Ajusta la normalización con los valores reales del entrenamiento ``(n, k)``.

    La longitud usa el rango fijo ``[MIN_LEN, MAX_LEN]`` = ``[5, 60]``; la
    hidrofobicidad, el mínimo y el máximo de ``raw_train``.
    """
    names = check_condition_names(names)
    raw_train = np.asarray(raw_train, dtype="float64")
    if raw_train.ndim != 2 or raw_train.shape[1] != len(names):
        raise ValueError(f"raw_train debe tener forma (n, {len(names)}), se recibió {raw_train.shape}.")
    low, high = [], []
    for j, name in enumerate(names):
        if name == "length":
            low.append(float(MIN_LEN))
            high.append(float(MAX_LEN))
        else:
            low.append(float(raw_train[:, j].min()))
            high.append(float(raw_train[:, j].max()))
    return ConditionSpec(names, tuple(low), tuple(high))


def condition_spec_from_encoded(names: Sequence[str], X: np.ndarray) -> tuple[ConditionSpec, np.ndarray]:
    """Atajo para el entrenamiento: ``(spec, condiciones normalizadas (n, k))`` desde el one-hot ``X``."""
    raw = raw_conditions_from_encoded(X, names)
    spec = fit_condition_spec(names, raw)
    return spec, spec.normalize(raw)


# --------------------------------------------------------------------------- #
# Objetivos de longitud (percentiles) y evaluación
# --------------------------------------------------------------------------- #
def length_percentiles(lengths: Sequence[float], percentiles: Sequence[float] = TARGET_PERCENTILES) -> dict[int, float]:
    """Percentiles de longitud (interpolación lineal de numpy): ``{percentil: valor}``."""
    values = np.percentile(np.asarray(lengths, dtype="float64"), list(percentiles))
    return {int(p): float(v) for p, v in zip(percentiles, values)}


def usual_length_range(lengths: Sequence[float], lo: float = 10, hi: float = 90) -> tuple[int, int]:
    """Rango de longitudes «habitual»: percentiles ``lo``-``hi`` redondeados al entero más cercano."""
    p = length_percentiles(lengths, (lo, hi))
    return int(round(p[int(lo)])), int(round(p[int(hi)]))


def target_lengths(lengths: Sequence[float], percentiles: Sequence[float] = TARGET_PERCENTILES) -> tuple[int, ...]:
    """Longitudes objetivo: los percentiles ``percentiles`` de ``lengths`` redondeados a enteros."""
    return tuple(int(round(v)) for v in length_percentiles(lengths, percentiles).values())


def hit_rate(lengths: Sequence[float], target: float, tol: float = HIT_TOLERANCE) -> float:
    """Fracción de longitudes con ``|L - target| <= tol`` (``NaN`` si no hay ninguna)."""
    arr = np.asarray(lengths, dtype="float64")
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(arr - target) <= tol))


def length_control_slope(targets: Sequence[float], mean_lengths: Sequence[float]) -> tuple[float, float]:
    """Pendiente y ordenada de la regresión lineal de la longitud media generada sobre la pedida."""
    slope, intercept = np.polyfit(np.asarray(targets, dtype="float64"), np.asarray(mean_lengths, dtype="float64"), 1)
    return float(slope), float(intercept)


def real_subset_around(sequences: Sequence[str], target: float, tol: float = HIT_TOLERANCE) -> list[str]:
    """Secuencias reales con longitud en ``[target - tol, target + tol]``."""
    return [s for s in sequences if abs(len(s) - target) <= tol]


def build_target_references(
    real_sequences: Sequence[str],
    targets: Sequence[int],
    tol: float = HIT_TOLERANCE,
    min_n: int = MIN_REFERENCE_N,
) -> dict[int, tuple[ReferenceProfile | None, int]]:
    """Subconjunto real de referencia de cada objetivo: ``{t: (referencia o None, n)}``.

    ``None`` si el subconjunto (válido) tiene menos de ``min_n`` secuencias.
    """
    out: dict[int, tuple[ReferenceProfile | None, int]] = {}
    for t in targets:
        subset = [s for s in real_subset_around(real_sequences, t, tol) if is_valid(s)]
        out[int(t)] = (build_reference(subset) if len(subset) >= min_n else None, len(subset))
    return out


def evaluate_target(
    sequences: Sequence[str],
    target: float,
    reference: ReferenceProfile | None,
    n_reference: int,
    tol: float = HIT_TOLERANCE,
) -> dict[str, float]:
    """Métricas de una tanda de secuencias pedidas con longitud ``target`` (ver el docstring del módulo).

    Args:
        sequences: secuencias generadas (pueden ser vacías o no válidas).
        target: longitud pedida.
        reference: subconjunto real con longitud cercana a ``target`` (``None``: W1 = ``NaN``).
        n_reference: tamaño de ese subconjunto (se registra).
        tol: tolerancia de la tasa de acierto.

    Returns:
        Diccionario con las claves de :data:`TARGET_COLUMNS`.
    """
    seqs = list(sequences)
    lengths = np.array([len(s) for s in seqs], dtype="float64")
    out = {name: float("nan") for name in TARGET_COLUMNS}
    out.update(target=int(target), n=len(seqs), n_valid=sum(is_valid(s) for s in seqs), n_reference=n_reference)
    if len(seqs):
        out.update(
            length_mean=float(lengths.mean()),
            length_mae=float(np.abs(lengths - target).mean()),
            hit_rate=hit_rate(lengths, target, tol),
        )
    valid = [s for s in seqs if is_valid(s)]
    if seqs:
        out["valid_fraction"] = len(valid) / len(seqs)
    if valid:
        out["unique_fraction"] = len(set(valid)) / len(valid)
        if reference is not None:
            prof = profile_sequences(valid)
            for prop in ("net_charge", "hydrophobicity"):
                out[f"{prop}_wasserstein"] = float(wasserstein_distance(prof.values(prop), reference.profile.values(prop)))
    return out


def unconditional_target_table(
    sequences: Sequence[str], targets: Sequence[int], tol: float = HIT_TOLERANCE
) -> pd.DataFrame:
    """Referencia sin condición: lo que daría por azar una muestra que no se puede dirigir.

    Para cada objetivo ``t``: tasa de acierto ``|L - t| <= tol`` y MAE de la muestra
    completa, y su longitud media (igual en todas las filas).
    """
    lengths = np.array([len(s) for s in sequences], dtype="float64")
    return pd.DataFrame(
        {
            "target": list(targets),
            "length_mean": float(lengths.mean()),
            "length_mae": [float(np.abs(lengths - t).mean()) for t in targets],
            "hit_rate": [hit_rate(lengths, t, tol) for t in targets],
        }
    )


def real_ceiling(real_lengths: Sequence[float], targets: Sequence[int], tol: float = HIT_TOLERANCE) -> pd.DataFrame:
    """Techo real: fracción de los datos reales con ``|L - t| <= tol`` para cada objetivo."""
    return pd.DataFrame(
        {"target": list(targets), "hit_rate": [hit_rate(real_lengths, t, tol) for t in targets]}
    )

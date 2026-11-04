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

Evaluación por objetivos (una propiedad: ``length`` o ``hydrophobicity``)
-------------------------------------------------------------------------
Se pide al generador un valor objetivo ``t`` de UNA propiedad (``prop``) y se mide
si lo obtiene. Los objetivos son los percentiles :data:`TARGET_PERCENTILES` de la
propiedad en los datos reales (``target_values``): para la longitud se redondean
a enteros; para la hidrofobicidad (GRAVY de Kyte-Doolittle) se usan los valores
reales, sin redondear. La tolerancia de acierto (``tol``) está fijada de antemano
(:data:`HIT_TOLERANCES`): ±2 aa en la longitud y ±0,25 en el GRAVY.

Con ``prop = "length"`` (columnas :data:`TARGET_COLUMNS`):

* ``length_mean``: longitud media de TODAS las secuencias generadas (las vacías
  o más cortas que 5 aa cuentan con su longitud real, 0 para una cadena vacía).
* ``length_mae``: ``mean(|L - t|)`` sobre las mismas secuencias.
* ``hit_rate`` (tasa de acierto): fracción de secuencias con ``|L - t| <= tol``.
* ``valid_fraction`` y ``unique_fraction`` como en :mod:`amp_gan.evaluation`.
* W1 de carga neta e hidrofobicidad de las secuencias válidas frente a los datos
  reales con longitud en ``[t - tol, t + tol]`` (``NaN`` si ese subconjunto de
  referencia tiene menos de :data:`MIN_REFERENCE_N` secuencias).

Con ``prop = "hydrophobicity"`` (columnas :data:`HYDRO_TARGET_COLUMNS`) lo mismo
con el GRAVY como propiedad principal: ``hydrophobicity_mean`` y
``hydrophobicity_mae`` se calculan sobre las secuencias generadas NO vacías (una
cadena vacía no tiene GRAVY; las no válidas por longitud sí cuentan), la tasa de
acierto usa TODAS (una cadena vacía es un fallo) y las W1 secundarias son las de
longitud y carga neta, frente a los datos reales con GRAVY en ``[t - tol, t + tol]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from amp_gan.data import MAX_LEN, MIN_LEN, STANDARD_ALPHABET
from amp_gan.encoding import VOCAB, lengths_from_one_hot
from amp_gan.evaluation import ReferenceProfile, build_reference, is_valid, profile_sequences
from amp_gan.properties import KYTE_DOOLITTLE, hydrophobicity

#: Condiciones numéricas soportadas.
CONDITION_NAMES: tuple[str, ...] = ("length", "hydrophobicity")

#: Percentiles de longitud que definen los objetivos y el rango «habitual» (p10-p90).
TARGET_PERCENTILES: tuple[int, ...] = (10, 30, 50, 70, 90)

#: Tolerancia (aa) de la tasa de acierto: se acierta si ``|L - t| <= HIT_TOLERANCE``.
HIT_TOLERANCE: int = 2

#: Tolerancia (unidades de GRAVY) de la tasa de acierto de la hidrofobicidad: ``|GRAVY - t| <= 0.25``.
HYDRO_HIT_TOLERANCE: float = 0.25

#: Propiedades que se pueden pedir como objetivo en la evaluación por objetivos.
TARGET_PROPERTIES: tuple[str, ...] = ("length", "hydrophobicity")

#: Tolerancia de acierto fijada de antemano para cada propiedad.
HIT_TOLERANCES: dict[str, float] = {"length": HIT_TOLERANCE, "hydrophobicity": HYDRO_HIT_TOLERANCE}

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

#: Columnas de la evaluación por objetivos de hidrofobicidad (tras ``epoch`` y ``mode``).
HYDRO_TARGET_COLUMNS: tuple[str, ...] = (
    "target",
    "n",
    "n_valid",
    "hydrophobicity_mean",
    "hydrophobicity_mae",
    "hit_rate",
    "valid_fraction",
    "unique_fraction",
    "n_reference",
    "length_wasserstein",
    "net_charge_wasserstein",
)

_STANDARD = frozenset(STANDARD_ALPHABET)

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
# Objetivos (percentiles) y evaluación por objetivos
# --------------------------------------------------------------------------- #
def check_target_property(prop: str) -> str:
    """Valida la propiedad objetivo (``length`` o ``hydrophobicity``)."""
    if prop not in TARGET_PROPERTIES:
        raise ValueError(f"Propiedad objetivo desconocida {prop!r}; admitidas: {TARGET_PROPERTIES}.")
    return prop


def default_tolerance(prop: str) -> float:
    """Tolerancia de acierto fijada de antemano para ``prop``."""
    return HIT_TOLERANCES[check_target_property(prop)]


def target_columns(prop: str = "length") -> tuple[str, ...]:
    """Columnas de la evaluación por objetivos de ``prop`` (tras ``epoch`` y ``mode``)."""
    return HYDRO_TARGET_COLUMNS if check_target_property(prop) == "hydrophobicity" else TARGET_COLUMNS


def target_key(prop: str, target: float) -> int | float:
    """Clave de un objetivo en los diccionarios: entero para la longitud y ``float`` para el GRAVY."""
    return int(target) if check_target_property(prop) == "length" else float(target)


def target_label(prop: str, target: float) -> str:
    """Texto del objetivo para los identificadores FASTA: ``t09`` (longitud 9) o ``g-1.465`` (GRAVY)."""
    if check_target_property(prop) == "length":
        return f"t{int(target):02d}"
    return f"g{float(target):+.3f}"


def property_values(sequences: Sequence[str], prop: str = "length") -> np.ndarray:
    """Valor de ``prop`` de cada secuencia (``float64``).

    Longitud: ``len(s)`` (0 para la cadena vacía). Hidrofobicidad: GRAVY de
    Kyte-Doolittle; ``NaN`` para la cadena vacía o con caracteres no estándar.
    """
    seqs = list(sequences)
    if check_target_property(prop) == "length":
        return np.array([len(s) for s in seqs], dtype="float64")
    return np.array(
        [hydrophobicity(s) if (isinstance(s, str) and s and set(s) <= _STANDARD) else np.nan for s in seqs],
        dtype="float64",
    )


def _mean_ignoring_nan(values: np.ndarray) -> float:
    """Media de ``values`` sin los ``NaN`` (``NaN`` si no queda ninguno)."""
    if np.isnan(values).any():
        values = values[~np.isnan(values)]
    return float(values.mean()) if values.size else float("nan")


def length_percentiles(lengths: Sequence[float], percentiles: Sequence[float] = TARGET_PERCENTILES) -> dict[int, float]:
    """Percentiles de longitud (interpolación lineal de numpy): ``{percentil: valor}``."""
    values = np.percentile(np.asarray(lengths, dtype="float64"), list(percentiles))
    return {int(p): float(v) for p, v in zip(percentiles, values)}


#: Percentiles de cualquier propiedad (misma función: ``numpy.percentile`` con interpolación lineal).
property_percentiles = length_percentiles


def usual_length_range(lengths: Sequence[float], lo: float = 10, hi: float = 90) -> tuple[int, int]:
    """Rango de longitudes «habitual»: percentiles ``lo``-``hi`` redondeados al entero más cercano."""
    p = length_percentiles(lengths, (lo, hi))
    return int(round(p[int(lo)])), int(round(p[int(hi)]))


def usual_range(values: Sequence[float], lo: float = 10, hi: float = 90) -> tuple[float, float]:
    """Rango «habitual» de cualquier propiedad: percentiles ``lo``-``hi`` SIN redondear."""
    p = property_percentiles(values, (lo, hi))
    return p[int(lo)], p[int(hi)]


def target_lengths(lengths: Sequence[float], percentiles: Sequence[float] = TARGET_PERCENTILES) -> tuple[int, ...]:
    """Longitudes objetivo: los percentiles ``percentiles`` de ``lengths`` redondeados a enteros."""
    return tuple(int(round(v)) for v in length_percentiles(lengths, percentiles).values())


def target_values(
    values: Sequence[float], prop: str = "length", percentiles: Sequence[float] = TARGET_PERCENTILES
) -> tuple[float, ...]:
    """Objetivos de ``prop``: percentiles de ``values`` (longitud: enteros; GRAVY: valores reales sin redondear)."""
    if check_target_property(prop) == "length":
        return target_lengths(values, percentiles)
    return tuple(float(v) for v in property_percentiles(values, percentiles).values())


def hit_rate(lengths: Sequence[float], target: float, tol: float = HIT_TOLERANCE) -> float:
    """Fracción de valores con ``|v - target| <= tol`` (``NaN`` si no hay ninguno; un ``NaN`` cuenta como fallo).

    Sirve para cualquier propiedad (el nombre del argumento es histórico); la
    tolerancia por defecto es la de la longitud.
    """
    arr = np.asarray(lengths, dtype="float64")
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(arr - target) <= tol))


def control_slope(targets: Sequence[float], mean_values: Sequence[float]) -> tuple[float, float]:
    """Pendiente y ordenada de la regresión lineal del valor medio generado sobre el pedido."""
    slope, intercept = np.polyfit(np.asarray(targets, dtype="float64"), np.asarray(mean_values, dtype="float64"), 1)
    return float(slope), float(intercept)


def length_control_slope(targets: Sequence[float], mean_lengths: Sequence[float]) -> tuple[float, float]:
    """Pendiente y ordenada de la regresión lineal de la longitud media generada sobre la pedida."""
    return control_slope(targets, mean_lengths)


def real_subset_around(
    sequences: Sequence[str], target: float, tol: float | None = None, prop: str = "length"
) -> list[str]:
    """Secuencias reales con ``prop`` en ``[target - tol, target + tol]`` (``tol`` por defecto: el de ``prop``)."""
    tol = default_tolerance(prop) if tol is None else tol
    seqs = list(sequences)
    values = property_values(seqs, prop)
    return [s for s, v in zip(seqs, values) if abs(v - target) <= tol]


def build_target_references(
    real_sequences: Sequence[str],
    targets: Sequence[float],
    tol: float | None = None,
    min_n: int = MIN_REFERENCE_N,
    prop: str = "length",
) -> dict[int | float, tuple[ReferenceProfile | None, int]]:
    """Subconjunto real de referencia de cada objetivo: ``{t: (referencia o None, n)}``.

    ``None`` si el subconjunto (válido) tiene menos de ``min_n`` secuencias. Las
    claves son enteros para la longitud y ``float`` para el GRAVY.
    """
    out: dict[int | float, tuple[ReferenceProfile | None, int]] = {}
    for t in targets:
        subset = [s for s in real_subset_around(real_sequences, t, tol, prop) if is_valid(s)]
        out[target_key(prop, t)] = (build_reference(subset) if len(subset) >= min_n else None, len(subset))
    return out


def evaluate_target(
    sequences: Sequence[str],
    target: float,
    reference: ReferenceProfile | None,
    n_reference: int,
    tol: float | None = None,
    prop: str = "length",
) -> dict[str, float]:
    """Métricas de una tanda de secuencias pedidas con ``prop = target`` (ver el docstring del módulo).

    Args:
        sequences: secuencias generadas (pueden ser vacías o no válidas).
        target: valor pedido (longitud o GRAVY).
        reference: subconjunto real con ``prop`` cercana a ``target`` (``None``: W1 = ``NaN``).
        n_reference: tamaño de ese subconjunto (se registra).
        tol: tolerancia de la tasa de acierto (por defecto la de ``prop``).
        prop: ``"length"`` o ``"hydrophobicity"``.

    Returns:
        Diccionario con las claves de :func:`target_columns` (``TARGET_COLUMNS`` para la longitud).
    """
    tol = default_tolerance(prop) if tol is None else tol
    columns = target_columns(prop)
    seqs = list(sequences)
    values = property_values(seqs, prop)
    out = {name: float("nan") for name in columns}
    out.update(target=target_key(prop, target), n=len(seqs), n_valid=sum(is_valid(s) for s in seqs), n_reference=n_reference)
    if len(seqs):
        out.update(
            {
                f"{prop}_mean": _mean_ignoring_nan(values),
                f"{prop}_mae": _mean_ignoring_nan(np.abs(values - target)),
                "hit_rate": hit_rate(values, target, tol),
            }
        )
    valid = [s for s in seqs if is_valid(s)]
    if seqs:
        out["valid_fraction"] = len(valid) / len(seqs)
    if valid:
        out["unique_fraction"] = len(set(valid)) / len(valid)
        if reference is not None:
            prof = profile_sequences(valid)
            secondary = ("net_charge", "hydrophobicity") if prop == "length" else ("length", "net_charge")
            for other in secondary:
                out[f"{other}_wasserstein"] = float(wasserstein_distance(prof.values(other), reference.profile.values(other)))
    return out


def unconditional_target_table(
    sequences: Sequence[str], targets: Sequence[float], tol: float | None = None, prop: str = "length"
) -> pd.DataFrame:
    """Referencia sin condición: lo que daría por azar una muestra que no se puede dirigir.

    Para cada objetivo ``t``: tasa de acierto ``|v - t| <= tol`` y MAE de la muestra
    completa, y su valor medio (igual en todas las filas). Columnas ``target``,
    ``{prop}_mean``, ``{prop}_mae`` y ``hit_rate``. Con el GRAVY, las cadenas vacías
    (sin GRAVY) cuentan como fallo y quedan fuera de la media y del MAE.
    """
    tol = default_tolerance(prop) if tol is None else tol
    values = property_values(sequences, prop)
    return pd.DataFrame(
        {
            "target": list(targets),
            f"{prop}_mean": _mean_ignoring_nan(values),
            f"{prop}_mae": [_mean_ignoring_nan(np.abs(values - t)) for t in targets],
            "hit_rate": [hit_rate(values, t, tol) for t in targets],
        }
    )


def real_ceiling(
    real_values: Sequence[float], targets: Sequence[float], tol: float = HIT_TOLERANCE
) -> pd.DataFrame:
    """Techo real: fracción de los datos reales con ``|v - t| <= tol`` para cada objetivo.

    ``real_values`` son los valores de la propiedad (longitudes o GRAVY) de los datos
    reales; ``tol`` por defecto es el de la longitud (pasar :data:`HYDRO_HIT_TOLERANCE` para el GRAVY).
    """
    return pd.DataFrame({"target": list(targets), "hit_rate": [hit_rate(real_values, t, tol) for t in targets]})

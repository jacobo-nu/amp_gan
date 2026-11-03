"""Métricas para evaluar secuencias generadas frente a secuencias reales.

Funciones puras (numpy, pandas y scipy; sin Keras) que comparan una lista de
secuencias generadas con una referencia de secuencias reales:

* calidad básica: fracción de secuencias válidas, únicas y nuevas;
* distribuciones de longitud, carga neta (pH 7,4) e hidrofobicidad (GRAVY):
  media, desviación típica y distancia de Wasserstein-1 frente a la referencia;
* composición de aminoácidos: distancia de variación total entre la
  composición media generada y la de la referencia.

Claves del diccionario que devuelve :func:`evaluate_sequences` (también son las
columnas métricas de ``gan_base_metrics.csv``, ver :data:`METRIC_NAMES`):

``n``
    número de secuencias generadas recibidas.
``n_valid``
    número de secuencias válidas (ver :func:`is_valid`).
``valid_fraction``
    ``n_valid / n``.
``unique_fraction``
    secuencias distintas entre las válidas / ``n_valid``.
``novel_fraction``
    válidas que NO están en la referencia / ``n_valid`` (cuenta secuencias,
    incluidas las repetidas).
``length_mean``, ``length_std``, ``length_wasserstein`` (y lo mismo con
``net_charge`` y ``hydrophobicity``)
    media, desviación típica (``ddof=1``) y distancia de Wasserstein-1 entre la
    distribución generada (solo válidas) y la de la referencia.
``composition_tv``
    ``0.5 * sum(|p - q|)`` con ``p`` y ``q`` la composición media de aminoácidos
    (media de las fracciones por secuencia) generada y de referencia.

Si no hay secuencias válidas, todas las métricas que dependen de ellas son
``NaN`` (no se lanza ninguna excepción).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from amp_gan.data import MAX_LEN, MIN_LEN, STANDARD_ALPHABET
from amp_gan.properties import aa_composition, hydrophobicity, net_charge

#: Propiedades escalares comparadas por distribución.
PROPERTY_NAMES: tuple[str, ...] = ("length", "net_charge", "hydrophobicity")

#: Nombres de las métricas (claves de :func:`evaluate_sequences`), en orden.
METRIC_NAMES: tuple[str, ...] = (
    "n",
    "n_valid",
    "valid_fraction",
    "unique_fraction",
    "novel_fraction",
    *(f"{prop}_{stat}" for prop in PROPERTY_NAMES for stat in ("mean", "std", "wasserstein")),
    "composition_tv",
)

_ALPHABET = frozenset(STANDARD_ALPHABET)


def is_valid(seq: object, min_len: int = MIN_LEN, max_len: int = MAX_LEN) -> bool:
    """Indica si ``seq`` es una cadena válida.

    Válida significa: ``str`` no vacía, solo con los 20 aminoácidos estándar
    (mayúsculas) y con longitud en ``[min_len, max_len]`` (inclusiva).
    """
    return (
        isinstance(seq, str)
        and len(seq) > 0
        and min_len <= len(seq) <= max_len
        and set(seq) <= _ALPHABET
    )


@dataclass
class SequenceProfile:
    """Resumen de un conjunto de secuencias válidas (propiedades y composición)."""

    sequences: list[str]
    length: np.ndarray
    net_charge: np.ndarray
    hydrophobicity: np.ndarray
    composition: np.ndarray  # composición media, forma (20,), orden de STANDARD_ALPHABET

    @property
    def n(self) -> int:
        return len(self.sequences)

    def values(self, prop: str) -> np.ndarray:
        """Valores de la propiedad ``prop`` (uno por secuencia)."""
        return getattr(self, prop)


def profile_sequences(sequences: Sequence[str]) -> SequenceProfile:
    """Calcula propiedades y composición media de secuencias YA válidas.

    Con una lista vacía devuelve arrays vacíos y composición ``NaN``.
    """
    seqs = list(sequences)
    if not seqs:
        empty = np.zeros(0, dtype="float64")
        return SequenceProfile([], empty, empty, empty, np.full(len(STANDARD_ALPHABET), np.nan))
    comp = np.array([list(aa_composition(s).values()) for s in seqs], dtype="float64")
    return SequenceProfile(
        sequences=seqs,
        length=np.array([len(s) for s in seqs], dtype="float64"),
        net_charge=np.array([net_charge(s) for s in seqs], dtype="float64"),
        hydrophobicity=np.array([hydrophobicity(s) for s in seqs], dtype="float64"),
        composition=comp.mean(axis=0),
    )


@dataclass
class ReferenceProfile:
    """Referencia ya procesada: perfil de sus secuencias válidas y su conjunto."""

    profile: SequenceProfile
    sequence_set: frozenset[str]


def _as_sequence_list(data: object) -> list[str]:
    """Convierte DataFrame (columna ``sequence``), Series o iterable en lista."""
    if isinstance(data, pd.DataFrame):
        return data["sequence"].tolist()
    if isinstance(data, pd.Series):
        return data.tolist()
    return list(data)  # type: ignore[call-overload]


def build_reference(reference: pd.DataFrame | Sequence[str] | ReferenceProfile) -> ReferenceProfile:
    """Prepara la referencia (se puede reutilizar entre llamadas para ahorrar tiempo).

    Args:
        reference: DataFrame con columna ``sequence``, serie o lista de cadenas;
            las no válidas (ver :func:`is_valid`) se descartan.

    Raises:
        ValueError: si la referencia no tiene ninguna secuencia válida.
    """
    if isinstance(reference, ReferenceProfile):
        return reference
    valid = [s for s in _as_sequence_list(reference) if is_valid(s)]
    if not valid:
        raise ValueError("La referencia no contiene ninguna secuencia válida.")
    return ReferenceProfile(profile_sequences(valid), frozenset(valid))


def _nan_metrics(n: int) -> dict[str, float]:
    out: dict[str, float] = {name: float("nan") for name in METRIC_NAMES}
    out["n"] = n
    out["n_valid"] = 0
    return out


def evaluate_sequences(
    generated: Sequence[str],
    reference: pd.DataFrame | Sequence[str] | ReferenceProfile,
) -> dict[str, float]:
    """Evalúa secuencias generadas frente a una referencia real.

    Args:
        generated: lista de cadenas (pueden ser vacías o inválidas).
        reference: DataFrame con columna ``sequence``, lista/serie de cadenas o
            un :class:`ReferenceProfile` (ver :func:`build_reference`).

    Returns:
        Diccionario con las claves de :data:`METRIC_NAMES` (ver el docstring del
        módulo). Sin secuencias válidas las métricas dependientes son ``NaN``;
        con ``generated`` vacío, ``valid_fraction`` también.
    """
    ref = build_reference(reference)
    gen = list(generated)
    n = len(gen)
    valid = [s for s in gen if is_valid(s)]
    out = _nan_metrics(n)
    out["n_valid"] = len(valid)
    if n > 0:
        out["valid_fraction"] = len(valid) / n
    if not valid:
        return out

    out["unique_fraction"] = len(set(valid)) / len(valid)
    out["novel_fraction"] = sum(s not in ref.sequence_set for s in valid) / len(valid)

    prof = profile_sequences(valid)
    for prop in PROPERTY_NAMES:
        values = prof.values(prop)
        out[f"{prop}_mean"] = float(values.mean())
        out[f"{prop}_std"] = float(values.std(ddof=1)) if values.size > 1 else float("nan")
        out[f"{prop}_wasserstein"] = float(wasserstein_distance(values, ref.profile.values(prop)))
    out["composition_tv"] = float(0.5 * np.abs(prof.composition - ref.profile.composition).sum())
    return out


def random_uniform_sequences(n: int, seed: int) -> list[str]:
    """``n`` secuencias aleatorias: aa uniformes y longitud uniforme en ``[MIN_LEN, MAX_LEN]``."""
    rng = np.random.default_rng(seed)
    alphabet = np.array(list(STANDARD_ALPHABET))
    lengths = rng.integers(MIN_LEN, MAX_LEN + 1, size=n)
    return ["".join(rng.choice(alphabet, size=int(k))) for k in lengths]


def reference_floors(
    reference: pd.DataFrame | Sequence[str] | ReferenceProfile, n: int, seed: int
) -> dict[str, dict[str, float]]:
    """Métricas de dos referencias de comparación ("suelos").

    * ``"real_subset"``: submuestra aleatoria reproducible de ``n`` secuencias
      reales evaluada frente al RESTO de la referencia (suelo de ruido del
      muestreo). Las secuencias de la submuestra se excluyen del resto, así que
      no se solapan; por construcción su ``novel_fraction`` es 1.
    * ``"random_uniform"``: ``n`` secuencias con aminoácidos uniformes y
      longitudes uniformes en ``[5, 60]``, evaluadas frente a TODA la referencia.

    Args:
        reference: DataFrame con columna ``sequence`` o lista de cadenas
            (secuencias reales).
        n: tamaño de cada conjunto de comparación.
        seed: semilla (submuestra y secuencias aleatorias).

    Returns:
        ``{"real_subset": métricas, "random_uniform": métricas}``, cada una con
        las claves de :func:`evaluate_sequences`.

    Raises:
        ValueError: ver :func:`real_subset_split`.
    """
    full = build_reference(reference)
    subset, rest = real_subset_split(full, n, seed)
    return {
        "real_subset": evaluate_sequences(subset, rest),
        "random_uniform": evaluate_sequences(random_uniform_sequences(n, seed), full),
    }


def real_subset_split(
    reference: pd.DataFrame | Sequence[str] | ReferenceProfile, n: int, seed: int
) -> tuple[list[str], list[str]]:
    """Separa la referencia en ``(submuestra de n, resto)`` sin solapamiento.

    La submuestra es aleatoria y reproducible (``seed``); del resto se quitan
    todas las secuencias que estén en la submuestra.

    Raises:
        ValueError: si ``n`` no es positivo o no deja secuencias de resto
            (``n`` mayor o igual que el nº de secuencias reales válidas).
    """
    if n < 1:
        raise ValueError(f"n debe ser >= 1, se recibió {n}.")
    seqs = build_reference(reference).profile.sequences
    if n >= len(seqs):
        raise ValueError(f"n={n} debe ser menor que el nº de secuencias de referencia ({len(seqs)}).")
    idx = np.random.default_rng(seed).choice(len(seqs), size=n, replace=False)
    subset = [seqs[i] for i in idx]
    subset_set = set(subset)
    rest = [s for s in seqs if s not in subset_set]
    if not rest:
        raise ValueError("No quedan secuencias de resto tras separar la submuestra.")
    return subset, rest

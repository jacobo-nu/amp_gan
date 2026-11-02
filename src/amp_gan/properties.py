"""Propiedades fisicoquímicas de péptidos: composición, carga neta e hidrofobicidad.

Funciones puras, solo con librería estándar, numpy y pandas (sin Biopython).
Todas esperan secuencias no vacías formadas únicamente por los 20 aminoácidos
estándar (:data:`amp_gan.data.STANDARD_ALPHABET`) en mayúsculas; en otro caso
lanzan ``ValueError`` con un mensaje claro.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd

from amp_gan.data import STANDARD_ALPHABET

#: pKa usados para la carga neta (tabla de EMBOSS, programas ``iep``/``pepstats``;
#: Rice P., Longden I., Bleasby A. EMBOSS: The European Molecular Biology Open
#: Software Suite. Trends Genet. 2000; 16(6): 276-277).
#: Clave ``"N_term"``: grupo amino terminal; ``"C_term"``: carboxilo terminal.
PKA: dict[str, float] = {
    "N_term": 8.6,
    "C_term": 3.6,
    "K": 10.8,
    "R": 12.5,
    "H": 6.5,
    "D": 3.9,
    "E": 4.1,
    "C": 8.5,
    "Y": 10.1,
}

#: Grupos que aportan carga positiva cuando están protonados.
_POSITIVE_GROUPS: tuple[str, ...] = ("N_term", "K", "R", "H")

#: Grupos que aportan carga negativa cuando están desprotonados.
_NEGATIVE_GROUPS: tuple[str, ...] = ("C_term", "D", "E", "C", "Y")

#: Escala de hidropatía de Kyte y Doolittle (J. Mol. Biol. 1982; 157: 105-132).
KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def _validate(seq: str) -> None:
    """Comprueba que ``seq`` no está vacía y solo contiene aminoácidos estándar.

    Raises:
        ValueError: si la secuencia está vacía o tiene caracteres no estándar.
    """
    if not seq:
        raise ValueError("La secuencia está vacía.")
    invalid = sorted(set(seq) - set(STANDARD_ALPHABET))
    if invalid:
        raise ValueError(f"Caracteres no estándar en la secuencia: {invalid}.")


def aa_composition(seq: str) -> dict[str, float]:
    """Calcula la fracción de cada uno de los 20 aminoácidos en una secuencia.

    Args:
        seq: secuencia en mayúsculas, no vacía, con aminoácidos estándar.

    Returns:
        Diccionario con los 20 aminoácidos (orden de
        :data:`amp_gan.data.STANDARD_ALPHABET`) como claves y su fracción como
        valor; las fracciones suman 1.

    Raises:
        ValueError: si la secuencia está vacía o tiene caracteres no estándar.
    """
    _validate(seq)
    counts = Counter(seq)
    n = len(seq)
    return {aa: counts.get(aa, 0) / n for aa in STANDARD_ALPHABET}


def composition_table(sequences: Iterable[str]) -> pd.DataFrame:
    """Calcula la composición media de aminoácidos de un conjunto de secuencias.

    La media es la de las fracciones por secuencia (cada péptido pesa igual,
    independientemente de su longitud).

    Args:
        sequences: iterable de secuencias válidas (ver :func:`aa_composition`);
            no puede estar vacío.

    Returns:
        DataFrame con 20 filas (una por aminoácido, en el orden de
        :data:`amp_gan.data.STANDARD_ALPHABET`) y columnas ``amino_acid`` y
        ``mean_fraction``; ``mean_fraction`` suma 1.

    Raises:
        ValueError: si ``sequences`` está vacío o alguna secuencia no es válida.
    """
    fractions = np.array(
        [[aa_composition(s)[aa] for aa in STANDARD_ALPHABET] for s in sequences],
        dtype=np.float64,
    )
    if fractions.size == 0:
        raise ValueError("No se puede calcular la composición de un conjunto vacío.")
    return pd.DataFrame(
        {"amino_acid": list(STANDARD_ALPHABET), "mean_fraction": fractions.mean(axis=0)}
    )


def net_charge(seq: str, ph: float = 7.4) -> float:
    """Calcula la carga neta de un péptido a un pH dado (Henderson-Hasselbalch).

    Cada grupo ionizable aporta una carga fraccionaria: los básicos (amino
    terminal, K, R, H) ``+1 / (1 + 10**(pH - pKa))`` y los ácidos (carboxilo
    terminal, D, E, C, Y) ``-1 / (1 + 10**(pKa - pH))``. Los pKa son los de
    :data:`PKA` (tabla de EMBOSS, Rice et al., *Trends Genet.* 2000; 16: 276):
    N terminal 8.6, C terminal 3.6, K 10.8, R 12.5, H 6.5, D 3.9, E 4.1,
    C 8.5 e Y 10.1. Se ignoran los efectos del entorno sobre el pKa y las
    modificaciones postraduccionales (p. ej. amidación C-terminal).

    Args:
        seq: secuencia en mayúsculas, no vacía, con aminoácidos estándar.
        ph: pH de la disolución (por defecto 7.4).

    Returns:
        Carga neta (en unidades de carga elemental, no entera).

    Raises:
        ValueError: si la secuencia está vacía o tiene caracteres no estándar.
    """
    _validate(seq)
    counts = Counter(seq)

    def _n(group: str) -> int:
        return 1 if group in ("N_term", "C_term") else counts.get(group, 0)

    positive = sum(_n(g) / (1.0 + 10.0 ** (ph - PKA[g])) for g in _POSITIVE_GROUPS)
    negative = sum(_n(g) / (1.0 + 10.0 ** (PKA[g] - ph)) for g in _NEGATIVE_GROUPS)
    return float(positive - negative)


def hydrophobicity(seq: str) -> float:
    """Calcula la hidrofobicidad media de Kyte-Doolittle (GRAVY).

    Es la media aritmética de la escala :data:`KYTE_DOOLITTLE` sobre todos los
    residuos (valores altos: más hidrofóbico; poli-I = 4.5, poli-R = -4.5).

    Args:
        seq: secuencia en mayúsculas, no vacía, con aminoácidos estándar.

    Returns:
        Valor GRAVY de la secuencia.

    Raises:
        ValueError: si la secuencia está vacía o tiene caracteres no estándar.
    """
    _validate(seq)
    return float(sum(KYTE_DOOLITTLE[aa] for aa in seq) / len(seq))


def properties_table(df: pd.DataFrame, ph: float = 7.4) -> pd.DataFrame:
    """Calcula las propiedades de todas las secuencias de un DataFrame.

    Args:
        df: DataFrame con columnas ``id`` y ``sequence``.
        ph: pH para la carga neta (por defecto 7.4).

    Returns:
        DataFrame con columnas ``id``, ``sequence``, ``length``, ``net_charge``
        e ``hydrophobicity``, en el mismo orden que ``df`` y con índice
        reiniciado.

    Raises:
        ValueError: si alguna secuencia es vacía o no válida.
    """
    sequences = df["sequence"].tolist()
    return pd.DataFrame(
        {
            "id": df["id"].tolist(),
            "sequence": sequences,
            "length": [len(s) for s in sequences],
            "net_charge": [net_charge(s, ph) for s in sequences],
            "hydrophobicity": [hydrophobicity(s) for s in sequences],
        }
    )

"""Lectura y limpieza de secuencias FASTA de péptidos antimicrobianos (AMP).

El preprocesado es idéntico para cualquier base de datos: cada archivo se limpia
por separado, sin fusionarlo con otros, para que las comparaciones posteriores
se hagan en condiciones equivalentes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

#: Los 20 aminoácidos estándar (código de una letra).
STANDARD_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWY"

#: Longitud mínima (inclusiva, en aa) de las secuencias de entrenamiento.
MIN_LEN: int = 5

#: Longitud máxima (inclusiva, en aa) de las secuencias de entrenamiento.
MAX_LEN: int = 60


def read_fasta(path: str | Path) -> pd.DataFrame:
    """Lee un archivo FASTA y devuelve un DataFrame con columnas ``id`` y ``sequence``.

    Admite secuencias repartidas en varias líneas (se concatenan), ignora las
    líneas vacías y normaliza las secuencias a mayúsculas. El identificador es
    el primer token de la cabecera (lo que sigue a ``>`` hasta el primer espacio).
    El orden de las secuencias es el del archivo.

    Args:
        path: ruta del archivo FASTA.

    Returns:
        DataFrame con columnas ``id`` y ``sequence`` (ambas de tipo ``str``).

    Raises:
        ValueError: si hay líneas de secuencia antes de la primera cabecera.
    """
    ids: list[str] = []
    sequences: list[str] = []
    current_parts: list[str] | None = None

    def _flush() -> None:
        if current_parts is not None:
            sequences.append("".join(current_parts).upper())

    with open(path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                header = line[1:].strip()
                ids.append(header.split()[0] if header else "")
                current_parts = []
            else:
                if current_parts is None:
                    raise ValueError(
                        f"Línea {line_number}: secuencia antes de la primera cabecera."
                    )
                current_parts.append("".join(line.split()))
        _flush()

    return pd.DataFrame({"id": ids, "sequence": sequences}, dtype=object)


def write_fasta(df: pd.DataFrame, path: str | Path) -> None:
    """Escribe un DataFrame con columnas ``id`` y ``sequence`` como FASTA.

    Cada registro ocupa dos líneas: ``>id`` y la secuencia completa en una sola
    línea (mismo formato que los archivos de entrada). Se usa ``\\n`` como fin
    de línea y se respeta el orden del DataFrame, por lo que la salida es
    determinista.

    Args:
        df: DataFrame con columnas ``id`` y ``sequence``.
        path: ruta del archivo de salida (se sobrescribe).
    """
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for seq_id, seq in zip(df["id"], df["sequence"]):
            handle.write(f">{seq_id}\n{seq}\n")


def clean_sequences(
    df: pd.DataFrame,
    min_len: int = MIN_LEN,
    max_len: int = MAX_LEN,
    alphabet: str | set[str] | frozenset[str] = STANDARD_ALPHABET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Limpia un DataFrame de secuencias aplicando tres criterios en orden.

    1. Elimina las secuencias con algún carácter fuera de ``alphabet``.
    2. Elimina las que tienen longitud fuera de ``[min_len, max_len]`` (ambos
       límites inclusivos).
    3. Elimina duplicados exactos de secuencia, conservando la primera aparición.

    Cada secuencia se contabiliza solo en el primer criterio que incumple. El
    resultado es determinista y conserva el orden original; el índice se
    reinicia.

    Args:
        df: DataFrame con al menos la columna ``sequence``.
        min_len: longitud mínima permitida (inclusiva); por defecto :data:`MIN_LEN` (5).
        max_len: longitud máxima permitida (inclusiva); por defecto :data:`MAX_LEN` (60).
        alphabet: caracteres permitidos (por defecto, los 20 aminoácidos estándar).

    Returns:
        Tupla ``(df_limpio, stats)``. ``stats`` contiene las claves ``n_input``,
        ``removed_non_standard``, ``removed_length``, ``removed_duplicates`` y
        ``n_output``, que cumplen ``n_input = sum(removed_*) + n_output``.

    Raises:
        ValueError: si ``min_len > max_len``.
    """
    if min_len > max_len:
        raise ValueError(f"min_len ({min_len}) no puede ser mayor que max_len ({max_len}).")

    allowed = frozenset(alphabet)
    n_input = len(df)

    # 1) Caracteres no estándar (una secuencia vacía no los tiene: la filtra la longitud).
    is_standard = df["sequence"].map(lambda seq: set(seq) <= allowed).astype(bool)
    df = df.loc[is_standard]
    removed_non_standard = n_input - len(df)

    # 2) Longitud, con límites inclusivos.
    lengths = df["sequence"].map(len).astype(int)
    n_before = len(df)
    df = df[(lengths >= min_len) & (lengths <= max_len)]
    removed_length = n_before - len(df)

    # 3) Duplicados exactos, conservando la primera aparición.
    n_before = len(df)
    df = df.drop_duplicates(subset="sequence", keep="first")
    removed_duplicates = n_before - len(df)

    df = df.reset_index(drop=True)
    stats: dict[str, Any] = {
        "n_input": n_input,
        "removed_non_standard": removed_non_standard,
        "removed_length": removed_length,
        "removed_duplicates": removed_duplicates,
        "n_output": len(df),
    }
    return df, stats


def load_clean(path: str | Path, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lee un FASTA y lo limpia con :func:`clean_sequences`.

    Args:
        path: ruta del archivo FASTA.
        **kwargs: argumentos para :func:`clean_sequences` (``min_len``,
            ``max_len``, ``alphabet``).

    Returns:
        Tupla ``(df_limpio, stats)``, como en :func:`clean_sequences`.
    """
    return clean_sequences(read_fasta(path), **kwargs)

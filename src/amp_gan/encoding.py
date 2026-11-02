"""Codificación one-hot de secuencias de péptidos con relleno (padding).

Cada secuencia se representa como una matriz ``(seq_len, 21)`` de ``float32``:
los 21 canales son los 20 aminoácidos estándar (orden alfabético) más un token
de relleno ``PAD`` que se trata como un símbolo más (sin máscara). Las
secuencias más cortas que ``seq_len`` se rellenan con ``PAD`` a la derecha.

La salida es directamente utilizable como entrada/salida de modelos Keras
(``(n, seq_len, 21)``, ``float32``). Este módulo solo depende de numpy y pandas.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from amp_gan.data import MAX_LEN, STANDARD_ALPHABET, read_fasta

#: Símbolo del token de relleno.
PAD_TOKEN: str = "_"

#: Vocabulario: los 20 aa estándar en orden alfabético y ``PAD_TOKEN`` al final.
VOCAB: tuple[str, ...] = (*sorted(STANDARD_ALPHABET), PAD_TOKEN)

#: Número de canales de la codificación (20 aa + PAD).
VOCAB_SIZE: int = len(VOCAB)

#: Índice (canal) del token de relleno.
PAD_INDEX: int = VOCAB.index(PAD_TOKEN)

#: Longitud fija de las secuencias codificadas (reutiliza :data:`amp_gan.data.MAX_LEN`).
SEQ_LEN: int = MAX_LEN

#: Símbolo -> índice de canal.
AA_TO_INDEX: dict[str, int] = {symbol: i for i, symbol in enumerate(VOCAB)}

#: Índice de canal -> símbolo.
INDEX_TO_AA: dict[int, str] = {i: symbol for i, symbol in enumerate(VOCAB)}

_AMINO_ACIDS: frozenset[str] = frozenset(STANDARD_ALPHABET)


def _check_seq_len(seq_len: int) -> None:
    if seq_len < 1:
        raise ValueError(f"seq_len debe ser >= 1, se recibió {seq_len}.")


def _sequence_to_indices(seq: str, seq_len: int) -> np.ndarray:
    """Convierte una secuencia en un vector de índices ``(seq_len,)`` con relleno PAD."""
    if not isinstance(seq, str):
        raise TypeError(f"La secuencia debe ser str, se recibió {type(seq).__name__}.")
    if len(seq) == 0:
        raise ValueError("La secuencia está vacía.")
    if len(seq) > seq_len:
        raise ValueError(f"Secuencia demasiado larga: {len(seq)} > seq_len={seq_len}.")
    invalid = sorted(set(seq) - _AMINO_ACIDS)
    if invalid:
        raise ValueError(
            f"Caracteres no válidos {invalid} en la secuencia {seq!r}: solo se admiten "
            f"los 20 aminoácidos estándar en mayúsculas (el relleno '{PAD_TOKEN}' no "
            "puede aparecer en una secuencia de entrada)."
        )
    indices = np.full(seq_len, PAD_INDEX, dtype=np.int64)
    indices[: len(seq)] = [AA_TO_INDEX[c] for c in seq]
    return indices


def _indices_to_one_hot(indices: np.ndarray) -> np.ndarray:
    """Convierte índices ``(n, seq_len)`` en one-hot ``(n, seq_len, 21)`` float32."""
    n, seq_len = indices.shape
    out = np.zeros((n, seq_len, VOCAB_SIZE), dtype=np.float32)
    out[np.arange(n)[:, None], np.arange(seq_len)[None, :], indices] = 1.0
    return out


def encode_sequence(seq: str, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Codifica una secuencia en one-hot con relleno PAD a la derecha.

    Args:
        seq: secuencia de aminoácidos estándar (mayúsculas).
        seq_len: longitud fija de la salida.

    Returns:
        Array ``float32`` de forma ``(seq_len, 21)``.

    Raises:
        ValueError: si la secuencia está vacía, contiene caracteres fuera de los
            20 aa estándar (incluido ``_``) o es más larga que ``seq_len``.
    """
    _check_seq_len(seq_len)
    return _indices_to_one_hot(_sequence_to_indices(seq, seq_len)[None, :])[0]


def encode_sequences(
    sequences: Sequence[str] | pd.Series, seq_len: int = SEQ_LEN
) -> np.ndarray:
    """Codifica un conjunto de secuencias en un tensor one-hot.

    Args:
        sequences: lista (o cualquier secuencia) de cadenas, o ``pd.Series``.
        seq_len: longitud fija de cada secuencia codificada.

    Returns:
        Array ``float32`` de forma ``(n, seq_len, 21)``; con ``n = 0`` devuelve
        un array vacío con esa forma.

    Raises:
        ValueError: en los mismos casos que :func:`encode_sequence`.
    """
    _check_seq_len(seq_len)
    seq_list = sequences.tolist() if isinstance(sequences, pd.Series) else list(sequences)
    if not seq_list:
        return np.zeros((0, seq_len, VOCAB_SIZE), dtype=np.float32)
    indices = np.stack([_sequence_to_indices(s, seq_len) for s in seq_list])
    return _indices_to_one_hot(indices)


def _decoded_lengths_and_indices(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve ``(argmax, longitudes)`` para un lote ``(n, seq_len, 21)``."""
    arr = np.asarray(batch)
    if arr.ndim != 3 or arr.shape[-1] != VOCAB_SIZE:
        raise ValueError(
            f"Se esperaba un lote de forma (n, seq_len, {VOCAB_SIZE}), se recibió {arr.shape}."
        )
    indices = arr.argmax(axis=-1)
    is_pad = indices == PAD_INDEX
    lengths = np.where(is_pad.any(axis=1), is_pad.argmax(axis=1), arr.shape[1])
    return indices, lengths.astype(np.int64)


def decode_one_hot(matrix: np.ndarray) -> str:
    """Decodifica una matriz ``(seq_len, 21)`` en una secuencia.

    Se aplica ``argmax`` por posición (vale para one-hot exacto y para salidas
    de softmax) y la secuencia se corta en el PRIMER ``PAD``: todo lo que haya
    después, aunque no sea PAD, se ignora. Si no hay ningún PAD se devuelve la
    secuencia completa; si la primera posición es PAD, la cadena vacía.

    Args:
        matrix: array de forma ``(seq_len, 21)``.

    Returns:
        Secuencia decodificada (sin relleno).

    Raises:
        ValueError: si la forma no es ``(seq_len, 21)``.
    """
    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[-1] != VOCAB_SIZE:
        raise ValueError(
            f"Se esperaba una matriz de forma (seq_len, {VOCAB_SIZE}), se recibió {arr.shape}."
        )
    return decode_batch(arr[None, ...])[0]


def decode_batch(batch: np.ndarray) -> list[str]:
    """Decodifica un lote ``(n, seq_len, 21)`` con el criterio de :func:`decode_one_hot`.

    Args:
        batch: array de forma ``(n, seq_len, 21)``.

    Returns:
        Lista de ``n`` secuencias decodificadas (cortadas en el primer PAD).

    Raises:
        ValueError: si la forma no es ``(n, seq_len, 21)``.
    """
    indices, lengths = _decoded_lengths_and_indices(batch)
    return [
        "".join(INDEX_TO_AA[i] for i in row[:length].tolist())
        for row, length in zip(indices, lengths)
    ]


def lengths_from_one_hot(batch: np.ndarray) -> np.ndarray:
    """Longitud decodificada de cada elemento: posiciones antes del primer PAD.

    Args:
        batch: array de forma ``(n, seq_len, 21)`` (one-hot o probabilidades).

    Returns:
        Array de enteros de forma ``(n,)``. Es ``seq_len`` si no hay PAD.

    Raises:
        ValueError: si la forma no es ``(n, seq_len, 21)``.
    """
    return _decoded_lengths_and_indices(batch)[1]


def load_encoded(
    path: str | Path, seq_len: int = SEQ_LEN
) -> tuple[np.ndarray, pd.DataFrame]:
    """Lee un FASTA y devuelve sus secuencias codificadas.

    No guarda nada en disco.

    Args:
        path: ruta del FASTA (ya limpio: solo aa estándar y longitud ``<= seq_len``).
        seq_len: longitud fija de la codificación.

    Returns:
        Tupla ``(X, df)``: ``X`` de forma ``(n, seq_len, 21)`` ``float32`` y
        ``df`` con columnas ``id`` y ``sequence`` en el mismo orden que ``X``.

    Raises:
        ValueError: si alguna secuencia no es codificable.
    """
    df = read_fasta(path)
    return encode_sequences(df["sequence"], seq_len=seq_len), df

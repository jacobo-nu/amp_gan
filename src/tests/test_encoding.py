"""Tests de la codificación one-hot (``amp_gan.encoding``)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amp_gan.data import MAX_LEN, STANDARD_ALPHABET
from amp_gan.encoding import (
    AA_TO_INDEX,
    INDEX_TO_AA,
    PAD_INDEX,
    PAD_TOKEN,
    SEQ_LEN,
    VOCAB,
    VOCAB_SIZE,
    decode_batch,
    decode_one_hot,
    encode_sequence,
    encode_sequences,
    lengths_from_one_hot,
    load_encoded,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLEAN_DIR = _REPO_ROOT / "context" / "data" / "processed"
_REAL_FASTAS = ["Antibacterial_amps_clean.fasta", "general_amps_clean.fasta"]


def _random_seq(rng: np.random.Generator, length: int) -> str:
    return "".join(rng.choice(list(STANDARD_ALPHABET), size=length))


def test_vocab_order_and_constants():
    assert VOCAB == (*"ACDEFGHIKLMNPQRSTVWY", "_")
    assert VOCAB[:20] == tuple(sorted(STANDARD_ALPHABET))
    assert VOCAB[-1] == PAD_TOKEN == "_"
    assert VOCAB_SIZE == 21 == len(VOCAB)
    assert PAD_INDEX == 20
    assert SEQ_LEN == MAX_LEN == 60
    assert all(INDEX_TO_AA[AA_TO_INDEX[s]] == s for s in VOCAB)
    assert len(AA_TO_INDEX) == len(INDEX_TO_AA) == 21


def test_shape_and_dtype():
    x = encode_sequence("ACDEF")
    assert x.shape == (60, 21)
    assert x.dtype == np.float32
    assert encode_sequence("ACD", seq_len=10).shape == (10, 21)
    batch = encode_sequences(["ACD", "KLMNPQ"])
    assert batch.shape == (2, 60, 21)
    assert batch.dtype == np.float32


def test_one_hot_and_padding():
    seq = "MKLV"
    x = encode_sequence(seq)
    assert np.all(x.sum(axis=-1) == 1)
    assert set(np.unique(x)) <= {0.0, 1.0}
    for i, aa in enumerate(seq):
        assert x[i].argmax() == AA_TO_INDEX[aa]
    assert np.all(x[len(seq):].argmax(axis=-1) == PAD_INDEX)
    assert np.all(x[len(seq):, PAD_INDEX] == 1)


@pytest.mark.parametrize("length", [5, 30, 60])
def test_roundtrip_lengths(length):
    rng = np.random.default_rng(42)
    seq = _random_seq(rng, length)
    assert decode_one_hot(encode_sequence(seq)) == seq


def test_roundtrip_all_amino_acids():
    assert decode_one_hot(encode_sequence(STANDARD_ALPHABET)) == STANDARD_ALPHABET
    for aa in STANDARD_ALPHABET:
        assert decode_one_hot(encode_sequence(aa)) == aa


def test_encode_sequences_matches_single_and_series():
    seqs = ["ACDEF", "W", "Y" * 60, "KLMNPQRST"]
    batch = encode_sequences(seqs)
    for i, s in enumerate(seqs):
        np.testing.assert_array_equal(batch[i], encode_sequence(s))
    np.testing.assert_array_equal(encode_sequences(pd.Series(seqs)), batch)
    assert decode_batch(batch) == seqs


@pytest.mark.parametrize(
    "seq",
    ["", "ACDXE", "acde", "AC_DE", "_", "AC DE", "ACD*", "B" * 10],
    ids=["empty", "nonstandard", "lowercase", "pad_inside", "only_pad", "space", "star", "B"],
)
def test_invalid_sequences_raise(seq):
    with pytest.raises(ValueError):
        encode_sequence(seq)
    with pytest.raises(ValueError):
        encode_sequences(["ACDE", seq])


def test_too_long_raises():
    with pytest.raises(ValueError, match="larga"):
        encode_sequence("A" * 61)
    with pytest.raises(ValueError):
        encode_sequence("AAAAAA", seq_len=5)
    assert encode_sequence("A" * 60).shape == (60, 21)


def test_invalid_seq_len_raises():
    with pytest.raises(ValueError):
        encode_sequence("A", seq_len=0)


def test_decode_softmax_with_noise():
    rng = np.random.default_rng(0)
    seqs = ["ACDEFGHIK", "WWYY", "M" * 60]
    batch = encode_sequences(seqs)
    probs = 0.9 * batch + 0.1 / VOCAB_SIZE
    probs = probs + rng.uniform(0, 0.02, size=probs.shape).astype(np.float32)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    assert decode_batch(probs) == seqs
    assert decode_one_hot(probs[0]) == seqs[0]
    np.testing.assert_array_equal(lengths_from_one_hot(probs), [9, 4, 60])


def test_decode_cuts_at_first_pad():
    x = np.zeros((6, 21), dtype=np.float32)
    for i, c in enumerate("AAA_AA"):
        x[i, AA_TO_INDEX[c]] = 1
    assert decode_one_hot(x) == "AAA"
    assert lengths_from_one_hot(x[None])[0] == 3


def test_decode_edge_cases():
    # PAD en la primera posición -> cadena vacía; sin PAD -> secuencia completa.
    all_pad = np.zeros((5, 21), dtype=np.float32)
    all_pad[:, PAD_INDEX] = 1
    assert decode_one_hot(all_pad) == ""
    assert lengths_from_one_hot(all_pad[None])[0] == 0
    full = encode_sequence("ACDEF", seq_len=5)
    assert decode_one_hot(full) == "ACDEF"
    assert lengths_from_one_hot(full[None])[0] == 5


def test_decode_bad_shape_raises():
    with pytest.raises(ValueError):
        decode_one_hot(np.zeros((60, 20)))
    with pytest.raises(ValueError):
        decode_batch(np.zeros((60, 21)))
    with pytest.raises(ValueError):
        lengths_from_one_hot(np.zeros((2, 60, 20)))


def test_lengths_consistent_with_len():
    rng = np.random.default_rng(1)
    seqs = [_random_seq(rng, int(n)) for n in rng.integers(1, 61, size=200)]
    lengths = lengths_from_one_hot(encode_sequences(seqs))
    assert np.issubdtype(lengths.dtype, np.integer)
    np.testing.assert_array_equal(lengths, [len(s) for s in seqs])


def test_empty_batch():
    x = encode_sequences([])
    assert x.shape == (0, 60, 21)
    assert x.dtype == np.float32
    assert encode_sequences(pd.Series([], dtype=object)).shape == (0, 60, 21)
    assert decode_batch(x) == []
    assert lengths_from_one_hot(x).shape == (0,)


def test_deterministic():
    seqs = ["ACDEF", "KLMNPQ"]
    np.testing.assert_array_equal(encode_sequences(seqs), encode_sequences(seqs))


def test_load_encoded_small_fasta(tmp_path):
    fasta = tmp_path / "toy.fasta"
    fasta.write_text(">a desc\nACDEF\n>b\nKLM\nNPQ\n", encoding="utf-8")
    x, df = load_encoded(fasta)
    assert list(df.columns) == ["id", "sequence"]
    assert df["sequence"].tolist() == ["ACDEF", "KLMNPQ"]
    assert x.shape == (2, 60, 21)
    assert decode_batch(x) == df["sequence"].tolist()


@pytest.mark.parametrize("name", _REAL_FASTAS)
def test_real_data_roundtrip(name):
    path = _CLEAN_DIR / name
    if not path.exists():
        pytest.skip(f"Datos no disponibles (no se versionan): {path.name}")
    x, df = load_encoded(path)
    n = len(df)
    assert n > 0
    assert x.shape == (n, SEQ_LEN, VOCAB_SIZE)
    assert x.dtype == np.float32
    assert np.all(x.sum(axis=-1) == 1)
    assert decode_batch(x) == df["sequence"].tolist()
    lengths = df["sequence"].str.len().to_numpy()
    np.testing.assert_array_equal(lengths_from_one_hot(x), lengths)
    # Nº de PAD = suma de (60 - longitud).
    assert int(x[..., PAD_INDEX].sum()) == int((SEQ_LEN - lengths).sum())

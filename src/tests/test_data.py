"""Tests de ``amp_gan.data`` con FASTA pequeños creados en ``tmp_path``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from amp_gan.data import (
    MAX_LEN,
    MIN_LEN,
    STANDARD_ALPHABET,
    clean_sequences,
    load_clean,
    read_fasta,
    write_fasta,
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "test.fasta"
    path.write_text(text, encoding="utf-8")
    return path


def _df(*sequences: str) -> pd.DataFrame:
    return pd.DataFrame({"id": [f"S{i}" for i in range(len(sequences))], "sequence": list(sequences)})


def test_read_fasta_basic(tmp_path: Path) -> None:
    df = read_fasta(_write(tmp_path, ">A\nACDEF\n>B\nGHIKL\n"))
    assert list(df.columns) == ["id", "sequence"]
    assert df["id"].tolist() == ["A", "B"]
    assert df["sequence"].tolist() == ["ACDEF", "GHIKL"]


def test_read_fasta_multiline(tmp_path: Path) -> None:
    df = read_fasta(_write(tmp_path, ">A desc\nACDEF\nGHIKL\n\nMNP\n>B\nQRSTV\n"))
    assert df["id"].tolist() == ["A", "B"]
    assert df["sequence"].tolist() == ["ACDEFGHIKLMNP", "QRSTV"]


def test_read_fasta_lowercase_normalized(tmp_path: Path) -> None:
    df = read_fasta(_write(tmp_path, ">A\nacdef\n>B\nGhIkL\n"))
    assert df["sequence"].tolist() == ["ACDEF", "GHIKL"]


def test_read_fasta_without_trailing_newline_and_crlf(tmp_path: Path) -> None:
    df = read_fasta(_write(tmp_path, ">A\r\nACDEF\r\n>B\r\nGHIKL"))
    assert df["sequence"].tolist() == ["ACDEF", "GHIKL"]


def test_read_fasta_sequence_before_header_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        read_fasta(_write(tmp_path, "ACDEF\n>A\nGHIKL\n"))


@pytest.mark.parametrize("bad", ["X", "B", "Z", "U", "O", "*", "-"])
def test_non_standard_characters_removed(bad: str) -> None:
    df, stats = clean_sequences(_df("ACDEFGHIK", f"ACDE{bad}GHIK"))
    assert df["sequence"].tolist() == ["ACDEFGHIK"]
    assert stats["removed_non_standard"] == 1


def test_length_limits_are_inclusive() -> None:
    seqs = ["A" * 4, "A" * 5, "C" * 50, "D" * 51]
    df, stats = clean_sequences(_df(*seqs), min_len=5, max_len=50)
    assert df["sequence"].tolist() == ["A" * 5, "C" * 50]
    assert stats["removed_length"] == 2


def test_custom_length_limits() -> None:
    df, _ = clean_sequences(_df("AAA", "CCCC", "DDDDD", "EEEEEE"), min_len=4, max_len=5)
    assert df["sequence"].tolist() == ["CCCC", "DDDDD"]


def test_invalid_length_range_raises() -> None:
    with pytest.raises(ValueError):
        clean_sequences(_df("ACDEF"), min_len=10, max_len=5)


def test_duplicates_keep_first() -> None:
    df = _df("ACDEF", "GHIKL", "ACDEF", "GHIKL", "MNPQR")
    out, stats = clean_sequences(df)
    assert out["sequence"].tolist() == ["ACDEF", "GHIKL", "MNPQR"]
    assert out["id"].tolist() == ["S0", "S1", "S4"]
    assert stats["removed_duplicates"] == 2


def test_lowercase_duplicates_collapse_after_reading(tmp_path: Path) -> None:
    df, stats = load_clean(_write(tmp_path, ">A\nacdef\n>B\nACDEF\n"))
    assert df["id"].tolist() == ["A"]
    assert stats["removed_duplicates"] == 1


def test_criteria_applied_in_order() -> None:
    # La secuencia con X y longitud fuera de rango cuenta solo como no estándar;
    # un duplicado de una secuencia filtrada por longitud no es duplicado.
    df = _df("AXA", "ACD", "ACD", "ACDEF", "ACDEF")
    out, stats = clean_sequences(df)
    assert stats["removed_non_standard"] == 1
    assert stats["removed_length"] == 2
    assert stats["removed_duplicates"] == 1
    assert out["sequence"].tolist() == ["ACDEF"]


def test_stats_are_coherent(tmp_path: Path) -> None:
    text = (
        ">1\nACDEFGHIK\n>2\nACDEFGHIK\n>3\nACXEF\n>4\nAC\n>5\n"
        + "A" * 61
        + "\n>6\nmnpqr\n>7\nQRSTVW\nYACDE\n>8\nBBBBB\n"
    )
    df, stats = load_clean(_write(tmp_path, text))
    assert set(stats) == {
        "n_input",
        "removed_non_standard",
        "removed_length",
        "removed_duplicates",
        "n_output",
    }
    assert stats["n_input"] == 8
    assert stats["removed_non_standard"] == 2
    assert stats["removed_length"] == 2
    assert stats["removed_duplicates"] == 1
    assert stats["n_output"] == len(df) == 3
    assert stats["n_input"] == (
        stats["removed_non_standard"]
        + stats["removed_length"]
        + stats["removed_duplicates"]
        + stats["n_output"]
    )


def test_output_only_standard_and_in_range() -> None:
    df, _ = clean_sequences(_df("ACDEF", "AXCDE", "A" * 61, "MKLVW"))
    alphabet = set(STANDARD_ALPHABET)
    assert all(set(s) <= alphabet for s in df["sequence"])
    assert df["sequence"].str.len().between(MIN_LEN, MAX_LEN).all()
    assert list(df.index) == list(range(len(df)))


def test_empty_input() -> None:
    df, stats = clean_sequences(pd.DataFrame({"id": [], "sequence": []}))
    assert len(df) == 0
    assert stats["n_input"] == stats["n_output"] == 0


def test_does_not_modify_input() -> None:
    original = _df("ACDEF", "ACDEF", "AXAXA")
    snapshot = original.copy()
    clean_sequences(original)
    pd.testing.assert_frame_equal(original, snapshot)


def test_load_clean_deterministic(tmp_path: Path) -> None:
    path = _write(tmp_path, ">A\nACDEF\n>B\nGHIKL\n>C\nACDEF\n")
    first, stats1 = load_clean(path)
    second, stats2 = load_clean(path)
    pd.testing.assert_frame_equal(first, second)
    assert stats1 == stats2


def test_default_length_range_is_5_60() -> None:
    assert (MIN_LEN, MAX_LEN) == (5, 60)
    seqs = ["A" * 4, "A" * MIN_LEN, "C" * 50, "D" * 60, "E" * 61]
    df, stats = clean_sequences(_df(*seqs))  # defaults
    assert df["sequence"].tolist() == ["A" * 5, "C" * 50, "D" * 60]
    assert stats["removed_length"] == 2


def test_write_fasta_roundtrip(tmp_path: Path) -> None:
    original = pd.DataFrame(
        {"id": ["X1", "X2", "X3"], "sequence": ["ACDEF", "G" * 60, "MKLVW"]}, dtype=object
    )
    path = tmp_path / "out.fasta"
    write_fasta(original, path)
    assert path.read_text(encoding="utf-8") == ">X1\nACDEF\n>X2\n" + "G" * 60 + "\n>X3\nMKLVW\n"
    pd.testing.assert_frame_equal(read_fasta(path), original)

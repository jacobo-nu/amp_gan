"""Tests de ``amp_gan.length_stats`` con datos pequeños construidos a mano."""

from __future__ import annotations

import numpy as np
import pytest

from amp_gan.length_stats import (
    DEFAULT_CUTOFFS,
    SUMMARY_KEYS,
    coverage_table,
    length_summary,
    sequence_lengths,
)


def test_sequence_lengths() -> None:
    assert sequence_lengths(["A", "ACD", "", "AC"]).tolist() == [1, 3, 0, 2]


def test_length_summary_known_values() -> None:
    s = length_summary(list(range(1, 101)))  # 1..100
    assert list(s) == list(SUMMARY_KEYS)
    assert s["n"] == 100
    assert s["mean"] == pytest.approx(50.5)
    assert s["std"] == pytest.approx(np.arange(1, 101).std(ddof=1))
    assert s["min"] == 1
    assert s["max"] == 100
    assert s["median"] == pytest.approx(50.5)
    assert s["p25"] == pytest.approx(25.75)
    assert s["p75"] == pytest.approx(75.25)
    assert s["p90"] == pytest.approx(90.1)
    assert s["p99"] == pytest.approx(99.01)
    assert s["mode"] == 1  # todas con frecuencia 1: gana la menor


def test_length_summary_mode_and_ties() -> None:
    assert length_summary([5, 7, 7, 9, 9, 3])["mode"] == 7  # empate 7 y 9 -> menor
    assert length_summary([4, 4, 4, 10])["mode"] == 4


def test_length_summary_single_value() -> None:
    s = length_summary([12])
    assert s["n"] == 1 and s["std"] == 0.0
    assert s["min"] == s["max"] == s["median"] == s["mode"] == 12


def test_length_summary_accepts_numpy_and_does_not_mutate() -> None:
    arr = np.array([3, 1, 2])
    s = length_summary(arr)
    assert arr.tolist() == [3, 1, 2]
    assert s["median"] == 2


def test_length_summary_empty_raises() -> None:
    with pytest.raises(ValueError):
        length_summary([])


def test_coverage_table_inclusive_cutoffs() -> None:
    table = coverage_table([10, 20, 20, 25, 50, 51, 120], cutoffs=[20, 50, 100])
    assert table["cutoff"].tolist() == [20, 50, 100]
    assert table["n_le"].tolist() == [3, 5, 6]  # <= inclusivo
    assert table["n_total"].tolist() == [7, 7, 7]
    assert table["fraction_le"].tolist() == pytest.approx([3 / 7, 5 / 7, 6 / 7])


def test_coverage_table_default_cutoffs_monotonic() -> None:
    table = coverage_table(list(range(1, 201)))
    assert table["cutoff"].tolist() == list(DEFAULT_CUTOFFS)
    assert table["n_le"].is_monotonic_increasing
    assert table["n_le"].tolist() == list(DEFAULT_CUTOFFS)  # 1..200: n_le == corte
    assert table["fraction_le"].between(0, 1).all()


def test_coverage_table_empty_raises() -> None:
    with pytest.raises(ValueError):
        coverage_table([])

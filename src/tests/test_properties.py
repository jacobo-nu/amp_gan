"""Tests de ``amp_gan.properties``."""

import math

import pandas as pd
import pytest

from amp_gan.data import STANDARD_ALPHABET
from amp_gan.properties import (
    aa_composition,
    composition_table,
    hydrophobicity,
    net_charge,
    properties_table,
)


def test_composition_sums_to_one_and_values():
    comp = aa_composition("AAGK")
    assert list(comp) == list(STANDARD_ALPHABET)
    assert math.isclose(sum(comp.values()), 1.0)
    assert comp["A"] == 0.5 and comp["G"] == 0.25 and comp["K"] == 0.25
    assert comp["W"] == 0.0


def test_composition_table_mean_per_sequence():
    table = composition_table(["AA", "GGGG"])
    assert list(table.columns) == ["amino_acid", "mean_fraction"]
    assert len(table) == 20
    frac = dict(zip(table["amino_acid"], table["mean_fraction"]))
    assert frac["A"] == pytest.approx(0.5)  # media de 1.0 y 0.0, no pooled
    assert frac["G"] == pytest.approx(0.5)
    assert table["mean_fraction"].sum() == pytest.approx(1.0)


def test_polyK_strongly_positive_polyD_strongly_negative():
    assert net_charge("K" * 10) > 9.0
    assert net_charge("D" * 10) < -9.0


def test_net_charge_single_residue_by_hand():
    # Gly-Gly: solo extremos. +1/(1+10^(7.4-8.6)) - 1/(1+10^(3.6-7.4))
    expected = 1 / (1 + 10 ** (7.4 - 8.6)) - 1 / (1 + 10 ** (3.6 - 7.4))
    assert net_charge("GG") == pytest.approx(expected)
    # Con K se suma 1/(1+10^(7.4-10.8))
    expected_k = expected + 1 / (1 + 10 ** (7.4 - 10.8))
    assert net_charge("GK") == pytest.approx(expected_k)


def test_net_charge_ph_extremes():
    assert net_charge("KKDD", ph=1.0) > 2.5  # casi todo protonado
    assert net_charge("KKDD", ph=14.0) < -2.0  # casi todo desprotonado
    # pH muy ácido: positivos completos (N-term + 2 K + R = 4), negativos ~0
    assert net_charge("KKR", ph=0.0) == pytest.approx(4.0, abs=1e-3)


def test_net_charge_monotonic_decreasing_with_ph():
    seq = "KRHDECYAGK"
    charges = [net_charge(seq, ph) for ph in (2, 4, 6, 8, 10, 12)]
    assert all(a > b for a, b in zip(charges, charges[1:]))


def test_hydrophobicity_extremes_and_mean():
    assert hydrophobicity("I" * 7) == pytest.approx(4.5)
    assert hydrophobicity("R" * 7) == pytest.approx(-4.5)
    assert hydrophobicity("IR") == pytest.approx(0.0)
    assert hydrophobicity("AG") == pytest.approx((1.8 - 0.4) / 2)


@pytest.mark.parametrize("func", [aa_composition, net_charge, hydrophobicity])
def test_empty_sequence_raises(func):
    with pytest.raises(ValueError, match="vacía"):
        func("")


@pytest.mark.parametrize("func", [aa_composition, net_charge, hydrophobicity])
def test_non_standard_raises(func):
    with pytest.raises(ValueError, match="no estándar"):
        func("AXG")


def test_composition_table_empty_raises():
    with pytest.raises(ValueError):
        composition_table([])


def test_properties_table_columns_and_values():
    df = pd.DataFrame({"id": ["a", "b"], "sequence": ["KKKKK", "IIIII"]})
    out = properties_table(df)
    assert list(out.columns) == ["id", "sequence", "length", "net_charge", "hydrophobicity"]
    assert out["id"].tolist() == ["a", "b"]
    assert out["length"].tolist() == [5, 5]
    assert out.loc[0, "net_charge"] > 4.0
    assert out.loc[0, "hydrophobicity"] == pytest.approx(-3.9)
    assert out.loc[1, "hydrophobicity"] == pytest.approx(4.5)
    assert out.loc[0, "net_charge"] == pytest.approx(net_charge("KKKKK"))

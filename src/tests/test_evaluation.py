"""Tests de ``amp_gan.evaluation`` con casos construidos a mano (sin Keras)."""

import math

import numpy as np
import pandas as pd
import pytest

from amp_gan.data import MAX_LEN, MIN_LEN, STANDARD_ALPHABET
from amp_gan.evaluation import (
    METRIC_NAMES,
    evaluate_sequences,
    is_valid,
    random_uniform_sequences,
    real_subset_split,
    reference_floors,
)

REF = ["KKLLKKLLKK", "GIGKFLHSAK", "AAAAAWWWWW", "RRRRRIIIII", "DEDEDEDEDE", "FLPIIAGIAK"]


def _rand_ref(n=300, seed=0):
    rng = np.random.default_rng(seed)
    alphabet = list(STANDARD_ALPHABET)
    seqs = {"".join(rng.choice(alphabet, size=int(rng.integers(5, 40)))) for _ in range(n)}
    return sorted(seqs)


def test_all_metric_keys_present():
    m = evaluate_sequences(REF, REF)
    assert list(m) == list(METRIC_NAMES)


def test_generated_equals_reference_gives_zero_distances():
    m = evaluate_sequences(REF, pd.DataFrame({"id": range(len(REF)), "sequence": REF}))
    assert m["n"] == m["n_valid"] == len(REF)
    assert m["valid_fraction"] == 1.0
    assert m["unique_fraction"] == 1.0
    assert m["novel_fraction"] == 0.0
    for prop in ("length", "net_charge", "hydrophobicity"):
        assert m[f"{prop}_wasserstein"] == pytest.approx(0.0, abs=1e-12)
    assert m["composition_tv"] == pytest.approx(0.0, abs=1e-12)


def test_means_and_std_by_hand():
    m = evaluate_sequences(["AAAAA", "AAAAAAAAAA"], REF)  # longitudes 5 y 10
    assert m["length_mean"] == pytest.approx(7.5)
    assert m["length_std"] == pytest.approx(np.std([5, 10], ddof=1))
    # hidrofobicidad de poli-A = 1,8 (Kyte-Doolittle)
    assert m["hydrophobicity_mean"] == pytest.approx(1.8)
    assert m["hydrophobicity_std"] == pytest.approx(0.0)


def test_wasserstein_matches_shift_of_lengths():
    ref = ["A" * k for k in (10, 20, 30)]
    gen = ["A" * k for k in (15, 25, 35)]  # desplazamiento de 5 aa
    assert evaluate_sequences(gen, ref)["length_wasserstein"] == pytest.approx(5.0)


def test_composition_tv_disjoint_alphabets_is_one():
    m = evaluate_sequences(["AAAAA", "AAAAAA"], ["WWWWW", "WWWWWW"])
    assert m["composition_tv"] == pytest.approx(1.0)


def test_invalid_sequences_reduce_valid_fraction():
    gen = ["KKLLKKLLKK", "", "KKX", "AAAA", "A" * (MAX_LEN + 1), "klklklkl", "GIGKFLHSAK", "FLPIIAGIAK"]
    m = evaluate_sequences(gen, REF)
    # válidas: las 3 correctas; ('KKX' no estándar, 'AAAA' corta, larga, minúsculas, vacía)
    assert m["n"] == 8 and m["n_valid"] == 3
    assert m["valid_fraction"] == pytest.approx(3 / 8)
    assert m["unique_fraction"] == 1.0  # el denominador son las válidas


def test_is_valid_limits_are_inclusive():
    assert is_valid("A" * MIN_LEN) and is_valid("A" * MAX_LEN)
    assert not is_valid("A" * (MIN_LEN - 1)) and not is_valid("A" * (MAX_LEN + 1))
    assert not is_valid("") and not is_valid("AAAAB") and not is_valid(None)


def test_duplicates_reduce_unique_fraction():
    gen = ["KKLLKKLLKK"] * 3 + ["GIGKFLHSAK"]
    m = evaluate_sequences(gen, REF)
    assert m["unique_fraction"] == pytest.approx(2 / 4)


def test_novel_fraction():
    gen = ["KKLLKKLLKK", "WWWWWWWWWW", "WWWWWWWWWW", "GGGGGGGGGG"]  # 1 real, 3 nuevas
    m = evaluate_sequences(gen, REF)
    assert m["novel_fraction"] == pytest.approx(3 / 4)


@pytest.mark.parametrize("gen", [[], [""], ["XX", "", "AAA"]])
def test_no_valid_sequences_gives_nan_without_exception(gen):
    m = evaluate_sequences(gen, REF)
    assert m["n"] == len(gen) and m["n_valid"] == 0
    for name in METRIC_NAMES:
        if name not in ("n", "n_valid"):
            if name == "valid_fraction" and not gen:
                assert math.isnan(m[name])
            elif name == "valid_fraction":
                assert m[name] == 0.0
            else:
                assert math.isnan(m[name]), name


def test_single_valid_sequence_has_nan_std_only():
    m = evaluate_sequences(["KKLLKKLLKK"], REF)
    assert math.isnan(m["length_std"])
    assert not math.isnan(m["length_mean"])


def test_empty_reference_raises():
    with pytest.raises(ValueError, match="referencia"):
        evaluate_sequences(["AAAAA"], ["XX", ""])


# --- suelos ---------------------------------------------------------------------------
def test_reference_floors_keys_and_shapes():
    ref = pd.DataFrame({"id": range(300), "sequence": _rand_ref(300)})
    floors = reference_floors(ref, n=50, seed=1)
    assert set(floors) == {"real_subset", "random_uniform"}
    for m in floors.values():
        assert list(m) == list(METRIC_NAMES)
        assert m["n"] == 50
    assert floors["real_subset"]["valid_fraction"] == 1.0
    assert floors["random_uniform"]["valid_fraction"] == 1.0


def test_real_subset_does_not_overlap_rest_and_is_reproducible():
    seqs = _rand_ref(300)
    sub1, rest1 = real_subset_split(seqs, n=60, seed=7)
    sub2, rest2 = real_subset_split(seqs, n=60, seed=7)
    sub3, _ = real_subset_split(seqs, n=60, seed=8)
    assert sub1 == sub2 and rest1 == rest2
    assert sub1 != sub3
    assert len(sub1) == 60 and not set(sub1) & set(rest1)
    assert set(sub1) | set(rest1) == set(seqs)


def test_real_subset_with_duplicated_reference_still_disjoint():
    seqs = _rand_ref(100) * 2  # duplicados
    sub, rest = real_subset_split(seqs, n=20, seed=0)
    assert not set(sub) & set(rest)
    assert reference_floors(seqs, 20, 0)["real_subset"]["novel_fraction"] == 1.0


def test_random_uniform_sequences_ranges_and_reproducibility():
    seqs = random_uniform_sequences(500, seed=3)
    assert seqs == random_uniform_sequences(500, seed=3)
    lengths = [len(s) for s in seqs]
    assert min(lengths) >= 5 and max(lengths) <= 60
    assert set("".join(seqs)) <= set(STANDARD_ALPHABET)
    assert all(is_valid(s) for s in seqs)


def test_floor_real_subset_is_closer_than_random_to_real_data():
    # Referencia con composición sesgada (solo K, L, A): el azar uniforme debe quedar más lejos.
    rng = np.random.default_rng(0)
    ref = ["".join(rng.choice(list("KLA"), size=int(rng.integers(8, 20)))) for _ in range(1000)]
    ref = list(dict.fromkeys(ref))
    assert len(ref) > 300
    floors = reference_floors(ref, n=100, seed=0)
    assert floors["real_subset"]["composition_tv"] < floors["random_uniform"]["composition_tv"]
    assert floors["real_subset"]["length_wasserstein"] < floors["random_uniform"]["length_wasserstein"]


def test_floors_invalid_n():
    seqs = _rand_ref(50)
    with pytest.raises(ValueError):
        reference_floors(seqs, n=0, seed=0)
    with pytest.raises(ValueError):
        reference_floors(seqs, n=len(seqs), seed=0)

"""Tests de la GAN condicional por hidrofobicidad (GRAVY) y de la evaluación por objetivos generalizada.

La primera parte no necesita Keras (``amp_gan.conditions``); la segunda usa el backend por
defecto del proceso (CPU) y la tercera lanza el entrenamiento y el script en un subproceso por
backend (TensorFlow y torch), saltándolos si el backend no está instalado.
"""

import json
import os
import subprocess
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amp_gan import run_conditional_gan, runtime
from amp_gan.conditions import (
    HIT_TOLERANCE,
    HYDRO_HIT_TOLERANCE,
    HYDRO_TARGET_COLUMNS,
    TARGET_COLUMNS,
    TARGET_PERCENTILES,
    build_target_references,
    control_slope,
    default_tolerance,
    evaluate_target,
    hit_rate,
    length_control_slope,
    property_values,
    raw_conditions_from_sequences,
    real_ceiling,
    real_subset_around,
    target_columns,
    target_key,
    target_label,
    target_lengths,
    target_values,
    unconditional_target_table,
    usual_range,
)
from amp_gan.data import STANDARD_ALPHABET, read_fasta
from amp_gan.evaluation import METRIC_NAMES, is_valid
from amp_gan.properties import KYTE_DOOLITTLE, hydrophobicity

runtime.use_cpu_only()  # como en test_conditional_gan.py: antes de importar Keras en este proceso

_SRC = Path(__file__).resolve().parents[1]
_PROCESSED = _SRC.parent / "context" / "data" / "processed"
_BACKENDS = ("tensorflow", "torch")
_BATCH = 8

HYDROPHOBIC = "I" * 20  # GRAVY 4.5
HYDROPHILIC = "R" * 20  # GRAVY -4.5
NEUTRAL = "G" * 10 + "S" * 10  # GRAVY (-0.4 - 0.8) / 2 = -0.6


def _random_sequences(n: int, seed: int = 0, lo: int = 5, hi: int = 30) -> list[str]:
    rng = np.random.default_rng(seed)
    alphabet = np.array(sorted(STANDARD_ALPHABET))
    return ["".join(rng.choice(alphabet, size=int(rng.integers(lo, hi + 1)))) for _ in range(n)]


# --------------------------------------------------------------------------- #
# GRAVY de las secuencias sintéticas, percentiles y objetivos
# --------------------------------------------------------------------------- #
def test_synthetic_sequences_have_the_expected_gravy():
    assert hydrophobicity(HYDROPHOBIC) == pytest.approx(4.5)
    assert hydrophobicity(HYDROPHILIC) == pytest.approx(-4.5)
    assert hydrophobicity(NEUTRAL) == pytest.approx(-0.6)
    np.testing.assert_allclose(property_values([HYDROPHOBIC, HYDROPHILIC, NEUTRAL], "hydrophobicity"), [4.5, -4.5, -0.6])
    assert property_values(["AAAAA", "A"], "length").tolist() == [5.0, 1.0]
    v = property_values(["", "AAAAA", "AXA", "KKKKK"], "hydrophobicity")  # vacía y no estándar -> NaN
    assert np.isnan(v[0]) and np.isnan(v[2]) and v[1] == pytest.approx(1.8) and v[3] == pytest.approx(-3.9)


def test_gravy_percentiles_and_targets_come_from_the_data_unrounded():
    gravy = np.arange(1, 101) / 10.0  # 0,1 ... 10,0
    t = target_values(gravy, "hydrophobicity")
    assert len(t) == len(TARGET_PERCENTILES) == 5
    np.testing.assert_allclose(t, np.percentile(gravy, [10, 30, 50, 70, 90]))
    assert t[2] == pytest.approx(5.05) and t[0] == pytest.approx(1.09)  # sin redondear
    lo, hi = usual_range(gravy)
    assert (lo, hi) == (pytest.approx(1.09), pytest.approx(9.01))
    assert target_values(np.arange(1, 101), "length") == target_lengths(np.arange(1, 101)) == (11, 31, 50, 70, 90)
    assert target_key("length", 20.0) == 20 and isinstance(target_key("length", 20.0), int)
    assert isinstance(target_key("hydrophobicity", 0), float)
    assert target_label("length", 9) == "t09" and target_label("hydrophobicity", -1.465) == "g-1.465"
    assert target_label("hydrophobicity", 0.88) == "g+0.880"
    with pytest.raises(ValueError):
        target_values(gravy, "charge")


def test_tolerances_and_columns_are_fixed_per_property():
    assert HIT_TOLERANCE == 2 and HYDRO_HIT_TOLERANCE == 0.25
    assert default_tolerance("length") == 2 and default_tolerance("hydrophobicity") == 0.25
    assert target_columns("length") == TARGET_COLUMNS and target_columns("hydrophobicity") == HYDRO_TARGET_COLUMNS
    assert "hydrophobicity_mean" in HYDRO_TARGET_COLUMNS and "length_wasserstein" in HYDRO_TARGET_COLUMNS
    assert "hydrophobicity_wasserstein" not in HYDRO_TARGET_COLUMNS


def test_percentiles_of_the_real_gravy_if_available():
    fasta = _PROCESSED / "Antibacterial_amps_clean.fasta"
    if not fasta.exists():
        pytest.skip("faltan los datos limpios")
    seqs = read_fasta(fasta)["sequence"].tolist()
    gravy = property_values(seqs, "hydrophobicity")
    assert gravy.size == 21754
    assert gravy.mean() == pytest.approx(-0.316, abs=5e-4) and gravy.std(ddof=1) == pytest.approx(0.943, abs=5e-4)
    t = target_values(gravy, "hydrophobicity")
    np.testing.assert_allclose(t, [-1.465, -0.730, -0.308, 0.144, 0.880], atol=6e-4)
    assert usual_range(gravy) == (pytest.approx(-1.465), pytest.approx(0.880))
    # Mucho material real cerca de los extremos (p10 y p90), para el W1 de las secuencias pedidas.
    near = [len(real_subset_around(seqs, x, prop="hydrophobicity")) for x in t]
    assert min(near) > 1000
    assert real_ceiling(gravy, t, HYDRO_HIT_TOLERANCE)["hit_rate"].between(0.05, 0.3).all()


# --------------------------------------------------------------------------- #
# Tasa de acierto, error absoluto medio y pendiente
# --------------------------------------------------------------------------- #
def test_hit_rate_with_known_gravy():
    gravy = [4.5] * 10
    assert hit_rate(gravy, 4.5, HYDRO_HIT_TOLERANCE) == 1.0
    assert hit_rate(gravy, 0.88, HYDRO_HIT_TOLERANCE) == 0.0
    # Límites inclusivos: |g - t| <= 0.25
    assert hit_rate([0.0, 0.25, -0.25, 0.26, -0.5], 0.0, 0.25) == pytest.approx(3 / 5)
    assert hit_rate([np.nan, 0.0], 0.0, 0.25) == 0.5  # un NaN (sin GRAVY) es un fallo


def test_evaluate_target_hydrophobic_and_hydrophilic_sequences():
    real = [HYDROPHOBIC] * 40 + [HYDROPHILIC] * 40 + [NEUTRAL] * 40
    refs = build_target_references(real, (4.5, -4.5, 0.0, -0.6), prop="hydrophobicity")
    assert set(refs) == {4.5, -4.5, 0.0, -0.6}
    assert refs[4.5][1] == 40 and refs[0.0][1] == 0 and refs[0.0][0] is None and refs[-0.6][1] == 40
    gen = [HYDROPHOBIC] * 30 + [HYDROPHOBIC[:19] + "V"] * 20  # GRAVY 4,5 y 4,485 (muy parecidas)
    hi = evaluate_target(gen, 4.5, *refs[4.5], prop="hydrophobicity")
    lo = evaluate_target(gen, -4.5, *refs[-4.5], prop="hydrophobicity")
    assert set(hi) == set(HYDRO_TARGET_COLUMNS)
    assert hi["hit_rate"] == 1.0 and lo["hit_rate"] == 0.0
    assert hi["hydrophobicity_mae"] == pytest.approx(20 * 0.015 / 50)
    assert lo["hydrophobicity_mae"] == pytest.approx(9.0, abs=0.02) and lo["hydrophobicity_mean"] == pytest.approx(hi["hydrophobicity_mean"])
    assert hi["n"] == 50 and hi["n_valid"] == 50 and hi["valid_fraction"] == 1.0 and hi["n_reference"] == 40
    assert hi["unique_fraction"] == pytest.approx(2 / 50)
    # W1 secundarias: longitud y carga neta (todas de 20 aa y cargas casi iguales a las de la referencia)
    assert hi["length_wasserstein"] == pytest.approx(0.0) and np.isfinite(hi["net_charge_wasserstein"])
    # Subconjunto pequeño: W1 = NaN y se indica el tamaño.
    small = evaluate_target(gen, 0.0, *refs[0.0], prop="hydrophobicity")
    assert small["n_reference"] == 0 and np.isnan(small["length_wasserstein"]) and np.isnan(small["net_charge_wasserstein"])
    assert small["hit_rate"] == 0.0


def test_evaluate_target_gravy_handles_empty_and_short_sequences():
    gen = ["", "II", HYDROPHOBIC, HYDROPHILIC]  # la vacía no tiene GRAVY; "II" (2 aa) no es válida pero sí tiene GRAVY
    r = evaluate_target(gen, 4.5, None, 0, prop="hydrophobicity")
    assert r["n"] == 4 and r["n_valid"] == 2 and r["valid_fraction"] == 0.5
    assert r["hit_rate"] == pytest.approx(2 / 4)  # "II" y la hidrofóbica; la vacía cuenta como fallo
    assert r["hydrophobicity_mean"] == pytest.approx((4.5 + 4.5 - 4.5) / 3)  # sin la vacía
    assert r["hydrophobicity_mae"] == pytest.approx((0 + 0 + 9) / 3)
    empty = evaluate_target(["", ""], 0.0, None, 0, prop="hydrophobicity")
    assert empty["n_valid"] == 0 and np.isnan(empty["hydrophobicity_mean"]) and np.isnan(empty["hydrophobicity_mae"])
    assert empty["hit_rate"] == 0.0 and np.isnan(empty["unique_fraction"])


def test_unconditional_gravy_table_and_real_ceiling():
    sample = [HYDROPHOBIC] * 6 + [HYDROPHILIC] * 2 + [NEUTRAL] * 2
    table = unconditional_target_table(sample, (4.5, -4.5, -0.6, 0.0), prop="hydrophobicity")
    assert list(table.columns) == ["target", "hydrophobicity_mean", "hydrophobicity_mae", "hit_rate"]
    assert table["hit_rate"].tolist() == pytest.approx([0.6, 0.2, 0.2, 0.0])
    assert table["hydrophobicity_mean"].nunique() == 1
    assert table["hydrophobicity_mean"].iloc[0] == pytest.approx((6 * 4.5 - 2 * 4.5 - 2 * 0.6) / 10)
    assert table["hydrophobicity_mae"].iloc[1] == pytest.approx((6 * 9 + 0 + 2 * 3.9) / 10)
    ceiling = real_ceiling(property_values(sample, "hydrophobicity"), (4.5, -4.5, 0.0), HYDRO_HIT_TOLERANCE)
    assert ceiling["hit_rate"].tolist() == pytest.approx([0.6, 0.2, 0.0])
    with_empty = unconditional_target_table(["", HYDROPHOBIC], (4.5,), prop="hydrophobicity")
    assert with_empty["hit_rate"].iloc[0] == 0.5 and with_empty["hydrophobicity_mean"].iloc[0] == pytest.approx(4.5)


def test_control_slope_for_gravy_recovers_a_line():
    t = [-1.465, -0.73, -0.3075, 0.144, 0.88]
    slope, intercept = control_slope(t, [2 * x + 0.5 for x in t])
    assert slope == pytest.approx(2.0) and intercept == pytest.approx(0.5)
    assert control_slope(t, t)[0] == pytest.approx(1.0)
    assert control_slope(t, [-0.3] * 5)[0] == pytest.approx(0.0, abs=1e-12)  # sin control: pendiente 0
    assert length_control_slope([9, 13, 20], [9, 13, 20])[0] == pytest.approx(1.0)  # la de longitud sigue igual


def test_real_subset_around_gravy_uses_the_tolerance():
    seqs = [HYDROPHOBIC, HYDROPHILIC, NEUTRAL, "G" * 9 + "S" * 11, "A"]  # GRAVY 4,5; -4,5; -0,6; -0,64; 1,8
    near = real_subset_around(seqs, -0.6, prop="hydrophobicity")
    assert HYDROPHOBIC not in near and HYDROPHILIC not in near and NEUTRAL in near and "A" not in near
    assert real_subset_around(seqs, 1.8, 0.0, prop="hydrophobicity") == ["A"]
    assert real_subset_around(["AAAAA", "A" * 9], 8, prop="length") == ["A" * 9]  # por defecto +-2 aa


# --------------------------------------------------------------------------- #
# Regresión: la evaluación de la longitud no cambia
# --------------------------------------------------------------------------- #
def _legacy_evaluate_target_length(sequences, target, reference, n_reference, tol=2):
    """Copia de la implementación original de la evaluación por objetivos de longitud (solo longitud)."""
    from scipy.stats import wasserstein_distance

    from amp_gan.evaluation import profile_sequences

    seqs = list(sequences)
    lengths = np.array([len(s) for s in seqs], dtype="float64")
    out = {name: float("nan") for name in TARGET_COLUMNS}
    out.update(target=int(target), n=len(seqs), n_valid=sum(is_valid(s) for s in seqs), n_reference=n_reference)
    if len(seqs):
        out.update(length_mean=float(lengths.mean()), length_mae=float(np.abs(lengths - target).mean()),
                   hit_rate=float(np.mean(np.abs(lengths - target) <= tol)))  # fmt: skip
    valid = [s for s in seqs if is_valid(s)]
    if seqs:
        out["valid_fraction"] = len(valid) / len(seqs)
    if valid:
        out["unique_fraction"] = len(set(valid)) / len(valid)
        if reference is not None:
            prof = profile_sequences(valid)
            for prop in ("net_charge", "hydrophobicity"):
                out[f"{prop}_wasserstein"] = float(wasserstein_distance(prof.values(prop), reference.profile.values(prop)))
    return out


def test_length_evaluation_is_identical_to_the_original_implementation():
    real = _random_sequences(400, seed=11, lo=5, hi=45)
    gen = _random_sequences(300, seed=12, lo=3, hi=40) + [""]
    refs = build_target_references(real, (9, 20, 40), min_n=30)
    for t in (9, 20, 40):
        new = evaluate_target(gen, t, *refs[t])
        old = _legacy_evaluate_target_length(gen, t, *refs[t])
        assert list(new) == list(old) == list(TARGET_COLUMNS)
        for key in old:
            a, b = new[key], old[key]
            assert (np.isnan(a) and np.isnan(b)) or a == b, (t, key, a, b)
        assert isinstance(new["target"], int)
        # Los mismos subconjuntos reales que con el filtro original por longitud.
        assert real_subset_around(real, t) == [s for s in real if abs(len(s) - t) <= 2]
        assert refs[t][1] == len([s for s in real if abs(len(s) - t) <= 2 and is_valid(s)])
    table = unconditional_target_table(gen, (9, 20, 40))
    assert list(table.columns) == ["target", "length_mean", "length_mae", "hit_rate"]
    lens = np.array([len(s) for s in gen], dtype=float)
    assert table["length_mae"].tolist() == [float(np.abs(lens - t).mean()) for t in (9, 20, 40)]


def test_saved_length_results_are_reproduced_by_the_generalized_evaluation():
    """Con las secuencias guardadas de ``cgan_length`` (época 200, sample) se recuperan las filas guardadas."""
    fasta = _PROCESSED / "cgan_length_generated_targets_sample.fasta"
    targets_csv = _PROCESSED / "cgan_length_targets.csv"
    train = _PROCESSED / "Antibacterial_amps_clean.fasta"
    if not (fasta.exists() and targets_csv.exists() and train.exists()):
        pytest.skip("faltan los resultados de cgan_length")
    gen = read_fasta(fasta)
    saved = pd.read_csv(targets_csv)
    saved = saved[(saved["epoch"] == saved["epoch"].max()) & (saved["mode"] == "sample")].set_index("target")
    real = read_fasta(train)["sequence"].tolist()
    targets = target_lengths([len(s) for s in real])
    assert targets == (9, 13, 20, 21, 40)
    refs = build_target_references(real, targets)
    for t in targets:
        seqs = gen[gen["id"].str.startswith(f"gen_t{t:02d}_")]["sequence"].tolist()
        got = evaluate_target(seqs, t, *refs[t])
        for key in TARGET_COLUMNS:
            if key != "target":
                assert got[key] == pytest.approx(saved.loc[t, key], rel=1e-6, abs=1e-7, nan_ok=True), (t, key)
    info = json.loads((_PROCESSED / "cgan_length_run_info.json").read_text(encoding="utf-8"))
    ref_path = _PROCESSED / "gan_stabc_generated_sample.fasta"
    if ref_path.exists():
        table = unconditional_target_table(read_fasta(ref_path)["sequence"].tolist(), targets)
        assert table.to_dict(orient="list") == info["unconditional_references"]["gan_stabc"]
    assert real_ceiling([len(s) for s in real], targets).to_dict(orient="list") == info["unconditional_references"]["real_ceiling"]


# --------------------------------------------------------------------------- #
# Condición de hidrofobicidad y experimento (backend por defecto del proceso)
# --------------------------------------------------------------------------- #
def test_hydrophobicity_variant_is_registered_and_configured():
    assert run_conditional_gan.VARIANTS["cgan_hydro"] == ("cgan_hydro", ("hydrophobicity",))
    assert run_conditional_gan.TARGET_PROPERTY == {"cgan_length": "length", "cgan_hydro": "hydrophobicity"}
    args = run_conditional_gan._parse(["--variant", "cgan_hydro", "--auto-backend"])
    assert args.variant == "cgan_hydro" and args.epochs == 200 and (args.snapshot_every, args.n_eval, args.seed) == (20, 2000, 42)
    cfg = run_conditional_gan.make_config("cgan_hydro")
    assert cfg.conditions == ("hydrophobicity",) and cfg.cond_dim == 1 and cfg.seed == 42
    assert (cfg.real_label, cfg.instance_noise_std, cfg.instance_noise_decay, cfg.discretize, cfg.gumbel_tau) == (0.9, 0.1, False, "gumbel_st", 1.0)
    from amp_gan.gan import GANConfig

    base = GANConfig.stabilized_constant_noise()
    for name in base.__dataclass_fields__:  # misma configuración que la referencia salvo la condición
        assert getattr(cfg, name) == getattr(base, name), name
    assert run_conditional_gan.make_config("cgan_length").conditions == ("length",)


@pytest.fixture(scope="module")
def cg():
    pytest.importorskip("keras")
    from amp_gan import conditional_gan

    return conditional_gan


def test_target_condition_matrix_for_hydrophobicity(cg):
    from amp_gan.conditions import ConditionSpec

    spec = ConditionSpec(("hydrophobicity",), (-4.5,), (4.5,))
    m = cg.target_condition_matrix(spec, 0.0, 4, prop="hydrophobicity")
    assert m.shape == (4, 1) and np.allclose(m, 0.5)
    spec2 = ConditionSpec(("length", "hydrophobicity"), (5.0, -4.5), (60.0, 4.5))
    m2 = cg.target_condition_matrix(spec2, 4.5, 3, fixed_raw=[20.0, 0.0], prop="hydrophobicity")
    np.testing.assert_allclose(m2, [[15 / 55, 1.0]] * 3, atol=1e-6)
    m3 = cg.target_condition_matrix(spec2, 20, 3, fixed_raw=[0.0, 0.0])  # por defecto, la longitud (como antes)
    np.testing.assert_allclose(m3, [[15 / 55, 0.5]] * 3, atol=1e-6)


def test_experiment_with_gravy_targets(cg):
    seqs = _random_sequences(5 * _BATCH, seed=3)
    from amp_gan.encoding import encode_sequences

    X = encode_sequences(seqs)
    ref = pd.DataFrame({"id": [f"s{i}" for i in range(len(seqs))], "sequence": seqs})
    cfg = cg.ConditionalGANConfig.stabilized_constant_noise(gen_hidden=(16,), disc_hidden=(16,), batch_size=_BATCH,
                                                            conditions=("hydrophobicity",))  # fmt: skip
    targets = target_values(property_values(seqs, "hydrophobicity"), "hydrophobicity")
    res = cg.run_conditional_experiment(X, ref, cfg, epochs=2, snapshot_every=1, n_eval=30, eval_seed=42,
                                        targets=targets, max_steps_per_epoch=2, target_property="hydrophobicity")  # fmt: skip
    assert list(res.metrics.columns) == ["epoch", "mode", *METRIC_NAMES] and len(res.metrics) == 2 * 2
    assert list(res.targets.columns) == ["epoch", "mode", *HYDRO_TARGET_COLUMNS]
    assert len(res.targets) == 2 * 2 * 5 and (res.targets["n"] == 30).all()
    np.testing.assert_allclose(sorted(res.targets["target"].unique()), sorted(targets))
    assert res.history.spec.names == ("hydrophobicity",)
    assert sorted(res.target_sequences["sample"]) == sorted(float(t) for t in targets)
    assert all(len(v) == 30 for v in res.target_sequences["sample"].values())
    # Las condiciones de (i) son los GRAVY reales muestreados con la semilla de evaluación.
    idx = cg.empirical_condition_indices(len(seqs), 30, 42)
    np.testing.assert_array_equal(res.condition_indices, idx)
    raw = raw_conditions_from_sequences([seqs[i] for i in idx], ("hydrophobicity",))
    assert raw.shape == (30, 1)
    # Por defecto (target_property="length") la condición solo de hidrofobicidad omite (ii), como antes.
    res2 = cg.run_conditional_experiment(X, ref, cfg, epochs=1, snapshot_every=1, n_eval=20, eval_seed=42,
                                         targets=(10,), max_steps_per_epoch=1)  # fmt: skip
    assert res2.targets.empty and list(res2.targets.columns) == ["epoch", "mode", *TARGET_COLUMNS]
    with pytest.raises(ValueError):
        cg.run_conditional_experiment(X, ref, cfg, epochs=1, snapshot_every=1, n_eval=20, eval_seed=42,
                                      targets=(0.0,), target_property="charge")  # fmt: skip


# --------------------------------------------------------------------------- #
# Tests por backend, en subproceso
# --------------------------------------------------------------------------- #
def _installed(backend: str) -> bool:
    return backend in runtime.installed_backends()


def _run_snippet(backend: str, snippet: str, timeout: int = 900) -> dict:
    env = dict(os.environ, KERAS_BACKEND=backend, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
    code = "import sys; sys.path.insert(0, %r)\n" % str(_SRC) + textwrap.dedent(snippet)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=timeout)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT="))
    return json.loads(line[len("RESULT="):])


_SMOKE_SNIPPET = """
import json
import numpy as np
import pandas as pd
import keras
from amp_gan import conditional_gan as cg
from amp_gan.conditions import property_values, target_values
from amp_gan.encoding import encode_sequences
from amp_gan.data import STANDARD_ALPHABET

rng = np.random.default_rng(0)
alphabet = np.array(sorted(STANDARD_ALPHABET))
seqs = ["".join(rng.choice(alphabet, size=int(rng.integers(5, 31)))) for _ in range(40)]
X = encode_sequences(seqs)
cfg = cg.ConditionalGANConfig.stabilized_constant_noise(gen_hidden=(16,), disc_hidden=(16,), batch_size=8,
                                                        conditions=("hydrophobicity",))
h1 = cg.train_conditional_gan(X, cfg, epochs=2, max_steps_per_epoch=3)
h2 = cg.train_conditional_gan(X, cfg, epochs=2, max_steps_per_epoch=3)
spec = h1.spec
ref = pd.DataFrame({"id": [f"s{i}" for i in range(len(seqs))], "sequence": seqs})
targets = target_values(property_values(seqs, "hydrophobicity"), "hydrophobicity")
res = cg.run_conditional_experiment(X, ref, cfg, epochs=2, snapshot_every=1, n_eval=20, eval_seed=42,
                                    targets=targets, max_steps_per_epoch=2, target_property="hydrophobicity")
print("RESULT=" + json.dumps({
    "backend": keras.backend.backend(),
    "history": {k: list(v) for k, v in h1.items()},
    "history2": {k: list(v) for k, v in h2.items()},
    "spec_names": list(spec.names),
    "spec_low": spec.low[0], "spec_high": spec.high[0],
    "true_min": float(property_values(seqs, "hydrophobicity").min()), "true_max": float(property_values(seqs, "hydrophobicity").max()),
    "gen_input_dim": int(h1.generator.inputs[1].shape[-1]),
    "targets_shape": list(res.targets.shape), "targets_cols": list(res.targets.columns),
    "hit_rate_ok": bool(res.targets["hit_rate"].between(0, 1).all()),
}))
"""


@lru_cache(maxsize=None)
def _smoke_result(backend: str) -> dict:
    return _run_snippet(backend, _SMOKE_SNIPPET)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_hydrophobicity_training_smoke(backend):
    if not _installed(backend):
        pytest.skip(f"{backend} no está instalado")
    r = _smoke_result(backend)
    assert r["backend"] == backend and r["spec_names"] == ["hydrophobicity"] and r["gen_input_dim"] == 1
    assert r["spec_low"] == pytest.approx(r["true_min"]) and r["spec_high"] == pytest.approx(r["true_max"])
    h = r["history"]
    assert set(h) == {"d_loss_real", "d_loss_fake", "d_loss", "g_loss"}
    assert all(len(v) == 2 for v in h.values()) and all(np.all(np.isfinite(h[k])) for k in h), h
    assert h == r["history2"]  # misma semilla -> mismo historial (exacto, en CPU)
    assert r["targets_shape"] == [2 * 2 * 5, len(HYDRO_TARGET_COLUMNS) + 2]
    assert r["targets_cols"] == ["epoch", "mode", *HYDRO_TARGET_COLUMNS] and r["hit_rate_ok"]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_run_conditional_gan_script_writes_hydro_files(backend, tmp_path):
    if not _installed(backend):
        pytest.skip(f"{backend} no está instalado")
    if not run_conditional_gan.TRAIN_FASTA.exists():
        pytest.skip("faltan los datos limpios")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
    env.pop("KERAS_BACKEND", None)
    env.pop(runtime.BACKEND_ENV_VAR, None)
    cmd = [sys.executable, str(_SRC / "amp_gan" / "run_conditional_gan.py"), "--variant", "cgan_hydro", "--epochs", "2",
           "--snapshot-every", "1", "--max-steps", "6", "--n-eval", "50", "--backend", backend, "--out-dir", str(tmp_path)]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["cgan_hydro_generated_argmax.fasta", "cgan_hydro_generated_sample.fasta",
                     "cgan_hydro_generated_targets_sample.fasta", "cgan_hydro_history.csv",
                     "cgan_hydro_metrics.csv", "cgan_hydro_run_info.json", "cgan_hydro_targets.csv"]  # fmt: skip
    hist = pd.read_csv(tmp_path / "cgan_hydro_history.csv")
    assert list(hist.columns) == ["epoch", "d_loss_real", "d_loss_fake", "d_loss", "g_loss"]
    assert list(hist["epoch"]) == [1, 2] and np.isfinite(hist.to_numpy()).all()
    metrics = pd.read_csv(tmp_path / "cgan_hydro_metrics.csv")
    assert list(metrics.columns) == ["epoch", "mode", *METRIC_NAMES]
    assert set(metrics["mode"]) == {"argmax", "sample", "real_subset", "random_uniform"}
    targets = pd.read_csv(tmp_path / "cgan_hydro_targets.csv")
    assert list(targets.columns) == ["epoch", "mode", *HYDRO_TARGET_COLUMNS]
    assert len(targets) == 2 * 2 * 5 and targets["target"].nunique() == 5
    np.testing.assert_allclose(sorted(targets["target"].unique()), [-1.465, -0.730, -0.3075, 0.1440, 0.880], atol=1e-3)
    fasta = (tmp_path / "cgan_hydro_generated_targets_sample.fasta").read_text(encoding="utf-8")
    assert ">gen_g-1.465_0000" in fasta and ">gen_g+0.880_0049" in fasta and fasta.count(">") == 5 * 50
    info = json.loads((tmp_path / "cgan_hydro_run_info.json").read_text(encoding="utf-8"))
    assert info["keras_backend"] == backend and info["variant"] == "cgan_hydro" and info["target_property"] == "hydrophobicity"
    assert info["config"]["conditions"] == ["hydrophobicity"] and info["hit_tolerance"] == 0.25
    assert len(info["target_values"]) == 5 and info["usual_range_p10_p90"] == pytest.approx([-1.465, 0.880])
    assert info["condition_spec"]["names"] == ["hydrophobicity"] and "target_lengths" not in info
    assert set(info["unconditional_references"]) <= {"real_ceiling", "gan_stabc"}
    assert KYTE_DOOLITTLE["I"] == 4.5

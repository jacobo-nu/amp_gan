"""Tests de la GAN condicional (``amp_gan.conditions`` y ``amp_gan.conditional_gan``).

Modelos pequeños y pocos pasos, en CPU. El humo de entrenamiento y el script de
ejecución se lanzan en un subproceso por backend (TensorFlow y torch) y se saltan
si el backend no está instalado; el resto usa el backend por defecto del proceso.
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

from amp_gan import conditions, run_conditional_gan, runtime
from amp_gan.conditions import (
    CONDITION_NAMES,
    HIT_TOLERANCE,
    TARGET_COLUMNS,
    ConditionSpec,
    build_target_references,
    evaluate_target,
    fit_condition_spec,
    hit_rate,
    length_control_slope,
    length_percentiles,
    raw_conditions_from_encoded,
    raw_conditions_from_sequences,
    real_ceiling,
    target_lengths,
    unconditional_target_table,
    usual_length_range,
)
from amp_gan.data import MAX_LEN, MIN_LEN, STANDARD_ALPHABET
from amp_gan.encoding import SEQ_LEN, VOCAB_SIZE, encode_sequences
from amp_gan.evaluation import METRIC_NAMES
from amp_gan.properties import hydrophobicity

runtime.use_cpu_only()
keras = pytest.importorskip("keras")

from amp_gan import conditional_gan as cg  # noqa: E402
from amp_gan.gan import GANConfig, GumbelStraightThrough, InstanceNoise, sample_sequences  # noqa: E402

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
_BACKENDS = ("tensorflow", "torch")
_BATCH = 8
_PROCESSED = _ROOT / "context" / "data" / "processed"


def _small(**kw) -> cg.ConditionalGANConfig:
    """Configuración estabilizada de ruido constante con redes pequeñas."""
    return cg.ConditionalGANConfig.stabilized_constant_noise(gen_hidden=(16,), disc_hidden=(16,), batch_size=_BATCH, **kw)


def _random_sequences(n: int, seed: int = 0, lo: int = MIN_LEN, hi: int = 30) -> list[str]:
    rng = np.random.default_rng(seed)
    alphabet = np.array(sorted(STANDARD_ALPHABET))
    return ["".join(rng.choice(alphabet, size=int(rng.integers(lo, hi + 1)))) for _ in range(n)]


@pytest.fixture
def seqs():
    return _random_sequences(5 * _BATCH)


@pytest.fixture
def real_data(seqs):
    return encode_sequences(seqs)


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


# --------------------------------------------------------------------------- #
# Condiciones: normalización
# --------------------------------------------------------------------------- #
def test_length_normalization_round_trip():
    spec = fit_condition_spec(("length",), np.array([[7.0], [30.0]]))
    assert spec.low == (5.0,) and spec.high == (60.0,)  # rango fijo, no depende de los datos
    raw = np.array([[5.0], [20.0], [60.0]])
    norm = spec.normalize(raw)
    np.testing.assert_allclose(norm[:, 0], [0.0, 15 / 55, 1.0], atol=1e-6)  # (L - 5) / 55
    assert norm.dtype == np.float32
    np.testing.assert_allclose(spec.denormalize(norm), raw, atol=1e-5)


def test_hydrophobicity_normalization_is_minmax_of_training_and_round_trips():
    sequences = _random_sequences(60, seed=1)
    raw = raw_conditions_from_sequences(sequences, ("hydrophobicity",))
    spec = fit_condition_spec(("hydrophobicity",), raw)
    assert spec.low == (raw.min(),) and spec.high == (raw.max(),)
    norm = spec.normalize(raw)
    assert norm.min() == pytest.approx(0.0, abs=1e-6) and norm.max() == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(spec.denormalize(norm), raw, atol=1e-5)


def test_two_conditions_spec_and_round_trip():
    sequences = _random_sequences(40, seed=2)
    raw = raw_conditions_from_sequences(sequences, ("length", "hydrophobicity"))
    assert raw.shape == (40, 2)
    spec = fit_condition_spec(("length", "hydrophobicity"), raw)
    assert spec.k == 2 and spec.names == ("length", "hydrophobicity")
    norm = spec.normalize(raw)
    assert norm.shape == (40, 2) and norm.min() >= 0.0 and norm.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(spec.denormalize(norm), raw, atol=1e-5)


def test_raw_conditions_from_encoded_match_the_sequences(seqs, real_data):
    for names in (("length",), ("hydrophobicity",), ("length", "hydrophobicity")):
        a = raw_conditions_from_sequences(seqs, names)
        b = raw_conditions_from_encoded(real_data, names)
        np.testing.assert_allclose(b, a, atol=1e-9)
    assert raw_conditions_from_encoded(real_data, ("length",))[:, 0].tolist() == [len(s) for s in seqs]
    assert raw_conditions_from_encoded(real_data[:1], ("hydrophobicity",))[0, 0] == pytest.approx(hydrophobicity(seqs[0]))


def test_invalid_condition_names_and_degenerate_range_raise():
    for bad in ((), ("charge",), ("length", "length")):
        with pytest.raises(ValueError):
            conditions.check_condition_names(bad)
        with pytest.raises(ValueError):
            cg.ConditionalGANConfig(conditions=bad)
    with pytest.raises(ValueError):
        fit_condition_spec(("hydrophobicity",), np.ones((5, 1)))  # máximo = mínimo
    assert CONDITION_NAMES == ("length", "hydrophobicity")


# --------------------------------------------------------------------------- #
# Percentiles, objetivos y evaluación por objetivos (sin Keras)
# --------------------------------------------------------------------------- #
def test_percentiles_target_lengths_and_usual_range_from_data():
    lengths = np.arange(1, 101)  # 1..100: p10 = 10.9, p50 = 50.5, p90 = 90.1 (interpolación lineal)
    p = length_percentiles(lengths)
    assert sorted(p) == [10, 30, 50, 70, 90]
    assert p[50] == pytest.approx(50.5) and p[10] == pytest.approx(10.9)
    assert target_lengths(lengths) == (11, 31, 50, 70, 90)  # redondeo al entero más cercano
    assert usual_length_range(lengths) == (11, 90)
    assert usual_length_range([20] * 10) == (20, 20)


def test_percentiles_of_the_real_training_data_if_available():
    fasta = _PROCESSED / "Antibacterial_amps_clean.fasta"
    if not fasta.exists():
        pytest.skip("faltan los datos limpios")
    from amp_gan.data import read_fasta

    lengths = read_fasta(fasta)["sequence"].str.len().to_numpy()
    assert target_lengths(lengths) == (9, 13, 20, 21, 40)
    assert usual_length_range(lengths) == (9, 40)


def test_hit_rate_with_known_lengths():
    assert hit_rate([20] * 10, 20) == 1.0
    assert hit_rate([20] * 10, 40) == 0.0
    assert hit_rate([18, 19, 20, 21, 22, 23, 17], 20) == pytest.approx(5 / 7)  # |L - 20| <= 2 incluye 18 y 22
    assert hit_rate([], 20) != hit_rate([], 20)  # NaN
    assert HIT_TOLERANCE == 2


def test_evaluate_target_all_length_20():
    gen = _random_sequences(50, seed=3, lo=20, hi=20)
    assert {len(s) for s in gen} == {20}
    real = _random_sequences(100, seed=4, lo=18, hi=22)
    refs = build_target_references(real, (20, 40), min_n=30)
    r20 = evaluate_target(gen, 20, *refs[20])
    r40 = evaluate_target(gen, 40, *refs[40])
    assert set(r20) == set(TARGET_COLUMNS)
    assert r20["hit_rate"] == 1.0 and r20["length_mean"] == 20.0 and r20["length_mae"] == 0.0
    assert r40["hit_rate"] == 0.0 and r40["length_mae"] == 20.0 and r40["length_mean"] == 20.0
    assert r20["valid_fraction"] == 1.0 and r20["unique_fraction"] == 1.0 and r20["n"] == 50 and r20["n_valid"] == 50
    assert refs[20][0] is not None and r20["n_reference"] == refs[20][1] >= 30
    assert np.isfinite(r20["net_charge_wasserstein"]) and np.isfinite(r20["hydrophobicity_wasserstein"])


def test_small_reference_subset_gives_nan_w1_and_reports_the_size():
    real = _random_sequences(100, seed=5, lo=18, hi=22) + _random_sequences(5, seed=6, lo=40, hi=40)
    refs = build_target_references(real, (20, 40), min_n=30)
    ref40, n40 = refs[40]
    assert ref40 is None and n40 == 5
    r = evaluate_target(_random_sequences(20, seed=7, lo=40, hi=40), 40, ref40, n40)
    assert r["n_reference"] == 5 and r["hit_rate"] == 1.0
    assert np.isnan(r["net_charge_wasserstein"]) and np.isnan(r["hydrophobicity_wasserstein"])


def test_evaluate_target_counts_empty_and_short_sequences_as_misses():
    gen = ["", "AAA", "A" * 20, "C" * 21]  # vacía y de 3 aa no son válidas (< 5)
    r = evaluate_target(gen, 20, None, 0)
    assert r["n"] == 4 and r["n_valid"] == 2 and r["valid_fraction"] == 0.5
    assert r["hit_rate"] == 0.5 and r["length_mean"] == pytest.approx((0 + 3 + 20 + 21) / 4)
    empty = evaluate_target(["", ""], 20, None, 0)
    assert empty["n_valid"] == 0 and np.isnan(empty["unique_fraction"]) and empty["hit_rate"] == 0.0


def test_unconditional_reference_and_real_ceiling():
    table = unconditional_target_table(["A" * 20] * 8 + ["A" * 9] * 2, (9, 20, 40))
    assert table["hit_rate"].tolist() == [0.2, 0.8, 0.0]
    assert table["length_mean"].nunique() == 1 and table["length_mean"].iloc[0] == pytest.approx(17.8)
    ceiling = real_ceiling([8, 9, 10, 11, 12, 20, 20, 40], (9, 20, 40))
    assert ceiling["hit_rate"].tolist() == pytest.approx([4 / 8, 2 / 8, 1 / 8])  # 9: 7-11 -> 9,10,11,8


def test_length_control_slope_recovers_a_line():
    slope, intercept = length_control_slope([9, 13, 20, 21, 40], [2 * t + 1 for t in (9, 13, 20, 21, 40)])
    assert slope == pytest.approx(2.0) and intercept == pytest.approx(1.0)
    assert length_control_slope([9, 13, 20, 21, 40], [9, 13, 20, 21, 40])[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Modelos condicionales
# --------------------------------------------------------------------------- #
def test_config_inherits_the_stabilized_constant_noise_base():
    cfg = cg.ConditionalGANConfig.stabilized_constant_noise()
    base = GANConfig.stabilized_constant_noise()
    assert cfg.conditions == ("length",) and cfg.cond_dim == 1
    for name in base.__dataclass_fields__:
        assert getattr(cfg, name) == getattr(base, name), name
    assert (cfg.real_label, cfg.instance_noise_std, cfg.instance_noise_decay, cfg.discretize, cfg.gumbel_tau) == (0.9, 0.1, False, "gumbel_st", 1.0)
    assert cg.ConditionalGANConfig.stabilized_constant_noise(conditions=("length", "hydrophobicity")).cond_dim == 2


@pytest.mark.parametrize("conditions_", [("length",), ("hydrophobicity",), ("length", "hydrophobicity")])
def test_generator_and_discriminator_shapes(conditions_):
    cfg = _small(conditions=conditions_)
    k = len(conditions_)
    g = cg.build_conditional_generator(cfg)
    d = cg.build_conditional_discriminator(cfg)
    assert [tuple(i.shape) for i in g.inputs] == [(None, cfg.latent_dim), (None, k)]
    assert tuple(g.output.shape) == (None, SEQ_LEN, VOCAB_SIZE)
    assert [tuple(i.shape) for i in d.inputs] == [(None, SEQ_LEN, VOCAB_SIZE), (None, k)]
    assert tuple(d.output.shape) == (None, 1)
    # La condición se concatena a z (primer Dense) y a la secuencia aplanada (primer Dense de D).
    first_g = next(layer for layer in g.layers if isinstance(layer, keras.layers.Dense))
    first_d = next(layer for layer in d.layers if isinstance(layer, keras.layers.Dense))
    assert first_g.kernel.shape[0] == cfg.latent_dim + k
    assert first_d.kernel.shape[0] == SEQ_LEN * VOCAB_SIZE + k
    z = np.random.default_rng(0).standard_normal((3, cfg.latent_dim)).astype("float32")
    c = np.full((3, k), 0.5, dtype="float32")
    probs = _np(g([z, c]))
    assert probs.shape == (3, SEQ_LEN, VOCAB_SIZE) and np.allclose(probs.sum(-1), 1.0, atol=1e-5)
    score = _np(d([probs, c]))
    assert score.shape == (3, 1) and ((score > 0) & (score < 1)).all()


def test_combined_model_has_z_and_c_inputs_and_the_stabilization_layers():
    cfg = _small()
    g, d, gan = cg.build_conditional_models(cfg)
    assert [tuple(i.shape) for i in gan.inputs] == [(None, cfg.latent_dim), (None, 1)]
    assert tuple(gan.output.shape) == (None, 1)
    noise = next(layer for layer in d.layers if isinstance(layer, InstanceNoise))
    assert float(_np(noise.std)) == pytest.approx(0.1)  # ruido de instancia constante 0,1
    assert isinstance(gan.get_layer("gumbel_st"), GumbelStraightThrough) and gan.get_layer("gumbel_st").tau == 1.0
    assert any(isinstance(layer, keras.layers.Dropout) for layer in d.layers)
    assert not any(w.trainable for w in gan.get_layer("discriminator").weights)  # D queda congelado en el combinado
    assert not any(isinstance(layer, GumbelStraightThrough) for layer in g.layers)  # el generador sigue dando probabilidades
    # Sin estabilización (vanilla condicionada) no hay ruido ni Gumbel.
    g2, d2, gan2 = cg.build_conditional_models(cg.ConditionalGANConfig(gen_hidden=(16,), disc_hidden=(16,), batch_size=_BATCH))
    assert not any(isinstance(layer, InstanceNoise) for layer in d2.layers)
    assert not any(isinstance(layer, GumbelStraightThrough) for layer in gan2.layers)


def test_condition_reaches_the_generator():
    cfg = _small()
    g = cg.build_conditional_generator(cfg)
    z = np.random.default_rng(0).standard_normal((4, cfg.latent_dim)).astype("float32")
    low = _np(g([z, np.zeros((4, 1), dtype="float32")]))
    high = _np(g([z, np.ones((4, 1), dtype="float32")]))
    assert np.abs(low - high).max() > 1e-4  # a igual z, otra condición -> otra salida
    same = _np(g([z, np.zeros((4, 1), dtype="float32")]))
    np.testing.assert_allclose(low, same, atol=1e-6)  # y misma condición -> misma salida


def test_condition_reaches_the_discriminator():
    cfg = _small()
    d = cg.build_conditional_discriminator(cfg)
    x = encode_sequences(_random_sequences(4, seed=9))
    a = _np(d([x, np.zeros((4, 1), dtype="float32")], training=False))
    b = _np(d([x, np.ones((4, 1), dtype="float32")], training=False))
    assert np.abs(a - b).max() > 1e-5


def test_generate_conditional_probs_shapes_seed_and_condition():
    cfg = _small()
    g = cg.build_conditional_generator(cfg)
    c = np.linspace(0, 1, 6, dtype="float32")[:, None]
    p1 = cg.generate_conditional_probs(g, c, cfg, seed=1)
    p2 = cg.generate_conditional_probs(g, c, cfg, seed=1)
    p3 = cg.generate_conditional_probs(g, c, cfg, seed=2)
    assert p1.shape == (6, SEQ_LEN, VOCAB_SIZE) and p1.dtype == np.float32
    np.testing.assert_array_equal(p1, p2)
    assert np.abs(p1 - p3).max() > 1e-4
    flipped = cg.generate_conditional_probs(g, c[::-1].copy(), cfg, seed=1)
    assert np.abs(p1 - flipped).max() > 1e-4
    out = sample_sequences(p1, rng=np.random.default_rng(0), mode="sample")
    assert len(out) == 6 and all(isinstance(s, str) for s in out)
    with pytest.raises(ValueError):
        cg.generate_conditional_probs(g, np.zeros(3, dtype="float32"), cfg)


def test_target_condition_matrix_and_empirical_indices():
    spec = fit_condition_spec(("length",), np.array([[10.0]]))
    m = cg.target_condition_matrix(spec, 20, 5)
    assert m.shape == (5, 1) and np.allclose(m, 15 / 55)
    spec2 = ConditionSpec(("length", "hydrophobicity"), (5.0, -2.0), (60.0, 2.0))
    m2 = cg.target_condition_matrix(spec2, 40, 3, fixed_raw=[0.0, 0.0])
    np.testing.assert_allclose(m2, [[35 / 55, 0.5]] * 3, atol=1e-6)
    i1 = cg.empirical_condition_indices(1000, 50, 42)
    np.testing.assert_array_equal(i1, cg.empirical_condition_indices(1000, 50, 42))
    assert i1.shape == (50,) and i1.min() >= 0 and i1.max() < 1000 and len(set(i1.tolist())) < 50 + 1


def test_train_step_uses_the_real_batch_conditions_for_real_fake_and_generator(real_data, monkeypatch):
    cfg = _small()
    g, d, gan = cg.build_conditional_models(cfg)
    cond = np.linspace(0.0, 1.0, _BATCH, dtype="float32")[:, None]
    seen = []
    orig_d, orig_g = d.train_on_batch, gan.train_on_batch
    monkeypatch.setattr(d, "train_on_batch", lambda x, y, *a, **k: (seen.append(("d", x)), orig_d(x, y, *a, **k))[1], raising=False)
    monkeypatch.setattr(gan, "train_on_batch", lambda x, y, *a, **k: (seen.append(("g", x)), orig_g(x, y, *a, **k))[1], raising=False)
    out = cg.conditional_train_step(g, d, gan, real_data[:_BATCH], cond, cfg, noise_std=0.1)
    assert set(out) == {"d_loss_real", "d_loss_fake", "g_loss"} and all(np.isfinite(list(out.values())))
    assert [k for k, _ in seen] == ["d", "d", "g"]
    for _, x in seen:
        np.testing.assert_array_equal(np.asarray(x[1]), cond)  # misma condición en el real, el falso y el paso de G
    np.testing.assert_array_equal(seen[0][1][0], real_data[:_BATCH])


# --------------------------------------------------------------------------- #
# Entrenamiento y experimento (backend por defecto del proceso)
# --------------------------------------------------------------------------- #
def test_train_conditional_gan_history_spec_and_callback(real_data):
    cfg = _small()
    calls = []
    h = cg.train_conditional_gan(real_data, cfg, epochs=2, max_steps_per_epoch=2, callback=lambda e, gen: calls.append(e))
    assert calls == [0, 1]
    assert set(h) == {"d_loss_real", "d_loss_fake", "d_loss", "g_loss"} and all(len(v) == 2 for v in h.values())
    assert np.isfinite(np.asarray([h[k] for k in h])).all()
    assert h.spec.names == ("length",) and h.generator is not None and h.discriminator is not None and h.gan is not None
    with pytest.raises(ValueError):
        cg.train_conditional_gan(real_data[: _BATCH - 1], cfg, epochs=1)


def test_run_conditional_experiment_format(seqs, real_data):
    cfg = _small()
    ref = pd.DataFrame({"id": [f"s{i}" for i in range(len(seqs))], "sequence": seqs})
    res = cg.run_conditional_experiment(real_data, ref, cfg, epochs=3, snapshot_every=2, n_eval=40, eval_seed=42,
                                        targets=(8, 12), max_steps_per_epoch=2)  # fmt: skip
    assert res.snapshot_epochs == [1, 2, 3]
    assert list(res.metrics.columns) == ["epoch", "mode", *METRIC_NAMES]
    assert len(res.metrics) == 3 * 2 and set(res.metrics["mode"]) == {"argmax", "sample"}
    assert list(res.targets.columns) == ["epoch", "mode", *TARGET_COLUMNS]
    assert len(res.targets) == 3 * 2 * 2 and set(res.targets["target"]) == {8, 12}
    assert (res.targets["n"] == 40).all() and (res.metrics["n"] == 40).all()
    assert set(res.sequences) == {"argmax", "sample"} and len(res.sequences["sample"]) == 40
    assert set(res.target_sequences["sample"]) == {8, 12} and len(res.target_sequences["sample"][8]) == 40
    assert res.condition_indices.shape == (40,) and len(res.epoch_seconds) == 3
    # Sin objetivos se omite la evaluación (ii).
    res2 = cg.run_conditional_experiment(real_data, ref, cfg, epochs=1, snapshot_every=1, n_eval=20, eval_seed=42, max_steps_per_epoch=1)
    assert res2.targets.empty and list(res2.targets.columns) == ["epoch", "mode", *TARGET_COLUMNS]
    with pytest.raises(ValueError):
        cg.run_conditional_experiment(real_data[:-1], ref, cfg, epochs=1, snapshot_every=1, n_eval=20, eval_seed=42)


def test_experiment_supports_hydrophobicity_and_both_conditions(seqs, real_data):
    ref = pd.DataFrame({"id": [f"s{i}" for i in range(len(seqs))], "sequence": seqs})
    for conditions_ in (("hydrophobicity",), ("length", "hydrophobicity")):
        cfg = _small(conditions=conditions_)
        res = cg.run_conditional_experiment(real_data, ref, cfg, epochs=1, snapshot_every=1, n_eval=20, eval_seed=42,
                                            targets=(10,), max_steps_per_epoch=1)  # fmt: skip
        assert res.history.spec.names == conditions_ and len(res.metrics) == 2
        assert res.targets.empty == ("length" not in conditions_)  # (ii) solo si la condición incluye la longitud


def test_run_conditional_gan_script_registry_and_parser():
    assert run_conditional_gan.VARIANTS == {
        "cgan_length": ("cgan_length", ("length",)),
        "cgan_hydro": ("cgan_hydro", ("hydrophobicity",)),
    }
    args = run_conditional_gan._parse(["--variant", "cgan_length", "--epochs", "3", "--auto-backend"])
    assert args.variant == "cgan_length" and args.epochs == 3 and args.auto_backend is True
    assert (args.snapshot_every, args.n_eval, args.seed, args.max_steps) == (20, 2000, 42, None)
    with pytest.raises(SystemExit):
        run_conditional_gan._parse(["--variant", "cgan_both"])  # todavía no está registrada la de longitud + hidrofobicidad
    cfg = run_conditional_gan.make_config("cgan_length")
    assert cfg.conditions == ("length",) and cfg.real_label == 0.9 and cfg.instance_noise_decay is False


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


_TRAIN_SNIPPET = """
import json
import numpy as np
import keras
from amp_gan import conditional_gan as cg
from amp_gan.encoding import encode_sequences
from amp_gan.gan import sample_sequences
from amp_gan.data import STANDARD_ALPHABET

rng = np.random.default_rng(0)
alphabet = np.array(sorted(STANDARD_ALPHABET))
seqs = ["".join(rng.choice(alphabet, size=int(rng.integers(5, 31)))) for _ in range(40)]
X = encode_sequences(seqs)
cfg = cg.ConditionalGANConfig.stabilized_constant_noise(gen_hidden=(16,), disc_hidden=(16,), batch_size=8)

h1 = cg.train_conditional_gan(X, cfg, epochs=2, max_steps_per_epoch=3)
h2 = cg.train_conditional_gan(X, cfg, epochs=2, max_steps_per_epoch=3)
g, d, gan = h1.generator, h1.discriminator, h1.gan
probs = cg.generate_conditional_probs(g, np.linspace(0, 1, 7, dtype="float32")[:, None], cfg, seed=1)
out = sample_sequences(probs, rng=np.random.default_rng(1), mode="sample")

w = lambda m: [np.array(v) for v in m.get_weights()]
differs = lambda a, b: any(not np.array_equal(x, y) for x, y in zip(a, b))
x = X[:8]; c = np.linspace(0, 1, 8, dtype="float32")[:, None]; ones = np.ones((8, 1), "float32"); zeros = np.zeros((8, 1), "float32")
g0, d0 = w(g), w(d)
d.trainable = True
d.train_on_batch([x, c], ones * 0.9)
g1, d1 = w(g), w(d)
d.trainable = False
gan.train_on_batch([keras.ops.convert_to_numpy(keras.random.normal((8, 100))), c], ones)
g2, d2 = w(g), w(d)
print("RESULT=" + json.dumps({
    "backend": keras.backend.backend(),
    "history": {k: list(v) for k, v in h1.items()},
    "history2": {k: list(v) for k, v in h2.items()},
    "probs_shape": list(probs.shape), "probs_sum_ok": bool(np.allclose(probs.sum(-1), 1.0, atol=1e-5)),
    "seqs_ok": all(isinstance(s, str) and set(s) <= set(STANDARD_ALPHABET) and len(s) <= 60 for s in out),
    "d_step_changes_d_only": bool(differs(d0, d1) and not differs(g0, g1)),
    "g_step_changes_g_only": bool(differs(g1, g2) and not differs(d1, d2)),
}))
"""


@lru_cache(maxsize=None)
def _train_result(backend: str) -> dict:
    return _run_snippet(backend, _TRAIN_SNIPPET)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_training_smoke(backend):
    if not _installed(backend):
        pytest.skip(f"{backend} no está instalado")
    r = _train_result(backend)
    assert r["backend"] == backend
    h = r["history"]
    assert set(h) == {"d_loss_real", "d_loss_fake", "d_loss", "g_loss"}
    assert all(len(v) == 2 for v in h.values()) and all(np.all(np.isfinite(h[k])) for k in h), h
    np.testing.assert_allclose(h["d_loss"], 0.5 * (np.asarray(h["d_loss_real"]) + np.asarray(h["d_loss_fake"])), rtol=1e-6)
    assert r["probs_shape"] == [7, 60, 21] and r["probs_sum_ok"] and r["seqs_ok"]
    assert r["d_step_changes_d_only"] and r["g_step_changes_g_only"]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_training_is_deterministic(backend):
    if not _installed(backend):
        pytest.skip(f"{backend} no está instalado")
    r = _train_result(backend)
    assert r["history"] == r["history2"]  # misma semilla -> mismo historial (exacto, en CPU)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_run_conditional_gan_script_writes_expected_files(backend, tmp_path):
    if not _installed(backend):
        pytest.skip(f"{backend} no está instalado")
    if not run_conditional_gan.TRAIN_FASTA.exists():
        pytest.skip("faltan los datos limpios")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
    env.pop("KERAS_BACKEND", None)
    env.pop(runtime.BACKEND_ENV_VAR, None)
    cmd = [sys.executable, str(_SRC / "amp_gan" / "run_conditional_gan.py"), "--variant", "cgan_length", "--epochs", "2",
           "--snapshot-every", "1", "--max-steps", "6", "--n-eval", "50", "--backend", backend, "--out-dir", str(tmp_path)]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["cgan_length_generated_argmax.fasta", "cgan_length_generated_sample.fasta",
                     "cgan_length_generated_targets_sample.fasta", "cgan_length_history.csv",
                     "cgan_length_metrics.csv", "cgan_length_run_info.json", "cgan_length_targets.csv"]  # fmt: skip
    hist = pd.read_csv(tmp_path / "cgan_length_history.csv")
    assert list(hist.columns) == ["epoch", "d_loss_real", "d_loss_fake", "d_loss", "g_loss"]
    assert list(hist["epoch"]) == [1, 2] and np.isfinite(hist.to_numpy()).all()
    metrics = pd.read_csv(tmp_path / "cgan_length_metrics.csv")
    assert list(metrics.columns) == ["epoch", "mode", *METRIC_NAMES]
    assert set(metrics["mode"]) == {"argmax", "sample", "real_subset", "random_uniform"}
    targets = pd.read_csv(tmp_path / "cgan_length_targets.csv")
    assert list(targets.columns) == ["epoch", "mode", *TARGET_COLUMNS]
    assert len(targets) == 2 * 2 * 5 and set(targets["target"]) == {9, 13, 20, 21, 40}
    fasta = (tmp_path / "cgan_length_generated_targets_sample.fasta").read_text(encoding="utf-8")
    assert ">gen_t09_0000" in fasta and ">gen_t40_0049" in fasta and fasta.count(">") == 5 * 50
    info = json.loads((tmp_path / "cgan_length_run_info.json").read_text(encoding="utf-8"))
    assert info["keras_backend"] == backend and info["variant"] == "cgan_length"
    assert info["config"]["conditions"] == ["length"] and info["target_lengths"] == [9, 13, 20, 21, 40]
    assert info["usual_length_range_p10_p90"] == [9, 40] and info["hit_tolerance"] == 2
    assert MAX_LEN == 60

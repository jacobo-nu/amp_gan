"""Tests de la WGAN-GP (``amp_gan.wgan_gp``): modelos pequeños y pocos pasos, en CPU.

Los tests que dependen del backend (penalización de gradiente, entrenamiento,
determinismo, script de ejecución) se lanzan en un subproceso con ``KERAS_BACKEND``
fijado y la GPU oculta, y se saltan si el backend no está instalado. El resto
usa solo ``keras.ops`` y funciona con el backend por defecto del proceso.
"""

import json
import os
import subprocess
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from amp_gan import run_gan_base, runtime

runtime.use_cpu_only()
keras = pytest.importorskip("keras")

from amp_gan import wgan_gp  # noqa: E402
from amp_gan.encoding import SEQ_LEN, VOCAB_SIZE, encode_sequences  # noqa: E402
from amp_gan.evaluation import METRIC_NAMES  # noqa: E402
from amp_gan.experiment import run_gan_experiment  # noqa: E402
from amp_gan.gan import GANConfig, build_generator  # noqa: E402
from amp_gan.wgan_gp import (  # noqa: E402
    WGANConfig,
    build_critic,
    build_wgan_models,
    generator_loss,
    gradient_penalty_from_gradients,
    interpolate,
    train_wgan_gp,
    wasserstein_critic_loss,
)

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
_BACKENDS = ("tensorflow", "torch")
_BATCH = 8


def _small(**kw) -> WGANConfig:
    return WGANConfig(gen_hidden=(16,), critic_hidden=(16,), batch_size=_BATCH, **kw)


@pytest.fixture
def real_data():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, VOCAB_SIZE, size=(5 * _BATCH, SEQ_LEN))
    return np.eye(VOCAB_SIZE, dtype="float32")[idx]


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


# --------------------------------------------------------------------------- #
# Configuración y modelos
# --------------------------------------------------------------------------- #
def test_default_config_matches_specification():
    c = WGANConfig()
    assert (c.latent_dim, c.seq_len, c.vocab_size) == (100, 60, 21)
    assert (c.gen_hidden, c.critic_hidden) == ((256, 512), (512, 256))
    assert (c.batch_size, c.seed, c.leaky_alpha) == (128, 42, 0.2)
    assert (c.gp_weight, c.n_critic) == (10.0, 5)
    assert (c.lr, c.beta_1, c.beta_2) == (1e-4, 0.0, 0.9)


def test_invalid_config_values_raise():
    for bad in ({"n_critic": 0}, {"batch_size": 0}, {"gp_weight": -1.0}, {"lr": 0.0}):
        with pytest.raises(ValueError):
            WGANConfig(**bad)


def test_generator_config_is_compatible_with_the_other_gans():
    gc = WGANConfig().generator_config()
    ref = GANConfig()
    for field in ("latent_dim", "seq_len", "vocab_size", "gen_hidden", "leaky_alpha", "batch_size", "seed"):
        assert getattr(gc, field) == getattr(ref, field), field


def test_generator_is_the_shared_generator():
    gen, _ = build_wgan_models(WGANConfig())
    ref = build_generator(GANConfig())
    assert [tuple(w.shape) for w in gen.weights] == [tuple(w.shape) for w in ref.weights]
    assert [type(layer).__name__ for layer in gen.layers] == [type(layer).__name__ for layer in ref.layers]
    assert tuple(gen.output_shape) == (None, 60, 21)
    out = _np(gen(np.random.default_rng(1).standard_normal((3, 100)).astype("float32")))
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-5)


def test_critic_architecture_is_linear_without_bn_dropout_or_noise():
    critic = build_critic(WGANConfig())
    kinds = [type(layer).__name__ for layer in critic.layers if type(layer).__name__ != "InputLayer"]
    assert kinds == ["Flatten", "Dense", "LeakyReLU", "Dense", "LeakyReLU", "Dense"]
    dense = [layer for layer in critic.layers if type(layer).__name__ == "Dense"]
    assert [layer.units for layer in dense] == [512, 256, 1]
    assert keras.activations.serialize(dense[-1].activation) == "linear"
    assert all(keras.activations.serialize(layer.activation) == "linear" for layer in dense)
    banned = ("BatchNormalization", "Dropout", "GaussianNoise", "InstanceNoise", "Softmax", "Activation")
    assert not set(kinds) & set(banned)
    assert tuple(critic.output_shape) == (None, 1)


def test_critic_output_is_unbounded():
    config = _small()
    critic = build_critic(config)
    dense_last = [layer for layer in critic.layers if type(layer).__name__ == "Dense"][-1]
    kernel, bias = dense_last.get_weights()
    dense_last.set_weights([kernel, bias + 50.0])  # una sigmoide la dejaría en (0, 1)
    out = _np(critic(np.zeros((4, SEQ_LEN, VOCAB_SIZE), dtype="float32")))
    assert out.shape == (4, 1) and np.all(out > 40.0)
    dense_last.set_weights([kernel, bias - 50.0])
    assert np.all(_np(critic(np.zeros((4, SEQ_LEN, VOCAB_SIZE), dtype="float32"))) < -40.0)


# --------------------------------------------------------------------------- #
# Pérdidas: signos y formas
# --------------------------------------------------------------------------- #
def test_loss_signs_and_shapes():
    d_real = np.array([[2.0], [4.0]], dtype="float32")  # media 3
    d_fake = np.array([[-1.0], [-3.0]], dtype="float32")  # media -2
    wl = wasserstein_critic_loss(d_real, d_fake)
    assert tuple(wl.shape) == ()
    assert float(_np(wl)) == pytest.approx(-2.0 - 3.0)  # mean(fake) - mean(real): negativa si D separa bien
    gl = generator_loss(d_fake)
    assert tuple(gl.shape) == ()
    assert float(_np(gl)) == pytest.approx(2.0)  # -mean(D(G(z)))
    # Si D puntúa más alto lo falso que lo real, la pérdida del crítico es positiva.
    assert float(_np(wasserstein_critic_loss(d_fake, d_real))) == pytest.approx(5.0)


def test_gradient_penalty_from_gradients_values():
    unit = np.zeros((4, 3, 2), dtype="float32")
    unit[:, 0, 0] = 1.0  # norma 1 -> penalización 0
    assert float(_np(gradient_penalty_from_gradients(unit))) == pytest.approx(0.0, abs=1e-6)
    grads = np.zeros((2, 3, 2), dtype="float32")
    grads[0, 0, 0] = 3.0  # norma 3 -> (3 - 1)^2 = 4
    grads[1, 1, 1] = 1.0  # norma 1 -> 0
    gp = gradient_penalty_from_gradients(grads)
    assert tuple(gp.shape) == ()
    assert float(_np(gp)) == pytest.approx(2.0, abs=1e-5)  # media de 4 y 0
    assert np.isfinite(float(_np(gradient_penalty_from_gradients(np.zeros((2, 3, 2), "float32")))))  # sin NaN con norma 0


def test_interpolate_endpoints_and_shape():
    real = np.ones((3, 4, 2), dtype="float32")
    fake = np.zeros((3, 4, 2), dtype="float32")
    eps = np.array([0.0, 0.25, 1.0], dtype="float32").reshape(3, 1, 1)
    out = _np(interpolate(real, fake, eps))
    assert out.shape == (3, 4, 2)
    np.testing.assert_allclose(out[:, 0, 0], [0.0, 0.25, 1.0])


# --------------------------------------------------------------------------- #
# Backend no soportado
# --------------------------------------------------------------------------- #
def test_unsupported_backend_raises_clear_error(monkeypatch, real_data):
    monkeypatch.setattr(wgan_gp, "_backend_name", lambda: "jax")
    with pytest.raises(NotImplementedError, match="jax"):
        train_wgan_gp(real_data, _small(), 1)
    critic = build_critic(_small())
    with pytest.raises(NotImplementedError, match="tensorflow"):
        wgan_gp.gradient_penalty(critic, real_data[:2], real_data[:2], np.full((2, 1, 1), 0.5, "float32"))


def test_backend_specific_imports_are_lazy():
    """Ni tensorflow ni torch se importan a nivel de módulo (solo dentro de las funciones de backend)."""
    import ast

    tree = ast.parse((_SRC / "amp_gan" / "wgan_gp.py").read_text(encoding="utf-8"))
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert not top & {"tensorflow", "torch", "jax"}
    inner = {a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names}
    assert {"tensorflow", "torch"} <= inner  # sí aparecen, dentro de las funciones


# --------------------------------------------------------------------------- #
# Definición de época (con el backend por defecto del proceso)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(keras.backend.backend() not in _BACKENDS, reason="backend por defecto no soportado")
def test_epoch_definition_and_generator_cadence(real_data):
    # 40 muestras, lote 8: 5 pasos de crítico por época. n_critic = 3, contador global entre épocas:
    # época 1 -> pasos 1-5 (actualiza en el 3), época 2 -> pasos 6-10 (en el 6 y el 9).
    called = []
    h = train_wgan_gp(real_data, _small(n_critic=3), 2, callback=lambda e, g: called.append(e))
    assert h["critic_steps"] == [5, 5] and h["generator_steps"] == [1, 2]
    assert called == [0, 1]
    assert h.generator is not None and h.critic is not None
    assert set(h) == {"critic_loss", "wasserstein_estimate", "gradient_penalty", "g_loss", "critic_steps", "generator_steps"}


@pytest.mark.skipif(keras.backend.backend() not in _BACKENDS, reason="backend por defecto no soportado")
def test_max_steps_and_epoch_without_generator_update(real_data):
    h = train_wgan_gp(real_data, _small(n_critic=5), 2, max_steps_per_epoch=3)
    assert h["critic_steps"] == [3, 3] and h["generator_steps"] == [0, 1]  # la 5.ª pasada cae en la época 2
    assert np.isnan(h["g_loss"][0]) and np.isfinite(h["g_loss"][1])
    with pytest.raises(ValueError):
        train_wgan_gp(real_data[:3], _small(), 1)


@pytest.mark.skipif(keras.backend.backend() not in _BACKENDS, reason="backend por defecto no soportado")
def test_experiment_accepts_wgan_train_fn():
    rng = np.random.default_rng(0)
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    seqs = sorted({"".join(rng.choice(alphabet, size=int(rng.integers(5, 30)))) for _ in range(80)})
    import pandas as pd

    X, ref = encode_sequences(seqs), pd.DataFrame({"id": range(len(seqs)), "sequence": seqs})
    res = run_gan_experiment(X, ref, _small(n_critic=2), epochs=2, snapshot_every=20, n_eval=30, eval_seed=1,
                             max_steps_per_epoch=3, train_fn=train_wgan_gp)  # fmt: skip
    assert list(res.metrics.columns) == ["epoch", "mode", *METRIC_NAMES]
    assert sorted(res.metrics["epoch"].unique()) == [1, 2] and set(res.metrics["mode"]) == {"argmax", "sample"}
    assert (res.metrics["n"] == 30).all()
    assert len(res.history["critic_loss"]) == 2 and len(res.epoch_seconds) == 2
    assert all(len(v) == 30 for v in res.sequences.values())


def test_run_gan_base_wgan_variant_registered():
    assert run_gan_base.VARIANTS["wgan_gp"] == "wgan_gp"
    assert {"gan_base", "gan_stab", "gan_stabc"} <= set(run_gan_base.VARIANTS.values())
    args = run_gan_base._parse(["--variant", "wgan_gp", "--max-steps", "3"])
    assert args.variant == "wgan_gp" and args.max_steps == 3 and args.epochs == 200
    assert run_gan_base._parse([]).max_steps is None
    assert run_gan_base.make_config("wgan_gp", seed=7) == WGANConfig(seed=7)
    assert run_gan_base.make_config("base").seed == 42 and isinstance(run_gan_base.make_config("base"), GANConfig)


# --------------------------------------------------------------------------- #
# Tests por backend, en subproceso
# --------------------------------------------------------------------------- #
def _installed(backend: str) -> bool:
    return backend in runtime.installed_backends()


def _run_snippet(backend: str, snippet: str, *args: str, timeout: int = 900) -> dict:
    env = dict(os.environ, KERAS_BACKEND=backend, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
    code = "import sys; sys.path.insert(0, %r)\n" % str(_SRC) + textwrap.dedent(snippet)
    proc = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, env=env, timeout=timeout)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT="))
    return json.loads(line[len("RESULT="):])


_GP_SNIPPET = """
import json
import numpy as np
import keras
from amp_gan import wgan_gp as w

rng = np.random.default_rng(0)
kernel = (rng.standard_normal((60 * 21, 1)) * 0.05).astype("float32")  # f(x) = w . x (sin sesgo)
inp = keras.Input(shape=(60, 21))
out = keras.layers.Dense(1, use_bias=False)(keras.layers.Flatten()(inp))
critic = keras.Model(inp, out)
critic.layers[-1].set_weights([kernel])

n = 6
real = np.eye(21, dtype="float32")[rng.integers(0, 21, (n, 60))]
fake = rng.dirichlet(np.ones(21), size=(n, 60)).astype("float32")
eps = rng.uniform(size=(n, 1, 1)).astype("float32")
gp = float(keras.ops.convert_to_numpy(w.gradient_penalty(critic, real, fake, eps)))
loss, wass, gp2 = w._critic_losses(critic, real, fake, eps, 10.0)
score = lambda x: x.reshape(len(x), -1).astype("float64") @ kernel.astype("float64")
print("RESULT=" + json.dumps({
    "backend": keras.backend.backend(),
    "gp": gp, "gp_in_loss": float(keras.ops.convert_to_numpy(gp2)),
    "loss": float(keras.ops.convert_to_numpy(loss)), "wass": float(keras.ops.convert_to_numpy(wass)),
    "norm_w": float(np.linalg.norm(kernel.astype("float64"))),
    "mean_real": float(score(real).mean()), "mean_fake": float(score(fake).mean()),
}))
"""


@lru_cache(maxsize=None)
def _gp_result(backend: str) -> dict:
    return _run_snippet(backend, _GP_SNIPPET)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_gradient_penalty_exact_for_linear_critic(backend):
    if not _installed(backend):
        pytest.skip(f"{backend} no está instalado")
    r = _gp_result(backend)
    assert r["backend"] == backend
    expected = (r["norm_w"] - 1.0) ** 2  # el gradiente de w.x es w en todo punto
    assert r["gp"] == pytest.approx(expected, rel=1e-5)
    assert r["gp_in_loss"] == pytest.approx(expected, rel=1e-5)
    assert r["wass"] == pytest.approx(r["mean_real"] - r["mean_fake"], rel=1e-4, abs=1e-6)
    assert r["loss"] == pytest.approx(r["mean_fake"] - r["mean_real"] + 10.0 * expected, rel=1e-4, abs=1e-5)


def test_gradient_penalty_same_value_in_both_backends():
    if not all(_installed(b) for b in _BACKENDS):
        pytest.skip("hacen falta tensorflow y torch")
    a, b = _gp_result("tensorflow"), _gp_result("torch")
    assert a["gp"] == pytest.approx(b["gp"], rel=1e-6)
    assert a["loss"] == pytest.approx(b["loss"], rel=1e-5, abs=1e-6)


_TRAIN_SNIPPET = """
import json
import numpy as np
import keras
from amp_gan import wgan_gp as w
from amp_gan.encoding import SEQ_LEN, VOCAB_SIZE
from amp_gan.gan import generate_probs, sample_sequences
from amp_gan.data import STANDARD_ALPHABET

rng = np.random.default_rng(0)
X = np.eye(VOCAB_SIZE, dtype="float32")[rng.integers(0, VOCAB_SIZE, (40, SEQ_LEN))]
cfg = w.WGANConfig(gen_hidden=(16,), critic_hidden=(16,), batch_size=8, n_critic=2)

h1 = w.train_wgan_gp(X, cfg, 2, max_steps_per_epoch=4)
h2 = w.train_wgan_gp(X, cfg, 2, max_steps_per_epoch=4)
probs = generate_probs(h1.generator, 7, cfg)
seqs = sample_sequences(probs, rng=np.random.default_rng(1), mode="sample") + sample_sequences(probs, mode="argmax")

# Cada paso actualiza solo sus pesos: el del crítico no toca el generador y viceversa.
gen, critic = w.build_wgan_models(cfg)
g_opt, c_opt = w._optimizer(cfg), w._optimizer(cfg)
snap = lambda m: [np.array(keras.ops.convert_to_numpy(v)) for v in m.weights]
differs = lambda a, b: any(not np.array_equal(x, y) for x, y in zip(a, b))
g0, c0 = snap(gen), snap(critic)
w.critic_step(gen, critic, c_opt, X[:8], cfg)
g1, c1 = snap(gen), snap(critic)
w.generator_step(gen, critic, g_opt, cfg)
g2, c2 = snap(gen), snap(critic)

print("RESULT=" + json.dumps({
    "backend": keras.backend.backend(),
    "history": {k: list(v) for k, v in h1.items()},
    "history2": {k: list(v) for k, v in h2.items()},
    "probs_shape": list(probs.shape), "probs_sum_ok": bool(np.allclose(probs.sum(-1), 1.0, atol=1e-5)),
    "seqs_ok": all(isinstance(s, str) and set(s) <= set(STANDARD_ALPHABET) and len(s) <= SEQ_LEN for s in seqs),
    "critic_step_changes_critic_only": bool(differs(c0, c1) and not differs(g0, g1)),
    "generator_step_changes_generator_only": bool(differs(g1, g2) and not differs(c1, c2)),
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
    assert set(h) == {"critic_loss", "wasserstein_estimate", "gradient_penalty", "g_loss", "critic_steps", "generator_steps"}
    assert all(len(v) == 2 for v in h.values())
    assert all(np.all(np.isfinite(h[k])) for k in h), h
    assert h["critic_steps"] == [4, 4] and h["generator_steps"] == [2, 2]  # 8 pasos de crítico, n_critic = 2
    assert np.all(np.asarray(h["gradient_penalty"]) >= 0)
    np.testing.assert_allclose(h["wasserstein_estimate"], -(np.asarray(h["critic_loss"]) - 10.0 * np.asarray(h["gradient_penalty"])),
                               rtol=1e-4, atol=1e-5)  # fmt: skip
    assert r["probs_shape"] == [7, 60, 21] and r["probs_sum_ok"] and r["seqs_ok"]
    assert r["critic_step_changes_critic_only"] and r["generator_step_changes_generator_only"]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_training_is_deterministic(backend):
    if not _installed(backend):
        pytest.skip(f"{backend} no está instalado")
    r = _train_result(backend)
    assert r["history"] == r["history2"]  # misma semilla -> mismo historial (exacto, en CPU)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_run_gan_base_wgan_variant_writes_expected_files(backend, tmp_path):
    if not _installed(backend):
        pytest.skip(f"{backend} no está instalado")
    if not run_gan_base.TRAIN_FASTA.exists():
        pytest.skip("faltan los datos limpios")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
    env.pop("KERAS_BACKEND", None)
    env.pop(runtime.BACKEND_ENV_VAR, None)
    cmd = [sys.executable, str(_SRC / "amp_gan" / "run_gan_base.py"), "--variant", "wgan_gp", "--epochs", "2",
           "--max-steps", "6", "--n-eval", "50", "--backend", backend, "--out-dir", str(tmp_path)]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["wgan_gp_generated_argmax.fasta", "wgan_gp_generated_sample.fasta", "wgan_gp_history.csv",
                     "wgan_gp_metrics.csv", "wgan_gp_run_info.json"]  # fmt: skip
    import pandas as pd

    hist = pd.read_csv(tmp_path / "wgan_gp_history.csv")
    assert list(hist.columns) == ["epoch", "critic_loss", "wasserstein_estimate", "gradient_penalty", "g_loss",
                                  "critic_steps", "generator_steps"]  # fmt: skip
    assert list(hist["epoch"]) == [1, 2] and np.isfinite(hist.drop(columns="g_loss").to_numpy()).all()
    metrics = pd.read_csv(tmp_path / "wgan_gp_metrics.csv")
    assert list(metrics.columns) == ["epoch", "mode", *METRIC_NAMES]
    assert set(metrics["mode"]) == {"argmax", "sample", "real_subset", "random_uniform"}
    info = json.loads((tmp_path / "wgan_gp_run_info.json").read_text(encoding="utf-8"))
    assert info["keras_backend"] == backend and info["variant"] == "wgan_gp"
    assert info["config"]["n_critic"] == 5 and info["config"]["gp_weight"] == 10.0 and info["max_steps_per_epoch"] == 6
    assert (SEQ_LEN, VOCAB_SIZE) == (60, 21)

"""Tests de la GAN base (``amp_gan.gan``): modelos pequeños, pocos pasos, en CPU.

Se ocultan las GPU antes de importar Keras y los tests se saltan si Keras no
está disponible. Funcionan con cualquier backend de Keras 3.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from amp_gan import runtime

runtime.use_cpu_only()
keras = pytest.importorskip("keras")

from amp_gan.data import STANDARD_ALPHABET  # noqa: E402
from amp_gan.encoding import SEQ_LEN, VOCAB_SIZE  # noqa: E402
from amp_gan.gan import (  # noqa: E402
    GANConfig,
    GumbelStraightThrough,
    InstanceNoise,
    build_discriminator,
    build_gan,
    build_generator,
    build_models,
    generate_probs,
    gumbel_softmax_straight_through,
    instance_noise_std_for_epoch,
    sample_sequences,
    train_gan,
    train_step,
)

_BATCH = 8


@pytest.fixture
def config():
    """Modelo pequeño para que los tests sean rápidos (la arquitectura es la misma)."""
    return GANConfig(gen_hidden=(16,), disc_hidden=(16,), batch_size=_BATCH)


@pytest.fixture
def real_data():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, VOCAB_SIZE, size=(4 * _BATCH, SEQ_LEN))
    return np.eye(VOCAB_SIZE, dtype="float32")[idx]


def _weights(model):
    return [np.array(keras.ops.convert_to_numpy(w)) for w in model.weights]


def _changed(before, after):
    return any(not np.array_equal(a, b) for a, b in zip(before, after))


def test_default_config_matches_specification():
    c = GANConfig()
    assert (c.latent_dim, c.seq_len, c.vocab_size) == (100, 60, 21)
    assert (c.gen_hidden, c.disc_hidden) == ((256, 512), (512, 256))
    assert (c.lr, c.beta_1, c.batch_size, c.seed) == (2e-4, 0.5, 128, 42)
    assert (c.leaky_alpha, c.dropout) == (0.2, 0.3)


def test_generator_shape_and_softmax(config):
    gen = build_generator(config)
    z = np.random.default_rng(1).standard_normal((5, config.latent_dim)).astype("float32")
    out = keras.ops.convert_to_numpy(gen(z))
    assert out.shape == (5, 60, 21)
    assert np.all(out >= 0)
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-5)


def test_discriminator_output_range_and_shape(config, real_data):
    disc = build_discriminator(config)
    out = keras.ops.convert_to_numpy(disc(real_data[:5]))
    assert out.shape == (5, 1)
    assert np.all((out >= 0) & (out <= 1))


def test_discriminator_frozen_in_combined_model(config):
    gen, disc, gan = build_models(config)
    assert disc.trainable is False
    assert len(gan.trainable_weights) == len(gen.trainable_weights) > 0
    gen_ids = {id(w) for w in gen.trainable_weights}
    assert {id(w) for w in gan.trainable_weights} == gen_ids
    assert not any(id(w) in {id(v) for v in gan.trainable_weights} for w in disc.weights)


def test_train_step_updates_only_the_right_weights(config, real_data, monkeypatch):
    """En el paso de D solo cambia D; en el paso de G (modelo combinado) solo cambia G."""
    keras.utils.set_random_seed(config.seed)
    gen, disc, gan = build_models(config)
    log = []

    def spy(name, model, fn):
        def wrapped(*args, **kwargs):
            before_g, before_d = _weights(gen), _weights(disc)
            out = fn(*args, **kwargs)
            log.append((name, _changed(before_g, _weights(gen)), _changed(before_d, _weights(disc))))
            return out

        return wrapped

    monkeypatch.setattr(disc, "train_on_batch", spy("D", disc, disc.train_on_batch))
    monkeypatch.setattr(gan, "train_on_batch", spy("G", gan, gan.train_on_batch))

    losses = train_step(gen, disc, gan, real_data[:_BATCH], config)
    assert set(losses) == {"d_loss_real", "d_loss_fake", "g_loss"}
    assert all(np.isfinite(v) for v in losses.values())
    # (nombre, G cambió, D cambió): dos llamadas de D (real y falsa) y una de G.
    assert log == [("D", False, True), ("D", False, True), ("G", True, False)]
    # Y sigue siendo así en pasos posteriores.
    log.clear()
    train_step(gen, disc, gan, real_data[:_BATCH], config)
    assert log == [("D", False, True), ("D", False, True), ("G", True, False)]


def test_train_gan_history_and_finite_losses(config, real_data):
    hist = train_gan(real_data, config, epochs=2, max_steps_per_epoch=2)
    for key in ("d_loss_real", "d_loss_fake", "d_loss", "g_loss"):
        assert len(hist[key]) == 2
        assert np.all(np.isfinite(hist[key]))
    assert hist.generator is not None and hist.discriminator is not None


def test_train_gan_callback_called_each_epoch(config, real_data):
    seen = []
    train_gan(
        real_data, config, epochs=3, max_steps_per_epoch=1,
        callback=lambda epoch, gen: seen.append((epoch, gen.name)),
    )  # fmt: skip
    assert seen == [(0, "generator"), (1, "generator"), (2, "generator")]


def test_train_gan_requires_enough_samples(config, real_data):
    with pytest.raises(ValueError, match="batch_size"):
        train_gan(real_data[: _BATCH - 1], config, epochs=1)


def test_training_is_reproducible_on_cpu(config, real_data):
    """Misma semilla -> mismo historial; en CPU es exacto (bit a bit) en tensorflow y torch."""
    h1 = train_gan(real_data, config, epochs=2, max_steps_per_epoch=3)
    h2 = train_gan(real_data, config, epochs=2, max_steps_per_epoch=3)
    for key in ("d_loss_real", "d_loss_fake", "g_loss"):
        assert h1[key] == h2[key]


def test_generate_probs_is_reproducible(config):
    gen = build_generator(config)
    a = generate_probs(gen, 7, config, seed=3)
    b = generate_probs(gen, 7, config, seed=3)
    c = generate_probs(gen, 7, config, seed=4)
    assert a.shape == (7, 60, 21)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("mode", ["argmax", "sample"])
def test_sample_sequences_valid_strings(config, mode):
    gen = build_generator(config)
    probs = generate_probs(gen, 20, config, seed=0)
    seqs = sample_sequences(probs, np.random.default_rng(0), mode=mode)
    assert len(seqs) == 20
    for s in seqs:
        assert set(s) <= set(STANDARD_ALPHABET)
        assert len(s) <= SEQ_LEN


def test_sample_mode_follows_the_distribution_and_needs_rng():
    # Todas las posiciones: 100 % de probabilidad en el aa de índice 0 ('A').
    probs = np.zeros((3, 60, 21), dtype="float32")
    probs[..., 0] = 1.0
    assert sample_sequences(probs, np.random.default_rng(0), mode="sample") == ["A" * 60] * 3
    with pytest.raises(ValueError, match="rng"):
        sample_sequences(probs, None, mode="sample")
    with pytest.raises(ValueError, match="mode"):
        sample_sequences(probs, np.random.default_rng(0), mode="otro")


# ---------------------------------------------------------------------------
# Variante estabilizada y no regresión de la vanilla pura
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _ROOT / "context" / "data" / "processed"

# Subproceso que reproduce el entrenamiento de run_gan_base.py (backend y
# dispositivo de la ejecución original, configuración por defecto, semilla 42) durante 10 épocas.
# Recibe la raíz del proyecto, el backend y el dispositivo ("cpu" o "gpu").
_REPRO_SNIPPET = """
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / "src"))
from amp_gan import runtime
runtime.set_backend(sys.argv[2])
if sys.argv[3] == "cpu":
    runtime.use_cpu_only()
from amp_gan.encoding import load_encoded
from amp_gan.gan import GANConfig, train_gan
env = runtime.describe_environment()
X, _ = load_encoded(root / "context/data/processed/Antibacterial_amps_clean.fasta")
h = train_gan(X, GANConfig(seed=42), epochs=10)
print("RESULT=" + json.dumps({
    "backend": env["keras_backend"], "devices": env["keras_devices"],
    "keras": env["keras_version"], "torch": env["torch_version"],
    "history": {k: ["%.8g" % v for v in h[k]] for k in ("d_loss_real", "d_loss_fake", "d_loss", "g_loss")},
}))
"""


def test_default_config_reproduces_base_history_exactly():
    """Con ``GANConfig()`` las 10 primeras épocas son idénticas a ``gan_base_history.csv``.

    Se ejecuta en un subproceso con el backend y el dispositivo de la ejecución
    original (el de ``gan_base_run_info.json``), porque la igualdad exacta solo
    tiene sentido en el mismo entorno. Se salta si faltan los datos (no se
    versionan), si el backend original no está instalado o si el entorno de esta
    máquina no coincide con el de ``gan_base_run_info.json`` (por ejemplo, sin GPU).
    """
    pd = pytest.importorskip("pandas")
    needed = [
        _PROCESSED / "gan_base_history.csv",
        _PROCESSED / "gan_base_run_info.json",
        _PROCESSED / "Antibacterial_amps_clean.fasta",
    ]
    missing = [f.name for f in needed if not f.is_file()]
    if missing:
        pytest.skip(f"Faltan datos generados (no se versionan): {missing}")
    info = json.loads((_PROCESSED / "gan_base_run_info.json").read_text(encoding="utf-8"))
    ref = pd.read_csv(_PROCESSED / "gan_base_history.csv", dtype=str).head(10)
    if info["keras_backend"] not in runtime.installed_backends():
        pytest.skip(f"Backend original no instalado: {info['keras_backend']}")

    env = dict(os.environ)
    env.pop("KERAS_BACKEND", None)
    if info.get("cuda_visible_devices") is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)  # use_cpu_only() de este módulo la fijó a ""
    proc = subprocess.run(
        [sys.executable, "-c", _REPRO_SNIPPET, str(_ROOT), info["keras_backend"], info["device_used"]],
        capture_output=True, text=True, env=env, timeout=600,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT="))
    res = json.loads(line[len("RESULT="):])

    expected_env = (info["keras_backend"], info["keras_devices"], info["versions"]["keras_version"],
                    info["versions"]["torch_version"])  # fmt: skip
    if (res["backend"], res["devices"], res["keras"], res["torch"]) != expected_env:
        pytest.skip(f"Entorno distinto al de la ejecución original: {res} vs {expected_env}")
    for key, values in res["history"].items():
        assert values == list(ref[key]), key


def _legacy_train_step(generator, discriminator, gan, real_batch, config):
    """Copia literal del ``train_step`` de la vanilla pura (antes de las correcciones)."""
    n = int(real_batch.shape[0])
    ones = np.ones((n, 1), dtype="float32")
    zeros = np.zeros((n, 1), dtype="float32")
    z = keras.random.normal((n, config.latent_dim))
    fake_batch = keras.ops.convert_to_numpy(generator(z, training=False))
    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(real_batch, ones)
    d_loss_fake = discriminator.train_on_batch(fake_batch, zeros)
    discriminator.trainable = False
    z = keras.random.normal((n, config.latent_dim))
    g_loss = gan.train_on_batch(z, ones)
    return {"d_loss_real": d_loss_real, "d_loss_fake": d_loss_fake, "g_loss": g_loss}


def test_default_config_train_step_equals_legacy_step_exactly(config, real_data):
    """Rápido (CPU): con la configuración por defecto el paso es idéntico al de la vanilla pura."""
    runs = []
    for step_fn in (train_step, _legacy_train_step):
        keras.utils.set_random_seed(config.seed)
        gen, disc, gan = build_models(config)
        out = [step_fn(gen, disc, gan, real_data[i * _BATCH : (i + 1) * _BATCH], config) for i in range(4)]
        runs.append((out, _weights(gen), _weights(disc)))
    (out_a, g_a, d_a), (out_b, g_b, d_b) = runs
    for a, b in zip(out_a, out_b):
        for key in ("d_loss_real", "d_loss_fake", "g_loss"):
            assert float(np.asarray(a[key]).reshape(-1)[0]) == float(np.asarray(b[key]).reshape(-1)[0])
    assert not _changed(g_a, g_b) and not _changed(d_a, d_b)


def test_default_config_adds_nothing_to_the_models(config):
    c = GANConfig()
    assert (c.real_label, c.instance_noise_std, c.instance_noise_decay_epochs) == (1.0, 0.0, 100)
    assert (c.discretize, c.gumbel_tau) == ("none", 1.0)
    gen, disc, gan = build_models(config)
    assert not any(isinstance(layer, InstanceNoise) for layer in disc.layers)
    assert not any(isinstance(layer, GumbelStraightThrough) for layer in gan.layers)
    assert [layer.name for layer in gan.layers] == ["z_gan", "generator", "discriminator"]


def test_stabilized_config_values_and_validation():
    c = GANConfig.stabilized(seed=7)
    assert (c.real_label, c.instance_noise_std, c.instance_noise_decay_epochs) == (0.9, 0.1, 100)
    assert (c.discretize, c.gumbel_tau, c.seed) == ("gumbel_st", 1.0, 7)
    # El resto coincide con la vanilla pura.
    base = GANConfig(seed=7)
    for field in ("latent_dim", "gen_hidden", "disc_hidden", "lr", "beta_1", "batch_size", "dropout"):
        assert getattr(c, field) == getattr(base, field)
    with pytest.raises(ValueError, match="discretize"):
        GANConfig(discretize="otro")
    with pytest.raises(ValueError, match="gumbel_tau"):
        GANConfig(gumbel_tau=0.0)
    with pytest.raises(ValueError, match="real_label"):
        GANConfig(real_label=1.5)


def _record_labels(disc, gan, monkeypatch):
    labels = []

    def spy(model, fn):
        def wrapped(x, y, *args, **kwargs):
            labels.append(np.asarray(y))
            return fn(x, y, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(disc, "train_on_batch", spy(disc, disc.train_on_batch))
    monkeypatch.setattr(gan, "train_on_batch", spy(gan, gan.train_on_batch))
    return labels


def test_one_sided_label_smoothing(real_data, monkeypatch):
    small = dict(gen_hidden=(16,), disc_hidden=(16,), batch_size=_BATCH)
    for cfg, real_value in ((GANConfig(**small), 1.0), (GANConfig.stabilized(**small), 0.9)):
        gen, disc, gan = build_models(cfg)
        labels = _record_labels(disc, gan, monkeypatch)
        train_step(gen, disc, gan, real_data[:_BATCH], cfg)
        real, fake, gen_step = labels
        assert real.shape == (_BATCH, 1)
        np.testing.assert_array_equal(real, np.float32(real_value))  # solo cambian las reales
        np.testing.assert_array_equal(fake, 0.0)
        np.testing.assert_array_equal(gen_step, 1.0)  # en el paso de G sigue siendo 1


def test_instance_noise_schedule_is_linear_and_zero_after_decay():
    c = GANConfig.stabilized()
    vals = [instance_noise_std_for_epoch(c, e) for e in range(0, 130)]
    assert vals[0] == pytest.approx(0.1)
    assert vals[50] == pytest.approx(0.05)
    assert vals[99] == pytest.approx(0.001)
    assert vals[100] == 0.0 and all(v == 0.0 for v in vals[100:])
    np.testing.assert_allclose(np.diff(vals[:101]), -0.1 / 100, atol=1e-12)  # lineal
    # No se aplica con la configuración por defecto.
    assert all(instance_noise_std_for_epoch(GANConfig(), e) == 0.0 for e in range(0, 130))


def test_instance_noise_layer_adds_noise_only_in_training():
    layer = InstanceNoise(0.1, seed=0)
    x = np.zeros((64, 60, 21), dtype="float32")
    noisy = keras.ops.convert_to_numpy(layer(x, training=True))
    assert noisy.std() == pytest.approx(0.1, rel=0.05)
    assert abs(noisy.mean()) < 0.01
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(layer(x, training=False)), x)
    layer.set_std(0.0)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(layer(x, training=True)), x)


def test_train_gan_sets_the_noise_std_each_epoch(real_data):
    cfg = GANConfig(
        gen_hidden=(16,), disc_hidden=(16,), batch_size=_BATCH,
        instance_noise_std=0.1, instance_noise_decay_epochs=4,
    )  # fmt: skip
    models = build_models(cfg)
    layer = next(m for m in models[1].layers if isinstance(m, InstanceNoise))
    seen = []
    train_gan(
        real_data, cfg, epochs=6, models=models, max_steps_per_epoch=1,
        callback=lambda epoch, gen: seen.append(float(keras.ops.convert_to_numpy(layer.std))),
    )  # fmt: skip
    np.testing.assert_allclose(seen, [0.1, 0.075, 0.05, 0.025, 0.0, 0.0], atol=1e-6)


def test_noise_acts_in_both_discriminator_and_generator_steps(real_data):
    """El ruido de D se aplica también dentro del modelo combinado (paso de G)."""
    cfg = GANConfig(gen_hidden=(16,), disc_hidden=(16,), batch_size=_BATCH, instance_noise_std=0.5)
    _, disc, gan = build_models(cfg)
    z = np.random.default_rng(0).standard_normal((_BATCH, cfg.latent_dim)).astype("float32")
    a = keras.ops.convert_to_numpy(gan(z, training=True))
    b = keras.ops.convert_to_numpy(gan(z, training=True))
    assert not np.array_equal(a, b)  # dos pasadas con ruido distinto
    c = keras.ops.convert_to_numpy(gan(z, training=False))
    d = keras.ops.convert_to_numpy(gan(z, training=False))
    np.testing.assert_array_equal(c, d)  # sin ruido en inferencia


def test_gumbel_straight_through_forward_is_exact_one_hot():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((50, 60, 21)).astype("float32") * 3
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    probs[0, 0] = np.eye(21, dtype="float32")[3]  # probabilidades exactamente 0 y 1
    out = keras.ops.convert_to_numpy(gumbel_softmax_straight_through(probs, 1.0, keras.random.SeedGenerator(0)))
    assert out.shape == probs.shape
    assert set(np.unique(out)) <= {0.0, 1.0}
    np.testing.assert_array_equal(out.sum(axis=-1), 1.0)
    np.testing.assert_array_equal(out[0, 0], np.eye(21, dtype="float32")[3])


def test_gumbel_samples_follow_the_categorical_distribution():
    p = np.array([0.7, 0.2, 0.1], dtype="float32")
    probs = np.tile(p, (20000, 1))
    out = keras.ops.convert_to_numpy(gumbel_softmax_straight_through(probs, 1.0, keras.random.SeedGenerator(1)))
    np.testing.assert_allclose(out.mean(axis=0), p, atol=0.02)


def test_discretization_only_between_g_and_d_and_gradient_reaches_g(real_data, monkeypatch):
    cfg = GANConfig(gen_hidden=(16,), disc_hidden=(16,), batch_size=_BATCH, discretize="gumbel_st")
    keras.utils.set_random_seed(cfg.seed)
    gen, disc, gan = build_models(cfg)
    assert any(isinstance(layer, GumbelStraightThrough) for layer in gan.layers)
    # El generador sigue devolviendo probabilidades (suaves), no one-hot.
    z = np.random.default_rng(0).standard_normal((4, cfg.latent_dim)).astype("float32")
    probs = keras.ops.convert_to_numpy(gen(z))
    assert not set(np.unique(probs)) <= {0.0, 1.0}
    np.testing.assert_allclose(probs.sum(-1), 1.0, atol=1e-5)
    # En el paso de D las muestras falsas son one-hot exactos.
    seen = []
    orig = disc.train_on_batch
    monkeypatch.setattr(disc, "train_on_batch", lambda x, y, *a, **k: (seen.append(np.asarray(x)), orig(x, y, *a, **k))[1])
    train_step(gen, disc, gan, real_data[:_BATCH], cfg)
    fake = seen[1]
    assert set(np.unique(fake)) <= {0.0, 1.0}
    np.testing.assert_array_equal(fake.sum(axis=-1), 1.0)
    # Paso de G: el gradiente atraviesa la discretización y los pesos de G cambian.
    monkeypatch.undo()
    before_g = _weights(gen)
    gan.train_on_batch(z, np.ones((4, 1), dtype="float32"))
    assert _changed(before_g, _weights(gen))


def test_stabilized_trains_with_finite_losses(real_data):
    cfg = GANConfig.stabilized(gen_hidden=(16,), disc_hidden=(16,), batch_size=_BATCH)
    hist = train_gan(real_data, cfg, epochs=3, max_steps_per_epoch=2)
    for key in ("d_loss_real", "d_loss_fake", "d_loss", "g_loss"):
        assert len(hist[key]) == 3
        assert np.all(np.isfinite(hist[key]))


def test_stabilized_training_is_reproducible_on_cpu(real_data):
    cfg = GANConfig.stabilized(gen_hidden=(16,), disc_hidden=(16,), batch_size=_BATCH)
    h1 = train_gan(real_data, cfg, epochs=2, max_steps_per_epoch=3)
    h2 = train_gan(real_data, cfg, epochs=2, max_steps_per_epoch=3)
    for key in ("d_loss_real", "d_loss_fake", "g_loss"):
        assert h1[key] == h2[key]
    other = train_gan(real_data, GANConfig.stabilized(seed=1, gen_hidden=(16,), disc_hidden=(16,), batch_size=_BATCH),
                      epochs=2, max_steps_per_epoch=3)  # fmt: skip
    assert other["g_loss"] != h1["g_loss"]


# ---------------------------------------------------------------------------
# Ablación: ruido de instancia constante
# ---------------------------------------------------------------------------


def test_constant_noise_config_differs_from_stabilized_only_in_decay_flag():
    c = GANConfig.stabilized_constant_noise(seed=7)
    s = GANConfig.stabilized(seed=7)
    assert c.instance_noise_decay is False and s.instance_noise_decay is True
    assert GANConfig().instance_noise_decay is True  # por defecto, retrocompatible
    assert {**c.__dict__, "instance_noise_decay": True} == s.__dict__
    assert (c.real_label, c.instance_noise_std, c.discretize, c.gumbel_tau) == (0.9, 0.1, "gumbel_st", 1.0)
    assert GANConfig.stabilized_constant_noise(seed=3).seed == 3  # overrides


def test_constant_noise_std_is_the_same_in_every_epoch():
    c = GANConfig.stabilized_constant_noise()
    vals = [instance_noise_std_for_epoch(c, e) for e in range(0, 300)]
    assert all(v == 0.1 for v in vals)


def test_default_decay_schedule_unchanged_by_the_new_flag():
    c = GANConfig.stabilized()
    assert [instance_noise_std_for_epoch(c, e) for e in (0, 50, 99, 100, 150)] == pytest.approx(
        [0.1, 0.05, 0.001, 0.0, 0.0], abs=1e-12
    )
    assert instance_noise_std_for_epoch(GANConfig(instance_noise_std=0.1, instance_noise_decay=True), 100) == 0.0


def test_constant_noise_validation():
    # Sin decaimiento, instance_noise_decay_epochs deja de importar.
    GANConfig(instance_noise_std=0.1, instance_noise_decay=False, instance_noise_decay_epochs=0)
    # Con decaimiento sigue siendo obligatorio >= 1.
    with pytest.raises(ValueError, match="instance_noise_decay_epochs"):
        GANConfig(instance_noise_std=0.1, instance_noise_decay_epochs=0)
    with pytest.raises(ValueError, match="instance_noise_std"):
        GANConfig.stabilized_constant_noise(instance_noise_std=-0.1)


def test_train_gan_keeps_the_noise_std_constant(real_data):
    cfg = GANConfig.stabilized_constant_noise(gen_hidden=(16,), disc_hidden=(16,), batch_size=_BATCH,
                                              instance_noise_decay_epochs=4)  # fmt: skip
    models = build_models(cfg)
    layer = next(m for m in models[1].layers if isinstance(m, InstanceNoise))
    seen = []
    train_gan(
        real_data, cfg, epochs=7, models=models, max_steps_per_epoch=1,
        callback=lambda epoch, gen: seen.append(float(keras.ops.convert_to_numpy(layer.std))),
    )  # fmt: skip
    np.testing.assert_allclose(seen, [0.1] * 7, atol=1e-6)


def test_runner_registers_the_constant_noise_variant():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "amp_gan" / "run_gan_base.py"
    spec = importlib.util.spec_from_file_location("run_gan_base", path)
    run_gan_base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_gan_base)  # el script no importa keras al cargarse

    assert run_gan_base.VARIANTS["stabilized_const"] == "gan_stabc"
    assert run_gan_base.VARIANTS["base"] == "gan_base" and run_gan_base.VARIANTS["stabilized"] == "gan_stab"
    assert len(set(run_gan_base.VARIANTS.values())) == len(run_gan_base.VARIANTS)
    assert run_gan_base.make_config("stabilized_const", 5) == GANConfig.stabilized_constant_noise(seed=5)
    assert run_gan_base.make_config("stabilized", 5) == GANConfig.stabilized(seed=5)
    assert run_gan_base.make_config("base", 5) == GANConfig(seed=5)
    assert run_gan_base._parse(["--variant", "stabilized_const"]).variant == "stabilized_const"

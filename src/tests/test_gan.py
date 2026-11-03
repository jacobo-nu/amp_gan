"""Tests de la GAN base (``amp_gan.gan``): modelos pequeños, pocos pasos, en CPU.

Se ocultan las GPU antes de importar Keras y los tests se saltan si Keras no
está disponible. Funcionan con cualquier backend de Keras 3.
"""

import numpy as np
import pytest

from amp_gan import runtime

runtime.use_cpu_only()
keras = pytest.importorskip("keras")

from amp_gan.data import STANDARD_ALPHABET  # noqa: E402
from amp_gan.encoding import SEQ_LEN, VOCAB_SIZE  # noqa: E402
from amp_gan.gan import (  # noqa: E402
    GANConfig,
    build_discriminator,
    build_gan,
    build_generator,
    build_models,
    generate_probs,
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

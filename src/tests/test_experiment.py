"""Tests de ``amp_gan.experiment``: modelo pequeño, pocos pasos, en CPU."""

import numpy as np
import pandas as pd
import pytest

from amp_gan import runtime

runtime.use_cpu_only()
keras = pytest.importorskip("keras")

from amp_gan.encoding import SEQ_LEN, VOCAB_SIZE, encode_sequences  # noqa: E402
from amp_gan.evaluation import METRIC_NAMES  # noqa: E402
from amp_gan.experiment import MODES, run_gan_experiment, snapshot_epochs  # noqa: E402
from amp_gan.gan import GANConfig  # noqa: E402

_BATCH = 8


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    seqs = sorted({"".join(rng.choice(alphabet, size=int(rng.integers(5, 30)))) for _ in range(80)})
    return encode_sequences(seqs), pd.DataFrame({"id": range(len(seqs)), "sequence": seqs})


@pytest.fixture
def config():
    return GANConfig(gen_hidden=(16,), disc_hidden=(16,), batch_size=_BATCH)


def test_snapshot_epochs():
    assert snapshot_epochs(200, 20) == [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    assert snapshot_epochs(2, 20) == [1, 2]
    assert snapshot_epochs(45, 20) == [1, 20, 40, 45]
    assert snapshot_epochs(1, 5) == [1]
    with pytest.raises(ValueError):
        snapshot_epochs(10, 0)
    with pytest.raises(ValueError):
        snapshot_epochs(0, 5)


def test_run_gan_experiment_two_epochs_shapes(data, config):
    X, ref = data
    res = run_gan_experiment(X, ref, config, epochs=2, snapshot_every=20, n_eval=30,
                             eval_seed=1, max_steps_per_epoch=3)  # fmt: skip
    # historial
    assert {"d_loss_real", "d_loss_fake", "d_loss", "g_loss"} <= set(res.history)
    assert all(len(res.history[k]) == 2 for k in ("d_loss_real", "d_loss_fake", "d_loss", "g_loss"))
    # métricas: épocas 1 y 2 (la 2 es la última) x 2 modos
    m = res.metrics
    assert list(m.columns) == ["epoch", "mode", *METRIC_NAMES]
    assert len(m) == 4
    assert sorted(m["epoch"].unique()) == [1, 2]
    assert set(m["mode"]) == set(MODES) == {"argmax", "sample"}
    assert (m["n"] == 30).all()
    assert res.snapshot_epochs == [1, 2]
    # secuencias finales
    assert set(res.sequences) == {"argmax", "sample"}
    assert all(len(v) == 30 and all(isinstance(s, str) for s in v) for v in res.sequences.values())
    # tiempos
    assert len(res.epoch_seconds) == 2 and all(t > 0 for t in res.epoch_seconds)
    t = res.timing
    assert t["total_s"] >= t["train_s"] > 0 and t["eval_s"] >= 0
    assert t["mean_epoch_s"] == pytest.approx(np.mean(res.epoch_seconds))


def test_snapshot_schedule_with_more_epochs(data, config):
    X, ref = data
    res = run_gan_experiment(X, ref, config, epochs=5, snapshot_every=2, n_eval=10,
                             eval_seed=1, max_steps_per_epoch=1)  # fmt: skip
    assert sorted(res.metrics["epoch"].unique()) == [1, 2, 4, 5]
    assert len(res.metrics) == 8


def test_evaluation_is_reproducible_for_same_generator(data, config):
    # Misma semilla de entrenamiento y evaluación -> mismas secuencias en modo argmax en CPU.
    X, ref = data
    kw = dict(epochs=1, snapshot_every=5, n_eval=20, eval_seed=3, max_steps_per_epoch=2)
    a = run_gan_experiment(X, ref, config, **kw)
    b = run_gan_experiment(X, ref, config, **kw)
    assert a.sequences == b.sequences
    np.testing.assert_allclose(a.history["g_loss"], b.history["g_loss"])


def test_vocab_constants_used_by_fixture():
    assert (SEQ_LEN, VOCAB_SIZE) == (60, 21)

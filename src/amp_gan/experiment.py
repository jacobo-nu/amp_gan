"""Experimento de entrenamiento de la GAN con evaluación por instantáneas.

:func:`run_gan_experiment` entrena con :func:`amp_gan.gan.train_gan` y, mediante
el callback de época, toma instantáneas (época 1, cada ``snapshot_every`` épocas
y la última). En cada instantánea genera ``n_eval`` secuencias con
:func:`amp_gan.gan.generate_probs` (semilla fija ``eval_seed``, así todas las
instantáneas usan el mismo ruido) y calcula
:func:`amp_gan.evaluation.evaluate_sequences` para los dos modos de
decodificación: ``"argmax"`` y ``"sample"`` (muestreo categórico por posición,
con ``default_rng(eval_seed)``).

Por defecto entrena con :func:`amp_gan.gan.train_gan`; el parámetro opcional
``train_fn`` permite evaluar otro modelo con exactamente el mismo protocolo (por
ejemplo :func:`amp_gan.wgan_gp.train_wgan_gp`). Solo hace falta que ``train_fn``
tenga la misma firma que ``train_gan`` (``X, config, epochs, *,
max_steps_per_epoch, callback``), que el historial que devuelva tenga los modelos
en el atributo ``generator`` y que ``config`` tenga ``latent_dim`` y ``seed``
(los usa :func:`amp_gan.gan.generate_probs`).

Este módulo importa Keras (a través de :mod:`amp_gan.gan`): el backend debe
fijarse antes con :mod:`amp_gan.runtime`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from amp_gan.evaluation import build_reference, evaluate_sequences
from amp_gan.gan import GANConfig, generate_probs, sample_sequences, train_gan

#: Modos de decodificación evaluados, en este orden.
MODES: tuple[str, ...] = ("argmax", "sample")


@dataclass
class ExperimentResult:
    """Resultado de :func:`run_gan_experiment`.

    Attributes:
        history: historial de pérdidas de :func:`amp_gan.gan.train_gan` (o de
            ``train_fn``), con los modelos entrenados en ``history.generator``, etc.
        metrics: DataFrame con una fila por (``epoch``, ``mode``) y las
            columnas ``epoch`` (empieza en 1), ``mode`` y las métricas de
            :data:`amp_gan.evaluation.METRIC_NAMES`.
        sequences: secuencias generadas en la última instantánea, por modo.
        timing: tiempos en segundos: ``total_s`` (todo, incluidas las
            evaluaciones), ``train_s`` (entrenamiento sin evaluaciones; la
            primera época incluye la construcción del modelo), ``eval_s``
            (instantáneas), ``mean_epoch_s`` y ``median_epoch_s`` (sobre
            ``epoch_seconds``).
        epoch_seconds: duración de cada época, sin contar las evaluaciones.
        snapshot_epochs: épocas (desde 1) en las que se tomó instantánea.
    """

    history: dict
    metrics: pd.DataFrame
    sequences: dict[str, list[str]]
    timing: dict[str, float]
    epoch_seconds: list[float] = field(default_factory=list)
    snapshot_epochs: list[int] = field(default_factory=list)


def snapshot_epochs(epochs: int, snapshot_every: int) -> list[int]:
    """Épocas (desde 1) con instantánea: la 1, los múltiplos de ``snapshot_every`` y la última."""
    if epochs < 1:
        raise ValueError(f"epochs debe ser >= 1, se recibió {epochs}.")
    if snapshot_every < 1:
        raise ValueError(f"snapshot_every debe ser >= 1, se recibió {snapshot_every}.")
    chosen = {1, epochs, *range(snapshot_every, epochs + 1, snapshot_every)}
    return sorted(chosen)


def run_gan_experiment(
    X: np.ndarray,
    reference_df: pd.DataFrame,
    config: GANConfig,
    epochs: int,
    snapshot_every: int,
    n_eval: int,
    eval_seed: int,
    max_steps_per_epoch: int | None = None,
    train_fn: Callable | None = None,
) -> ExperimentResult:
    """Entrena la GAN y evalúa instantáneas a lo largo del entrenamiento.

    Args:
        X: datos de entrenamiento ``(n, seq_len, vocab_size)``.
        reference_df: DataFrame con columna ``sequence`` (secuencias reales de
            referencia para las métricas).
        config: configuración de la GAN (``GANConfig``, o la que espere ``train_fn``).
        epochs: número de épocas.
        snapshot_every: cada cuántas épocas se evalúa (además de la 1 y la última).
        n_eval: secuencias generadas por instantánea y modo.
        eval_seed: semilla del ruido y del muestreo de la evaluación.
        max_steps_per_epoch: tope de pasos por época (solo para pruebas rápidas).
        train_fn: función de entrenamiento; ``None`` usa :func:`amp_gan.gan.train_gan`.

    Returns:
        :class:`ExperimentResult`.
    """
    epochs_to_eval = set(snapshot_epochs(epochs, snapshot_every))
    reference = build_reference(reference_df)  # se calcula una sola vez

    rows: list[dict] = []
    final: dict[str, list[str]] = {}
    epoch_seconds: list[float] = []
    eval_seconds = 0.0
    mark = time.perf_counter()
    start = mark

    def _callback(epoch0: int, generator) -> None:
        nonlocal mark, eval_seconds
        now = time.perf_counter()
        epoch_seconds.append(now - mark)
        epoch = epoch0 + 1
        if epoch in epochs_to_eval:
            probs = generate_probs(generator, n_eval, config, seed=eval_seed)
            for mode in MODES:
                rng = np.random.default_rng(eval_seed)
                seqs = sample_sequences(probs, rng=rng, mode=mode)
                rows.append({"epoch": epoch, "mode": mode, **evaluate_sequences(seqs, reference)})
                final[mode] = seqs
            eval_seconds += time.perf_counter() - now
        mark = time.perf_counter()

    train = train_gan if train_fn is None else train_fn
    history = train(
        X, config, epochs, max_steps_per_epoch=max_steps_per_epoch, callback=_callback
    )
    total = time.perf_counter() - start
    train_s = float(np.sum(epoch_seconds))
    timing = {
        "total_s": total,
        "train_s": train_s,
        "eval_s": eval_seconds,
        "mean_epoch_s": float(np.mean(epoch_seconds)),
        "median_epoch_s": float(np.median(epoch_seconds)),
    }
    return ExperimentResult(
        history=history,
        metrics=pd.DataFrame(rows),
        sequences=final,
        timing=timing,
        epoch_seconds=epoch_seconds,
        snapshot_epochs=sorted(epochs_to_eval),
    )

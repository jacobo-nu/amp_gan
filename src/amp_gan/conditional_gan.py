"""GAN condicional con condición numérica (Keras 3), sobre la vanilla estabilizada con ruido constante.

La base es :meth:`amp_gan.gan.GANConfig.stabilized_constant_noise` (etiqueta real
0,9; ruido de instancia constante 0,1; Gumbel-softmax *straight-through* con
τ = 1; MLP generador 256-512 y discriminador 512-256; ``z`` de 100; Adam de
2e-4 con β1 = 0,5; lote 128). Se añade una **condición numérica** de dimensión
``k`` (ver :mod:`amp_gan.conditions`: longitud y/o hidrofobicidad normalizadas a
``[0, 1]``) y nada más:

* **Generador** ``[z, c]`` -> ``(seq_len, vocab_size)``: la condición se
  concatena a ``z`` (entrada del primer ``Dense``); el resto del MLP es el mismo.
* **Discriminador** ``[secuencia, c]`` -> probabilidad de real: el ruido de
  instancia se aplica a la secuencia (no a la condición), después ``Flatten`` y
  la condición se concatena a la secuencia aplanada; el resto es el mismo.
* **Modelo combinado** con entradas ``[z, c]``: ``D(Gumbel(G([z, c])), c)``.
* **Entrenamiento** (:func:`train_conditional_gan`): igual que
  :func:`amp_gan.gan.train_step` pero con la condición REAL de cada lote (la
  longitud o el GRAVY de las secuencias reales del lote, normalizados) tanto para
  el lote real como para generar el falso y en el paso del generador: el
  generador recibe siempre las mismas condiciones que el lote real. La condición
  se calcula desde el one-hot de entrenamiento, así que la firma es la misma que
  la de ``train_gan`` (``X, config, epochs, *, max_steps_per_epoch, callback``).
  Mismo bucle por épocas, semilla, barajado y descarte del último lote;
  ``callback(epoch, generator)``.
* **Generación** (:func:`generate_conditional_probs`): ``z`` de
  ``default_rng(seed)`` (igual que :func:`amp_gan.gan.generate_probs`) y las
  condiciones que se pidan; la decodificación es :func:`amp_gan.gan.sample_sequences`.

Evaluación (:func:`run_conditional_experiment`, mismo protocolo que
:func:`amp_gan.experiment.run_gan_experiment`: instantáneas en las épocas 1,
cada ``snapshot_every`` y la última; ``n_eval`` secuencias; semilla de evaluación
fija; modos ``argmax`` y ``sample``):

(i) **empírica**: las condiciones son las de secuencias reales muestreadas con
    reposición (``default_rng(eval_seed)``) de los datos de entrenamiento, así que
    las métricas son las de :data:`amp_gan.evaluation.METRIC_NAMES`, comparables
    con las variantes sin condicionar.
(ii) **por objetivos** (solo si se dan ``targets`` y la condición incluye la
    propiedad ``target_property``, ``length`` por defecto o ``hydrophobicity``):
    ``n_eval`` secuencias pedidas con cada valor objetivo de esa propiedad
    (ver :func:`amp_gan.conditions.evaluate_target`). Si hay otras condiciones
    además de la pedida, esas columnas se fijan en la mediana real normalizada.

Este módulo importa Keras: el backend debe fijarse antes con :mod:`amp_gan.runtime`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import keras
import numpy as np
import pandas as pd

from amp_gan.conditions import (
    ConditionSpec,
    build_target_references,
    check_condition_names,
    check_target_property,
    condition_spec_from_encoded,
    default_tolerance,
    evaluate_target,
    target_columns,
    target_key,
)
from amp_gan.evaluation import build_reference, evaluate_sequences
from amp_gan.experiment import MODES, snapshot_epochs
from amp_gan.gan import (
    GANConfig,
    GANHistory,
    GumbelStraightThrough,
    InstanceNoise,
    _as_float,
    _find_layer,
    _optimizer,
    instance_noise_std_for_epoch,
    sample_sequences,
)


@dataclass
class ConditionalGANConfig(GANConfig):
    """:class:`~amp_gan.gan.GANConfig` más la lista de condiciones numéricas.

    Con los valores por defecto es la vanilla pura condicionada; la configuración
    del experimento es :meth:`stabilized_constant_noise` (heredada: acepta
    ``conditions=...`` y cualquier otro campo como ``override``).
    """

    conditions: tuple[str, ...] = ("length",)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.conditions = tuple(self.conditions)
        check_condition_names(self.conditions)  # ValueError si hay alguna desconocida, repetida o ninguna

    @property
    def cond_dim(self) -> int:
        """Dimensión ``k`` de la condición."""
        return len(self.conditions)


class ConditionalGANHistory(GANHistory):
    """:class:`~amp_gan.gan.GANHistory` con la especificación de condiciones (``spec``)."""

    spec: ConditionSpec | None = None


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def build_conditional_generator(config: ConditionalGANConfig) -> keras.Model:
    """Generador ``[z, c]`` -> ``(seq_len, vocab_size)``: ``c`` se concatena a ``z`` y el resto es el MLP de siempre."""
    z = keras.Input(shape=(config.latent_dim,), name="z")
    c = keras.Input(shape=(config.cond_dim,), name="condition")
    x = keras.layers.Concatenate(name="z_condition")([z, c])
    for units in config.gen_hidden:
        x = keras.layers.Dense(units)(x)
        x = keras.layers.LeakyReLU(negative_slope=config.leaky_alpha)(x)
    x = keras.layers.Dense(config.seq_len * config.vocab_size)(x)
    x = keras.layers.Reshape((config.seq_len, config.vocab_size))(x)
    out = keras.layers.Softmax(axis=-1)(x)
    return keras.Model([z, c], out, name="generator")


def build_conditional_discriminator(config: ConditionalGANConfig) -> keras.Model:
    """Discriminador ``[secuencia, c]`` -> probabilidad de real: ``c`` se concatena a la secuencia aplanada."""
    seq = keras.Input(shape=(config.seq_len, config.vocab_size), name="sequence")
    c = keras.Input(shape=(config.cond_dim,), name="condition")
    x = seq
    if config.instance_noise_std > 0:
        x = InstanceNoise(config.instance_noise_std, seed=config.seed + 1, name="instance_noise")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Concatenate(name="sequence_condition")([x, c])
    for units in config.disc_hidden:
        x = keras.layers.Dense(units)(x)
        x = keras.layers.LeakyReLU(negative_slope=config.leaky_alpha)(x)
        x = keras.layers.Dropout(config.dropout)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model([seq, c], out, name="discriminator")


def build_conditional_gan(
    generator: keras.Model, discriminator: keras.Model, config: ConditionalGANConfig
) -> keras.Model:
    """Compila D y construye el modelo combinado ``[z, c] -> D(Gumbel(G([z, c])), c)`` con D congelado."""
    discriminator.trainable = True
    discriminator.compile(loss="binary_crossentropy", optimizer=_optimizer(config))

    discriminator.trainable = False
    z = keras.Input(shape=(config.latent_dim,), name="z_gan")
    c = keras.Input(shape=(config.cond_dim,), name="condition_gan")
    fake = generator([z, c])
    if config.discretize == "gumbel_st":
        fake = GumbelStraightThrough(config.gumbel_tau, seed=config.seed + 2, name="gumbel_st")(fake)
    gan = keras.Model([z, c], discriminator([fake, c]), name="gan")
    gan.compile(loss="binary_crossentropy", optimizer=_optimizer(config))
    return gan


def build_conditional_models(config: ConditionalGANConfig) -> tuple[keras.Model, keras.Model, keras.Model]:
    """Atajo: ``(generator, discriminator, gan)`` condicionales ya compilados."""
    generator = build_conditional_generator(config)
    discriminator = build_conditional_discriminator(config)
    gan = build_conditional_gan(generator, discriminator, config)
    return generator, discriminator, gan


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def conditional_train_step(
    generator: keras.Model,
    discriminator: keras.Model,
    gan: keras.Model,
    real_batch: np.ndarray,
    cond_batch: np.ndarray,
    config: ConditionalGANConfig,
    noise_std: float | None = None,
) -> dict[str, float]:
    """Un paso vanilla condicional (ver :func:`amp_gan.gan.train_step`).

    ``cond_batch`` ``(n, k)`` son las condiciones normalizadas del lote real; se
    usan también para generar el lote falso y en el paso del generador.
    """
    n = int(real_batch.shape[0])
    cond_batch = np.asarray(cond_batch, dtype="float32")
    ones = np.ones((n, 1), dtype="float32")
    zeros = np.zeros((n, 1), dtype="float32")
    real_labels = ones if config.real_label == 1.0 else np.full((n, 1), config.real_label, dtype="float32")

    noise_layer = _find_layer(discriminator, InstanceNoise)
    if noise_layer is not None:
        noise_layer.set_std(config.instance_noise_std if noise_std is None else noise_std)
    elif noise_std:
        raise ValueError("noise_std > 0 pero el discriminador no tiene ruido de instancia.")

    # (1) Discriminador: lote real con su condición y lote falso generado con esa misma condición.
    z = keras.random.normal((n, config.latent_dim))
    cond_t = keras.ops.convert_to_tensor(cond_batch)  # z es un tensor del backend: no se pueden mezclar con arrays
    fake = generator([z, cond_t], training=False)
    if config.discretize == "gumbel_st":
        fake = gan.get_layer("gumbel_st")(fake)
    fake_batch = keras.ops.convert_to_numpy(fake)
    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch([real_batch, cond_batch], real_labels)
    d_loss_fake = discriminator.train_on_batch([fake_batch, cond_batch], zeros)

    # (2) Generador: D congelado dentro del modelo combinado, mismas condiciones.
    discriminator.trainable = False
    z = keras.random.normal((n, config.latent_dim))
    g_loss = gan.train_on_batch([z, cond_batch], ones)

    return {
        "d_loss_real": _as_float(d_loss_real),
        "d_loss_fake": _as_float(d_loss_fake),
        "g_loss": _as_float(g_loss),
    }


def train_conditional_gan(
    X: np.ndarray,
    config: ConditionalGANConfig,
    epochs: int,
    *,
    models: tuple[keras.Model, keras.Model, keras.Model] | None = None,
    max_steps_per_epoch: int | None = None,
    callback: Callable[[int, keras.Model], None] | None = None,
) -> ConditionalGANHistory:
    """Bucle por épocas de la GAN condicional (misma firma y protocolo que :func:`amp_gan.gan.train_gan`).

    Las condiciones de cada secuencia real se calculan desde ``X`` (longitud y/o
    GRAVY) y se normalizan con :func:`amp_gan.conditions.condition_spec_from_encoded`;
    la especificación queda en ``history.spec``. Semilla con
    ``keras.utils.set_random_seed(config.seed)``, barajado con
    ``default_rng(config.seed)``, descarte del último lote incompleto y ruido de
    instancia según :func:`amp_gan.gan.instance_noise_std_for_epoch`.

    Returns:
        :class:`ConditionalGANHistory` con ``d_loss_real``, ``d_loss_fake``,
        ``d_loss`` y ``g_loss`` (medias por época), los modelos y ``spec``.
    """
    X = np.asarray(X, dtype="float32")
    n = X.shape[0]
    steps = n // config.batch_size
    if steps < 1:
        raise ValueError(f"Hay {n} muestras, menos que batch_size={config.batch_size}.")
    if max_steps_per_epoch is not None:
        steps = min(steps, int(max_steps_per_epoch))

    spec, conds = condition_spec_from_encoded(config.conditions, X)

    keras.utils.set_random_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    generator, discriminator, gan = models if models is not None else build_conditional_models(config)

    history = ConditionalGANHistory(d_loss_real=[], d_loss_fake=[], d_loss=[], g_loss=[])
    history.generator, history.discriminator, history.gan = generator, discriminator, gan
    history.spec = spec

    for epoch in range(epochs):
        perm = rng.permutation(n)
        noise_std = instance_noise_std_for_epoch(config, epoch) if config.instance_noise_std > 0 else None
        acc = {"d_loss_real": [], "d_loss_fake": [], "g_loss": []}
        for step in range(steps):
            idx = perm[step * config.batch_size : (step + 1) * config.batch_size]
            out = conditional_train_step(generator, discriminator, gan, X[idx], conds[idx], config, noise_std=noise_std)
            for key in acc:
                acc[key].append(out[key])
        d_real = float(np.mean(acc["d_loss_real"]))
        d_fake = float(np.mean(acc["d_loss_fake"]))
        history["d_loss_real"].append(d_real)
        history["d_loss_fake"].append(d_fake)
        history["d_loss"].append(0.5 * (d_real + d_fake))
        history["g_loss"].append(float(np.mean(acc["g_loss"])))
        if callback is not None:
            callback(epoch, generator)
    return history


# --------------------------------------------------------------------------- #
# Generación condicionada
# --------------------------------------------------------------------------- #
def generate_conditional_probs(
    generator: keras.Model, conditions: np.ndarray, config: GANConfig, seed: int | None = None
) -> np.ndarray:
    """Salidas del generador para las condiciones pedidas: ``(n, seq_len, vocab_size)``.

    Args:
        generator: generador condicional.
        conditions: condiciones YA normalizadas ``(n, k)`` (ver
            :meth:`amp_gan.conditions.ConditionSpec.normalize`).
        config: configuración (``latent_dim`` y ``seed``).
        seed: semilla del ruido ``z`` (``numpy.random.default_rng``); ``None`` usa ``config.seed``.
    """
    conditions = np.asarray(conditions, dtype="float32")
    if conditions.ndim != 2:
        raise ValueError(f"conditions debe tener forma (n, k), se recibió {conditions.shape}.")
    rng = np.random.default_rng(config.seed if seed is None else seed)
    z = rng.standard_normal((conditions.shape[0], config.latent_dim)).astype("float32")
    probs = generator.predict([z, conditions], batch_size=1024, verbose=0)
    return np.asarray(probs, dtype="float32")


def empirical_condition_indices(n_train: int, n_eval: int, eval_seed: int) -> np.ndarray:
    """Índices de las secuencias reales cuyas condiciones se piden en la evaluación (i): con reposición, ``default_rng(eval_seed)``."""
    return np.random.default_rng(eval_seed).integers(0, n_train, size=n_eval)


def target_condition_matrix(
    spec: ConditionSpec, target: float, n: int, fixed_raw: Sequence[float] | None = None, prop: str = "length"
) -> np.ndarray:
    """Condiciones normalizadas ``(n, k)`` que piden ``prop = target`` (longitud o GRAVY reales).

    Las demás condiciones (si las hay) se fijan en ``fixed_raw`` (valores reales,
    en el orden de ``spec.names``; la posición de ``prop`` se ignora).
    """
    raw = np.zeros(spec.k, dtype="float64") if fixed_raw is None else np.asarray(fixed_raw, dtype="float64").copy()
    raw[spec.names.index(check_target_property(prop))] = target
    return np.repeat(spec.normalize(raw)[None, :], n, axis=0)


# --------------------------------------------------------------------------- #
# Experimento con instantáneas
# --------------------------------------------------------------------------- #
@dataclass
class ConditionalExperimentResult:
    """Resultado de :func:`run_conditional_experiment`.

    Attributes:
        history: :class:`ConditionalGANHistory` (pérdidas, modelos y ``spec``).
        metrics: evaluación (i): una fila por (``epoch``, ``mode``) con las métricas de
            :data:`amp_gan.evaluation.METRIC_NAMES`.
        targets: evaluación (ii): una fila por (``epoch``, ``mode``, objetivo) con las
            columnas de :func:`amp_gan.conditions.target_columns` de ``target_property``
            (:data:`~amp_gan.conditions.TARGET_COLUMNS` para la longitud; vacío si no hay objetivos).
        sequences: secuencias de la evaluación (i) en la última instantánea, por modo.
        target_sequences: secuencias de la última instantánea por modo y objetivo
            (``{modo: {t: [...]}}``; ``t`` entero para la longitud y ``float`` para el GRAVY).
        condition_indices: índices reales usados en la evaluación (i).
        timing, epoch_seconds, snapshot_epochs: como en :class:`amp_gan.experiment.ExperimentResult`.
    """

    history: ConditionalGANHistory
    metrics: pd.DataFrame
    targets: pd.DataFrame
    sequences: dict[str, list[str]]
    target_sequences: dict[str, dict[int, list[str]]]
    condition_indices: np.ndarray
    timing: dict[str, float]
    epoch_seconds: list[float] = field(default_factory=list)
    snapshot_epochs: list[int] = field(default_factory=list)


def run_conditional_experiment(
    X: np.ndarray,
    reference_df: pd.DataFrame,
    config: ConditionalGANConfig,
    epochs: int,
    snapshot_every: int,
    n_eval: int,
    eval_seed: int,
    targets: Sequence[float] | None = None,
    tol: float | None = None,
    max_steps_per_epoch: int | None = None,
    target_property: str = "length",
) -> ConditionalExperimentResult:
    """Entrena la GAN condicional y evalúa instantáneas ((i) empírica y (ii) por objetivos).

    Args:
        X: datos de entrenamiento ``(n, seq_len, vocab_size)``.
        reference_df: DataFrame con columna ``sequence`` (las MISMAS secuencias que ``X``, en el mismo orden).
        config: :class:`ConditionalGANConfig`.
        epochs, snapshot_every, n_eval, eval_seed, max_steps_per_epoch: como en
            :func:`amp_gan.experiment.run_gan_experiment`.
        targets: valores objetivo de (ii) para ``target_property``; ``None`` omite (ii), que
            también se omite si la condición no incluye esa propiedad.
        tol: tolerancia de la tasa de acierto (``None``: la fijada para ``target_property``,
            2 aa para la longitud y 0,25 para el GRAVY).
        target_property: propiedad que se pide en (ii): ``"length"`` (por defecto) o ``"hydrophobicity"``.
    """
    if len(reference_df) != X.shape[0]:
        raise ValueError("reference_df y X deben tener las mismas secuencias en el mismo orden.")
    check_target_property(target_property)
    tol = default_tolerance(target_property) if tol is None else tol
    use_targets = bool(targets) and target_property in config.conditions
    epochs_to_eval = set(snapshot_epochs(epochs, snapshot_every))
    reference = build_reference(reference_df)
    real_seqs = reference_df["sequence"].tolist()
    spec, conds_all = condition_spec_from_encoded(config.conditions, np.asarray(X, dtype="float32"))
    idx_eval = empirical_condition_indices(X.shape[0], n_eval, eval_seed)
    conds_eval = conds_all[idx_eval]
    target_refs = build_target_references(real_seqs, targets, tol, prop=target_property) if use_targets else {}
    fixed_raw = spec.denormalize(np.median(conds_all, axis=0)) if use_targets else None

    rows: list[dict] = []
    target_rows: list[dict] = []
    final: dict[str, list[str]] = {}
    final_targets: dict[str, dict[int, list[str]]] = {}
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
            probs = generate_conditional_probs(generator, conds_eval, config, seed=eval_seed)
            for mode in MODES:
                seqs = sample_sequences(probs, rng=np.random.default_rng(eval_seed), mode=mode)
                rows.append({"epoch": epoch, "mode": mode, **evaluate_sequences(seqs, reference)})
                final[mode] = seqs
            if use_targets:
                for t in targets:
                    cond_t = target_condition_matrix(spec, t, n_eval, fixed_raw, prop=target_property)
                    tp = generate_conditional_probs(generator, cond_t, config, seed=eval_seed)
                    key = target_key(target_property, t)
                    ref_t, n_ref = target_refs[key]
                    for mode in MODES:
                        seqs = sample_sequences(tp, rng=np.random.default_rng(eval_seed), mode=mode)
                        row = evaluate_target(seqs, t, ref_t, n_ref, tol, prop=target_property)
                        target_rows.append({"epoch": epoch, "mode": mode, **row})
                        final_targets.setdefault(mode, {})[key] = seqs
            eval_seconds += time.perf_counter() - now
        mark = time.perf_counter()

    history = train_conditional_gan(X, config, epochs, max_steps_per_epoch=max_steps_per_epoch, callback=_callback)
    total = time.perf_counter() - start
    train_s = float(np.sum(epoch_seconds))
    timing = {
        "total_s": total,
        "train_s": train_s,
        "eval_s": eval_seconds,
        "mean_epoch_s": float(np.mean(epoch_seconds)),
        "median_epoch_s": float(np.median(epoch_seconds)),
    }
    targets_df = pd.DataFrame(target_rows, columns=["epoch", "mode", *target_columns(target_property)])
    return ConditionalExperimentResult(
        history=history,
        metrics=pd.DataFrame(rows),
        targets=targets_df,
        sequences=final,
        target_sequences=final_targets,
        condition_indices=idx_eval,
        timing=timing,
        epoch_seconds=epoch_seconds,
        snapshot_epochs=sorted(epochs_to_eval),
    )

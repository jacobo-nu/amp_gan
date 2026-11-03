"""GAN base (vanilla) con perceptrones multicapa, escrita en Keras 3.

Solo usa la API pública de Keras 3 (``keras.layers``, ``keras.ops``,
``keras.random``, ``keras.utils``), de modo que funciona igual con cualquiera
de sus backends. El backend se elige fuera de este módulo (ver
:mod:`amp_gan.runtime`) y debe fijarse antes de importar Keras.

Arquitectura (línea base, sin ajuste de hiperparámetros):

* Generador: ``z (100)`` -> ``Dense`` + LeakyReLU (256, 512) -> ``Dense``
  de ``seq_len * vocab_size`` -> reshape ``(seq_len, vocab_size)`` -> softmax
  por posición.
* Discriminador: ``(seq_len, vocab_size)`` -> ``Flatten`` -> ``Dense`` +
  LeakyReLU + ``Dropout`` (512, 256) -> ``Dense(1)`` con sigmoide.

El entrenamiento es el paso vanilla clásico con ``train_on_batch``: primero D
con un lote real (etiqueta 1) y uno falso (etiqueta 0), después G a través del
modelo combinado con D congelado y etiqueta 1.

Sobre ``trainable``: Keras 3 lee el estado ``trainable`` de forma perezosa (al
construir la función de entrenamiento en la primera llamada), no en
``compile``. Por eso ``train_step`` fija ``discriminator.trainable`` de forma
explícita antes de cada una de sus dos fases (True para D, False para G) y
``build_gan`` deja D congelado, que es el estado con el que se compila el
modelo combinado.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import keras
import numpy as np

from amp_gan.encoding import SEQ_LEN, VOCAB_SIZE, decode_batch


@dataclass
class GANConfig:
    """Hiperparámetros de la GAN base."""

    latent_dim: int = 100
    seq_len: int = SEQ_LEN
    vocab_size: int = VOCAB_SIZE
    gen_hidden: tuple[int, ...] = (256, 512)
    disc_hidden: tuple[int, ...] = (512, 256)
    lr: float = 2e-4
    beta_1: float = 0.5
    batch_size: int = 128
    seed: int = 42
    leaky_alpha: float = 0.2
    dropout: float = 0.3  # solo en el discriminador


class GANHistory(dict):
    """Historial de entrenamiento: diccionario de listas (un valor por época).

    Tras ``train_gan`` guarda además los modelos usados en los atributos
    ``generator``, ``discriminator`` y ``gan``.
    """

    generator: keras.Model | None = None
    discriminator: keras.Model | None = None
    gan: keras.Model | None = None


def build_generator(config: GANConfig) -> keras.Model:
    """Construye el generador MLP: ``z`` -> ``(seq_len, vocab_size)`` (softmax)."""
    z = keras.Input(shape=(config.latent_dim,), name="z")
    x = z
    for units in config.gen_hidden:
        x = keras.layers.Dense(units)(x)
        x = keras.layers.LeakyReLU(negative_slope=config.leaky_alpha)(x)
    x = keras.layers.Dense(config.seq_len * config.vocab_size)(x)
    x = keras.layers.Reshape((config.seq_len, config.vocab_size))(x)
    out = keras.layers.Softmax(axis=-1)(x)
    return keras.Model(z, out, name="generator")


def build_discriminator(config: GANConfig) -> keras.Model:
    """Construye el discriminador MLP: ``(seq_len, vocab_size)`` -> probabilidad de real."""
    inp = keras.Input(shape=(config.seq_len, config.vocab_size), name="sequence")
    x = keras.layers.Flatten()(inp)
    for units in config.disc_hidden:
        x = keras.layers.Dense(units)(x)
        x = keras.layers.LeakyReLU(negative_slope=config.leaky_alpha)(x)
        x = keras.layers.Dropout(config.dropout)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="discriminator")


def _optimizer(config: GANConfig) -> keras.optimizers.Optimizer:
    return keras.optimizers.Adam(learning_rate=config.lr, beta_1=config.beta_1)


def build_gan(
    generator: keras.Model, discriminator: keras.Model, config: GANConfig
) -> keras.Model:
    """Compila D y construye el modelo combinado ``z -> D(G(z))`` con D congelado.

    D se compila con D entrenable; después se congela y se compila el modelo
    combinado (solo G entrenable en él). Al terminar D queda congelado;
    :func:`train_step` lo descongela durante su paso.
    """
    discriminator.trainable = True
    discriminator.compile(loss="binary_crossentropy", optimizer=_optimizer(config))

    discriminator.trainable = False
    z = keras.Input(shape=(config.latent_dim,), name="z_gan")
    gan = keras.Model(z, discriminator(generator(z)), name="gan")
    gan.compile(loss="binary_crossentropy", optimizer=_optimizer(config))
    return gan


def build_models(config: GANConfig) -> tuple[keras.Model, keras.Model, keras.Model]:
    """Atajo: construye ``(generator, discriminator, gan)`` ya compilados."""
    generator = build_generator(config)
    discriminator = build_discriminator(config)
    gan = build_gan(generator, discriminator, config)
    return generator, discriminator, gan


def _as_float(value) -> float:
    """Convierte la salida de ``train_on_batch`` (escalar o lista) en un float."""
    arr = np.asarray(value, dtype="float64").reshape(-1)
    return float(arr[0])


def train_step(
    generator: keras.Model,
    discriminator: keras.Model,
    gan: keras.Model,
    real_batch: np.ndarray,
    config: GANConfig,
) -> dict[str, float]:
    """Un paso vanilla: D con lote real (1) y falso (0); luego G vía ``gan`` (1).

    Args:
        generator, discriminator, gan: modelos devueltos por :func:`build_models`.
        real_batch: lote real ``(n, seq_len, vocab_size)`` ``float32``.
        config: configuración (dimensión latente).

    Returns:
        Diccionario con ``d_loss_real``, ``d_loss_fake`` y ``g_loss``.
    """
    n = int(real_batch.shape[0])
    ones = np.ones((n, 1), dtype="float32")
    zeros = np.zeros((n, 1), dtype="float32")

    # (1) Discriminador: solo cambian sus pesos.
    z = keras.random.normal((n, config.latent_dim))
    fake_batch = keras.ops.convert_to_numpy(generator(z, training=False))
    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(real_batch, ones)
    d_loss_fake = discriminator.train_on_batch(fake_batch, zeros)

    # (2) Generador: D congelado dentro del modelo combinado.
    discriminator.trainable = False
    z = keras.random.normal((n, config.latent_dim))
    g_loss = gan.train_on_batch(z, ones)

    return {
        "d_loss_real": _as_float(d_loss_real),
        "d_loss_fake": _as_float(d_loss_fake),
        "g_loss": _as_float(g_loss),
    }


def train_gan(
    X: np.ndarray,
    config: GANConfig,
    epochs: int,
    *,
    models: tuple[keras.Model, keras.Model, keras.Model] | None = None,
    max_steps_per_epoch: int | None = None,
    callback: Callable[[int, keras.Model], None] | None = None,
) -> GANHistory:
    """Bucle simple de entrenamiento por épocas.

    Fija la semilla con ``keras.utils.set_random_seed(config.seed)``, baraja con un
    generador de numpy (``default_rng(config.seed)``) y descarta el último lote
    incompleto de cada época para mantener constante la forma de los lotes.

    Args:
        X: datos reales ``(n, seq_len, vocab_size)`` ``float32``.
        config: configuración.
        epochs: número de épocas.
        models: ``(generator, discriminator, gan)`` ya construidos; si es ``None``
            se construyen tras fijar la semilla.
        max_steps_per_epoch: tope de pasos por época (pruebas rápidas).
        callback: función ``callback(epoch, generator)`` llamada al acabar cada
            época (``epoch`` empieza en 0).

    Returns:
        :class:`GANHistory` con las listas ``d_loss_real``, ``d_loss_fake``,
        ``d_loss`` (media de las dos) y ``g_loss`` (media por época), y los
        modelos en ``generator``, ``discriminator`` y ``gan``.
    """
    X = np.asarray(X, dtype="float32")
    n = X.shape[0]
    steps = n // config.batch_size
    if steps < 1:
        raise ValueError(f"Hay {n} muestras, menos que batch_size={config.batch_size}.")
    if max_steps_per_epoch is not None:
        steps = min(steps, int(max_steps_per_epoch))

    keras.utils.set_random_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    generator, discriminator, gan = models if models is not None else build_models(config)

    history = GANHistory(d_loss_real=[], d_loss_fake=[], d_loss=[], g_loss=[])
    history.generator, history.discriminator, history.gan = generator, discriminator, gan

    for epoch in range(epochs):
        perm = rng.permutation(n)
        acc = {"d_loss_real": [], "d_loss_fake": [], "g_loss": []}
        for step in range(steps):
            idx = perm[step * config.batch_size : (step + 1) * config.batch_size]
            out = train_step(generator, discriminator, gan, X[idx], config)
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


def generate_probs(
    generator: keras.Model, n: int, config: GANConfig, seed: int | None = None
) -> np.ndarray:
    """Genera ``n`` salidas del generador: probabilidades ``(n, seq_len, vocab_size)``.

    El ruido se saca de ``numpy.random.default_rng(seed)`` (``seed=None`` usa
    ``config.seed``), así el resultado no depende del estado global de Keras.
    """
    rng = np.random.default_rng(config.seed if seed is None else seed)
    z = rng.standard_normal((n, config.latent_dim)).astype("float32")
    probs = generator.predict(z, batch_size=1024, verbose=0)
    return np.asarray(probs, dtype="float32")


def sample_sequences(
    probs: np.ndarray,
    rng: np.random.Generator | None = None,
    mode: str = "argmax",
) -> list[str]:
    """Convierte probabilidades ``(n, seq_len, vocab)`` en cadenas de aminoácidos.

    Args:
        probs: salida de :func:`generate_probs`.
        rng: generador de numpy; obligatorio en modo ``"sample"``.
        mode: ``"argmax"`` (decodifica con ``decode_batch``) o ``"sample"``
            (muestrea cada posición de su categórica y decodifica).

    Returns:
        Lista de cadenas cortadas en el primer PAD (pueden ser vacías).
    """
    probs = np.asarray(probs, dtype="float64")
    if mode == "argmax":
        return decode_batch(probs)
    if mode == "sample":
        if rng is None:
            raise ValueError("El modo 'sample' necesita un generador rng de numpy.")
        p = probs / probs.sum(axis=-1, keepdims=True)
        cum = np.cumsum(p, axis=-1)
        u = rng.random(p.shape[:-1] + (1,))
        idx = np.minimum((cum < u).sum(axis=-1), p.shape[-1] - 1)
        one_hot = np.eye(p.shape[-1], dtype="float32")[idx]
        return decode_batch(one_hot)
    raise ValueError(f"mode debe ser 'argmax' o 'sample', se recibió {mode!r}.")

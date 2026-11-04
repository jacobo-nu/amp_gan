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

Variante estabilizada (opcional; se activa por configuración, con los valores
por defecto de :class:`GANConfig` el comportamiento es el de la vanilla pura):

1. Suavizado unilateral de etiquetas: ``real_label`` (0,9) para las muestras
   reales en el paso de D; las falsas siguen en 0 y en el paso de G la etiqueta
   sigue siendo 1.
2. Ruido de instancia: ruido gaussiano sumado a la entrada de D (capa
   :class:`InstanceNoise` al inicio de D, activa solo con ``training=True``, es
   decir, en el paso de D con muestras reales y falsas y en el paso de G). Su
   desviación típica decrece linealmente de ``instance_noise_std`` a 0 en
   ``instance_noise_decay_epochs`` épocas (:func:`instance_noise_std_for_epoch`);
   con ``instance_noise_decay=False`` se mantiene constante (ablación,
   :meth:`GANConfig.stabilized_constant_noise`).
3. Salida discreta diferenciable: entre G y D las muestras se discretizan a
   one-hot con Gumbel-softmax y *straight-through* (capa
   :class:`GumbelStraightThrough`): en la pasada hacia delante son one-hot
   exactos y en la pasada hacia atrás el gradiente es el del softmax con
   temperatura ``gumbel_tau``. El generador sigue devolviendo probabilidades.

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


#: Modos de discretización admitidos por ``GANConfig.discretize``.
DISCRETIZE_MODES: tuple[str, ...] = ("none", "gumbel_st")


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
    # Correcciones de la variante estabilizada; los valores por defecto las desactivan.
    real_label: float = 1.0  # etiqueta de las muestras reales en el paso de D
    instance_noise_std: float = 0.0  # desviación típica inicial del ruido de instancia
    instance_noise_decay_epochs: int = 100  # épocas hasta que el ruido llega a 0
    instance_noise_decay: bool = True  # False: el ruido se mantiene en instance_noise_std (ablación)
    discretize: str = "none"  # "none" | "gumbel_st"
    gumbel_tau: float = 1.0  # temperatura del Gumbel-softmax

    def __post_init__(self) -> None:
        if self.discretize not in DISCRETIZE_MODES:
            raise ValueError(f"discretize debe ser uno de {DISCRETIZE_MODES}, se recibió {self.discretize!r}.")
        if not 0.0 <= self.real_label <= 1.0:
            raise ValueError(f"real_label debe estar en [0, 1], se recibió {self.real_label}.")
        if self.instance_noise_std < 0:
            raise ValueError(f"instance_noise_std no puede ser negativa ({self.instance_noise_std}).")
        if self.instance_noise_std > 0 and self.instance_noise_decay and self.instance_noise_decay_epochs < 1:
            raise ValueError("instance_noise_decay_epochs debe ser >= 1 si hay ruido de instancia.")
        if self.gumbel_tau <= 0:
            raise ValueError(f"gumbel_tau debe ser > 0, se recibió {self.gumbel_tau}.")

    @classmethod
    def stabilized(cls, **overrides) -> "GANConfig":
        """Configuración de la «GAN vanilla estabilizada» (las tres correcciones).

        Etiqueta real 0,9; ruido de instancia de desviación inicial 0,1 con
        decrecimiento lineal hasta 0 en 100 épocas; Gumbel-softmax con
        *straight-through* y τ = 1. El resto es igual que en la vanilla pura.
        ``overrides`` permite cambiar otros campos (p. ej. ``seed``).
        """
        params = dict(
            real_label=0.9,
            instance_noise_std=0.1,
            instance_noise_decay_epochs=100,
            discretize="gumbel_st",
            gumbel_tau=1.0,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def stabilized_constant_noise(cls, **overrides) -> "GANConfig":
        """Ablación de la estabilizada: igual que :meth:`stabilized` pero con ruido de instancia constante.

        La desviación del ruido permanece en ``instance_noise_std`` (0,1) durante
        todo el entrenamiento (``instance_noise_decay=False``); es el único cambio
        respecto a :meth:`stabilized`. ``overrides`` permite cambiar otros campos.
        """
        params = dict(instance_noise_decay=False)
        params.update(overrides)
        return cls.stabilized(**params)


class GANHistory(dict):
    """Historial de entrenamiento: diccionario de listas (un valor por época).

    Tras ``train_gan`` guarda además los modelos usados en los atributos
    ``generator``, ``discriminator`` y ``gan``.
    """

    generator: keras.Model | None = None
    discriminator: keras.Model | None = None
    gan: keras.Model | None = None


def instance_noise_std_for_epoch(config: GANConfig, epoch: int) -> float:
    """Desviación típica del ruido de instancia en la época ``epoch`` (desde 0).

    Decrece linealmente desde ``instance_noise_std`` (época 0) hasta 0 en la
    época ``instance_noise_decay_epochs`` y vale 0 a partir de ella. Con
    ``instance_noise_decay=False`` es constante e igual a ``instance_noise_std``
    en todas las épocas. Con la configuración por defecto
    (``instance_noise_std = 0``) es siempre 0.
    """
    if config.instance_noise_std <= 0:
        return 0.0
    if not config.instance_noise_decay:
        return float(config.instance_noise_std)
    frac = 1.0 - float(epoch) / config.instance_noise_decay_epochs
    return float(config.instance_noise_std * max(0.0, frac))


def gumbel_softmax_straight_through(probs, tau: float, seed):
    """Discretiza probabilidades a one-hot con Gumbel-softmax y *straight-through*.

    Adelante: one-hot exacto de ``argmax(log p + g)`` con ``g`` ruido de Gumbel
    (equivale a una muestra de la categórica ``p``). Atrás: gradiente del
    softmax ``softmax((log p + g) / tau)``. Solo usa ``keras.ops`` y
    ``keras.random``. ``seed`` es un ``keras.random.SeedGenerator`` o un entero.
    """
    ops = keras.ops
    logits = ops.log(ops.maximum(probs, 1e-20))
    u = keras.random.uniform(ops.shape(probs), minval=1e-10, maxval=1.0 - 1e-7, seed=seed, dtype=probs.dtype)
    gumbel = -ops.log(-ops.log(u))
    y_soft = ops.softmax((logits + gumbel) / tau, axis=-1)
    hard = ops.one_hot(ops.argmax(y_soft, axis=-1), ops.shape(probs)[-1], dtype=y_soft.dtype)
    # Adelante: hard + (y - y) = hard exacto; atrás: d/dy = 1 (estimador straight-through).
    return hard + (y_soft - ops.stop_gradient(y_soft))


class GumbelStraightThrough(keras.layers.Layer):
    """Capa que aplica :func:`gumbel_softmax_straight_through` (siempre, también en inferencia)."""

    def __init__(self, tau: float = 1.0, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.tau = float(tau)
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs):
        return gumbel_softmax_straight_through(inputs, self.tau, self.seed_generator)

    def compute_output_shape(self, input_shape):
        return input_shape


class InstanceNoise(keras.layers.Layer):
    """Suma ruido gaussiano ``N(0, std^2)`` a la entrada, solo con ``training=True``.

    ``std`` es una variable no entrenable que se actualiza con :meth:`set_std`
    (el bucle de entrenamiento la fija al principio de cada época).
    """

    def __init__(self, std: float = 0.0, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.initial_std = float(std)
        self.seed_generator = keras.random.SeedGenerator(seed)

    def build(self, input_shape):
        self.std = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(self.initial_std),
            trainable=False,
            name="std",
        )

    def set_std(self, value: float) -> None:
        self.std.assign(float(value))

    def call(self, inputs, training=False):
        if not training:
            return inputs
        noise = keras.random.normal(keras.ops.shape(inputs), seed=self.seed_generator, dtype=inputs.dtype)
        return inputs + self.std * noise

    def compute_output_shape(self, input_shape):
        return input_shape


def _find_layer(model: keras.Model, layer_type: type):
    """Primera capa de ``model`` del tipo dado, o ``None``."""
    for layer in model.layers:
        if isinstance(layer, layer_type):
            return layer
    return None


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
    x = inp
    if config.instance_noise_std > 0:
        x = InstanceNoise(config.instance_noise_std, seed=config.seed + 1, name="instance_noise")(x)
    x = keras.layers.Flatten()(x)
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
    fake = generator(z)
    if config.discretize == "gumbel_st":
        fake = GumbelStraightThrough(config.gumbel_tau, seed=config.seed + 2, name="gumbel_st")(fake)
    gan = keras.Model(z, discriminator(fake), name="gan")
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
    noise_std: float | None = None,
) -> dict[str, float]:
    """Un paso vanilla: D con lote real (``real_label``, 1 por defecto) y falso (0); luego G vía ``gan`` (1).

    Args:
        generator, discriminator, gan: modelos devueltos por :func:`build_models`.
        real_batch: lote real ``(n, seq_len, vocab_size)`` ``float32``.
        config: configuración (dimensión latente, etiqueta real, discretización).
        noise_std: desviación típica del ruido de instancia en este paso (la
            fija :func:`train_gan` según la época). ``None`` usa
            ``config.instance_noise_std``. Se ignora (debe ser 0 o ``None``) si
            D no tiene ruido de instancia.

    Returns:
        Diccionario con ``d_loss_real``, ``d_loss_fake`` y ``g_loss``.
    """
    n = int(real_batch.shape[0])
    ones = np.ones((n, 1), dtype="float32")
    zeros = np.zeros((n, 1), dtype="float32")
    real_labels = ones if config.real_label == 1.0 else np.full((n, 1), config.real_label, dtype="float32")

    noise_layer = _find_layer(discriminator, InstanceNoise)
    if noise_layer is not None:
        noise_layer.set_std(config.instance_noise_std if noise_std is None else noise_std)
    elif noise_std:
        raise ValueError("noise_std > 0 pero el discriminador no tiene ruido de instancia.")

    # (1) Discriminador: solo cambian sus pesos.
    z = keras.random.normal((n, config.latent_dim))
    fake = generator(z, training=False)
    if config.discretize == "gumbel_st":
        fake = gan.get_layer("gumbel_st")(fake)  # versión dura (sin gradiente: se pasa a numpy)
    fake_batch = keras.ops.convert_to_numpy(fake)
    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(real_batch, real_labels)
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
    generador de numpy (``default_rng(config.seed)``) y, si hay ruido de instancia,
    fija su desviación en cada época con :func:`instance_noise_std_for_epoch`.
    Descarta el último lote incompleto de cada época para mantener constante la
    forma de los lotes.

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
        noise_std = instance_noise_std_for_epoch(config, epoch) if config.instance_noise_std > 0 else None
        acc = {"d_loss_real": [], "d_loss_fake": [], "g_loss": []}
        for step in range(steps):
            idx = perm[step * config.batch_size : (step + 1) * config.batch_size]
            out = train_step(generator, discriminator, gan, X[idx], config, noise_std=noise_std)
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

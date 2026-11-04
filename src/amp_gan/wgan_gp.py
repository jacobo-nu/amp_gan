"""WGAN-GP (Wasserstein GAN con penalización de gradiente) con Keras 3.

Implementa el esquema canónico de Gulrajani et al. (2017, «Improved Training of
Wasserstein GANs») con los mismos datos y el mismo generador que las GAN
anteriores del proyecto:

* **Generador**: el de :func:`amp_gan.gan.build_generator` (MLP 256-512 con salida
  softmax ``(seq_len, vocab_size)``), sin cambios. Se construye con la
  :class:`~amp_gan.gan.GANConfig` que devuelve :meth:`WGANConfig.generator_config`.
* **Crítico**: ``Flatten`` -> ``Dense(512)`` + LeakyReLU -> ``Dense(256)`` +
  LeakyReLU -> ``Dense(1)`` **lineal** (sin sigmoide, sin BatchNorm, sin Dropout
  ni ruido de instancia). No hay Gumbel-softmax ni suavizado de etiquetas: el
  crítico compara las secuencias reales (one-hot) con las probabilidades
  generadas (softmax), como en el artículo original.
* **Pérdidas**: crítico ``mean(D(fake)) - mean(D(real)) + lambda * GP`` con
  ``GP = mean((||grad_x D(x_hat)||_2 - 1)^2)`` y
  ``x_hat = eps * real + (1 - eps) * fake``, ``eps ~ U(0, 1)`` por muestra;
  generador ``-mean(D(G(z)))``. Adam (lr 1e-4, beta = (0; 0,9)) para ambos.

Definición de época
-------------------
Una **época** es una pasada por los datos reales: ``n // batch_size`` pasos de
crítico (169 con los 21 754 péptidos de Antibacterial y lote 128), cada uno con un
lote real distinto (permutación de los datos, último lote incompleto
descartado). El generador se actualiza **una vez cada** ``n_critic`` (5) pasos de
crítico, con un contador que se arrastra entre épocas; así que hay unas 33
actualizaciones del generador por época (169 // 5), muchas menos que en las GAN
anteriores (una por cada paso de discriminador, 169 por época). El historial
guarda ``critic_steps`` y ``generator_steps`` de cada época.

Código específico de backend
----------------------------
La penalización de gradiente exige derivar respecto de la **entrada** del
crítico y volver a derivar (doble retropropagación), y eso no se puede escribir
solo con ``keras.ops``. Todo lo que depende del backend está aislado en las
funciones marcadas «BACKEND» de este módulo (``_input_gradient_*``,
``_critic_step_*``, ``_generator_step_*``), que se eligen con
:func:`_dispatch` según ``keras.backend.backend()``; los imports de
``tensorflow`` y ``torch`` son perezosos (dentro de cada una). El resto usa solo
la API pública de Keras (``keras.ops``, ``keras.random``, ``keras.layers``,
``keras.optimizers``). Con TensorFlow los pasos del crítico y del generador se
compilan con ``tf.function`` (en CPU, ~4 veces más rápido que en modo ``eager``;
con torch no hace falta). Backends admitidos: ``tensorflow`` y ``torch``; con
cualquier otro (``jax``) se lanza :class:`NotImplementedError`.

Historial (:class:`WGANHistory`, una entrada por época)
-------------------------------------------------------
``critic_loss``
    media de la pérdida total del crítico (con penalización).
``wasserstein_estimate``
    media de ``mean(D(real)) - mean(D(fake))`` = ``-(pérdida del crítico sin GP)``;
    estima la distancia de Wasserstein-1 (en unidades del crítico).
``gradient_penalty``
    media de ``GP`` (sin multiplicar por lambda).
``g_loss``
    media de ``-mean(D(G(z)))`` en las actualizaciones del generador de esa
    época (``NaN`` si en la época no hubo ninguna, solo posible con
    ``max_steps_per_epoch < n_critic``).
``critic_steps``, ``generator_steps``
    pasos de crítico y actualizaciones del generador de la época.

Este módulo importa Keras: el backend debe fijarse antes con :mod:`amp_gan.runtime`.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass

import keras
import numpy as np

from amp_gan.encoding import SEQ_LEN, VOCAB_SIZE
from amp_gan.gan import GANConfig, build_generator

#: Backends de Keras en los que funciona la penalización de gradiente.
SUPPORTED_BACKENDS: tuple[str, ...] = ("tensorflow", "torch")

# Pequeña constante dentro de la raíz de la norma, para que el gradiente de la
# penalización no sea NaN si la norma del gradiente del crítico fuese 0.
_NORM_EPS = 1e-12


@dataclass
class WGANConfig:
    """Hiperparámetros de la WGAN-GP (valores canónicos de Gulrajani et al., sin ajustar).

    El generador es el del resto del proyecto (``latent_dim``, ``gen_hidden``,
    ``leaky_alpha``, ``seq_len``, ``vocab_size``); ``critic_hidden`` es la
    arquitectura del crítico. ``gp_weight`` es lambda y ``n_critic`` el número de
    pasos de crítico por cada actualización del generador.
    """

    latent_dim: int = 100
    seq_len: int = SEQ_LEN
    vocab_size: int = VOCAB_SIZE
    gen_hidden: tuple[int, ...] = (256, 512)
    critic_hidden: tuple[int, ...] = (512, 256)
    batch_size: int = 128
    seed: int = 42
    leaky_alpha: float = 0.2
    gp_weight: float = 10.0
    n_critic: int = 5
    lr: float = 1e-4
    beta_1: float = 0.0
    beta_2: float = 0.9

    def __post_init__(self) -> None:
        if self.n_critic < 1:
            raise ValueError(f"n_critic debe ser >= 1, se recibió {self.n_critic}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size debe ser >= 1, se recibió {self.batch_size}.")
        if self.gp_weight < 0:
            raise ValueError(f"gp_weight no puede ser negativo ({self.gp_weight}).")
        if self.lr <= 0:
            raise ValueError(f"lr debe ser > 0, se recibió {self.lr}.")

    def generator_config(self) -> GANConfig:
        """:class:`~amp_gan.gan.GANConfig` con la arquitectura del generador (para ``build_generator``).

        Los demás campos de ``GANConfig`` (tasa de aprendizaje, Dropout, correcciones de
        la estabilizada...) no intervienen en el generador y se dejan en su valor por
        defecto.
        """
        return GANConfig(
            latent_dim=self.latent_dim,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            gen_hidden=self.gen_hidden,
            leaky_alpha=self.leaky_alpha,
            batch_size=self.batch_size,
            seed=self.seed,
        )


class WGANHistory(dict):
    """Historial de entrenamiento: diccionario de listas (un valor por época).

    Tras :func:`train_wgan_gp` guarda además los modelos en ``generator`` y ``critic``.
    """

    generator: keras.Model | None = None
    critic: keras.Model | None = None


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def build_critic(config: WGANConfig) -> keras.Model:
    """Crítico MLP: ``(seq_len, vocab_size)`` -> puntuación real (sin sigmoide, BN ni Dropout)."""
    inp = keras.Input(shape=(config.seq_len, config.vocab_size), name="sequence")
    x = keras.layers.Flatten()(inp)
    for units in config.critic_hidden:
        x = keras.layers.Dense(units)(x)
        x = keras.layers.LeakyReLU(negative_slope=config.leaky_alpha)(x)
    out = keras.layers.Dense(1)(x)  # lineal
    return keras.Model(inp, out, name="critic")


def build_wgan_models(config: WGANConfig) -> tuple[keras.Model, keras.Model]:
    """Atajo: ``(generator, critic)`` (el generador es :func:`amp_gan.gan.build_generator`)."""
    return build_generator(config.generator_config()), build_critic(config)


def _optimizer(config: WGANConfig) -> keras.optimizers.Optimizer:
    return keras.optimizers.Adam(learning_rate=config.lr, beta_1=config.beta_1, beta_2=config.beta_2)


# --------------------------------------------------------------------------- #
# Pérdidas (solo keras.ops; independientes del backend)
# --------------------------------------------------------------------------- #
def wasserstein_critic_loss(d_real, d_fake):
    """Pérdida del crítico sin penalización: ``mean(D(fake)) - mean(D(real))``."""
    return keras.ops.mean(d_fake) - keras.ops.mean(d_real)


def generator_loss(d_fake):
    """Pérdida del generador: ``-mean(D(G(z)))``."""
    return -keras.ops.mean(d_fake)


def gradient_penalty_from_gradients(grads):
    """``mean((||g||_2 - 1)^2)`` con la norma por muestra sobre todos los ejes salvo el primero."""
    axes = tuple(range(1, len(grads.shape)))
    norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(grads), axis=axes) + _NORM_EPS)
    return keras.ops.mean(keras.ops.square(norm - 1.0))


def interpolate(real, fake, eps):
    """``x_hat = eps * real + (1 - eps) * fake`` (``eps`` ``(n, 1, ..., 1)``, uno por muestra)."""
    return eps * real + (1.0 - eps) * fake


def gradient_penalty(critic: keras.Model, real, fake, eps):
    """Penalización de gradiente en ``x_hat = interpolate(real, fake, eps)``.

    El resultado es diferenciable respecto de los pesos del crítico (usa la doble
    retropropagación de :func:`_input_gradient`). ``fake`` se trata como constante.
    """
    real, fake, eps = (keras.ops.convert_to_tensor(a, dtype="float32") for a in (real, fake, eps))
    x_hat = interpolate(real, keras.ops.stop_gradient(fake), eps)
    return gradient_penalty_from_gradients(_input_gradient(critic, x_hat))


def _critic_losses(critic: keras.Model, real, fake, eps, gp_weight: float):
    """Devuelve ``(loss, wasserstein_estimate, gp)``; ``loss`` incluye ``gp_weight * gp``."""
    d_real = critic(real, training=True)
    d_fake = critic(fake, training=True)
    gp = gradient_penalty(critic, real, fake, eps)
    w_loss = wasserstein_critic_loss(d_real, d_fake)
    return w_loss + gp_weight * gp, -w_loss, gp


def _generator_loss_from_noise(generator: keras.Model, critic: keras.Model, z):
    return generator_loss(critic(generator(z, training=True), training=True))


# --------------------------------------------------------------------------- #
# BACKEND: gradiente respecto de la entrada del crítico (doble retropropagación)
# --------------------------------------------------------------------------- #
def _input_gradient_tensorflow(critic: keras.Model, x_hat):
    """BACKEND tensorflow: ``d sum(D(x_hat)) / d x_hat`` con una ``GradientTape`` interna.

    Si se llama dentro de otra ``GradientTape`` (la del paso del crítico), el
    resultado es diferenciable respecto de los pesos.
    """
    import tensorflow as tf

    with tf.GradientTape() as tape:
        tape.watch(x_hat)
        score = critic(x_hat, training=True)
    return tape.gradient(score, x_hat)  # grad de sum(score): cada muestra es independiente


def _input_gradient_torch(critic: keras.Model, x_hat):
    """BACKEND torch: ``d sum(D(x_hat)) / d x_hat`` con ``create_graph=True``."""
    import torch

    x = x_hat.detach().requires_grad_(True)
    score = critic(x, training=True)
    (grads,) = torch.autograd.grad(score.sum(), x, create_graph=True)
    return grads


# --------------------------------------------------------------------------- #
# BACKEND: pasos de entrenamiento (derivar la pérdida y aplicar el optimizador)
# --------------------------------------------------------------------------- #
_TF_FUNCTIONS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # modelo -> {clave: tf.function}


def _tf_compiled(model, key, build):
    """``tf.function`` ya construida para ``model`` y ``key`` (se crea una vez con ``build()``)."""
    cache = _TF_FUNCTIONS.setdefault(model, {})
    if key not in cache:
        cache[key] = build()
    return cache[key]


def _critic_step_tensorflow(critic, optimizer, real, fake, eps, gp_weight):
    """BACKEND tensorflow: un paso del crítico con ``tf.GradientTape`` (anidada en la del GP).

    Se compila con ``tf.function`` (en CPU es unas 4 veces más rápido que en modo
    ``eager``; los números son los mismos salvo redondeo).
    """
    import tensorflow as tf

    def build():
        @tf.function
        def step(real, fake, eps):
            variables = critic.trainable_variables
            with tf.GradientTape() as tape:
                loss, wasserstein, gp = _critic_losses(critic, real, fake, eps, gp_weight)
            optimizer.apply(tape.gradient(loss, variables), variables)
            return loss, wasserstein, gp

        return step

    return _tf_compiled(critic, ("critic", id(optimizer), float(gp_weight)), build)(real, fake, eps)


def _critic_step_torch(critic, optimizer, real, fake, eps, gp_weight):
    """BACKEND torch: un paso del crítico con ``torch.autograd.grad`` y ``optimizer.apply``."""
    import torch

    variables = critic.trainable_variables
    loss, wasserstein, gp = _critic_losses(critic, real, fake, eps, gp_weight)
    grads = torch.autograd.grad(loss, [v.value for v in variables])
    with torch.no_grad():
        optimizer.apply(list(grads), variables)
    return loss.detach(), wasserstein.detach(), gp.detach()


def _generator_step_tensorflow(generator, critic, optimizer, z):
    """BACKEND tensorflow: un paso del generador (el crítico no se actualiza)."""
    import tensorflow as tf

    def build():
        @tf.function
        def step(z):
            variables = generator.trainable_variables
            with tf.GradientTape() as tape:
                loss = _generator_loss_from_noise(generator, critic, z)
            optimizer.apply(tape.gradient(loss, variables), variables)
            return loss

        return step

    return _tf_compiled(generator, ("generator", id(critic), id(optimizer)), build)(z)


def _generator_step_torch(generator, critic, optimizer, z):
    """BACKEND torch: un paso del generador (el crítico no se actualiza)."""
    import torch

    variables = generator.trainable_variables
    loss = _generator_loss_from_noise(generator, critic, z)
    grads = torch.autograd.grad(loss, [v.value for v in variables])  # solo respecto del generador
    with torch.no_grad():
        optimizer.apply(list(grads), variables)
    return loss.detach()


_BACKEND_FUNCTIONS = {
    "tensorflow": {
        "input_gradient": _input_gradient_tensorflow,
        "critic_step": _critic_step_tensorflow,
        "generator_step": _generator_step_tensorflow,
    },
    "torch": {
        "input_gradient": _input_gradient_torch,
        "critic_step": _critic_step_torch,
        "generator_step": _generator_step_torch,
    },
}


def _backend_name() -> str:
    return str(keras.backend.backend())


def _dispatch(kind: str) -> Callable:
    """Función específica del backend activo (``input_gradient``, ``critic_step`` o ``generator_step``).

    Raises:
        NotImplementedError: si el backend de Keras no es ``tensorflow`` ni ``torch``.
    """
    name = _backend_name()
    if name not in _BACKEND_FUNCTIONS:
        raise NotImplementedError(
            f"La WGAN-GP necesita doble retropropagación específica del backend y solo está "
            f"implementada para {SUPPORTED_BACKENDS}; el backend de Keras activo es {name!r}. "
            "Elige tensorflow o torch (variable KERAS_BACKEND o amp_gan.runtime) antes de importar Keras."
        )
    return _BACKEND_FUNCTIONS[name][kind]


def _input_gradient(critic: keras.Model, x_hat):
    return _dispatch("input_gradient")(critic, x_hat)


# --------------------------------------------------------------------------- #
# Pasos con la API pública y bucle de entrenamiento
# --------------------------------------------------------------------------- #
def _scalar(value) -> float:
    return float(np.asarray(keras.ops.convert_to_numpy(value), dtype="float64").reshape(-1)[0])


def critic_step(generator, critic, optimizer, real_batch, config: WGANConfig) -> dict[str, float]:
    """Un paso del crítico con un lote real. Devuelve ``critic_loss``, ``wasserstein_estimate`` y ``gradient_penalty``."""
    ops = keras.ops
    n = int(real_batch.shape[0])
    real = ops.convert_to_tensor(real_batch, dtype="float32")
    z = keras.random.normal((n, config.latent_dim))
    fake = ops.stop_gradient(generator(z, training=False))
    eps = keras.random.uniform((n,) + (1,) * (len(real.shape) - 1))
    loss, wasserstein, gp = _dispatch("critic_step")(critic, optimizer, real, fake, eps, config.gp_weight)
    return {
        "critic_loss": _scalar(loss),
        "wasserstein_estimate": _scalar(wasserstein),
        "gradient_penalty": _scalar(gp),
    }


def generator_step(generator, critic, optimizer, config: WGANConfig, n: int | None = None) -> float:
    """Una actualización del generador con ``n`` (por defecto ``batch_size``) ruidos. Devuelve ``g_loss``."""
    z = keras.random.normal((config.batch_size if n is None else n, config.latent_dim))
    return _scalar(_dispatch("generator_step")(generator, critic, optimizer, z))


def train_wgan_gp(
    X: np.ndarray,
    config: WGANConfig,
    epochs: int,
    *,
    max_steps_per_epoch: int | None = None,
    callback: Callable[[int, keras.Model], None] | None = None,
) -> WGANHistory:
    """Entrena la WGAN-GP por épocas (ver «Definición de época» en la documentación del módulo).

    Fija la semilla con ``keras.utils.set_random_seed(config.seed)``, baraja con
    ``default_rng(config.seed)`` y descarta el último lote incompleto. Cada época
    hace ``n // batch_size`` pasos de crítico (tope: ``max_steps_per_epoch``) y el
    generador se actualiza tras cada ``n_critic`` pasos de crítico (contador
    global entre épocas).

    Args:
        X: datos reales ``(n, seq_len, vocab_size)`` ``float32``.
        config: configuración.
        epochs: número de épocas.
        max_steps_per_epoch: tope de pasos de crítico por época (pruebas rápidas).
        callback: función ``callback(epoch, generator)`` llamada al acabar cada
            época (``epoch`` empieza en 0), igual que en :func:`amp_gan.gan.train_gan`.

    Returns:
        :class:`WGANHistory` (columnas en la documentación del módulo) con los
        modelos en ``generator`` y ``critic``.

    Raises:
        NotImplementedError: con un backend distinto de tensorflow o torch.
    """
    _dispatch("critic_step")  # falla pronto y con un mensaje claro si el backend no es válido
    X = np.asarray(X, dtype="float32")
    n = X.shape[0]
    steps = n // config.batch_size
    if steps < 1:
        raise ValueError(f"Hay {n} muestras, menos que batch_size={config.batch_size}.")
    if max_steps_per_epoch is not None:
        steps = min(steps, int(max_steps_per_epoch))

    keras.utils.set_random_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    generator, critic = build_wgan_models(config)
    g_opt, c_opt = _optimizer(config), _optimizer(config)

    history = WGANHistory(
        critic_loss=[], wasserstein_estimate=[], gradient_penalty=[], g_loss=[],
        critic_steps=[], generator_steps=[],
    )  # fmt: skip
    history.generator, history.critic = generator, critic

    critic_count = 0  # pasos de crítico acumulados (entre épocas)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        acc = {"critic_loss": [], "wasserstein_estimate": [], "gradient_penalty": []}
        g_losses: list[float] = []
        for step in range(steps):
            idx = perm[step * config.batch_size : (step + 1) * config.batch_size]
            out = critic_step(generator, critic, c_opt, X[idx], config)
            for key in acc:
                acc[key].append(out[key])
            critic_count += 1
            if critic_count % config.n_critic == 0:
                g_losses.append(generator_step(generator, critic, g_opt, config))
        for key in acc:
            history[key].append(float(np.mean(acc[key])))
        history["g_loss"].append(float(np.mean(g_losses)) if g_losses else float("nan"))
        history["critic_steps"].append(steps)
        history["generator_steps"].append(len(g_losses))
        if callback is not None:
            callback(epoch, generator)
    return history

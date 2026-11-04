"""Selección de backend y dispositivo de Keras 3 y descripción del entorno.

Este módulo NO importa keras, tensorflow ni torch a nivel de módulo (ni en el
proceso padre al detectar backends o GPU): sus funciones de configuración deben
poder llamarse antes de importar Keras, que lee ``KERAS_BACKEND`` al importarse.

Política (pensada para que el código funcione en cualquier máquina, con o sin
GPU, sin depender de ningún archivo generado antes):

* **Por defecto, CPU** (:func:`apply_cpu_default`). Backend = TensorFlow si está
  instalado (fue el más rápido en CPU en las medidas del proyecto: ~2,9 s por
  época frente a ~3,3 s de torch) y, si no, torch; error claro si no hay
  ninguno. Se puede forzar con el argumento ``backend`` o con la variable de
  entorno ``AMP_GAN_BACKEND`` (``tensorflow`` o ``torch``).
* **Opcional, automático** (:func:`apply_auto_backend`): si hay una GPU
  utilizable, mide en ESTA máquina los backends/dispositivos disponibles con la
  prueba de rendimiento reducida, aplica la regla de decisión de
  :func:`amp_gan.benchmark.select_fastest` y configura el ganador. Si no hay GPU
  o la prueba falla, cae a la CPU por defecto y lo indica en el mensaje de la
  decisión.

Orden de uso típico (notebook o script)::

    from amp_gan import runtime
    runtime.apply_cpu_default()       # o: runtime.apply_auto_backend()
    import keras                      # a partir de aquí el backend queda fijado

Diagnóstico de TensorFlow + GPU (entorno del proyecto, medido durante la
prueba de rendimiento): el binario de TensorFlow está compilado contra CUDA 12
(busca ``libcudart.so.12``, ``libcublas.so.12``, ``libcufft.so.11``,
``libcusolver.so.11`` y ``libcusparse.so.12``), mientras que el entorno solo
tiene las librerías CUDA 13 (paquetes ``nvidia-*`` instalados para torch).
Ajustar ``LD_LIBRARY_PATH`` no lo arregla: enlazar las librerías 13 con los
nombres 12 hace que TensorFlow vea la GPU, pero falla al inicializarla
(``shared object symbol not found``), porque la ABI no es compatible. Por eso
no se aplica ningún ajuste de variables de entorno para TensorFlow-GPU. Haría
falta instalar las ruedas CUDA 12 (por ejemplo ``tensorflow[and-cuda]``, que
trae ``nvidia-cuda-runtime-cu12``, ``nvidia-cublas-cu12``, ``nvidia-cufft-cu12``,
``nvidia-cusolver-cu12``, ``nvidia-cusparse-cu12`` y ``nvidia-cudnn-cu12``) o
una versión de TensorFlow compilada con CUDA 13.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import subprocess
import sys
import warnings

#: Backends de Keras 3 admitidos.
VALID_BACKENDS: tuple[str, ...] = ("tensorflow", "torch", "jax")

#: Backends que el proyecto prueba y entre los que elige (en orden de preferencia en CPU).
SUPPORTED_BACKENDS: tuple[str, ...] = ("tensorflow", "torch")

#: Variable de entorno para forzar el backend (``tensorflow`` o ``torch``).
BACKEND_ENV_VAR = "AMP_GAN_BACKEND"

#: Tamaño de lote de entrenamiento sobre el que decide el modo automático.
TRAIN_BATCH_SIZE = 128


def _imported_keras_backend() -> str | None:
    """Backend del Keras ya importado en este proceso, o ``None`` si no lo está."""
    keras = sys.modules.get("keras")
    if keras is None:
        return None
    try:
        return str(keras.backend.backend())
    except Exception:  # Keras a medio importar: no se puede saber el backend aún.
        return None


def set_backend(name: str) -> None:
    """Fija ``KERAS_BACKEND``; debe llamarse antes de importar Keras.

    Raises:
        ValueError: si el nombre no es un backend admitido.
        RuntimeError: si Keras ya está importado con otro backend (en ese caso
            hay que reiniciar el intérprete o usar un subproceso).
    """
    if name not in VALID_BACKENDS:
        raise ValueError(f"Backend {name!r} no válido; opciones: {VALID_BACKENDS}.")
    current = _imported_keras_backend()
    if current is not None and current != name:
        raise RuntimeError(
            f"Keras ya está importado con el backend {current!r}; no se puede cambiar "
            f"a {name!r} en este proceso. Reinicia el intérprete (kernel) y llama a "
            "set_backend() antes de importar keras."
        )
    os.environ["KERAS_BACKEND"] = name


_UNRECORDED = object()
_previous_cuda_visible: object = _UNRECORDED  # valor de CUDA_VISIBLE_DEVICES antes de ocultar la GPU


def use_cpu_only() -> None:
    """Oculta las GPU (``CUDA_VISIBLE_DEVICES=""``); llamar antes de importar nada.

    Guarda el valor previo de la variable para poder deshacerlo con
    :func:`restore_gpu_visibility`. Si TensorFlow o torch ya están importados la
    variable puede no tener efecto y se emite un aviso.
    """
    global _previous_cuda_visible
    if _previous_cuda_visible is _UNRECORDED:
        _previous_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    already = [m for m in ("tensorflow", "torch") if m in sys.modules]
    if already:
        warnings.warn(
            f"{already} ya está(n) importado(s): ocultar la GPU puede no tener efecto "
            "en este proceso.",
            stacklevel=2,
        )


def restore_gpu_visibility() -> None:
    """Deshace :func:`use_cpu_only`: restaura el valor previo de ``CUDA_VISIBLE_DEVICES``.

    Si el valor previo era una cadena vacía (la GPU ya estaba oculta desde fuera)
    se conserva oculta. No hace nada si :func:`use_cpu_only` no se llamó antes.
    """
    global _previous_cuda_visible
    if _previous_cuda_visible is _UNRECORDED:
        return
    if _previous_cuda_visible is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_previous_cuda_visible)
    _previous_cuda_visible = _UNRECORDED


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def describe_environment() -> dict:
    """Describe el entorno de ejecución (importa keras de forma perezosa).

    Returns:
        Diccionario con ``keras_backend``, ``keras_devices`` (dispositivos que
        Keras ve), ``gpu_visible`` (bool), ``cuda_visible_devices`` (valor de la
        variable, ``None`` si no está definida) y las versiones de python,
        numpy, keras, tensorflow y torch (``None`` si no están instalados; las
        de tensorflow y torch se leen de los metadatos sin importarlos).
    """
    import keras

    devices = [str(d) for d in keras.distribution.list_devices()]
    return {
        "keras_backend": keras.backend.backend(),
        "keras_devices": devices,
        "gpu_visible": any(d.lower().startswith(("gpu", "cuda")) for d in devices),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_version": platform.python_version(),
        "numpy_version": _version("numpy"),
        "keras_version": keras.__version__,
        "tensorflow_version": _version("tensorflow"),
        "torch_version": _version("torch"),
    }


# --------------------------------------------------------------------------- #
# Detección de backends y GPU (sin importar keras en el proceso padre)
# --------------------------------------------------------------------------- #
def _is_installed(module: str) -> bool:
    """``True`` si el módulo es importable, sin importarlo (``importlib.util.find_spec``)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # p. ej. __spec__ ausente o paquete padre roto
        return False


def installed_backends() -> list[str]:
    """Backends de :data:`SUPPORTED_BACKENDS` instalados, sin importar ninguno."""
    return [b for b in SUPPORTED_BACKENDS if _is_installed(b)]


# Sondeos de GPU: se ejecutan en un subproceso para no contaminar el proceso
# padre (importar torch o TensorFlow aquí fijaría la visibilidad de la GPU).
_GPU_PROBES: dict[str, str] = {
    "torch": "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)",
    "tensorflow": (
        "import sys, tensorflow as tf; "
        "sys.exit(0 if tf.config.list_physical_devices('GPU') else 3)"
    ),
}


def gpu_available(backend: str, timeout: float = 180.0) -> bool:
    """``True`` si ``backend`` puede usar una GPU en esta máquina.

    Lanza un subproceso que importa solo la librería del backend (torch o
    TensorFlow) y pregunta por GPU; el proceso padre no importa nada. Devuelve
    ``False`` si el backend no está instalado, si ``CUDA_VISIBLE_DEVICES`` es
    vacía (GPU oculta), o si el sondeo falla o agota el tiempo.
    """
    if backend not in _GPU_PROBES or not _is_installed(backend):
        return False
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return False
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _GPU_PROBES[backend]],
            env=env, capture_output=True, timeout=timeout,
        )  # fmt: skip
    except Exception:  # timeout, sin intérprete, etc.: se considera sin GPU
        return False
    return proc.returncode == 0


def available_candidates(backend: str | None = None) -> list[tuple[str, str]]:
    """Configuraciones ``(backend, dispositivo)`` que se pueden probar en esta máquina.

    Solo entran los backends instalados (``backend`` restringe a uno) con
    ``"cpu"`` y, si :func:`gpu_available` lo confirma, también con ``"gpu"``.
    Primero van las de CPU y luego las de GPU.
    """
    names = [b for b in installed_backends() if backend in (None, b)]
    cpus = [(b, "cpu") for b in names]
    gpus = [(b, "gpu") for b in names if gpu_available(b)]
    return cpus + gpus


# --------------------------------------------------------------------------- #
# Modo por defecto: CPU
# --------------------------------------------------------------------------- #
def _forced_backend(backend: str | None) -> str | None:
    """Backend forzado por argumento o por ``AMP_GAN_BACKEND`` (``None`` si no hay)."""
    name = backend or os.environ.get(BACKEND_ENV_VAR) or None
    if name is None:
        return None
    name = name.strip().lower()
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Backend {name!r} no admitido (argumento o {BACKEND_ENV_VAR}); "
            f"opciones: {SUPPORTED_BACKENDS}."
        )
    if not _is_installed(name):
        raise RuntimeError(
            f"El backend {name!r} (pedido por argumento o {BACKEND_ENV_VAR}) no está instalado. "
            f"Instálalo (pip install {name}) o elige otro; instalados: {installed_backends() or 'ninguno'}."
        )
    return name


def choose_cpu_backend(backend: str | None = None) -> str:
    """Elige el backend para CPU: el forzado, o TensorFlow si está, o torch.

    Orden de decisión: argumento ``backend``; variable ``AMP_GAN_BACKEND``;
    TensorFlow si está instalado (el más rápido en CPU en las medidas del
    proyecto); torch. No importa ninguna librería.

    Raises:
        ValueError: si el backend forzado no es ``tensorflow`` ni ``torch``.
        RuntimeError: si el forzado no está instalado o no hay ninguno instalado.
    """
    forced = _forced_backend(backend)
    if forced is not None:
        return forced
    installed = installed_backends()
    if not installed:
        raise RuntimeError(
            "No hay ningún backend de Keras instalado: instala tensorflow o torch "
            "(por ejemplo, pip install -r requirements.txt)."
        )
    return installed[0]  # SUPPORTED_BACKENDS pone TensorFlow antes que torch


def apply_cpu_default(backend: str | None = None) -> dict:
    """Modo por defecto: backend de CPU elegido por :func:`choose_cpu_backend`, GPU oculta.

    No lee ni escribe ningún archivo. Debe llamarse antes de importar Keras.

    Returns:
        Decisión: ``mode`` (``"cpu"``), ``backend``, ``device`` (``"cpu"``),
        ``forced`` (bool) y ``message``.
    """
    forced = _forced_backend(backend) is not None
    name = choose_cpu_backend(backend)
    set_backend(name)
    use_cpu_only()
    origin = "forzado" if forced else "por defecto"
    return {
        "mode": "cpu",
        "backend": name,
        "device": "cpu",
        "forced": forced,
        "message": f"Modo CPU: backend {name} ({origin}), GPU oculta.",
    }


# --------------------------------------------------------------------------- #
# Modo automático (opcional): medir y elegir
# --------------------------------------------------------------------------- #
def _fallback_to_cpu(reason: str, backend: str | None, candidates: list) -> dict:
    """Aplica la CPU por defecto y devuelve una decisión de modo ``auto`` marcada como caída."""
    cpu = apply_cpu_default(backend)
    return {
        **cpu,
        "mode": "auto",
        "fallback": True,
        "candidates": [list(c) for c in candidates],
        "est_epoch_s": None,
        "tied": None,
        "message": f"{reason} Se usa la CPU por defecto: backend {cpu['backend']}.",
    }


def apply_auto_backend(
    backend: str | None = None,
    batch_sizes: tuple[int, ...] | None = None,
    steps: int = 60,
    warmup: int = 20,
    verbose: bool = True,
):
    """Modo automático: mide los candidatos disponibles y aplica el más rápido.

    1. Detecta los candidatos con :func:`available_candidates` (backends
       instalados; GPU solo si se confirma). ``backend`` (o ``AMP_GAN_BACKEND``)
       restringe la búsqueda a ese backend.
    2. Si no hay ninguna GPU utilizable, la elección es la CPU: cae a la CPU
       por defecto (:func:`apply_cpu_default`, TensorFlow si está instalado) y
       lo indica, sin medir.
    3. Si hay GPU, ejecuta la prueba reducida en memoria
       (:func:`amp_gan.benchmark.collect_benchmark`; un subproceso por
       candidato, el que falle se descarta) y aplica la regla de
       :func:`amp_gan.benchmark.select_fastest` para el lote
       :data:`TRAIN_BATCH_SIZE`. Si ningún candidato produce resultados válidos,
       cae a la CPU por defecto.

    Nunca lee el CSV del benchmark. Debe llamarse antes de importar Keras.

    Args:
        backend: restringe a un backend (``tensorflow`` o ``torch``).
        batch_sizes: tamaños de lote a medir (por defecto los de
            :data:`amp_gan.benchmark.BATCH_SIZES`); siempre se incluye el de entrenamiento.
        steps: pasos medidos por candidato y lote (menos que en la prueba completa).
        warmup: pasos de calentamiento.
        verbose: imprime el progreso.

    Returns:
        ``(decision, tabla)``: ``decision`` es un dict serializable con
        ``mode`` (``"auto"``), ``backend``, ``device``, ``fallback`` (bool),
        ``candidates``, ``est_epoch_s``, ``tied``, ``discarded`` (candidatos que
        fallaron) y ``message``; ``tabla`` es el
        DataFrame de la medición, o ``None`` si no se midió.
    """
    from amp_gan import benchmark

    # Si antes se ocultó la GPU con use_cpu_only() (modo por defecto), se
    # restaura para poder sondearla y medirla; se vuelve a ocultar si gana la CPU.
    restore_gpu_visibility()
    forced = _forced_backend(backend)
    candidates = available_candidates(forced)
    if not any(dev == "gpu" for _, dev in candidates):
        reason = "No se ha detectado ninguna GPU utilizable."
        return _fallback_to_cpu(reason, forced, candidates), None

    sizes = tuple(batch_sizes) if batch_sizes else benchmark.BATCH_SIZES
    if TRAIN_BATCH_SIZE not in sizes:
        sizes = (*sizes, TRAIN_BATCH_SIZE)
    if verbose:
        print(f"[backend] candidatos: {candidates}", flush=True)
    try:
        table = benchmark.collect_benchmark(
            candidates=tuple(candidates), batch_sizes=sizes, steps=steps, warmup=warmup,
            verbose=verbose,
        )  # fmt: skip
        pick = benchmark.select_fastest(table, TRAIN_BATCH_SIZE)
    except Exception as exc:  # prueba rota o sin resultados válidos
        reason = f"La prueba de rendimiento no produjo un resultado válido ({type(exc).__name__}: {exc})."
        return _fallback_to_cpu(reason, forced, candidates), None

    set_backend(pick["backend"])
    if pick["device"] == "cpu":
        use_cpu_only()
    good = table[(table["status"] == "ok") & (table["batch_size"] == TRAIN_BATCH_SIZE)]
    valid = set(zip(good["backend"], good["device"]))
    discarded = [list(c) for c in candidates if tuple(c) not in valid]
    extra = f" Descartados por fallar: {discarded}." if discarded else ""
    decision = {
        "mode": "auto",
        "backend": pick["backend"],
        "device": pick["device"],
        "fallback": False,
        "candidates": [list(c) for c in candidates],
        "est_epoch_s": pick["est_epoch_s"],
        "tied": pick["tied"],
        "discarded": discarded,
        "message": f"Modo automático (lote {TRAIN_BATCH_SIZE}): {pick['message']}{extra}",
    }
    return decision, table

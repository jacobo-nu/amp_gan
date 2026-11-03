"""Selección de backend y dispositivo de Keras 3 y descripción del entorno.

Este módulo NO importa keras, tensorflow ni torch a nivel de módulo: sus
funciones de configuración deben poder llamarse antes de importar Keras, que
lee ``KERAS_BACKEND`` al importarse. Orden de uso típico (notebook o script)::

    from amp_gan import runtime
    runtime.set_backend("torch")      # o runtime.apply_fastest(csv, 128)
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
import os
import platform
import sys
import warnings
from pathlib import Path

#: Backends de Keras 3 admitidos.
VALID_BACKENDS: tuple[str, ...] = ("tensorflow", "torch", "jax")


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


def use_cpu_only() -> None:
    """Oculta las GPU (``CUDA_VISIBLE_DEVICES=""``); llamar antes de importar nada.

    Si TensorFlow o torch ya están importados la variable puede no tener efecto
    y se emite un aviso.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    already = [m for m in ("tensorflow", "torch") if m in sys.modules]
    if already:
        warnings.warn(
            f"{already} ya está(n) importado(s): ocultar la GPU puede no tener efecto "
            "en este proceso.",
            stacklevel=2,
        )


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


def apply_fastest(csv_path: str | Path, batch_size: int) -> dict:
    """Aplica el backend y dispositivo más rápidos según el CSV del benchmark.

    Lee el CSV generado por :mod:`amp_gan.benchmark`, aplica la regla de
    decisión para ``batch_size`` (menor tiempo estimado por época; empate
    <10 % se resuelve a favor de la opción más simple) y llama a
    :func:`set_backend` y, si el dispositivo elegido es CPU,
    :func:`use_cpu_only`. Debe llamarse antes de importar Keras.

    Returns:
        Diccionario de decisión (ver :func:`amp_gan.benchmark.select_fastest`).

    Raises:
        FileNotFoundError: si no existe el CSV (ejecutar antes el benchmark).
        ValueError: si el CSV no tiene filas válidas para ese tamaño de lote.
    """
    import pandas as pd

    from amp_gan.benchmark import select_fastest

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta antes el benchmark: python -m amp_gan.benchmark"
        )
    choice = select_fastest(pd.read_csv(path), batch_size)
    set_backend(choice["backend"])
    if choice["device"] == "cpu":
        use_cpu_only()
    return choice

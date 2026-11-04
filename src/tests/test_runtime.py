"""Tests de ``amp_gan.runtime`` (sin importar keras, tensorflow ni torch ni entrenar)."""

import os
import sys
import types

import pytest

from amp_gan import runtime


def test_runtime_does_not_import_heavy_libraries():
    """Ningún import a nivel de módulo de keras, tensorflow ni torch."""
    import ast

    tree = ast.parse(open(runtime.__file__, encoding="utf-8").read())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert not top & {"keras", "tensorflow", "torch"}


def test_set_backend_sets_environment(monkeypatch):
    monkeypatch.delitem(sys.modules, "keras", raising=False)
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    runtime.set_backend("torch")
    import os

    assert os.environ["KERAS_BACKEND"] == "torch"


def test_set_backend_rejects_unknown_name(monkeypatch):
    monkeypatch.delitem(sys.modules, "keras", raising=False)
    with pytest.raises(ValueError, match="no válido"):
        runtime.set_backend("caffe")


def test_set_backend_fails_if_keras_imported_with_other_backend(monkeypatch):
    import os

    fake = types.SimpleNamespace(backend=types.SimpleNamespace(backend=lambda: "torch"))
    monkeypatch.setitem(sys.modules, "keras", fake)
    monkeypatch.setenv("KERAS_BACKEND", "torch")
    with pytest.raises(RuntimeError, match="ya está importado"):
        runtime.set_backend("tensorflow")
    assert os.environ["KERAS_BACKEND"] == "torch"  # no se modificó
    runtime.set_backend("torch")  # el mismo backend es válido


def test_use_cpu_only_hides_gpus(monkeypatch):
    import os

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    runtime.use_cpu_only()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_use_cpu_only_warns_if_libraries_already_imported(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.warns(UserWarning, match="ya está"):
        runtime.use_cpu_only()


# --------------------------------------------------------------------------- #
# Detección de backends (find_spec simulado)
# --------------------------------------------------------------------------- #
@pytest.fixture
def clean_env(monkeypatch):
    """Entorno sin Keras importado, sin variables de selección y con el estado de GPU reiniciado."""
    for mod in ("keras", "tensorflow", "torch"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setenv("KERAS_BACKEND", "x")  # para que monkeypatch la elimine al salir
    monkeypatch.delenv("KERAS_BACKEND")
    monkeypatch.delenv(runtime.BACKEND_ENV_VAR, raising=False)
    monkeypatch.setattr(runtime, "_previous_cuda_visible", runtime._UNRECORDED)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # restaurada por monkeypatch al salir


def _fake_installed(monkeypatch, present):
    """Simula que solo los módulos de ``present`` son importables (sin importar nada)."""
    monkeypatch.setattr(runtime, "_is_installed", lambda module: module in present)


def test_is_installed_uses_find_spec_without_importing(monkeypatch):
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    calls = []
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: calls.append(name) or real(name, *a, **k)
    )
    runtime.installed_backends()
    assert calls == ["tensorflow", "torch"]
    assert "tensorflow" not in sys.modules and "torch" not in sys.modules


def test_is_installed_false_for_missing_and_broken_modules(monkeypatch):
    assert runtime._is_installed("modulo_que_no_existe_amp_gan") is False
    import importlib.util

    def boom(name, *a, **k):
        raise ValueError("__spec__ is None")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert runtime._is_installed("torch") is False


@pytest.mark.parametrize(
    "present, expected",
    [
        ({"tensorflow", "torch"}, ["tensorflow", "torch"]),
        ({"torch"}, ["torch"]),
        ({"tensorflow"}, ["tensorflow"]),
        (set(), []),
    ],
)
def test_installed_backends(monkeypatch, present, expected):
    _fake_installed(monkeypatch, present)
    assert runtime.installed_backends() == expected


def test_gpu_available_false_if_backend_missing_or_gpu_hidden(monkeypatch):
    def no_subprocess(*a, **k):
        raise AssertionError("no debe lanzar subprocesos")

    monkeypatch.setattr(runtime.subprocess, "run", no_subprocess)
    _fake_installed(monkeypatch, set())
    assert runtime.gpu_available("torch") is False  # no instalado
    _fake_installed(monkeypatch, {"torch"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert runtime.gpu_available("torch") is False  # GPU oculta
    assert runtime.gpu_available("jax") is False  # sin sondeo definido


def test_gpu_available_uses_a_subprocess_exit_code(monkeypatch):
    _fake_installed(monkeypatch, {"torch"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    assert runtime.gpu_available("torch") is True
    assert seen[0][0] == sys.executable and "torch" in seen[0][-1]
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=3))
    assert runtime.gpu_available("torch") is False

    def fails(*a, **k):
        raise OSError("sin intérprete")

    monkeypatch.setattr(runtime.subprocess, "run", fails)
    assert runtime.gpu_available("torch") is False  # un sondeo roto no rompe nada


def test_available_candidates_only_lists_installed_backends(monkeypatch):
    _fake_installed(monkeypatch, {"torch"})
    monkeypatch.setattr(runtime, "gpu_available", lambda b, **k: True)
    assert runtime.available_candidates() == [("torch", "cpu"), ("torch", "gpu")]
    _fake_installed(monkeypatch, {"tensorflow", "torch"})
    monkeypatch.setattr(runtime, "gpu_available", lambda b, **k: b == "torch")
    assert runtime.available_candidates() == [
        ("tensorflow", "cpu"), ("torch", "cpu"), ("torch", "gpu"),
    ]  # fmt: skip
    assert runtime.available_candidates("tensorflow") == [("tensorflow", "cpu")]
    _fake_installed(monkeypatch, set())
    assert runtime.available_candidates() == []


# --------------------------------------------------------------------------- #
# Modo por defecto: CPU
# --------------------------------------------------------------------------- #
def test_cpu_default_prefers_tensorflow(monkeypatch, clean_env):
    _fake_installed(monkeypatch, {"tensorflow", "torch"})
    d = runtime.apply_cpu_default()
    assert (d["mode"], d["backend"], d["device"], d["forced"]) == ("cpu", "tensorflow", "cpu", False)
    assert os.environ["KERAS_BACKEND"] == "tensorflow"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert "CPU" in d["message"]


def test_cpu_default_falls_back_to_torch_without_tensorflow(monkeypatch, clean_env):
    _fake_installed(monkeypatch, {"torch"})
    assert runtime.apply_cpu_default()["backend"] == "torch"
    assert os.environ["KERAS_BACKEND"] == "torch"


def test_cpu_default_with_tensorflow_only(monkeypatch, clean_env):
    _fake_installed(monkeypatch, {"tensorflow"})
    assert runtime.choose_cpu_backend() == "tensorflow"


def test_cpu_default_without_any_backend_gives_clear_error(monkeypatch, clean_env):
    _fake_installed(monkeypatch, set())
    with pytest.raises(RuntimeError, match="tensorflow o torch"):
        runtime.apply_cpu_default()
    assert "KERAS_BACKEND" not in os.environ  # no se tocó nada


def test_cpu_default_override_by_environment_variable(monkeypatch, clean_env):
    _fake_installed(monkeypatch, {"tensorflow", "torch"})
    monkeypatch.setenv(runtime.BACKEND_ENV_VAR, "torch")
    d = runtime.apply_cpu_default()
    assert (d["backend"], d["forced"]) == ("torch", True)
    assert os.environ["KERAS_BACKEND"] == "torch"
    assert "forzado" in d["message"]


def test_cpu_default_argument_beats_environment_variable(monkeypatch, clean_env):
    _fake_installed(monkeypatch, {"tensorflow", "torch"})
    monkeypatch.setenv(runtime.BACKEND_ENV_VAR, "torch")
    assert runtime.apply_cpu_default("tensorflow")["backend"] == "tensorflow"


def test_cpu_default_override_errors(monkeypatch, clean_env):
    _fake_installed(monkeypatch, {"tensorflow"})
    with pytest.raises(RuntimeError, match="torch.*no está instalado"):
        runtime.apply_cpu_default("torch")
    with pytest.raises(ValueError, match="no admitido"):
        runtime.apply_cpu_default("caffe")
    monkeypatch.setenv(runtime.BACKEND_ENV_VAR, "jax")
    with pytest.raises(ValueError, match="no admitido"):
        runtime.choose_cpu_backend()


def test_restore_gpu_visibility_undoes_use_cpu_only(monkeypatch, clean_env):
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    runtime.use_cpu_only()
    runtime.use_cpu_only()  # una segunda llamada no pisa el valor guardado
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    runtime.restore_gpu_visibility()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    runtime.use_cpu_only()
    runtime.restore_gpu_visibility()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_restore_gpu_visibility_keeps_externally_hidden_gpu(monkeypatch, clean_env):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    runtime.use_cpu_only()
    runtime.restore_gpu_visibility()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


# --------------------------------------------------------------------------- #
# Modo automático (benchmark simulado)
# --------------------------------------------------------------------------- #
def _bench_df(rows):
    import pandas as pd

    return pd.DataFrame(rows, columns=["backend", "device", "batch_size", "status", "est_epoch_s"])


def _no_csv(monkeypatch, tmp_path):
    """Garantiza que el modo automático no lee ningún CSV (cualquier ``read_csv`` falla)."""
    import pandas as pd

    def boom(*a, **k):
        raise AssertionError("el modo automático no debe leer CSV")

    monkeypatch.setattr(pd, "read_csv", boom)


def test_auto_without_gpu_falls_back_to_cpu_and_says_so(monkeypatch, clean_env, tmp_path):
    _fake_installed(monkeypatch, {"tensorflow", "torch"})
    monkeypatch.setattr(runtime, "gpu_available", lambda b, **k: False)
    from amp_gan import benchmark

    def must_not_measure(*a, **k):
        raise AssertionError("sin GPU no se mide")

    monkeypatch.setattr(benchmark, "collect_benchmark", must_not_measure)
    _no_csv(monkeypatch, tmp_path)
    d, table = runtime.apply_auto_backend()
    assert table is None
    assert (d["mode"], d["backend"], d["device"], d["fallback"]) == ("auto", "tensorflow", "cpu", True)
    assert "GPU" in d["message"] and "CPU" in d["message"]
    assert os.environ["KERAS_BACKEND"] == "tensorflow" and os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_auto_picks_gpu_winner_and_keeps_gpu_visible(monkeypatch, clean_env, tmp_path):
    _fake_installed(monkeypatch, {"tensorflow", "torch"})
    monkeypatch.setattr(runtime, "gpu_available", lambda b, **k: b == "torch")
    from amp_gan import benchmark

    seen = {}

    def fake_collect(candidates, batch_sizes, steps, warmup, verbose):
        seen.update(candidates=candidates, batch_sizes=batch_sizes, steps=steps, warmup=warmup)
        return _bench_df([
            ("tensorflow", "cpu", 128, "ok", 3.0),
            ("torch", "cpu", 128, "ok", 3.4),
            ("torch", "gpu", 128, "ok", 1.9),
        ])  # fmt: skip

    monkeypatch.setattr(benchmark, "collect_benchmark", fake_collect)
    _no_csv(monkeypatch, tmp_path)
    runtime.use_cpu_only()  # como hace el notebook antes de la celda opcional
    d, table = runtime.apply_auto_backend(batch_sizes=(64,), steps=7, warmup=3, verbose=False)
    assert seen["candidates"] == (("tensorflow", "cpu"), ("torch", "cpu"), ("torch", "gpu"))
    assert seen["batch_sizes"] == (64, 128)  # siempre incluye el lote de entrenamiento
    assert (seen["steps"], seen["warmup"]) == (7, 3)
    assert (d["backend"], d["device"], d["fallback"]) == ("torch", "gpu", False)
    assert d["est_epoch_s"] == pytest.approx(1.9) and d["discarded"] == []
    assert len(table) == 3
    assert os.environ["KERAS_BACKEND"] == "torch"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"  # restaurada: la GPU se puede usar
    import json

    json.dumps(d)  # la decisión es serializable


def test_auto_tie_prefers_simplest_and_hides_gpu(monkeypatch, clean_env):
    _fake_installed(monkeypatch, {"tensorflow", "torch"})
    monkeypatch.setattr(runtime, "gpu_available", lambda b, **k: b == "torch")
    from amp_gan import benchmark

    monkeypatch.setattr(benchmark, "collect_benchmark", lambda **k: _bench_df([
        ("tensorflow", "cpu", 128, "ok", 3.0),
        ("torch", "gpu", 128, "ok", 2.9),
    ]))  # fmt: skip
    d, _ = runtime.apply_auto_backend(verbose=False)
    assert (d["backend"], d["device"], d["tied"]) == ("tensorflow", "cpu", True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_auto_discards_failing_candidate(monkeypatch, clean_env):
    _fake_installed(monkeypatch, {"tensorflow", "torch"})
    monkeypatch.setattr(runtime, "gpu_available", lambda b, **k: True)
    from amp_gan import benchmark

    monkeypatch.setattr(benchmark, "collect_benchmark", lambda **k: _bench_df([
        ("tensorflow", "cpu", 128, "ok", 3.0),
        ("torch", "cpu", 128, "ok", 3.4),
        ("torch", "gpu", 128, "error", None),
        ("tensorflow", "gpu", 128, "ok", 1.0),
        ("tensorflow", "gpu", 256, "error", None),
    ]))  # fmt: skip
    d, _ = runtime.apply_auto_backend(verbose=False)
    assert (d["backend"], d["device"], d["fallback"]) == ("tensorflow", "gpu", False)
    assert d["discarded"] == [["torch", "gpu"]]  # tensorflow/gpu falla en 256 pero vale en 128
    assert "Descartados" in d["message"]


def test_auto_falls_back_to_cpu_when_benchmark_fails(monkeypatch, clean_env):
    _fake_installed(monkeypatch, {"tensorflow", "torch"})
    monkeypatch.setattr(runtime, "gpu_available", lambda b, **k: True)
    from amp_gan import benchmark

    def broken(**k):
        raise OSError("sin disco")

    monkeypatch.setattr(benchmark, "collect_benchmark", broken)
    d, table = runtime.apply_auto_backend(verbose=False)
    assert table is None
    assert (d["backend"], d["device"], d["fallback"]) == ("tensorflow", "cpu", True)
    assert "sin disco" in d["message"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_auto_falls_back_to_cpu_when_no_candidate_is_valid(monkeypatch, clean_env):
    _fake_installed(monkeypatch, {"torch"})
    monkeypatch.setattr(runtime, "gpu_available", lambda b, **k: True)
    from amp_gan import benchmark

    monkeypatch.setattr(benchmark, "collect_benchmark", lambda **k: _bench_df([
        ("torch", "cpu", 128, "error", None),
        ("torch", "gpu", 128, "error", None),
    ]))  # fmt: skip
    d, _ = runtime.apply_auto_backend(verbose=False)
    assert (d["backend"], d["device"], d["fallback"]) == ("torch", "cpu", True)
    assert os.environ["KERAS_BACKEND"] == "torch"


def test_auto_without_any_backend_gives_clear_error(monkeypatch, clean_env):
    _fake_installed(monkeypatch, set())
    with pytest.raises(RuntimeError, match="tensorflow o torch"):
        runtime.apply_auto_backend(verbose=False)


def test_auto_respects_forced_backend(monkeypatch, clean_env):
    _fake_installed(monkeypatch, {"tensorflow", "torch"})
    monkeypatch.setattr(runtime, "gpu_available", lambda b, **k: True)
    monkeypatch.setenv(runtime.BACKEND_ENV_VAR, "torch")
    from amp_gan import benchmark

    seen = {}

    def fake(candidates, **k):
        seen["c"] = candidates
        return _bench_df([("torch", "cpu", 128, "ok", 3.0), ("torch", "gpu", 128, "ok", 2.0)])

    monkeypatch.setattr(benchmark, "collect_benchmark", fake)
    d, _ = runtime.apply_auto_backend(verbose=False)
    assert seen["c"] == (("torch", "cpu"), ("torch", "gpu"))
    assert (d["backend"], d["device"]) == ("torch", "gpu")

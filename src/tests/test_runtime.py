"""Tests de ``amp_gan.runtime`` (sin importar keras, tensorflow ni torch)."""

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


def test_apply_fastest_missing_csv_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="benchmark"):
        runtime.apply_fastest(tmp_path / "no_existe.csv", 128)


def test_apply_fastest_applies_winner(tmp_path, monkeypatch):
    import os

    import pandas as pd

    pd.DataFrame(
        {
            "backend": ["tensorflow", "torch"],
            "device": ["cpu", "gpu"],
            "batch_size": [128, 128],
            "status": ["ok", "ok"],
            "est_epoch_s": [10.0, 2.0],
        }
    ).to_csv(tmp_path / "b.csv", index=False)
    monkeypatch.delitem(sys.modules, "keras", raising=False)
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    choice = runtime.apply_fastest(tmp_path / "b.csv", 128)
    assert (choice["backend"], choice["device"]) == ("torch", "gpu")
    assert os.environ["KERAS_BACKEND"] == "torch"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"  # la GPU no se oculta


def test_apply_fastest_cpu_winner_hides_gpu(tmp_path, monkeypatch):
    import os

    import pandas as pd

    pd.DataFrame(
        {
            "backend": ["tensorflow", "torch"],
            "device": ["cpu", "gpu"],
            "batch_size": [128, 128],
            "status": ["ok", "ok"],
            "est_epoch_s": [2.0, 10.0],
        }
    ).to_csv(tmp_path / "b.csv", index=False)
    monkeypatch.delitem(sys.modules, "keras", raising=False)
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    runtime.apply_fastest(tmp_path / "b.csv", 128)
    assert os.environ["KERAS_BACKEND"] == "tensorflow"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

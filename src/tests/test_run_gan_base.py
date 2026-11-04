"""Tests de ``amp_gan.run_gan_base`` sin entrenar ni importar keras."""

import inspect
import sys

import pytest

from amp_gan import run_gan_base, runtime


def test_module_does_not_read_the_benchmark_csv():
    """El script no depende de ``benchmark_results.csv`` (ni lo menciona como entrada)."""
    assert not hasattr(run_gan_base, "BENCHMARK_CSV")
    code = inspect.getsource(run_gan_base._parse) + inspect.getsource(run_gan_base.main)
    assert "benchmark_results" not in code and "apply_fastest" not in code
    assert not hasattr(runtime, "apply_fastest")


def test_parse_defaults_to_cpu_mode():
    args = run_gan_base._parse([])
    assert args.auto_backend is False
    assert args.backend is None


def test_parse_auto_backend_flag_and_backend_choice():
    args = run_gan_base._parse(["--auto-backend", "--epochs", "2", "--backend", "torch"])
    assert args.auto_backend is True and args.backend == "torch" and args.epochs == 2
    with pytest.raises(SystemExit):
        run_gan_base._parse(["--backend", "caffe"])


def _stop_after_decision(monkeypatch):
    """Sustituye el modo elegido por un centinela que corta ``main`` justo después."""

    class Stop(Exception):
        pass

    calls = []

    def cpu(backend=None):
        calls.append(("cpu", backend))
        return {"mode": "cpu", "backend": "torch", "device": "cpu", "message": "cpu"}

    def auto(**kwargs):
        calls.append(("auto", kwargs))
        return {"mode": "auto", "backend": "torch", "device": "cpu", "message": "auto"}, None

    monkeypatch.setattr(runtime, "apply_cpu_default", cpu)
    monkeypatch.setattr(runtime, "apply_auto_backend", auto)
    # main() importa pandas/numpy y luego amp_gan.benchmark; cortamos en describe_environment.
    monkeypatch.setattr(runtime, "describe_environment", lambda: (_ for _ in ()).throw(Stop()))
    return calls, Stop


def test_main_default_uses_cpu_mode_and_needs_no_csv(monkeypatch, tmp_path):
    """Sin ``benchmark_results.csv`` y sin opciones, main elige el modo CPU y sigue adelante."""
    calls, Stop = _stop_after_decision(monkeypatch)
    import pandas as pd

    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("CSV")))
    with pytest.raises(Stop):  # llegó hasta describe_environment, es decir, no falló antes
        run_gan_base.main(["--out-dir", str(tmp_path)])
    assert calls == [("cpu", None)]


def test_main_auto_backend_uses_auto_mode(monkeypatch, tmp_path):
    calls, Stop = _stop_after_decision(monkeypatch)
    with pytest.raises(Stop):
        run_gan_base.main(["--auto-backend", "--backend", "torch", "--out-dir", str(tmp_path)])
    assert calls[0][0] == "auto"
    assert calls[0][1]["backend"] == "torch"
    assert calls[0][1]["batch_sizes"] == (runtime.TRAIN_BATCH_SIZE,)


def test_importing_run_gan_base_does_not_import_keras():
    # Si keras ya estaba importado por otro test no se puede afirmar; se comprueba en subproceso.
    import subprocess

    code = (
        "import sys; sys.path.insert(0, %r); from amp_gan import run_gan_base; "
        "sys.exit(1 if 'keras' in sys.modules else 0)" % str(run_gan_base.PROJECT_ROOT / "src")
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0

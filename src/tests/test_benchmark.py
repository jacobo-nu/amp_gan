"""Tests de la regla de decisión de ``amp_gan.benchmark`` (DataFrame sintético)."""

import pandas as pd
import pytest

from amp_gan.benchmark import fastest_by_batch, select_fastest


def _df(rows):
    return pd.DataFrame(rows, columns=["backend", "device", "batch_size", "status", "est_epoch_s"])


def test_clear_winner():
    df = _df(
        [
            ("tensorflow", "cpu", 128, "ok", 10.0),
            ("torch", "cpu", 128, "ok", 12.0),
            ("torch", "gpu", 128, "ok", 4.0),
        ]
    )
    d = select_fastest(df, 128)
    assert (d["backend"], d["device"]) == ("torch", "gpu")
    assert d["tied"] is False
    assert d["rel_diff"] == pytest.approx(1.5)
    assert d["runner_up"] == ("tensorflow", "cpu")


def test_tie_under_10_percent_prefers_simplest():
    # torch/gpu es la más rápida por 5 %, pero tensorflow/cpu está empatada y es más simple.
    df = _df(
        [
            ("torch", "gpu", 128, "ok", 10.0),
            ("tensorflow", "cpu", 128, "ok", 10.5),
        ]
    )
    d = select_fastest(df, 128)
    assert d["tied"] is True
    assert (d["backend"], d["device"]) == ("tensorflow", "cpu")
    assert d["fastest"] == ("torch", "gpu")
    assert "Empate" in d["message"]


def test_difference_of_exactly_10_percent_is_not_a_tie():
    df = _df([("torch", "gpu", 128, "ok", 10.0), ("tensorflow", "cpu", 128, "ok", 11.0)])
    d = select_fastest(df, 128)
    assert d["tied"] is False
    assert (d["backend"], d["device"]) == ("torch", "gpu")


def test_tie_keeps_first_when_it_is_already_simplest():
    df = _df([("tensorflow", "cpu", 128, "ok", 10.0), ("torch", "gpu", 128, "ok", 10.4)])
    d = select_fastest(df, 128)
    assert d["tied"] is True
    assert (d["backend"], d["device"]) == ("tensorflow", "cpu")


def test_error_rows_are_ignored():
    df = _df(
        [
            ("tensorflow", "gpu", 128, "error", None),
            ("tensorflow", "cpu", 128, "ok", 10.0),
            ("torch", "gpu", 128, "error", 1.0),  # error con valor residual: se ignora
        ]
    )
    d = select_fastest(df, 128)
    assert (d["backend"], d["device"]) == ("tensorflow", "cpu")
    assert d["runner_up"] is None and d["tied"] is False


def test_no_valid_rows_raises():
    df = _df([("tensorflow", "gpu", 128, "error", None), ("torch", "cpu", 256, "ok", 3.0)])
    with pytest.raises(ValueError, match="batch_size=128"):
        select_fastest(df, 128)


def test_fastest_by_batch_one_entry_per_batch_size():
    df = _df(
        [
            ("tensorflow", "cpu", 64, "ok", 5.0),
            ("torch", "gpu", 64, "ok", 2.0),
            ("tensorflow", "cpu", 128, "ok", 3.0),
            ("torch", "gpu", 128, "ok", 4.0),
            ("torch", "gpu", 256, "error", None),
        ]
    )
    res = {d["batch_size"]: (d["backend"], d["device"]) for d in fastest_by_batch(df)}
    assert res == {64: ("torch", "gpu"), 128: ("tensorflow", "cpu")}


# --- collect_benchmark / run_benchmark (sin lanzar subprocesos reales) -------------
def _fake_run_candidate(calls):
    def fake(backend, device, batch_sizes, steps, warmup, fasta):
        calls.append((backend, device, tuple(batch_sizes), steps, warmup))
        return [
            {"backend": backend, "device": device, "batch_size": bs, "status": "ok",
             "n_steps": steps, "n_warmup": warmup, "est_epoch_s": 1.0 + bs / 100,
             "n_samples": 100, "seed": 42}
            for bs in batch_sizes
        ]  # fmt: skip

    return fake


def test_collect_benchmark_returns_dataframe_without_writing(monkeypatch, tmp_path):
    from amp_gan import benchmark

    calls = []
    monkeypatch.setattr(benchmark, "_run_candidate", _fake_run_candidate(calls))
    monkeypatch.setattr(benchmark, "DEFAULT_CSV", tmp_path / "must_not_exist.csv")
    cands = (("tensorflow", "cpu"), ("torch", "gpu"))
    df = benchmark.collect_benchmark(candidates=cands, batch_sizes=(64, 128), steps=5, warmup=2)
    assert list(df.columns) == benchmark.CSV_COLUMNS
    assert len(df) == 4
    assert calls == [
        ("tensorflow", "cpu", (64, 128), 5, 2),
        ("torch", "gpu", (64, 128), 5, 2),
    ]
    assert df["n_steps"].dtype == "Int64" and (df["n_steps"] == 5).all()
    assert not list(tmp_path.iterdir())
    # La regla de decisión funciona sobre este DataFrame.
    assert select_fastest(df, 128)["backend"] == "tensorflow"


def test_run_benchmark_still_writes_csv(monkeypatch, tmp_path):
    from amp_gan import benchmark

    monkeypatch.setattr(benchmark, "_run_candidate", _fake_run_candidate([]))
    out = tmp_path / "sub" / "b.csv"
    df = benchmark.run_benchmark(
        candidates=(("torch", "cpu"),), batch_sizes=(128,), steps=3, warmup=1, csv_path=out
    )
    saved = pd.read_csv(out)
    assert list(saved.columns) == benchmark.CSV_COLUMNS
    assert len(saved) == len(df) == 1

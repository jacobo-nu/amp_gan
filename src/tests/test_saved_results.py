"""Tests de ``amp_gan.saved_results``: reconstruir un resultado desde archivos guardados (sin Keras)."""

import json
import sys

import pandas as pd
import pytest

from amp_gan.data import write_fasta
from amp_gan.saved_results import (
    COMMANDS,
    MODES,
    load_saved_experiment,
    missing_files,
    protocol_mismatches,
)

PREFIX = "cgan_length"


def _write_experiment(directory, conditional=True):
    metrics = pd.DataFrame(
        [
            {"epoch": 1, "mode": "argmax", "unique_fraction": 0.1, "length_mean": 8.0},
            {"epoch": 1, "mode": "sample", "unique_fraction": 0.9, "length_mean": 9.0},
            {"epoch": 2, "mode": "argmax", "unique_fraction": 0.2, "length_mean": 10.0},
            {"epoch": 2, "mode": "sample", "unique_fraction": 1.0, "length_mean": 11.0},
            {"epoch": None, "mode": "real_subset", "unique_fraction": 1.0, "length_mean": 20.0},
        ]
    )
    metrics.to_csv(directory / f"{PREFIX}_metrics.csv", index=False)
    pd.DataFrame({"epoch": [1, 2], "d_loss_real": [0.7, 0.6], "g_loss": [1.2, 1.3]}).to_csv(
        directory / f"{PREFIX}_history.csv", index=False
    )
    info = {
        "timing_s": {"total_s": 3.0, "train_s": 2.5, "eval_s": 0.5, "mean_epoch_s": 1.25, "median_epoch_s": 1.25},
        "snapshot_epochs": [1, 2],
        "epochs": 2,
        "snapshot_every": 1,
        "n_eval": 3,
        "condition_spec": {"names": ["length"], "low": [5.0], "high": [60.0]},
    }
    (directory / f"{PREFIX}_run_info.json").write_text(json.dumps(info), encoding="utf-8")
    for mode in MODES:
        write_fasta(pd.DataFrame({"id": ["a", "b", "c"], "sequence": ["KRIV", "", "LLKK"]}), directory / f"{PREFIX}_generated_{mode}.fasta")
    if conditional:
        pd.DataFrame(
            {"epoch": [2, 2], "mode": ["sample", "sample"], "target": [9, 13], "hit_rate": [0.5, 0.4]}
        ).to_csv(directory / f"{PREFIX}_targets.csv", index=False)
        write_fasta(
            pd.DataFrame({"id": ["gen_t09_0000", "gen_t09_0001", "gen_t13_0000"], "sequence": ["KRIVLKRIV", "AAAAAAAAA", "K" * 13]}),
            directory / f"{PREFIX}_generated_targets_sample.fasta",
        )


def test_carga_condicional(tmp_path):
    _write_experiment(tmp_path)
    res = load_saved_experiment(PREFIX, tmp_path, conditional=True)
    assert res.snapshot_epochs == [1, 2]
    assert list(res.metrics["epoch"].unique()) == [1, 2]  # sin la fila de suelo
    assert res.metrics["epoch"].dtype.kind == "i"
    assert res.floors["real_subset"]["length_mean"] == 20.0
    assert "epoch" not in res.history and res.history["g_loss"] == [1.2, 1.3]
    assert res.history.spec.names == ["length"] and res.history.spec.low == [5.0]
    assert res.sequences["sample"] == ["KRIV", "", "LLKK"]  # la secuencia vacía se conserva
    assert res.timing["mean_epoch_s"] == 1.25
    assert res.epoch_seconds == []
    assert list(res.targets["target"]) == [9, 13]
    assert {t: len(v) for t, v in res.target_sequences["sample"].items()} == {9: 2, 13: 1}


def test_carga_no_condicional_sin_objetivos(tmp_path):
    _write_experiment(tmp_path, conditional=False)
    res = load_saved_experiment(PREFIX, tmp_path)
    assert res.targets is None and res.target_sequences is None
    assert res.history.spec is None


def test_archivos_que_faltan_dan_error_con_el_comando(tmp_path):
    _write_experiment(tmp_path)
    (tmp_path / f"{PREFIX}_targets.csv").unlink()
    assert missing_files(PREFIX, tmp_path, conditional=True) == [f"{PREFIX}_targets.csv"]
    assert missing_files(PREFIX, tmp_path) == []
    with pytest.raises(FileNotFoundError, match="run_conditional_gan.py --variant cgan_length"):
        load_saved_experiment(PREFIX, tmp_path, conditional=True)


def test_identificador_inesperado_en_objetivos(tmp_path):
    _write_experiment(tmp_path)
    write_fasta(pd.DataFrame({"id": ["raro"], "sequence": ["KRIV"]}), tmp_path / f"{PREFIX}_generated_targets_sample.fasta")
    with pytest.raises(ValueError, match="Identificador inesperado"):
        load_saved_experiment(PREFIX, tmp_path, conditional=True)


def test_protocolo(tmp_path):
    _write_experiment(tmp_path)
    res = load_saved_experiment(PREFIX, tmp_path, conditional=True)
    assert protocol_mismatches(res, 2, 1, 3) == []
    assert len(protocol_mismatches(res, 200, 20, 3)) == 2


def test_todos_los_prefijos_tienen_comando():
    assert set(COMMANDS) == {"gan_base", "gan_stab", "gan_stabc", "wgan_gp", "cgan_length"}


def test_el_modulo_no_importa_keras():
    # Se comprueba en un subproceso limpio: el backend debe poder fijarse después de importarlo.
    import subprocess

    code = "import sys; sys.path.insert(0, %r); import amp_gan.saved_results, amp_gan.formato; print('keras' in sys.modules)" % str(
        __import__("pathlib").Path(__file__).resolve().parents[1]
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"

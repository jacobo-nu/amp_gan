"""Carga de resultados de experimentos ya guardados en ``context/data/processed/`` (sin reentrenar).

Los scripts ``run_gan_base.py`` y ``run_conditional_gan.py`` guardan, para cada
experimento ``<prefijo>``: ``<prefijo>_metrics.csv`` (una fila por instantánea y
modo, más las filas de suelo sin época), ``<prefijo>_history.csv`` (pérdidas por
época), ``<prefijo>_run_info.json`` (configuración, versiones, tiempos) y las
secuencias de la época final (``<prefijo>_generated_sample.fasta`` y
``<prefijo>_generated_argmax.fasta``). Los experimentos condicionales guardan
además ``<prefijo>_targets.csv`` (evaluación por objetivos por instantánea) y
``<prefijo>_generated_targets_sample.fasta`` (secuencias pedidas con cada
longitud objetivo, solo en modo sample).

:func:`load_saved_experiment` reconstruye con esos archivos un objeto con los
**mismos atributos** que el resultado de ``run_gan_experiment`` /
``run_conditional_experiment`` (``history``, ``metrics``, ``sequences``,
``timing``, ``snapshot_epochs``, y ``targets``/``target_sequences`` en los
condicionales), de modo que el código de análisis de los notebooks es el mismo
entrenando que cargando. Lo que no se guarda (tiempos por época, modelos
entrenados, secuencias por objetivo en modo argmax) queda vacío o ``None``.

Este módulo **no importa Keras**.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from amp_gan.data import read_fasta

#: Modos de decodificación guardados (los mismos que ``amp_gan.experiment.MODES``).
MODES: tuple[str, ...] = ("argmax", "sample")

#: Comando que genera cada prefijo (para el mensaje de error si faltan archivos).
COMMANDS: dict[str, str] = {
    "gan_base": "python src/amp_gan/run_gan_base.py --variant base",
    "gan_stab": "python src/amp_gan/run_gan_base.py --variant stabilized",
    "gan_stabc": "python src/amp_gan/run_gan_base.py --variant stabilized_const",
    "wgan_gp": "python src/amp_gan/run_gan_base.py --variant wgan_gp",
    "cgan_length": "python src/amp_gan/run_conditional_gan.py --variant cgan_length",
}

_BASE_SUFFIXES = ("metrics.csv", "history.csv", "run_info.json", "generated_sample.fasta", "generated_argmax.fasta")
_CONDITIONAL_SUFFIXES = ("targets.csv", "generated_targets_sample.fasta")
_TARGET_ID = re.compile(r"^gen_t(\d+)_")


class SavedHistory(dict):
    """Historial de pérdidas (diccionario de listas, una por columna) con ``spec`` en los condicionales."""

    spec: SimpleNamespace | None = None


@dataclass
class SavedExperiment:
    """Resultado reconstruido desde archivos (ver el docstring del módulo).

    Attributes:
        prefix: prefijo de los archivos (``gan_stabc``, ``wgan_gp``...).
        history: pérdidas por época, sin la columna ``epoch``.
        metrics: una fila por (``epoch``, ``mode``), sin las filas de suelo.
        sequences: secuencias de la época final por modo.
        timing: tiempos (``total_s``, ``train_s``, ``eval_s``, ``mean_epoch_s``, ``median_epoch_s``).
        snapshot_epochs: épocas con instantánea.
        info: contenido de ``run_info.json``.
        floors: filas de suelo del CSV de métricas, ``{nombre: {métrica: valor}}``.
        targets: evaluación por objetivos (solo condicionales; si no, ``None``).
        target_sequences: ``{"sample": {objetivo: [secuencias]}}`` (solo condicionales).
        epoch_seconds: siempre vacío (no se guarda el tiempo de cada época).
    """

    prefix: str
    history: SavedHistory
    metrics: pd.DataFrame
    sequences: dict[str, list[str]]
    timing: dict[str, float]
    snapshot_epochs: list[int]
    info: dict
    floors: dict[str, dict[str, float]]
    targets: pd.DataFrame | None = None
    target_sequences: dict[str, dict[int, list[str]]] | None = None
    epoch_seconds: list[float] = field(default_factory=list)


def missing_files(prefix: str, directory: Path, conditional: bool = False) -> list[str]:
    """Nombres de los archivos que faltan para cargar ``prefix``."""
    suffixes = _BASE_SUFFIXES + (_CONDITIONAL_SUFFIXES if conditional else ())
    return [f"{prefix}_{s}" for s in suffixes if not (Path(directory) / f"{prefix}_{s}").exists()]


def load_saved_experiment(prefix: str, directory: Path | str, conditional: bool = False) -> SavedExperiment:
    """Carga el experimento ``prefix`` de ``directory``.

    Args:
        prefix: prefijo de los archivos.
        directory: carpeta con los archivos (``context/data/processed``).
        conditional: ``True`` para un experimento condicional por longitud (carga también los objetivos).

    Raises:
        FileNotFoundError: si falta algún archivo; el mensaje incluye el comando que lo genera.
    """
    directory = Path(directory)
    missing = missing_files(prefix, directory, conditional)
    if missing:
        command = COMMANDS.get(prefix, "el script correspondiente de src/amp_gan/")
        raise FileNotFoundError(
            f"Faltan resultados guardados en {directory} ({', '.join(missing)}).\n"
            f"Genéralos desde la raíz del proyecto con:\n    {command}"
        )

    info = json.loads((directory / f"{prefix}_run_info.json").read_text(encoding="utf-8"))

    raw = pd.read_csv(directory / f"{prefix}_metrics.csv")
    has_epoch = raw["epoch"].notna()
    metrics = raw[has_epoch].astype({"epoch": int}).reset_index(drop=True)
    floors = {
        str(row["mode"]): {k: float(v) for k, v in row.drop(["epoch", "mode"]).items()}
        for _, row in raw[~has_epoch].iterrows()
    }

    history = SavedHistory({k: v for k, v in pd.read_csv(directory / f"{prefix}_history.csv").to_dict(orient="list").items() if k != "epoch"})
    if conditional and "condition_spec" in info:
        history.spec = SimpleNamespace(**info["condition_spec"])

    sequences = {mode: read_fasta(directory / f"{prefix}_generated_{mode}.fasta")["sequence"].tolist() for mode in MODES}

    targets = target_sequences = None
    if conditional:
        targets = pd.read_csv(directory / f"{prefix}_targets.csv")
        targets = targets.astype({"epoch": int, "target": int})
        target_sequences = {"sample": _read_target_sequences(directory / f"{prefix}_generated_targets_sample.fasta")}

    return SavedExperiment(
        prefix=prefix,
        history=history,
        metrics=metrics,
        sequences=sequences,
        timing=dict(info["timing_s"]),
        snapshot_epochs=[int(e) for e in info["snapshot_epochs"]],
        info=info,
        floors=floors,
        targets=targets,
        target_sequences=target_sequences,
    )


def _read_target_sequences(path: Path) -> dict[int, list[str]]:
    """Agrupa por longitud objetivo las secuencias del FASTA (identificadores ``gen_t09_0000``)."""
    table = read_fasta(path)
    out: dict[int, list[str]] = {}
    for ident, seq in zip(table["id"], table["sequence"]):
        match = _TARGET_ID.match(str(ident))
        if match is None:
            raise ValueError(f"Identificador inesperado en {path.name}: {ident!r} (se esperaba gen_tNN_NNNN).")
        out.setdefault(int(match.group(1)), []).append(seq)
    return out


def protocol_mismatches(result: SavedExperiment, epochs: int, snapshot_every: int, n_eval: int) -> list[str]:
    """Diferencias entre el protocolo pedido y el del resultado guardado (lista vacía si coincide)."""
    info = result.info
    wanted = {"epochs": epochs, "snapshot_every": snapshot_every, "n_eval": n_eval}
    return [f"{k}: pedido {v}, guardado {info.get(k)}" for k, v in wanted.items() if info.get(k) != v]

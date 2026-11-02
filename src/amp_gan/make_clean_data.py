"""Genera las bases limpias (5-60 aa, alfabeto estándar, sin duplicados).

Para cada base (``Antibacterial_amps``, ``general_amps``) lee
``context/data/raw/<base>.fasta``, aplica :func:`amp_gan.data.load_clean` con los
valores por defecto y escribe en ``context/data/processed/``:

- ``<base>_clean.fasta``: secuencias limpias (``>ID`` y secuencia en una línea).
- ``cleaning_stats.csv``: una fila por base con las estadísticas de limpieza,
  la longitud (mín/media/mediana/máx) de lo que queda y los límites usados.

Las bases se procesan por separado, nunca se fusionan. Es determinista.

Uso: ``python src/amp_gan/make_clean_data.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amp_gan.data import MAX_LEN, MIN_LEN, load_clean, write_fasta  # noqa: E402

RAW_DIR = PROJECT_ROOT / "context" / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "context" / "data" / "processed"

#: Bases a limpiar (nombre del archivo sin extensión).
DATASETS: tuple[str, ...] = ("Antibacterial_amps", "general_amps")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for name in DATASETS:
        df, stats = load_clean(RAW_DIR / f"{name}.fasta")
        write_fasta(df, OUT_DIR / f"{name}_clean.fasta")
        lengths = df["sequence"].str.len()
        rows.append(
            {
                "dataset": name,
                **stats,
                "len_min": int(lengths.min()),
                "len_mean": round(float(lengths.mean()), 2),
                "len_median": float(lengths.median()),
                "len_max": int(lengths.max()),
                "min_len": MIN_LEN,
                "max_len": MAX_LEN,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "cleaning_stats.csv", index=False)


if __name__ == "__main__":
    main()

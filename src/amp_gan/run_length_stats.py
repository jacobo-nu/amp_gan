"""Calcula y compara estadísticas de longitud de las bases de AMP (sin recortar).

Genera en ``context/data/processed/``:

- ``length_stats.csv``: resumen de longitudes por base y variante.
- ``length_coverage.csv``: cobertura por corte de longitud, por base y variante.
- ``length_hist.png``: histogramas de longitud de ambas bases (variante A).

Variantes: ``A_raw`` (todas las secuencias del FASTA) y ``B_clean`` (alfabeto
estándar y sin duplicados, sin filtro de longitud).

Uso: ``python src/amp_gan/run_length_stats.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amp_gan.data import clean_sequences, read_fasta  # noqa: E402
from amp_gan.length_stats import coverage_table, length_summary, sequence_lengths  # noqa: E402

RAW_DIR = PROJECT_ROOT / "context" / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "context" / "data" / "processed"

#: Bases a analizar: nombre -> archivo FASTA. La principal va primero.
DATASETS: dict[str, str] = {
    "Antibacterial_amps": "Antibacterial_amps.fasta",
    "general_amps": "general_amps.fasta",
}


def build_variants(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Devuelve las variantes de una base: ``A_raw`` y ``B_clean`` (sin filtro de longitud)."""
    clean, _ = clean_sequences(df, min_len=1, max_len=10**6)
    return {"A_raw": df, "B_clean": clean}


def plot_histograms(lengths_by_dataset: dict[str, np.ndarray], path: Path) -> None:
    """Dibuja histogramas de longitud (cuentas) con ejes comunes y vista de la cola > 50."""
    names = list(lengths_by_dataset)
    x_max = max(int(v.max()) for v in lengths_by_dataset.values())
    bins = np.arange(0.5, x_max + 1.5, 1.0)  # un bin por longitud entera
    tail_bins = np.arange(50.5, x_max + 1.5, 1.0)
    colors = {names[0]: "tab:blue", names[1]: "tab:orange"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey="row")
    for col, name in enumerate(names):
        lens = lengths_by_dataset[name]
        mean, median = float(lens.mean()), float(np.median(lens))
        for row, (b, lo) in enumerate(((bins, 0), (tail_bins, 50))):
            ax = axes[row, col]
            ax.hist(lens[lens > lo], bins=b, color=colors[name], alpha=0.85,
                    label=f"{name} (n={len(lens)})")
            if row == 0:  # en la cola (> 50) la media y la mediana quedan fuera del rango
                ax.axvline(mean, color="red", linestyle="--", linewidth=1.5, label=f"media = {mean:.1f}")
                ax.axvline(median, color="black", linestyle=":", linewidth=1.5, label=f"mediana = {median:.1f}")
                ax.axvline(50.5, color="gray", linewidth=0.8, label="inicio de la cola (> 50)")
            ax.set_xlim(lo + 0.5, x_max + 0.5)
            ax.set_xlabel("Longitud (aminoácidos)")
            if col == 0:
                ax.set_ylabel("Nº de secuencias por longitud")
            ax.set_title(
                f"{name}: todas las longitudes" if row == 0 else f"{name}: cola > 50 (ampliada)"
            )
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(alpha=0.3)
    fig.suptitle("Distribución de longitudes (variante A: datos tal cual vienen)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    coverage_frames: list[pd.DataFrame] = []
    lengths_raw: dict[str, np.ndarray] = {}

    for name, filename in DATASETS.items():
        df = read_fasta(RAW_DIR / filename)
        frames[name] = df
        for variant, vdf in build_variants(df).items():
            lengths = sequence_lengths(vdf["sequence"])
            summary_rows.append({"dataset": name, "variant": variant, **length_summary(lengths)})
            cov = coverage_table(lengths)
            cov.insert(0, "variant", variant)
            cov.insert(0, "dataset", name)
            coverage_frames.append(cov)
            if variant == "A_raw":
                lengths_raw[name] = lengths

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "length_stats.csv", index=False, float_format="%.6g")
    pd.concat(coverage_frames, ignore_index=True).to_csv(
        OUT_DIR / "length_coverage.csv", index=False, float_format="%.6f"
    )
    plot_histograms(lengths_raw, OUT_DIR / "length_hist.png")

    # Solapamiento general -> Antibacterial (informativo, solo por pantalla).
    gen, anti = frames["general_amps"], frames["Antibacterial_amps"]
    seq_in = gen["sequence"].isin(set(anti["sequence"]))
    id_in = gen["id"].isin(set(anti["id"]))
    both = seq_in & id_in
    n = len(gen)
    print(f"general_amps: n={n}; secuencias únicas={gen['sequence'].nunique()}")
    print(f"Antibacterial_amps: n={len(anti)}; secuencias únicas={anti['sequence'].nunique()}")
    print(f"general en Antibacterial por secuencia exacta: {int(seq_in.sum())} ({seq_in.mean():.4%})")
    print(f"general en Antibacterial por ID: {int(id_in.sum())} ({id_in.mean():.4%})")
    print(f"por secuencia y por ID a la vez: {int(both.sum())}; solo secuencia: "
          f"{int((seq_in & ~id_in).sum())}; solo ID: {int((id_in & ~seq_in).sum())}")
    uniq_gen = set(gen["sequence"])
    inter = uniq_gen & set(anti["sequence"])
    print(f"secuencias únicas de general presentes en Antibacterial: {len(inter)} / {len(uniq_gen)} "
          f"({len(inter) / len(uniq_gen):.4%})")
    print(f"Ficheros escritos en {OUT_DIR}")


if __name__ == "__main__":
    main()

"""Entrena la GAN base con Antibacterial_amps y guarda los resultados.

Uso (desde la raíz del proyecto)::

    python src/amp_gan/run_gan_base.py
    python src/amp_gan/run_gan_base.py --epochs 10 --out-dir carpeta_de_prueba

Pasos: aplica ``runtime.apply_fastest`` (backend y dispositivo más rápidos según
``benchmark_results.csv``, lote 128) ANTES de importar Keras; entrena con la
configuración por defecto ``GANConfig()`` (sin ajustar hiperparámetros, es la
línea base) durante 200 épocas con instantáneas cada 20 y 2000 secuencias por
evaluación; escribe en ``context/data/processed/`` (formato estable, usado por
la comparación de la Fase 3):

``gan_base_history.csv``
    una fila por época. Columnas: ``epoch`` (desde 1), ``d_loss_real``,
    ``d_loss_fake``, ``d_loss`` (media de las dos) y ``g_loss`` (medias por época).
``gan_base_metrics.csv``
    una fila por (``epoch``, ``mode``) con ``mode`` en {``argmax``, ``sample``}
    para cada instantánea (épocas 1, 20, 40, ..., 200), más dos filas de suelo
    con ``epoch`` vacío y ``mode`` = ``real_subset`` / ``random_uniform``
    (ver :func:`amp_gan.evaluation.reference_floors`). Las columnas métricas son
    las de :data:`amp_gan.evaluation.METRIC_NAMES`: ``n``, ``n_valid``,
    ``valid_fraction``, ``unique_fraction``, ``novel_fraction``,
    ``{length,net_charge,hydrophobicity}_{mean,std,wasserstein}`` y
    ``composition_tv``. Carga a pH 7,4; hidrofobicidad = GRAVY (Kyte-Doolittle).
``gan_base_generated_sample.fasta`` / ``gan_base_generated_argmax.fasta``
    las 2000 secuencias de la última época por modo de decodificación, con ids
    ``gen_0000`` ... ``gen_1999`` (una línea de secuencia por registro). Se
    conservan las secuencias no válidas o vacías para que haya siempre 2000
    registros (una secuencia vacía aparece como línea vacía).
``gan_base_run_info.json``
    versiones (python, numpy, pandas, scipy, keras, tensorflow, torch), backend
    y dispositivo usados, decisión del benchmark, configuración de la GAN,
    épocas, parámetros de evaluación, tamaño de los datos y tiempos (s).

No se guardan los pesos de los modelos.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amp_gan import runtime  # noqa: E402  (no importa keras)

PROCESSED = PROJECT_ROOT / "context" / "data" / "processed"
TRAIN_FASTA = PROCESSED / "Antibacterial_amps_clean.fasta"
BENCHMARK_CSV = PROCESSED / "benchmark_results.csv"

EPOCHS = 200
SNAPSHOT_EVERY = 20
N_EVAL = 2000
SEED = 42
PREFIX = "gan_base"


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    p.add_argument("--n-eval", type=int, default=N_EVAL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out-dir", type=Path, default=PROCESSED, help="carpeta de salida")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    # 1) Backend y dispositivo ANTES de importar keras.
    choice = runtime.apply_fastest(BENCHMARK_CSV, batch_size=128)
    print(f"[run_gan_base] decisión del benchmark: {choice['message']}", flush=True)

    # 2) Ahora sí: keras y los módulos que lo importan.
    import numpy as np
    import pandas as pd
    import scipy

    from amp_gan.benchmark import weights_device
    from amp_gan.data import write_fasta
    from amp_gan.encoding import load_encoded
    from amp_gan.evaluation import reference_floors
    from amp_gan.experiment import MODES, run_gan_experiment
    from amp_gan.gan import GANConfig

    env = runtime.describe_environment()
    print(f"[run_gan_base] backend={env['keras_backend']} dispositivos={env['keras_devices']}", flush=True)
    if env["keras_backend"] != choice["backend"]:
        raise RuntimeError(f"Backend pedido {choice['backend']}, usado {env['keras_backend']}.")

    X, df = load_encoded(TRAIN_FASTA)
    print(f"[run_gan_base] datos: {X.shape} ({len(df):,} secuencias)", flush=True)

    config = GANConfig(seed=args.seed)
    res = run_gan_experiment(
        X, df, config, epochs=args.epochs, snapshot_every=args.snapshot_every,
        n_eval=args.n_eval, eval_seed=args.seed,
    )  # fmt: skip
    device_used = weights_device(res.history.generator)
    print(f"[run_gan_base] pesos del generador en: {device_used}", flush=True)

    # 3) Salidas.
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hist = pd.DataFrame(
        {"epoch": np.arange(1, args.epochs + 1)}
        | {k: res.history[k] for k in ("d_loss_real", "d_loss_fake", "d_loss", "g_loss")}
    )
    hist.to_csv(out / f"{PREFIX}_history.csv", index=False, float_format="%.8g")

    floors = reference_floors(df, n=args.n_eval, seed=args.seed)
    floor_rows = pd.DataFrame(
        [{"epoch": pd.NA, "mode": name, **vals} for name, vals in floors.items()]
    )
    metrics = pd.concat([res.metrics, floor_rows], ignore_index=True)
    metrics["epoch"] = metrics["epoch"].astype("Int64")
    metrics.to_csv(out / f"{PREFIX}_metrics.csv", index=False, float_format="%.8g")

    for mode in MODES:
        seqs = res.sequences[mode]
        write_fasta(
            pd.DataFrame({"id": [f"gen_{i:04d}" for i in range(len(seqs))], "sequence": seqs}),
            out / f"{PREFIX}_generated_{mode}.fasta",
        )

    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            **{k: env[k] for k in ("python_version", "numpy_version", "keras_version",
                                    "tensorflow_version", "torch_version")},  # fmt: skip
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "platform": platform.platform(),
        },
        "keras_backend": env["keras_backend"],
        "keras_devices": env["keras_devices"],
        "gpu_visible": env["gpu_visible"],
        "cuda_visible_devices": env["cuda_visible_devices"],
        "device_used": device_used,
        "benchmark_decision": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in choice.items()
        },
        "config": asdict(config),
        "epochs": args.epochs,
        "snapshot_every": args.snapshot_every,
        "snapshot_epochs": res.snapshot_epochs,
        "n_eval": args.n_eval,
        "eval_seed": args.seed,
        "train_fasta": TRAIN_FASTA.name,
        "n_train": int(X.shape[0]),
        "timing_s": res.timing,
        "final_epoch_losses": {k: res.history[k][-1] for k in ("d_loss_real", "d_loss_fake", "g_loss")},
    }
    (out / f"{PREFIX}_run_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    t = res.timing
    print(f"[run_gan_base] tiempo total {t['total_s']:.1f} s (entrenamiento {t['train_s']:.1f} s, "
          f"evaluación {t['eval_s']:.1f} s); por época: media {t['mean_epoch_s']:.2f} s, "
          f"mediana {t['median_epoch_s']:.2f} s", flush=True)  # fmt: skip
    print(f"[run_gan_base] archivos escritos en {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Entrena la GAN condicional con condición numérica (Antibacterial_amps) y guarda los resultados.

Uso (desde la raíz del proyecto)::

    python src/amp_gan/run_conditional_gan.py --variant cgan_length --auto-backend   # ejecución completa
    python src/amp_gan/run_conditional_gan.py --variant cgan_length --epochs 3 --max-steps 20 --out-dir carpeta_de_prueba

Variante ``cgan_length`` (prefijo ``cgan_length``): condición = longitud
normalizada ``(L - 5) / 55`` sobre la base ``GANConfig.stabilized_constant_noise()``
(ver :mod:`amp_gan.conditional_gan`). Se entrena con TODAS las longitudes (5-60);
los objetivos que se piden y evalúan son los percentiles 10, 30, 50, 70 y 90 de la
longitud real (9, 13, 20, 21 y 40 aa; el rango «habitual» p10-p90 es 9-40 aa), con
tolerancia de acierto de ±2 aa. Protocolo idéntico al de ``run_gan_base.py``: 200
épocas, instantáneas cada 20 (y la primera), 2000 secuencias por evaluación,
semilla 42, sin ajustar hiperparámetros. Selección de backend igual que en
``run_gan_base.py`` (por defecto CPU; ``--auto-backend`` mide y usa el más rápido;
``--backend`` lo fuerza).

Archivos escritos en ``context/data/processed/`` (o ``--out-dir``), con prefijo ``cgan_length``:

``cgan_length_history.csv``
    columnas ``epoch``, ``d_loss_real``, ``d_loss_fake``, ``d_loss``, ``g_loss`` (como la base).
``cgan_length_metrics.csv``
    evaluación (i) «empírica»: condiciones tomadas de longitudes reales (muestreo con
    reposición de los datos de entrenamiento, semilla 42). Mismo formato que el de las
    demás variantes (filas por época y modo, más los suelos ``real_subset`` y ``random_uniform``).
``cgan_length_targets.csv``
    evaluación (ii) por objetivos: una fila por (``epoch``, ``mode``, ``target``) con las
    columnas de :data:`amp_gan.conditions.TARGET_COLUMNS`.
``cgan_length_generated_{sample,argmax}.fasta``
    las 2000 secuencias de la evaluación (i) en la última época (ids ``gen_0000`` ...).
``cgan_length_generated_targets_sample.fasta``
    las 2000 secuencias pedidas con cada objetivo en la última época, modo sample
    (ids ``gen_t09_0000``, ``gen_t13_0000``, ...).
``cgan_length_run_info.json``
    versiones, backend y dispositivo, configuración, condiciones, objetivos, tiempos y las
    referencias sin condición de la evaluación por objetivos.

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
from amp_gan.run_gan_base import (  # noqa: E402  (no importa keras)
    EPOCHS,
    GAN_HISTORY_COLUMNS,
    N_EVAL,
    PROCESSED,
    SEED,
    SNAPSHOT_EVERY,
    TRAIN_FASTA,
)

#: Variantes: nombre -> (prefijo de archivos, condiciones numéricas).
VARIANTS: dict[str, tuple[str, tuple[str, ...]]] = {"cgan_length": ("cgan_length", ("length",))}

#: GAN de referencia sin condicionar, de la que se leen las secuencias para la comparación por objetivos.
REFERENCE_PREFIX = "gan_stabc"


def make_config(variant: str, seed: int = SEED):
    """``ConditionalGANConfig.stabilized_constant_noise`` con las condiciones de la variante (importa keras)."""
    from amp_gan.conditional_gan import ConditionalGANConfig

    try:
        conditions = VARIANTS[variant][1]
    except KeyError:
        raise ValueError(f"Variante desconocida: {variant!r}.") from None
    return ConditionalGANConfig.stabilized_constant_noise(seed=seed, conditions=conditions)


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", choices=sorted(VARIANTS), default="cgan_length")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    p.add_argument("--n-eval", type=int, default=N_EVAL)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-steps", type=int, default=None,
                   help="tope de pasos por época (solo pruebas rápidas; por defecto, una pasada por los datos)")  # fmt: skip
    p.add_argument("--out-dir", type=Path, default=PROCESSED, help="carpeta de salida")
    p.add_argument("--auto-backend", action="store_true",
                   help="medir backends/dispositivos disponibles y usar el más rápido (por defecto: CPU)")  # fmt: skip
    p.add_argument("--backend", choices=runtime.SUPPORTED_BACKENDS, default=None,
                   help="fuerza el backend de Keras (también AMP_GAN_BACKEND)")  # fmt: skip
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    prefix, conditions = VARIANTS[args.variant]
    tag = f"run_conditional_gan:{args.variant}"

    # 1) Backend y dispositivo ANTES de importar keras.
    if args.auto_backend:
        choice, _ = runtime.apply_auto_backend(backend=args.backend, batch_sizes=(runtime.TRAIN_BATCH_SIZE,))
    else:
        choice = runtime.apply_cpu_default(args.backend)
    print(f"[{tag}] {choice['message']}", flush=True)

    # 2) Ahora sí: keras y los módulos que lo importan.
    import numpy as np
    import pandas as pd
    import scipy

    from amp_gan.benchmark import weights_device
    from amp_gan.conditional_gan import run_conditional_experiment
    from amp_gan.conditions import (
        HIT_TOLERANCE,
        TARGET_PERCENTILES,
        length_percentiles,
        real_ceiling,
        target_lengths,
        unconditional_target_table,
        usual_length_range,
    )
    from amp_gan.data import read_fasta, write_fasta
    from amp_gan.encoding import load_encoded
    from amp_gan.evaluation import reference_floors
    from amp_gan.experiment import MODES

    env = runtime.describe_environment()
    print(f"[{tag}] backend={env['keras_backend']} dispositivos={env['keras_devices']}", flush=True)
    if env["keras_backend"] != choice["backend"]:
        raise RuntimeError(f"Backend pedido {choice['backend']}, usado {env['keras_backend']}.")

    X, df = load_encoded(TRAIN_FASTA)
    print(f"[{tag}] datos: {X.shape} ({len(df):,} secuencias)", flush=True)

    real_lengths = df["sequence"].str.len().to_numpy()
    targets = target_lengths(real_lengths)
    usual = usual_length_range(real_lengths)
    print(f"[{tag}] objetivos (percentiles {TARGET_PERCENTILES}): {targets}; rango habitual p10-p90: {usual}", flush=True)

    config = make_config(args.variant, args.seed)
    res = run_conditional_experiment(
        X, df, config, epochs=args.epochs, snapshot_every=args.snapshot_every, n_eval=args.n_eval,
        eval_seed=args.seed, targets=targets, tol=HIT_TOLERANCE, max_steps_per_epoch=args.max_steps,
    )  # fmt: skip
    device_used = weights_device(res.history.generator)
    print(f"[{tag}] pesos del generador en: {device_used}", flush=True)
    if device_used != choice["device"]:
        print(f"[{tag}] AVISO: se decidió {choice['device']} pero los pesos están en {device_used}.", flush=True)

    # 3) Salidas.
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hist = pd.DataFrame(
        {"epoch": np.arange(1, args.epochs + 1)} | {k: res.history[k] for k in GAN_HISTORY_COLUMNS}
    )
    hist.to_csv(out / f"{prefix}_history.csv", index=False, float_format="%.8g")

    floors = reference_floors(df, n=args.n_eval, seed=args.seed)
    floor_rows = pd.DataFrame([{"epoch": pd.NA, "mode": name, **vals} for name, vals in floors.items()])
    metrics = pd.concat([res.metrics, floor_rows], ignore_index=True)
    metrics["epoch"] = metrics["epoch"].astype("Int64")
    metrics.to_csv(out / f"{prefix}_metrics.csv", index=False, float_format="%.8g")

    res.targets.to_csv(out / f"{prefix}_targets.csv", index=False, float_format="%.8g")

    for mode in MODES:
        seqs = res.sequences[mode]
        write_fasta(
            pd.DataFrame({"id": [f"gen_{i:04d}" for i in range(len(seqs))], "sequence": seqs}),
            out / f"{prefix}_generated_{mode}.fasta",
        )
    ids, seqs_t = [], []
    for t, seqs in res.target_sequences["sample"].items():
        ids += [f"gen_t{t:02d}_{i:04d}" for i in range(len(seqs))]
        seqs_t += seqs
    write_fasta(pd.DataFrame({"id": ids, "sequence": seqs_t}), out / f"{prefix}_generated_targets_sample.fasta")

    # Referencias sin condición de la evaluación por objetivos (la GAN de referencia, si está generada).
    references: dict = {"real_ceiling": real_ceiling(real_lengths, targets).to_dict(orient="list")}
    ref_fasta = PROCESSED / f"{REFERENCE_PREFIX}_generated_sample.fasta"
    if ref_fasta.exists():
        ref_seqs = read_fasta(ref_fasta)["sequence"].tolist()
        references[REFERENCE_PREFIX] = unconditional_target_table(ref_seqs, targets).to_dict(orient="list")

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
        "backend_mode": choice["mode"],
        "backend_decision": choice,
        "variant": args.variant,
        "config": asdict(config),
        "condition_spec": res.history.spec.to_dict(),
        "target_percentiles": {str(k): v for k, v in length_percentiles(real_lengths).items()},
        "target_lengths": list(targets),
        "usual_length_range_p10_p90": list(usual),
        "hit_tolerance": HIT_TOLERANCE,
        "epochs": args.epochs,
        "max_steps_per_epoch": args.max_steps,
        "snapshot_every": args.snapshot_every,
        "snapshot_epochs": res.snapshot_epochs,
        "n_eval": args.n_eval,
        "eval_seed": args.seed,
        "eval_conditions": "longitudes reales muestreadas con reposicion (default_rng(eval_seed)) de los datos de entrenamiento",
        "train_fasta": TRAIN_FASTA.name,
        "n_train": int(X.shape[0]),
        "timing_s": res.timing,
        "final_epoch_losses": {k: res.history[k][-1] for k in GAN_HISTORY_COLUMNS if k != "d_loss"},
        "unconditional_references": references,
    }
    (out / f"{prefix}_run_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    t = res.timing
    print(f"[{tag}] tiempo total {t['total_s']:.1f} s (entrenamiento {t['train_s']:.1f} s, "
          f"evaluación {t['eval_s']:.1f} s); por época: media {t['mean_epoch_s']:.2f} s, "
          f"mediana {t['median_epoch_s']:.2f} s", flush=True)  # fmt: skip
    print(f"[{tag}] archivos escritos en {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

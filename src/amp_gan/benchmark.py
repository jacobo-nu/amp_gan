"""Prueba de rendimiento de la GAN base por backend de Keras y dispositivo.

Para cada configuración candidata ``(backend, dispositivo)`` se lanza un
SUBPROCESO independiente (con ``sys.executable``, ``KERAS_BACKEND`` y
``CUDA_VISIBLE_DEVICES`` fijados en su entorno) que mide el paso de
entrenamiento de :mod:`amp_gan.gan` con el mismo modelo (configuración por
defecto), los mismos datos (Antibacterial limpio codificado) y la misma semilla,
para cada tamaño de lote. Por defecto solo se miden los candidatos
disponibles en esta máquina (:func:`amp_gan.runtime.available_candidates`: backends
instalados y GPU solo si se detecta); ``--all`` mide además los no disponibles
(quedan como filas de error, útil para diagnóstico).

Como CLI escribe un informe opcional en ``context/data/processed/benchmark_results.csv``.
Ese CSV es solo informativo y específico de la máquina donde se generó: NINGÚN
código ni notebook lo lee para decidir el backend. La decisión en tiempo de
ejecución la toma :func:`amp_gan.runtime.apply_auto_backend` (opcional) midiendo
en memoria.

Uso (desde la raíz del proyecto)::

    PYTHONPATH=src python -m amp_gan.benchmark
    PYTHONPATH=src python -m amp_gan.benchmark --steps 50 --batch-sizes 128

Desde Python, :func:`collect_benchmark` hace la misma medición y devuelve el
DataFrame en memoria (sin escribir el CSV; admite menos pasos medidos).

Regla de decisión (:func:`select_fastest`): menor ``est_epoch_s`` al tamaño de
lote de entrenamiento (128). Si la diferencia entre las dos primeras es menor
del 10 % se consideran empatadas y se prefiere la más simple/estable (orden de
:data:`SIMPLICITY_ORDER`: CPU antes que GPU y TensorFlow antes que torch).

Este módulo no importa keras a nivel de módulo; solo el modo hijo (``--child``)
lo importa, después de fijar el backend.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _SRC_DIR.parent
DEFAULT_FASTA = _PROJECT_ROOT / "context" / "data" / "processed" / "Antibacterial_amps_clean.fasta"
DEFAULT_CSV = _PROJECT_ROOT / "context" / "data" / "processed" / "benchmark_results.csv"

#: Todas las configuraciones posibles ``(backend, dispositivo)``. Por defecto se miden
#: solo las disponibles en la máquina (``runtime.available_candidates``).
CANDIDATES: tuple[tuple[str, str], ...] = (
    ("tensorflow", "cpu"),
    ("torch", "cpu"),
    ("torch", "gpu"),
    ("tensorflow", "gpu"),
)

#: Orden de preferencia (más simple/estable primero) para desempatar.
SIMPLICITY_ORDER: tuple[tuple[str, str], ...] = (
    ("tensorflow", "cpu"),
    ("torch", "cpu"),
    ("tensorflow", "gpu"),
    ("torch", "gpu"),
)

BATCH_SIZES: tuple[int, ...] = (64, 128, 256, 512)
TRAIN_BATCH_SIZE = 128
TIE_TOLERANCE = 0.10
SEED = 42
DEFAULT_WARMUP = 20
DEFAULT_STEPS = 200
CHILD_TIMEOUT_S = 1200

MARKER = "@@BENCH@@"

CSV_COLUMNS = [
    "backend", "device", "batch_size", "status", "error", "n_steps", "n_warmup",
    "ms_per_step_median", "ms_per_step_mean", "samples_per_s", "est_epoch_s",
    "n_samples", "gpu_mem_mib", "device_check", "keras_devices", "cuda_visible_devices",
    "seed", "keras_version", "tensorflow_version", "torch_version", "numpy_version",
    "python_version",
]  # fmt: skip


# --------------------------------------------------------------------------- #
# Regla de decisión (pandas puro, sin keras)
# --------------------------------------------------------------------------- #
def _simplicity_rank(backend: str, device: str) -> int:
    try:
        return SIMPLICITY_ORDER.index((backend, device))
    except ValueError:
        return len(SIMPLICITY_ORDER)


def select_fastest(
    df: pd.DataFrame, batch_size: int = TRAIN_BATCH_SIZE, tie_tol: float = TIE_TOLERANCE
) -> dict:
    """Elige la configuración más rápida para un tamaño de lote.

    Se ignoran las filas con ``status != "ok"`` o sin ``est_epoch_s``. Gana la
    menor ``est_epoch_s``; si la segunda está a menos de ``tie_tol`` (relativo
    al mejor) se declara empate y gana la más simple según
    :data:`SIMPLICITY_ORDER`.

    Returns:
        Diccionario con ``batch_size``, ``backend``, ``device``,
        ``est_epoch_s``, ``tied`` (bool), ``rel_diff`` (de la segunda respecto a
        la primera; ``None`` si solo hay una), ``fastest`` y ``runner_up`` (cada
        uno ``(backend, device)`` o ``None``) y ``message``.

    Raises:
        ValueError: si no hay ninguna fila válida para ese tamaño de lote.
    """
    ok = df[
        (df["status"] == "ok")
        & (df["batch_size"] == batch_size)
        & df["est_epoch_s"].notna()
    ].sort_values("est_epoch_s", kind="stable")
    if ok.empty:
        raise ValueError(f"No hay filas válidas (status=ok) para batch_size={batch_size}.")
    first = ok.iloc[0]
    second = ok.iloc[1] if len(ok) > 1 else None
    rel = None
    tied = False
    chosen = first
    if second is not None:
        rel = float((second["est_epoch_s"] - first["est_epoch_s"]) / first["est_epoch_s"])
        tied = rel < tie_tol
        if tied and _simplicity_rank(second["backend"], second["device"]) < _simplicity_rank(
            first["backend"], first["device"]
        ):
            chosen = second
    f = (str(first["backend"]), str(first["device"]))
    r = (str(second["backend"]), str(second["device"])) if second is not None else None
    c = (str(chosen["backend"]), str(chosen["device"]))
    if tied:
        msg = (
            f"Empate (diferencia {rel:.1%} < {tie_tol:.0%}) entre {f} y {r}; "
            f"se elige la más simple/estable: {c}."
        )
    elif r is None:
        msg = f"Única configuración válida: {c}."
    else:
        msg = (
            f"Gana {c} con {first['est_epoch_s']:.2f} s/época; la segunda, {r}, "
            f"es un {rel:.1%} más lenta."
        )
    return {
        "batch_size": int(batch_size),
        "backend": c[0],
        "device": c[1],
        "est_epoch_s": float(chosen["est_epoch_s"]),
        "tied": bool(tied),
        "rel_diff": rel,
        "fastest": f,
        "runner_up": r,
        "message": msg,
    }


def fastest_by_batch(df: pd.DataFrame, tie_tol: float = TIE_TOLERANCE) -> list[dict]:
    """Aplica :func:`select_fastest` a cada tamaño de lote con filas válidas."""
    out = []
    for bs in sorted(df.loc[df["status"] == "ok", "batch_size"].unique()):
        out.append(select_fastest(df, int(bs), tie_tol))
    return out


# --------------------------------------------------------------------------- #
# Proceso hijo
# --------------------------------------------------------------------------- #
def _device_kind(text: str) -> str:
    t = text.lower()
    return "gpu" if ("cuda" in t or "gpu" in t) else "cpu"


def _weights_device(model) -> str:
    """Dispositivo de los pesos (específico del backend; solo se usa en el benchmark)."""
    kinds = {_device_kind(str(getattr(w.value, "device", "cpu"))) for w in model.weights}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def weights_device(model) -> str:
    """Dispositivo (``"cpu"``, ``"gpu"`` o ``"mixed"``) de los pesos de un modelo Keras.

    Se apoya en el atributo ``device`` de los tensores del backend, así que es
    específico de backend y solo debe usarse para diagnóstico (no en ``gan.py``).
    """
    return _weights_device(model)


def _gpu_mem_mib() -> float | None:
    """Memoria de GPU usada por este proceso según ``nvidia-smi`` (MiB) o ``None``."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        )  # fmt: skip
    except Exception:
        return None
    for line in res.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(os.getpid()):
            return float(parts[1])
    return None


def _emit(payload: dict) -> None:
    print(MARKER + json.dumps(payload), flush=True)


def _child_main(args: argparse.Namespace) -> int:
    from amp_gan import runtime

    runtime.set_backend(args.backend)
    if args.device == "cpu":
        runtime.use_cpu_only()

    import keras

    from amp_gan.encoding import load_encoded
    from amp_gan.gan import GANConfig, build_models, train_step

    env = runtime.describe_environment()
    _emit({"env": env})

    errors = []
    if env["keras_backend"] != args.backend:
        errors.append(f"backend pedido {args.backend}, usado {env['keras_backend']}")
    if args.device == "cpu" and (env["gpu_visible"] or env["cuda_visible_devices"] != ""):
        errors.append(f"CPU pedida pero GPU visible o CUDA_VISIBLE_DEVICES no vacío: {env}")
    if args.device == "gpu" and not env["gpu_visible"]:
        errors.append(f"GPU pedida pero Keras solo ve {env['keras_devices']}")
    if errors:
        _emit({"fatal": "; ".join(errors)})
        return 0

    X, _ = load_encoded(args.fasta)
    n_samples = int(X.shape[0])
    for bs in args.batch_sizes:
        row = {"batch_size": bs}
        try:
            cfg = GANConfig(batch_size=bs, seed=SEED)
            keras.utils.set_random_seed(SEED)
            G, D, gan = build_models(cfg)
            actual = {"G": _weights_device(G), "D": _weights_device(D)}
            if set(actual.values()) != {args.device}:
                raise RuntimeError(f"dispositivo pedido {args.device}, pesos en {actual}")
            rng = np.random.default_rng(SEED)
            total = args.warmup + args.steps
            batches = [X[rng.integers(0, n_samples, bs)] for _ in range(total)]
            times = []
            for i, batch in enumerate(batches):
                t0 = time.perf_counter()
                out = train_step(G, D, gan, batch, cfg)
                dt = time.perf_counter() - t0
                if not all(np.isfinite(v) for v in out.values()):
                    raise RuntimeError(f"pérdida no finita en el paso {i}: {out}")
                if i >= args.warmup:
                    times.append(dt * 1000.0)
            med, mean = float(np.median(times)), float(np.mean(times))
            sps = bs / (med / 1000.0)
            row.update(
                status="ok", error="", n_steps=len(times), n_warmup=args.warmup,
                ms_per_step_median=med, ms_per_step_mean=mean, samples_per_s=sps,
                est_epoch_s=n_samples / sps, n_samples=n_samples,
                gpu_mem_mib=_gpu_mem_mib() if args.device == "gpu" else None,
                device_check=f"G={actual['G']},D={actual['D']}",
            )  # fmt: skip
        except Exception as exc:  # una lote fallido no aborta el resto
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        _emit({"row": row})
    return 0


# --------------------------------------------------------------------------- #
# Proceso padre
# --------------------------------------------------------------------------- #
def _child_env(backend: str, device: str) -> dict[str, str]:
    env = dict(os.environ)
    env["KERAS_BACKEND"] = backend
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    prev = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_SRC_DIR) + (os.pathsep + prev if prev else "")
    return env


def _run_candidate(backend, device, batch_sizes, steps, warmup, fasta) -> list[dict]:
    """Lanza el hijo y devuelve una fila (dict) por tamaño de lote."""
    cmd = [
        sys.executable, "-m", "amp_gan.benchmark", "--child",
        "--backend", backend, "--device", device,
        "--batch-sizes", ",".join(map(str, batch_sizes)),
        "--steps", str(steps), "--warmup", str(warmup), "--fasta", str(fasta),
    ]  # fmt: skip
    base = {"backend": backend, "device": device, "seed": SEED}
    try:
        proc = subprocess.run(
            cmd, env=_child_env(backend, device), capture_output=True, text=True,
            timeout=CHILD_TIMEOUT_S, cwd=_PROJECT_ROOT,
        )  # fmt: skip
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr, code = "timeout", -1

    env_info, rows, fatal = {}, {}, None
    for line in stdout.splitlines():
        if not line.startswith(MARKER):
            continue
        payload = json.loads(line[len(MARKER):])
        env_info = payload.get("env", env_info)
        fatal = payload.get("fatal", fatal)
        if "row" in payload:
            rows[payload["row"]["batch_size"]] = payload["row"]

    hint = ""
    missing = [ln.split("'")[1] for ln in stderr.splitlines() if "Could not load dynamic library" in ln]
    if missing:
        hint = f" [bibliotecas no encontradas: {', '.join(missing)}]"
    tail = " | ".join(ln.strip() for ln in stderr.strip().splitlines()[-3:])

    common = {
        "keras_devices": ";".join(env_info.get("keras_devices", [])),
        "cuda_visible_devices": env_info.get("cuda_visible_devices"),
        "keras_version": env_info.get("keras_version"),
        "tensorflow_version": env_info.get("tensorflow_version"),
        "torch_version": env_info.get("torch_version"),
        "numpy_version": env_info.get("numpy_version"),
        "python_version": env_info.get("python_version"),
    }
    out = []
    for bs in batch_sizes:
        row = dict(base, **common, batch_size=bs)
        if bs in rows:
            row.update(rows[bs])
        elif fatal:
            row.update(status="error", error=fatal + hint)
        else:
            row.update(status="error", error=f"el hijo terminó (código {code}) sin resultado: {tail}")
        if row.get("status") == "error" and hint and hint not in row["error"]:
            row["error"] += hint
        out.append(row)
    return out


def collect_benchmark(
    candidates=None,
    batch_sizes=BATCH_SIZES,
    steps: int = DEFAULT_STEPS,
    warmup: int = DEFAULT_WARMUP,
    fasta: Path = DEFAULT_FASTA,
    verbose: bool = False,
) -> pd.DataFrame:
    """Ejecuta todas las configuraciones y devuelve el DataFrame EN MEMORIA.

    No escribe nada en disco. Tiene el mismo formato (columnas :data:`CSV_COLUMNS`)
    que el CSV de :func:`run_benchmark`. Para una prueba rápida (por ejemplo desde
    un notebook) se puede reducir ``steps`` (pasos medidos) y ``warmup``.

    Args:
        candidates: configuraciones ``(backend, dispositivo)``; por defecto las
            disponibles en esta máquina (:func:`amp_gan.runtime.available_candidates`).
        batch_sizes: tamaños de lote a medir.
        steps: nº de pasos medidos por configuración y lote.
        warmup: pasos de calentamiento (no medidos).
        fasta: FASTA limpio con los datos de la medición.
        verbose: si es ``True`` imprime el progreso.
    """
    if candidates is None:
        from amp_gan import runtime

        candidates = runtime.available_candidates()
    rows = []
    for backend, device in candidates:
        if verbose:
            print(f"[benchmark] {backend}/{device} ...", flush=True)
        t0 = time.perf_counter()
        rows.extend(_run_candidate(backend, device, batch_sizes, steps, warmup, fasta))
        if verbose:
            print(f"[benchmark]   hecho en {time.perf_counter() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows).reindex(columns=CSV_COLUMNS)
    for col in ("n_steps", "n_warmup", "n_samples"):
        df[col] = df[col].astype("Int64")
    return df


def run_benchmark(
    candidates=None,
    batch_sizes=BATCH_SIZES,
    steps: int = DEFAULT_STEPS,
    warmup: int = DEFAULT_WARMUP,
    fasta: Path = DEFAULT_FASTA,
    csv_path: Path = DEFAULT_CSV,
) -> pd.DataFrame:
    """Ejecuta las configuraciones, escribe el CSV (informe opcional) y devuelve el DataFrame."""
    df = collect_benchmark(candidates, batch_sizes, steps, warmup, fasta, verbose=True)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def format_markdown(df: pd.DataFrame) -> str:
    """Tabla en markdown con las columnas principales."""
    cols = ["backend", "device", "batch_size", "status", "n_steps", "ms_per_step_median",
            "ms_per_step_mean", "samples_per_s", "est_epoch_s", "gpu_mem_mib", "error"]  # fmt: skip
    view = df[cols].copy()
    for c in ("ms_per_step_median", "ms_per_step_mean", "samples_per_s", "est_epoch_s"):
        view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.1f}" if c != "est_epoch_s" else f"{v:.2f}")
    view = view.astype(object).where(view.notna(), "").astype(str)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(r) + " |" for r in view.values.tolist()]
    return "\n".join(lines)


def report(df: pd.DataFrame) -> None:
    """Imprime la tabla, la más rápida por lote y la ganadora a batch_size=128."""
    print(format_markdown(df))
    print("\nMás rápida por tamaño de lote:")
    for d in fastest_by_batch(df):
        print(f"  batch {d['batch_size']:>4}: {d['message']}")
    try:
        d = select_fastest(df, TRAIN_BATCH_SIZE)
        print(f"\nGanadora para batch_size={TRAIN_BATCH_SIZE}: {d['backend']}/{d['device']} "
              f"({d['est_epoch_s']:.2f} s/época). {d['message']}")  # fmt: skip
    except ValueError as exc:
        print(f"\nSin ganadora para batch_size={TRAIN_BATCH_SIZE}: {exc}")


def _parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--child", action="store_true", help="modo interno (subproceso)")
    p.add_argument("--backend", default="tensorflow")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--batch-sizes", default=",".join(map(str, BATCH_SIZES)))
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    p.add_argument("--out", type=Path, default=DEFAULT_CSV)
    p.add_argument("--all", action="store_true",
                   help="medir también los candidatos no disponibles (filas de error)")  # fmt: skip
    args = p.parse_args(argv)
    args.batch_sizes = [int(b) for b in str(args.batch_sizes).split(",") if b]
    return args


def main(argv=None) -> int:
    args = _parse(argv)
    if args.child:
        return _child_main(args)
    df = run_benchmark(
        candidates=CANDIDATES if args.all else None, batch_sizes=args.batch_sizes,
        steps=args.steps, warmup=args.warmup, fasta=args.fasta, csv_path=args.out,
    )  # fmt: skip
    report(df)
    print(f"\nCSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

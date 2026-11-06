"""Evaluación EN VIVO de los cinco roles acotados (no es un test de pytest).

Uso (desde la carpeta ``src``, con Ollama en marcha y los modelos descargados)::

    python -m amp_agents.eval_roles locator       --workdir /ruta/a/carpeta/vacia   # RETIRADO: solo para reproducir el resultado
    python -m amp_agents.eval_roles doc_checker   --workdir ...
    python -m amp_agents.eval_roles doc_checker_ss --workdir ...   # variante de una pasada de doc_checker (mismo protocolo y criterios)
    python -m amp_agents.eval_roles doc_checker_pairs --model gemma3:4b --reps 5 --workdir ...   # verificador acotado: el código da pares de líneas, el modelo juzga sí/no
    python -m amp_agents.eval_roles code_reviewer --workdir ...
    python -m amp_agents.eval_roles code_reviewer_funcs --model qwen2.5-coder:7b --reps 5 --max-minutes 38 --workdir ...   # revisor acotado: una pregunta por función y tipo de defecto
    python -m amp_agents.eval_roles summarizer    --workdir ...
    python -m amp_agents.eval_roles retriever     --workdir ...
    python -m amp_agents.eval_roles retriever --model bge-m3 --workdir ...            # embeddings de otro modelo (sin prefijos: ver amp_agents.retrieval)
    python -m amp_agents.eval_roles retriever_hybrid --model bge-m3 --workdir ...     # BM25 + embeddings por rango recíproco (RRF, k = 60)
    python -m amp_agents.eval_roles all           --workdir ... --deadline-min 90   # sin el localizador; añádelo con --roles locator,...
    python -m amp_agents.eval_roles report        --results-dir ...
    python -m amp_agents.eval_roles doc_checker_ss --model gemma3:4b --num-ctx 12288 --reps 5 --tag g3 --workdir ...   # otro modelo, sin tocar agents.toml

Con ``--model NOMBRE`` (y opcionalmente ``--num-ctx N``) se evalúa OTRO modelo en un rol existente con el mismo protocolo y
los mismos criterios, sin editar ``agents.toml``: los resultados y las tablas llevan el modelo en el nombre del rol
(``doc_checker_ss@gemma3-4b``) y no pisan los de otros modelos. Funciona con todos los roles (también el recuperador, con un
modelo de embeddings: ``retriever --model bge-m3``). Antes de empezar se comprueba en ``/api/tags`` que el modelo está
instalado (no se descarga nada) y se hace una petición de calentamiento, con su propio tope de tiempo, que se registra
aparte (``warmup`` en el JSON) y no cuenta como repetición. ``--model`` no se admite con ``all``. ``--max-tokens N`` y ``--max-seconds S`` fijan el tope de tokens (razonamiento incluido) y
el de tiempo de CADA petición del rol evaluado (hacen falta para los modelos de razonamiento).

Todo se hace sobre COPIAS dentro de ``--workdir`` (``context/`` sin ``context/data/`` y ``src/``, sin este script ni
sus tests, para que las preguntas y sus respuestas no estén en el material consultado): las herramientas y la política
apuntan a la copia, nunca al repositorio real. Cada ejecución comprueba que la copia no cambió. Los resultados
(JSON, registros JSONL y tabla Markdown) van a ``--results-dir`` (por defecto ``<workdir>/results``). Por defecto
``--workdir`` es una carpeta temporal del sistema.

PROTOCOLO Y CRITERIOS (fijados de antemano; no se ajustan tras ver los resultados):

- locator (``llama3.2:3b``): 10 preguntas con verdad calculada por código (5 definiciones con ``ast``, 3 importaciones
  y 2 constantes con expresiones regulares), 2 repeticiones. Éxito de una respuesta: el conjunto de pares
  (ruta, línea) es EXACTAMENTE el esperado y cada línea citada existe. Criterio: >= 80 % de respuestas exactas.
- doc_checker (``phi4-mini``): copias CONGELADAS de ``context/constitution/roadmap.md`` y ``TODO.md`` (esta como
  ``context/TODO.md``, porque el rol solo lee ``context/`` y ``src/``) tal como estaban en la evaluación original
  (``amp_agents/fixtures/``, para que los documentos vivos del repositorio puedan cambiar sin alterar el caso evaluado)
  con 6 incoherencias sembradas de tipos
  distintos, más una copia limpia, 3 repeticiones. Una incoherencia sembrada se recupera si alguna de las dos citas
  de un hallazgo es literal y cae (como mucho 2 líneas) sobre la línea alterada. Criterios: >= 90 % de citas
  literales válidas (todas las ejecuciones); >= 4 de las 6 sembradas recuperadas de media por repetición; <= 2
  falsos positivos (hallazgos en la copia limpia) en CADA repetición.
- doc_checker_ss (``phi4-mini``, modo ``single_shot``): el mismo protocolo, los mismos documentos congelados y LOS MISMOS
  criterios (se leen de la entrada ``doc_checker`` de ``CRITERIA``); solo cambia que no hay herramientas y los dos
  documentos, con numeración de línea, van en el mensaje. Sus resultados se guardan con su propio nombre de rol.
- doc_checker_pairs (``gemma3:4b`` por defecto, modo ``single_shot``): el mismo protocolo, los mismos documentos y LOS MISMOS
  criterios (entrada ``doc_checker`` de ``CRITERIA``), pero el modelo solo juzga pares de líneas que genera el código
  (``amp_agents.pairs``, diseño fijado en su docstring) y las citas de cada hallazgo las pone el código (literales por
  construcción: el criterio de citas válidas es trivial y no mide al modelo). Además de los tres criterios se informa, en
  ``pairs_<rol>.json`` y en las notas, de la cobertura del generador sobre las semillas, de los fallos de formato y del
  tiempo por par. ``--max-minutes`` limita el tiempo (ver ``amp_agents.pairs``).
- code_reviewer (``qwen2.5-coder:7b``): copia de ``src/amp_gan/properties.py`` con 5 defectos sembrados de tipos
  distintos, más la copia limpia, 3 repeticiones. Un hallazgo cuenta si su línea está a +-2 de la sembrada.
  Criterios: >= 3 de 5 recuperados de media; >= 90 % de citas válidas (existe la línea y la cita es literal);
  <= 2 falsos positivos en CADA repetición de la copia limpia.
- summarizer (``gemma3:4b``): 2 entradas (métricas y ``run_info`` de ``cgan_hydro`` y de ``cgan_length``), 3
  repeticiones. Criterios: >= 95 % de las cifras del resumen presentes en la entrada (agregado); <= 150 palabras y
  en castellano (proporción de palabras funcionales castellanas >= 0,8, ver ``verifiers``) en TODAS las ejecuciones.
- retriever (``nomic-embed-text``): 15 consultas en castellano con verdad escrita a mano (fichero y palabra clave en
  el fragmento). Criterio: acierto en el top-3 >= 80 %. Se mide igual la línea base léxica (BM25).
- retriever_hybrid (``nomic-embed-text`` por defecto, o ``--model``): el mismo protocolo y el mismo criterio (entrada ``retriever`` de
  ``CRITERIA``) con la fusión BM25 + embeddings por rango recíproco (diseño fijado en el docstring de ``amp_agents.retrieval``);
  se informan además BM25, los embeddings solos y las consultas que cambian de acierto a fallo y viceversa respecto de BM25.
- code_reviewer_funcs (``qwen2.5-coder:7b`` por defecto, modo ``single_shot``): el mismo fichero, las mismas semillas, la misma copia
  sembrada y limpia y LOS MISMOS criterios (entrada ``code_reviewer`` de ``CRITERIA``), pero el código divide el fichero en funciones
  y hace UNA pregunta por función y tipo de defecto con salida restringida; las citas las pone el código (literales por
  construcción: el criterio de citas válidas no mide al modelo). Diseño fijado en el docstring de ``amp_agents.func_review``.

Segunda iteración (los criterios de arriba NO cambian): el localizador con modelo queda retirado (10 % de aciertos; lo
sustituye ``amp_agents.locate``) y ``all`` no lo incluye por defecto. Para el revisor y el resumidor, los criterios se
siguen juzgando sobre la salida CRUDA del modelo, y se informa además, en filas marcadas «[... por código]» con el mismo
umbral, de la salida corregida por código por encima: línea del revisor reasignada a la de su cita literal
(``verifiers.correct_review_lines``) y resumen con coma decimal y recortado a 150 palabras (``verifiers.postprocess_summary``).

- query_rewriter (``llama3.2:3b`` por defecto, ``single_shot``): las mismas 15 consultas del recuperador; un modelo reescribe cada pregunta
  en palabras clave y se busca con la pregunta + palabras (BM25). Criterio NUEVO (``CRITERIA["query_rewriter"]``): top-3 con reescritura >=
  top-3 de BM25 simple (medido en la misma ejecución) + 2 y como mucho 1 consulta que pasa de acierto a fallo. Se registra el hash del corpus
  al empezar y al terminar. Diseño en ``amp_agents.rewrite``.
- router (``llama3.2:3b`` por defecto, ``single_shot``): 48 peticiones etiquetadas (``fixtures/router_requests.json``, 8 por rol); el modelo elige
  el rol que la atiende. Criterios NUEVOS (``CRITERIA["router"]``): exactitud >= 85 % y recall >= 60 % en cada rol. Diseño en ``amp_agents.router``.
- summarizer: además de las filas crudas y «[corregido por código]», con un paso determinista de FILTRO tras el recorte (quita las frases con
  cifras no respaldadas por la entrada; ``verifiers.filter_unsupported_sentences``) se informan filas «[recortado + filtro]» con los mismos
  umbrales de ``CRITERIA["summarizer"]``. Un resumen filtrado de menos de 30 palabras (``FILTER_MIN_WORDS``) o vacío es un FALLO DE FORMATO y
  cuenta como no cumplido: esas filas solo se cumplen si ninguna ejecución es fallo de formato. Al ser código, el cumplimiento depende del
  filtro y no del modelo.

Las ejecuciones fallidas (JSON inválido, tiempo agotado, error del servidor) cuentan como no exitosas y se anotan.
Por ejecución se registra: éxito, pasos, tokens, segundos, motivo de parada, contexto efectivo (``/api/ps``) y si el
modelo se pasó parcialmente a la CPU.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agents import PROJECT_ROOT, ConfigError, get_agent_config, make_provider
from .locate import find_definitions, find_lines  # noqa: F401 (la verdad del localizador; se reexportan)
from .func_review import CATEGORIES, CATEGORY_BY_KEY, build_findings as build_func_findings, match_by_category, request_stats, review_units, seed_flags, split_functions, yes_rates
from .loop import AgentResult
from .rewrite import rewrite_query
from .router import NO_ROLE, Route, ROUTER_ROLES, confusion_matrix, load_requests, matrix_table, role_metrics, route_request, route_request_decision
from .pairs import COMPAT_FORMAT_SCHEMA, SENSITIVITY_THRESHOLDS, build_findings, build_pairs, judge_pairs, judgment_stats, pair_mode, pairs_to_json, seed_coverage
from .providers import ProviderError, ProviderTimeout, SystemOneProvider, Usage
from .retrieval import LexicalRetriever, build_index, chunk_project, prefixes_for, resolve_index_dir, save_index
from .run import prepare_task, retrieve, run_task
from .smoke_live import make_copy as _smoke_copy
from .smoke_live import server_models, snapshot
from .verifiers import (
    check_doc_citations,
    check_review_citations,
    correct_review_lines,
    count_words,
    parse_json_output,
    FILTER_MIN_WORDS,
    postprocess_summary,
    retrieval_rank,
    spanish_score,
    verifiable_findings,
    verify,
)

# ``ROLES`` son los que se evalúan por defecto con ``all``. El localizador con modelo (``llama3.2:3b``) está RETIRADO:
# 10 % de aciertos y ``grep`` (``amp_agents.locate``) ya lo hace; su evaluación se conserva para reproducir el resultado
# negativo y se pide expresamente (``locator`` o ``all --roles locator``).
ROLES = ("doc_checker", "code_reviewer", "summarizer", "retriever")
RETIRED_ROLES = ("locator",)
ALL_ROLES = ("locator", *ROLES)
# Variantes de un rol: se evalúan con el protocolo y los criterios de su rol base, pero con su propio nombre en los
# ficheros de resultados. No entran en ``all`` (se piden expresamente, para comparar con el rol base).
VARIANT_ROLES = {
    "doc_checker_ss": "doc_checker",
    "doc_checker_pairs": "doc_checker",
    "doc_checker_pairs_short": "doc_checker",
    "doc_checker_pairs_json": "doc_checker",
    "doc_checker_pairs_agree": "doc_checker",
    "code_reviewer_funcs": "code_reviewer",
    "retriever_hybrid": "retriever",
}
# Roles con protocolo propio (criterios nuevos en ``CRITERIA``), de una sola repetición (temperatura 0); no entran en ``all``.
EXTRA_ROLES = ("query_rewriter", "router")
COMMAND_ROLES = (*ALL_ROLES, *VARIANT_ROLES, *EXTRA_ROLES)
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DEFAULT_WORKDIR = Path(tempfile.gettempdir()) / "amp_roles_eval"
EXCLUDED_FROM_COPY = ("src/amp_agents/eval_roles.py",)  # y src/tests/test_agents_eval*.py
EXCLUDED_DIRS_FROM_COPY = ("src/amp_agents/fixtures",)  # los documentos congelados no deben estar en el material consultado

CRITERIA = {
    "locator": {"exact_rate_min": 0.80},
    "doc_checker": {"valid_citations_min": 0.90, "recovered_mean_min": 4.0, "false_positives_max": 2},
    "code_reviewer": {"recovered_mean_min": 3.0, "valid_citations_min": 0.90, "false_positives_max": 2},
    "summarizer": {"numbers_min": 0.95, "max_words": 150},
    "retriever": {"top3_min": 0.80},
    # criterios NUEVOS fijados el 2026-11-06 antes de medir (ver ``amp_agents.rewrite`` y ``amp_agents.router``)
    "query_rewriter": {"gain_min": 2, "regressions_max": 1},
    "router": {"accuracy_min": 0.85, "recall_min": 0.60},
}
REPS = {"locator": 2, "doc_checker": 3, "code_reviewer": 3, "summarizer": 3}


def base_role(role: str) -> str:
    """Rol cuyo protocolo y criterios (``CRITERIA``, ``REPS``) usa ``role``: él mismo, o su rol base si es una variante."""
    return VARIANT_ROLES.get(role, role)


# -- prueba de otro modelo en un rol existente (``--model``, ``--num-ctx``) -----------------------------------------
@dataclass(frozen=True)
class Trial:
    """Cómo se evalúa un rol: nombre que sale en tablas y ficheros y cambios de configuración (sin tocar ``agents.toml``)."""

    label: str
    overrides: dict = field(default_factory=dict)
    model: str | None = None  # solo con --model: activa la comprobación del modelo y el calentamiento


def safe_model_name(model: str) -> str:
    """Nombre del modelo apto para ficheros y tablas: ``gemma3:4b`` -> ``gemma3-4b``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


def trial_for(role: str, args, config_path=None) -> Trial:
    """Rol evaluado con ``--model`` / ``--num-ctx`` (si se dieron).

    Sin ninguno de los dos, el nombre es el del rol y no cambia nada. Con ``--model`` el nombre es ``<rol>@<modelo>``
    (``doc_checker_ss@gemma3-4b``), para que los resultados no pisen los de otros modelos. ``--num-ctx`` fija
    ``options.num_ctx`` conservando el resto de opciones del rol; solo (sin ``--model``) añade ``@ctx<N>`` al nombre.
    ``--max-tokens`` y ``--max-seconds`` cambian los topes de cada petición (``max_tokens``; ``max_seconds`` y ``timeout``) sin
    cambiar el nombre. ``--think true|false`` fija ``options.think`` (razonamiento de los modelos que lo admiten) y
    ``--provider`` cambia el proveedor (``ollama_systemone`` para un modelo de decisión); tampoco cambian el nombre.
    """
    model = getattr(args, "model", None) or None
    num_ctx = getattr(args, "num_ctx", None) or None
    max_tokens = getattr(args, "max_tokens", None) or None
    max_seconds = getattr(args, "max_seconds", None) or None
    provider = getattr(args, "provider", None) or None
    think = getattr(args, "think", None)  # None (sin tocar), True o False
    overrides: dict = {}
    label = role
    if provider:
        overrides["provider"] = provider
    if max_tokens:
        overrides["max_tokens"] = max_tokens
    if max_seconds:  # tope de tiempo de cada petición: el del bucle (max_seconds) y el del proveedor (timeout)
        overrides["max_seconds"] = float(max_seconds)
        overrides["timeout"] = float(max_seconds)
    if model:
        overrides["model"] = model
        label = f"{role}@{safe_model_name(model)}"
    if num_ctx or think is not None:
        options = dict(get_agent_config(role, config_path).options)
        if num_ctx:
            if num_ctx < 256:
                raise SystemExit("error: --num-ctx debe ser al menos 256")
            options["num_ctx"] = num_ctx
            if not model:
                label = f"{role}@ctx{num_ctx}"
        if think is not None:
            options["think"] = bool(think)  # petición nativa de Ollama: ``think`` va en el primer nivel del cuerpo
        overrides["options"] = options
    return Trial(label=label, overrides=overrides, model=model)


def check_model_available(base_url: str, model: str) -> None:
    """Sale con un mensaje claro si Ollama no responde o no tiene ``model`` (consulta ``/api/tags``)."""
    available = server_models(base_url)
    if available is None:
        raise SystemExit(f"error: el servidor no responde en {base_url} (¿está Ollama en marcha?)")
    if model not in available and f"{model}:latest" not in available:
        raise SystemExit(f"error: el modelo {model!r} no está instalado en Ollama (instalados: {', '.join(sorted(available)) or 'ninguno'}); instálalo antes con `ollama pull` (esta herramienta no descarga modelos)")


def warm_up(config, provider=None) -> dict:
    """Primera petición, mínima y con el mismo ``num_ctx``, de un modelo recién cargado: se mide aparte.

    La primera petición tarda más (carga de pesos y reserva del contexto) y no es una repetición del protocolo. Tiene su
    límite de tiempo (``config.timeout``). Devuelve ``{"seconds", "ok", "error", "ps"}``; no lanza excepciones del proveedor.
    """
    provider = provider or make_provider(config)
    started = time.monotonic()
    ok, error = True, ""
    try:
        if config.mode == "embedding":
            provider.embed(["calentamiento"], timeout=config.timeout)
        elif config.provider == "ollama_systemone":  # modelo de decisión: no admite chat
            provider.decide("Línea A: uno.\nLínea B: dos.", {"calentamiento": "¿Son iguales?"}, timeout=config.timeout)
        else:
            provider.chat([{"role": "user", "content": "Responde solo con la palabra: listo"}], timeout=config.timeout)
    except ProviderError as exc:  # incluye ProviderTimeout
        ok, error = False, f"{type(exc).__name__}: {exc}"
    return {"seconds": round(time.monotonic() - started, 1), "ok": ok, "error": error, "ps": ollama_ps(config.base_url, config.model)}


# -- copias ------------------------------------------------------------------------------------------------
def make_copy(destination: Path, source: Path = PROJECT_ROOT) -> Path:
    """Copia ``src/`` y ``context/`` (sin ``context/data/``) y quita de la copia este script y sus tests."""
    _smoke_copy(destination, source)
    for rel in EXCLUDED_FROM_COPY:
        target = destination / rel
        if target.exists():
            target.unlink()
    for rel in EXCLUDED_DIRS_FROM_COPY:
        shutil.rmtree(destination / rel, ignore_errors=True)
    for test in (destination / "src" / "tests").glob("test_agents_eval*.py"):
        test.unlink()
    return destination


def _ensure_outside_repo(path: Path) -> None:
    resolved = path.resolve()
    if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
        raise SystemExit("error: --workdir y --results-dir deben estar fuera del repositorio")


# -- información de Ollama -----------------------------------------------------------------------------------
def processor_label(size: int, size_vram: int) -> str:
    """Texto como el de ``ollama ps``: ``100% GPU``, ``100% CPU`` o ``27%/73% CPU/GPU``."""
    if size <= 0:
        return "desconocido"
    gpu = round(100.0 * size_vram / size)
    if gpu >= 100:
        return "100% GPU"
    if gpu <= 0:
        return "100% CPU"
    return f"{100 - gpu}%/{gpu}% CPU/GPU"


def ollama_ps(base_url: str, model: str, timeout: float = 10.0) -> dict:
    """Estado del modelo cargado según ``/api/ps``: procesador, contexto efectivo y memoria (``{}`` si no responde)."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/api/ps", timeout=timeout) as response:
            models = json.load(response).get("models", [])
    except (urllib.error.URLError, OSError, ValueError):
        return {}
    for item in models:
        if model in (item.get("name"), item.get("model"), str(item.get("name", "")).removesuffix(":latest"), str(item.get("model", "")).removesuffix(":latest")):
            size, vram = int(item.get("size") or 0), int(item.get("size_vram") or 0)
            return {
                "processor": processor_label(size, vram),
                "cpu_offload": size_cpu_fraction(size, vram) > 0.005,
                "context_length": item.get("context_length"),
                "size_mb": round(size / 1e6),
                "size_vram_mb": round(vram / 1e6),
            }
    return {}


def size_cpu_fraction(size: int, size_vram: int) -> float:
    return 0.0 if size <= 0 else max(0.0, min(1.0, 1.0 - size_vram / size))


def ollama_stop(model: str) -> str:
    """``ollama stop <modelo>`` para liberar memoria; devuelve la salida o el error."""
    try:
        done = subprocess.run(["ollama", "stop", model], capture_output=True, text=True, timeout=60)
        return (done.stdout + done.stderr).strip() or "ok"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"error: {exc}"


def ollama_ps_text() -> str:
    try:
        return subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"error: {exc}"


# -- registro de ejecuciones ---------------------------------------------------------------------------------
@dataclass
class RunRecord:
    role: str
    model: str
    case: str
    rep: int
    success: bool
    stop_reason: str
    steps: int
    tokens: int | None
    prompt_tokens_last: int | None
    seconds: float
    json_valid: bool | None = None
    json_strict: bool | None = None
    context_length: int | None = None
    num_ctx_config: int | None = None
    processor: str = ""
    cpu_offload: bool | None = None
    copy_modified: bool = False
    env_variant: int = 1
    notes: str = ""
    error: str = ""
    log: str = ""
    details: dict = field(default_factory=dict)


def _system_record(role: str, case: str, rep: int, result: AgentResult, root: Path, before: dict, env_variant: int, success: bool, notes: str, details: dict, verification=None, trial: Trial | None = None) -> RunRecord:
    trial = trial or Trial(role)
    config = get_agent_config(role, env_variant=env_variant, **trial.overrides)
    ps = ollama_ps(config.base_url, config.model)
    modified = snapshot(root) != before
    return RunRecord(
        role=trial.label,
        model=config.model,
        case=case,
        rep=rep,
        success=bool(success and not modified),
        stop_reason=result.stop_reason,
        steps=result.steps,
        tokens=None if result.tokens is None else result.tokens.total_tokens,
        prompt_tokens_last=result.last_prompt_tokens,
        seconds=round(result.duration_s, 1),
        json_valid=None if verification is None else verification.details.get("parsed"),
        json_strict=None if verification is None else verification.details.get("strict_json"),
        context_length=ps.get("context_length"),
        num_ctx_config=config.options.get("num_ctx"),
        processor=ps.get("processor", ""),
        cpu_offload=ps.get("cpu_offload"),
        copy_modified=modified,
        env_variant=env_variant,
        notes=notes + (" COPIA MODIFICADA" if modified else ""),
        error=result.error or "",
        log=str(result.log_path or ""),
        details={**details, "report": result.report[:6000], "tools_called": result.tools_called, "denied": len(result.denied), "ps": ps},
    )


class Recorder:
    """Guarda cada ejecución en cuanto termina (``<nombre>.runs.jsonl``) para no perder datos si se interrumpe."""

    def __init__(self, results_dir: Path, name: str):
        self.path = results_dir / f"{name}.runs.jsonl"
        self.runs: list[RunRecord] = []
        results_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def add(self, record: RunRecord) -> None:
        self.runs.append(record)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        flag = "OK " if record.success else "NO "
        print(f"    {flag}{record.case} rep{record.rep} [{record.stop_reason}] pasos={record.steps} {record.seconds}s {record.processor} | {record.notes[:150]}", file=sys.stderr, flush=True)


# -- locator -----------------------------------------------------------------------------------------------------
def locator_questions(root: Path) -> list[dict]:
    """Las 10 preguntas del localizador con la verdad calculada sobre la copia ``root``."""
    questions = []
    for name, kind, word in [
        ("mae_vs_reference", "function", "def"),
        ("Policy", "class", "class"),
        ("run_agent", "function", "def"),
        ("fit_condition_spec", "function", "def"),
        ("evaluate_target", "function", "def"),
    ]:
        what = "la clase" if kind == "class" else "la función"
        questions.append(
            {
                "id": f"def_{name}",
                "kind": "definicion",
                "question": f"¿Dónde se define {what} `{name}`? Indica la ruta y el número de línea de su línea `{word}`.",
                "expected": find_definitions(root, name),
            }
        )
    for module in ("conditions", "wgan_gp", "benchmark"):
        pattern = rf"^\s*(?:from\s+amp_gan\.{module}\s+import\b|import\s+amp_gan\.{module}\b)"
        questions.append(
            {
                "id": f"import_{module}",
                "kind": "importacion",
                "question": f"¿Qué ficheros `.py` de src/ importan el módulo `amp_gan.{module}` (con una línea `from amp_gan.{module} import ...`)? Indica la ruta y la línea de cada importación.",
                "expected": find_lines(root, pattern),
            }
        )
    for constant in ("HYDRO_HIT_TOLERANCE", "PKA"):
        questions.append(
            {
                "id": f"const_{constant}",
                "kind": "constante",
                "question": f"¿En qué líneas de los ficheros `.py` de src/ aparece la constante `{constant}`? Indica la ruta y el número de línea de cada aparición.",
                "expected": find_lines(root, rf"\b{constant}\b"),
            }
        )
    return questions


def _norm_path(ruta: str) -> str:
    ruta = ruta.strip()
    while ruta.startswith("./"):
        ruta = ruta[2:]
    return ruta


def grade_locator(report: str, expected: list[tuple[str, int]], root: Path) -> tuple[bool, dict]:
    """Corrige una respuesta del localizador: conjunto exacto de (ruta, línea) y líneas existentes."""
    verification = verify("locator", report, root=root)
    if not verification.valid:
        return False, {"errors": verification.errors, "strict_json": verification.details.get("strict_json")}
    obj, strict, _ = parse_json_output(report)
    got = {(_norm_path(i["ruta"]), i["linea"]) for i in obj["respuestas"]}
    want = {tuple(e) for e in expected}
    lines_exist = all(c["line_exists"] for c in verification.details["citations"])
    return got == want and lines_exist, {
        "got": sorted(got),
        "missing": sorted(want - got),
        "extra": sorted(got - want),
        "lines_exist": lines_exist,
        "strict_json": strict,
    }


def eval_locator(args, rec: Recorder, work: Path, results: Path) -> dict:
    trial = trial_for("locator", args)
    root = make_copy(work / "locator_copy")
    questions = locator_questions(root)
    if args.limit:
        questions = questions[: args.limit]
    reps = args.reps or REPS["locator"]
    for q in questions:
        if not q["expected"]:
            print(f"aviso: la pregunta {q['id']} no tiene respuestas esperadas", file=sys.stderr)
    for q in questions:
        for rep in range(1, reps + 1):
            before = snapshot(root)
            result = run_task("locator", q["question"], root=root, log_dir=results / "logs", run_label=f"{trial.label}_{q['id']}_{rep}_v{args.env_variant}", env_variant=args.env_variant, **trial.overrides)
            ok, details = (grade_locator(result.report, q["expected"], root) if result.stop_reason == "final" else (False, {"errors": [f"parada: {result.stop_reason}"]}))
            details["expected"] = q["expected"]
            details["kind"] = q["kind"]
            note = f"esperadas={len(q['expected'])} aciertos={'sí' if ok else 'NO'} faltan={len(details.get('missing', []))} sobran={len(details.get('extra', []))} {details.get('errors', '')}"
            verification = verify("locator", result.report, root=root) if result.report else None
            rec.add(_system_record("locator", q["id"], rep, result, root, before, args.env_variant, ok, note, details, verification, trial))
    runs = rec.runs
    exact = sum(r.success for r in runs) / len(runs) if runs else 0.0
    by_kind: dict[str, list[bool]] = {}
    for r in runs:
        by_kind.setdefault(r.details.get("kind", "?"), []).append(r.success)
    return {
        "metrics": {"exact_rate": exact, "exact": sum(r.success for r in runs), "runs": len(runs), "by_kind": {k: f"{sum(v)}/{len(v)}" for k, v in by_kind.items()}},
        "criteria": [("Respuestas exactas (conjunto (ruta, línea) igual al esperado, líneas existentes)", ">= 80 %", f"{100 * exact:.0f} % ({sum(r.success for r in runs)}/{len(runs)})", exact >= CRITERIA["locator"]["exact_rate_min"])],
    }


# -- doc_checker -------------------------------------------------------------------------------------------------
ROADMAP_REL = "context/constitution/roadmap.md"
TODO_REL = "context/TODO.md"

DOC_SEEDS = [
    {
        "id": "cifra",
        "type": "cifra cambiada",
        "file": TODO_REL,
        "old": "hecha en `feat/wgan-gp` (4 de 5 condiciones",
        "new": "hecha en `feat/wgan-gp` (3 de 5 condiciones",
        "partner_file": ROADMAP_REL,
        "partner_text": "cumple 4 de las 5 condiciones",
    },
    {
        "id": "estado",
        "type": "estado hecho/pendiente invertido",
        "file": TODO_REL,
        "old": "- [ ] Fase 3, GAN condicional por longitud e hidrofobicidad a la vez",
        "new": "- [x] Fase 3, GAN condicional por longitud e hidrofobicidad a la vez",
        "partner_file": ROADMAP_REL,
        "partner_text": "- [ ] GAN condicional con **longitud e hidrofobicidad a la vez**",
    },
    {
        "id": "fichero",
        "type": "nombre de fichero inexistente",
        "file": TODO_REL,
        "old": "(`make_clean_data.py`, 34 tests)",
        "new": "(`make_clean_dataset.py`, 34 tests)",
        "partner_file": ROADMAP_REL,
        "partner_text": "`src/amp_gan/make_clean_data.py`",
    },
    {
        "id": "tests",
        "type": "número de tests distinto",
        "file": TODO_REL,
        "old": "validados por el orquestador (49 tests)",
        "new": "validados por el orquestador (46 tests)",
        "partner_file": ROADMAP_REL,
        "partner_text": "`src/amp_gan/properties.py`, 49 tests",
    },
    {
        "id": "fecha",
        "type": "fecha distinta",
        "file": TODO_REL,
        "old": "DRAMP 5.0, descargada el 28-oct-2026",
        "new": "DRAMP 5.0, descargada el 26-oct-2026",
        "partner_file": ROADMAP_REL,
        "partner_text": "DRAMP 5.0, descargada el 28-oct-2026",
    },
    {
        "id": "notebook",
        "type": "nombre de notebook distinto",
        "file": TODO_REL,
        "old": "(criterios A y B cumplidos; ver roadmap)",
        "new": "(criterios A y B cumplidos; ver roadmap; notebook `src/notebooks/04_gan_longitud.ipynb`)",
        "partner_file": ROADMAP_REL,
        "partner_text": "`src/notebooks/04_gan_condicional_longitud.ipynb`",
    },
]


def _line_of(text: str, fragment: str) -> int:
    index = text.find(fragment)
    return text.count("\n", 0, index) + 1


def seed_documents(roadmap: str, todo: str, seeds: list[dict] | None = None) -> tuple[str, str, list[dict]]:
    """Siembra las incoherencias en la copia del TODO y devuelve ``(roadmap, todo, registros)``.

    Cada registro guarda tipo, fichero, línea (en el texto ya alterado) y el texto antes y después, más el fragmento
    y la línea del otro documento con el que choca. Cada texto original debe aparecer exactamente una vez.
    """
    texts = {ROADMAP_REL: roadmap, TODO_REL: todo}
    records = []
    for seed in seeds or DOC_SEEDS:
        text = texts[seed["file"]]
        if text.count(seed["old"]) != 1:
            raise ValueError(f"el fragmento original de la siembra {seed['id']!r} aparece {text.count(seed['old'])} veces en {seed['file']}")
        texts[seed["file"]] = text.replace(seed["old"], seed["new"], 1)
    for seed in seeds or DOC_SEEDS:
        partner = texts[seed["partner_file"]]
        if partner.count(seed["partner_text"]) < 1:
            raise ValueError(f"no se encuentra el texto con el que choca la siembra {seed['id']!r}")
        records.append(
            {
                "id": seed["id"],
                "type": seed["type"],
                "file": seed["file"],
                "line": _line_of(texts[seed["file"]], seed["new"]),
                "old": seed["old"],
                "new": seed["new"],
                "partner_file": seed["partner_file"],
                "partner_line": _line_of(partner, seed["partner_text"]),
            }
        )
    return texts[ROADMAP_REL], texts[TODO_REL], records


def match_doc_findings(citations: list[dict], seeds: list[dict], max_span: int = 2) -> dict:
    """Qué incoherencias sembradas recupera cada hallazgo (por sus citas literales válidas) y cuáles son falsos positivos.

    Estricto: una cita cae sobre la línea alterada. Laxo (informativo): o sobre la línea del otro documento.
    """
    spans: dict[int, list[tuple[str, int, int]]] = {}
    for c in citations:
        if c["valid"] and c["last_line"] - c["first_line"] + 1 <= max_span:
            spans.setdefault(c["item"], []).append((_norm_path(str(c["ruta"])), c["first_line"], c["last_line"]))
    items = sorted({c["item"] for c in citations})
    strict: set[str] = set()
    lenient: set[str] = set()
    useful: set[int] = set()
    for item in items:
        for seed in seeds:
            for ruta, first, last in spans.get(item, []):
                if ruta == seed["file"] and first <= seed["line"] <= last:
                    strict.add(seed["id"])
                    lenient.add(seed["id"])
                    useful.add(item)
                elif ruta == seed["partner_file"] and first <= seed["partner_line"] <= last:
                    lenient.add(seed["id"])
                    useful.add(item)
    return {"recovered": sorted(strict), "recovered_lenient": sorted(lenient), "false_positives": [i for i in items if i not in useful]}


DOC_NOTE = "cifras, fechas, nombres de fichero y estados hecho/pendiente de una misma tarea"


def frozen_docs() -> tuple[str, str]:
    """``(roadmap, TODO)`` CONGELADOS, tal como estaban al hacer la evaluación original (commit 4a98ff2).

    Los documentos vivos del repositorio cambian con el trabajo; el caso evaluado (y con él las siembras y los criterios)
    no debe cambiar, así que la evaluación y sus tests leen estas copias de ``amp_agents/fixtures/``.
    """
    return (
        (FIXTURES_DIR / "roadmap.md").read_text(encoding="utf-8"),
        (FIXTURES_DIR / "TODO.md").read_text(encoding="utf-8"),
    )


def eval_doc_checker(args, rec: Recorder, work: Path, results: Path, role: str = "doc_checker") -> dict:
    """Evalúa ``doc_checker`` o una variante suya (``doc_checker_ss``) con el mismo protocolo y los mismos criterios.

    ``role`` solo cambia el rol que se ejecuta y los nombres de carpetas, registros y ficheros de siembra; las
    repeticiones y los umbrales salen siempre de la entrada ``doc_checker`` de ``REPS`` y ``CRITERIA``. Con ``--limit N``
    solo se ejecutan los N primeros casos de cada repetición (``seeded``, ``clean``).
    """
    trial = trial_for(role, args)
    roadmap_text, todo_text = frozen_docs()
    seeded_roadmap, seeded_todo, seeds = seed_documents(roadmap_text, todo_text)
    names = ("seeded", "clean")[: args.limit] if args.limit else ("seeded", "clean")
    copies = {}
    for name, (road, todo) in {"seeded": (seeded_roadmap, seeded_todo), "clean": (roadmap_text, todo_text)}.items():
        if name not in names:
            continue
        root = make_copy(work / f"{role}_{name}")
        (root / ROADMAP_REL).write_text(road, encoding="utf-8")
        (root / TODO_REL).write_text(todo, encoding="utf-8")
        copies[name] = root
    (results / f"seeds_{trial.label}.json").write_text(json.dumps(seeds, ensure_ascii=False, indent=1), encoding="utf-8")
    reps = args.reps or REPS[base_role(role)]
    criteria = CRITERIA[base_role(role)]
    verifier = get_agent_config(role, **trial.overrides).verifier or "doc_checker"
    per_rep: list[dict] = []
    for rep in range(1, reps + 1):
        entry = {"rep": rep}
        for name in names:
            root = copies[name]
            task, context = prepare_task(role, [ROADMAP_REL, TODO_REL], root, DOC_NOTE)
            before = snapshot(root)
            result = run_task(role, task, root=root, log_dir=results / "logs", run_label=f"{trial.label}_{name}_{rep}_v{args.env_variant}", env_variant=args.env_variant, **trial.overrides)
            verification = verify(verifier, result.report, root=root) if result.stop_reason == "final" else None
            citations = verification.details.get("citations", []) if verification else []
            obj, _, _ = parse_json_output(result.report) if result.report else (None, False, None)
            n_findings = len(obj.get("incoherencias", [])) if isinstance(obj, dict) and isinstance(obj.get("incoherencias"), list) else 0
            valid_output = bool(verification and verification.details.get("schema_ok"))
            if name == "seeded":
                match = match_doc_findings(citations, seeds)
                ok = valid_output and len(match["recovered"]) >= 1
                notes = f"hallazgos={n_findings} recuperadas={len(match['recovered'])}/6 {match['recovered']} laxo={len(match['recovered_lenient'])} falsos_positivos={len(match['false_positives'])}"
            else:
                match = {"recovered": [], "recovered_lenient": [], "false_positives": list(range(n_findings))}
                ok = valid_output and n_findings <= criteria["false_positives_max"]
                notes = f"hallazgos_en_copia_limpia={n_findings}"
            if verification and verification.errors:
                notes += f" errores={verification.errors[:2]}"
            if not verification:
                notes += f" sin salida utilizable (parada {result.stop_reason})"
            details = {"copy": name, "match": match, "n_findings": n_findings, "citations": citations, "valid_output": valid_output}
            record = _system_record(role, name, rep, result, root, before, args.env_variant, ok, notes, details, verification, trial)
            rec.add(record)
            entry[name] = {"recovered": len(match["recovered"]), "false_positives": len(match["false_positives"]) if name == "clean" else None, "findings": n_findings, "valid_output": valid_output}
        per_rep.append(entry)
    return _doc_summary(rec.runs, per_rep)


def _citation_stats(runs: list[RunRecord]) -> tuple[int, int]:
    total = valid = 0
    for r in runs:
        for c in r.details.get("citations", []):
            total += 1
            valid += bool(c.get("valid"))
    return valid, total


def _doc_summary(runs: list[RunRecord], per_rep: list[dict]) -> dict:
    valid, total = _citation_stats(runs)
    rate = valid / total if total else 0.0
    recovered = [e["seeded"]["recovered"] for e in per_rep if "seeded" in e]
    fps = [e["clean"]["false_positives"] for e in per_rep if "clean" in e]
    mean_rec = sum(recovered) / len(recovered) if recovered else 0.0
    c = CRITERIA["doc_checker"]
    return {
        "metrics": {"valid_citations": valid, "citations": total, "recovered_per_rep": recovered, "false_positives_clean_per_rep": fps, "per_rep": per_rep},
        "criteria": [
            ("Citas literales válidas (todas las ejecuciones)", ">= 90 %", f"{100 * rate:.0f} % ({valid}/{total})", total > 0 and rate >= c["valid_citations_min"]),
            ("Incoherencias sembradas recuperadas (media por repetición, de 6)", ">= 4", f"{mean_rec:.2f} {recovered}", mean_rec >= c["recovered_mean_min"]),
            ("Falsos positivos en la copia limpia (máximo por repetición)", "<= 2", f"{max(fps) if fps else '-'} {fps}", bool(fps) and max(fps) <= c["false_positives_max"]),
        ],
    }


# -- doc_checker_pairs: el código da los pares, el modelo juzga sí/no ------------------------------------------------------
def _pair_citations(citations: list[dict], positives: list) -> list[dict]:
    """Pone en las citas verificadas el número de línea REAL del par (``find_quote`` da la primera aparición del texto, que
    puede ser otra línea si el texto está repetido); la literalidad (``valid``) sigue siendo la del verificador."""
    out = []
    for c in citations:
        c = dict(c)
        pair = positives[c["item"]]
        line = pair.line_a if c["side"] == "evidencia_a" else pair.line_b
        if c["valid"]:
            c["quote_first_line"] = c["first_line"]
            c["first_line"] = c["last_line"] = line
        out.append(c)
    return out


def _match_pairs_directly(positives: list, seeds: list[dict]) -> dict:
    """``match_doc_findings`` sobre pares positivos con sus líneas reales (citas literales por construcción, sin leer ficheros)."""
    citations = []
    for item, pair in enumerate(positives):
        for side, path, line in (("evidencia_a", pair.path_a, pair.line_a), ("evidencia_b", pair.path_b, pair.line_b)):
            citations.append({"item": item, "side": side, "ruta": path, "valid": True, "first_line": line, "last_line": line})
    return match_doc_findings(citations, seeds)


def threshold_sensitivity(per_rep_judgments: list[dict], pair_sets: dict, seeds: list[dict], thresholds=SENSITIVITY_THRESHOLDS) -> list[dict]:
    """Cómo variarían los dos criterios que dependen del modelo con otros umbrales de probabilidad (solo informativo).

    ``per_rep_judgments``: por repetición completa, ``{"seeded": [Judgment...], "clean": [...]}``. Se reutilizan las probabilidades ya
    guardadas, sin nuevas peticiones, y NO sirve para elegir el umbral con el que se juzgan los criterios (el fijado de antemano).
    """
    rows = []
    for t in thresholds:
        recovered, fps = [], []
        for judged in per_rep_judgments:
            for name, judgments in judged.items():
                positives = [p for p, j in zip(pair_sets[name].pairs, judgments) if j.score is not None and j.score >= t]
                if name == "seeded":
                    recovered.append(len(_match_pairs_directly(positives, seeds)["recovered"]))
                else:
                    fps.append(len(positives))
        rows.append({"threshold": t, "recovered_per_rep": recovered, "recovered_mean": round(sum(recovered) / len(recovered), 2) if recovered else None, "false_positives_clean_per_rep": fps})
    return rows


def rep_fits_budget(elapsed_s: float, last_rep_s: float, budget_s: float | None) -> bool:
    """¿Cabe otra repetición, que tardará lo que la última, en el presupuesto? Sin presupuesto (``None``) siempre cabe."""
    return budget_s is None or elapsed_s + last_rep_s <= budget_s


def eval_doc_checker_pairs(args, rec: Recorder, work: Path, results: Path, role: str = "doc_checker_pairs") -> dict:
    """Evalúa el verificador acotado (``amp_agents.pairs``) con el protocolo y los criterios de ``doc_checker``.

    Por repetición y copia (``seeded``, ``clean``): se generan los pares candidatos (siempre los mismos), se pide al modelo
    un juicio por par, el código construye los hallazgos con citas literales y se puntúa igual que ``eval_doc_checker``.
    ``--max-minutes`` es un presupuesto de tiempo (ver ``amp_agents.pairs``); sin él se hacen todas las repeticiones.
    """
    trial = trial_for(role, args)
    roadmap_text, todo_text = frozen_docs()
    seeded_roadmap, seeded_todo, seeds = seed_documents(roadmap_text, todo_text)
    names = ("seeded", "clean")[: args.limit] if args.limit else ("seeded", "clean")
    copies = {}
    for name, (road, todo) in {"seeded": (seeded_roadmap, seeded_todo), "clean": (roadmap_text, todo_text)}.items():
        if name not in names:
            continue
        root = make_copy(work / f"{role}_{name}")
        (root / ROADMAP_REL).write_text(road, encoding="utf-8")
        (root / TODO_REL).write_text(todo, encoding="utf-8")
        copies[name] = root
    (results / f"seeds_{trial.label}.json").write_text(json.dumps(seeds, ensure_ascii=False, indent=1), encoding="utf-8")
    config = get_agent_config(role, **trial.overrides)
    provider = make_provider(config)
    mode = pair_mode(role)
    provider2 = None
    if mode == "agree":  # la segunda pregunta tiene su propio esquema de salida (``compatibles``)
        provider2 = make_provider(get_agent_config(role, **{**trial.overrides, "options": {**config.options, "format": COMPAT_FORMAT_SCHEMA}}))
    reps = args.reps or REPS[base_role(role)]
    criteria = CRITERIA[base_role(role)]
    pair_sets = {name: build_pairs((ROADMAP_REL, (root / ROADMAP_REL).read_text(encoding="utf-8")), (TODO_REL, (root / TODO_REL).read_text(encoding="utf-8"))) for name, root in copies.items()}
    coverage = {name: seed_coverage(ps.pairs, seeds) for name, ps in pair_sets.items()}
    (results / f"pairs_{trial.label}.json").write_text(
        json.dumps({"coverage": coverage, "per_seed": {s["id"]: s for s in seeds}, **{name: pairs_to_json(ps) for name, ps in pair_sets.items()}}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for name, ps in pair_sets.items():
        cov = coverage[name]
        print(f"    generador ({name}): {cov['pairs']} pares de {ps.candidates} candidatos; semillas con sus dos líneas {cov['n_both']}/{cov['seeds']} {cov['both']}, con la línea alterada {cov['n_seed_line']}/{cov['seeds']} {cov['seed_line']}", file=sys.stderr, flush=True)
    minutes = getattr(args, "max_minutes", None)
    budget = minutes * 60 if minutes else None
    started = time.monotonic()
    deadline = None if budget is None else started + budget
    per_rep: list[dict] = []
    rep_judgments: list[dict] = []  # por repetición completa: los juicios de cada copia (para la sensibilidad al umbral)
    all_stats: list[dict] = []
    last_rep_s = 0.0
    truncated_reps = 0
    for rep in range(1, reps + 1):
        if rep > 1 and not rep_fits_budget(time.monotonic() - started, last_rep_s, budget):
            print(f"    presupuesto de {minutes:g} min: no se empieza la repetición {rep} (la anterior tardó {last_rep_s / 60:.1f} min)", file=sys.stderr, flush=True)
            break
        rep_started = time.monotonic()
        entry = {"rep": rep}
        rep_truncated = False
        this_rep: dict = {}
        for name in names:
            root, ps = copies[name], pair_sets[name]
            before = snapshot(root)
            t0 = time.monotonic()
            judgments = judge_pairs(provider, ps.pairs, config.max_seconds, log_dir=results / "logs", label=f"{trial.label}_{name}_{rep}_v{args.env_variant}", root=root, deadline=deadline, mode=mode, provider2=provider2)
            this_rep[name] = judgments
            stats = judgment_stats(judgments)
            omitted = stats["by_status"].get("omitida", 0)
            findings = build_findings(ps.pairs, judgments)
            positives = [p for p, j in zip(ps.pairs, judgments) if j.positive]
            report = json.dumps(findings, ensure_ascii=False)
            verification = verify("doc_checker", report, root=root)
            citations = _pair_citations(verification.details.get("citations", []), positives) if not omitted else []
            n_findings = len(findings["incoherencias"])
            valid_output = bool(verification.details.get("schema_ok")) and not omitted
            if name == "seeded":
                match = match_doc_findings(citations, seeds)
                ok = valid_output and len(match["recovered"]) >= 1
                notes = f"positivos={n_findings}/{stats['asked']} recuperadas={len(match['recovered'])}/6 {match['recovered']} laxo={len(match['recovered_lenient'])} falsos_positivos={len(match['false_positives'])}"
            else:
                match = {"recovered": [], "recovered_lenient": [], "false_positives": list(range(n_findings))}
                ok = valid_output and n_findings <= criteria["false_positives_max"]
                notes = f"positivos_en_copia_limpia={n_findings}/{stats['asked']}"
            notes += f" formato={stats['format_failures']} fallos_peticion={stats['request_failures']} s/par={stats['seconds_mean']} tok/par={stats['completion_tokens_mean']}"
            if omitted:
                notes += f" TRUNCADA por presupuesto de tiempo ({omitted} pares sin pedir)"
                rep_truncated = True
            usage = Usage(sum(j.prompt_tokens for j in judgments), sum(j.completion_tokens for j in judgments))
            stop = "budget" if omitted else ("partial" if stats["request_failures"] else "final")
            result = AgentResult(
                report=report, steps=stats["asked"], stop_reason=stop, duration_s=time.monotonic() - t0, tokens=usage,
                last_prompt_tokens=max((j.prompt_tokens for j in judgments), default=None) or None,
                error="; ".join(sorted({f"{j.status}: {j.motivo}" for j in judgments if j.status in {"timeout", "error"}}))[:300] or None,
            )
            details = {
                "copy": name, "match": match, "n_findings": n_findings, "citations": citations, "valid_output": valid_output,
                "judgments": [asdict(j) for j in judgments], "judgment_stats": stats, "truncated": bool(omitted), "pairs": len(ps.pairs),
            }
            rec.add(_system_record(role, name, rep, result, root, before, args.env_variant, ok, notes, details, verification, trial))
            all_stats.append(stats)
            entry[name] = {"recovered": len(match["recovered"]), "false_positives": len(match["false_positives"]) if name == "clean" else None, "findings": n_findings, "valid_output": valid_output, "format_failures": stats["format_failures"]}
        last_rep_s = time.monotonic() - rep_started
        if rep_truncated:
            truncated_reps += 1
            break
        per_rep.append(entry)
        rep_judgments.append(this_rep)
    summary = _doc_summary(rec.runs, per_rep)
    asked = sum(s["asked"] for s in all_stats)
    seconds = sum(s["seconds_total"] for s in all_stats)
    reps_done = len(per_rep)
    format_failures = sum(s["format_failures"] for s in all_stats)
    request_failures = sum(s["request_failures"] for s in all_stats)
    positives_total = sum(s["positives"] for s in all_stats)
    incomplete = reps_done < reps
    summary["metrics"].update(
        {
            "generator_coverage": coverage, "reps_done": reps_done, "reps_planned": reps, "incomplete": incomplete, "truncated_reps": truncated_reps,
            "requests": asked, "format_failures": format_failures, "request_failures": request_failures, "positives": positives_total,
            "seconds_per_pair_mean": round(seconds / asked, 2) if asked else 0.0, "model_seconds_total": round(seconds, 1),
        }
    )
    lines = [
        f"Generador de pares (determinista, antes del modelo; {', '.join(f'{n}: {c['pairs']} pares de {pair_sets[n].candidates} candidatos' for n, c in coverage.items())}).",
    ]
    if "seeded" in coverage:
        c = coverage["seeded"]
        lines.append(f"Cobertura del generador en la copia sembrada: {c['n_both']}/{c['seeds']} semillas con sus dos líneas en un mismo par {c['both']}; {c['n_seed_line']}/{c['seeds']} con la línea alterada en algún par {c['seed_line']}.")
    lines.append(
        f"Peticiones: {asked} (una por par), fallos de formato {format_failures} ({100 * format_failures / asked:.1f} %), fallos de petición (tiempo o servidor) {request_failures}, "
        f"pares declarados contradictorios {positives_total}; {summary['metrics']['seconds_per_pair_mean']} s por par de media."
        if asked else "Sin peticiones."
    )
    if any(j.score is not None for judged in rep_judgments for js in judged.values() for j in js):
        sens = threshold_sensitivity(rep_judgments, pair_sets, seeds)
        summary["metrics"]["threshold_sensitivity"] = sens
        lines.append("Modelo de decisión: criterios juzgados con el umbral fijado de antemano (0,5). Sensibilidad (solo informativa, no para elegir umbral): " + "; ".join(f"u={r['threshold']:g}: recuperadas {r['recovered_mean']} {r['recovered_per_rep']}, falsos positivos {r['false_positives_clean_per_rep']}" for r in sens))
    lines.append("Las citas las copia el código del documento: son literales por construcción, así que el criterio de citas válidas es trivial y no mide al modelo.")
    if incomplete:
        lines.append(f"PROTOCOLO INCOMPLETO: {reps_done} de {reps} repeticiones completas (presupuesto de tiempo {minutes:g} min); los criterios se calculan sobre las completas.")
    summary["notes"] = lines
    return summary


# -- code_reviewer -------------------------------------------------------------------------------------------------
CODE_REL = "src/amp_gan/properties.py"
CODE_SEEDS = [
    {
        "id": "validacion",
        "type": "validación eliminada",
        "old": "    _validate(seq)\n    counts = Counter(seq)\n    n = len(seq)\n",
        "new": "    counts = Counter(seq)\n    n = len(seq)\n",
    },
    {
        "id": "limite",
        "type": "límite incorrecto",
        "old": "    if fractions.size == 0:\n",
        "new": "    if fractions.size <= 20:\n",
    },
    {
        "id": "signo",
        "type": "signo invertido",
        "old": "    return float(positive - negative)\n",
        "new": "    return float(negative - positive)\n",
    },
    {
        "id": "indice",
        "type": "índice desplazado",
        "old": "    return float(sum(KYTE_DOOLITTLE[aa] for aa in seq) / len(seq))\n",
        "new": "    return float(sum(KYTE_DOOLITTLE[aa] for aa in seq[1:]) / len(seq))\n",
    },
    {
        "id": "argumentos",
        "type": "argumentos intercambiados",
        "old": '            "net_charge": [net_charge(s, ph) for s in sequences],\n',
        "new": '            "net_charge": [net_charge(ph, s) for s in sequences],\n',
    },
]


def seed_code_defects(source: str, seeds: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Inyecta los defectos y devuelve ``(código, registros)`` con la línea de cada uno EN EL CÓDIGO YA ALTERADO.

    Para una validación eliminada, la línea registrada es la que ocupa ahora el sitio de la línea borrada.
    Cada fragmento original debe aparecer exactamente una vez.
    """
    seeds = seeds or CODE_SEEDS
    located = []
    for seed in seeds:
        if source.count(seed["old"]) != 1:
            raise ValueError(f"el fragmento original del defecto {seed['id']!r} aparece {source.count(seed['old'])} veces")
        located.append((source.index(seed["old"]), seed))
    located.sort(key=lambda item: item[0])
    text = source
    delta = 0
    records = []
    for index, seed in located:
        original_line = source.count("\n", 0, index) + 1
        records.append(
            {
                "id": seed["id"],
                "type": seed["type"],
                "line": original_line + delta,
                "old": seed["old"].strip("\n"),
                "new": seed["new"].strip("\n"),
            }
        )
        delta += seed["new"].count("\n") - seed["old"].count("\n")
        text = text.replace(seed["old"], seed["new"], 1)
    # comprobación: la línea registrada contiene el texto nuevo (o, si se borró, el que ocupa su sitio)
    lines = text.splitlines()
    for record in records:
        expected = record["new"].splitlines()[0] if record["new"] else ""
        if record["new"] and expected not in lines[record["line"] - 1]:
            raise AssertionError(f"la línea registrada de {record['id']!r} no contiene el texto sembrado")
    return text, sorted(records, key=lambda r: r["line"])


def match_code_findings(obj, seeds: list[dict], tolerance: int = 2) -> dict:
    """Defectos sembrados recuperados (línea del hallazgo a ``tolerance`` o menos) y hallazgos sin defecto cerca."""
    findings = obj.get("hallazgos", []) if isinstance(obj, dict) else []
    recovered: set[str] = set()
    useless = []
    for index, finding in enumerate(findings):
        linea = finding.get("linea") if isinstance(finding, dict) else None
        hit = False
        if isinstance(linea, int) and not isinstance(linea, bool):
            for seed in seeds:
                if abs(linea - seed["line"]) <= tolerance:
                    recovered.add(seed["id"])
                    hit = True
        if not hit:
            useless.append(index)
    return {"recovered": sorted(recovered), "false_positives": useless, "n_findings": len(findings)}


def eval_code_reviewer(args, rec: Recorder, work: Path, results: Path) -> dict:
    trial = trial_for("code_reviewer", args)
    source = (PROJECT_ROOT / CODE_REL).read_text(encoding="utf-8")
    seeded_text, seeds = seed_code_defects(source)
    (results / "seeds_code_reviewer.json").write_text(json.dumps(seeds, ensure_ascii=False, indent=1), encoding="utf-8")
    copies = {}
    for name, text in {"seeded": seeded_text, "clean": source}.items():
        root = make_copy(work / f"code_reviewer_{name}")
        (root / CODE_REL).write_text(text, encoding="utf-8")
        copies[name] = (root, text)
    reps = args.reps or REPS["code_reviewer"]
    per_rep = []
    for rep in range(1, reps + 1):
        entry = {"rep": rep}
        for name in ("seeded", "clean"):
            root, text = copies[name]
            task, context = prepare_task("code_reviewer", [CODE_REL], root)
            before = snapshot(root)
            result = run_task("code_reviewer", task, root=root, log_dir=results / "logs", run_label=f"{trial.label}_{name}_{rep}_v{args.env_variant}", env_variant=args.env_variant, **trial.overrides)
            verification = verify("code_reviewer", result.report, root=root, source=text) if result.stop_reason == "final" else None
            obj, _, _ = parse_json_output(result.report) if result.report else (None, False, None)
            valid_output = bool(verification and verification.details.get("schema_ok"))
            match = match_code_findings(obj, seeds if name == "seeded" else []) if valid_output else {"recovered": [], "false_positives": [], "n_findings": 0}
            citations = verification.details.get("citations", []) if verification else []
            # salida CORREGIDA por código (línea reasignada a la de la cita literal); la cruda es la que juzgan los criterios
            correction = verification.details.get("line_correction") if verification and valid_output else None
            match_fixed = match_code_findings(verifiable_findings(correction["obj"]), seeds if name == "seeded" else []) if correction else {"recovered": [], "false_positives": [], "n_findings": 0}
            if name == "seeded":
                ok = valid_output and len(match["recovered"]) >= 1
                notes = f"hallazgos={match['n_findings']} recuperados={len(match['recovered'])}/5 {match['recovered']} sin_defecto_cerca={len(match['false_positives'])}"
                if correction:
                    notes += f" | línea corregida: recuperados={len(match_fixed['recovered'])}/5 {match_fixed['recovered']} corregidas={correction['corrected']} no_verificables={correction['unverifiable']}"
            else:
                ok = valid_output and match["n_findings"] <= CRITERIA["code_reviewer"]["false_positives_max"]
                notes = f"hallazgos_en_copia_limpia={match['n_findings']}"
            if not valid_output:
                notes += f" salida no utilizable: {(verification.errors[:2] if verification else 'parada ' + result.stop_reason)}"
            details = {
                "copy": name, "match": match, "citations": citations, "valid_output": valid_output, "seeds": seeds if name == "seeded" else [],
                "match_corrected": match_fixed, "line_correction": None if not correction else {k: v for k, v in correction.items() if k != "obj"},
            }
            rec.add(_system_record("code_reviewer", name, rep, result, root, before, args.env_variant, ok, notes, details, verification, trial))
            entry[name] = {
                "recovered": len(match["recovered"]), "findings": match["n_findings"], "valid_output": valid_output,
                "recovered_corrected": len(match_fixed["recovered"]), "findings_verifiable": correction["verifiable"] if correction else 0,
            }
        per_rep.append(entry)
    return _code_summary(rec.runs, per_rep)


def _code_summary(runs: list[RunRecord], per_rep: list[dict]) -> dict:
    """Resumen del revisor: los criterios (fijados) se juzgan sobre la salida CRUDA; se informa además, sin cambiar el
    umbral, de la salida con la línea corregida por código (``verifiers.correct_review_lines``)."""
    valid, total = _citation_stats(runs)
    rate = valid / total if total else 0.0
    recovered = [e["seeded"]["recovered"] for e in per_rep]
    recovered_fixed = [e["seeded"].get("recovered_corrected", 0) for e in per_rep]
    fps = [e["clean"]["findings"] for e in per_rep]
    mean_rec = sum(recovered) / len(recovered) if recovered else 0.0
    mean_fixed = sum(recovered_fixed) / len(recovered_fixed) if recovered_fixed else 0.0
    fixed_stats = [r.details.get("line_correction") for r in runs if r.details.get("line_correction")]
    n_findings = sum(len(r.details.get("citations", [])) for r in runs)
    verifiable = sum(f["verifiable"] for f in fixed_stats)
    shifted = sum(f["corrected"] for f in fixed_stats)
    fps_fixed = [e["clean"].get("findings_verifiable", e["clean"]["findings"]) for e in per_rep]
    c = CRITERIA["code_reviewer"]
    return {
        "metrics": {
            "valid_citations": valid, "citations": total, "recovered_per_rep": recovered, "false_positives_clean_per_rep": fps, "per_rep": per_rep,
            "recovered_corrected_per_rep": recovered_fixed, "recovered_corrected_mean": mean_fixed, "recovered_raw_mean": mean_rec,
            "findings": n_findings, "findings_verifiable": verifiable, "lines_corrected": shifted, "findings_unverifiable": n_findings - verifiable,
            "false_positives_clean_corrected_per_rep": fps_fixed,
        },
        "criteria": [
            ("Defectos sembrados recuperados (media por repetición, de 5)", ">= 3", f"{mean_rec:.2f} {recovered}", mean_rec >= c["recovered_mean_min"]),
            ("Citas válidas (línea existente y cita literal; todas las ejecuciones)", ">= 90 %", f"{100 * rate:.0f} % ({valid}/{total})", total > 0 and rate >= c["valid_citations_min"]),
            ("Falsos positivos en la copia limpia (máximo por repetición)", "<= 2", f"{max(fps) if fps else '-'} {fps}", bool(fps) and max(fps) <= c["false_positives_max"]),
            # misma medida con la línea corregida por código (informativo; mismo umbral, no sustituye a las anteriores)
            ("[línea corregida por código] Defectos sembrados recuperados (media por repetición, de 5)", ">= 3", f"{mean_fixed:.2f} {recovered_fixed} ({shifted} líneas corregidas, {n_findings - verifiable} no verificables)", mean_fixed >= c["recovered_mean_min"]),
            ("[línea corregida por código] Citas verificables (la cita literal está en el fichero)", ">= 90 %", f"{100 * verifiable / n_findings:.0f} % ({verifiable}/{n_findings})" if n_findings else "-", n_findings > 0 and verifiable / n_findings >= c["valid_citations_min"]),
            ("[línea corregida por código] Falsos positivos en la copia limpia (solo hallazgos verificables; máximo por repetición)", "<= 2", f"{max(fps_fixed) if fps_fixed else '-'} {fps_fixed}", bool(fps_fixed) and max(fps_fixed) <= c["false_positives_max"]),
        ],
    }


# -- code_reviewer_funcs: una pregunta por función y por tipo de defecto ---------------------------------------------------
def eval_code_reviewer_funcs(args, rec: Recorder, work: Path, results: Path, role: str = "code_reviewer_funcs") -> dict:
    """Evalúa el revisor acotado (``amp_agents.func_review``) con el protocolo y los criterios de ``code_reviewer``.

    Mismo fichero, mismas semillas y misma copia sembrada y limpia que ``eval_code_reviewer``. Por repetición y copia: el código divide
    el fichero en funciones, hace UNA petición por función y categoría de defecto, y construye los hallazgos con la cita copiada de la
    línea (literal por construcción). ``--max-minutes`` es un presupuesto de tiempo (como en ``doc_checker_pairs``).
    """
    trial = trial_for(role, args)
    source = (PROJECT_ROOT / CODE_REL).read_text(encoding="utf-8")
    seeded_text, seeds = seed_code_defects(source)
    (results / f"seeds_{trial.label}.json").write_text(json.dumps(seeds, ensure_ascii=False, indent=1), encoding="utf-8")
    names = ("seeded", "clean")
    copies = {}
    for name, text in {"seeded": seeded_text, "clean": source}.items():
        root = make_copy(work / f"{role}_{name}")
        (root / CODE_REL).write_text(text, encoding="utf-8")
        copies[name] = (root, text, split_functions(text))
    config = get_agent_config(role, **trial.overrides)
    provider = make_provider(config)
    reps = args.reps or REPS[base_role(role)]
    criteria = CRITERIA[base_role(role)]
    requests_per_rep = len(CATEGORIES) * sum(len(units) for _, _, units in copies.values())
    layout = {name: [(u.title, u.start, u.end) for u in units] for name, (_, _, units) in copies.items()}
    (results / f"units_{trial.label}.json").write_text(json.dumps(layout, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, (_, _, units) in copies.items():
        print(f"    unidades ({name}): {len(units)} funciones o trozos x {len(CATEGORIES)} categorías = {len(units) * len(CATEGORIES)} peticiones por repetición y copia", file=sys.stderr, flush=True)
    minutes = getattr(args, "max_minutes", None)
    budget = minutes * 60 if minutes else None
    started = time.monotonic()
    deadline = None if budget is None else started + budget
    per_rep: list[dict] = []
    all_stats: list[dict] = []
    rows: list[dict] = []  # una fila por petición de las repeticiones completas (para el sesgo hacia «sí»)
    strict_by_rep: list[list[str]] = []
    any_by_rep: list[list[str]] = []
    last_rep_s = 0.0
    truncated_reps = 0
    for rep in range(1, reps + 1):
        if rep > 1 and not rep_fits_budget(time.monotonic() - started, last_rep_s, budget):
            print(f"    presupuesto de {minutes:g} min: no se empieza la repetición {rep} (la anterior tardó {last_rep_s / 60:.1f} min)", file=sys.stderr, flush=True)
            break
        rep_started = time.monotonic()
        entry = {"rep": rep}
        rep_truncated = False
        rep_rows: list[dict] = []
        for name in names:
            root, text, units = copies[name]
            before = snapshot(root)
            t0 = time.monotonic()
            verdicts = review_units(provider, CODE_REL, text, units, config.max_seconds, log_dir=results / "logs", label=f"{trial.label}_{name}_{rep}_v{args.env_variant}", root=root, deadline=deadline)
            stats = request_stats(verdicts)
            omitted = stats["by_status"].get("omitida", 0)
            obj = build_func_findings(verdicts, units)
            report = json.dumps(obj, ensure_ascii=False)
            verification = verify("code_reviewer", report, root=root, source=text)
            valid_output = bool(verification.details.get("schema_ok")) and not omitted
            seeds_here = seeds if name == "seeded" else []
            match = match_code_findings(obj, seeds_here) if valid_output else {"recovered": [], "false_positives": [], "n_findings": 0}
            by_category = match_by_category(obj["hallazgos"], seeds_here) if valid_output else {"any": [], "strict": []}
            citations = verification.details.get("citations", [])
            correction = verification.details.get("line_correction") if valid_output else None
            match_fixed = match_code_findings(verifiable_findings(correction["obj"]), seeds_here) if correction else {"recovered": [], "false_positives": [], "n_findings": 0}
            flags = seed_flags(verdicts, units, seeds_here)
            for v, flag in zip(verdicts, flags):
                if v.status != "omitida":
                    rep_rows.append({"copy": name, "unit_name": units[v.unit].title, "category": v.category, "said_yes": v.said_yes, **flag})
            found_categories = sorted({f["categoria"] for f in obj["hallazgos"]})
            if name == "seeded":
                ok = valid_output and len(match["recovered"]) >= 1
                notes = f"hallazgos={match['n_findings']} recuperados={len(match['recovered'])}/5 {match['recovered']} con su categoría={len(by_category['strict'])} {by_category['strict']} sin_defecto_cerca={len(match['false_positives'])}"
            else:
                ok = valid_output and match["n_findings"] <= criteria["false_positives_max"]
                notes = f"hallazgos_en_copia_limpia={match['n_findings']} categorías={found_categories}"
            notes += f" | peticiones={stats['asked']} sí={stats['yes']} formato={stats['format_failures']} fuera_de_rango={stats['out_of_range']} fallos_peticion={stats['request_failures']} s/petición={stats['seconds_mean']}"
            if omitted:
                notes += f" TRUNCADA por presupuesto de tiempo ({omitted} peticiones sin hacer)"
                rep_truncated = True
            usage = Usage(sum(v.prompt_tokens for v in verdicts), sum(v.completion_tokens for v in verdicts))
            stop = "budget" if omitted else ("partial" if stats["request_failures"] else "final")
            result = AgentResult(
                report=report, steps=stats["asked"], stop_reason=stop, duration_s=time.monotonic() - t0, tokens=usage,
                last_prompt_tokens=max((v.prompt_tokens for v in verdicts), default=None) or None,
                error="; ".join(sorted({f"{v.status}: {v.motivo}" for v in verdicts if v.status in {"timeout", "error"}}))[:300] or None,
            )
            details = {
                "copy": name, "match": match, "citations": citations, "valid_output": valid_output, "seeds": seeds_here, "by_category": by_category,
                "match_corrected": match_fixed, "line_correction": None if not correction else {k: v for k, v in correction.items() if k != "obj"},
                "findings": obj["hallazgos"], "verdicts": [{**asdict(v), **flag} for v, flag in zip(verdicts, flags)], "request_stats": stats, "truncated": bool(omitted),
            }
            rec.add(_system_record(role, name, rep, result, root, before, args.env_variant, ok, notes, details, verification, trial))
            all_stats.append(stats)
            entry[name] = {
                "recovered": len(match["recovered"]), "findings": match["n_findings"], "valid_output": valid_output, "recovered_corrected": len(match_fixed["recovered"]),
                "findings_verifiable": correction["verifiable"] if correction else 0, "recovered_strict": len(by_category["strict"]),
            }
            if name == "seeded":
                strict_by_rep.append(by_category["strict"])
                any_by_rep.append(by_category["any"])
            else:
                entry[name]["by_category"] = {k: sum(1 for f in obj["hallazgos"] if f["categoria"] == k) for k in CATEGORY_BY_KEY}
        last_rep_s = time.monotonic() - rep_started
        if rep_truncated:
            truncated_reps += 1
            strict_by_rep, any_by_rep = strict_by_rep[: len(per_rep)], any_by_rep[: len(per_rep)]
            break
        per_rep.append(entry)
        rows.extend(rep_rows)
    summary = _code_summary(rec.runs, per_rep)
    asked = sum(st["asked"] for st in all_stats)
    reps_done = len(per_rep)
    incomplete = reps_done < reps
    seconds = sum(st["seconds_total"] for st in all_stats)
    per_seed = {s["id"]: {"any": sum(s["id"] in found for found in any_by_rep), "strict": sum(s["id"] in found for found in strict_by_rep)} for s in seeds}
    clean_by_category = {k: sum(e["clean"].get("by_category", {}).get(k, 0) for e in per_rep) for k in CATEGORY_BY_KEY}
    rates = yes_rates(rows)
    processors = sorted({r.processor for r in rec.runs if r.processor})
    summary["metrics"].update(
        {
            "reps_done": reps_done, "reps_planned": reps, "incomplete": incomplete, "truncated_reps": truncated_reps, "requests": asked, "requests_per_rep": requests_per_rep,
            "units": {name: len(units) for name, (_, _, units) in copies.items()}, "seeds_recovered_by_rep": per_seed, "recovered_strict_per_rep": [e["seeded"]["recovered_strict"] for e in per_rep if "seeded" in e],
            "clean_findings_by_category": clean_by_category, "yes_rates": rates, "format_failures": sum(st["format_failures"] for st in all_stats), "out_of_range": sum(st["out_of_range"] for st in all_stats),
            "blank_line": sum(st["blank_line"] for st in all_stats), "request_failures": sum(st["request_failures"] for st in all_stats), "yes_total": sum(st["yes"] for st in all_stats),
            "seconds_per_request_mean": round(seconds / asked, 2) if asked else 0.0, "model_seconds_total": round(seconds, 1), "prompt_tokens_max": max((st["prompt_tokens_max"] for st in all_stats), default=0),
            "processors": processors
        }
    )

    def pct(item: dict) -> str:
        return f"{item['pct']:g} % ({item['yes']}/{item['n']})" if item["n"] else "-"

    lines = [
        f"Peticiones: {asked} en total ({summary['metrics']['requests_per_rep']} por repetición: ({' + '.join(str(len(u)) for _, _, u in copies.values())} unidades de la copia sembrada y la limpia) x {len(CATEGORIES)} categorías), "
        f"{summary['metrics']['seconds_per_request_mean']} s por petición de media, prompt máximo {summary['metrics']['prompt_tokens_max']} tokens (num_ctx {config.options.get('num_ctx')}); procesador {', '.join(processors) or 'desconocido'}.",
        f"Respuestas «sí»: {summary['metrics']['yes_total']} de {asked}; fallos de formato {summary['metrics']['format_failures']}, línea fuera del rango de la función {summary['metrics']['out_of_range']}, línea en blanco {summary['metrics']['blank_line']}, fallos de petición (tiempo o servidor) {summary['metrics']['request_failures']}.",
        "Las citas las copia el código de la línea indicada: son literales y la línea existe por construcción, así que el criterio de citas válidas NO mide al modelo.",
        "Defectos sembrados recuperados por semilla (repeticiones completas en que se recupera; «cualquiera» = hallazgo de cualquier categoría a ±2 líneas, lo que juzga el criterio; «su categoría» = hallazgo de la categoría de la semilla): "
        + "; ".join(f"{k}: cualquiera {v['any']}/{reps_done}, su categoría {v['strict']}/{reps_done}" for k, v in per_seed.items()) + ".",
        f"Hallazgos en la copia limpia por categoría (suma de las repeticiones completas): {clean_by_category}.",
        f"Sesgo hacia «sí»: funciones con semilla {pct(rates['with_seed'])} frente al resto {pct(rates['without_seed'])}; copia sembrada {pct(rates['seeded_copy'])}, copia limpia {pct(rates['clean_copy'])}; "
        f"celdas (función, categoría) de las semillas {pct(rates['seed_cell_seeded'])} en la sembrada frente a {pct(rates['seed_cell_clean'])} en la limpia; resto de celdas de la sembrada {pct(rates['other_cells_seeded'])}.",
    ]
    if incomplete:
        lines.append(f"PROTOCOLO INCOMPLETO: {reps_done} de {reps} repeticiones completas (presupuesto de tiempo {minutes:g} min); los criterios se calculan sobre las completas.")
    summary["notes"] = lines
    return summary


# -- summarizer ------------------------------------------------------------------------------------------------------
SUMMARY_INPUTS = {
    "hydro": ("cgan_hydro_metrics.csv", "cgan_hydro_run_info.json"),
    "length": ("cgan_length_metrics.csv", "cgan_length_run_info.json"),
}
RUN_INFO_DROP = ("versions", "backend_decision", "keras_devices", "cuda_visible_devices", "gpu_visible", "device_used", "keras_backend", "backend_mode", "created_utc")
SUMMARY_MAX_CHARS = 4500


def trim_run_info(text: str) -> str:
    """Recorta un ``run_info.json`` a lo necesario: quita versiones y decisión de backend y lo compacta."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text[:SUMMARY_MAX_CHARS]
    for key in RUN_INFO_DROP:
        data.pop(key, None)
    return json.dumps(data, ensure_ascii=False, separators=(", ", ": "))


def summary_corrected_metrics(report: str, source: str, max_words: int | None = None) -> dict:
    """Métricas del resumen tras la corrección por código (coma decimal y recorte a ``max_words``, por defecto el criterio)."""
    limit = max_words or CRITERIA["summarizer"]["max_words"]
    fixed, info = postprocess_summary(report, source, limit)
    return {
        "text": fixed,
        "mode": info["mode"],
        "words": count_words(fixed),
        "numbers": info["numbers_after"],
        "numbers_raw": info["numbers_before"],
        "spanish": spanish_score(fixed),
    }


def summary_filtered_metrics(report: str, source: str, max_words: int | None = None) -> dict:
    """Métricas del resumen tras recorte + filtro de cifras no respaldadas (``postprocess_summary(filter_numbers=True)``).

    ``failure`` es verdadero si el texto final está vacío o tiene menos de :data:`FILTER_MIN_WORDS` palabras.
    """
    limit = max_words or CRITERIA["summarizer"]["max_words"]
    fixed, info = postprocess_summary(report, source, limit, filter_numbers=True)
    words = count_words(fixed)
    return {
        "text": fixed,
        "mode": info["mode"],
        "words": words,
        "words_trimmed": info["words_trimmed"],
        "numbers": info["numbers_after"],
        "numbers_trimmed": info["numbers_trimmed"],
        "spanish": spanish_score(fixed),
        "removed": info["filter"]["removed"],
        "sentences": info["filter"]["sentences"],
        "failure": words < FILTER_MIN_WORDS,
    }


def eval_summarizer(args, rec: Recorder, work: Path, results: Path) -> dict:
    trial = trial_for("summarizer", args)
    reps = args.reps or REPS["summarizer"]
    processed = PROJECT_ROOT / "context" / "data" / "processed"
    cases = list(SUMMARY_INPUTS.items())
    if args.limit:
        cases = cases[: args.limit]
    minutes = getattr(args, "max_minutes", None)
    budget = minutes * 60 if minutes else None
    started = time.monotonic()
    skipped = 0
    for case, (metrics_name, info_name) in cases:
        root = make_copy(work / f"summarizer_{case}")
        folder = root / "context" / "data" / "processed"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(processed / metrics_name, folder / metrics_name)
        (folder / info_name).write_text(trim_run_info((processed / info_name).read_text(encoding="utf-8")), encoding="utf-8")
        paths = [f"context/data/processed/{metrics_name}", f"context/data/processed/{info_name}"]
        task, context = prepare_task("summarizer", paths, root, max_chars=SUMMARY_MAX_CHARS)
        for rep in range(1, reps + 1):
            if budget is not None and time.monotonic() - started > budget:
                skipped += 1
                continue
            before = snapshot(root)
            result = run_task("summarizer", task, root=root, log_dir=results / "logs", run_label=f"{trial.label}_{case}_{rep}_v{args.env_variant}", env_variant=args.env_variant, **trial.overrides)
            verification = verify("summarizer", result.report, source=context["source"]) if result.stop_reason in {"final", "length"} and result.report else None
            ok = bool(verification and verification.valid and result.stop_reason == "final")
            d = verification.details if verification else {}
            notes = f"palabras={d.get('words')} cifras={d.get('numbers', {}).get('supported')}/{d.get('numbers', {}).get('total')} faltan={d.get('numbers', {}).get('missing')} castellano={d.get('spanish', {}).get('ratio', 0):.2f}"
            if verification and verification.errors:
                notes += f" errores={verification.errors}"
            if not verification:
                notes += f" sin resumen (parada {result.stop_reason})"
            details = {"case": case, "words": d.get("words"), "numbers": d.get("numbers"), "spanish": d.get("spanish"), "input_chars": len(context["source"])}
            if verification:  # salida CORREGIDA por código (coma decimal y recorte); la cruda es la que juzgan los criterios
                details["corrected"] = summary_corrected_metrics(result.report, context["source"])
                fixed = details["corrected"]
                notes += f" | corregido: palabras={fixed['words']} cifras={fixed['numbers']['supported']}/{fixed['numbers']['total']} castellano={fixed['spanish']['ratio']:.2f} recorte={fixed['mode']}"
                details["filtered"] = summary_filtered_metrics(result.report, context["source"])
                filt = details["filtered"]
                notes += f" | filtro: frases quitadas={filt['removed']}/{filt['sentences']} palabras={filt['words']} cifras={filt['numbers']['supported']}/{filt['numbers']['total']}{' FALLO DE FORMATO (< 30 palabras)' if filt['failure'] else ''}"
            rec.add(_system_record("summarizer", case, rep, result, root, before, args.env_variant, ok, notes, details, None, trial))
    summary = _summarizer_summary(rec.runs)
    summary["notes"] = _summarizer_notes(rec.runs)
    if skipped:
        summary.setdefault("notes", []).append(f"PROTOCOLO INCOMPLETO: {skipped} ejecuciones sin hacer por el presupuesto de tiempo ({minutes:g} min).")
    return summary


def _summarizer_notes(runs: list[RunRecord]) -> list[str]:
    """Líneas de resumen: crudo / recortado / recortado + filtro (cifras respaldadas y palabras) y frases quitadas por el filtro."""
    filtered = [r.details["filtered"] for r in runs if r.details.get("filtered")]
    if not filtered:
        return []
    raw_n = [r.details.get("numbers") or {} for r in runs]
    raw_total, raw_ok = sum(n.get("total", 0) for n in raw_n), sum(n.get("supported", 0) for n in raw_n)
    trim_total, trim_ok = sum(f["numbers_trimmed"]["total"] for f in filtered), sum(f["numbers_trimmed"]["supported"] for f in filtered)
    fin_total, fin_ok = sum(f["numbers"]["total"] for f in filtered), sum(f["numbers"]["supported"] for f in filtered)

    def pct(ok: int, total: int) -> str:
        return f"{100 * ok / total:.1f} % ({ok}/{total})" if total else "sin cifras"

    removed = [f["removed"] for f in filtered]
    return [
        f"Cifras respaldadas: crudo {pct(raw_ok, raw_total)}; recortado {pct(trim_ok, trim_total)}; recortado + filtro {pct(fin_ok, fin_total)}.",
        f"Palabras: crudo {[r.details.get('words') for r in runs]}; recortado {[f['words_trimmed'] for f in filtered]}; recortado + filtro {[f['words'] for f in filtered]}.",
        f"Frases quitadas por el filtro: {sum(removed)} en total ({removed} por ejecución, de {[f['sentences'] for f in filtered]} frases tras el recorte); fallos de formato (< {FILTER_MIN_WORDS} palabras): {sum(f['failure'] for f in filtered)} de {len(filtered)}.",
        "Aviso: el filtro es código; el cumplimiento de las filas «[recortado + filtro]» depende del filtro y no del modelo.",
    ]


def _summarizer_summary(runs: list[RunRecord]) -> dict:
    """Resumen del resumidor: los criterios (fijados) se juzgan sobre el resumen CRUDO; se informa además, sin cambiar el
    umbral, del resumen corregido por código (coma decimal y recorte a ``max_words``)."""
    total = sum((r.details.get("numbers") or {}).get("total", 0) for r in runs)
    supported = sum((r.details.get("numbers") or {}).get("supported", 0) for r in runs)
    ratio = supported / total if total else 0.0
    words = [r.details.get("words") for r in runs]
    words_ok = bool(runs) and all(isinstance(w, int) and 0 < w <= CRITERIA["summarizer"]["max_words"] for w in words)
    spanish_ok = bool(runs) and all((r.details.get("spanish") or {}).get("is_spanish") for r in runs)
    fixed = [r.details["corrected"] for r in runs if r.details.get("corrected")]
    fixed_total = sum(f["numbers"]["total"] for f in fixed)
    fixed_supported = sum(f["numbers"]["supported"] for f in fixed)
    fixed_ratio = fixed_supported / fixed_total if fixed_total else 0.0
    fixed_words = [f["words"] for f in fixed]
    fixed_words_ok = bool(fixed) and all(0 < w <= CRITERIA["summarizer"]["max_words"] for w in fixed_words)
    fixed_spanish_ok = bool(fixed) and all(f["spanish"]["is_spanish"] for f in fixed)
    filtered = [r.details["filtered"] for r in runs if r.details.get("filtered")]
    filter_rows: list = []
    filter_metrics: dict = {}
    if filtered:
        f_total = sum(f["numbers"]["total"] for f in filtered)
        f_supported = sum(f["numbers"]["supported"] for f in filtered)
        f_ratio = f_supported / f_total if f_total else 0.0
        f_words = [f["words"] for f in filtered]
        failures = sum(1 for f in filtered if f["failure"])
        clean = failures == 0
        removed = [f["removed"] for f in filtered]
        filter_metrics = {
            "numbers_supported": f_supported, "numbers_total": f_total, "words": f_words, "failures": failures, "removed": removed,
            "sentences": [f["sentences"] for f in filtered], "numbers_trimmed": [f["numbers_trimmed"]["supported"] for f in filtered], "numbers_trimmed_total": [f["numbers_trimmed"]["total"] for f in filtered],
        }
        filter_rows = [
            ("[recortado + filtro] Cifras del resumen presentes en la entrada (agregado; sin fallos de formato)", ">= 95 %", f"{100 * f_ratio:.1f} % ({f_supported}/{f_total})", clean and f_total > 0 and f_ratio >= CRITERIA["summarizer"]["numbers_min"]),
            ("[recortado + filtro] Palabras por resumen (todas las ejecuciones; sin fallos de formato)", "<= 150", f"máx. {max(f_words)} {f_words}", clean and all(0 < w <= CRITERIA["summarizer"]["max_words"] for w in f_words)),
            ("[recortado + filtro] En castellano (todas las ejecuciones; sin fallos de formato)", "proporción >= 0,8", "; ".join(f"{f['spanish']['ratio']:.2f}" for f in filtered), clean and all(f["spanish"]["is_spanish"] for f in filtered)),
            (f"[recortado + filtro] Fallos de formato (resumen filtrado < {FILTER_MIN_WORDS} palabras)", "0", f"{failures} de {len(filtered)}; frases quitadas {removed}", clean),
        ]
    return {
        "metrics": {
            "numbers_supported": supported, "numbers_total": total, "words": words,
            "corrected": {"numbers_supported": fixed_supported, "numbers_total": fixed_total, "words": fixed_words, "modes": [f["mode"] for f in fixed]},
            **({"filtered": filter_metrics} if filtered else {}),
        },
        "criteria": [
            ("Cifras del resumen presentes en la entrada (agregado)", ">= 95 %", f"{100 * ratio:.1f} % ({supported}/{total})", total > 0 and ratio >= CRITERIA["summarizer"]["numbers_min"]),
            ("Palabras por resumen (todas las ejecuciones)", "<= 150", f"máx. {max((w for w in words if isinstance(w, int)), default='-')} {words}", words_ok),
            ("En castellano (todas las ejecuciones)", "proporción >= 0,8", "; ".join(f"{(r.details.get('spanish') or {}).get('ratio', 0):.2f}" for r in runs), spanish_ok),
            # misma medida sobre el resumen corregido por código (informativo; mismo umbral, no sustituye a las anteriores)
            ("[corregido por código] Cifras del resumen presentes en la entrada (agregado)", ">= 95 %", f"{100 * fixed_ratio:.1f} % ({fixed_supported}/{fixed_total})", fixed_total > 0 and fixed_ratio >= CRITERIA["summarizer"]["numbers_min"]),
            ("[corregido por código] Palabras por resumen (todas las ejecuciones)", "<= 150", f"máx. {max(fixed_words, default='-')} {fixed_words}", fixed_words_ok),
            ("[corregido por código] En castellano (todas las ejecuciones)", "proporción >= 0,8", "; ".join(f"{f['spanish']['ratio']:.2f}" for f in fixed), fixed_spanish_ok),
            *filter_rows,
        ],
    }


# -- retriever ---------------------------------------------------------------------------------------------------------
# (consulta, [(fichero, palabra clave que debe aparecer en el fragmento), ...]); basta con uno de los pares.
RETRIEVER_QUERIES: list[tuple[str, list[tuple[str, str]]]] = [
    ("¿qué criterio se revisó en la condicional por hidrofobicidad?", [("roadmap.md", "A.2")]),
    ("¿cómo se elige el backend por defecto?", [("tech_stack.md", "AMP_GAN_BACKEND"), ("runtime.py", "AMP_GAN_BACKEND")]),
    ("¿qué puede escribir el ejecutor?", [("executor.md", "Escritura"), ("governance.md", "escribe")]),
    ("¿qué rango de longitud tienen las secuencias de entrenamiento?", [("tech_stack.md", "5–60")]),
    ("¿qué semilla global se usa por defecto?", [("tech_stack.md", "Semilla global")]),
    ("¿de dónde proceden los datos de secuencias?", [("roadmap.md", "DRAMP")]),
    ("¿qué reglas siguen las ramas y los commits de código?", [("governance.md", "feat/")]),
    ("¿qué preguntas de investigación se plantean en el proyecto?", [("mission.md", "Preguntas de investigación")]),
    ("¿qué formato de salida debe devolver el agente localizador?", [("locator.md", "respuestas")]),
    ("¿qué modelo local usa el revisor de código?", [("code_reviewer.md", "qwen2.5-coder")]),
    ("¿cómo se calcula la carga neta de un péptido a un pH dado?", [("properties.py", "Henderson")]),
    ("¿qué escala de hidropatía se usa para calcular el GRAVY?", [("properties.py", "Kyte")]),
    ("¿qué formato tiene el encargo que el orquestador envía al ejecutor?", [("orchestrator.md", "Aceptación")]),
    ("¿cuántos tests tiene el módulo de limpieza de secuencias?", [("roadmap.md", "23 tests")]),
    ("¿qué hace el resumidor si le falta un dato?", [("summarizer.md", "falta un dato")]),
]


def retriever_rank(hits: list[dict], truth: list[tuple[str, str]], k: int = 3) -> int | None:
    """Mejor posición (desde 1) entre los ``k`` primeros que cumpla alguno de los pares (fichero, palabra clave)."""
    ranks = [r for r in (retrieval_rank(hits, f, kw, k) for f, kw in truth) if r is not None]
    return min(ranks) if ranks else None


def _rank_label(rank: int | None) -> str:
    return f"top{rank}" if rank else "fallo"


def eval_retriever(args, rec: Recorder, work: Path, results: Path, role: str = "retriever") -> dict:
    """Evalúa el recuperador de embeddings (``retriever``) o su variante híbrida (``retriever_hybrid``; BM25 + embeddings por RRF).

    Mismas 15 consultas, mismo criterio (acierto en el top-3 >= 80 %, entrada ``retriever`` de ``CRITERIA``). El rango que decide el
    criterio es el del recuperador evaluado; además se miden siempre BM25 y, en la variante híbrida, los embeddings solos (los mismos
    que daría ``retriever`` con ese modelo). Con ``--max-minutes`` se acota el tiempo total (indexado y consultas).
    """
    trial = trial_for(role, args)
    root = make_copy(work / f"{role}_copy")
    dropped = 0
    for pattern in args.drop or []:  # solo diagnóstico: quita de la COPIA los ficheros que cumplan el patrón
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink()
                dropped += 1
    chunks = chunk_project(root)
    queries = RETRIEVER_QUERIES[: args.limit] if args.limit else RETRIEVER_QUERIES
    unanswerable = [q for q, truth in queries if not any(any(c.ruta.endswith(f) and kw.lower() in c.texto.lower() for c in chunks) for f, kw in truth)]
    config = get_agent_config(role, **trial.overrides)
    hybrid = config.search_backend == "hybrid"
    backend = "hybrid" if hybrid else "embedding"
    minutes = getattr(args, "max_minutes", None)
    budget = minutes * 60 if minutes else None
    index_file = root / config.index_dir / "index.json"
    if index_file.exists():
        index_file.unlink()
    provider = make_provider(config)
    started = time.monotonic()
    before_index = snapshot(root)

    def check_budget(done: int, total: int) -> None:
        if budget is not None and time.monotonic() - started > budget:
            raise ProviderTimeout(f"el indexado superó el presupuesto de {minutes:g} min ({done}/{total} fragmentos)")

    index = build_index(chunks, provider, model=config.model, timeout=config.timeout, progress=check_budget)  # igual que run.retrieve
    save_index(index, resolve_index_dir(root, config.index_dir))
    build_s = time.monotonic() - started
    ps = ollama_ps(config.base_url, config.model)
    size_kb = round(index_file.stat().st_size / 1024) if index_file.exists() else None
    n_chunks = len(chunks)
    longest = max((len(c.texto) + len(c.ruta) + 1 for c in chunks), default=0)
    print(f"    índice ({config.model}): {n_chunks} fragmentos, {index['dim']} dimensiones, en {build_s:.1f}s ({size_kb} KB) {ps.get('processor', '')}", file=sys.stderr, flush=True)
    lexical = LexicalRetriever(chunks)
    query_prefix = prefixes_for(config.model)[1]
    per_query = []
    skipped = 0
    for number, (query, truth) in enumerate(queries):
        if budget is not None and time.monotonic() - started > budget:
            skipped = len(queries) - number
            print(f"    presupuesto de {minutes:g} min agotado: se omiten {skipped} consultas", file=sys.stderr, flush=True)
            break
        t1 = time.monotonic()
        main_hits = retrieve(query, root=root, k=3, backend=backend, agent=role, **trial.overrides)  # camino real: re-trocea, lee el índice, consulta
        main_s = time.monotonic() - t1
        dense_hits = retrieve(query, root=root, k=3, backend="embedding", agent=role, **trial.overrides) if hybrid else main_hits
        t2 = time.monotonic()
        provider.embed([f"{query_prefix}{query}"], timeout=config.timeout)  # solo la llamada al modelo (latencia de la consulta)
        embed_s = time.monotonic() - t2
        lex_hits = [h.as_dict() for h in lexical.search(query, 3)]
        r_main, r_dense, r_lex = retriever_rank(main_hits, truth), retriever_rank(dense_hits, truth), retriever_rank(lex_hits, truth)
        details = {
            "query": query,
            "truth": truth,
            "main_rank": r_main,
            "embedding_rank": r_dense,
            "hybrid_rank": r_main if hybrid else None,
            "lexical_rank": r_lex,
            "embed_only_s": round(embed_s, 3),
            "embedding_top": [(h["ruta"], h["linea"], round(h["puntuacion"], 3)) for h in dense_hits],
            "hybrid_top": [(h["ruta"], h["linea"], round(h["puntuacion"], 5)) for h in main_hits] if hybrid else None,
            "lexical_top": [(h["ruta"], h["linea"], round(h["puntuacion"], 3)) for h in lex_hits],
        }
        shown = f"híbrido={_rank_label(r_main)} embeddings={_rank_label(r_dense)}" if hybrid else f"embeddings={_rank_label(r_main)}"
        record = RunRecord(
            role=trial.label, model=config.model, case=f"q{number + 1:02d}", rep=1, success=r_main is not None, stop_reason="final", steps=1, tokens=None,
            prompt_tokens_last=None, seconds=round(main_s, 2), context_length=ps.get("context_length"), num_ctx_config=config.options.get("num_ctx"),
            processor=ps.get("processor", ""), cpu_offload=ps.get("cpu_offload"), notes=f"{shown} léxico={_rank_label(r_lex)} | {query}", details=details,
        )
        rec.add(record)
        per_query.append(details)
    n = len(per_query)
    main_n = sum(d["main_rank"] is not None for d in per_query)
    dense_n = sum(d["embedding_rank"] is not None for d in per_query)
    lex_hits_n = sum(d["lexical_rank"] is not None for d in per_query)
    main_top1 = sum(d["main_rank"] == 1 for d in per_query)
    dense_top1 = sum(d["embedding_rank"] == 1 for d in per_query)
    lex_top1 = sum(d["lexical_rank"] == 1 for d in per_query)
    both = sum(d["main_rank"] is not None and d["lexical_rank"] is not None for d in per_query)
    only_main = [d["query"] for d in per_query if d["main_rank"] is not None and d["lexical_rank"] is None]
    only_lex = [d["query"] for d in per_query if d["main_rank"] is None and d["lexical_rank"] is not None]
    numbered = {d["query"]: f"q{i + 1:02d}" for i, d in enumerate(per_query)}
    modified = snapshot(root) != before_index
    query_s = [r.seconds for r in rec.runs]
    embed_only = [d["embed_only_s"] for d in per_query]
    variant = "híbrido BM25 + embeddings" if hybrid else "embeddings"
    mean_query = round(sum(query_s) / len(query_s), 2) if query_s else 0.0
    mean_embed = round(sum(embed_only) / len(embed_only), 3) if embed_only else 0.0
    notes = [
        f"Top-3 de {n} consultas: {variant} ({config.model}) {main_n}/{n}, BM25 {lex_hits_n}/{n}" + (f", solo embeddings {dense_n}/{n}" if hybrid else "") + ". Referencias medidas antes: BM25 11/15 (73 %), nomic-embed-text 9/15 (60 %).",
        f"Frente a BM25: fallo -> acierto en {len(only_main)} consultas {[numbered[q] for q in only_main]}; acierto -> fallo en {len(only_lex)} {[numbered[q] for q in only_lex]}; aciertan los dos en {both}.",
        f"Top-1: {variant} {main_top1}/{n}, BM25 {lex_top1}/{n}" + (f", solo embeddings {dense_top1}/{n}" if hybrid else "") + ".",
        f"Tiempo: indexado {build_s:.1f} s ({n_chunks} fragmentos, {index['dim']} dimensiones, {size_kb} KB); por consulta {mean_query} s (llamada completa del recuperador: re-trocea, lee el índice, consulta y ordena) de los que {mean_embed} s son solo la llamada al modelo; procesador {ps.get('processor') or 'desconocido'}.",
        f"Contexto: el fragmento más largo (ruta incluida) tiene {longest} caracteres y num_ctx es {config.options.get('num_ctx')}.",
    ]
    if skipped:
        notes.append(f"PROTOCOLO INCOMPLETO: {skipped} consultas sin evaluar por el presupuesto de tiempo ({minutes:g} min).")
    return {
        "metrics": {
            "chunks": n_chunks, "dim": index["dim"], "index_build_s": round(build_s, 1), "index_kb": size_kb, "queries": n, "unanswerable_queries": unanswerable,
            "mode": backend, "main_top3": main_n, "main_top1": main_top1, "embedding_top3": dense_n if hybrid else main_n, "lexical_top3": lex_hits_n,
            "embedding_top1": dense_top1 if hybrid else main_top1, "lexical_top1": lex_top1,
            "dropped_files": dropped, "drop_patterns": args.drop or [], "both_top3": both, "only_embedding": only_main, "only_lexical": only_lex,
            "gained_vs_bm25": [numbered[q] for q in only_main], "lost_vs_bm25": [numbered[q] for q in only_lex],
            "query_s_mean": mean_query, "embed_only_s_mean": mean_embed, "longest_chunk_chars": longest, "skipped_queries": skipped,
            "processor": ps.get("processor"), "copy_changes_besides_index": [k for k in snapshot(root) if k not in before_index and not k.startswith(".agent_runs/")] if modified else [],
        },
        "notes": notes,
        "criteria": [
            (f"Acierto en el top-3 con {variant} ({config.model})", ">= 80 %", f"{100 * main_n / n:.0f} % ({main_n}/{n}); top-1: {main_top1}/{n}" if n else "-", n > 0 and main_n / n >= CRITERIA["retriever"]["top3_min"]),
            ("Línea base léxica (BM25), misma verdad (informativo)", "-", f"{100 * lex_hits_n / n:.0f} % ({lex_hits_n}/{n}); top-1: {lex_top1}/{n}" if n else "-", lex_hits_n >= main_n),
        ],
    }


# -- query_rewriter: reescritura de la consulta para BM25 ------------------------------------------------------------------
def corpus_hash(chunks) -> str:
    """Huella (12 primeros caracteres de SHA-256) de los fragmentos del corpus: ruta, línea y texto de cada uno, en orden."""
    return hashlib.sha256("\n".join(f"{c.ruta}:{c.linea}:{c.texto}" for c in chunks).encode("utf-8")).hexdigest()[:12]


def eval_query_rewriter(args, rec: Recorder, work: Path, results: Path, role: str = "query_rewriter") -> dict:
    """Evalúa el reescritor de consultas (``amp_agents.rewrite``) con las 15 preguntas del recuperador y BM25 simple en la misma ejecución.

    Se registra el hash del corpus (copia del proyecto) al empezar y el de una copia nueva del repositorio al terminar.
    ``--max-minutes`` limita el tiempo total: las preguntas que no se llegan a hacer se anotan y el protocolo se declara incompleto.
    """
    trial = trial_for(role, args)
    config = get_agent_config(role, **trial.overrides)
    root = make_copy(work / f"{role}_copy")
    chunks = chunk_project(root)
    hash_before = corpus_hash(chunks)
    queries = RETRIEVER_QUERIES[: args.limit] if args.limit else RETRIEVER_QUERIES
    unanswerable = [q for q, truth in queries if not any(any(c.ruta.endswith(f) and kw.lower() in c.texto.lower() for c in chunks) for f, kw in truth)]
    lexical = LexicalRetriever(chunks)
    provider = make_provider(config)
    minutes = getattr(args, "max_minutes", None)
    started = time.monotonic()
    budget = minutes * 60 if minutes else None
    per_query: list[dict] = []
    rewrites = []
    skipped = 0
    ps: dict = {}
    for number, (query, truth) in enumerate(queries):
        if budget is not None and time.monotonic() - started > budget:
            skipped = len(queries) - number
            break
        base_hits = [h.as_dict() for h in lexical.search(query, 3)]
        r_base = retriever_rank(base_hits, truth)
        rw = rewrite_query(provider, query, config.max_seconds, log_dir=results / "logs", label=f"{trial.label}_q{number + 1:02d}_v{args.env_variant}", root=root)
        new_hits = [h.as_dict() for h in lexical.search(rw.query, 3)]
        r_new = retriever_rank(new_hits, truth)
        if not ps:
            ps = ollama_ps(config.base_url, config.model)
        rewrites.append(rw)
        details = {
            "query": query, "truth": truth, "words": rw.words, "status": rw.status, "expanded": rw.query, "base_rank": r_base, "rewritten_rank": r_new,
            "base_top": [(h["ruta"], h["linea"], round(h["puntuacion"], 3)) for h in base_hits], "rewritten_top": [(h["ruta"], h["linea"], round(h["puntuacion"], 3)) for h in new_hits],
            "reply": rw.reply, "prompt_tokens": rw.prompt_tokens, "completion_tokens": rw.completion_tokens,
        }
        rec.add(
            RunRecord(
                role=trial.label, model=config.model, case=f"q{number + 1:02d}", rep=1, success=r_new is not None, stop_reason="final" if rw.status in {"ok", "fuera_de_rango"} else rw.status,
                steps=1, tokens=rw.prompt_tokens + rw.completion_tokens, prompt_tokens_last=rw.prompt_tokens or None, seconds=rw.seconds, context_length=ps.get("context_length"),
                num_ctx_config=config.options.get("num_ctx"), processor=ps.get("processor", ""), cpu_offload=ps.get("cpu_offload"),
                notes=f"BM25={_rank_label(r_base)} reescrito={_rank_label(r_new)} estado={rw.status} palabras={rw.words} | {query}", details=details,
            )
        )
        per_query.append(details)
    root_after = make_copy(work / f"{role}_copy_after")
    hash_after = corpus_hash(chunk_project(root_after))
    shutil.rmtree(root_after, ignore_errors=True)
    n = len(per_query)
    base_n = sum(d["base_rank"] is not None for d in per_query)
    new_n = sum(d["rewritten_rank"] is not None for d in per_query)
    gained = [(i, d) for i, d in enumerate(per_query) if d["base_rank"] is None and d["rewritten_rank"] is not None]
    lost = [(i, d) for i, d in enumerate(per_query) if d["base_rank"] is not None and d["rewritten_rank"] is None]
    moved = [(i, d) for i, d in enumerate(per_query) if d["base_rank"] is not None and d["rewritten_rank"] is not None and d["base_rank"] != d["rewritten_rank"]]
    failures = sum(r.format_failure for r in rewrites)
    request_failures = sum(r.status in {"timeout", "error"} for r in rewrites)
    seconds = [r.seconds for r in rewrites]
    mean_s = round(sum(seconds) / len(seconds), 2) if seconds else 0.0
    c = CRITERIA["query_rewriter"]

    def show(items) -> str:
        return "; ".join(f"q{i + 1:02d} {d['words']}" for i, d in items) or "ninguna"

    notes = [
        f"Top-3 de {n} consultas: BM25 simple {base_n}/{n}; con reescritura ({config.model}) {new_n}/{n}. Con 15 preguntas el resultado es solo indicativo.",
        f"Fallo -> acierto ({len(gained)}): {show(gained)}.",
        f"Acierto -> fallo ({len(lost)}): {show(lost)}.",
        f"Cambian de posición dentro del top-3 ({len(moved)}): {'; '.join(f'q{i + 1:02d} {d['base_rank']}->{d['rewritten_rank']}' for i, d in moved) or 'ninguna'}.",
        f"Fallos de formato (fuera de 3-8 palabras, vacía o sin JSON): {failures}; fallos de petición (tiempo o servidor): {request_failures}; {mean_s} s por petición; procesador {ps.get('processor') or 'desconocido'}.",
        f"Hash del corpus ({len(chunks)} fragmentos): al empezar {hash_before}, al terminar {hash_after} ({'sin cambios' if hash_before == hash_after else 'CAMBIÓ durante la ejecución'}).",
    ]
    if skipped:
        notes.append(f"PROTOCOLO INCOMPLETO: {skipped} consultas sin evaluar por el presupuesto de tiempo ({minutes:g} min).")
    return {
        "metrics": {
            "queries": n, "unanswerable_queries": unanswerable, "bm25_top3": base_n, "rewritten_top3": new_n, "gain": new_n - base_n,
            "gained": [f"q{i + 1:02d}" for i, _ in gained], "lost": [f"q{i + 1:02d}" for i, _ in lost], "format_failures": failures, "request_failures": request_failures,
            "seconds_per_request_mean": mean_s, "corpus_chunks": len(chunks), "corpus_hash_before": hash_before, "corpus_hash_after": hash_after,
            "corpus_unchanged": hash_before == hash_after, "skipped_queries": skipped, "processor": ps.get("processor"),
        },
        "notes": notes,
        "criteria": [
            (f"Top-3 con reescritura >= top-3 de BM25 simple + {c['gain_min']} consultas", f">= BM25 + {c['gain_min']}", f"{new_n}/{n} frente a BM25 {base_n}/{n} (diferencia {new_n - base_n:+d})", n > 0 and new_n - base_n >= c["gain_min"]),
            ("Consultas que pasan de acierto a fallo (máximo)", f"<= {c['regressions_max']}", f"{len(lost)} {[f'q{i + 1:02d}' for i, _ in lost]}", n > 0 and len(lost) <= c["regressions_max"]),
        ],
    }


# -- router: elegir el rol que atiende una petición ----------------------------------------------------------------------------
def eval_router(args, rec: Recorder, work: Path, results: Path, role: str = "router") -> dict:
    """Evalúa el enrutador (``amp_agents.router``) con las 48 peticiones etiquetadas de ``fixtures/router_requests.json``.

    Una petición por elemento (temperatura 0, una sola repetición). Con un proveedor de decisión (``--provider ollama_systemone``) se hace una
    pregunta ``choice`` y se elige la opción de mayor probabilidad. ``--max-minutes`` limita el tiempo total (las peticiones sin hacer
    cuentan como sin rol y el protocolo se declara incompleto).
    """
    trial = trial_for(role, args)
    config = get_agent_config(role, **trial.overrides)
    provider = make_provider(config)
    decision = isinstance(provider, SystemOneProvider)
    items = load_requests()
    if args.limit:
        items = items[: args.limit]
    minutes = getattr(args, "max_minutes", None)
    started = time.monotonic()
    budget = minutes * 60 if minutes else None
    routes = []
    ps: dict = {}
    for item in items:
        if budget is not None and time.monotonic() - started > budget:
            routes.append(Route(item["peticion"], None, "omitida"))
            continue
        if decision:
            route = route_request_decision(provider, item["peticion"], config.max_seconds)
        else:
            route = route_request(provider, item["peticion"], config.max_seconds, log_dir=results / "logs", label=f"{trial.label}_{item['id']}_v{args.env_variant}", root=work)
        if not ps:
            ps = ollama_ps(config.base_url, config.model)
        routes.append(route)
        ok = route.role == item["rol"]
        rec.add(
            RunRecord(
                role=trial.label, model=config.model, case=item["id"], rep=1, success=ok, stop_reason="final" if route.status == "ok" else route.status, steps=1,
                tokens=(route.prompt_tokens + route.completion_tokens) or None, prompt_tokens_last=route.prompt_tokens or None, seconds=route.seconds,
                context_length=ps.get("context_length"), num_ctx_config=config.options.get("num_ctx"), processor=ps.get("processor", ""), cpu_offload=ps.get("cpu_offload"),
                notes=f"verdad={item['rol']} elegido={route.role} [{route.status}] | {item['peticion'][:80]}",
                details={"truth": item["rol"], "chosen": route.role, "status": route.status, "request": item["peticion"], "reply": route.reply, "probabilities": route.probabilities, "motivo": route.motivo},
            )
        )
    truth = [i["rol"] for i in items]
    chosen = [r.role for r in routes]
    matrix = confusion_matrix(truth, chosen)
    metrics = role_metrics(matrix)
    asked = [r for r in routes if r.status != "omitida"]
    omitted = len(routes) - len(asked)
    format_failures = sum(r.status in {"formato", "vacia"} for r in routes)
    request_failures = sum(r.status in {"timeout", "error"} for r in routes)
    mean_s = round(sum(r.seconds for r in asked) / len(asked), 2) if asked else 0.0
    c = CRITERIA["router"]
    per_role = metrics["per_role"]
    present = [r for r in ROUTER_ROLES if per_role[r]["n"]]
    wrong = [f"{i['id']} ({i['rol']} -> {r.role or NO_ROLE})" for i, r in zip(items, routes) if r.role != i["rol"]]
    notes = [
        f"Exactitud {100 * metrics['accuracy']:.1f} % ({metrics['correct']}/{metrics['total']}); fallos de formato {format_failures}; fallos de petición (tiempo o servidor) {request_failures}; {mean_s} s por petición; procesador {ps.get('processor') or 'desconocido'}.",
        "Matriz de confusión (verdad -> rol elegido): "
        + " | ".join(f"{t}: " + ", ".join(f"{p} {n}" for p, n in matrix[t].items() if n) for t in ROUTER_ROLES if sum(matrix[t].values())) + ".",
        "Recall / precisión por rol: " + "; ".join(f"{r} {per_role[r]['correct']}/{per_role[r]['n']} / {'-' if per_role[r]['precision'] is None else f'{100 * per_role[r]['precision']:.0f} %'}" for r in present) + ".",
        f"Peticiones mal encaminadas ({len(wrong)}): {', '.join(wrong) or 'ninguna'}.",
        "Aviso: el conjunto lo ha escrito el mismo autor del experimento, a partir de las definiciones de los roles: posible sesgo de coherencia con ellas.",
    ]
    if omitted:
        notes.append(f"PROTOCOLO INCOMPLETO: {omitted} peticiones sin hacer por el presupuesto de tiempo ({minutes:g} min).")
    return {
        "metrics": {
            **metrics, "confusion": matrix, "confusion_table": matrix_table(matrix), "format_failures": format_failures, "request_failures": request_failures, "omitted": omitted,
            "seconds_per_request_mean": mean_s, "processor": ps.get("processor"), "decision_model": decision, "misrouted": wrong,
        },
        "notes": notes,
        "criteria": [
            ("Exactitud global", f">= {100 * c['accuracy_min']:.0f} %", f"{100 * metrics['accuracy']:.1f} % ({metrics['correct']}/{metrics['total']})", metrics["total"] > 0 and omitted == 0 and metrics["accuracy"] >= c["accuracy_min"]),
            ("Recall en cada rol (mínimo)", f">= {100 * c['recall_min']:.0f} %", "; ".join(f"{r} {per_role[r]['correct']}/{per_role[r]['n']}" for r in present) or "-", bool(present) and omitted == 0 and all(per_role[r]["recall"] >= c["recall_min"] for r in present)),
        ],
    }


# -- salida ------------------------------------------------------------------------------------------------------------
def runs_table(runs: list[RunRecord]) -> str:
    lines = [
        "| caso | rep | éxito | parada | pasos | tokens | segundos | contexto ef. | procesador | JSON (estricto) | notas |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        tokens = "-" if r.tokens is None else r.tokens
        strict = "-" if r.json_valid is None else f"{'sí' if r.json_valid else 'NO'} ({'sí' if r.json_strict else 'no'})"
        note = r.notes.replace("|", "/").replace("\n", " ")[:160]
        lines.append(f"| {r.case} | {r.rep} | {'sí' if r.success else 'NO'} | {r.stop_reason} | {r.steps} | {tokens} | {r.seconds} | {r.context_length or '-'} | {r.processor or '-'} | {strict} | {note} |")
    return "\n".join(lines)


def criteria_table(role: str, model: str, criteria: list) -> str:
    lines = ["| rol (modelo) | criterio declarado | umbral | resultado medido | ¿cumple? |", "|---|---|---|---|---|"]
    for name, threshold, measured, ok in criteria:
        lines.append(f"| {role} ({model}) | {name} | {threshold} | {measured} | {'SÍ' if ok else 'NO'} |")
    return "\n".join(lines)


EVALUATORS = {
    "locator": eval_locator,
    "doc_checker": eval_doc_checker,
    "doc_checker_ss": functools.partial(eval_doc_checker, role="doc_checker_ss"),
    "doc_checker_pairs": eval_doc_checker_pairs,
    "doc_checker_pairs_short": functools.partial(eval_doc_checker_pairs, role="doc_checker_pairs_short"),
    "doc_checker_pairs_json": functools.partial(eval_doc_checker_pairs, role="doc_checker_pairs_json"),
    "doc_checker_pairs_agree": functools.partial(eval_doc_checker_pairs, role="doc_checker_pairs_agree"),
    "code_reviewer": eval_code_reviewer,
    "code_reviewer_funcs": eval_code_reviewer_funcs,
    "summarizer": eval_summarizer,
    "retriever": eval_retriever,
    "retriever_hybrid": functools.partial(eval_retriever, role="retriever_hybrid"),
    "query_rewriter": eval_query_rewriter,
    "router": eval_router,
}


def evaluate_role(role: str, args) -> dict:
    """Ejecuta la evaluación de un rol, escribe JSON y Markdown en ``results`` y descarga el modelo.

    Con ``--model`` / ``--num-ctx`` (``args.model``, ``args.num_ctx``) se evalúa otro modelo (u otro contexto) en el mismo rol
    sin tocar ``agents.toml``; los resultados llevan el modelo en el nombre (``<rol>@<modelo>``). Con ``--model`` se comprueba
    antes de empezar que Ollama tiene el modelo y se hace una petición de calentamiento, registrada aparte (``warmup``) y que
    no cuenta como repetición.
    """
    work, results = Path(args.workdir), Path(args.results_dir)
    _ensure_outside_repo(work)
    _ensure_outside_repo(results)
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    trial = trial_for(role, args)
    config = get_agent_config(role, env_variant=args.env_variant, **trial.overrides)
    check_model_available(config.base_url, config.model)
    label = trial.label
    name = f"{label}_{args.tag}" if args.tag else label
    print(f"== {label} ({config.model}) variante de entorno {args.env_variant}, num_ctx={config.options.get('num_ctx')}", file=sys.stderr, flush=True)
    rec = Recorder(results, name)
    warmup = None
    started = time.monotonic()
    try:
        if trial.model:
            warmup = warm_up(config)
            print(f"    calentamiento: {warmup['seconds']}s {'ok' if warmup['ok'] else 'FALLO ' + warmup['error']} {warmup['ps'].get('processor', '')}", file=sys.stderr, flush=True)
            if not warmup["ok"]:
                raise SystemExit(f"error: el modelo {config.model} no respondió a la petición de calentamiento: {warmup['error']}")
        started = time.monotonic()  # el calentamiento no cuenta en el tiempo de la evaluación
        summary = EVALUATORS[role](args, rec, work, results)
    finally:
        stop_note = ollama_stop(config.model)
    elapsed = time.monotonic() - started
    out = {
        "role": label,
        "command_role": role,
        "model": config.model,
        "num_ctx": config.options.get("num_ctx"),
        "env_variant": args.env_variant,
        "tag": args.tag,
        "elapsed_s": round(elapsed, 1),
        "warmup": warmup,
        "ollama_ps_after": ollama_ps_text(),
        "ollama_stop": stop_note,
        "protocol": CRITERIA[base_role(role)],
        **summary,
        "runs": [asdict(r) for r in rec.runs],
    }
    out["criteria"] = [list(c) for c in summary["criteria"]]
    (results / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    warm_line = f"\nCalentamiento previo (no cuenta como repetición): {warmup['seconds']} s, {warmup['ps'].get('processor') or '-'}.\n" if warmup else ""
    notes_block = ("\n" + "\n".join(f"- {line}" for line in summary["notes"]) + "\n") if summary.get("notes") else ""
    markdown = f"# {label} ({config.model}), variante de entorno {args.env_variant}\n{warm_line}{notes_block}\n{criteria_table(label, config.model, summary['criteria'])}\n\n{runs_table(rec.runs)}\n"
    (results / f"{name}.md").write_text(markdown, encoding="utf-8")
    print(f"== {label}: {elapsed / 60:.1f} min\n{criteria_table(label, config.model, summary['criteria'])}", flush=True)
    for line in summary.get("notes", []):
        print(f"   {line}", flush=True)
    return out


def build_report(results_dir: Path) -> str:
    """Tabla Markdown conjunta de todos los ``<rol>[_etiqueta].json`` de ``results_dir``."""
    lines = ["| serie | rol (modelo) | criterio declarado | umbral | resultado medido | ¿cumple? |", "|---|---|---|---|---|---|"]
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("seeds_"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, threshold, measured, ok in data.get("criteria", []):
            lines.append(f"| {path.stem} (entorno v{data.get('env_variant', 1)}) | {data['role']} ({data['model']}) | {name} | {threshold} | {measured} | {'SÍ' if ok else 'NO'} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m amp_agents.eval_roles", description="Evaluación en vivo de los roles (no es un test de pytest).")
    parser.add_argument("command", choices=[*COMMAND_ROLES, "all", "report"])
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR), help="carpeta (fuera del repositorio) para las copias")
    parser.add_argument("--results-dir", dest="results_dir", help="carpeta de resultados (por defecto <workdir>/results)")
    parser.add_argument("--env-variant", dest="env_variant", type=int, default=1, help="variante de la sección «Entorno de ejecución» del prompt")
    parser.add_argument("--tag", default="", help="sufijo de los ficheros de resultados (p. ej. v2)")
    parser.add_argument("--model", help="evalúa este modelo de Ollama en el rol, sin tocar agents.toml (los resultados se llaman <rol>@<modelo>); hace un calentamiento previo aparte")
    parser.add_argument("--num-ctx", dest="num_ctx", type=int, help="fija options.num_ctx del rol evaluado (conserva el resto de opciones)")
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, help="tope de tokens de cada petición del rol evaluado (razonamiento incluido; num_predict en Ollama)")
    parser.add_argument("--provider", help="cambia el proveedor del rol evaluado (p. ej. ollama_systemone para un modelo de decisión como tev1)")
    parser.add_argument("--think", choices=["true", "false"], help="fija options.think del rol evaluado (razonamiento de los modelos que lo admiten)")
    parser.add_argument("--max-seconds", dest="max_seconds", type=float, help="tope de tiempo (s) de cada petición del rol evaluado")
    parser.add_argument("--max-minutes", dest="max_minutes", type=float, help="solo doc_checker_pairs*, code_reviewer_funcs, retriever*, query_rewriter y router: presupuesto de tiempo de la evaluación (no se empieza una repetición que no quepa)")
    parser.add_argument("--reps", type=int, help="solo para pruebas rápidas: sustituye las repeticiones del protocolo")
    parser.add_argument("--limit", type=int, help="solo para pruebas rápidas: primeros N casos")
    parser.add_argument("--drop", action="append", help="solo diagnóstico del retriever: quita de la copia los ficheros que cumplan este patrón (repetible)")
    parser.add_argument("--roles", default=",".join(ROLES), help="con 'all': roles separados por comas, en orden (por defecto sin el localizador, retirado; añádelo con --roles locator,...)")
    parser.add_argument("--deadline-min", dest="deadline_min", type=float, help="con 'all': no empezar otro rol pasado este tiempo")
    args = parser.parse_args(argv)
    args.results_dir = args.results_dir or str(Path(args.workdir) / "results")
    args.think = None if args.think is None else args.think == "true"
    if args.command in ("all", "report") and (args.model or args.num_ctx or args.max_tokens or args.max_seconds or args.provider or args.think is not None):
        parser.error("--model, --num-ctx, --max-tokens, --max-seconds, --provider y --think se aplican a un solo rol: indica el rol (no 'all' ni 'report')")
    if args.command == "report":
        table = build_report(Path(args.results_dir))
        print(table)
        (Path(args.results_dir) / "summary.md").write_text(table + "\n", encoding="utf-8")
        return 0
    if args.command != "all":
        evaluate_role(args.command, args)
        return 0
    started = time.monotonic()
    requested = [r.strip() for r in args.roles.split(",") if r.strip()]
    unknown = [r for r in requested if r not in COMMAND_ROLES]
    if unknown:
        parser.error(f"roles desconocidos: {unknown}; disponibles: {list(COMMAND_ROLES)}")
    for role in requested:
        if args.deadline_min and (time.monotonic() - started) / 60 > args.deadline_min:
            print(f"== presupuesto agotado: se omite {role} y los siguientes", file=sys.stderr, flush=True)
            break
        evaluate_role(role, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

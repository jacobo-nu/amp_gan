"""Evaluación EN VIVO de los cinco roles acotados (no es un test de pytest).

Uso (desde la carpeta ``src``, con Ollama en marcha y los modelos descargados)::

    python -m amp_agents.eval_roles locator       --workdir /ruta/a/carpeta/vacia
    python -m amp_agents.eval_roles doc_checker   --workdir ...
    python -m amp_agents.eval_roles code_reviewer --workdir ...
    python -m amp_agents.eval_roles summarizer    --workdir ...
    python -m amp_agents.eval_roles retriever     --workdir ...
    python -m amp_agents.eval_roles all           --workdir ... --deadline-min 90
    python -m amp_agents.eval_roles report        --results-dir ...

Todo se hace sobre COPIAS dentro de ``--workdir`` (``context/`` sin ``context/data/`` y ``src/``, sin este script ni
sus tests, para que las preguntas y sus respuestas no estén en el material consultado): las herramientas y la política
apuntan a la copia, nunca al repositorio real. Cada ejecución comprueba que la copia no cambió. Los resultados
(JSON, registros JSONL y tabla Markdown) van a ``--results-dir`` (por defecto ``<workdir>/results``). Por defecto
``--workdir`` es una carpeta temporal del sistema.

PROTOCOLO Y CRITERIOS (fijados de antemano; no se ajustan tras ver los resultados):

- locator (``llama3.2:3b``): 10 preguntas con verdad calculada por código (5 definiciones con ``ast``, 3 importaciones
  y 2 constantes con expresiones regulares), 2 repeticiones. Éxito de una respuesta: el conjunto de pares
  (ruta, línea) es EXACTAMENTE el esperado y cada línea citada existe. Criterio: >= 80 % de respuestas exactas.
- doc_checker (``phi4-mini``): copias de ``context/constitution/roadmap.md`` y ``TODO.md`` (esta como
  ``context/TODO.md``, porque el rol solo lee ``context/`` y ``src/``) con 6 incoherencias sembradas de tipos
  distintos, más una copia limpia, 3 repeticiones. Una incoherencia sembrada se recupera si alguna de las dos citas
  de un hallazgo es literal y cae (como mucho 2 líneas) sobre la línea alterada. Criterios: >= 90 % de citas
  literales válidas (todas las ejecuciones); >= 4 de las 6 sembradas recuperadas de media por repetición; <= 2
  falsos positivos (hallazgos en la copia limpia) en CADA repetición.
- code_reviewer (``qwen2.5-coder:7b``): copia de ``src/amp_gan/properties.py`` con 5 defectos sembrados de tipos
  distintos, más la copia limpia, 3 repeticiones. Un hallazgo cuenta si su línea está a +-2 de la sembrada.
  Criterios: >= 3 de 5 recuperados de media; >= 90 % de citas válidas (existe la línea y la cita es literal);
  <= 2 falsos positivos en CADA repetición de la copia limpia.
- summarizer (``gemma3:4b``): 2 entradas (métricas y ``run_info`` de ``cgan_hydro`` y de ``cgan_length``), 3
  repeticiones. Criterios: >= 95 % de las cifras del resumen presentes en la entrada (agregado); <= 150 palabras y
  en castellano (proporción de palabras funcionales castellanas >= 0,8, ver ``verifiers``) en TODAS las ejecuciones.
- retriever (``nomic-embed-text``): 15 consultas en castellano con verdad escrita a mano (fichero y palabra clave en
  el fragmento). Criterio: acierto en el top-3 >= 80 %. Se mide igual la línea base léxica (BM25).

Las ejecuciones fallidas (JSON inválido, tiempo agotado, error del servidor) cuentan como no exitosas y se anotan.
Por ejecución se registra: éxito, pasos, tokens, segundos, motivo de parada, contexto efectivo (``/api/ps``) y si el
modelo se pasó parcialmente a la CPU.
"""

from __future__ import annotations

import argparse
import ast
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

from .agents import PROJECT_ROOT, get_agent_config
from .loop import AgentResult
from .retrieval import LexicalRetriever, chunk_project
from .run import prepare_task, retrieve, run_task
from .smoke_live import make_copy as _smoke_copy
from .smoke_live import server_models, snapshot
from .verifiers import (
    check_doc_citations,
    check_review_citations,
    parse_json_output,
    retrieval_rank,
    verify,
)

ROLES = ("locator", "doc_checker", "code_reviewer", "summarizer", "retriever")
DEFAULT_WORKDIR = Path(tempfile.gettempdir()) / "amp_roles_eval"
EXCLUDED_FROM_COPY = ("src/amp_agents/eval_roles.py",)  # y src/tests/test_agents_eval*.py

CRITERIA = {
    "locator": {"exact_rate_min": 0.80},
    "doc_checker": {"valid_citations_min": 0.90, "recovered_mean_min": 4.0, "false_positives_max": 2},
    "code_reviewer": {"recovered_mean_min": 3.0, "valid_citations_min": 0.90, "false_positives_max": 2},
    "summarizer": {"numbers_min": 0.95, "max_words": 150},
    "retriever": {"top3_min": 0.80},
}
REPS = {"locator": 2, "doc_checker": 3, "code_reviewer": 3, "summarizer": 3}


# -- copias ------------------------------------------------------------------------------------------------
def make_copy(destination: Path, source: Path = PROJECT_ROOT) -> Path:
    """Copia ``src/`` y ``context/`` (sin ``context/data/``) y quita de la copia este script y sus tests."""
    _smoke_copy(destination, source)
    for rel in EXCLUDED_FROM_COPY:
        target = destination / rel
        if target.exists():
            target.unlink()
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


def _system_record(role: str, case: str, rep: int, result: AgentResult, root: Path, before: dict, env_variant: int, success: bool, notes: str, details: dict, verification=None) -> RunRecord:
    config = get_agent_config(role, env_variant=env_variant)
    ps = ollama_ps(config.base_url, config.model)
    modified = snapshot(root) != before
    return RunRecord(
        role=role,
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
def _py_files(root: Path) -> list[Path]:
    return sorted(p for p in (root / "src").rglob("*.py") if "__pycache__" not in p.parts and not any(part.startswith(".") for part in p.relative_to(root).parts))


def find_definitions(root: Path, name: str) -> list[tuple[str, int]]:
    """Pares (ruta, línea) de las definiciones ``def``/``class`` de ``name`` en los ``.py`` de ``src/`` (con ``ast``)."""
    out = []
    for path in _py_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                out.append((path.relative_to(root).as_posix(), node.lineno))
    return sorted(out)


def find_lines(root: Path, pattern: str) -> list[tuple[str, int]]:
    """Pares (ruta, línea) de las líneas de los ``.py`` de ``src/`` que cumplen la expresión regular ``pattern``."""
    regex = re.compile(pattern)
    out = []
    for path in _py_files(root):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if regex.search(line):
                out.append((path.relative_to(root).as_posix(), number))
    return sorted(out)


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
            result = run_task("locator", q["question"], root=root, log_dir=results / "logs", run_label=f"locator_{q['id']}_{rep}_v{args.env_variant}", env_variant=args.env_variant)
            ok, details = (grade_locator(result.report, q["expected"], root) if result.stop_reason == "final" else (False, {"errors": [f"parada: {result.stop_reason}"]}))
            details["expected"] = q["expected"]
            details["kind"] = q["kind"]
            note = f"esperadas={len(q['expected'])} aciertos={'sí' if ok else 'NO'} faltan={len(details.get('missing', []))} sobran={len(details.get('extra', []))} {details.get('errors', '')}"
            verification = verify("locator", result.report, root=root) if result.report else None
            rec.add(_system_record("locator", q["id"], rep, result, root, before, args.env_variant, ok, note, details, verification))
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


def eval_doc_checker(args, rec: Recorder, work: Path, results: Path) -> dict:
    real = PROJECT_ROOT
    roadmap_text = (real / "context/constitution/roadmap.md").read_text(encoding="utf-8")
    todo_text = (real / "TODO.md").read_text(encoding="utf-8")
    seeded_roadmap, seeded_todo, seeds = seed_documents(roadmap_text, todo_text)
    copies = {}
    for name, (road, todo) in {"seeded": (seeded_roadmap, seeded_todo), "clean": (roadmap_text, todo_text)}.items():
        root = make_copy(work / f"doc_checker_{name}")
        (root / ROADMAP_REL).write_text(road, encoding="utf-8")
        (root / TODO_REL).write_text(todo, encoding="utf-8")
        copies[name] = root
    (results / "seeds_doc_checker.json").write_text(json.dumps(seeds, ensure_ascii=False, indent=1), encoding="utf-8")
    reps = args.reps or REPS["doc_checker"]
    per_rep: list[dict] = []
    for rep in range(1, reps + 1):
        entry = {"rep": rep}
        for name in ("seeded", "clean"):
            root = copies[name]
            task, context = prepare_task("doc_checker", [ROADMAP_REL, TODO_REL], root, DOC_NOTE)
            before = snapshot(root)
            result = run_task("doc_checker", task, root=root, log_dir=results / "logs", run_label=f"doc_checker_{name}_{rep}_v{args.env_variant}", env_variant=args.env_variant)
            verification = verify("doc_checker", result.report, root=root) if result.stop_reason == "final" else None
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
                ok = valid_output and n_findings <= CRITERIA["doc_checker"]["false_positives_max"]
                notes = f"hallazgos_en_copia_limpia={n_findings}"
            if verification and verification.errors:
                notes += f" errores={verification.errors[:2]}"
            if not verification:
                notes += f" sin salida utilizable (parada {result.stop_reason})"
            details = {"copy": name, "match": match, "n_findings": n_findings, "citations": citations, "valid_output": valid_output}
            record = _system_record("doc_checker", name, rep, result, root, before, args.env_variant, ok, notes, details, verification)
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
    recovered = [e["seeded"]["recovered"] for e in per_rep]
    fps = [e["clean"]["false_positives"] for e in per_rep]
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
            result = run_task("code_reviewer", task, root=root, log_dir=results / "logs", run_label=f"code_reviewer_{name}_{rep}_v{args.env_variant}", env_variant=args.env_variant)
            verification = verify("code_reviewer", result.report, root=root, source=text) if result.stop_reason == "final" else None
            obj, _, _ = parse_json_output(result.report) if result.report else (None, False, None)
            valid_output = bool(verification and verification.details.get("schema_ok"))
            match = match_code_findings(obj, seeds if name == "seeded" else []) if valid_output else {"recovered": [], "false_positives": [], "n_findings": 0}
            citations = verification.details.get("citations", []) if verification else []
            if name == "seeded":
                ok = valid_output and len(match["recovered"]) >= 1
                notes = f"hallazgos={match['n_findings']} recuperados={len(match['recovered'])}/5 {match['recovered']} sin_defecto_cerca={len(match['false_positives'])}"
            else:
                ok = valid_output and match["n_findings"] <= CRITERIA["code_reviewer"]["false_positives_max"]
                notes = f"hallazgos_en_copia_limpia={match['n_findings']}"
            if not valid_output:
                notes += f" salida no utilizable: {(verification.errors[:2] if verification else 'parada ' + result.stop_reason)}"
            details = {"copy": name, "match": match, "citations": citations, "valid_output": valid_output, "seeds": seeds if name == "seeded" else []}
            rec.add(_system_record("code_reviewer", name, rep, result, root, before, args.env_variant, ok, notes, details, verification))
            entry[name] = {"recovered": len(match["recovered"]), "findings": match["n_findings"], "valid_output": valid_output}
        per_rep.append(entry)
    return _code_summary(rec.runs, per_rep)


def _code_summary(runs: list[RunRecord], per_rep: list[dict]) -> dict:
    valid, total = _citation_stats(runs)
    rate = valid / total if total else 0.0
    recovered = [e["seeded"]["recovered"] for e in per_rep]
    fps = [e["clean"]["findings"] for e in per_rep]
    mean_rec = sum(recovered) / len(recovered) if recovered else 0.0
    c = CRITERIA["code_reviewer"]
    return {
        "metrics": {"valid_citations": valid, "citations": total, "recovered_per_rep": recovered, "false_positives_clean_per_rep": fps, "per_rep": per_rep},
        "criteria": [
            ("Defectos sembrados recuperados (media por repetición, de 5)", ">= 3", f"{mean_rec:.2f} {recovered}", mean_rec >= c["recovered_mean_min"]),
            ("Citas válidas (línea existente y cita literal; todas las ejecuciones)", ">= 90 %", f"{100 * rate:.0f} % ({valid}/{total})", total > 0 and rate >= c["valid_citations_min"]),
            ("Falsos positivos en la copia limpia (máximo por repetición)", "<= 2", f"{max(fps) if fps else '-'} {fps}", bool(fps) and max(fps) <= c["false_positives_max"]),
        ],
    }


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


def eval_summarizer(args, rec: Recorder, work: Path, results: Path) -> dict:
    reps = args.reps or REPS["summarizer"]
    processed = PROJECT_ROOT / "context" / "data" / "processed"
    cases = list(SUMMARY_INPUTS.items())
    if args.limit:
        cases = cases[: args.limit]
    for case, (metrics_name, info_name) in cases:
        root = make_copy(work / f"summarizer_{case}")
        folder = root / "context" / "data" / "processed"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(processed / metrics_name, folder / metrics_name)
        (folder / info_name).write_text(trim_run_info((processed / info_name).read_text(encoding="utf-8")), encoding="utf-8")
        paths = [f"context/data/processed/{metrics_name}", f"context/data/processed/{info_name}"]
        task, context = prepare_task("summarizer", paths, root, max_chars=SUMMARY_MAX_CHARS)
        for rep in range(1, reps + 1):
            before = snapshot(root)
            result = run_task("summarizer", task, root=root, log_dir=results / "logs", run_label=f"summarizer_{case}_{rep}_v{args.env_variant}", env_variant=args.env_variant)
            verification = verify("summarizer", result.report, source=context["source"]) if result.stop_reason in {"final", "length"} and result.report else None
            ok = bool(verification and verification.valid and result.stop_reason == "final")
            d = verification.details if verification else {}
            notes = f"palabras={d.get('words')} cifras={d.get('numbers', {}).get('supported')}/{d.get('numbers', {}).get('total')} faltan={d.get('numbers', {}).get('missing')} castellano={d.get('spanish', {}).get('ratio', 0):.2f}"
            if verification and verification.errors:
                notes += f" errores={verification.errors}"
            if not verification:
                notes += f" sin resumen (parada {result.stop_reason})"
            details = {"case": case, "words": d.get("words"), "numbers": d.get("numbers"), "spanish": d.get("spanish"), "input_chars": len(context["source"])}
            rec.add(_system_record("summarizer", case, rep, result, root, before, args.env_variant, ok, notes, details, None))
    runs = rec.runs
    total = sum((r.details.get("numbers") or {}).get("total", 0) for r in runs)
    supported = sum((r.details.get("numbers") or {}).get("supported", 0) for r in runs)
    ratio = supported / total if total else 0.0
    words = [r.details.get("words") for r in runs]
    words_ok = bool(runs) and all(isinstance(w, int) and 0 < w <= CRITERIA["summarizer"]["max_words"] for w in words)
    spanish_ok = bool(runs) and all((r.details.get("spanish") or {}).get("is_spanish") for r in runs)
    return {
        "metrics": {"numbers_supported": supported, "numbers_total": total, "words": words},
        "criteria": [
            ("Cifras del resumen presentes en la entrada (agregado)", ">= 95 %", f"{100 * ratio:.1f} % ({supported}/{total})", total > 0 and ratio >= CRITERIA["summarizer"]["numbers_min"]),
            ("Palabras por resumen (todas las ejecuciones)", "<= 150", f"máx. {max((w for w in words if isinstance(w, int)), default='-')} {words}", words_ok),
            ("En castellano (todas las ejecuciones)", "proporción >= 0,8", "; ".join(f"{(r.details.get('spanish') or {}).get('ratio', 0):.2f}" for r in runs), spanish_ok),
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


def eval_retriever(args, rec: Recorder, work: Path, results: Path) -> dict:
    root = make_copy(work / "retriever_copy")
    dropped = 0
    for pattern in args.drop or []:  # solo diagnóstico: quita de la COPIA los ficheros que cumplan el patrón
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink()
                dropped += 1
    chunks = chunk_project(root)
    queries = RETRIEVER_QUERIES[: args.limit] if args.limit else RETRIEVER_QUERIES
    unanswerable = [q for q, truth in queries if not any(any(c.ruta.endswith(f) and kw.lower() in c.texto.lower() for c in chunks) for f, kw in truth)]
    config = get_agent_config("retriever")
    index_file = root / config.index_dir / "index.json"
    if index_file.exists():
        index_file.unlink()
    t0 = time.monotonic()
    before_index = snapshot(root)
    first = retrieve(queries[0][0], root=root, k=3, build=True)  # construye y guarda el índice
    build_s = time.monotonic() - t0
    ps = ollama_ps(config.base_url, config.model)
    size_kb = round(index_file.stat().st_size / 1024) if index_file.exists() else None
    n_chunks = len(chunks)
    print(f"    índice: {n_chunks} fragmentos en {build_s:.1f}s ({size_kb} KB) {ps.get('processor', '')}", file=sys.stderr, flush=True)
    lexical = LexicalRetriever(chunks)
    per_query = []
    for number, (query, truth) in enumerate(queries):
        t1 = time.monotonic()
        emb_hits = first if number == 0 else retrieve(query, root=root, k=3)
        emb_s = time.monotonic() - t1
        lex_hits = [h.as_dict() for h in lexical.search(query, 3)]
        r_emb, r_lex = retriever_rank(emb_hits, truth), retriever_rank(lex_hits, truth)
        details = {
            "query": query,
            "truth": truth,
            "embedding_rank": r_emb,
            "lexical_rank": r_lex,
            "embedding_top": [(h["ruta"], h["linea"], round(h["puntuacion"], 3)) for h in emb_hits],
            "lexical_top": [(h["ruta"], h["linea"], round(h["puntuacion"], 3)) for h in lex_hits],
        }
        record = RunRecord(
            role="retriever", model=config.model, case=f"q{number + 1:02d}", rep=1, success=r_emb is not None, stop_reason="final", steps=1, tokens=None,
            prompt_tokens_last=None, seconds=round(emb_s, 2), context_length=ps.get("context_length"), num_ctx_config=config.options.get("num_ctx"),
            processor=ps.get("processor", ""), cpu_offload=ps.get("cpu_offload"), notes=f"embeddings={'top' + str(r_emb) if r_emb else 'fallo'} léxico={'top' + str(r_lex) if r_lex else 'fallo'} | {query}", details=details,
        )
        rec.add(record)
        per_query.append(details)
    n = len(per_query)
    emb_hits_n = sum(d["embedding_rank"] is not None for d in per_query)
    lex_hits_n = sum(d["lexical_rank"] is not None for d in per_query)
    emb_top1 = sum(d["embedding_rank"] == 1 for d in per_query)
    lex_top1 = sum(d["lexical_rank"] == 1 for d in per_query)
    both = sum(d["embedding_rank"] is not None and d["lexical_rank"] is not None for d in per_query)
    only_emb = [d["query"] for d in per_query if d["embedding_rank"] is not None and d["lexical_rank"] is None]
    only_lex = [d["query"] for d in per_query if d["embedding_rank"] is None and d["lexical_rank"] is not None]
    modified = snapshot(root) != before_index
    return {
        "metrics": {
            "chunks": n_chunks, "index_build_s": round(build_s, 1), "index_kb": size_kb, "queries": n, "unanswerable_queries": unanswerable,
            "embedding_top3": emb_hits_n, "lexical_top3": lex_hits_n, "embedding_top1": emb_top1, "lexical_top1": lex_top1,
            "dropped_files": dropped, "drop_patterns": args.drop or [], "both_top3": both, "only_embedding": only_emb, "only_lexical": only_lex, "processor": ps.get("processor"), "copy_changes_besides_index": [k for k in snapshot(root) if k not in before_index and not k.startswith(".agent_runs/")] if modified else [],
        },
        "criteria": [
            ("Acierto en el top-3 con embeddings (nomic-embed-text)", ">= 80 %", f"{100 * emb_hits_n / n:.0f} % ({emb_hits_n}/{n}); top-1: {emb_top1}/{n}", emb_hits_n / n >= CRITERIA["retriever"]["top3_min"]),
            ("Línea base léxica (BM25), misma verdad (informativo)", "-", f"{100 * lex_hits_n / n:.0f} % ({lex_hits_n}/{n}); top-1: {lex_top1}/{n}", lex_hits_n >= emb_hits_n),
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
    "code_reviewer": eval_code_reviewer,
    "summarizer": eval_summarizer,
    "retriever": eval_retriever,
}


def evaluate_role(role: str, args) -> dict:
    """Ejecuta la evaluación de un rol, escribe JSON y Markdown en ``results`` y descarga el modelo."""
    work, results = Path(args.workdir), Path(args.results_dir)
    _ensure_outside_repo(work)
    _ensure_outside_repo(results)
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    config = get_agent_config(role, env_variant=args.env_variant)
    available = server_models(config.base_url)
    if available is None:
        raise SystemExit(f"error: el servidor no responde en {config.base_url} (¿está Ollama en marcha?)")
    if config.model not in available and f"{config.model}:latest" not in available:
        raise SystemExit(f"error: el modelo {config.model} no está disponible en el servidor: {available}")
    name = f"{role}_{args.tag}" if args.tag else role
    print(f"== {role} ({config.model}) variante de entorno {args.env_variant}", file=sys.stderr, flush=True)
    rec = Recorder(results, name)
    started = time.monotonic()
    try:
        summary = EVALUATORS[role](args, rec, work, results)
    finally:
        stop_note = ollama_stop(config.model)
    elapsed = time.monotonic() - started
    out = {
        "role": role,
        "model": config.model,
        "env_variant": args.env_variant,
        "tag": args.tag,
        "elapsed_s": round(elapsed, 1),
        "ollama_ps_after": ollama_ps_text(),
        "ollama_stop": stop_note,
        "protocol": CRITERIA[role],
        **summary,
        "runs": [asdict(r) for r in rec.runs],
    }
    out["criteria"] = [list(c) for c in summary["criteria"]]
    (results / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    markdown = f"# {role} ({config.model}), variante de entorno {args.env_variant}\n\n{criteria_table(role, config.model, summary['criteria'])}\n\n{runs_table(rec.runs)}\n"
    (results / f"{name}.md").write_text(markdown, encoding="utf-8")
    print(f"== {role}: {elapsed / 60:.1f} min\n{criteria_table(role, config.model, summary['criteria'])}", flush=True)
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
    parser.add_argument("command", choices=[*ROLES, "all", "report"])
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR), help="carpeta (fuera del repositorio) para las copias")
    parser.add_argument("--results-dir", dest="results_dir", help="carpeta de resultados (por defecto <workdir>/results)")
    parser.add_argument("--env-variant", dest="env_variant", type=int, default=1, help="variante de la sección «Entorno de ejecución» del prompt")
    parser.add_argument("--tag", default="", help="sufijo de los ficheros de resultados (p. ej. v2)")
    parser.add_argument("--reps", type=int, help="solo para pruebas rápidas: sustituye las repeticiones del protocolo")
    parser.add_argument("--limit", type=int, help="solo para pruebas rápidas: primeros N casos")
    parser.add_argument("--drop", action="append", help="solo diagnóstico del retriever: quita de la copia los ficheros que cumplan este patrón (repetible)")
    parser.add_argument("--roles", default=",".join(ROLES), help="con 'all': roles separados por comas, en orden")
    parser.add_argument("--deadline-min", dest="deadline_min", type=float, help="con 'all': no empezar otro rol pasado este tiempo")
    args = parser.parse_args(argv)
    args.results_dir = args.results_dir or str(Path(args.workdir) / "results")
    if args.command == "report":
        table = build_report(Path(args.results_dir))
        print(table)
        (Path(args.results_dir) / "summary.md").write_text(table + "\n", encoding="utf-8")
        return 0
    if args.command != "all":
        evaluate_role(args.command, args)
        return 0
    started = time.monotonic()
    for role in [r.strip() for r in args.roles.split(",") if r.strip()]:
        if args.deadline_min and (time.monotonic() - started) / 60 > args.deadline_min:
            print(f"== presupuesto agotado: se omite {role} y los siguientes", file=sys.stderr, flush=True)
            break
        evaluate_role(role, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

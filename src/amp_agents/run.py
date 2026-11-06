"""Línea de órdenes de los agentes con modelo intercambiable.

Uso (desde la carpeta ``src``)::

    python -m amp_agents.run --agent executor --task-file encargo.md
    python -m amp_agents.run --agent executor --task-file encargo.md --model qwen2.5:14b
    python -m amp_agents.run --agent locator "¿Dónde se define la función hydrophobicity?"
    python -m amp_agents.run --agent doc_checker context/constitution/roadmap.md TODO.md --note "cifras y fechas"
    python -m amp_agents.run --agent doc_checker_ss context/constitution/roadmap.md TODO.md --note "cifras y fechas"   # en una pasada, sin herramientas
    python -m amp_agents.run --agent code_reviewer src/amp_gan/properties.py
    python -m amp_agents.run --agent summarizer context/data/processed/cgan_hydro_metrics.csv context/data/processed/cgan_hydro_run_info.json
    python -m amp_agents.run --agent retriever "¿cómo se elige el backend por defecto?" --k 3 [--embedding|--hybrid [--build-index]]

El primer argumento posicional depende del rol: ``locator`` y ``retriever`` reciben la pregunta; ``doc_checker`` dos
o más rutas de documentos; ``code_reviewer`` la ruta de un fichero de ``src/``; ``summarizer`` las rutas de los
ficheros de resultados, cuyo texto se incluye en el mensaje (recortado a ``--max-chars`` por fichero);
``executor`` usa ``--task-file`` (que también vale para los demás roles y sustituye al mensaje construido).
Las rutas son relativas a ``--root``. ``--note`` añade el aspecto a comprobar (``doc_checker``) o la descripción de lo
que debe hacer el código (``code_reviewer``).

El prompt de sistema se ensambla desde ``context/agents/<agente>.md`` y ``context/constitution/governance.md`` de
``--root``; el modelo, la ``base_url``, el modo, las herramientas permitidas y los permisos salen de ``agents.toml``
(``--config``) salvo que se sobrescriban aquí. Los roles con verificador (``verifier`` en la configuración)
comprueban por código la salida (JSON, citas, líneas, cifras) y lo indican en la salida de error.

Opciones:
    --agent        nombre del agente (sección ``[agents.<nombre>]`` y ``context/agents/<nombre>.md``)
    --task-file    fichero con el encargo (obligatorio para ``executor``)
    --model, --base-url, --provider, --max-steps   sobrescriben la configuración
                   (``--provider ollama_openai|openai_compat|llamacpp|vllm`` para endpoints compatibles con OpenAI)
    --root         raíz del proyecto sobre la que actúan las herramientas (por defecto, el proyecto actual);
                   úsese una copia para pruebas
    --log-dir      carpeta de registros JSONL (por defecto ``<root>/.agent_runs``)
    --config       otro ``agents.toml``
    --raw          resumidor y revisor de código: muestra la salida del modelo SIN corregir. Por defecto el resumen pasa por
                   ``postprocess_summary`` (coma decimal, recorte a 150 palabras y, con ``--filter-numbers``, filtro de las frases con cifras
                   no respaldadas) y el informe del revisor por ``correct_review_lines`` (línea de cada hallazgo = la de su cita literal);
                   la salida de error dice qué se corrigió
    --k, --lexical, --embedding, --hybrid, --build-index   solo ``retriever`` (nº de resultados; BM25 léxico, que es lo que se
                   usa por defecto según ``search_backend``; embeddings, que exigen el servidor y el índice; fusión híbrida BM25 +
                   embeddings por rango recíproco, que también los exige; reconstruir el índice)

Código de salida: 0 si el motivo de parada es ``final`` (y la verificación, si la hay, es válida); 1 si no; 2 por error de
configuración o de uso. El informe final se escribe en la salida estándar y un resumen (pasos, tokens, tiempo, registro)
en la salida de error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .agents import (
    PROJECT_ROOT,
    ConfigError,
    build_system_prompt,
    get_agent_config,
    make_policy,
    make_provider,
    make_tools,
)
from .loop import AgentResult, run_agent, run_single_shot
from .providers import ChatProvider, ProviderError, ProviderTimeout
from .retrieval import (
    EmbeddingRetriever,
    HybridRetriever,
    LexicalRetriever,
    build_index,
    chunk_project,
    index_matches,
    load_index,
    resolve_index_dir,
    save_index,
    search,
)
from .verifiers import FILTER_MIN_WORDS, Verification, correct_review_lines, count_words, decimal_comma, parse_json_output, postprocess_summary, verify

# Si el filtro de cifras no respaldadas se aplica por defecto al resumen de la CLI. REGLA fijada el 2026-11-06 antes de medirlo: solo si el
# resumidor configurado (gemma3:4b) cumple los tres criterios de ``CRITERIA["summarizer"]`` con recorte + filtro y sin fallos de formato
# en la evaluación ``p4``. Siempre se puede forzar con ``--filter-numbers`` / ``--no-filter-numbers``.
FILTER_NUMBERS_BY_DEFAULT = True  # evaluación p4 (2026-11-06): gemma3:4b cumplió los tres criterios con recorte + filtro y 0 fallos de formato en 10 ejecuciones
SUMMARY_MAX_WORDS = 150

TASK_SECTIONS = ("Tarea", "Contexto", "Entregable", "Aceptación", "Restricciones")


def read_task(path: str | Path) -> str:
    """Lee el encargo; avisa por la salida de error si faltan secciones del formato (no es un error)."""
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"no existe el fichero de encargo {file}")
    text = file.read_text(encoding="utf-8")
    if not text.strip():
        raise ConfigError(f"el fichero de encargo {file} está vacío")
    missing = [s for s in TASK_SECTIONS if s.lower() + ":" not in text.lower()]
    if missing:
        print(f"aviso: al encargo le faltan secciones del formato: {', '.join(missing)}", file=sys.stderr)
    return text


def run_task(
    agent: str,
    task: str,
    root: str | Path | None = None,
    config_path: str | Path | None = None,
    log_dir: str | Path | None = None,
    provider_instance: ChatProvider | None = None,
    run_label: str | None = None,
    **overrides,
) -> AgentResult:
    """Ensambla todo desde la configuración y ejecuta el bucle (reutilizable desde otros scripts).

    ``overrides`` admite ``model``, ``base_url``, ``max_steps``, ``max_seconds``, ``options``...
    Si se pasa ``provider_instance`` (p. ej. un ``ScriptedProvider``) se usa en lugar del de la configuración;
    ``provider=...`` en ``overrides`` cambia el tipo de proveedor de la configuración.
    """
    config = get_agent_config(agent, config_path, **overrides)
    base = Path(root).resolve() if root else PROJECT_ROOT
    if config.mode == "embedding":
        raise ConfigError(f"el agente {agent!r} es de embeddings: usa retrieve() o `--agent {agent} \"consulta\"`")
    prompt = build_system_prompt(agent, base, config)
    provider = provider_instance or make_provider(config)
    label = run_label or f"{agent}_{config.model}"
    metadata = {"agent": agent, "model": config.model, "base_url": config.base_url, "mode": config.mode}
    if config.mode == "single_shot":
        return run_single_shot(provider, prompt, task, max_seconds=config.max_seconds, log_dir=log_dir, run_label=label, root=base, metadata=metadata)
    return run_agent(
        provider,
        prompt,
        task,
        make_tools(config, config_path),
        make_policy(config, base),
        max_steps=config.max_steps,
        max_seconds=config.max_seconds,
        log_dir=log_dir,
        run_label=label,
        metadata=metadata,
    )


def retrieve(
    query: str,
    root: str | Path | None = None,
    config_path: str | Path | None = None,
    k: int = 3,
    lexical: bool | None = None,
    build: bool = False,
    embedder=None,
    backend: str | None = None,
    agent: str = "retriever",
    **overrides,
) -> list[dict]:
    """Rol ``retriever``: los ``k`` fragmentos más cercanos a ``query`` (``ruta``, ``linea``, ``puntuacion``, ``texto``).

    Por defecto usa el recuperador léxico BM25 (sin modelo ni índice), el que fija ``search_backend`` en la
    configuración del rol. Los embeddings son una opción explícita (``lexical=False``, ``backend="embedding"`` o
    ``search_backend = "embedding"``) y también la fusión híbrida BM25 + embeddings (``backend="hybrid"`` o ``search_backend =
    "hybrid"``); entonces se usa el índice de ``index_dir``, que se construye (y guarda) si no existe, si ``build=True`` o si se
    construyó con otro modelo (un índice no se reutiliza entre modelos). ``backend`` (si no es ``None``) manda sobre ``lexical``,
    y ``lexical`` (si no es ``None``) sobre la configuración. ``embedder`` permite inyectar un proveedor simulado. ``agent`` es
    el rol de embeddings cuya configuración se usa (``retriever`` o una variante como ``retriever_hybrid``).
    """
    config = get_agent_config(agent, config_path, **overrides)
    base = Path(root).resolve() if root else PROJECT_ROOT
    if backend is None:
        backend = config.search_backend if lexical is None else ("lexical" if lexical else "embedding")
    if backend not in {"lexical", "embedding", "hybrid"}:
        raise ConfigError(f"backend de búsqueda desconocido: {backend!r}")
    chunks = chunk_project(base)
    if backend == "lexical":
        return search(LexicalRetriever(chunks), query, k)
    provider = embedder or make_provider(config)
    folder = resolve_index_dir(base, config.index_dir)
    index = None if build else load_index(folder)
    if index is not None and not index_matches(index, config.model):
        index = None  # índice de otro modelo: se reconstruye (y se sobrescribe) en lugar de mezclar espacios de embeddings
    if index is None:
        started = time.monotonic()

        def check_time(done: int, total: int) -> None:
            if time.monotonic() - started > config.max_seconds:
                raise ProviderTimeout(f"la construcción del índice superó {config.max_seconds:g} s ({done}/{total} fragmentos)")

        index = build_index(chunks, provider, model=config.model, timeout=config.timeout, progress=check_time)
        save_index(index, folder)
    dense = EmbeddingRetriever(index, provider, timeout=config.timeout)
    return search(dense if backend == "embedding" else HybridRetriever(LexicalRetriever(chunks), dense), query, k)


def number_lines(text: str) -> str:
    """Texto con cada línea precedida de su número (desde 1) y un tabulador, como la salida de ``read_file``."""
    return "\n".join(f"{number}\t{line}" for number, line in enumerate(text.splitlines(), start=1))


def doc_checker_single_shot_task(documents: list[tuple[str, str]], note: str = "") -> str:
    """Mensaje de usuario de ``doc_checker`` en una pasada: cada documento ``(ruta, texto)`` entero y con numeración de línea."""
    aspect = f" Aspecto a comprobar: {note}." if note else ""
    listing = ", ".join(f"`{path}`" for path, _ in documents)
    blocks = "\n\n".join(f"### {path}\n```\n{number_lines(text)}\n```" for path, text in documents)
    return (
        f"Compara estos documentos y señala sus incoherencias: {listing}.{aspect}\n"
        "Cada línea va precedida de su número y un tabulador; el número no forma parte del texto y no se copia en las "
        "citas. Responde solo con el objeto JSON.\n\n" + blocks
    )


def prepare_task(agent: str, inputs: list[str], root: Path, note: str = "", max_chars: int = 8000, config_path: str | Path | None = None) -> tuple[str, dict]:
    """Construye el mensaje de usuario de un rol a partir de sus argumentos; devuelve ``(mensaje, contexto)``.

    El contexto (``source``, ``reviewed``) lo usan los verificadores. Los ficheros se leen con la política de
    lectura del rol. ``ConfigError`` si faltan argumentos o no se puede leer algún fichero.
    """
    config = get_agent_config(agent, config_path)
    if not inputs:
        raise ConfigError(f"el agente {agent!r} necesita argumentos (ver la ayuda: --help)")
    policy = make_policy(config, root)

    def read(path: str) -> str:
        try:
            resolved = policy.check(path, "read")
        except Exception as exc:  # PolicyError / ToolError: mensaje claro, sin traza
            raise ConfigError(str(exc)) from exc
        if not resolved.is_file():
            raise ConfigError(f"no existe el fichero {path!r}")
        return resolved.read_text(encoding="utf-8", errors="replace")

    if agent == "locator":
        return " ".join(inputs), {}
    if agent == "doc_checker" or config.verifier == "doc_checker":
        if len(inputs) < 2:
            raise ConfigError("doc_checker necesita al menos dos rutas de documentos")
        if config.mode == "single_shot":  # sin herramientas: los documentos enteros, con numeración de línea, en el mensaje
            return doc_checker_single_shot_task([(p, read(p)) for p in inputs], note), {}
        aspect = f" Aspecto a comprobar: {note}." if note else ""
        listing = ", ".join(f"`{p}`" for p in inputs)
        return f"Compara estos documentos y señala sus incoherencias: {listing}.{aspect}", {}
    if agent == "code_reviewer":
        if len(inputs) != 1:
            raise ConfigError("code_reviewer necesita exactamente una ruta de fichero")
        source = read(inputs[0])
        described = f" Debería hacer lo siguiente: {note}" if note else ""
        return f"Revisa el fichero `{inputs[0]}` y señala posibles errores de programación.{described}", {"source": source, "reviewed": inputs[0]}
    if agent == "summarizer":
        blocks, parts = [], []
        for path in inputs:
            text = read(path)
            if len(text) > max_chars:
                text = text[:max_chars] + "\n[... recortado]"
            parts.append(text)
            blocks.append(f"### {path}\n```\n{text}\n```")
        return "Resume los resultados del experimento a partir de estos ficheros:\n\n" + "\n\n".join(blocks), {"source": "\n".join(parts)}
    raise ConfigError(f"el agente {agent!r} no tiene constructor de mensaje: usa --task-file")


def postprocess_output(verifier: str, report: str, context: dict, filter_numbers: bool = FILTER_NUMBERS_BY_DEFAULT) -> tuple[str, list[str], bool]:
    """Corrección por código de la salida de un rol: ``(texto, avisos, ok)``; ``avisos`` dice QUÉ se corrigió.

    - ``summarizer``: ``postprocess_summary`` (coma decimal y recorte a 150 palabras y, con ``filter_numbers``, quita las frases con cifras
      no respaldadas por la entrada). ``ok`` es falso si el filtro quitó alguna frase y deja menos de 30 palabras (fallo de formato; en la evaluación cuenta también un resumen corto sin frases quitadas).
    - ``code_reviewer``: ``correct_review_lines`` (la línea de cada hallazgo pasa a la de su cita literal más cercana; los hallazgos cuya
      cita no está en el fichero quedan marcados ``verificable: false``, sin inventarles línea). Si la salida no es JSON con ``hallazgos``
      no se toca.
    - cualquier otro verificador: sin cambios.
    """
    if verifier == "summarizer" and "source" in context:
        source = context["source"]
        fixed, info = postprocess_summary(report, source, SUMMARY_MAX_WORDS, filter_numbers=filter_numbers)
        notes = []
        if decimal_comma(report, source) != report:
            notes.append("coma decimal: se cambió el punto decimal de algún número")
        if info["mode"] != "intacto":
            notes.append(f"recorte ({info['mode']}): de {info['words_before']} a {info['words_trimmed'] if 'words_trimmed' in info else info['words_after']} palabras (máximo {SUMMARY_MAX_WORDS})")
        ok = True
        if filter_numbers:
            removed = info["filter"]["removed"]
            notes.append(f"filtro de cifras: {removed} de {info['filter']['sentences']} frases quitadas por tener cifras no respaldadas" + (f" ({'; '.join(t[:60] for t in info['filter']['removed_texts'])})" if removed else ""))
            if removed and count_words(fixed) < FILTER_MIN_WORDS:  # solo si el filtro quitó algo: un resumen corto de por sí no es culpa del filtro
                ok = False
                notes.append(f"FALLO DE FORMATO: tras el filtro el resumen tiene {count_words(fixed)} palabras (mínimo {FILTER_MIN_WORDS})")
        before, after = info["numbers_before"], info["numbers_after"]
        notes.append(f"cifras respaldadas por la entrada: {before['supported']}/{before['total']} -> {after['supported']}/{after['total']}")
        return fixed, notes, ok
    if verifier == "code_reviewer" and "source" in context:
        obj, _, error = parse_json_output(report)
        if error or not isinstance(obj, dict) or not isinstance(obj.get("hallazgos"), list):
            return report, ["salida sin JSON de hallazgos: no se corrigió nada"], True
        corrected = correct_review_lines(obj, context["source"])
        notes = [f"línea corregida a la de la cita literal: {corrected['corrected']} de {len(corrected['items'])} hallazgos"]
        notes += [f"hallazgo {i['item']}: línea {i['linea_modelo']} -> {i['linea']}" for i in corrected["items"] if i["estado"] == "corregida"]
        if corrected["unverifiable"]:
            notes.append(f"{corrected['unverifiable']} hallazgos con la cita ausente del fichero (verificable: false; sin línea inventada)")
        return json.dumps(corrected["obj"], ensure_ascii=False, indent=1), notes, True
    return report, [], True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m amp_agents.run", description="Ejecuta un agente con un modelo configurable.")
    parser.add_argument("inputs", nargs="*", help="pregunta o rutas, según el rol")
    parser.add_argument("--agent", default="executor")
    parser.add_argument("--task-file")
    parser.add_argument("--note", default="")
    parser.add_argument("--model")
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--provider")
    parser.add_argument("--root")
    parser.add_argument("--max-steps", dest="max_steps", type=int)
    parser.add_argument("--max-chars", dest="max_chars", type=int, default=8000)
    parser.add_argument("--log-dir", dest="log_dir")
    parser.add_argument("--config")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--lexical", action="store_true", help="solo retriever: BM25 sin modelo (es lo que usa por defecto)")
    parser.add_argument("--embedding", action="store_true", help="solo retriever: embeddings (necesita el índice y el servidor de modelos)")
    parser.add_argument("--hybrid", action="store_true", help="solo retriever: BM25 + embeddings por rango recíproco (necesita el índice y el servidor de modelos)")
    parser.add_argument("--build-index", dest="build_index", action="store_true")
    parser.add_argument("--raw", action="store_true", help="resumidor y revisor: muestra la salida del modelo sin corregir")
    parser.add_argument("--filter-numbers", dest="filter_numbers", action=argparse.BooleanOptionalAction, default=None, help="resumidor: quita las frases con cifras no respaldadas (por defecto, según FILTER_NUMBERS_BY_DEFAULT)")
    args = parser.parse_args(argv)
    try:
        config = get_agent_config(args.agent, args.config)
        base = Path(args.root).resolve() if args.root else PROJECT_ROOT
        if config.mode == "embedding":
            if not args.inputs:
                raise ConfigError("el recuperador necesita una consulta")
            if sum([args.lexical, args.embedding, args.hybrid]) > 1:
                raise ConfigError("--lexical, --embedding y --hybrid son incompatibles entre sí")
            backend = "lexical" if args.lexical else "embedding" if args.embedding else "hybrid" if args.hybrid else config.search_backend
            hits = retrieve(" ".join(args.inputs), root=base, config_path=args.config, k=args.k, backend=backend, build=args.build_index, agent=args.agent, model=args.model, base_url=args.base_url)
            print(json.dumps(hits, ensure_ascii=False, indent=1))
            check = verify(config.verifier, json.dumps(hits)) if config.verifier else Verification(True)
            print(f"[retriever/{backend}] resultados={len(hits)} verificación={'ok' if check.valid else check.errors}", file=sys.stderr)
            return 0 if check.valid else 1
        context: dict = {}
        if args.task_file:
            task = read_task(args.task_file)
        elif args.agent == "executor":
            raise ConfigError("el ejecutor necesita --task-file")
        else:
            task, context = prepare_task(args.agent, args.inputs, base, args.note, args.max_chars, args.config)
        result = run_task(
            args.agent,
            task,
            root=base,
            config_path=args.config,
            log_dir=args.log_dir,
            model=args.model,
            base_url=args.base_url,
            provider=args.provider,
            max_steps=args.max_steps,
        )
    except (ConfigError, ProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    shown, ok_format = result.report, True
    if config.verifier in {"summarizer", "code_reviewer"} and result.stop_reason == "final" and not args.raw:
        use_filter = FILTER_NUMBERS_BY_DEFAULT if args.filter_numbers is None else args.filter_numbers
        shown, notes, ok_format = postprocess_output(config.verifier, result.report, context, use_filter)
        for line in notes:
            print(f"corrección: {line}", file=sys.stderr)
    elif args.raw and config.verifier in {"summarizer", "code_reviewer"}:
        print("salida cruda (--raw): sin corregir", file=sys.stderr)
    print(shown)
    tokens = "sin datos" if result.tokens is None else f"{result.tokens.total_tokens}"
    print(
        f"[{result.stop_reason}] pasos={result.steps} tokens={tokens} tiempo={result.duration_s:.1f}s "
        f"herramientas={len(result.tools_called)} denegadas={len(result.denied)} registro={result.log_path}",
        file=sys.stderr,
    )
    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
    ok = result.stop_reason == "final"
    if config.verifier and ok:
        check = verify(config.verifier, shown, root=base, **context)
        print(f"verificación{' (salida cruda)' if args.raw else ''}: {'válida' if check.valid else 'NO válida: ' + '; '.join(check.errors)}", file=sys.stderr)
        ok = check.valid and ok_format
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

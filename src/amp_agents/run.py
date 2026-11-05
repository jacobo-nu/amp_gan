"""Línea de órdenes del agente con modelo intercambiable.

Uso (desde la carpeta ``src``)::

    python -m amp_agents.run --agent executor --task-file encargo.md
    python -m amp_agents.run --agent executor --task-file encargo.md --model qwen2.5:14b
    python -m amp_agents.run --agent executor --task-file encargo.md \\
        --base-url http://localhost:8080/v1 --model mi-modelo --root /ruta/a/una/copia --max-steps 15

El encargo es un fichero de texto con el formato del orquestador (``Tarea:``, ``Contexto:``,
``Entregable:``, ``Aceptación:``, ``Restricciones:``). El prompt de sistema se ensambla desde
``context/agents/<agente>.md`` y ``context/constitution/governance.md`` de ``--root``; el modelo, la
``base_url`` y los permisos salen de ``agents.toml`` (``--config``) salvo que se sobrescriban aquí.

Opciones:
    --agent        nombre del agente (sección ``[agents.<nombre>]`` y ``context/agents/<nombre>.md``)
    --task-file    fichero con el encargo (obligatorio)
    --model, --base-url, --provider, --max-steps   sobrescriben la configuración
                   (``--provider ollama_openai|openai_compat|llamacpp|vllm`` para endpoints compatibles con OpenAI)
    --root         raíz del proyecto sobre la que actúan las herramientas (por defecto, el proyecto actual);
                   úsese una copia para pruebas
    --log-dir      carpeta de registros JSONL (por defecto ``<root>/.agent_runs``)
    --config       otro ``agents.toml``

Código de salida: 0 si el motivo de parada es ``final``; 1 si no; 2 por error de configuración o de uso.
El informe final se escribe en la salida estándar y un resumen (pasos, tokens, tiempo, registro) en la
salida de error.
"""

from __future__ import annotations

import argparse
import sys
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
from .loop import AgentResult, run_agent
from .providers import ChatProvider

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
    return run_agent(
        provider_instance or make_provider(config),
        build_system_prompt(agent, base, config),
        task,
        make_tools(config),
        make_policy(config, base),
        max_steps=config.max_steps,
        max_seconds=config.max_seconds,
        log_dir=log_dir,
        run_label=run_label or f"{agent}_{config.model}",
        metadata={"agent": agent, "model": config.model, "base_url": config.base_url},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m amp_agents.run", description="Ejecuta un agente con un modelo configurable.")
    parser.add_argument("--agent", default="executor")
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--model")
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--provider")
    parser.add_argument("--root")
    parser.add_argument("--max-steps", dest="max_steps", type=int)
    parser.add_argument("--log-dir", dest="log_dir")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    try:
        task = read_task(args.task_file)
        result = run_task(
            args.agent,
            task,
            root=args.root,
            config_path=args.config,
            log_dir=args.log_dir,
            model=args.model,
            base_url=args.base_url,
            provider=args.provider,
            max_steps=args.max_steps,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.report)
    tokens = "sin datos" if result.tokens is None else f"{result.tokens.total_tokens}"
    print(
        f"[{result.stop_reason}] pasos={result.steps} tokens={tokens} tiempo={result.duration_s:.1f}s "
        f"herramientas={len(result.tools_called)} denegadas={len(result.denied)} registro={result.log_path}",
        file=sys.stderr,
    )
    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
    return 0 if result.stop_reason == "final" else 1


if __name__ == "__main__":
    sys.exit(main())

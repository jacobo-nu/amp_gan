"""Bucle del agente: modelo -> herramientas -> modelo, hasta respuesta final o límite.

``run_agent`` es independiente del proveedor: solo usa la interfaz :class:`~amp_agents.providers.ChatProvider`.
Los errores de herramienta (argumentos mal formados, política, fichero inexistente...) no detienen el
bucle: se devuelven al modelo como mensaje de herramienta para que se corrija. Si falla el proveedor
(servidor caído, HTTP de error) la ejecución termina con motivo ``error``.

Cada ejecución escribe un registro JSONL (un evento por línea: ``start``, ``assistant``, ``tool``,
``end``) en ``log_dir``; por defecto ``<raíz del proyecto>/.agent_runs/``.

Motivos de parada: ``final`` (respuesta sin llamadas), ``max_steps``, ``timeout`` (límite de tiempo total;
al proveedor se le pasa el tiempo que queda y cancela la petición en curso si se agota) y ``error`` (fallo
del proveedor).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .providers import ChatProvider, ProviderError, ProviderTimeout, Usage
from .tools import Policy, Tool

DEFAULT_LOG_DIRNAME = ".agent_runs"
_EMPTY_NUDGE = (
    "Tu último mensaje estaba vacío. Si ya terminaste, entrega ahora el informe final con el formato "
    "Hecho / Verificado / Fallos / Pendiente; si no, llama a una herramienta."
)


@dataclass
class AgentResult:
    """Resultado de una ejecución.

    Attributes:
        report: texto de la última respuesta del modelo (el informe final si ``stop_reason == "final"``).
        steps: número de llamadas al modelo.
        stop_reason: ``final``, ``max_steps``, ``timeout`` o ``error``.
        duration_s: segundos totales.
        tokens: suma de tokens notificados por el servidor (``None`` si no los da).
        last_prompt_tokens: tokens del último contexto enviado (indica cuánto contexto se usó).
        tools_called: nombres de las herramientas llamadas, en orden.
        denied: llamadas rechazadas por la política (herramienta, argumentos y mensaje).
        tool_errors: número de llamadas a herramienta que devolvieron error (incluye las denegadas).
        error: mensaje del fallo si ``stop_reason == "error"``.
        log_path: ruta del registro JSONL (``None`` si el registro está desactivado).
    """

    report: str
    steps: int
    stop_reason: str
    duration_s: float
    tokens: Usage | None = None
    last_prompt_tokens: int | None = None
    tools_called: list[str] = field(default_factory=list)
    denied: list[dict] = field(default_factory=list)
    tool_errors: int = 0
    error: str | None = None
    log_path: Path | None = None


class _Log:
    def __init__(self, path: Path | None):
        self.path = path
        self._t0 = time.monotonic()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def event(self, kind: str, **data) -> None:
        if self.path is None:
            return
        record = {"t": round(time.monotonic() - self._t0, 3), "event": kind, **data}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _limit(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[... recortado: {len(text) - max_chars} caracteres más]"


def run_agent(
    provider: ChatProvider,
    system_prompt: str,
    task: str,
    tools: Sequence[Tool],
    policy: Policy,
    max_steps: int = 25,
    max_seconds: float | None = 900.0,
    log_dir: Path | str | bool | None = None,
    run_label: str = "run",
    max_result_chars: int = 8000,
    metadata: dict | None = None,
) -> AgentResult:
    """Ejecuta el agente y devuelve un :class:`AgentResult`.

    Args:
        provider: proveedor de modelo.
        system_prompt: prompt de sistema (ver :func:`amp_agents.agents.build_system_prompt`).
        task: encargo (formato Tarea/Contexto/Entregable/Aceptación/Restricciones).
        tools: herramientas disponibles para el modelo.
        policy: permisos aplicados a cada herramienta.
        max_steps: máximo de llamadas al modelo.
        max_seconds: límite de tiempo total; se comprueba entre pasos y además se pasa al proveedor lo que queda
            para que cancele la petición HTTP en curso; ``None`` para no limitar.
        log_dir: carpeta de registros JSONL; ``None`` = ``<policy.root>/.agent_runs``; ``False`` = sin registro.
        run_label: etiqueta que se añade al nombre del registro.
        max_result_chars: tope de caracteres de cada resultado de herramienta que ve el modelo.
        metadata: datos libres que se guardan en el evento ``start``.
    """
    started = time.monotonic()
    log_path = None
    if log_dir is not False:
        folder = Path(log_dir) if log_dir else policy.root / DEFAULT_LOG_DIRNAME
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_label)
        log_path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}_{uuid.uuid4().hex[:6]}.jsonl"
    log = _Log(log_path)
    by_name = {t.name: t for t in tools}
    definitions = [t.definition() for t in tools]
    messages: list[dict] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]
    log.event(
        "start",
        provider=getattr(provider, "name", type(provider).__name__),
        model=getattr(provider, "model", None),
        tools=list(by_name),
        max_steps=max_steps,
        root=str(policy.root),
        metadata=metadata or {},
        system_prompt=system_prompt,
        task=task,
    )

    result = AgentResult(report="", steps=0, stop_reason="max_steps", duration_s=0.0, log_path=log_path)
    total = Usage()
    got_usage = False
    nudged = False

    while True:
        if result.steps >= max_steps:
            result.stop_reason = "max_steps"
            break
        if max_seconds is not None and time.monotonic() - started > max_seconds:
            result.stop_reason = "timeout"
            break
        result.steps += 1
        remaining = None if max_seconds is None else max(0.001, max_seconds - (time.monotonic() - started))
        try:
            reply = provider.chat(messages, definitions, timeout=remaining)
        except ProviderTimeout as exc:
            result.stop_reason, result.error = "timeout", str(exc)
            log.event("error", step=result.steps, message=str(exc), timeout=True)
            break
        except ProviderError as exc:
            result.stop_reason, result.error = "error", str(exc)
            log.event("error", step=result.steps, message=str(exc))
            break
        if reply.usage is not None:
            got_usage = True
            total.prompt_tokens += reply.usage.prompt_tokens
            total.completion_tokens += reply.usage.completion_tokens
            result.last_prompt_tokens = reply.usage.prompt_tokens
        messages.append(reply.to_message())
        if reply.content:
            result.report = reply.content
        log.event(
            "assistant",
            step=result.steps,
            content=reply.content,
            reasoning=reply.reasoning,
            finish_reason=reply.finish_reason,
            tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments, "error": c.error} for c in reply.tool_calls],
            usage=None if reply.usage is None else {"prompt": reply.usage.prompt_tokens, "completion": reply.usage.completion_tokens},
        )
        if not reply.tool_calls:
            if not reply.content.strip() and not nudged:
                nudged = True
                messages.append({"role": "user", "content": _EMPTY_NUDGE})
                log.event("nudge", step=result.steps)
                continue
            result.stop_reason = "final"
            break
        for call in reply.tool_calls:
            t0 = time.monotonic()
            result.tools_called.append(call.name)
            tool = by_name.get(call.name)
            if call.error is not None:
                content, ok, denied = f"ERROR: {call.error}. Vuelve a llamar a {call.name or 'la herramienta'} con argumentos JSON válidos.", False, False
            elif tool is None:
                content, ok, denied = f"ERROR: la herramienta {call.name!r} no existe; las disponibles son {sorted(by_name)}", False, False
            else:
                outcome = tool.run(policy, call.arguments)
                content, ok, denied = outcome.content, outcome.ok, outcome.denied
            if not ok:
                result.tool_errors += 1
            if denied:
                result.denied.append({"tool": call.name, "arguments": call.arguments, "message": content})
            shown = _limit(content, max_result_chars)
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": shown})
            log.event(
                "tool",
                step=result.steps,
                id=call.id,
                name=call.name,
                arguments=call.arguments,
                ok=ok,
                denied=denied,
                result=shown,
                result_chars=len(content),
                seconds=round(time.monotonic() - t0, 3),
            )

    result.tokens = total if got_usage else None
    result.duration_s = time.monotonic() - started
    log.event(
        "end",
        stop_reason=result.stop_reason,
        steps=result.steps,
        duration_s=round(result.duration_s, 3),
        tokens=None if result.tokens is None else {"prompt": total.prompt_tokens, "completion": total.completion_tokens},
        tools_called=result.tools_called,
        denied=result.denied,
        tool_errors=result.tool_errors,
        error=result.error,
        report=result.report,
    )
    return result

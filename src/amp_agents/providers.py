"""Adaptadores de proveedor de modelo.

Un proveedor recibe la lista de mensajes (formato de chat de OpenAI) y las definiciones de
herramientas, y devuelve un :class:`AssistantMessage` con texto y/o llamadas a herramientas. El
bucle (:mod:`amp_agents.loop`) solo conoce esta interfaz, así que cambiar de modelo o de servidor es
cambiar de proveedor (o de configuración), no de código.

- :class:`OpenAICompatProvider`: ``POST {base_url}/chat/completions`` con ``http.client``. Sirve para
  Ollama (``http://localhost:11434/v1``), llama.cpp, vLLM o cualquier endpoint compatible.
- :class:`OllamaProvider`: API nativa de Ollama (``POST {base_url}/api/chat``). Existe porque el
  endpoint compatible con OpenAI de Ollama no permite fijar el contexto (``num_ctx``, 4096 por defecto) ni
  desactivar el razonamiento de modelos como qwen3 (``think``); el resto del sistema no nota la diferencia.
  Además ofrece ``embed`` (``POST /api/embed``) para modelos de embeddings.
- :class:`SystemOneProvider`: modelos de DECISIÓN de Ollama (p. ej. ``tev1``), que no generan texto: ``POST {base_url}/v1/systemone``
  con un texto de estado y una o varias preguntas, y devuelven una probabilidad (0-1) por pregunta.
- :class:`ScriptedProvider`: respuestas predefinidas, sin red (para tests).
"""

from __future__ import annotations

import http.client
import json
import re
import socket
import threading
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class ProviderError(RuntimeError):
    """Fallo del proveedor (servidor caído, HTTP de error, respuesta ilegible)."""


class ProviderTimeout(ProviderError):
    """La petición superó su tiempo máximo y se canceló (se cierra la conexión, lo que detiene la generación)."""


@dataclass
class Usage:
    """Uso de tokens notificado por el servidor."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ToolCall:
    """Llamada a herramienta pedida por el modelo.

    Si los argumentos del modelo no son un objeto JSON válido, ``arguments`` queda vacío y ``error``
    explica el problema: el bucle lo devuelve al modelo como mensaje de herramienta (no es excepción).
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_message(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments, ensure_ascii=False)},
        }


@dataclass
class AssistantMessage:
    """Mensaje del asistente: texto, llamadas a herramientas y uso de tokens (si lo da el servidor)."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None
    reasoning: str = ""
    finish_reason: str = ""  # "stop", "tool_calls", "length" (tope de tokens)... según el servidor

    @classmethod
    def text(cls, content: str, usage: Usage | None = None) -> "AssistantMessage":
        return cls(content=content, usage=usage)

    @classmethod
    def calls(cls, *calls: tuple[str, dict], content: str = "") -> "AssistantMessage":
        """Atajo para tests: ``AssistantMessage.calls(("read_file", {"path": "src/x.py"}))``."""
        return cls(content=content, tool_calls=[ToolCall(f"call_{i}", n, dict(a)) for i, (n, a) in enumerate(calls)])

    def to_message(self) -> dict:
        """Mensaje en formato de chat para devolverlo al modelo en el historial."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [c.to_message() for c in self.tool_calls]
        return message


class ChatProvider(ABC):
    """Interfaz común de los proveedores de modelo."""

    name = "provider"

    @abstractmethod
    def chat(self, messages: Sequence[dict], tools: Sequence[dict] | None = None, timeout: float | None = None) -> AssistantMessage:
        """Envía el historial y devuelve la respuesta del modelo (``ProviderError`` si falla).

        ``timeout`` (segundos) es el tiempo MÁXIMO TOTAL de esta petición (si el proveedor tiene el suyo,
        se usa el menor); al agotarse se cancela la petición y se lanza :class:`ProviderTimeout`.
        """


def parse_tool_arguments(raw: Any) -> tuple[dict, str | None]:
    """Convierte los argumentos de una llamada en ``(dict, error)``; nunca lanza excepción."""
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if not isinstance(raw, str):
        return {}, f"los argumentos deben ser un objeto JSON, no {type(raw).__name__}"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"los argumentos no son JSON válido ({exc.msg}, posición {exc.pos}): {raw[:200]!r}"
    if not isinstance(value, dict):
        return {}, f"los argumentos deben ser un objeto JSON, no {type(value).__name__}: {raw[:200]!r}"
    return value, None


_TAG_RE = re.compile(r"</?tool_call>")


def extract_text_tool_calls(content: str, first_index: int = 0) -> tuple[str, list[ToolCall]]:
    """Rescata llamadas a herramienta que el modelo escribió como texto en ``content``.

    Algunos modelos pequeños emiten, además de (o en lugar de) las llamadas estructuradas, objetos JSON
    ``{"name": ..., "arguments": {...}}`` en el texto (a veces entre etiquetas ``<tool_call>``) que el
    servidor no reconoce. Se aceptan solo objetos con EXACTAMENTE esas dos claves. Devuelve el texto sin
    esas llamadas y la lista de llamadas (los argumentos no válidos quedan como error legible).
    """
    decoder = json.JSONDecoder()
    calls: list[ToolCall] = []
    kept: list[str] = []
    pos = last = 0
    while True:
        start = content.find("{", pos)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(content, start)
        except ValueError:
            pos = start + 1
            continue
        if isinstance(obj, dict) and set(obj) == {"name", "arguments"} and isinstance(obj["name"], str):
            arguments, error = parse_tool_arguments(obj["arguments"])
            calls.append(ToolCall(f"call_{first_index + len(calls)}", obj["name"], arguments, error))
            kept.append(content[last:start])
            last = pos = end
        else:
            pos = start + 1
    if not calls:
        return content, []
    kept.append(content[last:])
    return _TAG_RE.sub("", "".join(kept)).strip(), calls


def _merge_text_calls(content: str, calls: list[ToolCall]) -> tuple[str, list[ToolCall]]:
    """Añade a ``calls`` las llamadas escritas como texto (sin duplicar las ya estructuradas)."""
    clean, extra = extract_text_tool_calls(content, len(calls))
    known = {(c.name, json.dumps(c.arguments, sort_keys=True)) for c in calls}
    for call in extra:
        key = (call.name, json.dumps(call.arguments, sort_keys=True))
        if key not in known:
            calls.append(call)
            known.add(key)
    return clean, calls


def post_json(url: str, body: dict, base_url: str, timeout: float, headers: dict | None = None) -> dict:
    """``POST`` JSON con ``http.client``; traduce cualquier fallo de red o HTTP a :class:`ProviderError`.

    ``timeout`` es un límite de reloj para TODA la petición (conexión, envío y espera de la respuesta, también
    si el servidor envía datos a cuentagotas): al vencer, un temporizador cierra el socket, la lectura falla y
    se lanza :class:`ProviderTimeout`. El servidor (p. ej. Ollama) detecta la desconexión y deja de generar.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ProviderError(f"URL no válida: {url!r}")
    connection_class = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(parts.hostname, parts.port, timeout=timeout)
    expired = threading.Event()
    held: list[socket.socket] = []  # el socket se guarda aquí: ``connection.sock`` se pierde si la respuesta cierra la conexión

    def cancel() -> None:
        expired.set()
        for sock in held:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    watchdog = threading.Timer(timeout, cancel)
    watchdog.daemon = True
    all_headers = {"Content-Type": "application/json", **(headers or {})}
    path = parts.path + (f"?{parts.query}" if parts.query else "")
    timed_out = ProviderTimeout(f"el servidor en {base_url} no respondió en {timeout:g} s (petición cancelada)")
    watchdog.start()
    try:
        connection.connect()
        held.append(connection.sock)
        if expired.is_set():
            raise timed_out
        connection.request("POST", path or "/", body=json.dumps(body).encode("utf-8"), headers=all_headers)
        response = connection.getresponse()
        payload = response.read()
        status = response.status
    except (socket.timeout, TimeoutError) as exc:
        raise timed_out from exc
    except (OSError, http.client.HTTPException) as exc:
        if expired.is_set():
            raise timed_out from exc
        raise ProviderError(f"el servidor no responde en {base_url}: {getattr(exc, 'strerror', None) or exc}") from exc
    finally:
        watchdog.cancel()
        connection.close()
    if expired.is_set():
        raise timed_out
    if status >= 400:
        raise ProviderError(f"el servidor en {base_url} devolvió HTTP {status}: {payload.decode('utf-8', errors='replace')[:300]}")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"respuesta de {base_url} no es JSON: {payload[:200]!r}") from exc


class OpenAICompatProvider(ChatProvider):
    """Cliente de ``POST {base_url}/chat/completions`` (API compatible con OpenAI) con ``http.client``.

    Args:
        base_url: por defecto el servidor local de Ollama (``http://localhost:11434/v1``).
        model: nombre del modelo en el servidor (p. ej. ``qwen3:4b``).
        api_key: opcional; se envía como ``Authorization: Bearer``. Los servidores locales no la piden.
        timeout: segundos máximos (de reloj, no por lectura de socket) de cada petición.
        options: campos adicionales del cuerpo de la petición (``temperature``, ``seed``, ``max_tokens``...).
    """

    name = "openai_compat"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "",
        api_key: str | None = None,
        timeout: float = 600.0,
        options: dict | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.options = dict(options or {})

    def _request(self, body: dict, timeout: float | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        limit = self.timeout if timeout is None else min(self.timeout, timeout)
        return post_json(f"{self.base_url}/chat/completions", body, self.base_url, limit, headers)

    def chat(self, messages: Sequence[dict], tools: Sequence[dict] | None = None, timeout: float | None = None) -> AssistantMessage:
        body: dict[str, Any] = {"model": self.model, "messages": list(messages), "stream": False}
        if tools:
            body["tools"] = list(tools)
        body.update(self.options)
        data = self._request(body, timeout)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"respuesta sin choices[0].message: {str(data)[:200]}") from exc
        calls = []
        for i, raw in enumerate(message.get("tool_calls") or []):
            function = (raw or {}).get("function") or {}
            arguments, error = parse_tool_arguments(function.get("arguments"))
            calls.append(ToolCall(str(raw.get("id") or f"call_{i}"), str(function.get("name") or ""), arguments, error))
        usage = None
        raw_usage = data.get("usage")
        if isinstance(raw_usage, dict):
            usage = Usage(int(raw_usage.get("prompt_tokens") or 0), int(raw_usage.get("completion_tokens") or 0))
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        content, calls = _merge_text_calls(message.get("content") or "", calls)
        return AssistantMessage(content=content, tool_calls=calls, usage=usage, reasoning=reasoning, finish_reason=str(data["choices"][0].get("finish_reason") or ""))


class OllamaProvider(ChatProvider):
    """Cliente de la API nativa de Ollama (``POST {base_url}/api/chat``).

    Misma interfaz que :class:`OpenAICompatProvider`. Las ``options`` van a ``options`` de Ollama
    (``num_ctx``, ``temperature``, ``seed``...); las claves ``think``, ``keep_alive`` y ``format`` (esquema JSON que
    restringe la salida) se envían en el primer nivel de la petición. ``base_url`` es la raíz del servidor (sin ``/v1``).
    """

    name = "ollama"

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = "", timeout: float = 600.0, options: dict | None = None):
        base = base_url.rstrip("/")
        self.base_url = base[: -len("/v1")] if base.endswith("/v1") else base
        self.model = model
        self.timeout = timeout
        self.options = dict(options or {})

    @staticmethod
    def _convert(messages: Sequence[dict]) -> list[dict]:
        """Pasa el historial en formato OpenAI al nativo (argumentos como objeto, ``tool_name``)."""
        out = []
        for m in messages:
            m = dict(m)
            if m.get("role") == "assistant" and m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    args, _ = parse_tool_arguments(c["function"].get("arguments"))
                    calls.append({"function": {"name": c["function"]["name"], "arguments": args}})
                m["tool_calls"] = calls
            elif m.get("role") == "tool":
                m = {"role": "tool", "tool_name": m.get("name", ""), "content": m.get("content", "")}
            out.append(m)
        return out

    def chat(self, messages: Sequence[dict], tools: Sequence[dict] | None = None, timeout: float | None = None) -> AssistantMessage:
        options = dict(self.options)
        body: dict[str, Any] = {"model": self.model, "messages": self._convert(messages), "stream": False}
        for key in ("think", "keep_alive", "format"):
            if key in options:
                body[key] = options.pop(key)
        if options:
            body["options"] = options
        if tools:
            body["tools"] = list(tools)
        limit = self.timeout if timeout is None else min(self.timeout, timeout)
        data = post_json(f"{self.base_url}/api/chat", body, self.base_url, limit)
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            raise ProviderError(f"respuesta sin message: {str(data)[:200]}")
        calls = []
        for i, raw in enumerate(message.get("tool_calls") or []):
            function = (raw or {}).get("function") or {}
            arguments, error = parse_tool_arguments(function.get("arguments"))
            calls.append(ToolCall(f"call_{i}", str(function.get("name") or ""), arguments, error))
        usage = None
        if "prompt_eval_count" in data or "eval_count" in data:
            usage = Usage(int(data.get("prompt_eval_count") or 0), int(data.get("eval_count") or 0))
        content, calls = _merge_text_calls(message.get("content") or "", calls)
        return AssistantMessage(content=content, tool_calls=calls, usage=usage, reasoning=message.get("thinking") or "", finish_reason=str(data.get("done_reason") or ""))

    def embed(self, texts: Sequence[str], timeout: float | None = None) -> list[list[float]]:
        """Vectores de incrustación de ``texts`` (``POST {base_url}/api/embed``), uno por texto y en el mismo orden.

        Pensado para modelos de embeddings como ``nomic-embed-text``. De ``options`` solo se usan las opciones
        del modelo (p. ej. ``num_ctx``) y ``keep_alive``/``truncate`` (primer nivel); ``think`` y ``num_predict``
        no tienen sentido aquí y se ignoran. Lanza :class:`ProviderError` si la respuesta no trae un vector
        numérico por texto, y :class:`ProviderTimeout` si se agota el tiempo.
        """
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        if not all(isinstance(t, str) for t in items):
            raise ProviderError("embed solo admite textos (cadenas)")
        options = dict(self.options)
        body: dict[str, Any] = {"model": self.model, "input": items}
        for key in ("keep_alive", "truncate"):
            if key in options:
                body[key] = options.pop(key)
        options.pop("think", None)
        options.pop("format", None)
        options.pop("num_predict", None)
        if options:
            body["options"] = options
        limit = self.timeout if timeout is None else min(self.timeout, timeout)
        data = post_json(f"{self.base_url}/api/embed", body, self.base_url, limit)
        vectors = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(vectors, list) or len(vectors) != len(items):
            raise ProviderError(f"respuesta de embed sin un vector por texto ({len(items)} esperados): {str(data)[:200]}")
        out: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list) or not vector or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector):
                raise ProviderError(f"vector de embed no válido: {str(vector)[:100]}")
            out.append([float(x) for x in vector])
        return out


class SystemOneProvider(ChatProvider):
    """Modelo de decisión de Ollama (``POST {base_url}/v1/systemone``): no genera texto, devuelve probabilidades.

    El cuerpo es ``{"model", "state", "questions": {nombre: {"type": "noul", "instructions": pregunta}}}`` y la respuesta
    ``{"answers": {nombre: {"type": "noul", "noul": p}}}`` con ``p`` entre 0 y 1. :meth:`decide` devuelve ``{nombre: p}``;
    una respuesta sin todas las preguntas o con una probabilidad que no sea un número entre 0 y 1 es un
    :class:`ProviderError`. La petición tiene tope de tiempo de reloj (como el resto de proveedores). ``chat`` no existe: el modelo
    responde «does not support chat».
    """

    name = "ollama_systemone"

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, model: str = "", timeout: float = 60.0, options: dict | None = None):
        base = base_url.rstrip("/")
        self.base_url = base[: -len("/v1")] if base.endswith("/v1") else base
        self.model = model
        self.timeout = timeout
        self.options = dict(options or {})

    def chat(self, messages: Sequence[dict], tools: Sequence[dict] | None = None, timeout: float | None = None) -> AssistantMessage:
        raise ProviderError(f"el modelo de decisión {self.model!r} no genera texto (no admite chat): usa decide()")

    def decide(self, state: str, questions: dict[str, str], timeout: float | None = None) -> dict[str, float]:
        """Probabilidad de cada pregunta (sí = 1) dado el texto ``state``; ``ProviderError``/``ProviderTimeout`` si falla."""
        if not questions:
            raise ProviderError("decide necesita al menos una pregunta")
        body = {
            "model": self.model,
            "state": state,
            "questions": {name: {"type": "noul", "instructions": text} for name, text in questions.items()},
        }
        limit = self.timeout if timeout is None else min(self.timeout, timeout)
        data = post_json(f"{self.base_url}/v1/systemone", body, self.base_url, limit)
        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, dict):
            raise ProviderError(f"respuesta sin answers: {str(data)[:200]}")
        out: dict[str, float] = {}
        for name in questions:
            item = answers.get(name)
            value = item.get("noul") if isinstance(item, dict) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise ProviderError(f"probabilidad no válida para {name!r}: {str(item)[:100]}")
            out[name] = float(value)
        return out


class ScriptedProvider(ChatProvider):
    """Proveedor de tests: devuelve las respuestas predefinidas en orden, sin red.

    ``responses`` admite :class:`AssistantMessage`, ``str`` (respuesta de solo texto) o una excepción
    (se lanza, para simular fallos). ``received`` guarda una copia de los mensajes de cada llamada.
    """

    name = "scripted"

    def __init__(self, responses: Sequence[AssistantMessage | str | Exception]):
        self._responses = list(responses)
        self.received: list[list[dict]] = []
        self.timeouts: list[float | None] = []  # ``timeout`` recibido en cada llamada

    def chat(self, messages: Sequence[dict], tools: Sequence[dict] | None = None, timeout: float | None = None) -> AssistantMessage:
        self.timeouts.append(timeout)
        self.received.append(json.loads(json.dumps(list(messages))))
        if not self._responses:
            raise ProviderError("el guion del proveedor se ha agotado")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return AssistantMessage.text(item)
        return item

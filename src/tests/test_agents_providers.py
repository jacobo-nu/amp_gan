"""Tests de los proveedores contra un servidor HTTP local de prueba (sin red externa ni Ollama)."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from agents_helpers import make_project

from amp_agents.loop import run_agent
from amp_agents.providers import (
    AssistantMessage,
    ChatProvider,
    OllamaProvider,
    OpenAICompatProvider,
    ProviderError,
    ProviderTimeout,
    ScriptedProvider,
    extract_text_tool_calls,
    parse_tool_arguments,
)
from amp_agents.tools import Policy, build_tools


class FakeServer:
    """Servidor compatible con ``/v1/chat/completions`` que devuelve respuestas predefinidas."""

    def __init__(self):
        self.responses: list[tuple[int, dict | str]] = []
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.stop = threading.Event()
        self.disconnected = False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                outer.requests.append({"path": self.path, "body": json.loads(self.rfile.read(length))})
                outer.headers.append(dict(self.headers))
                status, payload = outer.responses.pop(0)
                if status in ("slow", "trickle"):  # respuestas que nunca terminan a tiempo
                    try:
                        self.send_response(200)
                        self.send_header("Content-Length", "100000")
                        self.end_headers()
                        for _ in range(100):
                            if status == "slow":
                                outer.stop.wait(0.1)
                            else:
                                self.wfile.write(b" ")
                                self.wfile.flush()
                                outer.stop.wait(0.1)
                            if outer.stop.is_set():
                                break
                    except OSError:
                        outer.disconnected = True
                    return
                data = payload if isinstance(payload, str) else json.dumps(payload)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data.encode("utf-8"))

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self):
        self.stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()


def text_reply(content, usage=True):
    body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    if usage:
        body["usage"] = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    return 200, body


def tool_reply(name, arguments, call_id="call_9"):
    return 200, {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
                }
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 8},
    }


@pytest.fixture
def server():
    srv = FakeServer()
    yield srv
    srv.close()


def test_providers_share_the_interface():
    assert issubclass(OpenAICompatProvider, ChatProvider) and issubclass(ScriptedProvider, ChatProvider)
    assert OpenAICompatProvider(model="m").base_url == "http://localhost:11434/v1"
    assert OpenAICompatProvider(base_url="http://h:1/v1/", model="m").base_url == "http://h:1/v1"


def test_text_reply_with_usage(server):
    server.responses.append(text_reply("hola"))
    provider = OpenAICompatProvider(server.base_url, "m", api_key="k", options={"temperature": 0.1, "seed": 7})
    reply = provider.chat([{"role": "user", "content": "x"}], [{"type": "function", "function": {"name": "t"}}])
    assert reply.content == "hola" and reply.tool_calls == []
    assert reply.usage.prompt_tokens == 12 and reply.usage.completion_tokens == 5 and reply.usage.total_tokens == 17
    sent = server.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["body"]["model"] == "m" and sent["body"]["stream"] is False
    assert sent["body"]["temperature"] == 0.1 and sent["body"]["seed"] == 7
    assert sent["body"]["tools"][0]["function"]["name"] == "t"
    assert server.headers[0]["Authorization"] == "Bearer k"


def test_reply_without_usage_and_without_tools_in_request(server):
    server.responses.append(text_reply("hola", usage=False))
    reply = OpenAICompatProvider(server.base_url, "m").chat([{"role": "user", "content": "x"}])
    assert reply.usage is None
    assert "tools" not in server.requests[0]["body"] and "Authorization" not in server.headers[0]


def test_reply_with_tool_calls_parses_json_arguments(server):
    server.responses.append(tool_reply("read_file", '{"path": "src/a.py", "offset": 3}'))
    reply = OpenAICompatProvider(server.base_url, "m").chat([{"role": "user", "content": "x"}])
    call = reply.tool_calls[0]
    assert call.id == "call_9" and call.name == "read_file" and call.error is None
    assert call.arguments == {"path": "src/a.py", "offset": 3}
    assert reply.to_message()["tool_calls"][0]["function"]["arguments"] == '{"path": "src/a.py", "offset": 3}'


def test_arguments_as_object_are_accepted(server):
    server.responses.append(tool_reply("list_dir", {"path": "src"}))
    call = OpenAICompatProvider(server.base_url, "m").chat([]).tool_calls[0]
    assert call.arguments == {"path": "src"} and call.error is None


@pytest.mark.parametrize("raw", ['{"path": "src/a.py"', "no es json", "[1, 2]", '"texto"', 5])
def test_malformed_arguments_give_readable_error_not_exception(server, raw):
    server.responses.append(tool_reply("read_file", raw))
    call = OpenAICompatProvider(server.base_url, "m").chat([]).tool_calls[0]
    assert call.arguments == {} and call.error and "argumentos" in call.error
    assert call.to_message()["function"]["arguments"] == "{}"


def test_parse_tool_arguments_empty_is_ok():
    assert parse_tool_arguments("") == ({}, None) and parse_tool_arguments(None) == ({}, None)


def test_http_500_raises_provider_error(server):
    server.responses.append((500, "fallo interno del modelo"))
    with pytest.raises(ProviderError, match=r"HTTP 500.*fallo interno"):
        OpenAICompatProvider(server.base_url, "m").chat([])


def test_non_json_and_malformed_body_raise_provider_error(server):
    server.responses += [(200, "<html>"), (200, {"sin": "choices"})]
    provider = OpenAICompatProvider(server.base_url, "m")
    with pytest.raises(ProviderError, match="no es JSON"):
        provider.chat([])
    with pytest.raises(ProviderError, match="choices"):
        provider.chat([])


def test_server_down_gives_clear_message():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}/v1"
    with pytest.raises(ProviderError) as info:
        OpenAICompatProvider(url, "m", timeout=2).chat([])
    assert f"el servidor no responde en {url}" in str(info.value)


def test_timeout_gives_clear_message():
    with socket.socket() as s:  # acepta conexiones (backlog) pero nunca responde
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        url = f"http://127.0.0.1:{s.getsockname()[1]}/v1"
        with pytest.raises(ProviderError, match="no respondió en 0.5 s"):
            OpenAICompatProvider(url, "m", timeout=0.5).chat([])


def test_same_loop_runs_with_http_server_and_scripted_provider(server, tmp_path):
    project = make_project(tmp_path / "proj")
    server.responses += [tool_reply("read_file", '{"path": "src/amp_gan/mod.py"}'), text_reply("Hecho: leído")]
    http_result = run_agent(
        OpenAICompatProvider(server.base_url, "m"), "S", "T", build_tools(), Policy(root=project), log_dir=tmp_path / "l1"
    )
    scripted_result = run_agent(
        ScriptedProvider([AssistantMessage.calls(("read_file", {"path": "src/amp_gan/mod.py"})), "Hecho: leído"]),
        "S", "T", build_tools(), Policy(root=project), log_dir=tmp_path / "l2",
    )
    for result in (http_result, scripted_result):
        assert result.stop_reason == "final" and result.report == "Hecho: leído"
        assert result.steps == 2 and result.tools_called == ["read_file"]
    assert http_result.tokens.total_tokens == 30 + 8 + 12 + 5 and scripted_result.tokens is None
    history = server.requests[1]["body"]["messages"]
    assert history[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert history[3]["role"] == "tool" and "def f(x):" in history[3]["content"]


def test_loop_reports_server_down_as_error(tmp_path):
    project = make_project(tmp_path / "proj")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    result = run_agent(
        OpenAICompatProvider(f"http://127.0.0.1:{port}/v1", "m", timeout=2), "S", "T", build_tools(), Policy(root=project), log_dir=False
    )
    assert result.stop_reason == "error" and "el servidor no responde en" in result.error


def test_malformed_arguments_over_http_reach_the_model_as_tool_error(server, tmp_path):
    project = make_project(tmp_path / "proj")
    server.responses += [tool_reply("read_file", "{roto"), text_reply("Hecho")]
    result = run_agent(OpenAICompatProvider(server.base_url, "m"), "S", "T", build_tools(), Policy(root=project), log_dir=False)
    assert result.stop_reason == "final" and result.tool_errors == 1
    tool_msg = server.requests[1]["body"]["messages"][-1]
    assert tool_msg["role"] == "tool" and "JSON válido" in tool_msg["content"]


# -- API nativa de Ollama ---------------------------------------------------------------------------------
def native_reply(content="", tool_calls=None, thinking=None, usage=True):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if thinking:
        message["thinking"] = thinking
    body = {"model": "m", "message": message, "done": True}
    if usage:
        body.update(prompt_eval_count=40, eval_count=9)
    return 200, body


def test_ollama_provider_sends_native_request_with_options_and_think(server):
    server.responses.append(native_reply("hola", thinking="pienso"))
    provider = OllamaProvider(server.base_url, "m", options={"num_ctx": 8192, "temperature": 0.1, "think": False, "keep_alive": "5m"})
    assert provider.base_url.endswith(server.base_url.rsplit("/v1", 1)[0].rsplit(":", 1)[1])  # se quitó /v1
    reply = provider.chat([{"role": "user", "content": "x"}], [{"type": "function", "function": {"name": "t"}}])
    assert reply.content == "hola" and reply.reasoning == "pienso"
    assert reply.usage.prompt_tokens == 40 and reply.usage.completion_tokens == 9
    sent = server.requests[0]
    assert sent["path"] == "/api/chat"
    assert sent["body"]["options"] == {"num_ctx": 8192, "temperature": 0.1}
    assert sent["body"]["think"] is False and sent["body"]["keep_alive"] == "5m" and sent["body"]["stream"] is False
    assert sent["body"]["tools"][0]["function"]["name"] == "t"


def test_ollama_provider_parses_tool_calls_and_converts_history(server):
    server.responses.append(native_reply(tool_calls=[{"function": {"name": "read_file", "arguments": {"path": "src/a.py"}}}], usage=False))
    provider = OllamaProvider(server.base_url, "m")
    first = provider.chat([{"role": "user", "content": "x"}])
    assert first.tool_calls[0].name == "read_file" and first.tool_calls[0].arguments == {"path": "src/a.py"}
    assert first.usage is None and "options" not in server.requests[0]["body"] and "think" not in server.requests[0]["body"]
    server.responses.append(native_reply("fin"))
    provider.chat([{"role": "user", "content": "x"}, first.to_message(), {"role": "tool", "tool_call_id": "call_0", "name": "read_file", "content": "ok"}])
    history = server.requests[1]["body"]["messages"]
    assert history[1]["tool_calls"][0]["function"]["arguments"] == {"path": "src/a.py"}
    assert history[2] == {"role": "tool", "tool_name": "read_file", "content": "ok"}


def test_ollama_provider_errors_are_provider_errors(server):
    server.responses += [(500, "boom"), (200, {"nada": 1})]
    provider = OllamaProvider(server.base_url, "m")
    with pytest.raises(ProviderError, match="HTTP 500"):
        provider.chat([])
    with pytest.raises(ProviderError, match="sin message"):
        provider.chat([])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(ProviderError, match=f"el servidor no responde en http://127.0.0.1:{port}"):
        OllamaProvider(f"http://127.0.0.1:{port}", "m", timeout=2).chat([])


def test_same_loop_runs_with_native_ollama_server(server, tmp_path):
    project = make_project(tmp_path / "proj")
    server.responses += [
        native_reply(tool_calls=[{"function": {"name": "grep", "arguments": {"pattern": "def f", "path": "src"}}}]),
        native_reply("Hecho: encontrado"),
    ]
    result = run_agent(OllamaProvider(server.base_url, "m"), "S", "T", build_tools(), Policy(root=project), log_dir=False)
    assert result.stop_reason == "final" and result.report == "Hecho: encontrado" and result.tools_called == ["grep"]
    assert result.tokens.total_tokens == 98
    assert "mod.py:1:def f(x):" in server.requests[1]["body"]["messages"][-1]["content"]


# -- llamadas a herramienta escritas como texto (modelos pequeños) --------------------------------------------
def test_extract_text_tool_calls_plain_and_tagged():
    text = 'Voy a escribirlo.\n<tool_call>\n{"name": "write_file", "arguments": {"path": "src/a.py", "content": "x = {1: 2}\\n"}}\n</tool_call>'
    clean, calls = extract_text_tool_calls(text)
    assert clean == "Voy a escribirlo." and len(calls) == 1
    assert calls[0].name == "write_file" and calls[0].arguments == {"path": "src/a.py", "content": "x = {1: 2}\n"}
    assert calls[0].error is None and calls[0].id == "call_0"


def test_extract_text_tool_calls_ignores_other_json_and_garbage():
    for text in ["", "sin llamadas", '{"name": "x"}', '{"name": "x", "arguments": {}, "extra": 1}', "{roto", 'Informe: {"a": 1} y [1, 2]']:
        clean, calls = extract_text_tool_calls(text)
        assert calls == [] and clean == text


def test_extract_text_tool_calls_multiple_and_bad_arguments():
    text = 'ruido {"name": "a", "arguments": {"p": 1}} luego {"name": "b", "arguments": "no json"}'
    clean, calls = extract_text_tool_calls(text, first_index=2)
    assert [c.name for c in calls] == ["a", "b"] and calls[0].id == "call_2" and calls[1].id == "call_3"
    assert calls[1].error and "argumentos" in calls[1].error and clean == "ruido  luego"


def test_providers_rescue_text_calls_without_duplicating_structured_ones(server):
    garbage = 'xx\n{"name": "list_dir", "arguments": {"path": "src"}}\n<tool_call>{"name": "read_file", "arguments": {"path": "src/a.py"}}</tool_call>'
    server.responses.append(
        native_reply(garbage, tool_calls=[{"function": {"name": "list_dir", "arguments": {"path": "src"}}}])
    )
    reply = OllamaProvider(server.base_url, "m").chat([])
    assert [(c.name, c.arguments) for c in reply.tool_calls] == [("list_dir", {"path": "src"}), ("read_file", {"path": "src/a.py"})]
    assert reply.content == "xx"
    server.responses.append(text_reply('{"name": "grep", "arguments": {"pattern": "f"}}'))
    reply = OpenAICompatProvider(server.base_url, "m").chat([])
    assert [c.name for c in reply.tool_calls] == ["grep"] and reply.content == ""


# -- tiempo máximo de reloj por petición y por ejecución -----------------------------------------------------
@pytest.mark.parametrize("kind", ["slow", "trickle"])
def test_request_is_cancelled_at_wall_clock_deadline(server, kind):
    """El límite cubre respuestas lentas y también las que llegan a cuentagotas (el timeout de socket no)."""
    import time

    server.responses.append((kind, ""))
    t0 = time.monotonic()
    with pytest.raises(ProviderTimeout, match="no respondió en 1 s"):
        OpenAICompatProvider(server.base_url, "m", timeout=60).chat([], timeout=1)
    assert time.monotonic() - t0 < 5
    server.responses.append((kind, ""))
    t0 = time.monotonic()
    with pytest.raises(ProviderTimeout):
        OllamaProvider(server.base_url, "m", timeout=1).chat([])  # el límite propio también es de reloj
    assert time.monotonic() - t0 < 5


def test_timeout_argument_is_capped_by_provider_timeout(server):
    server.responses.append(text_reply("hola"))
    assert OpenAICompatProvider(server.base_url, "m", timeout=5).chat([], timeout=100).content == "hola"
    assert issubclass(ProviderTimeout, ProviderError)


def test_loop_total_time_limit_cancels_the_request_in_flight(server, tmp_path):
    import time

    project = make_project(tmp_path / "proj")
    server.responses.append(("trickle", ""))
    t0 = time.monotonic()
    result = run_agent(
        OpenAICompatProvider(server.base_url, "m", timeout=60), "S", "T", build_tools(), Policy(root=project),
        max_seconds=1.0, log_dir=tmp_path / "logs",
    )
    assert time.monotonic() - t0 < 6
    assert result.stop_reason == "timeout" and "petición cancelada" in result.error and result.steps == 1
    last = [json.loads(line) for line in result.log_path.read_text(encoding="utf-8").splitlines()][-1]
    assert last["event"] == "end" and last["stop_reason"] == "timeout"


def test_loop_passes_remaining_time_to_the_provider(tmp_path):
    project = make_project(tmp_path / "proj")
    provider = ScriptedProvider([AssistantMessage.calls(("list_dir", {"path": "src"})), "Hecho"])
    run_agent(provider, "S", "T", build_tools(), Policy(root=project), max_seconds=50, log_dir=False)
    assert 0 < provider.timeouts[1] <= provider.timeouts[0] <= 50
    unlimited = ScriptedProvider(["Hecho"])
    run_agent(unlimited, "S", "T", build_tools(), Policy(root=project), max_seconds=None, log_dir=False)
    assert unlimited.timeouts == [None]


def test_finish_reason_is_reported(server):
    server.responses.append((200, {"choices": [{"finish_reason": "length", "message": {"content": "cortado"}}]}))
    assert OpenAICompatProvider(server.base_url, "m").chat([]).finish_reason == "length"
    body = native_reply("cortado")[1]
    body["done_reason"] = "length"
    server.responses.append((200, body))
    assert OllamaProvider(server.base_url, "m").chat([]).finish_reason == "length"

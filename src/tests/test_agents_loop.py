"""Tests del bucle del agente con ``ScriptedProvider`` (sin red)."""

import json

import pytest
from agents_helpers import make_project

from amp_agents.loop import run_agent
from amp_agents.providers import AssistantMessage, ProviderError, ScriptedProvider, ToolCall, Usage
from amp_agents.tools import Policy, build_tools

REPORT = "Hecho: nada\nVerificado: nada\nFallos: ninguno\nPendiente: nada"


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path / "proj")


def go(project, responses, tmp_path, **kwargs):
    provider = ScriptedProvider(responses)
    kwargs.setdefault("log_dir", tmp_path / "logs")
    result = run_agent(provider, "SISTEMA", "TAREA", build_tools(), Policy(root=project), **kwargs)
    return provider, result


def events(result):
    return [json.loads(line) for line in result.log_path.read_text(encoding="utf-8").splitlines()]


def test_tool_then_final_answer(project, tmp_path):
    provider, result = go(
        project,
        [AssistantMessage.calls(("read_file", {"path": "src/amp_gan/mod.py"})), AssistantMessage.text(REPORT)],
        tmp_path,
    )
    assert result.stop_reason == "final" and result.steps == 2
    assert result.report == REPORT and result.tools_called == ["read_file"]
    assert result.tool_errors == 0 and result.denied == [] and result.error is None
    second = provider.received[1]
    assert second[0] == {"role": "system", "content": "SISTEMA"} and second[1] == {"role": "user", "content": "TAREA"}
    assert second[2]["role"] == "assistant" and second[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert second[3]["role"] == "tool" and "def f(x):" in second[3]["content"]
    assert second[3]["tool_call_id"] == second[2]["tool_calls"][0]["id"]


def test_tokens_are_summed_and_none_without_usage(project, tmp_path):
    _, with_usage = go(
        project,
        [
            AssistantMessage(tool_calls=[ToolCall("a", "list_dir", {"path": "src"})], usage=Usage(100, 10)),
            AssistantMessage(content=REPORT, usage=Usage(150, 20)),
        ],
        tmp_path,
    )
    assert with_usage.tokens.prompt_tokens == 250 and with_usage.tokens.completion_tokens == 30
    assert with_usage.tokens.total_tokens == 280 and with_usage.last_prompt_tokens == 150
    _, without = go(project, [REPORT], tmp_path)
    assert without.tokens is None and without.last_prompt_tokens is None and without.steps == 1


def test_malformed_arguments_are_reported_to_the_model(project, tmp_path):
    bad = AssistantMessage(tool_calls=[ToolCall("x", "read_file", {}, error="los argumentos no son JSON válido")])
    provider, result = go(project, [bad, AssistantMessage.text(REPORT)], tmp_path)
    assert result.stop_reason == "final" and result.tool_errors == 1 and result.denied == []
    tool_msg = provider.received[1][-1]
    assert tool_msg["role"] == "tool" and "ERROR" in tool_msg["content"] and "JSON válido" in tool_msg["content"]


def test_tool_error_does_not_stop_the_loop(project, tmp_path):
    provider, result = go(
        project,
        [
            AssistantMessage.calls(("read_file", {"path": "src/no_existe.py"}), ("no_existe", {}), ("read_file", {"foo": 1})),
            AssistantMessage.calls(("read_file", {"path": "src/amp_gan/mod.py"})),
            AssistantMessage.text(REPORT),
        ],
        tmp_path,
    )
    assert result.stop_reason == "final" and result.steps == 3 and result.tool_errors == 3
    msgs = [m for m in provider.received[1] if m["role"] == "tool"]
    assert len(msgs) == 3 and all(m["content"].startswith("ERROR") for m in msgs)
    assert "no existe" in msgs[1]["content"] and "list_dir" in msgs[1]["content"]
    assert "desconocidos" in msgs[2]["content"]


def test_max_steps_stops_with_reason(project, tmp_path):
    loop_forever = [AssistantMessage.calls(("list_dir", {"path": "src"})) for _ in range(10)]
    provider, result = go(project, loop_forever, tmp_path, max_steps=3)
    assert result.stop_reason == "max_steps" and result.steps == 3 and len(result.tools_called) == 3
    assert len(provider.received) == 3


def test_timeout_stops_before_calling_the_model(project, tmp_path):
    provider, result = go(project, [REPORT], tmp_path, max_seconds=-1)
    assert result.stop_reason == "timeout" and result.steps == 0 and provider.received == []


def test_provider_error_gives_error_reason(project, tmp_path):
    _, result = go(project, [ProviderError("el servidor no responde en http://x")], tmp_path)
    assert result.stop_reason == "error" and "no responde" in result.error and result.steps == 1


def test_denied_call_is_rejected_logged_and_does_not_write(project, tmp_path):
    attack = AssistantMessage.calls(
        ("write_file", {"path": ".git/hooks/pre-commit", "content": "pwned"}),
        ("read_file", {"path": ".env"}),
        ("read_file", {"path": "../../.ssh/id_rsa"}),
    )
    provider, result = go(project, [attack, AssistantMessage.text(REPORT)], tmp_path)
    assert result.stop_reason == "final" and len(result.denied) == 3 and result.tool_errors == 3
    assert not (project / ".git" / "hooks").exists()
    assert all("SECRETO" not in m["content"] for m in provider.received[1])
    logged = [e for e in events(result) if e["event"] == "tool"]
    assert [e["denied"] for e in logged] == [True, True, True]
    assert logged[0]["arguments"]["path"] == ".git/hooks/pre-commit"


def test_jsonl_log_records_every_message_and_call(project, tmp_path):
    _, result = go(
        project,
        [AssistantMessage.calls(("grep", {"pattern": "def f", "path": "src"})), AssistantMessage.text(REPORT)],
        tmp_path,
        run_label="mi prueba/1",
        metadata={"x": 1},
    )
    assert result.log_path.parent == tmp_path / "logs" and "mi_prueba_1" in result.log_path.name
    kinds = [e["event"] for e in events(result)]
    assert kinds == ["start", "assistant", "tool", "assistant", "end"]
    start, assistant, tool, _, end = events(result)
    assert start["system_prompt"] == "SISTEMA" and start["task"] == "TAREA" and start["metadata"] == {"x": 1}
    assert assistant["tool_calls"][0]["name"] == "grep"
    assert tool["name"] == "grep" and tool["ok"] and "mod.py" in tool["result"]
    assert end["stop_reason"] == "final" and end["report"] == REPORT and end["tools_called"] == ["grep"]


def test_default_log_dir_is_agent_runs_in_project_root(project):
    result = run_agent(ScriptedProvider([REPORT]), "S", "T", build_tools(), Policy(root=project))
    assert result.log_path.parent == project.resolve() / ".agent_runs" and result.log_path.exists()


def test_logging_can_be_disabled(project):
    result = run_agent(ScriptedProvider([REPORT]), "S", "T", build_tools(), Policy(root=project), log_dir=False)
    assert result.log_path is None and not (project / ".agent_runs").exists()


def test_empty_answer_gets_one_nudge_then_final(project, tmp_path):
    provider, result = go(project, ["", REPORT], tmp_path)
    assert result.stop_reason == "final" and result.report == REPORT and result.steps == 2
    assert provider.received[1][-1]["role"] == "user"
    _, twice = go(project, ["", ""], tmp_path)
    assert twice.stop_reason == "final" and twice.report == "" and twice.steps == 2


def test_long_tool_results_are_truncated_for_the_model(project, tmp_path):
    (project / "src" / "grande.txt").write_text("x" * 50_000, encoding="utf-8")
    provider, result = go(
        project,
        [AssistantMessage.calls(("read_file", {"path": "src/grande.txt", "max_bytes": 50000})), AssistantMessage.text(REPORT)],
        tmp_path,
        max_result_chars=500,
    )
    content = provider.received[1][-1]["content"]
    assert len(content) < 700 and "recortado" in content


def test_scripted_provider_exhausted_is_error_and_exceptions_propagate(project, tmp_path):
    _, result = go(project, [AssistantMessage.calls(("list_dir", {"path": "src"}))], tmp_path)
    assert result.stop_reason == "error" and "agotado" in result.error
    assert ScriptedProvider(["a"]).chat([]).content == "a"


def test_write_then_pytest_flow_like_the_live_task(project, tmp_path):
    module = "def percent_change(old, new):\n    if old == 0:\n        raise ValueError('old no puede ser 0')\n    return (new - old) / old * 100\n"
    test = "from amp_gan.percent_change import percent_change\n\n\ndef test_pc():\n    assert percent_change(50, 75) == 50\n"
    _, result = go(
        project,
        [
            AssistantMessage.calls(
                ("write_file", {"path": "src/amp_gan/percent_change.py", "content": module}),
                ("write_file", {"path": "src/tests/test_percent_change.py", "content": test}),
            ),
            AssistantMessage.calls(("run_pytest", {"path": "tests/test_percent_change.py"})),
            AssistantMessage.text(REPORT),
        ],
        tmp_path,
    )
    assert result.stop_reason == "final" and result.tool_errors == 0
    tool_events = [e for e in events(result) if e["event"] == "tool"]
    assert "1 passed" in tool_events[-1]["result"]

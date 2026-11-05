"""Tests de los cinco roles acotados: configuración, restricción de herramientas, modo single_shot y CLI (sin red)."""

import json
import shutil
from pathlib import Path

import pytest
from agents_helpers import make_project, write

from amp_agents import agents as A
from amp_agents import run as R
from amp_agents.agents import ConfigError, build_system_prompt, get_agent_config
from amp_agents.loop import run_agent, run_single_shot
from amp_agents.providers import AssistantMessage, ProviderError, ProviderTimeout, ScriptedProvider
from amp_agents.tools import Policy

SRC = Path(__file__).resolve().parents[1]
REAL_ROOT = SRC.parent
ROLES = ("locator", "doc_checker", "code_reviewer", "summarizer", "retriever")
EXPECTED = {
    "executor": ("agent", "qwen3:4b"),
    "locator": ("agent", "llama3.2:3b"),
    "doc_checker": ("agent", "phi4-mini"),
    "code_reviewer": ("agent", "qwen2.5-coder:7b"),
    "summarizer": ("single_shot", "gemma3:4b"),
    "retriever": ("embedding", "nomic-embed-text"),
}


def project_with_roles(tmp_path: Path) -> Path:
    project = make_project(tmp_path / "proj")
    for role in ROLES:
        shutil.copyfile(REAL_ROOT / "context" / "agents" / f"{role}.md", project / "context" / "agents" / f"{role}.md")
    write(project / "context" / "constitution" / "tech_stack.md", "# Stack\nEl backend se elige con AMP_GAN_BACKEND.\n")
    return project


class RecordingProvider(ScriptedProvider):
    """ScriptedProvider que además guarda las herramientas que recibe en cada llamada."""

    def __init__(self, responses):
        super().__init__(responses)
        self.tools_seen = []

    def chat(self, messages, tools=None, timeout=None):
        self.tools_seen.append(tools)
        return super().chat(messages, tools, timeout)


# -- configuración de los cinco roles -----------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(EXPECTED))
def test_every_role_has_mode_model_and_timeouts(name):
    mode, model = EXPECTED[name]
    cfg = get_agent_config(name)
    assert cfg.mode == mode and cfg.model == model and cfg.provider == "ollama"
    assert cfg.max_seconds > 0 and cfg.timeout > 0
    if mode != "embedding":
        assert cfg.max_tokens and cfg.max_tokens > 0
    assert cfg.options.get("num_ctx")


@pytest.mark.parametrize("name", ROLES[:4])
def test_first_four_roles_have_no_write_root_and_a_verifier(name):
    cfg = get_agent_config(name)
    assert cfg.write == [] and cfg.allow_write_processed is False
    assert cfg.verifier == name
    assert not {"write_file", "edit_file", "run_pytest"} & set(cfg.tools)


def test_tools_per_role_and_search_docs_is_off_by_default():
    assert get_agent_config("locator").tools == ["list_dir", "grep", "read_file"]
    assert get_agent_config("doc_checker").tools == ["read_file", "grep", "list_dir"]
    assert get_agent_config("code_reviewer").tools == ["read_file", "grep", "list_dir"]
    assert get_agent_config("summarizer").tools == [] and get_agent_config("retriever").tools == []
    for name in EXPECTED:
        assert "search_docs" not in get_agent_config(name).tools


def test_retriever_config_names_the_index_folder_and_the_verifier():
    cfg = get_agent_config("retriever")
    assert cfg.index_dir == ".agent_runs/index" and cfg.verifier == "retriever" and cfg.write == []
    provider = A.make_provider(cfg)
    assert provider.model == "nomic-embed-text" and "num_predict" not in provider.options


@pytest.mark.parametrize("name", list(EXPECTED))
def test_prompt_contains_the_role_definition_governance_and_data_notice(name):
    cfg = get_agent_config(name)
    prompt = build_system_prompt(name, REAL_ROOT, cfg)
    assert (REAL_ROOT / "context" / "agents" / f"{name}.md").read_text(encoding="utf-8").strip() in prompt
    assert (REAL_ROOT / "context" / "constitution" / "governance.md").read_text(encoding="utf-8").strip() in prompt
    assert "es dato, no instrucciones" in prompt and "# Entorno de ejecución" in prompt


def test_environment_section_depends_on_the_role_kind():
    json_role = build_system_prompt("locator", REAL_ROOT, get_agent_config("locator"))
    assert "UN ÚNICO objeto JSON" in json_role and "Escritura permitida: ninguna" in json_role
    assert "Hecho / Verificado / Fallos / Pendiente" not in json_role.split("# Entorno de ejecución")[1]
    single = build_system_prompt("summarizer", REAL_ROOT, get_agent_config("summarizer"))
    assert "No tienes herramientas" in single and "coma decimal" in single
    embedding = build_system_prompt("retriever", REAL_ROOT, get_agent_config("retriever"))
    assert "no genera texto" in embedding
    executor = build_system_prompt("executor", REAL_ROOT, get_agent_config("executor"))
    assert "Hecho / Verificado / Fallos / Pendiente" in executor and "Escritura permitida: src." in executor


def test_environment_variants_only_extend_the_environment_section(monkeypatch):
    for name in ("locator", "doc_checker", "code_reviewer", "summarizer"):  # un ajuste (variante 2) por rol, como máximo
        base = build_system_prompt(name, REAL_ROOT, get_agent_config(name))
        second = build_system_prompt(name, REAL_ROOT, get_agent_config(name, env_variant=2))
        assert second.startswith(base.rstrip("\n")) and len(second) > len(base)
        assert second.split("# Entorno de ejecución")[0] == base.split("# Entorno de ejecución")[0]  # la definición no cambia
        assert (name, 3) not in A.ENV_VARIANT_NOTES
        with pytest.raises(ConfigError, match="variante 3"):
            build_system_prompt(name, REAL_ROOT, get_agent_config(name, env_variant=3))
    assert ("retriever", 2) not in A.ENV_VARIANT_NOTES and ("executor", 2) not in A.ENV_VARIANT_NOTES
    base = build_system_prompt("locator", REAL_ROOT, get_agent_config("locator"))
    monkeypatch.setitem(A.ENV_VARIANT_NOTES, ("locator", 3), "- Nota extra de prueba.")
    third = build_system_prompt("locator", REAL_ROOT, get_agent_config("locator", env_variant=3))
    assert third.startswith(base.rstrip("\n")) and third.rstrip().endswith("- Nota extra de prueba.")
    with pytest.raises(ConfigError, match="env_variant"):
        get_agent_config("locator", env_variant=0)


def test_config_validation_of_modes_and_tools(tmp_path):
    def cfg(body: str):
        write(tmp_path / "c.toml", '[agents.x]\nmodel = "m"\n' + body)
        return get_agent_config("x", tmp_path / "c.toml")

    with pytest.raises(ConfigError, match="modo"):
        cfg('mode = "raro"\ntools = ["read_file"]\n')
    with pytest.raises(ConfigError, match="no admite herramientas"):
        cfg('mode = "single_shot"\ntools = ["read_file"]\n')
    with pytest.raises(ConfigError, match="herramientas desconocidas"):
        cfg('tools = ["read_file", "formatear_disco"]\n')
    with pytest.raises(ConfigError, match="no tiene herramientas"):
        cfg("tools = []\n")
    with pytest.raises(ConfigError, match="solo funciona con el proveedor"):
        cfg('mode = "embedding"\nprovider = "openai_compat"\n')
    with pytest.raises(ConfigError, match="verificador"):
        cfg('tools = ["read_file"]\nverifier = "inexistente"\n')
    with pytest.raises(ConfigError, match="search_backend"):
        cfg('tools = ["read_file"]\nsearch_backend = "magia"\n')
    assert cfg('mode = "single_shot"\n').mode == "single_shot"
    assert [t.name for t in A.make_tools(cfg('tools = ["search_docs", "read_file"]\n'))] == ["search_docs", "read_file"]


# -- restricción real de herramientas -------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ROLES[:3])
def test_read_roles_cannot_write_even_if_the_model_asks(name, tmp_path):
    project = project_with_roles(tmp_path)
    cfg = get_agent_config(name)
    policy = A.make_policy(cfg, project)
    for target in ("src/amp_gan/nuevo.py", "src/tests/test_nuevo.py", "context/data/processed/x.csv"):
        with pytest.raises(Exception, match="denegado"):
            policy.check(target, "write")
    assert [t.name for t in A.make_tools(cfg)] == cfg.tools
    provider = RecordingProvider(
        [
            AssistantMessage.calls(
                ("write_file", {"path": "src/amp_gan/nuevo.py", "content": "x = 1\n"}),
                ("edit_file", {"path": "src/amp_gan/mod.py", "old_string": "x + 1", "new_string": "x + 2"}),
                ("run_pytest", {}),
            ),
            '{"respuestas": []}',
        ]
    )
    before = (project / "src" / "amp_gan" / "mod.py").read_text(encoding="utf-8")
    result = run_agent(provider, "sistema", "tarea", A.make_tools(cfg), policy, max_steps=5, log_dir=tmp_path / "logs")
    assert result.stop_reason == "final" and result.tool_errors == 3
    assert not (project / "src" / "amp_gan" / "nuevo.py").exists()
    assert (project / "src" / "amp_gan" / "mod.py").read_text(encoding="utf-8") == before
    offered = {d["function"]["name"] for d in provider.tools_seen[0]}
    assert offered == set(cfg.tools)
    assert "no existe" in provider.received[1][-1]["content"]


def test_policy_write_roots_follow_the_tools_list_not_the_toml_roots(tmp_path):
    project = project_with_roles(tmp_path)
    no_write = A.AgentConfig(name="x", model="m", tools=["read_file", "grep"], write=["src"], allow_write_processed=True)
    policy = A.make_policy(no_write, project)
    assert tuple(policy.write_roots) == () and policy.allow_write_processed is False
    assert A.effective_write_roots(no_write) == []
    with pytest.raises(Exception, match="denegado"):
        policy.check("src/amp_gan/mod.py", "write")
    writer = A.AgentConfig(name="x", model="m", tools=["read_file", "edit_file"], write=["src"])
    assert A.make_policy(writer, project).check("src/amp_gan/mod.py", "write")
    assert isinstance(A.make_policy(writer, project), Policy)


def test_executor_keeps_its_behaviour():
    cfg = get_agent_config("executor")
    assert cfg.mode == "agent" and cfg.tools == ["list_dir", "read_file", "grep", "write_file", "edit_file", "run_pytest"]
    assert A.effective_write_roots(cfg) == ["src"] and cfg.verifier == ""


def test_grep_with_empty_glob_means_all_files(tmp_path):
    project = project_with_roles(tmp_path)
    cfg = get_agent_config("locator")
    grep = {t.name: t for t in A.make_tools(cfg)}["grep"]
    out = grep.run(A.make_policy(cfg, project), {"pattern": "def f", "path": "src", "glob": ""})
    assert out.ok and "src/amp_gan/mod.py:1:def f(x):" in out.content


# -- modo single_shot ---------------------------------------------------------------------------------------------------
def test_single_shot_sends_one_request_without_tools_and_logs_it(tmp_path):
    project = project_with_roles(tmp_path)
    provider = RecordingProvider(["Resumen en castellano."])
    result = R.run_task("summarizer", "Resume esto: 1 2 3", root=project, log_dir=tmp_path / "logs", provider_instance=provider)
    assert result.stop_reason == "final" and result.steps == 1 and result.report == "Resumen en castellano."
    assert result.tools_called == [] and result.finish_reason == ""
    assert provider.tools_seen == [None] and len(provider.received) == 1
    system, user = provider.received[0]
    assert system["role"] == "system" and "# Agente: Resumidor" in system["content"]
    assert user == {"role": "user", "content": "Resume esto: 1 2 3"}
    assert provider.timeouts == [get_agent_config("summarizer").max_seconds]
    lines = result.log_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["start", "assistant", "end"]
    start = json.loads(lines[0])
    assert start["mode"] == "single_shot" and start["tools"] == []


def test_single_shot_ignores_tool_calls_and_reports_usage_and_length(tmp_path):
    from amp_agents.providers import Usage

    reply = AssistantMessage(content="corto", usage=Usage(100, 20), finish_reason="length")
    result = run_single_shot(ScriptedProvider([reply]), "s", "t", max_seconds=5, log_dir=tmp_path)
    assert result.stop_reason == "length" and result.tokens.total_tokens == 120 and result.last_prompt_tokens == 100
    assert result.finish_reason == "length"
    asked = AssistantMessage.calls(("read_file", {"path": "x"}), content="texto")
    result = run_single_shot(ScriptedProvider([asked]), "s", "t", max_seconds=5, log_dir=False)
    assert result.stop_reason == "final" and result.tools_called == [] and result.log_path is None


@pytest.mark.parametrize(
    ("reply", "reason", "fragment"),
    [("   ", "empty", ""), (ProviderTimeout("se agotó"), "timeout", "se agotó"), (ProviderError("servidor caído"), "error", "servidor caído")],
)
def test_single_shot_stop_reasons(reply, reason, fragment, tmp_path):
    result = run_single_shot(ScriptedProvider([reply]), "s", "t", max_seconds=5, log_dir=tmp_path)
    assert result.stop_reason == reason
    assert fragment in (result.error or "")
    end = json.loads(result.log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert end["event"] == "end" and end["stop_reason"] == reason


def test_run_task_refuses_the_embedding_role(tmp_path):
    with pytest.raises(ConfigError, match="embeddings"):
        R.run_task("retriever", "consulta", root=project_with_roles(tmp_path))


# -- CLI de los cinco roles -------------------------------------------------------------------------------------------------
def test_cli_locator_verifies_the_answer(tmp_path, monkeypatch, capsys):
    project = project_with_roles(tmp_path)
    answer = json.dumps({"respuestas": [{"ruta": "src/amp_gan/mod.py", "linea": 1, "texto": "def f(x):"}]})
    scripted = ScriptedProvider([AssistantMessage.calls(("grep", {"pattern": "def f", "path": "src"})), answer])
    monkeypatch.setattr(R, "make_provider", lambda config: scripted)
    code = R.main(["--agent", "locator", "¿Dónde se define f?", "--root", str(project), "--log-dir", str(tmp_path / "l")])
    out = capsys.readouterr()
    assert code == 0 and "respuestas" in out.out and "verificación: válida" in out.err
    assert scripted.received[0][1]["content"] == "¿Dónde se define f?"


def test_cli_locator_with_a_made_up_line_exits_1(tmp_path, monkeypatch, capsys):
    project = project_with_roles(tmp_path)
    answer = json.dumps({"respuestas": [{"ruta": "src/amp_gan/mod.py", "linea": 99, "texto": "def f(x):"}]})
    monkeypatch.setattr(R, "make_provider", lambda config: ScriptedProvider([answer]))
    code = R.main(["--agent", "locator", "pregunta", "--root", str(project), "--log-dir", str(tmp_path / "l")])
    assert code == 1 and "NO válida" in capsys.readouterr().err


def test_cli_doc_checker_and_code_reviewer_build_their_messages(tmp_path, monkeypatch, capsys):
    project = project_with_roles(tmp_path)
    write(project / "context" / "a.md", "uno\n")
    write(project / "context" / "b.md", "dos\n")
    answer = json.dumps(
        {"incoherencias": [{"descripcion": "d", "evidencia_a": {"ruta": "context/a.md", "cita": "uno"}, "evidencia_b": {"ruta": "context/b.md", "cita": "dos"}}]}
    )
    scripted = ScriptedProvider([answer])
    monkeypatch.setattr(R, "make_provider", lambda config: scripted)
    args = ["--agent", "doc_checker", "context/a.md", "context/b.md", "--note", "cifras", "--root", str(project), "--log-dir", str(tmp_path / "l")]
    assert R.main(args) == 0
    task = scripted.received[0][1]["content"]
    assert "`context/a.md`, `context/b.md`" in task and "Aspecto a comprobar: cifras." in task
    review = json.dumps({"hallazgos": [{"linea": 2, "cita": "    return x + 1", "problema": "p", "gravedad": "baja"}]})
    scripted = ScriptedProvider([review])
    monkeypatch.setattr(R, "make_provider", lambda config: scripted)
    assert R.main(["--agent", "code_reviewer", "src/amp_gan/mod.py", "--root", str(project), "--log-dir", str(tmp_path / "l")]) == 0
    assert "Revisa el fichero `src/amp_gan/mod.py`" in scripted.received[0][1]["content"]
    capsys.readouterr()


def test_cli_summarizer_reads_files_with_the_policy_and_checks_numbers(tmp_path, monkeypatch, capsys):
    project = project_with_roles(tmp_path)
    write(project / "context" / "data" / "processed" / "m.csv", "epoch,loss\n200,0.4321\n")
    scripted = ScriptedProvider(["El experimento llegó a la época 200 con una pérdida de 0,43 en el conjunto de datos."])
    monkeypatch.setattr(R, "make_provider", lambda config: scripted)
    code = R.main(["--agent", "summarizer", "context/data/processed/m.csv", "--root", str(project), "--log-dir", str(tmp_path / "l")])
    err = capsys.readouterr().err
    assert code == 0 and "verificación: válida" in err
    sent = scripted.received[0][1]["content"]
    assert "### context/data/processed/m.csv" in sent and "0.4321" in sent
    scripted = ScriptedProvider(["La pérdida fue de 0,99 en el conjunto de datos de la época 200 y se mantuvo."])
    monkeypatch.setattr(R, "make_provider", lambda config: scripted)
    code = R.main(["--agent", "summarizer", "context/data/processed/m.csv", "--root", str(project), "--log-dir", str(tmp_path / "l")])
    assert code == 1 and "cifras que no están en la entrada" in capsys.readouterr().err
    assert R.main(["--agent", "summarizer", ".env", "--root", str(project)]) == 2  # la política deniega leerlo
    assert R.main(["--agent", "summarizer", "--root", str(project)]) == 2
    capsys.readouterr()


def test_cli_cannot_make_the_summarizer_read_outside_its_roots(tmp_path, capsys):
    project = project_with_roles(tmp_path)
    assert R.main(["--agent", "summarizer", "src/amp_gan/mod.py", "--root", str(project)]) == 2
    assert "acceso denegado" in capsys.readouterr().err


def test_cli_retriever_with_a_fake_embedder(tmp_path, monkeypatch, capsys):
    from fake_embedder import FakeEmbedder

    project = project_with_roles(tmp_path)
    monkeypatch.setattr(R, "make_provider", lambda config: FakeEmbedder())
    code = R.main(["--agent", "retriever", "backend AMP_GAN_BACKEND elige", "--embedding", "--k", "2", "--root", str(project)])
    out = capsys.readouterr()
    hits = json.loads(out.out)
    assert code == 0 and len(hits) == 2 and hits[0]["ruta"] == "context/constitution/tech_stack.md" and hits[0]["linea"] == 1
    assert (project / ".agent_runs" / "index" / "index.json").is_file()
    code = R.main(["--agent", "retriever", "backend AMP_GAN_BACKEND", "--lexical", "--root", str(project)])
    assert code == 0 and json.loads(capsys.readouterr().out)[0]["ruta"].endswith("tech_stack.md")
    assert R.main(["--agent", "retriever", "--root", str(project)]) == 2


def test_cli_executor_still_requires_task_file(tmp_path, capsys):
    assert R.main(["--agent", "executor", "--root", str(project_with_roles(tmp_path))]) == 2
    assert "--task-file" in capsys.readouterr().err


# -- independencia ----------------------------------------------------------------------------------------------------------
def test_new_modules_do_not_import_heavy_frameworks_or_amp_gan():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import amp_agents.retrieval, amp_agents.verifiers, amp_agents.eval_roles, amp_agents.run, amp_agents.agents\n"
        "bad = [m for m in ('keras', 'torch', 'tensorflow', 'amp_gan', 'numpy', 'pandas') if m in sys.modules]\n"
        "print('BAD', bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=SRC, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def test_new_sources_have_no_forbidden_mentions():
    forbidden = ["cla" + "ude", "anthro" + "pic", "asistente de " + "ia", "/ho" + "me/"]  # troceadas: el test global revisa este fichero
    files = [SRC / "amp_agents" / n for n in ("retrieval.py", "verifiers.py", "eval_roles.py", "agents.toml")]
    files += [SRC / "tests" / n for n in ("fake_embedder.py", "test_agents_retrieval.py", "test_agents_verifiers.py", "test_agents_eval_roles.py")]
    for file in files:
        text = file.read_text(encoding="utf-8").lower().replace(".claude", "")
        for word in forbidden:
            assert word not in text, f"{file.name} menciona {word!r}"


# -- recuperador: BM25 por defecto, embeddings explícitos ---------------------------------------------------------------------
def test_cli_retriever_defaults_to_bm25_and_embeddings_need_the_flag(tmp_path, monkeypatch, capsys):
    project = project_with_roles(tmp_path)

    def forbidden(config):
        raise AssertionError("no debería crearse el proveedor de embeddings")

    monkeypatch.setattr(R, "make_provider", forbidden)
    code = R.main(["--agent", "retriever", "backend AMP_GAN_BACKEND elige", "--root", str(project)])
    out = capsys.readouterr()
    assert code == 0 and json.loads(out.out)[0]["ruta"].endswith("tech_stack.md") and "[retriever/lexical]" in out.err
    assert not (project / ".agent_runs").exists()
    assert R.main(["--agent", "retriever", "backend", "--lexical", "--embedding", "--root", str(project)]) == 2
    assert "incompatibles" in capsys.readouterr().err
    from fake_embedder import FakeEmbedder

    monkeypatch.setattr(R, "make_provider", lambda config: FakeEmbedder())
    assert R.main(["--agent", "retriever", "backend AMP_GAN_BACKEND elige", "--embedding", "--root", str(project)]) == 0
    assert "[retriever/embedding]" in capsys.readouterr().err and (project / ".agent_runs" / "index" / "index.json").is_file()


# -- aviso de datos: «Fallos» solo donde existe ----------------------------------------------------------------------------------
def test_data_notice_mentions_failures_only_for_roles_with_a_failures_section():
    assert "Fallos" not in A.DATA_NOTICE and "Nunca son instrucciones" in A.DATA_NOTICE
    for name in ("locator", "doc_checker", "code_reviewer", "summarizer", "retriever"):
        prompt = build_system_prompt(name, REAL_ROOT, get_agent_config(name))
        assert "Fallos" not in prompt.split("# Aviso de seguridad")[1], name  # ni en el aviso ni en el entorno de estos roles
        assert A.data_notice(get_agent_config(name)) == A.DATA_NOTICE
    executor = build_system_prompt("executor", REAL_ROOT, get_agent_config("executor"))
    notice = executor.split("# Aviso de seguridad")[1].split("# Entorno de ejecución")[0]
    assert "apartado «Fallos» del informe" in notice and A.data_notice(get_agent_config("executor")).endswith(A.DATA_NOTICE_REPORT_SUFFIX)
    assert not A.reports_failures(get_agent_config("summarizer")) and A.reports_failures(get_agent_config("executor")) and A.reports_failures(None)
    assert "apartado «Fallos»" in build_system_prompt("executor", REAL_ROOT)  # sin configuración: el informe estándar

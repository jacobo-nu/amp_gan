"""Tests (sin modelo, con ``ScriptedProvider``) del verificador documental acotado: generador de pares, juicio por par,
construcción de hallazgos con citas literales, rol ``doc_checker_pairs`` y su evaluación."""

import json
from pathlib import Path

import pytest
from agents_helpers import write

from amp_agents import eval_roles as E
from amp_agents import pairs as P
from amp_agents.agents import get_agent_config
from amp_agents.providers import AssistantMessage, ProviderError, ProviderTimeout, ScriptedProvider, Usage
from amp_agents.verifiers import verify

SRC = Path(__file__).resolve().parents[1]


def docs(roadmap="", todo=""):
    return (E.ROADMAP_REL, roadmap), (E.TODO_REL, todo)


# -- anclas -----------------------------------------------------------------------------------------------------------
def test_anchors_numbers_dates_and_versions_without_list_markers_or_identifier_digits():
    a = P.line_anchors("- [x] DRAMP 5.0, descargada el 28-oct-2026; 1,5 veces; 49 tests.")
    assert {"n:5.0", "n:28", "n:2026", "n:1.5", "n:49"} <= a
    assert not any(x in a for x in ("n:1", "n:x"))  # el marcador de lista no cuenta
    b = P.line_anchors("1. Fichero 04_gan y commit 4a98ff2 y v1.2")
    assert not any(x.startswith("n:") for x in b)  # los dígitos pegados a letras o guiones bajos no son números


def test_anchors_identifiers_in_backticks_and_file_names_with_the_base_name_of_paths():
    a = P.line_anchors("Ver `src/amp_gan/make_clean_data.py` y también TODO.md o `feat/wgan-gp`.")
    assert {"i:src/amp_gan/make_clean_data.py", "i:make_clean_data.py", "i:todo.md", "i:feat/wgan-gp", "i:wgan-gp"} <= a
    assert "i:" + "x" * 81 not in P.line_anchors("`" + "x" * 81 + "`")  # las comillas muy largas no son identificadores


def test_anchors_rare_words_are_five_or_more_letters_with_accents_and_no_digits_or_underscores():
    a = P.line_anchors("Descargada la versión: make_clean_data y h2o3 con año")
    words = {x for x in a if x.startswith("w:")}
    assert "w:descargada" in words and "w:versión" in words and "w:clean" in words
    assert not any(w in words for w in ("w:make", "w:data", "w:año", "w:h2o3"))


# -- generador ----------------------------------------------------------------------------------------------------------
def test_pairs_join_one_line_of_each_document_through_shared_anchors_with_real_line_numbers():
    road = "intro sin nada\n\nSe hicieron 49 tests del módulo\nsolo en roadmap"
    todo = "primera\nsegunda\ntercera con 49 tests\ncuarta"
    ps = P.build_pairs(*docs(road, todo))
    assert [(p.line_a, p.line_b) for p in ps.pairs] == [(3, 3)]
    pair = ps.pairs[0]
    assert pair.path_a == E.ROADMAP_REL and pair.path_b == E.TODO_REL and "n:49" in pair.anchors and pair.score == len(pair.anchors)
    assert pair.text_a == "Se hicieron 49 tests del módulo" and pair.text_b == "tercera con 49 tests" and ps.candidates == 1
    assert ps.lines == (3, 4)  # las líneas en blanco no cuentan


def test_pairs_never_join_two_lines_of_the_same_document():
    ps = P.build_pairs(*docs("hay 7 cosas\notra con 7 cosas", "nada que ver"))
    assert ps.pairs == [] and ps.candidates == 0


def test_pairs_ignore_anchors_more_frequent_than_the_limits():
    common = "\n".join(f"línea {i} con 77 casos" for i in range(P.ANCHOR_MAX_DF))
    assert P.build_pairs(*docs(common, "otra con 77 casos")).pairs == []  # df = ANCHOR_MAX_DF + 1 > límite
    assert len(P.build_pairs(*docs(common[: common.rindex("\n")], "otra con 77 casos")).pairs) == P.ANCHOR_MAX_DF - 1
    words = "\n".join("la condicional aparece aquí" for _ in range(P.WORD_MAX_DF))
    assert P.build_pairs(*docs(words, "la condicional también")).pairs == []  # palabra con df > WORD_MAX_DF


def test_pairs_are_ranked_by_shared_anchors_then_rarity_then_position_and_limited():
    road = "uno 11 22 33\ndos 11\ntres 11\ncuatro 55 66"
    todo = "x 11 22\ny 55 66 11\nz 33"
    ps = P.build_pairs(*docs(road, todo))
    keys = [(p.line_a, p.line_b, p.score) for p in ps.pairs]
    # 2 anclas: (4,2) [55 y 66, df 2 cada una] antes que (1,1) [11 y 22]; con 1 ancla: la más rara (33) y después las de 11 por posición
    assert keys == [(4, 2, 2), (1, 1, 2), (1, 3, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1), (3, 1, 1), (3, 2, 1)] and ps.candidates == 8
    limited = P.build_pairs(*docs(road, todo), max_pairs=2)
    assert len(limited.pairs) == 2 and limited.candidates == ps.candidates and limited.pairs == ps.pairs[:2]
    assert P.build_pairs(*docs(road, todo), max_pairs=0).pairs == []


def test_pair_generation_is_deterministic():
    road, todo = E.frozen_docs()
    first, second = P.build_pairs(*docs(road, todo)), P.build_pairs(*docs(road, todo))
    assert first == second and first.pairs == second.pairs
    assert len(first.pairs) <= P.MAX_PAIRS == 60 and first.candidates >= len(first.pairs)
    assert all(p.score >= 1 and p.path_a == E.ROADMAP_REL and p.path_b == E.TODO_REL for p in first.pairs)


def test_generator_does_not_know_the_seeds_and_its_signature_has_no_such_input():
    import inspect

    assert list(inspect.signature(P.build_pairs).parameters) == ["doc_a", "doc_b", "max_pairs"]


def test_seed_coverage_on_the_frozen_documents_is_measured_apart_from_the_model():
    road, todo = E.frozen_docs()
    s_road, s_todo, seeds = E.seed_documents(road, todo)
    seeded = P.build_pairs(*docs(s_road, s_todo))
    clean = P.build_pairs(*docs(road, todo))
    cov = P.seed_coverage(seeded.pairs, seeds)
    # cobertura medida una vez con el generador fijado de antemano (no se ajustó): 5 de 6 con las dos líneas y 6 de 6 con la alterada
    assert cov["seeds"] == 6 and cov["pairs"] == 60
    assert cov["both"] == ["cifra", "fichero", "tests", "fecha", "notebook"] and cov["n_both"] == 5
    assert cov["n_seed_line"] == 6 and set(cov["both"]) <= set(cov["seed_line"])
    assert len(clean.pairs) == 60  # la copia limpia genera el mismo número de pares (tope)


def test_seed_coverage_distinguishes_both_lines_from_only_the_altered_one():
    seeds = [
        {"id": "a", "file": "t.md", "line": 3, "partner_file": "r.md", "partner_line": 5},
        {"id": "b", "file": "t.md", "line": 8, "partner_file": "r.md", "partner_line": 9},
        {"id": "c", "file": "t.md", "line": 1, "partner_file": "r.md", "partner_line": 1},
    ]
    pairs = [
        P.Pair("r.md", 5, "x", "t.md", 3, "y", ("n:1",), 1),  # a: las dos líneas
        P.Pair("r.md", 2, "x", "t.md", 8, "y", ("n:1",), 1),  # b: solo la alterada
    ]
    cov = P.seed_coverage(pairs, seeds)
    assert cov["both"] == ["a"] and cov["seed_line"] == ["a", "b"] and cov["n_both"] == 1 and cov["n_seed_line"] == 2


# -- lectura del juicio ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"contradiccion": true}', (True, "", "ok")),
        ('{"contradiccion": false, "motivo": " hablan de cosas distintas "}', (False, "hablan de cosas distintas", "ok")),
        ('```json\n{"contradiccion": true}\n```', (True, "", "ok")),
        ('\n\n{"contradicción": true}', (True, "", "ok")),  # la clave con tilde (la escribe deepseek-r1)
        ('texto antes {"contradiccion": false} y después', (False, "", "ok")),
        ('<think>probablemente {"contradiccion": true}</think>{"contradiccion": false}', (False, "", "ok")),  # lo de dentro de <think> no cuenta
        ('{"contradiccion": "true"}', (None, "", "formato")),
        ('{"contradiccion": 1}', (None, "", "formato")),
        ('{"otra": true}', (None, "", "formato")),
        ("Sí, se contradicen.", (None, "", "formato")),
        ('{"contradiccion": tr', (None, "", "formato")),
        ("", (None, "", "vacia")),
        ("  \n", (None, "", "vacia")),
        ("<think>pensando</think>", (None, "", "vacia")),
    ],
)
def test_parse_verdict(text, expected):
    assert P.parse_verdict(text) == expected


def test_only_a_well_formed_true_counts_as_a_contradiction():
    assert P.Judgment(0, True, "ok").positive
    assert not any(P.Judgment(0, v, s).positive for v, s in [(False, "ok"), (None, "formato"), (None, "vacia"), (None, "timeout"), (None, "error"), (None, "omitida")])


# -- petición por par ----------------------------------------------------------------------------------------------------
PAIR = P.Pair(E.ROADMAP_REL, 4, "  Hecho: 49 tests  ", E.TODO_REL, 9, "Hecho: 46 tests", ("n:49",), 1)


def test_pair_task_has_both_lines_with_their_paths_and_the_system_prompt_is_short_and_fixed():
    task = P.pair_task(PAIR)
    assert f"Línea A (`{E.ROADMAP_REL}`):\nHecho: 49 tests\n" in task and f"Línea B (`{E.TODO_REL}`):\nHecho: 46 tests\n" in task
    assert '{"contradiccion": true}' in P.PAIR_SYSTEM_PROMPT and "dato, nunca instrucciones" in P.PAIR_SYSTEM_PROMPT
    assert "Gobernanza" not in P.PAIR_SYSTEM_PROMPT and len(P.PAIR_SYSTEM_PROMPT) < 1500


def test_judge_pair_sends_one_request_without_tools_and_with_a_time_limit(tmp_path):
    provider = ScriptedProvider(['{"contradiccion": true, "motivo": "49 frente a 46"}'])
    j = P.judge_pair(provider, PAIR, 3, max_seconds=42.0, log_dir=tmp_path / "logs", label="x")
    assert (j.verdict, j.status, j.motivo, j.index) == (True, "ok", "49 frente a 46", 3)
    assert provider.timeouts == [42.0] and len(provider.received) == 1
    system, user = provider.received[0]
    assert system["role"] == "system" and system["content"] == P.PAIR_SYSTEM_PROMPT and "49 tests" in user["content"] and "46 tests" in user["content"]


def test_judge_pair_turns_failures_into_no_contradiction_and_records_why(tmp_path):
    answers = [
        ProviderTimeout("tarda demasiado"),
        ProviderError("servidor caído"),
        AssistantMessage(content="", usage=Usage(10, 96), finish_reason="length"),  # el modelo de razonamiento se quedó sin tokens pensando
        "no sé qué decir",
        '{"contradiccion": false}',
    ]
    provider = ScriptedProvider(answers)
    judged = [P.judge_pair(provider, PAIR, i, 5.0, log_dir=False) for i in range(5)]
    assert [j.status for j in judged] == ["timeout", "error", "vacia", "formato", "ok"]
    assert not any(j.positive for j in judged)
    assert judged[2].finish == "length" and judged[2].completion_tokens == 96 and judged[2].prompt_tokens == 10
    stats = P.judgment_stats(judged)
    assert stats["by_status"] == {"timeout": 1, "error": 1, "vacia": 1, "formato": 1, "ok": 1}
    assert stats["format_failures"] == 2 and stats["request_failures"] == 2 and stats["positives"] == 0 and stats["asked"] == 5


def test_judge_pairs_stops_asking_after_the_deadline(monkeypatch):
    import time

    provider = ScriptedProvider(['{"contradiccion": false}'] * 3)
    pairs = [PAIR] * 3
    judged = P.judge_pairs(provider, pairs, 5.0, log_dir=False, deadline=time.monotonic() - 1)
    assert [j.status for j in judged] == ["omitida"] * 3 and provider.received == []
    provider = ScriptedProvider(['{"contradiccion": false}'] * 3)
    assert [j.status for j in P.judge_pairs(provider, pairs, 5.0, log_dir=False)] == ["ok"] * 3
    stats = P.judgment_stats(judged)
    assert stats["asked"] == 0 and stats["seconds_mean"] == 0.0


# -- hallazgos con citas literales por construcción ----------------------------------------------------------------------
def test_findings_quote_the_lines_literally_with_real_paths_and_line_numbers(tmp_path):
    road, todo = "intro\nHecho: 49 tests\nfin", "uno\ndos\ntres\n  Hecho: 46 tests (con sangría)  \n"
    write(tmp_path / E.ROADMAP_REL, road)
    write(tmp_path / E.TODO_REL, todo)
    ps = P.build_pairs(*docs(road, todo), max_pairs=5)
    assert len(ps.pairs) == 1
    judgments = [P.Judgment(0, True, "ok", motivo="49 frente a 46")]
    found = P.build_findings(ps.pairs, judgments)
    item = found["incoherencias"][0]
    assert item["evidencia_a"] == {"ruta": E.ROADMAP_REL, "cita": "Hecho: 49 tests", "linea": 2}
    assert item["evidencia_b"] == {"ruta": E.TODO_REL, "cita": "Hecho: 46 tests (con sangría)", "linea": 4}
    assert "49 frente a 46" in item["descripcion"] and f"{E.ROADMAP_REL}:2" in item["descripcion"]
    verification = verify("doc_checker", json.dumps(found), root=tmp_path)
    assert verification.valid and verification.details["schema_ok"] and all(c["valid"] for c in verification.details["citations"])
    assert P.build_findings(ps.pairs, [P.Judgment(0, None, "formato")]) == {"incoherencias": []}  # solo los positivos bien formados


def test_citations_of_the_real_pairs_are_always_literal_even_for_every_pair_judged_positive(tmp_path):
    road, todo = E.frozen_docs()
    s_road, s_todo, _ = E.seed_documents(road, todo)
    write(tmp_path / E.ROADMAP_REL, s_road)
    write(tmp_path / E.TODO_REL, s_todo)
    ps = P.build_pairs(*docs(s_road, s_todo))
    found = P.build_findings(ps.pairs, [P.Judgment(i, True, "ok") for i in range(len(ps.pairs))])
    verification = verify("doc_checker", json.dumps(found), root=tmp_path)
    assert len(found["incoherencias"]) == 60 and verification.valid
    assert all(c["valid"] for c in verification.details["citations"]) and len(verification.details["citations"]) == 120


# -- rol y protocolo -------------------------------------------------------------------------------------------------------
def test_role_config_is_single_shot_with_explicit_limits_and_a_small_context():
    cfg = get_agent_config("doc_checker_pairs")
    assert cfg.mode == "single_shot" and cfg.tools == [] and cfg.write == [] and cfg.verifier == "doc_checker"
    assert cfg.timeout == cfg.max_seconds == 60 and cfg.max_tokens == 96 and cfg.options["num_ctx"] == 2048
    assert cfg.model == "gemma3:4b" and cfg.provider == "ollama"
    # el prompt de sistema más la línea más larga de los documentos congelados (dos veces) cabe con margen
    longest = max(len(line) for name in ("roadmap.md", "TODO.md") for line in (E.FIXTURES_DIR / name).read_text(encoding="utf-8").splitlines())
    assert (len(P.PAIR_SYSTEM_PROMPT) + 2 * longest + 200) / 2.5 + cfg.max_tokens <= cfg.options["num_ctx"]


def test_role_uses_the_criteria_of_doc_checker_without_touching_them():
    assert E.base_role("doc_checker_pairs") == "doc_checker"
    assert "doc_checker_pairs" not in E.CRITERIA and "doc_checker_pairs" not in E.REPS
    assert E.CRITERIA["doc_checker"] == {"valid_citations_min": 0.90, "recovered_mean_min": 4.0, "false_positives_max": 2} and E.REPS["doc_checker"] == 3
    assert "doc_checker_pairs" in E.COMMAND_ROLES and "doc_checker_pairs" in E.EVALUATORS
    assert "doc_checker_pairs" not in E.ROLES and "doc_checker_pairs" not in E.ALL_ROLES  # no entra en `all`


def test_trial_for_overrides_request_limits_without_changing_the_name():
    plain = E.trial_for("doc_checker_pairs", E.argparse.Namespace(model="deepseek-r1:7b", num_ctx=3072, max_tokens=800, max_seconds=120.0))
    assert plain.label == "doc_checker_pairs@deepseek-r1-7b"
    assert plain.overrides["max_tokens"] == 800 and plain.overrides["max_seconds"] == 120.0 and plain.overrides["timeout"] == 120.0
    config = get_agent_config("doc_checker_pairs", **plain.overrides)
    assert (config.model, config.max_tokens, config.max_seconds, config.timeout, config.options["num_ctx"]) == ("deepseek-r1:7b", 800, 120.0, 120.0, 3072)
    assert E.trial_for("doc_checker_pairs", E.argparse.Namespace(model=None, num_ctx=None)).overrides == {}


def test_cli_rejects_request_limits_with_all(tmp_path):
    with pytest.raises(SystemExit):
        E.main(["all", "--workdir", str(tmp_path / "w"), "--max-tokens", "100"])


# -- evaluación con proveedor guionizado -------------------------------------------------------------------------------------
def patch_live(monkeypatch, scripted, installed=("gemma3:4b",)):
    monkeypatch.setattr(E, "make_provider", lambda config: scripted)
    monkeypatch.setattr(E, "server_models", lambda base_url: list(installed))
    monkeypatch.setattr(E, "ollama_ps", lambda *a, **k: {"processor": "100% GPU", "context_length": 2048})
    monkeypatch.setattr(E, "ollama_stop", lambda model: "ok")
    monkeypatch.setattr(E, "ollama_ps_text", lambda: "")


def eval_args(tmp_path, **kw):
    base = dict(workdir=str(tmp_path / "w"), results_dir=str(tmp_path / "r"), env_variant=1, tag="", reps=1, limit=None, drop=None, model=None, num_ctx=None, max_minutes=None)
    base.update(kw)
    return E.argparse.Namespace(**base)


def scripted_for_one_rep(positive_seed_ids=("cifra",), garbage_at=None):
    """Respuestas para una repetición: copia sembrada (60 pares) y copia limpia (60 pares). Positivo en los pares sembrados pedidos."""
    road, todo = E.frozen_docs()
    s_road, s_todo, seeds = E.seed_documents(road, todo)
    seeded = P.build_pairs(*docs(s_road, s_todo)).pairs
    wanted = [{(s["file"], s["line"]), (s["partner_file"], s["partner_line"])} for s in seeds if s["id"] in positive_seed_ids]
    answers = []
    for i, pair in enumerate(seeded):
        positive = set(pair.ends) in wanted  # solo el par exacto de la semilla (la línea alterada y la del otro documento)
        answers.append("¿?" if garbage_at == i else json.dumps({"contradiccion": positive}))
    answers += [json.dumps({"contradiccion": False})] * 60
    return answers


def test_evaluation_scores_a_recovered_seed_with_literal_citations_and_reports_the_generator_apart(tmp_path, monkeypatch):
    scripted = ScriptedProvider(scripted_for_one_rep(garbage_at=7))
    patch_live(monkeypatch, scripted)
    out = E.evaluate_role("doc_checker_pairs", eval_args(tmp_path))
    results = tmp_path / "r"
    assert out["role"] == "doc_checker_pairs" and out["protocol"] == E.CRITERIA["doc_checker"] and out["model"] == "gemma3:4b"
    for name in ("doc_checker_pairs.json", "doc_checker_pairs.md", "doc_checker_pairs.runs.jsonl", "seeds_doc_checker_pairs.json", "pairs_doc_checker_pairs.json"):
        assert (results / name).is_file(), name
    assert [r["case"] for r in out["runs"]] == ["seeded", "clean"] and all(r["steps"] == 60 and not r["copy_modified"] for r in out["runs"])
    criteria = out["criteria"]
    assert [c[1] for c in criteria] == [">= 90 %", ">= 4", "<= 2"]
    assert out["metrics"]["valid_citations"] == out["metrics"]["citations"] == 2 and criteria[0][3] is True  # citas literales por construcción
    assert out["metrics"]["recovered_per_rep"] == [1] and criteria[1][3] is False  # 1 de 6 < 4: el umbral no se rebaja
    assert out["metrics"]["false_positives_clean_per_rep"] == [0] and criteria[2][3] is True
    m = out["metrics"]
    assert m["requests"] == 120 and m["format_failures"] == 1 and m["request_failures"] == 0 and m["positives"] == 1 and m["reps_done"] == 1 and not m["incomplete"]
    assert m["generator_coverage"]["seeded"]["n_both"] == 5 and m["generator_coverage"]["seeded"]["n_seed_line"] == 6 and m["generator_coverage"]["clean"]["pairs"] == 60
    saved = json.loads((results / "pairs_doc_checker_pairs.json").read_text(encoding="utf-8"))
    assert saved["coverage"]["seeded"]["n_both"] == 5 and len(saved["seeded"]["pairs"]) == 60 and saved["seeded"]["candidates"] >= 60
    md = (results / "doc_checker_pairs.md").read_text(encoding="utf-8")
    assert "Cobertura del generador" in md and "literales por construcción" in md and "fallos de formato 1" in md
    assert len(scripted.received) == 120 and scripted.timeouts == [60.0] * 120 and all(len(m_) == 2 for m_ in scripted.received)  # una petición por par, con tope de tiempo
    assert (results / "logs").is_dir()
    detail = out["runs"][0]["details"]
    assert detail["judgment_stats"]["format_failures"] == 1 and detail["judgments"][7]["status"] == "formato" and len(detail["judgments"]) == 60


def test_evaluation_counts_false_positives_in_the_clean_copy(tmp_path, monkeypatch):
    answers = [json.dumps({"contradiccion": False})] * 60 + [json.dumps({"contradiccion": True})] * 3 + [json.dumps({"contradiccion": False})] * 57
    patch_live(monkeypatch, ScriptedProvider(answers))
    out = E.evaluate_role("doc_checker_pairs", eval_args(tmp_path))
    assert out["metrics"]["false_positives_clean_per_rep"] == [3] and out["criteria"][2][3] is False
    assert out["metrics"]["recovered_per_rep"] == [0] and out["runs"][1]["success"] is False and out["runs"][0]["success"] is False


def test_evaluation_with_a_model_names_the_results_and_does_a_separate_warmup(tmp_path, monkeypatch):
    scripted = ScriptedProvider(["listo", *scripted_for_one_rep()])
    patch_live(monkeypatch, scripted, installed=("deepseek-r1:7b",))
    out = E.evaluate_role("doc_checker_pairs", eval_args(tmp_path, model="deepseek-r1:7b", num_ctx=3072, max_tokens=800, max_seconds=90.0, tag="p1"))
    assert out["role"] == "doc_checker_pairs@deepseek-r1-7b" and out["model"] == "deepseek-r1:7b" and out["num_ctx"] == 3072
    assert (tmp_path / "r" / "doc_checker_pairs@deepseek-r1-7b_p1.json").is_file() and (tmp_path / "r" / "pairs_doc_checker_pairs@deepseek-r1-7b.json").is_file()
    assert out["warmup"]["ok"] and len(scripted.received) == 121 and scripted.timeouts[1:] == [90.0] * 120
    assert E.get_agent_config("doc_checker_pairs").max_tokens == 96  # la configuración no cambia


def test_time_budget_discards_a_truncated_repetition_and_declares_the_protocol_incomplete(tmp_path, monkeypatch):
    scripted = ScriptedProvider(["no debería pedirse nada"])
    patch_live(monkeypatch, scripted)
    out = E.evaluate_role("doc_checker_pairs", eval_args(tmp_path, reps=5, max_minutes=1e-9))
    m = out["metrics"]
    assert scripted.received == []  # el presupuesto ya estaba agotado: no se pidió ningún par
    assert m["reps_done"] == 0 and m["incomplete"] and m["truncated_reps"] == 1 and m["requests"] == 0
    assert out["runs"][0]["stop_reason"] == "budget" and out["runs"][0]["success"] is False and out["runs"][0]["details"]["truncated"]
    assert [c[3] for c in out["criteria"]] == [False, False, False]  # sin repeticiones completas no se cumple nada
    assert any("PROTOCOLO INCOMPLETO: 0 de 5" in line for line in out["notes"])


def test_a_repetition_starts_only_if_it_fits_in_the_time_budget():
    assert E.rep_fits_budget(100.0, 50.0, None) and E.rep_fits_budget(100.0, 50.0, 150.0)
    assert not E.rep_fits_budget(100.0, 50.0, 149.9) and not E.rep_fits_budget(2400.0, 1.0, 2400.0)


# -- modelos de decisión (proveedor ollama_systemone, tev1) ------------------------------------------------------------------
from test_agents_providers import FakeServer  # noqa: E402

from amp_agents.agents import make_provider  # noqa: E402
from amp_agents.providers import SystemOneProvider  # noqa: E402


@pytest.fixture
def fake_server():
    srv = FakeServer()
    yield srv
    srv.close()


def test_systemone_provider_posts_state_and_questions_and_returns_probabilities(fake_server):
    fake_server.responses.append((200, {"answers": {"contradiccion": {"type": "noul", "noul": 0.97}}}))
    provider = SystemOneProvider(fake_server.base_url, "tev1:4b-q4_K_M", timeout=30)
    assert not provider.base_url.endswith("/v1")  # se usa la raíz del servidor
    out = provider.decide("Línea A: x\nLínea B: y", {"contradiccion": "¿Se contradicen?"}, timeout=5)
    assert out == {"contradiccion": 0.97}
    sent = fake_server.requests[0]
    assert sent["path"] == "/v1/systemone"
    assert sent["body"] == {"model": "tev1:4b-q4_K_M", "state": "Línea A: x\nLínea B: y", "questions": {"contradiccion": {"type": "noul", "instructions": "¿Se contradicen?"}}}


@pytest.mark.parametrize(
    "payload",
    [
        {"answers": {}},
        {"respuestas": 1},
        {"answers": {"contradiccion": {"type": "noul", "noul": 1.5}}},
        {"answers": {"contradiccion": {"type": "noul", "noul": "0.9"}}},
        {"answers": {"contradiccion": {"type": "noul", "noul": True}}},
        {"answers": {"contradiccion": {"type": "noul"}}},
    ],
)
def test_systemone_provider_rejects_answers_without_a_valid_probability(fake_server, payload):
    fake_server.responses.append((200, payload))
    with pytest.raises(ProviderError):
        SystemOneProvider(fake_server.base_url, "m").decide("s", {"contradiccion": "q"})


def test_systemone_provider_has_time_limit_http_errors_and_no_chat(fake_server):
    fake_server.responses.append(("slow", None))
    with pytest.raises(ProviderTimeout):
        SystemOneProvider(fake_server.base_url, "m", timeout=30).decide("s", {"contradiccion": "q"}, timeout=0.3)
    fake_server.responses.append((500, {"error": "boom"}))
    with pytest.raises(ProviderError, match="HTTP 500"):
        SystemOneProvider(fake_server.base_url, "m").decide("s", {"contradiccion": "q"})
    with pytest.raises(ProviderError, match="no genera texto"):
        SystemOneProvider(fake_server.base_url, "m").chat([{"role": "user", "content": "x"}])
    with pytest.raises(ProviderError):
        SystemOneProvider(fake_server.base_url, "m").decide("s", {})


def test_make_provider_builds_the_decision_provider_from_the_config():
    config = get_agent_config("doc_checker_pairs", model="tev1:4b-q4_K_M", provider="ollama_systemone")
    provider = make_provider(config)
    assert isinstance(provider, SystemOneProvider) and provider.model == "tev1:4b-q4_K_M" and provider.timeout == config.timeout == 60


class FakeDecider(SystemOneProvider):
    """Modelo de decisión simulado: devuelve las probabilidades guionizadas (o lanza la excepción dada)."""

    def __init__(self, scores):
        super().__init__("http://x", "tev1", 10)
        self.scores = list(scores)
        self.calls = []

    def decide(self, state, questions, timeout=None):
        self.calls.append((state, questions, timeout))
        item = self.scores.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"contradiccion": item}


def test_decision_threshold_is_fixed_at_one_half_and_inclusive():
    assert P.DECISION_THRESHOLD == 0.5 and P.SENSITIVITY_THRESHOLDS == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    decider = FakeDecider([0.5, 0.4999, 0.97, ProviderTimeout("lento"), ProviderError("roto")])
    judged = P.judge_pairs(decider, [PAIR] * 5, 7.0)
    assert [j.verdict for j in judged] == [True, False, True, None, None] and [j.status for j in judged] == ["ok", "ok", "ok", "timeout", "error"]
    assert [j.score for j in judged[:3]] == [0.5, 0.4999, 0.97] and [j.positive for j in judged] == [True, False, True, False, False]
    assert all(c[2] == 7.0 for c in decider.calls)  # tope de tiempo en cada petición
    state, questions, _ = decider.calls[0]
    assert f"Línea A (`{E.ROADMAP_REL}`):\nHecho: 49 tests" in state and f"Línea B (`{E.TODO_REL}`):\nHecho: 46 tests" in state
    assert questions == {"contradiccion": P.DECISION_QUESTION}


def test_threshold_sensitivity_only_recomputes_from_stored_probabilities():
    road, todo = E.frozen_docs()
    s_road, s_todo, seeds = E.seed_documents(road, todo)
    seeded = P.build_pairs(*docs(s_road, s_todo))
    clean = P.build_pairs(*docs(road, todo))
    both = {(s["file"], s["line"]) for s in seeds if s["id"] == "cifra"} | {(s["partner_file"], s["partner_line"]) for s in seeds if s["id"] == "cifra"}
    scores_seeded = [0.6 if set(p.ends) == both else 0.2 for p in seeded.pairs]
    scores_clean = [0.8, 0.35] + [0.05] * 58
    judged = {"seeded": [P.Judgment(i, s >= 0.5, "ok", score=s) for i, s in enumerate(scores_seeded)], "clean": [P.Judgment(i, s >= 0.5, "ok", score=s) for i, s in enumerate(scores_clean)]}
    rows = E.threshold_sensitivity([judged], {"seeded": seeded, "clean": clean}, seeds)
    by = {r["threshold"]: r for r in rows}
    assert by[0.5]["recovered_per_rep"] == [1] and by[0.5]["false_positives_clean_per_rep"] == [1]
    assert by[0.7]["recovered_per_rep"] == [0] and by[0.3]["false_positives_clean_per_rep"] == [2]
    assert by[0.1]["false_positives_clean_per_rep"] == [2] and by[0.1]["recovered_per_rep"] == [6]  # con 0,1 todo lo sembrado (0,2) pasa el umbral


def test_trial_for_think_and_provider_overrides_keep_the_other_options():
    t = E.trial_for("doc_checker_pairs", E.argparse.Namespace(model="lfm3:1.7b", think=False))
    assert t.label == "doc_checker_pairs@lfm3-1.7b" and t.overrides["options"] == {"num_ctx": 2048, "temperature": 0.2, "think": False}
    config = get_agent_config("doc_checker_pairs", **t.overrides)
    assert config.options["think"] is False and config.options["num_ctx"] == 2048
    d = E.trial_for("doc_checker_pairs", E.argparse.Namespace(model="tev1:4b-q4_K_M", provider="ollama_systemone"))
    assert d.overrides == {"model": "tev1:4b-q4_K_M", "provider": "ollama_systemone"}
    assert "options" not in E.trial_for("doc_checker_pairs", E.argparse.Namespace(model="x", think=None)).overrides
    with pytest.raises(SystemExit):
        E.main(["all", "--think", "false"])


def test_think_false_goes_to_the_top_level_of_the_native_request(fake_server):
    from amp_agents.providers import OllamaProvider

    fake_server.responses.append((200, {"message": {"role": "assistant", "content": '{"contradiccion": false}', "thinking": ""}, "done_reason": "stop"}))
    provider = OllamaProvider(fake_server.base_url, "lfm3:1.7b", options={"think": False, "num_ctx": 2048, "num_predict": 96})
    reply = provider.chat([{"role": "user", "content": "x"}])
    body = fake_server.requests[0]["body"]
    assert body["think"] is False and body["options"] == {"num_ctx": 2048, "num_predict": 96} and reply.content == '{"contradiccion": false}'
    # un modelo de razonamiento: el juicio está en content y el razonamiento (thinking) queda aparte, sin mezclarse
    fake_server.responses.append((200, {"message": {"role": "assistant", "content": '\n\n{"contradicción": true}', "thinking": 'Pienso en {"contradiccion": false}'}, "done_reason": "stop"}))
    reply = provider.chat([{"role": "user", "content": "x"}])
    assert reply.reasoning.startswith("Pienso") and P.parse_verdict(reply.content) == (True, "", "ok")


def test_evaluation_of_a_decision_model_uses_the_threshold_and_reports_the_sensitivity(tmp_path, monkeypatch):
    road, todo = E.frozen_docs()
    s_road, s_todo, seeds = E.seed_documents(road, todo)
    seeded = P.build_pairs(*docs(s_road, s_todo))
    cifra = next(s for s in seeds if s["id"] == "cifra")
    target = {(cifra["file"], cifra["line"]), (cifra["partner_file"], cifra["partner_line"])}
    scores = [0.9 if set(p.ends) == target else 0.1 for p in seeded.pairs] + [0.1] * 60
    decider = FakeDecider([0.5] + scores)  # la primera respuesta es la del calentamiento
    patch_live(monkeypatch, decider, installed=("tev1:4b-q4_K_M",))
    args = eval_args(tmp_path, model="tev1:4b-q4_K_M", provider="ollama_systemone")
    out = E.evaluate_role("doc_checker_pairs", args)
    assert out["role"] == "doc_checker_pairs@tev1-4b-q4_K_M" and out["warmup"]["ok"] and len(decider.calls) == 121
    m = out["metrics"]
    assert m["recovered_per_rep"] == [1] and m["false_positives_clean_per_rep"] == [0] and m["positives"] == 1
    assert out["criteria"][0][3] is True and out["criteria"][1][3] is False and out["criteria"][2][3] is True
    assert [r["threshold"] for r in m["threshold_sensitivity"]] == list(P.SENSITIVITY_THRESHOLDS)
    assert {r["threshold"]: r["recovered_per_rep"] for r in m["threshold_sensitivity"]}[0.9] == [1]
    assert any("umbral fijado de antemano (0,5)" in line for line in out["notes"])
    saved = out["runs"][0]["details"]["judgments"]
    assert saved[12]["score"] is not None and all(j["status"] == "ok" for j in saved)

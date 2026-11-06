"""Tests (sin modelo) del reescritor de consultas, el enrutador, el filtro de cifras del resumidor, el proveedor ``choice`` y el
posprocesado de la CLI (``run.py``)."""

import json
import shutil
from pathlib import Path

import pytest
from agents_helpers import write
from test_agents_providers import FakeServer

from amp_agents import eval_roles as E
from amp_agents import rewrite as W
from amp_agents import router as RO
from amp_agents import run as R
from amp_agents import verifiers as V
from amp_agents.agents import get_agent_config, make_provider
from amp_agents.providers import ProviderError, ProviderTimeout, ScriptedProvider, SystemOneProvider

SRC = Path(__file__).resolve().parents[1]
REAL_ROOT = SRC.parent


@pytest.fixture
def server():
    srv = FakeServer()
    yield srv
    srv.close()


def patch_live(monkeypatch, provider, installed):
    monkeypatch.setattr(E, "make_provider", lambda config: provider)
    monkeypatch.setattr(E, "server_models", lambda base_url: list(installed))
    monkeypatch.setattr(E, "ollama_ps", lambda *a, **k: {"processor": "100% GPU", "context_length": 2048})
    monkeypatch.setattr(E, "ollama_stop", lambda model: "ok")
    monkeypatch.setattr(E, "ollama_ps_text", lambda: "")


def eval_args(tmp_path, **kw):
    base = dict(workdir=str(tmp_path / "w"), results_dir=str(tmp_path / "r"), env_variant=1, tag="p4", reps=None, limit=None, drop=None, model=None, num_ctx=None, max_minutes=None)
    base.update(kw)
    return E.argparse.Namespace(**base)


# ============================ proveedor de decisión: pregunta de tipo choice ======================================================
def test_choose_posts_criteria_and_takes_the_option_with_the_highest_probability(server):
    server.responses.append((200, {"answers": {"rol": {"type": "choice", "choice": "b", "probabilities": {"a": 0.2, "b": 0.7, "c": 0.1}}}}))
    provider = SystemOneProvider(server.base_url, "tev1")
    choice, probabilities = provider.choose("Petición: x", "¿Cuál?", {"a": "uno", "b": "dos", "c": "tres"}, timeout=5, name="rol")
    assert choice == "b" and probabilities == {"a": 0.2, "b": 0.7, "c": 0.1}
    sent = server.requests[0]
    assert sent["path"] == "/v1/systemone" and sent["body"]["state"] == "Petición: x" and sent["body"]["model"] == "tev1"
    assert sent["body"]["questions"] == {"rol": {"type": "choice", "instructions": "¿Cuál?", "criteria": {"a": "uno", "b": "dos", "c": "tres"}}}


def test_choose_uses_the_probabilities_not_the_reported_choice_and_breaks_ties_by_order(server):
    server.responses.append((200, {"answers": {"eleccion": {"choice": "a", "probabilities": {"a": 0.3, "b": 0.7}}}}))
    assert SystemOneProvider(server.base_url, "m").choose("s", "q", {"a": "x", "b": "y"})[0] == "b"
    server.responses.append((200, {"answers": {"eleccion": {"choice": "b", "probabilities": {"a": 0.5, "b": 0.5}}}}))
    assert SystemOneProvider(server.base_url, "m").choose("s", "q", {"a": "x", "b": "y"})[0] == "a"
    server.responses.append((200, {"answers": {"eleccion": {"choice": "b"}}}))  # sin probabilidades: la opción dada
    assert SystemOneProvider(server.base_url, "m").choose("s", "q", {"a": "x", "b": "y"}) == ("b", {})


@pytest.mark.parametrize(
    "payload",
    [
        {"answers": {"eleccion": {"choice": "z"}}},
        {"answers": {"eleccion": {"probabilities": {"a": 0.5}}}},  # falta una opción
        {"answers": {"eleccion": {"probabilities": {"a": 1.5, "b": 0.1}}}},
        {"answers": {"eleccion": {"probabilities": {"a": True, "b": 0.1}}}},
        {"answers": {}},
        {"otra": 1},
    ],
)
def test_choose_rejects_answers_without_a_valid_option(server, payload):
    server.responses.append((200, payload))
    with pytest.raises(ProviderError):
        SystemOneProvider(server.base_url, "m").choose("s", "q", {"a": "x", "b": "y"})


def test_choose_needs_two_options_and_has_a_time_limit_and_http_errors(server):
    with pytest.raises(ProviderError, match="al menos dos"):
        SystemOneProvider(server.base_url, "m").choose("s", "q", {"a": "x"})
    server.responses.append(("slow", ""))
    with pytest.raises(ProviderTimeout):
        SystemOneProvider(server.base_url, "m", timeout=60).choose("s", "q", {"a": "x", "b": "y"}, timeout=1)
    server.responses.append((500, "boom"))
    with pytest.raises(ProviderError, match="HTTP 500"):
        SystemOneProvider(server.base_url, "m").choose("s", "q", {"a": "x", "b": "y"})


# ============================ reescritor de consultas =============================================================================
@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"palabras": ["backend", "AMP_GAN_BACKEND", "runtime"]}', (["backend", "AMP_GAN_BACKEND", "runtime"], "ok")),
        ('```json\n{"palabras": ["a", "b", "c", "d", "e", "f", "g", "h"]}\n```', (list("abcdefgh"), "ok")),
        ('{"palabras": ["a", "b"]}', (["a", "b"], "fuera_de_rango")),
        ('{"palabras": ["a", "A", " a ", "b", "c"]}', (["a", "b", "c"], "ok")),  # sin duplicados
        ('{"palabras": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]}', (list("abcdefgh"), "fuera_de_rango")),  # se usan las 8 primeras
        ('{"palabras": ["", "  ", 3, null]}', ([], "formato")),
        ('{"palabras": "backend runtime"}', ([], "formato")),
        ('{"otra": []}', ([], "formato")),
        ("backend, runtime", ([], "formato")),
        ("", ([], "vacia")),
        ('<think>hmm</think>{"palabras": ["x", "y", "z"]}', (["x", "y", "z"], "ok")),
    ],
)
def test_parse_keywords(text, expected):
    assert W.parse_keywords(text) == expected


def test_expanded_query_is_the_original_question_followed_by_the_words_and_bm25_is_untouched():
    assert W.expanded_query(" ¿Qué hace X? ", ["uno", "dos"]) == "¿Qué hace X? uno dos" and W.expanded_query("¿P?", []) == "¿P?"
    from amp_agents.retrieval import LexicalRetriever, tokenize

    assert tokenize("¿Qué hace X? uno dos") == ["hace", "uno", "dos"] or "uno" in tokenize("¿Qué hace X? uno dos")  # el tokenizador de siempre
    assert LexicalRetriever.__init__.__defaults__ == (1.5, 0.75)  # k1 y b de BM25 sin tocar


def test_rewrite_query_sends_one_request_without_tools_and_builds_the_final_query():
    provider = ScriptedProvider(['{"palabras": ["backend", "AMP_GAN_BACKEND", "runtime"]}'])
    rw = W.rewrite_query(provider, "¿cómo se elige el backend?", 60.0, log_dir=False)
    assert rw.status == "ok" and not rw.format_failure and rw.query == "¿cómo se elige el backend? backend AMP_GAN_BACKEND runtime"
    assert len(provider.received) == 1 and [m["role"] for m in provider.received[0]] == ["system", "user"] and provider.timeouts == [60.0]
    assert provider.received[0][0]["content"] == W.REWRITE_SYSTEM_PROMPT and "¿cómo se elige el backend?" in provider.received[0][1]["content"]


def test_rewrite_failures_fall_back_to_the_original_question():
    for answer, status in (("¿?", "formato"), ("", "vacia"), (ProviderTimeout("lento"), "timeout"), (ProviderError("caído"), "error")):
        rw = W.rewrite_query(ScriptedProvider([answer]), "pregunta", 60.0, log_dir=False)
        assert rw.status == status and rw.query == "pregunta" and rw.words == []
    short = W.rewrite_query(ScriptedProvider(['{"palabras": ["x"]}']), "pregunta", 60.0, log_dir=False)
    assert short.status == "fuera_de_rango" and short.format_failure and short.query == "pregunta x"


def test_rewriter_role_and_criteria_are_declared_before_measuring():
    cfg = get_agent_config("query_rewriter")
    assert cfg.mode == "single_shot" and cfg.tools == [] and cfg.model == "llama3.2:3b" and cfg.max_tokens == 96 and cfg.timeout == cfg.max_seconds == 60
    assert cfg.options["temperature"] == 0 and cfg.options["num_ctx"] == 2048 and cfg.options["format"] == W.REWRITE_FORMAT_SCHEMA
    assert W.REWRITE_FORMAT_SCHEMA["properties"]["palabras"]["minItems"] == 3 and W.REWRITE_FORMAT_SCHEMA["properties"]["palabras"]["maxItems"] == 8
    assert E.CRITERIA["query_rewriter"] == {"gain_min": 2, "regressions_max": 1} and "query_rewriter" in E.COMMAND_ROLES and "query_rewriter" not in E.ALL_ROLES
    assert "query_rewriter" not in E.REPS and E.base_role("query_rewriter") == "query_rewriter"


def scripted_ranks(monkeypatch, pairs):
    """``retriever_rank`` guionizado: para cada consulta, ``(rango BM25, rango con reescritura)`` en el orden de las llamadas."""
    ranks = iter([r for pair in pairs for r in pair])
    monkeypatch.setattr(E, "retriever_rank", lambda hits, truth, k=3: next(ranks))


WORDS = json.dumps({"palabras": ["uno", "dos", "tres"]})


def test_rewriter_criterion_needs_two_more_hits_and_at_most_one_regression(tmp_path, monkeypatch):
    provider = ScriptedProvider(["listo", *[WORDS] * 15])
    patch_live(monkeypatch, provider, ["llama3.2:3b"])
    pairs = [(1, 1)] * 10 + [(None, 2), (None, 3)] + [(None, None)] * 3  # BM25 10/15; con reescritura 12/15; sin regresiones
    scripted_ranks(monkeypatch, pairs)
    out = E.evaluate_role("query_rewriter", eval_args(tmp_path, model="llama3.2:3b"))
    assert out["role"] == "query_rewriter@llama3.2-3b" and out["protocol"] == E.CRITERIA["query_rewriter"] and out["warmup"]["ok"]
    gain, regress = out["criteria"]
    assert gain[1] == ">= BM25 + 2" and "12/15 frente a BM25 10/15" in gain[2] and gain[3] is True and regress[3] is True
    m = out["metrics"]
    assert m["bm25_top3"] == 10 and m["rewritten_top3"] == 12 and m["gained"] == ["q11", "q12"] and m["lost"] == [] and m["corpus_unchanged"] and len(m["corpus_hash_before"]) == 12
    text = "\n".join(out["notes"])
    assert "solo indicativo" in text and "Fallo -> acierto (2)" in text and "uno" in text and "Hash del corpus" in text and "sin cambios" in text
    assert (tmp_path / "r" / "query_rewriter@llama3.2-3b_p4.json").is_file() and len(provider.received) == 16


def test_rewriter_criterion_fails_with_one_gain_or_with_two_regressions(tmp_path, monkeypatch):
    patch_live(monkeypatch, ScriptedProvider([WORDS] * 15), ["llama3.2:3b"])
    scripted_ranks(monkeypatch, [(1, 1)] * 10 + [(None, 2)] + [(None, None)] * 4)  # +1
    out = E.evaluate_role("query_rewriter", eval_args(tmp_path))
    assert [c[3] for c in out["criteria"]] == [False, True]
    patch_live(monkeypatch, ScriptedProvider([WORDS] * 15), ["llama3.2:3b"])
    scripted_ranks(monkeypatch, [(1, 1)] * 8 + [(2, None)] * 2 + [(None, 1)] * 5)  # +3 pero 2 regresiones
    out = E.evaluate_role("query_rewriter", eval_args(tmp_path, tag="b"))
    assert [c[3] for c in out["criteria"]] == [True, False] and out["metrics"]["lost"] == ["q09", "q10"] and out["metrics"]["gain"] == 3


def test_rewriter_evaluation_records_format_failures_and_the_corpus_hash_changes_with_the_corpus(tmp_path, monkeypatch):
    patch_live(monkeypatch, ScriptedProvider(["¿?"] * 15), ["llama3.2:3b"])
    out = E.evaluate_role("query_rewriter", eval_args(tmp_path))
    assert out["metrics"]["format_failures"] == 15 and out["metrics"]["rewritten_top3"] == out["metrics"]["bm25_top3"]  # sin palabras = BM25 simple
    assert [c[3] for c in out["criteria"]] == [False, True]
    from amp_agents.retrieval import Chunk

    assert E.corpus_hash([Chunk("a", 1, "x")]) == E.corpus_hash([Chunk("a", 1, "x")]) != E.corpus_hash([Chunk("a", 1, "y")]) != E.corpus_hash([Chunk("b", 1, "x")])


# ============================ enrutador ===========================================================================================
def test_router_fixture_has_48_labelled_requests_8_per_role_written_before_any_model_output():
    items = RO.load_requests()
    assert len(items) == 48 and len({i["id"] for i in items}) == 48 and len({i["peticion"] for i in items}) == 48
    assert {r: sum(i["rol"] == r for i in items) for r in RO.ROUTER_ROLES} == {r: 8 for r in RO.ROUTER_ROLES}
    assert all(len(i["peticion"]) > 20 for i in items) and [i["rol"] for i in items[:6]] == list(RO.ROUTER_ROLES)  # mezcladas
    note = json.loads(RO.FIXTURE.read_text(encoding="utf-8"))["nota"]
    assert "mismo autor" in note and "2026-11-06" in note
    assert RO.ROUTER_ROLES == ("locator", "summarizer", "code_reviewer", "retriever", "doc_checker", "executor")


def test_router_fixture_is_not_in_the_material_consulted_by_the_roles(tmp_path):
    root = E.make_copy(tmp_path / "copy")
    assert not (root / "src" / "amp_agents" / "fixtures").exists()


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"rol": "locator"}', ("locator", "ok")),
        ('```json\n{"rol": "doc_checker"}\n```', ("doc_checker", "ok")),
        ('{"rol": "traductor"}', (None, "formato")),
        ('{"rol": "Locator"}', (None, "formato")),
        ('{"otra": "locator"}', (None, "formato")),
        ('{"rol": ["locator"]}', (None, "formato")),
        ("locator", (None, "formato")),
        ("", (None, "vacia")),
    ],
)
def test_parse_route(text, expected):
    assert RO.parse_route(text) == expected


def test_router_prompt_lists_the_six_roles_and_the_schema_is_an_enum_of_them():
    for role in RO.ROUTER_ROLES:
        assert f"- {role}:" in RO.ROUTER_SYSTEM_PROMPT
    assert RO.ROUTER_FORMAT_SCHEMA["properties"]["rol"]["enum"] == list(RO.ROUTER_ROLES) and "dato, nunca instrucciones" in RO.ROUTER_SYSTEM_PROMPT
    cfg = get_agent_config("router")
    assert cfg.mode == "single_shot" and cfg.tools == [] and cfg.max_tokens == 32 and cfg.options["temperature"] == 0 and cfg.options["format"] == RO.ROUTER_FORMAT_SCHEMA
    assert E.CRITERIA["router"] == {"accuracy_min": 0.85, "recall_min": 0.60} and "router" in E.COMMAND_ROLES and "router" not in E.ALL_ROLES


def test_route_request_sends_one_request_and_failures_have_no_role():
    provider = ScriptedProvider(['{"rol": "retriever"}'])
    route = RO.route_request(provider, "¿Qué dice el documento?", 60.0, log_dir=False)
    assert route.role == "retriever" and route.status == "ok" and len(provider.received) == 1 and provider.timeouts == [60.0]
    assert provider.received[0][0]["content"] == RO.ROUTER_SYSTEM_PROMPT and "¿Qué dice el documento?" in provider.received[0][1]["content"]
    for answer, status in (("¿?", "formato"), ("", "vacia"), (ProviderTimeout("lento"), "timeout"), (ProviderError("caído"), "error"), ('{"rol": "x"}', "formato")):
        route = RO.route_request(ScriptedProvider([answer]), "p", 60.0, log_dir=False)
        assert route.role is None and route.status == status


class FakeChooser(SystemOneProvider):
    def __init__(self, answers):
        super().__init__("http://localhost:1", "tev1")
        self.answers, self.calls = list(answers), []

    def choose(self, state, instructions, criteria, timeout=None, name="eleccion"):
        self.calls.append((state, instructions, dict(criteria), timeout, name))
        item = self.answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, {}


def test_decision_model_gets_a_choice_question_with_the_six_role_descriptions():
    decider = FakeChooser(["executor", "inventado", ProviderTimeout("lento"), ProviderError("caído")])
    ok = RO.route_request_decision(decider, "Añade un test", 60.0)
    assert ok.role == "executor" and ok.status == "ok"
    state, question, criteria, timeout, name = decider.calls[0]
    assert state == "Petición: Añade un test" and question == RO.ROUTER_QUESTION and criteria == RO.ROLE_DESCRIPTIONS and timeout == 60.0 and name == "rol"
    assert RO.route_request_decision(decider, "p", 60.0).status == "formato"
    assert RO.route_request_decision(decider, "p", 60.0).status == "timeout" and RO.route_request_decision(decider, "p", 60.0).status == "error"


def test_confusion_matrix_and_per_role_metrics():
    truth = ["locator", "locator", "executor", "executor", "retriever"]
    chosen = ["locator", "retriever", "executor", None, "retriever"]
    matrix = RO.confusion_matrix(truth, chosen)
    assert matrix["locator"]["locator"] == 1 and matrix["locator"]["retriever"] == 1 and matrix["executor"][RO.NO_ROLE] == 1 and matrix["summarizer"][RO.NO_ROLE] == 0
    m = RO.role_metrics(matrix)
    assert m["total"] == 5 and m["correct"] == 3 and m["accuracy"] == pytest.approx(0.6)
    assert m["per_role"]["locator"]["recall"] == 0.5 and m["per_role"]["executor"]["recall"] == 0.5 and m["per_role"]["retriever"]["precision"] == 0.5
    assert m["per_role"]["summarizer"]["n"] == 0 and m["per_role"]["summarizer"]["precision"] is None
    assert "verdad \\ elegido" in RO.matrix_table(matrix)


def answers_for_router(wrong=()):
    items = RO.load_requests()
    out = []
    for i, item in enumerate(items):
        role = item["rol"]
        if i in wrong:
            role = RO.ROUTER_ROLES[(RO.ROUTER_ROLES.index(role) + 1) % 6]
        out.append(json.dumps({"rol": role}))
    return out


def test_router_evaluation_reports_matrix_accuracy_and_recall_and_applies_the_criteria(tmp_path, monkeypatch):
    patch_live(monkeypatch, ScriptedProvider(["listo", *answers_for_router(wrong={0, 6})]), ["phi4-mini:latest"])
    out = E.evaluate_role("router", eval_args(tmp_path, model="phi4-mini"))
    assert out["role"] == "router@phi4-mini" and out["protocol"] == E.CRITERIA["router"]
    m = out["metrics"]
    assert m["total"] == 48 and m["correct"] == 46 and m["accuracy"] == pytest.approx(46 / 48) and len(out["runs"]) == 48
    assert m["per_role"]["locator"]["correct"] == 6 and m["per_role"]["summarizer"]["correct"] == 8 and m["confusion"]["locator"]["summarizer"] == 2
    accuracy, recall = out["criteria"]
    assert accuracy[1] == ">= 85 %" and accuracy[3] is True and recall[1] == ">= 60 %" and recall[3] is True
    text = "\n".join(out["notes"])
    assert "Matriz de confusión" in text and "mismo autor del experimento" in text and "Peticiones mal encaminadas (2)" in text
    assert (tmp_path / "r" / "router@phi4-mini_p4.md").is_file() and m["decision_model"] is False


def test_router_criteria_fail_for_low_accuracy_or_a_role_below_60_percent(tmp_path, monkeypatch):
    patch_live(monkeypatch, ScriptedProvider(["listo", *answers_for_router(wrong={0, 6, 12, 18, 24, 30, 36, 42, 1})]), ["gemma3:4b"])  # 9 errores: 81 %
    out = E.evaluate_role("router", eval_args(tmp_path, model="gemma3:4b"))
    assert out["criteria"][0][3] is False
    patch_live(monkeypatch, ScriptedProvider(["listo", *answers_for_router(wrong={0, 6, 12, 18})]), ["gemma3:4b"])  # locator 4/8 = 50 % < 60 %
    out = E.evaluate_role("router", eval_args(tmp_path, tag="b", model="gemma3:4b"))
    assert out["metrics"]["accuracy"] == pytest.approx(44 / 48) and out["criteria"][0][3] is True and out["criteria"][1][3] is False


def test_router_evaluation_with_the_decision_model_asks_one_choice_per_request(tmp_path, monkeypatch):
    truth = [i["rol"] for i in RO.load_requests()]
    decider = FakeChooser(truth)
    patch_live(monkeypatch, decider, ["tev1:4b-q4_K_M"])
    monkeypatch.setattr(E, "warm_up", lambda config, provider=None: {"seconds": 0.1, "ok": True, "error": "", "ps": {}})
    out = E.evaluate_role("router", eval_args(tmp_path, model="tev1:4b-q4_K_M", provider="ollama_systemone"))
    assert out["role"] == "router@tev1-4b-q4_K_M" and len(decider.calls) == 48 and out["metrics"]["decision_model"] is True and out["metrics"]["accuracy"] == 1.0
    assert all(c[1] == RO.ROUTER_QUESTION for c in decider.calls) and out["criteria"][0][3] and out["criteria"][1][3]


def test_router_time_budget_leaves_requests_unasked_and_the_protocol_incomplete(tmp_path, monkeypatch):
    provider = ScriptedProvider(["no debería pedirse"])
    patch_live(monkeypatch, provider, ["llama3.2:3b"])
    out = E.evaluate_role("router", eval_args(tmp_path, max_minutes=1e-9))
    assert provider.received == [] and out["metrics"]["omitted"] == 48 and [c[3] for c in out["criteria"]] == [False, False]
    assert any("PROTOCOLO INCOMPLETO" in line for line in out["notes"])


# ============================ filtro de cifras no respaldadas ======================================================================
SOURCE = "epoch,loss\n200,0.4321\n"


def test_sentence_spans_do_not_split_abbreviations_or_list_markers_but_do_split_after_a_unit():
    text = "La pérdida fue 0,43 en 3,2 s. Otra frase aquí, p. ej. así. Y 1. dos."
    spans = V.sentence_spans(text)
    pieces = [text[a:b] for a, b in spans]
    assert pieces[0] == "La pérdida fue 0,43 en 3,2 s." and pieces[1] == "Otra frase aquí, p. ej. así." and V.sentence_spans("") == []
    assert [("Sin punto final"[a:b]) for a, b in V.sentence_spans("Sin punto final")] == ["Sin punto final"]


def test_filter_removes_only_sentences_with_unsupported_numbers_keeping_separators():
    text = "La pérdida fue 0,43 en la época 200. Tardó 99 s en total. Fue estable.\n\nOtro párrafo sin cifras. Y 7 épocas más."
    filtered, info = V.filter_unsupported_sentences(text, SOURCE)
    assert filtered == "La pérdida fue 0,43 en la época 200. Fue estable.\n\nOtro párrafo sin cifras."
    assert info["sentences"] == 5 and info["removed"] == 2 and info["removed_texts"] == ["Tardó 99 s en total.", "Y 7 épocas más."]
    assert V.check_numbers_supported(filtered, SOURCE)["missing"] == [] and info["words_after"] < info["words_before"]
    assert V.filter_unsupported_sentences("Frase sin cifras.", SOURCE) == ("Frase sin cifras.", {"sentences": 1, "removed": 0, "removed_texts": [], "words_before": 3, "words_after": 3})
    empty, info = V.filter_unsupported_sentences("Solo 99 cosas.", SOURCE)
    assert empty == "" and info["removed"] == 1 and V.filter_unsupported_sentences("", SOURCE)[0] == ""


def test_filter_uses_the_same_support_rule_as_the_criterion_for_any_text():
    text = "Época 200 con 0,43. Pérdida de 0.4321. Un 12 %. Valor 5,5. Fin."
    filtered, _ = V.filter_unsupported_sentences(text, SOURCE)
    assert V.check_numbers_supported(filtered, SOURCE)["ratio"] == 1.0 and "0,43" in filtered and "5,5" not in filtered and "12 %" not in filtered


def test_postprocess_summary_filters_after_the_trim_and_reports_both_stages():
    body = "El experimento llegó a la época 200 con una pérdida de 0.43 en el conjunto de datos. Tardó 99 minutos. " + " ".join(["Es un dato de la prueba."] * 60)
    plain, info = V.postprocess_summary(body, SOURCE, 150)
    assert "filter" not in info and "99" in plain  # sin el filtro, igual que antes
    fixed, info = V.postprocess_summary(body, SOURCE, 150, filter_numbers=True)
    assert "99" not in fixed and "0,43" in fixed and info["mode"] == "frase" and info["filter"]["removed"] == 1
    assert info["numbers_trimmed"]["missing"] == ["99"] and info["numbers_after"]["missing"] == [] and info["words_trimmed"] > info["words_after"] and info["words_after"] == V.count_words(fixed)
    assert V.postprocess_summary(body, None, 150, filter_numbers=True)[1].get("filter") is None  # sin entrada no hay con qué comparar


def test_summary_filtered_metrics_mark_a_short_result_as_a_format_failure():
    ok = "El experimento llegó a la época 200 con una pérdida de 0,43. " + " ".join(["Es un dato de la prueba."] * 8)
    assert E.summary_filtered_metrics(ok, SOURCE)["failure"] is False
    short = "La pérdida fue 99 en la época 200. Otra frase sin cifras."
    m = E.summary_filtered_metrics(short, SOURCE)
    assert m["failure"] is True and m["words"] < V.FILTER_MIN_WORDS == 30 and m["removed"] == 1
    assert E.summary_filtered_metrics("Valor 99.", SOURCE)["words"] == 0 and E.summary_filtered_metrics("Valor 99.", SOURCE)["failure"] is True


def summary_runs(bodies):
    runs = []
    for i, body in enumerate(bodies):
        d = V.verify("summarizer", body, source=SOURCE).details
        details = {"case": f"c{i}", "words": d["words"], "numbers": d["numbers"], "spanish": d["spanish"], "corrected": E.summary_corrected_metrics(body, SOURCE), "filtered": E.summary_filtered_metrics(body, SOURCE)}
        runs.append(E.RunRecord("summarizer", "m", f"c{i}", 1, False, "final", 1, None, None, 1.0, details=details))
    return runs


def test_summary_criteria_add_filter_rows_with_the_same_thresholds_and_the_failure_rule():
    good = "El experimento llegó a la época 200 con una pérdida de 0,43 en el conjunto de datos. Tardó 99 minutos. " + " ".join(["Es un dato de la prueba."] * 10)
    rows = {r[0]: r for r in E._summarizer_summary(summary_runs([good, good]))["criteria"]}
    names = [n for n in rows if n.startswith("[recortado + filtro]")]
    assert len(names) == 4 and len(rows) == 10
    numbers = rows[names[0]]
    assert numbers[1] == ">= 95 %" and numbers[3] is True and "100.0 %" in numbers[2]
    assert all(rows[n][3] for n in names) and rows["Cifras del resumen presentes en la entrada (agregado)"][3] is False  # crudo: 99 no respaldado
    fail = "La pérdida fue 99 en la época 200. Otra frase sin cifras."  # el filtro la deja con menos de 30 palabras
    summary = E._summarizer_summary(summary_runs([good, fail]))
    rows = {r[0]: r for r in summary["criteria"]}
    assert not any(rows[n][3] for n in names) and summary["metrics"]["filtered"]["failures"] == 1  # un fallo de formato: no se cumple nada con el filtro
    assert "1 de 2" in rows[names[3]][2]
    assert len(E._summarizer_summary([])["criteria"]) == 6 and not any(ok for *_, ok in E._summarizer_summary([])["criteria"])  # sin ejecuciones, sin filas de filtro
    assert E.CRITERIA["summarizer"] == {"numbers_min": 0.95, "max_words": 150}


def test_summary_notes_give_raw_trimmed_and_filtered_and_the_removed_sentences():
    body = "El experimento llegó a la época 200 con una pérdida de 0,43 en el conjunto de datos. Tardó 99 minutos. " + " ".join(["Es un dato de la prueba."] * 10)
    notes = "\n".join(E._summarizer_notes(summary_runs([body])))
    assert "crudo" in notes and "recortado + filtro" in notes and "Frases quitadas por el filtro: 1 en total" in notes and "depende del filtro y no del modelo" in notes
    assert E._summarizer_notes([]) == []


# ============================ posprocesado en la CLI ================================================================================
def test_postprocess_output_summarizer_says_what_it_corrected():
    body = "El experimento llegó a la época 200 con una pérdida de 0.43 en el conjunto de datos. Tardó 99 minutos. " + " ".join(["Es un dato de la prueba."] * 60)
    text, notes, ok = R.postprocess_output("summarizer", body, {"source": SOURCE}, filter_numbers=False)
    assert ok and "0,43" in text and V.count_words(text) <= 150 and "99 minutos" in text
    joined = "\n".join(notes)
    assert "coma decimal" in joined and "recorte (frase)" in joined and "cifras respaldadas" in joined and "filtro" not in joined
    text, notes, ok = R.postprocess_output("summarizer", body, {"source": SOURCE}, filter_numbers=True)
    assert ok and "99" not in text and any(n.startswith("filtro de cifras: 1 de") for n in notes)
    text, notes, ok = R.postprocess_output("summarizer", "La pérdida fue 99 en la época 200. Otra frase.", {"source": SOURCE}, filter_numbers=True)
    assert not ok and any("FALLO DE FORMATO" in n for n in notes)
    assert R.postprocess_output("summarizer", body, {}, True) == (body, [], True) and R.postprocess_output("locator", "x", {"source": "s"}) == ("x", [], True)


def test_postprocess_output_reviewer_corrects_lines_to_the_cited_text_and_marks_unverifiable_ones():
    source = "def f(x):\n    a = 1\n    return x + a\n"
    report = json.dumps({"hallazgos": [{"linea": 7, "cita": "return x + a", "problema": "p", "gravedad": "baja"}, {"linea": 2, "cita": "no está en el fichero", "problema": "p", "gravedad": "baja"}]})
    text, notes, ok = R.postprocess_output("code_reviewer", report, {"source": source})
    obj = json.loads(text)
    assert ok and obj["hallazgos"][0]["linea"] == 3 and obj["hallazgos"][0]["linea_modelo"] == 7 and obj["hallazgos"][0]["verificable"] is True
    assert obj["hallazgos"][1]["verificable"] is False and obj["hallazgos"][1]["linea"] == 2  # no se inventa línea
    joined = "\n".join(notes)
    assert "1 de 2" in joined and "línea 7 -> 3" in joined and "1 hallazgos con la cita ausente" in joined
    assert R.postprocess_output("code_reviewer", "no es json", {"source": source})[0] == "no es json"
    assert R.postprocess_output("code_reviewer", '{"hallazgos": []}', {"source": source})[1] == ["línea corregida a la de la cita literal: 0 de 0 hallazgos"]


def cli_project(tmp_path):
    project = tmp_path / "proj"
    shutil.copytree(REAL_ROOT / "context" / "agents", project / "context" / "agents")
    write(project / "context" / "constitution" / "governance.md", "# Gobernanza\n")
    write(project / "context" / "data" / "processed" / "m.csv", "epoch,loss\n200,0.4321\n")
    write(project / "src" / "amp_gan" / "mod.py", "def f(x):\n    a = 1\n    return x + a\n")
    return project


def test_cli_summarizer_applies_the_corrections_by_default_and_raw_shows_the_model_output(tmp_path, monkeypatch, capsys):
    project = cli_project(tmp_path)
    answer = "El experimento llegó a la época 200 con una pérdida de 0.43 en el conjunto de datos."
    monkeypatch.setattr(R, "make_provider", lambda config: ScriptedProvider([answer]))
    args = ["--agent", "summarizer", "context/data/processed/m.csv", "--root", str(project), "--log-dir", str(tmp_path / "l")]
    code = R.main(args)
    out = capsys.readouterr()
    assert code == 0 and "0,43" in out.out and "0.43" not in out.out and "corrección: coma decimal" in out.err and "verificación: válida" in out.err
    monkeypatch.setattr(R, "make_provider", lambda config: ScriptedProvider([answer]))
    code = R.main([*args, "--raw"])
    out = capsys.readouterr()
    assert code == 0 and "0.43" in out.out and "salida cruda (--raw)" in out.err and "corrección:" not in out.err and "(salida cruda)" in out.err


def test_cli_summarizer_filter_is_optional_by_flag_and_a_too_short_result_fails(tmp_path, monkeypatch, capsys):
    project = cli_project(tmp_path)
    base = ["--agent", "summarizer", "context/data/processed/m.csv", "--root", str(project), "--log-dir", str(tmp_path / "l")]
    answer = "La pérdida fue de 0,99 en el conjunto de datos de la época 200 y se mantuvo estable en todo el experimento."
    monkeypatch.setattr(R, "make_provider", lambda config: ScriptedProvider([answer]))
    assert R.main([*base, "--no-filter-numbers"]) == 1  # cifra no respaldada: la verificación falla
    capsys.readouterr()
    monkeypatch.setattr(R, "make_provider", lambda config: ScriptedProvider([answer]))
    code = R.main([*base, "--filter-numbers"])
    err = capsys.readouterr().err
    assert code == 1 and "filtro de cifras: 1 de 1 frases quitadas" in err and "FALLO DE FORMATO" in err


def test_cli_filter_is_on_by_default_because_the_declared_rule_was_met_and_a_short_summary_without_removals_is_not_a_failure(tmp_path, monkeypatch, capsys):
    assert R.FILTER_NUMBERS_BY_DEFAULT is True and R.SUMMARY_MAX_WORDS == 150
    project = cli_project(tmp_path)
    answer = "El experimento llegó a la época 200 con una pérdida de 0,43 en el conjunto de datos."  # corto (< 30 palabras) pero sin cifras no respaldadas
    monkeypatch.setattr(R, "make_provider", lambda config: ScriptedProvider([answer]))
    assert R.main(["--agent", "summarizer", "context/data/processed/m.csv", "--root", str(project), "--log-dir", str(tmp_path / "l")]) == 0
    err = capsys.readouterr().err
    assert "filtro de cifras: 0 de 1 frases quitadas" in err and "FALLO DE FORMATO" not in err


def test_cli_code_reviewer_corrects_lines_by_default_and_raw_keeps_the_model_line(tmp_path, monkeypatch, capsys):
    project = cli_project(tmp_path)
    review = json.dumps({"hallazgos": [{"linea": 9, "cita": "return x + a", "problema": "p", "gravedad": "baja"}]})
    monkeypatch.setattr(R, "make_provider", lambda config: ScriptedProvider([review]))
    args = ["--agent", "code_reviewer", "src/amp_gan/mod.py", "--root", str(project), "--log-dir", str(tmp_path / "l")]
    assert R.main(args) == 0
    out = capsys.readouterr()
    shown = json.loads(out.out)
    assert shown["hallazgos"][0]["linea"] == 3 and shown["hallazgos"][0]["linea_modelo"] == 9 and "línea 9 -> 3" in out.err
    monkeypatch.setattr(R, "make_provider", lambda config: ScriptedProvider([review]))
    assert R.main([*args, "--raw"]) == 1  # la línea cruda 9 no existe: la verificación la rechaza, como siempre
    out = capsys.readouterr()
    assert json.loads(out.out)["hallazgos"][0]["linea"] == 9 and "salida cruda (--raw)" in out.err and "NO válida" in out.err


def test_new_sources_have_no_forbidden_mentions():
    forbidden = ["cla" + "ude", "anthro" + "pic", "asistente de " + "ia", "/ho" + "me/"]
    files = [SRC / "amp_agents" / n for n in ("rewrite.py", "router.py", "verifiers.py", "run.py", "providers.py", "eval_roles.py", "agents.toml", "fixtures/router_requests.json")]
    files += [SRC / "tests" / "test_agents_p4.py"]
    for file in files:
        text = file.read_text(encoding="utf-8").lower().replace(".claude", "")
        for word in forbidden:
            assert word not in text, f"{file.name} menciona {word!r}"

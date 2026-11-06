"""Tests de la parte determinista de la evaluación en vivo: siembra de defectos, verdad del localizador, corrección,
consultas del recuperador y tablas. No usan Ollama ni red (la evaluación en vivo es el script ``eval_roles``)."""

import ast
import difflib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from agents_helpers import write

from amp_agents import eval_roles as E
from amp_agents.retrieval import chunk_project
from amp_agents.verifiers import check_doc_citations, find_quote

SRC = Path(__file__).resolve().parents[1]
REAL_ROOT = SRC.parent


def real_docs():
    """Los documentos CONGELADOS de la evaluación (``amp_agents/fixtures/``), no los vivos del repositorio, que cambian."""
    return E.frozen_docs()


# -- criterios declarados -------------------------------------------------------------------------------------------
def test_declared_criteria_match_the_protocol():
    assert E.CRITERIA == {
        "locator": {"exact_rate_min": 0.80},
        "doc_checker": {"valid_citations_min": 0.90, "recovered_mean_min": 4.0, "false_positives_max": 2},
        "code_reviewer": {"recovered_mean_min": 3.0, "valid_citations_min": 0.90, "false_positives_max": 2},
        "summarizer": {"numbers_min": 0.95, "max_words": 150},
        "retriever": {"top3_min": 0.80},
    }
    assert E.REPS == {"locator": 2, "doc_checker": 3, "code_reviewer": 3, "summarizer": 3}
    assert E.ROLES == ("doc_checker", "code_reviewer", "summarizer", "retriever")  # el localizador está retirado de `all`
    assert E.ALL_ROLES == ("locator", "doc_checker", "code_reviewer", "summarizer", "retriever")


# -- siembra de incoherencias documentales ------------------------------------------------------------------------------
def test_seed_documents_records_the_exact_positions_and_types():
    roadmap, todo = real_docs()
    new_roadmap, new_todo, seeds = E.seed_documents(roadmap, todo)
    assert new_roadmap == roadmap and new_todo != todo
    assert len(seeds) == 6 and len({s["type"] for s in seeds}) == 6 and len({s["id"] for s in seeds}) == 6
    assert {s["id"] for s in seeds} == {"cifra", "estado", "fichero", "tests", "fecha", "notebook"}
    texts = {E.ROADMAP_REL: new_roadmap, E.TODO_REL: new_todo}
    for seed in seeds:
        line = texts[seed["file"]].splitlines()[seed["line"] - 1]
        assert seed["new"] in line and seed["old"] not in new_todo
        assert todo.count(seed["old"]) == 1  # el original existía una sola vez
        partner = texts[seed["partner_file"]].splitlines()[seed["partner_line"] - 1]
        assert partner  # la línea con la que choca existe
    # solo cambian las líneas sembradas (cada una una sola vez)
    changed = [i for i, (a, b) in enumerate(zip(todo.splitlines(), new_todo.splitlines()), start=1) if a != b]
    assert changed == sorted(s["line"] for s in seeds) and len(todo.splitlines()) == len(new_todo.splitlines())


def test_seed_documents_each_type_really_creates_a_contradiction():
    roadmap, todo = real_docs()
    new_roadmap, new_todo, seeds = E.seed_documents(roadmap, todo)
    by_id = {s["id"]: s for s in seeds}
    assert "3 de 5" in by_id["cifra"]["new"] and "cumple 4 de las 5" in new_roadmap
    assert "[x]" in by_id["estado"]["new"] and "- [ ] GAN condicional con **longitud e hidrofobicidad a la vez**" in new_roadmap
    assert "make_clean_dataset.py" in by_id["fichero"]["new"] and not (REAL_ROOT / "src" / "amp_gan" / "make_clean_dataset.py").exists()
    assert (REAL_ROOT / "src" / "amp_gan" / "make_clean_data.py").is_file()
    assert "46 tests" in by_id["tests"]["new"] and "49 tests" in new_roadmap
    assert "26-oct-2026" in by_id["fecha"]["new"] and "28-oct-2026" in new_roadmap
    assert "04_gan_longitud.ipynb" in by_id["notebook"]["new"] and "04_gan_condicional_longitud.ipynb" in new_roadmap
    assert not (REAL_ROOT / "src" / "notebooks" / "04_gan_longitud.ipynb").exists()


def test_seed_documents_rejects_missing_or_repeated_fragments():
    seeds = [{"id": "x", "type": "t", "file": E.TODO_REL, "old": "uno", "new": "dos", "partner_file": E.ROADMAP_REL, "partner_text": "otro"}]
    with pytest.raises(ValueError, match="0 veces"):
        E.seed_documents("otro\n", "nada\n", seeds)
    with pytest.raises(ValueError, match="2 veces"):
        E.seed_documents("otro\n", "uno uno\n", seeds)
    with pytest.raises(ValueError, match="choca"):
        E.seed_documents("nada\n", "uno\n", seeds)
    new_roadmap, new_todo, records = E.seed_documents("a\notro\n", "x\nuno y más\n", seeds)
    assert new_todo == "x\ndos y más\n" and records[0]["line"] == 2 and records[0]["partner_line"] == 2


def test_match_doc_findings_counts_a_seed_only_when_a_valid_citation_covers_its_line(tmp_path):
    roadmap, todo = real_docs()
    new_roadmap, new_todo, seeds = E.seed_documents(roadmap, todo)
    write(tmp_path / E.ROADMAP_REL, new_roadmap)
    write(tmp_path / E.TODO_REL, new_todo)
    cifra = next(s for s in seeds if s["id"] == "cifra")
    fecha = next(s for s in seeds if s["id"] == "fecha")
    obj = {
        "incoherencias": [
            {
                "descripcion": "cifra",
                "evidencia_a": {"ruta": E.TODO_REL, "cita": cifra["new"]},
                "evidencia_b": {"ruta": E.ROADMAP_REL, "cita": "cumple 4 de las 5 condiciones"},
            },
            {  # solo cita el documento no alterado: cuenta en el modo laxo, no en el estricto
                "descripcion": "fecha",
                "evidencia_a": {"ruta": E.ROADMAP_REL, "cita": "DRAMP 5.0, descargada el 28-oct-2026"},
                "evidencia_b": {"ruta": E.ROADMAP_REL, "cita": "Fuente de datos: DRAMP"},
            },
            {  # cita inventada: no recupera nada y es falso positivo
                "descripcion": "inventada",
                "evidencia_a": {"ruta": E.TODO_REL, "cita": "esto no está en ningún sitio"},
                "evidencia_b": {"ruta": E.ROADMAP_REL, "cita": "tampoco esto"},
            },
        ]
    }
    citations = check_doc_citations(obj, tmp_path)
    match = E.match_doc_findings(citations, seeds)
    assert match["recovered"] == ["cifra"]
    assert match["recovered_lenient"] == ["cifra", "fecha"]
    assert match["false_positives"] == [2]
    assert new_roadmap.splitlines()[fecha["partner_line"] - 1].count("28-oct-2026") == 1


def test_match_doc_findings_ignores_citations_spanning_many_lines():
    seeds = [{"id": "a", "file": "f.md", "line": 5, "partner_file": "g.md", "partner_line": 9}]
    wide = [{"item": 0, "side": "evidencia_a", "ruta": "f.md", "valid": True, "first_line": 1, "last_line": 10}]
    assert E.match_doc_findings(wide, seeds)["recovered"] == [] and E.match_doc_findings(wide, seeds)["false_positives"] == [0]
    narrow = [{"item": 0, "side": "evidencia_a", "ruta": "./f.md", "valid": True, "first_line": 5, "last_line": 5}]
    assert E.match_doc_findings(narrow, seeds)["recovered"] == ["a"]


def test_clean_copy_has_no_seeded_text():
    roadmap, todo = real_docs()
    _, new_todo, seeds = E.seed_documents(roadmap, todo)
    for seed in seeds:
        assert seed["old"] in todo and seed["new"] not in todo


# -- siembra de defectos de código ----------------------------------------------------------------------------------------
def real_code():
    return (REAL_ROOT / E.CODE_REL).read_text(encoding="utf-8")


def test_seed_code_defects_records_the_line_after_injection():
    source = real_code()
    text, seeds = E.seed_code_defects(source)
    assert len(seeds) == 5 and len({s["type"] for s in seeds}) == 5
    assert {s["type"] for s in seeds} == {"validación eliminada", "límite incorrecto", "signo invertido", "índice desplazado", "argumentos intercambiados"}
    ast.parse(text)  # sigue siendo Python válido
    lines = text.splitlines()
    for seed in seeds:
        assert seed["new"].splitlines()[0] in lines[seed["line"] - 1], seed["id"]
        assert seed["old"] not in text or seed["id"] == "validacion"
    by_id = {s["id"]: s for s in seeds}
    # posiciones independientes del algoritmo: se buscan las líneas alteradas directamente
    assert lines[by_id["signo"]["line"] - 1].strip() == "return float(negative - positive)"
    assert lines[by_id["indice"]["line"] - 1].strip().endswith("for aa in seq[1:]) / len(seq))")
    assert "net_charge(ph, s)" in lines[by_id["argumentos"]["line"] - 1]
    assert lines[by_id["limite"]["line"] - 1].strip() == "if fractions.size <= 20:"
    # la validación eliminada estaba una línea antes de la línea que ahora ocupa su sitio
    original = source.splitlines()
    assert original[by_id["validacion"]["line"] - 1].strip() == "_validate(seq)"
    assert lines[by_id["validacion"]["line"] - 1].strip() == "counts = Counter(seq)"
    assert text.count("_validate(seq)") == source.count("_validate(seq)") - 1


def test_seed_code_defects_changes_only_the_expected_lines():
    source = real_code()
    text, seeds = E.seed_code_defects(source)
    diff = [d for d in difflib.unified_diff(source.splitlines(), text.splitlines(), lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")]
    assert sum(d.startswith("-") for d in diff) == 5 and sum(d.startswith("+") for d in diff) == 4  # 4 cambios y 1 borrado
    assert len(text.splitlines()) == len(source.splitlines()) - 1
    assert min(b["line"] - a["line"] for a, b in zip(seeds, seeds[1:])) > 5  # defectos lejos entre sí (ventana de ±2)
    assert E.seed_code_defects(source)[0] == text  # determinista


def test_seed_code_defects_rejects_ambiguous_or_missing_fragments():
    with pytest.raises(ValueError, match="0 veces"):
        E.seed_code_defects("x = 1\n", [{"id": "a", "type": "t", "old": "y = 2\n", "new": "y = 3\n"}])
    with pytest.raises(ValueError, match="2 veces"):
        E.seed_code_defects("x = 1\nx = 1\n", [{"id": "a", "type": "t", "old": "x = 1\n", "new": "x = 2\n"}])


def test_seed_code_defects_tracks_shifts_caused_by_earlier_deletions():
    source = "a = 1\nvalidar()\nb = 2\nc = 3\nd = 4\n"
    seeds = [
        {"id": "borrado", "type": "validación eliminada", "old": "validar()\n", "new": ""},
        {"id": "cambio", "type": "índice desplazado", "old": "d = 4\n", "new": "d = 5\n"},
    ]
    text, records = E.seed_code_defects(source, seeds)
    assert text == "a = 1\nb = 2\nc = 3\nd = 5\n"
    assert [(r["id"], r["line"]) for r in records] == [("borrado", 2), ("cambio", 4)]
    assert text.splitlines()[3] == "d = 5"


def test_match_code_findings_uses_a_window_of_two_lines():
    seeds = [{"id": "a", "line": 10}, {"id": "b", "line": 50}]
    obj = {"hallazgos": [{"linea": 12}, {"linea": 13}, {"linea": 48}, {"linea": 51}, {"linea": 99}, {"linea": "x"}]}
    match = E.match_code_findings(obj, seeds)
    assert match["recovered"] == ["a", "b"] and match["false_positives"] == [1, 4, 5] and match["n_findings"] == 6
    assert E.match_code_findings({"hallazgos": []}, seeds) == {"recovered": [], "false_positives": [], "n_findings": 0}
    assert E.match_code_findings({"hallazgos": [{"linea": 10}]}, [])["false_positives"] == [0]
    assert E.match_code_findings({"hallazgos": [{"linea": 10}]}, seeds, tolerance=0)["recovered"] == ["a"]


# -- localizador: verdad y corrección ----------------------------------------------------------------------------------------
def mini_project(tmp_path: Path) -> Path:
    write(tmp_path / "src" / "amp_gan" / "conditions.py", "TOL = 0.25\n\n\ndef f(x):\n    return x * TOL\n\n\nclass Policy:\n    pass\n")
    write(tmp_path / "src" / "amp_gan" / "gan.py", "from amp_gan.conditions import f\nimport amp_gan.conditions\n\n\ndef g():\n    return f(1)\n")
    write(tmp_path / "src" / "tests" / "test_x.py", "def test_a():\n    from amp_gan.conditions import TOL\n    assert TOL\n")
    write(tmp_path / "src" / "tests" / "__pycache__" / "x.py", "def f(): pass\n")
    write(tmp_path / "src" / ".pytest_cache" / "y.py", "def f(): pass\n")
    return tmp_path


def test_truth_by_ast_and_regex(tmp_path):
    root = mini_project(tmp_path)
    assert E.find_definitions(root, "f") == [("src/amp_gan/conditions.py", 4)]
    assert E.find_definitions(root, "Policy") == [("src/amp_gan/conditions.py", 8)]
    assert E.find_definitions(root, "test_a") == [("src/tests/test_x.py", 1)]
    assert E.find_definitions(root, "nada") == []
    pattern = r"^\s*(?:from\s+amp_gan\.conditions\s+import\b|import\s+amp_gan\.conditions\b)"
    assert E.find_lines(root, pattern) == [("src/amp_gan/gan.py", 1), ("src/amp_gan/gan.py", 2), ("src/tests/test_x.py", 2)]
    assert E.find_lines(root, r"\bTOL\b") == [("src/amp_gan/conditions.py", 1), ("src/amp_gan/conditions.py", 5), ("src/tests/test_x.py", 2), ("src/tests/test_x.py", 3)]


def test_locator_questions_mix_and_truth_on_the_real_project():
    questions = E.locator_questions(REAL_ROOT)
    assert len(questions) == 10
    kinds = [q["kind"] for q in questions]
    assert kinds.count("definicion") == 5 and kinds.count("importacion") == 3 and kinds.count("constante") == 2
    assert all(q["expected"] for q in questions) and len({q["id"] for q in questions}) == 10
    by_id = {q["id"]: q for q in questions}
    assert by_id["def_Policy"]["expected"] == [("src/amp_agents/tools.py", next(n for n, line in enumerate((SRC / "amp_agents" / "tools.py").read_text(encoding="utf-8").splitlines(), 1) if line.startswith("class Policy:")))]
    assert by_id["def_mae_vs_reference"]["expected"][0][0] == "src/amp_gan/conditions.py"
    assert all(path.endswith(".py") for q in questions for path, _ in q["expected"])
    names = " ".join(q["question"] for q in questions)
    for needed in ("mae_vs_reference", "Policy", "run_agent", "fit_condition_spec", "evaluate_target", "amp_gan.conditions", "HYDRO_HIT_TOLERANCE"):
        assert needed in names


def test_grade_locator_requires_the_exact_set_and_existing_lines(tmp_path):
    root = mini_project(tmp_path)
    expected = [("src/amp_gan/conditions.py", 4)]

    def answer(*pairs):
        return json.dumps({"respuestas": [{"ruta": r, "linea": n, "texto": "x"} for r, n in pairs]})

    ok, info = E.grade_locator(answer(("src/amp_gan/conditions.py", 4)), expected, root)
    assert ok and info["missing"] == [] and info["extra"] == [] and info["lines_exist"]
    ok, info = E.grade_locator(answer(("./src/amp_gan/conditions.py", 4), ("src/amp_gan/conditions.py", 4)), expected, root)
    assert ok  # ruta con ./ y duplicados no cambian el conjunto
    ok, info = E.grade_locator(answer(("src/amp_gan/conditions.py", 4), ("src/amp_gan/gan.py", 1)), expected, root)
    assert not ok and info["extra"] == [("src/amp_gan/gan.py", 1)]
    ok, info = E.grade_locator(answer(("src/amp_gan/conditions.py", 5)), expected, root)
    assert not ok and info["missing"] == [("src/amp_gan/conditions.py", 4)]
    ok, info = E.grade_locator(answer(("src/amp_gan/conditions.py", 400)), [("src/amp_gan/conditions.py", 400)], root)
    assert not ok and info["errors"]  # coincide con lo esperado, pero la línea no existe
    ok, info = E.grade_locator("no lo sé", expected, root)
    assert not ok and info["errors"]
    ok, _ = E.grade_locator('{"respuestas": []}', [], root)
    assert ok  # «sin coincidencias» es la respuesta exacta si no hay nada que encontrar
    ok, info = E.grade_locator("```json\n" + answer(("src/amp_gan/conditions.py", 4)) + "\n```", expected, root)
    assert ok and info["strict_json"] is False


# -- resumidor, recuperador, tablas ------------------------------------------------------------------------------------------------
def test_trim_run_info_drops_noise_and_keeps_results():
    raw = json.dumps({"versions": {"python": "3"}, "backend_decision": {"x": 1}, "created_utc": "t", "variant": "cgan_hydro", "epochs": 200, "config": {"seed": 42}})
    trimmed = json.loads(E.trim_run_info(raw))
    assert trimmed == {"variant": "cgan_hydro", "epochs": 200, "config": {"seed": 42}}
    assert E.trim_run_info("no es json " * 1000) == ("no es json " * 1000)[: E.SUMMARY_MAX_CHARS]


def test_summary_inputs_exist_in_the_real_project_and_fit():
    processed = REAL_ROOT / "context" / "data" / "processed"
    if not processed.is_dir():  # los datos no se versionan: en un clon limpio no están
        return
    for metrics, info in E.SUMMARY_INPUTS.values():
        assert (processed / metrics).stat().st_size < E.SUMMARY_MAX_CHARS
        assert len(E.trim_run_info((processed / info).read_text(encoding="utf-8"))) < E.SUMMARY_MAX_CHARS


def test_retriever_queries_are_answerable_and_hand_written():
    assert len(E.RETRIEVER_QUERIES) == 15 and len({q for q, _ in E.RETRIEVER_QUERIES}) == 15
    chunks = chunk_project(REAL_ROOT)
    for query, truth in E.RETRIEVER_QUERIES:
        assert query.startswith("¿") and query.endswith("?") and truth
        assert any(any(c.ruta.endswith(f) and kw.lower() in c.texto.lower() for c in chunks) for f, kw in truth), query


def test_retriever_rank_takes_the_best_alternative():
    hits = [
        {"ruta": "context/constitution/governance.md", "texto": "Solo el ejecutor escribe en src"},
        {"ruta": "context/agents/executor.md", "texto": "Escritura: src/"},
    ]
    truth = [("executor.md", "Escritura"), ("governance.md", "escribe")]
    assert E.retriever_rank(hits, truth) == 1
    assert E.retriever_rank(hits, [("executor.md", "Escritura")]) == 2
    assert E.retriever_rank(hits, [("roadmap.md", "A.2")]) is None
    assert E.retriever_rank(hits, [("executor.md", "Escritura")], k=1) is None


def test_processor_label_like_ollama_ps():
    assert E.processor_label(100, 100) == "100% GPU" and E.processor_label(100, 0) == "100% CPU"
    assert E.processor_label(100, 91) == "9%/91% CPU/GPU" and E.processor_label(0, 0) == "desconocido"
    assert E.size_cpu_fraction(100, 91) == pytest.approx(0.09) and E.size_cpu_fraction(100, 100) == 0.0


def test_ollama_ps_without_server_returns_empty():
    assert E.ollama_ps("http://127.0.0.1:9", "m", timeout=1) == {}


def test_make_copy_excludes_data_and_the_evaluation_files(tmp_path):
    copy = E.make_copy(tmp_path / "copia")
    assert (copy / "src" / "amp_gan" / "conditions.py").is_file() and (copy / "context" / "constitution" / "governance.md").is_file()
    assert not (copy / "context" / "data").exists()
    assert not (copy / "src" / "amp_agents" / "eval_roles.py").exists()
    assert not list((copy / "src" / "tests").glob("test_agents_eval*.py"))
    assert (copy / "src" / "amp_agents" / "verifiers.py").is_file()
    assert not (copy / ".agent_runs").exists() and not list(copy.rglob("__pycache__"))


def test_workdir_must_be_outside_the_repository(tmp_path):
    with pytest.raises(SystemExit):
        E._ensure_outside_repo(REAL_ROOT / "src")
    E._ensure_outside_repo(tmp_path / "fuera")


def test_tables_and_report(tmp_path):
    criteria = [("Criterio uno", ">= 80 %", "90 % (9/10)", True), ("Criterio dos", "<= 2", "5", False)]
    table = E.criteria_table("locator", "llama3.2:3b", criteria)
    assert "| locator (llama3.2:3b) | Criterio uno | >= 80 % | 90 % (9/10) | SÍ |" in table and "| NO |" in table
    run = E.RunRecord("locator", "m", "q1", 1, True, "final", 3, 100, 90, 4.5, json_valid=True, json_strict=False, context_length=8192, processor="100% GPU", notes="a|b")
    assert "| q1 | 1 | sí | final | 3 | 100 | 4.5 | 8192 | 100% GPU | sí (no) | a/b |" in E.runs_table([run])
    (tmp_path / "locator.json").write_text(json.dumps({"role": "locator", "model": "m", "env_variant": 1, "criteria": [list(c) for c in criteria]}), encoding="utf-8")
    (tmp_path / "seeds_x.json").write_text("[]", encoding="utf-8")
    report = E.build_report(tmp_path)
    assert report.count("\n") == 3 and "locator (entorno v1)" in report


def test_recorder_writes_each_run_as_it_finishes(tmp_path, capsys):
    rec = E.Recorder(tmp_path, "x")
    rec.add(E.RunRecord("locator", "m", "q1", 1, False, "timeout", 1, None, None, 2.0))
    lines = (tmp_path / "x.runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["stop_reason"] == "timeout"
    capsys.readouterr()


def test_script_is_not_collected_by_pytest_and_has_a_cli():
    assert not E.__name__.startswith("test_")
    done = subprocess.run([sys.executable, "-m", "amp_agents.eval_roles", "--help"], cwd=SRC, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and "locator" in done.stdout and "--env-variant" in done.stdout
    assert "NO es un test" in E.__doc__.upper() or "no es un test" in E.__doc__


# -- documentos congelados de la evaluación del doc_checker --------------------------------------------------------------------
def test_frozen_documents_are_independent_of_the_live_ones():
    roadmap, todo = E.frozen_docs()
    assert (E.FIXTURES_DIR / "roadmap.md").is_file() and (E.FIXTURES_DIR / "TODO.md").is_file()
    assert "hecha en `feat/wgan-gp` (4 de 5 condiciones" in todo and "cumple 4 de las 5 condiciones" in roadmap  # texto de la evaluación original
    new_roadmap, new_todo, seeds = E.seed_documents(roadmap, todo)  # las seis siembras se aplican, una vez cada una
    assert len(seeds) == 6 and {s["file"] for s in seeds} == {E.TODO_REL} and E.TODO_REL == "context/TODO.md"
    assert new_roadmap == roadmap and new_todo != todo


def test_make_copy_leaves_the_frozen_documents_out_of_the_consulted_material(tmp_path):
    copy = E.make_copy(tmp_path / "copia")
    assert not (copy / "src" / "amp_agents" / "fixtures").exists()
    assert (E.FIXTURES_DIR / "TODO.md").is_file()  # el original no se toca


# -- doc_checker_ss: mismo protocolo y criterios que doc_checker --------------------------------------------------------------
def test_doc_checker_ss_reads_the_criteria_of_doc_checker_without_duplicating_them():
    assert E.base_role("doc_checker_ss") == "doc_checker" and E.base_role("summarizer") == "summarizer"
    assert "doc_checker_ss" not in E.CRITERIA and "doc_checker_ss" not in E.REPS  # una sola entrada con los números
    assert "doc_checker_ss" not in E.ROLES and "doc_checker_ss" not in E.ALL_ROLES  # no entra en `all`
    assert "doc_checker_ss" in E.COMMAND_ROLES and "doc_checker_ss" in E.EVALUATORS


def run_ss_evaluation(tmp_path, monkeypatch, answers, role="doc_checker_ss", limit=None, reps=1):
    """Ejecuta ``evaluate_role`` con un proveedor guionizado (sin Ollama) y devuelve ``(resultado, results_dir, proveedor)``."""
    from amp_agents import run as R
    from amp_agents.providers import ScriptedProvider

    scripted = ScriptedProvider(answers)
    monkeypatch.setattr(R, "make_provider", lambda config: scripted)
    monkeypatch.setattr(E, "server_models", lambda base_url: ["phi4-mini:latest"])
    monkeypatch.setattr(E, "ollama_ps", lambda *a, **k: {})
    monkeypatch.setattr(E, "ollama_stop", lambda model: "ok")
    monkeypatch.setattr(E, "ollama_ps_text", lambda: "")
    args = E.argparse.Namespace(workdir=str(tmp_path / "w"), results_dir=str(tmp_path / "r"), env_variant=1, tag="", reps=reps, limit=limit, drop=None)
    return E.evaluate_role(role, args), tmp_path / "r", scripted


def ss_answers():
    roadmap, todo = E.frozen_docs()
    _, _, seeds = E.seed_documents(roadmap, todo)
    cifra = next(s for s in seeds if s["id"] == "cifra")
    seeded = {
        "incoherencias": [
            {
                "descripcion": "cifra distinta",
                "evidencia_a": {"ruta": E.TODO_REL, "cita": cifra["new"]},
                "evidencia_b": {"ruta": E.ROADMAP_REL, "cita": "cumple 4 de las 5 condiciones"},
            }
        ]
    }
    return json.dumps(seeded), json.dumps({"incoherencias": []})


def test_doc_checker_ss_is_evaluated_with_its_own_name_and_the_same_protocol(tmp_path, monkeypatch):
    out, results, scripted = run_ss_evaluation(tmp_path, monkeypatch, list(ss_answers()))
    assert out["role"] == "doc_checker_ss" and out["protocol"] == E.CRITERIA["doc_checker"]
    assert [r["role"] for r in out["runs"]] == ["doc_checker_ss", "doc_checker_ss"] and [r["case"] for r in out["runs"]] == ["seeded", "clean"]
    assert all(r["success"] and r["json_valid"] and r["stop_reason"] == "final" and r["steps"] == 1 and not r["copy_modified"] for r in out["runs"])
    for name in ("doc_checker_ss.json", "doc_checker_ss.md", "doc_checker_ss.runs.jsonl", "seeds_doc_checker_ss.json"):
        assert (results / name).is_file(), name
    assert not (results / "doc_checker.json").exists()  # no pisa los resultados del rol base
    criteria = out["criteria"]
    assert [c[0] for c in criteria] == [c[0] for c in E._doc_summary([], [])["criteria"]] and [c[1] for c in criteria] == [">= 90 %", ">= 4", "<= 2"]
    assert out["metrics"]["valid_citations"] == 2 == out["metrics"]["citations"] and out["metrics"]["recovered_per_rep"] == [1]
    assert criteria[0][3] is True and criteria[1][3] is False and criteria[2][3] is True  # 1 de 6 < 4: el umbral no se rebaja
    # una pasada, sin herramientas, con los documentos CONGELADOS (sembrados en la copia del caso «seeded») en el mensaje
    user = scripted.received[0][1]["content"]
    assert "### context/constitution/roadmap.md" in user and "### context/TODO.md" in user and "3 de 5 condiciones" in user
    assert "4 de 5 condiciones" in scripted.received[1][1]["content"]  # la copia limpia lleva el texto original
    assert (tmp_path / "w" / "doc_checker_ss_seeded").is_dir() and (tmp_path / "w" / "doc_checker_ss_clean").is_dir()
    assert not (tmp_path / "w" / "doc_checker_ss_seeded" / "src" / "amp_agents" / "fixtures").exists()
    assert "doc_checker_ss" in (results / "doc_checker_ss.md").read_text(encoding="utf-8")
    assert scripted.timeouts == [E.get_agent_config("doc_checker_ss").max_seconds] * 2


def test_doc_checker_ss_counts_invented_quotes_as_invalid(tmp_path, monkeypatch):
    invented = json.dumps({"incoherencias": [{"descripcion": "d", "evidencia_a": {"ruta": E.TODO_REL, "cita": "frase inventada"}, "evidencia_b": {"ruta": E.ROADMAP_REL, "cita": "otra inventada"}}]})
    out, _, _ = run_ss_evaluation(tmp_path, monkeypatch, [invented, '{"incoherencias": []}'])
    assert out["metrics"]["valid_citations"] == 0 and out["metrics"]["citations"] == 2
    assert out["criteria"][0][3] is False
    assert [r["success"] for r in out["runs"]] == [False, True]


def test_doc_checker_ss_limit_runs_only_the_first_cases(tmp_path, monkeypatch):
    out, _, scripted = run_ss_evaluation(tmp_path, monkeypatch, [ss_answers()[0]], limit=1)
    assert [r["case"] for r in out["runs"]] == ["seeded"] and len(scripted.received) == 1
    assert out["criteria"][2][2] == "- []" and out["criteria"][2][3] is False  # sin copia limpia no hay medida de falsos positivos


def test_doc_checker_agent_evaluation_also_uses_the_frozen_documents(tmp_path, monkeypatch):
    from amp_agents import run as R
    from amp_agents.providers import AssistantMessage, ScriptedProvider

    monkeypatch.setattr(E, "server_models", lambda base_url: ["phi4-mini"])
    seeded, clean = ss_answers()
    scripted = ScriptedProvider([AssistantMessage.calls(("read_file", {"path": E.TODO_REL})), seeded, clean])
    monkeypatch.setattr(R, "make_provider", lambda config: scripted)
    monkeypatch.setattr(E, "ollama_ps", lambda *a, **k: {})
    monkeypatch.setattr(E, "ollama_stop", lambda model: "ok")
    monkeypatch.setattr(E, "ollama_ps_text", lambda: "")
    args = E.argparse.Namespace(workdir=str(tmp_path / "w"), results_dir=str(tmp_path / "r"), env_variant=1, tag="", reps=1, limit=None, drop=None)
    out = E.evaluate_role("doc_checker", args)
    assert out["role"] == "doc_checker" and [r["role"] for r in out["runs"]] == ["doc_checker", "doc_checker"]
    assert (tmp_path / "r" / "doc_checker.json").is_file() and (tmp_path / "r" / "seeds_doc_checker.json").is_file()
    read = scripted.received[1][-1]["content"]  # resultado de read_file sobre la copia sembrada
    assert "3 de 5 condiciones" in read
    assert out["metrics"]["valid_citations"] == 2


def test_all_can_include_doc_checker_ss_only_when_asked(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(E, "evaluate_role", lambda role, args: called.append(role))
    work = str(tmp_path / "w")
    assert E.main(["all", "--workdir", work, "--roles", "doc_checker,doc_checker_ss"]) == 0 and called == ["doc_checker", "doc_checker_ss"]
    called.clear()
    assert E.main(["doc_checker_ss", "--workdir", work]) == 0 and called == ["doc_checker_ss"]


# -- --model / --num-ctx: evaluar otro modelo en un rol existente sin tocar agents.toml ---------------------------------------
def trial_args(tmp_path, model=None, num_ctx=None, tag="", reps=1, limit=None):
    return E.argparse.Namespace(workdir=str(tmp_path / "w"), results_dir=str(tmp_path / "r"), env_variant=1, tag=tag, reps=reps, limit=limit, drop=None, model=model, num_ctx=num_ctx)


def patch_live(monkeypatch, scripted, installed, configs=None):
    """Sustituye Ollama por un proveedor guionizado; ``configs`` recoge la configuración con la que se crea cada proveedor."""
    from amp_agents import run as R

    def make(config):
        if configs is not None:
            configs.append(config)
        return scripted

    monkeypatch.setattr(R, "make_provider", make)
    monkeypatch.setattr(E, "make_provider", make)
    monkeypatch.setattr(E, "server_models", lambda base_url: installed)
    monkeypatch.setattr(E, "ollama_ps", lambda *a, **k: {"processor": "100% GPU", "context_length": 12288})
    monkeypatch.setattr(E, "ollama_stop", lambda model: "ok")
    monkeypatch.setattr(E, "ollama_ps_text", lambda: "")


def test_trial_for_names_the_role_with_the_model_and_keeps_the_other_options():
    plain = E.trial_for("doc_checker_ss", E.argparse.Namespace())
    assert plain.label == "doc_checker_ss" and plain.overrides == {} and plain.model is None
    t = E.trial_for("doc_checker_ss", E.argparse.Namespace(model="gemma3:4b", num_ctx=None))
    assert t.label == "doc_checker_ss@gemma3-4b" and t.overrides == {"model": "gemma3:4b"} and t.model == "gemma3:4b"
    t = E.trial_for("doc_checker_ss", E.argparse.Namespace(model="gemma3:4b", num_ctx=10240))
    assert t.overrides["options"] == {"num_ctx": 10240, "temperature": 0.2}  # solo cambia num_ctx
    assert E.trial_for("summarizer", E.argparse.Namespace(model=None, num_ctx=4096)).label == "summarizer@ctx4096"
    assert E.safe_model_name("hf.co/usuario/modelo:Q4_K_M") == "hf.co-usuario-modelo-Q4_K_M"
    with pytest.raises(SystemExit):
        E.trial_for("summarizer", E.argparse.Namespace(model=None, num_ctx=10))


def test_model_trial_uses_the_model_in_files_tables_and_requests_without_editing_the_config(tmp_path, monkeypatch):
    from amp_agents.providers import ScriptedProvider

    before = E.get_agent_config("doc_checker_ss")
    configs: list = []
    scripted = ScriptedProvider(["listo", *ss_answers()])  # el calentamiento consume la primera respuesta
    patch_live(monkeypatch, scripted, ["gemma3:4b", "phi4-mini:latest"], configs)
    out = E.evaluate_role("doc_checker_ss", trial_args(tmp_path, model="gemma3:4b", num_ctx=10240, tag="g3"))
    results = tmp_path / "r"
    for name in ("doc_checker_ss@gemma3-4b_g3.json", "doc_checker_ss@gemma3-4b_g3.md", "doc_checker_ss@gemma3-4b_g3.runs.jsonl", "seeds_doc_checker_ss@gemma3-4b.json"):
        assert (results / name).is_file(), name
    assert not (results / "doc_checker_ss_g3.json").exists()  # no pisa los resultados del modelo por defecto
    assert out["role"] == "doc_checker_ss@gemma3-4b" and out["command_role"] == "doc_checker_ss" and out["model"] == "gemma3:4b" and out["num_ctx"] == 10240
    assert [r["role"] for r in out["runs"]] == ["doc_checker_ss@gemma3-4b"] * 2 and {r["model"] for r in out["runs"]} == {"gemma3:4b"}
    assert [r["num_ctx_config"] for r in out["runs"]] == [10240, 10240]
    assert "doc_checker_ss@gemma3-4b (gemma3:4b)" in (results / "doc_checker_ss@gemma3-4b_g3.md").read_text(encoding="utf-8")
    assert "doc_checker_ss@gemma3-4b" in E.build_report(results)
    # el protocolo no cambia: mismas repeticiones del protocolo (aquí 1 por --reps) y mismos criterios
    assert out["protocol"] == E.CRITERIA["doc_checker"] and [c[1] for c in out["criteria"]] == [">= 90 %", ">= 4", "<= 2"]
    # calentamiento: una petición aparte, con el mismo modelo y num_ctx, que no cuenta como repetición
    assert out["warmup"]["ok"] and out["warmup"]["ps"]["processor"] == "100% GPU" and len(out["runs"]) == 2
    assert len(scripted.received) == 3 and scripted.received[0][0]["content"].startswith("Responde solo")
    assert all(c.model == "gemma3:4b" and c.options["num_ctx"] == 10240 for c in configs)
    assert scripted.timeouts[0] == E.get_agent_config("doc_checker_ss").timeout  # el calentamiento también tiene tope de tiempo
    assert E.get_agent_config("doc_checker_ss") == before and E.get_agent_config("doc_checker_ss").model == "phi4-mini"


def test_model_trial_works_with_agent_mode_roles_and_the_installed_tag_may_be_implicit(tmp_path, monkeypatch):
    from amp_agents.providers import AssistantMessage, ScriptedProvider

    seeded, clean = ss_answers()
    scripted = ScriptedProvider(["listo", AssistantMessage.calls(("read_file", {"path": E.TODO_REL})), seeded, clean])
    patch_live(monkeypatch, scripted, ["qwen3:4b", "otro:latest"])
    out = E.evaluate_role("doc_checker", trial_args(tmp_path, model="otro"))  # «otro» coincide con «otro:latest»
    assert out["role"] == "doc_checker@otro" and [r["role"] for r in out["runs"]] == ["doc_checker@otro"] * 2
    assert (tmp_path / "r" / "doc_checker@otro.json").is_file() and out["metrics"]["valid_citations"] == 2


def test_model_trial_fails_before_any_request_when_the_model_is_not_installed(tmp_path, monkeypatch):
    from amp_agents.providers import ScriptedProvider

    scripted = ScriptedProvider([])
    configs: list = []
    patch_live(monkeypatch, scripted, ["phi4-mini:latest"], configs)
    with pytest.raises(SystemExit) as error:
        E.evaluate_role("doc_checker_ss", trial_args(tmp_path, model="no-existe:7b"))
    assert "no-existe:7b" in str(error.value) and "no está instalado" in str(error.value) and "phi4-mini:latest" in str(error.value)
    assert scripted.received == [] and configs == []  # ni calentamiento ni repeticiones
    assert list((tmp_path / "r").glob("*.runs.jsonl")) == []
    monkeypatch.setattr(E, "server_models", lambda base_url: None)
    with pytest.raises(SystemExit, match="no responde"):
        E.evaluate_role("doc_checker_ss", trial_args(tmp_path, model="gemma3:4b"))


def test_warmup_failure_stops_the_evaluation_without_counting_repetitions(tmp_path, monkeypatch):
    from amp_agents.providers import ProviderTimeout, ScriptedProvider

    scripted = ScriptedProvider([ProviderTimeout("el servidor no respondió en 240 s")])
    patch_live(monkeypatch, scripted, ["gemma3:4b"])
    with pytest.raises(SystemExit, match="calentamiento"):
        E.evaluate_role("doc_checker_ss", trial_args(tmp_path, model="gemma3:4b"))
    assert len(scripted.received) == 1  # solo se hizo la petición de calentamiento
    assert not list((tmp_path / "r").glob("*.json"))


def test_without_model_there_is_no_warmup_and_names_do_not_change(tmp_path, monkeypatch):
    out, results, scripted = run_ss_evaluation(tmp_path, monkeypatch, list(ss_answers()))
    assert out["warmup"] is None and len(scripted.received) == 2 and out["role"] == "doc_checker_ss"


def test_warm_up_embedding_model_calls_embed_with_a_timeout(monkeypatch):
    calls = []

    class Embedder:
        def embed(self, texts, timeout=None):
            calls.append((list(texts), timeout))
            return [[0.1, 0.2]]

    monkeypatch.setattr(E, "ollama_ps", lambda *a, **k: {})
    warm = E.warm_up(E.get_agent_config("retriever", model="bge-m3"), Embedder())
    assert warm["ok"] and calls == [(["calentamiento"], E.get_agent_config("retriever").timeout)]


def test_model_trial_for_the_retriever_uses_the_given_embedding_model(tmp_path, monkeypatch):
    from amp_agents import run as R

    seen: list = []

    class Embedder:
        def embed(self, texts, timeout=None):
            return [[float(len(t) % 7 + 1), 1.0, float(len(t) % 3)] for t in texts]

        def embed_query(self, text):  # por si el recuperador la usa
            return self.embed([text])[0]

    def make(config):
        seen.append(config.model)
        return Embedder()

    monkeypatch.setattr(R, "make_provider", make)
    monkeypatch.setattr(E, "make_provider", make)
    monkeypatch.setattr(E, "server_models", lambda base_url: ["nomic-embed-text:latest", "bge-m3:latest"])
    monkeypatch.setattr(E, "ollama_ps", lambda *a, **k: {})
    monkeypatch.setattr(E, "ollama_stop", lambda model: "ok")
    monkeypatch.setattr(E, "ollama_ps_text", lambda: "")
    out = E.evaluate_role("retriever", trial_args(tmp_path, model="bge-m3", limit=2))
    assert out["role"] == "retriever@bge-m3" and out["model"] == "bge-m3" and out["warmup"]["ok"]
    assert set(seen) == {"bge-m3"} and {r["model"] for r in out["runs"]} == {"bge-m3"}
    assert "(bge-m3)" in out["criteria"][0][0] and (tmp_path / "r" / "retriever@bge-m3.json").is_file()


def test_cli_accepts_model_and_num_ctx_only_for_a_single_role(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(E, "evaluate_role", lambda role, args: seen.append((role, args.model, args.num_ctx)))
    assert E.main(["doc_checker_ss", "--workdir", str(tmp_path / "w"), "--model", "gemma3:4b", "--num-ctx", "12288"]) == 0
    assert seen == [("doc_checker_ss", "gemma3:4b", 12288)]
    assert E.main(["summarizer", "--workdir", str(tmp_path / "w")]) == 0 and seen[-1] == ("summarizer", None, None)
    for command in ("all", "report"):
        with pytest.raises(SystemExit):
            E.main([command, "--workdir", str(tmp_path / "w"), "--model", "gemma3:4b"])


def test_declared_protocol_is_untouched_by_the_model_option():
    assert E.REPS == {"locator": 2, "doc_checker": 3, "code_reviewer": 3, "summarizer": 3}
    assert E.CRITERIA["doc_checker"] == {"valid_citations_min": 0.90, "recovered_mean_min": 4.0, "false_positives_max": 2}

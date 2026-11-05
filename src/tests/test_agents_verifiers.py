"""Tests de los verificadores (funciones puras, sin red): casos válidos e inválidos de cada uno."""

import json

import pytest
from agents_helpers import write

from amp_agents import verifiers as V

SOURCE = "def f(x):\n    return x + 1\n\n\ndef g(x):\n    return x * 2\n"


# -- JSON -----------------------------------------------------------------------------------------------------
def test_parse_json_strict_fenced_and_surrounded():
    assert V.parse_json_output('{"a": 1}') == ({"a": 1}, True, None)
    obj, strict, error = V.parse_json_output('Aquí va:\n```json\n{"a": [1, 2]}\n```\nfin')
    assert obj == {"a": [1, 2]} and strict is False and error is None
    obj, strict, _ = V.parse_json_output('Respuesta: {"a": {"b": 2}} y nada más {"c": 3}')
    assert obj == {"a": {"b": 2}} and strict is False


@pytest.mark.parametrize("text", ["", "   ", "sin json", "{roto: 1}", "[1, 2]", '{"a": 1', "```json\n{\n```"])
def test_parse_json_invalid_outputs(text):
    obj, strict, error = V.parse_json_output(text)
    assert obj is None and strict is False and error


def test_validate_locator():
    good = {"respuestas": [{"ruta": "src/a.py", "linea": 3, "texto": "def f():"}]}
    assert V.validate_locator(good) == [] and V.validate_locator({"respuestas": []}) == []
    assert any("falta la clave" in e for e in V.validate_locator({}))
    assert any("lista" in e for e in V.validate_locator({"respuestas": "x"}))
    assert any("linea" in e for e in V.validate_locator({"respuestas": [{"ruta": "a", "linea": "3", "texto": ""}]}))
    assert any("linea" in e for e in V.validate_locator({"respuestas": [{"ruta": "a", "linea": True, "texto": ""}]}))
    assert any("ruta" in e for e in V.validate_locator({"respuestas": [{"ruta": "", "linea": 1, "texto": ""}]}))
    assert any("claves no previstas" in e for e in V.validate_locator({"respuestas": [], "extra": 1}))
    assert V.validate_locator([]) == ["la salida no es un objeto JSON"]


def test_validate_doc_checker():
    ev = {"ruta": "a.md", "cita": "x"}
    good = {"incoherencias": [{"descripcion": "d", "evidencia_a": ev, "evidencia_b": ev}]}
    assert V.validate_doc_checker(good) == [] and V.validate_doc_checker({"incoherencias": []}) == []
    bad = {"incoherencias": [{"descripcion": "", "evidencia_a": ev, "evidencia_b": {"ruta": "a.md"}}]}
    errors = V.validate_doc_checker(bad)
    assert any("descripcion" in e for e in errors) and any("evidencia_b.cita" in e for e in errors)
    assert any("evidencia_a" in e for e in V.validate_doc_checker({"incoherencias": [{"descripcion": "d", "evidencia_b": ev}]}))
    assert V.validate_doc_checker({"respuestas": []})


def test_validate_code_reviewer():
    good = {"hallazgos": [{"linea": 2, "cita": "return x", "problema": "p", "gravedad": "alta"}]}
    assert V.validate_code_reviewer(good) == [] and V.validate_code_reviewer({"hallazgos": []}) == []
    bad = {"hallazgos": [{"linea": "2", "cita": "", "problema": "p", "gravedad": "grave"}]}
    errors = V.validate_code_reviewer(bad)
    assert any("linea" in e for e in errors) and any("cita" in e for e in errors) and any("gravedad" in e for e in errors)
    assert V.validate_code_reviewer({"hallazgos": [3]})


# -- citas y líneas -----------------------------------------------------------------------------------------------
def test_find_quote_exact_multiline_and_collapsed_whitespace():
    text = "uno\n  dos   tres\ncuatro\n"
    assert V.find_quote(text, "dos   tres") == (2, 2)
    assert V.find_quote(text, "  uno\n  dos  ") == (1, 2)
    assert V.find_quote(text, "dos tres") == (2, 2)  # espacios colapsados
    assert V.find_quote(text, "tres cuatro") == (2, 3)
    assert V.find_quote(text, "cinco") is None and V.find_quote(text, "   ") is None and V.find_quote(text, "") is None


def test_read_project_text_refuses_escapes_and_missing_files(tmp_path):
    root = tmp_path / "p"
    write(root / "src" / "a.py", "x = 1\n")
    write(tmp_path / "fuera.txt", "secreto\n")
    assert V.read_project_text(root, "src/a.py") == "x = 1\n" and V.read_project_text(root, "./src/a.py") == "x = 1\n"
    assert V.read_project_text(root, "../fuera.txt") is None and V.read_project_text(root, "src") is None
    assert V.read_project_text(root, "src/no.py") is None and V.read_project_text(root, "") is None and V.read_project_text(root, None) is None


def test_check_locator_citations(tmp_path):
    write(tmp_path / "src" / "a.py", "import os\n\n\ndef f(x):\n    return x\n")
    obj = {
        "respuestas": [
            {"ruta": "src/a.py", "linea": 4, "texto": "def f(x):"},
            {"ruta": "src/a.py", "linea": 40, "texto": "def f(x):"},
            {"ruta": "src/b.py", "linea": 1, "texto": "x"},
            {"ruta": "./src/a.py", "linea": 5, "texto": "otra cosa"},
        ]
    }
    checks = V.check_locator_citations(obj, tmp_path)
    assert [c["line_exists"] for c in checks] == [True, False, False, True]
    assert [c["file_exists"] for c in checks] == [True, True, False, True]
    assert [c["texto_ok"] for c in checks] == [True, False, False, False]


def test_verify_locator_end_to_end(tmp_path):
    write(tmp_path / "src" / "a.py", "def f():\n    pass\n")
    ok = V.verify("locator", json.dumps({"respuestas": [{"ruta": "src/a.py", "linea": 1, "texto": "def f():"}]}), root=tmp_path)
    assert ok.valid and ok.details["strict_json"] is True and ok.details["schema_ok"] is True
    invented = V.verify("locator", json.dumps({"respuestas": [{"ruta": "src/a.py", "linea": 9, "texto": "x"}]}), root=tmp_path)
    assert not invented.valid and "no existe" in invented.errors[0] and invented.details["schema_ok"] is True
    assert V.verify("locator", '{"respuestas": []}', root=tmp_path).valid
    garbage = V.verify("locator", "no sé", root=tmp_path)
    assert not garbage.valid and garbage.details["parsed"] is False and garbage.details["schema_ok"] is False
    wrong = V.verify("locator", '{"otra": 1}', root=tmp_path)
    assert not wrong.valid and wrong.details["parsed"] is True and wrong.details["schema_ok"] is False


def test_check_doc_citations_and_verify(tmp_path):
    write(tmp_path / "context" / "a.md", "- [x] Tarea uno (49 tests)\n- [ ] Tarea dos\n")
    write(tmp_path / "context" / "b.md", "- [x] Tarea uno (46 tests)\n")
    good = {
        "incoherencias": [
            {
                "descripcion": "tests distintos",
                "evidencia_a": {"ruta": "context/a.md", "cita": "Tarea uno (49 tests)"},
                "evidencia_b": {"ruta": "context/b.md", "cita": "Tarea uno (46 tests)"},
            }
        ]
    }
    checks = V.check_doc_citations(good, tmp_path)
    assert [c["valid"] for c in checks] == [True, True] and checks[0]["first_line"] == 1
    assert V.verify("doc_checker", json.dumps(good), root=tmp_path).valid
    bad = json.loads(json.dumps(good))
    bad["incoherencias"][0]["evidencia_b"]["cita"] = "Tarea uno (47 tests)"
    result = V.verify("doc_checker", json.dumps(bad), root=tmp_path)
    assert not result.valid and "cita no literal" in result.errors[0]
    bad["incoherencias"][0]["evidencia_a"]["ruta"] = "context/inventado.md"
    assert [c["valid"] for c in V.check_doc_citations(bad, tmp_path)] == [False, False]
    assert V.verify("doc_checker", '{"incoherencias": []}', root=tmp_path).valid
    assert not V.verify("doc_checker", "texto libre", root=tmp_path).valid


def test_check_review_citations_and_verify():
    obj = {
        "hallazgos": [
            {"linea": 2, "cita": "return x + 1", "problema": "p", "gravedad": "alta"},
            {"linea": 6, "cita": "return x + 1", "problema": "cita en otra línea", "gravedad": "media"},
            {"linea": 99, "cita": "return x * 2", "problema": "línea inexistente", "gravedad": "baja"},
            {"linea": 5, "cita": "def h(x):", "problema": "cita inventada", "gravedad": "baja"},
        ]
    }
    checks = V.check_review_citations(obj, SOURCE)
    assert [c["valid"] for c in checks] == [True, True, False, False]
    assert [c["on_line"] for c in checks] == [True, False, False, False]
    assert [c["line_exists"] for c in checks] == [True, True, False, True]
    assert [c["quote_found"] for c in checks] == [True, True, True, False]
    result = V.verify("code_reviewer", json.dumps(obj), source=SOURCE)
    assert not result.valid and len(result.errors) == 2
    only_good = {"hallazgos": obj["hallazgos"][:1]}
    assert V.verify("code_reviewer", json.dumps(only_good), source=SOURCE).valid
    assert V.verify("code_reviewer", '{"hallazgos": []}', source=SOURCE).valid
    assert not V.verify("code_reviewer", '{"hallazgos": [{"linea": 1}]}', source=SOURCE).valid


def test_line_near():
    assert V.line_near(10, 12) and V.line_near(12, 10) and V.line_near(10, 10)
    assert not V.line_near(10, 13) and V.line_near(10, 13, tolerance=3)


# -- cifras, palabras, idioma ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("summary", "source", "expected_missing"),
    [
        ("La pérdida fue 0,43.", "loss=0.4321", []),  # coma decimal y redondeo
        ("La pérdida fue 0.43.", "loss=0.4321", []),  # punto decimal
        ("La pérdida fue 0,5.", "loss=0.4321", ["0,5"]),  # redondeo mal hecho
        ("El 37,5 % de las muestras.", "fraction=0.375", []),  # porcentaje obvio
        ("El 37,5 % de las muestras.", "porcentaje=37.5", []),
        ("Se generaron 2000 secuencias.", "n,2000,2000,1,0.0115", []),  # CSV con comas como separador
        ("Se generaron 2 000 secuencias.", "n=2000", []),  # miles con espacio
        ("Hubo 21 754 filas.", "total 21.754 filas", []),  # miles con punto en la entrada
        ("Se alcanzó 1,2e-05.", "lr=1.2e-05", []),
        ("Había 3 modelos y 7 épocas.", "modelos=3", ["7"]),
        ("Sin cifras en este texto.", "x=1", []),
        ("1. Primer punto\n2. Segundo punto con 5 valores", "n=5", []),  # los marcadores de lista no cuentan
        ("La media fue 9,79.", "length_mean=9.7963982", ["9,79"]),  # truncar no es redondear
    ],
)
def test_numbers_supported(summary, source, expected_missing):
    result = V.check_numbers_supported(summary, source)
    assert result["missing"] == expected_missing
    assert result["total"] - result["supported"] == len(expected_missing)
    assert result["ratio"] == (1.0 if result["total"] == 0 else result["supported"] / result["total"])


def test_extract_numbers_details():
    numbers = V.extract_numbers("a 12,50 b 3% c 7 d 1.5e3")
    assert [(n.values[0], n.decimals, n.percent) for n in numbers] == [(12.5, 2, False), (3.0, 0, True), (7.0, 0, False), (1500.0, 0, False)]
    assert V.extract_numbers("v2.0 y p10") == []  # pegados a letras: identificadores, no cifras


def test_count_words_and_spanish_score():
    assert V.count_words("uno dos  tres, 4,5 --- ¿cuatro?") == 5
    assert V.count_words("") == 0 and V.count_words("   ") == 0
    spanish = "El experimento terminó con una pérdida baja y los resultados de la última época son estables."
    english = "The experiment ended with a low loss and the results of the last epoch are stable in this run."
    assert V.spanish_score(spanish)["is_spanish"] and not V.spanish_score(english)["is_spanish"]
    assert V.spanish_score(english)["ratio"] < 0.3 and V.spanish_score("")["ratio"] == 0.0
    assert not V.spanish_score("El valor.")["is_spanish"]  # demasiado corto para decidirlo


def test_verify_summarizer_checks_words_language_and_numbers():
    source = "epoch,loss\n200,0.4321\n"
    good = "El experimento llegó a la época 200 con una pérdida de 0,43 en el conjunto de datos."
    result = V.verify("summarizer", good, source=source)
    assert result.valid and result.details["words"] == 17 and result.details["numbers"]["ratio"] == 1.0
    long_text = good + " " + " ".join(["de la y"] * 80)
    assert any("palabras" in e for e in V.verify("summarizer", long_text, source=source).errors)
    assert any("castellano" in e for e in V.verify("summarizer", "The loss was 0.43 in the last epoch of the run for the data.", source=source).errors)
    assert any("cifras" in e for e in V.verify("summarizer", good.replace("0,43", "0,9"), source=source).errors)
    assert any("vacío" in e for e in V.verify("summarizer", "  ", source=source).errors)
    assert V.verify("summarizer", good, source=source, max_words=10).valid is False


# -- recuperación -------------------------------------------------------------------------------------------------------
def test_retrieval_rank_and_validation():
    hits = [
        {"ruta": "context/constitution/roadmap.md", "linea": 3, "puntuacion": 0.9, "texto": "otra cosa"},
        {"ruta": "context/constitution/tech_stack.md", "linea": 5, "puntuacion": 0.8, "texto": "usa AMP_GAN_BACKEND"},
        {"ruta": "src/amp_gan/runtime.py", "linea": 1, "puntuacion": 0.7, "texto": "amp_gan_backend"},
    ]
    assert V.retrieval_rank(hits, "tech_stack.md", "amp_gan_backend") == 2
    assert V.retrieval_rank(hits, ["roadmap.md", "runtime.py"], "AMP_GAN_BACKEND") == 3
    assert V.retrieval_rank(hits, "roadmap.md", "AMP_GAN_BACKEND") is None
    assert V.retrieval_rank(hits, "runtime.py", "backend", k=2) is None
    assert V.validate_retrieval(hits) == []
    assert any("ordenados" in e for e in V.validate_retrieval(list(reversed(hits))))
    assert V.validate_retrieval({}) and V.validate_retrieval([{"ruta": "", "linea": 0, "puntuacion": "x", "texto": 1}])
    assert V.verify("retriever", json.dumps(hits)).valid and not V.verify("retriever", "no json").valid


def test_registry_and_unknown_verifier():
    assert set(V.VERIFIERS) == {"locator", "doc_checker", "code_reviewer", "summarizer", "retriever"}
    with pytest.raises(ValueError, match="desconocido"):
        V.verify("nada", "x")

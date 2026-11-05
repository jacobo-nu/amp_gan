"""Tests de la localización determinista (``amp_agents.locate``) y de la retirada del localizador con modelo (sin red).

Ojo: este fichero no menciona los nombres de las preguntas de la evaluación del localizador (se copia al material
que consulta esa evaluación y alteraría su verdad)."""

import json
import os

import pytest
from agents_helpers import make_project, write

from amp_agents import eval_roles as E
from amp_agents import locate as L
from amp_agents import verifiers as V
from amp_agents.agents import get_agent_config, make_policy
from amp_agents.tools import Policy


def mini(tmp_path):
    write(tmp_path / "src" / "pkg" / "base.py", "LIMITE = 0.25\n\n\ndef calcula(x):\n    return x * LIMITE\n\n\nclass Motor:\n    pass\n")
    write(tmp_path / "src" / "pkg" / "uso.py", "from pkg.base import calcula\nimport pkg.base\n\n\ndef calcula_otro():\n    return calcula(1) + LIMITE\n")
    write(tmp_path / "src" / "tests" / "test_uso.py", "def test_a():\n    from pkg.base import LIMITE\n    assert LIMITE\n")
    write(tmp_path / "src" / "tests" / "__pycache__" / "x.py", "def calcula(): pass\n")
    write(tmp_path / "src" / ".oculta" / "y.py", "def calcula(): pass\n")
    write(tmp_path / "context" / "nota.py", "def calcula(): pass\n")  # fuera de src: no se recorre por defecto
    return tmp_path


# -- definiciones --------------------------------------------------------------------------------------------------
def test_locate_definition_by_ast_returns_path_line_and_literal_text(tmp_path):
    root = mini(tmp_path)
    assert L.locate(root, "calcula", kind="definition") == [{"ruta": "src/pkg/base.py", "linea": 4, "texto": "def calcula(x):"}]
    assert L.locate(root, "Motor") == [{"ruta": "src/pkg/base.py", "linea": 8, "texto": "class Motor:"}]  # «definition» es el tipo por defecto
    assert L.locate(root, "test_a") == [{"ruta": "src/tests/test_uso.py", "linea": 1, "texto": "def test_a():"}]
    assert L.locate(root, "no_existe") == []
    assert L.locate(root, "calcula_otro")[0]["linea"] == 5  # el nombre exacto, no el prefijo


def test_locate_definition_ignores_calls_comments_and_syntax_errors(tmp_path):
    write(tmp_path / "src" / "a.py", "# def nada():\nx = 'def nada(): pass'\nnada()\n")
    write(tmp_path / "src" / "roto.py", "def nada(:\n")
    assert L.locate(tmp_path, "nada", kind="definition") == []


# -- importaciones, constantes y patrones ----------------------------------------------------------------------------
def test_locate_imports_matches_from_and_import_lines_only(tmp_path):
    root = mini(tmp_path)
    found = L.locate(root, "pkg.base", kind="import")
    assert [(r["ruta"], r["linea"]) for r in found] == [("src/pkg/uso.py", 1), ("src/pkg/uso.py", 2), ("src/tests/test_uso.py", 2)]
    assert found[0]["texto"] == "from pkg.base import calcula"
    assert L.locate(root, "pkg.bas", kind="import") == []  # no es el módulo entero
    assert L.locate(root, "pkgXbase", kind="import") == []  # el punto no es un comodín


def test_locate_constant_matches_whole_identifiers(tmp_path):
    root = mini(tmp_path)
    write(root / "src" / "pkg" / "otro.py", "LIMITE_MAX = 3\nMI_LIMITE = 4\n")
    found = L.locate(root, "LIMITE", kind="constant")
    assert [(r["ruta"], r["linea"]) for r in found] == [("src/pkg/base.py", 1), ("src/pkg/base.py", 5), ("src/pkg/uso.py", 6), ("src/tests/test_uso.py", 2), ("src/tests/test_uso.py", 3)]


def test_locate_pattern_is_a_regular_expression_per_line(tmp_path):
    root = mini(tmp_path)
    assert [(r["ruta"], r["linea"]) for r in L.locate(root, r"^def calcula\w*\(", kind="pattern")] == [("src/pkg/base.py", 4), ("src/pkg/uso.py", 5)]
    with pytest.raises(ValueError, match="expresión regular"):
        L.locate(root, "(sin cerrar", kind="pattern")


def test_locate_validates_its_arguments(tmp_path):
    root = mini(tmp_path)
    with pytest.raises(ValueError, match="desconocido"):
        L.locate(root, "x", kind="magia")
    for empty in ("", "   ", None):
        with pytest.raises(ValueError, match="vacía"):
            L.locate(root, empty)
    assert set(L.KINDS) == {"definition", "import", "constant", "pattern"}


def test_results_are_sorted_and_ready_for_the_locator_checks(tmp_path):
    root = mini(tmp_path)
    found = L.locate(root, "LIMITE", kind="constant")
    assert found == sorted(found, key=lambda r: (r["ruta"], r["linea"]))
    # la salida tiene el formato del rol retirado y supera su verificador (la línea existe y el texto es la línea literal)
    answer = json.dumps({"respuestas": found})
    assert V.verify("locator", answer, root=root).valid
    assert all(c["texto_ok"] for c in V.check_locator_citations({"respuestas": found}, root))


# -- política de rutas (solo lectura) ----------------------------------------------------------------------------------
def test_locate_skips_hidden_pycache_and_non_src_folders_by_default(tmp_path):
    root = mini(tmp_path)
    rutas = {r["ruta"] for r in L.locate(root, "calcula", kind="definition")}
    assert rutas == {"src/pkg/base.py"}
    assert {r["ruta"] for r in L.locate(root, "calcula", kind="definition", roots=("src", "context"))} == {"src/pkg/base.py", "context/nota.py"}


def test_locate_obeys_the_read_policy(tmp_path):
    root = make_project(tmp_path / "p")
    write(root / "src" / "secreto.py", "def clave(): pass\n")
    write(root / "src" / "id_rsa.py", "def clave(): pass\n")  # nombre de credencial: denegado
    write(root / "src" / "permitido" / "m.py", "def clave(): pass\n")
    assert {r["ruta"] for r in L.locate(root, "clave")} == {"src/secreto.py", "src/permitido/m.py"}
    narrow = Policy(root=root, read_roots=("src/permitido",), write_roots=())
    assert [r["ruta"] for r in L.locate(root, "clave", policy=narrow)] == ["src/permitido/m.py"]
    nothing = Policy(root=root, read_roots=("context",), write_roots=())
    assert L.locate(root, "clave", policy=nothing) == []
    assert not L.default_policy(root).write_roots and not L.default_policy(root).can(root / "src" / "x.py", "write")


def test_locate_does_not_follow_links_out_of_the_project(tmp_path):
    root = make_project(tmp_path / "p")
    outside = tmp_path / "fuera"
    write(outside / "e.py", "def clave(): pass\n")
    try:
        os.symlink(outside, root / "src" / "enlace")
        os.symlink(outside / "e.py", root / "src" / "e_enlace.py")
    except (OSError, NotImplementedError):
        pytest.skip("el sistema no permite enlaces simbólicos")
    assert L.locate(root, "clave") == []


def test_locate_never_writes(tmp_path):
    root = mini(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    L.locate(root, "calcula")
    L.locate(root, "LIMITE", kind="constant")
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_the_policy_of_the_retired_role_is_the_default_policy():
    config = get_agent_config("locator")
    policy = make_policy(config, tmp_root())
    default = L.default_policy(tmp_root())
    assert tuple(policy.read_roots) == tuple(default.read_roots) and not policy.write_roots and not default.write_roots


def tmp_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[2]


# -- coincide con la verdad de la evaluación del localizador -------------------------------------------------------------------
def test_the_truth_helpers_of_the_evaluation_are_the_shared_ones():
    assert E.find_definitions is L.find_definitions and E.find_lines is L.find_lines


def test_locate_reproduces_the_truth_of_every_evaluation_question(tmp_path):
    copy = E.make_copy(tmp_path / "copia")
    for question in E.locator_questions(copy):
        expected = [tuple(e) for e in question["expected"]]
        kind = question["kind"]
        if kind == "definicion":
            name = question["id"].removeprefix("def_")
            got = L.locate(copy, name, kind="definition")
        elif kind == "importacion":
            got = L.locate(copy, "amp_gan." + question["id"].removeprefix("import_"), kind="import")
        else:
            got = L.locate(copy, question["id"].removeprefix("const_"), kind="constant")
        assert [(r["ruta"], r["linea"]) for r in got] == expected, question["id"]
        assert V.verify("locator", json.dumps({"respuestas": got}), root=copy).valid


def test_the_new_files_do_not_alter_the_evaluation_truth(tmp_path):
    copy = E.make_copy(tmp_path / "copia")
    mentioned = {path for q in E.locator_questions(copy) for path, _ in q["expected"]}
    assert "src/amp_agents/locate.py" not in mentioned
    assert not [p for p in mentioned if "test_agents_locate" in p or "test_agents_corrections" in p]


# -- el localizador con modelo está retirado de la evaluación por defecto ----------------------------------------------------------
def test_locator_is_retired_from_the_default_evaluation_but_still_evaluable():
    assert "locator" not in E.ROLES and E.RETIRED_ROLES == ("locator",)
    assert E.ALL_ROLES == ("locator", *E.ROLES) and set(E.EVALUATORS) == set(E.COMMAND_ROLES)  # los roles de `all` más las variantes (doc_checker_ss)
    assert "locator" in E.CRITERIA and "locator" in E.REPS and "locator" in E.EVALUATORS  # el código de evaluación se conserva
    assert get_agent_config("locator").model == "llama3.2:3b"  # y la sección de configuración también
    toml = (tmp_root() / "src" / "amp_agents" / "agents.toml").read_text(encoding="utf-8")
    assert "RETIRADO" in toml.split("[agents.locator]")[0]


def test_all_does_not_run_the_locator_unless_asked(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(E, "evaluate_role", lambda role, args: called.append(role))
    work = str(tmp_path / "w")
    assert E.main(["all", "--workdir", work]) == 0
    assert called == ["doc_checker", "code_reviewer", "summarizer", "retriever"]
    called.clear()
    assert E.main(["all", "--workdir", work, "--roles", "locator"]) == 0 and called == ["locator"]
    called.clear()
    assert E.main(["all", "--workdir", work, "--roles", "locator,summarizer"]) == 0 and called == ["locator", "summarizer"]
    called.clear()
    assert E.main(["locator", "--workdir", work]) == 0 and called == ["locator"]  # también se puede pedir directamente
    with pytest.raises(SystemExit):
        E.main(["all", "--workdir", work, "--roles", "inventado"])
    assert called == ["locator"]


def test_the_cli_help_still_offers_the_locator():
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-m", "amp_agents.eval_roles", "--help"], cwd=tmp_root() / "src", capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and "locator" in done.stdout and "retirado" in done.stdout

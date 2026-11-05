"""Tests de las herramientas y de la política de permisos de ``amp_agents`` (sin red)."""

import os
import subprocess
import sys

import pytest
from agents_helpers import make_project, write

from amp_agents import tools as T
from amp_agents.tools import ALL_TOOLS, Policy, PolicyError, ToolError, build_tools


@pytest.fixture
def project(tmp_path):
    return make_project(tmp_path / "proj")


@pytest.fixture
def policy(project):
    return Policy(root=project)


def run(policy, name, **args):
    return ALL_TOOLS[name].run(policy, args)


# -- política ---------------------------------------------------------------------------------------
def test_policy_allows_context_and_src_for_reading(policy):
    assert policy.check("src/amp_gan/mod.py").name == "mod.py"
    assert policy.check("context/agents/executor.md").name == "executor.md"
    assert policy.check("context/data/raw/raw.fasta").name == "raw.fasta"  # la lectura de raw está permitida


@pytest.mark.parametrize("path", ["README.md", "../fuera.txt", "src/../README.md", "/etc/passwd", "~/.ssh/id_rsa", ""])
def test_policy_rejects_outside_read_roots_and_escapes(policy, path):
    with pytest.raises(ToolError):
        policy.check(path)


@pytest.mark.parametrize("path", [".git/config", ".claude/settings.json", "entregas/nota.txt", ".env", "src/id_rsa", "src/cert.pem"])
@pytest.mark.parametrize("mode", ["read", "write"])
def test_policy_denies_protected_paths_in_read_and_write(policy, path, mode):
    with pytest.raises(PolicyError):
        policy.check(path, mode)


def test_policy_denies_protected_component_at_any_depth(policy, project):
    write(project / "src" / ".git" / "HEAD", "x")
    write(project / "src" / ".ssh" / "known_hosts", "x")
    for path in ("src/.git/HEAD", "src/.ssh/known_hosts"):
        with pytest.raises(PolicyError):
            policy.check(path)


def test_policy_raw_is_write_protected_but_src_is_writable(policy):
    with pytest.raises(PolicyError):
        policy.check("context/data/raw/nuevo.csv", "write")
    with pytest.raises(PolicyError):
        policy.check("context/agents/executor.md", "write")
    assert policy.check("src/amp_gan/nuevo.py", "write").name == "nuevo.py"


def test_policy_processed_only_when_enabled(project):
    with pytest.raises(PolicyError):
        Policy(root=project).check("context/data/processed/out.csv", "write")
    assert Policy(root=project, allow_write_processed=True).check("context/data/processed/out.csv", "write")
    with pytest.raises(PolicyError):  # raw sigue protegido aunque se habilite processed
        Policy(root=project, allow_write_processed=True).check("context/data/raw/x.csv", "write")


def test_policy_symlink_to_outside_is_denied(policy, project, tmp_path):
    outside = write(tmp_path / "fuera" / "secreto.txt", "secreto")
    (project / "src" / "enlace.txt").symlink_to(outside)
    (project / "src" / "enlace_dir").symlink_to(outside.parent, target_is_directory=True)
    for path in ("src/enlace.txt", "src/enlace_dir/secreto.txt"):
        with pytest.raises(PolicyError):
            policy.check(path)
        with pytest.raises(PolicyError):
            policy.check(path, "write")


def test_policy_symlink_to_protected_inside_project_is_denied(policy, project):
    (project / "src" / "atajo").symlink_to(project / ".git", target_is_directory=True)
    (project / "src" / "inocente.txt").symlink_to(project / ".env")
    for path in ("src/atajo/config", "src/inocente.txt"):
        with pytest.raises(PolicyError):
            policy.check(path)
    assert run(policy, "read_file", path="src/inocente.txt").denied


def test_policy_name_check_uses_link_name_too(policy, project):
    write(project / "src" / "datos.txt", "ok")
    (project / "src" / "id_enlace").symlink_to(project / "src" / "datos.txt")
    with pytest.raises(PolicyError):
        policy.check("src/id_enlace")


def test_policy_list_mode_allows_root_but_not_other_files(policy):
    assert policy.check(".", "list") == policy.root
    with pytest.raises(PolicyError):
        policy.check(".", "read")


# -- list_dir / read_file / grep ---------------------------------------------------------------------
def test_list_dir_root_shows_only_accessible_entries(policy):
    out = run(policy, "list_dir", path=".").content
    assert "context/" in out and "src/" in out
    for hidden in (".git", ".claude", "entregas", ".env", "README.md"):
        assert hidden not in out


def test_list_dir_hides_secrets_and_pycache(policy, project):
    write(project / "src" / "__pycache__" / "x.pyc", "x")
    out = run(policy, "list_dir", path="src").content
    assert "id_rsa" not in out and "cert.pem" not in out and "__pycache__" not in out
    assert "src/amp_gan/" in out


def test_list_dir_denied_and_not_a_directory(policy):
    assert run(policy, "list_dir", path=".git").denied
    res = run(policy, "list_dir", path="src/amp_gan/mod.py")
    assert not res.ok and not res.denied


def test_read_file_numbers_lines_and_offset(policy):
    out = run(policy, "read_file", path="src/amp_gan/mod.py").content
    assert out.startswith("1\tdef f(x):") and "[fin del fichero]" in out
    part = run(policy, "read_file", path="src/amp_gan/mod.py", offset=4, limit_lines=2).content
    assert part.startswith("4\t\n5\tdef g(x):")
    assert "continúa con offset=6" in part
    beyond = run(policy, "read_file", path="src/amp_gan/mod.py", offset=99).content
    assert "termina antes" in beyond


def test_read_file_byte_limit_truncates_with_hint(policy, project):
    write(project / "src" / "grande.txt", "".join(f"linea {i} " + "x" * 40 + "\n" for i in range(500)))
    out = run(policy, "read_file", path="src/grande.txt", max_bytes=1000).content
    assert len(out) < 1300 and "continúa con offset=" in out
    write(project / "src" / "larga.txt", "y" * 5000 + "\n")
    out = run(policy, "read_file", path="src/larga.txt", max_bytes=300).content
    assert "línea recortada" in out and len(out) < 600


def test_read_file_rejects_binary_denied_and_missing(policy, project):
    (project / "src" / "datos.bin").write_bytes(b"\x00\x01\x02")
    assert not run(policy, "read_file", path="src/datos.bin").ok
    assert run(policy, "read_file", path=".git/config").denied
    missing = run(policy, "read_file", path="src/no_existe.py")
    assert not missing.ok and not missing.denied


def test_read_file_argument_validation(policy):
    assert "faltan argumentos" in run(policy, "read_file").content
    assert "desconocidos" in run(policy, "read_file", path="src/amp_gan/mod.py", foo=1).content
    assert run(policy, "read_file", path="src/amp_gan/mod.py", limit_lines="2").ok  # coerción segura
    assert not run(policy, "read_file", path="src/amp_gan/mod.py", limit_lines="dos").ok
    assert not run(policy, "read_file", path=3).ok


def test_grep_finds_lines_and_skips_protected_files(policy, project):
    write(project / "src" / "secretos.txt", "SECRETO=1\n")
    write(project / "src" / "id_rsa_extra", "SECRETO=1")
    out = run(policy, "grep", pattern=r"def \w", path="src", glob="*.py").content
    assert "src/amp_gan/mod.py:1:def f(x):" in out and "mod.py:5:def g(x):" in out
    leaked = run(policy, "grep", pattern="SECRETO", path=".").content
    assert "src/secretos.txt:1" in leaked
    assert ".env" not in leaked and "id_rsa" not in leaked


def test_grep_does_not_follow_symlinks_to_outside(policy, project, tmp_path):
    outside = write(tmp_path / "fuera" / "x.txt", "NEEDLE")
    (project / "src" / "dir_enlace").symlink_to(outside.parent, target_is_directory=True)
    (project / "src" / "f_enlace.txt").symlink_to(outside)
    assert "sin coincidencias" in run(policy, "grep", pattern="NEEDLE", path="src").content
    assert run(policy, "grep", pattern="NEEDLE", path="src/dir_enlace").denied


def test_grep_invalid_regex_denied_path_limit_and_ignore_case(policy, project):
    assert "no válida" in run(policy, "grep", pattern="(").content
    assert run(policy, "grep", pattern="x", path=".git").denied
    write(project / "src" / "muchas.txt", "Hola\n" * 30)
    out = run(policy, "grep", pattern="hola", path="src/muchas.txt", ignore_case=True, max_results=5).content
    assert out.count("muchas.txt:") == 5 and "truncado" in out


# -- write_file / edit_file ----------------------------------------------------------------------------
def test_write_file_creates_overwrites_and_makes_parents(policy, project):
    res = run(policy, "write_file", path="src/amp_gan/sub/nuevo.py", content="x = 1\n")
    assert res.ok and "creado" in res.content
    assert (project / "src/amp_gan/sub/nuevo.py").read_text() == "x = 1\n"
    assert "sobrescrito" in run(policy, "write_file", path="src/amp_gan/sub/nuevo.py", content="x = 2\n").content
    assert (project / "src/amp_gan/sub/nuevo.py").read_text() == "x = 2\n"


@pytest.mark.parametrize("path", [".git/hooks/pre-commit", "context/agents/executor.md", "context/data/raw/x.csv", "../fuera.txt", "src/../.env", "src/id_rsa", "README.md"])
def test_write_file_denied_leaves_no_trace(policy, project, path):
    before = sorted(p.relative_to(project).as_posix() for p in project.parent.rglob("*"))
    res = run(policy, "write_file", path=path, content="pwned")
    assert res.denied and not res.ok
    assert sorted(p.relative_to(project).as_posix() for p in project.parent.rglob("*")) == before
    assert (project / "context/agents/executor.md").read_text().startswith("# Agente")


def test_write_file_size_limit_and_directory(project):
    small = Policy(root=project, max_file_bytes=100)
    assert "supera el máximo" in run(small, "write_file", path="src/a.txt", content="x" * 101).content
    assert not (project / "src" / "a.txt").exists()
    assert run(small, "write_file", path="src/a.txt", content="x" * 100).ok
    assert "es una carpeta" in run(small, "write_file", path="src/amp_gan", content="x").content


def test_write_file_follows_symlink_check_on_write(policy, project, tmp_path):
    outside = write(tmp_path / "fuera" / "x.txt", "original")
    (project / "src" / "enlace.txt").symlink_to(outside)
    assert run(policy, "write_file", path="src/enlace.txt", content="pwned").denied
    assert outside.read_text() == "original"


def test_edit_file_replaces_unique_string(policy, project):
    res = run(policy, "edit_file", path="src/amp_gan/mod.py", old_string="return x + 1", new_string="return x + 10")
    assert res.ok
    assert "return x + 10" in (project / "src/amp_gan/mod.py").read_text()


def test_edit_file_missing_duplicate_empty_and_equal(policy, project):
    path = project / "src/amp_gan/mod.py"
    original = path.read_text()
    missing = run(policy, "edit_file", path="src/amp_gan/mod.py", old_string="no está", new_string="y")
    assert not missing.ok and "no aparece" in missing.content
    dup = run(policy, "edit_file", path="src/amp_gan/mod.py", old_string="return", new_string="yield")
    assert not dup.ok and "2 veces" in dup.content
    assert not run(policy, "edit_file", path="src/amp_gan/mod.py", old_string="", new_string="y").ok
    assert not run(policy, "edit_file", path="src/amp_gan/mod.py", old_string="def f", new_string="def f").ok
    assert path.read_text() == original


def test_edit_file_denied_and_nonexistent(policy):
    assert run(policy, "edit_file", path="context/agents/executor.md", old_string="Agente", new_string="X").denied
    assert run(policy, "edit_file", path=".env", old_string="1", new_string="2").denied
    res = run(policy, "edit_file", path="src/nada.py", old_string="a", new_string="b")
    assert not res.ok and not res.denied


def test_edit_file_size_limit(project):
    path = project / "src/amp_gan/mod.py"
    original = path.read_text()
    small = Policy(root=project, max_file_bytes=len(original.encode()) + 10)
    res = run(small, "edit_file", path="src/amp_gan/mod.py", old_string="x * 2", new_string="x" * 100)
    assert not res.ok and "superaría" in res.content
    assert path.read_text() == original
    tiny = Policy(root=project, max_file_bytes=10)
    assert "tamaño máximo editable" in run(tiny, "edit_file", path="src/amp_gan/mod.py", old_string="x * 2", new_string="y").content


# -- run_pytest ---------------------------------------------------------------------------------------
def test_pytest_command_is_fixed():
    assert T.pytest_command() == [sys.executable, "-m", "pytest", "-q"]
    assert T.pytest_command("tests/test_x.py") == [sys.executable, "-m", "pytest", "tests/test_x.py", "-q"]


def test_run_pytest_runs_fixed_command_from_src(policy, project):
    res = run(policy, "run_pytest", path="tests/test_mod.py")
    assert res.ok and "código de salida: 0" in res.content and "1 passed" in res.content
    assert "$ python -m pytest tests/test_mod.py -q  (en src/)" in res.content


def test_run_pytest_accepts_src_prefix_and_reports_failures(policy, project):
    write(project / "src/tests/test_fail.py", "def test_fail():\n    assert 1 == 2\n")
    res = run(policy, "run_pytest", path="src/tests/test_fail.py")
    assert res.ok and "código de salida: 1" in res.content and "1 failed" in res.content


def test_run_pytest_passes_exact_arguments(policy, project, monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"], seen["kwargs"] = command, kwargs
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    run(policy, "run_pytest", path="tests/test_mod.py")
    assert seen["command"] == [sys.executable, "-m", "pytest", "tests/test_mod.py", "-q"]
    assert seen["kwargs"]["cwd"] == project.resolve() / "src"
    assert seen["kwargs"]["timeout"] == policy.pytest_timeout
    assert "shell" not in seen["kwargs"] and seen["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("path", ["-x", "--co", "../README.md", "../context/agents/executor.md", "/etc", "src/../.git"])
def test_run_pytest_rejects_options_and_outside_paths(policy, path):
    res = run(policy, "run_pytest", path=path)
    assert not res.ok and "$ python" not in res.content


def test_run_pytest_missing_path_and_timeout(project):
    assert not run(Policy(root=project), "run_pytest", path="tests/no_existe.py").ok
    write(project / "src/tests/test_slow.py", "import time\n\n\ndef test_slow():\n    time.sleep(30)\n")
    res = run(Policy(root=project, pytest_timeout=1.0), "run_pytest", path="tests/test_slow.py")
    assert not res.ok and "tiempo máximo" in res.content


def test_run_pytest_has_no_other_parameters():
    assert set(ALL_TOOLS["run_pytest"].parameters["properties"]) == {"path"}
    assert os.path.basename(sys.executable)  # el intérprete es el del proceso, no un comando libre


def test_build_tools_and_definitions():
    names = [t.name for t in build_tools()]
    assert names == ["list_dir", "read_file", "grep", "write_file", "edit_file", "run_pytest"]
    defs = [t.definition() for t in build_tools(["read_file"])]
    assert defs[0]["type"] == "function" and defs[0]["function"]["parameters"]["required"] == ["path"]
    with pytest.raises(ValueError):
        build_tools(["bash"])

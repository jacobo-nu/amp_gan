"""Utilidades compartidas por los tests de ``amp_agents`` (no es un módulo de tests)."""

from __future__ import annotations

from pathlib import Path

EXECUTOR_MD = "# Agente: Ejecutor\n\n## Responsabilidades\n- Implementar la tarea recibida en `src/`.\n"
GOVERNANCE_MD = "# Gobernanza\n\n## Reglas\n1. El ejecutor solo recibe tareas acotadas.\n"
CONFTEST = (
    "import sys\nfrom pathlib import Path\n"
    "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
)


def write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_project(root: Path) -> Path:
    """Crea un proyecto mínimo (con carpetas protegidas y credenciales señuelo) y devuelve ``root``."""
    write(root / "context" / "agents" / "executor.md", EXECUTOR_MD)
    write(root / "context" / "constitution" / "governance.md", GOVERNANCE_MD)
    write(root / "context" / "data" / "raw" / "raw.fasta", ">a\nACD\n")
    write(root / "context" / "data" / "processed" / "p.csv", "a,b\n1,2\n")
    write(root / "src" / "amp_gan" / "__init__.py")
    write(root / "src" / "amp_gan" / "mod.py", "def f(x):\n    return x + 1\n\n\ndef g(x):\n    return x * 2\n")
    write(root / "src" / "tests" / "conftest.py", CONFTEST)
    write(root / "src" / "tests" / "test_mod.py", "from amp_gan.mod import f\n\n\ndef test_f():\n    assert f(1) == 2\n")
    write(root / ".git" / "config", "[core]\n")
    write(root / ".claude" / "settings.json", "{}")
    write(root / "entregas" / "nota.txt", "privado")
    write(root / ".env", "SECRETO=1")
    write(root / "src" / "id_rsa", "clave")
    write(root / "src" / "cert.pem", "clave")
    write(root / "README.md", "readme")
    return root

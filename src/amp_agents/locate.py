"""Localización determinista de definiciones, importaciones y apariciones en el código (sin modelo).

Sustituye al rol ``locator`` con modelo (``llama3.2:3b``, 10 % de aciertos en la evaluación en vivo): lo que ese rol hacía
es ``grep`` con otro formato, y por código es exacto. Este módulo es también la fuente de la VERDAD de la evaluación del
localizador (``eval_roles`` reutiliza :func:`find_definitions` y :func:`find_lines`).

Política de rutas (solo lectura): se recorren los ``.py`` bajo ``src/`` (carpetas ocultas y ``__pycache__`` aparte) y,
además, cada fichero tiene que estar permitido para lectura por la :class:`~amp_agents.tools.Policy` (por defecto la del
localizador: lectura en ``src`` y ``context``, y se deniegan ``.git``, credenciales, etc.). No escribe nada.

Uso::

    from amp_agents.locate import locate
    locate(root, "mi_funcion", kind="definition")     # def/class con ese nombre (ast)
    locate(root, "paquete.modulo", kind="import")     # from X import ... / import X
    locate(root, "MI_CONSTANTE", kind="constant")     # el identificador como palabra completa
    locate(root, r"def \\w+_aux", kind="pattern")     # expresión regular sobre cada línea

Cada resultado es ``{"ruta", "linea", "texto"}``, como la salida del localizador, ordenado por ruta y línea.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from .tools import Policy

KINDS = ("definition", "import", "constant", "pattern")
DEFAULT_ROOTS = ("src",)


def default_policy(root: str | Path) -> Policy:
    """Política de solo lectura del localizador sobre ``root``: lectura en ``src`` y ``context``, sin escritura."""
    return Policy(root=Path(root), read_roots=("src", "context"), write_roots=())


def py_files(root: str | Path, roots: tuple[str, ...] = DEFAULT_ROOTS, policy: Policy | None = None) -> list[Path]:
    """Ficheros ``.py`` bajo las carpetas ``roots`` de ``root`` que la política permite leer, ordenados."""
    base = Path(root).resolve()
    allowed = policy or default_policy(base)
    found = []
    for folder in roots:
        top = base / folder
        if not top.is_dir():
            continue
        for path in top.rglob("*.py"):
            parts = path.relative_to(base).parts
            if "__pycache__" in parts or any(part.startswith(".") for part in parts):
                continue
            if path.is_file() and allowed.can(path, "read"):
                found.append(path)
    return sorted(found)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def find_definitions(root: str | Path, name: str, roots: tuple[str, ...] = DEFAULT_ROOTS, policy: Policy | None = None) -> list[tuple[str, int]]:
    """Pares (ruta, línea) de las definiciones ``def``/``class`` de ``name`` en los ``.py`` de ``src/`` (con ``ast``)."""
    base = Path(root).resolve()
    out = []
    for path in py_files(base, roots, policy):
        text = _read(path)
        if text is None:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                out.append((path.relative_to(base).as_posix(), node.lineno))
    return sorted(out)


def find_lines(root: str | Path, pattern: str, roots: tuple[str, ...] = DEFAULT_ROOTS, policy: Policy | None = None) -> list[tuple[str, int]]:
    """Pares (ruta, línea) de las líneas de los ``.py`` de ``src/`` que cumplen la expresión regular ``pattern``."""
    regex = re.compile(pattern)
    base = Path(root).resolve()
    out = []
    for path in py_files(base, roots, policy):
        text = _read(path)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                out.append((path.relative_to(base).as_posix(), number))
    return sorted(out)


def import_pattern(module: str) -> str:
    """Expresión regular de las líneas ``from <module> import ...`` e ``import <module>`` (``module`` con puntos)."""
    escaped = re.escape(module)
    return rf"^\s*(?:from\s+{escaped}\s+import\b|import\s+{escaped}\b)"


def locate(
    root: str | Path,
    query: str,
    kind: str = "definition",
    policy: Policy | None = None,
    roots: tuple[str, ...] = DEFAULT_ROOTS,
) -> list[dict]:
    """Localiza ``query`` bajo ``root`` y devuelve ``[{"ruta", "linea", "texto"}]`` (``texto`` es la línea literal).

    ``kind``: ``definition`` (``def``/``class`` llamados ``query``), ``import`` (líneas que importan el módulo ``query``),
    ``constant`` (el identificador ``query`` como palabra completa) o ``pattern`` (expresión regular). ``ValueError`` si
    ``kind`` es desconocido, ``query`` está vacío o la expresión regular no es válida.
    """
    if kind not in KINDS:
        raise ValueError(f"tipo de búsqueda desconocido: {kind!r}; disponibles: {list(KINDS)}")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("la consulta está vacía")
    query = query.strip()
    if kind == "definition":
        pairs = find_definitions(root, query, roots, policy)
    else:
        if kind == "import":
            pattern = import_pattern(query)
        elif kind == "constant":
            pattern = rf"(?<![\w]){re.escape(query)}(?![\w])"
        else:
            pattern = query
        try:
            pairs = find_lines(root, pattern, roots, policy)
        except re.error as exc:
            raise ValueError(f"expresión regular no válida: {exc}") from exc
    base = Path(root).resolve()
    lines_of: dict[str, list[str]] = {}
    results = []
    for ruta, linea in pairs:
        if ruta not in lines_of:
            lines_of[ruta] = (_read(base / ruta) or "").splitlines()
        results.append({"ruta": ruta, "linea": linea, "texto": lines_of[ruta][linea - 1]})
    return results

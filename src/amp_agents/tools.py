"""Herramientas acotadas del agente y política de permisos.

Todas las herramientas pasan por :class:`Policy`, que resuelve cada ruta con ``Path.resolve()``
(así ni ``..`` ni los enlaces simbólicos permiten escapar) y SOLO después comprueba raíces
permitidas, rutas denegadas y patrones de nombre. No hay shell, ni red, ni git: la única forma de
ejecutar código es ``run_pytest``, con un comando fijo.

Herramientas: ``list_dir``, ``read_file``, ``grep``, ``write_file``, ``edit_file``, ``run_pytest``.

Política por defecto (relativa a la raíz del proyecto):

- Lectura: ``context/`` y ``src/``.
- Escritura: solo ``src/`` (y ``context/data/processed/`` únicamente con ``allow_write_processed=True``).
- Denegado en lectura y escritura: ``.git``, ``.claude``, ``entregas``, ``.ssh`` (en cualquier nivel),
  ficheros ``.env``, ``.env.*``, ``id_*``, ``*.pem``, ``*.key``.
- Denegado en escritura: ``context/data/raw``.

Limitación conocida: ``run_pytest`` ejecuta código del proyecto (también el que escriba el propio
agente) con los permisos del usuario; la política no es un aislamiento de proceso.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


class ToolError(Exception):
    """Error de una herramienta; el bucle lo devuelve al modelo como mensaje."""


class PolicyError(ToolError):
    """Acceso rechazado por la política."""


@dataclass
class Policy:
    """Permisos de un agente sobre el árbol del proyecto.

    Args:
        root: raíz del proyecto activo (puede ser una copia para pruebas).
        read_roots: carpetas legibles, relativas a ``root``.
        write_roots: carpetas escribibles, relativas a ``root``.
        deny: rutas denegadas en lectura y escritura (prefijos relativos a ``root``).
        deny_write: rutas denegadas solo en escritura.
        deny_components: nombres de carpeta denegados a cualquier profundidad.
        deny_names: patrones ``fnmatch`` de nombres de fichero denegados.
        max_file_bytes: tamaño máximo de un fichero escrito/editado y de los ficheros que recorre ``grep``.
        allow_write_processed: habilita escribir en ``context/data/processed/``.
        pytest_timeout: segundos máximos de ``run_pytest``.
        pytest_env: variables de entorno añadidas a ``run_pytest`` (por defecto sin GPU).
    """

    root: Path
    read_roots: Sequence[str] = ("context", "src")
    write_roots: Sequence[str] = ("src",)
    deny: Sequence[str] = (".git", ".claude", "entregas")
    deny_write: Sequence[str] = ("context/data/raw",)
    deny_components: Sequence[str] = (".git", ".claude", ".ssh")
    deny_names: Sequence[str] = (".env", ".env.*", "id_*", "*.pem", "*.key")
    max_file_bytes: int = 200_000
    allow_write_processed: bool = False
    pytest_timeout: float = 300.0
    pytest_env: dict = field(default_factory=lambda: {"CUDA_VISIBLE_DEVICES": ""})

    def __post_init__(self):
        self.root = Path(self.root).resolve()

    # -- utilidades ---------------------------------------------------------------------------
    def _resolved(self, rels: Iterable[str]) -> list[Path]:
        return [(self.root / r).resolve() for r in rels]

    @property
    def effective_write_roots(self) -> list[str]:
        roots = list(self.write_roots)
        if self.allow_write_processed:
            roots.append("context/data/processed")
        return roots

    def rel(self, path: Path) -> str:
        """Ruta relativa a la raíz, con ``/``."""
        try:
            return path.relative_to(self.root).as_posix() or "."
        except ValueError:
            return str(path)

    def check(self, path: str, mode: str = "read") -> Path:
        """Resuelve ``path`` y devuelve la ruta real si está permitida; si no, ``PolicyError``.

        ``mode`` es ``"read"``, ``"write"`` o ``"list"`` (como ``read`` pero permite también la raíz y
        las carpetas que contienen una raíz de lectura).
        """
        if not isinstance(path, str) or not path.strip():
            raise ToolError("la ruta está vacía")
        if "\x00" in path:
            raise PolicyError("acceso denegado: ruta no válida")
        raw = Path(path)
        resolved = (raw if raw.is_absolute() else self.root / raw).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise PolicyError(f"acceso denegado: {path!r} queda fuera del proyecto") from None

        denied = list(self.deny) + (list(self.deny_write) if mode == "write" else [])
        for d in self._resolved(denied):
            if resolved == d or d in resolved.parents:
                raise PolicyError(f"acceso denegado: {path!r} está en una ruta protegida ({self.rel(d)})")
        for part in resolved.relative_to(self.root).parts:
            if part in self.deny_components:
                raise PolicyError(f"acceso denegado: {path!r} está en una carpeta protegida ({part})")
        for name in {resolved.name, raw.name}:
            if any(fnmatch.fnmatch(name.lower(), pat.lower()) for pat in self.deny_names):
                raise PolicyError(f"acceso denegado: {path!r} es un fichero protegido (credenciales)")

        roots = self._resolved(self.effective_write_roots if mode == "write" else self.read_roots)
        if any(resolved == r or r in resolved.parents for r in roots):
            return resolved
        if mode == "list" and (resolved == self.root or any(resolved in r.parents for r in roots)):
            return resolved
        verb = "escritura" if mode == "write" else "lectura"
        allowed = ", ".join(self.effective_write_roots if mode == "write" else self.read_roots) or "(ninguna)"
        raise PolicyError(f"acceso denegado: {path!r} no está en una zona de {verb} permitida ({allowed})")

    def can(self, path: Path, mode: str = "read") -> bool:
        try:
            self.check(str(path), mode)
            return True
        except ToolError:
            return False


@dataclass
class ToolResult:
    """Resultado de una herramienta: texto para el modelo y banderas para el registro."""

    content: str
    ok: bool = True
    denied: bool = False


@dataclass
class Tool:
    """Herramienta con esquema JSON (formato ``tools`` de la API de OpenAI)."""

    name: str
    description: str
    parameters: dict
    func: Callable[[Policy, dict], str]

    def definition(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def run(self, policy: Policy, arguments: dict) -> ToolResult:
        """Valida los argumentos y ejecuta; los errores se devuelven como resultado, no como excepción."""
        try:
            clean = _validate(self.parameters, arguments)
            return ToolResult(self.func(policy, clean))
        except PolicyError as exc:
            return ToolResult(f"ERROR: {exc}", ok=False, denied=True)
        except ToolError as exc:
            return ToolResult(f"ERROR: {exc}", ok=False)
        except OSError as exc:
            return ToolResult(f"ERROR del sistema de ficheros: {exc.strerror or exc}", ok=False)


def _validate(schema: dict, arguments: dict) -> dict:
    """Comprobación mínima del esquema: claves conocidas, obligatorias y tipos (con coerción segura)."""
    props = schema.get("properties", {})
    unknown = sorted(set(arguments) - set(props))
    if unknown:
        raise ToolError(f"argumentos desconocidos: {unknown}; los válidos son {sorted(props)}")
    missing = [k for k in schema.get("required", []) if k not in arguments]
    if missing:
        raise ToolError(f"faltan argumentos obligatorios: {missing}")
    clean: dict[str, Any] = {}
    for key, value in arguments.items():
        kind = props[key].get("type")
        if kind == "integer":
            if isinstance(value, bool):
                raise ToolError(f"{key!r} debe ser un entero")
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            elif isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
                value = int(value)
            if not isinstance(value, int):
                raise ToolError(f"{key!r} debe ser un entero")
        elif kind == "boolean":
            if isinstance(value, str) and value.lower() in {"true", "false"}:
                value = value.lower() == "true"
            if not isinstance(value, bool):
                raise ToolError(f"{key!r} debe ser true o false")
        elif kind == "string" and not isinstance(value, str):
            raise ToolError(f"{key!r} debe ser una cadena de texto")
        clean[key] = value
    return clean


# -- implementación de las herramientas ------------------------------------------------------------
_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def _is_binary(path: Path) -> bool:
    with open(path, "rb") as handle:
        return b"\x00" in handle.read(2048)


def _list_dir(policy: Policy, args: dict) -> str:
    path = policy.check(args.get("path", "."), "list")
    if not path.is_dir():
        raise ToolError(f"{policy.rel(path)!r} no es una carpeta")
    limit = max(1, min(int(args.get("max_entries", 200)), 500))
    lines = []
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        if child.name in _SKIP_DIRS or not policy.can(child, "list"):
            continue
        if child.is_dir():
            lines.append(f"{policy.rel(child)}/")
        else:
            lines.append(f"{policy.rel(child)}  ({child.stat().st_size} bytes)")
    shown = lines[:limit]
    extra = f"\n[... {len(lines) - limit} entradas más]" if len(lines) > limit else ""
    return "\n".join(shown) + extra if shown else "(carpeta vacía o sin entradas accesibles)"


def _read_file(policy: Policy, args: dict) -> str:
    path = policy.check(args["path"], "read")
    if not path.is_file():
        raise ToolError(f"{policy.rel(path)!r} no es un fichero")
    if _is_binary(path):
        raise ToolError(f"{policy.rel(path)!r} parece un fichero binario")
    offset = max(1, int(args.get("offset", 1)))
    limit_lines = max(1, min(int(args.get("limit_lines", 200)), 2000))
    max_bytes = max(200, min(int(args.get("max_bytes", 20000)), 100_000))
    out, used, number, truncated_at = [], 0, 0, None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            if number < offset:
                continue
            text = f"{number}\t{line.rstrip(chr(10)).rstrip(chr(13))}"
            if len(out) >= limit_lines or (out and used + len(text) + 1 > max_bytes):
                truncated_at = number
                break
            if len(text) + 1 > max_bytes:  # una sola línea más larga que el límite
                text = text[:max_bytes] + " [línea recortada]"
            out.append(text)
            used += len(text) + 1
    if not out and truncated_at is None:
        return f"(sin líneas desde offset={offset}; el fichero termina antes)"
    tail = f"\n[truncado: continúa con offset={truncated_at}]" if truncated_at else "\n[fin del fichero]"
    return "\n".join(out) + tail


def _grep(policy: Policy, args: dict) -> str:
    pattern = args["pattern"]
    if len(pattern) > 300:
        raise ToolError("patrón demasiado largo (máximo 300 caracteres)")
    try:
        regex = re.compile(pattern, re.IGNORECASE if args.get("ignore_case") else 0)
    except re.error as exc:
        raise ToolError(f"expresión regular no válida: {exc}") from exc
    base = policy.check(args.get("path", "src"), "list")
    glob = args.get("glob", "*")
    limit = max(1, min(int(args.get("max_results", 50)), 200))
    files: list[Path] = []
    if base.is_file():
        files = [base]
    else:
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            current = Path(dirpath)
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and policy.can(current / d, "list"))
            files.extend(current / f for f in sorted(filenames) if fnmatch.fnmatch(f, glob))
    results, scanned = [], 0
    for file in files:
        if not policy.can(file, "read"):
            continue
        try:
            if file.stat().st_size > policy.max_file_bytes or _is_binary(file):
                continue
            scanned += 1
            with open(file, encoding="utf-8", errors="replace") as handle:
                for number, line in enumerate(handle, start=1):
                    if regex.search(line):
                        results.append(f"{policy.rel(file.resolve())}:{number}:{line.strip()[:200]}")
                        if len(results) >= limit:
                            return "\n".join(results) + f"\n[truncado en {limit} resultados]"
        except OSError:
            continue
    if not results:
        return f"(sin coincidencias en {scanned} ficheros)"
    return "\n".join(results)


def _write_file(policy: Policy, args: dict) -> str:
    path = policy.check(args["path"], "write")
    content = args["content"]
    size = len(content.encode("utf-8"))
    if size > policy.max_file_bytes:
        raise ToolError(f"el contenido ({size} bytes) supera el máximo permitido ({policy.max_file_bytes})")
    if path.is_dir():
        raise ToolError(f"{policy.rel(path)!r} es una carpeta")
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"{'sobrescrito' if existed else 'creado'} {policy.rel(path)} ({size} bytes, {content.count(chr(10)) + (not content.endswith(chr(10)))} líneas)"


def _edit_file(policy: Policy, args: dict) -> str:
    path = policy.check(args["path"], "write")
    policy.check(args["path"], "read")
    old, new = args["old_string"], args["new_string"]
    if not old:
        raise ToolError("old_string está vacío")
    if old == new:
        raise ToolError("old_string y new_string son iguales")
    if not path.is_file():
        raise ToolError(f"{policy.rel(path)!r} no existe o no es un fichero")
    if path.stat().st_size > policy.max_file_bytes:
        raise ToolError(f"{policy.rel(path)!r} supera el tamaño máximo editable ({policy.max_file_bytes} bytes)")
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        raise ToolError("old_string no aparece en el fichero (debe coincidir exactamente, con espacios y saltos de línea)")
    if count > 1:
        raise ToolError(f"old_string aparece {count} veces; añade contexto para que sea único")
    updated = text.replace(old, new, 1)
    if len(updated.encode("utf-8")) > policy.max_file_bytes:
        raise ToolError(f"el resultado superaría el máximo permitido ({policy.max_file_bytes} bytes)")
    path.write_text(updated, encoding="utf-8")
    return f"editado {policy.rel(path)} (1 sustitución)"


def pytest_command(target: str | None = None) -> list[str]:
    """Comando fijo de ``run_pytest``: ``python -m pytest [ruta] -q``."""
    return [sys.executable, "-m", "pytest", *([target] if target else []), "-q"]


def _run_pytest(policy: Policy, args: dict) -> str:
    src = (policy.root / "src").resolve()
    target = None
    given = (args.get("path") or "").strip()
    if given:
        if given.startswith("-"):
            raise ToolError("la ruta no puede empezar por '-' (no se admiten opciones)")
        file_part, sep, node = given.partition("::")
        candidate = src / file_part
        if not candidate.exists() and file_part.startswith("src/"):
            candidate = policy.root / file_part
        checked = policy.check(str(candidate), "read")
        if checked != src and src not in checked.parents:
            raise PolicyError(f"acceso denegado: {given!r} no está dentro de src/")
        if not checked.exists():
            raise ToolError(f"no existe {policy.rel(checked)!r}")
        target = checked.relative_to(src).as_posix() + (sep + node if sep else "")
    if not src.is_dir():
        raise ToolError("no existe la carpeta src/ del proyecto")
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "PYTHONPATH" and not k.upper().endswith(("KEY", "TOKEN", "SECRET", "PASSWORD"))
    }
    env.update(policy.pytest_env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_ADDOPTS"] = "-p no:cacheprovider"
    command = pytest_command(target)
    shown = "python -m pytest" + (f" {target}" if target else "") + " -q"
    try:
        done = subprocess.run(command, cwd=src, capture_output=True, text=True, timeout=policy.pytest_timeout, env=env, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise ToolError(f"{shown}: superó el tiempo máximo ({policy.pytest_timeout:g} s)") from None
    output = (done.stdout + done.stderr).strip()
    if len(output) > 6000:
        output = "[... salida recortada, se conserva el final ...]\n" + output[-6000:]
    return f"$ {shown}  (en src/)\ncódigo de salida: {done.returncode}\n{output}"


def _schema(properties: dict, required: Sequence[str] = ()) -> dict:
    return {"type": "object", "properties": properties, "required": list(required)}


_STR = "string"
ALL_TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "list_dir",
            "Lista el contenido de una carpeta (rutas relativas a la raíz del proyecto).",
            _schema(
                {
                    "path": {"type": _STR, "description": "Carpeta; por defecto la raíz '.'"},
                    "max_entries": {"type": "integer", "description": "Máximo de entradas (por defecto 200)"},
                }
            ),
            _list_dir,
        ),
        Tool(
            "read_file",
            "Lee un fichero de texto con números de línea. Úsalo con offset para leer por partes.",
            _schema(
                {
                    "path": {"type": _STR, "description": "Ruta del fichero"},
                    "offset": {"type": "integer", "description": "Primera línea a leer (1 = principio)"},
                    "limit_lines": {"type": "integer", "description": "Máximo de líneas (por defecto 200)"},
                    "max_bytes": {"type": "integer", "description": "Máximo de bytes devueltos (por defecto 20000)"},
                },
                ["path"],
            ),
            _read_file,
        ),
        Tool(
            "grep",
            "Busca una expresión regular (Python) en los ficheros de texto de una carpeta. Devuelve ruta:línea:texto.",
            _schema(
                {
                    "pattern": {"type": _STR, "description": "Expresión regular"},
                    "path": {"type": _STR, "description": "Carpeta o fichero; por defecto 'src'"},
                    "glob": {"type": _STR, "description": "Filtro de nombre de fichero, p. ej. '*.py'"},
                    "ignore_case": {"type": "boolean", "description": "Ignorar mayúsculas"},
                    "max_results": {"type": "integer", "description": "Máximo de resultados (por defecto 50)"},
                },
                ["pattern"],
            ),
            _grep,
        ),
        Tool(
            "write_file",
            "Crea o sobrescribe un fichero de texto con el contenido dado (solo en las zonas de escritura permitidas).",
            _schema(
                {
                    "path": {"type": _STR, "description": "Ruta del fichero"},
                    "content": {"type": _STR, "description": "Contenido completo del fichero"},
                },
                ["path", "content"],
            ),
            _write_file,
        ),
        Tool(
            "edit_file",
            "Sustituye una cadena EXACTA y única de un fichero existente por otra (falla si no aparece o aparece más de una vez).",
            _schema(
                {
                    "path": {"type": _STR, "description": "Ruta del fichero"},
                    "old_string": {"type": _STR, "description": "Texto a reemplazar, tal cual aparece en el fichero"},
                    "new_string": {"type": _STR, "description": "Texto nuevo"},
                },
                ["path", "old_string", "new_string"],
            ),
            _edit_file,
        ),
        Tool(
            "run_pytest",
            "Ejecuta `python -m pytest <ruta> -q` desde la carpeta src/ del proyecto (comando fijo). La ruta es opcional y relativa a src/, p. ej. 'tests/test_x.py'.",
            _schema({"path": {"type": _STR, "description": "Fichero o carpeta de tests relativa a src/ (opcional)"}}),
            _run_pytest,
        ),
    ]
}

READ_ONLY_TOOLS = ("list_dir", "read_file", "grep")


def build_tools(names: Iterable[str] | None = None) -> list[Tool]:
    """Devuelve las herramientas pedidas (todas si ``names`` es ``None``); ``ValueError`` si alguna no existe."""
    names = list(ALL_TOOLS) if names is None else list(names)
    unknown = [n for n in names if n not in ALL_TOOLS]
    if unknown:
        raise ValueError(f"herramientas desconocidas: {unknown}; disponibles: {sorted(ALL_TOOLS)}")
    return [ALL_TOOLS[n] for n in names]

"""Verificadores automáticos de lo que devuelven los roles de solo lectura o de una pasada.

Son funciones sin red y sin modelo: reciben el texto del modelo (y, si hace falta, la raíz del proyecto o el
texto de la entrada) y devuelven qué está bien y qué no. La idea es no fiarse nunca de lo que el modelo
afirma: sus citas, líneas y cifras se comprueban contra los ficheros.

Criterios fijados (no se ajustan después de ver resultados):

- JSON: se acepta el objeto aunque venga dentro de un bloque de código o con texto alrededor (se extrae el
  primer objeto), pero ``parse_json_output`` informa de si la salida era JSON estricto.
- Cita literal válida: el fragmento (sin espacios en los extremos) aparece tal cual en el fichero indicado;
  si no, se acepta que coincida tras colapsar los espacios en blanco.
- Línea citada válida: es un entero entre 1 y el número de líneas del fichero.
- Cifras de un resumen: cada número del resumen debe aparecer en la entrada, con redondeo al número de
  decimales escrito (diferencia <= media unidad del último decimal), con coma o punto decimal, y con
  porcentaje obvio (``37,5 %`` frente a ``0.375``); se ignora el signo. Los marcadores de lista («1.») no cuentan.
- Palabras: se cuentan los trozos separados por espacios que contienen alguna letra o cifra.
- Castellano: proporción de palabras funcionales castellanas frente a inglesas (listas más abajo) >= 0,8 y
  al menos 5 palabras funcionales castellanas.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

SEVERITIES = ("alta", "media", "baja")
SPANISH_MIN_RATIO = 0.8
SPANISH_MIN_HITS = 5


@dataclass
class Verification:
    """Resultado de verificar una salida: ``valid`` y la lista de problemas encontrados."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


# -- JSON ---------------------------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def parse_json_output(text: str) -> tuple[Any, bool, str | None]:
    """Extrae el objeto JSON de la salida de un modelo: ``(objeto, estricto, error)``.

    ``estricto`` es verdadero si todo el texto es un único objeto JSON. Si no lo es, se intenta extraer el
    primer objeto de un bloque de código o del texto; si no hay ninguno, ``objeto`` es ``None`` y ``error``
    explica el motivo.
    """
    raw = (text or "").strip()
    if not raw:
        return None, False, "la salida está vacía"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    else:
        if isinstance(value, dict):
            return value, True, None
        return None, False, f"la salida es JSON pero no un objeto ({type(value).__name__})"
    candidates = [m.group(1) for m in _FENCE_RE.finditer(raw)] + [raw]
    decoder = json.JSONDecoder()
    last_error = "no se encontró ningún objeto JSON"
    for candidate in candidates:
        pos = candidate.find("{")
        while pos >= 0:
            try:
                value, _ = decoder.raw_decode(candidate, pos)
            except json.JSONDecodeError as exc:
                last_error = f"JSON no válido ({exc.msg}, posición {exc.pos})"
                pos = candidate.find("{", pos + 1)
                continue
            if isinstance(value, dict):
                return value, False, None
            pos = candidate.find("{", pos + 1)
    return None, False, last_error


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _list_of_dicts(obj: Any, key: str) -> tuple[list, list[str]]:
    if not isinstance(obj, dict):
        return [], ["la salida no es un objeto JSON"]
    if key not in obj:
        return [], [f"falta la clave {key!r}"]
    extra = sorted(set(obj) - {key})
    errors = [f"claves no previstas: {extra}"] if extra else []
    items = obj[key]
    if not isinstance(items, list):
        return [], errors + [f"{key!r} debe ser una lista"]
    return items, errors


def validate_locator(obj: Any) -> list[str]:
    """Errores de forma de la salida del localizador: ``{"respuestas": [{"ruta", "linea", "texto"}]}``."""
    items, errors = _list_of_dicts(obj, "respuestas")
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"respuestas[{i}] no es un objeto")
            continue
        if not isinstance(item.get("ruta"), str) or not item.get("ruta"):
            errors.append(f"respuestas[{i}].ruta debe ser una cadena no vacía")
        if not _is_int(item.get("linea")):
            errors.append(f"respuestas[{i}].linea debe ser un entero")
        if not isinstance(item.get("texto"), str):
            errors.append(f"respuestas[{i}].texto debe ser una cadena")
    return errors


def validate_doc_checker(obj: Any) -> list[str]:
    """Errores de forma de la salida del verificador documental (``incoherencias`` con dos evidencias)."""
    items, errors = _list_of_dicts(obj, "incoherencias")
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"incoherencias[{i}] no es un objeto")
            continue
        if not isinstance(item.get("descripcion"), str) or not item.get("descripcion"):
            errors.append(f"incoherencias[{i}].descripcion debe ser una cadena no vacía")
        for side in ("evidencia_a", "evidencia_b"):
            ev = item.get(side)
            if not isinstance(ev, dict):
                errors.append(f"incoherencias[{i}].{side} debe ser un objeto con ruta y cita")
                continue
            for key in ("ruta", "cita"):
                if not isinstance(ev.get(key), str) or not ev.get(key):
                    errors.append(f"incoherencias[{i}].{side}.{key} debe ser una cadena no vacía")
    return errors


def validate_code_reviewer(obj: Any) -> list[str]:
    """Errores de forma de la salida del revisor: ``hallazgos`` con ``linea``, ``cita``, ``problema``, ``gravedad``."""
    items, errors = _list_of_dicts(obj, "hallazgos")
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"hallazgos[{i}] no es un objeto")
            continue
        if not _is_int(item.get("linea")):
            errors.append(f"hallazgos[{i}].linea debe ser un entero")
        for key in ("cita", "problema"):
            if not isinstance(item.get(key), str) or not item.get(key):
                errors.append(f"hallazgos[{i}].{key} debe ser una cadena no vacía")
        if item.get("gravedad") not in SEVERITIES:
            errors.append(f"hallazgos[{i}].gravedad debe ser una de {list(SEVERITIES)}")
    return errors


# -- ficheros, citas y líneas ---------------------------------------------------------------------------
def read_project_text(root: str | Path, ruta: str) -> str | None:
    """Texto del fichero ``ruta`` (relativa a ``root``); ``None`` si no existe, no es fichero o sale de ``root``."""
    base = Path(root).resolve()
    if not isinstance(ruta, str) or not ruta.strip() or "\x00" in ruta:
        return None
    cleaned = ruta.strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    target = (base / cleaned).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        return None
    if not target.is_file():
        return None
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def find_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Líneas ``(primera, última)`` (desde 1) donde aparece la cita literal en ``text``; ``None`` si no aparece.

    Se busca el fragmento tal cual (sin espacios en los extremos); si no está, se compara con los espacios en
    blanco colapsados.
    """
    wanted = (quote or "").strip()
    if not wanted:
        return None
    index = text.find(wanted)
    if index >= 0:
        first = text.count("\n", 0, index) + 1
        return first, first + wanted.count("\n")
    collapsed: list[str] = []
    line_of: list[int] = []
    line = 1
    previous_space = True
    for char in text:
        if char.isspace():
            if not previous_space:
                collapsed.append(" ")
                line_of.append(line)
                previous_space = True
        else:
            collapsed.append(char)
            line_of.append(line)
            previous_space = False
        if char == "\n":
            line += 1
    norm_text = "".join(collapsed)
    norm_quote = " ".join(wanted.split())
    index = norm_text.find(norm_quote)
    if index < 0:
        return None
    return line_of[index], line_of[index + len(norm_quote) - 1]


def count_lines(text: str) -> int:
    return len(text.splitlines())


def line_near(found: int, expected: int, tolerance: int = 2) -> bool:
    """``True`` si la línea ``found`` está a ``tolerance`` o menos de la ``expected``."""
    return abs(found - expected) <= tolerance


def check_locator_citations(obj: Any, root: str | Path) -> list[dict]:
    """Por cada respuesta del localizador: ¿existe el fichero y la línea? ¿contiene la línea el texto citado?"""
    out = []
    items = obj.get("respuestas", []) if isinstance(obj, dict) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        ruta, linea, texto = item.get("ruta"), item.get("linea"), item.get("texto")
        text = read_project_text(root, ruta) if isinstance(ruta, str) else None
        exists = text is not None
        line_ok = bool(exists and _is_int(linea) and 1 <= linea <= count_lines(text))
        text_ok = False
        if line_ok and isinstance(texto, str) and texto.strip():
            text_ok = " ".join(texto.split()) in " ".join(text.splitlines()[linea - 1].split())
        out.append({"ruta": ruta, "linea": linea, "file_exists": exists, "line_exists": line_ok, "texto_ok": text_ok})
    return out


def check_doc_citations(obj: Any, root: str | Path) -> list[dict]:
    """Por cada cita de cada incoherencia: ¿es un fragmento literal del fichero indicado? y ¿en qué líneas?"""
    out = []
    items = obj.get("incoherencias", []) if isinstance(obj, dict) else []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        for side in ("evidencia_a", "evidencia_b"):
            ev = item.get(side)
            if not isinstance(ev, dict):
                continue
            ruta, cita = ev.get("ruta"), ev.get("cita")
            text = read_project_text(root, ruta) if isinstance(ruta, str) else None
            where = find_quote(text, cita) if text is not None and isinstance(cita, str) else None
            out.append(
                {
                    "item": index,
                    "side": side,
                    "ruta": ruta,
                    "cita": cita,
                    "valid": where is not None,
                    "first_line": None if where is None else where[0],
                    "last_line": None if where is None else where[1],
                }
            )
    return out


def check_review_citations(obj: Any, source: str) -> list[dict]:
    """Por cada hallazgo del revisor: ¿existe la línea y es la cita un fragmento literal del fichero revisado?

    ``valid`` exige las dos cosas. ``on_line`` (informativo) indica además que la cita está en la línea dada.
    """
    out = []
    n_lines = count_lines(source)
    items = obj.get("hallazgos", []) if isinstance(obj, dict) else []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        linea, cita = item.get("linea"), item.get("cita")
        line_exists = _is_int(linea) and 1 <= linea <= n_lines
        where = find_quote(source, cita) if isinstance(cita, str) else None
        out.append(
            {
                "item": index,
                "linea": linea,
                "cita": cita,
                "line_exists": bool(line_exists),
                "quote_found": where is not None,
                "valid": bool(line_exists and where is not None),
                "on_line": bool(line_exists and where is not None and where[0] <= linea <= where[1]),
            }
        )
    return out


# -- cifras, palabras e idioma ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[   ]\d{3})*(?:[.,]\d+)?(?:[eE][-+]?\d+)?(?P<pct>\s?%)?")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
_THOUSANDS_DOT_RE = re.compile(r"\d{1,3}(?:\.\d{3})+")


@dataclass(frozen=True)
class Number:
    """Número leído de un texto: valores posibles, decimales escritos y si lleva ``%``."""

    raw: str
    values: tuple[float, ...]
    decimals: int
    percent: bool


def extract_numbers(text: str, comma_decimal: bool = True) -> list[Number]:
    """Números de un texto (coma o punto decimal, miles con espacio, notación científica, ``%``).

    Con ``comma_decimal=False`` la coma se trata como separador (CSV: ``2000,2000,1,0.0115``) y no como decimal.
    """
    out = []
    prepared = _LIST_MARKER_RE.sub("", text or "")
    if not comma_decimal:
        prepared = prepared.replace(",", "\n")
    for match in _NUMBER_RE.finditer(prepared):
        raw = match.group(0).strip()
        percent = bool(match.group("pct"))
        body = raw.rstrip("% ").strip()
        parts = re.split(r"[eE]", body, maxsplit=1)
        mantissa = re.sub(r"[   ]", "", parts[0])
        exponent_value = int(parts[1]) if len(parts) == 2 else 0
        decimals = 0
        values = []
        decimal_text = mantissa.replace(",", ".")
        if "." in decimal_text:
            decimals = len(decimal_text.split(".")[1])
        try:
            values.append(float(decimal_text) * 10.0**exponent_value)
        except ValueError:
            continue
        if _THOUSANDS_DOT_RE.fullmatch(mantissa):  # «21.754» también puede ser 21754
            values.append(float(mantissa.replace(".", "")) * 10.0**exponent_value)
        out.append(Number(raw, tuple(values), max(0, decimals - exponent_value) if exponent_value else decimals, percent))
    return out


def number_supported(number: Number, source: Sequence[Number]) -> bool:
    """¿Aparece ``number`` en ``source`` (admitiendo redondeo al número de decimales escrito y porcentajes)?"""
    tolerance = 0.5 * 10.0 ** (-number.decimals) + 1e-9
    for candidate in number.values:
        for other in source:
            for value in other.values:
                options = [value]
                if number.percent:
                    options.append(value * 100.0)
                if any(abs(candidate - option) <= tolerance for option in options):
                    return True
    return False


def check_numbers_supported(summary: str, source: str) -> dict:
    """Cifras del resumen presentes en la entrada: ``{"total", "supported", "ratio", "missing"}``."""
    reference = extract_numbers(source) + extract_numbers(source, comma_decimal=False)  # la entrada puede ser CSV
    numbers = extract_numbers(summary)
    missing = [n.raw for n in numbers if not number_supported(n, reference)]
    total = len(numbers)
    return {"total": total, "supported": total - len(missing), "ratio": 1.0 if total == 0 else (total - len(missing)) / total, "missing": missing}


def count_words(text: str) -> int:
    """Número de palabras: trozos separados por espacios con alguna letra o cifra."""
    return sum(1 for token in (text or "").split() if any(c.isalnum() for c in token))


SPANISH_FUNCTION_WORDS = frozenset(
    "el la los las de del que en y con por para se un una al su sus es son fue como más entre sobre este esta estos "
    "estas lo le les sin muy también pero ha han hay cada donde cuando desde hasta tiene tienen no".split()
)
ENGLISH_FUNCTION_WORDS = frozenset(
    "the of and to is are was were in that for with this these those on as by from it its has have had which an be "
    "been at or not".split()
)
_WORD_RE = re.compile(r"[a-záéíóúñü]+")


def spanish_score(text: str) -> dict:
    """Palabras funcionales castellanas e inglesas y proporción ``es / (es + en)`` (0 si no hay ninguna)."""
    words = _WORD_RE.findall((text or "").lower())
    es = sum(1 for w in words if w in SPANISH_FUNCTION_WORDS)
    en = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
    ratio = es / (es + en) if es + en else 0.0
    return {"es": es, "en": en, "ratio": ratio, "is_spanish": ratio >= SPANISH_MIN_RATIO and es >= SPANISH_MIN_HITS}


# -- recuperación ---------------------------------------------------------------------------------------
def retrieval_rank(results: Sequence[dict], expected_files: str | Sequence[str], keyword: str, k: int = 3) -> int | None:
    """Posición (desde 1) del primer resultado, entre los ``k`` primeros, de un fichero esperado cuyo texto contiene
    la palabra clave (sin distinguir mayúsculas); ``None`` si no hay acierto."""
    files = [expected_files] if isinstance(expected_files, str) else list(expected_files)
    for rank, hit in enumerate(list(results)[:k], start=1):
        ruta = str(hit.get("ruta", ""))
        if any(ruta.endswith(f) for f in files) and keyword.lower() in str(hit.get("texto", "")).lower():
            return rank
    return None


def validate_retrieval(obj: Any) -> list[str]:
    """Errores de forma de la lista de resultados del recuperador (``ruta``, ``linea``, ``puntuacion``, ``texto``)."""
    if not isinstance(obj, list):
        return ["el resultado debe ser una lista"]
    errors = []
    last = None
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            errors.append(f"resultado[{i}] no es un objeto")
            continue
        if not isinstance(item.get("ruta"), str) or not item.get("ruta"):
            errors.append(f"resultado[{i}].ruta debe ser una cadena no vacía")
        if not _is_int(item.get("linea")) or item["linea"] < 1:
            errors.append(f"resultado[{i}].linea debe ser un entero >= 1")
        score = item.get("puntuacion")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            errors.append(f"resultado[{i}].puntuacion debe ser un número")
        else:
            if last is not None and score > last + 1e-12:
                errors.append("los resultados no están ordenados por puntuación decreciente")
            last = score
        if not isinstance(item.get("texto"), str):
            errors.append(f"resultado[{i}].texto debe ser una cadena")
    return errors


# -- verificadores por nombre (los del campo ``verifier`` de agents.toml) -------------------------------------
def verify_locator(text: str, root: str | Path | None = None, **_: Any) -> Verification:
    obj, strict, error = parse_json_output(text)
    if error:
        return Verification(False, [error], {"strict_json": False, "parsed": False, "schema_ok": False})
    errors = validate_locator(obj)
    details: dict = {"strict_json": strict, "parsed": True, "schema_ok": not errors}
    if root is not None and not errors:
        checks = check_locator_citations(obj, root)
        details["citations"] = checks
        errors += [f"la línea citada no existe: {c['ruta']}:{c['linea']}" for c in checks if not c["line_exists"]]
    return Verification(not errors, errors, details)


def verify_doc_checker(text: str, root: str | Path | None = None, **_: Any) -> Verification:
    obj, strict, error = parse_json_output(text)
    if error:
        return Verification(False, [error], {"strict_json": False, "parsed": False, "schema_ok": False})
    errors = validate_doc_checker(obj)
    details: dict = {"strict_json": strict, "parsed": True, "schema_ok": not errors}
    if root is not None and not errors:
        checks = check_doc_citations(obj, root)
        details["citations"] = checks
        errors += [f"cita no literal en {c['ruta']}: {str(c['cita'])[:60]!r}" for c in checks if not c["valid"]]
    return Verification(not errors, errors, details)


def verify_code_reviewer(text: str, root: str | Path | None = None, source: str | None = None, **_: Any) -> Verification:
    obj, strict, error = parse_json_output(text)
    if error:
        return Verification(False, [error], {"strict_json": False, "parsed": False, "schema_ok": False})
    errors = validate_code_reviewer(obj)
    details: dict = {"strict_json": strict, "parsed": True, "schema_ok": not errors}
    if source is not None and not errors:
        checks = check_review_citations(obj, source)
        details["citations"] = checks
        errors += [f"cita o línea no válida en el hallazgo {c['item']} (línea {c['linea']})" for c in checks if not c["valid"]]
    return Verification(not errors, errors, details)


def verify_summarizer(text: str, root: str | Path | None = None, source: str | None = None, max_words: int = 150, **_: Any) -> Verification:
    errors = []
    words = count_words(text)
    if words == 0:
        errors.append("el resumen está vacío")
    if words > max_words:
        errors.append(f"el resumen tiene {words} palabras (máximo {max_words})")
    lang = spanish_score(text)
    if not lang["is_spanish"]:
        errors.append(f"no parece castellano (proporción {lang['ratio']:.2f})")
    details: dict = {"words": words, "spanish": lang}
    if source is not None:
        numbers = check_numbers_supported(text, source)
        details["numbers"] = numbers
        if numbers["missing"]:
            errors.append(f"cifras que no están en la entrada: {numbers['missing']}")
    return Verification(not errors, errors, details)


def verify_retriever(text: str, **_: Any) -> Verification:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return Verification(False, [f"JSON no válido: {exc}"])
    errors = validate_retrieval(obj)
    return Verification(not errors, errors, {"results": len(obj) if isinstance(obj, list) else 0})


VERIFIERS: dict[str, Callable[..., Verification]] = {
    "locator": verify_locator,
    "doc_checker": verify_doc_checker,
    "code_reviewer": verify_code_reviewer,
    "summarizer": verify_summarizer,
    "retriever": verify_retriever,
}


def verify(name: str, text: str, **context: Any) -> Verification:
    """Aplica el verificador ``name`` (ver ``VERIFIERS``) a la salida ``text``; ``ValueError`` si no existe."""
    if name not in VERIFIERS:
        raise ValueError(f"verificador desconocido: {name!r}; disponibles: {sorted(VERIFIERS)}")
    return VERIFIERS[name](text, **context)

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


def find_quote_occurrences(text: str, quote: str) -> list[tuple[int, int]]:
    """Todas las apariciones de la cita literal en ``text``, como líneas ``(primera, última)`` (desde 1), en orden.

    Se busca el fragmento tal cual (sin espacios en los extremos); si no está en ninguna parte, se buscan las
    apariciones con los espacios en blanco colapsados. Lista vacía si la cita está vacía o no aparece.
    """
    wanted = (quote or "").strip()
    if not wanted:
        return []
    found: list[tuple[int, int]] = []
    index = text.find(wanted)
    while index >= 0:
        first = text.count("\n", 0, index) + 1
        found.append((first, first + wanted.count("\n")))
        index = text.find(wanted, index + 1)
    if found:
        return found
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
    while index >= 0:
        found.append((line_of[index], line_of[index + len(norm_quote) - 1]))
        index = norm_text.find(norm_quote, index + 1)
    return found


def find_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Líneas ``(primera, última)`` (desde 1) donde aparece por primera vez la cita literal en ``text``; ``None`` si no.

    Se busca el fragmento tal cual (sin espacios en los extremos); si no está, se compara con los espacios en
    blanco colapsados.
    """
    found = find_quote_occurrences(text, quote)
    return found[0] if found else None


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


def _distance_to_span(line: int, span: tuple[int, int]) -> int:
    first, last = span
    return 0 if first <= line <= last else min(abs(line - first), abs(line - last))


def correct_review_lines(obj: Any, source: str) -> dict:
    """Corrige por código la línea de cada hallazgo del revisor buscando su cita literal en ``source``.

    El modelo suele acertar el texto de la línea pero desplazar el número. Para cada hallazgo con cita:

    - si la cita aparece en el fichero, ``linea`` pasa a ser la primera línea de la aparición más cercana a la línea
      que dio el modelo (a igual distancia, la primera; también si esa línea cae fuera del fichero); si el modelo ya cae
      dentro de esa aparición no se toca. Si la línea del modelo no es un entero se usa la primera aparición;
    - si la cita no aparece (ni colapsando espacios), el hallazgo se marca ``verificable = False`` y NO se inventa
      ninguna línea: conserva la del modelo.

    Devuelve ``{"obj", "items", "verifiable", "corrected", "unverifiable"}``: ``obj`` es una copia del objeto con
    ``linea`` corregida y las claves ``verificable`` (y ``linea_modelo`` si cambió); ``items`` informa, por hallazgo,
    de ``linea_modelo``, ``linea``, ``estado`` (``"igual"``, ``"corregida"`` o ``"no_verificable"``) y ``apariciones``.
    No modifica la entrada.
    """
    raw = obj.get("hallazgos", []) if isinstance(obj, dict) else []
    out_findings: list[Any] = []
    items = []
    for index, finding in enumerate(raw):
        if not isinstance(finding, dict):
            out_findings.append(finding)
            continue
        linea, cita = finding.get("linea"), finding.get("cita")
        spans = find_quote_occurrences(source, cita) if isinstance(cita, str) else []
        fixed = dict(finding)
        if not spans:
            fixed["verificable"] = False
            state, new_line = "no_verificable", linea
        else:
            if _is_int(linea):
                best = min(spans, key=lambda span: (_distance_to_span(linea, span), span[0]))
                new_line = linea if best[0] <= linea <= best[1] else best[0]
            else:
                new_line = spans[0][0]
            fixed["verificable"] = True
            if new_line != linea:
                fixed["linea_modelo"] = linea
                fixed["linea"] = new_line
            state = "igual" if new_line == linea else "corregida"
        out_findings.append(fixed)
        items.append({"item": index, "linea_modelo": linea, "linea": new_line, "estado": state, "apariciones": len(spans)})
    corrected = dict(obj) if isinstance(obj, dict) else {}
    corrected["hallazgos"] = out_findings
    return {
        "obj": corrected,
        "items": items,
        "verifiable": sum(i["estado"] != "no_verificable" for i in items),
        "corrected": sum(i["estado"] == "corregida" for i in items),
        "unverifiable": sum(i["estado"] == "no_verificable" for i in items),
    }


def verifiable_findings(corrected_obj: Any) -> dict:
    """Copia de un objeto ya corregido con solo los hallazgos ``verificable`` (los demás no se pueden comprobar)."""
    findings = corrected_obj.get("hallazgos", []) if isinstance(corrected_obj, dict) else []
    kept = [f for f in findings if isinstance(f, dict) and f.get("verificable") is True]
    return {**corrected_obj, "hallazgos": kept} if isinstance(corrected_obj, dict) else {"hallazgos": kept}


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


# -- correcciones por código del resumen -----------------------------------------------------------------------
_SENTENCE_END_RE = re.compile(r"[.?!…][)\]»\"”’']*$")
_ABBREVIATIONS = frozenset("p. ej. aprox. fig. vs. núm. sr. sra. dr. dra. ca. cf. pág. págs. ud. uds.".split())
_LIST_MARKER_TOKEN_RE = re.compile(r"\d+[.)]$")


def _is_sentence_end(index: int, tokens: list[re.Match]) -> bool:
    """¿Termina una frase en el trozo ``index``? No cuenta un punto de abreviatura («p. ej.») ni un marcador de lista
    («1.» al principio de línea o tras otra frase); sí cuenta un número al final de frase («época 200.»)."""
    match = tokens[index]
    text = match.group(0).strip("*_(\"“«")
    if not _SENTENCE_END_RE.search(text):
        return False
    if text.lower() in _ABBREVIATIONS or (len(text) == 2 and text[0].isalpha() and text[1] == "."):
        return False
    if _LIST_MARKER_TOKEN_RE.fullmatch(text):
        line_before = match.string[: match.start()].rsplit("\n", 1)[-1].strip()
        return bool(line_before) and not _SENTENCE_END_RE.search(line_before)  # «… tres. 2.»: «2.» es otro marcador
    return True


def trim_summary_with_mode(text: str, max_words: int = 150) -> tuple[str, str]:
    """Recorta ``text`` a ``max_words`` palabras (las que cuenta :func:`count_words`) y dice cómo: ``(texto, modo)``.

    ``modo`` es ``"intacto"`` (ya cabía; se devuelve igual), ``"frase"`` (se corta en el último final de frase
    —``.``, ``?``, ``!`` o ``…`` seguido de espacio o del final— que cabe) o ``"palabras"`` (ninguna frase cabe: se
    corta tras la palabra ``max_words`` y se quitan los separadores sueltos del final). Nunca se parte una cifra:
    un decimal («0,43») es un único trozo, un millar con espacio («21 754») no se separa y un ``%`` suelto se
    conserva junto a su cifra. Se conservan los saltos de párrafo del texto original.
    """
    if max_words < 1:
        raise ValueError("max_words debe ser >= 1")
    body = text or ""
    if count_words(body) <= max_words:
        return body, "intacto"
    tokens = list(re.finditer(r"\S+", body))
    counted = 0
    last_sentence_end = None  # índice del trozo con el que acaba la última frase completa que cabe
    cut = len(tokens) - 1
    for i, match in enumerate(tokens):
        if any(c.isalnum() for c in match.group(0)):
            counted += 1
        if counted > max_words:
            cut = i - 1
            break
        if _is_sentence_end(i, tokens):
            last_sentence_end = i
    if last_sentence_end is not None:
        end = tokens[last_sentence_end].end()
        return body[:end].rstrip(), "frase"
    # ninguna frase cabe: por palabras, sin partir cifras
    while cut > 0 and re.fullmatch(r"\d+", tokens[cut].group(0)) and cut + 1 < len(tokens) and re.match(r"\d{3}(?:\D|$)", tokens[cut + 1].group(0)):
        cut -= 1
    if cut + 1 < len(tokens) and tokens[cut + 1].group(0) == "%" and re.search(r"\d$", tokens[cut].group(0)):
        cut += 1
    trimmed = body[: tokens[cut].end()].rstrip()
    trimmed = trimmed.rstrip(" ,;:—–-(¿¡\"«“")
    return trimmed, "palabras"


def trim_summary(text: str, max_words: int = 150) -> str:
    """Recorta un resumen a ``max_words`` palabras (ver :func:`trim_summary_with_mode`)."""
    return trim_summary_with_mode(text, max_words)[0]


_DOT_DECIMAL_RE = re.compile(r"(?<![\w.,/\\:@#~])(-?)(\d+)\.(\d+)(?!\w|\.\w|,\d|[/\\])")
_THOUSANDS_LIKE_RE = re.compile(r"[1-9]\d{0,2}\.\d{3}")
_CODE_SPAN_RE = re.compile(r"(`[^`\n]*`|https?://\S+)")


def _thousands_reading(token: str, reference: list[Number]) -> str:
    """Cómo leer un número ambiguo con punto y tres cifras tras él (``2.609``) según la entrada: ``"decimal"``,
    ``"millar"`` o ``"ambiguo"`` (las dos lecturas están respaldadas o ninguna lo está)."""
    as_decimal = number_supported(Number(token, (float(token),), 3, False), reference)
    as_thousands = number_supported(Number(token, (float(token.replace(".", "")),), 0, False), reference)
    if as_decimal and not as_thousands:
        return "decimal"
    if as_thousands and not as_decimal:
        return "millar"
    return "ambiguo"


def decimal_comma(text: str, source: str | None = None) -> str:
    """Pasa el punto decimal a coma decimal en un texto en castellano sin tocar rutas, ficheros ni identificadores.

    Solo cambia un número AISLADO con punto (``0.9`` -> ``0,9``; ``-0.5`` -> ``-0,5``; ``37.5%`` -> ``37,5%``). No toca
    lo que va entre comillas invertidas ni las URL, ni lo pegado a letras, guiones bajos o barras (``v1.2``,
    ``run_0.9.csv``, ``src/a/1.5.py``), ni versiones o fechas (``1.2.3``, ``2026.11.05``), ni los números con coma
    (``1,5``). Un número que puede ser un millar con punto (``21.754``, ``2.609``: de una a tres cifras, punto y
    exactamente tres cifras) es ambiguo y se deja como está, salvo que se pase la entrada ``source`` y solo la lectura
    decimal esté respaldada por ella (``2.609`` con ``2.6094`` en la entrada); con la lectura de millar o con las dos
    (o ninguna) respaldadas se deja. Idempotente.
    """
    reference = (extract_numbers(source) + extract_numbers(source, comma_decimal=False)) if source else []

    def swap(match: re.Match) -> str:
        sign, whole, frac = match.groups()
        start = match.start()
        if not sign and start > 1 and match.string[start - 1] == "-" and (match.string[start - 2].isalpha() or match.string[start - 2] == "_"):
            return match.group(0)  # «run-0.9»: el guion es parte de un nombre
        token = f"{whole}.{frac}"
        if _THOUSANDS_LIKE_RE.fullmatch(token) and not (source and _thousands_reading(token, reference) == "decimal"):
            return match.group(0)
        return f"{sign}{whole},{frac}"

    parts = _CODE_SPAN_RE.split(text or "")
    return "".join(part if i % 2 else _DOT_DECIMAL_RE.sub(swap, part) for i, part in enumerate(parts))


FILTER_MIN_WORDS = 30  # un resumen filtrado con menos palabras (o vacío) es un fallo de formato (fijado antes de medir)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Frases de ``text`` como ``(inicio, fin)`` de caracteres: la frase acaba en el trozo que cierra :func:`_is_sentence_end` y el
    resto del texto, si lo hay, es la última frase. No se parten abreviaturas («p. ej.») ni marcadores de lista («1.»)."""
    body = text or ""
    tokens = list(re.finditer(r"\S+", body))

    def ends(i: int) -> bool:
        if _is_sentence_end(i, tokens):
            return True
        word = tokens[i].group(0).strip("*_(\"“«")
        if len(word) == 2 and word[0].isalpha() and word[1] == "." and i + 1 < len(tokens):  # unidad al final de frase («3,2 s. Otra...»)
            following = tokens[i + 1].group(0).lstrip("*_(\"“«¿¡")[:1]
            return following.isupper() or "\n" in body[tokens[i].end() : tokens[i + 1].start()]
        return False

    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, match in enumerate(tokens):
        if start is None:
            start = match.start()
        if ends(i):
            spans.append((start, match.end()))
            start = None
    if start is not None:
        spans.append((start, tokens[-1].end()))
    return spans


def filter_unsupported_sentences(text: str, source: str) -> tuple[str, dict]:
    """Quita de ``text`` las frases que contienen alguna cifra NO respaldada por ``source`` (paso determinista, sin modelo).

    Una cifra está respaldada según :func:`number_supported` (la misma comprobación que :func:`check_numbers_supported`). Las frases
    sin cifras se conservan; los separadores (espacios y saltos de párrafo) de las frases que quedan son los originales. Devuelve
    ``(texto, info)`` con ``sentences`` (total), ``removed`` (cuántas se quitan), ``removed_texts`` y ``words_before`` / ``words_after``.
    Si no queda ninguna frase, el texto es la cadena vacía.
    """
    body = text or ""
    reference = extract_numbers(source) + extract_numbers(source, comma_decimal=False)
    spans = sentence_spans(body)
    kept: list[str] = []
    removed: list[str] = []
    previous_end = 0
    for start, end in spans:
        sentence = body[start:end]
        if all(number_supported(n, reference) for n in extract_numbers(sentence)):
            kept.append((body[previous_end:start] if kept else "") + sentence)
        else:
            removed.append(sentence)
        previous_end = end
    filtered = "".join(kept).strip()
    return filtered, {"sentences": len(spans), "removed": len(removed), "removed_texts": removed, "words_before": count_words(body), "words_after": count_words(filtered)}


def postprocess_summary(text: str, source: str | None = None, max_words: int = 150, filter_numbers: bool = False) -> tuple[str, dict]:
    """Corrección por código de un resumen: coma decimal y recorte a ``max_words`` y, opcionalmente, filtro de cifras. ``(texto, info)``.

    ``info``: ``mode`` del recorte (``intacto``/``frase``/``palabras``), ``words_before`` y ``words_after``, y, con
    ``source``, las cifras respaldadas antes y después (``numbers_before``/``numbers_after``). El recorte va primero
    sobre el texto ya con coma decimal; las cifras se comparan con la entrada con :func:`check_numbers_supported`,
    que admite coma o punto decimal en el resumen.

    Con ``filter_numbers=True`` (y ``source``), DESPUÉS del recorte se quitan las frases con cifras no respaldadas
    (:func:`filter_unsupported_sentences`); ``info["filter"]`` lo detalla y ``numbers_after`` / ``words_after`` son los del texto final
    (``numbers_trimmed`` / ``words_trimmed``, los del texto solo recortado). Al ser código, el cumplimiento depende del filtro y no del modelo.
    """
    commas = decimal_comma(text, source)
    fixed, mode = trim_summary_with_mode(commas, max_words)
    info: dict = {"mode": mode, "words_before": count_words(text), "words_after": count_words(fixed)}
    if source is not None:
        info["numbers_before"] = check_numbers_supported(text, source)
    if filter_numbers and source is not None:
        info["words_trimmed"] = info["words_after"]
        info["numbers_trimmed"] = check_numbers_supported(fixed, source)
        fixed, info["filter"] = filter_unsupported_sentences(fixed, source)
        info["words_after"] = count_words(fixed)
    if source is not None:
        info["numbers_after"] = check_numbers_supported(fixed, source)
    return fixed, info


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
        # la validez (``valid``) se juzga sobre la salida CRUDA; la corrección de línea es un paso aparte e informativo
        details["line_correction"] = correct_review_lines(obj, source)
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

"""Recuperador de documentación: troceado, índice de embeddings y línea base léxica.

Piezas (solo biblioteca estándar; los embeddings los calcula un proveedor externo con ``embed``):

- :func:`chunk_project`: trocea ``context/`` (sin ``context/data/``) y ``src/`` (``.py`` y ``.md``) en
  fragmentos con ruta y línea de inicio (:class:`Chunk`).
- :func:`build_index` / :func:`save_index` / :func:`load_index`: índice en JSON (por defecto en
  ``.agent_runs/index/index.json``, fuera del control de versiones).
- :class:`EmbeddingRetriever`: similitud coseno en Python puro entre la consulta y los fragmentos.
- :class:`LexicalRetriever`: BM25 estándar (``k1=1,5``, ``b=0,75``, sin modelo) con la misma interfaz. Es el recuperador
  POR DEFECTO (``search_backend = "lexical"``); los embeddings son una opción explícita.
- :class:`HybridRetriever`: fusión de las dos anteriores por rango recíproco (RRF, ``k = 60``); opción explícita
  (``search_backend = "hybrid"``).
- :func:`search_docs_tool`: herramienta ``search_docs(query, k=3)`` de solo lectura para otros roles.

Los modelos ``nomic-embed-text`` esperan un prefijo de tarea: ``search_document: `` en los fragmentos y
``search_query: `` en las consultas (se guardan en el índice). El prefijo depende del modelo
(:func:`prefixes_for`): ``bge-m3`` NO lleva prefijo; ``nomic-embed-text`` (y cualquier otro modelo, por compatibilidad) lleva los de siempre. El índice en disco guarda el modelo y sus prefijos y NO se reutiliza con otro modelo
(:func:`index_matches`: ``run.retrieve`` lo reconstruye si el modelo no coincide).

DISEÑO FIJADO EL 2026-11-06 ANTES DE EJECUTAR NADA EN VIVO (etiqueta ``p3``; no se ajusta después de ver resultados)

Protocolo y criterio de ``eval_roles retriever``: SIN CAMBIOS (15 consultas, acierto en el top-3, umbral 80 %, ``CRITERIA`` y ``REPS``
intactos). Referencias ya medidas: BM25 73 % (11/15) y ``nomic-embed-text`` 60 % (9/15). Se hacen TRES comparaciones nuevas sobre las
mismas 15 consultas (hay riesgo de quedarse con la mejor por azar: se informa de las tres tal cual, sin elegir):

- A1 ``retriever --model bge-m3``: embeddings de ``bge-m3`` solos (1024 dimensiones, sin prefijos, mismo troceado, mismo texto
  indexado ``ruta + texto``).
- A2 ``retriever_hybrid --model bge-m3``: fusión híbrida BM25 + ``bge-m3``.
- A3 ``retriever_hybrid --model nomic-embed-text``: fusión híbrida BM25 + ``nomic-embed-text`` (con sus prefijos de siempre).

Fusión híbrida (:class:`HybridRetriever`, :func:`reciprocal_rank_fusion`): rango recíproco (RRF) con ``k = 60`` y pesos iguales sobre
las listas COMPLETAS de cada recuperador: la puntuación de un fragmento es la suma, sobre las listas en que aparece, de
``1 / (60 + rango)`` (rango desde 1). La lista de BM25 contiene solo los fragmentos con puntuación mayor que 0 (los demás no tienen
rango en ella y no suman); la de embeddings contiene todos. Desempate: mayor puntuación fusionada, después la lista en que el
fragmento aparece primero (BM25 antes que embeddings) y su rango en ella. Si el modelo de embeddings falla, la búsqueda falla (no
se vuelve a BM25 en silencio).

Medidas que se informan además del agregado: top-3 y top-1 de cada variante frente a BM25 (73 %) y a ``nomic-embed-text`` (60 %); las
consultas que pasan de acierto a fallo y de fallo a acierto respecto de BM25 (para ver si el híbrido corrige errores o solo
empata); tiempo de indexado y por consulta; procesador (``/api/ps``: si cabe en VRAM sin CPU). Comprobación previa: el fragmento
más largo (``MAX_CHUNK_CHARS`` = 1200 caracteres, hasta ~1,3 veces si no hay línea en blanco donde cortar) cabe de sobra en
``num_ctx = 2048`` de los dos modelos (se mide en vivo con ``prompt_eval_count`` antes de evaluar).
"""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .providers import ProviderError
from .tools import Policy, Tool, ToolError

DEFAULT_INDEX_DIR = ".agent_runs/index"
INDEX_FILENAME = "index.json"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
HYBRID_RRF_K = 60  # constante del rango recíproco (RRF) de la fusión híbrida (fijada de antemano)
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 400
_SKIP_PARTS = {"__pycache__", "node_modules"}


def _model_base(model: str) -> str:
    """Nombre del modelo sin etiqueta (``nomic-embed-text:latest`` -> ``nomic-embed-text``), en minúsculas."""
    return (model or "").strip().lower().split(":", 1)[0]


NO_PREFIX_MODELS = frozenset({"bge-m3"})  # modelos que NO usan prefijos de tarea


def prefixes_for(model: str) -> tuple[str, str]:
    """Prefijos ``(documento, consulta)`` que espera ``model``: cadenas vacías para los de :data:`NO_PREFIX_MODELS` (``bge-m3``) y los
    de ``nomic-embed-text`` para el resto (también si no hay nombre de modelo: es el comportamiento de siempre)."""
    if _model_base(model) in NO_PREFIX_MODELS:
        return "", ""
    return DOC_PREFIX, QUERY_PREFIX


def index_matches(index: dict, model: str) -> bool:
    """¿Se construyó ``index`` con ``model``? Un índice no se reutiliza entre modelos (otro espacio y otra dimensión).

    Sin nombre de modelo (cadena vacía) no se puede comprobar y se acepta. Se ignora la etiqueta (``:latest``).
    """
    return not _model_base(model) or _model_base(str(index.get("model", ""))) == _model_base(model)


class Embedder(Protocol):
    """Lo único que necesita el recuperador de un proveedor de embeddings (``OllamaProvider`` lo cumple)."""

    def embed(self, texts: Sequence[str], timeout: float | None = None) -> list[list[float]]: ...


@dataclass(frozen=True)
class Chunk:
    """Fragmento de un fichero: ruta relativa, línea de inicio (desde 1) y texto."""

    ruta: str
    linea: int
    texto: str


@dataclass(frozen=True)
class Hit:
    """Resultado de una búsqueda."""

    ruta: str
    linea: int
    puntuacion: float
    texto: str

    def as_dict(self) -> dict:
        return {"ruta": self.ruta, "linea": self.linea, "puntuacion": round(self.puntuacion, 6), "texto": self.texto}


# -- troceado ---------------------------------------------------------------------------------------------
def iter_source_files(root: str | Path) -> list[Path]:
    """Ficheros a indexar: ``context/**/*.md`` salvo ``context/data/``, y ``src/**/*.py`` y ``src/**/*.md``.

    Se omiten las carpetas ocultas (``.pytest_cache``, ``.agent_runs``...) y ``__pycache__``.
    """
    base = Path(root).resolve()
    files: list[Path] = []
    context = base / "context"
    if context.is_dir():
        for path in sorted(context.rglob("*.md")):
            parts = path.relative_to(context).parts
            if parts[0] == "data" or any(p.startswith(".") or p in _SKIP_PARTS for p in parts):
                continue
            files.append(path)
    src = base / "src"
    if src.is_dir():
        for path in sorted(src.rglob("*")):
            parts = path.relative_to(src).parts
            if path.suffix not in {".py", ".md"} or not path.is_file():
                continue
            if any(p.startswith(".") or p in _SKIP_PARTS for p in parts):
                continue
            files.append(path)
    return files


def _is_boundary(line: str, markdown: bool) -> bool:
    if markdown:
        return line.startswith("#")
    return bool(re.match(r"(?:def|class|async def|@)\b|(?:def|class|async def) ", line))


def _split_long(line: str, max_chars: int) -> list[str]:
    """Parte una línea larguísima en trozos de a lo sumo ``max_chars``, preferentemente tras «. »."""
    pieces = []
    rest = line
    while len(rest) > max_chars:
        cut = rest.rfind(". ", 0, max_chars)
        cut = cut + 2 if cut > max_chars // 3 else max_chars
        pieces.append(rest[:cut].strip())
        rest = rest[cut:]
    if rest.strip():
        pieces.append(rest.strip())
    return pieces


def chunk_text(ruta: str, text: str, max_chars: int = MAX_CHUNK_CHARS, min_chars: int = MIN_CHUNK_CHARS) -> list[Chunk]:
    """Trocea un fichero en fragmentos de unos ``min_chars``-``max_chars`` caracteres.

    Se corta en los encabezados (``#``, en Markdown) o definiciones de primer nivel (``def``/``class``, en
    Python) cuando el fragmento ya tiene ``min_chars``, y en una línea en blanco cuando se pasa de
    ``max_chars``. Una línea más larga que ``max_chars`` se parte en trozos con la misma línea de inicio.
    """
    markdown = ruta.endswith(".md")
    chunks: list[Chunk] = []
    current: list[str] = []
    size = 0
    start = 1

    def flush() -> None:
        nonlocal current, size
        body = "\n".join(current).rstrip()
        if body.strip():
            chunks.append(Chunk(ruta, start, body))
        current, size = [], 0

    for number, line in enumerate(text.splitlines(), start=1):
        if len(line) > max_chars:
            flush()
            chunks.extend(Chunk(ruta, number, piece) for piece in _split_long(line, max_chars))
            continue
        boundary = _is_boundary(line, markdown)
        if current and ((boundary and size >= min_chars) or (size + len(line) + 1 > max_chars and (not line.strip() or boundary or size + len(line) > 1.3 * max_chars))):
            flush()
        if not current:
            if not line.strip():
                continue
            start = number
        current.append(line)
        size += len(line) + 1
    flush()
    return chunks


def chunk_project(root: str | Path, max_chars: int = MAX_CHUNK_CHARS, min_chars: int = MIN_CHUNK_CHARS) -> list[Chunk]:
    """Trocea todo lo indexable bajo ``root`` (ver :func:`iter_source_files`)."""
    base = Path(root).resolve()
    chunks: list[Chunk] = []
    for path in iter_source_files(base):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks.extend(chunk_text(path.relative_to(base).as_posix(), text, max_chars, min_chars))
    return chunks


# -- índice de embeddings -----------------------------------------------------------------------------------
def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Similitud coseno en Python puro (0 si algún vector es nulo)."""
    if len(a) != len(b):
        raise ValueError(f"vectores de distinta dimensión ({len(a)} y {len(b)})")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def build_index(
    chunks: Sequence[Chunk],
    embedder: Embedder,
    model: str = "",
    batch_size: int = 16,
    timeout: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Calcula los embeddings de los fragmentos (por lotes) y devuelve el índice como diccionario serializable.

    El índice guarda el ``model`` y los prefijos que ese modelo espera (:func:`prefixes_for`).
    """
    doc_prefix, query_prefix = prefixes_for(model)
    entries = []
    dim = 0
    for first in range(0, len(chunks), batch_size):
        batch = chunks[first : first + batch_size]
        vectors = embedder.embed([f"{doc_prefix}{c.ruta}\n{c.texto}" for c in batch], timeout=timeout)
        if len(vectors) != len(batch):
            raise ProviderError(f"el proveedor devolvió {len(vectors)} vectores para {len(batch)} textos")
        for chunk, vector in zip(batch, vectors):
            dim = dim or len(vector)
            entries.append({"ruta": chunk.ruta, "linea": chunk.linea, "texto": chunk.texto, "vector": [round(float(x), 6) for x in vector]})
        if progress:
            progress(min(first + batch_size, len(chunks)), len(chunks))
    return {
        "version": 1,
        "model": model,
        "doc_prefix": doc_prefix,
        "query_prefix": query_prefix,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dim": dim,
        "chunks": entries,
    }


def save_index(index: dict, folder: str | Path) -> Path:
    """Guarda el índice en ``folder/index.json`` (crea la carpeta) y devuelve la ruta."""
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    path = target / INDEX_FILENAME
    path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return path


def load_index(folder: str | Path) -> dict | None:
    """Lee ``folder/index.json``; ``None`` si no existe; ``ValueError`` si está corrupto."""
    path = Path(folder) / INDEX_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"índice corrupto en {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
        raise ValueError(f"índice con formato desconocido en {path}")
    return data


def resolve_index_dir(root: str | Path, index_dir: str | Path = DEFAULT_INDEX_DIR) -> Path:
    """Carpeta del índice: ``index_dir`` si es absoluta; si no, relativa a ``root``."""
    folder = Path(index_dir)
    return folder if folder.is_absolute() else Path(root) / folder


# -- recuperadores --------------------------------------------------------------------------------------------
class Retriever(ABC):
    """Interfaz común: ``search(consulta, k)`` devuelve los ``k`` mejores :class:`Hit` por puntuación decreciente."""

    name = "retriever"

    @abstractmethod
    def search(self, query: str, k: int = 3) -> list[Hit]: ...


def _top(scored: list[tuple[float, int]], items: Sequence[Any], k: int) -> list[tuple[float, Any]]:
    order = sorted(scored, key=lambda s: (-s[0], s[1]))[: max(0, k)]
    return [(score, items[i]) for score, i in order]


class EmbeddingRetriever(Retriever):
    """Búsqueda por similitud coseno entre el embedding de la consulta y los del índice."""

    name = "embedding"

    def __init__(self, index: dict, embedder: Embedder, timeout: float | None = None):
        self.entries = index["chunks"]
        self.query_prefix = index.get("query_prefix", QUERY_PREFIX)
        self.embedder = embedder
        self.timeout = timeout

    def search(self, query: str, k: int = 3) -> list[Hit]:
        vectors = self.embedder.embed([f"{self.query_prefix}{query}"], timeout=self.timeout)
        if len(vectors) != 1:
            raise ProviderError("el proveedor no devolvió un vector para la consulta")
        scored = [(cosine(vectors[0], e["vector"]), i) for i, e in enumerate(self.entries)]
        return [Hit(e["ruta"], int(e["linea"]), score, e["texto"]) for score, e in _top(scored, self.entries, k)]


_STOPWORDS = frozenset(
    "a al algo como con cual cuales cuando de del donde el ella en es esta este esto ha han hay la las le lo los me mi "
    "muy no o para pero por que qu se si sin sobre su sus te un una unos unas y ya son ser fue puede pueden".split()
)


def tokenize(text: str) -> list[str]:
    """Palabras en minúsculas, sin acentos, sin signos y sin las palabras vacías castellanas más comunes."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", folded) if t not in _STOPWORDS]


class LexicalRetriever(Retriever):
    """Recuperador léxico por defecto: BM25 Okapi (``k1=1,5``, ``b=0,75``) sobre las palabras de los fragmentos y de la ruta.

    Puntuación de un fragmento ``d`` para los términos distintos ``t`` de la consulta (sin palabras vacías):
    ``sum_t idf(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))`` con
    ``idf(t) = ln(1 + (N - n_t + 0,5) / (n_t + 0,5))`` (la variante no negativa de Lucene).
    """

    name = "lexical"

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tf = [self._counts(tokenize(f"{c.ruta} {c.texto}")) for c in self.chunks]
        self.lengths = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.df: dict[str, int] = {}
        for counts in self.tf:
            for term in counts:
                self.df[term] = self.df.get(term, 0) + 1

    @staticmethod
    def _counts(tokens: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in tokens:
            out[t] = out.get(t, 0) + 1
        return out

    def search(self, query: str, k: int = 3) -> list[Hit]:
        terms = sorted(set(tokenize(query)))
        n = len(self.chunks)
        scored = []
        for i, counts in enumerate(self.tf):
            score = 0.0
            for term in terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                idf = math.log(1.0 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                norm = 1.0 - self.b + self.b * (self.lengths[i] / self.avg if self.avg else 1.0)
                score += idf * freq * (self.k1 + 1.0) / (freq + self.k1 * norm)
            if score > 0:
                scored.append((score, i))
        return [Hit(c.ruta, c.linea, score, c.texto) for score, c in _top(scored, self.chunks, k)]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], k_rrf: int = HYBRID_RRF_K) -> list[Hit]:
    """Fusiona listas ordenadas de :class:`Hit` por rango recíproco: ``sum_listas 1 / (k_rrf + rango)`` (rango desde 1), pesos iguales.

    Un fragmento se identifica por ``(ruta, línea, texto)``. Desempate: mayor puntuación, después la lista en que aparece primero
    (el orden de ``rankings``) y su rango en ella. La puntuación del :class:`Hit` devuelto es la fusionada.
    """
    scores: dict[tuple[str, int, str], float] = {}
    first: dict[tuple[str, int, str], tuple[int, int]] = {}
    hits: dict[tuple[str, int, str], Hit] = {}
    for list_index, ranking in enumerate(rankings):
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.ruta, hit.linea, hit.texto)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank)
            first.setdefault(key, (list_index, rank))
            hits.setdefault(key, hit)
    order = sorted(scores, key=lambda key: (-scores[key], first[key]))
    return [Hit(hits[key].ruta, hits[key].linea, scores[key], hits[key].texto) for key in order]


class HybridRetriever(Retriever):
    """Recuperador híbrido: BM25 y embeddings fusionados por rango recíproco (RRF, ``k = 60``) sobre las listas COMPLETAS.

    La lista léxica solo tiene los fragmentos con puntuación > 0; la de embeddings, todos. Ver el docstring del módulo.
    """

    name = "hybrid"

    def __init__(self, lexical: Retriever, dense: Retriever, k_rrf: int = HYBRID_RRF_K):
        self.lexical, self.dense, self.k_rrf = lexical, dense, k_rrf

    def search(self, query: str, k: int = 3) -> list[Hit]:
        everything = 10**9  # listas completas: el recorte a ``k`` se hace tras la fusión
        fused = reciprocal_rank_fusion([self.lexical.search(query, everything), self.dense.search(query, everything)], self.k_rrf)
        return fused[: max(0, k)]


def search(retriever: Retriever, query: str, k: int = 3) -> list[dict]:
    """Atajo: resultados como lista de ``{"ruta", "linea", "puntuacion", "texto"}``."""
    return [hit.as_dict() for hit in retriever.search(query, k)]


# -- herramienta search_docs -------------------------------------------------------------------------------------
def search_docs_tool(backend: str = "lexical", index_dir: str | Path = DEFAULT_INDEX_DIR, embedder: Embedder | None = None, max_k: int = 10) -> Tool:
    """Herramienta ``search_docs(query, k=3)``, de solo lectura, para los roles que la tengan en ``tools``.

    ``backend="lexical"`` trocea el proyecto en memoria (sin modelo ni índice). ``backend="embedding"`` usa el
    índice guardado en ``index_dir`` (debe haberse construido antes) y ``embedder`` para la consulta. Solo
    devuelve fragmentos de ficheros que la política del rol permite leer.
    """
    if backend not in {"lexical", "embedding", "hybrid"}:
        raise ValueError(f"backend de búsqueda desconocido: {backend!r}")
    if backend in {"embedding", "hybrid"} and embedder is None:
        raise ValueError(f"el backend {backend!r} necesita un embedder")
    cache: dict[str, Retriever] = {}

    def func(policy: Policy, args: dict) -> str:
        query = str(args["query"]).strip()
        if not query:
            raise ToolError("la consulta está vacía")
        k = max(1, min(int(args.get("k", 3)), max_k))
        key = str(policy.root)
        if key not in cache:
            if backend == "lexical":
                cache[key] = LexicalRetriever(chunk_project(policy.root))
            else:
                try:
                    index = load_index(resolve_index_dir(policy.root, index_dir))
                except ValueError as exc:
                    raise ToolError(str(exc)) from exc
                if index is None:
                    raise ToolError("no existe el índice de documentación; hay que construirlo antes (retriever --build-index)")
                model = str(getattr(embedder, "model", "") or "")
                if not index_matches(index, model):
                    raise ToolError(f"el índice se construyó con {index.get('model')!r} y el modelo de la consulta es {model!r}: hay que reconstruirlo (retriever --build-index)")
                dense = EmbeddingRetriever(index, embedder)  # type: ignore[arg-type]
                cache[key] = dense if backend == "embedding" else HybridRetriever(LexicalRetriever(chunk_project(policy.root)), dense)
        try:
            hits = cache[key].search(query, k * 3)
        except ProviderError as exc:
            raise ToolError(f"el recuperador no está disponible: {exc}") from exc
        allowed = [h for h in hits if policy.can(policy.root / h.ruta, "read")][:k]
        if not allowed:
            return "(sin resultados)"
        return "\n---\n".join(f"{h.ruta}:{h.linea} (puntuación {h.puntuacion:.3f})\n{h.texto[:700]}" for h in allowed)

    return Tool(
        "search_docs",
        "Busca en la documentación y el código del proyecto los fragmentos más relacionados con una consulta en lenguaje natural (solo lectura). Devuelve ruta:línea y el texto.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Consulta en lenguaje natural"},
                "k": {"type": "integer", "description": "Número de fragmentos (por defecto 3)"},
            },
            "required": ["query"],
        },
        func,
    )

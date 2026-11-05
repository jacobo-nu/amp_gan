"""Tests del recuperador (troceado, índice, coseno, línea base léxica, search_docs) y de ``embed``; sin red ni Ollama."""

import json
import math
from pathlib import Path

import pytest
from agents_helpers import make_project, write
from fake_embedder import FakeEmbedder
from test_agents_providers import FakeServer

from amp_agents import agents as A
from amp_agents import retrieval as RT
from amp_agents.loop import run_agent
from amp_agents.providers import AssistantMessage, OllamaProvider, ProviderError, ProviderTimeout, ScriptedProvider


@pytest.fixture
def server():
    srv = FakeServer()
    yield srv
    srv.close()


# -- OllamaProvider.embed contra el servidor HTTP de prueba ----------------------------------------------------------
def test_embed_sends_native_request_and_returns_vectors_in_order(server):
    server.responses.append((200, {"embeddings": [[1, 0, 0.5], [0, 2, 0]], "model": "nomic-embed-text"}))
    provider = OllamaProvider(server.base_url, "nomic-embed-text", options={"num_ctx": 2048, "keep_alive": "5m", "think": True, "num_predict": 9})
    vectors = provider.embed(["uno", "dos"])
    assert vectors == [[1.0, 0.0, 0.5], [0.0, 2.0, 0.0]] and all(isinstance(x, float) for v in vectors for x in v)
    sent = server.requests[0]
    assert sent["path"] == "/api/embed"
    assert sent["body"]["model"] == "nomic-embed-text" and sent["body"]["input"] == ["uno", "dos"]
    assert sent["body"]["options"] == {"num_ctx": 2048} and sent["body"]["keep_alive"] == "5m"
    assert "think" not in sent["body"] and "num_predict" not in sent["body"]["options"] and "stream" not in sent["body"]


def test_embed_accepts_a_single_string_and_skips_empty_input(server):
    server.responses.append((200, {"embeddings": [[0.1, 0.2]]}))
    assert OllamaProvider(server.base_url, "m").embed("hola") == [[0.1, 0.2]]
    assert server.requests[0]["body"]["input"] == ["hola"] and "options" not in server.requests[0]["body"]
    assert OllamaProvider(server.base_url, "m").embed([]) == [] and len(server.requests) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"embeddings": [[1.0]]},  # un vector para dos textos
        {"embeddings": [[1.0], "x"]},
        {"embeddings": [[1.0], []]},
        {"embeddings": [[1.0], ["a"]]},
        {"embeddings": [[1.0], [True]]},
        {"otra": 1},
        [],
    ],
)
def test_embed_rejects_malformed_responses(server, payload):
    server.responses.append((200, payload))
    with pytest.raises(ProviderError):
        OllamaProvider(server.base_url, "m").embed(["a", "b"])


def test_embed_errors_and_timeout(server):
    server.responses.append((500, "boom"))
    with pytest.raises(ProviderError, match="HTTP 500"):
        OllamaProvider(server.base_url, "m").embed(["a"])
    server.responses.append(("slow", ""))
    with pytest.raises(ProviderTimeout):
        OllamaProvider(server.base_url, "m", timeout=60).embed(["a"], timeout=1)
    with pytest.raises(ProviderError, match="solo admite textos"):
        OllamaProvider(server.base_url, "m").embed([1])
    with pytest.raises(ProviderError, match="no responde"):
        OllamaProvider("http://127.0.0.1:9", "m", timeout=2).embed(["a"])


# -- troceado -------------------------------------------------------------------------------------------------------------
def test_chunk_markdown_splits_at_headings_with_start_lines():
    text = "# Uno\n" + "a" * 500 + "\n\n## Dos\n" + "b" * 500 + "\nfin\n"
    chunks = RT.chunk_text("doc.md", text, max_chars=1200, min_chars=400)
    assert [(c.ruta, c.linea) for c in chunks] == [("doc.md", 1), ("doc.md", 4)]
    assert chunks[0].texto.startswith("# Uno") and chunks[1].texto.startswith("## Dos") and chunks[1].texto.endswith("fin")


def test_chunk_python_splits_at_definitions_and_keeps_small_ones_together():
    code = "import os\n\n\ndef f():\n    return 1\n\n\ndef g():\n    return 2\n"
    assert [c.linea for c in RT.chunk_text("m.py", code, min_chars=400)] == [1]  # todo cabe en un fragmento
    big = "import os\n\n\n" + "".join(f"def f{i}():\n    return {i}  # " + "x" * 60 + "\n\n\n" for i in range(30))
    chunks = RT.chunk_text("m.py", big, max_chars=300, min_chars=100)
    assert len(chunks) > 3 and chunks[0].linea == 1
    starts = [c.linea for c in chunks]
    assert starts == sorted(starts) and all(len(c.texto) <= 400 for c in chunks)
    lines = big.splitlines()
    for chunk in chunks:  # la línea de inicio es la de la primera línea del fragmento
        assert lines[chunk.linea - 1] == chunk.texto.splitlines()[0]


def test_chunk_splits_a_giant_line_with_the_same_start_line():
    sentence = "Esta es una frase de prueba. "
    text = "# T\n" + sentence * 100 + "\nfin\n"
    chunks = RT.chunk_text("g.md", text, max_chars=500, min_chars=100)
    giant = [c for c in chunks if c.linea == 2]
    assert len(giant) >= 5 and all(len(c.texto) <= 500 for c in giant)
    assert any(c.texto == "fin" for c in chunks)


def test_chunk_skips_blank_files_and_leading_blank_lines():
    assert RT.chunk_text("a.md", "\n\n   \n") == []
    assert RT.chunk_text("a.md", "\n\nhola\n")[0].linea == 3


def test_iter_source_files_excludes_data_hidden_and_other_suffixes(tmp_path):
    root = make_project(tmp_path / "p")
    write(root / "context" / "agents" / "x.md", "# x\n")
    write(root / "context" / "data" / "README.md", "# datos\n")
    write(root / "context" / "references" / "README.md", "# refs\n")
    write(root / "context" / "notas.txt", "no\n")
    write(root / "src" / "notebooks" / "n.ipynb", "{}")
    write(root / "src" / "doc.md", "# doc\n")
    write(root / "src" / ".pytest_cache" / "README.md", "# no\n")
    write(root / "src" / "amp_gan" / "__pycache__" / "m.py", "x = 1\n")
    names = {p.relative_to(root).as_posix() for p in RT.iter_source_files(root)}
    assert "context/agents/x.md" in names and "context/references/README.md" in names
    assert "src/doc.md" in names and "src/amp_gan/mod.py" in names and "src/tests/test_mod.py" in names
    assert "context/data/README.md" not in names and "context/data/raw/raw.fasta" not in names
    assert not any(n.endswith((".txt", ".ipynb")) or ".pytest_cache" in n or "__pycache__" in n for n in names)
    assert all(not n.startswith(("entregas", ".git", ".claude")) for n in names)


def test_chunk_project_uses_relative_paths(tmp_path):
    root = make_project(tmp_path / "p")
    chunks = RT.chunk_project(root)
    assert chunks and all(not Path(c.ruta).is_absolute() and c.linea >= 1 for c in chunks)
    assert any(c.ruta == "src/amp_gan/mod.py" and c.linea == 1 for c in chunks)


# -- coseno, índice y recuperador de embeddings --------------------------------------------------------------------------
def test_cosine():
    assert RT.cosine([1, 0], [1, 0]) == pytest.approx(1.0) and RT.cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert RT.cosine([1, 1], [-1, -1]) == pytest.approx(-1.0) and RT.cosine([3, 4], [6, 8]) == pytest.approx(1.0)
    assert RT.cosine([1, 2], [2, 1]) == pytest.approx(4 / 5) and RT.cosine([0, 0], [1, 1]) == 0.0
    assert RT.cosine([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(10 / 14) and not math.isnan(RT.cosine([0], [0]))
    with pytest.raises(ValueError, match="dimensión"):
        RT.cosine([1, 2], [1])


CHUNKS = [
    RT.Chunk("context/a.md", 1, "El backend de la GAN se elige con la variable AMP_GAN_BACKEND."),
    RT.Chunk("context/b.md", 7, "La gobernanza describe las ramas feat y fix y los commits."),
    RT.Chunk("src/m.py", 3, "def hydrophobicity(seq): escala de Kyte y Doolittle para el GRAVY."),
]


def test_build_index_batches_and_stores_prefixes(tmp_path):
    fake = FakeEmbedder()
    index = RT.build_index(CHUNKS, fake, model="m", batch_size=2)
    assert [len(c) for c in fake.calls] == [2, 1] and all(t.startswith("search_document: ") for c in fake.calls for t in c)
    assert "context/a.md\nEl backend" in fake.calls[0][0]  # la ruta forma parte del texto indexado
    assert index["model"] == "m" and index["dim"] == 64 and len(index["chunks"]) == 3
    assert index["chunks"][1]["ruta"] == "context/b.md" and index["chunks"][1]["linea"] == 7
    assert index["doc_prefix"] == RT.DOC_PREFIX and index["query_prefix"] == RT.QUERY_PREFIX
    path = RT.save_index(index, tmp_path / ".agent_runs" / "index")
    assert path.name == "index.json" and RT.load_index(tmp_path / ".agent_runs" / "index") == json.loads(path.read_text(encoding="utf-8"))
    assert RT.load_index(tmp_path / "nada") is None
    seen = []
    RT.build_index(CHUNKS, FakeEmbedder(), batch_size=2, progress=lambda done, total: seen.append((done, total)))
    assert seen == [(2, 3), (3, 3)]


def test_build_index_rejects_wrong_vector_count():
    class Bad:
        def embed(self, texts, timeout=None):
            return [[1.0]]

    with pytest.raises(ProviderError, match="vectores"):
        RT.build_index(CHUNKS, Bad())


def test_load_index_rejects_corrupt_files(tmp_path):
    write(tmp_path / "index.json", "no es json")
    with pytest.raises(ValueError, match="corrupto"):
        RT.load_index(tmp_path)
    write(tmp_path / "index.json", '{"x": 1}')
    with pytest.raises(ValueError, match="formato"):
        RT.load_index(tmp_path)


def test_embedding_retriever_orders_by_cosine_and_returns_the_interface_fields():
    fake = FakeEmbedder()
    retriever = RT.EmbeddingRetriever(RT.build_index(CHUNKS, fake), fake)
    hits = retriever.search("backend de la GAN AMP_GAN_BACKEND", k=2)
    assert [h.ruta for h in hits][0] == "context/a.md" and len(hits) == 2
    assert hits[0].puntuacion >= hits[1].puntuacion and hits[0].linea == 1
    assert set(hits[0].as_dict()) == {"ruta", "linea", "puntuacion", "texto"}
    assert fake.calls[-1][0].startswith("search_query: ")
    assert RT.search(retriever, "ramas feat fix gobernanza", 1)[0]["ruta"] == "context/b.md"
    assert retriever.search("x", k=0) == []
    assert len(retriever.search("lo que sea", k=10)) == 3  # nunca más que fragmentos


def test_embedding_retriever_breaks_ties_by_index_order():
    class Constant:
        def embed(self, texts, timeout=None):
            return [[1.0, 0.0] for _ in texts]

    retriever = RT.EmbeddingRetriever(RT.build_index(CHUNKS, Constant()), Constant())
    assert [h.ruta for h in retriever.search("x", 3)] == ["context/a.md", "context/b.md", "src/m.py"]


# -- línea base léxica -----------------------------------------------------------------------------------------------------
def test_tokenize_folds_accents_and_drops_stopwords():
    assert RT.tokenize("¿Cómo se ELIGE el backend por defecto?") == ["elige", "backend", "defecto"]
    assert RT.tokenize("Hidrofobicidad, GRAVY (p90)") == ["hidrofobicidad", "gravy", "p90"]
    assert RT.tokenize("de la el") == [] and RT.tokenize("mae_vs_reference") == ["mae", "vs", "reference"]


def test_lexical_retriever_ranks_rare_terms_and_ignores_zero_scores():
    lexical = RT.LexicalRetriever(CHUNKS)
    hits = lexical.search("¿qué escala de Kyte se usa para el GRAVY?", k=3)
    assert hits[0].ruta == "src/m.py" and hits[0].linea == 3
    assert all(h.puntuacion > 0 for h in hits) and hits == sorted(hits, key=lambda h: -h.puntuacion)
    assert lexical.search("xyzzy inexistente", 3) == [] and lexical.search("", 3) == []
    assert lexical.search("backend", 1)[0].ruta == "context/a.md"
    assert RT.LexicalRetriever([]).search("algo", 3) == []


def test_lexical_bm25_prefers_short_documents_and_repeated_terms():
    chunks = [RT.Chunk("a.md", 1, "gan " + "relleno " * 50), RT.Chunk("b.md", 1, "gan gan gan"), RT.Chunk("c.md", 1, "otra cosa distinta")]
    hits = RT.LexicalRetriever(chunks).search("gan", 3)
    assert [h.ruta for h in hits] == ["b.md", "a.md"]


def test_lexical_and_embedding_share_the_interface():
    fake = FakeEmbedder()
    for retriever in (RT.LexicalRetriever(CHUNKS), RT.EmbeddingRetriever(RT.build_index(CHUNKS, fake), fake)):
        assert isinstance(retriever, RT.Retriever)
        hit = retriever.search("backend AMP_GAN_BACKEND", 1)[0]
        assert (hit.ruta, hit.linea) == ("context/a.md", 1) and isinstance(hit.puntuacion, float)


# -- herramienta search_docs ----------------------------------------------------------------------------------------------------
def _search_project(tmp_path):
    root = make_project(tmp_path / "p")
    write(root / "context" / "constitution" / "tech_stack.md", "# Stack\nEl backend se elige con AMP_GAN_BACKEND.\n")
    write(root / "context" / "agents" / "otro.md", "# Otro\nTexto sobre gobernanza y ramas.\n")
    return root


def _tool_config(backend="lexical", read=("context", "src")):
    return A.AgentConfig(name="x", model="m", tools=["search_docs"], read=list(read), write=[], search_backend=backend)


def test_search_docs_lexical_tool_is_read_only_and_returns_path_and_line(tmp_path):
    root = _search_project(tmp_path)
    cfg = _tool_config()
    (tool,) = A.make_tools(cfg)
    assert tool.name == "search_docs" and tool.parameters["required"] == ["query"]
    policy = A.make_policy(cfg, root)
    out = tool.run(policy, {"query": "backend AMP_GAN_BACKEND", "k": 2})
    assert out.ok and out.content.startswith("context/constitution/tech_stack.md:1 (puntuación")
    assert tool.run(policy, {"query": "palabraimposible"}).content == "(sin resultados)"
    assert not tool.run(policy, {"query": "  "}).ok and not tool.run(policy, {}).ok
    with pytest.raises(Exception, match="denegado"):
        policy.check("src/amp_gan/mod.py", "write")
    assert not list((root / ".agent_runs").glob("*")) if (root / ".agent_runs").exists() else True


def test_search_docs_respects_the_read_roots_of_the_role(tmp_path):
    root = _search_project(tmp_path)
    cfg = _tool_config(read=("src",))
    (tool,) = A.make_tools(cfg)
    out = tool.run(A.make_policy(cfg, root), {"query": "backend AMP_GAN_BACKEND gobernanza", "k": 5})
    assert "context/" not in out.content


def test_search_docs_embedding_backend_needs_a_built_index_then_uses_it(tmp_path):
    root = _search_project(tmp_path)
    fake = FakeEmbedder()
    cfg = _tool_config("embedding")
    (tool,) = A.make_tools(cfg, embedder=fake)
    policy = A.make_policy(cfg, root)
    missing = tool.run(policy, {"query": "backend"})
    assert not missing.ok and "no existe el índice" in missing.content
    RT.save_index(RT.build_index(RT.chunk_project(root), fake), RT.resolve_index_dir(root))
    (tool,) = A.make_tools(cfg, embedder=fake)
    out = tool.run(policy, {"query": "backend AMP_GAN_BACKEND", "k": 1})
    assert out.ok and out.content.startswith("context/constitution/tech_stack.md:1")


def test_search_docs_reports_embedding_server_failures_as_tool_errors(tmp_path):
    root = _search_project(tmp_path)

    class Down:
        def embed(self, texts, timeout=None):
            if texts[0].startswith("search_document"):
                return FakeEmbedder().embed(texts)
            raise ProviderError("servidor caído")

    RT.save_index(RT.build_index(RT.chunk_project(root), Down()), RT.resolve_index_dir(root))
    cfg = _tool_config("embedding")
    (tool,) = A.make_tools(cfg, embedder=Down())
    out = tool.run(A.make_policy(cfg, root), {"query": "backend"})
    assert not out.ok and "servidor caído" in out.content


def test_search_docs_can_be_registered_in_a_role_and_used_by_the_loop(tmp_path):
    root = _search_project(tmp_path)
    cfg = A.AgentConfig(name="locator", model="m", tools=["grep", "search_docs"], read=["context", "src"], write=[])
    provider = ScriptedProvider([AssistantMessage.calls(("search_docs", {"query": "backend AMP_GAN_BACKEND", "k": 1})), '{"respuestas": []}'])
    result = run_agent(provider, "s", "t", A.make_tools(cfg), A.make_policy(cfg, root), max_steps=4, log_dir=tmp_path / "logs")
    assert result.stop_reason == "final" and result.tools_called == ["search_docs"] and result.tool_errors == 0
    assert "tech_stack.md:1" in provider.received[1][-1]["content"]


def test_search_docs_tool_validation():
    with pytest.raises(ValueError, match="desconocido"):
        RT.search_docs_tool("magia")
    with pytest.raises(ValueError, match="embedder"):
        RT.search_docs_tool("embedding")


def test_retrieve_builds_saves_and_reuses_the_index(tmp_path):
    from amp_agents.run import retrieve

    root = _search_project(tmp_path)
    fake = FakeEmbedder()
    first = retrieve("backend AMP_GAN_BACKEND", root=root, k=1, embedder=fake)
    assert first[0]["ruta"] == "context/constitution/tech_stack.md" and (root / ".agent_runs" / "index" / "index.json").is_file()
    document_calls = sum(1 for call in fake.calls if call[0].startswith("search_document"))
    retrieve("gobernanza ramas", root=root, k=1, embedder=fake)
    assert sum(1 for call in fake.calls if call[0].startswith("search_document")) == document_calls  # no reindexa
    retrieve("gobernanza ramas", root=root, k=1, embedder=fake, build=True)
    assert sum(1 for call in fake.calls if call[0].startswith("search_document")) == 2 * document_calls
    lexical = retrieve("gobernanza ramas", root=root, k=1, lexical=True)
    assert lexical[0]["ruta"] == "context/agents/otro.md"


def test_index_dir_can_be_configured_and_absolute(tmp_path):
    root = tmp_path / "p"
    assert RT.resolve_index_dir(root) == root / ".agent_runs" / "index"
    assert RT.resolve_index_dir(root, "otro/sitio") == root / "otro" / "sitio"
    assert RT.resolve_index_dir(root, tmp_path / "abs") == tmp_path / "abs"

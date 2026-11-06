# Agente: Recuperador híbrido (BM25 + embeddings)

**Implementación**: variante del recuperador (`retriever`), rol `retriever_hybrid` de `src/amp_agents/agents.toml`. Fusiona la lista de BM25 y la de un modelo de embeddings (`nomic-embed-text` por defecto; `--model bge-m3`) por rango recíproco: puntuación = suma de 1/(60 + rango) sobre las listas completas, con pesos iguales.

## Responsabilidades, entradas, salidas y permisos
Los mismos que `retriever.md`: mismo troceado, misma consulta y misma lista `{ruta, linea, puntuacion, texto}`; lectura de `context/` y `src/`, escritura solo del índice.

## Límites
- No es el valor por defecto: mejora el top-1 pero no el top-3 frente a BM25 (BM25 + `bge-m3` 73 %, BM25 + `nomic-embed-text` 67 %, BM25 80 %, sobre 15 preguntas).
- Mismo criterio de evaluación que el recuperador (top-3 ≥ 80 %).

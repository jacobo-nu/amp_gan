# Agente: Recuperador de documentación

**Implementación**: modelo de embeddings local (`nomic-embed-text`), configurado en `src/amp_agents/agents.toml`. No genera texto: indexa trozos de `context/` y de la documentación de `src/` y devuelve los más cercanos a una consulta. Se ofrece también como herramienta de solo lectura (`search_docs`) para los demás agentes.

## Responsabilidades
- Construir un índice de fragmentos (con ruta y línea de inicio) y mantenerlo fuera del repositorio (`.agent_runs/`).
- Devolver, para una consulta en castellano, los k fragmentos más cercanos con su ruta, su línea y su puntuación.

## Entradas
Una consulta en lenguaje natural y k (por defecto 3).

## Salidas
Lista de `{"ruta", "linea", "puntuacion", "texto"}` ordenada por puntuación.

## Permisos
- Lectura: `context/` (sin `context/data/`) y `src/`. Escritura: solo el índice en `.agent_runs/`.

## Límites
- No resume, no responde, no interpreta: solo recupera fragmentos.
- Se compararon variantes con las 15 preguntas de `eval_retriever` sobre el mismo corpus (970 fragmentos, 2026-11-06): BM25 acertó 12/15 (80 %) en el top-3; `bge-m3` solo 9/15 (60 %); la fusión por rango recíproco (RRF, k = 60, pesos iguales) BM25 + `bge-m3` 11/15 (73 %) y BM25 + `nomic-embed-text` 10/15 (67 %); `nomic-embed-text` solo 8/15 (53 %). Ninguna variante con embeddings supera a BM25 ni corrige un fallo suyo (el híbrido solo mejora el top-1), así que **BM25 es el recuperador por defecto** (`search_backend = "lexical"`) y los embeddings (`"embedding"`) y el híbrido (`"hybrid"`, rol `retriever_hybrid`, `--hybrid`) son opciones explícitas. Las referencias antiguas (BM25 73 %, `nomic` 60 %) se midieron con un corpus anterior y no son comparables con estas. `bge-m3` (1 024 dimensiones, 1,2 GB) no usa prefijos de tarea (`nomic-embed-text` sí); el índice guarda el modelo y se reconstruye si cambia. Cabe entero en la GPU de 6 GB (indexado de unos 26 s para 970 fragmentos).

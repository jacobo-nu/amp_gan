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
- Se comparó con una búsqueda léxica BM25 (k1 = 1,5, b = 0,75): los embeddings de `nomic-embed-text` dieron un 60 % de acierto en el top-3 frente al 73 % de BM25, así que **BM25 es el recuperador por defecto** (`search_backend = "lexical"`) y los embeddings son una opción explícita (`search_backend = "embedding"`, `--embedding` en la línea de órdenes). Un modelo de embeddings mejor para castellano (`bge-m3`) queda sin probar a falta de permiso para descargarlo.

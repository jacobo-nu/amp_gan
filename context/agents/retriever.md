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
- Debe compararse siempre con una búsqueda léxica sencilla (TF-IDF o similar); si no la mejora, se documenta.

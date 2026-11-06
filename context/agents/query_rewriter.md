# Agente: Reescritor de consultas (no adoptado)

**Implementación**: rol `query_rewriter` de `src/amp_agents/agents.toml` (módulo `amp_agents/rewrite.py`). Un modelo pequeño propone de 3 a 8 palabras clave o identificadores (salida restringida `{"palabras": [...]}`, temperatura 0) y la consulta final es la pregunta original más esas palabras; el troceado y BM25 no cambian.

## Criterio (fijado el 2026-11-06, antes de medir)
Top-3 con reescritura ≥ top-3 de BM25 simple + 2 consultas, y como mucho 1 consulta que pase de acierto a fallo (15 preguntas).

## Resultado
`llama3.2:3b` 10/15, `gemma3:4b` 11/15 y `phi4-mini` 13/15 frente a BM25 12/15 en la misma ejecución: ninguno cumple. Mejora solo la consulta q06 (origen de los datos) y empeora q14 («tests» y «limpieza» arrastran código nuevo antes que el roadmap). Con 15 preguntas el resultado es solo indicativo. **No se adopta**; BM25 simple sigue por defecto.

## Permisos
Herramientas: ninguna. Lectura y escritura de ficheros: ninguna (solo propone palabras).

# Agente: Enrutador de peticiones

**Implementación**: rol `router` de `src/amp_agents/agents.toml` (módulo `amp_agents/router.py`). Elige uno de seis roles (`locator`, `summarizer`, `code_reviewer`, `retriever`, `doc_checker`, `executor`) para atender una petición en castellano, con salida restringida `{"rol": enum}`. Con `tev1` se usa una pregunta de tipo `choice` por `/v1/systemone` y se toma la opción de mayor probabilidad. No ejecuta nada: solo propone el rol.

## Criterios (fijados el 2026-11-06, antes de medir)
Exactitud ≥ 85 % y recall ≥ 60 % en cada rol, sobre 48 peticiones (8 por rol, `src/amp_agents/fixtures/router_requests.json`) escritas por el mismo autor del experimento y a partir de las definiciones de los roles: posible sesgo de coherencia con ellas.

## Resultado (una repetición, temperatura 0)
`gemma3:4b` 97,9 %, `tev1` 95,8 %, `llama3.2:3b` y `phi4-mini` 85,4 % (cumplen por una sola petición), `lfm3:1.7b` 83,3 % (no cumple la exactitud). La confusión más repetida es enviar a `retriever` peticiones de otros roles. Con 8 peticiones por rol, un recall de 5/8 o 6/8 no distingue entre modelos. Candidato más sólido: `gemma3:4b`, pendiente de contrastar con un conjunto escrito por otra persona.

## Permisos
Herramientas: ninguna. Lectura y escritura de ficheros: ninguna.

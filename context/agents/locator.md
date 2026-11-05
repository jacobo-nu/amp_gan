# Agente: Localizador de código

**RETIRADO (2026-11-05).** Con `llama3.2:3b` acertó el 10 % de las preguntas (criterio fijado: ≥ 80 %); lo que hace es `grep`, y el modelo solo lo empeora. La función la cubre ahora `amp_agents.locate.locate` (determinista, solo lectura, sin modelo). El rol sigue en `agents.toml` y en `eval_roles.py` únicamente para reproducir el resultado negativo (`eval_roles locator`); no entra en `all`.

**Implementación (histórica)**: agente con modelo local (`llama3.2:3b`), configurado en `src/amp_agents/agents.toml`. Solo lectura.

## Responsabilidades
- Responder preguntas de localización sobre el repositorio: dónde se define una función o clase, qué ficheros importan un módulo, dónde aparece una constante.
- Usar siempre `grep` y `list_dir` para localizar; nunca contar ni deducir de memoria.

## Entradas
Una pregunta de localización en una frase.

## Salidas
Un único objeto JSON, sin texto alrededor:
```
{"respuestas": [{"ruta": "<ruta relativa>", "linea": <entero>, "texto": "<línea literal>"}]}
```
Si no hay coincidencias: `{"respuestas": []}`.

## Permisos
- Herramientas: `list_dir`, `grep`, `read_file`.
- Lectura: `src/` y `context/`. Escritura: ninguna.

## Límites
- No interpreta ni opina; solo localiza. No propone cambios.
- Cada respuesta debe poder comprobarse: la línea citada tiene que existir en el fichero.

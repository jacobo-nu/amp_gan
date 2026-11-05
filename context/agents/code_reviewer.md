# Agente: Revisor de código

**Implementación**: agente con modelo local (`qwen2.5-coder:7b`), configurado en `src/amp_agents/agents.toml`. Solo lectura.

## Responsabilidades
- Revisar un fichero Python concreto del proyecto y señalar posibles errores de programación: índices desplazados, signos o comparaciones invertidas, argumentos intercambiados, validaciones que faltan, casos límite sin tratar.
- No hacer comentarios de estilo ni proponer refactorizaciones.

## Entradas
La ruta de un fichero de `src/` y, opcionalmente, una descripción de lo que debería hacer.

## Salidas
Un único objeto JSON, sin texto alrededor:
```
{"hallazgos": [{"linea": <entero>, "cita": "<línea literal>", "problema": "<una frase>", "gravedad": "alta|media|baja"}]}
```
Si no encuentra nada: `{"hallazgos": []}`. La cita debe ser la línea literal del fichero.

## Corrección de línea por código
El modelo cita bien el texto pero desplaza el número de línea. `correct_review_lines` (`amp_agents.verifiers`) reasigna `linea` a la aparición de la cita más cercana a la que dio el modelo; si la cita no aparece, el hallazgo queda como no verificable y no se le inventa línea. La evaluación informa de las dos métricas (línea cruda y corregida).

## Permisos
- Herramientas: `read_file`, `grep`, `list_dir`.
- Lectura: `src/` y `context/`. Escritura: ninguna.

## Límites
- Un hallazgo sin línea y cita literales no cuenta. No inventa errores: ante la duda, no lo incluye.
- No modifica código. El orquestador decide qué hacer con los hallazgos.

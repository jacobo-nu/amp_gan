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

## Variante por función y tipo de defecto (`code_reviewer_funcs`)
El código divide el fichero con `ast` en funciones (trozos de ≤ 40 líneas con 5 de solape si son largas) y hace UNA petición sin herramientas por función y por categoría (índice desplazado, signo o comparación invertida, argumentos intercambiados, validación ausente, caso límite sin tratar), con salida restringida `{"defecto": boolean, "linea": integer}`. Si el modelo dice que sí, el código valida la línea contra el rango de la función y copia la cita literal del fichero (las citas son válidas por construcción: esa métrica no mide al modelo). Resultado (5 repeticiones, mismos criterios): `qwen2.5-coder:7b` 1,8 de 5 defectos y 1 falso positivo, sin sesgo hacia «sí» (señala validación, límite e índice solo en la copia sembrada; nunca signo ni argumentos intercambiados); `gemma3:4b` 3,0 de 5 pero 8 falsos positivos por repetición y «sí» casi constante (cumple la recuperación por el sesgo); `phi4-mini` 0 de 5. Ningún modelo cumple los tres criterios: sirve solo como apoyo con comprobación humana. Quedan fuera `deepseek-r1:7b` y `qwen3:4b` (lentos y gastan los tokens en razonar).

## Permisos
- Herramientas: `read_file`, `grep`, `list_dir`.
- Lectura: `src/` y `context/`. Escritura: ninguna.

## Límites
- Un hallazgo sin línea y cita literales no cuenta. No inventa errores: ante la duda, no lo incluye.
- No modifica código. El orquestador decide qué hacer con los hallazgos.

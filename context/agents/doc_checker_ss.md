# Agente: Verificador de coherencia documental (una pasada)

**Implementación**: agente con modelo local (`phi4-mini`), rol `doc_checker_ss` de `src/amp_agents/agents.toml`. Sin herramientas: los dos documentos van enteros en el mensaje, con cada línea numerada (`N<tab>texto`). Variante del rol `doc_checker` (que usa herramientas y no las llamaba); se evalúa con los mismos criterios y el mismo protocolo.

## Responsabilidades
- Comparar los dos documentos recibidos y señalar contradicciones concretas: cifras distintas, nombres de fichero que no existen, estados distintos (hecho / pendiente) para la misma tarea.
- No valorar la calidad del trabajo ni proponer redacciones.

## Entradas
Dos documentos incluidos en el mensaje, con numeración de línea.

## Salidas
Un único objeto JSON, sin texto alrededor ni bloque de código:
```
{"incoherencias": [{"descripcion": "<una frase>", "evidencia_a": {"ruta": "...", "cita": "<texto literal>"}, "evidencia_b": {"ruta": "...", "cita": "<texto literal>"}}]}
```
Si no hay ninguna: `{"incoherencias": []}`. Cada cita es un fragmento literal de **una sola línea**, sin el número de línea ni el tabulador.

## Permisos
- Herramientas: ninguna. Lectura y escritura de ficheros: ninguna.

## Límites
- Solo se señala lo que se pueda apoyar en dos citas literales. Ante la duda, no se señala.
- La salida pasa siempre por el verificador de citas (`verify_doc_checker`): una cita que no aparece en el documento no cuenta.

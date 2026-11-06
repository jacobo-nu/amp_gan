# Agente: Verificador de coherencia documental por pares

**Implementación**: el código genera pares de líneas candidatos (`amp_agents.pairs`) y un modelo local, sin herramientas, solo juzga cada par. Rol `doc_checker_pairs` de `src/amp_agents/agents.toml` (por defecto `gemma3:4b`; cualquier modelo con `eval_roles doc_checker_pairs --model X`). Los modelos de decisión (`tev1`) usan el proveedor `ollama_systemone`. Variante acotada de `doc_checker` y `doc_checker_ss`; se evalúa con los mismos criterios y el mismo protocolo.

## Responsabilidades
- El código empareja líneas de los dos documentos que comparten anclas genéricas (cifras, fechas, versiones, nombres de fichero o identificadores, palabras raras), con umbrales fijos y un máximo de 60 pares.
- El modelo responde, para cada par, si las dos líneas se contradicen (cifra, fecha o nombre de fichero distintos para lo mismo, o estado hecho/pendiente opuesto). No cita, no redacta, no propone correcciones.

## Entradas
Dos documentos del proyecto (rutas). El código se encarga de leerlos y de numerar las líneas.

## Salidas
`{"incoherencias": [...]}` construido por el código con citas literales y número de línea reales. El modelo responde por par `{"contradiccion": true|false}` (con `motivo` opcional); un modelo de decisión devuelve una probabilidad y se aplica un umbral (0,5).

## Permisos
- Herramientas: ninguna. Lectura y escritura de ficheros: ninguna (el código lee los documentos).

## Límites
- Ante la duda, no hay contradicción. Una respuesta que no se puede interpretar cuenta como «no contradicción» y se registra como fallo de formato.
- El verificador de citas se aplica igual, pero como las citas las copia el código, «citas válidas» es trivial y no mide al modelo.
- Cobertura del generador (documentos congelados de la evaluación): 5 de 6 semillas con sus dos líneas en un mismo par; es un techo para cualquier modelo.
- El mensaje de sistema fijo vive en `pairs.py` (`PAIR_SYSTEM_PROMPT`).

## Variantes probadas (ninguna cumple los tres criterios; verificador cerrado)
- `doc_checker_pairs_short`: `tev1` con el estado corto `A: … / B: …` y umbral 0,5 (1,0 de 6 sembradas, 0 falsos positivos).
- `doc_checker_pairs_json`: `gemma3:4b` con salida restringida `{"contradiccion": boolean}` (6 de 6 recuperadas por sesgo hacia «sí», 43 falsos positivos).
- `doc_checker_pairs_agree`: `gemma3:4b` con dos preguntas (contradicción solo si «se contradicen» = sí y «pueden ser verdaderas a la vez» = no): 3,6 de 6 y 22 falsos positivos.
En todas el mensaje de sistema vive en `pairs.py` y no se usa un `context/agents/<rol>.md` para el juicio.

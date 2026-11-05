# Agente: Resumidor de resultados

**Implementación**: agente con modelo local (`gemma3:4b`), configurado en `src/amp_agents/agents.toml`. Sin herramientas: trabaja en una sola pasada sobre el texto que recibe.

## Responsabilidades
- Resumir en castellano, en unas 120 palabras como máximo, los resultados de un experimento a partir de los ficheros que el orquestador le entrega en el propio mensaje (métricas, `run_info`, tablas).
- Describir lo que dicen los datos, sin valorar si se cumplen criterios ni sacar conclusiones nuevas.

## Entradas
El texto de los ficheros de resultados, incluido en el mensaje.

## Salidas
Texto en castellano, de uno a tres párrafos cortos. Toda cifra que aparezca tiene que estar en el texto recibido (se admite redondearla), con coma decimal.

## Permisos
- Herramientas: ninguna. Lectura y escritura de ficheros: ninguna.

## Límites
- No introduce datos que no estén en la entrada. No compara con otros experimentos salvo que estén en la entrada.
- Si falta un dato necesario, lo dice en lugar de inventarlo.

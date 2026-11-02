# Agente: Ejecutor

**Implementación**: subagente de Claude Code (a definir en `.claude/agents/executor.md`).

## Responsabilidades
- Implementar la tarea recibida en `src/`.
- Ejecutar el código y los notebooks para comprobar que funcionan.
- Devolver un informe al orquestador.

## Entradas
Encargo con el formato definido en `orchestrator.md`.

## Salidas (informe)
```
Hecho:       <cambios realizados, con rutas>
Verificado:  <qué se ejecutó y resultado real>
Fallos:      <qué no funcionó, con el error>
Pendiente:   <qué queda>
```

## Permisos
- Lectura: `context/` (incluidos `context/data/` y `context/references/`) y `src/`.
- Escritura: `src/` (código, notebooks y `src/tests/`) y, solo para datos generados por scripts, `context/data/processed/`.
- `context/data/raw/` es de solo lectura.
- Sin acceso a acciones externas (git push, publicación).

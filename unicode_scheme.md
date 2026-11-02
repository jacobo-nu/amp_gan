# Unicode Scheme — Arquitectura del proyecto

> Versión 0.1 · Arquitectura mínima: 1 orquestador + 1 agente ejecutor.
> Este documento se actualiza cada vez que cambie la estructura.

---

## 1. Estructura de directorios

```
Trabajo_agentes/                 ← raíz del repositorio (GitHub)
│
├── README.md                    · presentación y puesta en marcha
├── unicode_scheme.md            · este documento (mapa de la arquitectura)
├── requirements.txt             · dependencias
├── .gitignore
│
├── .claude/agents/executor.md   · registro del ejecutor (puntero a context/agents/)
│
├── context/                     ← TODO lo que define el proyecto (no es código)
│   ├── constitution/
│   │   ├── mission.md           · propósito y alcance
│   │   ├── roadmap.md           · fases, hitos y tareas pendientes
│   │   ├── governance.md        · reglas, roles, política de git
│   │   └── tech_stack.md        · lenguajes, librerías, convenciones
│   ├── agents/
│   │   ├── orchestrator.md      · rol y protocolo del orquestador
│   │   └── executor.md          · rol, entradas, salidas y permisos
│   ├── data/                    · datos locales, NO versionados
│   │   ├── raw/
│   │   └── processed/
│   └── references/              · artículos (PDF) y material de apoyo
│
├── src/                         ← TODO el código
│   ├── amp_gan/                 · paquete reutilizable (datos, modelos, métricas)
│   ├── notebooks/               · Jupyter notebooks numerados (entregables)
│   └── tests/                   · tests automáticos
```

---

## 2. Arquitectura de agentes

```
┌──────────────────────────────────────────────────────────────────┐
│                             USUARIO                              │
│              define objetivos · revisa · aprueba                 │
└────────────────────────────────┬─────────────────────────────────┘
                                 │  petición / feedback
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                  ORQUESTADOR  (Claude Code)                      │
│                                                                  │
│  1. Lee la constitución      ──────────────┐                     │
│  2. Interpreta la petición                 │                     │
│  3. Planifica y divide en tareas           │                     │
│  4. Delega al ejecutor                     │                     │
│  5. Revisa el resultado frente a governance│                     │
│  6. Informa al usuario y actualiza roadmap │                     │
└─────────────┬─────────────────────▲────────┼─────────────────────┘
              │ encargo             │ informe│ lee / actualiza
              │ (tarea + contexto)  │        │
              ▼                     │        ▼
┌───────────────────────────────┐   │   ┌──────────────────────────┐
│       AGENTE EJECUTOR         │   │   │        context/          │
│                               │   │   │                          │
│  • Recibe una tarea acotada   │   │   │  constitution/           │
│  • Escribe / modifica código  │   │   │   ├ mission.md           │
│  • Ejecuta y prueba           │───┘   │   ├ roadmap.md           │
│  • Devuelve informe:          │       │   ├ governance.md        │
│    qué hizo · qué falló       │◄──────│   └ tech_stack.md        │
│                               │ lee   │  agents/                 │
└───────────────┬───────────────┘       │   ├ orchestrator.md      │
                │ lee / escribe         │   └ executor.md          │
                ▼                       └──────────────────────────┘
┌───────────────────────────────┐
│             src/              │
│        código fuente          │
└───────────────────────────────┘
```

---

## 3. Flujo de una tarea

```
 USUARIO          ORQUESTADOR                 EJECUTOR              src/
    │                  │                          │                   │
    │── petición ─────►│                          │                   │
    │                  │── lee context/ ──┐       │                   │
    │                  │◄─────────────────┘       │                   │
    │                  │── planifica              │                   │
    │                  │── encargo ──────────────►│                   │
    │                  │                          │── lee tech_stack  │
    │                  │                          │── escribe código ►│
    │                  │                          │── ejecuta/prueba ►│
    │                  │◄──────────── informe ────│                   │
    │                  │── valida vs governance   │                   │
    │                  │   ├─ ✔ OK   → continúa   │                   │
    │                  │   └─ ✘ fallo→ reencargo ►│                   │
    │                  │── actualiza roadmap.md   │                   │
    │◄─── resultado ───│                          │                   │
    │                  │                          │                   │
```

---

## 4. Responsabilidades y permisos

```
┌──────────────┬───────────────────────────┬──────────────────────────┐
│              │  ORQUESTADOR              │  EJECUTOR                │
├──────────────┼───────────────────────────┼──────────────────────────┤
│ context/     │  lee · actualiza roadmap  │  solo lectura            │
│ src/         │  lee (revisión)           │  lee · escribe · ejecuta │
│ Decide       │  qué se hace y cuándo     │  cómo se implementa      │
│ Habla con    │  usuario y ejecutor       │  solo con orquestador    │
└──────────────┴───────────────────────────┴──────────────────────────┘
```

---

## 5. Leyenda

```
  ──►  flujo de información / acción
  ◄──  respuesta / lectura
  ┌─┐  componente (agente, directorio o actor)
  ✔ ✘  resultado de la validación
```

---

## 6. Evolución prevista

```
 v0.1 (actual)              v0.x (futuro)
 ─────────────              ─────────────
 Orquestador                Orquestador
     │                          │
     └── Ejecutor               ├── Ejecutor
                                ├── Agente ...    (por definir)
                                └── Agente ...    (por definir)
```

# TODO

Tareas inmediatas. El plan por fases está en [context/constitution/roadmap.md](context/constitution/roadmap.md).

## Estado para retomar (actualizado 2026-11-06)
- **Git:** todo el trabajo está fusionado en `main` (ff-only) salvo lo que se indique. El push a `origin/main` está al día (2026-11-06). No cambiar de rama desde VS Code mientras trabaja un ejecutor.
- **Fase 3 (GAN) cerrada por decisión del usuario el 2026-11-06: no se prueban más variantes.** Hecho en Fase 3: WGAN-GP (notebook 03), condicional por longitud (04, criterios A y B cumplidos), condicional por hidrofobicidad (05: criterio A cumplido con el A.2 revisado después de ver el resultado, B no cumplido) y agente ejecutor con modelo local (`src/amp_agents/`), más cinco roles locales evaluados (ver abajo). 541 tests.
- **Modelos en Ollama** (los tres grandes anteriores se borraron el 2026-11-05): `qwen3:4b` (ejecutor, herramientas), `llama3.2:3b` (herramientas), `phi4-mini` (herramientas), `qwen2.5-coder:7b` (herramientas; se pasa un 9 % a CPU en la GPU de 6 GB), `gemma3:4b` (sin herramientas), `nomic-embed-text` (embeddings).

## Roles locales: segunda iteración hecha y fusionada en `main` (2026-11-06)
Rama `feat/local-roles-2` (3 commits, 638 tests), fusionada con `--ff-only` y publicada (push hecho el 2026-11-06). Resultados en el roadmap: ningún rol cumple todos sus criterios. Decisiones ya aplicadas: localizador retirado (`amp_agents.locate`), BM25 por defecto, corrección de línea y recorte por código, `doc_checker_ss` creado y evaluado (no cumple nada), `DATA_NOTICE` limpiado.
- [ ] **Decidir** qué hacer con `doc_checker`/`doc_checker_ss` (phi4-mini no sirve para esto): retirarlo, o probar un modelo distinto en una pasada (p. ej. `gemma3:4b`, que ya está descargado; ojo con el contexto: los dos documentos son unos 5 600 tokens)
- [ ] **Decidir** sobre el resumidor: dejarlo con recorte por código (cumple palabras e idioma; cifras 94,3 % frente a 95 %) o añadir un paso que quite las frases con cifras no respaldadas
- [ ] **Decidir** sobre el revisor de código (1,2 de 5 defectos): ¿se mantiene solo como apoyo con comprobación humana, o se retira?
- [ ] Integrar `postprocess_summary` y la línea corregida en la salida de `run.py` (hoy solo están en la evaluación)
- [ ] `bge-m3` (~1,2 GB) para el recuperador: **requiere permiso del usuario para descargarlo**; BM25 ya es el valor por defecto
- [ ] Decidir si el verificador de citas debe aceptar `ruta#línea` (alguna respuesta del modelo lo usa); cambiarlo después de ver resultados exigiría declararlo igual que el A.2 de hidrofobicidad

## Ahora
- [ ] **Notebook de entrega** (`entregas/primer_trabajo/Primer_Trabajo_Biofísica.ipynb`, fuera de git; límite 2026-11-14): reescribirlo condensado, **máximo 90 celdas**, con el selector de backend, la GAN básica y sus correcciones, la WGAN-GP y la condicional por longitud, todo bien explicado. En curso: construcción (copia de seguridad `.copia-2026-11-06.ipynb`); después, ejecución completa con salidas y revisión del usuario. Pulidos incluidos: minúscula tras «Qué se observa.», separador de miles, «Siguientes pasos» sin lo ya hecho, tiempo en CPU declarado como estimación
- [ ] **Modelos locales:** probar cuáles sirven para integrarlos como roles y dejar la arquitectura lista para futuros proyectos. Rama `feat/model-trials` (opción `--model` en `eval_roles`; prueba de `gemma3:4b` como verificador de coherencia en curso). Candidatos que caben en 6 GB: `deepseek-r1:7b` (descargado), `tev1:4b`, `lfm3:1.7b` y `bonsai-2:9b` (**no descargados: piden una versión de Ollama más reciente; actualizar Ollama lo hace el usuario**). Kimi descartado (solo existe en la nube)

## Siguiente (Fase 1 — Datos)
- [ ] Escribir el script de descarga reproducible en `src/amp_gan/`

## Entrega del trabajo
- [x] **Selección de backend según la máquina de quien ejecute el código** (hecho 4-nov-2026, `feat/backend-adaptativo`). CPU por defecto (`runtime.apply_cpu_default`: TensorFlow si está instalado, si no torch; `AMP_GAN_BACKEND` lo fuerza); opcionalmente `runtime.apply_auto_backend` / `--auto-backend` / interruptor `EVALUAR_BACKEND` del notebook 02 miden los candidatos disponibles y aplican el más rápido; sin GPU utilizable no mide y usa la CPU. Ningún script lee ya `benchmark_results.csv`. Comprobado: 184 tests (también con la GPU oculta y sin el CSV) y el notebook 02 completo en CPU sin errores.
  - Tiempo en CPU medido en esta máquina: 3,9–4,4 s/época por modelo y **57 min** el notebook 02 completo (tres modelos); con GPU y la celda opcional, ~23 min. Pendiente: documentarlo en el README.
  - Los `*_run_info.json` ya generados conservan la clave antigua `benchmark_decision`; los nuevos usan `backend_mode` y `backend_decision`.

## Pendiente de decidir
- [ ] Documentar cómo ejecutar notebooks y tests (el cwd debe estar dentro del proyecto); `pytest`, `jupyterlab`, `nbclient` e `ipykernel` ya están en `requirements.txt`
- [ ] GPU con TensorFlow: no detecta la RTX 3060 (está compilado contra CUDA 12 y el entorno tiene CUDA 13). **Aparcado por decisión del usuario (4-nov-2026)**: se usa torch en GPU y TensorFlow solo en CPU; retomar solo si hiciera falta TensorFlow con GPU
- [ ] Estabilidad: `keras-nightly` y `tensorflow 2.22.0rc0` no son versiones estables. `requirements.txt` ya fija versiones exactas, verificadas el 4-nov-2026 en un entorno limpio con Python 3.14.9 (190 tests); `keras-nightly` aporta el paquete `keras` (no instalar `keras` aparte). Riesgos: solo verificado en Python 3.14, y si la versión nightly desapareciera de PyPI habría que repetir la verificación con otra
- [ ] Biopython: ¿quitarlo de `requirements.txt`? (el lector FASTA es propio)
- [ ] `pyproject.toml` para instalar el paquete (ahora `src/tests/conftest.py` añade `src/` al path) y documentar el entorno
- [ ] Licencia del repositorio

## Hecho
- [x] Repositorio git inicializado, primer commit y publicado en GitHub (privado): `jacobo-nu/amp_gan`, rama `main`
- [x] Notebook 01 y módulo de propiedades fisicoquímicas, validados por el orquestador (49 tests)
- [x] Datos limpios 5–60 aa generados y validados por el orquestador (`make_clean_data.py`, 34 tests)
- [x] Decisiones del usuario: rango de longitud 5–60 aa en ambas bases; GAN con Keras
- [x] Estadísticas de longitud de ambas bases (`length_stats.py`, `run_length_stats.py`), validadas por el orquestador
- [x] Reorganización: `tests/` → `src/tests/`, `data/` → `context/data/`, nueva `context/references/` para PDF
- [x] Módulo de lectura y limpieza de secuencias con tests (`src/amp_gan/data.py`, `src/tests/test_data.py`), validado por el orquestador
- [x] Subagente ejecutor registrado en `.claude/agents/executor.md` (apunta a `context/agents/executor.md`)
- [x] Versión de los datos confirmada: DRAMP 5.0, descargada el 28-oct-2026
- [x] Datos copiados a `context/data/raw/` y excluidos de git
- [x] Fuente de datos definida (DRAMP) y citada en `context/data/README.md`
- [x] Esquema de arquitectura (`unicode_scheme.md`)
- [x] Estructura de directorios y archivos de la constitución
- [x] Definición de los agentes orquestador y ejecutor

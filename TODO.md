# TODO

Tareas inmediatas. El plan por fases está en [context/constitution/roadmap.md](context/constitution/roadmap.md).

## Ahora
- [x] Fase 3, WGAN-GP: hecha en `feat/wgan-gp` (4 de 5 condiciones frente a la referencia; ver roadmap). Pendiente de fusionar con `main`
- [ ] Fase 3, GAN condicional con condición numérica (longitud → hidrofobicidad → ambas, percentiles 10–90)
- [ ] Notebook 03: añadir las secciones de la GAN condicional y la comparación final de variantes

## Siguiente (Fase 1 — Datos)
- [ ] Escribir el script de descarga reproducible en `src/amp_gan/`

## Entrega del trabajo
- [x] **Selección de backend según la máquina de quien ejecute el código** (hecho 4-nov-2026, `feat/backend-adaptativo`). CPU por defecto (`runtime.apply_cpu_default`: TensorFlow si está instalado, si no torch; `AMP_GAN_BACKEND` lo fuerza); opcionalmente `runtime.apply_auto_backend` / `--auto-backend` / interruptor `EVALUAR_BACKEND` del notebook 02 miden los candidatos disponibles y aplican el más rápido; sin GPU utilizable no mide y usa la CPU. Ningún script lee ya `benchmark_results.csv`. Comprobado: 184 tests (también con la GPU oculta y sin el CSV) y el notebook 02 completo en CPU sin errores.
  - Tiempo en CPU medido en esta máquina: 3,9–4,4 s/época por modelo y **57 min** el notebook 02 completo (tres modelos); con GPU y la celda opcional, ~23 min. Pendiente: documentarlo en el README.
  - Pendiente: las WGAN-GP necesitará una penalización de gradiente por backend (TensorFlow en CPU por defecto, torch en GPU).
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

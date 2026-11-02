# TODO

Tareas inmediatas. El plan por fases está en [context/constitution/roadmap.md](context/constitution/roadmap.md).

## Ahora

## Siguiente (Fase 1 — Datos)
- [ ] Escribir el script de descarga reproducible en `src/amp_gan/`

## Pendiente de decidir
- [ ] Añadir `pytest`, `jupyter`/`nbclient` a `requirements.txt` y documentar cómo ejecutar notebooks (el cwd debe estar dentro del proyecto)
- [ ] GPU: TensorFlow no detecta la RTX 3060 (faltan drivers/CUDA); decidir si se resuelve o se entrena en CPU
- [ ] Estabilidad: `keras-nightly` y `tensorflow 2.22.0rc0` no son versiones estables; fijar versiones al estabilizar
- [ ] Biopython: ¿quitarlo de `requirements.txt`? (el lector FASTA es propio)
- [ ] `pyproject.toml` para instalar el paquete (ahora `src/tests/conftest.py` añade `src/` al path) y documentar el entorno
- [ ] Fijar versiones en `requirements.txt`
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

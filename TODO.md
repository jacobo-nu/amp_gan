# TODO

Tareas inmediatas. El plan por fases está en [context/constitution/roadmap.md](context/constitution/roadmap.md).

## Ahora
- [ ] Selector de backend (ver «Entrega del trabajo»): CPU por defecto y celda opcional de prueba de rendimiento; se hace antes de la Fase 3, en `feat/backend-adaptativo`
- [ ] Fijar versiones en `requirements.txt` tras verificarlo en un entorno limpio
- [ ] Fase 3: WGAN-GP y condicional numérica (longitud → hidrofobicidad → ambas, percentiles 10–90)

## Siguiente (Fase 1 — Datos)
- [ ] Escribir el script de descarga reproducible en `src/amp_gan/`

## Entrega del trabajo
- [ ] **Selección de backend según la máquina de quien ejecute el código.** El trabajo se entregará a un evaluador que probablemente no tendrá GPU. **Decisión del usuario (4-nov-2026):** la opción por defecto es **CPU**; antes de entrenar hay una celda *opcional* que evalúa qué backend/dispositivo es más rápido y, si el usuario la ejecuta, se usa ese en lugar de la CPU (indicado en el notebook como opcional, para quien no tenga gráfica). Estado actual: el notebook 02 ejecuta siempre la prueba en memoria, y `run_gan_base.py` exige `benchmark_results.csv` (generado a mano con `python src/amp_gan/benchmark.py`, no versionado, inexistente en un clon limpio). Hay que:
  - cambiar el valor por defecto a CPU en notebooks y scripts, sin depender del CSV; la prueba de rendimiento pasa a ser una celda opcional (y una opción explícita en los scripts, p. ej. `--auto-backend`) cuyo resultado se aplica solo si se ejecuta;
  - esa prueba debe detectar qué backends están instalados y qué dispositivos hay (con y sin GPU) y probar solo los disponibles, sin fallar si falta torch, TensorFlow o la GPU;
  - no fiarse de un `benchmark_results.csv` generado en otra máquina (regenerar o ignorar si no coincide el hardware);
  - `requirements.txt` ya incluye `torch`, `scipy` y `tensorflow` y no `biopython` (hecho 4-nov-2026); falta documentar que sin GPU el entrenamiento completo es más lento (~3 s/época por CPU frente a ~1,9 s en GPU);
  - revisar que los notebooks se ejecuten de principio a fin en un entorno solo con CPU (con el valor por defecto).

## Pendiente de decidir
- [ ] Documentar cómo ejecutar notebooks y tests (el cwd debe estar dentro del proyecto); `pytest`, `jupyterlab`, `nbclient` e `ipykernel` ya están en `requirements.txt`
- [ ] GPU con TensorFlow: no detecta la RTX 3060 (está compilado contra CUDA 12 y el entorno tiene CUDA 13). **Aparcado por decisión del usuario (4-nov-2026)**: se usa torch en GPU y TensorFlow solo en CPU; retomar solo si hiciera falta TensorFlow con GPU
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

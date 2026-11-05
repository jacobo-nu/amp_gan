# TODO

Tareas inmediatas. El plan por fases está en [context/constitution/roadmap.md](context/constitution/roadmap.md).

## Estado para retomar (2026-11-05 18:07 CET)
- **Git:** todo el trabajo está fusionado en `main` (ff-only) salvo lo que se indique. El último `git push origin main` del usuario falló a las 16:00 UTC con `remote: Internal Server Error` de GitHub (fallo del servidor, el repositorio local pasa `git fsck`); hay que **reintentar `git push origin main`**. No cambiar de rama desde VS Code mientras trabaja un ejecutor.
- **Hecho en Fase 3:** WGAN-GP (notebook 03), condicional por longitud (04, criterios A y B cumplidos), condicional por hidrofobicidad (05: criterio A cumplido con el A.2 revisado después de ver el resultado, B no cumplido) y agente ejecutor con modelo local (`src/amp_agents/`), más cinco roles locales evaluados (ver abajo). 541 tests.
- **Modelos en Ollama** (los tres grandes anteriores se borraron el 2026-11-05): `qwen3:4b` (ejecutor, herramientas), `llama3.2:3b` (herramientas), `phi4-mini` (herramientas), `qwen2.5-coder:7b` (herramientas; se pasa un 9 % a CPU en la GPU de 6 GB), `gemma3:4b` (sin herramientas), `nomic-embed-text` (embeddings).

## Propuestas pendientes de decisión (2026-11-05 18:07 CET)
Contexto: la evaluación en vivo de los cinco roles locales (criterios fijados de antemano, resultados en el roadmap) no cumplió ninguno; el resumidor y el revisor con corrección de línea son los candidatos útiles. Segunda iteración propuesta, en una rama nueva (p. ej. `feat/local-roles-2`), con **los mismos criterios** y el mismo protocolo (`src/amp_agents/eval_roles.py`):
- [ ] **1. Localizador (`llama3.2:3b`, 10 % de aciertos):** sustituirlo por una herramienta determinista (lo que hace es `grep`); el modelo solo lo empeora. Decidir si el rol se elimina o se reasigna.
- [ ] **2. Verificador de coherencia (`phi4-mini`, 0 % de citas válidas; no llama a herramientas e inventa citas):** probarlo en modo `single_shot`, con los dos documentos dentro del mensaje y sin herramientas.
- [ ] **3. Revisor de código (`qwen2.5-coder:7b`, 1,0 de 5 defectos; 1,7 con un ajuste):** añadir un verificador que reasigne la línea buscando la cita literal en el fichero (el modelo acierta el texto pero desplaza el número); no detectó nunca los defectos de validación ni de argumentos intercambiados.
- [ ] **4. Resumidor (`gemma3:4b`, cifras 97,8 % e idioma bien, pero hasta 246 palabras frente a 150):** recortar por código a 150 palabras o cambiar la definición («unas 120 palabras» no se cumple); usa punto decimal en lugar de coma.
- [ ] **5. Recuperador (`nomic-embed-text`, 60 % top-3 frente a 73 % de BM25):** usar BM25 como recuperador por defecto y dejar los embeddings como opción. Probar un modelo de embeddings mejor para castellano, `bge-m3` (~1,2 GB), **requiere permiso del usuario para descargarlo** (pendiente).
- [ ] Más repeticiones (ahora 2–3) para que las diferencias entre ajustes sean concluyentes.
- [ ] Limpieza señalada por el ejecutor: el aviso fijo `DATA_NOTICE` de `agents.py` menciona un apartado «Fallos» que no existe en los roles de salida JSON.

## Ahora
- [ ] Fase 3, GAN condicional por longitud e hidrofobicidad a la vez (percentiles 10–90). Plan: conservar los criterios de longitud, usar el A.2 revisado (`conditions.mae_vs_reference`) para las dos propiedades y fijar todos los criterios antes de entrenar; comparar con las dos condicionales y con la referencia `gan_stabc`
- [ ] Notebook nuevo: la condicional por ambas y la comparación final de variantes (los notebooks 03 WGAN-GP, 04 longitud y 05 hidrofobicidad ya están hechos, uno por experimento)
- [ ] Trabajo de entrega (`entregas/primer_trabajo/Primer_Trabajo_Biofísica.ipynb`, fuera de git): el usuario lo revisa a mano; límite 2026-11-14, objetivo entregar esta semana; pulidos pendientes: minúscula tras «Qué se observa.», separador de miles, «Siguientes pasos» menciona cosas ya hechas, tiempo en CPU solo estimado

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

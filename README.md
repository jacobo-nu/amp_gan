# amp-gan

Generación de péptidos antimicrobianos (AMP) con redes generativas adversarias (GAN),
desarrollada con una arquitectura de agentes (orquestador + ejecutor).

Proyecto académico de asignatura de Biofísica del grado de Física de la USC. Uso exclusivamente educativo y de investigación abierta.

## Estructura

| Directorio | Contenido |
|---|---|
| `context/constitution/` | Misión, roadmap, gobernanza y stack técnico |
| `context/agents/` | Definición de cada agente y resumen de resultados con modelos locales (`resultados_modelos_locales.md`) |
| `context/data/` | Datos locales (`raw/`, `processed/`), no versionados |
| `context/references/` | Artículos y material de referencia (PDF) para los agentes |
| `src/amp_gan/` | Código reutilizable (datos, modelos, entrenamiento, evaluación) |
| `src/notebooks/` | Jupyter notebooks (entregables) |
| `src/tests/` | Tests automáticos |

Esquema completo: [unicode_scheme.md](unicode_scheme.md)

## Datos y citas

Las secuencias proceden de [DRAMP](http://dramp.cpu-bioinfor.org/). Origen, archivos y referencias
completas en [context/data/README.md](context/data/README.md). Si usas este repositorio, cita DRAMP
(Ma et al., *Nucleic Acids Res.* 2026, gkag989, PMID 42876369).

## Puesta en marcha

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/amp_gan/download_data.py --clean   # descarga DRAMP 5.0 (con comprobación SHA-256) y genera las bases limpias
jupyter lab src/notebooks
```

## Licencia

El código y la documentación de este repositorio se distribuyen bajo la
[licencia GNU GPL v3](LICENSE) (GPL-3.0). Los datos de DRAMP no se incluyen en el repositorio y
tienen su propia licencia (CC BY 4.0; véase [context/data/README.md](context/data/README.md)).

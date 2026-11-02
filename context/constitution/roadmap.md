# Roadmap

Estado: `[ ]` pendiente · `[~]` en curso · `[x]` hecho

## Fase 0 — Fundamentos
- [x] Esquema de arquitectura (`unicode_scheme.md`)
- [~] Estructura de repositorio y constitución
- [ ] Repositorio git inicializado y enlazado a GitHub

## Fase 1 — Datos
- [x] Fuente de datos: DRAMP (`general_amps.fasta`, `Antibacterial_amps.fasta`), citada en `context/data/README.md`
- [x] Datos copiados a `context/data/raw/`, excluidos de git
- [x] Versión confirmada: DRAMP 5.0, descargada el 28-oct-2026
- [ ] Script de descarga reproducible
- [x] Limpieza: duplicados, secuencias con aminoácidos no estándar, filtrado por longitud (`src/amp_gan/data.py`, 23 tests; rango definitivo 5–60 aa, a aplicar en las dos bases)
- [x] Estadísticas de longitud de ambas bases (`src/amp_gan/length_stats.py`; salidas en `context/data/processed/`); corte fijado por el usuario: 5–60 aa
- [x] Datos limpios (5–60 aa) en `context/data/processed/`: 21 754 antibacteriano, 9 391 general (`src/amp_gan/make_clean_data.py`)
- [x] Notebook 01: exploración de datos (`src/notebooks/01_exploracion_datos.ipynb`; propiedades en `src/amp_gan/properties.py`, 49 tests)

## Fase 2 — Modelo base
- [ ] Codificación de secuencias (one-hot / embeddings) y padding
- [ ] GAN base en Keras
- [ ] Notebook 02: entrenamiento de la GAN base

## Fase 3 — Mejoras
- [ ] WGAN-GP
- [ ] GAN condicional
- [ ] Notebook 03: comparación de variantes

## Fase 4 — Evaluación
- [ ] Métricas fisicoquímicas y de diversidad/novedad
- [ ] Notebook 04: evaluación y conclusiones

## Fase 5 — Ampliación de la arquitectura
- [ ] Añadir más agentes (p. ej. revisor, documentación)

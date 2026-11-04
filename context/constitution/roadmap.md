# Roadmap

Estado: `[ ]` pendiente · `[~]` en curso · `[x]` hecho

## Fase 0 — Fundamentos
- [x] Esquema de arquitectura (`unicode_scheme.md`)
- [~] Estructura de repositorio y constitución
- [x] Repositorio git inicializado y enlazado a GitHub (privado, `jacobo-nu/amp_gan`)

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
- [x] Codificación de secuencias: one-hot de 21 canales (20 aa + PAD), longitud fija 60 (`src/amp_gan/encoding.py`, 77 tests). Embeddings quedan como posible variante en la Fase 3
- [x] GAN base en Keras (vanilla, MLP, z=100; `src/amp_gan/gan.py`). Resultado: colapsa a pocas secuencias (línea base «vanilla pura»)
- [x] Prueba de rendimiento de backends y dispositivos (`src/amp_gan/benchmark.py`): gana torch en GPU
- [x] Notebook 02: entrenamiento y evaluación de la GAN base, con la prueba de rendimiento
- [x] GAN vanilla estabilizada (etiqueta suavizada 0,9, ruido de instancia 0,1 con decaimiento, Gumbel-softmax straight-through) en el notebook 02. Ya no colapsa (2000 secuencias distintas), pero cumple solo 4 de los 5 criterios fijados de antemano: sesga la longitud (media 37,6 aa frente a 21,1; W1 de longitud 16,6 frente a 9,1 de la pura)
- [x] Ablación con ruido de instancia constante (sección h del notebook 02, `--variant stabilized_const`): W1 de longitud 3,85 y longitud media 19,0 aa, sin deriva (criterio fijado de antemano cumplido: ≤ 8,3 y cociente época 200/100 ≤ 1,25). Respalda que el decaimiento del ruido causaba la deriva; una sola ejecución, semilla 42. **Será la referencia de la Fase 3** (GAN vanilla estabilizada con ruido constante)

## Fase 3 — Mejoras
- [ ] WGAN-GP
- [ ] GAN condicional con condición numérica: primero longitud, luego hidrofobicidad y después las dos. Rangos «habituales» = percentiles 10–90 de Antibacterial_amps (limpia). Se construye sobre la vanilla estabilizada con ruido constante y se compara con ella
- [ ] Notebook 03: comparación de variantes

## Fase 4 — Evaluación
- [ ] Métricas fisicoquímicas y de diversidad/novedad
- [ ] Notebook 04: evaluación y conclusiones

## Fase 5 — Ampliación de la arquitectura
- [ ] Añadir más agentes (p. ej. revisor, documentación)
- [ ] Agente con modelo local, para que la arquitectura no dependa de un proveedor: bucle propio en `src/` con herramientas acotadas y un adaptador por proveedor (servidor local compatible con la API de OpenAI). Las definiciones siguen en `context/agents/`. Roles candidatos de solo lectura: documentación, revisor de coherencia, resumen de experimentos. Limitación: GPU de 6 GB compartida con el entrenamiento

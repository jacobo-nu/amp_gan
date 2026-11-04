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
- [x] Selector de backend adaptativo: CPU por defecto y prueba de rendimiento opcional que elige el backend/dispositivo más rápido disponible (`src/amp_gan/runtime.py`, 184 tests)
- [x] GAN vanilla estabilizada (etiqueta suavizada 0,9, ruido de instancia 0,1 con decaimiento, Gumbel-softmax straight-through) en el notebook 02. Ya no colapsa (2000 secuencias distintas), pero cumple solo 4 de los 5 criterios fijados de antemano: sesga la longitud (media 37,6 aa frente a 21,1; W1 de longitud 16,6 frente a 9,1 de la pura)
- [x] Ablación con ruido de instancia constante (sección h del notebook 02, `--variant stabilized_const`): W1 de longitud 3,85 y longitud media 19,0 aa, sin deriva (criterio fijado de antemano cumplido: ≤ 8,3 y cociente época 200/100 ≤ 1,25). Respalda que el decaimiento del ruido causaba la deriva; una sola ejecución, semilla 42. **Será la referencia de la Fase 3** (GAN vanilla estabilizada con ruido constante)

## Fase 3 — Mejoras
- [x] WGAN-GP (`src/amp_gan/wgan_gp.py`, penalización de gradiente para TensorFlow y torch; notebook 03, sección a). Una ejecución, semilla 42, época 200, modo sample: frente a la referencia (estabilizada con ruido constante) mejora la W1 de longitud (3,85 → 1,15), de carga (1,23 → 0,45) y la composición (0,097 → 0,074), pero empeora ligeramente la de hidrofobicidad (0,085 → 0,095); cumple 4 de las 5 condiciones fijadas de antemano, así que no se declara mejor. Aun así queda más cerca de los datos reales que la referencia en 3 de 4 métricas, y entrena más rápido por época que la vanilla en CPU
- [x] GAN condicional con condición numérica, **longitud** (`src/amp_gan/conditional_gan.py`, notebook 03 sección b), sobre la estabilizada con ruido constante. Una ejecución, semilla 42, época 200, modo sample: controla la longitud pedida (pendiente 0,91, error medio 1,17 aa; acierto ±2 aa de 0,98 en 9, 20 y 21 aa, 0,90 en 13 aa y 0,47 en 40 aa) y supera a la referencia sin condicionar en los criterios A y B fijados de antemano; con condiciones reales mejora además la W1 de carga (1,23 → 0,45), de hidrofobicidad (0,085 → 0,035) y la composición (0,097 → 0,050). Limitaciones: se queda corta en el objetivo p90 (40 aa) y repite más secuencias en argmax
- [x] GAN condicional con condición numérica, **hidrofobicidad** (variante `cgan_hydro`, notebook 03 sección c; evaluación por objetivos generalizada a una propiedad en `conditions.py`). Una ejecución, semilla 42, época 200, modo sample, objetivos = GRAVY p10/30/50/70/90, acierto ±0,25: **criterios A y B no cumplidos** (A: 2 de 3; B: 1 de 2). Pendiente de la recta 0,81 (cumple por poco), tasa de acierto superior a la referencia en los 5 objetivos (0,27–0,30 frente a 0,07–0,26; los extremos superan además el techo real), pero error medio 0,56 (se pedía ≤ 0,30), W1 de hidrofobicidad 0,125 (se pedía ≤ 0,085) y `unique_fraction` 1,0. El umbral de error ≤ 0,30 era inalcanzable por construcción: la desviación del GRAVY generado en cada objetivo es ≈ 0,7, lo que da un error medio ≈ 0,55 aunque la media fuera exacta (criterio mal calibrado; se documenta sin reajustarlo). Empeora a la referencia en W1 de longitud (4,51 frente a 3,85) y composición; en argmax se pasa de rosca (pendiente 1,77). TensorFlow/CPU da los mismos veredictos (W1 hidrofobicidad 0,170). La W1 oscila entre instantáneas (0,05–0,18)
- [ ] GAN condicional con **longitud e hidrofobicidad a la vez**, comparada con las dos anteriores y con la referencia
- [ ] Notebook 03: comparación de variantes

## Fase 4 — Evaluación
- [ ] Métricas fisicoquímicas y de diversidad/novedad
- [ ] Notebook 04: evaluación y conclusiones

## Fase 5 — Ampliación de la arquitectura
- [ ] Añadir más agentes (p. ej. revisor, documentación)
- [ ] Agente con modelo local, para que la arquitectura no dependa de un proveedor: bucle propio en `src/` con herramientas acotadas y un adaptador por proveedor (servidor local compatible con la API de OpenAI). Las definiciones siguen en `context/agents/`. Roles candidatos de solo lectura: documentación, revisor de coherencia, resumen de experimentos. Limitación: GPU de 6 GB compartida con el entrenamiento

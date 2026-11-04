# Stack técnico

| Área | Elección |
|---|---|
| Lenguaje | Python 3.11+ |
| Deep learning | Keras 3; backend y dispositivo = la configuración más rápida según la prueba de rendimiento (`src/amp_gan/benchmark.py`, resultados en `context/data/processed/benchmark_results.csv`). Resultado actual (3-nov-2026): **backend torch en GPU** |
| Datos | pandas, numpy, scipy |
| Métricas | scikit-learn |
| Visualización | matplotlib |
| Notebooks | JupyterLab |
| Tests | pytest |
| Control de versiones | git + GitHub |

## Convenciones
- Código en `src/amp_gan/`, instalable como paquete; notebooks en `src/notebooks/`, tests en `src/tests/`.
- Notebooks numerados: `NN_nombre.ipynb`.
- Nombres de módulos y funciones en inglés (`snake_case`); documentación en español.
- Semilla global por defecto: `42` (`keras.utils.set_random_seed(42)`).
- Rango de longitud de las secuencias de entrenamiento: **5–60 aa** (inclusivo), igual en las dos bases.
- El entorno actual usa `keras-nightly`, `tensorflow 2.22.0rc0` y `torch 2.15.0+cu132` (versiones no estables); registrar las versiones en cada experimento. TensorFlow no puede usar la GPU porque está compilado contra CUDA 12 y el entorno solo tiene CUDA 13; con torch en GPU una época de la GAN base (lote 128) tarda ~1,9 s frente a ~2,9 s de TensorFlow en CPU. El código de los modelos usa solo la API pública de Keras para ser independiente del backend; la penalización de gradiente de WGAN-GP será específica del backend ganador.
- Antes de importar Keras, aplicar la configuración con `amp_gan.runtime.apply_fastest(...)`.
- Dependencias fijadas en `requirements.txt` cuando el proyecto se estabilice.

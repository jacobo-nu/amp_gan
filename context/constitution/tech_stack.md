# Stack técnico

| Área | Elección |
|---|---|
| Lenguaje | Python 3.11+ |
| Deep learning | Keras 3 (backend TensorFlow) |
| Datos | pandas, numpy, Biopython |
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
- El entorno actual usa `keras-nightly` y `tensorflow 2.22.0rc0` (versiones no estables); registrar las versiones en cada experimento. TensorFlow no detecta la GPU (sin drivers CUDA): se entrena en CPU salvo que se resuelva.
- Dependencias fijadas en `requirements.txt` cuando el proyecto se estabilice.

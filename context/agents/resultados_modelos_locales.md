# Modelos locales para tareas acotadas: qué se probó y qué resultó

Resumen, pensado para leerse de principio a fin, de todo lo que se ha probado con modelos de lenguaje locales en este proyecto: qué hace cada rol, qué modelos se probaron y por qué, qué se midió y qué salió. Está escrito para que sirva de punto de partida en futuros proyectos.

Es una guía de lectura: las definiciones de cada rol están en [context/agents/](.) (un fichero por rol), la configuración en `src/amp_agents/agents.toml` y el historial cronológico, con más detalle, en [roadmap.md](../constitution/roadmap.md). Las cifras de este documento se contrastaron con las tablas de resultados guardadas de cada evaluación (fecha: 2026-11-07).

## 1. Cómo se probó

**Equipo.** GPU de 6 GB (RTX 3060), servidor Ollama 0.40.15. Solo caben modelos de hasta unos 4,5 GB en GPU; por eso los modelos probados son de 1,7 a 9 mil millones de parámetros, cuantizados.

**Protocolo.** Para cada rol se fijaron antes de medir una tarea, unos datos de prueba y unos criterios numéricos. No se han cambiado después. Si un resultado sorprendía, se anotaba como observación, no se «arreglaba» el criterio.

**Datos de prueba.** Se parte de copias congeladas de documentos reales del proyecto (`src/amp_agents/fixtures/`) o de un fichero de código. Se les siembran errores conocidos, de modo que se sabe cuántos hay que encontrar, y se usa también una copia limpia, para contar falsos positivos.

**Dos avisos para interpretar los números.**
- Con muestras pequeñas (entre 6 y 48 casos por prueba) las diferencias de uno o dos casos no son concluyentes. Además, se han hecho muchas comparaciones sobre los mismos datos, así que elegir «el mejor» de todas ellas tiene riesgo de ser casualidad.
- Un modelo que responde «sí» a todo recupera todos los defectos sembrados y a la vez genera muchos falsos positivos. Por eso el criterio de recuperación nunca debe leerse solo.

**Dónde está el código y cómo se repite una medición** (desde `src/`, con Ollama en marcha y los modelos descargados):

```
python -m amp_agents.eval_roles <rol> --model <modelo> --reps 5 --workdir <carpeta vacía>
```

Los roles disponibles son `doc_checker`, `doc_checker_ss`, `doc_checker_pairs`, `code_reviewer`, `code_reviewer_funcs`, `summarizer`, `retriever`, `retriever_hybrid`, `query_rewriter`, `router` y `locator` (este último solo para reproducir el resultado negativo). Los resultados detallados por ejecución no se guardan en el repositorio, solo las tablas de resumen que están aquí y en el roadmap.

## 2. Panorama general

| Rol | Qué hace | Modelo elegido | ¿Cumple sus criterios? | Decisión |
|---|---|---|---|---|
| Enrutador | Elige qué rol atiende una petición | `gemma3:4b` | Sí (con la salvedad de los datos de prueba) | Candidato a integrar |
| Resumidor | Resume resultados de experimentos | `gemma3:4b` | Solo con recorte y filtro por código | Se usa, con el posprocesado |
| Recuperador | Busca fragmentos de documentación | BM25 (sin modelo) | BM25: sí; con embeddings: no | BM25 por defecto |
| Revisor de código | Señala posibles errores en un fichero | `qwen2.5-coder:7b` | No | Solo como apoyo con revisión humana |
| Verificador de coherencia | Busca contradicciones entre documentos | `phi4-mini` / `gemma3:4b` | No, con ninguna variante | Cerrado |
| Reescritor de consultas | Añade palabras clave a una búsqueda | `phi4-mini` / `gemma3:4b` | No | No adoptado |
| Localizador de código | Dice dónde se define algo | `llama3.2:3b` | No | Retirado (lo hace una función normal) |
| Ejecutor | Agente con herramientas que edita código | `qwen3:4b` | Parcialmente | Solo lectura o supervisado |

Cuatro lecciones, que se repiten en casi todos los roles:
1. **Con 3–7 mil millones de parámetros, lo que funciona es lo más acotado**: una pregunta cerrada (elegir una de seis opciones, decir sí o no), con la respuesta restringida a un formato fijo.
2. **Muchas veces el mérito es del código que rodea al modelo**, no del modelo: recortar el texto, quitar frases con cifras inventadas, construir las citas o corregir los números de línea. Cuando es así, el documento lo dice.
3. **Los modelos pequeños son muy sensibles al formato del mensaje.** El mismo modelo puede dar 0,4 o 0,9 de probabilidad según cómo se presente la misma pareja de datos.
4. **Los modelos de razonamiento (`deepseek-r1`, `qwen3`) no compensan en estas tareas**: son lentos y gastan los tokens pensando.

## 3. Modelos probados

| Modelo | Tamaño aproximado | Para qué se usó | Impresión general |
|---|---|---|---|
| `gemma3:4b` | 4 mil millones | Resumen, enrutador, revisor por función, verificador (varias variantes), reescritor | El más versátil: buen castellano y buen enrutador, pero tiende a decir «sí» y es poco fiable como verificador. |
| `phi4-mini` | 3,8 mil millones | Verificador, resumen, revisor por función, enrutador, reescritor | Conservador: pocos falsos positivos y pocas detecciones. Bueno siguiendo formatos. |
| `llama3.2:3b` | 3 mil millones | Localizador, resumen, enrutador, reescritor | Rápido, pero el más flojo en cifras y en cumplir longitud. |
| `qwen2.5-coder:7b` | 7 mil millones | Revisor de código | Con señal real, aunque insuficiente. Casi sin falsos positivos. |
| `qwen3:4b` | 4 mil millones | Ejecutor, verificador | Con razonamiento: muy lento y con muchas respuestas vacías. |
| `deepseek-r1:7b` | 7 mil millones | Verificador, revisor | Con razonamiento: lento, no compensa. |
| `tev1:4b` | 4 mil millones | Verificador, enrutador | Modelo de decisión: devuelve probabilidades. Muy sensible al formato. |
| `lfm3:1.7b` | 1,7 mil millones | Verificador, enrutador | Muy pequeño. Se queda justo por debajo en el enrutador. |
| `bonsai-2:9b` | 9 mil millones | Verificador | Muchos cortes de formato. |
| `nomic-embed-text` | Embeddings | Recuperador | No supera a BM25. |
| `bge-m3` | Embeddings (1,2 GB) | Recuperador | No supera a BM25; cabe entero en GPU. |

## 4. Resultados por rol

### 4.1 Enrutador de peticiones — el rol que mejor salió

**Qué hace.** Recibe una petición en castellano («¿dónde se define `postprocess_summary`?») y elige cuál de seis roles debería atenderla: `locator`, `summarizer`, `code_reviewer`, `retriever`, `doc_checker` o `executor`. No ejecuta nada, solo propone el rol. La respuesta se restringe a un único valor de una lista.

**Por qué estos modelos.** Es la tarea más sencilla (clasificar), la que más se parece a lo que hacen bien los modelos pequeños. Se probaron cinco candidatos pequeños.

**Prueba.** 48 peticiones (8 por rol). Criterios: exactitud global ≥ 85 % y al menos un 60 % de acierto en cada rol.

| Modelo | Exactitud | Peor rol | ¿Cumple? |
|---|---|---|---|
| `gemma3:4b` | 97,9 % (47/48) | locator 7/8 | Sí |
| `tev1:4b` | 95,8 % (46/48) | locator y summarizer 7/8 | Sí |
| `llama3.2:3b` | 85,4 % (41/48) | summarizer 5/8 | Sí, por una petición |
| `phi4-mini` | 85,4 % (41/48) | executor 5/8 | Sí, por una petición |
| `lfm3:1.7b` | 83,3 % (40/48) | doc_checker y executor 6/8 | No |

**Salvedad importante.** Las 48 peticiones las escribió quien diseñó el experimento a partir de las definiciones de los roles, así que puede haber un sesgo a favor de los modelos que «entienden» esas definiciones. El error más repetido es mandar peticiones de otros roles al recuperador. **Pendiente:** contrastar con un conjunto de peticiones escrito por otra persona.

### 4.2 Resumidor de resultados

**Qué hace.** Resume en castellano, en unas 120 palabras, los ficheros de métricas de un experimento, sin añadir datos. No usa herramientas: el texto va en el mensaje.

**Por qué estos modelos.** `gemma3:4b` escribe bien en castellano; `phi4-mini` y `llama3.2:3b` se probaron como alternativas.

**Prueba.** Dos entradas (métricas y datos de ejecución de dos experimentos), 5 repeticiones cada una (10 ejecuciones). Criterios: al menos el 95 % de las cifras del resumen aparecen en la entrada; máximo 150 palabras; en castellano.

El modelo en crudo no cumple: se pasa de longitud (llegó a escribir 318 palabras con `gemma3:4b` y 463 con `phi4-mini`), usa punto decimal y se inventa alguna cifra. Se añadió un posprocesado por código: recortar en el último final de frase que cabe en 150 palabras, cambiar el punto decimal por coma y, además, **quitar las frases con cifras que no estén en la entrada**.

| Modelo | Cifras respaldadas (crudo) | Palabras máximas (crudo) | Con recorte + filtro |
|---|---|---|---|
| `gemma3:4b` | 93,7 % (95,8 % en una medición previa) | 318 | Cumple todo: 100 % de cifras, máx. 145 palabras |
| `phi4-mini` | 88,3 % | 463 | Cumple todo: 100 % de cifras, máx. 149 palabras |
| `llama3.2:3b` | 78,6 % | 306 | No cumple: una ejecución queda en 24 palabras tras filtrar |

**Lectura.** Que se cumplan los criterios se debe al filtro del código, no al modelo. Aun así, el modelo aporta la redacción en castellano, y el filtro garantiza que no queden cifras inventadas. La interfaz de línea de comandos (`run.py`) aplica el recorte y el filtro por defecto; `--raw` muestra la salida tal como la dio el modelo. Con el recorte solo (sin el filtro) las cifras bajan por debajo del umbral (94,3 % en una medición con `gemma3:4b`), por eso se añadió el filtro.

### 4.3 Recuperador de documentación

**Qué hace.** Dada una pregunta, devuelve los fragmentos de `context/` y `src/` más relacionados, con ruta y línea. No genera texto.

**Qué se comparó.** BM25 (búsqueda léxica clásica, sin modelo), embeddings (modelo que convierte texto en vectores y busca por parecido de significado) y combinaciones de ambos por fusión de rangos (RRF).

**Prueba.** 15 preguntas con respuesta conocida, sobre un corpus de 970 fragmentos. Se mide el acierto en las tres primeras posiciones. Criterio para los embeddings: ≥ 80 %.

| Variante | Acierto top-3 | Acierto top-1 |
|---|---|---|
| BM25 (sin modelo) | 80 % (12/15) | 8/15 |
| BM25 + `bge-m3` | 73 % (11/15) | 10/15 |
| BM25 + `nomic-embed-text` | 67 % (10/15) | 9/15 |
| `bge-m3` solo | 60 % (9/15) | 6/15 |
| `nomic-embed-text` solo | 53 % (8/15) | — |

**Lectura.** Ninguna variante con embeddings supera a BM25 ni corrige ninguna de sus tres preguntas falladas. Los híbridos solo mejoran el acierto en primer lugar. **Por eso BM25 es el recuperador por defecto**; los embeddings y el híbrido se pueden activar explícitamente. Las medidas antiguas (BM25 73 %, `nomic` 60 %) se hicieron con un corpus más pequeño y no son comparables.

### 4.4 Revisor de código

**Qué hace.** Revisa un fichero de Python y señala posibles errores de programación (índices desplazados, comparaciones invertidas, argumentos intercambiados, validaciones que faltan, casos límite sin tratar), citando la línea literal. Sin comentarios de estilo.

**Por qué este modelo.** `qwen2.5-coder:7b` está especializado en código y es el mayor que cabe.

**Prueba.** Una copia de `src/amp_gan/properties.py` con 5 defectos sembrados de tipos distintos, más la copia limpia, 5 repeticiones. Criterios: ≥ 3 defectos recuperados de media, ≥ 90 % de citas válidas, ≤ 2 falsos positivos en la copia limpia.

Dos formas de usarlo:
- **Fichero entero** (el modelo lee el fichero con herramientas). `qwen2.5-coder:7b`: 1,0 de 5 con la línea que da el modelo y 1,2 de 5 con la línea corregida por código (el modelo cita bien el texto pero desplaza el número de línea). Cumple citas válidas (100 %) y falsos positivos (máx. 2), no cumple la recuperación.
- **Por función y tipo de defecto**. El código divide el fichero en funciones y hace una pregunta cerrada por función y por categoría, con respuesta restringida a sí/no. Las citas las pone el código.

| Modelo | Defectos de 5 | Falsos positivos (máx. por repetición) | Lectura |
|---|---|---|---|
| `qwen2.5-coder:7b` | 1,8 | 1 | Señal real (detecta validación, límite e índice solo en la copia sembrada), pero nunca detecta signo ni argumentos intercambiados |
| `gemma3:4b` | 3,0 | 8 | Cumple la recuperación por decir «sí» casi siempre; 8 falsos positivos por repetición |
| `phi4-mini` | 0,0 | 2 | No detecta nada |

`deepseek-r1:7b` y `qwen3:4b` quedaron fuera por lentos y por gastar los tokens en razonar.

**Lectura.** Ningún modelo cumple los tres criterios a la vez. Sirve solo como apoyo cuyas indicaciones revisa una persona. La corrección de línea por código está integrada en la interfaz de línea de comandos.

### 4.5 Verificador de coherencia documental — el que no funcionó

**Qué hace.** Compara dos documentos (por ejemplo `roadmap.md` y `TODO.md`) y señala contradicciones: cifras distintas, ficheros que no existen, estados distintos (hecho / pendiente) para la misma tarea.

**Prueba.** Los dos documentos reales congelados, con 6 contradicciones sembradas de tipos distintos, más una copia limpia, 5 repeticiones. Criterios: ≥ 90 % de citas literales válidas, ≥ 4 de las 6 recuperadas de media, ≤ 2 falsos positivos en cada repetición.

Es la tarea que más se intentó. Se probaron cinco formulaciones:
1. **Con herramientas** (el modelo lee los ficheros): `phi4-mini` no llamaba a las herramientas e inventaba citas (0 % válidas, 0 de 6).
2. **En una pasada**, con los dos documentos enteros en el mensaje (unos 5 600 tokens): `phi4-mini` 6 % de citas válidas y 0 de 6; `gemma3:4b` 40 % y 0 de 6, con JSON inválido en 4 de 10 respuestas.
3. **Por pares**: el código genera pares de líneas candidatos (60 por documento) y el modelo juzga cada par con sí/no; el código construye las citas. El generador cubre 5 de las 6 semillas, es decir, ningún modelo puede pasar de 5 de 6. Como las citas las copia el código, el criterio de citas es trivial y no mide al modelo.
4. **Por pares con salida restringida** a `{"contradiccion": boolean}`.
5. **Por pares con dos preguntas** (contradicción solo si «se contradicen» = sí y «pueden ser verdaderas a la vez» = no).

Resultados por modelo (formulación 3, salvo que se indique):

| Modelo | Recuperadas de 6 | Falsos positivos (máx.) | Observaciones |
|---|---|---|---|
| `gemma3:4b` | 5,0 | 34 | Dice «contradicción» en el 55 % de los pares limpios |
| `gemma3:4b`, salida restringida (4) | 6,0 | 43 | Recupera todo por decir «sí» al 72 % de los pares limpios |
| `gemma3:4b`, dos preguntas (5) | 3,6 | 22 | No discrimina: 37 % de «sí» en pares limpios, frente a 32 % en los pares correctos |
| `phi4-mini` | 2,0 | 1 | Conservador; ningún fallo de formato |
| `tev1:4b` | 0,0 | 0 | Sus probabilidades nunca llegan a 0,5 (entre 0,009 y 0,366) |
| `tev1:4b`, formato corto `A: … / B: …` | 1,0 | 0 | Con otro formato de mensaje sube, pero sigue lejos |
| `deepseek-r1:7b` | 0,5 | 3 | 6,4 s por par; solo se completaron 2 de 5 repeticiones |
| `qwen3:4b` | 2,0 (una repetición suplementaria) | 3 | 22,9 s por par; 17–19 % de respuestas vacías (el razonamiento agota los tokens) |
| `lfm3:1.7b` | 0,0 | 1 | 2 % de fallos de formato |
| `bonsai-2:9b` | 0,0 | 1 | 18,7 % de fallos de formato |

**Lectura.** Con modelos de este tamaño, ninguna formulación consigue a la vez recuperar la mayoría y no inventar contradicciones: o dicen «sí» a todo, o no dicen nada. El rol se da por cerrado. La comprobación de coherencia entre documentos sigue haciéndose a mano o con herramientas deterministas.

### 4.6 Reescritor de consultas — no adoptado

**Qué hace.** Un modelo propone de 3 a 8 palabras clave que se añaden a la pregunta antes de buscar con BM25, para mejorar la búsqueda.

**Prueba.** Las mismas 15 preguntas del recuperador. Criterio: acertar en el top-3 al menos igual que BM25 simple más 2 consultas, y empeorar como mucho 1.

| Modelo | Top-3 (BM25 simple: 12/15) | Consultas que empeoran | ¿Cumple? |
|---|---|---|---|
| `phi4-mini` | 13/15 | 0 | No (+1, necesitaba +2) |
| `gemma3:4b` | 11/15 | 1 | No |
| `llama3.2:3b` | 10/15 | 3 | No |

**Lectura.** Mejora una consulta y empeora otra. No se adopta; BM25 simple sigue por defecto.

### 4.7 Localizador de código — retirado

**Qué hace.** Responde dónde se define una función o clase, qué ficheros importan un módulo o dónde aparece una constante, usando `grep` y `list_dir`.

**Prueba.** 10 preguntas con respuesta calculada por código, 2 repeticiones. Criterio: ≥ 80 % de respuestas exactas.

**Resultado.** `llama3.2:3b` acertó el 10 %. Lo que hace es un `grep`, y el modelo solo lo empeora.

**Decisión.** Lo sustituye `amp_agents.locate.locate`, una función determinista, sin modelo. El rol se conserva en el código solo para reproducir el resultado negativo.

### 4.8 Ejecutor (agente con herramientas)

**Qué hace.** Ejecuta tareas acotadas en `src/` con un bucle propio y herramientas con política de rutas: `list_dir`, `read_file`, `grep`, `write_file`, `edit_file` y `run_pytest`. Cambiar de proveedor de modelo es solo configuración.

**Prueba.** Tareas de seguridad (no salirse de las rutas permitidas), de escritura y de lectura, 3 repeticiones.

| Modelo | Seguridad | Escritura | Lectura |
|---|---|---|---|
| `qwen3:4b` | 6/6 | 2/3 | 0/3 |
| `qwen2.5:14b` (una repetición) | 1/1 | 0/1 | 0/1 |

**Lectura.** El modelo pequeño cuenta mal al leer y dio informes falsos (dos casos solo se detectaron con una comprobación independiente). Sirve para tareas de solo lectura o supervisadas, no para escribir sin validación.

## 5. Qué conviene usar

- **Integrar:** el enrutador (`gemma3:4b`), y el resumidor (`gemma3:4b` o `phi4-mini`) siempre con recorte y filtro por código.
- **Usar BM25** para recuperar documentación; `bge-m3` solo si se quiere probar el híbrido.
- **Mantener como opción no predeterminada:** el revisor de código (con revisión humana de lo que señale) y el recuperador híbrido.
- **No usar un modelo para** localizar código, comprobar la coherencia entre documentos ni reescribir consultas.

## 6. Pendiente y límites

- Contrastar el enrutador con peticiones escritas por otra persona.
- Decidir si el verificador de citas debe aceptar el formato `ruta#línea` (algún modelo lo usa). Cambiarlo ahora exigiría declararlo como cambio posterior, sin ocultarlo.
- Todos los resultados son de una GPU de 6 GB, con modelos cuantizados de hasta 9 mil millones de parámetros; con modelos mayores o con GPU más grande podrían cambiar.
- Las pruebas usan documentos y código de este proyecto; no está medido cómo se comportan en otros.

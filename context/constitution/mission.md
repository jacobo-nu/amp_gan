# Misión

## Propósito
Aprender arquitectura de agentes construyendo un proyecto serio y reproducible:
implementar redes GAN que generen secuencias de péptidos antimicrobianos (AMP)
a partir de bases de datos públicas, entregadas como Jupyter notebooks.

## Alcance
- Entrenar y comparar variantes de GAN (vanilla, WGAN-GP, condicional) sobre secuencias de AMP.
- **Comparar resultados** variando (a) la arquitectura de la GAN y (b) la base de datos de entrenamiento según
  lo específica o general que sea: `Antibacterial_amps.fasta` (específica, base principal) frente a
  `general_amps.fasta` (general, usada como contraste entre bases).
- Evaluar de forma cuantitativa: composición de aminoácidos, longitud, carga neta,
  hidrofobicidad, diversidad y novedad frente al conjunto de entrenamiento.
- Documentar cada experimento de forma que sea reproducible.

## Preguntas de investigación
1. ¿Cómo cambia la calidad de las secuencias generadas al variar la arquitectura (vanilla, WGAN-GP, condicional)?
2. ¿Cómo influye entrenar con una base específica (antibacteriana) frente a una general?
3. ¿Qué métricas distinguen mejor unas configuraciones de otras?

## Fuera de alcance
- Validación experimental en laboratorio.
- Cualquier uso distinto de la investigación académica abierta sobre péptidos con
  función terapéutica o antibacteriana.

## Criterios de éxito
1. Notebooks que se ejecutan de principio a fin en un entorno limpio.
2. Resultados reproducibles (semillas fijadas, versiones registradas).
3. Comparativas entre arquitecturas y entre bases de datos hechas en condiciones equivalentes (mismo
   preprocesado, semillas y métricas).
4. Repositorio limpio y sincronizado con GitHub.
5. El flujo orquestador → ejecutor funciona y queda documentado.

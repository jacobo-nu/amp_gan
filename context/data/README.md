# Datos

Los archivos de datos **no se versionan** (ver `.gitignore`). Este documento registra su origen.

## Fuente: DRAMP

Todas las secuencias proceden de **DRAMP** (Data Repository of Antimicrobial Peptides),
<http://dramp.cpu-bioinfor.org/>. Los identificadores `DRAMPxxxxx` de los archivos son los de esta base.

| Archivo | Contenido | Nº de secuencias |
|---|---|---|
| `general_amps.fasta` | Subconjunto general de AMP de DRAMP | 12 784 |
| `Antibacterial_amps.fasta` | Subconjunto de AMP con actividad antibacteriana de DRAMP | 28 702 |

Los dos archivos están en `context/data/raw/` (copiados de `Trabajo Redes Neuronales/`, trabajo previo). Son datos
de DRAMP sin modificar. **No se suben a GitHub** (`.gitignore`): el
repositorio solo los cita; quien lo clone debe descargarlos de DRAMP.

> **Versión:** DRAMP 5.0, descargada el 28-oct-2026 desde <https://dramp.cpu-bioinfor.org/>
> (la web indicaba última actualización el 20-oct-2026, 32 436 entradas).

## Licencia y condiciones de uso

Según la web de DRAMP (consultada el 30-oct-2026), los datos se publican bajo **CC BY 4.0**, y piden citar sus
publicaciones en toda descarga. Además indican:

- Para AMP generales y clínicos, citar también el artículo original de cada péptido.
- Para AMP procedentes de **patentes**, hace falta autorización.
- El sitio se ofrece "tal cual", sin garantías.

Los datos no se redistribuyen en el repositorio, por lo que las condiciones sobre patentes no afectan a su
publicación; solo hay que citar la base y a sus autores.

## Cómo citar

Citar la versión de DRAMP utilizada (la 5.0):

- Ma T, Huang Y, Liu H, Wu W, Ren L, Akhtar F, Nawaz M, Hua Z, Zheng H, Lao X.
  DRAMP 5.0: an update integrating sequence and structure in a novel FASTS format for AI-driven design of
  antimicrobial peptides. *Nucleic Acids Research* 2026; gkag989 (publicado el 15-oct-2026). PMID: 42876369.
  (La web no indica DOI; añadirlo cuando figure en la página del artículo.)

Publicaciones anteriores de DRAMP, que la web también pide citar y que respaldan versiones previas:

- Shi G, Kang X, Dong F, Liu Y, Zhu N, Hu Y, Xu H, Lao X, Zheng H. DRAMP 3.0: an enhanced comprehensive data
  repository of antimicrobial peptides. *Nucleic Acids Research* 2022; 50(D1): D488-D496.
  doi:[10.1093/nar/gkab651](https://doi.org/10.1093/nar/gkab651)
- Kang X, Dong F, Shi C, Liu S, Sun J, Chen J, Li H, Xu H, Lao X, Zheng H. DRAMP 2.0, an updated data
  repository of antimicrobial peptides. *Scientific Data* 2019; 6: 148.
  doi:[10.1038/s41597-019-0154-y](https://doi.org/10.1038/s41597-019-0154-y)
- Fan L, Sun J, Zhou M, Zhou J, Lao X, Zheng H, Xu H. DRAMP: a comprehensive data repository of
  antimicrobial peptides. *Scientific Reports* 2016; 6: 24482.
  doi:[10.1038/srep24482](https://doi.org/10.1038/srep24482)

BibTeX de la versión 5.0:

```bibtex
@article{ma2026dramp5,
  author  = {Ma, T. and Huang, Y. and Liu, H. and Wu, W. and Ren, L. and Akhtar, F. and Nawaz, M. and Hua, Z. and Zheng, H. and Lao, X.},
  title   = {{DRAMP} 5.0: an update integrating sequence and structure in a novel {FASTS} format for {AI}-driven design of antimicrobial peptides},
  journal = {Nucleic Acids Research},
  year    = {2026},
  pages   = {gkag989},
  note    = {PMID: 42876369}
}
```

## Preprocesado

Las dos bases se limpian con el mismo procedimiento (`src/amp_gan/data.py`), en este orden:

1. Se descartan las secuencias con algún carácter fuera de los 20 aminoácidos estándar.
2. Se conservan las de longitud entre **5 y 60 aa** (ambos inclusive).
3. Se eliminan los duplicados exactos de secuencia (se conserva la primera aparición).

El rango se fijó a la vista de las estadísticas de longitud (`processed/length_stats.csv`,
`processed/length_coverage.csv`, `processed/length_hist.png`): 5–60 aa conserva el ~93 % de las secuencias de
`Antibacterial_amps` y el ~94 % de `general_amps` (datos sin limpiar). Se recorta para reducir el espacio de
soluciones de las primeras GAN.

"""Descarga reproducible de las bases de DRAMP a ``context/data/raw/``.

Descarga los dos FASTA que usa el proyecto (``Antibacterial_amps`` y ``general_amps``) desde la
página oficial de descargas de DRAMP, comprueba que su huella SHA-256 coincide con la de la
versión con la que se trabajó (DRAMP 5.0, descargada el 28-oct-2026) y los deja donde espera
``make_clean_data.py``. Solo usa la biblioteca estándar.

- Es **idempotente**: un fichero que ya existe y tiene la huella esperada no se vuelve a descargar.
- Un fichero descargado con otra huella **no se instala** (se informa y el código de salida es 1):
  DRAMP puede haber actualizado la base, y entonces los resultados dejarían de ser comparables.
  ``--accept-changes`` lo instala igualmente e imprime la huella nueva para actualizar ``DATASETS``.
- La descarga va a un fichero temporal y se renombra al final: nunca queda un FASTA a medias.

Uso: ``python src/amp_gan/download_data.py [--force] [--accept-changes] [--clean]``
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "context" / "data" / "raw"

#: Página de descargas de DRAMP; las URL de los ficheros cuelgan de ella.
DOWNLOAD_BASE = "https://dramp.cpu-bioinfor.org/downloads/download.php?filename=download_data/DRAMP3.0_new/"

#: Segundos de espera máxima por operación de red.
TIMEOUT_S = 120

#: Tamaño del bloque de lectura (bytes).
CHUNK = 1 << 16


@dataclass(frozen=True)
class Dataset:
    """Un FASTA de DRAMP: nombre del fichero, huella esperada y tamaño esperado (bytes)."""

    name: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return f"{DOWNLOAD_BASE}{self.name}"


#: Versión de los datos con la que se trabajó (DRAMP 5.0, descargada el 28-oct-2026).
DATASETS: tuple[Dataset, ...] = (
    Dataset("Antibacterial_amps.fasta", "a0d484eb6176e123298d19d06f23ccc6398a1f5c75d169cccbbf7590a6e6c835", 1072625),
    Dataset("general_amps.fasta", "8326ff308eb3ae67cafbdbf3756c0e943d7d968e5e0a338298ce602269d1caf0", 477245),
)

#: Función que, dada una URL, devuelve el contenido como iterador de bloques de bytes.
Fetch = Callable[[str], Iterator[bytes]]


def sha256_of(path: Path) -> str:
    """Huella SHA-256 (hexadecimal) de un fichero, leído por bloques."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def http_fetch(url: str) -> Iterator[bytes]:
    """Descarga ``url`` por HTTPS y devuelve el contenido por bloques (agente de usuario genérico)."""
    request = urllib.request.Request(url, headers={"User-Agent": "amp-gan-data-download"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310 (URL fija, https)
        while block := response.read(CHUNK):
            yield block


def ensure_dataset(
    dataset: Dataset,
    raw_dir: Path,
    *,
    fetch: Fetch = http_fetch,
    force: bool = False,
    accept_changes: bool = False,
) -> str:
    """Deja ``dataset`` en ``raw_dir`` y devuelve qué se hizo: ``"presente"``, ``"descargado"`` o ``"distinto"``.

    ``"distinto"`` significa que lo descargado no tiene la huella esperada y **no** se ha instalado
    (salvo con ``accept_changes``, que lo instala y devuelve ``"descargado"``).

    Raises:
        OSError: si la red falla (se propaga el error de ``fetch``); no queda ningún fichero a medias.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / dataset.name
    if dest.exists() and not force and sha256_of(dest) == dataset.sha256:
        return "presente"

    fd, tmp_name = tempfile.mkstemp(dir=raw_dir, prefix=f".{dataset.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as handle:
            for block in fetch(dataset.url):
                digest.update(block)
                handle.write(block)
        if digest.hexdigest() != dataset.sha256 and not accept_changes:
            return "distinto"
        os.replace(tmp, dest)
        return "descargado"
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None, *, fetch: Fetch = http_fetch, raw_dir: Path = RAW_DIR) -> int:
    parser = argparse.ArgumentParser(description="Descarga las bases de DRAMP a context/data/raw/ y comprueba su huella.")
    parser.add_argument("--force", action="store_true", help="volver a descargar aunque el fichero ya esté y sea correcto")
    parser.add_argument("--accept-changes", action="store_true", help="instalar un fichero con huella distinta a la esperada (DRAMP actualizada)")
    parser.add_argument("--clean", action="store_true", help="ejecutar make_clean_data.py al terminar")
    args = parser.parse_args(argv)

    status = 0
    for dataset in DATASETS:
        try:
            result = ensure_dataset(dataset, raw_dir, fetch=fetch, force=args.force, accept_changes=args.accept_changes)
        except OSError as error:
            print(f"{dataset.name}: ERROR de red ({error}).", file=sys.stderr)
            status = 1
            continue
        dest = raw_dir / dataset.name
        if result == "presente":
            print(f"{dataset.name}: ya está y su huella es la esperada.")
        elif result == "descargado":
            actual = sha256_of(dest)
            note = "" if actual == dataset.sha256 else f" (HUELLA DISTINTA, nueva: {actual}; actualiza DATASETS si la aceptas)"
            print(f"{dataset.name}: descargado ({dest.stat().st_size} bytes){note}.")
        else:
            print(
                f"{dataset.name}: la huella descargada no coincide con la de DRAMP 5.0 (28-oct-2026); no se ha instalado. "
                "DRAMP puede haber actualizado la base; usa --accept-changes si es lo que quieres.",
                file=sys.stderr,
            )
            status = 1
    if args.clean and status == 0:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from amp_gan import make_clean_data  # noqa: PLC0415

        make_clean_data.main()
        print("Bases limpias escritas en context/data/processed/.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

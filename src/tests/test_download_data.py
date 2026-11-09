"""Tests de ``amp_gan.download_data`` sin red: el contenido llega por una función ``fetch`` falsa."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from amp_gan import download_data as dd

CONTENT = b">DRAMP00005\nSLGPAIKATRQVCPKATRFVTVSCKKSDCQ\n"
GOOD = dd.Dataset("demo.fasta", hashlib.sha256(CONTENT).hexdigest(), len(CONTENT))


def _fetch_of(data: bytes, calls: list[str] | None = None):
    def fetch(url: str):
        if calls is not None:
            calls.append(url)
        for i in range(0, len(data), 7):  # bloques pequeños: se comprueba que se concatenan bien
            yield data[i : i + 7]

    return fetch


def test_pinned_datasets_cover_the_two_dramp_bases_with_https_urls_and_valid_hashes():
    assert [d.name for d in dd.DATASETS] == ["Antibacterial_amps.fasta", "general_amps.fasta"]
    for d in dd.DATASETS:
        assert d.url.startswith("https://dramp.cpu-bioinfor.org/") and d.url.endswith(d.name)
        assert len(d.sha256) == 64 and int(d.sha256, 16) >= 0 and d.size > 0


def test_sha256_of_matches_hashlib_for_a_file_larger_than_one_chunk(tmp_path: Path):
    data = b"ACDEFGHIKL" * (dd.CHUNK // 5)
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert dd.sha256_of(path) == hashlib.sha256(data).hexdigest()


def test_downloads_a_missing_file_and_leaves_no_temporary_files(tmp_path: Path):
    calls: list[str] = []
    assert dd.ensure_dataset(GOOD, tmp_path, fetch=_fetch_of(CONTENT, calls)) == "descargado"
    assert (tmp_path / "demo.fasta").read_bytes() == CONTENT
    assert calls == [GOOD.url]
    assert [p.name for p in tmp_path.iterdir()] == ["demo.fasta"]


def test_creates_the_destination_folder(tmp_path: Path):
    target = tmp_path / "a" / "b"
    assert dd.ensure_dataset(GOOD, target, fetch=_fetch_of(CONTENT)) == "descargado"
    assert (target / "demo.fasta").exists()


def test_does_not_download_again_when_the_file_is_present_and_correct(tmp_path: Path):
    (tmp_path / "demo.fasta").write_bytes(CONTENT)
    calls: list[str] = []
    assert dd.ensure_dataset(GOOD, tmp_path, fetch=_fetch_of(b"otro", calls)) == "presente"
    assert calls == []
    assert (tmp_path / "demo.fasta").read_bytes() == CONTENT


def test_force_downloads_again_even_when_the_file_is_correct(tmp_path: Path):
    (tmp_path / "demo.fasta").write_bytes(CONTENT)
    calls: list[str] = []
    assert dd.ensure_dataset(GOOD, tmp_path, fetch=_fetch_of(CONTENT, calls), force=True) == "descargado"
    assert calls == [GOOD.url]


def test_replaces_a_corrupted_local_file_with_the_correct_download(tmp_path: Path):
    (tmp_path / "demo.fasta").write_bytes(b"corrupto")
    assert dd.ensure_dataset(GOOD, tmp_path, fetch=_fetch_of(CONTENT)) == "descargado"
    assert (tmp_path / "demo.fasta").read_bytes() == CONTENT


def test_a_download_with_a_different_hash_is_not_installed_and_keeps_the_existing_file(tmp_path: Path):
    (tmp_path / "demo.fasta").write_bytes(b"version vieja")
    assert dd.ensure_dataset(GOOD, tmp_path, fetch=_fetch_of(b">X\nAAAAA\n")) == "distinto"
    assert (tmp_path / "demo.fasta").read_bytes() == b"version vieja"
    assert [p.name for p in tmp_path.iterdir()] == ["demo.fasta"]


def test_a_different_download_with_no_previous_file_installs_nothing(tmp_path: Path):
    assert dd.ensure_dataset(GOOD, tmp_path, fetch=_fetch_of(b">X\nAAAAA\n")) == "distinto"
    assert list(tmp_path.iterdir()) == []


def test_accept_changes_installs_a_download_with_a_different_hash(tmp_path: Path):
    new = b">X\nAAAAA\n"
    assert dd.ensure_dataset(GOOD, tmp_path, fetch=_fetch_of(new), accept_changes=True) == "descargado"
    assert (tmp_path / "demo.fasta").read_bytes() == new


def test_a_network_error_in_the_middle_leaves_no_partial_file(tmp_path: Path):
    def broken(url: str):
        yield CONTENT[:10]
        raise OSError("conexión cortada")

    with pytest.raises(OSError, match="cortada"):
        dd.ensure_dataset(GOOD, tmp_path, fetch=broken)
    assert list(tmp_path.iterdir()) == []


def test_a_network_error_does_not_destroy_an_existing_file(tmp_path: Path):
    (tmp_path / "demo.fasta").write_bytes(b"version vieja")

    def broken(url: str):
        raise OSError("sin red")
        yield b""  # pragma: no cover

    with pytest.raises(OSError):
        dd.ensure_dataset(GOOD, tmp_path, fetch=broken)
    assert (tmp_path / "demo.fasta").read_bytes() == b"version vieja"


def test_main_downloads_both_files_then_reports_them_as_present(tmp_path: Path, monkeypatch, capsys):
    data = {d.name: f"contenido de {d.name}".encode() for d in dd.DATASETS}
    fake = tuple(dd.Dataset(d.name, hashlib.sha256(data[d.name]).hexdigest(), len(data[d.name])) for d in dd.DATASETS)
    monkeypatch.setattr(dd, "DATASETS", fake)
    by_url = {d.url: data[d.name] for d in fake}

    def fetch(url: str):
        yield by_url[url]

    assert dd.main([], fetch=fetch, raw_dir=tmp_path) == 0
    assert "descargado" in capsys.readouterr().out
    assert {p.name for p in tmp_path.iterdir()} == set(data)
    assert dd.main([], fetch=lambda url: iter(()), raw_dir=tmp_path) == 0  # segunda vez: no pide nada
    assert capsys.readouterr().out.count("ya está") == 2


def test_main_returns_1_when_a_hash_does_not_match_and_installs_nothing(tmp_path: Path, capsys):
    status = dd.main([], fetch=lambda url: iter([b"otra version"]), raw_dir=tmp_path)
    assert status == 1
    assert list(tmp_path.iterdir()) == []
    assert "no coincide" in capsys.readouterr().err


def test_main_returns_1_on_network_error_and_keeps_trying_the_other_file(tmp_path: Path, capsys):
    seen: list[str] = []

    def fetch(url: str):
        seen.append(url)
        raise OSError("sin red")
        yield b""  # pragma: no cover

    assert dd.main([], fetch=fetch, raw_dir=tmp_path) == 1
    assert len(seen) == 2
    assert capsys.readouterr().err.count("ERROR de red") == 2


def test_main_accept_changes_installs_and_warns_about_the_new_hash(tmp_path: Path, capsys):
    assert dd.main(["--accept-changes"], fetch=lambda url: iter([b">X\nAAAAA\n"]), raw_dir=tmp_path) == 0
    assert "HUELLA DISTINTA" in capsys.readouterr().out
    assert len(list(tmp_path.iterdir())) == 2


def test_the_pinned_hashes_match_the_local_raw_files_when_they_exist():
    """Si los datos locales están, deben ser la versión fijada (si DRAMP cambia, hay que actualizar DATASETS)."""
    for d in dd.DATASETS:
        local = dd.RAW_DIR / d.name
        if local.exists():
            assert local.stat().st_size == d.size and dd.sha256_of(local) == d.sha256


def test_main_clean_runs_make_clean_data_only_when_everything_went_well(tmp_path: Path, monkeypatch):
    from amp_gan import make_clean_data

    ran: list[bool] = []
    monkeypatch.setattr(make_clean_data, "main", lambda: ran.append(True))
    assert dd.main(["--clean"], fetch=lambda url: iter([b"otra version"]), raw_dir=tmp_path) == 1
    assert ran == []  # huella distinta: no se limpia nada
    assert dd.main(["--clean", "--accept-changes"], fetch=lambda url: iter([b"otra version"]), raw_dir=tmp_path) == 0
    assert ran == [True]

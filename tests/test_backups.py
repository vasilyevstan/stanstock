from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from django.test import override_settings

from stanstock.core.backups import (
    BackupError,
    create_backup_bundle,
    extract_verified_backup,
    verify_backup_bundle,
)
from stanstock.core.management.commands import backup as backup_command
from stanstock.core.management.commands import restore as restore_command


def test_backup_bundle_keeps_database_and_assets_together(tmp_path: Path) -> None:
    database = tmp_path / "source.sqlite3"
    database.write_bytes(b"database snapshot")
    data_dir = tmp_path / "data"
    (data_dir / "prices").mkdir(parents=True)
    (data_dir / "prices" / "sample.parquet").write_bytes(b"immutable prices")
    output = tmp_path / "backups" / "snapshot.tar.gz"

    create_backup_bundle(
        database_artifact=database,
        database_format="sqlite",
        data_dir=data_dir,
        output_path=output,
        metadata={"code_revision": "test"},
    )
    verified = verify_backup_bundle(output)
    extraction = tmp_path / "extracted"
    extract_verified_backup(output, extraction)

    assert verified.manifest["database"]["format"] == "sqlite"
    assert verified.asset_members == ("assets/prices/sample.parquet",)
    assert (extraction / verified.database_member).read_bytes() == b"database snapshot"
    assert (extraction / "assets" / "prices" / "sample.parquet").read_bytes() == b"immutable prices"


def test_backup_rejects_output_inside_data_directory(tmp_path: Path) -> None:
    database = tmp_path / "source.sqlite3"
    database.write_bytes(b"database snapshot")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(BackupError, match="outside STANSTOCK_DATA_DIR"):
        create_backup_bundle(
            database_artifact=database,
            database_format="sqlite",
            data_dir=data_dir,
            output_path=data_dir / "snapshot.tar.gz",
        )


def test_backup_rejects_unsafe_archive_member(tmp_path: Path) -> None:
    bundle = tmp_path / "unsafe.tar.gz"
    manifest = {
        "version": 1,
        "database": {"format": "sqlite", "path": "../database.sqlite3"},
        "files": [],
    }
    with tarfile.open(bundle, mode="w:gz") as archive:
        payload = json.dumps(manifest).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        unsafe = tarfile.TarInfo("../database.sqlite3")
        unsafe.size = 1
        archive.addfile(unsafe, io.BytesIO(b"x"))

    with pytest.raises(BackupError, match="Unsafe backup member path"):
        verify_backup_bundle(bundle)


def test_backup_rejects_duplicate_archive_members(tmp_path: Path) -> None:
    bundle = tmp_path / "duplicate.tar.gz"
    with tarfile.open(bundle, mode="w:gz") as archive:
        for payload in (b"{}", b'{"version": 1}'):
            info = tarfile.TarInfo("manifest.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(BackupError, match="duplicate member names"):
        verify_backup_bundle(bundle)


def test_postgresql_restore_is_fail_fast_and_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "database.dump"
    artifact.write_bytes(b"test")
    fake_connection = SimpleNamespace(
        settings_dict={
            "ENGINE": "django.db.backends.postgresql",
            "HOST": "db",
            "PORT": "5432",
            "USER": "stanstock",
            "PASSWORD": "secret",
            "NAME": "stanstock",
        },
        close=lambda: None,
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(restore_command, "connection", fake_connection)
    monkeypatch.setattr(restore_command.subprocess, "run", fake_run)

    restore_command.Command()._restore_database(artifact, "postgresql-custom")

    assert len(calls) == 1
    assert "--clean" in calls[0]
    assert "--exit-on-error" in calls[0]
    assert "--single-transaction" in calls[0]


def test_backup_command_uses_configured_writable_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    backup_dir = tmp_path / "private-backups"
    command = backup_command.Command()

    def fake_snapshot(temporary_dir: Path) -> tuple[Path, str]:
        artifact = temporary_dir / "database.sqlite3"
        artifact.write_bytes(b"database snapshot")
        return artifact, "sqlite"

    monkeypatch.setattr(command, "_snapshot_database", fake_snapshot)

    with override_settings(DATA_DIR=data_dir, BACKUP_DIR=backup_dir):
        command.handle(output=None)

    bundles = list(backup_dir.glob("stanstock-*.tar.gz"))
    assert len(bundles) == 1
    assert verify_backup_bundle(bundles[0]).manifest["database"]["format"] == "sqlite"

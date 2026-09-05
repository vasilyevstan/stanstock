from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

BUFFER_SIZE = 1024 * 1024


class BackupError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VerifiedBackup:
    manifest: dict[str, Any]
    database_member: str
    asset_members: tuple[str, ...]


def create_backup_bundle(
    *,
    database_artifact: Path,
    database_format: str,
    data_dir: Path,
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    database_artifact = database_artifact.resolve()
    data_dir = data_dir.resolve()
    output_path = output_path.resolve()
    if database_format not in {"sqlite", "postgresql-custom"}:
        raise BackupError(f"Unsupported database backup format: {database_format}")
    if not database_artifact.is_file():
        raise BackupError(f"Database artifact does not exist: {database_artifact}")
    if output_path == data_dir or data_dir in output_path.parents:
        raise BackupError("Backup output must be outside STANSTOCK_DATA_DIR")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stanstock-backup-") as temporary:
        staging = Path(temporary)
        database_member = (
            "database/database.sqlite3" if database_format == "sqlite" else "database/database.dump"
        )
        staged_database = staging / database_member
        staged_database.parent.mkdir(parents=True)
        shutil.copy2(database_artifact, staged_database)

        staged_files = [staged_database]
        if data_dir.exists():
            for source in sorted(data_dir.rglob("*")):
                if source.is_symlink():
                    raise BackupError(f"Symbolic links are not allowed in DATA_DIR: {source}")
                if not source.is_file():
                    continue
                relative = source.relative_to(data_dir)
                destination = staging / "assets" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                staged_files.append(destination)

        entries = [
            {
                "path": path.relative_to(staging).as_posix(),
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
            for path in staged_files
        ]
        manifest = {
            "version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "database": {
                "format": database_format,
                "path": database_member,
            },
            "files": entries,
            "metadata": metadata or {},
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_bundle:
            temporary_path = Path(temporary_bundle.name)
        try:
            with tarfile.open(temporary_path, mode="w:gz") as archive:
                archive.add(manifest_path, arcname="manifest.json", recursive=False)
                for path in staged_files:
                    archive.add(
                        path,
                        arcname=path.relative_to(staging).as_posix(),
                        recursive=False,
                    )
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    return output_path


def verify_backup_bundle(bundle_path: Path) -> VerifiedBackup:
    bundle_path = bundle_path.resolve()
    if not bundle_path.is_file():
        raise BackupError(f"Backup bundle does not exist: {bundle_path}")

    with tarfile.open(bundle_path, mode="r:gz") as archive:
        member_list = archive.getmembers()
        members = {member.name: member for member in member_list}
        if len(members) != len(member_list):
            raise BackupError("Backup contains duplicate member names")
        for member in members.values():
            _validate_member_name(member.name)
            if not member.isfile():
                raise BackupError(f"Backup contains a non-file member: {member.name}")

        manifest_member = members.get("manifest.json")
        if manifest_member is None:
            raise BackupError("Backup manifest is missing")
        manifest_stream = archive.extractfile(manifest_member)
        if manifest_stream is None:
            raise BackupError("Backup manifest cannot be read")
        try:
            manifest = json.load(manifest_stream)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BackupError("Backup manifest is invalid JSON") from exc

        if not isinstance(manifest, dict) or manifest.get("version") != 1:
            raise BackupError("Unsupported backup manifest version")
        database = manifest.get("database")
        files = manifest.get("files")
        if not isinstance(database, dict) or not isinstance(files, list):
            raise BackupError("Backup manifest has an invalid structure")
        database_format = database.get("format")
        if database_format not in {"sqlite", "postgresql-custom"}:
            raise BackupError("Backup manifest has an unsupported database format")
        database_member = database.get("path")
        if not isinstance(database_member, str):
            raise BackupError("Backup manifest has no database path")

        expected_names = {"manifest.json"}
        asset_members: list[str] = []
        for entry in files:
            if not isinstance(entry, dict):
                raise BackupError("Backup manifest contains an invalid file entry")
            name = entry.get("path")
            expected_sha = entry.get("sha256")
            expected_size = entry.get("size")
            if not isinstance(name, str) or not isinstance(expected_sha, str):
                raise BackupError("Backup manifest file metadata is incomplete")
            _validate_member_name(name)
            archive_member = members.get(name)
            if archive_member is None:
                raise BackupError(f"Backup member is missing: {name}")
            if archive_member.size != expected_size:
                raise BackupError(f"Backup member size does not match: {name}")
            stream = archive.extractfile(archive_member)
            if stream is None:
                raise BackupError(f"Backup member cannot be read: {name}")
            digest = hashlib.sha256()
            while chunk := stream.read(BUFFER_SIZE):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha:
                raise BackupError(f"Backup member checksum does not match: {name}")
            expected_names.add(name)
            if name.startswith("assets/"):
                asset_members.append(name)

        if database_member not in expected_names:
            raise BackupError("Database artifact is not listed in the manifest")
        unexpected_names = set(members) - expected_names
        if unexpected_names:
            raise BackupError(
                f"Backup contains unlisted members: {', '.join(sorted(unexpected_names))}"
            )

    return VerifiedBackup(
        manifest=manifest,
        database_member=database_member,
        asset_members=tuple(sorted(asset_members)),
    )


def extract_verified_backup(bundle_path: Path, destination: Path) -> VerifiedBackup:
    verified = verify_backup_bundle(bundle_path)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    members_to_extract = {
        "manifest.json",
        verified.database_member,
        *verified.asset_members,
    }
    with tarfile.open(bundle_path, mode="r:gz") as archive:
        for name in sorted(members_to_extract):
            member = archive.getmember(name)
            stream = archive.extractfile(member)
            if stream is None:
                raise BackupError(f"Backup member cannot be extracted: {name}")
            target = (destination / name).resolve()
            if destination != target and destination not in target.parents:
                raise BackupError(f"Backup member escapes extraction directory: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                shutil.copyfileobj(stream, output, length=BUFFER_SIZE)
    return verified


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or not path.parts:
        raise BackupError(f"Unsafe backup member path: {name}")

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from stanstock.core.backups import (
    BackupError,
    extract_verified_backup,
    verify_backup_bundle,
)


class Command(BaseCommand):
    help = "Verify or restore a StanStock database-and-assets backup bundle."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("bundle", type=Path)
        parser.add_argument("--verify-only", action="store_true")
        parser.add_argument(
            "--confirm",
            help='Required for restore; pass the exact value "RESTORE".',
        )

    def handle(self, *args: object, **options: object) -> None:
        bundle_option = options.get("bundle")
        if not isinstance(bundle_option, Path):
            raise CommandError("bundle must be a filesystem path")
        bundle = bundle_option
        try:
            verified = verify_backup_bundle(bundle)
            if options["verify_only"]:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Backup verified: {len(verified.asset_members)} asset files"
                    )
                )
                return
            if options["confirm"] != "RESTORE":
                raise CommandError("Restore requires --confirm RESTORE")

            with tempfile.TemporaryDirectory(prefix="stanstock-restore-") as temporary:
                staging = Path(temporary)
                verified = extract_verified_backup(bundle, staging)
                asset_plan = self._prepare_asset_restore(staging, verified.asset_members)
                self._restore_database(
                    staging / verified.database_member,
                    str(verified.manifest["database"]["format"]),
                )
                self._restore_assets(asset_plan)
        except (BackupError, OSError, subprocess.CalledProcessError) as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Restore completed. Restart every StanStock process before serving traffic."
            )
        )

    def _restore_database(self, artifact: Path, database_format: str) -> None:
        engine = connection.settings_dict["ENGINE"]
        if engine == "django.db.backends.sqlite3":
            if database_format != "sqlite":
                raise BackupError("A PostgreSQL backup cannot be restored into SQLite")
            database_path = Path(str(connection.settings_dict["NAME"])).resolve()
            if database_path.name == ":memory:":
                raise BackupError("Cannot restore into an in-memory SQLite database")
            database_path.parent.mkdir(parents=True, exist_ok=True)
            connection.close()
            temporary_database = database_path.with_suffix(f"{database_path.suffix}.restore")
            shutil.copy2(artifact, temporary_database)
            os.replace(temporary_database, database_path)
            return

        if engine == "django.db.backends.postgresql":
            if database_format != "postgresql-custom":
                raise BackupError("A SQLite backup cannot be restored into PostgreSQL")
            database = connection.settings_dict
            environment = os.environ.copy()
            password = database.get("PASSWORD")
            if password:
                environment["PGPASSWORD"] = str(password)
            connection.close()
            subprocess.run(
                [
                    "pg_restore",
                    "--clean",
                    "--if-exists",
                    "--exit-on-error",
                    "--single-transaction",
                    "--no-owner",
                    "--no-privileges",
                    "--host",
                    str(database.get("HOST") or "localhost"),
                    "--port",
                    str(database.get("PORT") or 5432),
                    "--username",
                    str(database.get("USER") or ""),
                    "--dbname",
                    str(database.get("NAME") or ""),
                    str(artifact),
                ],
                check=True,
                env=environment,
            )
            return

        raise BackupError(f"Unsupported database backend: {engine}")

    def _prepare_asset_restore(
        self,
        staging: Path,
        members: tuple[str, ...],
    ) -> list[tuple[Path, Path]]:
        data_dir = settings.DATA_DIR.resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        plan: list[tuple[Path, Path]] = []
        required_bytes = 0
        for member in members:
            relative_path = Path(*Path(member).parts[1:])
            source = staging / member
            target = (data_dir / relative_path).resolve()
            if data_dir != target and data_dir not in target.parents:
                raise BackupError(f"Asset escapes STANSTOCK_DATA_DIR: {member}")
            if not source.is_file():
                raise BackupError(f"Extracted backup asset is missing: {member}")
            target.parent.mkdir(parents=True, exist_ok=True)
            required_bytes += source.stat().st_size
            plan.append((source, target))

        available_bytes = shutil.disk_usage(data_dir).free
        if required_bytes > available_bytes:
            raise BackupError(
                f"Insufficient free space to restore assets: need {required_bytes} bytes, "
                f"have {available_bytes} bytes"
            )
        with tempfile.NamedTemporaryFile(
            dir=data_dir,
            prefix=".stanstock-restore-write-probe-",
        ) as probe:
            probe.write(b"ok")
            probe.flush()
        return plan

    def _restore_assets(self, plan: list[tuple[Path, Path]]) -> None:
        for source, target in plan:
            temporary_target = target.with_suffix(f"{target.suffix}.restore")
            shutil.copy2(source, temporary_target)
            os.replace(temporary_target, target)

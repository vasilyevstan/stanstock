from __future__ import annotations

import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import django
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from django.utils import timezone

from stanstock.core.backups import BackupError, create_backup_bundle


class Command(BaseCommand):
    help = "Create one checksummed bundle containing the database and DATA_DIR."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--output", type=Path)

    def handle(self, *args: object, **options: object) -> None:
        output_option = options.get("output")
        if output_option is not None and not isinstance(output_option, Path):
            raise CommandError("--output must be a filesystem path")
        output = (
            output_option
            if output_option is not None
            else settings.BACKUP_DIR / f"stanstock-{timezone.now():%Y%m%dT%H%M%SZ}.tar.gz"
        )

        try:
            with tempfile.TemporaryDirectory(prefix="stanstock-database-") as temporary:
                temporary_dir = Path(temporary)
                database_artifact, database_format = self._snapshot_database(temporary_dir)
                bundle = create_backup_bundle(
                    database_artifact=database_artifact,
                    database_format=database_format,
                    data_dir=settings.DATA_DIR,
                    output_path=output,
                    metadata={
                        "django_version": django.get_version(),
                        "code_revision": os.getenv(
                            "STANSTOCK_CODE_REVISION",
                            "working-tree",
                        ),
                    },
                )
        except (BackupError, OSError, subprocess.CalledProcessError) as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS(f"Backup created: {bundle}"))

    def _snapshot_database(self, temporary_dir: Path) -> tuple[Path, str]:
        engine = connection.settings_dict["ENGINE"]
        if engine == "django.db.backends.sqlite3":
            destination_path = temporary_dir / "database.sqlite3"
            connection.ensure_connection()
            source = connection.connection
            if not isinstance(source, sqlite3.Connection):
                raise BackupError("Django did not provide a SQLite connection")
            with sqlite3.connect(destination_path) as destination:
                source.backup(destination)
            return destination_path, "sqlite"

        if engine == "django.db.backends.postgresql":
            destination_path = temporary_dir / "database.dump"
            database = connection.settings_dict
            environment = os.environ.copy()
            password = database.get("PASSWORD")
            if password:
                environment["PGPASSWORD"] = str(password)
            subprocess.run(
                [
                    "pg_dump",
                    "--format=custom",
                    "--no-owner",
                    "--no-privileges",
                    "--file",
                    str(destination_path),
                    "--host",
                    str(database.get("HOST") or "localhost"),
                    "--port",
                    str(database.get("PORT") or 5432),
                    "--username",
                    str(database.get("USER") or ""),
                    "--dbname",
                    str(database.get("NAME") or ""),
                ],
                check=True,
                env=environment,
            )
            return destination_path, "postgresql-custom"

        raise BackupError(f"Unsupported database backend: {engine}")

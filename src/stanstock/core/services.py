from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from django.conf import settings
from django.db import connection


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    name: str
    ok: bool
    detail: str


def database_status() -> ComponentStatus:
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
    except Exception as exc:  # Django exposes backend-specific database exceptions.
        return ComponentStatus("Database", False, exc.__class__.__name__)
    return ComponentStatus("Database", True, "Connected")


def data_directory_status(data_dir: Path | None = None) -> ComponentStatus:
    directory = data_dir or settings.DATA_DIR
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=directory,
            prefix=".stanstock-write-probe-",
        ) as probe:
            probe.write(b"ok")
            probe.flush()
    except OSError as exc:
        return ComponentStatus("Data directory", False, exc.__class__.__name__)
    return ComponentStatus("Data directory", True, "Writable")


def system_status() -> list[dict[str, object]]:
    return [asdict(database_status()), asdict(data_directory_status())]

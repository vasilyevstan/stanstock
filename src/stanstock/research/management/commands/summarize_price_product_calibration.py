from __future__ import annotations

import json
import os
import stat
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from stanstock.research.price_product_calibration import (
    render_quantile_calibration,
    summarize_quantile_calibration,
)

MAX_STUDY_FILE_BYTES = 64 * 1024 * 1024


class _DuplicateKeyError(ValueError):
    pass


class Command(BaseCommand):
    help = (
        "Read-only per-quantile calibration ledger over a saved retrospective replay JSON "
        "document (replay_price_product --format json). It reads one local file and writes only "
        "to stdout: no database, asset store, provider, credential, or file write."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--study-file",
            required=True,
            type=Path,
            help="Existing regular (non-symlink) JSON file emitted by replay_price_product",
        )
        parser.add_argument(
            "--format",
            required=True,
            choices=("json", "text"),
            help="Explicit output format",
        )

    def handle(self, *args: object, **options: object) -> None:
        study_file = options["study_file"]
        if not isinstance(study_file, Path):
            raise CommandError("--study-file must be a filesystem path")
        try:
            document = _read_study_document(study_file)
            ledger = summarize_quantile_calibration(document)
            rendered = render_quantile_calibration(ledger, output_format=str(options["format"]))
        except OSError as exc:
            raise CommandError("Could not read --study-file") from exc
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(rendered)


def _read_study_document(path: Path) -> Any:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise CommandError("--study-file must be an existing regular file") from exc
    if stat.S_ISLNK(mode):
        raise CommandError("--study-file must not be a symbolic link")
    if not stat.S_ISREG(mode):
        raise CommandError("--study-file must be an existing regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise CommandError("--study-file must be an existing regular file")
        raw = handle.read(MAX_STUDY_FILE_BYTES + 1)
        if len(raw) > MAX_STUDY_FILE_BYTES:
            raise CommandError("--study-file exceeds the 64 MiB size limit")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except _DuplicateKeyError as exc:
        raise CommandError("--study-file contains a duplicate JSON object key") from exc
    except (RecursionError, ValueError) as exc:
        raise CommandError("--study-file is not valid UTF-8 JSON") from exc


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError("duplicate JSON object key")
        result[key] = value
    return result

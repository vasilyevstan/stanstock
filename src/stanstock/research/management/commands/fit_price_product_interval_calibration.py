from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from stanstock.research.management.commands.summarize_price_product_calibration import (
    _read_study_document,
)
from stanstock.research.price_product_interval_calibration_document import (
    fit_interval_calibration_document,
    render_interval_calibration,
)


class Command(BaseCommand):
    help = (
        "Fit a research-only interval calibration adjustment from a saved retrospective study. "
        "Reads one local file and writes stdout only; it does not register or apply parameters."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--study-file",
            required=True,
            type=Path,
            help="Existing regular (non-symlink) JSON file from replay_price_product",
        )
        parser.add_argument(
            "--format",
            required=True,
            choices=("json", "text"),
            help="Explicit output format",
        )

    def handle(self, *args: object, **options: Any) -> None:
        study_file = options["study_file"]
        if not isinstance(study_file, Path):
            raise CommandError("--study-file must be a filesystem path")
        try:
            document = _read_study_document(study_file)
            fit_document = fit_interval_calibration_document(document)
            rendered = render_interval_calibration(
                fit_document,
                output_format=str(options["format"]),
            )
        except OSError as exc:
            raise CommandError("Could not read --study-file") from exc
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(rendered)

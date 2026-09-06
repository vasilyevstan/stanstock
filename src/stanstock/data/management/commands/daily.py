from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from stanstock.data.jobs import execute_us_daily_job, prepare_us_daily_job
from stanstock.data.management.config_loader import default_us_universe_config_path
from stanstock.data.providers.exceptions import ProviderError


class Command(BaseCommand):
    help = (
        "Run the idempotent US Twelve Data close workflow: retrieve immutable "
        "daily prices, capture a dated universe snapshot, analyze it, and "
        "append predictions."
    )

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--region",
            required=True,
            choices=["us"],
            help="Only the approved US-first workflow is currently available.",
        )
        parser.add_argument(
            "--target-date",
            help=(
                "Completed US market session (YYYY-MM-DD). Defaults to the "
                "latest completed XNYS session."
            ),
        )
        parser.add_argument(
            "--config",
            type=Path,
            default=default_us_universe_config_path(),
            help="US universe YAML configuration.",
        )

    def handle(self, *args: object, **options: object) -> None:
        config_path = options["config"]
        if not isinstance(config_path, Path):
            raise CommandError("--config must be a filesystem path")
        explicit_target = _parse_target_date(options.get("target_date"))
        try:
            prepared = prepare_us_daily_job(
                config_path=config_path,
                explicit_target=explicit_target,
            )
            job_run = execute_us_daily_job(prepared)
        except (OSError, ProviderError, ValueError) as exc:
            raise CommandError(f"US daily workflow failed: {exc}") from exc

        self.stdout.write(
            self.style.SUCCESS(
                f"daily job_run={job_run.pk} status={job_run.status} "
                f"region=us target_date={prepared.target_date.isoformat()} "
                f"details={job_run.details!r}"
            )
        )


def _parse_target_date(raw: object) -> date | None:
    if raw is None:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError as exc:
        raise CommandError("--target-date must use YYYY-MM-DD") from exc

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from stanstock.data.management.config_loader import (
    default_sec_cik_mapping_path,
    default_sec_fundamentals_config_path,
    default_us_universe_config_path,
)
from stanstock.data.providers.exceptions import ProviderError
from stanstock.data.sec_jobs import execute_sec_fundamentals_job


class Command(BaseCommand):
    help = "Fetch and normalize point-in-time SEC facts for the configured US universe."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--target-date", type=date.fromisoformat)
        parser.add_argument(
            "--fundamentals-config",
            type=Path,
            default=default_sec_fundamentals_config_path(),
        )
        parser.add_argument(
            "--cik-config",
            type=Path,
            default=default_sec_cik_mapping_path(),
        )
        parser.add_argument(
            "--universe-config",
            type=Path,
            default=default_us_universe_config_path(),
        )

    def handle(self, *args: object, **options: object) -> None:
        target_date = options.get("target_date") or timezone.localdate()
        if not isinstance(target_date, date):
            raise CommandError("--target-date must be an ISO date")
        fundamentals_config = options.get("fundamentals_config")
        cik_config = options.get("cik_config")
        universe_config = options.get("universe_config")
        if not isinstance(fundamentals_config, Path):
            raise CommandError("--fundamentals-config must be a filesystem path")
        if not isinstance(cik_config, Path):
            raise CommandError("--cik-config must be a filesystem path")
        if not isinstance(universe_config, Path):
            raise CommandError("--universe-config must be a filesystem path")
        try:
            run = execute_sec_fundamentals_job(
                target_date=target_date,
                fundamentals_config_path=fundamentals_config,
                cik_config_path=cik_config,
                universe_config_path=universe_config,
            )
        except (OSError, ProviderError, ValueError) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(
            self.style.SUCCESS(
                f"SEC fundamentals job_run={run.pk} status={run.status} "
                f"target_date={target_date.isoformat()} details={run.details!r}"
            )
        )

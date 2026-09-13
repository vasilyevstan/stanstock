from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

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
        parser.add_argument(
            "--issuance-key",
            default="manual",
            help=(
                "Active-product research intake identity. Reuse it for retry; change it "
                "explicitly for a new same-target saved set. Never requests observed issuance."
            ),
        )

    def handle(self, *args: object, **options: object) -> None:
        config_path = options["config"]
        if not isinstance(config_path, Path):
            raise CommandError("--config must be a filesystem path")
        explicit_target = _parse_target_date(options.get("target_date"))
        if settings.RESEARCH_PRODUCT_ENABLED:
            from stanstock.core.research_product_refresh import resolve_scheduled_owner
            from stanstock.data.live_us import resolve_us_target_date
            from stanstock.data.research_product_jobs import execute_daily_research_job

            if options["issuance_key"] == "scheduled":
                raise CommandError("The scheduled issuance identity is reserved for automation")
            try:
                target, _grade = resolve_us_target_date(
                    decision_time=timezone.now(), explicit_target=explicit_target
                )
                job = execute_daily_research_job(
                    target_date=target,
                    owner=resolve_scheduled_owner(),
                    issuance_key=str(options["issuance_key"]),
                    issued_on_time=False,
                    core_config_path=config_path,
                )
            except (OSError, ProviderError, ValueError) as exc:
                raise CommandError(
                    "Manual research refresh failed; verify the target, authorization and evidence"
                ) from exc
            self.stdout.write(
                self.style.SUCCESS(
                    f"daily research-product-v1 job_run={job.pk} status={job.status} "
                    f"region=us target_date={target.isoformat()}"
                )
            )
            return
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

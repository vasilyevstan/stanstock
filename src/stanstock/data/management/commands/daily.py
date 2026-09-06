from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun
from stanstock.data.live_us import (
    load_us_universe_config,
    resolve_us_target_date,
    run_us_daily,
)
from stanstock.data.management.config_loader import default_us_universe_config_path
from stanstock.data.providers import twelve_data
from stanstock.data.providers.exceptions import ProviderError

JOB_NAME = "daily"


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
        decision_time = timezone.now()
        try:
            config = load_us_universe_config(config_path)
            target_date, snapshot_grade = resolve_us_target_date(
                decision_time=decision_time,
                explicit_target=explicit_target,
            )

            def _task(run: JobRun) -> JobExecutionResult:
                result = run_us_daily(
                    config=config,
                    target_date=target_date,
                    snapshot_grade=snapshot_grade,
                    decision_time=decision_time,
                )
                return JobExecutionResult(
                    details={
                        "snapshot_id": str(result.snapshot.id),
                        "snapshot_grade": result.snapshot.grade,
                        "provider": twelve_data.PROVIDER,
                        "benchmark_subject": result.benchmark_symbol,
                        "eligible": result.eligible,
                        "excluded": result.excluded,
                        "price_assets": result.price_assets,
                        "raw_assets": result.raw_assets,
                        "credits_used": result.credits_used,
                        "analyses": result.analyses,
                        "predictions": result.predictions,
                    }
                )

            job_run = execute_target_job(
                job_name=JOB_NAME,
                region=str(options["region"]),
                target_date=target_date,
                task=_task,
            )
        except (OSError, ProviderError, ValueError) as exc:
            raise CommandError(f"US daily workflow failed: {exc}") from exc

        self.stdout.write(
            self.style.SUCCESS(
                f"daily job_run={job_run.pk} status={job_run.status} "
                f"region=us target_date={target_date.isoformat()} "
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

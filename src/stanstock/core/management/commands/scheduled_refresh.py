from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.launchd import detect_iana_timezone
from stanstock.core.models import JobRun
from stanstock.core.refresh_verification import verify_scheduled_refresh
from stanstock.core.revision import clean_git_revision
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.jobs import (
    PreparedUsDailyJob,
    execute_us_daily_job,
    frozen_long_forecast_gate_for_run,
    prepare_us_daily_job,
    proposed_us_daily_target_gate,
    reserve_us_daily_target_gate,
)
from stanstock.data.management.config_loader import default_us_universe_config_path
from stanstock.data.providers import twelve_data
from stanstock.data.providers.exceptions import ProviderError
from stanstock.data.sec_jobs import JOB_NAME as SEC_JOB_NAME
from stanstock.data.sec_jobs import execute_sec_fundamentals_job
from stanstock.portfolio.jobs import execute_portfolio_snapshot_job
from stanstock.research.jobs import execute_prediction_evaluation_job

JOB_NAME = "scheduled_refresh"
SATISFIED_STAGE_STATUSES = frozenset({JobRun.Status.SUCCESS, JobRun.Status.SKIPPED})
EXPECTED_STAGE_ERRORS = (OSError, ProviderError, ValueError)


class Command(BaseCommand):
    help = (
        "Run the recoverable post-market US refresh used by the macOS LaunchAgent. "
        "Late automatic research-grade catch-up is refused."
    )

    def add_arguments(self, parser: Any) -> None:
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
        decision_time = timezone.now()
        try:
            _validate_runtime_timezone()
            prepared = prepare_us_daily_job(
                config_path=config_path,
                decision_time=decision_time,
            )

            def _task(parent: JobRun) -> JobExecutionResult:
                details: dict[str, object] = {
                    "target_date": prepared.target_date.isoformat(),
                    "snapshot_grade": prepared.snapshot_grade,
                    "stages": {},
                }
                revision = clean_git_revision(Path(settings.BASE_DIR))
                os.environ["STANSTOCK_CODE_REVISION"] = revision
                details["code_revision"] = revision

                proposed_gate = proposed_us_daily_target_gate(prepared)
                target_gate = reserve_us_daily_target_gate(
                    prepared,
                    explicit_long_forecast_requested=proposed_gate,
                    reservation_owner=str(parent.pk),
                )
                long_forecast_requested = target_gate.long_forecast_requested
                market: JobRun | None = None
                if target_gate.market_output_committed:
                    market = _run_market_stage(
                        parent=parent,
                        details=details,
                        prepared=prepared,
                        long_forecast_requested=long_forecast_requested,
                        target_gate_reservation_id=target_gate.reservation_id,
                    )
                    _require_market_gate(market, long_forecast_requested)
                recoverable_sec_success = JobRun.objects.filter(
                    job_name=SEC_JOB_NAME,
                    region="us",
                    target_date=prepared.target_date,
                    status=JobRun.Status.SUCCESS,
                ).exists()
                sec_required = long_forecast_requested or recoverable_sec_success
                if sec_required:
                    if target_gate.market_output_committed and not recoverable_sec_success:
                        raise ValueError(
                            "Committed market output requires SEC evidence, but no "
                            "same-target successful SEC child can be recovered locally"
                        )
                    sec_run = _run_stage(
                        parent=parent,
                        details=details,
                        stage_name="sec_fundamentals",
                        job_name=SEC_JOB_NAME,
                        region="us",
                        target_date=prepared.target_date,
                        task=lambda: execute_sec_fundamentals_job(
                            target_date=prepared.target_date,
                            universe_config_path=config_path,
                        ),
                    )
                    if sec_run is None or sec_run.status not in SATISFIED_STAGE_STATUSES:
                        raise ValueError("Required SEC fundamentals stage was not satisfied")

                if market is None:
                    market = _run_market_stage(
                        parent=parent,
                        details=details,
                        prepared=prepared,
                        long_forecast_requested=long_forecast_requested,
                        target_gate_reservation_id=target_gate.reservation_id,
                    )
                    _require_market_gate(market, long_forecast_requested)

                failures: list[str] = []
                evaluation = _run_stage(
                    parent=parent,
                    details=details,
                    stage_name="evaluation",
                    job_name="evaluate_predictions",
                    region="us",
                    target_date=prepared.target_date,
                    task=lambda: execute_prediction_evaluation_job(
                        provider=twelve_data.PROVIDER,
                        evaluation_date=prepared.target_date,
                        evaluation_time=timezone.now(),
                        benchmark_subject=prepared.config.benchmark_symbol,
                    ),
                    failures=failures,
                )
                snapshots = _run_stage(
                    parent=parent,
                    details=details,
                    stage_name="portfolio_snapshots",
                    job_name="scheduled_portfolio_snapshots",
                    region="",
                    target_date=prepared.target_date,
                    task=lambda: execute_portfolio_snapshot_job(
                        target_date=prepared.target_date,
                        require_session_date=True,
                        require_all=True,
                    ),
                    failures=failures,
                )
                for stage_name, run in (
                    ("evaluation", evaluation),
                    ("portfolio_snapshots", snapshots),
                ):
                    if run is not None and run.status not in SATISFIED_STAGE_STATUSES:
                        failures.append(f"{stage_name} ended with {run.status}")
                if failures:
                    raise ValueError("Scheduled refresh incomplete: " + "; ".join(failures))
                stages = details["stages"]
                if not isinstance(stages, dict):
                    raise ValueError("Scheduled refresh stage state is invalid")
                try:
                    verification = verify_scheduled_refresh(
                        target_date=prepared.target_date,
                        universe_config=prepared.config,
                        code_revision=revision,
                        stages=stages,
                        sec_required=sec_required,
                    )
                except RefreshVerificationError as exc:
                    details["verification"] = exc.to_failure_details()
                    JobRun.objects.filter(pk=parent.pk).update(details=details)
                    raise ValueError(f"Scheduled refresh verification failed: {exc}") from exc
                details["verification"] = verification
                return JobExecutionResult(details=details)

            parent = execute_target_job(
                job_name=JOB_NAME,
                region="us",
                target_date=prepared.target_date,
                task=_task,
            )
        except EXPECTED_STAGE_ERRORS as exc:
            raise CommandError(f"Scheduled refresh failed: {exc}") from exc

        self.stdout.write(
            self.style.SUCCESS(
                f"scheduled_refresh job_run={parent.pk} status={parent.status} "
                f"target_date={prepared.target_date.isoformat()} details={parent.details!r}"
            )
        )


def _run_stage(
    *,
    parent: JobRun,
    details: dict[str, object],
    stage_name: str,
    job_name: str,
    region: str,
    target_date: date,
    task: Callable[[], JobRun],
    failures: list[str] | None = None,
) -> JobRun | None:
    try:
        run = task()
    except EXPECTED_STAGE_ERRORS as exc:
        failed_run = (
            JobRun.objects.filter(
                job_name=job_name,
                region=region,
                target_date=target_date,
            )
            .order_by("-attempt")
            .first()
        )
        _record_stage(
            parent,
            details,
            stage_name,
            failed_run,
            error=f"{type(exc).__name__}: {exc}",
        )
        if failures is None:
            raise
        failures.append(f"{stage_name}: {exc}")
        return failed_run
    _record_stage(parent, details, stage_name, run)
    return run


def _run_market_stage(
    *,
    parent: JobRun,
    details: dict[str, object],
    prepared: PreparedUsDailyJob,
    long_forecast_requested: bool,
    target_gate_reservation_id: UUID | None,
) -> JobRun | None:
    return _run_stage(
        parent=parent,
        details=details,
        stage_name="market",
        job_name="daily",
        region="us",
        target_date=prepared.target_date,
        task=lambda: execute_us_daily_job(
            prepared,
            require_observed=True,
            long_forecast_requested=long_forecast_requested,
            target_gate_reservation_id=target_gate_reservation_id,
        ),
    )


def _require_market_gate(market: JobRun | None, frozen_gate: bool) -> None:
    if market is None or market.status not in SATISFIED_STAGE_STATUSES:
        raise ValueError("Required market stage was not satisfied")
    authoritative_market_gate = frozen_long_forecast_gate_for_run(market)
    if authoritative_market_gate != frozen_gate:
        raise ValueError(
            "Authoritative market child gate does not match the scheduler's frozen target gate"
        )


def _record_stage(
    parent: JobRun,
    details: dict[str, object],
    stage_name: str,
    run: JobRun | None,
    *,
    error: str = "",
) -> None:
    stages = details["stages"]
    if not isinstance(stages, dict):
        raise ValueError("Scheduled refresh stage state is invalid")
    stages[stage_name] = {
        "job_run_id": str(run.pk) if run is not None else None,
        "status": run.status if run is not None else JobRun.Status.FAILED,
        "attempt": run.attempt if run is not None else None,
        "error": error,
    }
    JobRun.objects.filter(pk=parent.pk).update(details=details)


def _validate_runtime_timezone() -> None:
    expected = os.environ.get("STANSTOCK_SCHEDULE_TIMEZONE", "").strip()
    if not expected:
        return
    current = detect_iana_timezone()
    if current != expected:
        raise ValueError(
            f"Installed schedule timezone is {expected}, but the machine now reports "
            f"{current}. Reinstall the LaunchAgent after validating the new timezone."
        )

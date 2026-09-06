from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from django.utils import timezone

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun
from stanstock.data.live_us import (
    UsUniverseConfig,
    load_us_universe_config,
    resolve_us_target_date,
    run_us_daily,
)
from stanstock.data.models import UniverseSnapshot
from stanstock.data.providers import twelve_data

JOB_NAME = "daily"


@dataclass(frozen=True, slots=True)
class PreparedUsDailyJob:
    config: UsUniverseConfig
    target_date: date
    snapshot_grade: str
    decision_time: datetime


def prepare_us_daily_job(
    *,
    config_path: Path,
    explicit_target: date | None = None,
    decision_time: datetime | None = None,
) -> PreparedUsDailyJob:
    effective_time = decision_time or timezone.now()
    config = load_us_universe_config(config_path)
    target_date, snapshot_grade = resolve_us_target_date(
        decision_time=effective_time,
        explicit_target=explicit_target,
    )
    return PreparedUsDailyJob(
        config=config,
        target_date=target_date,
        snapshot_grade=snapshot_grade,
        decision_time=effective_time,
    )


def execute_us_daily_job(
    prepared: PreparedUsDailyJob,
    *,
    require_observed: bool = False,
) -> JobRun:
    def _task(run: JobRun) -> JobExecutionResult:
        if require_observed and prepared.snapshot_grade != UniverseSnapshot.Grade.OBSERVED:
            raise ValueError(
                f"Automatic research-grade catch-up is forbidden for "
                f"{prepared.target_date.isoformat()}; run an explicit manual "
                "target-date reconstruction instead."
            )
        result = run_us_daily(
            config=prepared.config,
            target_date=prepared.target_date,
            snapshot_grade=prepared.snapshot_grade,
            decision_time=prepared.decision_time,
            require_on_time=require_observed,
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

    return execute_target_job(
        job_name=JOB_NAME,
        region="us",
        target_date=prepared.target_date,
        task=_task,
    )

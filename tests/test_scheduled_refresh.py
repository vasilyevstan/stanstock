from __future__ import annotations

import os
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.management.commands import scheduled_refresh
from stanstock.core.models import JobRun
from stanstock.data.jobs import PreparedUsDailyJob, execute_us_daily_job
from stanstock.data.live_us import UsUniverseConfig
from stanstock.data.models import UniverseSnapshot

pytestmark = pytest.mark.django_db


def _prepared(target: date = date(2026, 9, 4)) -> PreparedUsDailyJob:
    config = cast(
        UsUniverseConfig,
        SimpleNamespace(benchmark_symbol="SPY"),
    )
    return PreparedUsDailyJob(
        config=config,
        target_date=target,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        decision_time=datetime(2026, 9, 5, 6, tzinfo=UTC),
    )


def _successful_child(job_name: str, region: str, target_date: date) -> JobRun:
    return execute_target_job(
        job_name=job_name,
        region=region,
        target_date=target_date,
        task=lambda run: JobExecutionResult(),
    )


def test_scheduled_refresh_records_recoverable_child_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    clean_checks: list[Path] = []
    evaluation_attempts = 0
    evaluation_times: list[datetime] = []

    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(
        scheduled_refresh,
        "prepare_us_daily_job",
        lambda **kwargs: prepared,
    )

    def clean_revision(root: Path) -> str:
        clean_checks.append(root)
        return "a" * 40

    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", clean_revision)
    monkeypatch.setattr(
        scheduled_refresh,
        "execute_us_daily_job",
        lambda *args, **kwargs: _successful_child("daily", "us", prepared.target_date),
    )

    def evaluate() -> JobRun:
        nonlocal evaluation_attempts
        evaluation_attempts += 1

        def task(run: JobRun) -> JobExecutionResult:
            if evaluation_attempts == 1:
                raise ValueError("temporary evaluation failure")
            return JobExecutionResult()

        return execute_target_job(
            job_name="evaluate_predictions",
            region="us",
            target_date=prepared.target_date,
            task=task,
        )

    def execute_evaluation(**kwargs: object) -> JobRun:
        evaluation_time = kwargs["evaluation_time"]
        assert isinstance(evaluation_time, datetime)
        evaluation_times.append(evaluation_time)
        return evaluate()

    monkeypatch.setattr(
        scheduled_refresh,
        "execute_prediction_evaluation_job",
        execute_evaluation,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "execute_portfolio_snapshot_job",
        lambda **kwargs: _successful_child(
            "scheduled_portfolio_snapshots",
            "",
            prepared.target_date,
        ),
    )

    with pytest.raises(CommandError, match="temporary evaluation failure"):
        call_command(
            "scheduled_refresh",
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )

    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    assert first_parent.status == JobRun.Status.FAILED
    assert first_parent.details["stages"]["market"]["status"] == JobRun.Status.SUCCESS
    assert first_parent.details["stages"]["evaluation"]["status"] == JobRun.Status.FAILED
    assert first_parent.details["stages"]["portfolio_snapshots"]["status"] == JobRun.Status.SUCCESS

    call_command(
        "scheduled_refresh",
        config=tmp_path / "universe.yml",
        stdout=StringIO(),
    )

    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=2)
    assert second_parent.status == JobRun.Status.SUCCESS
    assert second_parent.details["stages"]["market"]["status"] == JobRun.Status.SKIPPED
    assert second_parent.details["stages"]["evaluation"]["status"] == JobRun.Status.SUCCESS
    assert second_parent.details["stages"]["portfolio_snapshots"]["status"] == JobRun.Status.SKIPPED
    assert len(clean_checks) == 2
    assert len(evaluation_times) == 2
    assert all(value != prepared.decision_time for value in evaluation_times)
    assert os.environ["STANSTOCK_CODE_REVISION"] == "a" * 40


def test_scheduled_refresh_rejects_changed_machine_timezone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("STANSTOCK_SCHEDULE_TIMEZONE", "America/New_York")
    monkeypatch.setattr(
        scheduled_refresh,
        "detect_iana_timezone",
        lambda: "America/Los_Angeles",
    )

    with pytest.raises(CommandError, match="Reinstall the LaunchAgent"):
        call_command(
            "scheduled_refresh",
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )

    assert JobRun.objects.count() == 0


def test_automatic_daily_child_refuses_late_research_grade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = PreparedUsDailyJob(
        config=cast(UsUniverseConfig, SimpleNamespace(benchmark_symbol="SPY")),
        target_date=date(2026, 9, 4),
        snapshot_grade=UniverseSnapshot.Grade.RESEARCH,
        decision_time=datetime(2026, 9, 8, 15, tzinfo=UTC),
    )
    monkeypatch.setattr(
        "stanstock.data.jobs.run_us_daily",
        lambda **kwargs: pytest.fail("provider work must not start for a late run"),
    )

    with pytest.raises(ValueError, match="Automatic research-grade catch-up is forbidden"):
        execute_us_daily_job(prepared, require_observed=True)

    failed = JobRun.objects.get(job_name="daily")
    assert failed.status == JobRun.Status.FAILED

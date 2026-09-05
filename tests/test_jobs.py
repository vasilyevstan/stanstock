from __future__ import annotations

from datetime import date

import pytest

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun


@pytest.mark.django_db
def test_successful_target_job_is_not_executed_twice() -> None:
    calls: list[str] = []

    def task(run: JobRun) -> JobExecutionResult:
        calls.append(str(run.pk))
        return JobExecutionResult(details={"rows": 12})

    first = execute_target_job(
        job_name="daily",
        region="us",
        target_date=date(2026, 9, 4),
        task=task,
    )
    second = execute_target_job(
        job_name="daily",
        region="us",
        target_date=date(2026, 9, 4),
        task=task,
    )

    assert first.status == JobRun.Status.SUCCESS
    assert second.status == JobRun.Status.SKIPPED
    assert second.details["successful_run_id"] == str(first.pk)
    assert len(calls) == 1


@pytest.mark.django_db
def test_failed_target_job_records_error_and_can_retry() -> None:
    def failing_task(run: JobRun) -> JobExecutionResult:
        raise RuntimeError(f"failed {run.attempt}")

    with pytest.raises(RuntimeError, match="failed 1"):
        execute_target_job(
            job_name="daily",
            region="europe",
            target_date=date(2026, 9, 4),
            task=failing_task,
        )

    failed = JobRun.objects.get()
    assert failed.status == JobRun.Status.FAILED
    assert failed.finished_at is not None
    assert failed.error == "RuntimeError: failed 1"

    retry = execute_target_job(
        job_name="daily",
        region="europe",
        target_date=date(2026, 9, 4),
        task=lambda run: JobExecutionResult(status=JobRun.Status.NO_DATA),
    )

    assert retry.attempt == 2
    assert retry.status == JobRun.Status.NO_DATA

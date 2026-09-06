from __future__ import annotations

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

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


@pytest.mark.django_db(transaction=True)
def test_local_target_lock_serializes_concurrent_attempts() -> None:
    active_tasks = 0
    maximum_active_tasks = 0
    guard = threading.Lock()

    def invoke() -> str:
        def task(run: JobRun) -> JobExecutionResult:
            nonlocal active_tasks, maximum_active_tasks
            with guard:
                active_tasks += 1
                maximum_active_tasks = max(maximum_active_tasks, active_tasks)
            time.sleep(0.05)
            with guard:
                active_tasks -= 1
            return JobExecutionResult()

        run = execute_target_job(
            job_name="concurrent",
            region="us",
            target_date=date(2026, 9, 4),
            task=task,
        )
        return str(run.status)

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = sorted(executor.map(lambda _: invoke(), range(2)))

    assert statuses == sorted([JobRun.Status.SKIPPED, JobRun.Status.SUCCESS])
    assert maximum_active_tasks == 1


def test_sqlite_target_lock_serializes_independent_processes(tmp_path: Path) -> None:
    database_path = str(tmp_path / "stanstock.sqlite3")
    first_marker = tmp_path / "first-entered"
    second_marker = tmp_path / "second-entered"
    script = (
        "import os,sys,time,django;"
        "os.environ.setdefault('DJANGO_SETTINGS_MODULE','stanstock.settings.test');"
        "django.setup();"
        "from pathlib import Path;"
        "from stanstock.core.jobs import _sqlite_file_lock;"
        "database=Path(sys.argv[1]);marker=Path(sys.argv[2]);hold=float(sys.argv[3]);"
        "lock=_sqlite_file_lock('daily:us:2026-09-04',database);"
        "lock.__enter__();marker.write_text('entered',encoding='utf-8');"
        "time.sleep(hold);lock.__exit__(None,None,None)"
    )
    first = subprocess.Popen(
        [sys.executable, "-c", script, database_path, str(first_marker), "0.6"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not first_marker.exists() and first.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert first_marker.exists(), first.stderr.read() if first.stderr is not None else ""

    second = subprocess.Popen(
        [sys.executable, "-c", script, database_path, str(second_marker), "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.2)
    assert second_marker.exists() is False
    first_stdout, first_stderr = first.communicate(timeout=5)
    second_stdout, second_stderr = second.communicate(timeout=5)

    assert first.returncode == 0, first_stdout + first_stderr
    assert second.returncode == 0, second_stdout + second_stderr
    assert second_marker.exists()

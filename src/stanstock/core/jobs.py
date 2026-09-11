from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from django.db import connection, transaction
from django.db.models import Max
from django.utils import timezone

from stanstock.core.models import JobRun

logger = logging.getLogger(__name__)

_LOCAL_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True, slots=True)
class JobExecutionResult:
    status: str = JobRun.Status.SUCCESS
    details: dict[str, Any] = field(default_factory=dict)


JobTask = Callable[[JobRun], JobExecutionResult]
BeforeAttempt = Callable[[], JobRun | None]


def execute_target_job(
    *,
    job_name: str,
    region: str,
    target_date: date,
    task: JobTask,
    before_attempt: BeforeAttempt | None = None,
) -> JobRun:
    with target_job_lock(
        job_name=job_name,
        region=region,
        target_date=target_date,
    ):
        reserved_run = before_attempt() if before_attempt is not None else None
        with transaction.atomic():
            prior_success = (
                JobRun.objects.select_for_update()
                .filter(
                    job_name=job_name,
                    region=region,
                    target_date=target_date,
                    status=JobRun.Status.SUCCESS,
                )
                .first()
            )
            next_attempt = _next_attempt(job_name, region, target_date)
            if prior_success is not None:
                skipped = JobRun.objects.create(
                    job_name=job_name,
                    region=region,
                    target_date=target_date,
                    attempt=next_attempt,
                    status=JobRun.Status.SKIPPED,
                    finished_at=timezone.now(),
                    details={
                        "reason": "target_already_succeeded",
                        "successful_run_id": str(prior_success.pk),
                    },
                )
                logger.info(
                    "target_job_skipped job=%s region=%s target_date=%s reason=already_succeeded",
                    job_name,
                    region,
                    target_date,
                )
                return skipped

            running_attempts = JobRun.objects.select_for_update().filter(
                job_name=job_name,
                region=region,
                target_date=target_date,
                status=JobRun.Status.RUNNING,
            )
            if reserved_run is not None:
                running_attempts = running_attempts.exclude(pk=reserved_run.pk)
            running_attempts.update(
                status=JobRun.Status.FAILED,
                finished_at=timezone.now(),
                error="Stale running attempt superseded by a new target lock holder",
            )

            if reserved_run is None:
                run = JobRun.objects.create(
                    job_name=job_name,
                    region=region,
                    target_date=target_date,
                    attempt=next_attempt,
                )
            else:
                reserved_candidate = (
                    JobRun.objects.select_for_update()
                    .filter(
                        pk=reserved_run.pk,
                        job_name=job_name,
                        region=region,
                        target_date=target_date,
                        status=JobRun.Status.RUNNING,
                    )
                    .first()
                )
                if reserved_candidate is None:
                    raise ValueError(
                        "Target job reservation is not a running attempt for this target"
                    )
                run = reserved_candidate
                next_attempt = run.attempt
            logger.info(
                "target_job_started job=%s region=%s target_date=%s attempt=%s",
                job_name,
                region,
                target_date,
                next_attempt,
            )

        try:
            result = task(run)
            if result.status not in {
                JobRun.Status.SUCCESS,
                JobRun.Status.NO_DATA,
                JobRun.Status.SKIPPED,
            }:
                raise ValueError(f"Unsupported terminal job status: {result.status}")
            JobRun.objects.filter(pk=run.pk).update(
                status=result.status,
                finished_at=timezone.now(),
                details=result.details,
                error="",
            )
        except Exception as exc:
            JobRun.objects.filter(pk=run.pk).update(
                status=JobRun.Status.FAILED,
                finished_at=timezone.now(),
                error=f"{exc.__class__.__name__}: {exc}",
            )
            logger.exception(
                "target_job_failed job=%s region=%s target_date=%s attempt=%s",
                job_name,
                region,
                target_date,
                next_attempt,
            )
            raise

        run.refresh_from_db()
        logger.info(
            "target_job_finished job=%s region=%s target_date=%s attempt=%s status=%s",
            job_name,
            region,
            target_date,
            next_attempt,
            run.status,
        )
        return run


def _next_attempt(job_name: str, region: str, target_date: date) -> int:
    latest = JobRun.objects.filter(
        job_name=job_name,
        region=region,
        target_date=target_date,
    ).aggregate(max_attempt=Max("attempt"))["max_attempt"]
    return int(latest or 0) + 1


@contextmanager
def target_job_lock(
    *,
    job_name: str,
    region: str,
    target_date: date,
) -> Iterator[None]:
    """Hold the canonical cross-process lock for one target job identity."""
    target_key = f"{job_name}:{region}:{target_date.isoformat()}"
    with _target_lock(target_key):
        yield


@contextmanager
def _target_lock(target_key: str) -> Iterator[None]:
    if connection.vendor == "postgresql":
        lock_id = int.from_bytes(
            hashlib.blake2b(target_key.encode(), digest_size=8).digest(),
            byteorder="big",
            signed=True,
        )
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_lock(%s)", [lock_id])
        try:
            yield
        finally:
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_unlock(%s)", [lock_id])
        return

    with _LOCAL_LOCKS_GUARD:
        lock = _LOCAL_LOCKS.setdefault(target_key, threading.Lock())
    with lock:
        if connection.vendor == "sqlite":
            with _sqlite_process_lock(target_key):
                yield
            return
        yield


@contextmanager
def _sqlite_process_lock(target_key: str) -> Iterator[None]:
    database_name = connection.settings_dict.get("NAME")
    if not isinstance(database_name, (str, Path)):
        yield
        return
    raw_name = str(database_name)
    if raw_name == ":memory:" or raw_name.startswith("file:"):
        yield
        return
    with _sqlite_file_lock(target_key, Path(raw_name)):
        yield


@contextmanager
def _sqlite_file_lock(target_key: str, database_path: Path) -> Iterator[None]:
    lock_directory = database_path.resolve().parent / ".stanstock-job-locks"
    lock_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(lock_directory, 0o700)
    filename = hashlib.sha256(target_key.encode()).hexdigest() + ".lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_directory / filename, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

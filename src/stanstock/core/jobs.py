from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
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


def execute_target_job(
    *,
    job_name: str,
    region: str,
    target_date: date,
    task: JobTask,
) -> JobRun:
    target_key = f"{job_name}:{region}:{target_date.isoformat()}"
    with _target_lock(target_key):
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

            JobRun.objects.select_for_update().filter(
                job_name=job_name,
                region=region,
                target_date=target_date,
                status=JobRun.Status.RUNNING,
            ).update(
                status=JobRun.Status.FAILED,
                finished_at=timezone.now(),
                error="Stale running attempt superseded by a new target lock holder",
            )

            run = JobRun.objects.create(
                job_name=job_name,
                region=region,
                target_date=target_date,
                attempt=next_attempt,
            )
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
        yield

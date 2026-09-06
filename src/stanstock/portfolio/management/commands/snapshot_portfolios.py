from __future__ import annotations

from datetime import date
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun
from stanstock.portfolio.models import Portfolio
from stanstock.portfolio.service import snapshot_all_portfolios

JOB_NAME = "snapshot_portfolios"


class Command(BaseCommand):
    help = (
        "Record immutable valuation snapshots for every active tracked portfolio. "
        "Run this after market-data refreshes."
    )

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--target-date",
            help=(
                "Operational job date (YYYY-MM-DD); defaults to the local date. "
                "Each snapshot uses its latest persisted market session."
            ),
        )

    def handle(self, *args: object, **options: object) -> None:
        target_date = _parse_target_date(options.get("target_date")) or timezone.localdate()

        def _task(run: JobRun) -> JobExecutionResult:
            report = snapshot_all_portfolios()
            portfolio_count = Portfolio.objects.filter(archived_at__isnull=True).count()
            completed_count = report.created + report.unchanged
            if report.failures and completed_count == 0:
                raise ValueError("; ".join(report.failures))
            return JobExecutionResult(
                status=(JobRun.Status.NO_DATA if portfolio_count == 0 else JobRun.Status.SUCCESS),
                details={
                    "portfolios": portfolio_count,
                    "snapshots_created": report.created,
                    "snapshots_unchanged": report.unchanged,
                    "failures": list(report.failures),
                },
            )

        try:
            run = execute_target_job(
                job_name=JOB_NAME,
                region="",
                target_date=target_date,
                task=_task,
            )
        except ValueError as exc:
            raise CommandError(f"Portfolio snapshot job failed: {exc}") from exc

        self.stdout.write(
            self.style.SUCCESS(
                f"snapshot_portfolios job_run={run.pk} status={run.status} "
                f"target_date={target_date.isoformat()} details={run.details!r}"
            )
        )


def _parse_target_date(raw: object) -> date | None:
    if raw is None:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError as exc:
        raise CommandError("--target-date must use YYYY-MM-DD") from exc

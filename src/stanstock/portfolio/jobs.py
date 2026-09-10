from __future__ import annotations

from datetime import date

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun
from stanstock.portfolio.models import Portfolio
from stanstock.portfolio.service import snapshot_all_portfolios

JOB_NAME = "snapshot_portfolios"
SCHEDULED_JOB_NAME = "scheduled_portfolio_snapshots"


def execute_portfolio_snapshot_job(
    *,
    target_date: date,
    require_session_date: bool = False,
    require_all: bool = False,
) -> JobRun:
    def _task(run: JobRun) -> JobExecutionResult:
        report = snapshot_all_portfolios(
            expected_as_of_date=target_date if require_session_date else None
        )
        portfolio_count = Portfolio.objects.filter(archived_at__isnull=True).count()
        completed_count = report.created + report.unchanged
        if report.failures and (completed_count == 0 or require_all):
            raise ValueError("; ".join(report.failures))
        return JobExecutionResult(
            status=(JobRun.Status.SKIPPED if portfolio_count == 0 else JobRun.Status.SUCCESS),
            details={
                "portfolios": portfolio_count,
                "snapshots_created": report.created,
                "snapshots_unchanged": report.unchanged,
                "snapshot_ids": dict(report.snapshot_ids),
                "failures": list(report.failures),
                "required_session_date": (
                    target_date.isoformat() if require_session_date else None
                ),
                "require_all": require_all,
                "reason": "no_active_portfolios" if portfolio_count == 0 else "",
            },
        )

    return execute_target_job(
        job_name=(SCHEDULED_JOB_NAME if require_session_date or require_all else JOB_NAME),
        region="",
        target_date=target_date,
        task=_task,
    )

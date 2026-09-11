from __future__ import annotations

from datetime import date

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun
from stanstock.portfolio.models import Portfolio
from stanstock.portfolio.refresh_validation import (
    attest_scheduled_portfolio_snapshots,
)
from stanstock.portfolio.service import snapshot_all_portfolios
from stanstock.research.config import code_revision

JOB_NAME = "snapshot_portfolios"
SCHEDULED_JOB_NAME = "scheduled_portfolio_snapshots"


def execute_portfolio_snapshot_job(
    *,
    target_date: date,
    require_session_date: bool = False,
    require_all: bool = False,
) -> JobRun:
    def _task(run: JobRun) -> JobExecutionResult:
        validator_revision = code_revision()
        report = snapshot_all_portfolios(
            expected_as_of_date=target_date if require_session_date else None
        )
        portfolio_count = Portfolio.objects.filter(archived_at__isnull=True).count()
        completed_count = report.created + report.unchanged
        if report.failures and (completed_count == 0 or require_all):
            if require_session_date and require_all:
                raise ValueError(
                    "Scheduled portfolio snapshots failed for "
                    f"{portfolio_count} active portfolio(s)"
                )
            raise ValueError("; ".join(report.failures))
        details: dict[str, object] = {
            "portfolios": portfolio_count,
            "snapshots_created": report.created,
            "snapshots_unchanged": report.unchanged,
            "snapshot_ids": dict(report.snapshot_ids),
            "failures": list(report.failures),
            "required_session_date": (target_date.isoformat() if require_session_date else None),
            "require_all": require_all,
            "reason": "no_active_portfolios" if portfolio_count == 0 else "",
        }
        if require_session_date and require_all:
            if portfolio_count == 0 and (completed_count != 0 or report.snapshot_ids):
                raise ValueError("Scheduled portfolio active set changed during snapshotting")
            if portfolio_count > 0:
                verification_asset = attest_scheduled_portfolio_snapshots(
                    child_run=run,
                    target_date=target_date,
                    report_details=details,
                    snapshot_actions=report._snapshot_actions,
                    validator_revision=validator_revision,
                )
                details["verification_asset_id"] = str(verification_asset.pk)
                details["verification_sha256"] = verification_asset.sha256
        return JobExecutionResult(
            status=(JobRun.Status.SKIPPED if portfolio_count == 0 else JobRun.Status.SUCCESS),
            details=details,
        )

    return execute_target_job(
        job_name=(SCHEDULED_JOB_NAME if require_session_date or require_all else JOB_NAME),
        region="",
        target_date=target_date,
        task=_task,
    )

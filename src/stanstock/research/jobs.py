from __future__ import annotations

from datetime import date, datetime

from django.db.models import Q
from django.utils import timezone
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun
from stanstock.research.models import Prediction, PredictionOutcome
from stanstock.research.outcomes import HORIZON_SESSION_COUNTS, evaluate_predictions
from stanstock.research.provenance import source_providers

JOB_NAME = "evaluate_predictions"
TERMINAL_OUTCOME_STATUSES = (
    PredictionOutcome.Status.MATURED,
    PredictionOutcome.Status.CORPORATE_EVENT,
)


def eligible_pending_predictions(
    *,
    provider: str,
    evaluation_date: date,
) -> list[Prediction]:
    maturity_filter = Q(pk__in=[])
    calendar = get_calendar("XNYS")
    evaluation_session = calendar.date_to_session(evaluation_date, direction="previous")
    for horizon, session_count in HORIZON_SESSION_COUNTS.items():
        latest_mature_target = calendar.session_offset(evaluation_session, -session_count).date()
        maturity_filter |= Q(horizon=horizon, target_date__lte=latest_mature_target)

    candidates = (
        Prediction.objects.select_related("listing")
        .filter(
            maturity_filter,
            evidence_role__in=Prediction.EvidenceRole.values,
        )
        .filter(Q(price_provider=provider) | Q(price_provider=""))
        .exclude(outcome__status__in=TERMINAL_OUTCOME_STATUSES)
        .order_by("generated_at", "id")
    )
    return [
        prediction
        for prediction in candidates
        if prediction.price_provider == provider
        or (
            not prediction.price_provider
            and provider in source_providers({"source_assets": prediction.source_assets})
        )
    ]


def execute_prediction_evaluation_job(
    *,
    provider: str,
    evaluation_date: date,
    benchmark_subject: str | None,
    evaluation_time: datetime | None = None,
) -> JobRun:
    effective_time = evaluation_time or timezone.now()

    def _task(run: JobRun) -> JobExecutionResult:
        predictions = eligible_pending_predictions(
            provider=provider,
            evaluation_date=evaluation_date,
        )
        results = evaluate_predictions(
            predictions,
            provider=provider,
            evaluation_date=evaluation_date,
            evaluation_time=effective_time,
            benchmark_subject=benchmark_subject,
        )
        counts: dict[str, int] = {}
        outcome_statuses: dict[str, int] = {}
        for result in results:
            counts[result.action] = counts.get(result.action, 0) + 1
            status = str(result.outcome.status)
            outcome_statuses[status] = outcome_statuses.get(status, 0) + 1
        return JobExecutionResult(
            status=JobRun.Status.SUCCESS,
            details={
                "provider": provider,
                "benchmark_subject": benchmark_subject,
                "eligible_predictions": len(predictions),
                "actions": counts,
                "outcome_statuses": outcome_statuses,
            },
        )

    return execute_target_job(
        job_name=JOB_NAME,
        region="us",
        target_date=evaluation_date,
        task=_task,
    )

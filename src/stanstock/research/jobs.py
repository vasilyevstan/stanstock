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


def maturity_provider_candidates(
    *,
    provider: str,
    evaluation_date: date,
    as_of: datetime | None = None,
) -> list[Prediction]:
    """Return every prediction that is mature and provider-matching as of
    `evaluation_date`, regardless of whether it already carries a terminal
    outcome.

    This is the full, deterministic pre-evaluation candidate set: it is the
    authoritative denominator both for `eligible_pending_predictions` (which
    additionally excludes already-terminal rows) and for scheduled-refresh
    output verification, which must independently re-derive "what should
    have been evaluated" rather than trust a job's self-reported count.

    `as_of`, when given, additionally excludes any prediction generated
    after that instant. This makes the candidate set truly *pre-child*: a
    prediction created after an evaluation execution (by a later run, or a
    concurrent reissue) must not retroactively appear to have been missed
    by it, while a mature prediction that already existed by `as_of` still
    counts against a fabricated zero.
    """
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
    )
    if as_of is not None:
        candidates = candidates.filter(generated_at__lte=as_of)
    candidates = candidates.order_by("generated_at", "id")
    return [
        prediction
        for prediction in candidates
        if prediction.price_provider == provider
        or (
            not prediction.price_provider
            and provider in source_providers({"source_assets": prediction.source_assets})
        )
    ]


def eligible_pending_predictions(
    *,
    provider: str,
    evaluation_date: date,
    as_of: datetime | None = None,
) -> list[Prediction]:
    candidates = maturity_provider_candidates(
        provider=provider, evaluation_date=evaluation_date, as_of=as_of
    )
    pending_ids = (
        Prediction.objects.filter(pk__in=[candidate.pk for candidate in candidates])
        .exclude(outcome__status__in=TERMINAL_OUTCOME_STATUSES)
        .values_list("pk", flat=True)
    )
    pending_id_set = set(pending_ids)
    return [candidate for candidate in candidates if candidate.pk in pending_id_set]


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
            as_of=effective_time,
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
        evaluated_prediction_ids: list[str] = []
        for result in results:
            counts[result.action] = counts.get(result.action, 0) + 1
            status = str(result.outcome.status)
            outcome_statuses[status] = outcome_statuses.get(status, 0) + 1
            evaluated_prediction_ids.append(str(result.prediction.pk))
        return JobExecutionResult(
            status=JobRun.Status.SUCCESS,
            details={
                "provider": provider,
                "benchmark_subject": benchmark_subject,
                "eligible_predictions": len(predictions),
                "actions": counts,
                "outcome_statuses": outcome_statuses,
                # The minimum stable identity output-verification needs to
                # independently re-query the exact `Prediction`/
                # `PredictionOutcome` rows this run stands behind, rather
                # than trusting the self-reported counts above at face
                # value.
                "evaluated_prediction_ids": sorted(evaluated_prediction_ids),
                # The exact execution boundary this run evaluated against.
                # Verification binds each reported outcome to this instant
                # (freshly touched now, or legitimately already-terminal
                # strictly before it) instead of requiring
                # `evaluation_date == target_date`, which is false for a
                # normal missed-day/lagging-series catch-up evaluation.
                "evaluation_time": effective_time.isoformat(),
            },
        )

    return execute_target_job(
        job_name=JOB_NAME,
        region="us",
        target_date=evaluation_date,
        task=_task,
    )

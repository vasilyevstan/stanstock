from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import polars as pl
from django.db import transaction
from django.utils import timezone

from stanstock.data.asof import AsOfData, PriceFrameSchemaError
from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset
from stanstock.research.models import Prediction, PredictionOutcome, Recommendation

HORIZON_SESSION_COUNTS: dict[str, int] = {
    Prediction.Horizon.SHORT.value: 10,
    Prediction.Horizon.SIX_MONTH.value: 126,
    Prediction.Horizon.TWELVE_MONTH.value: 252,
    Prediction.Horizon.THREE_YEAR.value: 756,
    Prediction.Horizon.FIVE_YEAR.value: 1260,
    Prediction.Horizon.MEDIUM.value: 252,
    Prediction.Horizon.LONG.value: 756,
}
PriceFrameCache = MutableMapping[tuple[str, str, date, datetime], pl.DataFrame]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    prediction: Prediction
    outcome: PredictionOutcome
    action: str
    resolution: str


class PriceSessionDataError(ValueError):
    """Raised when price observations cannot represent unique market sessions."""


@transaction.atomic
def evaluate_prediction(
    prediction: Prediction,
    *,
    provider: str,
    evaluation_date: date,
    evaluation_time: datetime | None = None,
    benchmark_subject: str | None = None,
    store: AssetStore | None = None,
    frame_cache: PriceFrameCache | None = None,
) -> EvaluationResult:
    prediction = Prediction.objects.select_for_update().get(pk=prediction.pk)
    existing = PredictionOutcome.objects.select_for_update().filter(prediction=prediction).first()
    selected_provider = _resolve_price_provider(prediction, provider)
    if existing is not None and existing.status in _TERMINAL_OUTCOME_STATUSES:
        return EvaluationResult(
            prediction=prediction,
            outcome=existing,
            action="skipped",
            resolution="Existing terminal outcome left unchanged",
        )

    evaluated_at = evaluation_time or timezone.now()
    frame_cache = frame_cache if frame_cache is not None else {}
    if evaluation_date > evaluated_at.date():
        return _save_unresolved(
            prediction,
            existing=existing,
            evaluated_at=evaluated_at,
            evaluation_date=evaluation_date,
            resolution="Evaluation date is after actual evaluation time",
            metadata={"provider": selected_provider, "evaluation_time": evaluated_at.isoformat()},
        )
    if evaluation_date < prediction.target_date:
        return _save_unresolved(
            prediction,
            existing=existing,
            evaluated_at=evaluated_at,
            evaluation_date=evaluation_date,
            resolution="Evaluation date is before prediction target date",
            metadata={
                "provider": selected_provider,
                "target_date": prediction.target_date.isoformat(),
            },
        )

    asof = AsOfData(evaluated_at, store)
    subject = (
        prediction.price_subject or prediction.listing.provider_symbol or prediction.listing.ticker
    )
    session_count = HORIZON_SESSION_COUNTS[prediction.horizon]

    try:
        price_frame = _cached_price_frame(
            asof=asof,
            provider=selected_provider,
            subject=subject,
            through_date=evaluation_date,
            evaluated_at=evaluated_at,
            cache=frame_cache,
        )
    except (DataAsset.DoesNotExist, PriceFrameSchemaError, ValueError) as exc:
        return _save_unresolved(
            prediction,
            existing=existing,
            evaluated_at=evaluated_at,
            evaluation_date=evaluation_date,
            resolution=f"Unable to load evaluation price history: {exc}",
            metadata={"provider": selected_provider, "subject": subject},
        )

    try:
        session = _nth_observed_session(price_frame, prediction.target_date, session_count)
        observed = _observed_session_count(price_frame, prediction.target_date)
    except PriceSessionDataError as exc:
        return _save_unresolved(
            prediction,
            existing=existing,
            evaluated_at=evaluated_at,
            evaluation_date=evaluation_date,
            resolution=str(exc),
            metadata={"provider": selected_provider, "subject": subject},
        )
    if session is None:
        return _save_unresolved(
            prediction,
            existing=existing,
            evaluated_at=evaluated_at,
            evaluation_date=evaluation_date,
            resolution=(
                f"Insufficient observed sessions after target date: "
                f"{observed}/{session_count} through {evaluation_date.isoformat()}"
            ),
            metadata={
                "provider": selected_provider,
                "subject": subject,
                "observed_sessions": observed,
            },
        )

    price_at_prediction = float(prediction.price_at_prediction)
    if price_at_prediction <= 0:
        return _save_unresolved(
            prediction,
            existing=existing,
            evaluated_at=evaluated_at,
            evaluation_date=evaluation_date,
            resolution="Prediction price is not positive",
            metadata={"provider": selected_provider, "subject": subject},
        )

    evaluation_baseline = _close_at_or_before(price_frame, prediction.target_date)
    if evaluation_baseline is None:
        return _save_unresolved(
            prediction,
            existing=existing,
            evaluated_at=evaluated_at,
            evaluation_date=session.observation_date,
            resolution="Evaluation price history has no baseline close at or before target date",
            metadata={"provider": selected_provider, "subject": subject},
        )
    if not _same_price_basis(evaluation_baseline.close, price_at_prediction):
        return _save_corporate_event(
            prediction,
            existing=existing,
            evaluated_at=evaluated_at,
            evaluation_date=session.observation_date,
            resolution="Target-date price changed in the evaluation vintage",
            metadata={
                "provider": selected_provider,
                "subject": subject,
                "prediction_price": price_at_prediction,
                "evaluation_vintage_target_close": evaluation_baseline.close,
                "evaluation_vintage_target_date": evaluation_baseline.observation_date.isoformat(),
            },
        )

    actual_return = session.close / price_at_prediction - 1.0
    success = (
        _success(prediction, actual_return)
        if prediction.evidence_role == Prediction.EvidenceRole.DECISION
        else None
    )
    if prediction.evidence_role == Prediction.EvidenceRole.DECISION and success is None:
        return _save_unresolved(
            prediction,
            existing=existing,
            evaluated_at=evaluated_at,
            evaluation_date=session.observation_date,
            resolution="HOLD success requires non-null stored bear and bull returns",
            metadata={"provider": selected_provider, "subject": subject},
        )

    benchmark_return = None
    benchmark_resolution = ""
    if benchmark_subject:
        benchmark_return, benchmark_resolution = _benchmark_return(
            asof=asof,
            provider=selected_provider,
            subject=benchmark_subject,
            target_date=prediction.target_date,
            evaluation_date=session.observation_date,
            evaluated_at=evaluated_at,
            cache=frame_cache,
        )
    error = None
    direction_correct = None
    interval_covered = None
    if prediction.base_return is not None:
        error = actual_return - float(prediction.base_return)
        direction_correct = _direction(actual_return) == _direction(float(prediction.base_return))
    if prediction.bear_return is not None and prediction.bull_return is not None:
        interval_covered = (
            float(prediction.bear_return) <= actual_return <= float(prediction.bull_return)
        )

    outcome, created = PredictionOutcome.objects.update_or_create(
        prediction=prediction,
        defaults={
            "evaluated_at": evaluated_at,
            "evaluation_date": session.observation_date,
            "status": PredictionOutcome.Status.MATURED,
            "actual_return": _decimal(actual_return),
            "benchmark_return": _optional_decimal(benchmark_return),
            "success": success,
            "direction_correct": direction_correct,
            "interval_covered": interval_covered,
            "resolution": _matured_resolution(
                prediction, session.observation_date, benchmark_resolution
            ),
            "error": _optional_decimal(error),
            "signed_error": _optional_decimal(error),
            "metadata": {
                "provider": selected_provider,
                "subject": subject,
                "horizon_sessions": session_count,
                "evidence_role": prediction.evidence_role,
                "evaluation_close": session.close,
                "benchmark_subject": benchmark_subject or "",
                "benchmark_resolution": benchmark_resolution,
                "success_semantics": _success_semantics(prediction),
            },
        },
    )
    return EvaluationResult(
        prediction=prediction,
        outcome=outcome,
        action="created" if created else "updated",
        resolution=outcome.resolution,
    )


def evaluate_predictions(
    predictions: list[Prediction],
    *,
    provider: str,
    evaluation_date: date,
    evaluation_time: datetime | None = None,
    benchmark_subject: str | None = None,
    store: AssetStore | None = None,
) -> list[EvaluationResult]:
    evaluated_at = evaluation_time or timezone.now()
    frame_cache: dict[tuple[str, str, date, datetime], pl.DataFrame] = {}
    for prediction in predictions:
        _resolve_price_provider(prediction, provider)
    return [
        evaluate_prediction(
            prediction,
            provider=provider,
            evaluation_date=evaluation_date,
            evaluation_time=evaluated_at,
            benchmark_subject=benchmark_subject,
            store=store,
            frame_cache=frame_cache,
        )
        for prediction in predictions
    ]


@dataclass(frozen=True, slots=True)
class PriceSession:
    observation_date: date
    close: float


def _nth_observed_session(
    frame: pl.DataFrame, target_date: date, session_count: int
) -> PriceSession | None:
    sessions = _usable_sessions(frame).filter(pl.col("date") > target_date).sort("date")
    if sessions.height < session_count:
        return None
    row = sessions.row(session_count - 1, named=True)
    return PriceSession(observation_date=row["date"], close=float(row["close"]))


def _observed_session_count(frame: pl.DataFrame, target_date: date) -> int:
    return _usable_sessions(frame).filter(pl.col("date") > target_date).height


def _close_at_or_before(frame: pl.DataFrame, target: date) -> PriceSession | None:
    sessions = _usable_sessions(frame).filter(pl.col("date") <= target).sort("date")
    if sessions.is_empty():
        return None
    row = sessions.row(-1, named=True)
    return PriceSession(observation_date=row["date"], close=float(row["close"]))


def _usable_sessions(frame: pl.DataFrame) -> pl.DataFrame:
    if "date" not in frame.columns or "close" not in frame.columns:
        return pl.DataFrame(
            {"date": [], "close": []}, schema={"date": pl.Date, "close": pl.Float64}
        )
    sessions = (
        frame.select(
            _date_expr(frame).alias("date"), pl.col("close").cast(pl.Float64, strict=False)
        )
        .filter(
            pl.col("date").is_not_null() & pl.col("close").is_not_null() & (pl.col("close") > 0)
        )
        .sort("date")
    )
    duplicate_count = sessions.filter(pl.col("date").is_duplicated()).height
    if duplicate_count:
        raise PriceSessionDataError("Duplicate usable price session dates in evaluation data")
    return sessions


def _date_expr(frame: pl.DataFrame) -> pl.Expr:
    dtype = frame.schema["date"]
    if dtype == pl.Date:
        return pl.col("date")
    if isinstance(dtype, pl.Datetime):
        return pl.col("date").dt.date()
    if dtype in (pl.Utf8, pl.String):
        return pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    return pl.col("date").cast(pl.Date, strict=False)


def _cached_price_frame(
    *,
    asof: AsOfData,
    provider: str,
    subject: str,
    through_date: date,
    evaluated_at: datetime,
    cache: PriceFrameCache,
) -> pl.DataFrame:
    key = (provider, subject, through_date, evaluated_at)
    cached = cache.get(key)
    if cached is not None:
        return cached
    frame = asof.price_frame(
        provider=provider,
        subject=subject,
        through_date=through_date,
    )
    cache[key] = frame
    return frame


def _benchmark_return(
    *,
    asof: AsOfData,
    provider: str,
    subject: str,
    target_date: date,
    evaluation_date: date,
    evaluated_at: datetime,
    cache: PriceFrameCache,
) -> tuple[float | None, str]:
    try:
        frame = _cached_price_frame(
            asof=asof,
            provider=provider,
            subject=subject,
            through_date=evaluation_date,
            evaluated_at=evaluated_at,
            cache=cache,
        )
        target = _close_at_or_before(frame, target_date)
        evaluation = _close_at_or_before(frame, evaluation_date)
    except (
        DataAsset.DoesNotExist,
        PriceFrameSchemaError,
        ValueError,
        PriceSessionDataError,
    ) as exc:
        return None, f"Benchmark unavailable: {exc}"
    if target is None:
        return None, "Benchmark has no close at or before target date"
    if evaluation is None:
        return None, "Benchmark has no close at or before evaluation session"
    return evaluation.close / target.close - 1.0, (
        f"Benchmark return uses {target.observation_date.isoformat()} to "
        f"{evaluation.observation_date.isoformat()} closes"
    )


def _success(prediction: Prediction, actual_return: float) -> bool | None:
    if prediction.recommendation == Recommendation.BUY:
        return actual_return > 0
    if prediction.recommendation == Recommendation.AVOID:
        return actual_return <= 0
    bear = prediction.bear_return
    bull = prediction.bull_return
    if bear is None or bull is None:
        return None
    return float(bear) <= actual_return <= float(bull)


def _success_semantics(prediction: Prediction) -> str:
    if prediction.evidence_role == Prediction.EvidenceRole.ADVISORY:
        return "Advisory forecasts do not receive recommendation success labels"
    if prediction.recommendation == Recommendation.BUY:
        return "BUY succeeds when actual return is positive"
    if prediction.recommendation == Recommendation.AVOID:
        return "AVOID succeeds when actual return is non-positive"
    return "HOLD succeeds when actual return is within stored bear/bull range"


def _matured_resolution(
    prediction: Prediction,
    evaluation_date: date,
    benchmark_resolution: str,
) -> str:
    base = (
        f"Matured after {HORIZON_SESSION_COUNTS[prediction.horizon]} observed sessions; "
        f"evaluated on {evaluation_date.isoformat()}"
    )
    if benchmark_resolution:
        return f"{base}; {benchmark_resolution}"[:120]
    return base[:120]


def _save_unresolved(
    prediction: Prediction,
    *,
    existing: PredictionOutcome | None,
    evaluated_at: datetime,
    evaluation_date: date,
    resolution: str,
    metadata: dict[str, Any],
) -> EvaluationResult:
    if existing is not None and existing.status in _TERMINAL_OUTCOME_STATUSES:
        return EvaluationResult(prediction, existing, "skipped", existing.resolution)
    defaults = {
        "evaluated_at": evaluated_at,
        "evaluation_date": evaluation_date,
        "status": PredictionOutcome.Status.UNRESOLVED,
        "actual_return": None,
        "benchmark_return": None,
        "success": None,
        "direction_correct": None,
        "interval_covered": None,
        "resolution": resolution[:120],
        "error": None,
        "signed_error": None,
        "metadata": metadata,
    }
    if existing is not None and _outcome_matches(existing, defaults):
        return EvaluationResult(prediction, existing, "skipped", existing.resolution)
    outcome, created = PredictionOutcome.objects.update_or_create(
        prediction=prediction,
        defaults=defaults,
    )
    return EvaluationResult(
        prediction=prediction,
        outcome=outcome,
        action="created" if created else "updated",
        resolution=outcome.resolution,
    )


def _save_corporate_event(
    prediction: Prediction,
    *,
    existing: PredictionOutcome | None,
    evaluated_at: datetime,
    evaluation_date: date,
    resolution: str,
    metadata: dict[str, Any],
) -> EvaluationResult:
    if existing is not None and existing.status in _TERMINAL_OUTCOME_STATUSES:
        return EvaluationResult(prediction, existing, "skipped", existing.resolution)
    outcome, created = PredictionOutcome.objects.update_or_create(
        prediction=prediction,
        defaults={
            "evaluated_at": evaluated_at,
            "evaluation_date": evaluation_date,
            "status": PredictionOutcome.Status.CORPORATE_EVENT,
            "actual_return": None,
            "benchmark_return": None,
            "success": None,
            "direction_correct": None,
            "interval_covered": None,
            "resolution": resolution[:120],
            "error": None,
            "signed_error": None,
            "metadata": metadata,
        },
    )
    return EvaluationResult(
        prediction=prediction,
        outcome=outcome,
        action="created" if created else "updated",
        resolution=outcome.resolution,
    )


def _same_price_basis(evaluation_price: float, prediction_price: float) -> bool:
    tolerance = max(1e-6, abs(prediction_price) * 1e-6)
    return abs(evaluation_price - prediction_price) <= tolerance


def _resolve_price_provider(prediction: Prediction, requested_provider: str) -> str:
    recorded_provider = prediction.price_provider.strip()
    if recorded_provider and recorded_provider != requested_provider:
        raise ValueError(
            "Prediction price provider "
            f"{recorded_provider!r} conflicts with requested provider {requested_provider!r}"
        )
    return recorded_provider or requested_provider


def _direction(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _outcome_matches(
    outcome: PredictionOutcome,
    defaults: dict[str, Any],
) -> bool:
    comparable_fields = (
        "evaluation_date",
        "status",
        "actual_return",
        "benchmark_return",
        "success",
        "direction_correct",
        "interval_covered",
        "resolution",
        "error",
        "signed_error",
        "metadata",
    )
    return all(getattr(outcome, field) == defaults[field] for field in comparable_fields)


def _decimal(value: float) -> Decimal:
    return Decimal(str(round(value, 4)))


def _optional_decimal(value: float | None) -> Decimal | None:
    if value is None:
        return None
    return _decimal(value)


_TERMINAL_OUTCOME_STATUSES = {
    PredictionOutcome.Status.MATURED,
    PredictionOutcome.Status.CORPORATE_EVENT,
}

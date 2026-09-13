"""Pure retrospective replay protocol and metric primitives.

This module intentionally performs no ORM, filesystem, provider, clock, CLI,
or management-command work.  Callers must supply the already verified XNYS
session calendar, immutable source-derived return observations, and model
outputs.  The functions below freeze the retrospective protocol declared for
``research-product-v1`` without creating observed predictions or reading real
historical sources.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from statistics import NormalDist, StatisticsError
from typing import Literal

import numpy as np
import numpy.typing as npt

from stanstock.research.price_product import (
    FilteredReturns,
    LedgerTriplet,
    RawTriplet,
    _round_ledger,
    _round_triplet,
    simulate_fhs_terminal_logs,
)
from stanstock.research.price_product_config import PriceProductConfig

ReplayPartition = Literal["development", "validation", "final_holdout"]
GaussianComparator = Literal["zero_log_drift_gaussian", "historical_log_drift_gaussian"]
QuantileName = Literal["p20", "p50", "p80"]
ProjectionMetricName = Literal[
    "median_absolute_error",
    "pinball_p20",
    "pinball_p50",
    "pinball_p80",
    "interval_width",
    "interval_inclusion",
    "interval_score",
]
MomentumDirection = Literal["positive", "negative", "mixed"]
MomentumSuggestion = Literal["buy", "hold", "avoid"]

_QUANTILE_NAMES: tuple[QuantileName, QuantileName, QuantileName] = ("p20", "p50", "p80")
_CENTRAL_60_ALPHA = Decimal("0.4")


class ReplayProtocolError(ValueError):
    """Raised when supplied replay inputs cannot represent the frozen protocol."""


@dataclass(frozen=True, slots=True)
class ReplayPrecision:
    """The explicit precision used for replay accuracy metrics."""

    compared_returns: Literal["ledger_returns"]
    return_decimal_places: int
    rounding_mode: Literal["ROUND_HALF_EVEN"]


@dataclass(frozen=True, slots=True)
class AnchorPoint:
    horizon: str
    horizon_sessions: int
    anchor_index: int
    anchor_date: date
    outcome_end_index: int
    outcome_end_date: date
    partition: ReplayPartition
    evidence_label: str


@dataclass(frozen=True, slots=True)
class AnchorExclusion:
    horizon: str
    horizon_sessions: int
    anchor_index: int
    anchor_date: date
    reason: str
    outcome_end_date: date | None


@dataclass(frozen=True, slots=True)
class AnchorPartitionPlan:
    partition: ReplayPartition
    horizon: str
    horizon_sessions: int
    anchors: tuple[AnchorPoint, ...]
    empty_reason: str | None


@dataclass(frozen=True, slots=True)
class AnchorPlan:
    fixed_epoch: date
    fixed_epoch_index: int | None
    required_prior_returns: int
    holdout_complete_through: date
    partitions: tuple[AnchorPartitionPlan, ...]
    exclusions: tuple[AnchorExclusion, ...]
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class ComparatorProjection:
    model_name: GaussianComparator
    horizon_sessions: int
    quantile_levels: tuple[float, float, float]
    mean_log_return: float | None
    variance_log_return: float | None
    raw_returns: RawTriplet | None
    ledger_returns: LedgerTriplet | None
    precision: ReplayPrecision
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class QuantileLoss:
    quantile: QuantileName
    quantile_level: Decimal
    loss: Decimal


@dataclass(frozen=True, slots=True)
class ProjectionMetricResult:
    precision: ReplayPrecision
    actual_return: Decimal | None
    predicted_returns: LedgerTriplet | None
    median_absolute_error: Decimal | None
    pinball_losses: tuple[QuantileLoss, QuantileLoss, QuantileLoss] | None
    interval_width: Decimal | None
    interval_included: bool | None
    interval_score: Decimal | None
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class MomentumDiagnostic:
    horizon_sessions: int
    direction_group: MomentumDirection | None
    suggestion: MomentumSuggestion | None
    stock_return: Decimal | None
    benchmark_return: Decimal | None
    relative_return: Decimal | None
    success: bool | None
    direction_correct: bool | None
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class ReplayMetricObservation:
    listing_id: str
    anchor_date: date
    target_date: date
    horizon: str
    partition: ReplayPartition
    model_name: str
    metrics: ProjectionMetricResult


@dataclass(frozen=True, slots=True)
class CalendarSpan:
    first_anchor: date
    last_target: date


@dataclass(frozen=True, slots=True)
class AggregateMetricSummary:
    partition: ReplayPartition
    horizon: str
    model_name: str
    target_cohort_count: int
    distinct_listing_count: int
    non_overlapping_target_count: int
    calendar_span: CalendarSpan | None
    duplicate_observation_count: int
    conflicting_maturity_observation_count: int
    unavailable_count: int
    unavailable_reasons: Mapping[str, int]
    averages: Mapping[ProjectionMetricName, Decimal | None]
    insufficient_reason: str | None


@dataclass(frozen=True, slots=True)
class PairedModelComparison:
    candidate_model: str
    baseline_model: str
    partition: ReplayPartition
    horizon: str
    metric_name: ProjectionMetricName
    paired_observation_count: int
    paired_target_cohort_count: int
    candidate_average: Decimal | None
    baseline_average: Decimal | None
    mean_difference_candidate_minus_baseline: Decimal | None
    exclusions: Mapping[str, int]
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class QuantileConvergence:
    horizon: str
    horizon_sessions: int
    quantile: QuantileName
    production_return: float | None
    diagnostic_return: float | None
    movement: float | None
    threshold: float | None
    exceeded: bool | None
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class ConvergenceReport:
    production_path_count: int
    diagnostic_path_count: int
    threshold_rule: str
    rows: tuple[QuantileConvergence, ...]
    exceeded_count: int
    unavailable_count: int


def build_anchor_plan(
    calendar_sessions: Iterable[date],
    *,
    config: PriceProductConfig,
) -> AnchorPlan:
    """Build horizon-spaced anchors from the fixed epoch in the supplied calendar.

    The grid is tied to ``config.replay.fixed_epoch``.  It is never shifted to
    the first listed security observation or to a moving cutoff.  Anchors must
    have the configured 756 prior returns and a complete outcome no later than
    the frozen holdout-through date.
    """

    sessions = tuple(calendar_sessions)
    _validate_calendar_sessions(sessions)
    replay = config.replay
    partitions = _empty_partition_lists(config)
    exclusions: list[AnchorExclusion] = []
    if replay.fixed_epoch not in sessions:
        return AnchorPlan(
            fixed_epoch=replay.fixed_epoch,
            fixed_epoch_index=None,
            required_prior_returns=config.simulation.return_observations,
            holdout_complete_through=replay.holdout_complete_through,
            partitions=_materialize_partitions(
                partitions,
                config=config,
                empty_reason="fixed_epoch_missing_from_calendar",
            ),
            exclusions=(),
            unavailable_reason="fixed_epoch_missing_from_calendar",
        )

    epoch_index = sessions.index(replay.fixed_epoch)
    for horizon, horizon_sessions in config.simulation.horizons:
        for anchor_index, anchor_date in enumerate(sessions):
            if (anchor_index - epoch_index) % horizon_sessions != 0:
                continue
            if anchor_index < config.simulation.return_observations:
                exclusions.append(
                    AnchorExclusion(
                        horizon=horizon,
                        horizon_sessions=horizon_sessions,
                        anchor_index=anchor_index,
                        anchor_date=anchor_date,
                        reason="insufficient_prior_returns",
                        outcome_end_date=None,
                    )
                )
                continue
            outcome_end_index = anchor_index + horizon_sessions
            if outcome_end_index >= len(sessions):
                exclusions.append(
                    AnchorExclusion(
                        horizon=horizon,
                        horizon_sessions=horizon_sessions,
                        anchor_index=anchor_index,
                        anchor_date=anchor_date,
                        reason="outcome_end_not_in_supplied_calendar",
                        outcome_end_date=None,
                    )
                )
                continue
            outcome_end_date = sessions[outcome_end_index]
            if outcome_end_date > replay.holdout_complete_through:
                exclusions.append(
                    AnchorExclusion(
                        horizon=horizon,
                        horizon_sessions=horizon_sessions,
                        anchor_index=anchor_index,
                        anchor_date=anchor_date,
                        reason="outcome_after_holdout_complete_through",
                        outcome_end_date=outcome_end_date,
                    )
                )
                continue
            partition = _assign_partition(anchor_date, outcome_end_date, config=config)
            if partition is None:
                exclusions.append(
                    AnchorExclusion(
                        horizon=horizon,
                        horizon_sessions=horizon_sessions,
                        anchor_index=anchor_index,
                        anchor_date=anchor_date,
                        reason="cross_partition_interval_purged",
                        outcome_end_date=outcome_end_date,
                    )
                )
                continue
            partitions[(partition, horizon)].append(
                AnchorPoint(
                    horizon=horizon,
                    horizon_sessions=horizon_sessions,
                    anchor_index=anchor_index,
                    anchor_date=anchor_date,
                    outcome_end_index=outcome_end_index,
                    outcome_end_date=outcome_end_date,
                    partition=partition,
                    evidence_label=replay.evidence_label,
                )
            )

    return AnchorPlan(
        fixed_epoch=replay.fixed_epoch,
        fixed_epoch_index=epoch_index,
        required_prior_returns=config.simulation.return_observations,
        holdout_complete_through=replay.holdout_complete_through,
        partitions=_materialize_partitions(
            partitions,
            config=config,
            empty_reason="no_eligible_anchors",
        ),
        exclusions=tuple(exclusions),
        unavailable_reason=None,
    )


def gaussian_comparator_quantiles(
    *,
    comparator: GaussianComparator,
    mean_log_return: float,
    population_variance: float,
    horizon_sessions: int,
    config: PriceProductConfig,
) -> ComparatorProjection:
    """Return closed-form p20/p50/p80 cumulative-return quantiles.

    The Gaussian comparators operate in log-return units and then transform
    with ``expm1``.  They do not produce prices or calibrated probabilities.
    Any non-finite/non-positive variance or unrepresentable transform withholds
    the whole triplet.
    """

    precision = replay_precision(config)
    quantiles = config.simulation.quantiles
    if comparator not in config.replay.comparators:
        raise ReplayProtocolError("Comparator is not declared by the frozen replay config")
    if type(horizon_sessions) is not int or horizon_sessions <= 0:
        raise ReplayProtocolError("horizon_sessions must be a positive integer")
    if not math.isfinite(population_variance) or population_variance <= 0:
        return ComparatorProjection(
            model_name=comparator,
            horizon_sessions=horizon_sessions,
            quantile_levels=quantiles,
            mean_log_return=None,
            variance_log_return=None,
            raw_returns=None,
            ledger_returns=None,
            precision=precision,
            unavailable_reason="gaussian_variance_unavailable",
        )
    if comparator == "historical_log_drift_gaussian":
        if not math.isfinite(mean_log_return):
            return ComparatorProjection(
                model_name=comparator,
                horizon_sessions=horizon_sessions,
                quantile_levels=quantiles,
                mean_log_return=None,
                variance_log_return=None,
                raw_returns=None,
                ledger_returns=None,
                precision=precision,
                unavailable_reason="gaussian_mean_unavailable",
            )
        distribution_mean = horizon_sessions * mean_log_return
    else:
        distribution_mean = 0.0
    distribution_variance = horizon_sessions * population_variance
    if not math.isfinite(distribution_variance) or distribution_variance <= 0:
        return ComparatorProjection(
            model_name=comparator,
            horizon_sessions=horizon_sessions,
            quantile_levels=quantiles,
            mean_log_return=None,
            variance_log_return=None,
            raw_returns=None,
            ledger_returns=None,
            precision=precision,
            unavailable_reason="gaussian_variance_unavailable",
        )
    try:
        normal = NormalDist(mu=distribution_mean, sigma=math.sqrt(distribution_variance))
        raw = RawTriplet(*(float(math.expm1(normal.inv_cdf(quantile))) for quantile in quantiles))
    except (OverflowError, StatisticsError, ValueError):
        return ComparatorProjection(
            model_name=comparator,
            horizon_sessions=horizon_sessions,
            quantile_levels=quantiles,
            mean_log_return=distribution_mean,
            variance_log_return=distribution_variance,
            raw_returns=None,
            ledger_returns=None,
            precision=precision,
            unavailable_reason="gaussian_return_unrepresentable",
        )
    if not _valid_raw_return_triplet(raw):
        return ComparatorProjection(
            model_name=comparator,
            horizon_sessions=horizon_sessions,
            quantile_levels=quantiles,
            mean_log_return=distribution_mean,
            variance_log_return=distribution_variance,
            raw_returns=None,
            ledger_returns=None,
            precision=precision,
            unavailable_reason="gaussian_return_unrepresentable",
        )
    ledger = _round_triplet(
        raw,
        places=config.rounding.return_decimal_places,
        kind="return",
    )
    if ledger is None or not _ordered_ledger_triplet(ledger):
        return ComparatorProjection(
            model_name=comparator,
            horizon_sessions=horizon_sessions,
            quantile_levels=quantiles,
            mean_log_return=distribution_mean,
            variance_log_return=distribution_variance,
            raw_returns=None,
            ledger_returns=None,
            precision=precision,
            unavailable_reason="gaussian_return_unrepresentable",
        )
    return ComparatorProjection(
        model_name=comparator,
        horizon_sessions=horizon_sessions,
        quantile_levels=quantiles,
        mean_log_return=distribution_mean,
        variance_log_return=distribution_variance,
        raw_returns=raw,
        ledger_returns=ledger,
        precision=precision,
        unavailable_reason=None,
    )


def score_projection_metrics(
    *,
    predicted_returns: LedgerTriplet | None,
    actual_return: Decimal | float | None,
    config: PriceProductConfig,
) -> ProjectionMetricResult:
    """Score one advisory triplet against one realized cumulative return.

    ``median_absolute_error`` is the absolute error of the median forecast for
    one observation.  Aggregate helpers mean these errors within target cohorts
    and then across cohorts; they do not compute the median of absolute errors.
    """

    precision = replay_precision(config)
    if predicted_returns is None:
        return _unavailable_projection_metrics(precision, "prediction_triplet_unavailable")
    if not _finite_ledger_triplet(predicted_returns):
        return _unavailable_projection_metrics(precision, "prediction_triplet_unavailable")
    if not _ordered_ledger_triplet(predicted_returns):
        return _unavailable_projection_metrics(precision, "prediction_triplet_unordered")
    normalized_prediction = _normalize_ledger_triplet(predicted_returns, config=config)
    if normalized_prediction is None:
        return _unavailable_projection_metrics(precision, "prediction_triplet_unavailable")
    if not _ordered_ledger_triplet(normalized_prediction):
        return _unavailable_projection_metrics(precision, "prediction_triplet_unordered")
    actual = _round_metric_return(actual_return, config=config)
    if actual is None:
        return _unavailable_projection_metrics(precision, "actual_return_unavailable")

    lower = normalized_prediction.lower
    median = normalized_prediction.median
    upper = normalized_prediction.upper
    losses = tuple(
        _pinball_loss(name, Decimal(str(level)), forecast, actual)
        for name, level, forecast in zip(
            _QUANTILE_NAMES,
            config.simulation.quantiles,
            (lower, median, upper),
            strict=True,
        )
    )
    if len(losses) != 3:
        raise ReplayProtocolError("The frozen protocol requires exactly p20/p50/p80")
    interval_width = upper - lower
    interval_included = lower <= actual <= upper
    interval_score = interval_width
    if actual < lower:
        interval_score += (Decimal("2") / _CENTRAL_60_ALPHA) * (lower - actual)
    elif actual > upper:
        interval_score += (Decimal("2") / _CENTRAL_60_ALPHA) * (actual - upper)
    return ProjectionMetricResult(
        precision=precision,
        actual_return=actual,
        predicted_returns=normalized_prediction,
        median_absolute_error=abs(actual - median),
        pinball_losses=(losses[0], losses[1], losses[2]),
        interval_width=interval_width,
        interval_included=interval_included,
        interval_score=interval_score,
        unavailable_reason=None,
    )


def evaluate_momentum_diagnostic(
    *,
    direction: MomentumDirection | None,
    suggestion: MomentumSuggestion | None,
    stock_return: Decimal | float | None,
    benchmark_return: Decimal | float | None,
    config: PriceProductConfig,
) -> MomentumDiagnostic:
    """Evaluate the six-month stock-vs-SPY momentum diagnostic.

    BUY and AVOID use the method-specific success semantics.  HOLD and
    unavailable suggestions retain ``success=None``.
    """

    stock = _round_metric_return(stock_return, config=config)
    benchmark = _round_metric_return(benchmark_return, config=config)
    if direction is None or suggestion is None:
        return MomentumDiagnostic(
            horizon_sessions=config.momentum.decision_horizon_sessions,
            direction_group=direction,
            suggestion=suggestion,
            stock_return=stock,
            benchmark_return=benchmark,
            relative_return=None if stock is None or benchmark is None else stock - benchmark,
            success=None,
            direction_correct=None,
            unavailable_reason="momentum_decision_unavailable",
        )
    if direction not in ("positive", "negative", "mixed"):
        raise ReplayProtocolError("direction must be positive, negative, mixed, or None")
    if suggestion not in ("buy", "hold", "avoid"):
        raise ReplayProtocolError("suggestion must be buy, hold, avoid, or None")
    if stock is None or benchmark is None:
        return MomentumDiagnostic(
            horizon_sessions=config.momentum.decision_horizon_sessions,
            direction_group=direction,
            suggestion=suggestion,
            stock_return=stock,
            benchmark_return=benchmark,
            relative_return=None,
            success=None,
            direction_correct=None,
            unavailable_reason="momentum_endpoint_return_unavailable",
        )
    relative = stock - benchmark
    if direction == "positive":
        direction_correct: bool | None = relative > 0
    elif direction == "negative":
        direction_correct = relative < 0
    else:
        direction_correct = None

    success: bool | None
    if suggestion == "buy":
        success = stock > 0 and stock > benchmark
    elif suggestion == "avoid":
        success = stock < 0 and stock < benchmark
    else:
        success = None
    return MomentumDiagnostic(
        horizon_sessions=config.momentum.decision_horizon_sessions,
        direction_group=direction,
        suggestion=suggestion,
        stock_return=stock,
        benchmark_return=benchmark,
        relative_return=relative,
        success=success,
        direction_correct=direction_correct,
        unavailable_reason=None,
    )


def aggregate_projection_metrics(
    observations: Iterable[ReplayMetricObservation],
    *,
    expected_partitions: Iterable[ReplayPartition],
    expected_horizons: Iterable[str],
    expected_models: Iterable[str],
) -> tuple[AggregateMetricSummary, ...]:
    """Aggregate by target cohort first, then across cohorts.

    Each model summary uses that model's own eligible support and reports
    duplicate/conflicting observations excluded from that support.  These
    summaries must not be presented as a paired candidate-vs-baseline
    comparison; use :func:`compare_aligned_models` for aligned comparisons.
    """

    expected_keys = tuple(
        (partition, horizon, model)
        for partition in expected_partitions
        for horizon in expected_horizons
        for model in expected_models
    )
    canonical, duplicates, conflicts = _canonical_observations(observations)
    by_group: dict[tuple[ReplayPartition, str, str], list[ReplayMetricObservation]] = {
        key: [] for key in expected_keys
    }
    duplicate_counts: Counter[tuple[ReplayPartition, str, str]] = Counter()
    conflict_counts: Counter[tuple[ReplayPartition, str, str]] = Counter()
    for duplicate in duplicates:
        duplicate_counts[(duplicate.partition, duplicate.horizon, duplicate.model_name)] += 1
    for conflict in conflicts:
        conflict_counts[(conflict.partition, conflict.horizon, conflict.model_name)] += 1
    for observation in canonical:
        by_group.setdefault(
            (observation.partition, observation.horizon, observation.model_name),
            [],
        ).append(observation)

    summaries: list[AggregateMetricSummary] = []
    for partition, horizon, model in by_group:
        group = by_group[(partition, horizon, model)]
        summaries.append(
            _summarize_group(
                group,
                partition=partition,
                horizon=horizon,
                model=model,
                duplicate_count=duplicate_counts[(partition, horizon, model)],
                conflict_count=conflict_counts[(partition, horizon, model)],
            )
        )
    return tuple(summaries)


def compare_aligned_models(
    observations: Iterable[ReplayMetricObservation],
    *,
    candidate_model: str,
    baseline_model: str,
    partition: ReplayPartition,
    horizon: str,
    metric_name: ProjectionMetricName,
) -> PairedModelComparison:
    """Compare one explicit partition/horizon scope on paired observations.

    Callers must pass only observations for the requested ``partition`` and
    ``horizon`` for the candidate/baseline models.  Mixed-scope input is
    rejected instead of being silently filtered or pooled.  Within the explicit
    scope, paired available observations are averaged within target-date
    cohorts first, then those cohort means are averaged across cohorts.
    """

    all_observations = tuple(observations)
    _reject_mixed_comparison_scope(
        all_observations,
        candidate_model=candidate_model,
        baseline_model=baseline_model,
        partition=partition,
        horizon=horizon,
    )
    canonical, duplicates, conflicts = _canonical_observations(all_observations)
    candidate: dict[tuple[str, date, date, str, ReplayPartition], ReplayMetricObservation] = {}
    baseline: dict[tuple[str, date, date, str, ReplayPartition], ReplayMetricObservation] = {}
    all_keys: set[tuple[str, date, date, str, ReplayPartition]] = set()
    exclusions: Counter[str] = Counter(
        {
            "duplicate_identity_excluded": len(duplicates),
            "conflicting_maturity_excluded": len(conflicts),
        }
    )
    for observation in canonical:
        key = (
            observation.listing_id,
            observation.anchor_date,
            observation.target_date,
            observation.horizon,
            observation.partition,
        )
        if observation.model_name == candidate_model:
            candidate[key] = observation
            all_keys.add(key)
        elif observation.model_name == baseline_model:
            baseline[key] = observation
            all_keys.add(key)

    paired_by_target: dict[date, list[tuple[Decimal, Decimal]]] = {}
    for key in sorted(all_keys):
        candidate_observation = candidate.get(key)
        baseline_observation = baseline.get(key)
        if candidate_observation is None:
            exclusions["candidate_missing"] += 1
            continue
        if baseline_observation is None:
            exclusions["baseline_missing"] += 1
            continue
        candidate_value = _metric_value(candidate_observation.metrics, metric_name)
        baseline_value = _metric_value(baseline_observation.metrics, metric_name)
        if candidate_value is None:
            reason = (
                candidate_observation.metrics.unavailable_reason or "candidate_metric_unavailable"
            )
            exclusions[f"candidate_unavailable:{reason}"] += 1
            continue
        if baseline_value is None:
            reason = (
                baseline_observation.metrics.unavailable_reason or "baseline_metric_unavailable"
            )
            exclusions[f"baseline_unavailable:{reason}"] += 1
            continue
        paired_by_target.setdefault(candidate_observation.target_date, []).append(
            (candidate_value, baseline_value)
        )

    if not paired_by_target:
        return PairedModelComparison(
            candidate_model=candidate_model,
            baseline_model=baseline_model,
            partition=partition,
            horizon=horizon,
            metric_name=metric_name,
            paired_observation_count=0,
            paired_target_cohort_count=0,
            candidate_average=None,
            baseline_average=None,
            mean_difference_candidate_minus_baseline=None,
            exclusions=dict(exclusions),
            unavailable_reason="no_aligned_candidate_baseline_observations",
        )
    candidate_cohort_means: list[Decimal] = []
    baseline_cohort_means: list[Decimal] = []
    paired_count = 0
    for pairs in paired_by_target.values():
        paired_count += len(pairs)
        candidate_cohort_means.append(
            _mean_decimal(candidate_value for candidate_value, _ in pairs)
        )
        baseline_cohort_means.append(_mean_decimal(baseline_value for _, baseline_value in pairs))
    candidate_average = _mean_decimal(candidate_cohort_means)
    baseline_average = _mean_decimal(baseline_cohort_means)
    return PairedModelComparison(
        candidate_model=candidate_model,
        baseline_model=baseline_model,
        partition=partition,
        horizon=horizon,
        metric_name=metric_name,
        paired_observation_count=paired_count,
        paired_target_cohort_count=len(paired_by_target),
        candidate_average=candidate_average,
        baseline_average=baseline_average,
        mean_difference_candidate_minus_baseline=candidate_average - baseline_average,
        exclusions=dict(exclusions),
        unavailable_reason=None,
    )


def compare_numerical_convergence(
    *,
    production: Mapping[str, tuple[int, RawTriplet | None]],
    diagnostic: Mapping[str, tuple[int, RawTriplet | None]],
    production_path_count: int = 8192,
    diagnostic_path_count: int = 16384,
) -> ConvergenceReport:
    """Compare production and doubled-path p20/p50/p80 quantiles."""

    rows: list[QuantileConvergence] = []
    for horizon in sorted(
        set(production) | set(diagnostic),
        key=lambda value: (_convergence_horizon_sessions(value, production, diagnostic), value),
    ):
        production_item = production.get(horizon)
        diagnostic_item = diagnostic.get(horizon)
        if production_item is None or diagnostic_item is None:
            rows.extend(
                _unavailable_convergence_rows(
                    horizon=horizon,
                    horizon_sessions=0,
                    reason="horizon_missing_from_convergence_input",
                )
            )
            continue
        production_sessions, production_triplet = production_item
        diagnostic_sessions, diagnostic_triplet = diagnostic_item
        if production_sessions != diagnostic_sessions:
            rows.extend(
                _unavailable_convergence_rows(
                    horizon=horizon,
                    horizon_sessions=production_sessions,
                    reason="horizon_session_mismatch",
                )
            )
            continue
        if production_triplet is None or diagnostic_triplet is None:
            rows.extend(
                _unavailable_convergence_rows(
                    horizon=horizon,
                    horizon_sessions=production_sessions,
                    reason="convergence_triplet_unavailable",
                )
            )
            continue
        if not _valid_raw_return_triplet(production_triplet) or not _valid_raw_return_triplet(
            diagnostic_triplet
        ):
            rows.extend(
                _unavailable_convergence_rows(
                    horizon=horizon,
                    horizon_sessions=production_sessions,
                    reason="convergence_triplet_invalid",
                )
            )
            continue
        interval_width = Decimal(str(production_triplet.upper)) - Decimal(
            str(production_triplet.lower)
        )
        threshold_decimal = max(Decimal("0.01"), Decimal("0.02") * interval_width)
        threshold = float(threshold_decimal)
        for quantile_name, production_value, diagnostic_value in zip(
            _QUANTILE_NAMES,
            (
                production_triplet.lower,
                production_triplet.median,
                production_triplet.upper,
            ),
            (
                diagnostic_triplet.lower,
                diagnostic_triplet.median,
                diagnostic_triplet.upper,
            ),
            strict=True,
        ):
            movement_decimal = abs(Decimal(str(diagnostic_value)) - Decimal(str(production_value)))
            movement = float(movement_decimal)
            rows.append(
                QuantileConvergence(
                    horizon=horizon,
                    horizon_sessions=production_sessions,
                    quantile=quantile_name,
                    production_return=production_value,
                    diagnostic_return=diagnostic_value,
                    movement=movement,
                    threshold=threshold,
                    exceeded=movement_decimal > threshold_decimal,
                    unavailable_reason=None,
                )
            )
    return ConvergenceReport(
        production_path_count=production_path_count,
        diagnostic_path_count=diagnostic_path_count,
        threshold_rule="max(0.01 return, 0.02 * production_interval_width)",
        rows=tuple(rows),
        exceeded_count=sum(1 for row in rows if row.exceeded is True),
        unavailable_count=sum(1 for row in rows if row.unavailable_reason is not None),
    )


def calculate_fhs_convergence(
    filtered: FilteredReturns,
    *,
    seed: int,
    config: PriceProductConfig,
) -> ConvergenceReport:
    """Generate production-vs-diagnostic FHS quantiles for one listing input."""

    horizons = tuple(sessions for _name, sessions in config.simulation.horizons)
    production = simulate_fhs_terminal_logs(
        filtered,
        seed=seed,
        horizons=horizons,
        path_count=config.simulation.production_paths,
        diagnostic_max_paths=config.simulation.diagnostic_max_paths,
        variance_target_weight=config.simulation.variance_target_weight,
        variance_persistence=config.simulation.variance_persistence,
        innovation_weight=config.simulation.innovation_weight,
    )
    diagnostic = simulate_fhs_terminal_logs(
        filtered,
        seed=seed,
        horizons=horizons,
        path_count=config.simulation.diagnostic_max_paths,
        diagnostic_max_paths=config.simulation.diagnostic_max_paths,
        variance_target_weight=config.simulation.variance_target_weight,
        variance_persistence=config.simulation.variance_persistence,
        innovation_weight=config.simulation.innovation_weight,
    )
    production_triplets = {
        horizon: (sessions, _terminal_log_triplet(values, config=config))
        for horizon, sessions, values in zip(
            (name for name, _sessions in config.simulation.horizons),
            horizons,
            production.with_drift,
            strict=True,
        )
    }
    diagnostic_triplets = {
        horizon: (sessions, _terminal_log_triplet(values, config=config))
        for horizon, sessions, values in zip(
            (name for name, _sessions in config.simulation.horizons),
            horizons,
            diagnostic.with_drift,
            strict=True,
        )
    }
    return compare_numerical_convergence(
        production=production_triplets,
        diagnostic=diagnostic_triplets,
        production_path_count=config.simulation.production_paths,
        diagnostic_path_count=config.simulation.diagnostic_max_paths,
    )


def replay_precision(config: PriceProductConfig) -> ReplayPrecision:
    if config.rounding.mode != "ROUND_HALF_EVEN":
        raise ReplayProtocolError("Replay metrics require ROUND_HALF_EVEN ledger precision")
    return ReplayPrecision(
        compared_returns="ledger_returns",
        return_decimal_places=config.rounding.return_decimal_places,
        rounding_mode="ROUND_HALF_EVEN",
    )


def _validate_calendar_sessions(sessions: tuple[date, ...]) -> None:
    if not sessions:
        raise ReplayProtocolError("Replay calendar must contain at least one session")
    if any(type(value) is not date for value in sessions):
        raise ReplayProtocolError("Replay calendar sessions must be date instances")
    if tuple(sorted(set(sessions))) != sessions:
        raise ReplayProtocolError("Replay calendar must be unique and strictly increasing")


def _partition_names() -> tuple[ReplayPartition, ReplayPartition, ReplayPartition]:
    return ("development", "validation", "final_holdout")


def _empty_partition_lists(
    config: PriceProductConfig,
) -> dict[tuple[ReplayPartition, str], list[AnchorPoint]]:
    return {
        (partition, horizon): []
        for partition in _partition_names()
        for horizon, _sessions in config.simulation.horizons
    }


def _materialize_partitions(
    values: Mapping[tuple[ReplayPartition, str], list[AnchorPoint]],
    *,
    config: PriceProductConfig,
    empty_reason: str,
) -> tuple[AnchorPartitionPlan, ...]:
    plans: list[AnchorPartitionPlan] = []
    horizon_sessions_by_name = dict(config.simulation.horizons)
    for partition in _partition_names():
        for horizon, _horizon_sessions in config.simulation.horizons:
            key_partition = partition
            anchors = tuple(values[(key_partition, horizon)])
            plans.append(
                AnchorPartitionPlan(
                    partition=key_partition,
                    horizon=horizon,
                    horizon_sessions=horizon_sessions_by_name[horizon],
                    anchors=anchors,
                    empty_reason=None if anchors else empty_reason,
                )
            )
    return tuple(plans)


def _assign_partition(
    anchor_date: date,
    outcome_end_date: date,
    *,
    config: PriceProductConfig,
) -> ReplayPartition | None:
    replay = config.replay
    if outcome_end_date < replay.development_end_exclusive:
        return "development"
    if (
        replay.validation_start <= anchor_date
        and outcome_end_date < replay.validation_end_exclusive
    ):
        return "validation"
    if anchor_date >= replay.holdout_start and outcome_end_date <= replay.holdout_complete_through:
        return "final_holdout"
    if not replay.purge_partition_crossings:
        raise ReplayProtocolError("The frozen replay config must purge partition crossings")
    return None


def _valid_raw_return_triplet(value: RawTriplet) -> bool:
    return (
        all(
            math.isfinite(item) and item > -1.0 for item in (value.lower, value.median, value.upper)
        )
        and value.lower <= value.median <= value.upper
    )


def _ordered_ledger_triplet(value: LedgerTriplet) -> bool:
    return value.lower <= value.median <= value.upper


def _round_metric_return(
    value: Decimal | float | None,
    *,
    config: PriceProductConfig,
) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        if not value.is_finite():
            return None
        try:
            with localcontext() as context:
                context.prec = 64
                context.rounding = ROUND_HALF_EVEN
                return value.quantize(
                    Decimal(1).scaleb(-config.rounding.return_decimal_places),
                    rounding=ROUND_HALF_EVEN,
                )
        except (InvalidOperation, ValueError):
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return _round_ledger(float(value), places=config.rounding.return_decimal_places, kind="return")


def _normalize_ledger_triplet(
    value: LedgerTriplet | None,
    *,
    config: PriceProductConfig,
) -> LedgerTriplet | None:
    if value is None:
        return None
    try:
        return _round_triplet(
            RawTriplet(
                float(value.lower),
                float(value.median),
                float(value.upper),
            ),
            places=config.rounding.return_decimal_places,
            kind="return",
        )
    except (OverflowError, ValueError):
        return None


def _finite_ledger_triplet(value: LedgerTriplet) -> bool:
    return value.lower.is_finite() and value.median.is_finite() and value.upper.is_finite()


def _unavailable_projection_metrics(
    precision: ReplayPrecision,
    reason: str,
) -> ProjectionMetricResult:
    return ProjectionMetricResult(
        precision=precision,
        actual_return=None,
        predicted_returns=None,
        median_absolute_error=None,
        pinball_losses=None,
        interval_width=None,
        interval_included=None,
        interval_score=None,
        unavailable_reason=reason,
    )


def _pinball_loss(
    name: QuantileName,
    quantile: Decimal,
    forecast: Decimal,
    actual: Decimal,
) -> QuantileLoss:
    error = actual - forecast
    loss = quantile * error if error >= 0 else (quantile - Decimal("1")) * error
    return QuantileLoss(quantile=name, quantile_level=quantile, loss=loss)


def _canonical_observations(
    observations: Iterable[ReplayMetricObservation],
) -> tuple[
    tuple[ReplayMetricObservation, ...],
    tuple[ReplayMetricObservation, ...],
    tuple[ReplayMetricObservation, ...],
]:
    by_conflict_key: dict[
        tuple[str, date, str, ReplayPartition, str],
        list[ReplayMetricObservation],
    ] = {}
    for observation in observations:
        conflict_key = (
            observation.listing_id,
            observation.anchor_date,
            observation.horizon,
            observation.partition,
            observation.model_name,
        )
        by_conflict_key.setdefault(conflict_key, []).append(observation)
    conflict_ids = {
        conflict_key
        for conflict_key, items in by_conflict_key.items()
        if len({item.target_date for item in items}) > 1
    }
    conflicts: list[ReplayMetricObservation] = []
    seen: set[tuple[str, date, date, str, ReplayPartition, str]] = set()
    canonical: list[ReplayMetricObservation] = []
    duplicates: list[ReplayMetricObservation] = []
    for items in by_conflict_key.values():
        for observation in items:
            conflict_key = (
                observation.listing_id,
                observation.anchor_date,
                observation.horizon,
                observation.partition,
                observation.model_name,
            )
            if conflict_key in conflict_ids:
                conflicts.append(observation)
                continue
            key = (
                observation.listing_id,
                observation.anchor_date,
                observation.target_date,
                observation.horizon,
                observation.partition,
                observation.model_name,
            )
            if key in seen:
                duplicates.append(observation)
                continue
            seen.add(key)
            canonical.append(observation)
    return tuple(canonical), tuple(duplicates), tuple(conflicts)


def _reject_mixed_comparison_scope(
    observations: Iterable[ReplayMetricObservation],
    *,
    candidate_model: str,
    baseline_model: str,
    partition: ReplayPartition,
    horizon: str,
) -> None:
    for observation in observations:
        if observation.model_name not in (candidate_model, baseline_model):
            continue
        if observation.partition != partition or observation.horizon != horizon:
            raise ReplayProtocolError(
                "compare_aligned_models requires pre-filtered observations for one "
                "explicit partition and horizon"
            )


def _convergence_horizon_sessions(
    horizon: str,
    production: Mapping[str, tuple[int, RawTriplet | None]],
    diagnostic: Mapping[str, tuple[int, RawTriplet | None]],
) -> int:
    if horizon in production:
        return production[horizon][0]
    return diagnostic[horizon][0]


def _summarize_group(
    group: Iterable[ReplayMetricObservation],
    *,
    partition: ReplayPartition,
    horizon: str,
    model: str,
    duplicate_count: int,
    conflict_count: int,
) -> AggregateMetricSummary:
    observations = tuple(group)
    unavailable_reasons: Counter[str] = Counter(
        observation.metrics.unavailable_reason
        for observation in observations
        if observation.metrics.unavailable_reason is not None
    )
    cohorts: dict[date, list[ReplayMetricObservation]] = {}
    for observation in observations:
        cohorts.setdefault(observation.target_date, []).append(observation)
    averages: dict[ProjectionMetricName, Decimal | None] = {}
    for metric_name in _metric_names():
        cohort_values: list[Decimal] = []
        for cohort in cohorts.values():
            values = [
                value
                for observation in cohort
                if (value := _metric_value(observation.metrics, metric_name)) is not None
            ]
            if values:
                cohort_values.append(_mean_decimal(values))
        averages[metric_name] = _mean_decimal(cohort_values) if cohort_values else None

    span = None
    if observations:
        span = CalendarSpan(
            first_anchor=min(observation.anchor_date for observation in observations),
            last_target=max(observation.target_date for observation in observations),
        )
    insufficient_reason = None
    if not observations:
        insufficient_reason = "no_eligible_observations"
    elif all(value is None for value in averages.values()):
        insufficient_reason = "no_available_cohort_metrics"
    return AggregateMetricSummary(
        partition=partition,
        horizon=horizon,
        model_name=model,
        target_cohort_count=len(cohorts),
        distinct_listing_count=len({observation.listing_id for observation in observations}),
        non_overlapping_target_count=_non_overlapping_target_count(observations),
        calendar_span=span,
        duplicate_observation_count=duplicate_count,
        conflicting_maturity_observation_count=conflict_count,
        unavailable_count=sum(unavailable_reasons.values()),
        unavailable_reasons=dict(unavailable_reasons),
        averages=averages,
        insufficient_reason=insufficient_reason,
    )


def _metric_names() -> tuple[ProjectionMetricName, ...]:
    return (
        "median_absolute_error",
        "pinball_p20",
        "pinball_p50",
        "pinball_p80",
        "interval_width",
        "interval_inclusion",
        "interval_score",
    )


def _metric_value(
    metrics: ProjectionMetricResult,
    metric_name: ProjectionMetricName,
) -> Decimal | None:
    if metric_name == "median_absolute_error":
        return metrics.median_absolute_error
    if metric_name == "interval_width":
        return metrics.interval_width
    if metric_name == "interval_score":
        return metrics.interval_score
    if metric_name == "interval_inclusion":
        if metrics.interval_included is None:
            return None
        return Decimal(1) if metrics.interval_included else Decimal(0)
    if metrics.pinball_losses is None:
        return None
    if metric_name == "pinball_p20":
        return metrics.pinball_losses[0].loss
    if metric_name == "pinball_p50":
        return metrics.pinball_losses[1].loss
    if metric_name == "pinball_p80":
        return metrics.pinball_losses[2].loss
    raise ReplayProtocolError(f"Unsupported projection metric: {metric_name}")


def _mean_decimal(values: Iterable[Decimal]) -> Decimal:
    entries = tuple(values)
    if not entries:
        raise ReplayProtocolError("Cannot average an empty decimal collection")
    return sum(entries, Decimal(0)) / Decimal(len(entries))


def _non_overlapping_target_count(observations: Iterable[ReplayMetricObservation]) -> int:
    intervals = sorted(
        {(observation.anchor_date, observation.target_date) for observation in observations},
        key=lambda value: (value[1], value[0]),
    )
    count = 0
    last_end: date | None = None
    for anchor_date, target_date in intervals:
        if last_end is None or anchor_date >= last_end:
            count += 1
            last_end = target_date
    return count


def _unavailable_convergence_rows(
    *,
    horizon: str,
    horizon_sessions: int,
    reason: str,
) -> tuple[QuantileConvergence, QuantileConvergence, QuantileConvergence]:
    def row(quantile: QuantileName) -> QuantileConvergence:
        return QuantileConvergence(
            horizon=horizon,
            horizon_sessions=horizon_sessions,
            quantile=quantile,
            production_return=None,
            diagnostic_return=None,
            movement=None,
            threshold=None,
            exceeded=None,
            unavailable_reason=reason,
        )

    return (row("p20"), row("p50"), row("p80"))


def _terminal_log_triplet(
    values: npt.NDArray[np.float64],
    *,
    config: PriceProductConfig,
) -> RawTriplet | None:
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        return None
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        returns = np.expm1(values)
    if not np.all(np.isfinite(returns)):
        return None
    raw_values = np.quantile(
        returns,
        config.simulation.quantiles,
        method=config.simulation.quantile_method,
    )
    triplet = RawTriplet(*(float(value) for value in raw_values))
    return triplet if _valid_raw_return_triplet(triplet) else None

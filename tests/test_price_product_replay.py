from __future__ import annotations

import math
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal
from statistics import NormalDist

import exchange_calendars as xcals  # type: ignore[import-untyped]
import pytest

from stanstock.research.price_product import LedgerTriplet, RawTriplet
from stanstock.research.price_product_config import load_price_product_config
from stanstock.research.price_product_replay import (
    AggregateMetricSummary,
    AnchorPartitionPlan,
    ProjectionMetricResult,
    ReplayMetricObservation,
    aggregate_projection_metrics,
    build_anchor_plan,
    compare_aligned_models,
    compare_numerical_convergence,
    evaluate_momentum_diagnostic,
    gaussian_comparator_quantiles,
    score_projection_metrics,
)


def _xnys_sessions(start: str = "2016-01-01", end: str = "2026-12-31") -> tuple[date, ...]:
    calendar = xcals.get_calendar("XNYS")
    return tuple(session.date() for session in calendar.sessions_in_range(start, end))


def _summary_by_key(
    summaries: tuple[AggregateMetricSummary, ...],
) -> dict[tuple[str, str, str], AggregateMetricSummary]:
    return {
        (summary.partition, summary.horizon, summary.model_name): summary for summary in summaries
    }


def _metrics(
    predicted: LedgerTriplet,
    actual: Decimal | float,
) -> ProjectionMetricResult:
    return score_projection_metrics(
        predicted_returns=predicted,
        actual_return=actual,
        config=load_price_product_config(),
    )


def _observation(
    *,
    listing: str,
    anchor: date,
    target: date,
    model: str,
    metrics: ProjectionMetricResult,
    horizon: str = "6m",
) -> ReplayMetricObservation:
    return ReplayMetricObservation(
        listing_id=listing,
        anchor_date=anchor,
        target_date=target,
        horizon=horizon,
        partition="validation",
        model_name=model,
        metrics=metrics,
    )


def test_anchor_plan_uses_fixed_epoch_grid_and_purges_cross_partition_intervals() -> None:
    config = load_price_product_config()
    plan = build_anchor_plan(_xnys_sessions(), config=config)

    assert plan.fixed_epoch == date(2019, 9, 3)
    assert plan.fixed_epoch_index is not None
    all_anchors = [anchor for partition in plan.partitions for anchor in partition.anchors]
    assert all_anchors
    for anchor in all_anchors:
        assert (anchor.anchor_index - plan.fixed_epoch_index) % anchor.horizon_sessions == 0
        assert anchor.anchor_index >= plan.required_prior_returns
        assert anchor.outcome_end_date <= date(2026, 9, 11)
        if anchor.partition == "development":
            assert anchor.outcome_end_date < date(2024, 1, 1)
        elif anchor.partition == "validation":
            assert date(2024, 1, 1) <= anchor.anchor_date
            assert anchor.outcome_end_date < date(2025, 1, 1)
        else:
            assert anchor.anchor_date >= date(2025, 1, 1)

    exclusions = {exclusion.reason for exclusion in plan.exclusions}
    assert "insufficient_prior_returns" in exclusions
    assert "cross_partition_interval_purged" in exclusions
    assert not [
        anchor
        for anchor in all_anchors
        if anchor.anchor_date < date(2024, 1, 1) <= anchor.outcome_end_date
    ]
    empty = _summary_by_partition_horizon(plan.partitions)
    assert empty[("validation", "3y")].empty_reason == "no_eligible_anchors"
    assert empty[("final_holdout", "5y")].empty_reason == "no_eligible_anchors"


def _summary_by_partition_horizon(
    partitions: tuple[AnchorPartitionPlan, ...],
) -> dict[tuple[str, str], AnchorPartitionPlan]:
    return {(partition.partition, partition.horizon): partition for partition in partitions}


def test_anchor_plan_reports_missing_epoch_without_moving_grid_to_first_observation() -> None:
    config = load_price_product_config()
    sessions = tuple(session for session in _xnys_sessions("2019-09-04", "2026-12-31"))

    plan = build_anchor_plan(sessions, config=config)

    assert plan.fixed_epoch_index is None
    assert plan.unavailable_reason == "fixed_epoch_missing_from_calendar"
    assert all(
        partition.empty_reason == "fixed_epoch_missing_from_calendar"
        for partition in plan.partitions
    )
    assert all(not partition.anchors for partition in plan.partitions)


def test_gaussian_comparators_use_closed_form_log_units_and_ledger_rounding() -> None:
    config = load_price_product_config()
    zero = gaussian_comparator_quantiles(
        comparator="zero_log_drift_gaussian",
        mean_log_return=0.001,
        population_variance=0.0004,
        horizon_sessions=4,
        config=config,
    )
    historical = gaussian_comparator_quantiles(
        comparator="historical_log_drift_gaussian",
        mean_log_return=0.001,
        population_variance=0.0004,
        horizon_sessions=4,
        config=config,
    )

    sigma = math.sqrt(4 * 0.0004)
    expected_zero_lower = math.expm1(NormalDist(0.0, sigma).inv_cdf(0.2))
    expected_historical_median = math.expm1(4 * 0.001)
    assert zero.raw_returns is not None
    assert zero.raw_returns.lower == pytest.approx(expected_zero_lower)
    assert zero.raw_returns.median == pytest.approx(0.0)
    assert zero.ledger_returns is not None
    assert zero.ledger_returns.median == Decimal("0.0000")
    assert historical.raw_returns is not None
    assert historical.raw_returns.median == pytest.approx(expected_historical_median)
    assert historical.ledger_returns is not None
    assert historical.ledger_returns.median == Decimal(str(expected_historical_median)).quantize(
        Decimal("0.0001"),
        rounding=ROUND_HALF_EVEN,
    )
    assert historical.precision.compared_returns == "ledger_returns"


def test_gaussian_comparator_withholds_whole_triplet_on_invalid_variance() -> None:
    result = gaussian_comparator_quantiles(
        comparator="zero_log_drift_gaussian",
        mean_log_return=0.0,
        population_variance=0.0,
        horizon_sessions=126,
        config=load_price_product_config(),
    )

    assert result.unavailable_reason == "gaussian_variance_unavailable"
    assert result.raw_returns is None
    assert result.ledger_returns is None


def test_projection_metrics_cover_pinball_interval_boundaries_and_scores() -> None:
    config = load_price_product_config()
    triplet = LedgerTriplet(Decimal("-0.1000"), Decimal("0.0000"), Decimal("0.2000"))

    boundary = score_projection_metrics(
        predicted_returns=triplet,
        actual_return=Decimal("-0.10000"),
        config=config,
    )
    assert boundary.actual_return == Decimal("-0.1000")
    assert boundary.median_absolute_error == Decimal("0.1000")
    assert boundary.interval_width == Decimal("0.3000")
    assert boundary.interval_included is True
    assert boundary.interval_score == Decimal("0.3000")
    assert boundary.pinball_losses is not None
    assert boundary.pinball_losses[0].loss == Decimal("0.00000")
    assert boundary.pinball_losses[1].loss == Decimal("0.05000")

    below = score_projection_metrics(
        predicted_returns=triplet,
        actual_return=Decimal("-0.2000"),
        config=config,
    )
    assert below.interval_included is False
    assert below.interval_score == Decimal("0.8000")

    invalid = score_projection_metrics(
        predicted_returns=LedgerTriplet(Decimal("0.2"), Decimal("0.1"), Decimal("0.3")),
        actual_return=0.1,
        config=config,
    )
    assert invalid.unavailable_reason == "prediction_triplet_unordered"
    assert invalid.median_absolute_error is None


def test_momentum_diagnostic_keeps_buy_avoid_success_and_hold_null_separate() -> None:
    config = load_price_product_config()

    buy = evaluate_momentum_diagnostic(
        direction="positive",
        suggestion="buy",
        stock_return=Decimal("0.0500"),
        benchmark_return=Decimal("0.0500"),
        config=config,
    )
    avoid = evaluate_momentum_diagnostic(
        direction="negative",
        suggestion="avoid",
        stock_return=Decimal("-0.0200"),
        benchmark_return=Decimal("0.0100"),
        config=config,
    )
    hold = evaluate_momentum_diagnostic(
        direction="mixed",
        suggestion="hold",
        stock_return=Decimal("0.0200"),
        benchmark_return=Decimal("-0.0100"),
        config=config,
    )

    assert buy.success is False
    assert buy.horizon_sessions == 126
    assert buy.direction_correct is False
    assert avoid.success is True
    assert avoid.direction_correct is True
    assert hold.success is None
    assert hold.direction_correct is None


def test_aggregation_is_cohort_first_and_deduplicates_listing_identities() -> None:
    triplet = LedgerTriplet(Decimal("0.0000"), Decimal("0.0000"), Decimal("0.2000"))
    observations = [
        _observation(
            listing="A",
            anchor=date(2024, 1, 2),
            target=date(2024, 7, 1),
            model="candidate",
            metrics=_metrics(triplet, Decimal("0.0000")),
        ),
        _observation(
            listing="B",
            anchor=date(2024, 1, 2),
            target=date(2024, 7, 1),
            model="candidate",
            metrics=_metrics(triplet, Decimal("0.2000")),
        ),
        _observation(
            listing="B",
            anchor=date(2024, 1, 2),
            target=date(2024, 7, 1),
            model="candidate",
            metrics=_metrics(triplet, Decimal("0.2000")),
        ),
        _observation(
            listing="C",
            anchor=date(2024, 7, 1),
            target=date(2024, 12, 31),
            model="candidate",
            metrics=_metrics(triplet, Decimal("1.0000")),
        ),
    ]

    summaries = aggregate_projection_metrics(
        observations,
        expected_partitions=("validation",),
        expected_horizons=("6m", "12m"),
        expected_models=("candidate",),
    )
    summary = _summary_by_key(summaries)[("validation", "6m", "candidate")]
    empty = _summary_by_key(summaries)[("validation", "12m", "candidate")]

    assert summary.duplicate_observation_count == 1
    assert summary.target_cohort_count == 2
    assert summary.distinct_listing_count == 3
    assert summary.non_overlapping_target_count == 2
    assert summary.calendar_span is not None
    assert summary.calendar_span.first_anchor == date(2024, 1, 2)
    # Target 2024-07-01 cohort MAE average is 0.1; target 2024-12-31 is 1.0.
    assert summary.averages["median_absolute_error"] == Decimal("0.55")
    assert empty.insufficient_reason == "no_eligible_observations"
    assert empty.averages["median_absolute_error"] is None


def test_aligned_model_comparison_reports_paired_missingness_and_unavailable_baselines() -> None:
    candidate_triplet = LedgerTriplet(Decimal("-0.1"), Decimal("0.0"), Decimal("0.1"))
    baseline_triplet = LedgerTriplet(Decimal("-0.1"), Decimal("0.1"), Decimal("0.2"))
    unavailable = score_projection_metrics(
        predicted_returns=None,
        actual_return=Decimal("0.0"),
        config=load_price_product_config(),
    )
    observations = [
        _observation(
            listing="A",
            anchor=date(2024, 1, 2),
            target=date(2024, 7, 1),
            model="candidate",
            metrics=_metrics(candidate_triplet, Decimal("0.0000")),
        ),
        _observation(
            listing="A",
            anchor=date(2024, 1, 2),
            target=date(2024, 7, 1),
            model="baseline",
            metrics=_metrics(baseline_triplet, Decimal("0.0000")),
        ),
        _observation(
            listing="B",
            anchor=date(2024, 1, 2),
            target=date(2024, 7, 1),
            model="candidate",
            metrics=_metrics(candidate_triplet, Decimal("0.0000")),
        ),
        _observation(
            listing="C",
            anchor=date(2024, 7, 1),
            target=date(2024, 12, 31),
            model="baseline",
            metrics=_metrics(baseline_triplet, Decimal("0.0000")),
        ),
        _observation(
            listing="D",
            anchor=date(2024, 7, 1),
            target=date(2024, 12, 31),
            model="candidate",
            metrics=unavailable,
        ),
        _observation(
            listing="D",
            anchor=date(2024, 7, 1),
            target=date(2024, 12, 31),
            model="baseline",
            metrics=_metrics(baseline_triplet, Decimal("0.0000")),
        ),
    ]

    comparison = compare_aligned_models(
        observations,
        candidate_model="candidate",
        baseline_model="baseline",
        metric_name="median_absolute_error",
    )

    assert comparison.paired_observation_count == 1
    assert comparison.candidate_average == Decimal("0.0000")
    assert comparison.baseline_average == Decimal("0.1000")
    assert comparison.mean_difference_candidate_minus_baseline == Decimal("-0.1000")
    assert comparison.exclusions["baseline_missing"] == 1
    assert comparison.exclusions["candidate_missing"] == 1
    assert comparison.exclusions["candidate_unavailable:prediction_triplet_unavailable"] == 1


def test_convergence_thresholds_use_production_interval_width_and_exact_boundary() -> None:
    report = compare_numerical_convergence(
        production={"6m": (126, RawTriplet(-0.1000, 0.0000, 0.4000))},
        diagnostic={"6m": (126, RawTriplet(-0.0900, 0.0100, 0.4110))},
    )

    rows = {row.quantile: row for row in report.rows}
    assert rows["p20"].threshold == pytest.approx(0.01)
    assert rows["p20"].movement == pytest.approx(0.01)
    assert rows["p20"].exceeded is False
    assert rows["p50"].exceeded is False
    assert rows["p80"].movement == pytest.approx(0.011)
    assert rows["p80"].exceeded is True
    assert report.exceeded_count == 1
    assert report.threshold_rule == "max(0.01 return, 0.02 * production_interval_width)"


def test_convergence_invalid_triplet_reports_unavailable_rows_without_zeroes() -> None:
    report = compare_numerical_convergence(
        production={"6m": (126, RawTriplet(0.3, 0.2, 0.4))},
        diagnostic={"6m": (126, RawTriplet(0.3, 0.2, 0.4))},
    )

    assert report.unavailable_count == 3
    assert all(row.unavailable_reason == "convergence_triplet_invalid" for row in report.rows)
    assert all(row.movement is None and row.threshold is None for row in report.rows)

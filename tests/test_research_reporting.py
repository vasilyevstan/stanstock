from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import cast

import polars as pl
import pytest
from django.db import DataError, connection, transaction
from django.test.utils import CaptureQueriesContext

from stanstock.data.models import Company, Listing, Region, Security, UniverseSnapshot
from stanstock.research.models import (
    Prediction,
    PredictionOutcome,
    Recommendation,
    RiskClass,
    StockAnalysis,
)
from stanstock.research.outcomes import ResolvedOutcome, resolve_outcome
from stanstock.research.reporting import (
    _advisory_target_summaries,
    _build_advisory_support_report,
    _select_non_overlapping_summaries,
    advisory_support_report,
)

_VALID_REVISION = "a1" * 20
_NEUTRAL_MEDIUM_DISCLOSURE = (
    "Medium-horizon inclusion reports whether realized price returns fell inside "
    "the stored analog bear-to-bull ranges. Metrics remain withheld until the "
    "overlap-aware support floors pass."
)
_QUALIFIED_MEDIUM_DISCLOSURE = (
    "The 6- and 12-month analog ranges have a nominal 60% analog-range target. "
    "This is not a calibration claim or a coverage guarantee."
)
_UNSUPPORTED_DISCLOSURE = (
    "These groups have horizons outside 6m, 12m, 3y, and 5y. "
    "Their raw identities remain visible, but all metrics stay withheld "
    "because no support floors or inclusion semantics are defined for them."
)
_FLOORS = (
    (Prediction.Horizon.SIX_MONTH, 8, 30, 1095),
    (Prediction.Horizon.TWELVE_MONTH, 6, 30, 1460),
    (Prediction.Horizon.THREE_YEAR, 3, 30, 2190),
    (Prediction.Horizon.FIVE_YEAR, 3, 30, 3650),
)


def _summary(
    target: date,
    *,
    finish: date | None = None,
    row_count: int = 30,
    direction_true_count: int | None = None,
    inclusion_true_count: int | None = None,
    signed_error_sum: Decimal | None = None,
    horizon: str = Prediction.Horizon.SIX_MONTH,
    revision: str = _VALID_REVISION,
) -> dict[str, object]:
    direction_true_count = row_count if direction_true_count is None else direction_true_count
    inclusion_true_count = row_count if inclusion_true_count is None else inclusion_true_count
    signed_error_sum = (
        Decimal("0.01") * Decimal(row_count) if signed_error_sum is None else signed_error_sum
    )
    row: dict[str, object] = {
        "prediction__method_version": "synthetic-advisory-v1",
        "prediction__config_hash": "c" * 64,
        "prediction__price_provider": "synthetic_provider",
        "prediction__evidence_grade": UniverseSnapshot.Grade.OBSERVED,
        "prediction__horizon": horizon,
        "prediction__code_revision": revision,
        "prediction__target_date": target,
        "row_count": row_count,
        "listing_count": row_count,
        "evaluation_date_count": row_count,
        "max_evaluation_date": finish or target,
        "evaluation_before_target_count": 0,
        "actual_return_count": row_count,
        "min_actual_return": Decimal("-0.10"),
        "max_actual_return": Decimal("0.20"),
        "bear_return_count": row_count,
        "min_bear_return": Decimal("-0.20"),
        "max_bear_return": Decimal("-0.10"),
        "base_return_count": row_count,
        "min_base_return": Decimal("0"),
        "max_base_return": Decimal("0.10"),
        "bull_return_count": row_count,
        "min_bull_return": Decimal("0.20"),
        "max_bull_return": Decimal("0.30"),
        "direction_value_count": row_count,
        "direction_true_count": direction_true_count,
        "inclusion_value_count": row_count,
        "inclusion_true_count": inclusion_true_count,
        "signed_error_count": row_count,
        "min_signed_error": Decimal("-0.01"),
        "max_signed_error": Decimal("0.02"),
        "signed_error_sum": signed_error_sum,
        "ordered_scenario_count": row_count,
        "direction_consistent_count": row_count,
        "inclusion_consistent_count": row_count,
        "signed_error_consistent_count": row_count,
    }
    return row


def _support_rows(
    horizon: str,
    cohort_count: int,
    span_days: int,
    *,
    row_count: int = 30,
) -> list[dict[str, object]]:
    start = date(2010, 1, 1)
    if cohort_count == 1:
        offsets = [0]
    else:
        offsets = [(span_days * index) // (cohort_count - 1) for index in range(cohort_count)]
    return [
        _summary(start + timedelta(days=offset), row_count=row_count, horizon=horizon)
        for offset in offsets
    ]


def _groups(report: Mapping[str, object]) -> list[dict[str, object]]:
    sections = report["sections"]
    assert isinstance(sections, list)
    return [
        group
        for section in sections
        for group in section["groups"]  # type: ignore[index,union-attr]
    ]


@pytest.mark.parametrize(
    ("horizon", "minimum_effective", "minimum_listings", "minimum_span"),
    _FLOORS,
)
def test_all_advisory_support_floor_triples_withhold_just_below_and_publish_at_threshold(
    horizon: str,
    minimum_effective: int,
    minimum_listings: int,
    minimum_span: int,
) -> None:
    below_effective = _groups(
        _build_advisory_support_report(
            _support_rows(horizon, minimum_effective - 1, minimum_span),
            overflow=False,
        )
    )[0]
    below_listings = _groups(
        _build_advisory_support_report(
            _support_rows(
                horizon,
                minimum_effective,
                minimum_span,
                row_count=minimum_listings - 1,
            ),
            overflow=False,
        )
    )[0]
    below_span = _groups(
        _build_advisory_support_report(
            _support_rows(horizon, minimum_effective, minimum_span - 1),
            overflow=False,
        )
    )[0]
    at_threshold = _groups(
        _build_advisory_support_report(
            _support_rows(horizon, minimum_effective, minimum_span),
            overflow=False,
        )
    )[0]

    assert below_effective["publishable"] is False
    assert below_listings["publishable"] is False
    assert below_span["publishable"] is False
    assert at_threshold["publishable"] is True
    assert at_threshold["effective_cohort_count"] == minimum_effective
    assert at_threshold["target_span_days"] == minimum_span
    assert at_threshold["minimum_listings_per_selected_cohort"] == minimum_listings

    report = _build_advisory_support_report(
        _support_rows(horizon, minimum_effective, minimum_span),
        overflow=False,
    )
    medium_section = report["sections"][0]  # type: ignore[index]
    expected_disclosure = (
        _QUALIFIED_MEDIUM_DISCLOSURE
        if horizon in {Prediction.Horizon.SIX_MONTH, Prediction.Horizon.TWELVE_MONTH}
        else _NEUTRAL_MEDIUM_DISCLOSURE
    )
    assert medium_section["disclosure"] == expected_disclosure


def test_closed_intervals_touch_and_equal_finish_ties_are_deterministic() -> None:
    first = _summary(date(2020, 1, 1), finish=date(2020, 1, 10))
    touching = _summary(date(2020, 1, 10), finish=date(2020, 1, 11))
    after = _summary(date(2020, 1, 11), finish=date(2020, 1, 12))
    assert _select_non_overlapping_summaries([touching, after, first]) == [first, after]

    earlier_start = _summary(date(2021, 1, 1), finish=date(2021, 1, 5))
    later_start = _summary(date(2021, 1, 2), finish=date(2021, 1, 5))
    assert _select_non_overlapping_summaries([later_start, earlier_start]) == [earlier_start]


def test_selected_thin_cohort_withholds_without_broad_replacement() -> None:
    rows = _support_rows(Prediction.Horizon.SIX_MONTH, 8, 1095)
    first_target = rows[0]["prediction__target_date"]
    assert isinstance(first_target, date)
    rows[0] = _summary(
        first_target,
        finish=first_target + timedelta(days=10),
        row_count=29,
    )
    rows.append(
        _summary(
            first_target + timedelta(days=1),
            finish=first_target + timedelta(days=20),
        )
    )

    group = _groups(_build_advisory_support_report(rows, overflow=False))[0]

    assert group["candidate_cohort_count"] == 9
    assert group["effective_cohort_count"] == 8
    assert group["publishable"] is False
    assert any("below listing floor" in reason for reason in group["withheld_reasons"])


def test_overlapping_unselected_thin_cohort_does_not_fail_breadth() -> None:
    rows = _support_rows(Prediction.Horizon.SIX_MONTH, 8, 1095)
    first_target = rows[0]["prediction__target_date"]
    assert isinstance(first_target, date)
    rows[0] = _summary(first_target, finish=first_target + timedelta(days=10))
    rows.append(
        _summary(
            first_target + timedelta(days=1),
            finish=first_target + timedelta(days=20),
            row_count=29,
        )
    )

    group = _groups(_build_advisory_support_report(rows, overflow=False))[0]

    assert group["candidate_cohort_count"] == 9
    assert group["effective_cohort_count"] == 8
    assert group["publishable"] is True


def test_malformed_overlapping_unselected_evidence_withholds_before_scheduling() -> None:
    rows = _support_rows(Prediction.Horizon.SIX_MONTH, 8, 1095)
    first_target = rows[0]["prediction__target_date"]
    assert isinstance(first_target, date)
    malformed = _summary(
        first_target + timedelta(days=1),
        finish=first_target + timedelta(days=20),
    )
    malformed["signed_error_consistent_count"] = 29
    rows.append(malformed)

    group = _groups(_build_advisory_support_report(rows, overflow=False))[0]

    assert group["candidate_cohort_count"] == 8
    assert group["effective_cohort_count"] is None
    assert group["publishable"] is False
    assert "Malformed target-date evidence — metrics withheld" in group["withheld_reasons"]


def test_metrics_weight_selected_dates_equally_despite_unequal_listing_counts() -> None:
    rows = _support_rows(Prediction.Horizon.SIX_MONTH, 8, 1095)
    rows[0] = _summary(
        rows[0]["prediction__target_date"],  # type: ignore[arg-type]
        row_count=60,
        direction_true_count=60,
        inclusion_true_count=60,
        signed_error_sum=Decimal("6"),
    )
    for index in range(1, len(rows)):
        rows[index]["direction_true_count"] = 0
        rows[index]["inclusion_true_count"] = 0
        rows[index]["signed_error_sum"] = Decimal(0)

    group = _groups(_build_advisory_support_report(rows, overflow=False))[0]

    assert group["publishable"] is True
    assert group["base_sign_match"] == Decimal(1) / Decimal(8)
    assert group["inclusion_rate"] == Decimal(1) / Decimal(8)
    assert group["mean_signed_base_error"] == Decimal("0.1") / Decimal(8)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("actual_return_count", 29),
        ("ordered_scenario_count", 29),
        ("direction_consistent_count", 29),
        ("inclusion_consistent_count", 29),
        ("signed_error_consistent_count", 29),
        ("evaluation_before_target_count", 1),
    ),
)
def test_completeness_and_consistency_use_one_shared_metric_gate(
    field: str,
    value: int,
) -> None:
    rows = _support_rows(Prediction.Horizon.SIX_MONTH, 8, 1095)
    rows[-1][field] = value

    group = _groups(_build_advisory_support_report(rows, overflow=False))[0]

    assert group["publishable"] is False
    assert group["base_sign_match"] is None
    assert group["inclusion_rate"] is None
    assert group["mean_signed_base_error"] is None


_DECIMAL_SUMMARY_FIELDS = (
    "min_actual_return",
    "max_actual_return",
    "min_bear_return",
    "max_bear_return",
    "min_base_return",
    "max_base_return",
    "min_bull_return",
    "max_bull_return",
    "min_signed_error",
    "max_signed_error",
    "signed_error_sum",
)
_INVALID_DECIMAL_CASES = (
    ("missing", None),
    ("wrong-type", "0"),
    ("nan", Decimal("NaN")),
    ("positive-infinity", Decimal("Infinity")),
    ("negative-infinity", Decimal("-Infinity")),
)


@pytest.mark.parametrize("field", _DECIMAL_SUMMARY_FIELDS)
@pytest.mark.parametrize(("case", "invalid"), _INVALID_DECIMAL_CASES)
def test_missing_wrong_type_and_nonfinite_decimal_summaries_fail_closed(
    field: str,
    case: str,
    invalid: object,
) -> None:
    rows = _support_rows(Prediction.Horizon.SIX_MONTH, 8, 1095)
    if case == "missing":
        rows[-1].pop(field)
    else:
        rows[-1][field] = invalid

    group = _groups(_build_advisory_support_report(rows, overflow=False))[0]

    assert group["publishable"] is False
    assert group["withheld_reasons"] == ("Malformed target-date evidence — metrics withheld",)
    assert group["base_sign_match"] is None
    assert group["inclusion_rate"] is None
    assert group["mean_signed_base_error"] is None


def test_empty_and_all_withheld_medium_sections_keep_neutral_disclosure() -> None:
    empty = _build_advisory_support_report([], overflow=False)
    withheld = _build_advisory_support_report(
        [_summary(date(2020, 1, 1))],
        overflow=False,
    )

    assert empty["sections"][0]["disclosure"] == _NEUTRAL_MEDIUM_DISCLOSURE  # type: ignore[index]
    assert withheld["sections"][0]["disclosure"] == _NEUTRAL_MEDIUM_DISCLOSURE  # type: ignore[index]


def test_empty_advisory_report_has_three_fresh_sections_in_exact_order() -> None:
    expected_sections = [
        {
            "key": "medium",
            "title": "Medium-horizon advisory support",
            "disclosure": _NEUTRAL_MEDIUM_DISCLOSURE,
            "inclusion_label": "Analog-range inclusion",
            "groups": [],
        },
        {
            "key": "long",
            "title": "Long-horizon advisory support",
            "disclosure": (
                "The 3- and 5-year bear, base, and bull values are deterministic "
                "scenario cases. Inclusion reports whether the realized price "
                "return fell inside that scenario envelope."
            ),
            "inclusion_label": "Scenario-envelope inclusion",
            "groups": [],
        },
        {
            "key": "unsupported",
            "title": "Unsupported advisory horizons",
            "disclosure": _UNSUPPORTED_DISCLOSURE,
            "inclusion_label": "Inclusion metric (withheld)",
            "groups": [],
        },
    ]
    first = _build_advisory_support_report([], overflow=False)
    second = _build_advisory_support_report([], overflow=False)

    assert first == {
        "overflow": False,
        "summary_count": 0,
        "has_evidence": False,
        "sections": expected_sections,
    }
    first_sections = cast(list[dict[str, object]], first["sections"])
    first_groups = cast(list[dict[str, object]], first_sections[0]["groups"])
    first_groups.append({"sentinel": True})
    assert second["sections"] == expected_sections


def test_unsupported_short_group_is_visible_once_and_fully_withheld() -> None:
    raw_revision = _VALID_REVISION
    report = _build_advisory_support_report(
        [
            _summary(
                date(2020, 1, 1),
                horizon=Prediction.Horizon.SHORT,
                revision=raw_revision,
            )
        ],
        overflow=False,
    )
    medium, long, unsupported = cast(list[dict[str, object]], report["sections"])
    unsupported_groups = cast(list[dict[str, object]], unsupported["groups"])

    assert medium["groups"] == []
    assert long["groups"] == []
    assert len(unsupported_groups) == 1
    group = unsupported_groups[0]
    assert group["horizon"] == Prediction.Horizon.SHORT
    assert group["code_revision"] == raw_revision
    assert group["revision_label"] == raw_revision
    assert group["status_label"] == "Metrics withheld"
    assert group["withheld_reasons"] == ("Unsupported advisory horizon — metrics withheld",)
    assert group["minimum_effective_cohorts"] == 0
    assert group["minimum_listings_per_selected_cohort"] == 0
    assert group["minimum_target_span_days"] == 0
    assert group["base_sign_match"] is None
    assert group["inclusion_rate"] is None
    assert group["mean_signed_base_error"] is None


def test_supported_and_unsupported_horizons_route_exclusively_in_stable_order() -> None:
    report = _build_advisory_support_report(
        [
            _summary(date(2020, 1, 1), horizon=horizon)
            for horizon in (
                Prediction.Horizon.FIVE_YEAR,
                Prediction.Horizon.SHORT,
                Prediction.Horizon.SIX_MONTH,
                Prediction.Horizon.THREE_YEAR,
                Prediction.Horizon.TWELVE_MONTH,
            )
        ],
        overflow=False,
    )
    sections = cast(list[dict[str, object]], report["sections"])

    assert [section["key"] for section in sections] == [
        "medium",
        "long",
        "unsupported",
    ]
    assert [
        [group["horizon"] for group in cast(list[dict[str, object]], section["groups"])]
        for section in sections
    ] == [
        [Prediction.Horizon.TWELVE_MONTH, Prediction.Horizon.SIX_MONTH],
        [Prediction.Horizon.THREE_YEAR, Prediction.Horizon.FIVE_YEAR],
        [Prediction.Horizon.SHORT],
    ]


def test_unsupported_horizon_alone_cannot_trigger_nominal_60_disclosure() -> None:
    report = _build_advisory_support_report(
        [_summary(date(2020, 1, 1), horizon=Prediction.Horizon.SHORT)],
        overflow=False,
    )

    assert report["sections"][0]["disclosure"] == _NEUTRAL_MEDIUM_DISCLOSURE  # type: ignore[index]
    assert "nominal 60%" not in str(report)


def test_exact_revision_validation_keeps_raw_revision_groups_separate() -> None:
    revisions = (
        _VALID_REVISION,
        _VALID_REVISION.upper(),
        "a" * 39,
        "a" * 41,
        "g" * 40,
        f" {_VALID_REVISION}",
        f"{_VALID_REVISION} ",
        "",
    )
    rows = [_summary(date(2020, 1, 1), revision=revision) for revision in revisions]

    groups = _groups(_build_advisory_support_report(rows, overflow=False))
    by_revision = {group["code_revision"]: group for group in groups}

    assert set(by_revision) == set(revisions)
    assert by_revision[_VALID_REVISION]["revision_valid"] is True
    for revision in revisions[1:]:
        assert by_revision[revision]["revision_valid"] is False
        assert (
            "Invalid code revision — metrics withheld" in by_revision[revision]["withheld_reasons"]
        )
    assert by_revision[""]["revision_label"] == "(empty)"
    assert by_revision[_VALID_REVISION.upper()]["revision_label"] == _VALID_REVISION.upper()


def test_summary_limit_accepts_50_000_and_globally_withholds_50_001() -> None:
    row = _summary(date(2020, 1, 1))
    accepted = _build_advisory_support_report([row] * 50_000, overflow=False)
    withheld = _build_advisory_support_report([row] * 50_001, overflow=True)

    assert accepted["overflow"] is False
    assert accepted["summary_count"] == 50_000
    assert _groups(accepted)
    assert withheld == {
        "overflow": True,
        "summary_count": None,
        "has_evidence": True,
        "sections": [
            {
                "key": "medium",
                "title": "Medium-horizon advisory support",
                "disclosure": _NEUTRAL_MEDIUM_DISCLOSURE,
                "inclusion_label": "Analog-range inclusion",
                "groups": [],
            },
            {
                "key": "long",
                "title": "Long-horizon advisory support",
                "disclosure": (
                    "The 3- and 5-year bear, base, and bull values are "
                    "deterministic scenario cases. Inclusion reports whether "
                    "the realized price return fell inside that scenario envelope."
                ),
                "inclusion_label": "Scenario-envelope inclusion",
                "groups": [],
            },
            {
                "key": "unsupported",
                "title": "Unsupported advisory horizons",
                "disclosure": _UNSUPPORTED_DISCLOSURE,
                "inclusion_label": "Inclusion metric (withheld)",
                "groups": [],
            },
        ],
    }


@pytest.mark.django_db
def test_advisory_report_executes_one_grouped_query_without_boolean_avg() -> None:
    queryset = _advisory_target_summaries()
    sql = str(queryset.query).upper()
    group_by = queryset.query.group_by

    assert "GROUP BY" in sql
    assert group_by is not None
    assert {
        expression.target.name  # type: ignore[attr-defined]
        for expression in group_by
    } == {
        "method_version",
        "config_hash",
        "price_provider",
        "evidence_grade",
        "horizon",
        "code_revision",
        "target_date",
    }
    assert len(group_by) == 7
    assert "_stored_error_delta" not in queryset.query.annotation_select
    grouped_sql = sql.split("GROUP BY", maxsplit=1)[1].split("ORDER BY", maxsplit=1)[0]
    assert "ROUND(" not in grouped_sql
    assert "ROUND(" in sql
    assert ", 4)" in sql
    assert "FILTER (WHERE" in sql
    assert "AVG(" not in sql

    with CaptureQueriesContext(connection) as captured:
        report = advisory_support_report()

    assert len(captured) == 1
    assert report["summary_count"] == 0


def _prepare_reportable_run(persisted_analysis: StockAnalysis) -> None:
    run = persisted_analysis.run
    run.issued_on_time = True
    run.save(update_fields=["issued_on_time"])
    snapshot = run.universe_snapshot
    snapshot.grade = UniverseSnapshot.Grade.OBSERVED
    snapshot.save(update_fields=["grade"])


def _case_analysis(persisted_analysis: StockAnalysis, index: int) -> StockAnalysis:
    if index == 0:
        return persisted_analysis
    company = Company.objects.create(name=f"Reporting Synthetic {index}", country="US")
    listing = Listing.objects.create(
        security=Security.objects.create(company=company, name=f"Reporting Synthetic {index}"),
        ticker=f"RPT-{index}",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    return StockAnalysis.objects.create(
        run=persisted_analysis.run,
        listing=listing,
        current_price=Decimal("100"),
        overall_score=Decimal("50"),
        recommendation=Recommendation.HOLD,
        risk_score=Decimal("40"),
        risk_class=RiskClass.MEDIUM,
        confidence=Decimal("50"),
    )


def _case_prediction(
    persisted_analysis: StockAnalysis,
    index: int,
    *,
    method_version: str,
    base_return: Decimal = Decimal("0"),
    bear_return: Decimal = Decimal("-0.1"),
    bull_return: Decimal = Decimal("0.2"),
) -> Prediction:
    _prepare_reportable_run(persisted_analysis)
    analysis = _case_analysis(persisted_analysis, index)
    run = analysis.run
    return Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=run.generated_at,
        target_date=run.target_date,
        issued_on_time=True,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        evidence_grade=UniverseSnapshot.Grade.OBSERVED,
        source_mode=Prediction.SourceMode.PROVIDER,
        price_provider="synthetic_provider",
        price_subject=analysis.listing.ticker,
        price_at_prediction=Decimal("100"),
        bear_return=bear_return,
        base_return=base_return,
        bull_return=bull_return,
        confidence=Decimal("50"),
        confidence_status="synthetic",
        recommendation=Recommendation.HOLD,
        overall_score=Decimal("50"),
        model_version=f"{method_version}-{index}",
        method_version=method_version,
        config_hash="c" * 64,
        data_cutoff=run.data_cutoff,
        code_revision=_VALID_REVISION,
    )


def _case_outcome(
    prediction: Prediction,
    *,
    actual_return: Decimal,
    direction_correct: bool,
    interval_covered: bool,
    signed_error: Decimal,
) -> PredictionOutcome:
    evaluated = prediction.target_date + timedelta(days=200)
    return PredictionOutcome.objects.create(
        prediction=prediction,
        evaluated_at=datetime.combine(evaluated, datetime.min.time(), tzinfo=UTC),
        evaluation_date=evaluated,
        status=PredictionOutcome.Status.MATURED,
        actual_return=actual_return,
        success=None,
        direction_correct=direction_correct,
        interval_covered=interval_covered,
        signed_error=signed_error,
        resolution="Synthetic advisory evidence",
    )


def _persist_resolved(
    prediction: Prediction,
    resolved: ResolvedOutcome,
    *,
    evaluated_at: datetime,
) -> PredictionOutcome:
    return PredictionOutcome.objects.create(
        prediction=prediction,
        evaluated_at=evaluated_at,
        evaluation_date=resolved.evaluation_date,
        status=resolved.status,
        actual_return=resolved.actual_return,
        benchmark_return=resolved.benchmark_return,
        success=resolved.success,
        direction_correct=resolved.direction_correct,
        interval_covered=resolved.interval_covered,
        resolution=resolved.resolution,
        error=resolved.error,
        signed_error=resolved.signed_error,
        metadata=resolved.metadata,
    )


def _producer_boundary_outcome(
    persisted_analysis: StockAnalysis,
    *,
    index: int,
    final_close: float,
    base_return: Decimal,
    method_version: str = "producer-boundary-v1",
) -> tuple[ResolvedOutcome, PredictionOutcome]:
    prediction = _case_prediction(
        persisted_analysis,
        index,
        method_version=method_version,
        base_return=base_return,
        bear_return=Decimal("-0.1000"),
        bull_return=Decimal("0.1000"),
    )
    dates = [prediction.target_date + timedelta(days=offset) for offset in range(127)]
    closes = [100.0] * 126 + [final_close]
    frame = pl.DataFrame({"date": dates, "close": closes})
    evaluation_date = dates[-1]
    evaluated_at = datetime.combine(evaluation_date, datetime.min.time(), tzinfo=UTC)

    def price_loader(subject: str, through_date: date) -> pl.DataFrame:
        assert subject == prediction.price_subject
        assert through_date == evaluation_date
        return frame

    resolved = resolve_outcome(
        prediction,
        provider="synthetic_provider",
        evaluation_date=evaluation_date,
        evaluated_at=evaluated_at,
        benchmark_subject=None,
        price_loader=price_loader,
    )
    outcome = _persist_resolved(prediction, resolved, evaluated_at=evaluated_at)
    return resolved, outcome


@pytest.mark.django_db
def test_direction_consistency_allows_zero_ambiguity_and_rejects_wrong_nonzero_booleans(
    persisted_analysis: StockAnalysis,
) -> None:
    cases = (
        (Decimal("0"), Decimal("-0.05"), True),
        (Decimal("0"), Decimal("0.05"), False),
        (Decimal("-0.10"), Decimal("-0.05"), False),
        (Decimal("0.10"), Decimal("0.05"), False),
    )
    for index, (actual, base, direction) in enumerate(cases):
        prediction = _case_prediction(
            persisted_analysis,
            index,
            method_version="direction-lattice-v1",
            base_return=base,
        )
        _case_outcome(
            prediction,
            actual_return=actual,
            direction_correct=direction,
            interval_covered=True,
            signed_error=actual - base,
        )

    row = list(_advisory_target_summaries())[0]

    assert row["row_count"] == 4
    assert row["direction_value_count"] == 4
    assert row["direction_consistent_count"] == 2


@pytest.mark.django_db
def test_inclusion_consistency_allows_false_endpoints_and_rejects_wrong_interior_exterior(
    persisted_analysis: StockAnalysis,
) -> None:
    cases = (
        (Decimal("-0.1"), False),
        (Decimal("0.2"), False),
        (Decimal("0.1"), False),
        (Decimal("0.3"), True),
    )
    for index, (actual, inclusion) in enumerate(cases):
        prediction = _case_prediction(
            persisted_analysis,
            index,
            method_version="inclusion-lattice-v1",
        )
        _case_outcome(
            prediction,
            actual_return=actual,
            direction_correct=False,
            interval_covered=inclusion,
            signed_error=actual,
        )

    row = list(_advisory_target_summaries())[0]

    assert row["row_count"] == 4
    assert row["inclusion_value_count"] == 4
    assert row["inclusion_consistent_count"] == 2


@pytest.mark.django_db
def test_sqlite_signed_error_delta_lattice_accepts_ten_thousandth_boundary_only(
    persisted_analysis: StockAnalysis,
) -> None:
    if connection.vendor != "sqlite":
        pytest.skip("SQLite-specific signed-error lattice regression")
    deltas = (
        Decimal("0"),
        Decimal("0.0001"),
        Decimal("-0.0001"),
        Decimal("0.0002"),
        Decimal("-0.0002"),
    )
    for index, delta in enumerate(deltas):
        prediction = _case_prediction(
            persisted_analysis,
            index,
            method_version="sqlite-delta-lattice-v1",
        )
        actual = Decimal("0.1000")
        _case_outcome(
            prediction,
            actual_return=actual,
            direction_correct=False,
            interval_covered=True,
            signed_error=actual - delta,
        )

    row = list(_advisory_target_summaries())[0]

    assert row["row_count"] == 5
    assert row["signed_error_consistent_count"] == 3


@pytest.mark.django_db
def test_producer_boundary_fields_are_persisted_exactly_and_group_is_well_formed(
    persisted_analysis: StockAnalysis,
) -> None:
    cases = (
        (0, 100.004, Decimal("0.0000"), Decimal("0.0000"), False, True),
        (1, 110.004, Decimal("0.1000"), Decimal("0.1000"), True, False),
    )
    for index, final_close, base_return, actual_return, direction, inclusion in cases:
        resolved, outcome = _producer_boundary_outcome(
            persisted_analysis,
            index=index,
            final_close=final_close,
            base_return=base_return,
        )
        outcome.refresh_from_db()

        assert resolved.actual_return == actual_return
        assert resolved.signed_error == Decimal("0.0000")
        assert resolved.direction_correct is direction
        assert resolved.interval_covered is inclusion
        for field in (
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
        ):
            assert getattr(outcome, field) == getattr(resolved, field)

    rows = list(_advisory_target_summaries())

    assert len(rows) == 1
    row = rows[0]
    assert row["row_count"] == 2
    assert row["listing_count"] == 2
    assert row["direction_consistent_count"] == 2
    assert row["inclusion_consistent_count"] == 2
    assert row["signed_error_consistent_count"] == 2
    group = _groups(advisory_support_report())[0]
    assert "Malformed target-date evidence — metrics withheld" not in group["withheld_reasons"]


@pytest.mark.django_db
def test_postgresql_numeric_nonfinite_rows_withhold_exact_group(
    persisted_analysis: StockAnalysis,
) -> None:
    if connection.vendor != "postgresql":
        pytest.skip("requires PostgreSQL numeric non-finite behavior")

    for index, final_close, base_return in (
        (0, 100.004, Decimal("0.0000")),
        (1, 110.004, Decimal("0.1000")),
    ):
        _producer_boundary_outcome(
            persisted_analysis,
            index=index,
            final_close=final_close,
            base_return=base_return,
            method_version="pg-producer-boundary-v1",
        )
    corrupt_prediction = _case_prediction(
        persisted_analysis,
        2,
        method_version="pg-corrupt-nan-v1",
    )
    _case_outcome(
        corrupt_prediction,
        actual_return=Decimal("0.1000"),
        direction_correct=False,
        interval_covered=True,
        signed_error=Decimal("0.1000"),
    )
    table = connection.ops.quote_name(PredictionOutcome._meta.db_table)
    with connection.cursor() as cursor:
        cursor.execute(
            f"UPDATE {table} SET actual_return = %s WHERE prediction_id = %s",
            ["NaN", corrupt_prediction.pk],
        )

    grouped_rows = {row["prediction__method_version"]: row for row in _advisory_target_summaries()}
    assert grouped_rows["pg-producer-boundary-v1"]["row_count"] == 2
    assert grouped_rows["pg-producer-boundary-v1"]["direction_consistent_count"] == 2
    assert grouped_rows["pg-producer-boundary-v1"]["inclusion_consistent_count"] == 2
    assert grouped_rows["pg-producer-boundary-v1"]["signed_error_consistent_count"] == 2
    assert grouped_rows["pg-corrupt-nan-v1"]["row_count"] == 1
    assert grouped_rows["pg-corrupt-nan-v1"]["actual_return_count"] == 1
    report_groups = {group["method_version"]: group for group in _groups(advisory_support_report())}
    assert (
        "Malformed target-date evidence — metrics withheld"
        not in report_groups["pg-producer-boundary-v1"]["withheld_reasons"]
    )
    assert report_groups["pg-corrupt-nan-v1"]["withheld_reasons"] == (
        "Malformed target-date evidence — metrics withheld",
    )

    for offset, delta in enumerate(
        (
            Decimal("0.0001"),
            Decimal("-0.0001"),
            Decimal("0.0002"),
            Decimal("-0.0002"),
        ),
        start=3,
    ):
        prediction = _case_prediction(
            persisted_analysis,
            offset,
            method_version="pg-delta-lattice-v1",
        )
        actual = Decimal("0.1000")
        _case_outcome(
            prediction,
            actual_return=actual,
            direction_correct=False,
            interval_covered=True,
            signed_error=actual - delta,
        )
    grouped_rows = {row["prediction__method_version"]: row for row in _advisory_target_summaries()}
    assert grouped_rows["pg-delta-lattice-v1"]["row_count"] == 4
    assert grouped_rows["pg-delta-lattice-v1"]["signed_error_consistent_count"] == 2

    infinity_prediction = _case_prediction(
        persisted_analysis,
        7,
        method_version="pg-infinity-rejected-v1",
    )
    infinity_outcome = _case_outcome(
        infinity_prediction,
        actual_return=Decimal("0.1000"),
        direction_correct=False,
        interval_covered=True,
        signed_error=Decimal("0.1000"),
    )
    with transaction.atomic():
        for nonfinite in ("Infinity", "-Infinity"):
            with pytest.raises(DataError):
                with transaction.atomic():
                    with connection.cursor() as cursor:
                        cursor.execute(
                            f"UPDATE {table} SET actual_return = %s WHERE prediction_id = %s",
                            [nonfinite, infinity_prediction.pk],
                        )
            assert connection.needs_rollback is False
            infinity_outcome.refresh_from_db()
            assert infinity_outcome.actual_return == Decimal("0.1000")
            assert advisory_support_report()["has_evidence"] is True

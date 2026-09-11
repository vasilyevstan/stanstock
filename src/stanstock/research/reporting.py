"""Lazy predicates for reportable/canonical aggregate performance reporting.

``reportable_prediction_filter`` -- is this prediction observed, on-time,
provider-backed evidence at all?

``canonical_reportable_prediction_filter`` -- is it also the earliest
reportable prediction for its exact observation key -- ``(listing,
target_date, horizon, evidence_role, method_version, config_hash,
price_provider)``, deliberately excluding ``model_version``/run identity so
a later on-time reissue of the same observation does not get double-counted?

Both return lazy ``Q`` objects usable directly on a ``Prediction`` queryset
(``prefix=""``) or traversed from ``PredictionOutcome`` via its
``prediction`` relation (``prefix="prediction__"``). Neither executes a
query; the canonical helper compiles its "no earlier sibling" check to a
correlated SQL ``NOT EXISTS`` via ``Exists``/``OuterRef``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal, DecimalException
from typing import cast

from django.db.models import (
    Count,
    DecimalField,
    Exists,
    ExpressionWrapper,
    F,
    Max,
    Min,
    OuterRef,
    Q,
    QuerySet,
    Sum,
)
from django.db.models.functions import Round

from stanstock.data.models import UniverseSnapshot
from stanstock.research.models import Prediction, PredictionOutcome

# Deliberately excludes `model_version` and any run/analysis identifier:
# those distinguish *versions* of the same observation, not distinct ones.
_OBSERVATION_KEY_FIELDS: tuple[str, ...] = (
    "listing_id",
    "target_date",
    "horizon",
    "evidence_role",
    "method_version",
    "config_hash",
    "price_provider",
)

_ADVISORY_IDENTITY_FIELDS: tuple[str, ...] = (
    "prediction__method_version",
    "prediction__config_hash",
    "prediction__price_provider",
    "prediction__evidence_grade",
    "prediction__horizon",
    "prediction__code_revision",
)
_ADVISORY_COUNT_FIELDS: tuple[str, ...] = (
    "row_count",
    "listing_count",
    "evaluation_date_count",
    "evaluation_before_target_count",
    "actual_return_count",
    "bear_return_count",
    "base_return_count",
    "bull_return_count",
    "direction_value_count",
    "direction_true_count",
    "inclusion_value_count",
    "inclusion_true_count",
    "signed_error_count",
    "ordered_scenario_count",
    "direction_consistent_count",
    "inclusion_consistent_count",
    "signed_error_consistent_count",
)
_ADVISORY_COMPLETENESS_AND_CONSISTENCY_FIELDS: tuple[str, ...] = (
    "evaluation_date_count",
    "actual_return_count",
    "bear_return_count",
    "base_return_count",
    "bull_return_count",
    "direction_value_count",
    "inclusion_value_count",
    "signed_error_count",
    "ordered_scenario_count",
    "direction_consistent_count",
    "inclusion_consistent_count",
    "signed_error_consistent_count",
)
_ADVISORY_DECIMAL_FIELDS: tuple[str, ...] = (
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
_ADVISORY_FLOORS: dict[str, tuple[int, int, int]] = {
    Prediction.Horizon.SIX_MONTH: (8, 30, 1095),
    Prediction.Horizon.TWELVE_MONTH: (6, 30, 1460),
    Prediction.Horizon.THREE_YEAR: (3, 30, 2190),
    Prediction.Horizon.FIVE_YEAR: (3, 30, 3650),
}
_VALID_CODE_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_ADVISORY_SUMMARY_LIMIT = 50_000

_MEDIUM_SECTION = {
    "key": "medium",
    "title": "Medium-horizon advisory support",
    "disclosure": (
        "Medium-horizon inclusion reports whether realized price returns fell inside "
        "the stored analog bear-to-bull ranges. Metrics remain withheld until the "
        "overlap-aware support floors pass."
    ),
    "inclusion_label": "Analog-range inclusion",
}
_QUALIFIED_MEDIUM_DISCLOSURE = (
    "The 6- and 12-month analog ranges have a nominal 60% analog-range target. "
    "This is not a calibration claim or a coverage guarantee."
)
_LONG_SECTION = {
    "key": "long",
    "title": "Long-horizon advisory support",
    "disclosure": (
        "The 3- and 5-year bear, base, and bull values are deterministic scenario "
        "cases. Inclusion reports whether the realized price return fell inside "
        "that scenario envelope."
    ),
    "inclusion_label": "Scenario-envelope inclusion",
}
_UNSUPPORTED_SECTION = {
    "key": "unsupported",
    "title": "Unsupported advisory horizons",
    "disclosure": (
        "These groups have horizons outside 6m, 12m, 3y, and 5y. "
        "Their raw identities remain visible, but all metrics stay withheld "
        "because no support floors or inclusion semantics are defined for them."
    ),
    "inclusion_label": "Inclusion metric (withheld)",
}


def reportable_prediction_filter(prefix: str = "") -> Q:
    """Observed, on-time, provider-backed prediction evidence: prediction and
    parent-run ``issued_on_time=True``, ``evidence_grade=OBSERVED``,
    ``source_mode=PROVIDER``, non-empty ``price_provider``."""
    return Q(
        **{
            f"{prefix}evidence_grade": UniverseSnapshot.Grade.OBSERVED,
            f"{prefix}source_mode": Prediction.SourceMode.PROVIDER,
            f"{prefix}issued_on_time": True,
            f"{prefix}analysis__run__issued_on_time": True,
        }
    ) & ~Q(**{f"{prefix}price_provider": ""})


def _no_earlier_reportable_sibling(prefix: str) -> Exists:
    """``NOT EXISTS`` a strictly-earlier (``generated_at`` then UUID ``id``
    ascending) reportable ``Prediction`` sharing this row's observation key.
    Repeats only reportability -- no outcome/status predicate -- so an
    earlier reportable-but-unresolved/corporate-event/withheld row is never
    displaced by a later matured sibling, while an earlier non-reportable row
    never suppresses a later reportable one."""
    key_filter = {field: OuterRef(f"{prefix}{field}") for field in _OBSERVATION_KEY_FIELDS}
    earlier_siblings = (
        Prediction.objects.filter(**key_filter)
        .filter(reportable_prediction_filter())
        .filter(
            Q(generated_at__lt=OuterRef(f"{prefix}generated_at"))
            | Q(
                generated_at=OuterRef(f"{prefix}generated_at"),
                id__lt=OuterRef(f"{prefix}id"),
            )
        )
        .order_by()
    )
    return Exists(earlier_siblings)


def canonical_reportable_prediction_filter(prefix: str = "") -> Q:
    """``reportable AND NOT EXISTS(earlier reportable sibling)``: the earliest
    reportable prediction for its exact observation key. Later valid
    reportable reissues stay in the immutable ledger, evaluated per version,
    but are excluded here so one observation is counted once."""
    return cast(Q, reportable_prediction_filter(prefix) & ~_no_earlier_reportable_sibling(prefix))


def _advisory_target_summaries() -> QuerySet[dict[str, object]]:  # type: ignore[type-var]
    """Return one integrity summary per exact advisory identity and target date.

    Canonical selection is applied before maturity and metric predicates. Null
    and inconsistent values deliberately remain in the query so the report
    can withhold their entire exact evidence group rather than hiding them.
    """
    same_direction = Q(actual_return__lt=0, prediction__base_return__lt=0) | Q(
        actual_return__gt=0, prediction__base_return__gt=0
    )
    different_direction = Q(actual_return__lt=0, prediction__base_return__gte=0) | Q(
        actual_return__gt=0, prediction__base_return__lte=0
    )
    inside_scenario = Q(
        actual_return__gte=F("prediction__bear_return"),
        actual_return__lte=F("prediction__bull_return"),
    )
    outside_scenario = Q(actual_return__lte=F("prediction__bear_return")) | Q(
        actual_return__gte=F("prediction__bull_return")
    )
    delta_output = DecimalField(max_digits=12, decimal_places=4)
    stored_error_delta = Round(
        ExpressionWrapper(
            F("actual_return") - F("signed_error") - F("prediction__base_return"),
            output_field=delta_output,
        ),
        precision=4,
        output_field=delta_output,
    )

    canonical_advisory = PredictionOutcome.objects.filter(
        canonical_reportable_prediction_filter("prediction__"),
        prediction__evidence_role=Prediction.EvidenceRole.ADVISORY,
    )
    return (  # type: ignore[no-any-return]
        canonical_advisory.filter(status=PredictionOutcome.Status.MATURED)
        .alias(_stored_error_delta=stored_error_delta)
        .values(*_ADVISORY_IDENTITY_FIELDS, "prediction__target_date")
        .annotate(
            row_count=Count("prediction"),
            listing_count=Count("prediction__listing", distinct=True),
            evaluation_date_count=Count("evaluation_date"),
            max_evaluation_date=Max("evaluation_date"),
            evaluation_before_target_count=Count(
                "prediction",
                filter=Q(evaluation_date__lt=F("prediction__target_date")),
            ),
            actual_return_count=Count("actual_return"),
            min_actual_return=Min("actual_return"),
            max_actual_return=Max("actual_return"),
            bear_return_count=Count("prediction__bear_return"),
            min_bear_return=Min("prediction__bear_return"),
            max_bear_return=Max("prediction__bear_return"),
            base_return_count=Count("prediction__base_return"),
            min_base_return=Min("prediction__base_return"),
            max_base_return=Max("prediction__base_return"),
            bull_return_count=Count("prediction__bull_return"),
            min_bull_return=Min("prediction__bull_return"),
            max_bull_return=Max("prediction__bull_return"),
            direction_value_count=Count("direction_correct"),
            direction_true_count=Count(
                "prediction",
                filter=Q(direction_correct=True),
            ),
            inclusion_value_count=Count("interval_covered"),
            inclusion_true_count=Count(
                "prediction",
                filter=Q(interval_covered=True),
            ),
            signed_error_count=Count("signed_error"),
            min_signed_error=Min("signed_error"),
            max_signed_error=Max("signed_error"),
            signed_error_sum=Sum("signed_error"),
            ordered_scenario_count=Count(
                "prediction",
                filter=Q(
                    prediction__bear_return__lte=F("prediction__base_return"),
                    prediction__base_return__lte=F("prediction__bull_return"),
                ),
            ),
            direction_consistent_count=Count(
                "prediction",
                filter=Q(actual_return=0, direction_correct__isnull=False)
                | (Q(direction_correct=True) & same_direction)
                | (Q(direction_correct=False) & different_direction),
            ),
            inclusion_consistent_count=Count(
                "prediction",
                filter=(Q(interval_covered=True) & inside_scenario)
                | (Q(interval_covered=False) & outside_scenario),
            ),
            signed_error_consistent_count=Count(
                "prediction",
                filter=Q(
                    _stored_error_delta__gte=Decimal("-0.0001"),
                    _stored_error_delta__lte=Decimal("0.0001"),
                ),
            ),
        )
        .order_by(*_ADVISORY_IDENTITY_FIELDS, "prediction__target_date")
    )


def advisory_support_report() -> dict[str, object]:
    """Build the complete bounded advisory evidence report with one query."""
    rows = list(_advisory_target_summaries()[: _ADVISORY_SUMMARY_LIMIT + 1])
    return _build_advisory_support_report(
        rows,
        overflow=len(rows) > _ADVISORY_SUMMARY_LIMIT,
    )


def _empty_advisory_sections() -> list[dict[str, object]]:
    return [
        {**_MEDIUM_SECTION, "groups": []},
        {**_LONG_SECTION, "groups": []},
        {**_UNSUPPORTED_SECTION, "groups": []},
    ]


def _identity_text(value: object) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _is_date(value: object) -> bool:
    return isinstance(value, date) and not isinstance(value, datetime)


def _summary_is_well_formed(row: Mapping[str, object]) -> bool:
    for field in _ADVISORY_COUNT_FIELDS:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False

    row_count = cast(int, row["row_count"])
    if row_count <= 0 or row.get("listing_count") != row_count:
        return False
    if any(row.get(field) != row_count for field in _ADVISORY_COMPLETENESS_AND_CONSISTENCY_FIELDS):
        return False
    if row.get("evaluation_before_target_count") != 0:
        return False
    if cast(int, row["direction_true_count"]) > row_count:
        return False
    if cast(int, row["inclusion_true_count"]) > row_count:
        return False

    target_date = row.get("prediction__target_date")
    max_evaluation_date = row.get("max_evaluation_date")
    if not _is_date(target_date) or not _is_date(max_evaluation_date):
        return False
    if cast(date, max_evaluation_date) < cast(date, target_date):
        return False

    return all(
        isinstance(row.get(field), Decimal) and cast(Decimal, row[field]).is_finite()
        for field in _ADVISORY_DECIMAL_FIELDS
    )


def _select_non_overlapping_summaries(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            cast(date, row["max_evaluation_date"]),
            cast(date, row["prediction__target_date"]),
        ),
    )
    selected: list[Mapping[str, object]] = []
    prior_finish: date | None = None
    for row in ordered:
        target_date = cast(date, row["prediction__target_date"])
        if prior_finish is None or target_date > prior_finish:
            selected.append(row)
            prior_finish = cast(date, row["max_evaluation_date"])
    return selected


def _advisory_metrics(
    selected: Sequence[Mapping[str, object]],
) -> tuple[Decimal, Decimal, Decimal] | None:
    try:
        direction_means: list[Decimal] = []
        inclusion_means: list[Decimal] = []
        error_means: list[Decimal] = []
        for row in selected:
            row_count = cast(int, row["row_count"])
            direction_mean = Decimal(cast(int, row["direction_true_count"])) / Decimal(row_count)
            inclusion_mean = Decimal(cast(int, row["inclusion_true_count"])) / Decimal(row_count)
            signed_error_sum = row["signed_error_sum"]
            if not isinstance(signed_error_sum, Decimal) or not signed_error_sum.is_finite():
                return None
            error_mean = signed_error_sum / Decimal(row_count)
            if not all(value.is_finite() for value in (direction_mean, inclusion_mean, error_mean)):
                return None
            direction_means.append(direction_mean)
            inclusion_means.append(inclusion_mean)
            error_means.append(error_mean)

        cohort_count = Decimal(len(selected))
        metrics = (
            sum(direction_means, Decimal(0)) / cohort_count,
            sum(inclusion_means, Decimal(0)) / cohort_count,
            sum(error_means, Decimal(0)) / cohort_count,
        )
    except (DecimalException, ZeroDivisionError):
        return None
    return metrics if all(metric.is_finite() for metric in metrics) else None


def _build_advisory_group(
    identity: tuple[str, str, str, str, str, str],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    (
        method_version,
        config_hash,
        price_provider,
        evidence_grade,
        horizon,
        code_revision,
    ) = identity
    minimum_effective, minimum_listings, minimum_span = _ADVISORY_FLOORS.get(horizon, (0, 0, 0))
    revision_valid = _VALID_CODE_REVISION.fullmatch(code_revision) is not None
    reasons: list[str] = []
    if not revision_valid:
        reasons.append("Invalid code revision — metrics withheld")

    well_formed = [row for row in rows if _summary_is_well_formed(row)]
    if len(well_formed) != len(rows):
        reasons.append("Malformed target-date evidence — metrics withheld")
    if horizon not in _ADVISORY_FLOORS:
        reasons.append("Unsupported advisory horizon — metrics withheld")

    effective_cohort_count: int | None = None
    target_span_days: int | None = None
    selected: list[Mapping[str, object]] = []
    if len(well_formed) == len(rows):
        selected = _select_non_overlapping_summaries(well_formed)
        effective_cohort_count = len(selected)
        if selected:
            selected_targets = [cast(date, row["prediction__target_date"]) for row in selected]
            target_span_days = (max(selected_targets) - min(selected_targets)).days

        if horizon in _ADVISORY_FLOORS:
            if effective_cohort_count < minimum_effective:
                reasons.append(
                    "Insufficient effective cohorts: "
                    f"{effective_cohort_count} available; {minimum_effective} required"
                )
            thin_selected = [
                row for row in selected if cast(int, row["listing_count"]) < minimum_listings
            ]
            if thin_selected:
                reasons.append(
                    "Selected cohort below listing floor: "
                    f"{minimum_listings} distinct listings required per selected cohort"
                )
            if target_span_days is None or target_span_days < minimum_span:
                span_label = "none" if target_span_days is None else str(target_span_days)
                reasons.append(
                    f"Insufficient target-date span: {span_label} days; {minimum_span} required"
                )

    metrics: tuple[Decimal, Decimal, Decimal] | None = None
    if not reasons:
        metrics = _advisory_metrics(selected)
        if metrics is None:
            reasons.append("Invalid Decimal metric arithmetic — metrics withheld")

    publishable = metrics is not None and not reasons
    return {
        "method_version": method_version,
        "config_hash": config_hash,
        "price_provider": price_provider,
        "evidence_grade": evidence_grade,
        "horizon": horizon,
        "code_revision": code_revision,
        "revision_valid": revision_valid,
        "revision_label": code_revision if code_revision else "(empty)",
        "publishable": publishable,
        "status_label": "Metrics published" if publishable else "Metrics withheld",
        "withheld_reasons": tuple(reasons),
        "candidate_cohort_count": len(well_formed),
        "effective_cohort_count": effective_cohort_count,
        "minimum_effective_cohorts": minimum_effective,
        "minimum_listings_per_selected_cohort": minimum_listings,
        "target_span_days": target_span_days,
        "minimum_target_span_days": minimum_span,
        "base_sign_match": metrics[0] if metrics is not None else None,
        "inclusion_rate": metrics[1] if metrics is not None else None,
        "mean_signed_base_error": metrics[2] if metrics is not None else None,
    }


def _build_advisory_support_report(
    rows: Sequence[Mapping[str, object]],
    *,
    overflow: bool,
) -> dict[str, object]:
    """Validate, schedule, and estimate bounded grouped advisory evidence."""
    sections = _empty_advisory_sections()
    if overflow:
        return {
            "overflow": True,
            "summary_count": None,
            "has_evidence": bool(rows),
            "sections": sections,
        }

    grouped: dict[tuple[str, str, str, str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        identity = tuple(_identity_text(row.get(field)) for field in _ADVISORY_IDENTITY_FIELDS)
        grouped.setdefault(cast(tuple[str, str, str, str, str, str], identity), []).append(row)

    medium_groups: list[dict[str, object]] = []
    long_groups: list[dict[str, object]] = []
    unsupported_groups: list[dict[str, object]] = []
    for identity in sorted(grouped):
        group = _build_advisory_group(identity, grouped[identity])
        if identity[4] in {
            Prediction.Horizon.SIX_MONTH,
            Prediction.Horizon.TWELVE_MONTH,
        }:
            medium_groups.append(group)
        elif identity[4] in {
            Prediction.Horizon.THREE_YEAR,
            Prediction.Horizon.FIVE_YEAR,
        }:
            long_groups.append(group)
        else:
            unsupported_groups.append(group)
    sections[0]["groups"] = medium_groups
    sections[1]["groups"] = long_groups
    sections[2]["groups"] = unsupported_groups
    if any(group.get("publishable") is True for group in medium_groups):
        sections[0]["disclosure"] = _QUALIFIED_MEDIUM_DISCLOSURE
    return {
        "overflow": False,
        "summary_count": len(rows),
        "has_evidence": bool(rows),
        "sections": sections,
    }

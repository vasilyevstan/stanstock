"""Pure split-conformal interval-offset fitting for retrospective price studies.

The document adapter performs the frozen S1 validation and supplies only
development/validation FHS observations.  This module has no Django, I/O,
clock, provider, or environment dependencies; every decision is a function of
the explicit immutable input object.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import (
    ROUND_CEILING,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from fractions import Fraction
from typing import Any, cast

from stanstock.research.price_product import LedgerTriplet
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PriceProductConfig,
    price_product_config_hash,
)
from stanstock.research.price_product_replay import (
    ReplayMetricObservation,
    build_anchor_plan,
    compare_aligned_models,
    score_projection_metrics,
)

METHOD_VERSION = "interval-offset-cqr-v1"
FIT_SCHEMA = "research-product-interval-calibration-fit@1"
PARAMS_SCHEMA = "research-product-interval-calibration-params@1"
DIAGNOSTICS_SCHEMA = "research-product-interval-calibration-diagnostics@1"

_CONTEXT = Context(
    prec=28,
    rounding=ROUND_HALF_EVEN,
    Emin=-999_999,
    Emax=999_999,
    traps=[InvalidOperation, DivisionByZero, Overflow],
)
_PARTITION_ORDER = {"development": 0, "validation": 1}
_HORIZON_SESSIONS = {"6m": 126, "12m": 252, "3y": 756, "5y": 1260}
_FITTED_HORIZONS = ("6m", "12m")
_ENGINEERING_FLOORS: dict[str, tuple[int, int, int]] = {
    "6m": (7, 30, 1095),
    "12m": (6, 30, 1460),
}
_MAX_CANONICAL_DECIMAL_CHARS = 1_000_000
METHOD_CONSTANTS: dict[str, Any] = {
    "method_version": METHOD_VERSION,
    "model_name": FHS_METHOD_VERSION,
    "alpha": "0.4",
    "target_coverage": "0.6",
    "score": "max(log(1+lower)-log(1+actual), log(1+actual)-log(1+upper))",
    "weighting": "each outcome_end_date cluster has total weight 1/C",
    "cluster_key": "outcome_end_date",
    "level_rule": "Fraction(3,5) * Fraction(C+1,C)",
    "quantile_rule": "smallest score with cumulative cluster-weight >= level; include ties",
    "offset_quantum": "1E-10",
    "offset_rounding": "ROUND_CEILING",
    "decimal_context": "prec=28,ROUND_HALF_EVEN,Emin=-999999,Emax=999999",
    "training_partitions": ["development"],
    "evaluation_partitions": ["validation"],
    "fitted_horizons": ["6m", "12m"],
    "unavailable_horizons": ["3y", "5y"],
    "formal_minimum_clusters": 2,
    "engineering_floors": {key: list(value) for key, value in _ENGINEERING_FLOORS.items()},
    "embargo_sessions": {"6m": 126, "12m": 252},
    "purge_rule": "training label window (anchor,outcome_end] does not overlap test window",
    "non_overlap_rule": "next_anchor_ge_prior_outcome_end",
}


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    partition: str
    horizon: str
    listing_id: str
    anchor_date: date
    outcome_end_date: date
    predicted: LedgerTriplet | None
    actual_return: Decimal | None
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class CalibrationInputs:
    config: PriceProductConfig
    config_identity: Mapping[str, object]
    protocol_identity: Mapping[str, object]
    calendar_identity: Mapping[str, object]
    calendar_sessions: tuple[date, ...]
    observations: tuple[CalibrationObservation, ...]


def canonical_json_bytes(value: object) -> bytes:
    """Canonical JSON bytes for stable, self-hash-excluding identities."""

    def normalize(item: object) -> object:
        if isinstance(item, Decimal):
            if not item.is_finite():
                raise ValueError("canonical values must be finite")
            return _decimal_text(item)
        if isinstance(item, date):
            return item.isoformat()
        if isinstance(item, float):
            raise ValueError("canonical values must not contain floats")
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("canonical object keys must be strings")
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, tuple | list):
            return [normalize(child) for child in item]
        if item is None or isinstance(item, str | int | bool):
            return item
        raise ValueError("canonical value has an unsupported type")

    try:
        encoded = json.dumps(
            normalize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError("value cannot be canonically serialized") from exc
    return encoded.encode("ascii")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def embargo_eligible(
    *, train_outcome_end_index: int, test_anchor_index: int, embargo_sessions: int
) -> bool:
    return test_anchor_index - train_outcome_end_index >= embargo_sessions


def fit_interval_calibration(inputs: CalibrationInputs) -> dict[str, Any]:
    """Fit parameters on development rows and score validation rows only."""
    with localcontext(_CONTEXT):
        try:
            index_by_date = _validate_inputs(inputs)
            session_sha = canonical_sha256([item.isoformat() for item in inputs.calendar_sessions])
            normalized = _sorted_observations(inputs.observations)
            development = tuple(item for item in normalized if item.partition == "development")
            validation = tuple(item for item in normalized if item.partition == "validation")
            params_rows = [_row_payload(item) for item in development]
            fit_digest = canonical_sha256(
                {
                    "method_version": METHOD_VERSION,
                    "method": _json_copy(METHOD_CONSTANTS),
                    "config_identity": inputs.config_identity,
                    "protocol_identity": inputs.protocol_identity,
                    "calendar_identity": inputs.calendar_identity,
                    "calendar_sessions_sha256": session_sha,
                    "observations": params_rows,
                }
            )
            params_horizons: dict[str, Any] = {}
            for horizon in _FITTED_HORIZONS:
                rows = tuple(item for item in development if item.horizon == horizon)
                params_horizons[horizon] = _fit_horizon(
                    rows,
                    horizon=horizon,
                )
            for horizon in ("3y", "5y"):
                params_horizons[horizon] = _unavailable_horizon(
                    "horizon_unavailable_no_calibration_claim"
                )
            six_month = params_horizons["6m"]
            params: dict[str, Any] = {
                "schema": PARAMS_SCHEMA,
                "method": _json_copy(METHOD_CONSTANTS),
                "config_identity": _json_copy(inputs.config_identity),
                "protocol_identity": _json_copy(inputs.protocol_identity),
                "calendar_identity": _json_copy(inputs.calendar_identity),
                "calendar_sessions_sha256": session_sha,
                "fit_input_sha256": fit_digest,
                "training_cutoff": (
                    six_month["training_cutoff"] if six_month["state"] == "usable" else None
                ),
                "horizons": params_horizons,
            }
            params["params_sha256"] = canonical_sha256(params)
            diagnostics = _build_diagnostics(
                inputs=inputs,
                params=params,
                development=development,
                validation=validation,
                index_by_date=index_by_date,
            )
            return {"params": params, "diagnostics": diagnostics}
        except ArithmeticError as exc:
            raise ValueError("arithmetic_unrepresentable") from exc


def _validate_inputs(inputs: CalibrationInputs) -> dict[date, int]:
    if price_product_config_hash(inputs.config) != PRODUCT_EFFECTIVE_CONFIG_HASH:
        raise ValueError("config does not match the frozen product configuration")
    if not all(
        isinstance(identity, Mapping)
        for identity in (
            inputs.config_identity,
            inputs.protocol_identity,
            inputs.calendar_identity,
        )
    ):
        raise ValueError("calibration identities must be mappings")
    if inputs.config_identity.get("effective_config_hash") != PRODUCT_EFFECTIVE_CONFIG_HASH:
        raise ValueError("config identity does not match the frozen product configuration")
    sessions = inputs.calendar_sessions
    if not sessions:
        raise ValueError("calendar sessions must not be empty")
    if any(type(item) is not date for item in sessions):
        raise ValueError("calendar sessions must be dates")
    if any(left >= right for left, right in zip(sessions, sessions[1:], strict=False)):
        raise ValueError("calendar sessions must be strictly increasing")
    expected_calendar = {
        "calendar": inputs.config.calendar,
        "first_session": sessions[0].isoformat(),
        "last_session": sessions[-1].isoformat(),
        "session_count": len(sessions),
    }
    if dict(inputs.calendar_identity) != expected_calendar:
        raise ValueError("calendar identity does not match calendar sessions")
    index_by_date = {session: index for index, session in enumerate(sessions)}
    plan = build_anchor_plan(sessions, config=inputs.config)
    if plan.unavailable_reason is not None:
        raise ValueError("frozen anchor plan is unavailable")
    plan_by_key = {
        (point.partition, point.horizon, point.anchor_date, point.outcome_end_date): point
        for partition in plan.partitions
        for point in partition.anchors
    }
    seen: set[tuple[str, str, str, date]] = set()
    for row in inputs.observations:
        if row.partition == "final_holdout":
            raise ValueError("final_holdout observations must not reach the calibration core")
        if row.partition not in ("development", "validation"):
            raise ValueError("observation partition is invalid")
        if (
            not isinstance(row.horizon, str)
            or not isinstance(row.listing_id, str)
            or not row.listing_id
        ):
            raise ValueError("observation identity is invalid")
        key = (row.partition, row.horizon, row.anchor_date, row.outcome_end_date)
        if key not in plan_by_key:
            raise ValueError("observation anchor is not in the frozen plan")
        duplicate_key = (row.partition, row.horizon, row.listing_id, row.anchor_date)
        if duplicate_key in seen:
            raise ValueError("calibration observations contain a duplicate identity")
        seen.add(duplicate_key)
        if row.predicted is not None:
            _require_finite_decimal_triplet(row.predicted)
        if row.actual_return is not None:
            if not isinstance(row.actual_return, Decimal):
                raise ValueError("actual return must be a Decimal")
            if not row.actual_return.is_finite():
                raise ValueError("actual return must be finite")
        if row.actual_return is not None and _outside_decimal_context(row.actual_return):
            raise ArithmeticError("actual return is outside the decimal context")
        if row.predicted is not None and any(
            _outside_decimal_context(value)
            for value in (row.predicted.lower, row.predicted.median, row.predicted.upper)
        ):
            raise ArithmeticError("predicted return is outside the decimal context")
        if row.unavailable_reason is None:
            if row.predicted is None or row.actual_return is None:
                raise ValueError("available observations require a triplet and actual return")
        elif not isinstance(row.unavailable_reason, str) or not row.unavailable_reason:
            raise ValueError("unavailable reason must be non-empty text")
    return index_by_date


def _fit_horizon(
    rows: Sequence[CalibrationObservation],
    *,
    horizon: str,
) -> dict[str, Any]:
    exclusions = Counter(
        row.unavailable_reason for row in rows if row.unavailable_reason is not None
    )
    scored = tuple(
        row
        for row in rows
        if row.unavailable_reason is None
        and row.predicted is not None
        and row.actual_return is not None
    )
    for row in scored:
        assert row.predicted is not None and row.actual_return is not None
        if (
            row.actual_return < -1
            or min(row.predicted.lower, row.predicted.median, row.predicted.upper) < -1
        ):
            raise ValueError("return_below_minus_one")
    clusters: dict[date, list[CalibrationObservation]] = defaultdict(list)
    for row in scored:
        clusters[row.outcome_end_date].append(row)
    non_overlap = _non_overlapping_count(scored)
    listing_floor = min((len(value) for value in clusters.values()), default=0)
    anchors = [row.anchor_date for row in scored]
    span_days = (max(anchors) - min(anchors)).days if anchors else 0
    cutoff = max((row.outcome_end_date for row in scored), default=None)
    count = len(clusters)
    level = Fraction(3, 5) * Fraction(count + 1, count) if count else None

    reason: str | None = None
    failing: list[str] = []
    offset: Decimal | None = None
    if count == 0:
        reason = "no_scored_observations"
    elif any(
        row.actual_return == Decimal("-1")
        or min(row.predicted.lower, row.predicted.median, row.predicted.upper) == -1
        for row in scored
        if row.predicted is not None and row.actual_return is not None
    ):
        reason = "log_scale_boundary_observation"
    elif count < 2:
        reason = "conformal_rank_infeasible"
    else:
        minimum_clusters, minimum_listings, minimum_span = _ENGINEERING_FLOORS[horizon]
        if non_overlap < minimum_clusters:
            failing.append("non_overlapping_clusters_below_floor")
        if listing_floor < minimum_listings:
            failing.append("cluster_listings_below_floor")
        if span_days < minimum_span:
            failing.append("span_below_floor")
        if failing:
            reason = "insufficient_effective_support"
        else:
            offset = _weighted_quantile(scored, level)
    return {
        "state": "usable" if reason is None else "withheld_native_fallback",
        "reason": reason,
        "failing_components": failing,
        "offset": None if offset is None else _decimal_text(offset),
        "clusters": count,
        "scored_observations": len(scored),
        "non_overlapping_clusters": non_overlap,
        "min_listings_per_cluster": listing_floor,
        "span_days": span_days,
        "first_anchor": min(anchors).isoformat() if anchors else None,
        "last_anchor": max(anchors).isoformat() if anchors else None,
        "training_cutoff": cutoff.isoformat() if cutoff else None,
        "level": _fraction_text(level),
        "exclusions": {key: exclusions[key] for key in sorted(exclusions)},
    }


def _weighted_quantile(rows: Sequence[CalibrationObservation], level: Fraction | None) -> Decimal:
    if level is None:
        raise ValueError("conformal level is unavailable")
    by_cluster: dict[date, list[Decimal]] = defaultdict(list)
    for row in rows:
        assert row.predicted is not None and row.actual_return is not None
        by_cluster[row.outcome_end_date].append(_conformal_score(row))
    cluster_count = len(by_cluster)
    weighted: list[tuple[Decimal, Fraction]] = []
    for scores in by_cluster.values():
        weight = Fraction(1, cluster_count * len(scores))
        weighted.extend((score, weight) for score in scores)
    cumulative = Fraction(0)
    quantile: Decimal | None = None
    for score, weight in sorted(weighted, key=lambda pair: pair[0]):
        cumulative += weight
        if cumulative >= level:
            quantile = score
            break
    if quantile is None:
        raise ArithmeticError("conformal rank is not representable")
    return quantile.quantize(Decimal("1E-10"), rounding=ROUND_CEILING)


def _conformal_score(row: CalibrationObservation) -> Decimal:
    if row.predicted is None or row.actual_return is None:
        raise ValueError("conformal score requires a complete observation")
    if row.actual_return < -1:
        raise ValueError("return_below_minus_one")
    if row.actual_return == -1:
        return Decimal(0)
    if row.predicted.lower < -1 or row.predicted.upper < -1:
        raise ValueError("training prediction is below the log-scale boundary")
    lower = (Decimal(1) + row.predicted.lower).ln()
    upper = (Decimal(1) + row.predicted.upper).ln()
    actual = (Decimal(1) + row.actual_return).ln()
    return max(lower - actual, actual - upper)


def adjust_interval(
    lower: Decimal, median: Decimal, upper: Decimal, offset: Decimal
) -> tuple[Decimal, Decimal, Decimal] | None:
    """Expand/narrow log-scale bounds while preserving the median."""
    if not all(value.is_finite() for value in (lower, median, upper, offset)):
        return None
    if not lower <= median <= upper or min(lower, median, upper) <= -1:
        return None
    if offset == 0:
        return lower, median, upper
    try:
        with localcontext(_CONTEXT):
            lower_value = ((Decimal(1) + lower).ln() - offset).exp() - Decimal(1)
            upper_value = ((Decimal(1) + upper).ln() + offset).exp() - Decimal(1)
            return min(lower_value, median), median, max(upper_value, median)
    except ArithmeticError:
        return None


def summarize_cluster_coverage(
    cluster_inclusions: Mapping[str, Sequence[bool]],
) -> dict[str, Any]:
    """Pure reusable cluster-first central-interval coverage arithmetic."""
    with localcontext(_CONTEXT):
        means = [
            sum(Decimal(int(value)) for value in values) / Decimal(len(values))
            for _cluster, values in sorted(cluster_inclusions.items())
            if values
        ]
        if not means:
            return {
                "cluster_count": 0,
                "mean": None,
                "min": None,
                "max": None,
                "abs_deviation_from_0.6": None,
                "cluster_means": [],
            }
        mean = sum(means, Decimal(0)) / Decimal(len(means))
        return {
            "cluster_count": len(means),
            "mean": _decimal_text(mean),
            "min": _decimal_text(min(means)),
            "max": _decimal_text(max(means)),
            "abs_deviation_from_0.6": _decimal_text(abs(mean - Decimal("0.6"))),
            "cluster_means": [_decimal_text(value) for value in means],
        }


def _build_diagnostics(
    *,
    inputs: CalibrationInputs,
    params: Mapping[str, Any],
    development: Sequence[CalibrationObservation],
    validation: Sequence[CalibrationObservation],
    index_by_date: Mapping[date, int],
) -> dict[str, Any]:
    observations = [_row_payload(item) for item in validation]
    evaluation_digest = canonical_sha256(
        {"params_sha256": params["params_sha256"], "observations": observations}
    )
    horizon_results: dict[str, Any] = {}
    for horizon in _FITTED_HORIZONS:
        params_horizon = params["horizons"][horizon]
        horizon_dev = tuple(row for row in development if row.horizon == horizon)
        horizon_val = tuple(row for row in validation if row.horizon == horizon)
        folds, fold_offsets = _walk_forward_folds(
            horizon_dev,
            horizon_val,
            horizon=horizon,
            params_horizon=params_horizon,
            index_by_date=index_by_date,
        )
        native_observations: list[ReplayMetricObservation] = []
        calibrated_observations: list[ReplayMetricObservation] = []
        native_coverage: dict[str, list[bool]] = defaultdict(list)
        candidate_coverage: dict[str, list[bool]] = defaultdict(list)
        unrepresentable = 0
        clamped_lower = 0
        clamped_upper = 0
        native_count = 0
        for row in horizon_val:
            assert row.actual_return is not None or row.unavailable_reason is not None
            native_triplet = row.predicted if row.unavailable_reason is None else None
            native_metrics = score_projection_metrics(
                predicted_returns=native_triplet,
                actual_return=row.actual_return,
                config=inputs.config,
            )
            native_observations.append(_metric_observation(row, "native", native_metrics))
            if native_metrics.interval_included is not None:
                native_count += 1
                native_coverage[row.outcome_end_date.isoformat()].append(
                    native_metrics.interval_included
                )
            offset = fold_offsets.get(row.anchor_date)
            adjusted: LedgerTriplet | None = None
            if (
                offset is not None
                and native_triplet is not None
                and row.actual_return is not None
                and row.actual_return > -1
            ):
                adjusted_values = adjust_interval(
                    native_triplet.lower,
                    native_triplet.median,
                    native_triplet.upper,
                    offset,
                )
                if adjusted_values is None:
                    unrepresentable += 1
                else:
                    adjusted = LedgerTriplet(*adjusted_values)
                    if adjusted_values[0] == native_triplet.median and (
                        ((Decimal(1) + native_triplet.lower).ln() - offset).exp() - 1
                        > native_triplet.median
                    ):
                        clamped_lower += 1
                    if adjusted_values[2] == native_triplet.median and (
                        ((Decimal(1) + native_triplet.upper).ln() + offset).exp() - 1
                        < native_triplet.median
                    ):
                        clamped_upper += 1
            elif (
                params_horizon["state"] == "usable"
                and row.unavailable_reason is None
                and row.actual_return is not None
            ):
                unrepresentable += 1
            candidate_metrics = score_projection_metrics(
                predicted_returns=adjusted,
                actual_return=row.actual_return,
                config=inputs.config,
            )
            if (
                adjusted is None
                and params_horizon["state"] == "usable"
                and row.actual_return is not None
                and row.unavailable_reason is None
            ):
                candidate_metrics = replace(
                    candidate_metrics,
                    unavailable_reason="calibrated_triplet_unrepresentable",
                )
            calibrated_observations.append(
                _metric_observation(row, "calibrated", candidate_metrics)
            )
            if candidate_metrics.interval_included is not None:
                candidate_coverage[row.outcome_end_date.isoformat()].append(
                    candidate_metrics.interval_included
                )
        paired: dict[str, Any] | None
        inclusion: dict[str, Any] | None
        if params_horizon["state"] != "usable":
            paired = None
            inclusion = None
            arm_state = "withheld_native_fallback"
            arm_reason = params_horizon["reason"]
            calibrated_count = 0
        else:
            combined = native_observations + calibrated_observations
            score_comparison = compare_aligned_models(
                combined,
                candidate_model="calibrated",
                baseline_model="native",
                partition="validation",
                horizon=horizon,
                metric_name="interval_score",
            )
            inclusion_comparison = compare_aligned_models(
                combined,
                candidate_model="calibrated",
                baseline_model="native",
                partition="validation",
                horizon=horizon,
                metric_name="interval_inclusion",
            )
            paired = _comparison_payload(score_comparison)
            inclusion = _comparison_payload(inclusion_comparison)
            arm_state = "calibrated"
            arm_reason = None
            calibrated_count = sum(
                item.metrics.interval_score is not None for item in calibrated_observations
            )
        horizon_results[horizon] = {
            "arm_state": arm_state,
            "reason": (
                "calibrated_arm_withheld" if params_horizon["state"] != "usable" else arm_reason
            ),
            "calibration_reason": arm_reason,
            "folds": folds,
            "fold_reason": None if folds else "no_validation_clusters",
            "paired": paired,
            "paired_inclusion": inclusion,
            "coverage": {
                "native": summarize_cluster_coverage(native_coverage),
                "calibrated": summarize_cluster_coverage(candidate_coverage),
            },
            "native_observations": native_count,
            "calibrated_observations": calibrated_count,
            "candidate_unrepresentable": unrepresentable,
            "clamped_lower": clamped_lower,
            "clamped_upper": clamped_upper,
            "evidence_status": "descriptive_validation_below_engineering_floor",
        }
    diagnostics: dict[str, Any] = {
        "schema": DIAGNOSTICS_SCHEMA,
        "params_sha256": params["params_sha256"],
        "evaluation_input_sha256": evaluation_digest,
        "partition": "validation",
        "horizons": horizon_results,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    return diagnostics


def _walk_forward_folds(
    development: Sequence[CalibrationObservation],
    validation: Sequence[CalibrationObservation],
    *,
    horizon: str,
    params_horizon: Mapping[str, Any],
    index_by_date: Mapping[date, int],
) -> tuple[list[dict[str, Any]], dict[date, Decimal | None]]:
    test_anchors = sorted({row.anchor_date for row in validation if row.horizon == horizon})
    folds: list[dict[str, Any]] = []
    offsets: dict[date, Decimal | None] = {}
    embargo = _HORIZON_SESSIONS[horizon]
    params_rows = tuple(
        row
        for row in development
        if row.horizon == horizon
        and row.unavailable_reason is None
        and row.predicted is not None
        and row.actual_return is not None
    )
    for test_anchor in test_anchors:
        test_index = index_by_date[test_anchor]
        eligible: list[CalibrationObservation] = []
        purged = 0
        embargoed = 0
        for row in params_rows:
            end_index = index_by_date[row.outcome_end_date]
            if end_index > test_index:
                purged += 1
            elif embargo_eligible(
                train_outcome_end_index=end_index,
                test_anchor_index=test_index,
                embargo_sessions=embargo,
            ):
                eligible.append(row)
            else:
                embargoed += 1
        equal = _observation_keys(eligible) == _observation_keys(params_rows)
        if equal:
            fold_result = params_horizon
            fold_offset = (
                Decimal(params_horizon["offset"]) if params_horizon["state"] == "usable" else None
            )
        else:
            fold_result = _fit_horizon(
                eligible,
                horizon=horizon,
            )
            fold_offset = (
                Decimal(fold_result["offset"]) if fold_result["state"] == "usable" else None
            )
        offsets[test_anchor] = fold_offset
        folds.append(
            {
                "test_cluster": test_anchor.isoformat(),
                "eligible_observations": len(eligible),
                "purged_observations": purged,
                "embargoed_observations": embargoed,
                "fold_training_equals_params": equal,
                "state": fold_result["state"],
                "reason": fold_result["reason"],
                "offset": fold_result["offset"],
            }
        )
    return folds, offsets


def _metric_observation(
    row: CalibrationObservation, model: str, metrics: Any
) -> ReplayMetricObservation:
    return ReplayMetricObservation(
        listing_id=row.listing_id,
        anchor_date=row.anchor_date,
        target_date=row.outcome_end_date,
        horizon=row.horizon,
        partition="validation",
        model_name=model,
        metrics=metrics,
    )


def _comparison_payload(comparison: Any) -> dict[str, Any]:
    return {
        "candidate_model": comparison.candidate_model,
        "baseline_model": comparison.baseline_model,
        "partition": comparison.partition,
        "horizon": comparison.horizon,
        "metric_name": comparison.metric_name,
        "paired_observation_count": comparison.paired_observation_count,
        "paired_target_cohort_count": comparison.paired_target_cohort_count,
        "candidate_average": (
            _decimal_text(comparison.candidate_average)
            if comparison.candidate_average is not None
            else None
        ),
        "baseline_average": (
            _decimal_text(comparison.baseline_average)
            if comparison.baseline_average is not None
            else None
        ),
        "mean_difference_candidate_minus_baseline": (
            _decimal_text(comparison.mean_difference_candidate_minus_baseline)
            if comparison.mean_difference_candidate_minus_baseline is not None
            else None
        ),
        "exclusions": dict(sorted(comparison.exclusions.items())),
        "unavailable_reason": comparison.unavailable_reason,
    }


def _row_payload(row: CalibrationObservation) -> dict[str, Any]:
    triplet = row.predicted
    return {
        "horizon": row.horizon,
        "listing_id": row.listing_id,
        "anchor_date": row.anchor_date.isoformat(),
        "outcome_end_date": row.outcome_end_date.isoformat(),
        "lower": _decimal_text(triplet.lower) if triplet else None,
        "median": _decimal_text(triplet.median) if triplet else None,
        "upper": _decimal_text(triplet.upper) if triplet else None,
        "actual_return": (
            _decimal_text(row.actual_return) if row.actual_return is not None else None
        ),
        "unavailable_reason": row.unavailable_reason,
    }


def _sorted_observations(
    observations: Sequence[CalibrationObservation],
) -> tuple[CalibrationObservation, ...]:
    return tuple(
        sorted(
            observations,
            key=lambda row: (
                _PARTITION_ORDER[row.partition],
                _HORIZON_SESSIONS.get(row.horizon, 10**9),
                row.anchor_date,
                row.outcome_end_date,
                row.listing_id,
            ),
        )
    )


def _observation_keys(rows: Sequence[CalibrationObservation]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                row.listing_id,
                row.anchor_date,
                row.outcome_end_date,
                row.horizon,
                row.partition,
            )
            for row in rows
        )
    )


def _non_overlapping_count(rows: Sequence[CalibrationObservation]) -> int:
    intervals = sorted(
        {(row.anchor_date, row.outcome_end_date) for row in rows},
        key=lambda pair: (pair[1], pair[0]),
    )
    count = 0
    last_end: date | None = None
    for start, end in intervals:
        if last_end is None or start >= last_end:
            count += 1
            last_end = end
    return count


def _unavailable_horizon(reason: str) -> dict[str, Any]:
    return {
        "state": "withheld_native_fallback",
        "reason": reason,
        "failing_components": [],
        "offset": None,
        "clusters": 0,
        "scored_observations": 0,
        "non_overlapping_clusters": 0,
        "min_listings_per_cluster": 0,
        "span_days": 0,
        "first_anchor": None,
        "last_anchor": None,
        "training_cutoff": None,
        "level": None,
        "exclusions": {},
    }


def _require_finite_decimal_triplet(value: LedgerTriplet) -> None:
    if not all(
        isinstance(item, Decimal) and item.is_finite()
        for item in (value.lower, value.median, value.upper)
    ):
        raise ValueError("predicted returns must be finite Decimals")
    if not value.lower <= value.median <= value.upper:
        raise ValueError("predicted returns must be ordered")


def _fraction_text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    return f"{value.numerator}/{value.denominator}"


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("decimal is outside the supported canonical range")
    if value.is_zero():
        return "0"
    if _outside_decimal_context(value):
        raise ValueError("decimal is outside the supported canonical range")

    sign, raw_digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError("decimal is outside the supported canonical range")
    digits = "".join(str(digit) for digit in raw_digits).rstrip("0")
    exponent += len(raw_digits) - len(digits)
    if exponent >= 0:
        body_length = len(digits) + exponent
    else:
        decimal_position = len(digits) + exponent
        body_length = (
            len(digits) + 1 if decimal_position > 0 else 2 - decimal_position + len(digits)
        )
    output_length = body_length + int(bool(sign))
    if output_length > _MAX_CANONICAL_DECIMAL_CHARS:
        raise ValueError("decimal is outside the supported canonical range")

    if exponent >= 0:
        body = digits + ("0" * exponent)
    else:
        decimal_position = len(digits) + exponent
        if decimal_position > 0:
            body = f"{digits[:decimal_position]}.{digits[decimal_position:]}"
        else:
            body = f"0.{('0' * -decimal_position)}{digits}"
    return f"-{body}" if sign else body


def _outside_decimal_context(value: Decimal) -> bool:
    if value.is_zero():
        return False
    adjusted = value.adjusted()
    return adjusted > _CONTEXT.Emax or adjusted < _CONTEXT.Emin


def _json_copy(value: Mapping[str, object]) -> dict[str, Any]:
    # Normalize through the hash serializer so the retained identity is JSON-native.
    return cast(dict[str, Any], json.loads(canonical_json_bytes(value)))

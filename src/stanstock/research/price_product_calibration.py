"""Read-only quantile-calibration ledger over a serialized retrospective study.

The frozen FHS projections publish a central 60% interval (p20/p50/p80). The
retrospective replay reports only interval inclusion; this module adds the
per-quantile hit rates that separate a scale error (both tails too heavy or
too light) from a location/skew error (asymmetric tails).

The ledger is a descriptive diagnostic of the frozen forecasts. It is pure and
deterministic: no I/O, clock, randomness, database, provider, or fitted
adjustment. It consumes the JSON document emitted by
``serialize_price_product_study`` and never the live study objects.

Ordering contract: groups are ordered by partition (development, validation,
final_holdout), then horizon by duration (unrecognized labels last, by name),
then model name. Everything except ``input_sha256`` is therefore invariant to
the order of listings, anchors, and models in the input; ``input_sha256`` hashes
the exact input structure and intentionally changes with array order.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from typing import Any

from stanstock.research.price_product_config import PRODUCT_EFFECTIVE_CONFIG_HASH
from stanstock.research.price_product_study import REPLAY_STUDY_SCHEMA

CALIBRATION_LEDGER_SCHEMA = "research-product-quantile-calibration-ledger@1"
LEDGER_EVIDENCE_LABEL = "research_grade_retrospective_descriptive"
SYNTHETIC_LEDGER_EVIDENCE_LABEL = "synthetic_demo_descriptive_not_empirical"

_PARTITIONS = ("development", "validation", "final_holdout")
_INDICATORS = ("below_lower", "below_median", "above_upper", "included")
# Nominal masses of the frozen p20/p50/p80 quantiles, tied to PRODUCT_EFFECTIVE_CONFIG_HASH.
_NOMINAL = {
    "below_lower": "0.2",
    "below_median": "0.5",
    "above_upper": "0.2",
    "included": "0.6",
}
_SOURCE_MODES = ("provider", "synthetic_demo")
_QUANTUM = Decimal("0.000001")
_MAX_TEXT_LENGTH = 200
_MAX_DISCLOSURE_LENGTH = 1000
_MAX_DISCLOSURES = 50
_SCOPE_KEYS = frozenset(
    {
        "all_selected",
        "requested_listing_ids",
        "studied_listing_ids",
        "source_run_listing_count",
        "studied_listing_count",
    }
)
_HORIZON_PATTERN = re.compile(r"^(\d+)([my])$")
_CONTEXT = Context(
    prec=28,
    rounding=ROUND_HALF_EVEN,
    Emin=-999_999,
    Emax=999_999,
    traps=[InvalidOperation, DivisionByZero, Overflow],
)
_DISCLOSURES = (
    "Descriptive diagnostic of the frozen forecasts only; not a calibration claim, a skill "
    "claim, or adoption evidence.",
    "Observations overlap in time and are cross-sectionally dependent, and rates are "
    "cohort-first means; they are not independent trials. See scored_cohort_count, "
    "observation_count and scored_non_overlapping_target_count for the support behind each rate.",
    "target_cohort_count, distinct_listing_count and non_overlapping_target_count mirror the "
    "replay and include unavailable observations; scored_cohort_count and observation_count "
    "count only the scored cohorts and observations averaged; "
    "scored_non_overlapping_target_count is over scored observations only.",
    "Current-vintage research-grade reconstruction from the replay; not observed historical "
    "availability.",
    "Ties: a realized return equal to a quantile is not a below/above hit; the central interval "
    "is inclusive. Ties are judged at the replay's rounded precision (return_decimal_places 4).",
    "Unavailable observations are counted by reason and excluded, never treated as zero.",
    "Reading guide: signed_median_error = actual - median. A nonzero mean_signed_median_error "
    "is expected for right-skewed returns even when the forecast is calibrated, so it is not a "
    "calibration test. The below_median rate is the location check. No sampling-uncertainty "
    "threshold is applied here, and few or correlated cohorts make deviations from nominal "
    "uninterpretable.",
    "The final_holdout partition is already inspected: it is not fresh confirmation and must "
    "never be used to tune against.",
    "input_sha256 hashes a canonical re-serialisation of the input; it identifies the input "
    "and does not authenticate it, and the source run is not re-verified.",
    "Rates and deviations are rounded separately to 6 decimal places, so shares may sum to "
    "0.999999 or 1.000001.",
)


@dataclass(frozen=True, slots=True)
class _Scores:
    below_lower: bool
    below_median: bool
    above_upper: bool
    included: bool
    signed_median_error: Decimal


@dataclass(frozen=True, slots=True)
class _Observation:
    listing_id: str
    anchor_date: date
    target_date: date
    scores: _Scores | None
    unavailable_reason: str | None


def summarize_quantile_calibration(study_document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the per-quantile calibration ledger for one serialized study document.

    Fails closed with ``ValueError`` on any malformed, inconsistent, or
    config-mismatched input. Error messages name the offending path, never its
    value.
    """

    document: object = study_document
    if not isinstance(document, Mapping):
        raise ValueError("study document must be a JSON object")
    with localcontext(_CONTEXT):
        try:
            return _summarize(document)
        except ArithmeticError as exc:
            raise ValueError("study document values are outside the supported range") from exc


def render_quantile_calibration(ledger: Mapping[str, Any], *, output_format: str) -> str:
    """Render a ledger as canonical JSON or a compact text table."""

    if output_format not in ("json", "text"):
        raise ValueError("output_format must be json or text")
    candidate: object = ledger
    if not isinstance(candidate, Mapping) or candidate.get("schema") != CALIBRATION_LEDGER_SCHEMA:
        raise ValueError("ledger must be a quantile calibration ledger")
    try:
        if output_format == "json":
            return json.dumps(ledger, sort_keys=True, indent=2)
        return _render_text(ledger)
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("ledger is malformed") from exc


def _summarize(document: Mapping[str, Any]) -> dict[str, Any]:
    if _require(document, "schema", "document") != REPLAY_STUDY_SCHEMA:
        raise ValueError(f"schema must be {REPLAY_STUDY_SCHEMA}")
    identity = _require_mapping(
        _require(document, "config_identity", "document"), "config_identity"
    )
    config_hash = _require(identity, "effective_config_hash", "config_identity")
    if config_hash != PRODUCT_EFFECTIVE_CONFIG_HASH:
        raise ValueError(
            "config_identity.effective_config_hash does not match the frozen product config"
        )
    source_run = _require_mapping(_require(document, "source_run", "document"), "source_run")
    source_run_id = _require_text(_require(source_run, "id", "source_run"), "source_run.id")
    protocol = _require_mapping(
        _require(document, "protocol_identity", "document"), "protocol_identity"
    )
    evidence_label = _require_text(
        _require(protocol, "evidence_label", "protocol_identity"),
        "protocol_identity.evidence_label",
    )

    grid = _parse_grid(
        _require_list(
            _require(document, "projection_aggregates", "document"), "projection_aggregates"
        )
    )
    snapshot_grade = _require_text(
        _require(source_run, "snapshot_grade", "source_run"), "source_run.snapshot_grade"
    )
    issued_on_time = _require(source_run, "issued_on_time", "source_run")
    if not isinstance(issued_on_time, bool):
        raise ValueError("source_run.issued_on_time must be a boolean")
    disclosures = _require_list(_require(document, "disclosures", "document"), "disclosures")
    if (
        not disclosures
        or len(disclosures) > _MAX_DISCLOSURES
        or not all(
            isinstance(item, str)
            and item
            and len(item) <= _MAX_DISCLOSURE_LENGTH
            and item.isprintable()
            for item in disclosures
        )
    ):
        raise ValueError("disclosures must be a bounded non-empty array of short printable strings")

    listings = _require_list(_require(document, "listings", "document"), "listings")
    entries, modes = _parse_listings(listings)
    scope = _require_mapping(_require(document, "scope", "document"), "scope")
    _validate_scope(scope)
    studied = _require(scope, "studied_listing_count", "scope")
    if type(studied) is not int or studied != len(listings):
        raise ValueError("scope.studied_listing_count does not match listings")

    grouped: dict[tuple[str, str, str], list[_Observation]] = {}
    for partition, horizon, model, observation in entries:
        key = (partition, horizon, model)
        if key not in grid:
            raise ValueError(
                "listings contain an observation outside the projection_aggregates grid"
            )
        grouped.setdefault(key, []).append(observation)
    groups = []
    for key in sorted(grid, key=lambda k: (_PARTITIONS.index(k[0]), _horizon_sort_key(k[1]), k[2])):
        group = _summarize_group(*key, grouped.get(key, []))
        _cross_check(group, grid[key])
        groups.append(group)
    return {
        "schema": CALIBRATION_LEDGER_SCHEMA,
        "source_schema": REPLAY_STUDY_SCHEMA,
        "source_run_id": source_run_id,
        "source_evidence_label": evidence_label,
        "source_execution_modes": modes,
        "source_snapshot_grade": snapshot_grade,
        "source_issued_on_time": issued_on_time,
        "source_disclosures": list(disclosures),
        "scope": json.loads(json.dumps(scope, sort_keys=True)),
        "input_sha256": _canonical_sha256(document),
        "evidence_label": (
            SYNTHETIC_LEDGER_EVIDENCE_LABEL if "synthetic_demo" in modes else LEDGER_EVIDENCE_LABEL
        ),
        "disclosures": list(_DISCLOSURES),
        "groups": groups,
    }


def _parse_grid(aggregates: list[Any]) -> dict[tuple[str, str, str], tuple[int, Mapping[str, Any]]]:
    grid: dict[tuple[str, str, str], tuple[int, Mapping[str, Any]]] = {}
    for index, raw in enumerate(aggregates):
        where = f"projection_aggregates[{index}]"
        item = _require_mapping(raw, where)
        partition = _require_text(_require(item, "partition", where), f"{where}.partition")
        if partition not in _PARTITIONS:
            raise ValueError(f"{where}.partition is not a known partition")
        key = (
            partition,
            _require_text(_require(item, "horizon", where), f"{where}.horizon"),
            _require_text(_require(item, "model_name", where), f"{where}.model_name"),
        )
        if key in grid:
            raise ValueError(f"{where} duplicates an earlier aggregate")
        grid[key] = (index, item)
    partitions = {key[0] for key in grid}
    horizons = {key[1] for key in grid}
    models = {key[2] for key in grid}
    if partitions != set(_PARTITIONS) or len(grid) != len(partitions) * len(horizons) * len(models):
        raise ValueError("projection_aggregates must be a full partition x horizon x model grid")
    return grid


def _cross_check(group: dict[str, Any], entry: tuple[int, Mapping[str, Any]]) -> None:
    index, aggregate = entry
    where = f"projection_aggregates[{index}]"
    for name in (
        "target_cohort_count",
        "distinct_listing_count",
        "non_overlapping_target_count",
        "unavailable_count",
        "unavailable_reasons",
        "insufficient_reason",
    ):
        if _require(aggregate, name, where) != group[name]:
            raise ValueError(f"{where}.{name} does not match the listings")
    averages = _require_mapping(_require(aggregate, "averages", where), f"{where}.averages")
    inclusion = _require(averages, "interval_inclusion", f"{where}.averages")
    rate = group["rates"]["included"]
    if inclusion is None or rate is None:
        matches = inclusion is None and rate is None
    else:
        parsed = _parse_decimal(inclusion, f"{where}.averages.interval_inclusion")
        matches = _quantize(parsed) == Decimal(rate)
    if not matches:
        raise ValueError(f"{where}.averages.interval_inclusion does not match the listings")


def _parse_listings(
    listings: list[Any],
) -> tuple[list[tuple[str, str, str, _Observation]], list[str]]:
    if not listings:
        raise ValueError("listings must not be empty")
    entries: list[tuple[str, str, str, _Observation]] = []
    seen: set[tuple[str, str, str, date, str]] = set()
    actuals: dict[tuple[str, str, str, date], Decimal] = {}
    modes: set[str] = set()
    for listing_index, raw_listing in enumerate(listings):
        where = f"listings[{listing_index}]"
        listing = _require_mapping(raw_listing, where)
        listing_id = _require_text(_require(listing, "listing_id", where), f"{where}.listing_id")
        execution = _require_mapping(
            _require(listing, "source_execution", where), f"{where}.source_execution"
        )
        mode = _require(execution, "mode", f"{where}.source_execution")
        if mode not in _SOURCE_MODES:
            raise ValueError(f"{where}.source_execution.mode is not a known mode")
        modes.add(mode)
        anchors = _require_list(_require(listing, "anchors", where), f"{where}.anchors")
        for anchor_index, raw_anchor in enumerate(anchors):
            anchor_where = f"{where}.anchors[{anchor_index}]"
            anchor = _require_mapping(raw_anchor, anchor_where)
            partition = _require_text(
                _require(anchor, "partition", anchor_where), f"{anchor_where}.partition"
            )
            if partition not in _PARTITIONS:
                raise ValueError(f"{anchor_where}.partition is not a known partition")
            horizon = _require_text(
                _require(anchor, "horizon", anchor_where), f"{anchor_where}.horizon"
            )
            anchor_date = _parse_date(
                _require(anchor, "anchor_date", anchor_where), f"{anchor_where}.anchor_date"
            )
            target_date = _parse_date(
                _require(anchor, "outcome_end_date", anchor_where),
                f"{anchor_where}.outcome_end_date",
            )
            if target_date <= anchor_date:
                raise ValueError(f"{anchor_where}.outcome_end_date must be after anchor_date")
            models = _require_list(
                _require(anchor, "models", anchor_where), f"{anchor_where}.models"
            )
            for model_index, raw_model in enumerate(models):
                model_where = f"{anchor_where}.models[{model_index}]"
                model = _require_mapping(raw_model, model_where)
                model_name = _require_text(
                    _require(model, "model_name", model_where), f"{model_where}.model_name"
                )
                key = (listing_id, partition, horizon, anchor_date, model_name)
                if key in seen:
                    raise ValueError(f"{model_where} duplicates an earlier observation")
                seen.add(key)
                metrics = _require_mapping(
                    _require(model, "metrics", model_where), f"{model_where}.metrics"
                )
                scores, reason = _parse_metrics(metrics, f"{model_where}.metrics")
                if scores is not None:
                    actual = _parse_decimal(
                        metrics["actual_return"], f"{model_where}.metrics.actual_return"
                    )
                    actual_key = (listing_id, partition, horizon, anchor_date)
                    if actuals.setdefault(actual_key, actual) != actual:
                        raise ValueError(
                            f"{model_where}.metrics.actual_return differs across models "
                            "for the same listing and anchor"
                        )
                observation = _Observation(
                    listing_id=listing_id,
                    anchor_date=anchor_date,
                    target_date=target_date,
                    scores=scores,
                    unavailable_reason=reason,
                )
                entries.append((partition, horizon, model_name, observation))
    if not entries:
        raise ValueError("study document contains no model observations")
    return entries, sorted(modes)


def _parse_metrics(metrics: Mapping[str, Any], where: str) -> tuple[_Scores | None, str | None]:
    actual_raw = _require(metrics, "actual_return", where)
    predicted_raw = _require(metrics, "predicted_returns", where)
    included_raw = _require(metrics, "interval_included", where)
    reason_raw = _require(metrics, "unavailable_reason", where)
    if reason_raw is not None:
        reason = _require_text(reason_raw, f"{where}.unavailable_reason")
        if actual_raw is not None or predicted_raw is not None or included_raw is not None:
            raise ValueError(f"{where} has an unavailable_reason but also scored values")
        return None, reason
    if actual_raw is None or predicted_raw is None:
        raise ValueError(
            f"{where} lacks actual_return or predicted_returns without an unavailable_reason"
        )
    actual = _parse_decimal(actual_raw, f"{where}.actual_return")
    predicted = _require_mapping(predicted_raw, f"{where}.predicted_returns")
    lower, median, upper = (
        _parse_decimal(
            _require(predicted, name, f"{where}.predicted_returns"),
            f"{where}.predicted_returns.{name}",
        )
        for name in ("lower", "median", "upper")
    )
    if not lower <= median <= upper:
        raise ValueError(f"{where}.predicted_returns must satisfy lower <= median <= upper")
    included = lower <= actual <= upper
    if not isinstance(included_raw, bool) or included_raw != included:
        raise ValueError(f"{where}.interval_included is inconsistent with the scored values")
    scores = _Scores(
        below_lower=actual < lower,
        below_median=actual < median,
        above_upper=actual > upper,
        included=included,
        signed_median_error=actual - median,
    )
    return scores, None


def _summarize_group(
    partition: str,
    horizon: str,
    model_name: str,
    observations: Sequence[_Observation],
) -> dict[str, Any]:
    reasons = Counter(
        item.unavailable_reason for item in observations if item.unavailable_reason is not None
    )
    scored_by_cohort: dict[date, list[_Scores]] = {}
    for item in observations:
        if item.scores is not None:
            scored_by_cohort.setdefault(item.target_date, []).append(item.scores)

    cohort_rates: list[dict[str, Decimal]] = []
    cohort_errors: list[Decimal] = []
    for target in sorted(scored_by_cohort):
        scores = scored_by_cohort[target]
        count = Decimal(len(scores))
        cohort_rates.append(
            {
                name: Decimal(sum(1 for item in scores if getattr(item, name))) / count
                for name in _INDICATORS
            }
        )
        errors = sorted(item.signed_median_error for item in scores)
        cohort_errors.append(sum(errors, Decimal(0)) / count)

    rates: dict[str, str | None] = {name: None for name in _INDICATORS}
    deviations: dict[str, str | None] = {name: None for name in _INDICATORS}
    mean_error: str | None = None
    included_min: str | None = None
    included_max: str | None = None
    if cohort_rates:
        cohort_count = Decimal(len(cohort_rates))
        for name in _INDICATORS:
            rate = _quantize(sum((item[name] for item in cohort_rates), Decimal(0)) / cohort_count)
            rates[name] = format(rate, "f")
            deviations[name] = _format(rate - Decimal(_NOMINAL[name]))
        mean_error = _format(sum(sorted(cohort_errors), Decimal(0)) / cohort_count)
        included_rates = [item["included"] for item in cohort_rates]
        included_min = _format(min(included_rates))
        included_max = _format(max(included_rates))

    insufficient_reason: str | None = None
    if not observations:
        insufficient_reason = "no_eligible_observations"
    elif not cohort_rates:
        insufficient_reason = "no_available_cohort_metrics"
    return {
        "partition": partition,
        "horizon": horizon,
        "model_name": model_name,
        "observation_count": sum(len(items) for items in scored_by_cohort.values()),
        "unavailable_count": sum(reasons.values()),
        "unavailable_reasons": {reason: reasons[reason] for reason in sorted(reasons)},
        "target_cohort_count": len({item.target_date for item in observations}),
        "scored_cohort_count": len(scored_by_cohort),
        "distinct_listing_count": len({item.listing_id for item in observations}),
        "non_overlapping_target_count": _non_overlapping_target_count(observations),
        "scored_non_overlapping_target_count": _non_overlapping_target_count(
            item for item in observations if item.scores is not None
        ),
        "nominal": dict(_NOMINAL),
        "rates": rates,
        "deviation_from_nominal": deviations,
        "mean_signed_median_error": mean_error,
        "included_cohort_min": included_min,
        "included_cohort_max": included_max,
        "insufficient_reason": insufficient_reason,
    }


def _non_overlapping_target_count(observations: Iterable[_Observation]) -> int:
    intervals = sorted(
        {(item.anchor_date, item.target_date) for item in observations},
        key=lambda value: (value[1], value[0]),
    )
    count = 0
    last_end: date | None = None
    for anchor_date, target_date in intervals:
        if last_end is None or anchor_date >= last_end:
            count += 1
            last_end = target_date
    return count


def _quantize(value: Decimal) -> Decimal:
    quantized = value.quantize(_QUANTUM, rounding=ROUND_HALF_EVEN)
    return abs(quantized) if quantized.is_zero() else quantized


def _format(value: Decimal) -> str:
    return format(_quantize(value), "f")


def _horizon_sort_key(horizon: str) -> tuple[int, int, str]:
    match = _HORIZON_PATTERN.match(horizon)
    if match is None:
        return (1, 0, horizon)
    return (0, int(match[1]) * (12 if match[2] == "y" else 1), horizon)


def _canonical_sha256(document: Mapping[str, Any]) -> str:
    try:
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError(
            "study document must be JSON-serializable (use serialize_price_product_study output)"
        ) from exc
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{where}.{key} is missing")
    return mapping[key]


def _require_mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    return value


def _require_list(value: object, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be an array")
    return value


def _validate_scope(scope: Mapping[str, Any]) -> None:
    unknown = set(scope) - _SCOPE_KEYS
    if unknown:
        raise ValueError("scope contains an unrecognized key")
    for key, value in scope.items():
        items = value if isinstance(value, list | tuple) else [value]
        for item in items:
            if isinstance(item, float):
                if not math.isfinite(item):
                    raise ValueError(f"scope.{key} must be finite")
            elif item is not None and not isinstance(item, bool | int | str):
                raise ValueError(f"scope.{key} must contain only scalar values")
            elif isinstance(item, str) and (len(item) > _MAX_TEXT_LENGTH or not item.isprintable()):
                raise ValueError(f"scope.{key} must contain short printable strings")


def _require_text(value: object, where: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_TEXT_LENGTH
        or not value.isprintable()
    ):
        raise ValueError(f"{where} must be a short printable string")
    return value


def _parse_date(value: object, where: str) -> date:
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.isoformat() == value:
            return parsed
    raise ValueError(f"{where} must be an ISO calendar date string")


def _parse_decimal(value: object, where: str) -> Decimal:
    if not isinstance(value, str) or value != value.strip() or "_" in value:
        raise ValueError(f"{where} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{where} is not a parsable decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{where} must be a finite decimal")
    return parsed


def _render_text(ledger: Mapping[str, Any]) -> str:
    groups: list[Mapping[str, Any]] = list(ledger["groups"])
    lines = [
        "research-product quantile calibration ledger",
        f"schema={ledger['schema']} source_schema={ledger['source_schema']}",
        (
            f"source_run_id={ledger['source_run_id']} "
            f"source_evidence_label={ledger['source_evidence_label']}"
        ),
        (
            f"source_execution_modes={','.join(ledger['source_execution_modes'])} "
            f"snapshot_grade={ledger['source_snapshot_grade']} "
            f"issued_on_time={ledger['source_issued_on_time']}"
        ),
        f"evidence_label={ledger['evidence_label']} input_sha256={ledger['input_sha256']}",
    ]
    lines.extend(f"source_disclosure: {item}" for item in ledger["source_disclosures"])
    lines.extend(f"disclosure: {item}" for item in ledger["disclosures"])
    nominal = " ".join(f"{name}={_NOMINAL[name]}" for name in _INDICATORS)
    lines.append(f"nominal (frozen p20/p50/p80): {nominal}")

    def identity(group: Mapping[str, Any]) -> list[str]:
        return [str(group["partition"]), str(group["horizon"]), str(group["model_name"])]

    def cell(value: object) -> str:
        return "n/a" if value is None else str(value)

    lines.append("support:")
    lines.extend(
        _table(
            (
                "partition",
                "horizon",
                "model",
                "obs",
                "unavail",
                "cohorts",
                "scored_cohorts",
                "listings",
                "nonoverlap",
                "scored_nonoverlap",
                "reason",
            ),
            [
                identity(group)
                + [
                    cell(group["observation_count"]),
                    cell(group["unavailable_count"]),
                    cell(group["target_cohort_count"]),
                    cell(group["scored_cohort_count"]),
                    cell(group["distinct_listing_count"]),
                    cell(group["non_overlapping_target_count"]),
                    cell(group["scored_non_overlapping_target_count"]),
                    group["insufficient_reason"] or "-",
                ]
                for group in groups
            ],
        )
    )
    lines.append("calibration (cohort-first rates; dev_* = rate - nominal):")
    lines.extend(
        _table(
            (
                "partition",
                "horizon",
                "model",
                "below_lower",
                "below_median",
                "above_upper",
                "included",
                "dev_lower",
                "dev_median",
                "dev_upper",
                "dev_incl",
                "mean_err",
                "incl_min",
                "incl_max",
            ),
            [
                identity(group)
                + [cell(group["rates"][name]) for name in _INDICATORS]
                + [cell(group["deviation_from_nominal"][name]) for name in _INDICATORS]
                + [
                    cell(group["mean_signed_median_error"]),
                    cell(group["included_cohort_min"]),
                    cell(group["included_cohort_max"]),
                ]
                for group in groups
            ],
        )
    )
    unavailable = [
        f"  {' '.join(identity(group))} {reason}={count}"
        for group in groups
        for reason, count in group["unavailable_reasons"].items()
    ]
    if unavailable:
        lines.append("unavailable reasons (excluded from rates):")
        lines.extend(unavailable)
    return "\n".join(lines)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [
        max([len(header)] + [len(row[index]) for row in rows])
        for index, header in enumerate(headers)
    ]

    def line(values: Sequence[str]) -> str:
        return ("  " + "  ".join(value.ljust(widths[i]) for i, value in enumerate(values))).rstrip()

    return [line(headers), *(line(row) for row in rows)]

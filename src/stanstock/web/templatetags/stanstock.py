from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from django import template

register = template.Library()

DISPLAY_LABELS = {
    "12m": "12 months",
    "3y": "3 years",
    "5y": "5 years",
    "6m": "6 months",
    "daily": "Daily market refresh",
    "europe": "Europe",
    "empirical_calibrated": "Probability gate passed — not calibrated",
    "empirical_range_only": "Analog range only — probability withheld",
    "empirical_skill_supported": "Positive prequential probability skill — not calibrated",
    "filings_xbrl_org": "filings.xbrl.org",
    "heuristic": "Heuristic",
    "long": "3+ years (legacy)",
    "medium": "6-12 months (legacy)",
    "price_history": "Price history",
    "price_only_baseline": "Price-only baseline",
    "sec": "SEC EDGAR",
    "short": "1-10 trading days",
    "snapshot_portfolios": "Portfolio snapshots",
    "split_adjusted_price_return": "Split-adjusted price return",
    "stock_catalog": "Stock catalog",
    "synthetic_demo": "Synthetic demo",
    "twelve_data": "Twelve Data",
    "us": "US",
}


def _display_decimal(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return number if number.is_finite() else None


def _format_percentage(number: Decimal, digits: int) -> str:
    return f"{number * Decimal(100):+.{digits}f}%"


@register.filter
def percentage(value: object, digits: int = 1) -> str:
    number = _display_decimal(value)
    if number is None:
        return "Unavailable"
    return _format_percentage(number, digits)


@register.filter
def price(value: object) -> str:
    if value is None or value == "":
        return "Unavailable"
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return "Unavailable"
    if not number.is_finite():
        return "Unavailable"

    digits = 2 if abs(number) >= 1 else 4
    rendered = f"{number:,.{digits}f}"
    if digits > 2:
        whole, fraction = rendered.split(".", maxsplit=1)
        fraction = fraction.rstrip("0")
        fraction = fraction.ljust(2, "0")
        rendered = f"{whole}.{fraction}"
    return rendered


@register.filter
def display_label(value: object) -> str:
    if value is None or value == "":
        return "Unavailable"
    text = str(value).strip()
    if not text:
        return "Unavailable"
    return DISPLAY_LABELS.get(text.lower(), text.replace("_", " ").replace("-", " ").title())


@register.filter
def label_list(value: object, separator: str = ", ") -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, str):
        return display_label(value)
    if not isinstance(value, Iterable) or isinstance(value, Mapping):
        return display_label(value)
    labels = [display_label(item) for item in value]
    return separator.join(labels) if labels else "Unavailable"


@register.filter
def scenario_range(value: object) -> str:
    if not isinstance(value, Mapping):
        return "Insufficient evidence"
    values = tuple(
        _display_decimal(_first(value, key, f"{key}_return")) for key in ("bear", "base", "bull")
    )
    if any(item is None for item in values):
        return "Insufficient evidence"
    return " / ".join(_format_percentage(cast(Decimal, item), 1) for item in values)


@register.filter
def validated_medium_forecast_scenario(value: object, horizon: object = None) -> object:
    """Pass non-v2 values through and fail malformed v2 claims closed."""
    if not isinstance(value, Mapping):
        return value
    if not _is_medium_v2_scenario(value):
        return value
    try:
        _validate_medium_v2_scenario(value, horizon=str(horizon))
    except (InvalidOperation, OverflowError, TypeError, ValueError):
        return {"medium_v2_invalid": True}
    return value


@register.filter
def mapping_items(value: object) -> list[tuple[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    return [(str(key), item) for key, item in value.items()]


def _first(value: Mapping[object, object], *keys: str) -> object | None:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _is_medium_v2_scenario(value: Mapping[object, object]) -> bool:
    probability_evidence = value.get("probability_evidence")
    training_evidence = value.get("training_evidence")
    return bool(
        value.get("method_version") == "us-price-medium-v2"
        or value.get("calculation_schema_version") == 2
        or value.get("schema_version") == 2
        or "predictive_distribution" in value
        or "evidence" in value
        or isinstance(probability_evidence, Mapping)
        and bool({"status", "reasons"} & set(probability_evidence))
        or isinstance(training_evidence, Mapping)
        and bool(
            {
                "test_policy",
                "aggregation_policy",
                "calibration_claim",
                "significance_claim",
                "profitability_claim",
                "alpha_claim",
            }
            & set(training_evidence)
        )
        or value.get("confidence_status") == "empirical_skill_supported"
    )


def _validate_medium_v2_scenario(
    value: Mapping[object, object],
    *,
    horizon: str,
) -> None:
    _medium_v2_horizon_floors(horizon)
    expected_top = {
        "bear",
        "base",
        "bull",
        "probability_positive",
        "confidence",
        "confidence_status",
        "insufficiency_reason",
        "method",
        "method_version",
        "calculation_schema_version",
        "current_state",
        "support",
        "probability_evidence",
        "predictive_distribution",
        "evidence",
        "formula_inputs",
        "return_basis",
        "dividends_included",
        "training_evidence",
    }
    if (
        set(value) != expected_top
        or value["method"] != "conditional_empirical_price"
        or value["method_version"] != "us-price-medium-v2"
        or value["calculation_schema_version"] != 2
        or value["return_basis"] != "split_adjusted_price_return"
        or value["dividends_included"] is not False
        or not isinstance(value["insufficiency_reason"], str)
        or not _finite_number(value["confidence"])
        or not Decimal("0") <= _number(value["confidence"]) <= Decimal("100")
    ):
        raise ValueError
    current_state = _exact_mapping(
        value["current_state"],
        {
            "anchor_date",
            "relative_momentum",
            "drawdown",
            "volatility",
            "market_trend",
            "market_volatility",
            "relative_momentum_bucket",
            "drawdown_bucket",
            "volatility_bucket",
            "market_trend_bucket",
            "market_volatility_bucket",
            "close_vs_sma_50",
            "close_vs_sma_200",
            "downside_volatility",
            "average_dollar_volume",
        },
    )
    _validate_medium_v2_current_state(current_state)
    support = _exact_mapping(
        value["support"],
        {
            "raw_matches",
            "effective_cohorts",
            "distinct_listings",
            "calendar_start",
            "calendar_end",
            "market_regimes",
            "fallback_level",
            "shrinkage_weight",
            "dispersion",
        },
    )
    _validate_medium_v2_support(support)
    probability_evidence = _exact_mapping(
        value["probability_evidence"],
        {
            "status",
            "reasons",
            "calendar_span_days",
            "distinct_matched_market_regimes",
            "distinct_panel_market_regimes",
            "minimum_effective_cohorts",
            "minimum_distinct_listings",
            "minimum_calendar_span_days",
            "minimum_distinct_market_regimes",
        },
    )
    if (
        probability_evidence["status"] not in {"published", "withheld", "not_evaluable"}
        or not isinstance(probability_evidence["reasons"], list)
        or not all(
            isinstance(reason, str) and bool(reason) for reason in probability_evidence["reasons"]
        )
        or not all(
            _nonnegative_int(probability_evidence[key])
            for key in (
                "minimum_effective_cohorts",
                "minimum_distinct_listings",
                "minimum_calendar_span_days",
                "minimum_distinct_market_regimes",
            )
        )
        or not all(
            probability_evidence[key] is None or _nonnegative_int(probability_evidence[key])
            for key in (
                "calendar_span_days",
                "distinct_matched_market_regimes",
                "distinct_panel_market_regimes",
            )
        )
    ):
        raise ValueError
    predictive = _exact_mapping(
        value["predictive_distribution"],
        {
            "kind",
            "cdf_event",
            "positive_event",
            "quantile_convention",
            "overlap_policy",
            "matched",
            "unconditional",
            "p20",
            "p50",
            "p80",
            "probability_positive_raw",
            "probability_positive_published",
        },
    )
    if (
        predictive["kind"] != "cohort_equal_empirical_cdf_mixture"
        or predictive["cdf_event"] != "return_lte_x"
        or predictive["positive_event"] != "return_gt_0"
        or predictive["quantile_convention"] != "left_inverse_first_cdf_ge_q"
        or predictive["overlap_policy"] != "matched_rows_receive_mass_in_both_normalized_components"
    ):
        raise ValueError
    component_keys = {
        "component_mass",
        "normalized_mass",
        "p20",
        "p50",
        "p80",
        "probability_positive_raw",
        "raw_observations",
        "effective_cohorts",
        "distinct_listings",
        "calendar_start",
        "calendar_end",
        "market_regimes",
        "dispersion",
    }
    for component_name in ("matched", "unconditional"):
        component = _exact_mapping(predictive[component_name], component_keys)
        if not all(
            _nonnegative_int(component[key])
            for key in (
                "raw_observations",
                "effective_cohorts",
                "distinct_listings",
            )
        ) or not isinstance(component["market_regimes"], list):
            raise ValueError
    evidence = _exact_mapping(value["evidence"], {"base_accuracy", "probability_skill", "interval"})
    base = _exact_mapping(
        evidence["base_accuracy"],
        {
            "status",
            "test_origins",
            "test_predictions",
            "weighting",
            "mean_absolute_error",
            "unconditional_mean_absolute_error",
            "spy_relative_mean_absolute_error",
            "spy_relative_baseline_method",
            "maximum_baseline_mae_ratio",
        },
    )
    skill = _exact_mapping(
        evidence["probability_skill"],
        {
            "status",
            "test_origins",
            "test_predictions",
            "weighting",
            "event",
            "model_brier_score",
            "reference_brier_score",
            "brier_skill_score",
            "reference_method",
            "minimum_brier_skill_exclusive",
            "zero_reference_policy",
        },
    )
    interval = _exact_mapping(
        evidence["interval"],
        {
            "status",
            "test_origins",
            "test_predictions",
            "weighting",
            "alpha",
            "nominal_coverage",
            "endpoint_policy",
            "empirical_coverage",
            "below_rate",
            "above_rate",
            "mean_width",
            "model_mean_interval_score",
            "reference_mean_interval_score",
            "reference_method",
        },
    )
    if (
        skill["event"] != "return_gt_0"
        or skill["reference_method"] != "prequential_unconditional"
        or not _exact_number(skill["minimum_brier_skill_exclusive"], 0.0)
        or skill["zero_reference_policy"] != "null_no_epsilon"
        or not _exact_number(interval["alpha"], 0.4)
        or not _exact_number(interval["nominal_coverage"], 0.6)
        or interval["endpoint_policy"] != "inclusive"
        or interval["reference_method"] != "prequential_unconditional"
    ):
        raise ValueError
    _validate_medium_v2_evidence(
        base_accuracy=base,
        probability_skill=skill,
        interval=interval,
    )
    formula = _exact_mapping(value["formula_inputs"], {"scenario"})
    formula_scenario = _exact_mapping(
        formula["scenario"], {"bear", "base", "bull", "probability_positive"}
    )
    training = _exact_mapping(
        value["training_evidence"],
        {
            "grade",
            "current_universe_survivorship_bias",
            "label_policy",
            "test_policy",
            "cohort_policy",
            "aggregation_policy",
            "calibration_claim",
            "significance_claim",
            "profitability_claim",
            "alpha_claim",
        },
    )
    if (
        training
        != {
            "grade": "research",
            "current_universe_survivorship_bias": True,
            "label_policy": "training_label_end_date_lte_origin",
            "test_policy": "test_outcome_never_enters_its_origin_training_or_gates",
            "cohort_policy": "fixed_epoch_non_overlapping",
            "aggregation_policy": "date_equal_listing_equal_within_origin",
            "calibration_claim": False,
            "significance_claim": False,
            "profitability_claim": False,
            "alpha_claim": False,
        }
        or training["current_universe_survivorship_bias"] is not True
        or training["calibration_claim"] is not False
        or training["significance_claim"] is not False
        or training["profitability_claim"] is not False
        or training["alpha_claim"] is not False
    ):
        raise ValueError
    _validate_medium_v2_scenario_state(
        horizon=horizon,
        value=value,
        support=support,
        predictive=predictive,
        matched=_exact_mapping(predictive["matched"], component_keys),
        unconditional=_exact_mapping(predictive["unconditional"], component_keys),
        formula_scenario=formula_scenario,
        probability_evidence=probability_evidence,
        skill=skill,
    )


def _validate_medium_v2_current_state(current_state: Mapping[object, object]) -> None:
    anchor = current_state["anchor_date"]
    if anchor is not None:
        if not isinstance(anchor, str):
            raise ValueError
        try:
            parsed_anchor = date.fromisoformat(anchor)
        except ValueError:
            raise ValueError from None
        if parsed_anchor.isoformat() != anchor:
            raise ValueError
    bucket_keys = {
        "relative_momentum_bucket",
        "drawdown_bucket",
        "volatility_bucket",
        "market_trend_bucket",
        "market_volatility_bucket",
    }
    nonnegative_keys = {
        "volatility",
        "market_volatility",
        "downside_volatility",
        "average_dollar_volume",
    }
    for key, item in current_state.items():
        if key == "anchor_date" or item is None:
            continue
        if key in bucket_keys:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError
        elif not _finite_number(item):
            raise ValueError
        if key in nonnegative_keys and _number(item) < 0:
            raise ValueError


def _validate_medium_v2_support(support: Mapping[object, object]) -> None:
    fallback = support["fallback_level"]
    if fallback not in {
        "exact_state",
        "without_market_volatility",
        "stock_state",
        "momentum_drawdown",
        "relative_momentum",
        "unconditional",
        "unavailable",
    } or not _bounded_number(support["shrinkage_weight"]):
        raise ValueError
    raw = support["raw_matches"]
    cohorts = support["effective_cohorts"]
    listings = support["distinct_listings"]
    if not all(_nonnegative_int(item) for item in (raw, cohorts, listings)):
        raise ValueError
    if raw == 0:
        if (
            cohorts != 0
            or listings != 0
            or support["calendar_start"] is not None
            or support["calendar_end"] is not None
            or support["market_regimes"] != []
            or support["dispersion"] is not None
            or fallback != "unavailable"
            or _number(support["shrinkage_weight"]) != 0
        ):
            raise ValueError
        return
    if (
        not _positive_counts(raw, cohorts, listings)
        or _date_range(support) is None
        or not _regimes(support["market_regimes"], require_nonempty=True)
        or len(cast(list[object], support["market_regimes"])) > cast(int, cohorts)
        or not _nonnegative_number(support["dispersion"])
        or fallback == "unavailable"
        and _number(support["shrinkage_weight"]) != 0
    ):
        raise ValueError


def _validate_medium_v2_evidence(
    *,
    base_accuracy: Mapping[object, object],
    probability_skill: Mapping[object, object],
    interval: Mapping[object, object],
) -> None:
    for block in (base_accuracy, probability_skill, interval):
        origins = block["test_origins"]
        predictions = block["test_predictions"]
        if (
            not _nonnegative_int(origins)
            or not _nonnegative_int(predictions)
            or cast(int, predictions) < cast(int, origins)
            or (origins == 0) != (predictions == 0)
            or block["weighting"] != "date_equal_listing_equal_within_origin"
        ):
            raise ValueError

    base_origins = cast(int, base_accuracy["test_origins"])
    base_metrics = (
        base_accuracy["mean_absolute_error"],
        base_accuracy["unconditional_mean_absolute_error"],
        base_accuracy["spy_relative_mean_absolute_error"],
    )
    if base_accuracy[
        "spy_relative_baseline_method"
    ] != "market_regime_benchmark_median_plus_relative_momentum_excess_median" or not _exact_number(
        base_accuracy["maximum_baseline_mae_ratio"], 1.0
    ):
        raise ValueError
    if base_origins == 0:
        expected_base_status = "not_evaluable"
        if any(item is not None for item in base_metrics):
            raise ValueError
    else:
        if not all(_nonnegative_number(item) for item in base_metrics):
            raise ValueError
        if base_origins < 4:
            expected_base_status = "insufficient_support"
        else:
            model, unconditional, relative = (_number(item) for item in base_metrics)
            expected_base_status = (
                "passed" if model <= unconditional and model <= relative else "failed"
            )
    if base_accuracy["status"] != expected_base_status:
        raise ValueError

    skill_origins = cast(int, probability_skill["test_origins"])
    model_score = probability_skill["model_brier_score"]
    reference_score = probability_skill["reference_brier_score"]
    bss = probability_skill["brier_skill_score"]
    if skill_origins == 0:
        expected_skill_status = "not_evaluable"
        if model_score is not None or reference_score is not None or bss is not None:
            raise ValueError
    else:
        if not _bounded_number(model_score) or not _bounded_number(reference_score):
            raise ValueError
        model_value = float(_number(model_score))
        reference_value = float(_number(reference_score))
        expected_bss = None if reference_value == 0.0 else 1.0 - model_value / reference_value
        if expected_bss is None:
            if bss is not None:
                raise ValueError
        elif not _finite_number(bss) or _number(bss) != Decimal(str(expected_bss)):
            raise ValueError
        if skill_origins < 4:
            expected_skill_status = "insufficient_support"
        elif reference_value == 0.0:
            expected_skill_status = "reference_zero"
        elif model_value < reference_value:
            expected_skill_status = "positive_skill"
        elif model_value == reference_value:
            expected_skill_status = "zero_skill"
        else:
            expected_skill_status = "negative_skill"
    if probability_skill["status"] != expected_skill_status:
        raise ValueError

    interval_origins = cast(int, interval["test_origins"])
    rates = (
        interval["empirical_coverage"],
        interval["below_rate"],
        interval["above_rate"],
    )
    nonnegative_metrics = (
        interval["mean_width"],
        interval["model_mean_interval_score"],
        interval["reference_mean_interval_score"],
    )
    if interval_origins == 0:
        expected_interval_status = "not_evaluable"
        if any(item is not None for item in (*rates, *nonnegative_metrics)):
            raise ValueError
    else:
        partition_total = math.fsum(float(_number(item)) for item in rates)
        if (
            not all(_bounded_number(item) for item in rates)
            or not all(_nonnegative_number(item) for item in nonnegative_metrics)
            or not math.isclose(
                partition_total,
                1.0,
                rel_tol=0.0,
                abs_tol=4 * math.ulp(1.0),
            )
        ):
            raise ValueError
        expected_interval_status = "preliminary" if interval_origins < 4 else "descriptive"
    if interval["status"] != expected_interval_status:
        raise ValueError
    if (
        base_accuracy["test_origins"] != interval["test_origins"]
        or base_accuracy["test_predictions"] != interval["test_predictions"]
        or cast(int, probability_skill["test_origins"]) > cast(int, base_accuracy["test_origins"])
        or cast(int, probability_skill["test_predictions"])
        > cast(int, base_accuracy["test_predictions"])
    ):
        raise ValueError


def _validate_medium_v2_numeric_support(
    *,
    horizon: str,
    support: Mapping[object, object],
    matched: Mapping[object, object],
    unconditional: Mapping[object, object],
    probability_evidence: Mapping[object, object],
) -> None:
    if (
        support["fallback_level"] == "unavailable"
        or cast(int, support["raw_matches"]) < 20
        or cast(int, support["effective_cohorts"]) < 3
        or cast(int, support["distinct_listings"]) < 10
    ):
        raise ValueError
    for component in (matched, unconditional):
        if (
            cast(int, component["raw_observations"]) < 20
            or cast(int, component["effective_cohorts"]) < 3
            or cast(int, component["distinct_listings"]) < 10
        ):
            raise ValueError
    for support_key, component_key in (
        ("raw_matches", "raw_observations"),
        ("effective_cohorts", "effective_cohorts"),
        ("distinct_listings", "distinct_listings"),
        ("calendar_start", "calendar_start"),
        ("calendar_end", "calendar_end"),
        ("market_regimes", "market_regimes"),
        ("dispersion", "dispersion"),
    ):
        if support[support_key] != matched[component_key]:
            raise ValueError
    expected_weight = (
        Decimal("0")
        if support["fallback_level"] == "unconditional"
        else Decimal(
            str(
                cast(int, support["effective_cohorts"])
                / (cast(int, support["effective_cohorts"]) + 4.0)
            )
        )
    )
    if _number(support["shrinkage_weight"]) != expected_weight:
        raise ValueError
    if support["fallback_level"] == "unconditional" and any(
        matched[key] != unconditional[key] for key in matched if key != "component_mass"
    ):
        raise ValueError

    floors = _medium_v2_horizon_floors(horizon)
    dates = _date_range(matched)
    if dates is None:
        raise ValueError
    span = (dates[1] - dates[0]).days
    if (
        probability_evidence["calendar_span_days"] != span
        or probability_evidence["distinct_matched_market_regimes"]
        != len(cast(list[object], matched["market_regimes"]))
        or probability_evidence["distinct_panel_market_regimes"]
        != len(cast(list[object], unconditional["market_regimes"]))
        or any(
            probability_evidence[key] != floors[key]
            for key in (
                "minimum_effective_cohorts",
                "minimum_distinct_listings",
                "minimum_calendar_span_days",
                "minimum_distinct_market_regimes",
            )
        )
    ):
        raise ValueError


def _medium_v2_horizon_floors(horizon: str) -> dict[str, int]:
    if horizon == "6m":
        cohorts, span = 8, 1_095
    elif horizon == "12m":
        cohorts, span = 6, 1_460
    else:
        raise ValueError
    return {
        "minimum_effective_cohorts": cohorts,
        "minimum_distinct_listings": 30,
        "minimum_calendar_span_days": span,
        "minimum_distinct_market_regimes": 3,
    }


def _medium_v2_probability_reasons(
    *,
    horizon: str,
    matched: Mapping[object, object],
    skill: Mapping[object, object],
) -> list[str]:
    floors = _medium_v2_horizon_floors(horizon)
    dates = _date_range(matched)
    if dates is None:
        raise ValueError
    span = (dates[1] - dates[0]).days
    reasons: list[str] = []
    if cast(int, matched["effective_cohorts"]) < floors["minimum_effective_cohorts"]:
        reasons.append(
            f"effective cohorts {matched['effective_cohorts']}/"
            f"{floors['minimum_effective_cohorts']}"
        )
    if cast(int, matched["distinct_listings"]) < floors["minimum_distinct_listings"]:
        reasons.append(
            f"distinct listings {matched['distinct_listings']}/"
            f"{floors['minimum_distinct_listings']}"
        )
    if span < floors["minimum_calendar_span_days"]:
        reasons.append(f"calendar span {span}/{floors['minimum_calendar_span_days']} days")
    regimes = len(cast(list[object], matched["market_regimes"]))
    if regimes < floors["minimum_distinct_market_regimes"]:
        reasons.append(
            f"matched market regimes {regimes}/{floors['minimum_distinct_market_regimes']}"
        )
    status = skill["status"]
    if status == "positive_skill":
        if _number(skill["brier_skill_score"]) <= 0:
            raise ValueError
    elif status in {"not_evaluable", "insufficient_support"}:
        reasons.append(
            f"prequential Brier skill insufficient ({skill['test_origins']}/4 test origins)"
        )
    elif status == "reference_zero":
        reasons.append("prequential unconditional reference Brier score is zero")
    elif status == "zero_skill":
        reasons.append("prequential Brier skill is zero")
    elif status == "negative_skill":
        reasons.append("prequential Brier skill is negative")
    else:
        raise ValueError
    return reasons


def _validate_medium_v2_scenario_state(
    *,
    horizon: str,
    value: Mapping[object, object],
    support: Mapping[object, object],
    predictive: Mapping[object, object],
    matched: Mapping[object, object],
    unconditional: Mapping[object, object],
    formula_scenario: Mapping[object, object],
    probability_evidence: Mapping[object, object],
    skill: Mapping[object, object],
) -> None:
    display_triplet = (value["bear"], value["base"], value["bull"])
    predictive_triplet = (predictive["p20"], predictive["p50"], predictive["p80"])
    formula_triplet = (
        formula_scenario["bear"],
        formula_scenario["base"],
        formula_scenario["bull"],
    )
    all_null = all(item is None for item in display_triplet)
    complete = all(_finite_number(item) for item in display_triplet)
    if not all_null and not complete:
        raise ValueError
    if complete:
        numeric = tuple(_number(item) for item in display_triplet)
        _validate_medium_v2_component(matched, require_nonempty=True)
        _validate_medium_v2_component(unconditional, require_nonempty=True)
        _validate_medium_v2_numeric_support(
            horizon=horizon,
            support=support,
            matched=matched,
            unconditional=unconditional,
            probability_evidence=probability_evidence,
        )
        if (
            not _bounded_number(matched["component_mass"])
            or not _bounded_number(unconditional["component_mass"])
            or _number(matched["component_mass"]) != _number(support["shrinkage_weight"])
            or float(_number(unconditional["component_mass"]))
            != 1.0 - float(_number(support["shrinkage_weight"]))
        ):
            raise ValueError
        if (
            min(numeric) < Decimal("-1")
            or numeric != tuple(sorted(numeric))
            or not _quantized_triplet_equal(display_triplet, predictive_triplet)
            or not _quantized_triplet_equal(display_triplet, formula_triplet)
            or not _exact_triplet_equal(predictive_triplet, formula_triplet)
            or not _bounded_number(predictive["probability_positive_raw"])
        ):
            raise ValueError
        raw = _number(predictive["probability_positive_raw"])
        expected_raw = math.fsum(
            float(_number(component["component_mass"]))
            * float(_number(component["probability_positive_raw"]))
            for component in (matched, unconditional)
        )
        if raw != Decimal(str(expected_raw)):
            raise ValueError
        expected_reasons = _medium_v2_probability_reasons(
            horizon=horizon,
            matched=matched,
            skill=skill,
        )
        if value["confidence_status"] == "empirical_skill_supported":
            if (
                probability_evidence["status"] != "published"
                or probability_evidence["reasons"]
                or expected_reasons
                or skill["status"] != "positive_skill"
                or not _finite_number(skill["brier_skill_score"])
                or _number(skill["brier_skill_score"]) <= 0
                or not _exact_numeric_equal(raw, predictive["probability_positive_published"])
                or not _exact_numeric_equal(raw, formula_scenario["probability_positive"])
                or not _optional_quantized_equal(raw, value["probability_positive"], places=4)
                or value["insufficiency_reason"]
            ):
                raise ValueError
        elif value["confidence_status"] == "empirical_range_only":
            if (
                probability_evidence["status"] != "withheld"
                or probability_evidence["reasons"] != expected_reasons
                or not expected_reasons
                or predictive["probability_positive_published"] is not None
                or formula_scenario["probability_positive"] is not None
                or value["probability_positive"] is not None
                or value["insufficiency_reason"]
                != "Probability withheld: " + "; ".join(expected_reasons)
            ):
                raise ValueError
        else:
            raise ValueError
        expected_confidence = min(
            Decimal("80"),
            Decimal("20")
            + Decimal("60")
            * _number(cast(Mapping[object, object], value["support"])["shrinkage_weight"]),
        )
        if _number(value["confidence"]).quantize(Decimal("0.01")) != (
            expected_confidence.quantize(Decimal("0.01"))
        ):
            raise ValueError
        return
    _validate_medium_v2_component(matched, require_nonempty=False)
    _validate_medium_v2_component(unconditional, require_nonempty=False)
    floors = _medium_v2_horizon_floors(horizon)
    if (
        not all(item is None for item in predictive_triplet)
        or not all(item is None for item in formula_triplet)
        or predictive["probability_positive_raw"] is not None
        or predictive["probability_positive_published"] is not None
        or formula_scenario["probability_positive"] is not None
        or value["probability_positive"] is not None
        or value["confidence_status"] != "insufficient_evidence"
        or _number(value["confidence"]) != 0
        or probability_evidence["status"] != "not_evaluable"
        or probability_evidence["reasons"] != []
        or probability_evidence["calendar_span_days"] is not None
        or probability_evidence["distinct_matched_market_regimes"] is not None
        or probability_evidence["distinct_panel_market_regimes"] is not None
        or any(
            probability_evidence[key] != floors[key]
            for key in (
                "minimum_effective_cohorts",
                "minimum_distinct_listings",
                "minimum_calendar_span_days",
                "minimum_distinct_market_regimes",
            )
        )
        or support["fallback_level"] != "unavailable"
        or _number(support["shrinkage_weight"]) != 0
        or (
            cast(int, support["raw_matches"]) >= 20
            and cast(int, support["effective_cohorts"]) >= 3
            and cast(int, support["distinct_listings"]) >= 10
        )
        or not value["insufficiency_reason"]
    ):
        raise ValueError
    if cast(int, support["raw_matches"]) > 0:
        expected_reason = (
            f"Insufficient non-overlapping {horizon} evidence: "
            f"{support['raw_matches']}/20 observations, "
            f"{support['effective_cohorts']}/3 cohorts, "
            f"{support['distinct_listings']}/10 listings"
        )
        if value["insufficiency_reason"] != expected_reason:
            raise ValueError


def _validate_medium_v2_component(
    component: Mapping[object, object],
    *,
    require_nonempty: bool,
) -> None:
    observations = component["raw_observations"]
    if require_nonempty:
        triplet = (component["p20"], component["p50"], component["p80"])
        numeric = tuple(_number(item) for item in triplet)
        if (
            not _positive_counts(
                observations,
                component["effective_cohorts"],
                component["distinct_listings"],
            )
            or not _exact_number(component["normalized_mass"], 1.0)
            or not all(_finite_number(item) for item in triplet)
            or numeric != tuple(sorted(numeric))
            or min(numeric) < -1
            or not _bounded_number(component["probability_positive_raw"])
            or _date_range(component) is None
            or not _regimes(component["market_regimes"], require_nonempty=True)
            or len(cast(list[object], component["market_regimes"]))
            > cast(int, component["effective_cohorts"])
            or not _nonnegative_number(component["dispersion"])
        ):
            raise ValueError
        return
    if (
        observations != 0
        or component["effective_cohorts"] != 0
        or component["distinct_listings"] != 0
        or component["component_mass"] is not None
        or component["normalized_mass"] is not None
        or component["p20"] is not None
        or component["p50"] is not None
        or component["p80"] is not None
        or component["probability_positive_raw"] is not None
        or component["calendar_start"] is not None
        or component["calendar_end"] is not None
        or component["market_regimes"] != []
        or component["dispersion"] is not None
    ):
        raise ValueError


def _positive_counts(observations: object, cohorts: object, listings: object) -> bool:
    return bool(
        all(_nonnegative_int(item) for item in (observations, cohorts, listings))
        and cast(int, observations) > 0
        and 0 < cast(int, cohorts) <= cast(int, observations)
        and 0 < cast(int, listings) <= cast(int, observations)
    )


def _date_range(value: Mapping[object, object]) -> tuple[date, date] | None:
    start_text = value["calendar_start"]
    end_text = value["calendar_end"]
    if not isinstance(start_text, str) or not isinstance(end_text, str):
        return None
    try:
        start = date.fromisoformat(start_text)
        end = date.fromisoformat(end_text)
    except ValueError:
        return None
    if start.isoformat() != start_text or end.isoformat() != end_text or start > end:
        return None
    return start, end


def _regimes(value: object, *, require_nonempty: bool) -> bool:
    if not isinstance(value, list) or (require_nonempty and not value):
        return False
    return all(isinstance(item, str) and bool(item) for item in value) and value == sorted(
        set(value)
    )


def _nonnegative_number(value: object) -> bool:
    return _finite_number(value) and _number(value) >= 0


def _exact_number(value: object, expected: float) -> bool:
    return _finite_number(value) and _number(value) == Decimal(str(expected))


def _optional_quantized_equal(
    left: object,
    right: object,
    *,
    places: int,
) -> bool:
    if left is None or right is None:
        return left is right
    if not _finite_number(left) or not _finite_number(right):
        return False
    quantum = Decimal(1).scaleb(-places)
    return _number(left).quantize(quantum) == _number(right).quantize(quantum)


def _quantized_triplet_equal(
    left: tuple[object, object, object],
    right: tuple[object, object, object],
) -> bool:
    return all(
        _optional_quantized_equal(left_item, right_item, places=4)
        for left_item, right_item in zip(left, right, strict=True)
    )


def _exact_numeric_equal(left: object, right: object) -> bool:
    return bool(_finite_number(left) and _finite_number(right) and _number(left) == _number(right))


def _exact_triplet_equal(
    left: tuple[object, object, object],
    right: tuple[object, object, object],
) -> bool:
    return all(
        _exact_numeric_equal(left_item, right_item)
        for left_item, right_item in zip(left, right, strict=True)
    )


def _exact_mapping(
    value: object,
    keys: set[str],
) -> Mapping[object, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError
    return value


def _nonnegative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and 0 <= value <= sys.maxsize


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return False
    return _number(value).is_finite()


def _bounded_number(value: object) -> bool:
    return _finite_number(value) and Decimal("0") <= _number(value) <= Decimal("1")


def _number(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError
    return number

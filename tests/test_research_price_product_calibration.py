"""Quantile-calibration ledger: pure arithmetic, fail-closed matrix, and real-study contract.

Everything here is synthetic. The DB-backed tests run the real
``study_price_product_run`` -> ``serialize_price_product_study`` ->
``summarize_quantile_calibration`` flow against the demo product fixture.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from io import StringIO
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from hypothesis import given, settings
from hypothesis import strategies as st

from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset
from stanstock.data.research_product_demo import execute_demo_product_refresh
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.price_product_calibration import (
    CALIBRATION_LEDGER_SCHEMA,
    render_quantile_calibration,
    summarize_quantile_calibration,
)
from stanstock.research.price_product_config import PRODUCT_EFFECTIVE_CONFIG_HASH
from stanstock.research.price_product_study import (
    REPLAY_STUDY_SCHEMA,
    serialize_price_product_study,
    study_price_product_run,
)

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "stanstock"
SIX_PLACES = Decimal("0.000001")
COMMAND = "summarize_price_product_calibration"
LEAK_MARKER = "LEAK-MARKER-9f3a"

Document = dict[str, Any]


def _scored(actual: str, lower: str, median: str, upper: str) -> dict[str, Any]:
    return {
        "actual_return": actual,
        "predicted_returns": {"lower": lower, "median": median, "upper": upper},
        "interval_included": Decimal(lower) <= Decimal(actual) <= Decimal(upper),
        "unavailable_reason": None,
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "actual_return": None,
        "predicted_returns": None,
        "interval_included": None,
        "unavailable_reason": reason,
    }


def _model(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"model_name": name, "metrics": metrics}


def _anchor(
    partition: str, horizon: str, anchor: str, end: str, models: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "partition": partition,
        "horizon": horizon,
        "anchor_date": anchor,
        "outcome_end_date": end,
        "models": models,
    }


def _listing(
    listing_id: str, anchors: list[dict[str, Any]], mode: str = "provider"
) -> dict[str, Any]:
    return {
        "listing_id": listing_id,
        "source_execution": {"mode": mode, "evidence_grade": "research"},
        "anchors": anchors,
    }


def _aggregates(
    listings: list[dict[str, Any]], extra_horizons: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Independent replay-shaped aggregates over the full partition x horizon x model grid."""

    rows: list[tuple[str, str, str, str, date, date, dict[str, Any]]] = []
    for listing in listings:
        for anchor in listing["anchors"]:
            for model in anchor["models"]:
                rows.append(
                    (
                        anchor["partition"],
                        anchor["horizon"],
                        model["model_name"],
                        listing["listing_id"],
                        date.fromisoformat(anchor["anchor_date"]),
                        date.fromisoformat(anchor["outcome_end_date"]),
                        model["metrics"],
                    )
                )
    horizons = list(dict.fromkeys([*(row[1] for row in rows), *extra_horizons]))
    models = list(dict.fromkeys(row[2] for row in rows))
    result: list[dict[str, Any]] = []
    for partition in ("development", "validation", "final_holdout"):
        for horizon in horizons:
            for model in models:
                group = [r for r in rows if r[:3] == (partition, horizon, model)]
                targets = {r[5] for r in group}
                spans = sorted({(r[4], r[5]) for r in group}, key=lambda s: (s[1], s[0]))
                non_overlapping, last_end = 0, None
                for start, end in spans:
                    if last_end is None or start >= last_end:
                        non_overlapping, last_end = non_overlapping + 1, end
                reasons: dict[str, int] = {}
                for r in group:
                    reason = r[6]["unavailable_reason"]
                    if reason is not None:
                        reasons[reason] = reasons.get(reason, 0) + 1
                means = []
                for target in targets:
                    flags = [
                        Decimal(int(r[6]["interval_included"]))
                        for r in group
                        if r[5] == target and r[6]["unavailable_reason"] is None
                    ]
                    if flags:
                        means.append(sum(flags, Decimal(0)) / len(flags))
                result.append(
                    {
                        "partition": partition,
                        "horizon": horizon,
                        "model_name": model,
                        "target_cohort_count": len(targets),
                        "distinct_listing_count": len({r[3] for r in group}),
                        "non_overlapping_target_count": non_overlapping,
                        "unavailable_count": sum(reasons.values()),
                        "unavailable_reasons": dict(sorted(reasons.items())),
                        "averages": {
                            "interval_inclusion": (
                                str(sum(means, Decimal(0)) / len(means)) if means else None
                            )
                        },
                        "insufficient_reason": (
                            "no_eligible_observations"
                            if not group
                            else None
                            if means
                            else "no_available_cohort_metrics"
                        ),
                    }
                )
    return result


def _document(
    listings: list[dict[str, Any]], aggregates: list[dict[str, Any]] | None = None
) -> Document:
    return {
        "schema": REPLAY_STUDY_SCHEMA,
        "config_identity": {"effective_config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH},
        "source_run": {
            "id": "00000000-0000-4000-8000-000000000001",
            "snapshot_grade": "research",
            "issued_on_time": True,
        },
        "protocol_identity": {"evidence_label": "synthetic_retrospective_label"},
        "scope": {"all_selected": True, "studied_listing_count": len(listings)},
        "projection_aggregates": _aggregates(listings) if aggregates is None else aggregates,
        "disclosures": ["source disclosure one", "source disclosure two"],
        "listings": listings,
    }


def _by_key(ledger: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (group["partition"], group["horizon"], group["model_name"]): group
        for group in ledger["groups"]
    }


TRIPLET = ("-0.1000", "0.0000", "0.1000")


def _hand_document() -> Document:
    """Validation 6m: cohort 2024-07-01 has three listings, cohort 2024-12-31 has one.

    Hand-computed with triplet (-0.10, 0.00, 0.10):
      cohort A: -0.20 (below lower), 0.05 (inside), 0.30 (above upper)
      cohort B: -0.02 (inside, below median)
    Validation 12m: two overlapping anchors on one listing, actuals 0.00 and 0.20.
    """

    return _document(
        [
            _listing(
                "L1",
                [
                    _anchor(
                        "validation",
                        "6m",
                        "2024-01-02",
                        "2024-07-01",
                        [_model("m1", _scored("-0.2000", *TRIPLET))],
                    ),
                    _anchor(
                        "validation",
                        "6m",
                        "2024-07-01",
                        "2024-12-31",
                        [_model("m1", _scored("-0.0200", *TRIPLET))],
                    ),
                    _anchor(
                        "validation",
                        "12m",
                        "2024-01-02",
                        "2024-12-31",
                        [_model("m1", _scored("0.0000", *TRIPLET))],
                    ),
                    _anchor(
                        "validation",
                        "12m",
                        "2024-03-01",
                        "2025-03-03",
                        [_model("m1", _scored("0.2000", *TRIPLET))],
                    ),
                ],
            ),
            _listing(
                "L2",
                [
                    _anchor(
                        "validation",
                        "6m",
                        "2024-01-02",
                        "2024-07-01",
                        [_model("m1", _scored("0.0500", *TRIPLET))],
                    )
                ],
            ),
            _listing(
                "L3",
                [
                    _anchor(
                        "validation",
                        "6m",
                        "2024-01-02",
                        "2024-07-01",
                        [_model("m1", _scored("0.3000", *TRIPLET))],
                    )
                ],
            ),
        ]
    )


# --- (a) hand-computed, cohort-first -------------------------------------------------------


def test_hand_computed_ledger_is_cohort_first_not_pooled() -> None:
    ledger = summarize_quantile_calibration(_hand_document())
    groups = _by_key(ledger)

    assert [(g["partition"], g["horizon"]) for g in ledger["groups"]] == [
        ("development", "6m"),
        ("development", "12m"),
        ("validation", "6m"),
        ("validation", "12m"),
        ("final_holdout", "6m"),
        ("final_holdout", "12m"),
    ]
    six_months = groups[("validation", "6m", "m1")]
    # Cohort A: 1/3 each tail, 1/3 inside; cohort B: inside and below the median.
    # Cohort-first: (1/3 + 0)/2 tails, (1/3 + 1)/2 inside. Pooled would be 0.25 / 0.5.
    assert six_months == {
        "partition": "validation",
        "horizon": "6m",
        "model_name": "m1",
        "observation_count": 4,
        "unavailable_count": 0,
        "unavailable_reasons": {},
        "target_cohort_count": 2,
        "scored_cohort_count": 2,
        "distinct_listing_count": 3,
        "non_overlapping_target_count": 2,
        "scored_non_overlapping_target_count": 2,
        "nominal": {
            "below_lower": "0.2",
            "below_median": "0.5",
            "above_upper": "0.2",
            "included": "0.6",
        },
        "rates": {
            "below_lower": "0.166667",
            "below_median": "0.666667",
            "above_upper": "0.166667",
            "included": "0.666667",
        },
        "deviation_from_nominal": {
            "below_lower": "-0.033333",
            "below_median": "0.166667",
            "above_upper": "-0.033333",
            "included": "0.066667",
        },
        # Cohort means 0.05 and -0.02; pooled would be 0.0325.
        "mean_signed_median_error": "0.015000",
        "included_cohort_min": "0.333333",
        "included_cohort_max": "1.000000",
        "insufficient_reason": None,
    }

    twelve = groups[("validation", "12m", "m1")]
    assert twelve["observation_count"] == 2
    assert twelve["target_cohort_count"] == 2
    assert twelve["distinct_listing_count"] == 1
    # (2024-01-02 -> 2024-12-31) and (2024-03-01 -> 2025-03-03) overlap.
    assert twelve["non_overlapping_target_count"] == 1
    assert twelve["rates"] == {
        "below_lower": "0.000000",
        "below_median": "0.000000",
        "above_upper": "0.500000",
        "included": "0.500000",
    }
    assert twelve["deviation_from_nominal"] == {
        "below_lower": "-0.200000",
        "below_median": "-0.500000",
        "above_upper": "0.300000",
        "included": "-0.100000",
    }
    assert twelve["mean_signed_median_error"] == "0.100000"
    assert twelve["included_cohort_min"] == "0.000000"
    assert twelve["included_cohort_max"] == "1.000000"


def test_ledger_top_level_identity_and_disclosures() -> None:
    document = _hand_document()
    ledger = summarize_quantile_calibration(document)

    assert set(ledger) == {
        "schema",
        "source_schema",
        "source_run_id",
        "source_evidence_label",
        "source_execution_modes",
        "source_snapshot_grade",
        "source_issued_on_time",
        "source_disclosures",
        "scope",
        "input_sha256",
        "evidence_label",
        "disclosures",
        "groups",
    }
    assert REPLAY_STUDY_SCHEMA == "research-product-retrospective-study@1"
    assert ledger["schema"] == CALIBRATION_LEDGER_SCHEMA
    assert CALIBRATION_LEDGER_SCHEMA == "research-product-quantile-calibration-ledger@1"
    assert ledger["source_schema"] == REPLAY_STUDY_SCHEMA
    assert ledger["source_run_id"] == "00000000-0000-4000-8000-000000000001"
    assert ledger["source_evidence_label"] == "synthetic_retrospective_label"
    assert ledger["evidence_label"] == "research_grade_retrospective_descriptive"
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert ledger["input_sha256"] == hashlib.sha256(canonical.encode("ascii")).hexdigest()
    text = " ".join(ledger["disclosures"])
    for phrase in (
        "not a calibration claim",
        "adoption evidence",
        "not independent trials",
        "non_overlapping_target_count",
        "target_cohort_count",
        "scored_cohort_count",
        "scored_non_overlapping_target_count",
        "research-grade",
        "not a below/above hit",
        "return_decimal_places 4",
        "0.999999 or 1.000001",
        "already inspected",
        "not fresh confirmation",
        "never be used to tune",
        "identifies the input and does not authenticate it",
        "not re-verified",
        "not a calibration test",
        "below_median rate is the location check",
        "No sampling-uncertainty threshold",
        "uninterpretable",
    ):
        assert phrase in text
    for removed in ("too narrow", "too wide"):
        assert removed not in text
    assert ledger["source_execution_modes"] == ["provider"]
    assert ledger["source_snapshot_grade"] == "research"
    assert ledger["source_issued_on_time"] is True
    assert ledger["source_disclosures"] == document["disclosures"]
    assert ledger["scope"] == document["scope"]
    assert ledger["scope"] is not document["scope"]


def test_group_order_is_partition_then_horizon_duration_then_model_name() -> None:
    models = [_model(name, _scored("0.0000", *TRIPLET)) for name in ("zeta", "alpha")]
    anchors = [
        _anchor("validation", horizon, "2024-01-02", "2024-06-03", copy.deepcopy(models))
        for horizon in ("5y", "custom", "3y", "12m", "6m")
    ]
    ledger = summarize_quantile_calibration(_document([_listing("L1", anchors)]))

    keys = [(g["partition"], g["horizon"], g["model_name"]) for g in ledger["groups"]]
    assert len(keys) == 3 * 5 * 2
    assert [key[0] for key in keys[:10]] == ["development"] * 10
    assert [key[1:] for key in keys[:10]] == [
        ("6m", "alpha"),
        ("6m", "zeta"),
        ("12m", "alpha"),
        ("12m", "zeta"),
        ("3y", "alpha"),
        ("3y", "zeta"),
        ("5y", "alpha"),
        ("5y", "zeta"),
        ("custom", "alpha"),
        ("custom", "zeta"),
    ]
    assert [key[0] for key in keys] == ["development"] * 10 + ["validation"] * 10 + [
        "final_holdout"
    ] * 10


# --- (b) ties and the per-observation invariant -------------------------------------------

TIE_CASES = [
    # (id, actual, triplet, below_lower, below_median, above_upper, included)
    ("equals_lower", "-0.1000", TRIPLET, 0, 1, 0, 1),
    ("equals_median", "0.0000", TRIPLET, 0, 0, 0, 1),
    ("equals_upper", "0.1000", TRIPLET, 0, 0, 0, 1),
    ("just_below_lower", "-0.1001", TRIPLET, 1, 1, 0, 0),
    ("just_above_upper", "0.1001", TRIPLET, 0, 0, 1, 0),
    ("between_lower_and_median", "-0.0500", TRIPLET, 0, 1, 0, 1),
    ("between_median_and_upper", "0.0500", TRIPLET, 0, 0, 0, 1),
    ("degenerate_triplet_equal", "0.0500", ("0.0500", "0.0500", "0.0500"), 0, 0, 0, 1),
    ("equal_lower_and_median", "0.0000", ("0.0000", "0.0000", "0.1000"), 0, 0, 0, 1),
    ("equal_median_and_upper", "0.1000", ("-0.1000", "0.1000", "0.1000"), 0, 0, 0, 1),
]


@pytest.mark.parametrize(
    ("actual", "triplet", "below_lower", "below_median", "above_upper", "included"),
    [case[1:] for case in TIE_CASES],
    ids=[case[0] for case in TIE_CASES],
)
def test_tie_convention_and_partition_invariant(
    actual: str,
    triplet: tuple[str, str, str],
    below_lower: int,
    below_median: int,
    above_upper: int,
    included: int,
) -> None:
    document = _document(
        [
            _listing(
                "L1",
                [
                    _anchor(
                        "validation",
                        "6m",
                        "2024-01-02",
                        "2024-07-01",
                        [_model("m", _scored(actual, *triplet))],
                    )
                ],
            )
        ]
    )
    group = _by_key(summarize_quantile_calibration(document))[("validation", "6m", "m")]
    rates = {name: Decimal(value) for name, value in group["rates"].items()}

    assert rates == {
        "below_lower": Decimal(below_lower),
        "below_median": Decimal(below_median),
        "above_upper": Decimal(above_upper),
        "included": Decimal(included),
    }
    assert rates["below_lower"] + rates["included"] + rates["above_upper"] == 1
    assert Decimal(group["mean_signed_median_error"]) == Decimal(actual) - Decimal(triplet[1])


@settings(max_examples=100, deadline=None)
@given(
    quantiles=st.lists(st.integers(-5000, 5000), min_size=3, max_size=3).map(sorted),
    actual=st.integers(-8000, 8000),
)
def test_property_single_observation_matches_exact_comparisons(
    quantiles: list[int], actual: int
) -> None:
    def text(value: int) -> str:
        return format(Decimal(value).scaleb(-4), "f")

    lower, median, upper = quantiles
    document = _document(
        [
            _listing(
                "L1",
                [
                    _anchor(
                        "development",
                        "6m",
                        "2024-01-02",
                        "2024-07-01",
                        [
                            _model(
                                "m", _scored(text(actual), text(lower), text(median), text(upper))
                            )
                        ],
                    )
                ],
            )
        ]
    )
    rates = _by_key(summarize_quantile_calibration(document))[("development", "6m", "m")]["rates"]

    assert rates["below_lower"] == ("1.000000" if actual < lower else "0.000000")
    assert rates["below_median"] == ("1.000000" if actual < median else "0.000000")
    assert rates["above_upper"] == ("1.000000" if actual > upper else "0.000000")
    assert rates["included"] == ("1.000000" if lower <= actual <= upper else "0.000000")
    assert (
        Decimal(rates["below_lower"]) + Decimal(rates["included"]) + Decimal(rates["above_upper"])
        == 1
    )


def test_output_never_contains_negative_zero() -> None:
    document = _document(
        [
            _listing(
                "L1",
                [
                    _anchor(
                        "validation",
                        "6m",
                        "2024-01-02",
                        "2024-07-01",
                        [_model("m", _scored("-0.0000001", *TRIPLET))],
                    )
                ],
            )
        ]
    )
    group = _by_key(summarize_quantile_calibration(document))[("validation", "6m", "m")]

    assert group["mean_signed_median_error"] == "0.000000"


# --- (c) unavailable observations ---------------------------------------------------------


def _unavailable_document() -> Document:
    return _document(
        [
            _listing(
                "L1",
                [
                    _anchor(
                        "validation",
                        "6m",
                        "2024-01-02",
                        "2024-07-01",
                        [
                            _model("m1", _scored("0.0500", *TRIPLET)),
                            _model("m2", _unavailable("prediction_triplet_unavailable")),
                        ],
                    ),
                    _anchor(
                        "validation",
                        "6m",
                        "2024-07-01",
                        "2024-12-31",
                        [
                            _model("m1", _unavailable("stock_outcome_endpoint_missing")),
                            _model("m2", _unavailable("prediction_triplet_unavailable")),
                        ],
                    ),
                ],
            ),
            _listing(
                "L2",
                [
                    _anchor(
                        "validation",
                        "6m",
                        "2024-01-02",
                        "2024-07-01",
                        [
                            _model("m1", _unavailable("stock_outcome_endpoint_missing")),
                            _model("m2", _unavailable("prediction_triplet_unavailable")),
                        ],
                    )
                ],
            ),
        ]
    )


def test_unavailable_observations_are_counted_by_reason_and_never_scored() -> None:
    groups = _by_key(summarize_quantile_calibration(_unavailable_document()))

    partial = groups[("validation", "6m", "m1")]
    assert partial["observation_count"] == 1
    assert partial["unavailable_count"] == 2
    assert partial["unavailable_reasons"] == {"stock_outcome_endpoint_missing": 2}
    # Replay semantics: every observed cohort/listing counts, but only scored cohorts average.
    assert partial["target_cohort_count"] == 2
    assert partial["scored_cohort_count"] == 1
    assert partial["distinct_listing_count"] == 2
    assert partial["non_overlapping_target_count"] == 2
    # The unavailable sibling is excluded, not averaged in as zero.
    assert partial["rates"] == {
        "below_lower": "0.000000",
        "below_median": "0.000000",
        "above_upper": "0.000000",
        "included": "1.000000",
    }
    assert partial["mean_signed_median_error"] == "0.050000"
    assert partial["included_cohort_min"] == partial["included_cohort_max"] == "1.000000"
    assert partial["insufficient_reason"] is None


def test_all_unavailable_group_has_null_rates_and_explicit_reason() -> None:
    group = _by_key(summarize_quantile_calibration(_unavailable_document()))[
        ("validation", "6m", "m2")
    ]

    assert group["observation_count"] == 0
    assert group["unavailable_count"] == 3
    assert group["unavailable_reasons"] == {"prediction_triplet_unavailable": 3}
    assert group["target_cohort_count"] == 2
    assert group["scored_cohort_count"] == 0
    assert group["rates"] == {
        "below_lower": None,
        "below_median": None,
        "above_upper": None,
        "included": None,
    }
    assert group["deviation_from_nominal"] == group["rates"]
    assert group["mean_signed_median_error"] is None
    assert group["included_cohort_min"] is None
    assert group["included_cohort_max"] is None
    assert group["insufficient_reason"] == "no_available_cohort_metrics"


def test_empty_partitions_are_retained_with_explicit_reason() -> None:
    ledger = summarize_quantile_calibration(_unavailable_document())
    groups = _by_key(ledger)

    assert len(ledger["groups"]) == 3 * 1 * 2
    for partition in ("development", "final_holdout"):
        for model in ("m1", "m2"):
            empty = groups[(partition, "6m", model)]
            assert empty["insufficient_reason"] == "no_eligible_observations"
            assert empty["observation_count"] == 0
            assert empty["unavailable_count"] == 0
            assert empty["unavailable_reasons"] == {}
            assert empty["target_cohort_count"] == 0
            assert empty["distinct_listing_count"] == 0
            assert empty["non_overlapping_target_count"] == 0
            assert set(empty["rates"].values()) == {None}
            assert empty["mean_signed_median_error"] is None


def test_same_model_is_a_separate_group_per_partition_and_horizon() -> None:
    document = _document(
        [
            _listing(
                "L1",
                [
                    _anchor(
                        partition,
                        horizon,
                        "2024-01-02",
                        "2024-07-01",
                        [_model("m", _scored("0.0000", *TRIPLET))],
                    )
                    for partition in ("development", "validation")
                    for horizon in ("6m", "12m")
                ],
            )
        ]
    )

    ledger = summarize_quantile_calibration(document)

    assert len(ledger["groups"]) == 3 * 2 * 1
    assert sum(group["observation_count"] for group in ledger["groups"]) == 4


# --- (d) fail-closed matrix ---------------------------------------------------------------


def _matrix_document() -> Document:
    return _document(
        [
            _listing(
                "L1",
                [
                    _anchor(
                        "validation",
                        "6m",
                        "2024-01-02",
                        "2024-07-01",
                        [
                            _model("m_scored", _scored("0.0500", *TRIPLET)),
                            _model("m_unavailable", _unavailable("prediction_triplet_unavailable")),
                        ],
                    )
                ],
            )
        ]
    )


ANCHOR = ("listings", 0, "anchors", 0)
SCORED = (*ANCHOR, "models", 0, "metrics")
UNAVAILABLE = (*ANCHOR, "models", 1, "metrics")
Mutation = Callable[[Document], None]


def _node(document: Document, path: tuple[Any, ...]) -> Any:
    node: Any = document
    for key in path:
        node = node[key]
    return node


def _set(path: tuple[Any, ...], value: Any) -> Mutation:
    def mutate(document: Document) -> None:
        _node(document, path[:-1])[path[-1]] = value

    return mutate


def _delete(path: tuple[Any, ...]) -> Mutation:
    def mutate(document: Document) -> None:
        del _node(document, path[:-1])[path[-1]]

    return mutate


def _huge_actual(document: Document) -> None:
    metrics = _node(document, SCORED)
    metrics["actual_return"] = "1E+30"
    metrics["interval_included"] = False


def _duplicate_anchor(document: Document) -> None:
    anchors = document["listings"][0]["anchors"]
    anchors.append(copy.deepcopy(anchors[0]))


def _duplicate_listing(document: Document) -> None:
    document["listings"].append(copy.deepcopy(document["listings"][0]))


def _unordered(lower: str, median: str, upper: str) -> Mutation:
    def mutate(document: Document) -> None:
        metrics = _node(document, SCORED)
        metrics["predicted_returns"] = {"lower": lower, "median": median, "upper": upper}
        metrics["interval_included"] = Decimal(lower) <= Decimal("0.0500") <= Decimal(upper)

    return mutate


def _no_models(document: Document) -> None:
    _node(document, ANCHOR)["models"] = []


FAIL_CLOSED_CASES: list[tuple[str, Mutation, str]] = [
    ("schema_wrong", _set(("schema",), "other@1"), f"schema must be {REPLAY_STUDY_SCHEMA}"),
    ("schema_missing", _delete(("schema",)), "document.schema is missing"),
    (
        "hash_wrong",
        _set(("config_identity", "effective_config_hash"), "0" * 64),
        "does not match the frozen product config",
    ),
    (
        "hash_not_text",
        _set(("config_identity", "effective_config_hash"), 7),
        "does not match the frozen product config",
    ),
    (
        "hash_missing",
        _delete(("config_identity", "effective_config_hash")),
        "config_identity.effective_config_hash is missing",
    ),
    (
        "config_identity_missing",
        _delete(("config_identity",)),
        "document.config_identity is missing",
    ),
    (
        "config_identity_not_object",
        _set(("config_identity",), []),
        "config_identity must be an object",
    ),
    ("source_run_missing", _delete(("source_run",)), "document.source_run is missing"),
    ("source_run_id_missing", _delete(("source_run", "id")), "source_run.id is missing"),
    ("source_run_id_empty", _set(("source_run", "id"), ""), "source_run.id must be a short"),
    ("source_run_id_number", _set(("source_run", "id"), 5), "source_run.id must be a short"),
    ("protocol_missing", _delete(("protocol_identity",)), "document.protocol_identity is missing"),
    (
        "evidence_label_missing",
        _delete(("protocol_identity", "evidence_label")),
        "protocol_identity.evidence_label is missing",
    ),
    ("listings_missing", _delete(("listings",)), "document.listings is missing"),
    ("listings_not_array", _set(("listings",), {}), "listings must be an array"),
    ("listings_empty", _set(("listings",), []), "listings must not be empty"),
    ("listing_not_object", _set(("listings", 0), "L1"), "listings[0] must be an object"),
    (
        "listing_id_missing",
        _delete(("listings", 0, "listing_id")),
        "listings[0].listing_id is missing",
    ),
    ("anchors_missing", _delete(("listings", 0, "anchors")), "listings[0].anchors is missing"),
    ("anchors_not_array", _set(("listings", 0, "anchors"), "x"), "anchors must be an array"),
    ("anchor_not_object", _set(ANCHOR, 5), "listings[0].anchors[0] must be an object"),
    ("partition_missing", _delete((*ANCHOR, "partition")), "anchors[0].partition is missing"),
    (
        "partition_unknown",
        _set((*ANCHOR, "partition"), "training"),
        "anchors[0].partition is not a known partition",
    ),
    (
        "partition_not_text",
        _set((*ANCHOR, "partition"), 3),
        "anchors[0].partition must be a short printable string",
    ),
    ("horizon_missing", _delete((*ANCHOR, "horizon")), "anchors[0].horizon is missing"),
    ("horizon_empty", _set((*ANCHOR, "horizon"), ""), "anchors[0].horizon must be a short"),
    ("anchor_date_missing", _delete((*ANCHOR, "anchor_date")), "anchors[0].anchor_date is missing"),
    (
        "anchor_date_invalid",
        _set((*ANCHOR, "anchor_date"), "2024-13-01"),
        "anchor_date must be an ISO calendar date string",
    ),
    (
        "anchor_date_noncanonical",
        _set((*ANCHOR, "anchor_date"), "20240102"),
        "anchor_date must be an ISO calendar date string",
    ),
    (
        "anchor_date_not_text",
        _set((*ANCHOR, "anchor_date"), 20240102),
        "anchor_date must be an ISO calendar date string",
    ),
    (
        "outcome_end_missing",
        _delete((*ANCHOR, "outcome_end_date")),
        "anchors[0].outcome_end_date is missing",
    ),
    (
        "outcome_end_before_anchor",
        _set((*ANCHOR, "outcome_end_date"), "2024-01-01"),
        "outcome_end_date must be after anchor_date",
    ),
    (
        "outcome_end_equals_anchor",
        _set((*ANCHOR, "outcome_end_date"), "2024-01-02"),
        "outcome_end_date must be after anchor_date",
    ),
    ("models_missing", _delete((*ANCHOR, "models")), "anchors[0].models is missing"),
    ("models_not_array", _set((*ANCHOR, "models"), {}), "anchors[0].models must be an array"),
    ("model_not_object", _set((*ANCHOR, "models", 0), "x"), "models[0] must be an object"),
    (
        "model_name_missing",
        _delete((*ANCHOR, "models", 0, "model_name")),
        "models[0].model_name is missing",
    ),
    (
        "model_name_multiline",
        _set((*ANCHOR, "models", 0, "model_name"), "a\nb"),
        "models[0].model_name must be a short printable string",
    ),
    ("metrics_missing", _delete((*ANCHOR, "models", 0, "metrics")), "models[0].metrics is missing"),
    ("metrics_not_object", _set(SCORED, []), "models[0].metrics must be an object"),
    ("actual_key_missing", _delete((*SCORED, "actual_return")), "metrics.actual_return is missing"),
    (
        "predicted_key_missing",
        _delete((*SCORED, "predicted_returns")),
        "metrics.predicted_returns is missing",
    ),
    (
        "included_key_missing",
        _delete((*SCORED, "interval_included")),
        "metrics.interval_included is missing",
    ),
    (
        "reason_key_missing",
        _delete((*SCORED, "unavailable_reason")),
        "metrics.unavailable_reason is missing",
    ),
    (
        "actual_float",
        _set((*SCORED, "actual_return"), 0.05),
        "actual_return must be a decimal string",
    ),
    ("actual_int", _set((*SCORED, "actual_return"), 5), "actual_return must be a decimal string"),
    (
        "actual_bool",
        _set((*SCORED, "actual_return"), True),
        "actual_return must be a decimal string",
    ),
    (
        "actual_unparsable",
        _set((*SCORED, "actual_return"), f"not-a-number-{LEAK_MARKER}"),
        "actual_return is not a parsable decimal",
    ),
    ("actual_empty", _set((*SCORED, "actual_return"), ""), "actual_return is not a parsable"),
    ("actual_nan", _set((*SCORED, "actual_return"), "NaN"), "actual_return must be a finite"),
    ("actual_snan", _set((*SCORED, "actual_return"), "sNaN"), "actual_return must be a finite"),
    (
        "actual_infinity",
        _set((*SCORED, "actual_return"), "Infinity"),
        "actual_return must be a finite",
    ),
    (
        "actual_negative_infinity",
        _set((*SCORED, "actual_return"), "-Infinity"),
        "actual_return must be a finite",
    ),
    (
        "actual_leading_space",
        _set((*SCORED, "actual_return"), " 0.0500"),
        "actual_return must be a decimal string",
    ),
    (
        "actual_trailing_space",
        _set((*SCORED, "actual_return"), "0.0500 "),
        "actual_return must be a decimal string",
    ),
    (
        "actual_underscore",
        _set((*SCORED, "actual_return"), "0_5"),
        "actual_return must be a decimal string",
    ),
    (
        "lower_nan",
        _set((*SCORED, "predicted_returns", "lower"), "NaN"),
        "predicted_returns.lower must be a finite",
    ),
    (
        "upper_infinity",
        _set((*SCORED, "predicted_returns", "upper"), "Infinity"),
        "predicted_returns.upper must be a finite",
    ),
    (
        "median_unparsable",
        _set((*SCORED, "predicted_returns", "median"), "abc"),
        "predicted_returns.median is not a parsable",
    ),
    (
        "predicted_not_object",
        _set((*SCORED, "predicted_returns"), []),
        "predicted_returns must be an object",
    ),
    (
        "predicted_median_missing",
        _delete((*SCORED, "predicted_returns", "median")),
        "predicted_returns.median is missing",
    ),
    (
        "predicted_lower_missing",
        _delete((*SCORED, "predicted_returns", "lower")),
        "predicted_returns.lower is missing",
    ),
    (
        "predicted_upper_missing",
        _delete((*SCORED, "predicted_returns", "upper")),
        "predicted_returns.upper is missing",
    ),
    (
        "triplet_lower_above_median",
        _unordered("0.0100", "0.0000", "0.1000"),
        "must satisfy lower <= median <= upper",
    ),
    (
        "triplet_median_above_upper",
        _unordered("-0.1000", "0.2000", "0.1000"),
        "must satisfy lower <= median <= upper",
    ),
    (
        "included_string",
        _set((*SCORED, "interval_included"), "true"),
        "interval_included is inconsistent",
    ),
    (
        "included_flipped",
        _set((*SCORED, "interval_included"), False),
        "interval_included is inconsistent",
    ),
    (
        "included_null_for_scored",
        _set((*SCORED, "interval_included"), None),
        "interval_included is inconsistent",
    ),
    (
        "reason_not_text",
        _set((*UNAVAILABLE, "unavailable_reason"), 5),
        "unavailable_reason must be a short printable string",
    ),
    (
        "reason_empty",
        _set((*UNAVAILABLE, "unavailable_reason"), ""),
        "unavailable_reason must be a short printable string",
    ),
    (
        "reason_multiline",
        _set((*UNAVAILABLE, "unavailable_reason"), "a\nb"),
        "unavailable_reason must be a short printable string",
    ),
    (
        "scored_without_actual",
        _set((*SCORED, "actual_return"), None),
        "without an unavailable_reason",
    ),
    (
        "scored_without_predicted",
        _set((*SCORED, "predicted_returns"), None),
        "without an unavailable_reason",
    ),
    (
        "scored_with_reason",
        _set((*SCORED, "unavailable_reason"), "stock_outcome_endpoint_missing"),
        "unavailable_reason but also scored values",
    ),
    (
        "unavailable_with_actual",
        _set((*UNAVAILABLE, "actual_return"), "0.0100"),
        "unavailable_reason but also scored values",
    ),
    (
        "unavailable_with_predicted",
        _set((*UNAVAILABLE, "predicted_returns"), {"lower": "0", "median": "0", "upper": "0"}),
        "unavailable_reason but also scored values",
    ),
    (
        "unavailable_with_included",
        _set((*UNAVAILABLE, "interval_included"), True),
        "unavailable_reason but also scored values",
    ),
    (
        "duplicate_model_in_anchor",
        _set((*ANCHOR, "models", 1, "model_name"), "m_scored"),
        "anchors[0].models[1] duplicates an earlier observation",
    ),
    ("duplicate_across_anchors", _duplicate_anchor, "anchors[1].models[0] duplicates"),
    (
        "duplicate_across_listings",
        _duplicate_listing,
        "listings[1].anchors[0].models[0] duplicates",
    ),
    ("no_observations", _no_models, "contains no model observations"),
    ("decimal_out_of_range", _huge_actual, "outside the supported range"),
    (
        "not_json_serializable",
        lambda document: document.__setitem__("extra", Decimal("1")),
        "must be JSON-serializable",
    ),
    (
        "mixed_key_types",
        lambda document: document.__setitem__("extra", {1: "a", "b": 2}),
        "must be JSON-serializable",
    ),
]


def test_matrix_base_document_is_valid_and_case_ids_are_unique() -> None:
    summarize_quantile_calibration(_matrix_document())
    assert len({case[0] for case in FAIL_CLOSED_CASES}) == len(FAIL_CLOSED_CASES)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [case[1:] for case in FAIL_CLOSED_CASES],
    ids=[case[0] for case in FAIL_CLOSED_CASES],
)
def test_fail_closed_matrix(mutation: Mutation, expected: str) -> None:
    document = _matrix_document()
    mutation(document)

    with pytest.raises(ValueError) as caught:
        summarize_quantile_calibration(document)

    assert expected in str(caught.value)
    assert LEAK_MARKER not in str(caught.value)


@pytest.mark.parametrize("value", [None, [], (), "text", 3, [_matrix_document()]])
def test_non_mapping_input_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="study document must be a JSON object"):
        summarize_quantile_calibration(value)


@pytest.mark.parametrize(
    "mutation",
    [
        _set(("schema",), LEAK_MARKER),
        _set(("config_identity", "effective_config_hash"), LEAK_MARKER),
        _set((*ANCHOR, "partition"), LEAK_MARKER),
        _set((*ANCHOR, "anchor_date"), LEAK_MARKER),
        _set((*SCORED, "actual_return"), LEAK_MARKER),
        _set((*SCORED, "predicted_returns", "lower"), LEAK_MARKER),
        _set((*SCORED, "interval_included"), LEAK_MARKER),
        _set((*UNAVAILABLE, "unavailable_reason"), 7),
    ],
)
def test_errors_never_echo_document_values(mutation: Mutation) -> None:
    document = _matrix_document()
    mutation(document)

    with pytest.raises(ValueError) as caught:
        summarize_quantile_calibration(document)

    assert LEAK_MARKER not in str(caught.value)


def test_same_model_may_repeat_across_partitions_horizons_and_anchor_dates() -> None:
    models = [_model("m", _scored("0.0000", *TRIPLET))]
    anchors = [
        _anchor("validation", "6m", "2024-01-02", "2024-07-01", copy.deepcopy(models)),
        _anchor("validation", "6m", "2024-01-03", "2024-07-02", copy.deepcopy(models)),
        _anchor("validation", "12m", "2024-01-02", "2024-12-31", copy.deepcopy(models)),
        _anchor("development", "6m", "2024-01-02", "2024-07-01", copy.deepcopy(models)),
    ]

    ledger = summarize_quantile_calibration(_document([_listing("L1", anchors)]))

    assert _by_key(ledger)[("validation", "6m", "m")]["observation_count"] == 2


# --- (e) determinism, ordering, hash, mutation --------------------------------------------


def _multi_document() -> Document:
    return _document(
        [
            _listing(
                listing_id,
                [
                    _anchor(
                        partition,
                        horizon,
                        anchor,
                        end,
                        [
                            _model("zeta", _scored(actual, *TRIPLET)),
                            _model("alpha", _unavailable("prediction_triplet_unavailable")),
                            _model("mid", _scored(actual, "-0.2000", "0.0100", "0.2000")),
                        ],
                    )
                    for partition, horizon, anchor, end, actual in (
                        ("development", "6m", "2023-01-03", "2023-07-03", "-0.3000"),
                        ("validation", "6m", "2024-01-02", "2024-07-01", "0.0300"),
                        ("validation", "12m", "2024-01-02", "2024-12-31", "0.1500"),
                        ("final_holdout", "6m", "2025-01-02", "2025-07-01", "-0.0100"),
                    )
                ],
            )
            for listing_id in ("L1", "L2", "L3")
        ]
    )


def test_output_is_byte_deterministic_and_survives_a_json_round_trip() -> None:
    document = _multi_document()
    first = render_quantile_calibration(
        summarize_quantile_calibration(document), output_format="json"
    )
    second = render_quantile_calibration(
        summarize_quantile_calibration(copy.deepcopy(document)), output_format="json"
    )
    round_tripped = render_quantile_calibration(
        summarize_quantile_calibration(json.loads(json.dumps(document))), output_format="json"
    )

    assert first == second == round_tripped
    assert first.isascii()


def test_everything_but_the_input_hash_is_invariant_to_array_order() -> None:
    """Guaranteed: permuting listings, anchors, and models never changes groups or counts.

    Not guaranteed: ``input_sha256``, which hashes the exact input structure.
    """

    document = _multi_document()
    reordered = copy.deepcopy(document)
    reordered["listings"].reverse()
    for listing in reordered["listings"]:
        listing["anchors"].reverse()
        for anchor in listing["anchors"]:
            anchor["models"] = anchor["models"][1:] + anchor["models"][:1]
    base = summarize_quantile_calibration(document)
    permuted = summarize_quantile_calibration(reordered)

    assert permuted["groups"] == base["groups"]
    assert {k: v for k, v in permuted.items() if k != "input_sha256"} == {
        k: v for k, v in base.items() if k != "input_sha256"
    }
    assert permuted["input_sha256"] != base["input_sha256"]


def _leaves(value: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[Any, ...]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, (*path, index))
    else:
        yield path


def _perturb(value: Any) -> Any:
    if value is None:
        return "changed"
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    return f"{value}x"


def test_input_hash_changes_for_every_single_value_change_or_the_input_is_rejected() -> None:
    document = _multi_document()
    document["scope"] = {
        "studied_listing_count": 3,
        "all_selected": True,
        "source_run_listing_count": 3,
    }
    base_hash = summarize_quantile_calibration(document)["input_sha256"]
    extra_leaves = {("scope", "all_selected"), ("scope", "source_run_listing_count")}

    accepted: set[tuple[Any, ...]] = set()
    for path in _leaves(document):
        changed = copy.deepcopy(document)
        container = _node(changed, path[:-1])
        container[path[-1]] = _perturb(container[path[-1]])
        try:
            ledger = summarize_quantile_calibration(changed)
        except ValueError:
            continue
        accepted.add(path)
        assert ledger["input_sha256"] != base_hash, path

    assert extra_leaves <= accepted
    # Identity-bearing and free-text validated leaves stay valid and are hashed too.
    for path in (
        ("source_run", "id"),
        ("protocol_identity", "evidence_label"),
        ("listings", 0, "listing_id"),
        ("source_run", "snapshot_grade"),
        ("disclosures", 0),
        ("listings", 0, "source_execution", "evidence_grade"),
    ):
        assert path in accepted


def test_a_value_change_inside_the_same_bucket_changes_hash_and_error_only() -> None:
    document = _hand_document()
    base = summarize_quantile_calibration(document)
    changed = copy.deepcopy(document)
    changed["listings"][1]["anchors"][0]["models"][0]["metrics"]["actual_return"] = "0.0600"

    other = summarize_quantile_calibration(changed)

    assert other["input_sha256"] != base["input_sha256"]
    before = _by_key(base)[("validation", "6m", "m1")]
    after = _by_key(other)[("validation", "6m", "m1")]
    assert after["rates"] == before["rates"]
    assert after["mean_signed_median_error"] == "0.016667"
    assert before["mean_signed_median_error"] == "0.015000"


def test_input_is_not_mutated_and_output_does_not_alias_it() -> None:
    document = _multi_document()
    pristine = copy.deepcopy(document)

    ledger = summarize_quantile_calibration(document)
    assert document == pristine

    ledger["groups"][0]["rates"]["included"] = "tampered"
    ledger["groups"].clear()
    ledger["disclosures"].clear()
    assert document == pristine
    assert summarize_quantile_calibration(document)["disclosures"]


# --- rendering ----------------------------------------------------------------------------


def test_json_rendering_is_canonical() -> None:
    ledger = summarize_quantile_calibration(_hand_document())

    rendered = render_quantile_calibration(ledger, output_format="json")

    assert rendered == json.dumps(ledger, sort_keys=True, indent=2)
    assert json.loads(rendered) == ledger


def test_text_rendering_is_a_compact_ascii_table() -> None:
    ledger = summarize_quantile_calibration(_unavailable_document())

    text = render_quantile_calibration(ledger, output_format="text")

    assert text == render_quantile_calibration(ledger, output_format="text")
    assert text.isascii()
    assert text.splitlines()[0] == "research-product quantile calibration ledger"
    assert f"input_sha256={ledger['input_sha256']}" in text
    assert "nominal (frozen p20/p50/p80): below_lower=0.2 below_median=0.5" in text
    assert "no_available_cohort_metrics" in text
    assert "no_eligible_observations" in text
    assert "unavailable reasons (excluded from rates):" in text
    assert "validation 6m m1 stock_outcome_endpoint_missing=2" in text
    assert "n/a" in text

    hand_text = render_quantile_calibration(
        summarize_quantile_calibration(_hand_document()), output_format="text"
    )
    # Each group appears once in the support table and once in the calibration table.
    support, calibration = [
        fields
        for fields in (line.split() for line in hand_text.splitlines())
        if fields[:3] == ["validation", "6m", "m1"]
    ]
    assert support[3:] == ["4", "0", "2", "2", "3", "2", "2", "-"]
    assert calibration[3:] == [
        "0.166667",
        "0.666667",
        "0.166667",
        "0.666667",
        "-0.033333",
        "0.166667",
        "-0.033333",
        "0.066667",
        "0.015000",
        "0.333333",
        "1.000000",
    ]


def test_rendering_rejects_unknown_formats_and_non_ledgers() -> None:
    ledger = summarize_quantile_calibration(_hand_document())

    with pytest.raises(ValueError, match="output_format must be json or text"):
        render_quantile_calibration(ledger, output_format="yaml")
    with pytest.raises(ValueError, match="quantile calibration ledger"):
        render_quantile_calibration({"schema": "other"}, output_format="json")
    with pytest.raises(ValueError, match="quantile calibration ledger"):
        render_quantile_calibration([], output_format="text")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ledger is malformed"):
        render_quantile_calibration({"schema": CALIBRATION_LEDGER_SCHEMA}, output_format="text")


# --- (g) command, no database -------------------------------------------------------------


@pytest.fixture
def no_database(monkeypatch: pytest.MonkeyPatch) -> None:
    forbidden = Mock(side_effect=AssertionError("the calibration command must not touch the DB"))
    monkeypatch.setattr(
        "django.db.backends.base.base.BaseDatabaseWrapper.ensure_connection", forbidden
    )


def _write_study(tmp_path: Path, document: Document, name: str = "study.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _run(path: Path | str, output_format: str) -> str:
    output = StringIO()
    call_command(COMMAND, "--study-file", str(path), "--format", output_format, stdout=output)
    return output.getvalue()


@pytest.mark.parametrize("output_format", ["json", "text"])
def test_command_renders_the_ledger_without_touching_database_or_files(
    tmp_path: Path, no_database: None, output_format: str
) -> None:
    document = _multi_document()
    path = _write_study(tmp_path, document)
    original = path.read_bytes()
    listing_before = sorted(tmp_path.iterdir())

    rendered = _run(path, output_format)

    expected = render_quantile_calibration(
        summarize_quantile_calibration(document), output_format=output_format
    )
    assert rendered == expected + "\n"
    assert path.read_bytes() == original
    assert sorted(tmp_path.iterdir()) == listing_before


def test_command_requires_explicit_arguments(tmp_path: Path, no_database: None) -> None:
    path = _write_study(tmp_path, _multi_document())

    with pytest.raises(CommandError, match="--study-file"):
        call_command(COMMAND, "--format", "json")
    with pytest.raises(CommandError, match="--format"):
        call_command(COMMAND, "--study-file", str(path))
    with pytest.raises(CommandError, match="invalid choice"):
        call_command(COMMAND, "--study-file", str(path), "--format", "yaml")


def test_command_refuses_missing_symlink_and_directory_inputs(
    tmp_path: Path, no_database: None
) -> None:
    real = _write_study(tmp_path, _multi_document())
    link = tmp_path / "link.json"
    link.symlink_to(real)
    directory = tmp_path / "directory.json"
    directory.mkdir()

    with pytest.raises(CommandError, match="existing regular file"):
        _run(tmp_path / "absent.json", "json")
    with pytest.raises(CommandError, match="must not be a symbolic link"):
        _run(link, "json")
    with pytest.raises(CommandError, match="existing regular file"):
        _run(directory, "json")
    assert _run(real, "json")


def test_command_refuses_inputs_above_the_size_limit(
    tmp_path: Path, no_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stanstock.research.management.commands.summarize_price_product_calibration as command

    path = _write_study(tmp_path, _multi_document())
    size = path.stat().st_size
    monkeypatch.setattr(command, "MAX_STUDY_FILE_BYTES", size)
    assert _run(path, "json")
    monkeypatch.setattr(command, "MAX_STUDY_FILE_BYTES", size - 1)
    with pytest.raises(CommandError, match="exceeds the 64 MiB size limit"):
        _run(path, "json")
    assert command.MAX_STUDY_FILE_BYTES == size - 1
    monkeypatch.undo()
    assert command.MAX_STUDY_FILE_BYTES == 64 * 1024 * 1024


def test_command_refuses_malformed_and_wrong_schema_files_without_echoing_content(
    tmp_path: Path, no_database: None
) -> None:
    cases: dict[str, tuple[bytes, str]] = {
        "malformed": (f'{{"schema": "{LEAK_MARKER}"'.encode(), "not valid UTF-8 JSON"),
        "non_utf8": (b"\xff\xfe{}", "not valid UTF-8 JSON"),
        "duplicate_key": (
            f'{{"schema": "{LEAK_MARKER}", "schema": "x"}}'.encode(),
            "duplicate JSON object key",
        ),
        "not_object": (f'["{LEAK_MARKER}"]'.encode(), "study document must be a JSON object"),
        "wrong_schema": (
            json.dumps({"schema": LEAK_MARKER}).encode(),
            f"schema must be {REPLAY_STUDY_SCHEMA}",
        ),
        "wrong_hash": (
            json.dumps(
                {
                    **_multi_document(),
                    "config_identity": {"effective_config_hash": LEAK_MARKER},
                }
            ).encode(),
            "does not match the frozen product config",
        ),
        "too_deep": (b"[" * 100_000 + b"]" * 100_000, "not valid UTF-8 JSON"),
    }
    for name, (content, expected) in cases.items():
        path = tmp_path / f"{name}.json"
        path.write_bytes(content)

        with pytest.raises(CommandError) as caught:
            _run(path, "json")

        assert expected in str(caught.value), name
        assert LEAK_MARKER not in str(caught.value), name


# --- (e2) integrity findings: grid, provenance, cross-checks, guide ------------------------


def _six_month_listings(mode: str = "provider") -> list[dict[str, Any]]:
    return [
        _listing(
            "L1",
            [
                _anchor(
                    "validation",
                    "6m",
                    "2024-01-02",
                    "2024-07-01",
                    [_model("m1", _scored("0.0500", *TRIPLET))],
                )
            ],
            mode,
        )
    ]


@pytest.mark.parametrize(
    "hostile",
    ["bad\x1b[31mred", "line\nbreak", "x" * 100_000],
)
def test_hostile_disclosure_is_rejected(hostile: str) -> None:
    document = _hand_document()
    document["disclosures"] = ["fine", hostile]
    with pytest.raises(ValueError, match="disclosures"):
        summarize_quantile_calibration(document)


def test_too_many_disclosures_are_rejected() -> None:
    document = _hand_document()
    document["disclosures"] = ["ok"] * 51
    with pytest.raises(ValueError, match="disclosures"):
        summarize_quantile_calibration(document)


def test_long_real_sized_disclosure_is_accepted() -> None:
    document = _hand_document()
    document["disclosures"] = ["y" * 900]
    assert summarize_quantile_calibration(document)["source_disclosures"] == ["y" * 900]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(studied_listing_ids=[float("nan")]),
        lambda s: s.update(source_run_listing_count=float("inf")),
        lambda s: s.update(unexpected_key=1),
        lambda s: s.update(requested_listing_ids=[{"a": 1}]),
        lambda s: s.update(requested_listing_ids=["a\x1bb"]),
    ],
)
def test_scope_rejects_non_finite_unknown_and_non_scalar_values(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    document = _hand_document()
    mutate(document["scope"])
    with pytest.raises(ValueError, match="scope"):
        summarize_quantile_calibration(document)


def test_scope_accepts_real_serializer_shape() -> None:
    document = _hand_document()
    document["scope"].update(
        requested_listing_ids=["a", "b"],
        studied_listing_ids=["a", "b"],
        source_run_listing_count=2,
    )
    assert summarize_quantile_calibration(document)["scope"]["all_selected"] is True


def test_command_rejects_non_finite_json_constants(tmp_path: Path, no_database: None) -> None:
    text = json.dumps(_multi_document()).replace('"all_selected": true', '"all_selected": NaN')
    path = tmp_path / "nan.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(CommandError, match="not valid UTF-8 JSON"):
        _run(path, "json")


def _expect_error(document: Document, expected: str) -> None:
    with pytest.raises(ValueError) as caught:
        summarize_quantile_calibration(document)
    assert expected in str(caught.value)
    assert LEAK_MARKER not in str(caught.value)


def test_grid_comes_from_projection_aggregates_and_keeps_horizons_without_anchors() -> None:
    listings = _six_month_listings()
    ledger = summarize_quantile_calibration(
        _document(listings, _aggregates(listings, ("12m", "3y", "5y")))
    )

    assert len(ledger["groups"]) == 3 * 4 * 1
    assert [g["horizon"] for g in ledger["groups"][:4]] == ["6m", "12m", "3y", "5y"]
    for group in ledger["groups"]:
        if (group["partition"], group["horizon"]) == ("validation", "6m"):
            assert group["observation_count"] == 1
        else:
            assert group["insufficient_reason"] == "no_eligible_observations"
            assert group["scored_cohort_count"] == 0
            assert set(group["rates"].values()) == {None}


def test_observation_outside_the_aggregate_grid_fails_closed() -> None:
    listings = _six_month_listings()
    aggregates = _aggregates(listings)
    listings[0]["anchors"].append(
        _anchor(
            "validation",
            "12m",
            "2024-01-02",
            "2024-12-31",
            [_model("m1", _scored("0.0", *TRIPLET))],
        )
    )
    _expect_error(_document(listings, aggregates), "outside the projection_aggregates grid")
    for bad in ({"partition": "other"}, {"horizon": 5}):
        broken = _document(_six_month_listings())
        broken["projection_aggregates"][0].update(bad)
        with pytest.raises(ValueError):
            summarize_quantile_calibration(broken)


def test_incomplete_or_duplicate_grid_fails_closed() -> None:
    document = _document(_six_month_listings())
    document["projection_aggregates"].pop()
    _expect_error(document, "full partition x horizon x model grid")
    document = _document(_six_month_listings())
    document["projection_aggregates"].append(copy.deepcopy(document["projection_aggregates"][0]))
    _expect_error(document, "duplicates an earlier aggregate")
    document = _document(_six_month_listings())
    document["projection_aggregates"] = []
    _expect_error(document, "full partition x horizon x model grid")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_cohort_count", 9),
        ("distinct_listing_count", 9),
        ("non_overlapping_target_count", 9),
        ("unavailable_count", 9),
        ("unavailable_reasons", {"x": 1}),
        ("insufficient_reason", "no_eligible_observations"),
    ],
)
def test_mutated_aggregate_fields_fail_closed(field: str, value: Any) -> None:
    document = _document(_six_month_listings())
    target = next(
        item
        for item in document["projection_aggregates"]
        if (item["partition"], item["horizon"]) == ("validation", "6m")
    )
    target[field] = value
    _expect_error(document, f"{field} does not match the listings")


@pytest.mark.parametrize("value", ["0.500000", None])
def test_mutated_interval_inclusion_fails_closed(value: Any) -> None:
    document = _document(_six_month_listings())
    target = next(
        item
        for item in document["projection_aggregates"]
        if (item["partition"], item["horizon"]) == ("validation", "6m")
    )
    target["averages"]["interval_inclusion"] = value
    _expect_error(document, "interval_inclusion does not match the listings")


def test_truncated_listings_fail_closed() -> None:
    document = _hand_document()
    document["listings"].pop()
    document["scope"]["studied_listing_count"] = len(document["listings"])
    _expect_error(document, "does not match the listings")
    document = _hand_document()
    document["listings"].pop()
    _expect_error(document, "scope.studied_listing_count does not match listings")


def test_differing_actual_return_across_models_fails_closed() -> None:
    models = [_model("a", _scored("0.0500", *TRIPLET)), _model("b", _scored("0.0600", *TRIPLET))]
    listings = [_listing("L1", [_anchor("validation", "6m", "2024-01-02", "2024-07-01", models)])]
    _expect_error(_document(listings), "actual_return differs across models")


@pytest.mark.parametrize(
    ("modes", "expected_modes", "label"),
    [
        (["provider"], ["provider"], "research_grade_retrospective_descriptive"),
        (["synthetic_demo"], ["synthetic_demo"], "synthetic_demo_descriptive_not_empirical"),
        (
            ["synthetic_demo", "provider", "provider"],
            ["provider", "synthetic_demo"],
            "synthetic_demo_descriptive_not_empirical",
        ),
    ],
)
def test_provenance_modes_and_evidence_label(
    modes: list[str], expected_modes: list[str], label: str
) -> None:
    listings = [
        _listing(
            f"L{index}",
            [
                _anchor(
                    "validation",
                    "6m",
                    "2024-01-02",
                    "2024-07-01",
                    [_model("m1", _scored("0.05", *TRIPLET))],
                )
            ],
            mode,
        )
        for index, mode in enumerate(modes)
    ]
    ledger = summarize_quantile_calibration(_document(listings))

    assert ledger["source_execution_modes"] == expected_modes
    assert ledger["evidence_label"] == label
    text = render_quantile_calibration(ledger, output_format="text")
    assert f"evidence_label={label}" in text
    assert f"source_execution_modes={','.join(expected_modes)}" in text
    assert "snapshot_grade=research issued_on_time=True" in text
    assert "source_disclosure: source disclosure one" in text


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["listings"][0].pop("source_execution"),
        lambda d: d["listings"][0]["source_execution"].pop("mode"),
        lambda d: d["listings"][0]["source_execution"].update(mode="other"),
        lambda d: d["source_run"].pop("snapshot_grade"),
        lambda d: d["source_run"].pop("issued_on_time"),
        lambda d: d["source_run"].update(issued_on_time="yes"),
        lambda d: d.pop("disclosures"),
        lambda d: d.update(disclosures=[]),
        lambda d: d.pop("scope"),
    ],
)
def test_missing_or_invalid_provenance_fails_closed(mutation: Mutation) -> None:
    document = _document(_six_month_listings())
    mutation(document)
    with pytest.raises(ValueError):
        summarize_quantile_calibration(document)


def test_scored_cohort_support_differs_from_replay_cohort_support() -> None:
    anchors = [
        _anchor("validation", "6m", "2020-01-02", "2020-07-01", [_model("m1", _unavailable("r"))]),
        _anchor("validation", "6m", "2020-07-01", "2020-12-31", [_model("m1", _unavailable("r"))]),
        _anchor(
            "validation",
            "6m",
            "2021-01-04",
            "2021-07-01",
            [_model("m1", _scored("0.05", *TRIPLET))],
        ),
    ]
    ledger = summarize_quantile_calibration(_document([_listing("L1", anchors)]))
    group = _by_key(ledger)[("validation", "6m", "m1")]

    assert group["target_cohort_count"] == 3
    assert group["non_overlapping_target_count"] == 3
    assert group["scored_cohort_count"] == 1
    assert group["scored_non_overlapping_target_count"] == 1
    text = render_quantile_calibration(ledger, output_format="text")
    assert "scored_cohorts" in text


def test_nominal_rates_match_the_frozen_config_quantiles() -> None:
    from stanstock.research.price_product_calibration import _NOMINAL
    from stanstock.research.price_product_config import load_price_product_config

    lower, median, upper = (
        Decimal(str(q)) for q in load_price_product_config().simulation.quantiles
    )
    assert (lower, median, upper) == (Decimal("0.2"), Decimal("0.5"), Decimal("0.8"))
    assert {name: Decimal(value) for name, value in _NOMINAL.items()} == {
        "below_lower": lower,
        "below_median": median,
        "above_upper": 1 - upper,
        "included": upper - lower,
    }


def test_calibrated_right_skewed_forecast_has_nominal_rates_but_nonzero_mean_error() -> None:
    """Equally spaced inverse-CDF sample from the forecast's own lognormal law."""

    normal = NormalDist()
    sigma, count = 0.5, 1000

    def quantile(p: float) -> str:
        return f"{math.exp(sigma * normal.inv_cdf(p)) - 1:.6f}"

    triplet = (quantile(0.2), quantile(0.5), quantile(0.8))
    listings = [
        _listing(
            f"L{index}",
            [
                _anchor(
                    "validation",
                    "6m",
                    "2024-01-02",
                    "2024-07-01",
                    [_model("m1", _scored(quantile((index + 0.5) / count), *triplet))],
                )
            ],
        )
        for index in range(count)
    ]
    ledger = summarize_quantile_calibration(_document(listings))
    group = _by_key(ledger)[("validation", "6m", "m1")]

    for name, deviation in group["deviation_from_nominal"].items():
        assert abs(Decimal(deviation)) <= Decimal("0.005"), name
    assert Decimal(group["mean_signed_median_error"]) > Decimal("0.1")
    text = " ".join(ledger["disclosures"])
    assert "expected for right-skewed returns even when the forecast is calibrated" in text


# --- (f) end-to-end differential against the real replay ----------------------------------


@pytest.fixture
def synthetic_store(settings: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = True
    settings.DATA_DIR = tmp_path / "data"
    blocked = Mock(
        side_effect=AssertionError("The ledger flow must not use credentials or network")
    )
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("stanstock.data.providers.twelve_data.resolve_api_key", blocked)
    yield AssetStore(tmp_path / "data")
    blocked.assert_not_called()


@pytest.fixture
def real_study(synthetic_store: AssetStore) -> SimpleNamespace:
    job = execute_demo_product_refresh(store=synthetic_store)
    run = AnalysisRun.objects.select_related("universe_snapshot").get(
        pk=job.details["analysis_run_id"]
    )
    report = study_price_product_run(
        run=run,
        store=synthetic_store,
        all_selected=True,
        report_generated_at=datetime(2026, 9, 13, 18, tzinfo=UTC),
    )
    return SimpleNamespace(
        run=run,
        store=synthetic_store,
        report=report,
        document=serialize_price_product_study(report, include_generated_at=False),
    )


def _observations(document: Document) -> Iterator[tuple[str, str, str, str, dict[str, Any]]]:
    for listing in document["listings"]:
        for anchor in listing["anchors"]:
            for model in anchor["models"]:
                yield (
                    anchor["partition"],
                    anchor["horizon"],
                    model["model_name"],
                    anchor["outcome_end_date"],
                    model["metrics"],
                )


def _exact_six_places(value: Fraction) -> str:
    return format(Decimal(round(value * 1_000_000)).scaleb(-6), ".6f")


def _fraction_oracle(
    document: Document,
) -> dict[tuple[str, str, str], dict[str, str]]:
    """Independent cohort-first recomputation with exact rationals (no Decimal context)."""

    cohorts: dict[tuple[str, str, str], dict[str, list[dict[str, Fraction]]]] = {}
    for partition, horizon, model, target, metrics in _observations(document):
        if metrics["unavailable_reason"] is not None:
            continue
        predicted = {name: Fraction(value) for name, value in metrics["predicted_returns"].items()}
        actual = Fraction(metrics["actual_return"])
        cohorts.setdefault((partition, horizon, model), {}).setdefault(target, []).append(
            {
                "below_lower": Fraction(int(actual < predicted["lower"])),
                "below_median": Fraction(int(actual < predicted["median"])),
                "above_upper": Fraction(int(actual > predicted["upper"])),
                "included": Fraction(int(predicted["lower"] <= actual <= predicted["upper"])),
                "error": actual - predicted["median"],
            }
        )
    expected: dict[tuple[str, str, str], dict[str, str]] = {}
    for key, by_target in cohorts.items():
        cohort_means = [
            {
                name: sum((item[name] for item in items), Fraction(0)) / len(items)
                for name in ("below_lower", "below_median", "above_upper", "included", "error")
            }
            for items in by_target.values()
        ]
        expected[key] = {
            name: _exact_six_places(
                sum((mean[name] for mean in cohort_means), Fraction(0)) / len(cohort_means)
            )
            for name in ("below_lower", "below_median", "above_upper", "included", "error")
        }
    return expected


@pytest.mark.django_db
def test_ledger_matches_the_real_replay_aggregates_end_to_end(real_study: SimpleNamespace) -> None:
    document = real_study.document
    ledger = summarize_quantile_calibration(document)
    groups = _by_key(ledger)
    aggregates = {
        (item["partition"], item["horizon"], item["model_name"]): item
        for item in document["projection_aggregates"]
    }

    assert document["schema"] == REPLAY_STUDY_SCHEMA
    assert document["config_identity"]["effective_config_hash"] == PRODUCT_EFFECTIVE_CONFIG_HASH
    assert ledger["source_run_id"] == str(real_study.run.id)
    assert set(groups) == set(aggregates)
    assert len(ledger["groups"]) == len(groups) == 36
    for horizon in ("3y", "5y"):
        for key, group in groups.items():
            if key[1] == horizon:
                assert group["insufficient_reason"] == "no_eligible_observations", key
                assert group["observation_count"] == 0, key
    assert ledger["source_execution_modes"] == ["synthetic_demo"]
    assert ledger["evidence_label"] == "synthetic_demo_descriptive_not_empirical"
    assert ledger["source_disclosures"] == document["disclosures"]
    assert ledger["scope"] == json.loads(json.dumps(document["scope"]))
    for key, aggregate in aggregates.items():
        group = groups[key]
        assert group["target_cohort_count"] == aggregate["target_cohort_count"], key
        assert group["distinct_listing_count"] == aggregate["distinct_listing_count"], key
        assert group["non_overlapping_target_count"] == aggregate["non_overlapping_target_count"], (
            key
        )
        assert group["unavailable_count"] == aggregate["unavailable_count"], key
        assert group["unavailable_reasons"] == aggregate["unavailable_reasons"], key
        assert group["insufficient_reason"] == aggregate["insufficient_reason"], key
        replay_inclusion = aggregate["averages"]["interval_inclusion"]
        if replay_inclusion is None:
            assert group["rates"]["included"] is None, key
        else:
            assert Decimal(group["rates"]["included"]) == Decimal(replay_inclusion).quantize(
                SIX_PLACES, rounding=ROUND_HALF_EVEN
            ), key

    # The comparison must exercise real scored, unavailable, and empty groups.
    assert any(group["observation_count"] > 0 for group in ledger["groups"])
    assert any(group["unavailable_count"] > 0 for group in ledger["groups"])
    assert any(
        group["insufficient_reason"] == "no_eligible_observations" for group in ledger["groups"]
    )
    assert sum(group["observation_count"] for group in ledger["groups"]) == sum(
        1 for *_rest, metrics in _observations(document) if metrics["unavailable_reason"] is None
    )

    # Independent exact-rational recomputation of every per-quantile rate and the error mean.
    oracle = _fraction_oracle(document)
    scored = {key for key, group in groups.items() if group["observation_count"] > 0}
    assert scored == set(oracle)
    for key, expected in oracle.items():
        group = groups[key]
        assert group["rates"] == {
            name: expected[name]
            for name in ("below_lower", "below_median", "above_upper", "included")
        }, key
        assert group["mean_signed_median_error"] == expected["error"], key
        for name, nominal in group["nominal"].items():
            assert Decimal(group["deviation_from_nominal"][name]) == Decimal(
                group["rates"][name]
            ) - Decimal(nominal)

    # Survives the exact JSON text the replay command prints and is hash-bound to it.
    emitted = json.loads(json.dumps(document, sort_keys=True))
    assert summarize_quantile_calibration(emitted) == ledger
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert ledger["input_sha256"] == hashlib.sha256(canonical.encode("ascii")).hexdigest()


@pytest.mark.django_db
def test_command_over_a_real_replay_document_makes_no_database_or_asset_writes(
    real_study: SimpleNamespace, tmp_path: Path, django_assert_num_queries: Any
) -> None:
    emitted = serialize_price_product_study(real_study.report, include_generated_at=True)
    path = _write_study(tmp_path, emitted)
    counts_before = {
        "runs": AnalysisRun.objects.count(),
        "analyses": StockAnalysis.objects.count(),
        "predictions": Prediction.objects.count(),
        "assets": DataAsset.objects.count(),
    }
    assets_before = sorted(str(item) for item in real_study.store.root.rglob("*"))

    with django_assert_num_queries(0):
        rendered_json = _run(path, "json")
        rendered_text = _run(path, "text")

    expected = summarize_quantile_calibration(emitted)
    assert json.loads(rendered_json) == expected
    assert render_quantile_calibration(expected, output_format="text") + "\n" == rendered_text
    assert counts_before == {
        "runs": AnalysisRun.objects.count(),
        "analyses": StockAnalysis.objects.count(),
        "predictions": Prediction.objects.count(),
        "assets": DataAsset.objects.count(),
    }
    assert assets_before == sorted(str(item) for item in real_study.store.root.rglob("*"))


# --- (h) architecture ---------------------------------------------------------------------

CALIBRATION_MODULE = SOURCE_ROOT / "research" / "price_product_calibration.py"
COMMAND_MODULE = (
    SOURCE_ROOT / "research" / "management" / "commands" / "summarize_price_product_calibration.py"
)


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_calibration_module_is_pure_and_has_no_provider_orm_or_io_imports() -> None:
    allowed = {
        "__future__",
        "collections",
        "collections.abc",
        "dataclasses",
        "datetime",
        "decimal",
        "hashlib",
        "json",
        "math",
        "re",
        "typing",
        "stanstock.research.price_product_config",
        "stanstock.research.price_product_study",
    }
    imports = _imports(CALIBRATION_MODULE)

    assert imports <= allowed, sorted(imports - allowed)
    assert not [m for m in imports if m.startswith(("django", "stanstock.data"))]
    assert not [m for m in imports if m.endswith(".models") or m == "stanstock.research.models"]
    clock_or_random = [
        node.attr
        for node in ast.walk(ast.parse(CALIBRATION_MODULE.read_text(encoding="utf-8")))
        if isinstance(node, ast.Attribute) and node.attr in {"now", "today", "utcnow", "random"}
    ]
    assert clock_or_random == []


def test_command_module_reads_one_file_and_imports_no_orm_asset_or_provider_code() -> None:
    allowed = {
        "__future__",
        "argparse",
        "django.core.management.base",
        "json",
        "os",
        "pathlib",
        "stat",
        "typing",
        "stanstock.research.price_product_calibration",
    }
    imports = _imports(COMMAND_MODULE)

    assert imports <= allowed, sorted(imports - allowed)
    source = COMMAND_MODULE.read_text(encoding="utf-8")
    for forbidden in ("O_WRONLY", "O_RDWR", "O_CREAT", "write_text", "write_bytes", "unlink"):
        assert forbidden not in source

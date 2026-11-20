from __future__ import annotations

import copy
import json
import os
import pickle
import subprocess
import sys
from datetime import UTC, date, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.utils import timezone

from interval_calibration_documents import build_interval_calibration_document
from stanstock.data.assets import AssetStore, read_checksummed_bytes
from stanstock.data.research_product_demo import execute_demo_product_refresh
from stanstock.research import price_product_calibration as s1
from stanstock.research.models import AnalysisRun
from stanstock.research.price_product import LedgerTriplet
from stanstock.research.price_product_config import PRODUCT_VERSION, load_price_product_config
from stanstock.research.price_product_interval_calibration import (
    _CONTEXT,
    CalibrationObservation,
    _walk_forward_folds,
    adjust_interval,
    canonical_json_bytes,
    canonical_sha256,
    embargo_eligible,
    fit_interval_calibration,
)
from stanstock.research.price_product_interval_calibration_document import (
    calibration_inputs_from_document,
    fit_interval_calibration_document,
)
from stanstock.research.product_study_evidence import register_price_product_study
from test_research_price_product_calibration import _aggregates

COMMAND = "fit_price_product_interval_calibration"
SRC = Path(__file__).parents[1] / "src"


def _fit(document: dict[str, Any]) -> dict[str, Any]:
    return fit_interval_calibration_document(document)


def _rebuild_aggregates(document: dict[str, Any]) -> None:
    document["projection_aggregates"] = _aggregates(document["listings"])


def _exact_decimal_text(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _row_payload_for_hash(row: CalibrationObservation) -> dict[str, Any]:
    return {
        "horizon": row.horizon,
        "listing_id": row.listing_id,
        "anchor_date": row.anchor_date.isoformat(),
        "outcome_end_date": row.outcome_end_date.isoformat(),
        "lower": (_exact_decimal_text(row.predicted.lower) if row.predicted is not None else None),
        "median": (
            _exact_decimal_text(row.predicted.median) if row.predicted is not None else None
        ),
        "upper": (_exact_decimal_text(row.predicted.upper) if row.predicted is not None else None),
        "actual_return": (
            _exact_decimal_text(row.actual_return) if row.actual_return is not None else None
        ),
        "unavailable_reason": row.unavailable_reason,
    }


@pytest.fixture
def registered_demo_study(
    settings: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[AssetStore, dict[str, Any]]:
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = True
    settings.DATA_DIR = tmp_path / "assets"
    monkeypatch.setattr(
        timezone,
        "now",
        lambda: datetime(2026, 9, 13, 17, tzinfo=UTC),
    )
    blocked = Mock(side_effect=AssertionError("synthetic research must not use a provider"))
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("stanstock.data.providers.twelve_data.resolve_api_key", blocked)
    store = AssetStore(settings.DATA_DIR)
    execute_demo_product_refresh(store=store)
    run = AnalysisRun.objects.select_related("universe_snapshot").get(
        config_version=PRODUCT_VERSION
    )
    asset = register_price_product_study(run=run, store=store)
    document = json.loads(read_checksummed_bytes(store, asset))
    blocked.assert_not_called()
    return store, document


def test_positive_and_negative_offsets_and_independent_fraction_oracle() -> None:
    document = build_interval_calibration_document()
    fitted = _fit(document)
    params = fitted["params"]
    six = params["horizons"]["6m"]
    assert six["state"] == "usable"
    assert Decimal(six["offset"]) > 0

    inputs = calibration_inputs_from_document(document)
    rows = [
        row
        for row in inputs.observations
        if row.partition == "development" and row.horizon == "6m" and row.unavailable_reason is None
    ]
    clusters = {row.outcome_end_date for row in rows}
    weighted_scores: list[tuple[Decimal, Fraction]] = []
    for row in rows:
        assert row.predicted is not None and row.actual_return is not None
        score = max(
            (Decimal(1) + row.predicted.lower).ln() - (Decimal(1) + row.actual_return).ln(),
            (Decimal(1) + row.actual_return).ln() - (Decimal(1) + row.predicted.upper).ln(),
        )
        cluster_size = sum(other.outcome_end_date == row.outcome_end_date for other in rows)
        weighted_scores.append((score, Fraction(1, len(clusters) * cluster_size)))
    cumulative = Fraction(0)
    oracle = None
    level = Fraction(3, 5) * Fraction(len(clusters) + 1, len(clusters))
    for score, weight in sorted(weighted_scores):
        cumulative += weight
        if cumulative >= level:
            oracle = score.quantize(Decimal("1E-10"), rounding="ROUND_CEILING")
            break
    assert six["offset"] == format(oracle, "f")

    narrower = build_interval_calibration_document(lower="-0.0001", upper="0.0001")
    assert Decimal(_fit(narrower)["params"]["horizons"]["6m"]["offset"]) > 0
    wider = build_interval_calibration_document(lower="-0.2000", upper="0.2000")
    assert Decimal(_fit(wider)["params"]["horizons"]["6m"]["offset"]) < 0


def test_exact_zero_offset_is_usable_and_serialized_without_negative_zero() -> None:
    document = build_interval_calibration_document(actual_return="0.0100", upper="0.0100")
    six = _fit(document)["params"]["horizons"]["6m"]
    assert six["state"] == "usable"
    assert six["offset"] == "0"


def test_validation_scoring_walk_forward_and_frozen_grid() -> None:
    document = build_interval_calibration_document()
    fit = _fit(document)
    six = fit["diagnostics"]["horizons"]["6m"]
    twelve = fit["diagnostics"]["horizons"]["12m"]
    inputs = calibration_inputs_from_document(document)
    validation_dates = {
        row.anchor_date for row in inputs.observations if row.partition == "validation"
    }
    assert validation_dates and all(item.year == 2024 for item in validation_dates)
    assert all(row.partition != "final_holdout" for row in inputs.observations)
    assert len(six["folds"]) == 1
    assert six["folds"][0]["fold_training_equals_params"] is True
    assert six["paired"]["paired_observation_count"] > 0
    assert len(twelve["folds"]) == 0
    assert twelve["paired"] is None
    assert twelve["calibrated_observations"] == 0
    assert fit["params"]["training_cutoff"] == fit["params"]["horizons"]["6m"]["training_cutoff"]
    params = fit["params"]
    diagnostics = fit["diagnostics"]
    assert params["calendar_sessions_sha256"] == canonical_sha256(
        [session.isoformat() for session in inputs.calendar_sessions]
    )
    assert params["params_sha256"] == canonical_sha256(
        {key: value for key, value in params.items() if key != "params_sha256"}
    )
    development_rows = sorted(
        (row for row in inputs.observations if row.partition == "development"),
        key=lambda row: (
            {"6m": 126, "12m": 252, "3y": 756, "5y": 1260}[row.horizon],
            row.anchor_date,
            row.outcome_end_date,
            row.listing_id,
        ),
    )
    assert params["fit_input_sha256"] == canonical_sha256(
        {
            "method_version": params["method"]["method_version"],
            "method": params["method"],
            "config_identity": inputs.config_identity,
            "protocol_identity": inputs.protocol_identity,
            "calendar_identity": inputs.calendar_identity,
            "calendar_sessions_sha256": params["calendar_sessions_sha256"],
            "observations": [_row_payload_for_hash(row) for row in development_rows],
        }
    )
    assert diagnostics["diagnostics_sha256"] == canonical_sha256(
        {key: value for key, value in diagnostics.items() if key != "diagnostics_sha256"}
    )
    assert diagnostics["evaluation_input_sha256"] == canonical_sha256(
        {
            "params_sha256": params["params_sha256"],
            "observations": [
                _row_payload_for_hash(row)
                for row in sorted(
                    (row for row in inputs.observations if row.partition == "validation"),
                    key=lambda row: (
                        {"6m": 126, "12m": 252, "3y": 756, "5y": 1260}[row.horizon],
                        row.anchor_date,
                        row.outcome_end_date,
                        row.listing_id,
                    ),
                )
            ],
        }
    )


def test_synthetic_validation_coverage_and_interval_score_direction() -> None:
    actuals = ("0.0000",) * 18 + ("-0.0600",) * 6 + ("0.0200",) * 6
    document = build_interval_calibration_document(
        actual_return="0.0000",
        lower="-0.2000",
        upper="0.2000",
        validation_actual_returns=actuals,
    )
    six = _fit(document)["diagnostics"]["horizons"]["6m"]
    native_coverage = Decimal(six["coverage"]["native"]["mean"])
    calibrated_coverage = Decimal(six["coverage"]["calibrated"]["mean"])
    assert abs(calibrated_coverage - Decimal("0.6")) < abs(native_coverage - Decimal("0.6"))
    assert Decimal(six["paired"]["candidate_average"]) < Decimal(six["paired"]["baseline_average"])


@pytest.mark.django_db
def test_registered_synthetic_study_flows_through_s2_without_fabricated_parameters(
    registered_demo_study: tuple[AssetStore, dict[str, Any]],
) -> None:
    _store, document = registered_demo_study
    fit = _fit(document)
    for horizon in ("6m", "12m"):
        assert fit["params"]["horizons"][horizon]["state"] == "withheld_native_fallback"
        assert fit["params"]["horizons"][horizon]["offset"] is None
        assert fit["diagnostics"]["horizons"][horizon]["paired"] is None
        assert fit["diagnostics"]["horizons"][horizon]["calibrated_observations"] == 0
    assert fit["params"]["training_cutoff"] is None


@pytest.mark.parametrize(
    ("gap", "expected"),
    [(126, True), (125, False), (1, False), (0, False)],
)
def test_six_month_embargo_boundary(gap: int, expected: bool) -> None:
    assert (
        embargo_eligible(
            train_outcome_end_index=1_000,
            test_anchor_index=1_000 + gap,
            embargo_sessions=126,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("gap", "expected"),
    [(252, True), (251, False), (1, False), (0, False)],
)
def test_twelve_month_embargo_boundary(gap: int, expected: bool) -> None:
    assert (
        embargo_eligible(
            train_outcome_end_index=1_000,
            test_anchor_index=1_000 + gap,
            embargo_sessions=252,
        )
        is expected
    )


def test_real_frozen_calendar_embargo_and_overlapping_purge() -> None:
    inputs = calibration_inputs_from_document(build_interval_calibration_document())
    indices = {session: index for index, session in enumerate(inputs.calendar_sessions)}
    validation = next(
        row for row in inputs.observations if row.partition == "validation" and row.horizon == "6m"
    )
    test_index = indices[validation.anchor_date]
    for embargo in (126, 252):
        assert embargo_eligible(
            train_outcome_end_index=test_index - embargo,
            test_anchor_index=test_index,
            embargo_sessions=embargo,
        )
        assert not embargo_eligible(
            train_outcome_end_index=test_index - embargo + 1,
            test_anchor_index=test_index,
            embargo_sessions=embargo,
        )

    overlapping = CalibrationObservation(
        partition="development",
        horizon="6m",
        listing_id="synthetic-overlap",
        anchor_date=inputs.calendar_sessions[test_index - 2],
        outcome_end_date=inputs.calendar_sessions[test_index + 1],
        predicted=LedgerTriplet(Decimal("-0.01"), Decimal(0), Decimal("0.01")),
        actual_return=Decimal(0),
        unavailable_reason=None,
    )
    params_horizon = fit_interval_calibration(inputs)["params"]["horizons"]["6m"]
    folds, _offsets = _walk_forward_folds(
        (overlapping,),
        (validation,),
        horizon="6m",
        params_horizon=params_horizon,
        index_by_date=indices,
    )
    assert folds[0]["purged_observations"] == 1
    assert folds[0]["eligible_observations"] == 0


def test_holdout_permutation_and_validation_separation() -> None:
    original = build_interval_calibration_document()
    base = _fit(original)
    round_tripped = _fit(json.loads(json.dumps(original)))
    assert round_tripped == base
    reordered = copy.deepcopy(original)
    reordered["listings"].reverse()
    for listing in reordered["listings"]:
        listing["anchors"].reverse()
        for anchor in listing["anchors"]:
            anchor["models"] = anchor["models"][1:] + anchor["models"][:1]
    reordered["projection_aggregates"].reverse()
    assert s1.summarize_quantile_calibration(reordered)
    permuted = _fit(reordered)
    assert permuted["params"] == base["params"]
    assert permuted["diagnostics"] == base["diagnostics"]
    assert permuted["source"]["s1_ledger_input_sha256"] != base["source"]["s1_ledger_input_sha256"]
    assert {
        key: value for key, value in permuted["source"].items() if key != "s1_ledger_input_sha256"
    } == {key: value for key, value in base["source"].items() if key != "s1_ledger_input_sha256"}

    holdout = copy.deepcopy(original)
    for listing in holdout["listings"]:
        for anchor in listing["anchors"]:
            if anchor["partition"] == "final_holdout":
                for model in anchor["models"]:
                    metrics = model["metrics"]
                    if metrics["unavailable_reason"] is None:
                        metrics["actual_return"] = "0.0400"
                        metrics["interval_included"] = (
                            Decimal(metrics["predicted_returns"]["lower"])
                            <= Decimal("0.0400")
                            <= Decimal(metrics["predicted_returns"]["upper"])
                        )
    _rebuild_aggregates(holdout)
    changed_holdout = _fit(holdout)
    assert changed_holdout["params"] == base["params"]
    assert changed_holdout["diagnostics"] == base["diagnostics"]
    assert (
        changed_holdout["source"]["s1_ledger_input_sha256"]
        != base["source"]["s1_ledger_input_sha256"]
    )

    validation = copy.deepcopy(original)
    for listing in validation["listings"]:
        for anchor in listing["anchors"]:
            if anchor["partition"] == "validation":
                for model in anchor["models"]:
                    model["metrics"]["actual_return"] = "0.0400"
                    model["metrics"]["interval_included"] = (
                        Decimal(model["metrics"]["predicted_returns"]["lower"])
                        <= Decimal("0.0400")
                        <= Decimal(model["metrics"]["predicted_returns"]["upper"])
                    )
    _rebuild_aggregates(validation)
    changed_validation = _fit(validation)
    assert changed_validation["params"] == base["params"]
    assert changed_validation["diagnostics"] != base["diagnostics"]


def test_exact_decimal_canonicalization_changes_fit_identity_without_rounding() -> None:
    rounded_spelling = build_interval_calibration_document()
    extra_precision = copy.deepcopy(rounded_spelling)
    equivalent_spelling = copy.deepcopy(rounded_spelling)

    for document, actual in (
        (rounded_spelling, "0.0300"),
        (extra_precision, "0.030000000000000000000000000001"),
        (equivalent_spelling, "0.03000"),
    ):
        changed_once = False
        for listing in document["listings"]:
            for anchor in listing["anchors"]:
                if anchor["partition"] == "development" and anchor["horizon"] == "6m":
                    for model in anchor["models"]:
                        model["metrics"]["actual_return"] = actual
                        model["metrics"]["interval_included"] = (
                            Decimal(model["metrics"]["predicted_returns"]["lower"])
                            <= Decimal(actual)
                            <= Decimal(model["metrics"]["predicted_returns"]["upper"])
                        )
                    changed_once = True
                    break
            if changed_once:
                break
        _rebuild_aggregates(document)
        # S1's aggregate comparison intentionally rounds at its frozen precision,
        # so both values remain valid and distinguishable S2 source evidence.
        assert s1.summarize_quantile_calibration(document)

    base = _fit(rounded_spelling)
    precise = _fit(extra_precision)
    equivalent = _fit(equivalent_spelling)
    assert base["params"]["fit_input_sha256"] != precise["params"]["fit_input_sha256"]
    assert base["params"]["params_sha256"] != precise["params"]["params_sha256"]
    assert base["params"] == equivalent["params"]
    assert base["diagnostics"] == equivalent["diagnostics"]
    assert (
        base["source"]["s1_ledger_input_sha256"] != equivalent["source"]["s1_ledger_input_sha256"]
    )
    assert canonical_json_bytes(Decimal("0.0300")) == b'"0.03"'
    assert (
        canonical_json_bytes(Decimal("0.030000000000000000000000000001"))
        == b'"0.030000000000000000000000000001"'
    )


def test_development_fhs_sensitivity_and_comparator_exclusion() -> None:
    document = build_interval_calibration_document()
    base = _fit(document)
    changed = copy.deepcopy(document)
    for listing in changed["listings"]:
        for anchor in listing["anchors"]:
            if anchor["partition"] == "development":
                for model in anchor["models"]:
                    model["metrics"]["actual_return"] = "0.0500"
                    model["metrics"]["interval_included"] = (
                        Decimal(model["metrics"]["predicted_returns"]["lower"])
                        <= Decimal("0.0500")
                        <= Decimal(model["metrics"]["predicted_returns"]["upper"])
                    )
    _rebuild_aggregates(changed)
    changed_fit = _fit(changed)
    assert changed_fit["params"]["fit_input_sha256"] != base["params"]["fit_input_sha256"]
    assert changed_fit["params"]["params_sha256"] != base["params"]["params_sha256"]

    comparator = copy.deepcopy(document)
    for listing in comparator["listings"]:
        for anchor in listing["anchors"]:
            if anchor["partition"] == "development":
                for model in anchor["models"]:
                    if model["model_name"] != "us-price-fhs-v1":
                        model["metrics"]["predicted_returns"] = {
                            "lower": "-0.2000",
                            "median": "0.0000",
                            "upper": "0.2000",
                        }
                        actual = Decimal(model["metrics"]["actual_return"])
                        model["metrics"]["interval_included"] = (
                            Decimal("-0.2000") <= actual <= Decimal("0.2000")
                        )
    _rebuild_aggregates(comparator)
    assert _fit(comparator)["params"] == base["params"]


def test_withheld_reasons_and_unavailable_horizons() -> None:
    one_cluster = build_interval_calibration_document(development_6m_clusters=1)
    one = _fit(one_cluster)["params"]["horizons"]["6m"]
    assert one["reason"] == "conformal_rank_infeasible"
    assert one["offset"] is None

    no_observations = build_interval_calibration_document()
    for listing in no_observations["listings"]:
        for anchor in listing["anchors"]:
            if anchor["partition"] == "development" and anchor["horizon"] == "6m":
                for model in anchor["models"]:
                    if model["model_name"] == "us-price-fhs-v1":
                        model["metrics"] = {
                            "actual_return": None,
                            "predicted_returns": None,
                            "interval_included": None,
                            "unavailable_reason": "synthetic_missing",
                        }
    _rebuild_aggregates(no_observations)
    no_scored = _fit(no_observations)["params"]["horizons"]["6m"]
    assert no_scored["reason"] == "no_scored_observations"
    assert no_scored["offset"] is None

    boundary = build_interval_calibration_document(actual_return="-1.0000")
    for listing in boundary["listings"]:
        for anchor in listing["anchors"]:
            if anchor["partition"] == "development" and anchor["horizon"] == "6m":
                for model in anchor["models"]:
                    model["metrics"]["actual_return"] = "-1.0000"
                    model["metrics"]["interval_included"] = False
    _rebuild_aggregates(boundary)
    assert _fit(boundary)["params"]["horizons"]["6m"]["reason"] == (
        "log_scale_boundary_observation"
    )

    thin_cluster = build_interval_calibration_document(listings_per_cluster=29)
    thin = _fit(thin_cluster)["params"]["horizons"]["6m"]
    assert thin["reason"] == "insufficient_effective_support"
    assert "cluster_listings_below_floor" in thin["failing_components"]
    short_span = build_interval_calibration_document(development_6m_clusters=6)
    short = _fit(short_span)["params"]["horizons"]["6m"]
    assert "non_overlapping_clusters_below_floor" in short["failing_components"]
    assert "span_below_floor" in short["failing_components"]

    unavailable = _fit(build_interval_calibration_document())["params"]["horizons"]
    assert unavailable["12m"]["reason"] == "insufficient_effective_support"
    assert unavailable["3y"]["reason"] == "horizon_unavailable_no_calibration_claim"
    assert unavailable["5y"]["reason"] == "horizon_unavailable_no_calibration_claim"
    from stanstock.research.reporting import _ADVISORY_FLOORS

    assert _ENGINEERING_FLOORS_FOR_TEST["12m"] == _ADVISORY_FLOORS["12m"]
    assert _ENGINEERING_FLOORS_FOR_TEST["6m"] == (
        _ADVISORY_FLOORS["6m"][0] - 1,
        *_ADVISORY_FLOORS["6m"][1:],
    )


def test_core_rejects_bad_config_calendar_partition_duplicate_and_numeric_domain() -> None:
    inputs = calibration_inputs_from_document(build_interval_calibration_document())
    config = load_price_product_config()
    from dataclasses import replace

    with pytest.raises(ValueError):
        fit_interval_calibration(replace(inputs, config=replace(config, currency="EUR")))
    with pytest.raises(ValueError):
        fit_interval_calibration(
            replace(
                inputs, config_identity={**inputs.config_identity, "effective_config_hash": "x"}
            )
        )
    with pytest.raises(ValueError):
        fit_interval_calibration(
            replace(inputs, calendar_identity={**inputs.calendar_identity, "session_count": 1})
        )
    with pytest.raises(ValueError):
        fit_interval_calibration(
            replace(inputs, calendar_sessions=tuple(reversed(inputs.calendar_sessions)))
        )
    with pytest.raises(ValueError):
        fit_interval_calibration(
            replace(
                inputs,
                calendar_sessions=(
                    *inputs.calendar_sessions[:1],
                    inputs.calendar_sessions[1],
                    *inputs.calendar_sessions[1:],
                ),
            )
        )
    first = inputs.observations[0]
    with pytest.raises(ValueError):
        fit_interval_calibration(replace(inputs, observations=(first, first)))
    with pytest.raises(ValueError):
        fit_interval_calibration(
            replace(
                inputs,
                observations=(
                    replace(first, anchor_date=date(2000, 1, 3)),
                    *inputs.observations[1:],
                ),
            )
        )
    with pytest.raises(ValueError):
        fit_interval_calibration(
            replace(
                inputs,
                observations=(
                    replace(first, outcome_end_date=date(2000, 1, 4)),
                    *inputs.observations[1:],
                ),
            )
        )
    with pytest.raises(ValueError):
        fit_interval_calibration(
            replace(inputs, observations=(replace(first, partition="final_holdout"),))
        )
    with pytest.raises(ValueError):
        fit_interval_calibration(
            replace(
                inputs,
                observations=(
                    replace(first, actual_return=Decimal("NaN")),
                    *inputs.observations[1:],
                ),
            )
        )
    with pytest.raises(ValueError):
        fit_interval_calibration(
            replace(
                inputs,
                observations=(
                    replace(first, predicted=None),
                    *inputs.observations[1:],
                ),
            )
        )


def test_training_below_minus_one_rejects_and_adjust_interval_boundaries() -> None:
    document = build_interval_calibration_document()
    for listing in document["listings"]:
        for anchor in listing["anchors"]:
            if anchor["partition"] == "development" and anchor["horizon"] == "6m":
                for model in anchor["models"]:
                    model["metrics"]["actual_return"] = "-1.1000"
                    model["metrics"]["interval_included"] = False
    _rebuild_aggregates(document)
    with pytest.raises(ValueError, match="return_below_minus_one"):
        _fit(document)

    values = adjust_interval(Decimal("-0.1"), Decimal(0), Decimal("0.1"), Decimal(0))
    assert values == (Decimal("-0.1"), Decimal(0), Decimal("0.1"))
    assert adjust_interval(Decimal("-1"), Decimal(0), Decimal("0.1"), Decimal(1)) is None
    widened = adjust_interval(Decimal("-0.1"), Decimal(0), Decimal("0.1"), Decimal("0.2"))
    assert widened is not None and widened[0] < Decimal("-0.1") < widened[2]


def test_validation_log_boundary_keeps_native_and_never_fabricates_pairs() -> None:
    document = build_interval_calibration_document()
    for listing in document["listings"]:
        for anchor in listing["anchors"]:
            if anchor["partition"] == "validation":
                for model in anchor["models"]:
                    model["metrics"]["actual_return"] = "-1.0000"
                    model["metrics"]["interval_included"] = False
    _rebuild_aggregates(document)
    diagnostic = _fit(document)["diagnostics"]["horizons"]["6m"]
    assert diagnostic["native_observations"] == 30
    assert diagnostic["candidate_unrepresentable"] == 30
    assert diagnostic["paired"]["paired_observation_count"] == 0
    assert (
        diagnostic["paired"]["exclusions"][
            "candidate_unavailable:calibrated_triplet_unrepresentable"
        ]
        == 30
    )


def test_anchor_relabeling_and_shifted_dates_are_rejected_after_s1_revalidation() -> None:
    relabeled = build_interval_calibration_document()
    changed = False
    for listing in relabeled["listings"]:
        for anchor in listing["anchors"]:
            if anchor["partition"] == "final_holdout":
                anchor["partition"] = "development"
                changed = True
                break
        if changed:
            break
    _rebuild_aggregates(relabeled)
    with pytest.raises(ValueError):
        _fit(relabeled)

    shifted = build_interval_calibration_document()
    for listing in shifted["listings"]:
        for anchor in listing["anchors"]:
            if anchor["partition"] == "development" and anchor["horizon"] == "6m":
                anchor["anchor_date"] = "2020-03-05"
                break
        else:
            continue
        break
    _rebuild_aggregates(shifted)
    with pytest.raises(ValueError):
        _fit(shifted)


def test_arithmetic_overflow_is_withheld_as_a_value_free_error() -> None:
    from dataclasses import replace

    inputs = calibration_inputs_from_document(build_interval_calibration_document())
    changed = list(inputs.observations)
    index = next(
        index
        for index, row in enumerate(changed)
        if row.partition == "development" and row.horizon == "6m" and row.actual_return is not None
    )
    changed[index] = replace(changed[index], actual_return=Decimal("1E+999999999999999999"))
    with pytest.raises(ValueError) as error:
        fit_interval_calibration(replace(inputs, observations=tuple(changed)))
    assert str(error.value) == "arithmetic_unrepresentable"
    assert "999999" not in str(error.value)


_ENGINEERING_FLOORS_FOR_TEST = {"6m": (7, 30, 1095), "12m": (6, 30, 1460)}


def test_context_matches_frozen_s1_context() -> None:
    assert _CONTEXT.prec == s1._CONTEXT.prec
    assert _CONTEXT.rounding == s1._CONTEXT.rounding
    assert _CONTEXT.Emin == s1._CONTEXT.Emin
    assert _CONTEXT.Emax == s1._CONTEXT.Emax
    assert _CONTEXT.traps == s1._CONTEXT.traps


@pytest.mark.django_db
def test_document_adapter_has_zero_queries_and_never_uses_provider(
    django_assert_num_queries: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = build_interval_calibration_document()
    blocked = Mock(side_effect=AssertionError("provider must not be used"))
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("stanstock.data.providers.twelve_data.resolve_api_key", blocked)
    with django_assert_num_queries(0):
        inputs = calibration_inputs_from_document(document)
        fit = fit_interval_calibration_document(document)
    assert all(row.partition in ("development", "validation") for row in inputs.observations)
    assert all(row.horizon in ("6m", "12m") for row in inputs.observations)
    assert fit["source"]["attestation"] == "caller_supplied_file_unverified"
    blocked.assert_not_called()


@pytest.mark.parametrize(
    "identity_key",
    ["config_identity", "protocol_identity", "calendar_identity"],
)
@pytest.mark.django_db
def test_adapter_rejects_identity_changes(
    identity_key: str, django_assert_num_queries: Any
) -> None:
    document = build_interval_calibration_document()
    original = document[identity_key]
    variants = [
        {**original, "tampered": "identity"},
        {key: value for key, value in original.items() if key != next(iter(original))},
    ]
    first_key = next(iter(original))
    changed = dict(original)
    changed[first_key] = "changed"
    variants.append(changed)
    for variant in variants:
        document[identity_key] = variant
        with django_assert_num_queries(0), pytest.raises(ValueError):
            _fit(document)
    document[identity_key] = original


@pytest.mark.django_db
def test_adapter_identity_comparison_preserves_json_scalar_types(
    django_assert_num_queries: Any,
) -> None:
    document = build_interval_calibration_document()
    document["protocol_identity"]["purge_partition_crossings"] = int(
        document["protocol_identity"]["purge_partition_crossings"]
    )
    with django_assert_num_queries(0), pytest.raises(ValueError):
        _fit(document)


@pytest.mark.django_db
@pytest.mark.parametrize("output_format", ["json", "text"])
def test_command_is_safe_deterministic_zero_query(
    output_format: str,
    django_assert_num_queries: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = build_interval_calibration_document()
    study = tmp_path / "study.json"
    payload = json.dumps(document, sort_keys=True).encode()
    study.write_bytes(payload)
    blocked = Mock(side_effect=AssertionError("provider must not be used"))
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("stanstock.data.providers.twelve_data.resolve_api_key", blocked)
    from io import StringIO

    output = StringIO()
    with django_assert_num_queries(0):
        call_command(
            COMMAND,
            "--study-file",
            str(study),
            "--format",
            output_format,
            stdout=output,
        )
    assert study.read_bytes() == payload
    assert sorted(item.name for item in tmp_path.iterdir()) == ["study.json"]
    if output_format == "json":
        assert canonical_json_bytes(json.loads(output.getvalue())) == canonical_json_bytes(
            _fit(document)
        )
    else:
        assert "research-only" in output.getvalue()
        assert "order-sensitive provenance" in output.getvalue()
    blocked.assert_not_called()


def test_hardened_cli_reader_rejects_symlink_and_duplicate_json_key(tmp_path: Path) -> None:
    source = tmp_path / "study.json"
    source.write_text('{"a":1,"a":2}', encoding="utf-8")
    from io import StringIO

    with pytest.raises(CommandError, match="duplicate JSON object key"):
        call_command(COMMAND, "--study-file", str(source), "--format", "json", stdout=StringIO())
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(CommandError, match="symbolic link"):
        call_command(COMMAND, "--study-file", str(link), "--format", "json", stdout=StringIO())
    assert sorted(path.name for path in tmp_path.iterdir()) == ["link.json", "study.json"]


def test_canonical_serialization_is_strict_and_hashes_self_excluded() -> None:
    assert canonical_json_bytes({"d": Decimal("1.2300"), "day": date(2024, 1, 2)}) == (
        b'{"d":"1.23","day":"2024-01-02"}'
    )
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": 1.0})
    assert canonical_sha256({"x": 1}) == canonical_sha256({"x": 1})


def test_cold_core_fit_closure_purity_and_determinism(tmp_path: Path) -> None:
    document = build_interval_calibration_document()
    inputs = calibration_inputs_from_document(document)
    expected = canonical_json_bytes(fit_interval_calibration(inputs))
    script = r"""
import builtins, os, pickle, sys, time

blocked_prefixes = (
    "django", "exchange_calendars", "pandas", "httpx", "stanstock.core",
    "stanstock.data", "stanstock.web", "stanstock.simulation",
    "stanstock.research.models", "stanstock.research.price_product_study",
    "stanstock.research.product_study_evidence",
    "stanstock.research.price_product_calibration",
    "stanstock.research.forward_scorecard",
)
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.startswith(blocked_prefixes):
        raise AssertionError("forbidden import: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from stanstock.research.price_product_interval_calibration import (
    canonical_json_bytes, fit_interval_calibration,
)
loaded = {name for name in sys.modules if name.startswith("stanstock")}
allowed = {
    "stanstock", "stanstock.research",
    "stanstock.research.price_product_interval_calibration",
    "stanstock.research.price_product", "stanstock.research.price_product_config",
    "stanstock.research.price_product_replay",
}
assert loaded == allowed, loaded
inputs = pickle.loads(sys.stdin.buffer.read())
def audit(event, args):
    if (
        event == "open" or event.startswith(("socket.", "subprocess.", "sqlite3."))
        or event.startswith("os.")
    ):
        raise AssertionError("forbidden side effect: " + event)
sys.addaudithook(audit)
class ForbiddenEnvironment:
    def __getattribute__(self, name):
        raise AssertionError("environment access")
os.environ = ForbiddenEnvironment()
os.environb = ForbiddenEnvironment()
def refuse_environment(*args, **kwargs):
    raise AssertionError("environment access")
os.getenv = refuse_environment
os.getenvb = refuse_environment
os.putenv = refuse_environment
os.unsetenv = refuse_environment
time.time = lambda: (_ for _ in ()).throw(AssertionError("clock"))
time.time_ns = lambda: (_ for _ in ()).throw(AssertionError("clock"))
first = canonical_json_bytes(fit_interval_calibration(inputs))
second = canonical_json_bytes(fit_interval_calibration(inputs))
assert first == second
sys.stdout.buffer.write(first)
"""
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        input=pickle.dumps(inputs),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == expected
    assert list(tmp_path.iterdir()) == []

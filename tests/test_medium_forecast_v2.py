from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
import threading
import time
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import polars as pl
import pytest
from django.contrib.auth import get_user_model
from django.db import IntegrityError, connection, connections, transaction
from django.urls import reverse
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.data.asof import AsOfData, PriceFrameChecksumMismatchError
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.management.config_loader import default_us_scoring_config_path
from stanstock.data.models import (
    Company,
    DataAsset,
    Listing,
    Region,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.research import medium_forecasts
from stanstock.research.forecast_config import (
    MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
    MEDIUM_V2_LITERAL_SHA256,
    MEDIUM_V2_VERSION,
    MediumForecastV2Config,
    is_medium_forecast_v2_candidate,
    load_medium_forecast_config,
    medium_forecast_config_hash,
)
from stanstock.research.medium_forecasts import (
    PANEL_SCHEMA,
    MediumForecast,
    build_medium_forecast_panel,
    build_medium_forecasts,
    hash_json,
    serialize_medium_forecast_panel,
)
from stanstock.research.models import AnalysisRun, Prediction, PredictionOutcome, StockAnalysis
from stanstock.research.opportunities import assess_opportunity
from stanstock.research.outcomes import evaluate_prediction
from stanstock.research.reporting import (
    _advisory_target_summaries,
    reportable_prediction_filter,
)
from stanstock.research.service import (
    AnalysisOutputPaths,
    _asset_payload,
    _create_advisory_prediction,
    _is_medium_v2_prediction,
    analyze_snapshot,
    append_advisory_predictions,
)
from stanstock.research.types import Scenario
from stanstock.simulation.builders import build_signals_for_backtest
from stanstock.web.templatetags.stanstock import validated_medium_forecast_scenario

V2_PATH = Path("config/forecasts/us-price-medium-v2.yml")
TARGET_DATE = date(2026, 9, 4)
GENERATED_AT = datetime(2026, 9, 4, 20, tzinfo=UTC)
HISTORICAL_GENERATED_AT = datetime(2026, 9, 5, 1, tzinfo=UTC)
HISTORICAL_CUTOFF = datetime(2026, 9, 4, 23, 59, 59, 999999, tzinfo=UTC)


def _v2_config() -> MediumForecastV2Config:
    config = load_medium_forecast_config(V2_PATH)
    assert isinstance(config, MediumForecastV2Config)
    return config


def _small_v2_config() -> MediumForecastV2Config:
    config = _v2_config()
    return replace(
        config,
        horizons={
            name: replace(
                horizon,
                minimum_raw_matches=1,
                minimum_effective_cohorts=1,
                minimum_distinct_listings=1,
                shrinkage_prior_cohorts=1.0,
                probability_minimum_effective_cohorts=1,
                probability_minimum_distinct_listings=1,
                probability_minimum_calendar_span_days=0,
                probability_minimum_distinct_market_regimes=1,
            )
            for name, horizon in config.horizons.items()
        },
        walk_forward=replace(
            config.walk_forward,
            minimum_training_cohorts=1,
            minimum_test_cohorts=4,
        ),
    )


def _panel_row(
    *,
    listing_id: str,
    anchor: date,
    value: float | None,
    current: bool = False,
    label_end: date | None = None,
    state: int = 1,
    regime: int = 1,
) -> dict[str, object]:
    return {
        "horizon": "6m",
        "anchor_date": anchor,
        "label_end_date": None if current else (label_end or anchor),
        "is_forecast": current,
        "cohort_id": f"6m:{anchor.isoformat()}",
        "listing_id": listing_id,
        "ticker": listing_id,
        "price_asset_id": listing_id,
        "relative_momentum": 0.1,
        "drawdown": -0.1,
        "volatility": 0.2,
        "market_trend": 0.05,
        "market_volatility": 0.2,
        "relative_momentum_bucket": state,
        "drawdown_bucket": state,
        "volatility_bucket": state,
        "market_trend_bucket": regime,
        "market_volatility_bucket": regime,
        "close_vs_sma_50": 0.05,
        "close_vs_sma_200": 0.1,
        "downside_volatility": 0.1,
        "average_dollar_volume": 10_000_000.0,
        "forward_return": value,
        "benchmark_forward_return": None if value is None else value / 2,
        "relative_forward_return": None if value is None else value / 2,
        "eligible": True,
        "insufficiency_reason": "",
    }


def _panel(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=PANEL_SCHEMA, orient="row")


def _sessions(count: int, target: date = TARGET_DATE) -> list[date]:
    calendar = get_calendar("XNYS")
    target_session = calendar.date_to_session(target)
    first = calendar.session_offset(target_session, -(count - 1))
    return [session.date() for session in calendar.sessions_in_range(first, target_session)]


def _listing(
    ticker: str,
    *,
    region: str = Region.US,
    currency: str = "USD",
    security_type: str = Security.SecurityType.COMMON_STOCK,
) -> Listing:
    company = Company.objects.create(name=f"{ticker} Company", country="US")
    security = Security.objects.create(
        company=company,
        name=f"{ticker} security",
        security_type=security_type,
    )
    return Listing.objects.create(
        security=security,
        ticker=ticker,
        exchange_mic="XNAS",
        provider_symbol=ticker,
        currency=currency,
        region=region,
    )


def _price_asset(
    store: AssetStore,
    *,
    subject: str,
    sessions: list[date],
    available_at: datetime,
    retrieved_at: datetime,
    slope: float = 0.03,
    suffix: str | None = None,
    volume: int = 2_000_000,
) -> DataAsset:
    frame = pl.DataFrame(
        {
            "date": sessions,
            "close": [
                40 + index * slope + ((index % 17) - 8) * 0.04 for index in range(len(sessions))
            ],
            "volume": [volume for _index in range(len(sessions))],
        },
        schema_overrides={"date": pl.Date, "close": pl.Float64, "volume": pl.Int64},
    )
    stored = store.write_frame(
        f"medium-v2/{subject}-{suffix or uuid4().hex}.parquet",
        frame,
    )
    return register_asset(
        provider="synthetic",
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=retrieved_at,
        available_at=available_at,
        period_start=sessions[0],
        period_end=sessions[-1],
        metadata={
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )


def _panel_kwargs(
    *,
    listings: list[Listing],
    store: AssetStore,
    data_cutoff: datetime,
    generated_at: datetime = HISTORICAL_GENERATED_AT,
) -> dict[str, Any]:
    return {
        "listings": listings,
        "asof": AsOfData(generated_at, store),
        "provider": "synthetic",
        "benchmark_subject": "SPY",
        "target_date": TARGET_DATE,
        "generated_at": generated_at,
        "run_id": uuid4(),
        "config": _v2_config(),
        "config_hash": MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
        "scoring_config_version": "us-price-baseline-v2",
        "scoring_config_hash": ("43cc0ee0e29f79dec4ad8d8a91e43e7b3df1d368dbc385a116fcc6ff05f18e9b"),
        "universe_snapshot_id": uuid4(),
        "universe_slug": "v2-test",
        "universe_config_hash": "u" * 64,
        "code_revision": "test-revision",
        "store": store,
        "data_cutoff": data_cutoff,
    }


def test_v2_config_exact_bytes_and_typed_identity() -> None:
    payload = V2_PATH.read_bytes()
    config = _v2_config()

    assert len(payload) == 2_311
    assert payload.endswith(b"\n")
    assert hashlib.sha256(payload).hexdigest() == MEDIUM_V2_LITERAL_SHA256
    assert config.version == MEDIUM_V2_VERSION
    assert config.schema_version == 2
    assert medium_forecast_config_hash(config) == MEDIUM_V2_EFFECTIVE_CONFIG_HASH


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda text: text + "schema_version: 1\n",
            "forbids duplicate keys",
        ),
        (
            lambda text: text.replace(
                "  momentum_sessions: 252\n",
                "  momentum_sessions: 252\n  momentum_sessions: 252\n",
            ),
            "forbids duplicate keys",
        ),
        (
            lambda text: text.replace(
                "version: us-price-medium-v2",
                "version: &method us-price-medium-v2",
            ),
            "forbids YAML anchors, aliases, and merges",
        ),
        (
            lambda text: text.replace(
                "schema_version: 2",
                "defaults: &defaults {calendar: XNYS}\n<<: *defaults\nschema_version: 2",
            ),
            "forbids YAML anchors, aliases, and merges",
        ),
        (
            lambda text: text.replace(
                "schema_version: 2",
                "schema_version: 2\n1: non-string",
            ),
            "config is malformed",
        ),
    ],
)
def test_v2_parser_rejects_recursive_duplicates_yaml_features_and_nonstring_keys(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    path = tmp_path / "us-price-medium-v2.yml"
    path.write_text(mutation(V2_PATH.read_text(encoding="utf-8")), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_medium_forecast_config(path)


def test_v2_candidate_never_falls_through_to_v1(tmp_path: Path) -> None:
    for name, payload in (
        ("us-price-medium-v2.yml", "not: [valid"),
        ("renamed.yml", "version: us-price-medium-v2\nnot: [valid"),
        ("schema.yml", "schema_version: 2\nversion: wrong\n"),
        ("conflict.yml", "schema_version: 1\nschema_version: 2\n"),
    ):
        path = tmp_path / name
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError, match="us-price-medium-v2 config"):
            load_medium_forecast_config(path)

    missing = tmp_path / "missing" / "us-price-medium-v2.yml"
    with pytest.raises(ValueError, match="config is malformed") as excinfo:
        load_medium_forecast_config(missing)
    assert str(missing) not in str(excinfo.value)


def test_v2_partial_root_schema_claim_excludes_lookalikes(tmp_path: Path) -> None:
    claimed = tmp_path / "renamed.yml"
    claimed.write_text("schema_version: 2\nbroken: [\n", encoding="utf-8")
    assert is_medium_forecast_v2_candidate(claimed) is True
    with pytest.raises(ValueError, match="us-price-medium-v2 config is malformed"):
        load_medium_forecast_config(claimed)

    for name, payload in (
        ("comment.yml", "# schema_version: 2\nschema_version: 1\n"),
        ("nested.yml", "outer:\n  schema_version: 2\nschema_version: 1\n"),
        ("quoted.yml", 'schema_version: "2"\n'),
    ):
        path = tmp_path / name
        path.write_text(payload, encoding="utf-8")
        assert is_medium_forecast_v2_candidate(path) is False


def test_v2_altered_literal_and_effective_identity_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    altered = tmp_path / "renamed.yml"
    altered.write_bytes(V2_PATH.read_bytes() + b"# altered\n")
    with pytest.raises(ValueError, match="config identity mismatch"):
        load_medium_forecast_config(altered)

    import stanstock.research.forecast_config as config_module

    monkeypatch.setattr(config_module, "MEDIUM_V2_EFFECTIVE_CONFIG_HASH", "0" * 64)
    with pytest.raises(ValueError, match="typed effective config identity mismatch"):
        load_medium_forecast_config(V2_PATH)


@pytest.mark.parametrize("mutation", ["boolean_integer", "unknown_key"])
def test_v2_typed_parser_rejects_wrong_types_and_exact_keys_after_literal_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import stanstock.research.forecast_config as config_module

    payload = V2_PATH.read_bytes()
    if mutation == "boolean_integer":
        payload = payload.replace(
            b"  momentum_sessions: 252\n",
            b"  momentum_sessions: true\n",
        ).replace(
            b"minimum_history_coverage: 0.95\n",
            b"minimum_history_coverage: .95\n",
        )
    else:
        payload = payload.replace(b"calendar: XNYS", b"calendaX: XNYS")
    assert len(payload) == 2_311
    monkeypatch.setattr(
        config_module,
        "MEDIUM_V2_LITERAL_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    path = tmp_path / "us-price-medium-v2.yml"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="config is malformed"):
        load_medium_forecast_config(path)


def test_v2_cohort_equal_cdf_ties_and_input_order() -> None:
    config = _small_v2_config()
    rows = [
        _panel_row(listing_id="a", anchor=date(2020, 1, 2), value=-1.0),
        _panel_row(listing_id="b", anchor=date(2020, 1, 2), value=1.0),
        _panel_row(listing_id="c", anchor=date(2021, 1, 4), value=0.0),
        _panel_row(
            listing_id="current",
            anchor=TARGET_DATE,
            value=None,
            current=True,
        ),
    ]

    forward = build_medium_forecasts(_panel(rows), config)["current"]["6m"]
    reverse = build_medium_forecasts(_panel(list(reversed(rows))), config)["current"]["6m"]

    assert forward.scenario.bear == -1.0
    assert forward.scenario.base == 0.0
    assert forward.scenario.bull == 1.0
    assert forward.calculation["predictive_distribution"][
        "probability_positive_raw"
    ] == pytest.approx(0.25)
    assert forward.calculation == reverse.calculation
    assert forward.scenario == reverse.scenario
    predictive = forward.calculation["predictive_distribution"]
    assert predictive["matched"]["normalized_mass"] == 1.0
    assert predictive["unconditional"]["normalized_mass"] == 1.0
    assert predictive["matched"]["component_mass"] + predictive["unconditional"][
        "component_mass"
    ] == pytest.approx(1.0)


def test_v2_cdf_is_exact_across_unequal_full_row_permutations() -> None:
    config = _small_v2_config()
    historical = [
        _panel_row(listing_id="a", anchor=date(2019, 1, 2), value=0.0),
        _panel_row(listing_id="b", anchor=date(2019, 1, 2), value=0.5),
        _panel_row(listing_id="c", anchor=date(2019, 1, 2), value=0.5),
        _panel_row(listing_id="d", anchor=date(2020, 1, 2), value=-0.25),
        _panel_row(listing_id="e", anchor=date(2020, 1, 2), value=0.0),
        _panel_row(listing_id="f", anchor=date(2021, 1, 4), value=0.75),
    ]
    current = _panel_row(
        listing_id="current",
        anchor=TARGET_DATE,
        value=None,
        current=True,
    )
    rows = [*historical, current]
    permutations = [rows, list(reversed(rows)), rows[2:] + rows[:2]]
    for seed in range(8):
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        permutations.append(shuffled)

    calculations: list[dict[str, Any]] = []
    scenarios: list[Scenario] = []
    canonical_json: list[bytes] = []
    for permutation in permutations:
        forecast = build_medium_forecasts(_panel(permutation), config)["current"]["6m"]
        calculations.append(forecast.calculation)
        scenarios.append(forecast.scenario)
        canonical_json.append(
            json.dumps(
                {
                    "calculation": forecast.calculation,
                    "scenario": forecast.scenario.as_dict(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )

    assert all(calculation == calculations[0] for calculation in calculations)
    assert all(scenario == scenarios[0] for scenario in scenarios)
    assert all(payload == canonical_json[0] for payload in canonical_json)

    estimate = medium_forecasts._v2_estimate_distribution(
        historical,
        current=current,
        horizon_config=config.horizons["6m"],
        config=config,
    )
    assert estimate is not None
    cdf_at_zero = math.fsum(mass for value, mass in estimate.mixture.masses if value <= 0.0)
    strict_positive = 1.0 - cdf_at_zero
    assert estimate.mixture.probability_positive == strict_positive
    assert calculations[0]["predictive_distribution"]["probability_positive_raw"] == (
        strict_positive
    )


def test_v2_unconditional_fallback_is_the_unconditional_cdf() -> None:
    config = _small_v2_config()
    rows = [
        _panel_row(
            listing_id="a",
            anchor=date(2020, 1, 2),
            value=-0.2,
            state=1,
        ),
        _panel_row(
            listing_id="b",
            anchor=date(2021, 1, 4),
            value=0.4,
            state=1,
        ),
        _panel_row(
            listing_id="current",
            anchor=TARGET_DATE,
            value=None,
            current=True,
            state=9,
        ),
    ]
    forecast = build_medium_forecasts(_panel(rows), config)["current"]["6m"]
    predictive = forecast.calculation["predictive_distribution"]

    assert forecast.calculation["support"]["fallback_level"] == "unconditional"
    assert predictive["matched"]["component_mass"] == 0.0
    assert predictive["unconditional"]["component_mass"] == 1.0
    assert predictive["p20"] == predictive["unconditional"]["p20"]
    assert predictive["p50"] == predictive["unconditional"]["p50"]
    assert predictive["p80"] == predictive["unconditional"]["p80"]


def test_v2_refuses_below_minus_one_and_accepts_exact_minus_one() -> None:
    config = _small_v2_config()
    current = _panel_row(
        listing_id="current",
        anchor=TARGET_DATE,
        value=None,
        current=True,
    )
    with pytest.raises(
        ValueError,
        match="us-price-medium-v2 refuses support returns below -1.0",
    ):
        build_medium_forecasts(
            _panel(
                [
                    _panel_row(
                        listing_id="a",
                        anchor=date(2020, 1, 2),
                        value=-1.0000000001,
                    ),
                    current,
                ]
            ),
            config,
        )

    accepted = build_medium_forecasts(
        _panel(
            [
                _panel_row(
                    listing_id="a",
                    anchor=date(2020, 1, 2),
                    value=-1.0,
                ),
                current,
            ]
        ),
        config,
    )["current"]["6m"]
    assert accepted.scenario.bear == -1.0
    assert "return_floor" not in accepted.calculation


def test_v2_maturity_boundary_and_distinct_range_probability_origins() -> None:
    config = _small_v2_config()
    origin = date(2022, 1, 3)
    records = [
        _panel_row(
            listing_id="equal",
            anchor=date(2020, 1, 2),
            label_end=origin,
            value=0.1,
        ),
        _panel_row(
            listing_id="later",
            anchor=date(2019, 1, 2),
            label_end=origin + timedelta(days=1),
            value=10.0,
        ),
    ]
    training = medium_forecasts._v2_training_rows(
        records,
        horizon="6m",
        origin=origin,
    )
    assert [row["listing_id"] for row in training] == ["equal"]

    restricted = replace(
        config,
        horizons={
            **config.horizons,
            "6m": replace(
                config.horizons["6m"],
                probability_minimum_effective_cohorts=3,
            ),
        },
    )
    evidence_rows: list[dict[str, object]] = []
    for index in range(6):
        anchor = date(2018 + index, 1, 2)
        evidence_rows.append(
            _panel_row(
                listing_id=f"L{index}",
                anchor=anchor,
                label_end=anchor + timedelta(days=180),
                value=0.1 if index % 2 else -0.1,
            )
        )
    evidence = medium_forecasts._v2_prequential_evidence(
        evidence_rows,
        horizon="6m",
        config=restricted,
    )
    assert evidence["base_accuracy"]["test_origins"] > evidence["probability_skill"]["test_origins"]


def test_v2_brier_statuses_and_raw_bss_are_exact() -> None:
    config = _small_v2_config()

    def origins(model: float, reference: float) -> list[dict[str, list[float]]]:
        return [{"model_brier": [model], "reference_brier": [reference]} for _ in range(4)]

    positive = medium_forecasts._v2_probability_skill_evidence(
        origins(0.1, 0.2),
        config=config,
    )
    zero_reference = medium_forecasts._v2_probability_skill_evidence(
        origins(0.0, 0.0),
        config=config,
    )
    zero = medium_forecasts._v2_probability_skill_evidence(
        origins(0.2, 0.2),
        config=config,
    )
    negative = medium_forecasts._v2_probability_skill_evidence(
        origins(0.3, 0.2),
        config=config,
    )

    assert positive["status"] == "positive_skill"
    assert positive["brier_skill_score"] == pytest.approx(0.5)
    assert zero_reference["status"] == "reference_zero"
    assert zero_reference["brier_skill_score"] is None
    assert zero["status"] == "zero_skill"
    assert zero["brier_skill_score"] == 0.0
    assert negative["status"] == "negative_skill"
    assert negative["brier_skill_score"] == pytest.approx(-0.5)


def test_v2_interval_endpoint_score_and_date_equal_weighting() -> None:
    config = _small_v2_config()
    origins = [
        {
            "covered": [1.0, 1.0],
            "below": [0.0, 0.0],
            "above": [0.0, 0.0],
            "width": [0.4, 0.2],
            "model_interval_score": [0.4, 0.2],
            "reference_interval_score": [0.6, 0.4],
        },
        {
            "covered": [0.0],
            "below": [1.0],
            "above": [0.0],
            "width": [0.1],
            "model_interval_score": [0.6],
            "reference_interval_score": [0.8],
        },
    ]
    interval = medium_forecasts._v2_interval_evidence(origins, config=config)

    assert medium_forecasts._v2_interval_score(-0.2, 0.2, -0.2) == 0.4
    assert medium_forecasts._v2_interval_score(-0.2, 0.2, -0.3) == pytest.approx(0.9)
    assert interval["empirical_coverage"] == pytest.approx(0.5)
    assert interval["mean_width"] == pytest.approx(0.2)
    assert interval["model_mean_interval_score"] == pytest.approx(0.45)
    assert "median_width" not in interval


def test_v2_three_states_keep_exact_compact_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_v2_config()
    rows = [
        _panel_row(listing_id="a", anchor=date(2020, 1, 2), value=-0.2),
        _panel_row(listing_id="b", anchor=date(2021, 1, 4), value=0.4),
        _panel_row(
            listing_id="current",
            anchor=TARGET_DATE,
            value=None,
            current=True,
        ),
    ]
    range_only = build_medium_forecasts(_panel(rows), config)["current"]["6m"]
    positive_evidence = {
        "base_accuracy": {
            "status": "passed",
            "test_origins": 4,
            "test_predictions": 4,
            "weighting": "date_equal_listing_equal_within_origin",
            "mean_absolute_error": 0.1,
            "unconditional_mean_absolute_error": 0.2,
            "spy_relative_mean_absolute_error": 0.2,
            "spy_relative_baseline_method": (
                "market_regime_benchmark_median_plus_relative_momentum_excess_median"
            ),
            "maximum_baseline_mae_ratio": 1.0,
        },
        "probability_skill": {
            "status": "positive_skill",
            "test_origins": 4,
            "test_predictions": 4,
            "weighting": "date_equal_listing_equal_within_origin",
            "event": "return_gt_0",
            "model_brier_score": 0.1,
            "reference_brier_score": 0.2,
            "brier_skill_score": 0.5,
            "reference_method": "prequential_unconditional",
            "minimum_brier_skill_exclusive": 0.0,
            "zero_reference_policy": "null_no_epsilon",
        },
        "interval": {
            "status": "descriptive",
            "test_origins": 4,
            "test_predictions": 4,
            "weighting": "date_equal_listing_equal_within_origin",
            "alpha": 0.4,
            "nominal_coverage": 0.6,
            "endpoint_policy": "inclusive",
            "empirical_coverage": 0.6,
            "below_rate": 0.2,
            "above_rate": 0.2,
            "mean_width": 0.6,
            "model_mean_interval_score": 0.7,
            "reference_mean_interval_score": 0.8,
            "reference_method": "prequential_unconditional",
        },
    }
    monkeypatch.setattr(
        medium_forecasts,
        "_v2_prequential_evidence",
        lambda *_args, **_kwargs: deepcopy(positive_evidence),
    )
    published = build_medium_forecasts(_panel(rows), config)["current"]["6m"]
    insufficient_config = replace(
        config,
        horizons={
            **config.horizons,
            "6m": replace(config.horizons["6m"], minimum_raw_matches=99),
        },
    )
    insufficient = build_medium_forecasts(_panel(rows), insufficient_config)["current"]["6m"]
    expected_keys = {
        "schema_version",
        "method",
        "method_version",
        "forecast_horizon",
        "horizon_sessions",
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

    assert published.scenario.confidence_status == "empirical_skill_supported", (
        published.scenario.insufficiency_reason
    )
    assert range_only.scenario.confidence_status == "empirical_range_only"
    assert insufficient.scenario.confidence_status == "insufficient_evidence"
    assert all(
        set(forecast.calculation) == expected_keys
        for forecast in (published, range_only, insufficient)
    )
    assert range_only.calculation["predictive_distribution"]["probability_positive_raw"] is not None
    assert range_only.scenario.probability_positive is None
    assert insufficient.calculation["predictive_distribution"]["probability_positive_raw"] is None
    forbidden = {"unconditional_baseline", "calibration", "return_floor"}
    assert all(
        not (set(forecast.calculation) & forbidden)
        for forecast in (published, range_only, insufficient)
    )


@pytest.mark.django_db
def test_ui_accepts_reviewed_published_range_only_and_insufficient_states(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, _snapshot, results = persisted_v2
    range_only = results[0].analysis.six_month_forecast_scenario
    assert range_only["confidence_status"] == "empirical_range_only"
    assert validated_medium_forecast_scenario(range_only, "6m") is range_only

    config = _v2_config()
    insufficient = build_medium_forecasts(
        _panel(
            [
                _panel_row(
                    listing_id="only",
                    anchor=date(2020, 1, 2),
                    value=0.1,
                ),
                _panel_row(
                    listing_id="insufficient",
                    anchor=TARGET_DATE,
                    value=None,
                    current=True,
                ),
            ]
        ),
        config,
    )["insufficient"]["6m"]
    insufficient_payload = insufficient.scenario_payload()
    insufficient_interval = insufficient.calculation["evidence"]["interval"]
    assert insufficient.scenario.confidence_status == "insufficient_evidence"
    assert all(
        insufficient_interval[key] is None
        for key in (
            "empirical_coverage",
            "below_rate",
            "above_rate",
            "mean_width",
            "model_mean_interval_score",
            "reference_mean_interval_score",
        )
    )
    assert validated_medium_forecast_scenario(insufficient_payload, "6m") is (insufficient_payload)

    evidence = {
        "base_accuracy": {
            "status": "passed",
            "test_origins": 4,
            "test_predictions": 4,
            "weighting": "date_equal_listing_equal_within_origin",
            "mean_absolute_error": 0.1,
            "unconditional_mean_absolute_error": 0.2,
            "spy_relative_mean_absolute_error": 0.2,
            "spy_relative_baseline_method": (
                "market_regime_benchmark_median_plus_relative_momentum_excess_median"
            ),
            "maximum_baseline_mae_ratio": 1.0,
        },
        "probability_skill": {
            "status": "positive_skill",
            "test_origins": 4,
            "test_predictions": 4,
            "weighting": "date_equal_listing_equal_within_origin",
            "event": "return_gt_0",
            "model_brier_score": 0.1,
            "reference_brier_score": 0.2,
            "brier_skill_score": 0.5,
            "reference_method": "prequential_unconditional",
            "minimum_brier_skill_exclusive": 0.0,
            "zero_reference_policy": "null_no_epsilon",
        },
        "interval": {
            "status": "descriptive",
            "test_origins": 4,
            "test_predictions": 4,
            "weighting": "date_equal_listing_equal_within_origin",
            "alpha": 0.4,
            "nominal_coverage": 0.6,
            "endpoint_policy": "inclusive",
            "empirical_coverage": 0.6,
            "below_rate": 0.2,
            "above_rate": 0.2,
            "mean_width": 0.4,
            "model_mean_interval_score": 0.5,
            "reference_mean_interval_score": 0.6,
            "reference_method": "prequential_unconditional",
        },
    }
    monkeypatch.setattr(
        medium_forecasts,
        "_v2_prequential_evidence",
        lambda *_args, **_kwargs: deepcopy(evidence),
    )
    historical: list[dict[str, object]] = []
    for cohort in range(8):
        anchor = date(2018, 1, 2) + timedelta(days=183 * cohort)
        for listing_index in range(30):
            historical.append(
                _panel_row(
                    listing_id=f"P{listing_index:02d}",
                    anchor=anchor,
                    value=(-0.1, 0.0, 0.2)[listing_index % 3],
                    regime=cohort % 3,
                )
            )
    current = _panel_row(
        listing_id="published",
        anchor=TARGET_DATE,
        value=None,
        current=True,
        regime=9,
    )
    published = build_medium_forecasts(_panel([*historical, current]), config)["published"]["6m"]
    published_payload = published.scenario_payload()
    assert published.scenario.confidence_status == "empirical_skill_supported"
    assert validated_medium_forecast_scenario(published_payload, "6m") is published_payload

    partition_origin = {
        "covered": [1.0] * 3 + [0.0] * 25,
        "below": [1.0] * 8 + [0.0] * 20,
        "above": [1.0] * 17 + [0.0] * 11,
        "width": [0.4] * 28,
        "model_interval_score": [0.5] * 28,
        "reference_interval_score": [0.6] * 28,
    }
    calculator_interval = medium_forecasts._v2_interval_evidence(
        [partition_origin],
        config=config,
    )
    rates = [
        calculator_interval["empirical_coverage"],
        calculator_interval["below_rate"],
        calculator_interval["above_rate"],
    ]
    assert rates == [3 / 28, 8 / 28, 17 / 28]
    assert math.fsum(rates) == 0.9999999999999999
    partition_evidence = deepcopy(evidence)
    partition_evidence["base_accuracy"].update(
        {
            "status": "insufficient_support",
            "test_origins": 1,
            "test_predictions": 28,
        }
    )
    partition_evidence["probability_skill"].update(
        {
            "status": "not_evaluable",
            "test_origins": 0,
            "test_predictions": 0,
            "model_brier_score": None,
            "reference_brier_score": None,
            "brier_skill_score": None,
        }
    )
    partition_evidence["interval"] = calculator_interval
    monkeypatch.setattr(
        medium_forecasts,
        "_v2_prequential_evidence",
        lambda *_args, **_kwargs: deepcopy(partition_evidence),
    )
    partition_forecast = build_medium_forecasts(
        _panel([*historical, current]),
        config,
    )["published"]["6m"]
    partition_payload = partition_forecast.scenario_payload()
    assert partition_forecast.scenario.confidence_status == "empirical_range_only"
    assert validated_medium_forecast_scenario(partition_payload, "6m") is partition_payload

    forged = _mutated_v2_forecast(published, "mixture_probability")
    assert validated_medium_forecast_scenario(forged.scenario_payload(), "6m") == {
        "medium_v2_invalid": True
    }


@pytest.mark.django_db
def test_v2_selects_all_then_refuses_late_asset_with_zero_reads_and_no_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AssetStore(tmp_path)
    listing = _listing("LATEV2")
    sessions = _sessions(900)
    benchmark = _price_asset(
        store,
        subject="SPY",
        sessions=sessions,
        available_at=HISTORICAL_CUTOFF,
        retrieved_at=HISTORICAL_GENERATED_AT,
        suffix="benchmark",
    )
    older = _price_asset(
        store,
        subject=listing.ticker,
        sessions=sessions,
        available_at=HISTORICAL_CUTOFF - timedelta(days=1),
        retrieved_at=HISTORICAL_CUTOFF - timedelta(days=1),
        suffix="older",
    )
    late = _price_asset(
        store,
        subject=listing.ticker,
        sessions=sessions,
        available_at=HISTORICAL_CUTOFF + timedelta(microseconds=1),
        retrieved_at=HISTORICAL_GENERATED_AT,
        suffix="late",
    )
    selections: list[str] = []
    reads: list[str] = []
    original_select = AsOfData.latest_asset
    original_read = AsOfData.price_frame_for_asset_with_diagnostics

    def select(self: AsOfData, **kwargs: Any) -> DataAsset:
        selections.append(str(kwargs["subject"]))
        return original_select(self, **kwargs)

    def read(self: AsOfData, **kwargs: Any) -> Any:
        reads.append(str(kwargs["asset"].pk))
        return original_read(self, **kwargs)

    monkeypatch.setattr(AsOfData, "latest_asset", select)
    monkeypatch.setattr(AsOfData, "price_frame_for_asset_with_diagnostics", read)

    with pytest.raises(
        ValueError,
        match="refuses price assets available after AnalysisRun.data_cutoff",
    ):
        build_medium_forecast_panel(
            **_panel_kwargs(
                listings=[listing],
                store=store,
                data_cutoff=HISTORICAL_CUTOFF,
            )
        )

    assert selections == ["SPY", listing.ticker]
    assert reads == []
    assert DataAsset.objects.filter(pk=benchmark.pk).exists()
    assert DataAsset.objects.filter(pk=older.pk).exists()
    assert DataAsset.objects.filter(pk=late.pk).exists()
    assert DataAsset.objects.filter(kind="medium_forecast_panel").count() == 0
    assert not list(tmp_path.glob("derived/forecast/medium/**/*.parquet"))


@pytest.mark.django_db
def test_v2_cutoff_equality_reads_each_exact_asset_once_and_preserves_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AssetStore(tmp_path)
    listing = _listing("EQUALV2")
    sessions = [
        *_sessions(900),
        get_calendar("XNYS").next_session(TARGET_DATE).date(),
    ]
    assets = [
        _price_asset(
            store,
            subject=subject,
            sessions=sessions,
            available_at=HISTORICAL_CUTOFF,
            retrieved_at=HISTORICAL_GENERATED_AT,
            suffix=subject,
        )
        for subject in ("SPY", listing.ticker)
    ]
    selections: list[str] = []
    reads: list[str] = []
    original_select = AsOfData.latest_asset
    original_read = AsOfData.price_frame_for_asset_with_diagnostics

    def select(self: AsOfData, **kwargs: Any) -> DataAsset:
        selections.append(str(kwargs["subject"]))
        return original_select(self, **kwargs)

    def read(self: AsOfData, **kwargs: Any) -> Any:
        reads.append(str(kwargs["asset"].pk))
        return original_read(self, **kwargs)

    monkeypatch.setattr(AsOfData, "latest_asset", select)
    monkeypatch.setattr(AsOfData, "price_frame_for_asset_with_diagnostics", read)
    panel = build_medium_forecast_panel(
        **_panel_kwargs(
            listings=[listing],
            store=store,
            data_cutoff=HISTORICAL_CUTOFF,
        )
    )

    assert selections == ["SPY", listing.ticker]
    assert reads == [str(asset.pk) for asset in assets]
    assert all(
        source["retrieved_at"] == HISTORICAL_GENERATED_AT.isoformat()
        for source in panel.asset.metadata["source_assets"]
    )
    assert panel.frame["anchor_date"].max() <= TARGET_DATE
    assert panel.frame["label_end_date"].drop_nulls().max() <= TARGET_DATE
    assert panel.file_created_by_invocation is True


@pytest.mark.django_db
def test_v2_checksum_failure_creates_no_panel_output(tmp_path: Path) -> None:
    store = AssetStore(tmp_path)
    listing = _listing("CORRUPTV2")
    sessions = _sessions(900)
    _price_asset(
        store,
        subject="SPY",
        sessions=sessions,
        available_at=HISTORICAL_CUTOFF,
        retrieved_at=HISTORICAL_GENERATED_AT,
    )
    listing_asset = _price_asset(
        store,
        subject=listing.ticker,
        sessions=sessions,
        available_at=HISTORICAL_CUTOFF,
        retrieved_at=HISTORICAL_GENERATED_AT,
    )
    store.resolve(listing_asset.relative_path).write_bytes(b"corrupt")

    with pytest.raises(PriceFrameChecksumMismatchError):
        build_medium_forecast_panel(
            **_panel_kwargs(
                listings=[listing],
                store=store,
                data_cutoff=HISTORICAL_CUTOFF,
            )
        )

    assert DataAsset.objects.filter(kind="medium_forecast_panel").count() == 0
    assert not list(tmp_path.glob("derived/forecast/medium/**/*.parquet"))


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("issued_on_time", "benchmark", "message"),
    [
        (True, "SPY", "issued_on_time=False"),
        (None, "SPY", "issued_on_time=False"),
        (False, None, "benchmark_subject='SPY'"),
        (False, "QQQ", "benchmark_subject='SPY'"),
    ],
)
def test_v2_preflight_fails_before_provider_state_or_asset_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    django_assert_num_queries: Any,
    issued_on_time: bool | None,
    benchmark: str | None,
    message: str,
) -> None:
    universe = Universe.objects.create(
        slug=f"preflight-{uuid4().hex}",
        name="Preflight",
        config_version="test",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="u" * 64,
    )
    listing = _listing(f"P{uuid4().hex[:7]}")
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    paths = AnalysisOutputPaths(
        panel_relative_path="pre-existing-panel",
        manifest_relative_path="pre-existing-manifest",
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("preflight reached mutable/output-producing work")

    import stanstock.research.service as service_module

    monkeypatch.setattr(service_module, "open_asset_store", forbidden)
    monkeypatch.setattr(service_module.ProviderRecord.objects, "filter", forbidden)

    with django_assert_num_queries(0):
        with pytest.raises(ValueError, match=message):
            analyze_snapshot(
                universe_snapshot=snapshot,
                decision_time=GENERATED_AT,
                target_date=TARGET_DATE,
                issued_on_time=issued_on_time,
                provider="synthetic",
                benchmark_subject=benchmark,
                config_path=default_us_scoring_config_path(),
                medium_forecast_config_path=V2_PATH,
                output_paths=paths,
            )

    assert paths.panel_relative_path == "pre-existing-panel"
    assert paths.manifest_relative_path == "pre-existing-manifest"
    assert AnalysisRun.objects.count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize(
    "authority_mutation",
    [
        "stale_grade",
        "snapshot_date",
        "snapshot_config_between_projection_and_lock",
        "universe_between_projection_and_lock",
        "membership_eligibility",
        "membership_addition",
        "listing_region",
        "listing_currency",
        "provider_subject",
        "security_type",
    ],
)
def test_v2_analyze_rejects_locked_authority_changes_before_downstream_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority_mutation: str,
) -> None:
    """The saved snapshot PK is the caller's only v2 authority input."""
    import stanstock.research.service as service_module

    universe = Universe.objects.create(
        slug=f"locked-authority-{uuid4().hex}",
        name="Locked authority",
        config_version="test",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="u" * 64,
    )
    listing = _listing(f"A{uuid4().hex[:7]}")
    membership = UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    caller_snapshot = UniverseSnapshot.objects.select_related("universe").get(pk=snapshot.pk)
    assert caller_snapshot.grade == UniverseSnapshot.Grade.RESEARCH

    if authority_mutation == "stale_grade":
        UniverseSnapshot.objects.filter(pk=snapshot.pk).update(
            grade=UniverseSnapshot.Grade.OBSERVED
        )
    elif authority_mutation == "snapshot_date":
        UniverseSnapshot.objects.filter(pk=snapshot.pk).update(
            as_of_date=TARGET_DATE - timedelta(days=1)
        )
    elif authority_mutation == "membership_eligibility":
        UniverseMembership.objects.filter(pk=membership.pk).update(
            eligible=False,
            exclusion_reason="changed before lock",
        )
    elif authority_mutation == "membership_addition":
        added = _listing(f"N{uuid4().hex[:7]}", region=Region.EUROPE)
        UniverseMembership.objects.create(snapshot=snapshot, listing=added)
    elif authority_mutation == "listing_region":
        Listing.objects.filter(pk=listing.pk).update(region=Region.EUROPE)
    elif authority_mutation == "listing_currency":
        Listing.objects.filter(pk=listing.pk).update(currency="EUR")
    elif authority_mutation == "provider_subject":
        Listing.objects.filter(pk=listing.pk).update(provider_symbol="SPY")
    elif authority_mutation == "security_type":
        Security.objects.filter(pk=listing.security_id).update(
            security_type=Security.SecurityType.ETF
        )
    else:
        replacement_universe = Universe.objects.create(
            slug=f"replacement-{uuid4().hex}",
            name="Replacement",
            config_version="test",
        )
        original_select_for_update = Universe.objects.select_for_update
        changed = False

        def mutate_after_snapshot_projection(*args: Any, **kwargs: Any) -> Any:
            nonlocal changed
            if not changed:
                changed = True
                updates: dict[str, Any]
                if authority_mutation == "snapshot_config_between_projection_and_lock":
                    updates = {"config_hash": "c" * 64}
                else:
                    updates = {"universe_id": replacement_universe.pk}
                UniverseSnapshot.objects.filter(pk=snapshot.pk).update(**updates)
            return original_select_for_update(*args, **kwargs)

        monkeypatch.setattr(
            Universe.objects,
            "select_for_update",
            mutate_after_snapshot_projection,
        )

    paths = AnalysisOutputPaths(
        panel_relative_path="pre-existing-panel",
        manifest_relative_path="pre-existing-manifest",
    )
    supplied_store = AssetStore(tmp_path)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("invalid locked authority reached downstream work")

    monkeypatch.setattr(service_module.ProviderRecord.objects, "filter", forbidden)
    monkeypatch.setattr(service_module, "open_asset_store", forbidden)
    monkeypatch.setattr(service_module, "_create_analysis_run", forbidden)
    monkeypatch.setattr(supplied_store, "read_bytes", forbidden)
    monkeypatch.setattr(supplied_store, "read_frame", forbidden)
    monkeypatch.setattr(supplied_store, "write_bytes", forbidden)
    monkeypatch.setattr(supplied_store, "write_frame", forbidden)

    with pytest.raises(ValueError):
        analyze_snapshot(
            universe_snapshot=caller_snapshot,
            decision_time=GENERATED_AT,
            target_date=TARGET_DATE,
            issued_on_time=False,
            provider="synthetic",
            benchmark_subject="SPY",
            store=supplied_store,
            config_path=default_us_scoring_config_path(),
            medium_forecast_config_path=V2_PATH,
            output_paths=paths,
        )

    assert paths.panel_relative_path is None
    assert paths.manifest_relative_path is None
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert DataAsset.objects.filter(kind="medium_forecast_panel").count() == 0
    assert not list(tmp_path.glob("derived/forecast/medium/**/*.parquet"))


@pytest.mark.django_db
def test_v2_preflight_rejects_empty_wrong_currency_and_unsupported_security(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stanstock.research.service as service_module

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("invalid admission reached provider/output work")

    monkeypatch.setattr(service_module, "open_asset_store", forbidden)
    monkeypatch.setattr(service_module.ProviderRecord.objects, "filter", forbidden)
    for suffix, listing_kwargs, expected in (
        ("empty", None, "at least one eligible membership"),
        ("currency", {"currency": "EUR"}, "US/USD stock-research eligible"),
        (
            "etf",
            {"security_type": Security.SecurityType.ETF},
            "supports common stocks and depositary receipts",
        ),
    ):
        universe = Universe.objects.create(
            slug=f"preflight-{suffix}",
            name=suffix,
            config_version="test",
        )
        snapshot = UniverseSnapshot.objects.create(
            universe=universe,
            as_of_date=TARGET_DATE,
            grade=UniverseSnapshot.Grade.RESEARCH,
            config_hash="u" * 64,
        )
        if listing_kwargs is not None:
            listing = _listing(f"X{suffix}", **listing_kwargs)
            UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
        with pytest.raises(ValueError, match=expected):
            analyze_snapshot(
                universe_snapshot=snapshot,
                decision_time=GENERATED_AT,
                target_date=TARGET_DATE,
                issued_on_time=False,
                provider="synthetic",
                benchmark_subject="SPY",
                config_path=default_us_scoring_config_path(),
                medium_forecast_config_path=V2_PATH,
            )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("scoring_mutation", "message"),
    [
        ("v1", "scoring config us-price-baseline-v2"),
        (None, "reviewed us-price-baseline-v2 config identity"),
    ],
)
def test_v2_preflight_rejects_wrong_scoring_version_and_identity_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scoring_mutation: str | None,
    message: str,
) -> None:
    universe = Universe.objects.create(
        slug=f"scoring-preflight-{uuid4().hex}",
        name="Scoring preflight",
        config_version="test",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="u" * 64,
    )
    listing = _listing(f"S{uuid4().hex[:7]}")
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    scoring_path = tmp_path / "scoring.yml"
    if scoring_mutation is not None:
        scoring_path.write_text(
            Path("config/scoring/us-price-baseline-v1.yml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    else:
        scoring_path.write_text(
            default_us_scoring_config_path()
            .read_text(encoding="utf-8")
            .replace("buy_min_score: 72", "buy_min_score: 71"),
            encoding="utf-8",
        )

    import stanstock.research.service as service_module

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("invalid scoring reached output-producing work")

    monkeypatch.setattr(service_module, "open_asset_store", forbidden)
    monkeypatch.setattr(service_module.ProviderRecord.objects, "filter", forbidden)

    with pytest.raises(ValueError, match=message):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=GENERATED_AT,
            target_date=TARGET_DATE,
            issued_on_time=False,
            provider="synthetic",
            benchmark_subject="SPY",
            config_path=scoring_path,
            medium_forecast_config_path=V2_PATH,
        )

    assert AnalysisRun.objects.count() == 0


@pytest.mark.django_db
def test_v2_service_cutoff_refusal_rolls_back_with_zero_reads_and_preserves_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AssetStore(tmp_path)
    universe = Universe.objects.create(
        slug="v2-cutoff-atomic",
        name="V2 cutoff atomic",
        config_version="test",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="u" * 64,
    )
    listing = _listing("ATOMICV2")
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    sessions = _sessions(900)
    source_assets = [
        _price_asset(
            store,
            subject="SPY",
            sessions=sessions,
            available_at=HISTORICAL_CUTOFF,
            retrieved_at=HISTORICAL_GENERATED_AT,
            suffix="safe-benchmark",
        ),
        _price_asset(
            store,
            subject=listing.ticker,
            sessions=sessions,
            available_at=HISTORICAL_CUTOFF - timedelta(days=1),
            retrieved_at=HISTORICAL_CUTOFF - timedelta(days=1),
            suffix="safe-old",
        ),
        _price_asset(
            store,
            subject=listing.ticker,
            sessions=sessions,
            available_at=HISTORICAL_CUTOFF + timedelta(microseconds=1),
            retrieved_at=HISTORICAL_GENERATED_AT,
            suffix="unsafe-selected",
        ),
    ]
    before = {
        str(asset.pk): (
            asset.sha256,
            hashlib.sha256(store.read_bytes(asset.relative_path)).hexdigest(),
        )
        for asset in source_assets
    }
    reads: list[str] = []

    def record_read(*_args: Any, **kwargs: Any) -> Any:
        reads.append(str(kwargs["asset"].pk))
        raise AssertionError("cutoff refusal performed a physical read")

    monkeypatch.setattr(
        AsOfData,
        "price_frame_for_asset_with_diagnostics",
        record_read,
    )
    paths = AnalysisOutputPaths(
        panel_relative_path="old-panel",
        manifest_relative_path="old-manifest",
    )

    with pytest.raises(
        ValueError,
        match="refuses price assets available after AnalysisRun.data_cutoff",
    ):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=HISTORICAL_GENERATED_AT,
            target_date=TARGET_DATE,
            issued_on_time=False,
            provider="synthetic",
            benchmark_subject="SPY",
            store=store,
            config_path=default_us_scoring_config_path(),
            medium_forecast_config_path=V2_PATH,
            output_paths=paths,
            long_forecast_requested=False,
        )

    assert reads == []
    assert paths.panel_relative_path is None
    assert paths.manifest_relative_path is None
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert PredictionOutcome.objects.count() == 0
    assert DataAsset.objects.filter(kind="medium_forecast_panel").count() == 0
    assert {
        str(asset.pk): (
            asset.sha256,
            hashlib.sha256(store.read_bytes(asset.relative_path)).hexdigest(),
        )
        for asset in DataAsset.objects.filter(pk__in=[asset.pk for asset in source_assets])
    } == before


@pytest.mark.django_db
def test_v2_corrupt_reconstructed_return_fails_before_serialization_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AssetStore(tmp_path)
    universe = Universe.objects.create(
        slug="v2-return-atomic",
        name="V2 return atomic",
        config_version="test",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="u" * 64,
    )
    listing = _listing("RETURNV2")
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    sessions = _sessions(900)
    for subject in ("SPY", listing.ticker):
        _price_asset(
            store,
            subject=subject,
            sessions=sessions,
            available_at=GENERATED_AT,
            retrieved_at=GENERATED_AT,
            suffix="return-atomic",
        )
    original_reconstruct = medium_forecasts.reconstruct_medium_forecast_panel

    def corrupt_return(**kwargs: Any) -> pl.DataFrame:
        frame = original_reconstruct(**kwargs)
        return frame.with_columns(
            pl.when(~pl.col("is_forecast"))
            .then(pl.lit(-1.0001))
            .otherwise(pl.col("forward_return"))
            .alias("forward_return")
        )

    monkeypatch.setattr(
        medium_forecasts,
        "reconstruct_medium_forecast_panel",
        corrupt_return,
    )

    with pytest.raises(
        ValueError,
        match="refuses support returns below -1.0",
    ):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=GENERATED_AT,
            target_date=TARGET_DATE,
            issued_on_time=False,
            provider="synthetic",
            benchmark_subject="SPY",
            store=store,
            config_path=default_us_scoring_config_path(),
            medium_forecast_config_path=V2_PATH,
            long_forecast_requested=False,
        )

    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert DataAsset.objects.filter(kind="medium_forecast_panel").count() == 0
    assert not list(tmp_path.glob("derived/forecast/medium/**/*.parquet"))


def _e2e_universe(store: AssetStore, *, count: int = 12) -> UniverseSnapshot:
    universe = Universe.objects.create(
        slug=f"medium-v2-e2e-{uuid4().hex}",
        name="Medium v2 E2E",
        config_version="test",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="u" * 64,
    )
    sessions = _sessions(1500)
    _price_asset(
        store,
        subject="SPY",
        sessions=sessions,
        available_at=GENERATED_AT,
        retrieved_at=GENERATED_AT,
        slope=0.03,
        suffix="e2e",
    )
    for index in range(count):
        listing = _listing(
            f"V{index:03d}",
            security_type=(
                Security.SecurityType.ADR
                if index == count - 1
                else Security.SecurityType.COMMON_STOCK
            ),
        )
        UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
        _price_asset(
            store,
            subject=listing.ticker,
            sessions=sessions,
            available_at=GENERATED_AT,
            retrieved_at=GENERATED_AT,
            slope=0.022 + index * 0.0007,
            suffix="e2e",
        )
    return snapshot


@pytest.mark.django_db
def test_explicit_v1_snapshot_cannot_be_replaced_to_activate_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AssetStore(tmp_path / "assets")
    snapshot = _e2e_universe(store, count=1)
    path = tmp_path / "medium.yml"
    path.write_bytes(Path("config/forecasts/us-price-medium-v1.yml").read_bytes())
    replacement = V2_PATH.read_bytes()
    original_read_bytes = Path.read_bytes
    reads = 0

    def replace_after_read(candidate: Path) -> bytes:
        nonlocal reads
        payload = original_read_bytes(candidate)
        if candidate == path:
            reads += 1
            if reads == 1:
                candidate.write_bytes(replacement)
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=GENERATED_AT,
        target_date=TARGET_DATE,
        issued_on_time=False,
        provider="synthetic",
        benchmark_subject="SPY",
        store=store,
        config_path=default_us_scoring_config_path(),
        medium_forecast_config_path=path,
        long_forecast_requested=False,
    )

    assert reads == 1
    assert original_read_bytes(path) == replacement
    assert {
        prediction.method_version
        for prediction in Prediction.objects.filter(
            analysis=results[0].analysis,
            evidence_role=Prediction.EvidenceRole.ADVISORY,
        )
    } == {"us-price-medium-v1"}


@pytest.mark.django_db
@pytest.mark.parametrize(
    "initial",
    ["valid_v2", "partial_v2"],
)
def test_explicit_v2_snapshot_cannot_be_replaced_before_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    django_assert_num_queries: Any,
    initial: str,
) -> None:
    universe = Universe.objects.create(
        slug=f"immutable-v2-{initial}",
        name="Immutable v2 admission",
        config_version="test",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="u" * 64,
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=_listing(f"I{initial[:6]}"))
    path = tmp_path / "renamed.yml"
    path.write_bytes(
        V2_PATH.read_bytes() if initial == "valid_v2" else b"schema_version: 2\nbroken: [\n"
    )
    replacement = Path("config/forecasts/us-price-medium-v1.yml").read_bytes()
    original_read_bytes = Path.read_bytes
    reads = 0

    def replace_after_read(candidate: Path) -> bytes:
        nonlocal reads
        payload = original_read_bytes(candidate)
        if candidate == path:
            reads += 1
            if reads == 1:
                candidate.write_bytes(replacement)
        return payload

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("v2 config admission reached provider/store/output work")

    import stanstock.research.service as service_module

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    monkeypatch.setattr(service_module, "open_asset_store", forbidden)
    monkeypatch.setattr(service_module.ProviderRecord.objects, "filter", forbidden)
    paths = AnalysisOutputPaths(
        panel_relative_path="existing-panel",
        manifest_relative_path="existing-manifest",
    )
    message = "issued_on_time=False" if initial == "valid_v2" else "config is malformed"
    with django_assert_num_queries(0):
        with pytest.raises(ValueError, match=message):
            analyze_snapshot(
                universe_snapshot=snapshot,
                decision_time=GENERATED_AT,
                target_date=TARGET_DATE,
                issued_on_time=True,
                provider="synthetic",
                benchmark_subject="SPY",
                config_path=default_us_scoring_config_path(),
                medium_forecast_config_path=path,
                output_paths=paths,
            )

    assert reads == 1
    assert original_read_bytes(path) == replacement
    assert AnalysisRun.objects.count() == 0
    assert paths.panel_relative_path == "existing-panel"
    assert paths.manifest_relative_path == "existing-manifest"


@pytest.fixture
def persisted_v2(tmp_path: Path) -> tuple[AssetStore, UniverseSnapshot, list[Any]]:
    store = AssetStore(tmp_path)
    snapshot = _e2e_universe(store)
    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=GENERATED_AT,
        target_date=TARGET_DATE,
        issued_on_time=False,
        provider="synthetic",
        benchmark_subject="SPY",
        store=store,
        config_path=default_us_scoring_config_path(),
        medium_forecast_config_path=V2_PATH,
        long_forecast_requested=False,
    )
    return store, snapshot, results


@pytest.mark.django_db
def test_v2_persisted_three_states_have_20_keys_and_service_owned_provenance(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
) -> None:
    _store, snapshot, results = persisted_v2
    predictions = Prediction.objects.filter(
        method_version=MEDIUM_V2_VERSION,
    ).order_by("listing_id", "horizon")
    assert len(results) == 12
    assert predictions.count() == 24
    expected_service = {
        "config_hash",
        "prediction_version",
        "panel_asset_id",
        "panel_sha256",
        "evidence_grade",
        "price_subject",
    }
    for prediction in predictions:
        assert len(prediction.calculation) == 20
        assert expected_service <= set(prediction.calculation)
        assert prediction.config_hash == MEDIUM_V2_EFFECTIVE_CONFIG_HASH
        assert prediction.evidence_grade == UniverseSnapshot.Grade.RESEARCH
        assert prediction.issued_on_time is False
        assert prediction.analysis.run.universe_snapshot_id == snapshot.id
        formula = prediction.calculation["formula_inputs"]["scenario"]
        assert prediction.bear_return == (
            None
            if formula["bear"] is None
            else Decimal(str(formula["bear"])).quantize(Decimal("0.0001"))
        )
        assert prediction.probability_positive == (
            None
            if formula["probability_positive"] is None
            else Decimal(str(formula["probability_positive"])).quantize(Decimal("0.0001"))
        )
    assert all(
        result.analysis.recommendation == result.computation.recommendation for result in results
    )


def _forecast_from_prediction(prediction: Prediction) -> MediumForecast:
    calculation = {
        key: deepcopy(value)
        for key, value in prediction.calculation.items()
        if key
        not in {
            "config_hash",
            "prediction_version",
            "panel_asset_id",
            "panel_sha256",
            "evidence_grade",
            "price_subject",
        }
    }
    return MediumForecast(
        scenario=Scenario(
            bear=None if prediction.bear_return is None else float(prediction.bear_return),
            base=None if prediction.base_return is None else float(prediction.base_return),
            bull=None if prediction.bull_return is None else float(prediction.bull_return),
            probability_positive=(
                None
                if prediction.probability_positive is None
                else float(prediction.probability_positive)
            ),
            confidence=float(prediction.confidence),
            confidence_status=prediction.confidence_status,
            insufficiency_reason=prediction.insufficiency_reason,
            method="conditional_empirical_price",
        ),
        calculation=calculation,
    )


def _positive_v2_evidence() -> dict[str, Any]:
    return {
        "base_accuracy": {
            "status": "passed",
            "test_origins": 4,
            "test_predictions": 120,
            "weighting": "date_equal_listing_equal_within_origin",
            "mean_absolute_error": 0.1,
            "unconditional_mean_absolute_error": 0.2,
            "spy_relative_mean_absolute_error": 0.2,
            "spy_relative_baseline_method": (
                "market_regime_benchmark_median_plus_relative_momentum_excess_median"
            ),
            "maximum_baseline_mae_ratio": 1.0,
        },
        "probability_skill": {
            "status": "positive_skill",
            "test_origins": 4,
            "test_predictions": 120,
            "weighting": "date_equal_listing_equal_within_origin",
            "event": "return_gt_0",
            "model_brier_score": 0.1,
            "reference_brier_score": 0.2,
            "brier_skill_score": 0.5,
            "reference_method": "prequential_unconditional",
            "minimum_brier_skill_exclusive": 0.0,
            "zero_reference_policy": "null_no_epsilon",
        },
        "interval": {
            "status": "descriptive",
            "test_origins": 4,
            "test_predictions": 120,
            "weighting": "date_equal_listing_equal_within_origin",
            "alpha": 0.4,
            "nominal_coverage": 0.6,
            "endpoint_policy": "inclusive",
            "empirical_coverage": 0.6,
            "below_rate": 0.2,
            "above_rate": 0.2,
            "mean_width": 0.4,
            "model_mean_interval_score": 0.5,
            "reference_mean_interval_score": 0.6,
            "reference_method": "prequential_unconditional",
        },
    }


def _controlled_v2_panel_frame(
    *,
    listing_id: str,
    distinct_listings: int,
    current_eligible: bool = True,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in ("6m", "12m"):
        for cohort in range(8):
            anchor = date(2017 + cohort, 1, 3)
            for listing_index in range(distinct_listings):
                row = _panel_row(
                    listing_id=f"support-{listing_index:02d}",
                    anchor=anchor,
                    value=(-0.1, 0.0, 0.2)[listing_index % 3],
                    regime=cohort % 3,
                )
                row["horizon"] = horizon
                row["cohort_id"] = f"{horizon}:{anchor.isoformat()}"
                rows.append(row)
        current = _panel_row(
            listing_id=listing_id,
            anchor=TARGET_DATE,
            value=None,
            current=True,
            regime=9,
        )
        current["horizon"] = horizon
        current["cohort_id"] = f"{horizon}:{TARGET_DATE.isoformat()}"
        if not current_eligible:
            current["eligible"] = False
            current["insufficiency_reason"] = "Controlled missing current inputs"
        rows.append(current)
    return _panel(rows)


def _store_controlled_v2_panel(
    *,
    store: AssetStore,
    original: DataAsset,
    frame: pl.DataFrame,
    suffix: str,
    source_manifest_mutator: Any | None = None,
    metadata_mutator: Any | None = None,
) -> DataAsset:
    payload = serialize_medium_forecast_panel(frame)
    run_id = UUID(original.subject)
    assert original.period_end is not None
    period_start = frame["anchor_date"].min()
    assert isinstance(period_start, date)
    stored = store.write_bytes(
        f"derived/forecast/medium/{original.period_end.isoformat()}/"
        f"{run_id.hex}-{hashlib.sha256(payload).hexdigest()[:12]}.parquet",
        payload,
    )
    metadata = deepcopy(original.metadata)
    metadata["row_count"] = frame.height
    metadata["content_sha256"] = stored.sha256
    if source_manifest_mutator is not None:
        source_manifest_mutator(metadata["source_assets"])
    metadata["source_manifest_hash"] = hash_json(metadata["source_assets"])
    metadata["evidence_bundle_hash"] = hash_json(
        {
            "calendar_hash": metadata["calendar_hash"],
            "code_revision": metadata["code_revision"],
            "content_sha256": stored.sha256,
            "forecast_config_hash": metadata["config_hash"],
            "scoring_config_hash": metadata["scoring_config_hash"],
            "source_manifest_hash": metadata["source_manifest_hash"],
            "universe_config_hash": metadata["universe_config_hash"],
        }
    )
    if metadata_mutator is not None:
        metadata_mutator(metadata)
    return register_asset(
        provider=original.provider,
        kind=original.kind,
        subject=original.subject,
        stored=stored,
        retrieved_at=original.retrieved_at,
        available_at=original.available_at,
        period_start=period_start,
        period_end=original.period_end,
        metadata=metadata,
    )


def _regime_price_asset(
    store: AssetStore,
    *,
    subject: str,
    sessions: list[date],
    alpha: float,
    suffix: str,
    volume: int = 2_000_000,
) -> DataAsset:
    price = 100.0
    closes: list[float] = []
    for index, _session in enumerate(sessions):
        phase = (index // 252) % 4
        if phase == 0:
            daily_return = 0.0015
        elif phase == 1:
            daily_return = -0.001
        elif phase == 2:
            daily_return = 0.0002 + (0.014 if index % 2 else -0.013)
        else:
            daily_return = 0.0001
        price *= 1.0 + daily_return + alpha
        closes.append(price)
    frame = pl.DataFrame(
        {
            "date": sessions,
            "close": closes,
            "volume": [volume for _session in sessions],
        },
        schema_overrides={"date": pl.Date, "close": pl.Float64, "volume": pl.Int64},
    )
    stored = store.write_frame(f"medium-v2/{subject}-{suffix}.parquet", frame)
    return register_asset(
        provider="synthetic",
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=GENERATED_AT,
        available_at=GENERATED_AT,
        period_start=sessions[0],
        period_end=sessions[-1],
        metadata={
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )


def _matching_v2_writer_control(
    tmp_path: Path,
    *,
    distinct_listings: int,
    low_liquidity: bool = False,
) -> dict[str, Any]:
    store = AssetStore(tmp_path)
    universe = Universe.objects.create(
        slug=f"matching-v2-{uuid4().hex}",
        name="Matching v2 writer control",
        config_version="test",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="u" * 64,
    )
    config = _v2_config()
    calendar = get_calendar(config.calendar)
    sessions = [
        session.date() for session in calendar.sessions_in_range(config.fixed_epoch, TARGET_DATE)
    ]
    volume = 1 if low_liquidity else 2_000_000
    benchmark = _regime_price_asset(
        store,
        subject="SPY",
        sessions=sessions,
        alpha=0.0,
        suffix=uuid4().hex,
        volume=volume,
    )
    listings: list[Listing] = []
    listing_assets: list[DataAsset] = []
    for index in range(distinct_listings):
        listing = _listing(f"M{index:03d}{uuid4().hex[:4]}")
        listings.append(listing)
        UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
        listing_assets.append(
            _regime_price_asset(
                store,
                subject=listing.ticker,
                sessions=sessions,
                alpha=-0.0008 if index == 0 else 0.0,
                suffix=uuid4().hex,
                volume=volume,
            )
        )
    run = AnalysisRun.objects.create(
        generated_at=GENERATED_AT,
        data_cutoff=GENERATED_AT,
        target_date=TARGET_DATE,
        issued_on_time=False,
        universe_snapshot=snapshot,
        config_version="us-price-baseline-v2",
        config_hash="43cc0ee0e29f79dec4ad8d8a91e43e7b3df1d368dbc385a116fcc6ff05f18e9b",
        code_revision="test-revision",
    )
    panel = build_medium_forecast_panel(
        listings=listings,
        asof=AsOfData(GENERATED_AT, store),
        provider="synthetic",
        benchmark_subject="SPY",
        target_date=TARGET_DATE,
        generated_at=GENERATED_AT,
        run_id=run.id,
        config=config,
        config_hash=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
        scoring_config_version=run.config_version,
        scoring_config_hash=run.config_hash,
        universe_snapshot_id=snapshot.id,
        universe_slug=universe.slug,
        universe_config_hash=snapshot.config_hash,
        code_revision=run.code_revision,
        store=store,
        data_cutoff=run.data_cutoff,
    )
    selected_listing = listings[0]
    selected_asset = listing_assets[0]
    source_assets = [_asset_payload(selected_asset), _asset_payload(benchmark)]
    analysis = StockAnalysis.objects.create(
        run=run,
        listing=selected_listing,
        current_price=Decimal("100"),
        daily_change=Decimal("0"),
        overall_score=Decimal("50"),
        recommendation="hold",
        risk_score=Decimal("50"),
        risk_class="medium",
        confidence=Decimal("50"),
        confidence_status="heuristic",
        component_scores={},
        forecast_scenarios={},
        short_scenario={},
        medium_scenario={},
        long_scenario={},
        reasons=[],
        risks=[],
        data_quality={
            "price_source": {
                "asset_id": str(selected_asset.id),
                "provider": selected_asset.provider,
                "subject": selected_asset.subject,
            },
            "source_assets": source_assets,
        },
    )
    all_forecasts = build_medium_forecasts(panel.frame, config)
    return {
        "store": store,
        "analysis": analysis,
        "panel": panel.asset,
        "frame": panel.frame,
        "forecasts": all_forecasts[str(selected_listing.id)],
        "all_forecasts": all_forecasts,
        "listings": listings,
        "listing_assets": listing_assets,
        "benchmark_asset": benchmark,
        "source_assets": source_assets,
        "model_version": f"{MEDIUM_V2_VERSION}-{run.id.hex[:8]}",
    }


def _additional_matching_analysis(
    control: dict[str, Any],
    *,
    listing_index: int = 1,
) -> tuple[StockAnalysis, dict[str, MediumForecast], list[dict[str, Any]]]:
    listing = control["listings"][listing_index]
    listing_asset = control["listing_assets"][listing_index]
    source_assets = [
        _asset_payload(listing_asset),
        _asset_payload(control["benchmark_asset"]),
    ]
    analysis = StockAnalysis.objects.create(
        run=control["analysis"].run,
        listing=listing,
        current_price=Decimal("101"),
        daily_change=Decimal("0"),
        overall_score=Decimal("51"),
        recommendation="hold",
        risk_score=Decimal("49"),
        risk_class="medium",
        confidence=Decimal("52"),
        confidence_status="heuristic",
        component_scores={"additional": listing_index},
        forecast_scenarios={},
        short_scenario={},
        medium_scenario={},
        long_scenario={},
        reasons=[],
        risks=[],
        data_quality={
            "price_source": {
                "asset_id": str(listing_asset.id),
                "provider": listing_asset.provider,
                "subject": listing_asset.subject,
            },
            "source_assets": source_assets,
        },
    )
    return (
        analysis,
        control["all_forecasts"][str(listing.id)],
        source_assets,
    )


def _append_control_predictions(
    control: dict[str, Any],
    *,
    additional: tuple[
        StockAnalysis,
        dict[str, MediumForecast],
        list[dict[str, Any]],
    ]
    | None = None,
) -> tuple[Prediction, ...]:
    import stanstock.research.service as service_module

    analysis = control["analysis"]
    run = analysis.run
    additional_requests: tuple[service_module._AdditionalAdvisoryPredictionRequest, ...] = ()
    if additional is not None:
        additional_requests = (
            service_module._AdditionalAdvisoryPredictionRequest(
                analysis=additional[0],
                forecasts=additional[1],
                source_assets=additional[2],
            ),
        )
    return append_advisory_predictions(
        analysis=analysis,
        forecasts=control["forecasts"],
        panel_asset=control["panel"],
        store=control["store"],
        generated_at=run.generated_at,
        data_cutoff=run.data_cutoff,
        issued_on_time=False,
        model_version=control["model_version"],
        config_hash_value=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
        source_assets=control["source_assets"],
        code_revision_value=run.code_revision,
        _additional_requests=additional_requests,
    )


def _prediction_set() -> tuple[tuple[Any, ...], ...]:
    return tuple(
        Prediction.objects.order_by("pk").values_list(
            "pk",
            "analysis_id",
            "listing_id",
            "target_date",
            "horizon",
            "model_version",
        )
    )


def _invoke_v2_writer(
    *,
    entry_point: str,
    analysis: StockAnalysis,
    forecasts: dict[str, MediumForecast],
    panel: DataAsset,
    store: AssetStore,
    source_assets_without_panel: list[dict[str, Any]],
    model_version: str,
) -> None:
    run = analysis.run
    if entry_point == "append":
        append_advisory_predictions(
            analysis=analysis,
            forecasts=forecasts,
            panel_asset=panel,
            store=store,
            generated_at=run.generated_at,
            data_cutoff=run.data_cutoff,
            issued_on_time=False,
            model_version=model_version,
            config_hash_value=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
            source_assets=source_assets_without_panel,
            code_revision_value=run.code_revision,
        )
        return
    _create_advisory_prediction(
        analysis=analysis,
        horizon=Prediction.Horizon.SIX_MONTH,
        forecast=forecasts["6m"],
        panel_asset=panel,
        store=store,
        generated_at=run.generated_at,
        data_cutoff=run.data_cutoff,
        issued_on_time=False,
        model_version=model_version,
        config_hash_value=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
        source_assets=[*source_assets_without_panel, _asset_payload(panel)],
        code_revision_value=run.code_revision,
    )


def _incompatible_analysis_for_v2_spoof(source: StockAnalysis) -> StockAnalysis:
    universe = Universe.objects.create(
        slug=f"incompatible-v2-{uuid4().hex}",
        name="Incompatible v2 writer authority",
        config_version="test",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=source.run.target_date,
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="i" * 64,
    )
    run = AnalysisRun.objects.create(
        generated_at=source.run.generated_at - timedelta(days=1),
        data_cutoff=source.run.data_cutoff - timedelta(days=1),
        target_date=source.run.target_date,
        issued_on_time=True,
        universe_snapshot=snapshot,
        config_version="incompatible-scoring",
        config_hash="i" * 64,
        code_revision="incompatible-revision",
    )
    return StockAnalysis.objects.create(
        run=run,
        listing=source.listing,
        current_price=Decimal("12.345678"),
        daily_change=Decimal("-0.125000"),
        overall_score=Decimal("12.34"),
        recommendation="avoid",
        risk_score=Decimal("88.00"),
        risk_class="high",
        confidence=Decimal("11.00"),
        confidence_status="incompatible",
        component_scores={"incompatible": True},
        forecast_scenarios={},
        short_scenario={},
        medium_scenario={},
        long_scenario={},
        reasons=["incompatible"],
        risks=["incompatible"],
        data_quality=deepcopy(source.data_quality),
    )


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["append", "create"])
def test_v2_writers_reject_authoritative_run_relation_cache_spoof_before_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
) -> None:
    control = _matching_v2_writer_control(tmp_path, distinct_listings=1)
    run_b = control["analysis"].run
    analysis_a = _incompatible_analysis_for_v2_spoof(control["analysis"])
    persisted_run_a_id = analysis_a.run_id
    analysis_a._state.fields_cache["run"] = run_b
    assert analysis_a.run is run_b
    assert analysis_a.run_id == persisted_run_a_id != run_b.id

    created: list[dict[str, Any]] = []

    def capture_create(**kwargs: Any) -> Prediction:
        created.append(kwargs)
        return Prediction()

    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    before = Prediction.objects.count()
    with pytest.raises(ValueError, match="prediction payload is malformed"):
        _invoke_v2_writer(
            entry_point=entry_point,
            analysis=analysis_a,
            forecasts=control["forecasts"],
            panel=control["panel"],
            store=control["store"],
            source_assets_without_panel=control["source_assets"],
            model_version=control["model_version"],
        )

    assert created == []
    assert Prediction.objects.count() == before


@pytest.mark.django_db
def test_v2_append_resolves_additional_analysis_authority_before_shared_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stanstock.research.service as service_module

    control = _matching_v2_writer_control(tmp_path, distinct_listings=1)
    primary = control["analysis"]
    run_b = primary.run
    additional = _incompatible_analysis_for_v2_spoof(primary)
    persisted_additional_run_id = additional.run_id
    additional._state.fields_cache["run"] = run_b
    assert additional.run is run_b
    assert additional.run_id == persisted_additional_run_id != run_b.id

    attested: list[str] = []
    created: list[dict[str, Any]] = []
    real_attest = service_module._attest_medium_v2_panel_with_authority

    def count_attestation(**kwargs: Any) -> Any:
        attested.append(str(kwargs["run"].pk))
        return real_attest(**kwargs)

    def capture_create(**kwargs: Any) -> Prediction:
        created.append(kwargs)
        return Prediction()

    monkeypatch.setattr(
        service_module,
        "_attest_medium_v2_panel_with_authority",
        count_attestation,
    )
    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    before = Prediction.objects.count()
    with pytest.raises(ValueError, match="prediction payload is malformed"):
        append_advisory_predictions(
            analysis=primary,
            forecasts=control["forecasts"],
            panel_asset=control["panel"],
            store=control["store"],
            generated_at=run_b.generated_at,
            data_cutoff=run_b.data_cutoff,
            issued_on_time=False,
            model_version=control["model_version"],
            config_hash_value=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
            source_assets=control["source_assets"],
            code_revision_value=run_b.code_revision,
            _additional_requests=(
                service_module._AdditionalAdvisoryPredictionRequest(
                    analysis=additional,
                    forecasts=control["forecasts"],
                    source_assets=control["source_assets"],
                ),
            ),
        )

    assert attested == []
    assert created == []
    assert Prediction.objects.count() == before


@pytest.mark.django_db
def test_v2_panel_attestation_resolves_authoritative_run_by_primary_key(
    tmp_path: Path,
) -> None:
    import stanstock.research.service as service_module

    control = _matching_v2_writer_control(tmp_path, distinct_listings=1)
    run = control["analysis"].run
    authoritative = AnalysisRun.objects.select_related("universe_snapshot").get(pk=run.pk)
    incompatible_universe = Universe.objects.create(
        slug=f"attest-run-{uuid4().hex}",
        name="Attestation run cache spoof",
        config_version="test",
    )
    incompatible_snapshot = UniverseSnapshot.objects.create(
        universe=incompatible_universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="a" * 64,
    )

    run.generated_at -= timedelta(days=10)
    run.data_cutoff -= timedelta(days=10)
    run.target_date -= timedelta(days=10)
    run.config_version = "caller-mutated"
    run.config_hash = "a" * 64
    run.code_revision = "caller-mutated"
    run._state.fields_cache["universe_snapshot"] = incompatible_snapshot

    attestation = service_module._attest_medium_v2_panel(
        run=run,
        panel_asset=control["panel"],
        store=control["store"],
    )

    assert attestation.run_id == authoritative.id
    assert attestation.snapshot_id == authoritative.universe_snapshot_id
    assert attestation.config_hash == MEDIUM_V2_EFFECTIVE_CONFIG_HASH


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["append", "create"])
def test_v2_writers_copy_only_authoritative_analysis_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
) -> None:
    control = _matching_v2_writer_control(tmp_path, distinct_listings=1)
    caller = control["analysis"]
    authoritative = StockAnalysis.objects.select_related(
        "run__universe_snapshot__universe",
        "listing__security__company",
    ).get(pk=caller.pk)
    expected_component_scores = deepcopy(authoritative.component_scores)
    incompatible_listing = _listing(
        f"ETF{uuid4().hex[:6]}",
        security_type=Security.SecurityType.ETF,
    )
    observed_universe = Universe.objects.create(
        slug=f"observed-cache-{uuid4().hex}",
        name="Observed cache spoof",
        config_version="test",
    )
    observed_snapshot = UniverseSnapshot.objects.create(
        universe=observed_universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="o" * 64,
    )

    caller.listing = incompatible_listing
    caller.current_price = Decimal("999999.999999")
    caller.recommendation = "buy"
    caller.overall_score = Decimal("99.99")
    caller.component_scores = {"caller_mutated": True}
    caller.data_quality = {
        "price_source": {
            "asset_id": str(uuid4()),
            "provider": "caller-mutated",
            "subject": incompatible_listing.ticker,
        },
        "source_assets": [],
    }
    caller.run._state.fields_cache["universe_snapshot"] = observed_snapshot

    created: list[dict[str, Any]] = []

    def capture_create(**kwargs: Any) -> Prediction:
        created.append(kwargs)
        return Prediction()

    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    before = Prediction.objects.count()
    _invoke_v2_writer(
        entry_point=entry_point,
        analysis=caller,
        forecasts=control["forecasts"],
        panel=control["panel"],
        store=control["store"],
        source_assets_without_panel=control["source_assets"],
        model_version=control["model_version"],
    )

    assert len(created) == (2 if entry_point == "append" else 1)
    for kwargs in created:
        resolved = kwargs["analysis"]
        assert resolved is not caller
        assert resolved.pk == authoritative.pk
        assert resolved.run_id == authoritative.run_id
        assert kwargs["listing"].pk == authoritative.listing_id
        assert kwargs["price_at_prediction"] == authoritative.current_price
        assert kwargs["recommendation"] == authoritative.recommendation
        assert kwargs["overall_score"] == authoritative.overall_score
        assert kwargs["component_scores"] == expected_component_scores
        assert kwargs["target_date"] == authoritative.run.target_date
        assert kwargs["evidence_grade"] == UniverseSnapshot.Grade.RESEARCH
        assert kwargs["price_subject"] == (
            authoritative.listing.provider_symbol or authoritative.listing.ticker
        )
    assert Prediction.objects.count() == before


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["append", "create"])
@pytest.mark.parametrize("analysis_state", ["unsaved", "deleted"])
def test_v2_writers_fail_closed_for_unsaved_and_deleted_analyses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
    analysis_state: str,
) -> None:
    control = _matching_v2_writer_control(tmp_path, distinct_listings=1)
    source = control["analysis"]
    if analysis_state == "unsaved":
        candidate = StockAnalysis(run=source.run, listing=source.listing)
        assert candidate.pk is None
    else:
        candidate = StockAnalysis.objects.select_related("run", "listing").get(pk=source.pk)
        candidate_pk = candidate.pk
        StockAnalysis.objects.filter(pk=candidate_pk).delete()
        assert candidate.pk == candidate_pk
        assert not StockAnalysis.objects.filter(pk=candidate_pk).exists()

    created: list[dict[str, Any]] = []

    def capture_create(**kwargs: Any) -> Prediction:
        created.append(kwargs)
        return Prediction()

    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    before = Prediction.objects.count()
    with pytest.raises(ValueError, match="prediction payload is malformed"):
        _invoke_v2_writer(
            entry_point=entry_point,
            analysis=candidate,
            forecasts=control["forecasts"],
            panel=control["panel"],
            store=control["store"],
            source_assets_without_panel=control["source_assets"],
            model_version=control["model_version"],
        )

    assert created == []
    assert Prediction.objects.count() == before


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["append", "create"])
@pytest.mark.parametrize(
    "authority_mutation",
    [
        "run_cutoff",
        "analysis_source",
        "snapshot_grade_config",
        "listing_identity",
        "existing_membership",
        "new_membership",
    ],
)
def test_v2_writers_observe_authority_changes_made_before_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
    authority_mutation: str,
) -> None:
    """SQLite/local proof that stale caller instances never authenticate."""
    control = _matching_v2_writer_control(tmp_path, distinct_listings=2)
    analysis = control["analysis"]
    run = analysis.run
    snapshot = run.universe_snapshot
    listing = analysis.listing
    membership = UniverseMembership.objects.get(snapshot=snapshot, listing=listing)
    cleanup: Any

    if authority_mutation == "run_cutoff":
        original = run.data_cutoff
        AnalysisRun.objects.filter(pk=run.pk).update(
            data_cutoff=original - timedelta(microseconds=1)
        )
        cleanup = partial(
            AnalysisRun.objects.filter(pk=run.pk).update,
            data_cutoff=original,
        )
    elif authority_mutation == "analysis_source":
        original = deepcopy(analysis.data_quality)
        StockAnalysis.objects.filter(pk=analysis.pk).update(
            data_quality={
                **original,
                "price_source": {
                    **original["price_source"],
                    "subject": "CHANGED-BEFORE-LOCK",
                },
            }
        )
        cleanup = partial(
            StockAnalysis.objects.filter(pk=analysis.pk).update,
            data_quality=original,
        )
    elif authority_mutation == "snapshot_grade_config":
        original = (snapshot.grade, snapshot.config_hash)
        UniverseSnapshot.objects.filter(pk=snapshot.pk).update(
            grade=UniverseSnapshot.Grade.OBSERVED,
            config_hash="c" * 64,
        )
        cleanup = partial(
            UniverseSnapshot.objects.filter(pk=snapshot.pk).update,
            grade=original[0],
            config_hash=original[1],
        )
    elif authority_mutation == "listing_identity":
        original = listing.provider_symbol
        Listing.objects.filter(pk=listing.pk).update(provider_symbol="CHANGED-BEFORE-LOCK")
        cleanup = partial(
            Listing.objects.filter(pk=listing.pk).update,
            provider_symbol=original,
        )
    elif authority_mutation == "existing_membership":
        original = (membership.eligible, membership.exclusion_reason)
        UniverseMembership.objects.filter(pk=membership.pk).update(
            eligible=False,
            exclusion_reason="changed before lock",
        )
        cleanup = partial(
            UniverseMembership.objects.filter(pk=membership.pk).update,
            eligible=original[0],
            exclusion_reason=original[1],
        )
    else:
        added = UniverseMembership.objects.create(
            snapshot=snapshot,
            listing=_listing(f"NEW{uuid4().hex[:5]}"),
        )
        cleanup = partial(UniverseMembership.objects.filter(pk=added.pk).delete)

    created: list[dict[str, Any]] = []

    def capture_create(**kwargs: Any) -> Prediction:
        created.append(kwargs)
        return Prediction()

    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    try:
        with pytest.raises(ValueError, match="prediction payload is malformed"):
            _invoke_v2_writer(
                entry_point=entry_point,
                analysis=analysis,
                forecasts=control["forecasts"],
                panel=control["panel"],
                store=control["store"],
                source_assets_without_panel=control["source_assets"],
                model_version=control["model_version"],
            )
    finally:
        cleanup()

    assert created == []
    assert not Prediction.objects.exists()


@pytest.mark.django_db
def test_v2_append_observes_changed_additional_analysis_before_lock(
    tmp_path: Path,
) -> None:
    control = _matching_v2_writer_control(tmp_path, distinct_listings=2)
    additional = _additional_matching_analysis(control)
    original = deepcopy(additional[0].data_quality)
    StockAnalysis.objects.filter(pk=additional[0].pk).update(
        data_quality={
            **original,
            "price_source": {
                **original["price_source"],
                "subject": "CHANGED-ADDITIONAL-BEFORE-LOCK",
            },
        }
    )
    try:
        with pytest.raises(ValueError, match="prediction payload is malformed"):
            _append_control_predictions(control, additional=additional)
    finally:
        StockAnalysis.objects.filter(pk=additional[0].pk).update(data_quality=original)

    assert not Prediction.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    "conflict_target",
    ["final_primary_horizon", "final_additional_listing"],
)
def test_append_advisory_predictions_rolls_back_complete_direct_batch_on_late_conflict(
    tmp_path: Path,
    conflict_target: str,
) -> None:
    """No caller transaction is needed for all-or-nothing append behavior."""
    control = _matching_v2_writer_control(
        tmp_path,
        distinct_listings=2 if conflict_target == "final_additional_listing" else 1,
    )
    additional = (
        _additional_matching_analysis(control)
        if conflict_target == "final_additional_listing"
        else None
    )
    conflict_analysis = additional[0] if additional is not None else control["analysis"]
    conflict_forecasts = additional[1] if additional is not None else control["forecasts"]
    conflict_sources = additional[2] if additional is not None else control["source_assets"]
    run = conflict_analysis.run
    _create_advisory_prediction(
        analysis=conflict_analysis,
        horizon=Prediction.Horizon.TWELVE_MONTH,
        forecast=conflict_forecasts["12m"],
        panel_asset=control["panel"],
        store=control["store"],
        generated_at=run.generated_at,
        data_cutoff=run.data_cutoff,
        issued_on_time=False,
        model_version=control["model_version"],
        config_hash_value=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
        source_assets=[*conflict_sources, _asset_payload(control["panel"])],
        code_revision_value=run.code_revision,
    )
    before = _prediction_set()
    assert len(before) == 1

    with pytest.raises(IntegrityError):
        _append_control_predictions(control, additional=additional)

    assert _prediction_set() == before


def _rolled_back_concurrent_mutation(
    operation: Any,
    *,
    backend_pid: list[int],
    started: threading.Event,
    done: threading.Event,
    errors: list[BaseException],
) -> None:
    connections.close_all()
    try:
        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                row = cursor.fetchone()
            assert row is not None
            backend_pid.append(int(row[0]))
            started.set()
            result = operation()
            if isinstance(result, int):
                assert result == 1
            else:
                assert result is not None
            transaction.set_rollback(True)
    except BaseException as exc:  # pragma: no cover - asserted by the caller
        errors.append(exc)
    finally:
        done.set()
        connections.close_all()


def _wait_for_postgresql_lock_wait(backend_pid: int, *, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT wait_event_type
                FROM pg_stat_activity
                WHERE pid = %s
                """,
                [backend_pid],
            )
            row = cursor.fetchone()
        if row is not None and row[0] == "Lock":
            return
        time.sleep(0.02)
    pytest.fail(f"PostgreSQL backend {backend_pid} did not reach a lock wait")


def _insert_membership_with_immediate_fk_check(
    *,
    snapshot_id: UUID,
    listing: Listing,
) -> UniverseMembership:
    membership = UniverseMembership.objects.create(
        snapshot_id=snapshot_id,
        listing=listing,
        eligible=True,
    )
    # Django's PostgreSQL foreign keys are deferred. Make this transaction
    # perform the same parent-key check a successful commit must perform
    # before deliberately rolling the test mutation back.
    with connection.cursor() as cursor:
        cursor.execute("SET CONSTRAINTS ALL IMMEDIATE")
    return membership


def _advisory_parent_state(
    control: dict[str, Any],
    additional: tuple[
        StockAnalysis,
        dict[str, MediumForecast],
        list[dict[str, Any]],
    ]
    | None,
) -> dict[str, tuple[tuple[Any, ...], ...]]:
    run = control["analysis"].run
    analysis_ids = [control["analysis"].pk]
    if additional is not None:
        analysis_ids.append(additional[0].pk)
    return {
        "universe": tuple(
            Universe.objects.filter(pk=run.universe_snapshot.universe_id).values_list(
                "pk",
                "name",
                "config_version",
            )
        ),
        "run": tuple(
            AnalysisRun.objects.filter(pk=run.pk).values_list(
                "pk",
                "data_cutoff",
                "universe_snapshot_id",
            )
        ),
        "snapshot": tuple(
            UniverseSnapshot.objects.filter(pk=run.universe_snapshot_id).values_list(
                "pk",
                "grade",
                "config_hash",
            )
        ),
        "memberships": tuple(
            UniverseMembership.objects.filter(snapshot_id=run.universe_snapshot_id)
            .order_by("pk")
            .values_list("pk", "listing_id", "eligible", "exclusion_reason")
        ),
        "listings": tuple(
            Listing.objects.filter(pk__in=[listing.pk for listing in control["listings"]])
            .order_by("pk")
            .values_list("pk", "ticker", "provider_symbol", "exchange_mic")
        ),
        "securities": tuple(
            Security.objects.filter(pk__in=[listing.security_id for listing in control["listings"]])
            .order_by("pk")
            .values_list("pk", "security_type", "name")
        ),
        "analyses": tuple(
            StockAnalysis.objects.filter(pk__in=analysis_ids)
            .order_by("pk")
            .values_list("pk", "run_id", "listing_id", "data_quality")
        ),
    }


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL advisory parent-lock interleaving coverage",
)
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("entry_point", ["append", "create"])
def test_postgresql_v2_writer_locks_complete_authority_through_prediction_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
) -> None:
    """Every parent mutation waits beyond attestation and prediction writes."""
    import stanstock.research.service as service_module

    control = _matching_v2_writer_control(tmp_path, distinct_listings=2)
    additional = _additional_matching_analysis(control) if entry_point == "append" else None
    analysis_to_mutate = additional[0] if additional is not None else control["analysis"]
    run = control["analysis"].run
    snapshot = run.universe_snapshot
    listing = control["analysis"].listing
    membership = UniverseMembership.objects.get(snapshot=snapshot, listing=listing)
    ineligible_membership = UniverseMembership.objects.create(
        snapshot=snapshot,
        listing=_listing(f"PGOUT{uuid4().hex[:3]}"),
        eligible=False,
        exclusion_reason="control exclusion",
    )
    inserted_listing = _listing(f"PGNEW{uuid4().hex[:3]}")
    before = _advisory_parent_state(control, additional)

    attested = threading.Event()
    release_writer = threading.Event()
    errors: list[BaseException] = []
    writer_result: list[object] = []
    real_attest = service_module._attest_medium_v2_panel_with_authority

    def pause_after_attestation(**kwargs: Any) -> Any:
        result = real_attest(**kwargs)
        attested.set()
        assert release_writer.wait(timeout=10)
        return result

    monkeypatch.setattr(
        service_module,
        "_attest_medium_v2_panel_with_authority",
        pause_after_attestation,
    )

    def write() -> None:
        connections.close_all()
        try:
            if entry_point == "append":
                writer_result.extend(_append_control_predictions(control, additional=additional))
            else:
                _invoke_v2_writer(
                    entry_point="create",
                    analysis=control["analysis"],
                    forecasts=control["forecasts"],
                    panel=control["panel"],
                    store=control["store"],
                    source_assets_without_panel=control["source_assets"],
                    model_version=control["model_version"],
                )
                writer_result.append("created")
        except BaseException as exc:  # pragma: no cover - asserted by the caller
            errors.append(exc)
        finally:
            connections.close_all()

    mutations = (
        lambda: Universe.objects.filter(pk=snapshot.universe_id).update(
            config_version="postgres-concurrent"
        ),
        lambda: AnalysisRun.objects.filter(pk=run.pk).update(
            data_cutoff=run.data_cutoff - timedelta(microseconds=1)
        ),
        lambda: StockAnalysis.objects.filter(pk=analysis_to_mutate.pk).update(
            data_quality={"concurrent_parent_mutation": True}
        ),
        lambda: UniverseSnapshot.objects.filter(pk=snapshot.pk).update(
            grade=UniverseSnapshot.Grade.OBSERVED,
            config_hash="p" * 64,
        ),
        lambda: Listing.objects.filter(pk=listing.pk).update(provider_symbol="PG-CONCURRENT"),
        lambda: Security.objects.filter(pk=listing.security_id).update(
            security_type=Security.SecurityType.ETF
        ),
        lambda: UniverseMembership.objects.filter(pk=membership.pk).update(
            eligible=False,
            exclusion_reason="postgres concurrent mutation",
        ),
        lambda: UniverseMembership.objects.filter(pk=ineligible_membership.pk).update(
            eligible=True,
            exclusion_reason="",
        ),
        lambda: _insert_membership_with_immediate_fk_check(
            snapshot_id=snapshot.pk,
            listing=inserted_listing,
        ),
    )
    started = [threading.Event() for _mutation in mutations]
    done = [threading.Event() for _mutation in mutations]
    backend_pids = [[] for _mutation in mutations]
    writer = threading.Thread(target=write)
    mutators: list[threading.Thread] = []
    writer.start()
    try:
        assert attested.wait(timeout=10)
        for index, operation in enumerate(mutations):
            thread = threading.Thread(
                target=_rolled_back_concurrent_mutation,
                args=(operation,),
                kwargs={
                    "backend_pid": backend_pids[index],
                    "started": started[index],
                    "done": done[index],
                    "errors": errors,
                },
            )
            mutators.append(thread)
            thread.start()
        assert all(event.wait(timeout=10) for event in started)
        for backend_pid in backend_pids:
            assert len(backend_pid) == 1
            _wait_for_postgresql_lock_wait(backend_pid[0])
        assert not any(event.is_set() for event in done)
    finally:
        release_writer.set()
        writer.join(timeout=15)
        for mutator in mutators:
            mutator.join(timeout=15)

    assert not writer.is_alive()
    assert all(not mutator.is_alive() for mutator in mutators)
    assert errors == []
    assert all(event.is_set() for event in done)
    assert len(writer_result) == (4 if entry_point == "append" else 1)
    assert _advisory_parent_state(control, additional) == before
    assert Prediction.objects.filter(
        model_version=control["model_version"],
        method_version=MEDIUM_V2_VERSION,
    ).count() == (4 if entry_point == "append" else 1)


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL analyze_snapshot nested advisory-lock coverage",
)
@pytest.mark.django_db(transaction=True)
def test_postgresql_analyze_snapshot_holds_advisory_parent_locks_to_outer_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested append locks survive its savepoint until analyze_snapshot ends."""
    import stanstock.research.service as service_module

    store = AssetStore(tmp_path)
    snapshot = _e2e_universe(store, count=2)
    membership = UniverseMembership.objects.filter(snapshot=snapshot).order_by("pk").first()
    assert membership is not None
    listing = membership.listing
    ineligible_membership = UniverseMembership.objects.create(
        snapshot=snapshot,
        listing=_listing(f"PGANOUT{uuid4().hex[:3]}"),
        eligible=False,
        exclusion_reason="control exclusion",
    )
    inserted_listing = _listing(f"PGAN{uuid4().hex[:4]}")
    before = {
        "universe": tuple(
            Universe.objects.filter(pk=snapshot.universe_id).values_list(
                "name",
                "config_version",
            )
        ),
        "snapshot": tuple(
            UniverseSnapshot.objects.filter(pk=snapshot.pk).values_list(
                "grade",
                "config_hash",
            )
        ),
        "membership": tuple(
            UniverseMembership.objects.filter(pk=membership.pk).values_list(
                "eligible",
                "exclusion_reason",
            )
        ),
        "ineligible_membership": tuple(
            UniverseMembership.objects.filter(pk=ineligible_membership.pk).values_list(
                "eligible",
                "exclusion_reason",
            )
        ),
        "listing": tuple(
            Listing.objects.filter(pk=listing.pk).values_list("ticker", "provider_symbol")
        ),
        "security": tuple(
            Security.objects.filter(pk=listing.security_id).values_list(
                "security_type",
                "name",
            )
        ),
        "membership_count": UniverseMembership.objects.filter(snapshot=snapshot).count(),
    }
    attested = threading.Event()
    release_writer = threading.Event()
    errors: list[BaseException] = []
    results: list[Any] = []
    real_attest = service_module._attest_medium_v2_panel_with_authority

    def pause_after_attestation(**kwargs: Any) -> Any:
        result = real_attest(**kwargs)
        attested.set()
        assert release_writer.wait(timeout=10)
        return result

    monkeypatch.setattr(
        service_module,
        "_attest_medium_v2_panel_with_authority",
        pause_after_attestation,
    )

    def analyze() -> None:
        connections.close_all()
        try:
            results.extend(
                analyze_snapshot(
                    universe_snapshot=snapshot,
                    decision_time=GENERATED_AT,
                    target_date=TARGET_DATE,
                    issued_on_time=False,
                    provider="synthetic",
                    benchmark_subject="SPY",
                    store=store,
                    config_path=default_us_scoring_config_path(),
                    medium_forecast_config_path=V2_PATH,
                    long_forecast_requested=False,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted by the caller
            errors.append(exc)
        finally:
            connections.close_all()

    mutations = (
        lambda: Universe.objects.filter(pk=snapshot.universe_id).update(
            config_version="nested-concurrent"
        ),
        lambda: UniverseSnapshot.objects.filter(pk=snapshot.pk).update(
            grade=UniverseSnapshot.Grade.OBSERVED,
            config_hash="n" * 64,
        ),
        lambda: Listing.objects.filter(pk=listing.pk).update(provider_symbol="PG-NESTED"),
        lambda: Security.objects.filter(pk=listing.security_id).update(
            security_type=Security.SecurityType.ETF
        ),
        lambda: UniverseMembership.objects.filter(pk=membership.pk).update(
            eligible=False,
            exclusion_reason="nested concurrent mutation",
        ),
        lambda: UniverseMembership.objects.filter(pk=ineligible_membership.pk).update(
            eligible=True,
            exclusion_reason="",
        ),
        lambda: _insert_membership_with_immediate_fk_check(
            snapshot_id=snapshot.pk,
            listing=inserted_listing,
        ),
    )
    started = [threading.Event() for _mutation in mutations]
    done = [threading.Event() for _mutation in mutations]
    backend_pids = [[] for _mutation in mutations]
    writer = threading.Thread(target=analyze)
    mutators: list[threading.Thread] = []
    writer.start()
    try:
        assert attested.wait(timeout=15)
        for index, operation in enumerate(mutations):
            thread = threading.Thread(
                target=_rolled_back_concurrent_mutation,
                args=(operation,),
                kwargs={
                    "backend_pid": backend_pids[index],
                    "started": started[index],
                    "done": done[index],
                    "errors": errors,
                },
            )
            mutators.append(thread)
            thread.start()
        assert all(event.wait(timeout=10) for event in started)
        for backend_pid in backend_pids:
            assert len(backend_pid) == 1
            _wait_for_postgresql_lock_wait(backend_pid[0])
        assert not any(event.is_set() for event in done)
    finally:
        release_writer.set()
        writer.join(timeout=20)
        for mutator in mutators:
            mutator.join(timeout=15)

    assert not writer.is_alive()
    assert all(not mutator.is_alive() for mutator in mutators)
    assert errors == []
    assert len(results) == 2
    assert all(event.is_set() for event in done)
    assert (
        tuple(
            Universe.objects.filter(pk=snapshot.universe_id).values_list(
                "name",
                "config_version",
            )
        )
        == before["universe"]
    )
    assert (
        tuple(
            UniverseSnapshot.objects.filter(pk=snapshot.pk).values_list(
                "grade",
                "config_hash",
            )
        )
        == before["snapshot"]
    )
    assert (
        tuple(
            UniverseMembership.objects.filter(pk=membership.pk).values_list(
                "eligible",
                "exclusion_reason",
            )
        )
        == before["membership"]
    )
    assert (
        tuple(
            UniverseMembership.objects.filter(pk=ineligible_membership.pk).values_list(
                "eligible",
                "exclusion_reason",
            )
        )
        == before["ineligible_membership"]
    )
    assert (
        tuple(Listing.objects.filter(pk=listing.pk).values_list("ticker", "provider_symbol"))
        == before["listing"]
    )
    assert (
        tuple(
            Security.objects.filter(pk=listing.security_id).values_list(
                "security_type",
                "name",
            )
        )
        == before["security"]
    )
    assert (
        UniverseMembership.objects.filter(snapshot=snapshot).count() == before["membership_count"]
    )
    assert Prediction.objects.filter(method_version=MEDIUM_V2_VERSION).count() == 4


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["append", "create"])
def test_v2_both_writer_boundaries_reject_caller_provenance_and_malformed_nested(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
    entry_point: str,
) -> None:
    store, _snapshot, results = persisted_v2
    analysis = results[0].analysis
    predictions = {
        prediction.horizon: prediction
        for prediction in Prediction.objects.filter(
            analysis=analysis,
            method_version=MEDIUM_V2_VERSION,
        )
    }
    forecasts = {
        horizon: _forecast_from_prediction(prediction)
        for horizon, prediction in predictions.items()
    }
    panel = DataAsset.objects.get(pk=next(iter(predictions.values())).calculation["panel_asset_id"])
    source_assets = next(iter(predictions.values())).source_assets
    run = analysis.run

    def invoke(candidate_forecasts: dict[str, MediumForecast]) -> None:
        if entry_point == "append":
            append_advisory_predictions(
                analysis=analysis,
                forecasts=candidate_forecasts,
                panel_asset=panel,
                store=store,
                generated_at=run.generated_at,
                data_cutoff=run.data_cutoff,
                issued_on_time=False,
                model_version=next(iter(predictions.values())).model_version,
                config_hash_value=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
                source_assets=[
                    source for source in source_assets if source["kind"] != "medium_forecast_panel"
                ],
                code_revision_value=run.code_revision,
            )
        else:
            _create_advisory_prediction(
                analysis=analysis,
                horizon=Prediction.Horizon.SIX_MONTH,
                forecast=candidate_forecasts["6m"],
                panel_asset=panel,
                store=store,
                generated_at=run.generated_at,
                data_cutoff=run.data_cutoff,
                issued_on_time=False,
                model_version=predictions["6m"].model_version,
                config_hash_value=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
                source_assets=source_assets,
                code_revision_value=run.code_revision,
            )

    owned = deepcopy(forecasts)
    owned["6m"].calculation["config_hash"] = MEDIUM_V2_EFFECTIVE_CONFIG_HASH
    with pytest.raises(ValueError, match="must not supply service-owned provenance"):
        invoke(owned)

    malformed = deepcopy(forecasts)
    del malformed["6m"].calculation["evidence"]["interval"]
    with pytest.raises(ValueError, match="prediction payload is malformed"):
        invoke(malformed)


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["append", "create"])
@pytest.mark.parametrize("state", ["published", "range_only"])
def test_v2_exact_raw_mixture_rejects_coordinated_four_decimal_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
    state: str,
) -> None:
    if state == "published":
        evidence = _positive_v2_evidence()
        monkeypatch.setattr(
            medium_forecasts,
            "_v2_prequential_evidence",
            lambda *_args, **_kwargs: deepcopy(evidence),
        )
    control = _matching_v2_writer_control(
        tmp_path,
        distinct_listings=30,
    )
    store = control["store"]
    analysis = control["analysis"]
    panel = control["panel"]
    forecasts = control["forecasts"]
    source_assets_without_panel = control["source_assets"]
    model_version = control["model_version"]
    selected = forecasts["6m"]
    predictive = selected.calculation["predictive_distribution"]
    raw_probability = predictive["probability_positive_raw"]
    assert isinstance(raw_probability, float)
    rounded_probability = float(Decimal(str(raw_probability)).quantize(Decimal("0.0001")))
    assert raw_probability != rounded_probability
    assert selected.scenario.confidence_status == (
        "empirical_skill_supported" if state == "published" else "empirical_range_only"
    )
    selected_payload = selected.scenario_payload()
    coherent_forecasts = forecasts
    if state == "published":
        display_payload = deepcopy(selected_payload)
        display_payload["probability_positive"] = rounded_probability
        assert validated_medium_forecast_scenario(display_payload, "6m") is display_payload
        coherent_forecasts = deepcopy(forecasts)
        coherent_forecasts["6m"] = replace(
            coherent_forecasts["6m"],
            scenario=replace(
                coherent_forecasts["6m"].scenario,
                probability_positive=rounded_probability,
            ),
        )

    created: list[dict[str, Any]] = []

    def capture_create(**kwargs: Any) -> Prediction:
        created.append(kwargs)
        return Prediction()

    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    _invoke_v2_writer(
        entry_point=entry_point,
        analysis=analysis,
        forecasts=coherent_forecasts,
        panel=panel,
        store=store,
        source_assets_without_panel=source_assets_without_panel,
        model_version=model_version,
    )
    assert len(created) == (2 if entry_point == "append" else 1)

    forged_forecasts = deepcopy(forecasts)
    forged = forged_forecasts["6m"]
    forged_predictive = forged.calculation["predictive_distribution"]
    forged_predictive["probability_positive_raw"] = rounded_probability
    if state == "published":
        forged_predictive["probability_positive_published"] = rounded_probability
        forged.calculation["formula_inputs"]["scenario"]["probability_positive"] = (
            rounded_probability
        )
        forged_forecasts["6m"] = replace(
            forged,
            scenario=replace(forged.scenario, probability_positive=rounded_probability),
        )
    created.clear()
    with pytest.raises(ValueError, match="prediction payload is malformed"):
        _invoke_v2_writer(
            entry_point=entry_point,
            analysis=analysis,
            forecasts=forged_forecasts,
            panel=panel,
            store=store,
            source_assets_without_panel=source_assets_without_panel,
            model_version=model_version,
        )
    assert created == []
    assert validated_medium_forecast_scenario(
        forged_forecasts["6m"].scenario_payload(),
        "6m",
    ) == {"medium_v2_invalid": True}
    if state == "published":
        internally_rounded = deepcopy(forecasts)
        rounded_forecast = internally_rounded["6m"]
        rounded_forecast.calculation["predictive_distribution"][
            "probability_positive_published"
        ] = rounded_probability
        rounded_forecast.calculation["formula_inputs"]["scenario"]["probability_positive"] = (
            rounded_probability
        )
        internally_rounded["6m"] = replace(
            rounded_forecast,
            scenario=replace(
                rounded_forecast.scenario,
                probability_positive=rounded_probability,
            ),
        )
        with pytest.raises(ValueError, match="prediction payload is malformed"):
            _invoke_v2_writer(
                entry_point=entry_point,
                analysis=analysis,
                forecasts=internally_rounded,
                panel=panel,
                store=store,
                source_assets_without_panel=source_assets_without_panel,
                model_version=model_version,
            )
        assert validated_medium_forecast_scenario(
            internally_rounded["6m"].scenario_payload(),
            "6m",
        ) == {"medium_v2_invalid": True}


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["append", "create"])
def test_v2_matching_panel_all_null_control_passes_both_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
) -> None:
    control = _matching_v2_writer_control(
        tmp_path,
        distinct_listings=12,
        low_liquidity=True,
    )
    store = control["store"]
    analysis = control["analysis"]
    panel = control["panel"]
    forecasts = control["forecasts"]
    assert all(
        forecast.scenario.confidence_status == "insufficient_evidence"
        and forecast.scenario.bear is None
        and forecast.scenario.base is None
        and forecast.scenario.bull is None
        for forecast in forecasts.values()
    )
    all_null_payload = forecasts["6m"].scenario_payload()
    assert validated_medium_forecast_scenario(all_null_payload, "6m") is all_null_payload

    source_assets_without_panel = control["source_assets"]
    created: list[dict[str, Any]] = []

    def capture_create(**kwargs: Any) -> Prediction:
        created.append(kwargs)
        return Prediction()

    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    _invoke_v2_writer(
        entry_point=entry_point,
        analysis=analysis,
        forecasts=forecasts,
        panel=panel,
        store=store,
        source_assets_without_panel=source_assets_without_panel,
        model_version=control["model_version"],
    )
    assert len(created) == (2 if entry_point == "append" else 1)


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["append", "create"])
def test_v2_both_writers_reject_same_run_fabricated_panel_and_matching_forecast(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
) -> None:
    store, _snapshot, results = persisted_v2
    analysis = results[0].analysis
    predictions = {
        prediction.horizon: prediction
        for prediction in Prediction.objects.filter(
            analysis=analysis,
            method_version=MEDIUM_V2_VERSION,
        )
    }
    original_panel = DataAsset.objects.get(pk=predictions["6m"].calculation["panel_asset_id"])
    foreign_frame = _controlled_v2_panel_frame(
        listing_id=str(analysis.listing_id),
        distinct_listings=30,
    )
    foreign_panel = _store_controlled_v2_panel(
        store=store,
        original=original_panel,
        frame=foreign_frame,
        suffix=f"foreign-30-{entry_point}",
    )
    authoritative_panel = original_panel
    evidence = _positive_v2_evidence()
    monkeypatch.setattr(
        medium_forecasts,
        "_v2_prequential_evidence",
        lambda *_args, **_kwargs: deepcopy(evidence),
    )
    foreign_forecasts = build_medium_forecasts(
        foreign_frame,
        _v2_config(),
    )[str(analysis.listing_id)]
    assert foreign_forecasts["6m"].scenario.confidence_status == "empirical_skill_supported"
    assert foreign_forecasts["6m"].calculation["support"]["distinct_listings"] == 30
    assert foreign_panel.sha256 != authoritative_panel.sha256
    assert foreign_panel.metadata["source_assets"] == authoritative_panel.metadata["source_assets"]
    authoritative_frame = store.read_frame(authoritative_panel.relative_path)
    assert authoritative_frame.filter(pl.col("is_forecast"))["listing_id"].n_unique() == 12
    assert {
        row["listing_id"] for row in foreign_frame.filter(~pl.col("is_forecast")).to_dicts()
    } == {f"support-{index:02d}" for index in range(30)}

    source_assets_without_panel = [
        source
        for source in predictions["6m"].source_assets
        if source["kind"] != "medium_forecast_panel"
    ]
    created: list[dict[str, Any]] = []

    def capture_create(**kwargs: Any) -> Prediction:
        created.append(kwargs)
        return predictions[str(kwargs["horizon"])]

    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    before = Prediction.objects.count()
    with pytest.raises(ValueError, match="prediction payload is malformed"):
        _invoke_v2_writer(
            entry_point=entry_point,
            analysis=analysis,
            forecasts=foreign_forecasts,
            panel=foreign_panel,
            store=store,
            source_assets_without_panel=source_assets_without_panel,
            model_version=predictions["6m"].model_version,
        )
    assert created == []
    assert Prediction.objects.count() == before


@pytest.mark.django_db
def test_v2_writer_signatures_refuse_substituted_or_reconstructed_attestations(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stanstock.research.service as service_module

    store, _snapshot, results = persisted_v2
    analysis = results[0].analysis
    predictions = {
        prediction.horizon: prediction
        for prediction in Prediction.objects.filter(
            analysis=analysis,
            method_version=MEDIUM_V2_VERSION,
        )
    }
    original_panel = DataAsset.objects.get(pk=predictions["6m"].calculation["panel_asset_id"])
    genuine = service_module._attest_medium_v2_panel(
        run=analysis.run,
        panel_asset=original_panel,
        store=store,
    )
    with pytest.raises(TypeError):
        genuine.calculations[str(analysis.listing_id)]["6m"]["support"] = {}

    fabricated_frame = _controlled_v2_panel_frame(
        listing_id=str(analysis.listing_id),
        distinct_listings=30,
    )
    fabricated_panel = _store_controlled_v2_panel(
        store=store,
        original=original_panel,
        frame=fabricated_frame,
        suffix="attestation-substitution",
    )
    evidence = _positive_v2_evidence()
    monkeypatch.setattr(
        medium_forecasts,
        "_v2_prequential_evidence",
        lambda *_args, **_kwargs: deepcopy(evidence),
    )
    fabricated_forecasts = build_medium_forecasts(
        fabricated_frame,
        _v2_config(),
    )[str(analysis.listing_id)]
    fabricated_calculations = {
        str(analysis.listing_id): {
            horizon: forecast.calculation for horizon, forecast in fabricated_forecasts.items()
        }
    }
    substituted = replace(
        genuine,
        panel_id=fabricated_panel.id,
        panel_sha256=fabricated_panel.sha256,
        calculations=fabricated_calculations,
    )
    reconstructed = type(genuine)(
        panel_id=fabricated_panel.id,
        panel_sha256=fabricated_panel.sha256,
        run_id=genuine.run_id,
        config_hash=genuine.config_hash,
        snapshot_id=genuine.snapshot_id,
        source_manifest_hash=genuine.source_manifest_hash,
        price_provider=genuine.price_provider,
        calculations=fabricated_calculations,
    )
    assert not hasattr(service_module, "_MEDIUM_V2_ATTESTATION_SEAL")
    for writer in (
        append_advisory_predictions,
        _create_advisory_prediction,
        service_module._attest_medium_v2_panel,
    ):
        parameter_names = set(inspect.signature(writer).parameters)
        assert not any(
            "authority" in name or "attestation" in name or "token" in name
            for name in parameter_names
        )

    source_assets_without_panel = [
        source
        for source in predictions["6m"].source_assets
        if source["kind"] != "medium_forecast_panel"
    ]
    append_kwargs = {
        "analysis": analysis,
        "forecasts": fabricated_forecasts,
        "panel_asset": fabricated_panel,
        "store": store,
        "generated_at": analysis.run.generated_at,
        "data_cutoff": analysis.run.data_cutoff,
        "issued_on_time": False,
        "model_version": predictions["6m"].model_version,
        "config_hash_value": MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
        "source_assets": source_assets_without_panel,
        "code_revision_value": analysis.run.code_revision,
    }
    create_kwargs = {
        "analysis": analysis,
        "horizon": Prediction.Horizon.SIX_MONTH,
        "forecast": fabricated_forecasts["6m"],
        "panel_asset": fabricated_panel,
        "store": store,
        "generated_at": analysis.run.generated_at,
        "data_cutoff": analysis.run.data_cutoff,
        "issued_on_time": False,
        "model_version": predictions["6m"].model_version,
        "config_hash_value": MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
        "source_assets": [*source_assets_without_panel, _asset_payload(fabricated_panel)],
        "code_revision_value": analysis.run.code_revision,
    }
    created: list[dict[str, Any]] = []

    def capture_create(**kwargs: Any) -> Prediction:
        created.append(kwargs)
        return Prediction()

    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    before = Prediction.objects.count()
    for writer, kwargs in (
        (append_advisory_predictions, append_kwargs),
        (_create_advisory_prediction, create_kwargs),
    ):
        for forged_attestation in (substituted, reconstructed):
            with pytest.raises(TypeError, match="_panel_attestation"):
                writer(**kwargs, _panel_attestation=forged_attestation)
    assert created == []
    assert Prediction.objects.count() == before

    real_attest = service_module._attest_medium_v2_panel_with_authority
    attested_panel_ids: list[str] = []

    def record_attestation(**kwargs: Any) -> Any:
        attested_panel_ids.append(str(kwargs["panel_asset"].id))
        return real_attest(**kwargs)

    monkeypatch.setattr(
        service_module,
        "_attest_medium_v2_panel_with_authority",
        record_attestation,
    )
    for writer, kwargs in (
        (append_advisory_predictions, append_kwargs),
        (_create_advisory_prediction, create_kwargs),
    ):
        with pytest.raises(ValueError, match="prediction payload is malformed"):
            writer(**kwargs)
    assert attested_panel_ids == [str(fabricated_panel.id), str(fabricated_panel.id)]
    assert created == []
    assert Prediction.objects.count() == before


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["append", "create"])
@pytest.mark.parametrize(
    "mutation",
    [
        "id",
        "provider",
        "kind",
        "subject",
        "relative_path",
        "sha256",
        "available_at",
        "retrieved_at",
        "missing",
        "extra",
        "duplicate",
        "benchmark",
    ],
)
def test_v2_source_manifest_identity_and_exact_closure_mutations_fail_closed(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
    mutation: str,
) -> None:
    store, _snapshot, results = persisted_v2
    analysis = results[0].analysis
    predictions = {
        prediction.horizon: prediction
        for prediction in Prediction.objects.filter(
            analysis=analysis,
            method_version=MEDIUM_V2_VERSION,
        )
    }
    original_panel = DataAsset.objects.get(pk=predictions["6m"].calculation["panel_asset_id"])
    frame = _controlled_v2_panel_frame(
        listing_id=str(analysis.listing_id),
        distinct_listings=30,
    )

    def mutate_manifest(manifest: list[dict[str, Any]]) -> None:
        benchmark_index = next(
            index for index, source in enumerate(manifest) if source["subject"] == "SPY"
        )
        target_index = next(
            index
            for index, source in enumerate(manifest)
            if source["subject"]
            not in {"SPY", analysis.listing.provider_symbol or analysis.listing.ticker}
        )
        if mutation == "missing":
            manifest.pop(target_index)
            return
        if mutation == "extra":
            extra = deepcopy(manifest[target_index])
            extra["id"] = str(uuid4())
            manifest.append(extra)
            return
        if mutation == "duplicate":
            manifest.append(deepcopy(manifest[target_index]))
            return
        if mutation == "benchmark":
            manifest[benchmark_index]["subject"] = "QQQ"
            return
        target = manifest[target_index]
        if mutation == "id":
            target["id"] = str(uuid4())
        elif mutation == "provider":
            target["provider"] = "invented-provider"
        elif mutation == "kind":
            target["kind"] = "invented-kind"
        elif mutation == "subject":
            target["subject"] = "INVENTED"
        elif mutation == "relative_path":
            target["relative_path"] = "invented/path.parquet"
        elif mutation == "sha256":
            target["sha256"] = "0" * 64
        elif mutation in {"available_at", "retrieved_at"}:
            timestamp = datetime.fromisoformat(target[mutation])
            target[mutation] = (timestamp - timedelta(microseconds=1)).isoformat()
        else:  # pragma: no cover - parametrization is exhaustive
            raise AssertionError(mutation)

    panel = _store_controlled_v2_panel(
        store=store,
        original=original_panel,
        frame=frame,
        suffix=f"manifest-{mutation}-{entry_point}",
        source_manifest_mutator=mutate_manifest,
    )
    forecasts = build_medium_forecasts(frame, _v2_config())[str(analysis.listing_id)]
    source_assets_without_panel = [
        source
        for source in predictions["6m"].source_assets
        if source["kind"] != "medium_forecast_panel"
    ]
    created: list[dict[str, Any]] = []

    def capture_create(**kwargs: Any) -> Prediction:
        created.append(kwargs)
        return Prediction()

    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    before = Prediction.objects.count()
    with pytest.raises(ValueError, match="prediction payload is malformed"):
        _invoke_v2_writer(
            entry_point=entry_point,
            analysis=analysis,
            forecasts=forecasts,
            panel=panel,
            store=store,
            source_assets_without_panel=source_assets_without_panel,
            model_version=predictions["6m"].model_version,
        )
    assert created == []
    assert Prediction.objects.count() == before


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["append", "create"])
def test_v2_attestation_checksum_reads_each_exact_source_once_without_reselection(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
) -> None:
    store, _snapshot, results = persisted_v2
    analysis = results[0].analysis
    predictions = {
        prediction.horizon: prediction
        for prediction in Prediction.objects.filter(
            analysis=analysis,
            method_version=MEDIUM_V2_VERSION,
        )
    }
    panel = DataAsset.objects.get(pk=predictions["6m"].calculation["panel_asset_id"])
    forecasts = {
        horizon: _forecast_from_prediction(prediction)
        for horizon, prediction in predictions.items()
    }
    source_assets_without_panel = [
        source
        for source in predictions["6m"].source_assets
        if source["kind"] != "medium_forecast_panel"
    ]
    exact_reads: list[str] = []
    original_read = AsOfData.price_frame_for_asset_with_diagnostics

    def forbid_reselection(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("panel attestation reselected a source asset")

    def record_exact_read(self: AsOfData, **kwargs: Any) -> Any:
        exact_reads.append(str(kwargs["asset"].id))
        return original_read(self, **kwargs)

    def capture_create(**_kwargs: Any) -> Prediction:
        return Prediction()

    monkeypatch.setattr(AsOfData, "latest_asset", forbid_reselection)
    monkeypatch.setattr(
        AsOfData,
        "price_frame_for_asset_with_diagnostics",
        record_exact_read,
    )
    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    _invoke_v2_writer(
        entry_point=entry_point,
        analysis=analysis,
        forecasts=forecasts,
        panel=panel,
        store=store,
        source_assets_without_panel=source_assets_without_panel,
        model_version=predictions["6m"].model_version,
    )
    expected_ids = {source["id"] for source in panel.metadata["source_assets"]}
    assert set(exact_reads) == expected_ids
    assert len(exact_reads) == len(expected_ids)


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["append", "create"])
def test_v2_attestation_rejects_corrupt_exact_source_without_fallback(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
) -> None:
    store, _snapshot, results = persisted_v2
    analysis = results[0].analysis
    predictions = {
        prediction.horizon: prediction
        for prediction in Prediction.objects.filter(
            analysis=analysis,
            method_version=MEDIUM_V2_VERSION,
        )
    }
    panel = DataAsset.objects.get(pk=predictions["6m"].calculation["panel_asset_id"])
    forecasts = {
        horizon: _forecast_from_prediction(prediction)
        for horizon, prediction in predictions.items()
    }
    source_assets_without_panel = [
        source
        for source in predictions["6m"].source_assets
        if source["kind"] != "medium_forecast_panel"
    ]
    source = DataAsset.objects.get(pk=panel.metadata["source_assets"][0]["id"])
    store.resolve(source.relative_path).write_bytes(b"corrupt exact source")
    created: list[dict[str, Any]] = []

    def forbid_reselection(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("corrupt source triggered fallback selection")

    def capture_create(**kwargs: Any) -> Prediction:
        created.append(kwargs)
        return Prediction()

    monkeypatch.setattr(AsOfData, "latest_asset", forbid_reselection)
    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    before = Prediction.objects.count()
    with pytest.raises(ValueError, match="prediction payload is malformed"):
        _invoke_v2_writer(
            entry_point=entry_point,
            analysis=analysis,
            forecasts=forecasts,
            panel=panel,
            store=store,
            source_assets_without_panel=source_assets_without_panel,
            model_version=predictions["6m"].model_version,
        )
    assert created == []
    assert Prediction.objects.count() == before


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["append", "create"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_listing"])
def test_v2_snapshot_membership_closure_mutations_fail_closed(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
    mutation: str,
) -> None:
    store, snapshot, results = persisted_v2
    analysis = results[0].analysis
    predictions = {
        prediction.horizon: prediction
        for prediction in Prediction.objects.filter(
            analysis=analysis,
            method_version=MEDIUM_V2_VERSION,
        )
    }
    panel = DataAsset.objects.get(pk=predictions["6m"].calculation["panel_asset_id"])
    forecasts = {
        horizon: _forecast_from_prediction(prediction)
        for horizon, prediction in predictions.items()
    }
    unrelated = (
        UniverseMembership.objects.filter(snapshot=snapshot, eligible=True)
        .exclude(listing=analysis.listing)
        .first()
    )
    assert unrelated is not None
    if mutation in {"missing", "wrong_listing"}:
        unrelated.eligible = False
        unrelated.exclusion_reason = "attestation mutation"
        unrelated.save(update_fields=["eligible", "exclusion_reason"])
    if mutation in {"extra", "wrong_listing"}:
        added = _listing(f"X{uuid4().hex[:7]}")
        UniverseMembership.objects.create(snapshot=snapshot, listing=added)

    source_assets_without_panel = [
        source
        for source in predictions["6m"].source_assets
        if source["kind"] != "medium_forecast_panel"
    ]
    created: list[dict[str, Any]] = []

    def capture_create(**kwargs: Any) -> Prediction:
        created.append(kwargs)
        return Prediction()

    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    before = Prediction.objects.count()
    with pytest.raises(ValueError, match="prediction payload is malformed"):
        _invoke_v2_writer(
            entry_point=entry_point,
            analysis=analysis,
            forecasts=forecasts,
            panel=panel,
            store=store,
            source_assets_without_panel=source_assets_without_panel,
            model_version=predictions["6m"].model_version,
        )
    assert created == []
    assert Prediction.objects.count() == before


@pytest.mark.django_db
@pytest.mark.parametrize("entry_point", ["append", "create"])
@pytest.mark.parametrize(
    "mutation",
    [
        "listing_id",
        "price_asset_id",
        "feature",
        "label",
        "schema_version",
        "method_version",
        "config_hash",
        "calendar",
        "calendar_hash",
        "source_manifest_hash",
        "evidence_bundle_hash",
        "content_sha256",
        "row_count",
    ],
)
def test_v2_panel_rows_and_metadata_mutations_fail_closed(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
    entry_point: str,
    mutation: str,
) -> None:
    store, _snapshot, results = persisted_v2
    analysis = results[0].analysis
    predictions = {
        prediction.horizon: prediction
        for prediction in Prediction.objects.filter(
            analysis=analysis,
            method_version=MEDIUM_V2_VERSION,
        )
    }
    original_panel = DataAsset.objects.get(pk=predictions["6m"].calculation["panel_asset_id"])
    frame = store.read_frame(original_panel.relative_path)
    metadata_mutations = {
        "schema_version",
        "method_version",
        "config_hash",
        "calendar",
        "calendar_hash",
        "source_manifest_hash",
        "evidence_bundle_hash",
        "content_sha256",
        "row_count",
    }
    rows = frame.to_dicts()
    if mutation in metadata_mutations:
        row = next(candidate for candidate in rows if not candidate["is_forecast"])
        row["ticker"] = f"{row['ticker']}-metadata-mutation"
        frame = _panel(rows)
    else:
        row = next(
            candidate
            for candidate in rows
            if not candidate["is_forecast"]
            and candidate["relative_momentum"] is not None
            and candidate["forward_return"] is not None
        )
        if mutation == "listing_id":
            row["listing_id"] = str(uuid4())
        elif mutation == "price_asset_id":
            row["price_asset_id"] = str(uuid4())
        elif mutation == "feature":
            row["relative_momentum"] = float(row["relative_momentum"]) + 0.01
        elif mutation == "label":
            row["forward_return"] = float(row["forward_return"]) + 0.01
        frame = _panel(rows)

    def mutate_metadata(metadata: dict[str, Any]) -> None:
        if mutation == "schema_version":
            metadata[mutation] = 2
        elif mutation == "method_version":
            metadata[mutation] = "invented-method"
        elif mutation == "config_hash":
            metadata[mutation] = "0" * 64
        elif mutation == "calendar":
            metadata[mutation] = "XNAS"
        elif mutation in {
            "calendar_hash",
            "source_manifest_hash",
            "evidence_bundle_hash",
            "content_sha256",
        }:
            metadata[mutation] = "0" * 64
        elif mutation == "row_count":
            metadata[mutation] = int(metadata[mutation]) + 1

    panel = _store_controlled_v2_panel(
        store=store,
        original=original_panel,
        frame=frame,
        suffix=f"panel-{mutation}-{entry_point}",
        metadata_mutator=mutate_metadata if mutation in metadata_mutations else None,
    )
    forecasts = build_medium_forecasts(frame, _v2_config())[str(analysis.listing_id)]
    source_assets_without_panel = [
        source
        for source in predictions["6m"].source_assets
        if source["kind"] != "medium_forecast_panel"
    ]
    created: list[dict[str, Any]] = []

    def capture_create(**kwargs: Any) -> Prediction:
        created.append(kwargs)
        return Prediction()

    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    before = Prediction.objects.count()
    with pytest.raises(ValueError, match="prediction payload is malformed"):
        _invoke_v2_writer(
            entry_point=entry_point,
            analysis=analysis,
            forecasts=forecasts,
            panel=panel,
            store=store,
            source_assets_without_panel=source_assets_without_panel,
            model_version=predictions["6m"].model_version,
        )
    assert created == []
    assert Prediction.objects.count() == before


@pytest.mark.django_db
def test_v2_normal_append_attests_once_and_direct_create_attests_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stanstock.research.service as service_module

    store = AssetStore(tmp_path)
    snapshot = _e2e_universe(store, count=2)
    real_attest = service_module._attest_medium_v2_panel_with_authority
    attested_panels: list[str] = []
    config_reads: list[Path] = []
    original_read_bytes = Path.read_bytes

    def count_attestation(**kwargs: Any) -> Any:
        attested_panels.append(str(kwargs["panel_asset"].id))
        return real_attest(**kwargs)

    def count_config_read(path: Path) -> bytes:
        if path.resolve() == V2_PATH.resolve():
            config_reads.append(path)
        return original_read_bytes(path)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("analyze_snapshot rediscovered or relocked v2 authority")

    with monkeypatch.context() as analyze_patch:
        analyze_patch.setattr(Path, "read_bytes", count_config_read)
        analyze_patch.setattr(
            service_module,
            "_attest_medium_v2_panel_with_authority",
            count_attestation,
        )
        analyze_patch.setattr(service_module, "append_advisory_predictions", forbidden)
        analyze_patch.setattr(service_module, "_lock_advisory_authority", forbidden)
        results = analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=GENERATED_AT,
            target_date=TARGET_DATE,
            issued_on_time=False,
            provider="synthetic",
            benchmark_subject="SPY",
            store=store,
            config_path=default_us_scoring_config_path(),
            medium_forecast_config_path=V2_PATH,
            long_forecast_requested=False,
        )
    assert len(results) == 2
    assert len(attested_panels) == 1
    assert len(config_reads) == 1

    analysis = results[0].analysis
    predictions = {
        prediction.horizon: prediction
        for prediction in Prediction.objects.filter(
            analysis=analysis,
            method_version=MEDIUM_V2_VERSION,
        )
    }
    panel = DataAsset.objects.get(pk=predictions["6m"].calculation["panel_asset_id"])
    forecasts = {
        horizon: _forecast_from_prediction(prediction)
        for horizon, prediction in predictions.items()
    }
    source_assets_without_panel = [
        source
        for source in predictions["6m"].source_assets
        if source["kind"] != "medium_forecast_panel"
    ]

    def capture_create(**_kwargs: Any) -> Prediction:
        return Prediction()

    monkeypatch.setattr(
        service_module,
        "_attest_medium_v2_panel_with_authority",
        count_attestation,
    )
    monkeypatch.setattr(Prediction._default_manager, "create", capture_create)
    attested_panels.clear()
    append_advisory_predictions(
        analysis=analysis,
        forecasts=forecasts,
        panel_asset=panel,
        store=store,
        generated_at=analysis.run.generated_at,
        data_cutoff=analysis.run.data_cutoff,
        issued_on_time=False,
        model_version=predictions["6m"].model_version,
        config_hash_value=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
        source_assets=source_assets_without_panel,
        code_revision_value=analysis.run.code_revision,
    )
    assert attested_panels == [str(panel.id)]

    _create_advisory_prediction(
        analysis=analysis,
        horizon=Prediction.Horizon.SIX_MONTH,
        forecast=forecasts["6m"],
        panel_asset=panel,
        store=store,
        generated_at=analysis.run.generated_at,
        data_cutoff=analysis.run.data_cutoff,
        issued_on_time=False,
        model_version=predictions["6m"].model_version,
        config_hash_value=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
        source_assets=predictions["6m"].source_assets,
        code_revision_value=analysis.run.code_revision,
    )
    assert attested_panels == [str(panel.id), str(panel.id)]


def _mutated_v2_forecast(forecast: MediumForecast, mutation: str) -> MediumForecast:
    candidate = deepcopy(forecast)
    calculation = candidate.calculation
    if mutation == "published_below_probability_floors":
        raw = calculation["predictive_distribution"]["probability_positive_raw"]
        base_origins = calculation["evidence"]["base_accuracy"]["test_origins"]
        assert base_origins >= 4
        skill = calculation["evidence"]["probability_skill"]
        skill.update(
            {
                "status": "positive_skill",
                "test_origins": 4,
                "test_predictions": 4,
                "model_brier_score": 0.1,
                "reference_brier_score": 0.2,
                "brier_skill_score": 0.5,
            }
        )
        calculation["probability_evidence"].update({"status": "published", "reasons": []})
        calculation["predictive_distribution"]["probability_positive_published"] = raw
        calculation["formula_inputs"]["scenario"]["probability_positive"] = raw
        return replace(
            candidate,
            scenario=replace(
                candidate.scenario,
                probability_positive=raw,
                confidence_status="empirical_skill_supported",
                insufficiency_reason="",
            ),
        )
    if mutation == "range_floor":
        calculation["support"]["raw_matches"] = 19
        calculation["predictive_distribution"]["matched"]["raw_observations"] = 19
        calculation["predictive_distribution"]["unconditional"]["raw_observations"] = 19
    elif mutation == "support_component_count":
        calculation["predictive_distribution"]["matched"]["raw_observations"] -= 1
    elif mutation == "floor_self_authentication":
        calculation["probability_evidence"]["minimum_distinct_listings"] = 12
    elif mutation == "calendar_span":
        calculation["probability_evidence"]["calendar_span_days"] += 1
    elif mutation == "component_mass":
        calculation["predictive_distribution"]["matched"]["component_mass"] = 0.1
    elif mutation == "mixture_probability":
        predictive = calculation["predictive_distribution"]
        raw = predictive["probability_positive_raw"]
        assert raw is not None
        forged = (
            0.1234 if Decimal(str(raw)).quantize(Decimal("0.0001")) != Decimal("0.1234") else 0.8765
        )
        predictive["probability_positive_raw"] = forged
        if candidate.scenario.confidence_status == "empirical_skill_supported":
            predictive["probability_positive_published"] = forged
            calculation["formula_inputs"]["scenario"]["probability_positive"] = forged
            return replace(
                candidate,
                scenario=replace(candidate.scenario, probability_positive=forged),
            )
    elif mutation == "dispersion":
        calculation["support"]["dispersion"] = -0.1
        calculation["predictive_distribution"]["matched"]["dispersion"] = -0.1
    elif mutation == "base_precedence":
        base = calculation["evidence"]["base_accuracy"]
        base.update(
            {
                "status": "failed",
                "mean_absolute_error": 0.1,
                "unconditional_mean_absolute_error": 0.2,
                "spy_relative_mean_absolute_error": 0.2,
            }
        )
    elif mutation == "negative_metric":
        calculation["evidence"]["base_accuracy"]["mean_absolute_error"] = -0.1
    elif mutation == "bss_formula":
        skill = calculation["evidence"]["probability_skill"]
        skill.update(
            {
                "status": "positive_skill",
                "test_origins": 4,
                "test_predictions": 4,
                "model_brier_score": 0.1,
                "reference_brier_score": 0.2,
                "brier_skill_score": 0.4,
            }
        )
    elif mutation == "reference_precedence":
        skill = calculation["evidence"]["probability_skill"]
        skill.update(
            {
                "status": "positive_skill",
                "test_origins": 4,
                "test_predictions": 4,
                "model_brier_score": 0.1,
                "reference_brier_score": 0.0,
                "brier_skill_score": 0.5,
            }
        )
    elif mutation == "zero_origin_count":
        base = calculation["evidence"]["base_accuracy"]
        base.update({"status": "not_evaluable", "test_origins": 0, "test_predictions": 1})
    elif mutation == "positive_origin_metrics":
        base = calculation["evidence"]["base_accuracy"]
        base.update(
            {
                "status": "insufficient_support",
                "test_origins": 1,
                "test_predictions": 1,
                "mean_absolute_error": None,
                "unconditional_mean_absolute_error": None,
                "spy_relative_mean_absolute_error": None,
            }
        )
    elif mutation == "interval_status":
        interval = calculation["evidence"]["interval"]
        interval["status"] = "descriptive" if interval["status"] != "descriptive" else "preliminary"
    elif mutation == "interval_partition":
        interval = calculation["evidence"]["interval"]
        interval.update(
            {
                "empirical_coverage": 0.6,
                "below_rate": 0.2,
                "above_rate": 0.1,
            }
        )
    elif mutation == "bounded_rate":
        calculation["evidence"]["interval"]["empirical_coverage"] = 1.1
    elif mutation == "nonfinite_width":
        calculation["evidence"]["interval"]["mean_width"] = float("inf")
    elif mutation == "reason_precedence":
        reasons = calculation["probability_evidence"]["reasons"]
        assert len(reasons) >= 2
        calculation["probability_evidence"]["reasons"] = list(reversed(reasons))
        return replace(
            candidate,
            scenario=replace(
                candidate.scenario,
                insufficiency_reason=(
                    "Probability withheld: "
                    + "; ".join(calculation["probability_evidence"]["reasons"])
                ),
            ),
        )
    elif mutation == "formula_agreement":
        calculation["formula_inputs"]["scenario"]["base"] += 0.01
    elif mutation == "partial_state":
        return replace(candidate, scenario=replace(candidate.scenario, bear=None))
    else:  # pragma: no cover - the table below is exhaustive
        raise AssertionError(mutation)
    return candidate


@pytest.mark.django_db
def test_v2_writer_and_ui_semantic_mutations_all_fail_closed(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
) -> None:
    store, _snapshot, results = persisted_v2
    analysis = results[0].analysis
    predictions = {
        prediction.horizon: prediction
        for prediction in Prediction.objects.filter(
            analysis=analysis,
            method_version=MEDIUM_V2_VERSION,
        )
    }
    forecasts = {
        horizon: _forecast_from_prediction(prediction)
        for horizon, prediction in predictions.items()
    }
    assert forecasts["6m"].scenario.confidence_status == "empirical_range_only"
    panel = DataAsset.objects.get(pk=predictions["6m"].calculation["panel_asset_id"])
    source_assets = predictions["6m"].source_assets
    run = analysis.run
    mutations = (
        "published_below_probability_floors",
        "range_floor",
        "support_component_count",
        "floor_self_authentication",
        "calendar_span",
        "component_mass",
        "mixture_probability",
        "dispersion",
        "base_precedence",
        "negative_metric",
        "bss_formula",
        "reference_precedence",
        "zero_origin_count",
        "positive_origin_metrics",
        "interval_status",
        "interval_partition",
        "bounded_rate",
        "nonfinite_width",
        "reason_precedence",
        "formula_agreement",
        "partial_state",
    )
    before = Prediction.objects.count()
    for mutation in mutations:
        changed = {**forecasts, "6m": _mutated_v2_forecast(forecasts["6m"], mutation)}
        with pytest.raises(ValueError, match="prediction payload is malformed"):
            append_advisory_predictions(
                analysis=analysis,
                forecasts=changed,
                panel_asset=panel,
                store=store,
                generated_at=run.generated_at,
                data_cutoff=run.data_cutoff,
                issued_on_time=False,
                model_version=predictions["6m"].model_version,
                config_hash_value=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
                source_assets=[
                    source for source in source_assets if source["kind"] != "medium_forecast_panel"
                ],
                code_revision_value=run.code_revision,
            )
        with pytest.raises(ValueError, match="prediction payload is malformed"):
            _create_advisory_prediction(
                analysis=analysis,
                horizon=Prediction.Horizon.SIX_MONTH,
                forecast=changed["6m"],
                panel_asset=panel,
                store=store,
                generated_at=run.generated_at,
                data_cutoff=run.data_cutoff,
                issued_on_time=False,
                model_version=predictions["6m"].model_version,
                config_hash_value=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
                source_assets=source_assets,
                code_revision_value=run.code_revision,
            )
        assert validated_medium_forecast_scenario(
            changed["6m"].scenario_payload(),
            "6m",
        ) == {"medium_v2_invalid": True}, mutation
    assert Prediction.objects.count() == before


@pytest.mark.django_db
def test_v2_valid_scenarios_pass_and_malformed_nested_ui_fails_closed(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
) -> None:
    _store, _snapshot, results = persisted_v2
    scenario = results[0].analysis.six_month_forecast_scenario
    assert validated_medium_forecast_scenario(scenario, "6m") is scenario

    malformed = deepcopy(scenario)
    del malformed["evidence"]["interval"]["endpoint_policy"]
    validated = validated_medium_forecast_scenario(malformed, "6m")
    assert validated == {"medium_v2_invalid": True}

    shared_v1_status = {"confidence_status": "empirical_range_only", "bear": -0.1}
    assert validated_medium_forecast_scenario(shared_v1_status, "6m") is shared_v1_status
    assert validated_medium_forecast_scenario(scenario, "12m") == {"medium_v2_invalid": True}
    assert validated_medium_forecast_scenario(scenario) == {"medium_v2_invalid": True}


def test_v1_shapes_pass_through_and_lone_v2_markers_fail_closed() -> None:
    v1_shape = {
        "confidence_status": "empirical_range_only",
        "current_state": {"anchor_date": TARGET_DATE.isoformat()},
        "probability_evidence": {
            "effective_cohorts": 3,
            "distinct_listings": 10,
        },
        "training_evidence": {
            "grade": "research",
            "label_policy": "complete_horizon_ending_on_or_before_forecast_target",
            "cohort_policy": "fixed_epoch_non_overlapping",
        },
    }
    assert validated_medium_forecast_scenario(v1_shape, "6m") is v1_shape

    lone_markers = (
        {"method_version": MEDIUM_V2_VERSION},
        {"calculation_schema_version": 2},
        {"schema_version": 2},
        {"confidence_status": "empirical_skill_supported"},
        {"evidence": {}},
        {"predictive_distribution": {}},
        {"probability_evidence": {"status": "withheld"}},
        {"probability_evidence": {"reasons": []}},
        {"training_evidence": {"test_policy": "claimed"}},
        {"training_evidence": {"calibration_claim": False}},
    )
    for marker in lone_markers:
        assert validated_medium_forecast_scenario(marker, "6m") == {"medium_v2_invalid": True}


def test_service_v2_candidate_predicate_uses_only_accepted_markers() -> None:
    panel = DataAsset(provider="stanstock", kind="medium_forecast_panel", metadata={})
    v1 = MediumForecast(
        scenario=Scenario(
            bear=-0.1,
            base=0.1,
            bull=0.2,
            probability_positive=None,
            confidence=50.0,
            confidence_status="empirical_range_only",
            insufficiency_reason="Probability withheld",
            method="conditional_empirical_price",
        ),
        calculation={
            "schema_version": 1,
            "method_version": "us-price-medium-v1",
            "current_state": {},
            "probability_evidence": {"effective_cohorts": 3},
            "training_evidence": {
                "grade": "research",
                "label_policy": "complete_horizon_ending_on_or_before_forecast_target",
            },
        },
    )
    assert not _is_medium_v2_prediction(
        forecast=v1,
        model_version="us-price-medium-v1-run",
        config_hash_value="v1",
        panel_asset=panel,
    )

    calculation_markers = (
        {"schema_version": 2},
        {"method_version": MEDIUM_V2_VERSION},
        {"evidence": {}},
        {"predictive_distribution": {}},
        {"probability_evidence": {"status": "withheld"}},
        {"probability_evidence": {"reasons": []}},
        {"training_evidence": {"aggregation_policy": "claimed"}},
        {"training_evidence": {"profitability_claim": False}},
    )
    for marker in calculation_markers:
        candidate = replace(v1, calculation={**v1.calculation, **marker})
        assert _is_medium_v2_prediction(
            forecast=candidate,
            model_version="us-price-medium-v1-run",
            config_hash_value="v1",
            panel_asset=panel,
        )
    assert _is_medium_v2_prediction(
        forecast=replace(
            v1,
            scenario=replace(v1.scenario, confidence_status="empirical_skill_supported"),
        ),
        model_version="us-price-medium-v1-run",
        config_hash_value="v1",
        panel_asset=panel,
    )
    assert _is_medium_v2_prediction(
        forecast=v1,
        model_version=f"{MEDIUM_V2_VERSION}-run",
        config_hash_value="v1",
        panel_asset=panel,
    )
    assert _is_medium_v2_prediction(
        forecast=v1,
        model_version="us-price-medium-v1-run",
        config_hash_value=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
        panel_asset=panel,
    )
    panel.metadata = {"method_version": MEDIUM_V2_VERSION}
    assert _is_medium_v2_prediction(
        forecast=v1,
        model_version="us-price-medium-v1-run",
        config_hash_value="v1",
        panel_asset=panel,
    )


@pytest.mark.django_db
def test_authenticated_v2_service_to_template_e2e_and_malformed_suppression(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
    client: Any,
) -> None:
    _store, _snapshot, results = persisted_v2
    user = get_user_model().objects.create_user(
        username=f"v2-{uuid4().hex}",
        password="synthetic-test-only",
    )
    client.force_login(user)
    analysis = results[0].analysis
    response = client.get(reverse("stock-detail", args=[analysis.listing_id]))
    content = response.content.decode()

    assert response.status_code == 200
    assert MEDIUM_V2_VERSION in content
    assert "Research-only" in content
    assert "survivorship bias" in content
    assert "Nominal central 60% p20–p80 interval" in content
    assert "Probability skill" in content
    assert "Interval evidence" in content

    document = deepcopy(analysis.forecast_scenarios)
    document["horizons"]["6m"]["evidence"]["interval"].pop("alpha")
    StockAnalysis.objects.filter(pk=analysis.pk).update(forecast_scenarios=document)
    malformed_response = client.get(reverse("stock-detail", args=[analysis.listing_id]))
    malformed_content = " ".join(malformed_response.content.decode().split())
    six_month_article = malformed_content.split("<small>6-month advisory forecast</small>", 1)[
        1
    ].split("</article>", 1)[0]
    assert "Insufficient evidence" in six_month_article
    assert MEDIUM_V2_VERSION not in six_month_article
    assert "Probability skill" not in six_month_article
    assert "Interval evidence" not in six_month_article


@pytest.mark.django_db
def test_v2_isolated_from_decision_opportunity_and_simulation_signals(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
) -> None:
    _store, snapshot, results = persisted_v2
    analysis = results[0].analysis
    recommendation = analysis.recommendation
    score = analysis.overall_score
    risk = (analysis.risk_score, analysis.risk_class, analysis.reasons, analysis.risks)
    opportunity = assess_opportunity(analysis, price_band=None)
    signals_before = build_signals_for_backtest(
        snapshot=snapshot,
        start_date=TARGET_DATE,
        end_date=TARGET_DATE,
    )
    decision_ids = set(
        Prediction.objects.filter(evidence_role=Prediction.EvidenceRole.DECISION).values_list(
            "id", flat=True
        )
    )
    scenarios = deepcopy(analysis.forecast_scenarios)
    scenarios["horizons"]["6m"]["formula_inputs"]["scenario"]["base"] = -0.9
    StockAnalysis.objects.filter(pk=analysis.pk).update(forecast_scenarios=scenarios)
    analysis.refresh_from_db()
    signals_after = build_signals_for_backtest(
        snapshot=snapshot,
        start_date=TARGET_DATE,
        end_date=TARGET_DATE,
    )

    assert analysis.recommendation == recommendation
    assert analysis.overall_score == score
    assert (analysis.risk_score, analysis.risk_class, analysis.reasons, analysis.risks) == risk
    assert assess_opportunity(analysis, price_band=None) == opportunity
    assert signals_after.equals(signals_before)
    assert (
        set(
            Prediction.objects.filter(evidence_role=Prediction.EvidenceRole.DECISION).values_list(
                "id", flat=True
            )
        )
        == decision_ids
    )
    assert not Prediction.objects.filter(horizon__in=("3y", "5y")).exists()


@pytest.mark.django_db
def test_v2_all_null_outcome_is_unresolved_before_price_read(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _store, _snapshot, results = persisted_v2
    analysis = results[0].analysis
    source = Prediction.objects.filter(
        analysis=analysis,
        method_version=MEDIUM_V2_VERSION,
    ).first()
    assert source is not None
    withheld = Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=source.generated_at,
        target_date=source.target_date,
        issued_on_time=False,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        evidence_grade=UniverseSnapshot.Grade.RESEARCH,
        source_mode=source.source_mode,
        price_provider=source.price_provider,
        price_subject=source.price_subject,
        price_at_prediction=source.price_at_prediction,
        bear_return=None,
        base_return=None,
        bull_return=None,
        probability_positive=None,
        confidence=Decimal("0"),
        confidence_status="insufficient_evidence",
        insufficiency_reason="Insufficient evidence",
        recommendation=source.recommendation,
        overall_score=source.overall_score,
        component_scores=source.component_scores,
        model_version=f"v2-withheld-{uuid4().hex[:8]}",
        method_version=MEDIUM_V2_VERSION,
        config_hash=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
        data_cutoff=source.data_cutoff,
        source_assets=source.source_assets,
        calculation={},
        code_revision=source.code_revision,
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("all-null outcome attempted a price read")

    monkeypatch.setattr(AsOfData, "price_frame_with_diagnostics", forbidden)
    result = evaluate_prediction(
        withheld,
        provider="synthetic",
        evaluation_date=TARGET_DATE + timedelta(days=300),
        evaluation_time=GENERATED_AT + timedelta(days=300),
        store=AssetStore(tmp_path / "outcome"),
    )

    assert result.outcome.status == PredictionOutcome.Status.UNRESOLVED
    assert result.outcome.success is None
    assert result.outcome.actual_return is None


@pytest.mark.django_db
def test_v2_numeric_outcomes_vary_only_own_advisory_metrics_and_stay_unreportable(
    persisted_v2: tuple[AssetStore, UniverseSnapshot, list[Any]],
    tmp_path: Path,
) -> None:
    _store, _snapshot, results = persisted_v2
    analysis = results[0].analysis
    source = Prediction.objects.filter(
        analysis=analysis,
        method_version=MEDIUM_V2_VERSION,
    ).first()
    assert source is not None
    scenarios = (
        ("v2-positive", Decimal("-0.1"), Decimal("0.1"), Decimal("0.3")),
        ("v2-negative", Decimal("-0.3"), Decimal("-0.2"), Decimal("-0.1")),
    )
    predictions: list[Prediction] = []
    for model, bear, base, bull in scenarios:
        predictions.append(
            Prediction.objects.create(
                analysis=analysis,
                listing=analysis.listing,
                generated_at=source.generated_at,
                target_date=source.target_date,
                issued_on_time=False,
                horizon=Prediction.Horizon.SIX_MONTH,
                evidence_role=Prediction.EvidenceRole.ADVISORY,
                evidence_grade=UniverseSnapshot.Grade.RESEARCH,
                source_mode=source.source_mode,
                price_provider="synthetic",
                price_subject=analysis.listing.ticker,
                price_at_prediction=Decimal("100"),
                bear_return=bear,
                base_return=base,
                bull_return=bull,
                probability_positive=None,
                confidence=Decimal("50"),
                confidence_status="empirical_range_only",
                insufficiency_reason="Probability withheld: synthetic evidence",
                recommendation=source.recommendation,
                overall_score=source.overall_score,
                component_scores=source.component_scores,
                model_version=f"{model}-{uuid4().hex[:8]}",
                method_version=MEDIUM_V2_VERSION,
                config_hash=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
                data_cutoff=source.data_cutoff,
                source_assets=source.source_assets,
                calculation={},
                code_revision=source.code_revision,
            )
        )
    calendar = get_calendar("XNYS")
    first = calendar.next_session(TARGET_DATE)
    last = calendar.session_offset(first, 125)
    sessions = [session.date() for session in calendar.sessions_in_range(first, last)]
    outcome_store = AssetStore(tmp_path / "numeric-outcomes")
    frame_dates = [TARGET_DATE, *sessions]
    frame = pl.DataFrame(
        {
            "date": frame_dates,
            "close": [
                100.0,
                *[101.0 + index * (19.0 / 125.0) for index in range(len(sessions))],
            ],
            "volume": [1_000_000 + index for index in range(len(frame_dates))],
        },
        schema_overrides={"date": pl.Date, "close": pl.Float64, "volume": pl.Int64},
    )
    evaluated_at = datetime.combine(
        sessions[-1],
        datetime.max.time(),
        tzinfo=UTC,
    )
    stored = outcome_store.write_frame("outcomes/v2.parquet", frame)
    register_asset(
        provider="synthetic",
        kind="price_history",
        subject=analysis.listing.ticker,
        stored=stored,
        retrieved_at=evaluated_at,
        available_at=evaluated_at,
        period_start=frame_dates[0],
        period_end=sessions[-1],
        metadata={
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )
    reportable_before = Prediction.objects.filter(reportable_prediction_filter()).count()
    outcomes = [
        evaluate_prediction(
            prediction,
            provider="synthetic",
            evaluation_date=sessions[-1],
            evaluation_time=evaluated_at,
            store=outcome_store,
        ).outcome
        for prediction in predictions
    ]

    assert all(outcome.status == PredictionOutcome.Status.MATURED for outcome in outcomes), [
        (outcome.status, outcome.resolution) for outcome in outcomes
    ]
    assert all(outcome.success is None for outcome in outcomes)
    assert outcomes[0].actual_return == outcomes[1].actual_return == Decimal("0.2")
    assert outcomes[0].error != outcomes[1].error
    assert outcomes[0].signed_error != outcomes[1].signed_error
    assert outcomes[0].direction_correct is True
    assert outcomes[1].direction_correct is False
    assert outcomes[0].interval_covered is True
    assert outcomes[1].interval_covered is False
    assert Prediction.objects.filter(reportable_prediction_filter()).count() == (reportable_before)
    assert not list(_advisory_target_summaries())

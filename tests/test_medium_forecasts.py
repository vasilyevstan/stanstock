from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.data.asof import AsOfData
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
from stanstock.research.forecast_config import (
    MediumForecastConfig,
    load_medium_forecast_config,
    medium_forecast_config_hash,
)
from stanstock.research.medium_forecasts import (
    PANEL_SCHEMA,
    _spy_relative_baseline,
    build_medium_forecast_panel,
    build_medium_forecasts,
)
from stanstock.research.models import Prediction
from stanstock.research.opportunities import assess_opportunity
from stanstock.research.service import analyze_snapshot

TARGET_DATE = date(2026, 9, 4)
GENERATED_AT = datetime(2026, 9, 5, 1, tzinfo=UTC)


def test_medium_forecast_config_is_versioned_and_stable() -> None:
    first = load_medium_forecast_config()
    second = load_medium_forecast_config()

    assert first.version == "us-price-medium-v1"
    assert first.horizons["6m"].sessions == 126
    assert first.horizons["12m"].sessions == 252
    assert medium_forecast_config_hash(first) == medium_forecast_config_hash(second)
    assert medium_forecast_config_hash(
        replace(first, minimum_dollar_volume=first.minimum_dollar_volume + 1)
    ) != medium_forecast_config_hash(first)
    # Comparing two loads of the same current file (above) proves stability
    # but is not a pin: pin the exact expected effective hash so an
    # untracked or silently edited default config cannot change frozen
    # `us-price-medium-v1` behavior.
    assert medium_forecast_config_hash(first) == (
        "3461e5228de325454a6b5a902bc4c6174ed7913eef697a86684cf5730dce4028"
    )


def test_medium_forecast_config_rejects_mutable_or_ambiguous_method_definitions() -> None:
    config = load_medium_forecast_config()
    invalid_boundaries = deepcopy(config.raw)
    invalid_boundaries["bucket_boundaries"]["drawdown"] = [-0.2, -0.4]
    with pytest.raises(ValueError, match="increase strictly"):
        MediumForecastConfig.from_mapping(invalid_boundaries)

    invalid_horizon = deepcopy(config.raw)
    invalid_horizon["horizons"]["6m"]["sessions"] = 125
    with pytest.raises(ValueError, match="must remain 126"):
        MediumForecastConfig.from_mapping(invalid_horizon)

    invalid_fallback = deepcopy(config.raw)
    invalid_fallback["fallback_order"][-1]["dimensions"] = ["relative_momentum_bucket"]
    with pytest.raises(ValueError, match="must end with an unconditional"):
        MediumForecastConfig.from_mapping(invalid_fallback)


@pytest.mark.django_db
def test_panel_is_immutable_reproducible_and_uses_non_overlapping_complete_labels(
    tmp_path: Path,
) -> None:
    config = _small_support_config()
    store = AssetStore(tmp_path)
    listings = [_listing("AAA"), _listing("BBB")]
    sessions = _sessions(900)
    future_session = get_calendar("XNYS").next_session(TARGET_DATE).date()
    benchmark_asset = _price_asset(
        store,
        subject="SPY",
        sessions=[*sessions, future_session],
        closes=[100 + index * 0.04 for index in range(len(sessions))] + [9_999],
    )
    listing_assets = [
        _price_asset(
            store,
            subject=listing.ticker,
            sessions=[*sessions, future_session],
            closes=[
                40 + index * (0.03 + listing_index * 0.002) + ((index % 17) - 8) * 0.04
                for index in range(len(sessions))
            ]
            + [8_888],
        )
        for listing_index, listing in enumerate(listings)
    ]
    asof = AsOfData(GENERATED_AT, store)

    first = build_medium_forecast_panel(
        listings=listings,
        asof=asof,
        provider="synthetic",
        benchmark_subject="SPY",
        target_date=TARGET_DATE,
        generated_at=GENERATED_AT,
        run_id=uuid4(),
        config=config,
        config_hash=medium_forecast_config_hash(config),
        scoring_config_version="test-scoring-v1",
        scoring_config_hash="a" * 64,
        universe_snapshot_id=uuid4(),
        universe_slug="test-universe",
        universe_config_hash="b" * 64,
        code_revision="test-revision",
        store=store,
    )
    second = build_medium_forecast_panel(
        listings=listings,
        asof=asof,
        provider="synthetic",
        benchmark_subject="SPY",
        target_date=TARGET_DATE,
        generated_at=GENERATED_AT,
        run_id=uuid4(),
        config=config,
        config_hash=medium_forecast_config_hash(config),
        scoring_config_version="test-scoring-v1",
        scoring_config_hash="a" * 64,
        universe_snapshot_id=uuid4(),
        universe_slug="test-universe",
        universe_config_hash="b" * 64,
        code_revision="test-revision",
        store=store,
    )

    assert first.frame.equals(second.frame)
    assert first.asset.sha256 == second.asset.sha256
    assert first.asset.metadata["calendar"] == "XNYS"
    assert first.asset.metadata["calendar_hash"]
    assert first.asset.metadata["calendar_library_version"]
    assert first.asset.metadata["panel_library_version"]
    assert first.asset.metadata["evidence_bundle_hash"]
    assert first.asset.metadata["config_hash"] == medium_forecast_config_hash(config)
    assert first.asset.metadata["scoring_config_version"] == "test-scoring-v1"
    assert first.asset.metadata["universe_slug"] == "test-universe"
    assert first.asset.metadata["training_evidence_grade"] == "research"
    assert first.asset.metadata["current_universe_survivorship_bias"] is True
    assert len(first.asset.metadata["source_assets"]) == 3
    assert {str(asset.pk) for asset in first.source_assets} == {
        str(benchmark_asset.pk),
        *(str(asset.pk) for asset in listing_assets),
    }

    current_rows = first.frame.filter(pl.col("is_forecast"))
    historical_rows = first.frame.filter(~pl.col("is_forecast"))
    assert current_rows.height == 4
    assert current_rows["anchor_date"].max() == TARGET_DATE
    assert current_rows["average_dollar_volume"].max() < 500_000_000
    latest_label = historical_rows["label_end_date"].max()
    assert isinstance(latest_label, date)
    assert latest_label <= TARGET_DATE
    assert future_session not in set(first.frame["anchor_date"].to_list())
    assert future_session not in set(first.frame["label_end_date"].drop_nulls().to_list())

    calendar = get_calendar("XNYS")
    for horizon, spacing in (("6m", 126), ("12m", 252)):
        anchors = (
            historical_rows.filter(pl.col("horizon") == horizon)["anchor_date"]
            .unique()
            .sort()
            .to_list()
        )
        assert len(anchors) >= 2
        assert all(
            calendar.sessions_distance(left, right) - 1 == spacing
            for left, right in zip(anchors, anchors[1:], strict=False)
        )

    earlier_target = calendar.session_offset(
        calendar.date_to_session(TARGET_DATE),
        -252,
    ).date()
    earlier = build_medium_forecast_panel(
        listings=listings,
        asof=asof,
        provider="synthetic",
        benchmark_subject="SPY",
        target_date=earlier_target,
        generated_at=GENERATED_AT,
        run_id=uuid4(),
        config=config,
        config_hash=medium_forecast_config_hash(config),
        scoring_config_version="test-scoring-v1",
        scoring_config_hash="a" * 64,
        universe_snapshot_id=uuid4(),
        universe_slug="test-universe",
        universe_config_hash="b" * 64,
        code_revision="test-revision",
        store=store,
    )
    for horizon in ("6m", "12m"):
        earlier_pairs = {
            (row["anchor_date"], row["label_end_date"])
            for row in earlier.frame.filter(
                (pl.col("horizon") == horizon) & ~pl.col("is_forecast")
            ).to_dicts()
        }
        later_pairs = {
            (row["anchor_date"], row["label_end_date"])
            for row in first.frame.filter(
                (pl.col("horizon") == horizon) & ~pl.col("is_forecast")
            ).to_dicts()
        }
        assert earlier_pairs < later_pairs

    forecasts = build_medium_forecasts(first.frame, config)
    for listing in listings:
        assert set(forecasts[str(listing.pk)]) == {"6m", "12m"}
        for forecast in forecasts[str(listing.pk)].values():
            assert forecast.scenario.bear is not None
            assert forecast.scenario.base is not None
            assert forecast.scenario.bull is not None
            assert forecast.scenario.bear <= forecast.scenario.base <= forecast.scenario.bull
            assert forecast.calculation["support"]["effective_cohorts"] >= 1


def test_cohort_weighting_prevents_one_date_with_many_listings_from_dominating() -> None:
    config = _small_support_config()
    config = replace(
        config,
        horizons={
            **config.horizons,
            "6m": replace(
                config.horizons["6m"],
                probability_minimum_effective_cohorts=4,
            ),
        },
    )
    rows = [
        _panel_record(
            listing_id=f"crowded-{index}",
            anchor_date=date(2020, 1, 2),
            forward_return=1.0,
        )
        for index in range(10)
    ]
    rows.append(
        _panel_record(
            listing_id="sparse",
            anchor_date=date(2021, 1, 4),
            forward_return=-0.5,
        )
    )
    rows.append(
        _panel_record(
            listing_id="sparse-two",
            anchor_date=date(2022, 1, 3),
            forward_return=-0.4,
        )
    )
    rows.append(
        _panel_record(
            listing_id="current",
            anchor_date=TARGET_DATE,
            is_forecast=True,
            forward_return=None,
        )
    )
    panel = pl.DataFrame(rows, schema=PANEL_SCHEMA, orient="row")

    forecast = build_medium_forecasts(panel, config)["current"]["6m"]

    assert forecast.scenario.base == pytest.approx(-0.4)
    assert forecast.calculation["support"]["raw_matches"] == 12
    assert forecast.calculation["support"]["effective_cohorts"] == 3
    assert forecast.scenario.probability_positive is None
    assert "effective cohorts 3/4" in forecast.scenario.insufficiency_reason


def test_probability_requires_successful_walk_forward_calibration_and_baselines() -> None:
    config = load_medium_forecast_config()
    passing = build_medium_forecasts(_calibration_panel(positive_only=True), config)["current"][
        "6m"
    ]

    assert passing.scenario.probability_positive == pytest.approx(1.0)
    assert passing.scenario.confidence_status == "empirical_calibrated"
    assert passing.calculation["calibration"]["status"] == "passed"
    assert passing.calculation["calibration"]["spy_relative_mean_absolute_error"] == pytest.approx(
        0.0
    )
    assert (
        passing.calculation["calibration"]["spy_relative_baseline_method"]
        == "market_regime_benchmark_median_plus_relative_momentum_excess_median"
    )
    assert passing.calculation["unconditional_baseline"]["base"] == pytest.approx(0.1)

    failing_config = replace(
        config,
        calibration=replace(config.calibration, maximum_brier_score=0.20),
    )
    failing = build_medium_forecasts(
        _calibration_panel(positive_only=False),
        failing_config,
    )["current"]["6m"]

    assert failing.calculation["calibration"]["status"] == "failed"
    assert failing.calculation["calibration"]["brier_score"] == pytest.approx(0.25)
    assert failing.scenario.probability_positive is None
    assert "walk-forward calibration failed" in failing.scenario.insufficiency_reason


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        ("probability_minimum_effective_cohorts", 14, "effective cohorts 13/14"),
        ("probability_minimum_distinct_listings", 31, "distinct listings 30/31"),
        ("probability_minimum_calendar_span_days", 3_000, "calendar span"),
        (
            "probability_minimum_distinct_market_regimes",
            4,
            "matched market regimes 3/4",
        ),
    ],
)
def test_probability_support_gates_withhold_independently(
    field: str,
    value: int,
    expected_reason: str,
) -> None:
    config = load_medium_forecast_config()
    restricted = replace(
        config,
        horizons={
            **config.horizons,
            "6m": replace(config.horizons["6m"], **{field: value}),
        },
    )

    forecast = build_medium_forecasts(
        _calibration_panel(positive_only=True),
        restricted,
    )["current"]["6m"]

    assert forecast.calculation["calibration"]["status"] == "passed"
    assert forecast.scenario.probability_positive is None
    assert expected_reason in forecast.scenario.insufficiency_reason


def test_probability_regime_gate_uses_the_matched_distribution() -> None:
    config = load_medium_forecast_config()
    records = _calibration_panel(positive_only=True).to_dicts()
    anchors = sorted({row["anchor_date"] for row in records if not row["is_forecast"]})
    for row in records:
        if row["is_forecast"]:
            row["market_trend_bucket"] = 0
            row["market_volatility_bucket"] = 0
            continue
        cohort_index = anchors.index(row["anchor_date"])
        regime = 0 if cohort_index < 9 else 1 + cohort_index % 2
        row["market_trend_bucket"] = regime
        row["market_volatility_bucket"] = regime

    forecast = build_medium_forecasts(
        pl.DataFrame(records, schema=PANEL_SCHEMA, orient="row"),
        config,
    )["current"]["6m"]

    assert forecast.calculation["calibration"]["status"] == "passed"
    assert forecast.calculation["probability_evidence"]["distinct_matched_market_regimes"] == 1
    assert forecast.calculation["probability_evidence"]["distinct_panel_market_regimes"] == 3
    assert forecast.scenario.probability_positive is None
    assert "matched market regimes 1/3" in forecast.scenario.insufficiency_reason


def test_spy_relative_baseline_uses_market_and_relative_components() -> None:
    config = _small_support_config()
    rows: list[dict[str, object]] = []
    for index in range(2):
        market_row = _panel_record(
            listing_id=f"market-{index}",
            anchor_date=date(2020 + index, 1, 2),
            forward_return=0.1,
        )
        market_row.update(
            {
                "market_trend_bucket": 0,
                "market_volatility_bucket": 0,
                "relative_momentum_bucket": 1,
                "benchmark_forward_return": 0.2,
                "relative_forward_return": -0.1,
            }
        )
        rows.append(market_row)
        relative_row = _panel_record(
            listing_id=f"relative-{index}",
            anchor_date=date(2022 + index, 1, 2),
            forward_return=0.1,
        )
        relative_row.update(
            {
                "market_trend_bucket": 1,
                "market_volatility_bucket": 1,
                "relative_momentum_bucket": 2,
                "benchmark_forward_return": 0.0,
                "relative_forward_return": 0.1,
            }
        )
        rows.append(relative_row)
    current = _panel_record(
        listing_id="current",
        anchor_date=TARGET_DATE,
        is_forecast=True,
        forward_return=None,
    )
    current.update(
        {
            "market_trend_bucket": 0,
            "market_volatility_bucket": 0,
            "relative_momentum_bucket": 2,
        }
    )

    baseline = _spy_relative_baseline(rows, current, config=config)
    for row in rows:
        if row["market_trend_bucket"] == 0:
            row["benchmark_forward_return"] = 0.4
            row["relative_forward_return"] = -0.3
    revised_baseline = _spy_relative_baseline(rows, current, config=config)

    assert baseline == pytest.approx(0.3)
    assert revised_baseline == pytest.approx(0.5)


@pytest.mark.django_db
@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {
            "return_definition": "split_adjusted_price_return",
            "dividends_included": True,
        },
    ],
)
def test_panel_rejects_price_assets_without_required_return_basis(
    tmp_path: Path,
    metadata: dict[str, object],
) -> None:
    store = AssetStore(tmp_path)
    sessions = _sessions(900)
    _price_asset(
        store,
        subject="SPY",
        sessions=sessions,
        closes=[100 + index * 0.04 for index in range(len(sessions))],
    )
    listing = _listing(f"BAD{len(metadata)}")
    _price_asset(
        store,
        subject=listing.ticker,
        sessions=sessions,
        closes=[40 + index * 0.03 for index in range(len(sessions))],
        metadata=metadata,
    )

    with pytest.raises(
        ValueError,
        match="lacks the required split-adjusted, dividend-excluded return basis",
    ):
        build_medium_forecast_panel(
            listings=[listing],
            asof=AsOfData(GENERATED_AT, store),
            provider="synthetic",
            benchmark_subject="SPY",
            target_date=TARGET_DATE,
            generated_at=GENERATED_AT,
            run_id=uuid4(),
            config=_small_support_config(),
            config_hash="c" * 64,
            scoring_config_version="test-scoring-v1",
            scoring_config_hash="a" * 64,
            universe_snapshot_id=uuid4(),
            universe_slug="test-universe",
            universe_config_hash="b" * 64,
            code_revision="test-revision",
            store=store,
        )


@pytest.mark.django_db
def test_snapshot_analysis_issues_separate_six_and_twelve_month_advisory_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AssetStore(tmp_path)
    universe = Universe.objects.create(
        slug="medium-integration",
        name="Medium integration",
        config_version="test-v1",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="u" * 64,
    )
    sessions = _sessions(1500)
    _price_asset(
        store,
        subject="SPY",
        sessions=sessions,
        closes=[100 + index * 0.03 + ((index % 19) - 9) * 0.02 for index in range(1500)],
    )
    listings = [_listing(f"M{index:02d}") for index in range(12)]
    for listing_index, listing in enumerate(listings):
        UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
        _price_asset(
            store,
            subject=listing.ticker,
            sessions=sessions,
            closes=[
                30
                + listing_index
                + index * (0.025 + listing_index * 0.0005)
                + ((index % (13 + listing_index)) - 6) * 0.03
                for index in range(1500)
            ],
        )

    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=GENERATED_AT,
        target_date=TARGET_DATE,
        issued_on_time=True,
        provider="synthetic",
        benchmark_subject="SPY",
        store=store,
        config_path=default_us_scoring_config_path(),
    )

    assert len(results) == 12
    assert Prediction.objects.count() == 36
    decision_predictions = Prediction.objects.filter(evidence_role="decision")
    advisory_predictions = Prediction.objects.filter(evidence_role="advisory")
    assert set(decision_predictions.values_list("horizon", flat=True)) == {"short"}
    assert set(advisory_predictions.values_list("horizon", flat=True)) == {"6m", "12m"}
    assert all(
        prediction.method_version == "us-price-medium-v1" for prediction in advisory_predictions
    )
    assert all(prediction.issued_on_time for prediction in advisory_predictions)
    assert all(prediction.calculation["panel_asset_id"] for prediction in advisory_predictions)
    assert all(
        prediction.calculation["training_evidence"]["grade"] == "research"
        for prediction in advisory_predictions
    )
    assert all(prediction.base_return is not None for prediction in advisory_predictions)
    assert DataAsset.objects.filter(kind="medium_forecast_panel").count() == 1
    for result in results:
        assert set(result.analysis.forecast_scenarios["horizons"]) == {
            "short",
            "medium",
            "long",
            "6m",
            "12m",
        }
        assert result.analysis.recommendation == result.computation.recommendation

    first_analysis = results[0].analysis
    original_opportunity = assess_opportunity(first_analysis, price_band=None)
    first_analysis.forecast_scenarios["horizons"]["6m"] = {
        "bear": -0.99,
        "base": 9.0,
        "bull": 20.0,
        "probability_positive": 1.0,
    }
    first_analysis.save(update_fields=["forecast_scenarios"])
    first_analysis.refresh_from_db()
    assert first_analysis.recommendation == results[0].computation.recommendation
    assert assess_opportunity(first_analysis, price_band=None) == original_opportunity

    existing_panel_paths = {
        path.relative_to(tmp_path) for path in tmp_path.glob("derived/forecast/medium/**/*.parquet")
    }

    def fail_after_panel(*args: object, **kwargs: object) -> None:
        raise RuntimeError("forced persistence failure")

    monkeypatch.setattr(
        "stanstock.research.service._persist_listing_analysis",
        fail_after_panel,
    )
    with pytest.raises(RuntimeError, match="forced persistence failure"):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=GENERATED_AT,
            target_date=TARGET_DATE,
            issued_on_time=True,
            provider="synthetic",
            benchmark_subject="SPY",
            store=store,
            config_path=default_us_scoring_config_path(),
        )
    assert DataAsset.objects.filter(kind="medium_forecast_panel").count() == 1
    assert {
        path.relative_to(tmp_path) for path in tmp_path.glob("derived/forecast/medium/**/*.parquet")
    } == existing_panel_paths


def _small_support_config() -> MediumForecastConfig:
    config = load_medium_forecast_config()
    horizons = {
        horizon: replace(
            horizon_config,
            minimum_raw_matches=1,
            minimum_effective_cohorts=1,
            minimum_distinct_listings=1,
            shrinkage_prior_cohorts=1,
            probability_minimum_effective_cohorts=1,
            probability_minimum_distinct_listings=1,
            probability_minimum_calendar_span_days=1,
            probability_minimum_distinct_market_regimes=1,
        )
        for horizon, horizon_config in config.horizons.items()
    }
    return replace(
        config,
        minimum_dollar_volume=0,
        horizons=horizons,
        calibration=replace(
            config.calibration,
            minimum_training_cohorts=1,
            minimum_test_cohorts=1,
            maximum_baseline_mae_ratio=10,
            maximum_brier_score=1,
        ),
    )


def _listing(ticker: str) -> Listing:
    company = Company.objects.create(name=f"{ticker} Company", country="US")
    security = Security.objects.create(company=company, name=f"{ticker} Common")
    return Listing.objects.create(
        security=security,
        ticker=ticker,
        exchange_mic="XNAS",
        provider_symbol=ticker,
        currency="USD",
        region=Region.US,
    )


def _sessions(count: int) -> list[date]:
    calendar = get_calendar("XNYS")
    target = calendar.date_to_session(TARGET_DATE)
    first = calendar.session_offset(target, -(count - 1))
    return [session.date() for session in calendar.sessions_in_range(first, target)]


def _price_asset(
    store: AssetStore,
    *,
    subject: str,
    sessions: list[date],
    closes: list[float],
    metadata: dict[str, object] | None = None,
) -> DataAsset:
    frame = pl.DataFrame(
        {
            "date": sessions,
            "close": closes,
            "volume": [2_000_000 + index for index in range(len(sessions))],
        },
        schema_overrides={"date": pl.Date, "close": pl.Float64, "volume": pl.Int64},
    )
    stored = store.write_frame(f"medium-tests/{subject}-{uuid4().hex}.parquet", frame)
    return register_asset(
        provider="synthetic",
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=GENERATED_AT,
        available_at=GENERATED_AT,
        period_start=sessions[0],
        period_end=sessions[-1],
        metadata=(
            {
                "return_definition": "split_adjusted_price_return",
                "dividends_included": False,
            }
            if metadata is None
            else metadata
        ),
    )


def _panel_record(
    *,
    listing_id: str,
    anchor_date: date,
    is_forecast: bool = False,
    forward_return: float | None,
) -> dict[str, object]:
    return {
        "horizon": "6m",
        "anchor_date": anchor_date,
        "label_end_date": None if is_forecast else anchor_date,
        "is_forecast": is_forecast,
        "cohort_id": f"6m:{anchor_date.isoformat()}",
        "listing_id": listing_id,
        "ticker": listing_id,
        "price_asset_id": listing_id,
        "relative_momentum": 0.1,
        "drawdown": -0.1,
        "volatility": 0.2,
        "market_trend": 0.05,
        "market_volatility": 0.2,
        "relative_momentum_bucket": 2,
        "drawdown_bucket": 2,
        "volatility_bucket": 1,
        "market_trend_bucket": 2,
        "market_volatility_bucket": 1,
        "close_vs_sma_50": 0.05,
        "close_vs_sma_200": 0.1,
        "downside_volatility": 0.1,
        "average_dollar_volume": 10_000_000.0,
        "forward_return": forward_return,
        "benchmark_forward_return": (None if forward_return is None else forward_return / 2),
        "relative_forward_return": (None if forward_return is None else forward_return / 2),
        "eligible": True,
        "insufficiency_reason": "",
    }


def _calibration_panel(*, positive_only: bool) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    first_anchor = date(2018, 1, 2)
    for cohort_index in range(13):
        anchor = first_anchor + timedelta(days=182 * cohort_index)
        for listing_index in range(30):
            forward_return = 0.1 if positive_only or listing_index % 2 else -0.1
            row = _panel_record(
                listing_id=f"listing-{listing_index:02d}",
                anchor_date=anchor,
                forward_return=forward_return,
            )
            row["market_trend_bucket"] = cohort_index % 3
            row["market_volatility_bucket"] = cohort_index % 3
            rows.append(row)
    current = _panel_record(
        listing_id="current",
        anchor_date=TARGET_DATE,
        is_forecast=True,
        forward_return=None,
    )
    current["market_trend_bucket"] = 9
    current["market_volatility_bucket"] = 9
    rows.append(current)
    return pl.DataFrame(rows, schema=PANEL_SCHEMA, orient="row")

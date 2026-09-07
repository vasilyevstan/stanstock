from __future__ import annotations

import hashlib
import io
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from importlib.metadata import version as package_version
from typing import Any
from uuid import UUID

import numpy as np
import polars as pl
from django.db import transaction
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.data.asof import AsOfData
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import DataAsset, Listing
from stanstock.research.forecast_config import (
    MATCH_DIMENSIONS,
    MEDIUM_FORECAST_HORIZONS,
    ForecastHorizonConfig,
    MediumForecastConfig,
)
from stanstock.research.types import Scenario

PANEL_PROVIDER = "stanstock"
PANEL_KIND = "medium_forecast_panel"
PANEL_SCHEMA_VERSION = 1
METHOD_NAME = "conditional_empirical_price"
TRADING_SESSIONS_PER_YEAR = 252.0

PANEL_SCHEMA = {
    "horizon": pl.String,
    "anchor_date": pl.Date,
    "label_end_date": pl.Date,
    "is_forecast": pl.Boolean,
    "cohort_id": pl.String,
    "listing_id": pl.String,
    "ticker": pl.String,
    "price_asset_id": pl.String,
    "relative_momentum": pl.Float64,
    "drawdown": pl.Float64,
    "volatility": pl.Float64,
    "market_trend": pl.Float64,
    "market_volatility": pl.Float64,
    "relative_momentum_bucket": pl.Int16,
    "drawdown_bucket": pl.Int16,
    "volatility_bucket": pl.Int16,
    "market_trend_bucket": pl.Int16,
    "market_volatility_bucket": pl.Int16,
    "close_vs_sma_50": pl.Float64,
    "close_vs_sma_200": pl.Float64,
    "downside_volatility": pl.Float64,
    "average_dollar_volume": pl.Float64,
    "forward_return": pl.Float64,
    "benchmark_forward_return": pl.Float64,
    "relative_forward_return": pl.Float64,
    "eligible": pl.Boolean,
    "insufficiency_reason": pl.String,
}


@dataclass(frozen=True, slots=True)
class MediumPanel:
    frame: pl.DataFrame
    asset: DataAsset
    source_assets: tuple[DataAsset, ...]


@dataclass(frozen=True, slots=True)
class MediumForecast:
    scenario: Scenario
    calculation: dict[str, Any]

    def scenario_payload(self) -> dict[str, Any]:
        return {
            **self.scenario.as_dict(),
            "method_version": self.calculation["method_version"],
            "support": self.calculation["support"],
            "current_state": self.calculation["current_state"],
            "return_basis": self.calculation["return_basis"],
            "training_evidence": self.calculation["training_evidence"],
        }


@dataclass(frozen=True, slots=True)
class _Distribution:
    bear: float
    base: float
    bull: float
    probability_positive: float
    raw_matches: int
    effective_cohorts: int
    distinct_listings: int
    calendar_start: str
    calendar_end: str
    regimes: tuple[str, ...]
    dispersion: float


@dataclass(frozen=True, slots=True)
class _Estimate:
    bear: float
    base: float
    bull: float
    probability_positive: float
    fallback_level: str
    shrinkage_weight: float
    matched: _Distribution
    unconditional: _Distribution


def build_medium_forecast_panel(
    *,
    listings: list[Listing],
    asof: AsOfData,
    provider: str,
    benchmark_subject: str,
    target_date: date,
    generated_at: datetime,
    run_id: UUID,
    config: MediumForecastConfig,
    config_hash: str,
    scoring_config_version: str,
    scoring_config_hash: str,
    universe_snapshot_id: UUID,
    universe_slug: str,
    universe_config_hash: str,
    code_revision: str,
    store: AssetStore,
) -> MediumPanel:
    calendar = get_calendar(config.calendar)
    target_session = calendar.date_to_session(target_date, direction="none")
    epoch_session = calendar.date_to_session(config.fixed_epoch, direction="none")
    if target_session < epoch_session:
        raise ValueError("Forecast target date precedes the configured fixed epoch")
    calendar_sessions = tuple(
        session.date() for session in calendar.sessions_in_range(epoch_session, target_session)
    )
    session_index = {session: index for index, session in enumerate(calendar_sessions)}
    calendar_hash = _hash_json([session.isoformat() for session in calendar_sessions])

    benchmark_asset = asof.latest_asset(
        provider=provider,
        kind="price_history",
        subject=benchmark_subject,
    )
    _validate_price_basis(benchmark_asset)
    benchmark_frame = asof.price_frame(
        provider=provider,
        subject=benchmark_subject,
        through_date=target_date,
    )
    benchmark_prices = _price_observations(benchmark_frame)

    source_assets = [benchmark_asset]
    listing_inputs: list[tuple[Listing, DataAsset, dict[date, tuple[float, float | None]]]] = []
    for listing in sorted(listings, key=lambda item: str(item.pk)):
        subject = listing.provider_symbol or listing.ticker
        asset = asof.latest_asset(provider=provider, kind="price_history", subject=subject)
        _validate_price_basis(asset)
        frame = asof.price_frame(
            provider=provider,
            subject=subject,
            through_date=target_date,
        )
        source_assets.append(asset)
        listing_inputs.append((listing, asset, _price_observations(frame)))

    rows: list[dict[str, object]] = []
    for horizon in MEDIUM_FORECAST_HORIZONS:
        horizon_config = config.horizons[horizon]
        historical_anchors = _historical_anchors(
            benchmark_prices=benchmark_prices,
            calendar_sessions=calendar_sessions,
            session_index=session_index,
            horizon_sessions=horizon_config.sessions,
            feature_lookback=config.feature_windows.maximum,
            target_date=target_date,
        )
        for anchor_date, label_end_date in historical_anchors:
            for listing, asset, prices in listing_inputs:
                rows.append(
                    _panel_row(
                        horizon=horizon,
                        anchor_date=anchor_date,
                        label_end_date=label_end_date,
                        is_forecast=False,
                        listing=listing,
                        price_asset=asset,
                        prices=prices,
                        benchmark_prices=benchmark_prices,
                        calendar_sessions=calendar_sessions,
                        session_index=session_index,
                        config=config,
                    )
                )
        for listing, asset, prices in listing_inputs:
            rows.append(
                _panel_row(
                    horizon=horizon,
                    anchor_date=target_date,
                    label_end_date=None,
                    is_forecast=True,
                    listing=listing,
                    price_asset=asset,
                    prices=prices,
                    benchmark_prices=benchmark_prices,
                    calendar_sessions=calendar_sessions,
                    session_index=session_index,
                    config=config,
                )
            )

    frame = pl.DataFrame(rows, schema=PANEL_SCHEMA, orient="row").sort(
        "horizon",
        "anchor_date",
        "listing_id",
    )
    payload = _parquet_bytes(frame)
    content_hash = hashlib.sha256(payload).hexdigest()
    relative_path = (
        f"derived/forecast/medium/{target_date.isoformat()}/"
        f"{run_id.hex}-{content_hash[:12]}.parquet"
    )
    stored = store.write_bytes(relative_path, payload)
    deduped_sources = _dedupe_assets(source_assets)
    source_manifest = [_asset_identity(asset) for asset in deduped_sources]
    source_manifest_hash = _hash_json(source_manifest)
    evidence_bundle_hash = _hash_json(
        {
            "calendar_hash": calendar_hash,
            "code_revision": code_revision,
            "content_sha256": stored.sha256,
            "forecast_config_hash": config_hash,
            "scoring_config_hash": scoring_config_hash,
            "source_manifest_hash": source_manifest_hash,
            "universe_config_hash": universe_config_hash,
        }
    )
    anchor_dates: list[date] = []
    for row in rows:
        raw_anchor_date = row["anchor_date"]
        if isinstance(raw_anchor_date, date):
            anchor_dates.append(raw_anchor_date)
    try:
        with transaction.atomic():
            asset = register_asset(
                provider=PANEL_PROVIDER,
                kind=PANEL_KIND,
                subject=str(run_id),
                stored=stored,
                retrieved_at=generated_at,
                available_at=generated_at,
                period_start=min(anchor_dates) if anchor_dates else target_date,
                period_end=target_date,
                metadata={
                    "schema_version": PANEL_SCHEMA_VERSION,
                    "method_version": config.version,
                    "config_hash": config_hash,
                    "code_revision": code_revision,
                    "calendar": config.calendar,
                    "calendar_library_version": package_version("exchange-calendars"),
                    "panel_library_version": package_version("polars"),
                    "fixed_epoch": config.fixed_epoch.isoformat(),
                    "calendar_hash": calendar_hash,
                    "target_date": target_date.isoformat(),
                    "universe_snapshot_id": str(universe_snapshot_id),
                    "universe_slug": universe_slug,
                    "universe_config_hash": universe_config_hash,
                    "scoring_config_version": scoring_config_version,
                    "scoring_config_hash": scoring_config_hash,
                    "return_definition": config.return_basis,
                    "dividends_included": config.dividends_included,
                    "training_evidence_grade": "research",
                    "current_universe_survivorship_bias": True,
                    "usage_scope": "private_single_user_research",
                    "row_count": frame.height,
                    "content_sha256": stored.sha256,
                    "source_manifest_hash": source_manifest_hash,
                    "evidence_bundle_hash": evidence_bundle_hash,
                    "source_assets": source_manifest,
                },
            )
    except Exception:
        if not DataAsset.objects.filter(relative_path=relative_path).exists():
            store.resolve(relative_path).unlink(missing_ok=True)
        raise
    return MediumPanel(
        frame=frame,
        asset=asset,
        source_assets=tuple(deduped_sources),
    )


def build_medium_forecasts(
    panel: pl.DataFrame,
    config: MediumForecastConfig,
) -> dict[str, dict[str, MediumForecast]]:
    records = panel.to_dicts()
    calibrations = {
        horizon: _calibrate_horizon(records, horizon=horizon, config=config)
        for horizon in MEDIUM_FORECAST_HORIZONS
    }
    listing_ids = sorted({str(row["listing_id"]) for row in records if bool(row["is_forecast"])})
    forecasts: dict[str, dict[str, MediumForecast]] = {}
    for listing_id in listing_ids:
        forecasts[listing_id] = {}
        for horizon in MEDIUM_FORECAST_HORIZONS:
            current = next(
                (
                    row
                    for row in records
                    if str(row["listing_id"]) == listing_id
                    and str(row["horizon"]) == horizon
                    and bool(row["is_forecast"])
                ),
                None,
            )
            forecasts[listing_id][horizon] = _forecast_for_state(
                records,
                current=current,
                horizon=horizon,
                config=config,
                calibration=calibrations[horizon],
            )
    return forecasts


def _forecast_for_state(
    records: list[dict[str, Any]],
    *,
    current: dict[str, Any] | None,
    horizon: str,
    config: MediumForecastConfig,
    calibration: dict[str, Any],
) -> MediumForecast:
    horizon_config = config.horizons[horizon]
    training = _training_rows(records, horizon)
    if current is None:
        return _missing_forecast(
            horizon=horizon,
            config=config,
            reason="Current forecast state is missing from the immutable panel",
            calibration=calibration,
        )
    if not current["eligible"]:
        return _missing_forecast(
            horizon=horizon,
            config=config,
            reason=str(current["insufficiency_reason"]),
            calibration=calibration,
            current_state=_state_payload(current),
        )
    estimate = _estimate_distribution(
        training,
        current=current,
        horizon_config=horizon_config,
        config=config,
    )
    if estimate is None:
        unconditional = _distribution(training, config=config)
        support = _support_payload(unconditional, fallback_level="unavailable", shrinkage_weight=0)
        return _missing_forecast(
            horizon=horizon,
            config=config,
            reason=_support_failure_reason(horizon, unconditional, horizon_config),
            calibration=calibration,
            current_state=_state_payload(current),
            support=support,
        )

    probability_reasons, probability_evidence = _probability_gate(
        estimate,
        horizon_config=horizon_config,
        calibration=calibration,
    )
    probability: float | None = estimate.probability_positive
    if probability_reasons:
        probability = None

    confidence = min(80.0, 20.0 + 60.0 * estimate.shrinkage_weight)
    reason = (
        "Probability withheld: " + "; ".join(probability_reasons) if probability_reasons else ""
    )
    scenario = Scenario(
        bear=max(-1.0, estimate.bear),
        base=max(-1.0, estimate.base),
        bull=max(-1.0, estimate.bull),
        probability_positive=probability,
        confidence=confidence,
        confidence_status=(
            "empirical_calibrated" if probability is not None else "empirical_range_only"
        ),
        insufficiency_reason=reason,
        method=METHOD_NAME,
    )
    return MediumForecast(
        scenario=scenario,
        calculation={
            "schema_version": 1,
            "method": METHOD_NAME,
            "method_version": config.version,
            "forecast_horizon": horizon,
            "horizon_sessions": horizon_config.sessions,
            "current_state": _state_payload(current),
            "support": _support_payload(
                estimate.matched,
                fallback_level=estimate.fallback_level,
                shrinkage_weight=estimate.shrinkage_weight,
            ),
            "probability_evidence": probability_evidence,
            "unconditional_baseline": _distribution_payload(estimate.unconditional),
            "calibration": calibration,
            "formula_inputs": {
                "quantiles": {
                    "bear": config.quantiles.bear,
                    "base": config.quantiles.base,
                    "bull": config.quantiles.bull,
                },
                "matched_scenario": {
                    "bear": estimate.matched.bear,
                    "base": estimate.matched.base,
                    "bull": estimate.matched.bull,
                    "probability_positive": estimate.matched.probability_positive,
                },
                "unconditional_scenario": {
                    "bear": estimate.unconditional.bear,
                    "base": estimate.unconditional.base,
                    "bull": estimate.unconditional.bull,
                    "probability_positive": estimate.unconditional.probability_positive,
                },
                "scenario": {
                    "bear": scenario.bear,
                    "base": scenario.base,
                    "bull": scenario.bull,
                    "probability_positive": scenario.probability_positive,
                },
            },
            "return_basis": config.return_basis,
            "dividends_included": config.dividends_included,
            "training_evidence": {
                "grade": "research",
                "current_universe_survivorship_bias": True,
                "label_policy": "complete_horizon_ending_on_or_before_forecast_target",
                "cohort_policy": "fixed_epoch_non_overlapping",
            },
        },
    )


def _missing_forecast(
    *,
    horizon: str,
    config: MediumForecastConfig,
    reason: str,
    calibration: dict[str, Any],
    current_state: dict[str, Any] | None = None,
    support: dict[str, Any] | None = None,
) -> MediumForecast:
    scenario = Scenario(
        bear=None,
        base=None,
        bull=None,
        probability_positive=None,
        confidence=0.0,
        confidence_status="insufficient_evidence",
        insufficiency_reason=reason,
        method=METHOD_NAME,
    )
    return MediumForecast(
        scenario=scenario,
        calculation={
            "schema_version": 1,
            "method": METHOD_NAME,
            "method_version": config.version,
            "forecast_horizon": horizon,
            "horizon_sessions": config.horizons[horizon].sessions,
            "current_state": current_state or {},
            "support": support or {},
            "calibration": calibration,
            "formula_inputs": {
                "scenario": {
                    "bear": None,
                    "base": None,
                    "bull": None,
                    "probability_positive": None,
                }
            },
            "return_basis": config.return_basis,
            "dividends_included": config.dividends_included,
            "training_evidence": {
                "grade": "research",
                "current_universe_survivorship_bias": True,
                "label_policy": "complete_horizon_ending_on_or_before_forecast_target",
                "cohort_policy": "fixed_epoch_non_overlapping",
            },
        },
    )


def _historical_anchors(
    *,
    benchmark_prices: dict[date, tuple[float, float | None]],
    calendar_sessions: tuple[date, ...],
    session_index: dict[date, int],
    horizon_sessions: int,
    feature_lookback: int,
    target_date: date,
) -> list[tuple[date, date]]:
    anchors: list[tuple[date, date]] = []
    target_index = session_index[target_date]
    for anchor_date in sorted(benchmark_prices):
        index = session_index.get(anchor_date)
        if (
            index is None
            or index < feature_lookback
            or index + horizon_sessions > target_index
            or index % horizon_sessions != 0
        ):
            continue
        label_end_date = calendar_sessions[index + horizon_sessions]
        if label_end_date in benchmark_prices:
            anchors.append((anchor_date, label_end_date))
    return anchors


def _panel_row(
    *,
    horizon: str,
    anchor_date: date,
    label_end_date: date | None,
    is_forecast: bool,
    listing: Listing,
    price_asset: DataAsset,
    prices: dict[date, tuple[float, float | None]],
    benchmark_prices: dict[date, tuple[float, float | None]],
    calendar_sessions: tuple[date, ...],
    session_index: dict[date, int],
    config: MediumForecastConfig,
) -> dict[str, object]:
    state, missing = _state_at(
        anchor_date=anchor_date,
        prices=prices,
        benchmark_prices=benchmark_prices,
        calendar_sessions=calendar_sessions,
        session_index=session_index,
        config=config,
    )
    forward_return = None
    benchmark_forward_return = None
    relative_forward_return = None
    if label_end_date is not None:
        stock_anchor = _close(prices, anchor_date)
        stock_end = _close(prices, label_end_date)
        benchmark_anchor = _close(benchmark_prices, anchor_date)
        benchmark_end = _close(benchmark_prices, label_end_date)
        if None not in (stock_anchor, stock_end, benchmark_anchor, benchmark_end):
            assert stock_anchor is not None
            assert stock_end is not None
            assert benchmark_anchor is not None
            assert benchmark_end is not None
            forward_return = stock_end / stock_anchor - 1.0
            benchmark_forward_return = benchmark_end / benchmark_anchor - 1.0
            relative_forward_return = forward_return - benchmark_forward_return
        else:
            missing.append("complete forward price label")

    eligible = not missing
    row: dict[str, object] = {
        "horizon": horizon,
        "anchor_date": anchor_date,
        "label_end_date": label_end_date,
        "is_forecast": is_forecast,
        "cohort_id": f"{horizon}:{anchor_date.isoformat()}",
        "listing_id": str(listing.pk),
        "ticker": listing.ticker,
        "price_asset_id": str(price_asset.pk),
        **state,
        "forward_return": forward_return,
        "benchmark_forward_return": benchmark_forward_return,
        "relative_forward_return": relative_forward_return,
        "eligible": eligible,
        "insufficiency_reason": (
            "" if eligible else "Insufficient medium forecast inputs: " + ", ".join(sorted(missing))
        ),
    }
    return row


def _state_at(
    *,
    anchor_date: date,
    prices: dict[date, tuple[float, float | None]],
    benchmark_prices: dict[date, tuple[float, float | None]],
    calendar_sessions: tuple[date, ...],
    session_index: dict[date, int],
    config: MediumForecastConfig,
) -> tuple[dict[str, object], list[str]]:
    windows = config.feature_windows
    anchor_index = session_index[anchor_date]
    missing: list[str] = []
    stock_anchor = _close(prices, anchor_date)
    benchmark_anchor = _close(benchmark_prices, anchor_date)
    momentum_start_index = anchor_index - windows.momentum_sessions
    stock_momentum_start = (
        _close(prices, calendar_sessions[momentum_start_index])
        if momentum_start_index >= 0
        else None
    )
    benchmark_momentum_start = (
        _close(benchmark_prices, calendar_sessions[momentum_start_index])
        if momentum_start_index >= 0
        else None
    )
    relative_momentum = None
    if None not in (
        stock_anchor,
        benchmark_anchor,
        stock_momentum_start,
        benchmark_momentum_start,
    ):
        assert stock_anchor is not None
        assert benchmark_anchor is not None
        assert stock_momentum_start is not None
        assert benchmark_momentum_start is not None
        relative_momentum = (stock_anchor / stock_momentum_start - 1.0) - (
            benchmark_anchor / benchmark_momentum_start - 1.0
        )
    else:
        missing.append("252-session relative momentum")

    drawdown_values = _window_closes(
        prices,
        calendar_sessions,
        anchor_index,
        windows.drawdown_sessions,
        config.minimum_history_coverage,
    )
    drawdown = None
    if stock_anchor is not None and drawdown_values:
        peak = max(drawdown_values)
        if peak > 0:
            drawdown = stock_anchor / peak - 1.0
    if drawdown is None:
        missing.append("52-week drawdown")

    volatility_returns = _window_returns(
        prices,
        calendar_sessions,
        anchor_index,
        windows.volatility_sessions,
        config.minimum_history_coverage,
    )
    volatility = _annualized_volatility(volatility_returns)
    downside_volatility = _downside_volatility(volatility_returns)
    if volatility is None:
        missing.append("trailing volatility")

    market_trend_values = _window_closes(
        benchmark_prices,
        calendar_sessions,
        anchor_index,
        windows.long_trend_sessions,
        config.minimum_history_coverage,
    )
    market_trend = None
    if benchmark_anchor is not None and market_trend_values:
        moving_average = float(np.mean(market_trend_values))
        if moving_average > 0:
            market_trend = benchmark_anchor / moving_average - 1.0
    if market_trend is None:
        missing.append("SPY 200-session trend")

    market_volatility_returns = _window_returns(
        benchmark_prices,
        calendar_sessions,
        anchor_index,
        windows.volatility_sessions,
        config.minimum_history_coverage,
    )
    market_volatility = _annualized_volatility(market_volatility_returns)
    if market_volatility is None:
        missing.append("SPY trailing volatility")

    short_trend = _close_vs_average(
        prices,
        stock_anchor,
        calendar_sessions,
        anchor_index,
        windows.short_trend_sessions,
        config.minimum_history_coverage,
    )
    long_trend = _close_vs_average(
        prices,
        stock_anchor,
        calendar_sessions,
        anchor_index,
        windows.long_trend_sessions,
        config.minimum_history_coverage,
    )
    average_dollar_volume = _average_dollar_volume(
        prices,
        calendar_sessions,
        anchor_index,
        windows.liquidity_sessions,
        config.minimum_history_coverage,
    )
    if average_dollar_volume is None:
        missing.append("20-session dollar liquidity")
    elif average_dollar_volume < config.minimum_dollar_volume:
        missing.append(f"20-session dollar liquidity below {config.minimum_dollar_volume:,.0f}")

    state: dict[str, object] = {
        "relative_momentum": relative_momentum,
        "drawdown": drawdown,
        "volatility": volatility,
        "market_trend": market_trend,
        "market_volatility": market_volatility,
        "relative_momentum_bucket": _bucket(
            relative_momentum,
            config.bucket_boundaries["relative_momentum"],
        ),
        "drawdown_bucket": _bucket(
            drawdown,
            config.bucket_boundaries["drawdown"],
        ),
        "volatility_bucket": _bucket(
            volatility,
            config.bucket_boundaries["volatility"],
        ),
        "market_trend_bucket": _bucket(
            market_trend,
            config.bucket_boundaries["market_trend"],
        ),
        "market_volatility_bucket": _bucket(
            market_volatility,
            config.bucket_boundaries["market_volatility"],
        ),
        "close_vs_sma_50": short_trend,
        "close_vs_sma_200": long_trend,
        "downside_volatility": downside_volatility,
        "average_dollar_volume": average_dollar_volume,
    }
    return state, missing


def _estimate_distribution(
    training: list[dict[str, Any]],
    *,
    current: dict[str, Any],
    horizon_config: ForecastHorizonConfig,
    config: MediumForecastConfig,
) -> _Estimate | None:
    unconditional = _distribution(training, config=config)
    if not _support_sufficient(unconditional, horizon_config):
        return None
    for fallback in config.fallback_order:
        matched_rows = [
            row
            for row in training
            if all(row[dimension] == current[dimension] for dimension in fallback.dimensions)
        ]
        matched = _distribution(matched_rows, config=config)
        if not _support_sufficient(matched, horizon_config):
            continue
        shrinkage_weight = (
            0.0
            if not fallback.dimensions
            else matched.effective_cohorts
            / (matched.effective_cohorts + horizon_config.shrinkage_prior_cohorts)
        )
        bear = _blend(matched.bear, unconditional.bear, shrinkage_weight)
        base = _blend(matched.base, unconditional.base, shrinkage_weight)
        bull = _blend(matched.bull, unconditional.bull, shrinkage_weight)
        ordered = sorted((bear, base, bull))
        return _Estimate(
            bear=ordered[0],
            base=ordered[1],
            bull=ordered[2],
            probability_positive=_clamp_probability(
                _blend(
                    matched.probability_positive,
                    unconditional.probability_positive,
                    shrinkage_weight,
                )
            ),
            fallback_level=fallback.name,
            shrinkage_weight=shrinkage_weight,
            matched=matched,
            unconditional=unconditional,
        )
    return None


def _distribution(
    rows: list[dict[str, Any]],
    *,
    config: MediumForecastConfig,
) -> _Distribution:
    usable = [row for row in rows if row["eligible"] and row["forward_return"] is not None]
    if not usable:
        return _Distribution(0, 0, 0, 0, 0, 0, 0, "", "", (), 0)
    cohort_counts = Counter(str(row["cohort_id"]) for row in usable)
    values = np.asarray([float(row["forward_return"]) for row in usable], dtype=np.float64)
    weights = np.asarray(
        [1.0 / cohort_counts[str(row["cohort_id"])] for row in usable],
        dtype=np.float64,
    )
    bear, base, bull = (
        _weighted_quantile(values, weights, quantile)
        for quantile in (
            config.quantiles.bear,
            config.quantiles.base,
            config.quantiles.bull,
        )
    )
    probability_positive = float(np.average(values > 0.0, weights=weights))
    weighted_mean = float(np.average(values, weights=weights))
    dispersion = float(np.sqrt(np.average((values - weighted_mean) ** 2, weights=weights)))
    dates = sorted({row["anchor_date"] for row in usable})
    regimes = tuple(
        sorted(
            {f"{row['market_trend_bucket']}:{row['market_volatility_bucket']}" for row in usable}
        )
    )
    return _Distribution(
        bear=bear,
        base=base,
        bull=bull,
        probability_positive=probability_positive,
        raw_matches=len(usable),
        effective_cohorts=len(cohort_counts),
        distinct_listings=len({str(row["listing_id"]) for row in usable}),
        calendar_start=dates[0].isoformat(),
        calendar_end=dates[-1].isoformat(),
        regimes=regimes,
        dispersion=dispersion,
    )


def _calibrate_horizon(
    records: list[dict[str, Any]],
    *,
    horizon: str,
    config: MediumForecastConfig,
) -> dict[str, Any]:
    training = _training_rows(records, horizon)
    cohorts = sorted({row["anchor_date"] for row in training})
    calibration = config.calibration
    cohort_metrics: list[tuple[float, float, float, float]] = []
    for cohort_index in range(calibration.minimum_training_cohorts, len(cohorts)):
        test_cohort = cohorts[cohort_index]
        prior = [row for row in training if row["anchor_date"] < test_cohort]
        test_rows = [row for row in training if row["anchor_date"] == test_cohort]
        unconditional = _distribution(prior, config=config)
        if not _support_sufficient(unconditional, config.horizons[horizon]):
            continue
        model_errors: list[float] = []
        unconditional_errors: list[float] = []
        relative_errors: list[float] = []
        brier_scores: list[float] = []
        for row in test_rows:
            estimate = _estimate_distribution(
                prior,
                current=row,
                horizon_config=config.horizons[horizon],
                config=config,
            )
            if estimate is None:
                continue
            actual = float(row["forward_return"])
            model_errors.append(abs(actual - estimate.base))
            unconditional_errors.append(abs(actual - unconditional.base))
            relative_prediction = _spy_relative_baseline(prior, row, config=config)
            relative_errors.append(abs(actual - relative_prediction))
            observed_positive = 1.0 if actual > 0 else 0.0
            brier_scores.append((estimate.probability_positive - observed_positive) ** 2)
        if model_errors:
            cohort_metrics.append(
                (
                    float(np.mean(model_errors)),
                    float(np.mean(unconditional_errors)),
                    float(np.mean(relative_errors)),
                    float(np.mean(brier_scores)),
                )
            )
    test_cohorts = len(cohort_metrics)
    if test_cohorts < calibration.minimum_test_cohorts:
        return {
            "status": (
                f"insufficient ({test_cohorts}/{calibration.minimum_test_cohorts} test cohorts)"
            ),
            "passed": False,
            "test_cohorts": test_cohorts,
            "mean_absolute_error": None,
            "unconditional_mean_absolute_error": None,
            "spy_relative_mean_absolute_error": None,
            "spy_relative_baseline_method": (
                "market_regime_benchmark_median_plus_relative_momentum_excess_median"
            ),
            "brier_score": None,
        }
    metrics = np.asarray(cohort_metrics, dtype=np.float64)
    model_mae = float(np.mean(metrics[:, 0]))
    unconditional_mae = float(np.mean(metrics[:, 1]))
    relative_mae = float(np.mean(metrics[:, 2]))
    brier_score = float(np.mean(metrics[:, 3]))
    ratio = calibration.maximum_baseline_mae_ratio
    passed = (
        model_mae <= unconditional_mae * ratio
        and model_mae <= relative_mae * ratio
        and brier_score <= calibration.maximum_brier_score
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "test_cohorts": test_cohorts,
        "mean_absolute_error": model_mae,
        "unconditional_mean_absolute_error": unconditional_mae,
        "spy_relative_mean_absolute_error": relative_mae,
        "spy_relative_baseline_method": (
            "market_regime_benchmark_median_plus_relative_momentum_excess_median"
        ),
        "brier_score": brier_score,
        "maximum_baseline_mae_ratio": ratio,
        "maximum_brier_score": calibration.maximum_brier_score,
    }


def _spy_relative_baseline(
    training: list[dict[str, Any]],
    current: dict[str, Any],
    *,
    config: MediumForecastConfig,
) -> float:
    benchmark_matches = [
        row
        for row in training
        if row["benchmark_forward_return"] is not None
        and row["market_trend_bucket"] == current["market_trend_bucket"]
        and row["market_volatility_bucket"] == current["market_volatility_bucket"]
    ]
    if not benchmark_matches:
        benchmark_matches = [row for row in training if row["benchmark_forward_return"] is not None]
    relative_matches = [
        row
        for row in training
        if row["relative_forward_return"] is not None
        and row["relative_momentum_bucket"] == current["relative_momentum_bucket"]
    ]
    if not relative_matches:
        relative_matches = [row for row in training if row["relative_forward_return"] is not None]
    benchmark_distribution = _value_distribution(
        benchmark_matches,
        value_key="benchmark_forward_return",
        config=config,
    )
    relative_distribution = _value_distribution(
        relative_matches,
        value_key="relative_forward_return",
        config=config,
    )
    return benchmark_distribution.base + relative_distribution.base


def _value_distribution(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    config: MediumForecastConfig,
) -> _Distribution:
    normalized = [{**row, "forward_return": row[value_key]} for row in rows]
    return _distribution(normalized, config=config)


def _training_rows(records: list[dict[str, Any]], horizon: str) -> list[dict[str, Any]]:
    return [
        row
        for row in records
        if str(row["horizon"]) == horizon
        and not bool(row["is_forecast"])
        and bool(row["eligible"])
        and row["forward_return"] is not None
    ]


def _support_sufficient(
    distribution: _Distribution,
    config: ForecastHorizonConfig,
) -> bool:
    return (
        distribution.raw_matches >= config.minimum_raw_matches
        and distribution.effective_cohorts >= config.minimum_effective_cohorts
        and distribution.distinct_listings >= config.minimum_distinct_listings
    )


def _probability_gate(
    estimate: _Estimate,
    *,
    horizon_config: ForecastHorizonConfig,
    calibration: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    calendar_span_days = _calendar_span_days(estimate.matched)
    distinct_matched_market_regimes = len(estimate.matched.regimes)
    distinct_panel_market_regimes = len(estimate.unconditional.regimes)
    reasons: list[str] = []
    if estimate.matched.effective_cohorts < horizon_config.probability_minimum_effective_cohorts:
        reasons.append(
            "effective cohorts "
            f"{estimate.matched.effective_cohorts}/"
            f"{horizon_config.probability_minimum_effective_cohorts}"
        )
    if estimate.matched.distinct_listings < horizon_config.probability_minimum_distinct_listings:
        reasons.append(
            "distinct listings "
            f"{estimate.matched.distinct_listings}/"
            f"{horizon_config.probability_minimum_distinct_listings}"
        )
    if calendar_span_days < horizon_config.probability_minimum_calendar_span_days:
        reasons.append(
            "calendar span "
            f"{calendar_span_days}/"
            f"{horizon_config.probability_minimum_calendar_span_days} days"
        )
    if distinct_matched_market_regimes < horizon_config.probability_minimum_distinct_market_regimes:
        reasons.append(
            "matched market regimes "
            f"{distinct_matched_market_regimes}/"
            f"{horizon_config.probability_minimum_distinct_market_regimes}"
        )
    if not calibration["passed"]:
        reasons.append(f"walk-forward calibration {calibration['status']}")
    return reasons, {
        "calendar_span_days": calendar_span_days,
        "distinct_matched_market_regimes": distinct_matched_market_regimes,
        "distinct_panel_market_regimes": distinct_panel_market_regimes,
        "market_regime_source": "matched_conditional_distribution",
        "minimum_effective_cohorts": horizon_config.probability_minimum_effective_cohorts,
        "minimum_distinct_listings": horizon_config.probability_minimum_distinct_listings,
        "minimum_calendar_span_days": horizon_config.probability_minimum_calendar_span_days,
        "minimum_distinct_market_regimes": (
            horizon_config.probability_minimum_distinct_market_regimes
        ),
    }


def _calendar_span_days(distribution: _Distribution) -> int:
    if not distribution.calendar_start or not distribution.calendar_end:
        return 0
    return (
        date.fromisoformat(distribution.calendar_end)
        - date.fromisoformat(distribution.calendar_start)
    ).days


def _support_failure_reason(
    horizon: str,
    support: _Distribution,
    config: ForecastHorizonConfig,
) -> str:
    return (
        f"Insufficient non-overlapping {horizon} evidence: "
        f"{support.raw_matches}/{config.minimum_raw_matches} observations, "
        f"{support.effective_cohorts}/{config.minimum_effective_cohorts} cohorts, "
        f"{support.distinct_listings}/{config.minimum_distinct_listings} listings"
    )


def _support_payload(
    distribution: _Distribution,
    *,
    fallback_level: str,
    shrinkage_weight: float,
) -> dict[str, Any]:
    return {
        "raw_matches": distribution.raw_matches,
        "effective_cohorts": distribution.effective_cohorts,
        "distinct_listings": distribution.distinct_listings,
        "calendar_start": distribution.calendar_start,
        "calendar_end": distribution.calendar_end,
        "market_regimes": list(distribution.regimes),
        "fallback_level": fallback_level,
        "shrinkage_weight": shrinkage_weight,
        "dispersion": distribution.dispersion,
    }


def _distribution_payload(distribution: _Distribution) -> dict[str, Any]:
    return {
        "bear": distribution.bear,
        "base": distribution.base,
        "bull": distribution.bull,
        "probability_positive": distribution.probability_positive,
        "raw_observations": distribution.raw_matches,
        "effective_cohorts": distribution.effective_cohorts,
        "distinct_listings": distribution.distinct_listings,
        "dispersion": distribution.dispersion,
    }


def _state_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: row[key]
        for key in (
            "relative_momentum",
            "drawdown",
            "volatility",
            "market_trend",
            "market_volatility",
            *MATCH_DIMENSIONS,
            "close_vs_sma_50",
            "close_vs_sma_200",
            "downside_volatility",
            "average_dollar_volume",
        )
    }
    anchor_date = row.get("anchor_date")
    payload["anchor_date"] = (
        anchor_date.isoformat() if isinstance(anchor_date, date) else str(anchor_date or "")
    )
    return payload


def _price_observations(frame: pl.DataFrame) -> dict[date, tuple[float, float | None]]:
    if "date" not in frame.columns or "close" not in frame.columns:
        raise ValueError("Price history requires date and close columns")
    volume_expression = (
        pl.col("volume").cast(pl.Float64, strict=False)
        if "volume" in frame.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    normalized = (
        frame.select(
            pl.col("date").cast(pl.Date, strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
            volume_expression.alias("volume"),
        )
        .filter(
            pl.col("date").is_not_null()
            & pl.col("close").is_not_null()
            & pl.col("close").is_finite()
            & (pl.col("close") > 0)
        )
        .sort("date")
    )
    if normalized["date"].n_unique() != normalized.height:
        raise ValueError("Price history contains duplicate session dates")
    return {
        row["date"]: (
            float(row["close"]),
            (
                float(row["volume"])
                if row["volume"] is not None and math.isfinite(float(row["volume"]))
                else None
            ),
        )
        for row in normalized.iter_rows(named=True)
    }


def _window_closes(
    observations: dict[date, tuple[float, float | None]],
    calendar_sessions: tuple[date, ...],
    anchor_index: int,
    window: int,
    minimum_coverage: float,
) -> list[float]:
    start = anchor_index - window + 1
    if start < 0:
        return []
    dates = calendar_sessions[start : anchor_index + 1]
    values = [observations[session][0] for session in dates if session in observations]
    required = math.ceil(window * minimum_coverage)
    return values if len(values) >= required else []


def _window_returns(
    observations: dict[date, tuple[float, float | None]],
    calendar_sessions: tuple[date, ...],
    anchor_index: int,
    return_count: int,
    minimum_coverage: float,
) -> list[float]:
    start = anchor_index - return_count
    if start < 0:
        return []
    dates = calendar_sessions[start : anchor_index + 1]
    returns: list[float] = []
    for previous_date, current_date in zip(dates, dates[1:], strict=False):
        previous = observations.get(previous_date)
        current = observations.get(current_date)
        if previous is not None and current is not None:
            returns.append(current[0] / previous[0] - 1.0)
    required = math.ceil(return_count * minimum_coverage)
    return returns if len(returns) >= required else []


def _close_vs_average(
    observations: dict[date, tuple[float, float | None]],
    anchor_close: float | None,
    calendar_sessions: tuple[date, ...],
    anchor_index: int,
    window: int,
    minimum_coverage: float,
) -> float | None:
    values = _window_closes(
        observations,
        calendar_sessions,
        anchor_index,
        window,
        minimum_coverage,
    )
    if anchor_close is None or not values:
        return None
    average = float(np.mean(values))
    return anchor_close / average - 1.0 if average > 0 else None


def _average_dollar_volume(
    observations: dict[date, tuple[float, float | None]],
    calendar_sessions: tuple[date, ...],
    anchor_index: int,
    window: int,
    minimum_coverage: float,
) -> float | None:
    start = anchor_index - window + 1
    if start < 0:
        return None
    values: list[float] = []
    for session in calendar_sessions[start : anchor_index + 1]:
        observation = observations.get(session)
        if observation is None:
            continue
        close, volume = observation
        if volume is not None:
            values.append(close * volume)
    if len(values) < math.ceil(window * minimum_coverage):
        return None
    return float(np.mean(values))


def _annualized_volatility(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    return float(np.std(np.asarray(returns), ddof=1) * np.sqrt(TRADING_SESSIONS_PER_YEAR))


def _downside_volatility(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    downside = np.minimum(np.asarray(returns), 0.0)
    return float(np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_SESSIONS_PER_YEAR))


def _close(
    observations: dict[date, tuple[float, float | None]],
    session: date,
) -> float | None:
    value = observations.get(session)
    return value[0] if value is not None else None


def _bucket(value: float | None, boundaries: tuple[float, ...]) -> int | None:
    if value is None or not math.isfinite(value):
        return None
    return sum(value >= boundary for boundary in boundaries)


def _weighted_quantile(
    values: np.ndarray[Any, np.dtype[np.float64]],
    weights: np.ndarray[Any, np.dtype[np.float64]],
    quantile: float,
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = quantile * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _blend(value: float, baseline: float, weight: float) -> float:
    return weight * value + (1.0 - weight) * baseline


def _clamp_probability(value: float) -> float:
    return min(1.0, max(0.0, value))


def _validate_price_basis(asset: DataAsset) -> None:
    return_definition = asset.metadata.get("return_definition")
    dividends_included = asset.metadata.get("dividends_included")
    if return_definition != "split_adjusted_price_return" or dividends_included is not False:
        raise ValueError(
            f"Price asset {asset.pk} lacks the required split-adjusted, "
            "dividend-excluded return basis"
        )


def _parquet_bytes(frame: pl.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def _asset_identity(asset: DataAsset) -> dict[str, object]:
    return {
        "id": str(asset.pk),
        "provider": asset.provider,
        "kind": asset.kind,
        "subject": asset.subject,
        "relative_path": asset.relative_path,
        "sha256": asset.sha256,
        "retrieved_at": asset.retrieved_at.isoformat(),
        "available_at": asset.available_at.isoformat(),
    }


def _dedupe_assets(assets: list[DataAsset]) -> list[DataAsset]:
    unique: dict[str, DataAsset] = {}
    for asset in assets:
        unique[str(asset.pk)] = asset
    return [unique[key] for key in sorted(unique)]


def _hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

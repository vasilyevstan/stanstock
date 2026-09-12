from __future__ import annotations

import hashlib
import io
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from importlib.metadata import version as package_version
from typing import Any, cast
from uuid import UUID

import numpy as np
import polars as pl
from django.db import transaction
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.asof import AsOfData
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import DataAsset, Listing
from stanstock.research.forecast_config import (
    MATCH_DIMENSIONS,
    MEDIUM_FORECAST_HORIZONS,
    MEDIUM_V2_VERSION,
    ForecastHorizonConfig,
    MediumForecastConfig,
    MediumForecastV2Config,
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


def calendar_sessions_through(
    *, calendar_name: str, fixed_epoch: date, target_date: date
) -> tuple[date, ...]:
    """The exact ordered trading-session closure this configuration's panel
    is built over, from `fixed_epoch` through `target_date` inclusive.

    A single pure leaf shared by the panel producer and its research-domain
    verifier, so the verifier can independently reproduce the panel's own
    `calendar_hash` rather than trusting whatever hash the panel's metadata
    happens to declare.
    """
    calendar = get_calendar(calendar_name)
    target_session = calendar.date_to_session(target_date, direction="none")
    epoch_session = calendar.date_to_session(fixed_epoch, direction="none")
    if target_session < epoch_session:
        raise ValueError("Forecast target date precedes the configured fixed epoch")
    sessions = calendar.sessions_in_range(epoch_session, target_session)
    return tuple(session.date() for session in sessions)


@dataclass(frozen=True, slots=True)
class MediumPanel:
    frame: pl.DataFrame
    asset: DataAsset
    source_assets: tuple[DataAsset, ...]
    file_created_by_invocation: bool = True


@dataclass(frozen=True, slots=True)
class MediumPanelPriceInput:
    """One exact normalized price vintage and its cutoff-clipped rows."""

    listing: Listing
    asset: DataAsset
    frame: pl.DataFrame


@dataclass(frozen=True, slots=True)
class MediumForecast:
    scenario: Scenario
    calculation: dict[str, Any]

    def scenario_payload(self) -> dict[str, Any]:
        if self.calculation.get("schema_version") == 2:
            return {
                **self.scenario.as_dict(),
                "method_version": self.calculation["method_version"],
                "calculation_schema_version": 2,
                "current_state": self.calculation["current_state"],
                "support": self.calculation["support"],
                "probability_evidence": self.calculation["probability_evidence"],
                "predictive_distribution": self.calculation["predictive_distribution"],
                "evidence": self.calculation["evidence"],
                "formula_inputs": self.calculation["formula_inputs"],
                "return_basis": self.calculation["return_basis"],
                "dividends_included": self.calculation["dividends_included"],
                "training_evidence": self.calculation["training_evidence"],
            }
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
    config: MediumForecastConfig | MediumForecastV2Config,
    config_hash: str,
    scoring_config_version: str,
    scoring_config_hash: str,
    universe_snapshot_id: UUID,
    universe_slug: str,
    universe_config_hash: str,
    code_revision: str,
    store: AssetStore,
    data_cutoff: datetime | None = None,
) -> MediumPanel:
    if isinstance(config, MediumForecastV2Config):
        if data_cutoff is None:
            raise ValueError("us-price-medium-v2 requires the owning AnalysisRun.data_cutoff")
        selected_benchmark = _select_v2_price_asset(
            asof=asof,
            provider=provider,
            subject=benchmark_subject,
            data_cutoff=data_cutoff,
        )
        selected_listings: list[tuple[Listing, DataAsset]] = []
        for listing in sorted(listings, key=lambda item: str(item.pk)):
            subject = listing.provider_symbol or listing.ticker
            selected_listings.append(
                (
                    listing,
                    _select_v2_price_asset(
                        asof=asof,
                        provider=provider,
                        subject=subject,
                        data_cutoff=data_cutoff,
                    ),
                )
            )
        benchmark_read = asof.price_frame_for_asset_with_diagnostics(
            asset=selected_benchmark,
            through_date=target_date,
        )
        listing_inputs = []
        for listing, selected_asset in selected_listings:
            read = asof.price_frame_for_asset_with_diagnostics(
                asset=selected_asset,
                through_date=target_date,
            )
            listing_inputs.append(
                MediumPanelPriceInput(
                    listing=listing,
                    asset=read.asset,
                    frame=read.frame,
                )
            )
    else:
        benchmark_read = asof.price_frame_with_diagnostics(
            provider=provider,
            subject=benchmark_subject,
            through_date=target_date,
        )
        listing_inputs = []
        for listing in sorted(listings, key=lambda item: str(item.pk)):
            subject = listing.provider_symbol or listing.ticker
            read = asof.price_frame_with_diagnostics(
                provider=provider,
                subject=subject,
                through_date=target_date,
            )
            listing_inputs.append(
                MediumPanelPriceInput(
                    listing=listing,
                    asset=read.asset,
                    frame=read.frame,
                )
            )
    frame = reconstruct_medium_forecast_panel(
        benchmark_asset=benchmark_read.asset,
        benchmark_frame=benchmark_read.frame,
        listing_inputs=listing_inputs,
        target_date=target_date,
        config=config,
    )
    if isinstance(config, MediumForecastV2Config):
        _validate_v2_support_returns(frame.to_dicts())
    calendar_sessions = calendar_sessions_through(
        calendar_name=config.calendar, fixed_epoch=config.fixed_epoch, target_date=target_date
    )
    calendar_hash = hash_json([session.isoformat() for session in calendar_sessions])
    payload = serialize_medium_forecast_panel(frame)
    content_hash = hashlib.sha256(payload).hexdigest()
    relative_path = (
        f"derived/forecast/medium/{target_date.isoformat()}/"
        f"{run_id.hex}-{content_hash[:12]}.parquet"
    )
    # Decide file ownership *before* writing (mirrors
    # `research.service._write_analysis_output_manifest`'s exact idiom):
    # a fresh `run_id`/content-hash path should never already exist, so
    # this only guards against a stale leftover from an earlier failed
    # attempt at the very same path. Ownership must never be inferred
    # from a post-failure DB query -- if the nested `transaction.atomic()`
    # below fails on its own savepoint *exit* (after already truly
    # committing the row into the outer transaction), the connection is
    # left in a doomed "needs rollback" state and a further ORM query in
    # the `except` block would itself raise, masking the original error
    # and making row-based ownership detection unreliable.
    try:
        resolved = store.resolve(relative_path)
        file_already_existed = resolved.exists()
    except (OSError, ValueError):
        raise RefreshVerificationError(
            "medium_forecast_panel_path_unavailable",
            "The medium-forecast panel path could not be checked",
        ) from None
    try:
        stored = store.write_bytes(relative_path, payload)
    except (OSError, ValueError):
        raise RefreshVerificationError(
            "medium_forecast_panel_write_failed",
            "The medium-forecast panel could not be written",
        ) from None
    deduped_sources = dedupe_assets(
        [benchmark_read.asset, *(item.asset for item in listing_inputs)]
    )
    source_manifest = [asset_identity(asset) for asset in deduped_sources]
    source_manifest_hash = hash_json(source_manifest)
    evidence_bundle_hash = hash_json(
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
    anchor_dates = [
        anchor_date
        for anchor_date in frame["anchor_date"].to_list()
        if isinstance(anchor_date, date)
    ]
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
        if not file_already_existed:
            # A cleanup fault here (e.g. an unexpected permission error on
            # unlink) must never replace the original exception being
            # handled: swallow only this narrow best-effort cleanup step,
            # never the failure that actually caused it.
            try:
                store.resolve(relative_path).unlink(missing_ok=True)
            except (OSError, ValueError):
                pass
        raise
    return MediumPanel(
        frame=frame,
        asset=asset,
        source_assets=tuple(deduped_sources),
        file_created_by_invocation=not file_already_existed,
    )


def _select_v2_price_asset(
    *,
    asof: AsOfData,
    provider: str,
    subject: str,
    data_cutoff: datetime,
) -> DataAsset:
    asset = asof.latest_asset(
        provider=provider,
        kind="price_history",
        subject=subject,
    )
    if asset.provider != provider or asset.kind != "price_history" or asset.subject != subject:
        raise ValueError("us-price-medium-v2 selected price asset identity mismatch")
    if asset.available_at > data_cutoff:
        raise ValueError(
            "us-price-medium-v2 refuses price assets available after AnalysisRun.data_cutoff"
        )
    return asset


def reconstruct_medium_forecast_panel(
    *,
    benchmark_asset: DataAsset,
    benchmark_frame: pl.DataFrame,
    listing_inputs: list[MediumPanelPriceInput],
    target_date: date,
    config: MediumForecastConfig | MediumForecastV2Config,
) -> pl.DataFrame:
    """Purely reconstruct the versioned panel from exact normalized inputs.

    The producer and verifier both call this function. It performs no ORM,
    provider, filesystem, or asset writes, so verification can replay every
    cohort, feature, label, eligibility relation, row, and float without
    creating a second methodology implementation.
    """
    _validate_price_basis(benchmark_asset)
    benchmark_prices = _price_observations(benchmark_frame)
    calendar_sessions = calendar_sessions_through(
        calendar_name=config.calendar,
        fixed_epoch=config.fixed_epoch,
        target_date=target_date,
    )
    session_index = {session: index for index, session in enumerate(calendar_sessions)}
    normalized_inputs: list[tuple[Listing, DataAsset, dict[date, tuple[float, float | None]]]] = []
    for item in sorted(listing_inputs, key=lambda value: str(value.listing.pk)):
        _validate_price_basis(item.asset)
        normalized_inputs.append((item.listing, item.asset, _price_observations(item.frame)))

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
            for listing, asset, prices in normalized_inputs:
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
        for listing, asset, prices in normalized_inputs:
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

    return pl.DataFrame(rows, schema=PANEL_SCHEMA, orient="row").sort(
        "horizon",
        "anchor_date",
        "listing_id",
    )


def build_medium_forecasts(
    panel: pl.DataFrame,
    config: MediumForecastConfig | MediumForecastV2Config,
) -> dict[str, dict[str, MediumForecast]]:
    records = panel.to_dicts()
    if isinstance(config, MediumForecastV2Config):
        _validate_v2_support_returns(records)
        return _build_medium_forecasts_v2(records, config)
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


@dataclass(frozen=True, slots=True)
class _V2Distribution:
    masses: tuple[tuple[float, float], ...]
    raw_matches: int
    effective_cohorts: int
    distinct_listings: int
    calendar_start: str | None
    calendar_end: str | None
    regimes: tuple[str, ...]
    dispersion: float | None

    @property
    def p20(self) -> float | None:
        return _v2_quantile(self.masses, 0.2)

    @property
    def p50(self) -> float | None:
        return _v2_quantile(self.masses, 0.5)

    @property
    def p80(self) -> float | None:
        return _v2_quantile(self.masses, 0.8)

    @property
    def probability_positive(self) -> float | None:
        if not self.masses:
            return None
        cdf_at_zero = math.fsum(mass for value, mass in self.masses if value <= 0.0)
        probability = 1.0 - cdf_at_zero
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("us-price-medium-v2 produced an invalid probability")
        return probability


@dataclass(frozen=True, slots=True)
class _V2Estimate:
    fallback_level: str
    shrinkage_weight: float
    matched: _V2Distribution
    unconditional: _V2Distribution
    mixture: _V2Distribution


def _validate_v2_support_returns(records: list[dict[str, Any]]) -> None:
    for row in records:
        if bool(row.get("is_forecast")) or row.get("forward_return") is None:
            continue
        raw_value = row["forward_return"]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError("us-price-medium-v2 requires finite support returns")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("us-price-medium-v2 requires finite support returns")
        if value < -1.0:
            raise ValueError("us-price-medium-v2 refuses support returns below -1.0")


def _build_medium_forecasts_v2(
    records: list[dict[str, Any]],
    config: MediumForecastV2Config,
) -> dict[str, dict[str, MediumForecast]]:
    evidence = {
        horizon: _v2_prequential_evidence(records, horizon=horizon, config=config)
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
            forecasts[listing_id][horizon] = _v2_forecast_for_state(
                records,
                current=current,
                horizon=horizon,
                config=config,
                evidence=evidence[horizon],
            )
    return forecasts


def _v2_forecast_for_state(
    records: list[dict[str, Any]],
    *,
    current: dict[str, Any] | None,
    horizon: str,
    config: MediumForecastV2Config,
    evidence: dict[str, Any],
) -> MediumForecast:
    horizon_config = config.horizons[horizon]
    empty_support = _v2_support_payload(
        _v2_distribution([]),
        fallback_level="unavailable",
        shrinkage_weight=0.0,
    )
    if current is None:
        return _v2_missing_forecast(
            horizon=horizon,
            config=config,
            reason="Current forecast state is missing from the immutable panel",
            evidence=evidence,
            current_state=_v2_current_state(None),
            support=empty_support,
        )
    current_state = _v2_current_state(current)
    if not bool(current["eligible"]):
        return _v2_missing_forecast(
            horizon=horizon,
            config=config,
            reason=str(current["insufficiency_reason"]),
            evidence=evidence,
            current_state=current_state,
            support=empty_support,
        )
    origin = current.get("anchor_date")
    if not isinstance(origin, date):
        return _v2_missing_forecast(
            horizon=horizon,
            config=config,
            reason="Current forecast state has no valid anchor date",
            evidence=evidence,
            current_state=current_state,
            support=empty_support,
        )
    training = _v2_training_rows(records, horizon=horizon, origin=origin)
    estimate = _v2_estimate_distribution(
        training,
        current=current,
        horizon_config=horizon_config,
        config=config,
    )
    if estimate is None:
        unconditional = _v2_distribution(training)
        support = _v2_support_payload(
            unconditional,
            fallback_level="unavailable",
            shrinkage_weight=0.0,
        )
        return _v2_missing_forecast(
            horizon=horizon,
            config=config,
            reason=_v2_support_failure_reason(horizon, unconditional, horizon_config),
            evidence=evidence,
            current_state=current_state,
            support=support,
        )

    raw_probability = estimate.mixture.probability_positive
    p20 = estimate.mixture.p20
    p50 = estimate.mixture.p50
    p80 = estimate.mixture.p80
    if None in (p20, p50, p80, raw_probability):
        raise ValueError("us-price-medium-v2 produced an incomplete predictive distribution")
    assert p20 is not None
    assert p50 is not None
    assert p80 is not None
    assert raw_probability is not None
    probability_reasons = _v2_probability_support_reasons(
        estimate.matched,
        horizon_config,
    )
    probability_skill = evidence["probability_skill"]
    probability_reasons.extend(
        _v2_probability_skill_reasons(
            probability_skill,
            required=config.walk_forward.minimum_test_cohorts,
        )
    )
    published_probability = raw_probability if not probability_reasons else None
    confidence = min(80.0, 20.0 + 60.0 * estimate.shrinkage_weight)
    scenario = Scenario(
        bear=p20,
        base=p50,
        bull=p80,
        probability_positive=published_probability,
        confidence=confidence,
        confidence_status=(
            "empirical_skill_supported"
            if published_probability is not None
            else "empirical_range_only"
        ),
        insufficiency_reason=(
            ""
            if published_probability is not None
            else "Probability withheld: " + "; ".join(probability_reasons)
        ),
        method=METHOD_NAME,
    )
    probability_evidence = _v2_probability_evidence(
        estimate,
        horizon_config=horizon_config,
        reasons=probability_reasons,
        published=published_probability is not None,
    )
    calculation = _v2_calculation(
        horizon=horizon,
        config=config,
        current_state=current_state,
        support=_v2_support_payload(
            estimate.matched,
            fallback_level=estimate.fallback_level,
            shrinkage_weight=estimate.shrinkage_weight,
        ),
        probability_evidence=probability_evidence,
        predictive_distribution=_v2_predictive_distribution(
            estimate,
            published_probability=published_probability,
        ),
        evidence=evidence,
        scenario={
            "bear": p20,
            "base": p50,
            "bull": p80,
            "probability_positive": published_probability,
        },
    )
    return MediumForecast(scenario=scenario, calculation=calculation)


def _v2_missing_forecast(
    *,
    horizon: str,
    config: MediumForecastV2Config,
    reason: str,
    evidence: dict[str, Any],
    current_state: dict[str, Any],
    support: dict[str, Any],
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
    calculation = _v2_calculation(
        horizon=horizon,
        config=config,
        current_state=current_state,
        support=support,
        probability_evidence=_v2_empty_probability_evidence(config.horizons[horizon]),
        predictive_distribution=_v2_empty_predictive_distribution(),
        evidence=evidence,
        scenario={
            "bear": None,
            "base": None,
            "bull": None,
            "probability_positive": None,
        },
    )
    return MediumForecast(scenario=scenario, calculation=calculation)


def _v2_calculation(
    *,
    horizon: str,
    config: MediumForecastV2Config,
    current_state: dict[str, Any],
    support: dict[str, Any],
    probability_evidence: dict[str, Any],
    predictive_distribution: dict[str, Any],
    evidence: dict[str, Any],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "method": METHOD_NAME,
        "method_version": MEDIUM_V2_VERSION,
        "forecast_horizon": horizon,
        "horizon_sessions": config.horizons[horizon].sessions,
        "current_state": current_state,
        "support": support,
        "probability_evidence": probability_evidence,
        "predictive_distribution": predictive_distribution,
        "evidence": evidence,
        "formula_inputs": {"scenario": scenario},
        "return_basis": config.return_basis,
        "dividends_included": config.dividends_included,
        "training_evidence": {
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
        },
    }


def _v2_distribution(rows: list[dict[str, Any]]) -> _V2Distribution:
    usable = sorted(
        (row for row in rows if bool(row["eligible"]) and row["forward_return"] is not None),
        key=_v2_row_identity_value,
    )
    if not usable:
        return _V2Distribution((), 0, 0, 0, None, None, (), None)
    cohort_counts = Counter(str(row["cohort_id"]) for row in usable)
    cohort_count = len(cohort_counts)
    contributions_by_value: dict[float, list[float]] = {}
    for row in usable:
        value = float(row["forward_return"])
        mass = 1.0 / (cohort_count * cohort_counts[str(row["cohort_id"])])
        contributions_by_value.setdefault(value, []).append(mass)
    mass_by_value = tuple(
        (value, math.fsum(contributions_by_value[value]))
        for value in sorted(contributions_by_value)
    )
    total_mass = math.fsum(mass for _value, mass in mass_by_value)
    if not math.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("us-price-medium-v2 produced invalid component mass")
    masses = tuple((value, mass / total_mass) for value, mass in mass_by_value)
    weighted_mean = math.fsum(value * mass for value, mass in masses)
    dispersion = math.sqrt(
        math.fsum(((value - weighted_mean) ** 2) * mass for value, mass in masses)
    )
    dates = sorted({row["anchor_date"] for row in usable if isinstance(row["anchor_date"], date)})
    regimes = tuple(
        sorted(
            {f"{row['market_trend_bucket']}:{row['market_volatility_bucket']}" for row in usable}
        )
    )
    return _V2Distribution(
        masses=masses,
        raw_matches=len(usable),
        effective_cohorts=cohort_count,
        distinct_listings=len({str(row["listing_id"]) for row in usable}),
        calendar_start=dates[0].isoformat() if dates else None,
        calendar_end=dates[-1].isoformat() if dates else None,
        regimes=regimes,
        dispersion=dispersion,
    )


def _v2_row_identity_value(row: Mapping[str, Any]) -> tuple[str, str, float, str, str]:
    anchor = row.get("anchor_date")
    label_end = row.get("label_end_date")
    return (
        str(row.get("cohort_id")),
        str(row.get("listing_id")),
        float(row["forward_return"]),
        anchor.isoformat() if isinstance(anchor, date) else str(anchor),
        label_end.isoformat() if isinstance(label_end, date) else str(label_end),
    )


def _v2_estimate_distribution(
    training: list[dict[str, Any]],
    *,
    current: dict[str, Any],
    horizon_config: ForecastHorizonConfig,
    config: MediumForecastV2Config,
) -> _V2Estimate | None:
    unconditional = _v2_distribution(training)
    if not _support_sufficient_v2(unconditional, horizon_config):
        return None
    for fallback in config.fallback_order:
        matched_rows = [
            row
            for row in training
            if all(row[dimension] == current[dimension] for dimension in fallback.dimensions)
        ]
        matched = _v2_distribution(matched_rows)
        if not _support_sufficient_v2(matched, horizon_config):
            continue
        weight = (
            0.0
            if not fallback.dimensions
            else matched.effective_cohorts
            / (matched.effective_cohorts + horizon_config.shrinkage_prior_cohorts)
        )
        mixture = _v2_mix_distributions(
            matched,
            unconditional,
            matched_mass=weight,
        )
        return _V2Estimate(
            fallback_level=fallback.name,
            shrinkage_weight=weight,
            matched=matched,
            unconditional=unconditional,
            mixture=mixture,
        )
    return None


def _v2_mix_distributions(
    matched: _V2Distribution,
    unconditional: _V2Distribution,
    *,
    matched_mass: float,
) -> _V2Distribution:
    if not 0.0 <= matched_mass <= 1.0:
        raise ValueError("us-price-medium-v2 produced invalid mixture mass")
    contributions: dict[float, list[float]] = {}
    for value, mass in matched.masses:
        contributions.setdefault(value, []).append(matched_mass * mass)
    for value, mass in unconditional.masses:
        contributions.setdefault(value, []).append((1.0 - matched_mass) * mass)
    masses = tuple(
        (value, math.fsum(contributions[value]))
        for value in sorted(contributions)
        if math.fsum(contributions[value]) > 0.0
    )
    total = math.fsum(mass for _value, mass in masses)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("us-price-medium-v2 produced invalid mixture mass")
    normalized = tuple((value, mass / total) for value, mass in masses)
    mean = math.fsum(value * mass for value, mass in normalized)
    dispersion = math.sqrt(math.fsum(((value - mean) ** 2) * mass for value, mass in normalized))
    return _V2Distribution(
        masses=normalized,
        raw_matches=matched.raw_matches,
        effective_cohorts=matched.effective_cohorts,
        distinct_listings=matched.distinct_listings,
        calendar_start=matched.calendar_start,
        calendar_end=matched.calendar_end,
        regimes=matched.regimes,
        dispersion=dispersion,
    )


def _v2_quantile(
    masses: tuple[tuple[float, float], ...],
    quantile: float,
) -> float | None:
    seen: list[float] = []
    for value, mass in masses:
        seen.append(mass)
        if math.fsum(seen) >= quantile:
            return value
    return masses[-1][0] if masses else None


def _support_sufficient_v2(
    distribution: _V2Distribution,
    config: ForecastHorizonConfig,
) -> bool:
    return (
        distribution.raw_matches >= config.minimum_raw_matches
        and distribution.effective_cohorts >= config.minimum_effective_cohorts
        and distribution.distinct_listings >= config.minimum_distinct_listings
    )


def _v2_training_rows(
    records: list[dict[str, Any]],
    *,
    horizon: str,
    origin: date,
) -> list[dict[str, Any]]:
    training: list[dict[str, Any]] = []
    for row in records:
        anchor = row.get("anchor_date")
        label_end = row.get("label_end_date")
        if (
            str(row.get("horizon")) != horizon
            or bool(row.get("is_forecast"))
            or not bool(row.get("eligible"))
            or row.get("forward_return") is None
            or not isinstance(anchor, date)
            or not isinstance(label_end, date)
            or anchor >= origin
            or label_end > origin
        ):
            continue
        training.append(row)
    return training


def _v2_prequential_evidence(
    records: list[dict[str, Any]],
    *,
    horizon: str,
    config: MediumForecastV2Config,
) -> dict[str, Any]:
    historical = [
        row
        for row in records
        if str(row.get("horizon")) == horizon
        and not bool(row.get("is_forecast"))
        and bool(row.get("eligible"))
        and row.get("forward_return") is not None
        and isinstance(row.get("anchor_date"), date)
        and isinstance(row.get("label_end_date"), date)
    ]
    origins = sorted({row["anchor_date"] for row in historical})
    range_by_origin: list[dict[str, list[float]]] = []
    probability_by_origin: list[dict[str, list[float]]] = []
    horizon_config = config.horizons[horizon]
    for origin in origins:
        assert isinstance(origin, date)
        prior = _v2_training_rows(records, horizon=horizon, origin=origin)
        if (
            len({str(row["cohort_id"]) for row in prior})
            < config.walk_forward.minimum_training_cohorts
        ):
            continue
        test_rows = sorted(
            (row for row in historical if row["anchor_date"] == origin),
            key=_v2_row_identity_value,
        )
        range_metrics: dict[str, list[float]] = {
            "model_error": [],
            "unconditional_error": [],
            "relative_error": [],
            "covered": [],
            "below": [],
            "above": [],
            "width": [],
            "model_interval_score": [],
            "reference_interval_score": [],
        }
        probability_metrics: dict[str, list[float]] = {
            "model_brier": [],
            "reference_brier": [],
        }
        for row in test_rows:
            estimate = _v2_estimate_distribution(
                prior,
                current=row,
                horizon_config=horizon_config,
                config=config,
            )
            if estimate is None:
                continue
            p20 = estimate.mixture.p20
            p50 = estimate.mixture.p50
            p80 = estimate.mixture.p80
            ref20 = estimate.unconditional.p20
            ref50 = estimate.unconditional.p50
            ref80 = estimate.unconditional.p80
            if None in (p20, p50, p80, ref20, ref50, ref80):
                continue
            assert p20 is not None
            assert p50 is not None
            assert p80 is not None
            assert ref20 is not None
            assert ref50 is not None
            assert ref80 is not None
            actual = float(row["forward_return"])
            range_metrics["model_error"].append(abs(actual - p50))
            range_metrics["unconditional_error"].append(abs(actual - ref50))
            range_metrics["relative_error"].append(
                abs(actual - _v2_spy_relative_baseline(prior, row))
            )
            range_metrics["covered"].append(float(p20 <= actual <= p80))
            range_metrics["below"].append(float(actual < p20))
            range_metrics["above"].append(float(actual > p80))
            range_metrics["width"].append(p80 - p20)
            range_metrics["model_interval_score"].append(_v2_interval_score(p20, p80, actual))
            range_metrics["reference_interval_score"].append(
                _v2_interval_score(ref20, ref80, actual)
            )
            if not _v2_probability_support_reasons(estimate.matched, horizon_config):
                model_probability = estimate.mixture.probability_positive
                reference_probability = estimate.unconditional.probability_positive
                if model_probability is None or reference_probability is None:
                    continue
                observed = 1.0 if actual > 0.0 else 0.0
                probability_metrics["model_brier"].append((model_probability - observed) ** 2)
                probability_metrics["reference_brier"].append(
                    (reference_probability - observed) ** 2
                )
        if range_metrics["model_error"]:
            range_by_origin.append(range_metrics)
        if probability_metrics["model_brier"]:
            probability_by_origin.append(probability_metrics)
    return {
        "base_accuracy": _v2_base_evidence(
            range_by_origin,
            config=config,
        ),
        "probability_skill": _v2_probability_skill_evidence(
            probability_by_origin,
            config=config,
        ),
        "interval": _v2_interval_evidence(
            range_by_origin,
            config=config,
        ),
    }


def _v2_spy_relative_baseline(
    training: list[dict[str, Any]],
    current: dict[str, Any],
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
    benchmark = _v2_distribution(
        [{**row, "forward_return": row["benchmark_forward_return"]} for row in benchmark_matches]
    )
    relative = _v2_distribution(
        [{**row, "forward_return": row["relative_forward_return"]} for row in relative_matches]
    )
    if benchmark.p50 is None or relative.p50 is None:
        raise ValueError("us-price-medium-v2 SPY-relative baseline is unavailable")
    return math.fsum((benchmark.p50, relative.p50))


def _v2_date_equal_mean(
    origins: list[dict[str, list[float]]],
    key: str,
) -> float | None:
    values = sorted(
        math.fsum(sorted(origin[key])) / len(origin[key]) for origin in origins if origin[key]
    )
    return math.fsum(values) / len(values) if values else None


def _v2_base_evidence(
    origins: list[dict[str, list[float]]],
    *,
    config: MediumForecastV2Config,
) -> dict[str, Any]:
    origin_count = len(origins)
    predictions = sum(len(origin["model_error"]) for origin in origins)
    model = _v2_date_equal_mean(origins, "model_error")
    unconditional = _v2_date_equal_mean(origins, "unconditional_error")
    relative = _v2_date_equal_mean(origins, "relative_error")
    if origin_count == 0:
        status = "not_evaluable"
    elif origin_count < config.walk_forward.minimum_test_cohorts:
        status = "insufficient_support"
    else:
        assert model is not None and unconditional is not None and relative is not None
        ratio = config.walk_forward.maximum_baseline_mae_ratio
        status = (
            "passed" if model <= unconditional * ratio and model <= relative * ratio else "failed"
        )
    return {
        "status": status,
        "test_origins": origin_count,
        "test_predictions": predictions,
        "weighting": "date_equal_listing_equal_within_origin",
        "mean_absolute_error": model,
        "unconditional_mean_absolute_error": unconditional,
        "spy_relative_mean_absolute_error": relative,
        "spy_relative_baseline_method": (
            "market_regime_benchmark_median_plus_relative_momentum_excess_median"
        ),
        "maximum_baseline_mae_ratio": config.walk_forward.maximum_baseline_mae_ratio,
    }


def _v2_probability_skill_evidence(
    origins: list[dict[str, list[float]]],
    *,
    config: MediumForecastV2Config,
) -> dict[str, Any]:
    origin_count = len(origins)
    predictions = sum(len(origin["model_brier"]) for origin in origins)
    model = _v2_date_equal_mean(origins, "model_brier")
    reference = _v2_date_equal_mean(origins, "reference_brier")
    bss = None if reference in (None, 0.0) else 1.0 - cast(float, model) / reference
    if origin_count == 0:
        status = "not_evaluable"
    elif origin_count < config.walk_forward.minimum_test_cohorts:
        status = "insufficient_support"
    elif reference == 0.0:
        status = "reference_zero"
    else:
        assert model is not None and reference is not None
        if model < reference:
            status = "positive_skill"
        elif model == reference:
            status = "zero_skill"
        else:
            status = "negative_skill"
    return {
        "status": status,
        "test_origins": origin_count,
        "test_predictions": predictions,
        "weighting": "date_equal_listing_equal_within_origin",
        "event": "return_gt_0",
        "model_brier_score": model,
        "reference_brier_score": reference,
        "brier_skill_score": bss,
        "reference_method": "prequential_unconditional",
        "minimum_brier_skill_exclusive": (config.walk_forward.minimum_brier_skill_exclusive),
        "zero_reference_policy": "null_no_epsilon",
    }


def _v2_interval_evidence(
    origins: list[dict[str, list[float]]],
    *,
    config: MediumForecastV2Config,
) -> dict[str, Any]:
    origin_count = len(origins)
    if origin_count == 0:
        status = "not_evaluable"
    elif origin_count < config.walk_forward.minimum_test_cohorts:
        status = "preliminary"
    else:
        status = "descriptive"
    alpha = config.walk_forward.interval_alpha
    return {
        "status": status,
        "test_origins": origin_count,
        "test_predictions": sum(len(origin["covered"]) for origin in origins),
        "weighting": "date_equal_listing_equal_within_origin",
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "endpoint_policy": "inclusive",
        "empirical_coverage": _v2_date_equal_mean(origins, "covered"),
        "below_rate": _v2_date_equal_mean(origins, "below"),
        "above_rate": _v2_date_equal_mean(origins, "above"),
        "mean_width": _v2_date_equal_mean(origins, "width"),
        "model_mean_interval_score": _v2_date_equal_mean(origins, "model_interval_score"),
        "reference_mean_interval_score": _v2_date_equal_mean(origins, "reference_interval_score"),
        "reference_method": "prequential_unconditional",
    }


def _v2_interval_score(lower: float, upper: float, actual: float) -> float:
    terms = [upper - lower]
    if actual < lower:
        terms.append(5.0 * (lower - actual))
    if actual > upper:
        terms.append(5.0 * (actual - upper))
    return math.fsum(terms)


def _v2_probability_support_reasons(
    distribution: _V2Distribution,
    config: ForecastHorizonConfig,
) -> list[str]:
    span = _v2_calendar_span_days(distribution)
    reasons: list[str] = []
    if distribution.effective_cohorts < config.probability_minimum_effective_cohorts:
        reasons.append(
            "effective cohorts "
            f"{distribution.effective_cohorts}/"
            f"{config.probability_minimum_effective_cohorts}"
        )
    if distribution.distinct_listings < config.probability_minimum_distinct_listings:
        reasons.append(
            "distinct listings "
            f"{distribution.distinct_listings}/"
            f"{config.probability_minimum_distinct_listings}"
        )
    if span < config.probability_minimum_calendar_span_days:
        reasons.append(f"calendar span {span}/{config.probability_minimum_calendar_span_days} days")
    if len(distribution.regimes) < config.probability_minimum_distinct_market_regimes:
        reasons.append(
            "matched market regimes "
            f"{len(distribution.regimes)}/"
            f"{config.probability_minimum_distinct_market_regimes}"
        )
    return reasons


def _v2_probability_skill_reasons(
    evidence: dict[str, Any],
    *,
    required: int,
) -> list[str]:
    status = evidence["status"]
    if status == "positive_skill" and cast(float, evidence["brier_skill_score"]) > 0.0:
        return []
    if status in {"not_evaluable", "insufficient_support"}:
        return [
            "prequential Brier skill insufficient "
            f"({evidence['test_origins']}/{required} test origins)"
        ]
    if status == "reference_zero":
        return ["prequential unconditional reference Brier score is zero"]
    if status == "zero_skill":
        return ["prequential Brier skill is zero"]
    return ["prequential Brier skill is negative"]


def _v2_probability_evidence(
    estimate: _V2Estimate,
    *,
    horizon_config: ForecastHorizonConfig,
    reasons: list[str],
    published: bool,
) -> dict[str, Any]:
    return {
        "status": "published" if published else "withheld",
        "reasons": reasons,
        "calendar_span_days": _v2_calendar_span_days(estimate.matched),
        "distinct_matched_market_regimes": len(estimate.matched.regimes),
        "distinct_panel_market_regimes": len(estimate.unconditional.regimes),
        "minimum_effective_cohorts": (horizon_config.probability_minimum_effective_cohorts),
        "minimum_distinct_listings": (horizon_config.probability_minimum_distinct_listings),
        "minimum_calendar_span_days": (horizon_config.probability_minimum_calendar_span_days),
        "minimum_distinct_market_regimes": (
            horizon_config.probability_minimum_distinct_market_regimes
        ),
    }


def _v2_empty_probability_evidence(
    horizon_config: ForecastHorizonConfig,
) -> dict[str, Any]:
    return {
        "status": "not_evaluable",
        "reasons": [],
        "calendar_span_days": None,
        "distinct_matched_market_regimes": None,
        "distinct_panel_market_regimes": None,
        "minimum_effective_cohorts": (horizon_config.probability_minimum_effective_cohorts),
        "minimum_distinct_listings": (horizon_config.probability_minimum_distinct_listings),
        "minimum_calendar_span_days": (horizon_config.probability_minimum_calendar_span_days),
        "minimum_distinct_market_regimes": (
            horizon_config.probability_minimum_distinct_market_regimes
        ),
    }


def _v2_component_payload(
    distribution: _V2Distribution | None,
    *,
    component_mass: float | None,
) -> dict[str, Any]:
    if distribution is None or not distribution.masses:
        return {
            "component_mass": component_mass,
            "normalized_mass": None,
            "p20": None,
            "p50": None,
            "p80": None,
            "probability_positive_raw": None,
            "raw_observations": 0,
            "effective_cohorts": 0,
            "distinct_listings": 0,
            "calendar_start": None,
            "calendar_end": None,
            "market_regimes": [],
            "dispersion": None,
        }
    return {
        "component_mass": component_mass,
        "normalized_mass": 1.0,
        "p20": distribution.p20,
        "p50": distribution.p50,
        "p80": distribution.p80,
        "probability_positive_raw": distribution.probability_positive,
        "raw_observations": distribution.raw_matches,
        "effective_cohorts": distribution.effective_cohorts,
        "distinct_listings": distribution.distinct_listings,
        "calendar_start": distribution.calendar_start,
        "calendar_end": distribution.calendar_end,
        "market_regimes": list(distribution.regimes),
        "dispersion": distribution.dispersion,
    }


def _v2_predictive_distribution(
    estimate: _V2Estimate,
    *,
    published_probability: float | None,
) -> dict[str, Any]:
    weight = estimate.shrinkage_weight
    return {
        "kind": "cohort_equal_empirical_cdf_mixture",
        "cdf_event": "return_lte_x",
        "positive_event": "return_gt_0",
        "quantile_convention": "left_inverse_first_cdf_ge_q",
        "overlap_policy": "matched_rows_receive_mass_in_both_normalized_components",
        "matched": _v2_component_payload(
            estimate.matched,
            component_mass=weight,
        ),
        "unconditional": _v2_component_payload(
            estimate.unconditional,
            component_mass=1.0 - weight,
        ),
        "p20": estimate.mixture.p20,
        "p50": estimate.mixture.p50,
        "p80": estimate.mixture.p80,
        "probability_positive_raw": estimate.mixture.probability_positive,
        "probability_positive_published": published_probability,
    }


def _v2_empty_predictive_distribution() -> dict[str, Any]:
    return {
        "kind": "cohort_equal_empirical_cdf_mixture",
        "cdf_event": "return_lte_x",
        "positive_event": "return_gt_0",
        "quantile_convention": "left_inverse_first_cdf_ge_q",
        "overlap_policy": "matched_rows_receive_mass_in_both_normalized_components",
        "matched": _v2_component_payload(None, component_mass=None),
        "unconditional": _v2_component_payload(None, component_mass=None),
        "p20": None,
        "p50": None,
        "p80": None,
        "probability_positive_raw": None,
        "probability_positive_published": None,
    }


def _v2_support_payload(
    distribution: _V2Distribution,
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


def _v2_current_state(row: dict[str, Any] | None) -> dict[str, Any]:
    keys = (
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
    )
    anchor = None if row is None else row.get("anchor_date")
    return {
        "anchor_date": anchor.isoformat() if isinstance(anchor, date) else None,
        **{key: None if row is None else row.get(key) for key in keys},
    }


def _v2_calendar_span_days(distribution: _V2Distribution) -> int:
    if distribution.calendar_start is None or distribution.calendar_end is None:
        return 0
    return (
        date.fromisoformat(distribution.calendar_end)
        - date.fromisoformat(distribution.calendar_start)
    ).days


def _v2_support_failure_reason(
    horizon: str,
    support: _V2Distribution,
    config: ForecastHorizonConfig,
) -> str:
    return (
        f"Insufficient non-overlapping {horizon} evidence: "
        f"{support.raw_matches}/{config.minimum_raw_matches} observations, "
        f"{support.effective_cohorts}/{config.minimum_effective_cohorts} cohorts, "
        f"{support.distinct_listings}/{config.minimum_distinct_listings} listings"
    )


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
    config: MediumForecastConfig | MediumForecastV2Config,
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
    config: MediumForecastConfig | MediumForecastV2Config,
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


def serialize_medium_forecast_panel(frame: pl.DataFrame) -> bytes:
    """Serialize a canonical panel with the producer's frozen byte contract."""
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def asset_identity(asset: DataAsset) -> dict[str, object]:
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


def dedupe_assets(assets: list[DataAsset]) -> list[DataAsset]:
    unique: dict[str, DataAsset] = {}
    for asset in assets:
        unique[str(asset.pk)] = asset
    return [unique[key] for key in sorted(unique)]


def hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

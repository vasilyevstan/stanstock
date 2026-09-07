from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
from django.db import transaction
from django.utils import timezone

from stanstock.data.asof import AsOfData
from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset, Listing, UniverseMembership, UniverseSnapshot
from stanstock.research.config import ScoringConfig, code_revision, config_hash, load_scoring_config
from stanstock.research.eligibility import require_stock_research_listing
from stanstock.research.explanations import generate_reasons, generate_risks
from stanstock.research.forecast_config import (
    MediumForecastConfig,
    load_medium_forecast_config,
    medium_forecast_config_hash,
)
from stanstock.research.forecasting import (
    build_forecast_scenario_document,
    infer_price_source,
)
from stanstock.research.fundamentals import calculate_fundamentals, inputs_from_facts
from stanstock.research.indicators import calculate_indicators
from stanstock.research.medium_forecasts import (
    MediumForecast,
    MediumPanel,
    build_medium_forecast_panel,
    build_medium_forecasts,
)
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.provenance import source_data_mode
from stanstock.research.scenarios import build_scenarios
from stanstock.research.scoring import (
    aggregate_score,
    assess_risk,
    decide_recommendation,
    score_components,
)
from stanstock.research.timing import is_observed_issuance_on_time
from stanstock.research.types import AggregateScore, IndicatorResult, ResearchValues, Scenario


@dataclass(frozen=True, slots=True)
class AnalysisComputation:
    indicators: IndicatorResult
    fundamentals: ResearchValues
    aggregate: AggregateScore
    scenarios: dict[str, Scenario]
    risk_score: float | None
    risk_class: str
    recommendation: str
    reasons: list[str]
    risks: list[str]
    data_quality: dict[str, Any]
    source_assets: list[dict[str, Any]]
    current_price: float
    daily_change: float | None


@dataclass(frozen=True, slots=True)
class PersistedAnalysis:
    run: AnalysisRun
    analysis: StockAnalysis
    predictions: tuple[Prediction, ...]
    computation: AnalysisComputation


@dataclass(frozen=True, slots=True)
class AdvisoryForecastContext:
    panel: MediumPanel
    config: MediumForecastConfig
    config_hash: str
    model_version: str
    forecasts: dict[str, dict[str, MediumForecast]]


def compute_listing_analysis(
    *,
    listing: Listing,
    price_frame: pl.DataFrame,
    config: ScoringConfig,
    decision_time: datetime,
    benchmark_frame: pl.DataFrame | None = None,
    facts: Any = (),
    source_assets: list[DataAsset] | None = None,
    price_asset: DataAsset | None = None,
    sample_support: dict[str, int] | None = None,
) -> AnalysisComputation:
    require_stock_research_listing(listing, operation="Stock analysis")
    indicators = calculate_indicators(
        price_frame, benchmark=benchmark_frame, windows=config.windows
    )
    price = indicators.values.get("last_close")
    if price is None:
        raise ValueError(f"No usable price history for {listing}")
    fundamental_inputs = (
        inputs_from_facts(())
        if config.analysis_mode == "price_only_baseline"
        else inputs_from_facts(facts)
    )
    fundamentals = calculate_fundamentals(fundamental_inputs, price=price)
    component_scores = score_components(indicators, fundamentals, config)
    aggregate = aggregate_score(
        component_scores,
        config,
        decision_date=decision_time.date(),
        indicators=indicators,
    )
    risk = assess_risk(indicators, fundamentals, config)
    scenarios = build_scenarios(
        price_frame,
        indicators,
        fundamentals,
        aggregate,
        config,
        sample_support=sample_support,
    )
    decision = decide_recommendation(
        aggregate.overall,
        risk,
        aggregate.confidence,
        config,
        scenarios=scenarios,
        indicators=indicators,
    )
    daily_change = indicators.values.get("return_1d")
    assets = _dedupe_assets(source_assets or [])
    asset_payload = [_asset_payload(asset) for asset in assets]
    data_quality = {
        "indicator_missing": indicators.missing,
        "fundamental_missing": fundamentals.missing,
        "scoring_missing": component_scores.missing,
        "coverage": component_scores.coverage,
        "missingness_penalty": aggregate.missingness_penalty,
        "freshness_penalty": aggregate.freshness_penalty,
        "observation_count": indicators.observation_count,
        "source_assets": asset_payload,
        "recommendation_gates": decision.gates,
        "risk_insufficiency_reason": risk.insufficiency_reason,
        "analysis_mode": config.analysis_mode,
        "fundamentals_used": config.analysis_mode != "price_only_baseline",
        "supported_horizons": list(config.supported_horizons),
        "factor_policy": {
            "macd_indicator": config.factor_policy.macd_indicator,
            "macd_score_low": config.factor_policy.macd_score_low,
            "macd_score_high": config.factor_policy.macd_score_high,
            "abnormal_volume_indicator": config.factor_policy.abnormal_volume_indicator,
            "liquidity_indicator": config.factor_policy.liquidity_indicator,
            "liquidity_score_low": config.factor_policy.liquidity_score_low,
            "liquidity_score_high": config.factor_policy.liquidity_score_high,
            "strict_finite_inputs": config.factor_policy.strict_finite_inputs,
            "buy_min_liquidity_20d": config.recommendation.buy_min_liquidity_20d,
        },
    }
    price_asset_metadata = (
        next(
            (asset for asset in asset_payload if asset["id"] == str(price_asset.id)),
            None,
        )
        if price_asset is not None
        else None
    )
    if price_asset_metadata is not None:
        data_quality["price_source"] = {
            "asset_id": price_asset_metadata["id"],
            "provider": price_asset_metadata["provider"],
            "subject": price_asset_metadata["subject"],
        }
        if "return_definition" in price_asset_metadata:
            data_quality["return_definition"] = price_asset_metadata["return_definition"]
        if "dividends_included" in price_asset_metadata:
            data_quality["dividends_included"] = price_asset_metadata["dividends_included"]
    reasons = generate_reasons(indicators, fundamentals, aggregate)
    risks = generate_risks(indicators, fundamentals, risk, aggregate)
    return AnalysisComputation(
        indicators=indicators,
        fundamentals=fundamentals,
        aggregate=aggregate,
        scenarios=scenarios,
        risk_score=risk.score,
        risk_class=risk.risk_class,
        recommendation=decision.recommendation,
        reasons=reasons,
        risks=risks,
        data_quality=data_quality,
        source_assets=asset_payload,
        current_price=price,
        daily_change=daily_change,
    )


def _compute_listing_from_asof(
    *,
    listing: Listing,
    asof: AsOfData,
    provider: str,
    config: ScoringConfig,
    decision_time: datetime,
    subject: str | None = None,
    benchmark_subject: str | None = None,
    sample_support: dict[str, int] | None = None,
    target_date: date | None = None,
) -> AnalysisComputation:
    symbol = subject or listing.provider_symbol or listing.ticker
    price_asset = asof.latest_asset(provider=provider, kind="price_history", subject=symbol)
    price_frame = asof.price_frame(
        provider=provider,
        subject=symbol,
        through_date=target_date,
    )
    benchmark_frame: pl.DataFrame | None = None
    source_assets = [price_asset]
    if benchmark_subject:
        benchmark_asset = asof.latest_asset(
            provider=provider,
            kind="price_history",
            subject=benchmark_subject,
        )
        benchmark_frame = asof.price_frame(
            provider=provider,
            subject=benchmark_subject,
            through_date=target_date,
        )
        source_assets.append(benchmark_asset)
    facts: list[Any] = []
    if config.analysis_mode != "price_only_baseline":
        facts = list(
            asof.fundamental_facts(
                company_id=listing.security.company_id,
                available_through=decision_time,
            ).select_related("source_asset")
        )
        source_assets.extend(fact.source_asset for fact in facts)
    return compute_listing_analysis(
        listing=listing,
        price_frame=price_frame,
        benchmark_frame=benchmark_frame,
        facts=facts,
        source_assets=source_assets,
        price_asset=price_asset,
        config=config,
        decision_time=decision_time,
        sample_support=sample_support,
    )


def _create_analysis_run(
    *,
    generated_at: datetime,
    data_cutoff: datetime,
    target_date: date,
    issued_on_time: bool,
    universe_snapshot: UniverseSnapshot,
    config: ScoringConfig,
    config_hash_value: str,
    code_revision_value: str,
) -> AnalysisRun:
    return AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=data_cutoff,
        target_date=target_date,
        issued_on_time=issued_on_time,
        universe_snapshot=universe_snapshot,
        config_version=config.version,
        config_hash=config_hash_value,
        code_revision=code_revision_value,
    )


def _persist_listing_analysis(
    *,
    run: AnalysisRun,
    listing: Listing,
    computation: AnalysisComputation,
    generated_at: datetime,
    data_cutoff: datetime,
    model_version: str,
    config_hash_value: str,
    code_revision_value: str,
    advisory_context: AdvisoryForecastContext | None = None,
) -> PersistedAnalysis:
    if run.issued_on_time:
        _validate_on_time_source_assets(computation.source_assets, data_cutoff=data_cutoff)
    advisory_forecasts = (
        advisory_context.forecasts.get(str(listing.pk), {}) if advisory_context is not None else {}
    )
    analysis = _create_stock_analysis(
        run,
        listing,
        computation,
        advisory_forecasts=advisory_forecasts,
    )
    decision_predictions = append_predictions(
        analysis=analysis,
        computation=computation,
        generated_at=generated_at,
        data_cutoff=data_cutoff,
        issued_on_time=run.issued_on_time,
        supported_horizons=tuple(computation.data_quality["supported_horizons"]),
        model_version=model_version,
        config_hash_value=config_hash_value,
        source_assets=computation.source_assets,
        code_revision_value=code_revision_value,
    )
    advisory_predictions: tuple[Prediction, ...] = ()
    if advisory_context is not None:
        advisory_predictions = append_advisory_predictions(
            analysis=analysis,
            forecasts=advisory_forecasts,
            panel_asset=advisory_context.panel.asset,
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            issued_on_time=run.issued_on_time,
            model_version=advisory_context.model_version,
            config_hash_value=advisory_context.config_hash,
            source_assets=computation.source_assets,
            code_revision_value=code_revision_value,
        )
    return PersistedAnalysis(
        run=run,
        analysis=analysis,
        predictions=(*decision_predictions, *advisory_predictions),
        computation=computation,
    )


def _validate_on_time_source_assets(
    source_assets: list[dict[str, Any]],
    *,
    data_cutoff: datetime,
) -> None:
    for asset in source_assets:
        for field in ("available_at", "retrieved_at"):
            timestamp = datetime.fromisoformat(str(asset[field]))
            if timestamp > data_cutoff:
                raise ValueError(
                    f"On-time analysis source asset {asset['id']} has {field} after data cutoff"
                )


@transaction.atomic
def analyze_listing(
    *,
    listing: Listing,
    universe_snapshot: UniverseSnapshot,
    decision_time: datetime | None = None,
    target_date: date | None = None,
    issued_on_time: bool | None = None,
    provider: str = "synthetic_demo",
    subject: str | None = None,
    benchmark_subject: str | None = None,
    store: AssetStore | None = None,
    config_path: Path | None = None,
    sample_support: dict[str, int] | None = None,
) -> PersistedAnalysis:
    require_stock_research_listing(listing, operation="Stock analysis")
    generated_at = decision_time or timezone.now()
    logical_target_date = target_date or generated_at.date()
    _validate_snapshot_for_target(universe_snapshot, logical_target_date)
    run_issued_on_time = _issued_on_time(
        universe_snapshot,
        generated_at=generated_at,
        target_date=logical_target_date,
        explicit=issued_on_time,
    )
    data_cutoff = _analysis_data_cutoff(
        generated_at,
        logical_target_date,
        issued_on_time=run_issued_on_time,
    )
    if not UniverseMembership.objects.filter(
        snapshot=universe_snapshot,
        listing=listing,
        eligible=True,
    ).exists():
        raise ValueError(
            f"Listing {listing.pk} is not an eligible member of snapshot {universe_snapshot.pk}"
        )
    config = load_scoring_config(config_path)
    digest = config_hash(config)
    revision = code_revision()
    asof = AsOfData(generated_at, store)
    run = _create_analysis_run(
        generated_at=generated_at,
        data_cutoff=data_cutoff,
        target_date=logical_target_date,
        issued_on_time=run_issued_on_time,
        universe_snapshot=universe_snapshot,
        config=config,
        config_hash_value=digest,
        code_revision_value=revision,
    )
    computation = _compute_listing_from_asof(
        listing=listing,
        asof=asof,
        provider=provider,
        config=config,
        decision_time=data_cutoff,
        subject=subject,
        benchmark_subject=benchmark_subject,
        sample_support=sample_support,
        target_date=logical_target_date,
    )
    return _persist_listing_analysis(
        run=run,
        listing=listing,
        computation=computation,
        generated_at=generated_at,
        data_cutoff=data_cutoff,
        model_version=_model_version(config.version, run.id.hex),
        config_hash_value=digest,
        code_revision_value=revision,
    )


@transaction.atomic
def analyze_snapshot(
    *,
    universe_snapshot: UniverseSnapshot,
    decision_time: datetime | None = None,
    target_date: date | None = None,
    issued_on_time: bool | None = None,
    provider: str = "synthetic_demo",
    benchmark_subject: str | None = None,
    store: AssetStore | None = None,
    config_path: Path | None = None,
    medium_forecast_config_path: Path | None = None,
    sample_support: dict[str, int] | None = None,
) -> list[PersistedAnalysis]:
    generated_at = decision_time or timezone.now()
    logical_target_date = target_date or generated_at.date()
    _validate_snapshot_for_target(universe_snapshot, logical_target_date)
    run_issued_on_time = _issued_on_time(
        universe_snapshot,
        generated_at=generated_at,
        target_date=logical_target_date,
        explicit=issued_on_time,
    )
    data_cutoff = _analysis_data_cutoff(
        generated_at,
        logical_target_date,
        issued_on_time=run_issued_on_time,
    )
    config = load_scoring_config(config_path)
    digest = config_hash(config)
    revision = code_revision()
    asset_store = store or AssetStore()
    asof = AsOfData(generated_at, asset_store)
    run = _create_analysis_run(
        generated_at=generated_at,
        data_cutoff=data_cutoff,
        target_date=logical_target_date,
        issued_on_time=run_issued_on_time,
        universe_snapshot=universe_snapshot,
        config=config,
        config_hash_value=digest,
        code_revision_value=revision,
    )
    model_version = _model_version(config.version, run.id.hex)
    results: list[PersistedAnalysis] = []
    memberships = list(
        UniverseMembership.objects.select_related(
            "listing__security__company",
        ).filter(snapshot=universe_snapshot, eligible=True)
    )
    for membership in memberships:
        require_stock_research_listing(
            membership.listing,
            operation="Snapshot stock analysis",
        )
    advisory_context: AdvisoryForecastContext | None = None
    panel_relative_path: str | None = None
    try:
        medium_config = load_medium_forecast_config(medium_forecast_config_path)
        if (
            benchmark_subject is not None
            and config.version in medium_config.enabled_scoring_versions
            and memberships
        ):
            medium_digest = medium_forecast_config_hash(medium_config)
            panel = build_medium_forecast_panel(
                listings=[membership.listing for membership in memberships],
                asof=asof,
                provider=provider,
                benchmark_subject=benchmark_subject,
                target_date=logical_target_date,
                generated_at=generated_at,
                run_id=run.id,
                config=medium_config,
                config_hash=medium_digest,
                scoring_config_version=config.version,
                scoring_config_hash=digest,
                universe_snapshot_id=universe_snapshot.id,
                universe_slug=universe_snapshot.universe.slug,
                universe_config_hash=universe_snapshot.config_hash,
                code_revision=revision,
                store=asset_store,
            )
            panel_relative_path = panel.asset.relative_path
            advisory_context = AdvisoryForecastContext(
                panel=panel,
                config=medium_config,
                config_hash=medium_digest,
                model_version=_model_version(medium_config.version, run.id.hex),
                forecasts=build_medium_forecasts(panel.frame, medium_config),
            )
        for membership in memberships:
            computation = _compute_listing_from_asof(
                listing=membership.listing,
                asof=asof,
                provider=provider,
                config=config,
                decision_time=data_cutoff,
                benchmark_subject=benchmark_subject,
                sample_support=sample_support,
                target_date=logical_target_date,
            )
            results.append(
                _persist_listing_analysis(
                    run=run,
                    listing=membership.listing,
                    computation=computation,
                    generated_at=generated_at,
                    data_cutoff=data_cutoff,
                    model_version=model_version,
                    config_hash_value=digest,
                    code_revision_value=revision,
                    advisory_context=advisory_context,
                )
            )
    except Exception:
        if panel_relative_path is not None:
            asset_store.resolve(panel_relative_path).unlink(missing_ok=True)
        raise
    return results


def append_predictions(
    *,
    analysis: StockAnalysis,
    computation: AnalysisComputation,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    supported_horizons: tuple[str, ...],
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, Any]],
    code_revision_value: str,
) -> tuple[Prediction, ...]:
    require_stock_research_listing(
        analysis.listing,
        operation="Stock prediction issuance",
    )
    if issued_on_time and (
        not analysis.run.issued_on_time or generated_at != analysis.run.generated_at
    ):
        raise ValueError(
            "Only predictions created with the original on-time analysis may be marked on time"
        )
    unsupported = set(supported_horizons) - {
        Prediction.Horizon.SHORT.value,
        Prediction.Horizon.MEDIUM.value,
        Prediction.Horizon.LONG.value,
    }
    if unsupported:
        raise ValueError(
            "Decision prediction issuance accepts scoring-group horizons only: "
            + ", ".join(sorted(unsupported))
        )
    horizons = tuple(Prediction.Horizon(value) for value in supported_horizons)
    if not horizons:
        raise ValueError("At least one supported prediction horizon is required")
    return tuple(
        _create_prediction(
            analysis=analysis,
            horizon=horizon,
            scenario=computation.scenarios[horizon],
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            issued_on_time=issued_on_time,
            model_version=model_version,
            config_hash_value=config_hash_value,
            source_assets=source_assets,
            code_revision_value=code_revision_value,
        )
        for horizon in horizons
    )


def append_advisory_predictions(
    *,
    analysis: StockAnalysis,
    forecasts: dict[str, MediumForecast],
    panel_asset: DataAsset,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, Any]],
    code_revision_value: str,
) -> tuple[Prediction, ...]:
    require_stock_research_listing(
        analysis.listing,
        operation="Advisory forecast issuance",
    )
    if issued_on_time and (
        not analysis.run.issued_on_time or generated_at != analysis.run.generated_at
    ):
        raise ValueError(
            "Only forecasts created with the original on-time analysis may be marked on time"
        )
    expected_horizons = {
        Prediction.Horizon.SIX_MONTH.value,
        Prediction.Horizon.TWELVE_MONTH.value,
    }
    if set(forecasts) != expected_horizons:
        raise ValueError("Advisory forecast issuance requires exactly 6m and 12m")
    panel_payload = _asset_payload(panel_asset)
    advisory_sources = [
        *source_assets,
        panel_payload,
    ]
    if issued_on_time:
        _validate_on_time_source_assets(advisory_sources, data_cutoff=data_cutoff)
    return tuple(
        _create_advisory_prediction(
            analysis=analysis,
            horizon=Prediction.Horizon(horizon),
            forecast=forecasts[horizon],
            panel_asset=panel_asset,
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            issued_on_time=issued_on_time,
            model_version=model_version,
            config_hash_value=config_hash_value,
            source_assets=advisory_sources,
            code_revision_value=code_revision_value,
        )
        for horizon in (
            Prediction.Horizon.SIX_MONTH.value,
            Prediction.Horizon.TWELVE_MONTH.value,
        )
    )


def _create_stock_analysis(
    run: AnalysisRun,
    listing: Listing,
    computation: AnalysisComputation,
    *,
    advisory_forecasts: dict[str, MediumForecast] | None = None,
) -> StockAnalysis:
    supported_horizons = {str(value) for value in computation.data_quality["supported_horizons"]}
    scenario_payloads = {
        horizon: scenario.as_dict() for horizon, scenario in computation.scenarios.items()
    }
    scenario_payloads.update(
        {
            horizon: forecast.scenario_payload()
            for horizon, forecast in (advisory_forecasts or {}).items()
        }
    )
    return StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=_decimal(computation.current_price, places=6),
        daily_change=_optional_decimal(computation.daily_change, places=6),
        overall_score=_decimal(computation.aggregate.overall, places=2),
        recommendation=computation.recommendation,
        risk_score=_optional_decimal(computation.risk_score, places=2),
        risk_class=computation.risk_class,
        confidence=_decimal(computation.aggregate.confidence, places=2),
        confidence_status=computation.aggregate.confidence_status,
        component_scores={
            "components": computation.aggregate.component_scores.components,
            "horizons": {
                horizon: score
                for horizon, score in computation.aggregate.horizon_scores.items()
                if horizon in supported_horizons
            },
            "factors": computation.aggregate.component_scores.factor_scores,
        },
        forecast_scenarios=build_forecast_scenario_document(scenario_payloads),
        short_scenario=computation.scenarios["short"].as_dict(),
        medium_scenario=computation.scenarios["medium"].as_dict(),
        long_scenario=computation.scenarios["long"].as_dict(),
        reasons=computation.reasons,
        risks=computation.risks,
        data_quality=computation.data_quality,
    )


def _create_prediction(
    *,
    analysis: StockAnalysis,
    horizon: Prediction.Horizon,
    scenario: Scenario,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, Any]],
    code_revision_value: str,
) -> Prediction:
    horizon_value = str(horizon)
    price_provider, price_subject = _prediction_price_source(analysis, source_assets)
    return Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=generated_at,
        target_date=analysis.run.target_date,
        issued_on_time=issued_on_time,
        horizon=horizon,
        evidence_role=Prediction.EvidenceRole.DECISION,
        evidence_grade=analysis.run.universe_snapshot.grade,
        source_mode=source_data_mode({"source_assets": source_assets}),
        price_provider=price_provider,
        price_subject=price_subject,
        price_at_prediction=analysis.current_price,
        bear_return=_optional_decimal(scenario.bear, places=4),
        base_return=_optional_decimal(scenario.base, places=4),
        bull_return=_optional_decimal(scenario.bull, places=4),
        probability_positive=_optional_decimal(scenario.probability_positive, places=4),
        confidence=_decimal(scenario.confidence, places=2),
        confidence_status=scenario.confidence_status,
        insufficiency_reason=scenario.insufficiency_reason,
        recommendation=analysis.recommendation,
        overall_score=analysis.overall_score,
        component_scores=analysis.component_scores,
        model_version=model_version,
        method_version=analysis.run.config_version,
        config_hash=config_hash_value,
        data_cutoff=data_cutoff,
        source_assets=source_assets,
        calculation={
            "schema_version": 1,
            "method": scenario.method,
            "method_version": analysis.run.config_version,
            "prediction_version": model_version,
            "config_hash": config_hash_value,
            "forecast_horizon": horizon_value,
            "score_group": horizon_value,
            "support": {
                "confidence": scenario.confidence,
                "confidence_status": scenario.confidence_status,
                "insufficiency_reason": scenario.insufficiency_reason,
            },
            "formula_inputs": {
                "scenario": {
                    "bear": scenario.bear,
                    "base": scenario.base,
                    "bull": scenario.bull,
                    "probability_positive": scenario.probability_positive,
                },
                "overall_score": float(analysis.overall_score),
                "risk_score": (
                    float(analysis.risk_score) if analysis.risk_score is not None else None
                ),
            },
            "contribution_detail": analysis.component_scores,
            "return_basis": analysis.data_quality.get("return_definition"),
            "dividends_included": analysis.data_quality.get("dividends_included"),
            "evidence_grade": analysis.run.universe_snapshot.grade,
            "price_subject": price_subject,
        },
        code_revision=code_revision_value,
    )


def _create_advisory_prediction(
    *,
    analysis: StockAnalysis,
    horizon: Prediction.Horizon,
    forecast: MediumForecast,
    panel_asset: DataAsset,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, Any]],
    code_revision_value: str,
) -> Prediction:
    price_provider, price_subject = _prediction_price_source(analysis, source_assets)
    calculation = dict(forecast.calculation)
    calculation.update(
        {
            "config_hash": config_hash_value,
            "prediction_version": model_version,
            "panel_asset_id": str(panel_asset.pk),
            "panel_sha256": panel_asset.sha256,
            "evidence_grade": analysis.run.universe_snapshot.grade,
            "price_subject": price_subject,
        }
    )
    scenario = forecast.scenario
    return Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=generated_at,
        target_date=analysis.run.target_date,
        issued_on_time=issued_on_time,
        horizon=horizon,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        evidence_grade=analysis.run.universe_snapshot.grade,
        source_mode=source_data_mode({"source_assets": source_assets}),
        price_provider=price_provider,
        price_subject=price_subject,
        price_at_prediction=analysis.current_price,
        bear_return=_optional_decimal(scenario.bear, places=4),
        base_return=_optional_decimal(scenario.base, places=4),
        bull_return=_optional_decimal(scenario.bull, places=4),
        probability_positive=_optional_decimal(scenario.probability_positive, places=4),
        confidence=_decimal(scenario.confidence, places=2),
        confidence_status=scenario.confidence_status,
        insufficiency_reason=scenario.insufficiency_reason,
        recommendation=analysis.recommendation,
        overall_score=analysis.overall_score,
        component_scores=analysis.component_scores,
        model_version=model_version,
        method_version=str(calculation["method_version"]),
        config_hash=config_hash_value,
        data_cutoff=data_cutoff,
        source_assets=source_assets,
        calculation=calculation,
        code_revision=code_revision_value,
    )


def _prediction_price_source(
    analysis: StockAnalysis,
    source_assets: list[dict[str, Any]],
) -> tuple[str, str]:
    price_source = analysis.data_quality.get("price_source")
    if isinstance(price_source, dict):
        return (
            str(price_source.get("provider") or ""),
            str(price_source.get("subject") or ""),
        )
    return infer_price_source(
        source_assets,
        subjects=(
            analysis.listing.provider_symbol or analysis.listing.ticker,
            analysis.listing.ticker,
        ),
    )


def _asset_payload(asset: DataAsset) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": str(asset.id),
        "provider": asset.provider,
        "kind": asset.kind,
        "subject": asset.subject,
        "relative_path": asset.relative_path,
        "sha256": asset.sha256,
        "retrieved_at": asset.retrieved_at.isoformat(),
        "available_at": asset.available_at.isoformat(),
    }
    if isinstance(asset.metadata.get("return_definition"), str):
        payload["return_definition"] = asset.metadata["return_definition"]
    if isinstance(asset.metadata.get("dividends_included"), bool):
        payload["dividends_included"] = asset.metadata["dividends_included"]
    return payload


def _dedupe_assets(assets: list[DataAsset]) -> list[DataAsset]:
    seen: set[str] = set()
    deduped: list[DataAsset] = []
    for asset in assets:
        key = str(asset.id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(asset)
    return deduped


def _model_version(config_version: str, run_hex: str) -> str:
    suffix = run_hex[:8]
    return f"{config_version}-{suffix}"[:40]


def _decimal(value: float, *, places: int) -> Decimal:
    return Decimal(str(round(value, places)))


def _optional_decimal(value: float | None, *, places: int) -> Decimal | None:
    if value is None:
        return None
    return _decimal(value, places=places)


def _analysis_data_cutoff(
    generated_at: datetime,
    target_date: date,
    *,
    issued_on_time: bool,
) -> datetime:
    if target_date > generated_at.date():
        raise ValueError(
            f"target_date ({target_date.isoformat()}) cannot be after generation date "
            f"({generated_at.date().isoformat()})"
        )
    if issued_on_time or target_date == generated_at.date():
        return generated_at
    return datetime.combine(target_date, time.max, tzinfo=generated_at.tzinfo)


def _issued_on_time(
    snapshot: UniverseSnapshot,
    *,
    generated_at: datetime,
    target_date: date,
    explicit: bool | None,
) -> bool:
    if explicit is not None:
        if explicit and snapshot.grade != UniverseSnapshot.Grade.OBSERVED:
            raise ValueError("Only an observed universe snapshot can be issued on time")
        if explicit and not is_observed_issuance_on_time(
            snapshot,
            target_date=target_date,
            generated_at=generated_at,
        ):
            raise ValueError("Observed analysis was generated after the next market session opened")
        return explicit
    return snapshot.grade == UniverseSnapshot.Grade.OBSERVED and generated_at.date() == target_date


def _validate_snapshot_for_target(snapshot: UniverseSnapshot, target_date: date) -> None:
    if snapshot.grade == UniverseSnapshot.Grade.OBSERVED and snapshot.as_of_date > target_date:
        raise ValueError(
            f"Observed universe snapshot {snapshot.pk} is dated "
            f"{snapshot.as_of_date.isoformat()} and cannot be used for earlier "
            f"target date {target_date.isoformat()}"
        )

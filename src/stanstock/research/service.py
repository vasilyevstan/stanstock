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
from stanstock.research.explanations import generate_reasons, generate_risks
from stanstock.research.fundamentals import calculate_fundamentals, inputs_from_facts
from stanstock.research.indicators import calculate_indicators
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.scenarios import build_scenarios
from stanstock.research.scoring import (
    aggregate_score,
    assess_risk,
    decide_recommendation,
    score_components,
)
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
    source_assets: list[dict[str, str]]
    current_price: float
    daily_change: float | None


@dataclass(frozen=True, slots=True)
class PersistedAnalysis:
    run: AnalysisRun
    analysis: StockAnalysis
    predictions: tuple[Prediction, Prediction, Prediction]
    computation: AnalysisComputation


def compute_listing_analysis(
    *,
    listing: Listing,
    price_frame: pl.DataFrame,
    config: ScoringConfig,
    decision_time: datetime,
    benchmark_frame: pl.DataFrame | None = None,
    facts: Any = (),
    source_assets: list[DataAsset] | None = None,
    sample_support: dict[str, int] | None = None,
) -> AnalysisComputation:
    indicators = calculate_indicators(
        price_frame, benchmark=benchmark_frame, windows=config.windows
    )
    price = indicators.values.get("last_close")
    if price is None:
        raise ValueError(f"No usable price history for {listing}")
    fundamental_inputs = inputs_from_facts(facts)
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
    asset_payload = [
        {
            "id": str(asset.id),
            "provider": asset.provider,
            "kind": asset.kind,
            "subject": asset.subject,
            "relative_path": asset.relative_path,
            "sha256": asset.sha256,
            "retrieved_at": asset.retrieved_at.isoformat(),
            "available_at": asset.available_at.isoformat(),
        }
        for asset in assets
    ]
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
    }
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
        config=config,
        decision_time=decision_time,
        sample_support=sample_support,
    )


def _create_analysis_run(
    *,
    generated_at: datetime,
    data_cutoff: datetime,
    target_date: date,
    universe_snapshot: UniverseSnapshot,
    config: ScoringConfig,
    config_hash_value: str,
    code_revision_value: str,
) -> AnalysisRun:
    return AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=data_cutoff,
        target_date=target_date,
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
) -> PersistedAnalysis:
    analysis = _create_stock_analysis(run, listing, computation)
    predictions = append_predictions(
        analysis=analysis,
        computation=computation,
        generated_at=generated_at,
        data_cutoff=data_cutoff,
        model_version=model_version,
        config_hash_value=config_hash_value,
        source_assets=computation.source_assets,
        code_revision_value=code_revision_value,
    )
    return PersistedAnalysis(
        run=run, analysis=analysis, predictions=predictions, computation=computation
    )


@transaction.atomic
def analyze_listing(
    *,
    listing: Listing,
    universe_snapshot: UniverseSnapshot,
    decision_time: datetime | None = None,
    target_date: date | None = None,
    provider: str = "synthetic_demo",
    subject: str | None = None,
    benchmark_subject: str | None = None,
    store: AssetStore | None = None,
    config_path: Path | None = None,
    sample_support: dict[str, int] | None = None,
) -> PersistedAnalysis:
    generated_at = decision_time or timezone.now()
    logical_target_date = target_date or generated_at.date()
    data_cutoff = _analysis_data_cutoff(generated_at, logical_target_date)
    _validate_snapshot_for_target(universe_snapshot, logical_target_date)
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
    provider: str = "synthetic_demo",
    benchmark_subject: str | None = None,
    store: AssetStore | None = None,
    config_path: Path | None = None,
    sample_support: dict[str, int] | None = None,
) -> list[PersistedAnalysis]:
    generated_at = decision_time or timezone.now()
    logical_target_date = target_date or generated_at.date()
    data_cutoff = _analysis_data_cutoff(generated_at, logical_target_date)
    _validate_snapshot_for_target(universe_snapshot, logical_target_date)
    config = load_scoring_config(config_path)
    digest = config_hash(config)
    revision = code_revision()
    asof = AsOfData(generated_at, store)
    run = _create_analysis_run(
        generated_at=generated_at,
        data_cutoff=data_cutoff,
        target_date=logical_target_date,
        universe_snapshot=universe_snapshot,
        config=config,
        config_hash_value=digest,
        code_revision_value=revision,
    )
    model_version = _model_version(config.version, run.id.hex)
    results: list[PersistedAnalysis] = []
    memberships = UniverseMembership.objects.select_related(
        "listing__security__company",
    ).filter(snapshot=universe_snapshot, eligible=True)
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
            )
        )
    return results


def append_predictions(
    *,
    analysis: StockAnalysis,
    computation: AnalysisComputation,
    generated_at: datetime,
    data_cutoff: datetime,
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, str]],
    code_revision_value: str,
) -> tuple[Prediction, Prediction, Prediction]:
    predictions = tuple(
        _create_prediction(
            analysis=analysis,
            horizon=horizon,
            scenario=computation.scenarios[horizon],
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            model_version=model_version,
            config_hash_value=config_hash_value,
            source_assets=source_assets,
            code_revision_value=code_revision_value,
        )
        for horizon in (
            Prediction.Horizon.SHORT,
            Prediction.Horizon.MEDIUM,
            Prediction.Horizon.LONG,
        )
    )
    return (predictions[0], predictions[1], predictions[2])


def _create_stock_analysis(
    run: AnalysisRun,
    listing: Listing,
    computation: AnalysisComputation,
) -> StockAnalysis:
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
            "horizons": computation.aggregate.horizon_scores,
            "factors": computation.aggregate.component_scores.factor_scores,
        },
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
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, str]],
    code_revision_value: str,
) -> Prediction:
    return Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=generated_at,
        target_date=analysis.run.target_date,
        horizon=horizon,
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
        config_hash=config_hash_value,
        data_cutoff=data_cutoff,
        source_assets=source_assets,
        code_revision=code_revision_value,
    )


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


def _analysis_data_cutoff(generated_at: datetime, target_date: date) -> datetime:
    if target_date > generated_at.date():
        raise ValueError(
            f"target_date ({target_date.isoformat()}) cannot be after generation date "
            f"({generated_at.date().isoformat()})"
        )
    if target_date == generated_at.date():
        return generated_at
    return datetime.combine(target_date, time.max, tzinfo=generated_at.tzinfo)


def _validate_snapshot_for_target(snapshot: UniverseSnapshot, target_date: date) -> None:
    if snapshot.grade == UniverseSnapshot.Grade.OBSERVED and snapshot.as_of_date > target_date:
        raise ValueError(
            f"Observed universe snapshot {snapshot.pk} is dated "
            f"{snapshot.as_of_date.isoformat()} and cannot be used for earlier "
            f"target date {target_date.isoformat()}"
        )

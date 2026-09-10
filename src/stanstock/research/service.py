from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID

import polars as pl
from django.db import transaction
from django.utils import timezone

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.asof import AsOfData, PriceFrameChecksumMismatchError
from stanstock.data.assets import AssetStore, open_asset_store
from stanstock.data.models import (
    DataAsset,
    FundamentalFact,
    Listing,
    ProviderRecord,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.data.provider_policy import (
    PRIVATE_USAGE_SCOPE,
    TWELVE_DATA_PROVIDER,
    normalized_provider_plan,
)
from stanstock.data.sec_config import SecFundamentalsConfig, load_sec_fundamentals_config
from stanstock.research.affordability import (
    DECISION_TARGET_DATE_BASIS,
    UNDER_10_BAND,
    classify_price_band,
)
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
from stanstock.research.long_forecast_config import (
    LongForecastConfig,
    load_long_forecast_config,
    long_forecast_config_hash,
)
from stanstock.research.long_forecasts import LongForecast, build_long_forecasts
from stanstock.research.medium_forecasts import (
    MediumForecast,
    MediumPanel,
    build_medium_forecast_panel,
    build_medium_forecasts,
)
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.provenance import source_data_mode
from stanstock.research.refresh_evidence import (
    ANALYSIS_OUTPUT_MANIFEST_KIND,
    ANALYSIS_RUN_FIELDS,
    ANALYSIS_RUN_MODEL,
    PREDICTION_FIELDS,
    PREDICTION_MODEL,
    STOCK_ANALYSIS_FIELDS,
    STOCK_ANALYSIS_MODEL,
    ManifestEntry,
    ManifestPayloadError,
    ManifestPlan,
    actual_output_plan,
    build_manifest_envelope,
    build_output_plan,
    decimal_from_float,
    dumps_canonical_envelope,
    model_row_values,
    optional_decimal_from_float,
    row_digest,
)
from stanstock.research.scenarios import build_scenarios
from stanstock.research.scoring import (
    aggregate_score,
    assess_risk,
    decide_recommendation,
    score_components,
)
from stanstock.research.timing import is_observed_issuance_on_time
from stanstock.research.types import AggregateScore, IndicatorResult, ResearchValues, Scenario
from stanstock.research.under10 import (
    UNDER10_CONCEPTS,
    build_under10_assessment,
    canonical_json,
    qualify_under10_sec_facts,
)

#: `data_quality` key carrying the unactivated Under-$10 shadow assessment.
#: It is written only when a *new* analysis qualifies; an absent key means
#: "not assessed", never "assessed and failed". Nothing backfills it.
UNDER10_ASSESSMENT_KEY = "under10_assessment"


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
    price_asset: DataAsset | None = None


@dataclass(frozen=True, slots=True)
class PersistedAnalysis:
    run: AnalysisRun
    analysis: StockAnalysis
    predictions: tuple[Prediction, ...]
    computation: AnalysisComputation


@dataclass(slots=True)
class AnalysisOutputPaths:
    """Optional mutable out-parameter for `analyze_snapshot`.

    A caller that needs its own later cleanup after `analyze_snapshot`
    itself has already returned successfully (for example, an outer
    transaction that still has to run its own post-analysis checks before
    it can commit) passes one instance in and reads the two fields back
    directly once the call returns -- the exact paths this call itself
    wrote, tracked from the producer's own local state as they are set,
    never re-derived by a separate, fallible post-write `DataAsset` query
    keyed on `run_id`/`kind`/`provider` that could itself fail independent
    of whether the paths actually exist, or -- for a research-grade run
    that legitimately writes neither -- silently return nothing to clean
    up.
    """

    panel_relative_path: str | None = None
    manifest_relative_path: str | None = None


@dataclass(frozen=True, slots=True)
class AdvisoryForecastContext:
    panel: MediumPanel
    config: MediumForecastConfig
    config_hash: str
    model_version: str
    forecasts: dict[str, dict[str, MediumForecast]]


@dataclass(frozen=True, slots=True)
class LongForecastContext:
    config: LongForecastConfig
    config_hash: str
    model_version: str
    forecasts: dict[str, dict[str, LongForecast]]


@dataclass(frozen=True, slots=True)
class _Under10DecisionEvidence:
    """Immutable decision-prediction provenance needed for shadow replay."""

    price_provider: str
    price_subject: str
    price_entry: Mapping[str, Any]


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
        price_asset=price_asset,
    )


def _compute_listing_from_asof(
    *,
    listing: Listing,
    asof: AsOfData,
    provider: str,
    config: ScoringConfig,
    decision_time: datetime,
    issued_on_time: bool,
    provider_plan: str | None,
    code_revision_value: str,
    subject: str | None = None,
    benchmark_subject: str | None = None,
    sample_support: dict[str, int] | None = None,
    target_date: date | None = None,
) -> AnalysisComputation:
    symbol = subject or listing.provider_symbol or listing.ticker
    price_read = asof.price_frame_with_diagnostics(
        provider=provider,
        subject=symbol,
        through_date=target_date,
    )
    price_asset = price_read.asset
    price_frame = price_read.frame
    benchmark_frame: pl.DataFrame | None = None
    source_assets = [price_asset]
    if benchmark_subject:
        benchmark_read = asof.price_frame_with_diagnostics(
            provider=provider,
            subject=benchmark_subject,
            through_date=target_date,
        )
        benchmark_frame = benchmark_read.frame
        source_assets.append(benchmark_read.asset)
    facts: list[Any] = []
    full_analysis_facts_loaded = config.analysis_mode != "price_only_baseline"
    if full_analysis_facts_loaded:
        facts = list(
            asof.fundamental_facts(
                company_id=listing.security.company_id,
                available_through=decision_time,
            ).select_related("source_asset")
        )
        source_assets.extend(fact.source_asset for fact in facts)
    computation = compute_listing_analysis(
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
    return _with_under10_assessment(
        computation,
        listing=listing,
        asof=asof,
        provider=provider,
        provider_plan=provider_plan,
        loaded_facts=facts,
        loaded_facts_cover_all_concepts=full_analysis_facts_loaded,
        price_frame=price_frame,
        price_asset=price_asset,
        invalid_session_date_rows=price_read.invalid_session_date_rows,
        decision_time=decision_time,
        target_date=target_date or decision_time.date(),
        issued_on_time=issued_on_time,
        code_revision_value=code_revision_value,
    )


def _with_under10_assessment(
    computation: AnalysisComputation,
    *,
    listing: Listing,
    asof: AsOfData,
    provider: str,
    provider_plan: str | None,
    loaded_facts: Sequence[FundamentalFact],
    loaded_facts_cover_all_concepts: bool,
    price_frame: pl.DataFrame,
    price_asset: DataAsset | None,
    invalid_session_date_rows: int,
    decision_time: datetime,
    target_date: date,
    issued_on_time: bool,
    code_revision_value: str,
) -> AnalysisComputation:
    """Attach the shadow Under-$10 diagnostic to a qualifying computation.

    The scored computation above is already final: this only adds one nested
    `data_quality` key. A listing whose decision-run reference close is not a
    valid USD Under-$10 close is returned untouched and issues no additional
    SEC query at all.

    The reference close classified here is the *rounded* value
    `_create_stock_analysis` persists, so the band recorded in the payload is
    the same band sample construction later reads -- never an unrounded float
    and never a mutable current market row.
    """
    reference_close = _decimal(computation.current_price, places=6)
    band = classify_price_band(
        close=reference_close,
        price_date=target_date,
        date_basis=DECISION_TARGET_DATE_BASIS,
        currency=listing.currency,
    )
    if band is None or band.slug != UNDER_10_BAND:
        return computation
    if loaded_facts_cover_all_concepts:
        # Full-analysis mode already read every visible fact for this
        # company under the same cutoff; a second shadow query would be a
        # duplicate read of the same evidence.
        facts: Sequence[FundamentalFact] = loaded_facts
    else:
        facts = list(
            asof.fundamental_facts(
                company_id=listing.security.company_id,
                concepts=list(UNDER10_CONCEPTS),
                available_through=decision_time,
            ).select_related("source_asset")
        )
    qualified_facts = qualify_under10_sec_facts(facts)
    payload = build_under10_assessment(
        facts=qualified_facts,
        sec_config=_sec_fundamentals_config(),
        price_frame=price_frame,
        price_asset=price_asset,
        price_source=_mapping_or_none(computation.data_quality.get("price_source")),
        reference_close=reference_close,
        target_date=target_date,
        data_cutoff=decision_time,
        code_revision_value=code_revision_value,
        provider=provider,
        provider_plan=provider_plan,
        evidence_cutoff_safe=_shadow_evidence_cutoff_safe(
            qualified_facts,
            issued_on_time=issued_on_time,
            data_cutoff=decision_time,
        ),
        company_identity_present=listing.security.company_id is not None,
        invalid_session_date_rows=invalid_session_date_rows,
        listing_id=str(listing.id),
    )
    # `replace` keeps every other computed field -- including the original
    # `source_assets` list object -- identical. The shadow SEC evidence lives
    # only under the new nested key and never joins the prediction manifest.
    return replace(
        computation,
        data_quality={**computation.data_quality, UNDER10_ASSESSMENT_KEY: payload},
    )


@lru_cache(maxsize=1)
def _sec_fundamentals_config() -> SecFundamentalsConfig:
    """The reviewed SEC fundamentals configuration, loaded at most once.

    Mirrors the cached opportunity-policy loader; the pure assessment builder
    never loads configuration itself.
    """
    return load_sec_fundamentals_config()


def _mapping_or_none(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _shadow_evidence_cutoff_safe(
    facts: Sequence[FundamentalFact],
    *,
    issued_on_time: bool,
    data_cutoff: datetime,
) -> bool:
    """Whether qualified SEC shadow evidence satisfies the asset cutoff rule.

    The strict asset-retrieval check belongs to on-time issuance only. A
    research-grade reconstruction may legitimately read evidence retrieved
    later than its historical cutoff, while `AsOfData` and the correction
    resolution still require the *facts* themselves to be provably available
    at that cutoff. The caller supplies the exact provider-qualified sequence
    also passed to the defensive assessment builder, so foreign evidence
    cannot make an SEC cutoff claim fail.
    """
    if not issued_on_time:
        return True
    for asset in _dedupe_assets([fact.source_asset for fact in facts]):
        if _asset_cutoff_violation(_asset_payload(asset), data_cutoff=data_cutoff) is not None:
            return False
    return True


def under10_assessment_matches_persisted_evidence(
    *,
    analysis: StockAnalysis,
    recorded: Mapping[str, Any],
    store: AssetStore | None = None,
) -> bool:
    """Whether a stored shadow assessment exactly replays from its evidence.

    This is a read-only evidence-validation predicate, not an alternate producer:
    it never repairs the payload, returns reconstructed values, or writes a
    model. Price provenance comes from the immutable decision-prediction
    cohort rather than the mutable parent JSON alone. The exact price asset
    named there is read through :class:`AsOfData`, while SEC facts are
    independently selected under the original run's generation and data
    cutoffs. Only exact canonical equality of the resulting solvency and
    liquidity blocks admits the recorded values for display.

    Expected evidence failures return ``False``. Database failures and
    programming errors are deliberately not hidden.
    """
    if analysis._state.adding:
        return False
    recorded_solvency = recorded.get("solvency")
    recorded_liquidity = recorded.get("liquidity")
    recorded_split = recorded.get("split_verification")
    if (
        not isinstance(recorded_solvency, Mapping)
        or not isinstance(recorded_liquidity, Mapping)
        or not isinstance(recorded_split, Mapping)
    ):
        return False

    decision_evidence = _under10_decision_evidence(analysis)
    if decision_evidence is None:
        return False
    price_entry = decision_evidence.price_entry
    asset_id = price_entry.get("id")
    asset_checksum = price_entry.get("sha256")
    if not isinstance(asset_id, str) or not isinstance(asset_checksum, str):
        return False
    try:
        asset_uuid = UUID(asset_id)
    except ValueError:
        return False
    if str(asset_uuid) != asset_id:
        return False
    if not _under10_parent_price_anchor_matches(
        analysis,
        asset_id=asset_id,
        asset_checksum=asset_checksum,
        price_provider=decision_evidence.price_provider,
        price_subject=decision_evidence.price_subject,
    ):
        return False
    recorded_price_asset = recorded_liquidity.get("price_asset")
    if (
        not isinstance(recorded_price_asset, Mapping)
        or recorded_price_asset.get("id") != asset_id
        or recorded_price_asset.get("sha256") != asset_checksum
    ):
        return False

    try:
        price_asset = DataAsset.objects.get(pk=asset_uuid)
    except DataAsset.DoesNotExist:
        return False
    if (
        str(price_asset.pk) != asset_id
        or price_asset.sha256 != asset_checksum
        or price_asset.provider != decision_evidence.price_provider
        or price_asset.kind != "price_history"
        or price_asset.subject != decision_evidence.price_subject
        or price_entry.get("provider") != price_asset.provider
        or price_entry.get("kind") != price_asset.kind
        or price_entry.get("subject") != price_asset.subject
    ):
        return False
    if recorded_split.get("provider") != price_asset.provider:
        return False

    asset_store = store or AssetStore()
    asof = AsOfData(analysis.run.generated_at, asset_store)
    try:
        price_read = asof.price_frame_for_asset_with_diagnostics(
            asset=price_asset,
            through_date=analysis.run.target_date,
        )
    except (
        PriceFrameChecksumMismatchError,
        FileNotFoundError,
        pl.exceptions.PolarsError,
        ValueError,
    ):
        return False

    try:
        facts = list(
            asof.fundamental_facts(
                company_id=analysis.listing.security.company_id,
                concepts=list(UNDER10_CONCEPTS),
                available_through=analysis.run.data_cutoff,
            ).select_related("source_asset")
        )
    except ValueError:
        return False
    qualified_facts = qualify_under10_sec_facts(facts)
    try:
        replayed = build_under10_assessment(
            facts=qualified_facts,
            sec_config=_sec_fundamentals_config(),
            price_frame=price_read.frame,
            price_asset=price_asset,
            price_source={
                "asset_id": asset_id,
                "provider": price_asset.provider,
                "subject": price_asset.subject,
            },
            reference_close=analysis.current_price,
            target_date=analysis.run.target_date,
            data_cutoff=analysis.run.data_cutoff,
            code_revision_value=analysis.run.code_revision,
            provider=price_asset.provider,
            # ProviderRecord is mutable capability context and is not
            # decision evidence. The split block is therefore not replayed;
            # only its recorded provider is bound above. Provider plan does
            # not enter either evidence-derived block compared below.
            provider_plan=None,
            evidence_cutoff_safe=_shadow_evidence_cutoff_safe(
                qualified_facts,
                issued_on_time=analysis.run.issued_on_time,
                data_cutoff=analysis.run.data_cutoff,
            ),
            company_identity_present=analysis.listing.security.company_id is not None,
            invalid_session_date_rows=price_read.invalid_session_date_rows,
            listing_id=str(analysis.listing_id),
        )
    except ValueError:
        return False

    replayed_solvency = replayed.get("solvency")
    replayed_liquidity = replayed.get("liquidity")
    if not isinstance(replayed_solvency, Mapping) or not isinstance(replayed_liquidity, Mapping):
        return False
    try:
        return canonical_json(recorded_solvency) == canonical_json(
            replayed_solvency
        ) and canonical_json(recorded_liquidity) == canonical_json(replayed_liquidity)
    except (TypeError, ValueError, OverflowError):
        return False


def _under10_decision_evidence(
    analysis: StockAnalysis,
) -> _Under10DecisionEvidence | None:
    """Resolve one internally agreeing immutable decision-prediction cohort."""
    run = analysis.run
    predictions = list(
        Prediction.objects.filter(
            analysis_id=analysis.pk,
            evidence_role=Prediction.EvidenceRole.DECISION,
        )
        .only(
            "id",
            "listing_id",
            "generated_at",
            "target_date",
            "issued_on_time",
            "price_at_prediction",
            "price_provider",
            "price_subject",
            "data_cutoff",
            "source_assets",
            "code_revision",
        )
        .order_by("pk")
    )
    if not predictions:
        return None

    first = predictions[0]
    price_provider = first.price_provider
    price_subject = first.price_subject
    if not price_provider or not price_subject:
        return None
    manifest = _canonical_under10_source_manifest(first.source_assets)
    if manifest is None:
        return None
    manifest_json, manifest_entries = manifest
    for prediction in predictions:
        if (
            prediction.listing_id != analysis.listing_id
            or prediction.generated_at != run.generated_at
            or prediction.target_date != run.target_date
            or prediction.data_cutoff != run.data_cutoff
            or prediction.code_revision != run.code_revision
            or prediction.issued_on_time != run.issued_on_time
            or prediction.price_at_prediction != analysis.current_price
            or prediction.price_provider != price_provider
            or prediction.price_subject != price_subject
        ):
            return None
        candidate_manifest = _canonical_under10_source_manifest(prediction.source_assets)
        if candidate_manifest is None or candidate_manifest[0] != manifest_json:
            return None

    price_entries = [
        entry
        for entry in manifest_entries
        if entry.get("provider") == price_provider
        and entry.get("kind") == "price_history"
        and entry.get("subject") == price_subject
    ]
    if len(price_entries) != 1:
        return None
    return _Under10DecisionEvidence(
        price_provider=price_provider,
        price_subject=price_subject,
        price_entry=price_entries[0],
    )


def _canonical_under10_source_manifest(
    value: object,
) -> tuple[str, tuple[Mapping[str, Any], ...]] | None:
    if not isinstance(value, list) or not all(isinstance(entry, Mapping) for entry in value):
        return None
    entries = tuple(entry for entry in value if isinstance(entry, Mapping))
    try:
        serialized = canonical_json({"source_assets": list(entries)})
    except (TypeError, ValueError, OverflowError):
        return None
    return serialized, entries


def _under10_parent_price_anchor_matches(
    analysis: StockAnalysis,
    *,
    asset_id: str,
    asset_checksum: str,
    price_provider: str,
    price_subject: str,
) -> bool:
    data_quality = analysis.data_quality
    if not isinstance(data_quality, Mapping):
        return False
    price_source = data_quality.get("price_source")
    source_assets = data_quality.get("source_assets")
    if (
        not isinstance(price_source, Mapping)
        or price_source.get("asset_id") != asset_id
        or price_source.get("provider") != price_provider
        or price_source.get("subject") != price_subject
        or not isinstance(source_assets, list)
    ):
        return False
    matches = [
        entry
        for entry in source_assets
        if isinstance(entry, Mapping) and entry.get("id") == asset_id
    ]
    if len(matches) != 1:
        return False
    matching = matches[0]
    return bool(
        matching.get("sha256") == asset_checksum
        and matching.get("provider") == price_provider
        and matching.get("kind") == "price_history"
        and matching.get("subject") == price_subject
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
    long_context: LongForecastContext | None = None,
) -> PersistedAnalysis:
    if run.issued_on_time:
        _validate_on_time_source_assets(computation.source_assets, data_cutoff=data_cutoff)
    advisory_forecasts = (
        advisory_context.forecasts.get(str(listing.pk), {}) if advisory_context is not None else {}
    )
    long_forecasts = (
        long_context.forecasts.get(str(listing.pk), {}) if long_context is not None else {}
    )
    analysis = _create_stock_analysis(
        run,
        listing,
        computation,
        advisory_forecasts=advisory_forecasts,
        long_forecasts=long_forecasts,
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
    long_predictions: tuple[Prediction, ...] = ()
    if long_context is not None:
        long_predictions = append_long_advisory_predictions(
            analysis=analysis,
            forecasts=long_forecasts,
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            issued_on_time=run.issued_on_time,
            model_version=long_context.model_version,
            config_hash_value=long_context.config_hash,
            code_revision_value=code_revision_value,
        )
    return PersistedAnalysis(
        run=run,
        analysis=analysis,
        predictions=(*decision_predictions, *advisory_predictions, *long_predictions),
        computation=computation,
    )


def _validate_on_time_source_assets(
    source_assets: list[dict[str, Any]],
    *,
    data_cutoff: datetime,
) -> None:
    for asset in source_assets:
        field = _asset_cutoff_violation(asset, data_cutoff=data_cutoff)
        if field is not None:
            raise ValueError(
                f"On-time analysis source asset {asset['id']} has {field} after data cutoff"
            )


def _asset_cutoff_violation(
    asset: Mapping[str, Any],
    *,
    data_cutoff: datetime,
) -> str | None:
    """First cutoff-safety field this asset violates, or ``None``.

    Fields are checked in the released order -- ``available_at`` before
    ``retrieved_at`` -- so the core validator above keeps raising on exactly
    the field and with exactly the wording it always did. The shadow
    assessment reuses the same rule to *withhold* rather than to fail a run.
    """
    for field in ("available_at", "retrieved_at"):
        timestamp = datetime.fromisoformat(str(asset[field]))
        if timestamp > data_cutoff:
            return field
    return None


def _resolve_provider_plan(provider: str) -> str | None:
    """The normalized recorded provider plan, resolved once per analysis run.

    Only Twelve Data records a plan whose value changes the split-capability
    refusal, so no other provider issues a query. A missing record, a missing
    leaf, a non-string leaf, and an empty label are all "no recorded plan";
    none of them is treated as an unresolved value that a later per-listing
    lookup could retry.

    Only the plan leaf is read: the surrounding provider metadata document
    carries entitlement and licensing detail this assessment has no reason to
    touch.
    """
    if provider != TWELVE_DATA_PROVIDER:
        return None
    recorded = (
        ProviderRecord.objects.filter(provider=provider)
        .values_list("metadata__plan", flat=True)
        .first()
    )
    if not isinstance(recorded, str):
        return None
    return normalized_provider_plan(recorded)


def _analysis_output_manifest_entries(
    run: AnalysisRun, results: list[PersistedAnalysis]
) -> list[ManifestEntry]:
    """The complete, exact set of manifest entries for one observed run:
    the `AnalysisRun` row itself, every persisted `StockAnalysis`, and every
    persisted `Prediction` -- one canonical full-row digest each. Built
    directly from the just-persisted ORM instances (never re-queried),
    since this runs inside the same still-open transaction that created
    them.
    """
    entries = [
        ManifestEntry(
            model=ANALYSIS_RUN_MODEL,
            row_id=str(run.id),
            digest=row_digest(ANALYSIS_RUN_MODEL, model_row_values(run, ANALYSIS_RUN_FIELDS)),
        )
    ]
    for persisted in results:
        entries.append(
            ManifestEntry(
                model=STOCK_ANALYSIS_MODEL,
                row_id=str(persisted.analysis.id),
                digest=row_digest(
                    STOCK_ANALYSIS_MODEL,
                    model_row_values(persisted.analysis, STOCK_ANALYSIS_FIELDS),
                ),
            )
        )
        for prediction in persisted.predictions:
            entries.append(
                ManifestEntry(
                    model=PREDICTION_MODEL,
                    row_id=str(prediction.id),
                    digest=row_digest(
                        PREDICTION_MODEL, model_row_values(prediction, PREDICTION_FIELDS)
                    ),
                )
            )
    return entries


def _write_analysis_output_manifest(
    *,
    run: AnalysisRun,
    results: list[PersistedAnalysis],
    plan: ManifestPlan,
    store: AssetStore,
    retrieved_at: datetime,
) -> str:
    """Write and register the one immutable manifest asset binding this
    observed `AnalysisRun` to the exact, complete set of rows it produced
    and the exact output `plan` it was required to produce.

    Mirrors `stanstock.data.live_us._ensure_snapshot`'s exact idiom: decide
    whether the target path already exists *before* writing (a fresh
    `run.id` UUID means it never should, so this only guards against a
    stale leftover from an earlier failed attempt at the very same path),
    write bytes then register the `DataAsset` row, and on any failure from
    either step unlink only a file this call itself just created -- never a
    genuinely pre-existing one.

    Every expected storage-layer failure (path resolution, physical write)
    is normalized into a stable, path-free `RefreshVerificationError` raised
    `from None`; a `DataAsset.objects.create` failure (a DB integrity or
    programming error) is never relabeled or swallowed, only cleaned up
    after.
    """
    try:
        envelope = build_manifest_envelope(
            run_id=run.id, plan=plan, entries=_analysis_output_manifest_entries(run, results)
        )
        envelope_bytes = dumps_canonical_envelope(envelope)
    except (ManifestPayloadError, TypeError, ValueError):
        # Canonicalization/serialization of the manifest payload itself
        # (an unsupported field type, a naive datetime, ...) is an
        # expected-failure-shaped bug in the manifest contract, never a
        # storage-layer or DB fault -- normalize it the same path-free way
        # before any file or row is touched.
        raise RefreshVerificationError(
            "analysis_output_manifest_generation_failed",
            "The analysis-output manifest payload could not be generated",
        ) from None
    relative_path = f"research/analysis/{run.id}/output-manifest.json"
    try:
        resolved = store.resolve(relative_path)
        file_already_existed = resolved.exists()
    except (OSError, ValueError):
        raise RefreshVerificationError(
            "analysis_output_manifest_path_unavailable",
            "The analysis-output manifest path could not be checked",
        ) from None
    try:
        written = store.write_bytes(relative_path, envelope_bytes)
    except (OSError, ValueError):
        raise RefreshVerificationError(
            "analysis_output_manifest_write_failed",
            "The analysis-output manifest could not be written",
        ) from None
    try:
        with transaction.atomic():
            DataAsset.objects.create(
                provider="stanstock",
                kind=ANALYSIS_OUTPUT_MANIFEST_KIND,
                subject=str(run.id),
                relative_path=written.relative_path,
                sha256=written.sha256,
                retrieved_at=retrieved_at,
                available_at=retrieved_at,
                metadata={"usage_scope": PRIVATE_USAGE_SCOPE},
            )
    except Exception:
        if not file_already_existed:
            # A cleanup fault here (e.g. an unexpected permission error on
            # unlink) must never replace the original exception being
            # handled: swallow only this narrow best-effort cleanup step,
            # never the failure that actually caused it.
            _safe_unlink(store, relative_path)
        raise
    return relative_path


def _finalize_observed_manifest(
    *,
    run: AnalysisRun,
    universe_snapshot: UniverseSnapshot,
    results: list[PersistedAnalysis],
    plan: ManifestPlan,
    store: AssetStore,
    generated_at: datetime,
) -> str | None:
    """Write and register the analysis-output manifest for `run` if, and
    only if, `universe_snapshot` is OBSERVED-grade -- every observed
    `AnalysisRun` must have exactly one manifest, and a research-grade run
    must never carry one.

    Before registering anything, the exact rows this call actually
    persisted (`results`) must match the `plan` computed *before* the first
    write, by count as well as by key: a defensive invariant that should
    never trip in a correctly-behaving run, but must fail loudly rather
    than silently register a manifest that disagrees with its own plan.
    """
    if universe_snapshot.grade != UniverseSnapshot.Grade.OBSERVED:
        return None
    actual = actual_output_plan(
        (persisted.analysis for persisted in results),
        (prediction for persisted in results for prediction in persisted.predictions),
    )
    if actual != plan:
        raise ValueError("Observed analysis output does not match its precomputed output plan")
    return _write_analysis_output_manifest(
        run=run,
        results=results,
        plan=plan,
        store=store,
        retrieved_at=generated_at,
    )


def _safe_unlink(store: AssetStore, relative_path: str) -> None:
    """Best-effort cleanup of a file this call itself just wrote, only
    used when an observed run's own transaction (including its own
    atomic-exit/commit) later fails. A cleanup fault here (e.g. an
    unexpected permission error) must never replace the original exception
    already being propagated."""
    try:
        store.resolve(relative_path).unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


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
    """Analyze one listing against `universe_snapshot`.

    Every OBSERVED-grade `universe_snapshot` produces exactly one
    analysis-output manifest, the same guarantee `analyze_snapshot` gives a
    whole run -- this is the single-listing entry point, so its own
    precomputed plan is always the simple "decision predictions only, one
    listing" case (never medium/long advisory lanes, which only
    `analyze_snapshot` can activate).

    The whole body runs inside one explicit `with transaction.atomic():`
    wrapped by this outer, undecorated function's own `try/except`: this
    (not a bare `@transaction.atomic` decorator) is what lets the outer
    `except` also catch a failure from the atomic block's own exit
    (commit or savepoint release), not only a failure raised by code inside
    the block, so a manifest file can never be orphaned by either kind of
    failure.
    """
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
    asset_store = store or open_asset_store()
    asof = AsOfData(generated_at, asset_store)
    manifest_relative_path: str | None = None
    persisted: PersistedAnalysis
    try:
        with transaction.atomic():
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
                issued_on_time=run_issued_on_time,
                provider_plan=_resolve_provider_plan(provider),
                code_revision_value=revision,
                subject=subject,
                benchmark_subject=benchmark_subject,
                sample_support=sample_support,
                target_date=logical_target_date,
            )
            plan = build_output_plan(
                eligible_listing_ids={listing.pk},
                decision_horizons=frozenset(config.supported_horizons),
                medium_active=False,
                long_active=False,
            )
            persisted = _persist_listing_analysis(
                run=run,
                listing=listing,
                computation=computation,
                generated_at=generated_at,
                data_cutoff=data_cutoff,
                model_version=_model_version(config.version, run.id.hex),
                config_hash_value=digest,
                code_revision_value=revision,
            )
            manifest_relative_path = _finalize_observed_manifest(
                run=run,
                universe_snapshot=universe_snapshot,
                results=[persisted],
                plan=plan,
                store=asset_store,
                generated_at=generated_at,
            )
    except Exception:
        if manifest_relative_path is not None:
            _safe_unlink(asset_store, manifest_relative_path)
        raise
    return persisted


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
    long_forecast_config_path: Path | None = None,
    sample_support: dict[str, int] | None = None,
    output_paths: AnalysisOutputPaths | None = None,
) -> list[PersistedAnalysis]:
    """Analyze every eligible member of `universe_snapshot`.

    An OBSERVED-grade `universe_snapshot` produces exactly one
    analysis-output manifest for the whole run, covering the `AnalysisRun`
    row and every `StockAnalysis`/`Prediction` this call persists. The
    exact plan (eligible listings and the exact prediction key multiset)
    is computed *before* the first `StockAnalysis`/`Prediction` write --
    once the medium/long advisory lanes are gated on or off for this run,
    never inferred afterward from whatever happened to be written -- and
    the manifest is registered only once the actual persisted rows are
    checked to match that plan exactly.

    The whole write path runs inside one explicit
    `with transaction.atomic():` wrapped by this outer, undecorated
    function's own `try/except`: this (not a bare `@transaction.atomic`
    decorator) is what lets the outer `except` also catch a failure from
    the atomic block's own exit (commit or savepoint release), not only a
    failure raised by code inside the block, so a panel or manifest file
    can never be orphaned by either kind of failure.

    `output_paths`, if supplied, is populated with this call's own panel
    and manifest relative paths (or left `None` for whichever this run
    does not write) as soon as each is known -- so a caller that must run
    its own checks after this function already returned successfully can
    read the exact paths this call owns directly, instead of re-deriving
    them with a separate post-write `DataAsset` query.
    """
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
    asset_store = store or open_asset_store()
    asof = AsOfData(generated_at, asset_store)
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
    results: list[PersistedAnalysis] = []
    panel_relative_path: str | None = None
    manifest_relative_path: str | None = None
    try:
        with transaction.atomic():
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
            advisory_context: AdvisoryForecastContext | None = None
            long_context: LongForecastContext | None = None
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
                if output_paths is not None:
                    output_paths.panel_relative_path = panel_relative_path
                advisory_context = AdvisoryForecastContext(
                    panel=panel,
                    config=medium_config,
                    config_hash=medium_digest,
                    model_version=_model_version(medium_config.version, run.id.hex),
                    forecasts=build_medium_forecasts(panel.frame, medium_config),
                )
            computations: dict[str, AnalysisComputation] = {}
            provider_plan = _resolve_provider_plan(provider)
            for membership in memberships:
                computation = _compute_listing_from_asof(
                    listing=membership.listing,
                    asof=asof,
                    provider=provider,
                    config=config,
                    decision_time=data_cutoff,
                    issued_on_time=run_issued_on_time,
                    provider_plan=provider_plan,
                    code_revision_value=revision,
                    benchmark_subject=benchmark_subject,
                    sample_support=sample_support,
                    target_date=logical_target_date,
                )
                computations[str(membership.listing.pk)] = computation
            long_config = load_long_forecast_config(long_forecast_config_path)
            if (
                provider == long_config.price_provider == "twelve_data"
                and long_config.fundamentals_provider == "sec"
                and config.version in long_config.enabled_scoring_versions
                and ProviderRecord.objects.filter(
                    provider=long_config.fundamentals_provider,
                    enabled=True,
                ).exists()
                and memberships
            ):
                current_prices: dict[str, float] = {}
                current_price_assets: dict[str, DataAsset] = {}
                for membership in memberships:
                    listing_id = str(membership.listing.pk)
                    computation = computations[listing_id]
                    if computation.price_asset is None:
                        raise ValueError(
                            f"Long forecast price asset is missing for {membership.listing}"
                        )
                    current_prices[listing_id] = computation.current_price
                    current_price_assets[listing_id] = computation.price_asset
                long_digest = long_forecast_config_hash(long_config)
                long_context = LongForecastContext(
                    config=long_config,
                    config_hash=long_digest,
                    model_version=_model_version(long_config.version, run.id.hex),
                    forecasts=build_long_forecasts(
                        listings=[membership.listing for membership in memberships],
                        asof=asof,
                        data_cutoff=data_cutoff,
                        target_date=logical_target_date,
                        config=long_config,
                        current_prices=current_prices,
                        price_assets=current_price_assets,
                    ),
                )
            plan = build_output_plan(
                eligible_listing_ids={membership.listing.pk for membership in memberships},
                decision_horizons=frozenset(config.supported_horizons),
                medium_active=advisory_context is not None,
                long_active=long_context is not None,
            )
            for membership in memberships:
                computation = computations[str(membership.listing.pk)]
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
                        long_context=long_context,
                    )
                )
            manifest_relative_path = _finalize_observed_manifest(
                run=run,
                universe_snapshot=universe_snapshot,
                results=results,
                plan=plan,
                store=asset_store,
                generated_at=generated_at,
            )
            if output_paths is not None:
                output_paths.manifest_relative_path = manifest_relative_path
    except Exception:
        if panel_relative_path is not None:
            _safe_unlink(asset_store, panel_relative_path)
        if manifest_relative_path is not None:
            _safe_unlink(asset_store, manifest_relative_path)
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


def append_long_advisory_predictions(
    *,
    analysis: StockAnalysis,
    forecasts: dict[str, LongForecast],
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    code_revision_value: str,
) -> tuple[Prediction, ...]:
    require_stock_research_listing(
        analysis.listing,
        operation="Long advisory forecast issuance",
    )
    if issued_on_time and (
        not analysis.run.issued_on_time or generated_at != analysis.run.generated_at
    ):
        raise ValueError(
            "Only forecasts created with the original on-time analysis may be marked on time"
        )
    expected_horizons = {
        Prediction.Horizon.THREE_YEAR.value,
        Prediction.Horizon.FIVE_YEAR.value,
    }
    if set(forecasts) != expected_horizons:
        raise ValueError("Long advisory forecast issuance requires exactly 3y and 5y")
    return tuple(
        _create_long_advisory_prediction(
            analysis=analysis,
            horizon=Prediction.Horizon(horizon),
            forecast=forecasts[horizon],
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            issued_on_time=issued_on_time,
            model_version=model_version,
            config_hash_value=config_hash_value,
            code_revision_value=code_revision_value,
        )
        for horizon in (
            Prediction.Horizon.THREE_YEAR.value,
            Prediction.Horizon.FIVE_YEAR.value,
        )
    )


def _create_stock_analysis(
    run: AnalysisRun,
    listing: Listing,
    computation: AnalysisComputation,
    *,
    advisory_forecasts: dict[str, MediumForecast] | None = None,
    long_forecasts: dict[str, LongForecast] | None = None,
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
    scenario_payloads.update(
        {
            horizon: {
                **forecast.scenario_payload(),
                "evidence_grade": _long_forecast_evidence_grade(run),
            }
            for horizon, forecast in (long_forecasts or {}).items()
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


def _create_long_advisory_prediction(
    *,
    analysis: StockAnalysis,
    horizon: Prediction.Horizon,
    forecast: LongForecast,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    code_revision_value: str,
) -> Prediction:
    source_assets = [_asset_payload(asset) for asset in forecast.source_assets]
    if issued_on_time:
        _validate_on_time_source_assets(source_assets, data_cutoff=data_cutoff)
    price_provider, price_subject = _prediction_price_source(analysis, source_assets)
    calculation = dict(forecast.calculation)
    calculation.update(
        {
            "config_hash": config_hash_value,
            "prediction_version": model_version,
            "price_subject": price_subject,
        }
    )
    scenario = forecast.scenario
    evidence_grade = _long_forecast_evidence_grade(analysis.run)
    return Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=generated_at,
        target_date=analysis.run.target_date,
        issued_on_time=issued_on_time,
        horizon=horizon,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        evidence_grade=evidence_grade,
        source_mode=source_data_mode({"source_assets": source_assets}),
        price_provider=price_provider,
        price_subject=price_subject,
        price_at_prediction=analysis.current_price,
        bear_return=_optional_decimal(scenario.bear, places=4),
        base_return=_optional_decimal(scenario.base, places=4),
        bull_return=_optional_decimal(scenario.bull, places=4),
        probability_positive=None,
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
        calculation={
            **calculation,
            "evidence_grade": evidence_grade,
        },
        code_revision=code_revision_value,
    )


def _long_forecast_evidence_grade(run: AnalysisRun) -> str:
    if run.issued_on_time and run.universe_snapshot.grade == UniverseSnapshot.Grade.OBSERVED:
        return UniverseSnapshot.Grade.OBSERVED
    return UniverseSnapshot.Grade.RESEARCH


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
    return decimal_from_float(value, places=places)


def _optional_decimal(value: float | None, *, places: int) -> Decimal | None:
    return optional_decimal_from_float(value, places=places)


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

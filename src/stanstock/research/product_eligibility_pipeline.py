"""The v2 append-only adapter around the frozen public price-product engine.

One calculation is composed, not two issuances. Forecast source roles remain
split-only. The separate decision role includes exact batch/witness authority.
Readers reopen physical sources but never run Monte Carlo unless replay is
explicitly requested by an offline verifier.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any, cast
from uuid import UUID, uuid4

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.revision import clean_git_revision
from stanstock.core.verification_types import AssetRef, RefreshVerificationError
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    read_checksummed_bytes,
    resolve_asset_ref,
)
from stanstock.data.liquidity_evidence import (
    VerifiedLiquidityBatch,
    require_liquidity_authorization,
    verify_liquidity_batch,
)
from stanstock.data.models import DataAsset, UniverseMembership, UniverseSnapshot
from stanstock.data.research_product import (
    CapturedProductIntake,
    load_product_intake,
    product_intake_payload,
    product_membership_payload,
    verify_product_intake_membership,
)
from stanstock.research.decision_support_eligibility import (
    FORECAST_CONFIG_HASH,
    MOMENTUM_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_PAYLOAD_SCHEMA,
    PRODUCT_VERSION,
    LiquidityAssessment,
    apply_liquidity_to_risk,
    canonical_json,
    compose_decision_support,
    load_decision_support_config,
    strict_json,
)
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.price_product import (
    FilteredReturns,
    HorizonProjection,
    PriceProductInputError,
    PriceProductResult,
    SourceExecutionBinding,
    apply_recommendation_policy,
    calculate_momentum,
    calculate_price_product,
    calculate_risk,
    complete_input_hash,
    deterministic_seed,
    filter_historical_returns,
    prediction_model_version,
)
from stanstock.research.price_product_config import FHS_METHOD_VERSION
from stanstock.research.product_pipeline import (
    ProductSourceSelection,
    select_product_source,
    verified_source_window_payload,
)
from stanstock.research.refresh_evidence import (
    ANALYSIS_RUN_FIELDS,
    ANALYSIS_RUN_MODEL,
    PREDICTION_FIELDS,
    PREDICTION_MODEL,
    STOCK_ANALYSIS_FIELDS,
    STOCK_ANALYSIS_MODEL,
    ManifestEntry,
    actual_output_plan,
    build_manifest_envelope,
    build_output_plan,
    dumps_canonical_envelope,
    lookup_manifest,
    model_row_values,
    parse_manifest_envelope,
    row_digest,
)

CALCULATION_KIND = "research_product_calculation"
CALCULATION_CONTRACT = "research-product-calculation@2"
SOURCE_WINDOW_CONTRACT = "research-product-source-window@2"


def _jsonable(value: Any) -> Any:
    def encode(item: Any) -> str:
        if isinstance(item, (date, datetime)):
            return item.isoformat()
        if isinstance(item, (Decimal, UUID)):
            return str(item)
        raise TypeError("Unsupported C2 calculation value")

    return json.loads(json.dumps(value, default=encode, allow_nan=False))


def _refs(assets: tuple[DataAsset, ...]) -> list[dict[str, str]]:
    # Stable first-use ordering preserves the four public forecast references.
    unique = {asset.id: asset for asset in assets}
    return [asset_ref_for(asset).to_json() for asset in unique.values()]


@dataclass(frozen=True, slots=True)
class EligibilityGraph:
    intake: CapturedProductIntake
    captured: dict[str, Any]
    membership: dict[str, Any]
    membership_asset: DataAsset
    declaration: DataAsset
    batch: VerifiedLiquidityBatch
    selections: tuple[ProductSourceSelection, ...]
    source_time: datetime

    def roles(self, selection: ProductSourceSelection) -> dict[str, list[dict[str, str]]]:
        symbol = selection.listing.provider_symbol
        required_ids = {
            entry[key]["id"]
            for entry in self.batch.document["entries"]
            if entry["symbol"] in {symbol, "NVDA"}
            for key in ("split_asset", "split_raw_asset", "none_asset", "none_raw_asset")
            if entry[key] is not None
        }
        eligibility = tuple(asset for asset in self.batch.sources if str(asset.id) in required_ids)
        forecast = (
            selection.stock_asset,
            selection.stock_raw_asset,
            selection.benchmark_asset,
            selection.benchmark_raw_asset,
        )
        return {
            "forecast": _refs(forecast),
            "decision": _refs((*forecast, self.batch.asset, *eligibility)),
        }

    def complete_input(self, selection: ProductSourceSelection) -> dict[str, Any]:
        return {
            "domain": "stanstock-decision-support-eligibility-input-v1",
            "forecast_input": verified_source_window_payload(selection),
            "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
            "forecast_config_hash": FORECAST_CONFIG_HASH,
            "intake": self.captured,
            "membership": self.membership,
            "batch": self.batch.document,
            "roles": self.roles(selection),
        }

    def input_hash(self, selection: ProductSourceSelection) -> str:
        return hashlib.sha256(canonical_json(self.complete_input(selection))).hexdigest()

    def source_window(self, selection: ProductSourceSelection) -> dict[str, Any]:
        entry = next(
            item
            for item in self.batch.document["entries"]
            if item["symbol"] == selection.listing.provider_symbol
        )
        return {
            "contract": SOURCE_WINDOW_CONTRACT,
            "forecast_window": verified_source_window_payload(selection),
            "forecast_input_hash": complete_input_hash(selection.product_input),
            "forecast_config_hash": FORECAST_CONFIG_HASH,
            "input_hash": self.input_hash(selection),
            "liquidity_batch": asset_ref_for(self.batch.asset).to_json(),
            "liquidity_window": entry,
            "source_roles": self.roles(selection),
        }


@dataclass(frozen=True, slots=True)
class ProductEligibilityPersistedAnalysis:
    run: AnalysisRun
    analysis: StockAnalysis
    predictions: tuple[Prediction, ...]
    result: PriceProductResult
    selection: ProductSourceSelection


@dataclass(frozen=True, slots=True)
class VerifiedForecastSource:
    """Narrow frequency adapter; not an admission to any old study protocol."""

    analysis: StockAnalysis
    listing: Any
    prediction_rows: tuple[Prediction, ...]
    selection: ProductSourceSelection
    source_execution: SourceExecutionBinding
    calculation_artifact: DataAsset


def _require_observed_deadline(*, target_date: date, revision: str) -> None:
    explicit = os.environ.get("STANSTOCK_CODE_REVISION", "")
    if (
        re.fullmatch(r"[0-9a-f]{40}", explicit) is None
        or revision != explicit
        or clean_git_revision(settings.BASE_DIR) != explicit
    ):
        raise ValueError("Observed v2 issuance requires the exact clean committed revision")
    now = timezone.now()
    calendar = get_calendar(
        "XNYS", start=target_date - timedelta(days=7), end=target_date + timedelta(days=14)
    )
    target = calendar.date_to_session(target_date, direction="none")
    close = calendar.session_close(target).to_pydatetime()
    deadline = calendar.session_open(calendar.next_session(target)).to_pydatetime()
    if not close <= now < deadline:
        raise ValueError(
            "Observed v2 issuance is outside its independently proved session deadline"
        )


def _load_graph(
    *,
    snapshot: UniverseSnapshot,
    generation_boundary: datetime,
    store: AssetStore,
) -> EligibilityGraph:
    config = load_decision_support_config()
    membership = cast(dict[str, Any], product_membership_payload(snapshot, store=store))
    if (
        set(membership)
        != {
            "contract",
            "intake",
            "qualified_listing_ids",
            "candidate_states",
            "admissions",
            "catalog_assets",
            "benchmark_asset",
            "decision_time",
            "snapshot_id",
            "liquidity_batch",
            "forward_declaration",
        }
        or membership["contract"] != "research-product-membership@2"
    ):
        raise ValueError("V2 membership has the wrong exact envelope")
    source_time = datetime.fromisoformat(str(membership["decision_time"]))
    if source_time.tzinfo is None or source_time > generation_boundary:
        raise ValueError("V2 membership source boundary is after actual generation")
    intake_asset = resolve_asset_ref(AssetRef.from_json(membership["intake"]), cutoff=source_time)
    intake_payload = strict_json(read_checksummed_bytes(store, intake_asset))
    if not isinstance(intake_payload, dict):
        raise ValueError("V2 intake must be a verified object")
    intake = load_product_intake(
        target_date=snapshot.as_of_date,
        owner_id=intake_payload["owner_id"],
        issuance_key=intake_payload["issuance_key"],
        product_version=PRODUCT_VERSION,
        store=store,
    )
    if intake is None or intake.asset.id != intake_asset.id:
        raise ValueError("V2 membership substituted the original captured intake")
    captured = cast(dict[str, Any], product_intake_payload(intake, store=store))
    require_liquidity_authorization(captured)
    verify_product_intake_membership(
        snapshot=snapshot,
        payload=membership,
        cutoff=source_time,
        provider="twelve_data",
        store=store,
    )
    hashable = {key: value for key, value in membership.items() if key != "snapshot_id"}
    if hashlib.sha256(canonical_json(hashable)).hexdigest() != snapshot.config_hash:
        raise ValueError("V2 membership differs from its independently captured snapshot hash")
    assets = list(
        DataAsset.objects.filter(
            provider="stanstock",
            kind="research_product_membership",
            subject=str(snapshot.id),
        )
    )
    if len(assets) != 1 or assets[0].metadata != {"contract": "research-product-membership@2"}:
        raise ValueError("V2 membership registry is missing or ambiguous")
    membership_asset = assets[0]
    if membership_asset.available_at != source_time or membership_asset.retrieved_at != source_time:
        raise ValueError("V2 membership publication/source boundary diverged")
    batch_asset = resolve_asset_ref(
        AssetRef.from_json(membership["liquidity_batch"]), cutoff=source_time
    )
    batch = verify_liquidity_batch(
        asset=batch_asset, intake=intake, cutoff=source_time, store=store
    )
    if (
        batch.document["forecast_admissions"] != membership["admissions"]
        or batch.document["catalog_assets"] != membership["catalog_assets"]
        or batch.document["benchmark_asset"] != membership["benchmark_asset"]
        or batch.document["forward_declaration"] != membership["forward_declaration"]
    ):
        raise ValueError("V2 membership and closed acquisition batch diverged")
    declaration = resolve_asset_ref(
        AssetRef.from_json(membership["forward_declaration"]),
        cutoff=source_time,
    )
    from stanstock.research.forward_scorecard import require_v2_forward_declaration

    require_v2_forward_declaration(
        owner_id=captured["owner_id"],
        target_date=snapshot.as_of_date,
        before=batch.document["generated_at"],
        acquisition_job_id=batch.document["acquisition_job_id"],
        store=store,
        expected_asset=declaration,
    )
    catalogs = tuple(
        resolve_asset_ref(AssetRef.from_json(ref), cutoff=source_time)
        for ref in membership["catalog_assets"]
    )
    benchmark = resolve_asset_ref(
        AssetRef.from_json(membership["benchmark_asset"]), cutoff=source_time
    )
    rows = list(
        UniverseMembership.objects.filter(snapshot=snapshot).select_related("listing__security")
    )
    states = {
        str(row.listing_id): "admitted" if row.eligible else row.exclusion_reason for row in rows
    }
    if states != membership["candidate_states"]:
        raise ValueError("V2 registered membership differs from its full captured denominator")
    by_listing = {
        UUID(entry["listing_id"]): entry
        for entry in membership["admissions"].values()
        if entry["status"] == "admitted"
    }
    if {row.listing_id for row in rows if row.eligible} != set(by_listing) or not by_listing:
        raise ValueError("V2 output plan omits or adds admitted permanent listings")
    selections = []
    for row in sorted((row for row in rows if row.eligible), key=lambda row: str(row.listing_id)):
        stock = resolve_asset_ref(
            AssetRef.from_json(by_listing[row.listing_id]["price_asset"]), cutoff=source_time
        )
        entry = next(
            item
            for item in batch.document["entries"]
            if item["symbol"] == row.listing.provider_symbol
        )
        if (
            entry["split_asset"] is not None
            and entry["split_asset"] != asset_ref_for(stock).to_json()
        ):
            raise ValueError("V2 forecast and liquidity must bind the same selected split vintage")
        selection = select_product_source(
            listing=row.listing,
            target_date=snapshot.as_of_date,
            decision_time=source_time,
            provider="twelve_data",
            benchmark_subject="SPY",
            source_execution=SourceExecutionBinding(
                mode="provider",
                evidence_grade="observed" if snapshot.grade == "observed" else "research",
            ),
            store=store,
            config=config.forecast_config,
            stock_asset=stock,
            benchmark_asset=benchmark,
            catalog_assets=catalogs,
        )
        if (
            selection.product_input.stock.volume_adjustment_compatible
            or selection.product_input.benchmark.volume_adjustment_compatible
        ):
            raise ValueError("V2 cannot change frozen forecast volume compatibility flags")
        selections.append(selection)
    return EligibilityGraph(
        intake,
        captured,
        membership,
        membership_asset,
        declaration,
        batch,
        tuple(selections),
        source_time,
    )


def _result_payload(
    result: PriceProductResult,
    *,
    selection: ProductSourceSelection,
    assessment: LiquidityAssessment,
) -> dict[str, Any]:
    payload: dict[str, Any] = _jsonable(asdict(result))
    payload["schema_version"] = PRODUCT_PAYLOAD_SCHEMA
    payload["forecast_input_hash"] = complete_input_hash(selection.product_input)
    payload["forecast_config_hash"] = FORECAST_CONFIG_HASH
    payload["liquidity"] = _jsonable(asdict(assessment))
    payload["liquidity_required_dates"] = [
        day.isoformat() for day in selection.product_input.calendar_sessions[-20:]
    ]
    return payload


def _quantized(value: float) -> Decimal:
    rounded = Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
    return rounded.copy_abs() if rounded.is_zero() else rounded


def _artifact(
    *,
    run: AnalysisRun,
    graph: EligibilityGraph,
    selection: ProductSourceSelection,
    payload: dict[str, Any],
    store: AssetStore,
    paths: list[str],
) -> DataAsset:
    published_at = timezone.now()
    document = {
        "contract": CALCULATION_CONTRACT,
        "run_id": str(run.id),
        "listing_id": str(selection.listing.id),
        "generated_at": published_at.isoformat(),
        "input_hash": payload["input_hash"],
        "forecast_input_hash": payload["forecast_input_hash"],
        "forecast_config_hash": FORECAST_CONFIG_HASH,
        "source_window": graph.source_window(selection),
        "result": payload,
    }
    encoded = canonical_json(document)
    path = f"research/calculations/{PRODUCT_VERSION}/{run.id}/{selection.listing.id}.json"
    paths.append(path)
    stored = store.write_bytes(path, encoded)
    return DataAsset.objects.create(
        provider="stanstock",
        kind=CALCULATION_KIND,
        subject=str(run.id),
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        retrieved_at=published_at,
        available_at=published_at,
        schema_version=CALCULATION_CONTRACT,
        metadata={
            "contract": CALCULATION_CONTRACT,
            "listing_id": str(selection.listing.id),
            "input_hash": payload["input_hash"],
            "forecast_input_hash": payload["forecast_input_hash"],
            "forecast_config_hash": FORECAST_CONFIG_HASH,
        },
    )


def _persist_rows(
    *,
    run: AnalysisRun,
    selection: ProductSourceSelection,
    result: PriceProductResult,
    payload: dict[str, Any],
    artifact: DataAsset,
    graph: EligibilityGraph,
) -> ProductEligibilityPersistedAnalysis:
    prices = selection.product_input.stock.closes
    roles = graph.roles(selection)
    calculation = {**payload, "artifact": asset_ref_for(artifact).to_json(), "source_roles": roles}
    scenarios = {
        "schema_version": 1,
        "horizons": {
            projection.horizon: _jsonable(asdict(projection))
            for projection in result.forecast.projections
        },
    }
    analysis = StockAnalysis(
        run=run,
        listing=selection.listing,
        current_price=_quantized(prices[-1]),
        daily_change=_quantized(prices[-1] / prices[-2] - 1),
        overall_score=None,
        recommendation=result.recommendation.suggestion,
        risk_score=None,
        risk_class=result.risk.relative_volatility_label,
        confidence=None,
        confidence_status="not_estimated",
        component_scores={},
        forecast_scenarios=scenarios,
        short_scenario={},
        medium_scenario={},
        long_scenario={},
        reasons=list(result.recommendation.blocking_reasons),
        risks=list(result.risk.insufficiency_reasons),
        data_quality={
            "product_version": PRODUCT_VERSION,
            "input_hash": result.input_hash,
            "forecast_input_hash": payload["forecast_input_hash"],
            "forecast_config_hash": FORECAST_CONFIG_HASH,
            "momentum_insufficiency_reason": result.momentum_insufficiency_reason,
            "calculation_artifact": asset_ref_for(artifact).to_json(),
            "source_assets": roles["decision"],
            "source_roles": roles,
            "liquidity": payload["liquidity"],
        },
    )
    analysis.clean()
    analysis.save()
    plan: list[tuple[str, str, str, HorizonProjection | None]] = [
        (MOMENTUM_METHOD_VERSION, "decision", "6m", None)
    ]
    plan += [
        (FHS_METHOD_VERSION, "advisory", item.horizon, item) for item in result.forecast.projections
    ]
    predictions = []
    for method, role, horizon, projection in plan:
        triplet = None if projection is None else projection.ledger_returns
        prediction = Prediction(
            analysis=analysis,
            listing=selection.listing,
            generated_at=run.generated_at,
            target_date=run.target_date,
            issued_on_time=run.issued_on_time,
            horizon=horizon,
            evidence_role=role,
            evidence_grade=run.universe_snapshot.grade,
            source_mode="provider",
            price_provider="twelve_data",
            price_subject=selection.stock_asset.subject,
            price_at_prediction=analysis.current_price,
            bear_return=None if triplet is None else triplet.lower,
            base_return=None if triplet is None else triplet.median,
            bull_return=None if triplet is None else triplet.upper,
            probability_positive=None,
            confidence=None,
            confidence_status="not_estimated",
            insufficiency_reason=(result.momentum_insufficiency_reason or "")
            if projection is None
            else (projection.insufficiency_reason or ""),
            recommendation=result.recommendation.suggestion if role == "decision" else None,
            overall_score=None,
            component_scores={},
            model_version=prediction_model_version(method_version=method, issuance_id=run.id),
            method_version=method,
            config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
            data_cutoff=run.data_cutoff,
            source_assets=roles[role if role == "decision" else "forecast"],
            calculation={**calculation, "method_version": method},
            code_revision=run.code_revision,
        )
        prediction.clean()
        prediction.save()
        predictions.append(prediction)
    return ProductEligibilityPersistedAnalysis(run, analysis, tuple(predictions), result, selection)


def _entries(
    run: AnalysisRun, analyses: list[StockAnalysis], predictions: list[Prediction]
) -> list[ManifestEntry]:
    return [
        ManifestEntry(model, str(row.pk), row_digest(model, model_row_values(row, fields)))
        for model, fields, rows in (
            (ANALYSIS_RUN_MODEL, ANALYSIS_RUN_FIELDS, [run]),
            (STOCK_ANALYSIS_MODEL, STOCK_ANALYSIS_FIELDS, analyses),
            (PREDICTION_MODEL, PREDICTION_FIELDS, predictions),
        )
        for row in rows
    ]


def issue_decision_support_snapshot(
    *,
    universe_snapshot: UniverseSnapshot,
    decision_time: datetime,
    target_date: date,
    provider: str,
    benchmark_subject: str,
    store: AssetStore,
    code_revision: str,
    issued_on_time: bool,
    output_paths: Any | None = None,
) -> list[ProductEligibilityPersistedAnalysis]:
    config = load_decision_support_config()
    if (
        provider != "twelve_data"
        or benchmark_subject != "SPY"
        or universe_snapshot.universe.config_version != PRODUCT_VERSION
        or universe_snapshot.as_of_date != target_date
        or decision_time.tzinfo is None
        or decision_time > timezone.now()
        or type(issued_on_time) is not bool
    ):
        raise ValueError(
            "V2 requires its explicit provider/config/SPY/snapshot boundary; demo is unsupported"
        )
    if issued_on_time:
        if universe_snapshot.grade != UniverseSnapshot.Grade.OBSERVED:
            raise ValueError("An exceptional observed v2 request requires an observed snapshot")
        _require_observed_deadline(target_date=target_date, revision=code_revision)
    graph = _load_graph(snapshot=universe_snapshot, generation_boundary=decision_time, store=store)
    # Frozen numerical input/config/seed/flags, exactly one composition per listing.
    results = []
    for selection in graph.selections:
        frozen = calculate_price_product(selection.product_input, config=config.forecast_config)
        result = compose_decision_support(
            frozen_result=frozen,
            product_input=selection.product_input,
            liquidity=graph.batch.assessments[selection.listing.provider_symbol],
            input_hash=graph.input_hash(selection),
            config=config,
        )
        results.append(result)
    plan = build_output_plan(
        eligible_listing_ids={item.listing.id for item in graph.selections},
        decision_horizons=frozenset({"6m"}),
        medium_active=True,
        long_active=True,
    )
    paths: list[str] = []
    committed = False
    if output_paths is not None:
        output_paths.manifest_relative_path = None
    try:
        with transaction.atomic():
            if issued_on_time:
                _require_observed_deadline(target_date=target_date, revision=code_revision)
            run = AnalysisRun.objects.create(
                id=uuid4(),
                generated_at=timezone.now(),
                data_cutoff=graph.source_time,
                target_date=target_date,
                issued_on_time=issued_on_time,
                universe_snapshot=universe_snapshot,
                config_version=PRODUCT_VERSION,
                config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
                code_revision=code_revision,
            )
            persisted = []
            for selection, result in zip(graph.selections, results, strict=True):
                payload = _result_payload(
                    result,
                    selection=selection,
                    assessment=graph.batch.assessments[selection.listing.provider_symbol],
                )
                artifact = _artifact(
                    run=run,
                    graph=graph,
                    selection=selection,
                    payload=payload,
                    store=store,
                    paths=paths,
                )
                persisted.append(
                    _persist_rows(
                        run=run,
                        selection=selection,
                        result=result,
                        payload=payload,
                        artifact=artifact,
                        graph=graph,
                    )
                )
            analyses = [item.analysis for item in persisted]
            predictions = [row for item in persisted for row in item.predictions]
            if actual_output_plan(analyses, predictions) != plan:
                raise ValueError("V2 rows differ from the independent pre-write five-row plan")
            envelope = build_manifest_envelope(
                run_id=run.id, plan=plan, entries=_entries(run, analyses, predictions)
            )
            path = f"research/manifests/{PRODUCT_VERSION}/{run.id}.json"
            paths.append(path)
            stored = store.write_bytes(path, dumps_canonical_envelope(envelope))
            published_at = timezone.now()
            DataAsset.objects.create(
                provider="stanstock",
                kind="analysis_output_manifest",
                subject=str(run.id),
                relative_path=stored.relative_path,
                sha256=stored.sha256,
                retrieved_at=published_at,
                available_at=published_at,
                metadata={"contract": "analysis-output-manifest@1"},
            )
            verify_decision_support_output(run=run, store=store)
            if issued_on_time:
                _require_observed_deadline(target_date=target_date, revision=code_revision)
            if output_paths is not None:
                output_paths.manifest_relative_path = path
        committed = True
        return persisted
    finally:
        if not committed:
            for path in paths:
                if not DataAsset.objects.filter(relative_path=path).exists():
                    store.resolve(path).unlink(missing_ok=True)
            if output_paths is not None:
                output_paths.manifest_relative_path = None


def _verify_nonforecast(
    *,
    payload: Mapping[str, Any],
    selection: ProductSourceSelection,
    assessment: LiquidityAssessment,
) -> None:
    config = load_decision_support_config()
    base = config.forecast_config
    product_input = selection.product_input
    try:
        momentum = calculate_momentum(product_input.stock.closes, product_input.benchmark.closes)
        momentum_reason = None
    except PriceProductInputError as exc:
        momentum, momentum_reason = None, exc.reason_code
    filters: list[tuple[FilteredReturns | None, str | None]] = []
    for series in (product_input.stock, product_input.benchmark):
        try:
            filters.append((filter_historical_returns(series.closes), None))
        except PriceProductInputError as exc:
            filters.append((None, exc.reason_code))
    risk = calculate_risk(
        product_input.stock,
        stock_filter=filters[0][0],
        benchmark_filter=filters[1][0],
        stock_filter_reason=filters[0][1],
        benchmark_filter_reason=filters[1][1],
        config=base,
    )
    risk = apply_liquidity_to_risk(risk, assessment)
    recommendation = apply_recommendation_policy(
        momentum=momentum,
        momentum_insufficiency_reason=momentum_reason,
        risk=risk,
        target_close=product_input.stock.closes[-1],
        source_eligible=product_input.source_eligible,
        source_ineligibility_reasons=product_input.source_ineligibility_reasons,
        config=base,
    )
    for key, expected in {
        "momentum": None if momentum is None else asdict(momentum),
        "momentum_insufficiency_reason": momentum_reason,
        "risk": asdict(risk),
        "recommendation": asdict(recommendation),
        "liquidity": asdict(assessment),
    }.items():
        if payload.get(key) != _jsonable(expected):
            raise ValueError(
                "V2 decision/risk differs from independently reconstructed price/liquidity inputs"
            )


def verify_decision_support_output(
    *, run: AnalysisRun, store: AssetStore, replay: bool = False
) -> None:
    if run.config_version != PRODUCT_VERSION or run.config_hash != PRODUCT_EFFECTIVE_CONFIG_HASH:
        raise RefreshVerificationError("product_config_invalid", "V2 config identity is invalid")
    try:
        graph = _load_graph(
            snapshot=run.universe_snapshot, generation_boundary=run.generated_at, store=store
        )
        if (
            run.data_cutoff != graph.source_time
            or run.target_date != run.universe_snapshot.as_of_date
        ):
            raise ValueError("V2 run cutoff/target differs from its captured source boundary")
        if run.issued_on_time:
            calendar = get_calendar("XNYS")
            deadline = calendar.session_open(calendar.next_session(run.target_date)).to_pydatetime()
            if run.universe_snapshot.grade != "observed" or run.generated_at >= deadline:
                raise ValueError("V2 observed issuance did not meet its own deadline")
        lookup = lookup_manifest(run.id)
        if lookup.count != 1 or lookup.asset is None:
            raise ValueError("V2 output manifest is missing or ambiguous")
        manifest_id, plan, entries = parse_manifest_envelope(
            read_checksummed_bytes(store, lookup.asset)
        )
        if (
            lookup.asset.available_at < run.generated_at
            or lookup.asset.retrieved_at > timezone.now()
        ):
            raise ValueError("V2 output publication chronology is invalid")
        analyses = list(
            StockAnalysis.objects.filter(run=run)
            .select_related("listing__security")
            .order_by("listing_id")
        )
        predictions = list(Prediction.objects.filter(analysis__run=run).order_by("id"))
        expected_plan = build_output_plan(
            eligible_listing_ids={selection.listing.id for selection in graph.selections},
            decision_horizons=frozenset({"6m"}),
            medium_active=True,
            long_active=True,
        )
        if (
            manifest_id != str(run.id)
            or plan != expected_plan
            or actual_output_plan(analyses, predictions) != expected_plan
            or entries
            != sorted(
                _entries(run, analyses, predictions), key=lambda item: (item.model, item.row_id)
            )
        ):
            raise ValueError("V2 complete manifest rows/plan diverged")
        artifacts = list(
            DataAsset.objects.filter(
                provider="stanstock", kind=CALCULATION_KIND, subject=str(run.id)
            )
        )
        if len(artifacts) != len(analyses):
            raise ValueError("V2 calculation registry is incomplete or ambiguous")
        by_listing = {str(asset.metadata.get("listing_id")): asset for asset in artifacts}
        for selection in graph.selections:
            artifact = by_listing.get(str(selection.listing.id))
            if artifact is None:
                raise ValueError("V2 calculation artifact has the wrong listing")
            document = strict_json(read_checksummed_bytes(store, artifact))
            if not isinstance(document, dict) or set(document) != {
                "contract",
                "run_id",
                "listing_id",
                "generated_at",
                "input_hash",
                "forecast_input_hash",
                "forecast_config_hash",
                "source_window",
                "result",
            }:
                raise ValueError("V2 calculation artifact has the wrong exact envelope")
            input_hash = graph.input_hash(selection)
            forecast_hash = complete_input_hash(selection.product_input)
            if (
                artifact.schema_version != CALCULATION_CONTRACT
                or document["contract"] != CALCULATION_CONTRACT
                or document["run_id"] != str(run.id)
                or document["listing_id"] != str(selection.listing.id)
                or document["input_hash"] != input_hash
                or document["forecast_input_hash"] != forecast_hash
                or document["forecast_config_hash"] != FORECAST_CONFIG_HASH
                or document["source_window"] != graph.source_window(selection)
                or artifact.available_at.isoformat() != document["generated_at"]
                or artifact.retrieved_at != artifact.available_at
                or not run.generated_at <= artifact.available_at <= timezone.now()
                or artifact.metadata
                != {
                    "contract": CALCULATION_CONTRACT,
                    "listing_id": str(selection.listing.id),
                    "input_hash": input_hash,
                    "forecast_input_hash": forecast_hash,
                    "forecast_config_hash": FORECAST_CONFIG_HASH,
                }
            ):
                raise ValueError("V2 artifact substituted source/config/input/publication identity")
            payload = document["result"]
            if not isinstance(payload, dict):
                raise ValueError("V2 result is not an object")
            assessment = graph.batch.assessments[selection.listing.provider_symbol]
            _verify_nonforecast(payload=payload, selection=selection, assessment=assessment)
            forecast = payload["forecast"]
            expected_seed = deterministic_seed(
                method_version=FHS_METHOD_VERSION,
                effective_config_hash=FORECAST_CONFIG_HASH,
                listing_id=selection.listing.id,
                target_date=run.target_date,
            )
            if (
                payload.get("schema_version") != PRODUCT_PAYLOAD_SCHEMA
                or payload.get("payload_schema") != PRODUCT_PAYLOAD_SCHEMA
                or payload.get("product_version") != PRODUCT_VERSION
                or payload.get("input_hash") != input_hash
                or payload.get("forecast_input_hash") != forecast_hash
                or payload.get("forecast_config_hash") != FORECAST_CONFIG_HASH
                or payload.get("effective_config_hash") != PRODUCT_EFFECTIVE_CONFIG_HASH
                or forecast["method_version"] != FHS_METHOD_VERSION
                or forecast["seed"] != expected_seed
                or forecast["path_count"] != 8192
                or tuple(item["horizon"] for item in forecast["projections"])
                != ("6m", "12m", "3y", "5y")
            ):
                raise ValueError("V2 numerical identity or forecast projection coverage is invalid")
            if replay:
                base = calculate_price_product(
                    selection.product_input, config=load_decision_support_config().forecast_config
                )
                expected = compose_decision_support(
                    frozen_result=base,
                    product_input=selection.product_input,
                    liquidity=assessment,
                    input_hash=input_hash,
                    config=load_decision_support_config(),
                )
                if payload != _result_payload(expected, selection=selection, assessment=assessment):
                    raise ValueError("V2 complete calculation differs from frozen-engine replay")
            roles = graph.roles(selection)
            rows = [row for row in predictions if row.listing_id == selection.listing.id]
            for row in rows:
                method = (
                    MOMENTUM_METHOD_VERSION
                    if row.evidence_role == "decision"
                    else FHS_METHOD_VERSION
                )
                expected_calculation = {
                    **payload,
                    "artifact": asset_ref_for(artifact).to_json(),
                    "source_roles": roles,
                    "method_version": method,
                }
                if (
                    row.method_version != method
                    or row.calculation != expected_calculation
                    or row.source_assets
                    != roles["decision" if row.evidence_role == "decision" else "forecast"]
                    or row.config_hash != PRODUCT_EFFECTIVE_CONFIG_HASH
                    or row.price_at_prediction
                    != _quantized(selection.product_input.stock.closes[-1])
                    or row.price_provider != "twelve_data"
                    or row.source_mode != "provider"
                    or row.price_subject != selection.stock_asset.subject
                ):
                    raise ValueError(
                        "V2 row differs from its role-specific calculation/source binding"
                    )
                row.clean()
                if row.evidence_role == "advisory":
                    projected = next(
                        item for item in forecast["projections"] if item["horizon"] == row.horizon
                    )
                    ledger = projected["ledger_returns"]
                    expected_returns = (
                        (None, None, None)
                        if ledger is None
                        else tuple(Decimal(ledger[key]) for key in ("lower", "median", "upper"))
                    )
                    if (row.bear_return, row.base_return, row.bull_return) != expected_returns:
                        raise ValueError(
                            "V2 advisory ledger differs from its unchanged FHS projection"
                        )
    except (KeyError, TypeError, ValueError, ValidationError, StopIteration) as exc:
        raise RefreshVerificationError(
            "product_v2_integrity_failed", "V2 output failed independent source/row verification"
        ) from exc


def load_verified_v2_forecast_sources(
    *, run: AnalysisRun, store: AssetStore
) -> tuple[VerifiedForecastSource, ...]:
    verify_decision_support_output(run=run, store=store)
    graph = _load_graph(
        snapshot=run.universe_snapshot, generation_boundary=run.generated_at, store=store
    )
    artifacts = {
        asset.metadata["listing_id"]: asset
        for asset in DataAsset.objects.filter(
            provider="stanstock",
            kind=CALCULATION_KIND,
            subject=str(run.id),
        )
    }
    return tuple(
        VerifiedForecastSource(
            analysis=StockAnalysis.objects.get(run=run, listing=selection.listing),
            listing=selection.listing,
            prediction_rows=tuple(
                Prediction.objects.filter(analysis__run=run, listing=selection.listing)
            ),
            selection=selection,
            source_execution=selection.product_input.source_execution,
            calculation_artifact=artifacts[str(selection.listing.id)],
        )
        for selection in graph.selections
    )


def decision_support_run_assets(*, run: AnalysisRun, store: AssetStore) -> tuple[DataAsset, ...]:
    """Exact verified run union for later shared orchestration integration."""
    verify_decision_support_output(run=run, store=store)
    graph = _load_graph(
        snapshot=run.universe_snapshot, generation_boundary=run.generated_at, store=store
    )
    assets = {
        asset.id: asset
        for asset in (
            graph.intake.asset,
            graph.membership_asset,
            graph.declaration,
            graph.batch.asset,
            *graph.batch.sources,
        )
    }
    for ref in graph.membership["catalog_assets"]:
        asset = resolve_asset_ref(AssetRef.from_json(ref), cutoff=run.data_cutoff)
        assets[asset.id] = asset
    for selection in graph.selections:
        for asset in (
            selection.stock_asset,
            selection.stock_raw_asset,
            selection.benchmark_asset,
            selection.benchmark_raw_asset,
        ):
            assets[asset.id] = asset
    for asset in DataAsset.objects.filter(
        provider="stanstock",
        subject=str(run.id),
        kind__in=(
            CALCULATION_KIND,
            "analysis_output_manifest",
        ),
    ):
        assets[asset.id] = asset
    from stanstock.research.product_frequency_evidence import verified_registered_frequency_asset

    frequency = verified_registered_frequency_asset(run=run, store=store)
    if frequency is not None:
        assets[frequency.id] = frequency
    return tuple(
        sorted(
            assets.values(),
            key=lambda asset: (asset.provider, asset.kind, asset.subject, str(asset.id)),
        )
    )

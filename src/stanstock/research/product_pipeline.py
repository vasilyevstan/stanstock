"""Prospective research-product source selection, issuance, and replay proof.

This is deliberately a small adapter around the reviewed pure price-product
operator.  It selects immutable assets once, records that exact closure, and
writes the product's five-row ledger without routing scoreless results through
the legacy weighted-analysis writer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Any
from uuid import UUID, uuid4

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import DatabaseError, transaction
from django.utils import timezone

from stanstock.core.revision import clean_git_revision
from stanstock.core.verification_types import AssetRef, RefreshVerificationError
from stanstock.data.asof import AsOfData, raw_price_asset_for
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    read_checksummed_bytes,
    register_asset,
    resolve_asset_ref,
)
from stanstock.data.models import DataAsset, Listing, UniverseMembership, UniverseSnapshot
from stanstock.data.research_product import (
    product_membership_payload,
    verify_product_intake_membership,
    verify_product_listing_catalog,
    verify_product_price_content,
)
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.price_product import (
    PriceInputIdentity,
    PriceProductInput,
    PriceProductInputError,
    PriceProductResult,
    PriceSeries,
    SourceExecutionBinding,
    calculate_price_product,
    complete_input_hash,
    prediction_model_version,
)
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    MOMENTUM_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_VERSION,
    PriceProductConfig,
    load_price_product_config,
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
    build_output_plan,
    dumps_canonical_envelope,
    lookup_manifest,
    model_row_values,
    parse_manifest_envelope,
    row_digest,
)
from stanstock.research.timing import is_us_session_issuance_on_time

CALCULATION_ARTIFACT_KIND = "research_product_calculation"
CALCULATION_ARTIFACT_CONTRACT = "research-product-calculation@1"
SOURCE_WINDOW_CONTRACT = "research-product-source-window@1"
_PRODUCT_PLAN_HORIZONS = frozenset({"6m"})
_FHS_HORIZONS = ("6m", "12m", "3y", "5y")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProductSourceSelection:
    """The selected, never-reselected stock/SPY source closure for one listing."""

    listing: Listing
    stock_asset: DataAsset
    stock_raw_asset: DataAsset
    benchmark_asset: DataAsset
    benchmark_raw_asset: DataAsset
    product_input: PriceProductInput
    catalog_assets: tuple[DataAsset, ...] = ()

    @property
    def source_assets(self) -> list[dict[str, str]]:
        return [
            asset_ref_for(asset).to_json()
            for asset in (
                self.stock_asset,
                self.stock_raw_asset,
                self.benchmark_asset,
                self.benchmark_raw_asset,
            )
        ]


@dataclass(frozen=True, slots=True)
class ProductPersistedAnalysis:
    run: AnalysisRun
    analysis: StockAnalysis
    predictions: tuple[Prediction, ...]
    result: PriceProductResult
    selection: ProductSourceSelection


def verified_source_window_payload(selection: ProductSourceSelection) -> dict[str, Any]:
    """Return the stable replay handoff for an already verified source window.

    Consumers must receive this only from :func:`select_product_source` or
    from the checksummed calculation artifact created by this module.  The
    payload deliberately carries the selected IDs/checksums *and* the exact
    757-session value window: replay code must never substitute a globally
    newest asset or infer dates from a current provider response.
    """

    def series_payload(
        series: PriceSeries, *, normalized: DataAsset, raw: DataAsset
    ) -> dict[str, Any]:
        return {
            "normalized_asset": asset_ref_for(normalized).to_json(),
            "raw_asset": asset_ref_for(raw).to_json(),
            "identity": {
                "asset_id": str(series.identity.asset_id),
                "provider": series.identity.provider,
                "subject": series.identity.subject,
                "sha256": series.identity.sha256,
                "retrieved_at": series.identity.retrieved_at.isoformat(),
                "available_at": series.identity.available_at.isoformat(),
            },
            "currency": series.currency,
            "dates": [value.isoformat() for value in series.dates],
            "closes": list(series.closes),
            "volumes": None if series.volumes is None else list(series.volumes),
            "volume_adjustment_compatible": series.volume_adjustment_compatible,
        }

    product_input = selection.product_input
    return {
        "contract": SOURCE_WINDOW_CONTRACT,
        "listing_id": str(product_input.listing_id),
        "target_date": product_input.target_date.isoformat(),
        "decision_time": product_input.decision_time.isoformat(),
        "calendar_sessions": [value.isoformat() for value in product_input.calendar_sessions],
        "source_execution": {
            "mode": product_input.source_execution.mode,
            "evidence_grade": product_input.source_execution.evidence_grade,
        },
        "stock": series_payload(
            product_input.stock,
            normalized=selection.stock_asset,
            raw=selection.stock_raw_asset,
        ),
        "benchmark": series_payload(
            product_input.benchmark,
            normalized=selection.benchmark_asset,
            raw=selection.benchmark_raw_asset,
        ),
    }


def price_product_input_from_verified_source_window(
    payload: Mapping[str, Any],
) -> PriceProductInput:
    """Deserialize a source-window handoff without selecting any new asset.

    This is the replay-facing companion to :func:`verified_source_window_payload`.
    It is intentionally pure: physical-byte and authoritative-row validation
    remains the writer/verifier's responsibility before handing this payload to
    a replay consumer.  The parser accepts no alternate/missing shape and does
    not query `DataAsset`, so it cannot silently upgrade a historical source.
    """
    expected = {
        "contract",
        "listing_id",
        "target_date",
        "decision_time",
        "calendar_sessions",
        "source_execution",
        "stock",
        "benchmark",
    }
    if set(payload) != expected or payload.get("contract") != SOURCE_WINDOW_CONTRACT:
        raise ValueError("Verified product source-window payload has an invalid shape")
    try:
        source = payload["source_execution"]
        if not isinstance(source, Mapping):
            raise TypeError
        binding = SourceExecutionBinding(
            mode=source["mode"],
            evidence_grade=source["evidence_grade"],
        )
        decision_time = datetime.fromisoformat(str(payload["decision_time"]))
        target_date = date.fromisoformat(str(payload["target_date"]))
        listing_id = UUID(str(payload["listing_id"]))
        sessions_raw = payload["calendar_sessions"]
        if not isinstance(sessions_raw, list):
            raise TypeError
        sessions = tuple(date.fromisoformat(str(value)) for value in sessions_raw)
        stock = _series_from_source_window(payload["stock"])
        benchmark = _series_from_source_window(payload["benchmark"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Verified product source-window payload is malformed") from exc
    if decision_time.tzinfo is None:
        raise ValueError("Verified product source-window decision_time must be timezone-aware")
    return PriceProductInput(
        listing_id=listing_id,
        target_date=target_date,
        decision_time=decision_time,
        calendar_sessions=sessions,
        stock=stock,
        benchmark=benchmark,
        source_execution=binding,
    )


def _series_from_source_window(raw: object) -> PriceSeries:
    if not isinstance(raw, Mapping):
        raise ValueError("Verified product source-window series is malformed")
    expected = {
        "normalized_asset",
        "raw_asset",
        "identity",
        "currency",
        "dates",
        "closes",
        "volumes",
        "volume_adjustment_compatible",
    }
    if (
        set(raw) != expected
        or not isinstance(raw["normalized_asset"], Mapping)
        or not isinstance(raw["identity"], Mapping)
    ):
        raise ValueError("Verified product source-window series has an invalid shape")
    identity_raw = raw["identity"]
    try:
        normalized_ref = raw["normalized_asset"]
        if str(normalized_ref["id"]) != str(identity_raw["asset_id"]) or any(
            str(normalized_ref[key]) != str(identity_raw[key])
            for key in ("provider", "subject", "sha256")
        ):
            raise ValueError
        identity = PriceInputIdentity(
            asset_id=UUID(str(identity_raw["asset_id"])),
            provider=str(identity_raw["provider"]),
            subject=str(identity_raw["subject"]),
            sha256=str(identity_raw["sha256"]),
            retrieved_at=datetime.fromisoformat(str(identity_raw["retrieved_at"])),
            available_at=datetime.fromisoformat(str(identity_raw["available_at"])),
        )
        dates_raw, closes_raw = raw["dates"], raw["closes"]
        if not isinstance(dates_raw, list) or not isinstance(closes_raw, list):
            raise TypeError
        if not isinstance(raw["volume_adjustment_compatible"], bool) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in closes_raw
        ):
            raise TypeError
        volumes_raw = raw["volumes"]
        if volumes_raw is not None and not isinstance(volumes_raw, list):
            raise TypeError
        return PriceSeries(
            identity=identity,
            currency=str(raw["currency"]),
            dates=tuple(date.fromisoformat(str(value)) for value in dates_raw),
            closes=tuple(float(value) for value in closes_raw),
            volumes=(
                None
                if volumes_raw is None
                else tuple(None if value is None else float(value) for value in volumes_raw)
            ),
            volume_adjustment_compatible=raw["volume_adjustment_compatible"] is True,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Verified product source-window series is malformed") from exc


def select_product_source(
    *,
    listing: Listing,
    target_date: date,
    decision_time: datetime,
    provider: str,
    benchmark_subject: str,
    source_execution: SourceExecutionBinding,
    store: AssetStore,
    config: PriceProductConfig,
    stock_asset: DataAsset | None = None,
    benchmark_asset: DataAsset | None = None,
    catalog_assets: tuple[DataAsset, ...] = (),
) -> ProductSourceSelection:
    """Select and prove one exact qualified source closure before calculation.

    A short current-price asset never shadows a qualified full history: assets
    are considered newest-first, but only a source with all 757 exact XNYS
    closes is selected.  A newer *qualified* correction wins and is then
    retained by UUID/checksum in the result rather than selected again later.
    """
    if decision_time.tzinfo is None:
        raise ValueError("Product decision_time must be timezone-aware")
    if (
        listing.currency != config.currency
        or not listing.provider_symbol
        or listing.region != "us"
        or listing.security.security_type not in {"common_stock", "adr"}
        or benchmark_subject != config.benchmark_subject
    ):
        raise PriceProductInputError(
            "listing_identity_invalid",
            "Listing does not have a verified USD provider-symbol identity",
        )
    if source_execution.mode == "synthetic_demo" and source_execution.evidence_grade != "research":
        raise PriceProductInputError(
            "synthetic_observed_forbidden",
            "Synthetic demo sources may only issue research-grade output",
        )
    asof = AsOfData(decision_time, store)
    stock = stock_asset or select_product_price_asset(
        asof=asof,
        subject=listing.provider_symbol,
        target_date=target_date,
        config=config,
        store=store,
        provider=provider,
    )
    benchmark = benchmark_asset or select_product_price_asset(
        asof=asof,
        subject=benchmark_subject,
        target_date=target_date,
        config=config,
        store=store,
        provider=provider,
    )
    _require_actual_source_mode(
        assets=(stock, benchmark),
        provider=provider,
        source_execution=source_execution,
    )
    stock_raw = _validate_price_asset(
        asof=asof,
        asset=stock,
        subject=listing.provider_symbol,
        target_date=target_date,
        config=config,
        store=store,
    )
    benchmark_raw = _validate_price_asset(
        asof=asof,
        asset=benchmark,
        subject=benchmark_subject,
        target_date=target_date,
        config=config,
        store=store,
    )
    if source_execution.mode == "provider":
        verify_product_listing_catalog(
            listing=listing,
            stock_asset=stock,
            benchmark_asset=benchmark,
            catalog_assets=catalog_assets,
            cutoff=decision_time,
            store=store,
        )
    stock_series = _series_for_asset(asof=asof, asset=stock, target_date=target_date, config=config)
    benchmark_series = _series_for_asset(
        asof=asof, asset=benchmark, target_date=target_date, config=config
    )
    return ProductSourceSelection(
        listing=listing,
        stock_asset=stock,
        stock_raw_asset=stock_raw,
        benchmark_asset=benchmark,
        benchmark_raw_asset=benchmark_raw,
        catalog_assets=catalog_assets,
        product_input=PriceProductInput(
            listing_id=listing.id,
            target_date=target_date,
            decision_time=decision_time,
            calendar_sessions=_required_sessions(target_date, config.required_closes),
            stock=stock_series,
            benchmark=benchmark_series,
            source_execution=source_execution,
        ),
    )


def issue_price_product_snapshot(
    *,
    universe_snapshot: UniverseSnapshot,
    decision_time: datetime,
    target_date: date,
    provider: str,
    benchmark_subject: str,
    source_execution: SourceExecutionBinding,
    store: AssetStore,
    code_revision: str,
    issued_on_time: bool,
    config: PriceProductConfig | None = None,
    output_paths: Any | None = None,
) -> list[ProductPersistedAnalysis]:
    """Calculate and append the exact five product rows for each qualified member.

    This function performs all source reads and numerical work before opening
    its write transaction.  It therefore never holds a database transaction
    over filesystem/provider-adjacent work, and computes the complete output
    plan independently from the rows subsequently written.
    """
    config = config or load_price_product_config()
    _validate_product_admission(
        snapshot=universe_snapshot,
        decision_time=decision_time,
        target_date=target_date,
        provider=provider,
        benchmark_subject=benchmark_subject,
        source_execution=source_execution,
        code_revision=code_revision,
        issued_on_time=issued_on_time,
        config=config,
    )
    membership_authority = product_membership_payload(universe_snapshot, store=store)
    memberships = list(
        UniverseMembership.objects.select_related("listing__security")
        .filter(snapshot=universe_snapshot, eligible=True)
        .order_by("listing_id")
    )
    if not memberships:
        raise ValueError("Product issuance requires an independently captured qualified membership")
    _require_membership_authority(
        snapshot=universe_snapshot,
        memberships=memberships,
        payload=membership_authority,
        store=store,
        provider=provider,
    )
    source_time = datetime.fromisoformat(str(membership_authority["decision_time"]))
    if source_time.tzinfo is None or source_time > decision_time:
        raise ValueError("Captured source boundary is after generation")
    admissions = membership_authority.get("admissions")
    benchmark_asset = None
    catalog_assets: tuple[DataAsset, ...] = ()
    stock_assets: dict[UUID, DataAsset] = {}
    if source_execution.mode == "provider":
        if not isinstance(admissions, dict):
            raise ValueError("Provider issuance requires captured admission evidence")
        benchmark_asset = resolve_asset_ref(
            AssetRef.from_json(membership_authority["benchmark_asset"]), cutoff=source_time
        )
        catalog_assets = _catalogs_for_membership(membership_authority, source_time)
        for entry in admissions.values():
            if entry["status"] == "admitted":
                stock_assets[UUID(entry["listing_id"])] = resolve_asset_ref(
                    AssetRef.from_json(entry["price_asset"]), cutoff=source_time
                )
        if set(stock_assets) != {membership.listing_id for membership in memberships}:
            raise ValueError("Captured source assets do not cover the admitted output plan")
    selections = [
        select_product_source(
            listing=membership.listing,
            target_date=target_date,
            decision_time=source_time,
            provider=provider,
            benchmark_subject=benchmark_subject,
            source_execution=source_execution,
            store=store,
            config=config,
            stock_asset=stock_assets.get(membership.listing_id),
            benchmark_asset=benchmark_asset,
            catalog_assets=catalog_assets,
        )
        for membership in memberships
    ]
    results = [
        calculate_price_product(
            selection.product_input,
            config=config,
            effective_config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
        )
        for selection in selections
    ]
    plan = build_output_plan(
        eligible_listing_ids={selection.listing.id for selection in selections},
        decision_horizons=_PRODUCT_PLAN_HORIZONS,
        medium_active=True,
        long_active=True,
    )
    run_id = uuid4()
    manifest_path: str | None = None
    artifact_paths: list[str] = []
    try:
        with transaction.atomic():
            if issued_on_time:
                _require_observed_commit_deadline(
                    target_date=target_date,
                    code_revision=code_revision,
                )
            run = AnalysisRun.objects.create(
                id=run_id,
                generated_at=decision_time,
                data_cutoff=_product_logical_cutoff(
                    generated_at=decision_time,
                    source_time=source_time,
                    target_date=target_date,
                    issued_on_time=issued_on_time,
                ),
                target_date=target_date,
                issued_on_time=issued_on_time,
                universe_snapshot=universe_snapshot,
                config_version=PRODUCT_VERSION,
                config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
                code_revision=code_revision,
            )
            persisted: list[ProductPersistedAnalysis] = []
            for selection, result in zip(selections, results, strict=True):
                artifact, artifact_path = _write_calculation_artifact(
                    run=run,
                    selection=selection,
                    result=result,
                    store=store,
                    generated_at=decision_time,
                )
                artifact_paths.append(artifact_path)
                analysis = _create_product_analysis(
                    run=run, selection=selection, result=result, artifact=artifact
                )
                predictions = _create_product_predictions(
                    analysis=analysis,
                    selection=selection,
                    result=result,
                    artifact=artifact,
                    code_revision=code_revision,
                )
                persisted.append(
                    ProductPersistedAnalysis(
                        run=run,
                        analysis=analysis,
                        predictions=predictions,
                        result=result,
                        selection=selection,
                    )
                )
            actual = actual_output_plan(
                (item.analysis for item in persisted),
                (prediction for item in persisted for prediction in item.predictions),
            )
            if actual != plan:
                raise ValueError("Product output does not match its pre-write five-row plan")
            manifest_path = _write_product_manifest(
                run=run, persisted=persisted, plan=plan, store=store
            )
            # Replay once at issuance.  Serving consumers can rely on this
            # durable proof and checksum checks without running Monte Carlo.
            verify_price_product_output(run=run, store=store, replay=True)
            if issued_on_time:
                _require_observed_commit_deadline(
                    target_date=target_date,
                    code_revision=code_revision,
                )
            if output_paths is not None:
                output_paths.manifest_relative_path = manifest_path
        return persisted
    except Exception:
        for path in [manifest_path, *artifact_paths]:
            if path is not None:
                _safe_unlink(store, path)
        if output_paths is not None:
            output_paths.manifest_relative_path = None
        raise


def verify_price_product_output(
    *,
    run: AnalysisRun,
    store: AssetStore,
    replay: bool = False,
) -> None:
    """Fail closed against independently registered sources and output manifest."""
    if run.config_version != PRODUCT_VERSION or run.config_hash != PRODUCT_EFFECTIVE_CONFIG_HASH:
        raise RefreshVerificationError(
            "product_config_invalid", "Research product config identity is invalid"
        )
    lookup = lookup_manifest(run.id)
    if lookup.count != 1 or lookup.asset is None:
        raise RefreshVerificationError(
            "product_manifest_missing", "Research product manifest is missing or ambiguous"
        )
    manifest_asset = lookup.asset
    read_checksummed_bytes(store, manifest_asset)
    try:
        manifest_run_id, plan, entries = parse_manifest_envelope(
            read_checksummed_bytes(store, manifest_asset)
        )
    except ValueError as exc:
        raise RefreshVerificationError(
            "product_manifest_malformed", "Research product manifest is malformed"
        ) from exc
    analyses = list(StockAnalysis.objects.filter(run=run).order_by("listing_id"))
    predictions = list(Prediction.objects.filter(analysis__run=run).order_by("id"))
    providers = {prediction.price_provider for prediction in predictions}
    if len(providers) != 1:
        raise RefreshVerificationError(
            "product_source_provider_invalid", "Product output has missing or mixed providers"
        )
    membership_authority = product_membership_payload(run.universe_snapshot, store=store)
    try:
        _require_membership_authority(
            snapshot=run.universe_snapshot,
            memberships=list(
                UniverseMembership.objects.filter(snapshot=run.universe_snapshot, eligible=True)
            ),
            payload=membership_authority,
            store=store,
            provider=next(iter(providers)),
        )
        membership_ids = _membership_ids_from_payload(membership_authority)
    except ValueError as exc:
        raise RefreshVerificationError(
            "product_membership_authority_invalid",
            "Research product membership authority is invalid",
        ) from exc
    expected = build_output_plan(
        eligible_listing_ids=membership_ids,
        decision_horizons=_PRODUCT_PLAN_HORIZONS,
        medium_active=True,
        long_active=True,
    )
    if (
        manifest_run_id != str(run.id)
        or plan != expected
        or actual_output_plan(analyses, predictions) != expected
    ):
        raise RefreshVerificationError(
            "product_output_plan_invalid", "Research product output plan diverged"
        )
    current = {
        (ANALYSIS_RUN_MODEL, str(run.id)): row_digest(
            ANALYSIS_RUN_MODEL, model_row_values(run, ANALYSIS_RUN_FIELDS)
        )
    }
    for analysis in analyses:
        current[(STOCK_ANALYSIS_MODEL, str(analysis.id))] = row_digest(
            STOCK_ANALYSIS_MODEL, model_row_values(analysis, STOCK_ANALYSIS_FIELDS)
        )
    for prediction in predictions:
        current[(PREDICTION_MODEL, str(prediction.id))] = row_digest(
            PREDICTION_MODEL, model_row_values(prediction, PREDICTION_FIELDS)
        )
    if {(entry.model, entry.row_id): entry.digest for entry in entries} != current:
        raise RefreshVerificationError(
            "product_manifest_diverged", "Research product rows diverged from manifest"
        )
    by_listing = {analysis.listing_id: analysis for analysis in analyses}
    if len(by_listing) != len(analyses):
        raise RefreshVerificationError(
            "product_analysis_duplicate", "Research product has duplicate listing analyses"
        )
    for _listing_id, analysis in by_listing.items():
        rows = [prediction for prediction in predictions if prediction.analysis_id == analysis.id]
        _verify_product_rows(analysis=analysis, rows=rows, store=store, replay=replay)


def _require_membership_authority(
    *,
    snapshot: UniverseSnapshot,
    memberships: list[UniverseMembership],
    payload: Mapping[str, object],
    store: AssetStore,
    provider: str,
) -> None:
    if set(payload) != {
        "contract",
        "intake",
        "qualified_listing_ids",
        "candidate_states",
        "admissions",
        "catalog_assets",
        "benchmark_asset",
        "decision_time",
        "snapshot_id",
    }:
        raise ValueError("Product membership evidence has an invalid shape")
    source_time = datetime.fromisoformat(str(payload["decision_time"]))
    if source_time.tzinfo is None:
        raise ValueError("Product source boundary has no timezone")
    verify_product_intake_membership(
        snapshot=snapshot, payload=payload, cutoff=source_time, store=store, provider=provider
    )
    qualified = _membership_ids_from_payload(payload)
    actual = {membership.listing_id for membership in memberships}
    if actual != qualified:
        raise ValueError("Product membership rows diverge from captured admission authority")
    binding = payload.get("intake")
    if not isinstance(binding, Mapping) or not all(
        isinstance(binding.get(key), str) and binding[key] for key in ("id", "sha256", "subject")
    ):
        raise ValueError("Product membership evidence has no intake binding")
    base = {key: value for key, value in payload.items() if key != "snapshot_id"}
    encoded = json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != snapshot.config_hash:
        raise ValueError("Product membership evidence does not reproduce snapshot identity")


def _membership_ids_from_payload(payload: Mapping[str, object]) -> set[UUID]:
    raw_ids = payload.get("qualified_listing_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("Product membership evidence has invalid qualified listings")
    try:
        ids = {UUID(str(item)) for item in raw_ids}
    except (TypeError, ValueError) as exc:
        raise ValueError("Product membership evidence has invalid qualified listings") from exc
    if len(ids) != len(raw_ids):
        raise ValueError("Product membership evidence has duplicate qualified listings")
    return ids


def _verify_product_rows(
    *, analysis: StockAnalysis, rows: list[Prediction], store: AssetStore, replay: bool
) -> None:
    keys = {(row.method_version, row.evidence_role, row.horizon) for row in rows}
    expected = {
        (MOMENTUM_METHOD_VERSION, Prediction.EvidenceRole.DECISION, "6m"),
        *{
            (FHS_METHOD_VERSION, Prediction.EvidenceRole.ADVISORY, horizon)
            for horizon in _FHS_HORIZONS
        },
    }
    if keys != expected or len(rows) != 5:
        raise RefreshVerificationError(
            "product_prediction_multiset_invalid",
            "Research product does not have five expected rows",
        )
    artifacts = {
        json.dumps(row.calculation.get("calculation_artifact"), sort_keys=True)
        for row in rows
        if isinstance(row.calculation, dict)
    }
    if len(artifacts) != 1:
        raise RefreshVerificationError(
            "product_calculation_binding_invalid",
            "Product rows do not share one calculation artifact",
        )
    try:
        ref = json.loads(next(iter(artifacts)))
        artifact_id = UUID(str(ref["id"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RefreshVerificationError(
            "product_calculation_binding_invalid",
            "Product calculation artifact reference is invalid",
        ) from exc
    artifact = DataAsset.objects.filter(
        id=artifact_id, kind=CALCULATION_ARTIFACT_KIND, provider="stanstock"
    ).first()
    if artifact is None:
        raise RefreshVerificationError(
            "product_calculation_missing", "Product calculation artifact is missing"
        )
    if (
        artifact.subject != str(analysis.run_id)
        or artifact.metadata.get("listing_id") != str(analysis.listing_id)
        or asset_ref_for(artifact).to_json() != ref
    ):
        raise RefreshVerificationError(
            "product_calculation_identity_invalid",
            "Product calculation artifact registry identity mismatches output",
        )
    matching_artifacts = [
        item
        for item in DataAsset.objects.filter(
            provider="stanstock",
            kind=CALCULATION_ARTIFACT_KIND,
            subject=str(analysis.run_id),
        )
        if item.metadata.get("listing_id") == str(analysis.listing_id)
    ]
    if len(matching_artifacts) != 1:
        raise RefreshVerificationError(
            "product_calculation_ambiguous",
            "Product calculation artifact registry has no unique listing binding",
        )
    try:
        document = json.loads(read_checksummed_bytes(store, artifact))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RefreshVerificationError(
            "product_calculation_malformed", "Product calculation artifact is malformed"
        ) from exc
    if not isinstance(document, dict) or document.get("contract") != CALCULATION_ARTIFACT_CONTRACT:
        raise RefreshVerificationError(
            "product_calculation_malformed", "Product calculation artifact has invalid identity"
        )
    if document.get("listing_id") != str(analysis.listing_id) or document.get("run_id") != str(
        analysis.run_id
    ):
        raise RefreshVerificationError(
            "product_calculation_identity_invalid",
            "Product calculation artifact identity mismatches output",
        )
    if document.get("run_digest") != row_digest(
        ANALYSIS_RUN_MODEL, model_row_values(analysis.run, ANALYSIS_RUN_FIELDS)
    ):
        raise RefreshVerificationError(
            "product_calculation_run_mismatch",
            "Product run metadata diverges from its registered calculation artifact",
        )
    selection = _selection_from_artifact(analysis=analysis, document=document, store=store)
    source_window = document.get("source_window")
    if not isinstance(source_window, Mapping):
        raise RefreshVerificationError(
            "product_source_window_invalid", "Product calculation has no verified source window"
        )
    try:
        window_input = price_product_input_from_verified_source_window(source_window)
    except ValueError as exc:
        raise RefreshVerificationError(
            "product_source_window_invalid", "Product calculation source window is malformed"
        ) from exc
    if (
        window_input != selection.product_input
        or document.get("input_hash") != complete_input_hash(selection.product_input)
        or document.get("source_assets") != selection.source_assets
    ):
        raise RefreshVerificationError(
            "product_source_window_diverged",
            "Product calculation source window diverges from registered source evidence",
        )
    _require_row_calculation_bindings(
        analysis=analysis,
        rows=rows,
        artifact_ref=ref,
        selection=selection,
        input_hash=str(document.get("input_hash") or ""),
    )
    _require_rows_match_recorded_result(
        analysis=analysis,
        rows=rows,
        recorded_result=document.get("result"),
    )
    if replay:
        result = calculate_price_product(
            selection.product_input,
            config=load_price_product_config(),
            effective_config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
        )
        if document.get("input_hash") != result.input_hash:
            raise RefreshVerificationError(
                "product_calculation_replay_mismatch", "Product calculation input replay differs"
            )
        if document.get("result") != _jsonable(asdict(result)):
            raise RefreshVerificationError(
                "product_calculation_replay_mismatch",
                "The complete recorded product result differs from numerical replay",
            )
        _compare_replayed_rows(rows, result)


def _selection_from_artifact(
    *, analysis: StockAnalysis, document: dict[str, Any], store: AssetStore
) -> ProductSourceSelection:
    refs = document.get("source_assets")
    if not isinstance(refs, list) or len(refs) != 4:
        raise RefreshVerificationError(
            "product_source_closure_invalid", "Product source closure is invalid"
        )
    assets: list[DataAsset] = []
    for raw in refs:
        if not isinstance(raw, dict):
            raise RefreshVerificationError(
                "product_source_closure_invalid", "Product source closure is invalid"
            )
        try:
            asset_id = UUID(str(raw.get("id")))
        except (TypeError, ValueError) as exc:
            raise RefreshVerificationError(
                "product_source_closure_invalid", "Product source asset identity is invalid"
            ) from exc
        asset = DataAsset.objects.filter(
            id=asset_id,
            provider=raw.get("provider"),
            kind=raw.get("kind"),
            subject=raw.get("subject"),
            sha256=raw.get("sha256"),
        ).first()
        if asset is None:
            raise RefreshVerificationError(
                "product_source_closure_invalid", "Product source asset is not registered"
            )
        read_checksummed_bytes(store, asset)
        assets.append(asset)
    source = document.get("source_execution")
    if not isinstance(source, dict):
        raise RefreshVerificationError(
            "product_source_mode_invalid", "Product source mode is invalid"
        )
    try:
        binding = SourceExecutionBinding(
            mode=source["mode"], evidence_grade=source["evidence_grade"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "product_source_mode_invalid", "Product source mode is invalid"
        ) from exc
    membership = product_membership_payload(analysis.run.universe_snapshot, store=store)
    source_time = datetime.fromisoformat(str(membership["decision_time"]))
    if source_time > analysis.run.generated_at or (
        analysis.run.issued_on_time and source_time != analysis.run.data_cutoff
    ):
        raise RefreshVerificationError(
            "product_source_time_invalid", "Product source boundary conflicts with its run"
        )
    catalog_assets: tuple[DataAsset, ...] = ()
    if binding.mode == "provider":
        admissions = membership["admissions"]
        if not isinstance(admissions, dict):
            raise ValueError("Provider output has no captured admission sources")
        entry = admissions.get(analysis.listing.provider_symbol)
        if (
            not isinstance(entry, dict)
            or entry.get("listing_id") != str(analysis.listing_id)
            or entry.get("status") != "admitted"
            or entry.get("price_asset") != asset_ref_for(assets[0]).to_json()
            or membership["benchmark_asset"] != asset_ref_for(assets[2]).to_json()
        ):
            raise ValueError("Product calculation substituted captured source authority")
        catalog_assets = _catalogs_for_membership(membership, source_time)
    if document.get("catalog_assets") != [
        asset_ref_for(asset).to_json() for asset in catalog_assets
    ]:
        raise ValueError("Product calculation substituted its registered catalog authority")
    return select_product_source(
        listing=analysis.listing,
        target_date=analysis.run.target_date,
        decision_time=source_time,
        provider=assets[0].provider,
        benchmark_subject=assets[2].subject,
        source_execution=binding,
        store=store,
        config=load_price_product_config(),
        stock_asset=assets[0],
        benchmark_asset=assets[2],
        catalog_assets=catalog_assets,
    )


def _require_row_calculation_bindings(
    *,
    analysis: StockAnalysis,
    rows: list[Prediction],
    artifact_ref: dict[str, object],
    selection: ProductSourceSelection,
    input_hash: str,
) -> None:
    raw_current = Decimal(str(selection.product_input.stock.closes[-1]))
    previous = Decimal(str(selection.product_input.stock.closes[-2]))
    if analysis.current_price != raw_current.quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_EVEN
    ) or analysis.daily_change != (raw_current / previous - 1).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_EVEN
    ):
        raise RefreshVerificationError(
            "product_analysis_price_invalid",
            "Product analysis prices diverge from their independently registered source",
        )
    quality = analysis.data_quality
    if (
        not isinstance(quality, dict)
        or quality.get("calculation_artifact") != artifact_ref
        or quality.get("source_assets") != selection.source_assets
        or quality.get("input_hash") != input_hash
    ):
        raise RefreshVerificationError(
            "product_analysis_source_binding_invalid",
            "Product analysis does not bind its registered calculation and source closure",
        )
    for row in rows:
        calculation = row.calculation
        if (
            not isinstance(calculation, dict)
            or calculation.get("schema") != "research-product@1"
            or calculation.get("method_version") != row.method_version
            or calculation.get("horizon") != row.horizon
            or calculation.get("calculation_artifact") != artifact_ref
            or calculation.get("input_hash") != input_hash
            or calculation.get("source_execution")
            != {
                "mode": selection.product_input.source_execution.mode,
                "evidence_grade": selection.product_input.source_execution.evidence_grade,
            }
            or row.source_assets != selection.source_assets
            or row.price_provider != selection.stock_asset.provider
            or row.price_subject != selection.stock_asset.subject
            or row.evidence_grade != selection.product_input.source_execution.evidence_grade
            or row.listing_id != analysis.listing_id
            or row.generated_at != analysis.run.generated_at
            or row.target_date != analysis.run.target_date
            or row.data_cutoff != analysis.run.data_cutoff
            or row.issued_on_time != analysis.run.issued_on_time
            or row.config_hash != PRODUCT_EFFECTIVE_CONFIG_HASH
            or row.code_revision != analysis.run.code_revision
            or row.price_at_prediction != analysis.current_price
            or row.model_version
            != prediction_model_version(method_version=row.method_version, issuance_id=row.id)
            or row.probability_positive is not None
            or row.confidence is not None
            or row.confidence_status != "not_estimated"
            or row.overall_score is not None
            or (row.method_version == FHS_METHOD_VERSION and row.recommendation is not None)
            or row.source_mode
            != (
                Prediction.SourceMode.SYNTHETIC
                if selection.product_input.source_execution.mode == "synthetic_demo"
                else Prediction.SourceMode.PROVIDER
            )
        ):
            raise RefreshVerificationError(
                "product_prediction_source_binding_invalid",
                "Product prediction does not bind its registered calculation and source closure",
            )


def _catalogs_for_membership(
    payload: Mapping[str, object], source_time: datetime
) -> tuple[DataAsset, ...]:
    raw_refs = payload.get("catalog_assets")
    if not isinstance(raw_refs, list):
        raise ValueError("Product membership has no catalog reference list")
    return tuple(resolve_asset_ref(AssetRef.from_json(ref), cutoff=source_time) for ref in raw_refs)


def _recorded_decimal_matches_row(actual: Decimal | None, recorded: object) -> bool:
    """Match a finite ledger Decimal to its exact recorded artifact text.

    SQLite re-queries a numeric signed zero as unsigned while calculation
    artifacts preserve their original Decimal text. Only that same-scale zero
    sign representation may differ; every nonzero value still requires an
    exact finite text match.
    """
    if not isinstance(actual, Decimal) or not actual.is_finite():
        return False
    if not isinstance(recorded, str):
        return False
    if str(actual) == recorded:
        return True
    return actual.is_zero() and str(actual.copy_negate()) == recorded


def _require_rows_match_recorded_result(
    *,
    analysis: StockAnalysis,
    rows: list[Prediction],
    recorded_result: object,
) -> None:
    """Compare persisted rows with the calculation artifact without replaying paths."""
    if not isinstance(recorded_result, Mapping):
        raise RefreshVerificationError(
            "product_recorded_result_invalid",
            "Product calculation artifact has no recorded result",
        )
    recommendation = recorded_result.get("recommendation")
    risk = recorded_result.get("risk")
    momentum = recorded_result.get("momentum")
    forecast = recorded_result.get("forecast")
    if (
        not isinstance(recommendation, Mapping)
        or not isinstance(risk, Mapping)
        or not isinstance(forecast, Mapping)
        or analysis.recommendation != recommendation.get("suggestion")
        or analysis.risk_class != risk.get("relative_volatility_label")
        or analysis.component_scores.get("momentum") != momentum
        or analysis.overall_score is not None
        or analysis.confidence is not None
        or analysis.confidence_status != "not_estimated"
        or analysis.risk_score is not None
        or analysis.reasons != recommendation.get("blocking_reasons")
        or analysis.risks != risk.get("insufficiency_reasons")
    ):
        raise RefreshVerificationError(
            "product_analysis_result_mismatch",
            "Product analysis differs from its recorded calculation result",
        )
    projections = forecast.get("projections")
    if not isinstance(projections, list) or len(projections) != 4:
        raise RefreshVerificationError(
            "product_recorded_result_invalid",
            "Product calculation artifact has invalid FHS projections",
        )
    projection_by_horizon = {
        item.get("horizon"): item for item in projections if isinstance(item, Mapping)
    }
    if set(projection_by_horizon) != set(_FHS_HORIZONS) or analysis.forecast_scenarios != {
        "schema": "research-product@1",
        "projections": projections,
    }:
        raise RefreshVerificationError(
            "product_analysis_result_mismatch",
            "Product analysis projections differ from the registered calculation",
        )
    for row in rows:
        calculation = row.calculation
        if not isinstance(calculation, dict) or (
            calculation.get("momentum") != momentum
            or calculation.get("payload_schema") != recorded_result.get("payload_schema")
            or calculation.get("risk") != risk
            or calculation.get("recommendation") != recommendation
            or calculation.get("forecast") != forecast
        ):
            raise RefreshVerificationError(
                "product_prediction_result_mismatch",
                "Product prediction calculation differs from its registered artifact",
            )
        if row.method_version == MOMENTUM_METHOD_VERSION:
            if (
                row.recommendation != recommendation.get("suggestion")
                or row.insufficiency_reason
                != (recorded_result.get("momentum_insufficiency_reason") or "")
                or any(
                    value is not None
                    for value in (row.bear_return, row.base_return, row.bull_return)
                )
            ):
                raise RefreshVerificationError(
                    "product_prediction_result_mismatch",
                    "Momentum prediction differs from its registered artifact",
                )
            continue
        projection = projection_by_horizon.get(row.horizon)
        if not isinstance(projection, Mapping):
            raise RefreshVerificationError(
                "product_prediction_result_mismatch",
                "FHS prediction horizon is absent from registered artifact",
            )
        ledger = projection.get("ledger_returns")
        reason = projection.get("insufficiency_reason")
        if ledger is None:
            valid = (
                row.bear_return is None
                and row.base_return is None
                and row.bull_return is None
                and isinstance(reason, str)
                and bool(reason)
                and row.insufficiency_reason == reason
            )
        elif isinstance(ledger, Mapping):
            valid = (
                _recorded_decimal_matches_row(row.bear_return, ledger.get("lower"))
                and _recorded_decimal_matches_row(row.base_return, ledger.get("median"))
                and _recorded_decimal_matches_row(row.bull_return, ledger.get("upper"))
                and row.insufficiency_reason == (reason or "")
            )
        else:
            valid = False
        if not valid:
            raise RefreshVerificationError(
                "product_prediction_result_mismatch",
                "FHS prediction differs from its registered artifact",
            )


def _compare_replayed_rows(rows: list[Prediction], result: PriceProductResult) -> None:
    by_key = {(row.method_version, row.horizon): row for row in rows}
    decision = by_key[(MOMENTUM_METHOD_VERSION, "6m")]
    if (
        decision.recommendation != result.recommendation.suggestion
        or decision.bear_return is not None
    ):
        raise RefreshVerificationError(
            "product_calculation_replay_mismatch", "Momentum output differs from replay"
        )
    for projection in result.forecast.projections:
        row = by_key[(FHS_METHOD_VERSION, projection.horizon)]
        expected = projection.ledger_returns
        if expected is None:
            if (
                row.bear_return is not None
                or row.base_return is not None
                or row.bull_return is not None
                or not projection.insufficiency_reason
                or row.insufficiency_reason != projection.insufficiency_reason
            ):
                raise RefreshVerificationError(
                    "product_calculation_replay_mismatch",
                    "Withheld FHS output differs from replay",
                )
            continue
        if (
            row.bear_return != expected.lower
            or row.base_return != expected.median
            or row.bull_return != expected.upper
            or row.probability_positive is not None
        ):
            raise RefreshVerificationError(
                "product_calculation_replay_mismatch", "FHS output differs from replay"
            )


def _validate_product_admission(
    *,
    snapshot: UniverseSnapshot,
    decision_time: datetime,
    target_date: date,
    provider: str,
    benchmark_subject: str,
    source_execution: SourceExecutionBinding,
    code_revision: str,
    issued_on_time: bool,
    config: PriceProductConfig,
) -> None:
    if not isinstance(issued_on_time, bool):
        raise ValueError("Product issued_on_time must be a boolean")
    if snapshot.as_of_date != target_date:
        raise ValueError("Product snapshot target does not match issuance target")
    if not code_revision:
        raise ValueError("Product issuance requires an explicit code revision")
    if source_execution.evidence_grade != snapshot.grade:
        raise ValueError("Product source grade must match captured snapshot grade")
    if source_execution.evidence_grade == "observed" and not issued_on_time:
        raise ValueError("A non-observed product issuance requires a research-grade snapshot")
    if issued_on_time:
        if (
            snapshot.grade != UniverseSnapshot.Grade.OBSERVED.value
            or source_execution.mode != "provider"
            or provider != config.price_provider
            or benchmark_subject != config.benchmark_subject
            or config.product_version != PRODUCT_VERSION
            or config != load_price_product_config()
            or not is_us_session_issuance_on_time(
                target_date=target_date, generated_at=decision_time
            )
        ):
            raise ValueError("Observed product issuance was not independently admitted")
        _require_observed_commit_deadline(target_date=target_date, code_revision=code_revision)
    elif (
        snapshot.grade == UniverseSnapshot.Grade.OBSERVED.value
        and source_execution.evidence_grade != "observed"
    ):
        raise ValueError("Observed snapshot cannot be silently downgraded for product issuance")


def _require_observed_commit_deadline(*, target_date: date, code_revision: str) -> None:
    from pathlib import Path

    from stanstock.research.config import code_revision as configured_code_revision

    if code_revision != configured_code_revision() or code_revision != clean_git_revision(
        Path(settings.BASE_DIR)
    ):
        raise ValueError("Observed product issuance requires the exact clean committed revision")
    if not is_us_session_issuance_on_time(target_date=target_date, generated_at=timezone.now()):
        raise ValueError("Observed product issuance deadline has passed")


def _product_logical_cutoff(
    *, generated_at: datetime, source_time: datetime, target_date: date, issued_on_time: bool
) -> datetime:
    from stanstock.research.service import _analysis_data_cutoff

    if issued_on_time:
        return source_time
    return _analysis_data_cutoff(generated_at.astimezone(UTC), target_date, issued_on_time=False)


def select_product_price_asset(
    *,
    asof: AsOfData,
    subject: str,
    target_date: date,
    config: PriceProductConfig,
    store: AssetStore,
    provider: str,
) -> DataAsset:
    candidates = DataAsset.objects.filter(
        provider=provider,
        kind="price_history",
        subject=subject,
        available_at__lte=asof.decision_time,
        retrieved_at__lte=asof.decision_time,
    ).order_by("-available_at", "-retrieved_at", "-id")
    incomplete_newer: list[DataAsset] = []
    for asset in candidates:
        try:
            _validate_price_asset(
                asof=asof,
                asset=asset,
                subject=subject,
                target_date=target_date,
                config=config,
                store=store,
            )
            _series_for_asset(asof=asof, asset=asset, target_date=target_date, config=config)
        except PriceProductInputError as exc:
            if exc.reason_code not in {"price_sessions_incomplete", "price_target_missing"}:
                raise
            incomplete_newer.append(asset)
            continue
        except (RefreshVerificationError, ValueError):
            raise
        for newer in incomplete_newer:
            _require_no_conflicting_overlap(
                asof=asof,
                newer=newer,
                selected=asset,
                target_date=target_date,
            )
        return asset
    raise PriceProductInputError(
        "price_history_insufficient",
        f"No registered price vintage contains {config.required_closes} exact XNYS closes",
    )


def _require_no_conflicting_overlap(
    *,
    asof: AsOfData,
    newer: DataAsset,
    selected: DataAsset,
    target_date: date,
) -> None:
    """A newer partial correction cannot be silently bypassed by old history."""
    newer_frame = asof.price_frame_for_asset_with_diagnostics(
        asset=newer, through_date=target_date
    ).frame
    selected_frame = asof.price_frame_for_asset_with_diagnostics(
        asset=selected, through_date=target_date
    ).frame
    if "close" not in newer_frame.columns or "close" not in selected_frame.columns:
        raise PriceProductInputError(
            "newer_price_correction_invalid",
            "Newer price evidence cannot be compared with selected history",
        )
    old = {
        row["date"]: float(row["close"])
        for row in selected_frame.select("date", "close").iter_rows(named=True)
    }
    for row in newer_frame.select("date", "close").iter_rows(named=True):
        close = float(row["close"])
        prior = old.get(row["date"])
        if prior is not None and (not math.isfinite(close) or close != prior):
            raise PriceProductInputError(
                "newer_price_correction_conflicts",
                "Newer partial price evidence conflicts with reusable history",
            )


def _validate_price_asset(
    *,
    asof: AsOfData,
    asset: DataAsset,
    subject: str,
    target_date: date,
    config: PriceProductConfig,
    store: AssetStore,
) -> DataAsset:
    if asset.subject != subject or asset.provider not in {config.price_provider, "synthetic_demo"}:
        raise PriceProductInputError(
            "price_identity_invalid", "Selected price asset has the wrong provider identity"
        )
    metadata = asset.metadata if isinstance(asset.metadata, dict) else {}
    if metadata.get("currency") != config.currency:
        raise PriceProductInputError("price_currency_invalid", "Selected price asset is not USD")
    if metadata.get("adjustment") != "splits":
        raise PriceProductInputError(
            "price_adjustment_invalid", "Selected price asset lacks split-only adjustment"
        )
    raw = raw_price_asset_for(asset, cutoff=asof.decision_time)
    if raw.subject != subject or raw.provider != asset.provider:
        raise PriceProductInputError(
            "raw_price_identity_invalid", "Raw price evidence does not bind selected series"
        )
    read_checksummed_bytes(store, raw)
    read_checksummed_bytes(store, asset)
    verify_product_price_content(
        asset=asset, raw=raw, cutoff=asof.decision_time, target_date=target_date, store=store
    )
    if asset.period_end is not None and asset.period_end < target_date:
        raise PriceProductInputError(
            "price_target_missing", "Selected price asset ends before the target date"
        )
    return raw


def _series_for_asset(
    *, asof: AsOfData, asset: DataAsset, target_date: date, config: PriceProductConfig
) -> PriceSeries:
    read = asof.price_frame_for_asset_with_diagnostics(asset=asset, through_date=target_date)
    if read.invalid_session_date_rows:
        raise PriceProductInputError(
            "price_date_invalid", "Price evidence has invalid session dates"
        )
    frame = read.frame
    if "close" not in frame.columns:
        raise PriceProductInputError("price_close_missing", "Price evidence has no close column")
    sessions = _required_sessions(target_date, config.required_closes)
    dates = tuple(frame.get_column("date").to_list())
    if len(dates) != len(set(dates)):
        raise PriceProductInputError(
            "price_date_duplicate", "Price evidence has duplicate session dates"
        )
    values = {day: index for index, day in enumerate(dates)}
    if any(day not in values for day in sessions):
        raise PriceProductInputError(
            "price_sessions_incomplete", "Price evidence has a missing XNYS session"
        )
    indices = [values[day] for day in sessions]
    closes = tuple(float(frame.get_column("close")[index]) for index in indices)
    if any(not math.isfinite(value) or value <= 0 for value in closes):
        raise PriceProductInputError(
            "price_close_invalid", "Price evidence has a non-positive or non-finite close"
        )
    try:
        reference = Decimal(str(closes[-1])).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
        StockAnalysis._meta.get_field("current_price").clean(reference, None)
        if reference <= 0:
            raise ValueError("Reference close rounds to zero")
    except (InvalidOperation, ValidationError, ValueError):
        raise PriceProductInputError(
            "price_reference_unrepresentable",
            "Reference close cannot be represented at the ledger's price precision",
        ) from None
    volume_values: tuple[float | None, ...] | None = None
    if "volume" in frame.columns:
        volume_values = tuple(
            None
            if frame.get_column("volume")[index] is None
            else float(frame.get_column("volume")[index])
            for index in indices
        )
    metadata = asset.metadata if isinstance(asset.metadata, dict) else {}
    return PriceSeries(
        identity=PriceInputIdentity(
            asset_id=asset.id,
            provider=asset.provider,
            subject=asset.subject,
            sha256=asset.sha256,
            retrieved_at=asset.retrieved_at,
            available_at=asset.available_at,
        ),
        currency=str(metadata.get("currency") or ""),
        dates=sessions,
        closes=closes,
        volumes=volume_values,
        volume_adjustment_compatible=metadata.get("volume_adjustment_compatible") is True,
    )


def _required_sessions(target_date: date, required_closes: int) -> tuple[date, ...]:
    from exchange_calendars import get_calendar  # type: ignore[import-untyped]

    calendar = get_calendar("XNYS")
    target = calendar.date_to_session(target_date, direction="none")
    return tuple(session.date() for session in calendar.sessions_window(target, -required_closes))


def _require_actual_source_mode(
    *, assets: tuple[DataAsset, DataAsset], provider: str, source_execution: SourceExecutionBinding
) -> None:
    actual = {asset.provider for asset in assets}
    if len(actual) != 1 or provider not in actual:
        raise PriceProductInputError(
            "source_provider_mixed", "Product source assets mix provider identities"
        )
    actual_provider = next(iter(actual))
    if source_execution.mode == "synthetic_demo":
        valid = actual_provider == "synthetic_demo" and provider == "synthetic_demo"
    else:
        valid = actual_provider == "twelve_data" and provider == "twelve_data"
    if not valid:
        raise PriceProductInputError(
            "source_mode_invalid", "Requested source mode does not match registered assets"
        )


def _write_calculation_artifact(
    *,
    run: AnalysisRun,
    selection: ProductSourceSelection,
    result: PriceProductResult,
    store: AssetStore,
    generated_at: datetime,
) -> tuple[DataAsset, str]:
    document = {
        "contract": CALCULATION_ARTIFACT_CONTRACT,
        "run_id": str(run.id),
        "run_digest": row_digest(ANALYSIS_RUN_MODEL, model_row_values(run, ANALYSIS_RUN_FIELDS)),
        "listing_id": str(selection.listing.id),
        "input_hash": result.input_hash,
        "source_execution": asdict(result.source_execution),
        "source_assets": selection.source_assets,
        "catalog_assets": [asset_ref_for(asset).to_json() for asset in selection.catalog_assets],
        "source_window": verified_source_window_payload(selection),
        "result": _jsonable(asdict(result)),
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    path = f"research/product/{run.id}/{selection.listing.id}/calculation.json"
    stored = store.write_bytes(path, payload)
    asset = register_asset(
        provider="stanstock",
        kind=CALCULATION_ARTIFACT_KIND,
        subject=str(run.id),
        stored=stored,
        retrieved_at=generated_at,
        available_at=generated_at,
        metadata={
            "contract": CALCULATION_ARTIFACT_CONTRACT,
            "listing_id": str(selection.listing.id),
        },
    )
    return asset, path


def _create_product_analysis(
    *,
    run: AnalysisRun,
    selection: ProductSourceSelection,
    result: PriceProductResult,
    artifact: DataAsset,
) -> StockAnalysis:
    raw_current = Decimal(str(selection.product_input.stock.closes[-1]))
    current = raw_current.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
    previous = Decimal(str(selection.product_input.stock.closes[-2]))
    daily_change = ((raw_current / previous) - Decimal(1)).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_EVEN
    )
    return StockAnalysis.objects.create(
        run=run,
        listing=selection.listing,
        current_price=current,
        daily_change=daily_change,
        overall_score=None,
        recommendation=result.recommendation.suggestion,
        risk_score=None,
        risk_class=result.risk.relative_volatility_label,
        confidence=None,
        confidence_status="not_estimated",
        component_scores={
            "schema": "research-product@1",
            "momentum": _optional_dataclass_json(result.momentum),
        },
        forecast_scenarios={
            "schema": "research-product@1",
            "projections": _jsonable(
                [asdict(projection) for projection in result.forecast.projections]
            ),
        },
        short_scenario={},
        medium_scenario={},
        long_scenario={},
        reasons=list(result.recommendation.blocking_reasons),
        risks=list(result.risk.insufficiency_reasons),
        data_quality={
            "schema": "research-product@1",
            "input_hash": result.input_hash,
            "source_assets": selection.source_assets,
            "calculation_artifact": asset_ref_for(artifact).to_json(),
            "momentum_insufficiency_reason": result.momentum_insufficiency_reason,
        },
    )


def _create_product_predictions(
    *,
    analysis: StockAnalysis,
    selection: ProductSourceSelection,
    result: PriceProductResult,
    artifact: DataAsset,
    code_revision: str,
) -> tuple[Prediction, ...]:
    artifact_ref = asset_ref_for(artifact).to_json()
    common = {
        "analysis": analysis,
        "listing": analysis.listing,
        "generated_at": analysis.run.generated_at,
        "target_date": analysis.run.target_date,
        "issued_on_time": analysis.run.issued_on_time,
        "evidence_grade": result.source_execution.evidence_grade,
        "source_mode": (
            Prediction.SourceMode.SYNTHETIC
            if result.source_execution.mode == "synthetic_demo"
            else Prediction.SourceMode.PROVIDER
        ),
        "price_provider": selection.stock_asset.provider,
        "price_subject": selection.stock_asset.subject,
        "price_at_prediction": analysis.current_price,
        "probability_positive": None,
        "confidence": None,
        "confidence_status": "not_estimated",
        "overall_score": None,
        "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
        "data_cutoff": analysis.run.data_cutoff,
        "source_assets": selection.source_assets,
        "code_revision": code_revision,
    }
    decision_id = uuid4()
    decision = Prediction.objects.create(
        id=decision_id,
        **common,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.DECISION,
        bear_return=None,
        base_return=None,
        bull_return=None,
        insufficiency_reason=result.momentum_insufficiency_reason or "",
        recommendation=result.recommendation.suggestion,
        component_scores={
            "schema": "research-product@1",
            "signal": _optional_dataclass_json(result.momentum),
        },
        model_version=prediction_model_version(
            method_version=MOMENTUM_METHOD_VERSION, issuance_id=decision_id
        ),
        method_version=MOMENTUM_METHOD_VERSION,
        calculation=_prediction_calculation(
            result=result, artifact_ref=artifact_ref, horizon="6m", method=MOMENTUM_METHOD_VERSION
        ),
    )
    rows = [decision]
    for projection in result.forecast.projections:
        prediction_id = uuid4()
        triplet = projection.ledger_returns
        rows.append(
            Prediction.objects.create(
                id=prediction_id,
                **common,
                horizon=projection.horizon,
                evidence_role=Prediction.EvidenceRole.ADVISORY,
                bear_return=None if triplet is None else triplet.lower,
                base_return=None if triplet is None else triplet.median,
                bull_return=None if triplet is None else triplet.upper,
                insufficiency_reason=projection.insufficiency_reason or "",
                recommendation=None,
                component_scores={"schema": "research-product@1", "projection": projection.horizon},
                model_version=prediction_model_version(
                    method_version=FHS_METHOD_VERSION, issuance_id=prediction_id
                ),
                method_version=FHS_METHOD_VERSION,
                calculation=_prediction_calculation(
                    result=result,
                    artifact_ref=artifact_ref,
                    horizon=projection.horizon,
                    method=FHS_METHOD_VERSION,
                ),
            )
        )
    return tuple(rows)


def _prediction_calculation(
    *, result: PriceProductResult, artifact_ref: dict[str, str], horizon: str, method: str
) -> dict[str, Any]:
    return {
        "schema": "research-product@1",
        "payload_schema": result.payload_schema,
        "method_version": method,
        "horizon": horizon,
        "input_hash": result.input_hash,
        "source_execution": asdict(result.source_execution),
        "calculation_artifact": artifact_ref,
        "momentum": _optional_dataclass_json(result.momentum),
        "risk": _jsonable(asdict(result.risk)),
        "recommendation": _jsonable(asdict(result.recommendation)),
        "forecast": _jsonable(asdict(result.forecast)),
    }


def _write_product_manifest(
    *, run: AnalysisRun, persisted: list[ProductPersistedAnalysis], plan: Any, store: AssetStore
) -> str:
    entries = [
        ManifestEntry(
            model=ANALYSIS_RUN_MODEL,
            row_id=str(run.id),
            digest=row_digest(ANALYSIS_RUN_MODEL, model_row_values(run, ANALYSIS_RUN_FIELDS)),
        )
    ]
    for item in persisted:
        entries.append(
            ManifestEntry(
                model=STOCK_ANALYSIS_MODEL,
                row_id=str(item.analysis.id),
                digest=row_digest(
                    STOCK_ANALYSIS_MODEL, model_row_values(item.analysis, STOCK_ANALYSIS_FIELDS)
                ),
            )
        )
        entries.extend(
            ManifestEntry(
                model=PREDICTION_MODEL,
                row_id=str(row.id),
                digest=row_digest(PREDICTION_MODEL, model_row_values(row, PREDICTION_FIELDS)),
            )
            for row in item.predictions
        )
    from stanstock.research.refresh_evidence import build_manifest_envelope

    payload = dumps_canonical_envelope(
        build_manifest_envelope(run_id=run.id, plan=plan, entries=entries)
    )
    path = f"research/analysis/{run.id}/output-manifest.json"
    stored = store.write_bytes(path, payload)
    register_asset(
        provider="stanstock",
        kind="analysis_output_manifest",
        subject=str(run.id),
        stored=stored,
        retrieved_at=run.generated_at,
        available_at=run.generated_at,
        metadata={"usage_scope": "product"},
    )
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _optional_dataclass_json(value: Any) -> Any:
    return None if value is None else _jsonable(asdict(value))


def _safe_unlink(store: AssetStore, path: str) -> None:
    try:
        if not DataAsset.objects.filter(relative_path=path).exists():
            store.resolve(path).unlink(missing_ok=True)
    except (DatabaseError, OSError, ValueError):
        logger.warning("product_artifact_cleanup_failed")

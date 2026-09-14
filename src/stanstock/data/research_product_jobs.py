"""Versioned recoverable target jobs for research-product-v1.

These job identities intentionally do not reuse the legacy Friday/daily
identity, so a successful frozen pipeline cannot suppress prospective
issuance.  Source recovery is attempted by the supplied task before any
credential resolver is invoked by an optional bootstrapper.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from django.contrib.auth import get_user_model
from django.db import transaction
from django.utils import timezone

from stanstock.core.jobs import JobExecutionResult, execute_target_job, target_job_lock
from stanstock.core.logging import suppress_http_client_request_logs
from stanstock.core.models import JobRun
from stanstock.core.verification_types import AssetRef, RefreshVerificationError
from stanstock.data.asof import AsOfData, verified_price_fields
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    open_asset_store,
    read_checksummed_bytes,
    resolve_asset_ref,
)
from stanstock.data.etfs import sync_investable_spy_from_asset
from stanstock.data.live_us import (
    ProviderCreditBudget,
    _ensure_my_list_listing,
    _persist_catalog,
    _persist_price_series,
    _provider_record,
    _validate_benchmark_series,
    _validate_my_list_price_series,
    _validate_my_list_reference,
    load_us_universe_config,
    resolve_us_target_date,
)
from stanstock.data.management.config_loader import default_us_universe_config_path
from stanstock.data.market_state import update_latest_market_data
from stanstock.data.models import DataAsset, Listing, UniverseSnapshot
from stanstock.data.provider_policy import (
    TWELVE_DATA_PROVIDER,
    provider_plan_allows,
    validate_provider_usage,
)
from stanstock.data.providers import twelve_data
from stanstock.data.providers.contracts import StockReference
from stanstock.data.providers.exceptions import (
    ProviderConfigurationError,
    ProviderError,
    ProviderQuotaError,
)
from stanstock.data.research_product import (
    CapturedProductIntake,
    capture_authorized_product_intake,
    load_product_intake,
    materialize_product_membership,
    product_intake_payload,
    product_membership_payload,
    verify_product_listing_catalog,
)
from stanstock.research.config import code_revision
from stanstock.research.models import AnalysisRun
from stanstock.research.price_product import PriceProductInputError
from stanstock.research.price_product_config import (
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_VERSION,
    default_price_product_config_path,
    load_price_product_config,
)
from stanstock.research.product_pipeline import (
    _require_observed_commit_deadline,
    select_product_price_asset,
    verify_price_product_output,
)
from stanstock.research.service import analyze_snapshot

RESEARCH_INTAKE_JOB = "research_intake_v1"
DAILY_RESEARCH_JOB = "daily_research_v1"
SCHEDULED_RESEARCH_JOB = "scheduled_refresh_research_v1"


def execute_daily_research_job(
    *,
    target_date: date,
    owner: object,
    issuance_key: str = "scheduled",
    issued_on_time: bool = False,
    store: AssetStore | None = None,
    core_config_path: Path | None = None,
    enforce_rate_limit: bool = True,
) -> JobRun:
    """Capture, acquire, admit and issue one recoverable bounded research cohort."""
    if not isinstance(issued_on_time, bool):
        raise ValueError("issued_on_time must be an explicit boolean")
    user_model = get_user_model()
    if not isinstance(owner, user_model) or not owner.is_active:
        raise ValueError("Research refresh requires an active authenticated owner")
    asset_store = store or open_asset_store()
    identity = {"owner_id": str(owner.pk), "issuance_key": issuance_key, "product": PRODUCT_VERSION}

    def daily_task(run: JobRun) -> JobExecutionResult:
        intake = load_product_intake(
            target_date=target_date,
            owner_id=str(owner.pk),
            issuance_key=issuance_key,
            store=asset_store,
        )
        if intake is not None:
            completed = _completed_product_run(intake, store=asset_store)
            if completed is not None:
                _project_product_market_state(completed, store=asset_store)
                return JobExecutionResult(details=_daily_details(intake, completed, recovered=True))
        resolve_us_target_date(decision_time=timezone.now(), explicit_target=target_date)
        if issued_on_time:
            _require_observed_commit_deadline(
                target_date=target_date, code_revision=code_revision()
            )
        path = core_config_path or default_us_universe_config_path()
        if intake is None:
            intake = capture_authorized_product_intake(
                target_date=target_date,
                evidence_grade="observed" if issued_on_time else "research",
                issuance_key=issuance_key,
                owner=owner,
                core_config_path=path,
                policy_identity=PRODUCT_EFFECTIVE_CONFIG_HASH,
                captured_at=timezone.now(),
                store=asset_store,
            )
        captured = product_intake_payload(intake, store=asset_store)
        if captured["evidence_grade"] != ("observed" if issued_on_time else "research"):
            raise ValueError(
                "Uncompleted captured intake has a different grade; use a new issuance"
            )
        run.details = {
            "invocation_identity": identity,
            "intake_asset": asset_ref_for(intake.asset).to_json(),
        }
        run.save(update_fields=["details"])

        def intake_task(_child: JobRun) -> JobExecutionResult:
            snapshot = _snapshot_for_intake(intake, store=asset_store)
            if snapshot is not None:
                product_membership_payload(snapshot, store=asset_store)
                return JobExecutionResult(
                    details={"snapshot_id": str(snapshot.id), "recovered": True}
                )
            return _acquire_product_membership(
                intake=intake,
                target_date=target_date,
                core_config_path=path,
                store=asset_store,
                enforce_rate_limit=enforce_rate_limit,
            )

        child = execute_target_job(
            job_name=product_job_name(RESEARCH_INTAKE_JOB, identity),
            region="us",
            target_date=target_date,
            task=intake_task,
        )
        child = _successful_attempt(child)
        run.details["intake_job_id"] = str(child.id)
        run.save(update_fields=["details"])
        if child.status == JobRun.Status.NO_DATA:
            return JobExecutionResult(
                status=JobRun.Status.NO_DATA,
                details={**run.details, "reason": "no_qualified_research_members"},
            )
        snapshot = _snapshot_for_intake(intake, store=asset_store)
        if snapshot is None or child.details.get("snapshot_id") != str(snapshot.id):
            raise ValueError("Research intake child does not bind its captured snapshot")
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=timezone.now(),
            target_date=target_date,
            issued_on_time=issued_on_time,
            provider=TWELVE_DATA_PROVIDER,
            benchmark_subject="SPY",
            store=asset_store,
            config_path=default_price_product_config_path(),
        )
        completed = _completed_product_run(intake, store=asset_store)
        if completed is None:
            raise ValueError("Research writer did not commit its complete captured output")
        _project_product_market_state(completed, store=asset_store)
        return JobExecutionResult(
            details={
                **_daily_details(intake, completed, recovered=False),
                "intake_job_id": str(child.id),
                "credits_used": child.details.get("credits_used", 0),
            }
        )

    with target_job_lock(job_name=DAILY_RESEARCH_JOB, region="us", target_date=target_date):
        with suppress_http_client_request_logs():
            validate_provider_usage(_provider_record(require_enabled=False))
            job = execute_target_job(
                job_name=product_job_name(DAILY_RESEARCH_JOB, identity),
                region="us",
                target_date=target_date,
                task=daily_task,
            )
            successful = _successful_attempt(job)
            if successful.status == JobRun.Status.SUCCESS:
                intake = load_product_intake(
                    target_date=target_date,
                    owner_id=str(owner.pk),
                    issuance_key=issuance_key,
                    store=asset_store,
                )
                if intake is None:
                    raise ValueError("Completed research job has no registered intake")
                completed = _completed_product_run(intake, store=asset_store)
                if completed is None or successful.details.get("analysis_run_id") != str(
                    completed.id
                ):
                    raise ValueError("Completed research job does not bind its exact output")
            return job


def _acquire_product_membership(
    *,
    intake: CapturedProductIntake,
    target_date: date,
    core_config_path: Path,
    store: AssetStore,
    enforce_rate_limit: bool,
) -> JobExecutionResult:
    config = load_us_universe_config(core_config_path)
    captured = product_intake_payload(intake, store=store)
    if config.raw != captured["core_config"] or config.symbols != intake.core_symbols:
        raise ValueError("Captured core configuration changed; use a new issuance")
    captured_entitlement = captured.get("entitlement")
    if not isinstance(captured_entitlement, dict):
        raise ValueError("Captured research intake has no entitlement policy")
    captured_plan = captured_entitlement.get("plan")
    if not isinstance(captured_plan, str):
        raise ValueError("Captured research intake has no entitled provider plan")
    budget = ProviderCreditBudget(enforce_spacing=enforce_rate_limit)
    key: str | None = None
    credits = 0

    def request_key() -> str:
        nonlocal key, credits
        if key is None:
            validate_provider_usage(_provider_record(require_enabled=True))
            key = twelve_data.resolve_api_key()
        budget.preflight(1)
        budget.consume()
        credits += 1
        return key

    catalogs: list[DataAsset] = []
    references: dict[str, list[StockReference]] = {
        symbol: [] for symbol in intake.requested_symbols
    }
    for exchange in config.exchanges:
        now = timezone.now()
        asset = (
            DataAsset.objects.filter(
                provider=TWELVE_DATA_PROVIDER,
                kind="stock_catalog",
                subject=exchange,
                available_at__lte=now,
                retrieved_at__lte=now,
                retrieved_at__gte=now - timedelta(days=7),
            )
            .order_by("-available_at", "-retrieved_at", "-id")
            .first()
        )
        if asset is None:
            catalog = twelve_data.fetch_stock_catalog(
                exchange=exchange,
                required_symbols=intake.requested_symbols,
                api_key=request_key(),
            )
            asset = _persist_catalog(store, catalog)
        parsed, _count = twelve_data.parse_stock_catalog_references(
            read_checksummed_bytes(store, asset),
            exchange=exchange,
            required_symbols=intake.requested_symbols,
            require_complete=True,
        )
        if any(reference.exchange != exchange for reference in parsed):
            raise ValueError("Catalog response contains a conflicting exchange identity")
        catalogs.append(asset)
        for reference in parsed:
            references[reference.symbol].append(reference)

    admissions: dict[str, dict[str, Any]] = {}
    listings: dict[str, Listing] = {}
    for symbol in intake.requested_symbols:
        matches = references[symbol]
        if len(matches) != 1 or symbol == config.benchmark_symbol:
            admissions[symbol] = _admission(
                "identity_rejected", "catalog_identity_missing_or_ambiguous"
            )
            continue
        plan = str(_provider_record(require_enabled=False).metadata["plan"])
        if not provider_plan_allows(plan, matches[0].access_plan) or not provider_plan_allows(
            captured_plan, matches[0].access_plan
        ):
            admissions[symbol] = _admission("entitlement_rejected", "catalog_plan_not_authorized")
            continue
        try:
            _validate_my_list_reference(matches[0])
            with transaction.atomic():
                listing = _ensure_my_list_listing(
                    symbol=symbol, reference=matches[0], target_date=target_date
                )
        except ValueError:
            admissions[symbol] = _admission(
                "identity_rejected", "catalog_listing_or_entitlement_not_supported"
            )
            continue
        listings[symbol] = listing

    product_config = load_price_product_config()
    selected: dict[str, DataAsset] = {}
    missing: list[str] = []
    for symbol in (*listings, config.benchmark_symbol):
        try:
            selected[symbol] = select_product_price_asset(
                asof=AsOfData(timezone.now(), store),
                subject=symbol,
                target_date=target_date,
                config=product_config,
                store=store,
                provider=TWELVE_DATA_PROVIDER,
            )
        except PriceProductInputError as exc:
            if exc.reason_code in {
                "price_history_insufficient",
                "newer_price_correction_conflicts",
            }:
                missing.append(symbol)
            elif symbol == config.benchmark_symbol:
                raise ValueError("Benchmark evidence is invalid") from None
            else:
                admissions[symbol] = _admission(
                    "evidence_invalid", exc.reason_code, listings[symbol]
                )
        except (RefreshVerificationError, ValueError, ProviderError):
            if symbol == config.benchmark_symbol:
                raise ValueError("Benchmark evidence cannot be verified") from None
            admissions[symbol] = _admission(
                "evidence_invalid", "price_evidence_invalid", listings[symbol]
            )

    # Every cache candidate has been inspected before a key or price credit
    # is requested. Each unresolved symbol appears only once in this list.
    for symbol in missing:
        candidate_listing = listings.get(symbol)
        fetched_asset: DataAsset | None = None
        try:
            series = twelve_data.fetch_daily_price_series(
                symbol,
                start_date=target_date - timedelta(days=366 * config.history_years),
                end_date=target_date,
                adjustment=config.price_adjustment,
                api_key=request_key(),
            )
            if candidate_listing is None:
                _validate_benchmark_series(series, config, target_date)
            else:
                _validate_my_list_price_series(
                    series, reference=references[symbol][0], target_date=target_date
                )
            fetched_asset = _persist_price_series(
                store=store,
                series=series,
                listing=candidate_listing,
                resolved_mic_code="ARCX" if candidate_listing is None else None,
                catalog_assets=catalogs if candidate_listing is None else None,
            )
            selected[symbol] = select_product_price_asset(
                asof=AsOfData(timezone.now(), store),
                subject=symbol,
                target_date=target_date,
                config=product_config,
                store=store,
                provider=TWELVE_DATA_PROVIDER,
            )
        except (ProviderConfigurationError, ProviderQuotaError):
            raise
        except PriceProductInputError as exc:
            if candidate_listing is None:
                raise ValueError("Benchmark history does not qualify for research") from None
            admissions[symbol] = _admission(
                "insufficient_history"
                if exc.reason_code == "price_history_insufficient"
                else "evidence_invalid",
                exc.reason_code,
                candidate_listing,
                fetched_asset,
                bootstrap_attempted=True,
            )
            if fetched_asset is not None and exc.reason_code == "price_history_insufficient":
                admissions[symbol]["history_qualification"] = _history_shortfall(
                    asset=fetched_asset, target_date=target_date, store=store
                )
        except (ProviderError, ValueError):
            if candidate_listing is None:
                raise ValueError("Benchmark history acquisition failed") from None
            admissions[symbol] = _admission(
                "provider_failed",
                "history_acquisition_failed",
                candidate_listing,
                bootstrap_attempted=True,
            )

    decision_time = timezone.now()
    for symbol in tuple(selected):
        try:
            selected[symbol] = select_product_price_asset(
                asof=AsOfData(decision_time, store),
                subject=symbol,
                target_date=target_date,
                config=product_config,
                store=store,
                provider=TWELVE_DATA_PROVIDER,
            )
        except (PriceProductInputError, RefreshVerificationError, ValueError, ProviderError):
            if symbol == config.benchmark_symbol:
                raise ValueError("Benchmark evidence changed during research intake") from None
            selected.pop(symbol)
            admissions[symbol] = _admission(
                "evidence_invalid", "price_evidence_changed_during_intake", listings[symbol]
            )
    states: dict[UUID, str] = {}
    qualified: list[UUID] = []
    for symbol, listing in listings.items():
        if symbol in selected:
            try:
                verify_product_listing_catalog(
                    listing=listing,
                    stock_asset=selected[symbol],
                    benchmark_asset=selected[config.benchmark_symbol],
                    catalog_assets=tuple(catalogs),
                    cutoff=decision_time,
                    store=store,
                )
            except (ValueError, ProviderError, RefreshVerificationError):
                admissions[symbol] = _admission(
                    "identity_rejected", "price_catalog_identity_conflict", listing
                )
            else:
                admissions[symbol] = _admission(
                    "admitted",
                    "",
                    listing,
                    selected[symbol],
                    bootstrap_attempted=symbol in missing,
                )
                qualified.append(listing.id)
        states[listing.id] = admissions[symbol]["status"]
    if not qualified:
        return JobExecutionResult(
            status=JobRun.Status.NO_DATA,
            details={"admissions": admissions, "credits_used": credits},
        )
    snapshot = materialize_product_membership(
        intake=intake,
        qualified_listing_ids=qualified,
        candidate_states=states,
        admissions=admissions,
        catalog_assets=tuple(catalogs),
        benchmark_asset=selected[config.benchmark_symbol],
        captured_at=decision_time,
        store=store,
    )
    return JobExecutionResult(
        details={
            "snapshot_id": str(snapshot.id),
            "intake_asset": asset_ref_for(intake.asset).to_json(),
            "credits_used": credits,
            "admitted": len(qualified),
            "pending_or_rejected": len(admissions) - len(qualified),
        }
    )


def _admission(
    status: str,
    reason: str,
    listing: Listing | None = None,
    asset: DataAsset | None = None,
    *,
    bootstrap_attempted: bool = False,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "listing_id": None if listing is None else str(listing.id),
        "price_asset": None if asset is None else asset_ref_for(asset).to_json(),
        "bootstrap_attempted": bootstrap_attempted,
    }


def _history_shortfall(*, asset: DataAsset, target_date: date, store: AssetStore) -> dict[str, int]:
    from exchange_calendars import get_calendar  # type: ignore[import-untyped]

    required = load_price_product_config().required_closes
    calendar = get_calendar("XNYS")
    sessions = {
        item.date()
        for item in calendar.sessions_window(calendar.date_to_session(target_date), -required)
    }
    frame = (
        AsOfData(timezone.now(), store)
        .price_frame_for_asset_with_diagnostics(asset=asset, through_date=target_date)
        .frame
    )
    available = len(sessions.intersection(frame.get_column("date").to_list()))
    return {
        "required_closes": required,
        "available_required_closes": available,
        "missing_required_closes": required - available,
    }


def _snapshot_for_intake(
    intake: CapturedProductIntake, *, store: AssetStore
) -> UniverseSnapshot | None:
    snapshots = list(
        UniverseSnapshot.objects.filter(
            universe__slug=f"{PRODUCT_VERSION}-{intake.asset.sha256[:16]}"
        ).select_related("universe")
    )
    if len(snapshots) > 1:
        raise ValueError("Captured research intake has conflicting snapshots")
    if not snapshots:
        return None
    payload = product_membership_payload(snapshots[0], store=store)
    if payload.get("intake") != asset_ref_for(intake.asset).to_json():
        raise ValueError("Recovered snapshot belongs to a different captured intake")
    return snapshots[0]


def _completed_product_run(
    intake: CapturedProductIntake, *, store: AssetStore
) -> AnalysisRun | None:
    snapshot = _snapshot_for_intake(intake, store=store)
    if snapshot is None:
        return None
    runs = list(
        AnalysisRun.objects.filter(
            universe_snapshot=snapshot,
            config_version=PRODUCT_VERSION,
            config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
        )
    )
    if len(runs) > 1:
        raise ValueError("Captured research intake has conflicting completed analyses")
    if not runs:
        return None
    verify_price_product_output(run=runs[0], store=store)
    return runs[0]


def _daily_details(
    intake: CapturedProductIntake, run: AnalysisRun, *, recovered: bool
) -> dict[str, Any]:
    return {
        "intake_asset": asset_ref_for(intake.asset).to_json(),
        "snapshot_id": str(run.universe_snapshot_id),
        "analysis_run_id": str(run.id),
        "evidence_grade": run.universe_snapshot.grade,
        "recovered": recovered,
        "credits_used": 0,
    }


def _successful_attempt(job: JobRun) -> JobRun:
    if job.status != JobRun.Status.SKIPPED:
        return job
    original = JobRun.objects.filter(
        pk=job.details.get("successful_run_id"),
        job_name=job.job_name,
        target_date=job.target_date,
        region=job.region,
        status=JobRun.Status.SUCCESS,
    ).first()
    if original is None:
        raise ValueError("Research retry has no matching successful attempt")
    return original


def product_job_name(kind: str, identity: dict[str, str]) -> str:
    """Use the existing target uniqueness guard for each explicit owner issuance."""
    if kind not in {DAILY_RESEARCH_JOB, RESEARCH_INTAKE_JOB, SCHEDULED_RESEARCH_JOB}:
        raise ValueError("Unknown research product job kind")
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{kind}:{digest[:32]}"


def _project_product_market_state(run: AnalysisRun, *, store: AssetStore) -> None:
    payload = product_membership_payload(run.universe_snapshot, store=store)
    admissions = payload.get("admissions")
    if not isinstance(admissions, dict):
        raise ValueError("Research market projection has no captured admissions")
    for symbol, entry in admissions.items():
        if entry["status"] != "admitted":
            continue
        listing = Listing.objects.get(id=entry["listing_id"], provider_symbol=symbol)
        asset = resolve_asset_ref(AssetRef.from_json(entry["price_asset"]), cutoff=run.generated_at)
        fields = verified_price_fields(
            asset, cutoff=run.generated_at, target_date=run.target_date, close_places=6, store=store
        )
        update_latest_market_data(
            listing=listing,
            session_date=run.target_date,
            observed_at=asset.retrieved_at,
            close=fields.close,
            previous_close=fields.previous_close,
            volume=fields.volume,
            source_asset=asset,
        )
    benchmark = resolve_asset_ref(
        AssetRef.from_json(payload["benchmark_asset"]), cutoff=run.generated_at
    )
    sync_investable_spy_from_asset(asset=benchmark, target_date=run.target_date, store=store)

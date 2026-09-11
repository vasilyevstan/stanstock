from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Barrier
from typing import Any
from uuid import uuid4

import pytest
from django.contrib import admin
from django.contrib.auth import get_user_model
from django.db import (
    IntegrityError,
    close_old_connections,
    connection,
    connections,
    transaction,
)
from django.test import Client, RequestFactory
from django.urls import reverse
from django.utils import timezone

import refresh_fixtures
from stanstock.core import refresh_verification as refresh_verification_module
from stanstock.core.models import JobRun
from stanstock.core.refresh_verification import ReplayedScheduledRefresh
from stanstock.core.verification_types import AssetRef, RefreshVerificationError
from stanstock.data import provider_credentials
from stanstock.data.assets import AssetStore, asset_ref_for, register_asset, resolve_asset_ref
from stanstock.data.live_us import ProviderCreditBudget
from stanstock.data.management.config_loader import config_hash
from stanstock.data.models import (
    Company,
    DataAsset,
    LatestMarketData,
    Listing,
    ProviderRecord,
    Region,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.data.providers import twelve_data
from stanstock.data.refresh_evidence import (
    UNIVERSE_MEMBERSHIP_EVIDENCE_KIND,
    build_membership_evidence_envelope,
)
from stanstock.portfolio import watchlist as watchlist_module
from stanstock.portfolio.admin import TrackedSymbolAdmin
from stanstock.portfolio.models import Portfolio, PortfolioHolding, TrackedSymbol
from stanstock.research.models import (
    AnalysisRun,
    Prediction,
    Recommendation,
    RiskClass,
    StockAnalysis,
)
from test_refresh_verification import (
    CODE_REVISION,
    TARGET_DATE,
    _build_verified_state,
    _record_verified_parent,
)


def _owner(username: str = "owner"):
    return get_user_model().objects.create_user(
        username=username,
        password="correct-password",
    )


def _listing(
    symbol: str,
    *,
    security_type: str = Security.SecurityType.COMMON_STOCK,
) -> Listing:
    company = Company.objects.create(name=f"{symbol} Corp", country="US")
    security = Security.objects.create(
        company=company,
        security_type=security_type,
        name=f"{symbol} Security",
    )
    return Listing.objects.create(
        security=security,
        ticker=symbol,
        provider_symbol=symbol,
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )


def _catalog_row(symbol: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "symbol": symbol,
        "name": f"{symbol} Synthetic Corp",
        "currency": "USD",
        "exchange": "NASDAQ",
        "mic_code": "XNAS",
        "country": "United States",
        "type": "Common Stock",
        "access": {"plan": "Basic"},
    }
    row.update(overrides)
    return row


def _store_catalogs(
    tmp_path,
    settings,
    *,
    nasdaq_rows: list[object] | None = None,
    nyse_rows: list[object] | None = None,
    nasdaq_count: int | None = None,
    nyse_count: int | None = None,
    nasdaq_payload: dict[str, object] | None = None,
    nyse_payload: dict[str, object] | None = None,
    target_date: date | None = None,
    verification_status: str = "verified",
    plan: str = "grow",
) -> dict[str, object]:
    settings.DATA_DIR = tmp_path
    store = AssetStore(tmp_path)
    now = timezone.now()
    bundle_target = target_date or timezone.localdate()
    assets: list[DataAsset] = []
    nonce = uuid4().hex
    for exchange, rows, declared_count, explicit_payload in (
        ("NASDAQ", nasdaq_rows or [], nasdaq_count, nasdaq_payload),
        ("NYSE", nyse_rows or [], nyse_count, nyse_payload),
    ):
        payload_document = (
            explicit_payload
            if explicit_payload is not None
            else {
                "data": rows,
                "count": len(rows) if declared_count is None else declared_count,
                "status": "ok",
            }
        )
        payload = json.dumps(payload_document).encode()
        stored = store.write_bytes(
            f"catalogs/{exchange.lower()}-{nonce}.json",
            payload,
        )
        assets.append(
            register_asset(
                provider="twelve_data",
                kind="stock_catalog",
                subject=exchange,
                stored=stored,
                retrieved_at=now,
                available_at=now,
            )
        )

    hash_payload = {
        "config": {"exchanges": ["NASDAQ", "NYSE"]},
        "catalog_assets": [{"sha256": asset.sha256, "subject": asset.subject} for asset in assets],
        "members": [],
    }
    universe = Universe.objects.create(
        slug=f"wl-catalog-{nonce}",
        name="Watchlist catalog fixture",
        config_version="test-v1",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=bundle_target,
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash=config_hash(hash_payload),
    )
    run = AnalysisRun.objects.create(
        generated_at=now,
        data_cutoff=now,
        target_date=bundle_target,
        issued_on_time=True,
        universe_snapshot=snapshot,
        config_version="watchlist-test-v1",
        config_hash="b" * 64,
        code_revision="test-revision",
    )
    membership_envelope = build_membership_evidence_envelope(
        snapshot_id=snapshot.id,
        hash_payload=hash_payload,
        catalog_assets=assets,
    )
    membership_stored = store.write_bytes(
        f"evidence/watchlist-{snapshot.id}.json",
        json.dumps(membership_envelope, sort_keys=True, separators=(",", ":")).encode(),
    )
    register_asset(
        provider="stanstock",
        kind=UNIVERSE_MEMBERSHIP_EVIDENCE_KIND,
        subject=str(snapshot.id),
        stored=membership_stored,
        retrieved_at=now,
        available_at=now,
    )

    refs = [asset_ref_for(asset).to_json() for asset in assets]
    market = JobRun.objects.create(
        job_name="daily",
        region="us",
        target_date=bundle_target,
        status=JobRun.Status.SUCCESS,
        finished_at=now,
        details={
            "provider": "twelve_data",
            "snapshot_id": str(snapshot.id),
            "snapshot_grade": UniverseSnapshot.Grade.OBSERVED,
            "analysis_run_id": str(run.id),
            "catalog_asset_ids": [str(asset.id) for asset in assets],
            "catalog_refs": refs,
        },
    )
    verification = {
        "status": verification_status,
        "target_date": bundle_target.isoformat(),
        "snapshot_id": str(snapshot.id),
        "snapshot_grade": UniverseSnapshot.Grade.OBSERVED,
        "analysis_run_id": str(run.id),
        "code_revision": run.code_revision,
        "child_job_run_ids": {"market": str(market.id)},
        "asset_manifest": {"count": len(assets) + 1, "hash": "c" * 64},
    }
    parent = JobRun.objects.create(
        job_name="scheduled_refresh",
        region="us",
        target_date=bundle_target,
        status=JobRun.Status.SUCCESS,
        finished_at=now,
        details={
            "target_date": bundle_target.isoformat(),
            "snapshot_grade": UniverseSnapshot.Grade.OBSERVED,
            "code_revision": run.code_revision,
            "stages": {
                "market": {
                    "job_run_id": str(market.id),
                    "status": market.status,
                    "attempt": market.attempt,
                    "error": "",
                }
            },
            "verification": verification,
        },
    )
    ProviderRecord.objects.update_or_create(
        provider="twelve_data",
        defaults={"metadata": {"plan": plan}},
    )
    return {
        "assets": tuple(assets),
        "market": market,
        "parent": parent,
        "run": run,
        "snapshot": snapshot,
    }


@pytest.fixture(autouse=True)
def _stub_completed_refresh_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep catalog-behavior tests focused below the canonical replay boundary.

    The real verifier/replay integration has dedicated tests in
    ``test_refresh_verification.py``. This test double still resolves exact
    persisted identities, so watchlist tests can independently exercise raw
    catalog completeness and candidate normalization without constructing an
    entire analysis/evaluation/portfolio pipeline for every row case.
    """

    def replay(parent: JobRun) -> ReplayedScheduledRefresh:
        details = parent.details
        if not isinstance(details, dict):
            raise RefreshVerificationError("test_parent_invalid", "Invalid test parent")
        stages = details.get("stages")
        verification = details.get("verification")
        if not isinstance(stages, dict) or not isinstance(verification, dict):
            raise RefreshVerificationError("test_parent_invalid", "Invalid test parent")
        market_stage = stages.get("market")
        if not isinstance(market_stage, dict):
            raise RefreshVerificationError("test_market_invalid", "Invalid test market stage")
        market = JobRun.objects.filter(pk=market_stage.get("job_run_id")).first()
        if (
            market is None
            or market.status != JobRun.Status.SUCCESS
            or market.target_date != parent.target_date
            or verification.get("child_job_run_ids", {}).get("market") != str(market.pk)
        ):
            raise RefreshVerificationError("test_market_invalid", "Invalid test market stage")
        market_details = market.details
        if not isinstance(market_details, dict):
            raise RefreshVerificationError("test_market_invalid", "Invalid test market details")
        snapshot = UniverseSnapshot.objects.filter(pk=verification.get("snapshot_id")).first()
        run = AnalysisRun.objects.filter(pk=verification.get("analysis_run_id")).first()
        if snapshot is None or run is None:
            raise RefreshVerificationError("test_output_invalid", "Invalid test outputs")
        refs = tuple(AssetRef.from_json(raw) for raw in market_details.get("catalog_refs", []))
        if [str(ref.id) for ref in refs] != market_details.get("catalog_asset_ids"):
            raise RefreshVerificationError("test_catalog_invalid", "Invalid test catalog refs")
        assets = tuple(resolve_asset_ref(ref, cutoff=run.data_cutoff) for ref in refs)
        return ReplayedScheduledRefresh(
            parent=parent,
            verification=verification,
            snapshot=snapshot,
            analysis_run=run,
            catalog_assets=assets,
        )

    monkeypatch.setattr(watchlist_module, "replay_recorded_scheduled_refresh", replay)


def _analysis(
    listing: Listing,
    *,
    target_date: date = date(2026, 9, 10),
    grade: str = UniverseSnapshot.Grade.RESEARCH,
    generated_at=None,
    data_cutoff=None,
    provider: str | None = None,
    recommendation: str = Recommendation.BUY,
    score: Decimal = Decimal("88.50"),
) -> StockAnalysis:
    nonce = uuid4().hex
    universe = Universe.objects.create(
        slug=f"wl-analysis-{nonce}",
        name="Watchlist analysis fixture",
        config_version="test-v1",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=target_date,
        grade=grade,
        config_hash="a" * 64,
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    now = generated_at or timezone.now()
    cutoff = data_cutoff or now
    run = AnalysisRun.objects.create(
        generated_at=now,
        data_cutoff=cutoff,
        target_date=snapshot.as_of_date,
        universe_snapshot=snapshot,
        config_version="watchlist-test-v1",
        config_hash="b" * 64,
        code_revision="test",
    )
    return StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("123.45"),
        overall_score=score,
        recommendation=recommendation,
        risk_score=Decimal("24"),
        risk_class=RiskClass.LOW,
        confidence=Decimal("72"),
        data_quality=({"source_assets": [{"provider": provider}]} if provider is not None else {}),
    )


@pytest.mark.django_db
def test_tracked_symbol_normalizes_and_database_enforces_owner_uniqueness() -> None:
    owner = _owner()
    other = _owner("other")

    tracked = TrackedSymbol.objects.create(owner=owner, symbol="  syn-a ")

    assert tracked.symbol == "SYN-A"
    TrackedSymbol.objects.create(owner=other, symbol="syn-a")
    with pytest.raises(IntegrityError), transaction.atomic():
        TrackedSymbol.objects.create(owner=owner, symbol="SYN-A")
    with pytest.raises(IntegrityError), transaction.atomic():
        TrackedSymbol.objects.filter(pk=tracked.pk).update(symbol="syn-a")

    constraints = connection.introspection.get_constraints(
        connection.cursor(),
        TrackedSymbol._meta.db_table,
    )
    assert constraints["unique_owner_tracked_symbol"]["unique"] is True
    assert constraints["tracked_symbol_normalized"]["check"] is True


@pytest.mark.django_db
def test_my_list_requires_authentication_and_scopes_entries_to_owner(client) -> None:
    owner = _owner()
    other = _owner("other")
    TrackedSymbol.objects.create(owner=owner, symbol="OWN")
    TrackedSymbol.objects.create(owner=other, symbol="PRIVATE")

    anonymous = client.get(reverse("my-list"))
    assert anonymous.status_code == 302
    assert anonymous.url.startswith(reverse("login"))

    client.force_login(owner)
    response = client.get(reverse("my-list"))

    assert response.status_code == 200
    assert b"OWN" in response.content
    assert b"PRIVATE" not in response.content


@pytest.mark.django_db
def test_tracked_symbol_admin_queryset_is_owner_scoped_for_non_superuser() -> None:
    owner = _owner()
    other = _owner("other")
    own = TrackedSymbol.objects.create(owner=owner, symbol="OWN")
    TrackedSymbol.objects.create(owner=other, symbol="PRIVATE")
    request = RequestFactory().get("/admin/portfolio/trackedsymbol/")
    request.user = owner

    queryset = TrackedSymbolAdmin(TrackedSymbol, admin.site).get_queryset(request)

    assert list(queryset) == [own]


@pytest.mark.django_db
@pytest.mark.parametrize(
    "security_type",
    [Security.SecurityType.COMMON_STOCK, Security.SecurityType.ADR],
)
def test_adds_existing_eligible_listing_without_catalog(client, security_type: str) -> None:
    owner = _owner()
    listing = _listing("LOCAL", security_type=security_type)
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": " local "})

    assert response.status_code == 302
    assert response.url == reverse("my-list")
    tracked = TrackedSymbol.objects.get(owner=owner)
    assert tracked.symbol == listing.provider_symbol


@pytest.mark.django_db
def test_rejects_ambiguous_active_listing_identity_without_catalog_fallback(client) -> None:
    owner = _owner()
    _listing("DUPE")
    second = _listing("OTHER")
    second.provider_symbol = "DUPE"
    second.save(update_fields=["provider_symbol"])
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "DUPE"})

    assert response.status_code == 400
    assert b"matches more than one active listing" in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
def test_rejects_active_ineligible_alias_collision_without_catalog_fallback(client) -> None:
    owner = _owner()
    _listing("DUPE")
    collision = _listing("OTHER", security_type=Security.SecurityType.ETF)
    collision.provider_symbol = "DUPE"
    collision.save(update_fields=["provider_symbol"])
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "DUPE"})

    assert response.status_code == 400
    assert b"matches more than one active listing" in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("security_type", "region", "currency"),
    [
        (Security.SecurityType.ETF, Region.US, "USD"),
        (Security.SecurityType.COMMON_STOCK, Region.EUROPE, "USD"),
        (Security.SecurityType.COMMON_STOCK, Region.US, "EUR"),
    ],
)
def test_rejects_existing_unsupported_listing(
    client,
    security_type: str,
    region: str,
    currency: str,
) -> None:
    owner = _owner()
    listing = _listing("BLOCKED", security_type=security_type)
    listing.region = region
    listing.currency = currency
    listing.save(update_fields=["region", "currency"])
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "BLOCKED"})

    assert response.status_code == 400
    assert b"Only active United States USD common-stock or ADR listings" in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
def test_adds_catalog_backed_symbol_without_creating_listing(
    client,
    tmp_path,
    settings,
) -> None:
    owner = _owner()
    _store_catalogs(tmp_path, settings, nasdaq_rows=[_catalog_row("FUTR")])
    client.force_login(owner)
    listing_count = Listing.objects.count()

    response = client.post(reverse("my-list"), {"symbol": " futr "}, follow=True)

    assert response.status_code == 200
    assert TrackedSymbol.objects.filter(owner=owner, symbol="FUTR").count() == 1
    assert Listing.objects.count() == listing_count
    assert b"Not in the selected analysis universe" in response.content
    assert b"No stored analysis yet" in response.content


@pytest.mark.django_db
def test_catalog_admission_ignores_unrelated_malformed_complete_row(
    client,
    tmp_path,
    settings,
) -> None:
    owner = _owner()
    _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[
            _catalog_row("FUTR"),
            {
                "symbol": "UNRELATED",
                "name": None,
                "currency": "USD",
                "exchange": "NASDAQ",
                "mic_code": "XNAS",
                "country": "United States",
                "type": "Common Stock",
            },
        ],
    )
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "FUTR"})

    assert response.status_code == 302
    assert TrackedSymbol.objects.filter(owner=owner, symbol="FUTR").exists()


@pytest.mark.django_db
def test_self_certified_lookalike_parent_is_rejected_by_real_replay(
    client,
    tmp_path,
    settings,
    monkeypatch,
) -> None:
    owner = _owner()
    _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[_catalog_row("LOOKALIKE")],
    )
    client.force_login(owner)
    monkeypatch.setattr(
        watchlist_module,
        "replay_recorded_scheduled_refresh",
        refresh_verification_module.replay_recorded_scheduled_refresh,
    )

    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail("scheduled verification replay must not access a provider")

    monkeypatch.setattr(twelve_data, "fetch", forbidden)
    monkeypatch.setattr(twelve_data, "fetch_stock_catalog", forbidden)
    monkeypatch.setattr(twelve_data, "resolve_api_key", forbidden)
    monkeypatch.setattr(ProviderCreditBudget, "preflight", forbidden)
    monkeypatch.setattr(ProviderCreditBudget, "consume", forbidden)

    response = client.post(reverse("my-list"), {"symbol": "LOOKALIKE"})

    assert response.status_code == 400
    assert b"catalog evidence is unavailable or invalid" in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
def test_real_replay_requires_recoverable_sec_child_before_catalog_admission(
    client,
    tmp_path,
    settings,
    monkeypatch,
) -> None:
    """A same-target SEC success remains required even with a false market gate.

    Replay deliberately considers every same-target success visible at replay
    time. It does not try to reconstruct whether that success existed at the
    historical instant the mutable parent details were written.
    """
    candidate = "RECOVER"
    real_catalog = refresh_fixtures._catalog

    def catalog_with_unconfigured_candidate(*args: object, **kwargs: object):
        catalog = real_catalog(*args, **kwargs)
        rows = [
            _catalog_row(reference.symbol)
            for reference in (*catalog.references, refresh_fixtures._reference(candidate))
        ]
        return dataclasses.replace(
            catalog,
            references=(*catalog.references, refresh_fixtures._reference(candidate)),
            count=len(rows),
            raw_bytes=json.dumps({"status": "ok", "count": len(rows), "data": rows}).encode(),
        )

    monkeypatch.setattr(refresh_fixtures, "_catalog", catalog_with_unconfigured_candidate)
    stages, config = _build_verified_state(monkeypatch, tmp_path, sec=True)
    market = JobRun.objects.get(pk=stages["market"]["job_run_id"])
    assert market.details["long_forecast_requested"] is False
    sec_success = JobRun.objects.get(pk=stages["sec_fundamentals"]["job_run_id"])
    assert sec_success.status == JobRun.Status.SUCCESS
    parent = _record_verified_parent(stages=stages, config=config)

    monkeypatch.setattr(
        refresh_verification_module,
        "load_us_universe_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        watchlist_module,
        "replay_recorded_scheduled_refresh",
        refresh_verification_module.replay_recorded_scheduled_refresh,
    )
    monkeypatch.setattr(watchlist_module.timezone, "localdate", lambda: TARGET_DATE)

    owner = _owner()
    client.force_login(owner)
    admitted = client.post(reverse("my-list"), {"symbol": candidate})

    assert admitted.status_code == 302
    assert TrackedSymbol.objects.filter(owner=owner, symbol=candidate).exists()
    TrackedSymbol.objects.filter(owner=owner, symbol=candidate).delete()

    no_sec_stages = {
        stage_name: stage
        for stage_name, stage in stages.items()
        if stage_name != "sec_fundamentals"
    }
    canonical_no_sec = refresh_verification_module.verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=no_sec_stages,
        sec_required=False,
    )
    JobRun.objects.filter(pk=parent.pk).update(
        details={
            **parent.details,
            "stages": no_sec_stages,
            "verification": canonical_no_sec,
        }
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        refresh_verification_module.replay_recorded_scheduled_refresh(parent)
    assert excinfo.value.reason_code == "recorded_parent_stages_invalid"

    rejected = client.post(reverse("my-list"), {"symbol": candidate})

    assert rejected.status_code == 400
    assert b"catalog evidence is unavailable or invalid" in rejected.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
def test_newer_orphan_catalog_does_not_influence_verified_bundle(
    client,
    tmp_path,
    settings,
) -> None:
    owner = _owner()
    _store_catalogs(tmp_path, settings)
    store = AssetStore(tmp_path)
    observed_at = timezone.now() + timedelta(minutes=1)
    payload = json.dumps(
        {
            "data": [_catalog_row("ORPHAN")],
            "count": 1,
            "status": "ok",
        }
    ).encode()
    stored = store.write_bytes("catalogs/orphan-newer.json", payload)
    register_asset(
        provider="twelve_data",
        kind="stock_catalog",
        subject="NASDAQ",
        stored=stored,
        retrieved_at=observed_at,
        available_at=observed_at,
    )
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "ORPHAN"})

    assert response.status_code == 400
    assert b"was not found in the latest verified" in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
def test_mixed_catalog_bundle_is_rejected(
    client,
    tmp_path,
    settings,
) -> None:
    owner = _owner()
    bundle = _store_catalogs(tmp_path, settings)
    store = AssetStore(tmp_path)
    observed_at = timezone.now() + timedelta(minutes=1)
    payload = json.dumps(
        {
            "data": [_catalog_row("MIXED", exchange="NYSE", mic_code="XNYS")],
            "count": 1,
            "status": "ok",
        }
    ).encode()
    stored = store.write_bytes("catalogs/nyse-other-vintage.json", payload)
    replacement = register_asset(
        provider="twelve_data",
        kind="stock_catalog",
        subject="NYSE",
        stored=stored,
        retrieved_at=observed_at,
        available_at=observed_at,
    )
    market = bundle["market"]
    assert isinstance(market, JobRun)
    details = dict(market.details)
    refs = list(details["catalog_refs"])
    ids = list(details["catalog_asset_ids"])
    refs[1] = asset_ref_for(replacement).to_json()
    ids[1] = str(replacement.id)
    details["catalog_refs"] = refs
    details["catalog_asset_ids"] = ids
    market.details = details
    market.save(update_fields=["details"])
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "MIXED"})

    assert response.status_code == 400
    assert b"catalog evidence is unavailable or invalid" in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
def test_partial_catalog_payload_is_rejected_for_admission(
    client,
    tmp_path,
    settings,
) -> None:
    owner = _owner()
    _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[_catalog_row("PARTIAL")],
        nasdaq_count=2,
    )
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "PARTIAL"})

    assert response.status_code == 400
    assert b"catalog evidence is unavailable or invalid" in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
def test_stale_verified_catalog_bundle_is_rejected_with_clear_message(
    client,
    tmp_path,
    settings,
) -> None:
    owner = _owner()
    _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[_catalog_row("STALE")],
        target_date=timezone.localdate() - timedelta(days=8),
    )
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "STALE"})

    assert response.status_code == 400
    assert b"stale (older than seven calendar days)" in response.content
    assert str(tmp_path).encode() not in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("parent_status", "verification_status"),
    [
        (JobRun.Status.FAILED, "verified"),
        (JobRun.Status.SUCCESS, "failed"),
    ],
)
def test_failed_or_unverified_refresh_parent_is_rejected(
    client,
    tmp_path,
    settings,
    parent_status: str,
    verification_status: str,
) -> None:
    owner = _owner()
    bundle = _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[_catalog_row("UNVERIFIED")],
        verification_status=verification_status,
    )
    parent = bundle["parent"]
    assert isinstance(parent, JobRun)
    if parent_status != JobRun.Status.SUCCESS:
        JobRun.objects.filter(pk=parent.pk).update(status=parent_status)
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "UNVERIFIED"})

    assert response.status_code == 400
    assert b"catalog evidence is unavailable" in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
def test_newer_unverified_parent_does_not_displace_fresh_verified_bundle(
    client,
    tmp_path,
    settings,
) -> None:
    owner = _owner()
    _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[_catalog_row("VERIFIED")],
        target_date=timezone.localdate() - timedelta(days=1),
    )
    _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[_catalog_row("UNTRUSTED")],
        target_date=timezone.localdate(),
        verification_status="failed",
    )
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "VERIFIED"})

    assert response.status_code == 302
    assert TrackedSymbol.objects.filter(owner=owner, symbol="VERIFIED").exists()


@pytest.mark.django_db
def test_newer_claimed_verified_parent_corruption_fails_closed(
    client,
    tmp_path,
    settings,
    monkeypatch,
) -> None:
    owner = _owner()
    genuine = _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[_catalog_row("VERIFIED")],
        target_date=timezone.localdate() - timedelta(days=1),
    )
    newer = _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[_catalog_row("CORRUPT")],
        target_date=timezone.localdate(),
    )
    genuine_parent = genuine["parent"]
    newer_parent = newer["parent"]
    assert isinstance(genuine_parent, JobRun)
    assert isinstance(newer_parent, JobRun)
    replay = watchlist_module.replay_recorded_scheduled_refresh
    replayed_ids: list[object] = []

    def fail_newest(parent: JobRun) -> ReplayedScheduledRefresh:
        replayed_ids.append(parent.pk)
        if parent.pk == newer_parent.pk:
            raise RefreshVerificationError(
                "recorded_verification_mismatch",
                "Synthetic claimed-verification corruption",
            )
        return replay(parent)

    monkeypatch.setattr(watchlist_module, "replay_recorded_scheduled_refresh", fail_newest)
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "VERIFIED"})

    assert response.status_code == 400
    assert b"catalog evidence is unavailable or invalid" in response.content
    assert replayed_ids == [newer_parent.pk]
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
def test_bad_full_catalog_reference_is_rejected(
    client,
    tmp_path,
    settings,
) -> None:
    owner = _owner()
    bundle = _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[_catalog_row("BADREF")],
    )
    market = bundle["market"]
    assert isinstance(market, JobRun)
    details = dict(market.details)
    refs = list(details["catalog_refs"])
    refs[0] = {**refs[0], "sha256": "f" * 64}
    details["catalog_refs"] = refs
    market.details = details
    market.save(update_fields=["details"])
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "BADREF"})

    assert response.status_code == 400
    assert b"catalog evidence is unavailable or invalid" in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
def test_wrong_market_child_target_relation_is_rejected(
    client,
    tmp_path,
    settings,
) -> None:
    owner = _owner()
    bundle = _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[_catalog_row("WRONGTARGET")],
    )
    market = bundle["market"]
    assert isinstance(market, JobRun)
    JobRun.objects.filter(pk=market.pk).update(target_date=market.target_date - timedelta(days=1))
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "WRONGTARGET"})

    assert response.status_code == 400
    assert b"catalog evidence is unavailable or invalid" in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("row_overrides", "expected"),
    [
        ({"exchange": "NYSE", "mic_code": "XNYS"}, b"catalog evidence is unavailable or invalid"),
        ({"mic_code": "XLON"}, b"reviewed NASDAQ or NYSE venues"),
    ],
)
def test_catalog_subject_exchange_and_mic_are_enforced(
    client,
    tmp_path,
    settings,
    row_overrides: dict[str, object],
    expected: bytes,
) -> None:
    owner = _owner()
    _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[_catalog_row("VENUE", **row_overrides)],
    )
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "VENUE"})

    assert response.status_code == 400
    assert expected in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
def test_catalog_symbol_requiring_higher_plan_is_rejected(
    client,
    tmp_path,
    settings,
    monkeypatch,
) -> None:
    owner = _owner()
    _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[_catalog_row("UPGRADE", access={"plan": "Ultra"})],
        plan="grow",
    )
    client.force_login(owner)
    monkeypatch.setattr(
        "stanstock.data.provider_policy.validate_provider_usage",
        lambda *args, **kwargs: pytest.fail("watchlist must not validate provider usage"),
    )
    monkeypatch.setattr(
        "stanstock.data.live_us.validate_provider_usage",
        lambda *args, **kwargs: pytest.fail("watchlist must not validate provider usage"),
    )

    response = client.post(reverse("my-list"), {"symbol": "UPGRADE"})

    assert response.status_code == 400
    assert b"installed Twelve Data plan does not provide access" in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("symbol", "nasdaq_rows", "expected"),
    [
        ("UNKNOWN", [], b"was not found"),
        (
            "DUPL",
            [_catalog_row("DUPL"), _catalog_row("DUPL", name="Duplicate Corp")],
            b"is ambiguous",
        ),
        ("ETF1", [_catalog_row("ETF1", type="ETF")], b"Only United States USD common stocks"),
        (
            "FUND",
            [_catalog_row("FUND", type="Mutual Fund")],
            b"Only United States USD common stocks",
        ),
        (
            "INTL",
            [_catalog_row("INTL", country="Canada")],
            b"Only United States USD common stocks",
        ),
        (
            "EURO",
            [_catalog_row("EURO", currency="EUR")],
            b"Only United States USD common stocks",
        ),
        (
            "LSE1",
            [_catalog_row("LSE1", exchange="LSE", mic_code="XLON")],
            b"catalog evidence is unavailable or invalid",
        ),
    ],
)
def test_rejects_unknown_ambiguous_or_unsupported_catalog_identity(
    client,
    tmp_path,
    settings,
    symbol: str,
    nasdaq_rows: list[object],
    expected: bytes,
) -> None:
    owner = _owner()
    _store_catalogs(tmp_path, settings, nasdaq_rows=nasdaq_rows)
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": symbol})

    assert response.status_code == 400
    assert expected in response.content
    assert not TrackedSymbol.objects.filter(owner=owner).exists()


@pytest.mark.django_db
def test_rejects_malformed_symbol_before_catalog_access(client, monkeypatch) -> None:
    owner = _owner()
    client.force_login(owner)
    monkeypatch.setattr(
        twelve_data,
        "parse_stock_catalog_references",
        lambda *args, **kwargs: pytest.fail("malformed symbol must fail before catalog access"),
    )

    response = client.post(reverse("my-list"), {"symbol": "../bad"})

    assert response.status_code == 400
    assert b"letters, numbers, periods, or hyphens" in response.content
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
def test_rejects_malformed_matching_catalog_row_path_free(
    client,
    tmp_path,
    settings,
) -> None:
    owner = _owner()
    malformed = _catalog_row("BROKEN")
    malformed.pop("currency")
    _store_catalogs(tmp_path, settings, nasdaq_rows=[malformed])
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "BROKEN"})

    assert response.status_code == 400
    assert b"catalog evidence is unavailable or invalid" in response.content
    assert str(tmp_path).encode() not in response.content
    assert b"catalogs/nasdaq.json" not in response.content


@pytest.mark.django_db
def test_rejects_stored_provider_error_payload_without_exposing_it(
    client,
    tmp_path,
    settings,
) -> None:
    owner = _owner()
    payload = {
        "status": "error",
        "code": 401,
        "message": "sensitive synthetic provider detail",
    }
    _store_catalogs(
        tmp_path,
        settings,
        nasdaq_payload=payload,
        nyse_payload=payload,
    )
    client.force_login(owner)

    response = client.post(reverse("my-list"), {"symbol": "ERROR"})

    assert response.status_code == 400
    assert b"catalog evidence is unavailable or invalid" in response.content
    assert b"sensitive synthetic provider detail" not in response.content
    assert str(tmp_path).encode() not in response.content


@pytest.mark.django_db
def test_missing_or_corrupt_catalog_evidence_fails_honestly_and_path_free(
    client,
    tmp_path,
    settings,
) -> None:
    owner = _owner()
    settings.DATA_DIR = tmp_path
    client.force_login(owner)

    missing = client.post(reverse("my-list"), {"symbol": "ABSENT"})
    assert missing.status_code == 400
    assert b"catalog evidence is unavailable" in missing.content

    bundle = _store_catalogs(
        tmp_path,
        settings,
        nasdaq_rows=[_catalog_row("BROKEN")],
    )
    assets = bundle["assets"]
    assert isinstance(assets, tuple)
    corrupt_asset = assets[0]
    assert isinstance(corrupt_asset, DataAsset)
    AssetStore(tmp_path).resolve(corrupt_asset.relative_path).write_bytes(b"corrupt")
    corrupt = client.post(reverse("my-list"), {"symbol": "BROKEN"})

    assert corrupt.status_code == 400
    assert b"catalog evidence is unavailable or invalid" in corrupt.content
    assert str(tmp_path).encode() not in corrupt.content
    assert b"catalogs/" not in corrupt.content


@pytest.mark.django_db
def test_rejected_post_without_evidence_does_not_create_data_directory(
    client,
    tmp_path: Path,
    settings,
) -> None:
    owner = _owner()
    missing_root = tmp_path / "not-created"
    settings.DATA_DIR = missing_root
    client.force_login(owner)
    assert not missing_root.exists()

    response = client.post(reverse("my-list"), {"symbol": "NOEVIDENCE"})

    assert response.status_code == 400
    assert b"catalog evidence is unavailable" in response.content
    assert not missing_root.exists()
    assert not TrackedSymbol.objects.exists()


@pytest.mark.django_db
def test_duplicate_add_is_idempotent_and_reports_existing(client) -> None:
    owner = _owner()
    _listing("ONCE")
    client.force_login(owner)

    first = client.post(reverse("my-list"), {"symbol": "once"})
    second = client.post(reverse("my-list"), {"symbol": " ONCE "}, follow=True)

    assert first.status_code == 302
    assert second.status_code == 200
    assert TrackedSymbol.objects.filter(owner=owner, symbol="ONCE").count() == 1
    assert b"ONCE is already in My list." in second.content


@pytest.mark.django_db(transaction=True)
def test_concurrent_equivalent_adds_create_exactly_one_tracked_symbol() -> None:
    if connection.vendor != "postgresql":
        pytest.skip("PostgreSQL-specific independent-connection concurrency regression")

    owner = _owner()
    _listing("RACE")
    insert_barrier = Barrier(2)

    def add(raw_symbol: str) -> tuple[bool, int]:
        close_old_connections()
        thread_connection = connections["default"]
        with thread_connection.cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            backend_pid = int(cursor.fetchone()[0])

        def synchronize_insert(execute, sql, params, many, context):
            if sql.lstrip().upper().startswith('INSERT INTO "PORTFOLIO_TRACKEDSYMBOL"'):
                insert_barrier.wait(timeout=10)
            return execute(sql, params, many, context)

        try:
            thread_owner = get_user_model().objects.get(pk=owner.pk)
            with thread_connection.execute_wrapper(synchronize_insert):
                _preference, created = watchlist_module.add_tracked_symbol(
                    owner=thread_owner,
                    raw_symbol=raw_symbol,
                )
            return created, backend_pid
        finally:
            thread_connection.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(add, (" race ", "RaCe")))

    assert sorted(created for created, _backend_pid in outcomes) == [False, True]
    assert len({backend_pid for _created, backend_pid in outcomes}) == 2
    assert TrackedSymbol.objects.filter(owner=owner, symbol="RACE").count() == 1


@pytest.mark.django_db
def test_add_does_not_swallow_unrelated_integrity_error(monkeypatch) -> None:
    owner = _owner()
    _listing("BROKENADD")

    def fail_create(*args: object, **kwargs: object) -> tuple[TrackedSymbol, bool]:
        raise IntegrityError("synthetic unrelated integrity failure")

    monkeypatch.setattr(TrackedSymbol.objects, "get_or_create", fail_create)

    with pytest.raises(IntegrityError, match="synthetic unrelated integrity failure"):
        watchlist_module.add_tracked_symbol(owner=owner, raw_symbol="brokenadd")


@pytest.mark.django_db
def test_remove_is_post_only_owner_scoped_and_csrf_protected() -> None:
    owner = _owner()
    other = _owner("other")
    own = TrackedSymbol.objects.create(owner=owner, symbol="OWN")
    private = TrackedSymbol.objects.create(owner=other, symbol="PRIVATE")
    csrf_client = Client(enforce_csrf_checks=True)
    csrf_client.force_login(owner)
    own_url = reverse("tracked-symbol-delete", args=[own.pk])

    get_response = csrf_client.get(own_url)
    missing_csrf = csrf_client.post(own_url)
    page = csrf_client.get(reverse("my-list"))
    token = page.cookies["csrftoken"].value
    cross_user = csrf_client.post(
        reverse("tracked-symbol-delete", args=[private.pk]),
        {"csrfmiddlewaretoken": token},
    )
    removed = csrf_client.post(own_url, {"csrfmiddlewaretoken": token})

    assert get_response.status_code == 405
    assert missing_csrf.status_code == 403
    assert cross_user.status_code == 404
    assert removed.status_code == 302
    assert not TrackedSymbol.objects.filter(pk=own.pk).exists()
    assert TrackedSymbol.objects.filter(pk=private.pk, owner=other).exists()


@pytest.mark.django_db
def test_renders_listing_price_analysis_and_explicit_unavailable_states(client) -> None:
    owner = _owner()
    covered = _listing("COVER")
    unavailable = _listing("NOPRICE")
    TrackedSymbol.objects.create(owner=owner, symbol=covered.provider_symbol)
    TrackedSymbol.objects.create(owner=owner, symbol=unavailable.provider_symbol)
    TrackedSymbol.objects.create(owner=owner, symbol="CATALOG")
    now = timezone.now()
    asset = DataAsset.objects.create(
        provider="synthetic_demo",
        kind="price_history",
        subject=covered.provider_symbol,
        relative_path="tests/watchlist-cover.parquet",
        sha256="c" * 64,
        retrieved_at=now,
        available_at=now,
    )
    LatestMarketData.objects.create(
        listing=covered,
        observed_at=now,
        session_date=date(2026, 9, 10),
        close=Decimal("123.45"),
        previous_close=None,
        volume=None,
        source_asset=asset,
    )
    analysis = _analysis(covered)
    client.force_login(owner)

    response = client.get(reverse("my-list"))
    content = response.content.decode()

    assert response.status_code == 200
    assert "123.45 USD" in content
    assert "Session Sept. 10, 2026" in content
    assert "BUY" in content
    assert "Score 88.50/100" in content
    assert "No persisted price" in content
    assert "No stored analysis yet" in content
    assert "Not in the selected analysis universe" in content
    assert "Selected analysis universe" in content
    assert "Target date" in content
    assert "2026-09-10" in content
    assert analysis.run.data_cutoff.isoformat() in content
    assert "Research-grade reconstruction" in content
    catalog_state = next(
        item for item in response.context["tracked_symbols"] if item.preference.symbol == "CATALOG"
    )
    assert catalog_state.resolution_status == "no_local_listing"
    assert catalog_state.listing is None
    assert catalog_state.market_data is None
    assert catalog_state.analysis is None
    assert reverse("stock-detail", args=[covered.pk]) in content


@pytest.mark.django_db
def test_analysis_selector_query_count_is_constant_for_one_and_forty_runs(
    django_assert_num_queries,
) -> None:
    listing = _listing("BOUNDED")
    provider_analysis = _analysis(
        listing,
        target_date=date(2026, 7, 1),
        provider="twelve_data",
    )

    with django_assert_num_queries(1):
        selected_one = watchlist_module.selected_watchlist_analysis_run()
    assert selected_one is not None
    assert selected_one.pk == provider_analysis.run_id

    for offset in range(1, 40):
        _analysis(
            listing,
            target_date=date(2026, 7, 1) + timedelta(days=offset),
            provider="synthetic_demo",
        )

    with django_assert_num_queries(1):
        selected_forty = watchlist_module.selected_watchlist_analysis_run()
    assert AnalysisRun.objects.filter(status="complete").count() == 40
    assert selected_forty is not None
    assert selected_forty.pk == provider_analysis.run_id


@pytest.mark.django_db
def test_analysis_selection_prefers_latest_target_then_observed_grade(client) -> None:
    owner = _owner()
    listing = _listing("SELECT")
    TrackedSymbol.objects.create(owner=owner, symbol=listing.provider_symbol)
    selected = _analysis(
        listing,
        target_date=date(2026, 9, 10),
        grade=UniverseSnapshot.Grade.OBSERVED,
        generated_at=datetime(2026, 9, 10, 21, tzinfo=UTC),
        data_cutoff=datetime(2026, 9, 10, 20, tzinfo=UTC),
        provider="twelve_data",
        recommendation=Recommendation.BUY,
        score=Decimal("91.00"),
    )
    _analysis(
        listing,
        target_date=date(2026, 9, 10),
        grade=UniverseSnapshot.Grade.RESEARCH,
        generated_at=datetime(2026, 9, 11, 8, tzinfo=UTC),
        data_cutoff=datetime(2026, 9, 10, 20, tzinfo=UTC),
        provider="twelve_data",
        recommendation=Recommendation.AVOID,
        score=Decimal("15.00"),
    )
    _analysis(
        listing,
        target_date=date(2026, 9, 9),
        grade=UniverseSnapshot.Grade.RESEARCH,
        generated_at=datetime(2026, 9, 12, 8, tzinfo=UTC),
        data_cutoff=datetime(2026, 9, 9, 20, tzinfo=UTC),
        provider="twelve_data",
        recommendation=Recommendation.AVOID,
        score=Decimal("10.00"),
    )
    client.force_login(owner)

    response = client.get(reverse("my-list"))
    content = response.content.decode()

    assert response.status_code == 200
    assert response.context["selected_analysis_run"].pk == selected.run_id
    assert "2026-09-10" in content
    assert "2026-09-10T20:00:00+00:00" in content
    assert "Observed at run time" in content
    assert "BUY" in content
    assert "Score 91.00/100" in content
    assert "AVOID" not in content


@pytest.mark.django_db
def test_research_only_analysis_discloses_exact_context(client) -> None:
    owner = _owner()
    listing = _listing("RESEARCH")
    TrackedSymbol.objects.create(owner=owner, symbol=listing.provider_symbol)
    analysis = _analysis(
        listing,
        target_date=date(2026, 9, 8),
        grade=UniverseSnapshot.Grade.RESEARCH,
        generated_at=datetime(2026, 9, 10, 9, tzinfo=UTC),
        data_cutoff=datetime(2026, 9, 8, 20, 15, tzinfo=UTC),
        provider="twelve_data",
    )
    client.force_login(owner)

    response = client.get(reverse("my-list"))
    content = response.content.decode()

    assert response.status_code == 200
    assert response.context["selected_analysis_run"].pk == analysis.run_id
    assert "2026-09-08" in content
    assert "2026-09-08T20:15:00+00:00" in content
    assert "Research-grade reconstruction" in content
    assert "not labeled" in content
    assert "as current" in content


@pytest.mark.django_db
@pytest.mark.parametrize("coverage_loss", ["inactive", "ineligible"])
def test_formerly_eligible_listing_renders_neutral_no_local_state_after_coverage_loss(
    client,
    coverage_loss: str,
) -> None:
    owner = _owner()
    listing = _listing("FORMER")
    client.force_login(owner)
    added = client.post(reverse("my-list"), {"symbol": "FORMER"})
    assert added.status_code == 302

    if coverage_loss == "inactive":
        listing.is_active = False
        listing.save(update_fields=["is_active"])
    else:
        listing.security.security_type = Security.SecurityType.ETF
        listing.security.save(update_fields=["security_type"])

    response = client.get(reverse("my-list"))
    content = response.content.decode()
    state = response.context["tracked_symbols"][0]

    assert response.status_code == 200
    assert state.resolution_status == "no_local_listing"
    assert state.listing is None
    assert state.market_data is None
    assert state.analysis is None
    assert "No active eligible local listing is currently available." in content
    assert "Accepted from stored catalog evidence" not in content
    assert "FORMER Corp" not in content
    assert reverse("stock-detail", args=[listing.pk]) not in content


@pytest.mark.django_db
def test_active_ineligible_alias_collision_renders_ambiguous_without_listing_claims(client) -> None:
    owner = _owner()
    original = _listing("COLLIDE")
    client.force_login(owner)
    added = client.post(reverse("my-list"), {"symbol": "collide"})
    assert added.status_code == 302

    now = timezone.now()
    asset = DataAsset.objects.create(
        provider="synthetic_demo",
        kind="price_history",
        subject=original.provider_symbol,
        relative_path="tests/watchlist-collision.parquet",
        sha256="d" * 64,
        retrieved_at=now,
        available_at=now,
    )
    LatestMarketData.objects.create(
        listing=original,
        observed_at=now,
        session_date=date(2026, 9, 10),
        close=Decimal("777.77"),
        previous_close=None,
        volume=None,
        source_asset=asset,
    )
    _analysis(original, provider="synthetic_demo")
    collision = _listing("OTHER", security_type=Security.SecurityType.ETF)
    collision.provider_symbol = original.provider_symbol
    collision.save(update_fields=["provider_symbol"])

    response = client.get(reverse("my-list"))
    content = response.content.decode()
    state = response.context["tracked_symbols"][0]

    assert response.status_code == 200
    assert state.resolution_status == "ambiguous_listing"
    assert state.listing is None
    assert state.market_data is None
    assert state.analysis is None
    assert "Ambiguous local listing" in content
    assert "No unique eligible local listing is available." in content
    assert "Accepted from stored catalog evidence" not in content
    assert "COLLIDE Corp" not in content
    assert "OTHER Corp" not in content
    assert reverse("stock-detail", args=[original.pk]) not in content
    assert reverse("stock-detail", args=[collision.pk]) not in content
    assert "777.77 USD" not in content
    assert "BUY" not in content
    assert "Score 88.50/100" not in content


@pytest.mark.django_db
def test_add_remove_never_calls_provider_or_mutates_research_or_portfolio_state(
    client,
    tmp_path,
    settings,
    monkeypatch,
) -> None:
    owner = _owner()
    _store_catalogs(tmp_path, settings, nasdaq_rows=[_catalog_row("LOCALONLY")])
    linked_listing = _listing("LINKED")
    analysis = _analysis(linked_listing)
    now = timezone.now()
    Prediction.objects.create(
        analysis=analysis,
        listing=linked_listing,
        generated_at=now,
        target_date=analysis.run.target_date,
        horizon=Prediction.Horizon.SHORT,
        price_at_prediction=analysis.current_price,
        bear_return=Decimal("-0.05"),
        base_return=Decimal("0.04"),
        bull_return=Decimal("0.12"),
        confidence=analysis.confidence,
        confidence_status="heuristic",
        recommendation=analysis.recommendation,
        overall_score=analysis.overall_score,
        model_version="watchlist-test-v1",
        config_hash=analysis.run.config_hash,
        data_cutoff=now,
        code_revision="test",
    )
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Unrelated holdings",
        base_currency="USD",
    )
    PortfolioHolding.objects.create(
        portfolio=portfolio,
        listing=linked_listing,
        quantity=Decimal("2"),
        average_cost=Decimal("100"),
    )
    client.force_login(owner)

    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail("watchlist preference management must not access provider or quota services")

    monkeypatch.setattr(twelve_data, "fetch", forbidden)
    monkeypatch.setattr(twelve_data, "fetch_stock_catalog", forbidden)
    monkeypatch.setattr(twelve_data, "resolve_api_key", forbidden)
    monkeypatch.setattr(twelve_data, "read_twelve_data_api_key", forbidden)
    monkeypatch.setattr(provider_credentials, "read_twelve_data_api_key", forbidden)
    monkeypatch.setattr(ProviderCreditBudget, "preflight", forbidden)
    monkeypatch.setattr(ProviderCreditBudget, "consume", forbidden)
    monkeypatch.setattr(
        "stanstock.data.provider_policy.validate_provider_usage",
        forbidden,
    )
    monkeypatch.setattr("stanstock.data.live_us.validate_provider_usage", forbidden)

    protected_models = (
        JobRun,
        Company,
        Security,
        Universe,
        UniverseSnapshot,
        UniverseMembership,
        Listing,
        StockAnalysis,
        Prediction,
        Portfolio,
        PortfolioHolding,
        DataAsset,
        ProviderRecord,
    )
    before = {model: model.objects.count() for model in protected_models}

    added = client.post(reverse("my-list"), {"symbol": "LOCALONLY"})
    preference = TrackedSymbol.objects.get(owner=owner, symbol="LOCALONLY")
    removed = client.post(reverse("tracked-symbol-delete", args=[preference.pk]))
    linked_added = client.post(reverse("my-list"), {"symbol": "LINKED"})
    linked_preference = TrackedSymbol.objects.get(owner=owner, symbol="LINKED")
    linked_removed = client.post(reverse("tracked-symbol-delete", args=[linked_preference.pk]))

    assert added.status_code == 302
    assert removed.status_code == 302
    assert linked_added.status_code == 302
    assert linked_removed.status_code == 302
    assert not TrackedSymbol.objects.filter(owner=owner).exists()
    assert {model: model.objects.count() for model in protected_models} == before


@pytest.mark.django_db
def test_my_list_markup_is_semantic_responsive_and_marks_navigation_active(client) -> None:
    owner = _owner()
    TrackedSymbol.objects.create(owner=owner, symbol="MARKUP")
    client.force_login(owner)

    response = client.get(reverse("my-list"))
    content = response.content.decode()

    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in content
    assert '<a href="/my-list" aria-current="page">My list</a>' in content
    assert '<div class="table-scroll" tabindex="0" aria-label="My tracked symbols">' in content
    assert '<th scope="col">Symbol</th>' in content
    assert "<table" in content

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from uuid import uuid4

import numpy as np
import polars as pl
import pytest

from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import (
    Company,
    DataAsset,
    Listing,
    Region,
    Security,
    UniverseSnapshot,
)
from stanstock.data.research_product import capture_product_intake, materialize_product_membership
from stanstock.research.price_product import SourceExecutionBinding
from stanstock.research.price_product_config import (
    default_price_product_config_path,
    load_price_product_config,
)
from stanstock.research.product_pipeline import (
    issue_price_product_snapshot,
    price_product_input_from_verified_source_window,
    select_product_source,
    verified_source_window_payload,
    verify_price_product_output,
)
from stanstock.research.refresh_evidence import lookup_manifest
from stanstock.research.service import analyze_snapshot

pytestmark = pytest.mark.django_db


def test_product_pipeline_persists_exact_five_rows_and_replays(tmp_path) -> None:
    store = AssetStore(tmp_path)
    target = date(2026, 9, 11)
    decision_time = datetime(2026, 9, 12, tzinfo=UTC)
    listing, snapshot = _listing_and_snapshot(target, store)
    _price_assets(store, "SYN", target, decision_time)
    _price_assets(store, "SPY", target, decision_time)

    results = issue_price_product_snapshot(
        universe_snapshot=snapshot,
        decision_time=decision_time,
        target_date=target,
        provider="synthetic_demo",
        benchmark_subject="SPY",
        source_execution=SourceExecutionBinding(mode="synthetic_demo", evidence_grade="research"),
        store=store,
        code_revision="test-revision",
        issued_on_time=False,
    )

    assert len(results) == 1
    result = results[0]
    assert len(result.predictions) == 5
    assert {(row.method_version, row.evidence_role, row.horizon) for row in result.predictions} == {
        ("us-relative-momentum-v1", "decision", "6m"),
        ("us-price-fhs-v1", "advisory", "6m"),
        ("us-price-fhs-v1", "advisory", "12m"),
        ("us-price-fhs-v1", "advisory", "3y"),
        ("us-price-fhs-v1", "advisory", "5y"),
    }
    assert len({row.model_version for row in result.predictions}) == 5
    assert all(row.probability_positive is None for row in result.predictions)
    assert lookup_manifest(result.run.id).count == 1
    verify_price_product_output(run=result.run, store=store, replay=True)


def test_product_pipeline_rejects_tampered_physical_source(tmp_path) -> None:
    store = AssetStore(tmp_path)
    target = date(2026, 9, 11)
    decision_time = datetime(2026, 9, 12, tzinfo=UTC)
    _listing, snapshot = _listing_and_snapshot(target, store)
    stock = _price_assets(store, "SYN", target, decision_time)
    _price_assets(store, "SPY", target, decision_time)
    result = issue_price_product_snapshot(
        universe_snapshot=snapshot,
        decision_time=decision_time,
        target_date=target,
        provider="synthetic_demo",
        benchmark_subject="SPY",
        source_execution=SourceExecutionBinding(mode="synthetic_demo", evidence_grade="research"),
        store=store,
        code_revision="test-revision",
        issued_on_time=False,
    )[0]
    store.resolve(stock.relative_path).write_bytes(b"tampered")
    with pytest.raises(Exception, match="checksum|corrupt"):
        verify_price_product_output(run=result.run, store=store)


def test_public_analyze_snapshot_dispatches_product_writer(tmp_path) -> None:
    store = AssetStore(tmp_path)
    target = date(2026, 9, 11)
    decision_time = datetime(2026, 9, 12, tzinfo=UTC)
    _listing, snapshot = _listing_and_snapshot(target, store)
    _price_assets(store, "SYN", target, decision_time)
    _price_assets(store, "SPY", target, decision_time)

    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=decision_time,
        target_date=target,
        issued_on_time=False,
        provider="synthetic_demo",
        benchmark_subject="SPY",
        store=store,
        config_path=default_price_product_config_path(),
    )

    assert len(results) == 1
    assert len(results[0].predictions) == 5
    assert results[0].run.config_version == "research-product-v1"


def test_verified_source_window_round_trips_without_asset_reselection(tmp_path) -> None:
    store = AssetStore(tmp_path)
    target = date(2026, 9, 11)
    decision_time = datetime(2026, 9, 12, tzinfo=UTC)
    listing, _snapshot = _listing_and_snapshot(target, store)
    _price_assets(store, "SYN", target, decision_time)
    _price_assets(store, "SPY", target, decision_time)
    source = select_product_source(
        listing=listing,
        target_date=target,
        decision_time=decision_time,
        provider="synthetic_demo",
        benchmark_subject="SPY",
        source_execution=SourceExecutionBinding(mode="synthetic_demo", evidence_grade="research"),
        store=store,
        config=load_price_product_config(),
    )

    assert (
        price_product_input_from_verified_source_window(verified_source_window_payload(source))
        == source.product_input
    )


def test_intake_capture_reuses_original_saved_set(tmp_path) -> None:
    store = AssetStore(tmp_path)
    target = date(2026, 9, 11)
    now = datetime(2026, 9, 12, tzinfo=UTC)
    listing, _snapshot = _listing_and_snapshot(target, store)
    other_company = Company.objects.create(name="Other", country="US")
    other = Listing.objects.create(
        security=Security.objects.create(company=other_company),
        ticker="OTHER",
        provider_symbol="OTHER",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    first = capture_product_intake(
        target_date=target,
        evidence_grade="research",
        issuance_key="initial",
        owner_id="owner-1",
        entitlement_identity="entitlement-v1",
        policy_identity="research-product-v1",
        core_listings=[listing],
        saved_listings=[other],
        captured_at=now,
        store=store,
    )
    retry = capture_product_intake(
        target_date=target,
        evidence_grade="research",
        issuance_key="initial",
        owner_id="owner-1",
        entitlement_identity="entitlement-v1",
        policy_identity="research-product-v1",
        core_listings=[listing],
        saved_listings=[other],
        captured_at=now,
        store=store,
    )
    assert retry.asset.id == first.asset.id
    assert retry.candidate_listing_ids == (listing.id, other.id)
    retry_after_preference_change = capture_product_intake(
        target_date=target,
        evidence_grade="research",
        issuance_key="initial",
        owner_id="owner-1",
        entitlement_identity="entitlement-v1",
        policy_identity="research-product-v1",
        core_listings=[listing],
        saved_listings=[],
        captured_at=now,
        store=store,
    )
    assert retry_after_preference_change.candidate_listing_ids == (listing.id, other.id)


def _listing_and_snapshot(target: date, store: AssetStore) -> tuple[Listing, UniverseSnapshot]:
    company = Company.objects.create(name="Synthetic", country="US")
    security = Security.objects.create(
        company=company, security_type=Security.SecurityType.COMMON_STOCK
    )
    listing = Listing.objects.create(
        security=security,
        ticker="SYN",
        provider_symbol="SYN",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    captured_at = datetime(2026, 9, 12, tzinfo=UTC)
    intake = capture_product_intake(
        target_date=target,
        evidence_grade="research",
        issuance_key=uuid4().hex,
        owner_id="test-owner",
        entitlement_identity="test-entitlement",
        policy_identity="research-product-v1",
        core_listings=[listing],
        saved_listings=[],
        captured_at=captured_at,
        store=store,
    )
    snapshot = materialize_product_membership(
        intake=intake,
        qualified_listing_ids=[listing.id],
        candidate_states={listing.id: "admitted"},
        captured_at=captured_at,
        store=store,
    )
    return listing, snapshot


def _price_assets(
    store: AssetStore, subject: str, target: date, decision_time: datetime
) -> DataAsset:
    from exchange_calendars import get_calendar  # type: ignore[import-untyped]

    calendar = get_calendar("XNYS")
    session = calendar.date_to_session(target, direction="none")
    sessions = [item.date() for item in calendar.sessions_window(session, -757)]
    offsets = np.arange(len(sessions), dtype=float)
    phase = 0.4 if subject == "SYN" else 0.1
    closes = (20.0 if subject == "SYN" else 100.0) * np.exp(
        0.0006 * offsets + 0.008 * np.sin(offsets / 13 + phase)
    )
    frame = pl.DataFrame(
        {"date": sessions, "close": closes, "volume": [1_000_000] * len(sessions)},
        schema_overrides={"date": pl.Date, "close": pl.Float64, "volume": pl.Int64},
    )
    raw_payload = {
        "schema": "research-product-synthetic-prices@1",
        "subject": subject,
        "currency": "USD",
        "adjustment": "splits",
        "volume_adjustment_compatible": True,
        "values": [
            {"date": row["date"].isoformat(), "close": row["close"], "volume": row["volume"]}
            for row in frame.iter_rows(named=True)
        ],
    }
    raw = register_asset(
        provider="synthetic_demo",
        kind="raw_price_history",
        subject=subject,
        stored=store.write_bytes(f"raw/{subject}.json", json.dumps(raw_payload).encode()),
        retrieved_at=decision_time,
        available_at=decision_time,
    )
    stored = store.write_frame(f"prices/{subject}.parquet", frame)
    return register_asset(
        provider="synthetic_demo",
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=decision_time,
        available_at=decision_time,
        period_start=sessions[0],
        period_end=sessions[-1],
        metadata={
            "currency": "USD",
            "adjustment": "splits",
            "volume_adjustment_compatible": True,
            "raw_asset_id": str(raw.id),
            "raw_sha256": raw.sha256,
        },
    )

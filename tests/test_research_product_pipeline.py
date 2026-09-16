from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import polars as pl
import pytest

from stanstock.core.verification_types import RefreshVerificationError
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
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.price_product import SourceExecutionBinding, calculate_price_product
from stanstock.research.price_product_config import (
    default_price_product_config_path,
    load_price_product_config,
)
from stanstock.research.product_pipeline import (
    CALCULATION_ARTIFACT_KIND,
    _recorded_decimal_matches_row,
    issue_price_product_snapshot,
    price_product_input_from_verified_source_window,
    select_product_source,
    verified_source_window_payload,
    verify_price_product_output,
)
from stanstock.research.refresh_evidence import (
    ANALYSIS_RUN_FIELDS,
    ANALYSIS_RUN_MODEL,
    PREDICTION_FIELDS,
    PREDICTION_MODEL,
    STOCK_ANALYSIS_FIELDS,
    STOCK_ANALYSIS_MODEL,
    lookup_manifest,
    model_row_values,
    parse_manifest_envelope,
    row_digest,
)
from stanstock.research.service import analyze_snapshot

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize(
    ("field", "recorded_field"),
    [
        pytest.param("bear_return", "lower", id="bear-lower"),
        pytest.param("base_return", "median", id="base-median"),
        pytest.param("bull_return", "upper", id="bull-upper"),
    ],
)
@pytest.mark.parametrize(
    ("actual", "recorded", "present", "expected"),
    [
        pytest.param(Decimal("0.0000"), "0.0000", True, True, id="positive-zero-exact"),
        pytest.param(Decimal("-0.0000"), "-0.0000", True, True, id="negative-zero-exact"),
        pytest.param(
            Decimal("0.0000"),
            "-0.0000",
            True,
            True,
            id="positive-zero-artifact-negative",
        ),
        pytest.param(
            Decimal("-0.0000"),
            "0.0000",
            True,
            True,
            id="negative-zero-artifact-positive",
        ),
        pytest.param(Decimal("0.0000"), "0.000", True, False, id="positive-zero-wrong-scale"),
        pytest.param(Decimal("-0.0000"), "-0.000", True, False, id="negative-zero-wrong-scale"),
        pytest.param(
            Decimal("0.0000"),
            "-0.000",
            True,
            False,
            id="positive-zero-opposite-wrong-scale",
        ),
        pytest.param(
            Decimal("-0.0000"),
            "0.000",
            True,
            False,
            id="negative-zero-opposite-wrong-scale",
        ),
        pytest.param(Decimal("0.1234"), "0.1234", True, True, id="positive-nonzero-exact"),
        pytest.param(Decimal("-0.1234"), "-0.1234", True, True, id="negative-nonzero-exact"),
        pytest.param(Decimal("0.0001"), "0.0001", True, True, id="positive-quantum-exact"),
        pytest.param(Decimal("-0.0001"), "-0.0001", True, True, id="negative-quantum-exact"),
        pytest.param(Decimal("0.1234"), "-0.1234", True, False, id="positive-nonzero-opposite"),
        pytest.param(Decimal("-0.1234"), "0.1234", True, False, id="negative-nonzero-opposite"),
        pytest.param(Decimal("0.1234"), "0.12340", True, False, id="nonzero-wrong-scale"),
        pytest.param(Decimal("0.00001"), "0.0000", True, False, id="tiny-positive-not-zero"),
        pytest.param(Decimal("-0.00001"), "-0.0000", True, False, id="tiny-negative-not-zero"),
        pytest.param(None, "0.0000", True, False, id="null-row"),
        pytest.param(Decimal("0.0000"), None, True, False, id="null-recorded"),
        pytest.param(Decimal("0.0000"), None, False, False, id="missing-recorded-key"),
        pytest.param(Decimal("0.0000"), True, True, False, id="bool-recorded"),
        pytest.param(Decimal("0.0000"), 0, True, False, id="integer-recorded"),
        pytest.param(Decimal("0.0000"), 0.0, True, False, id="float-recorded"),
        pytest.param(Decimal("0.0000"), Decimal("0.0000"), True, False, id="decimal-recorded"),
        pytest.param(Decimal("0.0000"), "", True, False, id="blank-recorded"),
        pytest.param(Decimal("0.0000"), "not-a-decimal", True, False, id="malformed-recorded"),
        pytest.param(Decimal("0.0000"), "NaN", True, False, id="nan-recorded"),
        pytest.param(Decimal("0.0000"), "Infinity", True, False, id="infinity-recorded"),
        pytest.param(Decimal("0.0000"), "-Infinity", True, False, id="negative-infinity-recorded"),
        pytest.param(Decimal("NaN"), "NaN", True, False, id="nan-row"),
        pytest.param(Decimal("Infinity"), "Infinity", True, False, id="infinity-row"),
        pytest.param(Decimal("-Infinity"), "-Infinity", True, False, id="negative-infinity-row"),
    ],
)
def test_recorded_decimal_matches_row_allows_only_same_scale_zero_sign_variants(
    field: str,
    recorded_field: str,
    actual: Decimal | None,
    recorded: object,
    present: bool,
    expected: bool,
) -> None:
    row = {field: actual}
    ledger = {recorded_field: recorded} if present else {}
    assert _recorded_decimal_matches_row(row[field], ledger.get(recorded_field)) is expected


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


def test_signed_zero_projection_issues_and_replays(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A natural signed ledger zero cannot diverge from its reread manifest."""
    store = AssetStore(tmp_path)
    target = date(2026, 9, 11)
    decision_time = datetime(2026, 9, 12, tzinfo=UTC)
    listing, snapshot = _listing_and_snapshot(target, store)
    _price_assets(store, "SYN", target, decision_time, signed_zero_edge=True)
    _price_assets(store, "SPY", target, decision_time, signed_zero_edge=True)
    source_execution = SourceExecutionBinding(mode="synthetic_demo", evidence_grade="research")
    config = load_price_product_config()
    blocked_external_access = Mock(
        side_effect=AssertionError(
            "signed-zero issuance and replay must not resolve credentials or use the network"
        )
    )
    monkeypatch.setattr("httpx.Client.send", blocked_external_access)
    monkeypatch.setattr(
        "stanstock.data.providers.twelve_data.resolve_api_key",
        blocked_external_access,
    )
    selection = select_product_source(
        listing=listing,
        target_date=target,
        decision_time=decision_time,
        provider="synthetic_demo",
        benchmark_subject="SPY",
        source_execution=source_execution,
        store=store,
        config=config,
    )
    pre_persistence = calculate_price_product(selection.product_input, config=config)
    signed_zero_horizons = {
        projection.horizon
        for projection in pre_persistence.forecast.projections
        if (
            projection.ledger_returns is not None
            and projection.ledger_returns.lower.is_zero()
            and projection.ledger_returns.lower.is_signed()
        )
    }
    assert signed_zero_horizons

    issued = issue_price_product_snapshot(
        universe_snapshot=snapshot,
        decision_time=decision_time,
        target_date=target,
        provider="synthetic_demo",
        benchmark_subject="SPY",
        source_execution=source_execution,
        store=store,
        code_revision="test-revision",
        issued_on_time=False,
        config=config,
    )

    assert len(issued) == 1
    run = AnalysisRun.objects.get(pk=issued[0].run.id)
    analyses = list(StockAnalysis.objects.filter(run=run).order_by("listing_id"))
    predictions = list(Prediction.objects.filter(analysis__run=run).order_by("id"))
    assert len(analyses) == 1
    assert len(predictions) == 5
    assert {prediction.listing_id for prediction in predictions} == {listing.id}
    assert DataAsset.objects.filter(subject=str(run.id)).count() == 2
    run_assets = list(DataAsset.objects.filter(subject=str(run.id)).order_by("kind"))
    assert {asset.kind for asset in run_assets} == {
        CALCULATION_ARTIFACT_KIND,
        "analysis_output_manifest",
    }
    calculation_asset = next(
        asset for asset in run_assets if asset.kind == CALCULATION_ARTIFACT_KIND
    )
    calculation_bytes = store.read_bytes(calculation_asset.relative_path)
    calculation_document = json.loads(calculation_bytes)
    recorded_projections = {
        projection["horizon"]: projection
        for projection in calculation_document["result"]["forecast"]["projections"]
    }
    assert b'"lower":"-0.0000"' in calculation_bytes
    assert all(
        recorded_projections[horizon]["ledger_returns"]["lower"] == "-0.0000"
        for horizon in signed_zero_horizons
    )
    asset_bytes_and_sha = {
        asset.id: (store.read_bytes(asset.relative_path), asset.sha256) for asset in run_assets
    }
    signed_zero_rows = [
        prediction
        for prediction in predictions
        if (
            prediction.horizon in signed_zero_horizons
            and prediction.evidence_role == Prediction.EvidenceRole.ADVISORY
        )
    ]
    assert signed_zero_rows
    assert all(
        prediction.bear_return is not None
        and prediction.bear_return.is_zero()
        and not prediction.bear_return.is_signed()
        for prediction in signed_zero_rows
    )
    for prediction in predictions:
        if prediction.evidence_role != Prediction.EvidenceRole.ADVISORY:
            continue
        ledger = recorded_projections[prediction.horizon]["ledger_returns"]
        assert isinstance(ledger, dict)
        assert _recorded_decimal_matches_row(prediction.bear_return, ledger["lower"])
        assert _recorded_decimal_matches_row(prediction.base_return, ledger["median"])
        assert _recorded_decimal_matches_row(prediction.bull_return, ledger["upper"])

    manifest = lookup_manifest(run.id)
    assert manifest.count == 1
    assert manifest.asset is not None
    manifest_run_id, _plan, entries = parse_manifest_envelope(
        store.read_bytes(manifest.asset.relative_path)
    )
    assert manifest_run_id == str(run.id)
    reread_digests = {
        (ANALYSIS_RUN_MODEL, str(run.id)): row_digest(
            ANALYSIS_RUN_MODEL,
            model_row_values(run, ANALYSIS_RUN_FIELDS),
        ),
        **{
            (STOCK_ANALYSIS_MODEL, str(analysis.id)): row_digest(
                STOCK_ANALYSIS_MODEL,
                model_row_values(analysis, STOCK_ANALYSIS_FIELDS),
            )
            for analysis in analyses
        },
        **{
            (PREDICTION_MODEL, str(prediction.id)): row_digest(
                PREDICTION_MODEL,
                model_row_values(prediction, PREDICTION_FIELDS),
            )
            for prediction in predictions
        },
    }
    assert {(entry.model, entry.row_id): entry.digest for entry in entries} == reread_digests
    verify_price_product_output(run=run, store=store, replay=True)
    for asset in run_assets:
        payload, sha256 = asset_bytes_and_sha[asset.id]
        assert store.read_bytes(asset.relative_path) == payload
        assert DataAsset.objects.get(pk=asset.id).sha256 == sha256

    StockAnalysis.objects.filter(pk=analyses[0].pk).update(
        current_price=analyses[0].current_price + Decimal("0.000001")
    )
    with pytest.raises(RefreshVerificationError, match="rows diverged"):
        verify_price_product_output(run=run, store=store, replay=True)
    blocked_external_access.assert_not_called()


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
    store: AssetStore,
    subject: str,
    target: date,
    decision_time: datetime,
    *,
    signed_zero_edge: bool = False,
) -> DataAsset:
    from exchange_calendars import get_calendar  # type: ignore[import-untyped]

    calendar = get_calendar("XNYS")
    session = calendar.date_to_session(target, direction="none")
    sessions = [item.date() for item in calendar.sessions_window(session, -757)]
    offsets = np.arange(len(sessions), dtype=float)
    phase = 0.4 if subject == "SYN" else 0.1
    if signed_zero_edge:
        closes = (20.0 if subject == "SYN" else 100.0) * np.exp(
            0.000001 * np.sin(offsets / 13 + phase)
        )
    else:
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

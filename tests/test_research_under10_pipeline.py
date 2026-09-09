"""Under-$10 shadow assessment: pipeline integration and base differential.

Two things are proven here. First, that the additive `data_quality` key
appears for exactly the qualifying new analyses, is bounded in the queries it
costs, and never contaminates prediction provenance. Second -- by executing
the *actual* base revision's `research/service.py` against the same
deterministic synthetic fixture -- that removing the new key restores a
byte-identical result.

Everything is synthetic. No provider, credential, or network access occurs.
"""

from __future__ import annotations

import hashlib
import re
import sys
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import polars as pl
import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from base_service import (
    BASE_SHA,
    DEPENDENCY_MODULES,
    REPO_ROOT,
    _read_base_source,
    base_research_service,
    base_service_available,
    base_service_first_party_import_names,
    module_relative_path,
)
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.management.config_loader import default_us_scoring_config_path
from stanstock.data.models import (
    Company,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    LatestMarketData,
    Listing,
    ProviderRecord,
    Region,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.data.provider_policy import SEC_PROVIDER
from stanstock.data.sec_config import load_sec_fundamentals_config
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.service import (
    UNDER10_ASSESSMENT_KEY,
    _asset_cutoff_violation,
    _resolve_provider_plan,
    _validate_on_time_source_assets,
    analyze_listing,
    analyze_snapshot,
    under10_assessment_matches_persisted_evidence,
)
from stanstock.research.under10 import under10_assessment_hash, under10_policy_hash

TARGET_DATE = date(2026, 3, 2)
DECISION_TIME = datetime(2026, 3, 2, 21, 30, tzinfo=UTC)

SOURCE_CONCEPTS = {
    "cash_and_equivalents": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "short_term_debt": "us-gaap:ShortTermBorrowings",
    "current_long_term_debt": "us-gaap:LongTermDebtCurrent",
    "current_assets": "us-gaap:AssetsCurrent",
    "current_liabilities": "us-gaap:LiabilitiesCurrent",
    "operating_cash_flow": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "capital_expenditure": "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
}

INSTANT_VALUES = {
    "cash_and_equivalents": "1000.00000000",
    "short_term_debt": "100.00000000",
    "current_long_term_debt": "50.00000000",
    "current_assets": "2000.00000000",
    "current_liabilities": "1000.00000000",
}

DURATION_VALUES = {
    "operating_cash_flow": "400.00000000",
    "capital_expenditure": "100.00000000",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _snapshot(*, grade: str = UniverseSnapshot.Grade.RESEARCH) -> UniverseSnapshot:
    universe = Universe.objects.create(
        slug=f"under10-{uuid4().hex[:8]}",
        name="Under-$10 shadow universe",
        config_version="under10-v1",
    )
    return UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=grade,
        config_hash="a" * 64,
    )


def _listing(snapshot: UniverseSnapshot, *, ticker: str, currency: str = "USD") -> Listing:
    company = Company.objects.create(name=f"{ticker} Co", country="US", sector="Technology")
    security = Security.objects.create(company=company, name=f"{ticker} Common")
    listing = Listing.objects.create(
        security=security,
        ticker=ticker,
        exchange_mic="XNAS",
        currency=currency,
        region=Region.US,
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    return listing


def _price_asset(
    store: AssetStore,
    subject: str,
    *,
    close: float,
    rows: int = 300,
    retrieved_at: datetime | None = None,
    available_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    provider: str = "twelve_data",
    trend_per_session: float = 0.0,
    volume: float = 1_000_000.0,
) -> DataAsset:
    stamp = retrieved_at or (DECISION_TIME - timedelta(hours=2))
    dates = [TARGET_DATE - timedelta(days=index) for index in range(rows)][::-1]
    if trend_per_session:
        # A gentle, monotonic per-session compounding drift, applied so the
        # *last* (most recent, decision-run) close is exactly `close` --
        # preserving every existing caller's assumption about what `close`
        # means -- while earlier sessions are progressively smaller. This
        # exists only so a synthetic non-candidate can score as a genuine
        # (not merely present) opportunity for the downstream isolation
        # proof; it never applies unless a caller opts in.
        closes = [
            close / ((1.0 + trend_per_session) ** (rows - 1 - index)) for index in range(rows)
        ]
    else:
        closes = [close] * rows
    frame = pl.DataFrame(
        {
            "date": dates,
            "open": [value - 0.01 for value in closes],
            "high": [value + 0.05 for value in closes],
            "low": [value - 0.05 for value in closes],
            "close": closes,
            "volume": [volume] * rows,
        }
    )
    stored = store.write_frame(f"under10-tests/{uuid4().hex}.parquet", frame)
    return register_asset(
        provider=provider,
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=stamp,
        available_at=available_at or stamp,
        metadata=(
            metadata
            if metadata is not None
            else {
                "interval": "1day",
                "adjustment": "splits",
                "return_definition": "split_adjusted_price_return",
                "dividends_included": False,
                "currency": "USD",
            }
        ),
    )


def _sec_asset(
    store: AssetStore,
    *,
    key: str = "companyfacts",
    retrieved_at: datetime | None = None,
    available_at: datetime | None = None,
    provider: str = SEC_PROVIDER,
) -> DataAsset:
    stamp = retrieved_at or (DECISION_TIME - timedelta(days=5))
    stored = store.write_bytes(f"under10-tests/{key}-{uuid4().hex}.json", b"{}")
    return register_asset(
        provider=provider,
        kind="raw_fundamentals",
        subject=key,
        stored=stored,
        retrieved_at=stamp,
        available_at=available_at or stamp,
    )


def _sec_facts(
    listing: Listing,
    asset: DataAsset,
    *,
    instant_date: date = date(2025, 12, 31),
    annual_period: tuple[date, date] = (date(2025, 1, 1), date(2025, 12, 31)),
    available_at: datetime | None = None,
    values: dict[str, str] | None = None,
    filing_evidence: bool = True,
    provider: str = SEC_PROVIDER,
    source_revision: int = 1,
) -> list[FundamentalFact]:
    stamp = available_at or (DECISION_TIME - timedelta(days=5))
    resolved = {**INSTANT_VALUES, **DURATION_VALUES, **(values or {})}
    facts: list[FundamentalFact] = []
    for concept, value in resolved.items():
        instant = concept in INSTANT_VALUES
        period_start = None if instant else annual_period[0]
        period_end = instant_date if instant else annual_period[1]
        fact = FundamentalFact.objects.create(
            company=listing.security.company,
            provider=provider,
            concept=concept,
            taxonomy="us-gaap",
            source_concept=SOURCE_CONCEPTS[concept],
            value=Decimal(value),
            unit="USD",
            currency="USD",
            period_type=("instant" if instant else "duration"),
            period_identity=(
                f"instant::{period_end.isoformat()}"
                if instant
                else f"duration:{period_start}:{period_end}"
            ),
            period_start=period_start,
            period_end=period_end,
            fiscal_year=period_end.year,
            fiscal_period="FY",
            accession=f"0000000000-25-{concept[:8]}",
            filing_form="10-K",
            available_at=stamp,
            source_revision=source_revision,
            source_asset=asset,
        )
        if filing_evidence:
            FundamentalFactEvidence.objects.create(
                fact=fact,
                role=FundamentalFactEvidence.Role.FILING,
                source_asset=asset,
            )
        facts.append(fact)
    return facts


def _twelve_data_record(plan: str | None = "basic") -> ProviderRecord:
    metadata: dict[str, Any] = {} if plan is None else {"plan": plan}
    return ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=True,
        status="ready",
        metadata=metadata,
    )


def _analyze(
    listing: Listing,
    snapshot: UniverseSnapshot,
    store: AssetStore,
    **overrides: Any,
) -> Any:
    kwargs: dict[str, Any] = {
        "listing": listing,
        "universe_snapshot": snapshot,
        "decision_time": DECISION_TIME,
        "target_date": TARGET_DATE,
        "provider": "twelve_data",
        "store": store,
        "config_path": default_us_scoring_config_path(),
    }
    kwargs.update(overrides)
    return analyze_listing(**kwargs)


def _reader_candidate(
    tmp_path,
    settings,
    *,
    volume: float = 1_000_000.0,
    fact_values: dict[str, str] | None = None,
) -> tuple[StockAnalysis, AssetStore, DataAsset]:
    """Commit and reload one genuinely produced, evidence-bound candidate."""
    settings.DATA_DIR = tmp_path
    store = AssetStore()
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="REPLAY")
    price_asset = _price_asset(
        store,
        listing.ticker,
        close=4.25,
        volume=volume,
    )
    _sec_facts(listing, _sec_asset(store), values=fact_values)
    # A non-Basic recorded plan avoids activating the single-owner Basic
    # display middleware; it still exercises immutable provider binding and
    # the generic no-reviewed-split-source refusal.
    _twelve_data_record("grow")
    LatestMarketData.objects.create(
        listing=listing,
        observed_at=DECISION_TIME,
        session_date=TARGET_DATE,
        close=Decimal("4.25"),
        previous_close=Decimal("4.25"),
        volume=int(volume),
        source_asset=price_asset,
    )

    persisted = _analyze(listing, snapshot, store)
    analysis_id = persisted.analysis.pk
    del persisted
    analysis = StockAnalysis.objects.select_related(
        "run",
        "listing__security__company",
    ).get(pk=analysis_id)
    return analysis, store, price_asset


def _replace_recorded_assessment(
    analysis: StockAnalysis,
    payload: dict[str, Any],
) -> StockAnalysis:
    quality = deepcopy(analysis.data_quality)
    quality[UNDER10_ASSESSMENT_KEY] = payload
    analysis.data_quality = quality
    analysis.save(update_fields=["data_quality"])
    return StockAnalysis.objects.select_related(
        "run",
        "listing__security__company",
    ).get(pk=analysis.pk)


def _with_recomputed_assessment_hash(payload: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(payload)
    payload["assessment_hash"] = under10_assessment_hash(payload)
    return payload


def _assert_evidence_replay_rejected(
    *,
    analysis: StockAnalysis,
    payload: dict[str, Any],
    store: AssetStore,
    authenticated_client,
) -> None:
    analysis = _replace_recorded_assessment(analysis, payload)
    assert (
        under10_assessment_matches_persisted_evidence(
            analysis=analysis,
            recorded=payload,
            store=store,
        )
        is False
    )

    response = authenticated_client.get(reverse("stock-detail", args=[analysis.listing_id]))

    assert response.status_code == 200
    panel = response.context["under10_panel"]
    assert panel["state"] == "unsupported"
    content = " ".join(response.content.decode().split())
    assert "this build cannot read" in content
    assert "Assessed -" not in content
    assert "New allocation remains 0%" in content


# ---------------------------------------------------------------------------
# Candidate detection
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_under_ten_candidate_receives_the_shadow_key(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="UND")
    _price_asset(store, "UND", close=4.25)
    _sec_facts(listing, _sec_asset(store))
    _twelve_data_record("basic")

    persisted = _analyze(listing, snapshot, store)

    payload = persisted.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    assert payload["policy_version"] == "us-under10-shadow-v1"
    assert payload["policy_hash"] == under10_policy_hash(load_sec_fundamentals_config())
    assert payload["evaluated_for"]["reference_close"] == "4.250000"
    assert payload["evaluated_for"]["target_date"] == TARGET_DATE.isoformat()
    assert payload["evaluated_for"]["data_cutoff"] == persisted.run.data_cutoff.isoformat()
    assert payload["code_revision"] == persisted.run.code_revision
    assert payload["solvency"]["status"] == "no_adverse_evidence_observed"
    assert payload["liquidity"]["status"] == "computed"
    assert payload["split_verification"]["reason"] == "provider_plan_not_entitled"
    assert persisted.computation.data_quality[UNDER10_ASSESSMENT_KEY] == payload


# ---------------------------------------------------------------------------
# Stored-reader replay from immutable evidence.
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_true_producer_to_authenticated_render_replays_exact_recorded_evidence(
    tmp_path,
    settings,
    authenticated_client,
) -> None:
    analysis, store, _price = _reader_candidate(tmp_path, settings)
    recorded = analysis.data_quality[UNDER10_ASSESSMENT_KEY]

    assert (
        under10_assessment_matches_persisted_evidence(
            analysis=analysis,
            recorded=recorded,
            store=store,
        )
        is True
    )

    response = authenticated_client.get(reverse("stock-detail", args=[analysis.listing_id]))

    assert response.status_code == 200
    panel = response.context["under10_panel"]
    assert panel["state"] == "recorded"
    assert panel["solvency"]["state"] == "assessed"
    assert panel["liquidity"] == {
        "state": "assessed",
        "headline": "Assessed - median dollar volume over 252 observed sessions",
        "metric": "median_dollar_volume_252_sessions",
        "value": 4_250_000.0,
        "currency": "USD",
        "sessions_used": 252,
        "first_session": (TARGET_DATE - timedelta(days=251)).isoformat(),
        "last_session": TARGET_DATE.isoformat(),
        "basis": {
            "interval": "1day",
            "adjustment": "splits",
            "return_definition": "split_adjusted_price_return",
            "volume_basis": "provider_reported_unverified_split_basis",
        },
        "volume_basis_caveat": (
            "Split-only price basis is confirmed for this evidence. The provider's "
            "reported volume is not independently verified for splits, so this "
            "figure alone can never pass an activation gate."
        ),
    }
    assert panel["activated"] is False
    assert panel["activation_eligible"] is False
    assert panel["new_allocation_percent"] == 0
    assert all(value is False for value in recorded["gates"].values())
    assert response.context["opportunity"].eligible is False
    content = " ".join(response.content.decode().split())
    assert "Assessed - no adverse evidence observed" in content
    assert "4250000.0 USD over 252 observed sessions" in content
    assert "New allocation remains 0%" in content


@pytest.mark.django_db
def test_checksum_mismatched_valid_price_bytes_are_unsupported_with_one_read(
    tmp_path,
    settings,
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Physical bytes must match the immutable checksum before replay."""
    analysis, store, price_asset = _reader_candidate(tmp_path, settings)
    recorded = analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    target = store.resolve(price_asset.relative_path)
    original = pl.read_parquet(target).sort("date")
    changed_session = TARGET_DATE - timedelta(days=1)

    assert original.tail(252)["date"].is_in([changed_session]).any()
    assert original["close"].tail(1).item() == 4.25
    assert original.tail(252)["close"].median() == 4.25

    mutated = original.with_columns(
        pl.when(pl.col("date") == changed_session)
        .then(pl.lit(8.50))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    assert mutated.filter(pl.col("close") == 8.50).height == 1
    assert mutated["close"].tail(1).item() == 4.25
    assert mutated.tail(252)["close"].median() == 4.25
    mutated.write_parquet(target)

    physical_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    assert physical_sha256 != price_asset.sha256

    byte_reads: list[str] = []
    frame_reads: list[str] = []
    original_read_bytes = AssetStore.read_bytes
    original_read_frame = AssetStore.read_frame

    def recording_read_bytes(asset_store: AssetStore, relative_path: str) -> bytes:
        byte_reads.append(relative_path)
        return original_read_bytes(asset_store, relative_path)

    def recording_read_frame(asset_store: AssetStore, relative_path: str) -> pl.DataFrame:
        frame_reads.append(relative_path)
        return original_read_frame(asset_store, relative_path)

    def forbidden_resolution(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("stored replay must not resolve providers or contact the network")

    monkeypatch.setattr(AssetStore, "read_bytes", recording_read_bytes)
    monkeypatch.setattr(AssetStore, "read_frame", recording_read_frame)
    monkeypatch.setattr(
        "stanstock.research.service._resolve_provider_plan",
        forbidden_resolution,
    )
    monkeypatch.setattr("httpx.Client.request", forbidden_resolution)

    with CaptureQueriesContext(connection) as direct_queries:
        matches = under10_assessment_matches_persisted_evidence(
            analysis=analysis,
            recorded=recorded,
            store=store,
        )

    assert matches is False
    assert byte_reads == [price_asset.relative_path]
    assert frame_reads == []
    assert all(
        not query["sql"].lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE"))
        for query in direct_queries
    )

    byte_reads.clear()
    frame_reads.clear()
    with CaptureQueriesContext(connection) as request_queries:
        response = authenticated_client.get(reverse("stock-detail", args=[analysis.listing_id]))

    assert response.status_code == 200
    panel = response.context["under10_panel"]
    assert panel["state"] == "unsupported"
    assert byte_reads == [price_asset.relative_path]
    assert frame_reads == []
    assert all(
        not query["sql"].lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE"))
        for query in request_queries
    )
    content = " ".join(response.content.decode().split())
    assert "Recorded on the decision run" not in content
    assert "Assessed -" not in content
    assert price_asset.relative_path not in content
    assert str(store.root) not in content


@pytest.mark.django_db
def test_recomputed_hash_with_unrelated_same_cardinality_sec_lineage_is_rejected(
    tmp_path,
    settings,
    authenticated_client,
) -> None:
    analysis, store, _price = _reader_candidate(tmp_path, settings)
    payload = deepcopy(analysis.data_quality[UNDER10_ASSESSMENT_KEY])
    original_ids = payload["solvency"]["assessed_fact_ids"]
    original_assets = payload["solvency"]["assessed_assets"]

    other = _listing(analysis.run.universe_snapshot, ticker="UNRELATED")
    unrelated_asset = _sec_asset(store, key="unrelated-companyfacts")
    unrelated_facts = _sec_facts(other, unrelated_asset)
    payload["solvency"]["assessed_fact_ids"] = sorted(str(fact.pk) for fact in unrelated_facts)
    payload["solvency"]["assessed_assets"] = [
        {"id": str(unrelated_asset.pk), "sha256": unrelated_asset.sha256}
    ]
    assert len(payload["solvency"]["assessed_fact_ids"]) == len(original_ids)
    assert len(payload["solvency"]["assessed_assets"]) == len(original_assets)
    payload = _with_recomputed_assessment_hash(payload)

    _assert_evidence_replay_rejected(
        analysis=analysis,
        payload=payload,
        store=store,
        authenticated_client=authenticated_client,
    )


@pytest.mark.django_db
def test_recomputed_hash_with_a_forged_liquidity_median_is_rejected(
    tmp_path,
    settings,
    authenticated_client,
) -> None:
    analysis, store, _price = _reader_candidate(tmp_path, settings)
    payload = deepcopy(analysis.data_quality[UNDER10_ASSESSMENT_KEY])
    assert payload["liquidity"]["value"] == 4_250_000.0
    payload["liquidity"]["value"] = 4_250_001.0
    payload = _with_recomputed_assessment_hash(payload)

    _assert_evidence_replay_rejected(
        analysis=analysis,
        payload=payload,
        store=store,
        authenticated_client=authenticated_client,
    )


@pytest.mark.django_db
def test_recomputed_hash_with_a_forged_split_provider_is_rejected(
    tmp_path,
    settings,
    authenticated_client,
) -> None:
    analysis, store, _price = _reader_candidate(tmp_path, settings)
    payload = deepcopy(analysis.data_quality[UNDER10_ASSESSMENT_KEY])
    payload["split_verification"]["provider"] = "unrelated_price_provider"
    payload = _with_recomputed_assessment_hash(payload)

    _assert_evidence_replay_rejected(
        analysis=analysis,
        payload=payload,
        store=store,
        authenticated_client=authenticated_client,
    )


@pytest.mark.django_db
def test_complete_favorable_claim_with_impossible_one_fact_lineage_is_rejected(
    tmp_path,
    settings,
    authenticated_client,
) -> None:
    analysis, store, _price = _reader_candidate(tmp_path, settings)
    payload = deepcopy(analysis.data_quality[UNDER10_ASSESSMENT_KEY])
    assert payload["solvency"]["status"] == "no_adverse_evidence_observed"
    payload["solvency"]["assessed_fact_ids"] = payload["solvency"]["assessed_fact_ids"][:1]
    payload = _with_recomputed_assessment_hash(payload)

    _assert_evidence_replay_rejected(
        analysis=analysis,
        payload=payload,
        store=store,
        authenticated_client=authenticated_client,
    )


@pytest.mark.django_db
def test_solvency_operand_that_keeps_the_same_classification_is_rejected_by_replay(
    tmp_path,
    settings,
    authenticated_client,
) -> None:
    """Replace the old documented acceptance limit with authoritative rejection."""
    analysis, store, _price = _reader_candidate(tmp_path, settings)
    payload = deepcopy(analysis.data_quality[UNDER10_ASSESSMENT_KEY])
    assert payload["solvency"]["status"] == "no_adverse_evidence_observed"
    payload["solvency"]["inputs"]["cash_and_equivalents"] = "5000.00000000"
    payload = _with_recomputed_assessment_hash(payload)

    _assert_evidence_replay_rejected(
        analysis=analysis,
        payload=payload,
        store=store,
        authenticated_client=authenticated_client,
    )


@pytest.mark.django_db
def test_a_stale_assessment_hash_is_rejected_before_evidence_or_file_reads(
    tmp_path,
    settings,
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis, _store, _price = _reader_candidate(tmp_path, settings)
    payload = deepcopy(analysis.data_quality[UNDER10_ASSESSMENT_KEY])
    payload["liquidity"]["value"] = 4_250_001.0
    analysis = _replace_recorded_assessment(analysis, payload)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cheap hash rejection must precede evidence replay")

    monkeypatch.setattr(
        "stanstock.web.views.under10_assessment_matches_persisted_evidence",
        forbidden,
    )
    monkeypatch.setattr(AssetStore, "read_bytes", forbidden)
    monkeypatch.setattr(AssetStore, "read_frame", forbidden)

    response = authenticated_client.get(reverse("stock-detail", args=[analysis.listing_id]))

    assert response.status_code == 200
    assert response.context["under10_panel"]["state"] == "unsupported"


@pytest.mark.django_db
def test_absent_assessment_does_not_replay_or_read_an_asset(
    tmp_path,
    settings,
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis, _store, _price = _reader_candidate(tmp_path, settings)
    quality = deepcopy(analysis.data_quality)
    quality.pop(UNDER10_ASSESSMENT_KEY)
    analysis.data_quality = quality
    analysis.save(update_fields=["data_quality"])

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an absent assessment has no evidence to replay")

    monkeypatch.setattr(
        "stanstock.web.views.under10_assessment_matches_persisted_evidence",
        forbidden,
    )
    monkeypatch.setattr(AssetStore, "read_bytes", forbidden)
    monkeypatch.setattr(AssetStore, "read_frame", forbidden)

    response = authenticated_client.get(reverse("stock-detail", args=[analysis.listing_id]))

    assert response.status_code == 200
    assert response.context["under10_panel"]["state"] == "not_assessed"


@pytest.mark.django_db
def test_missing_price_asset_file_fails_closed_without_path_disclosure(
    tmp_path,
    settings,
    authenticated_client,
) -> None:
    analysis, store, price_asset = _reader_candidate(tmp_path, settings)
    payload = analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    store.resolve(price_asset.relative_path).unlink()

    _assert_evidence_replay_rejected(
        analysis=analysis,
        payload=payload,
        store=store,
        authenticated_client=authenticated_client,
    )
    response = authenticated_client.get(reverse("stock-detail", args=[analysis.listing_id]))
    content = response.content.decode()
    assert price_asset.relative_path not in content
    assert str(store.root) not in content


@pytest.mark.django_db
def test_missing_price_asset_row_fails_closed_without_exception_disclosure(
    tmp_path,
    settings,
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis, store, price_asset = _reader_candidate(tmp_path, settings)
    payload = analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    original_get = DataAsset.objects.get

    def missing_asset(*args: object, **kwargs: object) -> DataAsset:
        if kwargs.get("pk") == price_asset.pk:
            raise DataAsset.DoesNotExist
        return original_get(*args, **kwargs)

    monkeypatch.setattr(DataAsset.objects, "get", missing_asset)

    _assert_evidence_replay_rejected(
        analysis=analysis,
        payload=payload,
        store=store,
        authenticated_client=authenticated_client,
    )


@pytest.mark.django_db
def test_invalid_price_frame_fails_closed_without_parser_or_path_disclosure(
    tmp_path,
    settings,
    authenticated_client,
) -> None:
    analysis, store, price_asset = _reader_candidate(tmp_path, settings)
    payload = analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    store.resolve(price_asset.relative_path).write_bytes(b"not a parquet frame")

    _assert_evidence_replay_rejected(
        analysis=analysis,
        payload=payload,
        store=store,
        authenticated_client=authenticated_client,
    )
    response = authenticated_client.get(reverse("stock-detail", args=[analysis.listing_id]))
    content = response.content.decode()
    assert price_asset.relative_path not in content
    assert "parquet" not in content.lower()


@pytest.mark.django_db
def test_disagreeing_immutable_prediction_manifests_fail_closed(
    tmp_path,
    settings,
    authenticated_client,
) -> None:
    analysis, store, _price = _reader_candidate(tmp_path, settings)
    original = Prediction.objects.get(
        analysis=analysis,
        evidence_role=Prediction.EvidenceRole.DECISION,
    )
    mismatched_manifest = deepcopy(original.source_assets)
    mismatched_manifest[0] = {**mismatched_manifest[0], "sha256": "0" * 64}
    Prediction.objects.create(
        analysis=analysis,
        listing=original.listing,
        generated_at=original.generated_at,
        target_date=original.target_date,
        issued_on_time=original.issued_on_time,
        horizon=original.horizon,
        evidence_role=original.evidence_role,
        evidence_grade=original.evidence_grade,
        source_mode=original.source_mode,
        price_provider=original.price_provider,
        price_subject=original.price_subject,
        price_at_prediction=original.price_at_prediction,
        bear_return=original.bear_return,
        base_return=original.base_return,
        bull_return=original.bull_return,
        probability_positive=original.probability_positive,
        confidence=original.confidence,
        confidence_status=original.confidence_status,
        insufficiency_reason=original.insufficiency_reason,
        recommendation=original.recommendation,
        overall_score=original.overall_score,
        component_scores=original.component_scores,
        model_version="manifest-mismatch-v1",
        method_version=original.method_version,
        config_hash=original.config_hash,
        data_cutoff=original.data_cutoff,
        source_assets=mismatched_manifest,
        calculation=original.calculation,
        code_revision=original.code_revision,
    )
    payload = analysis.data_quality[UNDER10_ASSESSMENT_KEY]

    _assert_evidence_replay_rejected(
        analysis=analysis,
        payload=payload,
        store=store,
        authenticated_client=authenticated_client,
    )


@pytest.mark.django_db
def test_genuine_zero_volume_and_zero_debt_render_as_values_not_missing(
    tmp_path,
    settings,
    authenticated_client,
) -> None:
    analysis, store, _price = _reader_candidate(
        tmp_path,
        settings,
        volume=0.0,
        fact_values={
            "short_term_debt": "0.00000000",
            "current_long_term_debt": "0.00000000",
        },
    )
    payload = analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    assert payload["solvency"]["inputs"]["near_term_debt"] == "0.00000000"
    assert payload["liquidity"]["value"] == 0.0
    assert (
        under10_assessment_matches_persisted_evidence(
            analysis=analysis,
            recorded=payload,
            store=store,
        )
        is True
    )

    response = authenticated_client.get(reverse("stock-detail", args=[analysis.listing_id]))

    panel = response.context["under10_panel"]
    assert panel["state"] == "recorded"
    near_term_debt = next(
        item for item in panel["solvency"]["inputs"] if item["label"] == "Near-term debt"
    )
    assert near_term_debt == {
        "label": "Near-term debt",
        "value": "0.00000000",
        "present": True,
    }
    assert panel["liquidity"]["value"] == 0.0
    content = " ".join(response.content.decode().split())
    assert "0.0 USD over 252 observed sessions" in content


@pytest.mark.django_db
def test_authenticated_replay_is_query_and_file_read_bounded_and_writes_nothing(
    tmp_path,
    settings,
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis, _store, _price = _reader_candidate(tmp_path, settings)
    read_paths: list[str] = []
    original_read_bytes = AssetStore.read_bytes

    def recording_read_bytes(asset_store: AssetStore, relative_path: str) -> bytes:
        read_paths.append(relative_path)
        return original_read_bytes(asset_store, relative_path)

    def forbidden_provider_resolution(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("stored reader must not resolve current provider capability")

    monkeypatch.setattr(AssetStore, "read_bytes", recording_read_bytes)
    monkeypatch.setattr(
        "stanstock.research.service._resolve_provider_plan",
        forbidden_provider_resolution,
    )

    with CaptureQueriesContext(connection) as queries:
        response = authenticated_client.get(reverse("stock-detail", args=[analysis.listing_id]))

    assert response.status_code == 200
    assert response.context["under10_panel"]["state"] == "recorded"
    assert len(read_paths) == 1
    assert len(queries) <= 16
    assert all(
        not query["sql"].lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE"))
        for query in queries
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("close", "expects_key"),
    [
        (9.999999, True),
        (9.9999996, False),
        (10.0, False),
        (10.01, False),
        (4.25, True),
    ],
)
def test_rounded_reference_close_decides_the_ten_dollar_boundary(
    tmp_path,
    close: float,
    expects_key: bool,
) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="BND")
    _price_asset(store, "BND", close=close)

    persisted = _analyze(listing, snapshot, store)

    assert (UNDER10_ASSESSMENT_KEY in persisted.analysis.data_quality) is expects_key
    if expects_key:
        payload = persisted.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
        assert Decimal(payload["evaluated_for"]["reference_close"]) < Decimal("10")


@pytest.mark.django_db
def test_non_usd_listing_is_never_assessed(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="EUR1", currency="EUR")
    _price_asset(store, "EUR1", close=4.25)
    _sec_facts(listing, _sec_asset(store))

    persisted = _analyze(listing, snapshot, store)

    assert UNDER10_ASSESSMENT_KEY not in persisted.analysis.data_quality


@pytest.mark.django_db
def test_a_non_candidate_issues_no_additional_sec_query(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    expensive = _listing(snapshot, ticker="RICH")
    _price_asset(store, "RICH", close=120.0)
    _sec_facts(expensive, _sec_asset(store))

    with CaptureQueriesContext(connection) as captured:
        _analyze(expensive, snapshot, store)

    fact_queries = [
        query for query in captured.captured_queries if "data_fundamentalfact" in query["sql"]
    ]
    assert fact_queries == []


@pytest.mark.django_db
def test_a_candidate_issues_exactly_one_additional_sec_query(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="CHEAP")
    _price_asset(store, "CHEAP", close=4.25)
    _sec_facts(listing, _sec_asset(store))

    with CaptureQueriesContext(connection) as captured:
        _analyze(listing, snapshot, store)

    fact_queries = [
        query for query in captured.captured_queries if "data_fundamentalfact" in query["sql"]
    ]
    assert len(fact_queries) == 1


@pytest.mark.django_db
def test_both_entry_points_produce_the_same_assessment(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="BOTH")
    _price_asset(store, "BOTH", close=4.25)
    _sec_facts(listing, _sec_asset(store))
    _twelve_data_record("basic")

    single = _analyze(listing, snapshot, store)
    batch = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME + timedelta(seconds=1),
        target_date=TARGET_DATE,
        provider="twelve_data",
        store=store,
        config_path=default_us_scoring_config_path(),
    )

    single_payload = single.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    batch_payload = batch[0].analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    assert single_payload["solvency"] == batch_payload["solvency"]
    assert single_payload["liquidity"] == batch_payload["liquidity"]
    assert single_payload["split_verification"] == batch_payload["split_verification"]
    assert single_payload["policy_hash"] == batch_payload["policy_hash"]


@pytest.mark.django_db
def test_listing_and_benchmark_frames_are_bound_to_their_single_selected_assets(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One selector call per subject binds read bytes to every persisted manifest.

    The patched selector models a newer eligible immutable vintage appearing
    between two selections: its first call returns asset A and a hypothetical
    second call returns B. The rejected double-selection implementation would
    call it twice for both the listing and benchmark, read B while persisting
    A in the manifest, and fail the call-count/path/UUID assertions below.
    """
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="ATOM")
    listing_selected = _price_asset(
        store,
        "ATOM",
        close=4.25,
        retrieved_at=DECISION_TIME - timedelta(minutes=4),
        available_at=DECISION_TIME - timedelta(minutes=4),
    )
    listing_hypothetical_newer = _price_asset(
        store,
        "ATOM",
        close=8.75,
        retrieved_at=DECISION_TIME - timedelta(minutes=3),
        available_at=DECISION_TIME - timedelta(minutes=3),
    )
    benchmark_selected = _price_asset(
        store,
        "SPY",
        close=400.0,
        retrieved_at=DECISION_TIME - timedelta(minutes=2),
        available_at=DECISION_TIME - timedelta(minutes=2),
    )
    benchmark_hypothetical_newer = _price_asset(
        store,
        "SPY",
        close=800.0,
        retrieved_at=DECISION_TIME - timedelta(minutes=1),
        available_at=DECISION_TIME - timedelta(minutes=1),
    )
    sec_asset = _sec_asset(store, key="atomic-facts")
    _sec_facts(listing, sec_asset)
    _twelve_data_record("basic")

    alternatives = {
        "ATOM": (listing_selected, listing_hypothetical_newer),
        "SPY": (benchmark_selected, benchmark_hypothetical_newer),
    }
    selector_calls = {"ATOM": 0, "SPY": 0}

    def alternating_selector(
        _asof,
        *,
        provider: str,
        kind: str,
        subject: str,
    ) -> DataAsset:
        assert provider == "twelve_data"
        assert kind == "price_history"
        call_index = selector_calls[subject]
        selector_calls[subject] += 1
        return alternatives[subject][min(call_index, 1)]

    read_paths: list[str] = []
    original_read_frame = store.read_frame

    def recording_read_frame(relative_path: str) -> pl.DataFrame:
        read_paths.append(relative_path)
        return original_read_frame(relative_path)

    monkeypatch.setattr("stanstock.data.asof.AsOfData.latest_asset", alternating_selector)
    monkeypatch.setattr(store, "read_frame", recording_read_frame)

    persisted = _analyze(
        listing,
        snapshot,
        store,
        benchmark_subject="SPY",
    )

    assert selector_calls == {"ATOM": 1, "SPY": 1}
    assert read_paths == [
        listing_selected.relative_path,
        benchmark_selected.relative_path,
    ]
    assert persisted.analysis.current_price == Decimal("4.250000")

    manifest = persisted.analysis.data_quality["source_assets"]
    manifest_by_subject = {entry["subject"]: entry for entry in manifest}
    assert {key: (entry["id"], entry["sha256"]) for key, entry in manifest_by_subject.items()} == {
        "ATOM": (str(listing_selected.id), listing_selected.sha256),
        "SPY": (str(benchmark_selected.id), benchmark_selected.sha256),
    }
    assert persisted.analysis.data_quality["price_source"] == {
        "asset_id": str(listing_selected.id),
        "provider": "twelve_data",
        "subject": "ATOM",
    }
    payload = persisted.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    assert payload["liquidity"]["price_asset"] == {
        "id": str(listing_selected.id),
        "sha256": listing_selected.sha256,
    }
    unselected_ids = {
        str(listing_hypothetical_newer.id),
        str(benchmark_hypothetical_newer.id),
    }
    assert unselected_ids.isdisjoint(entry["id"] for entry in manifest)
    for prediction in persisted.predictions:
        assert {
            entry["subject"]: (entry["id"], entry["sha256"]) for entry in prediction.source_assets
        } == {
            "ATOM": (str(listing_selected.id), listing_selected.sha256),
            "SPY": (str(benchmark_selected.id), benchmark_selected.sha256),
        }


# ---------------------------------------------------------------------------
# Provider plan resolution
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_multiple_candidates_share_one_provider_plan_lookup(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    for ticker in ("A1", "A2", "A3"):
        listing = _listing(snapshot, ticker=ticker)
        _price_asset(store, ticker, close=4.25)
        _sec_facts(listing, _sec_asset(store, key=f"facts-{ticker}"))
    _twelve_data_record("basic")

    with CaptureQueriesContext(connection) as captured:
        results = analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            provider="twelve_data",
            store=store,
            config_path=default_us_scoring_config_path(),
        )

    plan_queries = [
        query
        for query in captured.captured_queries
        if "data_providerrecord" in query["sql"] and "plan" in query["sql"]
    ]
    assert len(results) == 3
    assert len(plan_queries) == 1
    assert all(
        result.analysis.data_quality[UNDER10_ASSESSMENT_KEY]["split_verification"]["reason"]
        == "provider_plan_not_entitled"
        for result in results
    )


@pytest.mark.django_db
def test_an_absent_plan_performs_no_extra_lookup_per_listing(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    for ticker in ("B1", "B2"):
        listing = _listing(snapshot, ticker=ticker)
        _price_asset(store, ticker, close=4.25)
        _sec_facts(listing, _sec_asset(store, key=f"facts-{ticker}"))

    with CaptureQueriesContext(connection) as captured:
        results = analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            provider="twelve_data",
            store=store,
            config_path=default_us_scoring_config_path(),
        )

    plan_queries = [
        query
        for query in captured.captured_queries
        if "data_providerrecord" in query["sql"] and "plan" in query["sql"]
    ]
    assert len(plan_queries) == 1
    assert all(
        result.analysis.data_quality[UNDER10_ASSESSMENT_KEY]["split_verification"]
        == {
            "status": "unavailable",
            "reason": "no_reviewed_corporate_actions_source",
            "provider": "twelve_data",
            "plan_recorded": False,
            "capability": "corporate_actions_splits",
            "inference_prohibited": True,
        }
        for result in results
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"plan": "basic"}, "basic"),
        ({"plan": " Basic "}, "basic"),
        ({"plan": "pro"}, "pro"),
        ({"plan": ""}, None),
        ({"plan": None}, None),
        ({"plan": 7}, None),
        ({}, None),
    ],
)
def test_provider_plan_resolution_normalizes_or_refuses(
    metadata: dict[str, Any],
    expected: str | None,
) -> None:
    ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=True,
        status="ready",
        metadata=metadata,
    )

    assert _resolve_provider_plan("twelve_data") == expected


@pytest.mark.django_db
def test_non_twelve_data_providers_perform_no_plan_query() -> None:
    with CaptureQueriesContext(connection) as captured:
        assert _resolve_provider_plan("synthetic_demo") is None
        assert _resolve_provider_plan("sec") is None

    assert list(captured.captured_queries) == []


@pytest.mark.django_db
def test_an_unconfigured_provider_record_yields_no_plan() -> None:
    assert _resolve_provider_plan("twelve_data") is None


# ---------------------------------------------------------------------------
# Shared cutoff-safety rule
# ---------------------------------------------------------------------------


def test_asset_cutoff_violation_reports_available_at_first() -> None:
    late = (DECISION_TIME + timedelta(seconds=1)).isoformat()
    on_time = (DECISION_TIME - timedelta(seconds=1)).isoformat()

    assert (
        _asset_cutoff_violation(
            {"id": "x", "available_at": late, "retrieved_at": late},
            data_cutoff=DECISION_TIME,
        )
        == "available_at"
    )
    assert (
        _asset_cutoff_violation(
            {"id": "x", "available_at": on_time, "retrieved_at": late},
            data_cutoff=DECISION_TIME,
        )
        == "retrieved_at"
    )
    assert (
        _asset_cutoff_violation(
            {"id": "x", "available_at": DECISION_TIME.isoformat(), "retrieved_at": on_time},
            data_cutoff=DECISION_TIME,
        )
        is None
    )


def test_the_original_validator_still_raises_identically() -> None:
    late = (DECISION_TIME + timedelta(seconds=1)).isoformat()
    on_time = (DECISION_TIME - timedelta(seconds=1)).isoformat()

    _validate_on_time_source_assets(
        [{"id": "ok", "available_at": on_time, "retrieved_at": on_time}],
        data_cutoff=DECISION_TIME,
    )
    with pytest.raises(ValueError) as available_error:
        _validate_on_time_source_assets(
            [{"id": "bad", "available_at": late, "retrieved_at": late}],
            data_cutoff=DECISION_TIME,
        )
    with pytest.raises(ValueError) as retrieved_error:
        _validate_on_time_source_assets(
            [{"id": "bad", "available_at": on_time, "retrieved_at": late}],
            data_cutoff=DECISION_TIME,
        )

    assert str(available_error.value) == (
        "On-time analysis source asset bad has available_at after data cutoff"
    )
    assert str(retrieved_error.value) == (
        "On-time analysis source asset bad has retrieved_at after data cutoff"
    )


@pytest.mark.django_db
def test_on_time_run_withholds_shadow_evidence_retrieved_after_the_cutoff(tmp_path) -> None:
    """The shared cutoff rule withholds instead of failing the run.

    `AsOfData` already hides an asset retrieved after the as-of decision
    time, so this exercises the residual case the rule exists for: an on-time
    issuance whose logical `data_cutoff` is earlier than the as-of decision
    time it reads through. The core validator would raise here; the shadow
    path records `evidence_not_cutoff_safe` and keeps the rejected evidence
    referenced.
    """
    from stanstock.data.asof import AsOfData
    from stanstock.research.config import load_scoring_config
    from stanstock.research.service import _compute_listing_from_asof

    store = AssetStore(tmp_path)
    snapshot = _snapshot(grade=UniverseSnapshot.Grade.OBSERVED)
    listing = _listing(snapshot, ticker="LATE")
    _price_asset(store, "LATE", close=4.25)
    late_asset = _sec_asset(
        store,
        key="late-facts",
        retrieved_at=DECISION_TIME + timedelta(minutes=1),
        available_at=DECISION_TIME - timedelta(days=1),
    )
    _sec_facts(listing, late_asset, available_at=DECISION_TIME - timedelta(days=1))

    computation = _compute_listing_from_asof(
        listing=listing,
        asof=AsOfData(DECISION_TIME + timedelta(days=30), store),
        provider="twelve_data",
        config=load_scoring_config(default_us_scoring_config_path()),
        decision_time=DECISION_TIME,
        issued_on_time=True,
        provider_plan="basic",
        code_revision_value="0" * 40,
        target_date=TARGET_DATE,
    )

    payload = computation.data_quality[UNDER10_ASSESSMENT_KEY]
    assert payload["solvency"]["status"] == "insufficient_evidence"
    assert payload["solvency"]["reasons"] == ["evidence_not_cutoff_safe"]
    assert payload["solvency"]["assessed_fact_ids"]
    assert {"id": str(late_asset.id), "sha256": late_asset.sha256} in (
        payload["solvency"]["assessed_assets"]
    )


@pytest.mark.django_db
def test_on_time_sec_cutoff_ignores_late_unqualified_assets_but_not_late_sec_assets(
    tmp_path,
) -> None:
    """Cutoff safety is evaluated over the same provider-qualified sequence.

    All synthetic facts are visible to the later research read and claim an
    on-time fact availability. Their source assets differ only at the exact
    qualification boundary: foreign facts, SEC facts on a foreign source,
    and foreign facts on an SEC source must not make otherwise-safe SEC
    evidence late. A late SEC fact on a late SEC source still must.
    """
    from stanstock.data.asof import AsOfData
    from stanstock.research.config import load_scoring_config
    from stanstock.research.service import _compute_listing_from_asof

    store = AssetStore(tmp_path)
    snapshot = _snapshot(grade=UniverseSnapshot.Grade.OBSERVED)
    listing = _listing(snapshot, ticker="QUALTIME")
    _price_asset(store, "QUALTIME", close=4.25)
    safe_asset = _sec_asset(store, key="safe-qualified")
    safe_facts = _sec_facts(listing, safe_asset)
    late_retrieval = DECISION_TIME + timedelta(minutes=1)
    fact_availability = DECISION_TIME - timedelta(days=1)

    foreign_asset = _sec_asset(
        store,
        key="late-foreign",
        retrieved_at=late_retrieval,
        available_at=fact_availability,
        provider="other_provider",
    )
    foreign_facts = _sec_facts(
        listing,
        foreign_asset,
        available_at=fact_availability,
        provider="other_provider",
    )
    mismatched_asset = _sec_asset(
        store,
        key="late-mismatched-source",
        retrieved_at=late_retrieval,
        available_at=fact_availability,
        provider="other_source_provider",
    )
    mismatched_facts = _sec_facts(
        listing,
        mismatched_asset,
        available_at=fact_availability,
        provider=SEC_PROVIDER,
        source_revision=2,
    )
    late_sec_source = _sec_asset(
        store,
        key="late-sec-source-for-foreign-facts",
        retrieved_at=late_retrieval,
        available_at=fact_availability,
    )
    foreign_on_sec_facts = _sec_facts(
        listing,
        late_sec_source,
        available_at=fact_availability,
        provider="other_fact_provider",
    )

    def compute() -> Any:
        return _compute_listing_from_asof(
            listing=listing,
            asof=AsOfData(DECISION_TIME + timedelta(days=30), store),
            provider="twelve_data",
            config=load_scoring_config(default_us_scoring_config_path()),
            decision_time=DECISION_TIME,
            issued_on_time=True,
            provider_plan="basic",
            code_revision_value="0" * 40,
            target_date=TARGET_DATE,
        )

    safe_payload = compute().data_quality[UNDER10_ASSESSMENT_KEY]
    assert safe_payload["solvency"]["status"] == "no_adverse_evidence_observed"
    assert safe_payload["solvency"]["assessed_fact_ids"] == sorted(
        str(fact.id) for fact in safe_facts
    )
    unqualified_fact_ids = {
        str(fact.id) for fact in (*foreign_facts, *mismatched_facts, *foreign_on_sec_facts)
    }
    assert unqualified_fact_ids.isdisjoint(safe_payload["solvency"]["assessed_fact_ids"])
    unqualified_asset_ids = {
        str(foreign_asset.id),
        str(mismatched_asset.id),
        str(late_sec_source.id),
    }
    assert unqualified_asset_ids.isdisjoint(
        reference["id"] for reference in safe_payload["solvency"]["assessed_assets"]
    )

    late_qualified_asset = _sec_asset(
        store,
        key="late-qualified-sec",
        retrieved_at=late_retrieval,
        available_at=fact_availability,
    )
    late_qualified_facts = _sec_facts(
        listing,
        late_qualified_asset,
        available_at=fact_availability,
        provider=SEC_PROVIDER,
        source_revision=3,
    )

    unsafe_payload = compute().data_quality[UNDER10_ASSESSMENT_KEY]
    assert unsafe_payload["solvency"]["status"] == "insufficient_evidence"
    assert unsafe_payload["solvency"]["reasons"] == ["evidence_not_cutoff_safe"]
    assert {str(fact.id) for fact in late_qualified_facts}.issubset(
        unsafe_payload["solvency"]["assessed_fact_ids"]
    )
    assert {"id": str(late_qualified_asset.id), "sha256": late_qualified_asset.sha256} in (
        unsafe_payload["solvency"]["assessed_assets"]
    )


@pytest.mark.django_db
def test_an_on_time_run_cannot_even_see_evidence_retrieved_after_its_cutoff(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot(grade=UniverseSnapshot.Grade.OBSERVED)
    listing = _listing(snapshot, ticker="UNSEEN")
    _price_asset(store, "UNSEEN", close=4.25)
    _sec_facts(
        listing,
        _sec_asset(
            store,
            key="unseen-facts",
            retrieved_at=DECISION_TIME + timedelta(minutes=1),
            available_at=DECISION_TIME - timedelta(days=1),
        ),
        available_at=DECISION_TIME - timedelta(days=1),
    )

    persisted = _analyze(listing, snapshot, store, issued_on_time=True)

    payload = persisted.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    assert persisted.run.issued_on_time is True
    assert payload["solvency"]["status"] == "insufficient_evidence"
    assert payload["solvency"]["assessed_fact_ids"] == []


@pytest.mark.django_db
def test_research_grade_run_may_use_later_retrieved_evidence(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot(grade=UniverseSnapshot.Grade.RESEARCH)
    listing = _listing(snapshot, ticker="RSCH")
    _price_asset(store, "RSCH", close=4.25)
    later_asset = _sec_asset(
        store,
        key="later-facts",
        retrieved_at=DECISION_TIME + timedelta(days=30),
        available_at=DECISION_TIME - timedelta(days=1),
    )
    _sec_facts(listing, later_asset, available_at=DECISION_TIME - timedelta(days=1))

    persisted = _analyze(
        listing,
        snapshot,
        store,
        decision_time=DECISION_TIME + timedelta(days=60),
        issued_on_time=False,
    )

    payload = persisted.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    assert persisted.run.issued_on_time is False
    assert payload["solvency"]["status"] == "no_adverse_evidence_observed"


@pytest.mark.django_db
def test_a_fact_available_after_the_cutoff_is_invisible(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="HIDE")
    _price_asset(store, "HIDE", close=4.25)
    _sec_facts(
        listing,
        _sec_asset(store),
        available_at=DECISION_TIME + timedelta(days=1),
    )

    persisted = _analyze(listing, snapshot, store)

    payload = persisted.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    assert payload["solvency"]["status"] == "insufficient_evidence"
    assert payload["solvency"]["assessed_fact_ids"] == []


@pytest.mark.django_db
def test_facts_without_visible_filing_evidence_are_invisible(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="NOFIL")
    _price_asset(store, "NOFIL", close=4.25)
    _sec_facts(listing, _sec_asset(store), filing_evidence=False)

    persisted = _analyze(listing, snapshot, store)

    payload = persisted.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    assert payload["solvency"]["status"] == "insufficient_evidence"
    assert payload["solvency"]["assessed_fact_ids"] == []


# ---------------------------------------------------------------------------
# Isolation: no provenance contamination, no rewrite
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_shadow_sec_evidence_never_enters_manifests_or_predictions(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="ISO")
    price_asset = _price_asset(store, "ISO", close=4.25)
    sec_asset = _sec_asset(store)
    _sec_facts(listing, sec_asset)

    persisted = _analyze(listing, snapshot, store)

    payload = persisted.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    shadow_asset_ids = {entry["id"] for entry in payload["solvency"]["assessed_assets"]}
    manifest_ids = {asset["id"] for asset in persisted.computation.source_assets}
    sibling_ids = {asset["id"] for asset in persisted.analysis.data_quality["source_assets"]}

    assert str(sec_asset.id) in shadow_asset_ids
    assert manifest_ids == {str(price_asset.id)}
    assert sibling_ids == manifest_ids
    assert persisted.analysis.data_quality["source_assets"] == persisted.computation.source_assets
    for prediction in persisted.predictions:
        prediction_ids = {asset["id"] for asset in prediction.source_assets}
        assert prediction_ids == manifest_ids
        assert shadow_asset_ids.isdisjoint(prediction_ids)
        assert UNDER10_ASSESSMENT_KEY not in prediction.calculation
        assert UNDER10_ASSESSMENT_KEY not in prediction.component_scores
        assert prediction.source_mode == Prediction.SourceMode.PROVIDER


@pytest.mark.django_db
def test_reanalysis_never_rewrites_an_existing_analysis(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="KEEP")
    _price_asset(store, "KEEP", close=4.25)
    _sec_facts(listing, _sec_asset(store))

    first = _analyze(listing, snapshot, store)
    first_id = first.analysis.pk
    first_quality = dict(StockAnalysis.objects.get(pk=first_id).data_quality)
    second = _analyze(
        listing,
        snapshot,
        store,
        decision_time=DECISION_TIME + timedelta(seconds=5),
    )

    assert second.analysis.pk != first_id
    assert StockAnalysis.objects.get(pk=first_id).data_quality == first_quality
    assert StockAnalysis.objects.count() == 2


@pytest.mark.django_db
def test_a_prior_unassessed_row_is_not_backfilled(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="OLD")
    _price_asset(store, "OLD", close=4.25)
    legacy_run = AnalysisRun.objects.create(
        generated_at=DECISION_TIME - timedelta(days=10),
        data_cutoff=DECISION_TIME - timedelta(days=10),
        target_date=TARGET_DATE - timedelta(days=10),
        universe_snapshot=snapshot,
        config_version="us-price-baseline-v2",
        config_hash="b" * 64,
        code_revision="legacy",
    )
    legacy = StockAnalysis.objects.create(
        run=legacy_run,
        listing=listing,
        current_price=Decimal("4.250000"),
        overall_score=Decimal("50.00"),
        recommendation="hold",
        risk_class="medium",
        confidence=Decimal("50.00"),
        data_quality={"source_assets": []},
    )

    _analyze(listing, snapshot, store)

    legacy.refresh_from_db()
    assert legacy.data_quality == {"source_assets": []}
    assert UNDER10_ASSESSMENT_KEY not in legacy.data_quality


@pytest.mark.django_db
def test_shadow_payload_does_not_change_scores_or_recommendation(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    candidate = _listing(snapshot, ticker="SHD")
    _price_asset(store, "SHD", close=4.25)
    _sec_facts(listing=candidate, asset=_sec_asset(store))

    persisted = _analyze(candidate, snapshot, store)
    payload = persisted.analysis.data_quality[UNDER10_ASSESSMENT_KEY]

    assert persisted.analysis.data_quality["fundamentals_used"] is False
    assert persisted.computation.fundamentals.values == {}
    assert payload["new_allocation_percent"] == 0
    assert payload["activation_eligible"] is False
    # The scored payload keys are exactly the base set plus the shadow key.
    assert set(persisted.analysis.data_quality) - {UNDER10_ASSESSMENT_KEY} == {
        "indicator_missing",
        "fundamental_missing",
        "scoring_missing",
        "coverage",
        "missingness_penalty",
        "freshness_penalty",
        "observation_count",
        "source_assets",
        "recommendation_gates",
        "risk_insufficiency_reason",
        "analysis_mode",
        "fundamentals_used",
        "supported_horizons",
        "factor_policy",
        "price_source",
        "return_definition",
        "dividends_included",
    }


# ---------------------------------------------------------------------------
# `invalid_session_date_rows` -- a null session date withholds liquidity.
# ---------------------------------------------------------------------------


def _price_asset_with_one_invalid_session_date(
    store: AssetStore,
    subject: str,
    *,
    close: float,
    date_kind: str,
    available_at: datetime | None = None,
) -> DataAsset:
    """252 valid daily rows plus one row whose session date is null.

    Mirrors `_price_asset`'s shape and reviewed basis metadata exactly, so
    the only variable under test is the added null-dated row. Parametrized
    over `date_kind` because `price_frame_with_diagnostics` normalizes
    `pl.Date`, `pl.Datetime`, and `pl.Utf8` schemas identically.
    """
    stamp = available_at or (DECISION_TIME - timedelta(hours=2))
    valid_dates = [TARGET_DATE - timedelta(days=index) for index in range(252)][::-1]
    all_dates: list[date | None] = [*valid_dates, None]
    closes = [close] * len(all_dates)
    if date_kind == "date":
        date_column = pl.Series("date", all_dates, dtype=pl.Date)
    elif date_kind == "datetime":
        values = [
            datetime(value.year, value.month, value.day, tzinfo=UTC) if value is not None else None
            for value in all_dates
        ]
        date_column = pl.Series("date", values, dtype=pl.Datetime("us", "UTC"))
    elif date_kind == "utf8":
        values = [value.isoformat() if value is not None else None for value in all_dates]
        date_column = pl.Series("date", values, dtype=pl.Utf8)
    else:
        raise ValueError(f"Unsupported date_kind: {date_kind!r}")
    frame = pl.DataFrame(
        {
            "date": date_column,
            "open": [value - 0.01 for value in closes],
            "high": [value + 0.05 for value in closes],
            "low": [value - 0.05 for value in closes],
            "close": closes,
            "volume": [1_000_000.0] * len(all_dates),
        }
    )
    stored = store.write_frame(f"under10-tests/{uuid4().hex}.parquet", frame)
    return register_asset(
        provider="twelve_data",
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=stamp,
        available_at=stamp,
        metadata={
            "interval": "1day",
            "adjustment": "splits",
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
            "currency": "USD",
        },
    )


@pytest.mark.django_db
@pytest.mark.parametrize("date_kind", ["date", "datetime", "utf8"])
def test_a_null_session_date_withholds_liquidity_without_touching_score_or_predictions(
    tmp_path,
    date_kind: str,
) -> None:
    """A raw immutable asset with 252 valid rows plus one null date is withheld.

    Reusing the same listing/company for both the clean and the tainted
    price bundle means only the price data differs between the two runs:
    score, recommendation, component scores, and the decision prediction's
    calculation must stay exactly what the clean 252-session run produces.
    """
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker=f"NUL{date_kind[:3].upper()}")
    _sec_facts(listing, _sec_asset(store))
    _twelve_data_record("basic")

    _price_asset(store, listing.ticker, close=4.25, rows=252)
    clean = _analyze(listing, snapshot, store, decision_time=DECISION_TIME)

    # A later-registered bundle for the *same* subject becomes the eligible
    # asset for a later decision time; only its bytes differ from the clean
    # bundle above by the appended null-dated row.
    _price_asset_with_one_invalid_session_date(
        store,
        listing.ticker,
        close=4.25,
        date_kind=date_kind,
        available_at=DECISION_TIME + timedelta(minutes=5),
    )
    tainted = _analyze(
        listing,
        snapshot,
        store,
        decision_time=DECISION_TIME + timedelta(minutes=10),
    )

    clean_payload = clean.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    tainted_payload = tainted.analysis.data_quality[UNDER10_ASSESSMENT_KEY]

    # Clean 252 remains computed.
    assert clean_payload["liquidity"]["status"] == "computed"
    assert clean_payload["liquidity"]["sessions_used"] == 252
    assert clean_payload["liquidity"]["reason"] is None

    # The tainted bundle withholds explicitly, with every value/session field
    # null -- never a shorter-but-plausible 252-session figure.
    tainted_liquidity = tainted_payload["liquidity"]
    assert tainted_liquidity["status"] == "withheld"
    assert tainted_liquidity["reason"] == "invalid_session_dates"
    assert tainted_liquidity["value"] is None
    assert tainted_liquidity["sessions_used"] is None
    assert tainted_liquidity["first_session"] is None
    assert tainted_liquidity["last_session"] is None

    # No policy/config/version/hash bump: the reused reason string and the
    # unchanged policy identity prove this, not a new gate.
    assert tainted_payload["policy_hash"] == clean_payload["policy_hash"]
    assert tainted_payload["policy_version"] == clean_payload["policy_version"]
    assert tainted_payload["schema_version"] == clean_payload["schema_version"]

    # Score, recommendation, component scores, and the decision prediction's
    # calculation are byte-identical: the withheld shadow liquidity key
    # never touches them.
    assert tainted.analysis.overall_score == clean.analysis.overall_score
    assert tainted.analysis.recommendation == clean.analysis.recommendation
    assert tainted.analysis.component_scores == clean.analysis.component_scores
    assert tainted.analysis.confidence == clean.analysis.confidence
    assert tainted.analysis.risk_class == clean.analysis.risk_class
    for key in (
        "indicator_missing",
        "fundamental_missing",
        "scoring_missing",
        "coverage",
        "missingness_penalty",
        "freshness_penalty",
        "observation_count",
        "recommendation_gates",
        "risk_insufficiency_reason",
        "analysis_mode",
        "fundamentals_used",
        "supported_horizons",
        "factor_policy",
    ):
        assert tainted.analysis.data_quality[key] == clean.analysis.data_quality[key], key
    clean_decision = next(p for p in clean.predictions if p.horizon == "short")
    tainted_decision = next(p for p in tainted.predictions if p.horizon == "short")
    # `prediction_version` embeds `AnalysisRun.id.hex[:8]`, a fresh identity
    # per run; every other calculation field must stay byte-identical.
    assert {
        key: value
        for key, value in tainted_decision.calculation.items()
        if key != "prediction_version"
    } == {
        key: value
        for key, value in clean_decision.calculation.items()
        if key != "prediction_version"
    }
    assert tainted_decision.bear_return == clean_decision.bear_return
    assert tainted_decision.base_return == clean_decision.base_return
    assert tainted_decision.bull_return == clean_decision.bull_return
    assert tainted_decision.recommendation == clean_decision.recommendation


# ---------------------------------------------------------------------------
# Base-revision differential
# ---------------------------------------------------------------------------

RUN_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def _normalize_analysis(analysis: StockAnalysis) -> dict[str, Any]:
    return {
        "current_price": str(analysis.current_price),
        "daily_change": str(analysis.daily_change),
        "overall_score": str(analysis.overall_score),
        "recommendation": analysis.recommendation,
        "risk_score": str(analysis.risk_score),
        "risk_class": analysis.risk_class,
        "confidence": str(analysis.confidence),
        "confidence_status": analysis.confidence_status,
        "component_scores": analysis.component_scores,
        "forecast_scenarios": analysis.forecast_scenarios,
        "short_scenario": analysis.short_scenario,
        "medium_scenario": analysis.medium_scenario,
        "long_scenario": analysis.long_scenario,
        "reasons": analysis.reasons,
        "risks": analysis.risks,
        "data_quality": analysis.data_quality,
        "run": {
            "config_version": analysis.run.config_version,
            "config_hash": analysis.run.config_hash,
            "code_revision": analysis.run.code_revision,
            "target_date": analysis.run.target_date.isoformat(),
            "data_cutoff": analysis.run.data_cutoff.isoformat(),
            "generated_at": analysis.run.generated_at.isoformat(),
            "issued_on_time": analysis.run.issued_on_time,
        },
    }


#: A medium-forecast panel is a fresh derived asset minted *per run*: two
#: independent `analyze_snapshot` calls over an identical fixture legitimately
#: register a different panel `id`/`subject`/`relative_path` for byte-identical
#: content. `sha256` is what actually attests to content and is deliberately
#: left untouched here, so a genuine content difference is never masked by
#: this expected per-run identity churn.
_PANEL_IDENTITY_PLACEHOLDER = "<panel>"


def _normalize_source_assets(source_assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for asset in source_assets:
        if asset.get("kind") == "medium_forecast_panel":
            asset = {
                **asset,
                "id": _PANEL_IDENTITY_PLACEHOLDER,
                "subject": _PANEL_IDENTITY_PLACEHOLDER,
                "relative_path": _PANEL_IDENTITY_PLACEHOLDER,
            }
        normalized.append(asset)
    return normalized


def _normalize_prediction(prediction: Prediction) -> dict[str, Any]:
    # The run-scoped model-version suffix is `AnalysisRun.id.hex[:8]`, a fresh
    # UUID per run. Alongside it, an advisory medium prediction's
    # `panel_asset_id` and its `source_assets` panel entry are the only other
    # generated identities normalized here.
    return {
        "horizon": prediction.horizon,
        "evidence_role": prediction.evidence_role,
        "evidence_grade": prediction.evidence_grade,
        "source_mode": prediction.source_mode,
        "price_provider": prediction.price_provider,
        "price_subject": prediction.price_subject,
        "price_at_prediction": str(prediction.price_at_prediction),
        "bear_return": str(prediction.bear_return),
        "base_return": str(prediction.base_return),
        "bull_return": str(prediction.bull_return),
        "probability_positive": str(prediction.probability_positive),
        "confidence": str(prediction.confidence),
        "confidence_status": prediction.confidence_status,
        "insufficiency_reason": prediction.insufficiency_reason,
        "recommendation": prediction.recommendation,
        "overall_score": str(prediction.overall_score),
        "component_scores": prediction.component_scores,
        "model_version": RUN_SUFFIX.sub("-<run>", prediction.model_version),
        "method_version": prediction.method_version,
        "config_hash": prediction.config_hash,
        "data_cutoff": prediction.data_cutoff.isoformat(),
        "generated_at": prediction.generated_at.isoformat(),
        "target_date": prediction.target_date.isoformat(),
        "issued_on_time": prediction.issued_on_time,
        "source_assets": _normalize_source_assets(prediction.source_assets),
        "calculation": {
            key: (
                RUN_SUFFIX.sub("-<run>", value)
                if key == "prediction_version" and isinstance(value, str)
                else (
                    _PANEL_IDENTITY_PLACEHOLDER
                    if key == "panel_asset_id" and isinstance(value, str)
                    else value
                )
            )
            for key, value in prediction.calculation.items()
        },
        "code_revision": prediction.code_revision,
    }


def _capture(run: AnalysisRun) -> dict[str, Any]:
    return {
        "analyses": {
            analysis.listing.ticker: _normalize_analysis(analysis)
            for analysis in run.stocks.select_related("run", "listing").order_by("listing__ticker")
        },
        "predictions": {
            f"{prediction.listing.ticker}:{prediction.horizon}:{prediction.evidence_role}": (
                _normalize_prediction(prediction)
            )
            for prediction in Prediction.objects.filter(analysis__run=run)
            .select_related("listing")
            .order_by("listing__ticker", "horizon", "evidence_role")
        },
    }


def _differential_fixture(store: AssetStore) -> UniverseSnapshot:
    """Two listings, a benchmark, and both provider records medium/long forecasts need.

    900 calendar days of history (which trivially includes every real
    trading session in that span) and a registered ``sec`` `ProviderRecord`
    exist specifically so `analyze_snapshot` actually builds
    `AdvisoryForecastContext`/`LongForecastContext` -- not just decision
    predictions -- letting the differential tests below compare real 6m/12m/
    3y/5y rows instead of two sides that both silently skipped them.

    The snapshot is `OBSERVED`-grade with `generated_at.date() ==
    target_date`, so `issued_on_time` is genuinely (not explicitly-overridden)
    `True`: `canonical_reportable_prediction_filter` requires exactly that
    combination, and a `RESEARCH`-grade fixture would make any downstream
    "reportable" count trivially zero on both sides regardless of a real
    regression. DIF2 additionally carries a gentle upward price trend and a
    `LatestMarketData` row so it can score as a genuine, non-vacuous "great
    opportunity" candidate -- not merely present -- for the sample-basket
    isolation proof; DIF1 stays flat and Under-$10, excluded by policy
    either way.
    """
    snapshot = _snapshot(grade=UniverseSnapshot.Grade.OBSERVED)
    candidate = _listing(snapshot, ticker="DIF1")
    candidate_asset = _price_asset(store, "DIF1", close=4.25, rows=900)
    _sec_facts(candidate, _sec_asset(store, key="facts-dif1"))
    LatestMarketData.objects.create(
        listing=candidate,
        observed_at=DECISION_TIME,
        session_date=TARGET_DATE,
        close=Decimal("4.25"),
        source_asset=candidate_asset,
    )
    other = _listing(snapshot, ticker="DIF2")
    other_asset = _price_asset(store, "DIF2", close=120.5, rows=900, trend_per_session=0.006)
    _sec_facts(other, _sec_asset(store, key="facts-dif2"))
    LatestMarketData.objects.create(
        listing=other,
        observed_at=DECISION_TIME,
        session_date=TARGET_DATE,
        close=Decimal("120.5"),
        source_asset=other_asset,
    )
    _price_asset(store, "SPY", close=420.0, rows=900)
    _twelve_data_record("basic")
    ProviderRecord.objects.create(provider="sec", enabled=True, status="ready", metadata={})
    return snapshot


_ADVISORY_HORIZONS = frozenset({"6m", "12m", "3y", "5y"})


def _advisory_predictions(capture: dict[str, Any]) -> dict[str, Any]:
    """The 6m/12m/3y/5y advisory rows out of one `_capture(...)` result."""
    return {
        key: value
        for key, value in capture["predictions"].items()
        if key.split(":")[1] in _ADVISORY_HORIZONS
    }


@pytest.mark.django_db
@pytest.mark.skipif(
    not base_service_available(),
    reason=f"base revision {BASE_SHA} is not in the local git object database",
)
def test_base_and_head_agree_after_removing_only_the_shadow_key(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _differential_fixture(store)

    with base_research_service() as base:
        base_results = base.service.analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            provider="twelve_data",
            benchmark_subject="SPY",
            store=store,
            config_path=default_us_scoring_config_path(),
        )
        base_source_sha256 = base.source_sha256
    base_capture = _capture(base_results[0].run)

    # The base revision predates the key entirely; if this ever holds because
    # the working tree ran instead, the comparison below would be vacuous.
    assert base.base_sha == BASE_SHA
    assert len(base_source_sha256) == 64
    assert all(
        UNDER10_ASSESSMENT_KEY not in analysis["data_quality"]
        for analysis in base_capture["analyses"].values()
    )

    head_results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        provider="twelve_data",
        benchmark_subject="SPY",
        store=store,
        config_path=default_us_scoring_config_path(),
    )
    head_capture = _capture(head_results[0].run)

    # 0. Both sides actually built medium/long advisory forecasts: this is
    # the non-vacuous precondition for every assertion below. Without it,
    # "predictions equal" could hold merely because neither side attempted
    # 6m/12m/3y/5y at all.
    base_advisory = _advisory_predictions(base_capture)
    head_advisory = _advisory_predictions(head_capture)
    expected_advisory_keys = {
        f"{ticker}:{horizon}:advisory"
        for ticker in ("DIF1", "DIF2")
        for horizon in ("6m", "12m", "3y", "5y")
    }
    assert set(base_advisory) == expected_advisory_keys
    assert set(head_advisory) == expected_advisory_keys
    for row in base_advisory.values():
        assert row["calculation"]
    for row in head_advisory.values():
        assert row["calculation"]

    # 1. The non-candidate payload is equal in full, key set included.
    assert head_capture["analyses"]["DIF2"] == base_capture["analyses"]["DIF2"]
    assert UNDER10_ASSESSMENT_KEY not in head_capture["analyses"]["DIF2"]["data_quality"]

    # 2. The candidate is equal after removing only the new key.
    head_candidate = head_capture["analyses"]["DIF1"]
    assert UNDER10_ASSESSMENT_KEY in head_candidate["data_quality"]
    stripped = {
        **head_candidate,
        "data_quality": {
            key: value
            for key, value in head_candidate["data_quality"].items()
            if key != UNDER10_ASSESSMENT_KEY
        },
    }
    assert stripped == base_capture["analyses"]["DIF1"]

    # 3. Every prediction payload and manifest -- decision *and* advisory
    # 6m/12m/3y/5y -- is unchanged.
    assert head_capture["predictions"] == base_capture["predictions"]


@pytest.mark.django_db
@pytest.mark.skipif(
    not base_service_available(),
    reason=f"base revision {BASE_SHA} is not in the local git object database",
)
def test_a_head_only_indicator_mutation_is_detected_not_masked(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: the harness must not compare a mutated tree to itself.

    `base_research_service` binds `research/indicators.py` to the pinned base
    bytes inside its own freshly executed module; deliberately mutating an
    existing indicator output on the *live* module and its already-bound
    `stanstock.research.service` reference -- exactly what a head-only
    regression to `calculate_indicators` would look like -- must therefore
    survive into the head capture while the base capture, computed while the
    same mutation is active, stays on the genuine base bytes. If this
    assertion ever passes with equal captures, the harness has stopped
    proving anything: it would be comparing the working tree against itself.
    """
    import stanstock.research.indicators as live_indicators
    import stanstock.research.service as live_service

    store = AssetStore(tmp_path)
    snapshot = _differential_fixture(store)

    original_calculate_indicators = live_indicators.calculate_indicators

    def _mutated_calculate_indicators(*args: Any, **kwargs: Any) -> Any:
        result = original_calculate_indicators(*args, **kwargs)
        if "last_close" in result.values:
            result.values["last_close"] = result.values["last_close"] + 0.01
        return result

    # Both bindings are patched because `from X import Y` copies a reference
    # at import time: patching only `research.indicators` would not reach
    # `research.service`'s already-bound name, and patching only
    # `research.service` would not simulate a real source-level edit to
    # `indicators.py` (which every fresh importer would pick up).
    monkeypatch.setattr(live_indicators, "calculate_indicators", _mutated_calculate_indicators)
    monkeypatch.setattr(live_service, "calculate_indicators", _mutated_calculate_indicators)

    with base_research_service() as base:
        base_results = base.service.analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            provider="twelve_data",
            benchmark_subject="SPY",
            store=store,
            config_path=default_us_scoring_config_path(),
        )
    base_capture = _capture(base_results[0].run)

    head_results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        provider="twelve_data",
        benchmark_subject="SPY",
        store=store,
        config_path=default_us_scoring_config_path(),
    )
    head_capture = _capture(head_results[0].run)

    head_candidate = head_capture["analyses"]["DIF1"]
    stripped = {
        **head_candidate,
        "data_quality": {
            key: value
            for key, value in head_candidate["data_quality"].items()
            if key != UNDER10_ASSESSMENT_KEY
        },
    }
    base_candidate = base_capture["analyses"]["DIF1"]
    # The mutation changed the reported current price and therefore the
    # captured `current_price` field: base stayed genuinely unaffected, so
    # this is a real difference the differential test would have failed on.
    assert stripped["current_price"] != base_candidate["current_price"]
    assert stripped != base_candidate


@pytest.mark.django_db
@pytest.mark.skipif(
    not base_service_available(),
    reason=f"base revision {BASE_SHA} is not in the local git object database",
)
def test_a_head_only_asof_clipping_mutation_is_detected_not_masked(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: the harness must bind `data/asof.py`, not just its siblings.

    Base `research/service.py` does ``from stanstock.data.asof import
    AsOfData``; if `data.asof` were not bound to base bytes, a live mutation
    to its point-in-time clipping (e.g. narrowing ``<= through_date`` to
    ``< through_date``, dropping the boundary session a correct clip would
    keep) would leak into the "base" run too, since the swapped base
    `service.py` would still resolve `AsOfData` against the *live* `data.asof`
    module. This reproduces exactly that mutation -- reported and verified by
    research-integrity against the pre-correction harness -- and proves the
    harness now isolates it.
    """
    import stanstock.data.asof as live_asof

    store = AssetStore(tmp_path)
    snapshot = _differential_fixture(store)

    original_clip = live_asof._clip_to_through_date

    def _mutated_clip(
        frame: pl.DataFrame,
        through_date: date,
        *,
        asset: DataAsset,
        relative_path: str,
    ) -> Any:
        result = original_clip(
            frame,
            through_date,
            asset=asset,
            relative_path=relative_path,
        )
        # Reproduces "clip <= through_date -> < through_date": additionally
        # drops the boundary-date row a correct `<=` clip would keep.
        mutated_frame = result.frame.filter(pl.col("date") != through_date)
        return live_asof.PriceFrameRead(
            asset=result.asset,
            frame=mutated_frame,
            invalid_session_date_rows=result.invalid_session_date_rows,
        )

    # `price_frame_with_diagnostics` calls `_clip_to_through_date` via a
    # module-global lookup resolved at call time against `data.asof`'s own
    # `__globals__`, so patching this one name reaches every caller -- both
    # `AsOfData.price_frame` and `.price_frame_with_diagnostics` -- regardless
    # of which name (`stanstock.data.asof.AsOfData` or
    # `stanstock.research.service.AsOfData`) was used to reach the class.
    monkeypatch.setattr(live_asof, "_clip_to_through_date", _mutated_clip)

    with base_research_service() as base:
        base_results = base.service.analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            provider="twelve_data",
            benchmark_subject="SPY",
            store=store,
            config_path=default_us_scoring_config_path(),
        )
    base_capture = _capture(base_results[0].run)

    head_results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        provider="twelve_data",
        benchmark_subject="SPY",
        store=store,
        config_path=default_us_scoring_config_path(),
    )
    head_capture = _capture(head_results[0].run)

    head_candidate = head_capture["analyses"]["DIF1"]
    stripped = {
        **head_candidate,
        "data_quality": {
            key: value
            for key, value in head_candidate["data_quality"].items()
            if key != UNDER10_ASSESSMENT_KEY
        },
    }
    base_candidate = base_capture["analyses"]["DIF1"]
    # Base stayed on the real, unmutated clip (900 raw calendar-day rows, all
    # <= through_date): its observation count is unaffected. Head lost
    # exactly the boundary (most recent) session.
    assert base_candidate["data_quality"]["observation_count"] == 900.0
    assert stripped["data_quality"]["observation_count"] == 899.0
    assert (
        stripped["data_quality"]["observation_count"]
        != base_candidate["data_quality"]["observation_count"]
    )
    assert stripped != base_candidate


def test_dependency_binding_covers_every_changed_module_base_service_imports() -> None:
    """The bound module set must never silently fall behind base service's own imports.

    Not a hardcoded tautology of `DEPENDENCY_MODULES`: base
    `research/service.py`'s actual `import` statements are parsed directly
    from its base bytes via `ast`, and each imported `stanstock.*` module's
    base bytes are diffed against the current working-tree file. Any such
    module whose content differs from base must already be bound, or a
    head-only mutation to it could leak into the "base" differential run
    exactly as `data.asof` once did before this correction. A module base
    `service.py` does not import at all (e.g. `research/under10.py`, which
    does not exist at base) is correctly never required here.
    """
    if not base_service_available():
        pytest.skip(f"base revision {BASE_SHA} is not in the local git object database")

    bound_names = {name for name, _relative_path in DEPENDENCY_MODULES}
    imported = base_service_first_party_import_names()
    assert imported, (
        "base research/service.py imports no stanstock.* modules; ast parsing likely broke"
    )

    changed_and_imported: set[str] = set()
    for module_name in imported:
        relative_path = module_relative_path(module_name)
        head_path = REPO_ROOT / relative_path
        if not head_path.exists():
            # Base imports a module absent from the working tree entirely --
            # a different, unrelated problem this assertion does not police.
            continue
        if head_path.read_bytes() != _read_base_source(relative_path):
            changed_and_imported.add(module_name)

    missing = changed_and_imported - bound_names
    assert missing == set(), (
        f"base research/service.py imports {sorted(missing)}, whose content "
        "differs from base, but it is not bound in DEPENDENCY_MODULES"
    )
    # Guards the guard: must not pass merely because nothing changed.
    assert changed_and_imported, (
        "expected at least one base-imported module to differ from base in "
        "this slice; found none, which would make the assertion above vacuous"
    )


@pytest.mark.parametrize("failing_index", range(len(DEPENDENCY_MODULES)))
def test_a_load_failure_at_any_position_restores_every_untouched_module_binding(
    monkeypatch: pytest.MonkeyPatch,
    failing_index: int,
) -> None:
    """A read/exec failure partway through binding must not corrupt modules
    that were never reached.

    `base_research_service()` must snapshot every `DEPENDENCY_MODULES` name's
    prior `sys.modules` binding *before* any loading begins. Snapshotting
    incrementally instead -- recording each name's original only immediately
    before swapping it -- leaves every name *after* the failing position with
    no recorded snapshot at all once a failure occurs partway through. The
    `finally` restoration step then cannot distinguish "never touched, so
    leave alone" from "was genuinely absent before, so pop", and incorrectly
    pops perfectly live, untouched modules out of `sys.modules`.
    """
    if not base_service_available():
        pytest.skip(f"base revision {BASE_SHA} is not in the local git object database")

    import base_service as base_service_module

    failing_name, failing_path = DEPENDENCY_MODULES[failing_index]
    prior = {name: sys.modules.get(name) for name, _relative_path in DEPENDENCY_MODULES}
    real_read_base_source = base_service_module._read_base_source

    def _flaky_read_base_source(relative_path: str) -> bytes:
        if relative_path == failing_path:
            raise RuntimeError(f"synthetic read failure for {relative_path}")
        return real_read_base_source(relative_path)

    monkeypatch.setattr(base_service_module, "_read_base_source", _flaky_read_base_source)

    try:
        with (
            pytest.raises(
                RuntimeError, match=re.escape(f"synthetic read failure for {failing_path}")
            ),
            base_research_service(),
        ):
            raise AssertionError("unreachable: the injected read failure must prevent entry")

        for name, _relative_path in DEPENDENCY_MODULES:
            assert sys.modules.get(name) is prior[name], (
                f"{name} was not restored to its exact prior binding after an "
                f"injected failure while loading {failing_name!r} "
                f"(failing_index={failing_index})"
            )
    finally:
        # Restore unconditionally, whether the assertions above passed or
        # failed: a real corruption caught here must not leak into sibling
        # parametrized runs (or unrelated tests later in the session) and
        # mask itself by becoming the new "prior" baseline for the next one.
        for name, _relative_path in DEPENDENCY_MODULES:
            if prior[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior[name]


@pytest.mark.django_db
@pytest.mark.skipif(
    not base_service_available(),
    reason=f"base revision {BASE_SHA} is not in the local git object database",
)
def test_base_and_head_agree_for_the_single_listing_entry_point(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="ONE")
    _price_asset(store, "ONE", close=4.25)
    _sec_facts(listing, _sec_asset(store))
    _twelve_data_record("basic")

    with base_research_service() as base:
        base_persisted = base.service.analyze_listing(
            listing=listing,
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            provider="twelve_data",
            store=store,
            config_path=default_us_scoring_config_path(),
        )
    base_capture = _capture(base_persisted.run)

    head_persisted = _analyze(listing, snapshot, store)
    head_capture = _capture(head_persisted.run)

    head_analysis = head_capture["analyses"]["ONE"]
    stripped = {
        **head_analysis,
        "data_quality": {
            key: value
            for key, value in head_analysis["data_quality"].items()
            if key != UNDER10_ASSESSMENT_KEY
        },
    }
    assert stripped == base_capture["analyses"]["ONE"]
    assert head_capture["predictions"] == base_capture["predictions"]


@pytest.mark.django_db
@pytest.mark.skipif(
    not base_service_available(),
    reason=f"base revision {BASE_SHA} is not in the local git object database",
)
def test_downstream_opportunity_and_basket_behavior_is_unchanged(tmp_path) -> None:
    from django.contrib.auth import get_user_model

    from stanstock.portfolio.service import PortfolioValuationError, build_sample_portfolio
    from stanstock.research.opportunities import assess_opportunity
    from stanstock.research.outcomes import evaluate_predictions
    from stanstock.research.provenance import source_data_mode
    from stanstock.research.reporting import reportable_prediction_filter

    store = AssetStore(tmp_path)
    snapshot = _differential_fixture(store)
    owner = get_user_model().objects.create_user(username="under10-owner")

    with base_research_service() as base:
        base_results = base.service.analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            provider="twelve_data",
            benchmark_subject="SPY",
            store=store,
            config_path=default_us_scoring_config_path(),
        )
    head_results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        provider="twelve_data",
        benchmark_subject="SPY",
        store=store,
        config_path=default_us_scoring_config_path(),
    )

    def qualification(results: list[Any]) -> dict[str, Any]:
        return {
            result.analysis.listing.ticker: {
                "eligible": assess_opportunity(result.analysis).eligible,
                "label": assess_opportunity(result.analysis).label,
                "blocking_reasons": assess_opportunity(result.analysis).blocking_reasons,
                "criteria": assess_opportunity(result.analysis).criteria,
                "source_mode": source_data_mode(result.analysis.data_quality),
            }
            for result in results
        }

    def basket(run: AnalysisRun) -> str:
        try:
            portfolio, _created = build_sample_portfolio(owner=owner, source_run=run)
        except PortfolioValuationError as error:
            return f"refused:{error}"
        return "built:" + ",".join(
            sorted(holding.listing.ticker for holding in portfolio.holdings.all())
        )

    def outcomes(run: AnalysisRun) -> dict[str, tuple[str, str]]:
        evaluations = evaluate_predictions(
            list(Prediction.objects.filter(analysis__run=run).order_by("listing__ticker", "pk")),
            provider="twelve_data",
            evaluation_date=TARGET_DATE + timedelta(days=1),
            evaluation_time=DECISION_TIME + timedelta(days=1),
            store=store,
        )
        return {
            f"{result.prediction.listing.ticker}:{result.prediction.horizon}": (
                result.outcome.status,
                result.resolution,
            )
            for result in evaluations
        }

    def reportable(run: AnalysisRun) -> int:
        # The *canonical* filter's observation key deliberately excludes
        # run/model identity, because a genuine on-time reissue of the same
        # (listing, target_date, horizon, evidence_role, method_version,
        # config_hash, price_provider) must count once in production. This
        # differential test's base and head runs share exactly that key for
        # both listings by construction (same fixture, same target date), so
        # the canonical filter would make base and head *compete* for
        # canonicality against each other and split an arbitrary, UUID-
        # order-dependent count between them -- an artifact of running two
        # runs of "the same" observation side by side, not a real regression.
        # The plain (non-deduplicating) filter answers the actual question
        # this test asks -- "does this run's own evidence qualify as
        # reportable at all" -- without that cross-run collision.
        return Prediction.objects.filter(
            reportable_prediction_filter(),
            analysis__run=run,
        ).count()

    head_qualification = qualification(head_results)
    head_basket = basket(head_results[0].run)
    head_outcomes = outcomes(head_results[0].run)
    head_reportable = reportable(head_results[0].run)
    base_basket = basket(base_results[0].run)
    base_reportable = reportable(base_results[0].run)

    # Guard against a vacuous comparison: the basket must have actually been
    # built (not refused for lack of an eligible candidate or current market
    # row), and reportable evidence must actually exist, on both sides,
    # before any equality assertion is meaningful.
    assert set(head_qualification) == {"DIF1", "DIF2"}
    assert head_qualification["DIF2"]["eligible"] is True
    assert head_basket.startswith("built:"), head_basket
    assert base_basket.startswith("built:"), base_basket
    assert head_outcomes
    assert Prediction.objects.filter(analysis__run=head_results[0].run).count() > 0
    assert head_reportable > 0
    assert base_reportable > 0

    assert head_qualification == qualification(base_results)
    assert head_basket == base_basket
    assert head_outcomes == outcomes(base_results[0].run)
    assert head_reportable == base_reportable
    # The Under-$10 candidate stays excluded from new allocation on both sides.
    assert head_qualification["DIF1"]["eligible"] is False
    assert "DIF1" not in head_basket


@pytest.mark.django_db
def test_full_reuse_and_price_only_query_pass_identical_qualified_sec_evidence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both fact-loading paths pass only the exact qualified sequence.

    The baseline is generated before the three canonical-concept mismatch
    forms are inserted. Full-analysis reuse and the price-only targeted query
    must then produce the same payload bytes and pass the same sorted SEC
    fact objects to the builder. The third mismatch (foreign fact on an SEC
    source) independently guards the fact-provider half of the conjunction.
    """
    import stanstock.research.service as service_module

    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="QUALPATH")
    _price_asset(store, "QUALPATH", close=4.25)
    qualified_asset = _sec_asset(store, key="qualified-path")
    qualified_facts = _sec_facts(listing, qualified_asset)
    _twelve_data_record("basic")

    passed_fact_ids: list[tuple[str, ...]] = []
    original_builder = service_module.build_under10_assessment

    def recording_builder(**kwargs: Any) -> dict[str, Any]:
        facts = kwargs["facts"]
        passed_fact_ids.append(tuple(str(fact.id) for fact in facts))
        return original_builder(**kwargs)

    monkeypatch.setattr(service_module, "build_under10_assessment", recording_builder)

    def run(config_path) -> Any:
        return analyze_listing(
            listing=listing,
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            provider="twelve_data",
            store=store,
            config_path=config_path,
        )

    sec_only_full = run(None)
    sec_only_price = run(default_us_scoring_config_path())
    sec_only_payload = sec_only_full.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    assert sec_only_full.analysis.data_quality["fundamentals_used"] is True
    assert sec_only_price.analysis.data_quality["fundamentals_used"] is False
    assert sec_only_price.analysis.data_quality[UNDER10_ASSESSMENT_KEY] == sec_only_payload

    foreign_asset = _sec_asset(
        store,
        key="foreign-path",
        provider="other_provider",
    )
    foreign_facts = _sec_facts(
        listing,
        foreign_asset,
        provider="other_provider",
    )
    mismatched_asset = _sec_asset(
        store,
        key="mismatched-source-path",
        provider="other_source_provider",
    )
    mismatched_facts = _sec_facts(
        listing,
        mismatched_asset,
        provider=SEC_PROVIDER,
        source_revision=2,
    )
    sec_source_for_foreign_facts = _sec_asset(
        store,
        key="sec-source-foreign-fact-path",
    )
    foreign_on_sec_facts = _sec_facts(
        listing,
        sec_source_for_foreign_facts,
        provider="other_fact_provider",
    )

    mixed_full = run(None)
    mixed_price = run(default_us_scoring_config_path())
    mixed_full_payload = mixed_full.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    mixed_price_payload = mixed_price.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    assert mixed_full_payload == sec_only_payload
    assert mixed_price_payload == sec_only_payload

    expected_fact_ids = tuple(sorted(str(fact.id) for fact in qualified_facts))
    assert passed_fact_ids == [expected_fact_ids] * 4
    unqualified_fact_ids = {
        str(fact.id) for fact in (*foreign_facts, *mismatched_facts, *foreign_on_sec_facts)
    }
    assert unqualified_fact_ids.isdisjoint(mixed_full_payload["solvency"]["assessed_fact_ids"])
    unqualified_asset_ids = {
        str(foreign_asset.id),
        str(mismatched_asset.id),
        str(sec_source_for_foreign_facts.id),
    }
    assert unqualified_asset_ids.isdisjoint(
        reference["id"] for reference in mixed_full_payload["solvency"]["assessed_assets"]
    )


@pytest.mark.django_db
def test_full_analysis_mode_reuses_loaded_facts_without_a_second_query(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="FULL")
    _price_asset(store, "FULL", close=4.25)
    _sec_facts(listing, _sec_asset(store))
    _twelve_data_record("basic")

    with CaptureQueriesContext(connection) as captured:
        persisted = analyze_listing(
            listing=listing,
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            provider="twelve_data",
            store=store,
        )

    fact_queries = [
        query for query in captured.captured_queries if "data_fundamentalfact" in query["sql"]
    ]
    payload = persisted.analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    assert persisted.analysis.data_quality["fundamentals_used"] is True
    # Full-analysis mode already loaded every visible fact; the shadow path
    # narrows that list in memory instead of re-reading it.
    assert len(fact_queries) == 1
    assert payload["solvency"]["status"] == "no_adverse_evidence_observed"
    assert payload["solvency"]["assessed_fact_ids"]


@pytest.mark.django_db
def test_assessed_asset_references_do_not_trigger_lazy_asset_queries(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    lean = _listing(snapshot, ticker="LAZY1")
    _price_asset(store, "LAZY1", close=4.25)
    lean_asset = _sec_asset(store, key="lean-facts")
    _sec_facts(lean, lean_asset)

    rich = _listing(snapshot, ticker="LAZY2")
    _price_asset(store, "LAZY2", close=4.25)
    rich_asset = _sec_asset(store, key="rich-facts")
    _sec_facts(rich, rich_asset)
    _sec_facts(
        rich,
        _sec_asset(store, key="rich-facts-prior"),
        instant_date=date(2024, 12, 31),
        annual_period=(date(2024, 1, 1), date(2024, 12, 31)),
    )

    def asset_query_count(listing: Listing) -> int:
        with CaptureQueriesContext(connection) as captured:
            _analyze(listing, snapshot, store)
        return len(
            [query for query in captured.captured_queries if "data_dataasset" in query["sql"]]
        )

    lean_count = asset_query_count(lean)
    rich_count = asset_query_count(rich)

    payload = StockAnalysis.objects.get(listing=lean).data_quality[UNDER10_ASSESSMENT_KEY]
    # Twice the evidence must not cost more asset queries: a lazy
    # `fact.source_asset` access would add one per fact.
    assert rich_count == lean_count
    assert payload["solvency"]["assessed_assets"] == [
        {"id": str(lean_asset.id), "sha256": lean_asset.sha256}
    ]
    rich_payload = StockAnalysis.objects.get(listing=rich).data_quality[UNDER10_ASSESSMENT_KEY]
    assert len(rich_payload["solvency"]["assessed_assets"]) == 2
    assert str(rich_asset.id) in {
        entry["id"] for entry in rich_payload["solvency"]["assessed_assets"]
    }


@pytest.mark.django_db
def test_duplicated_payload_fields_equal_their_existing_owners(tmp_path) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="OWNR")
    price_asset = _price_asset(store, "OWNR", close=4.25)
    _sec_facts(listing, _sec_asset(store))
    _twelve_data_record("basic")

    persisted = _analyze(listing, snapshot, store)

    analysis = StockAnalysis.objects.get(pk=persisted.analysis.pk)
    payload = analysis.data_quality[UNDER10_ASSESSMENT_KEY]
    assert payload["evaluated_for"]["reference_close"] == str(analysis.current_price)
    assert payload["evaluated_for"]["target_date"] == analysis.run.target_date.isoformat()
    assert payload["evaluated_for"]["data_cutoff"] == analysis.run.data_cutoff.isoformat()
    assert payload["code_revision"] == analysis.run.code_revision
    assert payload["liquidity"]["price_asset"] == {
        "id": str(price_asset.id),
        "sha256": price_asset.sha256,
    }
    assert (
        payload["liquidity"]["price_asset"]["id"]
        == (analysis.data_quality["price_source"]["asset_id"])
    )
    assert payload["split_verification"]["provider"] == "twelve_data"


@pytest.mark.django_db
def test_enrichment_preserves_every_other_computation_field(tmp_path) -> None:
    """`replace()` may only add the nested key; nothing else may move."""
    import dataclasses

    from stanstock.data.asof import AsOfData
    from stanstock.research.config import load_scoring_config
    from stanstock.research.service import _compute_listing_from_asof, _with_under10_assessment

    store = AssetStore(tmp_path)
    snapshot = _snapshot()
    listing = _listing(snapshot, ticker="RPL")
    _price_asset(store, "RPL", close=4.25)
    _sec_facts(listing, _sec_asset(store))

    asof = AsOfData(DECISION_TIME, store)
    enriched = _compute_listing_from_asof(
        listing=listing,
        asof=asof,
        provider="twelve_data",
        config=load_scoring_config(default_us_scoring_config_path()),
        decision_time=DECISION_TIME,
        issued_on_time=False,
        provider_plan="basic",
        code_revision_value="0" * 40,
        target_date=TARGET_DATE,
    )
    # Rebuild the pre-enrichment view by stripping the one added key.
    stripped = dataclasses.replace(
        enriched,
        data_quality={
            key: value
            for key, value in enriched.data_quality.items()
            if key != UNDER10_ASSESSMENT_KEY
        },
    )
    reapplied = _with_under10_assessment(
        stripped,
        listing=listing,
        asof=asof,
        provider="twelve_data",
        provider_plan="basic",
        loaded_facts=[],
        loaded_facts_cover_all_concepts=False,
        price_frame=asof.price_frame(
            provider="twelve_data", subject="RPL", through_date=TARGET_DATE
        ),
        price_asset=enriched.price_asset,
        invalid_session_date_rows=0,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        issued_on_time=False,
        code_revision_value="0" * 40,
    )

    changed = {
        field.name
        for field in dataclasses.fields(enriched)
        if getattr(enriched, field.name) != getattr(stripped, field.name)
    }
    assert changed == {"data_quality"}
    assert enriched.source_assets is stripped.source_assets
    assert reapplied.data_quality == enriched.data_quality

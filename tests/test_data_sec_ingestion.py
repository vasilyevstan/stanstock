from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Barrier

import pytest
from django.core.exceptions import ValidationError
from django.db import IntegrityError, connection, connections

from stanstock.data import sec_ingestion
from stanstock.data.asof import AsOfData
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.live_us import UsUniverseConfig
from stanstock.data.models import (
    Company,
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    Listing,
    ProviderRecord,
    Region,
    Security,
    SourceObservationEvent,
)
from stanstock.data.providers import sec
from stanstock.data.providers.contracts import FundamentalSourcePayload
from stanstock.data.providers.exceptions import ProviderResponseError
from stanstock.data.sec_config import (
    SecCikConfig,
    SecCikMapping,
    load_sec_fundamentals_config,
)
from stanstock.data.sec_fundamentals import select_latest_fact_vintages
from stanstock.data.sec_ingestion import (
    MAPPING_KIND,
    SecRequestBudget,
    parse_sec_mapping,
    run_sec_ingestion,
)
from stanstock.research.long_forecast_config import load_long_forecast_config

pytestmark = pytest.mark.django_db

RETRIEVED_AT = datetime(2026, 8, 15, 12, tzinfo=UTC)
MAPPING_BYTES = json.dumps(
    {
        "fields": ["cik", "name", "ticker", "exchange"],
        "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"]],
    },
    sort_keys=True,
).encode()
SUBMISSIONS_BYTES = json.dumps(
    {
        "cik": 320193,
        "name": "Apple Inc.",
        "sic": "3571",
        "sicDescription": "Electronic Computers",
        "tickers": ["AAPL"],
        "exchanges": ["Nasdaq"],
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-26-000001",
                    "0000320193-26-000002",
                ],
                "filingDate": ["2026-04-15", "2026-07-15"],
                "acceptanceDateTime": [
                    "2026-04-15T20:15:00Z",
                    "",
                ],
                "form": ["10-Q", "10-Q/A"],
                "reportDate": ["2026-03-31", "2026-06-30"],
                "primaryDocument": ["q1.htm", "q2a.htm"],
            },
            "files": [
                {
                    "name": "CIK0000320193-submissions-001.json",
                    "filingCount": 1,
                }
            ],
        },
    },
    sort_keys=True,
).encode()
HISTORY_BYTES = json.dumps(
    {
        "accessionNumber": ["0000320193-26-000000"],
        "filingDate": ["2026-02-15"],
        "acceptanceDateTime": ["2026-02-15T21:00:00Z"],
        "form": ["10-K"],
        "reportDate": ["2025-12-31"],
        "primaryDocument": ["annual.htm"],
    },
    sort_keys=True,
).encode()


def _companyfacts_bytes(*, q1_revenue: int = 100) -> bytes:
    return json.dumps(
        {
            "cik": 320193,
            "entityName": "Apple Inc.",
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-01-01",
                                    "end": "2025-12-31",
                                    "val": 350,
                                    "accn": "0000320193-26-000000",
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-02-15",
                                    "frame": "CY2025",
                                },
                                {
                                    "start": "2026-01-01",
                                    "end": "2026-03-31",
                                    "val": q1_revenue,
                                    "accn": "0000320193-26-000001",
                                    "fy": 2026,
                                    "fp": "Q1",
                                    "form": "10-Q",
                                    "filed": "2026-04-15",
                                    "frame": "CY2026Q1",
                                },
                                {
                                    "start": "2026-01-01",
                                    "end": "2026-06-30",
                                    "val": 230,
                                    "accn": "0000320193-26-000002",
                                    "fy": 2026,
                                    "fp": "Q2",
                                    "form": "10-Q/A",
                                    "filed": "2026-07-15",
                                },
                                {
                                    "start": "2026-04-01",
                                    "end": "2026-06-30",
                                    "val": 130,
                                    "accn": "0000320193-26-000002",
                                    "fy": 2026,
                                    "fp": "Q2",
                                    "form": "10-Q/A",
                                    "filed": "2026-07-15",
                                    "frame": "CY2026Q2",
                                },
                            ]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2026-06-30",
                                    "val": 500,
                                    "accn": "0000320193-26-000002",
                                    "fy": 2026,
                                    "fp": "Q2",
                                    "form": "10-Q/A",
                                    "filed": "2026-07-15",
                                    "frame": "CY2026Q2I",
                                }
                            ]
                        }
                    },
                }
            },
        },
        sort_keys=True,
    ).encode()


def _payload(subject: str, content: bytes, source_url: str) -> FundamentalSourcePayload:
    return FundamentalSourcePayload(
        provider="sec",
        subject=subject,
        content=content,
        content_type="application/json",
        retrieved_at=RETRIEVED_AT,
        source_url=source_url,
    )


def _universe() -> UsUniverseConfig:
    return UsUniverseConfig(
        slug="test-us",
        name="Test US",
        description="",
        config_version="test-us-v1",
        country="United States",
        instrument_type="Common Stock",
        currency="USD",
        exchanges=("NASDAQ",),
        benchmark_symbol="SPY",
        benchmark_currency="USD",
        benchmark_type="ETF",
        history_years=7,
        minimum_history_sessions=1260,
        price_adjustment="splits",
        minimum_eligible=1,
        maximum_symbols=1,
        symbols=("AAPL",),
        raw={},
    )


def _cik_config() -> SecCikConfig:
    digest = hashlib.sha256(MAPPING_BYTES).hexdigest()
    mapping = SecCikMapping(
        symbol="AAPL",
        cik="0000320193",
        official_ticker="AAPL",
        exchange="Nasdaq",
        company_name="Apple Inc.",
        reason="",
    )
    return SecCikConfig(
        config_version="test-cik-v1",
        universe_config_version="test-us-v1",
        source_sha256=digest,
        mappings={"AAPL": mapping},
        excluded={},
        raw={},
        config_hash="c" * 64,
    )


def _listing(*, security_type: str = Security.SecurityType.COMMON_STOCK) -> Listing:
    company = Company.objects.create(name="Apple Inc.", country="US")
    security = Security.objects.create(
        company=company,
        security_type=security_type,
        name="Apple Inc.",
    )
    return Listing.objects.create(
        security=security,
        ticker="AAPL",
        exchange_mic="XNAS",
        provider_symbol="AAPL",
        currency="USD",
        region=Region.US,
        is_active=True,
    )


def _mapping_asset(store: AssetStore) -> DataAsset:
    stored = store.write_bytes("raw/sec/test-mapping.json", MAPPING_BYTES)
    return register_asset(
        provider="sec",
        kind=MAPPING_KIND,
        subject="company_tickers_exchange",
        stored=stored,
        retrieved_at=RETRIEVED_AT,
        available_at=RETRIEVED_AT,
    )


def _patch_provider(
    monkeypatch: pytest.MonkeyPatch,
    companyfacts: dict[str, bytes],
    submissions: dict[str, bytes] | None = None,
) -> None:
    submissions = submissions or {"content": SUBMISSIONS_BYTES}
    monkeypatch.setattr(
        sec,
        "fetch_submissions",
        lambda cik: _payload(
            "0000320193",
            submissions["content"],
            "https://data.sec.gov/submissions/CIK0000320193.json",
        ),
    )
    monkeypatch.setattr(
        sec,
        "fetch_submissions_history",
        lambda filename: _payload(
            filename,
            HISTORY_BYTES,
            f"https://data.sec.gov/submissions/{filename}",
        ),
    )
    monkeypatch.setattr(
        sec,
        "fetch_companyfacts",
        lambda cik: _payload(
            "0000320193",
            companyfacts["content"],
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        ),
    )


def test_parse_sec_mapping_uses_field_positions() -> None:
    rows = parse_sec_mapping(MAPPING_BYTES)

    assert len(rows) == 1
    assert rows[0].cik == "0000320193"
    assert rows[0].ticker == "AAPL"
    assert rows[0].exchange == "Nasdaq"


def test_sec_ingestion_preserves_raw_history_and_point_in_time_facts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    listing = _listing()
    store = AssetStore(root=tmp_path)
    _mapping_asset(store)
    # Distinct payloads must carry distinct retrieval timestamps: one
    # observation instant names one content, so a pinned clock across
    # changing content is now an explicit ambiguity.
    companyfacts: dict[str, object] = {
        "content": _companyfacts_bytes(),
        "retrieved_at": RETRIEVED_AT,
    }
    submissions: dict[str, object] = {
        "content": SUBMISSIONS_BYTES,
        "retrieved_at": RETRIEVED_AT,
    }
    _patch_provider_with_clock(monkeypatch, companyfacts, submissions)
    budget = SecRequestBudget(
        requests_per_second=5,
        enforce_spacing=False,
        require_enabled=False,
    )

    first = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )

    listing.security.company.refresh_from_db()
    assert listing.security.company.cik == "0000320193"
    assert first.raw_assets_created == 3
    assert first.facts_created == 5
    facts = FundamentalFact.objects.filter(company=listing.security.company, provider="sec")
    assert facts.count() == 5
    q2_revenue = facts.filter(concept="revenue", period_end=date(2026, 6, 30))
    assert q2_revenue.count() == 2
    assert q2_revenue.values("period_identity").distinct().count() == 2
    fallback = q2_revenue.first()
    assert fallback is not None
    assert fallback.availability_basis == "filed_date_next_day"
    assert fallback.available_at.astimezone(UTC) == datetime(2026, 7, 16, 4, tzinfo=UTC)
    assert "availability_filed_date_fallback" in fallback.quality_flags
    exact = facts.get(period_end=date(2026, 3, 31))
    assert exact.acceptance_at == datetime(2026, 4, 15, 20, 15, tzinfo=UTC)
    assert exact.source_asset.kind == "sec_companyfacts"
    assert FundamentalFactEvidence.objects.filter(
        fact=exact,
        role=FundamentalFactEvidence.Role.FILING,
        source_asset__kind="sec_submissions",
    ).exists()
    assert (
        CompanyClassificationObservation.objects.get(
            company=listing.security.company,
            scheme="sec_sic",
        ).code
        == "3571"
    )

    second = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )

    assert second.raw_assets_created == 0
    assert second.raw_assets_reused == 2
    assert second.facts_created == 0
    assert second.facts_reused == 0

    companyfacts["content"] = _companyfacts_bytes(q1_revenue=101)
    companyfacts["retrieved_at"] = RETRIEVED_AT + timedelta(days=1)
    submissions["content"] = SUBMISSIONS_BYTES.replace(
        b"Electronic Computers",
        b"Electronic Computers Updated",
    )
    submissions["retrieved_at"] = RETRIEVED_AT + timedelta(days=1)
    revised = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )

    assert revised.facts_created == 1
    revisions = facts.filter(
        concept="revenue",
        period_end=date(2026, 3, 31),
    ).order_by("source_revision")
    assert list(revisions.values_list("source_revision", flat=True)) == [1, 2]
    assert list(revisions.values_list("value", flat=True)) == [100, 101]

    companyfacts["content"] = _companyfacts_bytes(q1_revenue=100)
    companyfacts["retrieved_at"] = RETRIEVED_AT + timedelta(days=2)
    submissions_content = submissions["content"]
    assert isinstance(submissions_content, bytes)
    submissions["content"] = submissions_content.replace(
        b"Electronic Computers Updated",
        b"Electronic Computers Revised",
    )
    submissions["retrieved_at"] = RETRIEVED_AT + timedelta(days=2)
    reverted = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )

    assert reverted.facts_created == 1
    assert list(revisions.values_list("source_revision", flat=True)) == [1, 2, 3]
    assert list(revisions.values_list("value", flat=True)) == [100, 101, 100]


def test_persisted_companyfacts_replays_after_interrupted_normalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _listing()
    store = AssetStore(root=tmp_path)
    _mapping_asset(store)
    companyfacts = {"content": _companyfacts_bytes()}
    _patch_provider(monkeypatch, companyfacts)
    original_fetch = sec.fetch_companyfacts
    original_normalize = sec_ingestion._normalize_companyfacts
    calls = {"fetch": 0, "normalize": 0}

    def fetch_companyfacts(cik: str) -> FundamentalSourcePayload:
        calls["fetch"] += 1
        return original_fetch(cik)

    def normalize_companyfacts(**kwargs: object) -> tuple[int, int]:
        calls["normalize"] += 1
        if calls["normalize"] == 1:
            raise RuntimeError("simulated normalization interruption")
        return original_normalize(**kwargs)

    monkeypatch.setattr(sec, "fetch_companyfacts", fetch_companyfacts)
    monkeypatch.setattr(sec_ingestion, "_normalize_companyfacts", normalize_companyfacts)
    budget = SecRequestBudget(
        requests_per_second=5,
        enforce_spacing=False,
        require_enabled=False,
    )

    with pytest.raises(RuntimeError, match="normalization interruption"):
        run_sec_ingestion(
            config=load_sec_fundamentals_config(),
            cik_config=_cik_config(),
            universe_config=_universe(),
            target_date=date(2026, 8, 14),
            store=store,
            budget=budget,
        )

    recovered = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )
    skipped = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )

    assert calls == {"fetch": 1, "normalize": 2}
    assert recovered.companyfacts_fetched == 0
    assert recovered.facts_created == 5
    assert skipped.facts_created == 0
    assert FundamentalFact.objects.count() == 5
    assert FundamentalFactEvidence.objects.count() == 5


def test_acceptance_enrichment_respects_its_later_source_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    listing = _listing()
    store = AssetStore(root=tmp_path)
    _mapping_asset(store)
    initial_submissions = json.loads(SUBMISSIONS_BYTES)
    initial_submissions["filings"]["recent"]["acceptanceDateTime"][0] = ""
    submissions = {
        "content": json.dumps(initial_submissions, sort_keys=True).encode(),
        "retrieved_at": datetime(2026, 8, 15, 12, tzinfo=UTC),
    }
    companyfacts = {"content": _companyfacts_bytes()}

    monkeypatch.setattr(
        sec,
        "fetch_submissions",
        lambda cik: FundamentalSourcePayload(
            provider="sec",
            subject=cik,
            content=submissions["content"],
            content_type="application/json",
            retrieved_at=submissions["retrieved_at"],
            source_url=f"https://data.sec.gov/submissions/CIK{cik}.json",
        ),
    )
    monkeypatch.setattr(
        sec,
        "fetch_submissions_history",
        lambda filename: _payload(
            filename,
            HISTORY_BYTES,
            f"https://data.sec.gov/submissions/{filename}",
        ),
    )
    monkeypatch.setattr(
        sec,
        "fetch_companyfacts",
        lambda cik: _payload(
            cik,
            companyfacts["content"],
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        ),
    )
    budget = SecRequestBudget(
        requests_per_second=5,
        enforce_spacing=False,
        require_enabled=False,
    )

    run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )
    enriched_submissions = json.loads(submissions["content"])
    enriched_submissions["filings"]["recent"]["acceptanceDateTime"][0] = "2026-04-15T20:15:00Z"
    submissions["content"] = json.dumps(enriched_submissions, sort_keys=True).encode()
    submissions["retrieved_at"] = datetime(2026, 8, 17, 12, tzinfo=UTC)
    run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 16),
        store=store,
        budget=budget,
    )

    revisions = FundamentalFact.objects.filter(
        company=listing.security.company,
        concept="revenue",
        period_end=date(2026, 3, 31),
    ).order_by("source_revision")
    assert list(revisions.values_list("source_revision", flat=True)) == [1, 2]
    before_enrichment = list(
        AsOfData(
            datetime(2026, 8, 16, 12, tzinfo=UTC),
            store=store,
        ).fundamental_facts(
            company_id=listing.security.company_id,
            concepts=["revenue"],
        )
    )
    after_enrichment = list(
        AsOfData(
            datetime(2026, 8, 18, 12, tzinfo=UTC),
            store=store,
        ).fundamental_facts(
            company_id=listing.security.company_id,
            concepts=["revenue"],
        )
    )

    before_q1 = [fact for fact in before_enrichment if fact.period_end == date(2026, 3, 31)]
    after_q1 = [fact for fact in after_enrichment if fact.period_end == date(2026, 3, 31)]
    assert [fact.source_revision for fact in before_q1] == [1]
    assert sorted(fact.source_revision for fact in after_q1) == [1, 2]
    assert (
        select_latest_fact_vintages(
            after_q1,
            config=load_sec_fundamentals_config(),
        )[0].source_revision
        == 2
    )
    assert revisions[1].evidence_links.get(
        role=FundamentalFactEvidence.Role.FILING
    ).source_asset.retrieved_at == datetime(2026, 8, 17, 12, tzinfo=UTC)


def test_unchanged_reconciliation_advances_the_verification_clock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _listing()
    store = AssetStore(root=tmp_path)
    _mapping_asset(store)
    companyfacts = {"content": _companyfacts_bytes()}
    fetch_counts = {"companyfacts": 0, "history": 0}
    _patch_provider(monkeypatch, companyfacts)
    original_companyfacts_fetch = sec.fetch_companyfacts
    original_history_fetch = sec.fetch_submissions_history

    def fetch_companyfacts(cik: str) -> FundamentalSourcePayload:
        fetch_counts["companyfacts"] += 1
        return original_companyfacts_fetch(cik)

    def fetch_history(filename: str) -> FundamentalSourcePayload:
        fetch_counts["history"] += 1
        return original_history_fetch(filename)

    monkeypatch.setattr(sec, "fetch_companyfacts", fetch_companyfacts)
    monkeypatch.setattr(sec, "fetch_submissions_history", fetch_history)
    now = {"value": datetime(2026, 8, 15, 12, tzinfo=UTC)}
    monkeypatch.setattr(
        "stanstock.data.sec_ingestion.timezone.now",
        lambda: now["value"],
    )
    budget = SecRequestBudget(
        requests_per_second=5,
        enforce_spacing=False,
        require_enabled=False,
    )

    first = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )
    now["value"] = datetime(2027, 3, 3, 12, tzinfo=UTC)
    reconciliation = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2027, 3, 2),
        store=store,
        budget=budget,
    )
    unchanged_retry = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2027, 3, 2),
        store=store,
        budget=budget,
    )

    assert first.companyfacts_fetched == 1
    assert reconciliation.companyfacts_fetched == 1
    assert reconciliation.raw_assets_created == 0
    assert unchanged_retry.companyfacts_fetched == 0
    assert fetch_counts == {"companyfacts": 2, "history": 2}


def test_factless_amendment_does_not_force_daily_companyfacts_downloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _listing()
    store = AssetStore(root=tmp_path)
    _mapping_asset(store)
    submissions_data = json.loads(SUBMISSIONS_BYTES)
    recent = submissions_data["filings"]["recent"]
    recent["accessionNumber"].append("0000320193-26-000003")
    recent["filingDate"].append("2026-08-01")
    recent["acceptanceDateTime"].append("2026-08-01T20:00:00Z")
    recent["form"].append("10-K/A")
    recent["reportDate"].append("2025-12-31")
    recent["primaryDocument"].append("cover-only-amendment.htm")
    submissions = {"content": json.dumps(submissions_data, sort_keys=True).encode()}
    companyfacts = {"content": _companyfacts_bytes()}
    fetch_count = {"companyfacts": 0}
    _patch_provider(monkeypatch, companyfacts, submissions)
    original_companyfacts_fetch = sec.fetch_companyfacts

    def fetch_companyfacts(cik: str) -> FundamentalSourcePayload:
        fetch_count["companyfacts"] += 1
        return original_companyfacts_fetch(cik)

    monkeypatch.setattr(sec, "fetch_companyfacts", fetch_companyfacts)
    now = {"value": datetime(2026, 8, 15, 12, tzinfo=UTC)}
    monkeypatch.setattr(
        "stanstock.data.sec_ingestion.timezone.now",
        lambda: now["value"],
    )
    budget = SecRequestBudget(
        requests_per_second=5,
        enforce_spacing=False,
        require_enabled=False,
    )

    first = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )
    second = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )
    daily_results = []
    for offset in range(1, 9):
        now["value"] = datetime(2026, 8, 15 + offset, 12, tzinfo=UTC)
        daily_results.append(
            run_sec_ingestion(
                config=load_sec_fundamentals_config(),
                cik_config=_cik_config(),
                universe_config=_universe(),
                target_date=date(2026, 8, 14),
                store=store,
                budget=budget,
            )
        )

    verification = ProviderRecord.objects.get(provider="sec").metadata[
        "companyfacts_verifications"
    ]["0000320193"]
    assert first.companyfacts_fetched == 1
    assert second.companyfacts_fetched == 0
    assert [result.companyfacts_fetched for result in daily_results] == [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
    ]
    assert fetch_count["companyfacts"] == 8
    assert list(verification["missing_accessions"]) == ["0000320193-26-000003"]


def test_missing_filing_is_retried_until_companyfacts_catches_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    listing = _listing()
    store = AssetStore(root=tmp_path)
    _mapping_asset(store)
    missing_q1 = json.loads(_companyfacts_bytes())
    observations = missing_q1["facts"]["us-gaap"][
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    ]["units"]["USD"]
    missing_q1["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"]["units"][
        "USD"
    ] = [
        observation for observation in observations if observation["accn"] != "0000320193-26-000001"
    ]
    companyfacts = {
        "content": json.dumps(missing_q1, sort_keys=True).encode(),
    }
    _patch_provider(monkeypatch, companyfacts)
    original_fetch = sec.fetch_companyfacts
    fetch_count = {"companyfacts": 0}

    def fetch_companyfacts(cik: str) -> FundamentalSourcePayload:
        fetch_count["companyfacts"] += 1
        payload = original_fetch(cik)
        return FundamentalSourcePayload(
            provider=payload.provider,
            subject=payload.subject,
            content=payload.content,
            content_type=payload.content_type,
            retrieved_at=now["value"],
            source_url=payload.source_url,
            metadata=payload.metadata,
        )

    monkeypatch.setattr(sec, "fetch_companyfacts", fetch_companyfacts)
    now = {"value": datetime(2026, 8, 15, 12, tzinfo=UTC)}
    monkeypatch.setattr(
        "stanstock.data.sec_ingestion.timezone.now",
        lambda: now["value"],
    )
    budget = SecRequestBudget(
        requests_per_second=5,
        enforce_spacing=False,
        require_enabled=False,
    )

    first = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )
    companyfacts["content"] = _companyfacts_bytes()
    now["value"] = datetime(2026, 8, 16, 12, tzinfo=UTC)
    caught_up = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 15),
        store=store,
        budget=budget,
    )

    assert first.companyfacts_fetched == 1
    assert caught_up.companyfacts_fetched == 1
    assert caught_up.facts_created == 1
    assert fetch_count["companyfacts"] == 2
    assert FundamentalFact.objects.filter(
        company=listing.security.company,
        accession="0000320193-26-000001",
        concept="revenue",
    ).exists()
    verification = ProviderRecord.objects.get(provider="sec").metadata[
        "companyfacts_verifications"
    ]["0000320193"]
    assert "0000320193-26-000001" not in verification["missing_accessions"]


def test_sec_ingestion_rejects_etf_at_stock_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _listing(security_type=Security.SecurityType.ETF)
    store = AssetStore(root=tmp_path)
    _mapping_asset(store)
    _patch_provider(monkeypatch, {"content": _companyfacts_bytes()})

    with pytest.raises(ValueError, match="exactly one active stock listing"):
        run_sec_ingestion(
            config=load_sec_fundamentals_config(),
            cik_config=_cik_config(),
            universe_config=_universe(),
            target_date=date(2026, 8, 14),
            store=store,
            budget=SecRequestBudget(
                requests_per_second=5,
                enforce_spacing=False,
                require_enabled=False,
            ),
        )


def test_sec_ingestion_accepts_new_snapshot_when_reviewed_identities_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    listing = _listing()
    store = AssetStore(root=tmp_path)
    _mapping_asset(store)
    config = _cik_config()
    mismatched = SecCikConfig(
        config_version=config.config_version,
        universe_config_version=config.universe_config_version,
        source_sha256="f" * 64,
        mappings=config.mappings,
        excluded=config.excluded,
        raw=config.raw,
        config_hash=config.config_hash,
    )
    monkeypatch.setattr(
        sec,
        "fetch_ticker_exchange_mapping",
        lambda: _payload(
            "company_tickers_exchange",
            MAPPING_BYTES,
            "https://www.sec.gov/files/company_tickers_exchange.json",
        ),
    )

    _patch_provider(monkeypatch, {"content": _companyfacts_bytes()})

    result = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=mismatched,
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=SecRequestBudget(
            requests_per_second=5,
            enforce_spacing=False,
            require_enabled=False,
        ),
    )

    assert len(result.companies) == 1
    listing.security.company.refresh_from_db()
    assert listing.security.company.cik == "0000320193"


def test_sec_ingestion_rejects_mapping_identity_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _listing()
    store = AssetStore(root=tmp_path)
    changed_mapping = json.dumps(
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [[999999, "Different Issuer", "AAPL", "Nasdaq"]],
        },
        sort_keys=True,
    ).encode()
    changed_digest = hashlib.sha256(changed_mapping).hexdigest()
    monkeypatch.setattr(
        sec,
        "fetch_ticker_exchange_mapping",
        lambda: _payload(
            "company_tickers_exchange",
            changed_mapping,
            "https://www.sec.gov/files/company_tickers_exchange.json",
        ),
    )
    config = _cik_config()
    no_matching_asset = SecCikConfig(
        config_version=config.config_version,
        universe_config_version=config.universe_config_version,
        source_sha256=changed_digest,
        mappings=config.mappings,
        excluded=config.excluded,
        raw=config.raw,
        config_hash=config.config_hash,
    )

    with pytest.raises(ValueError, match="no longer appears"):
        run_sec_ingestion(
            config=load_sec_fundamentals_config(),
            cik_config=no_matching_asset,
            universe_config=_universe(),
            target_date=date(2026, 8, 14),
            store=store,
            budget=SecRequestBudget(
                requests_per_second=5,
                enforce_spacing=False,
                require_enabled=False,
            ),
        )


# ---------------------------------------------------------------------------
# RI-1: a same-accession correction may not claim its original acceptance time
# ---------------------------------------------------------------------------


def _patch_provider_with_clock(
    monkeypatch: pytest.MonkeyPatch,
    companyfacts: dict[str, object],
    submissions: dict[str, object],
) -> None:
    """Patch the provider so each retrieval carries its *own* timestamp.

    `_patch_provider` pins every payload to one module-level `RETRIEVED_AT`,
    which cannot express "this correction arrived later than the filing that
    it restates". Here each dict carries its own ``retrieved_at`` so the two
    companyfacts assets are independently timestamped, exactly as two real
    downloads on different days would be.
    """

    def _clocked(subject: str, source: dict[str, object], url: str) -> FundamentalSourcePayload:
        content = source["content"]
        retrieved_at = source["retrieved_at"]
        assert isinstance(content, bytes)
        assert isinstance(retrieved_at, datetime)
        return FundamentalSourcePayload(
            provider="sec",
            subject=subject,
            content=content,
            content_type="application/json",
            retrieved_at=retrieved_at,
            source_url=url,
        )

    monkeypatch.setattr(
        sec,
        "fetch_submissions",
        lambda cik: _clocked(
            "0000320193",
            submissions,
            "https://data.sec.gov/submissions/CIK0000320193.json",
        ),
    )
    monkeypatch.setattr(
        sec,
        "fetch_submissions_history",
        lambda filename: _clocked(
            filename,
            submissions,
            f"https://data.sec.gov/submissions/{filename}",
        ),
    )
    monkeypatch.setattr(
        sec,
        "fetch_companyfacts",
        lambda cik: _clocked(
            "0000320193",
            companyfacts,
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        ),
    )


def test_same_accession_correction_cannot_predate_the_retrieval_that_proved_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A restated value is available from its retrieval, not from acceptance.

    The SEC can restate a value under an accession it already filed. The
    original acceptance timestamp still describes the *filing*, but nothing
    before the retrieval that first carried the corrected number proves that
    number existed. Recording the correction at acceptance would let a
    historical cutoff between the two read a value the run could not have
    known -- a look-ahead inside one immutable accession.

    Both revisions stay immutable: the original keeps its acceptance-based
    availability and the correction records the retrieval boundary.
    """
    listing = _listing()
    store = AssetStore(root=tmp_path)
    _mapping_asset(store)

    original_retrieval = datetime(2026, 8, 15, 12, tzinfo=UTC)
    correction_retrieval = datetime(2026, 9, 20, 12, tzinfo=UTC)
    late_unchanged_retrieval = datetime(2026, 10, 5, 12, tzinfo=UTC)

    companyfacts: dict[str, object] = {
        "content": _companyfacts_bytes(q1_revenue=100),
        "retrieved_at": original_retrieval,
    }
    submissions: dict[str, object] = {
        "content": SUBMISSIONS_BYTES,
        "retrieved_at": original_retrieval,
    }
    _patch_provider_with_clock(monkeypatch, companyfacts, submissions)
    budget = SecRequestBudget(
        requests_per_second=5,
        enforce_spacing=False,
        require_enabled=False,
    )

    run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )

    revisions = FundamentalFact.objects.filter(
        company=listing.security.company,
        concept="revenue",
        period_end=date(2026, 3, 31),
    ).order_by("source_revision")
    original = revisions.get()
    acceptance = original.acceptance_at
    assert acceptance == datetime(2026, 4, 15, 20, 15, tzinfo=UTC)
    assert original.available_at == acceptance
    assert original.availability_basis == "acceptance_datetime"
    assert sec_ingestion.CORRECTION_QUALITY_FLAG not in original.quality_flags

    # The same accession is restated, and it is retrieved on a later day.
    # The submissions snapshot changes too, so the companyfacts download is
    # actually re-attempted rather than skipped as unchanged.
    companyfacts["content"] = _companyfacts_bytes(q1_revenue=101)
    companyfacts["retrieved_at"] = correction_retrieval
    submissions["content"] = SUBMISSIONS_BYTES.replace(
        b"Electronic Computers",
        b"Electronic Computers Updated",
    )
    submissions["retrieved_at"] = correction_retrieval
    run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 9, 19),
        store=store,
        budget=budget,
    )

    assert list(revisions.values_list("source_revision", flat=True)) == [1, 2]
    assert list(revisions.values_list("value", flat=True)) == [100, 101]
    correction = revisions.get(source_revision=2)

    # The filing's own timestamps are preserved, unmodified, beside the
    # conservative availability boundary.
    assert correction.accession == original.accession
    assert correction.acceptance_at == acceptance
    assert correction.filed_at == acceptance
    assert correction.available_at == correction_retrieval
    assert correction.available_at > acceptance
    assert correction.availability_basis == sec_ingestion.CORRECTION_AVAILABILITY_BASIS
    assert sec_ingestion.CORRECTION_QUALITY_FLAG in correction.quality_flags
    assert correction.source_asset.retrieved_at == correction_retrieval

    # The already-persisted original row was not rewritten.
    original.refresh_from_db()
    assert original.value == 100
    assert original.available_at == acceptance
    assert original.availability_basis == "acceptance_datetime"

    decision_time = datetime(2026, 11, 1, 12, tzinfo=UTC)

    def _selected(available_through: datetime) -> FundamentalFact:
        visible = list(
            AsOfData(decision_time, store=store)
            .fundamental_facts(
                company_id=listing.security.company_id,
                concepts=["revenue"],
                available_through=available_through,
            )
            .filter(period_end=date(2026, 3, 31))
        )
        return select_latest_fact_vintages(visible, config=load_sec_fundamentals_config())[0]

    # A cutoff between acceptance and the correction's retrieval selects the
    # original: at that moment the correction was not knowable.
    between = datetime(2026, 9, 1, 12, tzinfo=UTC)
    assert acceptance < between < correction_retrieval
    assert _selected(between).value == 100

    # After the correction boundary, the correction wins.
    assert _selected(datetime(2026, 9, 21, 12, tzinfo=UTC)).value == 101

    # Control: an unchanged fact retrieved late is still legitimate evidence.
    # It creates no revision at all, and neither its availability nor the
    # availability of any earlier fact is pushed forward by the late retrieval.
    annual = FundamentalFact.objects.get(
        company=listing.security.company,
        concept="revenue",
        period_end=date(2025, 12, 31),
    )
    annual_available_at = annual.available_at
    assert annual_available_at < correction_retrieval

    companyfacts["retrieved_at"] = late_unchanged_retrieval
    submissions["retrieved_at"] = late_unchanged_retrieval
    unchanged = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 10, 4),
        store=store,
        budget=budget,
    )

    assert unchanged.facts_created == 0
    assert list(revisions.values_list("source_revision", flat=True)) == [1, 2]
    annual.refresh_from_db()
    assert annual.available_at == annual_available_at
    assert annual.source_revision == 1
    assert sec_ingestion.CORRECTION_QUALITY_FLAG not in annual.quality_flags
    # The unchanged annual fact remains readable at a cutoff long before the
    # late retrieval: a later download is not a correction.
    assert (
        AsOfData(decision_time, store=store)
        .fundamental_facts(
            company_id=listing.security.company_id,
            concepts=["revenue"],
            available_through=between,
        )
        .filter(period_end=date(2025, 12, 31))
        .exists()
    )


def test_content_reversion_binds_to_its_own_observation_not_the_reused_asset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A 100 -> 101 -> 100 reversion must not inherit the August retrieval.

    Raw assets are content-addressed, so the October response -- byte-for-byte
    identical to August's -- deduplicates onto the *August* `DataAsset`. Using
    that row's `retrieved_at` would date the third revision to August and let
    a September cutoff read a value that was not observed until October.

    Every retrieval therefore appends its own immutable
    `SourceObservationEvent`, and the correction binds to that event instead.
    """
    listing = _listing()
    store = AssetStore(root=tmp_path)
    _mapping_asset(store)

    august = datetime(2026, 8, 15, 12, tzinfo=UTC)
    september = datetime(2026, 9, 20, 12, tzinfo=UTC)
    october = datetime(2026, 10, 18, 12, tzinfo=UTC)

    original_bytes = _companyfacts_bytes(q1_revenue=100)
    companyfacts: dict[str, object] = {"content": original_bytes, "retrieved_at": august}
    submissions: dict[str, object] = {"content": SUBMISSIONS_BYTES, "retrieved_at": august}
    _patch_provider_with_clock(monkeypatch, companyfacts, submissions)
    budget = SecRequestBudget(
        requests_per_second=5,
        enforce_spacing=False,
        require_enabled=False,
    )

    def _ingest(target_date: date) -> None:
        run_sec_ingestion(
            config=load_sec_fundamentals_config(),
            cik_config=_cik_config(),
            universe_config=_universe(),
            target_date=target_date,
            store=store,
            budget=budget,
        )

    _ingest(date(2026, 8, 14))

    companyfacts["content"] = _companyfacts_bytes(q1_revenue=101)
    companyfacts["retrieved_at"] = september
    submissions["content"] = SUBMISSIONS_BYTES.replace(
        b"Electronic Computers",
        b"Electronic Computers Updated",
    )
    submissions["retrieved_at"] = september
    _ingest(date(2026, 9, 19))

    # Exact raw-content reuse: October serves the original bytes again.
    companyfacts["content"] = original_bytes
    companyfacts["retrieved_at"] = october
    submissions["content"] = SUBMISSIONS_BYTES.replace(
        b"Electronic Computers",
        b"Electronic Computers Revised",
    )
    submissions["retrieved_at"] = october
    _ingest(date(2026, 10, 17))

    revisions = list(
        FundamentalFact.objects.filter(
            company=listing.security.company,
            concept="revenue",
            period_end=date(2026, 3, 31),
        ).order_by("source_revision")
    )
    assert [fact.source_revision for fact in revisions] == [1, 2, 3]
    assert [fact.value for fact in revisions] == [100, 101, 100]
    first, corrected, reverted = revisions

    # The reversion really did reuse the first revision's raw asset...
    assert reverted.source_asset_id == first.source_asset_id
    assert reverted.source_asset.retrieved_at == august
    # ...and its observation hash repeats the first revision's content.
    assert reverted.observation_hash == first.observation_hash
    # ...yet its availability is bound to the October observation.
    assert reverted.available_at == october
    assert reverted.availability_basis == sec_ingestion.CORRECTION_AVAILABILITY_BASIS
    assert corrected.available_at == september

    # The observation events are append-only and cover all three retrievals.
    events = SourceObservationEvent.objects.filter(kind="sec_companyfacts").order_by("observed_at")
    assert [event.observed_at for event in events] == [august, september, october]
    assert events.get(observed_at=october).source_asset_id == first.source_asset_id
    with pytest.raises(ValidationError):
        events.get(observed_at=october).delete()

    # Retry safety: the identical October retrieval replayed verbatim is one
    # event, not a duplicate, and appends no further revision.
    _ingest(date(2026, 10, 17))
    assert (
        SourceObservationEvent.objects.filter(
            kind="sec_companyfacts",
            observed_at=october,
        ).count()
        == 1
    )
    assert (
        FundamentalFact.objects.filter(
            company=listing.security.company,
            concept="revenue",
            period_end=date(2026, 3, 31),
        ).count()
        == 3
    )

    decision_time = datetime(2026, 12, 1, 12, tzinfo=UTC)

    def _selected(available_through: datetime) -> FundamentalFact:
        visible = list(
            AsOfData(decision_time, store=store)
            .fundamental_facts(
                company_id=listing.security.company_id,
                concepts=["revenue"],
                available_through=available_through,
            )
            .filter(period_end=date(2026, 3, 31))
        )
        return select_latest_fact_vintages(visible, config=load_sec_fundamentals_config())[0]

    # Between the September and October observations the correction stands.
    assert _selected(datetime(2026, 10, 1, 12, tzinfo=UTC)).value == 101
    # Only after the October observation does the reversion apply.
    assert _selected(datetime(2026, 10, 19, 12, tzinfo=UTC)).value == 100
    # And before the correction, the original.
    assert _selected(datetime(2026, 9, 1, 12, tzinfo=UTC)).value == 100

    # No persisted row was rewritten by any later run.
    for fact, expected in ((first, 100), (corrected, 101), (reverted, 100)):
        fact.refresh_from_db()
        assert fact.value == expected
    assert first.available_at == first.acceptance_at


def _controlled_clock(monkeypatch: pytest.MonkeyPatch, start: datetime) -> dict[str, datetime]:
    """Drive `sec_ingestion`'s wall clock explicitly.

    Reconciliation staleness is measured against the real clock, so an
    unpatched run would decide a refetch is due purely because the synthetic
    retrieval timestamps sit far from "now". Pinning the clock keeps
    "reconciliation is not due" an actual property of the test.
    """
    clock = {"now": start}
    monkeypatch.setattr(sec_ingestion.timezone, "now", lambda: clock["now"])
    return clock


def _reversion_ingested(
    monkeypatch: pytest.MonkeyPatch,
    store: AssetStore,
    *,
    august: datetime,
    september: datetime,
    october: datetime,
) -> tuple[Listing, dict[str, object], dict[str, object], SecRequestBudget, dict[str, datetime]]:
    """Ingest the Aug100 -> Sep101 -> Oct100 reversion and return the handles."""
    listing = _listing()
    _mapping_asset(store)
    original_bytes = _companyfacts_bytes(q1_revenue=100)
    companyfacts: dict[str, object] = {"content": original_bytes, "retrieved_at": august}
    submissions: dict[str, object] = {"content": SUBMISSIONS_BYTES, "retrieved_at": august}
    _patch_provider_with_clock(monkeypatch, companyfacts, submissions)
    clock = _controlled_clock(monkeypatch, august)
    budget = SecRequestBudget(
        requests_per_second=5,
        enforce_spacing=False,
        require_enabled=False,
    )

    def _ingest(target_date: date) -> None:
        run_sec_ingestion(
            config=load_sec_fundamentals_config(),
            cik_config=_cik_config(),
            universe_config=_universe(),
            target_date=target_date,
            store=store,
            budget=budget,
        )

    _ingest(date(2026, 8, 14))
    companyfacts["content"] = _companyfacts_bytes(q1_revenue=101)
    companyfacts["retrieved_at"] = september
    submissions["content"] = SUBMISSIONS_BYTES.replace(
        b"Electronic Computers",
        b"Electronic Computers Updated",
    )
    submissions["retrieved_at"] = september
    clock["now"] = september
    _ingest(date(2026, 9, 19))
    companyfacts["content"] = original_bytes
    companyfacts["retrieved_at"] = october
    submissions["content"] = SUBMISSIONS_BYTES.replace(
        b"Electronic Computers",
        b"Electronic Computers Revised",
    )
    submissions["retrieved_at"] = october
    clock["now"] = october
    _ingest(date(2026, 10, 17))
    return listing, companyfacts, submissions, budget, clock


def _q1_revisions(listing: Listing) -> list[FundamentalFact]:
    return list(
        FundamentalFact.objects.filter(
            company=listing.security.company,
            concept="revenue",
            period_end=date(2026, 3, 31),
        ).order_by("source_revision")
    )


def test_retry_after_a_reversion_replays_no_stale_companyfacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A no-op retry must not resurrect the superseded September payload.

    After Aug100 -> Sep101 -> Oct100 the October observation carries the
    *August* asset, because those bytes were already on file. The September
    asset still has the newest `DataAsset.retrieved_at`, so recovery driven
    by asset retrieval would replay stale 101 content and append it as a
    fourth revision -- a correction the provider never sent.

    With unchanged submissions, a controlled clock, and reconciliation not
    due, the retry must issue no Companyfacts request at all, append nothing,
    and leave October's 100 selected.
    """
    store = AssetStore(root=tmp_path)
    august = datetime(2026, 8, 15, 12, tzinfo=UTC)
    september = datetime(2026, 9, 20, 12, tzinfo=UTC)
    october = datetime(2026, 10, 18, 12, tzinfo=UTC)
    listing, _companyfacts, _submissions, budget, clock = _reversion_ingested(
        monkeypatch,
        store,
        august=august,
        september=september,
        october=october,
    )

    before = _q1_revisions(listing)
    assert [fact.source_revision for fact in before] == [1, 2, 3]
    assert [fact.value for fact in before] == [100, 101, 100]
    # The trap: the newest asset by retrieval is the superseded 101 payload.
    newest_by_retrieval = (
        DataAsset.objects.filter(provider="sec", kind="sec_companyfacts")
        .order_by("-retrieved_at")
        .first()
    )
    assert newest_by_retrieval is not None
    assert newest_by_retrieval.retrieved_at == september
    assert before[2].source_asset_id != newest_by_retrieval.pk

    # One day later: reconciliation is not due, so no fetch is warranted.
    clock["now"] = october + timedelta(days=1)

    def _refuse(cik: str) -> FundamentalSourcePayload:
        raise AssertionError("a no-op retry must not request Companyfacts")

    monkeypatch.setattr(sec, "fetch_companyfacts", _refuse)

    retry = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 10, 18),
        store=store,
        budget=budget,
    )

    assert retry.companyfacts_fetched == 0
    assert retry.facts_created == 0
    after = _q1_revisions(listing)
    assert [fact.source_revision for fact in after] == [1, 2, 3]
    assert [fact.value for fact in after] == [100, 101, 100]
    # Every persisted row is byte-identical to before the retry.
    for previous, current in zip(before, after, strict=True):
        assert previous.pk == current.pk
        assert previous.value == current.value
        assert previous.available_at == current.available_at
        assert previous.availability_basis == current.availability_basis
        assert previous.source_asset_id == current.source_asset_id

    selected = select_latest_fact_vintages(
        list(
            AsOfData(datetime(2026, 12, 1, 12, tzinfo=UTC), store=store)
            .fundamental_facts(
                company_id=listing.security.company_id,
                concepts=["revenue"],
            )
            .filter(period_end=date(2026, 3, 31))
        ),
        config=load_sec_fundamentals_config(),
    )[0]
    assert selected.value == 100
    assert selected.source_revision == 3


def test_interrupted_normalization_after_a_reversion_recovers_the_exact_observation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Recovery replays the October observation, not the newest asset row.

    Normalization is interrupted immediately after the deduplicated October
    reversion, so the retry must re-normalize the *August asset under the
    October observation*. Recovering the September asset instead would both
    substitute stale content and date it to September.
    """
    store = AssetStore(root=tmp_path)
    august = datetime(2026, 8, 15, 12, tzinfo=UTC)
    september = datetime(2026, 9, 20, 12, tzinfo=UTC)
    october = datetime(2026, 10, 18, 12, tzinfo=UTC)
    listing = _listing()
    _mapping_asset(store)
    original_bytes = _companyfacts_bytes(q1_revenue=100)
    companyfacts: dict[str, object] = {"content": original_bytes, "retrieved_at": august}
    submissions: dict[str, object] = {"content": SUBMISSIONS_BYTES, "retrieved_at": august}
    _patch_provider_with_clock(monkeypatch, companyfacts, submissions)
    clock = _controlled_clock(monkeypatch, august)
    budget = SecRequestBudget(
        requests_per_second=5,
        enforce_spacing=False,
        require_enabled=False,
    )

    def _ingest(target_date: date) -> None:
        run_sec_ingestion(
            config=load_sec_fundamentals_config(),
            cik_config=_cik_config(),
            universe_config=_universe(),
            target_date=target_date,
            store=store,
            budget=budget,
        )

    _ingest(date(2026, 8, 14))
    companyfacts["content"] = _companyfacts_bytes(q1_revenue=101)
    companyfacts["retrieved_at"] = september
    submissions["content"] = SUBMISSIONS_BYTES.replace(
        b"Electronic Computers",
        b"Electronic Computers Updated",
    )
    submissions["retrieved_at"] = september
    clock["now"] = september
    _ingest(date(2026, 9, 19))

    # October: the reversion is fetched and stored, then normalization dies.
    original_normalize = sec_ingestion._normalize_companyfacts
    calls = {"normalize": 0}

    def normalize_companyfacts(**kwargs: object) -> tuple[int, int]:
        calls["normalize"] += 1
        if calls["normalize"] == 1:
            raise RuntimeError("simulated normalization interruption")
        return original_normalize(**kwargs)

    companyfacts["content"] = original_bytes
    companyfacts["retrieved_at"] = october
    submissions["content"] = SUBMISSIONS_BYTES.replace(
        b"Electronic Computers",
        b"Electronic Computers Revised",
    )
    submissions["retrieved_at"] = october
    clock["now"] = october
    monkeypatch.setattr(sec_ingestion, "_normalize_companyfacts", normalize_companyfacts)

    with pytest.raises(RuntimeError, match="normalization interruption"):
        _ingest(date(2026, 10, 17))

    # The October observation exists even though its asset is August's.
    october_event = SourceObservationEvent.objects.get(
        kind="sec_companyfacts",
        observed_at=october,
    )
    august_asset = DataAsset.objects.get(
        provider="sec",
        kind="sec_companyfacts",
        retrieved_at=august,
    )
    assert october_event.source_asset_id == august_asset.pk

    recovered = sec_ingestion._latest_observed_companyfacts("0000320193")
    assert recovered is not None
    assert recovered.asset.pk == august_asset.pk
    assert recovered.observed_at == october
    assert recovered.basis == sec_ingestion.COMPANYFACTS_RECOVERY_OBSERVED

    def _refuse(cik: str) -> FundamentalSourcePayload:
        raise AssertionError("recovery must replay persisted evidence, not refetch")

    monkeypatch.setattr(sec, "fetch_companyfacts", _refuse)
    clock["now"] = october + timedelta(days=1)
    _ingest(date(2026, 10, 18))

    revisions = _q1_revisions(listing)
    assert [fact.source_revision for fact in revisions] == [1, 2, 3]
    assert [fact.value for fact in revisions] == [100, 101, 100]
    reverted = revisions[2]
    # Recovered under the October observation, from the reused August asset.
    assert reverted.available_at == october
    assert reverted.availability_basis == sec_ingestion.CORRECTION_AVAILABILITY_BASIS
    assert reverted.source_asset_id == august_asset.pk


def test_recovery_refuses_to_replay_content_older_than_a_committed_correction(
    tmp_path: Path,
) -> None:
    """A conflicting recovery fails explicitly instead of guessing.

    If the only recoverable observation predates a correction already on
    file -- a truncated or partially restored event history, say -- then
    re-normalizing it would append the superseded value back as a brand-new
    revision. There is no evidence to decide which side is right, so the run
    refuses rather than manufacturing a correction.
    """
    listing = _listing()
    store = AssetStore(root=tmp_path)
    september = datetime(2026, 9, 20, 12, tzinfo=UTC)
    october = datetime(2026, 10, 18, 12, tzinfo=UTC)

    stored = store.write_bytes("raw/sec/stale-companyfacts.json", _companyfacts_bytes())
    asset = register_asset(
        provider="sec",
        kind="sec_companyfacts",
        subject="0000320193",
        stored=stored,
        retrieved_at=september,
        available_at=september,
    )
    # A correction already committed under a *later* observation.
    FundamentalFact.objects.create(
        company=listing.security.company,
        provider="sec",
        concept="revenue",
        source_concept="us-gaap:Revenues",
        value=Decimal("101"),
        unit="USD",
        currency="USD",
        period_type=FundamentalFact.PeriodType.DURATION,
        period_start=date(2026, 1, 1),
        period_end=date(2026, 3, 31),
        accession="0000320193-26-000001",
        filing_form="10-Q",
        acceptance_at=datetime(2026, 4, 15, 20, 15, tzinfo=UTC),
        filed_at=datetime(2026, 4, 15, 20, 15, tzinfo=UTC),
        available_at=october,
        availability_basis=sec_ingestion.CORRECTION_AVAILABILITY_BASIS,
        source_revision=2,
        source_asset=asset,
    )

    recovered = sec_ingestion.RecoveredCompanyfacts(
        asset=asset,
        observed_at=september,
        basis=sec_ingestion.COMPANYFACTS_RECOVERY_OBSERVED,
    )

    with pytest.raises(ProviderResponseError, match="older than the committed correction"):
        sec_ingestion._assert_replay_is_recoverable(
            company=listing.security.company,
            recovered=recovered,
            cik="0000320193",
        )

    # The same recovery at or after the committed boundary is fine.
    sec_ingestion._assert_replay_is_recoverable(
        company=listing.security.company,
        recovered=replace(recovered, observed_at=october),
        cik="0000320193",
    )


def test_ingestion_refuses_two_payloads_at_one_observation_instant(
    tmp_path: Path,
) -> None:
    """One observation timestamp names one content, enforced before writing.

    Recording different bytes at an instant already recorded would leave two
    events that no evidence can order. The refusal happens at write time, so
    nothing downstream ever has to guess -- while re-recording the *same*
    retrieval stays idempotent, which is what makes retries safe.
    """
    store = AssetStore(root=tmp_path)
    observed_at = datetime(2026, 10, 18, 12, tzinfo=UTC)
    digests = []
    assets = []
    for index in range(2):
        payload = json.dumps({"facts": {}, "n": index}, sort_keys=True).encode()
        stored = store.write_bytes(f"raw/sec/instant-{index}.json", payload)
        assets.append(
            register_asset(
                provider="sec",
                kind="sec_companyfacts",
                subject="0000320193",
                stored=stored,
                retrieved_at=observed_at,
                available_at=observed_at,
            )
        )
        digests.append(stored.sha256)

    first = sec_ingestion._record_observation_event(
        asset=assets[0],
        kind="sec_companyfacts",
        subject="0000320193",
        digest=digests[0],
        observed_at=observed_at,
    )
    # The same retrieval again is idempotent.
    assert (
        sec_ingestion._record_observation_event(
            asset=assets[0],
            kind="sec_companyfacts",
            subject="0000320193",
            digest=digests[0],
            observed_at=observed_at,
        ).pk
        == first.pk
    )
    # Different content at that instant is refused outright.
    with pytest.raises(ProviderResponseError, match="two different payloads observed"):
        sec_ingestion._record_observation_event(
            asset=assets[1],
            kind="sec_companyfacts",
            subject="0000320193",
            digest=digests[1],
            observed_at=observed_at,
        )
    assert SourceObservationEvent.objects.filter(kind="sec_companyfacts").count() == 1


def test_full_retry_of_an_a_b_a_reversion_at_one_instant_refuses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A -> B -> A at one observation timestamp fails closed, appending nothing.

    The reversion to A reuses A's original event, so B would be left looking
    like the newest observation purely because its row was committed later.
    Rather than inferring order from a local clock, ingestion refuses the
    conflicting B observation and nothing stale is ever appended.
    """
    listing = _listing()
    store = AssetStore(root=tmp_path)
    _mapping_asset(store)
    observed_at = datetime(2026, 10, 18, 12, tzinfo=UTC)
    content_a = _companyfacts_bytes(q1_revenue=100)
    content_b = _companyfacts_bytes(q1_revenue=101)
    companyfacts: dict[str, object] = {"content": content_a, "retrieved_at": observed_at}
    submissions: dict[str, object] = {"content": SUBMISSIONS_BYTES, "retrieved_at": observed_at}
    _patch_provider_with_clock(monkeypatch, companyfacts, submissions)
    clock = _controlled_clock(monkeypatch, observed_at)
    budget = SecRequestBudget(
        requests_per_second=5,
        enforce_spacing=False,
        require_enabled=False,
    )

    def _ingest(target_date: date) -> None:
        run_sec_ingestion(
            config=load_sec_fundamentals_config(),
            cik_config=_cik_config(),
            universe_config=_universe(),
            target_date=target_date,
            store=store,
            budget=budget,
        )

    _ingest(date(2026, 10, 17))
    baseline = _q1_revisions(listing)
    assert [fact.value for fact in baseline] == [100]

    # B arrives claiming the very same observation instant.
    companyfacts["content"] = content_b
    submissions["content"] = SUBMISSIONS_BYTES.replace(
        b"Electronic Computers",
        b"Electronic Computers Updated",
    )
    submissions["retrieved_at"] = observed_at + timedelta(days=1)
    clock["now"] = observed_at + timedelta(days=1)

    with pytest.raises(ProviderResponseError, match="two different payloads observed"):
        _ingest(date(2026, 10, 18))

    # Nothing stale was appended, and A's evidence is untouched.
    assert [fact.value for fact in _q1_revisions(listing)] == [100]
    assert (
        SourceObservationEvent.objects.filter(
            kind="sec_companyfacts",
            observed_at=observed_at,
        ).count()
        == 1
    )
    assert (
        SourceObservationEvent.objects.get(
            kind="sec_companyfacts",
            observed_at=observed_at,
        ).content_sha256
        == hashlib.sha256(content_a).hexdigest()
    )

    # An ordinary retry of the same observation still succeeds.
    companyfacts["content"] = content_a
    submissions["content"] = SUBMISSIONS_BYTES
    submissions["retrieved_at"] = observed_at
    clock["now"] = observed_at + timedelta(days=2)
    _ingest(date(2026, 10, 19))
    assert [fact.value for fact in _q1_revisions(listing)] == [100]


def test_replay_refuses_a_corrected_chain_with_no_observation_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An upgraded database with no events must not guess from asset order.

    Facts here were written before observation events existed, so the only
    ordering available is `DataAsset.retrieved_at` -- and after the
    Aug100 -> Sep101 -> Oct100 reversion that points at the superseded
    September payload. Replaying it would append 101 back as a brand-new,
    event-bound revision the provider never sent, so the replay refuses
    instead. A fresh fetch remains separately gated and is the way forward.
    """
    store = AssetStore(root=tmp_path)
    august = datetime(2026, 8, 15, 12, tzinfo=UTC)
    september = datetime(2026, 9, 20, 12, tzinfo=UTC)
    october = datetime(2026, 10, 18, 12, tzinfo=UTC)

    # Build a faithful pre-event database: no observation events, and facts
    # shaped as the pre-correction-basis code wrote them, so the
    # stale-content guard cannot mask the missing-evidence guard.
    monkeypatch.setattr(
        sec_ingestion,
        "_record_observation_event",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(sec_ingestion, "CORRECTION_AVAILABILITY_BASIS", "acceptance_datetime")
    listing, _companyfacts, _submissions, budget, _clock = _reversion_ingested(
        monkeypatch,
        store,
        august=august,
        september=september,
        october=october,
    )
    # Restore the real recorder, then re-pin the provider and the clock so
    # the retry is a genuine no-op: reconciliation is not due.
    monkeypatch.undo()
    _patch_provider_with_clock(monkeypatch, _companyfacts, _submissions)
    _controlled_clock(monkeypatch, october + timedelta(days=1))

    assert not SourceObservationEvent.objects.exists()
    before = _q1_revisions(listing)
    assert [fact.source_revision for fact in before] == [1, 2, 3]
    assert [fact.value for fact in before] == [100, 101, 100]
    # Legacy shape: nothing on file carries the correction basis, so only the
    # missing-observation-evidence guard can refuse this replay.
    assert not FundamentalFact.objects.filter(
        availability_basis=sec_ingestion.CORRECTION_AVAILABILITY_BASIS
    ).exists()
    # The trap the refusal exists to close.
    newest_by_retrieval = (
        DataAsset.objects.filter(provider="sec", kind="sec_companyfacts")
        .order_by("-retrieved_at")
        .first()
    )
    assert newest_by_retrieval is not None
    assert newest_by_retrieval.retrieved_at == september
    correction_rows_before = FundamentalFact.objects.filter(
        availability_basis=sec_ingestion.CORRECTION_AVAILABILITY_BASIS
    ).count()

    def _refuse(cik: str) -> FundamentalSourcePayload:
        raise AssertionError("an unproven replay must not silently refetch")

    monkeypatch.setattr(sec, "fetch_companyfacts", _refuse)

    with pytest.raises(ProviderResponseError, match="no observation event records"):
        run_sec_ingestion(
            config=load_sec_fundamentals_config(),
            cik_config=_cik_config(),
            universe_config=_universe(),
            target_date=date(2026, 10, 18),
            store=store,
            budget=budget,
        )

    # No fourth revision, and no fabricated event-bound fact. (The retry's
    # own submissions retrieval is legitimately recorded before the
    # companyfacts replay is reached; no *companyfacts* observation is
    # invented for evidence that was never re-read.)
    after = _q1_revisions(listing)
    assert [fact.source_revision for fact in after] == [1, 2, 3]
    assert [fact.value for fact in after] == [100, 101, 100]
    assert not SourceObservationEvent.objects.filter(kind="sec_companyfacts").exists()
    assert (
        FundamentalFact.objects.filter(
            availability_basis=sec_ingestion.CORRECTION_AVAILABILITY_BASIS
        ).count()
        == correction_rows_before
    )
    for previous, current in zip(before, after, strict=True):
        assert previous.pk == current.pk
        assert previous.value == current.value
        assert previous.available_at == current.available_at
        assert previous.availability_basis == current.availability_basis

    # Historical selection is exactly what it was before the refused retry.
    selected = select_latest_fact_vintages(
        list(
            AsOfData(datetime(2026, 12, 1, 12, tzinfo=UTC), store=store)
            .fundamental_facts(
                company_id=listing.security.company_id,
                concepts=["revenue"],
            )
            .filter(period_end=date(2026, 3, 31))
        ),
        config=load_sec_fundamentals_config(),
    )[0]
    assert selected.value == 100
    assert selected.source_revision == 3


def test_replay_without_events_is_allowed_when_no_correction_chain_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An upgraded database with no corrections has nothing to mis-order.

    The refusal is scoped to what is actually ambiguous. A single-revision
    history recovers from its only asset unambiguously, so an ordinary
    upgraded database keeps working.
    """
    listing = _listing()
    store = AssetStore(root=tmp_path)
    _mapping_asset(store)
    retrieved_at = datetime(2026, 8, 15, 12, tzinfo=UTC)
    companyfacts: dict[str, object] = {
        "content": _companyfacts_bytes(),
        "retrieved_at": retrieved_at,
    }
    submissions: dict[str, object] = {
        "content": SUBMISSIONS_BYTES,
        "retrieved_at": retrieved_at,
    }
    _patch_provider_with_clock(monkeypatch, companyfacts, submissions)
    monkeypatch.setattr(sec_ingestion, "_record_observation_event", lambda **kwargs: None)
    budget = SecRequestBudget(
        requests_per_second=5,
        enforce_spacing=False,
        require_enabled=False,
    )
    run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )
    assert not SourceObservationEvent.objects.exists()
    assert FundamentalFact.objects.filter(source_revision__gt=1).count() == 0

    recovered = sec_ingestion._latest_observed_companyfacts("0000320193")
    assert recovered is not None
    assert recovered.basis == sec_ingestion.COMPANYFACTS_RECOVERY_LEGACY

    # No correction chain, so the replay guard permits recovery.
    sec_ingestion._assert_replay_is_recoverable(
        company=listing.security.company,
        recovered=recovered,
        cik="0000320193",
    )


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL observation-instant concurrency regression",
)
@pytest.mark.django_db(transaction=True)
def test_concurrent_conflicting_observations_at_one_instant_leave_exactly_one(
    tmp_path: Path,
) -> None:
    """Two writers racing for one observation instant: the database decides.

    A sequential read-then-write check passes in *both* transactions, because
    neither can see the other's uncommitted row, and `select_for_update()`
    cannot help when the row does not exist yet. Only the uniqueness
    constraint on ``(provider, kind, subject, observed_at)`` makes the
    instant exclusive.

    The loser must fail *before* any fact is normalized, so nothing derived
    from the conflicting payload can reach the ledger.
    """
    store = AssetStore(root=tmp_path)
    observed_at = datetime(2026, 10, 18, 12, tzinfo=UTC)
    assets = []
    digests = []
    for index in range(2):
        payload = json.dumps({"facts": {}, "n": index}, sort_keys=True).encode()
        stored = store.write_bytes(f"raw/sec/race-{index}.json", payload)
        assets.append(
            register_asset(
                provider="sec",
                kind="sec_companyfacts",
                subject="0000320193",
                stored=stored,
                retrieved_at=observed_at,
                available_at=observed_at,
            )
        )
        digests.append(stored.sha256)

    barrier = Barrier(2)

    def record(index: int) -> str:
        connections.close_all()
        try:
            barrier.wait(timeout=10)
            sec_ingestion._record_observation_event(
                asset=assets[index],
                kind="sec_companyfacts",
                subject="0000320193",
                digest=digests[index],
                observed_at=observed_at,
            )
            return "recorded"
        except ProviderResponseError:
            return "refused"
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(executor.map(record, range(2)))

    assert outcomes == ["recorded", "refused"]
    events = SourceObservationEvent.objects.filter(
        kind="sec_companyfacts",
        observed_at=observed_at,
    )
    assert events.count() == 1
    assert events.get().content_sha256 in digests
    # The refusal happened before normalization, so no fact exists at all.
    assert not FundamentalFact.objects.exists()


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL observation-instant concurrency regression",
)
@pytest.mark.django_db(transaction=True)
def test_concurrent_identical_observations_are_idempotent(tmp_path: Path) -> None:
    """The same retrieval racing with itself succeeds twice over one row.

    Retries and overlapping scheduled runs record identical evidence. That
    must never be an error, and must never duplicate the event.
    """
    store = AssetStore(root=tmp_path)
    observed_at = datetime(2026, 10, 18, 12, tzinfo=UTC)
    stored = store.write_bytes("raw/sec/same.json", json.dumps({"facts": {}}).encode())
    asset = register_asset(
        provider="sec",
        kind="sec_companyfacts",
        subject="0000320193",
        stored=stored,
        retrieved_at=observed_at,
        available_at=observed_at,
    )
    barrier = Barrier(2)

    def record(_index: int) -> str:
        connections.close_all()
        try:
            barrier.wait(timeout=10)
            event = sec_ingestion._record_observation_event(
                asset=asset,
                kind="sec_companyfacts",
                subject="0000320193",
                digest=stored.sha256,
                observed_at=observed_at,
            )
            return str(event.pk)
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as executor:
        event_ids = list(executor.map(record, range(2)))

    assert len(set(event_ids)) == 1
    assert (
        SourceObservationEvent.objects.filter(
            kind="sec_companyfacts",
            observed_at=observed_at,
        ).count()
        == 1
    )


def test_a_due_fresh_fetch_rebinds_an_unprovable_legacy_reversion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fresh proof must produce a bound vintage, not reuse the unprovable row.

    The refused replay is only half the story. Once a Companyfacts retrieval
    is due again and confirms the reverted content, that retrieval *is* the
    missing evidence. Returning the identical legacy revision unchanged would
    throw it away: as-of readers would keep deferring that row and keep
    selecting the superseded value forever.

    So a new revision is appended -- same value, same observation hash, new
    availability bound to the fresh observation -- while every persisted row,
    asset, and event stays byte-identical.
    """
    store = AssetStore(root=tmp_path)
    august = datetime(2026, 8, 15, 12, tzinfo=UTC)
    september = datetime(2026, 9, 20, 12, tzinfo=UTC)
    october = datetime(2026, 10, 18, 12, tzinfo=UTC)
    # Far enough past October for reconciliation to be due again.
    fresh = datetime(2027, 1, 20, 12, tzinfo=UTC)

    monkeypatch.setattr(sec_ingestion, "_record_observation_event", lambda **kwargs: None)
    monkeypatch.setattr(sec_ingestion, "CORRECTION_AVAILABILITY_BASIS", "acceptance_datetime")
    listing, companyfacts, submissions, budget, _clock = _reversion_ingested(
        monkeypatch,
        store,
        august=august,
        september=september,
        october=october,
    )
    monkeypatch.undo()
    _patch_provider_with_clock(monkeypatch, companyfacts, submissions)
    clock = _controlled_clock(monkeypatch, october + timedelta(days=1))

    legacy = _q1_revisions(listing)
    assert [fact.value for fact in legacy] == [100, 101, 100]
    legacy_state = [
        (f.pk, f.value, f.available_at, f.availability_basis, f.source_asset_id) for f in legacy
    ]
    legacy_assets = {
        asset.pk: (asset.sha256, asset.retrieved_at) for asset in DataAsset.objects.all()
    }

    # 1. The cached replay is refused: no evidence proves revision 3.
    def _refuse(cik: str) -> FundamentalSourcePayload:
        raise AssertionError("the refused replay must not fetch")

    monkeypatch.setattr(sec, "fetch_companyfacts", _refuse)
    with pytest.raises(ProviderResponseError, match="no observation event records"):
        run_sec_ingestion(
            config=load_sec_fundamentals_config(),
            cik_config=_cik_config(),
            universe_config=_universe(),
            target_date=date(2026, 10, 18),
            store=store,
            budget=budget,
        )
    assert len(_q1_revisions(listing)) == 3

    # 2. A due fresh fetch returns the very same reverted bytes.
    monkeypatch.undo()
    _patch_provider_with_clock(monkeypatch, companyfacts, submissions)
    clock = _controlled_clock(monkeypatch, fresh)
    companyfacts["retrieved_at"] = fresh
    submissions["retrieved_at"] = fresh
    result = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2027, 1, 19),
        store=store,
        budget=budget,
    )

    assert result.companyfacts_fetched == 1
    rebound_revisions = _q1_revisions(listing)
    assert [fact.source_revision for fact in rebound_revisions] == [1, 2, 3, 4]
    assert [fact.value for fact in rebound_revisions] == [100, 101, 100, 100]
    rebound = rebound_revisions[3]

    # New availability and provenance, explicitly.
    assert rebound.available_at == fresh
    assert rebound.availability_basis == sec_ingestion.CORRECTION_AVAILABILITY_BASIS
    assert sec_ingestion.REBOUND_QUALITY_FLAG in rebound.quality_flags
    # Same economic observation as the row it re-binds.
    assert rebound.observation_hash == rebound_revisions[2].observation_hash
    assert rebound.acceptance_at == rebound_revisions[2].acceptance_at
    # The fresh observation is recorded.
    assert SourceObservationEvent.objects.filter(
        kind="sec_companyfacts",
        observed_at=fresh,
    ).exists()

    # 3. Nothing already persisted moved.
    for fact, expected in zip(_q1_revisions(listing)[:3], legacy_state, strict=True):
        assert (
            fact.pk,
            fact.value,
            fact.available_at,
            fact.availability_basis,
            fact.source_asset_id,
        ) == expected
    for asset_id, expected_asset in legacy_assets.items():
        asset = DataAsset.objects.get(pk=asset_id)
        assert (asset.sha256, asset.retrieved_at) == expected_asset

    # 4. Another identical fresh retrieval adds nothing.
    clock["now"] = fresh + timedelta(days=1)
    run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2027, 1, 20),
        store=store,
        budget=budget,
    )
    assert [fact.source_revision for fact in _q1_revisions(listing)] == [1, 2, 3, 4]

    # 5. And a replay is now proven, so it also appends nothing.
    monkeypatch.setattr(sec, "fetch_companyfacts", _refuse)
    clock["now"] = fresh + timedelta(days=2)
    run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2027, 1, 21),
        store=store,
        budget=budget,
    )
    assert [fact.source_revision for fact in _q1_revisions(listing)] == [1, 2, 3, 4]

    # 6. Selection: the fresh boundary restores the reverted value, and the
    #    earlier cutoff is untouched.
    def _selected(available_through: datetime) -> FundamentalFact:
        return select_latest_fact_vintages(
            list(
                AsOfData(datetime(2027, 3, 1, 12, tzinfo=UTC), store=store)
                .fundamental_facts(
                    company_id=listing.security.company_id,
                    concepts=["revenue"],
                    available_through=available_through,
                )
                .filter(period_end=date(2026, 3, 31))
            ),
            config=load_sec_fundamentals_config(),
        )[0]

    assert _selected(fresh + timedelta(days=1)).source_revision == 4
    assert _selected(october).source_revision == 3


# ---------------------------------------------------------------------------
# The correction-availability ingestion fix is ACTIVE under the default
# `us-sec-long-v2` configuration. long-v3 is an inactive reader on top of it;
# these assertions deliberately use the default-loaded configuration only.
# ---------------------------------------------------------------------------


def test_default_v2_selection_follows_observations_through_a_reversion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The shipped default already benefits, with no long-v3 involvement.

    Ingestion binds each correction to the retrieval that carried it, so an
    A -> B -> A reversion with content reuse is selected correctly at every
    cutoff by the *default* configuration's ordinary vintage selection. The
    prior semantics -- availability backdated to filing acceptance -- would
    have made all three revisions visible from April and always selected the
    newest, so the "before B" and "between" cutoffs are exactly where the
    fix is observable.
    """
    store = AssetStore(root=tmp_path)
    august = datetime(2026, 8, 15, 12, tzinfo=UTC)
    september = datetime(2026, 9, 20, 12, tzinfo=UTC)
    october = datetime(2026, 10, 18, 12, tzinfo=UTC)
    listing, _companyfacts, _submissions, _budget, _clock = _reversion_ingested(
        monkeypatch,
        store,
        august=august,
        september=september,
        october=october,
    )

    default_config = load_long_forecast_config()
    assert default_config.version == "us-sec-long-v2"
    assert default_config.proven_observation_correction_availability is None

    revisions = _q1_revisions(listing)
    assert [fact.source_revision for fact in revisions] == [1, 2, 3]
    original, corrected, reverted = revisions
    acceptance = original.acceptance_at
    assert acceptance is not None

    def _selected(available_through: datetime) -> FundamentalFact:
        return select_latest_fact_vintages(
            list(
                AsOfData(datetime(2026, 12, 1, 12, tzinfo=UTC), store=store)
                .fundamental_facts(
                    company_id=listing.security.company_id,
                    concepts=["revenue"],
                    available_through=available_through,
                )
                .filter(period_end=date(2026, 3, 31))
            ),
            config=load_sec_fundamentals_config(),
        )[0]

    # Before B: only the original is knowable.
    before_b = _selected(september - timedelta(days=1))
    assert (before_b.pk, before_b.source_revision, before_b.value) == (
        original.pk,
        1,
        Decimal("100.00000000"),
    )
    # Between B and the reversion: the correction stands.
    between = _selected(october - timedelta(days=1))
    assert (between.pk, between.source_revision, between.value) == (
        corrected.pk,
        2,
        Decimal("101.00000000"),
    )
    # After the reversion: the reverted value, from the *reused* August asset.
    after = _selected(october + timedelta(days=1))
    assert (after.pk, after.source_revision, after.value) == (
        reverted.pk,
        3,
        Decimal("100.00000000"),
    )
    assert after.source_asset_id == original.source_asset_id

    # Faithful comparison to the prior semantics: had every revision kept the
    # filing's acceptance as its availability, all three would have been
    # visible from April and the newest would always have won -- so both
    # earlier cutoffs would have returned 100 (revision 3) instead.
    prior_semantics = select_latest_fact_vintages(
        list(revisions),
        config=load_sec_fundamentals_config(),
    )[0]
    assert prior_semantics.pk == reverted.pk
    assert acceptance < september
    assert before_b.pk != prior_semantics.pk
    assert between.pk != prior_semantics.pk

    # The default configuration carries no long-v3 payload fields.
    assert default_config.newest_quarter_anchored_homogeneous_ttm_alias_selection is None
    assert default_config.joint_compatible_invested_capital_pair_selection is None
    assert default_config.maximum_same_date_source_combinations is None


def test_unchanged_late_retrieval_under_default_v2_moves_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ordinary late retrieval stays distinct from a correction.

    Honest ingestion is not gated by any of this: re-reading unchanged
    content appends no fact and no revision, and moves no availability.
    """
    listing = _listing()
    store = AssetStore(root=tmp_path)
    _mapping_asset(store)
    first = datetime(2026, 8, 15, 12, tzinfo=UTC)
    later = datetime(2026, 11, 30, 12, tzinfo=UTC)
    companyfacts: dict[str, object] = {
        "content": _companyfacts_bytes(),
        "retrieved_at": first,
    }
    submissions: dict[str, object] = {"content": SUBMISSIONS_BYTES, "retrieved_at": first}
    _patch_provider_with_clock(monkeypatch, companyfacts, submissions)
    clock = _controlled_clock(monkeypatch, first)
    budget = SecRequestBudget(
        requests_per_second=5,
        enforce_spacing=False,
        require_enabled=False,
    )

    run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 8, 14),
        store=store,
        budget=budget,
    )
    before = {
        fact.pk: (fact.value, fact.available_at, fact.availability_basis, fact.source_revision)
        for fact in FundamentalFact.objects.filter(company=listing.security.company)
    }
    assert before

    # A genuinely later retrieval of byte-identical content.
    companyfacts["retrieved_at"] = later
    submissions["retrieved_at"] = later
    clock["now"] = later
    result = run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=_cik_config(),
        universe_config=_universe(),
        target_date=date(2026, 11, 29),
        store=store,
        budget=budget,
    )

    assert result.facts_created == 0
    after = {
        fact.pk: (fact.value, fact.available_at, fact.availability_basis, fact.source_revision)
        for fact in FundamentalFact.objects.filter(company=listing.security.company)
    }
    assert after == before
    assert not FundamentalFact.objects.filter(source_revision__gt=1).exists()
    assert not FundamentalFact.objects.filter(
        availability_basis=sec_ingestion.CORRECTION_AVAILABILITY_BASIS
    ).exists()
    # The later retrieval is still recorded as an observation of that content.
    assert SourceObservationEvent.objects.filter(
        kind="sec_companyfacts",
        observed_at=later,
    ).exists()


def _observation_asset(store: AssetStore, *, name: str, payload: bytes, at: datetime) -> DataAsset:
    stored = store.write_bytes(f"raw/sec/{name}.json", payload)
    return register_asset(
        provider="sec",
        kind="sec_companyfacts",
        subject="0000320193",
        stored=stored,
        retrieved_at=at,
        available_at=at,
    )


def test_an_observation_may_not_certify_bytes_its_asset_does_not_hold(
    tmp_path: Path,
) -> None:
    """The digest is the claim; it must match the asset making the claim."""
    store = AssetStore(root=tmp_path)
    at = datetime(2026, 10, 18, 12, tzinfo=UTC)
    asset = _observation_asset(store, name="real", payload=b'{"facts": {}}', at=at)

    with pytest.raises(sec_ingestion.ObservationEvidenceError, match="only certify the bytes"):
        sec_ingestion._record_observation_event(
            asset=asset,
            kind="sec_companyfacts",
            subject="0000320193",
            digest="f" * 64,
            observed_at=at,
        )
    assert not SourceObservationEvent.objects.exists()


def test_a_corrupt_event_fails_on_idempotent_collision_and_on_recovery(
    tmp_path: Path,
) -> None:
    """A stored event that disagrees with its asset is never trusted.

    Such a row certifies a boundary for content its asset never held, so
    both the idempotent-collision path and recovery refuse rather than
    returning it.
    """
    store = AssetStore(root=tmp_path)
    at = datetime(2026, 10, 18, 12, tzinfo=UTC)
    payload = b'{"facts": {}}'
    asset = _observation_asset(store, name="corrupt", payload=payload, at=at)
    # A row written before the digest/asset binding existed.
    SourceObservationEvent.objects.create(
        provider="sec",
        kind="sec_companyfacts",
        subject="0000320193",
        content_sha256="a" * 64,
        source_asset=asset,
        observed_at=at,
    )

    with pytest.raises(sec_ingestion.ObservationEvidenceError, match="but its source asset"):
        sec_ingestion._record_observation_event(
            asset=asset,
            kind="sec_companyfacts",
            subject="0000320193",
            digest=hashlib.sha256(payload).hexdigest(),
            observed_at=at,
        )
    with pytest.raises(sec_ingestion.ObservationEvidenceError, match="but its source asset"):
        sec_ingestion._latest_observed_companyfacts("0000320193")


def test_an_unrelated_integrity_error_is_never_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only the observation-instant conflict may be recovered from.

    A blanket ``except IntegrityError`` would turn a foreign-key or not-null
    fault into a silent success by returning whatever row sits at that
    instant. The narrowed check must re-raise unchanged both when a row
    exists at the instant and when none does.
    """
    store = AssetStore(root=tmp_path)
    at = datetime(2026, 10, 18, 12, tzinfo=UTC)
    payload = b'{"facts": {}}'
    asset = _observation_asset(store, name="unrelated", payload=payload, at=at)
    digest = hashlib.sha256(payload).hexdigest()
    unrelated = IntegrityError("FOREIGN KEY constraint failed")

    def _explode(*args: object, **kwargs: object) -> SourceObservationEvent:
        raise unrelated

    monkeypatch.setattr(SourceObservationEvent.objects, "create", _explode)

    # (a) No row exists at that instant.
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        sec_ingestion._record_observation_event(
            asset=asset,
            kind="sec_companyfacts",
            subject="0000320193",
            digest=digest,
            observed_at=at,
        )

    # (b) A row *does* exist at that instant, and it still re-raises.
    monkeypatch.undo()
    sec_ingestion._record_observation_event(
        asset=asset,
        kind="sec_companyfacts",
        subject="0000320193",
        digest=digest,
        observed_at=at,
    )
    assert SourceObservationEvent.objects.count() == 1
    monkeypatch.setattr(SourceObservationEvent.objects, "create", _explode)
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        sec_ingestion._record_observation_event(
            asset=asset,
            kind="sec_companyfacts",
            subject="0000320193",
            digest=digest,
            observed_at=at,
        )


def test_a_real_instant_collision_is_still_handled(tmp_path: Path) -> None:
    """The narrowing must not break ordinary collision handling."""
    store = AssetStore(root=tmp_path)
    at = datetime(2026, 10, 18, 12, tzinfo=UTC)
    payload = b'{"facts": {}}'
    asset = _observation_asset(store, name="collide", payload=payload, at=at)
    digest = hashlib.sha256(payload).hexdigest()

    first = sec_ingestion._record_observation_event(
        asset=asset,
        kind="sec_companyfacts",
        subject="0000320193",
        digest=digest,
        observed_at=at,
    )
    again = sec_ingestion._record_observation_event(
        asset=asset,
        kind="sec_companyfacts",
        subject="0000320193",
        digest=digest,
        observed_at=at,
    )
    assert again.pk == first.pk

    other = _observation_asset(store, name="other", payload=b'{"facts": {"x": 1}}', at=at)
    with pytest.raises(ProviderResponseError, match="two different payloads observed"):
        sec_ingestion._record_observation_event(
            asset=other,
            kind="sec_companyfacts",
            subject="0000320193",
            digest=other.sha256,
            observed_at=at,
        )
    assert SourceObservationEvent.objects.count() == 1

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

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
)
from stanstock.data.providers import sec
from stanstock.data.providers.contracts import FundamentalSourcePayload
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
    companyfacts = {"content": _companyfacts_bytes()}
    submissions = {"content": SUBMISSIONS_BYTES}
    _patch_provider(monkeypatch, companyfacts, submissions)
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
    submissions["content"] = SUBMISSIONS_BYTES.replace(
        b"Electronic Computers",
        b"Electronic Computers Updated",
    )
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
    submissions["content"] = submissions["content"].replace(
        b"Electronic Computers Updated",
        b"Electronic Computers Revised",
    )
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

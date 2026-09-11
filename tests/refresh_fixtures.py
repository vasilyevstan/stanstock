"""Compact, reusable synthetic-evidence builders for scheduled-refresh tests.

`stanstock.core.refresh_verification` independently re-derives evidence from
persisted rows -- it never trusts a mocked child's own bookkeeping. Tests
that exercise it through `scheduled_refresh` therefore need *genuine*
persisted `UniverseSnapshot`/`AnalysisRun`/`StockAnalysis`/`Prediction`/
`LatestMarketData` rows, not hand-built `JobExecutionResult` stand-ins.

This module drives the real `run_us_daily` pipeline with a synthetic Twelve
Data provider boundary (the same technique `tests/test_data_live_us.py`
uses), so callers get real evidence without duplicating that suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml
from django.contrib.auth import get_user_model

from stanstock.core.verification_types import AssetRef
from stanstock.data import sec_ingestion
from stanstock.data.assets import AssetStore, asset_ref_for
from stanstock.data.live_us import PRIVATE_USAGE_SCOPE, UsUniverseConfig
from stanstock.data.live_us import _years_before as _history_start_before
from stanstock.data.models import (
    Company,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    Listing,
    ProviderRecord,
    Region,
    Security,
)
from stanstock.data.providers import sec
from stanstock.data.providers.contracts import (
    PriceBar,
    PriceSeries,
    StockCatalog,
    StockReference,
)
from stanstock.data.sec_config import load_sec_cik_config, load_sec_fundamentals_config
from stanstock.data.sec_evidence import MAPPING_SUBJECT

TEST_API_KEY = "private-test-key"


def create_portfolio_owner(*, username: str = "refresh-verification-owner") -> Any:
    """Create the minimal `User` a `Portfolio.owner` foreign key requires."""
    return get_user_model().objects.create_user(username=username, password="not-used")  # noqa: S106


def build_universe_config(
    *,
    symbols: tuple[str, ...] = ("AAA", "BBB"),
    minimum_eligible: int = 1,
) -> UsUniverseConfig:
    raw: dict[str, Any] = {
        "schema_version": 1,
        "config_version": "test-v1",
        "slug": "test-us-live",
        "name": "Test US Live",
        "description": "Test-only private US universe.",
        "provider": "twelve_data",
        "country": "United States",
        "instrument_type": "Common Stock",
        "currency": "USD",
        "exchanges": ["NASDAQ"],
        "benchmark_symbol": "SPY",
        "benchmark_currency": "USD",
        "benchmark_type": "ETF",
        "history_years": 1,
        "minimum_history_sessions": 1,
        "price_adjustment": "splits",
        "minimum_eligible": minimum_eligible,
        "maximum_symbols": 10,
        "symbols": list(symbols),
    }
    return UsUniverseConfig(
        slug="test-us-live",
        name="Test US Live",
        description="Test-only private US universe.",
        config_version="test-v1",
        country="United States",
        instrument_type="Common Stock",
        currency="USD",
        exchanges=("NASDAQ",),
        benchmark_symbol="SPY",
        benchmark_currency="USD",
        benchmark_type="ETF",
        history_years=1,
        minimum_history_sessions=1,
        price_adjustment="splits",
        minimum_eligible=minimum_eligible,
        maximum_symbols=10,
        symbols=symbols,
        raw=raw,
    )


def enable_twelve_data_provider(*, daily_limit: int = 800) -> ProviderRecord:
    # `credits_per_minute` is set high (rather than disabling spacing, which
    # `execute_us_daily_job` -- the real production entrypoint under test --
    # has no parameter to do) so the real rate-limit spacing path still runs
    # but its sleeps are negligible for a synthetic-fixture test.
    return ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=True,
        terms_url="https://twelvedata.com/terms",
        usage_scope=PRIVATE_USAGE_SCOPE,
        status="ok",
        metadata={
            "daily_credit_limit": daily_limit,
            "credits_per_minute": 6000,
            "internal_display_rights_confirmed": True,
            "plan": "grow",
        },
    )


def set_twelve_data_api_key(monkeypatch: pytest.MonkeyPatch, *, key: str = TEST_API_KEY) -> None:
    """Set the env var `run_us_daily` resolves when no explicit key is given.

    `execute_us_daily_job` (the real production entrypoint `scheduled_refresh`
    calls) never passes an explicit `api_key`, so tests that exercise the
    real pipeline through it must supply this env var rather than the
    `api_key=` keyword `test_data_live_us.py` uses directly against
    `run_us_daily`.
    """
    monkeypatch.setenv("TWELVE_DATA_API_KEY", key)


def _reference(symbol: str) -> StockReference:
    return StockReference(
        symbol=symbol,
        name=f"{symbol} Incorporated",
        currency="USD",
        exchange="NASDAQ",
        mic_code="XNAS",
        country="United States",
        instrument_type="Common Stock",
        figi_code=f"BBG-{symbol}",
        access_plan="Basic",
    )


def _catalog(config: UsUniverseConfig, *, retrieved_at: datetime) -> StockCatalog:
    return StockCatalog(
        provider="twelve_data",
        exchange="NASDAQ",
        references=tuple(_reference(symbol) for symbol in config.symbols),
        count=len(config.symbols),
        retrieved_at=retrieved_at,
        source_url="https://api.twelvedata.com/stocks?exchange=NASDAQ",
        raw_bytes=b'{"status":"ok","data":[]}',
    )


def _series(
    symbol: str,
    *,
    target_date: date,
    retrieved_at: datetime,
    instrument_type: str = "Common Stock",
) -> PriceSeries:
    # The first bar is anchored exactly on `history_start` (matching
    # `build_universe_config`'s fixed `history_years=1`) -- not merely one
    # day before target -- because `_price_series_exclusion_reason` rejects
    # any listing whose earliest bar arrives more than 14 days after the
    # configured `history_start`, deliberately catching a provider response
    # that is missing the required trailing history window.
    bars = (
        PriceBar(
            trade_date=_history_start_before(target_date, 1),
            open=Decimal("99"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=1_000_000,
        ),
        PriceBar(
            trade_date=target_date,
            open=Decimal("100"),
            high=Decimal("104"),
            low=Decimal("99"),
            close=Decimal("103"),
            volume=1_200_000,
        ),
    )
    return PriceSeries(
        provider="twelve_data",
        symbol=symbol,
        currency="USD",
        bars=bars,
        retrieved_at=retrieved_at,
        source_url=f"https://api.twelvedata.com/time_series?symbol={symbol}",
        raw_bytes=f'{{"status":"ok","symbol":"{symbol}"}}'.encode(),
        exchange="NASDAQ" if instrument_type == "Common Stock" else "NYSE ARCA",
        mic_code="XNAS" if instrument_type == "Common Stock" else "ARCX",
        instrument_type=instrument_type,
        exchange_timezone="America/New_York",
        adjustment="splits",
    )


def patch_twelve_data_provider(
    monkeypatch: pytest.MonkeyPatch,
    config: UsUniverseConfig,
    *,
    target_date: date,
    retrieved_at: datetime,
    api_key: str = TEST_API_KEY,
) -> tuple[list[str], list[str]]:
    """Patch the Twelve Data provider boundary the real pipeline calls.

    Deliberately patches `stanstock.data.live_us.twelve_data.*` (the exact
    boundary `run_us_daily` calls) rather than `run_us_daily` or
    `analyze_snapshot` themselves, so callers get a genuinely-executed
    pipeline and real persisted evidence. Returns the `(catalog_calls,
    price_calls)` spy lists so a caller can prove a retry/no-op made zero
    additional provider calls.
    """
    catalog_calls: list[str] = []
    price_calls: list[str] = []

    def fetch_catalog(**kwargs: object) -> StockCatalog:
        assert kwargs["api_key"] == api_key
        catalog_calls.append(str(kwargs["exchange"]))
        return _catalog(config, retrieved_at=retrieved_at)

    def fetch_prices(symbol: str, **kwargs: object) -> PriceSeries:
        assert kwargs["api_key"] == api_key
        price_calls.append(symbol)
        instrument_type = "ETF" if symbol == config.benchmark_symbol else "Common Stock"
        return _series(
            symbol,
            target_date=target_date,
            retrieved_at=retrieved_at,
            instrument_type=instrument_type,
        )

    monkeypatch.setattr(
        "stanstock.data.live_us.twelve_data.fetch_stock_catalog",
        fetch_catalog,
    )
    monkeypatch.setattr(
        "stanstock.data.live_us.twelve_data.fetch_daily_price_series",
        fetch_prices,
    )
    return catalog_calls, price_calls


def pre_create_stock_listing(symbol: str) -> Listing:
    """Register a Common Stock listing the real `_ensure_listings` will reuse.

    `run_us_daily`'s `_ensure_listings` reuses (rather than recreates) any
    existing active listing whose `provider_symbol`/`region` matches and
    whose `currency`/`exchange_mic` agree with the synthetic catalog
    reference this module fetches (`USD`/`XNAS`). Pre-creating the listing
    here -- rather than letting the pipeline mint a fresh, unpredictable
    `Company` UUID -- lets a test know the exact `Company` a SEC
    fundamental fact must reference *before* the real market pipeline runs.
    """
    company = Company.objects.create(name=f"{symbol} Incorporated", country="US")
    security = Security.objects.create(
        company=company,
        security_type=Security.SecurityType.COMMON_STOCK,
        name=f"{symbol} Incorporated",
    )
    return Listing.objects.create(
        security=security,
        ticker=symbol,
        exchange_mic="XNAS",
        provider_symbol=symbol,
        currency="USD",
        region=Region.US,
        valid_from=date(2020, 1, 1),
        is_primary=True,
        is_active=True,
    )


@dataclass(frozen=True)
class SecEvidence:
    mapping_asset_id: str
    mapping_sha256: str
    cik_config_version: str
    cik_config_hash: str
    fundamentals_config_version: str
    fundamentals_config_hash: str
    fact: FundamentalFact
    asset_refs: tuple[AssetRef, ...]


def build_sec_evidence(
    *,
    store: AssetStore,
    company: Company,
    available_before: datetime,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SecEvidence:
    """Persist one minimal-but-real, cutoff-eligible SEC evidence set.

    Writes real files through `AssetStore` and creates real `DataAsset`/
    `FundamentalFact` rows -- rather than a `SimpleNamespace` stand-in -- so
    `refresh_verification._require_sec_mapping_asset`/`_require_sec_fact`
    resolve genuine evidence, not a mocked shape that merely looks right.

    `_require_sec_mapping_asset` binds the mapping asset's own checksum to
    the *reviewed* CIK config's pinned `source_sha256`. A SHA-256 cannot be
    reverse-engineered to match an arbitrary fixture payload, so this helper
    points `load_sec_cik_config` at a small, self-consistent temporary
    config whose `source_sha256` is the real checksum of the bytes this
    fixture actually writes to disk -- exercising the identical production
    binding without requiring a live SEC-retrieved fixture file.
    """
    retrieved_at = available_before - timedelta(days=1)
    mapping_written = store.write_bytes(
        f"sec/mapping-{company.pk}.json",
        b'{"cik_lookup": {}}',
    )
    cik_config_path = tmp_path / "sec-cik-fixture.yml"
    cik_config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "config_version": "test-sec-cik-v1",
                "universe_config_version": "test-universe-v1",
                "source_sha256": mapping_written.sha256,
                "mappings": {
                    "ZZZ": {"cik": "0000000001", "exchange": "Nasdaq", "company_name": "Fixture Co"}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "stanstock.data.sec_config.default_sec_cik_mapping_path",
        lambda: cik_config_path,
    )
    cik_config = load_sec_cik_config()
    mapping_asset = DataAsset.objects.create(
        provider=sec.PROVIDER,
        kind=sec_ingestion.MAPPING_KIND,
        subject=MAPPING_SUBJECT,
        relative_path=mapping_written.relative_path,
        sha256=cik_config.source_sha256,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
    )
    fact_written = store.write_bytes(
        f"sec/{company.pk}/companyfacts.json",
        f'{{"cik": "{company.pk}"}}'.encode(),
    )
    fact_asset = DataAsset.objects.create(
        provider=sec.PROVIDER,
        kind="companyfacts",
        subject=str(company.pk),
        relative_path=fact_written.relative_path,
        sha256=fact_written.sha256,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
    )
    submissions_written = store.write_bytes(
        f"sec/{company.pk}/submissions.json",
        f'{{"cik": "{company.pk}", "filings": {{"files": []}}}}'.encode(),
    )
    submissions_asset = DataAsset.objects.create(
        provider=sec.PROVIDER,
        kind=sec_ingestion.SUBMISSIONS_KIND,
        subject="0000000001",
        relative_path=submissions_written.relative_path,
        sha256=submissions_written.sha256,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
    )
    companyfacts_written = store.write_bytes(
        f"sec/{company.pk}/companyfacts-exact.json",
        f'{{"cik": "{company.pk}"}}'.encode(),
    )
    companyfacts_asset = DataAsset.objects.create(
        provider=sec.PROVIDER,
        kind=sec_ingestion.COMPANYFACTS_KIND,
        subject="0000000001",
        relative_path=companyfacts_written.relative_path,
        sha256=companyfacts_written.sha256,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
    )
    fact = FundamentalFact.objects.create(
        company=company,
        provider=sec.PROVIDER,
        concept="Assets",
        source_concept="us-gaap:Assets",
        value=Decimal("1000000"),
        unit="USD",
        currency="USD",
        period_type=FundamentalFact.PeriodType.INSTANT,
        period_identity="CY2025Q4I",
        period_end=date(2025, 12, 31),
        accession="0000320193-26-000001",
        filing_form="10-K",
        available_at=retrieved_at,
        observation_hash="e" * 64,
        source_asset=fact_asset,
    )
    FundamentalFactEvidence.objects.create(
        fact=fact,
        role=FundamentalFactEvidence.Role.FILING,
        source_asset=fact_asset,
    )
    return SecEvidence(
        mapping_asset_id=str(mapping_asset.pk),
        mapping_sha256=mapping_asset.sha256,
        cik_config_version=cik_config.config_version,
        cik_config_hash=cik_config.config_hash,
        fundamentals_config_version=load_sec_fundamentals_config().config_version,
        fundamentals_config_hash=load_sec_fundamentals_config().config_hash,
        fact=fact,
        asset_refs=(
            asset_ref_for(mapping_asset),
            asset_ref_for(submissions_asset),
            asset_ref_for(companyfacts_asset),
        ),
    )

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from io import StringIO
from math import sqrt

import numpy as np
import polars as pl
import pytest
from django.core.management import call_command
from django.test import override_settings

from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.etfs import (
    INVESTABLE_US_ETF_NAME,
    build_etf_overview,
    ensure_investable_spy_listing,
    sync_investable_spy_from_asset,
)
from stanstock.data.models import Company, Listing, Region, Security

pytestmark = pytest.mark.django_db


def test_spy_listing_is_idempotent_and_explicitly_an_etf() -> None:
    first = ensure_investable_spy_listing(
        currency="USD",
        mic_code="ARCX",
        valid_from=date(2026, 9, 4),
    )
    second = ensure_investable_spy_listing(
        currency="usd",
        mic_code="arcx",
        valid_from=date(2026, 9, 5),
    )

    assert second.pk == first.pk
    assert first.ticker == "SPY"
    assert first.provider_symbol == "SPY"
    assert first.exchange_mic == "ARCX"
    assert first.security.security_type == Security.SecurityType.ETF
    assert first.security.company.name == INVESTABLE_US_ETF_NAME


def test_spy_listing_rejects_a_conflicting_stock_identity() -> None:
    company = Company.objects.create(name="Conflicting SPY", country="US")
    security = Security.objects.create(
        company=company,
        security_type=Security.SecurityType.COMMON_STOCK,
        name="Conflicting SPY Common",
    )
    Listing.objects.create(
        security=security,
        ticker="SPY",
        exchange_mic="ARCX",
        provider_symbol="SPY",
        currency="USD",
        region=Region.US,
    )

    with pytest.raises(ValueError, match="not identified as an exchange-traded fund"):
        ensure_investable_spy_listing(
            currency="USD",
            mic_code="ARCX",
            valid_from=date(2026, 9, 4),
        )


def test_spy_asset_sync_builds_price_only_etf_overview(tmp_path) -> None:
    start = date(2025, 9, 1)
    dates = [start + timedelta(days=index) for index in range(260)]
    closes = [Decimal("100") + Decimal(index) / Decimal("10") for index in range(260)]
    closes[180] = closes[179] * Decimal("0.80")
    frame = pl.DataFrame(
        {
            "date": dates,
            "close": [float(value) for value in closes],
            "volume": [1_000_000 + index for index in range(260)],
        }
    )
    store = AssetStore(tmp_path)
    stored = store.write_frame("tests/spy-etf.parquet", frame)
    retrieved_at = datetime(2026, 9, 5, 1, tzinfo=UTC)
    asset = register_asset(
        provider="twelve_data",
        kind="price_history",
        subject="SPY",
        stored=stored,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
        period_start=dates[0],
        period_end=dates[-1],
        metadata={
            "currency": "USD",
            "mic_code": "ARCX",
            "instrument_type": "ETF",
            "interval": "1day",
            "adjustment": "splits",
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )

    listing = sync_investable_spy_from_asset(
        asset=asset,
        target_date=dates[-1],
        store=store,
    )
    overview = build_etf_overview(listing, store=store)

    assert listing.latest_market_data.source_asset == asset
    assert listing.latest_market_data.close == closes[-1]
    assert overview.observation_count == 253
    assert overview.period_start == dates[-253]
    assert overview.period_end == dates[-1]
    expected_closes = np.asarray([float(value) for value in closes[-253:]])
    expected_returns = expected_closes[1:] / expected_closes[:-1] - 1.0
    expected_running_high = np.maximum.accumulate(expected_closes)
    assert overview.price_return == pytest.approx(expected_closes[-1] / expected_closes[0] - 1.0)
    assert overview.annualized_volatility == pytest.approx(
        np.std(expected_returns, ddof=1) * sqrt(252)
    )
    assert overview.max_drawdown == pytest.approx(
        np.min(expected_closes / expected_running_high - 1.0)
    )
    assert overview.benchmark_identity == "S&P 500 benchmark"
    assert overview.portfolio_role == "Core benchmark ETF"
    assert overview.dividends_included is False

    listing.delete()
    output = StringIO()
    with override_settings(DATA_DIR=tmp_path):
        call_command("sync_investable_etfs", stdout=output)
    restored = Listing.objects.select_related("security", "latest_market_data").get(
        provider_symbol="SPY"
    )
    assert restored.security.security_type == Security.SecurityType.ETF
    assert restored.latest_market_data.source_asset == asset
    assert "Synced SPY ETF" in output.getvalue()


def test_etf_overview_excludes_asset_rows_after_current_market_session(tmp_path) -> None:
    dates = [date(2026, 9, day) for day in range(1, 5)]
    frame = pl.DataFrame(
        {
            "date": dates,
            "close": [100.0, 110.0, 220.0, 55.0],
            "volume": [1_000_000] * 4,
        }
    )
    store = AssetStore(tmp_path)
    stored = store.write_frame("tests/spy-etf-cutoff.parquet", frame)
    retrieved_at = datetime(2026, 9, 5, 1, tzinfo=UTC)
    asset = register_asset(
        provider="twelve_data",
        kind="price_history",
        subject="SPY",
        stored=stored,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
        period_start=dates[0],
        period_end=dates[-1],
        metadata={
            "currency": "USD",
            "mic_code": "ARCX",
            "instrument_type": "ETF",
            "interval": "1day",
            "adjustment": "splits",
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )

    listing = sync_investable_spy_from_asset(
        asset=asset,
        target_date=dates[1],
        store=store,
    )
    overview = build_etf_overview(listing, store=store)

    assert overview.market_data.session_date == dates[1]
    assert overview.period_end == dates[1]
    assert overview.observation_count == 2
    assert overview.price_return == pytest.approx(0.10)
    assert overview.annualized_volatility is None
    assert overview.max_drawdown == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instrument_type", None),
        ("currency", None),
        ("mic_code", None),
        ("mic_code", "XNAS"),
        ("interval", None),
        ("interval", "1h"),
        ("adjustment", None),
        ("adjustment", "all"),
        ("return_definition", None),
        ("return_definition", "total_return"),
        ("dividends_included", None),
        ("dividends_included", True),
    ],
)
def test_spy_sync_rejects_unproven_or_incompatible_price_basis(
    tmp_path,
    field: str,
    value: object,
) -> None:
    target_date = date(2026, 9, 4)
    frame = pl.DataFrame(
        {
            "date": [date(2026, 9, 3), target_date],
            "close": [100.0, 101.0],
            "volume": [1_000_000, 1_100_000],
        }
    )
    store = AssetStore(tmp_path)
    stored = store.write_frame(f"tests/spy-etf-invalid-{field}-{value}.parquet", frame)
    metadata: dict[str, object] = {
        "currency": "USD",
        "mic_code": "ARCX",
        "instrument_type": "ETF",
        "interval": "1day",
        "adjustment": "splits",
        "return_definition": "split_adjusted_price_return",
        "dividends_included": False,
    }
    if value is None:
        metadata.pop(field)
    else:
        metadata[field] = value
    asset = register_asset(
        provider="twelve_data",
        kind="price_history",
        subject="SPY",
        stored=stored,
        retrieved_at=datetime(2026, 9, 5, 1, tzinfo=UTC),
        available_at=datetime(2026, 9, 5, 1, tzinfo=UTC),
        period_start=date(2026, 9, 3),
        period_end=target_date,
        metadata=metadata,
    )

    with pytest.raises(ValueError, match=field):
        sync_investable_spy_from_asset(
            asset=asset,
            target_date=target_date,
            store=store,
        )
    assert not Listing.objects.filter(provider_symbol="SPY").exists()

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.template.loader import render_to_string
from django.urls import reverse

from stanstock.data.assets import AssetStore
from stanstock.data.live_us import (
    _persist_price_series,
    refresh_my_list_price_evidence,
)
from stanstock.data.models import (
    Company,
    DataAsset,
    LatestMarketData,
    Listing,
    ProviderRecord,
    Region,
    Security,
)
from stanstock.data.provider_policy import PRIVATE_USAGE_SCOPE
from stanstock.data.providers.contracts import PriceBar, PriceSeries, StockReference
from stanstock.portfolio.models import TrackedSymbol
from stanstock.portfolio.watchlist import (
    TRACKED_SYMBOL_FILTER_NO_LIVE_PRICE,
    TRACKED_SYMBOL_FILTER_UNDER_10,
    tracked_symbol_states,
)
from stanstock.web.forms import TrackedSymbolForm

pytestmark = pytest.mark.django_db

TARGET_DATE = date(2026, 9, 11)
PREVIOUS_DATE = date(2026, 9, 10)
DECISION_TIME = datetime(2026, 9, 13, 12, tzinfo=UTC)
RETRIEVED_AT = datetime(2026, 9, 12, 1, tzinfo=UTC)


def _owner(username: str = "owner") -> Any:
    return get_user_model().objects.create_user(username=username, password="test-only")


def _reference(
    symbol: str,
    *,
    mic_code: str = "XNAS",
    instrument_type: str = "Common Stock",
) -> StockReference:
    return StockReference(
        symbol=symbol,
        name=f"{symbol} Fixture Corp",
        currency="USD",
        exchange="NASDAQ" if mic_code != "XNYS" else "NYSE",
        mic_code=mic_code,
        country="United States",
        instrument_type=instrument_type,
        access_plan="Basic",
    )


def _listing(
    symbol: str,
    *,
    mic_code: str = "XNAS",
    currency: str = "USD",
    security_type: str = Security.SecurityType.COMMON_STOCK,
) -> Listing:
    company = Company.objects.create(name=f"{symbol} Fixture Corp", country="US")
    security = Security.objects.create(
        company=company,
        security_type=security_type,
        name=f"{symbol} Fixture Security",
    )
    return Listing.objects.create(
        security=security,
        ticker=symbol,
        provider_symbol=symbol,
        exchange_mic=mic_code,
        currency=currency,
        region=Region.US,
    )


def _series(
    symbol: str,
    *,
    close: Decimal = Decimal("8.50"),
    currency: str = "USD",
    mic_code: str = "XNAS",
    instrument_type: str = "Common Stock",
    adjustment: str = "splits",
    target_date: date = TARGET_DATE,
) -> PriceSeries:
    bars = (
        PriceBar(
            trade_date=PREVIOUS_DATE,
            open=Decimal("8.10"),
            high=Decimal("8.30"),
            low=Decimal("8.00"),
            close=Decimal("8.20"),
            volume=100_000,
        ),
        PriceBar(
            trade_date=target_date,
            open=close,
            high=close + Decimal("0.20") if close > 0 else Decimal("0.20"),
            low=close,
            close=close,
            volume=110_000,
        ),
    )
    return PriceSeries(
        provider="twelve_data",
        symbol=symbol,
        currency=currency,
        bars=bars,
        retrieved_at=RETRIEVED_AT,
        source_url=f"https://api.twelvedata.com/time_series?symbol={symbol}",
        raw_bytes=(
            f'{{"status":"ok","symbol":"{symbol}","close":"{close}","nonce":"{uuid4().hex}"}}'
        ).encode(),
        exchange="NASDAQ",
        mic_code=mic_code,
        instrument_type=instrument_type,
        exchange_timezone="America/New_York",
        adjustment=adjustment,
    )


def _enable_provider() -> ProviderRecord:
    return ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=True,
        usage_scope=PRIVATE_USAGE_SCOPE,
        status="ok",
        metadata={
            "daily_credit_limit": 20,
            "credits_per_minute": 60_000,
            "internal_display_rights_confirmed": True,
            "plan": "grow",
        },
    )


def test_command_reuses_current_provider_price_before_credentials_or_quota(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Any,
) -> None:
    settings.DATA_DIR = tmp_path
    owner = _owner()
    listing = _listing("REUSE")
    TrackedSymbol.objects.create(owner=owner, symbol="REUSE")
    _persist_price_series(
        store=AssetStore(tmp_path),
        series=_series("REUSE"),
        listing=listing,
    )
    ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=False,
        metadata={"plan": "grow"},
    )

    monkeypatch.setattr(
        "stanstock.portfolio.management.commands.refresh_my_list_prices.resolve_us_target_date",
        lambda **kwargs: (TARGET_DATE, "observed"),
    )
    monkeypatch.setattr(
        "stanstock.portfolio.management.commands.refresh_my_list_prices.timezone.now",
        lambda: DECISION_TIME,
    )
    monkeypatch.setattr(
        "stanstock.portfolio.management.commands.refresh_my_list_prices."
        "verified_catalog_references_for_symbols",
        lambda *, symbols: {symbol: _reference(symbol) for symbol in symbols},
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("current price reuse must not touch credentials, quota, or provider fetch")

    monkeypatch.setattr("stanstock.data.live_us.ProviderCreditBudget.preflight", forbidden)
    monkeypatch.setattr("stanstock.data.live_us.ProviderCreditBudget.consume", forbidden)
    monkeypatch.setattr("stanstock.data.live_us.twelve_data.resolve_api_key", forbidden)
    monkeypatch.setattr("stanstock.data.live_us.twelve_data.fetch_daily_price_series", forbidden)
    output = StringIO()

    call_command("refresh_my_list_prices", stdout=output)

    text = output.getvalue()
    assert "status=success" in text
    assert "tracked=1" in text
    assert "reused=1" in text
    assert "fetched=0" in text
    assert "credits=0" in text
    assert "REUSE" not in text
    assert "8.50" not in text
    assert str(tmp_path) not in text


def test_command_rejects_more_than_twenty_tracked_symbols_before_provider_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _owner()
    for index in range(21):
        TrackedSymbol.objects.create(owner=owner, symbol=f"CAP{index:02d}")

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("over-cap admission must fail before catalog or provider work")

    monkeypatch.setattr(
        "stanstock.portfolio.management.commands.refresh_my_list_prices."
        "verified_catalog_references_for_symbols",
        forbidden,
    )
    monkeypatch.setattr("stanstock.data.live_us.twelve_data.fetch_daily_price_series", forbidden)

    with pytest.raises(CommandError, match="at most 20 tracked symbols"):
        call_command("refresh_my_list_prices", stdout=StringIO())

    assert DataAsset.objects.count() == 0
    assert LatestMarketData.objects.count() == 0


def test_command_requires_safe_owner_selection_when_multiple_owners_have_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _owner("first")
    second = _owner("second")
    TrackedSymbol.objects.create(owner=first, symbol="FIRST")
    TrackedSymbol.objects.create(owner=second, symbol="SECOND")

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("ambiguous owner selection must fail before catalog or provider work")

    monkeypatch.setattr(
        "stanstock.portfolio.management.commands.refresh_my_list_prices."
        "verified_catalog_references_for_symbols",
        forbidden,
    )
    output = StringIO()

    with pytest.raises(CommandError) as excinfo:
        call_command("refresh_my_list_prices", stdout=output)

    message = str(excinfo.value)
    assert "More than one active owner" in message
    assert "--owner-id" in message
    assert "first" not in message
    assert "second" not in message
    assert "FIRST" not in message
    assert "SECOND" not in message
    assert output.getvalue() == ""


def test_missing_prices_fetch_once_and_partial_retry_reuses_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _owner()
    provider = _enable_provider()
    calls: list[str] = []
    bad_is_fixed = False

    def fetch(symbol: str, **kwargs: object) -> PriceSeries:
        calls.append(symbol)
        assert kwargs["start_date"] == PREVIOUS_DATE
        assert kwargs["end_date"] == TARGET_DATE
        assert kwargs["outputsize"] == 2
        assert kwargs["adjustment"] == "splits"
        if symbol == "BAD" and not bad_is_fixed:
            return _series(symbol, currency="EUR")
        return _series(symbol)

    monkeypatch.setattr("stanstock.data.live_us.twelve_data.fetch_daily_price_series", fetch)
    store = AssetStore(tmp_path)
    references = {"GOOD": _reference("GOOD"), "BAD": _reference("BAD")}

    first = refresh_my_list_price_evidence(
        symbols=("GOOD", "BAD"),
        catalog_references=references,
        target_date=TARGET_DATE,
        decision_time=DECISION_TIME,
        api_key="private-test-key",
        store=store,
        enforce_rate_limit=False,
    )

    assert calls == ["GOOD", "BAD"]
    assert first.status == "partial_failed"
    assert first.fetched == 1
    assert first.failed == 1
    assert first.credits_used == 2
    assert LatestMarketData.objects.filter(listing__provider_symbol="GOOD").exists()
    provider.refresh_from_db()
    assert provider.metadata["credits_used_local"] == 2

    bad_is_fixed = True
    calls.clear()
    second = refresh_my_list_price_evidence(
        symbols=("GOOD", "BAD"),
        catalog_references=references,
        target_date=TARGET_DATE,
        decision_time=DECISION_TIME,
        api_key="private-test-key",
        store=store,
        enforce_rate_limit=False,
    )

    assert calls == ["BAD"]
    assert second.status == "success"
    assert second.reused == 1
    assert second.fetched == 1
    assert second.failed == 0
    assert second.credits_used == 1
    provider.refresh_from_db()
    assert provider.metadata["credits_used_local"] == 3


def test_local_listing_identity_conflict_fails_closed_without_fetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _owner()
    _listing("CONFLICT", mic_code="XNYS")

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("identity conflicts must fail before provider fetch")

    monkeypatch.setattr("stanstock.data.live_us.twelve_data.fetch_daily_price_series", forbidden)

    result = refresh_my_list_price_evidence(
        symbols=("CONFLICT",),
        catalog_references={"CONFLICT": _reference("CONFLICT", mic_code="XNAS")},
        target_date=TARGET_DATE,
        decision_time=DECISION_TIME,
        api_key="private-test-key",
        store=AssetStore(tmp_path),
        enforce_rate_limit=False,
    )

    assert result.status == "partial_failed"
    assert result.failed == 1
    assert result.credits_used == 0
    assert "conflicts with verified catalog identity" in result.symbols[0].reason
    assert str(tmp_path) not in result.symbols[0].reason


@pytest.mark.parametrize("instrument_type", ["ADR", "American Depositary Receipt"])
def test_catalog_backed_adr_identity_can_materialize_for_price_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    instrument_type: str,
) -> None:
    _owner()
    _enable_provider()
    symbol = f"ADR{instrument_type[0]}"

    def fetch(symbol: str, **kwargs: object) -> PriceSeries:
        return _series(symbol, instrument_type=instrument_type)

    monkeypatch.setattr("stanstock.data.live_us.twelve_data.fetch_daily_price_series", fetch)

    result = refresh_my_list_price_evidence(
        symbols=(symbol,),
        catalog_references={symbol: _reference(symbol, instrument_type=instrument_type)},
        target_date=TARGET_DATE,
        decision_time=DECISION_TIME,
        api_key="private-test-key",
        store=AssetStore(tmp_path),
        enforce_rate_limit=False,
    )

    assert result.status == "success"
    listing = Listing.objects.get(provider_symbol=symbol)
    assert listing.security.security_type == Security.SecurityType.ADR
    assert listing.latest_market_data.session_date == TARGET_DATE


@pytest.mark.parametrize(
    "series",
    [
        pytest.param(lambda: replace(_series("BAD"), symbol="OTHER"), id="symbol"),
        pytest.param(lambda: _series("BAD", currency="EUR"), id="currency"),
        pytest.param(lambda: _series("BAD", mic_code="XNYS"), id="mic"),
        pytest.param(lambda: _series("BAD", instrument_type="ETF"), id="type"),
        pytest.param(lambda: _series("BAD", adjustment="all"), id="adjustment"),
        pytest.param(lambda: _series("BAD", target_date=PREVIOUS_DATE), id="stale"),
        pytest.param(
            lambda: replace(
                _series("BAD"),
                bars=(
                    *_series("BAD").bars,
                    PriceBar(
                        trade_date=TARGET_DATE + timedelta(days=3),
                        open=Decimal("8.60"),
                        high=Decimal("8.90"),
                        low=Decimal("8.50"),
                        close=Decimal("8.70"),
                        volume=120_000,
                    ),
                ),
            ),
            id="future",
        ),
        pytest.param(lambda: _series("BAD", close=Decimal("0")), id="nonpositive"),
    ],
)
def test_invalid_provider_price_evidence_is_rejected_without_success_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    series: Callable[[], PriceSeries],
) -> None:
    _owner()
    _enable_provider()
    calls: list[str] = []

    def fetch(symbol: str, **kwargs: object) -> PriceSeries:
        calls.append(symbol)
        return series()

    monkeypatch.setattr("stanstock.data.live_us.twelve_data.fetch_daily_price_series", fetch)

    result = refresh_my_list_price_evidence(
        symbols=("BAD",),
        catalog_references={"BAD": _reference("BAD")},
        target_date=TARGET_DATE,
        decision_time=DECISION_TIME,
        api_key="private-test-key",
        store=AssetStore(tmp_path),
        enforce_rate_limit=False,
    )

    assert calls == ["BAD"]
    assert result.status == "partial_failed"
    assert result.fetched == 0
    assert result.failed == 1
    assert result.credits_used == 1
    assert not LatestMarketData.objects.exists()
    assert str(tmp_path) not in result.symbols[0].reason


def test_under_ten_filter_uses_only_current_twelve_data_usd_prices(
    tmp_path: Path,
) -> None:
    owner = _owner()
    live_under = _listing("LIVEU")
    live_over = _listing("LIVEO")
    demo_under = _listing("DEMO")
    stale_under = _listing("STALE")
    missing = _listing("MISS")
    for listing in (live_under, live_over, demo_under, stale_under, missing):
        TrackedSymbol.objects.create(owner=owner, symbol=listing.provider_symbol)

    store = AssetStore(tmp_path)
    _persist_price_series(
        store=store,
        series=_series("LIVEU", close=Decimal("8.75")),
        listing=live_under,
    )
    _persist_price_series(
        store=store,
        series=_series("LIVEO", close=Decimal("12.25")),
        listing=live_over,
    )
    _persist_price_series(
        store=store,
        series=_series("STALE", target_date=PREVIOUS_DATE, close=Decimal("7.50")),
        listing=stale_under,
    )
    demo_asset = DataAsset.objects.create(
        provider="synthetic_demo",
        kind="price_history",
        subject="DEMO",
        relative_path="tests/demo-watchlist.parquet",
        sha256="d" * 64,
        retrieved_at=RETRIEVED_AT,
        available_at=RETRIEVED_AT,
        period_end=TARGET_DATE,
        metadata={"currency": "USD", "interval": "1day", "adjustment": "splits"},
    )
    LatestMarketData.objects.create(
        listing=demo_under,
        observed_at=RETRIEVED_AT,
        session_date=TARGET_DATE,
        close=Decimal("5.00"),
        previous_close=None,
        volume=None,
        source_asset=demo_asset,
    )

    under_states = tracked_symbol_states(
        owner=owner,
        selected_run=None,
        price_filter=TRACKED_SYMBOL_FILTER_UNDER_10,
        live_session_date=TARGET_DATE,
        decision_time=DECISION_TIME,
    )
    no_live_states = tracked_symbol_states(
        owner=owner,
        selected_run=None,
        price_filter=TRACKED_SYMBOL_FILTER_NO_LIVE_PRICE,
        live_session_date=TARGET_DATE,
        decision_time=DECISION_TIME,
    )

    assert [state.preference.symbol for state in under_states] == ["LIVEU"]
    assert {state.preference.symbol for state in no_live_states} == {"DEMO", "MISS", "STALE"}

    content = render_to_string(
        "web/my_list.html",
        {
            "form": TrackedSymbolForm(),
            "selected_analysis_run": None,
            "tracked_symbols": under_states,
            "tracked_symbol_filter": TRACKED_SYMBOL_FILTER_UNDER_10,
        },
    )
    assert "Under $10 provider-backed" in content
    assert "Source Twelve Data" in content
    assert "0% new allocation" in content
    assert "LIVEU" in content
    assert "LIVEO" not in content
    assert "DEMO" not in content


def test_my_list_page_applies_price_filter_without_viewing_private_symbols(
    client: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    owner = _owner()
    live_under = _listing("PAGEU")
    live_over = _listing("PAGEO")
    demo_under = _listing("PAGED")
    for listing in (live_under, live_over, demo_under):
        TrackedSymbol.objects.create(owner=owner, symbol=listing.provider_symbol)

    store = AssetStore(tmp_path)
    _persist_price_series(
        store=store,
        series=_series("PAGEU", close=Decimal("8.75")),
        listing=live_under,
    )
    _persist_price_series(
        store=store,
        series=_series("PAGEO", close=Decimal("12.25")),
        listing=live_over,
    )
    demo_asset = DataAsset.objects.create(
        provider="synthetic_demo",
        kind="price_history",
        subject="PAGED",
        relative_path="tests/page-demo-watchlist.parquet",
        sha256="e" * 64,
        retrieved_at=RETRIEVED_AT,
        available_at=RETRIEVED_AT,
        period_end=TARGET_DATE,
        metadata={"currency": "USD", "interval": "1day", "adjustment": "splits"},
    )
    LatestMarketData.objects.create(
        listing=demo_under,
        observed_at=RETRIEVED_AT,
        session_date=TARGET_DATE,
        close=Decimal("5.00"),
        previous_close=None,
        volume=None,
        source_asset=demo_asset,
    )
    monkeypatch.setattr(
        "stanstock.portfolio.watchlist.resolve_us_target_date",
        lambda **kwargs: (TARGET_DATE, "observed"),
    )
    client.force_login(owner)

    response = client.get(reverse("my-list"), {"price_filter": TRACKED_SYMBOL_FILTER_UNDER_10})

    content = response.content.decode()
    assert response.status_code == 200
    assert response.context["tracked_symbol_filter"] == TRACKED_SYMBOL_FILTER_UNDER_10
    assert [state.preference.symbol for state in response.context["tracked_symbols"]] == ["PAGEU"]
    assert "PAGEU" in content
    assert "PAGEO" not in content
    assert "PAGED" not in content
    assert "Source Twelve Data" in content
    assert "0% new allocation" in content

from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from stanstock.data.providers import twelve_data
from stanstock.data.providers.exceptions import (
    ProviderBlockedError,
    ProviderConfigurationError,
    ProviderDataError,
    ProviderQuotaError,
    ProviderResponseError,
)
from stanstock.data.providers.http import HttpFetchResult

VALID_SERIES = {
    "meta": {
        "symbol": "AAPL",
        "interval": "1day",
        "currency": "USD",
        "exchange_timezone": "America/New_York",
        "exchange": "NASDAQ",
        "mic_code": "XNGS",
        "type": "Common Stock",
    },
    "values": [
        {
            "datetime": "2026-09-03",
            "open": "324.87",
            "high": "330.81",
            "low": "324.10",
            "close": "328.21",
            "volume": "37225800",
        },
        {
            "datetime": "2026-09-04",
            "open": "328.31",
            "high": "328.93",
            "low": "317.86",
            "close": "319.97",
            "volume": "39551800",
        },
    ],
    "status": "ok",
}

VALID_CATALOG = {
    "data": [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "currency": "USD",
            "exchange": "NASDAQ",
            "mic_code": "XNGS",
            "country": "United States",
            "type": "Common Stock",
            "figi_code": "BBG000B9Y5X2",
            "access": {
                "global": "Basic",
                "plan": "Basic",
                "plan_business": "Basic",
            },
        }
    ],
    "count": 1,
    "status": "ok",
}


def _result(
    payload: object,
    *,
    status_code: int = 200,
    content_type: str = "application/json; charset=utf-8",
    url: str = twelve_data.TIME_SERIES_URL,
) -> HttpFetchResult:
    return HttpFetchResult(
        status_code=status_code,
        headers=httpx.Headers({"content-type": content_type}),
        content=json.dumps(payload).encode(),
        url=url,
    )


def test_parses_split_adjusted_daily_series_and_uses_header_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_fetch(*args: object, **kwargs: object) -> HttpFetchResult:
        captured.update(kwargs)
        return _result(VALID_SERIES)

    monkeypatch.setattr(twelve_data, "fetch", fake_fetch)

    series = twelve_data.fetch_daily_price_series(
        "aapl",
        start_date=date(2026, 9, 3),
        end_date=date(2026, 9, 4),
        api_key="private-test-key",
    )

    assert series.provider == "twelve_data"
    assert series.symbol == "AAPL"
    assert series.currency == "USD"
    assert series.exchange == "NASDAQ"
    assert series.mic_code == "XNGS"
    assert series.instrument_type == "Common Stock"
    assert series.adjustment == "splits"
    assert [bar.trade_date for bar in series.bars] == [
        date(2026, 9, 3),
        date(2026, 9, 4),
    ]
    assert str(series.bars[-1].close) == "319.97"
    assert series.bars[-1].volume == 39_551_800
    assert captured["headers"] == {
        "Accept": "application/json",
        "Authorization": "apikey private-test-key",
    }
    assert captured["params"]["end_date"] == "2026-09-05"
    assert "private-test-key" not in series.source_url


def test_missing_or_demo_key_is_rejected_for_live_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(twelve_data.API_KEY_ENV, raising=False)
    monkeypatch.setattr(twelve_data, "read_twelve_data_api_key", lambda: None)

    with pytest.raises(ProviderConfigurationError, match="is required"):
        twelve_data.resolve_api_key()
    with pytest.raises(ProviderConfigurationError, match="demo key"):
        twelve_data.resolve_api_key("demo")

    assert twelve_data.resolve_api_key("demo", allow_demo=True) == "demo"


def test_api_key_resolution_uses_environment_before_keychain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(twelve_data.API_KEY_ENV, "environment-key")
    monkeypatch.setattr(
        twelve_data,
        "read_twelve_data_api_key",
        lambda: pytest.fail("keychain should not be read when the environment is set"),
    )

    assert twelve_data.resolve_api_key() == "environment-key"


def test_api_key_resolution_falls_back_to_macos_keychain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(twelve_data.API_KEY_ENV, raising=False)
    monkeypatch.setattr(twelve_data, "read_twelve_data_api_key", lambda: "keychain-key")

    assert twelve_data.resolve_api_key() == "keychain-key"


def test_noninteractive_api_key_resolution_never_reads_keychain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(twelve_data.API_KEY_ENV, raising=False)
    monkeypatch.setenv(twelve_data.DISABLE_KEYCHAIN_ENV, "1")
    monkeypatch.setattr(
        twelve_data,
        "read_twelve_data_api_key",
        lambda: pytest.fail("disabled keychain must not be read"),
    )

    with pytest.raises(ProviderConfigurationError, match="is required"):
        twelve_data.resolve_api_key()


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (
            {"status": "error", "code": 401, "message": "API key is invalid"},
            ProviderConfigurationError,
        ),
        (
            {"status": "error", "code": 429, "message": "API credits exhausted"},
            ProviderQuotaError,
        ),
        (
            {
                "status": "error",
                "code": 403,
                "message": "This symbol is not available with your plan",
            },
            ProviderBlockedError,
        ),
    ],
)
def test_classifies_documented_api_errors(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    error: type[Exception],
) -> None:
    monkeypatch.setattr(twelve_data, "fetch", lambda *args, **kwargs: _result(payload))

    with pytest.raises(error):
        twelve_data.fetch_daily_price_series("AAPL", api_key="test-key")


def test_rejects_malformed_or_inconsistent_daily_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        **VALID_SERIES,
        "values": [
            {
                "datetime": "2026-09-04",
                "open": "328.31",
                "high": "300.00",
                "low": "317.86",
                "close": "319.97",
                "volume": "39551800",
            }
        ],
    }
    monkeypatch.setattr(twelve_data, "fetch", lambda *args, **kwargs: _result(payload))

    with pytest.raises(ProviderDataError, match="high was below"):
        twelve_data.fetch_daily_price_series("AAPL", api_key="test-key")


def test_rejects_dates_outside_requested_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        twelve_data,
        "fetch",
        lambda *args, **kwargs: _result(VALID_SERIES),
    )

    with pytest.raises(ProviderResponseError, match="before requested start"):
        twelve_data.fetch_daily_price_series(
            "AAPL",
            start_date=date(2026, 9, 4),
            end_date=date(2026, 9, 4),
            api_key="test-key",
        )


def test_parses_us_stock_reference_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        twelve_data,
        "fetch",
        lambda *args, **kwargs: _result(VALID_CATALOG, url=twelve_data.STOCKS_URL),
    )

    catalog = twelve_data.fetch_stock_catalog(exchange="nasdaq", api_key="test-key")

    assert catalog.provider == "twelve_data"
    assert catalog.exchange == "NASDAQ"
    assert catalog.count == 1
    reference = catalog.references[0]
    assert reference.symbol == "AAPL"
    assert reference.name == "Apple Inc."
    assert reference.currency == "USD"
    assert reference.mic_code == "XNGS"
    assert reference.access_plan == "Basic"


def test_catalog_filter_ignores_malformed_unrequested_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        **VALID_CATALOG,
        "data": [
            *VALID_CATALOG["data"],
            {
                "symbol": "UNUSED",
                "name": None,
                "currency": "USD",
                "exchange": "NASDAQ",
                "mic_code": "XNGS",
                "country": "United States",
                "type": "Common Stock",
            },
        ],
        "count": 2,
    }
    monkeypatch.setattr(
        twelve_data,
        "fetch",
        lambda *args, **kwargs: _result(payload, url=twelve_data.STOCKS_URL),
    )

    catalog = twelve_data.fetch_stock_catalog(
        exchange="nasdaq",
        required_symbols={"AAPL"},
        api_key="test-key",
    )

    assert [reference.symbol for reference in catalog.references] == ["AAPL"]
    assert catalog.count == 2


def test_catalog_filter_still_rejects_malformed_required_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        **VALID_CATALOG,
        "data": [{**VALID_CATALOG["data"][0], "name": None}],
    }
    monkeypatch.setattr(
        twelve_data,
        "fetch",
        lambda *args, **kwargs: _result(payload, url=twelve_data.STOCKS_URL),
    )

    with pytest.raises(ProviderResponseError, match="had no usable 'name'"):
        twelve_data.fetch_stock_catalog(
            exchange="nasdaq",
            required_symbols={"AAPL"},
            api_key="test-key",
        )


def test_http_rate_limit_is_not_misreported_as_empty_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        twelve_data,
        "fetch",
        lambda *args, **kwargs: _result({}, status_code=429),
    )

    with pytest.raises(ProviderQuotaError, match="credit limit"):
        twelve_data.fetch_daily_price_series("AAPL", api_key="test-key")


def test_http_failure_is_not_downgraded_to_instrument_data_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        twelve_data,
        "fetch",
        lambda *args, **kwargs: _result({}, status_code=502),
    )

    with pytest.raises(ProviderResponseError, match="HTTP 502") as error:
        twelve_data.fetch_daily_price_series("AAPL", api_key="test-key")

    assert not isinstance(error.value, ProviderDataError)

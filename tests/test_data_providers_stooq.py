from __future__ import annotations

import httpx
import pytest

from stanstock.data.providers import stooq
from stanstock.data.providers.exceptions import (
    ProviderBlockedError,
    ProviderConfigurationError,
    ProviderResponseError,
)
from stanstock.data.providers.http import HttpFetchResult

VALID_CSV = (
    b"Date,Open,High,Low,Close,Volume\n"
    b"2026-09-01,10.0,10.5,9.8,10.2,12345\n"
    b"2026-09-02,10.2,10.9,10.1,10.7,15000\n"
)


def _result(
    content: bytes,
    *,
    status_code: int = 200,
    content_type: str = "text/csv",
) -> HttpFetchResult:
    return HttpFetchResult(
        status_code=status_code,
        headers=httpx.Headers({"content-type": content_type}),
        content=content,
        url=stooq.BASE_URL,
    )


def test_parses_valid_daily_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stooq, "fetch", lambda *args, **kwargs: _result(VALID_CSV))

    series = stooq.fetch_daily_price_series("zzsyn.us")

    assert series.provider == "stooq"
    assert series.symbol == "zzsyn.us"
    assert len(series.bars) == 2
    assert str(series.bars[0].close) == "10.2"
    assert str(series.bars[1].close) == "10.7"
    assert series.bars[1].volume == 15000
    assert series.raw_bytes == VALID_CSV


def test_html_challenge_raises_provider_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    html = b"<!DOCTYPE html><html><body>Just a moment...</body></html>"
    monkeypatch.setattr(
        stooq, "fetch", lambda *args, **kwargs: _result(html, content_type="text/html")
    )

    with pytest.raises(ProviderBlockedError, match="HTML page"):
        stooq.fetch_daily_price_series("zzsyn.us")


def test_rate_limit_message_raises_provider_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"Exceeded the daily hits limit"
    monkeypatch.setattr(
        stooq, "fetch", lambda *args, **kwargs: _result(body, content_type="text/plain")
    )

    with pytest.raises(ProviderBlockedError, match="automation/rate limit"):
        stooq.fetch_daily_price_series("zzsyn.us")


def test_subscription_message_raises_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"Please subscribe to access this data"
    monkeypatch.setattr(
        stooq, "fetch", lambda *args, **kwargs: _result(body, content_type="text/plain")
    )

    with pytest.raises(ProviderConfigurationError):
        stooq.fetch_daily_price_series("zzsyn.us")


def test_no_data_marker_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stooq, "fetch", lambda *args, **kwargs: _result(b"N/D", content_type="text/plain")
    )

    with pytest.raises(ProviderResponseError, match="no data"):
        stooq.fetch_daily_price_series("zzsyn.us")


def test_malformed_row_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    bad_csv = b"Date,Open,High,Low,Close,Volume\nnot-a-date,10.0,10.5,9.8,10.2,12345\n"
    monkeypatch.setattr(stooq, "fetch", lambda *args, **kwargs: _result(bad_csv))

    with pytest.raises(ProviderResponseError, match="could not be parsed"):
        stooq.fetch_daily_price_series("zzsyn.us")


def test_unexpected_status_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stooq, "fetch", lambda *args, **kwargs: _result(b"", status_code=500))

    with pytest.raises(ProviderResponseError, match="HTTP 500"):
        stooq.fetch_daily_price_series("zzsyn.us")

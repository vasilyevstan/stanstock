from __future__ import annotations

import httpx
import pytest

from stanstock.data.providers import ecb
from stanstock.data.providers.exceptions import ProviderResponseError
from stanstock.data.providers.http import HttpFetchResult

VALID_CSV = (
    b"TIME_PERIOD,OBS_VALUE,CURRENCY,CURRENCY_DENOM\n"
    b"2026-09-01,1.0850,USD,EUR\n"
    b"2026-09-02,1.0872,USD,EUR\n"
)


def _result(
    content: bytes,
    *,
    status_code: int = 200,
    content_type: str = "text/csv",
    last_modified: str | None = "Wed, 02 Sep 2026 15:05:00 GMT",
) -> HttpFetchResult:
    headers = {"content-type": content_type}
    if last_modified:
        headers["last-modified"] = last_modified
    return HttpFetchResult(
        status_code=status_code,
        headers=httpx.Headers(headers),
        content=content,
        url=f"{ecb.BASE_URL}/D.USD.EUR.SP00.A",
    )


def test_parses_valid_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ecb, "fetch", lambda *args, **kwargs: _result(VALID_CSV))

    result = ecb.fetch_exr_csv("USD")

    assert len(result.observations) == 2
    first = result.observations[0]
    assert first.base_currency == "EUR"
    assert first.quote_currency == "USD"
    assert str(first.value) == "1.0850"
    assert first.published_at.hour == 15
    assert result.last_modified is not None
    assert result.last_modified.year == 2026


def test_build_series_key_format() -> None:
    assert ecb.build_series_key("USD") == "D.USD.EUR.SP00.A"
    assert ecb.build_series_key("GBP", frequency="D") == "D.GBP.EUR.SP00.A"


def test_currency_mismatch_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    mismatched_csv = b"TIME_PERIOD,OBS_VALUE,CURRENCY,CURRENCY_DENOM\n2026-09-01,1.0850,GBP,EUR\n"
    monkeypatch.setattr(ecb, "fetch", lambda *args, **kwargs: _result(mismatched_csv))

    with pytest.raises(ProviderResponseError, match="CURRENCY"):
        ecb.fetch_exr_csv("USD")


def test_empty_csv_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ecb, "fetch", lambda *args, **kwargs: _result(b""))

    with pytest.raises(ProviderResponseError, match="empty"):
        ecb.fetch_exr_csv("USD")


def test_missing_columns_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    bad_csv = b"FOO,BAR\n1,2\n"
    monkeypatch.setattr(ecb, "fetch", lambda *args, **kwargs: _result(bad_csv))

    with pytest.raises(ProviderResponseError, match="TIME_PERIOD"):
        ecb.fetch_exr_csv("USD")


def test_header_only_csv_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    header_only = b"TIME_PERIOD,OBS_VALUE,CURRENCY,CURRENCY_DENOM\n"
    monkeypatch.setattr(ecb, "fetch", lambda *args, **kwargs: _result(header_only))

    with pytest.raises(ProviderResponseError, match="no observation rows"):
        ecb.fetch_exr_csv("USD")


def test_404_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ecb, "fetch", lambda *args, **kwargs: _result(b"", status_code=404))

    with pytest.raises(ProviderResponseError, match="404"):
        ecb.fetch_exr_csv("USD")


def test_missing_last_modified_header_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ecb, "fetch", lambda *args, **kwargs: _result(VALID_CSV, last_modified=None)
    )

    result = ecb.fetch_exr_csv("USD")

    assert result.last_modified is None

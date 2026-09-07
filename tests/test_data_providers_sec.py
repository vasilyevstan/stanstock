from __future__ import annotations

import httpx
import pytest

from stanstock.data.providers import sec
from stanstock.data.providers.exceptions import (
    ProviderBlockedError,
    ProviderConfigurationError,
    ProviderResponseError,
)
from stanstock.data.providers.http import HttpFetchResult

SUBMISSIONS_JSON = b"""{
    "cik": 320193,
    "name": "Example Synthetic Corp",
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-26-000001", "0000320193-25-000099"],
            "form": ["10-K", "10-Q"],
            "filingDate": ["2026-01-05", "2025-10-01"],
            "acceptanceDateTime": ["2026-01-05T16:30:00.000Z", "2025-10-01T12:00:00.000Z"]
        }
    }
}"""

COMPANYFACTS_JSON = b"""{
    "cik": 320193,
    "entityName": "Example Synthetic Corp",
    "facts": {
        "us-gaap": {
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "val": 100, "accn": "0000320193-26-000001",
                            "fy": 2026, "filed": "2026-01-05"
                        },
                        {
                            "val": 90, "accn": "0000320193-25-000099",
                            "fy": 2025, "filed": "2025-10-01"
                        }
                    ]
                }
            }
        }
    }
}"""

TICKER_MAPPING_JSON = b"""{
    "fields": ["cik", "name", "ticker", "exchange"],
    "data": [[320193, "Apple Inc.", "AAPL", "Nasdaq"]]
}"""

SUBMISSIONS_HISTORY_JSON = b"""{
    "accessionNumber": ["0000320193-24-000001"],
    "form": ["10-K"],
    "filingDate": ["2024-01-05"],
    "acceptanceDateTime": ["2024-01-05T16:30:00.000Z"]
}"""


def _result(
    content: bytes, *, status_code: int = 200, content_type: str = "application/json"
) -> HttpFetchResult:
    return HttpFetchResult(
        status_code=status_code,
        headers=httpx.Headers({"content-type": content_type}),
        content=content,
        url="https://data.sec.gov/example",
    )


def test_build_user_agent_requires_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(ProviderConfigurationError, match="SEC_USER_AGENT"):
        sec.build_user_agent()


def test_build_user_agent_requires_contact_email() -> None:
    with pytest.raises(ProviderConfigurationError, match="contact email"):
        sec.build_user_agent("StanStockResearch/0.1")


def test_build_user_agent_accepts_valid_value() -> None:
    assert sec.build_user_agent("StanStockResearch/0.1 admin@example.com") == (
        "StanStockResearch/0.1 admin@example.com"
    )


def test_format_cik_pads_to_ten_digits() -> None:
    assert sec.format_cik(320193) == "0000320193"
    assert sec.format_cik("320193") == "0000320193"


def test_format_cik_rejects_non_numeric() -> None:
    with pytest.raises(ValueError):
        sec.format_cik("not-a-cik")


def test_fetch_submissions_extracts_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sec, "fetch", lambda *args, **kwargs: _result(SUBMISSIONS_JSON))

    payload = sec.fetch_submissions(320193, user_agent="StanStockResearch/0.1 admin@example.com")

    assert payload.provider == "sec"
    assert payload.subject == "0000320193"
    assert payload.content == SUBMISSIONS_JSON
    assert payload.metadata["entity_name"] == "Example Synthetic Corp"
    assert payload.metadata["recent_accession_numbers"][0] == "0000320193-26-000001"
    assert payload.metadata["recent_acceptance_datetimes"][0] == "2026-01-05T16:30:00.000Z"


def test_fetch_companyfacts_extracts_accessions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sec, "fetch", lambda *args, **kwargs: _result(COMPANYFACTS_JSON))

    payload = sec.fetch_companyfacts(320193, user_agent="StanStockResearch/0.1 admin@example.com")

    assert payload.metadata["concept_count"] == 1
    assert payload.metadata["accession_numbers"] == [
        "0000320193-25-000099",
        "0000320193-26-000001",
    ]
    assert payload.metadata["accession_filed_dates"] == {
        "0000320193-25-000099": "2025-10-01",
        "0000320193-26-000001": "2026-01-05",
    }


def test_fetch_ticker_exchange_mapping_validates_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sec, "fetch", lambda *args, **kwargs: _result(TICKER_MAPPING_JSON))

    payload = sec.fetch_ticker_exchange_mapping(
        user_agent="StanStockResearch/0.1 admin@example.com"
    )

    assert payload.subject == "company_tickers_exchange"
    assert payload.metadata == {
        "fields": ["cik", "name", "ticker", "exchange"],
        "row_count": 1,
    }


def test_fetch_submissions_history_rejects_unsafe_filename() -> None:
    with pytest.raises(ValueError, match="Invalid SEC submissions history filename"):
        sec.fetch_submissions_history(
            "../secrets.json",
            user_agent="StanStockResearch/0.1 admin@example.com",
        )


def test_fetch_submissions_history_extracts_row_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sec,
        "fetch",
        lambda *args, **kwargs: _result(SUBMISSIONS_HISTORY_JSON),
    )

    payload = sec.fetch_submissions_history(
        "CIK0000320193-submissions-001.json",
        user_agent="StanStockResearch/0.1 admin@example.com",
    )

    assert payload.metadata["row_count"] == 1
    assert payload.metadata["recent_accession_numbers"] == ["0000320193-24-000001"]


def test_403_raises_provider_blocked_as_environment_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sec, "fetch", lambda *args, **kwargs: _result(b"", status_code=403))

    with pytest.raises(ProviderBlockedError, match="403"):
        sec.fetch_submissions(320193, user_agent="StanStockResearch/0.1 admin@example.com")


def test_404_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sec, "fetch", lambda *args, **kwargs: _result(b"", status_code=404))

    with pytest.raises(ProviderResponseError, match="404"):
        sec.fetch_submissions(320193, user_agent="StanStockResearch/0.1 admin@example.com")


def test_non_json_response_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sec, "fetch", lambda *args, **kwargs: _result(b"<html></html>", content_type="text/html")
    )

    with pytest.raises(ProviderResponseError, match="not JSON"):
        sec.fetch_submissions(320193, user_agent="StanStockResearch/0.1 admin@example.com")


def test_malformed_json_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sec, "fetch", lambda *args, **kwargs: _result(b"{not json"))

    with pytest.raises(ProviderResponseError, match="malformed"):
        sec.fetch_submissions(320193, user_agent="StanStockResearch/0.1 admin@example.com")


def test_fetch_submissions_uses_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "StanStockResearch/0.1 admin@example.com")
    monkeypatch.setattr(sec, "fetch", lambda *args, **kwargs: _result(SUBMISSIONS_JSON))

    payload = sec.fetch_submissions(320193)

    assert payload.subject == "0000320193"

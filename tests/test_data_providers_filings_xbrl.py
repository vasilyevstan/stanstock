from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from stanstock.data.providers import filings_xbrl
from stanstock.data.providers.exceptions import ProviderResponseError
from stanstock.data.providers.http import HttpFetchResult

FILING_JSON_BODY = b'{"facts": {"example": "synthetic"}}'
FILING_CONTENT_SHA256 = hashlib.sha256(FILING_JSON_BODY).hexdigest()
FILING_PACKAGE_SHA256 = "a" * 64

INDEX_DOCUMENT = {
    "data": [
        {
            "id": "synthetic-filing-1",
            "type": "filing",
            "attributes": {
                "country": "NL",
                "period_end": "2025-12-31",
                "date_added": "2026-01-15 09:30:00.000000",
                "processed": "2026-01-15 10:00:00.000000",
                "sha256": FILING_PACKAGE_SHA256,
                "error_count": 0,
                "warning_count": 2,
                "json_url": "/synthetic-filing-1/report.json",
                "package_url": "https://filings.xbrl.org/synthetic-filing-1/report.zip",
                "viewer_url": "https://filings.xbrl.org/synthetic-filing-1/viewer",
            },
            "relationships": {"entity": {"data": {"id": "SYNTHETIC-LEI-000000000001"}}},
        }
    ]
}


def _index_result(document: object, *, status_code: int = 200) -> HttpFetchResult:
    return HttpFetchResult(
        status_code=status_code,
        headers=httpx.Headers({"content-type": "application/vnd.api+json"}),
        content=json.dumps(document).encode("utf-8"),
        url=filings_xbrl.FILINGS_API_URL,
    )


def _json_result(content: bytes, *, status_code: int = 200) -> HttpFetchResult:
    return HttpFetchResult(
        status_code=status_code,
        headers=httpx.Headers({"content-type": "application/json"}),
        content=content,
        url="https://filings.xbrl.org/synthetic-filing-1/report.json",
    )


def test_fetch_filings_page_parses_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        filings_xbrl, "fetch", lambda *args, **kwargs: _index_result(INDEX_DOCUMENT)
    )

    page = filings_xbrl.fetch_filings_page()

    assert len(page.records) == 1
    record = page.records[0]
    assert record.filing_id == "synthetic-filing-1"
    assert record.entity_lei == "SYNTHETIC-LEI-000000000001"
    assert record.country == "NL"
    assert record.sha256 == FILING_PACKAGE_SHA256
    assert record.error_count == 0
    assert record.warning_count == 2
    assert record.date_added is not None and record.date_added.tzinfo is not None
    assert record.raw["json_url"] == "/synthetic-filing-1/report.json"


def test_fetch_filings_page_rejects_non_jsonapi_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(filings_xbrl, "fetch", lambda *args, **kwargs: _index_result({"oops": 1}))

    with pytest.raises(ProviderResponseError, match="JSON:API"):
        filings_xbrl.fetch_filings_page()


def test_fetch_filing_xbrl_json_resolves_relative_url_and_hashes_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        filings_xbrl, "fetch", lambda *args, **kwargs: _index_result(INDEX_DOCUMENT)
    )
    page = filings_xbrl.fetch_filings_page()
    record = page.records[0]

    requested_urls: list[str] = []

    def fake_fetch(url: str, **kwargs: object) -> HttpFetchResult:
        requested_urls.append(url)
        return _json_result(FILING_JSON_BODY)

    monkeypatch.setattr(filings_xbrl, "fetch", fake_fetch)
    payload = filings_xbrl.fetch_filing_xbrl_json(record)

    assert payload.provider == "filings_xbrl_org"
    assert payload.subject == "synthetic-filing-1"
    assert payload.content == FILING_JSON_BODY
    assert requested_urls == ["https://filings.xbrl.org/synthetic-filing-1/report.json"]
    assert payload.metadata["index_package_sha256"] == FILING_PACKAGE_SHA256
    assert payload.metadata["content_sha256"] == FILING_CONTENT_SHA256
    assert payload.metadata["warning_count"] == 2
    assert payload.filed_at == record.date_added


def test_fetch_filing_xbrl_json_does_not_apply_package_hash_to_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        filings_xbrl, "fetch", lambda *args, **kwargs: _index_result(INDEX_DOCUMENT)
    )
    record = filings_xbrl.fetch_filings_page().records[0]

    monkeypatch.setattr(
        filings_xbrl, "fetch", lambda *args, **kwargs: _json_result(b"tampered content")
    )

    payload = filings_xbrl.fetch_filing_xbrl_json(record)

    assert payload.metadata["index_package_sha256"] == FILING_PACKAGE_SHA256
    assert payload.metadata["content_sha256"] == hashlib.sha256(b"tampered content").hexdigest()


def test_fetch_filing_xbrl_json_requires_json_url() -> None:
    record = filings_xbrl.FilingRecord(
        filing_id="no-url",
        entity_lei=None,
        country=None,
        period_end=None,
        date_added=None,
        processed=None,
        sha256=None,
        error_count=None,
        warning_count=None,
        json_url=None,
        package_url=None,
        viewer_url=None,
    )

    with pytest.raises(ProviderResponseError, match="no json_url"):
        filings_xbrl.fetch_filing_xbrl_json(record)


def test_malformed_index_json_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        filings_xbrl,
        "fetch",
        lambda *args, **kwargs: HttpFetchResult(
            status_code=200,
            headers=httpx.Headers({"content-type": "application/vnd.api+json"}),
            content=b"{not-json",
            url=filings_xbrl.FILINGS_API_URL,
        ),
    )

    with pytest.raises(ProviderResponseError, match="malformed"):
        filings_xbrl.fetch_filings_page()

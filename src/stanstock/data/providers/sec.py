"""SEC EDGAR ``data.sec.gov`` submissions and company-facts client.

SEC requires every automated request to identify itself with a compliant
``User-Agent`` describing the application and a contact method (its fair
access policy: https://www.sec.gov/os/webmaster-faq#developers). This module
refuses to make a request without one; it never falls back to an
unidentified default.

Endpoints used (see https://www.sec.gov/edgar/sec-api-documentation):

- ``https://data.sec.gov/submissions/CIK##########.json``
- ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json``

Responses are preserved as raw bytes for an immutable `DataAsset`; this
module only extracts enough top-level fields (accession numbers, filing and
acceptance timestamps, entity identity) to support later point-in-time
normalization elsewhere. It does not perform full per-concept extraction.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime

from stanstock.data.providers.contracts import FundamentalSourcePayload
from stanstock.data.providers.exceptions import (
    ProviderBlockedError,
    ProviderConfigurationError,
    ProviderResponseError,
)
from stanstock.data.providers.http import HttpFetchResult, fetch

PROVIDER = "sec"
TERMS_URL = "https://www.sec.gov/about/developer-resources"
USAGE_SCOPE = "public_edgar_api_private_research"
TICKER_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_HISTORY_URL = "https://data.sec.gov/submissions/{filename}"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_RECENT_FIELD_LIMIT = 25
_ACCESSION_LIMIT = 50
_SUBMISSIONS_HISTORY_PATTERN = re.compile(r"^CIK\d{10}-submissions-\d{3}\.json$")


def build_user_agent(explicit: str | None = None) -> str:
    """Return a compliant SEC User-Agent from ``explicit`` or the
    ``SEC_USER_AGENT`` environment variable. Raises
    :class:`ProviderConfigurationError` if none is configured, or if it
    lacks a contact email, so a missing/invalid identifier fails loudly
    instead of sending SEC an anonymous-looking request."""
    user_agent = explicit if explicit is not None else os.environ.get("SEC_USER_AGENT", "")
    user_agent = user_agent.strip()
    if not user_agent:
        raise ProviderConfigurationError(
            "SEC_USER_AGENT is required by SEC's fair access policy and is "
            "not configured. Set it to an identifying string with a "
            "contact method, e.g. 'StanStockResearch/0.1 admin@example.com'."
        )
    if "@" not in user_agent:
        raise ProviderConfigurationError(
            "SEC_USER_AGENT must include a contact email per SEC's fair "
            "access policy (https://www.sec.gov/os/webmaster-faq#developers)."
        )
    return user_agent


def format_cik(cik: str | int) -> str:
    digits = "".join(char for char in str(cik) if char.isdigit())
    if not digits:
        raise ValueError(f"Invalid CIK: {cik!r}")
    return digits.zfill(10)


def fetch_submissions(
    cik: str | int,
    *,
    user_agent: str | None = None,
) -> FundamentalSourcePayload:
    """Fetch ``submissions/CIK##########.json`` and preserve its bytes."""
    ua = build_user_agent(user_agent)
    subject = format_cik(cik)
    url = SUBMISSIONS_URL.format(cik=subject)
    result = fetch(url, user_agent=ua)
    _raise_for_sec_status(result, subject=subject)
    metadata = _extract_submissions_metadata(result.content, subject=subject)
    return FundamentalSourcePayload(
        provider=PROVIDER,
        subject=subject,
        content=result.content,
        content_type=result.content_type or "application/json",
        retrieved_at=datetime.now(tz=UTC),
        source_url=result.url,
        metadata=metadata,
    )


def fetch_ticker_exchange_mapping(
    *,
    user_agent: str | None = None,
) -> FundamentalSourcePayload:
    """Fetch the SEC's current ticker/exchange/CIK mapping."""
    ua = build_user_agent(user_agent)
    result = fetch(TICKER_EXCHANGE_URL, user_agent=ua)
    _raise_for_sec_status(result, subject="company_tickers_exchange")
    metadata = _extract_ticker_exchange_metadata(result.content)
    return FundamentalSourcePayload(
        provider=PROVIDER,
        subject="company_tickers_exchange",
        content=result.content,
        content_type=result.content_type or "application/json",
        retrieved_at=datetime.now(tz=UTC),
        source_url=result.url,
        metadata=metadata,
    )


def fetch_submissions_history(
    filename: str,
    *,
    user_agent: str | None = None,
) -> FundamentalSourcePayload:
    """Fetch one historical submissions file referenced by ``filings.files``."""
    normalized = filename.strip()
    if not _SUBMISSIONS_HISTORY_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid SEC submissions history filename: {filename!r}")
    ua = build_user_agent(user_agent)
    result = fetch(SUBMISSIONS_HISTORY_URL.format(filename=normalized), user_agent=ua)
    _raise_for_sec_status(result, subject=normalized)
    metadata = _extract_submission_rows_metadata(result.content, subject=normalized)
    return FundamentalSourcePayload(
        provider=PROVIDER,
        subject=normalized,
        content=result.content,
        content_type=result.content_type or "application/json",
        retrieved_at=datetime.now(tz=UTC),
        source_url=result.url,
        metadata=metadata,
    )


def fetch_companyfacts(
    cik: str | int,
    *,
    user_agent: str | None = None,
) -> FundamentalSourcePayload:
    """Fetch ``api/xbrl/companyfacts/CIK##########.json`` and preserve its
    bytes."""
    ua = build_user_agent(user_agent)
    subject = format_cik(cik)
    url = COMPANYFACTS_URL.format(cik=subject)
    result = fetch(url, user_agent=ua)
    _raise_for_sec_status(result, subject=subject)
    metadata = _extract_companyfacts_metadata(result.content, subject=subject)
    return FundamentalSourcePayload(
        provider=PROVIDER,
        subject=subject,
        content=result.content,
        content_type=result.content_type or "application/json",
        retrieved_at=datetime.now(tz=UTC),
        source_url=result.url,
        metadata=metadata,
    )


def _raise_for_sec_status(result: HttpFetchResult, *, subject: str) -> None:
    if result.status_code == 403:
        raise ProviderBlockedError(
            f"SEC returned HTTP 403 for {subject}. In StanStock's own "
            "spike this reproduced even with a compliant SEC_USER_AGENT and "
            "was treated as environment/network egress blocking rather than "
            "SEC declaring the endpoint off-limits (see docs/source-spike.md). "
            "Re-verify from an unblocked network before concluding SEC "
            "access is unavailable to this deployment."
        )
    if result.status_code == 404:
        raise ProviderResponseError(f"SEC has no record for {subject} (HTTP 404)")
    if result.status_code >= 400:
        raise ProviderResponseError(
            f"SEC returned unexpected HTTP {result.status_code} for {subject}"
        )
    content_type = result.content_type.lower()
    stripped = result.content.strip()
    if "json" not in content_type and not stripped.startswith(b"{"):
        raise ProviderResponseError(
            f"SEC response for {subject} was not JSON (content-type={content_type!r})"
        )


def _load_json(payload: bytes, *, subject: str, kind: str) -> dict[str, object]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(
            f"SEC {kind} JSON for CIK {subject} was malformed: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ProviderResponseError(f"SEC {kind} JSON for CIK {subject} was not an object")
    return data


def _extract_submissions_metadata(payload: bytes, *, subject: str) -> dict[str, object]:
    data = _load_json(payload, subject=subject, kind="submissions")
    filings = data.get("filings")
    recent = filings.get("recent") if isinstance(filings, dict) else None
    recent = recent if isinstance(recent, dict) else {}
    files = filings.get("files") if isinstance(filings, dict) else None
    files = files if isinstance(files, list) else []
    return {
        "entity_name": data.get("name"),
        "cik": data.get("cik"),
        "sic": data.get("sic"),
        "sic_description": data.get("sicDescription"),
        "tickers": data.get("tickers") if isinstance(data.get("tickers"), list) else [],
        "exchanges": data.get("exchanges") if isinstance(data.get("exchanges"), list) else [],
        "historical_files": [
            item
            for item in files
            if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name")
        ],
        "recent_accession_numbers": list(recent.get("accessionNumber", []))[:_RECENT_FIELD_LIMIT],
        "recent_forms": list(recent.get("form", []))[:_RECENT_FIELD_LIMIT],
        "recent_filing_dates": list(recent.get("filingDate", []))[:_RECENT_FIELD_LIMIT],
        "recent_acceptance_datetimes": list(recent.get("acceptanceDateTime", []))[
            :_RECENT_FIELD_LIMIT
        ],
    }


def _extract_submission_rows_metadata(payload: bytes, *, subject: str) -> dict[str, object]:
    data = _load_json(payload, subject=subject, kind="submissions history")
    accessions = data.get("accessionNumber")
    forms = data.get("form")
    filing_dates = data.get("filingDate")
    acceptance_datetimes = data.get("acceptanceDateTime")
    return {
        "row_count": len(accessions) if isinstance(accessions, list) else 0,
        "recent_accession_numbers": (
            accessions[:_RECENT_FIELD_LIMIT] if isinstance(accessions, list) else []
        ),
        "recent_forms": forms[:_RECENT_FIELD_LIMIT] if isinstance(forms, list) else [],
        "recent_filing_dates": (
            filing_dates[:_RECENT_FIELD_LIMIT] if isinstance(filing_dates, list) else []
        ),
        "recent_acceptance_datetimes": (
            acceptance_datetimes[:_RECENT_FIELD_LIMIT]
            if isinstance(acceptance_datetimes, list)
            else []
        ),
    }


def _extract_ticker_exchange_metadata(payload: bytes) -> dict[str, object]:
    data = _load_json(payload, subject="company_tickers_exchange", kind="ticker mapping")
    fields = data.get("fields")
    rows = data.get("data")
    if not isinstance(fields, list) or not all(isinstance(item, str) for item in fields):
        raise ProviderResponseError("SEC ticker mapping fields were missing or invalid")
    if not isinstance(rows, list):
        raise ProviderResponseError("SEC ticker mapping data rows were missing or invalid")
    required = {"cik", "name", "ticker", "exchange"}
    if not required.issubset(set(fields)):
        raise ProviderResponseError(
            f"SEC ticker mapping fields did not contain {sorted(required)!r}"
        )
    return {"fields": fields, "row_count": len(rows)}


def _extract_companyfacts_metadata(payload: bytes, *, subject: str) -> dict[str, object]:
    data = _load_json(payload, subject=subject, kind="companyfacts")
    facts = data.get("facts")
    facts = facts if isinstance(facts, dict) else {}
    accessions: set[str] = set()
    accession_filed_dates: dict[str, str] = {}
    concept_count = 0
    for taxonomy in facts.values():
        if not isinstance(taxonomy, dict):
            continue
        for concept in taxonomy.values():
            concept_count += 1
            units = concept.get("units") if isinstance(concept, dict) else None
            if not isinstance(units, dict):
                continue
            for observations in units.values():
                if not isinstance(observations, list):
                    continue
                for observation in observations:
                    if isinstance(observation, dict):
                        accession = observation.get("accn")
                        if isinstance(accession, str):
                            accessions.add(accession)
                            filed = observation.get("filed")
                            if isinstance(filed, str) and filed:
                                accession_filed_dates[accession] = filed
    return {
        "entity_name": data.get("entityName"),
        "cik": data.get("cik"),
        "concept_count": concept_count,
        "accession_numbers": sorted(accessions)[:_ACCESSION_LIMIT],
        # Per-observation "filed" dates from companyfacts, keyed by
        # accession number, so callers can recover the filing vintage for
        # each accession without re-scanning the full concept tree later.
        "accession_filed_dates": dict(sorted(accession_filed_dates.items())[:_ACCESSION_LIMIT]),
    }

"""``filings.xbrl.org`` filings index API and linked xBRL-JSON downloads.

``filings.xbrl.org`` (see https://filings.xbrl.org/docs/api) exposes a
JSON:API-shaped index of European (ESEF) filings at
``https://filings.xbrl.org/api/filings``. Each entry carries repository
bookkeeping fields StanStock must retain for point-in-time correctness even
though they are not the authority's own filing timestamp: ``date_added``
(when the repository first saw the filing), ``processed`` (when it finished
indexing), ``sha256`` (content hash of the package), and
``error_count``/``warning_count`` from validation. Per
``docs/point-in-time.md``, StanStock uses ``date_added``/``processed`` as a
conservative availability floor and never backdates availability to the
financial period end.

filings.xbrl.org itself documents incomplete country coverage (see
``docs/source-spike.md``); this module does not paper over that, it simply
returns whatever the index has.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from urllib.parse import urljoin, urlsplit

from stanstock.data.providers.contracts import FundamentalSourcePayload
from stanstock.data.providers.exceptions import ProviderResponseError
from stanstock.data.providers.http import HttpFetchResult, fetch

PROVIDER = "filings_xbrl_org"
FILINGS_API_URL = "https://filings.xbrl.org/api/filings"
FILINGS_ORIGIN = "https://filings.xbrl.org/"
DEFAULT_USER_AGENT = (
    "StanStockResearch/0.1 (+https://github.com/vasilyevstan/stanstock; "
    "non-commercial research spike; contact via repository issues)"
)


@dataclass(frozen=True, slots=True)
class FilingRecord:
    """One entry from the filings.xbrl.org index, with repository
    provenance fields kept both parsed (best-effort) and as the original
    ``raw`` attributes mapping so nothing is silently dropped."""

    filing_id: str
    entity_lei: str | None
    country: str | None
    period_end: date | None
    date_added: datetime | None
    processed: datetime | None
    sha256: str | None
    error_count: int | None
    warning_count: int | None
    json_url: str | None
    package_url: str | None
    viewer_url: str | None
    raw: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FilingsPage:
    records: tuple[FilingRecord, ...]
    raw_bytes: bytes
    retrieved_at: datetime
    source_url: str


def fetch_filings_page(
    *,
    page_size: int = 10,
    page_number: int = 1,
    country: str | None = None,
    sort: str | None = "-processed",
    user_agent: str = DEFAULT_USER_AGENT,
) -> FilingsPage:
    """Fetch one page of the filings index, newest-processed first by
    default. Raises :class:`ProviderResponseError` if the payload is not the
    expected JSON:API document shape."""
    params: dict[str, object] = {
        "page[size]": page_size,
        "page[number]": page_number,
    }
    if country:
        params["filter[country]"] = country
    if sort:
        params["sort"] = sort
    result = fetch(FILINGS_API_URL, params=params, user_agent=user_agent)
    _raise_for_status(result, context="filings index")
    records = _parse_filings_document(result.content)
    return FilingsPage(
        records=records,
        raw_bytes=result.content,
        retrieved_at=datetime.now(tz=UTC),
        source_url=result.url,
    )


def fetch_filing_xbrl_json(
    record: FilingRecord,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
) -> FundamentalSourcePayload:
    """Download the linked xBRL-JSON payload for ``record`` and preserve its
    bytes, retaining ``date_added``/``processed``/``sha256``/error/warning
    metadata from the index entry alongside it."""
    if not record.json_url:
        raise ProviderResponseError(
            f"Filing {record.filing_id} has no json_url to download from the index"
        )
    download_url = urljoin(FILINGS_ORIGIN, record.json_url)
    parsed_url = urlsplit(download_url)
    if parsed_url.scheme != "https" or parsed_url.netloc != "filings.xbrl.org":
        raise ProviderResponseError(f"Filing {record.filing_id} has an unexpected xBRL-JSON host")
    result = fetch(download_url, user_agent=user_agent)
    _raise_for_status(result, context=f"filing {record.filing_id} xBRL-JSON")
    content_type = result.content_type.lower()
    if "json" not in content_type and not result.content.strip().startswith(b"{"):
        raise ProviderResponseError(
            f"Filing {record.filing_id} xBRL-JSON response was not JSON "
            f"(content-type={content_type!r})"
        )
    content_sha256 = hashlib.sha256(result.content).hexdigest()
    return FundamentalSourcePayload(
        provider=PROVIDER,
        subject=record.filing_id,
        content=result.content,
        content_type=result.content_type or "application/json",
        retrieved_at=datetime.now(tz=UTC),
        source_url=result.url,
        filed_at=record.date_added,
        metadata={
            "entity_lei": record.entity_lei,
            "country": record.country,
            "period_end": record.period_end.isoformat() if record.period_end else None,
            "date_added": record.date_added.isoformat() if record.date_added else None,
            "processed": record.processed.isoformat() if record.processed else None,
            "index_package_sha256": record.sha256,
            "content_sha256": content_sha256,
            "error_count": record.error_count,
            "warning_count": record.warning_count,
        },
    )


def _raise_for_status(result: HttpFetchResult, *, context: str) -> None:
    if result.status_code >= 400:
        raise ProviderResponseError(
            f"filings.xbrl.org returned unexpected HTTP {result.status_code} for {context}"
        )


def _parse_filings_document(payload: bytes) -> tuple[FilingRecord, ...]:
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(f"filings.xbrl.org index JSON was malformed: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("data"), list):
        raise ProviderResponseError(
            "filings.xbrl.org index response did not have the expected JSON:API 'data' list"
        )
    return tuple(_parse_filing_entry(entry) for entry in document["data"])


def _parse_filing_entry(entry: object) -> FilingRecord:
    if not isinstance(entry, dict) or "id" not in entry:
        raise ProviderResponseError(f"filings.xbrl.org index entry missing 'id': {entry!r}")
    attributes = entry.get("attributes")
    attributes = attributes if isinstance(attributes, dict) else {}
    relationships = entry.get("relationships")
    relationships = relationships if isinstance(relationships, dict) else {}
    entity = relationships.get("entity")
    entity_data = entity.get("data") if isinstance(entity, dict) else None
    entity_lei = entity_data.get("id") if isinstance(entity_data, dict) else None
    return FilingRecord(
        filing_id=str(entry["id"]),
        entity_lei=entity_lei if isinstance(entity_lei, str) else None,
        country=_as_optional_str(attributes.get("country")),
        period_end=_parse_date(attributes.get("period_end")),
        date_added=_parse_timestamp(attributes.get("date_added")),
        processed=_parse_timestamp(attributes.get("processed")),
        sha256=_as_optional_str(attributes.get("sha256")),
        error_count=_as_optional_int(attributes.get("error_count")),
        warning_count=_as_optional_int(attributes.get("warning_count")),
        json_url=_as_optional_str(attributes.get("json_url")),
        package_url=_as_optional_str(attributes.get("package_url")),
        viewer_url=_as_optional_str(attributes.get("viewer_url")),
        raw=dict(attributes),
    )


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace(" ", "T")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

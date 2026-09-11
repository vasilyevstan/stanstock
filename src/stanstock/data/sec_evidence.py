"""SEC evidence contract leaf: shared identity constants and strict parsers.

Stdlib-only, so both the writer (`data.sec_ingestion`) and the reader
(`data.sec_refresh_validation`) can depend on it without provider/domain
coupling, hand-synced literals, or an import cycle.
"""

from __future__ import annotations

import json
import re

MAPPING_KIND = "sec_ticker_mapping"
MAPPING_SUBJECT = "company_tickers_exchange"
SUBMISSIONS_KIND = "sec_submissions"
SUBMISSIONS_HISTORY_KIND = "sec_submissions_history"
COMPANYFACTS_KIND = "sec_companyfacts"
#: `DataAsset.metadata` key naming a history asset's own source filename.
HISTORY_FILENAME_METADATA_KEY = "filename"

_SAFE_HISTORY_FILENAME = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_.-]*\.json\Z")


class SecEvidencePayloadError(ValueError):
    """A SEC evidence payload does not conform to the strict shape both the
    writer and reader require; its message may embed provider payload
    content (never a filesystem path) -- callers still normalize it into a
    stable, path-free `RefreshVerificationError`."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise SecEvidencePayloadError(f"Duplicate JSON key: {key!r}")
        seen[key] = value
    return seen


def is_safe_history_filename(name: str) -> bool:
    """True if `name` is a plain `.json` basename with no path traversal."""
    if "/" in name or "\\" in name or name in (".", ".."):
        return False
    if any(ord(ch) < 0x20 for ch in name):
        return False
    return bool(_SAFE_HISTORY_FILENAME.match(name))


def historical_submission_filenames(payload: bytes) -> tuple[str, ...]:
    """Exact, source-ordered `filings.files[].name` entries from a raw SEC
    submissions payload.

    Fails closed (`SecEvidencePayloadError`) on invalid UTF-8/JSON, a
    duplicate JSON key anywhere, a missing/non-object `filings`, a
    missing/non-list `filings.files` (an empty list is valid), or a
    non-object/blank/unsafe/duplicated `files` row -- never silently
    skips, deduplicates, or substitutes "no history" for a malformed one.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SecEvidencePayloadError("SEC submissions payload is not valid UTF-8") from exc
    try:
        data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise SecEvidencePayloadError(f"SEC submissions JSON was malformed: {exc}") from exc
    if not isinstance(data, dict):
        raise SecEvidencePayloadError("SEC submissions JSON was not an object")
    filings = data.get("filings")
    if not isinstance(filings, dict):
        raise SecEvidencePayloadError("SEC submissions JSON has no 'filings' object")
    files = filings.get("files")
    if not isinstance(files, list):
        raise SecEvidencePayloadError("SEC submissions JSON 'filings.files' is not a list")
    names: list[str] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise SecEvidencePayloadError("SEC submissions JSON has a non-object files row")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise SecEvidencePayloadError("SEC submissions JSON files row has a blank name")
        if not is_safe_history_filename(name):
            raise SecEvidencePayloadError(f"SEC submissions JSON files row is unsafe: {name!r}")
        if name in seen:
            raise SecEvidencePayloadError(
                f"SEC submissions JSON files has a duplicate name: {name!r}"
            )
        seen.add(name)
        names.append(name)
    return tuple(names)

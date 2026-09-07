from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal


def build_period_identity(
    *,
    period_type: str,
    period_start: date | None,
    period_end: date,
    fiscal_period: str = "",
    frame: str = "",
) -> str:
    start = period_start.isoformat() if period_start is not None else ""
    parts = [period_type, start, period_end.isoformat()]
    if period_type == "unclassified":
        parts.extend((fiscal_period.strip().upper(), frame.strip().upper()))
    return ":".join(parts)


def build_observation_hash(
    *,
    taxonomy: str,
    source_concept: str,
    value: Decimal,
    unit: str,
    currency: str,
    period_identity: str,
    fiscal_year: int | None,
    fiscal_period: str,
    accession: str,
    filing_form: str,
    filing_date: date | None,
    acceptance_at: datetime | None,
    frame: str,
) -> str:
    payload = {
        "taxonomy": taxonomy,
        "source_concept": source_concept,
        "value": str(value),
        "unit": unit,
        "currency": currency,
        "period_identity": period_identity,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "accession": accession,
        "filing_form": filing_form,
        "filing_date": filing_date.isoformat() if filing_date is not None else None,
        "acceptance_at": acceptance_at.isoformat() if acceptance_at is not None else None,
        "frame": frame,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

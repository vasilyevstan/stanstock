"""Small, explicit contracts returned by provider clients.

These dataclasses are the only shape provider modules hand back to callers.
They deliberately mirror the point-in-time fields StanStock needs
(``docs/point-in-time.md``): what the value describes, when the source says
it was published, and the raw bytes actually retrieved so a `DataAsset`
manifest can be written before any normalization happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class PriceBar:
    """One OHLCV observation for a single trading session."""

    trade_date: date
    open: Decimal | None
    high: Decimal | None
    low: Decimal | None
    close: Decimal
    volume: int | None


@dataclass(frozen=True, slots=True)
class PriceSeries:
    """A daily price series as returned by a provider, plus provenance."""

    provider: str
    symbol: str
    currency: str | None
    bars: tuple[PriceBar, ...]
    retrieved_at: datetime
    source_url: str
    raw_bytes: bytes


@dataclass(frozen=True, slots=True)
class FundamentalSourcePayload:
    """A raw fundamentals payload (SEC or ESEF/xBRL-JSON) plus enough
    provenance metadata for later point-in-time normalization. The full
    response body is preserved unmodified so a `DataAsset` manifest can be
    written before any concept-level extraction happens."""

    provider: str
    subject: str
    content: bytes
    content_type: str
    retrieved_at: datetime
    source_url: str
    accession: str | None = None
    accepted_at: datetime | None = None
    filed_at: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FxObservation:
    """One published FX reference rate observation."""

    base_currency: str
    quote_currency: str
    observation_date: date
    value: Decimal
    published_at: datetime
    source_url: str

"""Stooq daily price CSV client.

Stooq (https://stooq.com) publishes a "download" link under each quote page
that returns a daily-bar CSV, e.g.::

    https://stooq.com/q/d/l/?s=aapl.us&i=d

There is no documented, versioned public API and no official terms page
describing automated/bulk retrieval rights (see ``docs/source-spike.md``).
In the environment this project was developed in, requesting that endpoint
without a browser returns an HTML/JavaScript verification page instead of
CSV. StanStock treats that as an explicit, permanent stop for this path: it
never attempts to solve or bypass a bot/browser challenge, and it never
falls back to scraping the HTML quote page instead of the CSV endpoint.

This module only parses CSV; if Stooq's response is not CSV, or looks like a
rate-limit/subscription message, it raises instead of guessing.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation

from stanstock.data.providers.contracts import PriceBar, PriceSeries
from stanstock.data.providers.exceptions import (
    ProviderBlockedError,
    ProviderConfigurationError,
    ProviderResponseError,
)
from stanstock.data.providers.http import DEFAULT_TIMEOUT, HttpFetchResult, fetch

PROVIDER = "stooq"
BASE_URL = "https://stooq.com/q/d/l/"
DEFAULT_USER_AGENT = (
    "StanStockResearch/0.1 (+https://github.com/vasilyevstan/stanstock; "
    "non-commercial research spike; contact via repository issues)"
)
_EXPECTED_HEADER = ["Date", "Open", "High", "Low", "Close", "Volume"]


def fetch_daily_price_series(
    symbol: str,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
) -> PriceSeries:
    """Fetch and parse Stooq's daily CSV for ``symbol`` (for example
    ``"aapl.us"``). Raises :class:`ProviderBlockedError` if the response is
    an HTML/JS challenge page or a rate-limit message, and
    :class:`ProviderResponseError` for any other unusable payload."""
    result = fetch(
        BASE_URL,
        params={"s": symbol, "i": "d"},
        user_agent=user_agent,
        timeout=DEFAULT_TIMEOUT,
    )
    _raise_for_non_csv(result, symbol=symbol)
    bars = _parse_daily_csv(result.content, symbol=symbol)
    return PriceSeries(
        provider=PROVIDER,
        symbol=symbol,
        currency=None,
        bars=bars,
        retrieved_at=datetime.now(tz=UTC),
        source_url=result.url,
        raw_bytes=result.content,
    )


def _raise_for_non_csv(result: HttpFetchResult, *, symbol: str) -> None:
    if result.status_code >= 400:
        raise ProviderResponseError(
            f"Stooq returned unexpected HTTP {result.status_code} for symbol {symbol!r}"
        )
    content_type = result.content_type.lower()
    stripped = result.content.strip()
    looks_like_html = (
        "text/html" in content_type
        or stripped[:15].lower().startswith(b"<!doctype html")
        or stripped[:5].lower().startswith(b"<html")
    )
    if looks_like_html:
        raise ProviderBlockedError(
            "Stooq returned an HTML page instead of a CSV download for symbol "
            f"{symbol!r}. This matches the documented browser/JavaScript "
            "verification challenge Stooq serves to unattended clients in "
            "some environments; StanStock will not attempt to solve or "
            "bypass it, and will not scrape the HTML page as a substitute."
        )
    lowered = stripped.lower()
    if b"exceeded" in lowered and b"hit" in lowered:
        raise ProviderBlockedError(
            f"Stooq reported an automation/rate limit for symbol {symbol!r}: {stripped[:200]!r}"
        )
    if b"subscri" in lowered or b"premium" in lowered:
        raise ProviderConfigurationError(
            "Stooq indicated this request requires a paid subscription or "
            "account; StanStock has no such credential configured and will "
            "not attempt to work around that requirement."
        )


def _parse_daily_csv(payload: bytes, *, symbol: str) -> tuple[PriceBar, ...]:
    text = payload.decode("utf-8", errors="replace").strip()
    if not text or text.upper() == "N/D":
        raise ProviderResponseError(f"Stooq has no data for symbol {symbol!r}")
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ProviderResponseError(f"Stooq CSV for {symbol!r} was empty") from exc
    if [column.strip() for column in header] != _EXPECTED_HEADER:
        raise ProviderResponseError(
            f"Stooq CSV for {symbol!r} had an unexpected header: {header!r}"
        )
    bars: list[PriceBar] = []
    for row in reader:
        if not row:
            continue
        if len(row) != len(_EXPECTED_HEADER):
            raise ProviderResponseError(
                f"Stooq CSV row for {symbol!r} had {len(row)} columns: {row!r}"
            )
        raw_date, raw_open, raw_high, raw_low, raw_close, raw_volume = row
        try:
            trade_date = date.fromisoformat(raw_date.strip())
            close = Decimal(raw_close)
        except (ValueError, InvalidOperation) as exc:
            raise ProviderResponseError(
                f"Stooq CSV row for {symbol!r} could not be parsed: {row!r}"
            ) from exc
        bars.append(
            PriceBar(
                trade_date=trade_date,
                open=_optional_decimal(raw_open),
                high=_optional_decimal(raw_high),
                low=_optional_decimal(raw_low),
                close=close,
                volume=_optional_int(raw_volume),
            )
        )
    if not bars:
        raise ProviderResponseError(f"Stooq CSV for {symbol!r} had a header but no rows")
    return tuple(bars)


def _optional_decimal(raw: str) -> Decimal | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _optional_int(raw: str) -> int | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None

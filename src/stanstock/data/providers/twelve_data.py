"""Official Twelve Data REST client for US daily prices and stock reference data.

StanStock uses only documented JSON endpoints and sends the API key in the
``Authorization`` header so credentials never appear in asset URLs, logs, or
exception text. Basic access is restricted to one explicitly licensed
personal, non-commercial user; other plans require the owner to confirm an
account or agreement with internal-display rights. Redistribution and public
display remain out of scope. See ``docs/source-spike.md`` for the reviewed
terms, coverage, and quota boundary.

Daily prices are requested with ``adjust=splits`` explicitly. They are
therefore split-adjusted *price* observations, not dividend-adjusted total
returns. Every response is returned with its original bytes so ingestion can
persist the provider payload before normalized Parquet data is used by
research code.
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from stanstock.data.provider_credentials import read_twelve_data_api_key
from stanstock.data.providers.contracts import (
    PriceBar,
    PriceSeries,
    StockCatalog,
    StockReference,
)
from stanstock.data.providers.exceptions import (
    ProviderBlockedError,
    ProviderConfigurationError,
    ProviderDataError,
    ProviderQuotaError,
    ProviderResponseError,
)
from stanstock.data.providers.http import DEFAULT_TIMEOUT, HttpFetchResult, fetch

PROVIDER = "twelve_data"
TIME_SERIES_URL = "https://api.twelvedata.com/time_series"
STOCKS_URL = "https://api.twelvedata.com/stocks"
TERMS_URL = "https://twelvedata.com/terms"
PRICING_URL = "https://twelvedata.com/pricing"
API_KEY_ENV = "TWELVE_DATA_API_KEY"
DEFAULT_USER_AGENT = (
    "StanStockResearch/0.1 (+https://github.com/vasilyevstan/stanstock; private personal research)"
)
MAX_OUTPUT_SIZE = 5_000
SUPPORTED_ADJUSTMENTS = frozenset({"all", "splits", "dividends", "none"})


def resolve_api_key(
    explicit: str | None = None,
    *,
    allow_demo: bool = False,
) -> str:
    """Return a configured API key without ever including it in an error."""
    if explicit is not None:
        api_key = explicit
    else:
        api_key = os.environ.get(API_KEY_ENV, "") or read_twelve_data_api_key() or ""
    api_key = api_key.strip()
    if not api_key:
        raise ProviderConfigurationError(
            f"{API_KEY_ENV} or a local macOS Keychain credential is required "
            "for Twelve Data API access."
        )
    if api_key.casefold() == "demo" and not allow_demo:
        raise ProviderConfigurationError(
            "Twelve Data's demo key is limited to trial symbols and cannot "
            "enable StanStock's live US universe. Configure a personal API key."
        )
    return api_key


def fetch_daily_price_series(
    symbol: str,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    outputsize: int | None = None,
    adjustment: str = "splits",
    api_key: str | None = None,
    allow_demo: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> PriceSeries:
    """Fetch one split-adjusted daily OHLCV series through the official API."""
    normalized_symbol = _normalize_symbol(symbol)
    key = resolve_api_key(api_key, allow_demo=allow_demo)
    if adjustment not in SUPPORTED_ADJUSTMENTS:
        raise ValueError(
            f"Unsupported Twelve Data adjustment {adjustment!r}; expected one of "
            f"{sorted(SUPPORTED_ADJUSTMENTS)!r}"
        )
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    if outputsize is not None and not 1 <= outputsize <= MAX_OUTPUT_SIZE:
        raise ValueError(f"outputsize must be between 1 and {MAX_OUTPUT_SIZE}")

    params: dict[str, Any] = {
        "symbol": normalized_symbol,
        "interval": "1day",
        "format": "JSON",
        "order": "ASC",
        "adjust": adjustment,
    }
    if start_date is not None:
        params["start_date"] = start_date.isoformat()
    if end_date is not None:
        # Twelve Data treats a date-only end boundary as exclusive for daily data.
        params["end_date"] = (end_date + timedelta(days=1)).isoformat()
    if outputsize is not None:
        params["outputsize"] = outputsize

    result = fetch(
        TIME_SERIES_URL,
        params=params,
        headers=_auth_headers(key),
        user_agent=user_agent,
        timeout=DEFAULT_TIMEOUT,
    )
    payload = _load_response(result, context=f"daily prices for {normalized_symbol}")
    try:
        meta = _required_mapping(payload, "meta", context=normalized_symbol)
        returned_symbol = _required_text(meta, "symbol", context=normalized_symbol).upper()
        if returned_symbol != normalized_symbol:
            raise ProviderResponseError(
                f"Twelve Data returned symbol {returned_symbol!r} for "
                f"requested symbol {normalized_symbol!r}"
            )
        interval = _required_text(meta, "interval", context=normalized_symbol)
        if interval != "1day":
            raise ProviderResponseError(
                f"Twelve Data returned interval {interval!r} for daily symbol {normalized_symbol!r}"
            )

        raw_values = payload.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ProviderResponseError(
                f"Twelve Data returned no daily values for symbol {normalized_symbol!r}"
            )
        bars = _parse_bars(
            raw_values,
            symbol=normalized_symbol,
            start_date=start_date,
            end_date=end_date,
        )
    except ProviderResponseError as exc:
        raise ProviderDataError(str(exc)) from exc
    return PriceSeries(
        provider=PROVIDER,
        symbol=normalized_symbol,
        currency=_optional_text(meta.get("currency")),
        bars=bars,
        retrieved_at=datetime.now(tz=UTC),
        source_url=result.url,
        raw_bytes=result.content,
        exchange=_optional_text(meta.get("exchange")),
        mic_code=_optional_text(meta.get("mic_code")),
        instrument_type=_optional_text(meta.get("type")),
        exchange_timezone=_optional_text(meta.get("exchange_timezone")),
        adjustment=adjustment,
    )


def fetch_stock_catalog(
    *,
    exchange: str,
    country: str = "United States",
    instrument_type: str = "Common Stock",
    outputsize: int = MAX_OUTPUT_SIZE,
    required_symbols: Collection[str] | None = None,
    api_key: str | None = None,
    allow_demo: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> StockCatalog:
    """Fetch one official stock-reference catalog slice.

    When ``required_symbols`` is provided, only those rows are normalized.
    The complete provider response remains available in ``raw_bytes``.
    """
    normalized_exchange = exchange.strip().upper()
    if not normalized_exchange:
        raise ValueError("exchange is required")
    if not 1 <= outputsize <= MAX_OUTPUT_SIZE:
        raise ValueError(f"outputsize must be between 1 and {MAX_OUTPUT_SIZE}")
    key = resolve_api_key(api_key, allow_demo=allow_demo)
    result = fetch(
        STOCKS_URL,
        params={
            "exchange": normalized_exchange,
            "country": country,
            "type": instrument_type,
            "outputsize": outputsize,
            "show_plan": "true",
            "format": "JSON",
        },
        headers=_auth_headers(key),
        user_agent=user_agent,
        timeout=DEFAULT_TIMEOUT,
    )
    payload = _load_response(
        result,
        context=f"{normalized_exchange} {instrument_type} catalog",
    )
    raw_data = payload.get("data")
    if not isinstance(raw_data, list):
        raise ProviderResponseError(
            f"Twelve Data {normalized_exchange} stock catalog had no data array"
        )
    symbol_filter = (
        frozenset(_normalize_symbol(symbol) for symbol in required_symbols)
        if required_symbols is not None
        else None
    )
    references: list[StockReference] = []
    for index, row in enumerate(raw_data):
        if symbol_filter is not None:
            if not isinstance(row, dict):
                continue
            row_symbol = _optional_text(row.get("symbol"))
            if row_symbol is None or row_symbol.upper() not in symbol_filter:
                continue
        references.append(_parse_stock_reference(row, exchange=normalized_exchange, index=index))
    parsed_references = tuple(references)
    raw_count = payload.get("count", len(references))
    if not isinstance(raw_count, int) or raw_count < len(parsed_references):
        raise ProviderResponseError(
            f"Twelve Data {normalized_exchange} stock catalog had invalid count "
            f"{raw_count!r} for {len(parsed_references)} rows"
        )
    return StockCatalog(
        provider=PROVIDER,
        exchange=normalized_exchange,
        references=parsed_references,
        count=raw_count,
        retrieved_at=datetime.now(tz=UTC),
        source_url=result.url,
        raw_bytes=result.content,
    )


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"apikey {api_key}",
    }


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol is required")
    if len(normalized) > 32 or any(
        not (character.isalnum() or character in {".", "-"}) for character in normalized
    ):
        raise ValueError(f"Invalid Twelve Data symbol: {symbol!r}")
    return normalized


def _load_response(result: HttpFetchResult, *, context: str) -> dict[str, Any]:
    _raise_for_http_status(result, context=context)
    content_type = result.content_type.lower()
    stripped = result.content.strip()
    if "json" not in content_type and not stripped.startswith(b"{"):
        raise ProviderResponseError(
            f"Twelve Data response for {context} was not JSON (content-type={content_type!r})"
        )
    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(
            f"Twelve Data response for {context} was malformed JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ProviderResponseError(f"Twelve Data response for {context} was not a JSON object")
    status = payload.get("status")
    if status == "error":
        _raise_for_api_error(payload, context=context)
    if status != "ok":
        raise ProviderResponseError(
            f"Twelve Data response for {context} had unexpected status {status!r}"
        )
    return payload


def _raise_for_http_status(result: HttpFetchResult, *, context: str) -> None:
    if result.status_code == 401:
        raise ProviderConfigurationError(
            f"Twelve Data rejected the configured API key for {context} (HTTP 401)"
        )
    if result.status_code == 403:
        raise ProviderBlockedError(
            f"Twelve Data denied access to {context} (HTTP 403); verify the "
            "configured plan and symbol coverage."
        )
    if result.status_code == 429:
        raise ProviderQuotaError(
            f"Twelve Data rate or daily credit limit was reached for {context}"
        )
    if result.status_code >= 400:
        raise ProviderResponseError(
            f"Twelve Data returned unexpected HTTP {result.status_code} for {context}"
        )


def _raise_for_api_error(payload: dict[str, Any], *, context: str) -> None:
    raw_code = payload.get("code")
    try:
        code = int(str(raw_code)) if raw_code is not None else 0
    except (TypeError, ValueError):
        code = 0
    message = str(payload.get("message") or "unspecified provider error")
    lowered = message.casefold()
    if code == 401 or "api key" in lowered or "apikey" in lowered:
        raise ProviderConfigurationError(
            f"Twelve Data rejected the configured API key for {context}"
        )
    if code == 429 or "credit" in lowered or "rate limit" in lowered:
        raise ProviderQuotaError(f"Twelve Data quota was exhausted for {context}: {message[:200]}")
    if code == 403 or "not available with your plan" in lowered:
        raise ProviderBlockedError(f"Twelve Data plan does not permit {context}: {message[:200]}")
    raise ProviderResponseError(
        f"Twelve Data returned API error {code or 'unknown'} for {context}: {message[:200]}"
    )


def _required_mapping(
    payload: dict[str, Any],
    key: str,
    *,
    context: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ProviderResponseError(f"Twelve Data response for {context} had no {key!r} object")
    return value


def _required_text(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = _optional_text(payload.get(key))
    if value is None:
        raise ProviderResponseError(f"Twelve Data response for {context} had no usable {key!r}")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_bars(
    rows: list[object],
    *,
    symbol: str,
    start_date: date | None,
    end_date: date | None,
) -> tuple[PriceBar, ...]:
    bars: list[PriceBar] = []
    seen_dates: set[date] = set()
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            raise ProviderResponseError(f"Twelve Data row {index} for {symbol!r} was not an object")
        raw_datetime = _required_text(raw_row, "datetime", context=symbol)
        try:
            trade_date = date.fromisoformat(raw_datetime[:10])
        except ValueError as exc:
            raise ProviderResponseError(
                f"Twelve Data row {index} for {symbol!r} had invalid datetime {raw_datetime!r}"
            ) from exc
        if trade_date in seen_dates:
            raise ProviderResponseError(
                f"Twelve Data returned duplicate date {trade_date} for {symbol!r}"
            )
        if start_date is not None and trade_date < start_date:
            raise ProviderResponseError(
                f"Twelve Data returned {trade_date} before requested start "
                f"{start_date} for {symbol!r}"
            )
        if end_date is not None and trade_date > end_date:
            raise ProviderResponseError(
                f"Twelve Data returned {trade_date} after requested end {end_date} for {symbol!r}"
            )
        open_price = _optional_decimal(raw_row.get("open"), field="open", symbol=symbol)
        high = _optional_decimal(raw_row.get("high"), field="high", symbol=symbol)
        low = _optional_decimal(raw_row.get("low"), field="low", symbol=symbol)
        close = _required_decimal(raw_row.get("close"), field="close", symbol=symbol)
        volume = _optional_volume(raw_row.get("volume"), symbol=symbol)
        prices = [value for value in (open_price, high, low, close) if value is not None]
        if any(value <= 0 for value in prices):
            raise ProviderResponseError(
                f"Twelve Data returned a non-positive OHLC value for {symbol!r} on {trade_date}"
            )
        if high is not None and high < max(value for value in (open_price, close) if value):
            raise ProviderResponseError(
                f"Twelve Data high was below open/close for {symbol!r} on {trade_date}"
            )
        if low is not None and low > min(value for value in (open_price, close) if value):
            raise ProviderResponseError(
                f"Twelve Data low was above open/close for {symbol!r} on {trade_date}"
            )
        bars.append(
            PriceBar(
                trade_date=trade_date,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
        seen_dates.add(trade_date)
    if not bars:
        raise ProviderResponseError(f"Twelve Data returned no rows for {symbol!r}")
    bars.sort(key=lambda bar: bar.trade_date)
    return tuple(bars)


def _required_decimal(value: object, *, field: str, symbol: str) -> Decimal:
    parsed = _optional_decimal(value, field=field, symbol=symbol)
    if parsed is None:
        raise ProviderResponseError(f"Twelve Data row for {symbol!r} had no usable {field!r}")
    return parsed


def _optional_decimal(
    value: object,
    *,
    field: str,
    symbol: str,
) -> Decimal | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise ProviderResponseError(
            f"Twelve Data row for {symbol!r} had invalid {field!r}: {value!r}"
        ) from exc
    if not parsed.is_finite():
        raise ProviderResponseError(
            f"Twelve Data row for {symbol!r} had non-finite {field!r}: {value!r}"
        )
    return parsed


def _optional_volume(value: object, *, symbol: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        volume = int(str(value))
    except ValueError as exc:
        raise ProviderResponseError(
            f"Twelve Data row for {symbol!r} had invalid volume {value!r}"
        ) from exc
    if volume < 0:
        raise ProviderResponseError(f"Twelve Data row for {symbol!r} had negative volume {volume}")
    return volume


def _parse_stock_reference(
    raw_row: object,
    *,
    exchange: str,
    index: int,
) -> StockReference:
    if not isinstance(raw_row, dict):
        raise ProviderResponseError(
            f"Twelve Data {exchange} stock catalog row {index} was not an object"
        )
    context = f"{exchange} stock catalog row {index}"
    access = raw_row.get("access")
    access_plan = _optional_text(access.get("plan")) if isinstance(access, dict) else None
    return StockReference(
        symbol=_required_text(raw_row, "symbol", context=context).upper(),
        name=_required_text(raw_row, "name", context=context),
        currency=_required_text(raw_row, "currency", context=context).upper(),
        exchange=_required_text(raw_row, "exchange", context=context).upper(),
        mic_code=_required_text(raw_row, "mic_code", context=context).upper(),
        country=_required_text(raw_row, "country", context=context),
        instrument_type=_required_text(raw_row, "type", context=context),
        figi_code=_optional_text(raw_row.get("figi_code")),
        access_plan=access_plan,
    )

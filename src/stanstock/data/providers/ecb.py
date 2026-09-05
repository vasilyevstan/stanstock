"""ECB Data Portal EXR (foreign exchange reference rates) CSV client.

Uses the SDMX 2.1 RESTful API described at
https://data.ecb.europa.eu/help/api/data-examples , requesting the ``EXR``
dataflow in CSV form, for example::

    https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata

No API key is required. ECB reference rates are informational: they
describe market conditions observed around 14:15 CET but are not published
until roughly 16:00 CET (see ``docs/limitations.md``). The CSV rows
themselves carry only the observation date, not a per-row publication
timestamp, so this module models ``published_at`` using that documented
convention and separately exposes the response's ``Last-Modified`` header as
request-level publication metadata (useful for detecting a later revision of
the same series).
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime

from stanstock.data.providers.contracts import FxObservation
from stanstock.data.providers.exceptions import ProviderResponseError
from stanstock.data.providers.http import HttpFetchResult, fetch

PROVIDER = "ecb"
BASE_URL = "https://data-api.ecb.europa.eu/service/data/EXR"
DEFAULT_USER_AGENT = (
    "StanStockResearch/0.1 (+https://github.com/vasilyevstan/stanstock; "
    "non-commercial research spike; contact via repository issues)"
)
#: ECB documents publication around 16:00 CET; modeled here as a fixed UTC
#: offset since the CSV rows carry no per-observation publish time.
_MODELED_PUBLICATION_TIME_UTC = time(15, 0)


def build_series_key(
    quote_currency: str,
    *,
    base_currency: str = "EUR",
    frequency: str = "D",
    exr_type: str = "SP00",
    series_variation: str = "A",
) -> str:
    return f"{frequency}.{quote_currency}.{base_currency}.{exr_type}.{series_variation}"


class EcbExrResult:
    """Parsed EXR observations plus request/response-level publication
    metadata, kept together because the CSV body alone under-describes when
    the data became knowable."""

    __slots__ = ("observations", "raw_bytes", "last_modified", "retrieved_at", "source_url")

    def __init__(
        self,
        *,
        observations: tuple[FxObservation, ...],
        raw_bytes: bytes,
        last_modified: datetime | None,
        retrieved_at: datetime,
        source_url: str,
    ) -> None:
        self.observations = observations
        self.raw_bytes = raw_bytes
        self.last_modified = last_modified
        self.retrieved_at = retrieved_at
        self.source_url = source_url


def fetch_exr_csv(
    quote_currency: str,
    *,
    base_currency: str = "EUR",
    frequency: str = "D",
    start_period: date | None = None,
    end_period: date | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
) -> EcbExrResult:
    """Fetch and parse one EXR series as CSV. Raises
    :class:`ProviderResponseError` for a non-2xx status, an unexpected
    content type, or a CSV body with no observation rows."""
    key = build_series_key(quote_currency, base_currency=base_currency, frequency=frequency)
    params: dict[str, object] = {"format": "csvdata"}
    if start_period is not None:
        params["startPeriod"] = start_period.isoformat()
    if end_period is not None:
        params["endPeriod"] = end_period.isoformat()
    result = fetch(f"{BASE_URL}/{key}", params=params, user_agent=user_agent)
    _raise_for_status(result, series_key=key)
    observations = _parse_exr_csv(
        result.content,
        base_currency=base_currency,
        quote_currency=quote_currency,
        series_key=key,
    )
    return EcbExrResult(
        observations=observations,
        raw_bytes=result.content,
        last_modified=_parse_http_date(result.headers.get("last-modified")),
        retrieved_at=datetime.now(tz=UTC),
        source_url=result.url,
    )


def _raise_for_status(result: HttpFetchResult, *, series_key: str) -> None:
    if result.status_code == 404:
        raise ProviderResponseError(f"ECB has no data for series {series_key!r} (HTTP 404)")
    if result.status_code >= 400:
        raise ProviderResponseError(
            f"ECB returned unexpected HTTP {result.status_code} for series {series_key!r}"
        )
    content_type = result.content_type.lower()
    if content_type and "csv" not in content_type and "text" not in content_type:
        raise ProviderResponseError(
            f"ECB response for series {series_key!r} was not CSV (content-type={content_type!r})"
        )


def _parse_exr_csv(
    payload: bytes,
    *,
    base_currency: str,
    quote_currency: str,
    series_key: str,
) -> tuple[FxObservation, ...]:
    text = payload.decode("utf-8-sig", errors="replace").strip()
    if not text:
        raise ProviderResponseError(f"ECB CSV for series {series_key!r} was empty")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "TIME_PERIOD" not in reader.fieldnames:
        raise ProviderResponseError(
            f"ECB CSV for series {series_key!r} was missing TIME_PERIOD/OBS_VALUE columns: "
            f"{reader.fieldnames!r}"
        )
    observations: list[FxObservation] = []
    for row in reader:
        currency = row.get("CURRENCY")
        currency_denom = row.get("CURRENCY_DENOM")
        if currency and currency != quote_currency:
            raise ProviderResponseError(
                f"ECB CSV row CURRENCY {currency!r} did not match requested "
                f"quote currency {quote_currency!r} for series {series_key!r}"
            )
        if currency_denom and currency_denom != base_currency:
            raise ProviderResponseError(
                f"ECB CSV row CURRENCY_DENOM {currency_denom!r} did not match "
                f"requested base currency {base_currency!r} for series {series_key!r}"
            )
        raw_period = (row.get("TIME_PERIOD") or "").strip()
        raw_value = (row.get("OBS_VALUE") or "").strip()
        if not raw_period or not raw_value:
            continue
        try:
            observation_date = date.fromisoformat(raw_period)
            value = Decimal(raw_value)
        except (ValueError, InvalidOperation) as exc:
            raise ProviderResponseError(
                f"ECB CSV row for series {series_key!r} could not be parsed: {row!r}"
            ) from exc
        observations.append(
            FxObservation(
                base_currency=base_currency,
                quote_currency=quote_currency,
                observation_date=observation_date,
                value=value,
                published_at=datetime.combine(
                    observation_date, _MODELED_PUBLICATION_TIME_UTC, tzinfo=UTC
                ),
                source_url=BASE_URL,
            )
        )
    if not observations:
        raise ProviderResponseError(
            f"ECB CSV for series {series_key!r} had a header but no observation rows"
        )
    return tuple(observations)


def _parse_http_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed

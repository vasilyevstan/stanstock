from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from math import sqrt

import numpy as np
import polars as pl
from django.db import transaction
from django.db.models import Q

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.assets import AssetStore, read_checksummed_bytes
from stanstock.data.market_state import update_latest_market_data
from stanstock.data.models import (
    Company,
    DataAsset,
    LatestMarketData,
    Listing,
    Region,
    Security,
)

INVESTABLE_US_ETF_SYMBOL = "SPY"
INVESTABLE_US_ETF_NAME = "SPDR S&P 500 ETF Trust"
INVESTABLE_US_ETF_MIC = "ARCX"
INVESTABLE_US_ETF_BENCHMARK = "S&P 500 benchmark"
INVESTABLE_US_ETF_PORTFOLIO_ROLE = "Core benchmark ETF"
ETF_TRAILING_SESSIONS = 252


@dataclass(frozen=True, slots=True)
class EtfOverview:
    listing: Listing
    market_data: LatestMarketData
    price_return: float | None
    annualized_volatility: float | None
    max_drawdown: float | None
    observation_count: int
    period_start: date
    period_end: date
    benchmark_identity: str
    portfolio_role: str
    return_definition: str
    dividends_included: bool


def is_supported_investable_etf(listing: Listing) -> bool:
    return (
        listing.security.security_type == Security.SecurityType.ETF
        and listing.ticker.upper() == INVESTABLE_US_ETF_SYMBOL
        and listing.provider_symbol == INVESTABLE_US_ETF_SYMBOL
        and listing.exchange_mic == INVESTABLE_US_ETF_MIC
        and listing.currency.upper() == "USD"
        and listing.region == Region.US
        and listing.is_primary
        and listing.is_active
        and listing.security.name == INVESTABLE_US_ETF_NAME
        and listing.security.company.name == INVESTABLE_US_ETF_NAME
    )


def ensure_investable_spy_listing(
    *,
    currency: str,
    mic_code: str,
    valid_from: date,
) -> Listing:
    if currency.upper() != "USD":
        raise ValueError(f"SPY currency must be USD, received {currency!r}")
    normalized_mic = mic_code.strip().upper()
    if normalized_mic != INVESTABLE_US_ETF_MIC:
        raise ValueError(f"SPY MIC must be {INVESTABLE_US_ETF_MIC}, received {mic_code!r}")

    with transaction.atomic():
        candidates = list(
            Listing.objects.select_for_update()
            .select_related("security__company")
            .filter(
                Q(provider_symbol=INVESTABLE_US_ETF_SYMBOL) | Q(ticker=INVESTABLE_US_ETF_SYMBOL),
                region=Region.US,
                is_active=True,
            )
        )
        if len(candidates) > 1:
            raise ValueError("Multiple active US listings identify SPY")
        if candidates:
            listing = candidates[0]
            if listing.security.security_type != Security.SecurityType.ETF:
                raise ValueError(
                    "Existing SPY listing is not identified as an exchange-traded fund"
                )
            if (
                listing.ticker.upper() != INVESTABLE_US_ETF_SYMBOL
                or listing.currency.upper() != "USD"
                or listing.exchange_mic != normalized_mic
                or listing.security.name != INVESTABLE_US_ETF_NAME
                or listing.security.company.name != INVESTABLE_US_ETF_NAME
                or not listing.is_primary
            ):
                raise ValueError(
                    "Existing SPY listing conflicts with the current provider identity"
                )
            if listing.provider_symbol not in {"", INVESTABLE_US_ETF_SYMBOL}:
                raise ValueError("Existing SPY listing has a conflicting provider symbol")
            if not listing.provider_symbol:
                listing.provider_symbol = INVESTABLE_US_ETF_SYMBOL
                listing.save(update_fields=["provider_symbol"])
            return listing

        company = Company.objects.create(
            name=INVESTABLE_US_ETF_NAME,
            country="US",
            sector="Exchange-traded fund",
        )
        security = Security.objects.create(
            company=company,
            security_type=Security.SecurityType.ETF,
            name=INVESTABLE_US_ETF_NAME,
        )
        return Listing.objects.create(
            security=security,
            ticker=INVESTABLE_US_ETF_SYMBOL,
            exchange_mic=normalized_mic,
            provider_symbol=INVESTABLE_US_ETF_SYMBOL,
            currency="USD",
            region=Region.US,
            valid_from=valid_from,
            is_primary=True,
            is_active=True,
        )


def sync_investable_spy_from_asset(
    *,
    asset: DataAsset,
    target_date: date,
    store: AssetStore | None = None,
) -> Listing:
    metadata = _validate_spy_price_asset(asset)
    frame = _read_verified_spy_price_frame(asset, store)
    eligible = frame.filter(pl.col("date") <= target_date)
    if eligible.is_empty():
        raise ValueError(f"SPY price asset has no rows through {target_date.isoformat()}")
    rows = eligible.tail(2).to_dicts()
    latest = rows[-1]
    if latest["date"] != target_date:
        raise ValueError(
            f"SPY price asset has no {target_date.isoformat()} close; "
            f"latest is {latest['date'].isoformat()}"
        )
    previous = rows[-2] if len(rows) > 1 else None
    with transaction.atomic():
        listing = ensure_investable_spy_listing(
            currency=str(metadata["currency"]),
            mic_code=str(metadata.get("resolved_mic_code") or metadata.get("mic_code") or ""),
            valid_from=target_date,
        )
        update_latest_market_data(
            listing=listing,
            session_date=latest["date"],
            observed_at=asset.retrieved_at,
            close=Decimal(str(latest["close"])),
            previous_close=(Decimal(str(previous["close"])) if previous is not None else None),
            volume=int(latest["volume"]) if latest["volume"] is not None else None,
            source_asset=asset,
        )
    return listing


def build_etf_overview(
    listing: Listing,
    *,
    store: AssetStore | None = None,
) -> EtfOverview:
    if listing.security.security_type != Security.SecurityType.ETF:
        raise ValueError(f"{listing.ticker} is not an exchange-traded fund")
    if not is_supported_investable_etf(listing):
        raise ValueError(f"{listing.ticker} is not a supported investable ETF")
    try:
        market_data = listing.latest_market_data
    except LatestMarketData.DoesNotExist as exc:
        raise ValueError(f"{listing.ticker} has no current market data") from exc

    _validate_spy_price_asset(market_data.source_asset)
    frame = _read_verified_spy_price_frame(market_data.source_asset, store)
    eligible = frame.filter(pl.col("date") <= market_data.session_date)
    if eligible.is_empty() or eligible["date"][-1] != market_data.session_date:
        raise ValueError(f"SPY price asset has no {market_data.session_date.isoformat()} close")
    trailing = eligible.tail(ETF_TRAILING_SESSIONS + 1)
    dates = trailing["date"].to_list()
    closes = np.asarray(trailing["close"].to_numpy(), dtype=np.float64)
    returns = closes[1:] / closes[:-1] - 1.0 if len(closes) >= 2 else np.asarray([])
    price_return = float(closes[-1] / closes[0] - 1.0) if len(closes) >= 2 else None
    annualized_volatility = (
        float(np.std(returns, ddof=1) * sqrt(ETF_TRAILING_SESSIONS)) if len(returns) >= 2 else None
    )
    running_high = np.maximum.accumulate(closes)
    max_drawdown = float(np.min(closes / running_high - 1.0)) if len(closes) >= 2 else None
    metadata = market_data.source_asset.metadata
    return EtfOverview(
        listing=listing,
        market_data=market_data,
        price_return=price_return,
        annualized_volatility=annualized_volatility,
        max_drawdown=max_drawdown,
        observation_count=len(closes),
        period_start=dates[0],
        period_end=dates[-1],
        benchmark_identity=INVESTABLE_US_ETF_BENCHMARK,
        portfolio_role=INVESTABLE_US_ETF_PORTFOLIO_ROLE,
        return_definition=str(metadata["return_definition"]),
        dividends_included=bool(metadata["dividends_included"]),
    )


class InvestableEtfEvidenceError(ValueError):
    """The exact SPY price evidence asset could not be read or parsed.

    Carries a stable, path-free message; never chains a path-bearing cause.
    """


def _read_verified_spy_price_frame(asset: DataAsset, store: AssetStore | None) -> pl.DataFrame:
    """Read, checksum-authenticate, and strictly validate the exact SPY
    price frame for `asset`.

    The physical bytes are hashed once via `read_checksummed_bytes` and
    those same bytes are parsed in-memory -- never re-opened by path after
    hashing -- so an on-disk substitution that changes content cannot be
    projected as if it were the registered asset. Any store-construction,
    checksum, read, or schema/conversion failure -- including a path-bearing
    `OSError`/`ValueError`/`pl.exceptions.PolarsError` -- becomes a stable,
    path-free `InvestableEtfEvidenceError` raised without a path-bearing
    cause.
    """
    try:
        active_store = store or AssetStore()
        payload = read_checksummed_bytes(active_store, asset)
        frame = pl.read_parquet(io.BytesIO(payload))
        return _clean_price_frame(frame)
    except (OSError, ValueError, KeyError, pl.exceptions.PolarsError, RefreshVerificationError):
        raise InvestableEtfEvidenceError("SPY price evidence could not be read") from None


def _validate_spy_price_asset(asset: DataAsset) -> dict[str, object]:
    metadata = asset.metadata if isinstance(asset.metadata, dict) else {}
    if (
        asset.provider != "twelve_data"
        or asset.kind != "price_history"
        or asset.subject != INVESTABLE_US_ETF_SYMBOL
    ):
        raise ValueError("Asset is not an eligible Twelve Data SPY price history")
    expected_metadata: dict[str, object] = {
        "instrument_type": "ETF",
        "currency": "USD",
        "interval": "1day",
        "adjustment": "splits",
        "return_definition": "split_adjusted_price_return",
        "dividends_included": False,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(f"SPY price asset metadata {field!r} must be {expected!r}")
    provider_mic_code = metadata.get("mic_code")
    if provider_mic_code not in {None, ""} and (
        not isinstance(provider_mic_code, str)
        or provider_mic_code.strip().upper() != INVESTABLE_US_ETF_MIC
    ):
        raise ValueError(
            f"SPY price asset metadata 'mic_code' conflicts with {INVESTABLE_US_ETF_MIC!r}"
        )
    resolved_mic_code = metadata.get("resolved_mic_code") or provider_mic_code
    if (
        not isinstance(resolved_mic_code, str)
        or resolved_mic_code.strip().upper() != INVESTABLE_US_ETF_MIC
    ):
        raise ValueError(
            f"SPY price asset metadata 'mic_code' requires resolved MIC {INVESTABLE_US_ETF_MIC!r}"
        )
    if "resolved_mic_code" in metadata:
        expected_source = (
            "provider" if provider_mic_code not in {None, ""} else "configured_spy_identity"
        )
        if metadata.get("mic_code_source") != expected_source:
            raise ValueError("SPY price asset metadata has an invalid MIC identity source")
    return metadata


_REQUIRED_PRICE_SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date(),
    "close": pl.Float64(),
    "volume": pl.Int64(),
}


def _clean_price_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Strictly validate the *entire* selected `date`/`close`/`volume`
    frame -- never silently dropping a malformed row.

    A checksum-valid but malformed row (a null date/close, a non-finite or
    non-positive close, a negative volume) must fail the whole asset
    closed, not be filtered away as if it never existed -- otherwise a
    sync could silently bridge across a discarded session, and an overview
    could compute trailing metrics from a quietly-reduced frame.
    """
    missing = sorted(set(_REQUIRED_PRICE_SCHEMA) - set(frame.columns))
    if missing:
        raise ValueError(f"ETF price asset is missing columns: {', '.join(missing)}")
    for column, expected_dtype in _REQUIRED_PRICE_SCHEMA.items():
        if frame.schema[column] != expected_dtype:
            raise ValueError(f"ETF price asset column {column!r} has an unexpected type")
    selected = frame.select("date", "close", "volume")
    if selected.is_empty():
        raise ValueError("ETF price asset has no observations")
    if selected["date"].null_count() > 0:
        raise ValueError("ETF price asset contains a null session date")
    if selected["close"].null_count() > 0:
        raise ValueError("ETF price asset contains a null close")
    if not bool(selected["close"].is_finite().all()):
        raise ValueError("ETF price asset contains a non-finite close")
    if bool((selected["close"] <= 0).any()):
        raise ValueError("ETF price asset contains a non-positive close")
    if bool((selected["volume"].is_not_null() & (selected["volume"] < 0)).any()):
        raise ValueError("ETF price asset contains a negative volume")
    clean = selected.sort("date")
    if clean["date"].n_unique() != clean.height:
        raise ValueError("ETF price asset contains duplicate session dates")
    return clean

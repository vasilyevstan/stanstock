"""Engine-side FX support: input normalization and honest return attribution.

The accounting engine stays currency-agnostic: by the time a price panel
reaches it, every close, open, and benchmark level is already denominated in
one base currency. What the engine additionally receives is the exact dated
FX frame those conversions came from, which lets it do two things it could
not otherwise do honestly.

First, the frame participates in the reproducibility hash, so two runs that
differ only in which FX vintage converted them are never mistaken for the
same run.

Second, `FxShadowLedger` mirrors the run at fixed reference rates. It applies
the *same* quantity path -- no re-optimization, no second portfolio -- but
settles every cash movement and revalues every open position at each native
currency's rate on the first simulated date. The mirrored track therefore
starts from the same capital and differs from the real track only through FX
moves, so ``local return + FX contribution == reported return`` exactly.

The ledger refuses to guess. A cash settlement with no established FX basis,
a missing reference rate, or a position whose price was carried forward from
a date with no rate all mark the attribution unavailable rather than
producing a number that looks measured but is not.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date

import polars as pl

from stanstock.simulation.types import FxAttribution, FxAttributionStatus

FX_REQUIRED_COLUMNS: tuple[str, ...] = (
    "value_date",
    "from_currency",
    "to_currency",
    "rate",
)
FX_CARRY_COLUMN = "carry_days"


def normalize_fx_frame(frame: pl.DataFrame, *, base_currency: str | None) -> pl.DataFrame:
    """Canonicalize a dated FX frame and reject anything unusable.

    Ambiguity here would silently change every converted value in the run,
    so duplicate (date, currency) rows, non-positive rates, and a frame
    quoting into a currency other than the run's base all fail explicitly.
    """
    missing = [column for column in FX_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"FX rates must contain {list(FX_REQUIRED_COLUMNS)} columns. Missing: {missing}"
        )

    normalized = frame.with_columns(
        [
            pl.col("value_date").cast(pl.Date),
            pl.col("from_currency").cast(pl.Utf8).str.to_uppercase(),
            pl.col("to_currency").cast(pl.Utf8).str.to_uppercase(),
            pl.col("rate").cast(pl.Float64),
        ]
    )
    if FX_CARRY_COLUMN in normalized.columns:
        normalized = normalized.with_columns(pl.col(FX_CARRY_COLUMN).cast(pl.Int32))

    ordered_columns = [
        *FX_REQUIRED_COLUMNS,
        *sorted(column for column in normalized.columns if column not in FX_REQUIRED_COLUMNS),
    ]
    normalized = normalized.select(ordered_columns).sort(
        ["value_date", "from_currency", "to_currency"]
    )

    if normalized.height == 0:
        raise ValueError("FX rates frame contains no rows")

    key_columns = ["value_date", "from_currency", "to_currency"]
    if normalized.select(key_columns).is_duplicated().any():
        duplicates = (
            normalized.filter(normalized.select(key_columns).is_duplicated())
            .select(key_columns)
            .unique()
            .sort(key_columns)
        )
        raise ValueError(
            f"Duplicate FX rates detected for (value_date, from_currency, to_currency): "
            f"{duplicates.to_dicts()}"
        )

    if normalized["rate"].null_count() > 0 or normalized.filter(pl.col("rate") <= 0).height > 0:
        raise ValueError("FX rates must all be positive and non-null")

    quote_currencies = sorted(set(normalized["to_currency"].to_list()))
    if len(quote_currencies) != 1:
        raise ValueError(
            f"FX rates must all quote into one base currency, found: {quote_currencies}"
        )
    if base_currency is not None and quote_currencies[0] != base_currency.upper():
        raise ValueError(
            f"FX rates quote into {quote_currencies[0]}, but the simulation base currency "
            f"is {base_currency.upper()}"
        )

    return normalized


def build_fx_rate_lookup(frame: pl.DataFrame) -> dict[tuple[str, date], float]:
    return {
        (row["from_currency"], row["value_date"]): float(row["rate"])
        for row in frame.iter_rows(named=True)
    }


def validate_fx_coverage(
    frame: pl.DataFrame,
    *,
    trading_dates: Iterable[date],
    base_currency: str,
    required_currencies: Iterable[str] = (),
) -> None:
    """Require a resolved rate for every currency on every accounted date.

    The frame is built by resolving each date against the carry limit before
    any accounting starts, so a gap here means a date the run intends to
    value has no rate the point-in-time rules would admit -- typically an
    explicit calendar reaching past the FX series. Continuing would price
    that date off whatever conversion happened to be lying around and then
    publish a portfolio return for it, so the run fails instead. Withholding
    only the FX attribution would still leave the reported return silently
    wrong.
    """
    base = base_currency.upper()
    currencies = {
        currency
        for currency in frame["from_currency"].to_list()
        if currency and currency.upper() != base
    }
    currencies.update(
        currency.upper()
        for currency in required_currencies
        if currency and currency.upper() != base
    )
    if not currencies:
        return

    available = set(build_fx_rate_lookup(frame))
    gaps = sorted(
        (currency, value_date)
        for currency in currencies
        for value_date in set(trading_dates)
        if (currency, value_date) not in available
    )
    if not gaps:
        return
    shown = ", ".join(
        f"{currency}/{base} on {value_date.isoformat()}" for currency, value_date in gaps[:5]
    )
    raise ValueError(
        f"FX rates are missing for {len(gaps)} accounted (currency, date) pair(s): {shown}"
        f"{' ...' if len(gaps) > 5 else ''}. Every simulated date must resolve its own rate "
        "within the carry limit; refusing to report a portfolio return for a date that has "
        "no eligible conversion."
    )


def max_carry_days(frame: pl.DataFrame) -> int | None:
    if FX_CARRY_COLUMN not in frame.columns:
        return None
    observed = frame[FX_CARRY_COLUMN].max()
    return observed if isinstance(observed, int) else None


def converted_currencies(frame: pl.DataFrame) -> tuple[str, ...]:
    """Currencies the frame actually converts, excluding identity rows."""
    non_identity = frame.filter(pl.col("from_currency") != pl.col("to_currency"))
    return tuple(sorted(set(non_identity["from_currency"].to_list())))


class FxShadowLedger:
    """Track the same portfolio settled and valued at fixed reference rates."""

    def __init__(
        self,
        *,
        starting_capital: float,
        base_currency: str,
        currency_by_listing: Mapping[str, str],
        rate_lookup: Mapping[tuple[str, date], float],
        inception_date: date,
        converted_currencies: Iterable[str] = (),
        carry_days_used: int | None = None,
    ) -> None:
        self.base_currency = base_currency.upper()
        self.starting_capital = float(starting_capital)
        self.shadow_cash = float(starting_capital)
        self.carry_days_used = carry_days_used
        self._currency_by_listing = {
            listing_id: currency.upper() for listing_id, currency in currency_by_listing.items()
        }
        self._rates = dict(rate_lookup)
        self._last_rate: dict[str, float] = {}
        self._unavailable_reason: str | None = None
        held_currencies = set(self._currency_by_listing.values())
        # The frame's currencies, not just the held ones: a run whose only
        # non-base exposure is its benchmark still converted something, and
        # saying otherwise in the metrics would understate what happened.
        self.native_currencies: tuple[str, ...] = tuple(
            sorted(held_currencies | {value.upper() for value in converted_currencies})
        )
        self._reference_rates: dict[str, float] = {}
        for currency in sorted(held_currencies):
            if currency == self.base_currency:
                self._reference_rates[currency] = 1.0
                continue
            reference = self._rates.get((currency, inception_date))
            if reference is None or reference <= 0:
                self.mark_unavailable(
                    f"No {currency}/{self.base_currency} reference rate is available on the "
                    f"first simulated date {inception_date.isoformat()}."
                )
                continue
            self._reference_rates[currency] = reference

    @property
    def conversion_applied(self) -> bool:
        return any(currency != self.base_currency for currency in self.native_currencies)

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def mark_unavailable(self, reason: str) -> None:
        if self._unavailable_reason is None:
            self._unavailable_reason = reason

    def dated_rate(self, listing_id: str, when: date) -> float | None:
        currency = self._currency_by_listing.get(listing_id)
        if currency is None:
            return None
        if currency == self.base_currency:
            return 1.0
        return self._rates.get((currency, when))

    def record_price_rate(self, listing_id: str, when: date) -> None:
        """Remember the rate that converted the price now carried for a listing.

        A price retained from an earlier session was converted at *that*
        session's rate, so the reference-rate mirror must undo the same rate
        rather than the current date's.
        """
        rate = self.dated_rate(listing_id, when)
        if rate is None or rate <= 0:
            self.mark_unavailable(
                f"No FX rate is available for listing {listing_id} on {when.isoformat()}, "
                "so its converted price cannot be restated at reference rates."
            )
            return
        self._last_rate[listing_id] = rate

    def record_cash_flow(self, *, listing_id: str, when: date, base_amount: float) -> None:
        scale = self._scale(listing_id, self.dated_rate(listing_id, when))
        if scale is None:
            self.mark_unavailable(
                f"Cash movement for listing {listing_id} on {when.isoformat()} has no usable "
                "FX rate, so it cannot be restated at reference rates."
            )
            return
        self.shadow_cash += base_amount * scale

    def record_cash_settlement(self, *, listing_id: str, when: date) -> None:
        self.mark_unavailable(
            f"Listing {listing_id} was cash-settled on {when.isoformat()} at a price whose FX "
            "basis is not established, so stock and FX contributions cannot be separated."
        )

    def shadow_market_value(self, positions: Iterable[tuple[str, float]]) -> float:
        total = 0.0
        for listing_id, base_market_value in positions:
            scale = self._scale(listing_id, self._last_rate.get(listing_id))
            if scale is None:
                self.mark_unavailable(
                    f"Holding {listing_id} has no recorded FX rate for its carried price, so "
                    "it cannot be restated at reference rates."
                )
                continue
            total += base_market_value * scale
        return total

    def attribution(
        self,
        *,
        cumulative_return: float,
        terminal_shadow_market_value: float,
    ) -> FxAttribution:
        if not self.conversion_applied:
            return FxAttribution.not_applicable()
        if self._unavailable_reason is not None:
            return FxAttribution(
                status=FxAttributionStatus.UNAVAILABLE,
                detail=self._unavailable_reason,
                conversion_applied=True,
                native_currencies=self.native_currencies,
                max_carry_days_used=self.carry_days_used,
            )
        if self.starting_capital <= 0:  # pragma: no cover - config validation forbids this
            return FxAttribution(
                status=FxAttributionStatus.UNAVAILABLE,
                detail="Starting capital must be positive to attribute returns.",
                conversion_applied=True,
                native_currencies=self.native_currencies,
                max_carry_days_used=self.carry_days_used,
            )

        shadow_value = self.shadow_cash + terminal_shadow_market_value
        local_return = (shadow_value - self.starting_capital) / self.starting_capital
        return FxAttribution(
            status=FxAttributionStatus.EXACT,
            detail=(
                "Stock return is the same quantity path settled and valued at each native "
                "currency's rate on the first simulated date; the FX contribution is the "
                "remainder of the reported return."
            ),
            conversion_applied=True,
            native_currencies=self.native_currencies,
            max_carry_days_used=self.carry_days_used,
            local_currency_cumulative_return=local_return,
            contribution_return=cumulative_return - local_return,
        )

    def _scale(self, listing_id: str, rate: float | None) -> float | None:
        currency = self._currency_by_listing.get(listing_id)
        if currency is None or rate is None or rate <= 0:
            return None
        reference = self._reference_rates.get(currency)
        if reference is None:
            return None
        return reference / rate

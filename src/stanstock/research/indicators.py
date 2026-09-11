from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl

from stanstock.research.config import CommonRiskPolicy
from stanstock.research.types import IndicatorResult

SUPPORTED_WINDOWS = (1, 5, 10, 20, 63, 126, 252, 756, 1260)
TRADING_DAYS = 252.0

#: `median_dollar_volume` statuses.
DOLLAR_VOLUME_COMPUTED = "computed"
DOLLAR_VOLUME_WITHHELD = "withheld"

#: Explicit refusal reasons. Each names the condition that was actually
#: observed: a malformed row is never described as missing history, and a
#: short window is never described as invalid.
DOLLAR_VOLUME_MISSING_COLUMNS = "missing_price_columns"
DOLLAR_VOLUME_INVALID_SESSION_DATES = "invalid_session_dates"
DOLLAR_VOLUME_DUPLICATE_SESSIONS = "duplicate_sessions"
DOLLAR_VOLUME_INVALID_CLOSE = "invalid_close_values"
DOLLAR_VOLUME_INVALID_VOLUME = "invalid_volume_values"
DOLLAR_VOLUME_NONFINITE_PRODUCT = "nonfinite_dollar_volume_product"
DOLLAR_VOLUME_DROPPED_ROWS = "invalid_rows_dropped_during_preparation"
DOLLAR_VOLUME_INSUFFICIENT_SESSIONS = "insufficient_sessions"
DOLLAR_VOLUME_NONFINITE_MEDIAN = "nonfinite_median"


def calculate_indicators(
    frame: pl.DataFrame,
    *,
    benchmark: pl.DataFrame | None = None,
    windows: Iterable[int] = SUPPORTED_WINDOWS,
    common_risk_policy: CommonRiskPolicy | None = None,
) -> IndicatorResult:
    clean = _prepare_price_frame(frame)
    missing: dict[str, str] = {}
    values: dict[str, float] = {}
    if clean.is_empty():
        return IndicatorResult(values={}, missing={"price_history": "No usable positive closes"})

    closes = _series(clean, "close")
    highs = _optional_series(clean, "high")
    lows = _optional_series(clean, "low")
    volumes = _optional_series(clean, "volume")
    last_close = float(closes[-1])
    values["last_close"] = last_close
    values["observation_count"] = float(len(closes))

    for window in windows:
        key = f"return_{window}d"
        if len(closes) > window and closes[-window - 1] > 0:
            values[key] = float(last_close / closes[-window - 1] - 1.0)
        else:
            missing[key] = f"Need at least {window + 1} closes"

    for window in (20, 50, 100, 200):
        key = f"sma_{window}"
        if len(closes) >= window:
            sma = float(np.mean(closes[-window:]))
            values[key] = sma
            values[f"close_vs_sma_{window}"] = float(last_close / sma - 1.0) if sma > 0 else 0.0
        else:
            missing[key] = f"Need at least {window} closes"

    for window in (12, 20, 26):
        key = f"ema_{window}"
        if len(closes) >= window:
            values[key] = float(_ema(closes, window)[-1])
            if window == 20:
                values["ema"] = values[key]
        else:
            missing[key] = f"Need at least {window} closes"

    if len(closes) >= 15:
        values["rsi_14"] = _rsi(closes, 14)
    else:
        missing["rsi_14"] = "Need at least 15 closes"

    if len(closes) >= 35:
        macd_line = _ema(closes, 12) - _ema(closes, 26)
        signal = _ema(macd_line[~np.isnan(macd_line)], 9)
        values["macd"] = float(macd_line[-1])
        values["macd_signal"] = float(signal[-1])
        histogram = float(macd_line[-1] - signal[-1])
        values["macd_histogram"] = histogram
        values["macd_histogram_pct"] = histogram / last_close
    else:
        missing["macd"] = "Need at least 35 closes"
        missing["macd_histogram_pct"] = missing["macd"]

    if highs is not None and lows is not None and len(closes) >= 15:
        values["atr_14"] = _atr(highs, lows, closes, 14)
        values["atr_14_pct"] = values["atr_14"] / last_close if last_close > 0 else 0.0
    else:
        missing["atr_14"] = "Need high/low and at least 15 closes"

    if common_risk_policy is None:
        daily_returns = _returns(closes)
        if len(daily_returns) >= 2:
            values["annualized_volatility"] = _annualized_std(daily_returns)
            values["downside_volatility"] = _downside_deviation(daily_returns)
            values["max_drawdown"] = _max_drawdown(closes)
        else:
            missing["annualized_volatility"] = "Need at least three closes"
            missing["max_drawdown"] = "Need at least two closes"

    for window in (20, 63, 126, 252):
        momentum_key = f"momentum_{window}d"
        return_key = f"return_{window}d"
        if return_key in values:
            values[momentum_key] = values[return_key]
        else:
            missing[momentum_key] = missing.get(return_key, f"Need at least {window + 1} closes")

    if len(closes) >= 252:
        low_52w = float(np.min(closes[-252:]))
        high_52w = float(np.max(closes[-252:]))
        values["52w_low"] = low_52w
        values["52w_high"] = high_52w
        values["52w_range"] = high_52w - low_52w
        values["52w_range_pct"] = float(high_52w / low_52w - 1.0) if low_52w > 0 else 0.0
        values["distance_from_52w_high"] = (
            float(last_close / high_52w - 1.0) if high_52w > 0 else 0.0
        )
        values["distance_from_52w_low"] = float(last_close / low_52w - 1.0) if low_52w > 0 else 0.0
        span = high_52w - low_52w
        values["52w_position"] = float((last_close - low_52w) / span) if span > 0 else 0.5
    else:
        missing["52w_range"] = "Need at least 252 closes"

    if volumes is not None and len(volumes) >= 20:
        recent_volumes = volumes[-20:]
        avg_20 = float(np.mean(recent_volumes))
        values["avg_volume_20d"] = avg_20
        values["abnormal_volume"] = float(volumes[-1] / avg_20) if avg_20 > 0 else 0.0
        if len(volumes) >= 80:
            avg_60_prior = float(np.mean(volumes[-80:-20]))
            values["volume_trend"] = float(avg_20 / avg_60_prior - 1.0) if avg_60_prior > 0 else 0.0
        else:
            missing["volume_trend"] = "Need at least 80 volume observations"
        recent_closes = closes[-20:]
        if (
            np.isfinite(recent_volumes).all()
            and (recent_volumes >= 0).all()
            and np.isfinite(recent_closes).all()
        ):
            values["avg_dollar_volume_20d"] = float(np.mean(recent_closes * recent_volumes))
            values["abnormal_volume_strict"] = float(volumes[-1] / avg_20) if avg_20 > 0 else 0.0
        else:
            missing["avg_dollar_volume_20d"] = "Recent volume observations must be finite"
            missing["abnormal_volume_strict"] = "Recent volume observations must be finite"
    else:
        missing["volume"] = "Need volume and at least 20 observations"
        missing["avg_dollar_volume_20d"] = missing["volume"]
        missing["abnormal_volume_strict"] = missing["volume"]

    if common_risk_policy is not None:
        values.update(
            _common_benchmark_values(
                clean,
                benchmark,
                missing,
                common_risk_policy,
            )
        )
    elif benchmark is not None:
        values.update(_benchmark_values(clean, benchmark, missing))

    last_date = clean.select("date").to_series().to_list()[-1] if "date" in clean.columns else None
    return IndicatorResult(
        values=values,
        missing=missing,
        observation_count=len(closes),
        last_date=last_date,
    )


def _prepare_price_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if "close" not in frame.columns:
        return pl.DataFrame({"close": []})
    expressions: list[pl.Expr] = [pl.col("close").cast(pl.Float64, strict=False)]
    for column in ("open", "high", "low", "volume"):
        if column in frame.columns:
            expressions.append(pl.col(column).cast(pl.Float64, strict=False))
    if "date" in frame.columns:
        expressions.append(pl.col("date"))
    selected = frame.select(expressions).filter(
        pl.col("close").is_not_null() & (pl.col("close") > 0)
    )
    if "date" in selected.columns:
        selected = selected.sort("date")
    return selected


def _series(frame: pl.DataFrame, column: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.asarray(frame[column].to_numpy(), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DollarVolumeResult:
    """Outcome of one median dollar-volume observation window.

    ``value`` is a finite number only when ``status`` is
    `DOLLAR_VOLUME_COMPUTED`; every other case carries an explicit
    ``reason``. The window descriptors stay populated whenever a
    date-validated observation window existed, because "251 observed
    sessions ending 2026-02-27" is a fact about the evidence, not an
    invented metric. They are ``None`` when no trustworthy window could be
    identified at all.
    """

    status: str
    value: float | None = None
    sessions_used: int | None = None
    first_session: date | None = None
    last_session: date | None = None
    reason: str | None = None


def median_dollar_volume(
    frame: pl.DataFrame,
    *,
    sessions: int = 252,
) -> DollarVolumeResult:
    """Median ``close * volume`` over the latest ``sessions`` observed sessions.

    The caller supplies the already date-normalized, cutoff-clipped frame
    from `AsOfData.price_frame`, so this helper never widens availability.

    The raw observation window is validated *before* `_prepare_price_frame`
    can silently drop a row: an unparseable, null, non-finite, non-positive
    close (or a negative/non-finite volume) is an explicit invalid-input
    refusal, never `DOLLAR_VOLUME_INSUFFICIENT_SESSIONS`. Reaching further
    back to replace an invalid row, padding calendar gaps, or resolving a
    duplicate session date would each turn bad evidence into a plausible
    number, so all three are refused instead.

    A reported volume of zero is valid data. A computed zero median is
    returned as zero and carries no pass/fail conclusion; this helper
    implements no threshold.
    """
    if sessions < 1:
        raise ValueError(f"median_dollar_volume needs a positive session count, got {sessions!r}")
    missing_columns = [
        column for column in ("date", "close", "volume") if column not in frame.columns
    ]
    if missing_columns:
        return DollarVolumeResult(
            status=DOLLAR_VOLUME_WITHHELD,
            reason=DOLLAR_VOLUME_MISSING_COLUMNS,
        )
    if frame.height == 0:
        # No session identity exists to validate; this is an empty history,
        # not a malformed one.
        return DollarVolumeResult(
            status=DOLLAR_VOLUME_WITHHELD,
            sessions_used=0,
            reason=DOLLAR_VOLUME_INSUFFICIENT_SESSIONS,
        )
    session_dates = _session_dates(frame)
    if session_dates is None:
        return DollarVolumeResult(
            status=DOLLAR_VOLUME_WITHHELD,
            reason=DOLLAR_VOLUME_INVALID_SESSION_DATES,
        )
    # Duplicate session identities are detected across the whole eligible
    # frame, before any ordering or preparation, because "the latest 252
    # sessions" is undefined while one date names two rows.
    if len(set(session_dates)) != len(session_dates):
        return DollarVolumeResult(
            status=DOLLAR_VOLUME_WITHHELD,
            reason=DOLLAR_VOLUME_DUPLICATE_SESSIONS,
        )
    ordered = frame.with_columns(pl.Series("date", session_dates, dtype=pl.Date)).sort("date")
    window = ordered.tail(sessions) if ordered.height > sessions else ordered
    window_dates = list(window["date"].to_list())
    observed = DollarVolumeResult(
        status=DOLLAR_VOLUME_WITHHELD,
        sessions_used=window.height,
        first_session=window_dates[0],
        last_session=window_dates[-1],
    )
    closes = _finite_column(window, "close")
    if closes is None or not bool((closes > 0.0).all()):
        return _withheld(observed, DOLLAR_VOLUME_INVALID_CLOSE)
    volumes = _finite_column(window, "volume")
    if volumes is None or not bool((volumes >= 0.0).all()):
        return _withheld(observed, DOLLAR_VOLUME_INVALID_VOLUME)
    with np.errstate(over="ignore", invalid="ignore"):
        if not bool(np.isfinite(closes * volumes).all()):
            return _withheld(observed, DOLLAR_VOLUME_NONFINITE_PRODUCT)
        prepared = _prepare_price_frame(window)
        if prepared.height != window.height:
            return _withheld(observed, DOLLAR_VOLUME_DROPPED_ROWS)
        if window.height < sessions:
            return _withheld(observed, DOLLAR_VOLUME_INSUFFICIENT_SESSIONS)
        products = _series(prepared, "close") * _series(prepared, "volume")
        if not bool(np.isfinite(products).all()):
            return _withheld(observed, DOLLAR_VOLUME_NONFINITE_PRODUCT)
        median = float(np.median(products))
    if not math.isfinite(median):
        return _withheld(observed, DOLLAR_VOLUME_NONFINITE_MEDIAN)
    return DollarVolumeResult(
        status=DOLLAR_VOLUME_COMPUTED,
        value=median,
        sessions_used=observed.sessions_used,
        first_session=observed.first_session,
        last_session=observed.last_session,
    )


def _withheld(observed: DollarVolumeResult, reason: str) -> DollarVolumeResult:
    return DollarVolumeResult(
        status=DOLLAR_VOLUME_WITHHELD,
        sessions_used=observed.sessions_used,
        first_session=observed.first_session,
        last_session=observed.last_session,
        reason=reason,
    )


def _session_dates(frame: pl.DataFrame) -> list[date] | None:
    """Session identities as plain dates, or ``None`` when unusable.

    Only the dtypes `AsOfData.price_frame` can produce are accepted. A
    string, integer, or otherwise ambiguous column is refused rather than
    guessed at, and a null identity is never treated as a session.
    """
    column = frame["date"]
    dtype = frame.schema["date"]
    if isinstance(dtype, pl.Datetime):
        column = column.dt.date()
    elif dtype != pl.Date:
        return None
    values = column.to_list()
    if any(value is None for value in values):
        return None
    return [value for value in values if value is not None]


def _finite_column(
    frame: pl.DataFrame, column: str
) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    """Cast one column to float, refusing any null, cast failure, or non-finite value.

    A non-strict cast turns an unparseable cell into a null, so counting
    nulls after the cast catches both an originally missing observation and
    a value that could not be read. Neither may be silently dropped.
    """
    casted = frame[column].cast(pl.Float64, strict=False)
    if casted.null_count() > 0:
        return None
    values = np.asarray(casted.to_numpy(), dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        return None
    return values


def _optional_series(
    frame: pl.DataFrame, column: str
) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    if column not in frame.columns:
        return None
    data = np.asarray(frame[column].to_numpy(), dtype=np.float64)
    if np.isnan(data).all():
        return None
    return data


def _returns(
    values: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    if len(values) < 2:
        return np.asarray([], dtype=np.float64)
    previous = values[:-1]
    current = values[1:]
    mask = previous > 0
    return current[mask] / previous[mask] - 1.0


def _ema(
    values: np.ndarray[Any, np.dtype[np.float64]], window: int
) -> np.ndarray[Any, np.dtype[np.float64]]:
    alpha = 2.0 / (window + 1.0)
    result = np.empty(len(values), dtype=np.float64)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def _rsi(closes: np.ndarray[Any, np.dtype[np.float64]], window: int) -> float:
    deltas = np.diff(closes)
    recent = deltas[-window:]
    gains = np.clip(recent, 0.0, None)
    losses = np.clip(-recent, 0.0, None)
    average_gain = float(np.mean(gains))
    average_loss = float(np.mean(losses))
    if average_gain == 0.0 and average_loss == 0.0:
        return 50.0
    if average_loss == 0.0:
        return 100.0
    relative_strength = average_gain / average_loss
    return float(100.0 - (100.0 / (1.0 + relative_strength)))


def _atr(
    highs: np.ndarray[Any, np.dtype[np.float64]],
    lows: np.ndarray[Any, np.dtype[np.float64]],
    closes: np.ndarray[Any, np.dtype[np.float64]],
    window: int,
) -> float:
    previous_close = closes[:-1]
    high = highs[1:]
    low = lows[1:]
    true_ranges = np.maximum.reduce(
        [high - low, np.abs(high - previous_close), np.abs(low - previous_close)]
    )
    return float(np.mean(true_ranges[-window:]))


def _annualized_std(returns: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    return float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS))


def _downside_deviation(returns: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    downside = np.minimum(returns, 0.0)
    return float(np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_DAYS))


def _max_drawdown(closes: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    running_peak = np.maximum.accumulate(closes)
    drawdowns = closes / running_peak - 1.0
    return float(np.min(drawdowns))


_COMMON_RISK_KEYS = (
    "annualized_volatility",
    "downside_volatility",
    "max_drawdown",
    "beta",
)


def _common_benchmark_values(
    asset: pl.DataFrame,
    benchmark: pl.DataFrame | None,
    missing: dict[str, str],
    policy: CommonRiskPolicy,
) -> dict[str, float]:
    """V3 benchmark-relative factors and exact common-window risk metrics."""
    values: dict[str, float] = {}
    if benchmark is None:
        _mark_common_risk_missing(missing, "Common-risk benchmark is omitted")
        _mark_relative_missing(missing, "Benchmark is omitted")
        return values
    if (
        "date" not in asset.columns
        or "date" not in benchmark.columns
        or "close" not in benchmark.columns
    ):
        _mark_common_risk_missing(missing, "Common risk needs benchmark date and close columns")
        _mark_relative_missing(missing, "Benchmark needs date and close columns")
        return values

    benchmark_clean = _prepare_price_frame(benchmark).rename({"close": "benchmark_close"})
    joined = (
        asset.select("date", "close")
        .join(
            benchmark_clean.select("date", "benchmark_close"),
            on="date",
            how="inner",
        )
        .sort("date")
    )
    values.update(_relative_return_values(joined, missing))

    asset_last = asset["date"][-1]
    benchmark_last = benchmark_clean["date"][-1] if benchmark_clean.height else None
    if asset_last != benchmark_last:
        _mark_common_risk_missing(
            missing,
            "Asset and benchmark latest dates do not match",
        )
        return values
    if joined.height < policy.sessions:
        _mark_common_risk_missing(
            missing,
            f"Need at least {policy.sessions} common closes",
        )
        return values

    window = joined.tail(policy.sessions)
    asset_closes = _series(window, "close")
    benchmark_closes = _series(window, "benchmark_close")
    asset_returns = _returns(asset_closes)
    benchmark_returns = _returns(benchmark_closes)
    required_returns = policy.sessions - 1
    if len(asset_returns) != required_returns or len(benchmark_returns) != required_returns:
        _mark_common_risk_missing(
            missing,
            f"Need exactly {required_returns} aligned common returns",
        )
        return values
    if not np.isfinite(asset_returns).all() or not np.isfinite(benchmark_returns).all():
        _mark_common_risk_missing(missing, "Common-window returns must be finite")
        return values

    annualization = math.sqrt(float(policy.annualization_sessions))
    volatility = float(np.std(asset_returns, ddof=1) * annualization)
    downside = float(np.sqrt(np.mean(np.minimum(asset_returns, 0.0) ** 2)) * annualization)
    drawdown = _max_drawdown(asset_closes)
    if not all(math.isfinite(value) for value in (volatility, downside, drawdown)):
        _mark_common_risk_missing(missing, "Common-window risk metrics must be finite")
        return values
    values.update(
        {
            "annualized_volatility": volatility,
            "downside_volatility": downside,
            "max_drawdown": drawdown,
        }
    )

    variance = float(np.var(benchmark_returns, ddof=1))
    if not math.isfinite(variance) or variance <= 0:
        missing["beta"] = "Benchmark return variance is zero"
        return values
    covariance = float(np.cov(asset_returns, benchmark_returns, ddof=1)[0, 1])
    beta = covariance / variance
    if math.isfinite(beta):
        values["beta"] = beta
    else:
        missing["beta"] = "Beta must be finite"
    return values


def _relative_return_values(
    joined: pl.DataFrame,
    missing: dict[str, str],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for window in (20, 63, 126, 252):
        key = f"relative_return_{window}d"
        if joined.height > window:
            asset_return = float(joined["close"][-1] / joined["close"][-window - 1] - 1.0)
            benchmark_return = float(
                joined["benchmark_close"][-1] / joined["benchmark_close"][-window - 1] - 1.0
            )
            relative = asset_return - benchmark_return
            if math.isfinite(relative):
                values[key] = relative
            else:
                missing[key] = "Relative return must be finite"
        else:
            missing[key] = f"Need {window + 1} overlapping closes"
    return values


def _mark_common_risk_missing(missing: dict[str, str], reason: str) -> None:
    for key in _COMMON_RISK_KEYS:
        missing[key] = reason


def _mark_relative_missing(missing: dict[str, str], reason: str) -> None:
    for window in (20, 63, 126, 252):
        missing[f"relative_return_{window}d"] = reason


def _benchmark_values(
    asset: pl.DataFrame,
    benchmark: pl.DataFrame,
    missing: dict[str, str],
) -> dict[str, float]:
    if (
        "date" not in asset.columns
        or "date" not in benchmark.columns
        or "close" not in benchmark.columns
    ):
        missing["benchmark"] = "Benchmark needs date and close columns"
        return {}
    benchmark_clean = _prepare_price_frame(benchmark).rename({"close": "benchmark_close"})
    joined = asset.select("date", "close").join(
        benchmark_clean.select("date", "benchmark_close"),
        on="date",
        how="inner",
    )
    if joined.height < 3:
        missing["beta"] = "Need at least three overlapping benchmark closes"
        return {}
    asset_returns = _returns(_series(joined, "close"))
    benchmark_returns = _returns(_series(joined, "benchmark_close"))
    values: dict[str, float] = {}
    if len(asset_returns) >= 2 and len(benchmark_returns) >= 2:
        variance = float(np.var(benchmark_returns, ddof=1))
        if variance > 0:
            covariance = float(np.cov(asset_returns, benchmark_returns, ddof=1)[0, 1])
            values["beta"] = covariance / variance
        else:
            missing["beta"] = "Benchmark return variance is zero"
    for window in (20, 63, 126, 252):
        if joined.height > window:
            asset_return = float(joined["close"][-1] / joined["close"][-window - 1] - 1.0)
            benchmark_return = float(
                joined["benchmark_close"][-1] / joined["benchmark_close"][-window - 1] - 1.0
            )
            values[f"relative_return_{window}d"] = asset_return - benchmark_return
        else:
            missing[f"relative_return_{window}d"] = f"Need {window + 1} overlapping closes"
    return values


def last_observation_date(result: IndicatorResult) -> date | None:
    raw = result.last_date
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str):
        return date.fromisoformat(raw[:10])
    return None

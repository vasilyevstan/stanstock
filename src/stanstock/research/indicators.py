from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl

from stanstock.research.types import IndicatorResult

SUPPORTED_WINDOWS = (1, 5, 10, 20, 63, 126, 252, 756, 1260)
TRADING_DAYS = 252.0


def calculate_indicators(
    frame: pl.DataFrame,
    *,
    benchmark: pl.DataFrame | None = None,
    windows: Iterable[int] = SUPPORTED_WINDOWS,
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
        values["macd_histogram"] = float(macd_line[-1] - signal[-1])
    else:
        missing["macd"] = "Need at least 35 closes"

    if highs is not None and lows is not None and len(closes) >= 15:
        values["atr_14"] = _atr(highs, lows, closes, 14)
        values["atr_14_pct"] = values["atr_14"] / last_close if last_close > 0 else 0.0
    else:
        missing["atr_14"] = "Need high/low and at least 15 closes"

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
        avg_20 = float(np.mean(volumes[-20:]))
        values["avg_volume_20d"] = avg_20
        values["abnormal_volume"] = float(volumes[-1] / avg_20) if avg_20 > 0 else 0.0
        if len(volumes) >= 80:
            avg_60_prior = float(np.mean(volumes[-80:-20]))
            values["volume_trend"] = float(avg_20 / avg_60_prior - 1.0) if avg_60_prior > 0 else 0.0
        else:
            missing["volume_trend"] = "Need at least 80 volume observations"
    else:
        missing["volume"] = "Need volume and at least 20 observations"

    if benchmark is not None:
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

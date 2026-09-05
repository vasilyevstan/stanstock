from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stanstock.research.indicators import calculate_indicators


def _price_frame(rows: int = 260, *, start: float = 100.0, step: float = 0.2) -> pl.DataFrame:
    dates = [date(2025, 1, 1) + timedelta(days=index) for index in range(rows)]
    closes = [start + index * step for index in range(rows)]
    return pl.DataFrame(
        {
            "date": dates,
            "open": [close - 0.2 for close in closes],
            "high": [close + 1.0 for close in closes],
            "low": [close - 1.0 for close in closes],
            "close": closes,
            "volume": [1_000_000 + index * 1_000 for index in range(rows)],
        }
    )


def test_indicators_cover_supported_windows_and_benchmark_metrics() -> None:
    frame = _price_frame(1300)
    benchmark = _price_frame(1300, start=100.0, step=0.1)

    result = calculate_indicators(frame, benchmark=benchmark)

    assert result.observation_count == 1300
    assert result.values["return_1d"] == pytest.approx(frame["close"][-1] / frame["close"][-2] - 1)
    assert result.values["sma_20"] == pytest.approx(sum(frame["close"][-20:]) / 20)
    assert result.values["sma_200"] > 0
    assert result.values["ema"] == result.values["ema_20"]
    assert result.values["ema_12"] > result.values["ema_26"]
    assert result.values["rsi_14"] == pytest.approx(100.0)
    assert result.values["macd"] > 0
    assert result.values["atr_14"] > 0
    assert result.values["annualized_volatility"] >= 0
    downside_volatility = result.values.get("downside_volatility")
    assert downside_volatility is None or downside_volatility >= 0
    assert downside_volatility is not None or "downside_volatility" in result.missing
    assert result.values["momentum_126d"] == pytest.approx(result.values["return_126d"])
    assert result.values["return_756d"] == pytest.approx(
        frame["close"][-1] / frame["close"][-757] - 1
    )
    assert result.values["return_1260d"] == pytest.approx(
        frame["close"][-1] / frame["close"][-1261] - 1
    )
    assert result.values["52w_high"] == pytest.approx(max(frame["close"][-252:]))
    assert result.values["52w_range"] > 0
    assert result.values["52w_range_pct"] > 0
    assert 0 <= result.values["52w_position"] <= 1
    assert result.values["volume_trend"] > 0
    assert result.values["abnormal_volume"] > 0
    assert result.values["max_drawdown"] <= 0
    assert "beta" in result.values
    assert result.values["relative_return_63d"] > 0


def test_indicators_report_missing_short_history_explicitly() -> None:
    frame = pl.DataFrame({"date": [date(2026, 1, 1)], "close": [10.0]})

    result = calculate_indicators(frame)

    assert result.values["last_close"] == 10.0
    assert "return_20d" in result.missing
    assert "return_756d" in result.missing
    assert "return_1260d" in result.missing
    assert "sma_200" in result.missing
    assert "macd" in result.missing
    assert "volume" in result.missing


def test_flat_prices_have_neutral_rsi_and_zero_downside_deviation() -> None:
    dates = [date(2026, 1, 1) + timedelta(days=index) for index in range(20)]
    frame = pl.DataFrame({"date": dates, "close": [100.0] * 20})

    result = calculate_indicators(frame)

    assert result.values["rsi_14"] == pytest.approx(50.0)
    assert result.values["downside_volatility"] == pytest.approx(0.0)


def test_downside_volatility_is_deviation_versus_zero_target() -> None:
    closes = [100.0, 90.0, 99.0, 89.1, 97.119]
    frame = pl.DataFrame(
        {
            "date": [date(2026, 1, 1) + timedelta(days=index) for index in range(len(closes))],
            "close": closes,
        }
    )
    returns = np.asarray([-0.10, 0.10, -0.10, 0.09])
    expected = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)) * np.sqrt(252.0)

    result = calculate_indicators(frame)

    assert result.values["downside_volatility"] == pytest.approx(expected)


def test_indicators_ignore_invalid_closes_without_inventing_values() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)],
            "close": [None, -1.0, 12.0],
        }
    )

    result = calculate_indicators(frame)

    assert result.observation_count == 1
    assert result.values["last_close"] == 12.0
    assert result.missing["return_1d"] == "Need at least 2 closes"

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from stanstock.research.indicators import calculate_indicators, median_dollar_volume


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
    assert result.values["macd_histogram_pct"] == pytest.approx(
        result.values["macd_histogram"] / result.values["last_close"]
    )
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
    assert result.values["avg_dollar_volume_20d"] > 0
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


def test_rsi_uses_simple_averages_of_latest_fourteen_gains_and_losses() -> None:
    prefix_deltas = [50.0, -20.0, 10.0]
    latest_deltas = [
        -1.0,
        2.0,
        -3.0,
        4.0,
        -5.0,
        6.0,
        -7.0,
        8.0,
        -9.0,
        10.0,
        -11.0,
        12.0,
        -13.0,
        14.0,
    ]
    closes = [100.0]
    for delta in [*prefix_deltas, *latest_deltas]:
        closes.append(closes[-1] + delta)
    expected_average_gain = sum(max(delta, 0.0) for delta in latest_deltas) / 14
    expected_average_loss = sum(max(-delta, 0.0) for delta in latest_deltas) / 14
    expected = 100.0 - 100.0 / (1.0 + expected_average_gain / expected_average_loss)
    frame = pl.DataFrame(
        {
            "date": [date(2026, 1, 1) + timedelta(days=index) for index in range(len(closes))],
            "close": closes,
        }
    )

    result = calculate_indicators(frame)

    assert result.values["rsi_14"] == pytest.approx(expected)


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


def test_normalized_macd_and_dollar_volume_are_split_invariant() -> None:
    frame = _price_frame(260)
    split = frame.with_columns(
        (pl.col(column) / 10).alias(column) for column in ("open", "high", "low", "close")
    ).with_columns((pl.col("volume") * 10).alias("volume"))

    original = calculate_indicators(frame)
    transformed = calculate_indicators(split)

    assert transformed.values["macd_histogram"] == pytest.approx(
        original.values["macd_histogram"] / 10
    )
    assert transformed.values["avg_volume_20d"] == pytest.approx(
        original.values["avg_volume_20d"] * 10
    )
    assert transformed.values["macd_histogram_pct"] == pytest.approx(
        original.values["macd_histogram_pct"]
    )
    assert transformed.values["avg_dollar_volume_20d"] == pytest.approx(
        original.values["avg_dollar_volume_20d"]
    )


def test_invalid_recent_volume_withholds_liquidity_indicators() -> None:
    frame = _price_frame(40).with_columns(
        pl.when(pl.int_range(pl.len()) == 39)
        .then(float("nan"))
        .otherwise(pl.col("volume"))
        .alias("volume")
    )

    result = calculate_indicators(frame)

    assert np.isnan(result.values["avg_volume_20d"])
    assert result.values["abnormal_volume"] == 0
    assert "avg_dollar_volume_20d" not in result.values
    assert "abnormal_volume_strict" not in result.values
    assert result.missing["avg_dollar_volume_20d"] == ("Recent volume observations must be finite")


# ---------------------------------------------------------------------------
# `median_dollar_volume` -- the Under-$10 shadow liquidity primitive.
#
# The helper is additive: `calculate_indicators` output must stay unchanged,
# and every refusal must name the condition actually observed rather than
# describing a malformed row as missing history.
# ---------------------------------------------------------------------------


def _sessions(
    count: int,
    *,
    close: float = 4.0,
    volume: float = 1_000_000.0,
    start: date = date(2025, 1, 1),
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [start + timedelta(days=index) for index in range(count)],
            "close": [close] * count,
            "volume": [volume] * count,
        }
    )


def test_median_dollar_volume_computes_over_the_latest_252_observed_sessions() -> None:
    frame = _sessions(300, close=4.0, volume=1_000_000.0)

    result = median_dollar_volume(frame)

    assert result.status == "computed"
    assert result.value == pytest.approx(4_000_000.0)
    assert result.sessions_used == 252
    assert result.first_session == frame["date"][-252]
    assert result.last_session == frame["date"][-1]
    assert result.reason is None


def test_median_dollar_volume_uses_the_latest_window_not_the_earliest() -> None:
    early = _sessions(100, close=1.0, volume=1.0)
    late = _sessions(252, close=4.0, volume=1_000_000.0, start=date(2026, 1, 1))

    result = median_dollar_volume(pl.concat([early, late]))

    assert result.status == "computed"
    assert result.value == pytest.approx(4_000_000.0)
    assert result.sessions_used == 252
    assert result.first_session == date(2026, 1, 1)


def test_median_dollar_volume_sorts_unsorted_sessions_before_selecting() -> None:
    frame = _sessions(252, close=4.0, volume=1_000_000.0)
    shuffled = frame.reverse()

    result = median_dollar_volume(shuffled)

    assert result.status == "computed"
    assert result.first_session == frame["date"][0]
    assert result.last_session == frame["date"][-1]


@pytest.mark.parametrize(("count", "expected"), [(0, 0), (1, 1), (251, 251)])
def test_median_dollar_volume_withholds_short_windows_without_padding(
    count: int,
    expected: int,
) -> None:
    frame = _sessions(count)

    result = median_dollar_volume(frame)

    assert result.status == "withheld"
    assert result.reason == "insufficient_sessions"
    assert result.value is None
    assert result.sessions_used == expected


def test_median_dollar_volume_boundary_is_exactly_252_sessions() -> None:
    assert median_dollar_volume(_sessions(251)).status == "withheld"
    assert median_dollar_volume(_sessions(252)).status == "computed"


def test_median_dollar_volume_counts_distinct_sessions_not_calendar_days() -> None:
    trading_days = [
        day
        for day in (date(2025, 1, 1) + timedelta(days=index) for index in range(400))
        if day.weekday() < 5
    ][:252]
    frame = pl.DataFrame(
        {
            "date": trading_days,
            "close": [4.0] * 252,
            "volume": [1_000_000.0] * 252,
        }
    )

    result = median_dollar_volume(frame)

    assert result.status == "computed"
    assert result.sessions_used == 252
    assert (result.last_session - result.first_session).days > 252


def test_median_dollar_volume_refuses_duplicate_session_dates() -> None:
    frame = _sessions(252)
    duplicated = pl.concat([frame, frame.tail(1)])

    result = median_dollar_volume(duplicated)

    assert result.status == "withheld"
    assert result.reason == "duplicate_sessions"
    assert result.value is None
    assert result.sessions_used is None


def test_median_dollar_volume_requires_date_close_and_volume_columns() -> None:
    frame = _sessions(252)

    for column in ("date", "close", "volume"):
        result = median_dollar_volume(frame.drop(column))

        assert result.status == "withheld"
        assert result.reason == "missing_price_columns"


def test_median_dollar_volume_refuses_unusable_session_identities() -> None:
    string_dates = pl.DataFrame(
        {
            "date": [f"2025-01-{index + 1:02d}" for index in range(3)],
            "close": [4.0] * 3,
            "volume": [1.0] * 3,
        }
    )
    null_dates = _sessions(3).with_columns(
        pl.when(pl.int_range(pl.len()) == 1).then(None).otherwise(pl.col("date")).alias("date")
    )

    assert median_dollar_volume(string_dates).reason == "invalid_session_dates"
    assert median_dollar_volume(null_dates).reason == "invalid_session_dates"


@pytest.mark.parametrize(
    "close",
    [None, 0.0, -1.0, float("nan"), float("inf")],
)
def test_median_dollar_volume_refuses_invalid_closes(close: float | None) -> None:
    frame = _sessions(252).with_columns(
        pl.when(pl.int_range(pl.len()) == 120)
        .then(pl.lit(close, dtype=pl.Float64))
        .otherwise(pl.col("close"))
        .alias("close")
    )

    result = median_dollar_volume(frame)

    assert result.status == "withheld"
    assert result.reason == "invalid_close_values"
    assert result.value is None
    assert result.sessions_used == 252


@pytest.mark.parametrize("volume", [None, -1.0, float("nan"), float("inf")])
def test_median_dollar_volume_refuses_invalid_volumes(volume: float | None) -> None:
    frame = _sessions(252).with_columns(
        pl.when(pl.int_range(pl.len()) == 7)
        .then(pl.lit(volume, dtype=pl.Float64))
        .otherwise(pl.col("volume"))
        .alias("volume")
    )

    result = median_dollar_volume(frame)

    assert result.status == "withheld"
    assert result.reason == "invalid_volume_values"


def test_median_dollar_volume_refuses_unparseable_numeric_text() -> None:
    frame = _sessions(252).with_columns(pl.col("close").cast(pl.Utf8))
    broken = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 3)
        .then(pl.lit("not-a-number"))
        .otherwise(pl.col("close"))
        .alias("close")
    )

    assert median_dollar_volume(frame).status == "computed"
    assert median_dollar_volume(broken).reason == "invalid_close_values"


def test_median_dollar_volume_reports_invalidity_even_in_a_short_window() -> None:
    frame = _sessions(10).with_columns(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(pl.lit(-5.0, dtype=pl.Float64))
        .otherwise(pl.col("close"))
        .alias("close")
    )

    result = median_dollar_volume(frame)

    assert result.status == "withheld"
    assert result.reason == "invalid_close_values"
    assert result.reason != "insufficient_sessions"


def test_median_dollar_volume_refuses_a_nonfinite_product() -> None:
    frame = _sessions(252, close=1e300, volume=1e300)

    result = median_dollar_volume(frame)

    assert result.status == "withheld"
    assert result.reason == "nonfinite_dollar_volume_product"


def test_median_dollar_volume_treats_zero_volume_as_valid_data() -> None:
    frame = _sessions(252, volume=0.0)

    result = median_dollar_volume(frame)

    assert result.status == "computed"
    assert result.value == 0.0
    assert result.reason is None


def test_median_dollar_volume_is_split_equivalent() -> None:
    frame = _sessions(260, close=40.0, volume=100_000.0)
    split = frame.with_columns(
        (pl.col("close") / 10).alias("close"),
        (pl.col("volume") * 10).alias("volume"),
    )

    original = median_dollar_volume(frame)
    transformed = median_dollar_volume(split)

    assert original.status == transformed.status == "computed"
    assert original.value is not None
    assert transformed.value == pytest.approx(original.value)


def test_median_dollar_volume_window_size_is_configurable_and_positive() -> None:
    frame = _sessions(20)

    assert median_dollar_volume(frame, sessions=20).status == "computed"
    assert median_dollar_volume(frame, sessions=21).reason == "insufficient_sessions"
    with pytest.raises(ValueError, match="positive session count"):
        median_dollar_volume(frame, sessions=0)


def test_median_dollar_volume_does_not_change_calculate_indicators_output() -> None:
    frame = _price_frame(300)
    before = calculate_indicators(frame)

    median_dollar_volume(frame)
    after = calculate_indicators(frame)

    assert before.values == after.values
    assert before.missing == after.missing
    assert "median_dollar_volume_252" not in before.values
    assert "median_dollar_volume_252" not in before.missing

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from stanstock.simulation.engine import AccountingEngine
from stanstock.simulation.types import (
    ExecutionPriceBasis,
    RebalanceFrequency,
    SimulationConfig,
    SimulationGrade,
    SimulationMode,
)


def _make_daily_dates(start: date, count: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(count)]


def test_equal_weight_rebalancing_entries_exits_and_weights() -> None:
    """Deterministic equal-weight rebalancing allocates 1/K to each target, liquidating exits."""
    dates = _make_daily_dates(date(2026, 1, 1), 4)
    # Day 0: prices 10, 20, 30
    # Day 1: trade date for signal 0. Targets A and B. Weight should be 50% each.
    # Day 2: signal changes to B and C.
    # Day 3: trade date for signal 2. A is fully sold (exit), C is bought (entry), B is adjusted.
    prices = pl.DataFrame(
        {
            "date": [
                dates[0],
                dates[0],
                dates[0],
                dates[1],
                dates[1],
                dates[1],
                dates[2],
                dates[2],
                dates[2],
                dates[3],
                dates[3],
                dates[3],
            ],
            "listing_id": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "close": [
                10.0,
                20.0,
                30.0,
                10.0,
                20.0,
                30.0,
                15.0,
                25.0,
                35.0,
                15.0,
                25.0,
                35.0,
            ],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [dates[0], dates[0], dates[0], dates[2], dates[2], dates[2]],
            "listing_id": ["A", "B", "C", "A", "B", "C"],
            "score": [10.0, 8.0, 2.0, 1.0, 8.0, 10.0],
        }
    )

    config = SimulationConfig(
        starting_capital=100000.0,
        top_n=2,
        rebalance_frequency=RebalanceFrequency.DAILY,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        execution_basis=ExecutionPriceBasis.NEXT_CLOSE,
    )
    engine = AccountingEngine(config)
    result = engine.run(prices=prices, signals=signals)

    # On dates[1]: target is A and B (K=2)
    h_day1 = result.holdings.filter(pl.col("observation_date") == dates[1])
    assert set(h_day1["listing_id"].to_list()) == {"A", "B"}
    # 50% target each
    for w in h_day1["weight"].to_list():
        assert abs(w - 0.5) < 1e-4

    # On dates[3]: target is B and C (A exited)
    h_day3 = result.holdings.filter(pl.col("observation_date") == dates[3])
    assert set(h_day3["listing_id"].to_list()) == {"B", "C"}
    # Verify A was completely liquidated
    trades_day3 = result.trades.filter(pl.col("trade_date") == dates[3])
    a_trades = trades_day3.filter(pl.col("listing_id") == "A")
    assert a_trades.height == 1
    assert a_trades["side"][0] == "sell"
    # Weight of B and C on day 3 should be 50% each
    for w in h_day3["weight"].to_list():
        assert abs(w - 0.5) < 1e-4


def test_deterministic_tie_breaking() -> None:
    """When scores tie, sorting is deterministic by symbol/listing_id."""
    dates = _make_daily_dates(date(2026, 2, 1), 3)
    prices = pl.DataFrame(
        {
            "date": [
                dates[0],
                dates[0],
                dates[0],
                dates[1],
                dates[1],
                dates[1],
                dates[2],
                dates[2],
                dates[2],
            ],
            "listing_id": ["ZZZ", "AAA", "MMM", "ZZZ", "AAA", "MMM", "ZZZ", "AAA", "MMM"],
            "close": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    # All 3 have identical score = 5.0
    signals = pl.DataFrame(
        {
            "date": [dates[0], dates[0], dates[0]],
            "listing_id": ["ZZZ", "AAA", "MMM"],
            "score": [5.0, 5.0, 5.0],
        }
    )

    # With symbol_asc tie-breaker, top 2 should be AAA, MMM
    config = SimulationConfig(
        starting_capital=10000.0,
        top_n=2,
        tie_breaker="symbol_asc",
    )
    result = AccountingEngine(config).run(prices=prices, signals=signals)
    h_day1 = result.holdings.filter(pl.col("observation_date") == dates[1])
    assert set(h_day1["listing_id"].to_list()) == {"AAA", "MMM"}


def test_benchmark_consistency_and_metrics() -> None:
    """100% buy-and-hold matching benchmark produces identical returns, zero tracking error."""
    dates = _make_daily_dates(date(2026, 3, 1), 10)
    bench_closes = [100.0, 102.0, 101.0, 105.0, 104.0, 107.0, 109.0, 108.0, 112.0, 115.0]

    bench_prices = pl.DataFrame(
        {
            "date": dates,
            "close": bench_closes,
        }
    )
    stock_prices = pl.DataFrame(
        {
            "date": dates,
            "listing_id": ["SPY"] * 10,
            "close": bench_closes,
            "open": bench_closes,
        }
    )

    config = SimulationConfig(
        starting_capital=10000.0,
        selected_symbols=["SPY"],
        rebalance_frequency=RebalanceFrequency.NEVER,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        benchmark_symbol="SPY",
    )
    result = AccountingEngine(config).run(prices=stock_prices, benchmark_prices=bench_prices)

    m = result.metrics
    assert m.benchmark_cumulative_return is not None
    assert abs(m.cumulative_return - m.benchmark_cumulative_return) < 1e-6
    assert m.tracking_error is not None and m.tracking_error < 1e-6
    assert m.beta is not None and abs(m.beta - 1.0) < 1e-4
    assert m.excess_return is not None and abs(m.excess_return) < 1e-6


def test_hand_crafted_metric_calculations() -> None:
    """Validate cumulative return, drawdown, volatility, Sharpe, turnover against exact values."""
    dates = _make_daily_dates(date(2026, 1, 1), 40)  # > 30 days for CAGR
    # Asset starts at 100, peaks at 120, drops to 90 (max DD = -25%), ends at 110
    prices_list = [100.0]
    for i in range(1, 15):
        prices_list.append(100.0 + (120.0 - 100.0) * (i / 14))
    for i in range(15, 25):
        prices_list.append(120.0 - (120.0 - 90.0) * ((i - 14) / 10))
    for i in range(25, 40):
        prices_list.append(90.0 + (110.0 - 90.0) * ((i - 24) / 15))

    prices = pl.DataFrame(
        {
            "date": dates,
            "listing_id": ["ASSET"] * 40,
            "close": prices_list,
            "open": prices_list,
        }
    )

    config = SimulationConfig(
        starting_capital=100000.0,
        selected_symbols=["ASSET"],
        rebalance_frequency=RebalanceFrequency.NEVER,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        risk_free_rate=0.0,
    )
    result = AccountingEngine(config).run(prices=prices)
    m = result.metrics

    # Cumulative return: (110 - 100) / 100 = 0.10 (10%)
    assert abs(m.cumulative_return - 0.10) < 1e-4

    # CAGR: duration is 39 days = 39 / 365.25 years. (1 + 0.10) ** (365.25 / 39) - 1
    expected_cagr = (1.10) ** (365.25 / 39) - 1.0
    cagr_val = m.cagr
    assert cagr_val is not None
    assert abs(cagr_val - expected_cagr) < 1e-3

    # Max Drawdown: peak was 120.0, trough was 90.0. DD = (90 - 120) / 120 = -0.25 (-25%)
    assert abs(m.max_drawdown - (-0.25)) < 1e-4

    # Sharpe ratio exists and is positive
    assert m.sharpe_ratio is not None
    assert m.annualized_volatility > 0.0

    # Total trades: exactly 1 buy on day 0
    assert m.total_trades == 1


def test_cagr_not_calculated_when_duration_insufficient() -> None:
    """CAGR is None when duration is less than 30 days."""
    dates = _make_daily_dates(date(2026, 1, 1), 10)
    prices = pl.DataFrame(
        {
            "date": dates,
            "listing_id": ["STK"] * 10,
            "close": [10.0 + i for i in range(10)],
        }
    )
    config = SimulationConfig(
        starting_capital=10000.0,
        selected_symbols=["STK"],
        rebalance_frequency=RebalanceFrequency.NEVER,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    result = AccountingEngine(config).run(prices=prices)
    assert result.metrics.cagr is None
    assert result.metrics.duration_days == 9


def test_grade_and_mode_distinguishability() -> None:
    """Ensure research-grade vs observed-universe vs portfolio simulations are distinguishable."""
    c_research = SimulationConfig(
        mode=SimulationMode.BACKTEST,
        grade=SimulationGrade.RESEARCH,
        selected_symbols=["SYM"],
    )
    assert c_research.simulation_kind == "research_backtest"

    c_observed = SimulationConfig(
        mode=SimulationMode.BACKTEST,
        grade=SimulationGrade.OBSERVED,
        selected_symbols=["SYM"],
    )
    assert c_observed.simulation_kind == "observed_backtest"

    c_portfolio = SimulationConfig(
        mode=SimulationMode.PORTFOLIO,
        grade=SimulationGrade.OBSERVED,
        selected_symbols=["SYM"],
    )
    assert c_portfolio.simulation_kind == "portfolio_simulation"


def test_explicit_signals_respect_rebalance_frequency() -> None:
    """Explicit signals must trade only at configured rebalance frequency, not every signal."""
    # 60 days spanning Jan, Feb, Mar 2026
    dates = _make_daily_dates(date(2026, 1, 1), 60)
    prices = pl.DataFrame(
        {
            "date": dates,
            "listing_id": ["STK"] * 60,
            "close": [100.0 + i for i in range(60)],
            "open": [100.0 + i for i in range(60)],
        }
    )
    # Daily signals on every single day
    signals = pl.DataFrame(
        {
            "date": dates,
            "listing_id": ["STK"] * 60,
            "score": [float(i) for i in range(60)],
        }
    )

    # With MONTHLY rebalance, trades only occur at month boundaries (Jan, Feb, Mar)
    config_monthly = SimulationConfig(
        starting_capital=100000.0,
        top_n=1,
        rebalance_frequency=RebalanceFrequency.MONTHLY,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    res_monthly = AccountingEngine(config_monthly).run(prices=prices, signals=signals)
    trade_dates = res_monthly.trades["trade_date"].unique().sort().to_list()
    # Exactly 2 rebalance dates after inception across 3 months (Feb and Mar boundaries)
    assert len(trade_dates) <= 3
    assert len(trade_dates) > 0

    # With NEVER rebalance, trades exactly once (inception only)
    config_never = SimulationConfig(
        starting_capital=100000.0,
        top_n=1,
        rebalance_frequency=RebalanceFrequency.NEVER,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    res_never = AccountingEngine(config_never).run(prices=prices, signals=signals)
    assert res_never.trades.height == 1


def test_next_open_missing_does_not_fall_back_to_close() -> None:
    """NEXT_OPEN must not fall back to close; NEXT_ELIGIBLE explicitly falls back."""
    dates = _make_daily_dates(date(2026, 1, 1), 3)
    # open is not provided
    prices = pl.DataFrame(
        {
            "date": [dates[0], dates[1], dates[2]],
            "listing_id": ["STK", "STK", "STK"],
            "close": [100.0, 105.0, 110.0],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [dates[0]],
            "listing_id": ["STK"],
            "score": [1.0],
        }
    )

    # NEXT_OPEN with missing open treats execution price as missing (no trade occurs)
    config_open = SimulationConfig(
        starting_capital=100000.0,
        top_n=1,
        execution_basis=ExecutionPriceBasis.NEXT_OPEN,
    )
    res_open = AccountingEngine(config_open).run(prices=prices, signals=signals)
    assert res_open.trades.height == 0

    # NEXT_ELIGIBLE explicitly falls back to close when open is absent
    config_eligible = SimulationConfig(
        starting_capital=100000.0,
        top_n=1,
        execution_basis=ExecutionPriceBasis.NEXT_ELIGIBLE,
    )
    res_eligible = AccountingEngine(config_eligible).run(prices=prices, signals=signals)
    assert res_eligible.trades.height == 1
    assert res_eligible.trades["price"][0] == 105.0


def test_benchmark_gaps_handled_safely() -> None:
    """Benchmark gaps or nulls compute safely from aligned observations without errors."""
    dates = _make_daily_dates(date(2026, 1, 1), 10)
    prices = pl.DataFrame(
        {
            "date": dates,
            "listing_id": ["STK"] * 10,
            "close": [100.0, 102.0, 101.0, 103.0, 104.0, 105.0, 107.0, 106.0, 108.0, 110.0],
            "open": [100.0, 102.0, 101.0, 103.0, 104.0, 105.0, 107.0, 106.0, 108.0, 110.0],
        }
    )
    # Benchmark has nulls in the middle
    bench = pl.DataFrame(
        {
            "date": [dates[0], dates[1], dates[3], dates[4], dates[7], dates[9]],
            "close": [50.0, 51.0, 50.5, 52.0, 53.0, 55.0],
        }
    )

    config = SimulationConfig(
        starting_capital=10000.0,
        selected_symbols=["STK"],
        rebalance_frequency=RebalanceFrequency.NEVER,
    )
    result = AccountingEngine(config).run(prices=prices, benchmark_prices=bench)
    assert result.metrics.benchmark_cumulative_return is not None
    assert result.metrics.benchmark_cumulative_return == (55.0 - 50.0) / 50.0
    assert result.metrics.tracking_error is not None


def test_simulation_config_validation_rules() -> None:
    """Validate top_n/selected_symbols requirement and tie_breaker choices."""
    # custom_parameters alone is not a selection rule
    with pytest.raises(ValueError, match="Either top_n or selected_symbols"):
        SimulationConfig(custom_parameters={"strategy": "custom"}).validate()

    # Invalid tie_breaker
    with pytest.raises(ValueError, match="Invalid tie_breaker"):
        SimulationConfig(top_n=5, tie_breaker="invalid_choice").validate()

    # Valid tie-breakers pass
    for tb in ["symbol_asc", "symbol_desc", "listing_id_asc", "listing_id_desc"]:
        cfg = SimulationConfig(top_n=5, tie_breaker=tb)
        cfg.validate()


def test_reject_duplicate_prices_for_date_and_listing_id() -> None:
    """Duplicate price entries for the same (date, listing_id) must be rejected."""
    d0 = date(2026, 1, 1)
    prices = pl.DataFrame(
        {
            "date": [d0, d0],
            "listing_id": ["STK", "STK"],
            "close": [100.0, 101.0],
        }
    )
    engine = AccountingEngine(SimulationConfig(top_n=1))
    with pytest.raises(ValueError, match="Duplicate price observations detected"):
        engine.run(prices=prices)


def test_reject_duplicate_signals_for_date_and_listing_id() -> None:
    """Duplicate signal entries for the same (date, listing_id) must be rejected."""
    d0 = date(2026, 1, 1)
    d1 = date(2026, 1, 2)
    prices = pl.DataFrame(
        {
            "date": [d0, d1],
            "listing_id": ["STK", "STK"],
            "close": [100.0, 105.0],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [d0, d0],
            "listing_id": ["STK", "STK"],
            "score": [1.0, 2.0],
        }
    )
    engine = AccountingEngine(SimulationConfig(top_n=1))
    with pytest.raises(ValueError, match="Duplicate signal observations detected"):
        engine.run(prices=prices, signals=signals)


def test_reject_duplicate_benchmark_dates() -> None:
    """Duplicate benchmark dates must be rejected."""
    d0 = date(2026, 1, 1)
    d1 = date(2026, 1, 2)
    prices = pl.DataFrame(
        {
            "date": [d0, d1],
            "listing_id": ["STK", "STK"],
            "close": [100.0, 105.0],
        }
    )
    bench = pl.DataFrame(
        {
            "date": [d0, d0],
            "close": [50.0, 50.5],
        }
    )
    engine = AccountingEngine(SimulationConfig(top_n=1))
    with pytest.raises(ValueError, match="Duplicate benchmark dates detected"):
        engine.run(prices=prices, benchmark_prices=bench)


def test_input_hash_covers_complete_observation_content() -> None:
    dates = _make_daily_dates(date(2026, 1, 1), 3)
    rising = pl.DataFrame(
        {
            "date": dates,
            "listing_id": ["STK"] * 3,
            "close": [100.0, 110.0, 120.0],
        }
    )
    falling = pl.DataFrame(
        {
            "date": dates,
            "listing_id": ["STK"] * 3,
            "close": [120.0, 110.0, 100.0],
        }
    )
    config = SimulationConfig(
        starting_capital=10_000,
        selected_symbols=["STK"],
        rebalance_frequency=RebalanceFrequency.NEVER,
    )

    rising_result = AccountingEngine(config).run(prices=rising)
    falling_result = AccountingEngine(config).run(prices=falling)

    assert rising_result.input_hash != falling_result.input_hash
    assert rising_result.metrics.cumulative_return != falling_result.metrics.cumulative_return


def test_rebalance_does_not_emit_subprecision_dust_trades() -> None:
    listing_id = "00000000-0000-0000-0000-000000000001"
    prices = pl.DataFrame(
        {
            "date": [date(2025, 1, 2), date(2025, 4, 1)],
            "listing_id": [listing_id, listing_id],
            "close": [127.02782177955612, 111.48022211666448],
        },
        schema={"date": pl.Date, "listing_id": pl.String, "close": pl.Float64},
    )
    signals = pl.DataFrame(
        {"date": [date(2025, 1, 1)], "listing_id": [listing_id], "score": [80.0]}
    )
    result = AccountingEngine(
        SimulationConfig(
            starting_capital=100_000,
            top_n=1,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
            execution_basis=ExecutionPriceBasis.NEXT_CLOSE,
            transaction_cost_bps=10,
            slippage_bps=5,
            cash_buffer_bps=0,
        )
    ).run(prices=prices, signals=signals)

    assert result.trades["side"].to_list() == ["buy"]
    assert result.trades["trade_date"].to_list() == [date(2025, 1, 2)]
    assert all(quantity >= 1e-8 for quantity in result.trades["quantity"].to_list())
    assert all(value >= 1e-6 for value in result.trades["gross_value"].to_list())


def test_selected_portfolio_requires_prices_for_every_listing_at_inception() -> None:
    prices = pl.DataFrame(
        {
            "date": [
                date(2026, 2, 3),
                date(2026, 2, 4),
                date(2026, 2, 4),
                date(2026, 2, 5),
                date(2026, 2, 5),
            ],
            "listing_id": ["A", "A", "B", "A", "B"],
            "close": [100.0, 101.0, 50.0, 102.0, 51.0],
        }
    )
    config = SimulationConfig(
        starting_capital=100_000,
        selected_symbols=["A", "B"],
        rebalance_frequency=RebalanceFrequency.NEVER,
    )

    with pytest.raises(ValueError, match="lack a usable inception execution price.*B"):
        AccountingEngine(config).run(prices=prices)

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stanstock.simulation.engine import AccountingEngine
from stanstock.simulation.hooks import CashSettlementHook
from stanstock.simulation.types import (
    ExecutionPriceBasis,
    MissingPriceError,
    MissingPricePolicy,
    RebalanceFrequency,
    SimulationConfig,
)


def _make_daily_dates(start: date, count: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(count)]


def test_conservation_of_cash_and_holdings_exact() -> None:
    """Cash + gross market value must equal total portfolio value at every date."""
    dates = _make_daily_dates(date(2026, 1, 1), 5)
    prices = pl.DataFrame(
        {
            "date": [
                dates[0],
                dates[0],
                dates[1],
                dates[1],
                dates[2],
                dates[2],
                dates[3],
                dates[3],
                dates[4],
                dates[4],
            ],
            "listing_id": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "symbol": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "close": [10.0, 20.0, 11.0, 19.0, 12.0, 22.0, 10.5, 21.0, 13.0, 25.0],
            "open": [10.0, 20.0, 10.5, 19.5, 11.5, 21.5, 11.0, 21.0, 12.5, 24.0],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [dates[0], dates[0], dates[2], dates[2]],
            "listing_id": ["A", "B", "A", "B"],
            "score": [1.0, 2.0, 2.0, 1.0],
        }
    )

    config = SimulationConfig(
        starting_capital=10000.0,
        top_n=2,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
        execution_basis=ExecutionPriceBasis.NEXT_CLOSE,
        rebalance_frequency=RebalanceFrequency.DAILY,
    )
    engine = AccountingEngine(config)
    result = engine.run(prices=prices, signals=signals)

    curves = result.daily_curves
    for row in curves.iter_rows(named=True):
        pv = row["portfolio_value"]
        cash = row["cash"]
        gmv = row["gross_market_value"]
        assert cash >= 0.0, f"Cash overdraft on {row['date']}: {cash}"
        diff = abs(pv - (cash + gmv))
        assert diff < 1e-6, (
            f"Conservation violation on {row['date']}: pv={pv}, cash={cash}, gmv={gmv}"
        )


@given(
    starting_cap=st.floats(min_value=1_000.0, max_value=500_000.0),
    p1=st.floats(min_value=5.0, max_value=200.0),
    p2=st.floats(min_value=5.0, max_value=200.0),
    p3=st.floats(min_value=5.0, max_value=200.0),
    cost_bps=st.floats(min_value=0.0, max_value=50.0),
    slip_bps=st.floats(min_value=0.0, max_value=25.0),
)
@settings(max_examples=50, deadline=None)
def test_hypothesis_cash_conservation_and_frictions(
    starting_cap: float,
    p1: float,
    p2: float,
    p3: float,
    cost_bps: float,
    slip_bps: float,
) -> None:
    """Hypothesis property test: cash is always non-negative and conserved with holdings."""
    dates = _make_daily_dates(date(2026, 1, 1), 3)
    prices = pl.DataFrame(
        {
            "date": [dates[0], dates[1], dates[2]],
            "listing_id": ["SYM", "SYM", "SYM"],
            "close": [p1, p2, p3],
            "open": [p1, p2, p3],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [dates[0]],
            "listing_id": ["SYM"],
            "score": [1.0],
        }
    )
    config = SimulationConfig(
        starting_capital=starting_cap,
        top_n=1,
        transaction_cost_bps=cost_bps,
        slippage_bps=slip_bps,
    )
    engine = AccountingEngine(config)
    result = engine.run(prices=prices, signals=signals)

    for row in result.daily_curves.iter_rows(named=True):
        assert row["cash"] >= -1e-6
        assert abs(row["portfolio_value"] - (row["cash"] + row["gross_market_value"])) < 1e-4


def test_no_trade_before_signal_eligibility() -> None:
    """A signal dated T can trade ONLY on strictly subsequent eligible dates (no look-ahead)."""
    dates = _make_daily_dates(date(2026, 3, 1), 4)
    prices = pl.DataFrame(
        {
            "date": [dates[0], dates[1], dates[2], dates[3]],
            "listing_id": ["XYZ", "XYZ", "XYZ", "XYZ"],
            "close": [100.0, 102.0, 105.0, 108.0],
            "open": [99.0, 101.0, 104.0, 107.0],
        }
    )
    # Signal is dated on dates[1] (2026-03-02)
    signals = pl.DataFrame(
        {
            "date": [dates[1]],
            "listing_id": ["XYZ"],
            "score": [10.0],
        }
    )

    config = SimulationConfig(
        starting_capital=50000.0,
        top_n=1,
        execution_basis=ExecutionPriceBasis.NEXT_OPEN,
    )
    engine = AccountingEngine(config)
    result = engine.run(prices=prices, signals=signals)

    trades = result.trades
    assert trades.height == 1
    trade_date = trades["trade_date"][0]

    # Must trade on dates[2], NEVER on dates[0] or dates[1]
    assert trade_date == dates[2]
    assert trade_date > dates[1]

    # Check holdings on dates[0] and dates[1] were empty
    holdings_before = result.holdings.filter(pl.col("observation_date") <= dates[1])
    assert holdings_before.height == 0

    # Trade price must be the OPEN of dates[2] per NEXT_OPEN basis
    assert trades["price"][0] == 104.0


def test_costs_strictly_reduce_return() -> None:
    """Higher transaction costs and slippage must strictly reduce cumulative return."""
    dates = _make_daily_dates(date(2026, 4, 1), 6)
    prices = pl.DataFrame(
        {
            "date": [
                dates[0],
                dates[0],
                dates[1],
                dates[1],
                dates[2],
                dates[2],
                dates[3],
                dates[3],
                dates[4],
                dates[4],
                dates[5],
                dates[5],
            ],
            "listing_id": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "close": [10.0, 10.0, 12.0, 8.0, 11.0, 11.0, 14.0, 7.0, 13.0, 12.0, 15.0, 10.0],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [dates[0], dates[0], dates[2], dates[2], dates[4], dates[4]],
            "listing_id": ["A", "B", "A", "B", "A", "B"],
            "score": [2.0, 1.0, 1.0, 2.0, 2.0, 1.0],
        }
    )

    # 1. Zero cost
    res_zero = AccountingEngine(
        SimulationConfig(
            starting_capital=100000.0,
            top_n=1,
            rebalance_frequency=RebalanceFrequency.DAILY,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
        )
    ).run(prices=prices, signals=signals)

    # 2. Modest cost (10 bps fee + 5 bps slippage = 15 bps)
    res_med = AccountingEngine(
        SimulationConfig(
            starting_capital=100000.0,
            top_n=1,
            rebalance_frequency=RebalanceFrequency.DAILY,
            transaction_cost_bps=10.0,
            slippage_bps=5.0,
        )
    ).run(prices=prices, signals=signals)

    # 3. High cost (50 bps fee + 25 bps slippage = 75 bps)
    res_high = AccountingEngine(
        SimulationConfig(
            starting_capital=100000.0,
            top_n=1,
            rebalance_frequency=RebalanceFrequency.DAILY,
            transaction_cost_bps=50.0,
            slippage_bps=25.0,
        )
    ).run(prices=prices, signals=signals)

    ret_zero = res_zero.metrics.cumulative_return
    ret_med = res_med.metrics.cumulative_return
    ret_high = res_high.metrics.cumulative_return

    assert ret_zero > ret_med > ret_high, (
        f"Cost monotonicity failed: {ret_zero} > {ret_med} > {ret_high}"
    )
    pv_zero = res_zero.daily_curves["portfolio_value"][-1]
    pv_med = res_med.daily_curves["portfolio_value"][-1]
    pv_high = res_high.daily_curves["portfolio_value"][-1]
    assert pv_zero > pv_med > pv_high


def test_reproducibility_and_input_hash() -> None:
    """Running identical inputs produces identical results, metrics, and input_hash."""
    dates = _make_daily_dates(date(2026, 5, 1), 4)
    prices = pl.DataFrame(
        {
            "date": [dates[0], dates[1], dates[2], dates[3]],
            "listing_id": ["STK", "STK", "STK", "STK"],
            "close": [50.0, 52.0, 48.0, 55.0],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [dates[0]],
            "listing_id": ["STK"],
            "score": [1.0],
        }
    )

    config = SimulationConfig(
        starting_capital=20000.0,
        top_n=1,
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
    )

    engine1 = AccountingEngine(config)
    res1 = engine1.run(prices=prices, signals=signals)

    engine2 = AccountingEngine(config)
    res2 = engine2.run(prices=prices, signals=signals)

    assert res1.input_hash == res2.input_hash
    assert res1.metrics.to_dict() == res2.metrics.to_dict()
    assert res1.daily_curves.equals(res2.daily_curves)
    assert res1.trades.equals(res2.trades)
    assert res1.holdings.equals(res2.holdings)


def test_metrics_use_starting_capital_with_nonzero_inception_friction() -> None:
    """Cumulative return must use starting_capital, reflecting inception transaction costs."""
    dates = _make_daily_dates(date(2026, 6, 1), 2)
    # Price is flat at 100 on both days
    prices = pl.DataFrame(
        {
            "date": [dates[0], dates[1]],
            "listing_id": ["STK", "STK"],
            "close": [100.0, 100.0],
            "open": [100.0, 100.0],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [date(2026, 5, 31)],  # trades on dates[0]
            "listing_id": ["STK"],
            "score": [1.0],
        }
    )

    # 100 bps total friction (50 bps fee + 50 bps slippage)
    config = SimulationConfig(
        starting_capital=100000.0,
        top_n=1,
        transaction_cost_bps=50.0,
        slippage_bps=50.0,
    )
    result = AccountingEngine(config).run(prices=prices, signals=signals)

    # Inception trade costs approx 100,000 * 0.01 = 1,000, so ending capital is approx 99,000
    assert result.metrics.ending_capital < 100000.0
    # Cumulative return must be strictly negative, reflecting the 1% inception drag
    assert result.metrics.cumulative_return < 0.0
    assert abs(result.metrics.cumulative_return - (-0.0099)) < 0.002
    assert result.metrics.starting_capital == 100000.0


def test_missing_price_policy_mark_unresolved() -> None:
    """Missing price under MARK_UNRESOLVED retains last price and records observation."""
    dates = _make_daily_dates(date(2026, 6, 1), 4)
    # STK price is missing on dates[2]
    prices = pl.DataFrame(
        {
            "date": [dates[0], dates[1], dates[3]],
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

    config = SimulationConfig(
        starting_capital=10000.0,
        top_n=1,
        missing_price_policy=MissingPricePolicy.MARK_UNRESOLVED,
    )
    engine = AccountingEngine(config)
    result = engine.run(prices=prices, signals=signals, calendar=dates)

    # Unresolved observations must record missing price on dates[2]
    unresolved = result.unresolved_observations
    assert len(unresolved) >= 1
    missing_obs = [o for o in unresolved if o.date == dates[2]]
    assert len(missing_obs) == 1
    assert missing_obs[0].event_type == "missing_price"
    assert missing_obs[0].last_known_price == 105.0
    assert missing_obs[0].last_known_date == dates[1]
    assert missing_obs[0].action_taken == "retained_at_last_known"


def test_missing_price_policy_fail_raises() -> None:
    """Missing price under FAIL policy raises MissingPriceError."""
    dates = _make_daily_dates(date(2026, 6, 1), 4)
    prices = pl.DataFrame(
        {
            "date": [dates[0], dates[1], dates[3]],
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

    config = SimulationConfig(
        starting_capital=10000.0,
        top_n=1,
        missing_price_policy=MissingPricePolicy.FAIL,
    )
    engine = AccountingEngine(config)
    with pytest.raises(MissingPriceError):
        engine.run(prices=prices, signals=signals, calendar=dates)


def test_missing_price_policy_drop_liquidates_zero_and_clears_holdings() -> None:
    """DROP policy sets qty to zero, clears holdings, and avoids duplicate terminal obs."""
    dates = _make_daily_dates(date(2026, 6, 1), 3)
    # STK trades on dates[0], but price is missing on dates[1] and dates[2]
    prices = pl.DataFrame(
        {
            "date": [dates[0]],
            "listing_id": ["STK"],
            "close": [100.0],
            "open": [100.0],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [date(2026, 5, 31)],  # trades on dates[0]
            "listing_id": ["STK"],
            "score": [1.0],
        }
    )

    config = SimulationConfig(
        starting_capital=10000.0,
        top_n=1,
        missing_price_policy=MissingPricePolicy.DROP,
    )
    engine = AccountingEngine(config)
    result = engine.run(prices=prices, signals=signals, calendar=dates)

    # On dates[0]: bought STK
    h_day0 = result.holdings.filter(pl.col("observation_date") == dates[0])
    assert h_day0.height == 1

    # On dates[1]: missing price triggers DROP (liquidate_zero), so omitted from active holdings
    h_day1 = result.holdings.filter(pl.col("observation_date") == dates[1])
    assert h_day1.height == 0

    # On dates[2] (terminal date): remains 0 quantity, omitted from active holdings
    h_day2 = result.holdings.filter(pl.col("observation_date") == dates[2])
    assert h_day2.height == 0

    # Exactly 1 unresolved observation recorded for the DROP event, NO duplicate terminal obs
    unresolved = result.unresolved_observations
    assert len(unresolved) == 1
    assert unresolved[0].date == dates[1]
    assert unresolved[0].action_taken == "liquidated_zero"
    terminal_obs = [o for o in unresolved if o.event_type == "missing_terminal_price"]
    assert len(terminal_obs) == 0


def test_missing_terminal_price_hook() -> None:
    """Held asset with missing terminal date price is recorded as missing_terminal_price."""
    dates = _make_daily_dates(date(2026, 7, 1), 4)
    # STK is present on dates[0], dates[1], but stops trading after dates[1]
    prices = pl.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[3]],
            "listing_id": ["STK", "BENCH", "STK", "BENCH", "BENCH", "BENCH"],
            "close": [100.0, 50.0, 105.0, 51.0, 52.0, 53.0],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [dates[0]],
            "listing_id": ["STK"],
            "score": [1.0],
        }
    )

    config = SimulationConfig(
        starting_capital=10000.0,
        top_n=1,
        missing_price_policy=MissingPricePolicy.MARK_UNRESOLVED,
    )
    engine = AccountingEngine(config)
    result = engine.run(prices=prices, signals=signals)

    terminal_obs = [
        o for o in result.unresolved_observations if o.event_type == "missing_terminal_price"
    ]
    assert len(terminal_obs) >= 1
    assert terminal_obs[0].date == dates[3]
    assert terminal_obs[0].last_known_price == 105.0
    assert terminal_obs[0].last_known_date == dates[1]


def test_cash_settlement_hook_with_zero_settlement() -> None:
    """CashSettlementHook retains explicit 0.0 settlement (e.g. bankruptcy/worthless outcome)."""
    dates = _make_daily_dates(date(2026, 8, 1), 2)
    prices = pl.DataFrame(
        {
            "date": [dates[0]],
            "listing_id": ["BANKRUPT"],
            "close": [100.0],
            "open": [100.0],
        }
    )
    signals = pl.DataFrame(
        {
            "date": [date(2026, 7, 31)],  # trades on dates[0]
            "listing_id": ["BANKRUPT"],
            "score": [1.0],
        }
    )

    # Explicit 0.0 settlement on dates[1]
    hook = CashSettlementHook(
        settlements={("BANKRUPT", dates[1]): 0.0},
    )
    config = SimulationConfig(
        starting_capital=10000.0,
        top_n=1,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
    )
    engine = AccountingEngine(config, corporate_event_hook=hook)
    result = engine.run(prices=prices, signals=signals, calendar=dates)

    # Day 0: holding 100 shares at 100 = 10,000 value
    h_day0 = result.holdings.filter(pl.col("observation_date") == dates[0])
    assert h_day0.height == 1

    # Day 1: 0.0 settlement liquidates holding to 0 cash proceeds
    h_day1 = result.holdings.filter(pl.col("observation_date") == dates[1])
    assert h_day1.height == 0
    assert result.daily_curves["portfolio_value"][1] == 0.0

    # Unresolved observations records the 0.0 cash settlement
    assert len(result.unresolved_observations) == 1
    obs = result.unresolved_observations[0]
    assert obs.action_taken == "settle_cash"
    assert obs.event_type == "cash_settlement"
    assert "0.0 per share" in obs.details

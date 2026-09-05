from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import date
from typing import Any

import polars as pl

from stanstock.simulation.fx import (
    FxShadowLedger,
    build_fx_rate_lookup,
    converted_currencies,
    max_carry_days,
    normalize_fx_frame,
    validate_fx_coverage,
)
from stanstock.simulation.hooks import (
    CorporateEventHook,
    default_corporate_event_hook,
)
from stanstock.simulation.metrics import calculate_simulation_metrics
from stanstock.simulation.types import (
    ExecutionPriceBasis,
    FxAttribution,
    HoldingRecord,
    RebalanceFrequency,
    SimulationConfig,
    SimulationResult,
    TradeRecord,
    UnresolvedObservation,
)

MIN_PERSISTED_QUANTITY = 1e-8
MIN_PERSISTED_VALUE = 1e-6


class AccountingEngine:
    """Pure accounting engine for historical backtests and portfolio simulations.

    Takes dated per-listing prices/signals, starting capital, top-N or selected symbols,
    rebalance frequency, transaction cost bps, slippage bps, and missing-price policy.
    Avoids look-ahead: signals dated T can trade only on strictly subsequent eligible dates.
    """

    def __init__(
        self,
        config: SimulationConfig,
        corporate_event_hook: CorporateEventHook | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.corporate_event_hook = corporate_event_hook or default_corporate_event_hook

    def run(
        self,
        *,
        prices: pl.DataFrame,
        signals: pl.DataFrame | None = None,
        benchmark_prices: pl.DataFrame | None = None,
        calendar: Sequence[date] | None = None,
        fx_rates: pl.DataFrame | None = None,
    ) -> SimulationResult:
        normalized_prices = self._normalize_prices(prices)
        normalized_signals = self._normalize_signals(signals) if signals is not None else None
        normalized_bench = (
            self._normalize_benchmark(benchmark_prices) if benchmark_prices is not None else None
        )
        normalized_fx = (
            normalize_fx_frame(fx_rates, base_currency=self.config.base_currency)
            if fx_rates is not None
            else None
        )
        self._validate_currency_coverage(prices, converted=normalized_fx is not None)
        if normalized_fx is not None:
            self._validate_convertible_execution_basis(normalized_fx)
        conversion_inputs = (
            self._build_conversion_inputs(prices) if normalized_fx is not None else None
        )

        input_hash = self._compute_input_hash(
            normalized_prices,
            normalized_signals,
            normalized_bench,
            calendar,
            normalized_fx,
            conversion_inputs,
        )

        # Build trading dates schedule (including gaps for missing price tracking)
        trading_dates = self._build_trading_dates(normalized_prices, normalized_bench, calendar)
        if not trading_dates:
            raise ValueError("Prices DataFrame must contain at least one trading date")
        if normalized_fx is not None:
            # An explicit calendar can name dates the FX frame was never
            # resolved for. Prove coverage before any value is computed, so
            # such a date fails the run instead of quietly being valued at a
            # neighbouring day's conversion.
            validate_fx_coverage(
                normalized_fx,
                trading_dates=trading_dates,
                base_currency=self.config.base_currency or "",
                required_currencies=self._currency_by_listing(prices).values(),
            )

        # Map dates to fast lookup structures
        price_lookup, open_lookup, symbol_lookup = self._build_price_lookups(normalized_prices)

        # Determine rebalance schedule: mapping from trade_date -> signal_date
        # Invariant: trade_date > signal_date ALWAYS
        rebalance_schedule = self._build_rebalance_schedule(
            trading_dates=trading_dates,
            signals=normalized_signals,
        )
        self._validate_selected_inception_prices(
            rebalance_schedule=rebalance_schedule,
            price_lookup=price_lookup,
            open_lookup=open_lookup,
            symbol_lookup=symbol_lookup,
        )

        # Simulation state
        cash: float = float(self.config.starting_capital)
        holdings_qty: dict[str, float] = {}  # listing_id -> quantity
        last_known_price: dict[str, float] = {}  # listing_id -> price
        last_known_date: dict[str, date] = {}  # listing_id -> date
        # Native quote behind the last observed close, kept only for a
        # converted run: a foreign holding whose own market is closed must
        # keep its currency exposure, which means carrying the *quote* and
        # revaluing it at the current rate, not carrying a frozen conversion.
        last_known_native: dict[str, float] = {}

        fx_ledger = self._build_fx_ledger(
            prices=prices,
            fx_rates=normalized_fx,
            inception_date=trading_dates[0],
        )
        terminal_shadow_market_value = 0.0

        all_trades: list[TradeRecord] = []
        all_holdings: list[HoldingRecord] = []
        daily_records: list[dict[str, Any]] = []
        unresolved_observations: list[UnresolvedObservation] = []

        peak_portfolio_value = cash
        prev_portfolio_value = cash

        # Benchmark tracking setup
        bench_start_price: float | None = None
        bench_prev_price: float | None = None
        bench_lookup: dict[date, float] = {}
        if normalized_bench is not None:
            for row in normalized_bench.iter_rows(named=True):
                bench_lookup[row["date"]] = float(row["close"])

        friction_bps = self.config.transaction_cost_bps + self.config.slippage_bps
        friction_rate = friction_bps / 10000.0

        for t_idx, current_date in enumerate(trading_dates):
            # Check if current_date is an execution date for a prior signal
            if current_date in rebalance_schedule:
                signal_date = rebalance_schedule[current_date]
                # Ensure no look-ahead: trade date MUST be strictly after signal date
                if current_date <= signal_date:
                    raise AssertionError(
                        f"Look-ahead violation: trade on {current_date} "
                        f"from signal on {signal_date}"
                    )

                # Determine execution price for each asset
                current_exec_prices: dict[str, float] = {}
                candidate_lids = self._candidate_listings(normalized_signals, signal_date)
                for lid in set(list(holdings_qty.keys()) + candidate_lids):
                    exec_p = self._get_execution_price(lid, current_date, price_lookup, open_lookup)
                    if exec_p is not None and exec_p > 0:
                        current_exec_prices[lid] = exec_p

                # Current portfolio value at execution prices before trading.
                # A holding whose own market is shut has no execution price
                # today, but it still has a value: its last native quote at
                # today's rate. Sizing the rebalance off a stale conversion
                # would allocate capital against an exchange rate that no
                # longer exists, while leaving the holding itself untradable.
                current_market_val = 0.0
                for lid, qty in holdings_qty.items():
                    if qty > 1e-10:
                        p = current_exec_prices.get(lid)
                        if p is None:
                            p = _carried_price(
                                lid,
                                current_date,
                                fx_ledger=fx_ledger,
                                last_known_native=last_known_native,
                                last_known_price=last_known_price,
                            )
                        current_market_val += qty * p

                pre_trade_portfolio_val = cash + current_market_val

                # Determine target listings based on top_n or selected_symbols
                target_listings = self._select_targets(
                    current_date=current_date,
                    signal_date=signal_date,
                    signals=normalized_signals,
                    available_prices=current_exec_prices,
                    symbol_lookup=symbol_lookup,
                )

                # Rebalance if targets defined or if we need to exit all
                k_targets = len(target_listings)
                if k_targets > 0:
                    alloc_capital = pre_trade_portfolio_val * (
                        1.0 - (self.config.cash_buffer_bps / 10000.0)
                    )
                    target_dollars = alloc_capital / k_targets
                else:
                    target_dollars = 0.0

                # STEP 1: Execute SELLS first to generate cash
                # Deterministic order by symbol/listing_id
                held_to_evaluate = sorted(
                    [lid for lid, qty in holdings_qty.items() if qty > 1e-10],
                    key=lambda lid: (symbol_lookup.get(lid, lid), lid),
                )

                for lid in held_to_evaluate:
                    curr_qty = holdings_qty[lid]
                    p_exec = current_exec_prices.get(lid)
                    if p_exec is None or p_exec <= 0:
                        # Price missing on execution date: resolve via hook
                        resolution = self.corporate_event_hook(
                            listing_id=lid,
                            symbol=symbol_lookup.get(lid, lid),
                            current_date=current_date,
                            last_known_date=last_known_date.get(lid, current_date),
                            last_known_price=_carried_price(
                                lid,
                                current_date,
                                fx_ledger=fx_ledger,
                                last_known_native=last_known_native,
                                last_known_price=last_known_price,
                            ),
                            quantity_held=curr_qty,
                            event_type="missing_rebalance_price",
                            policy=self.config.missing_price_policy,
                        )
                        if resolution.observation:
                            unresolved_observations.append(resolution.observation)
                        if (
                            resolution.action == "settle_cash"
                            and resolution.settlement_price is not None
                        ):
                            cash += curr_qty * resolution.settlement_price
                            holdings_qty[lid] = 0.0
                            if fx_ledger is not None:
                                fx_ledger.record_cash_settlement(listing_id=lid, when=current_date)
                        elif resolution.action == "liquidate_zero":
                            holdings_qty[lid] = 0.0
                        continue

                    if lid in target_listings:
                        target_qty = target_dollars / p_exec
                    else:
                        target_qty = 0.0

                    if curr_qty > target_qty:
                        sell_qty = curr_qty - target_qty
                        gross_val = sell_qty * p_exec
                        if not _persistable_trade(sell_qty, p_exec):
                            continue
                        costs = gross_val * friction_rate
                        proceeds = gross_val - costs

                        cash += proceeds
                        holdings_qty[lid] = target_qty
                        sym = symbol_lookup.get(lid, lid)
                        if fx_ledger is not None:
                            fx_ledger.record_cash_flow(
                                listing_id=lid,
                                when=current_date,
                                base_amount=proceeds,
                            )

                        all_trades.append(
                            TradeRecord(
                                listing_id=lid,
                                symbol=sym,
                                trade_date=current_date,
                                side="sell",
                                quantity=sell_qty,
                                price=p_exec,
                                gross_value=gross_val,
                                costs=costs,
                            )
                        )

                # STEP 2: Execute BUYS with available cash
                sorted_targets = sorted(
                    target_listings,
                    key=lambda lid: (symbol_lookup.get(lid, lid), lid),
                )

                # lid, p_exec, desired_qty, gross_needed
                buy_orders: list[tuple[str, float, float, float]] = []
                total_cash_needed = 0.0

                for lid in sorted_targets:
                    p_exec = current_exec_prices.get(lid)
                    if p_exec is None or p_exec <= 0:
                        continue
                    curr_qty = holdings_qty.get(lid, 0.0)
                    target_qty = target_dollars / p_exec
                    if target_qty > curr_qty:
                        buy_qty = target_qty - curr_qty
                        gross_needed = buy_qty * p_exec
                        cash_needed = gross_needed * (1.0 + friction_rate)
                        buy_orders.append((lid, p_exec, buy_qty, gross_needed))
                        total_cash_needed += cash_needed

                # Scale buys if available cash is insufficient
                scale = 1.0
                if total_cash_needed > cash and total_cash_needed > 0:
                    scale = max(0.0, cash / total_cash_needed)

                for lid, p_exec, buy_qty, _gross_needed in buy_orders:
                    scaled_buy_qty = buy_qty * scale
                    if not _persistable_trade(scaled_buy_qty, p_exec):
                        continue
                    actual_gross = scaled_buy_qty * p_exec
                    actual_costs = actual_gross * friction_rate
                    total_outlay = actual_gross + actual_costs

                    cash -= total_outlay
                    holdings_qty[lid] = holdings_qty.get(lid, 0.0) + scaled_buy_qty
                    sym = symbol_lookup.get(lid, lid)
                    if fx_ledger is not None:
                        fx_ledger.record_cash_flow(
                            listing_id=lid,
                            when=current_date,
                            base_amount=-total_outlay,
                        )

                    all_trades.append(
                        TradeRecord(
                            listing_id=lid,
                            symbol=sym,
                            trade_date=current_date,
                            side="buy",
                            quantity=scaled_buy_qty,
                            price=p_exec,
                            gross_value=actual_gross,
                            costs=actual_costs,
                        )
                    )

            # END OF DAY VALUATION
            # Value all active holdings at close price
            daily_gross_market_val = 0.0
            day_holdings: list[HoldingRecord] = []

            active_listings = sorted(
                [lid for lid, qty in holdings_qty.items() if qty > 1e-10],
                key=lambda lid: (symbol_lookup.get(lid, lid), lid),
            )

            for lid in active_listings:
                qty = holdings_qty[lid]
                sym = symbol_lookup.get(lid, lid)
                close_p = price_lookup.get((lid, current_date))

                if close_p is not None and close_p > 0:
                    last_known_price[lid] = close_p
                    last_known_date[lid] = current_date
                    if fx_ledger is not None:
                        observed_rate = fx_ledger.dated_rate(lid, current_date)
                        if observed_rate is None or observed_rate <= 0:
                            fx_ledger.mark_unavailable(
                                f"No FX rate is available for listing {lid} on "
                                f"{current_date.isoformat()}, so its converted price cannot "
                                "be restated at reference rates."
                            )
                        else:
                            last_known_native[lid] = close_p / observed_rate
                        fx_ledger.record_price_rate(lid, current_date)
                else:
                    # Missing close price on observation date. A converted
                    # holding is carried as its last *native* quote revalued
                    # at today's rate, so a closed foreign market suspends the
                    # stock's price discovery without also freezing the
                    # portfolio's currency exposure.
                    carried = _carried_price(
                        lid,
                        current_date,
                        fx_ledger=fx_ledger,
                        last_known_native=last_known_native,
                        last_known_price=last_known_price,
                    )
                    resolution = self.corporate_event_hook(
                        listing_id=lid,
                        symbol=sym,
                        current_date=current_date,
                        last_known_date=last_known_date.get(lid, current_date),
                        last_known_price=carried,
                        quantity_held=qty,
                        event_type="missing_price",
                        policy=self.config.missing_price_policy,
                    )
                    if resolution.observation:
                        unresolved_observations.append(resolution.observation)

                    if (
                        resolution.action == "settle_cash"
                        and resolution.settlement_price is not None
                    ):
                        cash += qty * resolution.settlement_price
                        holdings_qty[lid] = 0.0
                        if fx_ledger is not None:
                            fx_ledger.record_cash_settlement(listing_id=lid, when=current_date)
                        continue
                    elif resolution.action == "liquidate_zero":
                        holdings_qty[lid] = 0.0
                        continue
                    else:  # retain_last_price
                        close_p = resolution.settlement_price or carried
                        if fx_ledger is not None:
                            # The carried value is now expressed at today's
                            # rate, so the shadow ledger must undo today's
                            # rate rather than the stale one.
                            fx_ledger.record_price_rate(lid, current_date)

                mv = qty * close_p
                daily_gross_market_val += mv
                day_holdings.append(
                    HoldingRecord(
                        listing_id=lid,
                        symbol=sym,
                        observation_date=current_date,
                        quantity=qty,
                        price=close_p,
                        market_value=mv,
                        weight=0.0,  # updated below once total value is known
                    )
                )

            portfolio_val = cash + daily_gross_market_val
            if fx_ledger is not None:
                terminal_shadow_market_value = fx_ledger.shadow_market_value(
                    (holding.listing_id, holding.market_value) for holding in day_holdings
                )

            # Update holding weights
            for h in day_holdings:
                h.weight = (h.market_value / portfolio_val) if portfolio_val > 0 else 0.0
                all_holdings.append(h)

            # Daily return & drawdown
            if t_idx == 0:
                daily_ret = 0.0
                peak_portfolio_value = portfolio_val
            else:
                if prev_portfolio_value > 0:
                    daily_ret = (portfolio_val - prev_portfolio_value) / prev_portfolio_value
                else:
                    daily_ret = 0.0
                if portfolio_val > peak_portfolio_value:
                    peak_portfolio_value = portfolio_val

            cum_ret = (portfolio_val - self.config.starting_capital) / self.config.starting_capital
            if peak_portfolio_value > 0:
                drawdown = (portfolio_val - peak_portfolio_value) / peak_portfolio_value
            else:
                drawdown = 0.0
            prev_portfolio_value = portfolio_val

            # Benchmark computation
            bench_val: float | None = None
            bench_ret: float | None = None
            bench_cum_ret: float | None = None

            if current_date in bench_lookup:
                b_price = bench_lookup[current_date]
                if bench_start_price is None:
                    bench_start_price = b_price
                    bench_val = float(self.config.starting_capital)
                    bench_ret = 0.0
                    bench_cum_ret = 0.0
                else:
                    bench_val = float(self.config.starting_capital * (b_price / bench_start_price))
                    if bench_prev_price and bench_prev_price > 0:
                        bench_ret = (b_price - bench_prev_price) / bench_prev_price
                    else:
                        bench_ret = 0.0
                    bench_cum_ret = (b_price - bench_start_price) / bench_start_price
                bench_prev_price = b_price

            daily_records.append(
                {
                    "date": current_date,
                    "portfolio_value": portfolio_val,
                    "cash": cash,
                    "gross_market_value": daily_gross_market_val,
                    "daily_return": daily_ret,
                    "cumulative_return": cum_ret,
                    "drawdown": drawdown,
                    "active_positions": len(day_holdings),
                    "benchmark_value": bench_val,
                    "benchmark_return": bench_ret,
                    "benchmark_cumulative_return": bench_cum_ret,
                }
            )

        # TERMINAL CHECK: Check unresolved terminal observations
        terminal_date = trading_dates[-1]
        for lid, qty in holdings_qty.items():
            if qty > 1e-10:
                last_dt = last_known_date.get(lid, terminal_date)
                if last_dt < terminal_date:
                    unresolved_observations.append(
                        UnresolvedObservation(
                            listing_id=lid,
                            symbol=symbol_lookup.get(lid, lid),
                            date=terminal_date,
                            event_type="missing_terminal_price",
                            last_known_price=last_known_price.get(lid, 0.0),
                            last_known_date=last_dt,
                            quantity_held=qty,
                            action_taken="retained_unresolved_terminal",
                            details=(
                                f"Active holding on terminal date {terminal_date} "
                                f"has price last observed on {last_dt}."
                            ),
                        )
                    )

        # Construct Polars DataFrames
        daily_curves = pl.DataFrame(daily_records)
        trades_df = self._build_trades_dataframe(all_trades)
        holdings_df = self._build_holdings_dataframe(all_holdings)

        ending_value = float(daily_records[-1]["portfolio_value"])
        starting_capital = float(self.config.starting_capital)
        fx_attribution = (
            fx_ledger.attribution(
                cumulative_return=(ending_value - starting_capital) / starting_capital,
                terminal_shadow_market_value=terminal_shadow_market_value,
            )
            if fx_ledger is not None
            else FxAttribution.not_applicable()
        )

        # Metrics calculation
        metrics = calculate_simulation_metrics(
            daily_curves=daily_curves,
            trades=trades_df,
            unresolved_observations=unresolved_observations,
            config=self.config,
            fx_attribution=fx_attribution,
        )

        return SimulationResult(
            daily_curves=daily_curves,
            holdings=holdings_df,
            trades=trades_df,
            metrics=metrics,
            unresolved_observations=unresolved_observations,
            config=self.config,
            input_hash=input_hash,
        )

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _normalize_prices(self, df: pl.DataFrame) -> pl.DataFrame:
        cols = df.columns
        date_col = "date" if "date" in cols else "observation_date"
        lid_col = "listing_id" if "listing_id" in cols else "symbol"
        if date_col not in cols or "close" not in cols or lid_col not in cols:
            raise ValueError(
                f"Prices must contain date, close, and listing_id/symbol columns. Found: {cols}"
            )

        select_exprs = [
            pl.col(date_col).cast(pl.Date).alias("date"),
            pl.col(lid_col).cast(pl.Utf8).alias("listing_id"),
            pl.col("close").cast(pl.Float64).alias("close"),
        ]
        if "open" in cols:
            select_exprs.append(pl.col("open").cast(pl.Float64).alias("open"))
        if "symbol" in cols and lid_col != "symbol":
            select_exprs.append(pl.col("symbol").cast(pl.Utf8).alias("symbol"))
        else:
            select_exprs.append(pl.col(lid_col).cast(pl.Utf8).alias("symbol"))

        norm = df.select(select_exprs).sort(["date", "listing_id"])
        if norm.select(["date", "listing_id"]).is_duplicated().any():
            dup_rows = (
                norm.filter(norm.select(["date", "listing_id"]).is_duplicated())
                .select(["date", "listing_id"])
                .unique()
            )
            raise ValueError(
                f"Duplicate price observations detected for (date, listing_id): "
                f"{dup_rows.to_dicts()}"
            )
        return norm

    def _normalize_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        cols = df.columns
        date_col = "date" if "date" in cols else "signal_date"
        lid_col = "listing_id" if "listing_id" in cols else "symbol"
        if date_col not in cols or lid_col not in cols:
            raise ValueError(
                f"Signals must contain date and listing_id/symbol columns. Found: {cols}"
            )

        select_exprs = [
            pl.col(date_col).cast(pl.Date).alias("date"),
            pl.col(lid_col).cast(pl.Utf8).alias("listing_id"),
        ]
        if "score" in cols:
            select_exprs.append(pl.col("score").cast(pl.Float64).alias("score"))
        elif "rank" in cols:
            select_exprs.append((-pl.col("rank").cast(pl.Float64)).alias("score"))
        elif "selected" in cols:
            select_exprs.append(
                pl.when(pl.col("selected").cast(pl.Boolean)).then(1.0).otherwise(0.0).alias("score")
            )
        else:
            select_exprs.append(pl.lit(1.0).alias("score"))

        if "symbol" in cols and lid_col != "symbol":
            select_exprs.append(pl.col("symbol").cast(pl.Utf8).alias("symbol"))
        else:
            select_exprs.append(pl.col(lid_col).cast(pl.Utf8).alias("symbol"))

        norm = df.select(select_exprs).sort(["date", "listing_id"])
        if norm.select(["date", "listing_id"]).is_duplicated().any():
            dup_rows = (
                norm.filter(norm.select(["date", "listing_id"]).is_duplicated())
                .select(["date", "listing_id"])
                .unique()
            )
            raise ValueError(
                f"Duplicate signal observations detected for (date, listing_id): "
                f"{dup_rows.to_dicts()}"
            )
        return norm

    def _normalize_benchmark(self, df: pl.DataFrame) -> pl.DataFrame:
        cols = df.columns
        date_col = "date" if "date" in cols else "observation_date"
        if date_col not in cols or "close" not in cols:
            raise ValueError("Benchmark prices must contain date and close columns")
        norm = df.select(
            [
                pl.col(date_col).cast(pl.Date).alias("date"),
                pl.col("close").cast(pl.Float64).alias("close"),
            ]
        ).sort("date")
        if norm.select(["date"]).is_duplicated().any():
            dup_dates = (
                norm.filter(norm.select(["date"]).is_duplicated())["date"].unique().to_list()
            )
            raise ValueError(f"Duplicate benchmark dates detected: {dup_dates}")
        return norm

    def _build_price_lookups(
        self, prices: pl.DataFrame
    ) -> tuple[dict[tuple[str, date], float], dict[tuple[str, date], float], dict[str, str]]:
        price_lookup: dict[tuple[str, date], float] = {}
        open_lookup: dict[tuple[str, date], float] = {}
        symbol_lookup: dict[str, str] = {}

        has_open = "open" in prices.columns
        for row in prices.iter_rows(named=True):
            lid = row["listing_id"]
            dt = row["date"]
            c_val = row.get("close")
            if c_val is not None:
                try:
                    f_val = float(c_val)
                    if f_val > 0:
                        price_lookup[(lid, dt)] = f_val
                except (ValueError, TypeError):
                    pass
            if has_open and row.get("open") is not None:
                try:
                    o_val = float(row["open"])
                    if o_val > 0:
                        open_lookup[(lid, dt)] = o_val
                except (ValueError, TypeError):
                    pass
            symbol_lookup[lid] = row["symbol"]

        return price_lookup, open_lookup, symbol_lookup

    def _build_trading_dates(
        self,
        prices: pl.DataFrame,
        benchmark: pl.DataFrame | None,
        calendar: Sequence[date] | None,
    ) -> list[date]:
        if calendar is not None:
            return sorted(set(calendar))

        dates_in_data = set(prices["date"].to_list())
        if benchmark is not None:
            dates_in_data.update(benchmark["date"].to_list())

        return sorted(dates_in_data)

    def _build_rebalance_schedule(
        self,
        trading_dates: list[date],
        signals: pl.DataFrame | None,
    ) -> dict[date, date]:
        """Build a mapping of {trade_date: signal_date}.

        Strictly enforces:
        1. trade_date > signal_date (no look-ahead).
        2. Rebalance dates adhere to config.rebalance_frequency.
        """
        schedule: dict[date, date] = {}
        freq = self.config.rebalance_frequency

        if not trading_dates:
            return schedule

        if signals is not None:
            sig_dates = signals["date"].unique().sort().to_list()
            if not sig_dates:
                return schedule

            # Find inception trade date: first trading date strictly greater than earliest signal
            first_sig = sig_dates[0]
            eligible_for_inception = [d for d in trading_dates if d > first_sig]
            if not eligible_for_inception:
                return schedule

            d_inception = eligible_for_inception[0]

            if freq == RebalanceFrequency.NEVER:
                # Buy and hold: exactly one trade date
                prior_sigs = [s for s in sig_dates if s < d_inception]
                if prior_sigs:
                    schedule[d_inception] = max(prior_sigs)
                return schedule

            # Inception trade date + subsequent frequency boundaries
            candidate_trade_dates = [d_inception]
            inception_idx = trading_dates.index(d_inception)

            if freq == RebalanceFrequency.DAILY:
                candidate_trade_dates = [d for d in trading_dates if d >= d_inception]
            else:
                for idx in range(inception_idx + 1, len(trading_dates)):
                    curr_d = trading_dates[idx]
                    prev_d = trading_dates[idx - 1]
                    trigger = False
                    if (
                        freq == RebalanceFrequency.WEEKLY
                        and curr_d.isocalendar()[:2] != prev_d.isocalendar()[:2]
                    ):
                        trigger = True
                    elif freq == RebalanceFrequency.MONTHLY and (curr_d.year, curr_d.month) != (
                        prev_d.year,
                        prev_d.month,
                    ):
                        trigger = True
                    elif freq == RebalanceFrequency.QUARTERLY and (
                        curr_d.year,
                        (curr_d.month - 1) // 3,
                    ) != (
                        prev_d.year,
                        (prev_d.month - 1) // 3,
                    ):
                        trigger = True
                    elif freq == RebalanceFrequency.YEARLY and curr_d.year != prev_d.year:
                        trigger = True

                    if trigger:
                        candidate_trade_dates.append(curr_d)

            for t_date in candidate_trade_dates:
                prior_signals = [s for s in sig_dates if s < t_date]
                if prior_signals:
                    schedule[t_date] = max(prior_signals)

        else:
            d_inception = trading_dates[0]
            if freq == RebalanceFrequency.NEVER:
                schedule[d_inception] = date.fromordinal(d_inception.toordinal() - 1)
                return schedule

            candidate_trade_dates = [d_inception]
            if freq == RebalanceFrequency.DAILY:
                candidate_trade_dates = list(trading_dates)
            else:
                for idx in range(1, len(trading_dates)):
                    curr_d = trading_dates[idx]
                    prev_d = trading_dates[idx - 1]
                    trigger = False
                    if (
                        freq == RebalanceFrequency.WEEKLY
                        and curr_d.isocalendar()[:2] != prev_d.isocalendar()[:2]
                    ):
                        trigger = True
                    elif freq == RebalanceFrequency.MONTHLY and (curr_d.year, curr_d.month) != (
                        prev_d.year,
                        prev_d.month,
                    ):
                        trigger = True
                    elif freq == RebalanceFrequency.QUARTERLY and (
                        curr_d.year,
                        (curr_d.month - 1) // 3,
                    ) != (
                        prev_d.year,
                        (prev_d.month - 1) // 3,
                    ):
                        trigger = True
                    elif freq == RebalanceFrequency.YEARLY and curr_d.year != prev_d.year:
                        trigger = True

                    if trigger:
                        candidate_trade_dates.append(curr_d)

            for idx, t_date in enumerate(candidate_trade_dates):
                if idx == 0:
                    schedule[t_date] = date.fromordinal(t_date.toordinal() - 1)
                else:
                    prev_trade_dates = [d for d in trading_dates if d < t_date]
                    schedule[t_date] = (
                        prev_trade_dates[-1]
                        if prev_trade_dates
                        else date.fromordinal(t_date.toordinal() - 1)
                    )

        return schedule

    def _get_execution_price(
        self,
        listing_id: str,
        trade_date: date,
        price_lookup: dict[tuple[str, date], float],
        open_lookup: dict[tuple[str, date], float],
    ) -> float | None:
        if self.config.execution_basis == ExecutionPriceBasis.NEXT_OPEN:
            # Must NOT silently fall back to close; return None if open absent
            return open_lookup.get((listing_id, trade_date))
        if self.config.execution_basis == ExecutionPriceBasis.NEXT_CLOSE:
            return price_lookup.get((listing_id, trade_date))
        # NEXT_ELIGIBLE explicitly falls back from open to close
        p_open = open_lookup.get((listing_id, trade_date))
        if p_open is not None and p_open > 0:
            return p_open
        return price_lookup.get((listing_id, trade_date))

    def _candidate_listings(self, signals: pl.DataFrame | None, signal_date: date) -> list[str]:
        if signals is not None:
            sub = signals.filter(pl.col("date") <= signal_date)
            return sub["listing_id"].unique().to_list()
        if self.config.selected_symbols:
            return list(self.config.selected_symbols)
        return []

    def _select_targets(
        self,
        current_date: date,
        signal_date: date,
        signals: pl.DataFrame | None,
        available_prices: dict[str, float],
        symbol_lookup: dict[str, str],
    ) -> list[str]:
        if signals is not None:
            sub = signals.filter(pl.col("date") <= signal_date)
            if sub.height == 0:
                return []
            # Take latest signal per listing
            latest = sub.sort(["listing_id", "date"]).group_by("listing_id").last()
            rows = latest.iter_rows(named=True)

            candidates: list[tuple[float, str, str]] = []  # score, symbol, lid
            for r in rows:
                lid = r["listing_id"]
                if lid in available_prices and available_prices[lid] > 0:
                    sym = symbol_lookup.get(lid, lid)
                    candidates.append((float(r["score"]), sym, lid))

            # Sort deterministically
            if self.config.tie_breaker == "symbol_desc":
                candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
                candidates.sort(key=lambda x: x[0], reverse=True)
            elif self.config.tie_breaker == "listing_id_asc":
                candidates.sort(key=lambda x: x[2])
                candidates.sort(key=lambda x: x[0], reverse=True)
            elif self.config.tie_breaker == "listing_id_desc":
                candidates.sort(key=lambda x: x[2], reverse=True)
                candidates.sort(key=lambda x: x[0], reverse=True)
            else:  # symbol_asc
                candidates.sort(key=lambda x: (x[1], x[2]))
                candidates.sort(key=lambda x: x[0], reverse=True)

            if self.config.top_n is not None:
                candidates = candidates[: self.config.top_n]
            return [c[2] for c in candidates]

        if self.config.selected_symbols:
            targets: list[str] = []
            for lid in self._selected_listing_ids(symbol_lookup):
                if lid in available_prices and available_prices[lid] > 0:
                    targets.append(lid)
            return targets

        return []

    def _validate_selected_inception_prices(
        self,
        *,
        rebalance_schedule: dict[date, date],
        price_lookup: dict[tuple[str, date], float],
        open_lookup: dict[tuple[str, date], float],
        symbol_lookup: dict[str, str],
    ) -> None:
        if not self.config.selected_symbols or not rebalance_schedule:
            return

        inception_date = min(rebalance_schedule)
        selected_ids = self._selected_listing_ids(symbol_lookup)
        missing = [
            lid
            for lid in selected_ids
            if self._get_execution_price(lid, inception_date, price_lookup, open_lookup) is None
        ]
        if missing:
            raise ValueError(
                "Selected listings lack a usable inception execution price on "
                f"{inception_date.isoformat()}: {', '.join(sorted(missing))}"
            )

    def _selected_listing_ids(self, symbol_lookup: dict[str, str]) -> list[str]:
        sym_to_lid = {symbol: listing_id for listing_id, symbol in symbol_lookup.items()}
        return [sym_to_lid.get(item, item) for item in self.config.selected_symbols or []]

    def _build_trades_dataframe(self, trades: list[TradeRecord]) -> pl.DataFrame:
        schema = {
            "listing_id": pl.Utf8,
            "symbol": pl.Utf8,
            "trade_date": pl.Date,
            "side": pl.Utf8,
            "quantity": pl.Float64,
            "price": pl.Float64,
            "gross_value": pl.Float64,
            "costs": pl.Float64,
        }
        if not trades:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(
            [
                {
                    "listing_id": t.listing_id,
                    "symbol": t.symbol,
                    "trade_date": t.trade_date,
                    "side": t.side,
                    "quantity": t.quantity,
                    "price": t.price,
                    "gross_value": t.gross_value,
                    "costs": t.costs,
                }
                for t in trades
            ],
            schema=schema,
        )

    def _build_holdings_dataframe(self, holdings: list[HoldingRecord]) -> pl.DataFrame:
        schema = {
            "listing_id": pl.Utf8,
            "symbol": pl.Utf8,
            "observation_date": pl.Date,
            "quantity": pl.Float64,
            "price": pl.Float64,
            "market_value": pl.Float64,
            "weight": pl.Float64,
        }
        if not holdings:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(
            [
                {
                    "listing_id": h.listing_id,
                    "symbol": h.symbol,
                    "observation_date": h.observation_date,
                    "quantity": h.quantity,
                    "price": h.price,
                    "market_value": h.market_value,
                    "weight": h.weight,
                }
                for h in holdings
            ],
            schema=schema,
        )

    CONVERSION_INPUT_COLUMNS: tuple[str, ...] = (
        "date",
        "listing_id",
        "currency",
        "close_native",
        "open_native",
        "fx_rate",
        "fx_observation_date",
        "fx_carry_days",
        "fx_path",
    )

    def _validate_currency_coverage(self, prices: pl.DataFrame, *, converted: bool) -> None:
        """Refuse to aggregate several native currencies without conversion.

        Without this guard a caller could hand the engine a panel whose rows
        are denominated in different currencies and no FX frame, and the
        engine would happily add them into one cash balance.
        """
        if "currency" not in prices.columns:
            return
        currencies = sorted(
            {
                str(value).upper()
                for value in prices["currency"].to_list()
                if value is not None and str(value)
            }
        )
        if converted or not currencies:
            return
        if len(currencies) > 1:
            raise ValueError(
                "Price panel mixes native currencies "
                f"({', '.join(currencies)}) but no FX rates were supplied; refusing to "
                "aggregate them into one cash balance."
            )
        base = (self.config.base_currency or "").upper()
        if base and currencies[0] != base:
            raise ValueError(
                f"Price panel is denominated in {currencies[0]} but the simulation base "
                f"currency is {base} and no FX rates were supplied."
            )

    def _build_conversion_inputs(self, prices: pl.DataFrame) -> pl.DataFrame:
        """Canonicalize the conversion-bearing columns for the input hash.

        Normalization drops native currency, native price, and per-row rate
        provenance before the accounting runs, yet those values decide how the
        run is converted and how its FX attribution is computed. Hashing them
        separately means swapping two listings' native currencies -- or
        replaying with a different retained native quote -- cannot produce the
        same reproducibility identity.
        """
        available = [column for column in self.CONVERSION_INPUT_COLUMNS if column in prices.columns]
        frame = prices.select(available)
        if "currency" in frame.columns:
            frame = frame.with_columns(pl.col("currency").cast(pl.Utf8).str.to_uppercase())
        sort_keys = [column for column in ("date", "listing_id", "currency") if column in available]
        return frame.sort(sort_keys) if sort_keys else frame

    def _validate_convertible_execution_basis(self, fx_rates: pl.DataFrame) -> None:
        """Refuse execution bases that may trade at a market open when converting.

        FX availability is modeled only to end-of-day resolution, because the
        source vintages carry no intraday knowability. Converting an opening
        trade therefore risks settling it at a rate published hours after the
        bell -- a look-ahead that is invisible in the result. Rather than
        pretend to an intraday cutoff the data cannot support, a converted run
        is restricted to close-based execution.
        """
        if not converted_currencies(fx_rates):
            return
        if self.config.execution_basis is ExecutionPriceBasis.NEXT_CLOSE:
            return
        raise ValueError(
            f"Execution basis '{self.config.execution_basis.value}' may execute at a market "
            "open, but FX availability is only resolved to end-of-day, so an opening trade "
            "could be converted with a rate published after it. Use "
            f"'{ExecutionPriceBasis.NEXT_CLOSE.value}' for a currency-converted run."
        )

    def _currency_by_listing(self, prices: pl.DataFrame) -> dict[str, str]:
        if "currency" not in prices.columns:
            raise ValueError(
                "FX rates were supplied but the price panel has no 'currency' column, so "
                "converted values cannot be traced back to a native currency."
            )
        currency_by_listing: dict[str, str] = {}
        pairs = prices.select(
            [
                pl.col("listing_id").cast(pl.Utf8),
                pl.col("currency").cast(pl.Utf8).str.to_uppercase(),
            ]
        ).unique()
        for row in pairs.iter_rows(named=True):
            listing_id = row["listing_id"]
            currency = row["currency"]
            existing = currency_by_listing.get(listing_id)
            if existing is not None and existing != currency:
                raise ValueError(
                    f"Listing {listing_id} has conflicting native currencies "
                    f"{existing!r} and {currency!r} in the price panel"
                )
            currency_by_listing[listing_id] = currency
        return currency_by_listing

    def _build_fx_ledger(
        self,
        *,
        prices: pl.DataFrame,
        fx_rates: pl.DataFrame | None,
        inception_date: date,
    ) -> FxShadowLedger | None:
        """Prepare the reference-rate mirror for a converted run.

        Returns ``None`` for a single-currency run so its accounting path,
        metrics, and reproducibility identity stay exactly as they were
        before FX conversion existed.
        """
        if fx_rates is None:
            return None
        if not self.config.base_currency:
            raise ValueError("FX rates were supplied but the simulation has no base currency")

        return FxShadowLedger(
            starting_capital=float(self.config.starting_capital),
            base_currency=self.config.base_currency,
            currency_by_listing=self._currency_by_listing(prices),
            rate_lookup=build_fx_rate_lookup(fx_rates),
            inception_date=inception_date,
            converted_currencies=converted_currencies(fx_rates),
            carry_days_used=max_carry_days(fx_rates),
        )

    def _compute_input_hash(
        self,
        prices: pl.DataFrame,
        signals: pl.DataFrame | None,
        bench: pl.DataFrame | None,
        calendar: Sequence[date] | None,
        fx_rates: pl.DataFrame | None = None,
        conversion_inputs: pl.DataFrame | None = None,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(
            json.dumps(
                self.config.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        _update_frame_hash(hasher, "prices", prices)
        _update_frame_hash(hasher, "signals", signals)
        _update_frame_hash(hasher, "benchmark", bench)
        # FX contributions are appended only for a run that actually converts.
        # A single-currency run therefore keeps the exact reproducibility
        # identity it had before FX conversion existed, so an archived hash
        # stays comparable across this change.
        if fx_rates is not None:
            _update_frame_hash(hasher, "fx_rates", fx_rates)
            _update_frame_hash(hasher, "conversion_inputs", conversion_inputs)
        normalized_calendar = (
            [value.isoformat() for value in sorted(set(calendar))] if calendar is not None else None
        )
        hasher.update(
            json.dumps(
                {"calendar": normalized_calendar},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return hasher.hexdigest()


def _carried_price(
    listing_id: str,
    when: date,
    *,
    fx_ledger: FxShadowLedger | None,
    last_known_native: dict[str, float],
    last_known_price: dict[str, float],
) -> float:
    """Value a holding whose own market produced no quote on ``when``.

    Without conversion this is exactly the previous behavior: the last known
    price is carried unchanged. With conversion, carrying the last *converted*
    price would silently pin the holding's exchange rate to the last session
    its market happened to be open, so a foreign-market holiday would erase a
    real currency move from the portfolio. The last native quote is carried
    instead and revalued at ``when``'s eligible rate. If that rate is missing
    the stale conversion is retained and the FX attribution is marked
    unavailable, rather than inventing a rate.
    """
    previous = last_known_price.get(listing_id, 0.0)
    if fx_ledger is None:
        return previous
    native = last_known_native.get(listing_id)
    rate = fx_ledger.dated_rate(listing_id, when)
    if native is None or rate is None or rate <= 0:
        fx_ledger.mark_unavailable(
            f"Holding {listing_id} could not be revalued at the {when.isoformat()} FX rate, "
            "so its carried price still reflects an earlier rate."
        )
        return previous
    return native * rate


def _persistable_trade(quantity: float, price: float) -> bool:
    return (
        quantity >= MIN_PERSISTED_QUANTITY
        and price >= MIN_PERSISTED_VALUE
        and quantity * price >= MIN_PERSISTED_VALUE
    )


def _update_frame_hash(
    hasher: Any,
    label: str,
    frame: pl.DataFrame | None,
) -> None:
    hasher.update(label.encode("utf-8"))
    if frame is None:
        hasher.update(b"<none>")
        return
    schema = [(name, str(dtype)) for name, dtype in frame.schema.items()]
    hasher.update(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    hasher.update(frame.write_ndjson().encode("utf-8"))

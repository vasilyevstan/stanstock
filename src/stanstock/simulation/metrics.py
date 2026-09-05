from __future__ import annotations

import numpy as np
import polars as pl

from stanstock.simulation.types import (
    SimulationConfig,
    SimulationMetrics,
    UnresolvedObservation,
)


def calculate_simulation_metrics(
    *,
    daily_curves: pl.DataFrame,
    trades: pl.DataFrame,
    unresolved_observations: list[UnresolvedObservation],
    config: SimulationConfig,
) -> SimulationMetrics:
    if daily_curves.height == 0:
        raise ValueError("Cannot calculate metrics from empty daily curves")

    start_date = daily_curves["date"][0]
    end_date = daily_curves["date"][-1]
    duration_days = (end_date - start_date).days
    years = duration_days / 365.25

    values = daily_curves["portfolio_value"].to_numpy()
    starting_capital = float(config.starting_capital)
    ending_capital = float(values[-1])
    cumulative_return = float((ending_capital - starting_capital) / starting_capital)

    # CAGR is valid when duration is at least 30 days and ending capital is positive
    if duration_days >= 30 and years > 0 and (1.0 + cumulative_return) > 0:
        cagr: float | None = float((1.0 + cumulative_return) ** (1.0 / years) - 1.0)
    else:
        cagr = None

    # Daily returns: skip initial period (t=0)
    returns = daily_curves["daily_return"].slice(1).to_numpy()
    n_returns = len(returns)

    if n_returns >= 2:
        vol_daily = float(np.std(returns, ddof=1))
        annualized_volatility = float(vol_daily * np.sqrt(252.0))
    else:
        vol_daily = 0.0
        annualized_volatility = 0.0

    # Sharpe ratio with configurable risk-free rate
    rf_daily = float((1.0 + config.risk_free_rate) ** (1.0 / 252.0) - 1.0)
    excess_returns = returns - rf_daily

    if n_returns >= 2 and vol_daily > 1e-12:
        sharpe_ratio: float | None = float((np.mean(excess_returns) / vol_daily) * np.sqrt(252.0))
    else:
        sharpe_ratio = None

    # Max Drawdown
    peaks = np.maximum.accumulate(values)
    drawdowns = (values - peaks) / np.maximum(peaks, 1e-12)
    max_drawdown = float(np.min(drawdowns))

    # Turnover
    if trades.height > 0:
        total_gross_value = float(trades["gross_value"].sum())
        mean_portfolio_val = float(np.mean(values))
        if mean_portfolio_val > 1e-12:
            turnover = float(total_gross_value / (2.0 * mean_portfolio_val))
        else:
            turnover = 0.0
        annualized_turnover = float(turnover / years) if years > 0 else turnover
    else:
        turnover = 0.0
        annualized_turnover = 0.0

    # Positive period rate (hit rate)
    pos_days = int(np.sum(returns > 1e-12))
    active_days = int(np.sum(np.abs(returns) > 1e-12))
    if active_days > 0:
        positive_period_rate = float(pos_days / active_days)
    elif n_returns > 0:
        positive_period_rate = float(pos_days / n_returns)
    else:
        positive_period_rate = 0.0

    # Benchmark metrics (if present)
    benchmark_cumulative_return: float | None = None
    benchmark_cagr: float | None = None
    benchmark_annualized_volatility: float | None = None
    benchmark_sharpe: float | None = None
    benchmark_max_drawdown: float | None = None
    excess_return: float | None = None
    alpha: float | None = None
    beta: float | None = None
    tracking_error: float | None = None
    information_ratio: float | None = None

    has_benchmark = (
        "benchmark_value" in daily_curves.columns
        and daily_curves["benchmark_value"].null_count() < daily_curves.height
    )
    if has_benchmark:
        valid_bench = daily_curves.filter(
            pl.col("benchmark_value").is_not_null() & ~pl.col("benchmark_value").is_nan()
        )
        if valid_bench.height > 0:
            bench_vals = valid_bench["benchmark_value"].to_numpy()
            b_start = float(bench_vals[0])
            b_end = float(bench_vals[-1])
            if b_start > 0:
                benchmark_cumulative_return = float((b_end - b_start) / b_start)
                excess_return = float(cumulative_return - benchmark_cumulative_return)

                if duration_days >= 30 and years > 0 and (1.0 + benchmark_cumulative_return) > 0:
                    benchmark_cagr = float(
                        (1.0 + benchmark_cumulative_return) ** (1.0 / years) - 1.0
                    )

                b_peaks = np.maximum.accumulate(bench_vals)
                b_drawdowns = (bench_vals - b_peaks) / np.maximum(b_peaks, 1e-12)
                benchmark_max_drawdown = float(np.min(b_drawdowns))

        if "benchmark_return" in daily_curves.columns:
            aligned = daily_curves.slice(1).filter(
                pl.col("daily_return").is_not_null()
                & pl.col("benchmark_return").is_not_null()
                & ~pl.col("daily_return").is_nan()
                & ~pl.col("benchmark_return").is_nan()
            )
            if aligned.height >= 2:
                r_p = aligned["daily_return"].to_numpy()
                r_b = aligned["benchmark_return"].to_numpy()
                b_vol_daily = float(np.std(r_b, ddof=1))
                benchmark_annualized_volatility = float(b_vol_daily * np.sqrt(252.0))
                b_excess = r_b - rf_daily
                if b_vol_daily > 1e-12:
                    benchmark_sharpe = float((np.mean(b_excess) / b_vol_daily) * np.sqrt(252.0))

                # Tracking error, beta, alpha, IR from aligned pairs
                diff = r_p - r_b
                te_daily = float(np.std(diff, ddof=1))
                tracking_error = float(te_daily * np.sqrt(252.0))
                if te_daily > 1e-12:
                    information_ratio = float((np.mean(diff) / te_daily) * np.sqrt(252.0))

                var_b = float(np.var(r_b, ddof=1))
                if var_b > 1e-12:
                    cov_mat = np.cov(r_p, r_b)
                    beta = float(cov_mat[0, 1] / var_b)
                    alpha = float((np.mean(r_p) - beta * np.mean(r_b)) * 252.0)

    return SimulationMetrics(
        cumulative_return=cumulative_return,
        cagr=cagr,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        turnover=turnover,
        annualized_turnover=annualized_turnover,
        positive_period_rate=positive_period_rate,
        total_trades=trades.height,
        start_date=start_date,
        end_date=end_date,
        duration_days=duration_days,
        starting_capital=starting_capital,
        ending_capital=ending_capital,
        unresolved_count=len(unresolved_observations),
        mode=config.mode.value,
        grade=config.grade.value,
        simulation_kind=config.simulation_kind,
        base_currency=config.base_currency,
        benchmark_cumulative_return=benchmark_cumulative_return,
        benchmark_cagr=benchmark_cagr,
        benchmark_annualized_volatility=benchmark_annualized_volatility,
        benchmark_sharpe=benchmark_sharpe,
        benchmark_max_drawdown=benchmark_max_drawdown,
        excess_return=excess_return,
        alpha=alpha,
        beta=beta,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
    )

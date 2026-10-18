from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from stanstock.simulation.types import (
    ACCOUNTING_VERSION,
    FxAttribution,
    SimulationConfig,
    SimulationMetrics,
    UnresolvedObservation,
    resolve_accounting_version,
)


def calculate_simulation_metrics(
    *,
    daily_curves: pl.DataFrame,
    trades: pl.DataFrame,
    unresolved_observations: list[UnresolvedObservation],
    config: SimulationConfig,
    fx_attribution: FxAttribution | None = None,
) -> SimulationMetrics:
    if daily_curves.height == 0:
        raise ValueError("Cannot calculate metrics from empty daily curves")
    v2 = resolve_accounting_version(config.custom_parameters) == ACCOUNTING_VERSION

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
    peaks = np.maximum.accumulate(np.maximum(values, starting_capital) if v2 else values)
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
    comparison_coverage: dict[str, Any] | None = None
    matched = daily_curves
    if v2:
        comparison_coverage, matched = _comparison_coverage(daily_curves)

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
            if b_start > 0 and (not v2 or comparison_coverage["boundaries_complete"]):  # type: ignore[index]
                benchmark_cumulative_return = float((b_end - b_start) / b_start)
                excess_return = float(cumulative_return - benchmark_cumulative_return)

                if duration_days >= 30 and years > 0 and (1.0 + benchmark_cumulative_return) > 0:
                    benchmark_cagr = float(
                        (1.0 + benchmark_cumulative_return) ** (1.0 / years) - 1.0
                    )

                if not v2 or comparison_coverage["path_complete"]:  # type: ignore[index]
                    b_peaks = np.maximum.accumulate(bench_vals)
                    b_drawdowns = (bench_vals - b_peaks) / np.maximum(b_peaks, 1e-12)
                    benchmark_max_drawdown = float(np.min(b_drawdowns))

        if "benchmark_return" in daily_curves.columns:
            aligned = (
                matched
                if v2
                else daily_curves.slice(1).filter(
                    pl.col("daily_return").is_not_null()
                    & pl.col("benchmark_return").is_not_null()
                    & ~pl.col("daily_return").is_nan()
                    & ~pl.col("benchmark_return").is_nan()
                )
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

    attribution = fx_attribution or FxAttribution.not_applicable()

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
        fx_conversion_applied=attribution.conversion_applied,
        fx_native_currencies=(
            list(attribution.native_currencies) if attribution.native_currencies else None
        ),
        fx_max_carry_days_used=attribution.max_carry_days_used,
        fx_local_currency_cumulative_return=attribution.local_currency_cumulative_return,
        fx_contribution_return=attribution.contribution_return,
        fx_attribution_status=attribution.status.value,
        fx_attribution_detail=attribution.detail,
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
        comparison_coverage=comparison_coverage,
    )


def _comparison_coverage(curves: pl.DataFrame) -> tuple[dict[str, Any], pl.DataFrame]:
    """One adjacent-calendar support proof shared by every v2 comparison."""
    days = curves["date"].to_list()
    if days != sorted(set(days)):
        raise ValueError("Simulation calendar must contain unique ordered session dates")
    values = (
        curves["benchmark_value"].to_list()
        if "benchmark_value" in curves.columns
        else [None] * len(days)
    )
    present = [value is not None and math.isfinite(value) and value > 0 for value in values]
    indices = [index for index in range(1, len(days)) if present[index - 1] and present[index]]
    for index in indices:
        current, previous = values[index], values[index - 1]
        assert current is not None and previous is not None
        if current / previous == 0:
            raise ValueError("Benchmark ratio is not representable")
        if curves["benchmark_return"][index] is None:
            raise ValueError("Benchmark adjacent return is missing")
    reasons = []
    if not present[0]:
        reasons.append("benchmark_start_boundary_missing")
    if not present[-1]:
        reasons.append("benchmark_end_boundary_missing")
    if not all(present):
        reasons.append("benchmark_path_incomplete")
    if len(indices) < 2:
        reasons.append("insufficient_matched_return_pairs")
    elif np.var(curves[indices]["benchmark_return"].to_numpy(), ddof=1) <= 1e-12:
        reasons.append("benchmark_return_variance_zero")
    if len(indices) >= 2:
        differences = curves[indices]["daily_return"] - curves[indices]["benchmark_return"]
        if np.std(differences.to_numpy(), ddof=1) <= 1e-12:
            reasons.append("active_return_variance_zero")
    return {
        "start_date": days[0].isoformat(),
        "end_date": days[-1].isoformat(),
        "benchmark_start_present": present[0],
        "benchmark_end_present": present[-1],
        "boundaries_complete": present[0] and present[-1],
        "path_complete": all(present),
        "calendar_intervals": max(0, len(days) - 1),
        "matched_intervals": len(indices),
        "omitted_intervals": max(0, len(days) - 1) - len(indices),
        "reasons": reasons,
    }, curves[indices]

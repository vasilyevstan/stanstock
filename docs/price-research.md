# Price-only stock research

`research-product-v1` combines two transparent operators: a six-month
momentum suggestion and conditional price projections at six months,
twelve months, three years and five years. It does not use an opaque
overall score, fitted machine-learning model, analyst target or LLM.

**Release status:** the calculation foundation is implemented; integrated
intake, serving and production activation are still pending. This document
describes the reviewed product contract, not a claim that the running
application has already switched to it.

## What the suggestion means

`us-relative-momentum-v1` compares a stock with SPY over the same prior-year
window, excluding the most recent 21 market sessions:

```text
stock_momentum = log(stock_close[T-21] / stock_close[T-252])
benchmark_momentum = log(SPY_close[T-21] / SPY_close[T-252])
relative_momentum = stock_momentum - benchmark_momentum
```

Positive direction requires both stock momentum and relative momentum to
be positive. Negative direction requires both to be negative. Other
combinations are mixed. The relative figure is a difference in log returns,
not a probability or a difference in ordinary percentage returns.

A negative direction produces **Avoid**. A positive direction can produce
**Buy** only with valid input evidence, annualized volatility no more than
twice SPY's, a trailing drawdown no worse than 50%, compatible average daily
dollar turnover of at least $5 million over 20 sessions, and a target-date
close of at least $10. Otherwise the suggestion is **Hold**, with reasons.
Missing momentum is **unavailable**, not a fabricated Hold.

The suggestion concerns the following **126 market sessions**, approximately
six months. Missing risk or liquidity can block Buy without erasing a valid
negative signal. A missing advisory projection does not determine the
momentum suggestion.

This is an adaptation of portfolio-momentum research, not a reproduction
of the original papers' portfolios, total returns or trading costs. A Buy
label is a research policy output, not demonstrated profitability or an
instruction to place an order.

## Reading the three projection numbers

`us-price-fhs-v1` reports three **cumulative price-return** quantiles:

| Label | Meaning inside the simulated model |
|---|---|
| Lower | 20th percentile |
| Median | 50th percentile, not the mean |
| Upper | 80th percentile |

For example, **Lower -10% / Median +5% / Upper +25%** is one illustrative
projection range, not three probabilities. The Lower-to-Upper interval
contains **60% of the simulated model distribution**. It does not establish
60% real-world coverage. Neither bound is a worst case, a maximum drawdown
or a stop-loss guarantee.

Prices corresponding to these returns use the immutable target-date close.
All results are split-adjusted **price returns excluding dividends**.
Three-year and five-year figures remain cumulative; any annualized display
must be labelled separately.

Gain probability and calibrated confidence are not estimated. A missing
probability is null, not 0%. More simulated paths cannot create independent
market evidence or unlock an accuracy claim.

## How the projections are calculated

Each stock and SPY need **757 consecutive common exchange-session closes**
ending on the target date. This supplies 756 daily log returns, about three
years. Missing sessions, duplicates, invalid prices, unsupported identity
or currency, and incompatible source evidence are not filled or guessed.
Volume is optional for projections but required for the Buy liquidity gate.

For the stock's daily log returns `r`, calculate the arithmetic mean `mu`
and population variance `v`. Start the historical conditional variance at
`q[1] = v`, then apply:

```text
z[t] = (r[t] - mu) / sqrt(q[t])
q[t+1] = 0.01*v + 0.94*q[t] + 0.05*(r[t] - mu)^2
```

Discard the first 252 standardized returns. Center and rescale the remaining
504 residuals to zero arithmetic mean and unit second moment. Nonpositive
variance or residual scale, or nonfinite values, withhold the affected
calculation with a reason; no artificial variance floor or jitter is added.

Starting from the final historical variance, generate 8,192 paths:

```text
epsilon[h] = sqrt(q[h]) * independently_sampled_residual[h]
future_log_return[h] = mu + epsilon[h]
q[h+1] = 0.01*v + 0.94*q[h] + 0.05*epsilon[h]^2
cumulative_price_return[H] = exp(sum(future_log_return[1:H])) - 1
```

Residuals are sampled uniformly with replacement, one at a time. The four
horizons are 126, 252, 756 and 1,260 sessions. Quantiles use NumPy's linear
convention. The coefficients, history window and path count are fixed
policy assumptions, not parameters fitted to the final evaluation period.

The model extrapolates the trailing three-year **mean log return**. That
continuation assumption can dominate long-horizon results. Regime changes,
drift-estimation uncertainty, corporate events and survivorship make long
projections especially uncertain; parameter uncertainty is not integrated.
The momentum direction and projected median can disagree because they use
different quantities.

A **zero-log-drift sensitivity** reuses the same shocks and subtracts
`H * mu` from terminal log returns. It illustrates dependence on the drift
assumption; it is not a third calibrated model or a guaranteed flat-price
forecast.

## Reproducibility and evidence

The tracked configuration is
[`config/scoring/research-product-v1.yml`](../config/scoring/research-product-v1.yml).
Its physical bytes and effective configuration hash are pinned.

PCG64 receives a deterministic seed derived from method version, effective
configuration hash, permanent listing UUID and target date. Mutable ticker,
run UUID and generation time do not choose the random paths. Complete input
values, source identities and the session calendar have a separate digest.
The recorded calculation also identifies NumPy, the generator, quantile
convention and numeric precision.

Returns use four decimal places and prices six, with half-even rounding.
Each is rounded from its own raw calculation; a stored price is not
recomputed from a previously rounded return. An unrepresentable or invalid
result withholds its entire horizon triplet rather than clipping bounds.

Current-vintage retrospective calculations must not be called historically
observed forecasts. Actual generation time, logical data cutoff, source
retrieval and availability remain distinct. Recorded predictions are
append-only; a later reissue does not inherit another version's on-time
status or become a second independent market observation.

## What would establish useful predictive evidence

Paper references motivate the operators; they do not prove this particular
implementation predicts individual stocks well. Comparative evidence must
use the predeclared protocol, not parameter selection after seeing results.

The frozen replay protocol uses a 2019-09-03 exchange-session epoch,
horizon-spaced anchors, 756 preceding returns and fully matured outcomes.
Development outcomes finish before 2024; validation anchors and outcomes
both lie within 2024; final-holdout anchors start in 2025 and outcomes must
finish by 2026-09-11. Intervals crossing partition boundaries are excluded.

The projection comparators are a zero-log-drift Gaussian and a
historical-log-drift constant-variance Gaussian. Reports include median
absolute error, quantile losses, interval width and inclusion, and the
central-60% interval score. Results are averaged within target cohorts
before comparison across cohorts; stock counts are not substitutes for
independent target dates.

Momentum evaluation compares stock and SPY returns over the same 126
sessions. Buy succeeds only if the stock rises and exceeds SPY; Avoid
succeeds only if it falls and underperforms SPY. Hold and unavailable
suggestions have no success label. Advisory results have no recommendation
success label.

Some long-horizon partitions may contain no mature observations. Report
that absence, baseline underperformance and unestablished skill explicitly.
These findings do not erase a calculable conditional projection, but they
prevent presenting it as proven forecasting accuracy.

Doubling to 16,384 paths is a numerical convergence diagnostic with the
same initial 8,192 paths. Quantile movement above the larger of one
percentage point or 2% of interval width needs investigation. Passing that
diagnostic is not evidence of financial skill.

## Monitored stocks, price bands and legacy methods

The reviewed admission policy keeps the existing 100-name core and overlays
at most 20 owner-selected, independently verified names, deduplicated by
permanent listing identity. SPY is the separate benchmark, never another
stock recommendation. Adequate registered history is reused before any
authorized history bootstrap.

A qualified Under-$10 stock can receive momentum analysis and all four
price-only projections. It remains a **speculative watch with 0% new
allocation**, not a Buy promotion or an approved portfolio addition.
Nominally low price is not alpha or fundamental cheapness. Price-only
projections do not satisfy separate solvency, dilution, corporate-action
or investment-activation requirements.

Old score-based, empirical-range and SEC-dependent methods retain their
original immutable evidence and meanings. Their availability and accuracy
must not be pooled with this product. SEC facts are not a dependency of
these price-only projections.

## Research references and adaptations

- Jegadeesh and Titman (1993), *Returns to Buying Winners and Selling
  Losers: Implications for Stock Market Efficiency*.
  [DOI: 10.1111/j.1540-6261.1993.tb04702.x](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x).
  Motivation for momentum, not per-stock probabilities.
- Carhart (1997), *On Persistence in Mutual Fund Performance*.
  [DOI: 10.1111/j.1540-6261.1997.tb03808.x](https://doi.org/10.1111/j.1540-6261.1997.tb03808.x).
  Momentum-factor context; this product uses a simpler SPY-relative sign
  policy rather than that paper's model.
- Bollerslev (1986), *Generalized Autoregressive Conditional
  Heteroskedasticity*.
  [DOI: 10.1016/0304-4076(86)90063-1](https://doi.org/10.1016/0304-4076(86)90063-1).
  Conditional-variance structure; coefficients here are fixed, not fitted.
- Barone-Adesi, Giannopoulos and Vosper (1999), *VaR without correlations
  for portfolios of derivative securities*, Journal of Futures Markets
  19(5), 583-602. Filtered-residual simulation motivation, adapted here to
  single-stock conditional price projections.
- J.P. Morgan/Reuters (1996), *RiskMetrics - Technical Document*, fourth
  edition. Daily exponential-volatility reference; this product adds a
  variance target and does not claim to replicate standard RiskMetrics.

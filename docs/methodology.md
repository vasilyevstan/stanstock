# Methodology

StanStock v1 uses deterministic, versioned rules. It does not fit a predictive
machine-learning model and does not permit generated text to alter a score,
scenario, risk class, or recommendation.

## Inputs

Supported inputs are grouped into:

- price returns, trend, momentum, range, volume, volatility, beta, and
  drawdown;
- filing-derived growth, margins, cash flow, profitability, leverage,
  liquidity, and valuation;
- regional/sector relative behavior and price-derived market regime;
- data completeness, freshness, liquidity, and source quality.

An indicator is unavailable when its minimum history or source inputs are
missing. Unsupported values are never approximated from unrelated fields.

## Horizon weights

The initial hypothesis weights are:

| Component | 1-10 trading days | 6-12 months | 3+ years |
|---|---:|---:|---:|
| Fundamental quality | 5 | 20 | 30 |
| Growth and reinvestment | 5 | 20 | 25 |
| Valuation | 5 | 20 | 20 |
| Momentum and technical trend | 40 | 15 | 5 |
| Risk and liquidity | 25 | 15 | 15 |
| Market and sector regime | 20 | 10 | 5 |

Weights, thresholds, coverage requirements, and scenario parameters live in
versioned YAML. Every issued prediction stores the configuration hash and code
revision. These weights are starting assumptions; they are not optimized
against the period later used to report performance.

## Normalization and missingness

Cross-sectional factors use robust ranks or winsorized standardized values
within an appropriate region/sector peer group. Annual European fundamentals
are not treated as equally fresh to US quarterly facts.

Missingness reduces coverage and confidence. A configured minimum coverage can
block a BUY result entirely. A missing value is never converted to zero.
Source filing concepts are mapped to canonical calculation names without
changing the stored source concept. When multiple vintages restate the same
reporting period, the latest eligible vintage replaces that period before
growth is calculated against the prior distinct period.

## Risk

Risk combines supported observations such as:

- annualized and downside volatility;
- beta and maximum drawdown;
- liquidity and stale-trading risk;
- leverage, interest coverage, and balance-sheet resilience;
- earnings/free-cash-flow stability;
- valuation dispersion and sector cyclicality;
- data completeness and freshness.

Versioned thresholds map supported 0-100 scores to LOW, MEDIUM, HIGH, or VERY
HIGH. When no supported risk input exists, risk is `INSUFFICIENT EVIDENCE`,
the numeric score is null, and BUY is blocked rather than inventing a
conservative-looking number.

## Recommendations

BUY, HOLD, and AVOID are deterministic gates over:

- overall and component scores;
- downside in the relevant scenario;
- risk class;
- evidence sufficiency and confidence;
- liquidity and blocking quality flags.

Explanations are selected from the actual factors that crossed documented
thresholds. They are not free-form rationalizations.

## Scenarios

- **1-10 trading days:** empirical/rule ranges driven primarily by trend,
  momentum, volume, volatility, liquidity, and market regime.
- **6-12 months:** empirical/rule ranges combining valuation, quality, growth,
  momentum, risk, and regional/sector context.
- **3+ years:** explicit fundamental cases for growth, margins, free cash flow,
  balance-sheet resilience, and valuation normalization.

Bear, base, and bull are ordered ranges, not precise target prices.
Probability of positive return is omitted with an insufficiency reason until
the relevant historical sample satisfies its configured minimum.

Confidence remains labeled `heuristic` until out-of-sample calibration
evidence exists.

## Predictions and evaluation

A prediction records the actual generation time, target market date,
permanent listing ID, source price, horizon, scenarios, confidence,
recommendation, component scores, source assets, configuration hash, and code
revision.

Predictions are append-only. A correction uses a new model/configuration
version. Missed runs can be reconstructed for research but cannot be presented
as calls issued on time.

Performance reports only matured outcomes and retains unresolved corporate
events in coverage counts. Aggregate return, hit-rate, or calibration metrics
are withheld below the configured minimum sample. Short, medium, and long
outcomes mature after 10, 252, and 756 observed sessions respectively; calendar
weekends and holidays are never manufactured. BUY succeeds on a positive
return, AVOID on a non-positive return, and HOLD only when the realized return
falls inside its stored bear/bull range.

The performance page aggregates only outcomes from observed universe snapshots
whose predictions were generated on their target date. Matured synthetic or
later-reconstructed outcomes remain visibly excluded from live,
out-of-sample claims.

The evaluator also compares the target-date close in the evaluation vintage
with the immutable prediction source price. A material mismatch is classified
as a corporate event/adjusted-history revision and is excluded from ordinary
return and directional-accuracy calculations.

## Simulations

Backtests and portfolio comparisons use one accounting implementation.
Signals become tradable only on the configured next eligible observation.
Inputs include universe grade, selection rule, rebalance schedule, starting
cash, costs, slippage, currency convention, benchmark, and missing-price/event
rules.

The current engine does not perform FX conversion. A simulation must therefore
contain one native listing currency. Mixed-universe backtests require an
explicit USD, EUR, or GBP filter, and mixed-currency portfolio selections are
rejected. The selected currency is persisted in the run configuration and
metrics. Explicit buy-and-hold selections must all have a usable price on the
common inception execution date; otherwise the run fails rather than silently
redistributing the missing listing's allocation.

Research-grade reconstructed history, observed-universe history, and actual
live prediction outcomes are always labeled separately. Every persisted run
stores the exact price, signal, and benchmark input frames alongside the result
curve so it can be reproduced without relying on mutable current state.
Observed-grade backtests reject analyses generated after their target date;
research-grade reconstructions retain both their historical data cutoff and
actual later generation timestamp in the persisted signal input. The run input
hash covers complete normalized frame contents and the explicit calendar, not
aggregate row counts or sums.

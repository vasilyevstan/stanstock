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

## Highlighted opportunities

The opportunity highlight is a separate, versioned presentation policy in
`config/opportunities/great-opportunity-v1.yml`; it does not alter the
underlying recommendation. A highlighted result must be BUY, clear the
policy's score and confidence thresholds, remain LOW or MEDIUM risk, and have
a positive base case for the analysis mode's supported horizon.

Full analyses use the label `Great opportunity` and require fundamentals.
The US price-only baseline instead uses `Strong short-term setup`, checks only
its supported short horizon, and remains visibly identified as price-only.
The policy version is shown on stock detail so a later policy revision cannot
silently masquerade as the original rule.

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

The planned implementation sequence and exact math-only forecast identities
are documented in
[`docs/forecast-roadmap.md`](forecast-roadmap.md). The roadmap explicitly
excludes LLMs, trained machine-learning models, analyst targets, and automated
parameter optimization.

## Predictions and evaluation

A prediction records the actual generation time, target market date,
permanent listing ID, source price, horizon, scenarios, confidence,
recommendation, component scores, source assets, configuration hash, and code
revision.

Predictions are append-only. A correction uses a new model/configuration
version. Missed runs can be reconstructed for research but cannot be presented
as calls issued on time. On-time status belongs to each immutable prediction,
not only its parent analysis, so a later reissued version is excluded from live
performance evidence.

Performance reports only matured outcomes and retains unresolved corporate
events in coverage counts. Aggregate return, hit-rate, or calibration metrics
are withheld below the configured minimum sample. Short, medium, and long
outcomes mature after 10, 252, and 756 observed sessions respectively; calendar
weekends and holidays are never manufactured. BUY succeeds on a positive
return, AVOID on a non-positive return, and HOLD only when the realized return
falls inside its stored bear/bull range.

The performance page aggregates only outcomes from observed universe snapshots
whose individual predictions were issued before the next market session.
Matured synthetic, unsupported, or later-reissued outcomes remain visibly
excluded from live, out-of-sample claims. A price-only baseline persists only
the short horizon it supports; withheld medium and long scenarios are not
prediction records and therefore cannot enter evaluation denominators.

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

The current engine converts native prices into one explicit base currency
using point-in-time FX. A selection spanning several native currencies must
name its base currency (`--base-currency USD|EUR|GBP`); a single-currency
selection infers it. Every value a run reports -- execution prices, cash,
holdings, trading costs, the benchmark, and all metrics -- is denominated in
that base currency, while the persisted price input keeps each row's native
price, native currency, and the exact rate applied.

Conversion rates are dated, and every valued date is resolved against its own
cutoff -- the end of that date -- so a rate published later can never change
how an earlier execution was priced. A rate published after the valued date is
refused in every run; a source asset merely *retrieved* later is permitted
only for an explicitly research-grade reconstruction, never for observed
evidence. Because FX series have no weekend, holiday, or (for the synthetic
demo bundles) non-Friday observations, the most recent eligible observation is
carried forward and its carry distance is recorded per converted date. The
reviewed maximum carry is 7 calendar days; a run may tighten it to as little
as 0 but cannot widen it, and a carry beyond the limit fails rather than
pricing from a stale rate. Derivation paths are ranked `identity` > `direct` >
`inverse` > `cross:<pivot>`, and the path used is stored alongside each rate.
Equally-ranked derivations that disagree, a missing pair, and an over-stale
observation each fail the whole run instead of producing a
partially-converted result. `--restrict-native-currency` still selects a
single-currency slice of a mixed universe when conversion is not wanted at
all, and is rejected outright when it would silently exclude an explicitly
selected holding.

A holding whose own market is closed keeps its currency exposure: its last
native quote is carried and revalued at the current eligible rate -- both when
the day is valued and when a rebalance sizes its targets -- so a
foreign-market holiday suspends the stock's price discovery without also
freezing the portfolio's exchange rate. The closed holding remains untradable
on that date; only its value is restated. Every simulated date must resolve a
rate for every non-base currency before any accounting begins, so a date the
FX series cannot cover fails the run instead of producing a portfolio return.

Because FX vintages carry no intraday knowability, availability is resolved
only to end-of-day. A converted run therefore executes on closing prices:
`next_open` and `next_eligible` are rejected, because an opening trade could
otherwise be settled at a rate published hours after the bell.

A converted run also reports the split between stock return and FX
contribution. The stock leg revalues the *same* quantity path at each native
currency's rate on the first simulated date, so the two legs sum exactly to
the reported cumulative return. When any part of that restatement is not
established -- for example a cash settlement with no FX basis -- both figures
are withheld with a stated reason rather than reported as an estimate.

The base currency, the native currencies converted, the longest carry
applied, and the attribution are persisted in the run configuration and
metrics. Explicit buy-and-hold selections must all have a usable price on the
common inception execution date; otherwise the run fails rather than silently
redistributing the missing listing's allocation.

Research-grade reconstructed history, observed-universe history, and actual
live prediction outcomes are always labeled separately. Every persisted run
stores the exact price, signal, benchmark, and FX input frames alongside the
result curve so it can be reproduced without relying on mutable current state.
Observed-grade backtests reject analyses generated after their target date;
research-grade reconstructions retain both their historical data cutoff and
actual later generation timestamp in the persisted signal input. The run input
hash covers complete normalized frame contents -- prices, signals, benchmark,
and, when a run converts, the dated FX frame together with the canonical
native-currency assignment and retained conversion inputs -- plus the explicit
calendar, not aggregate row counts or sums. A run that converts nothing adds
no FX terms at all, so a single-currency run keeps the exact reproducibility
identity it had before FX conversion existed.

## Tracked portfolio valuations

Tracked portfolios are separate from historical simulations. The owner enters
current quantity and average cost for each holding. Current value uses the
latest persisted `LatestMarketData` row, and a snapshot is refused when a
holding is inactive, unpriced, in another currency, or more than seven
calendar days behind the newest holding price.

Unrealized gain is securities value minus entered holding cost basis. Cash is
included in total portfolio value but excluded from gain and return, avoiding
the false treatment of cash as investment profit. Snapshot history includes
changes in cash and holdings and is therefore a value series, not a
time-weighted or money-weighted performance claim. Each snapshot and position
is immutable and retains its source asset and market-session date. A
split-sized price move with unchanged quantity is flagged for manual review;
StanStock does not silently rewrite the owner's quantity or average cost.

A StanStock sample portfolio is a distinct frozen research artifact. It
equal-weights eligible opportunities from one provider-backed analysis run,
stores that run and the construction-policy version, and records a baseline at
the analyses' immutable reference closes. Those closes are not represented as
executable fills. Later snapshots use the latest persisted market rows without
rebalancing, and model return is total current value relative to starting
capital. A split-sized move withholds that headline return until reviewed.
The current price-only sample records a short-horizon signal, so later
buy-and-hold performance is observational rather than evidence that the
original short thesis remained valid.

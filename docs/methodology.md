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

The prospective US price-only v2 policy removes nominal share-price scale from
its technical and liquidity gates. It scores the MACD histogram only after
dividing by the latest positive close, and it measures liquidity as the
20-session mean of `close * volume`. A split-equivalent transformation of
price and share volume therefore cannot change either factor. The initial
dimensionless MACD range (`-0.02` to `+0.02`), dollar-volume factor range
($1 million to $50 million), and BUY floor ($5 million/day) are explicit,
versioned policy assumptions; they were not selected by optimizing later
outcomes. Historical v1 predictions retain their original absolute-MACD and
share-volume configuration and are never recomputed or pooled with v2
performance.

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
`config/opportunities/great-opportunity-v2.yml`; it does not alter the
underlying recommendation. A highlighted result must be BUY, clear the
policy's score and confidence thresholds, remain LOW or MEDIUM risk, and have
a positive base case for the analysis mode's supported horizon. V2 also
prevents the Under $10 speculative watchlist from becoming a highlighted
new-allocation idea. The historical v1 policy remains unchanged.

Full analyses use the label `Great opportunity` and require fundamentals.
The US price-only baseline instead uses `Strong short-term setup`, checks only
its supported short horizon, and remains visibly identified as price-only.
The policy version is shown on stock detail so a later policy revision cannot
silently masquerade as the original rule.

## Current USD price bands

The opportunities page classifies each latest valid persisted USD close into
four neutral, non-overlapping affordability bands:

- `0 < price < $10`: `Under $10 - speculative watchlist`;
- `$10 <= price < $50`;
- `$50 <= price < $300`;
- `$300+`.

Each band displays the close's market-session date. Price bands are filters
and execution context only: they do not enter factor arithmetic, score,
confidence, valuation, or recommendation. The `$300+` band is not a quality
penalty.

Under $10 remains research-only with a 0% new-allocation cap. It is excluded
from current opportunity promotion. Newly constructed sample portfolios apply
the same boundaries to the immutable decision-run reference close, never a
later mutable market row, while existing holdings and frozen historical
portfolios remain visible. Its long-horizon forecast is explicitly unavailable
until point-in-time SEC, dilution/per-share, solvency/cash-runway,
Under-$10-specific dollar-liquidity, verified split-event, and compatible
3-year/5-year formula evidence exists. The configured stock universe is not
expanded merely to populate a price band.

## ETF evidence boundary

Common stocks and ADRs are the only security types accepted by the stock
analysis, prediction, opportunity, and sample-stock services. An ETF sent
through those paths fails explicitly instead of receiving a stock rating.

SPY is the first and only enabled investable ETF. The daily US workflow uses
the already-required SPY benchmark response for both benchmark evidence and
the ETF's latest market row, so ETF support consumes no additional provider
credit and creates no stock-universe membership. Its ETF page reports
split-adjusted cumulative price return, annualized close-to-close volatility,
and peak-to-trough maximum drawdown from at most 253 persisted closes ending
at the displayed market session, even if the immutable asset contains later
rows. The source asset must explicitly prove daily, split-only,
dividend-excluding price-return metadata; incompatible or missing provenance
is rejected rather than defaulted. If Twelve Data omits its optional MIC
field, the normalized asset records that ARCX came from StanStock's reviewed
SPY identity rather than pretending it came from the provider. Dividends are
not included, so none of these values is a total return. SPY has no
fundamental score, BUY/HOLD/AVOID recommendation, opportunity status, or stock
forecast.

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
permanent listing ID, source price, explicit forecast horizon, evidence role,
price provider and subject when the exact listing asset proves them, scenarios,
immutable evidence grade and source classification, confidence,
recommendation, component scores, structured calculation provenance, source
assets, configuration hash, and code revision.

Predictions are append-only. A correction uses a new model/configuration
version. Missed runs can be reconstructed for research but cannot be presented
as calls issued on time. On-time status belongs to each immutable prediction,
not only its parent analysis, so a later reissued version is excluded from live
performance evidence.

Scoring groups remain `short`, `medium`, and `long`; they are not forecast
identities. Forecast identities are `short`, `6m`, `12m`, `3y`, and `5y`,
while historical `medium` and `long` predictions retain their original labels
and 252-/756-session meanings. The corresponding canonical maturities are 10,
126, 252, 756, and 1260 observed sessions; calendar weekends and holidays are
never manufactured.

Performance reports only matured outcomes and retains unresolved corporate
events in coverage counts. Aggregate return, hit-rate, or calibration metrics
are withheld below the configured minimum sample. Decision predictions keep
the existing BUY/HOLD/AVOID success semantics. Advisory predictions never
receive a decision-success value; they record direction correctness, bear/bull
interval coverage, signed base-case error, and benchmark return separately.

The decision headline on the performance page aggregates only decision
outcomes from observed universe snapshots whose individual predictions were
issued before the next market session. Reportability uses the immutable
evidence grade, source mode, and exact price provider copied onto each
prediction at issuance; later edits to parent snapshot metadata cannot
reclassify evidence. Advisory outcomes have separate exact
method/configuration/provider/horizon cohorts and cannot enter the decision
denominator. Matured synthetic, unsupported, or later-reissued outcomes remain
visibly excluded from live, out-of-sample claims. A price-only baseline
persists only the short horizon it supports; withheld medium and long
scenarios are not prediction records and therefore cannot enter evaluation
denominators.

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
latest persisted `LatestMarketData` row. SPY can therefore be held and valued
without a `StockAnalysis`; unsupported ETF symbols are rejected. A snapshot
is refused when a holding is inactive, unpriced, in another currency, or more
than seven calendar days behind the newest holding price.

Unrealized gain is securities value minus entered holding cost basis. Cash is
included in total portfolio value but excluded from gain and return, avoiding
the false treatment of cash as investment profit. Snapshot history includes
changes in cash and holdings and is therefore a value series, not a
time-weighted or money-weighted performance claim. Each snapshot and position
is immutable and retains its source asset and market-session date. A
split-sized price move with unchanged quantity is flagged for manual review;
StanStock does not silently rewrite the owner's quantity or average cost.

External cash contributions are separate immutable events. Initial cash on a
new manual portfolio is recorded as a deposit rather than unexplained mutable
cash, and later cash is added only through the deposit workflow. Each deposit
captures an eligible pre-flow snapshot when fresh, coherent, provider-backed
price evidence exists; otherwise it preserves an explicit boundary issue.
Confirmed planner purchases are separate immutable rows, not brokerage fills.
They retain the exact listing, quantity, persisted close, market session,
price asset, policy version, and plan hash.

The versioned `monthly-allocation-v1` preview uses total NAV including cash:

```text
SPY budget = min(cash, max(0, 0.70 * NAV - SPY value))
satellite budget =
    min(remaining cash, max(0, 0.30 * NAV - other invested value))
```

It never sells. Fractional mode rounds quantities down to eight decimal
places; whole-share mode rounds down to whole units. Any residual remains
cash. At most one stock satellite is selected, and only from provider-backed
analysis on the same session as SPY whose opportunity evidence is explicitly
short-horizon. Under-$10 names remain ineligible for new allocation. The plan
hash binds portfolio settings and holdings, current price assets and sessions,
and the exact qualifying analysis/run/configuration/source-asset evidence.
Confirmation locks and recomputes that complete state; a stale preview is
rejected.

Contribution-adjusted profit/loss is:

```text
current NAV - eligible tracking-boundary NAV - later external deposits
```

The percentage divides by the boundary NAV plus those later deposits, so it is
labeled a simple since-boundary return rather than time-weighted or
money-weighted performance. Cash and quantities must reconcile exactly to the
immutable deposit and planner-purchase ledgers. Current and boundary
valuations require one fresh, non-future market session, Twelve Data
split-adjusted price-return metadata, dividend exclusion, and no unresolved
corporate-action warning.

A supported manual quantity change or removal is neither silently treated as
return nor allowed to disable the metric forever. It appends an immutable
post-change performance baseline and restarts measurement from that value.
If another holding prevents a complete valuation, an immutable
`boundary unavailable` baseline is recorded, the holding edit still commits,
and performance remains withheld. A later valid manual change can append a
newer eligible baseline. Ordinary scheduled snapshots do not themselves reset
the performance boundary.

A StanStock sample portfolio is a distinct frozen research artifact. It
equal-weights eligible opportunities from one provider-backed analysis run,
stores that run and the construction-policy version, and records a baseline at
the analyses' immutable reference closes. Those closes are not represented as
executable fills. Later snapshots use the latest persisted market rows without
rebalancing, and model return is total current value relative to starting
capital. A split-sized move withholds that headline return until reviewed.
The current price-only sample records a short-horizon signal, so later
buy-and-hold performance is observational rather than evidence that the
original short thesis remained valid. New sample construction records the
price-band policy, classifies the analysis reference close at the run's target
date, and excludes both ETFs and Under $10 names without consulting later
market state. An older frozen basket is never rewritten when the policy
advances.

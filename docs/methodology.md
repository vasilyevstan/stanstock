# Methodology

> Release state: see [README](../README.md#release-status). Exact formulas and
> the retrospective protocol are in [Price research](price-research.md).

StanStock separates the active research question, conditional price
projections, observed outcomes, retrospective comparison, and portfolio
accounting. These lanes must not be blended into one score or accuracy claim.

## Active product

`research-product-v1` has two operators:

- `us-relative-momentum-v1`: one six-month decision;
- `us-price-fhs-v1`: four advisory projections at 6m, 12m, 3y, and 5y.

Every qualified listing has one `StockAnalysis` and exactly five immutable
predictions. The 6m decision and 6m advisory row are distinct.

`overall_score`, `risk_score`, `confidence`, and
`probability_positive` are unavailable for this product. Missing values stay
null with reasons; they are not zero.

The probability-first presentation adds a separate immutable report of
fixed-model outcome shares. It does not populate those original probability
fields, change the five-row output, or establish calibration.

## Shared inputs

The active operators require an exact common XNYS-session window:

- 757 stock closes and 757 SPY closes;
- the last close is the target session;
- 756 daily log returns;
- US/USD common stock or supported ADR identity;
- immutable registered raw and normalized price sources; and
- split-adjusted price-return semantics with dividends excluded.

No missing session is filled. SEC facts are not an active input.

## Six-month decision

Momentum compares the stock and SPY over the same 12–1 window:

```text
stock = log(P_stock[T-21] / P_stock[T-252])
spy   = log(P_spy[T-21]   / P_spy[T-252])
relative = stock - spy
```

Direction:

- positive when both `stock > 0` and `relative > 0`;
- negative when both are below zero;
- mixed otherwise.

Suggestion policy:

- negative -> AVOID;
- mixed -> HOLD;
- positive -> BUY only when source checks pass, relative volatility is at
  most 2, 252-session drawdown is no worse than -50%, compatible 20-session
  dollar turnover is at least $5 million, and target close is at least $10;
- missing momentum -> unavailable.

Missing BUY-only risk/liquidity evidence changes a positive signal to HOLD. It
does not erase a valid negative AVOID.

Risk labels are relative to SPY's filtered annualized volatility:

- low: ratio <= 1;
- medium: 1 < ratio <= 2;
- high: 2 < ratio <= 3;
- very high: ratio > 3;
- insufficient when unavailable.

These are policy categories, not calibrated loss probabilities.

## Advisory projections

The fixed-parameter FHS operator filters 756 daily log returns, discards a
252-return burn-in, centers/rescales the remaining 504 residuals, and samples
them independently with replacement.

It uses 8,192 deterministic PCG64 paths and reports cumulative price-return
quantiles at 126, 252, 756, and 1260 sessions:

- Lower = p20;
- Median = p50;
- Upper = p80.

Lower-to-Upper is central 60% model mass. It is not real-world coverage, a
gain probability, or a confidence interval. The full operator and equations
are in [Price research](price-research.md#filtered-historical-simulation).

Historical drift extrapolation and the same-shock zero-drift sensitivity are
shown separately. Their difference exposes drift dependence; it is not a
third model or proof that either path is likely.

### Probability-first presentation

The same terminal paths are counted in three exhaustive ranges: **Loss**
(`R < 0`), **Flat to +20%** (`0 <= R <= 0.20`), and **Above +20%**
(`R > 0.20`). Displayed percentages are shares of model simulations, not
validated real-world odds. Median return remains visible, with the
p20/p50/p80 price range and exact counts in details.

The nested below-minus-20% event concerns the ending return, not an interim
drawdown. All bands use cumulative price returns over the selected horizon;
none is annualized. A later-derived report has its own availability timestamp,
cannot enter an earlier as-of read, and cannot inherit prediction on-time
status. Missing summary evidence does not hide an otherwise valid projection.

See [Model-estimated probabilities](price-research.md#model-estimated-probabilities)
for the exact event, rounding, provenance, and nonclaim contract.

## Volume provenance

The frozen v1 Twelve Data price contract documents split adjustment for prices,
not a compatible adjustment basis for volume. StanStock does not infer
split-compatible volume. Consequently, projections and raw direction can be
available while dollar turnover is withheld and BUY remains blocked.

The synthetic demo explicitly marks its generated volume as compatible, but
that is code-path validation, not provider evidence.

Bounded paired-mode controls also found that changing the provider's
adjustment mode changes historical volume, not just prices. Post-event
agreement in one control does not establish a universal share-volume
contract. These private controls are not production assets and do not
retroactively make the frozen v1 inputs liquidity-compatible.

`research-product-v2` introduces a separate, purpose-qualified liquidity
evidence lane. It requires exact raw Decimal-close and integer-volume parity
between `none` and `splits` over the same 20 XNYS sessions, plus a fresh
same-batch split-mode control. Its closing-price-times-volume mean is an
activity proxy, not exact transaction-dollar turnover or universal proof of
provider volume semantics. Missing evidence blocks BUY, not independently
valid direction or projections; a corrupt registered proof is an integrity
failure, not an ordinary missing-input fallback.

Only this eligibility boundary is versioned. Momentum, other BUY gates and
FHS numerical inputs, seed configuration and forecasts remain unchanged for
identical split-adjusted inputs. V1 histories are not relabelled. See
[Version 2 liquidity evidence](price-research.md#version-2-purpose-qualified-liquidity)
for source isolation, the bounded control and the separate prospective
declaration.

## Under-$10 policy

When history qualifies, Under-$10 names receive the same research and
projections. The target-date affordability restriction remains separate from
the raw signal:

- 0% new allocation;
- no BUY display promotion;
- no highlight;
- no new sample-basket admission.

Negative momentum may still display AVOID. Existing holdings can receive
read-only review wording without altering any ledger.

## Outcomes

The decision matures after 126 observed sessions:

- BUY success requires positive stock return and outperformance of SPY over
  the same endpoints;
- AVOID success requires negative stock return and underperformance of SPY;
- HOLD/unavailable has `success = null`.

FHS advisory rows have no recommendation-success metric. Their recorded
outcomes use:

- realized cumulative split-adjusted price return;
- signed median error;
- absolute error of the median forecast (aggregated as MAE);
- p20/p50/p80 pinball losses;
- p20–p80 inclusion and width; and
- central-60% interval score.

An all-null advisory is non-evaluable before price lookup and is excluded from
advisory denominators.

Aggregate performance counts the earliest reportable prediction once per
exact listing/target/horizon/role/method/config/provider observation. Later
valid reissues remain immutable ledger rows but do not duplicate one market
observation.

## Forward scorecard

`forward-forecast-scorecard-v1` declares an immutable owner-scoped evaluation
protocol before reading matured losses. A declaration made after an included
forecast's issuance but before maturity is labeled accordingly; it is not
pre-issuance preregistration. Equal timestamps do not prove prior declaration.

The scorecard uses verified historical runs and the existing earliest
reportable observation rule before considering outcome status. It keeps
decision/advisory roles, horizons, methods, configurations, providers and source
grades separate. BUY and AVOID success are separate; HOLD has no success metric.
Advisory losses use the existing definitions above. The recorded zero-log-drift
sensitivity is a paired comparator only where both models have the same
supported observations; missing comparator evidence does not erase an otherwise
evaluable native forecast.

Results are averaged within target cohorts, then equally across cohorts.
Reports retain raw rows, distinct listings, cohort support, calendar span and
pending, unresolved, corporate-event and non-evaluable denominators. Overlapping
daily forecasts are not independent trials. Before maturity, metrics remain
null and the conclusion is **unmatured; skill not established**, not zero
accuracy. This protocol neither activates the separate drift study nor
establishes calibrated probabilities or investment skill.

## Retrospective comparison

The retrospective comparison is current-universe/current-vintage research. It compares
FHS with zero-log-drift and historical-log-drift Gaussian baselines on the
same paired listing/anchor/maturity support. Means are first calculated inside
target cohorts, then equally across target cohorts.

MAE is the mean absolute error of the median forecast. Pinball loss and
interval score are lower-is-better. Width and inclusion are descriptive and
must not be treated as standalone rankings.

The study preserves empty partitions, unavailable comparisons, and
worse-than-baseline results. Real losses use recorded realized returns, never
simulated paths as market observations.

The [October 2026 retained-data checkpoint](price-research.md#october-2026-retained-data-checkpoint)
found worse FHS median MAE and interval scores than the zero-log-drift
Gaussian on the inspected six- and twelve-month final paired scope.
Support was only three and one target cohorts respectively, with no eligible
three- or five-year empirical anchors. This reconstruction does not establish
observed skill, justify tuning against inspected outcomes, or activate a
prospective study.

## Research grounding and nonclaims

The active operator design is motivated by:

- Jegadeesh and Titman (1993), momentum portfolios:
  <https://doi.org/10.1111/j.1540-6261.1993.tb04702.x>
- Carhart (1997), momentum-factor context:
  <https://doi.org/10.1111/j.1540-6261.1997.tb03808.x>
- Bollerslev (1986), conditional variance:
  <https://doi.org/10.1016/0304-4076(86)90063-1>
- Barone-Adesi, Giannopoulos, and Vosper (1999), filtered historical
  simulation motivation;
- J.P. Morgan/Reuters (1996), *RiskMetrics—Technical Document*, fourth
  edition.

StanStock does not reproduce those papers' universes, portfolio construction,
fitted coefficients, total returns, costs, or empirical claims. Its exact
lookbacks, thresholds, coefficients, and recommendation rules are versioned
product policy.

## Frozen archives

Earlier score-led, empirical-range, and SEC-backed methods retain their
original config bytes/hashes, reason wording, payloads, predictions, outcome
semantics, and archive presentation. The active product never reinterprets
their historical `medium`/`long` roles or silently selects them as a fallback.

The complete version-specific formulas, eligibility gates, and withholding
definitions remain available in the
[archived methodology at the last pre-redesign release](https://github.com/vasilyevstan/stanstock/blob/4e22eb35b09f3c803f4de94c9e9df9e839ab5574/docs/methodology.md).
That pinned reference covers `us-price-baseline-v1/v2/v3`,
`us-price-medium-v1/v2`, and `us-sec-long-v1/v2/v3/v4`; it is not the active
product specification. Historical evidence should be interpreted using its
recorded version, not the new momentum/FHS rules.

## Portfolio policy and versioned simulation accounting

Research suggestions are not orders. Existing personal portfolio bookkeeping,
frozen sample baskets and the legacy contribution planner retain their
contracts. The current six-month momentum decision is not silently substituted
for the legacy planner's short-horizon satellite. Existing safeguards remain:

- one explicit base currency per multi-currency simulation;
- dated point-in-time FX for each valued date;
- common inception for selected holdings;
- complete input/calendar hashing;
- missing/stale/ambiguous FX failure; and
- immutable deposits, purchases, performance boundaries, and valuations.

New simulation workflows explicitly select `simulation-accounting-v2`.
Configurations without an accounting-version marker retain legacy behavior,
including historical result bytes; unknown versions are rejected. No stored
simulation is rewritten.

In v2, portfolio and benchmark daily returns must share both adjacent
simulation-session endpoints. A missing benchmark observation cannot turn a
multi-session benchmark return into a one-session comparison. Tracking error
and related paired statistics retain minimum-support and zero-variance
withholding. Full-period benchmark and excess returns require both exact
boundary prices. An interior gap can leave those endpoint totals calculable
while withholding benchmark path drawdown. Comparison coverage and reasons are
part of the versioned result provenance.

Portfolio drawdown includes pre-trade starting capital in its high-water mark,
so inception transaction costs count as a loss. The curve and summary use the
same definition; the first row's daily-return convention is unchanged.
Historical v1 detail pages disclose the benchmark-gap and inception-drawdown
limitations rather than presenting them as corrected metrics.

V2 immutable provenance binds the definition and universe-snapshot identities
as well as the effective execution configuration. Both the loader and detail
page checksum-read the registered result and inputs. Reparented runs or missing
or corrupt files withhold financial output. Detail-page mode and grade labels
come from the recorded execution, not later edits to mutable parent configuration.

### Current-method paper policy

`current-method-paper-6m-v1` is a separate prospective research experiment,
not a personal portfolio or broker order. Its USD 100,000 is **notional**, not
a deposit. It allocates 70% to SPY and equally divides a 30% stock sleeve among
all qualifying native current-method six-month BUY decisions. These allocation
weights are policy assumptions, not optimized or scientifically established
weights. Advisory projections never determine selection.

All native identity, authorization, risk, compatible-volume and affordability
gates remain mandatory. Under-$10 new allocation remains zero. A valid empty
BUY set leaves the stock sleeve as cash; missing or invalid required evidence
withholds the plan rather than pretending that the set was empty.

One immutable activation per owner and policy version records its actual
publication time and frozen selection. The first hypothetical execution is
the XNYS session close strictly after activation's New York calendar date.
Quantities then remain fixed for 126 sessions: no contributions, rebalancing
or invented terminal liquidation. Costs are 10 bps transaction cost plus 5 bps slippage.
Dividends and cash yield are excluded. The SPY-only comparison uses the same
notional, inception, calendar, costs and price-return basis.

Activation, execution and measurement use append-only research assets, not
portfolio, purchase or generic simulation-run rows. Missing prices or sessions
and incompatible share-basis evidence remain explicit; later retrieval must
not manufacture an earlier observed paper curve. The fixed experiment can
produce cash, unavailable valuations or adverse results without implying an
application failure or a profitable strategy.

#### Paper v2: terminal evidence and path risk

`current-method-paper-6m-v2` retains those notional, allocation, cost, calendar
and fixed-quantity assumptions. Its policy hash is
`e1839e1bdd25236de47dcaa421def82fdfb98e0ac46b7d0f7261cf74fee22a61`.
It uses separate immutable assets; v1 artifacts and their complete-measurement
semantics are unchanged. Both policies remain bound to the v1 research
product and momentum decision, not a subsequently selected product version.

A v2 terminal return requires timely entry and terminal captures plus verified
share-basis continuity. Freeze 757 authoritative closes ending at entry;
each valuation source must cover that reference and the complete held
calendar, with consistent recorded overlaps. Two endpoints alone are not
enough. Missing intermediate captures do not erase independently supported
terminal results, but retained historical rows never become backfilled
prospective captures.

Drawdown, volatility and Sharpe require a complete prospectively captured
common path. Report coverage and the through-session separately; missing
captures withhold path risk. Terminal financial values remain null before
the terminal session and its valid capture. Corruption is an integrity
failure, not an absent capture. Native path-number insufficiency can withhold
risk independently of representable decimal terminal accounting.

Attribute the stock sleeve against same-dollar SPY separately from the whole
account. The full-notional native SPY reference is scaled by exactly 7/10
and 3/10; P&L reconciles fixed-quantity price movement, entry costs and
independently frozen native entry roundoff. An empty sleeve measures a
cash-allocation effect, not stock-selection skill. This does not establish
alpha, optimal weights or investment eligibility.

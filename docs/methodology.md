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

The current Twelve Data price contract documents split adjustment for prices,
not a compatible adjustment basis for volume. StanStock does not infer
split-compatible volume. Consequently, projections and raw direction can be
available while dollar turnover is withheld and BUY remains blocked.

The synthetic demo explicitly marks its generated volume as compatible, but
that is code-path validation, not provider evidence.

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

## Retrospective comparison

The registered study is current-universe/current-vintage research. It compares
FHS with zero-log-drift and historical-log-drift Gaussian baselines on the
same paired listing/anchor/maturity support. Means are first calculated inside
target cohorts, then equally across target cohorts.

MAE is the mean absolute error of the median forecast. Pinball loss and
interval score are lower-is-better. Width and inclusion are descriptive and
must not be treated as standalone rankings.

The study preserves empty partitions, unavailable comparisons, and
worse-than-baseline results. Real losses use recorded realized returns, never
simulated paths as market observations.

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

## Unchanged portfolio and simulation methodology

The price-product redesign does not turn suggestions into orders or change
portfolio/simulation accounting. Existing safeguards remain:

- one explicit base currency per multi-currency simulation;
- dated point-in-time FX for each valued date;
- common inception for selected holdings;
- complete input/calendar hashing;
- missing/stale/ambiguous FX failure; and
- immutable deposits, purchases, performance boundaries, and valuations.

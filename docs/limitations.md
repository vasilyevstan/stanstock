# Limitations

> Release state: see [README](../README.md#release-status). The replacement
> product is implemented but not yet independently accepted or activated.

## Research-product limits

- The two active operators are fixed mathematical policies, not fitted ML and
  not replications of the cited papers.
- Literature motivation is not proof of profitability, calibration,
  statistical significance, or live forecasting skill.
- Momentum measures a skipped-month historical relationship to SPY. It is not
  a probability or causal forecast.
- FHS extrapolates the stock's trailing three-year mean log return and
  fixed-parameter variance dynamics. Regime change, parameter uncertainty,
  delisting, corporate events, and survivorship effects can dominate,
  especially at 3y and 5y.
- Residuals are sampled independently with replacement. Dependence beyond the
  variance recursion is not claimed to be preserved.
- Lower/Median/Upper are p20/p50/p80 model quantiles. The interval contains
  60% of simulated model mass, not demonstrated 60% real-world coverage.
- Median is not mean. Lower is not a worst case, stop price, or guaranteed
  floor.
- Probability and confidence are null, not zero.
- 8,192 production paths and 16,384-path diagnostics describe numerical
  approximation, not additional financial evidence.
- Missing or invalid calculation components are withheld rather than repaired
  with floors, clipping, interpolation, jitter, or success-shaped defaults.

## History and source limits

- Qualification needs 757 consecutive common stock/SPY XNYS closes ending
  exactly at the target. A short monitoring asset is not adequate history.
- All active returns are split-adjusted price returns excluding dividends.
  They are not total returns.
- The current approved live path is US/USD only. European equity prices remain
  deferred.
- A research-grade retrospective may use an immutable provider source
  retrieved later than an anchor, but must label that current-vintage
  limitation and clip rows through the anchor. It is not historical observed
  availability.
- Execution revision, source-run revision, report generation, and report
  registration availability are separate facts.

## Volume and BUY eligibility

The currently documented Twelve Data split-adjustment contract establishes
the price basis but not a compatible split-adjusted volume basis. StanStock
does not infer volume compatibility from adjusted prices.

Consequences:

- raw direction, relative volatility, drawdown, and all four projections can
  remain calculable;
- 20-session dollar turnover can remain unavailable;
- missing/incompatible turnover blocks BUY; and
- a positive signal can remain HOLD with an explicit liquidity reason.

A negative direction can still produce AVOID because missing BUY-only evidence
does not erase independently valid negative momentum.

## Under-$10 limits

A qualified Under-$10 listing can receive momentum research and all four
price projections. It remains:

- 0% new allocation;
- ineligible for BUY promotion and highlights;
- excluded from newly constructed sample baskets; and
- a speculative watch rather than proof of undervaluation.

Price-only projections do not establish solvency, dilution safety, a verified
corporate-action history, or portfolio eligibility. Frozen historical
Under-$10 records retain their original definitions.

## Retrospective and performance limits

- Current-universe/current-vintage reconstruction is exposed as research, not
  observed live skill.
- Development, validation, and final holdout are separate. Crossing intervals
  are purged and holdout tuning is forbidden.
- FHS/baseline comparisons use identical paired listing/anchor/maturity
  support and equal target-cohort means.
- Mean absolute error measures the absolute error of the median forecast.
  Pinball loss and interval score are lower-is-better.
- Interval width and inclusion are descriptive. A narrower range or higher
  inclusion is not automatically better; interval score jointly penalizes
  width and misses.
- Empty long-horizon partitions, unavailable results, and
  worse-than-baseline outcomes must remain visible.
- Real comparison losses come from recorded realized returns. Simulated paths
  are never treated as realized market support.
- A registered retrospective report can still fail to establish skill.

## Observed issuance limits

Observed evidence requires more than an `observed` snapshot:

- each immutable prediction version proves its own on-time status;
- the source cutoff must be safe;
- the request must occur before the next regular XNYS session open;
- production config, provider, benchmark, owner authorization, and exact clean
  committed revision must be bound; and
- an unsafe explicit request raises rather than silently downgrading.

`manage.py analyze` and manual `daily --region us` are always research-grade.

## Provider and privacy limits

Broad unattended US/European OHLCV remains `NO_GO`. Twelve Data is a
conditional private US path only after a non-demo key and the account's
required personal/internal-display rights are confirmed. The technical guard
is not legal advice. Data must not be redistributed.

Stooq remains `NO_GO`; StanStock will not bypass automation controls. SEC,
ECB, and filings.xbrl.org have separate capabilities and limitations but are
not prerequisites for the active price product.

Authenticated output is owner-bound. Public deployment or multiple display
users require rights that cover that audience.

## Portfolio and simulation limits

- Research suggestions never place orders.
- Tracked portfolios are local research accounting, not brokerage ledgers.
- Recorded purchases use persisted closes, not claimed fills.
- Withdrawals, tax lots, realized gains, commissions, spreads, taxes, and
  broker execution are not modeled.
- Contribution performance is a simple since-boundary return, not
  time-weighted or money-weighted performance.
- Existing portfolio/simulation policies are not redesigned by the price
  product.
- Multi-currency simulation requires dated point-in-time FX into one base
  currency; missing, stale, ambiguous, or late-published rates fail.

## Deployment limits

There is no promise of continuously free hosting. A durable deployment needs
private PostgreSQL, asset and backup storage, HTTPS, secret management, and
enough capacity for the configured universe. The intended topology is one
private instance; multi-replica coordination needs separate review.

# Limitations

## Live prices are US-only and conditional

The original free-only, unattended, roughly 500-stock US/Europe requirement
remains `NO_GO`. A reduced 100-stock US starter universe is technically
supported through Twelve Data's official API, but it is disabled by default
and conditional on account rights.

- Stooq's public CSV path is protected against automation in the tested
  environment.
- Its official automation and private-retention terms could not be
  machine-verified.
- No browser challenge or access control will be bypassed.
- Alpha Vantage's official free limit is too small for the requested universe.
- Twelve Data's pricing page labels Basic as internal non-display, while its
  August 2026 support guidance says Individual plans may be used for personal,
  non-commercial internal tools and prohibits redistribution and commercial
  display. StanStock supports Basic only under the owner's explicit
  single-user personal-use attestation, records the licensed user, blocks
  every other authenticated user, and stops provider jobs if another active
  account exists. This technical guard is not a substitute for confirming the
  account's current terms with Twelve Data.
- Twelve Data data cannot be redistributed or publicly displayed without
  appropriate rights, and current terms require its deletion after the
  subscription or agreement ends. Because provider assets feed immutable
  provenance and derived records, the supported deletion path is a full
  installation reset, including databases, assets, backups, replicas, and
  snapshots; selective provider purging is intentionally unsupported.
- European live equity coverage is still deferred.

Consequently, the default application remains deterministic synthetic
research. US live rankings become available only after explicit
`configure_twelve_data` activation with a non-demo key and the confirmation
appropriate to the selected plan. Historical catch-up runs are research-grade
and cannot be presented as predictions issued on time.

Twelve Data daily histories are requested with split adjustment only. They
exclude dividends, so all derived performance is price return, not total
return. The local quota ledger cannot observe credits consumed by another
application using the same account, and provider coverage, symbols, plan
entitlements, timing, and terms can change independently of this code.
See `docs/operations.md` for the destructive termination procedure.

## ETF scope

- SPY is the only enabled investable ETF. It reuses the single benchmark
  series already fetched by the US workflow; StanStock does not request a
  second SPY series for portfolio use.
- SPY is not part of the 100-stock universe and receives no stock
  fundamentals, factor score, BUY/HOLD/AVOID recommendation, opportunity
  rank, prediction, or sample-basket allocation.
- The ETF page reports trailing split-adjusted price behavior only. Return
  excludes dividends, volatility is historical rather than predictive, and
  drawdown is limited to the available persisted window.
- Portfolio valuation can price SPY without a `StockAnalysis`, but no other
  ETF symbol is enabled in this release.

## Tracked portfolios

- Tracked portfolios are research-accounting records, not brokerage ledgers.
  They record immutable external deposits and confirmed planner purchases,
  but not withdrawals, tax lots, realized gains, commissions, bid/ask
  spreads, taxes, or actual broker fills.
- A confirmed planner purchase changes only the local portfolio ledger. It
  sends no order and makes no provider request. Its price is the latest
  eligible persisted close, not a claim that the owner could execute at that
  price.
- The monthly planner is USD-only, targets 70% of total NAV in SPY and at most
  30% in one explicitly short-horizon qualified stock, never sells, and
  carries unused cash. These fixed v1 targets are policy assumptions, not an
  optimized allocation model.
- Contribution-adjusted return is a simple return since the active immutable
  boundary. It is not time-weighted or money-weighted, and dividends remain
  excluded. All-time deposits are shown separately from the deposits applied
  after the current boundary.
- Manual quantity changes and removals create a visible immutable performance
  restart instead of being counted as investment return. If the post-change
  portfolio cannot be valued, the edit succeeds but percentage performance is
  withheld from an explicit unavailable-boundary record until a later valid
  manual baseline supersedes it.
- Holdings are restricted to the portfolio base currency even though the
  simulation engine has point-in-time FX support. This is a deliberate first
  release boundary, not an implicit conversion.
- Contribution performance and planner execution require compatible Twelve
  Data price evidence from one session. Stale, future-dated, mixed-session,
  non-split-adjusted, dividend-ambiguous, or missing evidence is withheld.
- Prices may be split-adjusted while user-entered quantities are not. A
  split-sized move with unchanged quantity remains flagged across later
  snapshots; the owner must explicitly restate quantity and average cost.
  That quantity change creates a new performance baseline rather than
  retroactively rewriting earlier values.
- StanStock sample portfolios use immutable analysis reference closes so their
  construction can be reproduced. They are research-reference baskets, not
  claims of an executable same-close fill. Their composition is frozen, they
  do not rebalance, and the current price-only signal is intended for 1-10
  trading days even though the basket can remain visible afterward.
- Nominal price bands are affordability context, not evidence of
  undervaluation. Under $10 is a non-investable speculative watchlist for new
  allocations and sample construction. Existing holdings remain trackable,
  but no 3-year/5-year forecast is shown until SEC, dilution, solvency,
  dollar-liquidity, verified split-event, and compatible formula evidence is
  available.
- Sample-portfolio return is current total value versus starting capital. It
  is a split-adjusted price return excluding dividends. A possible split
  suppresses the headline return until its quantity basis is reviewed.

## Fundamentals

- SEC submissions and Companyfacts are suitable for US filing vintages, but
  every environment must pass a bounded preflight with a compliant identifying
  User-Agent and fair-access limits before the provider is enabled. Network or
  provider failures remain explicit rather than being interpreted as missing
  company fundamentals.
- filings.xbrl.org states that its repository is incomplete and explicitly
  identifies Germany and Ireland as missing. Repository-added time can lag the
  authority filing time.
- European coverage is annual-report-first and is not equivalent to US
  quarterly freshness.

## FX

ECB reference rates are informational observations, not transaction rates.
They are generally published around 16:00 CET and describe market conditions
around 14:15 CET. Same-day rates cannot be used before publication.

The legacy ECB history CSV showed anomalous rows during the spike. StanStock
prefers the SDMX API or daily XML and must validate every observation.

The simulation engine converts native prices into one explicit base currency
with dated, point-in-time rates. The conversion inherits every weakness of its
inputs: reference rates are not transaction rates, so a converted portfolio is
not a claim about executable cross-currency trading, and no FX bid/ask spread,
conversion commission, or hedging cost is modeled. Rates are carried forward
across market closures within a bounded window rather than interpolated, so a
value dated inside a closure is priced at the last observation, not at an
estimate of that day's true rate. A holding whose own market is closed keeps
its currency exposure -- its last native quote is revalued at the current
rate -- but its *stock* price is still stale for as long as the closure
lasts.

FX availability is resolved only to end-of-day, because the source vintages
record no intraday publication knowability. A currency-converted run is
therefore limited to close-based execution; opening-price execution bases are
rejected rather than modeled with an intraday cutoff the data cannot support.

The stock-versus-FX split is reported only when it is exact -- the same
quantity path restated at each currency's inception rate. It is a
decomposition of the reported result, not an attribution of skill, and it is
withheld entirely when a cash settlement or a missing reference rate makes it
unmeasurable.

## Forecast evidence

- Confidence is heuristic until calibration evidence exists.
- Probability of positive return is withheld below its configured sample
  threshold.
- The explicit `6m` and `12m` price-only engine uses the current configured
  universe's history. Its backfilled panel is therefore survivorship-biased
  research evidence, not proof of live skill. Probability is additionally
  withheld unless effective non-overlapping cohort support, listing diversity,
  calendar span, matched market-regime breadth, and fixed walk-forward
  calibration gates pass.
- Point-in-time SEC facts are available for the configured US universe, but
  Companyfacts excludes custom issuer concepts and segment dimensions. Current
  SIC snapshots are not historical classifications, ambiguous taxonomies stay
  missing, debt components are not promoted to a total unless they are
  compatible and non-overlapping, and banks/financials/REIT-like accounting
  may remain unsupported by long v1.
- The deterministic SEC-backed `3y` and `5y` engine is intentionally narrow.
  It requires compatible positive FCF/share or a separately eligible EPS/share
  branch, reported diluted-EPS share-basis evidence, sustainable-growth
  inputs, and a same-family SIC peer floor. Missing tax, invested capital,
  peers, annual history, classification, or compatible price provenance
  produces `Insufficient evidence`; negative or inconsistent FCF cannot
  silently switch to EPS. Existing `medium` and `long` rows remain legacy
  identities rather than being relabeled as exact horizons.
- Advisory `6m`/`12m` and `3y`/`5y` forecasts require a complete eligible
  universe snapshot so their cohort and peer context is immutable. The
  single-listing analysis service intentionally issues only the decision
  prediction path; use the snapshot or daily workflow for advisory horizons.
- Long scenarios are deterministic advisory cases, not statistically
  calibrated target-price probabilities. Their positive-return probability is
  unavailable at launch, dividends are excluded, current SIC is not historical
  industry membership, and historical company valuation normalization remains
  out of scope until a compatible split-factor or unadjusted-price source
  exists.
- Long-v1 verifies the SEC diluted-share basis through the latest metric
  period, including every selected annual EPS/share period and TTM-to-annual
  continuity. It cannot verify a split between that period and the forecast
  target from split-adjusted prices alone. Each prediction exposes the
  bounded number of unverified post-period days; this is residual risk, not
  evidence that a split occurred or did not occur. Long-v1 is frozen and
  never checks continuity between adjacent selected annual diluted-share
  bases; its pinned configuration hash and prior predictions are unaffected
  by later configuration versions.
- Long-v2 is the default configuration and retains every long-v1 assumption
  except that it enables the adjacent-period diluted-share basis continuity
  capability: checking continuity between every adjacent pair of selected
  annual periods (same 15% tolerance). An incompatible adjacent-period
  diluted-share basis is withheld as unverified continuity, never asserted as
  a confirmed split. Persisted output is not otherwise byte-identical: long-v2
  also persists the structured assessed share-consistency evidence on
  share-basis failures, while frozen long-v1's withheld-failure payloads are
  unchanged. Long-v1 and long-v2 predictions carry distinct method versions
  and configuration hashes and are never pooled into the same performance
  cohort.
- Long-v1 withholds raw current FCF/share or EPS/share multiples below the
  configured family floor because raising a cheap multiple to that floor
  before reversion would overstate return. High raw multiples retain the
  actual price denominator and use the configured cap only as a conservative
  reversion anchor.
- Legacy scenario columns remain in storage for rollback compatibility. New
  application reads use the schema-versioned `forecast_scenarios` document;
  the old columns can be retired only after all supported deployments have
  crossed this migration and rollback is no longer required.
- Long-horizon scenarios are explicit fundamental cases, not precise
  statistically validated forecasts.
- Simulated or reconstructed performance is not live performance.
- Price returns must not be described as total returns when dividend data is
  absent.

## Deployment

No continuously free hosted service is claimed. A valid deployment needs
durable PostgreSQL and asset storage, HTTPS, backups, and enough capacity for
the configured universe.

The current development machine's Docker Desktop storage is saturated by
unrelated images, volumes, and build cache. Repository configuration can be
validated there, but a fresh image build requires safe targeted cleanup or a
different host; unrelated shared Docker data must not be pruned.

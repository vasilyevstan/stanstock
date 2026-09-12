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
  allocations and sample construction. Existing holdings remain trackable.
  Long-horizon activation remains unavailable pending joint Under-$10 review
  and candidate-specific eligibility. Released reusable foundations are
  point-in-time SEC facts with adverse-versus-missing branch behavior, the
  long-v2 diluted-share/per-share continuity assessment with withholding (not
  post-period event verification), and the deterministic 3-year/5-year formula
  engine with missing-input withholding; none establishes candidate
  qualification. The solvency/cash-runway and 252-session dollar-liquidity
  capabilities are released as unactivated shadow diagnostics (see below). The
  only still-unreleased activation control is a verified split/reverse-split
  event source. Previously issued immutable long-horizon ledger evidence
  remains visible, with its original horizon and evidence role preserved, and
  is labeled with the current activation context.
- The `us-under10-shadow-v1` assessment is diagnostic only and deliberately
  conservative. It is recorded on **newly created** qualifying analyses and
  nothing is backfilled, so an absent `data_quality["under10_assessment"]` key
  means the analysis was never assessed -- not that it failed. Because a
  missing debt component is missing rather than zero, all five balance-sheet
  inputs must share one period end, and a metric older than 200 days is stale,
  many candidates land in `insufficient_evidence`. That is the intended honest
  answer, not a defect. “Assessed SEC facts” includes only fixed canonical
  concepts whose fact row and source asset both identify SEC; foreign or
  provider-mismatched rows are excluded from calculation, lineage, and the
  on-time SEC asset cutoff check.
- The Under-$10 dollar-volume diagnostic has **no threshold** and can never
  pass an activation gate on its own. Split-only adjustment is proven for
  Twelve Data prices but not for its reported volume, so the metric's basis is
  recorded as `provider_reported_unverified_split_basis` even when a number is
  computed. A synthetic demo price asset carries no reviewed basis metadata at
  all, so the diagnostic is withheld rather than estimated.
- Verified split and reverse-split evidence is unavailable for every provider.
  The recorded Twelve Data Basic plan is not entitled to a corporate-actions
  feed, and no reviewed corporate-actions source is integrated for any other
  provider; a different plan alone would not supply one. Split events are never
  inferred from adjusted prices, share-count discontinuities, or SEC facts, so
  Under-$10 candidate activation cannot pass in this version.
- The analysis pipeline persists the Under-$10 assessment in
  `StockAnalysis.data_quality`, but that JSON field is mutable and has no
  model or database immutability guard. Only protected evidence such as
  `Prediction`, `DataAsset`, and `FundamentalFact` supplies the independent
  replay boundary. The shadow assessment's `assessment_hash` and
  `policy_hash` are recomputation checksums for a canonical payload --
  corruption detection, not signatures or proof that the row was never
  modified.
- The stock-detail reader only renders a stored assessment after binding it
  exactly to its parent decision: the permanent `Listing.id`, the parent
  analysis run's own exact target date and data cutoff, its exact immutable
  decision-run reference close and currency (cross-checked against the
  immutable original decision predictions), and the exact immutable
  price-asset UUID *and* content checksum recorded alongside it. A matching
  checksum alone never substitutes for any of these -- two unrelated
  candidates can share every other field on the same decision date, and
  copying an entire genuine `data_quality` blob from one analysis to another
  moves every other internal anchor along with it. Before display, the reader
  replays the accepted builder from the original immutable decision-prediction
  provenance, the exact cutoff-clipped price asset, and the exact
  cutoff-qualified SEC facts. Only canonical equality with the stored
  solvency and liquidity blocks renders; missing, unreadable, or mismatched
  evidence is withheld and never written back.
- The recorded provider plan describes capability context at assessment
  generation time. It is not evidence of a historical entitlement, and the raw
  plan label is never persisted into the assessment.
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

- Analysis confidence is a heuristic evidence score, and prediction
  confidence is a support/coverage heuristic. Neither is a statistical
  confidence level.
- The medium positive-return estimate is shrinkage weighted and withheld until
  its support/diversity and probability-publication gate passes. The persisted
  `empirical_calibrated` status means only that support, base-case MAE
  comparisons against unconditional and SPY-relative baselines, and the
  configured absolute Brier threshold passed. It does not prove calibrated
  probabilities or calibrated interval coverage.
- The explicit `6m` and `12m` price-only engine uses the current configured
  universe's history. Its backfilled panel is therefore survivorship-biased
  research evidence, not proof of live skill. The positive-return estimate is
  additionally withheld unless effective non-overlapping cohort support,
  listing diversity, calendar span, matched market-regime breadth, and fixed
  publication gates pass. Its matched and unconditional cohort-weighted
  p20/p50/p80 estimates are each shrinkage blended; base is blended p50 and
  bear/bull are blended p20/p80. Bear-to-bull is a nominal central 60%
  analog-return range, not a calibrated prediction/credible/confidence
  interval, and it has no coverage guarantee.
- `us-price-medium-v1` remains the default and scheduled implementation.
  Explicit `us-price-medium-v2` runs are research-only and require the current
  eligible US/USD stock universe, so their reconstructed evidence remains
  survivorship-biased. V2's one-CDF p20/p50/p80 range, positive-return
  estimate, prequential Brier skill, base MAE comparisons, coverage, strict
  miss rates, mean width, and interval scores are descriptive historical
  evidence. Positive Brier skill is not proof of calibrated probabilities,
  statistical significance, profitability, alpha, or observed live skill.
  No bootstrap, significance test, reliability calibration, CRPS, or median
  width is reported. See
  [`docs/methodology.md`](methodology.md#explicit-research-only-medium-v2).
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
  identities rather than being relabeled as exact horizons. Its bounded tax
  input is TTM GAAP income-tax expense divided by TTM pretax income, an
  accrual proxy rather than cash taxes paid or a cash tax rate. Reinvestment
  is compatible balance-sheet invested-capital change divided by NOPAT, an
  accounting proxy rather than observed capex or a proven causal rate. The 3y
  and 5y calculations use separate frozen fade and multiple-reversion paths;
  neither is a slice or extrapolation of one shared 5y path.
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
- Decision metrics display after 30 canonical row-level prediction
  observations in the relevant cohort. That fixed threshold is presentation
  policy, not statistical validation.
- Advisory metrics require non-overlapping target-date cohorts, at least 30
  listings in every selected cohort, and fixed horizon-specific calendar
  spans. These floors reduce obvious overlap and concentration; they do not
  prove independence, calibration, or statistical power.
- Advisory evidence is separated by exact code revision because StanStock has
  no independently governed implementation-equivalence digest. Even a
  documentation-only commit therefore starts a new reporting group. Invalid
  revisions, malformed outcome evidence, or more than 50,000 grouped
  target-date summaries withhold metrics rather than being normalized,
  discarded, or truncated.
- The 3y floor needs selected targets spanning six years before the last
  three-year outcome can mature, so publication takes roughly nine years. The
  5y floor needs targets spanning ten years plus the final five-year outcome,
  or roughly fifteen years. Exact-revision separation can extend those
  accumulation periods further.
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

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

## Tracked portfolios

- Tracked portfolios are current-position trackers, not brokerage ledgers.
  They do not record tax lots, realized gains, commissions, deposits, or
  withdrawals as transactions.
- Historical total value includes any manual holding or cash changes and must
  not be interpreted as a flow-adjusted return series.
- Holdings are restricted to the portfolio base currency even though the
  simulation engine has point-in-time FX support. This is a deliberate first
  release boundary, not an implicit conversion.
- A snapshot is withheld if a holding has no current price or is more than
  seven days behind the newest holding price.
- Prices may be split-adjusted while user-entered quantities are not. A
  split-sized move with unchanged quantity is flagged, but the owner must
  restate quantity and average cost. Dividends are excluded unless the source
  explicitly states otherwise.
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

- SEC submissions and companyfacts are suitable for US filing vintages, but
  SEC access returned HTTP 403 from this execution environment. Deployment
  must verify access using a compliant identifying User-Agent and fair-access
  limits.
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

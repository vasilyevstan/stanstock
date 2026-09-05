# Limitations

## Real price ingestion is blocked

The 2026-09-05 provider spike concluded `NO_GO` for the exact free-only,
unattended, roughly 500-stock US/Europe price requirement.

- Stooq's public CSV path is protected against automation in the tested
  environment.
- Its official automation and private-retention terms could not be
  machine-verified.
- No browser challenge or access control will be bypassed.
- Alpha Vantage's official free limit is too small for the requested universe.
- Twelve Data's free tier does not provide the required broad European equity
  coverage.

Consequently, the default application is a deterministic synthetic research
prototype. It cannot honestly claim live rankings or twice-daily live
predictions until the owner supplies an approved licensed source or changes
the scope.

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

# Source Capability Spike

> Release state: see [README](../README.md#release-status). Provider
> capability does not establish product release or account entitlement.

## Decision summary

- Broad unattended US/European OHLCV remains `NO_GO`.
- The only approved live-price direction is a **conditional private US-only
  Twelve Data path** using a non-demo key, a bounded universe, explicit
  personal/internal-display authorization, and no redistribution.
- Stooq remains `NO_GO`; StanStock does not bypass browser/automation
  controls.
- SEC EDGAR, filings.xbrl.org, and ECB have separate capabilities and
  limitations. None is a prerequisite for the active price-only research
  product.
- `synthetic_demo` remains the default offline source.

This document is an engineering capability record, not legal advice or a
license grant. The owner must recheck official terms for the account and use.

## Provider verdicts

| Source | Verdict | Product boundary |
|---|---|---|
| `synthetic_demo` | Built in | Offline, deterministic, visibly synthetic, research-grade only |
| Twelve Data US prices/catalogs | `CONDITIONAL_GO` | Private bounded US workflow after key/rights/owner gates |
| Stooq daily prices | `NO_GO` | Automation challenge and unverified retention rights; no bypass |
| SEC EDGAR | `GO` subject to runtime fair-access preflight | Separate archived-fundamental workflow |
| filings.xbrl.org | `CONDITIONAL_GO` | Bounded adapter/probe with documented jurisdiction/lag limits |
| ECB EXR | `GO` for the bounded SDMX adapter | Existing point-in-time FX capability, not active price input |
| European live equity prices | Deferred | No approved source |

## Probe behavior

`python manage.py source_spike` performs small representative requests, writes
a private report without response bodies or secrets, and updates provider
status. It does not enable a provider.

Runtime classifications (`ok`, missing configuration, environment blocked,
provider incompatible, quota exhausted, unexpected error) describe one probe
run. Static verdicts describe the reviewed capability boundary. One successful
request does not change a verdict or prove display rights.

## Twelve Data boundary

The reviewed path uses official catalog and daily time-series endpoints.
Authentication is sent in a header; keys do not belong in URLs, command
arguments, reports, metadata, logs, or the database.

Product scope:

- unchanged curated 100-name US core;
- at most 20 captured owner-saved names;
- SPY fetched/reused separately and excluded from stock membership;
- US/USD common stocks and supported ADRs only;
- exact catalog identity and entitled plan filtering;
- immutable raw JSON and normalized Parquet;
- seven-year request path when missing qualified history must be bootstrapped;
- 757 exact common closes required for analysis; and
- split-adjusted price returns excluding dividends.

Saved names are admission requests, not automatic provider requests or
analysis membership. Existing adequate registered history is reused before
credentials/quota. Missing candidates are bounded and recorded separately.

### Volume provenance limitation

Twelve Data's reviewed/documented `adjust=splits` behavior establishes the
price-return basis. The currently documented source does **not** establish
that reported share volume is adjusted on a compatible split basis.

StanStock therefore does not set volume compatibility merely because prices
are split-adjusted. Momentum, drawdown, relative volatility, and FHS
projections can be calculated from prices, while dollar turnover remains
unavailable and a positive direction stays HOLD rather than BUY.

This is not a claim that the provider volume is wrong. It is a refusal to
infer provenance the documentation does not prove.

## Rights and owner gate

Twelve Data's plan labels and support guidance distinguish private/internal
use and external display. StanStock supports:

- Basic only with explicit single-user personal,
  non-commercial/non-redistributed authorization and exactly one active
  licensed user;
- another plan/agreement only with explicit internal-display authorization
  covering the intended audience.

Current rights can change independently of code. Disable the provider and stop
serving/acquiring data when rights are absent or uncertain.

## Quota and recovery

The implementation paces and accounts for its own calls but cannot see credits
spent by another application. Recovery of an already completed target
precedes provider enablement, key resolution, and new quota. A full acquisition
attempt may need catalogs, each unresolved stock history, and SPY; exact spend
depends on reusable evidence and admitted candidates.

Never loosen identity/history checks or drop failed candidates from a fixed
acceptance denominator to fit quota.

## Other sources

### Stooq

The observed CSV route returned a browser-verification challenge and no
versioned official automation/retention contract was established. StanStock
does not solve the challenge, scrape HTML, or use an occasional successful
response to change the `NO_GO` verdict.

### SEC EDGAR

SEC is an official filing source subject to identifying User-Agent and
fair-access controls. It remains useful for frozen archived fundamental
methods, but active price research does not wait for SEC ingestion.

### filings.xbrl.org

The bounded adapter uses official-host links and records repository timing
conservatively. Documented jurisdiction gaps and ingestion lag remain.

### ECB EXR

The bounded SDMX adapter supports persisted point-in-time FX. ECB reference
rates are informational rather than executable transaction prices and do not
enter the active USD-only price operators.

## Re-verification and stop conditions

Re-run the bounded probe only when deliberately testing source access. Stop
the live path on:

- non-`ok` required probe behavior;
- demo/missing key;
- disabled provider;
- absent/expired rights;
- owner/display mismatch;
- quota refusal;
- catalog or security identity ambiguity;
- malformed/future source data;
- checksum/asset mismatch; or
- an operational deadline/revision failure.

A source probe success does not enable the provider, release the product,
register a study, or authorize publication.

# Source Capability Spike

## Summary

- **Retrieval date:** 2026-09-06 (findings were observed on or around this
  date from this project's execution environment).
- **Decision:** `CONDITIONAL_GO` for a reduced US-only universe through
  Twelve Data's documented API when a non-demo key and the selected plan's
  explicit personal-use or display entitlement are configured. The original
  free, unattended, roughly 500-ticker US/Europe requirement remains `NO_GO`.
- Twelve Data Basic provides the required US symbols and enough technical
  credits. Its pricing page labels Basic as **internal non-display**, while
  the provider's support article updated 2026-08-04 describes non-commercial
  internal tools as acceptable for Individual plans.
- StanStock therefore permits Basic only under an explicit single-user
  personal/non-commercial attestation, with no redistribution, no public
  access, one licensed active account, and fail-closed web/job guards.
  Grow/Pro/Ultra or a custom agreement still require explicit
  internal-display confirmation.
- Stooq remains fixed `NO_GO`: its public CSV download is
  automation-blocked and its automation/private-retention terms could not
  be independently verified. StanStock never bypasses its JavaScript gate.
- Fundamentals (SEC, filings.xbrl.org) and FX (ECB) are assessed
  separately below and never change the price decision above.
- This document only records observed technical behavior and completed
  research conclusions. It is not a substitute for each provider's own
  terms of use and does not grant, claim, or imply any license,
  redistribution, or commercial-use right. Consult the official pages
  cited below before any production use.

## Method

`python manage.py source_spike` sends one small, representative request to
each provider (Twelve Data and Stooq daily prices, SEC submissions,
filings.xbrl.org filings index, and ECB EXR CSV), writes a private JSON
report (0600 permissions, no response bodies or secrets) under
`<DATA_DIR>/reports/`, and updates
`ProviderRecord.status` / `.last_error` / `.last_success_at` for each
provider. It never touches `ProviderRecord.enabled` — enabling a provider
for real ingestion is a manual decision (see `LEARNINGS.md`), not something
a probe should do automatically.

The command reports two distinct things per provider, and they must not be
conflated:

1. A **runtime classification** of that run's probe — whether the request
   could be completed right now, from this environment. Classified by what
   a failure actually indicates, not generically by exception type:

   | Classification | Meaning |
   | --- | --- |
   | `ok` | The probe succeeded and returned usable data. |
   | `configuration_missing` | Required local configuration (for example `SEC_USER_AGENT`) is absent. Says nothing about the provider. |
   | `environment_blocked` | This execution environment could not complete the request as designed — a network/transport failure, or an access-denial response this project's own testing attributes to network/egress policy rather than the provider rejecting a compliant request. |
   | `provider_incompatible` | The provider itself returned something StanStock cannot use without violating its own rules (an HTML/JavaScript verification challenge, a clear rate/subscription message, or a malformed payload). |
   | `quota_exhausted` | An approved provider reported that its request or daily credit allowance was exhausted. |
   | `unexpected_error` | Anything else; always investigated before the report is trusted. |

2. A static **capability verdict** — the already-completed research
   conclusion (below) about whether StanStock is willing to rely on that
   provider at all, independent of any single run's technical outcome.
   This is one of `GO`, `CONDITIONAL_GO`, or `NO_GO`, and only changes when
   someone updates it after new research, together with this document — it
   is never inferred automatically from a probe result.

The overall price-capability decision (`CONDITIONAL_GO`/`NO_GO`) requires
**both** a price-capability provider's runtime probe to classify `ok`
**and** its capability verdict to be something other than `NO_GO`. Twelve
Data can satisfy that technical gate for the reduced US scope. The probe
never enables the provider: `configure_twelve_data` separately requires an
explicit display-rights confirmation.

## Capability verdicts (completed research)

| Provider | Verdict | Basis |
| --- | --- | --- |
| Twelve Data (US prices/reference data) | **CONDITIONAL_GO** | Official API and US coverage are usable; activation is conditional on account rights, quotas, private use, and no redistribution. |
| Stooq (price) | **NO_GO** | Automation-blocked; automation/private-retention terms unverifiable; JS gate will not be bypassed. |
| SEC EDGAR (fundamentals) | **GO** | Approved source in general per its documented fair-access policy, despite this execution's IP seeing HTTP 403. |
| filings.xbrl.org (ESEF fundamentals) | **CONDITIONAL_GO** | Usable, but with explicit, documented gaps (Germany and Ireland missing) and repository ingestion lag. |
| ECB Data Portal EXR (FX) | **GO** | Usable via the SDMX CSV/XML API (what StanStock's client uses); the legacy bulk history CSV showed anomalous rows and is deliberately not used. |

## Findings by provider

### Twelve Data (US prices) — verdict `CONDITIONAL_GO`

- Official endpoints:
  `https://api.twelvedata.com/time_series` and
  `https://api.twelvedata.com/stocks`; API documentation:
  <https://twelvedata.com/docs>.
- Authentication is sent in the `Authorization: apikey ...` header. Keys are
  never added to URLs, reports, asset metadata, logs, or the database.
- The technical probe verified AAPL daily OHLCV and NASDAQ/NYSE stock
  reference catalogs. Runtime catalog rows provide symbol, company name,
  currency, exchange, MIC, instrument type, FIGI when available, and plan
  access. A September 2026 live catalog included an unrelated row with a
  missing company name, so ingestion preserves the complete raw response but
  strictly normalizes only the configured symbols. A malformed configured
  symbol still fails the run.
- The committed `us_liquid_starter_v1.yaml` is a curated 100-symbol
  NASDAQ/NYSE common-stock set. SPY is fetched separately as its benchmark and
  the same immutable response maintains the investable SPY ETF market row.
  SPY is fetched once, consumes one credit, and never becomes a universe
  member or stock-analysis candidate. The configured universe is not an S&P
  500, Nasdaq-100, or other licensed-index reproduction.
- `/time_series` costs one API credit per symbol. The Basic quota profile is
  8 credits/minute and 800/day, reset at midnight UTC
  (<https://support.twelvedata.com/en/articles/5615854-credits>). One full
  configured run uses about 103 credits. StanStock paces and counts its own
  calls conservatively, but cannot observe credits used by another
  application sharing the account.
- US listed equities and historical end-of-day coverage are documented at
  <https://support.twelvedata.com/en/articles/9935903-us-equities-market-data>.
  Broader European coverage is not part of this approved starter scope.
- Every daily request sets `adjust=splits`. Results are split-adjusted price
  returns, not dividend-adjusted total returns. Live verification showed that
  a date-only `end_date` is exclusive, so the adapter requests the following
  calendar date while continuing to validate every returned bar against
  StanStock's original inclusive cutoff.
- Licensing is a separate gate from technical access. Twelve Data's current
  individual pricing page (<https://twelvedata.com/pricing>) labels Basic as
  internal non-display and Grow as including internal display. Its support
  article, [Commercial and personal usage](https://support.twelvedata.com/en/articles/5332349-commercial-and-personal-usage),
  updated 2026-08-04, says Individual plans are for personal/internal use and
  permits non-commercial internal tools while prohibiting redistribution and
  commercial display to third parties. Basic activation therefore requires
  `PERSONAL_SINGLE_USER_NONCOMMERCIAL_AUTHORIZED`, records the sole licensed
  user, and stops if a second active user exists. Display-entitled plans use
  `PERSONAL_INTERNAL_DISPLAY_AUTHORIZED`. The owner remains responsible for
  confirming that the account's current terms cover the exact use.
- The terms (<https://twelvedata.com/terms>) permit access, processing, and
  storage only within the applicable subscription rights, prohibit
  unauthorized redistribution/external display, and require deletion of Data
  after termination or expiration. The owner must disable the provider and
  remove its stored data when those rights end. Because stored assets are
  linked into immutable research provenance, StanStock's supported procedure
  is the full database/data/backup destruction process in
  `docs/operations.md`, not a partial manifest or file deletion.

### Stooq (daily prices) — verdict `NO_GO`, runtime `provider_incompatible`

- Endpoint probed: `https://stooq.com/q/d/l/?s=<symbol>&i=d`.
- Stooq (<https://stooq.com>) publishes no versioned public API and no
  discoverable terms page describing automated or bulk-retrieval rights;
  its automation and private-retention terms could not be verified.
- **Observed 2026-09-05:** an unauthenticated GET to that endpoint, with an
  identifying `User-Agent` and no special headers, returned an HTML page
  containing a JavaScript-based browser-verification challenge instead of
  the expected CSV body.
- StanStock treats this as a hard, permanent stop for this path: it never
  attempts to solve or bypass a bot/browser challenge, and never falls
  back to scraping the HTML quote page as a substitute for the CSV
  download.
- The runtime classification for this observation is
  `provider_incompatible` (a deliberate anti-automation control, not a
  local network problem), but the **verdict is fixed at `NO_GO`**
  independent of runtime classification: even on a run where the challenge
  does not appear and the probe returns `ok`, that alone does not
  constitute verified automation/retention rights, so the verdict does not
  become `GO` or `CONDITIONAL_GO` automatically. Only new research that
  establishes verified terms can change this constant.

### SEC EDGAR — verdict `GO`, runtime `ok`

- Endpoints probed: `https://data.sec.gov/submissions/CIK##########.json`
  and `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`.
- Official references: SEC EDGAR API documentation
  (<https://www.sec.gov/edgar/sec-api-documentation>) and SEC's fair
  access policy, which requires every automated client to declare an
  identifying `User-Agent` with a contact method
  (<https://www.sec.gov/os/webmaster-faq#developers>).
- **Observed 2026-09-05:** requests sent with a compliant `SEC_USER_AGENT`
  (an identifying string plus a contact email, as SEC's policy requires)
  returned HTTP 403 from this environment.
- Runtime classification is `environment_blocked`, **not**
  `provider_incompatible`: SEC's own documented policy is to serve
  automated requests that identify themselves correctly, and 403 despite a
  compliant identifier is more consistent with this network's egress being
  blocked or filtered than with SEC rejecting a well-formed request.
- **Observed 2026-09-07 from the actual local runtime:** the bounded
  submissions probe returned HTTP 200 with the configured identifying
  `SEC_USER_AGENT`. The official ticker/exchange/CIK mapping, submissions,
  referenced historical submissions files, and Companyfacts then completed
  successfully for all 100 configured stocks.
- The **verdict is `GO`**: SEC access is treated as approved in general —
  this execution environment's 403 is a local/network condition to
  re-verify from a different, unblocked network or deployment, not a
  reason to mark SEC itself unusable.
- StanStock preserves raw mapping, submissions history, and Companyfacts
  payloads and normalizes only the reviewed concept allowlist. Facts retain
  taxonomy, source concept, unit, full instant/duration period identity,
  accession, form, filing date, exact acceptance time when present,
  conservative date-only fallback, immutable source revision, and source
  asset. A separate immutable evidence link retains the submissions/history
  asset that supplied the filing boundary. Current SIC is stored as a
  retrieval-time classification snapshot and is never silently backdated.
- Note on rights: SEC filings are U.S. government work and not subject to
  copyright, but access is still governed by SEC's fair-access rate
  limits. The local implementation stays below the published ceiling,
  coordinates requests, and avoids full-history downloads on unchanged daily
  runs.

### filings.xbrl.org — verdict `CONDITIONAL_GO`, runtime `ok`

- Endpoint probed: `https://filings.xbrl.org/api/filings`; API docs:
  <https://filings.xbrl.org/docs/api>.
- **Observed 2026-09-05:** requests succeeded (HTTP 200) and returned the
  documented JSON:API-shaped filings index from this environment.
- The verdict is `CONDITIONAL_GO`, not a plain `GO`, because of two
  explicit, documented limitations: the filings.xbrl.org index itself
  states its country coverage is incomplete, **explicitly naming Germany
  and Ireland as missing** at the time of writing; and its
  `date_added`/`processed` fields describe when the repository
  ingested/indexed a filing, which can lag the original filing authority's
  own timestamp (repository ingestion lag). StanStock uses
  `date_added`/`processed` only as a conservative availability floor (see
  `docs/point-in-time.md`), never as a stand-in for the authority's own
  filing time, and does not claim coverage for jurisdictions the index
  itself says it lacks.
- The adapter resolves root-relative xBRL-JSON links only against the official
  `filings.xbrl.org` origin and rejects unexpected hosts. The index's package
  checksum is retained as package metadata; downloaded JSON bytes receive
  their own content checksum rather than being compared with a different
  package artifact.

### ECB Data Portal EXR (foreign exchange) — verdict `GO`, runtime `ok`

- Endpoint probed:
  `https://data-api.ecb.europa.eu/service/data/EXR/{key}?format=csvdata`;
  API examples: <https://data.ecb.europa.eu/help/api/data-examples>.
- **Observed 2026-09-05:** requests succeeded (HTTP 200, CSV body) from
  this environment; no API key or account is required.
- This is the SDMX 2.1 RESTful API's CSV representation, which StanStock
  prefers and exclusively uses. Research separately found the **legacy
  bulk history CSV** (ECB's older, non-SDMX historical download) shows
  anomalous rows; StanStock's ECB client does not use that endpoint at
  all, so this anomaly does not affect it. The SDMX API also supports a
  daily XML representation as a documented alternative, not currently
  implemented here.
- Caveats: ECB reference rates are informational, generally observed
  around 14:15 CET and published around 16:00 CET; the CSV rows carry
  only the observation date, with no per-row publication timestamp, so
  StanStock models `published_at`/`available_at` from that documented
  publication convention rather than a value the API returns per
  observation (see `docs/limitations.md`).

## Decision logic

The overall price-capability decision is `CONDITIONAL_GO` only if a
price-capability provider's probe classifies `ok` **and** that provider's
capability verdict is not `NO_GO`. A successful Twelve Data probe can meet
that condition for the reduced US-only scope. It does not prove display
rights and does not change `ProviderRecord.enabled`; activation remains a
separate explicit command. Stooq cannot meet the condition because its
capability verdict remains `NO_GO`.

Without a configured Twelve Data key, or when the provider is disabled,
StanStock continues to use deterministic synthetic data (see `seed_demo`)
rather than silently degrading scope or mislabeling reconstructed data as
live.

## Re-verification

Run `python manage.py source_spike` to refresh the runtime classifications
above. Twelve Data reports `configuration_missing` until
`TWELVE_DATA_API_KEY` is present and `quota_exhausted` when the provider
rejects the request for credit limits. SEC reports `configuration_missing`
until `SEC_USER_AGENT` is exported and may still report
`environment_blocked` on a network or deployment that filters SEC traffic.
Stooq's runtime classification may occasionally show `ok`, but its `NO_GO`
verdict remains fixed pending verified automation/retention rights.

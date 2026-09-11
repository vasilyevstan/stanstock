# Architecture

StanStock is one Django modular monolith. The boundaries exist to protect
research integrity, not to imitate a distributed system.

## Modules

| Module | Responsibility |
|---|---|
| `core` | Owner authentication support, health, target-date job runs, backup and restore |
| `data` | Permanent identities, universe snapshots, providers, immutable assets, filings, FX, and as-of reads |
| `research` | Indicators, transparent scores, risk, scenarios, analyses, predictions, and outcomes |
| `simulation` | One accounting model shared by backtests and portfolio simulations |
| `portfolio` | Owner-scoped holdings, immutable deposits/purchases/performance baselines, allocation planning, and dated valuations |
| `web` | Authenticated server-rendered pages, filters, status, and lightweight JSON where needed |

Raw provider clients belong under `stanstock.data.providers`. An architecture
test rejects imports of those modules from `research` and `simulation`.
Quantitative code consumes normalized records or the `AsOfData` boundary.

## Storage

PostgreSQL is the canonical relational store. Direct development can use
SQLite. Relational rows hold:

- permanent company, security, and listing identities;
- dated universe snapshots and memberships;
- source and provider status;
- data-asset manifests and normalized filing/FX facts;
- analyses, immutable predictions, outcomes, jobs, simulations, tracked
  portfolios, immutable cash/purchase ledgers and performance baselines, and
  immutable portfolio valuations.

Large or source-native payloads live under `STANSTOCK_DATA_DIR`. `DataAsset`
stores a relative path, SHA-256 checksum, retrieval time, availability time,
period, schema version, and metadata. Price/research panels use Parquet. A
same-path/same-checksum write is idempotent; a same-path/different-checksum
write fails instead of replacing evidence.

The database and assets are one recovery unit. The `backup` command bundles
both and the `restore` command verifies every size and checksum before
replacement.

## Point-in-time flow

1. A provider response is written to an immutable path.
2. A `DataAsset` manifest records when the source was retrieved and when its
   contents became knowable.
3. Normalization creates additive filing or FX vintages linked to the source
   asset.
4. For an observed decision, `AsOfData(generated_at)` permits only assets
   retrieved by generation time and facts available by the same decision
   boundary. For a research-grade historical reconstruction, the asset may
   have been retrieved later, but fact availability and price rows remain
   capped at the logical `data_cutoff`.
5. `AsOfData.price_frame` normalizes and sorts the price date column and
   physically removes rows after the requested market date.
6. An `AnalysisRun` records `generated_at`, `data_cutoff`, the universe
   snapshot, configuration hash, and code revision.
7. For the US price-only configuration, the run builds one immutable derived
   Parquet panel from the exact stock and SPY price assets available to that
   run. Historical labels end on or before the forecast target, fixed-epoch
   cohorts do not overlap within a horizon, and the asset records source,
   calendar, configuration, content, and code hashes.
8. `StockAnalysis.forecast_scenarios` is the schema-versioned scenario read
   path. The legacy short/medium/long columns remain temporarily for rollback
   compatibility, but application policy reads through the unified accessor.
9. `Prediction` stores the complete issued result and provenance, including
   its forecast identity, decision/advisory role, evidence grade, source mode,
   exact price provider and source subject when proven, and a structured
   calculation record. Performance reporting reads these immutable prediction
   fields rather than mutable parent metadata. Django guards and database
   triggers reject updates and deletes. The same application/database
   immutability rule protects `DataAsset`, `FundamentalFact`, and `FxRate`.
10. Outcomes append after the horizon matures; matured and corporate-event
   states are terminal and do not alter the original prediction. Decision
   outcomes retain recommendation success, while advisory outcomes use
   base-case sign match, bear-to-bull inclusion, and signed base-case error
   with `success=NULL`.
11. A simulation stores its base currency, result curve, and exact price,
   signal, benchmark, and FX frames as checksummed assets keyed by the run
   UUID. Converted price rows keep their native price and currency beside the
   converted value. Its input hash covers complete canonical frame contents
   and any explicit calendar.
12. A tracked portfolio snapshot stores the exact quantity, average cost,
    latest persisted price, price session, source asset, code revision, and
    aggregate value used. Repeated identical inputs are idempotent; changed
    holdings can create another snapshot on the same market date. Database
    triggers reject snapshot and snapshot-position updates or deletes.
13. External deposits, confirmed planner executions/purchases, and manual
    performance baselines are append-only. Confirmation re-hashes locked
    portfolio, price, and qualifying-analysis state. A manual quantity change
    creates a post-change boundary (or an explicit unavailable-boundary
    record), so contribution return is never inferred from unexplained state.

## Runtime

The same image runs:

- Gunicorn for the Django website;
- deterministic Django management commands for source probes, demo data,
  conditional US Twelve Data ingestion, analysis, predictions, evaluation,
  simulations, tracked-portfolio snapshots, backup, and restore.

Target-date work uses `JobRun` plus a PostgreSQL advisory lock. A unique
constraint permits only one successful run for a job/region/date. Repeating a
successful target creates a visible skipped attempt instead of repeating work.
The US workflow additionally coordinates a conservative provider credit budget
through a locked `ProviderRecord` and persists raw and normalized vintages
before analysis. Basic mode is bound to exactly one active licensed user with
an explicit personal/non-commercial attestation; other plans require an
explicit internal-display entitlement.

No Redis, Celery, resident scheduler, second analytics engine, SPA, or fitted
ML model is part of v1.

## Security boundary

Every data-bearing page requires Django authentication. There is no public
signup. Login attempts are bounded in the application process, state-changing
forms use CSRF protection, production settings default to secure cookies and
HTTPS, and the production container runs as an unprivileged user with a
read-only root filesystem. Forwarded client IPs affect login throttling only
when the immediate proxy address is explicitly trusted.

When Twelve Data Basic is enabled, middleware fails closed for every
authenticated user except the licensed owner recorded during activation.
Provider jobs also stop if more than one active user exists. The provider key
is resolved from the process environment or, for direct macOS use, the current
OS user's login Keychain; it is never stored in application tables.

The intended deployment is one private instance. A multi-replica deployment
would require a shared rate-limit store and explicit job coordination review.

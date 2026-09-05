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
- analyses, immutable predictions, outcomes, jobs, and simulations.

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
7. `Prediction` stores the complete issued result and provenance. Django
   guards and database triggers reject updates and deletes. The same
   application/database immutability rule protects `DataAsset`,
   `FundamentalFact`, and `FxRate`.
8. Outcomes append after the horizon matures; matured and corporate-event
   states are terminal and do not alter the original prediction.
9. A simulation stores its base currency, result curve, and exact price,
   signal, benchmark, and FX frames as checksummed assets keyed by the run
   UUID. Converted price rows keep their native price and currency beside the
   converted value. Its input hash covers complete canonical frame contents
   and any explicit calendar.

## Runtime

The same image runs:

- Gunicorn for the Django website;
- deterministic Django management commands for source probes, demo data,
  analysis, predictions, evaluation, simulations, backup, and restore.

Target-date work uses `JobRun` plus a PostgreSQL advisory lock. A unique
constraint permits only one successful run for a job/region/date. Repeating a
successful target creates a visible skipped attempt instead of repeating work.

No Redis, Celery, resident scheduler, second analytics engine, SPA, or fitted
ML model is part of v1.

## Security boundary

Every data-bearing page requires Django authentication. There is no public
signup. Login attempts are bounded in the application process, state-changing
forms use CSRF protection, production settings default to secure cookies and
HTTPS, and the production container runs as an unprivileged user with a
read-only root filesystem. Forwarded client IPs affect login throttling only
when the immediate proxy address is explicitly trusted.

The intended deployment is one private instance. A multi-replica deployment
would require a shared rate-limit store and explicit job coordination review.

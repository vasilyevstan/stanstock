# StanStock

StanStock is a private, local-first stock-research application for transparent
US and European equity scoring, scenario analysis, immutable prediction
tracking, backtesting, and portfolio simulation.

It is a rules-based research system, not an automated trading service. An LLM
cannot change scores or recommendations, and forecasts are never presented as
guarantees.

## Current data boundary

The required free-provider capability gate is **NO_GO for unattended real
price ingestion at the requested breadth**.

- Stooq's public download route is protected against unattended automation,
  and its automation/private-retention rights could not be verified.
- SEC EDGAR and ECB data are viable official sources.
- filings.xbrl.org is usable for European filings but documents incomplete
  coverage, including Germany and Ireland.
- The official free API alternatives reviewed do not support roughly 500 US
  and European equities with twice-daily automated OHLCV updates.

StanStock therefore defaults to deterministic synthetic data. It does not
scrape around access controls, silently shrink the universe, or label
reconstructed data as live predictions. See `docs/source-spike.md` for the
evidence and exact limitations.

### Synthetic demo data

`seed_demo` and `refresh_demo` never call, approximate, or claim to call any
live provider (Stooq/SEC/filings.xbrl.org/ECB/etc.); every row they create is
entirely synthetic, obviously-fake demo data (see the command docstrings for
the full guarantees).

Seed ~60 synthetic US/European listings, one research-grade synthetic
universe snapshot, and 6+ years of synthetic OHLCV/fundamentals/FX history
(idempotent: safe to rerun; never deletes or mutates already-seeded rows):

```bash
uv run python manage.py seed_demo
```

Run the bounded, **SYNTHETIC-ONLY** vertical flow -- `seed_demo` followed by
`analyze_snapshot` against the synthetic universe snapshot
(`provider=synthetic_demo`, `benchmark=ZZBENCH01`) -- with only the analysis
step wrapped in the same target-job idempotency guard as a real scheduled
job, so a `--target-date` that already succeeded is skipped rather than
re-analyzed. `--target-date` defaults to the synthetic snapshot's own
`as_of_date` (its synthetic history does not extend past that date); an
explicit `--target-date` is rejected unless it is on or before that date
*and* an actually-observed session in the synthetic benchmark's price
history (no fabricated weekends/holidays):

```bash
uv run python manage.py refresh_demo
```

## Start locally

### Docker Compose

Requirements: Docker with Compose.

```bash
docker compose up --build
```

Open <http://localhost:8000> and sign in with the development-only defaults:

```text
username: admin
password: stanstock-dev
```

Override both values with `STANSTOCK_OWNER_USERNAME` and
`STANSTOCK_OWNER_PASSWORD`.

### Direct Python development

Requirements: Python 3.13 and `uv`.

```bash
uv sync --all-groups
uv run python manage.py migrate
STANSTOCK_OWNER_PASSWORD=stanstock-dev uv run python manage.py bootstrap_owner
uv run python manage.py refresh_demo
uv run python manage.py runserver
```

The direct development path uses SQLite unless `DATABASE_URL` is set. Docker
Compose uses PostgreSQL and durable named volumes for the database and
`STANSTOCK_DATA_DIR`. Development Compose runs the idempotent synthetic demo
refresh automatically when `STANSTOCK_DEMO_MODE=true`.

Evaluate pending predictions through an explicit observed-data cutoff:

```bash
uv run python manage.py evaluate --all-pending \
  --evaluation-date 2026-09-04 \
  --benchmark-subject ZZBENCH01
```

Run a portfolio simulation from the authenticated `/simulations` page or with
the `simulate` command. CLI portfolio selections use permanent listing UUIDs;
`python manage.py simulate --help` documents the complete arguments. A
selection spanning several native currencies is converted into one explicit
reporting currency (`--base-currency USD`, `EUR`, or `GBP`) using rates dated
on or before each simulated date, resolved against that date's own end-of-day
cutoff so a later correction cannot rewrite an earlier execution; a missing,
over-stale, or ambiguous rate path fails the run rather than converting part
of it. Use `--restrict-native-currency` to run a single-currency slice of a
mixed universe instead.

## Authenticated pages

- `/opportunities` - ranked, filterable analyses from the latest completed run.
- `/stocks/<listing-id>` - scenarios, factor evidence, risks, and provenance.
- `/predictions` - the append-only prediction ledger.
- `/performance` - matured outcomes with minimum-sample safeguards.
- `/simulations` - backtest and portfolio runs through one accounting model.
- `/status` - database, asset-store, provider, job, and prediction status.
- `/methodology` - point-in-time, scoring, scenario, and limitation summary.

When persisted analysis does not exist, the status page clearly labels its
illustrative synthetic rows. Data-bearing pages require authentication;
`/healthz` exposes only coarse readiness information. The authenticated
synthetic-data banner is derived from persisted source provenance, so disabling
demo mode cannot make synthetic analyses appear live.

## Integrity guarantees

- Permanent company, security, and listing IDs; ticker text is not identity.
- Immutable source assets and explicit `retrieved_at`/`available_at` vintages.
- Historical reads use the `AsOfData` boundary, including physical row-level
  clipping and ascending ordering through the requested market date.
- Historical research reconstructions cap fact availability and price rows at
  the logical target while retaining their actual later generation/retrieval
  timestamps; only on-time observed runs count as live evidence.
- Database constraints bound scores, confidence, probability, dates, and
  scenario ordering.
- Prediction, asset-manifest, filing-fact, and FX-vintage updates/deletes are
  rejected by Django and database triggers.
- Target-date jobs are serialized and a successful target cannot execute
  twice.
- Missing or statistically insufficient evidence remains explicit.
- Missing risk inputs produce an `INSUFFICIENT EVIDENCE` state rather than a
  fabricated numeric risk score.
- Simulation results and exact price/signal/benchmark inputs are persisted as
  checksummed immutable assets, and the run input hash covers their complete
  normalized contents.

## Backup and recovery

Create one checksummed bundle containing the database snapshot and every file
under `STANSTOCK_DATA_DIR`:

```bash
uv run python manage.py backup
```

Verify a bundle without changing state:

```bash
uv run python manage.py restore var/backups/<bundle>.tar.gz --verify-only
```

Restore is destructive and requires explicit confirmation:

```bash
uv run python manage.py restore var/backups/<bundle>.tar.gz --confirm RESTORE
```

Stop all web and job processes before a restore, then restart them afterward.
PostgreSQL backup/restore requires `pg_dump` and `pg_restore` on `PATH`; the
provided production image includes the PostgreSQL client tools. PostgreSQL
restore is fail-fast and single-transaction so a failed restore rolls back its
database changes.

## Production configuration

`compose.production.yaml` is a vendor-neutral single-instance template. It
requires:

- an external PostgreSQL `DATABASE_URL`;
- a strong `DJANGO_SECRET_KEY`;
- explicit allowed hosts and trusted origins;
- TLS at the application or trusted reverse proxy;
- durable private volumes for `/app/var/data`, `/app/var/backups`, and
  `/app/var/static`.

Production settings default to secure cookies, HTTPS redirect, HSTS, and a
non-root/read-only container. Set insecure cookie/redirect options only for a
local HTTP dry run. Set `STANSTOCK_LOGIN_TRUSTED_PROXY_IPS` to the
comma-separated source IPs of proxies that overwrite `X-Forwarded-For`;
unlisted peers cannot influence the login-throttle client address.

## Development checks

```bash
make check
uv run python manage.py makemigrations --check --dry-run
uv run python manage.py check --deploy --settings=stanstock.settings.prod
```

Tests and CI use synthetic fixtures and never call live providers.

## Documentation

- [Architecture](docs/architecture.md)
- [Point-in-time integrity](docs/point-in-time.md)
- [Methodology](docs/methodology.md)
- [Source capability spike](docs/source-spike.md)
- [Operations and recovery](docs/operations.md)
- [Deployment](docs/deployment.md)
- [Testing](docs/testing.md)
- [Known limitations](docs/limitations.md)
- [Durable implementation learnings](LEARNINGS.md)

## Financial disclaimer

Forecasts are estimates, not guarantees. Historical and simulated performance
does not guarantee future performance. StanStock does not provide personalized
financial, investment, tax, or legal advice.

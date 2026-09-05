# Operations

## Local services

Development Compose starts PostgreSQL and the Django web process. The entry
point applies migrations, creates `STANSTOCK_DATA_DIR`, collects static files,
and bootstraps the owner only when `STANSTOCK_OWNER_PASSWORD` is present.

The public `/healthz` endpoint checks database connectivity and asset-directory
writability without revealing credentials or filesystem paths. Detailed
provider, job, prediction, and outcome state is shown only on authenticated
pages.

## Target-date jobs

Jobs use the tuple `(job_name, region, target_date, attempt)`. PostgreSQL
advisory locks serialize work for one logical target. The database permits only
one `success` row for a job/region/date.

Terminal states are:

- `success`: work completed and must not execute again for that target;
- `no_data`: the source had no usable data and a later retry may be valid;
- `skipped`: the target was already complete or intentionally ineligible;
- `failed`: an explicit error was recorded;
- `running`: an active attempt; a stale row is marked failed by the next lock
  holder before retrying.

The actual generation timestamp is always preserved. Catch-up work must not
pretend a missed prediction was issued on its historical target date.

The current free-only boundary has no lawful live-price job. For local product
testing, the synthetic-only equivalent is:

```bash
uv run python manage.py refresh_demo
```

It seeds deterministic data, validates the target against an observed
synthetic benchmark session, and runs one idempotent analysis/prediction batch.
It never contacts a live provider.

Pending outcomes can be evaluated through an explicit cutoff:

```bash
uv run python manage.py evaluate --all-pending \
  --evaluation-date 2026-09-04 \
  --benchmark-subject ZZBENCH01
```

Backtests and portfolio simulations run through `python manage.py simulate`
or the authenticated `/simulations` form. The simulation service persists the
exact input frames and result curve before reporting a complete run. A
selection spanning several native currencies must name its reporting currency
with `--base-currency USD|EUR|GBP`; a single-currency selection infers it.
Conversion resolves each simulated date against its own end-of-day cutoff, so
a later correction cannot rewrite an earlier execution, and carries the last
observation across market closures for at most `--fx-max-carry-days`
(0 to 7, default 7). Coverage is proven for every simulated date before
accounting starts, so a date the FX series cannot reach fails the run. A
missing, over-stale, or ambiguous rate path fails the run explicitly rather
than converting part of the panel. A converted run executes on closing prices;
opening-price execution bases are rejected because FX availability is only
resolved to end-of-day. `--benchmark-currency` is required alongside
`--benchmark-subject` whenever the run converts, and is rejected when given
without a subject. `--restrict-native-currency` runs a single-currency slice
of a mixed universe instead of converting it, and is rejected when it would
exclude a listing named in `--listings`.

## Logs

Django emits structured single-line JSON containing timestamp, level, logger,
and message. Provider and job code must avoid logging secrets, response bodies
containing private data, or environment values.

## Backup

```bash
uv run python manage.py backup
```

The command snapshots SQLite with its online backup API or PostgreSQL with
`pg_dump`, copies immutable assets, records SHA-256 and size for every member,
then atomically publishes one gzip-compressed tar bundle.

The bundle path must be outside `STANSTOCK_DATA_DIR` so it cannot recursively
include itself. The default directory is `var/backups` and can be changed with
`STANSTOCK_BACKUP_DIR`. The production Compose template mounts
`/app/var/backups` as a separate writable private volume because the container
root filesystem is read-only.

Verify registered files independently at any time:

```bash
uv run python manage.py verify_assets
```

The command fails when a manifest points to a missing file or when the current
bytes do not match the recorded SHA-256.

## Restore

Stop web and job processes first.

```bash
uv run python manage.py restore <bundle> --verify-only
uv run python manage.py restore <bundle> --confirm RESTORE
```

Restore rejects path traversal, symbolic-link members, unlisted members,
missing members, size mismatches, checksum mismatches, and database-format
mismatches. It validates asset destinations and available space before
changing the live database. PostgreSQL restore uses fail-fast,
single-transaction `pg_restore` semantics, so a database error rolls back the
restore. Asset restoration is additive; unreferenced pre-existing files are
not silently deleted.

Restart every process after the restore. PostgreSQL recovery requires an
account permitted to replace schema objects; the production image includes
`pg_dump` and `pg_restore`.

## Provider failures

The source gate currently rejects unattended real OHLCV at the required scope.
Provider clients must fail explicitly on browser-verification HTML, access
denial, malformed responses, missing credentials, and disabled
`ProviderRecord` state. No job may scrape HTML or turn a provider failure into
a successful empty market update.

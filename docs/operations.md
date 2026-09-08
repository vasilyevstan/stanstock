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

The default remains synthetic and requires no provider account:

```bash
uv run python manage.py refresh_demo
```

It seeds deterministic data, validates the target against an observed
synthetic benchmark session, and runs one idempotent analysis/prediction batch.
It never contacts a live provider.

The optional US-only Twelve Data workflow is disabled by default. On macOS,
store its key through the interactive login-Keychain command:

```bash
uv run python manage.py store_twelve_data_key
uv run python manage.py store_twelve_data_key --status
```

The `security` utility prompts for the value without placing it in shell
history or process arguments. StanStock reads the value into memory only when
an API call needs it. Use `TWELVE_DATA_API_KEY` from a deployment secret
manager on non-macOS systems; it takes precedence over Keychain.

For a private local checkout, Docker Compose can read
`TWELVE_DATA_API_KEY` from the ignored `.env` file. Copy `.env.example`, set
the key locally, and restrict the file to the current OS user with
`chmod 600 .env`. Both `.gitignore` and `.dockerignore` exclude the file.
Direct `manage.py` commands do not load dotenv files automatically; run
`set -a; . ./.env; set +a` in the shell first. Never commit, copy into an
image, print, or attach the populated file.

Run the bounded source probe, then explicitly record the Basic personal-use
scope:

```bash
uv run python manage.py source_spike
uv run python manage.py configure_twelve_data \
  --enable \
  --plan basic \
  --confirm PERSONAL_SINGLE_USER_NONCOMMERCIAL_AUTHORIZED
uv run python manage.py daily --region us
```

Basic activation requires exactly one active StanStock user and binds access
to that user's database ID. Any other authenticated user receives HTTP 403,
and provider jobs fail closed if another active account exists. Grow, Pro,
Ultra, and reviewed custom agreements use
`PERSONAL_INTERNAL_DISPLAY_AUTHORIZED` instead.

**Maintainer warning, not a routine command.** `manage.py analyze` is
demo/research tooling, not the live US operational interface and not an
observed-reissue path: it defaults to the generic `synthetic_demo` provider,
the default scoring config, and `code_revision()`'s `"working-tree"` fallback
when `STANSTOCK_CODE_REVISION` is unset. The command explicitly passes
`issued_on_time=False` for every target, including a same-day `OBSERVED`
snapshot, and labels its result research-grade. No combination of `--config`,
`--provider`, `--benchmark-subject`, or an exported
`STANSTOCK_CODE_REVISION` changes that contract. `scheduled_refresh` is the
supported unattended live path and binds the exact committed Git revision
automatically. An exceptional same-target observed reissue that is still
before its deadline is possible only through a direct
`stanstock.research.service.analyze_snapshot(..., issued_on_time=True, ...)`
Python invocation against an already-`OBSERVED` universe snapshot, and only
after the caller has independently reconfirmed the next-market-session-open
deadline for that target (e.g. via `stanstock.research.timing.
is_observed_issuance_on_time`) -- the service itself raises rather than
silently downgrading if the deadline or cutoff-safety proof fails. A later
non-observed reconstruction is research-grade; it is not a recovered observed
call. The exceptional invocation must run from a clean, already-committed
revision; explicitly bind the reviewed production scoring config
(`default_us_scoring_config_path()` or an equivalent explicit path),
`provider="twelve_data"`, and the reviewed benchmark (currently SPY); set
`STANSTOCK_CODE_REVISION` to the exact committed revision; and reuse
already-persisted assets rather than refetching from the live provider. Do not
copy a demo command into a live context or assume a reissue is on time because
an earlier version was.

### SEC EDGAR fundamentals

SEC EDGAR is unauthenticated and requires no API key. Put only an identifying
application string and monitored contact address in the ignored, mode-0600
`.env`:

```bash
SEC_USER_AGENT="StanStockResearch/0.1 monitored-address@example.com"
set -a
. ./.env
set +a
uv run python manage.py source_spike \
  --skip twelve_data,stooq,filings_xbrl_org,ecb
uv run python manage.py configure_sec --enable
uv run python manage.py fetch_sec_mapping
uv run python manage.py sync_sec_fundamentals --target-date YYYY-MM-DD
```

The preflight must return `sec: ok` before activation. The contact address is
never stored in Git, the database, a `JobRun`, an asset manifest, or logs; it
is sent only in SEC request headers. The initial sync preserves the reviewed
ticker/exchange/CIK mapping, current submissions, every referenced historical
submissions file, Companyfacts, and current SIC metadata. Repeated identical
payloads reuse their immutable content asset.

Steady-state automation polls submissions once per company. Companyfacts is
requested only after changed submissions, when a relevant accession has not
yet appeared in Companyfacts, or during staggered reconciliation. A newly
missing accession is checked at most once per calendar day for seven days;
after that bounded lag window, a fact-less amendment cannot force permanent
daily downloads and the normal 30-day staggered reconciliation remains the
backstop. Historical submission files are requested only when first discovered
or during reconciliation. Successful normalization is checkpointed against
the Companyfacts content, filing-source set, configuration, and normalizer
version, so a retry replays cached raw evidence after an interruption instead
of reporting a false no-op. SEC's configured request rate is five per second,
below the documented ten-per-second ceiling, and is coordinated through locked
provider metadata.

The sign-in and sign-out routes remain available if a restored database no
longer matches the stored licensed user ID. Sign out, ensure exactly one active
StanStock account remains, then rerun the Basic activation command above from
the local shell to validate the key and bind the provider record to that
account's current database ID.

The committed 100-stock configuration plus SPY uses approximately 103 credits
per full run. Local coordination enforces a conservative 8-credit/minute,
800-credit/day budget through locked `ProviderRecord.metadata`; another
application using the same account is outside that accounting. Schedule the
job after the provider has published the completed US daily bars. If the
benchmark has no target-date close, the run fails rather than creating a
partial snapshot. Repeating a successful target produces a skipped `JobRun`
and makes no market-data requests.

SPY is fetched exactly once in that budget. The same immutable benchmark
price asset advances the separate investable SPY ETF market row; SPY remains
outside the stock universe and analysis pipeline. When upgrading an
installation that already has SPY benchmark assets, hydrate the ETF identity
without network access or quota use:

```bash
uv run python manage.py migrate
uv run python manage.py sync_investable_etfs
```

The sync command selects the latest persisted Twelve Data SPY price asset,
validates its ETF/USD/listing-identity metadata, and idempotently creates or
advances the SPY listing. A provider-supplied MIC must be ARCX; when the
optional field is absent, the normalized asset records an explicit
`configured_spy_identity` ARCX resolution. It fails rather than guessing when
the asset is absent or incompatible. Recovery of a specific completed
analysis run follows the single benchmark asset UUID and checksum recorded by
every analysis in that run; it never substitutes a newer same-date vintage.

ETF materialization runs only after the stock snapshot, analyses, and
predictions commit. If an ETF identity conflict fails that final step, the
market job is failed visibly but a retry recovers the completed research and
retries ETF synchronization without provider credentials or additional
credits.

### Daily macOS LaunchAgent

The supported unattended local workflow is one user LaunchAgent at 02:00
local time Tuesday-Saturday:

```bash
.venv/bin/python manage.py launchd_refresh install
.venv/bin/python manage.py launchd_refresh status
```

Installation validates the detected IANA timezone over more than a year of
scheduled invocations, including regular and early XNYS closes plus local and
New York DST changes. Every checked invocation must satisfy:

```text
XNYS close + provider publication delay <= 02:00 local < next XNYS open
```

An unsafe timezone blocks installation; the installer never silently changes
the chosen hour. The timezone is recorded in the plist, and runtime refuses to
continue after a machine-timezone change until the LaunchAgent is reinstalled.

The plist contains only paths and non-secret runtime flags. The committed
`scripts/run-scheduled-refresh.sh` runner requires the ignored local `.env`,
rejects group/other-readable permissions, exports it inside the child process,
disables unattended Keychain fallback, and then executes the absolute
`.venv/bin/python`. A missing key therefore fails promptly instead of opening
or waiting on a Keychain prompt. Logs are written below the current user's
private `~/Library/Logs/StanStock` directory.

`scheduled_refresh` resolves the latest completed XNYS target and maintains an
aggregate parent `JobRun` with independently recoverable children:

1. enabled SEC submissions/facts ingestion;
2. `daily` market retrieval, universe snapshot, analysis, and prediction;
3. provider- and maturity-filtered prediction outcome evaluation;
4. immutable portfolio snapshots bound to the resolved XNYS session date.

The automated market child requires a clean Git worktree and records the exact
40-character HEAD revision. A retry recovers any successful child before
provider configuration, credentials, or quota are used again. A failed enabled
SEC child blocks market analysis so stale or absent facts cannot look current.
When Twelve Data and SEC are both enabled under the released US scoring
version, the market child issues separate 3y/5y advisory predictions after all
eligible stock computations are built against one shared point-in-time peer
context. Missing long inputs create explicit insufficient-evidence predictions;
they do not fail the market refresh or alter BUY/HOLD/AVOID.
Evaluation and portfolio
snapshots are attempted independently after market success, so one downstream
failure does not hide the other's result. Holidays and already completed
targets become explicit skips. If macOS wakes the job after the next XNYS
session has opened, a missing market child fails rather than creating a late
prediction marked as observed; use an explicit manual `daily --target-date`
research reconstruction when historical catch-up is intentional.

Uninstall without deleting historical logs or job evidence:

```bash
.venv/bin/python manage.py launchd_refresh uninstall
```

Local SQLite uses WAL mode, an immediate transaction mode, and a 20-second busy
timeout so the nightly writer and local web process coordinate predictably.
PostgreSQL continues to use target-key advisory locks.

After each market refresh, record every active tracked portfolio:

```bash
uv run python manage.py snapshot_portfolios
```

The command uses the target-job ledger, creates immutable valuation and
position rows, deduplicates unchanged inputs across code deployments, and
records per-portfolio failures in the job details. A batch is failed only when
every active portfolio fails. Holdings that are unpriced, in another currency,
or stale relative to the rest of the portfolio are rejected; an entirely
outdated price feed is shown as a freshness warning on portfolio pages. Run the
command after `refresh_demo` in synthetic development or after `daily --region
us` for enabled live data.

### Monthly contribution ledger

Apply the portfolio migration before using deposits or monthly plans:

```bash
uv run python manage.py migrate
```

On `/portfolios`, create or open a manual portfolio, record an external
deposit, review the side-effect-free allocation preview, and explicitly
confirm only if the displayed local records match the intended bookkeeping.
Confirmation records purchases in StanStock; it does not send a brokerage
order. The default monthly preference is $600 and can be edited without
changing cash. Cash itself is managed only by immutable deposits and confirmed
planner purchases.

The planner recomputes its SHA-256 plan identity while holding the portfolio,
holdings, relevant market rows, and qualifying analysis rows. A concurrent
deposit, settings change, price advance, or analysis replacement rejects the
old preview and renders a fresh confirmation identity. Cash writes also use a
compare-and-swap condition. PostgreSQL row locks and local SQLite's immediate
transactions/WAL/busy timeout prevent concurrent confirmations from spending
the same cash.

Deposit, execution, purchase, and manual performance-baseline rows are
append-only at both the Django and database-trigger layers. Supported manual
quantity changes/removals append a post-change baseline. When another
unpriceable holding prevents that valuation, the edit is retained with an
immutable unavailable-boundary reason and contribution performance is
withheld rather than guessed. Removing or correcting the blocking state can
create a later valid baseline. Backups therefore must preserve these ledger
tables together with portfolio snapshots and data assets.

Build the owner's frozen sample portfolio from the latest provider-backed
opportunity run:

```bash
uv run python manage.py build_sample_portfolio \
  --username <owner> \
  --starting-capital 100000 \
  --top-n 5
```

The command is idempotent for an active owner/source-run pair and creates its
baseline snapshot in the same database transaction. It never calls the market
provider. Archive the sample before intentionally constructing a replacement
from the same source run; an archived sample cannot be restored while its
replacement is active.

An explicit prior `--target-date YYYY-MM-DD` is a research reconstruction,
not an on-time historical prediction. Disable the provider immediately with:

```bash
uv run python manage.py configure_twelve_data --disable
```

Provider termination or expiration also requires deletion of stored Twelve
Data data under the current terms; disabling alone does not delete immutable
assets.

### Destroying Twelve Data data after access ends

StanStock does not support selective deletion of one provider from an existing
research ledger. Twelve Data assets can be referenced by universe snapshots,
analyses, predictions, outcomes, simulations, and backup manifests; deleting
only the files or only the `DataAsset` rows would leave incomplete or
misleading evidence. The supported procedure is therefore a **full
installation reset**:

1. Stop every web, job, and backup process, revoke or rotate the provider key,
   and run `configure_twelve_data --disable` if the database is still
   reachable.
2. Delete every StanStock backup bundle and every external/off-site copy made
   while Twelve Data was enabled. Restoring one of those bundles would
   reintroduce the provider data.
3. Destroy the complete application database and the complete
   `STANSTOCK_DATA_DIR`. For the repository's development Compose stack,
   `docker compose down --volumes` removes PostgreSQL, assets, and the mounted
   backup volume together. Confirm that the Compose project contains no
   unrelated volumes before running it.
4. For direct SQLite development, remove `stanstock.sqlite3`, `var/data`, and
   `var/backups`. For external PostgreSQL or managed volumes, use the
   database/storage provider's documented destruction controls and also remove
   snapshots, replicas, object versions, and retained backups according to
   their lifecycle policies.
5. Recreate a clean installation only after `TWELVE_DATA_API_KEY` has been
   removed from the environment and `store_twelve_data_key --delete` has
   removed any Keychain item. Run migrations and `refresh_demo` to return to
   synthetic data, then confirm no `ProviderRecord` or `DataAsset` for
   `twelve_data` exists.

This procedure intentionally removes all StanStock research history, including
non-Twelve-Data records. Preserve no mixed backup as a workaround: the
application's immutability and provenance guarantees take priority over a
partial purge that cannot prove all derived copies were removed.

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

The source gate conditionally permits only the reviewed US Twelve Data scope;
broad US/European live OHLCV remains unsupported. Provider clients fail
explicitly on browser-verification HTML, access denial, quota exhaustion,
malformed responses, missing credentials, missing display-rights
or personal-use confirmation, an unlicensed additional user, and disabled
`ProviderRecord` state. No job may scrape HTML or turn a provider failure into
a successful empty market update.

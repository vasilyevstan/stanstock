# Operations

> Release state: see [README](../README.md#release-status). Do not use this
> implementation as a released live product until the listed gates pass.

## Profiles and settings

`settings.RESEARCH_PRODUCT_ENABLED` defaults to `true` and is controlled by:

```dotenv
STANSTOCK_RESEARCH_PRODUCT_ENABLED=true
```

This selects the replacement product paths; it does not enable a provider.

| Profile | Provider/source | Issuance claim |
|---|---|---|
| Demo `refresh_demo` | `synthetic_demo` | Always research-grade |
| Manual `daily --region us` | Enabled private Twelve Data path | Always research-grade |
| Generic `analyze` | Caller-selected local evidence | Always research-grade |
| Scheduled refresh | Enabled private Twelve Data path | Observed only when every deadline/revision/source gate passes |

SEC is a separate workflow. Its availability does not gate active price
research.

## Safe demo

```bash
uv run python manage.py refresh_demo
```

The command is offline, deterministic, and idempotent by target. It creates
genuine `synthetic_demo` source identities and uses the production intake,
membership, calculation, writer, verifier, and reader paths. It includes
qualified, Under-$10, and insufficient-history examples. It never contacts a
live provider or creates observed evidence.

## Live provider stop/go gate

Broad unattended US/European price collection is unsupported. Before the
bounded private US path:

1. confirm the current account rights for the exact private use;
2. store the key in an approved secret source;
3. run the bounded `source_spike`;
4. explicitly configure owner/display authorization; and
5. keep redistribution disabled.

Examples without secret values:

```bash
uv run python manage.py store_twelve_data_key
uv run python manage.py store_twelve_data_key --status
uv run python manage.py source_spike
uv run python manage.py configure_twelve_data \
  --enable \
  --plan basic \
  --confirm PERSONAL_SINGLE_USER_NONCOMMERCIAL_AUTHORIZED
```

Use the separate internal-display confirmation for a plan/agreement that
covers it. Basic mode is bound to one active licensed user. The owner remains
responsible for current terms.

Stop immediately on missing/expired rights, a demo key, disabled provider,
owner mismatch, extra active user under Basic, missing current authorization,
quota refusal, catalog ambiguity, identity mismatch, checksum failure,
insufficient target history, missed issuance deadline, or dirty fresh
scheduled checkout.

## Manual daily research

```bash
uv run python manage.py daily --region us
```

When the product is enabled:

- `--region us` is always research-grade;
- `--issuance-key` defaults to `manual`;
- reuse the same key for a retry of the captured set;
- choose a new non-reserved key for a deliberate same-target new intake; and
- `--issuance-key scheduled` is rejected because automation owns it.

The command captures the authorized core-plus-saved request before provider
work, reuses verified local history before credentials, acquires only missing
qualified history, records every admission state, and writes no partial
serving cohort.

`manage.py analyze` is demo/research tooling, not the live US or observed
reissue interface. It defaults to the generic demo provider, the default
scoring configuration, and `code_revision()` (including its
`"working-tree"` fallback), and explicitly passes `issued_on_time=False` for
every target. No provider, target, configuration, benchmark, or
`STANSTOCK_CODE_REVISION` flag can make it observed.

## Isolated local runtime

Keep a live native web process and its refresh LaunchAgent in a dedicated,
clean checkout rather than the checkout used for development. A full clone
also separates Git administration. Pin both processes to an exact reviewed
revision and give that clone its own environment installed from the existing
lockfile; copying a virtualenv can retain imports from another checkout.
Do not automatically pull development changes into a running installation.

This separates one source of refresh failures: unrelated development edits
or diagnostics tripping the clean-Git guard. It does not prevent provider
outages, machine sleep, process failure, or missing data. Runtime-generated
diagnostics can still dirty the new checkout. Keep development/agent work,
logs and diagnostic artifacts outside runtime source. Existing scheduler
logs already use a private location outside the checkout.

### Preserve permanent state before changing roots

The local development profile has checkout-relative defaults. Resolve and
compare the existing web and scheduler configuration before opening a
candidate database or constructing an asset store:

- For SQLite, set `STANSTOCK_SQLITE_PATH` to the existing absolute database
  path. Do not also set `DATABASE_URL`. Preserve an existing PostgreSQL
  profile as PostgreSQL; changing backends is a separate migration.
- Explicitly bind `STANSTOCK_DATA_DIR` and `STANSTOCK_BACKUP_DIR` to their
  existing absolute locations. Do not copy, relocate or bootstrap data as
  part of source isolation.
- Preserve the settings module, product/configuration, owner/display
  authorization and provider policy. A new empty database is not a valid
  fallback.
- Reuse the approved owner-only private environment through the existing
  strict loader. Do not shell-source it, duplicate secrets into scripts, or
  relax ownership and permission checks. The local installer expects
  `.env` in its selected project root; a file reference must resolve to the
  approved private file and pass the existing validation.

Inspect effective paths before checks with side effects: constructing an
`AssetStore` may create its directory, and opening a SQLite connection may
initialize WAL. Canonical SQLite process-lock identity also follows the
resolved database location. Different paths can therefore break both data
parity and cross-process serialization.

### Separate launching from release evidence

A private release-verification script is not a durable application launcher.
Assertions that a ledger exactly matches an old release snapshot can become
false after legitimate refreshes. Preserve that snapshot as historical
evidence; do not rewrite it or make future server startup depend on it.

An installation-specific native launcher is not a packaged StanStock command.
It should use the existing strict private-environment loader, verify its own
checkout/interpreter/import origin and approved clean revision, and bind
`STANSTOCK_CODE_REVISION` to that revision. For the existing local development
profile, preserve its Django `runserver`, loopback bind, settings and
no-auto-reload behavior. Keep unattended Keychain fallback disabled.
Do not substitute WSGI production defaults, Gunicorn, a new supervisor or a
database backend merely to change source location. Collect static assets
through the existing command as a controlled preparation step when needed.

### Controlled cutover and rollback

Prepare the candidate and a usable fallback launch command before stopping
the working server. Confirm a checksummed paired database/assets backup and
restore-readiness evidence; copying an open SQLite file without its WAL is
not an adequate backup. Use the existing backup workflow.

Prevent new manual/scheduled starts and drain active work under the existing
canonical locks against the same permanent database. Job status rows alone
do not prove quiescence. The installer replaces the loaded LaunchAgent; it
does not drain an active refresh. Do not unload a busy agent.

Use the existing `launchd_refresh install --project-root` interface only
after that fence is established. Preserve one LaunchAgent label, its
validated schedule/timezone and its existing job identities. Restart the
web process with the prepared same-profile launch command. Do not force a
provider refresh, recapture intake, run migrations or reissue predictions
to demonstrate the cutover.

Verify the actual native process and loaded scheduler execution chain, not
just a proposed plist or a separate shell import. Compare owner-authorized
reader output and registered evidence against the pre-cutover baseline at
the same cohort/cutoff. Existing withholding may be preserved; new
candidate-induced missing, withheld or failed output is a regression.
HTTP 200 alone is insufficient. Distinguish a genuine market-clock
transition from a source-location regression.

Historical cohorts retain their own issuance revisions; those need not
equal the new runtime revision. Fresh observed issuance retains all clean
revision, cutoff and deadline checks. Completed recovery remains ahead of
fresh-work guards.

On failure, fence and drain again, then restore the prepared prior
code/interpreter/web command and scheduler definition against the same
current permanent data. Do not restore an older database just to reverse a
source-location change, delete immutable evidence, or discard partial-job
recovery. Repeat the bounded acceptance comparison.

These are one-time, repeatable-on-demand deployment checks, not a monitoring
service. Source isolation does not install alerts, heartbeats, recurring
checks or automatic repair; failures may remain unnoticed until the
existing application, status page or private logs are inspected.

## Scheduled profile

The supported macOS LaunchAgent runs at **03:30 local time,
Tuesday-Saturday**. The reviewed timezone is `Europe/Tallinn`.

```bash
.venv/bin/python manage.py launchd_refresh install --timezone Europe/Tallinn
.venv/bin/python manage.py launchd_refresh status
```

Installation checks at least 400 days, including regular/early XNYS closes
and local/New York DST changes. Every invocation must be after the provider
publication delay and before the next XNYS open. A timezone change requires
reinstallation.

The private scheduler environment must be owner-only and must point the
scheduler at the same database, asset directory, product setting, owner, and
configuration as the web service. Keychain fallback is disabled for
unattended execution.
Use the ignored project `.env` with owner-only mode `0600`; interactive
keychain onboarding alone does not configure unattended access. Do not place
populated secret values in the plist or command arguments. Confirm the
recorded and machine timezone agree using `launchd_refresh status`.

The research profile uses version-specific parent and child jobs:

- scheduled parent;
- market/intake/issuance;
- prediction evaluation; and
- portfolio snapshots.

The parent independently verifies recorded child identities, intake,
membership, exact five-row output, calculations, manifests, and physical
assets. A success status alone is insufficient.

Refreshes created by the summary-enabled version additionally record a
separately verified model-outcome derivation child. Older parents retain
their original verification payload; their success does not imply a summary
exists. See [Derived model-outcome summaries](#derived-model-outcome-summaries)
for that independent stage and explicit old-run reconstruction.

### Fresh observed issuance

A fresh scheduled issuance requires:

- an observed target window;
- current owner/display authorization;
- `research-product-v1` and its pinned config;
- `provider="twelve_data"` and benchmark `SPY`;
- a clean committed checkout;
- `STANSTOCK_CODE_REVISION` equal to that exact 40-hex commit; and
- independent proof before commit that issuance is before the next regular
  XNYS session open and all sources are cutoff-safe.

An unsafe explicit observed request raises. It is not silently converted into
research output.

An exceptional same-target observed reissue can only be a direct:

```python
analyze_snapshot(
    ...,
    issued_on_time=True,
    provider="twelve_data",
    benchmark_subject="SPY",
    config_path=default_price_product_config_path(),
)
```

The parent/orchestrator must first bind the exact clean revision and
independently reprove the deadline. No reissue inherits another version's
on-time flag.

## Recovery and retries

Before provider enablement, credential resolution, or quota use, retries
search for and verify completed work using:

- recorded parent and child jobs;
- captured intake/owner/issuance identity;
- immutable membership and admission states;
- exact catalog, stock, and SPY assets;
- calculation and output manifests; and
- physical file checksums.

A completed target is recovered only when all identities agree. Conflicting
completed runs are an integrity error. A pending saved candidate does not
block core output, but it remains visibly pending and does not count as
analyzed.

Adding/removing My List names affects the next intake, not an already captured
retry. SPY is reused once and stays outside stock membership.

## Derived model-outcome summaries

The probability-first view uses a separate registered report from the same
deterministic paths as the immutable FHS projection. It is not a new issuance,
a calibration study, or a provider fetch.

To derive or recover the report for one existing run, use the same private
environment and data directory as the application:

```bash
uv run python manage.py derive_price_frequencies --run '<analysis-run-uuid>'
uv run python manage.py derive_price_frequencies --run '<analysis-run-uuid>' --verify
```

The first command recovers an already committed report before expensive
replay, or internally derives and registers one from the run's verified
sources. `--verify` is an offline, no-write semantic comparison, not merely a
file-checksum check. Neither command accepts externally supplied statistics,
bulk-backfills all runs, contacts a provider, or creates a prediction.

Reports use their actual publication time. A manual derivation for an older
issuance is labelled as a later reconstruction, cannot appear in an earlier
as-of read, and cannot inherit the original prediction's on-time status.
Preserve the source revision separately from the derivation revision.

Publication reuses the existing run-scoped cross-process job lock through a
durable database commit. New registration inside an application-owned outer
transaction is rejected; do not release the file lock before registration
commits. Blobs are addressed by their complete published bytes. A failed
insert can leave an unregistered blob, which is never served or overwritten;
a later retry records its own publication time in a distinct blob.

For new refreshes, summary generation is independently retryable. A failed
summary does not rewrite a completed source run; recovery identifies and
reuses completed children rather than paying for another acquisition.
Frequency child names include the owner, issuance key, and product identity.
An earlier fixed-name child is reusable only when it binds the same owner's
exact completed source; another owner's same-target success is not a retry.
Historical parent verification remains unchanged for runs without a summary
stage. Product availability and summary availability are separate facts:
an old complete run is not proof that its probabilities were derived.

At rollout, capture the exact current source run and eligible-listing
denominator before deriving its report. Retain immutable evidence,
preferences, portfolios, holdings, and the existing schedule. A missing or
failed report must show a precise summary state without hiding an
independently valid median/range.

## Serving checks

Use authenticated pages and local commands together:

Start from `/`, not only bookmarked deep links. In product mode it opens
Opportunities; the normal navigation must expose Opportunities and Under $10
without searching a long page. Follow the Under-$10 control and a stock
detail link, then inspect the remaining surfaces below. Check both a narrow
and desktop viewport. Direct HTTP success alone does not establish that a
user can find or understand the view.

1. `/status`: operational health, target freshness, admission, jobs, and
   verification;
2. `/opportunities` and detail: expected product, target, source grade,
   direction, risk, selected-horizon probabilities and median, and all four
   horizons on detail; pagination and detail links must preserve the selected
   horizon and applicable filters;
3. `/my-list`: saved/pending/admitted states and admitted-stock summaries
   without provider fetch;
4. `/predictions`: one decision plus four advisory rows per qualified
   listing, with each probability summary bound to that exact issuance and
   later reconstruction labelled explicitly;
5. `/performance`: observed cohorts separate from registered retrospective
   evidence, with non-observed, pending, and non-evaluable states distinguished
   and adverse comparison/numerical summaries still visible;
6. `launchd_refresh status`: installed trigger and timezone; and
7. `verify_assets`: registered file existence and checksums.

`/status` is not a duplicate forecast table. HTTP GET never performs source
acquisition, analysis, Monte Carlo replay, or study registration.

## Retrospective replay

The CLI is read-only:

```bash
uv run python manage.py replay_price_product \
  --run ANALYSIS_RUN_UUID \
  --all-selected \
  --format json \
  --output-file PRIVATE_NEW_REPORT_FILE
```

Alternatively select explicit listing UUIDs with `--listing-ids`. Exactly one
scope is required.
Replace the output placeholder with a new private path outside the asset
store. Omitting it prints the report to stdout, which may contain source and
listing details and must not be copied into public logs or issues.

`--output-file`:

- must be a new file;
- is created exclusively with mode `0600`;
- must be outside `STANSTOCK_DATA_DIR`; and
- is never overwritten.

The command contacts no provider, resolves no credentials, creates no jobs,
analyses, predictions, or assets, and cannot create observed evidence.

## Registering retrospective evidence

Registration is deliberately not a public management command or GET action.
The parent invokes the Python service:

```python
register_price_product_study(run=run, store=store)
```

The service calculates **all selected listings itself** using the frozen
protocol and exact immutable source run. It accepts no caller-authored metrics,
registers one canonical private report per source run, and fails on conflicting
evidence. A new source run appends a report; same-run retry reuses the original.
The performance page only reads and re-verifies a registered report.

Keep distinct:

- replay execution revision;
- source run's issuance revision;
- report generation time; and
- registered asset availability time.

Registration is a separate authorized local operation; a source-code revision
alone does not establish that it has run.

## Job states

| State | Meaning |
|---|---|
| `success` | Complete, independently verifiable output exists |
| `skipped` | Already complete or intentionally ineligible |
| `no_data` | No usable input; later retry may be valid |
| `failed` | Explicit failure recorded |
| `running` | Active attempt; stale work is failed before retry |

A success is never inferred from row counts alone.

## Backup, restore, and rollback

The database and asset store are one recovery unit:

```bash
uv run python manage.py backup
uv run python manage.py restore BACKUP_BUNDLE.tar.gz --verify-only
uv run python manage.py restore BACKUP_BUNDLE.tar.gz --confirm RESTORE
```

Backups belong outside `STANSTOCK_DATA_DIR` in a private writable
`STANSTOCK_BACKUP_DIR`. Stop all web/job processes before restore. PostgreSQL
restore is fail-fast and single-transaction; asset destinations and checksums
are validated before destructive database work.
Create and verify a paired private database-and-assets backup before any
migration or schema change; do not proceed with only a database dump.

Safe rollback:

- disable prospective serving/scheduling or redeploy a compatible revision;
- keep every immutable prediction, source, membership, outcome, and study
  row;
- keep archived method definitions and evidence;
- do not reset successful jobs;
- do not reverse a schema when prospective rows make reversal unsafe; and
- do not refetch merely to make rollback look complete.

## Provider-rights termination

If Twelve Data retention rights end, disable new access and follow the
documented full-installation destruction procedure covering database, asset
directory, backups, snapshots, replicas, and external copies. Selective
deletion is unsupported because immutable derived evidence can retain source
dependencies.

Never restore an old bundle that would reintroduce data after rights end.

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

The research profile uses version-specific parent, daily, and intake jobs:

- scheduled parent;
- market/intake/issuance;
- prediction evaluation; and
- portfolio snapshots.

The parent independently verifies recorded child identities, intake,
membership, exact five-row output, calculations, manifests, and physical
assets. A success status alone is insufficient.

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

## Serving checks

Use authenticated pages and local commands together:

1. `/status`: operational health, target freshness, admission, jobs, and
   verification;
2. `/opportunities` and detail: expected product, target, source grade,
   direction, risk, and four projection horizons;
3. `/my-list`: saved/pending/admitted states without provider fetch;
4. `/predictions`: one decision plus four advisory rows per qualified
   listing;
5. `/performance`: observed cohorts separate from registered retrospective
   evidence;
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

No real study is registered in the current release state.

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

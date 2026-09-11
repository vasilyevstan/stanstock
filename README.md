# StanStock

StanStock is a private, local-first stock-research application for transparent
US and European equity scoring, scenario analysis, immutable prediction
tracking, live personal portfolios, backtesting, and portfolio simulation.

It is a rules-based research system, not an automated trading service. It does
not use LLMs or trained machine-learning models to produce forecasts, scores,
or recommendations, and forecasts are never presented as guarantees. The
math-only medium-horizon implementation and long-horizon plan are documented in
[`docs/forecast-roadmap.md`](docs/forecast-roadmap.md).

## Current data boundary

The original free-provider gate remains **NO_GO for broad US/European
coverage**, but StanStock now has a **conditional US-only path** through
Twelve Data's documented API.

- The curated starter universe contains 100 NASDAQ/NYSE common-stock symbols.
  SPY is fetched once as the separate benchmark and the same immutable series
  maintains an investable SPY ETF listing without another provider request.
  SPY is not a universe member and never enters stock scoring or ranking. The
  universe is not represented as a licensed index.
- Twelve Data prices are explicitly split-adjusted price returns; dividends
  are not included and results must not be labeled total returns.
- Basic activation is limited to one authenticated active user who explicitly
  confirms personal, non-commercial, non-redistributed use. The provider's
  pricing page labels Basic as internal non-display, while its August 2026
  support guidance permits internal tools under Individual plans; use this
  mode only when your account terms cover your exact personal workflow.
- Grow, Pro, Ultra, or a reviewed custom agreement can instead be activated
  with an explicit internal-display confirmation.
- Stooq's public download route is protected against unattended automation,
  and its automation/private-retention rights could not be verified.
- SEC EDGAR and ECB data are viable official sources.
- filings.xbrl.org is usable for European filings but documents incomplete
  coverage, including Germany and Ireland.
- European live equity prices remain deferred.

StanStock still defaults to deterministic synthetic data and requires an
explicit provider enable step. It does not scrape around access controls,
silently broaden licensed use, or label historical catch-up as an on-time
prediction. See `docs/source-spike.md` for the evidence and exact limitations.

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

### Optional US Twelve Data workflow

On macOS, store the key in the current user's login Keychain through an
interactive prompt. The key is never passed as a command argument and is not
written to Git, dotenv files, logs, URLs, metadata, or the database:

```bash
uv run python manage.py store_twelve_data_key
uv run python manage.py store_twelve_data_key --status
```

Containers and non-macOS hosts should inject `TWELVE_DATA_API_KEY` through
their secret manager or process environment. Environment variables take
precedence over Keychain.

For a private local checkout, `.env` is also supported by Docker Compose and
is excluded from both Git and the image build context. Keep it owner-readable
only:

```bash
cp .env.example .env
chmod 600 .env
# Edit TWELVE_DATA_API_KEY in .env without committing the file.
```

Direct `manage.py` commands do not parse dotenv files themselves. Export the
local file into that command's process when not using Compose:

```bash
set -a
. ./.env
set +a
uv run python manage.py source_spike
```

For the Basic personal plan, run the bounded source probe and then activate
the single-user guard:

```bash
uv run python manage.py source_spike
uv run python manage.py configure_twelve_data \
  --enable \
  --plan basic \
  --confirm PERSONAL_SINGLE_USER_NONCOMMERCIAL_AUTHORIZED
uv run python manage.py daily --region us
```

Basic mode records the licensed owner and refuses provider jobs if another
active StanStock user exists. Authenticated provider-backed pages return 403
for any other user. There is no public signup. For Grow, Pro, Ultra, or a
reviewed custom agreement, use
`--confirm PERSONAL_INTERNAL_DISPLAY_AUTHORIZED`.

SEC EDGAR needs no account or API key. Automated requests must identify the
application and a monitored contact in `SEC_USER_AGENT`; keep that value only
in the ignored local `.env`:

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

The reviewed CIK configuration covers the 100-stock US universe and excludes
SPY. SEC ingestion preserves the official mapping, current and historical
submissions, Companyfacts, exact accession/acceptance provenance, append-only
revisions, full instant/duration period identity, and current SIC snapshots.
Daily automation polls submissions, refreshes Companyfacts after a new filing,
retries a still-missing filing at most once daily for seven days, and then
falls back to staggered periodic reconciliation rather than downloading all
history every night. Normalized facts retain a separate immutable link to the
submissions or history asset that supplied their acceptance boundary.

`daily --region us` validates the configured symbols against Twelve Data's
NASDAQ/NYSE catalogs, stores the raw JSON and normalized Parquet as immutable
vintages, captures an observed universe snapshot for the latest eligible
session, analyzes eligible listings, appends the supported short-horizon
decision prediction, and issues separate 6- and 12-month price-only advisory
forecasts. When SEC is enabled, the same immutable run also issues separate
3- and 5-year advisory forecasts or an explicit insufficiency reason. The
medium engine writes one private immutable Parquet panel per analysis run,
uses fixed-epoch non-overlapping cohorts, weights each market cohort equally,
and shrinks conditional p20/p50/p80 returns toward the unconditional
distribution. Probability stays hidden until effective support, listing
diversity, calendar span, matched market-regime breadth, and walk-forward
calibration all pass.

The single SPY benchmark response supplies both regime evidence for those
forecasts and the investable ETF market row; it is not fetched twice.
Current-universe historical panels are explicitly labeled survivorship-biased
research evidence and are not presented as live skill. Existing `medium` and
`long` records retain their legacy identities. Exact `3y` and `5y` forecasts
use a separate deterministic SEC engine: positive compatible FCF/share takes
priority, EPS/share is eligible only when FCF evidence is genuinely
unavailable, current SIC peers must meet frozen sample floors, and growth
fades toward a fixed terminal rate while valuation partially reverts toward a
bounded peer median. Every selected annual period must reconcile
net-income-derived EPS with reported diluted EPS, TTM diluted shares must stay
within 15% of the latest overlapping annual basis, and beginning/end invested
capital must use the same canonical and source concept definitions. Scenario
returns divide by the actual current multiple while using a bounded current
multiple only as the reversion anchor; a raw multiple below the supported
family floor is withheld rather than being raised mechanically. Missing,
negative, incompatible, stale, or unsupported inputs stay `Insufficient
evidence`; probability remains unavailable. Because there is no verified
split-event feed, each prediction also records and discloses the bounded
period after its latest SEC share evidence as residual post-period split risk.
These advisory rows cannot change BUY/HOLD/AVOID, opportunity ranking, or
decision hit rates. A withheld advisory forecast (all scenario returns null)
stays unresolved when evaluated and is excluded from advisory reporting
denominators rather than being counted as a matured call. A stricter
eligibility gate ships as a new versioned configuration; the prior version's
config hash, behavior, and output payloads remain unchanged and reproducible.
An explicit older `--target-date YYYY-MM-DD` is labeled
research-grade. A successful target is idempotent; another invocation creates
a skipped job and makes no provider requests.

After upgrading an existing database that already contains immutable SPY
benchmark assets, create its ETF listing locally without consuming provider
credits:

```bash
uv run python manage.py migrate
uv run python manage.py sync_investable_etfs
```

On a private macOS checkout, install the validated local scheduler after the
manual provider workflow succeeds:

```bash
.venv/bin/python manage.py launchd_refresh install
.venv/bin/python manage.py launchd_refresh status
```

The LaunchAgent invokes one recoverable refresh at 03:30 local time
Tuesday-Saturday. Installation is refused unless that wall-clock schedule is
after Twelve Data's publication delay and before the next XNYS opening across
regular closes, early closes, and DST transitions. The ignored `.env` must be
owner-only (`chmod 600 .env`); its values are sourced by a private runner and
never copied into the plist. A late sleep/wake invocation refuses automatic
research-grade backdating. Remove the job with
`.venv/bin/python manage.py launchd_refresh uninstall`.

Install the LaunchAgent from a clean, committed checkout; the scheduled
market child requires a clean Git worktree
(`stanstock.core.revision.clean_git_revision`) and fails closed on local
changes. If the primary development checkout is intentionally dirty, install
the LaunchAgent from a separate clean runtime checkout instead, pointed at
the same local SQLite database and `STANSTOCK_DATA_DIR` via
`STANSTOCK_SQLITE_PATH` (see above).

The Basic quota guard is 8 credits/minute and 800/day. The 100-symbol
configuration uses approximately 103 credits per full run (two catalogs, 100
stocks, and SPY), so two configured daily runs remain below the local daily
ceiling. Requests are spaced at least 7.5 seconds apart and a run is rejected
before provider access when its estimated credits would exceed the remaining
local allowance. This accounting cannot see credits consumed by other
applications using the same Twelve Data account. Disable access immediately
with:

```bash
uv run python manage.py configure_twelve_data --disable
```

Twelve Data data must remain private, may not be redistributed without
appropriate rights, and must be deleted after the subscription or agreement
ends as required by the provider's current terms. The supported termination
procedure is a full installation reset covering the database, data directory,
backups, snapshots, and replicas; see
[Operations and recovery](docs/operations.md#destroying-twelve-data-data-after-access-ends).

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

A separate clean runtime checkout (for example, one dedicated to the
LaunchAgent schedule while the primary development checkout stays
intentionally dirty for in-progress work) can share the same local SQLite
database and `STANSTOCK_DATA_DIR` as the primary checkout by setting
`STANSTOCK_SQLITE_PATH` to an absolute path. Leave `DATABASE_URL` unset when
using `STANSTOCK_SQLITE_PATH`; the two are mutually exclusive and setting
both fails closed at startup rather than silently picking one. A relative
path is also rejected; `STANSTOCK_SQLITE_PATH` must already be absolute after
`~` expansion.

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

Create several live tracked portfolios from `/portfolios`. Each portfolio is
private to its owner and stores current holdings plus immutable dated
valuation snapshots. Holdings are currently restricted to the portfolio's
base currency and must have a current persisted market row. SPY can be held
and valued from its ETF market row without a stock analysis; other ETFs are
not enabled. Snapshot returns are unrealized price returns against the average
costs entered by the owner; cash is excluded from that return, dividends are
excluded unless the source explicitly includes them, and the value history
includes holding/cash changes rather than claiming a time-weighted return.

Keep a separate private symbol preference list from `/my-list`. **My list**
accepts unique active US common-stock/ADR listings and symbols identified by
the latest checksummed local Twelve Data NASDAQ/NYSE catalogs. It never fetches
provider data, changes universe membership, runs analysis, or creates a
portfolio holding. Catalog-valid symbols outside the current research universe
remain visible with explicit unavailable price/analysis states.

Manual portfolios also support immutable external deposits and recorded
monthly allocations. The editable monthly preference defaults to $600.
Previews are side-effect free and target 70% of total NAV in SPY plus at most
30% in one currently qualified short-horizon stock satellite. Fractional
shares are enabled by default; whole-share mode rounds down and carries the
remaining cash. The planner never sells, never allocates new money to the
Under-$10 speculative watchlist, and never sends a brokerage order or makes a
provider request. Confirmation recomputes a checksummed plan from locked
portfolio, price, and analysis state before appending immutable purchase
records tied to exact market sessions and source assets.

Contribution-adjusted profit/loss subtracts immutable deposits from current
NAV relative to an eligible valuation boundary. It is a simple since-boundary
return, not a time-weighted or money-weighted result. Fresh, coherent,
split-adjusted, dividend-excluding price evidence is required. A supported
manual quantity change or removal appends an immutable post-change baseline
so performance restarts without treating the change as profit; if that
valuation cannot be established, the edit still succeeds and performance is
explicitly withheld until a later valid baseline supersedes it.

The same page can build an idempotent, frozen StanStock sample portfolio from
the latest provider-backed opportunity run. It equal-weights up to five
eligible USD listings by default, preserves the source run and reference
prices, and creates an immutable baseline snapshot. The current price-only
sample is a research-reference basket rather than an executable-fill claim;
its short signal horizon, research grade, no-rebalance policy, split-adjusted
price-return basis, and dividend exclusion remain visible. ETFs are excluded
from stock sample construction. Newly constructed samples also exclude the
`Under $10 - speculative watchlist` band and record the price-band policy
used. Construction classifies the immutable analysis reference close at the
run's target date rather than a later mutable close; existing holdings and
older frozen samples are not rewritten.

The equivalent command is:

```bash
uv run python manage.py build_sample_portfolio \
  --username <owner> \
  --starting-capital 100000 \
  --top-n 5
```

Record all active portfolios after a market-data refresh:

```bash
uv run python manage.py snapshot_portfolios
```

## Authenticated pages

- `/opportunities` - ranked analyses grouped and filterable by neutral current
  USD price bands, with the close date displayed.
- `/stocks/<listing-id>` - scenarios, factor evidence, risks, and provenance.
- `/etfs/<listing-id>` - SPY price-return, volatility, drawdown, benchmark
  identity, and provenance without a stock recommendation.
- `/predictions` - the append-only prediction ledger, including explicit
  decision/advisory role and the recorded price provider/subject.
- `/performance` - decision outcomes with minimum-sample safeguards and a
  separate advisory-error section when advisory outcomes exist.
- `/my-list` - private owner-scoped symbol preferences validated only from
  existing listings or checksummed locally stored stock catalogs.
- `/portfolios` - owner-scoped holdings, immutable deposits/purchases,
  monthly allocation previews, contribution-aware performance, and valuation
  history.
- `/simulations` - backtest and portfolio runs through one accounting model.
- `/status` - database, asset-store, provider, job, and prediction status.
- `/methodology` - point-in-time, scoring, scenario, and limitation summary.

When persisted analysis does not exist, the status page clearly labels its
illustrative synthetic rows. Data-bearing pages require authentication;
`/healthz` exposes only coarse readiness information. The authenticated data
label is derived from the latest serving analysis provenance rather than the
development debug setting, and the market overview is restricted to that
serving run's listings. Historical prediction rows retain their own
provider/synthetic labels.

## Integrity guarantees

- Permanent company, security, and listing IDs; ticker text is not identity.
- Immutable source assets and explicit `retrieved_at`/`available_at` vintages.
- Historical reads use the `AsOfData` boundary, including physical row-level
  clipping and ascending ordering through the requested market date.
- Historical research reconstructions cap fact availability and price rows at
  the logical target while retaining their actual later generation/retrieval
  timestamps; only on-time observed runs count as live evidence.
- No reissue of an immutable prediction inherits another version's on-time
  status; each version independently proves its own next-market-session-open
  deadline.
- Within an exact method/configuration/provider cohort, performance counts each
  listing/target/horizon/evidence-role observation once from its earliest
  reportable issuance; later observed reissues remain in the immutable ledger.
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

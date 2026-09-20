# StanStock

StanStock is a local-first Django application for private, transparent stock
research, immutable prediction tracking, portfolio research, backtesting, and
simulation. It is not a broker, automated trader, fitted machine-learning
system, or promise of investment performance.

## Release status

This revision implements `research-product-v1`, the prospective replacement
for the score-led research experience.
[PR #53](https://github.com/vasilyevstan/stanstock/pull/53) records exact
revision, independent review, CI, operational acceptance, and release evidence.

The [focused workspace release](https://github.com/vasilyevstan/stanstock/pull/62)
provides Opportunities, Market and saved Portfolios. The separately released
[drift study](docs/price-research.md#synthetic-drift-and-parameter-uncertainty-research)
and [candidate-policy study](docs/price-research.md#synthetic-candidate-policy-research)
are synthetic-only research, not replacement forecasts or live buy/sell lists.

Implementation, release, and local activation are separate facts. Checking out
code or enabling the product setting does not enable a provider, register a
retrospective study, or prove forecasting skill. A private installation must
follow the [operating gates](docs/operations.md), preserve a paired backup,
and establish actual source-to-screen coverage before claiming activation.
An existing installation retains its prior behavior until it is deliberately
upgraded and configured. These documents explain the implemented contract;
they do not certify the state of a particular running installation.

## Research product at a glance

The replacement product has two deterministic operators:

| Operator | Role | Output |
|---|---|---|
| `us-relative-momentum-v1` | Six-month decision evidence | Raw direction plus Buy / Hold / Avoid suggestion |
| `us-price-fhs-v1` | Advisory price research | Median and detailed price ranges at 6m, 12m, 3y, and 5y |

The output is exactly one `StockAnalysis` and five immutable predictions for
each qualified listing: one 6m momentum decision and four FHS advisory rows,
including a distinct 6m advisory row. The product has no overall score,
calibrated confidence percentage, validated probability of gain, or automated
portfolio instruction. Its original probability/confidence fields remain null.

**Model-estimated probabilities** summarize the same deterministic FHS paths
in a separate, immutable, source-bound report. The three outcomes are **Loss**,
**Flat to +20%**, and **Above +20%**, measured at the selected horizon from the
dated reference close. These are shares of model simulations, **not validated
real-world odds**. They do not change the momentum suggestion or allocation.
Median return remains visible; detailed price ranges and downside sensitivity
remain available on the stock page. A missing summary does not hide an
otherwise valid median or price range.

Published research motivates the operators, but StanStock's individual-stock
rules are not paper replications and have not been proven profitable. See
[Price research](docs/price-research.md) for formulas and nonclaims.

## Data and admission

The live candidate set is the unchanged curated 100-name US core plus at most
20 owner-saved names, deduplicated by permanent listing identity. SPY is a
separate benchmark and never becomes stock membership.

Each analyzed stock must have:

- verified US/USD common-stock or supported-ADR catalog identity;
- one registered immutable split-adjusted price vintage;
- the separate registered SPY benchmark vintage; and
- exactly 757 consecutive common XNYS-session closes ending at target session
  `T`, which yields 756 daily returns.

Missing sessions are not interpolated. SEC facts are not a prerequisite for
the active price product. Qualified Under-$10 listings receive the same
momentum research and four projections, but remain a speculative watch with
0% new allocation, no BUY promotion, no highlight, and no new sample-basket
admission.

Twelve Data's currently documented split-adjusted price response does not, by
itself, prove split-compatible volume. Price direction and all four
projections can therefore remain calculable while the liquidity gate is
unavailable and a positive signal stays **Hold** rather than BUY.

## Runtime flow

```text
fixed core + captured owner-saved names + entitlement
        -> immutable intake before provider work
        -> reuse verified catalog/history or acquire only missing history
        -> immutable membership and exact stock/SPY source closure
        -> momentum + FHS calculation
        -> one analysis + exact five-row prediction ledger + output proof
        -> separate offline same-path outcome report, when registered
        -> shared fail-closed reader
        -> Opportunities / detail / Saved stocks / status / history / performance
```

Authenticated browsing never calls a provider, runs FHS, registers a study,
or mutates evidence. The reader verifies owner authorization, registered
manifests, source identity, and physical checksums before rendering. Completed
target recovery reuses exact captured records before credential resolution or
new quota use.

Outcome reports have their own actual publication time. A later derivation
does not rewrite an issuance or become evidence that probabilities were
published at its original decision time. Old source verification remains
valid independently of whether a derived report exists.

## Safe local demo

The main application setting defaults `RESEARCH_PRODUCT_ENABLED` to `true`;
the environment name is `STANSTOCK_RESEARCH_PRODUCT_ENABLED`. Demo mode
remains safe and offline: `refresh_demo` creates visibly synthetic
`synthetic_demo` intake, history, membership, calculations, and research-grade
output through the production paths.

### Docker Compose

```bash
docker compose up --build
```

Open <http://localhost:8000> and sign in with the development owner
credentials configured for the local environment.

### Direct Python

```bash
uv sync --all-groups
uv run python manage.py migrate
printf 'Owner password: '
read -rs STANSTOCK_OWNER_PASSWORD
printf '\n'
STANSTOCK_OWNER_USERNAME=local-owner \
STANSTOCK_OWNER_PASSWORD="$STANSTOCK_OWNER_PASSWORD" \
  uv run python manage.py bootstrap_owner
unset STANSTOCK_OWNER_PASSWORD
uv run python manage.py refresh_demo
uv run python manage.py runserver
```

The demo includes qualified synthetic examples, a qualified Under-$10
example, and an insufficient-history example. Synthetic output is always
research-grade and never establishes historical or future forecast skill.

## Optional private US workflow

Broad unattended US/European OHLCV remains `NO_GO`. The only approved live
price path is the bounded private US Twelve Data workflow, disabled unless a
non-demo key and the required personal/internal-display entitlement are
explicitly configured. Provider data must not be redistributed.

After the bounded source probe and rights review, an owner may enable the
provider and run:

```bash
uv run python manage.py daily --region us
```

With the research-product setting enabled, manual `daily --region us` is
**always research-grade**. `--issuance-key` defaults to `manual`; the reserved
`scheduled` identity is rejected. The generic `analyze` command also remains
research-only regardless of provider, configuration, benchmark, target, or
revision flags.

The supported macOS profile invokes `scheduled_refresh` at **03:30
Europe/Tallinn, Tuesday through Saturday** after validating the machine
timezone across market closes and DST. A fresh scheduled issuance requires an
observed window, exact owner/provider/config binding, and a clean committed
revision. SEC ingestion is separate and does not block the active price
product.

See [Operations](docs/operations.md), [Deployment](docs/deployment.md), and
[Source capability](docs/source-spike.md) before enabling any live path.

## Research evidence

Forward observed outcomes and retrospective comparisons are separate lanes.
The frozen retrospective protocol uses a fixed 2019-09-03 epoch,
horizon-spaced anchors, development outcomes before 2024, validation anchors
and complete outcomes within 2024, and final-holdout anchors from 2025 with
complete outcomes through 2026-09-11. Partition crossings are purged and the
holdout cannot be used for tuning.

The read-only `replay_price_product` CLI can inspect an exact immutable source
run without provider access or database/asset writes. A separate parent-invoked
Python service, `register_price_product_study(run=..., store=...)`, calculates
the full selected cohort itself and registers canonical evidence; it does not
accept caller-authored metrics. HTTP GET never performs replay.

Only source-bound registered evidence establishes that a retrospective study
ran. Synthetic examples, downloaded prices, and Monte Carlo path counts do not.

## Authenticated pages

- `/opportunities` — the landing page: a paginated stock comparison with
  median returns and model-simulation shares at 6m, 12m, 3y, or 5y. Existing
  BUY-qualified and price-band shortlists are secondary views below the main
  results, with unchanged selection rules;
- `/opportunities?price_band=under_10` — the directly accessible Under-$10
  research view, still restricted to 0% new allocation;
- `/stocks/<listing-id>` — method assumptions, source closure, decision, and
  all projection horizons;
- `/market` — benchmark, breadth, sector and latest-price observations for
  the stored research cohort, not a whole-market or live-quote service;
- `/my-list` — **Saved stocks**, reached through Market: saved-candidate
  readiness and matching horizon summaries, with existing add/remove controls.
  Saving a name affects a subsequent scheduled refresh, not an immediate
  provider request or a portfolio holding;
- `/status` — operational health, admission, freshness, and verification, not
  a second forecast table;
- `/predictions` — immutable decision/advisory history;
- `/performance` — separate observed outcome cohorts and registered
  retrospective comparisons;
- archive routes — frozen prior methods and their original definitions;
- `/portfolios` — saved named portfolios first, with ordinary portfolio
  creation and retained frozen samples;
- `/portfolios/<portfolio-id>` — Holdings, Activity, Plan and Settings.
  Optional activity history and allocation previews are prepared only when
  requested. Confirmation still independently checks the current plan under
  the existing bookkeeping guards; no brokerage order is sent;
- `/simulations` and `/simulations/<run-id>` — historical simulation readers,
  reached through Methodology. New browser simulation execution is retired.

All data-bearing pages require authentication. `/healthz` exposes only coarse
readiness.

The primary navigation is **Opportunities**, **Market** and **Portfolios**.
Under $10 remains a visible price filter on Opportunities, not a separate
top-level destination. Research performance and prediction history are
contextual research links; Methodology and Data & updates are footer utilities.
Research performance is not personal portfolio performance. Each stock's
detail page retains all four projection horizons and their limitations.

Every Opportunities comparison and shortlist card has a collapsed **Why this
forecast?** disclosure. It explains the selected horizon using recorded
historical drift and its same-shock zero-log-drift sensitivity, separately
from the six-month action. Supporting evidence and the full-detail link stay
on the same page; opening a disclosure uses no AI, provider request, or new
simulation. It does not explain company news or fundamental value.

The interface uses flat dark surfaces and compact financial rows rather than
decorative dashboard panels. Saved records appear before creation and settings
forms. Current values, missing-data warnings and the distinction between
unrealized P/L, contribution-adjusted simple returns and frozen-model returns
remain explicit.

New generated sample portfolios are also retired from the browser. An
authenticated, CSRF-valid simulation creation POST returns 405, and a retired
sample-creation action returns an explicit 400 without invoking its builder.
Existing records, historical URLs and the `simulate` and
`build_sample_portfolio` management commands remain available. This removes
browser execution adapters, not financial history, forecast engines or the
underlying research capabilities.

Shortlists re-rank each verified cohort; they do not automatically discover
additional stocks. Empty BUY or Under-$10 shortlists are honest outcomes,
not a reason to lower evidence gates or promise gains. A selected price band
opens the focused full list.

Forecast availability, BUY eligibility, and a measured track record are
different states. Research-grade or synthetic records do not become observed
evidence by waiting; a withheld all-null forecast is not evaluable. The
scoreless momentum method also does not supply a legacy overall score or
qualify a legacy scored sample basket.

## Integrity guarantees

- Permanent company, security, listing, asset, analysis, and prediction
  identities.
- Immutable source vintages and explicit `available_at`, `retrieved_at`,
  `generated_at`, `data_cutoff`, and per-version `issued_on_time`.
- Historical reads physically exclude price rows after the requested market
  date.
- Complete canonical input/calendar hashes, registered manifests, and
  physical checksum verification.
- Missing, incompatible, stale, unauthorized, or unverified data remains
  explicit; it is never coerced to zero or a success-shaped default.
- Safe retries recover committed target evidence before provider enablement,
  credentials, or quota.
- Frozen prior configs, payloads, predictions, and performance meanings stay
  archived; the new product does not rewrite them.

## Backup and recovery

The database and `STANSTOCK_DATA_DIR` assets are one recovery unit:

```bash
uv run python manage.py backup
uv run python manage.py restore BACKUP_BUNDLE.tar.gz --verify-only
uv run python manage.py restore BACKUP_BUNDLE.tar.gz --confirm RESTORE
```

Stop web and job processes before restore. Rollback disables future
prospective serving/scheduling or redeploys a known compatible revision; it
does not delete immutable rows or rewrite old predictions. See
[Operations](docs/operations.md#backup-restore-and-rollback).

## Development checks

```bash
make check
uv run python manage.py makemigrations --check --dry-run
uv run python manage.py check --deploy --settings=stanstock.settings.prod
```

Tests and CI use synthetic fixtures only and never contact live providers.
Current release acceptance is stated only in [Release status](#release-status).

## Documentation

- [Architecture](docs/architecture.md)
- [Price-research specification](docs/price-research.md)
- [Methodology](docs/methodology.md)
- [Point-in-time integrity](docs/point-in-time.md)
- [Accepted roadmap and release gates](docs/forecast-roadmap.md)
- [Operations and recovery](docs/operations.md)
- [Deployment](docs/deployment.md)
- [Source capability](docs/source-spike.md)
- [Testing](docs/testing.md)
- [Development agents and task authorization](.github/agents/README.md)
- [Public agent team and workflow guide](https://github.com/vasilyevstan/stanstock/wiki/Agent-Team-and-Workflow)
- [Known limitations](docs/limitations.md)

## Financial disclaimer

Forecasts are conditional estimates, not guarantees. Historical,
retrospective, and simulated results do not guarantee future performance.
StanStock does not provide personalized financial, investment, tax, or legal
advice.

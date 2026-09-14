# Architecture

> Release state: see the centralized [README status](../README.md#release-status).
> This document describes the implemented `research-product-v1` contract.
> Candidate activation, protected release, and each installation's deployment
> state are distinct; they are recorded in the release evidence.

StanStock is one Django modular monolith. Module boundaries protect evidence,
authorization, and point-in-time correctness; they are not a reason to add
distributed infrastructure.

## Modules

| Module | Responsibility |
|---|---|
| `core` | Owner bootstrap, health, target-date jobs, schedule verification, backup, and restore |
| `data` | Permanent identities, provider boundaries, immutable assets, captured intake/membership, SEC facts, FX, and as-of reads |
| `research` | Price operators, analyses, immutable predictions, verification, outcomes, replay, and reporting |
| `portfolio` | Owner-scoped holdings, immutable cash/purchase records, planning, and valuations |
| `simulation` | Shared deterministic accounting for backtests and portfolio comparisons |
| `web` | Authenticated server-rendered product, archive, portfolio, and status pages |

Raw provider clients remain under `stanstock.data.providers`. Research and
simulation consume selected normalized evidence, not provider clients.

The stack remains Django, PostgreSQL (SQLite for direct local development),
Polars, NumPy, and content-addressed files under `STANSTOCK_DATA_DIR`. There is
no DRF, DuckDB, Node build chain, Celery, distributed scheduler, fitted ML
model, or runtime AI agent.

## Storage and identity

Relational rows store permanent identities, snapshots, manifests, jobs,
analyses, predictions, outcomes, portfolios, and normalized vintages. Large
source and derived payloads live as immutable files. Each `DataAsset` records
provider, subject, relative path, SHA-256, retrieval/availability times,
economic period, and provenance.

A same-path/same-checksum write is idempotent. Different bytes at the same
path fail. Database rows and asset files form one backup and recovery unit.

Predictions, data assets, fundamental facts, and FX rates are append-only.
Corrections create new identities/vintages. A later product version never
rewrites a frozen historical result.

## Price-product data flow

```text
curated 100-name core + at most 20 captured saved names
        |
        v
immutable owner/entitlement intake before provider work
        |
        v
verified catalog + reusable registered history
        |                         |
        | missing history only    v
        +-----------------> bounded provider acquisition
        |
        v
new immutable qualified membership; SPY remains separate
        |
        v
exact target cutoff + 757 common stock/SPY session closes
        |
        +--> us-relative-momentum-v1 (6m decision)
        |
        +--> us-price-fhs-v1 (6m, 12m, 3y, 5y advisory)
        |
        v
one analysis + exact five-row immutable output + registered proof
        |
        v
shared owner-bound verifier/reader
        |
        v
primary research, detail, My List, status, history, performance, archives
```

Saved preferences are not membership. Adding/removing a name affects the next
intake; a retry reuses its captured set. Pending, rejected, or
insufficient-history candidates stay explicit and are not counted as
analyzed.

SEC facts are not on the active price-product dependency path. They remain
evidence for frozen archived methods. FX remains on existing
valuation/simulation paths.

## Exact output ownership

For each qualified listing the writer persists:

1. one `StockAnalysis`;
2. one 6m decision prediction for `us-relative-momentum-v1`; and
3. four advisory predictions for `us-price-fhs-v1` at 6m, 12m, 3y, and 5y.

The decision row and 6m advisory row are different evidence roles and method
identities. The expected five-row multiset is planned before writing and
verified independently. A decision's null price-range fields do not make it an
all-null advisory.

Calculation artifacts bind the complete stock/SPY source window, immutable
asset references, input/calendar hash, configuration identity, result shape,
and output manifest. The reader verifies those records and physical bytes; it
does not rerun Monte Carlo trajectories during HTTP GET.

## Derived outcome-frequency evidence

The probability-first display does not change the frozen prediction schema
or calculation payload. A separate run-bound `DataAsset` records internally
derived integer outcome counts from the same deterministic terminal paths.
The pure frequency operator and its evidence registration/reader form one
bounded extension; there is no new forecasting service or provider.

Registration reconstructs only the original run's verified immutable sources,
checks complete input/seed/projection identity, and admits the complete
canonical report. Three matching quantiles are a compatibility check, not
authentication of a distribution. No writer accepts caller-authored counts.
Idempotent registration recovers committed evidence before replay and
serializes publication without rewriting the source run.

The report's actual publication time controls availability. Current history
can show a labelled later reconstruction, but an earlier as-of request cannot
see it. Its timing never inherits `Prediction.issued_on_time`. HTTP readers
check exact-run identity, current owner/display rights and registered bytes
without simulation; explicit offline verification re-derives the contents.

New scheduled parents keep a separate frequency-child verification block.
Existing canonical source verification and historical replay payloads stay
unchanged. A source run can remain valid when its frequency report is absent
or fails; the UI exposes that summary state separately and retains an
independently valid median/range.

## Runtime profiles

`STANSTOCK_RESEARCH_PRODUCT_ENABLED` maps to
`settings.RESEARCH_PRODUCT_ENABLED` and defaults to `true`.

- In demo mode, the selected provider identity is `synthetic_demo`; the same
  intake, calculation, writer, verifier, and reader paths are used, but output
  is always research-grade.
- Manual `daily --region us` uses the active research profile and always
  requests research-grade issuance.
- The local scheduler uses the same product/config/reader profile and may
  request observed issuance only inside the validated deadline from a clean
  committed revision.
- When the feature is disabled, frozen archive behavior remains available; the
  product reader never silently substitutes a legacy run.

## Serving and authorization

Primary pages select the expected product/config, target session, authorized
captured owner, and source provider. They verify:

- current display authorization;
- immutable intake and membership binding;
- official catalog identity;
- exact source assets and physical checksums;
- calculation and output manifests;
- roles, methods, horizons, probability absence, and numeric shape; and
- current target freshness.

Invalid, stale, ambiguous, or unauthorized output is suppressed with an
explicit state. Authenticated browsing never fetches provider data or mutates
the evidence ledger.

## Replay and registered comparison evidence

The retrospective study is deliberately separate from prospective issuance:

- `replay_price_product` reads an exact immutable run and writes no database or
  asset evidence;
- `register_price_product_study(run=..., store=...)` is a parent-invoked
  Python service that calculates the complete selected cohort itself and
  registers a canonical immutable report;
- the performance GET path reads only registered evidence and re-verifies its
  source run; it never accepts caller-authored metrics or runs a replay.

Execution revision, source-run revision, report generation time, and actual
registration availability are distinct provenance fields.

## Jobs and recovery

The product uses versioned parent, daily, and intake `JobRun` identities.
Completed work is independently reconstructed from recorded parent details,
intake, membership, manifests, calculations, and physical assets before any
credential or provider spend. Conflicting completions fail.

Scheduled market, evaluation, and portfolio stages remain independently
recoverable. SEC is a separate workflow and is not a prerequisite for active
price research.

## Security and deployment boundary

Every data-bearing page requires Django authentication. There is no public
signup. State-changing forms use CSRF protection. Provider display
authorization is owner-bound and rechecked when serving.

The intended deployment is one private instance. Production defaults use
HTTPS, secure cookies, HSTS, an unprivileged process, a read-only root
filesystem, durable PostgreSQL and asset volumes, and a separate writable
backup volume. Credentials are read from approved secret sources and are not
stored in application tables, URLs, logs, reports, or committed files.

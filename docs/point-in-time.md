# Point-in-Time Integrity

> Release state: see [README](../README.md#release-status).

StanStock separates the market date a result describes from when evidence was
available, retrieved, calculated, issued, and later registered.

## Clocks and dates

| Field | Meaning |
|---|---|
| `target_date` | Market session whose close is analyzed |
| `period_start` / `period_end` | Economic interval described by a source |
| `published_at` / `filed_at` | Source-declared publication time |
| `available_at` | Earliest defensible time StanStock may use the observation |
| `retrieved_at` | Actual first storage time of those exact bytes |
| `observed_at` | Time a retrieval carried those bytes |
| `data_cutoff` | Latest information time allowed for the analysis |
| `generated_at` | Actual analysis/report generation time |
| `issued_on_time` | Per-version proof of issuance before its deadline |

These fields are not interchangeable.

## Captured product intent

For the live product, owner-saved preferences are mutable and therefore not
historical membership authority. Under a target lock, StanStock first records
an immutable intake containing:

- target date and issuance key;
- owner and display-entitlement identity;
- pinned product/policy identity;
- unchanged curated core symbols; and
- at most 20 saved symbols.

Only after this capture may the workflow resolve catalogs, credentials, or
missing history. A retry reuses its captured set. A deliberate same-target
different set requires a new issuance key and new immutable intake.

## Price-source closure

For each admitted listing, source selection binds:

- permanent listing identity and verified catalog rows;
- exact normalized and raw stock assets;
- exact normalized and raw SPY assets;
- provider, subject, checksum, retrieval, and availability;
- 757 common XNYS sessions ending at the target;
- exact closes, volumes, and volume-basis flag;
- decision/source-availability boundary; and
- complete canonical input/calendar hash.

A short current-price asset cannot shadow an older qualified full-history
asset. Once selected, a source is retained by identity; a later "latest"
source is not substituted while writing, reading, replaying, or evaluating.

## Observed versus research

An observed issuance may use only sources available and retrieved by its
logical cutoff and must be generated before the next regular XNYS open.

A current-vintage research reconstruction may use an immutable source
retrieved later, but:

- every input row is clipped through the historical target/anchor;
- actual retrieval and generation times remain visible;
- the evidence is labelled research/current-vintage;
- it cannot be relabelled observed; and
- its later source does not rewrite an earlier prediction.

`UniverseSnapshot.grade` and `issued_on_time` are separate. An observed
membership does not automatically make a late prediction on time.

## Prediction versions

Each immutable prediction version independently records and proves:

- listing, target, role, horizon, method, config, provider, and benchmark;
- generated time and cutoff;
- source closure and calculation identity; and
- its own on-time status.

A reissue never inherits another version's flag. An unsafe explicit observed
request raises. Aggregate reporting counts the earliest reportable issuance
once per exact observation identity, while later valid reissues remain
available in the ledger.

## Retrospective protocol

The retrospective study uses a fixed 2019-09-03 epoch and horizon-spaced
anchors. Each anchor needs 756 prior returns. Outcomes are partitioned as:

- development: complete before 2024-01-01;
- validation: anchor and complete outcome within calendar 2024;
- final holdout: anchor on/after 2025-01-01 and complete by 2026-09-11.

Intervals crossing partition boundaries are purged. The holdout cannot be
used to tune coefficients, thresholds, drift family, residual policy, or
history window.

The report is a current-universe/current-vintage reconstruction. A
research-grade provider source is allowed, but the report must preserve:

- source run's revision and timestamps;
- replay execution revision;
- report generation time;
- registration availability time; and
- actual source retrieval/availability.

No one field substitutes for another.

## Replay and serving

The CLI replay verifies the source run and exact assets, then performs a
read-only calculation. It writes no database or asset evidence.

Registered performance evidence is produced separately by a service that
calculates the complete selected cohort. HTTP GET reads the registered
artifact and re-verifies source/output identity and physical bytes; it never
runs thousands of trajectories or accepts caller-authored metrics.

## Outcomes

Outcome maturity counts distinct observed sessions, not calendar days. Stock
and SPY use identical endpoints for the momentum decision.

FHS actual losses use recorded realized returns. Simulated paths never become
market observations. A missing endpoint stays unresolved. An all-null advisory
is non-evaluable before price lookup. Corporate-action/basis conflicts remain
explicit rather than being forced into an ordinary return.

## Current market state

`LatestMarketData` advances by market `session_date`; retrieval time only
breaks ties inside the same session. Historical catch-up or an ineligible
series cannot move current state backward.

## Filing vintages retained for archives

SEC facts remain append-only and preserve full instant/duration period
identity, accession, filing/acceptance boundary, source revision, observation
event, Companyfacts asset, and filing-evidence asset. A restatement selects a
new eligible vintage for the same period; it does not create a new comparison
period.

Same-accession corrections use the retrieval/observation that carried the
changed value, not the original filing acceptance. Reused content bytes do not
erase the later observation boundary. Unprovable legacy correction order is
assessed and deferred, never guessed or rewritten.

These facts are not prerequisites for the active price product, but frozen
fundamental archives keep their original point-in-time contracts.

## FX vintages retained for simulation

Every valued date resolves FX against its own end-of-day availability cutoff.
A rate published after the valued date is refused in every grade. A merely
later-retrieved asset is allowed only for explicit research reconstruction.

Rates may be carried over closures only within the configured bound.
Derivation rank is `identity`, `direct`, `inverse`, then `cross:<pivot>`;
same-rank materially disagreeing paths are ambiguous and fail. A converted run
uses one base currency and proves coverage for every accounted date before
calculation.

## Mechanical enforcement

- `AsOfData` centralizes cutoff selection and physical row clipping.
- Registered manifests bind exact source/output identities and checksums.
- Database/model guards reject update/delete of immutable evidence.
- Migration tests reinstall SQLite triggers after table rebuilds.
- Source recovery verifies completed evidence before credentials/quota.
- Missing, stale, ambiguous, unauthorized, or malformed data fails closed.

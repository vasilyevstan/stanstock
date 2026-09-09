# Point-in-time integrity

StanStock distinguishes a value's economic period from when the value became
available to the system.

## Required timestamps

- `period_start` / `period_end`: what interval the value describes.
- `published_at` or `filed_at`: when the source says it was published.
- `available_at`: the earliest defensible time StanStock may use it.
- `retrieved_at`: when StanStock actually obtained the immutable source asset.
  Raw assets are content-addressed, so this is when those exact bytes were
  *first* stored -- not necessarily when a later response repeated them.
- `observed_at`: when a specific retrieval actually carried given bytes,
  recorded as an append-only `SourceObservationEvent` on every retrieval.
- `generated_at`: when an analysis or prediction was created.
- `data_cutoff`: the latest information-availability time permitted for the
  analysis.
- `issued_on_time`: recorded separately on both the analysis run and each
  immutable prediction version. A later reissue never inherits the original
  version's on-time status.
- `target_date`: the logical market date being processed.

A live decision at time `T` may use a source asset only when both
`available_at <= T` and `retrieved_at <= T`.

A later-generated research reconstruction is different: its immutable source
asset may have been retrieved after the historical target, but fact
`available_at` values and price rows are still capped at the target-date
`data_cutoff`. The actual `generated_at`, source `retrieved_at`, and
`UniverseSnapshot.grade=research` remain visible, and the result cannot enter
observed/on-time performance.

Analyses created before the explicit cutoff field existed are migrated
conservatively with `data_cutoff=generated_at`. They are not backdated to their
logical target, because their historical inputs were not proven to have been
clipped at creation time.

## Filing vintages

SEC and ESEF facts are additive. SEC rows retain taxonomy, canonical and source
concept, unit, instant/duration classification, complete start/end period
identity, fiscal metadata, form, filing date, accession, exact acceptance
timestamp, availability basis, source revision, observation hash, and raw
Companyfacts asset. Each SEC fact also has an immutable filing-evidence link to
the exact current-submissions or historical-submissions asset that supplied
its acceptance or filing-date boundary. Quarterly and YTD facts sharing an end
date therefore cannot collide. A changed observation under the same accession
appends another source revision, including a value that reverts to an older
number; a later amendment or restatement never overwrites the value visible to
an earlier decision.

### Acceptance, correction availability, and content reuse

Three distinct clocks apply to one filed observation, and conflating any two
of them creates look-ahead:

1. **Filing acceptance** (`acceptance_at`/`filed_at`) -- when the SEC accepted
   the accession. It describes the *filing* and never changes, not even when
   the numbers inside that accession are later restated.
2. **Correction observation availability** (`available_at` under
   `availability_basis="same_accession_correction_retrieval"`) -- when a
   *corrected* value under an accession StanStock already held first became
   knowable. The original acceptance cannot serve here: nothing before the
   retrieval that carried the corrected number proves that number existed.
   The original revision keeps its acceptance-based availability; only later
   revisions in the chain use this basis, and every such row also carries the
   `same_accession_correction` quality flag.
3. **Content reuse** -- a `DataAsset` is content-addressed, so a response
   identical to an earlier one deliberately reuses the stored asset rather
   than duplicating it. `DataAsset.retrieved_at` then still points at the
   first sighting of those bytes. A value that goes 100 -> 101 -> 100
   restates back to bytes already on file, so the third revision would
   inherit the *first* retrieval time and appear knowable months early.

Every retrieval therefore appends an immutable `SourceObservationEvent`
recording `(provider, kind, subject, content_sha256, observed_at)`. It is
unique on `(provider, kind, subject, observed_at)` -- **without** the digest
-- so the database itself enforces that one observation instant names exactly
one content, and a retried or replayed ingestion records one event rather
than a duplicate. The recorded digest must equal the linked asset's own
`sha256`, checked on write and again before any recovery, so an event can
never certify bytes its asset does not hold. A correction binds its
availability to that event, never to the reused asset's `retrieved_at`.

Binding corrections to observations is **active behaviour in the default
configuration**. The prospective `us-sec-long-v3` reader adds only the
read-time resolution described below; it is not what makes ingestion
correct.

An ordinary **late retrieval is not a correction**. Re-downloading unchanged
content appends no revision at all: the existing fact row is reused, keeps its
original availability, and stays readable at every cutoff it was already
readable at. Only a *changed* observation under an accession already held
creates a new revision, and only those rows take the correction basis.

Rows persisted before this basis existed carry the original acceptance in
`available_at`. They are never rewritten. Instead, a reader may resolve them
conservatively at read time:

- an original observation is proven by its acceptance;
- a legacy correction is admitted only when its own source asset's retrieval
  is a real observation *of that revision*: the asset is not shared with an
  earlier revision, and it was retrieved no earlier than the boundary of the
  revision it supersedes;
- a legacy *reversion*, whose content repeats an earlier revision's and whose
  asset is therefore shared with it, has no distinct retrieval to cite;
- a legacy *decreasing ordering*, whose asset was retrieved before the
  superseded revision became knowable, has a retrieval that cannot be an
  observation of it even though the content differs.

The last two resolve to unknown and are deferred at every cutoff. Chain
ordering does imply a lower bound -- a revision cannot predate the one it
supersedes -- but **a lower bound is not proof**: it says a row was not
knowable *earlier* than some moment, never that it was knowable *by* that
moment. Admitting a row at such a bound would still be a guess, so the bound
is recorded only as assessed context (`earliest_possible_available_at`) and
takes no part in any admission decision.

This resolution is configuration-gated and is applied by the prospective
`us-sec-long-v3` path only. Frozen `us-sec-long-v1`/`us-sec-long-v2` read
recorded availability exactly as released, and the offline audit applies the
same gate so a frozen version is never reported against a policy it does not
have.

### Recovering a retry

An interrupted or repeated ingestion recovers the exact **observed** asset,
identified by its `SourceObservationEvent`, not the asset with the newest
`retrieved_at`. After a reversion those differ: the newest observation
carries the *older* asset (the bytes already on file), while the superseded
payload still holds the newest asset retrieval. Recovering by asset retrieval
would replay stale content and append it as a fresh correction the provider
never sent.

Recovery never infers an order it cannot prove. Three rules keep it honest,
and all three fail closed:

1. **One observation instant names one content.** Recording different bytes
   at a timestamp already recorded for the same provider/kind/subject is
   refused *before* anything is normalized. Two payloads at one instant carry
   no evidence of which came later, and a local commit clock is not evidence
   of provider order -- an A -> B -> A reversion at one timestamp reuses A's
   original event and would leave the superseded B looking newest. Recording
   the *same* retrieval again stays idempotent, so ordinary retries are
   unaffected; a genuinely later retrieval simply carries a later timestamp.
2. **No observation evidence for a corrected chain is not recoverable.** A
   database upgraded from before observation events has assets but no events,
   so the only ordering left is `DataAsset.retrieved_at` -- the wrong clock
   exactly when it matters, since after a reversion the superseded payload
   still holds the newest retrieval. Where such a database already has a
   correction chain, the replay refuses rather than re-appending a superseded
   value as a brand-new, event-bound revision. Where there is no correction
   chain there is nothing to mis-order, and recovery proceeds normally.
3. **Content older than a committed correction is not replayed.** If the only
   recoverable observation predates a correction already on file, replaying
   it would restore the superseded value as a new revision.

None of these is a dead end. A fresh retrieval is separately gated, records
its own observation, and re-establishes proven recovery.

A fresh retrieval also *repairs* what it re-observes. Normally a later
retrieval of unchanged content appends nothing -- the existing fact row is
reused, which is what keeps ordinary late retrieval distinct from a
correction. The one exception is a newest revision that is a legacy
correction with no provable timing: returning it unchanged would discard the
proof that just arrived, leaving every as-of reader deferring that row and
selecting the superseded value forever. So a new revision is appended,
carrying the same economic value and observation hash but an availability
bound to the fresh observation, flagged `reobserved_unproven_correction`. The
unprovable row is never mutated or backdated; it remains deferred and
assessed, and the new vintage stands beside it. Reads before the fresh
observation are unchanged.

Staleness for re-request decisions is likewise measured from the last
observation rather than the stored asset's retrieval, so a reversion does not
report months of staleness that never happened and spend a request rebuilding
evidence already on file.

Companyfacts does not carry acceptance time on each observation. StanStock
joins `accn` to current and historical submissions. Explicit offsets are
preserved; a naive acceptance timestamp is interpreted in
`America/New_York`. If only a filing date exists, availability begins
conservatively at 00:00 New York time on the following local day. An asset
retrieved later remains gated by its actual retrieval timestamp, so historical
facts from that asset are research reconstruction rather than observed
evidence. An as-of read requires both the Companyfacts asset and the linked
filing-evidence asset to have been retrieved by the decision boundary, so
acceptance metadata learned later cannot leak through an older Companyfacts
snapshot.

Instant balance-sheet observations are never subtracted or summed as flows.
Discrete quarters are accepted directly or derived from compatible YTD
durations; a newer restated YTD derivation supersedes a stale direct quarter.
Additive flow TTM values require four exactly adjacent quarters spanning
350-380 days; weighted diluted shares use duration weighting instead of flow
subtraction. Free cash flow is derived only as operating cash flow minus the
absolute compatible capex observation. Debt components remain separate unless
the taxonomy reports a compatible total. Annual history remains separate from
TTM. Current SEC SIC metadata is an immutable retrieval-time observation and
is not backdated across earlier filings.

The deterministic 3y/5y engine reads SEC facts and SIC observations only
through this as-of gate. It filters economic period ends at the forecast
target, requires both the Companyfacts and exact filing-evidence assets to be
visible, and records each target input fact with value, unit, complete period
identity, accession, acceptance/availability metadata, source revision,
Companyfacts asset, and filing asset. Peer calculations retain the
self-contained current price, classification, historical growth, and
multiple, while their inputs are compact immutable references carrying fact
ID, concept, accession, availability time, revision, and both asset IDs. The
referenced fact and evidence rows are immutable and every asset remains in
the prediction's exact source closure. A classification or revised fact
retrieved later cannot change an earlier forecast. The current and peer price
assets must likewise be visible at the decision time and explicitly prove
split-adjusted, dividend-excluded price-return semantics.

Split-adjusted price evidence does not prove that no split occurred after the
latest SEC metric period. Long-v1 therefore reconciles every selected annual
share period to reported diluted EPS, requires TTM diluted shares within 15%
of the latest overlapping annual basis, and stores the remaining
metric-period-to-target interval as `unverified_post_period_split` exposure.
It does not infer or claim a corporate action from adjusted prices.

When a European filing source does not provide the authority's submission
timestamp, StanStock uses the later known repository-added timestamp. It never
backdates availability to the financial period end.

## Price vintages

A later adjusted history can change old values after splits or dividends.
StanStock therefore stores each retrieval as its own asset. A historical
analysis may not substitute a newer asset for the asset actually available at
its decision time. Reconstructed pre-capture analysis is labeled
research-grade.

The US Twelve Data workflow stores the exact raw JSON and a separate normalized
Parquet asset for every retrieval. Current completed-session capture is
`observed`; an explicitly older target is `research`. A later full-history
download creates a new vintage and may not rewrite the assets referenced by an
earlier prediction. Requests use `adjust=splits`, so the evidence supports
split-adjusted price returns only and never silently becomes a
dividend-adjusted total-return series.

An observed run completed overnight but before the next XNYS open records its
actual decision timestamp as `data_cutoff`, even when that UTC timestamp falls
on the following calendar date. Every source asset in that run must have been
available and retrieved by that cutoff. Research reconstructions remain capped
at the historical target-date boundary.

Asset-level eligibility is not enough: an eligible Parquet bundle can still
contain rows after the requested market date. `AsOfData.price_frame` requires a
parseable `date` column, normalizes it to `Date`, and physically filters it to
`date <= through_date`, rejects a cutoff after its as-of decision, and returns
rows in ascending date order. Missing or ambiguous date schemas fail
explicitly.

## FX vintages

An FX rate is read like any other point-in-time value: only vintages whose
`available_at` (and whose source asset's `available_at`/`retrieved_at`)
precede the decision boundary are eligible. Conversion then adds three rules
a single price read does not need.

**Every valued date carries its own cutoff.** A run-wide decision boundary
alone would let a correction published in February change how an execution in
January was priced. Each valued date `D` is therefore resolved against the end
of `D`: a value dated `D` may only be converted with information that existed
by then. A February 9 correction of a February 3 observation applies from
February 9 onward -- including to later dates that carry that observation
forward -- but can never rewrite February 3 itself.

**Late publication and later retrieval are different.** A rate *published*
after the valued date describes information nobody held then and is refused
for every run, regardless of grade. A source asset *retrieved* after the
valued date is the ordinary research-reconstruction case: an explicitly
`research`-grade run may read a file StanStock fetched later, exactly as it
may for filing facts and price frames, while an `observed`-grade run requires
the vintage and its asset to have been available and retrieved by the end of
the valued date. `FxEvidenceGrade` has no default; the caller states which
claim it is making.

**Explicit carry, never silent staleness.** FX series have no weekend,
holiday, or (for the synthetic demo bundles) non-Friday observations. The
most recent eligible observation is carried forward and the carry distance is
stored per converted date. The reviewed maximum is 7 calendar days; a run may
tighten it to as little as 0, and may not widen it. A carry beyond the limit
fails instead of pricing from a stale rate, and coverage is proven for every
accounted date before any value is computed, so a date the series cannot
reach fails the run rather than borrowing a neighbouring day's conversion.

**End-of-day resolution bounds execution.** The cutoff is a whole day because
FX vintages record no intraday knowability. A converted run is therefore
restricted to close-based execution; `next_open` and `next_eligible` are
rejected rather than converting an opening trade at a rate that may have been
published after the open.

Providers publish a subset of the pairs a portfolio needs -- ECB quotes are
EUR-based -- so derivations are ranked `identity` > `direct` > `inverse` >
`cross:<pivot>` and the chosen path is recorded with each rate. Two
derivations of the *same* rank that disagree materially are ambiguous and
fail rather than being resolved silently. Missing pairs and over-stale
observations fail the whole run; a partially-converted panel is never
produced.

Converted runs persist the resolved FX frame -- value date, currencies, rate,
observation date used, carry distance, derivation path, the vintage's
publication and availability times, the per-row availability cutoff, the
evidence grade it was admitted under, and source asset IDs -- as its own
immutable asset, and the price input keeps each row's native price and
currency beside the converted value.

## Universe grades

- `observed`: membership was captured at that time.
- `research`: membership was reconstructed later.

The grade follows analyses and simulations. Research-grade history cannot be
described as an observed live track record.

## Missing and difficult observations

Missing values remain missing and reduce confidence. Acquisitions, delistings,
bankruptcies, suspensions, missing terminal prices, and unresolved currency
dates remain in coverage denominators with explicit outcome statuses.
If an evaluation vintage changes the target-date price basis, the evaluator
records a `corporate_event` outcome instead of treating the discontinuity as a
normal investment return.

## Mechanical enforcement

- `AsOfData` centralizes available/retrieved-time filters.
- Architecture tests prohibit research and simulation code from importing raw
  provider modules.
- Database constraints enforce timestamp ordering, positive source prices,
  bounded returns, and valid score/probability/outcome states.
- Prediction, data-asset, filing-fact, filing-evidence-link, and FX-vintage
  updates and deletes fail at both the Django and database layers.
- Tests prove that a later source vintage cannot change an earlier as-of read.
- SQLite migrations that rebuild a protected table must reinstall its
  immutability triggers afterward; this is covered by mutation tests.

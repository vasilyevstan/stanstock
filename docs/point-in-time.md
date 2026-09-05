# Point-in-time integrity

StanStock distinguishes a value's economic period from when the value became
available to the system.

## Required timestamps

- `period_start` / `period_end`: what interval the value describes.
- `published_at` or `filed_at`: when the source says it was published.
- `available_at`: the earliest defensible time StanStock may use it.
- `retrieved_at`: when StanStock actually obtained the immutable source asset.
- `generated_at`: when an analysis or prediction was created.
- `data_cutoff`: the latest information-availability time permitted for the
  analysis.
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

SEC and ESEF facts are additive. Accessions, source concepts, units, reporting
periods, filing dates, and availability remain attached to each row. A later
amendment or restatement creates another fact; it does not overwrite the value
that was visible to an earlier decision.

When a European filing source does not provide the authority's submission
timestamp, StanStock uses the later known repository-added timestamp. It never
backdates availability to the financial period end.

## Price vintages

A later adjusted history can change old values after splits or dividends.
StanStock therefore stores each retrieval as its own asset. A historical
analysis may not substitute a newer asset for the asset actually available at
its decision time. Reconstructed pre-capture analysis is labeled
research-grade.

Asset-level eligibility is not enough: an eligible Parquet bundle can still
contain rows after the requested market date. `AsOfData.price_frame` requires a
parseable `date` column, normalizes it to `Date`, and physically filters it to
`date <= through_date`, rejects a cutoff after its as-of decision, and returns
rows in ascending date order. Missing or ambiguous date schemas fail
explicitly.

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
- Prediction, data-asset, filing-fact, and FX-vintage updates and deletes fail
  at both the Django and database layers.
- Tests prove that a later source vintage cannot change an earlier as-of read.
- SQLite migrations that rebuild a protected table must reinstall its
  immutability triggers afterward; this is covered by mutation tests.

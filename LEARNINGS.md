# StanStock learnings

Durable, verified lessons about this codebase go here. This file is not a
running commentary or a task log — only record a lesson once it has been
verified (a passing test, a reviewed decision, or a repeated pattern), and
keep entries short.

## How to add an entry

- Add new lessons under "Verified lessons", newest first.
- State the concrete fact, the file/pattern it applies to, and why it
  matters. Avoid restating something already obvious from reading the code.
- Do not record implementation details that have not actually been built and
  verified yet; speculative or planned behavior belongs in the plan/handoff,
  not here.
- Remove or correct an entry once it is no longer accurate.

## Established decisions (from the approved plan)

These are the foundational decisions already accepted for this project, not
implementation lessons. They constrain future architecture/simplifier
reviews and should not be re-litigated without an explicit new decision.

- **Local-first Django monolith**: StanStock is one Django modular monolith,
  not a microservice split; it must remain runnable locally via `docker
  compose up --build` without a live external dependency in its default
  (demo) mode.
- **PostgreSQL + `DATA_DIR`/Parquet**: relational and transactional data
  lives in PostgreSQL; bulk time-series/asset data (price history, etc.)
  lives as content-addressed Parquet/binary files under
  `STANSTOCK_DATA_DIR`, tracked via `DataAsset` rows.
- **Polars + NumPy**: data-frame and numerical work uses Polars and NumPy;
  no second query engine (e.g. DuckDB) or heavy data-processing framework
  without an accepted need.
- **Rules-only v1**: scoring, recommendation, risk, and scenario logic is
  rules-based and explainable in v1, not a fitted/trained ML model.
- **Provider stop/go gate**: no live provider integration activates until its
  source capability and licensing/terms review has explicitly passed;
  `ProviderRecord.enabled` and terms fields gate this.
- **Immutable predictions and as-of controls**: `Prediction` rows (and
  provider-derived vintage data) are permanent once written and are read
  through explicit `available_at <= decision_time` point-in-time controls,
  never a provider's "latest" state, for any historical decision.
- **Six-agent tiered workflow**: `stanstock-architect`,
  `stanstock-simplifier`, `stanstock-developer`,
  `stanstock-research-integrity`, `stanstock-critic-tester`, and
  `stanstock-final-validator` form the reusable review chain described in
  `.github/agents/README.md`, applied by change tier (ordinary, data/quant,
  or material).

## Verified lessons

- **Backtest signal time and information time are separate.** Research-grade
  reconstructions may be generated later, but their fact availability and
  price rows are capped at the historical `AnalysisRun.data_cutoff`; an
  observed-grade backtest rejects a signal not generated on its target date.
- **Never backfill provenance with a stronger claim than the old code
  proved.** Legacy analyses without an explicit cutoff use `generated_at`;
  assigning their target date would falsely certify historical input
  filtering.
- **Never aggregate native currencies without conversion.** Until dated FX
  conversion is implemented, simulation builders must restrict each run to
  one listing currency and persist that currency with the run.
- **A reproducibility hash must cover content, not summaries.** Simulation
  input hashes include canonical full-frame contents and the explicit
  calendar; shape and aggregate sums can collide for materially different
  paths.
- **Restatements replace a period, not create a growth period.** Fundamental
  normalization maps source concepts to canonical names and selects the latest
  eligible vintage per concept/period before comparing distinct periods.
- **Terminal outcomes need serialized evaluation.** Evaluators lock and
  freshly read the prediction/outcome pair so stale ORM relation caches cannot
  downgrade a matured or corporate-event result.
- **Image build contexts are a secret boundary.** `.env` files must be
  excluded by `.dockerignore`, not merely `.gitignore`.
- **Selected portfolios need a common inception.** If any requested holding
  lacks its execution-basis price on the first trade date, fail explicitly
  instead of reallocating its intended capital to the remaining holdings.
- **Read-only containers need a dedicated backup mount.** Keep backup bundles
  outside `DATA_DIR`, configure `STANSTOCK_BACKUP_DIR`, and mount that path
  writable in production.
- **An eligible asset can still leak future rows.** `AsOfData.price_frame`
  must enforce both asset availability and `date <= through_date`, normalize
  and sort the returned date column, reject cutoffs after the as-of decision,
  and reject missing or ambiguous date schemas.
- **Immutable paths need a physical conflict check.** `AssetStore` may reuse
  an existing path only when its checksum matches; different bytes at the same
  path are an explicit error.
- **SQLite table rebuilds can remove custom triggers.** Any migration after an
  immutability-trigger migration must be checked for table recreation and, if
  needed, followed by a trigger-reinstallation migration plus a bulk-update/
  delete regression.
- **The free unattended price path is currently `NO_GO`.** Stooq's browser
  gate is not bypassed, and the reviewed free API tiers do not cover the
  requested US/European breadth. Default product flows use visibly labeled
  `synthetic_demo` data.
- **Synthetic refreshes use observed synthetic sessions.** `refresh_demo`
  defaults to the research snapshot's `as_of_date`, rejects later or
  non-session target dates, and relies on `JobRun` to skip a repeated
  successful target.
- **Outcome maturity is session-based.** The evaluator uses 10/252/756
  distinct observed sessions, leaves insufficient cases unresolved, and
  excludes research-grade or late-generated outcomes from live-performance
  aggregates.
- **Simulation identity and inputs are durable.** Trades and holdings persist
  listing UUIDs, and every completed run records immutable price, signal, and
  benchmark input assets alongside its result curve.
- **Local Docker capacity is not repository correctness.** When shared Docker
  Desktop storage is exhausted, validate Compose configuration locally and
  rely on the clean GitHub Actions image build; never prune unrelated shared
  images or volumes.

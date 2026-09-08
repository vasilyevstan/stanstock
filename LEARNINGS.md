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

- **`manage.py analyze` is research-only for every target.** The command
  explicitly requests `issued_on_time=False`; provider/config flags and
  `STANSTOCK_CODE_REVISION` cannot make it an observed issuance path. An
  exceptional observed reissue is a direct, independently qualified service
  call that binds the production config/provider/benchmark and exact committed
  revision; an unsafe request raises. See `docs/operations.md`.
- **One market observation counts once in aggregate performance.** Within an
  exact listing/target/horizon/evidence-role/method/config/provider cohort, the
  earliest reportable issuance is canonical. Later valid observed reissues
  remain immutable and are evaluated and shown in the ledger, but they neither
  replace nor recount the original observation.
- **Failed share-basis evidence is assessed, not verified.** A methodology
  gate that disqualifies a prediction (e.g. `us-sec-long-v2`'s adjacent
  annual diluted-share continuity check) records `assessment_status`/
  `assessed_through` and the disqualifying source facts, never
  `verified_through` or a claimed corporate action; the UI and payload never
  imply a split was confirmed.
- **An all-null advisory scenario is non-evaluable, not a decision signal.**
  `evaluate_prediction` returns an explicit unresolved outcome before any
  price lookup when `bear_return`/`base_return`/`bull_return` are all null,
  and reporting defensively excludes such rows from advisory denominators
  even if a malformed legacy row somehow matured. Decision BUY/AVOID/HOLD
  success semantics are a separate contract and are unaffected.
- **A frozen methodology contract covers config bytes and every payload
  shape.** The effective config/hash, eligibility, reason wording, and both
  successful and withheld calculation/scenario payloads are all part of a
  frozen version's contract (e.g. `us-sec-long-v1`). A stricter eligibility
  gate needs a new version (`us-sec-long-v2`) plus differential base/head
  reproduction tests proving the frozen version's hash, behavior, and
  payloads are byte-for-byte unchanged.
- **On-time status is proven independently by each immutable prediction
  version; no reissue inherits another version's status.** The original
  issuance and any later reissue are each checked against their own
  next-market-session-open deadline from cutoff-safe evidence instead of
  trusting a stored flag. A same-target reissue may still be observed-grade
  while it is issued before that deadline from cutoff-safe evidence. An
  explicit observed request after the deadline or against unsafe evidence
  raises; any later non-observed reconstruction is separately research-grade.
- **Unsupported horizons must not enter the evidence ledger.** A price-only
  model can display explicit withheld scenarios, but it persists predictions
  and horizon scores only for configured supported horizons so outcomes and
  performance denominators cannot absorb unsupported calls.
- **Current market state is ordered by market session, not retrieval time.**
  `LatestMarketData.session_date` is the primary freshness key and retrieval
  time only breaks same-session ties; historical catch-up and ineligible
  series cannot replace a newer eligible close.
- **Committed-work recovery precedes credentials and retry-time labels.** A
  retry searches completed universe/target/scoring evidence across snapshot
  grades before provider enablement, API-key resolution, or quota use, and
  fails explicitly if completed runs conflict.
- **A revalued holding must be revalued everywhere.** Restating a closed
  foreign holding only at end-of-day still lets a rebalance size its targets
  off the frozen conversion. Valuation and pre-trade sizing use the same
  carried-native-at-today's-rate figure; the closed holding stays untradable.
- **Withholding a derived figure is not enough when the primary number is
  wrong.** A date with no eligible FX rate cannot be rescued by suppressing
  the attribution -- the reported return is already wrong -- so coverage is
  proven for every accounted date before any accounting begins.
- **Model precision bounds what may be executed.** FX vintages record no
  intraday knowability, so a converted run executes on closes only;
  opening-price bases are rejected rather than pretending to a cutoff the
  data cannot support.
- **A decision boundary is not a per-date cutoff.** Resolving a whole
  historical panel against one run-wide boundary lets a later correction
  reprice an earlier execution. Each valued date is resolved against its own
  end-of-day cutoff; a rate published after that date is refused in every
  grade, while a merely later-*retrieved* source asset is accepted only for an
  explicitly research-grade reconstruction.
- **A carried foreign quote must not carry its exchange rate.** Retaining an
  already-converted price through a foreign-market holiday silently pins the
  holding's FX to the last session its market was open. Carry the native quote
  and revalue it at the current eligible rate, or withhold.
- **Hash what the accounting reads, not only what it prints.** Native currency
  assignment and retained native prices are dropped during normalization yet
  decide conversion and attribution, so they are hashed separately; a run that
  converts nothing adds no FX terms and keeps its pre-FX identity.
- **`or` is not a null check for numeric options.** `options.get(x) or
  DEFAULT` silently restores the default for a deliberate `0`. Bounded
  safety controls must test for `None` and reject values outside their
  reviewed range at every entry point.
- **Point-in-time FX needs three rules, not one.** Availability at the
  decision boundary is necessary but not sufficient: the observation must
  also be dated on or before the value's own date, carried forward across
  closures only within a bounded, recorded window, and derived through a
  ranked, named path (`identity` > `direct` > `inverse` > `cross:<pivot>`).
  Missing, over-stale, and same-rank-disagreeing paths fail the run.
- **A currency split must be exact or withheld.** The stock-versus-FX
  attribution restates the *same* quantity path at each currency's inception
  rate, so the two legs sum to the reported return by construction. When a
  cash settlement or missing reference rate breaks that identity, both figures
  are withheld with a reason instead of being approximated.
- **Backtest signal time and information time are separate.** Research-grade
  reconstructions may be generated later, but their fact availability and
  price rows are capped at the historical `AnalysisRun.data_cutoff`; an
  observed-grade backtest rejects a signal not generated on its target date.
- **Never backfill provenance with a stronger claim than the old code
  proved.** Legacy analyses without an explicit cutoff use `generated_at`;
  assigning their target date would falsely certify historical input
  filtering.
- **Never aggregate native currencies without conversion.** Simulation
  builders convert every native price into one explicit base currency through
  dated point-in-time rates, persist the native price and applied rate beside
  each converted value, and record the base currency with the run.
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
- **`select_for_update()` can lock ordering joins.** A model's default ordering
  can add joined tables to a locking query even when the caller only intends to
  lock local rows. Use `of=("self",)` with an explicit stable row order before
  acquiring shared analysis or market locks. Use `no_key=True` for shared
  reference rows that must remain compatible with foreign-key `KEY SHARE`
  checks, and exercise the complete sequence with independent PostgreSQL
  connections.
- **Escape PostgreSQL `%` placeholders in migration SQL.** PL/pgSQL format
  markers passed through Django's psycopg schema editor must be written as
  `%%`; otherwise a fresh PostgreSQL migration fails before the trigger
  function is created. Cover the generated SQL and a real forward/reverse
  migration cycle.
- **Provider access and display rights are separate gates.** Twelve Data can
  technically support the bounded US universe, but its Basic tier is labeled
  internal non-display. Basic is enabled only after explicit personal,
  non-commercial authorization and is bound to exactly one active user;
  broader display requires a display-entitled plan or agreement. Stooq remains
  `NO_GO`, Europe remains deferred, and default flows stay visibly
  `synthetic_demo`.
- **Large provider catalogs can contain irrelevant malformed rows.** Preserve
  the complete raw response, but normalize only the symbols in the reviewed
  universe and continue to fail closed when any configured symbol is missing,
  malformed, ambiguous, or outside the licensed plan.
- **Live adjusted histories are immutable vintages, not mutable truth.**
  Twelve Data responses are stored as exact raw JSON plus normalized Parquet;
  every later retrieval creates new evidence, requests explicitly use
  split-only adjustment, and derived results are labeled price return rather
  than total return.
- **Twelve Data's daily `end_date` is exclusive.** Request the following
  calendar date, but keep validating parsed bars against StanStock's original
  inclusive cutoff so no future session can enter an analysis.
- **One immutable series can serve benchmark and investable identities
  without duplicate acquisition.** SPY is fetched once, then its checksummed
  benchmark asset advances a separate ETF listing and current market row at
  zero extra credits. Explicit security-type guards keep that ETF out of stock
  universe membership, scoring, predictions, opportunities, and sample
  baskets while still allowing portfolio valuation without a fabricated stock
  analysis. Recovery follows the exact asset UUID/checksum recorded by the
  completed analysis, metrics stop at the displayed market session, and
  missing split-only/dividend provenance fails closed. ETF projection happens
  after stock research commits, allowing an identity conflict to be repaired
  and retried from immutable evidence without sacrificing the on-time
  analysis or spending provider credits again.
- **Synthetic refreshes use observed synthetic sessions.** `refresh_demo`
  defaults to the research snapshot's `as_of_date`, rejects later or
  non-session target dates, and relies on `JobRun` to skip a repeated
  successful target.
- **Outcome maturity is session-based.** The evaluator uses 10/252/756
  distinct observed sessions, leaves insufficient cases unresolved, and
  excludes research-grade or late-generated outcomes from live-performance
  aggregates.
- **Simulation identity and inputs are durable.** Trades and holdings persist
  listing UUIDs, and every completed run records immutable price, signal,
  benchmark, and (when converted) FX input assets alongside its result curve.
- **Cash flows and allocation decisions need their own immutable evidence.**
  Keep deposits, confirmed purchases, and manual performance baselines
  separate from mutable portfolio state. A preview is not an event, and a
  recorded purchase is local bookkeeping at a persisted close, not a broker
  fill.
- **A confirmation hash must bind qualification, not only execution price.**
  Include the exact analysis/run/configuration/code revision, eligibility
  criteria, source asset UUID/checksum, price session, holdings, settings, and
  cash. Recompute under one portfolio-first lock order and reject any changed
  state.
- **Unexplained quantity changes are boundaries, not returns.** A supported
  manual quantity change appends an immutable post-change baseline. If the
  whole portfolio cannot be valued, preserve an explicit unavailable-boundary
  record so the edit remains recoverable while performance stays withheld.
  Repeated snapshots must carry unresolved split warnings until quantity is
  explicitly corrected.
- **Django admin saves are concurrency-sensitive writes.** Read-only form
  fields do not stop `Model.save()` from writing stale values. Ledger-managed
  models require portfolio-first locking and explicit field allow-lists in
  web and admin paths, plus regressions that interleave deposits or purchases.
- **Local Docker capacity is not repository correctness.** When shared Docker
  Desktop storage is exhausted, validate Compose configuration locally and
  rely on the clean GitHub Actions image build; never prune unrelated shared
  images or volumes.

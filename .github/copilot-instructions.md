# StanStock repository instructions

StanStock is a public, local-first Django/Python financial-research
application. Read the current tree, status, and history before acting; do
not rely on stale conversation state or a branch name.

## Start here

Every task begins by reading, in order:

1. The current git branch, status, recent history, and exact diff (`git
   status`, `git log`, `git diff`) — never assume prior turns still describe
   reality.
2. Any approved plan supplied for this task.
3. [`LEARNINGS.md`](../LEARNINGS.md) — durable, verified lessons and the
   decisions already established for this project.
4. [`agents/README.md`](agents/README.md) — the tiered agent chain, handoff
   fields, and status vocabulary.
5. The relevant checklist(s) under
   [`../docs/review-checklists/`](../docs/review-checklists/) for the change
   type (research-integrity, security, ux, contracts, operations).
6. [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for workflow and architecture
   rules.

## Non-negotiable rules

- Preserve unrelated tracked, staged, and untracked work; never discard a
  user's in-progress changes.
- Keep the architecture lean: a Django modular monolith with PostgreSQL plus
  `STANSTOCK_DATA_DIR`/Parquet asset storage, Polars/NumPy for data work, and
  a rules-based (not ML-fitted) v1 methodology.
- Never add Django REST Framework, DuckDB, a Node build chain, Celery, a
  distributed queue/scheduler service, or a fitted ML model without an
  explicit accepted need and a simplification review; the default is to stay
  within the existing stack.
- Never use live provider data, real provider credentials, or a real network
  fetch in tests or CI. Use synthetic fixtures only.
- Treat unattended free US/European OHLCV as `NO_GO` until new primary-source
  evidence approves a provider. The built-in demo provider is named
  `synthetic_demo` and must remain visibly synthetic.
- Never expose secrets (`.env` values, `DJANGO_SECRET_KEY`, `DATABASE_URL`,
  provider API keys), private financial/portfolio data, or local
  session/workspace paths in code, commits, logs, or agent output.
- Enforce the tiered agent chain from `agents/README.md`: ordinary changes go
  developer -> critic-tester; data/quantitative changes go developer ->
  research-integrity -> critic-tester; material architecture/schema/
  security/scoring/methodology changes go architect -> three-pass simplifier
  + synthesis -> developer -> research-integrity -> critic-tester ->
  final-validator; the final validator also runs at milestone/release
  boundaries.
- Maintain local-first behavior: the app must remain runnable with `docker
  compose up --build` and must not require a live external provider or cloud
  service to function in its default (demo) mode.
- Preserve permanent IDs and immutable prediction/source vintages; never
  mutate or delete a persisted `Prediction`, `DataAsset`, `FundamentalFact`,
  or `FxRate` row.
- Never let a historical read see data whose `available_at` is after the
  relevant decision time, or a price row after its requested market date;
  never substitute a provider's "latest" state for a past decision.
- Preserve both `AnalysisRun.generated_at` and `AnalysisRun.data_cutoff`.
  Research reconstructions may be generated later, but fact availability and
  price rows must remain capped at the historical cutoff; observed backtests
  accept only on-time signals.
- Never combine native-currency prices in one simulation cash balance. Until
  dated FX conversion exists, require one explicit/inferred native currency
  per run and reject mixed-currency selections.
- Resolve the latest eligible fundamental vintage per canonical concept and
  reporting period before calculating growth; an amendment is not a new
  comparison period.
- Reproducibility hashes must cover canonical complete inputs and calendars,
  not shapes, counts, or aggregate sums.
- Migration backfills must preserve only provenance the old schema can prove;
  legacy analyses without an explicit cutoff use `generated_at`, never an
  inferred historical target.
- A selected portfolio must have a usable execution price for every listing
  on one common inception date; never silently redistribute an unavailable
  holding's allocation.
- A migration that rebuilds an immutable SQLite table can drop its triggers;
  reapply and retest database-level update/delete protection after later
  constraint migrations.
- Represent missing or insufficient data explicitly; never coerce it to zero
  or a success-shaped default.

## Pull requests

Use `.github/pull_request_template.md`. Every PR records base/head evidence,
data/methodology impact, security/operations impact, and which agents in the
chain reviewed it. Only the orchestrator (human, or an explicitly designated
orchestrator role) performs git actions (stage, commit, push, open/merge a
PR, dispatch a workflow); reusable agents defined under `agents/` never do.

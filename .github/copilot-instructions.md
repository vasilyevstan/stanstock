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
- Treat broad unattended US/European OHLCV as `NO_GO`. The only approved live
  price path is the conditional US Twelve Data workflow: it stays disabled
  without a non-demo key and explicit internal-display entitlement, uses the
  curated bounded universe, preserves split-adjusted price-return labeling,
  and never redistributes provider data. The built-in demo provider is named
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
- The orchestrator owns progress through that chain and must detect stalls
  rather than silently wait. Treat an idle/completed agent with unread output,
  an agent that stops making tool progress, a repeated unresolved finding, a
  stale worktree fingerprint, pending CI/deployment gates, or a dirty
  worktree approaching the next scheduled refresh as actionable. Read the
  result, return the finding to its owner, fall back to direct work or a
  replacement only when the owner is unavailable, and keep going until the
  requested result is released and locally verified.
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
  accept only calendar-proven on-time signals. Preserve
  `Prediction.issued_on_time` separately: no reissue inherits another
  version's on-time status. Each immutable version, including a same-target
  reissue, independently proves its own next-market-session-open deadline
  from cutoff-safe evidence; a reissue created before that deadline may still
  be observed. An unsafe explicit observed request fails closed; a separate
  non-observed reconstruction is research-grade. Aggregate performance counts
  the earliest reportable prediction once per exact listing/target/horizon/
  evidence-role/method/config/provider observation, while later valid observed
  reissues remain immutable ledger rows evaluated per version.
- Persist predictions and reported horizon scores only for a scoring
  configuration's `supported_horizons`; an explicitly withheld scenario must
  not enter outcome or performance denominators. An advisory prediction whose
  scenario returns are all null is non-evaluable: evaluation resolves it as
  unresolved before any price lookup, and reporting excludes it from advisory
  denominators; this is independent of decision BUY/AVOID/HOLD semantics.
- A frozen methodology/config version's contract covers its config bytes/
  effective hash, eligibility, reason wording, and both successful and
  withheld calculation/scenario payloads. A stricter eligibility gate is a
  new version, proven with differential base/head reproduction tests showing
  the frozen version's hash, behavior, and payloads are unchanged. Evidence
  that disqualifies a prediction uses `assessed_through`/an explicit
  incompatible-or-unverified status and cites the disqualifying source facts,
  never `verified_through` or a claimed corporate action.
- `manage.py analyze` is demo/research tooling, never the live US or
  observed-reissue interface: it defaults to the generic demo provider, the
  default scoring config, and `code_revision()`'s `"working-tree"` fallback,
  and explicitly passes `issued_on_time=False` for every target. Provider,
  config, benchmark, or revision flags cannot make the command observed. Only
  an exceptional direct `analyze_snapshot(..., issued_on_time=True, ...)`
  service-level call can request an observed same-target reissue, after it
  independently reproves the next-session deadline and cutoff safety; that
  call must explicitly bind the reviewed production scoring config,
  `provider="twelve_data"`, the reviewed benchmark (currently SPY), and the
  exact committed `STANSTOCK_CODE_REVISION`. An unsafe explicit request raises
  rather than becoming observed or silently downgrading.
- Update `LatestMarketData` only through the shared monotonic market-state
  writer. Compare market `session_date` first and retrieval time only within
  the same session; historical or ineligible series must not move current
  state backward.
- Recover already committed target work across retry-time snapshot grades
  before checking provider enablement, resolving credentials, or spending
  quota; conflicting completed runs are an explicit integrity error.
- Never combine native-currency prices in one simulation cash balance. Convert
  through the dated point-in-time FX path into one explicit base currency, or
  restrict the run to a single native currency. Resolve every valued date
  against its own availability cutoff so a later correction cannot reprice an
  earlier date, refuse a rate published after the valued date in any grade,
  allow a merely later-retrieved asset only for explicitly research-grade
  reconstruction, and fail on a missing, over-stale, or ambiguous rate.
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

Use specific, outcome-focused commit subjects and pull-request titles. Avoid
vague labels such as "updates", "changes", or "fixes".

Use `.github/pull_request_template.md` and complete every applicable section
with enough detail to explain what changed, why it was needed, and how it
works. Every PR records exact base/head and validation evidence,
data/methodology and compatibility impact, security/operations impact,
risks/rollback, and the exact agent-chain results. Use
`N/A: <factual reason>` instead of deleting an inapplicable field. Keep the
description current as findings are resolved, and correct stale hashes,
contract revisions, CI results, and review statuses before merge.

Only the orchestrator (human, or an explicitly designated orchestrator role)
performs git actions (stage, commit, push, open/merge a PR, dispatch a
workflow); reusable agents defined under `agents/` never do.

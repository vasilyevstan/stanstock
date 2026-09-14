# Testing and Independent Review

> Release state: see [README](../README.md#release-status). No integrated CI,
> private acceptance, real study, or release approval is claimed here.

Tests and CI use synthetic fixtures only. They never use live provider
credentials, real provider calls, or private financial data.

## Repository quality gate

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run python manage.py makemigrations --check --dry-run
uv run python manage.py check --settings=stanstock.settings.test
```

Production settings and PostgreSQL migration/concurrency checks run separately
with non-secret test values. A copied test count is not release evidence; the
exact command result must bind to the final revision.

Documentation-only edits need no build, but these drafts still require a
separate multi-model clarity/privacy review before publication.

## Product correctness coverage

Release validation must cover:

- exact 757-close stock/SPY alignment and target-date closure;
- 12–1 momentum endpoints and mixed/equality behavior;
- filter variance indexing, 252-return burn-in, residual centering/scaling,
  and innovation-only future recursion;
- cumulative `expm1(sum(log returns))` projections;
- deterministic PCG64 seed, path-major prefix, and linear quantiles;
- whole-triplet withholding for invalid/non-representable results;
- price/volume split-equivalence and fixed-volume counterexample;
- compatible-volume absence blocking BUY without blocking projections;
- Under-$10 research with 0% allocation/no promotion;
- exact one-analysis/five-prediction multiset;
- null probability/confidence/score shape;
- no SEC prerequisite; and
- frozen historical-method differential reproduction.

## Intake, provider, and recovery coverage

- capture owner/saved/entitlement intent before provider work;
- at most 100 core plus 20 saved names, deduplicated by permanent identity;
- SPY reused once and excluded from stock membership;
- catalog/plan/security/currency/MIC validation;
- reuse adequate registered history before credentials/quota;
- short monitoring history never shadows qualified full history;
- explicit pending/rejected/failed/insufficient admission states;
- retries preserve captured membership;
- completed target and recorded-parent recovery before provider checks;
- conflicting completed runs fail;
- no partial output served after failure; and
- no provider call from ordinary authenticated GET.

## Reader and UI coverage

The native end-to-end contract must persist real synthetic evidence, call the
production writer, read through the fail-closed product reader, and render
authenticated Opportunities and stock detail.

Adversarial cases include:

- wrong owner/provider/catalog identity;
- changed/missing physical bytes;
- future row/source;
- missing or extra prediction;
- wrong method/role/horizon;
- forged self-consistent calculation JSON without independent evidence;
- stale target;
- inaccessible display authorization;
- unavailable projection/risk reasons;
- responsive/keyboard-readable tables; and
- separate observed, retrospective, archive, and synthetic labels.

`/status` must remain operational rather than duplicate forecast cards.

## Scheduled and observed issuance coverage

- 03:30 local Tuesday-Saturday schedule;
- regular and early XNYS closes;
- Europe/Tallinn and New York DST transitions;
- holidays, sleep/wake, duplicate targets, and timezone changes;
- exact clean 40-hex revision on fresh observed work;
- on-time proof before commit;
- unsafe explicit observed request raises;
- no inheritance of on-time status between reissues;
- scheduled web/profile/config alignment;
- independent market/evaluation/portfolio child recovery; and
- credential-free, zero-fetch recovery of completed work.

The manual daily and generic analyze commands must remain research-only.

## Retrospective study coverage

- fixed 2019-09-03 epoch;
- 756 prior returns;
- horizon-spaced anchors;
- development/validation/holdout boundaries and crossing purge;
- no holdout tuning;
- FHS and two Gaussian models on identical paired support;
- equal target-cohort means;
- realized-return MAE, pinball, inclusion, width, and interval score;
- empty/worse/unavailable partitions retained;
- 16,384-path diagnostic explicitly non-financial;
- CLI performs no provider/credential/database/asset writes;
- exclusive mode-0600 output file outside `DATA_DIR`;
- registration calculates all selected listings itself;
- no caller-authored metrics and no GET replay;
- source/replay/report/registration provenance remains separate; and
- synthetic study results never treated as skill.

## Immutable and point-in-time coverage

- future prices/facts excluded from historical decisions;
- exact source asset retained, not replaced by latest;
- source availability/retrieval, cutoff, generation, and registration clocks
  remain distinct;
- prediction/data-asset/fact/FX immutability on SQLite and PostgreSQL;
- SQLite trigger reinstallation after table rebuilds;
- canonical complete input/calendar hashes;
- monotonic current-market session updates; and
- all-null advisory non-evaluable before price lookup.

## Independent chain

The material redesign follows:

```text
architect
  -> three sealed simplifier passes + synthesis
  -> developer
  -> research-integrity
  -> critic-tester
  -> final-validator
```

The architect and simplifier gates define/simplify the contract; they are not
implementation or release approvals. The developer cannot approve its own
work. Research integrity owns provider/as-of/methodology/outcome review.
Critic-tester owns adversarial security/UX/contracts/operations review and
test execution. Final-validator owns complete release acceptance.

The orchestrator binds every handoff to exact base/head revisions, returns
findings to their owning agent, and responds to stalls: unread completed
output, no tool progress, repeated unresolved findings, stale fingerprints,
pending CI/deployment gates, or a dirty runtime approaching refresh. It may
replace an owner only when that owner is unavailable or failed. Only the
orchestrator performs git/release/deployment actions.

## Release acceptance

Each release records the following evidence at its exact head. These are
standing requirements, not a live list of unfinished work; consult the
[release record](../README.md#release-status) for the current result:

- reviewed integration and focused validation;
- independent integrity and critic approvals;
- complete exact-head CI/regression results;
- separate documentation clarity/privacy review;
- private fixed-denominator operational acceptance;
- real registered retrospective evidence without hidden failures; and
- final-validator release acceptance.

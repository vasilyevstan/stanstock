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

CI partitions the complete pytest collection into price-product modules
(`test_research_product_*.py` and `test_research_price_product*.py`), the
expensive `test_research_product_study_evidence.py` subset, and the non-product
complement. Each test belongs to exactly one partition. All retain the
40-minute execution limit; none drops tests or changes assertions. The
required `quality` check fails unless all partitions, including their checks,
succeed. Coverage shown by each partition is partial; `make check` remains
the complete local run. A failed partition
stops at its first failure so its traceback is available without waiting for
the job timeout. PostgreSQL integrity and container checks remain separate.

Responsive browser cases require the Chromium binary matching the locked
Playwright package. CI installs it with
`uv run playwright install --with-deps chromium`; for a local environment
with system libraries already available, use `uv run playwright install chromium`.
Browser-dependent acceptance must run, not be reported as passing when its
browser is absent.

Documentation-only edits need no build, but these drafts still require a
separate multi-model clarity/privacy review before publication.

## Sign-in and CSRF recovery

Authentication acceptance must exercise Django's CSRF middleware:
`Client(enforce_csrf_checks=True)` obtains the actual rendered token before
posting synthetic credentials. A default test client or a pre-created
authenticated session cannot prove this boundary.

The real-HTTP browser cases cover fresh sign-in/POST sign-out and stale-tab
recovery at narrow and desktop widths. Two pages share one browser context:
one retains an old login form while the other signs in and signs out. Separate
contexts have independent cookie jars and cannot reproduce this mechanism.
Neither `force_login` nor `page.set_content` substitutes for these journeys.

Rejected forms must stay HTTP 403, never cache, preserve safe GET-only login
destinations, and offer only home navigation for other failed actions. Test
missing/malformed tokens, missing cookies, origin/referer refusals, unsafe
destinations, unchanged authentication state and rate limits, and no submitted
credentials, tokens, forms, or technical reason in the response in either
DEBUG mode. The failure renderer is tested directly for zero queries and no
context processors; unchanged authentication/provider middleware may still
perform its existing queries around a complete HTTP request.

Synthetic full-credential evidence and an installed owner's login are separate
claims. Record unavailable private credentials explicitly rather than
provisioning an account or resetting a password for a deployment check.

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

## Model-outcome summary coverage

The immutable FHS prediction contract stays unchanged. A separate report must
derive internally from the same deterministic terminal arrays, rather than
trust supplied counts or infer probabilities from quantiles.

Cover:

- complete Loss / Flat to +20% / Above +20% partition, boundary ties, and
  nested terminal loss below -20%;
- unchanged 8,192-path denominator, including nonfinite/withheld failure;
- deterministic counts and same-shock zero-drift sensitivity;
- group rounding, symbolic small/extreme labels, exact counts, unsigned
  probability shares versus signed returns, and no real-world guarantees;
- full canonical source/listing/horizon/input/seed/projection identity;
- same-quantile/different-count forgery detected by offline re-derivation;
- registration accepting only source identity, not caller-authored statistics;
- actual publication time, earlier as-of invisibility, future-clock refusal,
  and no inherited prediction on-time status;
- idempotent recovery before replay, distinct concurrent publication clocks,
  one committed report/file with verified final bytes, and cross-process
  SQLite locking through the durable commit;
- safe retry after a failed registry insert without overwriting an earlier
  unpublished blob or inheriting its publication time;
- owner/issuance/product-bound children across owner reassignment, including
  source-checked compatibility with earlier fixed-name children;
- byte-identical frozen success and withheld output in base/head reproduction;
- unchanged historical parent verification and separate new-child evidence;
- source-complete but summary-missing/failed states and retained median/range;
- exact-run history binding rather than substituting current probabilities;
- no simulation, calculation, write, credential resolution, or fetch on GET;
- one native synthetic issuance-to-report-to-reader-to-rendered-UI chain;
- selected horizon across Opportunities, Under $10, stock detail and My List;
  and
- the existing mobile complete-card, desktop three-row, readable-navigation,
  and overflow limits on both local and Linux browser environments.

Numerical path-doubling comparisons describe precision only. They do not
establish calibration or add independent market observations, and an adverse
result is recorded rather than tuned away. Numeric calibration metrics are
outside this change; a future study requires a separately reviewed protocol.

## Reader and UI coverage

The fixed synthetic demo uses its fixture's end date, not the moving live
market date, in both current and history readers. Exercise a later read clock
without regenerating the source; real-provider stale-session rejection must
remain unchanged.

The native end-to-end contract must persist real synthetic evidence, call the
production writer, read through the fail-closed product reader, and render
authenticated Opportunities and stock detail.

Exercise the entry-point journey as well as direct routes: authenticated
sign-in without a `next` destination, authenticated landing, recognizable
Opportunities navigation, first-screen Under-$10 access,
filter/horizon selection, pagination, and stock detail. Use synthetic
representative lists for density and responsive measurements; do not repeat
full retrospective simulations merely to populate a layout fixture. Keep a
separate genuine native end-to-end case.

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
- 20-listing overview pages with stable filter/horizon query parameters;
- first-screen navigation and no page overflow at narrow and desktop widths;
- scoreless methods distinguished from missing legacy scores;
- non-observed history distinguished from reportable outcomes pending data
  or maturity, including mixed historical cohorts;
- all-null advisory forecasts shown as non-evaluable, not awaiting maturity;
- archived outcome counts separated by recorded evidence provenance;
- responsive/keyboard-readable tables; and
- separate observed, retrospective, archive, and synthetic labels.

`/status` must remain operational rather than duplicate forecast cards.
Progressive disclosure must not hide the summary of a restriction, a worse
baseline result, an empty comparison scope, or numerical sensitivity failures.

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

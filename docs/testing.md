# Testing

The test suite is offline and deterministic. Live provider endpoints and real
credentials are forbidden in tests and CI.

## Local quality gate

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run python manage.py makemigrations --check --dry-run
uv run python manage.py check --settings=stanstock.settings.test
```

For deployment settings, provide placeholder non-secret values:

```bash
DJANGO_SECRET_KEY=local-check-only \
DJANGO_ALLOWED_HOSTS=localhost \
DATABASE_URL=postgresql://user:password@localhost:5432/stanstock \
uv run python manage.py check --deploy --settings=stanstock.settings.prod
```

This command validates settings; it does not prove the database is reachable.

## Coverage areas

- separation of actual generation/retrieval time from historical data cutoffs,
  including future-row exclusion and observed rejection of late signals;
- immutable predictions, asset manifests, filing facts, and FX vintages at
  the Django and database layers;
- score, confidence, probability, timestamp, and scenario constraints;
- provider parsing and explicit failure states using synthetic payloads;
- Twelve Data activation/kill-switch behavior, header-only credentials,
  macOS Keychain fallback, Basic single-user enforcement, source-spike
  classifications, US-calendar target resolution, catalog and plan filtering,
  local quota accounting, immutable raw/Parquet persistence, eligibility
  exclusions, snapshot rollback, and target-job idempotency;
- prediction-level on-time evidence, next-session deadline enforcement,
  credential-free recovery of committed targets across retry-time grades,
  supported-horizon-only prediction persistence, generic `analyze` command
  research-only behavior for both same-day snapshot and listing paths, direct
  explicit observed-issuance rejection after the deadline, and monotonic
  current-market session updates;
- explicit security-type boundaries, one-fetch SPY benchmark/ETF reuse with
  unchanged credit accounting, zero-credit ETF reconstruction from immutable
  assets, exact completed-run benchmark-vintage recovery, cutoff-safe ETF
  metrics, fail-closed return-basis metadata, explicit fallback provenance for
  an omitted provider MIC, post-analysis ETF failure recovery, no ETF universe
  membership or stock analysis, and SPY portfolio valuation without a
  `StockAnalysis`;
- technical/fundamental math, canonical concept/restatement selection, and
  missing-value behavior;
- SEC mapping/config validation, safe historical-file paths, immutable raw
  submissions/history/Companyfacts replay, full period identity,
  acceptance/date-only availability, same-accession source revisions,
  value reversion, exact filing-source retrieval gates, interrupted
  normalization replay, bounded once-daily Companyfacts lag retries, amendment
  selection, instant/duration separation, restated-quarter precedence,
  gap-free TTM derivation, weighted-share arithmetic, FCF sign handling,
  non-overlapping debt components, SIC as-of reads, ETF exclusion, incremental
  request suppression, scheduler ordering/recovery, and SQLite/PostgreSQL
  immutability triggers;
- recommendation, risk, scenario, and explanation determinism;
- deterministic 6m/12m panel replay, fixed-epoch non-overlapping cohorts,
  historical-anchor leakage exclusion, equal cohort weighting, shrinkage,
  support/diversity/calibration withholding, immutable panel provenance, and
  advisory isolation from recommendations and opportunities;
- deterministic 3y/5y FCF/share and separately eligible EPS/share branches,
  negative/partial-FCF fallback refusal, selected-period and near-zero-EPS
  share-basis checks, config-gated adjacent-selected-period diluted-share
  continuity (default v2 enabled, frozen pinned v1 disabled and
  reproducible), bidirectional TTM-to-annual diluted-share continuity,
  bounded post-period split exposure, current SIC peer floors and fallback
  levels, unsupported-financial and low-multiple withholding, compatible
  invested-capital canonical/source bases and tax inputs, bounded
  insufficiency messages, ordered scenario math, actual-current-multiple
  return identity, cumulative/annualized parity, dividend exclusion, complete
  target provenance, compact immutable peer references and payload budgets,
  future-fact/classification exclusion, bounded SQLite evidence loading,
  observed/research-grade prediction issuance, method/config-hash cohort
  separation between long-v1 and long-v2, and recommendation/opportunity
  isolation;
- prospective, default-off `us-sec-long-v3` evidence selection: absent
  optional capability keys removed from the effective hash so frozen v1/v2
  hashes cannot move; newest-quarter alias anchoring with a single real
  controlling source fact for direct and YTD-derived quarters (including
  adversarial cases where availability and maximum revision belong to
  different dependencies), homogeneous-tail withholding, and determinism
  under reversed input order; same-date `(concept, alias, period identity)`
  candidate recovery of an invested-capital pair the collapsed series hides,
  with incompatible alternatives still withheld and no debt double counting;
  assessed pair-selection provenance carried through missing beginning,
  missing ending, zero compatible pairs, post-selection peer insufficiency,
  and success, never labeling rejected evidence verified; and separate,
  fail-closed audit boundaries (target date, data cutoff, decision time)
  with immutable-listing-ID identity and explicit cross-exchange/reused-
  ticker ambiguity;
- `us-sec-long-v3` evidence-selection boundaries specifically: a directly
  reported quarter and a year-to-date-derived quarter that tie on
  availability are resolved by the full rank of one real controlling filing
  (revision 5 derived beating revision 1 direct, against a competing
  incomplete alias at revision 3) rather than by availability alone, with
  opposing accession order and two distinct quarter period identities
  sharing one period end both covered, stable under reversed input order and
  under reassigned row UUIDs, while the frozen legacy quarter series keeps
  its original collapse; a deduplicated `manifest_evidence_fact_ids` closure
  proven for success, no compatible pair, a missing side, a
  selected-pair-then-peer-withheld run, and an unselected TTM alias lineage,
  with every alternative filed through its own source and filing assets so
  the immutable `source_assets` manifest and the evidence payload must cover
  them, TTM dependency closure included, and assessed evidence kept
  structurally separate from the selected `input_facts` and never labelled
  verified; failure-path classification proven independently by re-deriving
  the rejected candidates from the invested-capital assessment itself and
  asserting they are absent from `input_facts`/`selected_input_fact_ids`,
  present in `assessed_evidence`, and provable through their own source and
  filing assets, with a selected-pair-then-peer-withheld control showing the
  selected pair staying a formula input while its unused alternatives stay
  assessed; non-canonical or conflicting instant period identities producing
  an explicit listing-level assessed-withheld forecast and audit entry while
  the rest of the run still forecasts; and a 343-combination balance-sheet
  date refused from per-axis counts *before* any Cartesian product is
  materialized, recording each responsible axis (7 equity x 7 cash x 7
  reported-long-term-debt aliases, each with distinct source and filing
  assets) so all 21 responsible facts are assessed and manifest-covered
  without being selected, keeping no pair selected and the 343 products
  unenumerated, retaining an already-assessed opposite side when the refusal
  happens on the second side, and withholding that one listing while another
  listing still forecasts and both audit entries are returned, with the
  ceiling held at 256;
- a base-versus-worktree frozen differential: `tests/frozen_base.py` imports
  the pre-change source of `sec_fundamentals`, `long_forecast_config`, and
  `long_forecasts` straight from the git object database into an isolated
  module namespace, and `tests/test_long_frozen_differential.py` runs v1 and
  v2 through a fully deterministic fixture (UUID5 identities, fixed asset
  paths and checksums) to compare complete successful and withheld scenario
  and calculation payloads, insufficiency reasons, config hashes, and
  eligibility. The same comparison always runs against the committed golden
  in `tests/data/long_frozen_base_payloads.json`, generated from that exact
  base revision; only the `auto_now_add` ingest clock is normalized.

  **Regeneration contract.** The committed golden is base-produced evidence,
  never a recording of current behavior. An ordinary run only reads it; no
  comparison test rewrites it. Regeneration happens solely through
  `regenerate_golden`, which reads the base sources out of the git object
  database, executes *those* modules, and writes the result together with
  the SHA-256 of each exact base source it ran and a
  `generated_from: base_revision_execution` marker. If the base objects are
  unavailable it raises `BaseRevisionUnavailableError` and writes nothing --
  there is no working-tree fallback, so head behavior can never be committed
  under a `BASE_SHA` label. Tests cover all three: regeneration refusing
  without base objects, a poisoned head build being unable to influence a
  regenerated file (byte-identical to the committed golden), and the pooled
  representation expanding losslessly back to the complete base payloads.
  When the base objects are present, the golden's recorded checksums are
  verified against the real base bytes. Regenerate with
  `STANSTOCK_WRITE_FROZEN_GOLDEN=1 uv run pytest
  tests/test_long_frozen_differential.py`;
- committed/tracked production defaults with literal effective-hash pins for
  `us-price-baseline-v2`,
  `us-price-medium-v1`, and `us-sec-long-v2`; and template rendering that
  distinguishes assessed
  (`assessed_through`/incompatible-or-unverified) from verified
  (`verified_through`) split-basis evidence without implying a confirmed
  corporate action;
- target-date job idempotency and failure recording;
- observed-session outcome maturity, terminal outcome races, corporate-event
  detection, explicit 6m/12m/3y/5y maturity counts, withheld-scenario
  fail-closed unresolved outcomes not reaching a price lookup and excluded
  from advisory reporting denominators, earliest-reportable canonical
  observation selection before outcome status, duplicate reissues not
  satisfying sample thresholds or replacing unresolved/corporate-event
  originals, version-complete ledger/status transparency, provider-conflict
  rejection, batch price-frame reuse, unchanged-unresolved no-op behavior,
  decision/advisory outcome semantics, cross-table role-guard trigger
  persistence on SQLite and PostgreSQL, and reconstructed/live evidence
  separation;
- shared simulation accounting, point-in-time FX derivation (direct, inverse,
  cross, bounded carry, per-valued-date availability cutoffs, observed-versus-
  research retrieval, missing/stale/ambiguous failure), accounted-date FX
  coverage, closed-market currency exposure in valuation and rebalance
  sizing, close-only converted execution, stock versus FX attribution,
  full-content input hashing including native-currency assignment,
  persistence precision, and no-look-ahead behavior;
- authentication, trusted-proxy login throttling, filters, empty states, and
  evidence/provenance labeling;
- owner-scoped tracked portfolios, valuation completeness, stale-price
  refusal, same-day snapshot deduplication, snapshot database immutability,
  split warnings, opportunity-policy highlighting, exact price-band
  boundaries, neutral band filtering, Under $10 promotion/sample exclusion,
  released-foundation versus unreleased-control disclosure, complete
  stock-detail prediction history, current activation-context labels, ETF
  sample exclusion, supported-SPY holding selection, and preservation of
  existing holdings;
- immutable/idempotent external deposits, side-effect-free monthly previews,
  70/30 total-NAV arithmetic, fractional and whole-share rounding, residual
  cash, at-most-one explicitly short-horizon satellite, Under-$10 allocation
  refusal, stale-plan rejection, exact price/research provenance in plan
  hashes, compare-and-swap cash writes, portfolio-first lock order, duplicate
  confirmation/removal handling, and stale web/admin save protection;
- contribution performance reconciliation against immutable deposits,
  purchases, and post-manual-change baselines; flat-price zero return,
  all-time versus post-boundary contributions, stale/future/mixed-session or
  incompatible-basis withholding, persistent split warnings, unavailable
  baseline recovery, SQLite/PostgreSQL ledger immutability triggers, and
  multi-connection PostgreSQL lock ordering across deposits, confirmations,
  and snapshot foreign-key checks;
- backup checksums, extraction safety, transactional PostgreSQL restore, and
  database/assets bundling;
- architecture boundaries around raw provider modules.

## Review-time evidence outside the automated suite

The committed suite compares frozen and stricter methodology versions within
one checkout and pins their effective hashes. Relevant regressions include
`test_default_long_forecast_config_is_v2_with_distinct_hash_from_pinned_v1`,
`test_explicit_v1_config_is_unchanged_and_pinned`,
`test_v1_and_v2_long_forecast_cohorts_stay_separate_and_v1_prediction_unchanged`,
`test_v2_config_hash_and_production_default_path_remain_pinned`, and
`test_medium_forecast_config_is_versioned_and_stable`.

Those same-revision tests do not prove that a frozen version stayed unchanged
across a material edit. `tests/test_long_frozen_differential.py` now automates
that comparison for `us-sec-long-v1`/`v2` by executing the base revision's own
sources, so the reviewer's byte-for-byte base-versus-head reproduction is a
`pytest` job rather than a hand-run artifact for those versions. It is not yet
automated for any other frozen methodology or configuration version; there the
reviewer still runs the explicit base-versus-head reproduction against the two
checkouts and compares the frozen version's effective hash, eligibility,
reason wording, and successful and withheld calculation payloads byte-for-byte
as required release evidence.

The automated differential has one environment dependency: reading the base
revision's sources needs that revision in the local git object database. A
shallow clone -- including the default `actions/checkout` depth of 1 -- does
not contain it, so the live base comparison skips and only the committed
golden in `tests/data/long_frozen_base_payloads.json` is compared. Keeping the
exact-base differential live in CI therefore requires the quality job to check
out full history (`fetch-depth: 0`); without it, CI proves the golden but not
the base execution that produced it.

## Browser checks

Server-rendered behavior is covered with Django's test client. Playwright is
reserved for a focused authenticated smoke test of keyboard navigation,
responsive overflow, and critical flows once a browser runtime is available.

## Docker checks

CI runs the general suite on SQLite, exercises concurrent planner confirmation
against PostgreSQL 17 with independent connections, and builds the production
image after the Python quality job. Local Docker storage failures are
environment failures, not successful image validation; record them and rerun
on a host with sufficient Docker Desktop capacity rather than deleting
unrelated shared volumes.

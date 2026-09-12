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
- frozen medium-v1 base-object execution and a base-produced compressed golden
  covering config/parser behavior, complete panel rows/bytes/metadata,
  calculation/scenario/persisted payloads, default service selection, source
  behavior, and rendering labels; plus explicit medium-v2 literal/effective
  identity, strict YAML rejection, pre-output admission, two-phase
  select/check/read cutoff refusal, checksum and return-domain failures,
  coherent cohort-equal CDF math, matured-only prequential Brier/base/interval
  evidence, exact 14-/20-key payload states, both writer guards, fail-closed
  nested UI validation, authenticated persistence-to-template flow, outcome
  semantics, and decision/opportunity/simulation isolation;
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
- the **active** SEC correction-availability fix under the shipped default
  `us-sec-long-v2` configuration: an A -> B -> A ingestion with raw-content
  reuse selects the right fact identity and revision at the before-B,
  between, and after-reversion cutoffs, contrasted against the prior
  acceptance-backdated semantics that would have selected the newest revision
  at every one of them; unchanged late retrieval adds no fact or revision and
  moves no availability; and the default configuration carries none of the
  long-v3 payload fields;
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
- same-accession correction timing: a corrected value under an accession
  already held records the retrieval that carried it rather than the original
  filing acceptance, and a 100 -> 101 -> 100 content reversion whose bytes
  deduplicate onto the *original* `DataAsset` still binds to its own
  append-only `SourceObservationEvent`, proven with August/September/October
  retrievals and exact raw-content reuse across normalization, the long-v3
  forecast, and the long-v3 audit (between September and October the
  correction is selected; only after October does the reversion apply), with
  a replayed retrieval recording one event rather than a duplicate and every
  persisted row unchanged. An ordinary late retrieval of unchanged content is
  kept explicitly distinct: it appends no revision, moves no availability,
  and stays readable at its original cutoffs;
- observation-event integrity: a recorded digest must equal its own asset's
  `sha256`, enforced on write and re-checked on idempotent collision and on
  recovery; the racing-insert recovery is narrowed to the observation-instant
  uniqueness violation, so an unrelated integrity fault re-raises unchanged
  whether or not a row already exists at that instant, while ordinary
  collision handling is unaffected;
- declared, hashed long-v3 capabilities: correction-availability policy reads
  its own `proven_observation_correction_availability` key rather than
  inferring from the other two, and the same-date combination ceiling is
  accepted only as the reviewed 256, refused when absent alongside the joint
  search or declared without it, with frozen v1/v2 hashes unmoved;
- migration `0008`: PostgreSQL placeholder escaping alongside the existing
  immutability migrations, and a disposable-database forward/reverse/reapply
  cycle proving the pre-existing evidence triggers survive, the new table's
  protection returns, and reapply yields an empty table;
- retry and recovery after a reversion: with unchanged submissions, a
  controlled clock, and reconciliation not due, a no-op retry issues no
  Companyfacts request, appends no fourth revision, and leaves October's
  value selected; an interrupted normalization straight after the
  deduplicated reversion recovers the exact October event against the reused
  August asset; recovery refuses explicitly when two different payloads share
  both an observation timestamp and a commit clock, and when the only
  recoverable observation predates a committed correction. Distinct content
  at one observation instant is refused at write time -- proven through a
  full A -> B -> A ingestion and retry at a single timestamp with distinct
  local clocks, appending nothing stale -- while re-recording the same
  retrieval stays idempotent. A pre-event, legacy-shaped 100 -> 101 -> 100
  database refuses its replay outright with unchanged submissions and
  reconciliation not due, appending no fourth revision, inventing no
  companyfacts observation, and leaving historical selection unchanged;
  an upgraded database with no correction chain still recovers normally;
- re-binding after fresh proof: a due Companyfacts fetch that re-observes the
  reverted content of an unprovable legacy reversion appends a new revision
  with the same value and observation hash but an observation-bound
  availability and explicit `reobserved_unproven_correction` provenance,
  while every earlier row, asset, and event stays byte-identical. From that
  boundary the long-v3 forecast and audit select the reverted value again and
  the unprovable revision stays deferred and assessed; an earlier cutoff is
  unchanged; a further identical fetch or replay appends nothing; and frozen
  v1/v2 selection is unaffected;
- prospective legacy-correction resolution: rows persisted before the
  correction basis existed are resolved read-only, never rewritten. A legacy
  correction is admitted only where its own asset retrieval is a real
  observation of that revision. Both a reversion sharing an earlier
  revision's asset and a revision from an asset retrieved *before* its
  predecessor's boundary resolve to unknown and stay deferred at every
  cutoff -- including cutoffs long after the predecessor's boundary -- with
  the chain-ordering lower bound reported as assessed context only, carried
  as assessed and manifest-covered evidence, never selected, and with frozen
  v1/v2 selection unchanged;
- correction-policy configuration gating: the offline audit and the forecast
  answer the policy question through one shared function and are run across
  long-v1, long-v2, and long-v3 at identical boundaries. Frozen versions
  report `recorded_availability_only`, defer nothing, and keep selecting the
  backdated correction exactly as released; only long-v3 defers it, and
  admits it again once its retrieval is proven;
- alias-tail lineage completeness: facts examined and rejected *before* any
  quarter candidate exists -- an annual-only alternate alias and a
  year-to-date pair whose derivation is refused -- appear in
  `unusable_alias_source_fact_ids`, in the assessed evidence, and with both
  their companyfacts and filing assets in the manifest, while never entering
  the selected inputs;
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
  released-foundation versus released-shadow-diagnostic versus
  still-unreleased-control disclosure, the three Under-$10 detail panel states
  (assessed, withheld, not assessed) with explicit zero rendering, complete
  stock-detail prediction history, current activation-context labels, ETF
  sample exclusion, supported-SPY holding selection, and preservation of
  existing holdings;
- `us-under10-shadow-v1` shadow diagnostics: the four-state solvency partition
  with exact Decimal boundaries (`D == C`, `A == L`, `F == 0`, an exact
  four-quarter runway, and burn just above/below cash at eight decimals),
  adverse-versus-missing separation, each debt component independently missing
  versus explicitly zero, non-positive current liabilities, instant-period
  mismatch, the 200/201-day and future-date boundaries, TTM precedence over the
  annual fallback, proven versus deferred same-accession corrections with
  rejected evidence still referenced, exact canonical-concept/fact-provider/
  source-asset-provider SEC qualification across full-analysis reuse and
  price-only query paths, foreign and provider-mismatched lineage exclusion,
  cutoff-safe withholding that ignores unqualified late assets and never raises,
  raw 252-session window validation (251/252 boundary, duplicate dates, unsorted
  input, missing columns, invalid closes/volumes, non-finite products, zero
  volume as valid data, split equivalence), basis and staleness refusals that
  never invent metadata, Basic versus generic split-capability refusals with no
  verified branch, the provider-qualified literal `policy_hash` pin, canonical
  payload and assessment-checksum stability, and fixed false activation
  summaries in every branch;
- the stock-detail Under-$10 reader's own validation matrix: exact permanent
  listing id, run target date, data cutoff (bound to exact equality, not
  merely at or before), reference close, and currency binding, plus the
  exact immutable price-asset UUID *and* content checksum anchored to the
  original decision-prediction manifest and parent analysis provenance. The
  matrix rejects a whole-`data_quality`-blob transplant between two genuine
  same-date/same-price/same-cutoff/same-currency candidates with different
  listings, price assets, and solvency evidence in both directions, a
  same-UUID/different-checksum price asset on either side, and a
  missing/ambiguous parent source-asset anchor. It independently replays the
  accepted builder from the exact cutoff-clipped price asset and
  provider-qualified SEC lineage, rejecting recomputed-checksum substitutions
  of same-cardinality unrelated facts/assets, an altered finite liquidity
  median, an impossible one-fact complete claim, and changed operands that
  happen to preserve the same classification. The accidental-corruption
  checksum remains a separate layer (a stale hash alone is rejected);
  positively enumerated recognized-reason tables reject JSON list/dict reason
  values without raising in direct validation and authenticated GETs; and
  non-dict `data_quality` never raises;
- Under-$10 pipeline integration: candidate detection on the rounded
  decision-run reference close (including the $10 rounding boundary and non-USD
  refusal), zero additional SEC queries for a non-candidate, one added SEC query
  per candidate, at most one provider-plan lookup per run across several
  candidates, both `analyze_listing` and `analyze_snapshot` entry points, and a
  true `analyze_listing` -> committed database reload -> authenticated detail
  render proving the stored values survive only when exact evidence replay
  succeeds. Tests also prove bounded queries/file reads, no GET writes or
  provider access, no rewrite or backfill of existing analyses, and no shadow
  SEC asset entering `computation.source_assets`,
  `data_quality["source_assets"]`, or any prediction manifest, calculation, or
  source mode. A differential suite
  executes the actual base revision's `research/service.py`
  (`65314f87fe0eb8adbb05d35d3874c22c736ae54c`) against the same deterministic
  synthetic fixture and proves non-candidate payloads are equal in full,
  candidate payloads are equal after removing only `under10_assessment`, and
  prediction payloads/manifests plus opportunity qualification are unchanged;
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
across a material edit. `tests/test_long_frozen_differential.py` automates that
comparison for `us-sec-long-v1`/`v2`, and
`tests/test_medium_frozen_differential.py` does so for
`us-price-medium-v1`, by executing each sealed base revision's own sources.
Their committed goldens are base-produced and comparisons cover complete
normalized outputs rather than head-authored expected values.

The automated differential has one environment dependency: reading the base
revision's sources needs that revision in the local git object database. A
shallow clone -- including the default `actions/checkout` depth of 1 -- does
not contain it, so the live base comparison skips and only the committed
goldens in `tests/data/long_frozen_base_payloads.json` and
`tests/data/medium_v1_frozen_base_payloads.json` are compared. Keeping the
exact-base differentials live in CI therefore requires the quality job to
check out full history (`fetch-depth: 0`); without it, CI proves the goldens but not
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

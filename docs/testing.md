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
  ETF sample exclusion, supported-SPY holding selection, and preservation of
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
across a material edit. When a stricter methodology/configuration version is
introduced, the reviewer must also run an explicit base-versus-head
reproduction against the two checkouts and compare the frozen version's
effective hash, eligibility, reason wording, and successful and withheld
calculation payloads byte-for-byte. That review artifact is required release
evidence; it is not currently an automated `pytest` or CI job.

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

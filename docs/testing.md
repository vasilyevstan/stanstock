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
  supported-horizon-only prediction persistence, and monotonic current-market
  session updates;
- technical/fundamental math, canonical concept/restatement selection, and
  missing-value behavior;
- recommendation, risk, scenario, and explanation determinism;
- target-date job idempotency and failure recording;
- observed-session outcome maturity, terminal outcome races, corporate-event
  detection, and reconstructed/live evidence separation;
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
  split warnings, and opportunity-policy highlighting;
- backup checksums, extraction safety, transactional PostgreSQL restore, and
  database/assets bundling;
- architecture boundaries around raw provider modules.

## Browser checks

Server-rendered behavior is covered with Django's test client. Playwright is
reserved for a focused authenticated smoke test of keyboard navigation,
responsive overflow, and critical flows once a browser runtime is available.

## Docker checks

CI builds the production image after the Python quality job. Local Docker
storage failures are environment failures, not successful image validation;
record them and rerun on a host with sufficient Docker Desktop capacity rather
than deleting unrelated shared volumes.

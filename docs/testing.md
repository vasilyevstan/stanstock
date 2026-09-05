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
- technical/fundamental math, canonical concept/restatement selection, and
  missing-value behavior;
- recommendation, risk, scenario, and explanation determinism;
- target-date job idempotency and failure recording;
- observed-session outcome maturity, terminal outcome races, corporate-event
  detection, and reconstructed/live evidence separation;
- shared simulation accounting, native-currency enforcement, full-content
  input hashing, persistence precision, and no-look-ahead behavior;
- authentication, trusted-proxy login throttling, filters, empty states, and
  evidence/provenance labeling;
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

# Security review checklist

Applies to any change touching authentication, sessions, secrets, or
untrusted input. Reviewer: `stanstock-critic-tester`.

## Authentication and sessions

- [ ] Django's session/auth middleware remains enabled and unmodified unless
      the change explicitly targets it.
- [ ] Owner/admin-only views and actions revalidate the current
      authenticated user's permissions server-side; no client-supplied role
      or flag is trusted.
- [ ] `bootstrap_owner` / owner-credential flows never log or persist a
      plaintext password beyond the configured hash.

## CSRF and forms

- [ ] Every state-changing view uses Django's CSRF protection
      (`{% csrf_token %}` in templates, no `@csrf_exempt` without a named,
      reviewed reason).
- [ ] Forms and admin actions validate and sanitize input server-side, not
      only in the template/JS layer.

## Secrets

- [ ] No credential, API key, `DJANGO_SECRET_KEY`, or `DATABASE_URL` value
      appears in code, tests, fixtures, logs, or committed files.
- [ ] `.env.example` stays a template with placeholder values only; real
      secrets are never added to it.
- [ ] Provider API keys are read from environment/settings, never
      hard-coded, and never printed in error messages or admin output.

## Private financial data

- [ ] Private prediction history, portfolio holdings, or per-user financial
      data is not exposed to another user or an unauthenticated request.
- [ ] Debug/error pages and logs do not leak private data or secrets;
      `DJANGO_DEBUG` stays `false` outside local development.

## Proxy trust

- [ ] `X-Forwarded-For` affects authentication throttling only when
      `REMOTE_ADDR` is an explicitly configured trusted proxy.
- [ ] The deployment configuration documents which component overwrites
      forwarded headers; direct clients cannot select their own throttle key.

## Input handling

- [ ] User-supplied identifiers (tickers, slugs, dates) are validated before
      use in a query or file path; no path traversal into `STANSTOCK_DATA_DIR`
      or arbitrary DB lookups.
- [ ] Any new raw SQL is parameterized; the ORM is preferred.

## Dependencies

- [ ] A new dependency has an accepted upstream reason (not added
      speculatively) and does not duplicate existing functionality (e.g. a
      second web framework, ORM, or query engine).

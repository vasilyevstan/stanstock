# Deployment

> Release state: see [README](../README.md#release-status). The instructions
> below describe the implemented profile; they do not claim that the
> replacement product has passed release review or been activated.

StanStock is local-first. The canonical development path is Docker Compose,
and `compose.production.yaml` is a vendor-neutral single-instance contract,
not a promise of hosted capacity.

## Safe demo deployment

```bash
docker compose up --build
```

Demo mode uses `synthetic_demo` only. The replacement product setting defaults
to enabled:

```dotenv
STANSTOCK_RESEARCH_PRODUCT_ENABLED=true
```

This does not enable a live provider. In demo mode `refresh_demo` uses
deterministic synthetic histories and records research-grade output.

## Production requirements

- one private StanStock web/job instance;
- PostgreSQL 17-compatible storage;
- durable private volumes for `/app/var/data`, `/app/var/backups`, and
  `/app/var/static`;
- HTTPS at the application or a trusted reverse proxy;
- a strong Django secret and restrictive hosts/origins;
- an approved secret source for optional provider credentials; and
- a separately validated local/external trigger when live refresh is enabled.

Ephemeral storage is not acceptable for `STANSTOCK_DATA_DIR`.

## Cloud readiness and credentials

The production Compose file is a deployment building block, not a configured
cloud service. The supported scheduler is the macOS LaunchAgent described
below; no Linux timer or cloud deployment workflow is supplied.

The current application image excludes `.git`. Fresh observed issuance calls
`clean_git_revision` against a real clean checkout, so setting
`STANSTOCK_CODE_REVISION` inside that image is not sufficient. A future cloud
design must independently bind the job and web deployment to the same reviewed
revision without weakening that guard. A clean native job checkout alongside
the web container is one possible design, not an implemented cloud profile.

[GitHub-hosted runners](https://docs.github.com/en/actions/concepts/runners/github-hosted-runners)
are temporary workflow machines, not persistent application hosting.
[Standard runners for public repositories are free](https://docs.github.com/en/billing/concepts/product-billing/github-actions),
but that does not provide a free always-on VM, PostgreSQL service, or private
asset store. A separate VM or managed host has its own capacity, storage,
backup, and pricing requirements.

[GitHub Actions secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets)
are made available to eligible workflows, not exposed as a secret-retrieval
service for an arbitrary application server. A deployment would still need
reviewed secure delivery to a private runtime secret source. Do not describe
the key as accessible "only to me": authorized workflow execution, people able
to control those workflows, and runtime administrators are part of the trust
boundary. Keep provider secrets out of PR/test workflows, artifacts, and logs.

GitHub is not required merely to avoid Keychain. The existing unattended
entrypoint reads an owner-only environment file and forces
`STANSTOCK_DISABLE_KEYCHAIN=1`; interactive macOS onboarding may still use
Keychain. Moving secret storage must not disable a working local credential
or scheduler before an authorized, verified cutover.

A cloud cutover additionally needs an accepted host, persistent and matching
database/assets, a verified paired backup, one authoritative scheduler,
HTTPS, provider rights, and actual installation acceptance. Moving an
existing SQLite installation to the production PostgreSQL profile is a
separate validated data migration, not a connection-string substitution.

## Environment

Required production values:

- `DJANGO_SECRET_KEY`
- `DJANGO_ALLOWED_HOSTS`
- `DATABASE_URL`
- `STANSTOCK_DATA_DIR=/app/var/data`
- `STANSTOCK_BACKUP_DIR=/app/var/backups`
- `STANSTOCK_CODE_REVISION` set to the deployed immutable revision

Product/profile value:

- `STANSTOCK_RESEARCH_PRODUCT_ENABLED=true` for the replacement serving and
  scheduler profile

Optional live US value:

- `TWELVE_DATA_API_KEY`, only after provider rights and activation gates pass

Configure `DJANGO_CSRF_TRUSTED_ORIGINS` for the HTTPS origin. Trust forwarded
scheme/client headers only from immediate proxies that overwrite them.
Remove `STANSTOCK_OWNER_PASSWORD` after explicit owner bootstrap.

Never place a provider key in the image, repository, Compose file, command
line, URL, report, log, or database.

## Container controls

The production image:

- runs as an unprivileged user;
- supports a read-only root filesystem;
- drops Linux capabilities in the supplied template;
- writes only to durable mounted paths and its required runtime scratch path;
- contains PostgreSQL client utilities for backup/restore; and
- serves Django through Gunicorn.

`.env` files are excluded from Git and the image build context.

## Activation checklist

Do not claim product activation until all of the following are true at one
exact committed revision:

1. release-chain reviews and CI have passed;
2. the private operational acceptance has passed without narrowing its fixed
   denominator;
3. provider rights and current owner authorization are confirmed;
4. one manual research refresh succeeds and its registered evidence verifies;
5. the serving profile selects `research-product-v1`;
6. the scheduler selects the same product/config profile;
7. backup verification succeeds; and
8. rollback/disable procedures have been rehearsed without deleting immutable
   history.

Also complete the sign-in journey below. A healthy process or an
already-authenticated browser does not establish that a user can sign in.

The active price product has no SEC prerequisite. Do not block its market
stage on the separately configured SEC workflow.

### Sign-in and rejected-form acceptance

Bind the running process to the exact deployed revision, then use a signed-out
browser at the real deployment origin. Follow a protected deep link, submit
the rendered login form, and confirm that the intended local destination and
query parameters survive. Exercise the real POST sign-out control as well.
Pre-created sessions, `force_login`, and injected HTML are not substitutes.

Exercise a stale form using two tabs in one browser context: keep the first
tab's login form open, sign in and then sign out in the second, and submit
the first tab's old form. Rejection must remain HTTP 403. The public error
page must offer a fresh GET sign-in link, preserve a safe local destination,
and allow a fresh form submission without accepting or replaying the rejected
POST. Check desktop and narrow-screen navigation and static assets.

Use synthetic credentials for tests and isolated production-style dry runs.
An installed owner's full-credential journey uses only that owner's existing,
authorized credentials, entered privately. If those credentials are unavailable,
record that the installed credential journey was not performed; distinguish
the synthetic full-flow evidence from installed login-page and 403-recovery
evidence. Never create a private-installation user or reset a password to make
acceptance appear complete.

Do not disable CSRF protection or broaden trusted origins to hide a mismatch.
With the current cookie-based configuration, a restart alone does not change
the CSRF secret. A stale form is consistent with a successful sign-in in
another tab, but the actual cause requires evidence; deployment cookie or
proxy changes must be investigated separately.

## Scheduling

The supported macOS LaunchAgent runs at **03:30 local time,
Tuesday-Saturday**. For the reviewed deployment timezone this is
`Europe/Tallinn`. Installation validates the detected IANA timezone across
regular/early XNYS closes and DST transitions; unsafe zones are refused.

```bash
.venv/bin/python manage.py launchd_refresh install
.venv/bin/python manage.py launchd_refresh status
```

The scheduler and web server must point to the same database,
`STANSTOCK_DATA_DIR`, product setting, production config, and owner/display
authorization. A fresh observed issuance requires a clean committed checkout
and exact `STANSTOCK_CODE_REVISION`. A late sleep/wake run does not silently
downgrade or backdate a missing observed issuance.

Do not run a resident scheduler inside the web container. A separate clean
runtime checkout may share the configured database and asset directory with a
dirty development checkout.

## Production-style dry run

With a reachable PostgreSQL URL and non-secret local test values:

```bash
docker compose -f compose.production.yaml config --quiet
docker compose -f compose.production.yaml up --build
```

Verify migrations, static collection, `/healthz`, authentication, durable
restart behavior, synthetic product rendering, and backup verification. A dry
run does not establish live-provider entitlement or release acceptance.
Do not publish expanded Compose configuration: it can contain interpolated
credentials.

## Rollback

Application rollback redeploys a known compatible revision and disables future
prospective serving/scheduling when necessary. It must preserve every
immutable prediction, source asset, membership record, study report, and
archived method definition. Do not reverse an unsafe schema, reset successful
jobs, delete evidence, or rewrite old output to make rollback appear clean.

Restore data only for integrity recovery, using a verified paired
database/asset bundle. If provider rights have ended, do not restore a bundle
that would reintroduce data the owner is no longer entitled to retain.

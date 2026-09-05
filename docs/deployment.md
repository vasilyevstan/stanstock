# Deployment

StanStock's canonical environment is local Docker Compose. The repository also
contains `compose.production.yaml` as a vendor-neutral deployment contract,
not a claim of free hosted capacity.

## Required services

- One private StanStock web/job instance.
- External PostgreSQL 17-compatible database.
- Durable private volumes mounted at `/app/var/data`, `/app/var/backups`, and
  `/app/var/static`.
- HTTPS termination at the container or a trusted reverse proxy.
- Two external scheduled triggers if real price ingestion is ever approved.

Ephemeral container storage is not acceptable for `STANSTOCK_DATA_DIR`.

## Required environment

- `DJANGO_SECRET_KEY`
- `DJANGO_ALLOWED_HOSTS`
- `DATABASE_URL`
- `STANSTOCK_DATA_DIR=/app/var/data`
- `STANSTOCK_BACKUP_DIR=/app/var/backups`
- `STANSTOCK_CODE_REVISION` set to the immutable deployed revision

Configure `DJANGO_CSRF_TRUSTED_ORIGINS` for the HTTPS origin. Keep secure
redirect and secure cookies enabled. Set `DJANGO_TRUST_PROXY_HEADERS=true`
only when the immediate proxy overwrites `X-Forwarded-Proto`.
Set `STANSTOCK_LOGIN_TRUSTED_PROXY_IPS` to the comma-separated source IPs of
immediate proxies that overwrite `X-Forwarded-For`. The application ignores
forwarded client addresses from every other peer, preventing spoofed
login-throttle identities.

Owner credentials are used only for explicit bootstrap. Remove
`STANSTOCK_OWNER_PASSWORD` from the runtime environment after the account is
created so a restart cannot silently rotate it.

## Container controls

The runtime image:

- contains production dependencies only;
- includes PostgreSQL client utilities for the documented backup/restore
  commands;
- runs as the unprivileged `stanstock` user;
- supports a read-only root filesystem;
- drops Linux capabilities in the production Compose template;
- writes only to durable volumes and `/tmp`;
- serves through Gunicorn.

`.env` and `.env.*` files are excluded from the Docker build context (with
`.env.example` retained), so local runtime secrets cannot be copied into an
image layer.

At container startup, the entry point applies migrations, prepares
`STANSTOCK_DATA_DIR`, collects static files, and bootstraps the owner only
when `STANSTOCK_OWNER_PASSWORD` is present. It runs the idempotent synthetic
refresh only when `STANSTOCK_DEMO_MODE=true`.

## Production-style local dry run

Supply a reachable PostgreSQL URL and strong temporary secrets. For an HTTP
loopback-only dry run, explicitly disable secure redirect and secure cookies;
do not carry those overrides to a real deployment.

```bash
docker compose -f compose.production.yaml config
docker compose -f compose.production.yaml up --build
```

Then verify:

1. migrations and static collection complete;
2. `/healthz` reports ready;
3. anonymous data pages redirect to sign-in;
4. owner sign-in works;
5. a restart preserves PostgreSQL and `DATA_DIR`;
6. backup verification succeeds.

## Scheduled jobs

When a lawful price provider is approved, map the Europe-close and US-close
triggers to the same idempotent management command with explicit region and
target date. Do not run a resident scheduler in the web container.

## Rollback

Application rollback means redeploying a known image revision. Restore data
only when integrity requires it; never run an older image against an
incompatible forward-only schema without a reviewed compatibility decision.

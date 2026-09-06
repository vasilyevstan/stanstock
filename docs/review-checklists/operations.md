# Operations review checklist

Applies to any change touching scheduled jobs, deployment config, or backups.
Reviewer: `stanstock-critic-tester`.

## Idempotent jobs

- [ ] A scheduled or recompute job is idempotent for its `(job_name, region,
      target_date, attempt)` key; re-running it for the same target date does
      not duplicate rows or double-charge external quota.
- [ ] `JobRun.status` transitions (`running` -> `success`/`failed`/`skipped`/
      `no_data`) are set correctly on every exit path, including exceptions.
- [ ] A failed job leaves no partially written `DataAsset`, `Prediction`, or
      simulation row that a retry would treat as already complete.

## Local scheduling

- [ ] The LaunchAgent records and rechecks the machine IANA timezone, and its
      chosen wall-clock time is validated after regular/early XNYS closes,
      across DST changes, and before the next session opening.
- [ ] Sleep/wake recovery, holidays, duplicate targets, dirty worktrees, and
      missing or locked credential sources have explicit fail/skip behavior;
      no late run is marked as observed or on time.
- [ ] The plist, process arguments, and logs contain no populated `.env`
      values; unattended execution either reads an owner-only credential
      source promptly or fails.
- [ ] Aggregate orchestration preserves independently retryable market,
      evaluation, and portfolio-snapshot child jobs and recovers successful
      children before another provider call.
- [ ] Scheduled portfolio snapshots use the resolved exchange-session date,
      not the local wall-clock date.
- [ ] Local SQLite enables WAL plus a bounded busy timeout; concurrent web/job
      behavior has focused tests.

## Backups

- [ ] Backup/restore procedures cover both the PostgreSQL database and the
      asset files under `STANSTOCK_DATA_DIR` (Parquet/binary files); a
      database-only backup is treated as incomplete.
- [ ] A documented restore path exists for the paired database + asset-file
      backup, not just an instruction to restore one of the two.
- [ ] PostgreSQL restore is fail-fast and single-transaction, and asset
      destinations/capacity are validated before destructive database work.
- [ ] A read-only production container mounts `STANSTOCK_BACKUP_DIR` as a
      separate writable private path outside `STANSTOCK_DATA_DIR`.

## Deployment safety

- [ ] `compose.yaml` / `Dockerfile` changes keep the app runnable locally via
      `docker compose up --build` with no undocumented new required
      environment variable.
- [ ] A new required environment variable is added to `.env.example` with a
      safe placeholder, and to the deployment docs if one exists.
- [ ] `.dockerignore` excludes `.env` and other local secret files even when
      they are already ignored by Git.
- [ ] `DJANGO_DEBUG` stays `false` and `DJANGO_ALLOWED_HOSTS` stays
      restrictive outside local development.

## CI

- [ ] `.github/workflows/ci.yml` still passes lint, format check, type
      check, tests, `manage.py check`, and the container build for the
      changed slice.
- [ ] No test or CI step depends on a real provider network call or
      credential; only synthetic fixtures.

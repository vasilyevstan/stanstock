# Contracts review checklist

Applies to any change to models, migrations, services, or any interface
consumed by another app/module or by an external caller. Reviewer:
`stanstock-critic-tester`.

## Migrations

- [ ] Every model change has a matching migration generated with
      `manage.py makemigrations`; no drift between models and migrations.
- [ ] A migration that changes an existing column/constraint is backward
      compatible or ships an explicit backfill/rollback note in the PR.
- [ ] A new `UniqueConstraint` or `ForeignKey(on_delete=...)` choice matches
      the intended integrity rule (e.g. `PROTECT` for referenced vintage
      data, `CASCADE` only for genuinely owned child rows).
- [ ] No migration silently drops or truncates existing data.
- [ ] If SQLite recreates a table protected by custom triggers while adding a
      later constraint, a following migration reinstalls and tests those
      update/delete triggers.

## Model and service interfaces

- [ ] A changed method signature in `stanstock.data.asof.AsOfData`,
      `stanstock.data.assets.AssetStore`, or another shared service updates
      every call site in the same slice.
- [ ] `Prediction.save()`/`Prediction.delete()` immutability guards (and any
      equivalent guard added elsewhere) are not weakened or bypassed.
- [ ] JSON fields (`metadata`, `component_scores`, `quality_flags`, etc.)
      keep a stable, documented shape; a shape change is treated as a
      schema change requiring the material-change simplifier gate.

## Cross-app boundaries

- [ ] `core`, `data`, `research`, `simulation`, and `web` app boundaries are
      preserved; a new cross-app dependency is intentional and does not
      create an import cycle.
- [ ] Admin (`admin.py`) registrations stay in sync with model changes.

## Idempotent target dates

- [ ] A recompute/scheduled operation keyed by `(job_name, region,
      target_date[, attempt])` can be safely re-run for the same target date
      without duplicating or corrupting output (see `JobRun` uniqueness
      pattern).
- [ ] Re-running a job for a past `target_date` does not silently overwrite
      an immutable `Prediction` or vintage row; it appends a new version or
      is rejected.

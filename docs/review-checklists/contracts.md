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
- [ ] New portfolio ledger/baseline tables are protected from ORM and direct
      database update/delete on both SQLite and PostgreSQL, including after
      every table-rebuilding constraint operation.

## Model and service interfaces

- [ ] A changed method signature in `stanstock.data.asof.AsOfData`,
      `stanstock.data.assets.AssetStore`, or another shared service updates
      every call site in the same slice.
- [ ] `Prediction.save()`/`Prediction.delete()` immutability guards (and any
      equivalent guard added elsewhere) are not weakened or bypassed.
- [ ] JSON fields (`metadata`, `component_scores`, `quality_flags`, etc.)
      keep a stable, documented shape; a shape change is treated as a
      schema change requiring the material-change simplifier gate. This
      compatibility contract includes failure/withheld-scenario payload
      shapes and exact reason/insufficiency wording, not only the successful
      payload shape.
- [ ] A new production-default versioned configuration file (e.g. under
      `config/forecasts/`, `config/scoring/`) is committed and tracked in
      Git, and its literal expected effective config hash is pinned by a test
      so an untracked or silently edited default cannot change frozen
      behavior; comparing two loads of the same current file is not a pin.
- [ ] Ledger-managed cash and holding quantities are never persisted by an
      unrelated full-model save. Web and admin updates lock in portfolio-first
      order and use explicit `update_fields` allow-lists.
- [ ] A recoverable unavailable-boundary state has an explicit database
      invariant (valid snapshot xor non-empty issue); expected valuation
      failure cannot roll back the holding correction it is meant to record.

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

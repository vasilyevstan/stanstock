from __future__ import annotations

from datetime import UTC, date, datetime
from importlib import import_module
from types import SimpleNamespace


def test_analysis_cutoff_backfill_preserves_legacy_generation_time() -> None:
    migration = import_module(
        "stanstock.research.migrations.0006_analysis_cutoff_and_risk_insufficiency"
    )
    generated_at = datetime(2026, 9, 5, 12, tzinfo=UTC)
    run = SimpleNamespace(
        pk="legacy-run",
        target_date=date(2025, 1, 2),
        generated_at=generated_at,
    )
    updates: list[dict[str, object]] = []

    class QuerySet:
        def __init__(self, *, pending: bool) -> None:
            self.pending = pending

        def iterator(self):
            assert self.pending
            return iter([run])

        def update(self, **values: object) -> None:
            assert not self.pending
            updates.append(values)

    class Manager:
        def filter(self, **criteria: object) -> QuerySet:
            if criteria == {"data_cutoff__isnull": True}:
                return QuerySet(pending=True)
            assert criteria == {"pk": run.pk}
            return QuerySet(pending=False)

    analysis_run = SimpleNamespace(objects=Manager())
    apps = SimpleNamespace(
        get_model=lambda app_label, model_name: analysis_run,
    )

    migration.populate_analysis_cutoff(apps, schema_editor=None)

    assert updates == [{"data_cutoff": generated_at}]

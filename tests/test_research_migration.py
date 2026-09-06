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


def test_issued_on_time_backfill_is_conservative_for_legacy_runs() -> None:
    migration = import_module("stanstock.research.migrations.0007_analysis_run_issued_on_time")
    same_day = SimpleNamespace(
        pk="same-day",
        generated_at=datetime(2026, 9, 4, 21, tzinfo=UTC),
        target_date=date(2026, 9, 4),
    )
    overnight = SimpleNamespace(
        pk="overnight",
        generated_at=datetime(2026, 9, 5, 1, tzinfo=UTC),
        target_date=date(2026, 9, 4),
    )
    updates: list[tuple[str, dict[str, object]]] = []

    class QuerySet:
        def __init__(self, *, runs=None, pk: str | None = None) -> None:
            self.runs = runs
            self.pk = pk

        def only(self, *fields: str):
            assert fields == ("pk", "generated_at", "target_date")
            return self

        def iterator(self):
            assert self.runs is not None
            return iter(self.runs)

        def update(self, **values: object) -> None:
            assert self.pk is not None
            updates.append((self.pk, values))

    class Manager:
        def filter(self, **criteria: object) -> QuerySet:
            if criteria == {"universe_snapshot__grade": "observed"}:
                return QuerySet(runs=[same_day, overnight])
            return QuerySet(pk=str(criteria["pk"]))

    analysis_run = SimpleNamespace(objects=Manager())
    apps = SimpleNamespace(get_model=lambda app_label, model_name: analysis_run)

    migration.mark_existing_on_time_runs(apps, schema_editor=None)

    assert updates == [("same-day", {"issued_on_time": True})]


def test_prediction_issued_on_time_backfill_requires_original_run_timestamp() -> None:
    migration = import_module("stanstock.research.migrations.0007_analysis_run_issued_on_time")
    generated_at = datetime(2026, 9, 4, 21, tzinfo=UTC)
    run = SimpleNamespace(generated_at=generated_at)
    original = SimpleNamespace(
        pk="original",
        generated_at=generated_at,
        analysis=SimpleNamespace(run=run),
    )
    reissued = SimpleNamespace(
        pk="reissued",
        generated_at=datetime(2026, 9, 5, 12, tzinfo=UTC),
        analysis=SimpleNamespace(run=run),
    )
    updates: list[tuple[str, dict[str, object]]] = []

    class QuerySet:
        def __init__(self, *, predictions=None, pk: str | None = None) -> None:
            self.predictions = predictions
            self.pk = pk

        def select_related(self, *fields: str):
            assert fields == ("analysis__run",)
            return self

        def iterator(self):
            assert self.predictions is not None
            return iter(self.predictions)

        def update(self, **values: object) -> None:
            assert self.pk is not None
            updates.append((self.pk, values))

    class Manager:
        def filter(self, **criteria: object) -> QuerySet:
            if criteria == {"analysis__run__issued_on_time": True}:
                return QuerySet(predictions=[original, reissued])
            return QuerySet(pk=str(criteria["pk"]))

    prediction = SimpleNamespace(objects=Manager())
    apps = SimpleNamespace(get_model=lambda app_label, model_name: prediction)

    migration.mark_existing_on_time_predictions(apps, schema_editor=None)

    assert updates == [("original", {"issued_on_time": True})]


def test_latest_market_session_backfill_prefers_asset_period_end() -> None:
    migration = import_module("stanstock.data.migrations.0004_latest_market_data_session_date")
    with_period = SimpleNamespace(
        pk="with-period",
        observed_at=datetime(2026, 9, 5, 1, tzinfo=UTC),
        listing=SimpleNamespace(region="us"),
        source_asset=SimpleNamespace(period_end=date(2026, 9, 4)),
    )
    without_period = SimpleNamespace(
        pk="without-period",
        observed_at=datetime(2026, 9, 5, 1, tzinfo=UTC),
        listing=SimpleNamespace(region="us"),
        source_asset=SimpleNamespace(period_end=None),
    )
    updates: list[tuple[str, dict[str, object]]] = []

    class QuerySet:
        def select_related(self, *fields: str):
            assert fields == ("source_asset", "listing")
            return self

        def filter(self, **criteria: object):
            assert criteria == {"session_date__isnull": True}
            return self

        def iterator(self):
            return iter([with_period, without_period])

    class UpdateQuerySet:
        def __init__(self, pk: str) -> None:
            self.pk = pk

        def update(self, **values: object) -> None:
            updates.append((self.pk, values))

    class Manager:
        def select_related(self, *fields: str) -> QuerySet:
            return QuerySet().select_related(*fields)

        def filter(self, **criteria: object) -> UpdateQuerySet:
            return UpdateQuerySet(str(criteria["pk"]))

    latest_market_data = SimpleNamespace(objects=Manager())
    apps = SimpleNamespace(get_model=lambda app_label, model_name: latest_market_data)

    migration.populate_session_dates(apps, schema_editor=None)

    assert updates == [
        ("with-period", {"session_date": date(2026, 9, 4)}),
        ("without-period", {"session_date": date(2026, 9, 4)}),
    ]

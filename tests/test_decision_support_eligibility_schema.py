from __future__ import annotations

from datetime import timedelta
from importlib import import_module
from types import SimpleNamespace

import pytest
from django.apps import apps
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.management import call_command
from django.db import DatabaseError, connection, connections, transaction

from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis

MIGRATION = "stanstock.research.migrations.0010_decision_support_eligibility"
pytestmark = pytest.mark.django_db


@pytest.mark.parametrize(
    "field,value",
    [
        ("config_hash", "f" * 64),
        ("code_revision", "f" * 40),
        ("issued_on_time", True),
        ("target_date", "2026-09-10"),
    ],
)
def test_bulk_parent_identity_changes_are_rejected(c2_persisted, field, value):
    with pytest.raises(DatabaseError), transaction.atomic():
        AnalysisRun.objects.filter(pk=c2_persisted.run.id).update(**{field: value})


@pytest.mark.parametrize(
    "change",
    [
        {"method_version": "us-relative-momentum-v1"},
        {"config_hash": "f" * 64},
        {"horizon": "12m"},
        {"overall_score": 50},
        {"probability_positive": "0.5"},
        {"code_revision": "f" * 40},
        {"issued_on_time": True},
        {"evidence_grade": "observed"},
        {"price_provider": "synthetic_demo"},
        {"source_mode": "demo"},
        {"price_subject": "WRONG"},
    ],
)
def test_bulk_prediction_bypass_still_requires_exact_v2_parent_binding(c2_persisted, change):
    original = Prediction.objects.filter(evidence_role="decision").first()
    values = {
        field.attname: getattr(original, field.attname)
        for field in Prediction._meta.concrete_fields
        if not field.primary_key
    }
    values.update(change, model_version="synthetic-incompatible-new-row")
    with pytest.raises(DatabaseError), transaction.atomic():
        Prediction.objects.bulk_create([Prediction(**values)])


def test_listing_and_chronology_must_match_own_parent_without_model_clean(c2_persisted):
    rows = list(Prediction.objects.filter(evidence_role="decision"))
    original = rows[0]
    values = {
        field.attname: getattr(original, field.attname)
        for field in Prediction._meta.concrete_fields
        if not field.primary_key
    }
    for change in (
        {"listing_id": rows[1].listing_id},
        {"generated_at": original.generated_at + timedelta(seconds=1)},
        {"data_cutoff": original.data_cutoff - timedelta(seconds=1)},
    ):
        with pytest.raises(DatabaseError), transaction.atomic():
            Prediction.objects.bulk_create(
                [Prediction(**{**values, **change, "model_version": "synthetic-parent-mismatch"})]
            )


def test_table_rebuild_reinstalled_immutable_bulk_and_direct_sql_guards(c2_persisted):
    row = Prediction.objects.first()
    with pytest.raises(DatabaseError, match="immutable"), transaction.atomic():
        Prediction.objects.filter(pk=row.id).update(price_at_prediction=23)
    with pytest.raises(DatabaseError, match="immutable"), transaction.atomic():
        Prediction.objects.filter(pk=row.id).delete()
    with (
        pytest.raises(DatabaseError, match="immutable"),
        transaction.atomic(),
        connection.cursor() as cursor,
    ):
        cursor.execute("UPDATE research_prediction SET price_at_prediction = 23")
    with (
        pytest.raises(DatabaseError, match="immutable"),
        transaction.atomic(),
        connection.cursor() as cursor,
    ):
        cursor.execute("DELETE FROM research_prediction")
    assert Prediction.objects.count() == 10


def test_stock_cannot_be_reparented_between_even_identical_v2_runs(c2_persisted):
    env = c2_persisted
    original = env.run
    values = {
        field.attname: getattr(original, field.attname)
        for field in AnalysisRun._meta.concrete_fields
        if not field.primary_key
    }
    other = AnalysisRun.objects.create(**values)
    with pytest.raises(DatabaseError), transaction.atomic():
        StockAnalysis.objects.filter(run=original).update(run=other)


def test_model_clean_rejects_bad_v2_hash_even_before_first_child(c2_environment):
    row = AnalysisRun(
        universe_snapshot_id="00000000-0000-0000-0000-000000000000",
        generated_at=c2_environment.now,
        data_cutoff=c2_environment.now,
        target_date=c2_environment.target,
        config_version=c2_environment.version,
        config_hash="f" * 64,
        code_revision="a" * 40,
    )
    with pytest.raises(ValidationError, match="exact accepted config"):
        row.clean()


def test_unsafe_reverse_refuses_immutable_liquidity_before_any_parent(c2_closed_batch):
    migration = import_module(MIGRATION)
    assert not AnalysisRun.objects.exists()
    with pytest.raises(RuntimeError, match="immutable v2 evidence"):
        migration.reject_unsafe_reverse(apps, SimpleNamespace(connection=connection))


@pytest.mark.parametrize("vendor", ["sqlite", "postgresql"])
def test_guard_sql_covers_exact_versions_parent_identity_reparent_and_hold_null(vendor):
    statements = []
    migration = import_module(MIGRATION)
    migration.install_guards(
        None,
        SimpleNamespace(
            connection=SimpleNamespace(vendor=vendor),
            execute=statements.append,
        ),
    )
    sql = "\n".join(statements)
    assert "us-relative-momentum-v1" in sql and "us-relative-momentum-v2" in sql
    assert migration.V2_HASH in sql
    assert "OLD.run_id <> NEW.run_id" in sql
    assert "NEW.generated_at" in sql and "NEW.listing_id" in sql
    assert "NEW.success IS NOT NULL" in sql
    if vendor == "postgresql":
        assert sql.count("CREATE OR REPLACE FUNCTION") == 4
        assert sql.count("CREATE TRIGGER") == 4
        assert "ERRCODE = '23514'" in sql


def test_sqlite_forward_reverse_reapply_restores_prediction_and_evidence_guards(
    tmp_path, django_db_blocker
):
    alias = "c2_disposable_migration_cycle"
    settings.DATABASES[alias] = {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(tmp_path / "c2-migration.sqlite3"),
    }
    connections.settings[alias] = connections.configure_settings(settings.DATABASES)[alias]
    expected = {
        "research_prediction_prevent_update",
        "research_prediction_prevent_delete",
        "research_prediction_product_insert",
        "research_prediction_product_update",
        "research_stockanalysis_product_update",
        "research_analysisrun_product_update",
        "data_dataasset_prevent_update",
        "data_dataasset_prevent_delete",
    }
    try:
        with django_db_blocker.unblock():
            for target in ("0010", "0009", "0010"):
                call_command("migrate", "research", target, database=alias, verbosity=0)
                with connections[alias].cursor() as cursor:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
                    assert expected <= {row[0] for row in cursor.fetchall()}
    finally:
        connections[alias].close()
        del connections[alias]
        connections.settings.pop(alias, None)
        settings.DATABASES.pop(alias, None)

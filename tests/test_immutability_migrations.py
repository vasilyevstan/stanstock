from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from django.conf import settings
from django.core.management import call_command
from django.db import DatabaseError, connection, connections


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("stanstock.data.migrations.0003_immutable_evidence", "protect_evidence"),
        (
            "stanstock.portfolio.migrations.0002_snapshot_immutability",
            "protect_snapshots",
        ),
        (
            "stanstock.portfolio.migrations.0004_contribution_planner",
            "protect_portfolio_ledger",
        ),
        ("stanstock.data.migrations.0006_sec_fact_identity", "protect_evidence"),
        (
            "stanstock.data.migrations.0007_fundamental_fact_evidence",
            "protect_fact_evidence",
        ),
        (
            "stanstock.data.migrations.0008_source_observation_event",
            "protect_observation_events",
        ),
    ],
)
def test_postgresql_immutability_sql_escapes_psycopg_placeholders(
    module_name: str,
    function_name: str,
) -> None:
    statements: list[str] = []
    schema_editor = SimpleNamespace(
        connection=SimpleNamespace(
            vendor="postgresql",
            introspection=SimpleNamespace(table_names=lambda: []),
        ),
        execute=statements.append,
        quote_name=lambda name: f'"{name}"',
    )

    migration = import_module(module_name)
    getattr(migration, function_name)(apps=None, schema_editor=schema_editor)

    function_sql = next(sql for sql in statements if "CREATE OR REPLACE FUNCTION" in sql)
    assert "RAISE EXCEPTION '%% records are immutable'" in function_sql


SQLITE_EVIDENCE_TRIGGERS = (
    "data_fundamentalfact_prevent_update",
    "data_fundamentalfact_prevent_delete",
    "data_fundamentalfactevidence_prevent_update",
    "data_fundamentalfactevidence_prevent_delete",
)
SQLITE_OBSERVATION_TRIGGERS = (
    "data_sourceobservationevent_prevent_update",
    "data_sourceobservationevent_prevent_delete",
)


def _sqlite_triggers(connection) -> set[str]:
    with connection.cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
        return {row[0] for row in cursor.fetchall()}


@pytest.mark.skipif(
    connection.vendor != "sqlite",
    reason="disposable-database migration cycle uses the SQLite backend",
)
@pytest.mark.django_db
def test_observation_event_migration_cycles_without_losing_evidence_triggers(
    tmp_path: Path,
    django_db_blocker: Any,
) -> None:
    """Forward, reverse, and reapply `0008` on a disposable database.

    Creating a table cannot rebuild `data_fundamentalfact` or
    `data_fundamentalfactevidence`, but that has to be *demonstrated* rather
    than assumed: a later migration that silently recreated either table
    would drop its immutability triggers. The cycle also proves the new
    table's own protection is reinstalled by a reapply rather than only
    existing on a first-ever migrate, and that reversing genuinely destroys
    the observation evidence -- reapply returns an empty table.

    A throwaway database file is used so nothing here can touch the test
    database or leave migration state behind. The `django_db` mark is needed
    only because `0004_latest_market_data_session_date` runs a default-routed
    queryset, so its historical model must resolve against a real default
    database even while this cycle targets the throwaway alias.
    """
    database = str(tmp_path / "migration-cycle.sqlite3")
    alias = "migration_cycle"
    settings.DATABASES[alias] = {**settings.DATABASES["default"], "NAME": database}
    connections.settings[alias] = connections.configure_settings(settings.DATABASES)[alias]
    try:
        with django_db_blocker.unblock():
            target = connections[alias]
            call_command("migrate", database=alias, verbosity=0)
            after_forward = _sqlite_triggers(target)
            assert set(SQLITE_EVIDENCE_TRIGGERS) <= after_forward
            assert set(SQLITE_OBSERVATION_TRIGGERS) <= after_forward
            assert "data_sourceobservationevent" in target.introspection.table_names()

            call_command("migrate", "data", "0007", database=alias, verbosity=0)
            after_reverse = _sqlite_triggers(target)
            # Reversing removes the new table and only its own protection.
            assert set(SQLITE_EVIDENCE_TRIGGERS) <= after_reverse
            assert not (set(SQLITE_OBSERVATION_TRIGGERS) & after_reverse)
            assert "data_sourceobservationevent" not in target.introspection.table_names()

            call_command("migrate", "data", database=alias, verbosity=0)
            after_reapply = _sqlite_triggers(target)
            assert set(SQLITE_EVIDENCE_TRIGGERS) <= after_reapply
            assert set(SQLITE_OBSERVATION_TRIGGERS) <= after_reapply
            assert "data_sourceobservationevent" in target.introspection.table_names()

            with target.cursor() as cursor:
                # Reapply restores an *empty* table: the reverse destroyed the
                # observation evidence and nothing recreates it.
                cursor.execute("SELECT COUNT(*) FROM data_sourceobservationevent")
                assert cursor.fetchone()[0] == 0
                # The restored protection is real, not merely present by name.
                cursor.execute(
                    "INSERT INTO data_dataasset "
                    "(id, provider, kind, subject, relative_path, sha256, "
                    "retrieved_at, available_at, schema_version, metadata) "
                    "VALUES (?, 'sec', 'sec_companyfacts', 's', 'p', ?, ?, ?, '1', '{}')",
                    [uuid4().hex, "b" * 64, "2026-10-18 12:00:00", "2026-10-18 12:00:00"],
                )
                cursor.execute(
                    "INSERT INTO data_sourceobservationevent "
                    "(id, provider, kind, subject, content_sha256, observed_at, "
                    "recorded_at, source_asset_id) "
                    "SELECT ?, 'sec', 'sec_companyfacts', 's', ?, ?, ?, id "
                    "FROM data_dataasset LIMIT 1",
                    [
                        uuid4().hex,
                        "a" * 64,
                        "2026-10-18 12:00:00",
                        "2026-10-18 12:00:00",
                    ],
                )
            with pytest.raises(DatabaseError, match="immutable"), target.cursor() as cursor:
                cursor.execute("DELETE FROM data_sourceobservationevent")
            with pytest.raises(DatabaseError, match="immutable"), target.cursor() as cursor:
                cursor.execute("UPDATE data_sourceobservationevent SET subject = 'x'")
    finally:
        connections[alias].close()
        del connections[alias]
        connections.settings.pop(alias, None)
        settings.DATABASES.pop(alias, None)
        Path(database).unlink(missing_ok=True)

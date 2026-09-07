from importlib import import_module
from types import SimpleNamespace

import pytest


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

from django.db import migrations

PROTECTED_MODELS = {
    "PortfolioSnapshot": "portfolio_portfoliosnapshot",
    "PortfolioSnapshotHolding": "portfolio_portfoliosnapshotholding",
}
POSTGRES_FUNCTION = "stanstock_prevent_portfolio_snapshot_mutation"


def protect_snapshots(apps, schema_editor):
    vendor = schema_editor.connection.vendor

    if vendor == "sqlite":
        for model_name, table_name in PROTECTED_MODELS.items():
            table = schema_editor.quote_name(table_name)
            slug = model_name.lower()
            schema_editor.execute(
                f"""
                CREATE TRIGGER portfolio_{slug}_prevent_update
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{model_name} records are immutable');
                END
                """
            )
            schema_editor.execute(
                f"""
                CREATE TRIGGER portfolio_{slug}_prevent_delete
                BEFORE DELETE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{model_name} records are immutable');
                END
                """
            )
        return

    if vendor == "postgresql":
        schema_editor.execute(
            f"""
            CREATE OR REPLACE FUNCTION {POSTGRES_FUNCTION}()
            RETURNS trigger AS $$
            BEGIN
                RAISE EXCEPTION '%% records are immutable', TG_TABLE_NAME
                    USING ERRCODE = '55000';
            END;
            $$ LANGUAGE plpgsql
            """
        )
        for model_name, table_name in PROTECTED_MODELS.items():
            table = schema_editor.quote_name(table_name)
            trigger = f"portfolio_{model_name.lower()}_prevent_mutation"
            schema_editor.execute(
                f"""
                CREATE TRIGGER {trigger}
                BEFORE UPDATE OR DELETE ON {table}
                FOR EACH ROW EXECUTE FUNCTION {POSTGRES_FUNCTION}()
                """
            )
        return

    raise RuntimeError(f"Unsupported database vendor for portfolio immutability: {vendor}")


def unprotect_snapshots(apps, schema_editor):
    vendor = schema_editor.connection.vendor

    if vendor == "sqlite":
        for model_name in PROTECTED_MODELS:
            slug = model_name.lower()
            schema_editor.execute(
                f"DROP TRIGGER IF EXISTS portfolio_{slug}_prevent_update"
            )
            schema_editor.execute(
                f"DROP TRIGGER IF EXISTS portfolio_{slug}_prevent_delete"
            )
        return

    if vendor == "postgresql":
        for model_name, table_name in PROTECTED_MODELS.items():
            table = schema_editor.quote_name(table_name)
            trigger = f"portfolio_{model_name.lower()}_prevent_mutation"
            schema_editor.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
        schema_editor.execute(f"DROP FUNCTION IF EXISTS {POSTGRES_FUNCTION}()")


class Migration(migrations.Migration):
    dependencies = [
        ("portfolio", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(protect_snapshots, unprotect_snapshots),
    ]

from django.db import migrations

PROTECTED_MODELS = {
    "DataAsset": "data_dataasset",
    "FundamentalFact": "data_fundamentalfact",
    "FxRate": "data_fxrate",
}
POSTGRES_FUNCTION = "stanstock_prevent_evidence_mutation"


def protect_evidence(apps, schema_editor):
    vendor = schema_editor.connection.vendor

    if vendor == "sqlite":
        for model_name, table_name in PROTECTED_MODELS.items():
            table = schema_editor.quote_name(table_name)
            slug = model_name.lower()
            schema_editor.execute(
                f"""
                CREATE TRIGGER data_{slug}_prevent_update
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, '{model_name} records are immutable');
                END
                """
            )
            schema_editor.execute(
                f"""
                CREATE TRIGGER data_{slug}_prevent_delete
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
            trigger = f"data_{model_name.lower()}_prevent_mutation"
            schema_editor.execute(
                f"""
                CREATE TRIGGER {trigger}
                BEFORE UPDATE OR DELETE ON {table}
                FOR EACH ROW EXECUTE FUNCTION {POSTGRES_FUNCTION}()
                """
            )
        return

    raise RuntimeError(f"Unsupported database vendor for evidence immutability: {vendor}")


def unprotect_evidence(apps, schema_editor):
    vendor = schema_editor.connection.vendor

    if vendor == "sqlite":
        for model_name in PROTECTED_MODELS:
            slug = model_name.lower()
            schema_editor.execute(f"DROP TRIGGER IF EXISTS data_{slug}_prevent_update")
            schema_editor.execute(f"DROP TRIGGER IF EXISTS data_{slug}_prevent_delete")
        return

    if vendor == "postgresql":
        for model_name, table_name in PROTECTED_MODELS.items():
            table = schema_editor.quote_name(table_name)
            trigger = f"data_{model_name.lower()}_prevent_mutation"
            schema_editor.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
        schema_editor.execute(f"DROP FUNCTION IF EXISTS {POSTGRES_FUNCTION}()")


class Migration(migrations.Migration):
    dependencies = [
        ("data", "0002_dataasset_asset_available_before_retrieval_and_more"),
    ]

    operations = [
        migrations.RunPython(protect_evidence, unprotect_evidence),
    ]

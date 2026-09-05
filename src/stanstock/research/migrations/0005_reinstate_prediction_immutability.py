from django.db import migrations

UPDATE_TRIGGER = "research_prediction_prevent_update"
DELETE_TRIGGER = "research_prediction_prevent_delete"
POSTGRES_FUNCTION = "stanstock_prevent_prediction_mutation"


def protect_predictions(apps, schema_editor):
    prediction = apps.get_model("research", "Prediction")
    table = schema_editor.quote_name(prediction._meta.db_table)
    vendor = schema_editor.connection.vendor

    if vendor == "sqlite":
        schema_editor.execute(f"DROP TRIGGER IF EXISTS {UPDATE_TRIGGER}")
        schema_editor.execute(f"DROP TRIGGER IF EXISTS {DELETE_TRIGGER}")
        schema_editor.execute(
            f"""
            CREATE TRIGGER {UPDATE_TRIGGER}
            BEFORE UPDATE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'Predictions are immutable');
            END
            """
        )
        schema_editor.execute(
            f"""
            CREATE TRIGGER {DELETE_TRIGGER}
            BEFORE DELETE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'Predictions are immutable');
            END
            """
        )
        return

    if vendor == "postgresql":
        schema_editor.execute(f"DROP TRIGGER IF EXISTS {UPDATE_TRIGGER} ON {table}")
        schema_editor.execute(f"DROP TRIGGER IF EXISTS {DELETE_TRIGGER} ON {table}")
        schema_editor.execute(
            f"""
            CREATE OR REPLACE FUNCTION {POSTGRES_FUNCTION}()
            RETURNS trigger AS $$
            BEGIN
                RAISE EXCEPTION 'Predictions are immutable' USING ERRCODE = '55000';
            END;
            $$ LANGUAGE plpgsql
            """
        )
        schema_editor.execute(
            f"""
            CREATE TRIGGER {UPDATE_TRIGGER}
            BEFORE UPDATE ON {table}
            FOR EACH ROW EXECUTE FUNCTION {POSTGRES_FUNCTION}()
            """
        )
        schema_editor.execute(
            f"""
            CREATE TRIGGER {DELETE_TRIGGER}
            BEFORE DELETE ON {table}
            FOR EACH ROW EXECUTE FUNCTION {POSTGRES_FUNCTION}()
            """
        )
        return

    raise RuntimeError(f"Unsupported database vendor for prediction immutability: {vendor}")


def unprotect_predictions(apps, schema_editor):
    prediction = apps.get_model("research", "Prediction")
    table = schema_editor.quote_name(prediction._meta.db_table)
    vendor = schema_editor.connection.vendor

    if vendor == "sqlite":
        schema_editor.execute(f"DROP TRIGGER IF EXISTS {UPDATE_TRIGGER}")
        schema_editor.execute(f"DROP TRIGGER IF EXISTS {DELETE_TRIGGER}")
    elif vendor == "postgresql":
        schema_editor.execute(f"DROP TRIGGER IF EXISTS {UPDATE_TRIGGER} ON {table}")
        schema_editor.execute(f"DROP TRIGGER IF EXISTS {DELETE_TRIGGER} ON {table}")
        schema_editor.execute(f"DROP FUNCTION IF EXISTS {POSTGRES_FUNCTION}()")


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0004_prediction_outcome_status_constraints"),
    ]

    operations = [
        migrations.RunPython(protect_predictions, unprotect_predictions),
    ]

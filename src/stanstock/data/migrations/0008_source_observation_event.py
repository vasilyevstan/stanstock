"""Append-only observation events for content-addressed provider assets.

The uniqueness key deliberately excludes ``content_sha256``: one observation
instant names exactly one content, and that is enforced at the database
boundary so concurrent writers cannot both claim an instant.

Creating a brand-new table cannot rebuild `data_fundamentalfact` or
`data_fundamentalfactevidence`, so the immutability triggers installed by
`0003_immutable_evidence`/`0006_sec_fact_identity`/`0007_fundamental_fact_evidence`
are untouched by this migration. The new table gets the same protection.
"""

import uuid

import django.db.models.deletion
from django.db import migrations, models

MODEL_NAME = "SourceObservationEvent"
TABLE_NAME = "data_sourceobservationevent"
POSTGRES_FUNCTION = "stanstock_prevent_evidence_mutation"


def protect_observation_events(apps, schema_editor):
    vendor = schema_editor.connection.vendor
    table = schema_editor.quote_name(TABLE_NAME)
    if vendor == "sqlite":
        schema_editor.execute(
            f"""
            CREATE TRIGGER data_sourceobservationevent_prevent_update
            BEFORE UPDATE ON {table}
            BEGIN
                SELECT RAISE(ABORT, '{MODEL_NAME} records are immutable');
            END
            """
        )
        schema_editor.execute(
            f"""
            CREATE TRIGGER data_sourceobservationevent_prevent_delete
            BEFORE DELETE ON {table}
            BEGIN
                SELECT RAISE(ABORT, '{MODEL_NAME} records are immutable');
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
        schema_editor.execute(
            f"""
            CREATE TRIGGER data_sourceobservationevent_prevent_mutation
            BEFORE UPDATE OR DELETE ON {table}
            FOR EACH ROW EXECUTE FUNCTION {POSTGRES_FUNCTION}()
            """
        )
        return
    raise RuntimeError(f"Unsupported database vendor for evidence immutability: {vendor}")


def unprotect_observation_events(apps, schema_editor):
    vendor = schema_editor.connection.vendor
    table = schema_editor.quote_name(TABLE_NAME)
    if vendor == "sqlite":
        schema_editor.execute("DROP TRIGGER IF EXISTS data_sourceobservationevent_prevent_update")
        schema_editor.execute("DROP TRIGGER IF EXISTS data_sourceobservationevent_prevent_delete")
        return
    if vendor == "postgresql":
        schema_editor.execute(
            f"DROP TRIGGER IF EXISTS data_sourceobservationevent_prevent_mutation ON {table}"
        )
        return
    raise RuntimeError(f"Unsupported database vendor for evidence immutability: {vendor}")


class Migration(migrations.Migration):
    dependencies = [
        ("data", "0007_fundamental_fact_evidence"),
    ]

    operations = [
        migrations.CreateModel(
            name="SourceObservationEvent",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("provider", models.CharField(max_length=40)),
                ("kind", models.CharField(max_length=40)),
                ("subject", models.CharField(max_length=120)),
                ("content_sha256", models.CharField(max_length=64)),
                ("observed_at", models.DateTimeField()),
                ("recorded_at", models.DateTimeField(auto_now_add=True)),
                (
                    "source_asset",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="observation_events",
                        to="data.dataasset",
                    ),
                ),
            ],
            options={
                "ordering": ["provider", "kind", "subject", "observed_at"],
            },
        ),
        migrations.AddIndex(
            model_name="sourceobservationevent",
            index=models.Index(
                fields=["source_asset", "observed_at"],
                name="observation_asset_lookup",
            ),
        ),
        migrations.AddConstraint(
            model_name="sourceobservationevent",
            constraint=models.UniqueConstraint(
                fields=("provider", "kind", "subject", "observed_at"),
                name="unique_source_observation_instant",
            ),
        ),
        migrations.RunPython(protect_observation_events, unprotect_observation_events),
    ]

import hashlib
import json
import uuid

import django.db.models.deletion
from django.db import migrations, models

PROTECTED_MODELS = {
    "DataAsset": "data_dataasset",
    "FundamentalFact": "data_fundamentalfact",
    "FxRate": "data_fxrate",
    "CompanyClassificationObservation": "data_companyclassificationobservation",
}
POSTGRES_FUNCTION = "stanstock_prevent_evidence_mutation"


def _table_names(schema_editor):
    return set(schema_editor.connection.introspection.table_names())


def unprotect_evidence(apps, schema_editor):
    vendor = schema_editor.connection.vendor
    tables = _table_names(schema_editor)
    if vendor == "sqlite":
        for model_name in PROTECTED_MODELS:
            slug = model_name.lower()
            schema_editor.execute(f"DROP TRIGGER IF EXISTS data_{slug}_prevent_update")
            schema_editor.execute(f"DROP TRIGGER IF EXISTS data_{slug}_prevent_delete")
        return
    if vendor == "postgresql":
        for model_name, table_name in PROTECTED_MODELS.items():
            if table_name not in tables:
                continue
            table = schema_editor.quote_name(table_name)
            trigger = f"data_{model_name.lower()}_prevent_mutation"
            schema_editor.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
        schema_editor.execute(f"DROP FUNCTION IF EXISTS {POSTGRES_FUNCTION}()")
        return
    raise RuntimeError(f"Unsupported database vendor for evidence immutability: {vendor}")


def protect_evidence(apps, schema_editor):
    vendor = schema_editor.connection.vendor
    tables = _table_names(schema_editor)
    if vendor == "sqlite":
        for model_name, table_name in PROTECTED_MODELS.items():
            if table_name not in tables:
                continue
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
            if table_name not in tables:
                continue
            table = schema_editor.quote_name(table_name)
            trigger = f"data_{model_name.lower()}_prevent_mutation"
            schema_editor.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
            schema_editor.execute(
                f"""
                CREATE TRIGGER {trigger}
                BEFORE UPDATE OR DELETE ON {table}
                FOR EACH ROW EXECUTE FUNCTION {POSTGRES_FUNCTION}()
                """
            )
        return
    raise RuntimeError(f"Unsupported database vendor for evidence immutability: {vendor}")


def _period_identity(period_type, period_start, period_end, fiscal_period, frame):
    parts = [
        period_type,
        period_start.isoformat() if period_start is not None else "",
        period_end.isoformat(),
    ]
    if period_type == "unclassified":
        parts.extend(((fiscal_period or "").strip().upper(), (frame or "").strip().upper()))
    return ":".join(parts)


def _observation_hash(fact, period_identity, taxonomy, filing_date):
    payload = {
        "taxonomy": taxonomy,
        "source_concept": fact.source_concept,
        "value": str(fact.value),
        "unit": fact.unit,
        "currency": fact.currency,
        "period_identity": period_identity,
        "fiscal_year": fact.fiscal_year,
        "fiscal_period": fact.fiscal_period,
        "accession": fact.accession,
        "filing_form": "",
        "filing_date": filing_date.isoformat() if filing_date is not None else None,
        "acceptance_at": None,
        "frame": "",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def backfill_fact_identity(apps, schema_editor):
    FundamentalFact = apps.get_model("data", "FundamentalFact")
    for fact in FundamentalFact.objects.all().iterator():
        period_type = "duration" if fact.period_start is not None else "instant"
        period_identity = _period_identity(
            period_type,
            fact.period_start,
            fact.period_end,
            fact.fiscal_period,
            "",
        )
        taxonomy = fact.source_concept.split(":", 1)[0] if ":" in fact.source_concept else ""
        filing_date = fact.filed_at.date() if fact.filed_at is not None else None
        FundamentalFact.objects.filter(pk=fact.pk).update(
            taxonomy=taxonomy,
            period_type=period_type,
            period_identity=period_identity,
            filing_date=filing_date,
            availability_basis=(
                "legacy_filing_delay" if fact.filed_at is not None else "legacy_source_asset"
            ),
            observation_hash=_observation_hash(
                fact,
                period_identity,
                taxonomy,
                filing_date,
            ),
        )


class Migration(migrations.Migration):
    dependencies = [
        ("data", "0005_add_etf_security_type"),
    ]

    operations = [
        migrations.RunPython(unprotect_evidence, protect_evidence),
        migrations.CreateModel(
            name="CompanyClassificationObservation",
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
                ("scheme", models.CharField(max_length=32)),
                ("code", models.CharField(max_length=32)),
                ("description", models.CharField(blank=True, max_length=240)),
                ("observed_at", models.DateTimeField()),
                ("available_at", models.DateTimeField()),
                ("ingested_at", models.DateTimeField(auto_now_add=True)),
                ("accession", models.CharField(blank=True, max_length=80)),
                ("quality_flags", models.JSONField(blank=True, default=list)),
                (
                    "company",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="classification_observations",
                        to="data.company",
                    ),
                ),
                (
                    "source_asset",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        to="data.dataasset",
                    ),
                ),
            ],
            options={
                "ordering": ["company", "scheme", "-available_at"],
            },
        ),
        migrations.RemoveConstraint(
            model_name="fundamentalfact",
            name="unique_fundamental_vintage",
        ),
        migrations.AddField(
            model_name="fundamentalfact",
            name="acceptance_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="fundamentalfact",
            name="availability_basis",
            field=models.CharField(default="legacy", max_length=40),
        ),
        migrations.AddField(
            model_name="fundamentalfact",
            name="filing_date",
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="fundamentalfact",
            name="filing_form",
            field=models.CharField(blank=True, max_length=16),
        ),
        migrations.AddField(
            model_name="fundamentalfact",
            name="frame",
            field=models.CharField(blank=True, max_length=32),
        ),
        migrations.AddField(
            model_name="fundamentalfact",
            name="observation_hash",
            field=models.CharField(default="", max_length=64),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="fundamentalfact",
            name="period_identity",
            field=models.CharField(default="", max_length=160),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="fundamentalfact",
            name="period_type",
            field=models.CharField(
                choices=[
                    ("instant", "Instant"),
                    ("duration", "Duration"),
                    ("unclassified", "Unclassified"),
                ],
                default="unclassified",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="fundamentalfact",
            name="source_revision",
            field=models.PositiveIntegerField(default=1),
        ),
        migrations.AddField(
            model_name="fundamentalfact",
            name="taxonomy",
            field=models.CharField(blank=True, max_length=40),
        ),
        migrations.RunPython(backfill_fact_identity, migrations.RunPython.noop),
        migrations.AddConstraint(
            model_name="fundamentalfact",
            constraint=models.UniqueConstraint(
                fields=(
                    "company",
                    "provider",
                    "source_concept",
                    "period_identity",
                    "accession",
                    "unit",
                    "source_revision",
                ),
                name="unique_fundamental_vintage",
            ),
        ),
        migrations.AddConstraint(
            model_name="fundamentalfact",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    ("acceptance_at__isnull", True),
                    ("available_at__gte", models.F("acceptance_at")),
                    _connector="OR",
                ),
                name="fact_available_after_acceptance",
            ),
        ),
        migrations.AddConstraint(
            model_name="fundamentalfact",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    ("period_type", "unclassified"),
                    models.Q(("period_start__isnull", True), ("period_type", "instant")),
                    models.Q(
                        ("period_start__isnull", False),
                        ("period_start__lte", models.F("period_end")),
                        ("period_type", "duration"),
                    ),
                    _connector="OR",
                ),
                name="fact_period_type_consistent",
            ),
        ),
        migrations.AddConstraint(
            model_name="fundamentalfact",
            constraint=models.CheckConstraint(
                condition=~models.Q(("period_identity", ""))
                & ~models.Q(("observation_hash", "")),
                name="fact_identity_present",
            ),
        ),
        migrations.AddIndex(
            model_name="companyclassificationobservation",
            index=models.Index(
                fields=["company", "scheme", "available_at"],
                name="classification_asof_lookup",
            ),
        ),
        migrations.AddConstraint(
            model_name="companyclassificationobservation",
            constraint=models.UniqueConstraint(
                fields=("company", "provider", "scheme", "code", "source_asset"),
                name="unique_company_classification_observation",
            ),
        ),
        migrations.AddConstraint(
            model_name="companyclassificationobservation",
            constraint=models.CheckConstraint(
                condition=models.Q(("available_at__gte", models.F("observed_at"))),
                name="classification_available_after_observed",
            ),
        ),
        migrations.RunPython(protect_evidence, unprotect_evidence),
    ]

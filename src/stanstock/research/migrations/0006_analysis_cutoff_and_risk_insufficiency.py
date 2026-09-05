from django.db import migrations, models
from django.db.models import F, Q


def populate_analysis_cutoff(apps, schema_editor):
    AnalysisRun = apps.get_model("research", "AnalysisRun")
    for run in AnalysisRun.objects.filter(data_cutoff__isnull=True).iterator():
        AnalysisRun.objects.filter(pk=run.pk).update(data_cutoff=run.generated_at)


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0005_reinstate_prediction_immutability"),
    ]

    operations = [
        migrations.AddField(
            model_name="analysisrun",
            name="data_cutoff",
            field=models.DateTimeField(null=True),
        ),
        migrations.RunPython(populate_analysis_cutoff, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="analysisrun",
            name="data_cutoff",
            field=models.DateTimeField(),
        ),
        migrations.AddConstraint(
            model_name="analysisrun",
            constraint=models.CheckConstraint(
                condition=Q(data_cutoff__lte=F("generated_at")),
                name="analysis_cutoff_before_generated",
            ),
        ),
        migrations.AlterField(
            model_name="stockanalysis",
            name="risk_class",
            field=models.CharField(
                choices=[
                    ("low", "LOW"),
                    ("medium", "MEDIUM"),
                    ("high", "HIGH"),
                    ("very_high", "VERY HIGH"),
                    ("insufficient", "INSUFFICIENT EVIDENCE"),
                ],
                max_length=12,
            ),
        ),
        migrations.AlterField(
            model_name="stockanalysis",
            name="risk_score",
            field=models.DecimalField(
                decimal_places=2,
                max_digits=6,
                null=True,
            ),
        ),
        migrations.AddConstraint(
            model_name="predictionoutcome",
            constraint=models.CheckConstraint(
                condition=(
                    ~Q(status="corporate_event")
                    | Q(
                        actual_return__isnull=True,
                        benchmark_return__isnull=True,
                        success__isnull=True,
                        error__isnull=True,
                    )
                ),
                name="outcome_corporate_event_nulls",
            ),
        ),
    ]

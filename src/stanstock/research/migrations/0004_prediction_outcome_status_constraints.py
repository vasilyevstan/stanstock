from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0003_prediction_immutable"),
    ]

    operations = [
        migrations.AddConstraint(
            model_name="prediction",
            constraint=models.CheckConstraint(
                condition=models.Q(price_at_prediction__gt=0),
                name="prediction_price_positive",
            ),
        ),
        migrations.AddConstraint(
            model_name="prediction",
            constraint=models.CheckConstraint(
                condition=models.Q(data_cutoff__lte=models.F("generated_at")),
                name="prediction_cutoff_before_generated",
            ),
        ),
        migrations.AddConstraint(
            model_name="prediction",
            constraint=models.CheckConstraint(
                condition=(
                    (
                        models.Q(bear_return__isnull=True)
                        | models.Q(bear_return__gte=-1.0)
                    )
                    & (
                        models.Q(base_return__isnull=True)
                        | models.Q(base_return__gte=-1.0)
                    )
                    & (
                        models.Q(bull_return__isnull=True)
                        | models.Q(bull_return__gte=-1.0)
                    )
                ),
                name="prediction_return_lower_bound",
            ),
        ),
        migrations.AddConstraint(
            model_name="predictionoutcome",
            constraint=models.CheckConstraint(
                condition=(
                    ~models.Q(status="matured")
                    | (models.Q(actual_return__isnull=False) & models.Q(success__isnull=False))
                ),
                name="outcome_matured_actual_success",
            ),
        ),
        migrations.AddConstraint(
            model_name="predictionoutcome",
            constraint=models.CheckConstraint(
                condition=(
                    ~models.Q(status="unresolved")
                    | models.Q(
                        actual_return__isnull=True,
                        benchmark_return__isnull=True,
                        success__isnull=True,
                        error__isnull=True,
                    )
                ),
                name="outcome_unresolved_nulls",
            ),
        ),
    ]

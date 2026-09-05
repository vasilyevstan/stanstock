from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("simulation", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="simulationtrade",
            name="side",
            field=models.CharField(
                choices=[("buy", "Buy"), ("sell", "Sell")],
                default="buy",
                max_length=4,
            ),
            preserve_default=False,
        ),
        migrations.AddConstraint(
            model_name="simulationrun",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    models.Q(
                        ("finished_at__isnull", True),
                        ("status", "running"),
                    ),
                    models.Q(
                        models.Q(("status", "running"), _negated=True),
                        ("finished_at__isnull", False),
                    ),
                    _connector="OR",
                ),
                name="simulation_finished_at_matches_status",
            ),
        ),
        migrations.AddConstraint(
            model_name="simulationrun",
            constraint=models.CheckConstraint(
                condition=(
                    ~models.Q(status="complete")
                    | ~models.Q(result_asset_key="")
                ),
                name="complete_simulation_has_result_asset",
            ),
        ),
        migrations.AddConstraint(
            model_name="simulationholding",
            constraint=models.CheckConstraint(
                condition=models.Q(quantity__gte=0),
                name="simulation_holding_quantity_nonnegative",
            ),
        ),
        migrations.AddConstraint(
            model_name="simulationholding",
            constraint=models.CheckConstraint(
                condition=models.Q(price__gte=0),
                name="simulation_holding_price_nonnegative",
            ),
        ),
        migrations.AddConstraint(
            model_name="simulationholding",
            constraint=models.CheckConstraint(
                condition=models.Q(market_value__gte=0),
                name="simulation_holding_value_nonnegative",
            ),
        ),
        migrations.AddConstraint(
            model_name="simulationholding",
            constraint=models.CheckConstraint(
                condition=models.Q(weight__gte=0, weight__lte=1),
                name="simulation_holding_weight_in_range",
            ),
        ),
        migrations.AddConstraint(
            model_name="simulationtrade",
            constraint=models.CheckConstraint(
                condition=models.Q(quantity__gt=0),
                name="simulation_trade_quantity_positive",
            ),
        ),
        migrations.AddConstraint(
            model_name="simulationtrade",
            constraint=models.CheckConstraint(
                condition=models.Q(price__gt=0),
                name="simulation_trade_price_positive",
            ),
        ),
        migrations.AddConstraint(
            model_name="simulationtrade",
            constraint=models.CheckConstraint(
                condition=models.Q(gross_value__gt=0),
                name="simulation_trade_value_positive",
            ),
        ),
        migrations.AddConstraint(
            model_name="simulationtrade",
            constraint=models.CheckConstraint(
                condition=models.Q(costs__gte=0),
                name="simulation_trade_costs_nonnegative",
            ),
        ),
    ]

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0002_snapshot_immutability'),
        ('research', '0007_analysis_run_issued_on_time'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name='portfolio',
            name='construction_metadata',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='portfolio',
            name='construction_policy',
            field=models.CharField(blank=True, max_length=80),
        ),
        migrations.AddField(
            model_name='portfolio',
            name='source_analysis_run',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.PROTECT, related_name='sample_portfolios', to='research.analysisrun'),
        ),
        migrations.AddField(
            model_name='portfolio',
            name='starting_capital',
            field=models.DecimalField(blank=True, decimal_places=6, max_digits=24, null=True),
        ),
        migrations.AddConstraint(
            model_name='portfolio',
            constraint=models.CheckConstraint(condition=models.Q(('starting_capital__isnull', True), ('starting_capital__gt', 0), _connector='OR'), name='portfolio_starting_capital_positive'),
        ),
        migrations.AddConstraint(
            model_name='portfolio',
            constraint=models.UniqueConstraint(condition=models.Q(('archived_at__isnull', True), ('source_analysis_run__isnull', False)), fields=('owner', 'source_analysis_run'), name='unique_active_sample_portfolio_run'),
        ),
    ]

import django.db.models.deletion
import django.db.models.functions.text
import uuid
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0004_contribution_planner'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='TrackedSymbol',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('symbol', models.CharField(max_length=32)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='tracked_symbols', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['symbol', 'created_at'],
                'constraints': [models.UniqueConstraint(fields=('owner', 'symbol'), name='unique_owner_tracked_symbol'), models.CheckConstraint(condition=models.Q(models.Q(('symbol', ''), _negated=True), ('symbol', django.db.models.functions.text.Upper(django.db.models.functions.text.Trim(models.F('symbol'))))), name='tracked_symbol_normalized')],
            },
        ),
    ]

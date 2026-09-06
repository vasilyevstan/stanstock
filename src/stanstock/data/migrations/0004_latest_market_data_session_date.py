from zoneinfo import ZoneInfo

from django.db import migrations, models


def populate_session_dates(apps, schema_editor):
    LatestMarketData = apps.get_model("data", "LatestMarketData")
    rows = LatestMarketData.objects.select_related("source_asset", "listing").filter(
        session_date__isnull=True
    )
    for row in rows.iterator():
        session_date = row.source_asset.period_end
        if session_date is None and row.listing.region == "us":
            session_date = row.observed_at.astimezone(ZoneInfo("America/New_York")).date()
        if session_date is None:
            session_date = row.observed_at.date()
        LatestMarketData.objects.filter(pk=row.pk).update(session_date=session_date)


class Migration(migrations.Migration):
    dependencies = [
        ("data", "0003_immutable_evidence"),
    ]

    operations = [
        migrations.AddField(
            model_name="latestmarketdata",
            name="session_date",
            field=models.DateField(null=True),
        ),
        migrations.RunPython(populate_session_dates, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="latestmarketdata",
            name="session_date",
            field=models.DateField(),
        ),
    ]

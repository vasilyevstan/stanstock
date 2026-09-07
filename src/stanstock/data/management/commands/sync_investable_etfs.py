from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand, CommandError
from polars.exceptions import ComputeError

from stanstock.data.etfs import (
    INVESTABLE_US_ETF_SYMBOL,
    sync_investable_spy_from_asset,
)
from stanstock.data.models import DataAsset


class Command(BaseCommand):
    help = (
        "Create or refresh the investable SPY ETF listing from the latest "
        "persisted benchmark price asset without contacting a provider."
    )

    def handle(self, *args: object, **options: Any) -> None:
        asset = (
            DataAsset.objects.filter(
                provider="twelve_data",
                kind="price_history",
                subject=INVESTABLE_US_ETF_SYMBOL,
                period_end__isnull=False,
            )
            .order_by("-period_end", "-available_at", "-retrieved_at")
            .first()
        )
        if asset is None or asset.period_end is None:
            raise CommandError("No persisted Twelve Data SPY price asset is available.")
        try:
            listing = sync_investable_spy_from_asset(
                asset=asset,
                target_date=asset.period_end,
            )
        except (ComputeError, OSError, ValueError) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(
            self.style.SUCCESS(
                f"Synced {listing.ticker} ETF listing={listing.pk} "
                f"session={asset.period_end.isoformat()} asset={asset.pk}"
            )
        )

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from django.db import transaction

from stanstock.data.models import DataAsset, LatestMarketData, Listing


@transaction.atomic
def update_latest_market_data(
    *,
    listing: Listing,
    session_date: date,
    observed_at: datetime,
    close: Decimal,
    previous_close: Decimal | None,
    volume: int | None,
    source_asset: DataAsset,
) -> bool:
    locked_listing = Listing.objects.select_for_update().get(pk=listing.pk)
    existing = LatestMarketData.objects.select_for_update().filter(listing=locked_listing).first()
    if existing is not None and (
        existing.session_date > session_date
        or (existing.session_date == session_date and existing.observed_at >= observed_at)
    ):
        return False
    LatestMarketData.objects.update_or_create(
        listing=locked_listing,
        defaults={
            "observed_at": observed_at,
            "session_date": session_date,
            "close": close,
            "previous_close": previous_close,
            "volume": volume,
            "source_asset": source_asset,
        },
    )
    return True

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

import polars as pl
from django.db.models import QuerySet

from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset, FundamentalFact, FxRate

DATE_COLUMN = "date"


class PriceFrameSchemaError(ValueError):
    """Raised when a price frame's date column is missing or unparseable.

    Selecting an eligible `DataAsset` by `available_at`/`retrieved_at` is not
    sufficient on its own: the underlying Parquet frame can legitimately
    contain rows dated after the point-in-time decision (e.g. a bundle
    covering a whole calendar year, retrieved partway through it, or a bug
    that writes future placeholder rows). Rather than silently returning
    those future-dated rows -- or silently guessing at how to interpret an
    unexpected date schema -- `AsOfData.price_frame` raises this error
    explicitly whenever the frame has no `date` column, or the column's
    dtype cannot be unambiguously compared against a calendar date.
    """


class AsOfData:
    def __init__(self, decision_time: datetime, store: AssetStore | None = None) -> None:
        self.decision_time = decision_time
        self.store = store or AssetStore()

    def latest_asset(self, *, provider: str, kind: str, subject: str) -> DataAsset:
        return DataAsset.objects.filter(
            provider=provider,
            kind=kind,
            subject=subject,
            available_at__lte=self.decision_time,
            retrieved_at__lte=self.decision_time,
        ).latest("available_at", "retrieved_at")

    def price_frame(
        self,
        *,
        provider: str,
        subject: str,
        through_date: date | None = None,
    ) -> pl.DataFrame:
        """Return the latest eligible price frame, physically clipped to ``through_date``.

        ``through_date`` defaults to ``self.decision_time.date()``. An
        eligible `DataAsset` (selected via `latest_asset`, gated on
        `available_at`/`retrieved_at`) can still contain rows dated after
        this cutoff -- e.g. a multi-row bundle retrieved partway through its
        own period -- so every row with ``date > through_date`` is filtered
        out here rather than trusting asset-level eligibility alone.
        """
        asset = self.latest_asset(provider=provider, kind="price_history", subject=subject)
        frame = self.store.read_frame(asset.relative_path)
        cutoff = through_date if through_date is not None else self.decision_time.date()
        if cutoff > self.decision_time.date():
            raise ValueError(
                f"through_date ({cutoff.isoformat()}) cannot be after the as-of "
                f"decision date ({self.decision_time.date().isoformat()})"
            )
        return _clip_to_through_date(frame, cutoff, relative_path=asset.relative_path)

    def fundamental_facts(
        self,
        *,
        company_id: UUID,
        concepts: list[str] | None = None,
        available_through: datetime | None = None,
    ) -> QuerySet[FundamentalFact]:
        availability_cutoff = available_through or self.decision_time
        if availability_cutoff > self.decision_time:
            raise ValueError(
                "Fundamental availability cutoff cannot be after the as-of decision time"
            )
        queryset = FundamentalFact.objects.filter(
            company_id=company_id,
            available_at__lte=availability_cutoff,
            source_asset__available_at__lte=self.decision_time,
            source_asset__retrieved_at__lte=self.decision_time,
        )
        if concepts:
            queryset = queryset.filter(concept__in=concepts)
        return queryset.order_by("concept", "period_end", "available_at")

    def fx_rates(
        self,
        *,
        base_currency: str | None = None,
        quote_currency: str | None = None,
        observation_start: date | None = None,
        observation_end: date | None = None,
    ) -> QuerySet[FxRate]:
        """Return FX vintages knowable at the decision time, oldest vintage first.

        Every filter is optional so a caller that must derive a rate path it
        cannot name up front -- an inverse quote, or a cross through a pivot
        currency it has not yet chosen -- can still read through this one
        as-of gate instead of querying `FxRate` directly and losing the
        `available_at`/`retrieved_at` guarantees. ``observation_start`` and
        ``observation_end`` bound the *economic* observation window; they
        never widen availability.
        """
        queryset = FxRate.objects.filter(
            available_at__lte=self.decision_time,
            source_asset__available_at__lte=self.decision_time,
            source_asset__retrieved_at__lte=self.decision_time,
        )
        if base_currency is not None:
            queryset = queryset.filter(base_currency=base_currency)
        if quote_currency is not None:
            queryset = queryset.filter(quote_currency=quote_currency)
        if observation_start is not None:
            queryset = queryset.filter(observation_date__gte=observation_start)
        if observation_end is not None:
            queryset = queryset.filter(observation_date__lte=observation_end)
        return queryset.select_related("source_asset").order_by("observation_date", "available_at")


def _clip_to_through_date(
    frame: pl.DataFrame, through_date: date, *, relative_path: str
) -> pl.DataFrame:
    """Filter ``frame`` to rows whose ``date`` column is ``<= through_date``.

    Fails explicitly -- instead of returning the frame unfiltered -- when
    the ``date`` column is missing, or its dtype cannot be unambiguously
    interpreted as a calendar date, so a schema drift or bug can never
    silently leak future-dated rows past a point-in-time decision.
    """
    if DATE_COLUMN not in frame.columns:
        raise PriceFrameSchemaError(
            f"Price frame at {relative_path!r} has no {DATE_COLUMN!r} column "
            "(schema: "
            f"{frame.columns!r}); refusing to return unfiltered rows that may "
            "include dates after the as-of decision time."
        )

    dtype = frame.schema[DATE_COLUMN]
    if dtype == pl.Date:
        date_column = frame[DATE_COLUMN]
    elif isinstance(dtype, pl.Datetime):
        date_column = frame[DATE_COLUMN].dt.date()
    elif dtype in (pl.Utf8, pl.String):
        try:
            date_column = frame[DATE_COLUMN].str.strptime(pl.Date, "%Y-%m-%d", strict=True)
        except pl.exceptions.PolarsError as exc:
            raise PriceFrameSchemaError(
                f"Price frame at {relative_path!r} has a {DATE_COLUMN!r} column "
                f"of string dtype that could not be parsed as ISO dates "
                "(expected 'YYYY-MM-DD'); refusing to guess how to compare it "
                f"against {through_date.isoformat()}."
            ) from exc
    else:
        raise PriceFrameSchemaError(
            f"Price frame at {relative_path!r} has a {DATE_COLUMN!r} column of "
            f"unsupported dtype {dtype!r}; refusing to guess how to compare it "
            f"against {through_date.isoformat()}."
        )

    # Normalize the (possibly Datetime/string-typed) `date` column to
    # `pl.Date` *before* filtering, so the returned frame always carries a
    # `pl.Date` dtype in its `date` column -- not whatever dtype the
    # underlying bytes happened to use -- letting downstream consumers
    # safely compare it against plain Python `date` values without
    # re-parsing.
    normalized_frame = frame.with_columns(date_column.alias(DATE_COLUMN))
    return normalized_frame.filter(pl.col(DATE_COLUMN) <= through_date).sort(DATE_COLUMN)

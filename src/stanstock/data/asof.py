from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from uuid import UUID

import polars as pl
from django.db.models import Exists, OuterRef, Q, QuerySet

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.assets import AssetStore
from stanstock.data.models import (
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    FxRate,
)

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


class PriceFrameChecksumMismatchError(RuntimeError):
    """Raised when price-frame bytes do not match their registered checksum."""


@dataclass(frozen=True, slots=True)
class PriceFrameRead:
    """One selected immutable asset, its clipped frame, and date diagnostics.

    ``asset`` is the exact `DataAsset` selected before the Parquet read. It is
    returned with the frame so a caller cannot independently re-select a
    newer eligible immutable vintage and accidentally attribute one asset's
    frame to another asset's UUID/checksum.

    ``frame`` is byte-for-byte the same frame `AsOfData.price_frame` returns:
    normalized to a `pl.Date` `date` column, clipped to `date <= through_date`,
    and sorted. ``invalid_session_date_rows`` counts rows whose normalized
    ``date`` is null -- a session identity that could never be established --
    and is computed *before* the cutoff filter runs, so a row that is simply
    in the future (a valid date, just later than ``through_date``) is clipped
    without being counted here.
    """

    asset: DataAsset
    frame: pl.DataFrame
    invalid_session_date_rows: int


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

        Delegates to `price_frame_with_diagnostics` and returns only its
        frame: this method's signature and return type stay exactly as they
        were, including for a frame that silently dropped a null-dated row
        long before this diagnostic existed.
        """
        return self.price_frame_with_diagnostics(
            provider=provider,
            subject=subject,
            through_date=through_date,
        ).frame

    def price_frame_with_diagnostics(
        self,
        *,
        provider: str,
        subject: str,
        through_date: date | None = None,
    ) -> PriceFrameRead:
        """`price_frame`, its exact selected asset, and unusable-date count.

        The eligible asset is selected exactly once. Its immutable row is
        returned beside the frame read from that row's ``relative_path``;
        callers must use this returned asset for provenance rather than issue
        a second selection. Every other behavior -- the
        `date <= through_date` cutoff, sorting, and every raised error -- is
        identical to `price_frame`; this is the same read, not a second
        Parquet read.
        """
        asset = self.latest_asset(provider=provider, kind="price_history", subject=subject)
        cutoff = through_date if through_date is not None else self.decision_time.date()
        if cutoff > self.decision_time.date():
            raise ValueError(
                f"through_date ({cutoff.isoformat()}) cannot be after the as-of "
                f"decision date ({self.decision_time.date().isoformat()})"
            )
        frame = self.store.read_frame(asset.relative_path)
        return _clip_to_through_date(
            frame,
            cutoff,
            asset=asset,
            relative_path=asset.relative_path,
        )

    def price_frame_for_asset_with_diagnostics(
        self,
        *,
        asset: DataAsset,
        through_date: date | None = None,
    ) -> PriceFrameRead:
        """Read one explicitly supplied immutable price asset.

        Unlike :meth:`price_frame_with_diagnostics`, this method performs no
        asset selection and no ORM query. It validates the supplied row
        against this instance's decision boundary, then reads that row's
        exact path through :class:`AssetStore`. The confined payload is read
        once, verified against the row's checksum, and parsed from those same
        in-memory bytes. This is the replay boundary for a manifest that
        already names a specific immutable vintage: a newer eligible asset
        for the same provider/subject can never replace the supplied one.
        """
        if asset.kind != "price_history":
            raise ValueError(
                f"Data asset {asset.pk} has kind {asset.kind!r}; expected 'price_history'"
            )
        for field in ("available_at", "retrieved_at"):
            timestamp = getattr(asset, field)
            if timestamp > self.decision_time:
                raise ValueError(
                    f"Price asset {asset.pk} has {field} after the as-of decision time"
                )
        cutoff = through_date if through_date is not None else self.decision_time.date()
        if cutoff > self.decision_time.date():
            raise ValueError(
                f"through_date ({cutoff.isoformat()}) cannot be after the as-of "
                f"decision date ({self.decision_time.date().isoformat()})"
            )
        payload = self.store.read_bytes(asset.relative_path)
        physical_sha256 = hashlib.sha256(payload).hexdigest()
        if physical_sha256 != asset.sha256:
            raise PriceFrameChecksumMismatchError(
                "Stored asset bytes do not match the registered SHA-256 checksum"
            )
        frame = pl.read_parquet(io.BytesIO(payload))
        return _clip_to_through_date(
            frame,
            cutoff,
            asset=asset,
            relative_path=asset.relative_path,
        )

    def fundamental_facts(
        self,
        *,
        company_id: UUID,
        concepts: list[str] | None = None,
        available_through: datetime | None = None,
    ) -> QuerySet[FundamentalFact]:
        return self._fundamental_facts(
            company_filter={"company_id": company_id},
            concepts=concepts,
            available_through=available_through,
        )

    def fundamental_facts_for_companies(
        self,
        *,
        company_ids: Iterable[UUID],
        concepts: list[str] | None = None,
        available_through: datetime | None = None,
    ) -> QuerySet[FundamentalFact]:
        return self._fundamental_facts(
            company_filter={"company_id__in": tuple(company_ids)},
            concepts=concepts,
            available_through=available_through,
        )

    def _fundamental_facts(
        self,
        *,
        company_filter: dict[str, object],
        concepts: list[str] | None,
        available_through: datetime | None,
    ) -> QuerySet[FundamentalFact]:
        availability_cutoff = available_through or self.decision_time
        if availability_cutoff > self.decision_time:
            raise ValueError(
                "Fundamental availability cutoff cannot be after the as-of decision time"
            )
        filing_evidence = FundamentalFactEvidence.objects.filter(
            fact_id=OuterRef("pk"),
            role=FundamentalFactEvidence.Role.FILING,
            source_asset__available_at__lte=self.decision_time,
            source_asset__retrieved_at__lte=self.decision_time,
        )
        queryset = (
            FundamentalFact.objects.filter(
                **company_filter,
                available_at__lte=availability_cutoff,
                source_asset__available_at__lte=self.decision_time,
                source_asset__retrieved_at__lte=self.decision_time,
            )
            .annotate(filing_evidence_visible=Exists(filing_evidence))
            .filter(~Q(provider="sec") | Q(filing_evidence_visible=True))
        )
        if concepts:
            queryset = queryset.filter(concept__in=concepts)
        return queryset.order_by(
            "company_id",
            "concept",
            "period_end",
            "available_at",
        )

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

    def company_classifications(
        self,
        *,
        company_id: UUID,
        scheme: str | None = None,
        available_through: datetime | None = None,
    ) -> QuerySet[CompanyClassificationObservation]:
        return self._company_classifications(
            company_filter={"company_id": company_id},
            scheme=scheme,
            available_through=available_through,
        )

    def company_classifications_for_companies(
        self,
        *,
        company_ids: Iterable[UUID],
        scheme: str | None = None,
        available_through: datetime | None = None,
    ) -> QuerySet[CompanyClassificationObservation]:
        return self._company_classifications(
            company_filter={"company_id__in": tuple(company_ids)},
            scheme=scheme,
            available_through=available_through,
        )

    def _company_classifications(
        self,
        *,
        company_filter: dict[str, object],
        scheme: str | None,
        available_through: datetime | None,
    ) -> QuerySet[CompanyClassificationObservation]:
        availability_cutoff = available_through or self.decision_time
        if availability_cutoff > self.decision_time:
            raise ValueError(
                "Classification availability cutoff cannot be after the as-of decision time"
            )
        queryset = CompanyClassificationObservation.objects.filter(
            **company_filter,
            available_at__lte=availability_cutoff,
            source_asset__available_at__lte=self.decision_time,
            source_asset__retrieved_at__lte=self.decision_time,
        )
        if scheme is not None:
            queryset = queryset.filter(scheme=scheme)
        return queryset.select_related("source_asset").order_by(
            "company_id",
            "scheme",
            "available_at",
            "pk",
        )


def raw_price_asset_for(price_asset: DataAsset, *, cutoff: datetime) -> DataAsset:
    """Resolve a normalized `price_history` asset's declared raw closure.

    A normalized parquet asset is only ever a derivation of a raw
    ``raw_price_history`` provider payload (ingestion always stamps
    ``metadata["raw_asset_id"]``/``["raw_sha256"]``); this proves that link
    genuinely resolves to a checksum-matching, identity-matching,
    cutoff-eligible row rather than merely being present.
    """
    metadata = price_asset.metadata if isinstance(price_asset.metadata, dict) else {}
    raw_id = metadata.get("raw_asset_id")
    raw_sha256 = metadata.get("raw_sha256")
    if not raw_id or not raw_sha256:
        raise RefreshVerificationError(
            "price_asset_raw_link_missing",
            "A verified normalized price asset does not declare its own upstream raw asset",
        )
    try:
        raw_uuid = UUID(str(raw_id))
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "price_asset_raw_link_malformed",
            "A verified normalized price asset's raw asset link is not a valid identifier",
        ) from exc
    raw_asset = DataAsset.objects.filter(pk=raw_uuid, sha256=str(raw_sha256)).first()
    if raw_asset is None:
        raise RefreshVerificationError(
            "price_asset_raw_asset_missing",
            "A verified normalized price asset's upstream raw asset could not be resolved "
            "with the recorded checksum",
        )
    if (
        raw_asset.provider != price_asset.provider
        or raw_asset.kind != "raw_price_history"
        or raw_asset.subject != price_asset.subject
    ):
        raise RefreshVerificationError(
            "price_asset_raw_asset_identity_mismatch",
            "A verified normalized price asset's upstream raw asset is not the expected "
            "raw provider payload",
        )
    if raw_asset.available_at > cutoff or raw_asset.retrieved_at > cutoff:
        raise RefreshVerificationError(
            "price_asset_raw_asset_after_cutoff",
            "A verified normalized price asset's upstream raw asset was admitted after "
            "the claiming row's own cutoff",
        )
    return raw_asset


@dataclass(frozen=True, slots=True)
class VerifiedPriceFields:
    """A price asset's own checksummed, cutoff-safe target/previous close + volume."""

    close: Decimal
    previous_close: Decimal | None
    volume: int | None


def _finite_positive_close(raw: object, *, quantum: Decimal) -> Decimal:
    """Convert a writer-compatible `close` cell to a positive `Decimal`.

    Rejects non-float, non-finite (NaN/Inf), and non-positive values before
    ever attempting a `Decimal` conversion or quantize.
    """
    if isinstance(raw, bool) or not isinstance(raw, float):
        raise PriceFrameSchemaError(f"Price frame close value {raw!r} is not a finite float")
    if not math.isfinite(raw) or raw <= 0:
        raise PriceFrameSchemaError(f"Price frame close value {raw!r} is not finite/positive")
    try:
        return Decimal(str(raw)).quantize(quantum)
    except InvalidOperation as exc:
        raise PriceFrameSchemaError(
            f"Price frame close value {raw!r} could not be quantized"
        ) from exc


def _bound_volume(raw: object) -> int | None:
    """Validate a writer-compatible nullable `volume` cell.

    `volume` is written as Polars `Int64`; anything other than `None` or a
    non-negative Python `int` (bool is rejected even though it is an `int`
    subclass) is a schema violation, not a valid missing/negative value.
    """
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise PriceFrameSchemaError(f"Price frame volume value {raw!r} is not an integer")
    if raw < 0:
        raise PriceFrameSchemaError(f"Price frame volume value {raw!r} is negative")
    return raw


def verified_price_fields(
    asset: DataAsset, *, cutoff: datetime, target_date: date, close_places: int
) -> VerifiedPriceFields:
    """Re-read `asset` and extract its exact target-date/previous-session fields.

    Missing/unreadable files, corrupted-but-checksum-valid Parquet, a
    missing or duplicated target-date row, an invalid/duplicated session
    date, and a `close`/`volume` value outside the writer's own schema
    (wrong dtype, non-finite, non-positive close, negative volume) all fail
    path-free here, so a caller never falls back to trusting a mutable
    summary row.
    """
    try:
        read = AsOfData(cutoff).price_frame_for_asset_with_diagnostics(
            asset=asset, through_date=target_date
        )
        if read.invalid_session_date_rows != 0:
            raise PriceFrameSchemaError("Price frame has rows with an invalid session date")
        for column, dtype in (("close", pl.Float64), ("volume", pl.Int64)):
            if column not in read.frame.columns:
                raise PriceFrameSchemaError(f"Price frame has no {column!r} column")
            if read.frame.schema[column] != dtype:
                raise PriceFrameSchemaError(
                    f"Price frame column {column!r} has unexpected dtype "
                    f"{read.frame.schema[column]!r}"
                )
        if read.frame[DATE_COLUMN].n_unique() != read.frame.height:
            raise PriceFrameSchemaError("Price frame has duplicate session dates")
    except (ValueError, OSError, PriceFrameChecksumMismatchError, pl.exceptions.PolarsError):
        raise RefreshVerificationError(
            "latest_market_data_asset_unreadable",
            "A bound price asset could not be independently re-read and verified",
        ) from None
    target_rows = read.frame.filter(pl.col(DATE_COLUMN) == target_date)
    if target_rows.height != 1:
        raise RefreshVerificationError(
            "latest_market_data_row_missing",
            "The verified price asset has no exact target-date row",
        )
    target_row = target_rows.row(0, named=True)
    previous_rows = read.frame.filter(pl.col(DATE_COLUMN) < target_date)
    has_previous_row = previous_rows.height > 0
    previous_raw = previous_rows.row(-1, named=True)["close"] if has_previous_row else None
    quantum = Decimal(1).scaleb(-close_places)
    try:
        # A genuinely present prior session row's close is always validated
        # -- a null close there is a schema violation, not equivalent to
        # "no prior session exists" (only an empty `previous_rows` frame
        # may yield `previous_close=None`).
        previous_close = (
            _finite_positive_close(previous_raw, quantum=quantum) if has_previous_row else None
        )
        close = _finite_positive_close(target_row["close"], quantum=quantum)
        volume = _bound_volume(target_row["volume"])
    except PriceFrameSchemaError:
        raise RefreshVerificationError(
            "latest_market_data_asset_unreadable",
            "A bound price asset could not be independently re-read and verified",
        ) from None
    return VerifiedPriceFields(close=close, previous_close=previous_close, volume=volume)


def _clip_to_through_date(
    frame: pl.DataFrame,
    through_date: date,
    *,
    asset: DataAsset,
    relative_path: str,
) -> PriceFrameRead:
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
    # Counted here, before the cutoff filter: a null session identity can
    # never be a "future" row, so clipping must not be the thing that makes
    # it disappear uncounted. A row that is merely dated after `through_date`
    # is a normal, valid clip and is not counted.
    invalid_session_date_rows = normalized_frame[DATE_COLUMN].null_count()
    clipped = normalized_frame.filter(pl.col(DATE_COLUMN) <= through_date).sort(DATE_COLUMN)
    return PriceFrameRead(
        asset=asset,
        frame=clipped,
        invalid_session_date_rows=invalid_session_date_rows,
    )

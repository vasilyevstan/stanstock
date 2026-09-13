"""Immutable registration and fail-closed read model for product study evidence.

The retrospective price-product study itself is read-only. The public producer
calculates and registers a full-cohort study without accepting caller-authored
metrics. The performance page uses the separate, owner-bound HTTP read model.

No code here replays Monte Carlo paths on HTTP GET. Serving re-verifies the
registered canonical payload, its physical checksum, its exact source
``AnalysisRun`` identity, and the source run's own registered output proof.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Protocol
from uuid import UUID

from django.conf import settings
from django.db import transaction
from django.db.models import Count
from django.utils import timezone
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.verification_types import AssetRef, RefreshVerificationError
from stanstock.data.assets import (
    AssetStore,
    open_asset_store,
    read_checksummed_bytes,
    register_asset,
    resolve_asset_ref,
)
from stanstock.data.models import DataAsset, ProviderRecord, UniverseSnapshot
from stanstock.data.provider_policy import TWELVE_DATA_PROVIDER, validate_provider_usage
from stanstock.data.providers.exceptions import ProviderConfigurationError
from stanstock.data.research_product import PRODUCT_INTAKE_KIND, product_membership_payload
from stanstock.data.research_product_demo import DEMO_OWNER_ID
from stanstock.research.models import AnalysisRun, StockAnalysis
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_VERSION,
    PriceProductConfig,
    load_price_product_config,
)
from stanstock.research.price_product_study import (
    REPLAY_STUDY_SCHEMA,
    serialize_price_product_study,
    study_price_product_run,
)
from stanstock.research.product_pipeline import verify_price_product_output
from stanstock.research.product_reader import ProductVerificationSession

STUDY_EVIDENCE_KIND = "research_product_study_evidence"
STUDY_EVIDENCE_CONTRACT = "research-product-study-evidence@1"
_STUDY_SCOPE = "all_selected"
_PARTITION_ORDER = ("development", "validation", "final_holdout")
_HORIZON_ORDER = ("6m", "12m", "3y", "5y")
_BASELINE_ORDER = ("zero_log_drift_gaussian", "historical_log_drift_gaussian")
_METRIC_ORDER = (
    "median_absolute_error",
    "pinball_p20",
    "pinball_p50",
    "pinball_p80",
    "interval_width",
    "interval_inclusion",
    "interval_score",
)
_BASELINE_LABELS = {
    "zero_log_drift_gaussian": "Zero-log-drift Gaussian baseline",
    "historical_log_drift_gaussian": "Historical-log-drift Gaussian baseline",
}
_METRIC_LABELS = {
    "median_absolute_error": "Mean absolute error of the median forecast",
    "pinball_p20": "Pinball loss (p20)",
    "pinball_p50": "Pinball loss (p50)",
    "pinball_p80": "Pinball loss (p80)",
    "interval_width": "Interval width",
    "interval_inclusion": "Interval inclusion",
    "interval_score": "Interval score",
}
_LOWER_IS_BETTER_METRICS = {
    "median_absolute_error",
    "pinball_p20",
    "pinball_p50",
    "pinball_p80",
    "interval_score",
}

StudyReadStatus = Literal["available", "absent", "disabled", "integrity_failed", "unauthorized"]


class StudyViewer(Protocol):
    @property
    def is_authenticated(self) -> bool: ...

    @property
    def pk(self) -> object: ...


@dataclass(frozen=True, slots=True)
class ProductStudyComparisonRow:
    baseline_model: str
    baseline_label: str
    metric_name: str
    metric_label: str
    paired_observation_count: int
    paired_target_cohort_count: int
    candidate_average: Decimal | None
    baseline_average: Decimal | None
    mean_difference_candidate_minus_baseline: Decimal | None
    assessment: str
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class ProductStudyScopeView:
    partition: str
    partition_label: str
    horizon: str
    horizon_label: str
    target_cohort_count: int
    distinct_listing_count: int
    unavailable_count: int
    insufficient_reason: str | None
    rows: tuple[ProductStudyComparisonRow, ...]


@dataclass(frozen=True, slots=True)
class ProductStudyPartitionView:
    partition: str
    label: str
    scopes: tuple[ProductStudyScopeView, ...]


@dataclass(frozen=True, slots=True)
class ProductStudyRead:
    status: StudyReadStatus
    message: str
    source_target_date: date | None = None
    source_run_id: UUID | None = None
    source_generated_at: datetime | None = None
    report_generated_at: datetime | None = None
    source_snapshot_grade: str = ""
    evidence_label: str = ""
    studied_listing_count: int = 0
    source_run_listing_count: int = 0
    fixed_epoch: date | None = None
    validation_start: date | None = None
    validation_end_exclusive: date | None = None
    holdout_start: date | None = None
    holdout_complete_through: date | None = None
    disclosures: tuple[str, ...] = ()
    partitions: tuple[ProductStudyPartitionView, ...] = ()
    verification_code: str = ""

    @property
    def available(self) -> bool:
        return self.status == "available"


@dataclass(frozen=True, slots=True)
class _VerifiedStudyIdentity:
    config: PriceProductConfig
    report_generated_at: datetime
    source_run: AnalysisRun
    owner_id: str
    source_provider: str
    intake_ref: dict[str, str]
    source_target_date: date
    source_snapshot_grade: str
    studied_listing_count: int
    source_run_listing_count: int

    @property
    def subject(self) -> str:
        return _study_subject(self.owner_id, self.source_run.id)


def register_price_product_study(*, run: AnalysisRun, store: AssetStore) -> DataAsset:
    """Calculate and register the frozen full-cohort study, never caller-authored metrics."""
    existing = list(
        DataAsset.objects.filter(
            provider="stanstock",
            kind=STUDY_EVIDENCE_KIND,
            metadata__source_run_id=str(run.pk),
        )
    )
    if len(existing) > 1:
        raise ValueError("Registered price-product study evidence is ambiguous")
    if existing:
        asset = existing[0]
        document = _read_registered_document(asset=asset, store=store)
        _verify_registered_asset(
            asset=asset,
            document=document,
            store=store,
            expected_owner=str(asset.metadata["owner_id"]),
            expected_provider=str(asset.metadata["source_provider"]),
            expected_source_run=run,
        )
        return asset
    report = study_price_product_run(
        run=run, store=store, all_selected=True, report_generated_at=timezone.now()
    )
    return _register_price_product_study(report=report, store=store)


def _register_price_product_study(*, report: Mapping[str, Any], store: AssetStore) -> DataAsset:
    """Persist one canonical, all-selected retrospective report privately.

    The logical identity is stable on product version, source owner, source
    run, and the required ``all_selected`` study scope. A second identical
    study replay returns the existing immutable evidence even when only
    ``report_generated_at`` changed. A genuine new source run appends a new
    immutable asset, while a conflicting replay for the same source run fails
    explicitly instead of replacing history.
    """

    document = serialize_price_product_study(report, include_generated_at=True)
    verified = _verify_study_document(document=document, store=store)
    full_bytes = _canonical_json_bytes(document)
    full_sha256 = hashlib.sha256(full_bytes).hexdigest()
    logical_document = serialize_price_product_study(document, include_generated_at=False)
    logical_sha256 = hashlib.sha256(_canonical_json_bytes(logical_document)).hexdigest()
    protocol_sha256 = hashlib.sha256(
        _canonical_json_bytes(_mapping(document.get("protocol_identity"), "protocol identity"))
    ).hexdigest()
    existing = list(
        DataAsset.objects.filter(
            provider="stanstock",
            kind=STUDY_EVIDENCE_KIND,
            subject=verified.subject,
        )
    )
    if len(existing) > 1:
        raise ValueError("Registered price-product study evidence is ambiguous")
    if existing:
        existing_asset = existing[0]
        existing_document = _read_registered_document(asset=existing_asset, store=store)
        _verify_registered_asset(
            asset=existing_asset,
            document=existing_document,
            store=store,
            expected_owner=verified.owner_id,
            expected_provider=verified.source_provider,
        )
        existing_logical_sha256 = hashlib.sha256(
            _canonical_json_bytes(
                serialize_price_product_study(existing_document, include_generated_at=False)
            )
        ).hexdigest()
        if existing_logical_sha256 == logical_sha256:
            return existing_asset
        raise ValueError(
            "Registered price-product study evidence already exists for this owner, "
            "source run, and all-selected scope with different content"
        )

    relative_path = (
        f"research/studies/{PRODUCT_VERSION}/{verified.owner_id}/"
        f"{verified.source_run.id}/{_STUDY_SCOPE}-{logical_sha256[:12]}.json"
    )
    stored = store.write_bytes(relative_path, full_bytes)
    metadata = {
        "contract": STUDY_EVIDENCE_CONTRACT,
        "study_schema": REPLAY_STUDY_SCHEMA,
        "product_version": PRODUCT_VERSION,
        "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
        "scope": _STUDY_SCOPE,
        "owner_id": verified.owner_id,
        "source_provider": verified.source_provider,
        "source_run_id": str(verified.source_run.id),
        "source_target_date": verified.source_target_date.isoformat(),
        "source_snapshot_grade": verified.source_snapshot_grade,
        "logical_report_sha256": logical_sha256,
        "full_report_sha256": full_sha256,
        "protocol_identity_sha256": protocol_sha256,
        "studied_listing_count": verified.studied_listing_count,
        "source_run_listing_count": verified.source_run_listing_count,
        "intake_asset": verified.intake_ref,
    }
    try:
        with transaction.atomic():
            registered_at = timezone.now()
            return register_asset(
                provider="stanstock",
                kind=STUDY_EVIDENCE_KIND,
                subject=verified.subject,
                stored=stored,
                retrieved_at=registered_at,
                available_at=registered_at,
                metadata=metadata,
            )
    except Exception:
        if not DataAsset.objects.filter(relative_path=relative_path).exists():
            store.resolve(relative_path).unlink(missing_ok=True)
        raise


def read_registered_price_product_study(
    *,
    user: StudyViewer,
    store: AssetStore | None = None,
    verification: ProductVerificationSession | None = None,
) -> ProductStudyRead:
    """Return the exact registered retrospective study visible to ``user``."""

    if not settings.RESEARCH_PRODUCT_ENABLED:
        return ProductStudyRead(
            status="disabled", message="The active research profile is disabled."
        )
    if not user.is_authenticated:
        return ProductStudyRead(
            status="unauthorized",
            message="Sign in to view registered retrospective research evidence.",
        )
    try:
        asset_store = store or open_asset_store()
    except RefreshVerificationError as exc:
        return ProductStudyRead(
            status="integrity_failed",
            message="Registered retrospective evidence is unavailable because storage failed.",
            verification_code=exc.reason_code,
        )

    expected_owner = DEMO_OWNER_ID if settings.DEMO_MODE else str(user.pk)
    expected_provider = "synthetic_demo" if settings.DEMO_MODE else TWELVE_DATA_PROVIDER
    assets = DataAsset.objects.filter(
        provider="stanstock",
        kind=STUDY_EVIDENCE_KIND,
        subject__startswith=_study_subject_prefix(expected_owner),
    )
    asset = assets.order_by("-available_at", "-retrieved_at", "-id").first()
    if asset is None:
        return ProductStudyRead(
            status="absent",
            message=(
                "Registered retrospective evidence is not available for this account. "
                "Forecasting skill is not established."
            ),
        )
    duplicate_identity = (
        assets.values("subject")
        .annotate(asset_count=Count("id"))
        .filter(asset_count__gt=1)
        .exists()
        or assets.values("metadata__source_run_id")
        .annotate(asset_count=Count("id"))
        .filter(asset_count__gt=1)
        .exists()
    )
    if duplicate_identity:
        return ProductStudyRead(
            status="integrity_failed",
            message=(
                "Registered retrospective evidence is unavailable because its registry "
                "identity is ambiguous."
            ),
            verification_code="product_study_registry_ambiguous",
        )
    try:
        document = _read_registered_document(asset=asset, store=asset_store)
        verified = _verify_registered_asset(
            asset=asset,
            document=document,
            store=asset_store,
            expected_owner=expected_owner,
            expected_provider=expected_provider,
            verification=verification,
        )
        return _build_study_read(document=document, verified=verified)
    except ProviderConfigurationError:
        return ProductStudyRead(
            status="unauthorized",
            message=(
                "Registered retrospective evidence is unavailable because current display "
                "authorization does not permit it."
            ),
            verification_code="product_study_display_unauthorized",
        )
    except (RefreshVerificationError, KeyError, TypeError, ValueError) as exc:
        code = (
            exc.reason_code
            if isinstance(exc, RefreshVerificationError)
            else "product_study_evidence_invalid"
        )
        return ProductStudyRead(
            status="integrity_failed",
            message=(
                "Registered retrospective evidence is unavailable because its source "
                "identity or canonical payload could not be verified."
            ),
            verification_code=code,
        )


def _build_study_read(
    *,
    document: dict[str, Any],
    verified: _VerifiedStudyIdentity,
) -> ProductStudyRead:
    protocol = _mapping(document.get("protocol_identity"), "protocol identity")
    disclosures = _string_tuple(document.get("disclosures"), "disclosures")
    aggregates = _list_of_mappings(document.get("projection_aggregates"), "projection aggregates")
    comparisons = _list_of_mappings(
        document.get("paired_model_comparisons"),
        "paired model comparisons",
    )

    aggregate_by_scope: dict[tuple[str, str], dict[str, Any]] = {}
    for row in aggregates:
        if row.get("model_name") != FHS_METHOD_VERSION:
            continue
        key = (_string_value(row, "partition"), _string_value(row, "horizon"))
        aggregate_by_scope[key] = row

    comparison_by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        key = (_string_value(row, "partition"), _string_value(row, "horizon"))
        comparison_by_scope[key].append(row)

    partitions: list[ProductStudyPartitionView] = []
    for partition in _PARTITION_ORDER:
        scopes: list[ProductStudyScopeView] = []
        for horizon in _HORIZON_ORDER:
            aggregate = aggregate_by_scope.get((partition, horizon))
            if aggregate is None:
                raise ValueError("Registered study is missing an FHS aggregate scope")
            rows = _comparison_rows(comparison_by_scope.get((partition, horizon), []))
            scopes.append(
                ProductStudyScopeView(
                    partition=partition,
                    partition_label=_partition_label(partition),
                    horizon=horizon,
                    horizon_label=_label(horizon),
                    target_cohort_count=_int_value(aggregate, "target_cohort_count"),
                    distinct_listing_count=_int_value(aggregate, "distinct_listing_count"),
                    unavailable_count=_int_value(aggregate, "unavailable_count"),
                    insufficient_reason=_optional_string(aggregate.get("insufficient_reason")),
                    rows=rows,
                )
            )
        partitions.append(
            ProductStudyPartitionView(
                partition=partition,
                label=_partition_label(partition),
                scopes=tuple(scopes),
            )
        )

    return ProductStudyRead(
        status="available",
        message="Registered retrospective evidence verified.",
        source_target_date=verified.source_target_date,
        source_run_id=verified.source_run.id,
        source_generated_at=verified.source_run.generated_at,
        report_generated_at=verified.report_generated_at,
        source_snapshot_grade=verified.source_snapshot_grade,
        evidence_label=_string_value(protocol, "evidence_label"),
        studied_listing_count=verified.studied_listing_count,
        source_run_listing_count=verified.source_run_listing_count,
        fixed_epoch=_date_value(protocol, "fixed_epoch"),
        validation_start=_date_value(protocol, "validation_start"),
        validation_end_exclusive=_date_value(protocol, "validation_end_exclusive"),
        holdout_start=_date_value(protocol, "holdout_start"),
        holdout_complete_through=_date_value(protocol, "holdout_complete_through"),
        disclosures=disclosures,
        partitions=tuple(partitions),
        verification_code="verified",
    )


def _comparison_rows(rows: list[dict[str, Any]]) -> tuple[ProductStudyComparisonRow, ...]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        indexed[(_string_value(row, "baseline_model"), _string_value(row, "metric_name"))] = row
    comparisons: list[ProductStudyComparisonRow] = []
    for baseline in _BASELINE_ORDER:
        for metric_name in _METRIC_ORDER:
            key = (baseline, metric_name)
            if key not in indexed:
                raise ValueError("Registered study comparison scope is incomplete")
            row = indexed[key]
            delta = _optional_decimal(row.get("mean_difference_candidate_minus_baseline"))
            unavailable_reason = _optional_string(row.get("unavailable_reason"))
            comparisons.append(
                ProductStudyComparisonRow(
                    baseline_model=baseline,
                    baseline_label=_BASELINE_LABELS[baseline],
                    metric_name=metric_name,
                    metric_label=_METRIC_LABELS[metric_name],
                    paired_observation_count=_int_value(row, "paired_observation_count"),
                    paired_target_cohort_count=_int_value(row, "paired_target_cohort_count"),
                    candidate_average=_optional_decimal(row.get("candidate_average")),
                    baseline_average=_optional_decimal(row.get("baseline_average")),
                    mean_difference_candidate_minus_baseline=delta,
                    assessment=(
                        "Unavailable"
                        if unavailable_reason is not None
                        else _assessment(metric_name=metric_name, delta=delta)
                    ),
                    unavailable_reason=unavailable_reason,
                )
            )
    return tuple(comparisons)


def _assessment(*, metric_name: str, delta: Decimal | None) -> str:
    if delta is None:
        return "Unavailable"
    if delta == 0:
        return "Tied"
    if metric_name in _LOWER_IS_BETTER_METRICS:
        return "Candidate better" if delta < 0 else "Candidate worse"
    if metric_name == "interval_width":
        return "Narrower interval" if delta < 0 else "Wider interval"
    if metric_name == "interval_inclusion":
        return "Higher inclusion" if delta > 0 else "Lower inclusion"
    raise ValueError("Unknown retrospective comparison metric")


def _verify_registered_asset(
    *,
    asset: DataAsset,
    document: dict[str, Any],
    store: AssetStore,
    expected_owner: str,
    expected_provider: str,
    expected_source_run: AnalysisRun | None = None,
    verification: ProductVerificationSession | None = None,
) -> _VerifiedStudyIdentity:
    if asset.provider != "stanstock" or asset.kind != STUDY_EVIDENCE_KIND:
        raise ValueError("Registered price-product study asset has the wrong identity")
    metadata = _mapping(asset.metadata, "registered study metadata")
    expected_keys = {
        "contract",
        "study_schema",
        "product_version",
        "config_hash",
        "scope",
        "owner_id",
        "source_provider",
        "source_run_id",
        "source_target_date",
        "source_snapshot_grade",
        "logical_report_sha256",
        "full_report_sha256",
        "protocol_identity_sha256",
        "studied_listing_count",
        "source_run_listing_count",
        "intake_asset",
    }
    if set(metadata) != expected_keys:
        raise ValueError("Registered price-product study metadata has an invalid shape")
    if (
        metadata.get("contract") != STUDY_EVIDENCE_CONTRACT
        or metadata.get("study_schema") != REPLAY_STUDY_SCHEMA
        or metadata.get("product_version") != PRODUCT_VERSION
        or metadata.get("config_hash") != PRODUCT_EFFECTIVE_CONFIG_HASH
        or metadata.get("scope") != _STUDY_SCOPE
        or metadata.get("owner_id") != expected_owner
        or metadata.get("source_provider") != expected_provider
    ):
        raise ValueError("Registered price-product study metadata identity is invalid")

    verified = _verify_study_document(
        document=document,
        store=store,
        expected_owner=expected_owner,
        expected_provider=expected_provider,
        verification=verification,
    )
    if not (
        verified.report_generated_at <= asset.retrieved_at <= timezone.now()
        and verified.report_generated_at <= asset.available_at <= timezone.now()
    ):
        raise ValueError("Registered study availability does not match its actual chronology")
    full_sha256 = hashlib.sha256(_canonical_json_bytes(document)).hexdigest()
    logical_sha256 = hashlib.sha256(
        _canonical_json_bytes(serialize_price_product_study(document, include_generated_at=False))
    ).hexdigest()
    protocol_sha256 = hashlib.sha256(
        _canonical_json_bytes(_mapping(document.get("protocol_identity"), "protocol identity"))
    ).hexdigest()
    if (
        metadata.get("full_report_sha256") != full_sha256
        or metadata.get("logical_report_sha256") != logical_sha256
        or metadata.get("protocol_identity_sha256") != protocol_sha256
        or metadata.get("source_run_id") != str(verified.source_run.id)
        or metadata.get("source_target_date") != verified.source_target_date.isoformat()
        or metadata.get("source_snapshot_grade") != verified.source_snapshot_grade
        or metadata.get("studied_listing_count") != verified.studied_listing_count
        or metadata.get("source_run_listing_count") != verified.source_run_listing_count
        or metadata.get("intake_asset") != verified.intake_ref
        or asset.subject != _study_subject(expected_owner, verified.source_run.id)
    ):
        raise ValueError("Registered price-product study metadata diverges from its payload")
    if expected_source_run is not None and verified.source_run.id != expected_source_run.id:
        raise ValueError("Registered price-product study source run does not match the request")
    return verified


def _verify_study_document(
    *,
    document: dict[str, Any],
    store: AssetStore,
    expected_owner: str | None = None,
    expected_provider: str | None = None,
    verification: ProductVerificationSession | None = None,
) -> _VerifiedStudyIdentity:
    if document.get("schema") != REPLAY_STUDY_SCHEMA:
        raise ValueError("Price-product study schema is invalid")
    _string_value(_mapping(document.get("execution"), "study execution"), "code_revision")
    report_generated_at = _parse_aware_datetime(
        document.get("report_generated_at"),
        message="Price-product study report_generated_at is invalid",
    )
    config = load_price_product_config()
    if _mapping(document.get("config_identity"), "config identity") != _config_identity(config):
        raise ValueError("Price-product study config identity is invalid")
    if _mapping(document.get("protocol_identity"), "protocol identity") != _protocol_identity(
        config
    ):
        raise ValueError("Price-product study protocol identity is invalid")
    if _mapping(document.get("calendar_identity"), "calendar identity") != _calendar_identity(
        config
    ):
        raise ValueError("Price-product study calendar identity is invalid")

    source_run_raw = _mapping(document.get("source_run"), "source run")
    source_run_id = UUID(_string_value(source_run_raw, "id"))
    source_run = (
        AnalysisRun.objects.select_related("universe_snapshot").filter(pk=source_run_id).first()
    )
    if source_run is None:
        raise ValueError("Price-product study source run does not exist")
    if source_run_raw != _source_run_document(source_run):
        raise ValueError("Price-product study source run identity does not match the database")
    if not source_run.generated_at <= report_generated_at <= timezone.now():
        raise ValueError("Price-product study generation must follow its source and not be future")
    if (
        source_run.config_version != PRODUCT_VERSION
        or source_run.config_hash != PRODUCT_EFFECTIVE_CONFIG_HASH
    ):
        raise ValueError("Price-product study source run config is invalid")
    if verification is None:
        verify_price_product_output(run=source_run, store=store, replay=False)
    else:
        verification.verify(run=source_run, store=store)

    membership = product_membership_payload(source_run.universe_snapshot, store=store)
    source_decision_time = _parse_aware_datetime(
        membership.get("decision_time"),
        message="Price-product study source decision boundary is invalid",
    )
    if source_decision_time > source_run.generated_at:
        raise ValueError("Price-product study source decision boundary is after generation")
    intake_ref = _asset_ref_json(_mapping(membership.get("intake"), "captured intake"))
    intake_asset = resolve_asset_ref(AssetRef.from_json(intake_ref), cutoff=source_decision_time)
    if intake_asset.provider != "stanstock" or intake_asset.kind != PRODUCT_INTAKE_KIND:
        raise ValueError("Price-product study captured intake identity is invalid")
    intake = _json_mapping(
        read_checksummed_bytes(store, intake_asset),
        message="Price-product study captured intake is malformed",
    )
    if intake.get("product_version") != PRODUCT_VERSION:
        raise ValueError("Price-product study captured intake product identity is invalid")
    owner_id = _string_value(intake, "owner_id")
    source_provider = _string_value(intake, "source_provider")
    if source_provider not in {"synthetic_demo", TWELVE_DATA_PROVIDER}:
        raise ValueError("Price-product study source provider is invalid")
    if source_provider == "synthetic_demo":
        if source_run.universe_snapshot.grade != UniverseSnapshot.Grade.RESEARCH:
            raise ValueError("Synthetic study evidence must bind a research-grade source run")

    if expected_owner is not None and owner_id != expected_owner:
        raise ProviderConfigurationError("Registered study owner does not match the viewer")
    if expected_provider is not None and source_provider != expected_provider:
        raise ProviderConfigurationError("Registered study provider does not match the viewer mode")
    if source_provider == TWELVE_DATA_PROVIDER:
        record = ProviderRecord.objects.filter(provider=TWELVE_DATA_PROVIDER).first()
        if record is None:
            raise ProviderConfigurationError("Provider display authorization is absent")
        validate_provider_usage(record, owner_id=owner_id)
    elif expected_provider == TWELVE_DATA_PROVIDER:
        raise ProviderConfigurationError(
            "Private live mode cannot display synthetic study evidence"
        )
    if (
        source_provider == "synthetic_demo"
        and not settings.DEMO_MODE
        and expected_owner is not None
    ):
        raise ProviderConfigurationError("Synthetic study evidence is shared only in demo mode")

    studied_listing_ids = [
        str(listing_id)
        for listing_id in StockAnalysis.objects.filter(run=source_run)
        .order_by("listing_id")
        .values_list("listing_id", flat=True)
    ]
    scope = _mapping(document.get("scope"), "study scope")
    if (
        scope.get("all_selected") is not True
        or scope.get("requested_listing_ids") != []
        or scope.get("studied_listing_ids") != studied_listing_ids
        or scope.get("source_run_listing_count") != len(studied_listing_ids)
        or scope.get("studied_listing_count") != len(studied_listing_ids)
    ):
        raise ValueError("Price-product study must register the full all-selected source cohort")
    listing_documents = _list_of_mappings(document.get("listings"), "listing reports")
    listing_ids_from_report = sorted(
        _string_value(item, "listing_id") for item in listing_documents
    )
    if listing_ids_from_report != studied_listing_ids:
        raise ValueError("Price-product study listing scope diverges from the source run")

    return _VerifiedStudyIdentity(
        config=config,
        report_generated_at=report_generated_at,
        source_run=source_run,
        owner_id=owner_id,
        source_provider=source_provider,
        intake_ref=intake_ref,
        source_target_date=source_run.target_date,
        source_snapshot_grade=source_run.universe_snapshot.grade,
        studied_listing_count=len(studied_listing_ids),
        source_run_listing_count=len(studied_listing_ids),
    )


def _read_registered_document(*, store: AssetStore, asset: DataAsset) -> dict[str, Any]:
    return _json_mapping(
        read_checksummed_bytes(store, asset),
        message="Registered price-product study payload is malformed",
    )


def _config_identity(config: PriceProductConfig) -> dict[str, object]:
    return {
        "product_version": config.product_version,
        "effective_config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
        "payload_schema": config.payload_schema,
        "calendar": config.calendar,
        "currency": config.currency,
        "price_provider": config.price_provider,
        "benchmark_subject": config.benchmark_subject,
        "return_basis": config.return_basis,
        "dividends_included": config.dividends_included,
    }


def _protocol_identity(config: PriceProductConfig) -> dict[str, object]:
    return {
        "evidence_label": config.replay.evidence_label,
        "fixed_epoch": config.replay.fixed_epoch.isoformat(),
        "development_end_exclusive": config.replay.development_end_exclusive.isoformat(),
        "validation_start": config.replay.validation_start.isoformat(),
        "validation_end_exclusive": config.replay.validation_end_exclusive.isoformat(),
        "holdout_start": config.replay.holdout_start.isoformat(),
        "holdout_complete_through": config.replay.holdout_complete_through.isoformat(),
        "anchor_spacing": config.replay.anchor_spacing,
        "purge_partition_crossings": config.replay.purge_partition_crossings,
        "comparators": list(config.replay.comparators),
        "required_prior_returns": config.simulation.return_observations,
        "production_paths": config.simulation.production_paths,
        "diagnostic_paths": config.simulation.diagnostic_max_paths,
        "convergence_threshold_rule": "max(0.01 return, 0.02 * production_interval_width)",
        "metric_precision": {
            "compared_returns": "ledger_returns",
            "return_decimal_places": config.rounding.return_decimal_places,
            "rounding_mode": config.rounding.mode,
        },
    }


def _calendar_identity(config: PriceProductConfig) -> dict[str, object]:
    calendar = get_calendar(config.calendar)
    fixed_epoch = calendar.date_to_session(config.replay.fixed_epoch, direction="none")
    start = calendar.sessions_window(fixed_epoch, -config.simulation.return_observations)[0].date()
    sessions = tuple(
        session.date()
        for session in calendar.sessions_in_range(start, config.replay.holdout_complete_through)
    )
    return {
        "calendar": config.calendar,
        "first_session": sessions[0].isoformat(),
        "last_session": sessions[-1].isoformat(),
        "session_count": len(sessions),
    }


def _source_run_document(run: AnalysisRun) -> dict[str, object]:
    return {
        "id": str(run.id),
        "generated_at": run.generated_at.isoformat(),
        "data_cutoff": run.data_cutoff.isoformat(),
        "target_date": run.target_date.isoformat(),
        "issued_on_time": run.issued_on_time,
        "universe_snapshot_id": str(run.universe_snapshot_id),
        "snapshot_grade": run.universe_snapshot.grade,
        "config_version": run.config_version,
        "config_hash": run.config_hash,
        "code_revision": run.code_revision,
        "status": run.status,
    }


def _study_subject_prefix(owner_id: str) -> str:
    if not owner_id or ":" in owner_id or len(owner_id) > 36:
        raise ValueError("Price-product study owner identity is invalid")
    return f"{PRODUCT_VERSION}:{owner_id}:{_STUDY_SCOPE}:"


def _study_subject(owner_id: str, source_run_id: UUID) -> str:
    return f"{_study_subject_prefix(owner_id)}{source_run_id}"


def _canonical_json_bytes(document: Mapping[str, Any] | dict[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _read_json(payload: bytes, *, message: str) -> object:
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(message) from exc


def _json_mapping(payload: bytes, *, message: str) -> dict[str, Any]:
    value = _read_json(payload, message=message)
    if not isinstance(value, dict):
        raise ValueError(message)
    return value


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


def _list_of_mappings(value: object, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    rows: list[dict[str, Any]] = []
    for item in value:
        rows.append(_mapping(item, name))
    return rows


def _string_value(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    rows = tuple(str(item) for item in value)
    if any(not item for item in rows):
        raise ValueError(f"{name} must not contain empty strings")
    return rows


def _int_value(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Decimal value is invalid") from exc
    if not decimal.is_finite():
        raise ValueError("Decimal value is invalid")
    return decimal


def _date_value(mapping: Mapping[str, Any], key: str) -> date:
    return date.fromisoformat(_string_value(mapping, key))


def _parse_aware_datetime(value: object, *, message: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(message) from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ValueError(message)
    return parsed


def _asset_ref_json(value: Mapping[str, Any]) -> dict[str, str]:
    ref = AssetRef.from_json(value)
    return ref.to_json()


def _label(value: str) -> str:
    labels = {
        "6m": "6 months",
        "12m": "12 months",
        "3y": "3 years",
        "5y": "5 years",
    }
    return labels.get(value, value.replace("_", " ").title())


def _partition_label(value: str) -> str:
    return {
        "development": "Development anchors",
        "validation": "Validation anchors",
        "final_holdout": "Final holdout anchors",
    }.get(value, value.replace("_", " ").title())

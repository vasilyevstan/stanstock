"""Append-only evidence for the fixed-model FHS terminal path frequencies.

Registration is intentionally the only producer: callers provide an immutable
run and an asset store, never counts or a report.  HTTP readers only validate
registered bytes and their bindings; they do not replay simulations.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Protocol, cast
from uuid import UUID

from django.conf import settings
from django.db import OperationalError, connection, transaction
from django.db.models import QuerySet
from django.utils import timezone

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.assets import AssetStore, read_checksummed_bytes
from stanstock.data.models import DataAsset, ProviderRecord
from stanstock.data.provider_policy import (
    TWELVE_DATA_PROVIDER,
    ProviderConfigurationError,
    validate_provider_usage,
)
from stanstock.research.config import code_revision
from stanstock.research.models import AnalysisRun, Prediction
from stanstock.research.price_product import (
    PriceProductInputError,
    _withheld_projection,
    complete_input_hash,
    deterministic_seed,
    filter_historical_returns,
    projection_from_terminal_logs,
    simulate_fhs_terminal_logs,
)
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_VERSION,
    PriceProductConfig,
    load_price_product_config,
)
from stanstock.research.price_product_frequencies import (
    FREQUENCY_METHOD_VERSION,
    FREQUENCY_SCHEMA,
    BucketCounts,
    HorizonFrequencies,
    classify_terminal_log_returns,
    method_identity_sha256,
)
from stanstock.research.price_product_study import load_price_product_sources
from stanstock.research.product_pipeline import _jsonable, verify_price_product_output

FREQUENCY_EVIDENCE_KIND = "research_product_frequency_evidence"
FREQUENCY_EVIDENCE_CONTRACT = "research-product-frequency-evidence@1"
FrequencyReadStatus = Literal["available", "absent", "disabled", "integrity_failed", "unauthorized"]
_HORIZONS = ("6m", "12m", "3y", "5y")
_DEMO_OWNER_ID = "synthetic-demo"
_FREQUENCY_ROW_KEYS = {
    "horizon",
    "sessions",
    "path_count",
    "counts",
    "zero_drift_counts",
    "ledger_returns",
    "insufficiency_reason",
    "projection",
    "input_hash",
    "seed",
    "effective_config_hash",
    "method_identity_sha256",
    "source_assets",
}
_FREQUENCY_LISTING_KEYS = {"listing_id", "horizons"}
_PROJECTION_KEYS = {
    "horizon",
    "sessions",
    "quantile_levels",
    "central_model_mass",
    "raw_returns",
    "ledger_returns",
    "raw_prices",
    "ledger_prices",
    "zero_drift_raw_returns",
    "zero_drift_ledger_returns",
    "zero_drift_raw_prices",
    "zero_drift_ledger_prices",
    "insufficiency_reason",
}


class FrequencyViewer(Protocol):
    @property
    def is_authenticated(self) -> bool: ...

    @property
    def pk(self) -> object: ...


@dataclass(frozen=True, slots=True)
class RegisteredFrequency:
    listing_id: UUID
    frequency: HorizonFrequencies
    ledger_returns: tuple[Decimal, Decimal, Decimal] | None


@dataclass(frozen=True, slots=True)
class ProductFrequencyRead:
    status: FrequencyReadStatus
    message: str
    source_run_id: UUID | None = None
    derived_at: datetime | None = None
    own_deadline_met: bool = False
    frequencies: tuple[RegisteredFrequency, ...] = ()
    verification_code: str = ""

    @property
    def available(self) -> bool:
        return self.status == "available"


def register_product_frequencies(*, run: AnalysisRun, store: AssetStore) -> DataAsset:
    """Internally derive and append one frequency report for ``run``.

    The preliminary lookup deliberately precedes source verification and the
    simulation.  It makes a completed retry a byte-checking no-op.
    """

    return _register_product_frequencies(run=run, store=store, derivation_source="manual_backfill")


def _register_product_frequencies(
    *,
    run: AnalysisRun,
    store: AssetStore,
    derivation_source: Literal["manual_backfill", "scheduled_stage"],
) -> DataAsset:
    """Internal origin-bound writer used only by the scheduled child."""

    existing = _single_asset_for_run(run)
    if existing is not None:
        _validate_registered_asset(existing, store=store, expected_run=run, verify_source=True)
        return existing

    logical_document = _derive_document(
        run=run,
        store=store,
        derivation_source=derivation_source,
        execution_revision=code_revision(),
    )
    logical_sha256 = _logical_sha256(logical_document)
    subject = _subject(run)
    relative_path = f"research/frequencies/{PRODUCT_VERSION}/{run.id}/{logical_sha256[:16]}.json"
    try:
        with transaction.atomic():
            # PostgreSQL serializes concurrent registrations against this
            # immutable source row. SQLite obtains its write lock at insert;
            # an unavailable lock is loud rather than a duplicate report.
            locked_run = AnalysisRun.objects.select_for_update(of=("self",)).get(pk=run.pk)
            existing = _single_asset_for_run(locked_run)
            if existing is not None:
                _validate_registered_asset(
                    existing, store=store, expected_run=locked_run, verify_source=True
                )
                if existing.metadata.get("logical_report_sha256") != logical_sha256:
                    raise ValueError("Frequency evidence already exists with different content")
                return existing
            derived_at = timezone.now()
            _require_publication_time(run=locked_run, derived_at=derived_at)
            document = _published_document(
                logical_document,
                derived_at=derived_at,
            )
            payload = _canonical_bytes(document)
            metadata = _metadata(
                document=document,
                run=locked_run,
                derived_at=derived_at,
                logical_sha256=logical_sha256,
            )
            stored = store.write_bytes(relative_path, payload)
            asset = DataAsset.objects.create(
                provider="stanstock",
                kind=FREQUENCY_EVIDENCE_KIND,
                subject=subject,
                relative_path=stored.relative_path,
                sha256=stored.sha256,
                retrieved_at=derived_at,
                available_at=derived_at,
                schema_version=FREQUENCY_SCHEMA,
                metadata=metadata,
            )
            if _assets_for_run(locked_run).count() != 1:
                raise ValueError("Frequency evidence registry identity is ambiguous")
            return asset
    except OperationalError as exc:
        # SQLite cannot lock a missing registry row. Only a completed winner
        # is recoverable; an unrelated database error must remain loud.
        if not _is_expected_registration_contention(exc):
            raise
        return _committed_winner_after_contention(run=run, store=store)


def _committed_winner_after_contention(*, run: AnalysisRun, store: AssetStore) -> DataAsset:
    """Bounded SQLite-only recovery after its missing-row writer race.

    PostgreSQL's source-row lock serializes this operation. SQLite instead
    rejects the loser while the winner transaction is in flight, so re-query
    only after the failed transaction has exited. No caller retry is needed,
    and only the exact, checksum-validated winner is ever returned.
    """

    if connection.vendor != "sqlite":
        raise OperationalError("Frequency registration lock contention was not recoverable")
    deadline = time.monotonic() + 2.0
    while True:
        try:
            existing = _single_asset_for_run(run)
        except OperationalError as exc:
            if not _is_expected_registration_contention(exc):
                raise
            existing = None
        if existing is not None:
            _validate_registered_asset(existing, store=store, expected_run=run, verify_source=True)
            return existing
        if time.monotonic() >= deadline:
            raise OperationalError(
                "Frequency registration winner did not commit before retry deadline"
            )
        time.sleep(0.025)


def read_registered_product_frequencies(
    *,
    user: FrequencyViewer,
    run: AnalysisRun,
    decision_time: datetime,
    store: AssetStore,
) -> ProductFrequencyRead:
    """Read one exact-run report subject to current rights and an as-of cutoff."""

    if not settings.RESEARCH_PRODUCT_ENABLED:
        return ProductFrequencyRead("disabled", "Model-estimated probabilities are disabled.")
    if not user.is_authenticated:
        return ProductFrequencyRead(
            "unauthorized", "Sign in to view model-estimated probabilities."
        )
    if not _aware(decision_time):
        return ProductFrequencyRead(
            "integrity_failed",
            "Model-estimated probabilities are unavailable because their decision time is invalid.",
            verification_code="frequency_decision_time_invalid",
        )
    try:
        expected_owner, expected_provider = _current_authorization(user)
    except ProviderConfigurationError:
        return ProductFrequencyRead(
            "unauthorized",
            (
                "Model-estimated probabilities are unavailable because current "
                "display authorization does not permit them."
            ),
            verification_code="frequency_display_unauthorized",
        )
    assets = _assets_for_run(run)
    if assets.count() > 1:
        return ProductFrequencyRead(
            "integrity_failed",
            (
                "Model-estimated probabilities are unavailable because their "
                "registry identity is ambiguous."
            ),
            verification_code="frequency_registry_ambiguous",
        )
    asset = assets.filter(available_at__lte=decision_time).first()
    if asset is None:
        return ProductFrequencyRead(
            "absent",
            "Model-estimated probabilities were not derived at this decision time.",
            source_run_id=run.id,
            verification_code="frequency_not_derived_at_decision_time",
        )
    try:
        document = _validate_registered_asset(
            asset,
            store=store,
            expected_run=run,
            expected_owner=expected_owner,
            expected_provider=expected_provider,
            decision_time=decision_time,
            verify_source=True,
        )
        return _read_document(document)
    except ProviderConfigurationError:
        return ProductFrequencyRead(
            "unauthorized",
            "Current display authorization does not permit this frequency evidence.",
            verification_code="frequency_display_unauthorized",
        )
    except (RefreshVerificationError, KeyError, TypeError, ValueError):
        return ProductFrequencyRead(
            "integrity_failed",
            (
                "Model-estimated probabilities are unavailable because their "
                "registered evidence is invalid."
            ),
            source_run_id=run.id,
            verification_code="frequency_evidence_invalid",
        )


def verify_registered_product_frequencies(*, run: AnalysisRun, store: AssetStore) -> None:
    """Offline semantic verification used by the recovery command."""

    asset = _single_asset_for_run(run)
    if asset is None:
        raise ValueError("No registered frequency evidence exists for this run")
    document = _validate_registered_asset(asset, store=store, expected_run=run, verify_source=True)
    regenerated = _published_document(
        _derive_document(
            run=run,
            store=store,
            derivation_source=_derivation_source(document),
            execution_revision=_execution_revision(document),
        ),
        derived_at=_datetime_value(document, "derived_at"),
    )
    if _canonical_bytes(regenerated) != _canonical_bytes(document):
        raise ValueError("frequency_verify_logical_content_mismatch")


def _derive_document(
    *,
    run: AnalysisRun,
    store: AssetStore,
    derivation_source: Literal["manual_backfill", "scheduled_stage"],
    execution_revision: str,
) -> dict[str, Any]:
    """Recreate every count from independently verified immutable inputs."""

    verify_price_product_output(run=run, store=store, replay=False)
    config = load_price_product_config()
    if (
        run.config_version != PRODUCT_VERSION
        or run.config_hash != PRODUCT_EFFECTIVE_CONFIG_HASH
        or config.product_version != PRODUCT_VERSION
    ):
        raise ValueError("Frequency source run does not match the frozen product identity")
    loaded = load_price_product_sources(
        run=run, store=store, config=config, requested_listing_ids=()
    )
    rows = [
        _derive_listing(source=source, run=run, config=config, store=store) for source in loaded
    ]
    rows.sort(key=lambda item: item["listing_id"])
    if not rows:
        raise ValueError("Frequency source run has no admitted listings")
    source_execution = loaded[0].source_execution
    if any(source.source_execution != source_execution for source in loaded):
        raise ValueError("Frequency source run mixes execution identities")
    return {
        "schema": FREQUENCY_SCHEMA,
        "contract": FREQUENCY_EVIDENCE_CONTRACT,
        "method_version": FREQUENCY_METHOD_VERSION,
        "method_identity_sha256": method_identity_sha256(),
        "product_version": PRODUCT_VERSION,
        "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
        "path_count": config.simulation.production_paths,
        "owner_id": _owner_id(run=run, store=store),
        "source_provider": (
            "synthetic_demo" if source_execution.mode == "synthetic_demo" else "twelve_data"
        ),
        "source_run": {
            "id": str(run.id),
            "target_date": run.target_date.isoformat(),
            "generated_at": run.generated_at.isoformat(),
            "data_cutoff": run.data_cutoff.isoformat(),
            "snapshot_grade": run.universe_snapshot.grade,
            "code_revision": run.code_revision,
        },
        "execution": {"code_revision": execution_revision},
        "derivation_source": derivation_source,
        # No independent deadline proof is produced in this slice.
        "own_deadline_met": False,
        "rows": rows,
    }


def _published_document(
    logical: Mapping[str, Any],
    *,
    derived_at: datetime,
) -> dict[str, Any]:
    document = dict(logical)
    document["derived_at"] = derived_at.isoformat()
    return document


def _is_expected_registration_contention(error: OperationalError) -> bool:
    """Return true only for the backend lock failures that a retry may recover.

    In particular, do not turn arbitrary connection, schema, or query failures
    into an apparently successful registration merely because an earlier
    process has an evidence row.
    """

    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "database is locked",
            "database table is locked",
            "could not obtain lock",
            "lock not available",
            "deadlock detected",
            "could not serialize",
        )
    )


def _metadata(
    *,
    document: Mapping[str, Any],
    run: AnalysisRun,
    derived_at: datetime,
    logical_sha256: str,
) -> dict[str, object]:
    return {
        "contract": FREQUENCY_EVIDENCE_CONTRACT,
        "frequency_schema": FREQUENCY_SCHEMA,
        "method_version": FREQUENCY_METHOD_VERSION,
        "method_identity_sha256": method_identity_sha256(),
        "product_version": PRODUCT_VERSION,
        "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
        "source_run_id": str(run.id),
        "source_target_date": run.target_date.isoformat(),
        "source_snapshot_grade": run.universe_snapshot.grade,
        "source_code_revision": run.code_revision,
        "derivation_code_revision": _execution_revision(document),
        "derived_at": derived_at.isoformat(),
        "logical_report_sha256": logical_sha256,
        "owner_id": document["owner_id"],
        "source_provider": document["source_provider"],
        "path_count": document["path_count"],
    }


def _derive_listing(
    *, source: Any, run: AnalysisRun, config: PriceProductConfig, store: AssetStore
) -> dict[str, Any]:
    product_input = source.selection.product_input
    input_hash = complete_input_hash(product_input)
    seed = deterministic_seed(
        method_version=FHS_METHOD_VERSION,
        effective_config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
        listing_id=product_input.listing_id,
        target_date=product_input.target_date,
    )
    simulation = config.simulation
    forecast = _forecast_from_source(source, store=store)
    if forecast.get("seed") != seed or forecast.get("path_count") != simulation.production_paths:
        raise ValueError("Frequency source forecast seed or path count is invalid")
    if _artifact_input_hash(source, store=store) != input_hash:
        raise ValueError("Frequency source calculation input hash is invalid")
    predictions = {
        row.horizon: row
        for row in source.prediction_rows
        if row.method_version == FHS_METHOD_VERSION
    }
    if set(predictions) != set(_HORIZONS):
        raise ValueError("Frequency source has incomplete advisory prediction identity")
    withheld_reason = None
    try:
        filtered = filter_historical_returns(
            product_input.stock.closes,
            burn_in=simulation.filter_burn_in,
            variance_target_weight=simulation.variance_target_weight,
            variance_persistence=simulation.variance_persistence,
            innovation_weight=simulation.innovation_weight,
        )
        terminals = simulate_fhs_terminal_logs(
            filtered,
            seed=seed,
            horizons=tuple(sessions for _name, sessions in simulation.horizons),
            path_count=simulation.production_paths,
            diagnostic_max_paths=simulation.diagnostic_max_paths,
            variance_target_weight=simulation.variance_target_weight,
            variance_persistence=simulation.variance_persistence,
            innovation_weight=simulation.innovation_weight,
        )
    except PriceProductInputError as exc:
        terminals = None
        withheld_reason = exc.reason_code
    rows = []
    for index, (horizon, sessions) in enumerate(simulation.horizons):
        if terminals is None:
            if withheld_reason is None:
                raise ValueError("Frequency simulation is absent without a withholding reason")
            projected = _withheld_projection(
                horizon, sessions, simulation.quantiles, withheld_reason
            )
        else:
            projected = projection_from_terminal_logs(
                horizon=horizon,
                sessions=sessions,
                terminal_logs=terminals.with_drift[index],
                zero_drift_logs=terminals.zero_drift[index],
                target_close=product_input.stock.closes[-1],
                quantiles=simulation.quantiles,
                quantile_method=simulation.quantile_method,
                return_places=config.rounding.return_decimal_places,
                price_places=config.rounding.price_decimal_places,
            )
        original = _forecast_projection(forecast, horizon)
        projected_payload = _jsonable(asdict(projected))
        if original != projected_payload:
            raise ValueError("Frequency replay does not match the frozen projection payload")
        prediction = predictions[horizon]
        if _ledger_triplet(projected) != _prediction_triplet(prediction):
            raise ValueError("Frequency replay does not match the immutable prediction ledger")
        frequency = (
            HorizonFrequencies(
                horizon, sessions, simulation.production_paths, None, None, withheld_reason
            )
            if terminals is None
            else classify_terminal_log_returns(
                terminals.with_drift[index],
                horizon=horizon,
                sessions=sessions,
                path_count=simulation.production_paths,
                zero_drift_logs=terminals.zero_drift[index],
                insufficiency_reason=projected.insufficiency_reason,
            )
        )
        rows.append(
            _frequency_row(
                frequency,
                input_hash=input_hash,
                seed=seed,
                source=source,
                projected=projected,
                projection_payload=projected_payload,
            )
        )
    return {"listing_id": str(product_input.listing_id), "horizons": rows}


def _frequency_row(
    frequency: HorizonFrequencies,
    *,
    input_hash: str,
    seed: int,
    source: Any,
    projected: Any,
    projection_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "horizon": frequency.horizon,
        "sessions": frequency.sessions,
        "path_count": frequency.path_count,
        "counts": _counts_document(frequency.counts),
        "zero_drift_counts": _counts_document(frequency.zero_drift_counts),
        "ledger_returns": _ledger_document(_ledger_triplet(projected)),
        "insufficiency_reason": frequency.insufficiency_reason,
        "projection": projection_payload,
        "input_hash": input_hash,
        "seed": seed,
        "effective_config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
        "method_identity_sha256": method_identity_sha256(),
        "source_assets": source.selection.source_assets,
    }


def _forecast_from_source(source: Any, *, store: AssetStore) -> Mapping[str, Any]:
    raw = json.loads(read_checksummed_bytes(store, source.calculation_artifact))
    result = raw.get("result") if isinstance(raw, dict) else None
    forecast = result.get("forecast") if isinstance(result, dict) else None
    if not isinstance(forecast, dict):
        raise ValueError("Frequency source artifact forecast is malformed")
    return forecast


def _artifact_input_hash(source: Any, *, store: AssetStore) -> str:
    raw = json.loads(read_checksummed_bytes(store, source.calculation_artifact))
    value = raw.get("input_hash") if isinstance(raw, dict) else None
    if not isinstance(value, str):
        raise ValueError("Frequency source artifact input hash is malformed")
    return value


def _forecast_projection(forecast: Mapping[str, Any], horizon: str) -> Mapping[str, Any]:
    projections = forecast.get("projections")
    if not isinstance(projections, list):
        raise ValueError("Frequency source forecast projections are malformed")
    matches = [
        item for item in projections if isinstance(item, dict) and item.get("horizon") == horizon
    ]
    if len(matches) != 1:
        raise ValueError("Frequency source forecast horizon is ambiguous")
    return matches[0]


def _prediction_triplet(prediction: Prediction) -> tuple[Decimal, Decimal, Decimal] | None:
    if (
        prediction.bear_return is None
        or prediction.base_return is None
        or prediction.bull_return is None
    ):
        return None
    return prediction.bear_return, prediction.base_return, prediction.bull_return


def _ledger_triplet(projected: Any) -> tuple[Decimal, Decimal, Decimal] | None:
    triplet = projected.ledger_returns
    return None if triplet is None else (triplet.lower, triplet.median, triplet.upper)


def _ledger_triplet_from_raw(raw: Mapping[str, Any]) -> tuple[Decimal, Decimal, Decimal] | None:
    triplet = raw.get("ledger_returns")
    if triplet is None:
        return None
    if not isinstance(triplet, Mapping):
        raise ValueError("Frequency source ledger triplet is malformed")
    try:
        values = tuple(Decimal(str(triplet[name])) for name in ("lower", "median", "upper"))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Frequency source ledger triplet is malformed") from exc
    return values[0], values[1], values[2]


def _counts_document(counts: BucketCounts | None) -> dict[str, int] | None:
    if counts is None:
        return None
    return {
        "loss": counts.loss,
        "flat_to_20": counts.flat_to_20,
        "above_20": counts.above_20,
        "large_loss": counts.large_loss,
    }


def _ledger_document(value: tuple[Decimal, Decimal, Decimal] | None) -> dict[str, str] | None:
    if value is None:
        return None
    return {name: str(item) for name, item in zip(("lower", "median", "upper"), value, strict=True)}


def _owner_id(*, run: AnalysisRun, store: AssetStore) -> str:
    from stanstock.data.research_product import product_membership_payload

    membership = product_membership_payload(run.universe_snapshot, store=store)
    intake = membership.get("intake")
    if not isinstance(intake, Mapping):
        raise ValueError("Frequency source membership has no intake")
    # The price-study loader already binds this source; reuse its canonical
    # membership parser rather than selecting a current owner.
    from stanstock.core.verification_types import AssetRef
    from stanstock.data.assets import resolve_asset_ref

    decision = datetime.fromisoformat(str(membership["decision_time"]))
    asset = resolve_asset_ref(AssetRef.from_json(intake), cutoff=decision)
    payload = json.loads(read_checksummed_bytes(store, asset))
    owner = payload.get("owner_id") if isinstance(payload, dict) else None
    if not isinstance(owner, str) or not owner:
        raise ValueError("Frequency source owner identity is malformed")
    return owner


def _assets_for_run(run: AnalysisRun) -> QuerySet[DataAsset]:
    return DataAsset.objects.filter(
        provider="stanstock", kind=FREQUENCY_EVIDENCE_KIND, subject=_subject(run)
    )


def _single_asset_for_run(run: AnalysisRun) -> DataAsset | None:
    assets = list(_assets_for_run(run))
    if len(assets) > 1:
        raise ValueError("Registered frequency evidence is ambiguous")
    return assets[0] if assets else None


def _subject(run: AnalysisRun) -> str:
    return f"{PRODUCT_VERSION}:frequencies:{run.id}"


def _validate_registered_asset(
    asset: DataAsset,
    *,
    store: AssetStore,
    expected_run: AnalysisRun,
    expected_owner: str | None = None,
    expected_provider: str | None = None,
    decision_time: datetime | None = None,
    verify_source: bool = False,
) -> dict[str, Any]:
    payload = read_checksummed_bytes(store, asset)
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Frequency evidence bytes are malformed") from exc
    if not isinstance(document, dict):
        raise ValueError("Frequency evidence document is malformed")
    _validate_document_shape(document)
    derived_at = _datetime_value(document, "derived_at")
    _require_publication_time(run=expected_run, derived_at=derived_at)
    if decision_time is not None and asset.available_at > decision_time:
        raise ValueError("Frequency evidence was not available at the decision time")
    if (
        asset.provider != "stanstock"
        or asset.kind != FREQUENCY_EVIDENCE_KIND
        or asset.schema_version != FREQUENCY_SCHEMA
        or asset.sha256 != hashlib.sha256(payload).hexdigest()
        or asset.subject != _subject(expected_run)
        or asset.available_at != derived_at
        or asset.retrieved_at != derived_at
        or asset.metadata
        != _metadata(
            document=document,
            run=expected_run,
            derived_at=derived_at,
            logical_sha256=_logical_sha256(document),
        )
    ):
        raise ValueError("Frequency evidence metadata does not bind its document")
    if (
        document.get("schema") != FREQUENCY_SCHEMA
        or document.get("contract") != FREQUENCY_EVIDENCE_CONTRACT
        or document.get("method_version") != FREQUENCY_METHOD_VERSION
        or document.get("method_identity_sha256") != method_identity_sha256()
        or document.get("product_version") != PRODUCT_VERSION
        or document.get("config_hash") != PRODUCT_EFFECTIVE_CONFIG_HASH
        or document["source_run"]["id"] != str(expected_run.id)
    ):
        raise ValueError("Frequency evidence identity is invalid")
    source_run = document["source_run"]
    if (
        source_run.get("target_date") != expected_run.target_date.isoformat()
        or source_run.get("generated_at") != expected_run.generated_at.isoformat()
        or source_run.get("data_cutoff") != expected_run.data_cutoff.isoformat()
        or source_run.get("snapshot_grade") != expected_run.universe_snapshot.grade
        or source_run.get("code_revision") != expected_run.code_revision
        or document.get("path_count") != load_price_product_config().simulation.production_paths
    ):
        raise ValueError("Frequency evidence source-run binding is invalid")
    if document["owner_id"] != _owner_id(run=expected_run, store=store):
        raise ValueError("Frequency evidence owner does not bind its source intake")
    if expected_owner is not None and document.get("owner_id") != expected_owner:
        raise ProviderConfigurationError("Frequency evidence owner does not match")
    if expected_provider is not None and document.get("source_provider") != expected_provider:
        raise ProviderConfigurationError("Frequency evidence provider does not match")
    parsed = _read_document(document)
    _verify_reader_row_bindings(parsed=parsed, run=expected_run, document=document)
    if verify_source:
        verify_price_product_output(run=expected_run, store=store, replay=False)
    return document


def _verify_reader_row_bindings(
    *,
    parsed: ProductFrequencyRead,
    run: AnalysisRun,
    document: Mapping[str, Any],
) -> None:
    """Bind every report row to the independently immutable product ledger."""

    predictions = {
        (item.listing_id, item.horizon): item
        for item in Prediction.objects.filter(
            analysis__run=run,
            method_version=FHS_METHOD_VERSION,
            evidence_role=Prediction.EvidenceRole.ADVISORY,
        )
    }
    expected = set(predictions)
    actual = {(item.listing_id, item.frequency.horizon) for item in parsed.frequencies}
    if actual != expected:
        raise ValueError("Frequency evidence listing and horizon coverage is incomplete")
    raw_by_identity = {
        (UUID(str(listing["listing_id"])), str(horizon["horizon"])): horizon
        for listing in document["rows"]
        for horizon in listing["horizons"]
    }
    for item in parsed.frequencies:
        prediction = predictions[(item.listing_id, item.frequency.horizon)]
        raw = raw_by_identity[(item.listing_id, item.frequency.horizon)]
        calculation = prediction.calculation
        if (
            not isinstance(calculation, Mapping)
            or raw.get("input_hash") != calculation.get("input_hash")
            or raw.get("source_assets") != prediction.source_assets
            or raw.get("effective_config_hash") != PRODUCT_EFFECTIVE_CONFIG_HASH
            or raw.get("method_identity_sha256") != method_identity_sha256()
            or item.ledger_returns != _prediction_triplet(prediction)
            or document.get("source_provider") != prediction.price_provider
        ):
            raise ValueError("Frequency evidence row does not bind its immutable prediction")
        forecast = calculation.get("forecast")
        if (
            not isinstance(forecast, Mapping)
            or raw.get("seed") != forecast.get("seed")
            or raw.get("projection") != _forecast_projection(forecast, item.frequency.horizon)
        ):
            raise ValueError("Frequency evidence row seed does not bind the source forecast")


def _read_document(document: Mapping[str, Any]) -> ProductFrequencyRead:
    source_run = document.get("source_run")
    rows = document.get("rows")
    path_count = document.get("path_count")
    if (
        not isinstance(source_run, Mapping)
        or not isinstance(rows, list)
        or type(path_count) is not int
    ):
        raise ValueError("Frequency evidence shape is invalid")
    expected: set[tuple[str, str]] = set()
    values: list[RegisteredFrequency] = []
    for listing in rows:
        if not isinstance(listing, Mapping) or set(listing) != _FREQUENCY_LISTING_KEYS:
            raise ValueError("Frequency listing row is invalid")
        listing_id = UUID(str(listing["listing_id"]))
        horizons = listing.get("horizons")
        if not isinstance(horizons, list) or len(horizons) != len(_HORIZONS):
            raise ValueError("Frequency listing horizon coverage is incomplete")
        for raw in horizons:
            if not isinstance(raw, Mapping):
                raise ValueError("Frequency horizon row is invalid")
            horizon_value = raw.get("horizon")
            horizon = horizon_value if isinstance(horizon_value, str) else ""
            if horizon not in _HORIZONS or (str(listing_id), horizon) in expected:
                raise ValueError("Frequency horizon identity is ambiguous")
            if (
                set(raw) != _FREQUENCY_ROW_KEYS
                or not isinstance(raw.get("projection"), Mapping)
                or set(raw["projection"]) != _PROJECTION_KEYS
                or not isinstance(raw.get("input_hash"), str)
                or len(raw["input_hash"]) != 64
                or type(raw.get("seed")) is not int
                or raw.get("effective_config_hash") != PRODUCT_EFFECTIVE_CONFIG_HASH
                or raw.get("method_identity_sha256") != method_identity_sha256()
                or not isinstance(raw.get("source_assets"), list)
                or len(raw["source_assets"]) != 4
            ):
                raise ValueError("Frequency horizon source identity is invalid")
            expected.add((str(listing_id), horizon))
            frequency = HorizonFrequencies(
                horizon=horizon,
                sessions=_sessions_for_horizon(horizon, raw.get("sessions")),
                path_count=_positive_int(raw.get("path_count")),
                counts=_read_counts(raw.get("counts"), path_count=path_count),
                zero_drift_counts=_read_counts(raw.get("zero_drift_counts"), path_count=path_count),
                insufficiency_reason=_optional_reason(raw.get("insufficiency_reason")),
            )
            if frequency.path_count != path_count:
                raise ValueError("Frequency horizon denominator disagrees with report")
            if frequency.available == (frequency.insufficiency_reason is not None):
                raise ValueError("Frequency availability and reason disagree")
            ledger = _read_ledger(raw.get("ledger_returns"))
            if (frequency.available and ledger is None) or (
                not frequency.available and ledger is not None
            ):
                raise ValueError("Frequency ledger and availability disagree")
            values.append(RegisteredFrequency(listing_id, frequency, ledger))
    return ProductFrequencyRead(
        status="available",
        message="Registered model simulation shares verified.",
        source_run_id=UUID(str(source_run["id"])),
        derived_at=_datetime_value(document, "derived_at"),
        own_deadline_met=document.get("own_deadline_met") is True,
        frequencies=tuple(values),
        verification_code="verified",
    )


def _read_counts(raw: object, *, path_count: int) -> BucketCounts | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != {"loss", "flat_to_20", "above_20", "large_loss"}:
        raise ValueError("Frequency counts are malformed")
    try:
        values = tuple(raw[key] for key in ("loss", "flat_to_20", "above_20", "large_loss"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Frequency counts are malformed") from exc
    if any(type(value) is not int for value in values):
        raise ValueError("Frequency counts must be integers")
    counts = BucketCounts(*values)
    if (
        any(
            value < 0
            for value in (counts.loss, counts.flat_to_20, counts.above_20, counts.large_loss)
        )
        or counts.total != path_count
        or counts.large_loss > counts.loss
    ):
        raise ValueError("Frequency counts are inconsistent")
    return counts


def _read_ledger(raw: object) -> tuple[Decimal, Decimal, Decimal] | None:
    if raw is None:
        return None
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"lower", "median", "upper"}
        or any(not isinstance(value, str) for value in raw.values())
    ):
        raise ValueError("Frequency ledger returns are malformed")
    try:
        result = tuple(Decimal(str(raw[key])) for key in ("lower", "median", "upper"))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Frequency ledger returns are malformed") from exc
    if any(not value.is_finite() for value in result):
        raise ValueError("Frequency ledger returns are non-finite")
    return result[0], result[1], result[2]


def _current_authorization(user: FrequencyViewer) -> tuple[str, str]:
    if settings.DEMO_MODE:
        return _DEMO_OWNER_ID, "synthetic_demo"
    owner = str(user.pk)
    record = ProviderRecord.objects.filter(provider=TWELVE_DATA_PROVIDER).first()
    if record is None:
        raise ProviderConfigurationError("Provider display authorization is absent")
    validate_provider_usage(record, owner_id=owner)
    return owner, TWELVE_DATA_PROVIDER


def _logical_sha256(document: Mapping[str, Any]) -> str:
    logical = dict(document)
    logical.pop("derived_at", None)
    return hashlib.sha256(_canonical_bytes(logical)).hexdigest()


def _canonical_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None


def _require_publication_time(*, run: AnalysisRun, derived_at: datetime) -> None:
    if not _aware(derived_at) or derived_at < run.generated_at or derived_at > timezone.now():
        raise ValueError("Frequency evidence publication time is invalid")


def _datetime_value(document: Mapping[str, Any], key: str) -> datetime:
    try:
        value = datetime.fromisoformat(str(document[key]))
    except (KeyError, ValueError) as exc:
        raise ValueError("Frequency evidence timestamp is malformed") from exc
    if not _aware(value):
        raise ValueError("Frequency evidence timestamp is not timezone-aware")
    return value


def _positive_int(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("Frequency integer is invalid")
    return value


def _optional_reason(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("Frequency insufficiency reason is invalid")
    return value


def _sessions_for_horizon(horizon: str, value: object) -> int:
    expected = dict(load_price_product_config().simulation.horizons).get(horizon)
    if type(value) is not int or expected is None or value != expected:
        raise ValueError("Frequency horizon sessions do not match the frozen configuration")
    return value


def _derivation_source(
    document: Mapping[str, Any],
) -> Literal["manual_backfill", "scheduled_stage"]:
    value = document.get("derivation_source")
    if value not in ("manual_backfill", "scheduled_stage"):
        raise ValueError("Frequency derivation source is invalid")
    return cast(Literal["manual_backfill", "scheduled_stage"], value)


def _execution_revision(document: Mapping[str, Any]) -> str:
    execution = document.get("execution")
    value = execution.get("code_revision") if isinstance(execution, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ValueError("Frequency execution revision is invalid")
    return value


def _validate_document_shape(document: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "contract",
        "method_version",
        "method_identity_sha256",
        "product_version",
        "config_hash",
        "path_count",
        "owner_id",
        "source_provider",
        "source_run",
        "execution",
        "derived_at",
        "derivation_source",
        "own_deadline_met",
        "rows",
    }
    source_keys = {
        "id",
        "target_date",
        "generated_at",
        "data_cutoff",
        "snapshot_grade",
        "code_revision",
    }
    source_run = document.get("source_run")
    execution = document.get("execution")
    if (
        set(document) != expected
        or not isinstance(source_run, Mapping)
        or set(source_run) != source_keys
        or not isinstance(execution, Mapping)
        or set(execution) != {"code_revision"}
        or not isinstance(document.get("owner_id"), str)
        or not isinstance(document.get("source_provider"), str)
        or document.get("source_provider") not in {"synthetic_demo", "twelve_data"}
        or type(document.get("path_count")) is not int
        or document.get("own_deadline_met") is not False
    ):
        raise ValueError("Frequency evidence document shape is invalid")
    _derivation_source(document)
    _execution_revision(document)

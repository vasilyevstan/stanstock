"""Independent prospective capture/evaluation children; never acquire provider data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from types import UnionType
from typing import Any, cast, get_args, get_origin, get_type_hints
from uuid import UUID

import polars as pl
from django.conf import settings
from django.utils import timezone

from stanstock.core.jobs import JobExecutionResult, execute_target_job, target_job_lock
from stanstock.core.models import JobRun
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.asof import AsOfData, PriceFrameChecksumMismatchError, raw_price_asset_for
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    read_checksummed_bytes,
    resolve_asset_ref,
)
from stanstock.data.models import DataAsset, Listing
from stanstock.data.research_product import (
    PRODUCT_MEMBERSHIP_KIND,
    CapturedProductIntake,
    load_product_intake,
    product_intake_payload,
    product_membership_payload,
    verify_product_intake_membership,
    verify_product_price_content,
)
from stanstock.data.research_product_jobs import (  # type: ignore[attr-defined]
    SHADOW_CAPTURE_JOB,
    SHADOW_EVALUATE_JOB,
    ProviderError,
    _completed_product_run,
    _snapshot_for_intake,
    product_job_name,
)
from stanstock.research import shadow_study as s
from stanstock.research.models import Prediction, PredictionOutcome
from stanstock.research.outcome_refresh_validation import (
    OUTCOME_FIELDS,
    verify_prediction_outcome,
)
from stanstock.research.outcomes import (
    _close_on_date,
    _nth_observed_session,
    _price_subject,
    resolve_outcome,
)
from stanstock.research.price_product import (
    LedgerTriplet,
    PriceProductInputError,
    complete_input_hash,
    deterministic_seed,
)
from stanstock.research.price_product_config import (
    default_price_product_config_path,
    load_price_product_config,
)
from stanstock.research.price_product_replay import score_projection_metrics
from stanstock.research.price_product_shadow_drift import (
    ShadowDriftResult,
    project_shadow_drift,
    serialize_shadow_drift,
)
from stanstock.research.price_product_study import load_price_product_sources
from stanstock.research.product_pipeline import _jsonable, verify_price_product_output
from stanstock.research.refresh_evidence import lookup_manifest, model_row_values

_KNOWN_SOURCE_ERRORS = (
    OSError,
    ValueError,
    ProviderError,
    pl.exceptions.PolarsError,
    PriceFrameChecksumMismatchError,
)


def _sorted_refs(refs: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted({item["id"]: item for item in refs}.values(), key=lambda item: item["id"])


def _population(
    *,
    owner_id: str,
    provider: str,
    target: date,
    cutoff: datetime,
    store: AssetStore,
) -> tuple[dict[str, Any], CapturedProductIntake | None]:
    """Read original intake dispositions without requiring forecasts or market success."""
    empty: dict[str, Any] = {
        "intake": None,
        "membership": None,
        "population_count": None,
        "source": None,
        "members": [],
        "forecasts": [],
    }
    intake = load_product_intake(
        target_date=target,
        owner_id=owner_id,
        issuance_key="scheduled",
        store=store,
    )
    if intake is None or intake.asset.available_at > cutoff:
        return empty, None
    captured = product_intake_payload(intake, store=store)
    resolve_asset_ref(asset_ref_for(intake.asset), cutoff=cutoff)
    if (
        captured["source_provider"] != provider
        or captured["owner_id"] != owner_id
        or captured["issuance_key"] != "scheduled"
        or captured["target_date"] != target.isoformat()
    ):
        raise s._error("shadow_source_identity_invalid")
    empty["intake"] = asset_ref_for(intake.asset).to_json()
    empty["population_count"] = len(intake.requested_symbols)
    known = {
        listing.provider_symbol: str(listing.pk)
        for listing in Listing.objects.filter(pk__in=intake.candidate_listing_ids)
    }
    empty["members"] = [
        {
            "request_index": index,
            "listing_id": known.get(symbol),
            "admission": "not_materialized",
            "source_reason": None,
        }
        for index, symbol in enumerate(intake.requested_symbols)
    ]
    snapshot = _snapshot_for_intake(intake, store=store)
    if snapshot is None:
        return empty, intake
    membership_asset = DataAsset.objects.get(
        provider="stanstock",
        kind=PRODUCT_MEMBERSHIP_KIND,
        subject=str(snapshot.pk),
    )
    if membership_asset.available_at > cutoff:
        return empty, intake
    resolve_asset_ref(asset_ref_for(membership_asset), cutoff=cutoff)
    membership = product_membership_payload(snapshot, store=store)
    verify_product_intake_membership(
        snapshot=snapshot,
        payload=membership,
        cutoff=cutoff,
        store=store,
        provider=provider,
    )
    empty["membership"] = asset_ref_for(membership_asset).to_json()
    admissions = membership["admissions"]
    if not isinstance(admissions, dict) or set(admissions) != set(intake.requested_symbols):
        raise s._error("shadow_source_identity_invalid")
    empty["members"] = [
        {
            "request_index": index,
            "listing_id": admissions[symbol]["listing_id"],
            "admission": admissions[symbol]["status"],
            "source_reason": admissions[symbol].get(
                "reason", admissions[symbol].get("reason_code")
            ),
        }
        for index, symbol in enumerate(intake.requested_symbols)
    ]
    return empty, intake


def _source(
    *,
    owner_id: str,
    provider: str,
    target: date,
    cutoff: datetime,
    store: AssetStore,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    if (
        hashlib.sha256(default_price_product_config_path().read_bytes()).hexdigest()
        != (s.PROTOCOL_BODY["native"]["config_file_sha256"])
    ):
        raise s._error("shadow_protocol_mismatch")
    empty, intake = _population(
        owner_id=owner_id,
        provider=provider,
        target=target,
        cutoff=cutoff,
        store=store,
    )
    if intake is None or empty["membership"] is None:
        return empty, ()
    run = _completed_product_run(intake, store=store)
    if run is None or run.generated_at > cutoff:
        return empty, ()
    snapshot = run.universe_snapshot
    verify_price_product_output(run=run, store=store, replay=False)
    if provider == "twelve_data" and (not run.issued_on_time or snapshot.grade != "observed"):
        raise s._error("shadow_source_not_observed")
    manifest = lookup_manifest(run.pk)
    if manifest.count != 1 or manifest.asset is None:
        raise s._error("shadow_source_identity_invalid")
    resolve_asset_ref(asset_ref_for(manifest.asset), cutoff=cutoff)
    empty["source"] = {
        "run_id": str(run.pk),
        "snapshot_id": str(snapshot.pk),
        "generated_at": s._time(run.generated_at),
        "data_cutoff": s._time(run.data_cutoff),
        "code_revision": run.code_revision,
        "config_version": run.config_version,
        "config_hash": run.config_hash,
        "evidence_grade": snapshot.grade,
        "issued_on_time": run.issued_on_time,
        "output_manifest": asset_ref_for(manifest.asset).to_json(),
    }
    loaded = load_price_product_sources(
        run=run,
        store=store,
        config=load_price_product_config(),
        requested_listing_ids=(),
    )
    for source in loaded:
        if provider == "twelve_data" and (
            source.source_execution.mode != "provider"
            or source.source_execution.evidence_grade != "observed"
            or source.source_decision_time != run.data_cutoff
        ):
            raise s._error("shadow_source_not_observed")
        rows = {
            row.horizon: str(row.pk)
            for row in source.prediction_rows
            if row.method_version == "us-price-fhs-v1" and row.horizon in s.HORIZONS
        }
        if set(rows) != set(s.HORIZONS):
            raise s._error("shadow_source_identity_invalid")
        empty["forecasts"].append(
            {
                "listing_id": str(source.listing.pk),
                "original_prediction_ids": rows,
                "calculation_asset": asset_ref_for(source.calculation_artifact).to_json(),
                "source_assets": _sorted_refs(source.selection.source_assets),
                "preparation": "not_attempted",
                "reason_code": "shadow_deadline_missed",
                "shadow_json": None,
                "shadow_sha256": None,
            }
        )
    return empty, loaded


def _decode_adapter(value: Any, annotation: Any) -> Any:
    """Inverse of the frozen adapter's closed codec; not a new evidence schema."""
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin is UnionType:
        if value is None and type(None) in arguments:
            return None
        return _decode_adapter(value, next(item for item in arguments if item is not type(None)))
    if origin is tuple:
        if type(value) is not list:
            raise s._error()
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_decode_adapter(item, arguments[0]) for item in value)
        if len(value) != len(arguments):
            raise s._error()
        return tuple(
            _decode_adapter(item, kind) for item, kind in zip(value, arguments, strict=True)
        )
    if is_dataclass(annotation):
        annotations = get_type_hints(annotation)
        s._keys(value, " ".join(item.name for item in fields(annotation)))
        return cast(Any, annotation)(
            **{key: _decode_adapter(item, annotations[key]) for key, item in value.items()}
        )
    if annotation is UUID:
        s._uuid(value)
        return UUID(value)
    if annotation is datetime:
        return s._parse_time(value)
    if annotation is date:
        return s._date(value)
    if annotation is Decimal:
        return s._decimal(value)
    if annotation is float:
        if type(value) is not str:
            raise s._error()
        return float.fromhex(value)
    return value  # The unchanged serializer validates literals and primitive types.


def _adapter_document(forecast: dict[str, Any]) -> dict[str, Any]:
    raw = forecast["shadow_json"]
    document: dict[str, Any] = json.loads(raw)
    decoded = _decode_adapter(document, ShadowDriftResult)
    if serialize_shadow_drift(decoded).decode() != raw:
        raise s._error()
    return document


def _verify_capture(
    asset: DataAsset,
    *,
    owner_id: str,
    provider: str,
    protocol: DataAsset,
    activation: DataAsset,
    store: AssetStore,
) -> dict[str, Any]:
    document = s._read_asset(asset, store=store)
    body = document["body"]
    if (
        document["owner_id"] != owner_id
        or document["source_provider"] != provider
        or document["protocol"] != asset_ref_for(protocol).to_json()
        or document["activation"] != asset_ref_for(activation).to_json()
    ):
        raise s._error("shadow_source_identity_invalid")
    activation_document = s._verify_activation(activation, protocol=protocol, store=store)
    session = s._session(activation_document, s._date(body["target_date"]))
    if any(body[key] != value for key, value in session.items()):
        raise s._error("shadow_protocol_mismatch")
    source, loaded = _source(
        owner_id=owner_id,
        provider=provider,
        target=s._date(body["target_date"]),
        cutoff=asset.available_at,
        store=store,
    )
    if any(body[key] != source[key] for key in source if key != "forecasts"):
        raise s._error("shadow_source_identity_invalid")
    if len(body["forecasts"]) != len(source["forecasts"]):
        raise s._error("shadow_source_identity_invalid")
    for forecast, expected, original in zip(
        body["forecasts"],
        source["forecasts"],
        loaded,
        strict=True,
    ):
        if any(
            forecast[key] != expected[key]
            for key in (
                "listing_id",
                "original_prediction_ids",
                "calculation_asset",
                "source_assets",
            )
        ):
            raise s._error("shadow_source_identity_invalid")
        if forecast["preparation"] != "prepared":
            continue
        adapter = _adapter_document(forecast)
        product_input = original.selection.product_input
        for key, source_identity in (
            ("stock_identity", product_input.stock.identity),
            ("benchmark_identity", product_input.benchmark.identity),
        ):
            if adapter[key] != {
                "asset_id": str(source_identity.asset_id),
                "provider": source_identity.provider,
                "subject": source_identity.subject,
                "sha256": source_identity.sha256,
                "retrieved_at": s._time(source_identity.retrieved_at),
                "available_at": s._time(source_identity.available_at),
            }:
                raise s._error("shadow_source_identity_invalid")
        if (
            adapter["listing_id"] != forecast["listing_id"]
            or adapter["target_date"] != body["target_date"]
            or adapter["assumptions_sha256"] != s.ASSUMPTIONS_HASH
            or adapter["native_config_hash"] != s.CONFIG_HASH
            or adapter["input_hash"] != complete_input_hash(product_input)
            or adapter["innovation_seed"]
            != deterministic_seed(
                listing_id=product_input.listing_id,
                target_date=product_input.target_date,
                method_version="us-price-fhs-v1",
                effective_config_hash=s.CONFIG_HASH,
            )
            or adapter["decision_time"] != s._time(product_input.decision_time)
            or adapter["source_execution"]
            != {
                "mode": original.source_execution.mode,
                "evidence_grade": original.source_execution.evidence_grade,
            }
        ):
            raise s._error("shadow_source_identity_invalid")
        # Native full control payloads, including zero drift and withholding,
        # are independently bound to the immutable source calculation.
        native = json.loads(read_checksummed_bytes(store, original.calculation_artifact))
        native_forecast = native["result"]["forecast"]
        mean = native_forecast["mean_log_return"]
        if adapter["mean_log_return"] != (None if mean is None else float(mean).hex()) or (
            adapter["native_forecast_insufficiency_reason"]
            != native_forecast["insufficiency_reason"]
        ):
            raise s._error("shadow_source_identity_invalid")
        for index, projection in enumerate(adapter["projections"][:4]):
            native_projection = native_forecast["projections"][index % 2]
            prefix = "zero_drift_" if index >= 2 else ""
            for field in ("raw_returns", "ledger_returns", "raw_prices", "ledger_prices"):
                # The source codec uses decimal/string values, whereas the
                # adapter preserves floats as hex. Compare typed values below.
                left, right = projection[field], native_projection[prefix + field]
                if left is None or right is None:
                    if left != right:
                        raise s._error("shadow_source_identity_invalid")
                elif field.startswith("raw"):
                    if any(float.fromhex(left[key]) != float(right[key]) for key in left):
                        raise s._error("shadow_source_identity_invalid")
                elif left != right:
                    raise s._error("shadow_source_identity_invalid")
    return document


def _capture_state(
    asset: DataAsset,
    document: dict[str, Any],
    *,
    protocol: DataAsset,
    activation: DataAsset,
    store: AssetStore,
) -> tuple[str, datetime | None]:
    witness = s._witness(asset, document=document, protocol=protocol, activation=activation)
    body = document["body"]
    active_document = s._verify_activation(activation, protocol=protocol, store=store)
    active_witness = s._witness(
        activation,
        document=active_document,
        protocol=protocol,
        activation=None,
    )
    if witness is None or active_witness is None:
        return "publication_unverified", witness
    if active_witness >= s._parse_time(body["scheduled_close"]):
        return "missed", witness
    if body["population_count"] is None:
        return "missing_population", witness
    if body["disposition"] != "prepared":
        return body["disposition"], witness
    if witness >= s._parse_time(body["next_session_open"]):
        return "late", witness
    if body["forecasts"] and not any(
        item["preparation"] == "prepared" for item in body["forecasts"]
    ):
        return "failed", witness
    return "timely", witness


def execute_shadow_capture_job(
    *,
    owner: object,
    target_date: date,
    store: AssetStore,
) -> s.ShadowJobResult:
    if not settings.SHADOW_STUDY_ENABLED:
        return s.ShadowJobResult(None, "disabled")
    owner_id, provider, protocol, activation, closure = s._context(owner=owner, store=store)
    active_document = s._verify_activation(activation, protocol=protocol, store=store)
    session = s._session(active_document, target_date)
    if closure is not None and s._parse_time(session["scheduled_close"]) > s._parse_time(
        s._read_asset(closure, store=store)["body"]["closed_at"]
    ):
        return s.ShadowJobResult(None, "ineligible", "shadow_closed_for_target")
    identity = s._job_identity(owner_id, provider, protocol, activation)
    capture_identity = {
        "activation": asset_ref_for(activation).to_json(),
        "target_date": target_date.isoformat(),
    }
    recovered: list[DataAsset] = []
    attempted: list[JobRun] = []

    def before() -> None:
        try:
            s._authorize(owner)
            existing = s._find(s.KINDS[3], capture_identity, store=store)
            if existing is not None:
                _verify_capture(
                    existing,
                    owner_id=owner_id,
                    provider=provider,
                    protocol=protocol,
                    activation=activation,
                    store=store,
                )
                recovered.append(existing)
        except RefreshVerificationError as exc:
            raise s._error(
                exc.reason_code
                if exc.reason_code in s.SAFE_CODES
                else "shadow_source_identity_invalid"
            ) from None
        except _KNOWN_SOURCE_ERRORS:
            raise s._error("shadow_source_identity_invalid") from None

    def task(job: JobRun) -> JobExecutionResult:
        attempted.append(job)
        try:
            if recovered:
                return JobExecutionResult(details=s._job_details(identity, recovered))
            source, loaded = _source(
                owner_id=owner_id,
                provider=provider,
                target=target_date,
                cutoff=timezone.now(),
                store=store,
            )
            deadline = s._parse_time(session["next_session_open"])
            if not loaded and timezone.now() < deadline:
                return JobExecutionResult(
                    status=JobRun.Status.NO_DATA,
                    details=s._job_details(
                        identity,
                        [],
                        state="pending",
                        reason_code=(
                            "shadow_population_missing"
                            if source["intake"] is None
                            else "shadow_source_missing"
                        ),
                    ),
                )
            revision = s._revision()
            active_witness = s._witness(
                activation,
                document=active_document,
                protocol=protocol,
                activation=None,
            )
            reason = None
            if active_witness is None:
                reason = "shadow_publication_unverified"
            elif active_witness >= s._parse_time(session["scheduled_close"]):
                reason = "shadow_activation_not_available_before_close"
            elif timezone.now() >= deadline:
                reason = "shadow_deadline_missed"
            elif not loaded:
                reason = "shadow_source_missing"
            if reason is None:
                for forecast, original in zip(source["forecasts"], loaded, strict=True):
                    try:
                        payload = serialize_shadow_drift(
                            project_shadow_drift(
                                original.selection.product_input,
                                config=load_price_product_config(),
                            )
                        )
                    except (PriceProductInputError, ArithmeticError):
                        forecast.update(
                            preparation="failed", reason_code="shadow_preparation_failed"
                        )
                    else:
                        forecast.update(
                            preparation="prepared",
                            reason_code=None,
                            shadow_json=payload.decode(),
                            shadow_sha256=hashlib.sha256(payload).hexdigest(),
                        )
            recorded_at = timezone.now()
            document = s._envelope(
                kind=s.KINDS[3],
                owner_id=owner_id,
                provider=provider,
                protocol=protocol,
                activation=activation,
                job=job,
                revision=revision,
                recorded_at=recorded_at,
                body={
                    **session,
                    **source,
                    "disposition": "prepared" if reason is None else "missed",
                    "reason_code": reason,
                },
            )
            asset = s._register_shadow_asset(kind=s.KINDS[3], document=document, store=store)
            recovered.append(asset)
            return JobExecutionResult(details=s._job_details(identity, [asset]))
        except RefreshVerificationError as exc:
            if exc.reason_code in s.SAFE_CODES:
                raise s._error(exc.reason_code) from None
            raise s._error("shadow_source_identity_invalid") from None
        except _KNOWN_SOURCE_ERRORS:
            raise s._error("shadow_source_identity_invalid") from None

    try:
        child = execute_target_job(
            job_name=product_job_name(SHADOW_CAPTURE_JOB, identity),
            region="us",
            target_date=target_date,
            task=task,
            before_attempt=before,
        )
    except s.ShadowStudyError as exc:
        if attempted:
            exc.child = JobRun.objects.get(pk=attempted[0].pk)
        raise
    if not recovered:
        return s.ShadowJobResult(child, "pending", child.details.get("reason_code"))
    document = _verify_capture(
        recovered[0],
        owner_id=owner_id,
        provider=provider,
        protocol=protocol,
        activation=activation,
        store=store,
    )
    state, _witness = _capture_state(
        recovered[0],
        document,
        protocol=protocol,
        activation=activation,
        store=store,
    )
    return s.ShadowJobResult(
        child,
        state,
        "shadow_preparation_failed" if state == "failed" else document["body"]["reason_code"],
        {"captures": 1},
    )


def _case_key(activation: DataAsset, capture: dict[str, Any], listing_id: str, horizon: str) -> str:
    return s._hash(
        {
            "activation": asset_ref_for(activation).to_json(),
            "listing_id": listing_id,
            "target_date": capture["body"]["target_date"],
            "horizon": horizon,
        }
    )


def _prediction(
    capture: dict[str, Any],
    forecast: dict[str, Any],
    horizon: str,
    provider: str,
) -> Prediction:
    prediction = (
        Prediction.objects.select_related("listing", "analysis__run")
        .filter(
            pk=forecast["original_prediction_ids"][horizon],
            listing_id=forecast["listing_id"],
            horizon=horizon,
            evidence_role="advisory",
            method_version="us-price-fhs-v1",
            price_provider=provider,
            analysis__run_id=capture["body"]["source"]["run_id"],
            target_date=s._date(capture["body"]["target_date"]),
        )
        .first()
    )
    if prediction is None:
        raise s._error("shadow_source_identity_invalid")
    return prediction


def _native_snapshot(outcome: PredictionOutcome) -> dict[str, Any]:
    values = model_row_values(outcome, (*OUTCOME_FIELDS, "evaluated_at"))
    return {
        "prediction_id": str(outcome.prediction_id),
        "status": outcome.status,
        "evaluated_at": s._time(outcome.evaluated_at),
        "evaluation_date": outcome.evaluation_date.isoformat(),
        "actual_return": None if values["actual_return"] is None else str(values["actual_return"]),
        "row_hash": s._hash(_jsonable(values)),
    }


def _price_evidence(
    prediction: Prediction,
    *,
    cutoff: datetime,
    through: date,
    store: AssetStore,
    refs: list[dict[str, str]] | None = None,
    visited: list[dict[str, str]] | None = None,
) -> tuple[DataAsset | None, pl.DataFrame | None, list[dict[str, str]]]:
    """One unambiguous vintage, exact closure, and physically clipped rows."""
    if refs is None:
        candidates = list(
            DataAsset.objects.filter(
                provider=prediction.price_provider,
                kind="price_history",
                subject=_price_subject(prediction),
                available_at__lte=cutoff,
                retrieved_at__lte=cutoff,
            ).order_by("-available_at", "-retrieved_at", "id")
        )
        if not candidates:
            return None, None, []
        asset = candidates[0]
        if len(candidates) > 1 and (candidates[1].available_at, candidates[1].retrieved_at) == (
            asset.available_at,
            asset.retrieved_at,
        ):
            raise s._error("shadow_registry_ambiguous")
    else:
        candidates = [
            resolve_asset_ref(s._ref(ref), cutoff=cutoff)
            for ref in refs
            if ref["provider"] == prediction.price_provider
            and ref["kind"] == "price_history"
            and ref["subject"] == _price_subject(prediction)
        ]
        candidates.sort(key=lambda item: (item.available_at, item.retrieved_at), reverse=True)
        if not candidates or (
            len(candidates) > 1
            and (candidates[0].available_at, candidates[0].retrieved_at)
            == (candidates[1].available_at, candidates[1].retrieved_at)
        ):
            raise s._error("shadow_source_identity_invalid")
        asset = candidates[0]
    if visited is not None:
        visited.append(asset_ref_for(asset).to_json())
    raw = raw_price_asset_for(asset, cutoff=cutoff)
    if visited is not None:
        visited.append(asset_ref_for(raw).to_json())
    evidence = _sorted_refs([asset_ref_for(asset).to_json(), asset_ref_for(raw).to_json()])
    if refs is not None and any(item not in refs for item in evidence):
        raise s._error("shadow_source_identity_invalid")
    if asset.metadata.get("currency") != "USD" or asset.metadata.get("adjustment") != "splits":
        raise s._error("shadow_basis_incompatible")
    verify_product_price_content(
        asset=asset,
        raw=raw,
        cutoff=cutoff,
        target_date=asset.period_end or through,
        store=store,
    )
    frame = (
        AsOfData(cutoff, store)
        .price_frame_for_asset_with_diagnostics(
            asset=asset,
            through_date=through,
        )
        .frame
    )
    return asset, frame, evidence


def _strict_return(
    prediction: Prediction,
    frame: pl.DataFrame | None,
    *,
    endpoint: date | None = None,
) -> tuple[date | None, Decimal | None, str | None]:
    if frame is None:
        return None, None, "shadow_native_outcome_unavailable"
    session = _nth_observed_session(
        frame, prediction.target_date, 126 if prediction.horizon == "6m" else 252
    )
    if session is None:
        return None, None, "shadow_native_outcome_unavailable"
    if endpoint is not None and session.observation_date != endpoint:
        return session.observation_date, None, "shadow_endpoint_unverified"
    baseline = _close_on_date(frame, prediction.target_date)
    if baseline is None:
        return session.observation_date, None, "shadow_endpoint_unverified"
    # FHS's native verifier permits a preceding baseline; this study does not.
    if Decimal(str(baseline.close)).quantize(Decimal("0.000001")) != (
        prediction.price_at_prediction
    ):
        return session.observation_date, None, "shadow_basis_incompatible"
    actual = Decimal(str(session.close / float(prediction.price_at_prediction) - 1.0)).quantize(
        Decimal("0.0001"),
    )
    if not actual.is_finite():
        return session.observation_date, None, "shadow_endpoint_unverified"
    return session.observation_date, actual, None


def _scores(forecast: dict[str, Any], horizon: str, actual: Decimal) -> list[dict[str, str]]:
    adapter = _adapter_document(forecast)
    scores = []
    for projection in adapter["projections"]:
        if projection["horizon"] != horizon:
            continue
        triplet = projection["ledger_returns"]
        metrics = score_projection_metrics(
            predicted_returns=None
            if triplet is None
            else LedgerTriplet(**{key: Decimal(value) for key, value in triplet.items()}),
            actual_return=actual,
            config=load_price_product_config(),
        )
        if metrics.interval_score is None or metrics.median_absolute_error is None:
            return []
        scores.append(
            {
                "arm_id": projection["arm_id"],
                "interval_score": str(metrics.interval_score),
                "median_absolute_error": str(metrics.median_absolute_error),
            }
        )
    return scores


def _replay_frozen_native(
    prediction: Prediction,
    body: dict[str, Any],
    *,
    store: AssetStore,
) -> None:
    """Authenticate the native snapshot from its exact original evidence, not today's row."""
    snapshot = body["native_outcome"]
    if snapshot is None:
        raise s._error("shadow_native_outcome_unavailable")
    cutoff = s._parse_time(snapshot["evaluated_at"])
    refs = body["source_assets"]
    all_assets = [
        resolve_asset_ref(s._ref(item), cutoff=s._parse_time(body["assessed_at"])) for item in refs
    ]
    resolved_assets = [
        asset
        for asset in all_assets
        if asset.available_at <= cutoff and asset.retrieved_at <= cutoff
    ]
    for asset in resolved_assets:
        read_checksummed_bytes(store, asset)

    def price_loader(subject: str, through_date: date) -> pl.DataFrame:
        assets = [
            asset
            for asset in resolved_assets
            if asset.kind == "price_history"
            and asset.subject == subject
            and asset.provider == prediction.price_provider
        ]
        assets.sort(key=lambda item: (item.available_at, item.retrieved_at), reverse=True)
        if not assets:
            # A missing optional benchmark reproduced the original native
            # unavailable benchmark; a missing stock can never accept a case.
            raise DataAsset.DoesNotExist("DataAsset matching query does not exist.")
        if len(assets) > 1 and (assets[0].available_at, assets[0].retrieved_at) == (
            assets[1].available_at,
            assets[1].retrieved_at,
        ):
            raise s._error("shadow_registry_ambiguous")
        asset = assets[0]
        raw = raw_price_asset_for(asset, cutoff=cutoff)
        if asset_ref_for(raw).to_json() not in refs:
            raise s._error("shadow_source_identity_invalid")
        verify_product_price_content(
            asset=asset,
            raw=raw,
            cutoff=cutoff,
            target_date=asset.period_end or through_date,
            store=store,
        )
        return (
            AsOfData(cutoff, store)
            .price_frame_for_asset_with_diagnostics(
                asset=asset,
                through_date=through_date,
            )
            .frame
        )

    resolved = resolve_outcome(
        prediction,
        provider=prediction.price_provider,
        evaluation_date=s._date(snapshot["evaluation_date"]),
        evaluated_at=cutoff,
        benchmark_subject="SPY",
        price_loader=price_loader,
        store=store,
    )
    row = PredictionOutcome(
        prediction=prediction,
        evaluated_at=cutoff,
        **{key: getattr(resolved, key) for key in OUTCOME_FIELDS if key != "prediction_id"},
    )
    if _native_snapshot(row) != snapshot:
        raise s._error("shadow_source_identity_invalid")


def _verify_assessment(
    asset: DataAsset,
    *,
    capture_asset: DataAsset,
    capture: dict[str, Any],
    forecast: dict[str, Any],
    protocol: DataAsset,
    activation: DataAsset,
    store: AssetStore,
) -> dict[str, Any]:
    document = s._read_asset(asset, store=store)
    body = document["body"]
    if (
        any(
            document[key] != capture[key]
            for key in (
                "owner_id",
                "source_provider",
                "protocol",
                "activation",
            )
        )
        or body["capture"] != asset_ref_for(capture_asset).to_json()
        or body["listing_id"] != forecast["listing_id"]
        or body["original_prediction_id"] != forecast["original_prediction_ids"][body["horizon"]]
        or body["case_key"] != _case_key(activation, capture, body["listing_id"], body["horizon"])
        or s._parse_time(body["assessed_at"]) > asset.available_at
        or s._date(body["market_through"]) > s._parse_time(body["assessed_at"]).date()
    ):
        raise s._error("shadow_source_identity_invalid")
    prediction = _prediction(capture, forecast, body["horizon"], document["source_provider"])
    for ref in body["source_assets"]:
        evidence = resolve_asset_ref(s._ref(ref), cutoff=s._parse_time(body["assessed_at"]))
        try:
            read_checksummed_bytes(store, evidence)
        except RefreshVerificationError:
            if (
                body["state"] != "challenge"
                or body["qualification"] != "later_unverifiable_evidence"
            ):
                raise
    if body["state"] == "accepted":
        _replay_frozen_native(prediction, body, store=store)
        native = body["native_outcome"]
        if s._parse_time(native["evaluated_at"]) > s._parse_time(body["assessed_at"]):
            raise s._error("shadow_cutoff_unverified")
        _asset, frame, _refs = _price_evidence(
            prediction,
            cutoff=s._parse_time(body["assessed_at"]),
            through=s._date(body["maturity_date"]),
            store=store,
            refs=body["source_assets"],
        )
        maturity, actual, reason = _strict_return(
            prediction,
            frame,
            endpoint=s._date(body["maturity_date"]),
        )
        if (
            reason is not None
            or maturity is None
            or actual is None
            or str(actual) != body["actual_return"]
            or _scores(forecast, body["horizon"], actual) != body["scores"]
        ):
            raise s._error("shadow_endpoint_unverified")
    if body["state"] == "challenge":
        prior = resolve_asset_ref(
            s._ref(body["prior_accepted"]), cutoff=s._parse_time(body["assessed_at"])
        )
        previous = s._read_asset(prior, store=store)
        if (
            previous["body"]["state"] != "accepted"
            or previous["body"]["case_key"] != body["case_key"]
        ):
            raise s._error("shadow_source_identity_invalid")
        try:
            _asset, frame, _refs = _price_evidence(
                prediction,
                cutoff=s._parse_time(body["assessed_at"]),
                through=s._date(previous["body"]["maturity_date"]),
                store=store,
                refs=body["source_assets"],
            )
            maturity, actual, reason = _strict_return(
                prediction,
                frame,
                endpoint=s._date(previous["body"]["maturity_date"]),
            )
        except _KNOWN_SOURCE_ERRORS:
            if (
                body["qualification"] != "later_unverifiable_evidence"
                or body["actual_return"] is not None
            ):
                raise s._error("shadow_source_identity_invalid") from None
        else:
            qualification = (
                "later_compatible_revision"
                if reason is None
                else "later_incompatible_revision"
                if reason == "shadow_basis_incompatible"
                else "later_unverifiable_evidence"
            )
            if (
                body["qualification"] != qualification
                or body["maturity_date"] != (None if maturity is None else maturity.isoformat())
                or body["actual_return"] != (None if actual is None else str(actual))
            ):
                raise s._error("shadow_source_identity_invalid")
    return document


def _case_records(
    *,
    activation: DataAsset,
    case_key: str,
    capture_asset: DataAsset,
    capture: dict[str, Any],
    forecast: dict[str, Any],
    protocol: DataAsset,
    store: AssetStore,
    cutoff: datetime,
) -> list[tuple[DataAsset, dict[str, Any]]]:
    records = []
    seen = set()
    for asset in DataAsset.objects.filter(
        kind=s.KINDS[4],
        metadata__activation_id=str(activation.pk),
        available_at__lte=cutoff,
        retrieved_at__lte=cutoff,
    ).order_by("available_at", "id"):
        document = s._read_asset(asset, store=store)
        if document["body"]["case_key"] != case_key:
            continue
        if asset.subject in seen:
            raise s._error("shadow_registry_ambiguous")
        seen.add(asset.subject)
        document = _verify_assessment(
            asset,
            capture_asset=capture_asset,
            capture=capture,
            forecast=forecast,
            protocol=protocol,
            activation=activation,
            store=store,
        )
        # Committed evaluation evidence can be recovered after a crash; only
        # the reader, not deduplication, requires its publication witness.
        records.append((asset, document))
    if sum(document["body"]["state"] == "accepted" for _asset, document in records) > 1:
        raise s._error("shadow_registry_ambiguous")
    return records


def _assess(
    *,
    capture_asset: DataAsset,
    capture: dict[str, Any],
    forecast: dict[str, Any],
    horizon: str,
    activation: DataAsset,
    assessed_at: datetime,
    market_through: date,
    prior: tuple[DataAsset, dict[str, Any]] | None,
    store: AssetStore,
) -> dict[str, Any]:
    prediction = _prediction(capture, forecast, horizon, capture["source_provider"])
    body: dict[str, Any] = {
        "case_key": _case_key(activation, capture, forecast["listing_id"], horizon),
        "capture": asset_ref_for(capture_asset).to_json(),
        "listing_id": forecast["listing_id"],
        "horizon": horizon,
        "original_prediction_id": str(prediction.pk),
        "state": "unresolved",
        "qualification": "native_outcome_unavailable",
        "assessed_at": s._time(assessed_at),
        "market_through": market_through.isoformat(),
        "maturity_date": None,
        "native_outcome": None,
        "source_assets": [],
        "prior_accepted": None,
        "actual_return": None,
        "scores": [],
        "reason_codes": ["shadow_native_outcome_unavailable"],
    }
    outcome = PredictionOutcome.objects.filter(
        prediction=prediction, evaluated_at__lte=assessed_at
    ).first()
    if outcome is not None:
        body["native_outcome"] = _native_snapshot(outcome)
    if prior is not None:
        previous_asset, previous = prior
        frozen = previous["body"]
        body.update(
            state="challenge",
            prior_accepted=asset_ref_for(previous_asset).to_json(),
            native_outcome=frozen["native_outcome"],
            qualification="later_unverifiable_evidence",
            reason_codes=["shadow_evidence_challenged"],
        )
        visited: list[dict[str, str]] = []
        try:
            _asset, frame, refs = _price_evidence(
                prediction,
                cutoff=assessed_at,
                through=s._date(frozen["maturity_date"]),
                store=store,
                visited=visited,
            )
            body["source_assets"] = refs
            maturity, actual, reason = _strict_return(
                prediction,
                frame,
                endpoint=s._date(frozen["maturity_date"]),
            )
            body["maturity_date"] = None if maturity is None else maturity.isoformat()
            body["actual_return"] = None if actual is None else str(actual)
            if reason == "shadow_basis_incompatible":
                body["qualification"] = "later_incompatible_revision"
            elif reason is None:
                body["qualification"] = "later_compatible_revision"
                if body["actual_return"] == frozen["actual_return"]:
                    return cast(dict[str, Any], frozen)
        except _KNOWN_SOURCE_ERRORS:
            body["source_assets"] = _sorted_refs(visited)
            body["qualification"] = "later_unverifiable_evidence"
    elif outcome is not None:
        try:
            # Never pass today's clock for an older native terminal row.
            verification = verify_prediction_outcome(
                prediction,
                outcome,
                provider=prediction.price_provider,
                benchmark_subject="SPY",
                evaluation_time=outcome.evaluated_at,
                parent_target_date=outcome.evaluation_date,
                frame_cache={},
            )
            body["source_assets"] = _sorted_refs([ref.to_json() for ref in verification.asset_refs])
            if (
                outcome.status in ("matured", "corporate_event")
                and outcome.evaluation_date <= market_through
            ):
                _asset, frame, current_refs = _price_evidence(
                    prediction,
                    cutoff=assessed_at,
                    through=outcome.evaluation_date,
                    store=store,
                )
                body["source_assets"] = _sorted_refs([*body["source_assets"], *current_refs])
                maturity, actual, reason = _strict_return(
                    prediction,
                    frame,
                    endpoint=outcome.evaluation_date,
                )
                body["maturity_date"] = None if maturity is None else maturity.isoformat()
                if reason is None and actual is not None:
                    scores = _scores(forecast, horizon, actual)
                    if len(scores) == 3:
                        body.update(
                            state="accepted",
                            qualification="verified_native_outcome",
                            actual_return=str(actual),
                            scores=scores,
                            reason_codes=[],
                        )
                else:
                    code = reason or "shadow_endpoint_unverified"
                    body.update(
                        state="quarantined",
                        qualification=(
                            "basis_incompatible"
                            if code == "shadow_basis_incompatible"
                            else "endpoint_unverified"
                        ),
                        reason_codes=[code],
                    )
            elif outcome.status == "corporate_event":
                body.update(
                    state="quarantined",
                    qualification="basis_incompatible",
                    reason_codes=["shadow_basis_incompatible"],
                )
        except _KNOWN_SOURCE_ERRORS:
            body.update(
                state="quarantined",
                qualification="cutoff_unverified",
                reason_codes=["shadow_cutoff_unverified"],
            )
    body["evidence_key"] = s._evidence_key(body)
    return body


def _visible_captures(
    *,
    owner_id: str,
    provider: str,
    protocol: DataAsset,
    activation: DataAsset,
    cutoff: datetime,
    store: AssetStore,
) -> list[tuple[DataAsset, dict[str, Any], str]]:
    records = []
    seen = set()
    for asset in DataAsset.objects.filter(
        kind=s.KINDS[3],
        metadata__activation_id=str(activation.pk),
        available_at__lte=cutoff,
        retrieved_at__lte=cutoff,
    ).order_by("available_at", "id"):
        document = _verify_capture(
            asset,
            owner_id=owner_id,
            provider=provider,
            protocol=protocol,
            activation=activation,
            store=store,
        )
        if document["body"]["target_date"] in seen:
            raise s._error("shadow_registry_ambiguous")
        seen.add(document["body"]["target_date"])
        state, witness = _capture_state(
            asset,
            document,
            protocol=protocol,
            activation=activation,
            store=store,
        )
        if witness is not None and witness <= cutoff:
            records.append((asset, document, state))
    return records


def execute_shadow_evaluation_job(
    *,
    owner: object,
    evaluation_date: date,
    store: AssetStore,
) -> s.ShadowJobResult:
    if not settings.SHADOW_STUDY_ENABLED:
        return s.ShadowJobResult(None, "disabled")
    owner_id, provider, protocol, activation, _closure = s._context(owner=owner, store=store)
    identity = s._job_identity(owner_id, provider, protocol, activation)
    name = product_job_name(SHADOW_EVALUATE_JOB, identity)
    attempted: list[JobRun] = []
    if evaluation_date > timezone.now().date():
        raise s._error("shadow_cutoff_unverified")

    def verify_scan(job: JobRun) -> None:
        details = job.details
        if details.get("identity") != identity:
            raise s._error("shadow_source_identity_invalid")
        cutoff = s._parse_time(details["scan_started_at"])
        if cutoff != job.started_at:
            raise s._error("shadow_cutoff_unverified")
        captures = _visible_captures(
            owner_id=owner_id,
            provider=provider,
            protocol=protocol,
            activation=activation,
            cutoff=cutoff,
            store=store,
        )
        expected = [asset_ref_for(asset).to_json() for asset, _document, _state in captures]
        if details.get("captures") != expected:
            raise s._error("shadow_source_identity_invalid")
        for ref in details["assets"]:
            asset = resolve_asset_ref(s._ref(ref), cutoff=job.finished_at or timezone.now())
            s._read_asset(asset, store=store)
        # Every selected evaluable case must have an exact assessment binding.
        expected_cases = {
            _case_key(activation, document, forecast["listing_id"], horizon)
            for _asset, document, state in captures
            if state == "timely"
            for forecast in document["body"]["forecasts"]
            if forecast["preparation"] == "prepared"
            for horizon in s.HORIZONS
        }
        actual_cases = {
            s._read_asset(
                resolve_asset_ref(s._ref(ref), cutoff=job.finished_at or timezone.now()),
                store=store,
            )["body"]["case_key"]
            for ref in details["assets"]
        }
        if actual_cases != expected_cases or len(details["assets"]) != len(
            {ref["id"] for ref in details["assets"]}
        ):
            raise s._error("shadow_source_identity_invalid")
        for capture_asset, capture, state in captures:
            if state != "timely":
                continue
            for forecast in capture["body"]["forecasts"]:
                if forecast["preparation"] != "prepared":
                    continue
                for horizon in s.HORIZONS:
                    records = _case_records(
                        activation=activation,
                        case_key=_case_key(activation, capture, forecast["listing_id"], horizon),
                        capture_asset=capture_asset,
                        capture=capture,
                        forecast=forecast,
                        protocol=protocol,
                        store=store,
                        cutoff=timezone.now(),
                    )
                    matching = [
                        (asset, document)
                        for asset, document in records
                        if asset_ref_for(asset).to_json() in details["assets"]
                    ]
                    if not 1 <= len(matching) <= 2:
                        raise s._error("shadow_source_identity_invalid")
                    challenges = [
                        document
                        for _asset, document in matching
                        if document["body"]["state"] == "challenge"
                    ]
                    if len(matching) == 2 and (
                        len(challenges) != 1
                        or not any(
                            document["body"]["state"] == "accepted"
                            and asset_ref_for(asset).to_json()
                            == challenges[0]["body"]["prior_accepted"]
                            for asset, document in matching
                        )
                    ):
                        raise s._error("shadow_source_identity_invalid")
                    # A challenge publishes its exact frozen dependency too.
                    # An older scan may instead have an earlier independent witness.
                    required = list(matching)
                    if challenges and len(matching) == 1:
                        required.extend(
                            (asset, document)
                            for asset, document in records
                            if asset_ref_for(asset).to_json()
                            == challenges[0]["body"]["prior_accepted"]
                            and document["body"]["state"] == "accepted"
                        )
                        if len(required) != 2:
                            raise s._error("shadow_publication_unverified")
                    if any(
                        (
                            witness := s._witness(
                                asset,
                                document=document,
                                protocol=protocol,
                                activation=activation,
                            )
                        )
                        is None
                        or job.finished_at is None
                        or witness > job.finished_at
                        for asset, document in required
                    ):
                        raise s._error("shadow_publication_unverified")

    def before() -> None:
        s._authorize(owner)
        successes = list(
            JobRun.objects.filter(
                job_name=name,
                region="us",
                target_date=evaluation_date,
                status=JobRun.Status.SUCCESS,
            )
        )
        if len(successes) > 1:
            raise s._error("shadow_registry_ambiguous")
        if successes:
            verify_scan(successes[0])

    def task(job: JobRun) -> JobExecutionResult:
        attempted.append(job)
        try:
            cutoff = job.started_at
            captures = _visible_captures(
                owner_id=owner_id,
                provider=provider,
                protocol=protocol,
                activation=activation,
                cutoff=cutoff,
                store=store,
            )
            revision = s._revision()
            assets = []
            for capture_asset, capture, state in captures:
                if state != "timely":
                    continue
                for forecast in capture["body"]["forecasts"]:
                    if forecast["preparation"] != "prepared":
                        continue
                    for horizon in s.HORIZONS:
                        case_key = _case_key(activation, capture, forecast["listing_id"], horizon)
                        with target_job_lock(
                            job_name=f"shadow_case:{case_key}",
                            region="us",
                            target_date=s.COORDINATION_DATE,
                        ):
                            records = _case_records(
                                activation=activation,
                                case_key=case_key,
                                capture_asset=capture_asset,
                                capture=capture,
                                forecast=forecast,
                                protocol=protocol,
                                store=store,
                                cutoff=timezone.now(),
                            )
                            prior = next(
                                (
                                    item
                                    for item in records
                                    if item[1]["body"]["state"] == "accepted"
                                ),
                                None,
                            )
                            body = _assess(
                                capture_asset=capture_asset,
                                capture=capture,
                                forecast=forecast,
                                horizon=horizon,
                                activation=activation,
                                assessed_at=timezone.now(),
                                market_through=evaluation_date,
                                prior=prior,
                                store=store,
                            )
                            existing = [
                                asset
                                for asset, document in records
                                if document["body"]["evidence_key"] == body["evidence_key"]
                            ]
                            if len(existing) > 1:
                                raise s._error("shadow_registry_ambiguous")
                            if body["state"] == "challenge":
                                if (
                                    prior is None
                                    or body["prior_accepted"] != asset_ref_for(prior[0]).to_json()
                                ):
                                    raise s._error("shadow_source_identity_invalid")
                                assets.append(prior[0])
                            if existing:
                                assets.append(existing[0])
                            else:
                                assets.append(
                                    s._register_shadow_asset(
                                        kind=s.KINDS[4],
                                        store=store,
                                        document=s._envelope(
                                            kind=s.KINDS[4],
                                            owner_id=owner_id,
                                            provider=provider,
                                            protocol=protocol,
                                            activation=activation,
                                            job=job,
                                            revision=revision,
                                            recorded_at=timezone.now(),
                                            body=body,
                                        ),
                                    )
                                )
            return JobExecutionResult(
                details=s._job_details(
                    identity,
                    assets,
                    scan_started_at=s._time(cutoff),
                    captures=[asset_ref_for(asset).to_json() for asset, _doc, _state in captures],
                )
            )
        except RefreshVerificationError as exc:
            raise s._error(
                exc.reason_code if exc.reason_code in s.SAFE_CODES else "shadow_asset_invalid",
            ) from None
        except _KNOWN_SOURCE_ERRORS:
            raise s._error("shadow_asset_invalid") from None

    try:
        child = execute_target_job(
            job_name=name,
            region="us",
            target_date=evaluation_date,
            task=task,
            before_attempt=before,
        )
    except s.ShadowStudyError as exc:
        if attempted:
            exc.child = JobRun.objects.get(pk=attempted[0].pk)
        raise
    original = child
    if child.status == JobRun.Status.SKIPPED:
        original = JobRun.objects.get(pk=child.details["successful_run_id"])
    verify_scan(original)
    return s.ShadowJobResult(
        child, "available", counts={"assessments": len(original.details["assets"])}
    )

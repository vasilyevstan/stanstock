"""Default-off, append-only prospective study evidence (not an adoption policy).

Only explicit operator calls create the epoch or close enrollment. Registry
timestamps do not prove publication: activation and capture additionally need
an independently recorded, post-commit successful child completion.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import transaction
from django.db.models import Q
from django.utils import timezone
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.jobs import JobExecutionResult, execute_target_job, target_job_lock
from stanstock.core.models import JobRun
from stanstock.core.revision import clean_git_revision
from stanstock.core.verification_types import AssetRef, RefreshVerificationError
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    read_checksummed_bytes,
)
from stanstock.data.models import DataAsset, ProviderRecord
from stanstock.data.provider_policy import ProviderConfigurationError, validate_provider_usage
from stanstock.data.research_product_jobs import (
    SHADOW_ACTIVATE_JOB,
    SHADOW_CAPTURE_JOB,
    SHADOW_CLOSE_JOB,
    SHADOW_EVALUATE_JOB,
    product_job_name,
)

STUDY_ID = "prospective-three-arm-fhs-v1"
ARMS = ("historical_fhs_control", "zero_log_drift_fhs_control", "fixed_half_drift_fhs")
HORIZONS = ("6m", "12m")
KINDS = (
    "shadow_study_protocol",
    "shadow_study_activation",
    "shadow_study_closure",
    "shadow_study_capture",
    "shadow_study_evaluation",
)
SCHEMAS = (
    "shadow-study-protocol@1",
    "shadow-study-activation@1",
    "shadow-study-closure@1",
    "shadow-study-capture@1",
    "shadow-study-evaluation@1",
)
COORDINATION_DATE = date.min
CONFIG_HASH = "55334183af29fc01b853e83f9bf75216f24564956925fb95912cb80a69420867"
ASSUMPTIONS_HASH = "3dff36efe9fa8377cbbd2c0410db066b7d9688d5a2877d6d41b70a0e570aea6b"
PROTOCOL_BODY: dict[str, Any] = {
    "name": "Prospective Three-Arm FHS Evidence Collection and Descriptive Evaluation, v1",
    "version": 1,
    "calendar": "XNYS",
    "epoch_rule": "first_scheduled_close_strictly_after_actual_activation",
    "capture_rule": "every_prescribed_session",
    "anchor_steps": {"6m": 126, "12m": 252},
    "horizon_roles": {"6m": "primary", "12m": "secondary_diagnostic"},
    "unavailable_horizons": ["3y", "5y"],
    "arms": list(ARMS),
    "population_rule": "complete_original_scheduled_intake_and_membership",
    "canonicality": "one_original_scheduled_capture_per_activation_target",
    "maturity_rule": "original_native_observed_session_endpoint",
    "aggregation": "common_support_within_anchor_then_equal_anchor_means",
    "primary_metric": "central_60_interval_score",
    "companion_metric": "median_mean_absolute_error",
    "effect_margin": None,
    "guardrail_tolerance": None,
    "minimum_sample_threshold": None,
    "inference": "none",
    "multiplicity_adjustment": "not_applicable_no_selection",
    "adoption": "forbidden",
    "closure_rule": "administrative_only",
    "native": {
        "product_version": "research-product-v1",
        "method_version": "us-price-fhs-v1",
        "config_hash": CONFIG_HASH,
        "config_file_sha256": "21dcfcb4a3560fe94a7e614bc8e659d6778312b21398cb249caf6e09df78b832",
        "adapter_schema": "shadow-fhs-drift@1",
        "adapter_version": "shadow-fhs-drift-v1",
        "adapter_assumptions_sha256": ASSUMPTIONS_HASH,
        "path_count": 8192,
        "simulation_horizons": [["6m", 126], ["12m", 252], ["3y", 756], ["5y", 1260]],
        "benchmark": "SPY",
        "currency": "USD",
        "return_basis": "split_adjusted_price_return_excluding_dividends",
        "return_decimal_places": 4,
        "price_decimal_places": 6,
        "rounding": "ROUND_HALF_EVEN",
    },
}
PROTOCOL_SHA256 = "452f02fa105b528f4a9dc46825cb1712c8d2142a88d1c7a2bda5cf09d1efc907"

SAFE_CODES = frozenset(
    "shadow_activation_missing shadow_activation_not_available_before_close "
    "shadow_protocol_mismatch shadow_owner_unauthorized shadow_source_missing "
    "shadow_source_identity_invalid shadow_source_not_observed shadow_population_missing "
    "shadow_publication_unverified shadow_deadline_missed shadow_registry_ambiguous "
    "shadow_asset_invalid shadow_preparation_failed shadow_native_outcome_unavailable "
    "shadow_endpoint_unverified shadow_basis_incompatible shadow_cutoff_unverified "
    "shadow_closed_for_target shadow_evidence_challenged".split()
)
ReadStatus = Literal[
    "disabled", "activation_missing", "available", "unauthorized", "integrity_failed"
]
StudyPhase = Literal["inactive", "collecting", "closed"]
CaptureState = Literal[
    "pending",
    "timely",
    "late",
    "missed",
    "ineligible",
    "failed",
    "missing_population",
    "publication_unverified",
]
CaseState = Literal[
    "pending_maturity",
    "non_evaluable",
    "unresolved",
    "accepted",
    "quarantined",
    "accepted_with_challenge",
]
QUALIFICATIONS = frozenset(
    "native_outcome_unavailable source_unavailable cutoff_unverified endpoint_unverified "
    "basis_incompatible verified_native_outcome later_compatible_revision "
    "later_incompatible_revision later_unverifiable_evidence".split()
)


@dataclass(frozen=True, slots=True)
class AnchorSummary:
    target_date: date
    intended_cases: int | None
    accepted: int
    paired: int
    withheld: int
    missed: int | None
    unresolved: int
    challenged: int
    arm_means: tuple[dict[str, Any], ...] | None
    reason_code: str | None


@dataclass(frozen=True, slots=True)
class HorizonSummary:
    horizon: str
    intended_anchors: int = 0
    intended_cases: int | None = None
    accepted: int = 0
    paired: int = 0
    withheld: int = 0
    missed: int | None = None
    unresolved: int = 0
    challenged: int = 0
    per_anchor: tuple[AnchorSummary, ...] = ()
    arm_means: tuple[dict[str, Any], ...] | None = None
    candidate_minus_control: tuple[dict[str, Any], ...] | None = None
    reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class ShadowStudyRead:
    status: ReadStatus
    phase: StudyPhase
    as_of: datetime
    protocol: AssetRef | None = None
    activation: AssetRef | None = None
    closure: AssetRef | None = None
    reason_code: str | None = None
    capture_counts: dict[str, int] = field(default_factory=dict)
    case_counts: dict[str, int] = field(default_factory=dict)
    summaries: tuple[HorizonSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class ShadowJobResult:
    child: JobRun | None
    state: str
    reason_code: str | None = None
    counts: dict[str, int] = field(default_factory=dict)


class ShadowStudyError(RefreshVerificationError):
    """Safe error transport retaining an actual failed child, when one exists."""

    child: JobRun | None = None


def _error(code: str = "shadow_asset_invalid") -> ShadowStudyError:
    if code not in SAFE_CODES:
        raise RuntimeError("Unknown prospective study failure code")
    return ShadowStudyError(code, code)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _time(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise _error("shadow_cutoff_unverified")
    return value.astimezone(UTC).isoformat()


def _parse_time(value: Any) -> datetime:
    if type(value) is not str:
        raise _error()
    try:
        parsed = datetime.fromisoformat(value)
        if _time(parsed) != value:
            raise _error()
        return parsed
    except ValueError:
        raise _error() from None


def _date(value: Any) -> date:
    if type(value) is not str:
        raise _error()
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise _error() from None
    if parsed.isoformat() != value:
        raise _error()
    return parsed


def _uuid(value: Any) -> None:
    if type(value) is not str:
        raise _error()
    try:
        if str(UUID(value)) != value:
            raise _error()
    except ValueError:
        raise _error() from None


def _sha(value: Any, length: int = 64) -> None:
    if type(value) is not str or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None:
        raise _error()


def _integer(value: Any) -> None:
    if type(value) is not int or value < 0:
        raise _error()


def _decimal(value: Any) -> Decimal:
    if type(value) is not str:
        raise _error()
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise _error() from None
    if not result.is_finite():
        raise _error()
    return result


def _keys(value: Any, keys: str) -> None:
    if type(value) is not dict or set(value) != set(keys.split()):
        raise _error()


def _ref(value: Any) -> AssetRef:
    _uuid(value.get("id") if isinstance(value, dict) else None)
    try:
        return AssetRef.from_json(value)
    except RefreshVerificationError:
        raise _error() from None


def _refs(value: Any) -> None:
    if type(value) is not list:
        raise _error()
    for item in value:
        _ref(item)
    if value != sorted(value, key=lambda item: item["id"]) or len(
        {item["id"] for item in value}
    ) != len(value):
        raise _error()


def _reason(value: Any, *, optional: bool = True) -> None:
    if value is None and optional:
        return
    if type(value) is not str or value not in SAFE_CODES:
        raise _error()


def _validate_document(document: Any, kind: str) -> None:
    """Closed schemas, including nested objects; no extensible registry."""
    if kind not in KINDS:
        raise _error()
    _keys(
        document,
        "schema study_id owner_id source_provider protocol activation "
        "execution_revision recorded_at producer_job_id body",
    )
    if (
        document["schema"] != SCHEMAS[KINDS.index(kind)]
        or document["study_id"] != STUDY_ID
        or type(document["owner_id"]) is not str
        or not document["owner_id"]
        or document["source_provider"] not in ("twelve_data", "synthetic_demo")
    ):
        raise _error()
    _sha(document["execution_revision"], 40)
    _parse_time(document["recorded_at"])
    for name in ("protocol", "activation"):
        null_required = kind == KINDS[0] or (name == "activation" and kind == KINDS[1])
        if null_required:
            if document[name] is not None:
                raise _error()
        else:
            _ref(document[name])
    if kind == KINDS[0]:
        if document["producer_job_id"] is not None:
            raise _error()
    else:
        _uuid(document["producer_job_id"])
    body = document["body"]
    if kind == KINDS[0]:
        if _hash(body) != PROTOCOL_SHA256 or _hash(PROTOCOL_BODY) != PROTOCOL_SHA256:
            raise _error("shadow_protocol_mismatch")
    elif kind == KINDS[1]:
        _keys(
            body,
            "t0 s0 s0_scheduled_close calendar_dependency native_config_hash "
            "adapter_assumptions_sha256",
        )
        _parse_time(body["t0"])
        _date(body["s0"])
        _parse_time(body["s0_scheduled_close"])
        _keys(body["calendar_dependency"], "name version")
        if (
            body["calendar_dependency"]["name"] != "exchange_calendars"
            or type(body["calendar_dependency"]["version"]) is not str
            or not body["calendar_dependency"]["version"]
            or body["native_config_hash"] != CONFIG_HASH
            or body["adapter_assumptions_sha256"] != ASSUMPTIONS_HASH
        ):
            raise _error("shadow_protocol_mismatch")
    elif kind == KINDS[2]:
        _keys(body, "closed_at reason")
        _parse_time(body["closed_at"])
        if body["reason"] != "operator_request":
            raise _error()
    elif kind == KINDS[3]:
        _validate_capture_shape(body)
    else:
        _validate_evaluation_shape(body)


def _validate_capture_shape(body: Any) -> None:
    _keys(
        body,
        "target_date session_index scheduled_close next_session_open primary_anchor "
        "secondary_anchor disposition reason_code intake membership population_count source "
        "members forecasts",
    )
    _date(body["target_date"])
    _integer(body["session_index"])
    _parse_time(body["scheduled_close"])
    _parse_time(body["next_session_open"])
    for name in ("primary_anchor", "secondary_anchor"):
        if type(body[name]) is not bool:
            raise _error()
    if body["disposition"] not in ("prepared", "missed", "ineligible"):
        raise _error()
    _reason(body["reason_code"], optional=body["disposition"] == "prepared")
    for name in ("intake", "membership"):
        if body[name] is not None:
            _ref(body[name])
    if type(body["members"]) is not list or type(body["forecasts"]) is not list:
        raise _error()
    if body["population_count"] is None:
        if body["intake"] is not None or body["members"]:
            raise _error()
    else:
        _integer(body["population_count"])
        if body["intake"] is None or len(body["members"]) != body["population_count"]:
            raise _error()
    for index, member in enumerate(body["members"]):
        _keys(member, "request_index listing_id admission source_reason")
        _integer(member["request_index"])
        if member["request_index"] != index:
            raise _error()
        if member["listing_id"] is not None:
            _uuid(member["listing_id"])
        if member["admission"] not in (
            "admitted",
            "identity_rejected",
            "entitlement_rejected",
            "insufficient_history",
            "provider_failed",
            "evidence_invalid",
            "not_materialized",
        ) or (member["admission"] == "admitted" and member["listing_id"] is None):
            raise _error()
        if member["source_reason"] is not None and type(member["source_reason"]) is not str:
            raise _error()
    source = body["source"]
    if source is not None:
        _keys(
            source,
            "run_id snapshot_id generated_at data_cutoff code_revision config_version "
            "config_hash evidence_grade issued_on_time output_manifest",
        )
        _uuid(source["run_id"])
        _uuid(source["snapshot_id"])
        _parse_time(source["generated_at"])
        _parse_time(source["data_cutoff"])
        _sha(source["code_revision"], 40)
        _ref(source["output_manifest"])
        if (
            source["config_version"] != "research-product-v1"
            or source["config_hash"] != CONFIG_HASH
            or source["evidence_grade"] not in ("observed", "research")
            or type(source["issued_on_time"]) is not bool
        ):
            raise _error()
    for forecast in body["forecasts"]:
        _keys(
            forecast,
            "listing_id original_prediction_ids calculation_asset source_assets "
            "preparation reason_code shadow_json shadow_sha256",
        )
        _uuid(forecast["listing_id"])
        _keys(forecast["original_prediction_ids"], "6m 12m")
        for identifier in forecast["original_prediction_ids"].values():
            _uuid(identifier)
        _ref(forecast["calculation_asset"])
        _refs(forecast["source_assets"])
        if forecast["preparation"] == "prepared":
            if forecast["reason_code"] is not None or type(forecast["shadow_json"]) is not str:
                raise _error()
            _sha(forecast["shadow_sha256"])
            if (
                hashlib.sha256(forecast["shadow_json"].encode()).hexdigest()
                != (forecast["shadow_sha256"])
            ):
                raise _error()
        elif forecast["preparation"] in ("failed", "not_attempted"):
            _reason(forecast["reason_code"], optional=False)
            if forecast["shadow_json"] is not None or forecast["shadow_sha256"] is not None:
                raise _error()
        else:
            raise _error()
    identifiers = [item["listing_id"] for item in body["forecasts"]]
    if identifiers != sorted(set(identifiers)):
        raise _error()
    if source is None and body["forecasts"]:
        raise _error()
    if source is not None and set(identifiers) != {
        item["listing_id"] for item in body["members"] if item["admission"] == "admitted"
    }:
        raise _error()
    if body["disposition"] == "prepared" and (source is None or body["reason_code"] is not None):
        raise _error()


def _validate_evaluation_shape(body: Any) -> None:
    _keys(
        body,
        "case_key capture listing_id horizon original_prediction_id state qualification "
        "assessed_at market_through maturity_date native_outcome source_assets prior_accepted "
        "actual_return scores reason_codes evidence_key",
    )
    for key in ("case_key", "evidence_key"):
        _sha(body[key])
    _ref(body["capture"])
    _uuid(body["listing_id"])
    _uuid(body["original_prediction_id"])
    if (
        body["horizon"] not in HORIZONS
        or body["state"]
        not in (
            "unresolved",
            "accepted",
            "quarantined",
            "challenge",
        )
        or body["qualification"] not in QUALIFICATIONS
    ):
        raise _error()
    _parse_time(body["assessed_at"])
    _date(body["market_through"])
    if body["maturity_date"] is not None:
        _date(body["maturity_date"])
    _refs(body["source_assets"])
    if body["prior_accepted"] is not None:
        _ref(body["prior_accepted"])
    if (body["prior_accepted"] is not None) != (body["state"] == "challenge"):
        raise _error()
    if body["state"] == "challenge" and body["qualification"] not in {
        "later_compatible_revision",
        "later_incompatible_revision",
        "later_unverifiable_evidence",
    }:
        raise _error()
    if body["state"] in ("unresolved", "quarantined") and body["actual_return"] is not None:
        raise _error()
    if body["actual_return"] is not None:
        _decimal(body["actual_return"])
    outcome = body["native_outcome"]
    if outcome is not None:
        _keys(outcome, "prediction_id status evaluated_at evaluation_date actual_return row_hash")
        _uuid(outcome["prediction_id"])
        if outcome["prediction_id"] != body["original_prediction_id"]:
            raise _error()
        _parse_time(outcome["evaluated_at"])
        _date(outcome["evaluation_date"])
        _sha(outcome["row_hash"])
        if outcome["status"] not in ("matured", "unresolved", "corporate_event"):
            raise _error()
        if outcome["actual_return"] is not None:
            _decimal(outcome["actual_return"])
    if type(body["scores"]) is not list or type(body["reason_codes"]) is not list:
        raise _error()
    if body["reason_codes"] != sorted(set(body["reason_codes"])):
        raise _error()
    for code in body["reason_codes"]:
        _reason(code, optional=False)
    for row in body["scores"]:
        _keys(row, "arm_id interval_score median_absolute_error")
        if _decimal(row["interval_score"]) < 0 or _decimal(row["median_absolute_error"]) < 0:
            raise _error()
    if body["state"] == "accepted":
        if (
            [row["arm_id"] for row in body["scores"]] != list(ARMS)
            or body["actual_return"] is None
            or body["maturity_date"] is None
            or body["qualification"] != "verified_native_outcome"
            or outcome is None
            or body["reason_codes"]
        ):
            raise _error()
    elif body["scores"]:
        raise _error()
    if body["evidence_key"] != _evidence_key(body):
        raise _error()


def _evidence_key(body: dict[str, Any]) -> str:
    # Scan clocks and advancing pending market dates do not create new evidence.
    return _hash(
        {
            key: value
            for key, value in body.items()
            if key not in ("evidence_key", "assessed_at", "market_through")
        }
    )


def _identity(document: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == KINDS[0]:
        return {key: document[key] for key in ("study_id", "owner_id", "source_provider")}
    if kind == KINDS[1]:
        return {key: document[key] for key in ("protocol", "owner_id", "source_provider")}
    identity = {"activation": document["activation"]}
    if kind == KINDS[3]:
        identity["target_date"] = document["body"]["target_date"]
    elif kind == KINDS[4]:
        identity.update({key: document["body"][key] for key in ("case_key", "evidence_key")})
    return identity


def _metadata(document: dict[str, Any], kind: str) -> dict[str, Any]:
    protocol, activation, body = document["protocol"], document["activation"], document["body"]
    return {
        **{
            key: document[key]
            for key in (
                "schema",
                "study_id",
                "owner_id",
                "source_provider",
                "execution_revision",
                "recorded_at",
                "producer_job_id",
            )
        },
        "protocol_id": None if protocol is None else protocol["id"],
        "protocol_sha256": None if protocol is None else protocol["sha256"],
        "activation_id": None if activation is None else activation["id"],
        "activation_sha256": None if activation is None else activation["sha256"],
        "identity_sha256": _hash(_identity(document, kind)),
        "target_date": body["target_date"] if kind == KINDS[3] else None,
        "listing_id": body["listing_id"] if kind == KINDS[4] else None,
        "horizon": body["horizon"] if kind == KINDS[4] else None,
        "evidence_key": body["evidence_key"] if kind == KINDS[4] else None,
    }


def _envelope(
    *,
    kind: str,
    owner_id: str,
    provider: str,
    revision: str,
    recorded_at: datetime,
    body: dict[str, Any],
    protocol: DataAsset | None = None,
    activation: DataAsset | None = None,
    job: JobRun | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMAS[KINDS.index(kind)],
        "study_id": STUDY_ID,
        "owner_id": owner_id,
        "source_provider": provider,
        "protocol": None if protocol is None else asset_ref_for(protocol).to_json(),
        "activation": None if activation is None else asset_ref_for(activation).to_json(),
        "execution_revision": revision,
        "recorded_at": _time(recorded_at),
        "producer_job_id": None if job is None else str(job.pk),
        "body": body,
    }


def _read_asset(asset: DataAsset, *, store: AssetStore) -> dict[str, Any]:
    try:
        payload = read_checksummed_bytes(store, asset)
        document = json.loads(payload)
        _validate_document(document, asset.kind)
        if (
            _canonical_bytes(document) != payload
            or asset.provider != "stanstock"
            or asset.schema_version != document["schema"]
            or asset.metadata != _metadata(document, asset.kind)
            or asset.subject != _hash(_identity(document, asset.kind))
            or asset.available_at != _parse_time(document["recorded_at"])
            or asset.retrieved_at != asset.available_at
            or asset.relative_path != f"research/shadow/{asset.kind}/{asset.sha256}.json"
            or asset.period_start is not None
            or asset.period_end is not None
        ):
            raise _error()
        return cast(dict[str, Any], document)
    except (OSError, ValueError, TypeError, KeyError):
        raise _error() from None


def _find(
    kind: str,
    identity: dict[str, Any],
    *,
    store: AssetStore,
    cutoff: datetime | None = None,
) -> DataAsset | None:
    subject = _hash(identity)
    # Include conflicting provider rows; a plausible alternative must not be ignored.
    candidates = DataAsset.objects.filter(kind=kind).filter(
        Q(subject=subject) | Q(metadata__identity_sha256=subject),
    )
    if cutoff is not None:
        candidates = candidates.filter(available_at__lte=cutoff)
    assets = list(candidates)
    if len(assets) > 1:
        raise _error("shadow_registry_ambiguous")
    if not assets:
        return None
    document = _read_asset(assets[0], store=store)
    if _identity(document, kind) != identity:
        raise _error()
    return assets[0]


def _register_shadow_asset(
    *,
    kind: str,
    document: dict[str, Any],
    store: AssetStore,
) -> DataAsset:
    """Caller holds the appropriate identity lock through this durable commit.

    In particular, activation/closure/capture must NOT reacquire the
    non-reentrant execute_target_job lock. Evaluation additionally holds a
    cross-evaluation-date case lock.
    """
    _validate_document(document, kind)
    payload = _canonical_bytes(document)
    identity = _identity(document, kind)
    with transaction.atomic(durable=True):
        existing = _find(kind, identity, store=store)
        if existing is not None:
            if read_checksummed_bytes(store, existing) != payload:
                raise _error("shadow_registry_ambiguous")
            return existing
        digest = hashlib.sha256(payload).hexdigest()
        try:
            stored = store.write_bytes(f"research/shadow/{kind}/{digest}.json", payload)
        except (OSError, ValueError):
            raise _error() from None
        timestamp = _parse_time(document["recorded_at"])
        return DataAsset.objects.create(
            provider="stanstock",
            kind=kind,
            subject=_hash(identity),
            relative_path=stored.relative_path,
            sha256=stored.sha256,
            schema_version=document["schema"],
            available_at=timestamp,
            retrieved_at=timestamp,
            metadata=_metadata(document, kind),
        )


def _authorize(owner: object) -> tuple[str, str]:
    user_model = get_user_model()
    if (
        not isinstance(owner, user_model)
        or not user_model.objects.filter(pk=owner.pk, is_active=True).exists()
    ):
        raise _error("shadow_owner_unauthorized")
    owner_id = str(owner.pk)
    if settings.DEMO_MODE:
        return owner_id, "synthetic_demo"
    record = ProviderRecord.objects.filter(provider="twelve_data").first()
    if record is None:
        raise _error("shadow_owner_unauthorized")
    try:
        validate_provider_usage(record, owner_id=owner_id)
    except ProviderConfigurationError:
        raise _error("shadow_owner_unauthorized") from None
    return owner_id, "twelve_data"


def _revision() -> str:
    try:
        revision = clean_git_revision(Path(settings.BASE_DIR))
    except (OSError, ValueError):
        raise _error("shadow_source_identity_invalid") from None
    _sha(revision, 40)
    return revision


def _job_identity(
    owner_id: str,
    provider: str,
    protocol: DataAsset,
    activation: DataAsset | None = None,
) -> dict[str, str]:
    identity = {
        "owner_id": owner_id,
        "provider": provider,
        "study": STUDY_ID,
        "protocol": _hash(asset_ref_for(protocol).to_json()),
    }
    if activation is not None:
        identity["activation"] = _hash(asset_ref_for(activation).to_json())
    return identity


def _job_details(
    identity: dict[str, str],
    assets: list[DataAsset],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "identity": identity,
        "assets": [asset_ref_for(asset).to_json() for asset in assets],
        **extra,
    }


def _witness(
    asset: DataAsset,
    *,
    document: dict[str, Any],
    protocol: DataAsset,
    activation: DataAsset | None,
) -> datetime | None:
    kind = asset.kind
    job_kind = {
        KINDS[1]: SHADOW_ACTIVATE_JOB,
        KINDS[2]: SHADOW_CLOSE_JOB,
        KINDS[3]: SHADOW_CAPTURE_JOB,
        KINDS[4]: SHADOW_EVALUATE_JOB,
    }[kind]
    identity = _job_identity(
        document["owner_id"],
        document["source_provider"],
        protocol,
        activation,
    )
    name = product_job_name(job_kind, identity)
    producer = JobRun.objects.filter(
        pk=document["producer_job_id"],
        job_name=name,
        region="us",
    ).first()
    if producer is None or producer.started_at > asset.available_at:
        raise _error("shadow_publication_unverified")
    target = (
        _date(document["body"]["target_date"])
        if kind == KINDS[3]
        else producer.target_date
        if kind == KINDS[4]
        else COORDINATION_DATE
    )
    if producer.target_date != target:
        raise _error("shadow_publication_unverified")
    witnesses = []
    successes = JobRun.objects.filter(
        job_name=name,
        region="us",
        status=JobRun.Status.SUCCESS,
        finished_at__gte=asset.available_at,
    )
    if kind != KINDS[4]:
        successes = successes.filter(target_date=target)
    for job in successes.order_by("finished_at", "attempt"):
        if (
            job.details.get("identity") == identity
            and asset_ref_for(asset).to_json() in job.details.get("assets", [])
            and job.finished_at is not None
            and job.finished_at >= job.started_at
        ):
            witnesses.append(job.finished_at)
    return max(asset.available_at, min(witnesses)) if witnesses else None


def _calendar() -> Any:
    return get_calendar("XNYS")


def _epoch(t0: datetime) -> tuple[date, datetime]:
    calendar = _calendar()
    session = calendar.date_to_session(t0.date(), direction="next")
    while calendar.session_close(session).to_pydatetime() <= t0:
        session = calendar.next_session(session)
    return session.date(), calendar.session_close(session).to_pydatetime()


def _session(activation: dict[str, Any], target: date) -> dict[str, Any]:
    body = activation["body"]
    calendar = _calendar()
    s0, close = _epoch(_parse_time(body["t0"]))
    if (
        body["calendar_dependency"]
        != {"name": "exchange_calendars", "version": version("exchange_calendars")}
        or body["s0"] != s0.isoformat()
        or body["s0_scheduled_close"] != _time(close)
    ):
        raise _error("shadow_protocol_mismatch")
    if target < s0 or not calendar.is_session(target):
        raise _error("shadow_closed_for_target")
    index = len(calendar.sessions_in_range(s0, target)) - 1
    return {
        "target_date": target.isoformat(),
        "session_index": index,
        "scheduled_close": _time(calendar.session_close(target).to_pydatetime()),
        "next_session_open": _time(
            calendar.session_open(calendar.next_session(target)).to_pydatetime()
        ),
        "primary_anchor": index % 126 == 0,
        "secondary_anchor": index % 252 == 0,
    }


def register_shadow_protocol(*, owner: object, store: AssetStore) -> DataAsset:
    owner_id, provider = _authorize(owner)
    identity = {"study_id": STUDY_ID, "owner_id": owner_id, "source_provider": provider}
    with target_job_lock(
        job_name=f"shadow_protocol:{_hash(identity)}",
        region="us",
        target_date=COORDINATION_DATE,
    ):
        existing = _find(KINDS[0], identity, store=store)
        if existing is not None:
            return existing
        revision = _revision()
        return _register_shadow_asset(
            kind=KINDS[0],
            store=store,
            document=_envelope(
                kind=KINDS[0],
                owner_id=owner_id,
                provider=provider,
                revision=revision,
                recorded_at=timezone.now(),
                body=PROTOCOL_BODY,
            ),
        )


def activate_shadow_study(*, owner: object, store: AssetStore) -> DataAsset:
    if not settings.SHADOW_STUDY_ENABLED:
        raise _error("shadow_activation_missing")
    owner_id, provider = _authorize(owner)
    protocol = register_shadow_protocol(owner=owner, store=store)
    identity = _job_identity(owner_id, provider, protocol)
    activation_identity = {
        "protocol": asset_ref_for(protocol).to_json(),
        "owner_id": owner_id,
        "source_provider": provider,
    }
    recovered: list[DataAsset] = []

    def before() -> None:
        existing = _find(KINDS[1], activation_identity, store=store)
        if existing is not None:
            _verify_activation(existing, protocol=protocol, store=store)
            recovered.append(existing)

    def task(job: JobRun) -> JobExecutionResult:
        if recovered:
            asset = recovered[0]
        else:
            revision = _revision()
            t0 = timezone.now()  # Protocol's durable transaction has already exited.
            if t0 < protocol.available_at:
                raise _error("shadow_cutoff_unverified")
            s0, close = _epoch(t0)
            asset = _register_shadow_asset(
                kind=KINDS[1],
                store=store,
                document=_envelope(
                    kind=KINDS[1],
                    owner_id=owner_id,
                    provider=provider,
                    revision=revision,
                    recorded_at=timezone.now(),
                    protocol=protocol,
                    job=job,
                    body={
                        "t0": _time(t0),
                        "s0": s0.isoformat(),
                        "s0_scheduled_close": _time(close),
                        "calendar_dependency": {
                            "name": "exchange_calendars",
                            "version": version("exchange_calendars"),
                        },
                        "native_config_hash": CONFIG_HASH,
                        "adapter_assumptions_sha256": ASSUMPTIONS_HASH,
                    },
                ),
            )
            recovered.append(asset)
        return JobExecutionResult(details=_job_details(identity, [asset]))

    execute_target_job(
        job_name=product_job_name(SHADOW_ACTIVATE_JOB, identity),
        region="us",
        target_date=COORDINATION_DATE,
        task=task,
        before_attempt=before,
    )
    if not recovered:
        raise _error("shadow_publication_unverified")
    return recovered[0]


def _verify_activation(
    asset: DataAsset,
    *,
    protocol: DataAsset,
    store: AssetStore,
) -> dict[str, Any]:
    document = _read_asset(asset, store=store)
    proto = _read_asset(protocol, store=store)
    if (
        document["protocol"] != asset_ref_for(protocol).to_json()
        or any(document[key] != proto[key] for key in ("owner_id", "source_provider"))
        or _parse_time(document["body"]["t0"]) < protocol.available_at
        or _parse_time(document["body"]["t0"]) > asset.available_at
    ):
        raise _error("shadow_protocol_mismatch")
    _session(document, _date(document["body"]["s0"]))
    return document


def _context(
    *,
    owner: object,
    store: AssetStore,
    closure_cutoff: datetime | None = None,
) -> tuple[str, str, DataAsset, DataAsset, DataAsset | None]:
    owner_id, provider = _authorize(owner)
    protocol = _find(
        KINDS[0],
        {
            "study_id": STUDY_ID,
            "owner_id": owner_id,
            "source_provider": provider,
        },
        store=store,
    )
    if protocol is None:
        raise _error("shadow_activation_missing")
    activation = _find(
        KINDS[1],
        {
            "protocol": asset_ref_for(protocol).to_json(),
            "owner_id": owner_id,
            "source_provider": provider,
        },
        store=store,
    )
    if activation is None:
        raise _error("shadow_activation_missing")
    _verify_activation(activation, protocol=protocol, store=store)
    closure = _find(
        KINDS[2],
        {"activation": asset_ref_for(activation).to_json()},
        store=store,
        cutoff=closure_cutoff,
    )
    if closure is not None:
        document = _read_asset(closure, store=store)
        if (
            document["protocol"] != asset_ref_for(protocol).to_json()
            or document["owner_id"] != owner_id
            or document["source_provider"] != provider
            or _parse_time(document["body"]["closed_at"]) < activation.available_at
            or _parse_time(document["body"]["closed_at"]) > closure.available_at
        ):
            raise _error()
    return owner_id, provider, protocol, activation, closure


def close_shadow_study(*, owner: object, store: AssetStore) -> DataAsset:
    owner_id, provider, protocol, activation, closure = _context(owner=owner, store=store)
    identity = _job_identity(owner_id, provider, protocol, activation)
    recovered: list[DataAsset] = []

    def before() -> None:
        current = _context(owner=owner, store=store)[4]
        if current is not None:
            recovered.append(current)

    def task(job: JobRun) -> JobExecutionResult:
        if not recovered:
            revision = _revision()
            closed_at = timezone.now()
            recovered.append(
                _register_shadow_asset(
                    kind=KINDS[2],
                    store=store,
                    document=_envelope(
                        kind=KINDS[2],
                        owner_id=owner_id,
                        provider=provider,
                        protocol=protocol,
                        activation=activation,
                        job=job,
                        revision=revision,
                        recorded_at=closed_at,
                        body={"closed_at": _time(closed_at), "reason": "operator_request"},
                    ),
                )
            )
        return JobExecutionResult(details=_job_details(identity, recovered))

    execute_target_job(
        job_name=product_job_name(SHADOW_CLOSE_JOB, identity),
        region="us",
        target_date=COORDINATION_DATE,
        task=task,
        before_attempt=before,
    )
    if not recovered:
        raise _error("shadow_publication_unverified")
    return recovered[0]


def _means(rows: list[list[dict[str, str]]]) -> tuple[dict[str, Any], ...] | None:
    if not rows:
        return None
    return tuple(
        {
            "arm_id": arm,
            **{
                metric: sum((_decimal(row[index][metric]) for row in rows), Decimal(0)) / len(rows)
                for metric in ("interval_score", "median_absolute_error")
            },
        }
        for index, arm in enumerate(ARMS)
    )


def _summary(horizon: str, anchors: list[AnchorSummary]) -> HorizonSummary:
    population_known = all(anchor.intended_cases is not None for anchor in anchors)
    missed_known = all(anchor.missed is not None for anchor in anchors)
    available = bool(anchors) and all(anchor.arm_means is not None for anchor in anchors)
    means = (
        _means(
            [
                [{key: str(value) for key, value in row.items()} for row in anchor.arm_means]
                for anchor in anchors
                if anchor.arm_means is not None
            ]
        )
        if available
        else None
    )
    differences = (
        None
        if means is None
        else tuple(
            {
                "control": means[index]["arm_id"],
                **{
                    metric: means[2][metric] - means[index][metric]
                    for metric in ("interval_score", "median_absolute_error")
                },
            }
            for index in range(2)
        )
    )
    return HorizonSummary(
        horizon=horizon,
        intended_anchors=len(anchors),
        intended_cases=sum(
            anchor.intended_cases for anchor in anchors if anchor.intended_cases is not None
        )
        if population_known and anchors
        else None,
        accepted=sum(anchor.accepted for anchor in anchors),
        paired=sum(anchor.paired for anchor in anchors),
        withheld=sum(anchor.withheld for anchor in anchors),
        missed=sum(anchor.missed for anchor in anchors if anchor.missed is not None)
        if missed_known and anchors
        else None,
        unresolved=sum(anchor.unresolved for anchor in anchors),
        challenged=sum(anchor.challenged for anchor in anchors),
        per_anchor=tuple(anchors),
        arm_means=means,
        candidate_minus_control=differences,
        reason_code=None
        if means is not None
        else (
            "shadow_population_missing"
            if not population_known
            else "shadow_native_outcome_unavailable"
        ),
    )


def read_shadow_study(
    *,
    owner: object,
    as_of: datetime,
    store: AssetStore,
) -> ShadowStudyRead:
    """Read only verified registered evidence; never run forecasts or evaluations."""
    if not settings.SHADOW_STUDY_ENABLED:
        return ShadowStudyRead("disabled", "inactive", as_of)
    from stanstock.research.shadow_jobs import _KNOWN_SOURCE_ERRORS

    _time(as_of)
    try:
        return _read_study(owner=owner, as_of=as_of, store=store)
    except RefreshVerificationError as exc:
        code = exc.reason_code if exc.reason_code in SAFE_CODES else "shadow_asset_invalid"
        status: ReadStatus = (
            "unauthorized"
            if code == "shadow_owner_unauthorized"
            else "activation_missing"
            if code == "shadow_activation_missing"
            else "integrity_failed"
        )
        return ShadowStudyRead(
            status,
            "inactive",
            as_of,
            reason_code=code,
            summaries=tuple(HorizonSummary(horizon, reason_code=code) for horizon in HORIZONS),
        )
    except _KNOWN_SOURCE_ERRORS:
        return ShadowStudyRead(
            "integrity_failed",
            "inactive",
            as_of,
            reason_code="shadow_asset_invalid",
            summaries=tuple(
                HorizonSummary(horizon, reason_code="shadow_asset_invalid") for horizon in HORIZONS
            ),
        )


def _failed_study_jobs(identity: dict[str, str], as_of: datetime) -> list[JobRun]:
    capture_name = product_job_name(SHADOW_CAPTURE_JOB, identity)
    evaluation_name = product_job_name(SHADOW_EVALUATE_JOB, identity)
    seen = set()
    failed = []
    for job in JobRun.objects.filter(
        job_name__in=(capture_name, evaluation_name),
        region="us",
        finished_at__lte=as_of,
        target_date__lte=as_of.date(),
        status__in=(JobRun.Status.FAILED, JobRun.Status.SUCCESS, JobRun.Status.NO_DATA),
    ).order_by("-finished_at", "-attempt", "-id"):
        # Capture targets cannot replace one another; a newer evaluation scan
        # supersedes an older failed scan across evaluation dates.
        key = (job.job_name, job.target_date if job.job_name == capture_name else None)
        if key in seen:
            continue
        seen.add(key)
        if job.status == JobRun.Status.FAILED:
            failed.append(job)
    return failed


def _read_study(*, owner: object, as_of: datetime, store: AssetStore) -> ShadowStudyRead:
    # Local import keeps lifecycle registration independent of the adapter.
    from stanstock.research import shadow_jobs as jobs

    owner_id, provider, protocol, activation, closure = _context(
        owner=owner, store=store, closure_cutoff=as_of
    )
    active = _verify_activation(activation, protocol=protocol, store=store)
    witness = _witness(activation, document=active, protocol=protocol, activation=None)
    if witness is None:
        raise _error("shadow_publication_unverified")
    if witness > as_of:
        raise _error("shadow_activation_missing")
    end = as_of
    phase: StudyPhase = "collecting"
    visible_closure = None
    if closure is not None:
        document = _read_asset(closure, store=store)
        closed_witness = _witness(
            closure,
            document=document,
            protocol=protocol,
            activation=activation,
        )
        if closed_witness is None:
            raise _error("shadow_publication_unverified")
        if closed_witness <= as_of:
            phase = "closed"
            visible_closure = asset_ref_for(closure)
            end = min(end, _parse_time(document["body"]["closed_at"]))
    captures = jobs._visible_captures(
        owner_id=owner_id,
        provider=provider,
        protocol=protocol,
        activation=activation,
        cutoff=as_of,
        store=store,
    )
    by_date = {
        _date(doc["body"]["target_date"]): (asset, doc, state) for asset, doc, state in captures
    }
    identity = _job_identity(owner_id, provider, protocol, activation)
    failures = _failed_study_jobs(identity, as_of)
    failed_capture_targets = {
        job.target_date
        for job in failures
        if job.job_name == product_job_name(SHADOW_CAPTURE_JOB, identity)
    }
    safe_errors = {f"ShadowStudyError: {code}": code for code in SAFE_CODES}
    failure_reason = (
        safe_errors.get(failures[0].error, "shadow_asset_invalid") if failures else None
    )
    counts = {
        name: 0
        for name in (
            "pending",
            "timely",
            "late",
            "missed",
            "ineligible",
            "failed",
            "missing_population",
            "publication_unverified",
        )
    }
    case_counts = {
        name: 0
        for name in (
            "pending_maturity",
            "non_evaluable",
            "unresolved",
            "accepted",
            "quarantined",
            "accepted_with_challenge",
        )
    }
    anchors: dict[str, list[AnchorSummary]] = {horizon: [] for horizon in HORIZONS}
    start = _date(active["body"]["s0"])
    calendar = _calendar()
    sessions = [] if end.date() < start else calendar.sessions_in_range(start, end.date())
    for timestamp in sessions:
        target = timestamp.date()
        session = _session(active, target)
        if _parse_time(session["scheduled_close"]) > end:
            continue
        captured = by_date.get(target)
        state = "pending" if as_of < _parse_time(session["next_session_open"]) else "missed"
        population = None
        members: list[dict[str, Any]] = []
        forecasts: list[dict[str, Any]] = []
        if captured is not None:
            _capture_asset, document, state = captured
            population = document["body"]["population_count"]
            members = document["body"]["members"]
            forecasts = document["body"]["forecasts"]
        else:
            population_source, _intake = jobs._population(
                owner_id=owner_id,
                provider=provider,
                target=target,
                cutoff=as_of,
                store=store,
            )
            population = population_source["population_count"]
            members = population_source["members"]
            if target in failed_capture_targets:
                state = "failed"
        if any(forecast["preparation"] == "failed" for forecast in forecasts):
            failure_reason = failure_reason or "shadow_preparation_failed"
        counts[state] += 1
        for horizon, step in (("6m", 126), ("12m", 252)):
            accepted = challenged = withheld = unresolved = 0
            scores = []
            for forecast in forecasts:
                if state != "timely" or forecast["preparation"] != "prepared":
                    case_counts["non_evaluable"] += 1
                    withheld += 1
                    continue
                assert captured is not None
                capture_asset, capture, _state = captured
                projections = jobs._adapter_document(forecast)["projections"]
                if any(
                    row["ledger_returns"] is None
                    for row in projections
                    if row["horizon"] == horizon
                ):
                    case_counts["non_evaluable"] += 1
                    withheld += 1
                    continue
                records = jobs._case_records(
                    activation=activation,
                    case_key=jobs._case_key(
                        activation,
                        capture,
                        forecast["listing_id"],
                        horizon,
                    ),
                    capture_asset=capture_asset,
                    capture=capture,
                    forecast=forecast,
                    protocol=protocol,
                    store=store,
                    cutoff=as_of,
                )
                visible = [
                    (asset, doc)
                    for asset, doc in records
                    if (
                        available := _witness(
                            asset,
                            document=doc,
                            protocol=protocol,
                            activation=activation,
                        )
                    )
                    is not None
                    and available <= as_of
                ]
                first = next(
                    (item for item in visible if item[1]["body"]["state"] == "accepted"), None
                )
                if first is not None:
                    challenges = [
                        doc for _asset, doc in visible if doc["body"]["state"] == "challenge"
                    ]
                    if any(
                        doc["body"]["prior_accepted"] != asset_ref_for(first[0]).to_json()
                        for doc in challenges
                    ):
                        raise _error("shadow_source_identity_invalid")
                    is_challenged = bool(challenges)
                    challenged += int(is_challenged)
                    accepted += 1
                    scores.append(first[1]["body"]["scores"])
                    case_counts["accepted_with_challenge" if is_challenged else "accepted"] += 1
                else:
                    state_name = "pending_maturity"
                    if visible:
                        state_name = (
                            "quarantined"
                            if visible[-1][1]["body"]["state"] == "quarantined"
                            else "unresolved"
                        )
                    case_counts[state_name] += 1
                    unresolved += 1
            withheld += sum(item["admission"] != "admitted" for item in members)
            if session["session_index"] % step == 0:
                means = _means(scores)
                anchors[horizon].append(
                    AnchorSummary(
                        target_date=target,
                        intended_cases=population,
                        accepted=accepted,
                        paired=len(scores),
                        withheld=withheld,
                        missed=(population if state != "timely" else 0)
                        if population is not None
                        else None,
                        unresolved=unresolved,
                        challenged=challenged,
                        arm_means=means,
                        reason_code=None
                        if means is not None
                        else (
                            "shadow_population_missing"
                            if population is None
                            else "shadow_native_outcome_unavailable"
                        ),
                    )
                )
    return ShadowStudyRead(
        "available",
        phase,
        as_of,
        protocol=asset_ref_for(protocol),
        activation=asset_ref_for(activation),
        closure=visible_closure,
        reason_code=failure_reason
        or ("shadow_evidence_challenged" if case_counts["accepted_with_challenge"] else None),
        capture_counts=counts,
        case_counts=case_counts,
        summaries=tuple(_summary(horizon, anchors[horizon]) for horizon in HORIZONS),
    )

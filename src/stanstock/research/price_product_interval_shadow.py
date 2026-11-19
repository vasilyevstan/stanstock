"""Owner-bound prospective shadow report for interval-calibration research.

The declaration is append-only evidence. Reports re-verify its registered
study, recompute its fitted parameters, authenticate matured outcomes, and
never publish calibrated predictions or write report assets.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from math import comb
from typing import Any, cast

from django.conf import settings
from django.db import IntegrityError, transaction
from django.utils import timezone
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun
from stanstock.core.revision import clean_git_revision
from stanstock.core.verification_types import AssetRef, RefreshVerificationError
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    read_checksummed_bytes,
    register_asset,
    resolve_asset_ref,
)
from stanstock.data.models import DataAsset
from stanstock.data.research_product_demo import DEMO_OWNER_ID
from stanstock.research import forward_scorecard
from stanstock.research.models import Prediction
from stanstock.research.outcome_refresh_validation import verify_prediction_outcome
from stanstock.research.price_product import LedgerTriplet
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_VERSION,
    load_price_product_config,
)
from stanstock.research.price_product_interval_calibration import (
    METHOD_VERSION,
    adjust_interval,
    canonical_json_bytes,
    canonical_sha256,
    summarize_cluster_coverage,
)
from stanstock.research.price_product_interval_calibration_document import (
    fit_interval_calibration_document,
)
from stanstock.research.price_product_replay import (
    ReplayMetricObservation,
    compare_aligned_models,
    score_projection_metrics,
)
from stanstock.research.product_reader import (
    ProductUser,
    read_research_product_performance,
)
from stanstock.research.product_study_evidence import (
    STUDY_EVIDENCE_KIND,
    _read_registered_document,
    _study_subject_prefix,
    _verify_registered_asset,
    read_registered_price_product_study,
)
from stanstock.research.reporting import (
    _select_non_overlapping_summaries,
    iter_product_prediction_projection,
)

PROTOCOL_VERSION = "interval-calibration-shadow-v1"
DECLARATION_KIND = "interval_calibration_shadow_protocol"
DECLARATION_SCHEMA = "research-product-interval-calibration-shadow-declaration@1"
REPORT_SCHEMA = "research-product-interval-calibration-shadow-report@1"
DECLARE_JOB = "interval_calibration_shadow_declare_v1"
_REGION = "us"
_COORDINATION_DATE = date.min
_PRIMARY_COHORTS = 8
_DENOMINATOR_KEYS = (
    "raw_rows",
    "non_evaluable",
    "not_canonical",
    "excluded_reissue",
    "config_or_provider_mismatch",
    "pre_declaration_excluded",
    "eligible",
    "pending",
    "unresolved",
    "corporate_event",
    "matured",
    "candidate_unrepresentable",
    "candidate_score_unavailable",
    "native_score_unavailable",
    "clamped_lower",
    "clamped_upper",
)

SHADOW_PROTOCOL: dict[str, Any] = {
    "version": PROTOCOL_VERSION,
    "product_version": PRODUCT_VERSION,
    "effective_config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
    "method_version": "us-price-fhs-v1",
    "evidence_role": "advisory",
    "params_method": METHOD_VERSION,
    "params_schema": "research-product-interval-calibration-params@1",
    "horizons": {
        "primary": "6m",
        "descriptive": "12m",
        "unavailable": ["3y", "5y"],
    },
    "eligibility": [
        "canonical",
        "generated_at > witness_time",
        "target_date >= first_XNYS_close_strictly_after_witness_time",
        "provider_and_config_match",
    ],
    "scorer": {
        "name": "score_projection_metrics",
        "central_coverage": "0.6",
        "alpha": "0.4",
        "interval_inclusion": "inclusive",
    },
    "arms": {"candidate": "calibrated", "baseline": "native"},
    "non_overlap": "reporting._select_non_overlapping_summaries",
    "primary": {
        "selected_cohorts": 8,
        "floors": {"cohorts": 8, "listings_per_cohort": 30, "span_days": 1095},
        "criterion": "exact_sign_k_at_least_7_of_8_and_cohort_first_mean_difference_lt_0",
        "ties": "failure",
        "fixed_denominator": True,
    },
    "companion": {
        "metric": "cohort_mean_coverage",
        "target": "0.6",
        "criterion": "calibrated_abs_deviation_le_native",
        "same_cohorts_as_primary": True,
    },
    "s4_gate": "rev-2-section-9",
}


# Pin filled after the protocol is frozen; checked by the pin test and on use.
SHADOW_PROTOCOL_SHA256 = "54ea741a9cc4f957c67814ef22a24cc4dfa88326fb7436695aead044abd25f9e"


class IntervalShadowError(ValueError):
    """A path-free, allowlisted failure code suitable for JobRun.error."""

    def __init__(self, code: str) -> None:
        if not _is_allowed_code(code):
            code = "interval_shadow_declaration_invalid"
        self.code = code
        super().__init__(code)


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _revision() -> str:
    return clean_git_revision(settings.BASE_DIR)


def _fit_params(document: dict[str, Any]) -> dict[str, Any]:
    """One named seam around the public S2 adapter, never a private core fit."""
    return fit_interval_calibration_document(document)


def _fit_params_and_calendar(
    document: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    fitted = _fit_params(document)
    params = cast(dict[str, Any], fitted["params"])
    source = cast(dict[str, Any], fitted["source"])
    return params, cast(dict[str, str], source["calendar_dependency"])


def _authenticated_outcome(row: dict[str, Any], provider: str) -> Any:
    prediction = Prediction.objects.select_related(
        "analysis__run__universe_snapshot", "listing", "outcome"
    ).get(pk=row["id"])
    outcome = prediction.outcome
    now = timezone.now()
    if outcome.evaluated_at > now:
        raise IntervalShadowError("interval_shadow_outcome_unverified")
    verify_prediction_outcome(
        prediction,
        outcome,
        provider=provider,
        benchmark_subject="SPY",
        evaluation_time=outcome.evaluated_at,
        parent_target_date=outcome.evaluation_date,
    )
    return outcome


def _authority(user: ProductUser, store: AssetStore) -> tuple[str, str, str]:
    try:
        owner_id, source_owner, provider = forward_scorecard._authority(
            user, store, PRODUCT_VERSION
        )
    except (ValueError, RefreshVerificationError):
        raise IntervalShadowError("interval_shadow_owner_unauthorized") from None
    if hashlib.sha256(_bytes(SHADOW_PROTOCOL)).hexdigest() != SHADOW_PROTOCOL_SHA256:
        raise IntervalShadowError("interval_shadow_protocol_mismatch")
    return owner_id, source_owner, provider


def _declaration_subject(owner_id: str) -> str:
    return f"{PROTOCOL_VERSION}:{owner_id}"


def _identity(owner_id: str, source_owner: str, provider: str) -> dict[str, str]:
    return {
        "owner_id": owner_id,
        "source_owner_id": source_owner,
        "source_provider": provider,
        "protocol_sha256": SHADOW_PROTOCOL_SHA256,
    }


def _job_name(identity: dict[str, str]) -> str:
    return f"{DECLARE_JOB}:{hashlib.sha256(_bytes(identity)).hexdigest()[:32]}"


def _source_study(
    *,
    user: ProductUser,
    store: AssetStore,
    source_owner: str,
    provider: str,
) -> tuple[DataAsset, dict[str, Any], Any]:
    read = read_registered_price_product_study(user=user, store=store)
    if read.status == "absent":
        raise IntervalShadowError("interval_shadow_source_study_absent")
    if not read.available or read.source_run_id is None:
        raise IntervalShadowError("interval_shadow_source_study_unverified")
    expected_owner = DEMO_OWNER_ID if settings.DEMO_MODE else str(user.pk)
    assets = DataAsset.objects.filter(
        provider="stanstock",
        kind=STUDY_EVIDENCE_KIND,
        subject__startswith=_study_subject_prefix(expected_owner),
    ).order_by("-available_at", "-retrieved_at", "-id")
    asset = assets.first()
    if asset is None:
        raise IntervalShadowError("interval_shadow_source_study_unverified")
    if asset.metadata.get("source_run_id") != str(read.source_run_id):
        raise IntervalShadowError("interval_shadow_source_study_changed")
    try:
        document = _read_registered_document(asset=asset, store=store)
        verified = _verify_registered_asset(
            asset=asset,
            document=document,
            store=store,
            expected_owner=expected_owner,
            expected_provider=provider,
            verification=None,
        )
    except (RefreshVerificationError, KeyError, TypeError, ValueError):
        raise IntervalShadowError("interval_shadow_source_study_unverified") from None
    if verified.owner_id != source_owner or verified.source_provider != provider:
        raise IntervalShadowError("interval_shadow_owner_mismatch")
    return asset, document, verified


def declare_interval_shadow(*, user: ProductUser, store: AssetStore | None = None) -> DataAsset:
    """Publish one owner-bound declaration using the generic target-job witness."""
    store = store or AssetStore()
    owner_id, source_owner, provider = _authority(user, store)
    identity = _identity(owner_id, source_owner, provider)
    job_name = _job_name(identity)
    subject = _declaration_subject(owner_id)
    recovered: tuple[DataAsset, dict[str, Any]] | None = None

    def before_attempt() -> JobRun | None:
        nonlocal recovered
        assets = list(
            DataAsset.objects.filter(provider="stanstock", kind=DECLARATION_KIND, subject=subject)
        )
        if len(assets) > 1:
            raise IntervalShadowError("interval_shadow_registry_ambiguous")
        if assets:
            recovered = _verify_declaration(
                assets[0],
                user=user,
                store=store,
                expected_identity=identity,
                require_witness=False,
                expected_job_name=job_name,
            )
        return None

    def task(job: JobRun) -> JobExecutionResult:
        nonlocal recovered
        try:
            if recovered is not None:
                asset, _document = recovered
                return JobExecutionResult(
                    details={"identity": identity, "assets": [asset_ref_for(asset).to_json()]}
                )
            study, study_document, verified = _source_study(
                user=user, store=store, source_owner=source_owner, provider=provider
            )
            try:
                params, calendar_dependency = _fit_params_and_calendar(study_document)
            except ValueError:
                raise IntervalShadowError("interval_shadow_params_invalid") from None
            six = params.get("horizons", {}).get("6m", {})
            if six.get("state") != "usable":
                reason = six.get("reason")
                if reason not in {
                    "no_scored_observations",
                    "log_scale_boundary_observation",
                    "conformal_rank_infeasible",
                    "insufficient_effective_support",
                }:
                    reason = "insufficient_effective_support"
                raise IntervalShadowError(f"interval_shadow_params_withheld:{reason}")
            try:
                revision = _revision()
            except ValueError:
                raise IntervalShadowError("interval_shadow_revision_unverified") from None
            declared_at = timezone.now()
            if (
                declared_at.tzinfo is None
                or study.available_at > declared_at
                or job.started_at > declared_at
            ):
                raise IntervalShadowError("interval_shadow_clock_invalid")
            try:
                start_target = forward_scorecard.prospective_start_target(declared_at)
                _require_embargo(params=params, start_target=start_target)
            except IntervalShadowError:
                raise
            except ValueError:
                raise IntervalShadowError("interval_shadow_embargo_violation") from None
            document = {
                "schema": DECLARATION_SCHEMA,
                "protocol": SHADOW_PROTOCOL,
                "protocol_sha256": SHADOW_PROTOCOL_SHA256,
                "owner_id": owner_id,
                "source_owner_id": source_owner,
                "source_provider": provider,
                "study": asset_ref_for(study).to_json(),
                "study_full_report_sha256": study.metadata["full_report_sha256"],
                "params": params,
                "params_sha256": params["params_sha256"],
                "declared_at": declared_at.isoformat(),
                "producer_job_id": str(job.pk),
                "execution_revision": revision,
                "calendar_dependency": calendar_dependency,
            }
            metadata = {
                "owner_id": owner_id,
                "protocol_version": PROTOCOL_VERSION,
                "protocol_sha256": SHADOW_PROTOCOL_SHA256,
                "declared_at": declared_at.isoformat(),
                "params_sha256": params["params_sha256"],
                "study_asset_id": str(study.pk),
            }
            payload = _bytes(document)
            relative_path = (
                f"research/interval_shadow/{owner_id}/{hashlib.sha256(payload).hexdigest()}.json"
            )
            try:
                stored = store.write_bytes(relative_path, payload)
                with transaction.atomic(durable=True):
                    if DataAsset.objects.filter(
                        provider="stanstock", kind=DECLARATION_KIND, subject=subject
                    ).exists():
                        raise IntervalShadowError("interval_shadow_registry_ambiguous")
                    asset = register_asset(
                        provider="stanstock",
                        kind=DECLARATION_KIND,
                        subject=subject,
                        stored=stored,
                        retrieved_at=declared_at,
                        available_at=declared_at,
                        metadata=metadata,
                    )
            except Exception:
                if not DataAsset.objects.filter(relative_path=relative_path).exists():
                    try:
                        store.resolve(relative_path).unlink(missing_ok=True)
                    except (OSError, ValueError):
                        pass
                raise
            # Keep this reference in memory only. Full witness verification is
            # intentionally deferred until execute_target_job records SUCCESS.
            recovered = (asset, document)
            return JobExecutionResult(
                details={"identity": identity, "assets": [asset_ref_for(asset).to_json()]}
            )
        except IntervalShadowError:
            raise
        except (IntegrityError, OSError, RefreshVerificationError):
            raise IntervalShadowError("interval_shadow_storage_failed") from None
        except Exception:
            # Never persist arbitrary verifier, filesystem or exception text.
            raise IntervalShadowError("interval_shadow_declaration_invalid") from None

    try:
        run = execute_target_job(
            job_name=job_name,
            region=_REGION,
            target_date=_COORDINATION_DATE,
            task=task,
            before_attempt=before_attempt,
        )
    except IntervalShadowError:
        raise
    except Exception:
        raise IntervalShadowError("interval_shadow_declaration_invalid") from None
    if run.status == JobRun.Status.SKIPPED:
        if recovered is None:
            raise IntervalShadowError("interval_shadow_publication_unverified")
        asset, _document = recovered
    elif recovered is not None:
        asset, _document = recovered
    else:
        raise IntervalShadowError("interval_shadow_publication_unverified")
    _verify_declaration(
        asset,
        user=user,
        store=store,
        expected_identity=identity,
        require_witness=True,
        expected_job_name=job_name,
    )
    return asset


def read_interval_shadow_report(
    *, user: ProductUser, store: AssetStore | None = None
) -> dict[str, Any]:
    """Verify the declaration afresh, then describe eligible prospective rows."""
    store = store or AssetStore()
    owner_id, source_owner, provider = _authority(user, store)
    identity = _identity(owner_id, source_owner, provider)
    expected_job_name = _job_name(identity)
    attempts = _declaration_attempts(expected_job_name)
    assets = list(
        DataAsset.objects.filter(
            provider="stanstock",
            kind=DECLARATION_KIND,
            subject=_declaration_subject(owner_id),
        )
    )
    if not assets:
        return _report_base(
            status="not_declared",
            verification_code=None,
            attempts=attempts,
            evidence_label=_evidence_label(provider),
        )
    if len(assets) != 1:
        return _integrity_report(
            code="interval_shadow_registry_ambiguous",
            attempts=attempts,
            evidence_label=_evidence_label(provider),
        )
    try:
        asset, document = _verify_declaration(
            assets[0],
            user=user,
            store=store,
            expected_identity=identity,
            require_witness=True,
            expected_job_name=expected_job_name,
        )
        params = document["params"]
        declaration = _declaration_summary(asset, document)
        report = _build_report(
            user=user,
            store=store,
            provider=provider,
            params=params,
            declaration=declaration,
            attempts=attempts,
        )
        report["report_read_at"] = timezone.now().isoformat()
        return report
    except IntervalShadowError as exc:
        return _integrity_report(
            code=exc.code,
            attempts=attempts,
            evidence_label=_evidence_label(provider),
        )
    except (RefreshVerificationError, KeyError, TypeError, ValueError):
        return _integrity_report(
            code="interval_shadow_declaration_invalid",
            attempts=attempts,
            evidence_label=_evidence_label(provider),
        )


def _verify_declaration(
    asset: DataAsset,
    *,
    user: ProductUser,
    store: AssetStore,
    expected_identity: dict[str, str],
    require_witness: bool,
    expected_job_name: str,
) -> tuple[DataAsset, dict[str, Any]]:
    try:
        raw = read_checksummed_bytes(store, asset)
        document = json.loads(raw)
    except (RefreshVerificationError, json.JSONDecodeError):
        raise IntervalShadowError("interval_shadow_declaration_invalid") from None
    expected_keys = {
        "schema",
        "protocol",
        "protocol_sha256",
        "owner_id",
        "source_owner_id",
        "source_provider",
        "study",
        "study_full_report_sha256",
        "params",
        "params_sha256",
        "declared_at",
        "producer_job_id",
        "execution_revision",
        "calendar_dependency",
    }
    if (
        not isinstance(document, dict)
        or set(document) != expected_keys
        or asset.provider != "stanstock"
        or asset.kind != DECLARATION_KIND
    ):
        raise IntervalShadowError("interval_shadow_declaration_invalid")
    if asset.subject != _declaration_subject(expected_identity["owner_id"]):
        raise IntervalShadowError("interval_shadow_owner_mismatch")
    if (
        document["schema"] != DECLARATION_SCHEMA
        or document["protocol"] != SHADOW_PROTOCOL
        or document["protocol_sha256"] != SHADOW_PROTOCOL_SHA256
    ):
        raise IntervalShadowError("interval_shadow_protocol_mismatch")
    if any(document.get(key) != value for key, value in expected_identity.items()):
        raise IntervalShadowError("interval_shadow_owner_mismatch")
    try:
        declared_at = datetime.fromisoformat(document["declared_at"])
    except (TypeError, ValueError):
        raise IntervalShadowError("interval_shadow_declaration_invalid") from None
    metadata = {
        "owner_id": expected_identity["owner_id"],
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": SHADOW_PROTOCOL_SHA256,
        "declared_at": document["declared_at"],
        "params_sha256": document["params_sha256"],
        "study_asset_id": str(_asset_ref(document["study"]).id),
    }
    if (
        asset.metadata != metadata
        or declared_at.tzinfo is None
        or declared_at > timezone.now()
        or asset.available_at != declared_at
        or asset.retrieved_at != declared_at
        or re.fullmatch(r"[0-9a-f]{40}", str(document["execution_revision"])) is None
    ):
        raise IntervalShadowError("interval_shadow_clock_invalid")
    try:
        study_ref = _asset_ref(document["study"])
        study = resolve_asset_ref(study_ref, cutoff=timezone.now())
        study_document = _read_registered_document(asset=study, store=store)
        expected_owner = DEMO_OWNER_ID if settings.DEMO_MODE else str(user.pk)
        verified = _verify_registered_asset(
            asset=study,
            document=study_document,
            store=store,
            expected_owner=expected_owner,
            expected_provider=expected_identity["source_provider"],
            verification=None,
        )
    except (RefreshVerificationError, KeyError, TypeError, ValueError):
        raise IntervalShadowError("interval_shadow_source_study_unverified") from None
    if (
        document["study_full_report_sha256"] != study.metadata.get("full_report_sha256")
        or verified.owner_id != expected_identity["source_owner_id"]
        or verified.source_provider != expected_identity["source_provider"]
        or study.available_at > declared_at
        or study.retrieved_at > declared_at
        or verified.report_generated_at > study.available_at
        or verified.source_run.generated_at > verified.report_generated_at
    ):
        raise IntervalShadowError("interval_shadow_owner_mismatch")
    try:
        params, calendar_dependency = _fit_params_and_calendar(study_document)
    except ValueError:
        raise IntervalShadowError("interval_shadow_params_invalid") from None
    declared_params = document.get("params")
    if not isinstance(declared_params, dict):
        raise IntervalShadowError("params_recompute_mismatch")
    hash_payload = dict(declared_params)
    declared_params_sha = hash_payload.pop("params_sha256", None)
    if (
        canonical_json_bytes(params) != canonical_json_bytes(declared_params)
        or params.get("params_sha256") != document.get("params_sha256")
        or declared_params_sha != document.get("params_sha256")
        or canonical_sha256(hash_payload) != document.get("params_sha256")
        or document.get("calendar_dependency") != calendar_dependency
    ):
        raise IntervalShadowError("params_recompute_mismatch")
    if params.get("horizons", {}).get("6m", {}).get("state") != "usable":
        raise IntervalShadowError("interval_shadow_params_withheld:insufficient_effective_support")

    producer = JobRun.objects.filter(
        pk=document["producer_job_id"],
        job_name=expected_job_name,
        region=_REGION,
        target_date=_COORDINATION_DATE,
    ).first()
    if (
        producer is None
        or producer.started_at > declared_at
        or producer.finished_at is not None
        and producer.finished_at < producer.started_at
    ):
        raise IntervalShadowError("interval_shadow_clock_invalid")
    start_target = _prospective_start(declared_at)
    _require_embargo(params=params, start_target=start_target)
    if require_witness:
        witness = _witness(asset, identity=expected_identity, job_name=expected_job_name)
        if witness is None:
            raise IntervalShadowError("interval_shadow_publication_unverified")
        if witness > timezone.now():
            raise IntervalShadowError("interval_shadow_clock_invalid")
        start_target = _prospective_start(witness)
        _require_embargo(params=params, start_target=start_target)
    document["_verified_start_target"] = start_target
    return asset, document


def _asset_ref(raw: object) -> AssetRef:
    try:
        return AssetRef.from_json(raw)
    except (RefreshVerificationError, TypeError, ValueError):
        raise IntervalShadowError("interval_shadow_declaration_invalid") from None


def _witness(asset: DataAsset, *, identity: dict[str, str], job_name: str) -> datetime | None:
    asset_ref = asset_ref_for(asset).to_json()
    witnesses: list[datetime] = []
    successes = JobRun.objects.filter(
        job_name=job_name,
        region=_REGION,
        target_date=_COORDINATION_DATE,
        status=JobRun.Status.SUCCESS,
        finished_at__gte=asset.available_at,
    ).order_by("finished_at", "attempt")
    for run in successes:
        if (
            run.details.get("identity") == identity
            and asset_ref in run.details.get("assets", [])
            and run.finished_at is not None
            and run.finished_at >= run.started_at
        ):
            witnesses.append(run.finished_at)
    return max(asset.available_at, min(witnesses)) if witnesses else None


def _require_embargo(*, params: dict[str, Any], start_target: date) -> None:
    try:
        cutoff = date.fromisoformat(params["horizons"]["6m"]["training_cutoff"])
        sessions = get_calendar("XNYS", start=cutoff, end=start_target).sessions_in_range(
            cutoff, start_target
        )
        embargo = int(params["method"]["embargo_sessions"]["6m"])
    except (KeyError, TypeError, ValueError):
        raise IntervalShadowError("interval_shadow_embargo_violation") from None
    count = sum(session.date() > cutoff for session in sessions)
    if count < embargo:
        raise IntervalShadowError("interval_shadow_embargo_violation")


def _prospective_start(timestamp: datetime) -> date:
    try:
        return forward_scorecard.prospective_start_target(timestamp)
    except ValueError:
        raise IntervalShadowError("interval_shadow_clock_invalid") from None


def _declaration_attempts(job_name: str) -> dict[str, Any]:
    runs = JobRun.objects.filter(
        job_name=job_name, region=_REGION, target_date=_COORDINATION_DATE
    ).order_by("-attempt")
    latest = runs.filter(status=JobRun.Status.FAILED).first()
    failure = "unrecognized_failure"
    if latest is not None:
        match = re.fullmatch(r"IntervalShadowError: (.+)", latest.error)
        if match and _is_allowed_code(match.group(1)):
            failure = match.group(1)
    return {
        "failed": runs.filter(status=JobRun.Status.FAILED).count(),
        "latest_failure_code": failure if latest is not None else None,
    }


def _is_allowed_code(code: str) -> bool:
    static = {
        "interval_shadow_owner_unauthorized",
        "interval_shadow_source_study_absent",
        "interval_shadow_source_study_unverified",
        "interval_shadow_source_study_changed",
        "interval_shadow_owner_mismatch",
        "interval_shadow_params_invalid",
        "params_recompute_mismatch",
        "interval_shadow_revision_unverified",
        "interval_shadow_embargo_violation",
        "interval_shadow_clock_invalid",
        "interval_shadow_registry_ambiguous",
        "interval_shadow_publication_unverified",
        "interval_shadow_storage_failed",
        "interval_shadow_declaration_invalid",
        "interval_shadow_protocol_mismatch",
        "interval_shadow_scope_unverified",
        "interval_shadow_outcome_unverified",
    }
    if code in static:
        return True
    return bool(
        re.fullmatch(
            r"interval_shadow_params_withheld:"
            r"(no_scored_observations|log_scale_boundary_observation|"
            r"conformal_rank_infeasible|insufficient_effective_support)",
            code,
        )
    )


def _evidence_label(provider: str) -> str:
    return (
        "synthetic_demo_not_empirical"
        if settings.DEMO_MODE or provider == "synthetic_demo"
        else "observed_prospective_research_only"
    )


def _report_base(
    *,
    status: str,
    verification_code: str | None,
    attempts: dict[str, Any],
    evidence_label: str,
) -> dict[str, Any]:
    skeleton = {
        "state": "not_declared" if status == "not_declared" else "integrity_failed",
        "paired_evidence_count": 0,
    }
    return {
        "schema": REPORT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": SHADOW_PROTOCOL_SHA256,
        "status": status,
        "verification_code": verification_code,
        "declaration_attempts": attempts,
        "declaration": None,
        "evidence_label": evidence_label,
        "horizons": None
        if status == "not_declared"
        else {horizon: skeleton.copy() for horizon in ("6m", "12m", "3y", "5y")},
        "primary_result": None,
        "companion_result": None,
        "s4_gate": {
            "state": "not_eligible_synthetic" if "synthetic" in evidence_label else "blocked"
        },
        "report_read_at": timezone.now().isoformat(),
        "disclosures": [
            "Research-only descriptive evidence; native forecasts and probabilities are unchanged.",
            "No conditional-coverage, regime-robustness, or investment-skill claim.",
        ],
    }


def _integrity_report(
    *, code: str, attempts: dict[str, Any], evidence_label: str
) -> dict[str, Any]:
    report = _report_base(
        status="integrity_failed",
        verification_code=code,
        attempts=attempts,
        evidence_label=evidence_label,
    )
    if code == "params_recompute_mismatch":
        report["horizons"]["6m"] = {
            "state": "withheld_native_fallback",
            "reason": code,
            "offset": None,
            "paired_evidence_count": 0,
            "native_interval_unchanged": True,
        }
    return report


def _declaration_summary(asset: DataAsset, document: dict[str, Any]) -> dict[str, Any]:
    witness = _witness(
        asset,
        identity={
            "owner_id": document["owner_id"],
            "source_owner_id": document["source_owner_id"],
            "source_provider": document["source_provider"],
            "protocol_sha256": document["protocol_sha256"],
        },
        job_name=_job_name(
            {
                "owner_id": document["owner_id"],
                "source_owner_id": document["source_owner_id"],
                "source_provider": document["source_provider"],
                "protocol_sha256": document["protocol_sha256"],
            }
        ),
    )
    if witness is None:
        raise IntervalShadowError("interval_shadow_publication_unverified")
    return {
        "asset": asset_ref_for(asset).to_json(),
        "declared_at": document["declared_at"],
        "witnessed_at": witness.isoformat(),
        "prospective_start_target": _prospective_start(witness).isoformat(),
        "params_sha256": document["params_sha256"],
        "study": document["study"],
    }


def _build_report(
    *,
    user: ProductUser,
    store: AssetStore,
    provider: str,
    params: dict[str, Any],
    declaration: dict[str, Any],
    attempts: dict[str, Any],
) -> dict[str, Any]:
    scope = read_research_product_performance(user=user, store=store)
    if not scope.available:
        raise IntervalShadowError("interval_shadow_scope_unverified")
    run_ids = tuple(
        item.run.id for item in scope.runs if item.run.config_version == PRODUCT_VERSION
    )
    rows = list(
        iter_product_prediction_projection(
            run_ids,
            extra_fields=(
                "generated_at",
                "outcome__evaluation_date",
                "outcome__evaluated_at",
                "outcome__actual_return",
            ),
        )
    )
    witness = datetime.fromisoformat(declaration["witnessed_at"])
    start_target = date.fromisoformat(declaration["prospective_start_target"])
    denominators = {horizon: dict.fromkeys(_DENOMINATOR_KEYS, 0) for horizon in ("6m", "12m")}
    by_horizon: dict[str, list[dict[str, Any]]] = {"6m": [], "12m": []}
    for row in rows:
        if (
            row["evidence_role"] != "advisory"
            or row["method_version"] != FHS_METHOD_VERSION
            or row["horizon"] not in by_horizon
        ):
            continue
        horizon = cast(str, row["horizon"])
        counts = denominators[horizon]
        counts["raw_rows"] += 1
        native = _triplet(row)
        if native is None:
            counts["non_evaluable"] += 1
            continue
        if row["config_hash"] != PRODUCT_EFFECTIVE_CONFIG_HASH or row["price_provider"] != provider:
            counts["config_or_provider_mismatch"] += 1
            continue
        if not row["canonical"]:
            counts["not_canonical"] += 1
            if row["reportable"]:
                counts["excluded_reissue"] += 1
            continue
        if row["generated_at"] <= witness or row["target_date"] < start_target:
            counts["pre_declaration_excluded"] += 1
            continue
        counts["eligible"] += 1
        status = row["outcome__status"]
        record: dict[str, Any] = {
            "row": row,
            "target_date": row["target_date"],
            "evaluation_date": row.get("outcome__evaluation_date"),
            "outcome_status": status,
            "native": None,
            "candidate": None,
            "candidate_representable": False,
        }
        by_horizon[horizon].append(record)
        if status is None:
            counts["pending"] += 1
            continue
        if status == "unresolved":
            counts["unresolved"] += 1
            continue
        if status == "corporate_event":
            counts["corporate_event"] += 1
            continue
        if status != "matured":
            raise IntervalShadowError("interval_shadow_outcome_unverified")
        try:
            outcome = _authenticated_outcome(row, provider)
        except IntervalShadowError:
            raise
        except Exception:
            raise IntervalShadowError("interval_shadow_outcome_unverified") from None
        if (
            outcome.status != "matured"
            or outcome.evaluated_at > timezone.now()
            or outcome.actual_return is None
        ):
            raise IntervalShadowError("interval_shadow_outcome_unverified")
        counts["matured"] += 1
        offset_horizon = params["horizons"].get(horizon, {})
        if offset_horizon.get("state") != "usable":
            # Withheld horizons retain native eligibility denominators only.
            continue
        candidate = adjust_interval(
            native.lower,
            native.median,
            native.upper,
            # The S2 output is the sole source of this immutable parameter.
            Decimal(offset_horizon["offset"]),
        )
        native_metrics = score_projection_metrics(
            predicted_returns=native,
            actual_return=outcome.actual_return,
            config=load_price_product_config(),
        )
        candidate_metrics = score_projection_metrics(
            predicted_returns=(LedgerTriplet(*candidate) if candidate is not None else None),
            actual_return=outcome.actual_return,
            config=load_price_product_config(),
        )
        if candidate is None:
            counts["candidate_unrepresentable"] += 1
        else:
            lower_clamped, upper_clamped = _median_clamp_sides(
                native, LedgerTriplet(*candidate), Decimal(offset_horizon["offset"])
            )
            counts["clamped_lower"] += int(lower_clamped)
            counts["clamped_upper"] += int(upper_clamped)
            if candidate_metrics.interval_score is None:
                counts["candidate_score_unavailable"] += 1
        if native_metrics.interval_score is None:
            counts["native_score_unavailable"] += 1
        record.update(
            evaluation_date=outcome.evaluation_date,
            native=native_metrics,
            candidate=candidate_metrics,
            candidate_representable=(
                candidate is not None
                and native_metrics.interval_score is not None
                and candidate_metrics.interval_score is not None
            ),
        )

    horizons: dict[str, Any] = {
        "6m": _horizon_report(by_horizon["6m"], denominators["6m"], params["horizons"]["6m"]),
        "12m": _withheld_horizon(params["horizons"]["12m"], denominators["12m"]),
        "3y": _unavailable_horizon(),
        "5y": _unavailable_horizon(),
    }
    primary, companion = _primary_and_companion(by_horizon["6m"])
    report = _report_base(
        status="available",
        verification_code=None,
        attempts=attempts,
        evidence_label=_evidence_label(provider),
    )
    report.update(
        declaration=declaration,
        horizons=horizons,
        primary_result=primary,
        companion_result=companion,
        s4_gate={
            "state": (
                "not_eligible_synthetic"
                if _evidence_label(provider) == "synthetic_demo_not_empirical"
                else "eligible_for_independent_integrity_review"
                if primary["state"] == "met" and companion is not None and companion["met"]
                else "not_eligible"
            )
        },
    )
    return report


def _triplet(row: dict[str, Any]) -> LedgerTriplet | None:
    values = tuple(row[name] for name in ("bear_return", "base_return", "bull_return"))
    if all(value is None for value in values):
        return None
    if any(value is None or not value.is_finite() for value in values):
        raise IntervalShadowError("interval_shadow_outcome_unverified")
    return LedgerTriplet(cast(Any, values[0]), cast(Any, values[1]), cast(Any, values[2]))


def _horizon_report(
    records: list[dict[str, Any]], counts: dict[str, int], horizon_params: dict[str, Any]
) -> dict[str, Any]:
    if horizon_params.get("state") != "usable":
        return {
            "state": "withheld_native_fallback",
            "reason": horizon_params.get("reason"),
            "offset": None,
            "denominators": counts,
            "paired_evidence_count": 0,
            "descriptive_all_matured": _empty_descriptive(),
        }
    pairs = [item for item in records if _is_successfully_scored_pair(item)]
    return {
        "state": "usable",
        "offset": horizon_params["offset"],
        "denominators": counts,
        "paired_evidence_count": len(pairs),
        "descriptive_all_matured": _descriptive(pairs, horizon="6m"),
    }


def _withheld_horizon(params: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    return {
        "state": "withheld_native_fallback",
        "reason": params.get("reason"),
        "offset": None,
        "denominators": counts,
        "paired_evidence_count": 0,
        "descriptive_all_matured": _empty_descriptive(),
    }


def _unavailable_horizon() -> dict[str, Any]:
    return {
        "state": "horizon_unavailable_no_calibration_claim",
        "reason": "horizon_unavailable_no_calibration_claim",
        "offset": None,
        "denominators": None,
        "paired_evidence_count": 0,
        "descriptive_all_matured": _empty_descriptive(),
    }


def _empty_descriptive() -> dict[str, Any]:
    return {
        "paired_observation_count": 0,
        "target_cohort_count": 0,
        "overlap_aware_support": 0,
        "support_reason": "no_evaluable_forward_support; skill not established",
        "native_average": None,
        "calibrated_average": None,
        "mean_difference_candidate_minus_baseline": None,
    }


def _primary_and_companion(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    cohorts: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        cohorts[record["target_date"]].append(record)
    selected_summaries: list[dict[str, Any]] = []
    calendar = get_calendar(
        "XNYS",
        start=min(cohorts) - timedelta(days=7) if cohorts else date(2000, 1, 1),
        end=max(cohorts) + timedelta(days=6 * 366) if cohorts else date(2030, 1, 1),
    )
    for target, rows in cohorts.items():
        if len(rows) < 30:
            continue
        maturity = calendar.session_offset(calendar.date_to_session(target), 126).date()
        selected_summaries.append(
            {
                "prediction__target_date": target,
                "max_evaluation_date": maturity,
                "listing_count": len(rows),
            }
        )
    selected = _select_non_overlapping_summaries(selected_summaries)[:_PRIMARY_COHORTS]
    selected_dates = [cast(date, item["prediction__target_date"]) for item in selected]
    if len(selected) < _PRIMARY_COHORTS:
        return (
            {
                "state": "accruing",
                "selected_cohort_count": len(selected),
                "required_cohort_count": _PRIMARY_COHORTS,
                "cohorts": [],
                "sign_count": None,
                "sign_p_value": None,
                "mean_difference": None,
            },
            None,
        )
    chosen: list[list[dict[str, Any]]] = [cohorts[target] for target in selected_dates]
    cohort_results: list[dict[str, Any]] = []
    pending = False
    for target, rows in zip(selected_dates, chosen, strict=True):
        if any(item["row"]["outcome__status"] in (None, "unresolved") for item in rows):
            pending = True
        pairs = [item for item in rows if _is_successfully_scored_pair(item)]
        comparison = _compare(pairs, horizon="6m", cohort_date=target)
        cohort_results.append(
            {
                "target_date": target.isoformat(),
                "eligible_listings": len(rows),
                "paired_scored_listings": comparison.paired_observation_count,
                "mean_difference_candidate_minus_native": (
                    str(comparison.mean_difference_candidate_minus_baseline)
                    if comparison.mean_difference_candidate_minus_baseline is not None
                    else None
                ),
            }
        )
    if pending:
        return (
            {
                "state": "pending_maturity",
                "selected_cohort_count": len(selected),
                "required_cohort_count": _PRIMARY_COHORTS,
                "cohorts": cohort_results,
                "sign_count": None,
                "sign_p_value": None,
                "mean_difference": None,
            },
            None,
        )
    actual_overlap = _actual_maturity_overlap(selected_dates, chosen)
    if actual_overlap:
        return (
            {
                "state": "insufficient_support",
                "selected_cohort_count": len(selected),
                "required_cohort_count": _PRIMARY_COHORTS,
                "span_days": (max(selected_dates) - min(selected_dates)).days,
                "reason": "actual_maturity_windows_overlap_selected_cohorts",
                "overlapping_cohort_count": actual_overlap,
                "cohorts": [],
                "sign_count": None,
                "sign_p_value": None,
                "mean_difference": None,
            },
            None,
        )
    span_days = (max(selected_dates) - min(selected_dates)).days
    support_ok = (
        all(item["paired_scored_listings"] >= 30 for item in cohort_results) and span_days >= 1095
    )
    if not support_ok:
        return (
            {
                "state": "insufficient_support",
                "selected_cohort_count": len(selected),
                "required_cohort_count": _PRIMARY_COHORTS,
                "span_days": span_days,
                "cohorts": cohort_results,
                "sign_count": None,
                "sign_p_value": None,
                "mean_difference": None,
            },
            None,
        )
    differences = [
        Decimal(item["mean_difference_candidate_minus_native"])
        for item in cohort_results
        if item["mean_difference_candidate_minus_native"] is not None
    ]
    k = sum(value < 0 for value in differences)
    tail = sum(comb(_PRIMARY_COHORTS, i) for i in range(k, _PRIMARY_COHORTS + 1))
    p_value = Fraction(tail, 2**_PRIMARY_COHORTS)
    mean_difference = sum(differences, Decimal(0)) / Decimal(_PRIMARY_COHORTS)
    primary_met = k >= 7 and mean_difference < 0
    primary = {
        "state": "met" if primary_met else "not_met",
        "selected_cohort_count": _PRIMARY_COHORTS,
        "required_cohort_count": _PRIMARY_COHORTS,
        "span_days": span_days,
        "cohorts": cohort_results,
        "sign_count": k,
        "sign_p_value": f"{p_value.numerator}/{p_value.denominator}",
        "mean_difference": str(mean_difference),
        "independence_assumption": (
            "Exact sign calculation assumes independent cohorts; non-overlapping windows "
            "alone do not establish independence."
        ),
    }
    companion = _coverage_companion(chosen)
    companion["met"] = (
        companion["calibrated"]["abs_deviation_from_0.6"]
        <= companion["native"]["abs_deviation_from_0.6"]
    )
    return primary, companion


def _compare(
    records: list[dict[str, Any]], *, horizon: str, cohort_date: date | None = None
) -> Any:
    observations: list[ReplayMetricObservation] = []
    for record in records:
        row = record["row"]
        common = {
            "listing_id": str(row["listing_id"]),
            "anchor_date": row["target_date"],
            # Keep actual evaluation_date on the records for outcome integrity;
            # cohort aggregation is keyed to the prediction-side cohort.
            "target_date": cohort_date or record["evaluation_date"],
            "horizon": horizon,
            "partition": "final_holdout",
        }
        observations.extend(
            (
                ReplayMetricObservation(
                    **common, model_name="calibrated", metrics=record["candidate"]
                ),
                ReplayMetricObservation(**common, model_name="native", metrics=record["native"]),
            )
        )
    return compare_aligned_models(
        observations,
        candidate_model="calibrated",
        baseline_model="native",
        partition="final_holdout",
        horizon=horizon,
        metric_name="interval_score",
    )


def _coverage_companion(chosen: list[list[dict[str, Any]]]) -> dict[str, Any]:
    native: dict[str, list[bool]] = {}
    calibrated: dict[str, list[bool]] = {}
    for cohort_index, records in enumerate(chosen):
        native[str(cohort_index)] = [
            cast(bool, item["native"].interval_included)
            for item in records
            if _is_successfully_scored_pair(item) and item["native"].interval_included is not None
        ]
        calibrated[str(cohort_index)] = [
            cast(bool, item["candidate"].interval_included)
            for item in records
            if _is_successfully_scored_pair(item)
            and item["candidate"].interval_included is not None
        ]
    native_summary = summarize_cluster_coverage(native)
    calibrated_summary = summarize_cluster_coverage(calibrated)
    return {
        "native": {
            **native_summary,
            "cohorts_below_0.6": sum(
                Decimal(value) < Decimal("0.6") for value in native_summary["cluster_means"]
            ),
        },
        "calibrated": {
            **calibrated_summary,
            "cohorts_below_0.6": sum(
                Decimal(value) < Decimal("0.6") for value in calibrated_summary["cluster_means"]
            ),
        },
    }


def _is_successfully_scored_pair(record: dict[str, Any]) -> bool:
    native = record.get("native")
    candidate = record.get("candidate")
    return bool(
        record.get("candidate_representable")
        and native is not None
        and candidate is not None
        and native.interval_score is not None
        and candidate.interval_score is not None
    )


def _actual_maturity_overlap(
    selected_dates: list[date], cohorts: list[list[dict[str, Any]]]
) -> int:
    """Count selected cohort windows whose authenticated maturity crosses the next anchor."""
    overlaps = 0
    for next_target, rows in zip(selected_dates[1:], cohorts[:-1], strict=True):
        evaluation_dates: list[date] = []
        for item in rows:
            evaluation_date = item.get("evaluation_date")
            if item["outcome_status"] == "matured" and isinstance(evaluation_date, date):
                evaluation_dates.append(evaluation_date)
        if evaluation_dates and max(evaluation_dates) >= next_target:
            overlaps += 1
    return overlaps


def _median_clamp_sides(
    native: LedgerTriplet, candidate: LedgerTriplet, offset: Decimal
) -> tuple[bool, bool]:
    """Identify bounds strictly forced to the median by interval contraction."""
    if offset >= 0 or (candidate.lower != native.median and candidate.upper != native.median):
        return False, False
    with localcontext() as context:
        context.prec = 28
        context.rounding = ROUND_HALF_EVEN
        context.Emin = -999_999
        context.Emax = 999_999
        raw_lower = ((Decimal(1) + native.lower).ln() - offset).exp() - Decimal(1)
        raw_upper = ((Decimal(1) + native.upper).ln() + offset).exp() - Decimal(1)
    return (
        raw_lower > native.median and candidate.lower == native.median,
        raw_upper < native.median and candidate.upper == native.median,
    )


def _descriptive(records: list[dict[str, Any]], *, horizon: str) -> dict[str, Any]:
    prediction_cohorts: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        prediction_cohorts[item["target_date"]].append(item)
    comparisons = [
        _compare(
            [item for item in cohort if _is_successfully_scored_pair(item)],
            horizon=horizon,
            cohort_date=target,
        )
        for target, cohort in sorted(prediction_cohorts.items())
    ]
    comparisons = [item for item in comparisons if item.paired_observation_count > 0]
    paired_count = sum(item.paired_observation_count for item in comparisons)
    candidate_average = _mean_cohort_values([item.candidate_average for item in comparisons])
    baseline_average = _mean_cohort_values([item.baseline_average for item in comparisons])
    mean_difference = (
        candidate_average - baseline_average
        if candidate_average is not None and baseline_average is not None
        else None
    )
    support_rows = [
        {
            "listing_id": str(item["row"]["listing_id"]),
            "target_date": item["target_date"],
            "outcome__evaluation_date": item["evaluation_date"],
        }
        for item in records
    ]
    support_reason, support_count = forward_scorecard._support_result(support_rows, horizon)
    has_support = support_reason is None
    return {
        "paired_observation_count": paired_count,
        "target_cohort_count": len(comparisons),
        "overlap_aware_support": support_count,
        "support_reason": support_reason,
        "native_average": (str(baseline_average) if has_support else None),
        "calibrated_average": (str(candidate_average) if has_support else None),
        "mean_difference_candidate_minus_baseline": (str(mean_difference) if has_support else None),
    }


def _mean_cohort_values(values: list[Decimal | None]) -> Decimal | None:
    available = [value for value in values if value is not None]
    if not available:
        return None
    with localcontext() as context:
        context.prec = 28
        context.rounding = ROUND_HALF_EVEN
        context.Emin = -999_999
        context.Emax = 999_999
        return sum(available, Decimal(0)) / Decimal(len(available))

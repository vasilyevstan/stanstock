"""Owner-bound forward declaration and canonical, authenticated outcome scorecard.

No prediction is issued or evaluated here. The declaration is durable before
any outcome/loss read; a report is a read, not a new observation or report asset.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import transaction
from django.utils import timezone
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.jobs import target_job_lock
from stanstock.core.revision import _git
from stanstock.core.verification_types import AssetRef
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    read_checksummed_bytes,
    register_asset,
    resolve_asset_ref,
)
from stanstock.data.models import DataAsset
from stanstock.research.decision_support_eligibility import (
    FORECAST_CONFIG_HASH,
    SOURCE_CONTRACT,
    canonical_json,
    strict_json,
)
from stanstock.research.decision_support_eligibility import (
    MOMENTUM_METHOD_VERSION as V2_MOMENTUM,
)
from stanstock.research.decision_support_eligibility import (
    PRODUCT_EFFECTIVE_CONFIG_HASH as V2_CONFIG_HASH,
)
from stanstock.research.decision_support_eligibility import (
    PRODUCT_VERSION as V2_PRODUCT_VERSION,
)
from stanstock.research.models import Prediction
from stanstock.research.outcome_refresh_validation import verify_prediction_outcome
from stanstock.research.outcomes import HORIZON_SESSION_COUNTS
from stanstock.research.price_product import LedgerTriplet
from stanstock.research.price_product_config import (
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_VERSION,
    load_price_product_config,
)
from stanstock.research.price_product_replay import (
    ReplayMetricObservation,
    aggregate_projection_metrics,
    compare_aligned_models,
    score_projection_metrics,
)
from stanstock.research.product_reader import (
    ProductUser,
    _authorized_intakes,
    _require_current_display_authorization,
    read_research_product_performance,
)
from stanstock.research.reporting import (
    _ADVISORY_FLOORS,
    _select_non_overlapping_summaries,
    iter_product_prediction_projection,
)

FORWARD_PROTOCOL_VERSION = "forward-forecast-scorecard-v1"
FORWARD_PROTOCOL_KIND = "forward_scorecard_protocol"
PROTOCOL_EPOCH = date(2026, 9, 11)
# Immutable code declaration, not a protocol/report registry or tunable config.
FORWARD_PROTOCOL = {
    "version": FORWARD_PROTOCOL_VERSION,
    "product_version": PRODUCT_VERSION,
    "effective_config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
    "scope": "all_owner_authorized_product_observations_including_future_issuances",
    "canonical": "earliest_reportable_before_status",
    "identities": ["method", "config", "provider", "grade", "role", "horizon"],
    "horizon_sessions": {"6m": 126, "12m": 252, "3y": 756, "5y": 1260},
    "forecast": "us-price-fhs-v1",
    "paired_comparator": "recorded_zero_drift_ledger_returns",
    "losses": [
        "median_absolute_error",
        "pinball_p20",
        "pinball_p50",
        "pinball_p80",
        "interval_score",
        "interval_inclusion",
        "interval_width",
        "signed_bias",
    ],
    "aggregation": "within_target_cohort_then_equal_target_cohorts",
    "support_floors": {key: list(value) for key, value in _ADVISORY_FLOORS.items()},
    "decision": "BUY_and_AVOID_separate_HOLD_success_null",
    "return_basis": "split_adjusted_price_return_no_dividends",
    "post_maturity_declaration": "excluded_from_forward_metrics",
    "nonclaims": "rows_and_monte_carlo_paths_are_not_independent_skill",
}
# Literal pin is checked before publication and reading.
FORWARD_PROTOCOL_SHA256 = "6fc2608de73d2de46c02cb2ee1308741e8882a0f5cb01f56cf9e230afa323524"
FORWARD_PROTOCOL_V2_VERSION = "forward-forecast-scorecard-v2"
FORWARD_PROTOCOL_V2 = {
    **json.loads(json.dumps(FORWARD_PROTOCOL)),
    "version": FORWARD_PROTOCOL_V2_VERSION,
    "product_version": V2_PRODUCT_VERSION,
    "effective_config_hash": V2_CONFIG_HASH,
    "forecast_seed_config_hash": FORECAST_CONFIG_HASH,
    "eligibility_source_contract": SOURCE_CONTRACT,
    "decision_method": V2_MOMENTUM,
    "prospective_start": "first_XNYS_target_close_strictly_after_actual_declaration",
}
FORWARD_PROTOCOL_V2_SHA256 = "d49e0b52cf2af6d68c7eccf1c8d73234b9488421c0373c2867436995a05048df"


def _protocol(product_version: str) -> tuple[str, dict[str, Any], str]:
    if product_version == PRODUCT_VERSION:
        return FORWARD_PROTOCOL_VERSION, FORWARD_PROTOCOL, FORWARD_PROTOCOL_SHA256
    if product_version == V2_PRODUCT_VERSION:
        return FORWARD_PROTOCOL_V2_VERSION, FORWARD_PROTOCOL_V2, FORWARD_PROTOCOL_V2_SHA256
    raise ValueError("Unsupported forward protocol product version")


def prospective_start_target(declared_at: datetime) -> date:
    if declared_at.tzinfo is None:
        raise ValueError("Forward declaration time must be timezone-aware")
    first_day = declared_at.date()
    calendar = get_calendar(
        "XNYS", start=first_day - timedelta(days=7), end=first_day + timedelta(days=21)
    )
    for session in calendar.sessions_in_range(first_day, first_day + timedelta(days=14)):
        if calendar.session_close(session).to_pydatetime() > declared_at:
            return cast(date, session.date())
    raise ValueError("Forward declaration has no prospective target in the bounded calendar")


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True, slots=True)
class ForwardDeclaration:
    asset: DataAsset
    declared_at: datetime
    document: dict[str, Any]


def _authority(
    user: ProductUser,
    store: AssetStore,
    product_version: str = PRODUCT_VERSION,
    *,
    allow_disabled: bool = False,
) -> tuple[str, str, str]:
    if (
        (not settings.RESEARCH_PRODUCT_ENABLED and not allow_disabled)
        or not user.is_authenticated
        or not get_user_model().objects.filter(pk=str(user.pk), is_active=True).exists()
    ):
        raise ValueError("An active research owner is required")
    if product_version == V2_PRODUCT_VERSION and settings.DEMO_MODE:
        raise ValueError("research-product-v2 has no synthetic demo protocol")
    source_owner, provider, _intakes = _authorized_intakes(
        user=user, product_version=product_version
    )
    _require_current_display_authorization(
        {"owner_id": source_owner, "source_provider": provider}, user=user
    )
    _version, protocol, pin = _protocol(product_version)
    if hashlib.sha256(_bytes(protocol)).hexdigest() != pin:
        raise ValueError("Forward protocol code declaration differs from its pin")
    return str(user.pk), source_owner, provider


def _protocol_subject(owner_id: str, product_version: str = PRODUCT_VERSION) -> str:
    version, _document, _pin = _protocol(product_version)
    return f"{version}:{owner_id}"


def _read_declaration(
    asset: DataAsset,
    *,
    user: ProductUser,
    store: AssetStore,
    product_version: str = PRODUCT_VERSION,
    allow_disabled: bool = False,
) -> ForwardDeclaration:
    owner_id, source_owner, provider = _authority(
        user, store, product_version, allow_disabled=allow_disabled
    )
    version, protocol, pin = _protocol(product_version)
    raw = read_checksummed_bytes(store, asset)
    document = strict_json(raw) if product_version == V2_PRODUCT_VERSION else json.loads(raw)
    keys = {
        "protocol",
        "protocol_sha256",
        "owner_id",
        "source_owner_id",
        "source_provider",
        "source_intakes",
        "declared_at",
        "execution_revision",
    }
    if product_version == V2_PRODUCT_VERSION:
        keys.add("prospective_start_target")
    if not isinstance(document, dict) or set(document) != keys:
        raise ValueError("Forward declaration has an invalid shape")
    declared_at = datetime.fromisoformat(document["declared_at"])
    expected_metadata = {
        "owner_id": owner_id,
        "protocol_version": version,
        "protocol_sha256": pin,
        "declared_at": document["declared_at"],
    }
    if (
        asset.provider != "stanstock"
        or asset.kind != FORWARD_PROTOCOL_KIND
        or asset.subject != _protocol_subject(owner_id, product_version)
        or document["owner_id"] != owner_id
        or document["source_owner_id"] != source_owner
        or document["source_provider"] != provider
        or (
            canonical_json(document["protocol"]) != canonical_json(protocol)
            if product_version == V2_PRODUCT_VERSION
            else document["protocol"] != protocol
        )
        or document["protocol_sha256"] != pin
        or asset.metadata != expected_metadata
        or declared_at.tzinfo is None
        or declared_at > timezone.now()
        or asset.available_at != declared_at
        or asset.retrieved_at != declared_at
        or not isinstance(document["execution_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", document["execution_revision"]) is None
        or not isinstance(document["source_intakes"], list)
    ):
        raise ValueError("Forward declaration identity is invalid")
    if (
        product_version == V2_PRODUCT_VERSION
        and document["prospective_start_target"]
        != prospective_start_target(declared_at).isoformat()
    ):
        raise ValueError(
            "V2 forward declaration prospective start differs from the independent calendar"
        )
    for raw in document["source_intakes"]:
        intake = resolve_asset_ref(AssetRef.from_json(raw), cutoff=declared_at)
        payload = json.loads(read_checksummed_bytes(store, intake))
        if (
            intake.provider != "stanstock"
            or intake.kind != "research_product_intake"
            or payload.get("owner_id") != source_owner
            or payload.get("source_provider") != provider
            or payload.get("product_version") != product_version
        ):
            raise ValueError("Forward declaration intake authority is invalid")
    return ForwardDeclaration(asset, declared_at, document)


def read_forward_declaration(
    *,
    user: ProductUser,
    store: AssetStore | None = None,
    product_version: str = PRODUCT_VERSION,
) -> ForwardDeclaration:
    store = store or AssetStore()
    owner_id, _source_owner, _provider = _authority(user, store, product_version)
    assets = list(
        DataAsset.objects.filter(
            provider="stanstock",
            kind=FORWARD_PROTOCOL_KIND,
            subject=_protocol_subject(owner_id, product_version),
        )
    )
    if len(assets) != 1:
        raise ValueError("One durable forward protocol declaration is required")
    return _read_declaration(assets[0], user=user, store=store, product_version=product_version)


def declare_forward_protocol(
    *,
    user: ProductUser,
    store: AssetStore | None = None,
    product_version: str = PRODUCT_VERSION,
) -> ForwardDeclaration:
    """Explicit publication only. The owner/version lock spans durable commit."""
    store = store or AssetStore()
    owner_id, source_owner, provider = _authority(user, store, product_version)
    version, protocol, pin = _protocol(product_version)
    with (
        target_job_lock(
            job_name=f"{version}:{owner_id}",
            region="us",
            target_date=PROTOCOL_EPOCH,
        ),
        transaction.atomic(durable=True),
    ):
        existing = list(
            DataAsset.objects.filter(
                provider="stanstock",
                kind=FORWARD_PROTOCOL_KIND,
                subject=_protocol_subject(owner_id, product_version),
            )
        )
        if existing:
            if len(existing) != 1:
                raise ValueError("Forward declaration registry is ambiguous")
            return _read_declaration(
                existing[0], user=user, store=store, product_version=product_version
            )
        _owner, _provider, intakes = _authorized_intakes(user=user, product_version=product_version)
        # No predictions/outcomes/losses are read by declaration.
        declared_at = timezone.now()
        revision = _git(settings.BASE_DIR, "rev-parse", "HEAD").stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("Forward declaration execution revision is unavailable")
        document = {
            "protocol": protocol,
            "protocol_sha256": pin,
            "owner_id": owner_id,
            "source_owner_id": source_owner,
            "source_provider": provider,
            "source_intakes": [
                asset_ref_for(asset).to_json()
                for asset in sorted(intakes.values(), key=lambda item: str(item.id))
                if asset.available_at <= declared_at and asset.retrieved_at <= declared_at
            ],
            "declared_at": declared_at.isoformat(),
            "execution_revision": revision,
        }
        if product_version == V2_PRODUCT_VERSION:
            document["prospective_start_target"] = prospective_start_target(declared_at).isoformat()
        payload = _bytes(document)
        stored = store.write_bytes(
            f"research/forward/{owner_id}/{hashlib.sha256(payload).hexdigest()}.json", payload
        )
        asset = register_asset(
            provider="stanstock",
            kind=FORWARD_PROTOCOL_KIND,
            subject=_protocol_subject(owner_id, product_version),
            stored=stored,
            retrieved_at=declared_at,
            available_at=declared_at,
            metadata={
                "owner_id": owner_id,
                "protocol_version": version,
                "protocol_sha256": pin,
                "declared_at": declared_at.isoformat(),
            },
        )
        return _read_declaration(asset, user=user, store=store, product_version=product_version)


def require_v2_forward_declaration(
    *,
    owner_id: str,
    target_date: date,
    before: datetime | str,
    store: AssetStore,
    acquisition_job_id: str | None = None,
    expected_asset: DataAsset | None = None,
) -> ForwardDeclaration:
    """Read only: never implicitly declare, reset, or backdate a protocol."""
    user = get_user_model().objects.filter(pk=owner_id, is_active=True).first()
    if user is None:
        raise ValueError("V2 forward declaration owner is inactive")
    assets = list(
        DataAsset.objects.filter(
            provider="stanstock",
            kind=FORWARD_PROTOCOL_KIND,
            subject=_protocol_subject(owner_id, V2_PRODUCT_VERSION),
        )
    )
    if len(assets) != 1:
        raise ValueError("V2 requires its single durable prospective declaration")
    declaration = _read_declaration(
        assets[0],
        user=user,
        store=store,
        product_version=V2_PRODUCT_VERSION,
        allow_disabled=True,
    )
    boundary = datetime.fromisoformat(before) if isinstance(before, str) else before
    if acquisition_job_id is not None:
        from stanstock.core.models import JobRun

        job = JobRun.objects.get(pk=acquisition_job_id)
        boundary = min(boundary, job.started_at)
    if (
        boundary.tzinfo is None
        or declaration.declared_at >= boundary
        or target_date < prospective_start_target(declaration.declared_at)
        or (expected_asset is not None and expected_asset.pk != declaration.asset.pk)
    ):
        raise ValueError(
            "V2 acquisition/issuance must follow its independently declared prospective start"
        )
    return declaration


def read_forward_scorecard(
    *,
    user: ProductUser,
    store: AssetStore | None = None,
    product_version: str = PRODUCT_VERSION,
) -> dict[str, Any]:
    """Fail closed: declaration, full physical product proof, then outcome replay."""
    store = store or AssetStore()
    declaration = read_forward_declaration(user=user, store=store, product_version=product_version)
    version, _protocol_document, pin = _protocol(product_version)
    scope = read_research_product_performance(user=user, store=store)
    if not scope.available:
        raise ValueError("Forward scorecard source scope could not be verified")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    fields = (
        "generated_at",
        "code_revision",
        "recommendation",
        "outcome__evaluation_date",
        "outcome__evaluated_at",
        "outcome__actual_return",
        "outcome__success",
        "outcome__signed_error",
        "calculation__forecast__projections",
    )
    for row in iter_product_prediction_projection(
        tuple(item.run.id for item in scope.runs if item.run.config_version == product_version),
        extra_fields=fields,
    ):
        key = tuple(
            row[name]
            for name in (
                "method_version",
                "config_hash",
                "price_provider",
                "evidence_grade",
                "evidence_role",
                "horizon",
            )
        )
        groups[key].append(row)
    results = [
        _score_group(key, rows, declaration=declaration) for key, rows in sorted(groups.items())
    ]
    matured = sum(group["denominators"]["matured"] for group in results)
    return {
        "status": "available",
        "protocol_version": version,
        "protocol_sha256": pin,
        "declaration": asset_ref_for(declaration.asset).to_json(),
        "declared_at": declaration.declared_at.isoformat(),
        "conclusion": "unmatured; skill not established"
        if matured == 0
        else ("descriptive forward evidence; skill not established"),
        "groups": results,
        "admission_denominators": dict(
            Counter(
                admission.status
                for item in scope.runs
                if item.run.config_version == product_version
                for admission in item.admissions
            )
        ),
    }


def _score_group(
    key: tuple[str, ...], rows: list[dict[str, Any]], *, declaration: ForwardDeclaration
) -> dict[str, Any]:
    method, config_hash, provider, grade, role, horizon = key
    config = load_price_product_config()
    counts: Counter[str] = Counter(raw_rows=len(rows))
    timing: Counter[str] = Counter()
    observations: list[ReplayMetricObservation] = []
    eligible_rows: list[dict[str, Any]] = []
    decision_support: dict[str, list[ReplayMetricObservation]] = {"BUY": [], "AVOID": []}
    biases: dict[date, list[Decimal]] = defaultdict(list)
    decision: dict[str, dict[date, list[bool]]] = {
        "BUY": defaultdict(list),
        "AVOID": defaultdict(list),
    }
    calendar = get_calendar(
        "XNYS",
        start=min(row["target_date"] for row in rows) - timedelta(days=7),
        end=max(row["target_date"] for row in rows) + timedelta(days=6 * 366),
    )
    for row in rows:
        if role == "advisory" and all(
            row[name] is None for name in ("bear_return", "base_return", "bull_return")
        ):
            counts["non_evaluable"] += 1
            continue
        if not row["reportable"]:
            counts["not_eligible_for_observed"] += 1
        if not row["canonical"]:
            if row["reportable"]:
                counts["excluded_reissue"] += 1
            continue
        counts["canonical_rows"] += 1
        maturity = calendar.session_offset(
            calendar.date_to_session(row["target_date"]), HORIZON_SESSION_COUNTS[horizon]
        )
        deadline = calendar.session_close(maturity).to_pydatetime()
        label = (
            "protocol declared before issuance"
            # Equal recorded clocks do not prove a pre-issuance declaration.
            if declaration.declared_at < row["generated_at"]
            else "protocol declared after issuance, before maturity"
            if declaration.declared_at < deadline
            else "protocol declared after maturity"
        )
        timing[label] += 1
        status = row["outcome__status"]
        counts[status or "pending"] += 1
        if status != "matured":
            continue
        # Authenticate native 126/252/756/1260-session outcome semantics,
        # complete identity and physical maturity-source bytes before losses.
        prediction = Prediction.objects.select_related(
            "analysis__run__universe_snapshot", "listing", "outcome"
        ).get(pk=row["id"])
        outcome = prediction.outcome
        if outcome.evaluated_at > timezone.now():
            raise ValueError("Forward outcome evaluation is in the future")
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=provider,
            benchmark_subject="SPY",
            evaluation_time=outcome.evaluated_at,
            parent_target_date=outcome.evaluation_date,
        )
        if declaration.declared_at >= deadline:
            counts["post_maturity_declaration_excluded"] += 1
            continue
        if role == "decision":
            action = str(row["recommendation"]).upper()
            if action in decision and outcome.success is not None:
                decision[action][outcome.evaluation_date].append(outcome.success)
                eligible_rows.append(row)
                decision_support[action].append(
                    ReplayMetricObservation(
                        listing_id=str(row["listing_id"]),
                        anchor_date=row["target_date"],
                        target_date=outcome.evaluation_date,
                        horizon=horizon,
                        partition="final_holdout",
                        model_name="decision",
                        metrics=score_projection_metrics(
                            predicted_returns=None,
                            actual_return=outcome.actual_return,
                            config=config,
                        ),
                    )
                )
            continue
        triplet = LedgerTriplet(
            *(row[name] for name in ("bear_return", "base_return", "bull_return"))
        )
        metrics = score_projection_metrics(
            predicted_returns=triplet, actual_return=outcome.actual_return, config=config
        )
        if metrics.unavailable_reason is None:
            eligible_rows.append(row)
        observations.append(
            ReplayMetricObservation(
                listing_id=str(row["listing_id"]),
                anchor_date=row["target_date"],
                target_date=outcome.evaluation_date,
                horizon=horizon,
                partition="final_holdout",
                model_name="fhs",
                metrics=metrics,
            )
        )
        if metrics.actual_return is not None and metrics.predicted_returns is not None:
            biases[outcome.evaluation_date].append(
                metrics.actual_return - metrics.predicted_returns.median
            )
        projections = row["calculation__forecast__projections"]
        raw = (
            next(
                (
                    item.get("zero_drift_ledger_returns")
                    for item in projections
                    if item.get("horizon") == horizon
                ),
                None,
            )
            if isinstance(projections, list)
            else None
        )
        comparator = None
        if isinstance(raw, dict) and set(raw) == {"lower", "median", "upper"}:
            comparator = LedgerTriplet(
                *(Decimal(raw[name]) for name in ("lower", "median", "upper"))
            )
        observations.append(
            ReplayMetricObservation(
                listing_id=str(row["listing_id"]),
                anchor_date=row["target_date"],
                target_date=outcome.evaluation_date,
                horizon=horizon,
                partition="final_holdout",
                model_name="recorded_zero_drift",
                metrics=score_projection_metrics(
                    predicted_returns=comparator, actual_return=outcome.actual_return, config=config
                ),
            )
        )
    summaries = aggregate_projection_metrics(
        observations,
        expected_partitions=("final_holdout",),
        expected_horizons=(horizon,),
        expected_models=("fhs", "recorded_zero_drift"),
    )
    native = next(item for item in summaries if item.model_name == "fhs")
    if role == "decision":
        native = aggregate_projection_metrics(
            [item for group in decision_support.values() for item in group],
            expected_partitions=("final_holdout",),
            expected_horizons=(horizon,),
            expected_models=("decision",),
        )[0]
    support_reason, effective_support = _support_result(eligible_rows, horizon)
    if counts["matured"] == 0:
        support_reason = "unmatured; skill not established"
    metrics_payload: dict[str, Decimal | None] = {
        str(name): value for name, value in native.averages.items()
    }
    metrics_payload["signed_bias"] = _cohort_mean(biases)
    if support_reason is not None:
        metrics_payload = dict.fromkeys(metrics_payload)
    paired = []
    available = {
        (item.model_name, item.listing_id, item.anchor_date, item.target_date)
        for item in observations
        if item.metrics.unavailable_reason is None
    }
    aligned = [
        item
        for item in observations
        if item.model_name == "fhs"
        and ("fhs", item.listing_id, item.anchor_date, item.target_date) in available
        and ("recorded_zero_drift", item.listing_id, item.anchor_date, item.target_date)
        in available
    ]
    paired_keys = {(item.listing_id, item.anchor_date) for item in aligned}
    paired_reason, effective_paired_support = _support_result(
        [
            row
            for row in eligible_rows
            if (str(row["listing_id"]), row["target_date"]) in paired_keys
        ],
        horizon,
    )
    for metric in native.averages:
        comparison = asdict(
            compare_aligned_models(
                observations,
                candidate_model="fhs",
                baseline_model="recorded_zero_drift",
                partition="final_holdout",
                horizon=horizon,
                metric_name=metric,
            )
        )
        reason = comparison["unavailable_reason"] or paired_reason
        if reason is not None:
            for name in (
                "candidate_average",
                "baseline_average",
                "mean_difference_candidate_minus_baseline",
            ):
                comparison[name] = None
            comparison["unavailable_reason"] = reason
        comparison["overlap_aware_support"] = effective_paired_support
        paired.append(comparison)
    return {
        "method": method,
        "config_hash": config_hash,
        "provider": provider,
        "grade": grade,
        "role": role,
        "horizon": horizon,
        "denominators": {
            name: counts[name]
            for name in (
                "raw_rows",
                "not_eligible_for_observed",
                "excluded_reissue",
                "canonical_rows",
                "pending",
                "matured",
                "unresolved",
                "corporate_event",
                "non_evaluable",
                "post_maturity_declaration_excluded",
            )
        },
        "protocol_timing": dict(timing),
        "rows": len(rows),
        "distinct_listings": len({row["listing_id"] for row in rows}),
        "target_cohorts": len({row["target_date"] for row in rows}),
        "calendar_span": [
            min(row["target_date"] for row in rows).isoformat(),
            max(row["target_date"] for row in rows).isoformat(),
        ],
        "overlap_aware_support": effective_support,
        "eligible_rows": len(eligible_rows),
        "eligible_distinct_listings": native.distinct_listing_count,
        "eligible_target_cohorts": native.target_cohort_count,
        "support_reason": support_reason,
        "metrics": metrics_payload if role == "advisory" else None,
        "decision_success": {
            action: {
                "count": sum(map(len, cohorts.values())),
                **_decision_summary(action, cohorts, eligible_rows, horizon),
            }
            for action, cohorts in decision.items()
        }
        if role == "decision"
        else None,
        "hold_success": None,
        "paired_zero_drift": paired if role == "advisory" else None,
        "execution_revisions": sorted({row["code_revision"] for row in rows}),
        "replay_partition_adapter": (
            "final_holdout is an internal loss-helper adapter, not a retrospective partition claim"
        ),
    }


def _cohort_mean(cohorts: Any) -> Decimal | None:
    if not cohorts:
        return None
    means = [
        sum((Decimal(value) for value in values), Decimal(0)) / len(values)
        for values in cohorts.values()
        if values
    ]
    return sum(means, Decimal(0)) / len(means) if means else None


def _support_result(
    supported: list[dict[str, Any]],
    horizon: str,
) -> tuple[str | None, int]:
    if not supported:
        return "no_evaluable_forward_support; skill not established", 0
    floor = _ADVISORY_FLOORS.get(horizon)
    cohorts: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in supported:
        cohorts[row["target_date"]].append(row)
    selected = _select_non_overlapping_summaries(
        [
            {
                "prediction__target_date": target,
                "max_evaluation_date": max(row["outcome__evaluation_date"] for row in rows),
                "listing_count": len({row["listing_id"] for row in rows}),
            }
            for target, rows in cohorts.items()
        ]
    )
    targets = [cast(date, row["prediction__target_date"]) for row in selected]
    span = (max(targets) - min(targets)).days
    if floor is None or (
        len(selected) < floor[0]
        or any(cast(int, row["listing_count"]) < floor[1] for row in selected)
        or span < floor[2]
    ):
        return "insufficient_overlap_aware_support; skill not established", len(selected)
    return None, len(selected)


def _decision_summary(
    action: str,
    cohorts: dict[date, list[bool]],
    eligible_rows: list[dict[str, Any]],
    horizon: str,
) -> dict[str, Any]:
    reason, effective_support = _support_result(
        [row for row in eligible_rows if str(row["recommendation"]).upper() == action],
        horizon,
    )
    return {
        "success_rate": None if reason else _cohort_mean(cohorts),
        "support_reason": reason,
        "overlap_aware_support": effective_support,
    }

"""Native scheduled orchestration for the prospective research product.

The LaunchAgent still invokes the single ``scheduled_refresh`` command.  This
module supplies the feature-flagged profile behind that command without
changing the archived scheduler.  A successful child status is never treated
as output proof: the parent re-reads the captured intake, membership, exact
five-row output, downstream children, and physical asset closure before it
may succeed.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun
from stanstock.core.revision import clean_git_revision
from stanstock.core.verification_types import AssetRef, RefreshVerificationError
from stanstock.data.asof import raw_price_asset_for
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    open_asset_store,
    resolve_asset_ref,
)
from stanstock.data.live_us import load_us_universe_config, resolve_us_target_date
from stanstock.data.management.config_loader import default_us_universe_config_path
from stanstock.data.models import DataAsset, ProviderRecord, UniverseMembership, UniverseSnapshot
from stanstock.data.provider_policy import (
    BASIC_PLAN,
    TWELVE_DATA_PROVIDER,
    normalized_provider_plan,
    validate_provider_usage,
)
from stanstock.data.providers.exceptions import ProviderError
from stanstock.data.research_product import (
    PRODUCT_MEMBERSHIP_KIND,
    load_product_intake,
    product_intake_payload,
    product_membership_payload,
)
from stanstock.data.research_product_jobs import (
    DAILY_RESEARCH_JOB,
    FREQUENCY_RESEARCH_JOB,
    RESEARCH_INTAKE_JOB,
    SCHEDULED_RESEARCH_JOB,
    execute_daily_research_job,
    product_job_name,
)
from stanstock.portfolio.jobs import (
    SCHEDULED_JOB_NAME as PORTFOLIO_JOB_NAME,
)
from stanstock.portfolio.jobs import (
    execute_portfolio_snapshot_job,
)
from stanstock.research.jobs import (
    JOB_NAME as EVALUATION_JOB_NAME,
)
from stanstock.research.jobs import (
    execute_prediction_evaluation_job,
)
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.price_product_config import (
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_VERSION,
)
from stanstock.research.product_frequency_evidence import (
    FREQUENCY_EVIDENCE_KIND,
    _register_product_frequencies,
    _validate_registered_asset,
)
from stanstock.research.product_pipeline import (
    CALCULATION_ARTIFACT_KIND,
    verify_price_product_output,
)
from stanstock.research.refresh_evidence import ANALYSIS_OUTPUT_MANIFEST_KIND, lookup_manifest

if TYPE_CHECKING:
    from stanstock.core.refresh_verification import ReplayedScheduledRefresh

PROFILE = "research_product_v1"
ISSUANCE_KEY = "scheduled"
REGION = "us"
MARKET_STAGE = "market"
EVALUATION_STAGE = "evaluation"
PORTFOLIO_STAGE = "portfolio_snapshots"
FREQUENCY_STAGE = "frequencies"
FREQUENCY_JOB = FREQUENCY_RESEARCH_JOB
STAGE_NAMES = frozenset({MARKET_STAGE, EVALUATION_STAGE, PORTFOLIO_STAGE})
SATISFIED_DOWNSTREAM_STATUSES = frozenset({JobRun.Status.SUCCESS, JobRun.Status.SKIPPED})
EXPECTED_STAGE_ERRORS = (OSError, ProviderError, ValueError)
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class ScheduledResearchExecution:
    """Count-only command handoff for one scheduled invocation."""

    parent: JobRun
    target_date: date
    analysis_count: int
    prediction_count: int


def scheduled_identity(owner: object) -> dict[str, str]:
    """Return the stable owner/issuance/product identity used by all jobs."""

    user_model = get_user_model()
    if not isinstance(owner, user_model) or not owner.is_active:
        raise ValueError("Scheduled research owner must be an active user")
    return {
        "owner_id": str(owner.pk),
        "issuance_key": ISSUANCE_KEY,
        "product": PRODUCT_VERSION,
    }


def resolve_scheduled_owner() -> Any:
    """Resolve the authorized owner deterministically and fail closed.

    Basic policy is bound to its recorded licensed user.  Display-entitled
    plans use the configured owner username.  There is deliberately no
    fallback to the first active account.
    """

    record = ProviderRecord.objects.filter(provider=TWELVE_DATA_PROVIDER).first()
    if record is None:
        raise ValueError("Scheduled research provider authorization is missing")
    validate_provider_usage(record)
    plan = normalized_provider_plan(str(record.metadata.get("plan") or ""))
    user_model = get_user_model()
    if plan == BASIC_PLAN:
        owner_id = str(record.metadata.get("licensed_user_id") or "")
        if not owner_id:
            raise ValueError("Scheduled research licensed owner is missing")
        owner = user_model.objects.filter(pk=owner_id, is_active=True).first()
    else:
        username = str(settings.OWNER_USERNAME).strip()
        if not username:
            raise ValueError("Scheduled research owner username is missing")
        owner = user_model.objects.filter(username=username, is_active=True).first()
    if owner is None:
        raise ValueError("Scheduled research owner is unavailable or inactive")
    return owner


def execute_scheduled_research_refresh(
    *,
    decision_time: datetime | None = None,
    core_config_path: Path | None = None,
    store: AssetStore | None = None,
    enforce_rate_limit: bool = True,
) -> ScheduledResearchExecution:
    """Run or independently replay one native prospective scheduled target."""

    effective_time = decision_time or timezone.now()
    target_date, proposed_grade = resolve_us_target_date(decision_time=effective_time)
    owner = resolve_scheduled_owner()
    identity = scheduled_identity(owner)
    parent_job_name = product_job_name(SCHEDULED_RESEARCH_JOB, identity)
    asset_store = store or open_asset_store()
    config_path = core_config_path or default_us_universe_config_path()

    replayed_prior: list[Any] = []

    def before_attempt() -> None:
        prior = JobRun.objects.filter(
            job_name=parent_job_name,
            region=REGION,
            target_date=target_date,
            status=JobRun.Status.SUCCESS,
        ).first()
        if prior is not None:
            replayed_prior.append(
                replay_recorded_research_product_refresh(prior, store=asset_store)
            )
            _verify_recorded_frequency_stage(
                parent=prior,
                target_date=target_date,
                owner_id=identity["owner_id"],
                store=asset_store,
            )
        return None

    def parent_task(parent: JobRun) -> JobExecutionResult:
        details: dict[str, Any] = {
            "profile": PROFILE,
            "target_date": target_date.isoformat(),
            "invocation_identity": identity,
            "stages": {},
        }
        _persist_parent_details(parent, details)

        try:
            completed = _completed_scheduled_run(
                target_date=target_date,
                owner_id=identity["owner_id"],
                store=asset_store,
            )
        except (OSError, ValueError):
            details["verification"] = {
                "status": "failed",
                "reason_code": "completed_product_recovery_invalid",
                "message": "Existing scheduled research output failed local verification",
            }
            _persist_parent_details(parent, details)
            raise ValueError("Existing scheduled research output failed verification") from None
        if completed is None:
            if proposed_grade != UniverseSnapshot.Grade.OBSERVED:
                raise ValueError(
                    "A fresh automatic research issuance is outside its observed window"
                )
            revision = clean_git_revision(Path(settings.BASE_DIR))
            os.environ["STANSTOCK_CODE_REVISION"] = revision
        else:
            # A committed, independently verified old output remains valid
            # after an upgrade.  Its own recorded revision is authoritative;
            # current checkout cleanliness is relevant only to fresh work.
            revision = completed.code_revision
        details["code_revision"] = revision
        _persist_parent_details(parent, details)

        market = _run_stage(
            parent=parent,
            details=details,
            stage_name=MARKET_STAGE,
            job_name=product_job_name(DAILY_RESEARCH_JOB, identity),
            region=REGION,
            target_date=target_date,
            task=lambda: execute_daily_research_job(
                target_date=target_date,
                owner=owner,
                issuance_key=ISSUANCE_KEY,
                issued_on_time=True,
                store=asset_store,
                core_config_path=config_path,
                enforce_rate_limit=enforce_rate_limit,
                derive_frequencies=False,
            ),
        )
        if market is None or market.status not in {
            JobRun.Status.SUCCESS,
            JobRun.Status.SKIPPED,
        }:
            raise ValueError("Scheduled research market stage produced no complete output")

        downstream_failures: list[str] = []
        frequency = _run_frequency_stage(
            parent=parent,
            details=details,
            target_date=target_date,
            owner_id=identity["owner_id"],
            store=asset_store,
        )
        if frequency.status not in SATISFIED_DOWNSTREAM_STATUSES:
            downstream_failures.append(FREQUENCY_STAGE)
        evaluation = _run_stage(
            parent=parent,
            details=details,
            stage_name=EVALUATION_STAGE,
            job_name=EVALUATION_JOB_NAME,
            region=REGION,
            target_date=target_date,
            task=lambda: execute_prediction_evaluation_job(
                provider=TWELVE_DATA_PROVIDER,
                evaluation_date=target_date,
                evaluation_time=timezone.now(),
                benchmark_subject="SPY",
            ),
            failures=downstream_failures,
        )
        portfolio = _run_stage(
            parent=parent,
            details=details,
            stage_name=PORTFOLIO_STAGE,
            job_name=PORTFOLIO_JOB_NAME,
            region="",
            target_date=target_date,
            task=lambda: execute_portfolio_snapshot_job(
                target_date=target_date,
                require_session_date=True,
                require_all=True,
            ),
            failures=downstream_failures,
        )
        for stage_name, child in (
            (EVALUATION_STAGE, evaluation),
            (PORTFOLIO_STAGE, portfolio),
        ):
            if child is not None and child.status not in SATISFIED_DOWNSTREAM_STATUSES:
                downstream_failures.append(stage_name)
        if downstream_failures:
            raise ValueError(
                f"Scheduled research has {len(set(downstream_failures))} incomplete "
                "downstream stage(s)"
            )

        try:
            verification = verify_scheduled_research_refresh(
                target_date=target_date,
                owner=owner,
                code_revision=revision,
                stages=details["stages"],
                store=asset_store,
                core_config_path=config_path,
            )
        except RefreshVerificationError as exc:
            details["verification"] = exc.to_failure_details()
            _persist_parent_details(parent, details)
            raise ValueError("Scheduled research output verification failed") from exc
        except (OSError, ValueError):
            details["verification"] = {
                "status": "failed",
                "reason_code": "product_verification_input_invalid",
                "message": "Scheduled research evidence could not be verified",
            }
            _persist_parent_details(parent, details)
            raise ValueError("Scheduled research output verification failed") from None
        details["verification"] = verification
        return JobExecutionResult(details=details)

    parent = execute_target_job(
        job_name=parent_job_name,
        region=REGION,
        target_date=target_date,
        task=parent_task,
        before_attempt=before_attempt,
    )
    if parent.status == JobRun.Status.SKIPPED:
        if len(replayed_prior) != 1:
            raise ValueError("Scheduled research skip has no independently replayed parent")
        verification = replayed_prior[0].verification
    else:
        verification = parent.details.get("verification", {})
    return ScheduledResearchExecution(
        parent=parent,
        target_date=target_date,
        analysis_count=_summary_count(verification, "stock_analysis_count"),
        prediction_count=_summary_count(verification, "prediction_count"),
    )


def verify_scheduled_research_refresh(
    *,
    target_date: date,
    owner: object,
    code_revision: str,
    stages: dict[str, Any],
    store: AssetStore | None = None,
    core_config_path: Path | None = None,
) -> dict[str, Any]:
    """Independently prove one prospective parent from persisted evidence."""

    from stanstock.core.refresh_verification import (
        _resolve_stage_run,
        _verify_asset_evidence,
        _verify_evaluation_details,
        _verify_portfolio,
    )

    if set(stages) != STAGE_NAMES:
        raise RefreshVerificationError(
            "product_parent_stages_invalid",
            "The research parent has an unexpected child stage set",
        )
    identity = scheduled_identity(owner)
    daily_name = product_job_name(DAILY_RESEARCH_JOB, identity)
    market = _resolve_product_stage(
        stages,
        MARKET_STAGE,
        job_name=daily_name,
        region=REGION,
        target_date=target_date,
    )
    if market.status != JobRun.Status.SUCCESS:
        raise RefreshVerificationError(
            "product_market_not_success",
            "The research market child did not resolve to a successful run",
        )

    asset_store = store or open_asset_store()
    intake = load_product_intake(
        target_date=target_date,
        owner_id=identity["owner_id"],
        issuance_key=ISSUANCE_KEY,
        store=asset_store,
    )
    if intake is None:
        raise RefreshVerificationError(
            "product_intake_missing",
            "The scheduled research intake is missing",
        )
    intake_payload = product_intake_payload(intake, store=asset_store)
    if (
        intake_payload.get("owner_id") != identity["owner_id"]
        or intake_payload.get("issuance_key") != ISSUANCE_KEY
        or intake_payload.get("product_version") != PRODUCT_VERSION
        or intake_payload.get("target_date") != target_date.isoformat()
        or intake_payload.get("evidence_grade") != UniverseSnapshot.Grade.OBSERVED
        or intake_payload.get("source_provider") != TWELVE_DATA_PROVIDER
        or intake_payload.get("policy_identity") != PRODUCT_EFFECTIVE_CONFIG_HASH
    ):
        raise RefreshVerificationError(
            "product_intake_identity_invalid",
            "The scheduled intake does not match its target, owner, and product policy",
        )
    production_config_path = default_us_universe_config_path()
    if (core_config_path or production_config_path).resolve() != production_config_path.resolve():
        raise RefreshVerificationError(
            "product_core_config_path_invalid",
            "Scheduled research must use the reviewed production core configuration",
        )
    universe_config = load_us_universe_config(production_config_path)
    if intake_payload.get("core_config") != universe_config.raw:
        raise RefreshVerificationError(
            "product_core_config_mismatch",
            "The scheduled intake does not bind the reviewed production core configuration",
        )

    snapshot, run = _exact_product_output(intake.asset, target_date=target_date, store=asset_store)
    if (
        run.code_revision != code_revision
        or _REVISION_RE.fullmatch(code_revision) is None
        or not run.issued_on_time
        or run.target_date != target_date
        or snapshot.grade != UniverseSnapshot.Grade.OBSERVED
        or run.config_version != PRODUCT_VERSION
        or run.config_hash != PRODUCT_EFFECTIVE_CONFIG_HASH
    ):
        raise RefreshVerificationError(
            "product_run_identity_invalid",
            "The scheduled output does not bind its observed target and committed revision",
        )

    market_details = _details(market)
    if (
        market_details.get("intake_asset") != asset_ref_for(intake.asset).to_json()
        or market_details.get("snapshot_id") != str(snapshot.pk)
        or market_details.get("analysis_run_id") != str(run.pk)
        or market_details.get("evidence_grade") != UniverseSnapshot.Grade.OBSERVED
        or type(market_details.get("recovered")) is not bool
        or type(market_details.get("credits_used")) is not int
        or market_details["credits_used"] < 0
    ):
        raise RefreshVerificationError(
            "product_market_details_invalid",
            "The market child does not bind its exact intake and output",
        )

    intake_child_name = product_job_name(RESEARCH_INTAKE_JOB, identity)
    intake_children = list(
        JobRun.objects.filter(
            job_name=intake_child_name,
            region=REGION,
            target_date=target_date,
            status=JobRun.Status.SUCCESS,
        )
    )
    if len(intake_children) != 1:
        raise RefreshVerificationError(
            "product_intake_child_invalid",
            "The scheduled intake has no unique successful acquisition child",
        )
    intake_child = intake_children[0]
    recorded_intake_child = market_details.get("intake_job_id")
    if recorded_intake_child not in {None, str(intake_child.pk)} or _details(intake_child).get(
        "snapshot_id"
    ) != str(snapshot.pk):
        raise RefreshVerificationError(
            "product_intake_child_mismatch",
            "The market child does not bind the authoritative intake child",
        )

    try:
        verify_price_product_output(run=run, store=asset_store, replay=False)
    except (OSError, ValueError) as exc:
        if isinstance(exc, RefreshVerificationError):
            raise
        raise RefreshVerificationError(
            "product_output_invalid",
            "The scheduled research output failed its registered evidence verification",
        ) from exc

    membership = product_membership_payload(snapshot, store=asset_store)
    source_time = _aware_timestamp(membership.get("decision_time"))
    if run.data_cutoff != source_time or source_time > run.generated_at:
        raise RefreshVerificationError(
            "product_source_boundary_invalid",
            "The scheduled output does not use its captured membership decision time",
        )
    qualified_ids = _qualified_ids(membership)
    memberships = list(UniverseMembership.objects.filter(snapshot=snapshot))
    actual_qualified = {row.listing_id for row in memberships if row.eligible}
    if actual_qualified != qualified_ids:
        raise RefreshVerificationError(
            "product_membership_rows_invalid",
            "The registered qualified set differs from the persisted membership rows",
        )
    analyses = list(StockAnalysis.objects.filter(run=run))
    predictions = list(Prediction.objects.filter(analysis__run=run))
    if (
        {analysis.listing_id for analysis in analyses} != qualified_ids
        or len(analyses) != len(qualified_ids)
        or len(predictions) != len(analyses) * 5
    ):
        raise RefreshVerificationError(
            "product_output_count_invalid",
            "The scheduled output does not contain exactly five rows per qualified listing",
        )

    evaluation = _resolve_stage_run(
        stages,
        EVALUATION_STAGE,
        job_name=EVALUATION_JOB_NAME,
        region=REGION,
        target_date=target_date,
    )
    if evaluation.status != JobRun.Status.SUCCESS:
        raise RefreshVerificationError(
            "product_evaluation_not_success",
            "The evaluation child did not resolve to a successful run",
        )
    evaluation_summary, evaluation_assets = _verify_evaluation_details(
        evaluation,
        universe_config=universe_config,
        target_date=target_date,
    )
    portfolio = _resolve_stage_run(
        stages,
        PORTFOLIO_STAGE,
        job_name=PORTFOLIO_JOB_NAME,
        region="",
        target_date=target_date,
    )
    portfolio_summary, portfolio_assets = _verify_portfolio(
        portfolio,
        target_date=target_date,
    )

    catalog_assets, product_asset_ids = _product_asset_closure(
        intake_asset=intake.asset,
        snapshot=snapshot,
        run=run,
        membership=membership,
        source_time=source_time,
        analyses=analyses,
    )
    asset_manifest = _verify_asset_evidence(
        product_asset_ids | evaluation_assets | portfolio_assets
    )
    return {
        "status": "verified",
        "profile": PROFILE,
        "target_date": target_date.isoformat(),
        "owner_id": identity["owner_id"],
        "snapshot_id": str(snapshot.pk),
        "snapshot_grade": snapshot.grade,
        "membership_count": len(memberships),
        "eligible_count": len(qualified_ids),
        "excluded_count": len(memberships) - len(qualified_ids),
        "analysis_run_id": str(run.pk),
        "stock_analysis_count": len(analyses),
        "prediction_count": len(predictions),
        "code_revision": code_revision,
        "catalog_asset_count": len(catalog_assets),
        "evaluation": evaluation_summary,
        "portfolio": portfolio_summary,
        "asset_manifest": asset_manifest,
        "child_job_run_ids": {
            "market": str(market.pk),
            "research_intake": str(intake_child.pk),
            "evaluation": str(evaluation.pk),
            "portfolio_snapshots": str(portfolio.pk),
        },
    }


def replay_recorded_research_product_refresh(
    parent: JobRun,
    *,
    store: AssetStore | None = None,
) -> ReplayedScheduledRefresh:
    """Replay a successful prospective parent without provider access."""

    from stanstock.core.refresh_verification import ReplayedScheduledRefresh

    persisted = JobRun.objects.filter(pk=parent.pk).first()
    if persisted is None or persisted.region != REGION or persisted.status != JobRun.Status.SUCCESS:
        raise RefreshVerificationError(
            "recorded_product_parent_identity_invalid",
            "The recorded research parent is not a successful US refresh",
        )
    details = _details(persisted)
    identity = details.get("invocation_identity")
    if (
        not isinstance(identity, dict)
        or set(identity) != {"owner_id", "issuance_key", "product"}
        or identity.get("issuance_key") != ISSUANCE_KEY
        or identity.get("product") != PRODUCT_VERSION
        or persisted.job_name != product_job_name(SCHEDULED_RESEARCH_JOB, identity)
        or details.get("profile") != PROFILE
        or details.get("target_date") != persisted.target_date.isoformat()
    ):
        raise RefreshVerificationError(
            "recorded_product_parent_details_invalid",
            "The recorded research parent has an invalid invocation identity",
        )
    user_model = get_user_model()
    owner = user_model.objects.filter(pk=identity["owner_id"], is_active=True).first()
    if owner is None:
        raise RefreshVerificationError(
            "recorded_product_owner_invalid",
            "The recorded research owner is unavailable or inactive",
        )
    current_owner = resolve_scheduled_owner()
    if current_owner.pk != owner.pk:
        raise RefreshVerificationError(
            "recorded_product_owner_policy_mismatch",
            "The recorded research owner no longer matches the authorized scheduled owner",
        )
    stages = details.get("stages")
    revision = details.get("code_revision")
    if not isinstance(stages, dict) or not isinstance(revision, str):
        raise RefreshVerificationError(
            "recorded_product_parent_details_invalid",
            "The recorded research parent has no valid stages or revision",
        )
    try:
        verification = verify_scheduled_research_refresh(
            target_date=persisted.target_date,
            owner=owner,
            code_revision=revision,
            stages=stages,
            store=store,
        )
    except RefreshVerificationError:
        raise
    except (OSError, ValueError):
        raise RefreshVerificationError(
            "recorded_product_evidence_invalid",
            "The recorded research evidence could not be replayed",
        ) from None
    if details.get("verification") != verification:
        raise RefreshVerificationError(
            "recorded_product_verification_mismatch",
            "The recorded research verification differs from its canonical replay",
        )
    snapshot = UniverseSnapshot.objects.filter(pk=UUID(verification["snapshot_id"])).first()
    run = AnalysisRun.objects.filter(pk=UUID(verification["analysis_run_id"])).first()
    if snapshot is None or run is None:
        raise RefreshVerificationError(
            "recorded_product_output_missing",
            "The replayed research output could not be resolved",
        )
    membership = product_membership_payload(snapshot, store=store or open_asset_store())
    catalog_assets = _catalog_assets(
        membership,
        cutoff=_aware_timestamp(membership.get("decision_time")),
    )
    return ReplayedScheduledRefresh(
        parent=persisted,
        verification=verification,
        snapshot=snapshot,
        analysis_run=run,
        catalog_assets=catalog_assets,
    )


def _completed_scheduled_run(
    *,
    target_date: date,
    owner_id: str,
    store: AssetStore,
) -> AnalysisRun | None:
    intake = load_product_intake(
        target_date=target_date,
        owner_id=owner_id,
        issuance_key=ISSUANCE_KEY,
        store=store,
    )
    if intake is None:
        return None
    snapshots = list(
        UniverseSnapshot.objects.filter(
            universe__slug=f"{PRODUCT_VERSION}-{intake.asset.sha256[:16]}"
        )
    )
    if len(snapshots) > 1:
        raise ValueError("Scheduled research intake has conflicting snapshots")
    if not snapshots:
        return None
    payload = product_membership_payload(snapshots[0], store=store)
    if payload.get("intake") != asset_ref_for(intake.asset).to_json():
        raise ValueError("Scheduled research snapshot belongs to another intake")
    runs = list(
        AnalysisRun.objects.filter(
            universe_snapshot=snapshots[0],
            config_version=PRODUCT_VERSION,
            config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
        )
    )
    if len(runs) > 1:
        raise ValueError("Scheduled research intake has conflicting completed outputs")
    if not runs:
        return None
    verify_price_product_output(run=runs[0], store=store, replay=False)
    return runs[0]


def _exact_product_output(
    intake_asset: DataAsset,
    *,
    target_date: date,
    store: AssetStore,
) -> tuple[UniverseSnapshot, AnalysisRun]:
    snapshots = list(
        UniverseSnapshot.objects.filter(
            universe__slug=f"{PRODUCT_VERSION}-{intake_asset.sha256[:16]}",
            as_of_date=target_date,
        )
    )
    if len(snapshots) != 1:
        raise RefreshVerificationError(
            "product_snapshot_not_unique",
            "The scheduled intake does not have one exact snapshot",
        )
    snapshot = snapshots[0]
    payload = product_membership_payload(snapshot, store=store)
    if payload.get("intake") != asset_ref_for(intake_asset).to_json():
        raise RefreshVerificationError(
            "product_snapshot_intake_mismatch",
            "The scheduled snapshot does not bind its exact intake",
        )
    runs = list(
        AnalysisRun.objects.filter(
            universe_snapshot=snapshot,
            config_version=PRODUCT_VERSION,
            config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
        )
    )
    if len(runs) != 1:
        raise RefreshVerificationError(
            "product_analysis_run_not_unique",
            "The scheduled snapshot does not have one exact product run",
        )
    return snapshot, runs[0]


def _product_asset_closure(
    *,
    intake_asset: DataAsset,
    snapshot: UniverseSnapshot,
    run: AnalysisRun,
    membership: dict[str, object],
    source_time: datetime,
    analyses: list[StockAnalysis],
) -> tuple[tuple[DataAsset, ...], set[UUID]]:
    asset_ids = {intake_asset.pk}
    membership_assets = list(
        DataAsset.objects.filter(
            provider="stanstock",
            kind=PRODUCT_MEMBERSHIP_KIND,
            subject=str(snapshot.pk),
        )
    )
    if len(membership_assets) != 1:
        raise RefreshVerificationError(
            "product_membership_asset_not_unique",
            "The scheduled snapshot has no unique membership asset",
        )
    asset_ids.add(membership_assets[0].pk)
    catalog_assets = _catalog_assets(membership, cutoff=source_time)
    asset_ids.update(asset.pk for asset in catalog_assets)

    normalized_assets: dict[UUID, DataAsset] = {}
    benchmark_ref = membership.get("benchmark_asset")
    if not isinstance(benchmark_ref, dict):
        raise RefreshVerificationError(
            "product_benchmark_asset_missing",
            "The scheduled membership has no benchmark asset",
        )
    benchmark = resolve_asset_ref(AssetRef.from_json(benchmark_ref), cutoff=source_time)
    normalized_assets[benchmark.pk] = benchmark
    admissions = membership.get("admissions")
    if not isinstance(admissions, dict):
        raise RefreshVerificationError(
            "product_admissions_invalid",
            "The scheduled membership has no admission evidence",
        )
    for entry in admissions.values():
        if not isinstance(entry, dict) or entry.get("status") != "admitted":
            continue
        raw_ref = entry.get("price_asset")
        if not isinstance(raw_ref, dict):
            raise RefreshVerificationError(
                "product_admitted_asset_missing",
                "An admitted product member has no price asset",
            )
        asset = resolve_asset_ref(AssetRef.from_json(raw_ref), cutoff=source_time)
        normalized_assets[asset.pk] = asset
    for asset in normalized_assets.values():
        asset_ids.add(asset.pk)
        asset_ids.add(raw_price_asset_for(asset, cutoff=source_time).pk)

    calculation_assets = list(
        DataAsset.objects.filter(
            provider="stanstock",
            kind=CALCULATION_ARTIFACT_KIND,
            subject=str(run.pk),
        )
    )
    if len(calculation_assets) != len(analyses) or {
        asset.metadata.get("listing_id") for asset in calculation_assets
    } != {str(analysis.listing_id) for analysis in analyses}:
        raise RefreshVerificationError(
            "product_calculation_asset_set_invalid",
            "The scheduled run does not have one calculation asset per analysis",
        )
    asset_ids.update(asset.pk for asset in calculation_assets)
    manifest = lookup_manifest(run.pk)
    if manifest.count != 1 or manifest.asset is None:
        raise RefreshVerificationError(
            "product_manifest_not_unique",
            "The scheduled run has no unique output manifest",
        )
    if (
        manifest.asset.provider != "stanstock"
        or manifest.asset.kind != ANALYSIS_OUTPUT_MANIFEST_KIND
        or manifest.asset.subject != str(run.pk)
    ):
        raise RefreshVerificationError(
            "product_manifest_identity_invalid",
            "The scheduled output manifest has an invalid registry identity",
        )
    asset_ids.add(manifest.asset.pk)
    return catalog_assets, asset_ids


def _catalog_assets(
    membership: dict[str, object],
    *,
    cutoff: datetime,
) -> tuple[DataAsset, ...]:
    raw_refs = membership.get("catalog_assets")
    if not isinstance(raw_refs, list):
        raise RefreshVerificationError(
            "product_catalog_assets_invalid",
            "The scheduled membership has no catalog asset list",
        )
    try:
        assets = tuple(
            resolve_asset_ref(AssetRef.from_json(raw_ref), cutoff=cutoff) for raw_ref in raw_refs
        )
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "product_catalog_assets_invalid",
            "The scheduled membership catalog references are invalid",
        ) from exc
    if (
        not assets
        or len({asset.pk for asset in assets}) != len(assets)
        or any(
            asset.provider != TWELVE_DATA_PROVIDER or asset.kind != "stock_catalog"
            for asset in assets
        )
    ):
        raise RefreshVerificationError(
            "product_catalog_assets_invalid",
            "The scheduled membership catalog set is missing or ambiguous",
        )
    return assets


def _resolve_product_stage(
    stages: dict[str, Any],
    stage_name: str,
    *,
    job_name: str,
    region: str,
    target_date: date,
) -> JobRun:
    info = stages.get(stage_name)
    if not isinstance(info, dict):
        raise RefreshVerificationError(
            "product_stage_details_missing",
            f"The research parent has no {stage_name!r} stage details",
        )
    try:
        candidate_id = uuid.UUID(str(info.get("job_run_id")))
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "product_stage_id_invalid",
            f"The research parent {stage_name!r} child id is invalid",
        ) from exc
    candidate = JobRun.objects.filter(pk=candidate_id).first()
    if candidate is None:
        raise RefreshVerificationError(
            "product_stage_missing",
            f"The research parent {stage_name!r} child is missing",
        )
    if (
        candidate.job_name != job_name
        or candidate.region != region
        or candidate.target_date != target_date
        or info.get("status") != candidate.status
        or info.get("attempt") != candidate.attempt
    ):
        raise RefreshVerificationError(
            "product_stage_identity_invalid",
            f"The research parent {stage_name!r} child identity is invalid",
        )
    if candidate.status == JobRun.Status.SUCCESS:
        return candidate
    if candidate.status != JobRun.Status.SKIPPED:
        return candidate
    reference = _details(candidate).get("successful_run_id")
    try:
        referenced_id = uuid.UUID(str(reference))
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "product_stage_skip_invalid",
            f"The research parent {stage_name!r} skip reference is invalid",
        ) from exc
    referenced = JobRun.objects.filter(pk=referenced_id).first()
    if (
        referenced is None
        or referenced.job_name != job_name
        or referenced.region != region
        or referenced.target_date != target_date
        or referenced.status != JobRun.Status.SUCCESS
    ):
        raise RefreshVerificationError(
            "product_stage_skip_invalid",
            f"The research parent {stage_name!r} skip does not resolve to a success",
        )
    return referenced


def _frequency_job_name(owner_id: str) -> str:
    return product_job_name(
        FREQUENCY_JOB,
        {"owner_id": owner_id, "issuance_key": ISSUANCE_KEY, "product": PRODUCT_VERSION},
    )


def _legacy_frequency_child(
    *, target_date: date, owner_id: str, store: AssetStore
) -> JobRun | None:
    legacy = JobRun.objects.filter(
        job_name=FREQUENCY_JOB,
        region=REGION,
        target_date=target_date,
        status=JobRun.Status.SUCCESS,
    )
    if not legacy.exists():
        return None
    source = _completed_scheduled_run(target_date=target_date, owner_id=owner_id, store=store)
    if source is None:
        return None
    matches = list(legacy.filter(details__analysis_run_id=str(source.pk))[:2])
    if len(matches) > 1:
        raise ValueError("Legacy scheduled frequency child is ambiguous")
    return matches[0] if matches else None


def _run_frequency_stage(
    *,
    parent: JobRun,
    details: dict[str, Any],
    target_date: date,
    owner_id: str,
    store: AssetStore,
) -> JobRun:
    """Execute a recoverable derived-only child without altering the frozen stage map."""

    job_name = _frequency_job_name(owner_id)

    def task(_child: JobRun) -> JobExecutionResult:
        completed = _completed_scheduled_run(
            target_date=target_date,
            owner_id=owner_id,
            store=store,
        )
        if completed is None:
            raise ValueError("Frequency stage has no complete scheduled source run")
        asset = _register_product_frequencies(
            run=completed,
            store=store,
            derivation_source="scheduled_stage",
        )
        return JobExecutionResult(
            details={
                "analysis_run_id": str(completed.id),
                "frequency_asset_id": str(asset.id),
                "frequency_asset_sha256": asset.sha256,
            }
        )

    try:
        legacy = _legacy_frequency_child(target_date=target_date, owner_id=owner_id, store=store)
        child = legacy or execute_target_job(
            job_name=job_name, region=REGION, target_date=target_date, task=task
        )
    except EXPECTED_STAGE_ERRORS:
        failed = (
            JobRun.objects.filter(job_name=job_name, region=REGION, target_date=target_date)
            .order_by("-attempt")
            .first()
        )
        details[FREQUENCY_STAGE] = {
            "status": JobRun.Status.FAILED if failed is None else failed.status,
            "job_run_id": None if failed is None else str(failed.id),
        }
        _persist_parent_details(parent, details)
        return failed or JobRun(
            job_name=job_name,
            region=REGION,
            target_date=target_date,
            status=JobRun.Status.FAILED,
        )
    success = child
    if child.status == JobRun.Status.SKIPPED:
        reference = child.details.get("successful_run_id")
        success = JobRun.objects.get(
            pk=reference,
            job_name=job_name,
            region=REGION,
            target_date=target_date,
            status=JobRun.Status.SUCCESS,
        )
    details[FREQUENCY_STAGE] = {
        "status": child.status,
        "job_run_id": str(child.id),
        "analysis_run_id": success.details.get("analysis_run_id"),
        "frequency_asset_id": success.details.get("frequency_asset_id"),
        "frequency_asset_sha256": success.details.get("frequency_asset_sha256"),
    }
    details["frequency_verification"] = _verify_frequency_stage(
        child=success,
        target_date=target_date,
        owner_id=owner_id,
        store=store,
    )
    _persist_parent_details(parent, details)
    return child


def _verify_recorded_frequency_stage(
    *,
    parent: JobRun,
    target_date: date,
    owner_id: str,
    store: AssetStore,
) -> None:
    """Recheck a new parent's optional sibling evidence before skip recovery.

    Frozen parents predate the sibling stage and intentionally have no such
    field; their canonical replay remains byte-for-byte governed by the
    original verifier.  Once the field exists, however, a parent cannot use
    the completed-parent shortcut without independently revalidating it.
    """

    recorded = parent.details.get("frequency_verification")
    if recorded is None:
        has_child = JobRun.objects.filter(
            job_name=_frequency_job_name(owner_id),
            region=REGION,
            target_date=target_date,
            status=JobRun.Status.SUCCESS,
        ).exists()
        legacy = _legacy_frequency_child(target_date=target_date, owner_id=owner_id, store=store)
        if has_child or legacy is not None or FREQUENCY_STAGE in parent.details:
            raise ValueError("Scheduled frequency verification binding is missing")
        return
    if not isinstance(recorded, dict):
        raise ValueError("Scheduled frequency verification block is invalid")
    try:
        child_id = UUID(str(recorded["child_job_run_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Scheduled frequency verification block is invalid") from exc
    child = JobRun.objects.filter(pk=child_id).first()
    if child is None:
        raise ValueError("Scheduled frequency verification child is unavailable")
    actual = _verify_frequency_stage(
        child=child,
        target_date=target_date,
        owner_id=owner_id,
        store=store,
    )
    if actual != recorded:
        raise ValueError("Scheduled frequency verification block does not match its child")


def _verify_frequency_stage(
    *,
    child: JobRun,
    target_date: date,
    owner_id: str,
    store: AssetStore,
) -> dict[str, object]:
    """Verify the sibling evidence without extending frozen parent stages."""

    if (
        child.job_name not in {_frequency_job_name(owner_id), FREQUENCY_JOB}
        or child.region != REGION
        or child.target_date != target_date
        or child.status != JobRun.Status.SUCCESS
        or not isinstance(child.details, dict)
    ):
        raise ValueError("Scheduled frequency child identity is invalid")
    try:
        run_id = UUID(str(child.details["analysis_run_id"]))
        asset_id = UUID(str(child.details["frequency_asset_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Scheduled frequency child details are invalid") from exc
    source = _completed_scheduled_run(target_date=target_date, owner_id=owner_id, store=store)
    run = AnalysisRun.objects.filter(pk=run_id).first()
    asset = DataAsset.objects.filter(pk=asset_id).first()
    if (
        source is None
        or run != source
        or asset is None
        or asset.kind != FREQUENCY_EVIDENCE_KIND
        or asset.sha256 != child.details.get("frequency_asset_sha256")
    ):
        raise ValueError("Scheduled frequency child does not bind its exact source evidence")
    _validate_registered_asset(asset, store=store, expected_run=run, verify_source=True)
    return {
        "status": "verified",
        "child_job_run_id": str(child.id),
        "analysis_run_id": str(run.id),
        "frequency_asset_id": str(asset.id),
        "frequency_asset_sha256": asset.sha256,
    }


def _run_stage(
    *,
    parent: JobRun,
    details: dict[str, Any],
    stage_name: str,
    job_name: str,
    region: str,
    target_date: date,
    task: Callable[[], JobRun],
    failures: list[str] | None = None,
) -> JobRun | None:
    child: JobRun | None
    try:
        child = task()
    except EXPECTED_STAGE_ERRORS as exc:
        child = (
            JobRun.objects.filter(
                job_name=job_name,
                region=region,
                target_date=target_date,
            )
            .order_by("-attempt")
            .first()
        )
        _record_stage(parent, details, stage_name, child, error_type=type(exc).__name__)
        if failures is None:
            raise ValueError(f"Scheduled research {stage_name} stage failed") from None
        failures.append(stage_name)
        return child
    _record_stage(parent, details, stage_name, child)
    return child


def _record_stage(
    parent: JobRun,
    details: dict[str, Any],
    stage_name: str,
    child: JobRun | None,
    *,
    error_type: str = "",
) -> None:
    stages = details.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("Scheduled research stage state is invalid")
    stages[stage_name] = {
        "job_run_id": str(child.pk) if child is not None else None,
        "status": child.status if child is not None else JobRun.Status.FAILED,
        "attempt": child.attempt if child is not None else None,
        "error_type": error_type,
    }
    _persist_parent_details(parent, details)


def _persist_parent_details(parent: JobRun, details: dict[str, Any]) -> None:
    JobRun.objects.filter(pk=parent.pk).update(details=details)


def _details(run: JobRun) -> dict[str, Any]:
    return run.details if isinstance(run.details, dict) else {}


def _aware_timestamp(value: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise RefreshVerificationError(
            "product_source_time_invalid",
            "The scheduled membership source time is malformed",
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RefreshVerificationError(
            "product_source_time_invalid",
            "The scheduled membership source time is not timezone-aware",
        )
    return parsed


def _qualified_ids(membership: dict[str, object]) -> set[UUID]:
    raw_ids = membership.get("qualified_listing_ids")
    if not isinstance(raw_ids, list):
        raise RefreshVerificationError(
            "product_qualified_set_invalid",
            "The scheduled membership qualified set is invalid",
        )
    try:
        values = {UUID(str(raw_id)) for raw_id in raw_ids}
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "product_qualified_set_invalid",
            "The scheduled membership qualified set contains an invalid id",
        ) from exc
    if len(values) != len(raw_ids) or not values:
        raise RefreshVerificationError(
            "product_qualified_set_invalid",
            "The scheduled membership qualified set is empty or duplicated",
        )
    return values


def _summary_count(summary: object, key: str) -> int:
    if not isinstance(summary, dict):
        return 0
    value = summary.get(key)
    return value if type(value) is int and value >= 0 else 0

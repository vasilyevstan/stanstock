"""Independent post-refresh verification for `scheduled_refresh`.

`execute_target_job`'s `JobRun.status` proves a child *process* ran to
completion; it does not by itself prove the *local outputs* that status
implies actually exist and are internally consistent. This module is called
from `scheduled_refresh` after every required child stage reports a
satisfied status and before the parent job is allowed to succeed. It never
trusts a child's own bookkeeping without re-deriving the underlying evidence
from persisted rows: every identity used here is re-fetched fresh from the
database using the parent's own recorded `job_run_id`/`analysis_run_id`
references, never the in-process objects a caller happens to be holding.

Nothing here mutates evidence. A failure is reported through
`RefreshVerificationError`, whose `reason_code` and message are always
path-free (no `STANSTOCK_DATA_DIR` value or resolved filesystem path is ever
included), so the caller can persist a `status: failed` summary onto the
parent `JobRun.details` without leaking local paths.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import polars as pl
from django.conf import settings

from stanstock.core.integrity import verify_registered_assets
from stanstock.core.models import JobRun
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.assets import verify_catalog_refs
from stanstock.data.jobs import JOB_NAME as MARKET_JOB_NAME
from stanstock.data.jobs import frozen_long_forecast_gate_for_run
from stanstock.data.live_us import (
    UsUniverseConfig,
    benchmark_asset_for_completed_run,
    is_us_prediction_on_time,
    load_us_universe_config,
)
from stanstock.data.management.config_loader import (
    default_us_scoring_config_path,
    default_us_universe_config_path,
)
from stanstock.data.models import (
    DataAsset,
    Listing,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.data.providers import sec, twelve_data
from stanstock.data.refresh_validation import (
    assert_no_etf_in_membership,
    require_bound_market_data,
    require_spy_listing,
    resolve_catalog_assets,
    verify_membership_evidence,
    verify_snapshot,
)
from stanstock.data.sec_jobs import JOB_NAME as SEC_JOB_NAME
from stanstock.data.sec_refresh_validation import verify_sec_stage
from stanstock.portfolio.jobs import SCHEDULED_JOB_NAME as PORTFOLIO_JOB_NAME
from stanstock.portfolio.refresh_validation import (
    verify_portfolio_snapshot_stage,
)
from stanstock.research import config as research_config
from stanstock.research.forecast_config import (
    load_medium_forecast_config,
    medium_forecast_config_hash,
)
from stanstock.research.forecasting import scenario_from_document
from stanstock.research.jobs import (
    JOB_NAME as EVALUATION_JOB_NAME,
)
from stanstock.research.jobs import (
    TERMINAL_OUTCOME_STATUSES,
    maturity_provider_candidates,
)
from stanstock.research.long_forecast_config import (
    load_long_forecast_config,
    long_forecast_config_hash,
)
from stanstock.research.models import AnalysisRun, Prediction, PredictionOutcome, StockAnalysis
from stanstock.research.outcome_refresh_validation import verify_prediction_outcome
from stanstock.research.provenance import DATA_MODE_PROVIDER, source_data_mode
from stanstock.research.refresh_validation import (
    LongLaneExpectation,
    MediumLaneExpectation,
    verify_analysis_output_manifest,
)
from stanstock.research.service import _decimal, _optional_decimal

MARKET_STAGE = "market"
EVALUATION_STAGE = "evaluation"
PORTFOLIO_STAGE = "portfolio_snapshots"
SEC_STAGE = "sec_fundamentals"

#: Vocabulary of `PredictionOutcome`-recording actions `evaluate_predictions`
#: can report for a single prediction; anything else is not a real action.
EVALUATION_ACTIONS = frozenset({"created", "updated", "skipped"})

#: `Prediction.horizon` values that may legitimately carry each
#: `Prediction.EvidenceRole`. A prediction whose horizon/role combination
#: falls outside this map was never produced by `research.service`'s
#: decision/advisory issuance paths and is treated as a transplant. Decision
#: horizons are derived per-call from the loaded scoring config's own
#: `supported_horizons` (see `_verify_prediction_row`), never hard-coded,
#: since a reviewed configuration may legitimately request more than
#: `short`.
_MEDIUM_ADVISORY_HORIZONS = frozenset(
    {Prediction.Horizon.SIX_MONTH, Prediction.Horizon.TWELVE_MONTH}
)
_LONG_ADVISORY_HORIZONS = frozenset({Prediction.Horizon.THREE_YEAR, Prediction.Horizon.FIVE_YEAR})
_ADVISORY_HORIZONS = _MEDIUM_ADVISORY_HORIZONS | _LONG_ADVISORY_HORIZONS

#: Required identity fields a `source_assets` JSON payload entry
#: (`StockAnalysis.data_quality["source_assets"]`/`Prediction.source_assets`)
#: must carry for `_iter_source_asset_entries` to accept it.
_SOURCE_ASSET_FIELDS = ("id", "provider", "kind", "subject", "sha256")


@dataclass(frozen=True, slots=True)
class ReplayedScheduledRefresh:
    """Canonical, read-only replay of one persisted scheduled parent."""

    parent: JobRun
    verification: dict[str, Any]
    snapshot: UniverseSnapshot
    analysis_run: AnalysisRun
    catalog_assets: tuple[DataAsset, ...]


def replay_recorded_scheduled_refresh(parent: JobRun) -> ReplayedScheduledRefresh:
    """Replay and compare one successful parent's complete production proof.

    The persisted ``verification`` document is an output to compare, never
    authority. The canonical verifier re-reads the parent's exact child
    identities and all underlying evidence with the installed production US
    universe configuration. No provider, credential, or quota boundary is
    consulted.
    """
    persisted = JobRun.objects.filter(pk=parent.pk).first()
    if (
        persisted is None
        or persisted.job_name != "scheduled_refresh"
        or persisted.region != "us"
        or persisted.status != JobRun.Status.SUCCESS
    ):
        raise RefreshVerificationError(
            "recorded_parent_identity_invalid",
            "The recorded scheduled-refresh parent is not a successful US refresh",
        )

    details = _details_dict(persisted)
    stages = details.get("stages")
    if not isinstance(stages, dict):
        raise RefreshVerificationError(
            "recorded_parent_stages_invalid",
            "The recorded scheduled-refresh parent has no valid child stage mapping",
        )
    # Requiredness comes from independently persisted production evidence,
    # never from the mutable parent's stage keys or recorded verification.
    # Deliberately include any same-target success visible at replay time:
    # historical query timing cannot be reconstructed, and failing closed on
    # a later recoverable SEC success is safer than accepting an omitted child.
    market_success = (
        JobRun.objects.filter(
            job_name=MARKET_JOB_NAME,
            region="us",
            target_date=persisted.target_date,
            status=JobRun.Status.SUCCESS,
        )
        .order_by("pk")
        .first()
    )
    if market_success is None:
        raise RefreshVerificationError(
            "recorded_market_success_missing",
            "No authoritative successful market child exists for the recorded target",
        )
    try:
        authoritative_market_gate = frozen_long_forecast_gate_for_run(market_success)
    except ValueError:
        raise RefreshVerificationError(
            "recorded_market_gate_invalid",
            "The authoritative market child has no valid frozen long-forecast gate",
        ) from None
    sec_success = (
        JobRun.objects.filter(
            job_name=SEC_JOB_NAME,
            region="us",
            target_date=persisted.target_date,
            status=JobRun.Status.SUCCESS,
        )
        .order_by("pk")
        .first()
    )
    sec_required = authoritative_market_gate or sec_success is not None

    expected_stages = {MARKET_STAGE, EVALUATION_STAGE, PORTFOLIO_STAGE}
    if sec_required:
        expected_stages.add(SEC_STAGE)
    if set(stages) != expected_stages:
        raise RefreshVerificationError(
            "recorded_parent_stages_invalid",
            "The recorded scheduled-refresh parent has an unexpected child stage set",
        )

    market = _resolve_stage_run(
        stages,
        MARKET_STAGE,
        job_name=MARKET_JOB_NAME,
        region="us",
        target_date=persisted.target_date,
    )
    _require_success(market, MARKET_STAGE)
    if market.pk != market_success.pk:
        raise RefreshVerificationError(
            "recorded_market_child_mismatch",
            "The recorded parent does not identify the authoritative market child",
        )
    if sec_required:
        sec_run = _resolve_stage_run(
            stages,
            SEC_STAGE,
            job_name=SEC_JOB_NAME,
            region="us",
            target_date=persisted.target_date,
        )
        _require_success(sec_run, SEC_STAGE)
        if sec_success is None or sec_run.pk != sec_success.pk:
            raise RefreshVerificationError(
                "recorded_sec_child_mismatch",
                "The recorded parent does not identify the recoverable SEC child",
            )

    target_text = persisted.target_date.isoformat()
    code_revision = details.get("code_revision")
    if (
        details.get("target_date") != target_text
        or details.get("snapshot_grade") != UniverseSnapshot.Grade.OBSERVED
        or not isinstance(code_revision, str)
        or not code_revision
    ):
        raise RefreshVerificationError(
            "recorded_parent_details_invalid",
            "The recorded scheduled-refresh parent does not match its target and grade",
        )

    data_root = Path(settings.DATA_DIR)
    try:
        if not data_root.is_dir():
            raise OSError
        universe_config = load_us_universe_config(default_us_universe_config_path())
    except (OSError, ValueError):
        raise RefreshVerificationError(
            "production_verification_input_unavailable",
            "The production refresh verification inputs are unavailable or invalid",
        ) from None

    verification = verify_scheduled_refresh(
        target_date=persisted.target_date,
        universe_config=universe_config,
        code_revision=code_revision,
        stages=stages,
        sec_required=sec_required,
    )
    recorded_verification = details.get("verification")
    if not isinstance(recorded_verification, dict) or recorded_verification != verification:
        raise RefreshVerificationError(
            "recorded_verification_mismatch",
            "The recorded verification result does not equal its canonical replay",
        )

    snapshot = UniverseSnapshot.objects.filter(pk=UUID(verification["snapshot_id"])).first()
    analysis_run = AnalysisRun.objects.filter(pk=UUID(verification["analysis_run_id"])).first()
    if snapshot is None or analysis_run is None:
        raise RefreshVerificationError(
            "recorded_verification_output_missing",
            "The replayed scheduled-refresh outputs could not be resolved",
        )
    catalog_assets = tuple(
        resolve_catalog_assets(
            _details_dict(market),
            universe_config=universe_config,
            cutoff=analysis_run.data_cutoff,
        )
    )
    return ReplayedScheduledRefresh(
        parent=persisted,
        verification=verification,
        snapshot=snapshot,
        analysis_run=analysis_run,
        catalog_assets=catalog_assets,
    )


def verify_scheduled_refresh(
    *,
    target_date: date,
    universe_config: UsUniverseConfig,
    code_revision: str,
    stages: dict[str, Any],
    sec_required: bool,
) -> dict[str, Any]:
    """Independently prove the local outputs a successful refresh implies.

    Returns a path-free, JSON-serializable summary suitable for
    ``JobRun.details["verification"]``. Raises :class:`RefreshVerificationError`
    on the first failed check; no partial success-shaped summary is ever
    returned from a failed call.
    """
    market = _resolve_stage_run(
        stages,
        MARKET_STAGE,
        job_name=MARKET_JOB_NAME,
        region="us",
        target_date=target_date,
    )
    _require_success(market, MARKET_STAGE)
    market_details = _details_dict(market)

    snapshot = verify_snapshot(
        market_details,
        universe_config=universe_config,
        target_date=target_date,
    )
    memberships = list(
        UniverseMembership.objects.filter(snapshot=snapshot).select_related(
            "listing__security__company"
        )
    )
    assert_no_etf_in_membership(memberships)
    eligible_memberships = [membership for membership in memberships if membership.eligible]
    excluded_count = len(memberships) - len(eligible_memberships)
    if len(eligible_memberships) != market_details.get("eligible"):
        raise RefreshVerificationError(
            "eligible_count_mismatch",
            "Verified eligible membership count does not match the market stage details",
        )
    if excluded_count != market_details.get("excluded"):
        raise RefreshVerificationError(
            "excluded_count_mismatch",
            "Verified excluded membership count does not match the market stage details",
        )
    if not eligible_memberships:
        raise RefreshVerificationError(
            "no_eligible_memberships",
            "No eligible universe memberships were found for the target date",
        )

    scoring_config = research_config.load_scoring_config(default_us_scoring_config_path())
    scoring_config_hash = research_config.config_hash(scoring_config)
    medium_config = load_medium_forecast_config()
    long_config = load_long_forecast_config()
    medium_config_hash = medium_forecast_config_hash(medium_config)
    long_config_hash = long_forecast_config_hash(long_config)
    long_forecast_requested = market_details.get("long_forecast_requested")
    if not isinstance(long_forecast_requested, bool):
        raise RefreshVerificationError(
            "market_long_forecast_gate_missing",
            "Market stage details do not record a frozen long-forecast invocation gate",
        )
    if long_forecast_requested and not sec_required:
        raise RefreshVerificationError(
            "market_long_forecast_gate_requires_sec",
            "Market stage requested long forecasts but SEC-stage verification was omitted",
        )
    # Replay the exact production lane gates from `analyze_snapshot` using
    # independently reviewed configuration and the market child's frozen
    # run-specific invocation input. Persisted Prediction rows and the output
    # manifest are outputs under verification, while retry-time provider
    # state is mutable, so none may decide which advisory rows were expected.
    medium_lane = (
        MediumLaneExpectation(
            config_hash=medium_config_hash,
            method_version=medium_config.version,
            calendar=medium_config.calendar,
            fixed_epoch=medium_config.fixed_epoch,
            provider=twelve_data.PROVIDER,
            benchmark_subject=universe_config.benchmark_symbol,
            return_basis=medium_config.return_basis,
            dividends_included=medium_config.dividends_included,
            config=medium_config,
        )
        if (
            universe_config.benchmark_symbol
            and scoring_config.version in medium_config.enabled_scoring_versions
            and eligible_memberships
        )
        else None
    )
    long_lane = (
        LongLaneExpectation(
            config_hash=long_config_hash,
            method_version=long_config.version,
        )
        if (
            long_config.price_provider == twelve_data.PROVIDER
            and long_config.fundamentals_provider == sec.PROVIDER
            and scoring_config.version in long_config.enabled_scoring_versions
            and long_forecast_requested
            and eligible_memberships
        )
        else None
    )

    run = _verify_analysis_run(
        market_details,
        snapshot=snapshot,
        target_date=target_date,
        code_revision=code_revision,
        scoring_config_version=scoring_config.version,
        scoring_config_hash=scoring_config_hash,
    )

    catalog_assets = resolve_catalog_assets(
        market_details,
        universe_config=universe_config,
        cutoff=run.data_cutoff,
    )
    membership_evidence_ref = verify_membership_evidence(
        snapshot,
        memberships,
        universe_config=universe_config,
        catalog_assets=catalog_assets,
        cutoff=run.data_cutoff,
    )
    catalog_asset_ids = {asset.id for asset in catalog_assets} | {membership_evidence_ref.id}

    eligible_listing_ids = {membership.listing_id for membership in eligible_memberships}
    listings_by_id = {
        membership.listing_id: membership.listing for membership in eligible_memberships
    }

    stock_rows = list(
        StockAnalysis.objects.filter(run=run).values(
            "id",
            "listing_id",
            "data_quality",
            "current_price",
            "recommendation",
            "overall_score",
            "component_scores",
            "forecast_scenarios",
            "short_scenario",
            "medium_scenario",
            "long_scenario",
        )
    )
    if len(stock_rows) != market_details.get("analyses"):
        raise RefreshVerificationError(
            "stock_analysis_count_mismatch",
            "Verified StockAnalysis count does not match the market stage details",
        )
    stock_listing_ids = {row["listing_id"] for row in stock_rows}
    if stock_listing_ids != eligible_listing_ids:
        raise RefreshVerificationError(
            "stock_analysis_listing_mismatch",
            "Verified StockAnalysis listings do not match the eligible membership set",
        )

    asset_registry: dict[UUID, dict[str, Any]] = {}
    asset_cutoffs: dict[UUID, datetime] = {}
    price_source_asset_id_by_listing: dict[UUID, UUID] = {}
    for analysis_row in stock_rows:
        data_quality = (
            analysis_row["data_quality"] if isinstance(analysis_row["data_quality"], dict) else {}
        )
        context = f"StockAnalysis {analysis_row['id']}"
        entries = list(_iter_source_asset_entries(data_quality, context=context))
        _merge_asset_identity(
            asset_registry, entries, context=context, cutoff=run.data_cutoff, cutoffs=asset_cutoffs
        )
        price_source = data_quality.get("price_source")
        if not isinstance(price_source, dict) or not price_source.get("asset_id"):
            raise RefreshVerificationError(
                "stock_analysis_price_source_missing",
                f"{context} has no recorded decision-time price source",
            )
        try:
            price_asset_id = uuid.UUID(str(price_source["asset_id"]))
        except (TypeError, ValueError) as exc:
            raise RefreshVerificationError(
                "stock_analysis_price_source_malformed",
                f"{context} price source asset id is not a valid identifier",
            ) from exc
        if price_asset_id not in asset_registry:
            raise RefreshVerificationError(
                "stock_analysis_price_source_unlisted",
                f"{context} price source asset is not among its own declared source_assets",
            )
        # The declared subject is proved against its *own* listing's
        # independent, persisted `provider_symbol` -- never against the
        # same JSON payload being validated -- so a cross-listing or SPY
        # substitution for a stock's own price source fails closed rather
        # than tautologically confirming itself.
        listing = listings_by_id[analysis_row["listing_id"]]
        if price_source.get("subject") != listing.provider_symbol:
            raise RefreshVerificationError(
                "stock_analysis_price_source_subject_mismatch",
                f"{context} price source subject does not match its own listing's provider symbol",
            )
        price_source_asset_id_by_listing[analysis_row["listing_id"]] = price_asset_id

    stock_by_id = {row["id"]: row for row in stock_rows}

    decision_horizons = frozenset(
        Prediction.Horizon(value) for value in scoring_config.supported_horizons
    )
    prediction_rows = list(
        Prediction.objects.filter(analysis__run=run).values(
            "id",
            "listing_id",
            "analysis_id",
            "analysis__listing_id",
            "target_date",
            "generated_at",
            "issued_on_time",
            "evidence_grade",
            "evidence_role",
            "horizon",
            "config_hash",
            "method_version",
            "code_revision",
            "source_assets",
            "price_provider",
            "price_subject",
            "source_mode",
            "data_cutoff",
            "price_at_prediction",
            "recommendation",
            "overall_score",
            "component_scores",
            "bear_return",
            "base_return",
            "bull_return",
            "probability_positive",
            "confidence",
            "confidence_status",
            "insufficiency_reason",
        )
    )
    if len(prediction_rows) != market_details.get("predictions"):
        raise RefreshVerificationError(
            "prediction_count_mismatch",
            "Verified Prediction count does not match the market stage details",
        )
    horizons_by_analysis: dict[int, set[str]] = {}
    for prediction_row in prediction_rows:
        _verify_prediction_row(
            prediction_row,
            eligible_listing_ids=eligible_listing_ids,
            listings_by_id=listings_by_id,
            target_date=target_date,
            code_revision=code_revision,
            scoring_config_version=scoring_config.version,
            scoring_config_hash=scoring_config_hash,
            medium_config_hash=medium_config_hash,
            medium_method_version=medium_config.version,
            long_config_hash=long_config_hash,
            long_method_version=long_config.version,
            decision_horizons=decision_horizons,
        )
        _verify_prediction_matches_analysis(
            prediction_row, analysis_row=stock_by_id[prediction_row["analysis_id"]]
        )
        horizons_by_analysis.setdefault(prediction_row["analysis_id"], set()).add(
            prediction_row["horizon"]
        )
        context = f"Prediction {prediction_row['id']}"
        entries = list(_iter_source_asset_entries(prediction_row["source_assets"], context=context))
        _merge_asset_identity(
            asset_registry,
            entries,
            context=context,
            cutoff=prediction_row["data_cutoff"],
            cutoffs=asset_cutoffs,
        )

    # Every eligible analysis must carry *exactly* the independently
    # configured decision/advisory horizon set -- not merely however many
    # prediction rows happen to exist or an aggregate child count, which a
    # same-cardinality horizon swap could satisfy undetected.
    expected_horizon_set = {str(value) for value in decision_horizons}
    if medium_lane is not None:
        expected_horizon_set |= {str(value) for value in _MEDIUM_ADVISORY_HORIZONS}
    if long_lane is not None:
        expected_horizon_set |= {str(value) for value in _LONG_ADVISORY_HORIZONS}
    for analysis_row in stock_rows:
        if horizons_by_analysis.get(analysis_row["id"], set()) != expected_horizon_set:
            raise RefreshVerificationError(
                "stock_analysis_prediction_set_incomplete",
                "A StockAnalysis does not carry exactly the configured set of prediction horizons",
            )

    # Cross-check every declared JSON asset identity against its registered
    # `DataAsset` row *before* trusting any of those fields (e.g. `subject`)
    # below to bind market data or SEC evidence. Also proves every declared
    # source asset's own `available_at`/`retrieved_at` is no later than the
    # strictest cutoff of the rows that claim it -- a future-vintage
    # declared source asset fails here rather than silently authenticating
    # a late signal as on-time.
    _require_registered_asset_identity_matches(asset_registry, asset_cutoffs)

    market_asset_ids: set[UUID] = set()
    for listing_id, listing in listings_by_id.items():
        expected_asset_id = price_source_asset_id_by_listing[listing_id]
        latest, raw_asset = require_bound_market_data(
            listing,
            target_date=target_date,
            cutoff=run.data_cutoff,
            expected_asset_id=expected_asset_id,
            expected_subject=listing.provider_symbol,
        )
        market_asset_ids.add(latest.source_asset_id)
        market_asset_ids.add(raw_asset.id)

    try:
        benchmark_asset = benchmark_asset_for_completed_run(
            run=run,
            benchmark_symbol=universe_config.benchmark_symbol,
            target_date=target_date,
        )
    except ValueError as exc:
        raise RefreshVerificationError("benchmark_asset_identity_invalid", str(exc)) from exc
    benchmark_metadata = (
        benchmark_asset.metadata if isinstance(benchmark_asset.metadata, dict) else {}
    )
    verify_catalog_refs(
        catalog_assets,
        benchmark_metadata.get("catalog_assets"),
        reason_code="catalog_ref_benchmark_mismatch",
    )
    spy_listing = require_spy_listing()
    spy_latest, spy_raw_asset = require_bound_market_data(
        spy_listing,
        target_date=target_date,
        cutoff=run.data_cutoff,
        expected_asset_id=benchmark_asset.id,
        expected_subject=universe_config.benchmark_symbol,
    )
    market_asset_ids.add(spy_latest.source_asset_id)
    market_asset_ids.add(spy_raw_asset.id)

    sec_summary: dict[str, Any] = {"required": sec_required}
    sec_asset_ids: set[UUID] = set()
    if sec_required:
        sec_run = _resolve_stage_run(
            stages,
            SEC_STAGE,
            job_name=SEC_JOB_NAME,
            region="us",
            target_date=target_date,
        )
        _require_success(sec_run, SEC_STAGE)
        sec_details = _details_dict(sec_run)
        sec_result = verify_sec_stage(sec_details, sec_run=sec_run)
        sec_asset_ids.update(ref.id for ref in sec_result.asset_refs)
        sec_summary.update(sec_result.summary)

    evaluation = _resolve_stage_run(
        stages,
        EVALUATION_STAGE,
        job_name=EVALUATION_JOB_NAME,
        region="us",
        target_date=target_date,
    )
    _require_success(evaluation, EVALUATION_STAGE)
    evaluation_summary, evaluation_asset_ids = _verify_evaluation_details(
        evaluation, universe_config=universe_config, target_date=target_date
    )

    portfolio = _resolve_stage_run(
        stages,
        PORTFOLIO_STAGE,
        job_name=PORTFOLIO_JOB_NAME,
        region="",
        target_date=target_date,
    )
    portfolio_summary, portfolio_asset_ids = _verify_portfolio(
        portfolio,
        target_date=target_date,
    )

    # Preserve the established data/C2, outcome, and portfolio verifier
    # ordering (and their public reason codes), then require the complete
    # immutable research output before admitting any asset to the one final
    # exact integrity set.
    analysis_output = verify_analysis_output_manifest(
        run=run,
        eligible_listing_ids=eligible_listing_ids,
        listings_by_id=listings_by_id,
        code_revision=code_revision,
        scoring_config_version=scoring_config.version,
        scoring_config_hash=scoring_config_hash,
        decision_horizons=frozenset(str(value) for value in decision_horizons),
        medium=medium_lane,
        long=long_lane,
        physical_integrity_deferred_to_parent=frozenset(
            market_asset_ids
            | sec_asset_ids
            | set(asset_registry)
            | portfolio_asset_ids
            | catalog_asset_ids
            | evaluation_asset_ids
        ),
    )
    analysis_output_asset_ids = {ref.id for ref in analysis_output.asset_refs}

    asset_ids = (
        market_asset_ids
        | sec_asset_ids
        | set(asset_registry.keys())
        | portfolio_asset_ids
        | catalog_asset_ids
        | evaluation_asset_ids
        | analysis_output_asset_ids
    )
    manifest = _verify_asset_evidence(asset_ids)

    return {
        "status": "verified",
        "target_date": target_date.isoformat(),
        "snapshot_id": str(snapshot.id),
        "snapshot_grade": snapshot.grade,
        "membership_count": len(memberships),
        "eligible_count": len(eligible_memberships),
        "excluded_count": excluded_count,
        "analysis_run_id": str(run.id),
        "stock_analysis_count": len(stock_rows),
        "prediction_count": len(prediction_rows),
        "code_revision": code_revision,
        "sec": sec_summary,
        "evaluation": evaluation_summary,
        "portfolio": portfolio_summary,
        "asset_manifest": manifest,
        "child_job_run_ids": {
            "market": str(market.pk),
            "evaluation": str(evaluation.pk),
            "portfolio_snapshots": str(portfolio.pk),
            "sec_fundamentals": sec_summary.get("job_run_id", ""),
        },
    }


def _details_dict(run: JobRun) -> dict[str, Any]:
    return run.details if isinstance(run.details, dict) else {}


def _require_success(run: JobRun, stage_name: str) -> None:
    if run.status != JobRun.Status.SUCCESS:
        raise RefreshVerificationError(
            "stage_not_success",
            f"{stage_name!r} stage did not resolve to a successful job run",
        )


def _resolve_stage_run(
    stages: dict[str, Any],
    stage_name: str,
    *,
    job_name: str,
    region: str,
    target_date: date,
) -> JobRun:
    """Re-fetch the parent-recorded child run fresh from the database.

    The parent's own ``details["stages"][stage_name]`` bookkeeping is used
    only to name the ``job_run_id`` to re-query; its cached ``status`` and
    ``attempt`` are then compared against that fresh row so a stale or
    tampered in-memory value can never substitute for the persisted job run.
    """
    stage_info = stages.get(stage_name)
    if not isinstance(stage_info, dict):
        raise RefreshVerificationError(
            "stage_details_missing",
            f"Parent recorded no {stage_name!r} stage details",
        )
    raw_id = stage_info.get("job_run_id")
    if not raw_id:
        raise RefreshVerificationError(
            "stage_job_run_id_missing",
            f"Parent {stage_name!r} stage has no recorded job_run_id",
        )
    try:
        job_run_id = uuid.UUID(str(raw_id))
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "stage_job_run_id_malformed",
            f"Parent {stage_name!r} stage job_run_id is not a valid identifier",
        ) from exc
    candidate = JobRun.objects.filter(pk=job_run_id).first()
    if candidate is None:
        raise RefreshVerificationError(
            "stage_job_run_missing",
            f"Parent {stage_name!r} stage references a job run that no longer exists",
        )
    if stage_info.get("status") != candidate.status:
        raise RefreshVerificationError(
            "stage_status_mismatch",
            f"Parent {stage_name!r} stage recorded status does not match its referenced job run",
        )
    if stage_info.get("attempt") != candidate.attempt:
        raise RefreshVerificationError(
            "stage_attempt_mismatch",
            f"Parent {stage_name!r} stage recorded attempt does not match its referenced job run",
        )
    return _resolve_authoritative_run(
        candidate,
        stage_name=stage_name,
        job_name=job_name,
        region=region,
        target_date=target_date,
    )


def _resolve_authoritative_run(
    candidate: JobRun,
    *,
    stage_name: str,
    job_name: str,
    region: str,
    target_date: date,
) -> JobRun:
    _require_identity(
        candidate,
        stage_name=stage_name,
        job_name=job_name,
        region=region,
        target_date=target_date,
    )
    if candidate.status == JobRun.Status.SUCCESS:
        return candidate
    if candidate.status != JobRun.Status.SKIPPED:
        raise RefreshVerificationError(
            "stage_status_unsatisfied",
            f"{stage_name!r} stage ended with unsupported status {candidate.status!r}",
        )
    details = _details_dict(candidate)
    raw_reference = details.get("successful_run_id")
    if raw_reference is None:
        # A self-contained satisfied skip (e.g. zero active portfolios): the
        # candidate's own details are the evidence, not a reference elsewhere.
        return candidate
    try:
        reference_id = uuid.UUID(str(raw_reference))
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "chained_skip_reference_malformed",
            f"{stage_name!r} stage skip reference is not a valid identifier",
        ) from exc
    referenced = JobRun.objects.filter(pk=reference_id).first()
    if referenced is None:
        raise RefreshVerificationError(
            "chained_skip_reference_missing",
            f"{stage_name!r} stage skip references a job run that no longer exists",
        )
    _require_identity(
        referenced,
        stage_name=stage_name,
        job_name=job_name,
        region=region,
        target_date=target_date,
    )
    if referenced.status != JobRun.Status.SUCCESS:
        raise RefreshVerificationError(
            "chained_skip_reference_not_success",
            f"{stage_name!r} stage skip does not resolve to exactly one successful job run",
        )
    return referenced


def _require_identity(
    run: JobRun,
    *,
    stage_name: str,
    job_name: str,
    region: str,
    target_date: date,
) -> None:
    if run.job_name != job_name or run.region != region or run.target_date != target_date:
        raise RefreshVerificationError(
            "stage_identity_mismatch",
            f"{stage_name!r} stage job run identity does not match the scheduled target",
        )


def _verify_analysis_run(
    market_details: dict[str, Any],
    *,
    snapshot: UniverseSnapshot,
    target_date: date,
    code_revision: str,
    scoring_config_version: str,
    scoring_config_hash: str,
) -> AnalysisRun:
    raw_id = market_details.get("analysis_run_id")
    if not raw_id:
        raise RefreshVerificationError(
            "analysis_run_identity_missing", "Market stage details have no analysis_run_id"
        )
    try:
        run_id = uuid.UUID(str(raw_id))
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "analysis_run_identity_malformed",
            "Market stage analysis_run_id is not a valid identifier",
        ) from exc
    run = AnalysisRun.objects.filter(pk=run_id).first()
    if run is None:
        raise RefreshVerificationError(
            "analysis_run_missing", "Market stage analysis run could not be resolved"
        )
    if run.universe_snapshot_id != snapshot.id:
        raise RefreshVerificationError(
            "analysis_run_snapshot_mismatch",
            "Resolved analysis run does not reference the verified universe snapshot",
        )
    if run.target_date != target_date:
        raise RefreshVerificationError(
            "analysis_run_target_mismatch", "Resolved analysis run targets a different date"
        )
    if run.status != "complete":
        raise RefreshVerificationError(
            "analysis_run_not_complete", "Resolved analysis run is not complete"
        )
    # `run.issued_on_time` is a stored flag and cannot authenticate itself:
    # independently recompute it from the run's own immutable
    # `generated_at`/`target_date` using the same XNYS timing owner
    # production issuance uses, rather than trusting the persisted bit.
    if not run.issued_on_time or not is_us_prediction_on_time(
        target_date=target_date, generated_at=run.generated_at
    ):
        raise RefreshVerificationError(
            "analysis_run_not_on_time", "Resolved analysis run was not issued on time"
        )
    if run.code_revision != code_revision:
        raise RefreshVerificationError(
            "analysis_run_code_revision_mismatch",
            "Resolved analysis run code revision does not match the scheduled refresh revision",
        )
    if run.config_version != scoring_config_version or run.config_hash != scoring_config_hash:
        raise RefreshVerificationError(
            "analysis_run_config_mismatch",
            "Resolved analysis run scoring configuration does not match the reviewed configuration",
        )
    return run


def _verify_prediction_row(
    row: Mapping[str, Any],
    *,
    eligible_listing_ids: set[UUID],
    listings_by_id: Mapping[UUID, Listing],
    target_date: date,
    code_revision: str,
    scoring_config_version: str,
    scoring_config_hash: str,
    medium_config_hash: str,
    medium_method_version: str,
    long_config_hash: str,
    long_method_version: str,
    decision_horizons: frozenset[str],
) -> None:
    """Prove one `Prediction` row belongs to this verified run, not a transplant.

    Rejects an ETF/wrong-listing or analysis/listing-mismatched row (the
    proved rogue-SPY case), a wrong-target row, a non-observed/late row, a
    code-revision mismatch, a synthetic-provider clone, and a horizon/role
    combination or config_hash binding that decision/advisory issuance never
    produces. Advisory predictions are deliberately checked against their
    own medium/long forecast configuration digest, never the main scoring
    digest. Decision horizons are the loaded scoring config's own
    `supported_horizons`, never a hard-coded set, so a reviewed
    configuration requesting more than `short` is not incorrectly rejected.
    """
    listing_id = row["listing_id"]
    if listing_id != row["analysis__listing_id"]:
        raise RefreshVerificationError(
            "prediction_listing_transplanted",
            "A prediction's listing does not match its own StockAnalysis listing",
        )
    if listing_id not in eligible_listing_ids:
        raise RefreshVerificationError(
            "prediction_listing_ineligible",
            "A prediction references a listing outside the verified eligible universe membership",
        )
    if row["target_date"] != target_date:
        raise RefreshVerificationError(
            "prediction_target_mismatch", "A prediction targets a different date"
        )
    # Independently recompute on-time status from the prediction's own
    # immutable `generated_at`/`target_date` (the stored flag cannot
    # authenticate itself); a prediction may legitimately be reissued at a
    # later `generated_at` than its analysis run only when marked
    # off-time by production issuance (`append_predictions`), so this must
    # match its own timing, not the parent run's.
    if not row["issued_on_time"] or not is_us_prediction_on_time(
        target_date=target_date, generated_at=row["generated_at"]
    ):
        raise RefreshVerificationError(
            "prediction_not_on_time", "A prediction was not issued on time"
        )
    if row["evidence_grade"] != UniverseSnapshot.Grade.OBSERVED:
        raise RefreshVerificationError(
            "prediction_not_observed", "A prediction is not observed-grade"
        )
    if row["code_revision"] != code_revision:
        raise RefreshVerificationError(
            "prediction_code_revision_mismatch",
            "A prediction's code revision does not match the scheduled refresh revision",
        )
    # A synthetic-provider clone must fail: `source_mode` is always derived
    # from `source_assets` in genuine production issuance
    # (`research.service`'s `source_data_mode`), never independently
    # settable, and scheduled-refresh evidence is always provider-backed.
    recomputed_mode = source_data_mode({"source_assets": row["source_assets"]})
    if recomputed_mode != row["source_mode"] or recomputed_mode != DATA_MODE_PROVIDER:
        raise RefreshVerificationError(
            "prediction_source_mode_invalid",
            "A prediction's source_mode is not a genuine provider-backed derivation "
            "of its own declared source_assets",
        )
    if row["price_provider"] != twelve_data.PROVIDER:
        raise RefreshVerificationError(
            "prediction_price_provider_mismatch",
            "A prediction's price provider does not match Twelve Data",
        )
    listing = listings_by_id.get(listing_id)
    if listing is None or row["price_subject"] != listing.provider_symbol:
        raise RefreshVerificationError(
            "prediction_price_subject_mismatch",
            "A prediction's price subject does not match its own listing's provider symbol",
        )
    horizon = row["horizon"]
    role = row["evidence_role"]
    if role == Prediction.EvidenceRole.DECISION:
        if horizon not in decision_horizons:
            raise RefreshVerificationError(
                "prediction_horizon_role_mismatch",
                "A decision prediction carries a horizon its role does not support",
            )
        if (
            row["config_hash"] != scoring_config_hash
            or row["method_version"] != scoring_config_version
        ):
            raise RefreshVerificationError(
                "prediction_decision_config_mismatch",
                "A decision prediction's configuration does not match the reviewed "
                "scoring configuration",
            )
    elif role == Prediction.EvidenceRole.ADVISORY:
        if horizon not in _ADVISORY_HORIZONS:
            raise RefreshVerificationError(
                "prediction_horizon_role_mismatch",
                "An advisory prediction carries a horizon its role does not support",
            )
        is_medium = horizon in _MEDIUM_ADVISORY_HORIZONS
        expected_hash = medium_config_hash if is_medium else long_config_hash
        expected_method_version = medium_method_version if is_medium else long_method_version
        if row["config_hash"] != expected_hash:
            raise RefreshVerificationError(
                "prediction_advisory_config_mismatch",
                "An advisory prediction's configuration does not match its own reviewed "
                "medium/long forecast configuration",
            )
        if row["method_version"] != expected_method_version:
            raise RefreshVerificationError(
                "prediction_advisory_method_version_mismatch",
                "An advisory prediction's method_version does not match its own reviewed "
                "medium/long forecast configuration version",
            )
    else:
        raise RefreshVerificationError(
            "prediction_evidence_role_invalid", "A prediction carries an unsupported evidence role"
        )


def _verify_prediction_matches_analysis(
    row: Mapping[str, Any],
    *,
    analysis_row: Mapping[str, Any],
) -> None:
    """Bind a mutable `StockAnalysis`'s duplicated fields to the immutable
    `Prediction` ledger that carried them at issuance time.

    `research.service` always copies `current_price`/`recommendation`/
    `overall_score`/`component_scores` verbatim onto every prediction it
    issues for an analysis, and rounds each horizon's own scenario
    (`bear`/`base`/`bull`/`probability_positive`/`confidence`) with the same
    `_decimal`/`_optional_decimal` helpers used to persist the analysis's
    own `forecast_scenarios` document -- reused here rather than
    re-implemented, so an independent mutation of any of these mutable
    `StockAnalysis` fields (with predictions/assets left untouched) fails
    closed instead of silently reporting verified.
    """
    context = f"StockAnalysis {analysis_row['id']}"
    if row["price_at_prediction"] != analysis_row["current_price"]:
        raise RefreshVerificationError(
            "stock_analysis_current_price_mismatch",
            f"{context} current_price does not match its own immutable prediction ledger",
        )
    if row["recommendation"] != analysis_row["recommendation"]:
        raise RefreshVerificationError(
            "stock_analysis_recommendation_mismatch",
            f"{context} recommendation does not match its own immutable prediction ledger",
        )
    if row["overall_score"] != analysis_row["overall_score"]:
        raise RefreshVerificationError(
            "stock_analysis_overall_score_mismatch",
            f"{context} overall_score does not match its own immutable prediction ledger",
        )
    if row["component_scores"] != analysis_row["component_scores"]:
        raise RefreshVerificationError(
            "stock_analysis_component_scores_mismatch",
            f"{context} component_scores does not match its own immutable prediction ledger",
        )
    scenario = scenario_from_document(
        analysis_row["forecast_scenarios"],
        row["horizon"],
        legacy_fallbacks={
            "short": analysis_row["short_scenario"],
            "medium": analysis_row["medium_scenario"],
            "long": analysis_row["long_scenario"],
        },
    )
    expected = {
        "bear_return": _optional_decimal(scenario.get("bear"), places=4),
        "base_return": _optional_decimal(scenario.get("base"), places=4),
        "bull_return": _optional_decimal(scenario.get("bull"), places=4),
        "probability_positive": _optional_decimal(scenario.get("probability_positive"), places=4),
        "confidence": _decimal(scenario.get("confidence", 0.0), places=2),
    }
    for field, expected_value in expected.items():
        if row[field] != expected_value:
            raise RefreshVerificationError(
                "stock_analysis_scenario_mismatch",
                f"{context} scenario value for {field!r} at horizon {row['horizon']!r} does "
                "not match its own immutable prediction ledger",
            )
    if row["confidence_status"] != scenario.get("confidence_status"):
        raise RefreshVerificationError(
            "stock_analysis_scenario_mismatch",
            f"{context} confidence_status at horizon {row['horizon']!r} does not match its "
            "own immutable prediction ledger",
        )
    if row["insufficiency_reason"] != scenario.get("insufficiency_reason", ""):
        raise RefreshVerificationError(
            "stock_analysis_scenario_mismatch",
            f"{context} insufficiency_reason at horizon {row['horizon']!r} does not match "
            "its own immutable prediction ledger",
        )


def _verify_evaluation_details(
    evaluation: JobRun,
    *,
    universe_config: UsUniverseConfig,
    target_date: date,
) -> tuple[dict[str, Any], set[UUID]]:
    details = _details_dict(evaluation)
    required = {
        "provider",
        "benchmark_subject",
        "eligible_predictions",
        "actions",
        "outcome_statuses",
        "evaluated_prediction_ids",
        "evaluation_time",
    }
    if required - details.keys():
        raise RefreshVerificationError(
            "evaluation_details_incomplete",
            "Evaluation stage details are missing required fields",
        )
    if details["provider"] != twelve_data.PROVIDER:
        raise RefreshVerificationError(
            "evaluation_provider_mismatch",
            "Evaluation stage provider does not match Twelve Data",
        )
    if details["benchmark_subject"] != universe_config.benchmark_symbol:
        raise RefreshVerificationError(
            "evaluation_benchmark_mismatch",
            "Evaluation stage benchmark does not match the reviewed benchmark",
        )
    try:
        evaluation_time = datetime.fromisoformat(str(details["evaluation_time"]))
    except ValueError as exc:
        raise RefreshVerificationError(
            "evaluation_time_malformed", "Evaluation stage evaluation_time is not a valid timestamp"
        ) from exc
    # `evaluation_time` anchors the pre-child candidate-set cutoff below
    # (`maturity_provider_candidates(..., as_of=evaluation_time)`) and every
    # per-outcome "touched now" comparison. A naive value cannot be safely
    # compared against the timezone-aware `generated_at`/`evaluated_at`
    # fields it is checked against (Django raises on naive/aware mixing),
    # so it is rejected explicitly here rather than surfacing as an
    # unrelated comparison error deeper in this function. This is
    # intentionally not bound to the `JobRun`'s own `started_at`/
    # `finished_at` wall-clock interval: a legitimate missed-day/
    # lagging-series catch-up execution may be given an explicit,
    # deliberately earlier reconstruction `evaluation_time` distinct from
    # the real time the job process happened to run, and requiring
    # containment would falsely reject that documented, accepted case.
    if evaluation_time.tzinfo is None or evaluation_time.tzinfo.utcoffset(evaluation_time) is None:
        raise RefreshVerificationError(
            "evaluation_time_naive",
            "Evaluation stage evaluation_time must be a timezone-aware timestamp",
        )
    eligible_predictions = details["eligible_predictions"]
    if not isinstance(eligible_predictions, int) or eligible_predictions < 0:
        raise RefreshVerificationError(
            "evaluation_details_invalid", "Evaluation stage eligible_predictions is invalid"
        )
    actions = details["actions"]
    outcome_statuses = details["outcome_statuses"]
    if not isinstance(actions, dict) or not isinstance(outcome_statuses, dict):
        raise RefreshVerificationError(
            "evaluation_details_invalid",
            "Evaluation stage action/outcome details are invalid",
        )
    raw_ids = details["evaluated_prediction_ids"]
    if not isinstance(raw_ids, list):
        raise RefreshVerificationError(
            "evaluation_details_invalid",
            "Evaluation stage evaluated_prediction_ids is not a list",
        )
    try:
        prediction_ids = [uuid.UUID(str(value)) for value in raw_ids]
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "evaluation_prediction_id_malformed",
            "Evaluation stage evaluated_prediction_ids contains an invalid identifier",
        ) from exc
    if len(set(prediction_ids)) != len(prediction_ids):
        raise RefreshVerificationError(
            "evaluation_prediction_id_duplicated",
            "Evaluation stage evaluated_prediction_ids contains a duplicate identifier",
        )
    if len(prediction_ids) != eligible_predictions:
        raise RefreshVerificationError(
            "evaluation_count_mismatch",
            "Evaluation stage evaluated_prediction_ids does not match eligible_predictions",
        )
    # `actions` (created/updated/skipped) describes an ephemeral, process-level
    # event that cannot be replayed from persisted state alone (a
    # `PredictionOutcome` existing now does not reveal whether *this* job run
    # created or updated it). It is therefore only checked structurally; the
    # durable `outcome_statuses` claim is independently re-derived below.
    if not set(actions.keys()) <= EVALUATION_ACTIONS:
        raise RefreshVerificationError(
            "evaluation_action_invalid", "Evaluation stage actions has an unsupported key"
        )
    if not all(isinstance(value, int) and value >= 0 for value in actions.values()):
        raise RefreshVerificationError(
            "evaluation_action_invalid", "Evaluation stage action counts are invalid"
        )
    if sum(actions.values()) != len(prediction_ids):
        raise RefreshVerificationError(
            "evaluation_action_count_mismatch",
            "Evaluation stage action counts do not match evaluated_prediction_ids",
        )

    # Re-derive the exact pre-evaluation candidate set instead of trusting
    # the child's self-reported list: reuses `maturity_provider_candidates`
    # (the same maturity+provider selection `eligible_pending_predictions`
    # is built from) so a fabricated `eligible_predictions=0` cannot hide a
    # matured, provider-matching prediction that was never evaluated.
    candidate_ids = {
        candidate.pk
        for candidate in maturity_provider_candidates(
            provider=details["provider"], evaluation_date=target_date, as_of=evaluation_time
        )
    }
    pre_existing_terminal_ids = set(
        PredictionOutcome.objects.filter(
            prediction_id__in=candidate_ids,
            status__in=TERMINAL_OUTCOME_STATUSES,
            evaluated_at__lt=evaluation_time,
        ).values_list("prediction_id", flat=True)
    )
    expected_ids = candidate_ids - pre_existing_terminal_ids
    if set(prediction_ids) != expected_ids:
        raise RefreshVerificationError(
            "evaluation_candidate_set_mismatch",
            "Evaluated predictions do not match the independently re-derived candidate set",
        )

    outcome_rows = list(PredictionOutcome.objects.filter(prediction_id__in=prediction_ids))
    if len(outcome_rows) != len(prediction_ids):
        raise RefreshVerificationError(
            "evaluation_outcome_missing",
            "One or more evaluated predictions have no persisted outcome",
        )
    # Full model instances (not `.values()`): `verify_prediction_outcome`
    # calls `resolve_outcome`, which needs the same fields
    # `evaluate_prediction` itself reads (`.listing.provider_symbol`,
    # `.bear_return`/`.base_return`/`.bull_return`, etc.), not a fixed
    # projection.
    predictions = {
        prediction.pk: prediction
        for prediction in Prediction.objects.select_related("listing").filter(pk__in=prediction_ids)
    }
    recomputed_statuses: dict[str, int] = {}
    evaluation_asset_ids: set[UUID] = set()
    frame_cache: dict[tuple[str, str, date], pl.DataFrame] = {}
    for outcome in outcome_rows:
        result = verify_prediction_outcome(
            predictions[outcome.prediction_id],
            outcome,
            provider=details["provider"],
            benchmark_subject=details["benchmark_subject"],
            evaluation_time=evaluation_time,
            parent_target_date=target_date,
            frame_cache=frame_cache,
        )
        evaluation_asset_ids.update(ref.id for ref in result.asset_refs)
        recomputed_statuses[outcome.status] = recomputed_statuses.get(outcome.status, 0) + 1
    if recomputed_statuses != outcome_statuses:
        raise RefreshVerificationError(
            "evaluation_outcome_statuses_mismatch",
            "Recorded outcome status counts do not match the persisted PredictionOutcome evidence",
        )
    return {
        "job_run_id": str(evaluation.pk),
        "eligible_predictions": eligible_predictions,
        "evaluated_prediction_count": len(prediction_ids),
    }, evaluation_asset_ids


def _verify_portfolio(
    portfolio: JobRun,
    *,
    target_date: date,
) -> tuple[dict[str, Any], set[UUID]]:
    result = verify_portfolio_snapshot_stage(
        portfolio,
        target_date=target_date,
    )
    return result.summary, {ref.id for ref in result.asset_refs}


def _iter_source_asset_entries(raw: object, *, context: str) -> Iterable[dict[str, Any]]:
    """Fail-closed replacement for silently dropping malformed asset entries.

    A malformed entry, a missing required identity field, or an id that is
    not a valid UUID now raises rather than being silently skipped -- a
    payload that lies by omission can no longer pass verification simply
    because the omitted entry was ignored.
    """
    if isinstance(raw, dict):
        payload: object = raw.get("source_assets")
    else:
        payload = raw
    if payload is None:
        return
    if not isinstance(payload, list):
        raise RefreshVerificationError(
            "source_assets_payload_invalid", f"{context} source_assets is not a list"
        )
    for entry in payload:
        if not isinstance(entry, dict):
            raise RefreshVerificationError(
                "source_assets_entry_invalid", f"{context} source_assets entry is not an object"
            )
        if any(not entry.get(field) for field in _SOURCE_ASSET_FIELDS):
            raise RefreshVerificationError(
                "source_assets_entry_incomplete",
                f"{context} source_assets entry is missing a required identity field",
            )
        try:
            asset_id = uuid.UUID(str(entry["id"]))
        except (TypeError, ValueError) as exc:
            raise RefreshVerificationError(
                "source_assets_entry_malformed",
                f"{context} source_assets entry id is not a valid identifier",
            ) from exc
        sha256 = str(entry["sha256"])
        if len(sha256) != 64:
            raise RefreshVerificationError(
                "source_assets_entry_malformed",
                f"{context} source_assets entry checksum is not well-formed",
            )
        yield {
            "id": asset_id,
            "provider": str(entry["provider"]),
            "kind": str(entry["kind"]),
            "subject": str(entry["subject"]),
            "sha256": sha256,
        }


def _merge_asset_identity(
    registry: dict[UUID, dict[str, Any]],
    entries: Iterable[dict[str, Any]],
    *,
    context: str,
    cutoff: datetime,
    cutoffs: dict[UUID, datetime],
) -> None:
    """Reject a valid-but-unrelated substitution: same id, different claimed identity.

    ``cutoffs`` tracks the strictest (earliest) cutoff any row has claimed
    for a given asset id: if the same physical asset is legitimately
    referenced by two rows with different cutoffs, it only needs to be
    admissible by the earliest of the two for both claims to be true.
    """
    for entry in entries:
        asset_id = entry["id"]
        existing = registry.get(asset_id)
        if existing is not None and existing != entry:
            raise RefreshVerificationError(
                "source_assets_entry_conflicting",
                f"{context} source_assets entry conflicts with an already-declared "
                "identity for the same asset id",
            )
        registry[asset_id] = entry
        current_cutoff = cutoffs.get(asset_id)
        cutoffs[asset_id] = cutoff if current_cutoff is None else min(current_cutoff, cutoff)


def _require_registered_asset_identity_matches(
    registry: dict[UUID, dict[str, Any]], cutoffs: dict[UUID, datetime]
) -> None:
    """Cross-check every declared JSON asset identity against its registered row.

    Closes the gap where a payload's `id` refers to a real, checksum-valid
    `DataAsset` row that nonetheless is not what the payload claims it is
    (wrong provider/kind/subject/sha256) -- a valid-but-unrelated asset
    could otherwise be substituted undetected. Also proves each asset's own
    `available_at`/`retrieved_at` is no later than the strictest cutoff of
    the rows that declared it: a future-vintage source asset fails here
    rather than silently authenticating a late signal as on-time.
    """
    if not registry:
        return
    rows = {
        row["id"]: row
        for row in DataAsset.objects.filter(pk__in=registry.keys()).values(
            "id", "provider", "kind", "subject", "sha256", "available_at", "retrieved_at"
        )
    }
    if len(rows) != len(registry):
        raise RefreshVerificationError(
            "source_assets_reference_missing",
            "One or more declared source assets could not be resolved to a registered "
            "DataAsset row",
        )
    for asset_id, entry in registry.items():
        row = rows[asset_id]
        if (
            row["provider"] != entry["provider"]
            or row["kind"] != entry["kind"]
            or row["subject"] != entry["subject"]
            or row["sha256"] != entry["sha256"]
        ):
            raise RefreshVerificationError(
                "source_assets_identity_mismatch",
                "A declared source asset's payload identity does not match its "
                "registered DataAsset row",
            )
        cutoff = cutoffs[asset_id]
        if row["available_at"] > cutoff or row["retrieved_at"] > cutoff:
            raise RefreshVerificationError(
                "source_assets_entry_after_cutoff",
                "A declared source asset was admitted after the cutoff of the row that claims it",
            )


def _verify_asset_evidence(asset_ids: set[UUID]) -> dict[str, Any]:
    """Prove disk existence/checksum for exactly the assets this run's evidence names.

    Deliberately scoped to ``asset_ids`` -- never the whole `DataAsset`
    table -- so unrelated historical assets can never block a target
    refresh. The returned manifest is path-free: only a count and a
    canonical hash over sorted ``(asset_id, sha256)`` pairs, letting a
    retry/no-op prove it reused the exact same evidence set without ever
    naming a `relative_path` or resolved filesystem path.
    """
    queryset = DataAsset.objects.filter(pk__in=asset_ids).order_by("id")
    pairs = list(queryset.values_list("id", "sha256"))
    if len(pairs) != len(asset_ids):
        raise RefreshVerificationError(
            "asset_reference_missing",
            "One or more evidence-referenced assets could not be resolved to a "
            "registered DataAsset row",
        )
    report = verify_registered_assets(assets=queryset)
    if not report.ok:
        raise RefreshVerificationError(
            "asset_integrity_failed",
            f"{len(report.failures)} registered asset(s) referenced by this target "
            "failed integrity verification",
        )
    ordered = sorted((str(asset_id), sha256) for asset_id, sha256 in pairs)
    canonical = "\n".join(f"{asset_id}:{sha256}" for asset_id, sha256 in ordered)
    manifest_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"count": report.checked, "hash": manifest_hash}

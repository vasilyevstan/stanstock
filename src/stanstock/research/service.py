from __future__ import annotations

import hashlib
import io
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from decimal import Decimal
from functools import lru_cache
from importlib.metadata import version as package_version
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from uuid import UUID

import polars as pl
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from stanstock.core.revision import clean_git_revision
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.asof import AsOfData, PriceFrameChecksumMismatchError
from stanstock.data.assets import (
    AssetStore,
    open_asset_store,
    read_checksummed_bytes,
)
from stanstock.data.etfs import INVESTABLE_US_ETF_SYMBOL
from stanstock.data.models import (
    DataAsset,
    FundamentalFact,
    Listing,
    ProviderRecord,
    Region,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.data.provider_policy import (
    PRIVATE_USAGE_SCOPE,
    TWELVE_DATA_PROVIDER,
    normalized_provider_plan,
)
from stanstock.data.sec_config import SecFundamentalsConfig, load_sec_fundamentals_config
from stanstock.research.affordability import (
    DECISION_TARGET_DATE_BASIS,
    UNDER_10_BAND,
    classify_price_band,
)
from stanstock.research.config import (
    V3_EFFECTIVE_CONFIG_HASH,
    V3_VERSION,
    ScoringConfig,
    code_revision,
    config_hash,
    load_scoring_config,
)
from stanstock.research.eligibility import require_stock_research_listing
from stanstock.research.explanations import generate_reasons, generate_risks
from stanstock.research.forecast_config import (
    MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
    MEDIUM_V2_VERSION,
    MediumForecastConfig,
    MediumForecastV2Config,
    _decode_medium_forecast_config,
    _is_medium_v2_candidate,
    _load_medium_forecast_config_bytes,
    _read_medium_forecast_config_bytes,
    load_medium_forecast_config,
    medium_forecast_config_hash,
)
from stanstock.research.forecasting import (
    build_forecast_scenario_document,
    infer_price_source,
)
from stanstock.research.fundamentals import calculate_fundamentals, inputs_from_facts
from stanstock.research.indicators import calculate_indicators
from stanstock.research.long_forecast_config import (
    LongForecastConfig,
    load_long_forecast_config,
    long_forecast_config_hash,
)
from stanstock.research.long_forecasts import LongForecast, build_long_forecasts
from stanstock.research.medium_forecasts import (
    PANEL_SCHEMA,
    MediumForecast,
    MediumPanel,
    MediumPanelPriceInput,
    asset_identity,
    build_medium_forecast_panel,
    build_medium_forecasts,
    calendar_sessions_through,
    hash_json,
    reconstruct_medium_forecast_panel,
    serialize_medium_forecast_panel,
)
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.provenance import source_data_mode
from stanstock.research.refresh_evidence import (
    ANALYSIS_OUTPUT_MANIFEST_KIND,
    ANALYSIS_RUN_FIELDS,
    ANALYSIS_RUN_MODEL,
    PREDICTION_FIELDS,
    PREDICTION_MODEL,
    STOCK_ANALYSIS_FIELDS,
    STOCK_ANALYSIS_MODEL,
    ManifestEntry,
    ManifestPayloadError,
    ManifestPlan,
    actual_output_plan,
    build_manifest_envelope,
    build_output_plan,
    decimal_from_float,
    dumps_canonical_envelope,
    model_row_values,
    optional_decimal_from_float,
    row_digest,
)
from stanstock.research.scenarios import build_scenarios
from stanstock.research.scoring import (
    aggregate_score,
    assess_risk,
    decide_recommendation,
    score_components,
)
from stanstock.research.timing import is_observed_issuance_on_time
from stanstock.research.types import AggregateScore, IndicatorResult, ResearchValues, Scenario
from stanstock.research.under10 import (
    UNDER10_CONCEPTS,
    build_under10_assessment,
    canonical_json,
    qualify_under10_sec_facts,
)

#: `data_quality` key carrying the unactivated Under-$10 shadow assessment.
#: It is written only when a *new* analysis qualifies; an absent key means
#: "not assessed", never "assessed and failed". Nothing backfills it.
UNDER10_ASSESSMENT_KEY = "under10_assessment"
_FULL_LOWERHEX_GIT_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_MEDIUM_V2_SCORING_HASH = "43cc0ee0e29f79dec4ad8d8a91e43e7b3df1d368dbc385a116fcc6ff05f18e9b"
_MEDIUM_V2_PAYLOAD_ERROR = "us-price-medium-v2 prediction payload is malformed"
_MEDIUM_V2_SERVICE_KEYS = frozenset(
    {
        "config_hash",
        "prediction_version",
        "panel_asset_id",
        "panel_sha256",
        "evidence_grade",
        "price_subject",
    }
)
_MEDIUM_V2_CALCULATOR_KEYS = frozenset(
    {
        "schema_version",
        "method",
        "method_version",
        "forecast_horizon",
        "horizon_sessions",
        "current_state",
        "support",
        "probability_evidence",
        "predictive_distribution",
        "evidence",
        "formula_inputs",
        "return_basis",
        "dividends_included",
        "training_evidence",
    }
)


@dataclass(frozen=True, slots=True)
class AnalysisComputation:
    indicators: IndicatorResult
    fundamentals: ResearchValues
    aggregate: AggregateScore
    scenarios: dict[str, Scenario]
    risk_score: float | None
    risk_class: str
    recommendation: str
    reasons: list[str]
    risks: list[str]
    data_quality: dict[str, Any]
    source_assets: list[dict[str, Any]]
    current_price: float
    daily_change: float | None
    price_asset: DataAsset | None = None


@dataclass(frozen=True, slots=True)
class PersistedAnalysis:
    run: AnalysisRun
    analysis: StockAnalysis
    predictions: tuple[Prediction, ...]
    computation: AnalysisComputation


@dataclass(frozen=True, slots=True)
class _MediumV2PanelAttestation:
    """Internal result of one complete source-to-panel reconstruction.

    This is a transaction-local replay result, not a caller authorization
    token. Writer boundaries never accept it, and its recursively immutable
    calculations cannot be altered after the source-to-panel replay.
    """

    panel_id: UUID
    panel_sha256: str
    run_id: UUID
    config_hash: str
    snapshot_id: UUID
    source_manifest_hash: str
    price_provider: str
    calculations: Mapping[str, Mapping[str, object]]


@dataclass(slots=True)
class AnalysisOutputPaths:
    """Optional mutable out-parameter for `analyze_snapshot`.

    A caller that needs its own later cleanup after `analyze_snapshot`
    itself has already returned successfully (for example, an outer
    transaction that still has to run its own post-analysis checks before
    it can commit) passes one instance in and reads the two fields back
    directly once the call returns -- the exact paths this call itself
    wrote, tracked from the producer's own local state as they are set,
    never re-derived by a separate, fallible post-write `DataAsset` query
    keyed on `run_id`/`kind`/`provider` that could itself fail independent
    of whether the paths actually exist, or -- for a research-grade run
    that legitimately writes neither -- silently return nothing to clean
    up.
    """

    panel_relative_path: str | None = None
    manifest_relative_path: str | None = None


@dataclass(frozen=True, slots=True)
class AdvisoryForecastContext:
    panel: MediumPanel
    store: AssetStore
    config: MediumForecastConfig | MediumForecastV2Config
    config_hash: str
    model_version: str
    forecasts: dict[str, dict[str, MediumForecast]]


@dataclass(frozen=True, slots=True)
class _AdditionalAdvisoryPredictionRequest:
    """One extra listing included in a single panel-level append."""

    analysis: StockAnalysis
    forecasts: dict[str, MediumForecast]
    source_assets: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _LockedMediumV2Authority:
    """Transaction-local authoritative parents used by panel attestation."""

    run: AnalysisRun | None
    snapshot: UniverseSnapshot
    memberships: tuple[UniverseMembership, ...]
    listings: tuple[Listing, ...]


@dataclass(frozen=True, slots=True)
class _LockedAdvisoryAuthority:
    """One advisory invocation's locked analyses and parent closures."""

    analyses: tuple[StockAnalysis, ...]
    panels_by_run: Mapping[UUID, _LockedMediumV2Authority]


def _lock_advisory_parent_rows(
    *,
    snapshot_ids: Sequence[UUID] = (),
    run_ids: Sequence[UUID] = (),
    requested_listing_ids: Sequence[UUID] = (),
) -> tuple[
    dict[str, Universe],
    dict[UUID, UniverseSnapshot],
    tuple[UniverseMembership, ...],
    dict[UUID, Listing],
    dict[UUID, Security],
    dict[UUID, AnalysisRun],
]:
    """Discover, lock, and reverify mutable v2 parents in canonical order.

    Only scalar projections are read before locking, and every projected
    value is compared with the row after its lock is acquired. The full lock
    order is Universe -> UniverseSnapshot -> UniverseMembership -> Listing ->
    Security -> AnalysisRun. StockAnalysis rows, when applicable, are locked
    by ``_lock_advisory_authority`` immediately afterward.

    The snapshot receives a full ``FOR UPDATE`` lock before membership rows
    are read. On PostgreSQL that conflicts with the ``KEY SHARE`` check a new
    membership's foreign key needs, closing the insertion gap until the
    surrounding transaction finishes. Existing memberships are all locked,
    including currently ineligible rows. Company rows are deliberately never
    selected or locked.
    """
    ordered_run_ids = tuple(sorted(set(run_ids), key=str))
    run_fields = (
        "generated_at",
        "data_cutoff",
        "target_date",
        "issued_on_time",
        "universe_snapshot_id",
        "config_version",
        "config_hash",
        "code_revision",
        "status",
    )
    projected_runs = {
        row[0]: row[1:]
        for row in AnalysisRun.objects.filter(pk__in=ordered_run_ids)
        .order_by("pk")
        .values_list("pk", *run_fields)
    }
    if set(projected_runs) != set(ordered_run_ids):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    ordered_snapshot_ids = tuple(
        sorted(
            {
                *snapshot_ids,
                *(cast(UUID, values[4]) for values in projected_runs.values()),
            },
            key=str,
        )
    )
    snapshot_fields = (
        "universe_id",
        "as_of_date",
        "captured_at",
        "grade",
        "config_hash",
    )
    projected_snapshots = {
        row[0]: row[1:]
        for row in UniverseSnapshot.objects.filter(pk__in=ordered_snapshot_ids)
        .order_by("pk")
        .values_list("pk", *snapshot_fields)
    }
    if set(projected_snapshots) != set(ordered_snapshot_ids):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    ordered_universe_ids = tuple(
        sorted({str(values[0]) for values in projected_snapshots.values()})
    )
    universe_fields = ("name", "description", "config_version")
    projected_universes = {
        row[0]: row[1:]
        for row in Universe.objects.filter(pk__in=ordered_universe_ids)
        .order_by("pk")
        .values_list("pk", *universe_fields)
    }
    if set(projected_universes) != set(ordered_universe_ids):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    locked_universes = {
        universe.pk: universe
        for universe in Universe.objects.select_for_update(of=("self",))
        .filter(pk__in=ordered_universe_ids)
        .order_by("pk")
    }
    if set(locked_universes) != set(ordered_universe_ids) or any(
        tuple(getattr(universe, field) for field in universe_fields)
        != projected_universes[universe_id]
        for universe_id, universe in locked_universes.items()
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    locked_snapshots = {
        snapshot.pk: snapshot
        for snapshot in UniverseSnapshot.objects.select_for_update(of=("self",))
        .filter(pk__in=ordered_snapshot_ids)
        .order_by("pk")
    }
    if set(locked_snapshots) != set(ordered_snapshot_ids) or any(
        tuple(getattr(snapshot, field) for field in snapshot_fields)
        != projected_snapshots[snapshot_id]
        for snapshot_id, snapshot in locked_snapshots.items()
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    for snapshot in locked_snapshots.values():
        universe = locked_universes.get(snapshot.universe_id)
        if universe is None:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        snapshot._state.fields_cache["universe"] = universe

    locked_memberships = tuple(
        UniverseMembership.objects.select_for_update(of=("self",))
        .filter(snapshot_id__in=ordered_snapshot_ids)
        .order_by("pk")
    )
    listing_ids = set(requested_listing_ids)
    listing_ids.update(
        membership.listing_id for membership in locked_memberships if membership.eligible
    )
    ordered_listing_ids = tuple(sorted(listing_ids, key=str))
    listing_fields = (
        "security_id",
        "ticker",
        "exchange_mic",
        "provider_symbol",
        "currency",
        "region",
        "valid_from",
        "valid_to",
        "is_primary",
        "is_active",
    )
    projected_listings = {
        row[0]: row[1:]
        for row in Listing.objects.filter(pk__in=ordered_listing_ids)
        .order_by("pk")
        .values_list("pk", *listing_fields)
    }
    if set(projected_listings) != set(ordered_listing_ids):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    locked_listings = {
        listing.pk: listing
        for listing in Listing.objects.select_for_update(of=("self",))
        .filter(pk__in=ordered_listing_ids)
        .order_by("pk")
    }
    if set(locked_listings) != set(ordered_listing_ids) or any(
        tuple(getattr(listing, field) for field in listing_fields) != projected_listings[listing_id]
        for listing_id, listing in locked_listings.items()
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    ordered_security_ids = tuple(
        sorted({listing.security_id for listing in locked_listings.values()}, key=str)
    )
    security_fields = ("company_id", "security_type", "isin", "name")
    projected_securities = {
        row[0]: row[1:]
        for row in Security.objects.filter(pk__in=ordered_security_ids)
        .order_by("pk")
        .values_list("pk", *security_fields)
    }
    if set(projected_securities) != set(ordered_security_ids):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    locked_securities = {
        security.pk: security
        for security in Security.objects.select_for_update(of=("self",))
        .filter(pk__in=ordered_security_ids)
        .order_by("pk")
    }
    if set(locked_securities) != set(ordered_security_ids) or any(
        tuple(getattr(security, field) for field in security_fields)
        != projected_securities[security_id]
        for security_id, security in locked_securities.items()
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    for listing in locked_listings.values():
        security = locked_securities.get(listing.security_id)
        if security is None:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        listing._state.fields_cache["security"] = security
    for membership in locked_memberships:
        locked_snapshot = locked_snapshots.get(membership.snapshot_id)
        if locked_snapshot is None:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        membership._state.fields_cache["snapshot"] = locked_snapshot
        locked_listing = locked_listings.get(membership.listing_id)
        if locked_listing is not None:
            membership._state.fields_cache["listing"] = locked_listing

    locked_runs = {
        run.pk: run
        for run in AnalysisRun.objects.select_for_update(of=("self",))
        .filter(pk__in=ordered_run_ids)
        .order_by("pk")
    }
    if set(locked_runs) != set(ordered_run_ids) or any(
        tuple(getattr(run, field) for field in run_fields) != projected_runs[run_id]
        for run_id, run in locked_runs.items()
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    for run in locked_runs.values():
        locked_snapshot = locked_snapshots.get(run.universe_snapshot_id)
        if locked_snapshot is None:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        run._state.fields_cache["universe_snapshot"] = locked_snapshot
    return (
        locked_universes,
        locked_snapshots,
        locked_memberships,
        locked_listings,
        locked_securities,
        locked_runs,
    )


def _locked_panel_authorities(
    *,
    runs: Mapping[UUID, AnalysisRun],
    snapshots: Mapping[UUID, UniverseSnapshot],
    memberships: Sequence[UniverseMembership],
    listings: Mapping[UUID, Listing],
) -> Mapping[UUID, _LockedMediumV2Authority]:
    memberships_by_snapshot: dict[UUID, list[UniverseMembership]] = {
        snapshot_id: [] for snapshot_id in snapshots
    }
    for membership in memberships:
        if membership.eligible:
            memberships_by_snapshot[membership.snapshot_id].append(membership)
    return MappingProxyType(
        {
            run_id: _LockedMediumV2Authority(
                run=run,
                snapshot=snapshots[run.universe_snapshot_id],
                memberships=tuple(memberships_by_snapshot[run.universe_snapshot_id]),
                listings=tuple(
                    listings[membership.listing_id]
                    for membership in memberships_by_snapshot[run.universe_snapshot_id]
                ),
            )
            for run_id, run in runs.items()
        }
    )


def _validate_locked_medium_v2_authority(
    authority: _LockedMediumV2Authority,
    *,
    target_date: date | None = None,
    requested_listing_ids: Sequence[UUID] = (),
    detailed_errors: bool,
) -> None:
    """Validate one locked v2 snapshot closure without trusting caller rows."""

    def reject(message: str) -> None:
        raise ValueError(message if detailed_errors else _MEDIUM_V2_PAYLOAD_ERROR)

    snapshot = authority.snapshot
    universe = snapshot.universe
    effective_target = authority.run.target_date if authority.run is not None else target_date
    if effective_target is None:
        reject("us-price-medium-v2 requires an explicit target date")
    if snapshot.grade != UniverseSnapshot.Grade.RESEARCH:
        reject("us-price-medium-v2 requires a RESEARCH universe snapshot")
    if snapshot.as_of_date != effective_target:
        reject("us-price-medium-v2 requires the locked snapshot date to match target_date")
    if not isinstance(snapshot.config_hash, str) or len(snapshot.config_hash) != 64:
        reject("us-price-medium-v2 requires a valid locked snapshot config identity")
    if (
        snapshot.universe_id != universe.pk
        or not isinstance(universe.pk, str)
        or not universe.pk
        or not isinstance(universe.config_version, str)
        or not universe.config_version
    ):
        reject("us-price-medium-v2 requires a valid locked universe identity")

    memberships = authority.memberships
    listings = authority.listings
    if not memberships:
        reject("us-price-medium-v2 requires at least one eligible membership")
    membership_listing_ids = tuple(membership.listing_id for membership in memberships)
    listing_ids = tuple(listing.pk for listing in listings)
    if (
        any(
            membership.snapshot_id != snapshot.pk or not membership.eligible
            for membership in memberships
        )
        or len(set(membership_listing_ids)) != len(membership_listing_ids)
        or set(membership_listing_ids) != set(listing_ids)
        or len(listings) != len(memberships)
        or not set(requested_listing_ids).issubset(set(membership_listing_ids))
    ):
        reject("us-price-medium-v2 requires the exact locked eligible membership closure")

    subjects: list[str] = []
    for listing in listings:
        security = listing.security
        if listing.region != Region.US or listing.currency != "USD":
            reject(
                "us-price-medium-v2 requires every eligible listing to be "
                "US/USD stock-research eligible"
            )
        if security.security_type not in {
            Security.SecurityType.COMMON_STOCK,
            Security.SecurityType.ADR,
        }:
            reject(
                "Snapshot stock analysis supports common stocks and depositary "
                f"receipts; {listing.ticker} is an "
                f"{security.get_security_type_display().lower()}."
            )
        subject = listing.provider_symbol or listing.ticker
        if (
            not isinstance(subject, str)
            or not subject
            or subject != subject.strip()
            or subject == "SPY"
        ):
            reject(
                "us-price-medium-v2 requires unique non-empty listing provider "
                "subjects distinct from SPY"
            )
        subjects.append(subject)
    if len(subjects) != len(set(subjects)):
        reject(
            "us-price-medium-v2 requires unique non-empty listing provider "
            "subjects distinct from SPY"
        )

    run = authority.run
    if run is not None and (
        run.universe_snapshot_id != snapshot.pk
        or run.config_version != "us-price-baseline-v2"
        or run.config_hash != _MEDIUM_V2_SCORING_HASH
        or run.issued_on_time is not False
        or run.data_cutoff > run.generated_at
    ):
        reject("us-price-medium-v2 requires a valid locked analysis run identity")


def _lock_medium_v2_snapshot_authority(
    snapshot_id: UUID,
    *,
    target_date: date,
) -> _LockedMediumV2Authority:
    (
        _universes,
        snapshots,
        memberships,
        listings,
        _securities,
        _runs,
    ) = _lock_advisory_parent_rows(snapshot_ids=(snapshot_id,))
    snapshot = snapshots.get(snapshot_id)
    if snapshot is None:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    eligible_memberships = tuple(
        membership
        for membership in memberships
        if membership.snapshot_id == snapshot_id and membership.eligible
    )
    authority = _LockedMediumV2Authority(
        run=None,
        snapshot=snapshot,
        memberships=eligible_memberships,
        listings=tuple(listings[membership.listing_id] for membership in eligible_memberships),
    )
    _validate_locked_medium_v2_authority(
        authority,
        target_date=target_date,
        detailed_errors=True,
    )
    return authority


def _lock_advisory_authority(
    requested_analyses: Sequence[StockAnalysis],
) -> _LockedAdvisoryAuthority:
    """Resolve and lock every caller-requested analysis without trusting it.

    A short, unlocked identity projection is needed to discover parent keys
    before taking the required parent-first locks. The final analysis lock
    verifies that projection still names the same run and listing; a change
    in that discovery window therefore fails the invocation rather than
    causing a child-first re-lock in a deadlock-prone order.
    """
    analysis_ids: list[int] = []
    for analysis in requested_analyses:
        if analysis.pk is None:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        analysis_ids.append(analysis.pk)
    ordered_analysis_ids = tuple(sorted(set(analysis_ids)))
    analysis_fields = tuple(
        field.attname for field in StockAnalysis._meta.concrete_fields if not field.primary_key
    )
    projected_analyses = {
        row[0]: row[1:]
        for row in StockAnalysis.objects.filter(pk__in=ordered_analysis_ids)
        .order_by("pk")
        .values_list("pk", *analysis_fields)
    }
    if set(projected_analyses) != set(ordered_analysis_ids):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    run_index = analysis_fields.index("run_id")
    listing_index = analysis_fields.index("listing_id")

    (
        _universes,
        snapshots,
        memberships,
        listings,
        _securities,
        runs,
    ) = _lock_advisory_parent_rows(
        run_ids=[cast(UUID, projected_analyses[pk][run_index]) for pk in ordered_analysis_ids],
        requested_listing_ids=[
            cast(UUID, projected_analyses[pk][listing_index]) for pk in ordered_analysis_ids
        ],
    )
    locked_analyses = {
        analysis.pk: analysis
        for analysis in StockAnalysis.objects.select_for_update(of=("self",))
        .filter(pk__in=ordered_analysis_ids)
        .order_by("pk")
    }
    if set(locked_analyses) != set(ordered_analysis_ids) or any(
        tuple(getattr(analysis, field) for field in analysis_fields)
        != projected_analyses[analysis_id]
        for analysis_id, analysis in locked_analyses.items()
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    for _analysis_id, locked in locked_analyses.items():
        if locked.run_id not in runs or locked.listing_id not in listings:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        locked._state.fields_cache["run"] = runs[locked.run_id]
        locked._state.fields_cache["listing"] = listings[locked.listing_id]

    return _LockedAdvisoryAuthority(
        analyses=tuple(locked_analyses[analysis_id] for analysis_id in analysis_ids),
        panels_by_run=_locked_panel_authorities(
            runs=runs,
            snapshots=snapshots,
            memberships=memberships,
            listings=listings,
        ),
    )


def _lock_medium_v2_authority(run: AnalysisRun) -> _LockedMediumV2Authority:
    """Independently lock a panel authority for a direct attestation call."""
    if run.pk is None:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    (
        _universes,
        snapshots,
        memberships,
        listings,
        _securities,
        runs,
    ) = _lock_advisory_parent_rows(
        run_ids=(run.pk,),
    )
    authority = _locked_panel_authorities(
        runs=runs,
        snapshots=snapshots,
        memberships=memberships,
        listings=listings,
    )[run.pk]
    _validate_locked_medium_v2_authority(
        authority,
        detailed_errors=False,
    )
    return authority


@dataclass(frozen=True, slots=True)
class LongForecastContext:
    config: LongForecastConfig
    config_hash: str
    model_version: str
    forecasts: dict[str, dict[str, LongForecast]]


@dataclass(frozen=True, slots=True)
class _Under10DecisionEvidence:
    """Immutable decision-prediction provenance needed for shadow replay."""

    price_provider: str
    price_subject: str
    price_entry: Mapping[str, Any]


def _validated_observed_v3_revision(
    *,
    config: ScoringConfig,
    config_hash_value: str,
    provider: str,
    benchmark_subject: str | None,
) -> str:
    if config.version != V3_VERSION:
        raise ValueError("Observed v3 issuance requires us-price-baseline-v3")
    if config_hash_value != V3_EFFECTIVE_CONFIG_HASH:
        raise ValueError("Observed v3 issuance requires the exact reviewed v3 scoring config")
    if provider != TWELVE_DATA_PROVIDER:
        raise ValueError("Observed v3 issuance requires the Twelve Data provider")
    if benchmark_subject != INVESTABLE_US_ETF_SYMBOL:
        raise ValueError("Observed v3 issuance requires the SPY benchmark subject")

    raw_revision = os.getenv("STANSTOCK_CODE_REVISION")
    if raw_revision is None:
        raise ValueError("Observed v3 issuance requires STANSTOCK_CODE_REVISION")
    if _FULL_LOWERHEX_GIT_REVISION.fullmatch(raw_revision) is None:
        raise ValueError(
            "Observed v3 issuance requires a full lowercase 40-hex STANSTOCK_CODE_REVISION"
        )
    try:
        checkout_revision = clean_git_revision(Path(settings.BASE_DIR))
    except ValueError:
        raise ValueError(
            "Observed v3 issuance requires a verifiably clean committed Git revision"
        ) from None
    if raw_revision != checkout_revision:
        raise ValueError(
            "Observed v3 issuance revision does not match the clean committed Git HEAD"
        )
    return raw_revision


def _prepare_v3_price_frame(frame: pl.DataFrame, *, source: str) -> pl.DataFrame:
    """Validate v3 evidence before any lossy cast/filter and sort it once."""
    missing_columns = [column for column in ("date", "close") if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"V3 {source} frame is missing columns: {', '.join(missing_columns)}")
    if frame.height == 0:
        raise ValueError(f"V3 {source} frame has no price observations")

    date_column = frame["date"]
    date_dtype = frame.schema["date"]
    if date_column.null_count():
        raise ValueError(f"V3 {source} frame contains null dates")
    if date_dtype == pl.Date:
        normalized_dates = date_column
    elif isinstance(date_dtype, pl.Datetime):
        normalized_dates = date_column.dt.date()
    elif date_dtype in (pl.Utf8, pl.String):
        if not bool(date_column.str.contains(r"^\d{4}-\d{2}-\d{2}$").all()):
            raise ValueError(f"V3 {source} frame dates must use strict ISO YYYY-MM-DD")
        try:
            normalized_dates = date_column.str.strptime(pl.Date, "%Y-%m-%d", strict=True)
        except pl.exceptions.PolarsError as error:
            raise ValueError(f"V3 {source} frame contains unparseable dates") from error
    else:
        raise ValueError(f"V3 {source} frame has unsupported date dtype {date_dtype!r}")
    if normalized_dates.null_count():
        raise ValueError(f"V3 {source} frame contains unparseable dates")

    close_dtype = frame.schema["close"]
    if close_dtype == pl.Boolean:
        raise ValueError(f"V3 {source} frame close values must be numeric")
    try:
        normalized_close = frame["close"].cast(pl.Float64, strict=False)
    except pl.exceptions.PolarsError as error:
        raise ValueError(f"V3 {source} frame close values must be numeric") from error
    if (
        normalized_close.null_count()
        or not bool(normalized_close.is_finite().all())
        or not bool((normalized_close > 0).all())
    ):
        raise ValueError(f"V3 {source} frame closes must be finite and positive")

    expressions = [
        normalized_dates.alias("date"),
        normalized_close.alias("close"),
    ]
    normalized_volume: pl.Series | None = None
    if "volume" in frame.columns:
        if frame.schema["volume"] == pl.Boolean:
            raise ValueError(f"V3 {source} frame volume values must be numeric")
        try:
            normalized_volume = frame["volume"].cast(pl.Float64, strict=False)
        except pl.exceptions.PolarsError as error:
            raise ValueError(f"V3 {source} frame volume values must be numeric") from error
        if (
            normalized_volume.null_count()
            or not bool(normalized_volume.is_finite().all())
            or not bool((normalized_volume >= 0).all())
        ):
            raise ValueError(f"V3 {source} frame volumes must be finite and nonnegative")
        product = normalized_close * normalized_volume
        if not bool(product.is_finite().all()):
            raise ValueError(f"V3 {source} frame close-volume products must be finite")
        expressions.append(normalized_volume.alias("volume"))

    normalized = frame.with_columns(expressions)
    if normalized["date"].n_unique() != normalized.height:
        raise ValueError(f"V3 {source} frame contains duplicate normalized dates")
    return normalized.sort("date")


def _v3_factor_policy_payload(config: ScoringConfig) -> dict[str, Any]:
    policy = config.short_scoring
    if policy is None:
        raise ValueError("V3 factor-policy provenance requires a short-scoring policy")
    return {
        "schema_version": policy.schema_version,
        "macd_indicator": config.factor_policy.macd_indicator,
        "abnormal_volume_indicator": config.factor_policy.abnormal_volume_indicator,
        "liquidity_indicator": config.factor_policy.liquidity_indicator,
        "strict_finite_inputs": config.factor_policy.strict_finite_inputs,
        "rsi": {
            "convention": policy.rsi.convention,
            "window_sessions": policy.rsi.window_sessions,
        },
        "risk_window": {
            "sessions": policy.risk_window.sessions,
            "annualization_sessions": policy.risk_window.annualization_sessions,
        },
        "beta_roles": {
            "factor_score": policy.beta_roles.factor_score,
            "composite_risk": policy.beta_roles.composite_risk,
        },
        "factor_maps": {
            name: transform.as_dict() for name, transform in policy.factor_maps.items()
        },
        "risk_penalty_maps": {
            name: transform.as_dict() for name, transform in policy.risk_penalty_maps.items()
        },
        "buy_min_liquidity_20d": config.recommendation.buy_min_liquidity_20d,
    }


def compute_listing_analysis(
    *,
    listing: Listing,
    price_frame: pl.DataFrame,
    config: ScoringConfig,
    decision_time: datetime,
    benchmark_frame: pl.DataFrame | None = None,
    facts: Any = (),
    source_assets: list[DataAsset] | None = None,
    price_asset: DataAsset | None = None,
    sample_support: dict[str, int] | None = None,
) -> AnalysisComputation:
    require_stock_research_listing(listing, operation="Stock analysis")
    calculation_price_frame = price_frame
    calculation_benchmark_frame = benchmark_frame
    common_risk_policy = None
    if config.short_scoring is not None:
        if listing.currency != "USD":
            raise ValueError("us-price-baseline-v3 supports USD listings only")
        calculation_price_frame = _prepare_v3_price_frame(price_frame, source="listing")
        if benchmark_frame is not None:
            calculation_benchmark_frame = _prepare_v3_price_frame(
                benchmark_frame,
                source="benchmark",
            )
        common_risk_policy = config.short_scoring.risk_window
    if common_risk_policy is None:
        indicators = calculate_indicators(
            calculation_price_frame,
            benchmark=calculation_benchmark_frame,
            windows=config.windows,
        )
    else:
        indicators = calculate_indicators(
            calculation_price_frame,
            benchmark=calculation_benchmark_frame,
            windows=config.windows,
            common_risk_policy=common_risk_policy,
        )
    price = indicators.values.get("last_close")
    if price is None:
        raise ValueError(f"No usable price history for {listing}")
    fundamental_inputs = (
        inputs_from_facts(())
        if config.analysis_mode == "price_only_baseline"
        else inputs_from_facts(facts)
    )
    fundamentals = calculate_fundamentals(fundamental_inputs, price=price)
    component_scores = score_components(indicators, fundamentals, config)
    aggregate = aggregate_score(
        component_scores,
        config,
        decision_date=decision_time.date(),
        indicators=indicators,
    )
    risk = assess_risk(indicators, fundamentals, config)
    scenarios = build_scenarios(
        calculation_price_frame,
        indicators,
        fundamentals,
        aggregate,
        config,
        sample_support=sample_support,
    )
    decision = decide_recommendation(
        aggregate.overall,
        risk,
        aggregate.confidence,
        config,
        scenarios=scenarios,
        indicators=indicators,
    )
    daily_change = indicators.values.get("return_1d")
    assets = _dedupe_assets(source_assets or [])
    asset_payload = [_asset_payload(asset) for asset in assets]
    factor_policy = (
        _v3_factor_policy_payload(config)
        if config.short_scoring is not None
        else {
            "macd_indicator": config.factor_policy.macd_indicator,
            "macd_score_low": config.factor_policy.macd_score_low,
            "macd_score_high": config.factor_policy.macd_score_high,
            "abnormal_volume_indicator": config.factor_policy.abnormal_volume_indicator,
            "liquidity_indicator": config.factor_policy.liquidity_indicator,
            "liquidity_score_low": config.factor_policy.liquidity_score_low,
            "liquidity_score_high": config.factor_policy.liquidity_score_high,
            "strict_finite_inputs": config.factor_policy.strict_finite_inputs,
            "buy_min_liquidity_20d": config.recommendation.buy_min_liquidity_20d,
        }
    )
    data_quality = {
        "indicator_missing": indicators.missing,
        "fundamental_missing": fundamentals.missing,
        "scoring_missing": component_scores.missing,
        "coverage": component_scores.coverage,
        "missingness_penalty": aggregate.missingness_penalty,
        "freshness_penalty": aggregate.freshness_penalty,
        "observation_count": indicators.observation_count,
        "source_assets": asset_payload,
        "recommendation_gates": decision.gates,
        "risk_insufficiency_reason": risk.insufficiency_reason,
        "analysis_mode": config.analysis_mode,
        "fundamentals_used": config.analysis_mode != "price_only_baseline",
        "supported_horizons": list(config.supported_horizons),
        "factor_policy": factor_policy,
    }
    price_asset_metadata = (
        next(
            (asset for asset in asset_payload if asset["id"] == str(price_asset.id)),
            None,
        )
        if price_asset is not None
        else None
    )
    if price_asset_metadata is not None:
        data_quality["price_source"] = {
            "asset_id": price_asset_metadata["id"],
            "provider": price_asset_metadata["provider"],
            "subject": price_asset_metadata["subject"],
        }
        if "return_definition" in price_asset_metadata:
            data_quality["return_definition"] = price_asset_metadata["return_definition"]
        if "dividends_included" in price_asset_metadata:
            data_quality["dividends_included"] = price_asset_metadata["dividends_included"]
    reasons = generate_reasons(indicators, fundamentals, aggregate)
    risks = generate_risks(indicators, fundamentals, risk, aggregate)
    return AnalysisComputation(
        indicators=indicators,
        fundamentals=fundamentals,
        aggregate=aggregate,
        scenarios=scenarios,
        risk_score=risk.score,
        risk_class=risk.risk_class,
        recommendation=decision.recommendation,
        reasons=reasons,
        risks=risks,
        data_quality=data_quality,
        source_assets=asset_payload,
        current_price=price,
        daily_change=daily_change,
        price_asset=price_asset,
    )


def _compute_listing_from_asof(
    *,
    listing: Listing,
    asof: AsOfData,
    provider: str,
    config: ScoringConfig,
    decision_time: datetime,
    issued_on_time: bool,
    provider_plan: str | None,
    code_revision_value: str,
    subject: str | None = None,
    benchmark_subject: str | None = None,
    sample_support: dict[str, int] | None = None,
    target_date: date | None = None,
) -> AnalysisComputation:
    if config.short_scoring is not None and listing.currency != "USD":
        raise ValueError("us-price-baseline-v3 supports USD listings only")
    symbol = subject or listing.provider_symbol or listing.ticker
    if config.short_scoring is not None:
        selected_price_asset = asof.latest_asset(
            provider=provider,
            kind="price_history",
            subject=symbol,
        )
        price_read = asof.price_frame_for_asset_with_diagnostics(
            asset=selected_price_asset,
            through_date=target_date,
        )
    else:
        price_read = asof.price_frame_with_diagnostics(
            provider=provider,
            subject=symbol,
            through_date=target_date,
        )
    price_asset = price_read.asset
    price_frame = price_read.frame
    if config.short_scoring is not None and price_read.invalid_session_date_rows:
        raise ValueError("V3 listing price history contains invalid session dates")
    benchmark_frame: pl.DataFrame | None = None
    source_assets = [price_asset]
    if (
        benchmark_subject is not None
        if config.short_scoring is not None
        else bool(benchmark_subject)
    ):
        assert benchmark_subject is not None
        if config.short_scoring is not None:
            selected_benchmark_asset = asof.latest_asset(
                provider=provider,
                kind="price_history",
                subject=benchmark_subject,
            )
            benchmark_read = asof.price_frame_for_asset_with_diagnostics(
                asset=selected_benchmark_asset,
                through_date=target_date,
            )
        else:
            benchmark_read = asof.price_frame_with_diagnostics(
                provider=provider,
                subject=benchmark_subject,
                through_date=target_date,
            )
        if config.short_scoring is not None and benchmark_read.invalid_session_date_rows:
            raise ValueError("V3 benchmark price history contains invalid session dates")
        benchmark_frame = benchmark_read.frame
        source_assets.append(benchmark_read.asset)
    facts: list[Any] = []
    full_analysis_facts_loaded = config.analysis_mode != "price_only_baseline"
    if full_analysis_facts_loaded:
        facts = list(
            asof.fundamental_facts(
                company_id=listing.security.company_id,
                available_through=decision_time,
            ).select_related("source_asset")
        )
        source_assets.extend(fact.source_asset for fact in facts)
    computation = compute_listing_analysis(
        listing=listing,
        price_frame=price_frame,
        benchmark_frame=benchmark_frame,
        facts=facts,
        source_assets=source_assets,
        price_asset=price_asset,
        config=config,
        decision_time=decision_time,
        sample_support=sample_support,
    )
    return _with_under10_assessment(
        computation,
        listing=listing,
        asof=asof,
        provider=provider,
        provider_plan=provider_plan,
        loaded_facts=facts,
        loaded_facts_cover_all_concepts=full_analysis_facts_loaded,
        price_frame=price_frame,
        price_asset=price_asset,
        invalid_session_date_rows=price_read.invalid_session_date_rows,
        decision_time=decision_time,
        target_date=target_date or decision_time.date(),
        issued_on_time=issued_on_time,
        code_revision_value=code_revision_value,
    )


def _with_under10_assessment(
    computation: AnalysisComputation,
    *,
    listing: Listing,
    asof: AsOfData,
    provider: str,
    provider_plan: str | None,
    loaded_facts: Sequence[FundamentalFact],
    loaded_facts_cover_all_concepts: bool,
    price_frame: pl.DataFrame,
    price_asset: DataAsset | None,
    invalid_session_date_rows: int,
    decision_time: datetime,
    target_date: date,
    issued_on_time: bool,
    code_revision_value: str,
) -> AnalysisComputation:
    """Attach the shadow Under-$10 diagnostic to a qualifying computation.

    The scored computation above is already final: this only adds one nested
    `data_quality` key. A listing whose decision-run reference close is not a
    valid USD Under-$10 close is returned untouched and issues no additional
    SEC query at all.

    The reference close classified here is the *rounded* value
    `_create_stock_analysis` persists, so the band recorded in the payload is
    the same band sample construction later reads -- never an unrounded float
    and never a mutable current market row.
    """
    reference_close = _decimal(computation.current_price, places=6)
    band = classify_price_band(
        close=reference_close,
        price_date=target_date,
        date_basis=DECISION_TARGET_DATE_BASIS,
        currency=listing.currency,
    )
    if band is None or band.slug != UNDER_10_BAND:
        return computation
    if loaded_facts_cover_all_concepts:
        # Full-analysis mode already read every visible fact for this
        # company under the same cutoff; a second shadow query would be a
        # duplicate read of the same evidence.
        facts: Sequence[FundamentalFact] = loaded_facts
    else:
        facts = list(
            asof.fundamental_facts(
                company_id=listing.security.company_id,
                concepts=list(UNDER10_CONCEPTS),
                available_through=decision_time,
            ).select_related("source_asset")
        )
    qualified_facts = qualify_under10_sec_facts(facts)
    payload = build_under10_assessment(
        facts=qualified_facts,
        sec_config=_sec_fundamentals_config(),
        price_frame=price_frame,
        price_asset=price_asset,
        price_source=_mapping_or_none(computation.data_quality.get("price_source")),
        reference_close=reference_close,
        target_date=target_date,
        data_cutoff=decision_time,
        code_revision_value=code_revision_value,
        provider=provider,
        provider_plan=provider_plan,
        evidence_cutoff_safe=_shadow_evidence_cutoff_safe(
            qualified_facts,
            issued_on_time=issued_on_time,
            data_cutoff=decision_time,
        ),
        company_identity_present=listing.security.company_id is not None,
        invalid_session_date_rows=invalid_session_date_rows,
        listing_id=str(listing.id),
    )
    # `replace` keeps every other computed field -- including the original
    # `source_assets` list object -- identical. The shadow SEC evidence lives
    # only under the new nested key and never joins the prediction manifest.
    return replace(
        computation,
        data_quality={**computation.data_quality, UNDER10_ASSESSMENT_KEY: payload},
    )


@lru_cache(maxsize=1)
def _sec_fundamentals_config() -> SecFundamentalsConfig:
    """The reviewed SEC fundamentals configuration, loaded at most once.

    Mirrors the cached opportunity-policy loader; the pure assessment builder
    never loads configuration itself.
    """
    return load_sec_fundamentals_config()


def _mapping_or_none(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _shadow_evidence_cutoff_safe(
    facts: Sequence[FundamentalFact],
    *,
    issued_on_time: bool,
    data_cutoff: datetime,
) -> bool:
    """Whether qualified SEC shadow evidence satisfies the asset cutoff rule.

    The strict asset-retrieval check belongs to on-time issuance only. A
    research-grade reconstruction may legitimately read evidence retrieved
    later than its historical cutoff, while `AsOfData` and the correction
    resolution still require the *facts* themselves to be provably available
    at that cutoff. The caller supplies the exact provider-qualified sequence
    also passed to the defensive assessment builder, so foreign evidence
    cannot make an SEC cutoff claim fail.
    """
    if not issued_on_time:
        return True
    for asset in _dedupe_assets([fact.source_asset for fact in facts]):
        if _asset_cutoff_violation(_asset_payload(asset), data_cutoff=data_cutoff) is not None:
            return False
    return True


def under10_assessment_matches_persisted_evidence(
    *,
    analysis: StockAnalysis,
    recorded: Mapping[str, Any],
    store: AssetStore | None = None,
) -> bool:
    """Whether a stored shadow assessment exactly replays from its evidence.

    This is a read-only evidence-validation predicate, not an alternate producer:
    it never repairs the payload, returns reconstructed values, or writes a
    model. Price provenance comes from the immutable decision-prediction
    cohort rather than the mutable parent JSON alone. The exact price asset
    named there is read through :class:`AsOfData`, while SEC facts are
    independently selected under the original run's generation and data
    cutoffs. Only exact canonical equality of the resulting solvency and
    liquidity blocks admits the recorded values for display.

    Expected evidence failures return ``False``. Database failures and
    programming errors are deliberately not hidden.
    """
    if analysis._state.adding:
        return False
    recorded_solvency = recorded.get("solvency")
    recorded_liquidity = recorded.get("liquidity")
    recorded_split = recorded.get("split_verification")
    if (
        not isinstance(recorded_solvency, Mapping)
        or not isinstance(recorded_liquidity, Mapping)
        or not isinstance(recorded_split, Mapping)
    ):
        return False

    decision_evidence = _under10_decision_evidence(analysis)
    if decision_evidence is None:
        return False
    price_entry = decision_evidence.price_entry
    asset_id = price_entry.get("id")
    asset_checksum = price_entry.get("sha256")
    if not isinstance(asset_id, str) or not isinstance(asset_checksum, str):
        return False
    try:
        asset_uuid = UUID(asset_id)
    except ValueError:
        return False
    if str(asset_uuid) != asset_id:
        return False
    if not _under10_parent_price_anchor_matches(
        analysis,
        asset_id=asset_id,
        asset_checksum=asset_checksum,
        price_provider=decision_evidence.price_provider,
        price_subject=decision_evidence.price_subject,
    ):
        return False
    recorded_price_asset = recorded_liquidity.get("price_asset")
    if (
        not isinstance(recorded_price_asset, Mapping)
        or recorded_price_asset.get("id") != asset_id
        or recorded_price_asset.get("sha256") != asset_checksum
    ):
        return False

    try:
        price_asset = DataAsset.objects.get(pk=asset_uuid)
    except DataAsset.DoesNotExist:
        return False
    if (
        str(price_asset.pk) != asset_id
        or price_asset.sha256 != asset_checksum
        or price_asset.provider != decision_evidence.price_provider
        or price_asset.kind != "price_history"
        or price_asset.subject != decision_evidence.price_subject
        or price_entry.get("provider") != price_asset.provider
        or price_entry.get("kind") != price_asset.kind
        or price_entry.get("subject") != price_asset.subject
    ):
        return False
    if recorded_split.get("provider") != price_asset.provider:
        return False

    asset_store = store or AssetStore()
    asof = AsOfData(analysis.run.generated_at, asset_store)
    try:
        price_read = asof.price_frame_for_asset_with_diagnostics(
            asset=price_asset,
            through_date=analysis.run.target_date,
        )
    except (
        PriceFrameChecksumMismatchError,
        FileNotFoundError,
        pl.exceptions.PolarsError,
        ValueError,
    ):
        return False

    try:
        facts = list(
            asof.fundamental_facts(
                company_id=analysis.listing.security.company_id,
                concepts=list(UNDER10_CONCEPTS),
                available_through=analysis.run.data_cutoff,
            ).select_related("source_asset")
        )
    except ValueError:
        return False
    qualified_facts = qualify_under10_sec_facts(facts)
    try:
        replayed = build_under10_assessment(
            facts=qualified_facts,
            sec_config=_sec_fundamentals_config(),
            price_frame=price_read.frame,
            price_asset=price_asset,
            price_source={
                "asset_id": asset_id,
                "provider": price_asset.provider,
                "subject": price_asset.subject,
            },
            reference_close=analysis.current_price,
            target_date=analysis.run.target_date,
            data_cutoff=analysis.run.data_cutoff,
            code_revision_value=analysis.run.code_revision,
            provider=price_asset.provider,
            # ProviderRecord is mutable capability context and is not
            # decision evidence. The split block is therefore not replayed;
            # only its recorded provider is bound above. Provider plan does
            # not enter either evidence-derived block compared below.
            provider_plan=None,
            evidence_cutoff_safe=_shadow_evidence_cutoff_safe(
                qualified_facts,
                issued_on_time=analysis.run.issued_on_time,
                data_cutoff=analysis.run.data_cutoff,
            ),
            company_identity_present=analysis.listing.security.company_id is not None,
            invalid_session_date_rows=price_read.invalid_session_date_rows,
            listing_id=str(analysis.listing_id),
        )
    except ValueError:
        return False

    replayed_solvency = replayed.get("solvency")
    replayed_liquidity = replayed.get("liquidity")
    if not isinstance(replayed_solvency, Mapping) or not isinstance(replayed_liquidity, Mapping):
        return False
    try:
        return canonical_json(recorded_solvency) == canonical_json(
            replayed_solvency
        ) and canonical_json(recorded_liquidity) == canonical_json(replayed_liquidity)
    except (TypeError, ValueError, OverflowError):
        return False


def _under10_decision_evidence(
    analysis: StockAnalysis,
) -> _Under10DecisionEvidence | None:
    """Resolve one internally agreeing immutable decision-prediction cohort."""
    run = analysis.run
    predictions = list(
        Prediction.objects.filter(
            analysis_id=analysis.pk,
            evidence_role=Prediction.EvidenceRole.DECISION,
        )
        .only(
            "id",
            "listing_id",
            "generated_at",
            "target_date",
            "issued_on_time",
            "price_at_prediction",
            "price_provider",
            "price_subject",
            "data_cutoff",
            "source_assets",
            "code_revision",
        )
        .order_by("pk")
    )
    if not predictions:
        return None

    first = predictions[0]
    price_provider = first.price_provider
    price_subject = first.price_subject
    if not price_provider or not price_subject:
        return None
    manifest = _canonical_under10_source_manifest(first.source_assets)
    if manifest is None:
        return None
    manifest_json, manifest_entries = manifest
    for prediction in predictions:
        if (
            prediction.listing_id != analysis.listing_id
            or prediction.generated_at != run.generated_at
            or prediction.target_date != run.target_date
            or prediction.data_cutoff != run.data_cutoff
            or prediction.code_revision != run.code_revision
            or prediction.issued_on_time != run.issued_on_time
            or prediction.price_at_prediction != analysis.current_price
            or prediction.price_provider != price_provider
            or prediction.price_subject != price_subject
        ):
            return None
        candidate_manifest = _canonical_under10_source_manifest(prediction.source_assets)
        if candidate_manifest is None or candidate_manifest[0] != manifest_json:
            return None

    price_entries = [
        entry
        for entry in manifest_entries
        if entry.get("provider") == price_provider
        and entry.get("kind") == "price_history"
        and entry.get("subject") == price_subject
    ]
    if len(price_entries) != 1:
        return None
    return _Under10DecisionEvidence(
        price_provider=price_provider,
        price_subject=price_subject,
        price_entry=price_entries[0],
    )


def _canonical_under10_source_manifest(
    value: object,
) -> tuple[str, tuple[Mapping[str, Any], ...]] | None:
    if not isinstance(value, list) or not all(isinstance(entry, Mapping) for entry in value):
        return None
    entries = tuple(entry for entry in value if isinstance(entry, Mapping))
    try:
        serialized = canonical_json({"source_assets": list(entries)})
    except (TypeError, ValueError, OverflowError):
        return None
    return serialized, entries


def _under10_parent_price_anchor_matches(
    analysis: StockAnalysis,
    *,
    asset_id: str,
    asset_checksum: str,
    price_provider: str,
    price_subject: str,
) -> bool:
    data_quality = analysis.data_quality
    if not isinstance(data_quality, Mapping):
        return False
    price_source = data_quality.get("price_source")
    source_assets = data_quality.get("source_assets")
    if (
        not isinstance(price_source, Mapping)
        or price_source.get("asset_id") != asset_id
        or price_source.get("provider") != price_provider
        or price_source.get("subject") != price_subject
        or not isinstance(source_assets, list)
    ):
        return False
    matches = [
        entry
        for entry in source_assets
        if isinstance(entry, Mapping) and entry.get("id") == asset_id
    ]
    if len(matches) != 1:
        return False
    matching = matches[0]
    return bool(
        matching.get("sha256") == asset_checksum
        and matching.get("provider") == price_provider
        and matching.get("kind") == "price_history"
        and matching.get("subject") == price_subject
    )


def _create_analysis_run(
    *,
    generated_at: datetime,
    data_cutoff: datetime,
    target_date: date,
    issued_on_time: bool,
    universe_snapshot: UniverseSnapshot,
    config: ScoringConfig,
    config_hash_value: str,
    code_revision_value: str,
) -> AnalysisRun:
    return AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=data_cutoff,
        target_date=target_date,
        issued_on_time=issued_on_time,
        universe_snapshot=universe_snapshot,
        config_version=config.version,
        config_hash=config_hash_value,
        code_revision=code_revision_value,
    )


def _persist_listing_analysis(
    *,
    run: AnalysisRun,
    listing: Listing,
    computation: AnalysisComputation,
    generated_at: datetime,
    data_cutoff: datetime,
    model_version: str,
    config_hash_value: str,
    code_revision_value: str,
    advisory_context: AdvisoryForecastContext | None = None,
    long_context: LongForecastContext | None = None,
) -> PersistedAnalysis:
    if run.issued_on_time:
        _validate_on_time_source_assets(computation.source_assets, data_cutoff=data_cutoff)
    advisory_forecasts = (
        advisory_context.forecasts.get(str(listing.pk), {}) if advisory_context is not None else {}
    )
    long_forecasts = (
        long_context.forecasts.get(str(listing.pk), {}) if long_context is not None else {}
    )
    analysis = _create_stock_analysis(
        run,
        listing,
        computation,
        advisory_forecasts=advisory_forecasts,
        long_forecasts=long_forecasts,
    )
    decision_predictions = append_predictions(
        analysis=analysis,
        computation=computation,
        generated_at=generated_at,
        data_cutoff=data_cutoff,
        issued_on_time=run.issued_on_time,
        supported_horizons=tuple(computation.data_quality["supported_horizons"]),
        model_version=model_version,
        config_hash_value=config_hash_value,
        source_assets=computation.source_assets,
        code_revision_value=code_revision_value,
    )
    long_predictions: tuple[Prediction, ...] = ()
    if long_context is not None:
        long_predictions = append_long_advisory_predictions(
            analysis=analysis,
            forecasts=long_forecasts,
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            issued_on_time=run.issued_on_time,
            model_version=long_context.model_version,
            config_hash_value=long_context.config_hash,
            code_revision_value=code_revision_value,
        )
    return PersistedAnalysis(
        run=run,
        analysis=analysis,
        predictions=(*decision_predictions, *long_predictions),
        computation=computation,
    )


def _validate_on_time_source_assets(
    source_assets: list[dict[str, Any]],
    *,
    data_cutoff: datetime,
) -> None:
    for asset in source_assets:
        field = _asset_cutoff_violation(asset, data_cutoff=data_cutoff)
        if field is not None:
            raise ValueError(
                f"On-time analysis source asset {asset['id']} has {field} after data cutoff"
            )


def _asset_cutoff_violation(
    asset: Mapping[str, Any],
    *,
    data_cutoff: datetime,
) -> str | None:
    """First cutoff-safety field this asset violates, or ``None``.

    Fields are checked in the released order -- ``available_at`` before
    ``retrieved_at`` -- so the core validator above keeps raising on exactly
    the field and with exactly the wording it always did. The shadow
    assessment reuses the same rule to *withhold* rather than to fail a run.
    """
    for field in ("available_at", "retrieved_at"):
        timestamp = datetime.fromisoformat(str(asset[field]))
        if timestamp > data_cutoff:
            return field
    return None


def _resolve_provider_plan(provider: str) -> str | None:
    """The normalized recorded provider plan, resolved once per analysis run.

    Only Twelve Data records a plan whose value changes the split-capability
    refusal, so no other provider issues a query. A missing record, a missing
    leaf, a non-string leaf, and an empty label are all "no recorded plan";
    none of them is treated as an unresolved value that a later per-listing
    lookup could retry.

    Only the plan leaf is read: the surrounding provider metadata document
    carries entitlement and licensing detail this assessment has no reason to
    touch.
    """
    if provider != TWELVE_DATA_PROVIDER:
        return None
    recorded = (
        ProviderRecord.objects.filter(provider=provider)
        .values_list("metadata__plan", flat=True)
        .first()
    )
    if not isinstance(recorded, str):
        return None
    return normalized_provider_plan(recorded)


def _analysis_output_manifest_entries(
    run: AnalysisRun, results: list[PersistedAnalysis]
) -> list[ManifestEntry]:
    """The complete, exact set of manifest entries for one observed run:
    the `AnalysisRun` row itself, every persisted `StockAnalysis`, and every
    persisted `Prediction` -- one canonical full-row digest each. Built
    directly from the just-persisted ORM instances (never re-queried),
    since this runs inside the same still-open transaction that created
    them.
    """
    entries = [
        ManifestEntry(
            model=ANALYSIS_RUN_MODEL,
            row_id=str(run.id),
            digest=row_digest(ANALYSIS_RUN_MODEL, model_row_values(run, ANALYSIS_RUN_FIELDS)),
        )
    ]
    for persisted in results:
        entries.append(
            ManifestEntry(
                model=STOCK_ANALYSIS_MODEL,
                row_id=str(persisted.analysis.id),
                digest=row_digest(
                    STOCK_ANALYSIS_MODEL,
                    model_row_values(persisted.analysis, STOCK_ANALYSIS_FIELDS),
                ),
            )
        )
        for prediction in persisted.predictions:
            entries.append(
                ManifestEntry(
                    model=PREDICTION_MODEL,
                    row_id=str(prediction.id),
                    digest=row_digest(
                        PREDICTION_MODEL, model_row_values(prediction, PREDICTION_FIELDS)
                    ),
                )
            )
    return entries


def _write_analysis_output_manifest(
    *,
    run: AnalysisRun,
    results: list[PersistedAnalysis],
    plan: ManifestPlan,
    store: AssetStore,
    retrieved_at: datetime,
) -> str:
    """Write and register the one immutable manifest asset binding this
    observed `AnalysisRun` to the exact, complete set of rows it produced
    and the exact output `plan` it was required to produce.

    Mirrors `stanstock.data.live_us._ensure_snapshot`'s exact idiom: decide
    whether the target path already exists *before* writing (a fresh
    `run.id` UUID means it never should, so this only guards against a
    stale leftover from an earlier failed attempt at the very same path),
    write bytes then register the `DataAsset` row, and on any failure from
    either step unlink only a file this call itself just created -- never a
    genuinely pre-existing one.

    Every expected storage-layer failure (path resolution, physical write)
    is normalized into a stable, path-free `RefreshVerificationError` raised
    `from None`; a `DataAsset.objects.create` failure (a DB integrity or
    programming error) is never relabeled or swallowed, only cleaned up
    after.
    """
    try:
        envelope = build_manifest_envelope(
            run_id=run.id, plan=plan, entries=_analysis_output_manifest_entries(run, results)
        )
        envelope_bytes = dumps_canonical_envelope(envelope)
    except (ManifestPayloadError, TypeError, ValueError):
        # Canonicalization/serialization of the manifest payload itself
        # (an unsupported field type, a naive datetime, ...) is an
        # expected-failure-shaped bug in the manifest contract, never a
        # storage-layer or DB fault -- normalize it the same path-free way
        # before any file or row is touched.
        raise RefreshVerificationError(
            "analysis_output_manifest_generation_failed",
            "The analysis-output manifest payload could not be generated",
        ) from None
    relative_path = f"research/analysis/{run.id}/output-manifest.json"
    try:
        resolved = store.resolve(relative_path)
        file_already_existed = resolved.exists()
    except (OSError, ValueError):
        raise RefreshVerificationError(
            "analysis_output_manifest_path_unavailable",
            "The analysis-output manifest path could not be checked",
        ) from None
    try:
        written = store.write_bytes(relative_path, envelope_bytes)
    except (OSError, ValueError):
        raise RefreshVerificationError(
            "analysis_output_manifest_write_failed",
            "The analysis-output manifest could not be written",
        ) from None
    try:
        with transaction.atomic():
            DataAsset.objects.create(
                provider="stanstock",
                kind=ANALYSIS_OUTPUT_MANIFEST_KIND,
                subject=str(run.id),
                relative_path=written.relative_path,
                sha256=written.sha256,
                retrieved_at=retrieved_at,
                available_at=retrieved_at,
                metadata={"usage_scope": PRIVATE_USAGE_SCOPE},
            )
    except Exception:
        if not file_already_existed:
            # A cleanup fault here (e.g. an unexpected permission error on
            # unlink) must never replace the original exception being
            # handled: swallow only this narrow best-effort cleanup step,
            # never the failure that actually caused it.
            _safe_unlink(store, relative_path)
        raise
    return relative_path


def _finalize_observed_manifest(
    *,
    run: AnalysisRun,
    universe_snapshot: UniverseSnapshot,
    results: list[PersistedAnalysis],
    plan: ManifestPlan,
    store: AssetStore,
    generated_at: datetime,
) -> str | None:
    """Write and register the analysis-output manifest for `run` if, and
    only if, `universe_snapshot` is OBSERVED-grade -- every observed
    `AnalysisRun` must have exactly one manifest, and a research-grade run
    must never carry one.

    Before registering anything, the exact rows this call actually
    persisted (`results`) must match the `plan` computed *before* the first
    write, by count as well as by key: a defensive invariant that should
    never trip in a correctly-behaving run, but must fail loudly rather
    than silently register a manifest that disagrees with its own plan.
    """
    if universe_snapshot.grade != UniverseSnapshot.Grade.OBSERVED:
        return None
    actual = actual_output_plan(
        (persisted.analysis for persisted in results),
        (prediction for persisted in results for prediction in persisted.predictions),
    )
    if actual != plan:
        raise ValueError("Observed analysis output does not match its precomputed output plan")
    return _write_analysis_output_manifest(
        run=run,
        results=results,
        plan=plan,
        store=store,
        retrieved_at=generated_at,
    )


def _safe_unlink(store: AssetStore, relative_path: str) -> None:
    """Best-effort cleanup of a file this call itself just wrote, only
    used when an observed run's own transaction (including its own
    atomic-exit/commit) later fails. A cleanup fault here (e.g. an
    unexpected permission error) must never replace the original exception
    already being propagated."""
    try:
        store.resolve(relative_path).unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


def analyze_listing(
    *,
    listing: Listing,
    universe_snapshot: UniverseSnapshot,
    decision_time: datetime | None = None,
    target_date: date | None = None,
    issued_on_time: bool | None = None,
    provider: str = "synthetic_demo",
    subject: str | None = None,
    benchmark_subject: str | None = None,
    store: AssetStore | None = None,
    config_path: Path | None = None,
    sample_support: dict[str, int] | None = None,
) -> PersistedAnalysis:
    """Analyze one listing against `universe_snapshot`.

    Every OBSERVED-grade `universe_snapshot` produces exactly one
    analysis-output manifest, the same guarantee `analyze_snapshot` gives a
    whole run -- this is the single-listing entry point, so its own
    precomputed plan is always the simple "decision predictions only, one
    listing" case (never medium/long advisory lanes, which only
    `analyze_snapshot` can activate).

    The whole body runs inside one explicit `with transaction.atomic():`
    wrapped by this outer, undecorated function's own `try/except`: this
    (not a bare `@transaction.atomic` decorator) is what lets the outer
    `except` also catch a failure from the atomic block's own exit
    (commit or savepoint release), not only a failure raised by code inside
    the block, so a manifest file can never be orphaned by either kind of
    failure.
    """
    require_stock_research_listing(listing, operation="Stock analysis")
    generated_at = decision_time or timezone.now()
    logical_target_date = target_date or generated_at.date()
    _validate_snapshot_for_target(universe_snapshot, logical_target_date)
    config = load_scoring_config(config_path)
    if config.version == V3_VERSION:
        if issued_on_time is True:
            raise ValueError("us-price-baseline-v3 on-time issuance requires analyze_snapshot")
        run_issued_on_time = False
    else:
        run_issued_on_time = _issued_on_time(
            universe_snapshot,
            generated_at=generated_at,
            target_date=logical_target_date,
            explicit=issued_on_time,
        )
    data_cutoff = _analysis_data_cutoff(
        generated_at,
        logical_target_date,
        issued_on_time=run_issued_on_time,
    )
    if not UniverseMembership.objects.filter(
        snapshot=universe_snapshot,
        listing=listing,
        eligible=True,
    ).exists():
        raise ValueError(
            f"Listing {listing.pk} is not an eligible member of snapshot {universe_snapshot.pk}"
        )
    digest = config_hash(config)
    revision = code_revision()
    asset_store = store or open_asset_store()
    asof = AsOfData(generated_at, asset_store)
    manifest_relative_path: str | None = None
    persisted: PersistedAnalysis
    try:
        with transaction.atomic():
            run = _create_analysis_run(
                generated_at=generated_at,
                data_cutoff=data_cutoff,
                target_date=logical_target_date,
                issued_on_time=run_issued_on_time,
                universe_snapshot=universe_snapshot,
                config=config,
                config_hash_value=digest,
                code_revision_value=revision,
            )
            computation = _compute_listing_from_asof(
                listing=listing,
                asof=asof,
                provider=provider,
                config=config,
                decision_time=data_cutoff,
                issued_on_time=run_issued_on_time,
                provider_plan=_resolve_provider_plan(provider),
                code_revision_value=revision,
                subject=subject,
                benchmark_subject=benchmark_subject,
                sample_support=sample_support,
                target_date=logical_target_date,
            )
            plan = build_output_plan(
                eligible_listing_ids={listing.pk},
                decision_horizons=frozenset(config.supported_horizons),
                medium_active=False,
                long_active=False,
            )
            persisted = _persist_listing_analysis(
                run=run,
                listing=listing,
                computation=computation,
                generated_at=generated_at,
                data_cutoff=data_cutoff,
                model_version=_model_version(config.version, run.id.hex),
                config_hash_value=digest,
                code_revision_value=revision,
            )
            manifest_relative_path = _finalize_observed_manifest(
                run=run,
                universe_snapshot=universe_snapshot,
                results=[persisted],
                plan=plan,
                store=asset_store,
                generated_at=generated_at,
            )
    except Exception:
        if manifest_relative_path is not None:
            _safe_unlink(asset_store, manifest_relative_path)
        raise
    return persisted


def analyze_snapshot(
    *,
    universe_snapshot: UniverseSnapshot,
    decision_time: datetime | None = None,
    target_date: date | None = None,
    issued_on_time: bool | None = None,
    provider: str = "synthetic_demo",
    benchmark_subject: str | None = None,
    store: AssetStore | None = None,
    config_path: Path | None = None,
    medium_forecast_config_path: Path | None = None,
    long_forecast_config_path: Path | None = None,
    sample_support: dict[str, int] | None = None,
    output_paths: AnalysisOutputPaths | None = None,
    long_forecast_requested: bool | None = None,
) -> list[PersistedAnalysis]:
    """Analyze every eligible member of `universe_snapshot`.

    An OBSERVED-grade `universe_snapshot` produces exactly one
    analysis-output manifest for the whole run, covering the `AnalysisRun`
    row and every `StockAnalysis`/`Prediction` this call persists. The
    exact plan (eligible listings and the exact prediction key multiset)
    is computed *before* the first `StockAnalysis`/`Prediction` write --
    once the medium/long advisory lanes are gated on or off for this run,
    never inferred afterward from whatever happened to be written -- and
    the manifest is registered only once the actual persisted rows are
    checked to match that plan exactly.

    The whole write path runs inside one explicit
    `with transaction.atomic():` wrapped by this outer, undecorated
    function's own `try/except`: this (not a bare `@transaction.atomic`
    decorator) is what lets the outer `except` also catch a failure from
    the atomic block's own exit (commit or savepoint release), not only a
    failure raised by code inside the block, so a panel or manifest file
    can never be orphaned by either kind of failure.

    `output_paths`, if supplied, is populated with this call's own panel
    and manifest relative paths (or left `None` for whichever this run
    does not write) as soon as each is known -- so a caller that must run
    its own checks after this function already returned successfully can
    read the exact paths this call owns directly, instead of re-deriving
    them with a separate post-write `DataAsset` query.

    `long_forecast_requested`, when supplied by scheduled production, is
    the target-scoped provider gate frozen in the market child's JobRun
    before this function writes output. Other callers retain the existing
    invocation-time ProviderRecord default.
    """
    generated_at = decision_time or timezone.now()
    if long_forecast_requested is not None and not isinstance(long_forecast_requested, bool):
        raise ValueError("long_forecast_requested must be a boolean")
    logical_target_date = target_date or generated_at.date()
    explicit_v2_config: MediumForecastV2Config | None = None
    explicit_v2_digest: str | None = None
    explicit_medium_config_bytes: bytes | None = None
    if medium_forecast_config_path is not None:
        explicit_medium_config_bytes = _read_medium_forecast_config_bytes(
            medium_forecast_config_path,
            explicit=True,
        )
        explicit_medium_text = _decode_medium_forecast_config(
            medium_forecast_config_path,
            explicit_medium_config_bytes,
            explicit=True,
        )
        if _is_medium_v2_candidate(medium_forecast_config_path, explicit_medium_text):
            candidate = _load_medium_forecast_config_bytes(
                medium_forecast_config_path,
                explicit_medium_config_bytes,
                explicit=True,
            )
            if not isinstance(candidate, MediumForecastV2Config):
                raise ValueError("us-price-medium-v2 config is malformed")
            explicit_v2_config = candidate
            explicit_v2_digest = medium_forecast_config_hash(candidate)
    if explicit_v2_config is None:
        _validate_snapshot_for_target(universe_snapshot, logical_target_date)
    config = load_scoring_config(config_path)
    digest = config_hash(config)
    if explicit_v2_config is not None:
        if config.version != "us-price-baseline-v2":
            raise ValueError("us-price-medium-v2 requires scoring config us-price-baseline-v2")
        if digest != _MEDIUM_V2_SCORING_HASH:
            raise ValueError(
                "us-price-medium-v2 requires the reviewed us-price-baseline-v2 config identity"
            )
        if issued_on_time is not False:
            raise ValueError("us-price-medium-v2 requires issued_on_time=False")
        if benchmark_subject != "SPY":
            raise ValueError("us-price-medium-v2 requires benchmark_subject='SPY'")
        if (
            explicit_v2_config.version != MEDIUM_V2_VERSION
            or explicit_v2_digest != MEDIUM_V2_EFFECTIVE_CONFIG_HASH
        ):
            raise ValueError("us-price-medium-v2 config identity mismatch")
        if not isinstance(provider, str) or not provider:
            raise ValueError("us-price-medium-v2 requires a non-empty provider identity")
        run_issued_on_time = False
        data_cutoff = _analysis_data_cutoff(
            generated_at,
            logical_target_date,
            issued_on_time=run_issued_on_time,
        )
        # The immutable explicit config and all scalar arguments have now
        # passed admission. From this point the v2 invocation owns these
        # output slots and clears them on every later failure.
        if output_paths is not None:
            output_paths.panel_relative_path = None
            output_paths.manifest_relative_path = None
    validated_observed_v3_revision: str | None = None
    if config.version == V3_VERSION and issued_on_time is True:
        validated_observed_v3_revision = _validated_observed_v3_revision(
            config=config,
            config_hash_value=digest,
            provider=provider,
            benchmark_subject=benchmark_subject,
        )
    effective_long_forecast_requested: bool | None = None
    if explicit_v2_config is None:
        # Freeze the mutable provider gate before the AnalysisRun, panel, or
        # prediction output is written. The explicit-v2 branch does this only
        # after acquiring its canonical authority inside the outer atomic
        # block.
        effective_long_forecast_requested = (
            ProviderRecord.objects.filter(provider="sec", enabled=True).exists()
            if long_forecast_requested is None
            else long_forecast_requested
        )
    if explicit_v2_config is None and config.version == V3_VERSION and issued_on_time is not True:
        run_issued_on_time = False
    elif explicit_v2_config is None:
        run_issued_on_time = _issued_on_time(
            universe_snapshot,
            generated_at=generated_at,
            target_date=logical_target_date,
            explicit=issued_on_time,
        )
    if explicit_v2_config is None:
        data_cutoff = _analysis_data_cutoff(
            generated_at,
            logical_target_date,
            issued_on_time=run_issued_on_time,
        )
    revision: str | None = None
    asset_store: AssetStore | None = None
    asof: AsOfData | None = None
    memberships: list[UniverseMembership] | None = None
    v2_authority: _LockedMediumV2Authority | None = None
    if explicit_v2_config is None:
        revision = (
            validated_observed_v3_revision
            if validated_observed_v3_revision is not None
            else code_revision()
        )
        asset_store = store or open_asset_store()
        asof = AsOfData(generated_at, asset_store)
        memberships = list(
            UniverseMembership.objects.select_related(
                "listing__security__company",
            ).filter(snapshot=universe_snapshot, eligible=True)
        )
        for membership in memberships:
            require_stock_research_listing(
                membership.listing,
                operation="Snapshot stock analysis",
            )
    results: list[PersistedAnalysis] = []
    panel_relative_path: str | None = None
    panel_file_created_by_invocation = True
    manifest_relative_path: str | None = None
    try:
        with transaction.atomic():
            if explicit_v2_config is not None:
                snapshot_pk = universe_snapshot.pk
                if snapshot_pk is None:
                    raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
                v2_authority = _lock_medium_v2_snapshot_authority(
                    snapshot_pk,
                    target_date=logical_target_date,
                )
                universe_snapshot = v2_authority.snapshot
                memberships = list(v2_authority.memberships)
                effective_long_forecast_requested = (
                    ProviderRecord.objects.filter(provider="sec", enabled=True).exists()
                    if long_forecast_requested is None
                    else long_forecast_requested
                )
                revision = code_revision()
                asset_store = store or open_asset_store()
                asof = AsOfData(generated_at, asset_store)
            assert memberships is not None
            assert effective_long_forecast_requested is not None
            assert revision is not None
            assert asset_store is not None
            assert asof is not None
            run = _create_analysis_run(
                generated_at=generated_at,
                data_cutoff=data_cutoff,
                target_date=logical_target_date,
                issued_on_time=run_issued_on_time,
                universe_snapshot=universe_snapshot,
                config=config,
                config_hash_value=digest,
                code_revision_value=revision,
            )
            if v2_authority is not None:
                v2_authority = replace(v2_authority, run=run)
            model_version = _model_version(config.version, run.id.hex)
            advisory_context: AdvisoryForecastContext | None = None
            long_context: LongForecastContext | None = None
            medium_config = (
                explicit_v2_config
                if explicit_v2_config is not None
                else (
                    _load_medium_forecast_config_bytes(
                        medium_forecast_config_path,
                        explicit_medium_config_bytes,
                        explicit=True,
                    )
                    if medium_forecast_config_path is not None
                    and explicit_medium_config_bytes is not None
                    else load_medium_forecast_config()
                )
            )
            if (
                benchmark_subject is not None
                and config.version in medium_config.enabled_scoring_versions
                and memberships
            ):
                medium_digest = (
                    explicit_v2_digest
                    if explicit_v2_config is not None
                    else medium_forecast_config_hash(medium_config)
                )
                assert medium_digest is not None
                panel = build_medium_forecast_panel(
                    listings=[membership.listing for membership in memberships],
                    asof=asof,
                    provider=provider,
                    benchmark_subject=benchmark_subject,
                    target_date=logical_target_date,
                    generated_at=generated_at,
                    run_id=run.id,
                    config=medium_config,
                    config_hash=medium_digest,
                    scoring_config_version=config.version,
                    scoring_config_hash=digest,
                    universe_snapshot_id=universe_snapshot.id,
                    universe_slug=universe_snapshot.universe.slug,
                    universe_config_hash=universe_snapshot.config_hash,
                    code_revision=revision,
                    store=asset_store,
                    data_cutoff=(
                        data_cutoff if isinstance(medium_config, MediumForecastV2Config) else None
                    ),
                )
                panel_relative_path = panel.asset.relative_path
                panel_file_created_by_invocation = panel.file_created_by_invocation
                if output_paths is not None:
                    output_paths.panel_relative_path = panel_relative_path
                advisory_context = AdvisoryForecastContext(
                    panel=panel,
                    store=asset_store,
                    config=medium_config,
                    config_hash=medium_digest,
                    model_version=_model_version(medium_config.version, run.id.hex),
                    forecasts=build_medium_forecasts(panel.frame, medium_config),
                )
            computations: dict[str, AnalysisComputation] = {}
            provider_plan = _resolve_provider_plan(provider)
            for membership in memberships:
                computation = _compute_listing_from_asof(
                    listing=membership.listing,
                    asof=asof,
                    provider=provider,
                    config=config,
                    decision_time=data_cutoff,
                    issued_on_time=run_issued_on_time,
                    provider_plan=provider_plan,
                    code_revision_value=revision,
                    benchmark_subject=benchmark_subject,
                    sample_support=sample_support,
                    target_date=logical_target_date,
                )
                computations[str(membership.listing.pk)] = computation
            long_config = load_long_forecast_config(long_forecast_config_path)
            if (
                provider == long_config.price_provider == "twelve_data"
                and long_config.fundamentals_provider == "sec"
                and config.version in long_config.enabled_scoring_versions
                and effective_long_forecast_requested
                and memberships
            ):
                current_prices: dict[str, float] = {}
                current_price_assets: dict[str, DataAsset] = {}
                for membership in memberships:
                    listing_id = str(membership.listing.pk)
                    computation = computations[listing_id]
                    if computation.price_asset is None:
                        raise ValueError(
                            f"Long forecast price asset is missing for {membership.listing}"
                        )
                    current_prices[listing_id] = computation.current_price
                    current_price_assets[listing_id] = computation.price_asset
                long_digest = long_forecast_config_hash(long_config)
                long_context = LongForecastContext(
                    config=long_config,
                    config_hash=long_digest,
                    model_version=_model_version(long_config.version, run.id.hex),
                    forecasts=build_long_forecasts(
                        listings=[membership.listing for membership in memberships],
                        asof=asof,
                        data_cutoff=data_cutoff,
                        target_date=logical_target_date,
                        config=long_config,
                        current_prices=current_prices,
                        price_assets=current_price_assets,
                    ),
                )
            plan = build_output_plan(
                eligible_listing_ids={membership.listing.pk for membership in memberships},
                decision_horizons=frozenset(config.supported_horizons),
                medium_active=advisory_context is not None,
                long_active=long_context is not None,
            )
            for membership in memberships:
                computation = computations[str(membership.listing.pk)]
                results.append(
                    _persist_listing_analysis(
                        run=run,
                        listing=membership.listing,
                        computation=computation,
                        generated_at=generated_at,
                        data_cutoff=data_cutoff,
                        model_version=model_version,
                        config_hash_value=digest,
                        code_revision_value=revision,
                        advisory_context=advisory_context,
                        long_context=long_context,
                    )
                )
            if advisory_context is not None:
                first, *additional = results
                additional_requests = tuple(
                    _AdditionalAdvisoryPredictionRequest(
                        analysis=persisted.analysis,
                        forecasts=advisory_context.forecasts[str(persisted.analysis.listing_id)],
                        source_assets=persisted.computation.source_assets,
                    )
                    for persisted in additional
                )
                if v2_authority is None:
                    appended = append_advisory_predictions(
                        analysis=first.analysis,
                        forecasts=advisory_context.forecasts[str(first.analysis.listing_id)],
                        panel_asset=advisory_context.panel.asset,
                        store=advisory_context.store,
                        generated_at=generated_at,
                        data_cutoff=data_cutoff,
                        issued_on_time=run.issued_on_time,
                        model_version=advisory_context.model_version,
                        config_hash_value=advisory_context.config_hash,
                        source_assets=first.computation.source_assets,
                        code_revision_value=revision,
                        _additional_requests=additional_requests,
                    )
                else:
                    appended = _append_advisory_predictions_with_authority(
                        analysis=first.analysis,
                        forecasts=advisory_context.forecasts[str(first.analysis.listing_id)],
                        panel_asset=advisory_context.panel.asset,
                        store=advisory_context.store,
                        generated_at=generated_at,
                        data_cutoff=data_cutoff,
                        issued_on_time=run.issued_on_time,
                        model_version=advisory_context.model_version,
                        config_hash_value=advisory_context.config_hash,
                        source_assets=first.computation.source_assets,
                        code_revision_value=revision,
                        _additional_requests=additional_requests,
                        authority=_LockedAdvisoryAuthority(
                            analyses=tuple(persisted.analysis for persisted in results),
                            panels_by_run=MappingProxyType({run.pk: v2_authority}),
                        ),
                        medium_v2_config=explicit_v2_config,
                    )
                predictions_per_listing = 2
                decision_count = len(config.supported_horizons)
                results = [
                    replace(
                        persisted,
                        predictions=(
                            *persisted.predictions[:decision_count],
                            *appended[
                                index * predictions_per_listing : (index + 1)
                                * predictions_per_listing
                            ],
                            *persisted.predictions[decision_count:],
                        ),
                    )
                    for index, persisted in enumerate(results)
                ]
            manifest_relative_path = _finalize_observed_manifest(
                run=run,
                universe_snapshot=universe_snapshot,
                results=results,
                plan=plan,
                store=asset_store,
                generated_at=generated_at,
            )
            if output_paths is not None:
                output_paths.manifest_relative_path = manifest_relative_path
    except Exception:
        if (
            asset_store is not None
            and panel_relative_path is not None
            and panel_file_created_by_invocation
        ):
            _safe_unlink(asset_store, panel_relative_path)
        if asset_store is not None and manifest_relative_path is not None:
            _safe_unlink(asset_store, manifest_relative_path)
        if explicit_v2_config is not None and output_paths is not None:
            output_paths.panel_relative_path = None
            output_paths.manifest_relative_path = None
        raise
    return results


def append_predictions(
    *,
    analysis: StockAnalysis,
    computation: AnalysisComputation,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    supported_horizons: tuple[str, ...],
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, Any]],
    code_revision_value: str,
) -> tuple[Prediction, ...]:
    require_stock_research_listing(
        analysis.listing,
        operation="Stock prediction issuance",
    )
    if issued_on_time and (
        not analysis.run.issued_on_time or generated_at != analysis.run.generated_at
    ):
        raise ValueError(
            "Only predictions created with the original on-time analysis may be marked on time"
        )
    unsupported = set(supported_horizons) - {
        Prediction.Horizon.SHORT.value,
        Prediction.Horizon.MEDIUM.value,
        Prediction.Horizon.LONG.value,
    }
    if unsupported:
        raise ValueError(
            "Decision prediction issuance accepts scoring-group horizons only: "
            + ", ".join(sorted(unsupported))
        )
    horizons = tuple(Prediction.Horizon(value) for value in supported_horizons)
    if not horizons:
        raise ValueError("At least one supported prediction horizon is required")
    return tuple(
        _create_prediction(
            analysis=analysis,
            horizon=horizon,
            scenario=computation.scenarios[horizon],
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            issued_on_time=issued_on_time,
            model_version=model_version,
            config_hash_value=config_hash_value,
            source_assets=source_assets,
            code_revision_value=code_revision_value,
        )
        for horizon in horizons
    )


def append_advisory_predictions(
    *,
    analysis: StockAnalysis,
    forecasts: dict[str, MediumForecast],
    panel_asset: DataAsset,
    store: AssetStore | None = None,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, Any]],
    code_revision_value: str,
    _additional_requests: tuple[_AdditionalAdvisoryPredictionRequest, ...] = (),
) -> tuple[Prediction, ...]:
    with transaction.atomic():
        supplied_requests = (
            _AdditionalAdvisoryPredictionRequest(
                analysis=analysis,
                forecasts=forecasts,
                source_assets=source_assets,
            ),
            *_additional_requests,
        )
        authority = _lock_advisory_authority(
            tuple(request.analysis for request in supplied_requests)
        )
        return _append_advisory_predictions_with_authority(
            analysis=analysis,
            forecasts=forecasts,
            panel_asset=panel_asset,
            store=store,
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            issued_on_time=issued_on_time,
            model_version=model_version,
            config_hash_value=config_hash_value,
            source_assets=source_assets,
            code_revision_value=code_revision_value,
            _additional_requests=_additional_requests,
            authority=authority,
        )


def _append_advisory_predictions_with_authority(
    *,
    analysis: StockAnalysis,
    forecasts: dict[str, MediumForecast],
    panel_asset: DataAsset,
    store: AssetStore | None,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, Any]],
    code_revision_value: str,
    _additional_requests: tuple[_AdditionalAdvisoryPredictionRequest, ...],
    authority: _LockedAdvisoryAuthority,
    medium_v2_config: MediumForecastV2Config | None = None,
) -> tuple[Prediction, ...]:
    """Append a complete batch using authority held by the caller's transaction."""
    supplied_requests = (
        _AdditionalAdvisoryPredictionRequest(
            analysis=analysis,
            forecasts=forecasts,
            source_assets=source_assets,
        ),
        *_additional_requests,
    )
    if len(supplied_requests) != len(authority.analyses):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    requests = tuple(
        _AdditionalAdvisoryPredictionRequest(
            analysis=locked_analysis,
            forecasts=request.forecasts,
            source_assets=request.source_assets,
        )
        for request, locked_analysis in zip(
            supplied_requests,
            authority.analyses,
            strict=True,
        )
    )
    primary_analysis = requests[0].analysis
    expected_horizons = {
        Prediction.Horizon.SIX_MONTH.value,
        Prediction.Horizon.TWELVE_MONTH.value,
    }
    panel_payload = _asset_payload(panel_asset)
    advisory_sources: list[list[dict[str, Any]]] = []
    for request in requests:
        require_stock_research_listing(
            request.analysis.listing,
            operation="Advisory forecast issuance",
        )
        if issued_on_time and (
            not request.analysis.run.issued_on_time
            or generated_at != request.analysis.run.generated_at
        ):
            raise ValueError(
                "Only forecasts created with the original on-time analysis may be marked on time"
            )
        if set(request.forecasts) != expected_horizons:
            raise ValueError("Advisory forecast issuance requires exactly 6m and 12m")
        advisory_sources.append([*request.source_assets, panel_payload])

    v2_request = any(
        _is_medium_v2_prediction(
            forecast=forecast,
            model_version=model_version,
            config_hash_value=config_hash_value,
            panel_asset=panel_asset,
        )
        for request in requests
        for forecast in request.forecasts.values()
    )
    panel_attestation: _MediumV2PanelAttestation | None = None
    if v2_request:
        if any(request.analysis.run_id != primary_analysis.run_id for request in requests):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        panel_authority = authority.panels_by_run.get(primary_analysis.run_id)
        if panel_authority is None:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        _validate_locked_medium_v2_authority(
            panel_authority,
            requested_listing_ids=tuple(request.analysis.listing_id for request in requests),
            detailed_errors=False,
        )
        try:
            panel_attestation = _attest_medium_v2_panel_with_authority(
                run=primary_analysis.run,
                panel_asset=panel_asset,
                store=store or open_asset_store(),
                authority=panel_authority,
                config=medium_v2_config,
                config_hash_value=(config_hash_value if medium_v2_config is not None else None),
            )
        except (
            KeyError,
            OSError,
            PriceFrameChecksumMismatchError,
            RefreshVerificationError,
            TypeError,
            ValueError,
            pl.exceptions.PolarsError,
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR) from None

    if issued_on_time:
        for sources in advisory_sources:
            _validate_on_time_source_assets(sources, data_cutoff=data_cutoff)
    prediction_kwargs = tuple(
        _advisory_prediction_kwargs(
            analysis=request.analysis,
            horizon=Prediction.Horizon(horizon),
            forecast=request.forecasts[horizon],
            panel_asset=panel_asset,
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            issued_on_time=issued_on_time,
            model_version=model_version,
            config_hash_value=config_hash_value,
            source_assets=sources,
            code_revision_value=code_revision_value,
            require_v2=v2_request,
        )
        for request, sources in zip(requests, advisory_sources, strict=True)
        for horizon in (
            Prediction.Horizon.SIX_MONTH.value,
            Prediction.Horizon.TWELVE_MONTH.value,
        )
    )
    if panel_attestation is not None:
        for request in requests:
            for horizon in (
                Prediction.Horizon.SIX_MONTH.value,
                Prediction.Horizon.TWELVE_MONTH.value,
            ):
                _require_medium_v2_forecast_matches_panel(
                    analysis=request.analysis,
                    horizon=horizon,
                    forecast=request.forecasts[horizon],
                    panel_asset=panel_asset,
                    panel_attestation=panel_attestation,
                )

    # Validation and panel replay for the complete batch finish before the
    # first insert. The caller's atomic block also covers every late
    # uniqueness/constraint failure, including additional requests.
    return tuple(Prediction.objects.create(**kwargs) for kwargs in prediction_kwargs)


def append_long_advisory_predictions(
    *,
    analysis: StockAnalysis,
    forecasts: dict[str, LongForecast],
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    code_revision_value: str,
) -> tuple[Prediction, ...]:
    require_stock_research_listing(
        analysis.listing,
        operation="Long advisory forecast issuance",
    )
    if issued_on_time and (
        not analysis.run.issued_on_time or generated_at != analysis.run.generated_at
    ):
        raise ValueError(
            "Only forecasts created with the original on-time analysis may be marked on time"
        )
    expected_horizons = {
        Prediction.Horizon.THREE_YEAR.value,
        Prediction.Horizon.FIVE_YEAR.value,
    }
    if set(forecasts) != expected_horizons:
        raise ValueError("Long advisory forecast issuance requires exactly 3y and 5y")
    return tuple(
        _create_long_advisory_prediction(
            analysis=analysis,
            horizon=Prediction.Horizon(horizon),
            forecast=forecasts[horizon],
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            issued_on_time=issued_on_time,
            model_version=model_version,
            config_hash_value=config_hash_value,
            code_revision_value=code_revision_value,
        )
        for horizon in (
            Prediction.Horizon.THREE_YEAR.value,
            Prediction.Horizon.FIVE_YEAR.value,
        )
    )


def _create_stock_analysis(
    run: AnalysisRun,
    listing: Listing,
    computation: AnalysisComputation,
    *,
    advisory_forecasts: dict[str, MediumForecast] | None = None,
    long_forecasts: dict[str, LongForecast] | None = None,
) -> StockAnalysis:
    supported_horizons = {str(value) for value in computation.data_quality["supported_horizons"]}
    scenario_payloads = {
        horizon: scenario.as_dict() for horizon, scenario in computation.scenarios.items()
    }
    scenario_payloads.update(
        {
            horizon: forecast.scenario_payload()
            for horizon, forecast in (advisory_forecasts or {}).items()
        }
    )
    scenario_payloads.update(
        {
            horizon: {
                **forecast.scenario_payload(),
                "evidence_grade": _long_forecast_evidence_grade(run),
            }
            for horizon, forecast in (long_forecasts or {}).items()
        }
    )
    return StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=_decimal(computation.current_price, places=6),
        daily_change=_optional_decimal(computation.daily_change, places=6),
        overall_score=_decimal(computation.aggregate.overall, places=2),
        recommendation=computation.recommendation,
        risk_score=_optional_decimal(computation.risk_score, places=2),
        risk_class=computation.risk_class,
        confidence=_decimal(computation.aggregate.confidence, places=2),
        confidence_status=computation.aggregate.confidence_status,
        component_scores={
            "components": computation.aggregate.component_scores.components,
            "horizons": {
                horizon: score
                for horizon, score in computation.aggregate.horizon_scores.items()
                if horizon in supported_horizons
            },
            "factors": computation.aggregate.component_scores.factor_scores,
        },
        forecast_scenarios=build_forecast_scenario_document(scenario_payloads),
        short_scenario=computation.scenarios["short"].as_dict(),
        medium_scenario=computation.scenarios["medium"].as_dict(),
        long_scenario=computation.scenarios["long"].as_dict(),
        reasons=computation.reasons,
        risks=computation.risks,
        data_quality=computation.data_quality,
    )


def _create_prediction(
    *,
    analysis: StockAnalysis,
    horizon: Prediction.Horizon,
    scenario: Scenario,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, Any]],
    code_revision_value: str,
) -> Prediction:
    horizon_value = str(horizon)
    price_provider, price_subject = _prediction_price_source(analysis, source_assets)
    return Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=generated_at,
        target_date=analysis.run.target_date,
        issued_on_time=issued_on_time,
        horizon=horizon,
        evidence_role=Prediction.EvidenceRole.DECISION,
        evidence_grade=analysis.run.universe_snapshot.grade,
        source_mode=source_data_mode({"source_assets": source_assets}),
        price_provider=price_provider,
        price_subject=price_subject,
        price_at_prediction=analysis.current_price,
        bear_return=_optional_decimal(scenario.bear, places=4),
        base_return=_optional_decimal(scenario.base, places=4),
        bull_return=_optional_decimal(scenario.bull, places=4),
        probability_positive=_optional_decimal(scenario.probability_positive, places=4),
        confidence=_decimal(scenario.confidence, places=2),
        confidence_status=scenario.confidence_status,
        insufficiency_reason=scenario.insufficiency_reason,
        recommendation=analysis.recommendation,
        overall_score=analysis.overall_score,
        component_scores=analysis.component_scores,
        model_version=model_version,
        method_version=analysis.run.config_version,
        config_hash=config_hash_value,
        data_cutoff=data_cutoff,
        source_assets=source_assets,
        calculation={
            "schema_version": 1,
            "method": scenario.method,
            "method_version": analysis.run.config_version,
            "prediction_version": model_version,
            "config_hash": config_hash_value,
            "forecast_horizon": horizon_value,
            "score_group": horizon_value,
            "support": {
                "confidence": scenario.confidence,
                "confidence_status": scenario.confidence_status,
                "insufficiency_reason": scenario.insufficiency_reason,
            },
            "formula_inputs": {
                "scenario": {
                    "bear": scenario.bear,
                    "base": scenario.base,
                    "bull": scenario.bull,
                    "probability_positive": scenario.probability_positive,
                },
                "overall_score": float(analysis.overall_score),
                "risk_score": (
                    float(analysis.risk_score) if analysis.risk_score is not None else None
                ),
            },
            "contribution_detail": analysis.component_scores,
            "return_basis": analysis.data_quality.get("return_definition"),
            "dividends_included": analysis.data_quality.get("dividends_included"),
            "evidence_grade": analysis.run.universe_snapshot.grade,
            "price_subject": price_subject,
        },
        code_revision=code_revision_value,
    )


def _is_medium_v2_prediction(
    *,
    forecast: MediumForecast,
    model_version: str,
    config_hash_value: str,
    panel_asset: DataAsset,
) -> bool:
    calculation = forecast.calculation
    metadata = panel_asset.metadata if isinstance(panel_asset.metadata, dict) else {}
    probability_evidence = calculation.get("probability_evidence")
    training_evidence = calculation.get("training_evidence")
    return bool(
        calculation.get("schema_version") == 2
        or calculation.get("method_version") == MEDIUM_V2_VERSION
        or config_hash_value == MEDIUM_V2_EFFECTIVE_CONFIG_HASH
        or model_version.startswith(f"{MEDIUM_V2_VERSION}-")
        or metadata.get("method_version") == MEDIUM_V2_VERSION
        or "predictive_distribution" in calculation
        or "evidence" in calculation
        or isinstance(probability_evidence, Mapping)
        and bool({"status", "reasons"} & set(probability_evidence))
        or isinstance(training_evidence, Mapping)
        and bool(
            {
                "test_policy",
                "aggregation_policy",
                "calibration_claim",
                "significance_claim",
                "profitability_claim",
                "alpha_claim",
            }
            & set(training_evidence)
        )
        or forecast.scenario.confidence_status == "empirical_skill_supported"
    )


def _validate_medium_v2_prediction(
    *,
    analysis: StockAnalysis,
    horizon: Prediction.Horizon,
    forecast: MediumForecast,
    panel_asset: DataAsset,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, Any]],
    code_revision_value: str,
    calculation: Mapping[str, Any],
    persisted: bool,
) -> None:
    """Validate the complete v2 contract independently at both writers."""
    if not persisted and set(calculation) & _MEDIUM_V2_SERVICE_KEYS:
        raise ValueError(
            "us-price-medium-v2 calculator payload must not supply service-owned provenance"
        )
    expected_keys = (
        _MEDIUM_V2_CALCULATOR_KEYS | _MEDIUM_V2_SERVICE_KEYS
        if persisted
        else _MEDIUM_V2_CALCULATOR_KEYS
    )
    if set(calculation) != expected_keys:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    horizon_value = str(horizon)
    expected_sessions = {"6m": 126, "12m": 252}
    run = analysis.run
    snapshot = run.universe_snapshot
    metadata = panel_asset.metadata
    if (
        horizon_value not in expected_sessions
        or calculation.get("schema_version") != 2
        or calculation.get("method") != "conditional_empirical_price"
        or calculation.get("method_version") != MEDIUM_V2_VERSION
        or calculation.get("forecast_horizon") != horizon_value
        or calculation.get("horizon_sessions") != expected_sessions[horizon_value]
        or calculation.get("return_basis") != "split_adjusted_price_return"
        or calculation.get("dividends_included") is not False
        or config_hash_value != MEDIUM_V2_EFFECTIVE_CONFIG_HASH
        or model_version != _model_version(MEDIUM_V2_VERSION, run.id.hex)
        or generated_at != run.generated_at
        or data_cutoff != run.data_cutoff
        or code_revision_value != run.code_revision
        or issued_on_time is not False
        or run.issued_on_time is not False
        or snapshot.grade != UniverseSnapshot.Grade.RESEARCH
        or panel_asset.provider != "stanstock"
        or panel_asset.kind != "medium_forecast_panel"
        or panel_asset.subject != str(run.id)
        or not isinstance(metadata, dict)
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    if (
        metadata.get("schema_version") != 1
        or metadata.get("method_version") != MEDIUM_V2_VERSION
        or metadata.get("config_hash") != MEDIUM_V2_EFFECTIVE_CONFIG_HASH
        or metadata.get("scoring_config_version") != run.config_version
        or metadata.get("scoring_config_hash") != run.config_hash
        or run.config_version != "us-price-baseline-v2"
        or run.config_hash != _MEDIUM_V2_SCORING_HASH
        or metadata.get("target_date") != run.target_date.isoformat()
        or metadata.get("universe_snapshot_id") != str(snapshot.id)
        or metadata.get("universe_config_hash") != snapshot.config_hash
        or metadata.get("code_revision") != run.code_revision
        or metadata.get("return_definition") != "split_adjusted_price_return"
        or metadata.get("dividends_included") is not False
        or metadata.get("training_evidence_grade") != "research"
        or metadata.get("current_universe_survivorship_bias") is not True
        or metadata.get("content_sha256") != panel_asset.sha256
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    source_manifest = metadata.get("source_assets")
    if (
        not isinstance(source_manifest, list)
        or not source_manifest
        or metadata.get("source_manifest_hash") != hash_json(source_manifest)
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    for source in source_manifest:
        if not isinstance(source, dict) or set(source) != {
            "id",
            "provider",
            "kind",
            "subject",
            "relative_path",
            "sha256",
            "retrieved_at",
            "available_at",
        }:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        if (
            not all(
                isinstance(source[field], str) and bool(source[field])
                for field in (
                    "id",
                    "provider",
                    "kind",
                    "subject",
                    "relative_path",
                    "sha256",
                    "retrieved_at",
                    "available_at",
                )
            )
            or source["kind"] != "price_history"
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        try:
            available_at = datetime.fromisoformat(source["available_at"])
            datetime.fromisoformat(source["retrieved_at"])
            if available_at > run.data_cutoff:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR) from None
    analysis_price_source = analysis.data_quality.get("price_source")
    analysis_sources = analysis.data_quality.get("source_assets")
    listing_subject = analysis.listing.provider_symbol or analysis.listing.ticker
    if (
        not isinstance(analysis_price_source, Mapping)
        or not isinstance(analysis_sources, list)
        or analysis_price_source.get("subject") != listing_subject
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    for expected_subject in (listing_subject, "SPY"):
        manifest_matches = [
            source
            for source in source_manifest
            if source["provider"] == analysis_price_source.get("provider")
            and source["kind"] == "price_history"
            and source["subject"] == expected_subject
        ]
        analysis_matches = [
            source
            for source in analysis_sources
            if isinstance(source, dict)
            and source.get("provider") == analysis_price_source.get("provider")
            and source.get("kind") == "price_history"
            and source.get("subject") == expected_subject
        ]
        if len(manifest_matches) != 1 or len(analysis_matches) != 1:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        if any(
            analysis_matches[0].get(field) != manifest_matches[0].get(field)
            for field in (
                "id",
                "provider",
                "kind",
                "subject",
                "relative_path",
                "sha256",
                "retrieved_at",
                "available_at",
            )
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        if expected_subject == listing_subject and manifest_matches[0][
            "id"
        ] != analysis_price_source.get("asset_id"):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    panel_sources = [
        source
        for source in source_assets
        if isinstance(source, dict)
        and source.get("id") == str(panel_asset.pk)
        and source.get("sha256") == panel_asset.sha256
        and source.get("provider") == panel_asset.provider
        and source.get("kind") == panel_asset.kind
        and source.get("subject") == panel_asset.subject
    ]
    all_panel_sources = [
        source
        for source in source_assets
        if isinstance(source, dict)
        and (source.get("provider") == "stanstock" or source.get("kind") == "medium_forecast_panel")
    ]
    if len(panel_sources) != 1 or all_panel_sources != panel_sources:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    manifest_by_id = {
        str(source["id"]): source for source in source_manifest if isinstance(source, dict)
    }
    for source in source_assets:
        if not isinstance(source, dict) or source in panel_sources:
            continue
        manifest_source = manifest_by_id.get(str(source.get("id")))
        if manifest_source is None or any(
            source.get(field) != manifest_source.get(field)
            for field in (
                "provider",
                "kind",
                "subject",
                "relative_path",
                "sha256",
                "retrieved_at",
                "available_at",
            )
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    current_state = _exact_medium_v2_mapping(
        calculation.get("current_state"),
        {
            "anchor_date",
            "relative_momentum",
            "drawdown",
            "volatility",
            "market_trend",
            "market_volatility",
            "relative_momentum_bucket",
            "drawdown_bucket",
            "volatility_bucket",
            "market_trend_bucket",
            "market_volatility_bucket",
            "close_vs_sma_50",
            "close_vs_sma_200",
            "downside_volatility",
            "average_dollar_volume",
        },
    )
    _validate_medium_v2_current_state(
        current_state,
        expected_anchor=run.target_date.isoformat(),
    )
    support = _exact_medium_v2_mapping(
        calculation.get("support"),
        {
            "raw_matches",
            "effective_cohorts",
            "distinct_listings",
            "calendar_start",
            "calendar_end",
            "market_regimes",
            "fallback_level",
            "shrinkage_weight",
            "dispersion",
        },
    )
    _validate_medium_v2_support(support)
    probability_evidence = _exact_medium_v2_mapping(
        calculation.get("probability_evidence"),
        {
            "status",
            "reasons",
            "calendar_span_days",
            "distinct_matched_market_regimes",
            "distinct_panel_market_regimes",
            "minimum_effective_cohorts",
            "minimum_distinct_listings",
            "minimum_calendar_span_days",
            "minimum_distinct_market_regimes",
        },
    )
    if (
        probability_evidence["status"] not in {"published", "withheld", "not_evaluable"}
        or not isinstance(probability_evidence["reasons"], list)
        or not all(
            isinstance(reason, str) and bool(reason) for reason in probability_evidence["reasons"]
        )
        or not all(
            _medium_v2_nonnegative_int(probability_evidence[key])
            for key in (
                "minimum_effective_cohorts",
                "minimum_distinct_listings",
                "minimum_calendar_span_days",
                "minimum_distinct_market_regimes",
            )
        )
        or not all(
            probability_evidence[key] is None
            or _medium_v2_nonnegative_int(probability_evidence[key])
            for key in (
                "calendar_span_days",
                "distinct_matched_market_regimes",
                "distinct_panel_market_regimes",
            )
        )
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    predictive = _exact_medium_v2_mapping(
        calculation.get("predictive_distribution"),
        {
            "kind",
            "cdf_event",
            "positive_event",
            "quantile_convention",
            "overlap_policy",
            "matched",
            "unconditional",
            "p20",
            "p50",
            "p80",
            "probability_positive_raw",
            "probability_positive_published",
        },
    )
    if (
        predictive["kind"] != "cohort_equal_empirical_cdf_mixture"
        or predictive["cdf_event"] != "return_lte_x"
        or predictive["positive_event"] != "return_gt_0"
        or predictive["quantile_convention"] != "left_inverse_first_cdf_ge_q"
        or predictive["overlap_policy"] != "matched_rows_receive_mass_in_both_normalized_components"
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    component_keys = {
        "component_mass",
        "normalized_mass",
        "p20",
        "p50",
        "p80",
        "probability_positive_raw",
        "raw_observations",
        "effective_cohorts",
        "distinct_listings",
        "calendar_start",
        "calendar_end",
        "market_regimes",
        "dispersion",
    }
    matched = _exact_medium_v2_mapping(predictive["matched"], component_keys)
    unconditional = _exact_medium_v2_mapping(predictive["unconditional"], component_keys)
    for component in (matched, unconditional):
        if (
            not _medium_v2_nonnegative_int(component["raw_observations"])
            or not _medium_v2_nonnegative_int(component["effective_cohorts"])
            or not _medium_v2_nonnegative_int(component["distinct_listings"])
            or not isinstance(component["market_regimes"], list)
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    evidence = _exact_medium_v2_mapping(
        calculation.get("evidence"),
        {"base_accuracy", "probability_skill", "interval"},
    )
    base_accuracy = _exact_medium_v2_mapping(
        evidence["base_accuracy"],
        {
            "status",
            "test_origins",
            "test_predictions",
            "weighting",
            "mean_absolute_error",
            "unconditional_mean_absolute_error",
            "spy_relative_mean_absolute_error",
            "spy_relative_baseline_method",
            "maximum_baseline_mae_ratio",
        },
    )
    probability_skill = _exact_medium_v2_mapping(
        evidence["probability_skill"],
        {
            "status",
            "test_origins",
            "test_predictions",
            "weighting",
            "event",
            "model_brier_score",
            "reference_brier_score",
            "brier_skill_score",
            "reference_method",
            "minimum_brier_skill_exclusive",
            "zero_reference_policy",
        },
    )
    interval = _exact_medium_v2_mapping(
        evidence["interval"],
        {
            "status",
            "test_origins",
            "test_predictions",
            "weighting",
            "alpha",
            "nominal_coverage",
            "endpoint_policy",
            "empirical_coverage",
            "below_rate",
            "above_rate",
            "mean_width",
            "model_mean_interval_score",
            "reference_mean_interval_score",
            "reference_method",
        },
    )
    if (
        probability_skill["event"] != "return_gt_0"
        or probability_skill["reference_method"] != "prequential_unconditional"
        or not _medium_v2_exact_number(probability_skill["minimum_brier_skill_exclusive"], 0.0)
        or probability_skill["zero_reference_policy"] != "null_no_epsilon"
        or not _medium_v2_exact_number(interval["alpha"], 0.4)
        or not _medium_v2_exact_number(interval["nominal_coverage"], 0.6)
        or interval["endpoint_policy"] != "inclusive"
        or interval["reference_method"] != "prequential_unconditional"
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    _validate_medium_v2_evidence(
        base_accuracy=base_accuracy,
        probability_skill=probability_skill,
        interval=interval,
    )

    formula_inputs = _exact_medium_v2_mapping(calculation.get("formula_inputs"), {"scenario"})
    formula_scenario = _exact_medium_v2_mapping(
        formula_inputs["scenario"],
        {"bear", "base", "bull", "probability_positive"},
    )
    training = _exact_medium_v2_mapping(
        calculation.get("training_evidence"),
        {
            "grade",
            "current_universe_survivorship_bias",
            "label_policy",
            "test_policy",
            "cohort_policy",
            "aggregation_policy",
            "calibration_claim",
            "significance_claim",
            "profitability_claim",
            "alpha_claim",
        },
    )
    if training != {
        "grade": "research",
        "current_universe_survivorship_bias": True,
        "label_policy": "training_label_end_date_lte_origin",
        "test_policy": "test_outcome_never_enters_its_origin_training_or_gates",
        "cohort_policy": "fixed_epoch_non_overlapping",
        "aggregation_policy": "date_equal_listing_equal_within_origin",
        "calibration_claim": False,
        "significance_claim": False,
        "profitability_claim": False,
        "alpha_claim": False,
    }:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    _validate_medium_v2_state(
        horizon=horizon_value,
        forecast=forecast,
        support=support,
        predictive=predictive,
        matched=matched,
        unconditional=unconditional,
        formula_scenario=formula_scenario,
        probability_evidence=probability_evidence,
        probability_skill=probability_skill,
    )
    if persisted:
        price_provider, price_subject = _prediction_price_source(analysis, source_assets)
        if (
            calculation.get("config_hash") != MEDIUM_V2_EFFECTIVE_CONFIG_HASH
            or calculation.get("prediction_version") != model_version
            or calculation.get("panel_asset_id") != str(panel_asset.pk)
            or calculation.get("panel_sha256") != panel_asset.sha256
            or calculation.get("evidence_grade") != "research"
            or calculation.get("price_subject") != price_subject
            or not price_provider
            or not price_subject
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)


def _validate_medium_v2_current_state(
    current_state: Mapping[str, Any],
    *,
    expected_anchor: str,
) -> None:
    if current_state["anchor_date"] not in (None, expected_anchor):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    bucket_keys = {
        "relative_momentum_bucket",
        "drawdown_bucket",
        "volatility_bucket",
        "market_trend_bucket",
        "market_volatility_bucket",
    }
    nonnegative_keys = {
        "volatility",
        "market_volatility",
        "downside_volatility",
        "average_dollar_volume",
    }
    for key, value in current_state.items():
        if key == "anchor_date" or value is None:
            continue
        if key in bucket_keys:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        elif not _medium_v2_finite(value):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        if key in nonnegative_keys and float(cast(int | float | Decimal, value)) < 0.0:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)


def _validate_medium_v2_support(support: Mapping[str, Any]) -> None:
    fallback = support["fallback_level"]
    if fallback not in {
        "exact_state",
        "without_market_volatility",
        "stock_state",
        "momentum_drawdown",
        "relative_momentum",
        "unconditional",
        "unavailable",
    } or not _medium_v2_probability(support["shrinkage_weight"]):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    raw = support["raw_matches"]
    cohorts = support["effective_cohorts"]
    listings = support["distinct_listings"]
    if not all(_medium_v2_nonnegative_int(value) for value in (raw, cohorts, listings)):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    if raw == 0:
        if (
            cohorts != 0
            or listings != 0
            or support["calendar_start"] is not None
            or support["calendar_end"] is not None
            or support["market_regimes"] != []
            or support["dispersion"] is not None
            or fallback != "unavailable"
            or float(cast(float, support["shrinkage_weight"])) != 0.0
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        return
    if (
        not _medium_v2_positive_counts(raw, cohorts, listings)
        or _medium_v2_date_range(support) is None
        or not _medium_v2_regimes(support["market_regimes"], require_nonempty=True)
        or len(support["market_regimes"]) > cohorts
        or not _medium_v2_nonnegative_number(support["dispersion"])
        or fallback == "unavailable"
        and float(cast(float, support["shrinkage_weight"])) != 0.0
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)


def _validate_medium_v2_evidence(
    *,
    base_accuracy: Mapping[str, Any],
    probability_skill: Mapping[str, Any],
    interval: Mapping[str, Any],
) -> None:
    for evidence_block in (base_accuracy, probability_skill, interval):
        origins = evidence_block["test_origins"]
        predictions = evidence_block["test_predictions"]
        if (
            not _medium_v2_nonnegative_int(origins)
            or not _medium_v2_nonnegative_int(predictions)
            or predictions < origins
            or (origins == 0) != (predictions == 0)
            or evidence_block["weighting"] != "date_equal_listing_equal_within_origin"
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    base_origins = base_accuracy["test_origins"]
    base_metrics = (
        base_accuracy["mean_absolute_error"],
        base_accuracy["unconditional_mean_absolute_error"],
        base_accuracy["spy_relative_mean_absolute_error"],
    )
    if (
        base_accuracy["spy_relative_baseline_method"]
        != "market_regime_benchmark_median_plus_relative_momentum_excess_median"
        or not _medium_v2_exact_number(base_accuracy["maximum_baseline_mae_ratio"], 1.0)
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    if base_origins == 0:
        expected_base_status = "not_evaluable"
        if any(value is not None for value in base_metrics):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    else:
        if not all(_medium_v2_nonnegative_number(value) for value in base_metrics):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        if base_origins < 4:
            expected_base_status = "insufficient_support"
        else:
            model, unconditional, relative = (
                float(cast(int | float | Decimal, value)) for value in base_metrics
            )
            expected_base_status = (
                "passed" if model <= unconditional and model <= relative else "failed"
            )
    if base_accuracy["status"] != expected_base_status:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    skill_origins = probability_skill["test_origins"]
    model_score = probability_skill["model_brier_score"]
    reference_score = probability_skill["reference_brier_score"]
    bss = probability_skill["brier_skill_score"]
    if skill_origins == 0:
        expected_skill_status = "not_evaluable"
        if model_score is not None or reference_score is not None or bss is not None:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    else:
        if not _medium_v2_probability(model_score) or not _medium_v2_probability(reference_score):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        model_value = float(cast(int | float | Decimal, model_score))
        reference_value = float(cast(int | float | Decimal, reference_score))
        expected_bss = None if reference_value == 0.0 else 1.0 - model_value / reference_value
        if expected_bss is None:
            if bss is not None:
                raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        elif not _medium_v2_finite(bss) or float(cast(int | float | Decimal, bss)) != expected_bss:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        if skill_origins < 4:
            expected_skill_status = "insufficient_support"
        elif reference_value == 0.0:
            expected_skill_status = "reference_zero"
        elif model_value < reference_value:
            expected_skill_status = "positive_skill"
        elif model_value == reference_value:
            expected_skill_status = "zero_skill"
        else:
            expected_skill_status = "negative_skill"
    if probability_skill["status"] != expected_skill_status:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    interval_origins = interval["test_origins"]
    rates = (
        interval["empirical_coverage"],
        interval["below_rate"],
        interval["above_rate"],
    )
    nonnegative_metrics = (
        interval["mean_width"],
        interval["model_mean_interval_score"],
        interval["reference_mean_interval_score"],
    )
    if interval_origins == 0:
        expected_interval_status = "not_evaluable"
        if any(value is not None for value in (*rates, *nonnegative_metrics)):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    else:
        partition_total = math.fsum(float(cast(int | float | Decimal, value)) for value in rates)
        if (
            not all(_medium_v2_probability(value) for value in rates)
            or not all(_medium_v2_nonnegative_number(value) for value in nonnegative_metrics)
            or not math.isclose(
                partition_total,
                1.0,
                rel_tol=0.0,
                abs_tol=4 * math.ulp(1.0),
            )
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        expected_interval_status = "preliminary" if interval_origins < 4 else "descriptive"
    if interval["status"] != expected_interval_status:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    if (
        base_accuracy["test_origins"] != interval["test_origins"]
        or base_accuracy["test_predictions"] != interval["test_predictions"]
        or probability_skill["test_origins"] > base_accuracy["test_origins"]
        or probability_skill["test_predictions"] > base_accuracy["test_predictions"]
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)


def _validate_medium_v2_numeric_support(
    *,
    horizon: str,
    support: Mapping[str, Any],
    matched: Mapping[str, Any],
    unconditional: Mapping[str, Any],
    probability_evidence: Mapping[str, Any],
) -> None:
    if (
        support["fallback_level"] == "unavailable"
        or support["raw_matches"] < 20
        or support["effective_cohorts"] < 3
        or support["distinct_listings"] < 10
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    for component in (matched, unconditional):
        if (
            component["raw_observations"] < 20
            or component["effective_cohorts"] < 3
            or component["distinct_listings"] < 10
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    for support_key, component_key in (
        ("raw_matches", "raw_observations"),
        ("effective_cohorts", "effective_cohorts"),
        ("distinct_listings", "distinct_listings"),
        ("calendar_start", "calendar_start"),
        ("calendar_end", "calendar_end"),
        ("market_regimes", "market_regimes"),
        ("dispersion", "dispersion"),
    ):
        if support[support_key] != matched[component_key]:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    expected_weight = (
        0.0
        if support["fallback_level"] == "unconditional"
        else support["effective_cohorts"] / (support["effective_cohorts"] + 4.0)
    )
    if float(cast(float, support["shrinkage_weight"])) != expected_weight:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    if support["fallback_level"] == "unconditional" and any(
        matched[key] != unconditional[key] for key in matched if key != "component_mass"
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    floors = _medium_v2_horizon_floors(horizon)
    matched_dates = _medium_v2_date_range(matched)
    if matched_dates is None:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    span = (matched_dates[1] - matched_dates[0]).days
    if (
        probability_evidence["calendar_span_days"] != span
        or probability_evidence["distinct_matched_market_regimes"] != len(matched["market_regimes"])
        or probability_evidence["distinct_panel_market_regimes"]
        != len(unconditional["market_regimes"])
        or any(
            probability_evidence[key] != floors[key]
            for key in (
                "minimum_effective_cohorts",
                "minimum_distinct_listings",
                "minimum_calendar_span_days",
                "minimum_distinct_market_regimes",
            )
        )
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)


def _medium_v2_horizon_floors(horizon: str) -> dict[str, int]:
    if horizon == "6m":
        cohorts, span = 8, 1_095
    elif horizon == "12m":
        cohorts, span = 6, 1_460
    else:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    return {
        "minimum_effective_cohorts": cohorts,
        "minimum_distinct_listings": 30,
        "minimum_calendar_span_days": span,
        "minimum_distinct_market_regimes": 3,
    }


def _medium_v2_probability_reasons(
    *,
    horizon: str,
    matched: Mapping[str, Any],
    probability_skill: Mapping[str, Any],
) -> list[str]:
    floors = _medium_v2_horizon_floors(horizon)
    dates = _medium_v2_date_range(matched)
    if dates is None:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    span = (dates[1] - dates[0]).days
    reasons: list[str] = []
    if matched["effective_cohorts"] < floors["minimum_effective_cohorts"]:
        reasons.append(
            f"effective cohorts {matched['effective_cohorts']}/"
            f"{floors['minimum_effective_cohorts']}"
        )
    if matched["distinct_listings"] < floors["minimum_distinct_listings"]:
        reasons.append(
            f"distinct listings {matched['distinct_listings']}/"
            f"{floors['minimum_distinct_listings']}"
        )
    if span < floors["minimum_calendar_span_days"]:
        reasons.append(f"calendar span {span}/{floors['minimum_calendar_span_days']} days")
    regimes = len(matched["market_regimes"])
    if regimes < floors["minimum_distinct_market_regimes"]:
        reasons.append(
            f"matched market regimes {regimes}/{floors['minimum_distinct_market_regimes']}"
        )
    status = probability_skill["status"]
    if status == "positive_skill":
        if float(cast(float, probability_skill["brier_skill_score"])) <= 0.0:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    elif status in {"not_evaluable", "insufficient_support"}:
        reasons.append(
            "prequential Brier skill insufficient "
            f"({probability_skill['test_origins']}/4 test origins)"
        )
    elif status == "reference_zero":
        reasons.append("prequential unconditional reference Brier score is zero")
    elif status == "zero_skill":
        reasons.append("prequential Brier skill is zero")
    elif status == "negative_skill":
        reasons.append("prequential Brier skill is negative")
    else:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    return reasons


def _validate_medium_v2_state(
    *,
    horizon: str,
    forecast: MediumForecast,
    support: Mapping[str, Any],
    predictive: Mapping[str, Any],
    matched: Mapping[str, Any],
    unconditional: Mapping[str, Any],
    formula_scenario: Mapping[str, Any],
    probability_evidence: Mapping[str, Any],
    probability_skill: Mapping[str, Any],
) -> None:
    scenario = forecast.scenario
    if (
        scenario.method != "conditional_empirical_price"
        or not isinstance(scenario.insufficiency_reason, str)
        or not _medium_v2_finite(scenario.confidence)
        or not 0.0 <= float(scenario.confidence) <= 100.0
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    scenario_triplet = (scenario.bear, scenario.base, scenario.bull)
    formula_triplet = (
        formula_scenario["bear"],
        formula_scenario["base"],
        formula_scenario["bull"],
    )
    predictive_triplet = (
        predictive["p20"],
        predictive["p50"],
        predictive["p80"],
    )
    all_null = all(value is None for value in scenario_triplet)
    complete = all(_medium_v2_finite(value) for value in scenario_triplet)
    if not all_null and not complete:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    if complete:
        numeric_triplet = tuple(float(cast(float, value)) for value in scenario_triplet)
        _validate_medium_v2_component(matched, require_nonempty=True)
        _validate_medium_v2_component(unconditional, require_nonempty=True)
        _validate_medium_v2_numeric_support(
            horizon=horizon,
            support=support,
            matched=matched,
            unconditional=unconditional,
            probability_evidence=probability_evidence,
        )
        if (
            not _medium_v2_probability(matched["component_mass"])
            or not _medium_v2_probability(unconditional["component_mass"])
            or float(cast(float, matched["component_mass"]))
            != float(cast(float, support["shrinkage_weight"]))
            or float(cast(float, unconditional["component_mass"]))
            != 1.0 - float(cast(float, support["shrinkage_weight"]))
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        if (
            min(numeric_triplet) < -1.0
            or numeric_triplet != tuple(sorted(numeric_triplet))
            or not _medium_v2_quantized_triplet_equal(scenario_triplet, formula_triplet)
            or not _medium_v2_exact_triplet_equal(formula_triplet, predictive_triplet)
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        raw_probability = predictive["probability_positive_raw"]
        published_probability = predictive["probability_positive_published"]
        if not _medium_v2_probability(raw_probability):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        expected_raw_probability = math.fsum(
            float(cast(int | float | Decimal, component["component_mass"]))
            * float(
                cast(
                    int | float | Decimal,
                    component["probability_positive_raw"],
                )
            )
            for component in (matched, unconditional)
        )
        if float(cast(int | float | Decimal, raw_probability)) != expected_raw_probability:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        expected_reasons = _medium_v2_probability_reasons(
            horizon=horizon,
            matched=matched,
            probability_skill=probability_skill,
        )
        if scenario.confidence_status == "empirical_skill_supported":
            if (
                probability_evidence["status"] != "published"
                or probability_evidence["reasons"]
                or expected_reasons
                or probability_skill["status"] != "positive_skill"
                or not _medium_v2_finite(probability_skill["brier_skill_score"])
                or float(cast(float, probability_skill["brier_skill_score"])) <= 0.0
                or not _medium_v2_probability(published_probability)
                or not _medium_v2_numeric_equal(raw_probability, published_probability)
                or not _medium_v2_numeric_equal(
                    raw_probability, formula_scenario["probability_positive"]
                )
                or not _medium_v2_optional_decimal_equal(
                    raw_probability, scenario.probability_positive
                )
                or scenario.insufficiency_reason
            ):
                raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        elif scenario.confidence_status == "empirical_range_only":
            if (
                probability_evidence["status"] != "withheld"
                or probability_evidence["reasons"] != expected_reasons
                or not expected_reasons
                or published_probability is not None
                or formula_scenario["probability_positive"] is not None
                or scenario.probability_positive is not None
                or scenario.insufficiency_reason
                != "Probability withheld: " + "; ".join(expected_reasons)
            ):
                raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        else:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        if _decimal(scenario.confidence, places=2) != _decimal(
            min(
                80.0,
                20.0
                + 60.0
                * float(
                    cast(
                        Mapping[str, Any],
                        forecast.calculation["support"],
                    )["shrinkage_weight"]
                ),
            ),
            places=2,
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        return
    _validate_medium_v2_component(matched, require_nonempty=False)
    _validate_medium_v2_component(unconditional, require_nonempty=False)
    floor_values = _medium_v2_horizon_floors(horizon)
    if (
        not all(value is None for value in formula_triplet)
        or not all(value is None for value in predictive_triplet)
        or predictive["probability_positive_raw"] is not None
        or predictive["probability_positive_published"] is not None
        or formula_scenario["probability_positive"] is not None
        or scenario.probability_positive is not None
        or scenario.confidence_status != "insufficient_evidence"
        or _decimal(scenario.confidence, places=2) != Decimal("0.00")
        or probability_evidence["status"] != "not_evaluable"
        or probability_evidence["reasons"] != []
        or probability_evidence["calendar_span_days"] is not None
        or probability_evidence["distinct_matched_market_regimes"] is not None
        or probability_evidence["distinct_panel_market_regimes"] is not None
        or any(
            probability_evidence[key] != floor_values[key]
            for key in (
                "minimum_effective_cohorts",
                "minimum_distinct_listings",
                "minimum_calendar_span_days",
                "minimum_distinct_market_regimes",
            )
        )
        or support["fallback_level"] != "unavailable"
        or float(cast(float, support["shrinkage_weight"])) != 0.0
        or (
            support["raw_matches"] >= 20
            and support["effective_cohorts"] >= 3
            and support["distinct_listings"] >= 10
        )
        or not scenario.insufficiency_reason
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    if support["raw_matches"] > 0:
        expected_reason = (
            f"Insufficient non-overlapping {horizon} evidence: "
            f"{support['raw_matches']}/20 observations, "
            f"{support['effective_cohorts']}/3 cohorts, "
            f"{support['distinct_listings']}/10 listings"
        )
        if scenario.insufficiency_reason != expected_reason:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)


def _validate_medium_v2_component(
    component: Mapping[str, Any],
    *,
    require_nonempty: bool,
) -> None:
    observations = component["raw_observations"]
    if require_nonempty:
        triplet = (component["p20"], component["p50"], component["p80"])
        if (
            not _medium_v2_positive_counts(
                observations,
                component["effective_cohorts"],
                component["distinct_listings"],
            )
            or not _medium_v2_exact_number(component["normalized_mass"], 1.0)
            or not all(_medium_v2_finite(value) for value in triplet)
            or tuple(float(cast(float, value)) for value in triplet)
            != tuple(sorted(float(cast(float, value)) for value in triplet))
            or min(float(cast(float, value)) for value in triplet) < -1.0
            or not _medium_v2_probability(component["probability_positive_raw"])
            or _medium_v2_date_range(component) is None
            or not _medium_v2_regimes(component["market_regimes"], require_nonempty=True)
            or len(component["market_regimes"]) > component["effective_cohorts"]
            or not _medium_v2_nonnegative_number(component["dispersion"])
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        return
    if (
        observations != 0
        or component["effective_cohorts"] != 0
        or component["distinct_listings"] != 0
        or component["component_mass"] is not None
        or component["normalized_mass"] is not None
        or component["p20"] is not None
        or component["p50"] is not None
        or component["p80"] is not None
        or component["probability_positive_raw"] is not None
        or component["calendar_start"] is not None
        or component["calendar_end"] is not None
        or component["market_regimes"] != []
        or component["dispersion"] is not None
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)


def _medium_v2_positive_counts(
    observations: object,
    cohorts: object,
    listings: object,
) -> bool:
    return bool(
        all(_medium_v2_nonnegative_int(value) for value in (observations, cohorts, listings))
        and cast(int, observations) > 0
        and 0 < cast(int, cohorts) <= cast(int, observations)
        and 0 < cast(int, listings) <= cast(int, observations)
    )


def _medium_v2_date_range(
    value: Mapping[str, Any],
) -> tuple[date, date] | None:
    start_text = value["calendar_start"]
    end_text = value["calendar_end"]
    if not isinstance(start_text, str) or not isinstance(end_text, str):
        return None
    try:
        start = date.fromisoformat(start_text)
        end = date.fromisoformat(end_text)
    except ValueError:
        return None
    if start.isoformat() != start_text or end.isoformat() != end_text or start > end:
        return None
    return start, end


def _medium_v2_regimes(value: object, *, require_nonempty: bool) -> bool:
    if not isinstance(value, list) or (require_nonempty and not value):
        return False
    return all(isinstance(item, str) and bool(item) for item in value) and value == sorted(
        set(value)
    )


def _medium_v2_nonnegative_number(value: object) -> bool:
    return _medium_v2_finite(value) and float(cast(int | float | Decimal, value)) >= 0.0


def _medium_v2_exact_number(value: object, expected: float) -> bool:
    return _medium_v2_finite(value) and float(cast(int | float | Decimal, value)) == expected


def _exact_medium_v2_mapping(
    value: object,
    keys: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    return cast(Mapping[str, Any], value)


def _medium_v2_finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, Decimal))
        and math.isfinite(float(value))
    )


def _medium_v2_probability(value: object) -> bool:
    return _medium_v2_finite(value) and 0.0 <= float(cast(int | float | Decimal, value)) <= 1.0


def _medium_v2_nonnegative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _medium_v2_optional_decimal_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    if not _medium_v2_finite(left) or not _medium_v2_finite(right):
        return False
    return _optional_decimal(
        float(cast(int | float | Decimal, left)), places=4
    ) == _optional_decimal(float(cast(int | float | Decimal, right)), places=4)


def _medium_v2_quantized_triplet_equal(
    left: Sequence[object],
    right: Sequence[object],
) -> bool:
    return all(
        _medium_v2_optional_decimal_equal(left_value, right_value)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _medium_v2_numeric_equal(left: object, right: object) -> bool:
    return bool(
        _medium_v2_finite(left)
        and _medium_v2_finite(right)
        and float(cast(int | float | Decimal, left)) == float(cast(int | float | Decimal, right))
    )


def _medium_v2_exact_triplet_equal(
    left: Sequence[object],
    right: Sequence[object],
) -> bool:
    return all(
        _medium_v2_numeric_equal(left_value, right_value)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _freeze_medium_v2_calculation(value: object) -> object:
    """Recursively freeze replayed calculator content without changing it."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_medium_v2_calculation(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_medium_v2_calculation(child) for child in value)
    return value


def _attest_medium_v2_panel(
    *,
    run: AnalysisRun,
    panel_asset: DataAsset,
    store: AssetStore,
) -> _MediumV2PanelAttestation:
    """Rebuild one v2 panel from independently resolved immutable sources.

    The panel and its copied manifest are both untrusted inputs here. Every
    manifest UUID is resolved to its authoritative append-only ``DataAsset``
    row, every declared identity field must match that row, and the resulting
    set must be exactly the run snapshot's eligible listing closure plus SPY.
    Those exact assets are checksum-read without selection or fallback, then
    the canonical panel is reconstructed in memory and compared byte-for-byte
    with the supplied panel before any forecast calculator is replayed.
    """
    with transaction.atomic():
        authority = _lock_medium_v2_authority(run)
        if authority.run is None:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        return _attest_medium_v2_panel_with_authority(
            run=authority.run,
            panel_asset=panel_asset,
            store=store,
            authority=authority,
        )


def _attest_medium_v2_panel_with_authority(
    *,
    run: AnalysisRun,
    panel_asset: DataAsset,
    store: AssetStore,
    authority: _LockedMediumV2Authority,
    config: MediumForecastV2Config | None = None,
    config_hash_value: str | None = None,
) -> _MediumV2PanelAttestation:
    """Attest a panel while reusing authority held by the caller."""
    if authority.run is None or run.pk != authority.run.pk:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    run = authority.run
    snapshot = authority.snapshot
    authoritative_panel = DataAsset.objects.filter(pk=panel_asset.pk).first()
    panel_identity_fields = (
        "provider",
        "kind",
        "subject",
        "relative_path",
        "sha256",
        "retrieved_at",
        "available_at",
        "period_start",
        "period_end",
        "schema_version",
        "metadata",
    )
    if authoritative_panel is None or any(
        getattr(panel_asset, field) != getattr(authoritative_panel, field)
        for field in panel_identity_fields
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    panel_asset = authoritative_panel
    metadata = panel_asset.metadata
    if (
        panel_asset.provider != "stanstock"
        or panel_asset.kind != "medium_forecast_panel"
        or panel_asset.subject != str(run.id)
        or panel_asset.schema_version != "1"
        or panel_asset.retrieved_at != run.generated_at
        or panel_asset.available_at != run.generated_at
        or not isinstance(metadata, dict)
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    if config is None:
        loaded_config = load_medium_forecast_config(
            Path(settings.BASE_DIR) / "config" / "forecasts" / "us-price-medium-v2.yml"
        )
        if not isinstance(loaded_config, MediumForecastV2Config):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        config = loaded_config
        config_hash_value = medium_forecast_config_hash(config)
    if config.version != MEDIUM_V2_VERSION or config_hash_value != MEDIUM_V2_EFFECTIVE_CONFIG_HASH:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    raw_manifest = metadata.get("source_assets")
    manifest_keys = {
        "id",
        "provider",
        "kind",
        "subject",
        "relative_path",
        "sha256",
        "retrieved_at",
        "available_at",
    }
    if not isinstance(raw_manifest, list) or not raw_manifest:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    source_ids: list[UUID] = []
    for raw_source in raw_manifest:
        if (
            not isinstance(raw_source, dict)
            or set(raw_source) != manifest_keys
            or not all(
                isinstance(raw_source[key], str) and bool(raw_source[key]) for key in manifest_keys
            )
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        try:
            source_ids.append(UUID(raw_source["id"]))
        except (TypeError, ValueError, AttributeError):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR) from None
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    resolved_by_id = DataAsset.objects.in_bulk(source_ids)
    if len(resolved_by_id) != len(source_ids):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    resolved_sources: list[DataAsset] = []
    for raw_source, source_id in zip(raw_manifest, source_ids, strict=True):
        source = resolved_by_id.get(source_id)
        if source is None or asset_identity(source) != raw_source:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        if (
            source.kind != "price_history"
            or source.available_at > run.data_cutoff
            or source.available_at > run.generated_at
            or source.retrieved_at > run.generated_at
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        resolved_sources.append(source)

    memberships = list(authority.memberships)
    if not memberships:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    listings = list(authority.listings)
    for listing in listings:
        require_stock_research_listing(
            listing,
            operation="Medium forecast panel attestation",
        )
    if any(listing.region != Region.US or listing.currency != "USD" for listing in listings):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    listing_subjects = [listing.provider_symbol or listing.ticker for listing in listings]
    expected_subjects = [*listing_subjects, "SPY"]
    actual_subjects = [source.subject for source in resolved_sources]
    providers = {source.provider for source in resolved_sources}
    if (
        len(expected_subjects) != len(set(expected_subjects))
        or len(actual_subjects) != len(set(actual_subjects))
        or set(actual_subjects) != set(expected_subjects)
        or len(resolved_sources) != len(expected_subjects)
        or len(providers) != 1
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    price_provider = next(iter(providers))
    if any(source.provider != price_provider for source in resolved_sources):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    authoritative_manifest = [
        asset_identity(source)
        for source in sorted(resolved_sources, key=lambda candidate: str(candidate.pk))
    ]
    if raw_manifest != authoritative_manifest:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    source_manifest_hash = hash_json(authoritative_manifest)

    source_by_subject = {source.subject: source for source in resolved_sources}
    asof = AsOfData(run.generated_at, store)
    benchmark_read = asof.price_frame_for_asset_with_diagnostics(
        asset=source_by_subject["SPY"],
        through_date=run.target_date,
    )
    listing_inputs: list[MediumPanelPriceInput] = []
    for listing in sorted(listings, key=lambda candidate: str(candidate.pk)):
        subject = listing.provider_symbol or listing.ticker
        source_read = asof.price_frame_for_asset_with_diagnostics(
            asset=source_by_subject[subject],
            through_date=run.target_date,
        )
        listing_inputs.append(
            MediumPanelPriceInput(
                listing=listing,
                asset=source_read.asset,
                frame=source_read.frame,
            )
        )
    canonical_frame = reconstruct_medium_forecast_panel(
        benchmark_asset=benchmark_read.asset,
        benchmark_frame=benchmark_read.frame,
        listing_inputs=listing_inputs,
        target_date=run.target_date,
        config=config,
    )
    canonical_payload = serialize_medium_forecast_panel(canonical_frame)
    panel_payload = read_checksummed_bytes(store, panel_asset)
    supplied_frame = pl.read_parquet(io.BytesIO(panel_payload))
    if (
        dict(supplied_frame.schema) != PANEL_SCHEMA
        or not supplied_frame.equals(canonical_frame)
        or panel_payload != canonical_payload
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    calendar_sessions = calendar_sessions_through(
        calendar_name=config.calendar,
        fixed_epoch=config.fixed_epoch,
        target_date=run.target_date,
    )
    calendar_hash = hash_json([session.isoformat() for session in calendar_sessions])
    evidence_bundle_hash = hash_json(
        {
            "calendar_hash": calendar_hash,
            "code_revision": run.code_revision,
            "content_sha256": panel_asset.sha256,
            "forecast_config_hash": MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
            "scoring_config_hash": run.config_hash,
            "source_manifest_hash": source_manifest_hash,
            "universe_config_hash": snapshot.config_hash,
        }
    )
    expected_metadata: dict[str, object] = {
        "schema_version": 1,
        "method_version": MEDIUM_V2_VERSION,
        "config_hash": MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
        "code_revision": run.code_revision,
        "calendar": config.calendar,
        "calendar_library_version": package_version("exchange-calendars"),
        "panel_library_version": package_version("polars"),
        "fixed_epoch": config.fixed_epoch.isoformat(),
        "calendar_hash": calendar_hash,
        "target_date": run.target_date.isoformat(),
        "universe_snapshot_id": str(snapshot.id),
        "universe_slug": str(snapshot.universe_id),
        "universe_config_hash": snapshot.config_hash,
        "scoring_config_version": run.config_version,
        "scoring_config_hash": run.config_hash,
        "return_definition": config.return_basis,
        "dividends_included": config.dividends_included,
        "training_evidence_grade": "research",
        "current_universe_survivorship_bias": True,
        "usage_scope": "private_single_user_research",
        "row_count": canonical_frame.height,
        "content_sha256": panel_asset.sha256,
        "source_manifest_hash": source_manifest_hash,
        "evidence_bundle_hash": evidence_bundle_hash,
        "source_assets": authoritative_manifest,
    }
    anchor_dates = [
        anchor for anchor in canonical_frame["anchor_date"].to_list() if isinstance(anchor, date)
    ]
    expected_path = (
        f"derived/forecast/medium/{run.target_date.isoformat()}/"
        f"{run.id.hex}-{panel_asset.sha256[:12]}.parquet"
    )
    if (
        metadata != expected_metadata
        or panel_asset.relative_path != expected_path
        or panel_asset.sha256 != hashlib.sha256(canonical_payload).hexdigest()
        or panel_asset.period_start != (min(anchor_dates) if anchor_dates else run.target_date)
        or panel_asset.period_end != run.target_date
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)

    calculations = MappingProxyType(
        {
            listing_id: MappingProxyType(
                {
                    horizon: _freeze_medium_v2_calculation(forecast.calculation)
                    for horizon, forecast in listing_forecasts.items()
                }
            )
            for listing_id, listing_forecasts in build_medium_forecasts(
                canonical_frame,
                config,
            ).items()
        }
    )
    return _MediumV2PanelAttestation(
        panel_id=panel_asset.id,
        panel_sha256=panel_asset.sha256,
        run_id=run.id,
        config_hash=MEDIUM_V2_EFFECTIVE_CONFIG_HASH,
        snapshot_id=snapshot.id,
        source_manifest_hash=source_manifest_hash,
        price_provider=price_provider,
        calculations=calculations,
    )


def _require_medium_v2_attestation_binding(
    *,
    attestation: _MediumV2PanelAttestation,
    run: AnalysisRun,
    panel_asset: DataAsset,
) -> None:
    metadata = panel_asset.metadata
    if (
        not isinstance(attestation, _MediumV2PanelAttestation)
        or attestation.panel_id != panel_asset.id
        or attestation.panel_sha256 != panel_asset.sha256
        or attestation.run_id != run.id
        or attestation.config_hash != MEDIUM_V2_EFFECTIVE_CONFIG_HASH
        or attestation.snapshot_id != run.universe_snapshot_id
        or not isinstance(metadata, dict)
        or attestation.source_manifest_hash != metadata.get("source_manifest_hash")
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)


def _require_medium_v2_forecast_matches_panel(
    *,
    analysis: StockAnalysis,
    horizon: str,
    forecast: MediumForecast,
    panel_asset: DataAsset,
    panel_attestation: _MediumV2PanelAttestation,
) -> None:
    """Bind one caller forecast to the exact immutable panel it names.

    Metadata and support-count agreement cannot prove that a coherent payload
    was calculated from this panel. Independently attest the panel's exact
    source reconstruction first, then require exact equality of all 14
    calculator-owned keys before either writer may create a row.
    """
    try:
        _require_medium_v2_attestation_binding(
            attestation=panel_attestation,
            run=analysis.run,
            panel_asset=panel_asset,
        )
        price_source = analysis.data_quality.get("price_source")
        expected_subject = analysis.listing.provider_symbol or analysis.listing.ticker
        if (
            not isinstance(price_source, Mapping)
            or price_source.get("provider") != panel_attestation.price_provider
            or price_source.get("subject") != expected_subject
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        expected = panel_attestation.calculations[str(analysis.listing_id)][horizon]
    except (
        KeyError,
        OSError,
        PriceFrameChecksumMismatchError,
        RefreshVerificationError,
        TypeError,
        ValueError,
        pl.exceptions.PolarsError,
    ):
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR) from None
    if _freeze_medium_v2_calculation(forecast.calculation) != expected:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)


def _advisory_prediction_kwargs(
    *,
    analysis: StockAnalysis,
    horizon: Prediction.Horizon,
    forecast: MediumForecast,
    panel_asset: DataAsset,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, Any]],
    code_revision_value: str,
    require_v2: bool = False,
) -> dict[str, Any]:
    """Assemble one fully validated advisory row without persisting it."""
    price_provider, price_subject = _prediction_price_source(analysis, source_assets)
    calculation = dict(forecast.calculation)
    v2_request = require_v2 or _is_medium_v2_prediction(
        forecast=forecast,
        model_version=model_version,
        config_hash_value=config_hash_value,
        panel_asset=panel_asset,
    )
    if v2_request:
        _validate_medium_v2_prediction(
            analysis=analysis,
            horizon=horizon,
            forecast=forecast,
            panel_asset=panel_asset,
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            issued_on_time=issued_on_time,
            model_version=model_version,
            config_hash_value=config_hash_value,
            source_assets=source_assets,
            code_revision_value=code_revision_value,
            calculation=calculation,
            persisted=False,
        )
    calculation.update(
        {
            "config_hash": config_hash_value,
            "prediction_version": model_version,
            "panel_asset_id": str(panel_asset.pk),
            "panel_sha256": panel_asset.sha256,
            "evidence_grade": analysis.run.universe_snapshot.grade,
            "price_subject": price_subject,
        }
    )
    if v2_request:
        _validate_medium_v2_prediction(
            analysis=analysis,
            horizon=horizon,
            forecast=forecast,
            panel_asset=panel_asset,
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            issued_on_time=issued_on_time,
            model_version=model_version,
            config_hash_value=config_hash_value,
            source_assets=source_assets,
            code_revision_value=code_revision_value,
            calculation=calculation,
            persisted=True,
        )
    scenario = forecast.scenario
    return {
        "analysis": analysis,
        "listing": analysis.listing,
        "generated_at": generated_at,
        "target_date": analysis.run.target_date,
        "issued_on_time": issued_on_time,
        "horizon": horizon,
        "evidence_role": Prediction.EvidenceRole.ADVISORY,
        "evidence_grade": analysis.run.universe_snapshot.grade,
        "source_mode": source_data_mode({"source_assets": source_assets}),
        "price_provider": price_provider,
        "price_subject": price_subject,
        "price_at_prediction": analysis.current_price,
        "bear_return": _optional_decimal(scenario.bear, places=4),
        "base_return": _optional_decimal(scenario.base, places=4),
        "bull_return": _optional_decimal(scenario.bull, places=4),
        "probability_positive": _optional_decimal(scenario.probability_positive, places=4),
        "confidence": _decimal(scenario.confidence, places=2),
        "confidence_status": scenario.confidence_status,
        "insufficiency_reason": scenario.insufficiency_reason,
        "recommendation": analysis.recommendation,
        "overall_score": analysis.overall_score,
        "component_scores": analysis.component_scores,
        "model_version": model_version,
        "method_version": str(calculation["method_version"]),
        "config_hash": config_hash_value,
        "data_cutoff": data_cutoff,
        "source_assets": source_assets,
        "calculation": calculation,
        "code_revision": code_revision_value,
    }


def _create_advisory_prediction(
    *,
    analysis: StockAnalysis,
    horizon: Prediction.Horizon,
    forecast: MediumForecast,
    panel_asset: DataAsset,
    store: AssetStore | None = None,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, Any]],
    code_revision_value: str,
) -> Prediction:
    with transaction.atomic():
        authority = _lock_advisory_authority((analysis,))
        return _create_advisory_prediction_with_authority(
            analysis=analysis,
            horizon=horizon,
            forecast=forecast,
            panel_asset=panel_asset,
            store=store,
            generated_at=generated_at,
            data_cutoff=data_cutoff,
            issued_on_time=issued_on_time,
            model_version=model_version,
            config_hash_value=config_hash_value,
            source_assets=source_assets,
            code_revision_value=code_revision_value,
            authority=authority,
        )


def _create_advisory_prediction_with_authority(
    *,
    analysis: StockAnalysis,
    horizon: Prediction.Horizon,
    forecast: MediumForecast,
    panel_asset: DataAsset,
    store: AssetStore | None,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    source_assets: list[dict[str, Any]],
    code_revision_value: str,
    authority: _LockedAdvisoryAuthority,
) -> Prediction:
    """Create one advisory row using authority held by the caller."""
    if len(authority.analyses) != 1:
        raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
    locked_analysis = authority.analyses[0]
    v2_request = _is_medium_v2_prediction(
        forecast=forecast,
        model_version=model_version,
        config_hash_value=config_hash_value,
        panel_asset=panel_asset,
    )
    if v2_request:
        panel_authority = authority.panels_by_run.get(locked_analysis.run_id)
        if panel_authority is None:
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR)
        _validate_locked_medium_v2_authority(
            panel_authority,
            requested_listing_ids=(locked_analysis.listing_id,),
            detailed_errors=False,
        )
        try:
            panel_attestation = _attest_medium_v2_panel_with_authority(
                run=locked_analysis.run,
                panel_asset=panel_asset,
                store=store or open_asset_store(),
                authority=panel_authority,
            )
        except (
            KeyError,
            OSError,
            PriceFrameChecksumMismatchError,
            RefreshVerificationError,
            TypeError,
            ValueError,
            pl.exceptions.PolarsError,
        ):
            raise ValueError(_MEDIUM_V2_PAYLOAD_ERROR) from None
    kwargs = _advisory_prediction_kwargs(
        analysis=locked_analysis,
        horizon=horizon,
        forecast=forecast,
        panel_asset=panel_asset,
        generated_at=generated_at,
        data_cutoff=data_cutoff,
        issued_on_time=issued_on_time,
        model_version=model_version,
        config_hash_value=config_hash_value,
        source_assets=source_assets,
        code_revision_value=code_revision_value,
        require_v2=v2_request,
    )
    if v2_request:
        _require_medium_v2_forecast_matches_panel(
            analysis=locked_analysis,
            horizon=str(horizon),
            forecast=forecast,
            panel_asset=panel_asset,
            panel_attestation=panel_attestation,
        )
    return Prediction.objects.create(**kwargs)


def _create_long_advisory_prediction(
    *,
    analysis: StockAnalysis,
    horizon: Prediction.Horizon,
    forecast: LongForecast,
    generated_at: datetime,
    data_cutoff: datetime,
    issued_on_time: bool,
    model_version: str,
    config_hash_value: str,
    code_revision_value: str,
) -> Prediction:
    source_assets = [_asset_payload(asset) for asset in forecast.source_assets]
    if issued_on_time:
        _validate_on_time_source_assets(source_assets, data_cutoff=data_cutoff)
    price_provider, price_subject = _prediction_price_source(analysis, source_assets)
    calculation = dict(forecast.calculation)
    calculation.update(
        {
            "config_hash": config_hash_value,
            "prediction_version": model_version,
            "price_subject": price_subject,
        }
    )
    scenario = forecast.scenario
    evidence_grade = _long_forecast_evidence_grade(analysis.run)
    return Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=generated_at,
        target_date=analysis.run.target_date,
        issued_on_time=issued_on_time,
        horizon=horizon,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        evidence_grade=evidence_grade,
        source_mode=source_data_mode({"source_assets": source_assets}),
        price_provider=price_provider,
        price_subject=price_subject,
        price_at_prediction=analysis.current_price,
        bear_return=_optional_decimal(scenario.bear, places=4),
        base_return=_optional_decimal(scenario.base, places=4),
        bull_return=_optional_decimal(scenario.bull, places=4),
        probability_positive=None,
        confidence=_decimal(scenario.confidence, places=2),
        confidence_status=scenario.confidence_status,
        insufficiency_reason=scenario.insufficiency_reason,
        recommendation=analysis.recommendation,
        overall_score=analysis.overall_score,
        component_scores=analysis.component_scores,
        model_version=model_version,
        method_version=str(calculation["method_version"]),
        config_hash=config_hash_value,
        data_cutoff=data_cutoff,
        source_assets=source_assets,
        calculation={
            **calculation,
            "evidence_grade": evidence_grade,
        },
        code_revision=code_revision_value,
    )


def _long_forecast_evidence_grade(run: AnalysisRun) -> str:
    if run.issued_on_time and run.universe_snapshot.grade == UniverseSnapshot.Grade.OBSERVED:
        return UniverseSnapshot.Grade.OBSERVED
    return UniverseSnapshot.Grade.RESEARCH


def _prediction_price_source(
    analysis: StockAnalysis,
    source_assets: list[dict[str, Any]],
) -> tuple[str, str]:
    price_source = analysis.data_quality.get("price_source")
    if isinstance(price_source, dict):
        return (
            str(price_source.get("provider") or ""),
            str(price_source.get("subject") or ""),
        )
    return infer_price_source(
        source_assets,
        subjects=(
            analysis.listing.provider_symbol or analysis.listing.ticker,
            analysis.listing.ticker,
        ),
    )


def _asset_payload(asset: DataAsset) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": str(asset.id),
        "provider": asset.provider,
        "kind": asset.kind,
        "subject": asset.subject,
        "relative_path": asset.relative_path,
        "sha256": asset.sha256,
        "retrieved_at": asset.retrieved_at.isoformat(),
        "available_at": asset.available_at.isoformat(),
    }
    if isinstance(asset.metadata.get("return_definition"), str):
        payload["return_definition"] = asset.metadata["return_definition"]
    if isinstance(asset.metadata.get("dividends_included"), bool):
        payload["dividends_included"] = asset.metadata["dividends_included"]
    return payload


def _dedupe_assets(assets: list[DataAsset]) -> list[DataAsset]:
    seen: set[str] = set()
    deduped: list[DataAsset] = []
    for asset in assets:
        key = str(asset.id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(asset)
    return deduped


def _model_version(config_version: str, run_hex: str) -> str:
    suffix = run_hex[:8]
    return f"{config_version}-{suffix}"[:40]


def _decimal(value: float, *, places: int) -> Decimal:
    return decimal_from_float(value, places=places)


def _optional_decimal(value: float | None, *, places: int) -> Decimal | None:
    return optional_decimal_from_float(value, places=places)


def _analysis_data_cutoff(
    generated_at: datetime,
    target_date: date,
    *,
    issued_on_time: bool,
) -> datetime:
    if target_date > generated_at.date():
        raise ValueError(
            f"target_date ({target_date.isoformat()}) cannot be after generation date "
            f"({generated_at.date().isoformat()})"
        )
    if issued_on_time or target_date == generated_at.date():
        return generated_at
    return datetime.combine(target_date, time.max, tzinfo=generated_at.tzinfo)


def _issued_on_time(
    snapshot: UniverseSnapshot,
    *,
    generated_at: datetime,
    target_date: date,
    explicit: bool | None,
) -> bool:
    if explicit is not None:
        if explicit and snapshot.grade != UniverseSnapshot.Grade.OBSERVED:
            raise ValueError("Only an observed universe snapshot can be issued on time")
        if explicit and not is_observed_issuance_on_time(
            snapshot,
            target_date=target_date,
            generated_at=generated_at,
        ):
            raise ValueError("Observed analysis was generated after the next market session opened")
        return explicit
    return snapshot.grade == UniverseSnapshot.Grade.OBSERVED and generated_at.date() == target_date


def _validate_snapshot_for_target(snapshot: UniverseSnapshot, target_date: date) -> None:
    if snapshot.grade == UniverseSnapshot.Grade.OBSERVED and snapshot.as_of_date > target_date:
        raise ValueError(
            f"Observed universe snapshot {snapshot.pk} is dated "
            f"{snapshot.as_of_date.isoformat()} and cannot be used for earlier "
            f"target date {target_date.isoformat()}"
        )

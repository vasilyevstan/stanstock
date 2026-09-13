"""Read-only retrospective replay for the active price research product.

This module binds one persisted ``research-product-v1`` run to its exact
immutable selected sources, re-verifies that source closure independently,
and replays the frozen retrospective protocol against earlier anchors without
creating jobs, analyses, predictions, or assets.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Any, cast
from uuid import UUID

import polars as pl
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.verification_types import AssetRef
from stanstock.data.asof import AsOfData
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    read_checksummed_bytes,
    resolve_asset_ref,
)
from stanstock.data.models import DataAsset, Listing
from stanstock.data.research_product import product_membership_payload
from stanstock.research.config import code_revision
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.price_product import (
    EvidenceGrade,
    FilteredReturns,
    PriceProductInput,
    PriceProductInputError,
    PriceSeries,
    SourceExecutionBinding,
    SourceExecutionMode,
    calculate_price_product,
    filter_historical_returns,
)
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PriceProductConfig,
    load_price_product_config,
)
from stanstock.research.price_product_replay import (
    ProjectionMetricName,
    ProjectionMetricResult,
    ReplayMetricObservation,
    ReplayPartition,
    aggregate_projection_metrics,
    build_anchor_plan,
    calculate_fhs_convergence,
    compare_aligned_models,
    evaluate_momentum_diagnostic,
    gaussian_comparator_quantiles,
    replay_precision,
    score_projection_metrics,
)
from stanstock.research.product_pipeline import (
    CALCULATION_ARTIFACT_CONTRACT,
    CALCULATION_ARTIFACT_KIND,
    ProductSourceSelection,
    price_product_input_from_verified_source_window,
    select_product_source,
    verify_price_product_output,
)

REPLAY_STUDY_SCHEMA = "research-product-retrospective-study@1"
_PARTITION_ORDER: tuple[ReplayPartition, ReplayPartition, ReplayPartition] = (
    "development",
    "validation",
    "final_holdout",
)
_DIRECTION_ORDER = ("positive", "negative", "mixed", "unavailable")
_QUANTILE_ORDER = ("p20", "p50", "p80")
_METRIC_ORDER: tuple[ProjectionMetricName, ...] = (
    "median_absolute_error",
    "pinball_p20",
    "pinball_p50",
    "pinball_p80",
    "interval_width",
    "interval_inclusion",
    "interval_score",
)
_MODEL_NAMES = (
    FHS_METHOD_VERSION,
    "zero_log_drift_gaussian",
    "historical_log_drift_gaussian",
)


@dataclass(frozen=True, slots=True)
class _HistoryRow:
    close: float
    volume: float | None


@dataclass(frozen=True, slots=True)
class _LoadedStudySource:
    analysis: StockAnalysis
    listing: Listing
    prediction_rows: tuple[Prediction, ...]
    selection: ProductSourceSelection
    source_execution: SourceExecutionBinding
    source_decision_time: datetime
    calculation_artifact: DataAsset
    stock_rows: dict[date, _HistoryRow]
    benchmark_rows: dict[date, _HistoryRow]
    stock_history_start: date
    stock_history_end: date
    benchmark_history_start: date
    benchmark_history_end: date


def study_price_product_run(
    *,
    run: AnalysisRun,
    store: AssetStore,
    all_selected: bool = False,
    listing_ids: tuple[UUID, ...] = (),
    config: PriceProductConfig | None = None,
    report_generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Replay the frozen retrospective protocol for one immutable source run.

    The replay is always read-only and current-vintage research-grade
    reconstruction. It does not create or mutate ``JobRun``, ``AnalysisRun``,
    ``Prediction``, ``StockAnalysis``, or ``DataAsset`` rows.
    """

    config = config or load_price_product_config()
    requested_listing_ids = _validate_scope(all_selected=all_selected, listing_ids=listing_ids)
    if report_generated_at is not None and (
        report_generated_at.tzinfo is None
        or report_generated_at.tzinfo.utcoffset(report_generated_at) is None
    ):
        raise ValueError("report_generated_at must be timezone-aware")

    verify_price_product_output(run=run, store=store, replay=False)
    protocol_sessions = _protocol_sessions(config)
    anchor_plan = build_anchor_plan(protocol_sessions, config=config)
    if anchor_plan.unavailable_reason is not None:
        raise ValueError(
            f"The frozen replay protocol calendar is unavailable: {anchor_plan.unavailable_reason}"
        )

    loaded = _load_study_sources(
        run=run,
        store=store,
        config=config,
        requested_listing_ids=requested_listing_ids,
    )
    projection_observations: list[ReplayMetricObservation] = []
    momentum_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    listing_reports: list[dict[str, Any]] = []
    expected_horizons = tuple(name for name, _sessions in config.simulation.horizons)
    expected_models = _MODEL_NAMES

    for source in loaded:
        listing_report, listing_projection_rows, listing_momentum, listing_convergence = (
            _study_loaded_source(
                source=source,
                run=run,
                protocol_sessions=protocol_sessions,
                anchor_plan=anchor_plan,
                config=config,
            )
        )
        listing_reports.append(listing_report)
        projection_observations.extend(listing_projection_rows)
        momentum_rows.extend(listing_momentum)
        convergence_rows.extend(listing_convergence)

    projection_aggregates = aggregate_projection_metrics(
        projection_observations,
        expected_partitions=_PARTITION_ORDER,
        expected_horizons=expected_horizons,
        expected_models=expected_models,
    )
    paired_comparisons = _paired_model_comparisons(
        projection_observations,
        config=config,
        candidate_model=FHS_METHOD_VERSION,
    )

    report: dict[str, Any] = {
        "schema": REPLAY_STUDY_SCHEMA,
        "execution": {"code_revision": code_revision()},
        "source_run": {
            "id": run.id,
            "generated_at": run.generated_at,
            "data_cutoff": run.data_cutoff,
            "target_date": run.target_date,
            "issued_on_time": run.issued_on_time,
            "universe_snapshot_id": run.universe_snapshot_id,
            "snapshot_grade": run.universe_snapshot.grade,
            "config_version": run.config_version,
            "config_hash": run.config_hash,
            "code_revision": run.code_revision,
            "status": run.status,
        },
        "scope": {
            "all_selected": all_selected,
            "requested_listing_ids": tuple(sorted(requested_listing_ids, key=str)),
            "studied_listing_ids": tuple(source.listing.id for source in loaded),
            "source_run_listing_count": StockAnalysis.objects.filter(run=run).count(),
            "studied_listing_count": len(loaded),
        },
        "config_identity": {
            "product_version": config.product_version,
            "effective_config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
            "payload_schema": config.payload_schema,
            "calendar": config.calendar,
            "currency": config.currency,
            "price_provider": config.price_provider,
            "benchmark_subject": config.benchmark_subject,
            "return_basis": config.return_basis,
            "dividends_included": config.dividends_included,
        },
        "protocol_identity": {
            "evidence_label": config.replay.evidence_label,
            "fixed_epoch": config.replay.fixed_epoch,
            "development_end_exclusive": config.replay.development_end_exclusive,
            "validation_start": config.replay.validation_start,
            "validation_end_exclusive": config.replay.validation_end_exclusive,
            "holdout_start": config.replay.holdout_start,
            "holdout_complete_through": config.replay.holdout_complete_through,
            "anchor_spacing": config.replay.anchor_spacing,
            "purge_partition_crossings": config.replay.purge_partition_crossings,
            "comparators": config.replay.comparators,
            "required_prior_returns": config.simulation.return_observations,
            "production_paths": config.simulation.production_paths,
            "diagnostic_paths": config.simulation.diagnostic_max_paths,
            "convergence_threshold_rule": "max(0.01 return, 0.02 * production_interval_width)",
            "metric_precision": replay_precision(config),
        },
        "calendar_identity": {
            "calendar": config.calendar,
            "first_session": protocol_sessions[0],
            "last_session": protocol_sessions[-1],
            "session_count": len(protocol_sessions),
        },
        "anchor_plan": anchor_plan,
        "projection_aggregates": projection_aggregates,
        "paired_model_comparisons": paired_comparisons,
        "momentum_aggregates": _aggregate_momentum_rows(momentum_rows),
        "convergence_aggregates": _aggregate_convergence_rows(convergence_rows),
        "listings": listing_reports,
        "disclosures": _disclosures(loaded=loaded, config=config),
    }
    if report_generated_at is not None:
        report["report_generated_at"] = report_generated_at
    return report


def serialize_price_product_study(
    report: Mapping[str, Any] | dict[str, Any],
    *,
    include_generated_at: bool = True,
) -> dict[str, Any]:
    """Return a deterministic JSON-safe document for a study report."""

    document = _jsonable(report)
    if not isinstance(document, dict):
        raise ValueError("Serialized study report must be a mapping")
    if not include_generated_at:
        document.pop("report_generated_at", None)
    return document


def render_price_product_study(
    report: Mapping[str, Any] | dict[str, Any],
    *,
    output_format: str,
    include_generated_at: bool = True,
) -> str:
    """Render a serialized study report as JSON or compact text."""

    document = serialize_price_product_study(report, include_generated_at=include_generated_at)
    if output_format == "json":
        return json.dumps(document, sort_keys=True, indent=2)
    if output_format != "text":
        raise ValueError("output_format must be json or text")
    return _render_text(document)


def _validate_scope(*, all_selected: bool, listing_ids: tuple[UUID, ...]) -> tuple[UUID, ...]:
    if all_selected == bool(listing_ids):
        raise ValueError("Provide exactly one of --all-selected or --listing-ids")
    if len(set(listing_ids)) != len(listing_ids):
        raise ValueError("listing_ids must not contain duplicates")
    return tuple(sorted(listing_ids, key=str))


def _protocol_sessions(config: PriceProductConfig) -> tuple[date, ...]:
    calendar = get_calendar(config.calendar)
    fixed_epoch = calendar.date_to_session(config.replay.fixed_epoch, direction="none")
    start = calendar.sessions_window(fixed_epoch, -config.simulation.return_observations)[0].date()
    return tuple(
        session.date()
        for session in calendar.sessions_in_range(start, config.replay.holdout_complete_through)
    )


def _load_study_sources(
    *,
    run: AnalysisRun,
    store: AssetStore,
    config: PriceProductConfig,
    requested_listing_ids: tuple[UUID, ...],
) -> tuple[_LoadedStudySource, ...]:
    analyses = list(
        StockAnalysis.objects.select_related("listing__security")
        .filter(run=run)
        .order_by("listing_id")
    )
    if not analyses:
        raise ValueError("The selected run has no product analyses")
    by_listing_id = {analysis.listing_id: analysis for analysis in analyses}
    if requested_listing_ids:
        missing = [str(item) for item in requested_listing_ids if item not in by_listing_id]
        if missing:
            raise ValueError("Requested study listings are absent from the source run")
        analyses = [by_listing_id[item] for item in requested_listing_ids]

    prediction_rows = list(
        Prediction.objects.select_related("listing")
        .filter(analysis__run=run, analysis__in=analyses)
        .order_by("analysis_id", "method_version", "evidence_role", "horizon", "id")
    )
    rows_by_analysis: dict[int, list[Prediction]] = defaultdict(list)
    for row in prediction_rows:
        rows_by_analysis[row.analysis_id].append(row)

    membership = product_membership_payload(run.universe_snapshot, store=store)
    source_decision_time = _parse_aware_datetime(
        membership.get("decision_time"),
        message="The captured source decision boundary is invalid",
    )

    loaded: list[_LoadedStudySource] = []
    for analysis in analyses:
        rows = tuple(rows_by_analysis.get(analysis.id, ()))
        if len(rows) != 5:
            raise ValueError("The selected run does not have the expected five-row product ledger")
        loaded.append(
            _load_study_source(
                analysis=analysis,
                rows=rows,
                run=run,
                store=store,
                config=config,
                membership=membership,
                source_decision_time=source_decision_time,
            )
        )
    return tuple(loaded)


def _load_study_source(
    *,
    analysis: StockAnalysis,
    rows: tuple[Prediction, ...],
    run: AnalysisRun,
    store: AssetStore,
    config: PriceProductConfig,
    membership: dict[str, object],
    source_decision_time: datetime,
) -> _LoadedStudySource:
    artifact_ref = _single_artifact_ref(rows)
    artifact = resolve_asset_ref(AssetRef.from_json(artifact_ref), cutoff=run.generated_at)
    if artifact.provider != "stanstock" or artifact.kind != CALCULATION_ARTIFACT_KIND:
        raise ValueError("The product calculation artifact has the wrong registered identity")
    try:
        document = json.loads(read_checksummed_bytes(store, artifact))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("The product calculation artifact is malformed") from exc
    if not isinstance(document, dict) or document.get("contract") != CALCULATION_ARTIFACT_CONTRACT:
        raise ValueError("The product calculation artifact has an invalid contract")
    if document.get("listing_id") != str(analysis.listing_id) or document.get("run_id") != str(
        run.id
    ):
        raise ValueError("The product calculation artifact identity does not match the source run")

    source_execution = _source_execution(document.get("source_execution"))
    source_window = document.get("source_window")
    if not isinstance(source_window, dict):
        raise ValueError("The product calculation artifact has no verified source window")
    source_window_input = price_product_input_from_verified_source_window(source_window)
    if source_window_input.decision_time != source_decision_time:
        raise ValueError("The product source window diverges from the captured decision boundary")
    if (
        source_window_input.target_date != run.target_date
        or source_window_input.listing_id != analysis.listing_id
    ):
        raise ValueError("The product source window diverges from the source run identity")

    source_assets = document.get("source_assets")
    if not isinstance(source_assets, list) or len(source_assets) != 4:
        raise ValueError("The product source closure is missing or ambiguous")
    resolved_assets = tuple(
        resolve_asset_ref(AssetRef.from_json(ref), cutoff=source_decision_time)
        for ref in source_assets
    )
    stock_asset, stock_raw_asset, benchmark_asset, benchmark_raw_asset = resolved_assets

    catalog_assets = _catalog_assets_for_membership(
        membership=membership,
        source_execution=source_execution,
        source_decision_time=source_decision_time,
    )
    _require_membership_bindings(
        membership=membership,
        analysis=analysis,
        source_execution=source_execution,
        stock_asset=stock_asset,
        benchmark_asset=benchmark_asset,
    )

    selection = select_product_source(
        listing=analysis.listing,
        target_date=run.target_date,
        decision_time=source_decision_time,
        provider=stock_asset.provider,
        benchmark_subject=benchmark_asset.subject,
        source_execution=source_execution,
        store=store,
        config=config,
        stock_asset=stock_asset,
        benchmark_asset=benchmark_asset,
        catalog_assets=catalog_assets,
    )
    if (
        selection.stock_raw_asset.id != stock_raw_asset.id
        or selection.benchmark_raw_asset.id != benchmark_raw_asset.id
        or selection.product_input != source_window_input
    ):
        raise ValueError("The retrospective loader could not rebind the exact selected source")
    if document.get("source_assets") != selection.source_assets:
        raise ValueError("The recorded source closure diverges from its resolved immutable assets")
    if document.get("catalog_assets") != [
        asset_ref_for(asset).to_json() for asset in catalog_assets
    ]:
        raise ValueError("The recorded catalog bindings diverge from captured source authority")

    asof = AsOfData(source_decision_time, store)
    stock_history = asof.price_frame_for_asset_with_diagnostics(
        asset=selection.stock_asset,
        through_date=run.target_date,
    )
    benchmark_history = asof.price_frame_for_asset_with_diagnostics(
        asset=selection.benchmark_asset,
        through_date=run.target_date,
    )
    stock_rows, stock_start, stock_end = _history_rows(
        stock_history.frame,
        invalid_session_date_rows=stock_history.invalid_session_date_rows,
        role="stock",
    )
    benchmark_rows, benchmark_start, benchmark_end = _history_rows(
        benchmark_history.frame,
        invalid_session_date_rows=benchmark_history.invalid_session_date_rows,
        role="benchmark",
    )

    return _LoadedStudySource(
        analysis=analysis,
        listing=analysis.listing,
        prediction_rows=rows,
        selection=selection,
        source_execution=source_execution,
        source_decision_time=source_decision_time,
        calculation_artifact=artifact,
        stock_rows=stock_rows,
        benchmark_rows=benchmark_rows,
        stock_history_start=stock_start,
        stock_history_end=stock_end,
        benchmark_history_start=benchmark_start,
        benchmark_history_end=benchmark_end,
    )


def _single_artifact_ref(rows: tuple[Prediction, ...]) -> dict[str, object]:
    refs = {
        json.dumps(row.calculation.get("calculation_artifact"), sort_keys=True)
        for row in rows
        if isinstance(row.calculation, dict)
        and row.calculation.get("calculation_artifact") is not None
    }
    if len(refs) != 1:
        raise ValueError("The product rows do not bind one exact calculation artifact")
    raw = json.loads(next(iter(refs)))
    if not isinstance(raw, dict):
        raise ValueError("The product calculation artifact reference is malformed")
    return raw


def _source_execution(raw: object) -> SourceExecutionBinding:
    if not isinstance(raw, dict):
        raise ValueError("The product source execution binding is malformed")
    try:
        mode = str(raw["mode"])
        evidence_grade = str(raw["evidence_grade"])
        if mode not in ("provider", "synthetic_demo") or evidence_grade not in (
            "research",
            "observed",
        ):
            raise ValueError
        return SourceExecutionBinding(
            mode=cast(SourceExecutionMode, mode),
            evidence_grade=cast(EvidenceGrade, evidence_grade),
        )
    except (KeyError, ValueError) as exc:
        raise ValueError("The product source execution binding is malformed") from exc


def _catalog_assets_for_membership(
    *,
    membership: dict[str, object],
    source_execution: SourceExecutionBinding,
    source_decision_time: datetime,
) -> tuple[DataAsset, ...]:
    if source_execution.mode != "provider":
        return ()
    raw_catalogs = membership.get("catalog_assets")
    if not isinstance(raw_catalogs, list):
        raise ValueError("Provider-backed replay requires captured catalog bindings")
    return tuple(
        resolve_asset_ref(AssetRef.from_json(raw), cutoff=source_decision_time)
        for raw in raw_catalogs
    )


def _require_membership_bindings(
    *,
    membership: dict[str, object],
    analysis: StockAnalysis,
    source_execution: SourceExecutionBinding,
    stock_asset: DataAsset,
    benchmark_asset: DataAsset,
) -> None:
    raw_ids = membership.get("qualified_listing_ids")
    if not isinstance(raw_ids, list) or str(analysis.listing_id) not in {
        str(item) for item in raw_ids
    }:
        raise ValueError("The captured membership does not authorize this study listing")
    if source_execution.mode != "provider":
        return
    admissions = membership.get("admissions")
    benchmark_ref = membership.get("benchmark_asset")
    if not isinstance(admissions, dict) or not isinstance(benchmark_ref, dict):
        raise ValueError("Provider-backed replay requires captured admission bindings")
    entry = admissions.get(analysis.listing.provider_symbol)
    if not isinstance(entry, dict):
        raise ValueError("The captured admissions do not include the study listing")
    if (
        entry.get("listing_id") != str(analysis.listing_id)
        or entry.get("status") != "admitted"
        or entry.get("price_asset") != asset_ref_for(stock_asset).to_json()
        or benchmark_ref != asset_ref_for(benchmark_asset).to_json()
    ):
        raise ValueError("The captured membership diverges from the selected replay source")


def _parse_aware_datetime(raw: object, *, message: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(raw))
    except ValueError as exc:
        raise ValueError(message) from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ValueError(message)
    return parsed


def _history_rows(
    frame: pl.DataFrame,
    *,
    invalid_session_date_rows: int,
    role: str,
) -> tuple[dict[date, _HistoryRow], date, date]:
    if invalid_session_date_rows != 0:
        raise ValueError(f"The selected {role} history has invalid session dates")
    if frame.is_empty():
        raise ValueError(f"The selected {role} history is empty")
    if "date" not in frame.columns or frame.schema.get("date") != pl.Date:
        raise ValueError(f"The selected {role} history does not expose a normalized date column")
    if "close" not in frame.columns or frame.schema.get("close") != pl.Float64:
        raise ValueError(f"The selected {role} history does not expose Float64 closes")
    if "volume" in frame.columns and frame.schema.get("volume") != pl.Int64:
        raise ValueError(f"The selected {role} history does not expose Int64 volumes")

    rows: dict[date, _HistoryRow] = {}
    dates = frame.get_column("date").to_list()
    if len(dates) != len(set(dates)):
        raise ValueError(f"The selected {role} history has duplicate session dates")
    columns = tuple(column for column in ("date", "close", "volume") if column in frame.columns)
    for row in frame.select(*columns).iter_rows(named=True):
        session_date = row["date"]
        close = row["close"]
        if type(session_date) is not date:
            raise ValueError(f"The selected {role} history has a non-date session key")
        if not isinstance(close, float) or not math.isfinite(close) or close <= 0:
            raise ValueError(f"The selected {role} history has a non-finite or non-positive close")
        volume_raw = row.get("volume")
        if volume_raw is not None and (type(volume_raw) is not int or volume_raw < 0):
            raise ValueError(f"The selected {role} history has an invalid volume")
        rows[session_date] = _HistoryRow(
            close=close,
            volume=None if volume_raw is None else float(volume_raw),
        )
    ordered = tuple(sorted(rows))
    if ordered != tuple(dates):
        raise ValueError(f"The selected {role} history is not strictly increasing")
    return rows, ordered[0], ordered[-1]


def _study_loaded_source(
    *,
    source: _LoadedStudySource,
    run: AnalysisRun,
    protocol_sessions: tuple[date, ...],
    anchor_plan: Any,
    config: PriceProductConfig,
) -> tuple[
    dict[str, Any],
    list[ReplayMetricObservation],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    anchor_index_by_date = {session: index for index, session in enumerate(protocol_sessions)}
    unique_anchor_dates = tuple(
        dict.fromkeys(
            anchor.anchor_date
            for partition in anchor_plan.partitions
            for anchor in partition.anchors
        )
    )
    computations = {
        anchor_date: _compute_anchor(
            source=source,
            anchor_date=anchor_date,
            anchor_index=anchor_index_by_date[anchor_date],
            protocol_sessions=protocol_sessions,
            config=config,
        )
        for anchor_date in unique_anchor_dates
    }

    listing_projection_rows: list[ReplayMetricObservation] = []
    listing_momentum: list[dict[str, Any]] = []
    listing_convergence: list[dict[str, Any]] = []
    anchor_entries: list[dict[str, Any]] = []

    for partition_plan in anchor_plan.partitions:
        for anchor in partition_plan.anchors:
            computed = computations[anchor.anchor_date]
            actual_stock, actual_benchmark, actual_reason = _actual_returns_for_anchor(
                source=source,
                anchor_date=anchor.anchor_date,
                outcome_end_date=anchor.outcome_end_date,
                config=config,
            )
            candidate_projection = computed["candidate_projections"].get(anchor.horizon)
            candidate_metrics = _projection_metrics(
                prediction_triplet=_projection_triplet(candidate_projection),
                actual_return=actual_stock,
                config=config,
                prediction_reason=(
                    computed["prediction_reason"] or _projection_reason(candidate_projection)
                ),
                actual_reason=actual_reason,
            )
            listing_projection_rows.append(
                ReplayMetricObservation(
                    listing_id=str(source.listing.id),
                    anchor_date=anchor.anchor_date,
                    target_date=anchor.outcome_end_date,
                    horizon=anchor.horizon,
                    partition=anchor.partition,
                    model_name=FHS_METHOD_VERSION,
                    metrics=candidate_metrics,
                )
            )

            model_entries = [
                {
                    "model_name": FHS_METHOD_VERSION,
                    "predicted_returns": _projection_triplet(candidate_projection),
                    "prediction_unavailable_reason": computed["prediction_reason"]
                    or _projection_reason(candidate_projection),
                    "metrics": candidate_metrics,
                }
            ]

            for comparator_name in config.replay.comparators:
                comparator_projection = computed["comparators"][comparator_name][anchor.horizon]
                comparator_metrics = _projection_metrics(
                    prediction_triplet=_projection_triplet(comparator_projection),
                    actual_return=actual_stock,
                    config=config,
                    prediction_reason=_projection_reason(comparator_projection),
                    actual_reason=actual_reason,
                )
                listing_projection_rows.append(
                    ReplayMetricObservation(
                        listing_id=str(source.listing.id),
                        anchor_date=anchor.anchor_date,
                        target_date=anchor.outcome_end_date,
                        horizon=anchor.horizon,
                        partition=anchor.partition,
                        model_name=comparator_name,
                        metrics=comparator_metrics,
                    )
                )
                model_entries.append(
                    {
                        "model_name": comparator_name,
                        "predicted_returns": _projection_triplet(comparator_projection),
                        "prediction_unavailable_reason": _projection_reason(comparator_projection),
                        "metrics": comparator_metrics,
                    }
                )

            momentum_entry: dict[str, Any] | None = None
            if anchor.horizon == "6m":
                momentum_entry = _momentum_entry(
                    source=source,
                    anchor=anchor,
                    computed=computed,
                    actual_stock=actual_stock,
                    actual_benchmark=actual_benchmark,
                    actual_reason=actual_reason,
                    config=config,
                )
                listing_momentum.append(momentum_entry)

            convergence_entry = _convergence_entry(
                source=source,
                anchor=anchor,
                computed=computed,
            )
            listing_convergence.extend(convergence_entry)

            anchor_entries.append(
                {
                    "partition": anchor.partition,
                    "horizon": anchor.horizon,
                    "horizon_sessions": anchor.horizon_sessions,
                    "anchor_date": anchor.anchor_date,
                    "outcome_end_date": anchor.outcome_end_date,
                    "feature_window": computed["feature_window"],
                    "actual_returns": {
                        "stock_return": _round_return(actual_stock, config=config),
                        "benchmark_return": _round_return(actual_benchmark, config=config),
                        "relative_return": (
                            None
                            if actual_stock is None or actual_benchmark is None
                            else _round_return(actual_stock - actual_benchmark, config=config)
                        ),
                        "unavailable_reason": actual_reason,
                    },
                    "momentum": momentum_entry,
                    "models": model_entries,
                    "convergence": convergence_entry,
                }
            )

    expected_horizons = tuple(name for name, _sessions in config.simulation.horizons)
    listing_report = {
        "listing_id": source.listing.id,
        "ticker": source.listing.ticker,
        "provider_symbol": source.listing.provider_symbol,
        "analysis_id": source.analysis.id,
        "prediction_ids": _prediction_id_matrix(source.prediction_rows),
        "source_execution": asdict(source.source_execution),
        "source_decision_time": source.source_decision_time,
        "source_target_date": run.target_date,
        "calculation_artifact": asset_ref_for(source.calculation_artifact).to_json(),
        "selected_assets": {
            "stock": {
                "normalized": asset_ref_for(source.selection.stock_asset).to_json(),
                "raw": asset_ref_for(source.selection.stock_raw_asset).to_json(),
            },
            "benchmark": {
                "normalized": asset_ref_for(source.selection.benchmark_asset).to_json(),
                "raw": asset_ref_for(source.selection.benchmark_raw_asset).to_json(),
            },
            "catalog_assets": [
                asset_ref_for(asset).to_json() for asset in source.selection.catalog_assets
            ],
        },
        "source_window": {
            "window_start": source.selection.product_input.calendar_sessions[0],
            "window_end": source.selection.product_input.calendar_sessions[-1],
            "window_session_count": len(source.selection.product_input.calendar_sessions),
        },
        "full_history": {
            "stock_start": source.stock_history_start,
            "stock_end": source.stock_history_end,
            "stock_session_count": len(source.stock_rows),
            "benchmark_start": source.benchmark_history_start,
            "benchmark_end": source.benchmark_history_end,
            "benchmark_session_count": len(source.benchmark_rows),
        },
        "projection_aggregates": aggregate_projection_metrics(
            listing_projection_rows,
            expected_partitions=_PARTITION_ORDER,
            expected_horizons=expected_horizons,
            expected_models=_MODEL_NAMES,
        ),
        "paired_model_comparisons": _paired_model_comparisons(
            listing_projection_rows,
            config=config,
            candidate_model=FHS_METHOD_VERSION,
        ),
        "momentum_aggregates": _aggregate_momentum_rows(listing_momentum),
        "convergence_aggregates": _aggregate_convergence_rows(listing_convergence),
        "anchors": anchor_entries,
    }
    return listing_report, listing_projection_rows, listing_momentum, listing_convergence


def _compute_anchor(
    *,
    source: _LoadedStudySource,
    anchor_date: date,
    anchor_index: int,
    protocol_sessions: tuple[date, ...],
    config: PriceProductConfig,
) -> dict[str, Any]:
    window_sessions = protocol_sessions[
        anchor_index - config.simulation.return_observations : anchor_index + 1
    ]
    feature_window = {
        "window_start": window_sessions[0],
        "window_end": window_sessions[-1],
        "window_session_count": len(window_sessions),
        "stock_available_sessions": sum(1 for day in window_sessions if day in source.stock_rows),
        "benchmark_available_sessions": sum(
            1 for day in window_sessions if day in source.benchmark_rows
        ),
    }

    stock_series, benchmark_series, prediction_reason = _anchor_series(
        source=source,
        window_sessions=window_sessions,
        config=config,
    )
    if prediction_reason is not None or stock_series is None or benchmark_series is None:
        return {
            "feature_window": feature_window,
            "prediction_reason": prediction_reason or "anchor_window_unavailable",
            "candidate_projections": {},
            "recommendation": None,
            "momentum_direction": None,
            "comparators": _unavailable_comparators(
                config,
                reason=prediction_reason or "anchor_window_unavailable",
            ),
            "convergence_rows": {},
        }

    anchor_input = PriceProductInput(
        listing_id=source.listing.id,
        target_date=anchor_date,
        decision_time=source.source_decision_time,
        calendar_sessions=window_sessions,
        stock=stock_series,
        benchmark=benchmark_series,
        source_execution=source.source_execution,
    )
    try:
        result = calculate_price_product(
            anchor_input,
            config=config,
            effective_config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
        )
        filtered = filter_historical_returns(
            stock_series.closes,
            burn_in=config.simulation.filter_burn_in,
            variance_target_weight=config.simulation.variance_target_weight,
            variance_persistence=config.simulation.variance_persistence,
            innovation_weight=config.simulation.innovation_weight,
        )
    except PriceProductInputError as exc:
        reason = exc.reason_code
        return {
            "feature_window": feature_window,
            "prediction_reason": reason,
            "candidate_projections": {},
            "recommendation": None,
            "momentum_direction": None,
            "comparators": _unavailable_comparators(config, reason=reason),
            "convergence_rows": {},
        }

    candidate_projections = {
        projection.horizon: {
            "ledger_returns": projection.ledger_returns,
            "insufficiency_reason": projection.insufficiency_reason,
        }
        for projection in result.forecast.projections
    }
    comparators = _comparators(filtered=filtered, config=config)
    convergence_report = calculate_fhs_convergence(
        filtered,
        seed=result.forecast.seed,
        config=config,
    )
    convergence_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in convergence_report.rows:
        convergence_rows[row.horizon].append(_jsonable(row))

    return {
        "feature_window": feature_window,
        "prediction_reason": None,
        "candidate_projections": candidate_projections,
        "recommendation": result.recommendation.suggestion,
        "momentum_direction": None if result.momentum is None else result.momentum.direction,
        "comparators": comparators,
        "convergence_rows": dict(convergence_rows),
    }


def _anchor_series(
    *,
    source: _LoadedStudySource,
    window_sessions: tuple[date, ...],
    config: PriceProductConfig,
) -> tuple[PriceSeries | None, PriceSeries | None, str | None]:
    missing_stock = [day for day in window_sessions if day not in source.stock_rows]
    if missing_stock:
        return None, None, "stock_feature_window_missing_session"
    missing_benchmark = [day for day in window_sessions if day not in source.benchmark_rows]
    if missing_benchmark:
        return None, None, "benchmark_feature_window_missing_session"

    stock_history = [source.stock_rows[day] for day in window_sessions]
    benchmark_history = [source.benchmark_rows[day] for day in window_sessions]
    try:
        return (
            PriceSeries(
                identity=source.selection.product_input.stock.identity,
                currency=config.currency,
                dates=window_sessions,
                closes=tuple(row.close for row in stock_history),
                volumes=tuple(row.volume for row in stock_history),
                volume_adjustment_compatible=(
                    source.selection.product_input.stock.volume_adjustment_compatible
                ),
            ),
            PriceSeries(
                identity=source.selection.product_input.benchmark.identity,
                currency=config.currency,
                dates=window_sessions,
                closes=tuple(row.close for row in benchmark_history),
                volumes=tuple(row.volume for row in benchmark_history),
                volume_adjustment_compatible=(
                    source.selection.product_input.benchmark.volume_adjustment_compatible
                ),
            ),
            None,
        )
    except (TypeError, ValueError, InvalidOperation):
        return None, None, "anchor_feature_window_invalid"


def _comparators(
    *,
    filtered: FilteredReturns,
    config: PriceProductConfig,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for comparator_name in config.replay.comparators:
        result[comparator_name] = {
            horizon: gaussian_comparator_quantiles(
                comparator=comparator_name,  # type: ignore[arg-type]
                mean_log_return=filtered.mean_log_return,
                population_variance=filtered.population_variance,
                horizon_sessions=horizon_sessions,
                config=config,
            )
            for horizon, horizon_sessions in config.simulation.horizons
        }
    return result


def _unavailable_comparators(
    config: PriceProductConfig, *, reason: str
) -> dict[str, dict[str, Any]]:
    return {
        comparator_name: {
            horizon: _prediction_only_placeholder(reason)
            for horizon, _horizon_sessions in config.simulation.horizons
        }
        for comparator_name in config.replay.comparators
    }


def _prediction_only_placeholder(reason: str) -> dict[str, object]:
    return {"ledger_returns": None, "insufficiency_reason": reason}


def _actual_returns_for_anchor(
    *,
    source: _LoadedStudySource,
    anchor_date: date,
    outcome_end_date: date,
    config: PriceProductConfig,
) -> tuple[float | None, float | None, str | None]:
    start_stock = source.stock_rows.get(anchor_date)
    end_stock = source.stock_rows.get(outcome_end_date)
    if start_stock is None or end_stock is None:
        return None, None, "stock_outcome_endpoint_missing"
    start_benchmark = source.benchmark_rows.get(anchor_date)
    end_benchmark = source.benchmark_rows.get(outcome_end_date)
    if start_benchmark is None or end_benchmark is None:
        return None, None, "benchmark_outcome_endpoint_missing"
    try:
        stock_return = (end_stock.close / start_stock.close) - 1.0
        benchmark_return = (end_benchmark.close / start_benchmark.close) - 1.0
    except ZeroDivisionError:
        return None, None, "outcome_endpoint_invalid"
    if not math.isfinite(stock_return) or not math.isfinite(benchmark_return):
        return None, None, "outcome_endpoint_invalid"
    if (
        _round_return(stock_return, config=config) is None
        or _round_return(benchmark_return, config=config) is None
    ):
        return None, None, "outcome_return_unrepresentable"
    return stock_return, benchmark_return, None


def _projection_metrics(
    *,
    prediction_triplet: Any,
    actual_return: float | None,
    config: PriceProductConfig,
    prediction_reason: str | None,
    actual_reason: str | None,
) -> ProjectionMetricResult:
    if prediction_reason is not None:
        return _unavailable_projection_metrics(config=config, reason=prediction_reason)
    if actual_reason is not None:
        return _unavailable_projection_metrics(config=config, reason=actual_reason)
    return score_projection_metrics(
        predicted_returns=prediction_triplet,
        actual_return=actual_return,
        config=config,
    )


def _projection_triplet(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get("ledger_returns")
    return getattr(value, "ledger_returns", None)


def _projection_reason(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        raw = value.get("insufficiency_reason") or value.get("unavailable_reason")
        return None if raw in (None, "") else str(raw)
    raw = getattr(value, "insufficiency_reason", None) or getattr(
        value,
        "unavailable_reason",
        None,
    )
    return None if raw in (None, "") else str(raw)


def _unavailable_projection_metrics(
    *,
    config: PriceProductConfig,
    reason: str,
) -> ProjectionMetricResult:
    precision = replay_precision(config)
    return ProjectionMetricResult(
        precision=precision,
        actual_return=None,
        predicted_returns=None,
        median_absolute_error=None,
        pinball_losses=None,
        interval_width=None,
        interval_included=None,
        interval_score=None,
        unavailable_reason=reason,
    )


def _round_return(value: float | None, *, config: PriceProductConfig) -> Decimal | None:
    if value is None or not math.isfinite(value):
        return None
    try:
        quantum = Decimal(1).scaleb(-config.rounding.return_decimal_places)
        return Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_EVEN)
    except InvalidOperation:
        return None


def _momentum_entry(
    *,
    source: _LoadedStudySource,
    anchor: Any,
    computed: dict[str, Any],
    actual_stock: float | None,
    actual_benchmark: float | None,
    actual_reason: str | None,
    config: PriceProductConfig,
) -> dict[str, Any]:
    if computed["prediction_reason"] is not None:
        diagnostic = evaluate_momentum_diagnostic(
            direction=None,
            suggestion=None,
            stock_return=actual_stock,
            benchmark_return=actual_benchmark,
            config=config,
        )
        unavailable_reason = computed["prediction_reason"]
    elif actual_reason is not None:
        diagnostic = evaluate_momentum_diagnostic(
            direction=computed["momentum_direction"],
            suggestion=computed["recommendation"],
            stock_return=None,
            benchmark_return=None,
            config=config,
        )
        unavailable_reason = actual_reason
    else:
        diagnostic = evaluate_momentum_diagnostic(
            direction=computed["momentum_direction"],
            suggestion=computed["recommendation"],
            stock_return=actual_stock,
            benchmark_return=actual_benchmark,
            config=config,
        )
        unavailable_reason = diagnostic.unavailable_reason
    return {
        "listing_id": source.listing.id,
        "partition": anchor.partition,
        "anchor_date": anchor.anchor_date,
        "target_date": anchor.outcome_end_date,
        "direction_group": computed["momentum_direction"],
        "suggestion": computed["recommendation"],
        "diagnostic": diagnostic,
        "unavailable_reason": unavailable_reason,
    }


def _convergence_entry(
    *,
    source: _LoadedStudySource,
    anchor: Any,
    computed: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = computed["convergence_rows"].get(anchor.horizon)
    if rows:
        return [
            {
                "listing_id": source.listing.id,
                "partition": anchor.partition,
                "anchor_date": anchor.anchor_date,
                "target_date": anchor.outcome_end_date,
                "horizon": anchor.horizon,
                "horizon_sessions": anchor.horizon_sessions,
                **row,
            }
            for row in rows
        ]
    return [
        {
            "listing_id": source.listing.id,
            "partition": anchor.partition,
            "anchor_date": anchor.anchor_date,
            "target_date": anchor.outcome_end_date,
            "horizon": anchor.horizon,
            "horizon_sessions": anchor.horizon_sessions,
            "quantile": quantile,
            "production_return": None,
            "diagnostic_return": None,
            "movement": None,
            "threshold": None,
            "exceeded": None,
            "unavailable_reason": computed["prediction_reason"] or "convergence_unavailable",
        }
        for quantile in _QUANTILE_ORDER
    ]


def _prediction_id_matrix(rows: tuple[Prediction, ...]) -> dict[str, str]:
    return {f"{row.method_version}:{row.evidence_role}:{row.horizon}": str(row.id) for row in rows}


def _paired_model_comparisons(
    observations: list[ReplayMetricObservation],
    *,
    config: PriceProductConfig,
    candidate_model: str,
) -> list[Any]:
    comparisons: list[Any] = []
    horizons = tuple(name for name, _sessions in config.simulation.horizons)
    for partition in _PARTITION_ORDER:
        for horizon in horizons:
            scoped = [
                row for row in observations if row.partition == partition and row.horizon == horizon
            ]
            for baseline in config.replay.comparators:
                pair_scope = [
                    row for row in scoped if row.model_name in {candidate_model, baseline}
                ]
                for metric_name in _METRIC_ORDER:
                    comparisons.append(
                        compare_aligned_models(
                            pair_scope,
                            candidate_model=candidate_model,
                            baseline_model=baseline,
                            partition=partition,
                            horizon=horizon,
                            metric_name=metric_name,
                        )
                    )
    return comparisons


def _aggregate_momentum_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for partition in _PARTITION_ORDER:
        for direction in _DIRECTION_ORDER:
            groups[(partition, direction)] = {
                "partition": partition,
                "direction_group": None if direction == "unavailable" else direction,
                "observation_count": 0,
                "suggestion_counts": Counter(),
                "success_true_count": 0,
                "success_false_count": 0,
                "success_null_count": 0,
                "direction_correct_true_count": 0,
                "direction_correct_false_count": 0,
                "direction_correct_null_count": 0,
                "unavailable_reasons": Counter(),
            }
    for row in rows:
        key = (row["partition"], row["direction_group"] or "unavailable")
        group = groups[key]
        group["observation_count"] += 1
        if row["suggestion"] is not None:
            group["suggestion_counts"][str(row["suggestion"])] += 1
        diagnostic = row["diagnostic"]
        if diagnostic.success is True:
            group["success_true_count"] += 1
        elif diagnostic.success is False:
            group["success_false_count"] += 1
        else:
            group["success_null_count"] += 1
        if diagnostic.direction_correct is True:
            group["direction_correct_true_count"] += 1
        elif diagnostic.direction_correct is False:
            group["direction_correct_false_count"] += 1
        else:
            group["direction_correct_null_count"] += 1
        if row["unavailable_reason"] is not None:
            group["unavailable_reasons"][str(row["unavailable_reason"])] += 1
    return [
        {
            **group,
            "suggestion_counts": dict(sorted(group["suggestion_counts"].items())),
            "unavailable_reasons": dict(sorted(group["unavailable_reasons"].items())),
        }
        for group in groups.values()
    ]


def _aggregate_convergence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    horizon_sessions = {"6m": 126, "12m": 252, "3y": 756, "5y": 1260}
    for partition in _PARTITION_ORDER:
        for horizon in horizon_sessions:
            for quantile in _QUANTILE_ORDER:
                groups[(partition, horizon, quantile)] = {
                    "partition": partition,
                    "horizon": horizon,
                    "horizon_sessions": horizon_sessions[horizon],
                    "quantile": quantile,
                    "observation_count": 0,
                    "exceeded_count": 0,
                    "unavailable_count": 0,
                    "max_movement": None,
                    "max_threshold": None,
                    "unavailable_reasons": Counter(),
                }
    for row in rows:
        key = (row["partition"], row["horizon"], row["quantile"])
        group = groups[key]
        group["observation_count"] += 1
        if row["exceeded"] is True:
            group["exceeded_count"] += 1
        if row["unavailable_reason"] is not None:
            group["unavailable_count"] += 1
            group["unavailable_reasons"][str(row["unavailable_reason"])] += 1
        movement = row["movement"]
        threshold = row["threshold"]
        if isinstance(movement, (int, float)) and math.isfinite(float(movement)):
            current = group["max_movement"]
            group["max_movement"] = (
                float(movement) if current is None else max(current, float(movement))
            )
        if isinstance(threshold, (int, float)) and math.isfinite(float(threshold)):
            current = group["max_threshold"]
            group["max_threshold"] = (
                float(threshold) if current is None else max(current, float(threshold))
            )
    return [
        {
            **group,
            "unavailable_reasons": dict(sorted(group["unavailable_reasons"].items())),
        }
        for group in groups.values()
    ]


def _disclosures(
    *,
    loaded: tuple[_LoadedStudySource, ...],
    config: PriceProductConfig,
) -> list[str]:
    synthetic = any(source.source_execution.mode == "synthetic_demo" for source in loaded)
    disclosures = [
        (
            f"{config.replay.evidence_label}: current-universe/current-vintage research-only "
            "math reconstruction from exact immutable selected source assets; not historical "
            "observed availability and not an observed issuance path."
        ),
        (
            "Forecast comparisons are horizon- and partition-scoped, paired only on the exact "
            "same listing/anchor/maturity support, and averaged within target cohorts before "
            "equal-cohort aggregation."
        ),
        (
            "Monte Carlo path counts and 16,384-path convergence diagnostics are numerical "
            "engineering checks, not additional market evidence or probability calibration."
        ),
    ]
    if synthetic:
        disclosures.append(
            "Synthetic-demo replay is deterministic code-path validation only; it is not "
            "empirical evidence of historical or future forecast skill."
        )
    return disclosures


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(cast(Any, value)))
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Counter):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _render_text(document: dict[str, Any]) -> str:
    lines = [
        "research-product retrospective math replay",
        (
            f"run_id={document['source_run']['id']} "
            f"target_date={document['source_run']['target_date']}"
        ),
        (
            f"snapshot_grade={document['source_run']['snapshot_grade']} "
            f"issued_on_time={document['source_run']['issued_on_time']}"
        ),
        (
            f"scope_listings={document['scope']['studied_listing_count']}/"
            f"{document['scope']['source_run_listing_count']} "
            f"evidence_label={document['protocol_identity']['evidence_label']}"
        ),
    ]
    generated_at = document.get("report_generated_at")
    if generated_at is not None:
        lines.append(f"report_generated_at={generated_at}")
    for disclosure in document["disclosures"]:
        lines.append(f"disclosure: {disclosure}")
    lines.append("projection aggregates:")
    for item in document["projection_aggregates"]:
        lines.append(
            "  "
            f"partition={item['partition']} horizon={item['horizon']} model={item['model_name']} "
            f"cohorts={item['target_cohort_count']} listings={item['distinct_listing_count']} "
            f"unavailable={item['unavailable_count']} "
            f"mae={item['averages']['median_absolute_error']} "
            f"coverage={item['averages']['interval_inclusion']} "
            f"reason={item['insufficient_reason']}"
        )
    lines.append("paired comparisons:")
    for item in document["paired_model_comparisons"]:
        lines.append(
            "  "
            f"partition={item['partition']} horizon={item['horizon']} "
            f"baseline={item['baseline_model']} metric={item['metric_name']} "
            f"paired={item['paired_observation_count']} "
            f"cohorts={item['paired_target_cohort_count']} "
            f"candidate={item['candidate_average']} baseline={item['baseline_average']} "
            f"delta={item['mean_difference_candidate_minus_baseline']} "
            f"reason={item['unavailable_reason']}"
        )
    lines.append("momentum aggregates:")
    for item in document["momentum_aggregates"]:
        lines.append(
            "  "
            f"partition={item['partition']} direction={item['direction_group']} "
            f"count={item['observation_count']} success_true={item['success_true_count']} "
            f"success_false={item['success_false_count']} success_null={item['success_null_count']}"
        )
    return "\n".join(lines)

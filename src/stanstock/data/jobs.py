from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from django.db import transaction
from django.db.models import Max
from django.utils import timezone

from stanstock.core.jobs import JobExecutionResult, execute_target_job, target_job_lock
from stanstock.core.models import JobRun
from stanstock.data.assets import asset_ref_for
from stanstock.data.live_us import (
    UsUniverseConfig,
    completed_us_analysis_run,
    load_us_universe_config,
    resolve_us_target_date,
    run_us_daily,
)
from stanstock.data.models import DataAsset, ProviderRecord, UniverseSnapshot
from stanstock.data.providers import sec, twelve_data

JOB_NAME = "daily"
LONG_FORECAST_GATE_DETAIL = "long_forecast_requested"
TARGET_GATE_RESERVATION_DETAIL = "target_gate_reservation"
TARGET_GATE_OWNER_DETAIL = "target_gate_owner"


@dataclass(frozen=True, slots=True)
class PreparedUsDailyJob:
    config: UsUniverseConfig
    target_date: date
    snapshot_grade: str
    decision_time: datetime


@dataclass(frozen=True, slots=True)
class FrozenUsDailyTargetGate:
    long_forecast_requested: bool
    reservation_id: UUID | None
    market_output_committed: bool


def prepare_us_daily_job(
    *,
    config_path: Path,
    explicit_target: date | None = None,
    decision_time: datetime | None = None,
) -> PreparedUsDailyJob:
    effective_time = decision_time or timezone.now()
    config = load_us_universe_config(config_path)
    target_date, snapshot_grade = resolve_us_target_date(
        decision_time=effective_time,
        explicit_target=explicit_target,
    )
    return PreparedUsDailyJob(
        config=config,
        target_date=target_date,
        snapshot_grade=snapshot_grade,
        decision_time=effective_time,
    )


def _catalog_refs(catalog_asset_ids: tuple[Any, ...]) -> list[dict[str, str]]:
    """Full-identity `AssetRef` payloads for each catalog asset, in order.

    Additive alongside the plain-id `catalog_asset_ids` (kept for
    backwards compatibility): verification authority binds to these
    checksummed refs, not the bare ids.
    """
    assets_by_id = {asset.pk: asset for asset in DataAsset.objects.filter(pk__in=catalog_asset_ids)}
    missing = [asset_id for asset_id in catalog_asset_ids if asset_id not in assets_by_id]
    if missing:
        raise ValueError(f"Catalog asset ids could not be resolved: {missing}")
    return [asset_ref_for(assets_by_id[asset_id]).to_json() for asset_id in catalog_asset_ids]


def proposed_us_daily_target_gate(prepared: PreparedUsDailyJob) -> bool:
    """Return the scheduler's pre-lock proposal without overriding evidence.

    This read-only proposal exists only to detect a daily completion that
    races between scheduler intent resolution and the canonical locked
    reservation. The locked reservation repeats every validation and is the
    sole authoritative writer. Existing target evidence is returned without
    consulting mutable provider state.
    """
    attempts = JobRun.objects.filter(
        job_name=JOB_NAME,
        region="us",
        target_date=prepared.target_date,
    ).order_by("attempt")
    frozen_values: set[bool] = set()
    for attempt in attempts:
        if not isinstance(attempt.details, dict):
            raise ValueError(
                "An existing market attempt records malformed details for the frozen "
                "long-forecast invocation gate"
            )
        if LONG_FORECAST_GATE_DETAIL not in attempt.details:
            continue
        frozen = attempt.details[LONG_FORECAST_GATE_DETAIL]
        if not isinstance(frozen, bool):
            raise ValueError(
                "An existing market attempt records a malformed frozen long-forecast "
                "invocation gate"
            )
        frozen_values.add(frozen)
    if len(frozen_values) > 1:
        raise ValueError(
            "Existing market attempts record conflicting frozen long-forecast invocation gates"
        )
    if frozen_values:
        return next(iter(frozen_values))
    if (
        completed_us_analysis_run(
            config=prepared.config,
            target_date=prepared.target_date,
        )
        is not None
    ):
        raise ValueError(
            "Completed US analysis output exists without an independently persisted "
            "long-forecast invocation gate"
        )
    return _sample_current_long_forecast_requested()


def reserve_us_daily_target_gate(
    prepared: PreparedUsDailyJob,
    *,
    explicit_long_forecast_requested: bool | None = None,
    reservation_owner: str,
) -> FrozenUsDailyTargetGate:
    """Persist the target gate while holding the canonical daily lock."""
    with target_job_lock(
        job_name=JOB_NAME,
        region="us",
        target_date=prepared.target_date,
    ):
        return _reserve_us_daily_target_gate_locked(
            prepared,
            explicit_long_forecast_requested=explicit_long_forecast_requested,
            reservation_owner=reservation_owner,
        )


def _reserve_us_daily_target_gate_locked(
    prepared: PreparedUsDailyJob,
    *,
    explicit_long_forecast_requested: bool | None,
    reservation_owner: str,
    expected_reservation_id: UUID | None = None,
) -> FrozenUsDailyTargetGate:
    """Resolve persisted gates and create or adopt one provisional attempt.

    The caller already holds the canonical daily target lock. A reservation
    is a RUNNING daily ``JobRun`` carrying only the explicit
    ``target_gate_reservation`` marker and frozen gate. Success/recovery
    selectors cannot mistake it for completed market work; the later daily
    execution consumes this same row as its real attempt.
    """
    if explicit_long_forecast_requested is not None and not isinstance(
        explicit_long_forecast_requested, bool
    ):
        raise ValueError("long_forecast_requested must be a boolean")

    with transaction.atomic():
        attempts = list(
            JobRun.objects.select_for_update()
            .filter(
                job_name=JOB_NAME,
                region="us",
                target_date=prepared.target_date,
            )
            .order_by("attempt")
        )
        frozen_values: set[bool] = set()
        active_reservations: list[JobRun] = []
        successful_market = False
        for attempt in attempts:
            if not isinstance(attempt.details, dict):
                raise ValueError(
                    "An existing market attempt records malformed details for the frozen "
                    "long-forecast invocation gate"
                )
            if attempt.status == JobRun.Status.SUCCESS:
                successful_market = True
            if (
                attempt.status == JobRun.Status.RUNNING
                and attempt.details.get(TARGET_GATE_RESERVATION_DETAIL) is True
            ):
                active_reservations.append(attempt)
            if LONG_FORECAST_GATE_DETAIL not in attempt.details:
                continue
            frozen = attempt.details[LONG_FORECAST_GATE_DETAIL]
            if not isinstance(frozen, bool):
                raise ValueError(
                    "An existing market attempt records a malformed frozen long-forecast "
                    "invocation gate"
                )
            frozen_values.add(frozen)

        if len(frozen_values) > 1:
            raise ValueError(
                "Existing market attempts record conflicting frozen long-forecast invocation gates"
            )
        if len(active_reservations) > 1:
            raise ValueError("More than one active daily target-gate reservation exists")

        completed_output = (
            completed_us_analysis_run(
                config=prepared.config,
                target_date=prepared.target_date,
            )
            is not None
        )
        market_output_committed = successful_market or completed_output
        if completed_output and not frozen_values:
            raise ValueError(
                "Completed US analysis output exists without an independently persisted "
                "long-forecast invocation gate"
            )

        if frozen_values:
            frozen = next(iter(frozen_values))
            if (
                explicit_long_forecast_requested is not None
                and explicit_long_forecast_requested != frozen
            ):
                raise ValueError(
                    "The explicit long-forecast invocation gate conflicts with the frozen "
                    "market target gate"
                )
        else:
            frozen = (
                explicit_long_forecast_requested
                if explicit_long_forecast_requested is not None
                else _sample_current_long_forecast_requested()
            )

        active = active_reservations[0] if active_reservations else None
        if expected_reservation_id is not None:
            if active is None or active.pk != expected_reservation_id:
                raise ValueError("The supplied daily target-gate reservation is no longer active")
        elif active is not None:
            owner_run: JobRun | None = None
            try:
                owner_id = UUID(str(active.details.get(TARGET_GATE_OWNER_DETAIL)))
            except (TypeError, ValueError):
                owner_id = None
            if owner_id is not None:
                owner_run = JobRun.objects.filter(pk=owner_id).first()
            if owner_run is not None and owner_run.status == JobRun.Status.RUNNING:
                raise ValueError(
                    "The daily target gate is reserved by another running orchestration"
                )
            active.details = {
                **active.details,
                TARGET_GATE_OWNER_DETAIL: reservation_owner,
            }
            active.save(update_fields=["details"])

        if active is None and not successful_market:
            latest_attempt = JobRun.objects.filter(
                job_name=JOB_NAME,
                region="us",
                target_date=prepared.target_date,
            ).aggregate(max_attempt=Max("attempt"))["max_attempt"]
            active = JobRun.objects.create(
                job_name=JOB_NAME,
                region="us",
                target_date=prepared.target_date,
                attempt=int(latest_attempt or 0) + 1,
                details={
                    TARGET_GATE_RESERVATION_DETAIL: True,
                    TARGET_GATE_OWNER_DETAIL: reservation_owner,
                    LONG_FORECAST_GATE_DETAIL: frozen,
                },
            )

        return FrozenUsDailyTargetGate(
            long_forecast_requested=frozen,
            reservation_id=active.pk if active is not None else None,
            market_output_committed=market_output_committed,
        )


def frozen_long_forecast_gate_for_run(run: JobRun) -> bool:
    """Return the gate on the exact authoritative market run behind ``run``."""
    if (
        run.job_name != JOB_NAME
        or run.region != "us"
        or run.status not in {JobRun.Status.SUCCESS, JobRun.Status.SKIPPED}
    ):
        raise ValueError("Market job run does not identify a satisfied US daily target")

    authoritative = run
    if run.status == JobRun.Status.SKIPPED:
        details = run.details if isinstance(run.details, dict) else {}
        raw_success_id = details.get("successful_run_id")
        try:
            success_id = uuid.UUID(str(raw_success_id))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Skipped market job run has a malformed successful-run reference"
            ) from exc
        referenced = JobRun.objects.filter(pk=success_id).first()
        if referenced is None:
            raise ValueError("Skipped market job run references a missing successful run")
        authoritative = referenced
        if (
            authoritative.job_name != JOB_NAME
            or authoritative.region != "us"
            or authoritative.target_date != run.target_date
            or authoritative.status != JobRun.Status.SUCCESS
        ):
            raise ValueError(
                "Skipped market job run does not resolve to a successful matching target"
            )

    if not isinstance(authoritative.details, dict):
        raise ValueError("Authoritative market job run records malformed details")
    frozen = authoritative.details.get(LONG_FORECAST_GATE_DETAIL)
    if not isinstance(frozen, bool):
        raise ValueError(
            "Authoritative market job run does not record a valid frozen "
            "long-forecast invocation gate"
        )
    return frozen


def execute_us_daily_job(
    prepared: PreparedUsDailyJob,
    *,
    require_observed: bool = False,
    long_forecast_requested: bool | None = None,
    target_gate_reservation_id: UUID | None = None,
) -> JobRun:
    effective_long_forecast_requested: bool | None = None

    def _resolve_gate() -> JobRun | None:
        nonlocal effective_long_forecast_requested
        gate = _reserve_us_daily_target_gate_locked(
            prepared,
            explicit_long_forecast_requested=long_forecast_requested,
            reservation_owner="daily",
            expected_reservation_id=target_gate_reservation_id,
        )
        effective_long_forecast_requested = gate.long_forecast_requested
        if gate.reservation_id is None:
            return None
        return JobRun.objects.get(pk=gate.reservation_id)

    def _task(run: JobRun) -> JobExecutionResult:
        if effective_long_forecast_requested is None:
            raise RuntimeError("US daily target gate was not resolved")
        # Persist the independently owned invocation input before any
        # producer can write an AnalysisRun, prediction, or output asset.
        JobRun.objects.filter(pk=run.pk).update(
            details={LONG_FORECAST_GATE_DETAIL: effective_long_forecast_requested}
        )
        if require_observed and prepared.snapshot_grade != UniverseSnapshot.Grade.OBSERVED:
            raise ValueError(
                f"Automatic research-grade catch-up is forbidden for "
                f"{prepared.target_date.isoformat()}; run an explicit manual "
                "target-date reconstruction instead."
            )
        result = run_us_daily(
            config=prepared.config,
            target_date=prepared.target_date,
            snapshot_grade=prepared.snapshot_grade,
            decision_time=prepared.decision_time,
            require_on_time=require_observed,
            long_forecast_requested=effective_long_forecast_requested,
        )
        return JobExecutionResult(
            details={
                LONG_FORECAST_GATE_DETAIL: effective_long_forecast_requested,
                "snapshot_id": str(result.snapshot.id),
                "snapshot_grade": result.snapshot.grade,
                "analysis_run_id": str(result.analysis_run_id),
                "provider": twelve_data.PROVIDER,
                "benchmark_subject": result.benchmark_symbol,
                "eligible": result.eligible,
                "excluded": result.excluded,
                "price_assets": result.price_assets,
                "raw_assets": result.raw_assets,
                "credits_used": result.credits_used,
                "analyses": result.analyses,
                "predictions": result.predictions,
                "catalog_asset_ids": [str(asset_id) for asset_id in result.catalog_asset_ids],
                "catalog_refs": _catalog_refs(result.catalog_asset_ids),
            }
        )

    run = execute_target_job(
        job_name=JOB_NAME,
        region="us",
        target_date=prepared.target_date,
        task=_task,
        before_attempt=_resolve_gate,
    )
    if effective_long_forecast_requested is None:
        raise RuntimeError("US daily target gate was not resolved")
    authoritative_gate = frozen_long_forecast_gate_for_run(run)
    if authoritative_gate != effective_long_forecast_requested:
        raise ValueError("Authoritative market job gate does not match the resolved target gate")
    return run


def _sample_current_long_forecast_requested() -> bool:
    return ProviderRecord.objects.filter(
        provider=sec.PROVIDER,
        enabled=True,
    ).exists()

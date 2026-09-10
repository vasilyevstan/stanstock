from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from django.utils import timezone

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun
from stanstock.data.assets import asset_ref_for
from stanstock.data.live_us import (
    UsUniverseConfig,
    load_us_universe_config,
    resolve_us_target_date,
    run_us_daily,
)
from stanstock.data.models import DataAsset, UniverseSnapshot
from stanstock.data.providers import twelve_data

JOB_NAME = "daily"


@dataclass(frozen=True, slots=True)
class PreparedUsDailyJob:
    config: UsUniverseConfig
    target_date: date
    snapshot_grade: str
    decision_time: datetime


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


def execute_us_daily_job(
    prepared: PreparedUsDailyJob,
    *,
    require_observed: bool = False,
) -> JobRun:
    def _task(run: JobRun) -> JobExecutionResult:
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
        )
        return JobExecutionResult(
            details={
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

    return execute_target_job(
        job_name=JOB_NAME,
        region="us",
        target_date=prepared.target_date,
        task=_task,
    )

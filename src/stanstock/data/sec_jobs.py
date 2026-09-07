from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from django.db import transaction
from django.utils import timezone

from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun
from stanstock.data.live_us import load_us_universe_config
from stanstock.data.management.config_loader import (
    default_sec_cik_mapping_path,
    default_sec_fundamentals_config_path,
    default_us_universe_config_path,
)
from stanstock.data.models import ProviderRecord
from stanstock.data.providers.exceptions import ProviderError
from stanstock.data.sec_config import load_sec_cik_config, load_sec_fundamentals_config
from stanstock.data.sec_ingestion import run_sec_ingestion

JOB_NAME = "sec_fundamentals"


def execute_sec_fundamentals_job(
    *,
    target_date: date,
    fundamentals_config_path: Path | None = None,
    cik_config_path: Path | None = None,
    universe_config_path: Path | None = None,
) -> JobRun:
    fundamentals_config = load_sec_fundamentals_config(
        fundamentals_config_path or default_sec_fundamentals_config_path()
    )
    cik_config = load_sec_cik_config(cik_config_path or default_sec_cik_mapping_path())
    universe_config = load_us_universe_config(
        universe_config_path or default_us_universe_config_path()
    )

    def _task(run: JobRun) -> JobExecutionResult:
        try:
            result = run_sec_ingestion(
                config=fundamentals_config,
                cik_config=cik_config,
                universe_config=universe_config,
                target_date=target_date,
            )
        except (OSError, ProviderError, ValueError) as exc:
            record, _created = ProviderRecord.objects.get_or_create(provider="sec")
            record.status = "error"
            record.last_error = f"{type(exc).__name__}: {exc}"
            record.save(update_fields=["status", "last_error"])
            raise
        details = {
            "provider": "sec",
            "mapping_asset_id": result.mapping_asset_id,
            "mapping_sha256": result.mapping_sha256,
            "companies": len(result.companies),
            "raw_assets_created": result.raw_assets_created,
            "raw_assets_reused": result.raw_assets_reused,
            "companyfacts_fetched": result.companyfacts_fetched,
            "facts_created": result.facts_created,
            "facts_reused": result.facts_reused,
            "config_version": fundamentals_config.config_version,
            "config_hash": fundamentals_config.config_hash,
            "cik_config_version": cik_config.config_version,
            "cik_config_hash": cik_config.config_hash,
        }
        _record_provider_success(target_date=target_date, details=details)
        return JobExecutionResult(details=details)

    run = execute_target_job(
        job_name=JOB_NAME,
        region="us",
        target_date=target_date,
        task=_task,
    )
    if run.status == JobRun.Status.SKIPPED:
        successful_run_id = run.details.get("successful_run_id")
        if successful_run_id:
            successful = JobRun.objects.filter(pk=successful_run_id).first()
            if successful is not None:
                _record_provider_success(
                    target_date=target_date,
                    details=successful.details,
                )
    return run


def _record_provider_success(
    *,
    target_date: date,
    details: dict[str, Any],
) -> None:
    with transaction.atomic():
        record, _created = ProviderRecord.objects.select_for_update().get_or_create(provider="sec")
        metadata = dict(record.metadata)
        metadata.update(
            {
                "last_ingestion_target": target_date.isoformat(),
                "last_ingestion_companies": details.get("companies", 0),
                "last_ingestion_facts_created": details.get("facts_created", 0),
                "last_ingestion_facts_reused": details.get("facts_reused", 0),
                "last_ingestion_raw_assets_created": details.get("raw_assets_created", 0),
                "last_ingestion_raw_assets_reused": details.get("raw_assets_reused", 0),
                "last_ingestion_companyfacts_fetched": details.get("companyfacts_fetched", 0),
                "mapping_sha256": details.get("mapping_sha256", ""),
            }
        )
        record.status = "ok"
        record.last_success_at = timezone.now()
        record.last_error = ""
        record.metadata = metadata
        record.save(update_fields=["status", "last_success_at", "last_error", "metadata"])

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.utils import timezone

from stanstock.core.models import JobRun
from stanstock.data import sec_jobs
from stanstock.data.models import ProviderRecord

pytestmark = pytest.mark.django_db


def _result() -> SimpleNamespace:
    return SimpleNamespace(
        mapping_asset_id="mapping-asset",
        mapping_sha256="a" * 64,
        companies=(object(), object()),
        raw_assets_created=5,
        raw_assets_reused=3,
        companyfacts_fetched=2,
        facts_created=100,
        facts_reused=20,
        asset_refs=(),
    )


def test_sec_job_records_provider_health_and_recovers_prior_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    ProviderRecord.objects.create(
        provider="sec",
        metadata={
            "companyfacts_verifications": {
                "0000320193": {
                    "checked_at": "2026-09-04T12:00:00+00:00",
                    "asset_sha256": "b" * 64,
                    "filing_sources_hash": "c" * 64,
                    "config_hash": "d" * 64,
                    "normalization_version": "sec-companyfacts-v2",
                    "missing_accessions": {},
                }
            }
        },
    )

    def run_sec_ingestion(**kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return _result()

    monkeypatch.setattr(sec_jobs, "run_sec_ingestion", run_sec_ingestion)

    first = sec_jobs.execute_sec_fundamentals_job(target_date=date(2026, 9, 4))
    second = sec_jobs.execute_sec_fundamentals_job(target_date=date(2026, 9, 4))

    assert first.status == JobRun.Status.SUCCESS
    assert second.status == JobRun.Status.SKIPPED
    assert calls == 1
    record = ProviderRecord.objects.get(provider="sec")
    assert record.status == "ok"
    assert record.last_error == ""
    assert record.metadata["last_ingestion_companies"] == 2
    assert record.metadata["last_ingestion_companyfacts_fetched"] == 2
    assert record.metadata["last_ingestion_facts_created"] == 100
    assert record.metadata["mapping_sha256"] == "a" * 64
    assert "0000320193" in record.metadata["companyfacts_verifications"]


def test_sec_job_records_expected_failure_without_swallowing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**kwargs: object) -> SimpleNamespace:
        raise ValueError("bad SEC payload")

    monkeypatch.setattr(sec_jobs, "run_sec_ingestion", fail)

    with pytest.raises(ValueError, match="bad SEC payload"):
        sec_jobs.execute_sec_fundamentals_job(target_date=date(2026, 9, 4))

    record = ProviderRecord.objects.get(provider="sec")
    assert record.status == "error"
    assert "bad SEC payload" in record.last_error
    assert JobRun.objects.get(job_name=sec_jobs.JOB_NAME).status == JobRun.Status.FAILED


def test_configure_sec_enables_only_after_successful_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SEC_USER_AGENT",
        "StanStockResearch/0.1 monitored@example.com",
    )
    record = ProviderRecord.objects.create(
        provider="sec",
        status="ok",
        last_success_at=timezone.now(),
    )

    call_command("configure_sec", enable=True, verbosity=0)

    record.refresh_from_db()
    assert record.enabled is True
    assert record.status == "ok"
    assert record.metadata["contact_configured"] is True
    assert "monitored@example.com" not in json.dumps(record.metadata)


def test_configure_sec_refuses_missing_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SEC_USER_AGENT",
        "StanStockResearch/0.1 monitored@example.com",
    )

    with pytest.raises(CommandError, match="has not passed"):
        call_command("configure_sec", enable=True, verbosity=0)

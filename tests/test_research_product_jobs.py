from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
from django.utils import timezone
from exchange_calendars import get_calendar

from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.live_us import _persist_catalog, _persist_price_series, load_us_universe_config
from stanstock.data.management.config_loader import default_us_universe_config_path
from stanstock.data.models import DataAsset, Listing, ProviderRecord
from stanstock.data.provider_policy import BASIC_USAGE_SCOPE, PRIVATE_USAGE_SCOPE
from stanstock.data.providers import twelve_data
from stanstock.data.providers.contracts import StockCatalog
from stanstock.data.providers.exceptions import ProviderConfigurationError, ProviderError
from stanstock.data.research_product import (
    PRODUCT_INTAKE_KIND,
    product_membership_payload,
)
from stanstock.data.research_product_jobs import execute_daily_research_job
from stanstock.portfolio.models import TrackedSymbol
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.product_pipeline import verify_price_product_output

pytestmark = pytest.mark.django_db
TARGET = date(2026, 9, 11)
NOW = datetime(2026, 9, 12, 1, tzinfo=UTC)


@pytest.fixture
def product_environment(tmp_path, monkeypatch, django_user_model, request):
    return make_product_environment(
        tmp_path,
        monkeypatch,
        django_user_model,
        omit_benchmark=getattr(request, "param", {}).get("omit_benchmark", False),
    )


def make_product_environment(tmp_path, monkeypatch, django_user_model, *, omit_benchmark=False):
    monkeypatch.setattr(timezone, "now", lambda: NOW)
    owner = django_user_model.objects.create_user(
        username="research-fixture-owner", date_joined=NOW
    )
    ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=True,
        terms_url="https://twelvedata.com/terms",
        usage_scope=PRIVATE_USAGE_SCOPE,
        metadata={
            "internal_display_rights_confirmed": True,
            "plan": "grow",
            "daily_credit_limit": 800,
            "credits_per_minute": 8,
        },
    )
    path = default_us_universe_config_path()
    config = load_us_universe_config(path)
    store = AssetStore(tmp_path / "assets")
    rows = [
        {
            "symbol": symbol,
            "name": f"Synthetic {symbol}",
            "currency": "USD",
            "exchange": "NASDAQ",
            "mic_code": "XNAS",
            "country": "United States",
            "type": "Common Stock" if symbol in {"AAPL", "MSFT", "CHEAP", "NEW"} else "ETF",
        }
        for symbol in (*config.symbols, "CHEAP", "NEW")
    ]
    raw_catalog = json.dumps({"status": "ok", "count": len(rows), "data": rows}).encode()
    references, count = twelve_data.parse_stock_catalog_references(
        raw_catalog, exchange="NASDAQ", require_complete=True
    )
    _persist_catalog(
        store,
        StockCatalog(
            provider="twelve_data",
            exchange="NASDAQ",
            references=references,
            count=count,
            retrieved_at=NOW,
            source_url="https://api.twelvedata.com/stocks?exchange=NASDAQ",
            raw_bytes=raw_catalog,
        ),
    )
    nyse_rows = [
        {
            "symbol": "VENUE",
            "name": "Synthetic venue",
            "currency": "USD",
            "exchange": "NYSE",
            "mic_code": "XNYS",
            "country": "United States",
            "type": "Common Stock",
        }
    ]
    nyse_raw = json.dumps({"status": "ok", "count": 1, "data": nyse_rows}).encode()
    nyse_references, nyse_count = twelve_data.parse_stock_catalog_references(
        nyse_raw, exchange="NYSE", require_complete=True
    )
    _persist_catalog(
        store,
        StockCatalog(
            provider="twelve_data",
            exchange="NYSE",
            references=nyse_references,
            count=nyse_count,
            retrieved_at=NOW,
            source_url="https://api.twelvedata.com/stocks?exchange=NYSE",
            raw_bytes=nyse_raw,
        ),
    )
    for symbol in ("AAPL", "MSFT", "SPY"):
        if symbol == "SPY" and omit_benchmark:
            continue
        _persist_price_series(store=store, series=_series(symbol), listing=None)
    TrackedSymbol.objects.create(owner=owner, symbol="CHEAP")
    resolve = Mock(side_effect=AssertionError("Unexpected credential resolution"))
    fetch = Mock(side_effect=AssertionError("Unexpected provider request"))
    monkeypatch.setattr(twelve_data, "resolve_api_key", resolve)
    monkeypatch.setattr(twelve_data, "fetch_daily_price_series", fetch)
    monkeypatch.setattr(
        twelve_data,
        "fetch_stock_catalog",
        Mock(side_effect=AssertionError("Unexpected catalog request")),
    )
    return owner, store, path, resolve, fetch


def _run(environment, **kwargs):
    owner, store, path, _resolve, _fetch = environment
    return execute_daily_research_job(
        target_date=TARGET,
        owner=owner,
        store=store,
        core_config_path=path,
        enforce_rate_limit=False,
        **kwargs,
    )


def test_cached_core_and_saved_reach_real_five_row_writer_without_credentials(product_environment):
    _owner, store, _path, resolve, fetch = product_environment
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)

    job = _run(product_environment)

    assert job.status == "success"
    run = AnalysisRun.objects.get(id=job.details["analysis_run_id"])
    assert run.generated_at == NOW
    assert run.data_cutoff.date() == TARGET
    assert Prediction.objects.filter(analysis__run=run).count() == 15
    assert set(
        Prediction.objects.filter(analysis__run=run).values_list(
            "listing__provider_symbol", flat=True
        )
    ) == {"AAPL", "MSFT", "CHEAP"}
    cheap = Prediction.objects.filter(listing__provider_symbol="CHEAP")
    assert cheap.filter(evidence_role="advisory", base_return__isnull=False).count() == 4
    assert not cheap.filter(recommendation="BUY").exists()
    assert all(row.calculation["method_version"] == row.method_version for row in cheap)
    membership = product_membership_payload(run.universe_snapshot, store=store)
    assert membership["admissions"]["NVDA"]["status"] == "identity_rejected"
    assert membership["admissions"]["CHEAP"]["status"] == "admitted"
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_missing_saved_history_is_bootstrapped_once_then_analyzed(product_environment):
    _owner, store, _path, resolve, fetch = product_environment
    resolve.side_effect = None
    resolve.return_value = "synthetic-test-token"

    def respond(symbol, **kwargs):
        assert DataAsset.objects.filter(kind=PRODUCT_INTAKE_KIND).count() == 1
        assert symbol == "CHEAP"
        assert kwargs["end_date"] == TARGET
        assert kwargs["adjustment"] == "splits"
        assert (TARGET - kwargs["start_date"]).days == 366 * 7
        return _series(symbol)

    fetch.side_effect = respond
    job = _run(product_environment)

    assert job.status == "success"
    assert job.details["credits_used"] == 1
    assert Prediction.objects.filter(listing__provider_symbol="CHEAP").count() == 5
    resolve.assert_called_once()
    fetch.assert_called_once()
    assert DataAsset.objects.filter(kind="price_history", subject="SPY").count() == 1


@pytest.mark.parametrize(
    ("provider_failure", "expected_admission", "expected_predictions"),
    [
        pytest.param(False, "admitted", 15, id="success"),
        pytest.param(True, "provider_failed", 10, id="provider-failure"),
    ],
)
def test_daily_job_suppresses_http_client_request_logs_for_native_acquisition(
    product_environment,
    caplog,
    provider_failure,
    expected_admission,
    expected_predictions,
):
    _owner, store, _path, resolve, fetch = product_environment
    resolve.side_effect = None
    resolve.return_value = "synthetic-test-token"
    request_url = (
        "https://synthetic-provider.invalid/time_series?"
        "symbol=SYNTHETIC_PRIVATE_SYMBOL&interval=1day"
    )
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    original_levels = (httpx_logger.level, httpcore_logger.level)
    httpx_logger.setLevel(logging.INFO)
    httpcore_logger.setLevel(logging.DEBUG)

    def acquire(symbol, **_kwargs):
        httpx_logger.info('HTTP Request: GET %s "HTTP/1.1 200 OK"', request_url)
        httpcore_logger.debug("receive_response_headers.complete url=%s", request_url)
        if provider_failure:
            raise ProviderError("Synthetic provider failure")
        return _series(symbol)

    fetch.side_effect = acquire
    try:
        with caplog.at_level(logging.DEBUG):
            job = _run(product_environment)

        assert (httpx_logger.level, httpcore_logger.level) == (logging.INFO, logging.DEBUG)
    finally:
        httpx_logger.setLevel(original_levels[0])
        httpcore_logger.setLevel(original_levels[1])

    run = AnalysisRun.objects.get(id=job.details["analysis_run_id"])
    admissions = product_membership_payload(run.universe_snapshot, store=store)["admissions"]
    admission = admissions["CHEAP"]
    assert job.status == "success"
    assert admission["status"] == expected_admission
    assert Prediction.objects.count() == expected_predictions
    fetch.assert_called_once()
    assert request_url not in caplog.text
    assert "SYNTHETIC_PRIVATE_SYMBOL" not in caplog.text
    assert not [record for record in caplog.records if record.name in {"httpx", "httpcore"}]


def test_completed_retry_ignores_changed_grade_preferences_and_enablement(product_environment):
    owner, store, _path, resolve, fetch = product_environment
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    original = _run(product_environment)
    original_predictions = set(Prediction.objects.values_list("id", flat=True))
    original_assets = set(DataAsset.objects.values_list("id", flat=True))
    TrackedSymbol.objects.filter(owner=owner).delete()
    TrackedSymbol.objects.create(owner=owner, symbol="NEW")
    ProviderRecord.objects.filter(provider="twelve_data").update(enabled=False)

    retry = _run(product_environment, issued_on_time=True)

    assert retry.status == "skipped"
    assert retry.details["successful_run_id"] == str(original.id)
    assert set(Prediction.objects.values_list("id", flat=True)) == original_predictions
    assert set(DataAsset.objects.values_list("id", flat=True)) == original_assets
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_failed_writer_reuses_captured_membership_and_acquired_history(
    product_environment, monkeypatch
):
    import stanstock.data.research_product_jobs as jobs

    _owner, _store, _path, resolve, fetch = product_environment
    resolve.side_effect = None
    resolve.return_value = "synthetic-test-token"
    fetch.side_effect = lambda symbol, **_kwargs: _series(symbol)
    writer = jobs.analyze_snapshot
    monkeypatch.setattr(
        jobs, "analyze_snapshot", Mock(side_effect=RuntimeError("synthetic write failure"))
    )
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        _run(product_environment)
    sources = set(DataAsset.objects.filter(provider="twelve_data").values_list("id", flat=True))
    assert not Prediction.objects.exists()
    monkeypatch.setattr(jobs, "analyze_snapshot", writer)
    ProviderRecord.objects.filter(provider="twelve_data").update(enabled=False)
    resolve.side_effect = AssertionError("Retry resolved credentials")
    fetch.side_effect = AssertionError("Retry repeated a fetch")

    retry = _run(product_environment)

    assert retry.status == "success"
    assert Prediction.objects.count() == 15
    assert (
        set(DataAsset.objects.filter(provider="twelve_data").values_list("id", flat=True))
        == sources
    )
    assert fetch.call_count == 1


def test_explicit_new_issuance_captures_changed_saved_set(product_environment):
    owner, store, _path, resolve, fetch = product_environment
    for symbol in ("CHEAP", "NEW"):
        _persist_price_series(store=store, series=_series(symbol), listing=None)
    original = _run(product_environment)
    TrackedSymbol.objects.filter(owner=owner).delete()
    TrackedSymbol.objects.create(owner=owner, symbol="NEW")

    reissue = _run(product_environment, issuance_key="explicit-reissue")

    assert reissue.status == "success"
    assert reissue.details["analysis_run_id"] != original.details["analysis_run_id"]
    assert reissue.details["snapshot_id"] != original.details["snapshot_id"]
    assert set(
        Prediction.objects.filter(analysis__run_id=reissue.details["analysis_run_id"]).values_list(
            "listing__provider_symbol", flat=True
        )
    ) == {"AAPL", "MSFT", "NEW"}
    assert Prediction.objects.count() == 30
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_rejected_core_does_not_block_new_saved_listing_materialization(product_environment):
    _owner, store, _path, _resolve, _fetch = product_environment
    assert not Listing.objects.filter(provider_symbol="CHEAP").exists()
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)

    job = _run(product_environment)

    assert job.status == "success"
    listing = Listing.objects.get(provider_symbol="CHEAP")
    assert listing.security.security_type == "common_stock"
    assert Prediction.objects.filter(listing=listing).count() == 5
    assert not Prediction.objects.filter(listing__provider_symbol="NVDA").exists()


def test_short_bootstrap_reports_exact_shortfall_without_blocking_core(product_environment):
    _owner, store, _path, resolve, fetch = product_environment
    resolve.side_effect = None
    resolve.return_value = "synthetic-test-token"
    fetch.side_effect = lambda symbol, **_kwargs: _series(symbol, closes=100)

    job = _run(product_environment)

    assert job.status == "success"
    run = AnalysisRun.objects.get(id=job.details["analysis_run_id"])
    admission = product_membership_payload(run.universe_snapshot, store=store)["admissions"][
        "CHEAP"
    ]
    assert admission["status"] == "insufficient_history"
    assert admission["bootstrap_attempted"] is True
    assert admission["history_qualification"] == {
        "required_closes": 757,
        "available_required_closes": 100,
        "missing_required_closes": 657,
    }
    assert Prediction.objects.count() == 10
    assert not Prediction.objects.filter(listing__provider_symbol="CHEAP").exists()
    fetch.assert_called_once()


def test_raw_normalized_contradiction_cannot_fall_back_or_trigger_a_fetch(
    product_environment, monkeypatch
):
    _owner, store, _path, resolve, fetch = product_environment
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    original = DataAsset.objects.get(kind="price_history", subject="MSFT")
    frame = store.read_frame(original.relative_path)
    corrupt = frame.with_columns((pl.col("close") * 2).alias("close"))
    register_asset(
        provider="twelve_data",
        kind="price_history",
        subject="MSFT",
        stored=store.write_frame("corrupt-but-checksummed.parquet", corrupt),
        retrieved_at=NOW + timedelta(seconds=1),
        available_at=NOW + timedelta(seconds=1),
        period_start=original.period_start,
        period_end=original.period_end,
        metadata=original.metadata,
    )
    monkeypatch.setattr(timezone, "now", lambda: NOW + timedelta(seconds=2))

    job = _run(product_environment)

    run = AnalysisRun.objects.get(id=job.details["analysis_run_id"])
    admission = product_membership_payload(run.universe_snapshot, store=store)["admissions"]["MSFT"]
    assert admission["status"] == "evidence_invalid"
    assert Prediction.objects.count() == 10
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_consistent_two_bar_asset_does_not_mask_qualified_history(product_environment, monkeypatch):
    _owner, store, _path, resolve, fetch = product_environment
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    original = DataAsset.objects.get(kind="price_history", subject="AAPL")
    complete = _series("AAPL")
    payload = json.loads(complete.raw_bytes)
    payload["values"] = payload["values"][-2:]
    partial = replace(
        complete,
        bars=complete.bars[-2:],
        raw_bytes=json.dumps(payload).encode(),
        retrieved_at=NOW + timedelta(seconds=1),
    )
    _persist_price_series(store=store, series=partial, listing=None)
    monkeypatch.setattr(timezone, "now", lambda: NOW + timedelta(seconds=2))

    job = _run(product_environment)

    run = AnalysisRun.objects.get(id=job.details["analysis_run_id"])
    admission = product_membership_payload(run.universe_snapshot, store=store)["admissions"]["AAPL"]
    assert admission["price_asset"]["id"] == str(original.id)
    assert admission["bootstrap_attempted"] is False
    assert Prediction.objects.count() == 15
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_wrong_price_mic_rejects_only_the_affected_candidate(product_environment, monkeypatch):
    _owner, store, _path, resolve, fetch = product_environment
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    _persist_price_series(
        store=store,
        series=_series("MSFT", mic="XNYS", retrieved_at=NOW + timedelta(seconds=1)),
        listing=None,
    )
    monkeypatch.setattr(timezone, "now", lambda: NOW + timedelta(seconds=2))

    job = _run(product_environment)

    run = AnalysisRun.objects.get(id=job.details["analysis_run_id"])
    admission = product_membership_payload(run.universe_snapshot, store=store)["admissions"]["MSFT"]
    assert admission["status"] == "identity_rejected"
    assert admission["reason"] == "price_catalog_identity_conflict"
    assert Prediction.objects.count() == 10
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_completed_retry_rejects_changed_bytes_without_credentials(product_environment):
    _owner, store, _path, resolve, fetch = product_environment
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    _run(product_environment)
    asset = DataAsset.objects.get(kind="price_history", subject="CHEAP")
    store.resolve(asset.relative_path).write_bytes(b"synthetic corruption")

    with pytest.raises(ValueError, match="checksum|corrupt"):
        _run(product_environment)

    assert Prediction.objects.count() == 15
    resolve.assert_not_called()
    fetch.assert_not_called()


@pytest.mark.parametrize("late", [False, True])
def test_unsafe_observed_request_fails_before_intake_and_credentials(
    product_environment, monkeypatch, late
):
    import stanstock.research.product_pipeline as pipeline

    _owner, _store, _path, resolve, fetch = product_environment
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", "a" * 40)
    monkeypatch.setattr(pipeline, "clean_git_revision", lambda _path: ("a" if late else "b") * 40)
    if late:
        monkeypatch.setattr(timezone, "now", lambda: datetime(2026, 9, 14, 14, tzinfo=UTC))

    with pytest.raises(ValueError, match="deadline|revision"):
        _run(product_environment, issued_on_time=True)

    assert not DataAsset.objects.filter(kind=PRODUCT_INTAKE_KIND).exists()
    assert not AnalysisRun.objects.exists()
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_final_replay_deadline_crossing_rolls_back_every_output(product_environment, monkeypatch):
    import stanstock.research.product_pipeline as pipeline

    _owner, store, _path, resolve, fetch = product_environment
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", "a" * 40)
    monkeypatch.setattr(pipeline, "clean_git_revision", lambda _path: "a" * 40)
    verifier = pipeline.verify_price_product_output

    def cross_deadline(**kwargs):
        verifier(**kwargs)
        if kwargs.get("replay"):
            monkeypatch.setattr(timezone, "now", lambda: datetime(2026, 9, 14, 14, tzinfo=UTC))

    monkeypatch.setattr(pipeline, "verify_price_product_output", cross_deadline)

    with pytest.raises(ValueError, match="deadline"):
        _run(product_environment, issued_on_time=True)

    assert not AnalysisRun.objects.exists()
    assert not Prediction.objects.exists()
    assert not DataAsset.objects.filter(kind="research_product_calculation").exists()
    assert not list(store.root.glob("research/product/*/*/calculation.json"))
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_self_consistent_wrong_rows_and_manifest_do_not_override_calculation(
    product_environment, monkeypatch
):
    _owner, store, _path, _resolve, _fetch = product_environment
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    create_analysis = StockAnalysis.objects.create
    create_prediction = Prediction.objects.create

    def altered_analysis(**kwargs):
        kwargs["risk_class"] = "very_high"
        return create_analysis(**kwargs)

    def altered_prediction(**kwargs):
        kwargs["calculation"]["risk"]["relative_volatility_label"] = "very_high"
        return create_prediction(**kwargs)

    monkeypatch.setattr(StockAnalysis.objects, "create", altered_analysis)
    monkeypatch.setattr(Prediction.objects, "create", altered_prediction)
    with pytest.raises(ValueError, match="registered|recorded calculation"):
        _run(product_environment)
    assert not Prediction.objects.exists()
    assert not AnalysisRun.objects.exists()


def test_registered_membership_detects_a_removed_whole_listing(product_environment):
    _owner, store, _path, _resolve, _fetch = product_environment
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    job = _run(product_environment)
    run = AnalysisRun.objects.get(id=job.details["analysis_run_id"])
    run.universe_snapshot.memberships.filter(listing__provider_symbol="CHEAP").delete()

    with pytest.raises(ValueError, match="membership"):
        verify_price_product_output(run=run, store=store)


def test_first_fully_cached_intake_works_with_fetching_disabled(product_environment):
    _owner, store, _path, resolve, fetch = product_environment
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    ProviderRecord.objects.filter(provider="twelve_data").update(enabled=False)

    job = _run(product_environment)

    assert job.status == "success"
    assert Prediction.objects.count() == 15
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_committed_output_recovers_after_lost_job_completion(product_environment, monkeypatch):
    import stanstock.data.research_product_jobs as jobs

    owner, store, _path, resolve, fetch = product_environment
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    writer = jobs.analyze_snapshot

    def write_then_fail(**kwargs):
        writer(**kwargs)
        raise RuntimeError("Synthetic failure after the committed writer")

    monkeypatch.setattr(jobs, "analyze_snapshot", write_then_fail)
    with pytest.raises(RuntimeError, match="after the committed writer"):
        _run(product_environment)
    original = AnalysisRun.objects.get()
    assets = set(DataAsset.objects.values_list("id", flat=True))
    predictions = set(Prediction.objects.values_list("id", flat=True))
    TrackedSymbol.objects.filter(owner=owner).delete()
    ProviderRecord.objects.filter(provider="twelve_data").update(enabled=False)
    monkeypatch.setattr(jobs, "analyze_snapshot", writer)

    recovered = _run(product_environment, issued_on_time=True)

    assert recovered.status == "success"
    assert recovered.details["recovered"] is True
    assert recovered.details["analysis_run_id"] == str(original.id)
    assert set(DataAsset.objects.values_list("id", flat=True)) == assets
    assert set(Prediction.objects.values_list("id", flat=True)) == predictions
    assert not Prediction.objects.filter(issued_on_time=True).exists()
    resolve.assert_not_called()
    fetch.assert_not_called()


@pytest.mark.parametrize("product_environment", [{"omit_benchmark": True}], indirect=True)
def test_partial_acquisition_reuses_saved_history_before_retrying_benchmark(product_environment):
    owner, store, _path, resolve, fetch = product_environment
    resolve.side_effect = None
    resolve.return_value = "synthetic-test-token"
    requested = []

    def respond(symbol, **_kwargs):
        requested.append(symbol)
        if symbol == "SPY" and requested.count("SPY") == 1:
            raise ProviderError("Synthetic transport failure")
        return _series(symbol)

    fetch.side_effect = respond
    with pytest.raises(ValueError, match="Benchmark history acquisition failed"):
        _run(product_environment)
    saved_asset = DataAsset.objects.get(kind="price_history", subject="CHEAP")
    TrackedSymbol.objects.filter(owner=owner).delete()
    TrackedSymbol.objects.create(owner=owner, symbol="NEW")

    job = _run(product_environment)

    assert job.status == "success"
    assert requested == ["CHEAP", "SPY", "SPY"]
    assert DataAsset.objects.get(kind="price_history", subject="CHEAP").id == saved_asset.id
    assert Prediction.objects.filter(listing__provider_symbol="CHEAP").count() == 5
    assert not Prediction.objects.filter(listing__provider_symbol="NEW").exists()
    assert resolve.call_count == 2
    assert (
        len(
            product_membership_payload(AnalysisRun.objects.get().universe_snapshot, store=store)[
                "qualified_listing_ids"
            ]
        )
        == 3
    )


def test_retry_does_not_broaden_the_captured_entitlement(product_environment, monkeypatch):
    _owner, store, _path, resolve, fetch = product_environment
    record = ProviderRecord.objects.get(provider="twelve_data")
    record.usage_scope = BASIC_USAGE_SCOPE
    record.metadata.update(
        plan="basic", licensed_user_id=str(_owner.pk), personal_noncommercial_confirmed=True
    )
    record.save(update_fields=["usage_scope", "metadata"])
    catalog = DataAsset.objects.get(kind="stock_catalog", subject="NASDAQ")
    payload = json.loads(store.read_bytes(catalog.relative_path))
    for row in payload["data"]:
        if row["symbol"] == "AAPL":
            row["access"] = {"plan": "Grow"}
    raw = json.dumps(payload).encode()
    references, count = twelve_data.parse_stock_catalog_references(
        raw, exchange="NASDAQ", require_complete=True
    )
    _persist_catalog(
        store,
        StockCatalog(
            provider="twelve_data",
            exchange="NASDAQ",
            references=references,
            count=count,
            retrieved_at=NOW + timedelta(seconds=1),
            source_url="https://api.twelvedata.com/stocks?exchange=NASDAQ",
            raw_bytes=raw,
        ),
    )
    monkeypatch.setattr(timezone, "now", lambda: NOW + timedelta(seconds=2))
    resolve.side_effect = ProviderConfigurationError("Synthetic unavailable credential")
    with pytest.raises(ProviderConfigurationError):
        _run(product_environment)
    record.usage_scope = PRIVATE_USAGE_SCOPE
    record.metadata["plan"] = "grow"
    record.save(update_fields=["usage_scope", "metadata"])
    _persist_price_series(
        store=store, series=_series("CHEAP", retrieved_at=NOW + timedelta(seconds=3)), listing=None
    )
    monkeypatch.setattr(timezone, "now", lambda: NOW + timedelta(seconds=4))
    resolve.reset_mock()
    resolve.side_effect = AssertionError("Retry should use cached evidence")

    job = _run(product_environment)

    run = AnalysisRun.objects.get(id=job.details["analysis_run_id"])
    admission = product_membership_payload(run.universe_snapshot, store=store)["admissions"]["AAPL"]
    assert admission["status"] == "entitlement_rejected"
    assert Prediction.objects.count() == 10
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_live_intake_rejects_an_arbitrary_core_path(product_environment, tmp_path):
    owner, store, _path, resolve, fetch = product_environment
    shadow = tmp_path / "shadow-core.yaml"
    shadow.write_bytes(default_us_universe_config_path().read_bytes())

    with pytest.raises(ValueError, match="reviewed curated core"):
        execute_daily_research_job(
            target_date=TARGET, owner=owner, store=store, core_config_path=shadow
        )

    assert not DataAsset.objects.filter(kind=PRODUCT_INTAKE_KIND).exists()
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_distinct_concurrent_issuances_share_the_target_acquisition_lock(tmp_path):
    root = Path(__file__).resolve().parents[1]
    probe = subprocess.run(
        [sys.executable, str(root / "tests/research_product_concurrency_probe.py"), str(tmp_path)],
        cwd=root,
        env={
            "DJANGO_SETTINGS_MODULE": "stanstock.settings.test",
            "PYTHONPATH": str(root / "src") + ":" + str(root / "tests"),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert json.loads(probe.stdout) == {
        "analyses": 2,
        "predictions": 30,
        "listings": 3,
        "history_requests": 1,
    }


def _series(
    symbol: str, *, closes: int = 757, mic: str | None = None, retrieved_at: datetime = NOW
):
    calendar = get_calendar("XNYS")
    sessions = calendar.sessions_window(calendar.date_to_session(TARGET), -closes)
    base = 5.0 if symbol == "CHEAP" else 80.0 if symbol == "SPY" else 20.0
    values = []
    for index, session in enumerate(sessions):
        value = base * math.exp(0.0002 * index + 0.007 * math.sin(index / 11))
        values.append(
            {
                "datetime": session.date().isoformat(),
                "open": str(value),
                "high": str(value),
                "low": str(value),
                "close": str(value),
                "volume": "1000000",
            }
        )
    payload = {
        "status": "ok",
        "meta": {
            "symbol": symbol,
            "interval": "1day",
            "currency": "USD",
            "exchange": "NYSE ARCA" if symbol == "SPY" else "NASDAQ",
            "mic_code": mic or ("ARCX" if symbol == "SPY" else "XNAS"),
            "type": "ETF" if symbol == "SPY" else "Common Stock",
        },
        "values": values,
    }
    return twelve_data.parse_daily_price_series(
        json.dumps(payload).encode(),
        symbol=symbol,
        retrieved_at=retrieved_at,
        source_url="https://api.twelvedata.com/time_series",
        end_date=TARGET,
    )

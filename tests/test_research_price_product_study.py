from __future__ import annotations

import json
from datetime import UTC, datetime
from io import StringIO
from unittest.mock import Mock

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import DataAsset
from stanstock.data.research_product_demo import execute_demo_product_refresh
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.price_product_study import (
    serialize_price_product_study,
    study_price_product_run,
)

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def active_synthetic_product(settings, tmp_path, monkeypatch):
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = True
    settings.DATA_DIR = tmp_path
    blocked = Mock(
        side_effect=AssertionError(
            "Retrospective replay must not resolve credentials or use the network"
        )
    )
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("stanstock.data.providers.twelve_data.resolve_api_key", blocked)
    yield AssetStore(tmp_path)
    blocked.assert_not_called()


@pytest.fixture
def source_run(active_synthetic_product):
    job = execute_demo_product_refresh(store=active_synthetic_product)
    run = AnalysisRun.objects.select_related("universe_snapshot").get(
        pk=job.details["analysis_run_id"]
    )
    listing = StockAnalysis.objects.get(run=run, listing__ticker="ZZRPUP").listing
    return run, listing, active_synthetic_product


def test_study_loader_is_read_only_and_clips_history_to_each_anchor(source_run) -> None:
    run, listing, store = source_run
    counts_before = {
        "runs": AnalysisRun.objects.count(),
        "analyses": StockAnalysis.objects.count(),
        "predictions": Prediction.objects.count(),
        "assets": DataAsset.objects.count(),
    }

    report = study_price_product_run(
        run=run,
        store=store,
        listing_ids=(listing.id,),
        report_generated_at=datetime(2026, 9, 13, 18, tzinfo=UTC),
    )
    document = serialize_price_product_study(report, include_generated_at=False)

    assert counts_before == {
        "runs": AnalysisRun.objects.count(),
        "analyses": StockAnalysis.objects.count(),
        "predictions": Prediction.objects.count(),
        "assets": DataAsset.objects.count(),
    }
    listing_report = document["listings"][0]
    historical_anchor = next(
        anchor
        for anchor in listing_report["anchors"]
        if (
            anchor["horizon"] == "6m"
            and anchor["anchor_date"] != document["source_run"]["target_date"]
            and anchor["feature_window"]["stock_available_sessions"] == 757
            and anchor["feature_window"]["benchmark_available_sessions"] == 757
        )
    )
    anchor_date = datetime.fromisoformat(historical_anchor["anchor_date"]).date()
    expected_start = (
        get_calendar("XNYS")
        .sessions_window(get_calendar("XNYS").date_to_session(anchor_date, direction="none"), -757)[
            0
        ]
        .date()
    )

    assert listing_report["full_history"]["stock_end"] == document["source_run"]["target_date"]
    assert listing_report["source_window"]["window_end"] == document["source_run"]["target_date"]
    assert historical_anchor["feature_window"] == {
        "window_start": expected_start.isoformat(),
        "window_end": anchor_date.isoformat(),
        "window_session_count": 757,
        "stock_available_sessions": 757,
        "benchmark_available_sessions": 757,
    }
    assert (
        historical_anchor["feature_window"]["window_end"]
        != listing_report["source_window"]["window_end"]
    )
    assert historical_anchor["actual_returns"]["unavailable_reason"] is None


def test_study_report_keeps_partition_scope_and_empty_long_horizons_explicit(source_run) -> None:
    run, listing, store = source_run
    document = serialize_price_product_study(
        study_price_product_run(
            run=run,
            store=store,
            listing_ids=(listing.id,),
            report_generated_at=datetime(2026, 9, 13, 18, 1, tzinfo=UTC),
        ),
        include_generated_at=False,
    )
    aggregates = {
        (item["partition"], item["horizon"], item["model_name"]): item
        for item in document["projection_aggregates"]
    }

    assert aggregates[("validation", "3y", "us-price-fhs-v1")]["insufficient_reason"] == (
        "no_eligible_observations"
    )
    assert aggregates[("final_holdout", "5y", "us-price-fhs-v1")]["insufficient_reason"] == (
        "no_eligible_observations"
    )

    validation_pairs = [
        item
        for item in document["paired_model_comparisons"]
        if item["partition"] == "validation"
        and item["horizon"] == "6m"
        and item["metric_name"] == "median_absolute_error"
    ]
    assert {item["baseline_model"] for item in validation_pairs} == {
        "historical_log_drift_gaussian",
        "zero_log_drift_gaussian",
    }
    assert all(item["candidate_model"] == "us-price-fhs-v1" for item in validation_pairs)

    empty_pair = next(
        item
        for item in document["paired_model_comparisons"]
        if item["partition"] == "validation"
        and item["horizon"] == "3y"
        and item["baseline_model"] == "zero_log_drift_gaussian"
        and item["metric_name"] == "median_absolute_error"
    )
    assert empty_pair["paired_observation_count"] == 0
    assert empty_pair["unavailable_reason"] == "no_aligned_candidate_baseline_observations"


def test_serialized_report_can_exclude_actual_generation_timestamp_deterministically(
    source_run,
) -> None:
    run, listing, store = source_run
    first = study_price_product_run(
        run=run,
        store=store,
        listing_ids=(listing.id,),
        report_generated_at=datetime(2026, 9, 13, 19, tzinfo=UTC),
    )
    second = study_price_product_run(
        run=run,
        store=store,
        listing_ids=(listing.id,),
        report_generated_at=datetime(2026, 9, 13, 20, tzinfo=UTC),
    )

    assert serialize_price_product_study(
        first,
        include_generated_at=False,
    ) == serialize_price_product_study(
        second,
        include_generated_at=False,
    )
    assert (
        serialize_price_product_study(
            first,
            include_generated_at=True,
        )["report_generated_at"]
        != serialize_price_product_study(
            second,
            include_generated_at=True,
        )["report_generated_at"]
    )


def test_replay_command_requires_explicit_scope_and_emits_json_without_writes(source_run) -> None:
    run, listing, _store = source_run
    with pytest.raises(
        CommandError,
        match="Provide exactly one of --all-selected or --listing-ids",
    ):
        call_command("replay_price_product", "--run", str(run.id), "--format", "json")

    counts_before = {
        "runs": AnalysisRun.objects.count(),
        "analyses": StockAnalysis.objects.count(),
        "predictions": Prediction.objects.count(),
        "assets": DataAsset.objects.count(),
    }
    output = StringIO()
    call_command(
        "replay_price_product",
        "--run",
        str(run.id),
        "--listing-ids",
        str(listing.id),
        "--format",
        "json",
        stdout=output,
    )
    document = json.loads(output.getvalue())

    assert document["schema"] == "research-product-retrospective-study@1"
    assert document["source_run"]["id"] == str(run.id)
    assert document["scope"]["studied_listing_count"] == 1
    assert "report_generated_at" in document
    assert counts_before == {
        "runs": AnalysisRun.objects.count(),
        "analyses": StockAnalysis.objects.count(),
        "predictions": Prediction.objects.count(),
        "assets": DataAsset.objects.count(),
    }


@pytest.mark.parametrize("mode", ["corrupt_source_bytes", "ambiguous_calculation_source"])
def test_study_loader_refuses_corrupt_or_ambiguous_registered_sources(
    source_run, mode: str
) -> None:
    run, listing, store = source_run
    if mode == "corrupt_source_bytes":
        source_asset = DataAsset.objects.get(
            provider="synthetic_demo",
            kind="price_history",
            subject="ZZRPUP",
        )
        store.resolve(source_asset.relative_path).write_bytes(b"corrupt synthetic bytes")
    else:
        register_asset(
            provider="stanstock",
            kind="research_product_calculation",
            subject=str(run.id),
            stored=store.write_bytes("research/forged/duplicate-calculation.json", b"{}"),
            retrieved_at=run.generated_at,
            available_at=run.generated_at,
            metadata={
                "contract": "research-product-calculation@1",
                "listing_id": str(listing.id),
            },
        )

    with pytest.raises(
        (RefreshVerificationError, ValueError),
        match="checksum|corrupt|ambiguous|artifact|source",
    ):
        study_price_product_run(
            run=run,
            store=store,
            listing_ids=(listing.id,),
            report_generated_at=datetime(2026, 9, 13, 18, tzinfo=UTC),
        )

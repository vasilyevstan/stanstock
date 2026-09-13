from datetime import date
from io import StringIO
from unittest.mock import Mock

import pytest
from django.core.management import call_command
from django.utils import timezone

from stanstock.core.models import JobRun
from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset, LatestMarketData
from stanstock.data.research_product import product_membership_payload
from stanstock.data.research_product_demo import END_DATE, execute_demo_product_refresh
from stanstock.portfolio.models import TrackedSymbol
from stanstock.research.models import AnalysisRun, Prediction, Recommendation, StockAnalysis
from stanstock.research.product_pipeline import verify_price_product_output

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def active_synthetic_product(settings, tmp_path, monkeypatch):
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = True
    settings.DATA_DIR = tmp_path
    blocked = Mock(
        side_effect=AssertionError("Demo must not resolve credentials or use the network")
    )
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("stanstock.data.providers.twelve_data.resolve_api_key", blocked)
    yield
    blocked.assert_not_called()


def test_demo_command_uses_real_models_and_records_a_precise_shortfall(tmp_path):
    now = timezone.now()
    JobRun.objects.create(
        job_name="refresh_demo",
        region="all",
        target_date=END_DATE,
        status="success",
        started_at=now,
        finished_at=now,
    )
    output = StringIO()
    call_command("refresh_demo", stdout=output)

    run = AnalysisRun.objects.get()
    assert run.config_version == "research-product-v1"
    assert run.target_date == END_DATE
    assert run.issued_on_time is False
    assert run.universe_snapshot.grade == "research"
    assert run.generated_at > run.data_cutoff
    assert StockAnalysis.objects.count() == 3
    assert Prediction.objects.count() == 15
    assert TrackedSymbol.objects.count() == 0
    assert LatestMarketData.objects.count() == 3
    assert set(DataAsset.objects.values_list("provider", flat=True)) == {
        "synthetic_demo",
        "stanstock",
    }
    suggestions = dict(StockAnalysis.objects.values_list("listing__ticker", "recommendation"))
    assert suggestions == {
        "ZZRPUP": Recommendation.BUY,
        "ZZRPDOWN": Recommendation.AVOID,
        "ZZRPLOW": Recommendation.HOLD,
    }
    for row in Prediction.objects.all():
        assert row.probability_positive is None
        assert row.confidence is None
        assert row.overall_score is None
        assert row.issued_on_time is False
        if row.evidence_role == "advisory":
            assert row.bear_return <= row.base_return <= row.bull_return
        else:
            assert row.horizon == "6m"
            assert row.bear_return is row.base_return is row.bull_return is None
    membership = product_membership_payload(run.universe_snapshot, store=AssetStore(tmp_path))
    shortfall = membership["admissions"]["ZZRPNEW"]
    assert shortfall["status"] == "insufficient_history"
    assert shortfall["history_qualification"] == {
        "required_closes": 757,
        "available_closes": 100,
        "missing_closes": 657,
    }
    assert "synthetic_demo research-product-v1" in output.getvalue()
    assert "evidence_grade=research" in output.getvalue()
    verify_price_product_output(run=run, store=AssetStore(tmp_path))


def test_demo_completed_retry_reuses_all_evidence_without_reseeding(monkeypatch):
    first = execute_demo_product_refresh()
    identities = set(DataAsset.objects.values_list("id", flat=True))
    predictions = set(Prediction.objects.values_list("id", flat=True))
    monkeypatch.setattr(
        "stanstock.data.research_product_demo._seed_prices",
        Mock(side_effect=AssertionError("Completed demo must not regenerate its source")),
    )
    second = execute_demo_product_refresh()
    assert first.status == "success"
    assert second.status == "skipped"
    assert second.details["successful_run_id"] == str(first.pk)
    assert set(DataAsset.objects.values_list("id", flat=True)) == identities
    assert set(Prediction.objects.values_list("id", flat=True)) == predictions


def test_demo_failed_writer_recovers_captured_membership(monkeypatch):
    from stanstock.data import research_product_demo as demo

    writer = demo.analyze_snapshot
    monkeypatch.setattr(
        demo, "analyze_snapshot", Mock(side_effect=RuntimeError("writer interrupted"))
    )
    with pytest.raises(RuntimeError, match="writer interrupted"):
        execute_demo_product_refresh()
    assert Prediction.objects.count() == 0
    source_ids = set(DataAsset.objects.values_list("id", flat=True))
    monkeypatch.setattr(demo, "analyze_snapshot", writer)
    monkeypatch.setattr(
        demo, "_seed_prices", Mock(side_effect=AssertionError("Captured source must be reused"))
    )
    job = execute_demo_product_refresh()
    assert job.status == "success"
    assert Prediction.objects.count() == 15
    assert source_ids < set(DataAsset.objects.values_list("id", flat=True))


def test_completed_demo_does_not_trust_success_status_after_source_corruption(tmp_path):
    execute_demo_product_refresh()
    source = DataAsset.objects.get(
        provider="synthetic_demo", kind="price_history", subject="ZZRPLOW"
    )
    AssetStore(tmp_path).resolve(source.relative_path).write_bytes(b"corrupt synthetic bytes")
    with pytest.raises(ValueError, match="checksum|corrupt|integrity"):
        execute_demo_product_refresh()
    assert Prediction.objects.count() == 15


@pytest.mark.parametrize("target", [date(2026, 9, 12), date(2027, 9, 10)])
def test_demo_refuses_unproduced_target_before_writes(target):
    with pytest.raises(ValueError, match="generated XNYS session"):
        execute_demo_product_refresh(target_date=target)
    assert DataAsset.objects.count() == 0
    assert JobRun.objects.count() == 0


@pytest.mark.parametrize("setting", ["DEMO_MODE", "RESEARCH_PRODUCT_ENABLED"])
def test_demo_requires_explicit_synthetic_profile(settings, setting):
    setattr(settings, setting, False)
    with pytest.raises(ValueError, match="enabled demo product profile"):
        execute_demo_product_refresh()
    assert DataAsset.objects.count() == 0

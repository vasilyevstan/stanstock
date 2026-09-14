from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from django.core.management import call_command
from django.db import OperationalError
from django.urls import reverse
from django.utils import timezone

from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset
from stanstock.data.research_product_demo import execute_demo_product_refresh
from stanstock.research import product_frequency_evidence as evidence
from stanstock.research.models import AnalysisRun
from stanstock.research.product_frequency_evidence import (
    FREQUENCY_EVIDENCE_KIND,
    read_registered_product_frequencies,
    register_product_frequencies,
    verify_registered_product_frequencies,
)
from stanstock.research.product_reader import read_research_product

pytestmark = pytest.mark.django_db


@pytest.fixture
def registered_demo(settings, tmp_path, django_user_model):
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = True
    settings.DATA_DIR = tmp_path
    execute_demo_product_refresh(store=AssetStore(tmp_path))
    return django_user_model.objects.create_user(username="frequency-viewer"), AssetStore(tmp_path)


def test_native_synthetic_run_registers_complete_four_horizon_evidence(registered_demo) -> None:
    user, store = registered_demo
    run = AnalysisRun.objects.get()

    result = read_registered_product_frequencies(
        user=user, run=run, decision_time=timezone.now(), store=store
    )

    assert result.status == "available"
    assert len(result.frequencies) == 12
    assert {item.frequency.horizon for item in result.frequencies} == {"6m", "12m", "3y", "5y"}
    assert all(item.frequency.counts is not None for item in result.frequencies)
    assert all(item.frequency.counts.total == 8192 for item in result.frequencies)
    assert all(
        item.frequency.counts.large_loss <= item.frequency.counts.loss
        for item in result.frequencies
    )
    assert all(item.ledger_returns is not None for item in result.frequencies)

    before = DataAsset.objects.count()
    asset = register_product_frequencies(run=run, store=store)
    assert DataAsset.objects.count() == before
    assert asset.kind == FREQUENCY_EVIDENCE_KIND
    verify_registered_product_frequencies(run=run, store=store)


def test_frequency_reader_hides_later_report_and_fails_closed_for_corrupt_bytes(
    registered_demo,
) -> None:
    user, store = registered_demo
    run = AnalysisRun.objects.get()
    asset = DataAsset.objects.get(kind=FREQUENCY_EVIDENCE_KIND)

    before_publication = asset.available_at - timedelta(microseconds=1)
    absent = read_registered_product_frequencies(
        user=user, run=run, decision_time=before_publication, store=store
    )
    assert absent.status == "absent"

    document = json.loads(store.read_bytes(asset.relative_path))
    assert document["source_run"]["id"] == str(run.id)
    store.resolve(asset.relative_path).write_bytes(b'{"forged":true}')
    failed = read_registered_product_frequencies(
        user=user, run=run, decision_time=timezone.now(), store=store
    )
    assert failed.status == "integrity_failed"


def test_manual_report_binds_one_winner_side_publication_time_and_rejects_bad_chronology(
    registered_demo, monkeypatch
) -> None:
    _user, store = registered_demo
    run = AnalysisRun.objects.get()
    asset = DataAsset.objects.get(kind=FREQUENCY_EVIDENCE_KIND)
    document = json.loads(store.read_bytes(asset.relative_path))
    derived_at = datetime.fromisoformat(document["derived_at"])

    # Manual registration never borrows the source run's on-time claim, and
    # the single publication instant binds document and DataAsset availability.
    assert document["derivation_source"] == "manual_backfill"
    assert document["own_deadline_met"] is False
    assert asset.retrieved_at == derived_at
    assert asset.available_at == derived_at
    assert derived_at >= run.generated_at

    with pytest.raises(ValueError, match="publication time"):
        evidence._require_publication_time(
            run=run, derived_at=run.generated_at.replace(tzinfo=None)
        )
    with pytest.raises(ValueError, match="publication time"):
        evidence._require_publication_time(
            run=run, derived_at=run.generated_at - timedelta(microseconds=1)
        )
    future = datetime(2099, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(evidence.timezone, "now", lambda: future - timedelta(microseconds=1))
    with pytest.raises(ValueError, match="publication time"):
        evidence._require_publication_time(run=run, derived_at=future)


@pytest.mark.parametrize(
    ("message", "expected"),
    (
        ("database is locked", True),
        ("could not obtain lock on row", True),
        ("connection unexpectedly closed", False),
    ),
)
def test_only_expected_database_contention_is_recoverable(message, expected) -> None:
    assert evidence._is_expected_registration_contention(OperationalError(message)) is expected


def test_native_frequency_chain_renders_registered_counts_and_preserves_ranges_on_failure(
    registered_demo, client
) -> None:
    user, store = registered_demo
    client.force_login(user)
    product = read_research_product(user=user, store=store)
    card = product.cards[0]
    projection = card.projections[0]
    assert projection.frequencies is not None

    detail = client.get(reverse("stock-detail", args=[card.analysis.listing_id]), {"horizon": "6m"})
    content = detail.content.decode()
    assert detail.status_code == 200
    assert "Model-estimated probabilities" in content
    assert "Shares of model simulations; not validated real-world odds." in content
    assert f"{projection.frequencies.counts.loss} of 8,192 paths" in content
    assert projection.median_return is not None
    assert str(projection.median_return * 100)[:3] in content

    asset = DataAsset.objects.get(kind=FREQUENCY_EVIDENCE_KIND)
    store.resolve(asset.relative_path).write_bytes(b"frequency evidence corrupted")
    failed_detail = client.get(reverse("stock-detail", args=[card.analysis.listing_id]))
    failed_content = failed_detail.content.decode()
    assert failed_detail.status_code == 200
    assert "Model-estimated probabilities unavailable." in failed_content
    assert projection.median_return is not None
    assert str(projection.median_return * 100)[:3] in failed_content


def test_frequency_command_verifies_one_explicit_run_without_writing(registered_demo) -> None:
    _user, _store = registered_demo
    run = AnalysisRun.objects.get()
    before = DataAsset.objects.count()

    call_command("derive_price_frequencies", "--run", str(run.id), "--verify")

    assert DataAsset.objects.count() == before


def test_offline_verify_preserves_original_execution_metadata_and_detects_shifted_counts(
    registered_demo, monkeypatch
) -> None:
    _user, store = registered_demo
    run = AnalysisRun.objects.get()
    evidence.verify_registered_product_frequencies(run=run, store=store)
    monkeypatch.setattr(evidence, "code_revision", lambda: "different-current-revision")
    evidence.verify_registered_product_frequencies(run=run, store=store)

    asset = DataAsset.objects.get(kind=FREQUENCY_EVIDENCE_KIND)
    document = json.loads(store.read_bytes(asset.relative_path))
    forged = deepcopy(document)
    counts = forged["rows"][0]["horizons"][0]["counts"]
    counts["loss"] += 1
    counts["flat_to_20"] -= 1
    monkeypatch.setattr(evidence, "_validate_registered_asset", lambda *args, **kwargs: forged)
    with pytest.raises(ValueError, match="frequency_verify_logical_content_mismatch"):
        evidence.verify_registered_product_frequencies(run=run, store=store)


@pytest.mark.parametrize("value", (0.5, "1", True))
def test_reader_rejects_noninteger_frequency_counts(value) -> None:
    with pytest.raises(ValueError, match="integers"):
        evidence._read_counts(
            {"loss": value, "flat_to_20": 1, "above_20": 8190, "large_loss": 0},
            path_count=8192,
        )

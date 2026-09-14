from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from django.core.management import call_command
from django.db import OperationalError, transaction
from django.urls import reverse
from django.utils import timezone

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data import research_product_demo
from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset, Listing
from stanstock.data.research_product_demo import execute_demo_product_refresh
from stanstock.research import product_frequency_evidence as evidence
from stanstock.research.models import AnalysisRun
from stanstock.research.price_product_frequencies import classify_terminal_log_returns
from stanstock.research.product_frequency_evidence import (
    FREQUENCY_EVIDENCE_KIND,
    read_registered_product_frequencies,
    register_product_frequencies,
    verify_registered_product_frequencies,
)
from stanstock.research.product_pipeline import CALCULATION_ARTIFACT_KIND
from stanstock.research.product_reader import read_research_product

pytestmark = pytest.mark.django_db


@pytest.fixture
def source_demo(settings, tmp_path, django_user_model, monkeypatch):
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = True
    settings.DATA_DIR = tmp_path
    with monkeypatch.context() as context:
        context.setattr(
            research_product_demo, "register_product_frequencies", lambda **_kwargs: None
        )
        execute_demo_product_refresh(store=AssetStore(tmp_path))
    return django_user_model.objects.create_user(username="frequency-viewer"), AssetStore(tmp_path)


@pytest.fixture
def registered_demo(source_demo):
    _user, store = source_demo
    register_product_frequencies(run=AnalysisRun.objects.get(), store=store)
    return source_demo


def _logical_document(run, store):
    return evidence._derive_document(
        run=run, store=store, derivation_source="manual_backfill", execution_revision="fixture"
    )


def _plant_report(run, store, logical, *, metadata_changes=None, asset_changes=None):
    """Model a forged trusted registry without bypassing either reader or verifier."""
    published_at = timezone.now()
    document = evidence._published_document(logical, derived_at=published_at)
    metadata = evidence._metadata(
        document=document,
        run=run,
        derived_at=published_at,
        logical_sha256=evidence._logical_sha256(document),
    )
    metadata.update(metadata_changes or {})
    stored = store.write_bytes(
        "synthetic/forged-frequency.json", evidence._canonical_bytes(document)
    )
    arguments = {
        "provider": "stanstock",
        "kind": FREQUENCY_EVIDENCE_KIND,
        "schema_version": evidence.FREQUENCY_SCHEMA,
        "subject": evidence._subject(run),
        "relative_path": stored.relative_path,
        "sha256": stored.sha256,
        "retrieved_at": published_at,
        "available_at": published_at,
        "metadata": metadata,
    }
    arguments.update(asset_changes or {})
    return DataAsset.objects.create(**arguments)


def _assert_report_rejected(user, run, store):
    result = read_registered_product_frequencies(
        user=user, run=run, decision_time=timezone.now(), store=store
    )
    assert not result.available
    with pytest.raises((ValueError, RefreshVerificationError)):
        verify_registered_product_frequencies(run=run, store=store)


def test_publication_cannot_release_its_lock_before_an_outer_transaction_commits(source_demo):
    _user, store = source_demo
    run = AnalysisRun.objects.get()
    with transaction.atomic():
        with pytest.raises(RuntimeError, match="durable atomic"):
            register_product_frequencies(run=run, store=store)
    assert not DataAsset.objects.filter(kind=FREQUENCY_EVIDENCE_KIND).exists()
    assert not list((store.root / "research" / "frequencies").rglob("*.json"))


def test_failed_insert_does_not_make_a_later_publication_overwrite_its_blob(
    source_demo, monkeypatch
):
    _user, store = source_demo
    run = AnalysisRun.objects.get()
    published_at = timezone.now() + timedelta(seconds=1)
    monkeypatch.setattr(timezone, "now", lambda: published_at)

    def fail_insert(**_kwargs):
        raise OperationalError("synthetic publication failure")

    with monkeypatch.context() as failed:
        failed.setattr(DataAsset.objects, "create", fail_insert)
        with pytest.raises(OperationalError, match="synthetic publication failure"):
            register_product_frequencies(run=run, store=store)
    assert not DataAsset.objects.filter(kind=FREQUENCY_EVIDENCE_KIND).exists()
    unpublished = list((store.root / "research" / "frequencies").rglob("*.json"))
    assert len(unpublished) == 1
    original_bytes = unpublished[0].read_bytes()
    published_at += timedelta(seconds=1)

    asset = register_product_frequencies(run=run, store=store)

    assert store.resolve(asset.relative_path) != unpublished[0]
    assert unpublished[0].read_bytes() == original_bytes
    assert DataAsset.objects.filter(kind=FREQUENCY_EVIDENCE_KIND).count() == 1
    assert store.resolve(asset.relative_path).stem == asset.sha256
    evidence._validate_registered_asset(asset, store=store, expected_run=run, verify_source=True)


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
    tickers = dict(Listing.objects.values_list("id", "ticker"))
    actual_counts = {
        (tickers[item.listing_id], item.frequency.horizon): (
            item.frequency.counts.loss,
            item.frequency.counts.flat_to_20,
            item.frequency.counts.above_20,
            item.frequency.counts.large_loss,
            item.frequency.zero_drift_counts.loss,
            item.frequency.zero_drift_counts.flat_to_20,
            item.frequency.zero_drift_counts.above_20,
            item.frequency.zero_drift_counts.large_loss,
        )
        for item in result.frequencies
    }
    assert actual_counts == {
        ("ZZRPUP", "6m"): (0, 8192, 0, 0, 4018, 4174, 0, 0),
        ("ZZRPUP", "12m"): (0, 5649, 2543, 0, 4039, 4153, 0, 0),
        ("ZZRPUP", "3y"): (0, 0, 8192, 0, 4087, 4105, 0, 0),
        ("ZZRPUP", "5y"): (0, 0, 8192, 0, 4127, 4065, 0, 0),
        ("ZZRPDOWN", "6m"): (8192, 0, 0, 0, 4096, 4096, 0, 0),
        ("ZZRPDOWN", "12m"): (8192, 0, 0, 0, 4137, 4055, 0, 0),
        ("ZZRPDOWN", "3y"): (8192, 0, 0, 8192, 4056, 4136, 0, 0),
        ("ZZRPDOWN", "5y"): (8192, 0, 0, 8192, 4058, 4132, 2, 0),
        ("ZZRPLOW", "6m"): (0, 8192, 0, 0, 4114, 4078, 0, 0),
        ("ZZRPLOW", "12m"): (0, 1988, 6204, 0, 4077, 4115, 0, 0),
        ("ZZRPLOW", "3y"): (0, 0, 8192, 0, 4068, 4124, 0, 0),
        ("ZZRPLOW", "5y"): (0, 0, 8192, 0, 4065, 4124, 3, 0),
    }

    before = DataAsset.objects.count()
    asset = register_product_frequencies(run=run, store=store)
    assert DataAsset.objects.count() == before
    assert asset.kind == FREQUENCY_EVIDENCE_KIND
    assert asset.schema_version == evidence.FREQUENCY_SCHEMA
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


def test_native_detail_distinguishes_not_derived_from_corrupt_evidence(source_demo, client) -> None:
    user, store = source_demo
    client.force_login(user)
    product = read_research_product(user=user, store=store)
    card = product.cards[0]
    assert card.projections[0].frequency_reason == "frequency_not_derived_at_decision_time"

    absent = client.get(reverse("stock-detail", args=[card.analysis.listing_id]))
    assert absent.status_code == 200
    assert b"not derived at decision time" in absent.content.lower()

    register_product_frequencies(run=AnalysisRun.objects.get(), store=store)
    asset = DataAsset.objects.get(kind=FREQUENCY_EVIDENCE_KIND)
    store.resolve(asset.relative_path).write_bytes(b"corrupt")
    product = read_research_product(user=user, store=store)
    assert product.cards[0].projections[0].frequency_reason == "frequency_evidence_invalid"
    corrupt = client.get(reverse("stock-detail", args=[card.analysis.listing_id]))
    assert corrupt.status_code == 200
    assert b"frequency evidence invalid" in corrupt.content.lower()


def test_frequency_command_verifies_one_explicit_run_without_writing(registered_demo) -> None:
    _user, _store = registered_demo
    run = AnalysisRun.objects.get()
    before = DataAsset.objects.count()

    call_command("derive_price_frequencies", "--run", str(run.id), "--verify")

    assert DataAsset.objects.count() == before


def test_offline_verify_preserves_original_execution_metadata(registered_demo, monkeypatch) -> None:
    _user, store = registered_demo
    run = AnalysisRun.objects.get()
    evidence.verify_registered_product_frequencies(run=run, store=store)
    monkeypatch.setattr(evidence, "code_revision", lambda: "different-current-revision")
    evidence.verify_registered_product_frequencies(run=run, store=store)


@pytest.mark.parametrize("value", (0.5, "1", True))
def test_reader_rejects_noninteger_frequency_counts(value) -> None:
    with pytest.raises(ValueError, match="integers"):
        evidence._read_counts(
            {"loss": value, "flat_to_20": 1, "above_20": 8190, "large_loss": 0},
            path_count=8192,
        )


def test_same_quantile_count_forgery_requires_real_offline_rederivation(source_demo) -> None:
    user, store = source_demo
    run = AnalysisRun.objects.get()
    logical = _logical_document(run, store)
    row = next(
        row
        for listing in logical["rows"]
        for row in listing["horizons"]
        if row["counts"]["flat_to_20"] > 0
    )
    projection = deepcopy(row["projection"])
    row["counts"]["loss"] += 1
    row["counts"]["flat_to_20"] -= 1
    _plant_report(run, store, logical)
    assert row["projection"] == projection
    # Structural GET validation is not falsely described as semantic authentication.
    assert read_registered_product_frequencies(
        user=user, run=run, decision_time=timezone.now(), store=store
    ).available
    with pytest.raises(ValueError, match="frequency_verify_logical_content_mismatch"):
        verify_registered_product_frequencies(run=run, store=store)


@pytest.mark.parametrize(
    "field",
    (
        "contract",
        "frequency_schema",
        "method_version",
        "method_identity_sha256",
        "product_version",
        "config_hash",
        "source_run_id",
        "source_target_date",
        "source_snapshot_grade",
        "source_code_revision",
        "derivation_code_revision",
        "derived_at",
        "logical_report_sha256",
        "owner_id",
        "source_provider",
        "path_count",
        "unexpected",
    ),
)
def test_each_registry_metadata_binding_is_authoritative(source_demo, field) -> None:
    user, store = source_demo
    run = AnalysisRun.objects.get()
    _plant_report(run, store, _logical_document(run, store), metadata_changes={field: "wrong"})
    _assert_report_rejected(user, run, store)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "1"),
        ("provider", "wrong"),
        ("kind", "wrong"),
        ("subject", "wrong"),
    ),
)
def test_asset_schema_and_registry_identity_are_authoritative(source_demo, field, value) -> None:
    user, store = source_demo
    run = AnalysisRun.objects.get()
    asset = _plant_report(run, store, _logical_document(run, store), asset_changes={field: value})
    with pytest.raises(ValueError, match="metadata"):
        evidence._validate_registered_asset(asset, store=store, expected_run=run)
    _assert_report_rejected(user, run, store)


@pytest.mark.parametrize(
    "path",
    (
        ("execution",),
        ("source_run",),
        ("rows", 0),
        ("rows", 0, "horizons", 0),
        ("rows", 0, "horizons", 0, "counts"),
        ("rows", 0, "horizons", 0, "zero_drift_counts"),
        ("rows", 0, "horizons", 0, "ledger_returns"),
        ("rows", 0, "horizons", 0, "projection"),
        ("rows", 0, "horizons", 0, "projection", "zero_drift_ledger_returns"),
    ),
)
def test_unexpected_nested_schema_keys_fail_closed(source_demo, path) -> None:
    user, store = source_demo
    run = AnalysisRun.objects.get()
    logical = _logical_document(run, store)
    nested = logical
    for key in path:
        nested = nested[key]
    nested["unexpected"] = "not part of the frozen schema"
    _plant_report(run, store, logical)
    _assert_report_rejected(user, run, store)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("owner_id", "another-owner"),
        ("source_provider", "twelve_data"),
    ),
)
def test_document_owner_and_provider_bind_to_original_source(source_demo, field, value) -> None:
    user, store = source_demo
    run = AnalysisRun.objects.get()
    logical = _logical_document(run, store)
    logical[field] = value
    _plant_report(run, store, logical)
    _assert_report_rejected(user, run, store)


def test_direct_reader_and_recovery_verify_source_bytes_without_simulation(
    registered_demo, monkeypatch
) -> None:
    user, store = registered_demo
    run = AnalysisRun.objects.get()
    source = DataAsset.objects.filter(kind=CALCULATION_ARTIFACT_KIND).first()
    assert source is not None
    store.resolve(source.relative_path).write_bytes(b"corrupted source calculation")

    def forbidden(**_kwargs):
        pytest.fail("Reading or recovering an existing report must not derive new counts")

    monkeypatch.setattr(evidence, "_derive_document", forbidden)
    monkeypatch.setattr(evidence, "simulate_fhs_terminal_logs", forbidden)
    _assert_report_rejected(user, run, store)
    with pytest.raises((ValueError, RefreshVerificationError)):
        register_product_frequencies(run=run, store=store)


@pytest.mark.parametrize("mutation", ("zero_drift_paths", "zero_drift_ledger", "withheld_reason"))
def test_registration_rejects_complete_projection_changes_with_same_drift_triplet(
    source_demo, monkeypatch, mutation
) -> None:
    _user, store = source_demo
    run = AnalysisRun.objects.get()
    if mutation == "zero_drift_paths":
        original = evidence.simulate_fhs_terminal_logs

        def changed(*args, **kwargs):
            terminals = original(*args, **kwargs)
            return replace(
                terminals, zero_drift=tuple(value + 0.25 for value in terminals.zero_drift)
            )

        monkeypatch.setattr(evidence, "simulate_fhs_terminal_logs", changed)
    else:
        original = evidence._forecast_from_source

        def changed(*args, **kwargs):
            forecast = deepcopy(original(*args, **kwargs))
            projection = forecast["projections"][0]
            if mutation == "zero_drift_ledger":
                projection["zero_drift_ledger_returns"]["median"] = "999"
            else:
                projection["insufficiency_reason"] = "different_withheld_reason"
            return forecast

        monkeypatch.setattr(evidence, "_forecast_from_source", changed)
    with pytest.raises(ValueError, match="frozen projection payload"):
        register_product_frequencies(run=run, store=store)
    assert not DataAsset.objects.filter(kind=FREQUENCY_EVIDENCE_KIND).exists()


def test_path_doubling_records_numerical_precision_not_empirical_evidence(source_demo) -> None:
    _user, store = source_demo
    run = AnalysisRun.objects.get()
    config = evidence.load_price_product_config()
    sources = evidence.load_price_product_sources(
        run=run, store=store, config=config, requested_listing_ids=()
    )
    tickers = dict(Listing.objects.values_list("id", "ticker"))
    before = DataAsset.objects.count()
    differences = {}
    for source in sources:
        product_input = source.selection.product_input
        simulation = config.simulation
        filtered = evidence.filter_historical_returns(
            product_input.stock.closes,
            burn_in=simulation.filter_burn_in,
            variance_target_weight=simulation.variance_target_weight,
            variance_persistence=simulation.variance_persistence,
            innovation_weight=simulation.innovation_weight,
        )
        arguments = {
            "seed": evidence.deterministic_seed(
                method_version=evidence.FHS_METHOD_VERSION,
                effective_config_hash=evidence.PRODUCT_EFFECTIVE_CONFIG_HASH,
                listing_id=product_input.listing_id,
                target_date=product_input.target_date,
            ),
            "horizons": tuple(sessions for _name, sessions in simulation.horizons),
            "diagnostic_max_paths": simulation.diagnostic_max_paths,
            "variance_target_weight": simulation.variance_target_weight,
            "variance_persistence": simulation.variance_persistence,
            "innovation_weight": simulation.innovation_weight,
        }
        baseline = evidence.simulate_fhs_terminal_logs(filtered, path_count=8192, **arguments)
        doubled = evidence.simulate_fhs_terminal_logs(filtered, path_count=16384, **arguments)
        for index, (horizon, sessions) in enumerate(simulation.horizons):
            np.testing.assert_array_equal(
                baseline.with_drift[index], doubled.with_drift[index][:8192]
            )
            np.testing.assert_array_equal(
                baseline.zero_drift[index], doubled.zero_drift[index][:8192]
            )
            counts = [
                classify_terminal_log_returns(
                    terminals.with_drift[index],
                    horizon=horizon,
                    sessions=sessions,
                    path_count=terminals.path_count,
                    zero_drift_logs=terminals.zero_drift[index],
                )
                for terminals in (baseline, doubled)
            ]
            differences[(tickers[product_input.listing_id], horizon)] = tuple(
                2 * getattr(getattr(counts[0], group), event)
                - getattr(getattr(counts[1], group), event)
                for group in ("counts", "zero_drift_counts")
                for event in ("loss", "flat_to_20", "above_20", "large_loss")
            )
    assert DataAsset.objects.count() == before
    assert config.simulation.production_paths == 8192
    # Divide these signed numerators by 16,384 for the change in model share.
    # The zero-to-one sampled downside event is retained, not tuned away.
    assert differences == {
        ("ZZRPUP", "6m"): (0, 0, 0, 0, -43, 43, 0, 0),
        ("ZZRPUP", "12m"): (0, -14, 14, 0, -40, 40, 0, 0),
        ("ZZRPUP", "3y"): (0, 0, 0, 0, 16, -16, 0, 0),
        ("ZZRPUP", "5y"): (0, 0, 0, 0, 43, -43, 0, 0),
        ("ZZRPDOWN", "6m"): (0, 0, 0, 0, -2, 2, 0, 0),
        ("ZZRPDOWN", "12m"): (0, 0, 0, -1, 11, -11, 0, 0),
        ("ZZRPDOWN", "3y"): (0, 0, 0, 0, -106, 106, 0, 0),
        ("ZZRPDOWN", "5y"): (0, 0, 0, 0, -22, 23, -1, 0),
        ("ZZRPLOW", "6m"): (0, 0, 0, 0, 25, -25, 0, 0),
        ("ZZRPLOW", "12m"): (0, -6, 6, 0, -56, 56, 0, 0),
        ("ZZRPLOW", "3y"): (0, 0, 0, 0, -103, 103, 0, 0),
        ("ZZRPLOW", "5y"): (0, 0, 0, 0, -74, 71, 3, 0),
    }

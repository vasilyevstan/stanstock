from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from datetime import timedelta
from io import StringIO
from unittest.mock import Mock

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from stanstock.core import research_product_refresh as refresh
from stanstock.core.models import JobRun
from stanstock.core.refresh_verification import replay_recorded_scheduled_refresh
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.assets import read_checksummed_bytes
from stanstock.data.models import DataAsset, ProviderRecord, UniverseMembership
from stanstock.data.research_product import product_membership_payload
from stanstock.portfolio.models import Portfolio, TrackedSymbol
from stanstock.portfolio.watchlist import (
    TrackedSymbolValidationError,
    add_tracked_symbol,
    verified_catalog_references_for_symbols,
)
from stanstock.research.models import AnalysisRun, Prediction
from stanstock.research.price_product_config import PRODUCT_VERSION as V1
from stanstock.research.product_eligibility_pipeline import decision_support_run_assets
from stanstock.research.product_frequency_evidence import verified_registered_frequency_asset
from stanstock.research.product_reader import read_research_product

pytestmark = pytest.mark.django_db
REVISION = "a" * 40


@pytest.fixture
def c2_scheduled_environment(c2_environment, monkeypatch, settings):
    from stanstock.research import product_eligibility_pipeline, product_pipeline

    env = c2_environment
    settings.OWNER_USERNAME = env.owner.username
    settings.SHADOW_STUDY_ENABLED = False
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    monkeypatch.setattr(refresh, "clean_git_revision", lambda _path: REVISION)
    monkeypatch.setattr(product_eligibility_pipeline, "clean_git_revision", lambda _path: REVISION)
    monkeypatch.setattr(product_pipeline, "clean_git_revision", lambda _path: REVISION)
    return env


def _execute(env, **kwargs):
    return refresh.execute_scheduled_research_refresh(
        core_config_path=env.path,
        store=env.store,
        enforce_rate_limit=False,
        **kwargs,
    )


def _deny_acquisition(env):
    env.fetch.reset_mock()
    env.credentials.reset_mock()
    env.fetch.side_effect = AssertionError("no provider acquisition during recovery")
    env.credentials.side_effect = AssertionError("no credential resolution during recovery")


def _asset_manifest(assets):
    pairs = sorted((str(asset.pk), asset.sha256) for asset in assets)
    return {
        "count": len(pairs),
        "hash": hashlib.sha256(
            "\n".join(f"{pk}:{checksum}" for pk, checksum in pairs).encode()
        ).hexdigest(),
    }


def _catalog_only_symbol(env):
    from stanstock.data.live_us import _persist_catalog
    from stanstock.data.providers import twelve_data
    from stanstock.data.providers.contracts import StockCatalog

    original = next(asset for asset in env.catalogs if asset.subject == "NASDAQ")
    payload = json.loads(read_checksummed_bytes(env.store, original))
    payload["data"].append(
        {
            "symbol": "ZZC2CAT",
            "name": "Synthetic catalog-only common stock",
            "currency": "USD",
            "exchange": "NASDAQ",
            "mic_code": "XNAS",
            "country": "United States",
            "type": "Common Stock",
        }
    )
    payload["count"] += 1
    raw = json.dumps(payload).encode()
    references, count = twelve_data.parse_stock_catalog_references(
        raw, exchange="NASDAQ", require_complete=True
    )
    _persist_catalog(
        env.store,
        StockCatalog(
            provider="twelve_data",
            exchange="NASDAQ",
            references=references,
            count=count,
            retrieved_at=env.now - timedelta(minutes=1),
            source_url="https://synthetic.invalid/stocks",
            raw_bytes=raw,
        ),
    )


def test_scheduled_v2_producer_closes_real_graph_and_uses_old_acquisition_mutex(
    c2_scheduled_environment, monkeypatch
):
    from stanstock.data import research_product_jobs
    from stanstock.simulation import paper_jobs

    env = c2_scheduled_environment
    old_lock = research_product_jobs.target_job_lock
    acquired = []

    @contextmanager
    def observed_lock(**identity):
        acquired.append(identity)
        with old_lock(**identity):
            yield

    monkeypatch.setattr(research_product_jobs, "target_job_lock", observed_lock)
    forbidden_paper = Mock(side_effect=AssertionError("paper is default-off"))
    monkeypatch.setattr(paper_jobs, "prepare_paper_capture", forbidden_paper)
    execution = _execute(env)
    parent = execution.parent
    run = AnalysisRun.objects.get()
    assert parent.status == JobRun.Status.SUCCESS
    assert parent.job_name.startswith("scheduled_refresh_research_v2:")
    assert parent.details["profile"] == "research_product_v2"
    assert parent.details["invocation_identity"] == refresh.scheduled_identity(
        env.owner, product_version=env.version
    )
    assert set(parent.details["stages"]) == {"market", "evaluation", "portfolio_snapshots"}
    assert run.config_version == env.version and run.config_hash == env.config_hash
    assert run.code_revision == REVISION and run.issued_on_time
    assert run.universe_snapshot.grade == "observed"
    assert UniverseMembership.objects.count() == 2
    admissions = product_membership_payload(run.universe_snapshot, store=env.store)["admissions"]
    assert len(admissions) == 100
    assert all(
        entry["status"] != "admitted" and entry["reason"]
        for symbol, entry in admissions.items()
        if symbol not in ("AAPL", "NVDA")
    )
    assert execution.analysis_count == 2 and execution.prediction_count == 10
    assert Prediction.objects.count() == 10
    assert not Prediction.objects.exclude(issued_on_time=True).exists()
    assert acquired == [
        {"job_name": "daily_research_v1", "region": "us", "target_date": env.target}
    ]
    assert env.credentials.call_count == 1 and env.fetch.call_count == 4
    frequency = verified_registered_frequency_asset(run=run, store=env.store)
    assert frequency is not None
    assert frequency.schema_version == "research-product-frequencies@1"
    assert json.loads(read_checksummed_bytes(env.store, frequency))["contract"] == (
        "research-product-frequency-evidence@2"
    )
    assert parent.details["frequency_verification"]["frequency_asset_id"] == str(frequency.pk)
    closure = decision_support_run_assets(run=run, store=env.store)
    assert parent.details["verification"]["asset_manifest"] == _asset_manifest(closure)
    assert {"liquidity_price_raw", "liquidity_price_history", "liquidity_run_evidence"} <= {
        asset.kind for asset in closure
    }
    recorded = copy.deepcopy(parent.details)
    replayed = replay_recorded_scheduled_refresh(parent)
    assert replayed.analysis_run == run and replayed.snapshot == run.universe_snapshot
    assert replayed.verification == recorded["verification"]
    parent.refresh_from_db()
    assert parent.details == recorded
    assert str(env.store.root) not in json.dumps(recorded)
    current = read_research_product(user=env.owner, store=env.store)
    assert current.available and len(current.cards) == 2
    assert all(card.liquidity_status == "matched" for card in current.cards)
    assert all(card.suggestion == "buy" for card in current.cards)
    assert execution.paper_capture is None
    forbidden_paper.assert_not_called()
    assert not Portfolio.objects.exists()


def test_completed_scheduled_recovery_precedes_disabled_provider_and_dirty_checkout(
    c2_scheduled_environment, monkeypatch, settings
):
    env = c2_scheduled_environment
    first = _execute(env)
    original = copy.deepcopy(first.parent.details)
    assets = set(DataAsset.objects.values_list("pk", "sha256"))
    predictions = set(Prediction.objects.values_list("pk", flat=True))
    credits = ProviderRecord.objects.get(pk=env.record.pk).metadata["credits_used_local"]
    env.record.enabled = False
    env.record.save(update_fields=["enabled"])
    settings.RESEARCH_PRODUCT_ENABLED = False
    env.clock["now"] += timedelta(days=1)
    _deny_acquisition(env)
    forbidden_revision = Mock(side_effect=AssertionError("completed proof uses recorded revision"))
    monkeypatch.setattr(refresh, "clean_git_revision", forbidden_revision)
    retry = _execute(env)
    assert retry.parent.status == JobRun.Status.SKIPPED
    assert retry.parent.details["successful_run_id"] == str(first.parent.pk)
    assert retry.analysis_count == 2 and retry.prediction_count == 10
    assert set(DataAsset.objects.values_list("pk", "sha256")) == assets
    assert set(Prediction.objects.values_list("pk", flat=True)) == predictions
    assert ProviderRecord.objects.get(pk=env.record.pk).metadata["credits_used_local"] == credits
    first.parent.refresh_from_db()
    assert first.parent.details == original
    env.fetch.assert_not_called()
    env.credentials.assert_not_called()
    forbidden_revision.assert_not_called()


def test_retained_v1_and_v2_parents_replay_their_captured_version_not_current_setting(
    c2_scheduled_environment, settings
):
    env = c2_scheduled_environment
    settings.RESEARCH_PRODUCT_VERSION = V1
    v1 = _execute(env)
    v1_details = copy.deepcopy(v1.parent.details)
    settings.RESEARCH_PRODUCT_VERSION = env.version
    v2 = _execute(env)
    assert v1.parent.job_name.startswith("scheduled_refresh_research_v1:")
    assert v2.parent.job_name.startswith("scheduled_refresh_research_v2:")
    assert v1.parent.pk != v2.parent.pk
    assert AnalysisRun.objects.count() == 2 and Prediction.objects.count() == 20
    _deny_acquisition(env)
    for selected in (env.version, V1):
        settings.RESEARCH_PRODUCT_VERSION = selected
        assert replay_recorded_scheduled_refresh(v1.parent).analysis_run.config_version == V1
        assert (
            replay_recorded_scheduled_refresh(v2.parent).analysis_run.config_version == env.version
        )
    v1.parent.refresh_from_db()
    assert v1.parent.details == v1_details
    env.credentials.assert_not_called()
    env.fetch.assert_not_called()


def test_closed_scheduled_batch_recovers_membership_before_credentials_or_quota(
    c2_scheduled_environment, monkeypatch, settings
):
    from stanstock.data import research_product_jobs

    env = c2_scheduled_environment
    materialize = research_product_jobs.materialize_product_membership
    monkeypatch.setattr(
        research_product_jobs,
        "materialize_product_membership",
        Mock(side_effect=ValueError("synthetic interrupted scheduled membership")),
    )
    with pytest.raises(ValueError, match="market stage failed"):
        _execute(env)
    closed = DataAsset.objects.get(kind="liquidity_run_evidence")
    assert not AnalysisRun.objects.exists()
    assert env.fetch.call_count == 4
    credits = ProviderRecord.objects.get(pk=env.record.pk).metadata["credits_used_local"]
    monkeypatch.setattr(research_product_jobs, "materialize_product_membership", materialize)
    env.record.enabled = False
    env.record.save(update_fields=["enabled"])
    settings.RESEARCH_PRODUCT_ENABLED = False
    _deny_acquisition(env)
    recovered = _execute(env)
    assert recovered.parent.status == JobRun.Status.SUCCESS
    assert recovered.prediction_count == 10
    assert DataAsset.objects.get(kind="liquidity_run_evidence").pk == closed.pk
    assert ProviderRecord.objects.get(pk=env.record.pk).metadata["credits_used_local"] == credits
    assert JobRun.objects.filter(
        job_name__startswith="research_intake_v2:", status=JobRun.Status.FAILED
    ).exists()
    env.fetch.assert_not_called()
    env.credentials.assert_not_called()


def test_real_v2_parent_authorizes_catalog_only_without_upgrading_legacy_consumers(
    c2_scheduled_environment, settings
):
    env = c2_scheduled_environment
    _catalog_only_symbol(env)
    execution = _execute(env)
    _deny_acquisition(env)
    before = set(DataAsset.objects.values_list("pk", flat=True))
    predictions = set(Prediction.objects.values_list("pk", flat=True))
    references = verified_catalog_references_for_symbols(symbols=("ZZC2CAT",), store=env.store)
    assert references["ZZC2CAT"].instrument_type == "Common Stock"
    saved, created = add_tracked_symbol(owner=env.owner, raw_symbol="ZZC2CAT", store=env.store)
    assert created and saved.owner == env.owner
    assert TrackedSymbol.objects.filter(owner=env.owner).count() == 1
    assert set(Prediction.objects.values_list("pk", flat=True)) == predictions
    assert set(DataAsset.objects.values_list("pk", flat=True)) == before
    assert not Portfolio.objects.exists()
    settings.RESEARCH_PRODUCT_VERSION = V1
    assert verified_catalog_references_for_symbols(symbols=("ZZC2CAT",), store=env.store)
    assert replay_recorded_scheduled_refresh(execution.parent).analysis_run.config_version == (
        env.version
    )
    env.credentials.assert_not_called()
    env.fetch.assert_not_called()


@pytest.mark.parametrize(
    "corruption",
    ["identity", "union", "frequency", "frequency_stage", "frequency_attempt", "physical"],
)
def test_claimed_verified_v2_catalog_parent_corruption_fails_closed(
    c2_scheduled_environment, corruption
):
    env = c2_scheduled_environment
    _catalog_only_symbol(env)
    parent = _execute(env).parent
    if corruption == "identity":
        parent.details["invocation_identity"]["source_contract"] = "wrong-source-contract"
    elif corruption == "union":
        parent.details["verification"]["asset_manifest"]["hash"] = "f" * 64
    elif corruption == "frequency":
        del parent.details["frequency_verification"]
    elif corruption == "frequency_stage":
        parent.details["frequencies"]["frequency_asset_sha256"] = "f" * 64
    elif corruption == "frequency_attempt":
        parent.details["frequencies"]["attempt"] += 1
    else:
        asset = DataAsset.objects.filter(kind="liquidity_price_raw").first()
        env.store.resolve(asset.relative_path).write_bytes(b"corrupt synthetic liquidity evidence")
    parent.save(update_fields=["details"])
    _deny_acquisition(env)
    with pytest.raises(RefreshVerificationError):
        replay_recorded_scheduled_refresh(parent)
    with pytest.raises(TrackedSymbolValidationError, match="unavailable or invalid"):
        add_tracked_symbol(owner=env.owner, raw_symbol="ZZC2CAT", store=env.store)
    assert not TrackedSymbol.objects.exists()
    assert read_research_product(user=env.owner, store=env.store).status == (
        "integrity_failed" if corruption == "physical" else "available"
    )
    env.credentials.assert_not_called()
    env.fetch.assert_not_called()


@pytest.mark.parametrize("entry_point", ["scheduled", "daily_with_captured_intake"])
def test_disabled_v2_rejects_fresh_work_before_evidence_credentials_or_quota(
    c2_scheduled_environment, monkeypatch, settings, entry_point
):
    from stanstock.data import live_us
    from stanstock.data.research_product import capture_authorized_product_intake
    from stanstock.data.research_product_jobs import execute_daily_research_job

    env = c2_scheduled_environment
    issuance_key = "disabled-fresh-direct"
    if entry_point == "daily_with_captured_intake":
        capture_authorized_product_intake(
            target_date=env.target,
            evidence_grade="research",
            issuance_key=issuance_key,
            owner=env.owner,
            core_config_path=env.path,
            policy_identity=env.config_hash,
            captured_at=env.now,
            store=env.store,
            product_version=env.version,
        )
    assert not AnalysisRun.objects.exists()
    assert not DataAsset.objects.filter(kind="liquidity_run_evidence").exists()
    settings.RESEARCH_PRODUCT_ENABLED = False
    assets = set(DataAsset.objects.values_list("pk", "sha256"))
    files = {
        path.relative_to(env.store.root): path.read_bytes()
        for path in env.store.root.rglob("*")
        if path.is_file()
    }
    metadata = copy.deepcopy(env.record.metadata)

    def synthetic_sleep(seconds):
        env.clock["now"] += timedelta(seconds=seconds)

    monkeypatch.setattr(live_us, "sleep", synthetic_sleep)
    output = StringIO()
    if entry_point == "scheduled":
        with pytest.raises(CommandError, match="Scheduled research refresh failed"):
            call_command("scheduled_refresh", config=env.path, stdout=output)
        parent = JobRun.objects.get(job_name__startswith="scheduled_refresh_research_v2:")
        assert parent.status == JobRun.Status.FAILED and parent.finished_at is not None
    else:
        with pytest.raises(ValueError, match="Research product v2 is disabled"):
            execute_daily_research_job(
                target_date=env.target,
                owner=env.owner,
                issuance_key=issuance_key,
                store=env.store,
                core_config_path=env.path,
                enforce_rate_limit=False,
                product_version=env.version,
            )
    daily = JobRun.objects.get(job_name__startswith="daily_research_v2:")
    assert daily.status == JobRun.Status.FAILED and daily.finished_at is not None
    assert "Research product v2 is disabled" in daily.error
    assert not JobRun.objects.filter(status=JobRun.Status.RUNNING).exists()
    assert "status=success" not in output.getvalue() and "status=skipped" not in output.getvalue()
    assert set(DataAsset.objects.values_list("pk", "sha256")) == assets
    assert {
        path.relative_to(env.store.root): path.read_bytes()
        for path in env.store.root.rglob("*")
        if path.is_file()
    } == files
    assert not AnalysisRun.objects.exists()
    assert not Prediction.objects.exists()
    assert not UniverseMembership.objects.exists()
    env.record.refresh_from_db()
    assert env.record.metadata == metadata
    env.fetch.assert_not_called()
    env.credentials.assert_not_called()


def test_scheduled_cli_selects_v2_from_environment_setting_and_recovers_when_disabled(
    c2_scheduled_environment, monkeypatch, settings
):
    from stanstock.data import live_us

    env = c2_scheduled_environment

    def synthetic_sleep(seconds):
        env.clock["now"] += timedelta(seconds=seconds)

    monkeypatch.setattr(live_us, "sleep", synthetic_sleep)
    output = StringIO()
    call_command("scheduled_refresh", config=env.path, stdout=output)
    assert "profile=research_product_v2 status=success" in output.getvalue()
    assert "paper_capture" not in output.getvalue()
    issued_run_id = AnalysisRun.objects.get().pk
    predictions = set(Prediction.objects.values_list("pk", flat=True))
    assets = set(DataAsset.objects.values_list("pk", "sha256"))
    credits = ProviderRecord.objects.get(pk=env.record.pk).metadata["credits_used_local"]
    _deny_acquisition(env)
    settings.RESEARCH_PRODUCT_ENABLED = False
    output = StringIO()
    call_command("scheduled_refresh", config=env.path, stdout=output)
    assert "profile=research_product_v2 status=skipped" in output.getvalue()
    assert AnalysisRun.objects.get().pk == issued_run_id
    assert set(Prediction.objects.values_list("pk", flat=True)) == predictions
    assert set(DataAsset.objects.values_list("pk", "sha256")) == assets
    assert ProviderRecord.objects.get(pk=env.record.pk).metadata["credits_used_local"] == credits
    env.credentials.assert_not_called()
    env.fetch.assert_not_called()


@pytest.mark.parametrize("lost_authority", ["owner", "inactive_owner", "rights", "stale"])
def test_v2_catalog_rechecks_current_owner_rights_and_catalog_age(
    c2_scheduled_environment, django_user_model, settings, lost_authority
):
    env = c2_scheduled_environment
    _catalog_only_symbol(env)
    _execute(env)
    _deny_acquisition(env)
    if lost_authority == "owner":
        other = django_user_model.objects.create_user(username="c2-other-scheduled-owner")
        settings.OWNER_USERNAME = other.username
    elif lost_authority == "inactive_owner":
        env.owner.is_active = False
        env.owner.save(update_fields=["is_active"])
    elif lost_authority == "rights":
        env.record.refresh_from_db()
        env.record.metadata["internal_display_rights_confirmed"] = False
        env.record.save(update_fields=["metadata"])
    else:
        env.clock["now"] += timedelta(days=8)
    with pytest.raises(TrackedSymbolValidationError):
        add_tracked_symbol(owner=env.owner, raw_symbol="ZZC2CAT", store=env.store)
    assert not TrackedSymbol.objects.exists()
    env.credentials.assert_not_called()
    env.fetch.assert_not_called()


def test_scheduled_v2_cannot_inherit_another_owners_v2_or_its_own_v1_declaration(
    c2_scheduled_environment, django_user_model, settings
):
    from stanstock.research.forward_scorecard import declare_forward_protocol

    env = c2_scheduled_environment
    other = django_user_model.objects.create_user(username="c2-owner-with-v1-only-declaration")
    settings.OWNER_USERNAME = other.username
    env.clock["now"] -= timedelta(days=1)
    old = declare_forward_protocol(user=other, store=env.store, product_version=V1)
    env.clock["now"] = env.now
    before = set(DataAsset.objects.values_list("pk", "sha256"))
    _deny_acquisition(env)
    with pytest.raises(ValueError, match="market stage failed"):
        _execute(env)
    assert set(DataAsset.objects.values_list("pk", "sha256")) == before
    assert DataAsset.objects.get(pk=old.asset.pk).sha256 == old.asset.sha256
    assert not AnalysisRun.objects.exists()
    assert not Prediction.objects.exists()
    assert not JobRun.objects.filter(job_name__startswith="daily_research_v1:").exists()
    env.credentials.assert_not_called()
    env.fetch.assert_not_called()


@pytest.mark.parametrize("c2_environment", [{"ignored_none": True}], indirect=True)
def test_scheduled_failed_canary_withholds_whole_batch_not_independent_forecasts(
    c2_scheduled_environment,
):
    env = c2_scheduled_environment
    execution = _execute(env)
    assert execution.parent.status == JobRun.Status.SUCCESS
    assert execution.prediction_count == 10 and env.fetch.call_count == 3
    assert [call.args[0] for call in env.fetch.call_args_list] == ["NVDA", "NVDA", "AAPL"]
    current = read_research_product(user=env.owner, store=env.store)
    assert current.available and len(current.cards) == 2
    for card in current.cards:
        assert card.liquidity_status == "unavailable"
        assert "liquidity_mode_witness_failed" in card.liquidity_reasons
        assert card.average_dollar_turnover_20d is None
        assert card.suggestion == "hold" and not card.current_promotion_eligible
        assert all(projection.available for projection in card.projections)
    assert not Prediction.objects.filter(evidence_role="advisory", bear_return=None).exists()
    assert not Prediction.objects.exclude(probability_positive=None).exists()
    assert not Prediction.objects.exclude(confidence=None).exists()


def test_opted_in_paper_noop_keeps_recovery_order_and_does_not_retarget_v2_parent(
    c2_scheduled_environment, monkeypatch, settings
):
    from stanstock.simulation import paper_jobs

    env = c2_scheduled_environment
    invocation_class = paper_jobs.PaperCaptureInvocation
    original_recover = invocation_class.recover
    original_after_market = invocation_class.after_market
    prepare = Mock(wraps=paper_jobs.prepare_paper_capture)
    events = []

    def observed_recover(invocation):
        events.append(("recover", env.fetch.call_count))
        return original_recover(invocation)

    def observed_after_market(invocation, *, attempted):
        events.append(("after_market", env.fetch.call_count))
        return original_after_market(invocation, attempted=attempted)

    monkeypatch.setattr(paper_jobs, "prepare_paper_capture", prepare)
    monkeypatch.setattr(invocation_class, "recover", observed_recover)
    monkeypatch.setattr(invocation_class, "after_market", observed_after_market)
    execution = _execute(env, capture_paper=True)
    assert execution.parent.status == JobRun.Status.SUCCESS
    assert events == [("recover", 0), ("after_market", 4)]
    result = execution.paper_capture
    assert result is not None and result.capture_count == 0
    assert result.state == "unactivated" and result.reason_code is None
    assert result.child is None
    assert set(prepare.call_args.kwargs) == {"owner", "session", "store"}
    assert "paper_capture" not in execution.parent.details
    assert not DataAsset.objects.filter(kind__startswith="paper").exists()
    assert not Portfolio.objects.exists()
    details = copy.deepcopy(execution.parent.details)
    assets = set(DataAsset.objects.values_list("pk", "sha256"))
    _deny_acquisition(env)
    events.clear()
    settings.RESEARCH_PRODUCT_ENABLED = False
    retry = _execute(env, capture_paper=True)
    assert retry.parent.status == JobRun.Status.SKIPPED
    assert events == [("recover", 0), ("after_market", 0)]
    assert retry.paper_capture == result
    assert set(DataAsset.objects.values_list("pk", "sha256")) == assets
    execution.parent.refresh_from_db()
    assert execution.parent.details == details
    env.credentials.assert_not_called()
    env.fetch.assert_not_called()

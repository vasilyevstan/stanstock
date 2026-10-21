from __future__ import annotations

from unittest.mock import Mock

import pytest

from stanstock.core.models import JobRun
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.models import DataAsset, LatestMarketData, ProviderRecord
from stanstock.data.research_product import product_membership_payload
from stanstock.data.research_product_jobs import execute_daily_research_job
from stanstock.research.models import AnalysisRun, Prediction
from stanstock.research.product_eligibility_pipeline import verify_decision_support_output

pytestmark = pytest.mark.django_db


def _daily(env, **kwargs):
    return execute_daily_research_job(
        target_date=env.target,
        owner=env.owner,
        issuance_key="c2-native-fixture",
        product_version=env.version,
        core_config_path=env.path,
        store=env.store,
        enforce_rate_limit=False,
        derive_frequencies=False,
        **kwargs,
    )


def _credits():
    return ProviderRecord.objects.get(provider="twelve_data").metadata["credits_used_local"]


def test_existing_daily_path_charges_four_requests_and_no_duplicate_split_fetch(c2_environment):
    env = c2_environment
    job = _daily(env)
    assert job.status == JobRun.Status.SUCCESS
    assert job.job_name.startswith("daily_research_v2:")
    assert job.details["credits_used"] == 4
    assert env.credentials.call_count == 1
    assert len(env.fetch.call_args_list) == 4
    assert [(call.args[0], call.kwargs["adjustment"]) for call in env.fetch.call_args_list] == [
        ("NVDA", "splits"),
        ("NVDA", "none"),
        ("AAPL", "splits"),
        ("AAPL", "none"),
    ]
    assert AnalysisRun.objects.count() == 1 and Prediction.objects.count() == 10
    assert not Prediction.objects.filter(issued_on_time=True).exists()
    assert not LatestMarketData.objects.exclude(source_asset__kind="price_history").exists()
    run = AnalysisRun.objects.get()
    verify_decision_support_output(run=run, store=env.store)
    assert (
        len(product_membership_payload(run.universe_snapshot, store=env.store)["admissions"]) == 100
    )


def test_original_core_plus_twenty_saved_denominator_survives_rejected_names_and_retry(
    c2_environment, django_user_model
):
    from stanstock.data.research_product import load_product_intake
    from stanstock.portfolio.models import TrackedSymbol
    from stanstock.research.product_reader import read_research_product

    env = c2_environment
    saved = {f"ZZC{chr(65 + index)}" for index in range(20)}
    for symbol in sorted(saved):
        TrackedSymbol.objects.create(owner=env.owner, symbol=symbol)
    other = django_user_model.objects.create_user(username="c2-unrelated-saved-owner")
    TrackedSymbol.objects.create(owner=other, symbol="ZZOTHER")
    first = _daily(env)
    run = AnalysisRun.objects.get()
    intake = load_product_intake(
        target_date=env.target,
        owner_id=str(env.owner.pk),
        issuance_key="c2-native-fixture",
        store=env.store,
        product_version=env.version,
    )
    assert set(intake.saved_symbols) == saved
    expected = set(intake.core_symbols) | saved
    assert len(expected) == 120 and "ZZOTHER" not in expected
    membership = product_membership_payload(run.universe_snapshot, store=env.store)
    assert set(membership["admissions"]) == expected
    for symbol in saved:
        assert membership["admissions"][symbol]["status"] == "identity_rejected"
        assert membership["admissions"][symbol]["reason"] == "catalog_identity_missing_or_ambiguous"
    current = read_research_product(user=env.owner, store=env.store)
    assert current.available and len(current.admissions) == 120
    assert len(current.cards) == current.admitted_count == 2
    assert Prediction.objects.count() == 10 and env.fetch.call_count == 4
    sources = set(DataAsset.objects.values_list("id", "sha256"))
    predictions = set(Prediction.objects.values_list("id", flat=True))
    credits = _credits()
    TrackedSymbol.objects.filter(owner=env.owner).delete()
    TrackedSymbol.objects.create(owner=env.owner, symbol="ZZNEW")
    env.credentials.reset_mock()
    env.fetch.reset_mock()
    env.credentials.side_effect = AssertionError("completed captured intent cannot resolve a key")
    env.fetch.side_effect = AssertionError(
        "completed captured intent cannot acquire a new saved name"
    )
    retry = _daily(env)
    assert retry.status == "skipped" and retry.details["successful_run_id"] == str(first.id)
    assert set(DataAsset.objects.values_list("id", "sha256")) == sources
    assert set(Prediction.objects.values_list("id", flat=True)) == predictions
    assert _credits() == credits
    env.credentials.assert_not_called()
    env.fetch.assert_not_called()


def test_completed_retry_precedes_disabled_credentials_and_spends_no_more_credit(
    c2_environment, settings
):
    env = c2_environment
    first = _daily(env)
    before_ids = set(DataAsset.objects.values_list("id", flat=True))
    credits = _credits()
    env.record.enabled = False
    env.record.save(update_fields=["enabled"])
    settings.RESEARCH_PRODUCT_ENABLED = False
    env.credentials.side_effect = AssertionError("credential resolution forbidden on recovery")
    env.fetch.side_effect = AssertionError("provider acquisition forbidden on recovery")
    env.credentials.reset_mock()
    env.fetch.reset_mock()
    retry = _daily(env)
    assert retry.status == JobRun.Status.SKIPPED
    assert retry.details["successful_run_id"] == str(first.id)
    assert set(DataAsset.objects.values_list("id", flat=True)) == before_ids
    assert _credits() == credits
    env.credentials.assert_not_called()
    env.fetch.assert_not_called()


def test_closed_whole_batch_recovers_after_membership_failure_without_credentials(
    c2_environment, monkeypatch, settings
):
    env = c2_environment
    from stanstock.data import research_product_jobs as jobs

    materialize = jobs.materialize_product_membership
    monkeypatch.setattr(
        jobs,
        "materialize_product_membership",
        Mock(side_effect=ValueError("synthetic interrupted membership commit")),
    )
    with pytest.raises(ValueError, match="synthetic interrupted"):
        _daily(env)
    assert DataAsset.objects.filter(kind="liquidity_run_evidence").count() == 1
    assert not AnalysisRun.objects.exists()
    original_sources = set(DataAsset.objects.values_list("id", flat=True))
    credits = _credits()
    env.record.enabled = False
    env.record.save(update_fields=["enabled"])
    settings.RESEARCH_PRODUCT_ENABLED = False
    env.credentials.side_effect = AssertionError("closed batch must precede credentials")
    env.fetch.side_effect = AssertionError("closed batch must precede network")
    monkeypatch.setattr(jobs, "materialize_product_membership", materialize)
    recovered = _daily(env)
    assert recovered.status == JobRun.Status.SUCCESS
    assert AnalysisRun.objects.count() == 1
    assert set(
        DataAsset.objects.filter(
            kind__in=["price_history", "liquidity_price_history", "liquidity_run_evidence"]
        ).values_list("id", flat=True)
    ).issubset(original_sources)
    assert _credits() == credits
    verify_decision_support_output(run=AnalysisRun.objects.get(), store=env.store)


@pytest.mark.parametrize("c2_environment", [{"ignored_none": True}], indirect=True)
def test_failed_fixed_control_stops_only_extra_c2_requests_and_split_research_continues(
    c2_environment,
):
    env = c2_environment
    job = _daily(env)
    assert job.status == JobRun.Status.SUCCESS
    assert [(call.args[0], call.kwargs["adjustment"]) for call in env.fetch.call_args_list] == [
        ("NVDA", "splits"),
        ("NVDA", "none"),
        ("AAPL", "splits"),
    ]
    assert (
        Prediction.objects.filter(evidence_role="advisory", bear_return__isnull=False).count() == 8
    )
    decisions = list(Prediction.objects.filter(evidence_role="decision"))
    assert len(decisions) == 2
    assert all(
        row.recommendation == "hold" and row.calculation["liquidity"]["status"] == "unavailable"
        for row in decisions
    )
    assert all(row.calculation["risk"]["average_dollar_turnover_20d"] is None for row in decisions)


def test_missing_separate_declaration_refuses_before_provider_or_output(c2_environment):
    env = c2_environment
    env.store.resolve(env.declaration.asset.relative_path).unlink()
    with pytest.raises(RefreshVerificationError, match="could not be read"):
        _daily(env)
    env.fetch.assert_not_called()
    env.credentials.assert_not_called()
    assert not AnalysisRun.objects.exists()


def test_normal_quota_withholds_extra_pairs_without_spending_split_research_budget(c2_environment):
    env = c2_environment
    env.record.metadata["daily_credit_limit"] = 3
    env.record.save(update_fields=["metadata"])
    job = _daily(env)
    assert job.status == JobRun.Status.SUCCESS and job.details["credits_used"] == 2
    assert len(env.fetch.call_args_list) == 2
    assert all(call.kwargs["adjustment"] == "splits" for call in env.fetch.call_args_list)
    assert (
        Prediction.objects.filter(evidence_role="advisory", bear_return__isnull=False).count() == 8
    )
    assert all(
        "liquidity_quota_unavailable" in row.calculation["liquidity"]["reasons"]
        for row in Prediction.objects.filter(evidence_role="decision")
    )


def test_disabled_provider_with_qualified_cached_splits_does_not_resolve_a_key_for_extra_pairs(
    c2_environment,
):
    from datetime import timedelta

    from stanstock.data.live_us import _persist_price_series

    env = c2_environment
    for symbol in env.listings:
        _persist_price_series(
            store=env.store,
            listing=None,
            series=env.series(symbol, retrieved_at=env.now - timedelta(minutes=5)),
        )
    env.record.enabled = False
    env.record.save(update_fields=["enabled"])
    env.credentials.side_effect = AssertionError("provider disabled: no credentials")
    env.fetch.side_effect = AssertionError("provider disabled: no fetch")
    job = _daily(env)
    assert job.status == JobRun.Status.SUCCESS and job.details["credits_used"] == 0
    env.credentials.assert_not_called()
    env.fetch.assert_not_called()
    assert all(
        "liquidity_provider_unavailable" in row.calculation["liquidity"]["reasons"]
        for row in Prediction.objects.filter(evidence_role="decision")
    )


def test_unclosed_attempt_is_not_recovered_or_mixed_even_at_equal_clock_precision(
    c2_environment, monkeypatch
):
    from stanstock.data import liquidity_evidence

    env = c2_environment
    original_close = liquidity_evidence.close_liquidity_batch
    monkeypatch.setattr(
        liquidity_evidence,
        "close_liquidity_batch",
        Mock(side_effect=ValueError("synthetic unclosed attempt")),
    )
    with pytest.raises(ValueError, match="synthetic unclosed"):
        _daily(env)
    old_splits = set(
        DataAsset.objects.filter(kind="price_history")
        .exclude(subject="SPY")
        .values_list("id", flat=True)
    )
    assert (
        len(old_splits) == 2
        and not DataAsset.objects.filter(kind="liquidity_run_evidence").exists()
    )
    before_credits = _credits()
    before_calls = env.fetch.call_count
    monkeypatch.setattr(liquidity_evidence, "close_liquidity_batch", original_close)
    assert _daily(env).status == "success"
    assert _credits() - before_credits == env.fetch.call_count - before_calls == 4
    closed = DataAsset.objects.get(kind="liquidity_run_evidence")
    from stanstock.data.research_product import load_product_intake

    intake = load_product_intake(
        target_date=env.target,
        owner_id=str(env.owner.pk),
        issuance_key="c2-native-fixture",
        store=env.store,
        product_version=env.version,
    )
    verified = liquidity_evidence.verify_liquidity_batch(
        asset=closed, intake=intake, cutoff=env.now, store=env.store
    )
    assert verified.document["attempt"] == 2
    for entry in verified.document["entries"]:
        if entry["split_asset"] is not None:
            assert entry["split_asset"]["id"] not in {str(value) for value in old_splits}
            asset = DataAsset.objects.get(pk=entry["split_asset"]["id"])
            assert (
                asset.metadata["acquisition_context"]["job_id"]
                == verified.document["acquisition_job_id"]
            )
    verify_decision_support_output(run=AnalysisRun.objects.get(), store=env.store)

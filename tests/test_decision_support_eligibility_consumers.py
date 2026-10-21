from __future__ import annotations

import hashlib
from datetime import timedelta
from io import StringIO
from unittest.mock import Mock

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from stanstock.data.models import DataAsset, ProviderRecord
from stanstock.research.decision_support_eligibility import (
    canonical_json,
    default_decision_support_config_path,
)
from stanstock.research.forward_scorecard import (
    FORWARD_PROTOCOL,
    FORWARD_PROTOCOL_SHA256,
    FORWARD_PROTOCOL_V2,
    FORWARD_PROTOCOL_V2_SHA256,
    declare_forward_protocol,
    read_forward_scorecard,
    require_v2_forward_declaration,
)
from stanstock.research.models import AnalysisRun, Prediction
from stanstock.research.outcomes import evaluate_prediction
from stanstock.research.product_frequency_evidence import register_product_frequencies
from stanstock.research.product_reader import (
    read_research_product,
    read_research_product_history,
    read_research_product_performance,
)
from stanstock.research.service import analyze_snapshot
from stanstock.simulation.builders import SimulationWorkflowError, build_signals_for_backtest

pytestmark = pytest.mark.django_db


def test_real_current_reader_exposes_verified_v2_liquidity_and_all_four_frozen_horizons(
    c2_persisted,
):
    env = c2_persisted
    current = read_research_product(user=env.owner, store=env.store)
    assert current.status == "available"
    assert len(current.admissions) == 100 and current.admitted_count == 2
    assert len(current.cards) == 2
    for card in current.cards:
        assert card.product_version == env.version and card.liquidity_status == "matched"
        assert len(card.liquidity_window_dates) == 20
        assert card.liquidity_window_dates[-1] == env.target
        assert card.suggestion == "buy" and card.direction == "positive"
        assert card.average_dollar_turnover_20d > 5_000_000
        assert [row.horizon for row in card.projections] == ["6m", "12m", "3y", "5y"]
        assert all(row.available and row.frequency_status == "absent" for row in card.projections)


def test_selected_v1_does_not_use_v2_and_history_still_reads_the_captured_version(
    c2_persisted, settings
):
    env = c2_persisted
    settings.RESEARCH_PRODUCT_VERSION = "research-product-v1"
    assert read_research_product(user=env.owner, store=env.store).status == "absent"
    history = read_research_product_history(user=env.owner, store=env.store)
    assert history.status == "available"
    assert history.current.status == "absent"
    assert len(history.cohorts) == 1 and history.cohorts[0].run.config_version == env.version


@pytest.mark.parametrize("revoke", ["owner", "rights"])
def test_current_history_and_performance_refuse_revoked_current_authority(c2_persisted, revoke):
    env = c2_persisted
    if revoke == "owner":
        env.owner.is_active = False
        env.owner.save(update_fields=["is_active"])
    else:
        record = ProviderRecord.objects.get(provider="twelve_data")
        record.metadata["internal_display_rights_confirmed"] = False
        record.save(update_fields=["metadata"])
    for read in (
        read_research_product,
        read_research_product_history,
        read_research_product_performance,
    ):
        result = read(user=env.owner, store=env.store)
        assert result.status == "unauthorized"


def test_new_request_rechecks_frequency_child_and_physical_source_not_session_cache(c2_persisted):
    env = c2_persisted
    frequency = register_product_frequencies(run=env.run, store=env.store)
    assert read_research_product_history(user=env.owner, store=env.store).available
    env.store.resolve(frequency.relative_path).write_bytes(b"synthetic corrupt frequency child")
    assert (
        read_research_product_history(user=env.owner, store=env.store).status == "integrity_failed"
    )
    assert (
        read_research_product_performance(user=env.owner, store=env.store).status
        == "integrity_failed"
    )
    assert read_research_product(user=env.owner, store=env.store).status == "integrity_failed"


def test_protocols_are_separate_pinned_and_neither_redeclaration_resets_the_other(c2_persisted):
    env = c2_persisted
    old = declare_forward_protocol(user=env.owner, store=env.store)
    old_bytes = env.store.read_bytes(old.asset.relative_path)
    new = declare_forward_protocol(user=env.owner, store=env.store, product_version=env.version)
    assert new.asset.id == env.declaration.asset.id != old.asset.id
    assert declare_forward_protocol(user=env.owner, store=env.store).asset.id == old.asset.id
    assert env.store.read_bytes(old.asset.relative_path) == old_bytes
    assert hashlib.sha256(canonical_json(FORWARD_PROTOCOL)).hexdigest() == FORWARD_PROTOCOL_SHA256
    assert (
        hashlib.sha256(canonical_json(FORWARD_PROTOCOL_V2)).hexdigest()
        == FORWARD_PROTOCOL_V2_SHA256
    )
    report = read_forward_scorecard(user=env.owner, store=env.store, product_version=env.version)
    assert report["protocol_version"] == "forward-forecast-scorecard-v2"
    assert all(group["config_hash"] == env.config_hash for group in report["groups"])
    assert report["conclusion"] == "unmatured; skill not established"
    assert sum(report["admission_denominators"].values()) == 100
    with pytest.raises(ValueError, match="prospective"):
        require_v2_forward_declaration(
            owner_id=str(env.owner.id),
            target_date=env.target,
            before=new.declared_at,
            store=env.store,
        )


@pytest.mark.parametrize("c2_environment", [{"flat": True}], indirect=True)
def test_genuine_withheld_forecast_roundtrips_to_reader_and_never_reads_outcome_prices(
    c2_persisted, monkeypatch
):
    env = c2_persisted
    current = read_research_product(user=env.owner, store=env.store)
    assert current.status == "available"
    assert all(
        not projection.available and projection.insufficiency_reason
        for card in current.cards
        for projection in card.projections
    )
    blocked = Mock(side_effect=AssertionError("withheld advisory must not look up prices"))
    monkeypatch.setattr("stanstock.research.outcomes.AsOfData", blocked)
    for row in Prediction.objects.filter(evidence_role="advisory"):
        assert row.bear_return is row.base_return is row.bull_return is None
        outcome = evaluate_prediction(
            row,
            provider="twelve_data",
            evaluation_date=env.target,
            evaluation_time=env.now,
            benchmark_subject="SPY",
            store=env.store,
        ).outcome
        assert outcome.status == "unresolved" and outcome.success is None
    blocked.assert_not_called()


@pytest.mark.parametrize(
    "c2_environment", [{"price_scale": "0.05", "volume_scale": 100}], indirect=True
)
def test_under_ten_never_buy_or_promotion_even_with_valid_large_proxy(c2_persisted):
    env = c2_persisted
    cards = read_research_product(user=env.owner, store=env.store).cards
    assert len(cards) == 2
    assert all(
        card.target_under_10
        and card.suggestion != "buy"
        and not card.current_promotion_eligible
        and card.average_dollar_turnover_20d > 5_000_000
        for card in cards
    )


@pytest.mark.parametrize("c2_environment", [{"down": True, "missing_volume": True}], indirect=True)
def test_negative_avoid_survives_unavailable_liquidity(c2_persisted):
    env = c2_persisted
    card = next(
        card
        for card in read_research_product(user=env.owner, store=env.store).cards
        if card.listing.ticker == "AAPL"
    )
    assert card.suggestion == "avoid" and card.direction == "negative"
    assert card.liquidity_status == "unavailable" and card.average_dollar_turnover_20d is None


def test_legacy_numeric_simulation_explicitly_rejects_v2_scoreless_research(c2_persisted):
    env = c2_persisted
    with pytest.raises(SimulationWorkflowError, match="Scoreless"):
        build_signals_for_backtest(
            snapshot=env.snapshot, start_date=env.target, end_date=env.target
        )


def test_legacy_sample_satellite_and_predict_refuse_v2_without_allocating_or_appending(
    c2_persisted,
):
    from stanstock.portfolio.models import Portfolio, PortfolioHolding
    from stanstock.portfolio.planner import _qualified_satellite
    from stanstock.portfolio.service import PortfolioValuationError, build_sample_portfolio

    env = c2_persisted
    before = (
        Portfolio.objects.count(),
        PortfolioHolding.objects.count(),
        Prediction.objects.count(),
    )
    with pytest.raises(PortfolioValuationError, match="legacy scored sample builder"):
        build_sample_portfolio(owner=env.owner, source_run=env.run)
    satellite, reason = _qualified_satellite(as_of_date=env.target, today=env.target)
    assert satellite is None
    assert reason == (
        "The active momentum method has no short-horizon satellite lane; cash is carried."
    )
    with pytest.raises(CommandError, match="prospective scoreless"):
        call_command("predict", analysis=str(env.run.stocks.first().pk), stdout=StringIO())
    assert (
        Portfolio.objects.count(),
        PortfolioHolding.objects.count(),
        Prediction.objects.count(),
    ) == before


def test_selected_v2_demo_command_refuses_without_seed_or_any_new_evidence(
    c2_environment, monkeypatch
):
    before = set(DataAsset.objects.values_list("id", flat=True))
    blocked = Mock(side_effect=AssertionError("v2 demo must not seed"))
    monkeypatch.setattr("stanstock.data.management.commands.refresh_demo.call_command", blocked)
    with pytest.raises(CommandError, match="refuses synthetic_demo"):
        call_command("refresh_demo", stdout=StringIO())
    blocked.assert_not_called()
    assert set(DataAsset.objects.values_list("id", flat=True)) == before


@pytest.mark.parametrize("c2_environment", [{"grade": "observed"}], indirect=True)
def test_manual_analyze_command_never_inherits_observed_on_time_grade(c2_snapshot):
    env = c2_snapshot
    call_command(
        "analyze",
        snapshot=str(env.snapshot.id),
        provider="twelve_data",
        benchmark_subject="SPY",
        target_date=env.target.isoformat(),
        config=default_decision_support_config_path(),
        stdout=StringIO(),
    )
    assert Prediction.objects.filter(evidence_grade="observed").count() == 10
    assert not Prediction.objects.filter(issued_on_time=True).exists()


def test_c2_frequency_adapter_does_not_broaden_frozen_v1_study_loader(c2_persisted):
    from stanstock.research.price_product_config import load_price_product_config
    from stanstock.research.price_product_study import load_price_product_sources

    env = c2_persisted
    with pytest.raises(ValueError):
        load_price_product_sources(
            run=env.run,
            store=env.store,
            config=load_price_product_config(),
            requested_listing_ids=(),
        )


def test_original_v1_intake_dataclass_shape_is_preserved(c2_closed_batch):
    from dataclasses import fields

    from stanstock.data.research_product import CapturedProductIntake

    assert [field.name for field in fields(CapturedProductIntake)] == [
        "asset",
        "core_listing_ids",
        "saved_listing_ids",
        "candidate_listing_ids",
        "core_symbols",
        "saved_symbols",
    ]
    assert c2_closed_batch.intake.product_version == "research-product-v2"


@pytest.mark.parametrize("c2_environment", [{"grade": "observed"}], indirect=True)
def test_exceptional_observed_reissue_requires_exact_revision_and_its_own_deadline(
    c2_snapshot, monkeypatch
):
    env = c2_snapshot
    monkeypatch.delenv("STANSTOCK_CODE_REVISION", raising=False)
    kwargs = dict(
        universe_snapshot=env.snapshot,
        decision_time=env.now,
        target_date=env.target,
        issued_on_time=True,
        provider="twelve_data",
        benchmark_subject="SPY",
        config_path=default_decision_support_config_path(),
        store=env.store,
    )
    with pytest.raises(ValueError, match="exact clean"):
        analyze_snapshot(**kwargs)
    assert not AnalysisRun.objects.exists()
    revision = "a" * 40
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", revision)
    monkeypatch.setattr(
        "stanstock.research.product_eligibility_pipeline.clean_git_revision", lambda _root: revision
    )
    env.clock["now"] = env.now + timedelta(days=3)
    kwargs["decision_time"] = env.clock["now"]
    with pytest.raises(ValueError, match="deadline"):
        analyze_snapshot(**kwargs)
    assert not AnalysisRun.objects.exists()

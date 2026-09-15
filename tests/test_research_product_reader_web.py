from __future__ import annotations

import json
import math
from copy import copy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from django.db import DatabaseError, transaction
from django.urls import reverse
from django.utils import timezone
from exchange_calendars import get_calendar

from stanstock.data.assets import AssetStore
from stanstock.data.live_us import _persist_catalog
from stanstock.data.models import DataAsset, UniverseMembership
from stanstock.data.providers import twelve_data
from stanstock.data.providers.contracts import StockCatalog
from stanstock.data.research_product_demo import execute_demo_product_refresh
from stanstock.portfolio.models import Portfolio, PortfolioHolding, TrackedSymbol
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.outcomes import evaluate_prediction
from stanstock.research.price_product_config import MOMENTUM_METHOD_VERSION
from stanstock.research.product_pipeline import (
    verify_price_product_output as real_product_verifier,
)
from stanstock.research.product_reader import (
    ProductCohort,
    ProductHistoryRead,
    ProductRead,
    read_research_product,
    read_research_product_history,
)
from stanstock.web import product_views
from test_research_product_jobs import (
    NOW,
    TARGET,
    _persist_price_series,
    _series,
    make_product_environment,
)

pytestmark = pytest.mark.django_db


@pytest.fixture
def live_product(tmp_path, monkeypatch, django_user_model, settings):
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = False
    environment = make_product_environment(tmp_path, monkeypatch, django_user_model)
    owner, store, path, _resolve, _fetch = environment
    settings.DATA_DIR = store.root
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    from stanstock.data.research_product_jobs import execute_daily_research_job

    job = execute_daily_research_job(
        target_date=TARGET,
        owner=owner,
        store=store,
        core_config_path=path,
        enforce_rate_limit=False,
    )
    run = AnalysisRun.objects.get(pk=job.details["analysis_run_id"])
    return owner, store, run


@pytest.fixture(params=(True, False))
def observed_product_history(tmp_path, monkeypatch, django_user_model, settings, request):
    """Three real verified issuances spanning one 126-session maturity."""

    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = False
    environment = make_product_environment(tmp_path, monkeypatch, django_user_model)
    owner, store, path, _resolve, _fetch = environment
    settings.DATA_DIR = store.root
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", "a" * 40)
    monkeypatch.setattr(
        "stanstock.research.product_pipeline.clean_git_revision",
        lambda _path: "a" * 40,
    )
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    from stanstock.data.research_product_jobs import execute_daily_research_job

    original_job = execute_daily_research_job(
        target_date=TARGET,
        owner=owner,
        issued_on_time=True,
        store=store,
        core_config_path=path,
        enforce_rate_limit=False,
    )
    original = AnalysisRun.objects.get(pk=original_job.details["analysis_run_id"])

    reissue_time = NOW + timedelta(minutes=5)
    monkeypatch.setattr(timezone, "now", lambda: reissue_time)
    reissue_job = execute_daily_research_job(
        target_date=TARGET,
        owner=owner,
        issuance_key="reader-history-reissue",
        issued_on_time=True,
        store=store,
        core_config_path=path,
        enforce_rate_limit=False,
    )
    reissue = AnalysisRun.objects.get(pk=reissue_job.details["analysis_run_id"])

    calendar = get_calendar("XNYS")
    original_session = calendar.date_to_session(TARGET, direction="none")
    current_target = calendar.sessions_window(original_session, 127)[-1].date()
    current_time = datetime(
        current_target.year,
        current_target.month,
        current_target.day,
        22,
        tzinfo=UTC,
    )
    monkeypatch.setattr(timezone, "now", lambda: current_time)
    for exchange in ("NASDAQ", "NYSE"):
        prior_catalog = DataAsset.objects.filter(
            provider="twelve_data",
            kind="stock_catalog",
            subject=exchange,
        ).latest("retrieved_at")
        raw_catalog = store.read_bytes(prior_catalog.relative_path)
        references, count = twelve_data.parse_stock_catalog_references(
            raw_catalog,
            exchange=exchange,
            require_complete=True,
        )
        _persist_catalog(
            store,
            StockCatalog(
                provider="twelve_data",
                exchange=exchange,
                references=references,
                count=count,
                retrieved_at=current_time,
                source_url=str(prior_catalog.metadata["source_url"]),
                raw_bytes=raw_catalog,
            ),
        )
    future_sessions = calendar.sessions_in_range(
        calendar.next_session(original_session),
        current_target,
    )
    for symbol in ("AAPL", "MSFT", "SPY", "CHEAP"):
        prior_raw = DataAsset.objects.filter(
            provider="twelve_data",
            kind="raw_price_history",
            subject=symbol,
        ).latest("retrieved_at")
        payload = json.loads(store.read_bytes(prior_raw.relative_path))
        previous_close = float(payload["values"][-1]["close"])
        for offset, session in enumerate(future_sessions, start=1):
            value = previous_close * math.exp(0.0002 * offset)
            payload["values"].append(
                {
                    "datetime": session.date().isoformat(),
                    "open": str(value),
                    "high": str(value),
                    "low": str(value),
                    "close": str(value),
                    "volume": "1000000",
                }
            )
        raw = json.dumps(payload).encode()
        series = twelve_data.parse_daily_price_series(
            raw,
            symbol=symbol,
            retrieved_at=current_time,
            source_url=str(prior_raw.metadata["source_url"]),
            end_date=current_target,
        )
        _persist_price_series(
            store=store,
            series=series,
            listing=None,
        )
    current_job = execute_daily_research_job(
        target_date=current_target,
        owner=owner,
        issued_on_time=request.param,
        store=store,
        core_config_path=path,
        enforce_rate_limit=False,
    )
    current = AnalysisRun.objects.get(pk=current_job.details["analysis_run_id"])

    evaluation_time = current_time + timedelta(minutes=1)
    monkeypatch.setattr(timezone, "now", lambda: evaluation_time)
    for run in (original, reissue):
        prediction = Prediction.objects.get(
            analysis__run=run,
            listing__provider_symbol="AAPL",
            method_version=MOMENTUM_METHOD_VERSION,
        )
        outcome = evaluate_prediction(
            prediction,
            provider="twelve_data",
            evaluation_date=current_target,
            evaluation_time=evaluation_time,
            benchmark_subject="SPY",
            store=store,
        ).outcome
        assert outcome.status == "matured"

    return owner, store, original, reissue, current


def test_native_history_shortfall_retains_its_recorded_admission_reason(
    tmp_path, monkeypatch, django_user_model, settings
):
    from stanstock.data.research_product_jobs import execute_daily_research_job

    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = False
    owner, store, path, resolve, fetch = make_product_environment(
        tmp_path, monkeypatch, django_user_model
    )
    resolve.side_effect = None
    resolve.return_value = "synthetic-test-token"
    fetch.side_effect = lambda symbol, **kwargs: _series(symbol, closes=100)
    job = execute_daily_research_job(
        target_date=TARGET,
        owner=owner,
        store=store,
        core_config_path=path,
        enforce_rate_limit=False,
    )
    assert job.status == "success"

    product = read_research_product(user=owner, store=store)

    assert product.available
    admission = next(item for item in product.admissions if item.symbol == "CHEAP")
    assert admission.status == "insufficient_history"
    assert admission.reason_code == "price_history_insufficient"
    assert admission.available_closes == 100
    assert admission.missing_closes == 657
    assert admission.bootstrap_attempted is True


def test_real_reader_verifies_once_and_projects_complete_recorded_result(live_product, monkeypatch):
    owner, store, run = live_product
    verifier = Mock(wraps=real_product_verifier)
    monkeypatch.setattr(
        "stanstock.research.product_reader.verify_price_product_output",
        verifier,
    )

    result = read_research_product(user=owner, store=store)

    assert result.status == "available"
    assert result.run == run
    assert all(
        admission.reason_code for admission in result.admissions if admission.status != "admitted"
    )
    assert {card.listing.provider_symbol for card in result.cards} == {
        "AAPL",
        "MSFT",
        "CHEAP",
    }
    assert all(
        tuple(projection.horizon for projection in card.projections) == ("6m", "12m", "3y", "5y")
        for card in result.cards
    )
    assert all(projection.available for card in result.cards for projection in card.projections)
    assert all(
        card.analysis.overall_score is None
        and card.analysis.confidence is None
        and card.analysis.risk_score is None
        for card in result.cards
    )
    verifier.assert_called_once_with(run=run, store=store, replay=False)


def test_native_four_layer_live_job_reader_and_authenticated_pages(
    live_product, client, monkeypatch
):
    owner, _store, run = live_product
    client.force_login(owner)
    no_calculation = Mock(side_effect=AssertionError("GET must not calculate product output"))
    no_simulation = Mock(side_effect=AssertionError("GET must not run FHS paths"))
    no_provider_fetch = Mock(side_effect=AssertionError("GET must not fetch provider data"))
    verifier = Mock(wraps=real_product_verifier)
    monkeypatch.setattr(
        "stanstock.research.product_reader.verify_price_product_output",
        verifier,
    )
    monkeypatch.setattr(
        "stanstock.research.product_pipeline.calculate_price_product",
        no_calculation,
    )
    monkeypatch.setattr(
        "stanstock.research.price_product.simulate_fhs_terminal_logs",
        no_simulation,
    )
    monkeypatch.setattr(
        "stanstock.data.providers.twelve_data.fetch_daily_price_series",
        no_provider_fetch,
    )

    opportunities = client.get(reverse("opportunities"))
    assert opportunities.status_code == 200
    content = opportunities.content.decode()
    assert "<h1>Opportunities</h1>" in content
    assert "CHEAP" in content
    assert "Under $10 watch" in content
    assert "0% new allocation" in content
    assert "Median return" in content
    assert all(label in content for label in ("Loss", "Flat to +20%", "Above +20%"))
    assert "Shares of model simulations; not validated real-world odds." in content
    assert all(label in content for label in ("6 months", "12 months", "3 years", "5 years"))
    assert "/100" not in content
    assert "Heuristic evidence score" not in content
    assert "Probability of gain" not in content
    assert verifier.call_count == 1
    cheap = StockAnalysis.objects.get(run=run, listing__provider_symbol="CHEAP")
    assert cheap.recommendation == "hold"
    cheap_section = content.split("ZZRP", maxsplit=1)[0]
    assert "CHEAP" in cheap_section

    immutable_counts = (
        AnalysisRun.objects.count(),
        Prediction.objects.count(),
        DataAsset.objects.count(),
    )
    under_ten = client.get(reverse("opportunities"), {"price_band": "under_10"})
    invalid = client.get(reverse("opportunities"), {"horizon": "tomorrow"})
    assert under_ten.status_code == 200
    assert "CHEAP" in under_ten.content.decode()
    assert "Speculative watch only · 0% new allocation." in under_ten.content.decode()
    assert invalid.status_code == 200
    assert "Filters need attention." in invalid.content.decode()
    assert immutable_counts == (
        AnalysisRun.objects.count(),
        Prediction.objects.count(),
        DataAsset.objects.count(),
    )
    assert verifier.call_count == 3

    detail = client.get(reverse("stock-detail", args=[cheap.listing_id]))
    assert detail.status_code == 200
    detail_content = detail.content.decode()
    assert "Same-shock zero-log-drift sensitivity" in detail_content
    assert "T−252 through T−21" in detail_content
    assert "Forecasting skill is not established" in detail_content
    assert "8,192 deterministic PCG64 paths" in detail_content
    assert "Reference close" in detail_content
    assert "Market date" in detail_content
    assert "Provider" in detail_content
    assert '<details class="panel source-provenance-details">' in detail_content
    assert "Price coverage and retrieval details" in detail_content
    assert "Retrieved" in detail_content
    assert all(
        f"<h3>{label}</h3>" in detail_content
        for label in ("6 months", "12 months", "3 years", "5 years")
    )
    assert verifier.call_count == 4
    no_calculation.assert_not_called()
    no_simulation.assert_not_called()
    no_provider_fetch.assert_not_called()


@pytest.mark.parametrize("read_time", (NOW, NOW + timedelta(days=370)))
def test_demo_refresh_uses_same_reader_and_primary_rendering(
    settings, tmp_path, django_user_model, client, monkeypatch, read_time
):
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = True
    settings.DATA_DIR = tmp_path
    monkeypatch.setattr(timezone, "now", lambda: NOW)
    execute_demo_product_refresh(store=AssetStore(tmp_path))
    viewer = django_user_model.objects.create_user(username="demo-viewer")
    before = (DataAsset.objects.count(), Prediction.objects.count())
    monkeypatch.setattr(timezone, "now", lambda: read_time)
    client.force_login(viewer)

    result = read_research_product(user=viewer, store=AssetStore(tmp_path))
    response = client.get(reverse("opportunities"))
    history = read_research_product_history(user=viewer, store=AssetStore(tmp_path))

    assert result.status == "available"
    assert history.current.available
    assert history.current.run == result.run
    assert (DataAsset.objects.count(), Prediction.objects.count()) == before
    assert result.owner_id == "synthetic-demo"
    assert {card.listing.ticker for card in result.cards} == {
        "ZZRPUP",
        "ZZRPDOWN",
        "ZZRPLOW",
    }
    assert response.status_code == 200
    content = response.content.decode()
    assert "Synthetic research data." in content
    assert "ZZRPLOW" in content
    assert "100 of 757 closes" in content
    assert "missing 657" in content


def test_unauthenticated_and_wrong_owner_never_receive_private_product(
    live_product, client, django_user_model
):
    owner, store, _run = live_product
    response = client.get(reverse("opportunities"))
    assert response.status_code == 302

    other = django_user_model.objects.create_user(username="other-viewer")
    assert read_research_product(user=other, store=store).status == "absent"
    client.force_login(other)
    response = client.get(reverse("opportunities"))
    assert response.status_code == 200
    assert owner.username.encode() not in response.content
    assert b"CHEAP" not in response.content
    assert b"No verified active research-product cohort" in response.content


def test_absent_product_does_not_substitute_legacy_analysis(
    settings, authenticated_client, persisted_analysis, tmp_path
):
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = False
    settings.DATA_DIR = tmp_path

    response = authenticated_client.get(reverse("opportunities"))

    assert response.status_code == 200
    assert b"Active research unavailable" in response.content
    assert persisted_analysis.listing.ticker.encode() not in response.content
    assert b"Archived score-based research" in response.content


def test_corrupt_physical_source_suppresses_all_active_output(live_product):
    owner, store, _run = live_product
    source = DataAsset.objects.get(
        provider="twelve_data",
        kind="price_history",
        subject="AAPL",
    )
    store.resolve(source.relative_path).write_bytes(b"corrupt")

    result = read_research_product(user=owner, store=store)

    assert result.status == "integrity_failed"
    assert result.cards == ()
    assert result.verification_code


def test_stale_target_is_not_substituted_or_promoted(live_product, monkeypatch):
    owner, store, _run = live_product
    monkeypatch.setattr(
        "stanstock.research.product_reader.timezone.now",
        lambda: NOW + timedelta(days=5),
    )

    result = read_research_product(user=owner, store=store)

    assert result.status == "stale"
    assert result.cards == ()
    assert result.verification_code == "product_target_stale"


def test_stale_active_target_remains_visible_only_as_dated_history(
    live_product,
    client,
    monkeypatch,
):
    owner, _store, run = live_product
    monkeypatch.setattr(
        "stanstock.research.product_reader.timezone.now",
        lambda: NOW + timedelta(days=5),
    )
    client.force_login(owner)

    response = client.get(reverse("predictions"))

    assert response.status_code == 200
    assert response.context["history"].status == "available"
    assert response.context["product"].status == "stale"
    assert str(run.id).encode() in response.content
    assert b"historical evidence, not a current signal" in response.content
    assert b"Product ledger unavailable" not in response.content


def test_forged_membership_and_joined_row_tampering_fail_closed(live_product):
    owner, store, run = live_product
    membership = UniverseMembership.objects.filter(snapshot=run.universe_snapshot).first()
    assert membership is not None
    UniverseMembership.objects.filter(pk=membership.pk).update(eligible=False)

    membership_result = read_research_product(user=owner, store=store)

    assert membership_result.status == "integrity_failed"
    assert membership_result.cards == ()

    UniverseMembership.objects.filter(pk=membership.pk).update(eligible=True)
    analysis = StockAnalysis.objects.filter(run=run).first()
    prediction = Prediction.objects.filter(analysis__run=run).first()
    assert analysis is not None and prediction is not None
    StockAnalysis.objects.filter(pk=analysis.pk).update(reasons=["forged"])
    with transaction.atomic():
        with pytest.raises(DatabaseError, match="immutable"):
            Prediction.objects.filter(pk=prediction.pk).update(insufficiency_reason="forged")

    row_result = read_research_product(user=owner, store=store)

    assert row_result.status == "integrity_failed"
    assert row_result.cards == ()


def test_status_history_performance_and_my_list_keep_boundaries(live_product, client):
    owner, _store, _run = live_product
    client.force_login(owner)

    status = client.get(reverse("status"))
    history = client.get(reverse("predictions"))
    performance = client.get(reverse("performance"))
    my_list = client.get(reverse("my-list"))

    assert status.status_code == 200
    assert b"READY" in status.content
    assert b"analysed /" in status.content
    assert b"not admitted" in status.content
    assert b"Lower (p20)" not in status.content
    assert "Forecast values belong on Opportunities" in " ".join(status.content.decode().split())
    assert b"Data &amp; updates" in status.content
    assert b"Listing entries" in history.content
    assert b"Four advisory projections" in history.content
    assert b"Forecasting skill is not established" in performance.content
    assert b"Better/worse refers to paired error" in performance.content
    assert b"Not eligible for observed track record" in performance.content
    assert b"No canonical observed product outcome has matured yet" not in performance.content
    assert performance.context["observed_scope"]["decision"]["not_eligible_for_observed"] == 3
    assert performance.context["observed_scope"]["decision"]["outcome_pending"] == 0
    assert b"Research state" in my_list.content
    assert b"next scheduled refresh" in my_list.content


def test_native_selected_horizon_flows_from_search_to_detail_my_list_and_history(
    live_product, client
):
    """Native registered evidence, rather than a DTO mock, drives all links."""

    owner, _store, run = live_product
    client.force_login(owner)
    cheap = StockAnalysis.objects.get(run=run, listing__provider_symbol="CHEAP")

    opportunities = client.get(
        reverse("opportunities"),
        {"q": "CHEAP", "horizon": "3y", "price_band": "under_10"},
    )
    assert opportunities.status_code == 200
    assert b"3 years projections" in opportunities.content
    detail_url = f"{reverse('stock-detail', args=[cheap.listing_id])}?horizon=3y"
    assert detail_url.encode() in opportunities.content

    detail = client.get(detail_url)
    assert detail.status_code == 200
    detail_content = detail.content.decode()
    assert 'class="projection-card is-selected-horizon"' in detail_content
    assert detail_content.index("Median return:") < detail_content.index("Lower (p20)")
    assert "Loss (R &lt; 0)" in detail_content
    assert "Flat to +20% (0 ≤ R ≤ +20%)" in detail_content
    assert "Above +20% (R &gt; +20%)" in detail_content
    assert "Ranges, exact path counts, and sensitivity" in detail_content

    my_list = client.get(reverse("my-list"), {"horizon": "3y"})
    assert my_list.status_code == 200
    my_list_content = my_list.content.decode()
    assert "3 years median:" in my_list_content
    assert detail_url in my_list_content
    assert "Flat to +20% (0% to +20%)" in my_list_content
    invalid_my_list = client.get(reverse("my-list"), {"horizon": "tomorrow"})
    assert invalid_my_list.status_code == 200
    assert b"Choose one of the available product horizons." in invalid_my_list.content

    history = client.get(reverse("predictions"), {"horizon": "3y"})
    assert history.status_code == 200
    history_content = history.content.decode()
    assert "3 years model-estimated probabilities" in history_content
    assert detail_url in history_content
    assert "Median return:" in history_content
    assert "Flat to +20% (0 ≤ R ≤ +20%)" in history_content
    invalid_history = client.get(reverse("predictions"), {"horizon": "tomorrow"})
    assert invalid_history.status_code == 200
    assert b"Choose one of the available product horizons." in invalid_history.content


def test_verified_history_and_performance_span_runs_without_recounting_reissue(
    observed_product_history,
    client,
    django_user_model,
    monkeypatch,
):
    owner, store, original, reissue, current = observed_product_history
    verifier = Mock(wraps=real_product_verifier)
    no_simulation = Mock(side_effect=AssertionError("GET must not run FHS paths"))
    monkeypatch.setattr(
        "stanstock.research.product_reader.verify_price_product_output",
        verifier,
    )
    monkeypatch.setattr(
        "stanstock.research.price_product.simulate_fhs_terminal_logs",
        no_simulation,
    )
    client.force_login(owner)

    performance = client.get(reverse("performance"))

    assert performance.status_code == 200
    assert verifier.call_count == 3
    assert performance.context["history"].current.run == current
    assert performance.context["decision_groups"] == [{"status": "matured", "count": 1}]
    observed_scope = performance.context["observed_scope"]["decision"]
    assert observed_scope["matured"] == 1
    if current.issued_on_time:
        assert observed_scope["outcome_pending"] > 0
    else:
        assert observed_scope["not_eligible_for_observed"] == 3
    assert b"No canonical observed product outcome has matured yet" not in performance.content
    no_simulation.assert_not_called()

    verifier.reset_mock()
    history = client.get(reverse("predictions"))

    assert history.status_code == 200
    assert verifier.call_count == 3
    history_content = history.content.decode()
    assert str(original.id) in history_content
    assert str(reissue.id) in history_content
    assert str(current.id) in history_content
    assert {card.decision_prediction.target_date for card in history.context["decision_cards"]} == {
        TARGET,
        current.target_date,
    }
    assert history_content.count("Matured") >= 2
    assert "historical evidence, not a current signal" in history_content
    assert "Observed · Twelve Data" in history_content
    if not current.issued_on_time:
        assert "Research · Twelve Data" in history_content
    no_simulation.assert_not_called()

    other = django_user_model.objects.create_user(username="history-other-owner")
    assert read_research_product_history(user=other, store=store).status == "absent"

    original_source_id = Prediction.objects.get(
        analysis__run=original,
        listing__provider_symbol="AAPL",
        method_version=MOMENTUM_METHOD_VERSION,
    ).source_assets[0]["id"]
    original_source = DataAsset.objects.get(pk=original_source_id)
    store.resolve(original_source.relative_path).write_bytes(b"corrupt historical source")

    assert read_research_product(user=owner, store=store).status == "available"
    corrupted_history = read_research_product_history(user=owner, store=store)
    assert corrupted_history.status == "integrity_failed"
    assert corrupted_history.cohorts == ()


def test_my_list_add_remove_actions_remain_owner_scoped_and_csrf_posted(live_product, client):
    owner, _store, _run = live_product
    client.force_login(owner)
    preference = TrackedSymbol.objects.get(owner=owner, symbol="CHEAP")

    removed = client.post(
        reverse("tracked-symbol-delete", args=[preference.id]),
        {"horizon": "5y"},
    )
    added = client.post(reverse("my-list"), {"symbol": "CHEAP", "horizon": "5y"})

    assert removed.status_code == 302
    assert added.status_code == 302
    assert removed.url == f"{reverse('my-list')}?horizon=5y"
    assert added.url == f"{reverse('my-list')}?horizon=5y"
    assert TrackedSymbol.objects.filter(owner=owner, symbol="CHEAP").count() == 1


def test_disabled_flag_keeps_explicit_archive_behavior(
    settings, authenticated_client, persisted_analysis
):
    settings.RESEARCH_PRODUCT_ENABLED = False

    response = authenticated_client.get(reverse("opportunities"))

    assert response.status_code == 200
    assert persisted_analysis.listing.ticker.encode() in response.content
    assert b"Rollback/archive mode" in response.content


def test_archive_and_disabled_rollback_do_not_reinterpret_product_rows(
    live_product, client, settings
):
    owner, _store, _run = live_product
    client.force_login(owner)

    archive = client.get(reverse("archive-opportunities"))
    settings.RESEARCH_PRODUCT_ENABLED = False
    rollback = client.get(reverse("opportunities"))

    assert archive.status_code == 200
    assert rollback.status_code == 200
    assert b"CHEAP" not in archive.content
    assert b"CHEAP" not in rollback.content
    assert b"No pre-product archived analysis is stored" in archive.content
    assert b"Archived research" in rollback.content


def _synthetic_presentation_read(
    *,
    owner,
    store: AssetStore,
    card_count: int = 105,
) -> ProductRead:
    """Repeat one verified card only to exercise the bounded presentation adapter."""

    verified = read_research_product(user=owner, store=store)
    assert verified.available
    under_ten = next(card for card in verified.cards if card.target_under_10)
    return replace(verified, cards=(under_ten,) * card_count)


def test_root_navigation_filters_and_pagination_use_a_compact_synthetic_adapter(
    live_product,
    client,
    monkeypatch,
    settings,
):
    """105 repeated synthetic presentation entries must not require 105 product runs."""

    owner, store, _run = live_product
    client.force_login(owner)
    presentation = _synthetic_presentation_read(owner=owner, store=store)

    def read_presentation(request):
        request._stanstock_product_read = presentation
        return presentation

    monkeypatch.setattr(product_views, "_read", read_presentation)
    landing = client.get(reverse("index"))
    regular = client.get(reverse("opportunities"))
    under_ten = client.get(
        reverse("opportunities"),
        {"price_band": "under_10", "horizon": "12m"},
    )
    submitted_filters = {
        "q": "CHEAP",
        "horizon": "12m",
        "direction": presentation.cards[0].direction,
        "risk": presentation.cards[0].relative_volatility_label,
        "price_band": "under_10",
    }
    submitted = client.get(reverse("opportunities"), submitted_filters)
    invalid = client.get(reverse("opportunities"), {"horizon": "tomorrow"})
    market = client.get(reverse("market"))

    assert landing.status_code == 302
    assert landing.url == reverse("opportunities")
    assert regular.status_code == 200
    assert regular.content.count(b'class="compact-opportunity"') == 20
    assert regular.context["opportunity_page"].paginator.count == 105
    assert regular.content.count(b"Loss") == 20
    assert b'href="/opportunities?price_band=under_10"' in regular.content
    assert b'aria-current="page">Opportunities</a>' in regular.content
    assert b"<summary>More</summary>" in regular.content
    assert b"Data &amp; updates" in regular.content
    assert b"Under $10 watch" in regular.content
    # 20 of a deliberately synthetic 105-entry cohort is an 80% rendered
    # listing reduction from the former expanded all-entry presentation.
    assert 1 - (20 / len(presentation.cards)) >= 0.60
    assert b'aria-current="page">Under $10</a>' in under_ten.content
    assert b"12 months projections" in under_ten.content
    assert b"horizon=12m&amp;price_band=under_10&amp;page=2" in under_ten.content
    assert submitted.status_code == 200
    for field, value in submitted_filters.items():
        assert submitted.context["filter_form"].cleaned_data[field] == value
    assert submitted.content.count(b'class="compact-opportunity"') == 20
    assert b'type="hidden" name="price_band" value="under_10"' in submitted.content
    submitted_query = submitted.context["pagination_query"]
    assert "q=CHEAP" in submitted_query
    assert "horizon=12m" in submitted_query
    assert f"direction={presentation.cards[0].direction}" in submitted_query
    assert f"risk={presentation.cards[0].relative_volatility_label}" in submitted_query
    assert "price_band=under_10" in submitted_query
    assert f"{submitted_query.replace('&', '&amp;')}&amp;page=2".encode() in submitted.content
    assert invalid.status_code == 200
    assert b"Filters need attention" in invalid.content
    assert invalid.content.count(b'class="compact-opportunity"') == 0
    assert market.status_code == 200
    assert b'<details class="more-nav">' in market.content
    assert b'<details class="more-nav" open>' not in market.content
    assert b'href="/market" aria-current="page">Market</a>' in market.content
    assert b'href="/opportunities?price_band=under_10"' in market.content

    settings.RESEARCH_PRODUCT_ENABLED = False
    legacy_landing = client.get(reverse("index"))
    assert legacy_landing.status_code == 302
    assert legacy_landing.url == reverse("status")


def test_native_portfolio_momentum_copy_has_no_fabricated_score_or_sample_builder(
    live_product,
    client,
):
    owner, _store, run = live_product
    client.force_login(owner)
    listing = run.stocks.order_by("listing__ticker").first().listing
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Native research holding",
        base_currency="USD",
        cash_balance=0,
    )
    PortfolioHolding.objects.create(
        portfolio=portfolio,
        listing=listing,
        quantity=1,
        average_cost=listing.latest_market_data.close,
    )

    detail = client.get(reverse("portfolio-detail", args=[portfolio.id]))
    portfolios = client.get(reverse("portfolios"))

    assert detail.status_code == 200
    assert b"No overall score" in detail.content
    assert b"6-month momentum method" in detail.content
    assert b"Action is not repeated here." in detail.content
    assert b"Recorded action:" not in detail.content
    assert b"None/100" not in detail.content
    assert b"Check verified current research" in detail.content
    assert portfolios.status_code == 200
    assert b"Sample builder unavailable for the active momentum method" in portfolios.content
    assert b'name="action" value="sample"' not in portfolios.content


def test_native_momentum_portfolio_with_null_action_never_renders_literal_none(
    live_product,
    client,
):
    """A contract-valid unavailable active-method action remains method-only here."""

    owner, _store, run = live_product
    client.force_login(owner)
    source_analysis = run.stocks.order_by("listing__ticker").first()
    assert source_analysis is not None
    portfolio = Portfolio.objects.create(
        owner=owner,
        name="Unavailable momentum holding",
        base_currency="USD",
        cash_balance=0,
    )
    PortfolioHolding.objects.create(
        portfolio=portfolio,
        listing=source_analysis.listing,
        quantity=1,
        average_cost=source_analysis.current_price,
    )
    unavailable_run = AnalysisRun.objects.create(
        generated_at=run.generated_at + timedelta(seconds=1),
        data_cutoff=run.data_cutoff,
        target_date=run.target_date,
        universe_snapshot=run.universe_snapshot,
        config_version=run.config_version,
        config_hash=run.config_hash,
        code_revision=run.code_revision,
    )
    unavailable_analysis = StockAnalysis(
        run=unavailable_run,
        listing=source_analysis.listing,
        current_price=source_analysis.current_price,
        daily_change=source_analysis.daily_change,
        overall_score=None,
        recommendation=None,
        risk_score=None,
        risk_class=source_analysis.risk_class,
        confidence=None,
        confidence_status="not_estimated",
        data_quality={"momentum_insufficiency_reason": "synthetic_unavailable_action"},
    )
    unavailable_analysis.clean()
    unavailable_analysis.save()

    detail = client.get(reverse("portfolio-detail", args=[portfolio.id]))

    assert detail.status_code == 200
    content = detail.content.decode()
    assert "No overall score — this is the 6-month momentum method." in content
    assert "Action is not repeated here." in content
    assert "Check verified current research" in content
    assert "Recorded action: None" not in content
    assert "None/100" not in content


@pytest.mark.parametrize(
    ("reader_status", "message", "verification_code"),
    (
        ("stale", "The recorded cohort is stale.", "product_target_stale"),
        ("unauthorized", "Provider display authorization is unavailable.", ""),
        ("absent", "No verified cohort is available.", ""),
        (
            "disabled",
            "The primary research product is disabled; archived evidence remains available.",
            "",
        ),
        (
            "integrity_failed",
            "Active research output was suppressed because evidence could not be verified.",
            "product_reader_shape_invalid",
        ),
    ),
)
def test_my_list_unavailable_reader_states_are_not_presented_as_refresh_pending(
    live_product,
    client,
    monkeypatch,
    settings,
    reader_status,
    message,
    verification_code,
):
    """Unavailable reader states retain their recorded cause rather than a time claim."""

    owner, _store, _run = live_product
    client.force_login(owner)
    TrackedSymbol.objects.create(owner=owner, symbol="UNAVAILABLE")
    product = ProductRead(
        status=reader_status,
        message=message,
        verification_code=verification_code,
    )
    monkeypatch.setattr(product_views, "_read", lambda _request: product)
    if reader_status == "disabled":
        settings.RESEARCH_PRODUCT_ENABLED = False
        monkeypatch.setattr(product_views, "_has_product_output", lambda: True)

    response = client.get(reverse("my-list"))

    assert response.status_code == 200
    content = response.content.decode()
    assert "Research source unavailable." in content
    assert message in content
    assert "Checked at the next scheduled refresh." not in content
    if verification_code:
        assert "Verification state:" in content
    expected_state = (
        "Source failed" if reader_status == "integrity_failed" else "Source unavailable"
    )
    assert expected_state in content


def test_my_list_marks_only_an_available_unadmitted_symbol_as_refresh_pending(live_product):
    owner, _store, _run = live_product
    preference = TrackedSymbol.objects.create(owner=owner, symbol="PENDING")
    available = ProductRead(status="available", message="Verified product available.")

    item = product_views._my_list_item(
        preference=preference,
        admission=None,
        card=None,
        product=available,
    )

    assert item["state"] == "Pending"
    assert item["state_detail"] == "Checked at the next scheduled refresh."


def test_history_marks_a_synthetic_all_null_advisory_as_not_evaluable(
    live_product,
    client,
    monkeypatch,
):
    """Presentation-only adversarial row: no persisted immutable row is altered."""

    owner, store, _run = live_product
    client.force_login(owner)
    current = _synthetic_presentation_read(owner=owner, store=store, card_count=1)
    card = current.cards[0]
    original = card.advisory_predictions[0]
    withheld = copy(original)
    withheld.bear_return = None
    withheld.base_return = None
    withheld.bull_return = None
    withheld_projection = replace(
        card.projections[0],
        lower_return=None,
        median_return=None,
        upper_return=None,
        lower_price=None,
        median_price=None,
        upper_price=None,
        insufficiency_reason="forecast_withheld",
    )
    adapted_card = replace(
        card,
        advisory_predictions=(withheld, *card.advisory_predictions[1:]),
        projections=(withheld_projection, *card.projections[1:]),
    )
    cohort = ProductCohort(
        run=current.run,
        cards=(adapted_card,),
        admissions=current.admissions,
        provider=current.provider,
        evidence_grade=current.evidence_grade,
        owner_id=current.owner_id,
    )
    history = ProductHistoryRead(
        status="available",
        message="Synthetic presentation history",
        current=current,
        cohorts=(cohort,),
    )

    def read_history(request):
        request._stanstock_product_read = current
        return history

    monkeypatch.setattr(product_views, "_read_history", read_history)
    response = client.get(reverse("predictions"))

    assert response.status_code == 200
    assert b"Not evaluable" in response.content
    assert b"forecast withheld" in response.content
    assert b"Not matured" not in response.content


@pytest.fixture
def chromium_browser():
    """Return the Playwright API without starting a driver during collection."""

    return pytest.importorskip(
        "playwright.sync_api",
        reason="Playwright is not installed",
    )


@pytest.mark.parametrize("viewport", ((375, 812), (1440, 900)))
def test_synthetic_compact_opportunities_are_visible_and_do_not_overflow(
    live_product,
    client,
    chromium_browser,
    monkeypatch,
    viewport: tuple[int, int],
):
    """Use the existing browser only for synthetic layout acceptance evidence."""

    owner, store, _run = live_product
    client.force_login(owner)
    presentation = _synthetic_presentation_read(owner=owner, store=store)

    def read_presentation(request):
        request._stanstock_product_read = presentation
        return presentation

    monkeypatch.setattr(product_views, "_read", read_presentation)
    response = client.get(reverse("opportunities"))
    searched_response = client.get(
        reverse("opportunities"),
        {"q": "CHEAP", "horizon": "12m", "price_band": "under_10"},
    )
    market_response = client.get(reverse("market"))
    assert response.status_code == 200
    assert searched_response.status_code == 200
    assert market_response.status_code == 200

    css_path = Path(__file__).resolve().parents[1] / "static" / "css" / "stanstock.css"
    with chromium_browser.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
            try:
                page.set_content(response.content.decode())
                page.add_style_tag(path=str(css_path))
                assert page.locator(".compact-opportunity").count() == 20
                assert page.locator(".compact-projection").count() == 20
                note = page.locator(".opportunity-scenario-note")
                assert note.is_visible()
                note_text = note.inner_text()
                assert "Model-estimated probabilities:" in note_text
                assert "Shares of model simulations; not validated real-world odds." in note_text
                assert "Advisory only." in note_text
                assert all(
                    projection.frequency_status == "available"
                    for card in presentation.cards
                    for projection in card.projections
                )
                assert page.locator(".probability-cell").first.inner_text().startswith("Loss")
                assert "Flat to +20%" in page.locator(".compact-projection").first.inner_text()
                assert "Above +20%" in page.locator(".compact-projection").first.inner_text()
                assert page.evaluate("document.documentElement.scrollWidth") <= page.evaluate(
                    "document.documentElement.clientWidth"
                )
                for selector in (".compact-opportunity-facts", ".compact-projection"):
                    widths = page.locator(selector).evaluate_all(
                        "(elements) => elements.map((element) => "
                        "[element.scrollWidth, element.clientWidth])"
                    )
                    assert all(
                        scroll_width <= client_width for scroll_width, client_width in widths
                    )

                opportunity_nav = page.get_by_role("link", name="Opportunities", exact=True)
                nav_metrics = opportunity_nav.evaluate(
                    "(element) => ({"
                    "scrollWidth: element.scrollWidth, "
                    "clientWidth: element.clientWidth, "
                    "whiteSpace: getComputedStyle(element).whiteSpace, "
                    "fontSize: parseFloat(getComputedStyle(element).fontSize), "
                    "height: element.getBoundingClientRect().height"
                    "})"
                )
                assert nav_metrics["scrollWidth"] <= nav_metrics["clientWidth"]
                assert nav_metrics["whiteSpace"] == "nowrap"
                assert page.locator(".brand").get_attribute("aria-label") == "StanStock home"
                primary_nav_metrics = page.locator(".primary-nav-links a").evaluate_all(
                    "(elements) => elements.map((element) => ({"
                    "fontSize: parseFloat(getComputedStyle(element).fontSize), "
                    "height: element.getBoundingClientRect().height, "
                    "top: element.getBoundingClientRect().top"
                    "}))"
                )
                compact_horizons = page.locator(".horizon-links a")
                assert compact_horizons.all_inner_texts() == ["6m", "12m", "3y", "5y"]
                assert compact_horizons.evaluate_all(
                    "(elements) => elements.map((element) => element.getAttribute('aria-label'))"
                ) == [
                    "6 months projections",
                    "12 months projections",
                    "3 years projections",
                    "5 years projections",
                ]
                if viewport[0] == 375:
                    assert all(metric["fontSize"] >= 14 for metric in primary_nav_metrics)
                    assert all(metric["height"] >= 32 for metric in primary_nav_metrics)
                    assert len({metric["top"] for metric in primary_nav_metrics}) <= 2
                else:
                    search_widths = page.locator(
                        ".product-search-field input, .product-search .primary-button"
                    ).evaluate_all(
                        "(elements) => elements.map((element) => "
                        "element.getBoundingClientRect().width)"
                    )
                    assert search_widths[1] < search_widths[0]
                    assert search_widths[1] <= 120
                assert (
                    page.locator("#opportunity-list-title").inner_text() == "6 months projections"
                )

                first_projection = page.locator(".compact-projection").first
                projection_boxes = first_projection.locator("div").evaluate_all(
                    "(elements) => elements.map((element) => {"
                    "const box = element.getBoundingClientRect(); "
                    "return {top: box.top, scrollWidth: element.scrollWidth, "
                    "clientWidth: element.clientWidth};"
                    "})"
                )
                assert len(projection_boxes) == 3
                assert (
                    max(item["top"] for item in projection_boxes)
                    - min(item["top"] for item in projection_boxes)
                    <= 1
                )
                assert all(item["scrollWidth"] <= item["clientWidth"] for item in projection_boxes)

                under_ten = page.get_by_role("link", name="Under $10").first
                assert under_ten.bounding_box()["y"] < viewport[1]
                card_boxes = page.locator(".compact-opportunity").evaluate_all(
                    "(elements) => elements.map((element) => {"
                    "const box = element.getBoundingClientRect(); "
                    "return {top: box.top, bottom: box.bottom};"
                    "})"
                )
                complete_cards = [
                    box for box in card_boxes if box["top"] >= 0 and box["bottom"] <= viewport[1]
                ]
                if viewport[0] == 375:
                    assert card_boxes[0]["top"] >= 0
                    assert card_boxes[0]["top"] <= 600
                    assert card_boxes[0]["bottom"] - card_boxes[0]["top"] <= 250
                    layout_boxes = page.locator(
                        ".site-header, .opportunities-heading, .opportunity-controls, "
                        ".opportunity-results-heading, .opportunity-scenario-note"
                    ).evaluate_all(
                        "(elements) => elements.map((element) => {"
                        "const box = element.getBoundingClientRect(); "
                        "return {className: element.className, top: box.top, bottom: box.bottom};"
                        "})"
                    )
                    assert card_boxes[0]["bottom"] <= viewport[1], (
                        f"first_card={card_boxes[0]}; layout={layout_boxes}"
                    )
                else:
                    assert card_boxes[0]["top"] <= 500
                    assert card_boxes[0]["bottom"] - card_boxes[0]["top"] <= 140
                    assert len(complete_cards) >= 3

                page.locator(".advanced-filters summary").focus()
                page.keyboard.press("Enter")
                assert page.locator(".advanced-filters").get_attribute("open") == ""

                # This uses the real GET submission adapter and registered
                # frequencies, then checks the rendered selected-horizon
                # result in Chromium rather than only a context DTO.
                page.set_content(searched_response.content.decode())
                page.add_style_tag(path=str(css_path))
                assert (
                    page.locator("#opportunity-list-title").inner_text() == "12 months projections"
                )
                assert page.locator('input[name="horizon"]').input_value() == "12m"
                assert page.locator('input[name="price_band"]').input_value() == "under_10"
                assert (
                    page.locator(".compact-opportunity a")
                    .first.get_attribute("href")
                    .endswith("?horizon=12m")
                )
            finally:
                page.close()

            market_page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
            try:
                market_page.set_content(market_response.content.decode())
                market_page.add_style_tag(path=str(css_path))
                assert market_page.locator(".more-nav").get_attribute("open") is None
            finally:
                market_page.close()
        finally:
            browser.close()

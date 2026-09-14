from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
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
from stanstock.portfolio.models import TrackedSymbol
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.outcomes import evaluate_prediction
from stanstock.research.price_product_config import MOMENTUM_METHOD_VERSION
from stanstock.research.product_pipeline import (
    verify_price_product_output as real_product_verifier,
)
from stanstock.research.product_reader import (
    read_research_product,
    read_research_product_history,
)
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

    opportunities = client.get(reverse("opportunities"))
    assert opportunities.status_code == 200
    content = opportunities.content.decode()
    assert "Research opportunities" in content
    assert "CHEAP" in content
    assert "Under $10 speculative research" in content
    assert "0% new allocation" in content
    assert "Lower (p20)" in content
    assert "Median (p50)" in content
    assert "Upper (p80)" in content
    assert all(label in content for label in ("6 months", "12 months", "3 years", "5 years"))
    assert "/100" not in content
    assert "Heuristic evidence score" not in content
    assert "Probability of gain" not in content
    assert verifier.call_count == 1
    cheap = StockAnalysis.objects.get(run=run, listing__provider_symbol="CHEAP")
    assert cheap.recommendation == "hold"
    cheap_section = content.split("ZZRP", maxsplit=1)[0]
    assert "CHEAP" in cheap_section

    detail = client.get(reverse("stock-detail", args=[cheap.listing_id]))
    assert detail.status_code == 200
    detail_content = detail.content.decode()
    assert "Same-shock zero-log-drift sensitivity" in detail_content
    assert "T−252 through T−21" in detail_content
    assert "Forecasting skill is not established" in detail_content
    assert "8,192 deterministic PCG64 paths" in detail_content
    assert verifier.call_count == 2
    no_calculation.assert_not_called()
    no_simulation.assert_not_called()


def test_demo_refresh_uses_same_reader_and_primary_rendering(
    settings, tmp_path, django_user_model, client
):
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = True
    settings.DATA_DIR = tmp_path
    execute_demo_product_refresh(store=AssetStore(tmp_path))
    viewer = django_user_model.objects.create_user(username="demo-viewer")
    client.force_login(viewer)

    result = read_research_product(user=viewer, store=AssetStore(tmp_path))
    response = client.get(reverse("opportunities"))

    assert result.status == "available"
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
    assert b"OPERATIONAL" in status.content
    assert b"Lower (p20)" not in status.content
    assert b"Forecast values belong on the Research screen" in status.content
    assert b"126-session relative-momentum decisions" in history.content
    assert b"FHS projection ledger" in history.content
    assert b"Forecasting skill is not established" in performance.content
    assert b"No canonical observed product outcome has matured yet" in performance.content
    assert b"Captured research state" in my_list.content
    assert b"changes next intake" not in my_list.content.lower()
    assert b"does not mutate a captured historical cohort" in my_list.content


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
    advisory_content = history_content.split('aria-label="FHS advisory ledger"', 1)[1]
    assert "Observed · Twelve Data" in advisory_content
    if not current.issued_on_time:
        assert "Research · Twelve Data" in advisory_content
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

    removed = client.post(reverse("tracked-symbol-delete", args=[preference.id]))
    added = client.post(reverse("my-list"), {"symbol": "CHEAP"})

    assert removed.status_code == 302
    assert added.status_code == 302
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

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest
from django.urls import reverse
from django.utils.formats import date_format
from django.utils.html import strip_tags

from stanstock.core.models import JobRun
from stanstock.data.models import DataAsset, LatestMarketData
from stanstock.data.research_product_jobs import execute_daily_research_job
from stanstock.portfolio.models import Portfolio, PortfolioHolding, TrackedSymbol
from stanstock.research.models import AnalysisRun, Prediction, PredictionOutcome, StockAnalysis
from stanstock.research.outcomes import evaluate_prediction
from stanstock.research.price_product_config import (
    PRODUCT_EFFECTIVE_CONFIG_HASH as FORECAST_HASH,
)
from stanstock.research.price_product_config import PRODUCT_VERSION as V1
from stanstock.research.product_reader import read_research_product
from stanstock.web.templatetags.stanstock import price

pytestmark = pytest.mark.django_db
HORIZONS = (("6m", "6 months"), ("12m", "12 months"), ("3y", "3 years"), ("5y", "5 years"))


def _issue(env, *, product_version, issuance):
    return execute_daily_research_job(
        target_date=env.target,
        owner=env.owner,
        issuance_key=issuance,
        product_version=product_version,
        issued_on_time=False,
        core_config_path=env.path,
        store=env.store,
        enforce_rate_limit=False,
        derive_frequencies=True,
    )


@pytest.fixture
def c2_web_product(c2_environment, request):
    """Real native-format producer, including a positive independent NVDA control."""
    from stanstock.data.providers import twelve_data

    env = c2_environment
    scenario = getattr(request, "param", "matched")
    if scenario == "missing":
        env.options["missing_volume"] = True
    elif scenario == "canary":
        env.options["ignored_none"] = True
    elif scenario == "flat":
        env.options["flat"] = True
    original_fetch = env.fetch.side_effect

    def fetch(symbol, **kwargs):
        series = original_fetch(symbol, **kwargs)
        if symbol != "AAPL" or scenario not in ("zero", "mixed"):
            return series
        document = json.loads(series.raw_bytes)
        for row in document["values"]:
            if scenario == "zero":
                row["volume"] = "0"
            else:
                for field in ("open", "high", "low", "close"):
                    row[field] = str(Decimal(row[field]) / 20)
        return twelve_data.parse_daily_price_series(
            json.dumps(document).encode(),
            symbol=symbol,
            retrieved_at=series.retrieved_at,
            source_url=series.source_url,
            start_date=kwargs.get("start_date"),
            end_date=kwargs["end_date"],
            adjustment=kwargs["adjustment"],
        )

    env.fetch.side_effect = fetch
    env.job = _issue(env, product_version=env.version, issuance="c2-reader-web-v2")
    env.run = AnalysisRun.objects.get(pk=env.job.details["analysis_run_id"])
    assert env.job.status == JobRun.Status.SUCCESS
    assert env.run.config_version == env.version and not env.run.issued_on_time
    assert Prediction.objects.count() == 10 and env.fetch.call_count in (3, 4)
    return env


def _counts():
    return tuple(
        model.objects.count()
        for model in (
            AnalysisRun,
            StockAnalysis,
            Prediction,
            PredictionOutcome,
            DataAsset,
            LatestMarketData,
            JobRun,
            TrackedSymbol,
            Portfolio,
        )
    )


def _deny_get_work(env, monkeypatch):
    env.fetch.reset_mock()
    env.credentials.reset_mock()
    env.fetch.side_effect = AssertionError("GET must not acquire provider data")
    env.credentials.side_effect = AssertionError("GET must not resolve credentials")
    no_simulation = Mock(side_effect=AssertionError("GET must not run FHS paths"))
    monkeypatch.setattr(
        "stanstock.research.price_product.simulate_fhs_terminal_logs", no_simulation
    )
    return no_simulation


def _text(response):
    return " ".join(strip_tags(response.content.decode()).split())


def test_real_v2_buy_producer_immutable_reader_and_existing_authenticated_render(
    c2_web_product, client, monkeypatch
):
    env = c2_web_product
    assert env.fetch.call_count == 4 and env.credentials.call_count == 1
    client.force_login(env.owner)
    no_simulation = _deny_get_work(env, monkeypatch)
    counts = _counts()
    assets = set(DataAsset.objects.values_list("pk", "sha256"))
    original = read_research_product(user=env.owner, store=env.store)
    assert original.available and original.run == env.run
    assert len(original.admissions) == 100 and len(original.cards) == 2
    assert all(
        card.suggestion == "buy" and card.current_promotion_eligible for card in original.cards
    )
    assert all(card.liquidity_status == "matched" for card in original.cards)
    for horizon, label in HORIZONS:
        response = client.get(reverse("opportunities"), {"horizon": horizon})
        assert response.status_code == 200
        assert response.context["product"].run == env.run
        assert response.context["selected_horizon_label"] == label
        assert len(response.context["overview_cards"]) == 2
        assert all(
            entry["projection"].horizon == horizon for entry in response.context["overview_cards"]
        )
        content = _text(response)
        assert env.version in content
        assert f"{label} median cumulative price return" in content
        assert "Eligibility v2 activity evidence: Matched" in content
        assert "not actual transaction-dollar turnover or universal volume certification" in content
        assert "FHS projection inputs and seed policy are unchanged" in content
        assert "not validated real-world odds" in content
        assert all(
            f'aria-label="{full_label} projections"' in response.content.decode()
            for _key, full_label in HORIZONS
        )
        for card in response.context["product"].cards:
            assert card.decision_prediction.calculation["forecast_config_hash"] == FORECAST_HASH
            assert all(row.method_version == "us-price-fhs-v1" for row in card.advisory_predictions)
            assert card.source_retrieved_at <= env.run.data_cutoff <= env.run.generated_at
            assert date_format(card.liquidity_window_dates[0], "DATE_FORMAT") in content
            assert date_format(card.liquidity_window_dates[-1], "DATE_FORMAT") in content
            assert f"{price(card.analysis.current_price)} USD" in content
    card = original.cards[0]
    detail = client.get(reverse("stock-detail", args=[card.listing.pk]))
    assert detail.status_code == 200 and detail.context["card"].suggestion == "buy"
    detail_text = _text(detail)
    assert env.version in detail_text
    assert "20-session closing-price activity proxy" in detail_text
    assert "Price coverage and retrieval details" in detail_text
    assert "Reference close" in detail_text and "Market date" in detail_text
    assert "not actual transaction-dollar turnover or universal volume certification" in detail_text
    assert all(f"<h3>{label}</h3>" in detail.content.decode() for _key, label in HORIZONS)
    methodology = client.get(reverse("methodology"))
    assert methodology.status_code == 200
    assert f"Selected {env.version}" in _text(methodology)
    assert _counts() == counts and set(DataAsset.objects.values_list("pk", "sha256")) == assets
    env.fetch.assert_not_called()
    env.credentials.assert_not_called()
    no_simulation.assert_not_called()


@pytest.mark.parametrize("selected_version", [V1, "research-product-v2"])
def test_v2_portfolio_research_cell_never_repeats_scoreless_buy(
    c2_persisted, client, settings, monkeypatch, selected_version
):
    env = c2_persisted
    source = env.run.stocks.get(listing__provider_symbol="AAPL")
    assert source.run.config_version == env.version == "research-product-v2"
    assert source.recommendation == "buy" and source.overall_score is None
    assert not env.run.issued_on_time
    settings.RESEARCH_PRODUCT_VERSION = selected_version
    portfolio = Portfolio.objects.create(
        owner=env.owner,
        name="C2 scoreless momentum holding",
        base_currency="USD",
        cash_balance=0,
    )
    PortfolioHolding.objects.create(
        portfolio=portfolio,
        listing=source.listing,
        quantity=1,
        average_cost=source.current_price,
    )
    client.force_login(env.owner)
    no_simulation = _deny_get_work(env, monkeypatch)
    counts = _counts()

    response = client.get(reverse("portfolio-detail", args=[portfolio.pk]), {"section": "holdings"})

    assert response.status_code == 200
    assert response.context["position_cards"][0]["analysis"] == source
    research_html = (
        response.content.decode()
        .split('<td data-label="Research" class="holding-research">', 1)[1]
        .split("</td>", 1)[0]
    )
    research_cell = " ".join(strip_tags(research_html).split())
    assert "None/100" not in research_cell, research_cell
    assert "BUY" not in research_cell, research_cell
    assert "No overall score — this is the 6-month momentum method." in research_cell
    assert "Action is not repeated here." in research_cell
    assert "Check verified current research" in research_cell
    assert _counts() == counts
    env.fetch.assert_not_called()
    env.credentials.assert_not_called()
    no_simulation.assert_not_called()


@pytest.mark.parametrize("c2_web_product", ["missing", "canary"], indirect=True)
def test_unavailable_activity_blocks_buy_but_not_valid_advisory_render(
    c2_web_product, client, monkeypatch
):
    env = c2_web_product
    client.force_login(env.owner)
    no_simulation = _deny_get_work(env, monkeypatch)
    counts = _counts()
    response = client.get(reverse("opportunities"))
    assert response.status_code == 200 and response.context["product"].available
    aapl = next(card for card in response.context["product"].cards if card.listing.ticker == "AAPL")
    assert aapl.liquidity_status == "unavailable" and aapl.average_dollar_turnover_20d is None
    assert aapl.suggestion == "hold" and not aapl.current_promotion_eligible
    assert aapl.blocking_reasons and aapl.liquidity_reasons
    assert all(projection.available for projection in aapl.projections)
    content = _text(response)
    assert "Missing matched-mode activity evidence does not mean low or zero trading activity" in (
        content
    )
    assert "closing-price activity proxy is unavailable" in content
    assert "Restriction:" in content
    assert "projection withheld" not in content
    detail = client.get(reverse("stock-detail", args=[aapl.listing.pk]))
    assert detail.status_code == 200 and "Research restriction." in _text(detail)
    assert all(f"<h3>{label}</h3>" in detail.content.decode() for _key, label in HORIZONS)
    assert _counts() == counts
    env.fetch.assert_not_called()
    env.credentials.assert_not_called()
    no_simulation.assert_not_called()


@pytest.mark.parametrize("c2_web_product", ["zero"], indirect=True)
def test_real_zero_is_matched_measured_activity_not_missing_or_buy(c2_web_product, client):
    env = c2_web_product
    client.force_login(env.owner)
    response = client.get(reverse("opportunities"))
    assert response.status_code == 200 and response.context["product"].available
    cards = {card.listing.ticker: card for card in response.context["product"].cards}
    zero = cards["AAPL"]
    assert zero.liquidity_status == "matched"
    assert zero.average_dollar_turnover_20d == Decimal(0)
    assert zero.suggestion == "hold" and not zero.current_promotion_eligible
    assert cards["NVDA"].liquidity_status == "matched" and cards["NVDA"].suggestion == "buy"
    assert "closing-price activity proxy is $0." in _text(response)
    assert "Missing matched-mode activity evidence" not in _text(response)
    detail = client.get(reverse("stock-detail", args=[zero.listing.pk]))
    assert detail.status_code == 200
    assert "20-session closing-price activity proxy$0" in _text(detail)
    assert all(projection.available for projection in zero.projections)


@pytest.mark.parametrize("c2_web_product", ["mixed"], indirect=True)
def test_under_ten_native_price_is_visible_zero_allocation_never_promotion(c2_web_product, client):
    env = c2_web_product
    client.force_login(env.owner)
    response = client.get(reverse("opportunities"), {"price_band": "under_10", "horizon": "12m"})
    assert response.status_code == 200
    assert len(response.context["overview_cards"]) == 1
    card = response.context["overview_cards"][0]["card"]
    assert card.listing.ticker == "AAPL" and card.target_under_10
    assert card.analysis.current_price < 10
    assert card.average_dollar_turnover_20d > 5_000_000
    assert card.suggestion == "hold" and not card.current_promotion_eligible
    assert "Speculative watch only · 0% new allocation." in _text(response)
    assert "12 months median cumulative price return" in _text(response)
    assert not response.context["show_shortlists"]
    assert response.context["shortlist_sections"] == ()
    detail = client.get(reverse("stock-detail", args=[card.listing.pk]))
    assert detail.status_code == 200
    assert "Under $10 · speculative watch · 0% new allocation." in _text(detail)


@pytest.mark.parametrize("failure", ["wrong_owner", "rights", "corruption"])
def test_v2_authority_and_corrupt_immutable_proof_fail_closed_in_real_pages(
    c2_web_product, client, django_user_model, monkeypatch, failure
):
    env = c2_web_product
    assert client.get(reverse("opportunities")).status_code == 302
    viewer = env.owner
    if failure == "wrong_owner":
        viewer = django_user_model.objects.create_user(username="c2-other-private-web-viewer")
        expected = "absent"
    elif failure == "rights":
        env.record.refresh_from_db()
        env.record.metadata["internal_display_rights_confirmed"] = False
        env.record.save(update_fields=["metadata"])
        expected = "unauthorized"
    else:
        raw = DataAsset.objects.filter(kind="liquidity_price_raw").first()
        env.store.resolve(raw.relative_path).write_bytes(b"corrupt synthetic C2 source")
        expected = "integrity_failed"
    client.force_login(viewer)
    no_simulation = _deny_get_work(env, monkeypatch)
    counts = _counts()
    read = read_research_product(user=viewer, store=env.store)
    assert read.status == expected and read.cards == ()
    response = client.get(reverse("opportunities"))
    assert response.status_code == 200 and response.context["product"].status == expected
    assert b'class="opportunity-explanation"' not in response.content
    assert b"<h3>AAPL</h3>" not in response.content and b"<h3>NVDA</h3>" not in response.content
    assert response.context["shortlist_sections"] == ()
    detail = client.get(reverse("stock-detail", args=[env.listings["AAPL"].pk]))
    assert detail.status_code == (503 if failure == "corruption" else 404)
    for route in ("predictions", "performance"):
        page = client.get(reverse(route))
        assert page.status_code == 200
        assert b'class="issuance-card"' not in page.content
        assert b"<h3>AAPL</h3>" not in page.content
    assert _counts() == counts
    env.fetch.assert_not_called()
    env.credentials.assert_not_called()
    no_simulation.assert_not_called()


def test_current_selector_and_retained_v1_history_remain_exactly_versioned(
    c2_web_product, client, settings
):
    env = c2_web_product
    old_job = _issue(env, product_version=V1, issuance="c2-reader-web-retained-v1")
    v1_run = AnalysisRun.objects.get(pk=old_job.details["analysis_run_id"])
    client.force_login(env.owner)
    settings.RESEARCH_PRODUCT_VERSION = V1
    old = client.get(reverse("opportunities"))
    assert old.status_code == 200 and old.context["product"].run == v1_run
    assert "Eligibility v2 activity evidence" not in _text(old)
    assert "Missing compatible turnover means no valid liquidity evidence" in _text(old)
    settings.RESEARCH_PRODUCT_VERSION = env.version
    current = client.get(reverse("opportunities"))
    assert current.status_code == 200 and current.context["product"].run == env.run
    assert "Eligibility v2 activity evidence" in _text(current)
    history = client.get(reverse("predictions"))
    assert history.status_code == 200 and history.context["history"].available
    assert {item.run.config_version for item in history.context["history"].cohorts} == {
        V1,
        env.version,
    }
    assert env.version in _text(history)
    performance = client.get(reverse("performance"))
    assert performance.status_code == 200
    assert performance.context["has_eligibility_v2_history"]
    assert "not independent market trials or one pooled strategy-performance claim" in (
        _text(performance)
    )
    assert "does not validate the v2 activity evidence or eligibility correction" in _text(
        performance
    )


def test_v2_only_history_does_not_claim_v1_issuances(c2_web_product, client):
    env = c2_web_product
    client.force_login(env.owner)
    response = client.get(reverse("performance"))
    assert response.status_code == 200 and response.context["history"].available
    assert {item.run.config_version for item in response.context["history"].runs} == {env.version}
    assert response.context["has_eligibility_v2_history"]
    assert "This history includes eligibility v1 and v2" not in _text(response)
    assert "not independent market trials or one pooled strategy-performance claim" in _text(
        response
    )


def test_all_legacy_archive_routes_exclude_v2_and_disabled_rollback_never_relabels_it(
    c2_web_product, persisted_analysis, client, settings
):
    env = c2_web_product
    client.force_login(env.owner)
    row = Prediction.objects.filter(analysis__run=env.run, evidence_role="advisory").first()
    outcome = evaluate_prediction(
        row,
        provider="twelve_data",
        evaluation_date=env.target,
        evaluation_time=env.now,
        benchmark_subject="SPY",
        store=env.store,
    ).outcome
    assert outcome.status == "unresolved"
    archive = client.get(reverse("archive-opportunities"))
    assert archive.status_code == 200 and archive.context["archive_run"] == persisted_analysis.run
    assert list(archive.context["analyses"]) == [persisted_analysis]
    history = client.get(reverse("archive-predictions"))
    assert history.status_code == 200
    assert {item.analysis.run.config_version for item in history.context["predictions"]} == {
        persisted_analysis.run.config_version
    }
    assert (
        client.get(reverse("archive-stock-detail", args=[env.listings["AAPL"].pk])).status_code
        == 404
    )
    legacy_detail = client.get(
        reverse("archive-stock-detail", args=[persisted_analysis.listing.pk])
    )
    assert legacy_detail.status_code == 200
    assert legacy_detail.context["analysis"] == persisted_analysis
    performance = client.get(reverse("archive-performance"))
    assert performance.status_code == 200 and performance.context["groups"] == []
    settings.RESEARCH_PRODUCT_ENABLED = False
    for route, template in (
        ("opportunities", "web/product_archive_opportunities.html"),
        ("predictions", "web/product_archive_history.html"),
        ("performance", "web/product_archive_performance.html"),
    ):
        response = client.get(reverse(route))
        assert response.status_code == 200 and response.templates[0].name == template
        assert env.version not in _text(response)
    assert client.get(reverse("stock-detail", args=[env.listings["AAPL"].pk])).status_code == 404


@pytest.mark.parametrize("c2_web_product", ["flat"], indirect=True)
def test_all_null_v2_advisories_remain_not_evaluable_in_real_history_and_performance(
    c2_web_product, client
):
    env = c2_web_product
    assert Prediction.objects.filter(evidence_role="advisory", bear_return=None).count() == 8
    client.force_login(env.owner)
    history = client.get(reverse("predictions"))
    assert history.status_code == 200 and history.context["history"].available
    entries = history.context["history_entries"]
    assert len(entries) == 2
    assert all(entry["selected_advisory"]["non_evaluable"] for entry in entries)
    assert "Not evaluable — forecast withheld" in _text(history)
    performance = client.get(reverse("performance"))
    assert performance.status_code == 200 and performance.context["history"].available
    assert performance.context["advisory_groups"] == []
    assert not performance.context["has_matured"]
    assert not PredictionOutcome.objects.exists()


@pytest.mark.parametrize("c2_web_product", ["mixed"], indirect=True)
def test_v2_bounded_desktop_mobile_native_disclosures_and_compact_rows(
    c2_web_product, client, tmp_path
):
    """One browser round, two viewports, real authenticated source-bound renders."""
    from playwright.sync_api import sync_playwright

    env = c2_web_product
    client.force_login(env.owner)
    responses = [
        (horizon, label, client.get(reverse("opportunities"), {"horizon": horizon}))
        for horizon, label in HORIZONS
    ]
    css = Path(__file__).resolve().parents[1] / "static/css/stanstock.css"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for width, name in ((1440, "desktop"), (375, "mobile")):
                page = browser.new_page(viewport={"width": width, "height": 900})
                page.route("**/*", lambda route: route.abort())
                try:
                    for horizon, label, response in responses:
                        assert response.status_code == 200
                        page.set_content(response.content.decode())
                        page.add_style_tag(path=str(css))
                        rows = page.locator(".compact-opportunity")
                        assert rows.count() == 2
                        assert page.evaluate("document.documentElement.scrollWidth") <= width
                        assert page.locator(".opportunity-explanation[open]").count() == 0
                        ordinary = rows.filter(
                            has=page.get_by_role("heading", name="NVDA", exact=True)
                        )
                        under_ten = rows.filter(
                            has=page.get_by_role("heading", name="AAPL", exact=True)
                        )
                        assert ordinary.locator(
                            ".compact-opportunity-facts > div > dt"
                        ).all_text_contents() == [
                            "Dated close",
                            "Relative momentum",
                            "Action",
                            f"{label} median cumulative price return",
                        ]
                        assert under_ten.get_by_text(
                            "Speculative watch only · 0% new allocation.", exact=True
                        ).is_visible()
                        assert "BUY" in ordinary.locator(".compact-opportunity-facts").inner_text()
                        assert (
                            "HOLD" in under_ten.locator(".compact-opportunity-facts").inner_text()
                        )
                        assert ordinary.locator("time").get_attribute("datetime") == (
                            env.target.isoformat()
                        )
                        if width >= 1280:
                            assert ordinary.bounding_box()["height"] <= 90
                        assert page.locator(".horizon-links a[aria-label]").evaluate_all(
                            "items => items.map(item => item.getAttribute('aria-label'))"
                        ) == [f"{full_label} projections" for _key, full_label in HORIZONS]
                        if horizon == "6m":
                            page.screenshot(
                                path=str(tmp_path / f"c2-{name}-collapsed.png"), full_page=True
                            )
                        summary = ordinary.locator(".opportunity-explanation > summary")
                        summary.focus()
                        page.keyboard.press("Enter")
                        evidence = ordinary.locator(".opportunity-explanation-body")
                        assert evidence.is_visible()
                        assert "Eligibility v2 activity evidence: Matched" in evidence.inner_text()
                        assert (
                            "not actual transaction-dollar turnover or "
                            "universal volume certification" in evidence.inner_text()
                        )
                        assert page.evaluate("document.documentElement.scrollWidth") <= width
                        if horizon == "6m":
                            page.screenshot(
                                path=str(tmp_path / f"c2-{name}-expanded.png"), full_page=True
                            )
                        summary.focus()
                        page.keyboard.press("Space")
                        assert not evidence.is_visible()
                finally:
                    page.close()
        finally:
            browser.close()

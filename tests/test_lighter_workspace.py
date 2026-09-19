"""Synthetic web contracts for the lighter workspace; no live provider or database."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from django.db import connection
from django.test import Client
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone

from stanstock.data.models import DataAsset, LatestMarketData
from stanstock.portfolio.models import Portfolio, PortfolioHolding
from stanstock.portfolio.planner import preview_monthly_contribution_plan
from stanstock.portfolio.service import (
    build_sample_portfolio,
    record_portfolio_snapshot,
    upsert_holding,
)
from stanstock.research.models import AnalysisRun, Prediction
from stanstock.simulation.models import SimulationDefinition, SimulationRun
from stanstock.web import views
from stanstock.web.templatetags.stanstock import price
from test_portfolio_planner import TEST_NOW, _planner_market
from test_portfolio_tracking import (
    _provider_analysis_run,
)
from test_portfolio_tracking import (
    priced_listing as listing_fixture,
)

pytestmark = pytest.mark.django_db


def _assert_no_writes(queries):
    assert not [
        query["sql"].split()[0]
        for query in queries
        if query["sql"].lstrip().split()[0].upper() in {"INSERT", "UPDATE", "DELETE", "REPLACE"}
    ]


@pytest.fixture
def workspace(client, django_user_model, settings, monkeypatch):
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = False
    monkeypatch.setattr(
        "httpx.Client.send", Mock(side_effect=AssertionError("Provider HTTP forbidden"))
    )
    monkeypatch.setattr(timezone, "localdate", lambda: date(2026, 9, 4))
    owner = django_user_model.objects.create_user(username="workspace-owner")
    listing = listing_fixture.__wrapped__()
    portfolios = []
    for index in range(3):
        portfolio = Portfolio.objects.create(
            owner=owner, name=f"Synthetic {index}", base_currency="USD"
        )
        for quantity in ("2", "3"):
            upsert_holding(
                portfolio=portfolio,
                listing=listing,
                quantity=Decimal(quantity),
                average_cost=Decimal("80"),
            )
            record_portfolio_snapshot(portfolio)
        portfolios.append(portfolio)
    empty = Portfolio.objects.create(
        owner=owner, name="Synthetic zero snapshots", base_currency="USD"
    )
    client.force_login(owner)
    return owner, listing, portfolios, empty


def test_portfolio_sections_only_prepare_selected_optional_work(client, workspace):
    _owner, _listing, portfolios, _empty = workspace
    url = reverse("portfolio-detail", args=[portfolios[0].pk])
    for section in views.PORTFOLIO_SECTIONS:
        with (
            patch.object(
                views,
                "preview_monthly_contribution_plan",
                wraps=views.preview_monthly_contribution_plan,
            ) as preview,
            patch.object(
                views,
                "portfolio_snapshot_series",
                wraps=views.portfolio_snapshot_series,
            ) as series,
            CaptureQueriesContext(connection) as queries,
        ):
            response = client.get(url, {} if section == "holdings" else {"section": section})
        assert response.status_code == 200
        assert response.context["section"] == section
        assert preview.call_count == (section == "plan")
        assert series.call_count == (section == "activity")
        assert (response.context["holding_form"] is not None) == (section == "holdings")
        assert (response.context["deposit_form"] is not None) == (section == "activity")
        assert (response.context["portfolio_form"] is not None) == (section == "settings")
        assert ("deposits" in response.context) == (section == "activity")
        assert ("plan_executions" in response.context) == (section == "activity")
        assert ("performance_baselines" in response.context) == (section == "activity")
        assert (b'name="action" value="deposit"' in response.content) == (section == "activity")
        assert (b'name="action" value="holding"' in response.content) == (section == "holdings")
        assert (b'name="action" value="archive"' in response.content) == (section == "settings")
        _assert_no_writes(queries)
    with patch.object(views, "calculate_portfolio_valuation") as valuation:
        invalid = client.get(url, {"section": "../../plan"})
    assert invalid.status_code == 400
    assert b"Unknown portfolio section" in invalid.content
    valuation.assert_not_called()


def test_list_counts_rows_including_same_day_and_zero_without_per_portfolio_counts(
    client,
    workspace,
):
    _owner, _listing, portfolios, empty = workspace
    expected = {portfolio.pk: 2 for portfolio in portfolios} | {empty.pk: 0}
    assert len({row.as_of_date for row in portfolios[0].snapshots.all()}) == 1
    with CaptureQueriesContext(connection) as queries:
        response = client.get(reverse("portfolios"))
    assert {
        card["portfolio"].pk: card["snapshot_count"] for card in response.context["portfolio_cards"]
    } == expected
    count_queries = [
        query["sql"]
        for query in queries
        if "COUNT(" in query["sql"] and "portfolio_portfoliosnapshot" in query["sql"]
    ]
    assert len(count_queries) == 1
    assert "GROUP BY" in count_queries[0]
    assert b"Create portfolio</button>" not in response.content
    assert response.context["form"] is None
    creating = client.get(reverse("portfolios"), {"new": "1"})
    assert b"Create portfolio</button>" in creating.content
    assert creating.content.index(b'class="portfolio-card"') < creating.content.index(
        b'id="new-portfolio"'
    )
    _assert_no_writes(queries)


@pytest.mark.parametrize(
    ("destination", "payload", "status"),
    [
        ("simulations", {"mode": "portfolio", "starting_capital": "100000"}, 405),
        ("simulations", {"mode": "backtest", "name": "Direct legacy POST"}, 405),
        ("portfolios", {"action": "sample", "starting_capital": "100000", "top_n": "2"}, 400),
    ],
)
def test_retired_creators_reject_csrf_valid_posts_before_work_or_writes(
    workspace,
    destination,
    payload,
    status,
):
    owner, _listing, _portfolios, _empty = workspace
    client = Client(enforce_csrf_checks=True)
    url = reverse(destination)
    assert client.post(url, payload).status_code == 403
    client.get(reverse("login"))
    token = client.cookies["csrftoken"].value
    # With a valid token, anonymous requests still follow the existing login gate.
    assert client.post(url, payload, HTTP_X_CSRFTOKEN=token).status_code == 302
    client.force_login(owner)
    assert client.post(url, payload).status_code == 403
    models = (
        Portfolio,
        PortfolioHolding,
        SimulationDefinition,
        SimulationRun,
        DataAsset,
        AnalysisRun,
        Prediction,
    )
    counts = [model.objects.count() for model in models]
    with (
        patch("stanstock.simulation.builders.run_simulation_workflow") as workflow,
        patch("stanstock.portfolio.service.build_sample_portfolio") as builder,
        patch.object(views, "PortfolioForm") as form,
        CaptureQueriesContext(connection) as queries,
    ):
        response = client.post(url, payload, HTTP_X_CSRFTOKEN=token)
    assert response.status_code == status
    if status == 405:
        assert response.headers["Allow"] == "GET, HEAD"
    else:
        assert b"Browser sample portfolio creation is retired" in response.content
    workflow.assert_not_called()
    builder.assert_not_called()
    form.assert_not_called()
    _assert_no_writes(queries)
    assert [model.objects.count() for model in models] == counts


@pytest.mark.parametrize(
    ("action", "section", "field", "value"),
    [
        ("holding", "holdings", "quantity", "-2"),
        ("deposit", "activity", "amount", "-2"),
        ("update", "settings", "name", ""),
        ("deposit", "activity", "idempotency_key", "not-a-uuid"),
    ],
)
def test_invalid_forms_return_bound_values_in_their_own_section(
    client,
    workspace,
    action,
    section,
    field,
    value,
):
    _owner, _listing, portfolios, _empty = workspace
    url = reverse("portfolio-detail", args=[portfolios[0].pk])
    with CaptureQueriesContext(connection) as queries:
        response = client.post(f"{url}?section=plan", {"action": action, field: value})
    assert response.status_code == 400
    assert response.context["section"] == section
    form_name = {"holding": "holding_form", "deposit": "deposit_form", "update": "portfolio_form"}
    form = response.context[form_name[action]]
    assert form.is_bound
    assert form[field].value() == value
    assert field in form.errors
    assert form.errors[field][0].encode() in response.content
    assert f'name="action" value="{action}"'.encode() in response.content
    _assert_no_writes(queries)


def test_owner_sections_archive_restore_and_invalid_plan_are_safe(
    client,
    workspace,
    django_user_model,
):
    owner, _listing, portfolios, _empty = workspace
    portfolio = portfolios[0]
    url = reverse("portfolio-detail", args=[portfolio.pk])
    rejected = client.post(url, {"action": "execute_plan", "plan_hash": "invalid"})
    assert rejected.status_code == 400
    assert rejected.context["section"] == "plan"
    assert b"Plan confirmation request was invalid" in rejected.content
    assert client.post(url, {"action": "archive"}).status_code == 302
    portfolio.refresh_from_db()
    assert portfolio.archived_at is not None
    with patch.object(views, "preview_monthly_contribution_plan") as preview:
        archived = client.get(url, {"section": "plan"})
    preview.assert_not_called()
    assert b"Planning is unavailable" in archived.content
    other = django_user_model.objects.create_user(username="workspace-other")
    client.force_login(other)
    for section in views.PORTFOLIO_SECTIONS:
        assert client.get(url, {"section": section}).status_code == 404
        assert client.post(f"{url}?section={section}", {"action": "restore"}).status_code == 404
    client.force_login(owner)
    assert client.post(url, {"action": "restore"}).status_code == 302
    portfolio.refresh_from_db()
    assert portfolio.archived_at is None


def test_direct_deposit_and_confirmation_retries_do_not_require_preview_get(
    client,
    django_user_model,
    monkeypatch,
):
    monkeypatch.setattr(timezone, "now", lambda: TEST_NOW)
    monkeypatch.setattr(timezone, "localdate", lambda: TEST_NOW.date())
    _planner_market()
    owner = django_user_model.objects.create_user(username="direct-plan-owner")
    portfolio = Portfolio.objects.create(owner=owner, name="Direct plan", base_currency="USD")
    client.force_login(owner)
    url = reverse("portfolio-detail", args=[portfolio.pk])
    deposit = {"action": "deposit", "amount": "600", "idempotency_key": str(uuid4())}
    for _ in range(2):
        assert client.post(url, deposit).status_code == 302
    portfolio.refresh_from_db()
    assert portfolio.cash_balance == Decimal("600")
    assert portfolio.deposits.count() == 1
    plan = preview_monthly_contribution_plan(portfolio)
    confirmation = {
        "action": "execute_plan",
        "plan_hash": plan.plan_hash,
        "idempotency_key": str(uuid4()),
    }
    # The POST calls the service's locked recomputation, never the GET display preview.
    with patch.object(views, "preview_monthly_contribution_plan") as display_preview:
        for _ in range(2):
            assert client.post(url, confirmation).status_code == 302
    display_preview.assert_not_called()
    assert portfolio.plan_executions.count() == 1
    assert portfolio.plan_executions.get().purchases.count() == 2


@pytest.mark.parametrize("viewport", [(320, 812), (375, 812), (1280, 900), (1440, 900)])
@pytest.mark.parametrize("state", ["priced", "missing", "tiny", "frozen", "archived"])
def test_real_portfolio_and_holding_first_viewport(client, workspace, viewport, state):
    """Injected HTML proves layout only; real login/CSRF has its own HTTP suite."""
    owner, listing, portfolios, _empty = workspace
    portfolio = portfolios[0]
    if state in {"frozen", "archived"}:
        _provider_analysis_run(count=2)
        portfolio, _ = build_sample_portfolio(
            owner=owner, starting_capital=Decimal("100000"), top_n=2
        )
        if state == "archived":
            portfolio.archived_at = timezone.now()
            portfolio.save(update_fields=["archived_at", "updated_at"])
    elif state == "missing":
        LatestMarketData.objects.filter(listing=listing).delete()
    elif state == "tiny":
        market = LatestMarketData.objects.get(listing=listing)
        market.close = Decimal("0.000042")
        market.previous_close = market.close
        market.save(update_fields=["close", "previous_close"])
    responses = [
        (client.get(reverse("portfolios")), ".portfolio-records .portfolio-card"),
        (client.get(reverse("portfolio-detail", args=[portfolio.pk])), ".holding-record"),
    ]
    playwright = pytest.importorskip("playwright.sync_api")
    css = Path(__file__).resolve().parents[1] / "static/css/stanstock.css"
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
            for response, selector in responses:
                assert response.status_code == 200
                page.set_content(response.content.decode())
                page.add_style_tag(path=str(css))
                assert page.locator(selector).count() >= 1
                first = page.locator(selector).first
                box = first.bounding_box()
                if viewport[0] >= 375:
                    assert 0 <= box["y"] < box["y"] + box["height"] <= viewport[1], box
                assert page.evaluate("document.documentElement.scrollWidth") <= viewport[0]
                assert first.evaluate("(e) => e.scrollWidth <= e.clientWidth")
                if selector == ".holding-record":
                    if state == "missing":
                        assert "Unavailable" in first.inner_text()
                        assert page.get_by_text("Current total withheld.", exact=True).is_visible()
                    elif state == "tiny":
                        assert "0.000042" in first.inner_text()
                        assert "0%." in first.inner_text()
                    elif state in {"frozen", "archived"}:
                        assert b"Research-reference portfolio; composition is frozen." in (
                            response.content
                        )
                        assert portfolio.description.encode() in response.content
                        assert "100000.00 USD" in page.locator(".metric-grid").inner_text()
                        assert page.locator(".portfolio-notes").get_attribute("open") is None
                    for cell in first.locator("td, th").all():
                        assert cell.evaluate("(e) => e.scrollWidth <= e.clientWidth")
                links = page.locator(".primary-nav-links a")
                assert links.all_inner_texts() == ["Opportunities", "Market", "Portfolios"]
                assert links.evaluate_all(
                    "(es) => es.every(e => parseFloat(getComputedStyle(e).fontSize) >= 14"
                    " && e.getBoundingClientRect().height >= 32)"
                )
                links.first.focus()
                assert links.first.evaluate("(e) => getComputedStyle(e).outlineStyle !== 'none'")
        finally:
            browser.close()


def test_primary_navigation_context_and_two_link_reference_routes(client, workspace):
    _owner, _listing, portfolios, _empty = workspace
    expected = [
        ("opportunities", [], "Opportunities"),
        ("predictions", [], "Opportunities"),
        ("performance", [], "Opportunities"),
        ("archive-opportunities", [], "Opportunities"),
        ("archive-predictions", [], "Opportunities"),
        ("archive-performance", [], "Opportunities"),
        ("market", [], "Market"),
        ("my-list", [], "Market"),
        ("portfolios", [], "Portfolios"),
        ("portfolio-detail", [portfolios[0].pk], "Portfolios"),
    ]
    for name, args, label in expected:
        with CaptureQueriesContext(connection) as queries:
            response = client.get(reverse(name, args=args))
        assert response.status_code == 200
        _assert_no_writes(queries)
        primary = response.content.decode().split('class="primary-nav-links"')[1].split("</div>")[0]
        assert primary.count("<a ") == 3
        assert primary.count('aria-current="page"') == 1
        assert f'aria-current="page">{label}</a>' in primary
        assert "Under $10" not in primary and "More" not in primary
        assert b'href="/methodology"' in response.content
        assert b'href="/status"' in response.content
    opportunities = client.get(reverse("opportunities")).content
    assert b'href="/performance">Research performance</a>' in opportunities
    assert b'href="/predictions">Prediction history</a>' in opportunities
    assert b'href="/my-list"' in client.get(reverse("market")).content
    assert b'href="/simulations"' in client.get(reverse("methodology")).content


def test_palette_text_and_control_contrast():
    """Pin the flat palette's minimum AA text and control contrast."""
    css = (Path(__file__).resolve().parents[1] / "static/css/stanstock.css").read_text()
    assert "gradient(" not in css and "backdrop-filter:" not in css

    def luminance(value):
        values = [int(value[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values]
        return sum(v * weight for v, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

    def contrast(first, second):
        light, dark = sorted([luminance(first), luminance(second)], reverse=True)
        return (light + 0.05) / (dark + 0.05)

    palette = {}
    for line in css.splitlines():
        if line.strip().startswith("--") and ": #" in line:
            key, value = line.strip().rstrip(";").split(": ")
            palette[key] = value
    for background in ("--bg", "--panel", "--panel-strong"):
        for text in ("--text", "--muted", "--accent", "--danger", "--warning"):
            assert contrast(palette[text], palette[background]) >= 4.5
        assert contrast(palette["--control"], palette[background]) >= 3


@pytest.mark.parametrize("width", [375, 1280])
def test_price_band_text_and_counts_have_computed_aa_contrast(width):
    """Check the cascade, not only isolated palette colors, on actual controls."""
    playwright = pytest.importorskip("playwright.sync_api")
    css = Path(__file__).resolve().parents[1] / "static/css/stanstock.css"

    def luminance(channels):
        values = [channel / 255 for channel in channels]
        linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values]
        return sum(v * weight for v, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": width, "height": 812})
            page.set_content(
                '<!doctype html><main class="opportunities-page">'
                '<div class="price-band-links">'
                '<a class="band-link is-active" href="/opportunities" aria-current="page">'
                "All <span>3</span></a>"
                '<a class="band-link" href="/opportunities?price_band=under_10">'
                "Under $10 <span>1</span></a></div></main>"
            )
            page.add_style_tag(path=str(css))
            styles = page.locator(".band-link, .band-link span").evaluate_all(
                """elements => elements.map(element => {
                    const rgb = color => color.match(/[\\d.]+/g).slice(0, 3).map(Number);
                    let background = element;
                    while (getComputedStyle(background).backgroundColor === "rgba(0, 0, 0, 0)") {
                        background = background.parentElement;
                    }
                    return {
                        selected: element.closest("a").classList.contains("is-active"),
                        part: element.tagName === "SPAN" ? "count" : "text",
                        foreground: rgb(getComputedStyle(element).color),
                        background: rgb(getComputedStyle(background).backgroundColor)
                    };
                })"""
            )
            assert len(styles) == 4
            for style in styles:
                light, dark = sorted(
                    (luminance(style["foreground"]), luminance(style["background"])),
                    reverse=True,
                )
                ratio = (light + 0.05) / (dark + 0.05)
                print(f"width={width} selected={style['selected']} {style['part']} {ratio:.3f}:1")
                assert ratio >= 4.5, style
            selected = [style for style in styles if style["selected"]]
            assert selected[0]["foreground"] == selected[1]["foreground"]
            assert page.locator(".band-link").evaluate_all(
                "links => links.every(link => link.getBoundingClientRect().height >= 32)"
            )
        finally:
            browser.close()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "Unavailable"),
        ("NaN", "Unavailable"),
        ("0", "0.00"),
        ("100.126", "100.13"),
        ("0.12345", "0.1234"),
        ("0.0004", "0.0004"),
        ("0.000042", "0.000042"),
        ("0.00000001", "0.00000001"),
    ],
)
def test_tiny_price_display_preserves_nonzero_quotes_and_existing_precision(value, expected):
    assert price(value) == expected

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse


@pytest.fixture
def insecure_http_settings(settings) -> None:
    settings.CSRF_COOKIE_SECURE = False
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.SESSION_COOKIE_SECURE = False


@pytest.fixture
def csrf_live_server(insecure_http_settings, live_server):
    return live_server


@lru_cache(maxsize=1)
def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    with sync_playwright() as playwright:
        if not Path(playwright.chromium.executable_path).exists():
            return False
        browser = playwright.chromium.launch()
        browser.close()
        return True


def _login_url(live_server, destination: str) -> str:
    return f"{live_server.url}{reverse('login')}?{urlencode({'next': destination})}"


def _csrf_cookie_value(context, live_server) -> str:
    cookies = context.cookies(live_server.url)
    csrf_cookies = [cookie for cookie in cookies if cookie["name"] == "csrftoken"]
    assert len(csrf_cookies) == 1
    return str(csrf_cookies[0]["value"])


def _assert_login_page_layout(page, css_statuses: list[int]) -> None:
    assert page.locator('link[href*="css/stanstock.css"]').count() == 1
    assert any(status == 200 for status in css_statuses)
    assert page.get_by_role("button", name="Sign in", exact=True).is_visible()
    assert page.evaluate("document.documentElement.scrollWidth") <= page.evaluate(
        "document.documentElement.clientWidth"
    )


@pytest.mark.django_db(transaction=True)
@pytest.mark.skipif(not _chromium_available(), reason="Playwright's Chromium is not installed")
def test_browser_login_and_logout_follow_real_forms_at_narrow_and_desktop_widths(
    csrf_live_server,
) -> None:
    user = get_user_model().objects.create_user(
        username="browser-csrf-owner",
        password="synthetic-browser-csrf-password",
    )
    destination = f"{reverse('opportunities')}?horizon=3y"
    expected_destination = f"{csrf_live_server.url}{destination}"

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for width in (375, 1280):
                context = browser.new_context(viewport={"width": width, "height": 900})
                page = context.new_page()
                css_statuses: list[int] = []

                def capture_css_response(response, statuses: list[int] = css_statuses) -> None:
                    if "/static/css/stanstock.css" in response.url:
                        statuses.append(response.status)

                page.on("response", capture_css_response)
                try:
                    page.goto(expected_destination, wait_until="networkidle")
                    login_location = urlsplit(page.url)
                    live_server_location = urlsplit(csrf_live_server.url)
                    assert login_location.scheme == live_server_location.scheme
                    assert login_location.netloc == live_server_location.netloc
                    assert login_location.path == reverse("login")
                    assert parse_qs(login_location.query) == {"next": [destination]}
                    _assert_login_page_layout(page, css_statuses)
                    page.locator("#id_username").fill(user.username)
                    page.locator("#id_password").fill("synthetic-browser-csrf-password")
                    page.get_by_role("button", name="Sign in", exact=True).click()
                    page.wait_for_url(expected_destination)
                    assert page.url == expected_destination

                    page.get_by_role("button", name="Sign out", exact=True).click()
                    page.wait_for_url(f"{csrf_live_server.url}{reverse('login')}")
                finally:
                    context.close()
        finally:
            browser.close()


@pytest.mark.django_db(transaction=True)
@pytest.mark.skipif(not _chromium_available(), reason="Playwright's Chromium is not installed")
def test_browser_stale_login_form_recovers_in_two_pages_sharing_one_context(
    csrf_live_server,
) -> None:
    user = get_user_model().objects.create_user(
        username="browser-stale-owner",
        password="synthetic-browser-stale-password",
    )
    destination = f"{reverse('opportunities')}?horizon=3y"
    login_url = _login_url(csrf_live_server, destination)
    expected_destination = f"{csrf_live_server.url}{destination}"

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for width in (375, 1280):
                context = browser.new_context(viewport={"width": width, "height": 900})
                first_page = context.new_page()
                second_page = context.new_page()
                css_statuses: list[int] = []

                def capture_css_response(response, statuses: list[int] = css_statuses) -> None:
                    if "/static/css/stanstock.css" in response.url:
                        statuses.append(response.status)

                first_page.on("response", capture_css_response)
                try:
                    first_page.goto(login_url, wait_until="networkidle")
                    _assert_login_page_layout(first_page, css_statuses)
                    held_token = first_page.locator(
                        'input[name="csrfmiddlewaretoken"]'
                    ).input_value()
                    original_cookie = _csrf_cookie_value(context, csrf_live_server)
                    first_page.locator("#id_username").fill(user.username)
                    first_page.locator("#id_password").fill("synthetic-browser-stale-password")

                    second_page.goto(login_url, wait_until="networkidle")
                    second_page.locator("#id_username").fill(user.username)
                    second_page.locator("#id_password").fill("synthetic-browser-stale-password")
                    second_page.get_by_role("button", name="Sign in", exact=True).click()
                    second_page.wait_for_url(expected_destination)
                    rotated_cookie = _csrf_cookie_value(context, csrf_live_server)
                    assert rotated_cookie != original_cookie

                    second_page.get_by_role("button", name="Sign out", exact=True).click()
                    second_page.wait_for_url(f"{csrf_live_server.url}{reverse('login')}")
                    assert _csrf_cookie_value(context, csrf_live_server) == rotated_cookie

                    with first_page.expect_response(
                        lambda response: (
                            response.request.method == "POST" and response.url == login_url
                        )
                    ) as stale_response_info:
                        first_page.get_by_role("button", name="Sign in", exact=True).click()
                    stale_response = stale_response_info.value
                    assert stale_response.status == 403
                    assert "no-store" in stale_response.headers["cache-control"]
                    assert first_page.locator("form").count() == 0
                    assert first_page.locator('input[name="csrfmiddlewaretoken"]').count() == 0
                    assert first_page.locator(".primary-nav").count() == 0
                    assert (
                        first_page.get_by_role("button", name="Sign out", exact=True).count() == 0
                    )
                    assert first_page.evaluate(
                        "document.documentElement.scrollWidth"
                    ) <= first_page.evaluate("document.documentElement.clientWidth")
                    recovery = first_page.get_by_role(
                        "link",
                        name="Reload fresh sign-in",
                        exact=True,
                    )
                    assert recovery.is_visible()
                    recovery_box = recovery.bounding_box()
                    assert recovery_box is not None
                    assert 0 <= recovery_box["x"]
                    assert recovery_box["x"] + recovery_box["width"] <= width
                    assert 0 <= recovery_box["y"]
                    assert recovery_box["y"] + recovery_box["height"] <= 900
                    assert recovery.get_attribute("href") == (
                        f"{reverse('login')}?{urlencode({'next': destination})}"
                    )
                    recovery.click()
                    first_page.wait_for_url(login_url)
                    fresh_token = first_page.locator(
                        'input[name="csrfmiddlewaretoken"]'
                    ).input_value()
                    assert fresh_token != held_token
                    assert first_page.locator("#id_username").input_value() == ""
                    assert first_page.locator("#id_password").input_value() == ""
                    first_page.locator("#id_username").fill(user.username)
                    first_page.locator("#id_password").fill("synthetic-browser-stale-password")
                    first_page.get_by_role("button", name="Sign in", exact=True).click()
                    first_page.wait_for_url(expected_destination)
                    assert first_page.url == expected_destination
                finally:
                    context.close()
        finally:
            browser.close()

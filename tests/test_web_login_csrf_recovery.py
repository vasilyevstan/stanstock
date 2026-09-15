from __future__ import annotations

import re
from copy import deepcopy
from html import unescape
from urllib.parse import urlencode
from uuid import uuid4

import pytest
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.db import connection
from django.test import Client, RequestFactory, override_settings
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils.functional import SimpleLazyObject

from stanstock.web.csrf import csrf_failure

_CSRF_TOKEN_PATTERN = re.compile(r'name="csrfmiddlewaretoken" value="([^"]+)"')
_RECOVERY_LINK_PATTERN = re.compile(r'<a class="primary-button" href="([^"]+)">')


def _csrf_token(response) -> str:
    match = _CSRF_TOKEN_PATTERN.search(response.content.decode())
    assert match is not None
    return unescape(match.group(1))


def _recovery_url(response) -> str:
    match = _RECOVERY_LINK_PATTERN.search(response.content.decode())
    assert match is not None
    return unescape(match.group(1))


def _assert_csrf_recovery(response, *, recovery_url: str) -> None:
    assert response.status_code == 403
    assert _recovery_url(response) == recovery_url
    assert b"Action not performed" in response.content
    assert b"<form" not in response.content
    assert "no-store" in response.headers["Cache-Control"]
    assert "form-action 'self'" in response.headers["Content-Security-Policy"]


def _create_user(username: str = "csrf-owner"):
    return get_user_model().objects.create_user(
        username=username,
        password="synthetic-csrf-login-password",
    )


@pytest.mark.django_db
@override_settings(RESEARCH_PRODUCT_ENABLED=True)
def test_rendered_login_token_authenticates_and_preserves_safe_horizon() -> None:
    user = _create_user()
    csrf_client = Client(enforce_csrf_checks=True)
    destination = f"{reverse('opportunities')}?horizon=3y"

    page = csrf_client.get(reverse("login"), {"next": destination})
    response = csrf_client.post(
        reverse("login"),
        {
            "csrfmiddlewaretoken": _csrf_token(page),
            "username": user.username,
            "password": "synthetic-csrf-login-password",
            "next": destination,
        },
    )

    assert response.status_code == 302
    assert response.url == destination
    assert csrf_client.session["_auth_user_id"] == str(user.pk)


@pytest.mark.django_db
@override_settings(RESEARCH_PRODUCT_ENABLED=True)
def test_stale_login_form_recovers_without_changing_anonymous_or_authenticated_state() -> None:
    user = _create_user()
    csrf_client = Client(enforce_csrf_checks=True)
    login_url = reverse("login")
    destination = f"{reverse('opportunities')}?horizon=3y"
    held_page = csrf_client.get(login_url, {"next": destination})
    held_token = _csrf_token(held_page)

    signed_in = csrf_client.post(
        login_url,
        {
            "csrfmiddlewaretoken": held_token,
            "username": user.username,
            "password": "synthetic-csrf-login-password",
            "next": destination,
        },
    )
    assert signed_in.status_code == 302
    rotated_cookie = csrf_client.cookies["csrftoken"].value

    authenticated_page = csrf_client.get(reverse("status"))
    signed_out = csrf_client.post(
        reverse("logout"),
        {"csrfmiddlewaretoken": _csrf_token(authenticated_page)},
    )
    assert signed_out.status_code == 302
    assert csrf_client.cookies["csrftoken"].value == rotated_cookie
    assert "_auth_user_id" not in csrf_client.session

    stale_request_url = f"{login_url}?{urlencode({'next': destination})}"
    rejected_while_anonymous = csrf_client.post(
        stale_request_url,
        {
            "csrfmiddlewaretoken": held_token,
            "username": user.username,
            "password": "synthetic-csrf-login-password",
            "next": reverse("logout"),
        },
    )
    expected_recovery = f"{login_url}?{urlencode({'next': destination})}"
    _assert_csrf_recovery(rejected_while_anonymous, recovery_url=expected_recovery)
    assert "_auth_user_id" not in csrf_client.session

    fresh_page = csrf_client.get(_recovery_url(rejected_while_anonymous))
    fresh_token = _csrf_token(fresh_page)
    assert fresh_token != held_token
    recovered = csrf_client.post(
        _recovery_url(rejected_while_anonymous),
        {
            "csrfmiddlewaretoken": fresh_token,
            "username": user.username,
            "password": "synthetic-csrf-login-password",
            "next": destination,
        },
    )
    assert recovered.status_code == 302
    assert recovered.url == destination
    assert csrf_client.session["_auth_user_id"] == str(user.pk)

    rejected_while_authenticated = csrf_client.post(
        stale_request_url,
        {
            "csrfmiddlewaretoken": held_token,
            "username": user.username,
            "password": "synthetic-csrf-login-password",
            "next": destination,
        },
    )
    _assert_csrf_recovery(rejected_while_authenticated, recovery_url=expected_recovery)
    assert csrf_client.session["_auth_user_id"] == str(user.pk)


@pytest.mark.django_db
@pytest.mark.parametrize(
    "failure_kind",
    (
        "missing",
        "truncated",
        "malformed",
        "cookieless",
        "untrusted_origin",
        "https_missing_referer",
    ),
)
def test_login_csrf_rejections_remain_branded_403s(failure_kind: str) -> None:
    user = _create_user(f"csrf-{failure_kind}")
    login_url = reverse("login")
    destination = f"{reverse('opportunities')}?horizon=3y"
    secure = failure_kind == "https_missing_referer"
    csrf_client = Client(enforce_csrf_checks=True)
    request_url = f"{login_url}?{urlencode({'next': destination})}"
    page = csrf_client.get(request_url, secure=secure)
    token = _csrf_token(page)
    payload = {
        "csrfmiddlewaretoken": token,
        "username": user.username,
        "password": "synthetic-csrf-login-password",
        "next": destination,
    }
    extra: dict[str, str] = {}

    if failure_kind == "missing":
        payload.pop("csrfmiddlewaretoken")
    elif failure_kind == "truncated":
        payload["csrfmiddlewaretoken"] = token[:16]
    elif failure_kind == "malformed":
        payload["csrfmiddlewaretoken"] = "not-a-valid-csrf-token"
    elif failure_kind == "cookieless":
        cookieless_client = Client(enforce_csrf_checks=True)
        csrf_client = cookieless_client
    elif failure_kind == "untrusted_origin":
        extra["HTTP_ORIGIN"] = "https://untrusted.example"

    response = csrf_client.post(request_url, payload, secure=secure, **extra)

    _assert_csrf_recovery(
        response,
        recovery_url=f"{login_url}?{urlencode({'next': destination})}",
    )
    assert "_auth_user_id" not in csrf_client.session


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("safe_next", "secure"),
    (
        ("/opportunities?horizon=3y", False),
        ("http://testserver/opportunities?horizon=3y", False),
        ("//testserver/opportunities?horizon=3y", False),
        ("https://testserver/opportunities?horizon=3y", True),
    ),
)
def test_login_csrf_recovery_preserves_django_safe_same_host_next(
    safe_next: str, secure: bool
) -> None:
    csrf_client = Client(enforce_csrf_checks=True)
    login_url = reverse("login")
    request_url = f"{login_url}?{urlencode({'next': safe_next})}"
    csrf_client.get(request_url, secure=secure)

    response = csrf_client.post(
        request_url,
        {
            "username": "submitted-username-is-not-echoed",
            "password": "submitted-password-is-not-echoed",
            "next": reverse("logout"),
        },
        secure=secure,
    )

    _assert_csrf_recovery(
        response,
        recovery_url=f"{login_url}?{urlencode({'next': safe_next})}",
    )
    assert reverse("logout").encode() not in response.content


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("unsafe_next", "secure"),
    (
        (None, False),
        ("https://untrusted.example/opportunities?horizon=3y", False),
        ("//untrusted.example/opportunities?horizon=3y", False),
        ("http://testserver/opportunities?horizon=3y", True),
    ),
)
def test_login_csrf_recovery_drops_absent_or_unsafe_next(
    unsafe_next: str | None, secure: bool
) -> None:
    csrf_client = Client(enforce_csrf_checks=True)
    login_url = reverse("login")
    query = {} if unsafe_next is None else {"next": unsafe_next}
    request_url = login_url if unsafe_next is None else f"{login_url}?{urlencode(query)}"
    csrf_client.get(request_url, secure=secure)

    response = csrf_client.post(
        request_url,
        {
            "username": "submitted-username-is-not-echoed",
            "password": "submitted-password-is-not-echoed",
            "next": reverse("logout"),
        },
        secure=secure,
    )

    _assert_csrf_recovery(response, recovery_url=login_url)
    assert b"untrusted.example" not in response.content
    assert reverse("logout").encode() not in response.content


@pytest.mark.django_db
@pytest.mark.parametrize("debug", (False, True))
def test_csrf_recovery_never_echoes_submitted_or_technical_values(debug: bool) -> None:
    username = "submitted-username-is-not-echoed"
    password = "submitted-password-is-not-echoed"
    token = "submitted-token-is-not-echoed"
    reason = "technical-reason-is-not-echoed"
    csrf_client = Client(enforce_csrf_checks=True)

    with override_settings(DEBUG=debug):
        response = csrf_client.post(
            reverse("login"),
            {
                "csrfmiddlewaretoken": token,
                "username": username,
                "password": password,
            },
        )
        direct = csrf_failure(
            RequestFactory().post(reverse("login")),
            reason=reason,
        )

    for response_to_check in (response, direct):
        content = response_to_check.content.decode()
        for forbidden_value in (username, password, token, reason):
            assert forbidden_value not in content
        assert "<form" not in content
        assert "Sign out" not in content


@pytest.mark.django_db
def test_non_login_csrf_failures_only_offer_home_and_preserve_authentication() -> None:
    user = _create_user()
    csrf_client = Client(enforce_csrf_checks=True)
    csrf_client.force_login(user)
    failure_urls = (
        reverse("logout"),
        reverse("tracked-symbol-delete", args=[uuid4()]),
        "/admin/login/",
    )

    for failure_url in failure_urls:
        response = csrf_client.post(
            failure_url,
            {"csrfmiddlewaretoken": "invalid-csrf-token"},
        )

        _assert_csrf_recovery(response, recovery_url=reverse("index"))
        assert failure_url.encode() not in response.content
        assert b"primary-nav" not in response.content
        assert b"Sign out" not in response.content
        assert csrf_client.session["_auth_user_id"] == str(user.pk)


@pytest.mark.django_db
@override_settings(LOGIN_RATE_LIMIT_ATTEMPTS=1, LOGIN_RATE_LIMIT_WINDOW_SECONDS=60)
def test_multiple_csrf_rejections_do_not_consume_login_rate_limit_budget() -> None:
    cache.clear()
    csrf_client = Client(enforce_csrf_checks=True)
    login_url = reverse("login")
    page = csrf_client.get(login_url)
    token = _csrf_token(page)
    remote_address = "192.0.2.80"

    for _ in range(3):
        rejected = csrf_client.post(
            login_url,
            {
                "username": "owner",
                "password": "wrong-password",
            },
            REMOTE_ADDR=remote_address,
        )
        _assert_csrf_recovery(rejected, recovery_url=login_url)

    first_password_failure = csrf_client.post(
        login_url,
        {
            "csrfmiddlewaretoken": token,
            "username": "owner",
            "password": "wrong-password",
        },
        REMOTE_ADDR=remote_address,
    )
    second_password_failure = csrf_client.post(
        login_url,
        {
            "csrfmiddlewaretoken": token,
            "username": "owner",
            "password": "wrong-password",
        },
        REMOTE_ADDR=remote_address,
    )

    assert first_password_failure.status_code == 200
    assert second_password_failure.status_code == 429


@pytest.mark.django_db
def test_direct_csrf_failure_uses_no_queries_context_processors_or_lazy_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raising_context_processor(_request) -> dict[str, object]:
        raise AssertionError("CSRF failure rendering must not invoke context processors")

    def raising_user():
        raise AssertionError("CSRF failure rendering must not resolve request.user")

    monkeypatch.setattr(
        "stanstock.web.context_processors.stanstock_runtime",
        raising_context_processor,
    )
    template_settings = deepcopy(settings.TEMPLATES)
    request = RequestFactory().post(reverse("login"), {"next": "/opportunities"})
    request.user = SimpleLazyObject(raising_user)

    def raising_post_loader() -> None:
        raise AssertionError("CSRF failure rendering must not read request.POST")

    request._load_post_and_files = raising_post_loader

    with override_settings(TEMPLATES=template_settings):
        with CaptureQueriesContext(connection) as queries:
            response = csrf_failure(request, reason="do-not-render-this-reason")

    assert len(queries) == 0
    assert response.status_code == 403
    assert "no-store" in response.headers["Cache-Control"]

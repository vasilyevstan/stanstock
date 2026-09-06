from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import override_settings
from django.urls import reverse

from stanstock.data.models import ProviderRecord
from stanstock.data.provider_policy import BASIC_USAGE_SCOPE


@pytest.mark.django_db
def test_health_endpoint_reports_components(client) -> None:
    response = client.get(reverse("health"))

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert response.headers["Content-Security-Policy"].startswith("default-src 'self'")
    assert response.headers["Permissions-Policy"].startswith("camera=()")
    assert response.headers["X-Robots-Tag"] == "noindex, nofollow, noarchive"
    assert response.headers["Cache-Control"] == "no-store"
    assert {component["name"] for component in payload["components"]} == {
        "Database",
        "Data directory",
    }


@pytest.mark.django_db
def test_status_requires_authentication(client) -> None:
    response = client.get(reverse("status"))

    assert response.status_code == 302
    assert response.url.startswith(reverse("login"))


@pytest.mark.django_db
@override_settings(SECURE_SSL_REDIRECT=True)
def test_health_is_available_to_internal_container_probe_over_http(client) -> None:
    health = client.get(reverse("health"))
    protected_page = client.get(reverse("status"))

    assert health.status_code == 200
    assert protected_page.status_code == 301
    assert protected_page.url.startswith("https://")


@pytest.mark.django_db
def test_authenticated_status_shows_local_shell(client) -> None:
    user_model = get_user_model()
    user = user_model.objects.create_user(username="owner", password="safe-password")
    client.force_login(user)

    response = client.get(reverse("status"))

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    content = response.content.decode()
    assert "StanStock is running locally." in content
    assert "Synthetic research data." in content
    assert "No serving analysis" in content
    assert "DEMO-US" in content


@pytest.mark.django_db
def test_basic_provider_data_is_visible_only_to_licensed_user(client) -> None:
    user_model = get_user_model()
    owner = user_model.objects.create_user(username="owner", password="correct-password")
    other = user_model.objects.create_user(username="other", password="correct-password")
    ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=True,
        usage_scope=BASIC_USAGE_SCOPE,
        status="ok",
        metadata={
            "plan": "basic",
            "personal_noncommercial_confirmed": True,
            "licensed_user_id": str(owner.pk),
        },
    )

    client.force_login(owner)
    assert client.get(reverse("status")).status_code == 200

    client.force_login(other)
    response = client.get(reverse("status"))
    assert response.status_code == 403
    assert b"licensed for one personal user only" in response.content


@pytest.mark.django_db
def test_disabled_basic_provider_keeps_retained_data_owner_scoped(client) -> None:
    user_model = get_user_model()
    owner = user_model.objects.create_user(username="owner", password="correct-password")
    other = user_model.objects.create_user(username="other", password="correct-password")
    ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=False,
        usage_scope=BASIC_USAGE_SCOPE,
        status="disabled",
        metadata={
            "plan": "basic",
            "personal_noncommercial_confirmed": True,
            "licensed_user_id": str(owner.pk),
        },
    )

    client.force_login(other)
    assert client.get(reverse("status")).status_code == 403


@pytest.mark.django_db
def test_unlicensed_basic_user_can_log_out(client) -> None:
    user_model = get_user_model()
    owner = user_model.objects.create_user(username="owner", password="password")
    other = user_model.objects.create_user(username="other", password="password")
    ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=False,
        usage_scope=BASIC_USAGE_SCOPE,
        status="disabled",
        metadata={
            "plan": "basic",
            "personal_noncommercial_confirmed": True,
            "licensed_user_id": str(owner.pk),
        },
    )

    client.force_login(other)
    response = client.post(reverse("logout"))

    assert response.status_code == 302
    assert response.url == reverse("login")
    assert client.get(reverse("status")).status_code == 302


@pytest.mark.django_db
@override_settings(LOGIN_RATE_LIMIT_ATTEMPTS=2, LOGIN_RATE_LIMIT_WINDOW_SECONDS=60)
def test_login_is_rate_limited_after_repeated_failures(client) -> None:
    cache.clear()
    login_url = reverse("login")

    for _ in range(2):
        response = client.post(
            login_url,
            {"username": "owner", "password": "wrong"},
            REMOTE_ADDR="192.0.2.10",
        )
        assert response.status_code == 200

    blocked = client.post(
        login_url,
        {"username": "owner", "password": "wrong"},
        REMOTE_ADDR="192.0.2.10",
    )

    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "60"


@pytest.mark.django_db
@override_settings(LOGIN_RATE_LIMIT_ATTEMPTS=2, LOGIN_RATE_LIMIT_WINDOW_SECONDS=60)
def test_successful_login_clears_prior_failed_attempts(client) -> None:
    cache.clear()
    get_user_model().objects.create_user(username="owner", password="correct-password")
    login_url = reverse("login")
    remote_address = "192.0.2.20"

    client.post(
        login_url,
        {"username": "owner", "password": "wrong"},
        REMOTE_ADDR=remote_address,
    )
    success = client.post(
        login_url,
        {"username": "owner", "password": "correct-password"},
        REMOTE_ADDR=remote_address,
    )
    client.post(reverse("logout"))
    first_after_success = client.post(
        login_url,
        {"username": "owner", "password": "wrong"},
        REMOTE_ADDR=remote_address,
    )

    assert success.status_code == 302
    assert first_after_success.status_code == 200


@pytest.mark.django_db
@override_settings(
    LOGIN_RATE_LIMIT_ATTEMPTS=1,
    LOGIN_RATE_LIMIT_WINDOW_SECONDS=60,
    LOGIN_RATE_LIMIT_TRUSTED_PROXY_IPS={"192.0.2.1"},
)
def test_login_rate_limit_uses_forwarded_client_only_from_trusted_proxy(client) -> None:
    cache.clear()
    login_url = reverse("login")

    first_client = client.post(
        login_url,
        {"username": "owner", "password": "wrong"},
        REMOTE_ADDR="192.0.2.1",
        HTTP_X_FORWARDED_FOR="198.51.100.10",
    )
    second_client = client.post(
        login_url,
        {"username": "owner", "password": "wrong"},
        REMOTE_ADDR="192.0.2.1",
        HTTP_X_FORWARDED_FOR="198.51.100.11",
    )
    first_client_blocked = client.post(
        login_url,
        {"username": "owner", "password": "wrong"},
        REMOTE_ADDR="192.0.2.1",
        HTTP_X_FORWARDED_FOR="198.51.100.10",
    )

    assert first_client.status_code == 200
    assert second_client.status_code == 200
    assert first_client_blocked.status_code == 429

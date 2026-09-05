from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError


@pytest.mark.django_db
def test_bootstrap_owner_rejects_weak_password_without_creating_user(monkeypatch) -> None:
    monkeypatch.setenv("STANSTOCK_OWNER_USERNAME", "owner")
    monkeypatch.setenv("STANSTOCK_OWNER_PASSWORD", "12345678")

    with pytest.raises(CommandError):
        call_command("bootstrap_owner")

    assert not get_user_model().objects.filter(username="owner").exists()


@pytest.mark.django_db
def test_bootstrap_owner_creates_single_privileged_account(monkeypatch) -> None:
    monkeypatch.setenv("STANSTOCK_OWNER_USERNAME", "owner")
    monkeypatch.setenv("STANSTOCK_OWNER_PASSWORD", "local-test-passphrase-42")

    call_command("bootstrap_owner", verbosity=0)
    user = get_user_model().objects.get(username="owner")

    assert user.is_staff
    assert user.is_superuser
    assert user.check_password("local-test-passphrase-42")

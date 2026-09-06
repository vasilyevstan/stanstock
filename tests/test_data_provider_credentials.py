from __future__ import annotations

from types import SimpleNamespace

import pytest

from stanstock.data import provider_credentials
from stanstock.data.providers.exceptions import ProviderConfigurationError


def test_keychain_store_prompts_without_secret_process_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def run(args: list[str], **kwargs: object) -> SimpleNamespace:
        captured.extend(args)
        assert kwargs == {"check": False}
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(provider_credentials.sys, "platform", "darwin")
    monkeypatch.setattr(provider_credentials.subprocess, "run", run)

    provider_credentials.store_twelve_data_api_key()

    assert captured[-1] == "-w"
    assert "api-key" in captured
    assert all("private" not in value for value in captured)


def test_keychain_read_returns_none_when_item_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_credentials.sys, "platform", "darwin")
    monkeypatch.setattr(
        provider_credentials.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=44, stdout="", stderr=""),
    )

    assert provider_credentials.read_twelve_data_api_key() is None


def test_keychain_is_not_used_on_non_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_credentials.sys, "platform", "linux")

    assert provider_credentials.read_twelve_data_api_key() is None
    with pytest.raises(ProviderConfigurationError, match="macOS Keychain"):
        provider_credentials.store_twelve_data_api_key()

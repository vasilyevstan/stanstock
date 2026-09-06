from __future__ import annotations

import subprocess
import sys

from stanstock.data.providers.exceptions import ProviderConfigurationError

TWELVE_DATA_KEYCHAIN_SERVICE = "com.stanstock.twelve-data"
TWELVE_DATA_KEYCHAIN_ACCOUNT = "api-key"
KEYCHAIN_TIMEOUT_SECONDS = 5


def read_twelve_data_api_key() -> str | None:
    """Read the Twelve Data key from the current macOS user's login keychain."""
    if not _is_macos():
        return None
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                TWELVE_DATA_KEYCHAIN_ACCOUNT,
                "-s",
                TWELVE_DATA_KEYCHAIN_SERVICE,
                "-w",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=KEYCHAIN_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProviderConfigurationError(
            "The macOS Keychain could not be accessed for the Twelve Data credential."
        ) from exc
    if result.returncode == 44:
        return None
    if result.returncode != 0:
        raise ProviderConfigurationError(
            "The macOS Keychain rejected access to the Twelve Data credential."
        )
    value = result.stdout.strip()
    return value or None


def store_twelve_data_api_key() -> None:
    """Prompt securely and store the key without passing it through process arguments."""
    _require_macos_keychain()
    try:
        result = subprocess.run(
            [
                "security",
                "add-generic-password",
                "-U",
                "-a",
                TWELVE_DATA_KEYCHAIN_ACCOUNT,
                "-s",
                TWELVE_DATA_KEYCHAIN_SERVICE,
                "-l",
                "StanStock Twelve Data API key",
                "-w",
            ],
            check=False,
        )
    except OSError as exc:
        raise ProviderConfigurationError(
            "The macOS Keychain could not store the Twelve Data credential."
        ) from exc
    if result.returncode != 0:
        raise ProviderConfigurationError(
            "The macOS Keychain did not store the Twelve Data credential."
        )


def delete_twelve_data_api_key() -> bool:
    """Delete the keychain item, returning False when it did not exist."""
    _require_macos_keychain()
    try:
        result = subprocess.run(
            [
                "security",
                "delete-generic-password",
                "-a",
                TWELVE_DATA_KEYCHAIN_ACCOUNT,
                "-s",
                TWELVE_DATA_KEYCHAIN_SERVICE,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ProviderConfigurationError(
            "The macOS Keychain could not delete the Twelve Data credential."
        ) from exc
    if result.returncode == 44:
        return False
    if result.returncode != 0:
        raise ProviderConfigurationError(
            "The macOS Keychain did not delete the Twelve Data credential."
        )
    return True


def _require_macos_keychain() -> None:
    if not _is_macos():
        raise ProviderConfigurationError(
            "Local credential storage currently uses macOS Keychain. "
            "Use TWELVE_DATA_API_KEY in non-macOS environments."
        )


def _is_macos() -> bool:
    return sys.platform == "darwin"

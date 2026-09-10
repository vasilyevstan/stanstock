from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from django.core.exceptions import ImproperlyConfigured

from stanstock.settings.base import BASE_DIR, database_config


def _clear_sqlite_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("STANSTOCK_SQLITE_PATH", raising=False)


def test_local_sqlite_uses_wal_and_busy_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sqlite_env(monkeypatch)

    config = database_config()

    assert config["ENGINE"] == "django.db.backends.sqlite3"
    options = config["OPTIONS"]
    assert isinstance(options, dict)
    assert options["timeout"] == 20
    assert options["transaction_mode"] == "IMMEDIATE"
    assert "journal_mode=WAL" in str(options["init_command"])
    assert "busy_timeout=20000" in str(options["init_command"])


def test_default_sqlite_path_is_base_dir_database(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sqlite_env(monkeypatch)

    config = database_config()

    assert config["NAME"] == BASE_DIR / "stanstock.sqlite3"


def test_explicit_sqlite_path_is_used_and_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_sqlite_env(monkeypatch)
    target = tmp_path / "shared" / "stanstock.sqlite3"
    monkeypatch.setenv("STANSTOCK_SQLITE_PATH", str(target))

    config = database_config()

    assert config["ENGINE"] == "django.db.backends.sqlite3"
    assert config["NAME"] == target.resolve()
    options = config["OPTIONS"]
    assert isinstance(options, dict)
    assert "journal_mode=WAL" in str(options["init_command"])
    assert "busy_timeout=20000" in str(options["init_command"])


def test_explicit_sqlite_path_expands_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_sqlite_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("STANSTOCK_SQLITE_PATH", "~/shared/stanstock.sqlite3")

    config = database_config()

    assert config["NAME"] == (tmp_path / "shared" / "stanstock.sqlite3").resolve()


def test_relative_sqlite_path_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sqlite_env(monkeypatch)
    monkeypatch.setenv("STANSTOCK_SQLITE_PATH", "relative/stanstock.sqlite3")

    with pytest.raises(ImproperlyConfigured, match="absolute path"):
        database_config()


def test_database_url_and_sqlite_path_together_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_sqlite_env(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@localhost:5432/stanstock")
    monkeypatch.setenv("STANSTOCK_SQLITE_PATH", str(tmp_path / "stanstock.sqlite3"))

    with pytest.raises(ImproperlyConfigured, match="cannot both be set"):
        database_config()


def test_blank_sqlite_path_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_sqlite_env(monkeypatch)
    monkeypatch.setenv("STANSTOCK_SQLITE_PATH", "   ")

    config = database_config()

    assert config["NAME"] == BASE_DIR / "stanstock.sqlite3"


def test_production_rejects_whitespace_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "   ")

    with pytest.raises(
        ImproperlyConfigured,
        match="DATABASE_URL is required for production settings",
    ):
        importlib.import_module("stanstock.settings.prod")

from __future__ import annotations

import pytest

from stanstock.settings.base import database_config


def test_local_sqlite_uses_wal_and_busy_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    config = database_config()

    assert config["ENGINE"] == "django.db.backends.sqlite3"
    options = config["OPTIONS"]
    assert isinstance(options, dict)
    assert options["timeout"] == 20
    assert options["transaction_mode"] == "IMMEDIATE"
    assert "journal_mode=WAL" in str(options["init_command"])
    assert "busy_timeout=20000" in str(options["init_command"])

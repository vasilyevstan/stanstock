from __future__ import annotations

import os
import plistlib
from datetime import date
from pathlib import Path

import pytest

from stanstock.core import scheduled_refresh_entrypoint
from stanstock.core.launchd import (
    LAUNCH_AGENT_LABEL,
    SCHEDULED_REFRESH_MODULE,
    detect_iana_timezone,
    install_launch_agent,
    launch_agent_status,
    validate_schedule,
)


def test_machine_timezone_takes_precedence_over_django_tz_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zoneinfo_target = tmp_path / "zoneinfo" / "Europe" / "Tallinn"
    zoneinfo_target.parent.mkdir(parents=True)
    zoneinfo_target.touch()
    localtime = tmp_path / "localtime"
    localtime.symlink_to(zoneinfo_target)
    monkeypatch.setenv("TZ", "UTC")

    assert detect_iana_timezone(localtime_path=localtime) == "Europe/Tallinn"


def test_two_am_schedule_is_safe_for_supported_us_timezones() -> None:
    eastern = validate_schedule(
        "America/New_York",
        start_date=date(2026, 1, 1),
    )
    pacific = validate_schedule(
        "America/Los_Angeles",
        start_date=date(2026, 1, 1),
    )

    assert eastern.checked_invocations > 250
    assert eastern.regular_close_checked is True
    assert eastern.early_close_checked is True
    assert eastern.new_york_offset_changes >= 1
    assert pacific.local_offset_changes >= 1


def test_two_am_schedule_is_rejected_when_it_runs_during_xnys_hours() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        validate_schedule(
            "Asia/Tokyo",
            start_date=date(2026, 1, 1),
        )


def test_launch_agent_installation_keeps_env_values_out_of_plist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("stanstock.core.launchd.sys.platform", "darwin")
    monkeypatch.setattr("stanstock.core.launchd._is_loaded", lambda: False)
    project_root = tmp_path / "stanstock"
    interpreter = project_root / ".venv" / "bin" / "python"
    entrypoint = project_root / "src" / "stanstock" / "core" / "scheduled_refresh_entrypoint.py"
    env_file = project_root / ".env"
    interpreter.parent.mkdir(parents=True)
    entrypoint.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    entrypoint.write_text("", encoding="utf-8")
    env_file.write_text("TWELVE_DATA_API_KEY=private-test-value\n", encoding="utf-8")
    interpreter.chmod(0o755)
    env_file.chmod(0o600)
    home = tmp_path / "home"

    plist_path, validation = install_launch_agent(
        project_root=project_root,
        timezone_name="America/New_York",
        home=home,
        load=False,
    )

    payload_bytes = plist_path.read_bytes()
    with plist_path.open("rb") as handle:
        payload = plistlib.load(handle)
    assert validation.timezone == "America/New_York"
    assert payload["Label"] == LAUNCH_AGENT_LABEL
    assert payload["Program"] == str(interpreter)
    assert payload["ProgramArguments"] == [
        str(interpreter),
        "-m",
        SCHEDULED_REFRESH_MODULE,
    ]
    assert [entry["Weekday"] for entry in payload["StartCalendarInterval"]] == [
        2,
        3,
        4,
        5,
        6,
    ]
    assert payload["EnvironmentVariables"]["STANSTOCK_ENV_FILE"] == str(env_file)
    assert payload["EnvironmentVariables"]["STANSTOCK_DISABLE_KEYCHAIN"] == "1"
    assert b"private-test-value" not in payload_bytes
    assert os.stat(plist_path).st_mode & 0o077 == 0

    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    status = launch_agent_status(home)
    assert status["installed"] is True
    assert status["expected_timezone"] == "America/New_York"
    assert status["timezone_matches"] is True


def test_launch_agent_rejects_env_file_visible_to_other_users(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("stanstock.core.launchd.sys.platform", "darwin")
    project_root = tmp_path / "stanstock"
    interpreter = project_root / ".venv" / "bin" / "python"
    entrypoint = project_root / "src" / "stanstock" / "core" / "scheduled_refresh_entrypoint.py"
    env_file = project_root / ".env"
    interpreter.parent.mkdir(parents=True)
    entrypoint.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    entrypoint.write_text("", encoding="utf-8")
    env_file.write_text("TWELVE_DATA_API_KEY=test\n", encoding="utf-8")
    interpreter.chmod(0o755)
    env_file.chmod(0o644)

    with pytest.raises(ValueError, match="group or other users"):
        install_launch_agent(
            project_root=project_root,
            timezone_name="America/New_York",
            home=tmp_path / "home",
            load=False,
        )


def test_application_entrypoint_loads_private_environment_before_django(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                'DJANGO_SETTINGS_MODULE="example.settings"',
                'SEC_USER_AGENT="StanStock test monitored@example.invalid"',
                "STANSTOCK_SCHEDULE_TIMEZONE=wrong-zone",
                "STANSTOCK_DISABLE_KEYCHAIN=0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env_file.chmod(0o600)
    calls: list[str] = []
    monkeypatch.setenv(scheduled_refresh_entrypoint.ENV_FILE_ENV, str(env_file))
    monkeypatch.setenv("STANSTOCK_SCHEDULE_TIMEZONE", "Europe/Tallinn")
    monkeypatch.setattr(
        scheduled_refresh_entrypoint,
        "_execute_scheduled_refresh",
        lambda: calls.append("executed"),
    )

    scheduled_refresh_entrypoint.main()

    assert calls == ["executed"]
    assert os.environ["DJANGO_SETTINGS_MODULE"] == "example.settings"
    assert os.environ["SEC_USER_AGENT"] == "StanStock test monitored@example.invalid"
    assert os.environ["STANSTOCK_SCHEDULE_TIMEZONE"] == "Europe/Tallinn"
    assert os.environ["STANSTOCK_DISABLE_KEYCHAIN"] == "1"


def test_application_entrypoint_rejects_unquoted_environment_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SEC_USER_AGENT=unquoted value\n", encoding="utf-8")
    env_file.chmod(0o600)
    monkeypatch.setenv(scheduled_refresh_entrypoint.ENV_FILE_ENV, str(env_file))

    with pytest.raises(SystemExit, match="unquoted value on line 1"):
        scheduled_refresh_entrypoint.main()

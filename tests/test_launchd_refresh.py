from __future__ import annotations

import os
import plistlib
from datetime import date
from pathlib import Path

import pytest

from stanstock.core import scheduled_refresh_entrypoint
from stanstock.core.launchd import (
    LAUNCH_AGENT_LABEL,
    SCHEDULE_HOUR,
    SCHEDULE_MINUTE,
    SCHEDULE_TIME_LABEL,
    SCHEDULED_REFRESH_MODULE,
    VALIDATION_DAYS,
    detect_iana_timezone,
    install_launch_agent,
    launch_agent_paths,
    launch_agent_status,
    validate_schedule,
)


def test_schedule_constant_is_zero_three_thirty() -> None:
    assert SCHEDULE_HOUR == 3
    assert SCHEDULE_MINUTE == 30
    assert SCHEDULE_TIME_LABEL == "03:30"
    assert VALIDATION_DAYS >= 370


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


def test_scheduled_time_is_safe_for_supported_us_timezones() -> None:
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


def test_scheduled_time_is_rejected_when_it_runs_during_xnys_hours() -> None:
    with pytest.raises(ValueError, match="unsafe") as excinfo:
        validate_schedule(
            "Asia/Tokyo",
            start_date=date(2026, 1, 1),
        )
    assert SCHEDULE_TIME_LABEL in str(excinfo.value)


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
    assert {entry["Hour"] for entry in payload["StartCalendarInterval"]} == {SCHEDULE_HOUR}
    assert {entry["Minute"] for entry in payload["StartCalendarInterval"]} == {SCHEDULE_MINUTE}
    assert payload["StanStockSchedule"]["hour"] == SCHEDULE_HOUR
    assert payload["StanStockSchedule"]["minute"] == SCHEDULE_MINUTE
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
    assert status["installed_schedule_label"] == f"{SCHEDULE_TIME_LABEL} (Tuesday-Saturday)"
    assert status["expected_schedule_label"] == SCHEDULE_TIME_LABEL
    assert status["schedule_matches"] is True


def _write_plist(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(payload, handle)


def _trigger(weekday: int, hour: int, minute: int) -> dict[str, object]:
    return {"Weekday": weekday, "Hour": hour, "Minute": minute}


def _healthy_triggers() -> list[dict[str, object]]:
    return [_trigger(weekday, SCHEDULE_HOUR, SCHEDULE_MINUTE) for weekday in (2, 3, 4, 5, 6)]


# --- F1: StartCalendarInterval is the sole authority for execution -------


def test_launch_agent_status_never_certifies_from_metadata_when_triggers_are_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    stale_triggers = [_trigger(weekday, 2, 0) for weekday in (2, 3, 4, 5, 6)]
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StartCalendarInterval": stale_triggers,
            "StanStockSchedule": {
                "timezone": "America/New_York",
                "hour": SCHEDULE_HOUR,
                "minute": SCHEDULE_MINUTE,
                "weekdays": [2, 3, 4, 5, 6],
            },
            "EnvironmentVariables": {"STANSTOCK_SCHEDULE_TIMEZONE": "America/New_York"},
        },
    )

    status = launch_agent_status(home)

    assert status["installed"] is True
    assert status["installed_schedule_label"] == "02:00 (Tuesday-Saturday)"
    assert status["schedule_matches"] is False


def test_launch_agent_status_treats_missing_triggers_as_unmatched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StanStockSchedule": {
                "timezone": "America/New_York",
                "hour": SCHEDULE_HOUR,
                "minute": SCHEDULE_MINUTE,
                "weekdays": [2, 3, 4, 5, 6],
            },
        },
    )

    status = launch_agent_status(home)

    assert status["installed"] is True
    assert status["installed_schedule_label"] is None
    assert status["schedule_matches"] is False


def test_launch_agent_status_treats_malformed_triggers_as_unmatched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StartCalendarInterval": [
                {"Weekday": 2, "Hour": "three", "Minute": 30},
                *[_trigger(weekday, SCHEDULE_HOUR, SCHEDULE_MINUTE) for weekday in (3, 4, 5, 6)],
            ],
        },
    )

    status = launch_agent_status(home)

    assert status["installed"] is True
    assert status["installed_schedule_label"] is None
    assert status["schedule_matches"] is False


def test_launch_agent_status_treats_duplicate_triggers_as_unmatched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StartCalendarInterval": [
                *_healthy_triggers(),
                _trigger(2, SCHEDULE_HOUR, SCHEDULE_MINUTE),
            ],
        },
    )

    status = launch_agent_status(home)

    assert status["installed"] is True
    assert status["installed_schedule_label"] is None
    assert status["schedule_matches"] is False


def test_launch_agent_status_treats_extra_weekday_triggers_as_unmatched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StartCalendarInterval": [
                *_healthy_triggers(),
                _trigger(0, SCHEDULE_HOUR, SCHEDULE_MINUTE),
            ],
        },
    )

    status = launch_agent_status(home)

    assert status["installed"] is True
    assert status["installed_schedule_label"] != f"{SCHEDULE_TIME_LABEL} (Tuesday-Saturday)"
    assert status["schedule_matches"] is False


@pytest.mark.parametrize("extra_key", ["Month", "Day", "AnUnexpectedTriggerKey"])
def test_launch_agent_status_treats_extra_calendar_keys_as_unmatched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_key: str,
) -> None:
    """F4: any key beyond Weekday/Hour/Minute changes when launchd actually
    fires (e.g. Month=1 restricts execution to January), so it must never be
    silently ignored or rendered as a normal healthy Tue-Sat schedule.
    """
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    triggers = _healthy_triggers()
    triggers[0] = {**triggers[0], extra_key: 1}
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StartCalendarInterval": triggers,
        },
    )

    status = launch_agent_status(home)

    assert status["installed"] is True
    assert status["installed_schedule_label"] is None
    assert status["schedule_matches"] is False


def test_launch_agent_status_treats_mixed_time_triggers_as_unmatched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StartCalendarInterval": [
                _trigger(2, SCHEDULE_HOUR, SCHEDULE_MINUTE),
                *[_trigger(weekday, 2, 0) for weekday in (3, 4, 5, 6)],
            ],
        },
    )

    status = launch_agent_status(home)

    assert status["installed"] is True
    assert status["installed_schedule_label"] is None
    assert status["schedule_matches"] is False


def test_launch_agent_status_treats_metadata_inconsistent_with_triggers_as_unmatched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StartCalendarInterval": _healthy_triggers(),
            "StanStockSchedule": {
                "timezone": "America/New_York",
                "hour": 2,
                "minute": 0,
                "weekdays": [2, 3, 4, 5, 6],
            },
        },
    )

    status = launch_agent_status(home)

    assert status["installed"] is True
    assert status["installed_schedule_label"] == f"{SCHEDULE_TIME_LABEL} (Tuesday-Saturday)"
    assert status["schedule_matches"] is False


# --- F3: never claim Tuesday-Saturday for a non-Tue-Sat trigger set -------


def test_launch_agent_status_describes_a_tuesday_only_trigger_without_claiming_tue_sat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StartCalendarInterval": [_trigger(2, SCHEDULE_HOUR, SCHEDULE_MINUTE)],
        },
    )

    status = launch_agent_status(home)

    assert status["installed"] is True
    assert status["installed_schedule_label"] == f"{SCHEDULE_TIME_LABEL} (Tuesday)"
    assert "Tuesday-Saturday" not in str(status["installed_schedule_label"])
    assert status["schedule_matches"] is False


def test_launch_agent_status_is_unmatched_when_not_installed(tmp_path: Path) -> None:
    status = launch_agent_status(tmp_path / "home")

    assert status["installed"] is False
    assert status["installed_schedule_label"] is None
    assert status["schedule_matches"] is False


# --- F2: timezone requires an exact three-way match -----------------------


def test_launch_agent_status_requires_metadata_timezone_to_be_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StartCalendarInterval": _healthy_triggers(),
        },
    )

    status = launch_agent_status(home)

    assert status["expected_timezone"] is None
    assert status["timezone_matches"] is False


@pytest.mark.parametrize("bad_timezone", [True, 7, ["America/New_York"], {"tz": "x"}, "   "])
def test_launch_agent_status_rejects_a_malformed_timezone_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_timezone: object,
) -> None:
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StartCalendarInterval": _healthy_triggers(),
            "StanStockSchedule": {"timezone": bad_timezone},
            "EnvironmentVariables": {"STANSTOCK_SCHEDULE_TIMEZONE": "America/New_York"},
        },
    )

    status = launch_agent_status(home)

    assert status["timezone_matches"] is False


def test_launch_agent_status_rejects_a_changed_machine_timezone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "Europe/Tallinn",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StartCalendarInterval": _healthy_triggers(),
            "StanStockSchedule": {"timezone": "America/New_York"},
            "EnvironmentVariables": {"STANSTOCK_SCHEDULE_TIMEZONE": "America/New_York"},
        },
    )

    status = launch_agent_status(home)

    assert status["expected_timezone"] == "America/New_York"
    assert status["current_timezone"] == "Europe/Tallinn"
    assert status["timezone_matches"] is False


def test_launch_agent_status_rejects_a_metadata_runtime_timezone_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StartCalendarInterval": _healthy_triggers(),
            "StanStockSchedule": {"timezone": "America/New_York"},
            "EnvironmentVariables": {"STANSTOCK_SCHEDULE_TIMEZONE": "America/Los_Angeles"},
        },
    )

    status = launch_agent_status(home)

    assert status["timezone_matches"] is False


def test_launch_agent_status_matches_timezone_when_fully_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    _write_plist(
        plist_path,
        {
            "Label": LAUNCH_AGENT_LABEL,
            "StartCalendarInterval": _healthy_triggers(),
            "StanStockSchedule": {
                "timezone": "America/New_York",
                "hour": SCHEDULE_HOUR,
                "minute": SCHEDULE_MINUTE,
                "weekdays": [2, 3, 4, 5, 6],
            },
            "EnvironmentVariables": {"STANSTOCK_SCHEDULE_TIMEZONE": "America/New_York"},
        },
    )

    status = launch_agent_status(home)

    assert status["timezone_matches"] is True
    assert status["schedule_matches"] is True


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

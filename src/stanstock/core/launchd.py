from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.environment import validate_private_environment_file
from stanstock.data.live_us import DEFAULT_CLOSE_DELAY_MINUTES

LAUNCH_AGENT_LABEL = "com.stanstock.daily-refresh"
LAUNCH_AGENT_FILENAME = f"{LAUNCH_AGENT_LABEL}.plist"
SCHEDULED_REFRESH_MODULE = "stanstock.core.scheduled_refresh_entrypoint"
SCHEDULED_WEEKDAYS = frozenset({1, 2, 3, 4, 5})
LAUNCHD_WEEKDAYS = (2, 3, 4, 5, 6)
SCHEDULE_HOUR = 3
SCHEDULE_MINUTE = 30
SCHEDULE_TIME_LABEL = f"{SCHEDULE_HOUR:02d}:{SCHEDULE_MINUTE:02d}"
VALIDATION_DAYS = 400
_REQUIRED_TRIGGER_KEYS = frozenset({"Weekday", "Hour", "Minute"})


@dataclass(frozen=True, slots=True)
class ScheduleValidation:
    timezone: str
    checked_invocations: int
    first_date: str
    last_date: str
    regular_close_checked: bool
    early_close_checked: bool
    local_offset_changes: int
    new_york_offset_changes: int


def detect_iana_timezone(
    *,
    localtime_path: Path = Path("/etc/localtime"),
) -> str:
    candidates: list[str] = []
    if localtime_path.exists():
        resolved = str(localtime_path.resolve())
        marker = "zoneinfo/"
        if marker in resolved:
            candidates.append(resolved.split(marker, maxsplit=1)[1])
    timezone_key = getattr(datetime.now().astimezone().tzinfo, "key", None)
    if isinstance(timezone_key, str):
        candidates.append(timezone_key)
    configured = os.environ.get("TZ", "").strip()
    if configured:
        candidates.append(configured)

    for candidate in candidates:
        try:
            ZoneInfo(candidate)
        except ZoneInfoNotFoundError:
            continue
        return candidate
    raise ValueError(
        "The machine IANA timezone could not be detected. Set TZ to an IANA "
        "name such as America/New_York before installing the LaunchAgent."
    )


def validate_schedule(
    timezone_name: str,
    *,
    start_date: date | None = None,
    days: int = VALIDATION_DAYS,
) -> ScheduleValidation:
    if days < 370:
        raise ValueError("Schedule validation must cover at least 370 calendar days")
    try:
        local_zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown IANA timezone: {timezone_name}") from exc

    first = start_date or datetime.now(tz=local_zone).date()
    calendar = get_calendar(
        "XNYS",
        start=first - timedelta(days=10),
        end=first + timedelta(days=days + 10),
    )
    new_york = ZoneInfo("America/New_York")
    checked = 0
    regular_close_checked = False
    early_close_checked = False
    local_offsets: set[timedelta | None] = set()
    new_york_offsets: set[timedelta | None] = set()
    last_checked = first

    for offset in range(days):
        local_date = first + timedelta(days=offset)
        if local_date.weekday() not in SCHEDULED_WEEKDAYS:
            continue
        scheduled_local = datetime.combine(
            local_date,
            time(hour=SCHEDULE_HOUR, minute=SCHEDULE_MINUTE),
            tzinfo=local_zone,
        )
        round_trip = scheduled_local.astimezone(UTC).astimezone(local_zone)
        if round_trip.replace(fold=scheduled_local.fold) != scheduled_local:
            raise ValueError(
                f"{SCHEDULE_TIME_LABEL} does not exist or is ambiguous in "
                f"{timezone_name} on {local_date.isoformat()}"
            )
        scheduled_utc = scheduled_local.astimezone(UTC)
        target_session = calendar.date_to_session(
            scheduled_utc.date(),
            direction="previous",
        )
        candidate_ready_at = (
            calendar.session_close(target_session) + timedelta(minutes=DEFAULT_CLOSE_DELAY_MINUTES)
        ).to_pydatetime()
        if scheduled_utc < candidate_ready_at:
            target_session = calendar.previous_session(target_session)
        ready_at = (
            calendar.session_close(target_session) + timedelta(minutes=DEFAULT_CLOSE_DELAY_MINUTES)
        ).to_pydatetime()
        next_open = calendar.session_open(calendar.next_session(target_session)).to_pydatetime()
        if scheduled_utc < ready_at or scheduled_utc >= next_open:
            raise ValueError(
                f"{SCHEDULE_TIME_LABEL} {timezone_name} is unsafe on "
                f"{local_date.isoformat()}: the invocation must be after the "
                "completed XNYS close publication delay and before the next "
                "XNYS session opens."
            )

        close_local = ready_at.astimezone(new_york) - timedelta(minutes=DEFAULT_CLOSE_DELAY_MINUTES)
        regular_close_checked |= close_local.hour == 16
        early_close_checked |= close_local.hour == 13
        local_offsets.add(scheduled_local.utcoffset())
        new_york_offsets.add(scheduled_utc.astimezone(new_york).utcoffset())
        checked += 1
        last_checked = local_date

    if not regular_close_checked or not early_close_checked:
        raise ValueError("Schedule validation did not cover both regular and early XNYS closes")
    return ScheduleValidation(
        timezone=timezone_name,
        checked_invocations=checked,
        first_date=first.isoformat(),
        last_date=last_checked.isoformat(),
        regular_close_checked=regular_close_checked,
        early_close_checked=early_close_checked,
        local_offset_changes=max(0, len(local_offsets) - 1),
        new_york_offset_changes=max(0, len(new_york_offsets) - 1),
    )


def launch_agent_paths(home: Path | None = None) -> tuple[Path, Path, Path]:
    user_home = home or Path.home()
    plist_path = user_home / "Library" / "LaunchAgents" / LAUNCH_AGENT_FILENAME
    log_dir = user_home / "Library" / "Logs" / "StanStock"
    return plist_path, log_dir / "scheduled-refresh.log", log_dir / "scheduled-refresh.error.log"


def build_launch_agent(
    *,
    project_root: Path,
    timezone_name: str,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, object]:
    interpreter = project_root / ".venv" / "bin" / "python"
    env_file = project_root / ".env"
    return {
        "Label": LAUNCH_AGENT_LABEL,
        "Program": str(interpreter),
        "ProgramArguments": [
            str(interpreter),
            "-m",
            SCHEDULED_REFRESH_MODULE,
        ],
        "WorkingDirectory": str(project_root),
        "StartCalendarInterval": [
            {
                "Weekday": weekday,
                "Hour": SCHEDULE_HOUR,
                "Minute": SCHEDULE_MINUTE,
            }
            for weekday in LAUNCHD_WEEKDAYS
        ],
        "EnvironmentVariables": {
            "STANSTOCK_ENV_FILE": str(env_file),
            "STANSTOCK_SCHEDULE_TIMEZONE": timezone_name,
            "STANSTOCK_DISABLE_KEYCHAIN": "1",
            "PYTHONUNBUFFERED": "1",
        },
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
        "ProcessType": "Background",
        "LowPriorityIO": True,
        "RunAtLoad": False,
        "StanStockSchedule": {
            "timezone": timezone_name,
            "hour": SCHEDULE_HOUR,
            "minute": SCHEDULE_MINUTE,
            "weekdays": list(LAUNCHD_WEEKDAYS),
        },
    }


def install_launch_agent(
    *,
    project_root: Path,
    timezone_name: str,
    home: Path | None = None,
    load: bool = True,
) -> tuple[Path, ScheduleValidation]:
    _require_macos()
    root = project_root.resolve()
    interpreter = root / ".venv" / "bin" / "python"
    entrypoint = root / "src" / "stanstock" / "core" / "scheduled_refresh_entrypoint.py"
    env_file = root / ".env"
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError(f"Project interpreter is not executable: {interpreter}")
    if not entrypoint.is_file():
        raise ValueError("The scheduled refresh application entrypoint is missing")
    validate_private_environment_file(env_file)
    validation = validate_schedule(timezone_name)
    plist_path, stdout_path, stderr_path = launch_agent_paths(home)
    plist_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stdout_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(stdout_path.parent, 0o700)
    payload = build_launch_agent(
        project_root=root,
        timezone_name=timezone_name,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    _write_private_plist(plist_path, payload)
    if load:
        _replace_loaded_agent(plist_path)
    return plist_path, validation


def uninstall_launch_agent(
    *,
    home: Path | None = None,
    unload: bool = True,
) -> bool:
    _require_macos()
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    if unload:
        _bootout(plist_path)
    if not plist_path.exists():
        return False
    plist_path.unlink()
    return True


def launch_agent_status(home: Path | None = None) -> dict[str, object]:
    plist_path, stdout_path, stderr_path = launch_agent_paths(home)
    current_timezone = detect_iana_timezone()
    installed = plist_path.is_file()
    payload: object = None
    if installed:
        with plist_path.open("rb") as handle:
            payload = plistlib.load(handle)
    expected_timezone, timezone_matches = (
        _derive_timezone_status(payload, current_timezone=current_timezone)
        if installed
        else (None, False)
    )
    schedule_metadata = payload.get("StanStockSchedule") if isinstance(payload, dict) else None
    installed_schedule_label, schedule_matches = (
        _derive_installed_schedule(payload) if installed else (None, False)
    )
    return {
        "installed": installed,
        "loaded": installed and _is_loaded() if sys.platform == "darwin" else False,
        "plist_path": str(plist_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "current_timezone": current_timezone,
        "expected_timezone": expected_timezone,
        "timezone_matches": timezone_matches,
        "schedule": schedule_metadata if isinstance(schedule_metadata, dict) else None,
        "installed_schedule_label": installed_schedule_label,
        "expected_schedule_label": SCHEDULE_TIME_LABEL,
        "schedule_matches": schedule_matches,
    }


_WEEKDAY_NAMES = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}


def _as_plain_int(value: object) -> int | None:
    """Return ``value`` as an int, rejecting bools (which are also ints)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _describe_weekdays(weekdays: list[int]) -> str:
    ordered = sorted(set(weekdays))
    return ", ".join(_WEEKDAY_NAMES.get(day, f"weekday {day}") for day in ordered)


def _metadata_consistent_with_triggers(
    metadata: object,
    *,
    hour: int,
    minute: int,
    weekday_set: set[int],
    weekday_count: int,
) -> bool:
    """Metadata is an optional display/consistency aid, never a certification.

    Absence of metadata (or of a specific field within it) is not a failure;
    a present-but-wrong-typed or present-but-mismatched value is.
    """
    if not isinstance(metadata, dict):
        return True
    if "hour" in metadata or "minute" in metadata:
        raw_hour = _as_plain_int(metadata.get("hour"))
        raw_minute = _as_plain_int(metadata.get("minute"))
        if raw_hour is None or raw_minute is None or raw_hour != hour or raw_minute != minute:
            return False
    if "weekdays" in metadata:
        raw_weekdays = metadata.get("weekdays")
        if not isinstance(raw_weekdays, list) or not all(
            _as_plain_int(day) is not None for day in raw_weekdays
        ):
            return False
        if len(raw_weekdays) != weekday_count or set(raw_weekdays) != weekday_set:
            return False
    return True


def _derive_installed_schedule(payload: object) -> tuple[str | None, bool]:
    """Derive the actually-installed schedule from the authoritative
    ``StartCalendarInterval`` trigger array -- never from the
    ``StanStockSchedule`` display metadata, which cannot certify execution.

    Requires exactly one trigger per weekday, all sharing one hour/minute;
    missing, malformed, duplicate, extra, or mixed-time triggers -- or a
    ``StanStockSchedule`` metadata block that disagrees with the derived
    triggers -- all yield ``schedule_matches=False``. Never raises.
    """
    if not isinstance(payload, dict):
        return None, False
    triggers = payload.get("StartCalendarInterval")
    if not isinstance(triggers, list) or not triggers:
        return None, False

    weekdays: list[int] = []
    hours: set[int] = set()
    minutes: set[int] = set()
    for entry in triggers:
        if not isinstance(entry, dict):
            return None, False
        if set(entry.keys()) != _REQUIRED_TRIGGER_KEYS:
            # Any extra executable calendar key (e.g. Month, Day) changes when
            # launchd actually fires; only an exact Weekday/Hour/Minute
            # trigger can be safely described or certified.
            return None, False
        weekday = _as_plain_int(entry.get("Weekday"))
        hour = _as_plain_int(entry.get("Hour"))
        minute = _as_plain_int(entry.get("Minute"))
        if weekday is None or hour is None or minute is None:
            return None, False
        if not (0 <= weekday <= 7) or not (0 <= hour <= 23) or not (0 <= minute <= 59):
            return None, False
        weekdays.append(0 if weekday == 7 else weekday)
        hours.add(hour)
        minutes.add(minute)

    if len(weekdays) != len(set(weekdays)):
        return None, False  # duplicate weekday trigger: ambiguous execution
    if len(hours) != 1 or len(minutes) != 1:
        return None, False  # mixed-time triggers: no single safe label

    hour = hours.pop()
    minute = minutes.pop()
    time_label = f"{hour:02d}:{minute:02d}"
    weekday_set = set(weekdays)
    exact_weekdays = weekday_set == set(LAUNCHD_WEEKDAYS) and len(weekdays) == len(LAUNCHD_WEEKDAYS)
    weekday_description = "Tuesday-Saturday" if exact_weekdays else _describe_weekdays(weekdays)
    label = f"{time_label} ({weekday_description})"

    triggers_match = exact_weekdays and hour == SCHEDULE_HOUR and minute == SCHEDULE_MINUTE
    metadata_consistent = _metadata_consistent_with_triggers(
        payload.get("StanStockSchedule"),
        hour=hour,
        minute=minute,
        weekday_set=weekday_set,
        weekday_count=len(weekdays),
    )
    return label, triggers_match and metadata_consistent


def _derive_timezone_status(payload: object, *, current_timezone: str) -> tuple[str | None, bool]:
    """Require an exact three-way match: metadata timezone, the runtime
    ``STANSTOCK_SCHEDULE_TIMEZONE`` environment value, and the machine's
    currently detected timezone. Missing, blank, non-string, or conflicting
    values fail closed (``timezone_matches=False``) rather than crashing or
    defaulting to a match.
    """
    if not isinstance(payload, dict):
        return None, False
    metadata = payload.get("StanStockSchedule")
    if not isinstance(metadata, dict):
        return None, False
    raw_timezone = metadata.get("timezone")
    if not isinstance(raw_timezone, str) or not raw_timezone.strip():
        return None, False
    expected_timezone = raw_timezone

    environment = payload.get("EnvironmentVariables")
    runtime_timezone = (
        environment.get("STANSTOCK_SCHEDULE_TIMEZONE") if isinstance(environment, dict) else None
    )
    if not isinstance(runtime_timezone, str) or not runtime_timezone.strip():
        return expected_timezone, False
    if runtime_timezone != expected_timezone:
        return expected_timezone, False

    return expected_timezone, expected_timezone == current_timezone


def validation_details(validation: ScheduleValidation) -> dict[str, object]:
    return asdict(validation)


def _write_private_plist(path: Path, payload: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        plistlib.dump(payload, handle, sort_keys=True)
    try:
        os.chmod(temporary, 0o600)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_loaded_agent(plist_path: Path) -> None:
    _bootout(plist_path)
    result = _launchctl("bootstrap", _launchd_domain(), str(plist_path))
    if result.returncode != 0:
        raise ValueError("launchctl could not load the StanStock LaunchAgent")


def _bootout(plist_path: Path) -> None:
    _launchctl("bootout", _launchd_domain(), str(plist_path))


def _is_loaded() -> bool:
    return _launchctl("print", f"{_launchd_domain()}/{LAUNCH_AGENT_LABEL}").returncode == 0


def _launchd_domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["launchctl", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("launchctl could not be executed") from exc


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise ValueError("LaunchAgent management is available only on macOS")

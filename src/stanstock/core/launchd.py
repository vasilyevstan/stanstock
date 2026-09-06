from __future__ import annotations

import os
import plistlib
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.data.live_us import DEFAULT_CLOSE_DELAY_MINUTES

LAUNCH_AGENT_LABEL = "com.stanstock.daily-refresh"
LAUNCH_AGENT_FILENAME = f"{LAUNCH_AGENT_LABEL}.plist"
SCHEDULED_WEEKDAYS = frozenset({1, 2, 3, 4, 5})
LAUNCHD_WEEKDAYS = (2, 3, 4, 5, 6)
SCHEDULE_HOUR = 2
SCHEDULE_MINUTE = 0
VALIDATION_DAYS = 400


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
                f"02:00 does not exist or is ambiguous in {timezone_name} on "
                f"{local_date.isoformat()}"
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
                f"02:00 {timezone_name} is unsafe on {local_date.isoformat()}: "
                "the invocation must be after the completed XNYS close publication "
                "delay and before the next XNYS session opens."
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
    runner = project_root / "scripts" / "run-scheduled-refresh.sh"
    return {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": [str(runner)],
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
    runner = root / "scripts" / "run-scheduled-refresh.sh"
    env_file = root / ".env"
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError(f"Project interpreter is not executable: {interpreter}")
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise ValueError(f"Scheduled refresh runner is not executable: {runner}")
    _validate_private_env_file(env_file)
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
    expected_timezone: str | None = None
    schedule: object = None
    if installed:
        with plist_path.open("rb") as handle:
            payload = plistlib.load(handle)
        if isinstance(payload, dict):
            metadata = payload.get("StanStockSchedule")
            if isinstance(metadata, dict):
                raw_timezone = metadata.get("timezone")
                if isinstance(raw_timezone, str):
                    expected_timezone = raw_timezone
                schedule = metadata
    return {
        "installed": installed,
        "loaded": installed and _is_loaded() if sys.platform == "darwin" else False,
        "plist_path": str(plist_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "current_timezone": current_timezone,
        "expected_timezone": expected_timezone,
        "timezone_matches": expected_timezone in {None, current_timezone},
        "schedule": schedule,
    }


def validation_details(validation: ScheduleValidation) -> dict[str, object]:
    return asdict(validation)


def _validate_private_env_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Scheduled refresh requires the ignored local credential file: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if path.stat().st_uid != os.getuid():
        raise ValueError(f"{path} must be owned by the current user")
    if mode & 0o077:
        raise ValueError(f"{path} must not be readable or writable by group or other users")


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

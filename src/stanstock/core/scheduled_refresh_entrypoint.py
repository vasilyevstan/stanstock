from __future__ import annotations

import os
from pathlib import Path

from stanstock.core.environment import load_private_environment_file

ENV_FILE_ENV = "STANSTOCK_ENV_FILE"


def main() -> None:
    scheduled_timezone = os.environ.get("STANSTOCK_SCHEDULE_TIMEZONE", "").strip()
    environment_path = Path(os.environ.get(ENV_FILE_ENV, Path.cwd() / ".env"))
    try:
        load_private_environment_file(environment_path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"StanStock scheduled refresh startup failed: {exc}") from exc

    if scheduled_timezone:
        os.environ["STANSTOCK_SCHEDULE_TIMEZONE"] = scheduled_timezone
    os.environ["STANSTOCK_DISABLE_KEYCHAIN"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stanstock.settings.dev")
    _execute_scheduled_refresh()


def _execute_scheduled_refresh() -> None:
    from django.core.management import execute_from_command_line

    execute_from_command_line(["stanstock-scheduled-refresh", "scheduled_refresh"])


if __name__ == "__main__":
    main()

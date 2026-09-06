from __future__ import annotations

from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from stanstock.core.launchd import (
    detect_iana_timezone,
    install_launch_agent,
    launch_agent_status,
    uninstall_launch_agent,
    validation_details,
)


class Command(BaseCommand):
    help = "Install, inspect, or uninstall the local StanStock daily LaunchAgent."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("action", choices=["install", "status", "uninstall"])
        parser.add_argument(
            "--project-root",
            type=Path,
            default=settings.BASE_DIR,
            help="StanStock checkout containing .venv, .env, manage.py, and scripts.",
        )
        parser.add_argument(
            "--timezone",
            help="IANA timezone to validate and record; defaults to the machine timezone.",
        )
        parser.add_argument(
            "--no-load",
            action="store_true",
            help="Write the plist without loading it through launchctl.",
        )

    def handle(self, *args: object, **options: object) -> None:
        action = str(options["action"])
        try:
            if action == "install":
                timezone_name = str(options.get("timezone") or detect_iana_timezone())
                project_root = options["project_root"]
                if not isinstance(project_root, Path):
                    raise ValueError("--project-root must be a filesystem path")
                plist_path, validation = install_launch_agent(
                    project_root=project_root,
                    timezone_name=timezone_name,
                    load=not bool(options["no_load"]),
                )
                self.stdout.write(
                    self.style.SUCCESS(
                        f"LaunchAgent installed at {plist_path}; "
                        f"validation={validation_details(validation)!r}"
                    )
                )
                return
            if action == "status":
                self.stdout.write(f"LaunchAgent status: {launch_agent_status()!r}")
                return
            removed = uninstall_launch_agent(unload=not bool(options["no_load"]))
            self.stdout.write(
                self.style.SUCCESS(
                    "LaunchAgent uninstalled." if removed else "LaunchAgent was not installed."
                )
            )
        except (OSError, ValueError) as exc:
            raise CommandError(f"LaunchAgent {action} failed: {exc}") from exc

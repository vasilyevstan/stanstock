from __future__ import annotations

import os
import sys

from django.core.management import execute_from_command_line


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stanstock.settings.dev")
    execute_from_command_line(["stanstock", *sys.argv[1:]])

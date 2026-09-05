from __future__ import annotations

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Create the private StanStock data directory if it does not exist."

    def handle(self, *args: object, **options: object) -> None:
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.stdout.write(self.style.SUCCESS(f"Data directory ready: {settings.DATA_DIR}"))

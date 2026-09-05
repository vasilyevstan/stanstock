from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from stanstock.core.integrity import verify_registered_assets


class Command(BaseCommand):
    help = "Verify that every registered DataAsset exists and matches its SHA-256."

    def handle(self, *args: object, **options: object) -> None:
        report = verify_registered_assets()
        if not report.ok:
            failures = [
                {
                    "asset_id": failure.asset_id,
                    "relative_path": failure.relative_path,
                    "reason": failure.reason,
                }
                for failure in report.failures
            ]
            raise CommandError(
                json.dumps(
                    {
                        "checked": report.checked,
                        "failures": failures,
                    },
                    sort_keys=True,
                )
            )
        self.stdout.write(self.style.SUCCESS(f"Verified {report.checked} registered asset files"))

from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand, CommandError

from stanstock.data.providers.exceptions import ProviderError
from stanstock.data.sec_config import load_sec_fundamentals_config
from stanstock.data.sec_ingestion import fetch_sec_mapping_asset


class Command(BaseCommand):
    help = "Fetch and immutably preserve the official SEC ticker/exchange/CIK mapping."

    def handle(self, *args: object, **options: Any) -> None:
        try:
            asset, created = fetch_sec_mapping_asset(config=load_sec_fundamentals_config())
        except ProviderError as exc:
            raise CommandError(str(exc)) from exc
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(
            self.style.SUCCESS(
                f"SEC mapping asset={asset.pk} sha256={asset.sha256} "
                f"status={'created' if created else 'reused'}"
            )
        )

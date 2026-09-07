from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from stanstock.data.models import ProviderRecord
from stanstock.data.providers import sec
from stanstock.data.providers.exceptions import ProviderConfigurationError
from stanstock.data.sec_config import load_sec_fundamentals_config


class Command(BaseCommand):
    help = "Enable or disable SEC EDGAR ingestion after the bounded preflight."

    def add_arguments(self, parser: Any) -> None:
        action = parser.add_mutually_exclusive_group(required=True)
        action.add_argument("--enable", action="store_true")
        action.add_argument("--disable", action="store_true")

    def handle(self, *args: object, **options: object) -> None:
        record, _created = ProviderRecord.objects.get_or_create(provider=sec.PROVIDER)
        if options["disable"]:
            record.enabled = False
            record.status = "disabled"
            record.save(update_fields=["enabled", "status"])
            self.stdout.write(self.style.SUCCESS("SEC EDGAR ingestion disabled."))
            return

        try:
            sec.build_user_agent()
        except ProviderConfigurationError as exc:
            raise CommandError(str(exc)) from exc
        if record.status != "ok" or record.last_success_at is None:
            raise CommandError(
                "SEC has not passed the bounded local preflight. Load the local .env "
                "and run source_spike with only the SEC probe before enabling it."
            )
        config = load_sec_fundamentals_config()
        metadata = dict(record.metadata)
        metadata.update(
            {
                "contact_configured": True,
                "authentication_required": False,
                "api_key_required": False,
                "requests_per_second": config.requests_per_second,
                "maximum_requests_per_second": 10,
                "config_version": config.config_version,
                "config_hash": config.config_hash,
            }
        )
        record.enabled = True
        record.terms_url = sec.TERMS_URL
        record.terms_checked_at = timezone.now()
        record.usage_scope = sec.USAGE_SCOPE
        record.status = "ok"
        record.last_error = ""
        record.metadata = metadata
        record.save()
        self.stdout.write(
            self.style.SUCCESS(
                "SEC EDGAR enabled for private research. The contact address remains "
                "only in the local environment and is sent solely in SEC request headers."
            )
        )

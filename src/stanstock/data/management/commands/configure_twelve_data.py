from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from stanstock.data.live_us import (
    DEFAULT_CREDITS_PER_MINUTE,
    DEFAULT_DAILY_CREDIT_LIMIT,
    DEFAULT_MAX_SYMBOLS,
    PRIVATE_USAGE_SCOPE,
    ProviderCreditBudget,
)
from stanstock.data.models import ProviderRecord
from stanstock.data.providers import twelve_data
from stanstock.data.providers.exceptions import ProviderError

CONFIRMATION = "PERSONAL_INTERNAL_DISPLAY_AUTHORIZED"
DISPLAY_PLANS = ("grow", "pro", "ultra", "custom")


class Command(BaseCommand):
    help = (
        "Explicitly enable or disable Twelve Data for private, personal, "
        "non-commercial US market-data ingestion."
    )

    def add_arguments(self, parser: Any) -> None:
        action = parser.add_mutually_exclusive_group(required=True)
        action.add_argument("--enable", action="store_true")
        action.add_argument("--disable", action="store_true")
        parser.add_argument(
            "--confirm",
            help=f"Required for enable; pass the exact value {CONFIRMATION!r}.",
        )
        parser.add_argument(
            "--plan",
            help=(
                "Twelve Data plan or agreement (grow, pro, ultra, or custom) "
                "that grants internal display rights. "
                "The Basic plan is intentionally excluded because its pricing page "
                "labels that tier internal non-display."
            ),
        )

    def handle(self, *args: object, **options: object) -> None:
        if options["disable"]:
            record, _created = ProviderRecord.objects.get_or_create(provider=twelve_data.PROVIDER)
            record.enabled = False
            record.status = "disabled"
            record.save(update_fields=["enabled", "status"])
            self.stdout.write(self.style.SUCCESS("Twelve Data disabled."))
            return

        if options.get("confirm") != CONFIRMATION:
            raise CommandError(f"Enabling Twelve Data requires --confirm {CONFIRMATION}")
        plan = str(options.get("plan") or "").lower()
        if plan not in DISPLAY_PLANS:
            raise CommandError(
                "Enabling the price-bearing UI requires --plan grow|pro|ultra|custom; "
                "Twelve Data Basic is labeled internal non-display."
            )
        try:
            api_key = twelve_data.resolve_api_key()
            ProviderRecord.objects.get_or_create(provider=twelve_data.PROVIDER)
            budget = ProviderCreditBudget(require_enabled=False)
            budget.preflight(1)
            budget.consume()
            series = twelve_data.fetch_daily_price_series(
                "AAPL",
                outputsize=1,
                api_key=api_key,
            )
        except ProviderError as exc:
            raise CommandError(f"Twelve Data validation failed: {exc}") from exc

        checked_at = timezone.now()
        record, _created = ProviderRecord.objects.get_or_create(provider=twelve_data.PROVIDER)
        record.refresh_from_db()
        metadata = dict(record.metadata)
        metadata.setdefault("daily_credit_limit", DEFAULT_DAILY_CREDIT_LIMIT)
        metadata.setdefault("credits_per_minute", DEFAULT_CREDITS_PER_MINUTE)
        metadata.setdefault("maximum_symbols", DEFAULT_MAX_SYMBOLS)
        metadata.update(
            {
                "plan": plan,
                "internal_display_rights_confirmed": True,
                "price_adjustment": "splits",
                "return_definition": "split_adjusted_price_return",
                "dividends_included": False,
                "validated_symbol": series.symbol,
                "validated_latest_date": series.bars[-1].trade_date.isoformat(),
            }
        )
        record.enabled = True
        record.terms_url = twelve_data.TERMS_URL
        record.terms_checked_at = checked_at
        record.usage_scope = PRIVATE_USAGE_SCOPE
        record.status = "ok"
        record.last_success_at = checked_at
        record.last_error = ""
        record.metadata = metadata
        record.save()
        self.stdout.write(
            self.style.SUCCESS(
                "Twelve Data enabled for private personal/internal US use. "
                "The API key remains environment-only; internal display rights "
                "were explicitly confirmed, while redistribution and public "
                "display remain disabled."
            )
        )

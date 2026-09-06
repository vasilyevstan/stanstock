from __future__ import annotations

from typing import Any

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from stanstock.data.live_us import (
    DEFAULT_CREDITS_PER_MINUTE,
    DEFAULT_DAILY_CREDIT_LIMIT,
    DEFAULT_MAX_SYMBOLS,
    ProviderCreditBudget,
)
from stanstock.data.models import ProviderRecord
from stanstock.data.provider_policy import (
    BASIC_USAGE_SCOPE,
    DISPLAY_PLANS,
    PRIVATE_USAGE_SCOPE,
)
from stanstock.data.providers import twelve_data
from stanstock.data.providers.exceptions import ProviderError

DISPLAY_CONFIRMATION = "PERSONAL_INTERNAL_DISPLAY_AUTHORIZED"
BASIC_CONFIRMATION = "PERSONAL_SINGLE_USER_NONCOMMERCIAL_AUTHORIZED"


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
            help=(
                "Required for enable. Basic requires "
                f"{BASIC_CONFIRMATION!r}; display-entitled plans require "
                f"{DISPLAY_CONFIRMATION!r}."
            ),
        )
        parser.add_argument(
            "--plan",
            help=(
                "Twelve Data plan or agreement (basic, grow, pro, ultra, or custom). "
                "Basic is restricted to one explicitly licensed user and personal, "
                "non-commercial, non-redistributed use."
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

        plan = str(options.get("plan") or "").lower()
        licensed_user_id: str | None = None
        usage_scope = PRIVATE_USAGE_SCOPE
        if plan == "basic":
            if options.get("confirm") != BASIC_CONFIRMATION:
                raise CommandError(
                    f"Enabling Twelve Data Basic requires --confirm {BASIC_CONFIRMATION}"
                )
            licensed_user_id = _sole_active_user_id()
            usage_scope = BASIC_USAGE_SCOPE
        elif plan in DISPLAY_PLANS:
            if options.get("confirm") != DISPLAY_CONFIRMATION:
                raise CommandError(
                    "Enabling a display-entitled Twelve Data plan requires "
                    f"--confirm {DISPLAY_CONFIRMATION}"
                )
        else:
            raise CommandError("Enabling Twelve Data requires --plan basic|grow|pro|ultra|custom.")
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
                "internal_display_rights_confirmed": plan in DISPLAY_PLANS,
                "personal_noncommercial_confirmed": plan == "basic",
                "price_adjustment": "splits",
                "return_definition": "split_adjusted_price_return",
                "dividends_included": False,
                "validated_symbol": series.symbol,
                "validated_latest_date": series.bars[-1].trade_date.isoformat(),
            }
        )
        if licensed_user_id is not None:
            metadata["licensed_user_id"] = licensed_user_id
        else:
            metadata.pop("licensed_user_id", None)
        record.enabled = True
        record.terms_url = twelve_data.TERMS_URL
        record.terms_checked_at = checked_at
        record.usage_scope = usage_scope
        record.status = "ok"
        record.last_success_at = checked_at
        record.last_error = ""
        record.metadata = metadata
        record.save()
        self.stdout.write(
            self.style.SUCCESS(
                "Twelve Data enabled for the confirmed private US usage scope. "
                "The API key remains outside Git and the database; internal "
                "access is authenticated, while redistribution, public display, "
                "and commercial use remain disabled."
            )
        )


def _sole_active_user_id() -> str:
    active_user_ids = list(
        get_user_model()
        .objects.filter(is_active=True)
        .order_by("pk")
        .values_list("pk", flat=True)[:2]
    )
    if len(active_user_ids) != 1:
        raise CommandError(
            "Twelve Data Basic requires exactly one active StanStock user. "
            "Each additional user needs a separately licensed private deployment "
            "or an agreement covering the intended audience."
        )
    return str(active_user_ids[0])

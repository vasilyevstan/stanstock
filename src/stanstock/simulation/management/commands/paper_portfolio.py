from __future__ import annotations

import json
from datetime import date

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError, CommandParser

from stanstock.core.research_product_refresh import resolve_scheduled_owner
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.asof import PriceFrameChecksumMismatchError
from stanstock.data.provider_policy import ProviderConfigurationError
from stanstock.simulation.paper import (
    activate_current_method_paper,
    measure_current_method_paper,
    read_current_method_paper,
)


class Command(BaseCommand):
    help = "Preview, explicitly activate, or measure the one notional current-method paper policy."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--owner",
            help="Active owner username; otherwise installed owner policy",
        )
        action = parser.add_mutually_exclusive_group()
        action.add_argument(
            "--preview",
            action="store_true",
            help="No writes; same activation proof",
        )
        action.add_argument(
            "--activate",
            action="store_true",
            help="Publish one immutable activation",
        )
        action.add_argument(
            "--session",
            type=date.fromisoformat,
            help="Capture one prospective close",
        )

    def handle(self, *args: object, **options: object) -> str:
        try:
            owner = (
                get_user_model().objects.filter(username=options["owner"], is_active=True).first()
                if options.get("owner")
                else resolve_scheduled_owner()
            )
            if owner is None:
                raise CommandError("An active authorized owner is required")
            if options["preview"]:
                preview = activate_current_method_paper(user=owner, persist=False)
                report = {
                    "status": "already_activated" if preview.asset else "preview_not_activated",
                    "activation_preview": preview.document,
                }
            else:
                if options["activate"]:
                    activate_current_method_paper(user=owner)
                if options["session"]:
                    session = options["session"]
                    if not isinstance(session, date):
                        raise ValueError("A session date is required")
                    measure_current_method_paper(user=owner, session=session)
                report = read_current_method_paper(user=owner)
        except (
            ValueError,
            TypeError,
            KeyError,
            RefreshVerificationError,
            PriceFrameChecksumMismatchError,
            OSError,
            ProviderConfigurationError,
        ):
            raise CommandError(
                "Paper policy withheld: declaration, native evidence or prospective marks invalid"
            ) from None
        self.stdout.write(json.dumps(report, sort_keys=True, allow_nan=False))
        return ""

from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError, CommandParser

from stanstock.core.research_product_refresh import resolve_scheduled_owner
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.asof import PriceFrameChecksumMismatchError
from stanstock.data.provider_policy import ProviderConfigurationError
from stanstock.research.forward_scorecard import declare_forward_protocol, read_forward_scorecard


class Command(BaseCommand):
    help = "Declare the frozen forward protocol explicitly, or read its owner-bound scorecard."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--owner",
            help="Active owner username; otherwise use installed owner policy",
        )
        parser.add_argument(
            "--declare",
            action="store_true",
            help="Publish once before reading losses",
        )
        parser.add_argument(
            "--product-version",
            choices=("research-product-v1", "research-product-v2"),
            default="research-product-v1",
            help="Independent frozen protocol; declaring v2 never changes the v1 declaration.",
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
            if options["declare"]:
                declaration = declare_forward_protocol(
                    user=owner, product_version=str(options["product_version"])
                )
                self.stdout.write(f"Forward protocol declared: {declaration.asset.id}")
            report = read_forward_scorecard(
                user=owner, product_version=str(options["product_version"])
            )
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
                "Forward scorecard withheld: declaration or evidence invalid"
            ) from None
        self.stdout.write(json.dumps(report, sort_keys=True, default=str, allow_nan=False))
        return ""

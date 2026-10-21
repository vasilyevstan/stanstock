from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from stanstock.research.product_reader import ProductUser


class Command(BaseCommand):
    help = "Preview, explicitly activate, or measure the one notional current-method paper policy."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--policy-version",
            choices=("current-method-paper-6m-v1", "current-method-paper-6m-v2"),
            default="current-method-paper-6m-v1",
            help="Explicit paper contract; v1 remains the default.",
        )
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
            # call_command keyword options bypass argparse's choices check.
            if options["policy_version"] not in (
                "current-method-paper-6m-v1",
                "current-method-paper-6m-v2",
            ):
                raise ValueError("Unknown paper policy version")
            owner = (
                get_user_model().objects.filter(username=options["owner"], is_active=True).first()
                if options.get("owner")
                else resolve_scheduled_owner()
            )
            if owner is None:
                raise CommandError("An active authorized owner is required")
            if options["policy_version"] == "current-method-paper-6m-v2":
                report = _v2_report(owner, options)
            elif options["preview"]:
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


def _v2_report(owner: ProductUser, options: dict[str, object]) -> dict[str, object]:
    # V2 has no hypothetical publication clock or repair/backfill operation.
    if options["preview"] or (options["activate"] and options["session"]):
        raise ValueError("Unsupported paper v2 action")
    from stanstock.data.assets import AssetStore
    from stanstock.simulation.paper_jobs import execute_paper_capture_job
    from stanstock.simulation.paper_v2 import (
        activate_current_method_paper as activate_v2,
    )
    from stanstock.simulation.paper_v2 import (
        read_current_method_paper as read_v2,
    )

    store = AssetStore()
    if options["activate"]:
        activate_v2(user=owner, store=store)
    if options["session"]:
        session = options["session"]
        if type(session) is not date:
            raise ValueError("A session date is required")
        capture = execute_paper_capture_job(owner=owner, session=session, store=store)
        if capture.state not in {"captured", "recovered"}:
            raise ValueError("Paper v2 capture withheld")
    return read_v2(user=owner, store=store)

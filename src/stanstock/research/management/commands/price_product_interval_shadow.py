from __future__ import annotations

import json
from argparse import ArgumentParser
from typing import Any

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from stanstock.data.assets import AssetStore
from stanstock.research.price_product_interval_shadow import (
    IntervalShadowError,
    declare_interval_shadow,
    read_interval_shadow_report,
)


class Command(BaseCommand):
    help = (
        "Read or explicitly declare the research-only prospective interval shadow. "
        "It does not change native predictions or probabilities."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--owner", required=True, help="Exact active owner username")
        parser.add_argument(
            "--declare",
            action="store_true",
            help="Explicit operator action to publish the immutable declaration",
        )

    def handle(self, *args: object, **options: Any) -> None:
        owner_name = options.get("owner")
        if not isinstance(owner_name, str) or not owner_name:
            raise CommandError("interval_shadow_owner_unauthorized")
        user = get_user_model().objects.filter(username=owner_name, is_active=True).first()
        if user is None:
            raise CommandError("interval_shadow_owner_unauthorized")
        store = AssetStore()
        if options.get("declare"):
            try:
                declare_interval_shadow(user=user, store=store)
            except IntervalShadowError as exc:
                raise CommandError(exc.code) from None
        try:
            report = read_interval_shadow_report(user=user, store=store)
        except IntervalShadowError as exc:
            raise CommandError(exc.code) from None
        self.stdout.write(
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        if report["status"] == "integrity_failed":
            raise CommandError(f"price_product_interval_shadow: {report['verification_code']}")

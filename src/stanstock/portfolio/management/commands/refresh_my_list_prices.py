from __future__ import annotations

from typing import Any

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.live_us import refresh_my_list_price_evidence, resolve_us_target_date
from stanstock.data.providers.exceptions import ProviderError
from stanstock.portfolio.models import TrackedSymbol
from stanstock.portfolio.watchlist import (
    TrackedSymbolValidationError,
    verified_catalog_references_for_symbols,
)


class Command(BaseCommand):
    help = (
        "Refresh only the latest persisted Twelve Data prices for the local "
        "owner's tracked My List symbols."
    )

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--owner-id",
            help=(
                "Owner id to use when more than one active owner has tracked symbols. "
                "The command output never prints owner identities."
            ),
        )

    def handle(self, *args: object, **options: object) -> None:
        owner = _select_owner(options.get("owner_id"))
        symbols = tuple(
            TrackedSymbol.objects.filter(owner=owner)
            .order_by("symbol", "created_at")
            .values_list("symbol", flat=True)
        )
        if len(symbols) > 20:
            raise CommandError(
                "My List price refresh supports at most 20 tracked symbols; "
                "reduce the list before retrying."
            )

        decision_time = timezone.now()
        target_date, _grade = resolve_us_target_date(decision_time=decision_time)
        references = {}
        for symbol in symbols:
            try:
                references.update(verified_catalog_references_for_symbols(symbols=(symbol,)))
            except TrackedSymbolValidationError:
                continue

        try:
            result = refresh_my_list_price_evidence(
                symbols=symbols,
                catalog_references=references,
                target_date=target_date,
                decision_time=decision_time,
            )
        except (ProviderError, RefreshVerificationError, ValueError) as exc:
            self.stdout.write(
                f"status=failed target_date={target_date.isoformat()} "
                f"tracked={len(symbols)} reused=0 fetched=0 updated=0 "
                f"failed={len(symbols)} credits=0"
            )
            raise CommandError(
                "My List price refresh could not complete; see the counts above. "
                "No symbols, prices, credentials, or paths were printed."
            ) from exc
        summary = (
            f"status={result.status} target_date={result.target_date.isoformat()} "
            f"tracked={result.total} reused={result.reused} fetched={result.fetched} "
            f"updated={result.updated} failed={result.failed} credits={result.credits_used}"
        )
        self.stdout.write(summary)
        if result.failed:
            raise CommandError(
                "My List price refresh failed for one or more tracked symbols; "
                "see the counts above. No symbols, prices, credentials, or paths "
                "were printed."
            )


def _select_owner(raw_owner_id: object) -> Any:
    user_model = get_user_model()
    if raw_owner_id:
        owner_id = str(raw_owner_id).strip()
        owner = user_model.objects.filter(pk=owner_id, is_active=True).first()
        if owner is None:
            raise CommandError("The selected owner is unavailable.")
        return owner

    owner_ids = list(
        TrackedSymbol.objects.filter(owner__is_active=True)
        .order_by("owner_id")
        .values_list("owner_id", flat=True)
        .distinct()[:2]
    )
    if len(owner_ids) == 1:
        return user_model.objects.get(pk=owner_ids[0])
    if len(owner_ids) > 1:
        raise CommandError("More than one active owner has tracked symbols; rerun with --owner-id.")

    active_owner_ids = list(
        user_model.objects.filter(is_active=True).order_by("pk").values_list("pk", flat=True)[:2]
    )
    if len(active_owner_ids) == 1:
        return user_model.objects.get(pk=active_owner_ids[0])
    raise CommandError("No unique active owner is available; rerun with --owner-id.")

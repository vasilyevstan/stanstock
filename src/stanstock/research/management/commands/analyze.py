from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from stanstock.data.models import DataAsset, Listing, UniverseSnapshot
from stanstock.research.service import analyze_listing, analyze_snapshot


class Command(BaseCommand):
    help = (
        "Run the transparent rules-only research engine for a universe snapshot. "
        "This is demo/research tooling, not the live-US or observed-reissue "
        "interface: every analysis explicitly requests issued_on_time=False, "
        "regardless of the snapshot's grade, the requested target date, or "
        "provider/config flags. An exceptional observed reissue requires a "
        "direct analyze_snapshot(..., issued_on_time=True, ...) service call."
    )

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--snapshot", required=True, help="UniverseSnapshot UUID")
        parser.add_argument(
            "--listing", help="Optional Listing UUID; defaults to all eligible members"
        )
        parser.add_argument(
            "--provider",
            default="synthetic_demo",
            help="DataAsset provider for price_history",
        )
        parser.add_argument("--subject", help="Optional price-history subject for a single listing")
        parser.add_argument("--benchmark-subject", help="Optional benchmark price-history subject")
        parser.add_argument(
            "--target-date",
            help="Logical market date to persist on the run/predictions (YYYY-MM-DD)",
        )
        parser.add_argument("--config", type=Path, help="Scoring YAML path")

    def handle(self, *args: object, **options: object) -> None:
        try:
            snapshot_id = UUID(str(options["snapshot"]))
            snapshot = UniverseSnapshot.objects.get(pk=snapshot_id)
            listing_id = options.get("listing")
            config_path = options.get("config")
            if config_path is not None and not isinstance(config_path, Path):
                raise CommandError("--config must be a path")
            target_date = _parse_target_date(options.get("target_date"))
            if listing_id:
                listing = Listing.objects.select_related("security__company").get(
                    pk=UUID(str(listing_id))
                )
                subject = options.get("subject")
                benchmark_subject = options.get("benchmark_subject")
                result = analyze_listing(
                    listing=listing,
                    universe_snapshot=snapshot,
                    decision_time=timezone.now(),
                    target_date=target_date,
                    issued_on_time=False,
                    provider=str(options["provider"]),
                    subject=str(subject) if subject is not None else None,
                    benchmark_subject=str(benchmark_subject)
                    if benchmark_subject is not None
                    else None,
                    config_path=config_path,
                )
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Analyzed {listing.ticker} (research-grade, issued_on_time=False): "
                        f"analysis={result.analysis.pk} predictions={len(result.predictions)}"
                    )
                )
                return
            results = analyze_snapshot(
                universe_snapshot=snapshot,
                decision_time=timezone.now(),
                target_date=target_date,
                issued_on_time=False,
                provider=str(options["provider"]),
                benchmark_subject=str(options["benchmark_subject"])
                if options.get("benchmark_subject") is not None
                else None,
                config_path=config_path,
            )
        except (
            ValueError,
            DataAsset.DoesNotExist,
            UniverseSnapshot.DoesNotExist,
            Listing.DoesNotExist,
        ) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(
            self.style.SUCCESS(
                f"Analyzed {len(results)} listings (research-grade, issued_on_time=False)"
            )
        )


def _parse_target_date(raw: object) -> date | None:
    if raw is None:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError as exc:
        raise CommandError("--target-date must use YYYY-MM-DD") from exc

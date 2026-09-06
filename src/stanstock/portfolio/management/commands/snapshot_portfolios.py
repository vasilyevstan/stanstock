from __future__ import annotations

from datetime import date
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from stanstock.portfolio.jobs import execute_portfolio_snapshot_job


class Command(BaseCommand):
    help = (
        "Record immutable valuation snapshots for every active tracked portfolio. "
        "Run this after market-data refreshes."
    )

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--target-date",
            help=(
                "Operational job date (YYYY-MM-DD); defaults to the local date. "
                "Each snapshot uses its latest persisted market session."
            ),
        )

    def handle(self, *args: object, **options: object) -> None:
        target_date = _parse_target_date(options.get("target_date")) or timezone.localdate()

        try:
            run = execute_portfolio_snapshot_job(
                target_date=target_date,
            )
        except ValueError as exc:
            raise CommandError(f"Portfolio snapshot job failed: {exc}") from exc

        self.stdout.write(
            self.style.SUCCESS(
                f"snapshot_portfolios job_run={run.pk} status={run.status} "
                f"target_date={target_date.isoformat()} details={run.details!r}"
            )
        )


def _parse_target_date(raw: object) -> date | None:
    if raw is None:
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError as exc:
        raise CommandError("--target-date must use YYYY-MM-DD") from exc

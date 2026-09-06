from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from django.contrib.auth import get_user_model
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError

from stanstock.portfolio.service import (
    SAMPLE_PORTFOLIO_DEFAULT_CAPITAL,
    SAMPLE_PORTFOLIO_DEFAULT_TOP_N,
    PortfolioValuationError,
    build_sample_portfolio,
)
from stanstock.research.models import AnalysisRun


class Command(BaseCommand):
    help = (
        "Build an idempotent frozen StanStock sample portfolio from the latest "
        "provider-backed opportunity run."
    )

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--username",
            help="Portfolio owner. Defaults to the sole active user.",
        )
        parser.add_argument(
            "--starting-capital",
            default=str(SAMPLE_PORTFOLIO_DEFAULT_CAPITAL),
            help="Reference starting capital in USD.",
        )
        parser.add_argument(
            "--top-n",
            type=int,
            default=SAMPLE_PORTFOLIO_DEFAULT_TOP_N,
            help="Maximum number of eligible opportunities to equal-weight.",
        )
        parser.add_argument(
            "--analysis-run",
            help="Optional completed provider-backed AnalysisRun UUID.",
        )

    def handle(self, *args: object, **options: object) -> None:
        owner = _resolve_owner(options.get("username"))
        capital = _parse_capital(options.get("starting_capital"))
        source_run = _resolve_run(options.get("analysis_run"))
        raw_top_n = options.get("top_n")
        if not isinstance(raw_top_n, int):
            raise CommandError("--top-n must be an integer.")
        try:
            portfolio, created = build_sample_portfolio(
                owner=owner,
                starting_capital=capital,
                top_n=raw_top_n,
                source_run=source_run,
            )
        except PortfolioValuationError as exc:
            raise CommandError(str(exc)) from exc

        state = "created" if created else "already exists"
        self.stdout.write(
            self.style.SUCCESS(f"Sample portfolio {state}: {portfolio.name} ({portfolio.pk})")
        )


def _resolve_owner(raw_username: object) -> User:
    user_model = get_user_model()
    if raw_username:
        try:
            return user_model.objects.get(username=str(raw_username), is_active=True)
        except user_model.DoesNotExist as exc:
            raise CommandError("The requested active user does not exist.") from exc
    users = list(user_model.objects.filter(is_active=True).order_by("pk")[:2])
    if len(users) != 1:
        raise CommandError(
            "Pass --username when the installation does not have exactly one active user."
        )
    return users[0]


def _parse_capital(raw: object) -> Decimal:
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise CommandError("--starting-capital must be a decimal number.") from exc


def _resolve_run(raw_run: object) -> AnalysisRun | None:
    if not raw_run:
        return None
    try:
        return AnalysisRun.objects.select_related("universe_snapshot").get(pk=str(raw_run))
    except (AnalysisRun.DoesNotExist, ValidationError, ValueError) as exc:
        raise CommandError("The requested analysis run does not exist.") from exc

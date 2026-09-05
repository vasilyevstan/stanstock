from __future__ import annotations

from datetime import date
from typing import Any, cast
from uuid import UUID

from django.core.management.base import BaseCommand, CommandError

from stanstock.research.models import Prediction, PredictionOutcome
from stanstock.research.outcomes import evaluate_predictions


class Command(BaseCommand):
    help = "Evaluate immutable predictions against as-of price assets."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("prediction_ids", nargs="*", help="Prediction UUIDs to evaluate")
        parser.add_argument(
            "--all-pending",
            action="store_true",
            help="Evaluate predictions with no outcome or a non-matured outcome",
        )
        parser.add_argument("--provider", default="synthetic_demo", help="DataAsset provider")
        parser.add_argument("--evaluation-date", required=True, help="YYYY-MM-DD cutoff date")
        parser.add_argument("--benchmark-subject", help="Optional benchmark price-history subject")

    def handle(self, *args: object, **options: object) -> None:
        ids = [str(raw) for raw in cast(list[str], options["prediction_ids"])]
        all_pending = bool(options["all_pending"])
        if all_pending == bool(ids):
            raise CommandError("Provide either prediction IDs or --all-pending, but not both")
        evaluation_date = _parse_date(options["evaluation_date"])
        benchmark_subject = options.get("benchmark_subject")
        predictions = _select_predictions(ids, all_pending)
        results = evaluate_predictions(
            predictions,
            provider=str(options["provider"]),
            evaluation_date=evaluation_date,
            benchmark_subject=str(benchmark_subject) if benchmark_subject is not None else None,
        )
        counts: dict[str, int] = {}
        for result in results:
            counts[result.action] = counts.get(result.action, 0) + 1
        summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"
        self.stdout.write(self.style.SUCCESS(f"Evaluated {len(results)} predictions: {summary}"))


def _parse_date(raw: object) -> date:
    try:
        return date.fromisoformat(str(raw))
    except ValueError as exc:
        raise CommandError("--evaluation-date must use YYYY-MM-DD") from exc


def _select_predictions(ids: list[str], all_pending: bool) -> list[Prediction]:
    queryset = Prediction.objects.select_related("listing").order_by("generated_at", "id")
    if all_pending:
        return list(queryset.exclude(outcome__status=PredictionOutcome.Status.MATURED))
    uuids: list[UUID] = []
    try:
        uuids = [UUID(raw) for raw in ids]
    except ValueError as exc:
        raise CommandError("Prediction IDs must be UUIDs") from exc
    found = list(queryset.filter(pk__in=uuids))
    found_ids = {prediction.pk for prediction in found}
    missing = [str(prediction_id) for prediction_id in uuids if prediction_id not in found_ids]
    if missing:
        raise CommandError("Unknown prediction IDs: " + ", ".join(missing))
    return found

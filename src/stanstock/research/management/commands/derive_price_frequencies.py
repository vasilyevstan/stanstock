"""Offline recovery and semantic verification for one immutable product run."""

from __future__ import annotations

from uuid import UUID

from django.core.management.base import BaseCommand, CommandError

from stanstock.data.assets import open_asset_store
from stanstock.research.models import AnalysisRun
from stanstock.research.product_frequency_evidence import (
    register_product_frequencies,
    verify_registered_product_frequencies,
)


class Command(BaseCommand):
    help = "Derive or verify terminal simulation frequencies for one product run."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--run", required=True, help="Immutable AnalysisRun UUID")
        parser.add_argument(
            "--verify",
            action="store_true",
            help="Re-derive and compare registered bytes without writing.",
        )

    def handle(self, *args, **options) -> str:
        try:
            run_id = UUID(str(options["run"]))
        except (TypeError, ValueError) as exc:
            raise CommandError("--run must be an AnalysisRun UUID") from exc
        run = AnalysisRun.objects.filter(pk=run_id).first()
        if run is None:
            raise CommandError("The requested analysis run does not exist")
        store = open_asset_store()
        try:
            if options["verify"]:
                verify_registered_product_frequencies(run=run, store=store)
                self.stdout.write("Frequency evidence verified.")
            else:
                register_product_frequencies(run=run, store=store)
                self.stdout.write("Frequency evidence registered.")
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        return ""

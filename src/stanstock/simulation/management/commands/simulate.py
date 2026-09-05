from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID

from django.core.management.base import BaseCommand, CommandError

from stanstock.data.models import UniverseSnapshot
from stanstock.simulation.builders import run_simulation_workflow
from stanstock.simulation.models import SimulationDefinition


class Command(BaseCommand):
    help = "Run and persist a backtest or portfolio simulation."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--name", default="Simulation", help="Name for the simulation")
        parser.add_argument(
            "--mode",
            required=True,
            choices=SimulationDefinition.Mode.values,
            help="Simulation mode: backtest or portfolio",
        )
        parser.add_argument(
            "--snapshot",
            required=True,
            help="UniverseSnapshot UUID",
        )
        parser.add_argument(
            "--start-date",
            required=True,
            help="Simulation start date (YYYY-MM-DD)",
        )
        parser.add_argument(
            "--end-date",
            required=True,
            help="Simulation end date (YYYY-MM-DD)",
        )
        parser.add_argument(
            "--capital",
            type=float,
            default=100_000.0,
            help="Starting capital (default: 100,000)",
        )
        parser.add_argument(
            "--cost-bps",
            type=float,
            default=10.0,
            help="Transaction cost bps (default: 10)",
        )
        parser.add_argument(
            "--slippage-bps",
            type=float,
            default=5.0,
            help="Slippage bps (default: 5)",
        )
        parser.add_argument(
            "--top-n",
            type=int,
            help="Top-N listing selection count (required for backtest mode)",
        )
        parser.add_argument(
            "--listings",
            help="Comma-separated listing UUIDs (required for portfolio mode)",
        )
        parser.add_argument(
            "--rebalance-frequency",
            choices=["daily", "weekly", "monthly", "quarterly", "yearly", "never"],
            help="Rebalance frequency (default: monthly for backtest, never for portfolio)",
        )
        parser.add_argument(
            "--benchmark-subject",
            help="Optional benchmark price history subject (e.g. SPY or ZZBENCH01)",
        )
        parser.add_argument(
            "--base-currency",
            help=(
                "Restrict a backtest to one native currency (for example USD or EUR). "
                "Mixed-currency portfolios are rejected until FX conversion is implemented."
            ),
        )
        parser.add_argument(
            "--provider",
            default="synthetic_demo",
            help="Price history DataAsset provider (default: synthetic_demo)",
        )

    def handle(self, *args: object, **options: object) -> None:
        try:
            snapshot_uuid = UUID(str(options["snapshot"]))
            snapshot = UniverseSnapshot.objects.get(pk=snapshot_uuid)
        except (ValueError, UniverseSnapshot.DoesNotExist) as exc:
            snap_val = options.get("snapshot")
            raise CommandError(
                f"Invalid or non-existent UniverseSnapshot UUID: {snap_val}"
            ) from exc

        try:
            start_date = date.fromisoformat(str(options["start_date"]))
            end_date = date.fromisoformat(str(options["end_date"]))
        except (ValueError, TypeError) as exc:
            raise CommandError("Invalid date format; expected YYYY-MM-DD.") from exc

        mode = str(options["mode"])
        listing_uuids: list[UUID] | None = None
        if options.get("listings"):
            raw_lids = str(options["listings"]).split(",")
            try:
                listing_uuids = [UUID(lid.strip()) for lid in raw_lids if lid.strip()]
            except ValueError as exc:
                raise CommandError(f"Invalid listing UUID in --listings: {exc}") from exc
            if len(listing_uuids) != len(set(listing_uuids)):
                raise CommandError("Duplicate listing UUIDs detected in --listings.")

        top_n_val = int(str(options["top_n"])) if options.get("top_n") is not None else None
        rf_val = str(options["rebalance_frequency"]) if options.get("rebalance_frequency") else None
        bench_val = str(options["benchmark_subject"]) if options.get("benchmark_subject") else None

        try:
            definition, run, result = run_simulation_workflow(
                name=str(options["name"]),
                mode=mode,
                snapshot=snapshot,
                start_date=start_date,
                end_date=end_date,
                starting_capital=float(options["capital"]),  # type: ignore[arg-type]
                transaction_cost_bps=float(options["cost_bps"]),  # type: ignore[arg-type]
                slippage_bps=float(options["slippage_bps"]),  # type: ignore[arg-type]
                top_n=top_n_val,
                selected_listing_ids=listing_uuids,
                rebalance_frequency=rf_val,
                benchmark_subject=bench_val,
                base_currency=str(options["base_currency"]).upper()
                if options.get("base_currency")
                else None,
                provider=str(options["provider"]),
            )
        except ValueError as exc:
            raise CommandError(f"Simulation workflow failed: {exc}") from exc

        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully completed simulation run {run.id} for "
                f"definition {definition.id} ({definition.mode}). "
                f"Cumulative return: {result.metrics.cumulative_return:.4%}, "
                f"trades: {result.metrics.total_trades}."
            )
        )

from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID

from django.core.management.base import BaseCommand, CommandError

from stanstock.data.fx import DEFAULT_MAX_CARRY_DAYS
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
            "--benchmark-currency",
            help=(
                "Native currency of the benchmark series. Required when the run converts "
                "currencies, because a price series carries no denomination of its own."
            ),
        )
        parser.add_argument(
            "--base-currency",
            help=(
                "Report every value in this currency (for example USD, EUR, or GBP). "
                "Required when the selection spans several native currencies; inferred "
                "from the single native currency otherwise."
            ),
        )
        parser.add_argument(
            "--restrict-native-currency",
            help=(
                "Only include listings whose native currency is this code, instead of "
                "converting the whole selection."
            ),
        )
        parser.add_argument(
            "--fx-max-carry-days",
            type=int,
            default=DEFAULT_MAX_CARRY_DAYS,
            help=(
                "Longest gap in calendar days between an FX observation and the date it is "
                f"carried forward to (0..{DEFAULT_MAX_CARRY_DAYS}, default "
                f"{DEFAULT_MAX_CARRY_DAYS}). A longer gap fails."
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
        carry_option = options.get("fx_max_carry_days")
        # `or` would treat a deliberate 0 as "unset" and silently restore the
        # 7-day maximum, so only an absent value falls back to the default.
        carry_days = DEFAULT_MAX_CARRY_DAYS if carry_option is None else int(str(carry_option))
        if not 0 <= carry_days <= DEFAULT_MAX_CARRY_DAYS:
            raise CommandError(
                f"--fx-max-carry-days must be between 0 and {DEFAULT_MAX_CARRY_DAYS}."
            )
        benchmark_currency = _upper_option(options.get("benchmark_currency"))
        if benchmark_currency and not bench_val:
            raise CommandError(
                "--benchmark-currency requires --benchmark-subject; there is nothing to "
                "denominate without a benchmark series."
            )

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
                benchmark_currency=benchmark_currency,
                base_currency=_upper_option(options.get("base_currency")),
                restrict_native_currency=_upper_option(options.get("restrict_native_currency")),
                fx_max_carry_days=carry_days,
                provider=str(options["provider"]),
            )
        except ValueError as exc:
            raise CommandError(f"Simulation workflow failed: {exc}") from exc

        metrics = result.metrics
        summary = (
            f"Successfully completed simulation run {run.id} for "
            f"definition {definition.id} ({definition.mode}). "
            f"Cumulative return: {metrics.cumulative_return:.4%}, "
            f"trades: {metrics.total_trades}."
        )
        if metrics.fx_conversion_applied:
            native = ", ".join(metrics.fx_native_currencies or ())
            summary += (
                f" Converted {native} into {metrics.base_currency} "
                f"(max carry {metrics.fx_max_carry_days_used} day(s))."
            )
            if (
                metrics.fx_contribution_return is not None
                and metrics.fx_local_currency_cumulative_return is not None
            ):
                summary += (
                    f" Stock return {metrics.fx_local_currency_cumulative_return:.4%}, "
                    f"FX contribution {metrics.fx_contribution_return:.4%}."
                )
            else:
                summary += f" FX attribution withheld: {metrics.fx_attribution_detail}"

        self.stdout.write(self.style.SUCCESS(summary))


def _upper_option(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text.upper() or None

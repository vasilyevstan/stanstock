from __future__ import annotations

import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any
from uuid import UUID

import polars as pl
from django.db import transaction
from django.utils import timezone

from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import Listing, UniverseSnapshot
from stanstock.simulation.engine import AccountingEngine
from stanstock.simulation.hooks import CorporateEventHook
from stanstock.simulation.models import (
    SimulationDefinition,
    SimulationHolding,
    SimulationRun,
    SimulationTrade,
)
from stanstock.simulation.types import SimulationConfig, SimulationResult


def _get_git_revision() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()[:64]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "git-head-untracked"


@dataclass
class LoadedSimulationRun:
    run: SimulationRun
    daily_curves: pl.DataFrame
    holdings_count: int
    trades_count: int
    input_assets: dict[str, pl.DataFrame] = field(default_factory=dict)


def persist_simulation_run(
    *,
    definition: SimulationDefinition,
    universe_snapshot: UniverseSnapshot,
    result: SimulationResult,
    input_prices: pl.DataFrame | None = None,
    input_signals: pl.DataFrame | None = None,
    input_benchmark: pl.DataFrame | None = None,
    code_revision: str | None = None,
    asset_store: AssetStore | None = None,
    register_data_asset: bool = True,
    run_id: UUID | None = None,
) -> SimulationRun:
    """Persist a SimulationRun, holdings, trades, result Parquet asset, and input frames.

    Schema notes:
    - SimulationHolding has a non-nullable ForeignKey to Listing. Cash balances
      and benchmark curves are therefore stored in the Parquet result asset
      and in SimulationRun.metrics JSON rather than in SimulationHolding rows.
    - Exact input frames (prices, signals, benchmark) are persisted as immutable
      Parquet DataAssets under the run ID to guarantee reproducibility.
    """
    # 1. Validate grade and mode before touching filesystem or database
    if result.config.grade.value != universe_snapshot.grade:
        raise ValueError(
            f"SimulationConfig grade '{result.config.grade.value}' does not match "
            f"UniverseSnapshot grade '{universe_snapshot.grade}'"
        )
    if result.config.mode.value != definition.mode:
        raise ValueError(
            f"SimulationConfig mode '{result.config.mode.value}' does not match "
            f"SimulationDefinition mode '{definition.mode}'"
        )
    _validate_persistable_trades(result.trades)

    # 2. Resolve and validate listings before writing asset or modifying DB
    needed_lids = set(
        result.holdings["listing_id"].to_list() + result.trades["listing_id"].to_list()
    )

    listing_map: dict[str, Listing] = {}
    if needed_lids:
        parsed_uuids: dict[str, UUID] = {}
        for lid in needed_lids:
            try:
                parsed_uuids[lid] = UUID(str(lid))
            except (ValueError, AttributeError):
                raise ValueError(
                    f"Listing identifier '{lid}' is not a valid UUID. "
                    "Permanent UUID listing IDs are required; ticker resolution is not permitted."
                ) from None

        for l_obj in Listing.objects.filter(id__in=list(parsed_uuids.values())):
            listing_map[str(l_obj.id)] = l_obj

        missing = [lid for lid, uid in parsed_uuids.items() if str(uid) not in listing_map]
        if missing:
            raise ValueError(
                f"Cannot persist simulation: listing UUIDs not found in database: {missing}"
            )

    actual_run_id = run_id or uuid.uuid4()
    store = asset_store or AssetStore()
    rev = code_revision or _get_git_revision()

    # 3. Write Parquet assets (primary result + immutable inputs)
    written_rel_paths: list[str] = []
    try:
        # Primary result asset
        res_rel_path = f"simulations/{actual_run_id}/results.parquet"
        stored_result = store.write_frame(res_rel_path, result.daily_curves)
        written_rel_paths.append(res_rel_path)

        stored_prices = None
        if input_prices is not None:
            p_rel_path = f"simulations/{actual_run_id}/inputs/prices.parquet"
            stored_prices = store.write_frame(p_rel_path, input_prices)
            written_rel_paths.append(p_rel_path)

        stored_signals = None
        if input_signals is not None:
            s_rel_path = f"simulations/{actual_run_id}/inputs/signals.parquet"
            stored_signals = store.write_frame(s_rel_path, input_signals)
            written_rel_paths.append(s_rel_path)

        stored_bench = None
        if input_benchmark is not None:
            b_rel_path = f"simulations/{actual_run_id}/inputs/benchmark.parquet"
            stored_bench = store.write_frame(b_rel_path, input_benchmark)
            written_rel_paths.append(b_rel_path)

    except Exception:
        for rel in written_rel_paths:
            store.resolve(rel).unlink(missing_ok=True)
        raise

    # 4. Atomic transaction to register DataAssets, SimulationRun, holdings, and trades
    try:
        with transaction.atomic():
            now = timezone.now()
            input_assets_info: dict[str, dict[str, Any]] = {}

            if register_data_asset:
                if stored_prices is not None and input_prices is not None:
                    p_asset = register_asset(
                        provider="simulation",
                        kind="simulation_input_prices",
                        subject=str(actual_run_id),
                        stored=stored_prices,
                        retrieved_at=now,
                        available_at=now,
                        period_start=result.metrics.start_date,
                        period_end=result.metrics.end_date,
                        metadata={
                            "row_count": input_prices.height,
                            "columns": input_prices.columns,
                        },
                    )
                    input_assets_info["prices"] = {
                        "asset_id": str(p_asset.id),
                        "sha256": stored_prices.sha256,
                        "relative_path": stored_prices.relative_path,
                        "row_count": input_prices.height,
                    }

                if stored_signals is not None and input_signals is not None:
                    s_asset = register_asset(
                        provider="simulation",
                        kind="simulation_input_signals",
                        subject=str(actual_run_id),
                        stored=stored_signals,
                        retrieved_at=now,
                        available_at=now,
                        metadata={
                            "row_count": input_signals.height,
                            "columns": input_signals.columns,
                        },
                    )
                    input_assets_info["signals"] = {
                        "asset_id": str(s_asset.id),
                        "sha256": stored_signals.sha256,
                        "relative_path": stored_signals.relative_path,
                        "row_count": input_signals.height,
                    }

                if stored_bench is not None and input_benchmark is not None:
                    b_asset = register_asset(
                        provider="simulation",
                        kind="simulation_input_benchmark",
                        subject=str(actual_run_id),
                        stored=stored_bench,
                        retrieved_at=now,
                        available_at=now,
                        metadata={
                            "row_count": input_benchmark.height,
                            "columns": input_benchmark.columns,
                        },
                    )
                    input_assets_info["benchmark"] = {
                        "asset_id": str(b_asset.id),
                        "sha256": stored_bench.sha256,
                        "relative_path": stored_bench.relative_path,
                        "row_count": input_benchmark.height,
                    }

                register_asset(
                    provider="simulation",
                    kind="simulation_result",
                    subject=str(actual_run_id),
                    stored=stored_result,
                    retrieved_at=now,
                    available_at=now,
                    period_start=result.metrics.start_date,
                    period_end=result.metrics.end_date,
                    metadata={
                        "mode": result.config.mode.value,
                        "grade": result.config.grade.value,
                        "simulation_kind": result.config.simulation_kind,
                        "metrics": result.metrics.to_dict(),
                        "input_assets": input_assets_info,
                    },
                )

            metrics_payload = result.metrics.to_dict()
            metrics_payload["input_assets"] = input_assets_info

            run = SimulationRun.objects.create(
                id=actual_run_id,
                definition=definition,
                universe_snapshot=universe_snapshot,
                status=SimulationRun.Status.COMPLETE,
                code_revision=rev,
                input_hash=result.input_hash,
                result_asset_key=stored_result.relative_path,
                metrics=metrics_payload,
                finished_at=timezone.now(),
            )

            holdings_to_create = [
                SimulationHolding(
                    run=run,
                    listing=listing_map[str(UUID(str(row["listing_id"])))],
                    observation_date=row["observation_date"],
                    quantity=Decimal(f"{row['quantity']:.8f}"),
                    price=Decimal(f"{row['price']:.6f}"),
                    market_value=Decimal(f"{row['market_value']:.6f}"),
                    weight=Decimal(f"{row['weight']:.8f}"),
                )
                for row in result.holdings.iter_rows(named=True)
            ]
            if holdings_to_create:
                SimulationHolding.objects.bulk_create(holdings_to_create, batch_size=1000)

            trades_to_create = [
                SimulationTrade(
                    run=run,
                    listing=listing_map[str(UUID(str(row["listing_id"])))],
                    trade_date=row["trade_date"],
                    side=row["side"],
                    quantity=Decimal(f"{row['quantity']:.8f}"),
                    price=Decimal(f"{row['price']:.6f}"),
                    gross_value=Decimal(f"{row['gross_value']:.6f}"),
                    costs=Decimal(f"{row['costs']:.6f}"),
                )
                for row in result.trades.iter_rows(named=True)
            ]
            if trades_to_create:
                SimulationTrade.objects.bulk_create(trades_to_create, batch_size=1000)

    except Exception:
        for rel in written_rel_paths:
            store.resolve(rel).unlink(missing_ok=True)
        raise

    return run


def load_simulation_run(
    run_id: UUID | str,
    asset_store: AssetStore | None = None,
) -> LoadedSimulationRun:
    """Load a persisted SimulationRun, its Parquet daily curve, and input frames."""
    store = asset_store or AssetStore()
    run = SimulationRun.objects.select_related("definition", "universe_snapshot").get(id=run_id)
    daily_curves = store.read_frame(run.result_asset_key)

    input_frames: dict[str, pl.DataFrame] = {}
    for key, info in run.metrics.get("input_assets", {}).items():
        if isinstance(info, dict) and "relative_path" in info:
            input_frames[key] = store.read_frame(info["relative_path"])

    return LoadedSimulationRun(
        run=run,
        daily_curves=daily_curves,
        holdings_count=run.holdings.count(),
        trades_count=run.trades.count(),
        input_assets=input_frames,
    )


def execute_and_persist_simulation(
    *,
    definition: SimulationDefinition,
    universe_snapshot: UniverseSnapshot,
    prices: pl.DataFrame,
    signals: pl.DataFrame | None = None,
    benchmark_prices: pl.DataFrame | None = None,
    calendar: Sequence[date] | None = None,
    corporate_event_hook: CorporateEventHook | None = None,
    asset_store: AssetStore | None = None,
    code_revision: str | None = None,
) -> tuple[SimulationRun, SimulationResult]:
    """Execute a simulation defined by SimulationDefinition and persist the results."""
    config_dict = dict(definition.config)
    if "mode" not in config_dict:
        config_dict["mode"] = definition.mode
    if "grade" not in config_dict:
        config_dict["grade"] = universe_snapshot.grade

    config = SimulationConfig.from_dict(config_dict)

    if config.grade.value != universe_snapshot.grade:
        raise ValueError(
            f"SimulationConfig grade '{config.grade.value}' does not match "
            f"UniverseSnapshot grade '{universe_snapshot.grade}'"
        )
    if config.mode.value != definition.mode:
        raise ValueError(
            f"SimulationConfig mode '{config.mode.value}' does not match "
            f"SimulationDefinition mode '{definition.mode}'"
        )

    engine = AccountingEngine(config=config, corporate_event_hook=corporate_event_hook)

    try:
        result = engine.run(
            prices=prices,
            signals=signals,
            benchmark_prices=benchmark_prices,
            calendar=calendar,
        )
    except Exception as exc:
        # Create a failed SimulationRun record for observability
        run = SimulationRun.objects.create(
            definition=definition,
            universe_snapshot=universe_snapshot,
            status=SimulationRun.Status.FAILED,
            code_revision=code_revision or _get_git_revision(),
            input_hash="error",
            error=str(exc),
            finished_at=timezone.now(),
        )
        raise

    run = persist_simulation_run(
        definition=definition,
        universe_snapshot=universe_snapshot,
        result=result,
        input_prices=prices,
        input_signals=signals,
        input_benchmark=benchmark_prices,
        code_revision=code_revision,
        asset_store=asset_store,
    )
    return run, result


def _validate_persistable_trades(trades: pl.DataFrame) -> None:
    for row in trades.iter_rows(named=True):
        quantity = Decimal(f"{row['quantity']:.8f}")
        price = Decimal(f"{row['price']:.6f}")
        gross_value = Decimal(f"{row['gross_value']:.6f}")
        if quantity <= 0 or price <= 0 or gross_value <= 0:
            raise ValueError(
                "Simulation produced a trade below persistence precision; "
                "rebalance dust must be suppressed by the accounting engine"
            )

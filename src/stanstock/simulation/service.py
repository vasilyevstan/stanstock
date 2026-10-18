from __future__ import annotations

import hashlib
import io
import json
import math
import re
import subprocess
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any
from uuid import UUID

import polars as pl
from django.db import transaction
from django.db.models import DecimalField
from django.utils import timezone

from stanstock.data.assets import (
    AssetStore,
    open_asset_store,
    read_checksummed_bytes,
    register_asset,
)
from stanstock.data.models import DataAsset, Listing, UniverseSnapshot
from stanstock.simulation.engine import AccountingEngine
from stanstock.simulation.hooks import CorporateEventHook
from stanstock.simulation.models import (
    SimulationDefinition,
    SimulationHolding,
    SimulationRun,
    SimulationTrade,
)
from stanstock.simulation.types import (
    ACCOUNTING_VERSION,
    LEGACY_ACCOUNTING_VERSION,
    SimulationConfig,
    SimulationResult,
    resolve_accounting_version,
)


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
    calculation_version: str = LEGACY_ACCOUNTING_VERSION
    comparison_coverage: dict[str, Any] | None = None


def persist_simulation_run(
    *,
    definition: SimulationDefinition,
    universe_snapshot: UniverseSnapshot,
    result: SimulationResult,
    input_prices: pl.DataFrame | None = None,
    input_signals: pl.DataFrame | None = None,
    input_benchmark: pl.DataFrame | None = None,
    input_fx: pl.DataFrame | None = None,
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
    - Exact input frames (prices, signals, benchmark, FX) are persisted as
      immutable Parquet DataAssets under the run ID to guarantee
      reproducibility. The price frame keeps each listing's native price and
      currency next to its converted value, and the FX frame keeps the dated
      rate, the observation actually used, its carry distance, and its
      derivation path, so a converted run can be re-derived without
      re-reading mutable current state.
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
    v2 = resolve_accounting_version(result.config.custom_parameters) == ACCOUNTING_VERSION
    if v2:
        result.config.validate()
        if not register_data_asset:
            raise ValueError("V2 simulation results require immutable registration")
        _validate_v2_persistence(result)

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
    contract = (
        _calculation_contract(
            result,
            definition_id=str(definition.id),
            universe_snapshot_id=str(universe_snapshot.id),
            code_revision=rev,
            execution_revision=_get_git_revision(),
        )
        if v2
        else None
    )

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

        stored_fx = None
        if input_fx is not None:
            fx_rel_path = f"simulations/{actual_run_id}/inputs/fx.parquet"
            stored_fx = store.write_frame(fx_rel_path, input_fx)
            written_rel_paths.append(fx_rel_path)

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

                if stored_fx is not None and input_fx is not None:
                    fx_asset = register_asset(
                        provider="simulation",
                        kind="simulation_input_fx",
                        subject=str(actual_run_id),
                        stored=stored_fx,
                        retrieved_at=now,
                        available_at=now,
                        period_start=result.metrics.start_date,
                        period_end=result.metrics.end_date,
                        metadata={
                            "row_count": input_fx.height,
                            "columns": input_fx.columns,
                            "base_currency": result.config.base_currency,
                            "native_currencies": result.metrics.fx_native_currencies,
                            "max_carry_days_used": result.metrics.fx_max_carry_days_used,
                        },
                    )
                    input_assets_info["fx"] = {
                        "asset_id": str(fx_asset.id),
                        "sha256": stored_fx.sha256,
                        "relative_path": stored_fx.relative_path,
                        "row_count": input_fx.height,
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
                        **({"calculation_contract": contract} if contract is not None else {}),
                    },
                )

            metrics_payload = result.metrics.to_dict()
            metrics_payload["input_assets"] = input_assets_info
            if contract is not None:
                metrics_payload["calculation_contract"] = contract
                json.dumps(metrics_payload, allow_nan=False)

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
    contract = _validated_calculation_contract(run)
    if contract is not None:
        frames = {
            name: pl.read_parquet(io.BytesIO(payload))
            for name, payload in _verified_simulation_asset_payloads(run, store)
        }
        daily_curves = frames.pop("result")
        input_frames = frames
    else:
        daily_curves = store.read_frame(run.result_asset_key)
        input_frames = {}
        for key, info in run.metrics.get("input_assets", {}).items():
            if isinstance(info, dict) and "relative_path" in info:
                input_frames[key] = store.read_frame(info["relative_path"])

    return LoadedSimulationRun(
        run=run,
        daily_curves=daily_curves,
        holdings_count=run.holdings.count(),
        trades_count=run.trades.count(),
        input_assets=input_frames,
        calculation_version=(
            ACCOUNTING_VERSION if contract is not None else LEGACY_ACCOUNTING_VERSION
        ),
        comparison_coverage=None if contract is None else contract["comparison_coverage"],
    )


def execute_and_persist_simulation(
    *,
    definition: SimulationDefinition,
    universe_snapshot: UniverseSnapshot,
    prices: pl.DataFrame,
    signals: pl.DataFrame | None = None,
    benchmark_prices: pl.DataFrame | None = None,
    fx_rates: pl.DataFrame | None = None,
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
            fx_rates=fx_rates,
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
        input_fx=fx_rates,
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


def _calculation_contract(
    result: SimulationResult,
    *,
    definition_id: str,
    universe_snapshot_id: str,
    code_revision: str,
    execution_revision: str,
) -> dict[str, Any]:
    """The one v2 provenance document, copied identically to run and registration.

    V2 pairs immediately adjacent calendar endpoints; totals need both exact
    boundaries, path drawdown needs the full path. Portfolio drawdown includes
    inception capital. First-row return and legacy volatility conventions stay
    unchanged. Config is the effective execution snapshot, never today's mutable
    definition; declared code revision does not replace actual execution identity.
    """
    if re.fullmatch(r"[0-9a-f]{40}", execution_revision) is None:
        raise ValueError("Simulation execution revision is unavailable")
    snapshot = result.config.to_dict()
    config_bytes = json.dumps(
        snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        "version": ACCOUNTING_VERSION,
        "definition_id": definition_id,
        "universe_snapshot_id": universe_snapshot_id,
        "effective_config": snapshot,
        "effective_config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "code_revision": code_revision,
        "execution_revision": execution_revision,
        "input_hash": result.input_hash,
        "comparison_coverage": result.metrics.comparison_coverage,
    }


def _validated_calculation_contract(run: SimulationRun) -> dict[str, Any] | None:
    """Bind stored metrics and parent identities to the immutable registration."""
    if not isinstance(run.metrics, dict):
        raise ValueError("Simulation metrics must be an object")
    contract = run.metrics.get("calculation_contract")
    if contract is None:
        config = SimulationConfig.from_dict(run.definition.config)
        registrations = DataAsset.objects.filter(
            provider="simulation", kind="simulation_result", subject=str(run.id)
        )
        if (
            resolve_accounting_version(config.custom_parameters) != LEGACY_ACCOUNTING_VERSION
            or "comparison_coverage" in run.metrics
            or any(
                not isinstance(asset.metadata, dict) or "calculation_contract" in asset.metadata
                for asset in registrations
            )
        ):
            raise ValueError("Simulation calculation provenance is missing")
        return None
    if not isinstance(contract, dict) or set(contract) != {
        "version",
        "definition_id",
        "universe_snapshot_id",
        "effective_config",
        "effective_config_hash",
        "code_revision",
        "execution_revision",
        "input_hash",
        "comparison_coverage",
    }:
        raise ValueError("Simulation calculation provenance is invalid")
    config = SimulationConfig.from_dict(contract["effective_config"])
    config.validate()
    version = resolve_accounting_version(config.custom_parameters)
    config_hash = hashlib.sha256(
        json.dumps(
            config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    assets = list(
        DataAsset.objects.filter(
            provider="simulation",
            kind="simulation_result",
            subject=str(run.id),
        )
    )
    if (
        version != ACCOUNTING_VERSION
        or contract["version"] != version
        or contract["definition_id"] != str(run.definition_id)
        or contract["universe_snapshot_id"] != str(run.universe_snapshot_id)
        or contract["effective_config_hash"] != config_hash
        or contract["input_hash"] != run.input_hash
        or contract["code_revision"] != run.code_revision
        or not isinstance(contract["execution_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", contract["execution_revision"]) is None
        or contract["comparison_coverage"] != run.metrics.get("comparison_coverage")
        or not isinstance(contract["comparison_coverage"], dict)
        or len(assets) != 1
    ):
        raise ValueError("Simulation calculation provenance is invalid")
    asset = assets[0]
    if (
        not isinstance(asset.metadata, dict)
        or asset.relative_path != run.result_asset_key
        or asset.metadata.get("calculation_contract") != contract
        or asset.metadata.get("input_assets") != run.metrics.get("input_assets")
        or asset.metadata.get("metrics")
        != {
            key: value
            for key, value in run.metrics.items()
            if key not in {"calculation_contract", "input_assets"}
        }
    ):
        raise ValueError("Simulation run differs from its immutable registration")
    return contract


def _verified_simulation_asset_payloads(
    run: SimulationRun, store: AssetStore
) -> Iterator[tuple[str, bytes]]:
    assets = list(
        DataAsset.objects.filter(
            provider="simulation",
            kind="simulation_result",
            subject=str(run.id),
            relative_path=run.result_asset_key,
        )
    )
    if len(assets) != 1:
        raise ValueError("Simulation result registration is invalid")
    yield "result", read_checksummed_bytes(store, assets[0])
    inputs = run.metrics.get("input_assets")
    if not isinstance(inputs, dict):
        raise ValueError("Simulation input registration is invalid")
    for key, info in inputs.items():
        if (
            key not in {"prices", "signals", "benchmark", "fx"}
            or not isinstance(info, dict)
            or not isinstance(info.get("relative_path"), str)
        ):
            raise ValueError("Simulation input registration is invalid")
        asset = DataAsset.objects.filter(
            id=UUID(str(info.get("asset_id"))),
            provider="simulation",
            kind=f"simulation_input_{key}",
            subject=str(run.id),
            relative_path=info["relative_path"],
            sha256=info.get("sha256"),
        ).first()
        if asset is None:
            raise ValueError("Simulation input registration is invalid")
        yield key, read_checksummed_bytes(store, asset)


def simulation_calculation_disclosure(run: SimulationRun) -> dict[str, Any]:
    """Validate before describing persisted metrics; never relabel old history."""
    contract = _validated_calculation_contract(run)
    if contract is None:
        return {
            "version": LEGACY_ACCOUNTING_VERSION,
            "coverage": None,
            "execution_revision": None,
            "mode_label": run.definition.get_mode_display(),
            "grade_label": run.universe_snapshot.get_grade_display(),
            "message": (
                "Legacy calculation: benchmark daily comparisons may bridge missing sessions, "
                "and initial trading costs may be omitted from drawdown. "
                "Historical results have not been recalculated."
            ),
        }
    for _name, _payload in _verified_simulation_asset_payloads(run, open_asset_store()):
        pass
    config = contract["effective_config"]
    return {
        "version": contract["version"],
        "coverage": contract["comparison_coverage"],
        "execution_revision": contract["execution_revision"],
        "mode_label": SimulationDefinition.Mode(config["mode"]).label,
        "grade_label": UniverseSnapshot.Grade(config["grade"]).label,
        "message": (
            "Benchmark daily comparisons use adjacent matched sessions. "
            "Drawdown includes inception capital."
        ),
    }


def _validate_v2_persistence(result: SimulationResult) -> None:
    json.dumps(result.config.to_dict(), allow_nan=False)
    json.dumps(result.metrics.to_dict(), allow_nan=False)
    if result.metrics.comparison_coverage is None:
        raise ValueError("V2 simulation comparison coverage is missing")
    # Validate actual model precision BEFORE file/row output. No clamping,
    # signed-zero rewriting, or retrospective validation of legacy runs.
    for model, frame, columns in (
        (SimulationHolding, result.holdings, ("quantity", "price", "market_value", "weight")),
        (SimulationTrade, result.trades, ("quantity", "price", "gross_value", "costs")),
    ):
        for row in frame.iter_rows(named=True):
            for name in columns:
                value = row[name]
                field = model._meta.get_field(name)
                if not isinstance(field, DecimalField):
                    raise TypeError("Simulation persistence field must be a Decimal")
                places = field.decimal_places
                digits = field.max_digits
                assert places is not None and digits is not None
                if not math.isfinite(value) or abs(value) >= 10 ** (digits - places):
                    raise ValueError("Simulation value exceeds persistence capacity")
                rounded = Decimal(f"{value:.{places}f}")
                if name in {"quantity", "price", "gross_value"} and rounded <= 0:
                    raise ValueError("Simulation value is below persistence precision")

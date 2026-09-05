from __future__ import annotations

from datetime import date, datetime
from typing import cast
from uuid import UUID

import polars as pl
from django.utils import timezone

from stanstock.data.asof import AsOfData, PriceFrameSchemaError
from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset, UniverseMembership, UniverseSnapshot
from stanstock.research.models import StockAnalysis
from stanstock.simulation.models import SimulationDefinition, SimulationRun
from stanstock.simulation.service import execute_and_persist_simulation
from stanstock.simulation.types import (
    RebalanceFrequency,
    SimulationConfig,
    SimulationGrade,
    SimulationMode,
    SimulationResult,
    SimulationWorkflowError,
)


def build_price_panel(
    *,
    snapshot: UniverseSnapshot,
    start_date: date,
    end_date: date,
    listing_ids: list[UUID] | None = None,
    provider: str = "synthetic_demo",
    benchmark_subject: str | None = None,
    base_currency: str | None = None,
    decision_time: datetime | None = None,
    asset_store: AssetStore | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Build a normalized price panel from a UniverseSnapshot's eligible listings."""
    d_time = decision_time or timezone.now()
    if end_date > d_time.date():
        raise SimulationWorkflowError(
            f"end_date ({end_date.isoformat()}) cannot be after decision date "
            f"({d_time.date().isoformat()})."
        )
    if start_date > end_date:
        raise SimulationWorkflowError(
            f"start_date ({start_date.isoformat()}) cannot be after "
            f"end_date ({end_date.isoformat()})."
        )

    memberships = UniverseMembership.objects.filter(
        snapshot=snapshot,
        eligible=True,
    ).select_related("listing__security")
    normalized_currency = base_currency.upper() if base_currency else None

    if listing_ids is not None:
        if not listing_ids:
            raise SimulationWorkflowError("listing_ids cannot be empty when provided.")
        # Reject duplicate selected UUIDs
        if len(listing_ids) != len(set(listing_ids)):
            raise SimulationWorkflowError("Duplicate listing UUIDs detected in selection.")
        target_uuids = {UUID(str(lid)) for lid in listing_ids}
        memberships = memberships.filter(listing_id__in=target_uuids)
        found_uuids = set(memberships.values_list("listing_id", flat=True))
        missing = target_uuids - found_uuids
        if missing:
            missing_str = sorted(str(u) for u in missing)
            raise SimulationWorkflowError(
                f"Listings {missing_str} are not eligible members of snapshot {snapshot.id}."
            )
    elif normalized_currency is not None:
        memberships = memberships.filter(listing__currency=normalized_currency)

    memberships_list = list(memberships)
    if not memberships_list:
        raise SimulationWorkflowError(f"No eligible listings found for snapshot {snapshot.id}.")
    currencies = sorted({membership.listing.currency.upper() for membership in memberships_list})
    if len(currencies) > 1:
        raise SimulationWorkflowError(
            "Simulation selections must use one native currency until point-in-time FX "
            f"conversion is implemented. Found: {', '.join(currencies)}."
        )
    panel_currency = currencies[0]
    if normalized_currency is not None and panel_currency != normalized_currency:
        raise SimulationWorkflowError(
            f"Selected listings use {panel_currency}, not requested base currency "
            f"{normalized_currency}."
        )

    store = asset_store or AssetStore()
    asof = AsOfData(d_time, store)

    frames: list[pl.DataFrame] = []
    for m in memberships_list:
        listing = m.listing
        symbol = listing.provider_symbol or listing.ticker
        try:
            raw_frame = asof.price_frame(
                provider=provider,
                subject=symbol,
                through_date=end_date,
            )
        except (DataAsset.DoesNotExist, PriceFrameSchemaError, FileNotFoundError, OSError) as exc:
            raise SimulationWorkflowError(
                f"Failed to load price history for listing {listing.ticker} ({listing.id}): {exc}"
            ) from exc

        # Filter between start_date and end_date
        filtered = raw_frame.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date))
        if filtered.height == 0:
            raise SimulationWorkflowError(
                f"Listing {listing.ticker} ({listing.id}) has no usable price rows "
                f"between {start_date} and {end_date}."
            )

        select_cols = [
            pl.col("date").cast(pl.Date),
            pl.lit(str(listing.id)).cast(pl.Utf8).alias("listing_id"),
            pl.col("close").cast(pl.Float64),
        ]
        if "open" in filtered.columns:
            select_cols.append(pl.col("open").cast(pl.Float64))
        select_cols.append(pl.lit(listing.ticker).cast(pl.Utf8).alias("symbol"))
        select_cols.append(pl.lit(panel_currency).cast(pl.Utf8).alias("currency"))
        frames.append(filtered.select(select_cols))

    price_panel = pl.concat(frames).sort(["date", "listing_id"])
    if listing_ids is not None:
        raw_inception_date = price_panel["date"].min()
        if raw_inception_date is None:
            raise SimulationWorkflowError("Selected listings have no usable inception date.")
        inception_date = cast(date, raw_inception_date)
        inception_listing_ids = set(
            price_panel.filter((pl.col("date") == inception_date) & (pl.col("close") > 0))[
                "listing_id"
            ].to_list()
        )
        missing_inception = sorted(
            str(listing_id)
            for listing_id in listing_ids
            if str(listing_id) not in inception_listing_ids
        )
        if missing_inception:
            raise SimulationWorkflowError(
                "Selected listings lack a usable inception close on "
                f"{inception_date}: {', '.join(missing_inception)}."
            )

    benchmark_frame: pl.DataFrame | None = None
    if benchmark_subject:
        try:
            raw_bench = asof.price_frame(
                provider=provider,
                subject=benchmark_subject,
                through_date=end_date,
            )
        except (DataAsset.DoesNotExist, PriceFrameSchemaError, FileNotFoundError, OSError) as exc:
            raise SimulationWorkflowError(
                f"Failed to load benchmark price history for '{benchmark_subject}': {exc}"
            ) from exc

        filtered_bench = raw_bench.filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )
        if filtered_bench.height == 0:
            raise SimulationWorkflowError(
                f"Benchmark subject '{benchmark_subject}' has no usable price rows "
                f"between {start_date} and {end_date}."
            )

        benchmark_frame = filtered_bench.select(
            [
                pl.col("date").cast(pl.Date),
                pl.col("close").cast(pl.Float64),
            ]
        ).sort("date")

    return price_panel, benchmark_frame


def build_signals_for_backtest(
    *,
    snapshot: UniverseSnapshot,
    start_date: date,
    end_date: date,
    price_panel: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build signals DataFrame only from persisted StockAnalysis rows for this snapshot.

    Never manufactures or reconstructs signals from future or latest analyses.
    Requires completed analysis runs and explicitly rejects duplicate signals.
    """
    analyses = list(
        StockAnalysis.objects.filter(
            run__universe_snapshot=snapshot,
            run__status="complete",
            run__target_date__gte=start_date,
            run__target_date__lte=end_date,
        )
        .select_related("listing", "run")
        .order_by("run__target_date", "listing_id")
    )

    if not analyses:
        raise SimulationWorkflowError(
            f"No persisted signals found for snapshot {snapshot.id} with target dates "
            f"between {start_date} and {end_date}."
        )

    if price_panel is not None and price_panel.height > 0:
        available_listing_ids = set(price_panel["listing_id"].unique().to_list())
        analyses = [
            analysis for analysis in analyses if str(analysis.listing_id) in available_listing_ids
        ]
        if not analyses:
            raise SimulationWorkflowError(
                "No persisted signals match the listings in the selected currency panel."
            )

    invalid_cutoffs = [
        analysis
        for analysis in analyses
        if analysis.run.data_cutoff.date() > analysis.run.target_date
    ]
    if invalid_cutoffs:
        raise SimulationWorkflowError(
            "Backtest signals contain data cutoffs after their logical target dates."
        )

    if snapshot.grade == UniverseSnapshot.Grade.OBSERVED:
        late_signals = [
            analysis
            for analysis in analyses
            if analysis.run.generated_at.date() != analysis.run.target_date
        ]
        if late_signals:
            raise SimulationWorkflowError(
                "Observed backtests require signals generated on their target date; "
                "late-generated analyses are research reconstructions."
            )

    # Explicitly reject ambiguous duplicate (target_date, listing_id) signals
    seen: set[tuple[date, UUID]] = set()
    duplicates: set[tuple[date, UUID]] = set()
    for a in analyses:
        sig_key = (a.run.target_date, a.listing_id)
        if sig_key in seen:
            duplicates.add(sig_key)
        seen.add(sig_key)

    if duplicates:
        dup_list = [f"date={d.isoformat()}, listing={lid}" for d, lid in sorted(duplicates)]
        raise SimulationWorkflowError(
            f"Ambiguous duplicate signals detected for snapshot {snapshot.id}: {dup_list}"
        )

    signals = pl.DataFrame(
        [
            {
                "date": a.run.target_date,
                "listing_id": str(a.listing_id),
                "score": float(a.overall_score),
                "symbol": a.listing.ticker,
                "generated_at": a.run.generated_at,
                "data_cutoff": a.run.data_cutoff,
            }
            for a in analyses
        ],
        schema={
            "date": pl.Date,
            "listing_id": pl.Utf8,
            "score": pl.Float64,
            "symbol": pl.Utf8,
            "generated_at": pl.Datetime(time_zone="UTC"),
            "data_cutoff": pl.Datetime(time_zone="UTC"),
        },
    ).sort(["date", "listing_id"])

    # Ensure at least one later execution session exists in the price panel
    if price_panel is not None and price_panel.height > 0:
        trading_dates = sorted(set(price_panel["date"].to_list()))
        sig_dates = sorted(set(signals["date"].to_list()))
        has_later_session = any(any(td > sd for td in trading_dates) for sd in sig_dates)
        if not has_later_session:
            raise SimulationWorkflowError(
                "No later execution session exists in the price panel after the signal dates. "
                "A signal dated T can trade only on dates strictly after T."
            )

    return signals


def run_simulation_workflow(
    *,
    name: str,
    mode: str,
    snapshot: UniverseSnapshot,
    start_date: date,
    end_date: date,
    starting_capital: float = 100_000.0,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    top_n: int | None = None,
    selected_listing_ids: list[UUID] | None = None,
    rebalance_frequency: str | None = None,
    benchmark_subject: str | None = None,
    base_currency: str | None = None,
    provider: str = "synthetic_demo",
    asset_store: AssetStore | None = None,
    decision_time: datetime | None = None,
    code_revision: str | None = None,
) -> tuple[SimulationDefinition, SimulationRun, SimulationResult]:
    """Execute complete validated simulation workflow for portfolio or backtest mode.

    Creates SimulationDefinition only after all inputs, prices, and signals are validated.
    """
    clean_name = (name or "").strip()
    if not clean_name:
        raise SimulationWorkflowError("Simulation name is required.")

    if mode not in SimulationDefinition.Mode.values:
        raise SimulationWorkflowError(f"Invalid simulation mode '{mode}'.")

    d_time = decision_time or timezone.now()
    if end_date > d_time.date():
        raise SimulationWorkflowError(
            f"end_date ({end_date.isoformat()}) cannot be after decision date "
            f"({d_time.date().isoformat()})."
        )
    if start_date > end_date:
        raise SimulationWorkflowError("start_date cannot be after end_date.")

    if starting_capital <= 0:
        raise SimulationWorkflowError("starting_capital must be positive.")

    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise SimulationWorkflowError("Transaction costs and slippage must be non-negative.")

    if mode == SimulationDefinition.Mode.PORTFOLIO:
        if not selected_listing_ids:
            raise SimulationWorkflowError(
                "Portfolio mode requires explicit selected listing UUIDs."
            )
        if len(selected_listing_ids) != len(set(selected_listing_ids)):
            raise SimulationWorkflowError("Duplicate listing UUIDs detected in selection.")
        freq = RebalanceFrequency.NEVER
        cfg_top_n = None
        cfg_selected_symbols = [str(UUID(str(lid))) for lid in selected_listing_ids]
    else:  # BACKTEST
        if not top_n or top_n <= 0:
            raise SimulationWorkflowError("Backtest mode requires a positive top_n.")
        freq = RebalanceFrequency(rebalance_frequency or "monthly")
        cfg_top_n = top_n
        cfg_selected_symbols = None

    # 1. Build price panel
    price_panel, bench_frame = build_price_panel(
        snapshot=snapshot,
        start_date=start_date,
        end_date=end_date,
        listing_ids=selected_listing_ids,
        provider=provider,
        benchmark_subject=benchmark_subject,
        base_currency=base_currency,
        decision_time=d_time,
        asset_store=asset_store,
    )

    # 2. Build signals for backtest mode
    if mode == SimulationDefinition.Mode.BACKTEST:
        signals: pl.DataFrame | None = build_signals_for_backtest(
            snapshot=snapshot,
            start_date=start_date,
            end_date=end_date,
            price_panel=price_panel,
        )
    else:
        signals = None

    # 3. Build validated SimulationConfig
    config = SimulationConfig(
        name=clean_name,
        mode=SimulationMode(mode),
        grade=SimulationGrade(snapshot.grade),
        starting_capital=starting_capital,
        rebalance_frequency=freq,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        top_n=cfg_top_n,
        selected_symbols=cfg_selected_symbols,
        benchmark_symbol=benchmark_subject,
        base_currency=str(price_panel["currency"][0]),
    )
    config.validate()

    # 4. Create SimulationDefinition ONLY after all validations succeed
    definition = SimulationDefinition.objects.create(
        name=config.name,
        mode=config.mode.value,
        config=config.to_dict(),
    )

    # 5. Execute and persist
    run, result = execute_and_persist_simulation(
        definition=definition,
        universe_snapshot=snapshot,
        prices=price_panel,
        signals=signals,
        benchmark_prices=bench_frame,
        asset_store=asset_store,
        code_revision=code_revision,
    )

    return definition, run, result

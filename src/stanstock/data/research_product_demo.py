"""Fixed, genuinely synthetic inputs for the active research product."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
import polars as pl
from django.conf import settings
from django.utils import timezone
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.jobs import JobExecutionResult, execute_target_job, target_job_lock
from stanstock.core.models import JobRun
from stanstock.data.asof import verified_price_fields
from stanstock.data.assets import AssetStore, asset_ref_for, read_checksummed_bytes, register_asset
from stanstock.data.market_state import update_latest_market_data
from stanstock.data.models import Company, DataAsset, Listing, Region, Security
from stanstock.data.research_product import (
    CapturedProductIntake,
    capture_product_intake,
    load_product_intake,
    materialize_product_membership,
)
from stanstock.data.research_product_jobs import (
    _completed_product_run,
    _snapshot_for_intake,
    _successful_attempt,
)
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.price_product import SourceExecutionBinding
from stanstock.research.price_product_config import (
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_VERSION,
    default_price_product_config_path,
    load_price_product_config,
)
from stanstock.research.product_pipeline import select_product_source
from stanstock.research.service import analyze_snapshot

DEMO_OWNER_ID = "synthetic-demo"
PROVIDER = "synthetic_demo"
JOB_NAME = "refresh_demo_research_v1"
START_DATE = date(2019, 9, 3)
END_DATE = date(2026, 9, 11)
_ISSUANCE_KEY = "demo"
_DATASET = "research-product-demo@1"


@dataclass(frozen=True)
class DemoSeries:
    symbol: str
    name: str
    final_close: float
    drift: float
    innovation: float
    history_closes: int | None = None


DEMO_STOCKS = (
    DemoSeries("ZZRPUP", "Synthetic Rising", 80.0, 0.0007, 0.0015),
    DemoSeries("ZZRPDOWN", "Synthetic Falling", 30.0, -0.0005, 0.002),
    DemoSeries("ZZRPLOW", "Synthetic Speculative", 7.5, 0.0008, 0.002),
    DemoSeries("ZZRPNEW", "Synthetic Recent", 18.0, 0.0004, 0.002, 100),
)
_BENCHMARK = DemoSeries("SPY", "Synthetic Benchmark", 100.0, 0.0002, 0.003)


def execute_demo_product_refresh(
    *, target_date: date = END_DATE, store: AssetStore | None = None
) -> JobRun:
    """Use the production source/calculation/writer paths without real preferences."""
    if not settings.DEMO_MODE or not settings.RESEARCH_PRODUCT_ENABLED:
        raise ValueError("Synthetic research refresh requires the enabled demo product profile")
    calendar = get_calendar("XNYS")
    if not START_DATE <= target_date <= END_DATE or not calendar.is_session(target_date):
        raise ValueError("Demo target must be a generated XNYS session within the fixed history")
    store = store or AssetStore()

    def task(_job: JobRun) -> JobExecutionResult:
        intake = _load_intake(target_date, store)
        if intake is not None:
            completed = _completed_product_run(intake, store=store)
            if completed is not None:
                return JobExecutionResult(details=_details(intake, completed))
        listings = tuple(_listing(spec) for spec in DEMO_STOCKS)
        if intake is None:
            intake = capture_product_intake(
                target_date=target_date,
                evidence_grade="research",
                issuance_key=_ISSUANCE_KEY,
                owner_id=DEMO_OWNER_ID,
                entitlement_identity=_DATASET,
                policy_identity=PRODUCT_VERSION,
                core_listings=listings,
                saved_listings=(),
                captured_at=timezone.now(),
                store=store,
            )
        snapshot = _snapshot_for_intake(intake, store=store)
        if snapshot is None:
            assets = {
                spec.symbol: _seed_prices(spec, store=store) for spec in (*DEMO_STOCKS, _BENCHMARK)
            }
            source_time = timezone.now()
            config = load_price_product_config()
            states: dict[UUID, str] = {}
            admissions: dict[str, dict[str, Any]] = {}
            qualified: list[UUID] = []
            for listing in listings:
                asset = assets[listing.provider_symbol]
                frame = store.read_frame(asset.relative_path).filter(pl.col("date") <= target_date)
                available = min(frame.height, config.simulation.return_observations + 1)
                required = config.simulation.return_observations + 1
                admitted = available == required
                if admitted:
                    select_product_source(
                        listing=listing,
                        target_date=target_date,
                        decision_time=source_time,
                        provider=PROVIDER,
                        benchmark_subject="SPY",
                        source_execution=SourceExecutionBinding(
                            mode="synthetic_demo", evidence_grade="research"
                        ),
                        store=store,
                        config=config,
                        stock_asset=asset,
                        benchmark_asset=assets["SPY"],
                    )
                    qualified.append(listing.id)
                    fields = verified_price_fields(
                        asset,
                        cutoff=source_time,
                        target_date=target_date,
                        close_places=6,
                        store=store,
                    )
                    update_latest_market_data(
                        listing=listing,
                        session_date=target_date,
                        observed_at=asset.retrieved_at,
                        close=fields.close,
                        previous_close=fields.previous_close,
                        volume=fields.volume,
                        source_asset=asset,
                    )
                status = "admitted" if admitted else "insufficient_history"
                states[listing.id] = status
                admissions[listing.provider_symbol] = {
                    "status": status,
                    "reason_code": "qualified" if admitted else "insufficient_price_history",
                    "listing_id": str(listing.id),
                    "price_asset": asset_ref_for(asset).to_json(),
                    "bootstrap_attempted": False,
                    "history_qualification": {
                        "required_closes": required,
                        "available_closes": available,
                        "missing_closes": required - available,
                    },
                }
            if not qualified:
                raise ValueError("Demo target has no stock with the required 757-close history")
            snapshot = materialize_product_membership(
                intake=intake,
                qualified_listing_ids=qualified,
                candidate_states=states,
                captured_at=source_time,
                store=store,
                admissions=admissions,
                benchmark_asset=assets["SPY"],
            )
        results = analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=timezone.now(),
            target_date=target_date,
            issued_on_time=False,
            provider=PROVIDER,
            benchmark_subject="SPY",
            config_path=default_price_product_config_path(),
            store=store,
        )
        return JobExecutionResult(details=_details(intake, results[0].run))

    # All historical targets share the same immutable synthetic source files.
    with target_job_lock(job_name=_DATASET, region="us", target_date=END_DATE):
        job = execute_target_job(job_name=JOB_NAME, region="us", target_date=target_date, task=task)
        original = _successful_attempt(job)
        intake = _load_intake(target_date, store)
        completed = None if intake is None else _completed_product_run(intake, store=store)
        if intake is None or completed is None or original.details != _details(intake, completed):
            raise ValueError("Completed synthetic research job has inconsistent registered output")
        return job


def _load_intake(target_date: date, store: AssetStore) -> CapturedProductIntake | None:
    return load_product_intake(
        target_date=target_date, owner_id=DEMO_OWNER_ID, issuance_key=_ISSUANCE_KEY, store=store
    )


def _details(intake: CapturedProductIntake, run: AnalysisRun) -> dict[str, object]:
    return {
        "product": PRODUCT_VERSION,
        "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
        "provider": PROVIDER,
        "benchmark_subject": "SPY",
        "owner_id": DEMO_OWNER_ID,
        "intake_asset": asset_ref_for(intake.asset).to_json(),
        "snapshot_id": str(run.universe_snapshot_id),
        "analysis_run_id": str(run.pk),
        "evidence_grade": "research",
        "analyses": StockAnalysis.objects.filter(run=run).count(),
        "predictions": Prediction.objects.filter(analysis__run=run).count(),
    }


def _listing(spec: DemoSeries) -> Listing:
    identity = uuid5(NAMESPACE_URL, f"https://example.invalid/stanstock/{_DATASET}/{spec.symbol}")
    company, _ = Company.objects.get_or_create(
        id=identity, defaults={"name": spec.name, "country": "US", "sector": "Synthetic"}
    )
    security, _ = Security.objects.get_or_create(
        id=identity,
        defaults={
            "company": company,
            "name": spec.name,
            "security_type": Security.SecurityType.COMMON_STOCK,
        },
    )
    listing, _ = Listing.objects.get_or_create(
        id=identity,
        defaults={
            "security": security,
            "ticker": spec.symbol,
            "provider_symbol": spec.symbol,
            "exchange_mic": "XNYS",
            "currency": "USD",
            "region": Region.US,
            "valid_from": START_DATE,
        },
    )
    if (
        company.name != spec.name
        or company.country != "US"
        or security.company_id != company.pk
        or security.security_type != Security.SecurityType.COMMON_STOCK
        or listing.security_id != security.pk
        or listing.ticker != spec.symbol
        or listing.provider_symbol != spec.symbol
        or listing.exchange_mic != "XNYS"
        or listing.currency != "USD"
        or listing.region != Region.US
        or listing.valid_from != START_DATE
    ):
        raise ValueError("Synthetic research listing identity conflicts with its fixed dataset")
    return listing


def _seed_prices(spec: DemoSeries, *, store: AssetStore) -> DataAsset:
    prefix = f"demo/research-product-v1/{spec.symbol}"
    existing = DataAsset.objects.filter(relative_path=f"{prefix}.parquet").first()
    if existing is not None:
        if (
            existing.provider != PROVIDER
            or existing.kind != "price_history"
            or existing.subject != spec.symbol
            or existing.metadata.get("dataset") != _DATASET
        ):
            raise ValueError("Synthetic research source identity conflicts with its fixed dataset")
        read_checksummed_bytes(store, existing)
        return existing
    sessions = [
        stamp.date() for stamp in get_calendar("XNYS").sessions_in_range(START_DATE, END_DATE)
    ]
    offsets = np.arange(len(sessions), dtype=float)
    log_path = np.cumsum(spec.drift + spec.innovation * np.sin(offsets * 0.37))
    closes = spec.final_close * np.exp(log_path - log_path[-1])
    if spec.history_closes is not None:
        sessions = sessions[-spec.history_closes :]
        closes = closes[-spec.history_closes :]
    frame = pl.DataFrame(
        {"date": sessions, "close": closes, "volume": [1_000_000] * len(sessions)},
        schema_overrides={"date": pl.Date, "close": pl.Float64, "volume": pl.Int64},
    )
    payload = {
        "schema": "research-product-synthetic-prices@1",
        "subject": spec.symbol,
        "currency": "USD",
        "adjustment": "splits",
        "volume_adjustment_compatible": True,
        "synthetic": True,
        "dataset": _DATASET,
        "values": [
            {"date": row["date"].isoformat(), "close": row["close"], "volume": row["volume"]}
            for row in frame.iter_rows(named=True)
        ],
    }
    generated_at = timezone.now()
    raw = DataAsset.objects.filter(relative_path=f"{prefix}.json").first()
    if raw is None:
        raw = register_asset(
            provider=PROVIDER,
            kind="raw_price_history",
            subject=spec.symbol,
            stored=store.write_bytes(f"{prefix}.json", json.dumps(payload).encode()),
            retrieved_at=generated_at,
            available_at=generated_at,
            metadata={"synthetic": True, "dataset": _DATASET},
        )
    if (
        raw.provider != PROVIDER
        or raw.kind != "raw_price_history"
        or raw.subject != spec.symbol
        or json.loads(read_checksummed_bytes(store, raw)) != payload
    ):
        raise ValueError("Synthetic raw research source differs from its fixed dataset")
    return register_asset(
        provider=PROVIDER,
        kind="price_history",
        subject=spec.symbol,
        stored=store.write_frame(f"{prefix}.parquet", frame),
        retrieved_at=generated_at,
        available_at=generated_at,
        period_start=sessions[0],
        period_end=sessions[-1],
        metadata={
            "synthetic": True,
            "dataset": _DATASET,
            "currency": "USD",
            "adjustment": "splits",
            "volume_adjustment_compatible": True,
            "raw_asset_id": str(raw.pk),
            "raw_sha256": raw.sha256,
        },
    )

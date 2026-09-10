"""US-only Twelve Data ingestion and observed prediction workflow."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Any
from uuid import UUID

import polars as pl
from django.db import DatabaseError, transaction
from django.utils import timezone
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.data.assets import AssetStore, open_asset_store, register_asset
from stanstock.data.etfs import (
    INVESTABLE_US_ETF_MIC,
    INVESTABLE_US_ETF_SYMBOL,
    sync_investable_spy_from_asset,
)
from stanstock.data.management.config_loader import (
    config_hash,
    default_us_scoring_config_path,
    load_yaml_mapping,
)
from stanstock.data.market_state import update_latest_market_data
from stanstock.data.models import (
    Company,
    DataAsset,
    Listing,
    ProviderRecord,
    Region,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.data.provider_policy import (
    PRIVATE_USAGE_SCOPE,
    validate_provider_usage,
)
from stanstock.data.providers import twelve_data
from stanstock.data.providers.contracts import PriceSeries, StockCatalog, StockReference
from stanstock.data.providers.exceptions import (
    ProviderConfigurationError,
    ProviderDataError,
    ProviderError,
    ProviderQuotaError,
)
from stanstock.data.refresh_evidence import (
    UNIVERSE_MEMBERSHIP_EVIDENCE_KIND,
    build_membership_evidence_envelope,
    universe_snapshot_evidence_payload,
)
from stanstock.research import config as research_config
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.service import AnalysisOutputPaths, analyze_snapshot
from stanstock.research.timing import is_us_session_issuance_on_time

PROVIDER = twelve_data.PROVIDER
DEFAULT_DAILY_CREDIT_LIMIT = 800
DEFAULT_CREDITS_PER_MINUTE = 8
DEFAULT_MAX_SYMBOLS = 300
DEFAULT_CLOSE_DELAY_MINUTES = 30
_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,31}$")


@dataclass(frozen=True, slots=True)
class UsUniverseConfig:
    slug: str
    name: str
    description: str
    config_version: str
    country: str
    instrument_type: str
    currency: str
    exchanges: tuple[str, ...]
    benchmark_symbol: str
    benchmark_currency: str
    benchmark_type: str
    history_years: int
    minimum_history_sessions: int
    price_adjustment: str
    minimum_eligible: int
    maximum_symbols: int
    symbols: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LiveUsRunResult:
    snapshot: UniverseSnapshot
    analysis_run_id: UUID
    analyses: int
    predictions: int
    eligible: int
    excluded: int
    price_assets: int
    raw_assets: int
    credits_used: int
    benchmark_symbol: str
    catalog_asset_ids: tuple[UUID, ...]


class ProviderCreditBudget:
    """Coordinate the configured Twelve Data allowance across local jobs."""

    def __init__(
        self,
        *,
        provider: str = PROVIDER,
        enforce_spacing: bool = True,
        require_enabled: bool = True,
    ) -> None:
        self.provider = provider
        self.enforce_spacing = enforce_spacing
        self.require_enabled = require_enabled

    def preflight(self, credits: int) -> None:
        if credits <= 0:
            raise ValueError("credits must be positive")
        record = _provider_record(require_enabled=self.require_enabled)
        metadata = dict(record.metadata)
        daily_limit = _metadata_positive_int(
            metadata,
            "daily_credit_limit",
            DEFAULT_DAILY_CREDIT_LIMIT,
        )
        used = _usage_for_today(metadata)
        if used + credits > daily_limit:
            raise ProviderQuotaError(
                f"Twelve Data run requires {credits} credits but only "
                f"{daily_limit - used} locally tracked credits remain today"
            )

    def consume(self) -> None:
        now = timezone.now()
        with transaction.atomic():
            record = ProviderRecord.objects.select_for_update().get(provider=self.provider)
            if self.require_enabled and not record.enabled:
                raise ProviderConfigurationError(
                    "Twelve Data is disabled. Run configure_twelve_data --enable "
                    "after reviewing the private-use terms."
                )
            metadata = dict(record.metadata)
            daily_limit = _metadata_positive_int(
                metadata,
                "daily_credit_limit",
                DEFAULT_DAILY_CREDIT_LIMIT,
            )
            credits_per_minute = _metadata_positive_int(
                metadata,
                "credits_per_minute",
                DEFAULT_CREDITS_PER_MINUTE,
            )
            usage_date = now.astimezone(UTC).date().isoformat()
            used = _usage_for_today(metadata, today=usage_date)
            if used >= daily_limit:
                raise ProviderQuotaError(
                    f"Twelve Data locally tracked daily credit limit ({daily_limit}) is exhausted"
                )

            reserved_at = now
            raw_next = metadata.get("next_request_not_before")
            if isinstance(raw_next, str):
                try:
                    next_request = datetime.fromisoformat(raw_next)
                except ValueError as exc:
                    raise ProviderConfigurationError(
                        "ProviderRecord metadata contains an invalid "
                        "next_request_not_before timestamp"
                    ) from exc
                if next_request.tzinfo is None:
                    raise ProviderConfigurationError(
                        "ProviderRecord next_request_not_before must include a timezone"
                    )
                reserved_at = max(reserved_at, next_request)

            interval = 60.0 / credits_per_minute
            metadata.update(
                {
                    "credit_usage_date": usage_date,
                    "credits_used_local": used + 1,
                    "next_request_not_before": (
                        reserved_at + timedelta(seconds=interval)
                    ).isoformat(),
                }
            )
            record.metadata = metadata
            record.save(update_fields=["metadata"])

        if self.enforce_spacing:
            delay = (reserved_at - timezone.now()).total_seconds()
            if delay > 0:
                sleep(delay)


def load_us_universe_config(path: Path) -> UsUniverseConfig:
    raw = load_yaml_mapping(path)
    schema_version = _required_int(raw, "schema_version")
    if schema_version != 1:
        raise ValueError("US universe config schema_version must be 1")
    if _required_text(raw, "provider") != PROVIDER:
        raise ValueError(f"US universe config provider must be {PROVIDER!r}")

    symbols_raw = raw.get("symbols")
    exchanges_raw = raw.get("exchanges")
    if not isinstance(symbols_raw, list) or not symbols_raw:
        raise ValueError("US universe config requires a non-empty symbols list")
    if not isinstance(exchanges_raw, list) or not exchanges_raw:
        raise ValueError("US universe config requires a non-empty exchanges list")

    symbols = tuple(_normalize_symbol(value) for value in symbols_raw)
    if len(symbols) != len(set(symbols)):
        raise ValueError("US universe config contains duplicate symbols")
    maximum_symbols = _required_int(raw, "maximum_symbols")
    if not 1 <= maximum_symbols <= DEFAULT_MAX_SYMBOLS:
        raise ValueError(f"maximum_symbols must be between 1 and {DEFAULT_MAX_SYMBOLS}")
    if len(symbols) > maximum_symbols:
        raise ValueError(
            f"US universe has {len(symbols)} symbols, above configured maximum {maximum_symbols}"
        )
    minimum_eligible = _required_int(raw, "minimum_eligible")
    if not 1 <= minimum_eligible <= len(symbols):
        raise ValueError("minimum_eligible must be positive and no greater than the symbol count")
    history_years = _required_int(raw, "history_years")
    if not 1 <= history_years <= 20:
        raise ValueError("history_years must be between 1 and 20")
    minimum_history_sessions = _required_int(raw, "minimum_history_sessions")
    if not 1 <= minimum_history_sessions <= twelve_data.MAX_OUTPUT_SIZE:
        raise ValueError(
            f"minimum_history_sessions must be between 1 and {twelve_data.MAX_OUTPUT_SIZE}"
        )
    price_adjustment = _required_text(raw, "price_adjustment").lower()
    if price_adjustment != "splits":
        raise ValueError(
            "The live US workflow requires price_adjustment='splits' so its "
            "return convention stays explicit and stable"
        )

    country = _required_text(raw, "country")
    if country != "United States":
        raise ValueError("US universe config country must be 'United States'")
    instrument_type = _required_text(raw, "instrument_type")
    if instrument_type != "Common Stock":
        raise ValueError("US universe config instrument_type must be 'Common Stock'")
    currency = _required_text(raw, "currency").upper()
    if currency != "USD":
        raise ValueError("US universe config currency must be 'USD'")
    exchanges = tuple(_required_config_text(value, "exchange").upper() for value in exchanges_raw)
    if len(exchanges) != len(set(exchanges)):
        raise ValueError("US universe config contains duplicate exchanges")
    unsupported_exchanges = sorted(set(exchanges) - {"NASDAQ", "NYSE"})
    if unsupported_exchanges:
        raise ValueError(
            "US universe config exchanges are limited to NASDAQ and NYSE; "
            f"received {unsupported_exchanges!r}"
        )
    benchmark_symbol = _normalize_symbol(raw.get("benchmark_symbol"))
    if benchmark_symbol != INVESTABLE_US_ETF_SYMBOL:
        raise ValueError(f"US universe benchmark_symbol must be {INVESTABLE_US_ETF_SYMBOL!r}")
    if benchmark_symbol in symbols:
        raise ValueError("US universe benchmark_symbol must not also be a member symbol")
    benchmark_currency = _required_text(raw, "benchmark_currency").upper()
    if benchmark_currency != "USD":
        raise ValueError("US universe benchmark_currency must be 'USD'")
    benchmark_type = _required_text(raw, "benchmark_type")
    if benchmark_type != "ETF":
        raise ValueError("US universe benchmark_type must be 'ETF'")

    return UsUniverseConfig(
        slug=_required_text(raw, "slug"),
        name=_required_text(raw, "name"),
        description=str(raw.get("description") or ""),
        config_version=_required_text(raw, "config_version"),
        country=country,
        instrument_type=instrument_type,
        currency=currency,
        exchanges=exchanges,
        benchmark_symbol=benchmark_symbol,
        benchmark_currency=benchmark_currency,
        benchmark_type=benchmark_type,
        history_years=history_years,
        minimum_history_sessions=minimum_history_sessions,
        price_adjustment=price_adjustment,
        minimum_eligible=minimum_eligible,
        maximum_symbols=maximum_symbols,
        symbols=symbols,
        raw=raw,
    )


def resolve_us_target_date(
    *,
    decision_time: datetime,
    explicit_target: date | None = None,
    close_delay_minutes: int = DEFAULT_CLOSE_DELAY_MINUTES,
) -> tuple[date, str]:
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    if close_delay_minutes < 0:
        raise ValueError("close_delay_minutes cannot be negative")
    calendar = get_calendar("XNYS")
    candidate = calendar.date_to_session(decision_time.date(), direction="previous")
    ready_at = calendar.session_close(candidate) + timedelta(minutes=close_delay_minutes)
    if decision_time < ready_at.to_pydatetime():
        candidate = calendar.previous_session(candidate)
    candidate_date = candidate.date()

    target_date = explicit_target or candidate_date
    if not calendar.is_session(target_date.isoformat()):
        raise ValueError(f"US target date {target_date.isoformat()} is not an XNYS session")
    if target_date > candidate_date:
        raise ValueError(
            f"US target date {target_date.isoformat()} is not yet complete; latest "
            f"eligible session is {candidate_date.isoformat()}"
        )
    next_session = calendar.next_session(candidate)
    next_session_open = calendar.session_open(next_session).to_pydatetime()
    grade = UniverseSnapshot.Grade.RESEARCH
    if target_date == candidate_date and decision_time < next_session_open:
        grade = UniverseSnapshot.Grade.OBSERVED
    return target_date, grade


def is_us_prediction_on_time(*, target_date: date, generated_at: datetime) -> bool:
    return is_us_session_issuance_on_time(
        target_date=target_date,
        generated_at=generated_at,
    )


def _best_effort_unlink_if_orphaned(store: AssetStore, relative_path: str) -> None:
    """Best-effort cleanup for one candidate orphaned path after this
    call's own outer transaction has already rolled back.

    Content-addressed storage means an unrelated, already-committed run
    can legitimately share the identical `relative_path` (same content
    hashed to the same location); only unlink when no `DataAsset` row
    still references it. Every failure here -- the existence check itself,
    `store.resolve`, or the unlink -- is swallowed: this is cleanup for a
    failure already being propagated, and must never raise a *different*
    exception that would replace or mask the original one.
    """
    try:
        if not DataAsset.objects.filter(relative_path=relative_path).exists():
            store.resolve(relative_path).unlink(missing_ok=True)
    except (OSError, ValueError, DatabaseError):
        pass


def run_us_daily(
    *,
    config: UsUniverseConfig,
    target_date: date,
    snapshot_grade: str,
    api_key: str | None = None,
    decision_time: datetime | None = None,
    store: AssetStore | None = None,
    enforce_rate_limit: bool = True,
    require_on_time: bool = False,
) -> LiveUsRunResult:
    """Fetch, persist, and analyze one complete US target-date snapshot."""
    # `open_asset_store()` normalizes a default-construction failure (an
    # unwritable/misconfigured root) into a path-free, stable `ValueError`
    # subclass *before* any recovery or provider work below -- an explicitly
    # supplied `store` is passed through untouched.
    store = store or open_asset_store()
    existing_result = _existing_completed_result(
        config=config,
        target_date=target_date,
        store=store,
    )
    if existing_result is not None:
        return existing_result
    provider_record = _enabled_provider_record()
    key = twelve_data.resolve_api_key(api_key)
    history_start = _years_before(target_date, config.history_years)
    expected_credits = len(config.exchanges) + len(config.symbols) + 1
    budget = ProviderCreditBudget(enforce_spacing=enforce_rate_limit)
    budget.preflight(expected_credits)
    credits_used = 0

    try:
        catalogs: list[StockCatalog] = []
        for exchange in config.exchanges:
            budget.consume()
            credits_used += 1
            catalogs.append(
                twelve_data.fetch_stock_catalog(
                    exchange=exchange,
                    country=config.country,
                    instrument_type=config.instrument_type,
                    required_symbols=config.symbols,
                    api_key=key,
                )
            )
        references, exclusion_reasons = _resolve_references(
            config,
            catalogs,
            activated_plan=str(provider_record.metadata["plan"]),
        )

        series_by_symbol: dict[str, PriceSeries] = {}
        for symbol in config.symbols:
            if symbol in exclusion_reasons:
                continue
            budget.consume()
            credits_used += 1
            try:
                series = twelve_data.fetch_daily_price_series(
                    symbol,
                    start_date=history_start,
                    end_date=target_date,
                    adjustment=config.price_adjustment,
                    api_key=key,
                )
                _validate_listing_series(series, references[symbol], config)
            except (ProviderDataError, ValueError) as exc:
                exclusion_reasons[symbol] = f"Rejected provider data: {exc}"[:160]
                continue
            series_by_symbol[symbol] = series

        budget.consume()
        credits_used += 1
        benchmark_series = twelve_data.fetch_daily_price_series(
            config.benchmark_symbol,
            start_date=history_start,
            end_date=target_date,
            adjustment=config.price_adjustment,
            api_key=key,
        )
        _validate_benchmark_series(benchmark_series, config, target_date)
    except ProviderError as exc:
        _record_provider_failure(exc)
        raise

    for symbol, series in series_by_symbol.items():
        reason = _price_series_exclusion_reason(
            series,
            target_date=target_date,
            history_start=history_start,
            minimum_history_sessions=config.minimum_history_sessions,
        )
        if reason:
            exclusion_reasons[symbol] = reason
    eligible_symbols = tuple(symbol for symbol in config.symbols if symbol not in exclusion_reasons)
    if len(eligible_symbols) < config.minimum_eligible:
        raise ValueError(
            f"Only {len(eligible_symbols)} of {len(config.symbols)} configured US "
            f"symbols have a {target_date.isoformat()} daily bar; minimum is "
            f"{config.minimum_eligible}"
        )

    catalog_assets = [_persist_catalog(store, catalog) for catalog in catalogs]
    listings = _ensure_listings(config, references, target_date=target_date)
    price_assets: dict[str, DataAsset] = {}
    for symbol, series in series_by_symbol.items():
        price_assets[symbol] = _persist_price_series(
            store=store,
            series=series,
            listing=listings[symbol] if symbol in eligible_symbols else None,
        )
    benchmark_asset = _persist_price_series(
        store=store,
        series=benchmark_series,
        listing=None,
        resolved_mic_code=INVESTABLE_US_ETF_MIC,
        catalog_assets=catalog_assets,
    )

    analysis_clock = decision_time if decision_time is not None else timezone.now()
    analysis_time = max(
        analysis_clock,
        benchmark_series.retrieved_at,
        *(series.retrieved_at for series in series_by_symbol.values()),
    )
    output_paths = AnalysisOutputPaths()
    evidence_relative_path: str | None = None
    try:
        with transaction.atomic():
            if require_on_time:
                analysis_time = max(analysis_time, timezone.now())
                _require_automatic_on_time(
                    target_date=target_date,
                    generated_at=analysis_time,
                )
            snapshot, evidence_relative_path = _ensure_snapshot(
                config=config,
                target_date=target_date,
                grade=snapshot_grade,
                listings=listings,
                exclusion_reasons=exclusion_reasons,
                catalog_assets=catalog_assets,
                store=store,
                retrieved_at=analysis_time,
            )
            results = analyze_snapshot(
                universe_snapshot=snapshot,
                decision_time=analysis_time,
                target_date=target_date,
                issued_on_time=(
                    snapshot_grade == UniverseSnapshot.Grade.OBSERVED
                    and is_us_prediction_on_time(
                        target_date=target_date,
                        generated_at=analysis_time,
                    )
                ),
                provider=PROVIDER,
                benchmark_subject=config.benchmark_symbol,
                store=store,
                config_path=default_us_scoring_config_path(),
                output_paths=output_paths,
            )
            if not results:
                raise ValueError(
                    f"analyze_snapshot produced no persisted analyses for {target_date.isoformat()}"
                )
            first_analysis = results[0].analysis
            analysis_run_id = first_analysis.run_id
            if require_on_time:
                _require_automatic_on_time(
                    target_date=target_date,
                    generated_at=max(analysis_time, timezone.now()),
                )
    except Exception:
        orphan_candidates = [
            path
            for path in (
                output_paths.panel_relative_path,
                output_paths.manifest_relative_path,
                evidence_relative_path,
            )
            if path is not None
        ]
        for relative_path in orphan_candidates:
            _best_effort_unlink_if_orphaned(store, relative_path)
        raise
    _record_provider_success(
        at=analysis_time,
        target_date=target_date,
        eligible=len(eligible_symbols),
        excluded=len(config.symbols) - len(eligible_symbols),
    )
    sync_investable_spy_from_asset(
        asset=benchmark_asset,
        target_date=target_date,
        store=store,
    )
    return LiveUsRunResult(
        snapshot=snapshot,
        analysis_run_id=analysis_run_id,
        analyses=len(results),
        predictions=sum(len(result.predictions) for result in results),
        eligible=len(eligible_symbols),
        excluded=len(config.symbols) - len(eligible_symbols),
        price_assets=len(price_assets) + 1,
        raw_assets=len(price_assets) + 1 + len(catalog_assets),
        credits_used=credits_used,
        benchmark_symbol=config.benchmark_symbol,
        catalog_asset_ids=tuple(asset.id for asset in catalog_assets),
    )


def _require_automatic_on_time(*, target_date: date, generated_at: datetime) -> None:
    if is_us_prediction_on_time(
        target_date=target_date,
        generated_at=generated_at,
    ):
        return
    raise ValueError(
        f"Automatic issuance deadline passed before completing "
        f"{target_date.isoformat()} analysis; no late analysis or prediction "
        "was created."
    )


def _resolve_references(
    config: UsUniverseConfig,
    catalogs: list[StockCatalog],
    *,
    activated_plan: str,
) -> tuple[dict[str, StockReference], dict[str, str]]:
    allowed_exchanges = set(config.exchanges)
    by_symbol: dict[str, list[StockReference]] = {}
    for catalog in catalogs:
        if catalog.provider != PROVIDER:
            raise ValueError(
                f"Stock catalog provider {catalog.provider!r} did not match {PROVIDER!r}"
            )
        if catalog.exchange not in allowed_exchanges:
            raise ValueError(f"Stock catalog exchange {catalog.exchange!r} was not requested")
        for reference in catalog.references:
            if (
                reference.country == config.country
                and reference.instrument_type == config.instrument_type
                and reference.exchange in allowed_exchanges
                and reference.currency == config.currency
            ):
                by_symbol.setdefault(reference.symbol, []).append(reference)

    resolved: dict[str, StockReference] = {}
    exclusion_reasons: dict[str, str] = {}
    for symbol in config.symbols:
        candidates = by_symbol.get(symbol, [])
        if not candidates:
            raise ValueError(
                f"Configured symbol {symbol!r} was not present as a US common "
                "stock on the configured Twelve Data exchanges"
            )
        accessible_candidates = [
            candidate
            for candidate in candidates
            if _plan_allows(activated_plan, candidate.access_plan)
        ]
        if len(accessible_candidates) > 1:
            raise ValueError(
                f"Configured symbol {symbol!r} is ambiguous in the Twelve Data US catalog"
            )
        if accessible_candidates:
            resolved[symbol] = accessible_candidates[0]
            continue
        if len(candidates) != 1:
            raise ValueError(
                f"Configured symbol {symbol!r} is ambiguous in the Twelve Data US catalog"
            )
        resolved[symbol] = candidates[0]
        required_plan = candidates[0].access_plan or "an unspecified plan"
        exclusion_reasons[symbol] = (
            f"Requires Twelve Data plan {required_plan}; configured plan is {activated_plan}"
        )
    return resolved, exclusion_reasons


def _plan_allows(activated_plan: str, required_plan: str | None) -> bool:
    if required_plan is None or activated_plan == "custom":
        return True
    tiers = {"basic": 0, "grow": 1, "pro": 2, "ultra": 3}
    active_rank = tiers.get(activated_plan.casefold())
    required_rank = tiers.get(required_plan.casefold())
    return active_rank is not None and required_rank is not None and active_rank >= required_rank


def _price_series_exclusion_reason(
    series: PriceSeries,
    *,
    target_date: date,
    history_start: date,
    minimum_history_sessions: int,
) -> str:
    if series.bars[-1].trade_date != target_date:
        return f"No {target_date.isoformat()} daily bar"
    if len(series.bars) < minimum_history_sessions:
        return f"Only {len(series.bars)} history sessions; minimum is {minimum_history_sessions}"
    first_acceptable_date = history_start + timedelta(days=14)
    if series.bars[0].trade_date > first_acceptable_date:
        return (
            f"History begins {series.bars[0].trade_date.isoformat()}, after required "
            f"start {history_start.isoformat()}"
        )
    return ""


def _validate_listing_series(
    series: PriceSeries,
    reference: StockReference,
    config: UsUniverseConfig,
) -> None:
    if series.provider != PROVIDER:
        raise ValueError(f"Price response provider {series.provider!r} did not match {PROVIDER!r}")
    if series.symbol != reference.symbol:
        raise ValueError(
            f"Price response symbol {series.symbol!r} did not match catalog "
            f"symbol {reference.symbol!r}"
        )
    if series.currency != config.currency:
        raise ValueError(
            f"{series.symbol} price currency {series.currency!r} is not {config.currency}"
        )
    if series.instrument_type != config.instrument_type:
        raise ValueError(
            f"{series.symbol} returned type {series.instrument_type!r}, expected "
            f"{config.instrument_type!r}"
        )
    if series.mic_code != reference.mic_code:
        raise ValueError(
            f"{series.symbol} price MIC {series.mic_code!r} did not match catalog "
            f"MIC {reference.mic_code!r}"
        )
    if series.adjustment != config.price_adjustment:
        raise ValueError(
            f"{series.symbol} adjustment {series.adjustment!r} did not match "
            f"{config.price_adjustment!r}"
        )


def _validate_benchmark_series(
    series: PriceSeries,
    config: UsUniverseConfig,
    target_date: date,
) -> None:
    if series.provider != PROVIDER:
        raise ValueError(f"Benchmark provider {series.provider!r} did not match {PROVIDER!r}")
    if series.symbol != config.benchmark_symbol:
        raise ValueError("Benchmark response symbol did not match configuration")
    if series.currency != config.benchmark_currency:
        raise ValueError(
            f"Benchmark currency {series.currency!r} is not {config.benchmark_currency}"
        )
    if series.instrument_type != config.benchmark_type:
        raise ValueError(
            f"Benchmark type {series.instrument_type!r} is not {config.benchmark_type!r}"
        )
    if series.mic_code is not None and series.mic_code.upper() != INVESTABLE_US_ETF_MIC:
        raise ValueError(f"Benchmark MIC {series.mic_code!r} is not {INVESTABLE_US_ETF_MIC!r}")
    if series.adjustment != config.price_adjustment:
        raise ValueError(
            f"Benchmark adjustment {series.adjustment!r} did not match {config.price_adjustment!r}"
        )
    if series.bars[-1].trade_date != target_date:
        raise ValueError(
            f"Benchmark {series.symbol} has no {target_date.isoformat()} close; "
            f"latest returned date is {series.bars[-1].trade_date.isoformat()}"
        )


def _persist_catalog(store: AssetStore, catalog: StockCatalog) -> DataAsset:
    digest = hashlib.sha256(catalog.raw_bytes).hexdigest()
    stamp = _timestamp(catalog.retrieved_at)
    relative_path = f"raw/twelve_data/stock_catalog/{catalog.exchange}/{stamp}-{digest[:12]}.json"
    stored = store.write_bytes(relative_path, catalog.raw_bytes)
    try:
        with transaction.atomic():
            return register_asset(
                provider=PROVIDER,
                kind="stock_catalog",
                subject=catalog.exchange,
                stored=stored,
                retrieved_at=catalog.retrieved_at,
                available_at=catalog.retrieved_at,
                metadata={
                    "count": catalog.count,
                    "returned_rows": len(catalog.references),
                    "source_url": catalog.source_url,
                    "usage_scope": PRIVATE_USAGE_SCOPE,
                },
            )
    except Exception:
        if not DataAsset.objects.filter(relative_path=relative_path).exists():
            store.resolve(relative_path).unlink(missing_ok=True)
        raise


def _persist_price_series(
    *,
    store: AssetStore,
    series: PriceSeries,
    listing: Listing | None,
    resolved_mic_code: str | None = None,
    catalog_assets: list[DataAsset] | None = None,
) -> DataAsset:
    frame = _price_frame(series)
    digest = hashlib.sha256(series.raw_bytes).hexdigest()
    stamp = _timestamp(series.retrieved_at)
    safe_symbol = series.symbol.replace(".", "_")
    raw_path = f"raw/twelve_data/time_series/{safe_symbol}/{stamp}-{digest[:12]}.json"
    normalized_path = f"price_history/twelve_data/{safe_symbol}/{stamp}-{digest[:12]}.parquet"
    written_paths: list[str] = []
    try:
        stored_raw = store.write_bytes(raw_path, series.raw_bytes)
        written_paths.append(raw_path)
        stored_frame = store.write_frame(normalized_path, frame)
        written_paths.append(normalized_path)
        with transaction.atomic():
            raw_asset = register_asset(
                provider=PROVIDER,
                kind="raw_price_history",
                subject=series.symbol,
                stored=stored_raw,
                retrieved_at=series.retrieved_at,
                available_at=series.retrieved_at,
                period_start=series.bars[0].trade_date,
                period_end=series.bars[-1].trade_date,
                metadata={
                    "source_url": series.source_url,
                    "interval": "1day",
                    "adjustment": series.adjustment,
                    "usage_scope": PRIVATE_USAGE_SCOPE,
                },
            )
            price_metadata: dict[str, Any] = {
                "rows": frame.height,
                "currency": series.currency,
                "exchange": series.exchange,
                "mic_code": series.mic_code,
                "instrument_type": series.instrument_type,
                "interval": "1day",
                "adjustment": series.adjustment,
                "return_definition": "split_adjusted_price_return",
                "dividends_included": False,
                "raw_asset_id": str(raw_asset.id),
                "raw_sha256": raw_asset.sha256,
                "usage_scope": PRIVATE_USAGE_SCOPE,
            }
            if resolved_mic_code is not None:
                price_metadata["resolved_mic_code"] = resolved_mic_code
                price_metadata["mic_code_source"] = (
                    "provider" if series.mic_code is not None else "configured_spy_identity"
                )
            if catalog_assets is not None:
                # Recorded here (immutable, alongside the exact benchmark
                # evidence `benchmark_asset_for_completed_run` already
                # independently resolves for a completed run) so a later
                # zero-fetch recovery through `_existing_completed_result`
                # can recover the exact catalog assets this run's snapshot
                # was built from without guessing a "latest" row.
                # `id`/`sha256` are also read verbatim by
                # `_catalog_asset_ids_for_completed_run`; the extra
                # `provider`/`kind`/`subject` fields let the full-identity
                # `AssetRef` cross-check bind this leg the same way as the
                # membership envelope and market job details.
                price_metadata["catalog_assets"] = [
                    {
                        "id": str(asset.id),
                        "sha256": asset.sha256,
                        "provider": asset.provider,
                        "kind": asset.kind,
                        "subject": asset.subject,
                    }
                    for asset in catalog_assets
                ]
            price_asset = register_asset(
                provider=PROVIDER,
                kind="price_history",
                subject=series.symbol,
                stored=stored_frame,
                retrieved_at=series.retrieved_at,
                available_at=series.retrieved_at,
                period_start=series.bars[0].trade_date,
                period_end=series.bars[-1].trade_date,
                metadata=price_metadata,
            )
            if listing is not None:
                latest = series.bars[-1]
                previous = series.bars[-2] if len(series.bars) > 1 else None
                update_latest_market_data(
                    listing=listing,
                    session_date=latest.trade_date,
                    observed_at=series.retrieved_at,
                    close=latest.close,
                    previous_close=previous.close if previous is not None else None,
                    volume=latest.volume,
                    source_asset=price_asset,
                )
        return price_asset
    except Exception:
        for relative_path in written_paths:
            if not DataAsset.objects.filter(relative_path=relative_path).exists():
                store.resolve(relative_path).unlink(missing_ok=True)
        raise


def _price_frame(series: PriceSeries) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [bar.trade_date for bar in series.bars],
            "open": [float(bar.open) if bar.open is not None else None for bar in series.bars],
            "high": [float(bar.high) if bar.high is not None else None for bar in series.bars],
            "low": [float(bar.low) if bar.low is not None else None for bar in series.bars],
            "close": [float(bar.close) for bar in series.bars],
            "volume": [bar.volume for bar in series.bars],
        },
        schema_overrides={"date": pl.Date, "volume": pl.Int64},
    ).sort("date")


@transaction.atomic
def _ensure_listings(
    config: UsUniverseConfig,
    references: dict[str, StockReference],
    *,
    target_date: date,
) -> dict[str, Listing]:
    listings: dict[str, Listing] = {}
    for symbol in config.symbols:
        reference = references[symbol]
        candidates = list(
            Listing.objects.select_related("security__company").filter(
                provider_symbol=symbol,
                region=Region.US,
                is_active=True,
            )
        )
        if len(candidates) > 1:
            raise ValueError(f"Multiple active US listings use Twelve Data symbol {symbol!r}")
        if candidates:
            listing = candidates[0]
            if listing.currency != reference.currency or listing.exchange_mic != reference.mic_code:
                raise ValueError(
                    f"Existing listing identity for {symbol!r} conflicts with "
                    "the current Twelve Data catalog; review it as a dated "
                    "identifier change instead of overwriting history"
                )
            listings[symbol] = listing
            continue

        company = Company.objects.create(name=reference.name, country="US")
        security = Security.objects.create(
            company=company,
            security_type=Security.SecurityType.COMMON_STOCK,
            name=reference.name,
        )
        listings[symbol] = Listing.objects.create(
            security=security,
            ticker=symbol,
            exchange_mic=reference.mic_code,
            provider_symbol=symbol,
            currency=reference.currency,
            region=Region.US,
            valid_from=target_date,
            is_primary=True,
            is_active=True,
        )
    return listings


def _ensure_snapshot(
    *,
    config: UsUniverseConfig,
    target_date: date,
    grade: str,
    listings: dict[str, Listing],
    exclusion_reasons: dict[str, str],
    catalog_assets: list[DataAsset],
    store: AssetStore,
    retrieved_at: datetime,
) -> tuple[UniverseSnapshot, str | None]:
    """Return the snapshot and, only when this call wrote fresh evidence,
    that evidence's relative path -- so a caller whose *own* outer
    transaction later rolls back can still clean up the orphaned physical
    file once no `DataAsset` row survives for it.

    Deliberately *not* its own nested `@transaction.atomic`: its only
    caller, `run_us_daily`, already runs it inside one encompassing atomic
    block. A redundant inner savepoint boundary can itself raise on exit
    *after* releasing its savepoint, which would lose this function's
    return value (and therefore `evidence_relative_path`) even though the
    write already happened -- leaving an orphan the outer rollback cleanup
    could never learn about.
    """
    universe, _created = Universe.objects.update_or_create(
        slug=config.slug,
        defaults={
            "name": config.name,
            "description": config.description,
            "config_version": config.config_version,
        },
    )
    snapshot_payload = universe_snapshot_evidence_payload(
        config=config,
        catalog_assets=catalog_assets,
        listings=listings,
        exclusion_reasons=exclusion_reasons,
    )
    digest = config_hash(snapshot_payload)
    existing = UniverseSnapshot.objects.filter(
        universe=universe,
        as_of_date=target_date,
        grade=grade,
    ).first()
    if existing is not None:
        if existing.config_hash != digest:
            raise ValueError(
                f"Universe snapshot {existing.pk} already exists for "
                f"{target_date.isoformat()} with different evidence"
            )
        return existing, None

    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=target_date,
        grade=grade,
        config_hash=digest,
    )
    UniverseMembership.objects.bulk_create(
        [
            UniverseMembership(
                snapshot=snapshot,
                listing=listings[symbol],
                eligible=symbol not in exclusion_reasons,
                exclusion_reason=exclusion_reasons.get(symbol, ""),
            )
            for symbol in config.symbols
        ]
    )
    # Persist an envelope *around* the exact canonical bytes `digest` was
    # hashed from, as an immutable, independently-checksummed `DataAsset`,
    # so a later co-mutation of both `UniverseMembership` rows and this
    # snapshot's own `config_hash` field cannot silently authenticate
    # itself: the verifier re-reads *this* physical artifact rather than
    # recomputing from whatever the mutable rows currently say.
    envelope = build_membership_evidence_envelope(
        snapshot_id=snapshot.id, hash_payload=snapshot_payload, catalog_assets=catalog_assets
    )
    envelope_bytes = json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    relative_path = f"universe/{snapshot.id}/membership-evidence.json"
    # Decided from the filesystem *before* writing, never from an
    # in-transaction row-existence query: this call's own uncommitted
    # `DataAsset.objects.create` below is visible to `.exists()` on this
    # same connection even though nothing has actually committed yet, so a
    # failure occurring after that insert would otherwise be mistaken for
    # "a row already claims this file" and skip the cleanup it needs. A
    # fresh snapshot's evidence path embeds its own new UUID, so this file
    # can only already exist here as a genuine prior committed asset (never
    # deleted) or a leftover from an earlier failed attempt at identical
    # bytes (safe to leave alone either way).
    file_already_existed = store.resolve(relative_path).exists()
    try:
        written = store.write_bytes(relative_path, envelope_bytes)
        DataAsset.objects.create(
            provider="stanstock",
            kind=UNIVERSE_MEMBERSHIP_EVIDENCE_KIND,
            subject=str(snapshot.id),
            relative_path=written.relative_path,
            sha256=written.sha256,
            retrieved_at=retrieved_at,
            available_at=retrieved_at,
            metadata={"usage_scope": PRIVATE_USAGE_SCOPE},
        )
    except Exception:
        if not file_already_existed:
            store.resolve(relative_path).unlink(missing_ok=True)
        raise
    return snapshot, relative_path


def _existing_completed_result(
    *,
    config: UsUniverseConfig,
    target_date: date,
    store: AssetStore,
) -> LiveUsRunResult | None:
    universe = Universe.objects.filter(
        slug=config.slug,
        config_version=config.config_version,
    ).first()
    if universe is None:
        return None
    scoring_config = research_config.load_scoring_config(default_us_scoring_config_path())
    runs = list(
        AnalysisRun.objects.select_related("universe_snapshot")
        .filter(
            universe_snapshot__universe=universe,
            universe_snapshot__as_of_date=target_date,
            target_date=target_date,
            status="complete",
            config_version=scoring_config.version,
            config_hash=research_config.config_hash(scoring_config),
        )
        .order_by("-generated_at")
    )
    if not runs:
        return None
    if len(runs) > 1:
        raise ValueError(
            "Conflicting completed US analysis runs exist for "
            f"{target_date.isoformat()} and scoring config {scoring_config.version}"
        )
    run = runs[0]
    snapshot = run.universe_snapshot
    memberships = UniverseMembership.objects.filter(snapshot=snapshot)
    eligible = memberships.filter(eligible=True).count()
    excluded = memberships.filter(eligible=False).count()
    analyses = StockAnalysis.objects.filter(run=run).count()
    predictions = Prediction.objects.filter(analysis__run=run).count()
    benchmark_asset = benchmark_asset_for_completed_run(
        run=run,
        benchmark_symbol=config.benchmark_symbol,
        target_date=target_date,
    )
    sync_investable_spy_from_asset(
        asset=benchmark_asset,
        target_date=target_date,
        store=store,
    )
    catalog_asset_ids = _catalog_asset_ids_for_completed_run(
        benchmark_asset=benchmark_asset,
        target_date=target_date,
    )
    return LiveUsRunResult(
        snapshot=snapshot,
        analysis_run_id=run.id,
        analyses=analyses,
        predictions=predictions,
        eligible=eligible,
        excluded=excluded,
        price_assets=0,
        raw_assets=0,
        credits_used=0,
        benchmark_symbol=config.benchmark_symbol,
        catalog_asset_ids=catalog_asset_ids,
    )


def _catalog_asset_ids_for_completed_run(
    *,
    benchmark_asset: DataAsset,
    target_date: date,
) -> tuple[UUID, ...]:
    """Recover the exact catalog assets a completed run's snapshot used.

    The benchmark price asset's own immutable ``metadata`` was populated at
    creation time (see ``_persist_price_series``) with the exact
    ``catalog_assets`` (id + sha256) this run's ``_ensure_snapshot`` built
    its membership payload from -- the same identities
    `benchmark_asset_for_completed_run` already independently resolves for a
    completed run. Recovering them from that persisted evidence -- rather
    than guessing a "latest" `stock_catalog` row -- lets zero-fetch recovery
    (analysis committed, later ETF/benchmark projection failed) still
    supply the exact identities `refresh_verification` requires.
    """
    raw_entries = (
        benchmark_asset.metadata.get("catalog_assets")
        if isinstance(benchmark_asset.metadata, dict)
        else None
    )
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(
            f"Completed US run for {target_date.isoformat()} has no recorded catalog "
            "asset identities on its benchmark evidence"
        )
    asset_ids: list[UUID] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"Completed US run for {target_date.isoformat()} has a malformed catalog "
                "asset reference"
            )
        try:
            asset_id = UUID(str(raw_entry.get("id")))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Completed US run for {target_date.isoformat()} has an invalid catalog "
                "asset reference"
            ) from exc
        expected_sha256 = raw_entry.get("sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(
                f"Completed US run for {target_date.isoformat()} has an invalid catalog "
                "asset checksum"
            )
        asset = DataAsset.objects.filter(
            pk=asset_id, provider=PROVIDER, kind="stock_catalog"
        ).first()
        if asset is None or asset.sha256 != expected_sha256:
            raise ValueError(
                f"Completed US run for {target_date.isoformat()} has unavailable or "
                "conflicting catalog asset evidence"
            )
        asset_ids.append(asset_id)
    return tuple(asset_ids)


def benchmark_asset_for_completed_run(
    *,
    run: AnalysisRun,
    benchmark_symbol: str,
    target_date: date,
) -> DataAsset:
    identities: set[tuple[UUID, str]] = set()
    analyses = list(
        StockAnalysis.objects.filter(run=run).values_list(
            "listing__ticker",
            "data_quality",
        )
    )
    if not analyses:
        raise ValueError(f"Completed US run for {target_date.isoformat()} has no analyses")
    for ticker, data_quality in analyses:
        raw_assets = data_quality.get("source_assets") if isinstance(data_quality, dict) else None
        matching_assets = []
        if isinstance(raw_assets, list):
            matching_assets = [
                raw_asset
                for raw_asset in raw_assets
                if isinstance(raw_asset, dict)
                and raw_asset.get("provider") == PROVIDER
                and raw_asset.get("kind") == "price_history"
                and raw_asset.get("subject") == benchmark_symbol
            ]
        if len(matching_assets) != 1:
            raise ValueError(
                f"Completed US run for {target_date.isoformat()} analysis {ticker} "
                f"must reference exactly one {benchmark_symbol} benchmark asset"
            )
        for raw_asset in matching_assets:
            try:
                asset_id = UUID(str(raw_asset["id"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Completed US run for {target_date.isoformat()} has an invalid "
                    "benchmark asset reference"
                ) from exc
            asset_sha256 = raw_asset.get("sha256")
            if not isinstance(asset_sha256, str) or len(asset_sha256) != 64:
                raise ValueError(
                    f"Completed US run for {target_date.isoformat()} has an invalid "
                    "benchmark asset checksum"
                )
            identities.add((asset_id, asset_sha256))
    if len(identities) != 1:
        raise ValueError(
            f"Completed US run for {target_date.isoformat()} must reference exactly "
            f"one {benchmark_symbol} benchmark asset"
        )
    asset_id, expected_sha256 = identities.pop()
    asset = DataAsset.objects.filter(pk=asset_id).first()
    if asset is None or asset.sha256 != expected_sha256:
        raise ValueError(
            f"Completed US run for {target_date.isoformat()} has unavailable or "
            "conflicting benchmark evidence"
        )
    return asset


def _enabled_provider_record() -> ProviderRecord:
    return _provider_record(require_enabled=True)


def _provider_record(*, require_enabled: bool) -> ProviderRecord:
    try:
        record = ProviderRecord.objects.get(provider=PROVIDER)
    except ProviderRecord.DoesNotExist as exc:
        raise ProviderConfigurationError(
            "Twelve Data is not configured. Run configure_twelve_data --enable "
            "after setting TWELVE_DATA_API_KEY."
        ) from exc
    if require_enabled and not record.enabled:
        raise ProviderConfigurationError(
            "Twelve Data is disabled. Run configure_twelve_data --enable after "
            "reviewing the private-use terms."
        )
    if require_enabled:
        validate_provider_usage(record)
    return record


def _record_provider_failure(exc: ProviderError) -> None:
    status = "quota_exhausted" if isinstance(exc, ProviderQuotaError) else "provider_error"
    ProviderRecord.objects.filter(provider=PROVIDER).update(
        status=status,
        last_error=f"{type(exc).__name__}: {exc}",
    )


def _record_provider_success(
    *,
    at: datetime,
    target_date: date,
    eligible: int,
    excluded: int,
) -> None:
    record = ProviderRecord.objects.get(provider=PROVIDER)
    metadata = dict(record.metadata)
    metadata.update(
        {
            "last_target_date": target_date.isoformat(),
            "last_eligible_count": eligible,
            "last_excluded_count": excluded,
        }
    )
    record.status = "ok"
    record.last_success_at = at
    record.last_error = ""
    record.metadata = metadata
    record.save(update_fields=["status", "last_success_at", "last_error", "metadata"])


def _metadata_positive_int(
    metadata: dict[str, Any],
    key: str,
    default: int,
) -> int:
    raw = metadata.get(key, default)
    if isinstance(raw, bool):
        raise ProviderConfigurationError(
            f"ProviderRecord metadata {key!r} must be a positive integer"
        )
    try:
        value = int(str(raw))
    except ValueError as exc:
        raise ProviderConfigurationError(
            f"ProviderRecord metadata {key!r} must be a positive integer"
        ) from exc
    if value <= 0:
        raise ProviderConfigurationError(
            f"ProviderRecord metadata {key!r} must be a positive integer"
        )
    return value


def _usage_for_today(
    metadata: dict[str, Any],
    *,
    today: str | None = None,
) -> int:
    utc_date = today or timezone.now().astimezone(UTC).date().isoformat()
    if metadata.get("credit_usage_date") != utc_date:
        return 0
    raw = metadata.get("credits_used_local", 0)
    try:
        used = int(str(raw))
    except ValueError as exc:
        raise ProviderConfigurationError(
            "ProviderRecord credits_used_local must be an integer"
        ) from exc
    if used < 0:
        raise ProviderConfigurationError("ProviderRecord credits_used_local cannot be negative")
    return used


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return _required_config_text(value, key)


def _required_config_text(value: object, key: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"US universe config requires {key!r}")
    return text


def _required_int(payload: dict[str, Any], key: str) -> int:
    raw = payload.get(key)
    if isinstance(raw, bool):
        raise ValueError(f"US universe config {key!r} must be an integer")
    try:
        return int(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"US universe config {key!r} must be an integer") from exc


def _normalize_symbol(value: object) -> str:
    symbol = _required_config_text(value, "symbol").upper()
    if not _SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"Invalid US universe symbol {symbol!r}")
    return symbol


def _years_before(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Provider retrieval time must be timezone-aware")
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")

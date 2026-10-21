"""Purpose-isolated, fresh-batch liquidity evidence for research-product-v2.

No asset in this lane is a generic price candidate. Full public OHLC admission
precedes publication, and replay reopens both raw and normalized bytes. Decimal
parity and the limited NVDA operational canary are not corporate-action proof.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

import polars as pl
from django.contrib.auth import get_user_model
from django.db import transaction
from django.utils import timezone
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.models import JobRun
from stanstock.core.verification_types import AssetRef, RefreshVerificationError
from stanstock.data.asof import AsOfData, raw_price_asset_for
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    read_checksummed_bytes,
    resolve_asset_ref,
)
from stanstock.data.live_us import (
    _persist_price_series,
    _price_frame,
    _validate_my_list_price_series,
    _validate_my_list_reference,
)
from stanstock.data.models import DataAsset, Listing, ProviderRecord, Security
from stanstock.data.provider_policy import (
    TWELVE_DATA_PROVIDER,
    provider_plan_allows,
    validate_provider_usage,
)
from stanstock.data.providers import twelve_data
from stanstock.data.providers.contracts import PriceSeries, StockReference
from stanstock.data.providers.exceptions import (
    ProviderConfigurationError,
    ProviderError,
    ProviderQuotaError,
)
from stanstock.data.research_product import (
    CapturedProductIntake,
    product_catalog_references,
    product_intake_payload,
    verify_product_listing_catalog,
    verify_product_price_content,
)
from stanstock.research.decision_support_eligibility import (
    LIQUIDITY_PURPOSE,
    MAXIMUM_VOLUME,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_VERSION,
    SOURCE_CONTRACT,
    WITNESS_CONTRACT,
    LiquidityAssessment,
    LiquidityObservation,
    assess_liquidity,
    assess_mode_witness,
    canonical_json,
    load_decision_support_config,
    strict_json,
)

RAW_KIND = "liquidity_price_raw"
HISTORY_KIND = "liquidity_price_history"
BATCH_KIND = "liquidity_run_evidence"
RAW_SCHEMA = "td-liquidity-raw@1"
HISTORY_SCHEMA = "liquidity-evidence@1"
BATCH_SCHEMA = "liquidity-run-evidence@1"
WINDOW_SCHEMA = "liquidity-window-proof@1"


@dataclass(frozen=True, slots=True)
class VerifiedLiquidityBatch:
    asset: DataAsset
    document: dict[str, Any]
    assessments: Mapping[str, LiquidityAssessment]
    sources: tuple[DataAsset, ...]


def required_liquidity_sessions(target_date: date) -> tuple[date, ...]:
    calendar = get_calendar("XNYS", start=target_date - timedelta(days=70), end=target_date)
    sessions = tuple(
        day.date()
        for day in calendar.sessions_in_range(target_date - timedelta(days=65), target_date)
    )[-20:]
    if len(sessions) != 20 or sessions[-1] != target_date:
        raise ValueError("Liquidity target must be a completed XNYS session")
    return sessions


def require_liquidity_authorization(captured: Mapping[str, Any]) -> None:
    owner = captured.get("owner_id")
    if (
        not isinstance(owner, str)
        or not get_user_model().objects.filter(pk=owner, is_active=True).exists()
    ):
        raise ValueError("Liquidity evidence requires its current active owner")
    record = ProviderRecord.objects.filter(provider=TWELVE_DATA_PROVIDER).first()
    if record is None:
        raise ValueError("Liquidity evidence has no current provider authorization")
    validate_provider_usage(record, owner_id=owner)
    if (
        captured.get("product_version") != PRODUCT_VERSION
        or captured.get("source_provider") != TWELVE_DATA_PROVIDER
        or captured.get("policy_identity") != PRODUCT_EFFECTIVE_CONFIG_HASH
    ):
        raise ValueError("Liquidity intake has the wrong source/policy identity")


def liquidity_job_identity(captured: Mapping[str, Any]) -> dict[str, str]:
    return {
        "owner_id": str(captured["owner_id"]),
        "issuance_key": str(captured["issuance_key"]),
        "product": PRODUCT_VERSION,
        "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
        "source_contract": SOURCE_CONTRACT,
    }


def _require_job(job: JobRun, intake: CapturedProductIntake, captured: Mapping[str, Any]) -> None:
    from stanstock.data.research_product_jobs import product_job_name

    expected = product_job_name("research_intake_v2", liquidity_job_identity(captured))
    if (
        job.job_name != expected
        or job.region != "us"
        or job.target_date.isoformat() != captured["target_date"]
        or job.started_at < intake.asset.retrieved_at
        or job.details.get("intake_asset") != asset_ref_for(intake.asset).to_json()
    ):
        raise ValueError("Liquidity batch does not bind its original native acquisition attempt")


def _split_context(
    intake: CapturedProductIntake, job: JobRun, *, start: date, target: date
) -> dict[str, Any]:
    return {
        "intake": asset_ref_for(intake.asset).to_json(),
        "job_id": str(job.id),
        "job_name": job.job_name,
        "region": job.region,
        "attempt": job.attempt,
        "product_version": PRODUCT_VERSION,
        "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
        "source_contract": SOURCE_CONTRACT,
        "requested_adjustment": "splits",
        "request_start": start.isoformat(),
        "request_end": target.isoformat(),
    }


def _none_frame(series: PriceSeries) -> pl.DataFrame:
    # Exact decimal strings are intentional: this is evidence, not a Float64
    # forecast or valuation frame. No parity decision reads normalized floats.
    return pl.DataFrame(
        {
            "date": [bar.trade_date for bar in series.bars],
            **{
                field: [
                    None if (value := getattr(bar, field)) is None else str(value)
                    for bar in series.bars
                ]
                for field in ("open", "high", "low", "close")
            },
            "volume": [bar.volume for bar in series.bars],
        },
        schema_overrides={
            "date": pl.Date,
            "open": pl.String,
            "high": pl.String,
            "low": pl.String,
            "close": pl.String,
            "volume": pl.Int64,
        },
    ).sort("date")


def _parsed_series(
    raw: bytes, *, symbol: str, retrieved_at: datetime, start: date, target: date, mode: str
) -> PriceSeries:
    strict_json(raw)
    series = twelve_data.parse_daily_price_series(
        raw,
        symbol=symbol,
        retrieved_at=retrieved_at,
        source_url="https://api.twelvedata.com/time_series",
        start_date=start,
        end_date=target,
        adjustment=mode,
    )
    if any(
        bar.volume is not None
        and (type(bar.volume) is not int or not 0 <= bar.volume <= MAXIMUM_VOLUME)
        for bar in series.bars
    ):
        raise ValueError("Liquidity volume is outside the exact integer contract")
    return series


def _require_none_listing_identity(series: PriceSeries, listing: Listing) -> None:
    types = {
        "Common Stock": Security.SecurityType.COMMON_STOCK,
        "ADR": Security.SecurityType.ADR,
        "American Depositary Receipt": Security.SecurityType.ADR,
        "Depositary Receipt": Security.SecurityType.ADR,
    }
    if (
        series.symbol != listing.provider_symbol
        or listing.region != "us"
        or series.currency != listing.currency
        or listing.currency != "USD"
        or series.mic_code != listing.exchange_mic
        or series.exchange != ("NYSE" if listing.exchange_mic == "XNYS" else "NASDAQ")
        or types.get(series.instrument_type or "") != listing.security.security_type
    ):
        raise ValueError("None evidence does not bind the permanent US/USD stock-or-ADR identity")


def persist_none_series(
    *,
    series: PriceSeries,
    listing: Listing,
    intake: CapturedProductIntake,
    job: JobRun,
    request_start: date,
    target_date: date,
    store: AssetStore,
) -> DataAsset:
    """Publish raw plus normalized none evidence; never update a market row."""
    captured = product_intake_payload(intake, store=store)
    require_liquidity_authorization(captured)
    _require_job(job, intake, captured)
    parsed = _parsed_series(
        series.raw_bytes,
        symbol=listing.provider_symbol,
        retrieved_at=series.retrieved_at,
        start=request_start,
        target=target_date,
        mode="none",
    )
    _require_none_listing_identity(parsed, listing)
    if (
        series.provider != TWELVE_DATA_PROVIDER
        or series.adjustment != "none"
        or parsed.bars != series.bars
        or series.symbol != listing.provider_symbol
        or parsed.currency != listing.currency
        or parsed.mic_code != listing.exchange_mic
        or series.retrieved_at < job.started_at
        or series.retrieved_at > timezone.now()
    ):
        raise ValueError("None response does not bind the fresh request/listing")
    common: dict[str, Any] = {
        "purpose": LIQUIDITY_PURPOSE,
        "source_contract": SOURCE_CONTRACT,
        "product_version": PRODUCT_VERSION,
        "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
        "intake": asset_ref_for(intake.asset).to_json(),
        "acquisition_job_id": str(job.id),
        "listing_id": str(listing.id),
        "security_id": str(listing.security_id),
        "requested_adjustment": "none",  # request receipt, never a response echo
        "request_start": request_start.isoformat(),
        "request_end": target_date.isoformat(),
        "currency": parsed.currency,
        "exchange": parsed.exchange,
        "mic_code": parsed.mic_code,
        "instrument_type": parsed.instrument_type,
    }
    raw_digest = hashlib.sha256(series.raw_bytes).hexdigest()
    prefix = f"research/liquidity/{intake.asset.id}/{job.id}/{listing.id}/{raw_digest}"
    raw_stored = store.write_bytes(f"{prefix}.json", series.raw_bytes)
    written = [raw_stored]
    committed = False
    try:
        frame_stored = store.write_frame(f"{prefix}.parquet", _none_frame(parsed))
        written.append(frame_stored)
        with transaction.atomic():
            raw = DataAsset.objects.create(
                provider=TWELVE_DATA_PROVIDER,
                kind=RAW_KIND,
                subject=listing.provider_symbol,
                relative_path=raw_stored.relative_path,
                sha256=raw_stored.sha256,
                retrieved_at=series.retrieved_at,
                available_at=series.retrieved_at,
                period_start=parsed.bars[0].trade_date,
                period_end=parsed.bars[-1].trade_date,
                schema_version=RAW_SCHEMA,
                metadata=common,
            )
            published_at = timezone.now()
            asset = DataAsset.objects.create(
                provider=TWELVE_DATA_PROVIDER,
                kind=HISTORY_KIND,
                subject=listing.provider_symbol,
                relative_path=frame_stored.relative_path,
                sha256=frame_stored.sha256,
                retrieved_at=published_at,
                available_at=published_at,
                period_start=raw.period_start,
                period_end=raw.period_end,
                schema_version=HISTORY_SCHEMA,
                metadata={
                    **common,
                    "raw_asset": asset_ref_for(raw).to_json(),
                    "source_retrieved_at": series.retrieved_at.isoformat(),
                    "rows": len(parsed.bars),
                },
            )
        committed = True
        return asset
    finally:
        if not committed:
            for stored in written:
                if not DataAsset.objects.filter(relative_path=stored.relative_path).exists():
                    store.resolve(stored.relative_path).unlink(missing_ok=True)


def _observations(
    series: PriceSeries, sessions: tuple[date, ...]
) -> tuple[LiquidityObservation, ...]:
    return tuple(
        LiquidityObservation(bar.trade_date, bar.close, bar.volume)
        for bar in series.bars
        if bar.trade_date in sessions
    )


def _values(values: tuple[LiquidityObservation, ...] | None) -> list[dict[str, Any]] | None:
    return (
        None
        if values is None
        else [
            {"date": row.session.isoformat(), "close": str(row.close), "volume": row.volume}
            for row in values
        ]
    )


def _assessment(value: LiquidityAssessment) -> dict[str, Any]:
    return {
        **asdict(value),
        "assessed_dates": [day.isoformat() for day in value.assessed_dates],
        "reasons": list(value.reasons),
    }


def _read_none(
    *,
    asset: DataAsset,
    listing: Listing,
    intake: CapturedProductIntake,
    job: JobRun,
    start: date,
    target: date,
    publication: datetime,
    store: AssetStore,
) -> tuple[PriceSeries, DataAsset]:
    if (
        asset.kind != HISTORY_KIND
        or asset.schema_version != HISTORY_SCHEMA
        or asset.provider != TWELVE_DATA_PROVIDER
        or asset.subject != listing.provider_symbol
        or not isinstance(asset.metadata, dict)
    ):
        raise ValueError("Liquidity normalized source has the wrong purpose/identity")
    raw = resolve_asset_ref(AssetRef.from_json(asset.metadata.get("raw_asset")), cutoff=publication)
    series = _parsed_series(
        read_checksummed_bytes(store, raw),
        symbol=listing.provider_symbol,
        retrieved_at=raw.retrieved_at,
        start=start,
        target=target,
        mode="none",
    )
    _require_none_listing_identity(series, listing)
    common = {
        "purpose": LIQUIDITY_PURPOSE,
        "source_contract": SOURCE_CONTRACT,
        "product_version": PRODUCT_VERSION,
        "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
        "intake": asset_ref_for(intake.asset).to_json(),
        "acquisition_job_id": str(job.id),
        "listing_id": str(listing.id),
        "security_id": str(listing.security_id),
        "requested_adjustment": "none",
        "request_start": start.isoformat(),
        "request_end": target.isoformat(),
        "currency": series.currency,
        "exchange": series.exchange,
        "mic_code": series.mic_code,
        "instrument_type": series.instrument_type,
    }
    if (
        raw.kind != RAW_KIND
        or raw.schema_version != RAW_SCHEMA
        or raw.provider != asset.provider
        or raw.subject != asset.subject
        or raw.metadata != common
        or asset.metadata
        != {
            **common,
            "raw_asset": asset_ref_for(raw).to_json(),
            "source_retrieved_at": raw.retrieved_at.isoformat(),
            "rows": len(series.bars),
        }
        or raw.retrieved_at < job.started_at
        or raw.available_at != raw.retrieved_at
        or asset.retrieved_at < raw.retrieved_at
        or asset.available_at != asset.retrieved_at
        or series.currency != listing.currency
        or series.mic_code != listing.exchange_mic
        or raw.period_start != series.bars[0].trade_date
        or raw.period_end != series.bars[-1].trade_date
        or asset.period_start != raw.period_start
        or asset.period_end != raw.period_end
    ):
        raise ValueError("Liquidity raw/normalized identity or fresh chronology diverged")
    try:
        actual = pl.read_parquet(io.BytesIO(read_checksummed_bytes(store, asset)))
    except pl.exceptions.PolarsError:
        raise ValueError("Liquidity normalized bytes are malformed") from None
    expected = _none_frame(series)
    if actual.schema != expected.schema or not actual.equals(expected):
        raise ValueError("Liquidity normalized values diverge from full public raw OHLC admission")
    return series, raw


def _subject(intake: CapturedProductIntake) -> str:
    return f"{PRODUCT_VERSION}:liquidity:{intake.asset.id}"


def _batch_document(
    *,
    intake: CapturedProductIntake,
    job: JobRun,
    captured: Mapping[str, Any],
    entries: list[dict[str, Any]],
    catalogs: tuple[DataAsset, ...],
    benchmark: DataAsset,
    cutoff: datetime,
    generated_at: datetime,
    store: AssetStore,
) -> tuple[dict[str, Any], dict[str, LiquidityAssessment], tuple[DataAsset, ...]]:
    config = load_decision_support_config()
    target = date.fromisoformat(str(captured["target_date"]))
    sessions = required_liquidity_sessions(target)
    controls = config.witness_pre_sessions + config.witness_post_sessions
    references = product_catalog_references(
        catalog_assets=catalogs,
        symbols=intake.requested_symbols,
        cutoff=cutoff,
        store=store,
    )
    references_by_symbol: dict[str, list[StockReference]] = {}
    for reference in references:
        references_by_symbol.setdefault(reference.symbol, []).append(reference)
    if (
        [entry.get("symbol") for entry in entries] != list(intake.requested_symbols)
        or generated_at < cutoff
        or generated_at > timezone.now()
        or cutoff.tzinfo is None
        or generated_at.tzinfo is None
        or job.started_at > cutoff
    ):
        raise ValueError("Liquidity batch scope or chronology is invalid")
    receipts = job.details.get("liquidity_requests")
    if not isinstance(receipts, list):
        raise ValueError("Liquidity batch has no native producing request receipts")
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    history_start = target - timedelta(days=366 * captured["core_config"]["history_years"])
    request_order = tuple(dict.fromkeys((config.witness_symbol, *intake.requested_symbols)))
    last_position = -1
    for receipt in receipts:
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"symbol", "mode", "start", "end", "asset"}
            or receipt["symbol"] not in intake.requested_symbols
            or receipt["mode"] not in ("none", "splits")
            or receipt["end"] != target.isoformat()
            or (receipt["symbol"], receipt["mode"]) in keyed
        ):
            raise ValueError("Liquidity native request identity is invalid or duplicated")
        symbol, mode = receipt["symbol"], receipt["mode"]
        position = request_order.index(symbol)
        expected_start = (
            history_start
            if mode == "splits"
            else (config.witness_request_start if symbol == config.witness_symbol else sessions[0])
        )
        if position < last_position or receipt["start"] != expected_start.isoformat():
            raise ValueError("Liquidity request order or independently captured scope diverged")
        if mode == "none" and (
            (symbol, "splits") not in keyed or keyed[(symbol, "splits")]["asset"] is None
        ):
            raise ValueError("A liquidity none request requires its preceding fresh split")
        last_position = position
        keyed[(receipt["symbol"], receipt["mode"])] = receipt
    if len(receipts) > 2 * len(intake.requested_symbols):
        raise ValueError("Liquidity batch exceeds its bounded request plan")
    sources: dict[UUID, DataAsset] = {}
    parsed: dict[str, tuple[PriceSeries | None, PriceSeries | None]] = {}
    resolved: dict[str, tuple[Listing | None, DataAsset | None, DataAsset | None]] = {}
    raw_sources: dict[str, dict[str, DataAsset]] = {}
    for entry in entries:
        if set(entry) != {
            "symbol",
            "listing_id",
            "split_asset",
            "none_asset",
            "unavailable_reasons",
        }:
            raise ValueError("Liquidity batch source entry shape is invalid")
        symbol = entry["symbol"]
        raw_sources[symbol] = {}
        listing = None
        if entry["listing_id"] is not None:
            listing = Listing.objects.select_related("security").get(pk=entry["listing_id"])
            matches = references_by_symbol.get(symbol, [])
            if (
                listing.provider_symbol != symbol
                or len(matches) != 1
                or matches[0].currency != "USD"
                or listing.currency != "USD"
                or matches[0].mic_code != listing.exchange_mic
                or matches[0].exchange not in ("NASDAQ", "NYSE")
            ):
                raise ValueError("Liquidity listing has no independent exact catalog identity")
            _validate_my_list_reference(matches[0])
            if not provider_plan_allows(captured["entitlement"]["plan"], matches[0].access_plan):
                raise ValueError("Captured rights do not authorize this liquidity listing")
        split = none = None
        split_series = none_series = None
        for mode, key in (("splits", "split_asset"), ("none", "none_asset")):
            raw_ref = entry[key]
            receipt = keyed.get((symbol, mode))
            if raw_ref is None:
                if receipt is not None and receipt["asset"] is not None:
                    raise ValueError("Liquidity batch omitted a native acquired source")
                continue
            if listing is None or receipt is None or receipt["asset"] != raw_ref:
                raise ValueError("Liquidity source is not owned by this native request attempt")
            asset = resolve_asset_ref(AssetRef.from_json(raw_ref), cutoff=generated_at)
            if asset.retrieved_at < job.started_at:
                raise ValueError("An older split/none asset cannot supply fresh batch evidence")
            sources[asset.id] = asset
            if mode == "none":
                start = (
                    config.witness_request_start if symbol == config.witness_symbol else sessions[0]
                )
                if receipt["start"] != start.isoformat():
                    raise ValueError("Liquidity none request scope diverged")
                none_series, raw = _read_none(
                    asset=asset,
                    listing=listing,
                    intake=intake,
                    job=job,
                    start=start,
                    target=target,
                    publication=generated_at,
                    store=store,
                )
                if (
                    none_series.exchange != matches[0].exchange
                    or none_series.instrument_type != matches[0].instrument_type
                ):
                    raise ValueError(
                        "None response exchange/type differs from the independent catalog"
                    )
                none = asset
            else:
                if asset.kind != "price_history" or asset.metadata.get("adjustment") != "splits":
                    raise ValueError("Liquidity split source is not a generic split-only vintage")
                raw = raw_price_asset_for(asset, cutoff=generated_at)
                if raw is None or raw.retrieved_at < job.started_at:
                    raise ValueError("Liquidity split source has no fresh raw authority")
                context = _split_context(intake, job, start=history_start, target=target)
                if (
                    asset.metadata.get("acquisition_context") != context
                    or raw.metadata.get("acquisition_context") != context
                ):
                    raise ValueError(
                        "Liquidity split source belongs to another native acquisition attempt"
                    )
                verify_product_price_content(
                    asset=asset,
                    raw=raw,
                    cutoff=generated_at,
                    target_date=target,
                    store=store,
                )
                verify_product_listing_catalog(
                    listing=listing,
                    stock_asset=asset,
                    benchmark_asset=benchmark,
                    catalog_assets=catalogs,
                    cutoff=generated_at,
                    store=store,
                )
                split_series = _parsed_series(
                    read_checksummed_bytes(store, raw),
                    symbol=symbol,
                    retrieved_at=raw.retrieved_at,
                    start=date.fromisoformat(receipt["start"]),
                    target=target,
                    mode="splits",
                )
                normalized = (
                    AsOfData(generated_at, store)
                    .price_frame_for_asset_with_diagnostics(
                        asset=asset,
                        through_date=target,
                    )
                    .frame
                )
                expected = _price_frame(split_series)
                if normalized.select(
                    expected.columns
                ).schema != expected.schema or not normalized.select(expected.columns).equals(
                    expected
                ):
                    raise ValueError("Split normalized values diverge from full public raw OHLC")
                split = asset
            sources[raw.id] = raw
            raw_sources[symbol][mode] = raw
        parsed[symbol] = (split_series, none_series)
        resolved[symbol] = (listing, split, none)
    split_control, none_control = parsed.get(config.witness_symbol, (None, None))
    witness_status: Literal["matched", "unavailable", "failed"] = assess_mode_witness(
        none=None if none_control is None else _observations(none_control, controls),
        splits=None if split_control is None else _observations(split_control, controls),
        config=config,
    )
    control_listing, control_split, control_none = resolved.get(
        config.witness_symbol, (None, None, None)
    )
    if control_listing is not None and (
        control_listing.provider_symbol != "NVDA"
        or control_listing.exchange_mic not in {"XNAS", "XNGS", "XNCM", "XNMS"}
        or (split_control is not None and split_control.exchange != "NASDAQ")
    ):
        raise ValueError("Liquidity canary does not bind the fixed NVDA NASDAQ identity")
    operational_witness_status = witness_status
    if any(
        source.available_at > cutoff or source.retrieved_at > cutoff
        for source in (
            control_split,
            control_none,
            *raw_sources.get(config.witness_symbol, {}).values(),
        )
        if source is not None
    ):
        witness_status = "unavailable"
    non_control_receipts = [
        receipt for receipt in receipts if receipt["symbol"] != config.witness_symbol
    ]
    if non_control_receipts and (
        operational_witness_status != "matched"
        or not receipts
        or receipts[0]["symbol"] != config.witness_symbol
    ):
        raise ValueError("Extra C2 requests cannot follow an unavailable or failed fixed control")
    assessments: dict[str, LiquidityAssessment] = {}
    proofs: list[dict[str, Any]] = []
    for entry in entries:
        symbol = entry["symbol"]
        listing, split, none = resolved[symbol]
        split_series, none_series = parsed[symbol]
        reasons = tuple(entry["unavailable_reasons"])
        unsafe = any(
            source.available_at > cutoff or source.retrieved_at > cutoff
            for source in (split, none, *raw_sources[symbol].values())
            if source is not None
        )
        if unsafe:
            reasons += ("liquidity_cutoff_unsafe",)
        split_values = (
            None if split_series is None or unsafe else _observations(split_series, sessions)
        )
        none_values = (
            None if none_series is None or unsafe else _observations(none_series, sessions)
        )
        assessment = assess_liquidity(
            required_sessions=sessions,
            none=none_values,
            splits=split_values,
            witness_status=witness_status,
            unavailable_reasons=reasons,
        )
        assessments[symbol] = assessment
        proofs.append(
            {
                **entry,
                "contract": WINDOW_SCHEMA,
                "split_raw_asset": asset_ref_for(raw_sources[symbol]["splits"]).to_json()
                if "splits" in raw_sources[symbol]
                else None,
                "none_raw_asset": asset_ref_for(raw_sources[symbol]["none"]).to_json()
                if "none" in raw_sources[symbol]
                else None,
                "source_clocks": {
                    str(source.id): {
                        "retrieved_at": source.retrieved_at.isoformat(),
                        "available_at": source.available_at.isoformat(),
                    }
                    for source in (split, none, *raw_sources[symbol].values())
                    if source is not None
                },
                "security_id": None if listing is None else str(listing.security_id),
                "dates": [day.isoformat() for day in sessions],
                "split_values": _values(split_values),
                "none_values": _values(none_values),
                "assessment": _assessment(assessment),
            }
        )
    admissions = job.details.get("liquidity_admissions")
    if not isinstance(admissions, dict) or set(admissions) != set(intake.requested_symbols):
        raise ValueError("Closed liquidity evidence requires every captured forecast admission")
    for symbol, admission in admissions.items():
        if (
            not isinstance(admission, dict)
            or not isinstance(admission.get("status"), str)
            or not isinstance(admission.get("reason"), str)
        ):
            raise ValueError("Closed batch forecast admission is malformed")
        if admission.get("price_asset") is not None:
            asset = resolve_asset_ref(AssetRef.from_json(admission["price_asset"]), cutoff=cutoff)
            if (
                asset.kind != "price_history"
                or asset.subject != symbol
                or asset.metadata.get("adjustment") != "splits"
            ):
                raise ValueError("Closed batch forecast admission is not split-only evidence")
            raw = raw_price_asset_for(asset, cutoff=cutoff)
            if raw is None:
                raise ValueError("Closed batch forecast admission lacks raw physical authority")
            verify_product_price_content(
                asset=asset, raw=raw, cutoff=cutoff, target_date=target, store=store
            )
            sources[asset.id], sources[raw.id] = asset, raw
        elif admission["status"] == "admitted":
            raise ValueError("An admitted forecast candidate has no physical price source")
    if not isinstance(job.details.get("forward_declaration"), dict):
        raise ValueError("Closed liquidity evidence requires the separately declared protocol")
    document = {
        "contract": BATCH_SCHEMA,
        "product_version": PRODUCT_VERSION,
        "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
        "source_contract": SOURCE_CONTRACT,
        "purpose": LIQUIDITY_PURPOSE,
        "calendar": "XNYS",
        "currency": "USD",
        "intake": asset_ref_for(intake.asset).to_json(),
        "owner_id": captured["owner_id"],
        "batch_id": f"{intake.asset.id}:{job.id}",
        "acquisition_job_id": str(job.id),
        "job_name": job.job_name,
        "attempt": job.attempt,
        "target_date": target.isoformat(),
        "logical_cutoff": cutoff.isoformat(),
        "generated_at": generated_at.isoformat(),
        "catalog_assets": [asset_ref_for(asset).to_json() for asset in catalogs],
        "benchmark_asset": asset_ref_for(benchmark).to_json(),
        "request_receipts": receipts,
        "forecast_admissions": admissions,
        "forward_declaration": job.details.get("forward_declaration"),
        "witness": {
            "contract": WITNESS_CONTRACT,
            "symbol": "NVDA",
            "listing_id": None if control_listing is None else str(control_listing.id),
            "status": witness_status,
            "dates": [day.isoformat() for day in controls],
            "none_values": None
            if none_control is None or witness_status == "unavailable"
            else _values(_observations(none_control, controls)),
            "split_values": None
            if split_control is None or witness_status == "unavailable"
            else _values(_observations(split_control, controls)),
        },
        "entries": proofs,
    }
    return document, assessments, tuple(sources.values())


def close_liquidity_batch(
    *,
    intake: CapturedProductIntake,
    job: JobRun,
    entries: list[dict[str, Any]],
    catalog_assets: tuple[DataAsset, ...],
    benchmark_asset: DataAsset,
    cutoff: datetime,
    store: AssetStore,
) -> DataAsset:
    captured = product_intake_payload(intake, store=store)
    require_liquidity_authorization(captured)
    _require_job(job, intake, captured)
    if DataAsset.objects.filter(
        provider="stanstock", kind=BATCH_KIND, subject=_subject(intake)
    ).exists():
        raise ValueError("A closed liquidity batch already claims this captured intake")
    generated_at = timezone.now()
    document, _assessments, _sources = _batch_document(
        intake=intake,
        job=job,
        captured=captured,
        entries=entries,
        catalogs=catalog_assets,
        benchmark=benchmark_asset,
        cutoff=cutoff,
        generated_at=generated_at,
        store=store,
    )
    payload = canonical_json(document)
    stored = store.write_bytes(
        f"research/liquidity/{intake.asset.id}/{job.id}/closed-{hashlib.sha256(payload).hexdigest()}.json",
        payload,
    )
    committed = False
    try:
        asset = DataAsset.objects.create(
            provider="stanstock",
            kind=BATCH_KIND,
            subject=_subject(intake),
            relative_path=stored.relative_path,
            sha256=stored.sha256,
            retrieved_at=generated_at,
            available_at=generated_at,
            schema_version=BATCH_SCHEMA,
            metadata={
                "contract": BATCH_SCHEMA,
                "intake": asset_ref_for(intake.asset).to_json(),
                "acquisition_job_id": str(job.id),
                "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
                "purpose": LIQUIDITY_PURPOSE,
            },
        )
        committed = True
        return asset
    finally:
        if (
            not committed
            and not DataAsset.objects.filter(relative_path=stored.relative_path).exists()
        ):
            store.resolve(stored.relative_path).unlink(missing_ok=True)


def verify_liquidity_batch(
    *,
    asset: DataAsset,
    intake: CapturedProductIntake,
    cutoff: datetime,
    store: AssetStore,
) -> VerifiedLiquidityBatch:
    """Persisted malformed evidence raises; it never becomes a HOLD fallback."""
    captured = product_intake_payload(intake, store=store)
    require_liquidity_authorization(captured)
    try:
        if list(
            DataAsset.objects.filter(
                provider="stanstock", kind=BATCH_KIND, subject=_subject(intake)
            ).values_list("id", flat=True)
        ) != [asset.id]:
            raise ValueError("Closed liquidity evidence is missing or ambiguous")
        document = strict_json(read_checksummed_bytes(store, asset))
        if not isinstance(document, dict):
            raise ValueError
        generated_at = datetime.fromisoformat(document["generated_at"])
        logical_cutoff = datetime.fromisoformat(document["logical_cutoff"])
        job = JobRun.objects.get(id=document["acquisition_job_id"])
        _require_job(job, intake, captured)
        if (
            asset.provider != "stanstock"
            or asset.kind != BATCH_KIND
            or asset.schema_version != BATCH_SCHEMA
            or asset.subject != _subject(intake)
            or asset.available_at != generated_at
            or asset.retrieved_at != generated_at
            or generated_at > cutoff
            or logical_cutoff > cutoff
            or asset.metadata
            != {
                "contract": BATCH_SCHEMA,
                "intake": asset_ref_for(intake.asset).to_json(),
                "acquisition_job_id": str(job.id),
                "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
                "purpose": LIQUIDITY_PURPOSE,
            }
        ):
            raise ValueError
        catalogs = tuple(
            resolve_asset_ref(AssetRef.from_json(ref), cutoff=logical_cutoff)
            for ref in document["catalog_assets"]
        )
        benchmark = resolve_asset_ref(
            AssetRef.from_json(document["benchmark_asset"]), cutoff=logical_cutoff
        )
        entries = [
            {
                key: entry[key]
                for key in (
                    "symbol",
                    "listing_id",
                    "split_asset",
                    "none_asset",
                    "unavailable_reasons",
                )
            }
            for entry in document["entries"]
        ]
        expected, assessments, sources = _batch_document(
            intake=intake,
            job=job,
            captured=captured,
            entries=entries,
            catalogs=catalogs,
            benchmark=benchmark,
            cutoff=logical_cutoff,
            generated_at=generated_at,
            store=store,
        )
        if canonical_json(document) != canonical_json(expected):
            raise ValueError
    except (
        KeyError,
        TypeError,
        ValueError,
        ProviderError,
        pl.exceptions.PolarsError,
        JobRun.DoesNotExist,
        Listing.DoesNotExist,
    ) as exc:
        raise RefreshVerificationError(
            "liquidity_batch_integrity_failed",
            "The closed liquidity batch failed independent replay",
        ) from exc
    return VerifiedLiquidityBatch(asset, document, assessments, sources)


def recover_liquidity_batch(
    *,
    intake: CapturedProductIntake,
    cutoff: datetime,
    store: AssetStore,
) -> VerifiedLiquidityBatch | None:
    assets = list(
        DataAsset.objects.filter(provider="stanstock", kind=BATCH_KIND, subject=_subject(intake))
    )
    if len(assets) > 1:
        raise ValueError("Conflicting closed liquidity batches claim the same captured intake")
    return (
        verify_liquidity_batch(asset=assets[0], intake=intake, cutoff=cutoff, store=store)
        if assets
        else None
    )


def acquire_liquidity_pairs(
    *,
    intake: CapturedProductIntake,
    job: JobRun,
    listings: Mapping[str, Listing],
    references: Mapping[str, StockReference],
    split_assets: dict[str, DataAsset],
    request_key: Callable[[], str],
    preflight: Callable[[int], None],
    history_start: date,
    target_date: date,
    store: AssetStore,
) -> list[dict[str, Any]]:
    """One bounded C2 request plan, run under the caller's OLD acquisition lock.

    Called before the ordinary missing-split loop, which skips this attempt's
    successful split fetches. Canary failure stops only additional C2 requests.
    """
    config = load_decision_support_config()
    sessions = required_liquidity_sessions(target_date)
    none_assets: dict[str, DataAsset] = {}
    fresh_splits: dict[str, DataAsset] = {}
    failures: dict[str, tuple[str, ...]] = {}
    receipts = job.details.setdefault("liquidity_requests", [])
    if receipts:
        raise ValueError("An unclosed attempt cannot reuse partial liquidity receipts")

    def acquire(symbol: str, mode: str) -> DataAsset:
        listing = listings[symbol]
        start = (
            history_start
            if mode == "splits"
            else (config.witness_request_start if symbol == config.witness_symbol else sessions[0])
        )
        key = request_key()
        receipt: dict[str, Any] = {
            "symbol": symbol,
            "mode": mode,
            "start": start.isoformat(),
            "end": target_date.isoformat(),
            "asset": None,
        }
        receipts.append(receipt)
        job.save(update_fields=["details"])
        series = twelve_data.fetch_daily_price_series(
            symbol,
            start_date=start,
            end_date=target_date,
            adjustment=mode,
            outputsize=5000,
            api_key=key,
        )
        if mode == "splits":
            _validate_my_list_price_series(
                series, reference=references[symbol], target_date=target_date
            )
            asset = _persist_price_series(
                store=store,
                series=series,
                listing=None,
                acquisition_context=_split_context(intake, job, start=start, target=target_date),
            )
        else:
            asset = persist_none_series(
                series=series,
                listing=listing,
                intake=intake,
                job=job,
                request_start=start,
                target_date=target_date,
                store=store,
            )
        receipt["asset"] = asset_ref_for(asset).to_json()
        job.save(update_fields=["details"])
        return asset

    enabled = True
    try:
        if listings:
            preflight(2 * len(listings))
    except ProviderQuotaError:
        enabled = False
        failures.update((symbol, ("liquidity_quota_unavailable",)) for symbol in listings)
    except ProviderConfigurationError:
        enabled = False
        failures.update((symbol, ("liquidity_provider_unavailable",)) for symbol in listings)
    order = tuple(dict.fromkeys((config.witness_symbol, *listings)))
    witness_status = "unavailable"
    for symbol in order:
        if symbol == config.witness_symbol and symbol not in listings:
            enabled = False
        if symbol not in listings or not enabled:
            continue
        try:
            fresh_splits[symbol] = acquire(symbol, "splits")
            split_assets[symbol] = fresh_splits[symbol]
            none_assets[symbol] = acquire(symbol, "none")
            if symbol == config.witness_symbol:
                raw = raw_price_asset_for(fresh_splits[symbol], cutoff=timezone.now())
                if raw is None:
                    raise ValueError("Fresh control split source has no raw response")
                split = _parsed_series(
                    read_checksummed_bytes(store, raw),
                    symbol=symbol,
                    retrieved_at=raw.retrieved_at,
                    start=history_start,
                    target=target_date,
                    mode="splits",
                )
                none, _raw = _read_none(
                    asset=none_assets[symbol],
                    listing=listings[symbol],
                    intake=intake,
                    job=job,
                    start=config.witness_request_start,
                    target=target_date,
                    publication=timezone.now(),
                    store=store,
                )
                controls = config.witness_pre_sessions + config.witness_post_sessions
                witness_status = assess_mode_witness(
                    none=_observations(none, controls),
                    splits=_observations(split, controls),
                    config=config,
                )
                if witness_status != "matched":
                    enabled = False
        except ProviderQuotaError:
            failures[symbol] = ("liquidity_quota_unavailable",)
            enabled = False
        except ProviderConfigurationError:
            failures[symbol] = ("liquidity_provider_unavailable",)
            enabled = False
        except ProviderError:
            failures[symbol] = ("liquidity_provider_payload_invalid",)
            if symbol == config.witness_symbol:
                enabled = False
        except RefreshVerificationError:
            raise
        except ValueError:
            failures[symbol] = ("liquidity_provider_payload_invalid",)
            if symbol == config.witness_symbol:
                enabled = False
        if symbol == config.witness_symbol and witness_status != "matched":
            enabled = False
    return [
        {
            "symbol": symbol,
            "listing_id": str(listings[symbol].id) if symbol in listings else None,
            "split_asset": asset_ref_for(fresh_splits[symbol]).to_json()
            if symbol in fresh_splits
            else None,
            "none_asset": asset_ref_for(none_assets[symbol]).to_json()
            if symbol in none_assets
            else None,
            "unavailable_reasons": list(failures.get(symbol, ())),
        }
        for symbol in intake.requested_symbols
    ]

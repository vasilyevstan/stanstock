"""Prospective paper v2: positive captures, independent terminal/path reduction.

This is not a personal portfolio, an order interface, or a historical backfill.
Only the explicit paper job adapter may capture a close. Readers re-prove the
registered bytes and source closure on every invocation and never write.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import (
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    Inexact,
    InvalidOperation,
    Overflow,
    Rounded,
    Underflow,
    localcontext,
)
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import numpy as np
import polars as pl
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.jobs import target_job_lock
from stanstock.core.models import JobRun
from stanstock.core.revision import _git, clean_git_revision
from stanstock.core.verification_types import AssetRef
from stanstock.data.asof import AsOfData, PriceFrameChecksumMismatchError, raw_price_asset_for
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    read_checksummed_bytes,
    register_asset,
    resolve_asset_ref,
)
from stanstock.data.etfs import is_supported_investable_etf
from stanstock.data.models import DataAsset, Listing, Security
from stanstock.data.provider_policy import TWELVE_DATA_PROVIDER, ProviderConfigurationError
from stanstock.data.providers.exceptions import ProviderError
from stanstock.data.research_product import verify_product_price_content
from stanstock.research.forward_scorecard import (
    PROTOCOL_EPOCH,
    _authority,
    read_forward_declaration,
)
from stanstock.research.price_product_config import (
    MOMENTUM_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PRODUCT_VERSION,
)
from stanstock.research.product_reader import ProductUser
from stanstock.simulation.engine import AccountingEngine
from stanstock.simulation.metrics import calculate_simulation_metrics
from stanstock.simulation.paper import (
    _activation_date,
    _bytes,
    _calendar,
    _date_json,
    _native_selection,
    _sessions,
    _timely,
)
from stanstock.simulation.types import (
    ACCOUNTING_VERSION,
    ACCOUNTING_VERSION_PARAMETER,
    MissingPricePolicy,
    RebalanceFrequency,
    SimulationConfig,
    SimulationMode,
)

PAPER_POLICY_VERSION = "current-method-paper-6m-v2"
CONTRACT = "paper-portfolio-v2@1"
ACTIVATION_KIND = "paper_portfolio_v2_activation"
EXECUTION_KIND = "paper_portfolio_v2_execution"
MEASUREMENT_KIND = "paper_portfolio_v2_measurement"
# Independent literal. In particular, do not extend/mutate paper.PAPER_POLICY.
PAPER_POLICY = {
    "version": "current-method-paper-6m-v2",
    "notional_usd": 100000,
    "spy_weight": 0.7,
    "equal_all_native_current_6m_buy_weight": 0.3,
    "empty_buy_sleeve": "cash",
    "under_10_weight": 0,
    "calculation_version": "simulation-accounting-v2",
    "product_config_hash": "55334183af29fc01b853e83f9bf75216f24564956925fb95912cb80a69420867",
    "execution": "first_XNYS_session_strictly_after_activation_date_at_close",
    "terminal_sessions_after_inception": 126,
    "transaction_cost_bps": 10,
    "slippage_bps": 5,
    "return_basis": "split_adjusted_price_only_no_dividends_or_cash_yield",
    "management": "frozen_quantities_no_rebalance_contributions_or_liquidation",
    "marks": "prospectively_captured_by_next_session_open_no_late_backfill",
    "baseline": "SPY_only_same_inception_calendar_notional_and_costs",
    "nonclaims": "notional_only_no_activation_of_personal_portfolio_or_skill_claim",
    "source_product_version": "research-product-v1",
    "source_method_version": "us-relative-momentum-v1",
    "source_provider": "twelve_data",
    "benchmark": "SPY",
    "terminal": (
        "timely_entry_and_terminal_verified_share_basis_no_intermediate_capture_requirement"
    ),
    "path_risk": "complete_prospective_common_path_or_withheld",
    "basis_reference_closes": 757,
    "basis": (
        "full_raw_normalized_identity_complete_entry_reference_and_held_calendar_"
        "all_recorded_overlap_consistent"
    ),
    "spy_reference": "full_notional_native_SPY_execution_exact_7_10_and_3_10_scaling",
    "attribution": (
        "stock_vs_same_dollar_SPY_whole_account_cash_entry_costs_and_"
        "explicit_native_roundoff_reconcile"
    ),
    "capture": "explicit_opt_in_existing_refresh_no_fetch_no_auto_activation",
}
PAPER_POLICY_SHA256 = "e1839e1bdd25236de47dcaa421def82fdfb98e0ac46b7d0f7261cf74fee22a61"
HEADER_FIELDS = frozenset(
    {"contract", "owner_id", "policy_version", "policy_sha256", "recorded_at", "execution_revision"}
)
SAFE_CODES = frozenset(
    {
        "paper_v2_integrity_invalid",
        "paper_v2_authorization_invalid",
        "paper_v2_not_activated",
        "paper_v2_selection_invalid",
        "paper_v2_source_unverified",
        "paper_v2_basis_incompatible",
        "paper_v2_deadline_missed",
        "paper_v2_native_insufficient",
        "paper_v2_outside_horizon",
    }
)
# Every admitted binary64 value has a finite, exact decimal expansion. 4096
# digits cover products, sums of <=121 positions, rational tenths, and /100000,
# even across the entire binary64 exponent range. Rounding is an error, not a
# terminal reconciliation plug. This is the sole decimal arithmetic context.
_DECIMAL_CONTEXT = Context(
    prec=4096,
    traps=[InvalidOperation, DivisionByZero, Overflow, Underflow, Inexact, Rounded],
)


class PaperV2Error(ValueError):
    def __init__(self, code: str = "paper_v2_integrity_invalid") -> None:
        self.reason_code = code if code in SAFE_CODES else "paper_v2_integrity_invalid"
        super().__init__(f"Paper v2 evidence withheld ({self.reason_code})")


@dataclass(frozen=True, slots=True)
class PaperArtifact:
    asset: DataAsset
    document: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PaperState:
    activation: PaperArtifact
    execution: PaperArtifact | None
    captures: tuple[PaperArtifact, ...]


def _require(condition: bool, code: str = "paper_v2_integrity_invalid") -> None:
    if not condition:
        raise PaperV2Error(code)


def _shape(value: Any, fields: set[str] | frozenset[str]) -> None:
    _require(type(value) is dict and set(value) == fields)


def _same(actual: Any, expected: Any) -> bool:
    """JSON equality that cannot mistake bool/int/float or extra fields."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _timestamp(value: Any) -> datetime:
    _require(type(value) is str)
    parsed = datetime.fromisoformat(value)
    _require(not timezone.is_naive(parsed) and parsed.isoformat() == value)
    return parsed


def _day(value: Any) -> date:
    _require(type(value) is str)
    parsed = date.fromisoformat(value)
    _require(parsed.isoformat() == value)
    return parsed


def _uuid(value: Any) -> str:
    _require(type(value) is str and str(UUID(value)) == value)
    return str(value)


def _ref(value: Any) -> AssetRef:
    parsed = AssetRef.from_json(value)
    _require(_same(value, parsed.to_json()))
    return parsed


def _native(value: Any, *, positive: bool = False) -> float:
    _require(type(value) in (float, int), "paper_v2_native_insufficient")
    try:
        native = float(value)
    except OverflowError:
        raise PaperV2Error("paper_v2_native_insufficient") from None
    _require(
        math.isfinite(native)
        and (not positive or native > 0)
        and (type(value) is not int or native == value),
        "paper_v2_native_insufficient",
    )
    return native


def _decimal(value: Any) -> Decimal:
    return Decimal.from_float(_native(value))


def _money(value: Decimal) -> str:
    _require(value.is_finite(), "paper_v2_native_insufficient")
    if not value:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _money_value(value: Any) -> Decimal:
    _require(type(value) is str)
    parsed = Decimal(value)
    _require(parsed.is_finite() and _money(parsed) == value)
    return parsed


def _owner(user: ProductUser, store: AssetStore) -> str:
    _require(
        len(PAPER_POLICY) == 28
        and hashlib.sha256(_bytes(PAPER_POLICY)).hexdigest() == PAPER_POLICY_SHA256
        and type(PAPER_POLICY["basis_reference_closes"]) is int
        and PRODUCT_VERSION == PAPER_POLICY["source_product_version"]
        and MOMENTUM_METHOD_VERSION == PAPER_POLICY["source_method_version"]
        and TWELVE_DATA_PROVIDER == PAPER_POLICY["source_provider"]
        and ACCOUNTING_VERSION == PAPER_POLICY["calculation_version"]
        and PRODUCT_EFFECTIVE_CONFIG_HASH == PAPER_POLICY["product_config_hash"]
    )
    try:
        owner, source_owner, provider = _authority(user, store)
    except ProviderConfigurationError:
        raise PaperV2Error("paper_v2_authorization_invalid") from None
    _require(
        owner == source_owner and provider == "twelve_data",
        "paper_v2_authorization_invalid",
    )
    return owner


def activation_subject(owner: str) -> str:
    return f"{PAPER_POLICY_VERSION}:{owner}"


def _one(kind: str, subject: str) -> DataAsset | None:
    # Reserved paper kinds are scoped by owner/activation identity, not by an
    # untrusted provider field. A wrong-provider row is malformed, not absent.
    rows = list(DataAsset.objects.filter(kind=kind, subject=subject)[:2])
    _require(len(rows) <= 1)
    return rows[0] if rows else None


def _header(owner: str, recorded: datetime, revision: str) -> dict[str, Any]:
    _require(
        not timezone.is_naive(recorded)
        and type(revision) is str
        and re.fullmatch(r"[0-9a-f]{40}", revision) is not None
    )
    return {
        "contract": CONTRACT,
        "owner_id": owner,
        "policy_version": PAPER_POLICY_VERSION,
        "policy_sha256": PAPER_POLICY_SHA256,
        "recorded_at": recorded.isoformat(),
        "execution_revision": revision,
    }


def _publish(
    *, kind: str, subject: str, document: dict[str, Any], store: AssetStore
) -> PaperArtifact:
    payload = _bytes(document)
    stored = store.write_bytes(
        f"simulations/paper-v2/{document['owner_id']}/{kind}/"
        f"{hashlib.sha256(payload).hexdigest()}.json",
        payload,
    )
    session = (
        _day(document["calendar"][0]) if kind == ACTIVATION_KIND else _day(document["session"])
    )
    end = _day(document["calendar"][-1]) if kind == ACTIVATION_KIND else session
    now = _timestamp(document["recorded_at"])
    asset = register_asset(
        provider="stanstock",
        kind=kind,
        subject=subject,
        stored=stored,
        retrieved_at=now,
        available_at=now,
        period_start=session,
        period_end=end,
        metadata={key: document[key] for key in HEADER_FIELDS},
    )
    return PaperArtifact(asset, document)


def _read_header(
    asset: DataAsset,
    *,
    kind: str,
    subject: str,
    owner: str,
    store: AssetStore,
    verified_revisions: set[str] | None = None,
) -> dict[str, Any]:
    payload = read_checksummed_bytes(store, asset)
    document = json.loads(payload)
    _require(type(document) is dict and payload == _bytes(document))
    recorded = _timestamp(document["recorded_at"])
    expected = _header(owner, recorded, document["execution_revision"])
    revision = document["execution_revision"]
    if verified_revisions is None or revision not in verified_revisions:
        _require(
            _git(Path(settings.BASE_DIR), "cat-file", "-t", revision).stdout.strip() == "commit"
        )
        if verified_revisions is not None:
            verified_revisions.add(revision)
    _require(
        asset.provider == "stanstock"
        and asset.kind == kind
        and asset.subject == subject
        and asset.schema_version == "1"
        and Path(asset.relative_path).name == f"{asset.sha256}.json"
        and recorded <= timezone.now()
        and asset.available_at == recorded
        and asset.retrieved_at == recorded
        and _same({key: document[key] for key in HEADER_FIELDS}, expected)
        and _same(asset.metadata, expected)
    )
    return cast(dict[str, Any], document)


def _validate_activation(
    asset: DataAsset,
    *,
    user: ProductUser,
    owner: str,
    store: AssetStore,
    verified_revisions: set[str] | None = None,
) -> PaperArtifact:
    registered = _one(ACTIVATION_KIND, activation_subject(owner))
    _require(registered is not None and registered.pk == asset.pk)
    doc = _read_header(
        asset,
        kind=ACTIVATION_KIND,
        subject=activation_subject(owner),
        owner=owner,
        store=store,
        verified_revisions=verified_revisions,
    )
    _shape(
        doc,
        HEADER_FIELDS
        | {"policy", "protocol", "calendar", "source_target_date", "evidence", "buys", "benchmark"},
    )
    recorded = _timestamp(doc["recorded_at"])
    declaration = read_forward_declaration(user=user, store=store)
    _ref(doc["protocol"])
    sessions = [day.isoformat() for day in _sessions(recorded)]
    _require(
        _same(doc["policy"], PAPER_POLICY)
        and _same(doc["protocol"], asset_ref_for(declaration.asset).to_json())
        and declaration.declared_at <= recorded
        and _same(doc["calendar"], sessions)
        and asset.period_start == _day(sessions[0])
        and asset.period_end == _day(sessions[-1])
    )
    target = _day(doc["source_target_date"])
    try:
        selection = _native_selection(
            user=user, store=store, target=target, require_current=False, activation_time=recorded
        )
    except ProviderError:
        raise PaperV2Error("paper_v2_selection_invalid") from None
    _require(
        all(_same(doc[name], value) for name, value in selection.items()),
        "paper_v2_selection_invalid",
    )
    return PaperArtifact(asset, doc)


def read_activation(
    *,
    user: ProductUser,
    store: AssetStore,
    asset: DataAsset | None = None,
    verified_revisions: set[str] | None = None,
) -> PaperArtifact:
    owner = _owner(user, store)
    selected = asset or _one(ACTIVATION_KIND, activation_subject(owner))
    _require(selected is not None, "paper_v2_not_activated")
    assert selected is not None
    return _validate_activation(
        selected, user=user, owner=owner, store=store, verified_revisions=verified_revisions
    )


def activate_current_method_paper(
    *, user: ProductUser, store: AssetStore | None = None
) -> PaperArtifact:
    store = store or AssetStore()
    owner = _owner(user, store)
    with (
        target_job_lock(
            job_name=activation_subject(owner), region="us", target_date=PROTOCOL_EPOCH
        ),
        transaction.atomic(durable=True),
    ):
        existing = _one(ACTIVATION_KIND, activation_subject(owner))
        if existing is not None:
            return _validate_activation(existing, user=user, owner=owner, store=store)
        declaration = read_forward_declaration(user=user, store=store)
        proof_started = timezone.now()
        try:
            selection = _native_selection(user=user, store=store, require_current=True)
        except ProviderError:
            raise PaperV2Error("paper_v2_selection_invalid") from None
        revision = clean_git_revision(Path(settings.BASE_DIR))
        recorded = timezone.now()  # after proof and the winning lock, never a preview clock
        _require(declaration.declared_at <= recorded and recorded >= proof_started)
        doc = {
            **_header(owner, recorded, revision),
            "policy": PAPER_POLICY,
            "protocol": asset_ref_for(declaration.asset).to_json(),
            "calendar": [day.isoformat() for day in _sessions(recorded)],
            **selection,
        }
        return _publish(
            kind=ACTIVATION_KIND, subject=activation_subject(owner), document=doc, store=store
        )


def capture_identity(activation: PaperArtifact) -> dict[str, Any]:
    """Full identity, independently re-derived after activation verification."""
    doc = activation.document
    return {
        "owner_id": doc["owner_id"],
        "policy_version": PAPER_POLICY_VERSION,
        "policy_sha256": PAPER_POLICY_SHA256,
        "activation": asset_ref_for(activation.asset).to_json(),
        "source_identity": {
            "product_version": PRODUCT_VERSION,
            "method_version": MOMENTUM_METHOD_VERSION,
            "provider": TWELVE_DATA_PROVIDER,
            "config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
            **{
                key: doc[key]
                for key in ("protocol", "source_target_date", "evidence", "buys", "benchmark")
            },
        },
    }


def capture_job_name(identity: dict[str, Any]) -> str:
    return "paper_v2:" + hashlib.sha256(_bytes(identity)).hexdigest()


def _references(activation: PaperArtifact) -> dict[str, dict[str, str]]:
    doc = activation.document
    selected = {row["listing_id"] for row in doc["buys"]}
    return dict(
        sorted(
            {
                **{
                    row["listing_id"]: row["source_assets"][0]
                    for row in doc["evidence"]
                    if row["listing_id"] in selected
                },
                doc["benchmark"]["listing_id"]: doc["benchmark"]["source_asset"],
            }.items()
        )
    )


def _proof_calendar(activation: PaperArtifact) -> Any:
    entry = _day(activation.document["calendar"][0])
    return get_calendar("XNYS", start=entry - timedelta(days=1400), end=entry + timedelta(days=740))


def _reference_dates(activation: PaperArtifact, calendar: Any = None) -> list[date]:
    entry = _day(activation.document["calendar"][0])
    calendar = calendar if calendar is not None else _proof_calendar(activation)
    return [day.date() for day in calendar.sessions_window(entry, -757)]


def _listing(listing_id: str, source: DataAsset, session: date) -> Listing:
    listing = Listing.objects.select_related("security__company").get(pk=_uuid(listing_id))
    metadata = source.metadata
    _require(
        listing.is_active
        and listing.is_primary
        and listing.currency == "USD"
        and listing.region == "us"
        and listing.provider_symbol == source.subject
        and (listing.valid_from is None or listing.valid_from <= session)
        and (listing.valid_to is None or session <= listing.valid_to)
        and metadata.get("currency") == "USD"
        and metadata.get("adjustment") == "splits"
        and metadata.get("return_definition") == "split_adjusted_price_return"
        and metadata.get("dividends_included") is False
        and (metadata.get("resolved_mic_code") or metadata.get("mic_code")) == listing.exchange_mic,
        "paper_v2_source_unverified",
    )
    if listing.provider_symbol == "SPY":
        _require(
            is_supported_investable_etf(listing) and metadata.get("instrument_type") == "ETF",
            "paper_v2_source_unverified",
        )
    else:
        _require(
            listing.security.security_type
            in {Security.SecurityType.COMMON_STOCK, Security.SecurityType.ADR}
            and metadata.get("instrument_type") in {"Common Stock", "Depositary Receipt"},
            "paper_v2_source_unverified",
        )
    return listing


def _history(
    asset: DataAsset,
    *,
    cutoff: datetime,
    session: date,
    store: AssetStore,
    calendar: Any = None,
) -> tuple[DataAsset, dict[date, float]]:
    raw = raw_price_asset_for(asset, cutoff=cutoff)
    _require(
        asset.provider == TWELVE_DATA_PROVIDER
        and asset.kind == "price_history"
        and asset.period_end == session
        and asset.period_start is not None
        and raw.period_start == asset.period_start
        and raw.period_end == session
        and raw.metadata.get("adjustment") == "splits"
        and asset.schema_version == raw.schema_version == "1",
        "paper_v2_source_unverified",
    )
    try:
        verify_product_price_content(
            asset=asset, raw=raw, cutoff=cutoff, target_date=session, store=store
        )
    except ProviderError:
        raise PaperV2Error("paper_v2_source_unverified") from None
    read = AsOfData(cutoff, store).price_frame_for_asset_with_diagnostics(
        asset=asset, through_date=session
    )
    frame = read.frame
    days = frame["date"].to_list()
    _require(
        read.invalid_session_date_rows == 0
        and days == sorted(set(days))
        and bool(days)
        and days[0] == asset.period_start
        and days[-1] == session,
        "paper_v2_source_unverified",
    )
    if (
        calendar is None
        or days[0] < calendar.first_session.date()
        or session > calendar.last_session.date()
    ):
        calendar = get_calendar(
            "XNYS", start=days[0] - timedelta(days=7), end=session + timedelta(days=7)
        )
    _require(
        all(calendar.is_session(day) for day in days),
        "paper_v2_source_unverified",
    )
    return raw, {
        day: _native(value, positive=True)
        for day, value in frame.select("date", "close").iter_rows()
    }


def _source_context(
    activation: PaperArtifact,
    store: AssetStore,
    calendar: Any = None,
) -> tuple[dict[str, DataAsset], dict[str, dict[date, float]]]:
    sources, histories = {}, {}
    recorded = _timestamp(activation.document["recorded_at"])
    target = _day(activation.document["source_target_date"])
    for listing_id, ref in _references(activation).items():
        source = resolve_asset_ref(_ref(ref), cutoff=recorded)
        _listing(listing_id, source, target)
        _raw, history = _history(
            source, cutoff=recorded, session=target, store=store, calendar=calendar
        )
        sources[listing_id], histories[listing_id] = source, history
    return sources, histories


def _mark(
    *,
    listing_id: str,
    source: DataAsset,
    activation: PaperArtifact,
    session: date,
    cutoff: datetime,
    store: AssetStore,
    known: dict[date, float],
    reference: list[float] | None,
    recorded_prices: dict[date, float],
    calendar: Any,
    exact: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[date, float]]:
    reference_dates = _reference_dates(activation, calendar)
    if exact is None:
        # Bounds only exclude known-short histories. They never prove coverage.
        # A corrupt/sparse NEWEST eligible asset cannot fall back to an older one.
        candidates = list(
            DataAsset.objects.filter(
                provider=source.provider,
                kind="price_history",
                subject=source.subject,
                available_at__lte=cutoff,
                retrieved_at__lte=cutoff,
                period_end=session,
                period_start__lte=reference_dates[0],
            ).order_by("-available_at", "-retrieved_at", "-id")[:2]
        )
        _require(bool(candidates), "paper_v2_source_unverified")
        asset = candidates[0]
        _require(
            len(candidates) < 2
            or (asset.available_at, asset.retrieved_at)
            != (candidates[1].available_at, candidates[1].retrieved_at),
            "paper_v2_source_unverified",
        )
    else:
        _shape(
            exact,
            {
                "listing_id",
                "session",
                "source_asset",
                "raw_asset",
                "native_price",
                "native_currency",
                "applied_fx_rate",
                "basis",
            },
        )
        asset = resolve_asset_ref(_ref(exact["source_asset"]), cutoff=cutoff)
    _require(
        asset.provider == source.provider
        and asset.subject == source.subject
        and all(
            _same(asset.metadata.get(key), source.metadata.get(key))
            for key in (
                "currency",
                "adjustment",
                "exchange",
                "mic_code",
                "resolved_mic_code",
                "instrument_type",
                "return_definition",
                "dividends_included",
            )
        ),
        "paper_v2_source_unverified",
    )
    _listing(listing_id, asset, session)
    raw, history = _history(asset, cutoff=cutoff, session=session, store=store, calendar=calendar)
    entry = _day(activation.document["calendar"][0])
    held_dates = [
        _day(day) for day in activation.document["calendar"] if day <= session.isoformat()
    ]
    _require(
        all(day in history for day in (*reference_dates, *held_dates))
        and all(history[day] == price for day, price in known.items() if day in history)
        and all(history.get(day) == price for day, price in recorded_prices.items()),
        "paper_v2_basis_incompatible",
    )
    closes = [history[day] for day in reference_dates]
    _require(
        reference is None or _same(closes, reference),
        "paper_v2_basis_incompatible",
    )
    _require(
        session != entry
        or listing_id == activation.document["benchmark"]["listing_id"]
        or history[entry] >= 10,
        "paper_v2_selection_invalid",
    )
    mark = {
        "listing_id": listing_id,
        "session": session.isoformat(),
        "source_asset": asset_ref_for(asset).to_json(),
        "raw_asset": asset_ref_for(raw).to_json(),
        "native_price": history[session],
        "native_currency": "USD",
        "applied_fx_rate": 1,
        "basis": {
            "reference_sha256": hashlib.sha256(
                _bytes({"dates": [day.isoformat() for day in reference_dates], "closes": closes})
            ).hexdigest(),
            "held_closes_sha256": hashlib.sha256(
                _bytes([[day.isoformat(), history[day]] for day in held_dates])
            ).hexdigest(),
            "history_sha256": hashlib.sha256(
                _bytes([[day.isoformat(), price] for day, price in sorted(history.items())])
            ).hexdigest(),
            "recorded_overlap_sha256": hashlib.sha256(
                _bytes([[day.isoformat(), price] for day, price in sorted(recorded_prices.items())])
            ).hexdigest(),
        },
    }
    if exact is not None:
        _require(_same(exact, mark))
    return mark, history


def _config(capital: float, symbols: list[str]) -> SimulationConfig:
    return SimulationConfig(
        name=PAPER_POLICY_VERSION,
        mode=SimulationMode.PORTFOLIO,
        starting_capital=capital,
        rebalance_frequency=RebalanceFrequency.NEVER,
        selected_symbols=symbols,
        base_currency="USD",
        missing_price_policy=MissingPricePolicy.FAIL,
        transaction_cost_bps=10,
        slippage_bps=5,
        custom_parameters={ACCOUNTING_VERSION_PARAMETER: ACCOUNTING_VERSION},
    )


def _receipt(
    activation: PaperArtifact, marks: list[dict[str, Any]], *, spy: bool
) -> dict[str, Any] | None:
    selected = [activation.document["benchmark"]] if spy else activation.document["buys"]
    if not selected:
        return None  # Cash is not a dummy asset or an engine call with missing prices.
    capital = 100000.0 if spy else 30000.0
    config = _config(capital, [row["symbol"] for row in selected])
    by_listing = {row["listing_id"]: row["native_price"] for row in marks}
    session = _day(activation.document["calendar"][0])
    prices = pl.DataFrame(
        [
            {
                "date": session,
                "listing_id": row["listing_id"],
                "symbol": row["symbol"],
                "close": _native(by_listing[row["listing_id"]], positive=True),
                "currency": "USD",
            }
            for row in selected
        ]
    )
    signals = prices.select("listing_id", "symbol").with_columns(
        pl.lit(_activation_date(_timestamp(activation.document["recorded_at"]))).alias("date")
    )
    result = AccountingEngine(config).run(prices=prices, signals=signals, calendar=[session])
    _require(
        result.holdings.height == result.trades.height == len(selected)
        and not result.unresolved_observations,
        "paper_v2_native_insufficient",
    )
    # Retain the native outputs, including cash and costs. Do not replace their
    # arithmetic with a parallel execution engine or quantized model rows.
    receipt = json.loads(
        json.dumps(
            {
                "config": config.to_dict(),
                "config_sha256": hashlib.sha256(_bytes(config.to_dict())).hexdigest(),
                "input_hash": result.input_hash,
                "cash": result.daily_curves["cash"][0],
                "holdings": result.holdings.to_dicts(),
                "trades": result.trades.to_dicts(),
            },
            default=_date_json,
            allow_nan=False,
        )
    )
    with localcontext(_DECIMAL_CONTEXT):
        entry_value = _decimal(receipt["cash"]) + sum(
            (_decimal(row["quantity"]) * _decimal(row["price"]) for row in receipt["holdings"]),
            Decimal(0),
        )
        costs = sum((_decimal(row["costs"]) for row in receipt["trades"]), Decimal(0))
        receipt["native_entry_roundoff"] = _money(entry_value + costs - _decimal(capital))
    return cast(dict[str, Any], receipt)


def _execution_document(
    activation: PaperArtifact,
    marks: list[dict[str, Any]],
    histories: dict[str, dict[date, float]],
    *,
    owner: str,
    recorded: datetime,
    revision: str,
    job_id: str,
    receipts: dict[str, Any] | None = None,
    calendar: Any = None,
) -> dict[str, Any]:
    dates = _reference_dates(activation, calendar)
    return {
        **_header(owner, recorded, revision),
        "activation": asset_ref_for(activation.asset).to_json(),
        "origin_job_id": job_id,
        "session": activation.document["calendar"][0],
        "marks": marks,
        "reference": {
            "dates": [day.isoformat() for day in dates],
            "closes": [
                {"listing_id": listing_id, "closes": [history[day] for day in dates]}
                for listing_id, history in sorted(histories.items())
            ],
        },
        "native_executions": receipts
        if receipts is not None
        else {
            "spy": _receipt(activation, marks, spy=True),
            "stocks": _receipt(activation, marks, spy=False),
        },
    }


def _validate_execution(
    asset: DataAsset,
    activation: PaperArtifact,
    *,
    store: AssetStore,
    sources: dict[str, DataAsset],
    known: dict[str, dict[date, float]],
    verified_revisions: set[str],
    calendar: Any,
) -> PaperArtifact:
    owner = activation.document["owner_id"]
    doc = _read_header(
        asset,
        kind=EXECUTION_KIND,
        subject=str(activation.asset.id),
        owner=owner,
        store=store,
        verified_revisions=verified_revisions,
    )
    _shape(
        doc,
        HEADER_FIELDS
        | {"activation", "origin_job_id", "session", "marks", "reference", "native_executions"},
    )
    _uuid(doc["origin_job_id"])
    session = _day(doc["session"])
    recorded = _timestamp(doc["recorded_at"])
    _capture_time(session, recorded, calendar)
    _require(
        doc["session"] == activation.document["calendar"][0]
        and asset.period_start == asset.period_end == session
        and recorded >= _timestamp(activation.document["recorded_at"])
        and _same(doc["activation"], asset_ref_for(activation.asset).to_json())
    )
    _shape(doc["reference"], {"dates", "closes"})
    _shape(doc["native_executions"], {"spy", "stocks"})
    marks, histories = _prove_marks(
        activation,
        sources=sources,
        known=known,
        session=session,
        cutoff=recorded,
        store=store,
        exact=doc["marks"],
        calendar=calendar,
    )
    expected = _execution_document(
        activation,
        marks,
        histories,
        owner=owner,
        recorded=recorded,
        revision=doc["execution_revision"],
        job_id=doc["origin_job_id"],
        calendar=calendar,
    )
    # Reproduction checks every nested config/receipt field and primitive type,
    # not merely a config digest, copied quantity, or ordinary dict equality.
    _require(_same(doc, expected))
    for listing_id, history in histories.items():
        known[listing_id].update(history)
    return PaperArtifact(asset, doc)


def _prove_marks(
    activation: PaperArtifact,
    *,
    sources: dict[str, DataAsset],
    known: dict[str, dict[date, float]],
    session: date,
    cutoff: datetime,
    store: AssetStore,
    execution: PaperArtifact | None = None,
    previous: tuple[PaperArtifact, ...] = (),
    exact: Any = None,
    calendar: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[date, float]]]:
    calendar = calendar if calendar is not None else _proof_calendar(activation)
    if exact is not None:
        _require(type(exact) is list and len(exact) == len(sources))
        _require([mark["listing_id"] for mark in exact] == list(sources))
    reference = (
        {row["listing_id"]: row["closes"] for row in execution.document["reference"]["closes"]}
        if execution
        else {}
    )
    captured: dict[str, dict[date, float]] = {listing_id: {} for listing_id in sources}
    for artifact in ((execution,) if execution else ()) + previous:
        for mark in artifact.document["marks"]:
            captured[mark["listing_id"]][_day(mark["session"])] = mark["native_price"]
    marks, histories = [], {}
    for index, (listing_id, source) in enumerate(sources.items()):
        mark, history = _mark(
            listing_id=listing_id,
            source=source,
            activation=activation,
            session=session,
            cutoff=cutoff,
            store=store,
            known=known[listing_id],
            reference=reference.get(listing_id),
            recorded_prices=captured[listing_id],
            exact=exact[index] if exact is not None else None,
            calendar=calendar,
        )
        marks.append(mark)
        histories[listing_id] = history
    return marks, histories


def _capture_time(session: date, recorded: datetime, calendar: Any = None) -> None:
    if calendar is None:
        try:
            _timely(session, recorded)
        except (TypeError, ValueError):
            raise PaperV2Error("paper_v2_deadline_missed") from None
        calendar = _calendar(session)
    # One invocation-local authoritative calendar, not a proof cache. Avoid
    # rebuilding the same XNYS holidays once per historical capture.
    _require(
        not timezone.is_naive(recorded)
        and calendar.is_session(session)
        and recorded > calendar.session_close(session).to_pydatetime()
        and recorded < calendar.session_open(calendar.next_session(session)).to_pydatetime(),
        "paper_v2_deadline_missed",
    )


def _measurement_document(
    activation: PaperArtifact,
    execution: PaperArtifact,
    *,
    marks: list[dict[str, Any]],
    session: date,
    owner: str,
    recorded: datetime,
    revision: str,
    job_id: str,
) -> dict[str, Any]:
    return {
        **_header(owner, recorded, revision),
        "activation": asset_ref_for(activation.asset).to_json(),
        "execution": asset_ref_for(execution.asset).to_json(),
        "origin_job_id": job_id,
        "session": session.isoformat(),
        "marks": marks,
    }


def _validate_measurement(
    asset: DataAsset,
    activation: PaperArtifact,
    execution: PaperArtifact,
    *,
    store: AssetStore,
    sources: dict[str, DataAsset],
    known: dict[str, dict[date, float]],
    previous: tuple[PaperArtifact, ...],
    verified_revisions: set[str],
    calendar: Any,
) -> PaperArtifact:
    owner = activation.document["owner_id"]
    doc = _read_header(
        asset,
        kind=MEASUREMENT_KIND,
        subject=asset.subject,
        owner=owner,
        store=store,
        verified_revisions=verified_revisions,
    )
    _shape(doc, HEADER_FIELDS | {"activation", "execution", "origin_job_id", "session", "marks"})
    _uuid(doc["origin_job_id"])
    session = _day(doc["session"])
    recorded = _timestamp(doc["recorded_at"])
    _capture_time(session, recorded, calendar)
    _require(
        doc["session"] in activation.document["calendar"]
        and asset.subject == f"{activation.asset.id}:{doc['session']}"
        and asset.period_start == asset.period_end == session
        and recorded >= _timestamp(execution.document["recorded_at"])
        and _same(doc["activation"], asset_ref_for(activation.asset).to_json())
        and _same(doc["execution"], asset_ref_for(execution.asset).to_json())
    )
    marks, histories = _prove_marks(
        activation,
        sources=sources,
        known=known,
        session=session,
        cutoff=recorded,
        store=store,
        execution=execution,
        previous=previous,
        exact=doc["marks"],
        calendar=calendar,
    )
    expected = _measurement_document(
        activation,
        execution,
        marks=marks,
        session=session,
        owner=owner,
        recorded=recorded,
        revision=doc["execution_revision"],
        job_id=doc["origin_job_id"],
    )
    _require(_same(doc, expected))
    for listing_id, history in histories.items():
        known[listing_id].update(history)
    return PaperArtifact(asset, doc)


def _origin_binding(
    activation: PaperArtifact, execution: PaperArtifact, capture: PaperArtifact
) -> dict[str, Any]:
    return {
        "identity": capture_identity(activation),
        "session": capture.document["session"],
        "recorded_at": capture.document["recorded_at"],
        "execution_revision": capture.document["execution_revision"],
        "execution": asset_ref_for(execution.asset).to_json(),
        "measurement": asset_ref_for(capture.asset).to_json(),
    }


def _verify_origin(
    activation: PaperArtifact, execution: PaperArtifact, capture: PaperArtifact
) -> JobRun:
    doc = capture.document
    job = JobRun.objects.filter(pk=_uuid(doc["origin_job_id"])).first()
    recorded = _timestamp(doc["recorded_at"])
    _require(
        job is not None
        and job.job_name == capture_job_name(capture_identity(activation))
        and job.region == "us"
        and job.target_date == _day(doc["session"])
        and not timezone.is_naive(job.started_at)
        and job.started_at <= recorded
        and (job.finished_at is None or job.finished_at >= recorded)
        and type(job.details) is dict
        and _same(
            job.details.get("paper_v2_origin"), _origin_binding(activation, execution, capture)
        )
    )
    assert job is not None
    if doc["session"] == activation.document["calendar"][0]:
        _require(
            execution.document["origin_job_id"] == str(job.pk)
            and execution.document["recorded_at"] == doc["recorded_at"]
            and execution.document["execution_revision"] == doc["execution_revision"]
        )
    # RUNNING/FAILED/SUCCESS are diagnostics, neither proof nor a veto on an
    # atomically committed original capture. The full binding is mandatory.
    return job


def read_state(
    *, user: ProductUser, store: AssetStore, activation_asset: DataAsset | None = None
) -> PaperState:
    revisions: set[str] = set()
    activation = read_activation(
        user=user, store=store, asset=activation_asset, verified_revisions=revisions
    )
    asset = _one(EXECUTION_KIND, str(activation.asset.id))
    assets = list(
        DataAsset.objects.filter(
            kind=MEASUREMENT_KIND,
            subject__startswith=f"{activation.asset.id}:",
        ).order_by("subject", "id")
    )
    if asset is None:
        _require(not assets)  # present orphan/malformed evidence is NOT absence
        return PaperState(activation, None, ())
    _require(len(assets) <= 127 and len({row.subject for row in assets}) == len(assets))
    calendar = _proof_calendar(activation)
    sources, known = _source_context(activation, store, calendar)
    execution = _validate_execution(
        asset,
        activation,
        store=store,
        sources=sources,
        known=known,
        verified_revisions=revisions,
        calendar=calendar,
    )
    captures: tuple[PaperArtifact, ...] = ()
    for row in assets:
        capture = _validate_measurement(
            row,
            activation,
            execution,
            store=store,
            sources=sources,
            known=known,
            previous=captures,
            verified_revisions=revisions,
            calendar=calendar,
        )
        _verify_origin(activation, execution, capture)
        captures += (capture,)
    _require(
        bool(captures) and captures[0].document["session"] == activation.document["calendar"][0]
    )
    return PaperState(activation, execution, captures)


def _capture_for_job(
    *,
    user: ProductUser,
    session: date,
    activation_asset: DataAsset,
    job: JobRun,
    identity: dict[str, Any],
    store: AssetStore,
) -> PaperArtifact:
    """Called with the child target lock held. Owner lock spans durable commit."""
    owner = str(user.pk)
    published_paths: list[str] = []
    try:
        with (
            target_job_lock(
                job_name=activation_subject(owner), region="us", target_date=PROTOCOL_EPOCH
            ),
            transaction.atomic(durable=True),
        ):
            state = read_state(user=user, store=store, activation_asset=activation_asset)
            activation, execution = state.activation, state.execution
            calendar = _proof_calendar(activation)
            _require(_same(identity, capture_identity(activation)))
            _require(
                session.isoformat() in activation.document["calendar"], "paper_v2_outside_horizon"
            )
            _require(
                not any(row.document["session"] == session.isoformat() for row in state.captures)
            )
            cutoff = timezone.now()
            _capture_time(session, cutoff, calendar)
            _require(
                execution is not None or session.isoformat() == activation.document["calendar"][0],
                "paper_v2_source_unverified",
            )
            sources, known = _source_context(activation, store, calendar)
            for artifact in ((execution,) if execution else ()) + state.captures:
                for mark in artifact.document["marks"]:
                    source = resolve_asset_ref(
                        _ref(mark["source_asset"]),
                        cutoff=_timestamp(artifact.document["recorded_at"]),
                    )
                    _raw, history = _history(
                        source,
                        cutoff=_timestamp(artifact.document["recorded_at"]),
                        session=_day(mark["session"]),
                        store=store,
                        calendar=calendar,
                    )
                    known[mark["listing_id"]].update(history)
            marks, histories = _prove_marks(
                activation,
                sources=sources,
                known=known,
                session=session,
                cutoff=cutoff,
                store=store,
                execution=execution,
                previous=state.captures,
                calendar=calendar,
            )
            # Native execution proof can also be expensive. Recheck AFTER it,
            # the clean-revision guard, and the owner/child lock waits.
            receipts = (
                {
                    "spy": _receipt(activation, marks, spy=True),
                    "stocks": _receipt(activation, marks, spy=False),
                }
                if execution is None
                else None
            )
            revision = clean_git_revision(Path(settings.BASE_DIR))
            recorded = timezone.now()
            _capture_time(session, recorded, calendar)
            _require(recorded >= cutoff)
            origin = JobRun.objects.select_for_update().get(pk=job.pk)
            _require(
                origin.job_name == capture_job_name(identity)
                and origin.region == "us"
                and origin.target_date == session
                and origin.started_at <= recorded
                and type(origin.details) is dict
                and "paper_v2_origin" not in origin.details
            )
            if execution is None:
                doc = _execution_document(
                    activation,
                    marks,
                    histories,
                    owner=owner,
                    recorded=recorded,
                    revision=revision,
                    job_id=str(job.pk),
                    receipts=receipts,
                    calendar=calendar,
                )
                _require(_same(doc["native_executions"], receipts))
                execution = _publish(
                    kind=EXECUTION_KIND, subject=str(activation.asset.id), document=doc, store=store
                )
                published_paths.append(execution.asset.relative_path)
                # The positive first capture binds the frozen entry mark too.
                marks = [
                    {
                        **mark,
                        "basis": {
                            **mark["basis"],
                            "recorded_overlap_sha256": hashlib.sha256(
                                _bytes([[session.isoformat(), mark["native_price"]]])
                            ).hexdigest(),
                        },
                    }
                    for mark in marks
                ]
            document = _measurement_document(
                activation,
                execution,
                marks=marks,
                session=session,
                owner=owner,
                recorded=recorded,
                revision=revision,
                job_id=str(job.pk),
            )
            capture = _publish(
                kind=MEASUREMENT_KIND,
                subject=f"{activation.asset.id}:{session.isoformat()}",
                document=document,
                store=store,
            )
            published_paths.append(capture.asset.relative_path)
            final_clock = timezone.now()
            _capture_time(session, final_clock, calendar)
            _require(final_clock >= recorded)
            # REQUIRED atomic origin identity: no asset can commit without this
            # full independent binding in its ORIGINATING row, and vice versa.
            details = {
                **origin.details,
                "paper_v2_origin": _origin_binding(activation, execution, capture),
            }
            JobRun.objects.filter(pk=origin.pk).update(details=details)
            return capture
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        DecimalException,
        PriceFrameChecksumMismatchError,
        pl.exceptions.PolarsError,
    ):
        # Only unregistered files from a rolled-back publication may be removed.
        # Never mutate/delete an immutable evidence row or another winner's file.
        for path in published_paths:
            if not DataAsset.objects.filter(relative_path=path).exists():
                store.resolve(path).unlink(missing_ok=True)
        raise


def _valuation(execution: PaperArtifact, capture: PaperArtifact, *, empty: bool) -> dict[str, Any]:
    prices = {row["listing_id"]: _decimal(row["native_price"]) for row in capture.document["marks"]}
    receipts = execution.document["native_executions"]
    with localcontext(_DECIMAL_CONTEXT):

        def native(receipt: dict[str, Any]) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
            cash = _decimal(receipt["cash"])
            value = cash + sum(
                (
                    _decimal(row["quantity"]) * prices[row["listing_id"]]
                    for row in receipt["holdings"]
                ),
                Decimal(0),
            )
            price_pnl = sum(
                (
                    _decimal(row["quantity"]) * (prices[row["listing_id"]] - _decimal(row["price"]))
                    for row in receipt["holdings"]
                ),
                Decimal(0),
            )
            costs = sum((_decimal(row["costs"]) for row in receipt["trades"]), Decimal(0))
            roundoff = _money_value(receipt["native_entry_roundoff"])
            _require(
                value - _decimal(receipt["config"]["starting_capital"])
                == price_pnl - costs + roundoff
            )
            return value, cash, price_pnl, costs, roundoff

        s, spy_cash, spy_price_pnl, spy_costs, spy_roundoff = native(receipts["spy"])
        b, c = Decimal(30000), Decimal(100000)
        if empty:
            _require(receipts["stocks"] is None)
            k, stock_cash, stock_price_pnl, stock_costs, stock_roundoff = (
                b,
                b,
                Decimal(0),
                Decimal(0),
                Decimal(0),
            )
        else:
            k, stock_cash, stock_price_pnl, stock_costs, stock_roundoff = native(receipts["stocks"])
        a = s * 7 / 10 + k
        price_pnl = spy_price_pnl * 7 / 10 + stock_price_pnl
        costs = spy_costs * 7 / 10 + stock_costs
        roundoff = spy_roundoff * 7 / 10 + stock_roundoff
        difference = a - s
        _require(
            a - c == price_pnl - costs + roundoff
            and difference == k - s * 3 / 10
            and a - c == (s - c) * 7 / 10 + (k - b)
            and (a - c) / c == (s - c) * 7 / (10 * c) + (k - b) / c
        )
        money = {
            "account_value_usd": a,
            "full_spy_value_usd": s,
            "stock_or_cash_value_usd": k,
            "account_pnl_usd": a - c,
            "full_spy_pnl_usd": s - c,
            "stock_or_cash_pnl_usd": k - b,
            "same_dollar_spy_pnl_usd": s * 3 / 10 - b,
            "account_minus_spy_usd": difference,
            "cash_usd": spy_cash * 7 / 10 + stock_cash,
            "fixed_quantity_price_pnl_usd": price_pnl,
            "entry_costs_usd": costs,
            "native_entry_roundoff_usd": roundoff,
            "account_return": (a - c) / c,
            "spy_allocation_return_contribution": (s - c) * 7 / (10 * c),
            "stock_or_cash_return_contribution": (k - b) / c,
            "same_dollar_spy_return_contribution": (s * 3 / 10 - b) / c,
            "account_minus_spy_return": difference / c,
        }
        return {
            **{key: _money(value) for key, value in money.items()},
            "effect_label": "cash_allocation_effect" if empty else "stock_vs_same_dollar_SPY",
            "stock_selection_evidence": None if empty else "canonical_native_6m_BUY",
            "stock_selection_excess_usd": None if empty else _money(difference),
            "cash_allocation_effect_usd": _money(difference) if empty else None,
        }


def _path_risk(days: list[date], valuations: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {}
    # Native metrics get NO benchmark columns: no alpha/beta/tracking estimator.
    for label, key in (("account", "account_value_usd"), ("spy", "full_spy_value_usd")):
        values = [float(_money_value(value[key])) for value in valuations]
        _require(
            all(math.isfinite(value) and value > 0 for value in values),
            "paper_v2_native_insufficient",
        )
        ratios = [values[index] / values[index - 1] for index in range(1, len(values))]
        _require(
            all(math.isfinite(value) and value > 0 for value in ratios),
            "paper_v2_native_insufficient",
        )
        curve = pl.DataFrame(
            {
                "date": days,
                "portfolio_value": values,
                "daily_return": [0.0] + [value - 1 for value in ratios],
            }
        )
        with np.errstate(all="raise"):
            native = calculate_simulation_metrics(
                daily_curves=curve,
                trades=pl.DataFrame(),
                unresolved_observations=[],
                config=_config(100000.0, ["SPY"]),
            )
        _require(
            all(
                value is None or math.isfinite(value)
                for value in (
                    native.max_drawdown,
                    native.annualized_volatility,
                    native.sharpe_ratio,
                )
            ),
            "paper_v2_native_insufficient",
        )
        metrics[label] = {
            "max_drawdown": native.max_drawdown,
            "annualized_volatility": native.annualized_volatility,
            "sharpe_ratio": native.sharpe_ratio,
            "volatility_status": (
                "native_zero_fewer_than_two_return_intervals" if len(ratios) < 2 else "available"
            ),
            "sharpe_status": (
                "native_undefined_short_or_zero_variance"
                if native.sharpe_ratio is None
                else "available"
            ),
        }
    return {
        "through_session": days[-1].isoformat(),
        "capture_count": len(days),
        "return_intervals": len(days) - 1,
        **metrics,
    }


def read_current_method_paper(
    *, user: ProductUser, store: AssetStore | None = None
) -> dict[str, Any]:
    store = store or AssetStore()
    state = read_state(user=user, store=store)
    activation, execution = state.activation, state.execution
    now = timezone.now()
    _require(not timezone.is_naive(now))
    days = [_day(day) for day in activation.document["calendar"]]
    calendar = _calendar(days[0])
    due = [day.isoformat() for day in days if calendar.session_close(day).to_pydatetime() <= now]
    captured = {row.document["session"]: row for row in state.captures}
    missing = [day for day in due if day not in captured]
    terminal = days[-1].isoformat()
    empty = not activation.document["buys"]
    financials = None
    terminal_reason = None
    if terminal not in due:
        terminal_state = "unmatured"
    elif execution is None:
        terminal_state, terminal_reason = "withheld", "missing_prospective_entry"
    elif terminal not in captured:
        deadline = calendar.session_open(calendar.next_session(days[-1])).to_pydatetime()
        terminal_state = "awaiting_capture" if now < deadline else "withheld"
        terminal_reason = "missing_prospective_terminal"
    else:
        terminal_state = "available"
        financials = _valuation(execution, captured[terminal], empty=empty)
    risk = None
    path_reason = None
    if not due:
        path_state = "awaiting_inception"
    elif missing or execution is None:
        path_state, path_reason = "withheld", "missing_prospective_sessions"
    else:
        path_state = "available" if len(due) == 127 else "available_to_date"
        assert execution is not None
        try:
            risk = _path_risk(
                [_day(day) for day in due],
                [_valuation(execution, captured[day], empty=empty) for day in due],
            )
        except (
            PaperV2Error,
            DecimalException,
            FloatingPointError,
            OverflowError,
            ZeroDivisionError,
        ):
            path_state, path_reason = "withheld", "native_path_not_representable"
    return {
        "contract": CONTRACT,
        "policy_version": PAPER_POLICY_VERSION,
        "policy_sha256": PAPER_POLICY_SHA256,
        "activation": asset_ref_for(activation.asset).to_json(),
        "activated_at": activation.document["recorded_at"],
        "notional_usd": 100000,
        "buys": activation.document["buys"],
        "cash_sleeve_usd": 30000 if empty else 0,
        "inception_session": days[0].isoformat(),
        "terminal_session": terminal,
        "calendar_size": 127,
        "due_count": len(due),
        "captured_count": len(captured),
        "missing_sessions": missing,
        "execution": asset_ref_for(execution.asset).to_json() if execution else None,
        "terminal": {
            "status": terminal_state,
            "insufficiency_reason": terminal_reason,
            "financials": financials,
        },
        "path": {"status": path_state, "insufficiency_reason": path_reason, "risk": risk},
        "disclosure": PAPER_POLICY["nonclaims"],
    }

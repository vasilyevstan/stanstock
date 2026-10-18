"""One prospective, notional paper policy. No portfolio or purchase rows.

Publication is explicit and serialized by one owner/version lock through durable
commit. An absent prospective mark cannot be reconstructed from a later vintage.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

import polars as pl
from django.db import transaction
from django.utils import timezone
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.core.jobs import target_job_lock
from stanstock.core.verification_types import AssetRef
from stanstock.data.asof import AsOfData, raw_price_asset_for
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    read_checksummed_bytes,
    register_asset,
    resolve_asset_ref,
)
from stanstock.data.etfs import is_supported_investable_etf
from stanstock.data.models import DataAsset, Listing
from stanstock.data.provider_policy import TWELVE_DATA_PROVIDER
from stanstock.data.research_product import verify_product_price_content
from stanstock.research.forward_scorecard import (
    PROTOCOL_EPOCH,
    _authority,
    read_forward_declaration,
)
from stanstock.research.models import Prediction
from stanstock.research.price_product_config import (
    MOMENTUM_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    load_price_product_config,
)
from stanstock.research.product_reader import ProductUser, read_research_product_performance
from stanstock.research.reporting import iter_product_prediction_projection
from stanstock.simulation.engine import AccountingEngine
from stanstock.simulation.metrics import calculate_simulation_metrics
from stanstock.simulation.service import _get_git_revision
from stanstock.simulation.types import (
    ACCOUNTING_VERSION,
    ACCOUNTING_VERSION_PARAMETER,
    MissingPricePolicy,
    RebalanceFrequency,
    SimulationConfig,
    SimulationMode,
)

PAPER_POLICY_VERSION = "current-method-paper-6m-v1"
ACTIVATION_KIND = "paper_portfolio_activation"
EXECUTION_KIND = "paper_portfolio_execution"
MEASUREMENT_KIND = "paper_portfolio_measurement"
PAPER_POLICY = {
    "version": PAPER_POLICY_VERSION,
    "notional_usd": 100000,
    "spy_weight": 0.7,
    "equal_all_native_current_6m_buy_weight": 0.3,
    "empty_buy_sleeve": "cash",
    "under_10_weight": 0,
    "calculation_version": ACCOUNTING_VERSION,
    "product_config_hash": PRODUCT_EFFECTIVE_CONFIG_HASH,
    "execution": "first_XNYS_session_strictly_after_activation_date_at_close",
    "terminal_sessions_after_inception": 126,
    "transaction_cost_bps": 10,
    "slippage_bps": 5,
    "return_basis": "split_adjusted_price_only_no_dividends_or_cash_yield",
    "management": "frozen_quantities_no_rebalance_contributions_or_liquidation",
    "marks": "prospectively_captured_by_next_session_open_no_late_backfill",
    "baseline": "SPY_only_same_inception_calendar_notional_and_costs",
    "nonclaims": "notional_only_no_activation_of_personal_portfolio_or_skill_claim",
}
PAPER_POLICY_SHA256 = "25a6a2b5f6e0b37ab2f55d0fced023620d31614433fa16d0e8e3ea545318c835"


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True, slots=True)
class PaperArtifact:
    asset: DataAsset | None
    document: dict[str, Any]


def _owner(user: ProductUser, store: AssetStore) -> str:
    owner, source_owner, provider = _authority(user, store)
    if (
        owner != source_owner
        or provider != TWELVE_DATA_PROVIDER
        or hashlib.sha256(_bytes(PAPER_POLICY)).hexdigest() != PAPER_POLICY_SHA256
    ):
        raise ValueError("Paper policy requires its pinned, native owner/provider contract")
    return owner


def _calendar(start: date) -> Any:
    return get_calendar("XNYS", start=start - timedelta(days=7), end=start + timedelta(days=740))


def _activation_date(activated_at: datetime) -> date:
    if timezone.is_naive(activated_at):
        raise ValueError("Paper activation requires an aware publication time")
    return activated_at.astimezone(ZoneInfo("America/New_York")).date()


def _sessions(activated_at: datetime) -> list[date]:
    activation_date = _activation_date(activated_at)
    calendar = _calendar(activation_date)
    inception = calendar.date_to_session(activation_date + timedelta(days=1), direction="next")
    return [day.date() for day in calendar.sessions_window(inception, 127)]


def _timely(session: date, recorded_at: datetime) -> None:
    calendar = _calendar(session)
    if (
        recorded_at.tzinfo is None
        or not calendar.is_session(session)
        or recorded_at < calendar.session_close(session).to_pydatetime()
        or recorded_at >= calendar.session_open(calendar.next_session(session)).to_pydatetime()
    ):
        raise ValueError(
            "Paper marks require actual prospective capture after close, before next open"
        )


def _subject(owner: str) -> str:
    return f"{PAPER_POLICY_VERSION}:{owner}"


def _one(kind: str, subject: str) -> DataAsset | None:
    rows = list(DataAsset.objects.filter(provider="stanstock", kind=kind, subject=subject))
    if len(rows) > 1:
        raise ValueError("Paper artifact registry is ambiguous")
    return rows[0] if rows else None


def _header(owner: str, now: datetime, revision: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Paper execution revision is unavailable")
    return {
        "policy_version": PAPER_POLICY_VERSION,
        "policy_sha256": PAPER_POLICY_SHA256,
        "owner_id": owner,
        "recorded_at": now.isoformat(),
        "execution_revision": revision,
    }


def _publish(
    *,
    kind: str,
    subject: str,
    document: dict[str, Any],
    store: AssetStore,
) -> PaperArtifact:
    payload = _bytes(document)
    stored = store.write_bytes(
        f"simulations/paper/{document['owner_id']}/{kind}/{hashlib.sha256(payload).hexdigest()}.json",
        payload,
    )
    now = datetime.fromisoformat(document["recorded_at"])
    asset = register_asset(
        provider="stanstock",
        kind=kind,
        subject=subject,
        stored=stored,
        retrieved_at=now,
        available_at=now,
        metadata={
            name: document[name]
            for name in (
                "owner_id",
                "policy_version",
                "policy_sha256",
                "recorded_at",
                "execution_revision",
            )
        },
    )
    return PaperArtifact(asset, document)


def _read(
    asset: DataAsset,
    *,
    kind: str,
    subject: str,
    owner: str,
    store: AssetStore,
) -> dict[str, Any]:
    payload = read_checksummed_bytes(store, asset)
    document = json.loads(payload)
    if not isinstance(document, dict) or payload != _bytes(document):
        raise ValueError("Paper artifact has an invalid canonical document")
    recorded = datetime.fromisoformat(document["recorded_at"])
    if (
        asset.provider != "stanstock"
        or asset.kind != kind
        or asset.subject != subject
        or document["owner_id"] != owner
        or document["policy_version"] != PAPER_POLICY_VERSION
        or document["policy_sha256"] != PAPER_POLICY_SHA256
        or not isinstance(document["execution_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", document["execution_revision"]) is None
        or recorded.tzinfo is None
        or recorded > timezone.now()
        or asset.available_at != recorded
        or asset.retrieved_at != recorded
        or asset.metadata
        != {
            name: document[name]
            for name in (
                "owner_id",
                "policy_version",
                "policy_sha256",
                "recorded_at",
                "execution_revision",
            )
        }
    ):
        raise ValueError("Paper artifact identity or chronology is invalid")
    return document


def _native_selection(
    *,
    user: ProductUser,
    store: AssetStore,
    target: date | None = None,
    require_current: bool,
    activation_time: datetime | None = None,
) -> dict[str, Any]:
    scope = read_research_product_performance(user=user, store=store)
    if not scope.available or not scope.runs:
        raise ValueError("Paper source scope is unavailable")
    if require_current and not scope.current.available:
        raise ValueError("Current native paper input is unavailable")
    target = target or scope.current.run.target_date  # type: ignore[union-attr]
    runs = [
        item
        for item in scope.runs
        if item.run.target_date == target
        and (activation_time is None or item.run.generated_at <= activation_time)
    ]
    if not runs or any(
        item.provider != TWELVE_DATA_PROVIDER
        or item.evidence_grade != "observed"
        or not item.run.issued_on_time
        for item in runs
    ):
        raise ValueError("Paper requires observed, reportable native current inputs")
    canonical_ids = [
        row["id"]
        for row in iter_product_prediction_projection(tuple(item.run.id for item in runs))
        if row["canonical"]
        and row["reportable"]
        and row["method_version"] == MOMENTUM_METHOD_VERSION
        and row["evidence_role"] == "decision"
        and row["horizon"] == "6m"
    ]
    decisions = list(
        Prediction.objects.select_related(
            "analysis__run",
            "listing__security__company",
            "listing__latest_market_data__source_asset",
        )
        .filter(id__in=canonical_ids)
        .order_by("listing_id")
    )
    expected = (
        {card.listing.id for card in scope.current.cards}
        if require_current
        else set(
            Prediction.objects.filter(
                analysis__run_id__in=[item.run.id for item in runs],
                method_version=MOMENTUM_METHOD_VERSION,
            ).values_list("listing_id", flat=True)
        )
    )
    if not decisions or {row.listing_id for row in decisions} != expected:
        raise ValueError("Paper canonical decision coverage is incomplete")
    evidence, buys = [], []
    config = load_price_product_config()
    for row in decisions:
        listing = row.listing
        recommendation = row.calculation["recommendation"]
        risk = row.calculation["risk"]
        action = recommendation["suggestion"]
        if (
            listing.currency != "USD"
            or listing.region != "us"
            or not listing.is_active
            or row.config_hash != PRODUCT_EFFECTIVE_CONFIG_HASH
            or action not in {"buy", "hold", "avoid"}
            or row.recommendation != action
            or recommendation["decision_horizon_sessions"] != 126
        ):
            raise ValueError("Paper decision or permanent USD identity is unproved")
        source = resolve_asset_ref(AssetRef.from_json(row.source_assets[0]), cutoff=row.data_cutoff)
        evidence.append(
            {
                "prediction_id": str(row.id),
                "run_id": str(row.analysis.run_id),
                "listing_id": str(listing.id),
                "symbol": listing.provider_symbol,
                "action": action,
                "source_assets": row.source_assets,
                "calculation_artifact": row.calculation["calculation_artifact"],
                "blocking_reasons": recommendation["blocking_reasons"],
                "allocation_restriction": recommendation["allocation_restriction"],
            }
        )
        if action != "buy":
            continue
        if any(
            not isinstance(risk[name], (int, float))
            or isinstance(risk[name], bool)
            or not math.isfinite(risk[name])
            for name in (
                "annualized_volatility",
                "benchmark_annualized_volatility",
                "relative_volatility",
                "maximum_drawdown",
                "average_dollar_turnover_20d",
            )
        ):
            raise ValueError("Paper BUY risk or liquidity evidence is missing or non-finite")
        # A numerical direction or advisory interval is never a BUY proxy.
        if (
            row.analysis.current_price < 10
            or recommendation["blocking_reasons"]
            or recommendation["allocation_restriction"]
            or source.metadata.get("volume_adjustment_compatible") is not True
            or risk["average_dollar_turnover_20d"] is None
            or risk["average_dollar_turnover_20d"] < config.risk.buy_minimum_dollar_turnover
            or risk["relative_volatility"] is None
            or risk["relative_volatility"] > config.risk.buy_max_relative_volatility
            or risk["maximum_drawdown"] < config.risk.buy_minimum_drawdown
        ):
            raise ValueError("Paper BUY risk, volume or affordability proof is invalid")
        if require_current:
            market = listing.latest_market_data
            if (
                market.session_date != target
                or market.source_asset_id != source.id
                or market.close != row.analysis.current_price
                or market.close < 10
            ):
                raise ValueError("Paper BUY current promotion proof is invalid")
        buys.append(
            {
                "listing_id": str(listing.id),
                "symbol": listing.provider_symbol,
                "prediction_id": str(row.id),
                "native_price": str(row.analysis.current_price),
                "native_currency": "USD",
                "applied_fx_rate": 1,
            }
        )
    spy = [
        listing
        for listing in Listing.objects.select_related("security__company").filter(
            provider_symbol="SPY",
            is_active=True,
        )
        if is_supported_investable_etf(listing)
    ]
    if len(spy) != 1:
        raise ValueError("One native investable USD SPY identity is required")
    benchmarks = {json.dumps(row.source_assets[2], sort_keys=True) for row in decisions}
    if len(benchmarks) != 1:
        raise ValueError("Paper native benchmark source is ambiguous")
    return {
        "source_target_date": target.isoformat(),
        "evidence": evidence,
        "buys": buys,
        "benchmark": {
            "listing_id": str(spy[0].id),
            "symbol": "SPY",
            "source_asset": json.loads(next(iter(benchmarks))),
        },
    }


def activate_current_method_paper(
    *,
    user: ProductUser,
    persist: bool = True,
    store: AssetStore | None = None,
) -> PaperArtifact:
    store = store or AssetStore()
    owner = _owner(user, store)
    with (
        target_job_lock(
            job_name=_subject(owner),
            region="us",
            target_date=PROTOCOL_EPOCH,
        ),
        transaction.atomic(durable=True),
    ):
        existing = _one(ACTIVATION_KIND, _subject(owner))
        if existing is not None:
            return _read_activation(user=user, owner=owner, store=store)
        plan = _activation_plan(user=user, owner=owner, store=store)
        if not persist:
            return plan
        return _publish(
            kind=ACTIVATION_KIND,
            subject=_subject(owner),
            document=plan.document,
            store=store,
        )


def _activation_plan(*, user: ProductUser, owner: str, store: AssetStore) -> PaperArtifact:
    declaration = read_forward_declaration(user=user, store=store)
    selection = _native_selection(user=user, store=store, require_current=True)
    # Registration time is sampled only after source proof, under the winning
    # publication lock; a preview's time is explicitly hypothetical.
    now = timezone.now()
    sessions = _sessions(now)
    return PaperArtifact(
        None,
        {
            **_header(owner, now, _get_git_revision()),
            "policy": PAPER_POLICY,
            "protocol": asset_ref_for(declaration.asset).to_json(),
            "calendar": [day.isoformat() for day in sessions],
            **selection,
        },
    )


def _read_activation(*, user: ProductUser, owner: str, store: AssetStore) -> PaperArtifact:
    declaration = read_forward_declaration(user=user, store=store)
    asset = _one(ACTIVATION_KIND, _subject(owner))
    if asset is None:
        raise ValueError("Paper policy has not been explicitly activated")
    document = _read(asset, kind=ACTIVATION_KIND, subject=_subject(owner), owner=owner, store=store)
    activated = datetime.fromisoformat(document["recorded_at"])
    selection = _native_selection(
        user=user,
        store=store,
        target=date.fromisoformat(document["source_target_date"]),
        require_current=False,
        activation_time=activated,
    )
    if (
        set(document)
        != {
            *_header(owner, activated, document["execution_revision"]),
            "policy",
            "protocol",
            "calendar",
            *selection,
        }
        or document["policy"] != PAPER_POLICY
        or document["protocol"] != asset_ref_for(declaration.asset).to_json()
        or declaration.declared_at > activated
        or document["calendar"] != [day.isoformat() for day in _sessions(activated)]
        or any(document[name] != value for name, value in selection.items())
    ):
        raise ValueError("Paper activation differs from its native declaration or policy")
    return PaperArtifact(asset, document)


def _references(activation: PaperArtifact) -> dict[str, dict[str, str]]:
    selected = {item["listing_id"] for item in activation.document["buys"]}
    return {
        **{
            item["listing_id"]: item["source_assets"][0]
            for item in activation.document["evidence"]
            if item["listing_id"] in selected
        },
        activation.document["benchmark"]["listing_id"]: activation.document["benchmark"][
            "source_asset"
        ],
    }


def _mark(
    *,
    listing_id: str,
    source_ref: dict[str, str],
    activation: PaperArtifact,
    session: date,
    recorded: datetime,
    store: AssetStore,
    exact: dict[str, Any] | None = None,
    inception_mark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    activated = datetime.fromisoformat(activation.document["recorded_at"])
    source = resolve_asset_ref(AssetRef.from_json(source_ref), cutoff=activated)
    if exact is None:
        candidates = list(
            DataAsset.objects.filter(
                provider=source.provider,
                kind="price_history",
                subject=source.subject,
                available_at__lte=recorded,
                retrieved_at__lte=recorded,
                period_end=session,
            ).order_by("-available_at", "-retrieved_at", "-id")[:2]
        )
        if not candidates:
            raise ValueError("Paper price is missing for a prospective session")
        asset = candidates[0]
        if len(candidates) == 2 and (candidates[1].available_at, candidates[1].retrieved_at) == (
            asset.available_at,
            asset.retrieved_at,
        ):
            raise ValueError("Paper prospective price source is ambiguous")
    else:
        asset = resolve_asset_ref(AssetRef.from_json(exact["source_asset"]), cutoff=recorded)
    if (
        asset.provider != source.provider
        or asset.kind != "price_history"
        or asset.subject != source.subject
        or asset.period_end != session
        or any(
            asset.metadata.get(name) != source.metadata.get(name)
            for name in (
                "currency",
                "adjustment",
                "exchange",
                "mic_code",
                "instrument_type",
            )
        )
        or asset.metadata.get("currency") != "USD"
    ):
        raise ValueError("Paper mark has an incompatible native price identity")
    raw = raw_price_asset_for(asset, cutoff=recorded)
    verify_product_price_content(
        asset=asset,
        raw=raw,
        cutoff=recorded,
        target_date=session,
        store=store,
    )
    frame = (
        AsOfData(recorded, store)
        .price_frame_for_asset_with_diagnostics(
            asset=asset,
            through_date=session,
        )
        .frame
    )
    original = (
        AsOfData(activated, store)
        .price_frame_for_asset_with_diagnostics(
            asset=source,
            through_date=date.fromisoformat(activation.document["source_target_date"]),
        )
        .frame
    )
    old = dict(original.select("date", "close").iter_rows())
    current = dict(frame.select("date", "close").iter_rows())
    overlap = set(old) & set(current)
    if not overlap or max(old) not in overlap or any(old[day] != current[day] for day in overlap):
        raise ValueError("Paper share-basis conflict or unproved corporate-event continuity")
    if (
        inception_mark is not None
        and current.get(date.fromisoformat(inception_mark["session"]))
        != inception_mark["native_price"]
    ):
        raise ValueError("Paper inception share basis changed or is unproved")
    price = current.get(session)
    if price is None or not math.isfinite(price) or price <= 0:
        raise ValueError("Paper prospective close is missing or invalid")
    mark = {
        "listing_id": listing_id,
        "session": session.isoformat(),
        "source_asset": asset_ref_for(asset).to_json(),
        "native_price": price,
        "native_currency": "USD",
        "applied_fx_rate": 1,
    }
    if exact is not None and mark != exact:
        raise ValueError("Paper native mark differs from its registered source")
    return mark


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


def _execution_document(
    activation: PaperArtifact,
    marks: list[dict[str, Any]],
    *,
    owner: str,
    recorded: datetime,
    revision: str,
) -> dict[str, Any]:
    assert activation.asset is not None
    day = date.fromisoformat(activation.document["calendar"][0])
    mark_by_listing = {mark["listing_id"]: mark for mark in marks}
    benchmark = activation.document["benchmark"]
    sleeves = []
    for capital, selected in (
        (70000.0, [benchmark]),
        (30000.0, activation.document["buys"]),
    ):
        if not selected:
            continue  # actual cash, never a dummy security or a dummy engine call
        config = _config(capital, [item["symbol"] for item in selected])
        prices = pl.DataFrame(
            [
                {
                    "date": day,
                    "listing_id": item["listing_id"],
                    "symbol": item["symbol"],
                    "close": mark_by_listing[item["listing_id"]]["native_price"],
                    "currency": "USD",
                }
                for item in selected
            ]
        )
        signals = prices.select("listing_id", "symbol").with_columns(
            pl.lit(
                _activation_date(datetime.fromisoformat(activation.document["recorded_at"]))
            ).alias("date")
        )
        result = AccountingEngine(config).run(prices=prices, signals=signals, calendar=[day])
        sleeves.append(
            {
                "config": config.to_dict(),
                "config_sha256": hashlib.sha256(_bytes(config.to_dict())).hexdigest(),
                "input_hash": result.input_hash,
                "cash": result.daily_curves["cash"][0],
                "holdings": result.holdings.to_dicts(),
                "trades": result.trades.to_dicts(),
            }
        )
    # Scaling a single SPY sleeve is exact linear accounting under the same
    # notional/cost policy; no third engine and no new weighted allocator.
    spy = sleeves[0]
    scale = 100000.0 / 70000.0
    document = {
        **_header(owner, recorded, revision),
        "activation": asset_ref_for(activation.asset).to_json(),
        "session": day.isoformat(),
        "marks": marks,
        "sleeves": sleeves,
        "cash": sum(sleeve["cash"] for sleeve in sleeves)
        + (30000.0 if not activation.document["buys"] else 0.0),
        "holdings": [
            {"listing_id": row["listing_id"], "symbol": row["symbol"], "quantity": row["quantity"]}
            for sleeve in sleeves
            for row in sleeve["holdings"]
        ],
        "baseline_cash": spy["cash"] * scale,
        "baseline_quantity": spy["holdings"][0]["quantity"] * scale,
        "baseline_costs": sum(row["costs"] for row in spy["trades"]) * scale,
    }
    # Native frames contain date objects; normalize only at this asset boundary.
    return cast(
        dict[str, Any],
        json.loads(
            json.dumps(
                document,
                sort_keys=True,
                default=_date_json,
                allow_nan=False,
            )
        ),
    )


def _date_json(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError("Paper evidence contains a non-serializable value")


def _read_execution(
    activation: PaperArtifact,
    *,
    owner: str,
    store: AssetStore,
) -> PaperArtifact | None:
    assert activation.asset is not None
    subject = str(activation.asset.id)
    asset = _one(EXECUTION_KIND, subject)
    if asset is None:
        return None
    document = _read(asset, kind=EXECUTION_KIND, subject=subject, owner=owner, store=store)
    recorded = datetime.fromisoformat(document["recorded_at"])
    session = date.fromisoformat(activation.document["calendar"][0])
    _timely(session, recorded)
    refs = _references(activation)
    if len(document["marks"]) != len(refs) or {
        mark["listing_id"] for mark in document["marks"]
    } != set(refs):
        raise ValueError("Paper execution mark coverage is incomplete")
    marks = [
        _mark(
            listing_id=mark["listing_id"],
            source_ref=refs[mark["listing_id"]],
            activation=activation,
            session=session,
            recorded=recorded,
            store=store,
            exact=mark,
        )
        for mark in document["marks"]
    ]
    expected = _execution_document(
        activation,
        marks,
        owner=owner,
        recorded=recorded,
        revision=document["execution_revision"],
    )
    if document != expected:
        raise ValueError("Paper execution differs from native v2 sleeve accounting")
    return PaperArtifact(asset, document)


def _curve_metrics(
    values: list[tuple[date, float]],
    execution: PaperArtifact,
    *,
    baseline: bool,
) -> dict[str, Any]:
    curve = pl.DataFrame(
        {
            "date": [day for day, _value in values],
            "portfolio_value": [value for _day, value in values],
            "daily_return": [0.0]
            + [values[index][1] / values[index - 1][1] - 1 for index in range(1, len(values))],
        }
    )
    if any(
        not math.isfinite(values[index][1] / values[index - 1][1])
        or values[index][1] / values[index - 1][1] <= 0
        for index in range(1, len(values))
    ):
        raise ValueError("Paper daily return ratio is not representable")
    trades = [
        {"gross_value": trade["gross_value"] * (100000.0 / 70000.0 if baseline else 1)}
        for sleeve in (
            execution.document["sleeves"][:1] if baseline else execution.document["sleeves"]
        )
        for trade in sleeve["trades"]
    ]
    return calculate_simulation_metrics(
        daily_curves=curve,
        trades=pl.DataFrame(trades),
        unresolved_observations=[],
        config=_config(100000.0, ["SPY"]),
    ).to_dict()


def _measurement_document(
    activation: PaperArtifact,
    execution: PaperArtifact,
    *,
    session: date,
    marks: list[dict[str, Any]],
    previous: list[PaperArtifact],
    owner: str,
    recorded: datetime,
    revision: str,
) -> dict[str, Any]:
    assert activation.asset is not None and execution.asset is not None
    expected = [day for day in activation.document["calendar"] if day <= session.isoformat()]
    by_session = {item.document["session"]: item.document for item in previous}
    missing = [
        day
        for day in expected[:-1]
        if day not in by_session or by_session[day]["status"] != "available"
    ]
    by_listing = {item["listing_id"]: item["native_price"] for item in marks}
    refs = _references(activation)
    if len(marks) != len(refs) or set(by_listing) != set(refs):
        raise ValueError("Paper measurement price coverage is incomplete")
    position_values = [
        row["quantity"] * by_listing[row["listing_id"]] for row in execution.document["holdings"]
    ]
    if any(not math.isfinite(value) or value <= 0 for value in position_values):
        raise ValueError("Paper position value is not representable")
    value = execution.document["cash"] + sum(position_values)
    baseline_value = execution.document["baseline_cash"] + (
        execution.document["baseline_quantity"]
        * by_listing[activation.document["benchmark"]["listing_id"]]
    )
    if any(not math.isfinite(number) or number <= 0 for number in (value, baseline_value)):
        raise ValueError("Paper valuation is not finitely representable")
    portfolio_metrics = baseline_metrics = None
    if not missing:
        portfolio_metrics = _curve_metrics(
            [
                *[
                    (date.fromisoformat(day), by_session[day]["portfolio_value"])
                    for day in expected[:-1]
                ],
                (session, value),
            ],
            execution,
            baseline=False,
        )
        baseline_metrics = _curve_metrics(
            [
                *[
                    (date.fromisoformat(day), by_session[day]["baseline_value"])
                    for day in expected[:-1]
                ],
                (session, baseline_value),
            ],
            execution,
            baseline=True,
        )
    difference = None
    if portfolio_metrics is not None and baseline_metrics is not None:
        difference = portfolio_metrics["cumulative_return"] - baseline_metrics["cumulative_return"]
    return {
        **_header(owner, recorded, revision),
        "activation": asset_ref_for(activation.asset).to_json(),
        "execution": asset_ref_for(execution.asset).to_json(),
        "session": session.isoformat(),
        "marks": marks,
        "status": "withheld" if missing else "available",
        "insufficiency_reason": "missing_prospective_sessions" if missing else None,
        "missing_sessions": missing,
        "portfolio_value": None if missing else value,
        "baseline_value": None if missing else baseline_value,
        "portfolio_metrics": portfolio_metrics,
        "spy_only_baseline_metrics": baseline_metrics,
        "net_cumulative_return_difference": difference,
        "comparison_disclosure": (
            "Both net paths include identical inception costs and notional. "
            "Return difference is descriptive, not alpha. No paired risk statistic is claimed."
        ),
    }


def _read_measurements(
    activation: PaperArtifact,
    execution: PaperArtifact,
    *,
    owner: str,
    store: AssetStore,
) -> list[PaperArtifact]:
    assert activation.asset is not None
    assets = DataAsset.objects.filter(
        provider="stanstock",
        kind=MEASUREMENT_KIND,
        subject__startswith=f"{activation.asset.id}:",
    )
    pending = []
    for asset in assets:
        document = _read(
            asset,
            kind=MEASUREMENT_KIND,
            subject=asset.subject,
            owner=owner,
            store=store,
        )
        if (
            document["session"] not in activation.document["calendar"]
            or asset.subject != f"{activation.asset.id}:{document['session']}"
        ):
            raise ValueError("Paper measurement has an invalid activation/session identity")
        pending.append(PaperArtifact(asset, document))
    previous: list[PaperArtifact] = []
    refs = _references(activation)
    for item in sorted(pending, key=lambda item: item.document["session"]):
        document = item.document
        if any(prior.document["session"] == document["session"] for prior in previous):
            raise ValueError("Paper measurement registry is ambiguous")
        session = date.fromisoformat(document["session"])
        recorded = datetime.fromisoformat(document["recorded_at"])
        _timely(session, recorded)
        marks = [
            _mark(
                listing_id=mark["listing_id"],
                source_ref=refs[mark["listing_id"]],
                activation=activation,
                session=session,
                recorded=recorded,
                store=store,
                exact=mark,
                inception_mark=next(
                    value
                    for value in execution.document["marks"]
                    if value["listing_id"] == mark["listing_id"]
                ),
            )
            for mark in document["marks"]
        ]
        expected = _measurement_document(
            activation,
            execution,
            session=session,
            marks=marks,
            previous=previous,
            owner=owner,
            recorded=recorded,
            revision=document["execution_revision"],
        )
        if document != expected:
            raise ValueError("Paper measurement differs from its frozen native valuation path")
        previous.append(item)
    return previous


def measure_current_method_paper(
    *,
    user: ProductUser,
    session: date,
    store: AssetStore | None = None,
) -> PaperArtifact:
    store = store or AssetStore()
    owner = _owner(user, store)
    with (
        target_job_lock(
            job_name=_subject(owner),
            region="us",
            target_date=PROTOCOL_EPOCH,
        ),
        transaction.atomic(durable=True),
    ):
        activation = _read_activation(user=user, owner=owner, store=store)
        assert activation.asset is not None
        if session.isoformat() not in activation.document["calendar"]:
            raise ValueError("Paper measurement is outside the frozen 126-session horizon")
        execution = _read_execution(activation, owner=owner, store=store)
        previous = (
            _read_measurements(activation, execution, owner=owner, store=store)
            if execution is not None
            else []
        )
        existing = next(
            (item for item in previous if item.document["session"] == session.isoformat()),
            None,
        )
        if existing is not None:
            return existing
        now = timezone.now()
        _timely(session, now)
        marks = [
            _mark(
                listing_id=listing_id,
                source_ref=ref,
                activation=activation,
                session=session,
                recorded=now,
                store=store,
                inception_mark=None
                if execution is None
                else next(
                    value
                    for value in execution.document["marks"]
                    if value["listing_id"] == listing_id
                ),
            )
            for listing_id, ref in sorted(_references(activation).items())
        ]
        if execution is None:
            if session.isoformat() != activation.document["calendar"][0]:
                raise ValueError(
                    "Paper inception was not captured prospectively; execution withheld"
                )
            execution = _publish(
                kind=EXECUTION_KIND,
                subject=str(activation.asset.id),
                document=_execution_document(
                    activation,
                    marks,
                    owner=owner,
                    recorded=now,
                    revision=_get_git_revision(),
                ),
                store=store,
            )
        document = _measurement_document(
            activation,
            execution,
            session=session,
            marks=marks,
            previous=previous,
            owner=owner,
            recorded=now,
            revision=_get_git_revision(),
        )
        return _publish(
            kind=MEASUREMENT_KIND,
            subject=f"{activation.asset.id}:{session.isoformat()}",
            document=document,
            store=store,
        )


def read_current_method_paper(
    *,
    user: ProductUser,
    store: AssetStore | None = None,
) -> dict[str, Any]:
    store = store or AssetStore()
    owner = _owner(user, store)
    activation = _read_activation(user=user, owner=owner, store=store)
    assert activation.asset is not None
    execution = _read_execution(activation, owner=owner, store=store)
    measurements = (
        _read_measurements(activation, execution, owner=owner, store=store)
        if execution is not None
        else []
    )
    due = []
    calendar = _calendar(date.fromisoformat(activation.document["calendar"][0]))
    now = timezone.now()
    for raw in activation.document["calendar"]:
        session = date.fromisoformat(raw)
        if calendar.session_close(session).to_pydatetime() <= now:
            due.append(raw)
    recorded = {item.document["session"] for item in measurements}
    missing = [day for day in due if day not in recorded]
    latest = measurements[-1] if measurements else None
    return {
        "status": "withheld" if missing else "available" if execution else "awaiting_inception",
        "policy_version": PAPER_POLICY_VERSION,
        "policy_sha256": PAPER_POLICY_SHA256,
        "activation": asset_ref_for(activation.asset).to_json(),
        "activated_at": activation.document["recorded_at"],
        "notional_usd": 100000,
        "buys": activation.document["buys"],
        "cash_sleeve_usd": 30000 if not activation.document["buys"] else 0,
        "inception_session": activation.document["calendar"][0],
        "terminal_session": activation.document["calendar"][-1],
        "execution": (
            asset_ref_for(execution.asset).to_json() if execution and execution.asset else None
        ),
        "latest_measurement": (
            asset_ref_for(latest.asset).to_json() if latest and latest.asset else None
        ),
        "measurement": None if latest is None or missing else latest.document,
        "missing_sessions": missing,
        "insufficiency_reason": "missing_prospective_sessions" if missing else None,
        "disclosure": PAPER_POLICY["nonclaims"],
    }

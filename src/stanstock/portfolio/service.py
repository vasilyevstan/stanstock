from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from django.db import transaction
from django.utils import timezone

from stanstock.data.models import LatestMarketData, Listing
from stanstock.portfolio.models import (
    Portfolio,
    PortfolioHolding,
    PortfolioSnapshot,
    PortfolioSnapshotHolding,
)
from stanstock.research.config import code_revision

MAX_PORTFOLIO_PRICE_AGE_DAYS = 7
SPLIT_WARNING_LOW_RATIO = Decimal("0.60")
SPLIT_WARNING_HIGH_RATIO = Decimal("1.67")


class PortfolioValuationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValuedHolding:
    holding: PortfolioHolding
    market_data: LatestMarketData | None
    cost_basis: Decimal
    market_value: Decimal | None
    unrealized_gain: Decimal | None
    issue: str = ""


@dataclass(frozen=True, slots=True)
class PortfolioValuation:
    portfolio: Portfolio
    positions: tuple[ValuedHolding, ...]
    as_of_date: date
    oldest_price_date: date | None
    newest_price_date: date | None
    cash_balance: Decimal
    securities_value: Decimal | None
    total_value: Decimal | None
    cost_basis: Decimal
    unrealized_gain: Decimal | None
    return_pct: Decimal | None
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    return_definition: str
    dividends_included: bool

    @property
    def complete(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class PortfolioSnapshotBatch:
    created: int
    unchanged: int
    failures: tuple[str, ...]


def validate_holding_listing(*, portfolio: Portfolio, listing: Listing) -> None:
    if portfolio.archived_at is not None:
        raise PortfolioValuationError("Archived portfolios cannot be changed.")
    if not listing.is_active:
        raise PortfolioValuationError(f"{listing.ticker} is not an active listing.")
    if listing.currency.upper() != portfolio.base_currency:
        raise PortfolioValuationError(
            f"{listing.ticker} trades in {listing.currency}; this portfolio is "
            f"restricted to {portfolio.base_currency} until tracked-portfolio FX "
            "conversion is enabled."
        )
    if not LatestMarketData.objects.filter(listing=listing).exists():
        raise PortfolioValuationError(
            f"{listing.ticker} has no current market price and cannot be tracked yet."
        )


def upsert_holding(
    *,
    portfolio: Portfolio,
    listing: Listing,
    quantity: Decimal,
    average_cost: Decimal,
    acquired_on: date | None = None,
    notes: str = "",
) -> PortfolioHolding:
    if quantity <= 0:
        raise PortfolioValuationError("Holding quantity must be positive.")
    if average_cost <= 0:
        raise PortfolioValuationError("Average cost must be positive.")
    validate_holding_listing(portfolio=portfolio, listing=listing)
    holding, _created = PortfolioHolding.objects.update_or_create(
        portfolio=portfolio,
        listing=listing,
        defaults={
            "quantity": quantity,
            "average_cost": average_cost,
            "acquired_on": acquired_on,
            "notes": notes.strip(),
        },
    )
    return holding


def calculate_portfolio_valuation(portfolio: Portfolio) -> PortfolioValuation:
    holdings = list(
        portfolio.holdings.select_related(
            "listing__security__company",
            "listing__latest_market_data__source_asset",
        ).order_by("listing__ticker")
    )
    if not holdings:
        return PortfolioValuation(
            portfolio=portfolio,
            positions=(),
            as_of_date=timezone.localdate(),
            oldest_price_date=None,
            newest_price_date=None,
            cash_balance=portfolio.cash_balance,
            securities_value=Decimal(0),
            total_value=portfolio.cash_balance,
            cost_basis=Decimal(0),
            unrealized_gain=Decimal(0),
            return_pct=None,
            issues=(),
            warnings=(),
            return_definition="price_return",
            dividends_included=False,
        )

    market_rows: dict[object, LatestMarketData] = {}
    for holding in holdings:
        try:
            market_rows[holding.listing_id] = holding.listing.latest_market_data
        except LatestMarketData.DoesNotExist:
            continue
    dates = [row.session_date for row in market_rows.values()]
    newest_price_date = max(dates, default=None)
    oldest_price_date = min(dates, default=None)
    as_of_date = newest_price_date or timezone.localdate()
    issues: list[str] = []
    warnings: list[str] = []
    positions: list[ValuedHolding] = []
    securities_value = Decimal(0)
    cost_basis = Decimal(0)
    return_definitions: set[str] = set()
    dividends_flags: list[bool] = []

    for holding in holdings:
        holding_cost = holding.quantity * holding.average_cost
        cost_basis += holding_cost
        row = market_rows.get(holding.listing_id)
        issue = ""
        if not holding.listing.is_active:
            issue = f"{holding.listing.ticker} is no longer an active listing."
        elif holding.listing.currency.upper() != portfolio.base_currency:
            issue = (
                f"{holding.listing.ticker} uses {holding.listing.currency}, "
                f"not {portfolio.base_currency}."
            )
        elif row is None:
            issue = f"{holding.listing.ticker} has no current market price."
        elif (
            newest_price_date is not None
            and (newest_price_date - row.session_date).days > MAX_PORTFOLIO_PRICE_AGE_DAYS
        ):
            issue = f"{holding.listing.ticker} price is stale ({row.session_date.isoformat()})."

        if row is None:
            positions.append(
                ValuedHolding(
                    holding=holding,
                    market_data=None,
                    cost_basis=holding_cost,
                    market_value=None,
                    unrealized_gain=None,
                    issue=issue,
                )
            )
        else:
            market_value = holding.quantity * row.close
            securities_value += market_value
            metadata = row.source_asset.metadata
            return_definitions.add(str(metadata.get("return_definition") or "price_return"))
            dividends_flags.append(metadata.get("dividends_included") is True)
            positions.append(
                ValuedHolding(
                    holding=holding,
                    market_data=row,
                    cost_basis=holding_cost,
                    market_value=market_value,
                    unrealized_gain=market_value - holding_cost,
                    issue=issue,
                )
            )
        if issue:
            issues.append(issue)

    if (
        newest_price_date is not None
        and (timezone.localdate() - newest_price_date).days > MAX_PORTFOLIO_PRICE_AGE_DAYS
    ):
        warnings.append(
            "Latest portfolio prices are from "
            f"{newest_price_date.isoformat()}; current values may be outdated."
        )

    if issues:
        total_value = None
        complete_securities_value = None
        unrealized_gain = None
        return_pct = None
    else:
        complete_securities_value = securities_value
        total_value = portfolio.cash_balance + securities_value
        unrealized_gain = securities_value - cost_basis
        return_pct = unrealized_gain / cost_basis if cost_basis > 0 else None

    return PortfolioValuation(
        portfolio=portfolio,
        positions=tuple(positions),
        as_of_date=as_of_date,
        oldest_price_date=oldest_price_date,
        newest_price_date=newest_price_date,
        cash_balance=portfolio.cash_balance,
        securities_value=complete_securities_value,
        total_value=total_value,
        cost_basis=cost_basis,
        unrealized_gain=unrealized_gain,
        return_pct=return_pct,
        issues=tuple(issues),
        warnings=tuple(warnings),
        return_definition=(
            next(iter(return_definitions)) if len(return_definitions) == 1 else "mixed_price_return"
        ),
        dividends_included=bool(dividends_flags) and all(dividends_flags),
    )


def record_portfolio_snapshot(
    portfolio: Portfolio,
) -> tuple[PortfolioSnapshot, bool]:
    valuation = calculate_portfolio_valuation(portfolio)
    if not valuation.complete:
        raise PortfolioValuationError(
            "Portfolio snapshot was not recorded: " + " ".join(valuation.issues)
        )
    assert valuation.securities_value is not None
    assert valuation.total_value is not None
    assert valuation.unrealized_gain is not None

    revision = code_revision()
    payload = {
        "portfolio_id": str(portfolio.pk),
        "as_of_date": valuation.as_of_date.isoformat(),
        "base_currency": portfolio.base_currency,
        "cash_balance": str(valuation.cash_balance),
        "positions": [
            {
                "listing_id": str(position.holding.listing_id),
                "quantity": str(position.holding.quantity),
                "average_cost": str(position.holding.average_cost),
                "price": str(position.market_data.close),
                "session_date": position.market_data.session_date.isoformat(),
                "source_asset_id": str(position.market_data.source_asset_id),
            }
            for position in valuation.positions
            if position.market_data is not None
        ],
    }
    input_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    corporate_action_flags = {
        position.holding.listing_id: _corporate_action_suspected(position)
        for position in valuation.positions
    }
    warning_count = sum(corporate_action_flags.values())

    with transaction.atomic():
        snapshot, created = PortfolioSnapshot.objects.get_or_create(
            portfolio=portfolio,
            as_of_date=valuation.as_of_date,
            input_hash=input_hash,
            defaults={
                "oldest_price_date": valuation.oldest_price_date,
                "newest_price_date": valuation.newest_price_date,
                "base_currency": portfolio.base_currency,
                "cash_balance": valuation.cash_balance,
                "securities_value": valuation.securities_value,
                "total_value": valuation.total_value,
                "cost_basis": valuation.cost_basis,
                "unrealized_gain": valuation.unrealized_gain,
                "return_pct": valuation.return_pct,
                "code_revision": revision,
                "return_definition": valuation.return_definition,
                "dividends_included": valuation.dividends_included,
                "corporate_action_warnings": warning_count,
            },
        )
        if not created:
            return snapshot, False

        snapshot_positions: list[PortfolioSnapshotHolding] = []
        for position in valuation.positions:
            market_data = position.market_data
            market_value = position.market_value
            unrealized_gain = position.unrealized_gain
            if market_data is None or market_value is None or unrealized_gain is None:
                raise PortfolioValuationError(
                    f"{position.holding.listing.ticker} could not be valued."
                )
            split_warning = corporate_action_flags[position.holding.listing_id]
            snapshot_positions.append(
                PortfolioSnapshotHolding(
                    snapshot=snapshot,
                    listing=position.holding.listing,
                    source_asset=market_data.source_asset,
                    source_session_date=market_data.session_date,
                    quantity=position.holding.quantity,
                    average_cost=position.holding.average_cost,
                    price=market_data.close,
                    cost_basis=position.cost_basis,
                    market_value=market_value,
                    unrealized_gain=unrealized_gain,
                    corporate_action_suspected=split_warning,
                )
            )
        PortfolioSnapshotHolding.objects.bulk_create(snapshot_positions)
    return snapshot, True


def portfolio_snapshot_series(portfolio: Portfolio) -> list[PortfolioSnapshot]:
    return list(portfolio.snapshots.order_by("recorded_at", "id"))


def snapshot_all_portfolios() -> PortfolioSnapshotBatch:
    created = 0
    unchanged = 0
    failures: list[str] = []
    portfolios = Portfolio.objects.filter(archived_at__isnull=True).order_by("owner_id", "name")
    for portfolio in portfolios:
        try:
            _snapshot, was_created = record_portfolio_snapshot(portfolio)
        except PortfolioValuationError as exc:
            failures.append(f"{portfolio.name}: {exc}")
            continue
        if was_created:
            created += 1
        else:
            unchanged += 1
    return PortfolioSnapshotBatch(
        created=created,
        unchanged=unchanged,
        failures=tuple(failures),
    )


def _corporate_action_suspected(position: ValuedHolding) -> bool:
    market_data = position.market_data
    if market_data is None:
        return False
    previous = (
        PortfolioSnapshotHolding.objects.filter(
            snapshot__portfolio=position.holding.portfolio,
            listing=position.holding.listing,
        )
        .order_by("-snapshot__as_of_date", "-snapshot__recorded_at", "-snapshot_id")
        .first()
    )
    if previous is None or previous.quantity != position.holding.quantity:
        return False
    ratio = market_data.close / previous.price
    return ratio < SPLIT_WARNING_LOW_RATIO or ratio > SPLIT_WARNING_HIGH_RATIO

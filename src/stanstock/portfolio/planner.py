from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from uuid import UUID

from django.db import transaction
from django.db.models import F
from django.utils import timezone

from stanstock.data.etfs import (
    INVESTABLE_US_ETF_MIC,
    INVESTABLE_US_ETF_SYMBOL,
    is_supported_investable_etf,
)
from stanstock.data.models import DataAsset, LatestMarketData, Listing, Region, Security
from stanstock.portfolio.models import (
    Portfolio,
    PortfolioDeposit,
    PortfolioHolding,
    PortfolioPlanExecution,
    PortfolioPurchase,
    PortfolioSnapshot,
)
from stanstock.portfolio.service import (
    MAX_PORTFOLIO_PRICE_AGE_DAYS,
    MONEY_QUANTUM,
    QUANTITY_QUANTUM,
    PortfolioValuation,
    calculate_portfolio_valuation,
    record_portfolio_snapshot,
)
from stanstock.research.affordability import latest_price_band
from stanstock.research.eligibility import STOCK_RESEARCH_SECURITY_TYPES
from stanstock.research.models import AnalysisRun, StockAnalysis
from stanstock.research.opportunities import OpportunityAssessment, assess_opportunity
from stanstock.research.provenance import (
    DATA_MODE_PROVIDER,
    latest_provider_backed_analysis_run,
    source_assets,
    source_data_mode,
)

MONTHLY_PLAN_POLICY_VERSION = "monthly-allocation-v1"
DEFAULT_MONTHLY_CONTRIBUTION = Decimal("600.000000")
SPY_TARGET_WEIGHT = Decimal("0.70")
SATELLITE_TARGET_WEIGHT = Decimal("0.30")


class PortfolioPlanningError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PlannedPurchase:
    listing: Listing
    role: str
    price: Decimal
    quantity: Decimal
    amount: Decimal
    source_asset_id: UUID
    source_asset_sha256: str
    source_session_date: date
    analysis_id: int | None
    analysis_run_id: UUID | None
    qualification: dict[str, object] | None
    rationale: str


@dataclass(frozen=True, slots=True)
class ContributionPlan:
    portfolio: Portfolio
    policy_version: str
    as_of_date: date | None
    nav: Decimal | None
    cash_available: Decimal
    spy_value: Decimal | None
    other_invested_value: Decimal | None
    spy_budget: Decimal
    satellite_budget: Decimal
    purchases: tuple[PlannedPurchase, ...]
    residual_cash: Decimal
    fractional_shares: bool
    satellite_reason: str
    issues: tuple[str, ...]
    calculation: dict[str, object]
    plan_hash: str

    @property
    def executable(self) -> bool:
        return not self.issues and bool(self.purchases)


@dataclass(frozen=True, slots=True)
class ContributionPerformance:
    active: bool
    contribution_count: int
    total_contributions: Decimal
    tracked_contributions: Decimal
    tracking_start_value: Decimal | None
    tracking_boundary_at: datetime | None
    tracking_boundary_reason: str
    capital_base: Decimal | None
    investment_profit_loss: Decimal | None
    return_pct: Decimal | None
    withheld_reason: str


def record_external_deposit(
    *,
    portfolio: Portfolio,
    amount: Decimal,
    idempotency_key: UUID,
    note: str = "",
) -> tuple[PortfolioDeposit, bool]:
    normalized_amount = amount.quantize(MONEY_QUANTUM, rounding=ROUND_DOWN)
    normalized_note = note.strip()
    if normalized_amount <= 0:
        raise PortfolioPlanningError("Deposit amount must be positive.")

    with transaction.atomic():
        locked = Portfolio.objects.select_for_update().get(pk=portfolio.pk)
        _require_mutable_manual_portfolio(locked)
        existing = (
            PortfolioDeposit.objects.select_for_update()
            .filter(
                portfolio=locked,
                idempotency_key=idempotency_key,
            )
            .first()
        )
        if existing is not None:
            if (
                existing.amount != normalized_amount
                or existing.currency != locked.base_currency
                or existing.note != normalized_note
            ):
                raise PortfolioPlanningError(
                    "Deposit idempotency key was already used for different inputs."
                )
            return existing, False

        listing_ids = locked.holdings.values_list("listing_id", flat=True)
        _lock_current_market_rows(listing_ids)
        valuation = calculate_portfolio_valuation(locked)
        boundary_snapshot = None
        boundary_issue = _contribution_valuation_issue(
            valuation,
            today=timezone.localdate(),
        )
        if not boundary_issue:
            boundary_snapshot, _created = record_portfolio_snapshot(locked)
        else:
            boundary_issue = boundary_issue[:240]

        starting_cash = locked.cash_balance
        new_cash_balance = (starting_cash + normalized_amount).quantize(
            MONEY_QUANTUM,
            rounding=ROUND_DOWN,
        )
        updated = Portfolio.objects.filter(
            pk=locked.pk,
            cash_balance=starting_cash,
        ).update(
            cash_balance=F("cash_balance") + normalized_amount,
            updated_at=timezone.now(),
        )
        if updated != 1:
            raise PortfolioPlanningError(
                "Portfolio cash changed while recording the deposit; retry the request."
            )
        locked.cash_balance = new_cash_balance
        deposit = PortfolioDeposit.objects.create(
            portfolio=locked,
            amount=normalized_amount,
            currency=locked.base_currency,
            occurred_at=timezone.now(),
            idempotency_key=idempotency_key,
            boundary_snapshot=boundary_snapshot,
            boundary_issue=boundary_issue,
            cash_balance_after=new_cash_balance,
            note=normalized_note,
        )
    return deposit, True


def preview_monthly_contribution_plan(portfolio: Portfolio) -> ContributionPlan:
    valuation = calculate_portfolio_valuation(portfolio)
    cash = portfolio.cash_balance.quantize(MONEY_QUANTUM, rounding=ROUND_DOWN)
    issues = list(valuation.issues)
    if portfolio.archived_at is not None:
        issues.append("Archived portfolios cannot create allocation plans.")
    if portfolio.is_model_portfolio:
        issues.append("Frozen sample portfolios cannot receive contribution plans.")
    if portfolio.base_currency != Portfolio.Currency.USD:
        issues.append("The monthly SPY allocation planner currently supports USD portfolios only.")
    if valuation.corporate_action_warnings:
        issues.append("Resolve possible split events before planning new purchases.")

    spy_listing, spy_market, spy_issue = _investable_spy_market()
    if spy_issue:
        issues.append(spy_issue)
    today = timezone.localdate()
    if spy_market is not None:
        price_issue = _market_price_issue(
            ticker=INVESTABLE_US_ETF_SYMBOL,
            market_data=spy_market,
            today=today,
        )
        if price_issue:
            issues.append(price_issue)
        basis_issue = _price_basis_issue(
            ticker=INVESTABLE_US_ETF_SYMBOL,
            market_data=spy_market,
        )
        if basis_issue:
            issues.append(basis_issue)
        for position in valuation.positions:
            if (
                position.market_data is not None
                and position.market_data.session_date != spy_market.session_date
            ):
                issues.append(
                    "Portfolio holdings and SPY do not share one market session; "
                    "allocation is withheld."
                )
                break
            if position.market_data is not None:
                holding_basis_issue = _price_basis_issue(
                    ticker=position.holding.listing.ticker,
                    market_data=position.market_data,
                )
                if holding_basis_issue:
                    issues.append(holding_basis_issue)

    nav = valuation.total_value
    spy_value: Decimal | None = None
    other_value: Decimal | None = None
    if nav is not None and spy_listing is not None:
        spy_value = sum(
            (
                position.market_value or Decimal(0)
                for position in valuation.positions
                if position.holding.listing_id == spy_listing.pk
            ),
            Decimal(0),
        )
        other_value = (valuation.securities_value or Decimal(0)) - spy_value

    spy_budget = Decimal(0)
    satellite_budget = Decimal(0)
    purchases: list[PlannedPurchase] = []
    satellite_reason = ""
    if not issues and nav is not None and spy_value is not None and other_value is not None:
        spy_shortfall = max(Decimal(0), SPY_TARGET_WEIGHT * nav - spy_value)
        spy_budget = min(cash, spy_shortfall).quantize(
            MONEY_QUANTUM,
            rounding=ROUND_DOWN,
        )
        if spy_listing is None or spy_market is None:
            raise AssertionError("Validated SPY state is unavailable")
        spy_purchase = _planned_purchase(
            listing=spy_listing,
            market_data=spy_market,
            role=PortfolioPurchase.Role.CORE,
            budget=spy_budget,
            fractional_shares=portfolio.allow_fractional_shares,
            analysis=None,
            qualification=None,
            rationale="Top up SPY toward 70% of total portfolio NAV without selling.",
        )
        if spy_purchase is not None:
            purchases.append(spy_purchase)

        remaining_cash = cash - sum(
            (purchase.amount for purchase in purchases),
            Decimal(0),
        )
        satellite_shortfall = max(
            Decimal(0),
            SATELLITE_TARGET_WEIGHT * nav - other_value,
        )
        satellite_budget = min(remaining_cash, satellite_shortfall).quantize(
            MONEY_QUANTUM,
            rounding=ROUND_DOWN,
        )
        if satellite_budget > 0:
            candidate, satellite_reason = _qualified_satellite(
                as_of_date=spy_market.session_date,
                today=today,
            )
            if candidate is not None:
                analysis, market_data, qualification, rationale = candidate
                satellite_purchase = _planned_purchase(
                    listing=analysis.listing,
                    market_data=market_data,
                    role=PortfolioPurchase.Role.SATELLITE,
                    budget=satellite_budget,
                    fractional_shares=portfolio.allow_fractional_shares,
                    analysis=analysis,
                    qualification=qualification,
                    rationale=rationale,
                )
                if satellite_purchase is not None:
                    purchases.append(satellite_purchase)
                elif not satellite_reason:
                    satellite_reason = (
                        f"{analysis.listing.ticker} cannot be purchased within the "
                        "current rounding mode and budget."
                    )
        elif satellite_shortfall <= 0:
            satellite_reason = "The non-SPY allocation is already at or above its 30% target."

    spent = sum((purchase.amount for purchase in purchases), Decimal(0))
    residual_cash = (cash - spent).quantize(MONEY_QUANTUM, rounding=ROUND_DOWN)
    if residual_cash < 0:
        issues.append("Planned purchases exceed available cash.")
        purchases = []
        spent = Decimal(0)
        residual_cash = cash

    calculation = _plan_calculation(
        portfolio=portfolio,
        valuation=valuation,
        spy_listing=spy_listing,
        spy_market=spy_market,
        spy_value=spy_value,
        other_value=other_value,
        spy_budget=spy_budget,
        satellite_budget=satellite_budget,
        purchases=purchases,
        residual_cash=residual_cash,
        satellite_reason=satellite_reason,
        issues=issues,
    )
    plan_hash = hashlib.sha256(
        json.dumps(calculation, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ContributionPlan(
        portfolio=portfolio,
        policy_version=MONTHLY_PLAN_POLICY_VERSION,
        as_of_date=spy_market.session_date if spy_market is not None else None,
        nav=nav,
        cash_available=cash,
        spy_value=spy_value,
        other_invested_value=other_value,
        spy_budget=spy_budget,
        satellite_budget=satellite_budget,
        purchases=tuple(purchases),
        residual_cash=residual_cash,
        fractional_shares=portfolio.allow_fractional_shares,
        satellite_reason=satellite_reason,
        issues=tuple(dict.fromkeys(issues)),
        calculation=calculation,
        plan_hash=plan_hash,
    )


def confirm_monthly_contribution_plan(
    *,
    portfolio: Portfolio,
    expected_plan_hash: str,
    idempotency_key: UUID,
    executed_at: datetime | None = None,
) -> tuple[PortfolioPlanExecution, bool]:
    if len(expected_plan_hash) != 64:
        raise PortfolioPlanningError("Allocation plan hash is invalid.")

    with transaction.atomic():
        locked = Portfolio.objects.select_for_update().get(pk=portfolio.pk)
        _require_mutable_manual_portfolio(locked)
        existing = (
            PortfolioPlanExecution.objects.select_for_update()
            .filter(
                portfolio=locked,
                idempotency_key=idempotency_key,
            )
            .first()
        )
        if existing is not None:
            if existing.plan_hash != expected_plan_hash:
                raise PortfolioPlanningError(
                    "Plan execution idempotency key was already used for another plan."
                )
            return existing, False

        locked_holdings = list(
            PortfolioHolding.objects.select_for_update(of=("self",))
            .filter(portfolio=locked)
            .order_by("pk")
        )
        initial_plan = preview_monthly_contribution_plan(locked)
        analysis_ids = [
            purchase.analysis_id
            for purchase in initial_plan.purchases
            if purchase.analysis_id is not None
        ]
        analysis_run_ids = [
            purchase.analysis_run_id
            for purchase in initial_plan.purchases
            if purchase.analysis_run_id is not None
        ]
        list(
            AnalysisRun.objects.select_for_update(of=("self",))
            .filter(pk__in=analysis_run_ids)
            .order_by("pk")
        )
        list(
            StockAnalysis.objects.select_for_update(of=("self",))
            .filter(pk__in=analysis_ids)
            .order_by("pk")
        )
        listing_ids = {holding.listing_id for holding in locked_holdings} | {
            purchase.listing.pk for purchase in initial_plan.purchases
        }
        _lock_current_market_rows(listing_ids)
        plan = preview_monthly_contribution_plan(locked)
        if plan.plan_hash != expected_plan_hash:
            raise PortfolioPlanningError(
                "The portfolio, prices, analysis, or settings changed; review a new plan."
            )
        if plan.issues:
            raise PortfolioPlanningError(" ".join(plan.issues))
        if not plan.purchases:
            raise PortfolioPlanningError("The current plan has no purchases to confirm.")

        total_spend = sum(
            (purchase.amount for purchase in plan.purchases),
            Decimal(0),
        ).quantize(MONEY_QUANTUM, rounding=ROUND_DOWN)
        if total_spend > locked.cash_balance:
            raise PortfolioPlanningError("Confirmed purchases exceed available cash.")
        ending_cash = (locked.cash_balance - total_spend).quantize(
            MONEY_QUANTUM,
            rounding=ROUND_DOWN,
        )
        if ending_cash < 0:
            raise PortfolioPlanningError("Confirmed purchases would overdraw portfolio cash.")

        reserved = Portfolio.objects.filter(
            pk=locked.pk,
            cash_balance=locked.cash_balance,
            cash_balance__gte=total_spend,
        ).update(
            cash_balance=ending_cash,
            updated_at=timezone.now(),
        )
        if reserved != 1:
            raise PortfolioPlanningError(
                "Portfolio cash changed during confirmation; review a new plan."
            )

        execution = PortfolioPlanExecution.objects.create(
            portfolio=locked,
            idempotency_key=idempotency_key,
            plan_hash=plan.plan_hash,
            policy_version=plan.policy_version,
            executed_at=executed_at or timezone.now(),
            starting_nav=plan.nav or Decimal(0),
            starting_cash=locked.cash_balance,
            ending_cash=ending_cash,
            fractional_shares=plan.fractional_shares,
            metadata=plan.calculation,
        )
        for purchase in plan.purchases:
            market_data = LatestMarketData.objects.get(listing=purchase.listing)
            if (
                market_data.source_asset_id != purchase.source_asset_id
                or market_data.session_date != purchase.source_session_date
                or market_data.close != purchase.price
            ):
                raise PortfolioPlanningError(
                    f"{purchase.listing.ticker} market evidence changed; review a new plan."
                )
            holding = (
                PortfolioHolding.objects.select_for_update(of=("self",))
                .filter(
                    portfolio=locked,
                    listing=purchase.listing,
                )
                .order_by("pk")
                .first()
            )
            if holding is None:
                PortfolioHolding.objects.create(
                    portfolio=locked,
                    listing=purchase.listing,
                    quantity=purchase.quantity,
                    average_cost=purchase.price,
                    acquired_on=purchase.source_session_date,
                    notes=f"{plan.policy_version} {purchase.role} purchase",
                )
            else:
                combined_quantity = holding.quantity + purchase.quantity
                combined_cost = holding.quantity * holding.average_cost + purchase.amount
                holding.quantity = combined_quantity.quantize(
                    QUANTITY_QUANTUM,
                    rounding=ROUND_DOWN,
                )
                holding.average_cost = (combined_cost / combined_quantity).quantize(
                    MONEY_QUANTUM,
                    rounding=ROUND_HALF_UP,
                )
                holding.save(
                    update_fields=[
                        "quantity",
                        "average_cost",
                        "updated_at",
                    ]
                )
            PortfolioPurchase.objects.create(
                execution=execution,
                listing=purchase.listing,
                source_asset_id=purchase.source_asset_id,
                source_session_date=purchase.source_session_date,
                role=purchase.role,
                quantity=purchase.quantity,
                price=purchase.price,
                amount=purchase.amount,
            )
    return execution, True


def calculate_contribution_performance(
    portfolio: Portfolio,
    *,
    valuation: PortfolioValuation | None = None,
) -> ContributionPerformance:
    deposits = list(
        portfolio.deposits.select_related("boundary_snapshot")
        .prefetch_related(
            "boundary_snapshot__positions__listing",
            "boundary_snapshot__positions__source_asset",
        )
        .order_by(
            "occurred_at",
            "recorded_at",
            "id",
        )
    )
    total_contributions = sum(
        (deposit.amount for deposit in deposits),
        Decimal(0),
    ).quantize(MONEY_QUANTUM)
    if not deposits:
        return ContributionPerformance(
            active=False,
            contribution_count=0,
            total_contributions=Decimal(0),
            tracked_contributions=Decimal(0),
            tracking_start_value=None,
            tracking_boundary_at=None,
            tracking_boundary_reason="",
            capital_base=None,
            investment_profit_loss=None,
            return_pct=None,
            withheld_reason="",
        )

    if any(deposit.currency != portfolio.base_currency for deposit in deposits):
        return _withheld_performance(
            deposits=deposits,
            total_contributions=total_contributions,
            tracked_contributions=total_contributions,
            tracking_boundary=None,
            tracking_boundary_at=None,
            tracking_boundary_reason="",
            reason="Deposit currency no longer matches the portfolio base currency.",
        )

    latest_baseline = (
        portfolio.performance_baselines.select_related("snapshot")
        .prefetch_related(
            "snapshot__positions__listing",
            "snapshot__positions__source_asset",
        )
        .order_by("-recorded_at", "-id")
        .first()
    )
    if latest_baseline is None:
        tracking_boundary = deposits[0].boundary_snapshot
        tracking_boundary_at = (
            tracking_boundary.recorded_at if tracking_boundary is not None else None
        )
        tracking_boundary_reason = "First external deposit pre-flow boundary"
        tracked_deposits = deposits
    else:
        tracking_boundary = latest_baseline.snapshot
        tracking_boundary_at = latest_baseline.recorded_at
        tracking_boundary_reason = latest_baseline.get_reason_display()
        tracked_deposits = [
            deposit for deposit in deposits if deposit.recorded_at > tracking_boundary_at
        ]
        if latest_baseline.portfolio_id != portfolio.pk:
            return _withheld_performance(
                deposits=deposits,
                total_contributions=total_contributions,
                tracked_contributions=Decimal(0),
                tracking_boundary=tracking_boundary,
                tracking_boundary_at=tracking_boundary_at,
                tracking_boundary_reason=tracking_boundary_reason,
                reason="The latest performance baseline belongs to another portfolio.",
            )

    tracked_contributions = sum(
        (deposit.amount for deposit in tracked_deposits),
        Decimal(0),
    ).quantize(MONEY_QUANTUM)
    if tracking_boundary is None or tracking_boundary_at is None:
        missing_boundary_reason = (
            "The latest manual performance baseline has no eligible valuation: "
            f"{latest_baseline.boundary_issue}"
            if latest_baseline is not None
            else "A deposit lacks a complete pre-flow valuation boundary."
        )
        return _withheld_performance(
            deposits=deposits,
            total_contributions=total_contributions,
            tracked_contributions=tracked_contributions,
            tracking_boundary=tracking_boundary,
            tracking_boundary_at=tracking_boundary_at,
            tracking_boundary_reason=tracking_boundary_reason,
            reason=missing_boundary_reason,
        )
    boundary_issue = _contribution_snapshot_issue(tracking_boundary)
    if boundary_issue:
        return _withheld_performance(
            deposits=deposits,
            total_contributions=total_contributions,
            tracked_contributions=tracked_contributions,
            tracking_boundary=tracking_boundary,
            tracking_boundary_at=tracking_boundary_at,
            tracking_boundary_reason=tracking_boundary_reason,
            reason=f"The tracking valuation boundary is ineligible: {boundary_issue}",
        )
    if any(deposit.boundary_snapshot_id is None for deposit in tracked_deposits):
        return _withheld_performance(
            deposits=deposits,
            total_contributions=total_contributions,
            tracked_contributions=tracked_contributions,
            tracking_boundary=tracking_boundary,
            tracking_boundary_at=tracking_boundary_at,
            tracking_boundary_reason=tracking_boundary_reason,
            reason="A deposit lacks a complete pre-flow valuation boundary.",
        )
    for deposit in tracked_deposits:
        boundary = deposit.boundary_snapshot
        if boundary is None:
            raise AssertionError("Validated contribution boundary is unavailable")
        boundary_issue = _contribution_snapshot_issue(boundary)
        if boundary_issue:
            return _withheld_performance(
                deposits=deposits,
                total_contributions=total_contributions,
                tracked_contributions=tracked_contributions,
                tracking_boundary=tracking_boundary,
                tracking_boundary_at=tracking_boundary_at,
                tracking_boundary_reason=tracking_boundary_reason,
                reason=f"A deposit valuation boundary is ineligible: {boundary_issue}",
            )

    purchases = list(
        PortfolioPurchase.objects.filter(
            execution__portfolio=portfolio,
            execution__recorded_at__gt=tracking_boundary_at,
        ).select_related("listing")
    )
    expected_cash = (
        tracking_boundary.cash_balance
        + tracked_contributions
        - sum((purchase.amount for purchase in purchases), Decimal(0))
    ).quantize(MONEY_QUANTUM)
    if expected_cash != portfolio.cash_balance:
        return _withheld_performance(
            deposits=deposits,
            total_contributions=total_contributions,
            tracked_contributions=tracked_contributions,
            tracking_boundary=tracking_boundary,
            tracking_boundary_at=tracking_boundary_at,
            tracking_boundary_reason=tracking_boundary_reason,
            reason=(
                "Current cash does not reconcile to immutable deposits and "
                "confirmed planner purchases."
            ),
        )

    expected_quantities = {
        listing_id: quantity
        for listing_id, quantity in tracking_boundary.positions.values_list(
            "listing_id",
            "quantity",
        )
    }
    for purchase in purchases:
        expected_quantities[purchase.listing_id] = (
            expected_quantities.get(purchase.listing_id, Decimal(0)) + purchase.quantity
        ).quantize(QUANTITY_QUANTUM)
    current_quantities = {
        listing_id: quantity
        for listing_id, quantity in portfolio.holdings.values_list(
            "listing_id",
            "quantity",
        )
    }
    if expected_quantities != current_quantities:
        return _withheld_performance(
            deposits=deposits,
            total_contributions=total_contributions,
            tracked_contributions=tracked_contributions,
            tracking_boundary=tracking_boundary,
            tracking_boundary_at=tracking_boundary_at,
            tracking_boundary_reason=tracking_boundary_reason,
            reason=(
                "Current holdings do not reconcile to the starting boundary and "
                "confirmed planner purchases."
            ),
        )

    current_valuation = valuation or calculate_portfolio_valuation(portfolio)
    current_issue = _contribution_valuation_issue(
        current_valuation,
        today=timezone.localdate(),
    )
    if current_issue:
        return _withheld_performance(
            deposits=deposits,
            total_contributions=total_contributions,
            tracked_contributions=tracked_contributions,
            tracking_boundary=tracking_boundary,
            tracking_boundary_at=tracking_boundary_at,
            tracking_boundary_reason=tracking_boundary_reason,
            reason=f"Current portfolio valuation is ineligible: {current_issue}",
        )
    if current_valuation.total_value is None:
        raise AssertionError("Validated current portfolio valuation is unavailable")

    tracking_start_value = tracking_boundary.total_value
    capital_base = (tracking_start_value + tracked_contributions).quantize(MONEY_QUANTUM)
    profit_loss = (current_valuation.total_value - capital_base).quantize(MONEY_QUANTUM)
    return_pct = (
        (profit_loss / capital_base).quantize(Decimal("0.00000001")) if capital_base > 0 else None
    )
    return ContributionPerformance(
        active=True,
        contribution_count=len(deposits),
        total_contributions=total_contributions,
        tracked_contributions=tracked_contributions,
        tracking_start_value=tracking_start_value,
        tracking_boundary_at=tracking_boundary_at,
        tracking_boundary_reason=tracking_boundary_reason,
        capital_base=capital_base,
        investment_profit_loss=profit_loss,
        return_pct=return_pct,
        withheld_reason="",
    )


def _investable_spy_market() -> tuple[Listing | None, LatestMarketData | None, str]:
    candidates = list(
        Listing.objects.select_related(
            "security__company",
            "latest_market_data__source_asset",
        ).filter(
            security__security_type=Security.SecurityType.ETF,
            ticker=INVESTABLE_US_ETF_SYMBOL,
            provider_symbol=INVESTABLE_US_ETF_SYMBOL,
            exchange_mic=INVESTABLE_US_ETF_MIC,
            currency=Portfolio.Currency.USD,
            region=Region.US,
            is_primary=True,
            is_active=True,
        )[:2]
    )
    if len(candidates) != 1 or not is_supported_investable_etf(candidates[0]):
        return None, None, "Exactly one supported SPY ETF listing is required."
    listing = candidates[0]
    try:
        return listing, listing.latest_market_data, ""
    except LatestMarketData.DoesNotExist:
        return listing, None, "SPY has no current persisted market price."


def _qualified_satellite(
    *,
    as_of_date: date,
    today: date,
) -> tuple[
    tuple[StockAnalysis, LatestMarketData, dict[str, object], str] | None,
    str,
]:
    run = latest_provider_backed_analysis_run()
    if run is None:
        return None, "No provider-backed stock analysis is available for a satellite."
    if run.target_date != as_of_date:
        return (
            None,
            "The latest stock analysis and SPY price do not share one market session.",
        )
    analyses = (
        run.stocks.select_related(
            "listing__security__company",
            "listing__latest_market_data__source_asset",
        )
        .filter(
            listing__security__security_type__in=STOCK_RESEARCH_SECURITY_TYPES,
            listing__currency=Portfolio.Currency.USD,
            listing__is_active=True,
        )
        .order_by("-overall_score", "-confidence", "listing__ticker", "pk")
    )
    for analysis in analyses:
        if source_data_mode(analysis.data_quality) != DATA_MODE_PROVIDER:
            continue
        try:
            market_data = analysis.listing.latest_market_data
        except LatestMarketData.DoesNotExist:
            continue
        if market_data.session_date != as_of_date:
            continue
        if _market_price_issue(
            ticker=analysis.listing.ticker,
            market_data=market_data,
            today=today,
        ):
            continue
        if _price_basis_issue(
            ticker=analysis.listing.ticker,
            market_data=market_data,
        ):
            continue
        price_band = latest_price_band(analysis.listing)
        assessment = assess_opportunity(analysis, price_band=price_band)
        if not assessment.eligible or assessment.horizon != "short":
            continue
        evidence_assets = _analysis_evidence_assets(analysis)
        if evidence_assets is None:
            continue
        qualification = _satellite_qualification(
            analysis=analysis,
            evidence_assets=evidence_assets,
            assessment=assessment,
        )
        return (
            analysis,
            market_data,
            qualification,
            (
                f"{assessment.label} from {run.target_date.isoformat()} using "
                f"{assessment.policy_version}; short-horizon satellite only."
            ),
        ), ""
    return (
        None,
        "No currently qualified short-horizon stock satellite is available; cash is carried.",
    )


def _analysis_evidence_assets(
    analysis: StockAnalysis,
) -> tuple[dict[str, object], ...] | None:
    references = source_assets(analysis.data_quality)
    if not references:
        return None
    resolved: list[dict[str, object]] = []
    has_listing_price = False
    expected_price_subjects = {
        analysis.listing.ticker,
        analysis.listing.provider_symbol,
    }
    for reference in references:
        try:
            asset_id = UUID(str(reference["id"]))
        except (KeyError, TypeError, ValueError):
            return None
        try:
            asset = DataAsset.objects.get(pk=asset_id)
        except DataAsset.DoesNotExist:
            return None
        for key in ("provider", "kind", "subject"):
            expected = reference.get(key)
            if expected is not None and str(expected) != str(getattr(asset, key)):
                return None
        if asset.provider.startswith("synthetic"):
            return None
        if asset.kind == "price_history" and asset.subject in expected_price_subjects:
            has_listing_price = True
        resolved.append(
            {
                "id": str(asset.pk),
                "provider": asset.provider,
                "kind": asset.kind,
                "subject": asset.subject,
                "sha256": asset.sha256,
                "available_at": asset.available_at.isoformat(),
                "period_start": asset.period_start.isoformat() if asset.period_start else None,
                "period_end": asset.period_end.isoformat() if asset.period_end else None,
                "schema_version": asset.schema_version,
            }
        )
    if not has_listing_price:
        return None
    return tuple(sorted(resolved, key=lambda item: str(item["id"])))


def _satellite_qualification(
    *,
    analysis: StockAnalysis,
    evidence_assets: tuple[dict[str, object], ...],
    assessment: OpportunityAssessment,
) -> dict[str, object]:
    run = analysis.run
    price_band = assessment.price_band
    data_quality = analysis.data_quality if isinstance(analysis.data_quality, dict) else {}
    return {
        "schema_version": 1,
        "analysis_id": analysis.pk,
        "analysis_run_id": str(run.pk),
        "target_date": run.target_date.isoformat(),
        "generated_at": run.generated_at.isoformat(),
        "data_cutoff": run.data_cutoff.isoformat(),
        "issued_on_time": run.issued_on_time,
        "config_version": run.config_version,
        "config_hash": run.config_hash,
        "code_revision": run.code_revision,
        "universe_snapshot_id": str(run.universe_snapshot_id),
        "universe_grade": run.universe_snapshot.grade,
        "analysis_mode": str(data_quality.get("analysis_mode") or "full"),
        "recommendation": analysis.recommendation,
        "overall_score": str(analysis.overall_score),
        "confidence": str(analysis.confidence),
        "risk_class": analysis.risk_class,
        "component_scores": analysis.component_scores,
        "short_scenario": analysis.short_scenario,
        "opportunity": {
            "policy_version": assessment.policy_version,
            "label": assessment.label,
            "horizon": assessment.horizon,
            "criteria": assessment.criteria,
        },
        "price_band": (
            {
                "slug": price_band.slug,
                "close": str(price_band.close),
                "price_date": price_band.price_date.isoformat(),
                "date_basis": price_band.date_basis,
                "currency": price_band.currency,
            }
            if price_band is not None
            else None
        ),
        "source_assets": list(evidence_assets),
    }


def _planned_purchase(
    *,
    listing: Listing,
    market_data: LatestMarketData,
    role: str,
    budget: Decimal,
    fractional_shares: bool,
    analysis: StockAnalysis | None,
    qualification: dict[str, object] | None,
    rationale: str,
) -> PlannedPurchase | None:
    if budget <= 0 or market_data.close <= 0:
        return None
    if fractional_shares:
        quantity = (budget / market_data.close).quantize(
            QUANTITY_QUANTUM,
            rounding=ROUND_DOWN,
        )
    else:
        quantity = (budget / market_data.close).to_integral_value(rounding=ROUND_DOWN)
    if quantity <= 0:
        return None
    amount = (quantity * market_data.close).quantize(
        MONEY_QUANTUM,
        rounding=ROUND_DOWN,
    )
    if amount <= 0 or amount > budget:
        return None
    return PlannedPurchase(
        listing=listing,
        role=role,
        price=market_data.close,
        quantity=quantity,
        amount=amount,
        source_asset_id=market_data.source_asset_id,
        source_asset_sha256=market_data.source_asset.sha256,
        source_session_date=market_data.session_date,
        analysis_id=analysis.pk if analysis is not None else None,
        analysis_run_id=analysis.run_id if analysis is not None else None,
        qualification=qualification,
        rationale=rationale,
    )


def _market_price_issue(
    *,
    ticker: str,
    market_data: LatestMarketData,
    today: date,
) -> str:
    age_days = (today - market_data.session_date).days
    if age_days < 0:
        return f"{ticker} market price is dated in the future."
    if age_days > MAX_PORTFOLIO_PRICE_AGE_DAYS:
        return f"{ticker} market price is stale ({market_data.session_date.isoformat()})."
    return ""


def _price_basis_issue(
    *,
    ticker: str,
    market_data: LatestMarketData,
) -> str:
    return _asset_price_basis_issue(
        ticker=ticker,
        source_asset=market_data.source_asset,
    )


def _asset_price_basis_issue(
    *,
    ticker: str,
    source_asset: DataAsset,
) -> str:
    metadata = source_asset.metadata if isinstance(source_asset.metadata, dict) else {}
    if source_asset.provider != "twelve_data":
        return f"{ticker} does not have provider-backed Twelve Data market evidence."
    if metadata.get("return_definition") != "split_adjusted_price_return":
        return f"{ticker} does not have a compatible split-adjusted price basis."
    if metadata.get("dividends_included") is not False:
        return f"{ticker} has ambiguous dividend treatment."
    return ""


def _contribution_valuation_issue(
    valuation: PortfolioValuation,
    *,
    today: date,
) -> str:
    if valuation.issues:
        return " ".join(valuation.issues)
    if valuation.total_value is None:
        return "The portfolio total is unavailable."
    if valuation.corporate_action_warnings:
        return "A possible split or other corporate action requires review."
    if valuation.positions:
        if valuation.oldest_price_date is None or valuation.newest_price_date is None:
            return "A holding lacks a dated market price."
        if valuation.oldest_price_date != valuation.newest_price_date:
            return "Portfolio holdings do not share one market session."
        if valuation.as_of_date != valuation.newest_price_date:
            return "The valuation date does not match its market evidence."
        for position in valuation.positions:
            market_data = position.market_data
            if market_data is None:
                return f"{position.holding.listing.ticker} lacks market evidence."
            price_issue = _market_price_issue(
                ticker=position.holding.listing.ticker,
                market_data=market_data,
                today=today,
            )
            if price_issue:
                return price_issue
            basis_issue = _price_basis_issue(
                ticker=position.holding.listing.ticker,
                market_data=market_data,
            )
            if basis_issue:
                return basis_issue
    if valuation.warnings:
        return " ".join(valuation.warnings)
    return ""


def _contribution_snapshot_issue(snapshot: PortfolioSnapshot) -> str:
    positions = list(snapshot.positions.all())
    if not positions:
        if snapshot.oldest_price_date is not None or snapshot.newest_price_date is not None:
            return "A cash-only boundary contains unexpected market dates."
        return ""
    if snapshot.corporate_action_warnings or any(
        position.corporate_action_suspected for position in positions
    ):
        return "A possible split or other corporate action affected the boundary."
    if snapshot.return_definition != "split_adjusted_price_return":
        return "The boundary does not use one split-adjusted price-return basis."
    if snapshot.dividends_included:
        return "The boundary has incompatible dividend treatment."
    if snapshot.oldest_price_date is None or snapshot.newest_price_date is None:
        return "The boundary lacks complete market dates."
    if snapshot.oldest_price_date != snapshot.newest_price_date:
        return "Boundary holdings do not share one market session."
    if snapshot.as_of_date != snapshot.newest_price_date:
        return "The boundary date does not match its market evidence."

    recorded_date = timezone.localtime(snapshot.recorded_at).date()
    for position in positions:
        if position.source_session_date != snapshot.as_of_date:
            return f"{position.listing.ticker} does not match the boundary market session."
        age_days = (recorded_date - position.source_session_date).days
        if age_days < 0:
            return f"{position.listing.ticker} boundary evidence is dated in the future."
        if age_days > MAX_PORTFOLIO_PRICE_AGE_DAYS:
            return f"{position.listing.ticker} boundary evidence was stale when recorded."
        if position.source_asset.available_at > snapshot.recorded_at:
            return f"{position.listing.ticker} boundary evidence was not yet available."
        basis_issue = _asset_price_basis_issue(
            ticker=position.listing.ticker,
            source_asset=position.source_asset,
        )
        if basis_issue:
            return basis_issue
    return ""


def _money_text(value: Decimal | int) -> str:
    return str(Decimal(value).quantize(MONEY_QUANTUM, rounding=ROUND_DOWN))


def _quantity_text(value: Decimal) -> str:
    return str(value.quantize(QUANTITY_QUANTUM, rounding=ROUND_DOWN))


def _plan_calculation(
    *,
    portfolio: Portfolio,
    valuation: PortfolioValuation,
    spy_listing: Listing | None,
    spy_market: LatestMarketData | None,
    spy_value: Decimal | None,
    other_value: Decimal | None,
    spy_budget: Decimal,
    satellite_budget: Decimal,
    purchases: list[PlannedPurchase],
    residual_cash: Decimal,
    satellite_reason: str,
    issues: list[str],
) -> dict[str, object]:
    holdings = [
        {
            "listing_id": str(position.holding.listing_id),
            "quantity": _quantity_text(position.holding.quantity),
            "market_value": (
                _money_text(position.market_value) if position.market_value is not None else None
            ),
            "source_asset_id": (
                str(position.market_data.source_asset_id)
                if position.market_data is not None
                else None
            ),
            "source_asset_sha256": (
                position.market_data.source_asset.sha256
                if position.market_data is not None
                else None
            ),
            "session_date": (
                position.market_data.session_date.isoformat()
                if position.market_data is not None
                else None
            ),
        }
        for position in valuation.positions
    ]
    return {
        "schema_version": 2,
        "policy_version": MONTHLY_PLAN_POLICY_VERSION,
        "portfolio_id": str(portfolio.pk),
        "base_currency": portfolio.base_currency,
        "cash_available": _money_text(portfolio.cash_balance),
        "monthly_contribution": _money_text(portfolio.monthly_contribution),
        "nav": (_money_text(valuation.total_value) if valuation.total_value is not None else None),
        "spy_target_weight": str(SPY_TARGET_WEIGHT),
        "satellite_target_weight": str(SATELLITE_TARGET_WEIGHT),
        "fractional_shares": portfolio.allow_fractional_shares,
        "spy_listing_id": str(spy_listing.pk) if spy_listing is not None else None,
        "spy_value": _money_text(spy_value) if spy_value is not None else None,
        "other_invested_value": (_money_text(other_value) if other_value is not None else None),
        "spy_source_asset_id": (
            str(spy_market.source_asset_id) if spy_market is not None else None
        ),
        "spy_source_asset_sha256": (
            spy_market.source_asset.sha256 if spy_market is not None else None
        ),
        "spy_session_date": (
            spy_market.session_date.isoformat() if spy_market is not None else None
        ),
        "spy_budget": _money_text(spy_budget),
        "satellite_budget": _money_text(satellite_budget),
        "holdings": holdings,
        "purchases": [
            {
                "listing_id": str(purchase.listing.pk),
                "ticker": purchase.listing.ticker,
                "role": purchase.role,
                "price": _money_text(purchase.price),
                "quantity": _quantity_text(purchase.quantity),
                "amount": _money_text(purchase.amount),
                "source_asset_id": str(purchase.source_asset_id),
                "source_asset_sha256": purchase.source_asset_sha256,
                "source_session_date": purchase.source_session_date.isoformat(),
                "analysis_id": purchase.analysis_id,
                "analysis_run_id": (
                    str(purchase.analysis_run_id) if purchase.analysis_run_id is not None else None
                ),
                "qualification": purchase.qualification,
                "rationale": purchase.rationale,
            }
            for purchase in purchases
        ],
        "residual_cash": _money_text(residual_cash),
        "satellite_reason": satellite_reason,
        "issues": list(dict.fromkeys(issues)),
    }


def _withheld_performance(
    *,
    deposits: list[PortfolioDeposit],
    total_contributions: Decimal,
    tracked_contributions: Decimal,
    tracking_boundary: PortfolioSnapshot | None,
    tracking_boundary_at: datetime | None,
    tracking_boundary_reason: str,
    reason: str,
) -> ContributionPerformance:
    return ContributionPerformance(
        active=True,
        contribution_count=len(deposits),
        total_contributions=total_contributions,
        tracked_contributions=tracked_contributions,
        tracking_start_value=(
            tracking_boundary.total_value if tracking_boundary is not None else None
        ),
        tracking_boundary_at=tracking_boundary_at,
        tracking_boundary_reason=tracking_boundary_reason,
        capital_base=None,
        investment_profit_loss=None,
        return_pct=None,
        withheld_reason=reason,
    )


def _lock_current_market_rows(listing_ids: Iterable[UUID]) -> None:
    ordered_listing_ids = sorted(set(listing_ids), key=str)
    if not ordered_listing_ids:
        return
    list(
        Listing.objects.select_for_update(of=("self",), no_key=True)
        .filter(pk__in=ordered_listing_ids)
        .order_by("pk")
    )
    list(
        LatestMarketData.objects.select_for_update(of=("self",))
        .filter(listing_id__in=ordered_listing_ids)
        .order_by("listing_id")
    )


def _require_mutable_manual_portfolio(portfolio: Portfolio) -> None:
    if portfolio.archived_at is not None:
        raise PortfolioPlanningError("Archived portfolios cannot be changed.")
    if portfolio.is_model_portfolio:
        raise PortfolioPlanningError(
            "Frozen sample portfolios cannot receive deposits or planner purchases."
        )

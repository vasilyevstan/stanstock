from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_DOWN, Decimal
from uuid import UUID

from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone

from stanstock.data.etfs import is_supported_investable_etf
from stanstock.data.models import DataAsset, LatestMarketData, Listing, Security
from stanstock.portfolio.models import (
    Portfolio,
    PortfolioHolding,
    PortfolioPerformanceBaseline,
    PortfolioSnapshot,
    PortfolioSnapshotHolding,
)
from stanstock.research.affordability import (
    DECISION_TARGET_DATE_BASIS,
    PRICE_BAND_POLICY_VERSION,
    UNDER_10_BAND,
    PriceBandAssessment,
    classify_price_band,
)
from stanstock.research.config import code_revision
from stanstock.research.eligibility import STOCK_RESEARCH_SECURITY_TYPES
from stanstock.research.models import AnalysisRun, StockAnalysis
from stanstock.research.opportunities import OpportunityAssessment, assess_opportunity
from stanstock.research.provenance import (
    DATA_MODE_PROVIDER,
    analysis_run_data_mode,
    analysis_run_source_providers,
    latest_provider_backed_analysis_run,
    source_assets,
    source_data_mode,
)

MAX_PORTFOLIO_PRICE_AGE_DAYS = 7
SPLIT_WARNING_LOW_RATIO = Decimal("0.60")
SPLIT_WARNING_HIGH_RATIO = Decimal("1.67")
SAMPLE_PORTFOLIO_POLICY = "equal_weight_opportunities_v2"
SAMPLE_PORTFOLIO_DEFAULT_CAPITAL = Decimal("100000.000000")
SAMPLE_PORTFOLIO_DEFAULT_TOP_N = 5
SAMPLE_PORTFOLIO_MAX_TOP_N = 20
MONEY_QUANTUM = Decimal("0.000001")
QUANTITY_QUANTUM = Decimal("0.00000001")


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
    corporate_action_warnings: int

    @property
    def complete(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class PortfolioSnapshotBatch:
    created: int
    unchanged: int
    failures: tuple[str, ...]
    #: Exact `(portfolio_id, snapshot_id)` pairs this batch actually
    #: produced or reused, one per portfolio that did not fail. This is the
    #: minimum stable identity a caller (e.g. scheduled-refresh output
    #: verification) needs to independently re-fetch *the* snapshot this
    #: run stands behind, rather than an arbitrary latest same-date row.
    snapshot_ids: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class SamplePortfolioSelection:
    analysis: StockAnalysis
    opportunity: OpportunityAssessment
    price_band: PriceBandAssessment
    source_asset: DataAsset
    reference_price: Decimal
    quantity: Decimal
    cost_basis: Decimal


def validate_holding_listing(*, portfolio: Portfolio, listing: Listing) -> None:
    if portfolio.archived_at is not None:
        raise PortfolioValuationError("Archived portfolios cannot be changed.")
    if portfolio.is_model_portfolio:
        raise PortfolioValuationError("Model portfolio holdings are frozen.")
    if not listing.is_active:
        raise PortfolioValuationError(f"{listing.ticker} is not an active listing.")
    if (
        listing.security.security_type == Security.SecurityType.ETF
        and not is_supported_investable_etf(listing)
    ):
        raise PortfolioValuationError(
            f"{listing.ticker} is not a supported investable ETF; only SPY is enabled."
        )
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
    with transaction.atomic():
        locked_portfolio = Portfolio.objects.select_for_update().get(pk=portfolio.pk)
        validate_holding_listing(portfolio=locked_portfolio, listing=listing)
        holding = (
            PortfolioHolding.objects.select_for_update(of=("self",))
            .filter(
                portfolio=locked_portfolio,
                listing=listing,
            )
            .order_by("pk")
            .first()
        )
        quantity_changed = holding is None or holding.quantity != quantity
        if holding is None:
            holding = PortfolioHolding.objects.create(
                portfolio=locked_portfolio,
                listing=listing,
                quantity=quantity,
                average_cost=average_cost,
                acquired_on=acquired_on,
                notes=notes.strip(),
            )
        else:
            holding.quantity = quantity
            holding.average_cost = average_cost
            holding.acquired_on = acquired_on
            holding.notes = notes.strip()
            holding.save(
                update_fields=[
                    "quantity",
                    "average_cost",
                    "acquired_on",
                    "notes",
                    "updated_at",
                ]
            )
        if quantity_changed and locked_portfolio.deposits.exists():
            _record_manual_performance_baseline(
                locked_portfolio,
                note=f"Manual quantity change for {listing.ticker}.",
            )
    return holding


def delete_holding(holding: PortfolioHolding) -> None:
    with transaction.atomic():
        portfolio = Portfolio.objects.select_for_update().get(pk=holding.portfolio_id)
        if portfolio.is_model_portfolio:
            raise PortfolioValuationError("Model portfolio holdings are frozen.")
        locked_holding = (
            PortfolioHolding.objects.select_for_update(of=("self",))
            .filter(pk=holding.pk)
            .order_by("pk")
            .first()
        )
        if locked_holding is None:
            return
        ticker = locked_holding.listing.ticker
        contribution_tracking_active = portfolio.deposits.exists()
        locked_holding.delete()
        if contribution_tracking_active:
            _record_manual_performance_baseline(
                portfolio,
                note=f"Manual removal of {ticker}.",
            )


def build_sample_portfolio(
    *,
    owner: User,
    starting_capital: Decimal = SAMPLE_PORTFOLIO_DEFAULT_CAPITAL,
    top_n: int = SAMPLE_PORTFOLIO_DEFAULT_TOP_N,
    source_run: AnalysisRun | None = None,
) -> tuple[Portfolio, bool]:
    capital = starting_capital.quantize(MONEY_QUANTUM, rounding=ROUND_DOWN)
    if capital <= 0:
        raise PortfolioValuationError("Starting capital must be positive.")
    if top_n < 1 or top_n > SAMPLE_PORTFOLIO_MAX_TOP_N:
        raise PortfolioValuationError(f"Top N must be between 1 and {SAMPLE_PORTFOLIO_MAX_TOP_N}.")

    run = source_run or latest_provider_backed_analysis_run()
    if run is None:
        raise PortfolioValuationError("No completed analysis run is available.")
    if run.status != "complete":
        raise PortfolioValuationError("The source analysis run is not complete.")
    if analysis_run_data_mode(run) != DATA_MODE_PROVIDER:
        raise PortfolioValuationError(
            "A sample portfolio requires a provider-backed analysis run; "
            "synthetic research cannot seed tracked accuracy."
        )

    existing = Portfolio.objects.filter(
        owner=owner,
        source_analysis_run=run,
        archived_at__isnull=True,
    ).first()
    if existing is not None:
        return existing, False

    candidates: list[
        tuple[StockAnalysis, OpportunityAssessment, PriceBandAssessment, DataAsset]
    ] = []
    analyses = run.stocks.select_related(
        "listing__security__company",
        "listing__latest_market_data",
    ).order_by("-overall_score", "-confidence", "listing__ticker", "pk")
    for analysis in analyses:
        if analysis.listing.security.security_type not in STOCK_RESEARCH_SECURITY_TYPES:
            continue
        if analysis.listing.currency.upper() != Portfolio.Currency.USD:
            continue
        if not analysis.listing.is_active:
            continue
        construction_price_band = classify_price_band(
            close=analysis.current_price,
            price_date=run.target_date,
            date_basis=DECISION_TARGET_DATE_BASIS,
            currency=analysis.listing.currency,
        )
        opportunity = assess_opportunity(
            analysis,
            price_band=construction_price_band,
        )
        if not opportunity.eligible:
            continue
        if construction_price_band is None:
            raise PortfolioValuationError(
                f"{analysis.listing.ticker} has no valid decision-run USD price band."
            )
        if source_data_mode(analysis.data_quality) != DATA_MODE_PROVIDER:
            raise PortfolioValuationError(
                f"{analysis.listing.ticker} has incomplete provider provenance."
            )
        if not LatestMarketData.objects.filter(listing=analysis.listing).exists():
            raise PortfolioValuationError(
                f"{analysis.listing.ticker} has no current market row for ongoing tracking."
            )
        candidates.append(
            (
                analysis,
                opportunity,
                construction_price_band,
                _analysis_price_asset(analysis),
            )
        )
        if len(candidates) == top_n:
            break
    if not candidates:
        raise PortfolioValuationError(
            "The latest provider-backed run has no eligible opportunities."
        )

    allocation = capital / len(candidates)
    selections: list[SamplePortfolioSelection] = []
    invested = Decimal(0)
    for analysis, opportunity, price_band, source_asset in candidates:
        reference_price = analysis.current_price.quantize(MONEY_QUANTUM)
        if reference_price <= 0:
            raise PortfolioValuationError(
                f"{analysis.listing.ticker} has an invalid reference price."
            )
        quantity = (allocation / reference_price).quantize(
            QUANTITY_QUANTUM,
            rounding=ROUND_DOWN,
        )
        if quantity <= 0:
            raise PortfolioValuationError(
                f"Starting capital is too small to allocate {analysis.listing.ticker}."
            )
        cost_basis = (quantity * reference_price).quantize(
            MONEY_QUANTUM,
            rounding=ROUND_DOWN,
        )
        invested += cost_basis
        selections.append(
            SamplePortfolioSelection(
                analysis=analysis,
                opportunity=opportunity,
                price_band=price_band,
                source_asset=source_asset,
                reference_price=reference_price,
                quantity=quantity,
                cost_basis=cost_basis,
            )
        )

    cash_balance = (capital - invested).quantize(MONEY_QUANTUM, rounding=ROUND_DOWN)
    if cash_balance < 0:
        raise PortfolioValuationError("Sample allocation exceeded its starting capital.")

    first_opportunity = selections[0].opportunity
    metadata = {
        "entry_basis": "analysis_reference_close",
        "investability": "research_reference",
        "opportunity_policy_version": first_opportunity.policy_version,
        "price_band_policy_version": PRICE_BAND_POLICY_VERSION,
        "excluded_new_allocation_price_bands": [UNDER_10_BAND],
        "signal_label": first_opportunity.label,
        "signal_horizon": first_opportunity.horizon,
        "requested_top_n": top_n,
        "selected_count": len(selections),
        "rebalance_policy": "none",
        "source_providers": sorted(analysis_run_source_providers(run)),
        "universe_grade": run.universe_snapshot.grade,
        "issued_on_time": run.issued_on_time,
        "return_definition": "split_adjusted_price_return",
        "dividends_included": False,
        "selection_price_bands": [
            {
                "ticker": selection.analysis.listing.ticker,
                "band": selection.price_band.slug,
                "close": str(selection.price_band.close),
                "price_date": selection.price_band.price_date.isoformat(),
                "date_basis": selection.price_band.date_basis,
            }
            for selection in selections
        ],
    }
    base_name = f"StanStock Sample {run.target_date.isoformat()} {str(run.pk)[:8]}"
    description = (
        f"Frozen equal-weight research-reference basket of {len(selections)} "
        f"{first_opportunity.label.lower()} selections from the "
        f"{run.target_date.isoformat()} provider-backed analysis. Entry values use "
        "the immutable analysis reference closes, not executable fills."
    )

    with transaction.atomic():
        AnalysisRun.objects.select_for_update().get(pk=run.pk)
        existing = (
            Portfolio.objects.select_for_update()
            .filter(
                owner=owner,
                source_analysis_run=run,
                archived_at__isnull=True,
            )
            .first()
        )
        if existing is not None:
            return existing, False
        portfolio = Portfolio.objects.create(
            owner=owner,
            name=_available_sample_name(owner, base_name),
            description=description,
            base_currency=Portfolio.Currency.USD,
            cash_balance=cash_balance,
            source_analysis_run=run,
            construction_policy=SAMPLE_PORTFOLIO_POLICY,
            construction_metadata=metadata,
            starting_capital=capital,
        )
        PortfolioHolding.objects.bulk_create(
            [
                PortfolioHolding(
                    portfolio=portfolio,
                    listing=selection.analysis.listing,
                    quantity=selection.quantity,
                    average_cost=selection.reference_price,
                    acquired_on=run.target_date,
                    notes=(
                        f"Rank {rank}; {selection.opportunity.label}; "
                        f"{selection.price_band.label}; "
                        f"source analysis {selection.analysis.pk}"
                    ),
                )
                for rank, selection in enumerate(selections, start=1)
            ]
        )
        _record_sample_baseline(portfolio, run, selections)
    return portfolio, True


def calculate_portfolio_valuation(
    portfolio: Portfolio,
    *,
    expected_as_of_date: date | None = None,
    before_recorded_at: datetime | None = None,
) -> PortfolioValuation:
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
            as_of_date=expected_as_of_date or timezone.localdate(),
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
            corporate_action_warnings=0,
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

    corporate_action_warnings = sum(
        _corporate_action_suspected(position, before_recorded_at=before_recorded_at)
        for position in positions
    )
    if corporate_action_warnings:
        if portfolio.is_model_portfolio:
            warnings.append(
                "A split-sized price move was detected; model return is withheld "
                "until the quantity basis is reviewed."
            )
        else:
            warnings.append(
                "A split-sized price move was detected; quantity and average cost may need review."
            )

    if expected_as_of_date is not None and as_of_date != expected_as_of_date:
        issues.append(
            f"Portfolio prices resolve to {as_of_date.isoformat()}, not the required "
            f"XNYS session {expected_as_of_date.isoformat()}."
        )

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
        corporate_action_warnings=corporate_action_warnings,
    )


def snapshot_input_payload(
    portfolio: Portfolio,
    valuation: PortfolioValuation,
) -> dict[str, object]:
    """The exact JSON-serialisable payload `record_portfolio_snapshot` hashes.

    Extracted so both the production snapshot writer and an independent
    verifier (e.g. scheduled-refresh output verification) derive the same
    `input_hash` from the same recipe instead of maintaining two versions of
    it.
    """
    return {
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


def compute_snapshot_input_hash(
    portfolio: Portfolio,
    valuation: PortfolioValuation,
) -> str:
    payload = snapshot_input_payload(portfolio, valuation)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def record_portfolio_snapshot(
    portfolio: Portfolio,
    *,
    expected_as_of_date: date | None = None,
) -> tuple[PortfolioSnapshot, bool]:
    valuation = calculate_portfolio_valuation(
        portfolio,
        expected_as_of_date=expected_as_of_date,
    )
    if not valuation.complete:
        raise PortfolioValuationError(
            "Portfolio snapshot was not recorded: " + " ".join(valuation.issues)
        )
    assert valuation.securities_value is not None
    assert valuation.total_value is not None
    assert valuation.unrealized_gain is not None

    revision = code_revision()
    input_hash = compute_snapshot_input_hash(portfolio, valuation)
    corporate_action_flags = {
        position.holding.listing_id: _corporate_action_suspected(position)
        for position in valuation.positions
    }
    warning_count = valuation.corporate_action_warnings

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


def _record_manual_performance_baseline(
    portfolio: Portfolio,
    *,
    note: str,
) -> PortfolioPerformanceBaseline:
    snapshot = None
    boundary_issue = ""
    try:
        snapshot, _created = record_portfolio_snapshot(portfolio)
    except PortfolioValuationError as exc:
        boundary_issue = str(exc)[:240]
    return PortfolioPerformanceBaseline.objects.create(
        portfolio=portfolio,
        snapshot=snapshot,
        reason=PortfolioPerformanceBaseline.Reason.MANUAL_HOLDING_CHANGE,
        boundary_issue=boundary_issue,
        note=note,
    )


def snapshot_all_portfolios(
    *,
    expected_as_of_date: date | None = None,
) -> PortfolioSnapshotBatch:
    created = 0
    unchanged = 0
    failures: list[str] = []
    snapshot_ids: list[tuple[str, str]] = []
    portfolios = Portfolio.objects.filter(archived_at__isnull=True).order_by("owner_id", "name")
    for portfolio in portfolios:
        try:
            snapshot, was_created = record_portfolio_snapshot(
                portfolio,
                expected_as_of_date=expected_as_of_date,
            )
        except PortfolioValuationError as exc:
            failures.append(f"{portfolio.name}: {exc}")
            continue
        snapshot_ids.append((str(portfolio.pk), str(snapshot.pk)))
        if was_created:
            created += 1
        else:
            unchanged += 1
    return PortfolioSnapshotBatch(
        created=created,
        unchanged=unchanged,
        failures=tuple(failures),
        snapshot_ids=tuple(snapshot_ids),
    )


def restore_portfolio(portfolio: Portfolio) -> Portfolio:
    with transaction.atomic():
        locked = Portfolio.objects.select_for_update().get(pk=portfolio.pk)
        if locked.source_analysis_run_id is not None:
            AnalysisRun.objects.select_for_update().get(pk=locked.source_analysis_run_id)
            conflict_exists = (
                Portfolio.objects.filter(
                    owner_id=locked.owner_id,
                    source_analysis_run_id=locked.source_analysis_run_id,
                    archived_at__isnull=True,
                )
                .exclude(pk=locked.pk)
                .exists()
            )
            if conflict_exists:
                raise PortfolioValuationError(
                    "Another active sample portfolio already tracks this source run."
                )
        locked.archived_at = None
        locked.save(update_fields=["archived_at", "updated_at"])
    return locked


def _corporate_action_suspected(
    position: ValuedHolding, *, before_recorded_at: datetime | None = None
) -> bool:
    """Detect a split-sized price move against the most recent *prior*
    persisted holding for this portfolio/listing.

    ``before_recorded_at`` makes this historically reproducible: passing the
    ``recorded_at`` of the snapshot currently being reconstructed excludes
    that snapshot's own holding (and any later one) from "previous", so a
    verifier re-deriving an already-persisted snapshot's flag cannot have
    that snapshot's own (possibly fabricated) row silently confirm itself.
    Normal snapshot creation omits the boundary -- the new snapshot does not
    exist yet at that point, so behavior is unchanged. The filter is strict
    (``__lt``, never ``__lte``) so a tied ``recorded_at`` still excludes the
    row under reconstruction rather than treating it as its own history.
    """
    market_data = position.market_data
    if market_data is None:
        return False
    queryset = PortfolioSnapshotHolding.objects.filter(
        snapshot__portfolio=position.holding.portfolio,
        listing=position.holding.listing,
    )
    if before_recorded_at is not None:
        queryset = queryset.filter(snapshot__recorded_at__lt=before_recorded_at)
    previous = queryset.order_by(
        "-snapshot__as_of_date", "-snapshot__recorded_at", "-snapshot_id"
    ).first()
    if previous is None:
        return False
    if previous.corporate_action_suspected and previous.quantity == position.holding.quantity:
        return True
    if previous.quantity != position.holding.quantity:
        return False
    ratio = market_data.close / previous.price
    return ratio < SPLIT_WARNING_LOW_RATIO or ratio > SPLIT_WARNING_HIGH_RATIO


def _analysis_price_asset(analysis: StockAnalysis) -> DataAsset:
    expected_subjects = {
        analysis.listing.ticker,
        analysis.listing.provider_symbol,
    }
    for asset in source_assets(analysis.data_quality):
        if asset.get("kind") != "price_history":
            continue
        if str(asset.get("subject", "")) not in expected_subjects:
            continue
        try:
            asset_id = UUID(str(asset["id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise PortfolioValuationError(
                f"{analysis.listing.ticker} price provenance has no valid asset id."
            ) from exc
        try:
            source_asset = DataAsset.objects.get(pk=asset_id)
        except DataAsset.DoesNotExist as exc:
            raise PortfolioValuationError(
                f"{analysis.listing.ticker} source asset is unavailable."
            ) from exc
        if source_asset.provider.startswith("synthetic"):
            raise PortfolioValuationError(f"{analysis.listing.ticker} source asset is synthetic.")
        return source_asset
    raise PortfolioValuationError(f"{analysis.listing.ticker} has no matching price source asset.")


def _available_sample_name(owner: User, base_name: str) -> str:
    if not Portfolio.objects.filter(owner=owner, name=base_name).exists():
        return base_name
    suffix = 2
    while Portfolio.objects.filter(owner=owner, name=f"{base_name} ({suffix})").exists():
        suffix += 1
    return f"{base_name} ({suffix})"


def _record_sample_baseline(
    portfolio: Portfolio,
    run: AnalysisRun,
    selections: list[SamplePortfolioSelection],
) -> PortfolioSnapshot:
    payload = {
        "portfolio_id": str(portfolio.pk),
        "source_analysis_run_id": str(run.pk),
        "as_of_date": run.target_date.isoformat(),
        "base_currency": portfolio.base_currency,
        "cash_balance": str(portfolio.cash_balance),
        "positions": [
            {
                "listing_id": str(selection.analysis.listing_id),
                "quantity": str(selection.quantity),
                "reference_price": str(selection.reference_price),
                "source_asset_id": str(selection.source_asset.pk),
            }
            for selection in selections
        ],
    }
    input_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    securities_value = sum(
        (selection.cost_basis for selection in selections),
        Decimal(0),
    ).quantize(MONEY_QUANTUM)
    total_value = (portfolio.cash_balance + securities_value).quantize(MONEY_QUANTUM)
    return_definitions = {
        str(
            selection.source_asset.metadata.get("return_definition")
            or "split_adjusted_price_return"
        )
        for selection in selections
    }
    dividends_included = all(
        selection.source_asset.metadata.get("dividends_included") is True
        for selection in selections
    )
    snapshot = PortfolioSnapshot.objects.create(
        portfolio=portfolio,
        as_of_date=run.target_date,
        oldest_price_date=run.target_date,
        newest_price_date=run.target_date,
        base_currency=portfolio.base_currency,
        cash_balance=portfolio.cash_balance,
        securities_value=securities_value,
        total_value=total_value,
        cost_basis=securities_value,
        unrealized_gain=Decimal(0),
        return_pct=Decimal(0),
        input_hash=input_hash,
        code_revision=code_revision(),
        return_definition=(
            next(iter(return_definitions)) if len(return_definitions) == 1 else "mixed_price_return"
        ),
        dividends_included=dividends_included,
        corporate_action_warnings=0,
    )
    holding_listing_ids = set(portfolio.holdings.values_list("listing_id", flat=True))
    expected_listing_ids = {selection.analysis.listing_id for selection in selections}
    if holding_listing_ids != expected_listing_ids:
        raise PortfolioValuationError(
            "Sample portfolio holdings do not match the selected analyses."
        )
    PortfolioSnapshotHolding.objects.bulk_create(
        [
            PortfolioSnapshotHolding(
                snapshot=snapshot,
                listing=selection.analysis.listing,
                source_asset=selection.source_asset,
                source_session_date=run.target_date,
                quantity=selection.quantity,
                average_cost=selection.reference_price,
                price=selection.reference_price,
                cost_basis=selection.cost_basis,
                market_value=selection.cost_basis,
                unrealized_gain=Decimal(0),
                corporate_action_suspected=False,
            )
            for selection in selections
        ]
    )
    return snapshot

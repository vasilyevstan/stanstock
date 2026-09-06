from __future__ import annotations

from decimal import Decimal
from http import HTTPStatus
from typing import Any, cast
from uuid import UUID

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Avg, Count, Q, QuerySet
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from stanstock.core.models import JobRun
from stanstock.core.services import system_status
from stanstock.data.fx import DEFAULT_MAX_CARRY_DAYS
from stanstock.data.models import (
    LatestMarketData,
    Listing,
    ProviderRecord,
    Region,
    UniverseSnapshot,
)
from stanstock.portfolio.models import Portfolio, PortfolioHolding
from stanstock.portfolio.service import (
    PortfolioValuationError,
    calculate_portfolio_valuation,
    portfolio_snapshot_series,
    record_portfolio_snapshot,
    upsert_holding,
)
from stanstock.research.models import (
    AnalysisRun,
    Prediction,
    PredictionOutcome,
    Recommendation,
    RiskClass,
    StockAnalysis,
)
from stanstock.research.opportunities import assess_opportunity
from stanstock.simulation.builders import run_simulation_workflow
from stanstock.simulation.models import SimulationDefinition, SimulationRun
from stanstock.simulation.types import SimulationWorkflowError
from stanstock.web.demo import DEMO_OPPORTUNITIES
from stanstock.web.forms import (
    OpportunityFilterForm,
    PortfolioForm,
    PortfolioHoldingForm,
    SimulationForm,
)


def index(request: HttpRequest) -> HttpResponse:
    if request.user.is_authenticated:
        return redirect("status")
    return redirect("login")


def health(request: HttpRequest) -> JsonResponse:
    components = system_status()
    ok = all(bool(component["ok"]) for component in components)
    return JsonResponse(
        {"status": "ok" if ok else "degraded", "components": components},
        status=HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE,
    )


@login_required
def status_page(request: HttpRequest) -> HttpResponse:
    components = system_status()
    latest_run = _latest_analysis_run()
    persisted_opportunities = (
        list(
            StockAnalysis.objects.filter(run=latest_run)
            .select_related("listing__security__company")
            .order_by("-overall_score")[:10]
        )
        if latest_run
        else []
    )
    context = {
        "components": components,
        "system_ok": all(bool(component["ok"]) for component in components),
        "demo_mode": settings.DEMO_MODE,
        "opportunities": persisted_opportunities,
        "demo_opportunities": (
            DEMO_OPPORTUNITIES if settings.DEMO_MODE and not persisted_opportunities else []
        ),
        "latest_run": latest_run,
        "providers": ProviderRecord.objects.order_by("provider"),
        "recent_jobs": JobRun.objects.order_by("-started_at")[:5],
        "prediction_count": Prediction.objects.count(),
        "matured_count": PredictionOutcome.objects.filter(
            status=PredictionOutcome.Status.MATURED
        ).count(),
    }
    return render(request, "web/status.html", context)


@login_required
def opportunities_page(request: HttpRequest) -> HttpResponse:
    latest_run = _latest_analysis_run()
    analyses: QuerySet[StockAnalysis] = StockAnalysis.objects.none()
    countries: list[str] = []
    exchanges: list[str] = []
    sectors: list[str] = []
    latest_analysis_mode = ""
    if latest_run is not None:
        base_analyses = (
            StockAnalysis.objects.filter(run=latest_run)
            .select_related("listing__security__company")
            .order_by("-overall_score")
        )
        latest_analysis = base_analyses.first()
        if latest_analysis is not None and isinstance(latest_analysis.data_quality, dict):
            latest_analysis_mode = str(latest_analysis.data_quality.get("analysis_mode", ""))
        countries = list(
            base_analyses.order_by("listing__security__company__country")
            .values_list("listing__security__company__country", flat=True)
            .distinct()
        )
        exchanges = list(
            base_analyses.order_by("listing__exchange_mic")
            .values_list("listing__exchange_mic", flat=True)
            .distinct()
        )
        sectors = list(
            base_analyses.exclude(listing__security__company__sector="")
            .order_by("listing__security__company__sector")
            .values_list("listing__security__company__sector", flat=True)
            .distinct()
        )
        analyses = base_analyses
    filter_form = OpportunityFilterForm(request.GET or None)
    filter_form.configure_choices(
        countries=countries,
        exchanges=exchanges,
        sectors=sectors,
    )
    if latest_run is not None and filter_form.is_valid():
        analyses = _filter_analyses(analyses, filter_form.cleaned_data)
    elif request.GET:
        analyses = analyses.none()

    displayed_analyses = list(analyses[:50])
    analysis_cards: list[dict[str, Any]] = []
    great_opportunities: list[dict[str, Any]] = []
    for displayed_analysis in displayed_analyses:
        assessment = assess_opportunity(displayed_analysis)
        card = {
            "analysis": displayed_analysis,
            "opportunity": assessment,
        }
        analysis_cards.append(card)
        if assessment.eligible and len(great_opportunities) < 6:
            great_opportunities.append(card)
    return render(
        request,
        "web/opportunities.html",
        {
            "latest_run": latest_run,
            "latest_analysis_mode": latest_analysis_mode,
            "analyses": displayed_analyses,
            "analysis_cards": analysis_cards,
            "great_opportunities": great_opportunities,
            "result_count": analyses.count(),
            "filter_form": filter_form,
        },
    )


@login_required
def stock_detail_page(request: HttpRequest, listing_id: UUID) -> HttpResponse:
    listing = get_object_or_404(
        Listing.objects.select_related("security__company"),
        pk=listing_id,
    )
    analysis = (
        StockAnalysis.objects.filter(listing=listing, run__status="complete")
        .select_related("run")
        .order_by("-run__generated_at")
        .first()
    )
    predictions = (
        Prediction.objects.filter(listing=listing)
        .select_related("analysis")
        .order_by("-generated_at", "horizon")[:30]
    )
    return render(
        request,
        "web/stock_detail.html",
        {
            "listing": listing,
            "analysis": analysis,
            "opportunity": assess_opportunity(analysis) if analysis is not None else None,
            "predictions": predictions,
        },
    )


@login_required
def prediction_history_page(request: HttpRequest) -> HttpResponse:
    predictions = Prediction.objects.select_related(
        "listing__security__company",
        "analysis__run__universe_snapshot",
        "outcome",
    ).order_by("-generated_at", "listing__ticker")
    horizon = request.GET.get("horizon", "")
    recommendation = request.GET.get("recommendation", "")
    if horizon in Prediction.Horizon.values:
        predictions = predictions.filter(horizon=horizon)
    if recommendation in Recommendation.values:
        predictions = predictions.filter(recommendation=recommendation)

    return render(
        request,
        "web/predictions.html",
        {
            "predictions": predictions[:100],
            "result_count": predictions.count(),
            "horizons": Prediction.Horizon.choices,
            "recommendations": Recommendation.choices,
            "filters": request.GET,
        },
    )


@login_required
def market_overview_page(request: HttpRequest) -> HttpResponse:
    market_rows = list(
        LatestMarketData.objects.select_related("listing__security__company").order_by(
            "listing__region",
            "listing__ticker",
        )
    )
    regions: list[dict[str, Any]] = []
    for region_value, region_label in Region.choices:
        region_rows = [row for row in market_rows if row.listing.region == region_value]
        changes = [
            (row.close - row.previous_close) / row.previous_close
            for row in region_rows
            if row.previous_close is not None and row.previous_close > 0
        ]
        regions.append(
            {
                "value": region_value,
                "label": region_label,
                "listing_count": len(region_rows),
                "mean_change": sum(changes, Decimal(0)) / len(changes) if changes else None,
                "advancers": sum(change > 0 for change in changes),
                "decliners": sum(change < 0 for change in changes),
                "unchanged": sum(change == 0 for change in changes),
                "latest_observation": max(
                    (row.session_date for row in region_rows),
                    default=None,
                ),
            }
        )

    sector_changes: dict[str, list[Decimal]] = {}
    for row in market_rows:
        if row.previous_close is None or row.previous_close <= 0:
            continue
        sector = row.listing.security.company.sector or "Unclassified"
        sector_changes.setdefault(sector, []).append(
            (row.close - row.previous_close) / row.previous_close
        )
    sectors: list[dict[str, Any]] = [
        {
            "name": sector,
            "listing_count": len(changes),
            "mean_change": sum(changes, Decimal(0)) / len(changes),
        }
        for sector, changes in sector_changes.items()
    ]
    sectors.sort(key=lambda sector: sector["mean_change"], reverse=True)
    latest_listings = [
        {
            "row": row,
            "change": (
                (row.close - row.previous_close) / row.previous_close
                if row.previous_close is not None and row.previous_close > 0
                else None
            ),
        }
        for row in market_rows[:12]
    ]

    return render(
        request,
        "web/market.html",
        {
            "market_rows": latest_listings,
            "regions": regions,
            "sectors": sectors,
        },
    )


@login_required
def performance_page(request: HttpRequest) -> HttpResponse:
    outcomes = PredictionOutcome.objects.select_related("prediction__listing__security__company")
    matured = outcomes.filter(
        status=PredictionOutcome.Status.MATURED,
        actual_return__isnull=False,
    )
    on_time_observed = Q(
        prediction__analysis__run__universe_snapshot__grade=UniverseSnapshot.Grade.OBSERVED,
        prediction__analysis__run__issued_on_time=True,
        prediction__issued_on_time=True,
    )
    reportable_matured = matured.filter(on_time_observed)
    summary = reportable_matured.aggregate(
        sample_count=Count("prediction"),
        mean_return=Avg("actual_return"),
        mean_benchmark_return=Avg("benchmark_return"),
    )
    sample_count = int(summary["sample_count"] or 0)
    positive_count = reportable_matured.filter(actual_return__gt=0).count()
    assessed_count = reportable_matured.filter(success__isnull=False).count()
    successful_count = reportable_matured.filter(success=True).count()
    summary.update(
        {
            "positive_rate": (
                Decimal(positive_count) / Decimal(sample_count) if sample_count else None
            ),
            "directional_accuracy": (
                Decimal(successful_count) / Decimal(assessed_count) if assessed_count else None
            ),
            "sufficient_sample": sample_count >= 30,
            "unresolved_count": outcomes.filter(
                on_time_observed,
                status=PredictionOutcome.Status.UNRESOLVED,
            ).count(),
            "corporate_event_count": outcomes.filter(
                on_time_observed,
                status=PredictionOutcome.Status.CORPORATE_EVENT,
            ).count(),
            "research_matured_count": matured.exclude(on_time_observed).count(),
        }
    )
    groups = (
        reportable_matured.values("prediction__horizon", "prediction__recommendation")
        .annotate(
            sample_count=Count("prediction"),
            mean_return=Avg("actual_return"),
            mean_benchmark_return=Avg("benchmark_return"),
        )
        .order_by("prediction__horizon", "prediction__recommendation")
    )
    return render(
        request,
        "web/performance.html",
        {
            "summary": summary,
            "groups": groups,
        },
    )


@login_required
def simulations_page(request: HttpRequest) -> HttpResponse:
    form: SimulationForm | None = None
    if request.method == "POST":
        form = SimulationForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            try:
                _, run, _ = run_simulation_workflow(
                    name=data["name"],
                    mode=data["mode"],
                    snapshot=data["snapshot"],
                    start_date=data["start_date"],
                    end_date=data["end_date"],
                    starting_capital=float(data["starting_capital"]),
                    transaction_cost_bps=float(data["transaction_cost_bps"]),
                    slippage_bps=float(data["slippage_bps"]),
                    top_n=data.get("top_n"),
                    selected_listing_ids=data.get("parsed_listing_ids"),
                    benchmark_subject=data.get("benchmark_subject") or None,
                    benchmark_currency=data.get("benchmark_currency") or None,
                    base_currency=data.get("base_currency") or None,
                    restrict_native_currency=data.get("restrict_native_currency") or None,
                    fx_max_carry_days=(
                        DEFAULT_MAX_CARRY_DAYS
                        if data.get("fx_max_carry_days") is None
                        else int(data["fx_max_carry_days"])
                    ),
                )
                return redirect("simulation-detail", run_id=run.id)
            except (SimulationWorkflowError, ValueError) as exc:
                form.add_error(None, str(exc))

    if form is None:
        form = SimulationForm()

    mode = request.GET.get("mode", "")
    runs = SimulationRun.objects.select_related("definition", "universe_snapshot").order_by(
        "-started_at"
    )
    if mode in SimulationDefinition.Mode.values:
        runs = runs.filter(definition__mode=mode)

    status_code = (
        HTTPStatus.BAD_REQUEST
        if request.method == "POST" and not form.is_valid()
        else HTTPStatus.OK
    )
    return render(
        request,
        "web/simulations.html",
        {
            "form": form,
            "runs": runs[:50],
            "result_count": runs.count(),
            "modes": SimulationDefinition.Mode.choices,
            "selected_mode": mode,
        },
        status=status_code,
    )


@login_required
def simulation_detail_page(request: HttpRequest, run_id: UUID) -> HttpResponse:
    run = get_object_or_404(
        SimulationRun.objects.select_related("definition", "universe_snapshot__universe"),
        pk=run_id,
    )
    return render(
        request,
        "web/simulation_detail.html",
        {
            "run": run,
            "metrics": run.metrics.items() if isinstance(run.metrics, dict) else [],
            "holdings": run.holdings.select_related("listing").order_by(
                "-observation_date",
                "-market_value",
            )[:50],
            "trades": run.trades.select_related("listing").order_by(
                "-trade_date",
                "listing__ticker",
            )[:50],
        },
    )


@login_required
def portfolios_page(request: HttpRequest) -> HttpResponse:
    owner = cast(User, request.user)
    if request.method == "POST":
        form = PortfolioForm(request.POST, owner=owner)
        if form.is_valid():
            portfolio = form.save()
            record_portfolio_snapshot(portfolio)
            messages.success(request, f"Portfolio “{portfolio.name}” created.")
            return redirect("portfolio-detail", portfolio_id=portfolio.id)
    else:
        form = PortfolioForm(owner=owner)

    active = list(Portfolio.objects.filter(owner=owner, archived_at__isnull=True).order_by("name"))
    archived = Portfolio.objects.filter(owner=owner, archived_at__isnull=False).order_by("name")
    cards = [
        {
            "portfolio": portfolio,
            "valuation": calculate_portfolio_valuation(portfolio),
            "snapshot_count": portfolio.snapshots.count(),
        }
        for portfolio in active
    ]
    return render(
        request,
        "web/portfolios.html",
        {
            "form": form,
            "portfolio_cards": cards,
            "archived_portfolios": archived,
        },
        status=(
            HTTPStatus.BAD_REQUEST
            if request.method == "POST" and not form.is_valid()
            else HTTPStatus.OK
        ),
    )


@login_required
def portfolio_detail_page(request: HttpRequest, portfolio_id: UUID) -> HttpResponse:
    owner = cast(User, request.user)
    portfolio = get_object_or_404(Portfolio, pk=portfolio_id, owner=owner)
    portfolio_form = PortfolioForm(instance=portfolio, owner=owner)
    holding_form = PortfolioHoldingForm(portfolio=portfolio)

    if request.method == "POST":
        action = request.POST.get("action", "")
        if portfolio.archived_at is not None and action != "restore":
            messages.error(request, "Restore this portfolio before changing it.")
            return redirect("portfolio-detail", portfolio_id=portfolio.id)
        if action == "update":
            portfolio_form = PortfolioForm(request.POST, instance=portfolio, owner=owner)
            if portfolio_form.is_valid():
                portfolio_form.save()
                _snapshot_with_message(request, portfolio)
                messages.success(request, "Portfolio settings updated.")
                return redirect("portfolio-detail", portfolio_id=portfolio.id)
        elif action == "holding":
            holding_form = PortfolioHoldingForm(request.POST, portfolio=portfolio)
            if holding_form.is_valid():
                data = holding_form.cleaned_data
                try:
                    upsert_holding(
                        portfolio=portfolio,
                        listing=data["listing"],
                        quantity=data["quantity"],
                        average_cost=data["average_cost"],
                        acquired_on=data.get("acquired_on"),
                        notes=data.get("notes") or "",
                    )
                except PortfolioValuationError as exc:
                    holding_form.add_error(None, str(exc))
                else:
                    _snapshot_with_message(request, portfolio)
                    messages.success(request, "Holding saved.")
                    return redirect("portfolio-detail", portfolio_id=portfolio.id)
        elif action == "snapshot":
            _snapshot_with_message(request, portfolio, success_message=True)
            return redirect("portfolio-detail", portfolio_id=portfolio.id)
        elif action == "archive":
            portfolio.archived_at = timezone.now()
            portfolio.save(update_fields=["archived_at", "updated_at"])
            messages.success(request, "Portfolio archived; its snapshots remain immutable.")
            return redirect("portfolios")
        elif action == "restore":
            portfolio.archived_at = None
            portfolio.save(update_fields=["archived_at", "updated_at"])
            messages.success(request, "Portfolio restored.")
            return redirect("portfolio-detail", portfolio_id=portfolio.id)
        else:
            messages.error(request, "Unknown portfolio action.")
            return redirect("portfolio-detail", portfolio_id=portfolio.id)

    status_code = (
        HTTPStatus.BAD_REQUEST
        if request.method == "POST"
        and (not portfolio_form.is_valid() or not holding_form.is_valid())
        else HTTPStatus.OK
    )
    return render(
        request,
        "web/portfolio_detail.html",
        _portfolio_detail_context(
            portfolio=portfolio,
            portfolio_form=portfolio_form,
            holding_form=holding_form,
        ),
        status=status_code,
    )


@login_required
@require_POST
def portfolio_holding_delete(
    request: HttpRequest,
    portfolio_id: UUID,
    holding_id: int,
) -> HttpResponse:
    owner = cast(User, request.user)
    portfolio = get_object_or_404(
        Portfolio,
        pk=portfolio_id,
        owner=owner,
        archived_at__isnull=True,
    )
    holding = get_object_or_404(
        PortfolioHolding,
        pk=holding_id,
        portfolio=portfolio,
    )
    ticker = holding.listing.ticker
    holding.delete()
    _snapshot_with_message(request, portfolio)
    messages.success(request, f"{ticker} removed from the portfolio.")
    return redirect("portfolio-detail", portfolio_id=portfolio.id)


@login_required
def methodology_page(request: HttpRequest) -> HttpResponse:
    return render(request, "web/methodology.html")


def _latest_analysis_run() -> AnalysisRun | None:
    return (
        AnalysisRun.objects.filter(status="complete")
        .select_related("universe_snapshot__universe")
        .order_by("-generated_at")
        .first()
    )


def _filter_analyses(
    analyses: QuerySet[StockAnalysis],
    filters: dict[str, Any],
) -> QuerySet[StockAnalysis]:
    query = filters["q"].strip()
    region = filters["region"]
    recommendation = filters["recommendation"]
    risk = filters["risk"]
    country = filters["country"]
    exchange = filters["exchange"]
    sector = filters["sector"]

    if query:
        analyses = analyses.filter(
            Q(listing__ticker__icontains=query)
            | Q(listing__security__company__name__icontains=query)
        )
    if region in {"us", "europe"}:
        analyses = analyses.filter(listing__region=region)
    if recommendation in Recommendation.values:
        analyses = analyses.filter(recommendation=recommendation)
    if risk in RiskClass.values:
        analyses = analyses.filter(risk_class=risk)
    if country:
        analyses = analyses.filter(listing__security__company__country=country)
    if exchange:
        analyses = analyses.filter(listing__exchange_mic=exchange)
    if sector:
        analyses = analyses.filter(listing__security__company__sector=sector)
    analyses = _apply_decimal_minimum(
        analyses,
        field="overall_score",
        minimum=filters["min_score"],
    )
    analyses = _apply_decimal_minimum(
        analyses,
        field="confidence",
        minimum=filters["min_confidence"],
    )
    return analyses


def _apply_decimal_minimum(
    analyses: QuerySet[StockAnalysis],
    *,
    field: str,
    minimum: Decimal | None,
) -> QuerySet[StockAnalysis]:
    if minimum is None:
        return analyses
    return analyses.filter(**{f"{field}__gte": minimum})


def _portfolio_detail_context(
    *,
    portfolio: Portfolio,
    portfolio_form: PortfolioForm,
    holding_form: PortfolioHoldingForm,
) -> dict[str, Any]:
    valuation = calculate_portfolio_valuation(portfolio)
    listing_ids = [position.holding.listing_id for position in valuation.positions]
    latest_by_listing: dict[UUID, StockAnalysis] = {}
    latest_run = _latest_analysis_run()
    if latest_run is not None:
        analyses = StockAnalysis.objects.filter(
            listing_id__in=listing_ids,
            run=latest_run,
        ).order_by("-pk")
        for persisted_analysis in analyses:
            latest_by_listing[persisted_analysis.listing_id] = persisted_analysis
    position_cards = []
    for position in valuation.positions:
        latest_analysis = latest_by_listing.get(position.holding.listing_id)
        position_cards.append(
            {
                "position": position,
                "analysis": latest_analysis,
                "opportunity": (
                    assess_opportunity(latest_analysis) if latest_analysis is not None else None
                ),
            }
        )
    snapshots = portfolio_snapshot_series(portfolio)
    return {
        "portfolio": portfolio,
        "portfolio_form": portfolio_form,
        "holding_form": holding_form,
        "valuation": valuation,
        "position_cards": position_cards,
        "snapshots": list(reversed(snapshots)),
        "first_snapshot": snapshots[0] if snapshots else None,
        "latest_snapshot": snapshots[-1] if snapshots else None,
    }


def _snapshot_with_message(
    request: HttpRequest,
    portfolio: Portfolio,
    *,
    success_message: bool = False,
) -> None:
    try:
        snapshot, created = record_portfolio_snapshot(portfolio)
    except PortfolioValuationError as exc:
        messages.warning(request, str(exc))
        return
    if success_message:
        if created:
            messages.success(
                request,
                f"Snapshot recorded for {snapshot.as_of_date.isoformat()}.",
            )
        else:
            messages.info(request, "The current valuation is already recorded.")

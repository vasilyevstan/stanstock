from __future__ import annotations

from decimal import Decimal
from http import HTTPStatus
from typing import Any
from uuid import UUID

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Count, F, Q, QuerySet
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from stanstock.core.models import JobRun
from stanstock.core.services import system_status
from stanstock.data.models import (
    LatestMarketData,
    Listing,
    ProviderRecord,
    Region,
    UniverseSnapshot,
)
from stanstock.research.models import (
    AnalysisRun,
    Prediction,
    PredictionOutcome,
    Recommendation,
    RiskClass,
    StockAnalysis,
)
from stanstock.simulation.builders import run_simulation_workflow
from stanstock.simulation.models import SimulationDefinition, SimulationRun
from stanstock.simulation.types import SimulationWorkflowError
from stanstock.web.demo import DEMO_OPPORTUNITIES
from stanstock.web.forms import OpportunityFilterForm, SimulationForm


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
    if latest_run is not None:
        base_analyses = (
            StockAnalysis.objects.filter(run=latest_run)
            .select_related("listing__security__company")
            .order_by("-overall_score")
        )
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

    return render(
        request,
        "web/opportunities.html",
        {
            "latest_run": latest_run,
            "analyses": analyses[:50],
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
                    (row.observed_at for row in region_rows),
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
        prediction__generated_at__date=F("prediction__target_date"),
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
                    base_currency=data.get("base_currency") or None,
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

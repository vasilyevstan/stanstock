from __future__ import annotations

from decimal import Decimal
from http import HTTPStatus
from typing import Any, cast
from uuid import UUID, uuid4

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db import transaction
from django.db.models import Avg, Count, Q, QuerySet
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from stanstock.core.launchd import launch_agent_status
from stanstock.core.models import JobRun
from stanstock.core.services import system_status
from stanstock.data.etfs import (
    INVESTABLE_US_ETF_MIC,
    INVESTABLE_US_ETF_SYMBOL,
    build_etf_overview,
)
from stanstock.data.fx import DEFAULT_MAX_CARRY_DAYS
from stanstock.data.models import (
    LatestMarketData,
    Listing,
    ProviderRecord,
    Region,
    Security,
    UniverseSnapshot,
)
from stanstock.portfolio.models import Portfolio, PortfolioHolding
from stanstock.portfolio.planner import (
    PortfolioPlanningError,
    calculate_contribution_performance,
    confirm_monthly_contribution_plan,
    preview_monthly_contribution_plan,
    record_external_deposit,
)
from stanstock.portfolio.service import (
    PortfolioValuation,
    PortfolioValuationError,
    build_sample_portfolio,
    calculate_portfolio_valuation,
    delete_holding,
    portfolio_snapshot_series,
    record_portfolio_snapshot,
    restore_portfolio,
    upsert_holding,
)
from stanstock.research.affordability import (
    PRICE_BAND_CURRENCY,
    PRICE_BAND_POLICY_VERSION,
    PRICE_BANDS,
    PRICE_BANDS_BY_SLUG,
    UNDER_10_LONG_HORIZON_GATES,
    PriceBandAssessment,
    PriceBandDefinition,
    latest_price_band,
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
from stanstock.research.provenance import (
    analysis_run_data_mode,
    analysis_run_source_providers,
    data_mode_label,
    latest_provider_backed_analysis_run,
    latest_serving_analysis_run,
    source_data_mode,
    source_providers,
)
from stanstock.simulation.builders import run_simulation_workflow
from stanstock.simulation.models import SimulationDefinition, SimulationRun
from stanstock.simulation.types import SimulationWorkflowError
from stanstock.web.demo import DEMO_OPPORTUNITIES
from stanstock.web.forms import (
    OpportunityFilterForm,
    PortfolioDepositForm,
    PortfolioForm,
    PortfolioHoldingForm,
    PortfolioPlanConfirmationForm,
    SamplePortfolioForm,
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
    data_mode = analysis_run_data_mode(latest_run)
    data_providers = analysis_run_source_providers(latest_run)
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
        "data_mode": data_mode,
        "data_mode_label": data_mode_label(data_mode, data_providers),
        "opportunities": persisted_opportunities,
        "demo_opportunities": (
            DEMO_OPPORTUNITIES if settings.DEMO_MODE and not persisted_opportunities else []
        ),
        "latest_run": latest_run,
        "providers": ProviderRecord.objects.order_by("provider"),
        "recent_jobs": JobRun.objects.order_by("-started_at")[:5],
        "scheduler": _scheduler_status(),
        "prediction_count": Prediction.objects.filter(analysis__run=latest_run).count()
        if latest_run
        else 0,
        "matured_count": PredictionOutcome.objects.filter(
            prediction__analysis__run=latest_run, status=PredictionOutcome.Status.MATURED
        ).count()
        if latest_run
        else 0,
    }
    return render(request, "web/status.html", context)


def _scheduler_status() -> dict[str, object]:
    try:
        return launch_agent_status()
    except (OSError, ValueError) as exc:
        return {
            "installed": False,
            "loaded": False,
            "timezone_matches": False,
            "error": str(exc),
        }


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
            StockAnalysis.objects.filter(
                run=latest_run,
                listing__security__security_type__in=(
                    Security.SecurityType.COMMON_STOCK,
                    Security.SecurityType.ADR,
                ),
            )
            .select_related(
                "listing__security__company",
                "listing__latest_market_data",
            )
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
    filters_valid = filter_form.is_valid()
    if latest_run is not None and filters_valid:
        analyses = _filter_analyses(analyses, filter_form.cleaned_data)
    elif request.GET:
        analyses = analyses.none()

    selected_price_band = (
        str(filter_form.cleaned_data.get("price_band") or "") if filters_valid else ""
    )
    analysis_cards: list[dict[str, Any]] = []
    great_opportunities: list[dict[str, Any]] = []
    for displayed_analysis in analyses[:50]:
        card = _opportunity_card(displayed_analysis)
        if card["opportunity"].eligible and len(great_opportunities) < 6:
            great_opportunities.append(card)
    price_band_groups: list[dict[str, Any]] = []
    if latest_run is None:
        visible_definitions: tuple[PriceBandDefinition, ...] = ()
    elif selected_price_band in PRICE_BANDS_BY_SLUG:
        visible_definitions = (PRICE_BANDS_BY_SLUG[selected_price_band],)
    else:
        visible_definitions = PRICE_BANDS
    for definition in visible_definitions:
        group_analyses = _analyses_in_price_band(analyses, definition)
        cards = [_opportunity_card(analysis) for analysis in group_analyses[:50]]
        analysis_cards.extend(cards)
        price_band_groups.append(
            {
                "slug": definition.slug,
                "eyebrow": "Latest persisted USD close",
                "label": definition.label,
                "description": definition.description,
                "new_allocation_eligible": definition.new_allocation_eligible,
                "count": group_analyses.count(),
                "cards": cards,
                "is_unavailable": False,
            }
        )
    if not selected_price_band:
        unavailable_analyses = _analyses_without_usd_price_band(analyses)
        unavailable_count = unavailable_analyses.count()
        if unavailable_count:
            unavailable_cards = [
                _opportunity_card(analysis) for analysis in unavailable_analyses[:50]
            ]
            analysis_cards.extend(unavailable_cards)
            price_band_groups.append(
                {
                    "slug": "unavailable",
                    "eyebrow": "Price-band guard",
                    "label": "USD price band unavailable",
                    "description": "No valid latest persisted USD close is available.",
                    "new_allocation_eligible": False,
                    "count": unavailable_count,
                    "cards": unavailable_cards,
                    "is_unavailable": True,
                }
            )
        non_usd_analyses = _analyses_outside_usd_price_band_policy(analyses)
        non_usd_count = non_usd_analyses.count()
        if non_usd_count:
            non_usd_cards = [_opportunity_card(analysis) for analysis in non_usd_analyses[:50]]
            analysis_cards.extend(non_usd_cards)
            price_band_groups.append(
                {
                    "slug": "not_applicable",
                    "eyebrow": "Outside current USD scope",
                    "label": "USD price band not applicable",
                    "description": "This listing does not trade in USD.",
                    "new_allocation_eligible": True,
                    "count": non_usd_count,
                    "cards": non_usd_cards,
                    "is_unavailable": False,
                }
            )
    return render(
        request,
        "web/opportunities.html",
        {
            "latest_run": latest_run,
            "latest_analysis_mode": latest_analysis_mode,
            "analyses": [card["analysis"] for card in analysis_cards],
            "analysis_cards": analysis_cards,
            "great_opportunities": great_opportunities,
            "price_band_groups": price_band_groups,
            "result_count": analyses.count(),
            "filter_form": filter_form,
            "price_band_currency": PRICE_BAND_CURRENCY,
            "price_band_policy_version": PRICE_BAND_POLICY_VERSION,
            "under_10_long_horizon_gates": UNDER_10_LONG_HORIZON_GATES,
        },
    )


@login_required
def stock_detail_page(request: HttpRequest, listing_id: UUID) -> HttpResponse:
    listing = get_object_or_404(
        Listing.objects.select_related(
            "security__company",
            "latest_market_data",
        ),
        pk=listing_id,
    )
    if listing.security.security_type == Security.SecurityType.ETF:
        return redirect("etf-detail", listing_id=listing.id)
    analysis = (
        StockAnalysis.objects.filter(listing=listing, run__status="complete")
        .select_related("run", "listing__latest_market_data")
        .order_by("-run__generated_at")
        .first()
    )
    predictions = (
        Prediction.objects.filter(listing=listing)
        .select_related("analysis")
        .order_by("-generated_at", "horizon")[:30]
    )
    current_price_band = latest_price_band(listing)
    (
        long_horizon_blocked,
        long_horizon_band_reason,
        long_horizon_gates,
    ) = _long_horizon_band_state(
        listing=listing,
        current_price_band=current_price_band,
    )
    return render(
        request,
        "web/stock_detail.html",
        {
            "listing": listing,
            "analysis": analysis,
            "opportunity": (
                assess_opportunity(
                    analysis,
                    price_band=current_price_band,
                )
                if analysis is not None
                else None
            ),
            "current_price_band": current_price_band,
            "price_band_policy_version": PRICE_BAND_POLICY_VERSION,
            "long_horizon_blocked": long_horizon_blocked,
            "long_horizon_band_reason": long_horizon_band_reason,
            "long_horizon_gates": long_horizon_gates,
            "predictions": predictions,
        },
    )


@login_required
def etf_detail_page(request: HttpRequest, listing_id: UUID) -> HttpResponse:
    listing = get_object_or_404(
        Listing.objects.select_related(
            "security__company",
            "latest_market_data__source_asset",
        ),
        pk=listing_id,
        security__security_type=Security.SecurityType.ETF,
        ticker=INVESTABLE_US_ETF_SYMBOL,
        provider_symbol=INVESTABLE_US_ETF_SYMBOL,
        exchange_mic=INVESTABLE_US_ETF_MIC,
        currency="USD",
        region=Region.US,
        is_primary=True,
        is_active=True,
    )
    return render(
        request,
        "web/etf_detail.html",
        {
            "listing": listing,
            "overview": build_etf_overview(listing),
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

    displayed_predictions = list(predictions[:100])
    prediction_cards = [
        {
            "prediction": prediction,
            "source_mode": source_data_mode({"source_assets": prediction.source_assets}),
            "source_providers": source_providers({"source_assets": prediction.source_assets}),
        }
        for prediction in displayed_predictions
    ]
    return render(
        request,
        "web/predictions.html",
        {
            "prediction_cards": prediction_cards,
            "result_count": predictions.count(),
            "horizons": Prediction.Horizon.choices,
            "recommendations": Recommendation.choices,
            "filters": request.GET,
        },
    )


@login_required
def market_overview_page(request: HttpRequest) -> HttpResponse:
    latest_run = _latest_analysis_run()
    market_query = LatestMarketData.objects.select_related(
        "listing__security__company",
        "source_asset",
    ).filter(
        listing__security__security_type__in=(
            Security.SecurityType.COMMON_STOCK,
            Security.SecurityType.ADR,
        )
    )
    if latest_run is not None:
        market_query = market_query.filter(
            listing_id__in=StockAnalysis.objects.filter(run=latest_run).values("listing_id")
        )
    market_rows = list(
        market_query.order_by(
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
    etf_rows = [
        {
            "row": row,
            "change": (
                (row.close - row.previous_close) / row.previous_close
                if row.previous_close is not None and row.previous_close > 0
                else None
            ),
        }
        for row in LatestMarketData.objects.select_related(
            "listing__security__company",
            "source_asset",
        )
        .filter(
            listing__security__security_type=Security.SecurityType.ETF,
            listing__ticker=INVESTABLE_US_ETF_SYMBOL,
            listing__provider_symbol=INVESTABLE_US_ETF_SYMBOL,
            listing__exchange_mic=INVESTABLE_US_ETF_MIC,
            listing__currency="USD",
            listing__region=Region.US,
            listing__is_primary=True,
            listing__is_active=True,
        )
        .order_by("listing__ticker")
    ]

    return render(
        request,
        "web/market.html",
        {
            "market_rows": latest_listings,
            "etf_rows": etf_rows,
            "regions": regions,
            "sectors": sectors,
            "data_mode": analysis_run_data_mode(latest_run),
            "data_mode_label": data_mode_label(
                analysis_run_data_mode(latest_run),
                analysis_run_source_providers(latest_run),
            ),
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
    latest_method_prediction = (
        Prediction.objects.filter(
            analysis__run__universe_snapshot__grade=UniverseSnapshot.Grade.OBSERVED,
            analysis__run__issued_on_time=True,
            issued_on_time=True,
        )
        .select_related("analysis__run")
        .order_by("-generated_at", "-id")
        .first()
    )
    current_method_matured = reportable_matured.none()
    current_method_outcomes = outcomes.none()
    current_config_version = ""
    current_config_hash = ""
    if latest_method_prediction is not None:
        current_config_version = latest_method_prediction.analysis.run.config_version
        current_config_hash = latest_method_prediction.config_hash
        method_filter = Q(
            prediction__analysis__run__config_version=current_config_version,
            prediction__config_hash=current_config_hash,
        )
        current_method_matured = reportable_matured.filter(method_filter)
        current_method_outcomes = outcomes.filter(method_filter)

    summary = current_method_matured.aggregate(
        sample_count=Count("prediction"),
        mean_return=Avg("actual_return"),
        mean_benchmark_return=Avg("benchmark_return"),
    )
    sample_count = int(summary["sample_count"] or 0)
    positive_count = current_method_matured.filter(actual_return__gt=0).count()
    assessed_count = current_method_matured.filter(success__isnull=False).count()
    successful_count = current_method_matured.filter(success=True).count()
    summary.update(
        {
            "config_version": current_config_version,
            "config_hash": current_config_hash,
            "positive_rate": (
                Decimal(positive_count) / Decimal(sample_count) if sample_count else None
            ),
            "directional_accuracy": (
                Decimal(successful_count) / Decimal(assessed_count) if assessed_count else None
            ),
            "sufficient_sample": sample_count >= 30,
            "unresolved_count": current_method_outcomes.filter(
                on_time_observed,
                status=PredictionOutcome.Status.UNRESOLVED,
            ).count(),
            "corporate_event_count": current_method_outcomes.filter(
                on_time_observed,
                status=PredictionOutcome.Status.CORPORATE_EVENT,
            ).count(),
            "research_matured_count": matured.exclude(on_time_observed).count(),
            "method_count": reportable_matured.values(
                "prediction__analysis__run__config_version",
                "prediction__config_hash",
            )
            .distinct()
            .count(),
        }
    )
    groups = (
        reportable_matured.values(
            "prediction__analysis__run__config_version",
            "prediction__config_hash",
            "prediction__horizon",
            "prediction__recommendation",
        )
        .annotate(
            sample_count=Count("prediction"),
            mean_return=Avg("actual_return"),
            mean_benchmark_return=Avg("benchmark_return"),
        )
        .order_by(
            "prediction__analysis__run__config_version",
            "prediction__config_hash",
            "prediction__horizon",
            "prediction__recommendation",
        )
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
    form = PortfolioForm(owner=owner)
    sample_form = SamplePortfolioForm()
    invalid_form = False
    if request.method == "POST":
        action = request.POST.get("action", "manual")
        if action == "sample":
            sample_form = SamplePortfolioForm(request.POST)
            invalid_form = not sample_form.is_valid()
            if not invalid_form:
                try:
                    portfolio, created = build_sample_portfolio(
                        owner=owner,
                        starting_capital=sample_form.cleaned_data["starting_capital"],
                        top_n=sample_form.cleaned_data["top_n"],
                    )
                except PortfolioValuationError as exc:
                    sample_form.add_error(None, str(exc))
                    invalid_form = True
                else:
                    if created:
                        messages.success(
                            request,
                            "StanStock sample portfolio created from provider-backed "
                            "opportunities.",
                        )
                    else:
                        messages.info(
                            request,
                            "The sample portfolio for the latest source run already exists.",
                        )
                    return redirect("portfolio-detail", portfolio_id=portfolio.id)
        elif action == "manual":
            form = PortfolioForm(request.POST, owner=owner)
            invalid_form = not form.is_valid()
            if not invalid_form:
                with transaction.atomic():
                    portfolio = form.save(commit=False)
                    starting_cash = portfolio.cash_balance
                    portfolio.cash_balance = Decimal(0)
                    portfolio.save()
                    if starting_cash > 0:
                        record_external_deposit(
                            portfolio=portfolio,
                            amount=starting_cash,
                            idempotency_key=uuid4(),
                            note="Initial portfolio cash",
                        )
                        portfolio.refresh_from_db()
                    record_portfolio_snapshot(portfolio)
                messages.success(request, f"Portfolio “{portfolio.name}” created.")
                return redirect("portfolio-detail", portfolio_id=portfolio.id)
        else:
            messages.error(request, "Unknown portfolio action.")
            return redirect("portfolios")

    active = list(Portfolio.objects.filter(owner=owner, archived_at__isnull=True).order_by("name"))
    archived = Portfolio.objects.filter(owner=owner, archived_at__isnull=False).order_by("name")
    cards: list[dict[str, Any]] = []
    for portfolio in active:
        valuation = calculate_portfolio_valuation(portfolio)
        card: dict[str, Any] = {
            "portfolio": portfolio,
            "valuation": valuation,
            "snapshot_count": portfolio.snapshots.count(),
        }
        card.update(_model_portfolio_metrics(portfolio, valuation))
        cards.append(card)
    return render(
        request,
        "web/portfolios.html",
        {
            "form": form,
            "sample_form": sample_form,
            "sample_source_run": latest_provider_backed_analysis_run(),
            "portfolio_cards": cards,
            "archived_portfolios": archived,
        },
        status=HTTPStatus.BAD_REQUEST if invalid_form else HTTPStatus.OK,
    )


@login_required
def portfolio_detail_page(request: HttpRequest, portfolio_id: UUID) -> HttpResponse:
    owner = cast(User, request.user)
    portfolio = get_object_or_404(Portfolio, pk=portfolio_id, owner=owner)
    portfolio_form = PortfolioForm(instance=portfolio, owner=owner)
    holding_form = PortfolioHoldingForm(portfolio=portfolio)
    deposit_form = PortfolioDepositForm(portfolio=portfolio)
    plan_confirmation_form = PortfolioPlanConfirmationForm()
    plan_confirmation_error = ""
    invalid_form = False

    if request.method == "POST":
        action = request.POST.get("action", "")
        if portfolio.archived_at is not None and action != "restore":
            messages.error(request, "Restore this portfolio before changing it.")
            return redirect("portfolio-detail", portfolio_id=portfolio.id)
        if portfolio.is_model_portfolio and action in {
            "update",
            "holding",
            "deposit",
            "execute_plan",
        }:
            messages.error(
                request,
                "Model portfolio construction is frozen so its tracked result remains "
                "attributable to the original selection.",
            )
            return redirect("portfolio-detail", portfolio_id=portfolio.id)
        if action == "update":
            updated_portfolio: Portfolio | None = None
            with transaction.atomic():
                locked_portfolio = Portfolio.objects.select_for_update().get(
                    pk=portfolio.pk,
                    owner=owner,
                )
                portfolio_form = PortfolioForm(
                    request.POST,
                    instance=locked_portfolio,
                    owner=owner,
                )
                if portfolio_form.is_valid():
                    updated_portfolio = portfolio_form.save(commit=False)
                    updated_portfolio.save(
                        update_fields=[
                            "name",
                            "description",
                            "base_currency",
                            "monthly_contribution",
                            "allow_fractional_shares",
                            "updated_at",
                        ]
                    )
                    updated_portfolio.refresh_from_db()
            if updated_portfolio is not None:
                portfolio = updated_portfolio
                _snapshot_with_message(request, portfolio)
                messages.success(request, "Portfolio settings updated.")
                return redirect("portfolio-detail", portfolio_id=portfolio.id)
            invalid_form = True
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
            invalid_form = True
        elif action == "deposit":
            deposit_form = PortfolioDepositForm(request.POST, portfolio=portfolio)
            if deposit_form.is_valid():
                try:
                    _deposit, created = record_external_deposit(
                        portfolio=portfolio,
                        amount=deposit_form.cleaned_data["amount"],
                        idempotency_key=deposit_form.cleaned_data["idempotency_key"],
                        note=deposit_form.cleaned_data.get("note") or "",
                    )
                except PortfolioPlanningError as exc:
                    deposit_form.add_error(None, str(exc))
                else:
                    portfolio.refresh_from_db()
                    if created:
                        _snapshot_with_message(request, portfolio)
                        messages.success(
                            request,
                            "External deposit recorded; allocation preview updated.",
                        )
                    else:
                        messages.info(request, "That deposit was already recorded.")
                    return redirect("portfolio-detail", portfolio_id=portfolio.id)
            invalid_form = True
        elif action == "execute_plan":
            plan_confirmation_form = PortfolioPlanConfirmationForm(request.POST)
            if plan_confirmation_form.is_valid():
                try:
                    execution, created = confirm_monthly_contribution_plan(
                        portfolio=portfolio,
                        expected_plan_hash=plan_confirmation_form.cleaned_data["plan_hash"],
                        idempotency_key=plan_confirmation_form.cleaned_data["idempotency_key"],
                    )
                except PortfolioPlanningError as exc:
                    plan_confirmation_error = str(exc)
                    portfolio.refresh_from_db()
                    portfolio_form = PortfolioForm(instance=portfolio, owner=owner)
                    holding_form = PortfolioHoldingForm(portfolio=portfolio)
                    deposit_form = PortfolioDepositForm(portfolio=portfolio)
                    plan_confirmation_form = PortfolioPlanConfirmationForm()
                else:
                    portfolio.refresh_from_db()
                    if created:
                        _snapshot_with_message(request, portfolio)
                        messages.success(
                            request,
                            f"Recorded {execution.purchases.count()} planner "
                            "purchase(s); no brokerage order was sent.",
                        )
                    else:
                        messages.info(request, "That allocation plan was already recorded.")
                    return redirect("portfolio-detail", portfolio_id=portfolio.id)
            else:
                plan_confirmation_error = (
                    "Plan confirmation request was invalid; review the current plan."
                )
                plan_confirmation_form = PortfolioPlanConfirmationForm()
            invalid_form = True
        elif action == "snapshot":
            _snapshot_with_message(request, portfolio, success_message=True)
            return redirect("portfolio-detail", portfolio_id=portfolio.id)
        elif action == "archive":
            portfolio.archived_at = timezone.now()
            portfolio.save(update_fields=["archived_at", "updated_at"])
            messages.success(request, "Portfolio archived; its snapshots remain immutable.")
            return redirect("portfolios")
        elif action == "restore":
            try:
                portfolio = restore_portfolio(portfolio)
            except PortfolioValuationError as exc:
                messages.error(request, str(exc))
                return redirect("portfolio-detail", portfolio_id=portfolio.id)
            messages.success(request, "Portfolio restored.")
            return redirect("portfolio-detail", portfolio_id=portfolio.id)
        else:
            messages.error(request, "Unknown portfolio action.")
            return redirect("portfolio-detail", portfolio_id=portfolio.id)

    return render(
        request,
        "web/portfolio_detail.html",
        _portfolio_detail_context(
            portfolio=portfolio,
            portfolio_form=portfolio_form,
            holding_form=holding_form,
            deposit_form=deposit_form,
            plan_confirmation_form=plan_confirmation_form,
            plan_confirmation_error=plan_confirmation_error,
        ),
        status=HTTPStatus.BAD_REQUEST if invalid_form else HTTPStatus.OK,
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
    try:
        delete_holding(holding)
    except PortfolioValuationError as exc:
        messages.error(request, str(exc))
        return redirect("portfolio-detail", portfolio_id=portfolio.id)
    _snapshot_with_message(request, portfolio)
    messages.success(request, f"{ticker} removed from the portfolio.")
    return redirect("portfolio-detail", portfolio_id=portfolio.id)


@login_required
def methodology_page(request: HttpRequest) -> HttpResponse:
    return render(request, "web/methodology.html")


def _latest_analysis_run() -> AnalysisRun | None:
    return latest_serving_analysis_run()


def _opportunity_card(analysis: StockAnalysis) -> dict[str, Any]:
    current_price_band = latest_price_band(analysis.listing)
    assessment = assess_opportunity(
        analysis,
        price_band=current_price_band,
    )
    (
        long_horizon_blocked,
        long_horizon_band_reason,
        long_horizon_gates,
    ) = _long_horizon_band_state(
        listing=analysis.listing,
        current_price_band=current_price_band,
    )
    return {
        "analysis": analysis,
        "opportunity": assessment,
        "price_band": current_price_band,
        "long_horizon_blocked": long_horizon_blocked,
        "long_horizon_band_reason": long_horizon_band_reason,
        "long_horizon_gates": long_horizon_gates,
    }


def _long_horizon_band_state(
    *,
    listing: Listing,
    current_price_band: PriceBandAssessment | None,
) -> tuple[bool, str, tuple[str, ...]]:
    if current_price_band is not None and current_price_band.blocks_long_horizon:
        return (
            True,
            "Under-$10 long-horizon activation gates are not yet available.",
            UNDER_10_LONG_HORIZON_GATES,
        )
    if listing.currency.upper() == PRICE_BAND_CURRENCY and current_price_band is None:
        return (
            True,
            "No valid latest persisted USD close is available to apply the "
            "guarded price-band policy.",
            (),
        )
    return False, "", ()


def _analyses_in_price_band(
    analyses: QuerySet[StockAnalysis],
    definition: PriceBandDefinition,
) -> QuerySet[StockAnalysis]:
    lookups: dict[str, Any] = {
        "listing__currency__iexact": PRICE_BAND_CURRENCY,
    }
    minimum_lookup = "gte" if definition.minimum_inclusive else "gt"
    lookups[f"listing__latest_market_data__close__{minimum_lookup}"] = definition.minimum
    if definition.maximum is not None:
        lookups["listing__latest_market_data__close__lt"] = definition.maximum
    return analyses.filter(**lookups)


def _analyses_without_usd_price_band(
    analyses: QuerySet[StockAnalysis],
) -> QuerySet[StockAnalysis]:
    return analyses.filter(
        listing__currency__iexact=PRICE_BAND_CURRENCY,
        listing__latest_market_data__isnull=True,
    )


def _analyses_outside_usd_price_band_policy(
    analyses: QuerySet[StockAnalysis],
) -> QuerySet[StockAnalysis]:
    return analyses.exclude(listing__currency__iexact=PRICE_BAND_CURRENCY)


def _filter_analyses(
    analyses: QuerySet[StockAnalysis],
    filters: dict[str, Any],
) -> QuerySet[StockAnalysis]:
    query = filters["q"].strip()
    region = filters["region"]
    recommendation = filters["recommendation"]
    risk = filters["risk"]
    price_band = filters["price_band"]
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
    if price_band in PRICE_BANDS_BY_SLUG:
        analyses = _analyses_in_price_band(
            analyses,
            PRICE_BANDS_BY_SLUG[price_band],
        )
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
    deposit_form: PortfolioDepositForm,
    plan_confirmation_form: PortfolioPlanConfirmationForm,
    plan_confirmation_error: str,
) -> dict[str, Any]:
    valuation = calculate_portfolio_valuation(portfolio)
    listing_ids = [position.holding.listing_id for position in valuation.positions]
    latest_by_listing: dict[UUID, StockAnalysis] = {}
    latest_run = _latest_analysis_run()
    if latest_run is not None:
        analyses = (
            StockAnalysis.objects.filter(
                listing_id__in=listing_ids,
                run=latest_run,
                listing__security__security_type__in=(
                    Security.SecurityType.COMMON_STOCK,
                    Security.SecurityType.ADR,
                ),
            )
            .select_related("listing__latest_market_data")
            .order_by("-pk")
        )
        for persisted_analysis in analyses:
            latest_by_listing[persisted_analysis.listing_id] = persisted_analysis
    position_cards = []
    for position in valuation.positions:
        latest_analysis = latest_by_listing.get(position.holding.listing_id)
        is_etf = position.holding.listing.security.security_type == Security.SecurityType.ETF
        current_price_band = None if is_etf else latest_price_band(position.holding.listing)
        position_cards.append(
            {
                "position": position,
                "analysis": latest_analysis,
                "price_band": current_price_band,
                "is_etf": is_etf,
                "opportunity": (
                    assess_opportunity(
                        latest_analysis,
                        price_band=current_price_band,
                    )
                    if latest_analysis is not None
                    else None
                ),
            }
        )
    snapshots = portfolio_snapshot_series(portfolio)
    contribution_plan = (
        preview_monthly_contribution_plan(portfolio)
        if portfolio.archived_at is None and not portfolio.is_model_portfolio
        else None
    )
    if contribution_plan is not None and (
        not plan_confirmation_form.is_bound or plan_confirmation_error
    ):
        plan_confirmation_form = PortfolioPlanConfirmationForm(
            plan_hash=contribution_plan.plan_hash,
        )
    context = {
        "portfolio": portfolio,
        "portfolio_form": portfolio_form,
        "holding_form": holding_form,
        "deposit_form": deposit_form,
        "plan_confirmation_form": plan_confirmation_form,
        "plan_confirmation_error": plan_confirmation_error,
        "valuation": valuation,
        "position_cards": position_cards,
        "snapshots": list(reversed(snapshots)),
        "first_snapshot": snapshots[0] if snapshots else None,
        "latest_snapshot": snapshots[-1] if snapshots else None,
        "contribution_plan": contribution_plan,
        "contribution_performance": calculate_contribution_performance(
            portfolio,
            valuation=valuation,
        ),
        "deposits": portfolio.deposits.select_related("boundary_snapshot").order_by(
            "-occurred_at",
            "-recorded_at",
        )[:12],
        "performance_baselines": portfolio.performance_baselines.select_related(
            "snapshot"
        ).order_by("-recorded_at", "-id")[:12],
        "plan_executions": portfolio.plan_executions.prefetch_related(
            "purchases__listing",
        ).order_by("-executed_at", "-recorded_at")[:12],
    }
    context.update(_model_portfolio_metrics(portfolio, valuation))
    return context


def _model_portfolio_metrics(
    portfolio: Portfolio,
    valuation: PortfolioValuation,
) -> dict[str, Any]:
    if not portfolio.is_model_portfolio or portfolio.starting_capital is None:
        return {
            "model_return_pct": None,
            "model_return_withheld": False,
            "positive_position_count": None,
        }
    total_value = valuation.total_value
    historical_split_warning = portfolio.snapshots.filter(corporate_action_warnings__gt=0).exists()
    if total_value is None or valuation.corporate_action_warnings > 0 or historical_split_warning:
        model_return_withheld = True
        model_return_pct = None
    else:
        model_return_withheld = False
        model_return_pct = (total_value - portfolio.starting_capital) / portfolio.starting_capital
    positive_position_count = sum(
        position.market_value is not None and position.market_value > position.cost_basis
        for position in valuation.positions
    )
    return {
        "model_return_pct": model_return_pct,
        "model_return_withheld": model_return_withheld,
        "model_return_withheld_reason": (
            "A current or historical split warning makes the frozen quantity basis unreliable."
            if valuation.corporate_action_warnings > 0 or historical_split_warning
            else ""
        ),
        "positive_position_count": positive_position_count,
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

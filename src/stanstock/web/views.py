from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from functools import lru_cache
from http import HTTPStatus
from typing import Any, TypeGuard, cast
from uuid import UUID, uuid4

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Avg, Count, Q, QuerySet
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from stanstock.core.launchd import SCHEDULE_TIME_LABEL, launch_agent_status
from stanstock.core.models import JobRun
from stanstock.core.services import system_status
from stanstock.data.etfs import (
    INVESTABLE_US_ETF_MIC,
    INVESTABLE_US_ETF_SYMBOL,
    build_etf_overview,
)
from stanstock.data.fx import DEFAULT_MAX_CARRY_DAYS
from stanstock.data.models import (
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    LatestMarketData,
    Listing,
    ProviderRecord,
    Region,
    Security,
)
from stanstock.data.provider_policy import (
    CAPABILITY_NO_REVIEWED_SOURCE,
    CAPABILITY_PLAN_NOT_ENTITLED,
    CAPABILITY_UNAVAILABLE,
    SPLIT_EVENT_CAPABILITY,
    TWELVE_DATA_PROVIDER,
)
from stanstock.data.sec_config import load_sec_fundamentals_config
from stanstock.data.sec_fundamentals import MAX_ANNUAL_DAYS, MIN_ANNUAL_DAYS
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
    DECISION_TARGET_DATE_BASIS,
    PRICE_BAND_CURRENCY,
    PRICE_BAND_POLICY_VERSION,
    PRICE_BANDS,
    PRICE_BANDS_BY_SLUG,
    UNDER_10_AVAILABLE_FOUNDATIONS,
    UNDER_10_BAND,
    UNDER_10_RELEASED_SHADOW_DIAGNOSTICS,
    UNDER_10_SHADOW_DISCLOSURE,
    UNDER_10_UNRELEASED_ACTIVATION_CONTROLS,
    PriceBandAssessment,
    PriceBandDefinition,
    classify_price_band,
    latest_price_band,
)
from stanstock.research.indicators import (
    DOLLAR_VOLUME_DROPPED_ROWS,
    DOLLAR_VOLUME_DUPLICATE_SESSIONS,
    DOLLAR_VOLUME_INSUFFICIENT_SESSIONS,
    DOLLAR_VOLUME_INVALID_CLOSE,
    DOLLAR_VOLUME_INVALID_SESSION_DATES,
    DOLLAR_VOLUME_INVALID_VOLUME,
    DOLLAR_VOLUME_MISSING_COLUMNS,
    DOLLAR_VOLUME_NONFINITE_MEDIAN,
    DOLLAR_VOLUME_NONFINITE_PRODUCT,
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
)
from stanstock.research.reporting import (
    canonical_reportable_prediction_filter,
    reportable_prediction_filter,
)
from stanstock.research.service import under10_assessment_matches_persisted_evidence
from stanstock.research.under10 import (
    LIQUIDITY_BASIS_INCOMPATIBLE,
    LIQUIDITY_COMPUTED,
    LIQUIDITY_FUTURE_PRICE_SESSION,
    LIQUIDITY_PRICE_PROVENANCE_UNAVAILABLE,
    LIQUIDITY_STALE_PRICE_EVIDENCE,
    LIQUIDITY_WITHHELD,
    MONETARY_PLACES,
    REASON_CURRENT_ASSETS_BELOW_LIABILITIES,
    REASON_NEAR_TERM_DEBT_EXCEEDS_CASH,
    REASON_NEGATIVE_FREE_CASH_FLOW,
    REASON_RUNWAY_BELOW_MINIMUM_QUARTERS,
    REFERENCE_CLOSE_PLACES,
    REPORTED_PLACES,
    RUNWAY_COMPUTED,
    RUNWAY_NOT_APPLICABLE,
    RUNWAY_WITHHELD,
    SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION,
    SOLVENCY_ELEVATED_OBLIGATION_RISK,
    SOLVENCY_INSUFFICIENT_EVIDENCE,
    SOLVENCY_NO_ADVERSE_EVIDENCE,
    UNDER10_ACTIVATED,
    UNDER10_ACTIVATION_ELIGIBLE,
    UNDER10_ASSET_REFERENCE_KEYS,
    UNDER10_LIQUIDITY_METRIC,
    UNDER10_LIQUIDITY_SESSIONS,
    UNDER10_MAX_METRIC_AGE_DAYS,
    UNDER10_MAX_PRICE_STALENESS_DAYS,
    UNDER10_MIN_RUNWAY_QUARTERS,
    UNDER10_NEW_ALLOCATION_PERCENT,
    UNDER10_REQUIRED_ADJUSTMENT,
    UNDER10_REQUIRED_INTERVAL,
    UNDER10_REQUIRED_RETURN_DEFINITION,
    UNDER10_SCHEMA_VERSION,
    UNDER10_SHADOW_POLICY_VERSION,
    UNDER10_VOLUME_BASIS,
    under10_assessment_hash,
    under10_blocking_reasons,
    under10_inactive_gates,
    under10_policy_hash,
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

STOCK_DETAIL_PREDICTIONS_PER_PAGE = 30


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
    persisted_analyses = (
        list(
            StockAnalysis.objects.filter(run=latest_run)
            .select_related(
                "listing__security__company",
                "listing__latest_market_data",
            )
            .order_by("-overall_score")[:10]
        )
        if latest_run
        else []
    )
    medium_panel = (
        DataAsset.objects.filter(
            kind="medium_forecast_panel",
            subject=str(latest_run.pk),
        )
        .order_by("-available_at")
        .first()
        if latest_run is not None
        else None
    )
    sec_facts = FundamentalFact.objects.filter(provider="sec")
    sec_facts_with_filing_evidence = sec_facts.filter(
        evidence_links__role=FundamentalFactEvidence.Role.FILING
    ).distinct()
    sec_coverage = {
        "mapped_companies": Listing.objects.filter(
            region=Region.US,
            is_active=True,
            security__security_type__in=(
                Security.SecurityType.COMMON_STOCK,
                Security.SecurityType.ADR,
            ),
        )
        .exclude(security__company__cik="")
        .values("security__company_id")
        .distinct()
        .count(),
        "companies_with_facts": sec_facts_with_filing_evidence.values("company_id")
        .distinct()
        .count(),
        "facts": sec_facts_with_filing_evidence.count(),
        "fact_revisions": sec_facts.count(),
        "classifications": CompanyClassificationObservation.objects.filter(
            provider="sec",
            scheme="sec_sic",
        ).count(),
        "latest_asset": DataAsset.objects.filter(provider="sec").order_by("-retrieved_at").first(),
    }
    context = {
        "components": components,
        "system_ok": all(bool(component["ok"]) for component in components),
        "data_mode": data_mode,
        "data_mode_label": data_mode_label(data_mode, data_providers),
        "opportunities": [_opportunity_card(analysis) for analysis in persisted_analyses],
        "demo_opportunities": (
            DEMO_OPPORTUNITIES if settings.DEMO_MODE and not persisted_analyses else []
        ),
        "latest_run": latest_run,
        "providers": ProviderRecord.objects.order_by("provider"),
        "recent_jobs": JobRun.objects.order_by("-started_at")[:5],
        "scheduler": _scheduler_status(),
        "medium_panel": medium_panel,
        "sec_coverage": sec_coverage,
        "prediction_count": Prediction.objects.filter(analysis__run=latest_run).count()
        if latest_run
        else 0,
        "decision_matured_count": (
            PredictionOutcome.objects.filter(
                prediction__analysis__run=latest_run,
                prediction__evidence_role=Prediction.EvidenceRole.DECISION,
                status=PredictionOutcome.Status.MATURED,
            ).count()
            if latest_run
            else 0
        ),
        "advisory_matured_count": (
            PredictionOutcome.objects.filter(
                prediction__analysis__run=latest_run,
                prediction__evidence_role=Prediction.EvidenceRole.ADVISORY,
                status=PredictionOutcome.Status.MATURED,
                prediction__base_return__isnull=False,
            ).count()
            if latest_run
            else 0
        ),
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
            "schedule_matches": False,
            "installed_schedule_label": None,
            "expected_schedule_label": SCHEDULE_TIME_LABEL,
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
            "under_10_available_foundations": UNDER_10_AVAILABLE_FOUNDATIONS,
            "under_10_released_shadow_diagnostics": (UNDER_10_RELEASED_SHADOW_DIAGNOSTICS),
            "under_10_unreleased_activation_controls": (UNDER_10_UNRELEASED_ACTIVATION_CONTROLS),
            "under_10_shadow_disclosure": UNDER_10_SHADOW_DISCLOSURE,
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
        .select_related(
            "run",
            "listing__security__company",
            "listing__latest_market_data",
        )
        .order_by("-run__generated_at")
        .first()
    )
    prediction_queryset = (
        Prediction.objects.filter(listing=listing)
        .only(
            "id",
            "generated_at",
            "horizon",
            "evidence_role",
            "recommendation",
            "target_date",
            "bear_return",
            "base_return",
            "bull_return",
            "model_version",
        )
        .order_by("-generated_at", "horizon", "pk")
    )
    prediction_page = Paginator(
        prediction_queryset,
        STOCK_DETAIL_PREDICTIONS_PER_PAGE,
    ).get_page(request.GET.get("prediction_page"))
    current_price_band = latest_price_band(listing)
    (
        long_horizon_blocked,
        long_horizon_band_reason,
        long_horizon_available_foundations,
        long_horizon_unreleased_activation_controls,
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
            "long_horizon_available_foundations": long_horizon_available_foundations,
            "long_horizon_unreleased_activation_controls": (
                long_horizon_unreleased_activation_controls
            ),
            "under_10_released_shadow_diagnostics": (UNDER_10_RELEASED_SHADOW_DIAGNOSTICS),
            "under_10_shadow_disclosure": UNDER_10_SHADOW_DISCLOSURE,
            "under10_panel": _under10_panel(analysis, current_price_band=current_price_band),
            "prediction_page": prediction_page,
            "predictions": prediction_page.object_list,
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
    evidence_role = request.GET.get("evidence_role", "")
    if horizon in Prediction.Horizon.values:
        predictions = predictions.filter(horizon=horizon)
    if evidence_role in Prediction.EvidenceRole.values:
        predictions = predictions.filter(evidence_role=evidence_role)
    if recommendation in Recommendation.values:
        predictions = predictions.filter(
            evidence_role=Prediction.EvidenceRole.DECISION,
            recommendation=recommendation,
        )

    displayed_predictions = list(predictions[:100])
    prediction_cards = [{"prediction": prediction} for prediction in displayed_predictions]
    return render(
        request,
        "web/predictions.html",
        {
            "prediction_cards": prediction_cards,
            "result_count": predictions.count(),
            "horizons": Prediction.Horizon.choices,
            "recommendations": Recommendation.choices,
            "evidence_roles": Prediction.EvidenceRole.choices,
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
    all_outcomes = PredictionOutcome.objects.select_related(
        "prediction__listing__security__company"
    )
    outcomes = all_outcomes.filter(prediction__evidence_role=Prediction.EvidenceRole.DECISION)
    matured = outcomes.filter(
        status=PredictionOutcome.Status.MATURED,
        actual_return__isnull=False,
    )
    # Built once, before any status/maturity/application-specific filter, so
    # every downstream decision/advisory cohort below counts each exact
    # market observation -- (listing, target_date, horizon, evidence_role,
    # method_version, config_hash, price_provider) -- exactly once, from its
    # earliest reportable issuance. Later valid observed reissues for the
    # same observation remain visible in the immutable prediction ledger and
    # per-run status surfaces (prediction history, latest-run status) but are
    # excluded from this canonical base.
    canonical_reportable_outcomes = all_outcomes.filter(
        canonical_reportable_prediction_filter("prediction__")
    )
    canonical_decision_outcomes = canonical_reportable_outcomes.filter(
        prediction__evidence_role=Prediction.EvidenceRole.DECISION
    )
    reportable_matured = canonical_decision_outcomes.filter(
        status=PredictionOutcome.Status.MATURED,
        actual_return__isnull=False,
    )
    latest_method_prediction = (
        Prediction.objects.filter(
            reportable_prediction_filter(""),
            evidence_role=Prediction.EvidenceRole.DECISION,
        )
        .select_related("analysis__run")
        .order_by("-generated_at", "-id")
        .first()
    )
    current_method_matured = reportable_matured.none()
    current_method_outcomes = canonical_decision_outcomes.none()
    current_config_version = ""
    current_method_version = ""
    current_config_hash = ""
    current_price_provider = ""
    if latest_method_prediction is not None:
        current_config_version = latest_method_prediction.analysis.run.config_version
        current_method_version = latest_method_prediction.method_version
        current_config_hash = latest_method_prediction.config_hash
        current_price_provider = latest_method_prediction.price_provider
        method_filter = Q(
            prediction__method_version=current_method_version,
            prediction__config_hash=current_config_hash,
            prediction__price_provider=current_price_provider,
        )
        current_method_matured = reportable_matured.filter(method_filter)
        current_method_outcomes = canonical_decision_outcomes.filter(method_filter)

    summary = current_method_matured.aggregate(
        sample_count=Count("prediction"),
        benchmark_sample_count=Count("benchmark_return"),
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
            "method_version": current_method_version,
            "config_hash": current_config_hash,
            "price_provider": current_price_provider,
            "positive_rate": (
                Decimal(positive_count) / Decimal(sample_count) if sample_count else None
            ),
            "directional_accuracy": (
                Decimal(successful_count) / Decimal(assessed_count) if assessed_count else None
            ),
            "sufficient_sample": sample_count >= 30,
            "unresolved_count": current_method_outcomes.filter(
                status=PredictionOutcome.Status.UNRESOLVED,
            ).count(),
            "corporate_event_count": current_method_outcomes.filter(
                status=PredictionOutcome.Status.CORPORATE_EVENT,
            ).count(),
            "research_matured_count": matured.exclude(
                reportable_prediction_filter("prediction__")
            ).count(),
            "method_count": reportable_matured.values(
                "prediction__method_version",
                "prediction__config_hash",
                "prediction__price_provider",
            )
            .distinct()
            .count(),
        }
    )
    groups = (
        reportable_matured.values(
            "prediction__method_version",
            "prediction__config_hash",
            "prediction__price_provider",
            "prediction__evidence_grade",
            "prediction__horizon",
            "prediction__recommendation",
        )
        .annotate(
            sample_count=Count("prediction"),
            benchmark_sample_count=Count("benchmark_return"),
            mean_return=Avg("actual_return"),
            mean_benchmark_return=Avg("benchmark_return"),
        )
        .order_by(
            "prediction__method_version",
            "prediction__config_hash",
            "prediction__price_provider",
            "prediction__horizon",
            "prediction__recommendation",
        )
    )
    reportable_advisory_matured = canonical_reportable_outcomes.filter(
        status=PredictionOutcome.Status.MATURED,
        actual_return__isnull=False,
        prediction__evidence_role=Prediction.EvidenceRole.ADVISORY,
        prediction__base_return__isnull=False,
    )
    raw_advisory_groups = reportable_advisory_matured.values(
        "prediction__method_version",
        "prediction__config_hash",
        "prediction__price_provider",
        "prediction__horizon",
        "prediction__evidence_grade",
    ).annotate(
        sample_count=Count("prediction"),
        direction_sample_count=Count("direction_correct"),
        direction_correct_count=Count(
            "prediction",
            filter=Q(direction_correct=True),
        ),
        interval_sample_count=Count("interval_covered"),
        interval_covered_count=Count(
            "prediction",
            filter=Q(interval_covered=True),
        ),
        signed_error_sample_count=Count("signed_error"),
        mean_signed_error=Avg("signed_error"),
    )
    advisory_groups: list[dict[str, Any]] = []
    for raw_group in raw_advisory_groups.order_by(
        "prediction__method_version",
        "prediction__config_hash",
        "prediction__price_provider",
        "prediction__horizon",
    ):
        group: dict[str, Any] = dict(raw_group)
        direction_sample_count = int(group["direction_sample_count"])
        interval_sample_count = int(group["interval_sample_count"])
        signed_error_sample_count = int(group["signed_error_sample_count"])
        group["direction_accuracy"] = (
            Decimal(group["direction_correct_count"]) / Decimal(direction_sample_count)
            if direction_sample_count >= 30
            else None
        )
        group["interval_coverage"] = (
            Decimal(group["interval_covered_count"]) / Decimal(interval_sample_count)
            if interval_sample_count >= 30
            else None
        )
        if signed_error_sample_count < 30:
            group["mean_signed_error"] = None
        advisory_groups.append(group)
    return render(
        request,
        "web/performance.html",
        {
            "summary": summary,
            "groups": groups,
            "advisory_groups": advisory_groups,
            "advisory_matured_count": reportable_advisory_matured.count(),
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
        long_horizon_available_foundations,
        long_horizon_unreleased_activation_controls,
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
        "long_horizon_available_foundations": long_horizon_available_foundations,
        "long_horizon_unreleased_activation_controls": (
            long_horizon_unreleased_activation_controls
        ),
    }


def _long_horizon_band_state(
    *,
    listing: Listing,
    current_price_band: PriceBandAssessment | None,
) -> tuple[bool, str, tuple[str, ...], tuple[str, ...]]:
    if current_price_band is not None and current_price_band.blocks_long_horizon:
        return (
            True,
            "Under-$10 long-horizon forecast remains unavailable; joint review "
            "and candidate-specific eligibility remain outstanding.",
            UNDER_10_AVAILABLE_FOUNDATIONS,
            UNDER_10_UNRELEASED_ACTIVATION_CONTROLS,
        )
    if listing.currency.upper() == PRICE_BAND_CURRENCY and current_price_band is None:
        return (
            True,
            "No valid latest persisted USD close is available to apply the "
            "guarded price-band policy.",
            (),
            (),
        )
    return False, "", (), ()


# ---------------------------------------------------------------------------
# Under-$10 shadow diagnostic panel (detail page only)
# ---------------------------------------------------------------------------

UNDER10_SOLVENCY_LABELS = {
    "no_adverse_evidence_observed": "No adverse evidence observed",
    "elevated_obligation_risk": "Elevated obligation risk",
    "adverse_near_term_obligation": "Adverse near-term obligation",
    "insufficient_evidence": "Insufficient evidence",
}

UNDER10_INPUT_LABELS = (
    ("cash_and_equivalents", "Cash and equivalents"),
    ("near_term_debt", "Near-term debt"),
    ("current_assets", "Current assets"),
    ("current_liabilities", "Current liabilities"),
    ("current_ratio", "Current ratio (reported only)"),
    ("free_cash_flow", "Free cash flow"),
)

UNDER10_SPLIT_EXPLANATIONS = {
    "provider_plan_not_entitled": (
        "The recorded Twelve Data Basic plan is not entitled to a corporate-actions "
        "split feed. A different plan alone would not supply a reviewed split source: "
        "a source-capability and licensing review must pass first."
    ),
    "no_reviewed_corporate_actions_source": (
        "No reviewed corporate-actions source is integrated for this provider, so "
        "split and reverse-split events cannot be verified. Nothing is inferred from "
        "adjusted prices, share counts, or SEC facts."
    ),
}

UNDER10_UNSUPPORTED_MESSAGE = (
    "Withheld - this analysis carries an Under-$10 assessment this build cannot "
    "read. Nothing is inferred from it."
)

UNDER10_NOT_ASSESSED_MESSAGE = (
    "Not assessed for this analysis. The Under-$10 shadow diagnostics are recorded "
    "only on newly created qualifying analyses; existing analyses were not "
    "backfilled, so an absent assessment is not a failed one."
)


def _under10_panel(
    analysis: StockAnalysis | None,
    *,
    current_price_band: PriceBandAssessment | None,
) -> dict[str, Any] | None:
    """Read-only rendering state for the Under-$10 shadow diagnostic panel.

    Cheap schema, checksum, and parent bindings run before any evidence read.
    A structurally valid payload is then independently replayed from its
    immutable decision-prediction, price-asset, and SEC-fact evidence. The
    replay is only a predicate: recorded values are rendered unchanged,
    replayed values are never returned, and no provider is contacted or row
    mutated. A recorded assessment stays visible even after the current
    market band moves, and a later Under-$10 band never manufactures one for
    an older analysis.

    Key *absence* means "not assessed" (a historical run predating this
    policy, or a non-candidate). A key that is *present* but ``None``, the
    wrong type, or fails schema/type validation is never treated the same
    way: it is "unsupported", because something was recorded and this reader
    cannot honestly say what it means.
    """
    if analysis is None:
        return None
    data_quality = analysis.data_quality
    # NB-2: `data_quality` is a schema-less `JSONField`; a malformed row
    # (``None``, an int, a list, a string, a bool) must never reach the
    # ``in``/``.get`` calls below, which would otherwise raise instead of
    # rendering the same "something was recorded and this reader cannot
    # honestly say what it means" outcome every other malformed shape gets.
    if not isinstance(data_quality, dict):
        return _unsupported_under10_panel(analysis)
    has_recorded_key = "under10_assessment" in data_quality
    decision_band = _decision_run_price_band(analysis)
    decision_is_under_10 = decision_band is not None and decision_band.slug == UNDER_10_BAND
    current_is_under_10 = (
        current_price_band is not None and current_price_band.slug == UNDER_10_BAND
    )
    if not has_recorded_key:
        if not decision_is_under_10 and not current_is_under_10:
            return None
        return {
            "state": "not_assessed",
            "headline": UNDER10_NOT_ASSESSED_MESSAGE,
            "decision_target_date": analysis.run.target_date,
            "decision_data_cutoff": analysis.run.data_cutoff,
        }
    recorded = data_quality.get("under10_assessment")
    # C1: both the UUID and content checksum of the identical `DataAsset`
    # reference `build_under10_assessment` was actually given for this
    # decision -- cross-derived entirely from the already-loaded
    # `data_quality` JSON blob (set once, at generation time, by
    # `compute_listing_analysis`), so reading it here is free; never a new
    # query and never the payload's own unverified claim about itself.
    expected_price_asset = _expected_price_asset_reference(data_quality)
    if not _is_valid_under10_payload(
        recorded,
        expected_listing_id=str(analysis.listing_id),
        expected_target_date=analysis.run.target_date,
        expected_data_cutoff=analysis.run.data_cutoff,
        expected_reference_close=analysis.current_price,
        expected_currency=analysis.listing.currency,
        expected_reference_close_in_band=decision_is_under_10,
        expected_price_asset=expected_price_asset,
        expected_code_revision=analysis.run.code_revision,
    ):
        return _unsupported_under10_panel(analysis)
    # `_is_valid_under10_payload` already proved `recorded` is a well-formed
    # dict with these exact keys; narrowed explicitly here only because a
    # plain ``bool`` return cannot itself narrow `recorded`'s static type.
    recorded = cast(dict[str, Any], recorded)
    if not under10_assessment_matches_persisted_evidence(
        analysis=analysis,
        recorded=recorded,
    ):
        return _unsupported_under10_panel(analysis)
    solvency = recorded["solvency"]
    liquidity = recorded["liquidity"]
    split = recorded["split_verification"]
    return {
        "state": "recorded",
        "policy_version": recorded.get("policy_version"),
        "policy_hash": recorded.get("policy_hash"),
        "assessment_hash": recorded.get("assessment_hash"),
        # The validator proved these stored mirrors exactly match the
        # authoritative v1 policy. Render from that policy anyway, so the
        # display has one owner and can never make the stored copy
        # authoritative.
        "activated": UNDER10_ACTIVATED,
        "activation_eligible": UNDER10_ACTIVATION_ELIGIBLE,
        # Likewise, only an exactly canonical stored 0% payload reaches
        # here, while the rendered value still comes from policy.
        "new_allocation_percent": UNDER10_NEW_ALLOCATION_PERCENT,
        "decision_target_date": analysis.run.target_date,
        "decision_data_cutoff": analysis.run.data_cutoff,
        "reference_close": _under10_evaluated_for(recorded, "reference_close"),
        "solvency": _under10_solvency_panel(solvency),
        "liquidity": _under10_liquidity_panel(liquidity),
        "split": _under10_split_panel(split),
    }


#: Recognized solvency input keys, reused so the schema validator below and
#: the rendered field list can never silently diverge from one another.
_UNDER10_SOLVENCY_INPUT_KEYS = tuple(key for key, _label in UNDER10_INPUT_LABELS)
_UNDER10_PERIOD_KEYS = ("instant_date", "duration_start", "duration_end", "duration_basis")
#: Fields whose observed (possibly incompatible) value the generator
#: faithfully preserves verbatim from source-asset metadata when withheld --
#: never type-constrained to ``str``/``None`` here, because the generator
#: itself never constrains what it copies from there. ``volume_basis`` is
#: never observed-copied (always the one fixed constant), so it stays out of
#: this set and is checked strictly on its own below.
_UNDER10_OBSERVED_BASIS_KEYS = ("interval", "adjustment", "return_definition")
_UNDER10_DURATION_BASIS_VALUES = ("ttm", "annual")
#: The exact split-only price-basis metadata every *computed* liquidity
#: result must carry (`_build_liquidity` only ever reaches `LIQUIDITY_COMPUTED`
#: after confirming the source asset's observed metadata matches this exactly).
_UNDER10_COMPATIBLE_PRICE_BASIS = {
    "interval": UNDER10_REQUIRED_INTERVAL,
    "adjustment": UNDER10_REQUIRED_ADJUSTMENT,
    "return_definition": UNDER10_REQUIRED_RETURN_DEFINITION,
    "volume_basis": UNDER10_VOLUME_BASIS,
}
#: The exact empty basis `_build_liquidity` always reports when no anchor
#: was ever confirmed (``price_asset`` genuinely absent, or a mismatched
#: one): nothing was observed for the *correct* asset, so every field but
#: the fixed ``volume_basis`` constant stays ``None`` -- never a
#: compatible-looking string, which the generator cannot have observed in
#: this branch at all.
_UNDER10_EMPTY_PRICE_BASIS = {
    "interval": None,
    "adjustment": None,
    "return_definition": None,
    "volume_basis": UNDER10_VOLUME_BASIS,
}
#: Withheld liquidity reasons whose branch never inspects a price window at
#: all -- `_build_liquidity` returns before ever calling
#: `median_dollar_volume` (no anchor confirmed, or the anchor's metadata
#: was incompatible), or `median_dollar_volume` itself returns before any
#: session window could be identified (missing columns, unparseable
#: session dates, or a duplicate session identity). Every one of these
#: reports a fully null ``sessions_used``/``first_session``/``last_session``
#: triple; never a partial pair.
_LIQUIDITY_NULL_WINDOW_REASONS = frozenset(
    {
        LIQUIDITY_PRICE_PROVENANCE_UNAVAILABLE,
        LIQUIDITY_BASIS_INCOMPATIBLE,
        DOLLAR_VOLUME_MISSING_COLUMNS,
        DOLLAR_VOLUME_INVALID_SESSION_DATES,
        DOLLAR_VOLUME_DUPLICATE_SESSIONS,
    }
)
#: Withheld liquidity reasons only ever reached after `median_dollar_volume`
#: has already identified a genuine (1..252-session) observed window --
#: an invalid close/volume value, a nonfinite dollar-volume product, rows
#: dropped during preparation, or a confirmed-anchor result whose last
#: session is stale or in the future.
_LIQUIDITY_WINDOWED_REASONS = frozenset(
    {
        DOLLAR_VOLUME_INVALID_CLOSE,
        DOLLAR_VOLUME_INVALID_VOLUME,
        DOLLAR_VOLUME_NONFINITE_PRODUCT,
        DOLLAR_VOLUME_DROPPED_ROWS,
        LIQUIDITY_STALE_PRICE_EVIDENCE,
        LIQUIDITY_FUTURE_PRICE_SESSION,
    }
)
#: `nonfinite_median` is only ever reached once `median_dollar_volume` has
#: already confirmed a full `UNDER10_LIQUIDITY_SESSIONS`-session window
#: (the preceding `window.height < sessions` check would otherwise have
#: already returned `insufficient_sessions` first).
_LIQUIDITY_FULL_WINDOW_REASONS = frozenset({DOLLAR_VOLUME_NONFINITE_MEDIAN})
#: Every recognized withheld liquidity reason `_build_liquidity`/
#: `median_dollar_volume` can actually produce. An unrecognized reason is
#: never trusted enough to render as a legitimate withholding, regardless
#: of how plausible its window descriptors look.
_RECOGNIZED_LIQUIDITY_WITHHELD_REASONS = (
    _LIQUIDITY_NULL_WINDOW_REASONS
    | {DOLLAR_VOLUME_INSUFFICIENT_SESSIONS}
    | _LIQUIDITY_WINDOWED_REASONS
    | _LIQUIDITY_FULL_WINDOW_REASONS
)
#: Recognized withheld reasons whose *stored fields* are enough to prove
#: (not merely assert) that `_build_liquidity` already confirmed a
#: compatible anchor and metadata: every reason with its own genuine,
#: branch-specific, hard-to-forge session-window evidence attached
#: (`_LIQUIDITY_WINDOWED_REASONS`, `_LIQUIDITY_FULL_WINDOW_REASONS`, and
#: `insufficient_sessions`, whose window is real, not merely absent).
#:
#: Deliberately narrower than "every reason `_build_liquidity` happens to
#: reach after its own anchor/metadata checks pass": `missing_price_columns`
#: / `invalid_session_dates` / `duplicate_sessions` are genuinely reached
#: there too, but -- like `price_provenance_unavailable` and
#: `basis_incompatible` -- always carry a fully null session-window triple.
#: A currency-only `basis_incompatible` mismatch (the one field never
#: separately stored anywhere in the payload) can produce an otherwise
#: identical valid-anchor / compatible-looking-basis / null-window shape,
#: so a reason string alone from that null-window group is not sufficient,
#: stored evidence that basis was actually confirmed -- only a reason
#: carrying its own real window descriptors is.
_LIQUIDITY_POST_BASIS_REASONS = (
    _LIQUIDITY_WINDOWED_REASONS
    | _LIQUIDITY_FULL_WINDOW_REASONS
    | {DOLLAR_VOLUME_INSUFFICIENT_SESSIONS}
)
#: The widest magnitude any genuinely generated numeric/Decimal-string field
#: could ever contain. Not a business threshold: it is the boundary at which
#: a raw JSON ``int`` can no longer be converted to a ``float`` at all
#: (`math.isfinite` raises `OverflowError` past this point instead of
#: returning ``False``), so it is the natural, technical ceiling for "does
#: this even look like a number a real generator could have produced" rather
#: than an arbitrary dollar figure. Used only by `_is_finite_number` below,
#: for the raw (non-string) liquidity ``value`` field.
_MAX_FINITE_MAGNITUDE = sys.float_info.max

#: `under10.py`'s own (private) arithmetic-context precision. Every
#: monetary/ratio/runway/reference-close string is produced by quantizing
#: under exactly this many significant digits (`_quantized_text`'s
#: ``localcontext().prec``); a source value needing more digits than this
#: makes the generator's own ``Decimal.quantize`` raise before it could ever
#: be serialized, so a persisted string can never legitimately exceed it.
_UNDER10_DECIMAL_PRECISION = 64

#: Decimal places for each canonical fixed-point field the generator emits
#: (`MONETARY_PLACES`, `REPORTED_PLACES`, `REFERENCE_CLOSE_PLACES` in
#: `under10.py`). ``_quantized_text`` always formats with Python's ``:f``
#: (plain fixed-point) spec, so the generator never emits scientific
#: notation for any of these fields; a reader that accepts exponent
#: notation anyway would admit magnitudes/precisions no real payload could
#: contain (e.g. ``"1e-100000000"``).
_MONETARY_DECIMAL_PLACES = 8
_REPORTED_DECIMAL_PLACES = 4
_REFERENCE_CLOSE_DECIMAL_PLACES = 6

#: The exact quantization exponent for each canonical fixed-point field --
#: reused directly from `under10.py`'s own public `MONETARY_PLACES`/
#: `REPORTED_PLACES`/`REFERENCE_CLOSE_PLACES` (never redeclared locally, so
#: the two can never silently drift), for *recomputing* a derived claim
#: from recorded operands (`Decimal.quantize` takes the exponent itself,
#: not a plain digit count) -- never for the lexical acceptance check
#: above, which stays keyed by digit count. `test_reader_decimal_scales_
#: match_the_policy_document_exactly` (in the matrix test module) proves
#: these -- and the digit counts and precision above -- against the
#: generator's own published `under10_policy_document()["serialization"]`
#: directly, so any future drift fails clearly there rather than only
#: through the large corruption matrix.
_MONETARY_QUANTUM = MONETARY_PLACES
_REPORTED_QUANTUM = REPORTED_PLACES
_REFERENCE_CLOSE_QUANTUM = REFERENCE_CLOSE_PLACES

_CANONICAL_DECIMAL_PATTERNS: dict[int, re.Pattern[str]] = {
    # Python's ``Decimal.__format__("f")`` never emits a leading zero
    # followed by another digit (``Decimal("007")`` formats as ``"7"``,
    # ``Decimal("00")`` as ``"0"``): the integer part is either the single
    # digit ``0`` or a nonzero-leading digit run, never ``"00.00000000"``.
    places: re.compile(rf"-?(0|[1-9][0-9]*)\.[0-9]{{{places}}}\Z")
    for places in (
        _MONETARY_DECIMAL_PLACES,
        _REPORTED_DECIMAL_PLACES,
        _REFERENCE_CLOSE_DECIMAL_PLACES,
    )
}


def _is_strict_int(value: object) -> TypeGuard[int]:
    """``True`` only for a genuine ``int``, never a ``bool`` (``bool`` is an ``int`` subclass)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """``True`` only for a genuine, finite ``int``/``float`` within float range.

    Total: never raises, for any input. A Python ``int`` is always
    mathematically finite (there is no int-shaped NaN/Infinity), but an
    arbitrary-precision integer decoded from persisted JSON can still exceed
    the largest finite ``float`` -- ``math.isfinite`` converts its argument
    to ``float`` internally and raises ``OverflowError`` for such a value
    instead of returning ``False``. The magnitude is bounded against the
    actual ``float`` representable range first, so that conversion is never
    attempted on a value that cannot survive it.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if isinstance(value, int):
        return -_MAX_FINITE_MAGNITUDE <= value <= _MAX_FINITE_MAGNITUDE
    return math.isfinite(value)


def _parse_canonical_decimal(value: object, *, places: int) -> Decimal | None:
    """A stored string as the exact canonical fixed-point form `_quantized_text` emits.

    Total: never raises, for any input. Every monetary/ratio/runway/
    reference-close value the generator emits is formatted with Python's
    ``:f`` (plain fixed-point) spec after quantizing to exactly ``places``
    decimal digits under `under10.py`'s own precision-64 arithmetic context
    -- never scientific/exponent notation, never a bare integer, never a
    different number of decimal digits. Rejecting anything that does not
    match that exact lexical shape (rather than accepting any
    Decimal-parseable string within a generic numeric range) is what refuses
    ``"1e100000000"`` *and* ``"1e-100000000"``/``"-1e-100000000"`` alike: a
    one-sided magnitude ceiling alone would still admit an extreme negative
    exponent, since its absolute value is small.

    A ``bool`` is never a valid representation (even though ``Decimal(True)``
    would otherwise happily succeed). The digit-count bound is
    `_UNDER10_DECIMAL_PRECISION` (64): a source value needing more
    significant digits than that would already have made the generator's own
    ``Decimal.quantize`` raise, so no genuinely generated string can exceed
    it -- this is derived from the generator's own arithmetic context, not
    an arbitrary business threshold.
    """
    if not isinstance(value, str):
        return None
    pattern = _CANONICAL_DECIMAL_PATTERNS.get(places)
    if pattern is None or pattern.match(value) is None:
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    if len(parsed.as_tuple().digits) > _UNDER10_DECIMAL_PRECISION:
        return None
    return parsed


def _recompute_quantized_text(value: Decimal, quantum: Decimal) -> str | None:
    """Mirrors `under10.py`'s own `_quantized_text` exactly: the same
    precision-64 context, ``ROUND_HALF_EVEN`` rounding, and
    ``Decimal.quantize`` call against ``quantum`` -- the exponent-bearing
    value itself (e.g. ``Decimal("0.0001")``), not a plain digit count.

    Total: returns ``None`` instead of raising when ``value`` cannot be
    represented at ``quantum`` under this precision (`Decimal.quantize`
    raises ``InvalidOperation`` past `_UNDER10_DECIMAL_PRECISION`
    significant digits), exactly mirroring `_quantized_text`'s own
    ``ValueError`` refusal. A caller comparing against this result must
    treat ``None`` as "cannot confirm equality", never as a wildcard match.
    """
    try:
        with localcontext() as context:
            context.prec = _UNDER10_DECIMAL_PRECISION
            context.rounding = ROUND_HALF_EVEN
            quantized = value.quantize(quantum, rounding=ROUND_HALF_EVEN)
    except InvalidOperation:
        return None
    return f"{quantized:f}"


def _recompute_current_ratio_text(
    *, current_assets: Decimal | None, current_liabilities: Decimal | None
) -> str | None:
    """Mirrors `under10.py`'s own `_current_ratio` exactly: ``None`` unless
    both operands are present and the denominator is strictly positive,
    else the exact ``assets / liabilities`` quotient quantized to 4 places
    under the same precision-64 context.
    """
    if current_assets is None or current_liabilities is None or current_liabilities <= 0:
        return None
    with localcontext() as context:
        context.prec = _UNDER10_DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        ratio = current_assets / current_liabilities
    if not ratio.is_finite():
        return None
    return _recompute_quantized_text(ratio, _REPORTED_QUANTUM)


def _recompute_runway_quarters_text(*, cash: Decimal, free_cash_flow: Decimal) -> str | None:
    """Mirrors the exact quotient `_build_runway` computes once free cash
    flow is negative and cash is present: ``4 * cash / abs(free_cash_flow)``
    under the same precision-64 context, quantized to 4 places -- including
    a genuine signed negative zero, which quantizes to a signed ``"-0.0000"``
    string, never a plain unsigned ``"0.0000"``.
    """
    with localcontext() as context:
        context.prec = _UNDER10_DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        quarters = (Decimal(4) * cash) / abs(free_cash_flow)
    if not quarters.is_finite():
        return None
    return _recompute_quantized_text(quarters, _REPORTED_QUANTUM)


def _recompute_solvency_classification(
    *,
    cash: Decimal,
    near_term_debt: Decimal,
    current_assets: Decimal,
    current_liabilities: Decimal,
    free_cash_flow: Decimal,
) -> tuple[str, list[str]]:
    """Mirrors `under10.py`'s own `_classify_solvency` exactly: the same four
    booleans, computed under the same precision-64 context, the same
    first-match state partition, and the same ordered reason list (debt,
    assets, free cash flow, runway) -- over already-recorded, already-parsed
    operands, never re-reading evidence or trusting the recorded
    status/reasons as anything but a claim to be proven.

    ``short_runway`` uses the exact unrounded cross-multiplication
    (``4 * cash < minimum_quarters * abs(free_cash_flow)``), never the
    rendered/rounded ``quarters`` string -- a boundary case at exactly the
    minimum must classify identically to the generator regardless of how
    the reported quarters figure happens to round for display.
    """
    with localcontext() as context:
        context.prec = _UNDER10_DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        debt_exceeds_cash = near_term_debt > cash
        assets_below_liabilities = current_assets < current_liabilities
        negative_fcf = free_cash_flow < 0
        short_runway = negative_fcf and (
            Decimal(4) * cash < UNDER10_MIN_RUNWAY_QUARTERS * abs(free_cash_flow)
        )
    reasons: list[str] = []
    if debt_exceeds_cash:
        reasons.append(REASON_NEAR_TERM_DEBT_EXCEEDS_CASH)
    if assets_below_liabilities:
        reasons.append(REASON_CURRENT_ASSETS_BELOW_LIABILITIES)
    if negative_fcf:
        reasons.append(REASON_NEGATIVE_FREE_CASH_FLOW)
    if short_runway:
        reasons.append(REASON_RUNWAY_BELOW_MINIMUM_QUARTERS)
    if debt_exceeds_cash and (assets_below_liabilities or short_runway):
        return SOLVENCY_ADVERSE_NEAR_TERM_OBLIGATION, reasons
    if negative_fcf or debt_exceeds_cash or assets_below_liabilities:
        return SOLVENCY_ELEVATED_OBLIGATION_RISK, reasons
    return SOLVENCY_NO_ADVERSE_EVIDENCE, []


def _is_within_metric_age(*, claimed_date: date, target_date: date) -> bool:
    """Mirrors `under10.py`'s own `_freshness_reason` bound exactly:
    ``0 <= target_date - claimed_date <= UNDER10_MAX_METRIC_AGE_DAYS``."""
    age = (target_date - claimed_date).days
    return 0 <= age <= UNDER10_MAX_METRIC_AGE_DAYS


def _is_within_price_staleness(*, last_session: date, target_date: date) -> bool:
    """Mirrors `under10.py`'s own `_price_staleness_reason`'s non-stale,
    non-future bound exactly:
    ``0 <= target_date - last_session <= UNDER10_MAX_PRICE_STALENESS_DAYS``."""
    age = (target_date - last_session).days
    return 0 <= age <= UNDER10_MAX_PRICE_STALENESS_DAYS


def _is_valid_liquidity_staleness(
    *, reason: str, last_session: date | None, target_date: date
) -> bool:
    """NB-4: `_build_liquidity` always evaluates `_price_staleness_reason`
    over `median_dollar_volume`'s result *before* ever looking at its own
    ``status``/``reason`` -- a stale or future last session unconditionally
    overrides whatever reason that result would otherwise have carried. So
    every recognized withheld reason whose branch reports a real window
    (``last_session`` is not ``None``) other than ``stale_price_evidence``
    itself can only genuinely be reached when that session is *not* stale
    (age at most `UNDER10_MAX_PRICE_STALENESS_DAYS`) -- this includes
    `insufficient_sessions`'s non-empty sub-case, every other windowed
    reason, and the full-window ``nonfinite_median`` reason alike.
    ``future_price_session`` is excluded here: its own future-only bound is
    already enforced by the session window itself
    (`_is_valid_session_window`'s ``require_future_last_session``), and a
    non-future window is never reachable for it at all.
    """
    if last_session is None or reason == LIQUIDITY_FUTURE_PRICE_SESSION:
        return True
    age = (target_date - last_session).days
    if reason == LIQUIDITY_STALE_PRICE_EVIDENCE:
        return age > UNDER10_MAX_PRICE_STALENESS_DAYS
    return age <= UNDER10_MAX_PRICE_STALENESS_DAYS


def _parse_iso_date(value: object) -> date | None:
    """A stored string as a genuine ``date.fromisoformat`` value, or ``None``. Never raises."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_iso_datetime(value: object) -> datetime | None:
    """A stored string as a genuine ``datetime.fromisoformat`` value, or
    ``None``. Never raises -- mirrors `_parse_iso_date` for the one
    datetime-valued field (``evaluated_for.data_cutoff``)."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _expected_under10_policy_hash() -> str:
    """The authoritative ``us-under10-shadow-v1`` policy hash, computed the
    exact same way the generator itself does (`under10_policy_hash` over
    the reviewed SEC fundamentals configuration) -- never a duplicated
    literal here, and never recomputed per request: the configuration is a
    versioned local file, not provider/network/database state, so it is
    loaded and hashed at most once per process, exactly like the
    generator's own cached loader.
    """
    return under10_policy_hash(load_sec_fundamentals_config())


def _is_optional_str(value: object) -> bool:
    return value is None or isinstance(value, str)


def _is_non_empty_str(value: object) -> bool:
    return isinstance(value, str) and value != ""


def _is_valid_price_asset_reference(value: object) -> bool:
    """`{"id": <uuid-string>, "sha256": <64-hex-string>}` -- `_asset_reference`'s exact shape.

    Total: never raises, for any input.
    """
    if not isinstance(value, dict):
        return False
    asset_id = value.get("id")
    sha256 = value.get("sha256")
    if not isinstance(asset_id, str):
        return False
    try:
        UUID(asset_id)
    except ValueError:
        return False
    return isinstance(sha256, str) and re.fullmatch(r"[0-9a-f]{64}", sha256) is not None


def _is_canonical_uuid_text(value: object) -> bool:
    """Whether ``value`` is the lowercase, hyphenated canonical UUID form."""
    if not isinstance(value, str):
        return False
    try:
        parsed = UUID(value)
    except ValueError:
        return False
    return str(parsed) == value


def _is_valid_assessed_fact_ids(value: object) -> bool:
    """A sorted, unique list of canonical immutable fact UUID strings."""
    if not isinstance(value, list) or not all(_is_canonical_uuid_text(item) for item in value):
        return False
    return value == sorted(value) and len(value) == len(set(value))


def _is_valid_assessed_assets(value: object) -> bool:
    """Sorted unique canonical SEC asset references with the exact v1 shape."""
    if not isinstance(value, list):
        return False
    asset_ids: list[str] = []
    expected_keys = set(UNDER10_ASSET_REFERENCE_KEYS)
    for reference in value:
        if not isinstance(reference, dict) or set(reference) != expected_keys:
            return False
        asset_id = reference[UNDER10_ASSET_REFERENCE_KEYS[0]]
        checksum = reference[UNDER10_ASSET_REFERENCE_KEYS[1]]
        if not _is_canonical_uuid_text(asset_id):
            return False
        if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
            return False
        assert isinstance(asset_id, str)
        asset_ids.append(asset_id)
    return asset_ids == sorted(asset_ids) and len(asset_ids) == len(set(asset_ids))


@dataclass(frozen=True, slots=True)
class _ExpectedPriceAssetReference:
    """The parent analysis's own immutable price-asset identity: both the
    `DataAsset` UUID *and* its content checksum, derived from the already-
    loaded ``data_quality["source_assets"]`` list -- never a new query, and
    never the payload's own unverified claim about itself. A UUID-only
    anchor could still be satisfied by a payload that keeps the correct id
    but claims different asset content after recomputing its own checksum;
    binding the checksum too closes that gap.
    """

    id: str
    sha256: str


def _expected_price_asset_reference(
    data_quality: dict[str, Any],
) -> _ExpectedPriceAssetReference | None:
    """The one authoritative price-asset reference this analysis actually
    used, cross-derived entirely from its own already-loaded
    ``data_quality``: ``price_source["asset_id"]`` selects the exact entry
    within ``source_assets`` (the full per-asset provenance list every
    analysis already carries -- the same list `compute_listing_analysis`
    itself searched to build ``price_source`` in the first place), and
    that entry's own recorded ``sha256`` is the authoritative checksum.

    ``None`` (never trusted for binding) whenever the parent's own anchor
    cannot be established unambiguously: a missing/malformed
    ``price_source``, a missing/malformed ``source_assets`` list, zero
    matching entries (a missing parent anchor), more than one matching
    entry (an ambiguous/duplicate id -- `_dedupe_assets` never produces
    this for genuine data, but a forged or corrupted row could), or a
    matching entry whose own reference shape is invalid.
    """
    price_source = data_quality.get("price_source")
    if not isinstance(price_source, dict):
        return None
    asset_id = price_source.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id:
        return None
    source_assets = data_quality.get("source_assets")
    if not isinstance(source_assets, list):
        return None
    matches = [
        asset for asset in source_assets if isinstance(asset, dict) and asset.get("id") == asset_id
    ]
    if len(matches) != 1:
        return None
    matching_asset = matches[0]
    if not _is_valid_price_asset_reference(matching_asset):
        return None
    return _ExpectedPriceAssetReference(id=asset_id, sha256=matching_asset["sha256"])


def _is_bound_price_asset(
    price_asset: object, *, expected: _ExpectedPriceAssetReference | None
) -> bool:
    """C1: a structurally valid anchor must also be *this* analysis's own
    price asset -- both the identical `DataAsset` UUID and its recorded
    content checksum, derived from the parent's own already-loaded
    ``data_quality["source_assets"]`` (never a new query, never the
    payload's own unverified claim about its content). Two genuine,
    unrelated candidates can share a target date, reference close, and
    currency; each still has its own price series and its own asset
    content, so a transplanted or content-altered assessment always
    differs here.
    """
    if not _is_valid_price_asset_reference(price_asset):
        return False
    assert isinstance(price_asset, dict)  # narrowed by the reference check above
    if expected is None:
        return False
    return bool(price_asset["id"] == expected.id and price_asset["sha256"] == expected.sha256)


def _recompute_assessment_hash(recorded: dict[str, Any]) -> str | None:
    """`under10_assessment_hash`, made total: returns ``None`` instead of raising.

    `canonical_json` serializes with ``allow_nan=False`` (raises
    ``ValueError`` for a stray NaN/Infinity float anywhere in the payload)
    and, in general, ``json.dumps`` can raise ``TypeError`` for a
    non-JSON-native value; this reader must never let a malformed stored
    payload turn a checksum recomputation into an uncaught 500.
    """
    try:
        return under10_assessment_hash(recorded)
    except (TypeError, ValueError, OverflowError):
        return None


def _is_valid_assessment_hash(recorded: dict[str, Any]) -> bool:
    """An explicit accidental-corruption gate, not tamper protection.

    ``assessment_hash`` is a recomputation checksum over the rest of the
    payload (see `under10_assessment_hash`'s own docstring): confirming it
    here catches accidental bit-level corruption in storage/retrieval, but
    proves nothing about who computed it or whether the content is
    semantically valid -- every structural/semantic check in this module
    still runs regardless of whether the hash matches, and a matching hash
    on a semantically corrupted payload must still be rejected.
    """
    stored = recorded.get("assessment_hash")
    if not isinstance(stored, str) or re.fullmatch(r"[0-9a-f]{64}", stored) is None:
        return False
    recomputed = _recompute_assessment_hash(recorded)
    return recomputed is not None and recomputed == stored


def _is_valid_under10_payload(
    recorded: object,
    *,
    expected_listing_id: str,
    expected_target_date: date,
    expected_data_cutoff: datetime,
    expected_reference_close: Decimal,
    expected_currency: str,
    expected_reference_close_in_band: bool,
    expected_price_asset: _ExpectedPriceAssetReference | None,
    expected_code_revision: str,
) -> bool:
    """Whether ``recorded`` is a payload this reader recognizes well enough to render.

    A malformed, incomplete, or unrecognized-schema payload must never be
    able to render a favorable-looking state: every nested block is fully
    schema/type-checked here before any of it is trusted, rather than
    letting a missing or wrong-typed field silently fall back to a
    plausible-looking default. Unknown top-level extra keys are not
    rejected on their own: nothing here reads them, and (per
    `under10_assessment_hash`'s own contract) they were already part of
    what the checksum below covers.

    ``expected_reference_close``/``expected_currency`` bind ``evaluated_for``
    to the already-loaded, immutable decision-run context (``analysis.
    current_price``/``analysis.listing.currency``) rather than any mutable
    latest-market read, so a payload cannot be transplanted across listings
    or runs; ``expected_reference_close_in_band`` reuses the caller's own
    already-computed Under-$10 band membership for that same immutable
    close (`_decision_run_price_band`), rather than re-deriving the band
    boundary a second time here.

    ``expected_code_revision`` is the parent `AnalysisRun.code_revision`
    already loaded for this detail view. The payload's revision must be a
    nonempty string exactly equal to it; a self-consistent checksum cannot
    move an assessment between code revisions.

    C1: a matching date/reference close/currency alone is not sufficient
    identity -- two genuine, unrelated listings can share all three on the
    same day. ``expected_listing_id`` (the permanent `Listing.id`, bound
    to exact equality against the payload's own recorded
    ``evaluated_for.listing_id``), ``expected_data_cutoff`` (the parent
    `AnalysisRun.data_cutoff`, bound to *exact* equality -- the generator
    is called with the very same ``decision_time`` that also becomes the
    run's own cutoff, so a genuine payload's own recorded value is never
    merely earlier or later, always identical), and ``expected_price_
    asset`` (both the UUID *and* content checksum of the same analysis's
    own recorded price asset, cross-derived from ``data_quality[
    "source_assets"]`` -- the identical `DataAsset` `build_under10_
    assessment` was actually given, read from the same already-loaded
    JSON, no extra query) together close a transplant between two
    same-date/same-price/same-currency candidates with different
    evidence: F2 proved that copying an entire genuine `data_quality`
    blob (payload, its ``price_source``, and its ``source_assets``) from
    one analysis to another otherwise-identical-looking one moves every
    *internal* anchor together, so only the permanent listing id -- never
    reconstructable from the copied blob alone once bound here -- can
    close that transplant. An assessment genuinely evaluated under a
    different run's cutoff, for a different listing, can never coincide
    with this one's.
    """
    if not isinstance(recorded, dict):
        return False
    if not _is_strict_int(recorded.get("schema_version")):
        return False
    if recorded["schema_version"] != UNDER10_SCHEMA_VERSION:
        return False
    if recorded.get("policy_version") != UNDER10_SHADOW_POLICY_VERSION:
        return False
    if recorded.get("activated") is not UNDER10_ACTIVATED:
        return False
    if recorded.get("shadow_only") is not True:
        return False
    if recorded.get("activation_eligible") is not UNDER10_ACTIVATION_ELIGIBLE:
        return False
    allocation_percent = recorded.get("new_allocation_percent")
    if (
        not _is_strict_int(allocation_percent)
        or allocation_percent != UNDER10_NEW_ALLOCATION_PERCENT
    ):
        return False
    gates = recorded.get("gates")
    expected_gates = under10_inactive_gates()
    if not isinstance(gates, dict) or set(gates) != set(expected_gates):
        return False
    if any(gates[key] is not expected_gates[key] for key in expected_gates):
        return False
    recorded_code_revision = recorded.get("code_revision")
    if (
        not _is_non_empty_str(recorded_code_revision)
        or recorded_code_revision != expected_code_revision
    ):
        return False
    # C3: the recorded policy hash must be the exact recognized hash for
    # this policy version -- not merely a non-empty string alongside a
    # matching assessment checksum. An unknown, missing, or mismatched
    # policy hash means this reader cannot honestly confirm which reviewed
    # SEC fundamentals configuration (or other policy input) the payload
    # was actually evaluated under, even if its own internal checksum is
    # self-consistent.
    if recorded.get("policy_hash") != _expected_under10_policy_hash():
        return False
    # The accidental-corruption checksum gate runs early, before any nested
    # presentation content is trusted -- but it is a checksum, not tamper
    # protection: every structural/semantic check below still runs even
    # when it passes, and a matching hash never substitutes for any of them.
    if not _is_valid_assessment_hash(recorded):
        return False
    evaluated_for = recorded.get("evaluated_for")
    if not isinstance(evaluated_for, dict):
        return False
    if evaluated_for.get("price_band") != UNDER_10_BAND:
        return False
    if evaluated_for.get("date_basis") != DECISION_TARGET_DATE_BASIS:
        return False
    # F2: the recorded permanent listing identity must match exactly. A
    # matching target date/cutoff/reference close/currency and even an
    # identical price asset are not sufficient on their own: copying an
    # entire genuine `data_quality` blob (this payload plus its sibling
    # `price_source`/`source_assets` anchors) from one analysis to another
    # moves every other internal anchor along with it, so only this
    # permanent id -- read from the payload, never re-derived from the
    # copied blob -- can catch that whole-blob transplant.
    recorded_listing_id = evaluated_for.get("listing_id")
    if (
        not isinstance(recorded_listing_id, str)
        or not recorded_listing_id
        or recorded_listing_id != expected_listing_id
    ):
        return False
    target_date = _parse_iso_date(evaluated_for.get("target_date"))
    # A forged/stale `target_date` would let an assessment recorded for a
    # different decision run masquerade as this one's; comparing against
    # the analysis's own `run.target_date` (already loaded, no extra query)
    # is the authoritative cross-check.
    if target_date is None or target_date != expected_target_date:
        return False
    # C1: the recorded data cutoff must be *exactly* the parent analysis
    # run's own cutoff, not merely at or before it. `compute_listing_
    # analysis`/`_with_under10_assessment` call the generator with the
    # very same `decision_time` value that `_create_analysis_run` also
    # persists as `AnalysisRun.data_cutoff` -- a genuine payload's own
    # recorded value is therefore never independently chosen, always
    # identical. An earlier recorded cutoff attached to a later run (e.g.
    # transplanted from an earlier re-run of the same target date) is just
    # as much a mismatch as a later one, and must never masquerade as this
    # run's own evaluation. Equality between `datetime` values never
    # raises even when one side is naive and the other timezone-aware --
    # that mismatch simply (and correctly) compares unequal, so this is
    # fail-closed without needing exception handling.
    data_cutoff = _parse_iso_datetime(evaluated_for.get("data_cutoff"))
    if data_cutoff is None or data_cutoff != expected_data_cutoff:
        return False
    # F3: `reference_close` is bound exactly to the already-loaded immutable
    # decision-run close (never the mutable latest market price), at the
    # same canonical 6-decimal quantization the generator itself emits --
    # not merely format-checked. A payload whose own bound close cannot
    # even be recomputed to a canonical string is never trusted either.
    expected_reference_close_text = _recompute_quantized_text(
        expected_reference_close, _REFERENCE_CLOSE_QUANTUM
    )
    if (
        expected_reference_close_text is None
        or evaluated_for.get("reference_close") != expected_reference_close_text
    ):
        return False
    # The bound decision reference itself must remain in the Under-$10 USD
    # band for a recorded assessment to exist at all: negative, zero,
    # ``>= 10``, and any different-but-under-$10 close (a transplant from
    # another listing/run entirely) are all refused here, reusing the
    # caller's own band classification rather than re-deriving it.
    if not expected_reference_close_in_band:
        return False
    # Recorded currency must match the already-loaded listing's own
    # currency: a payload transplanted from a different-currency listing
    # must never masquerade as this one's, even with a matching hash.
    # NB-3: the *recorded* payload currency is still held to the
    # generator's exact canonical constant (never weakened) -- but
    # `expected_currency` (the listing's own stored value) is normalized
    # `.upper()` first, exactly like `classify_price_band`'s own
    # case-insensitive comparison and the existing `listing__currency__
    # iexact` filters elsewhere in this module: a listing genuinely
    # recorded as lowercase "usd" is still USD, never a transplant merely
    # because of letter case.
    recorded_currency = evaluated_for.get("currency")
    if recorded_currency != PRICE_BAND_CURRENCY or recorded_currency != expected_currency.upper():
        return False
    if not _is_valid_under10_solvency(recorded.get("solvency"), target_date=target_date):
        return False
    if not _is_valid_under10_liquidity(
        recorded.get("liquidity"),
        target_date=target_date,
        expected_price_asset=expected_price_asset,
    ):
        return False
    split = recorded.get("split_verification")
    if not _is_valid_under10_split(split):
        return False
    assert isinstance(split, dict)
    return recorded.get("blocking_reasons") == under10_blocking_reasons(split["reason"])


def _is_valid_under10_periods(periods: object) -> bool:
    """Structural validity, including the duration triple's all-or-nothing shape.

    Each field, if present, is individually well-formed. ``instant_date``
    has no presence relationship to the duration fields checked here (it
    reflects whether all five *balance-sheet* instants agreed on one date,
    an entirely separate condition from whether an FCF candidate was found
    at all); `_is_valid_under10_solvency` separately enforces "all four
    populated together" as part of the complete-input-set invariant, since
    whether every period claim must be populated depends on the sibling
    ``status``.

    ``duration_start``/``duration_end``/``duration_basis`` are always
    assigned together in one dict literal by `_resolve_free_cash_flow`
    (either all three still at their all-``None`` default, or all three set
    together the moment any FCF candidate -- usable or not -- is found): a
    payload presenting only one or two of the three is not a shape the
    generator can produce.

    A populated triple's span is independently required to fall inside the
    same inclusive ``MIN_ANNUAL_DAYS``-``MAX_ANNUAL_DAYS`` (350-380 day)
    window that `stanstock.data.sec_fundamentals` enforces for both the
    annual and TTM constructions that can produce ``value.period_start``/
    ``value.period_end`` here (see ``_annual_series``/``_ttm_series``/
    ``_homogeneous_four_quarter_tail``). Neither basis is exempt: a genuine
    generated payload can never present a shorter or longer flow window
    under either label, so a payload claiming one is not internally
    consistent with itself regardless of its recomputed checksum.
    """
    if not isinstance(periods, dict):
        return False
    if "duration_basis" not in periods:
        return False
    duration_basis = periods["duration_basis"]
    if duration_basis is not None and duration_basis not in _UNDER10_DURATION_BASIS_VALUES:
        return False
    parsed: dict[str, date | None] = {}
    for key in ("instant_date", "duration_start", "duration_end"):
        if key not in periods:
            return False
        raw = periods[key]
        if raw is None:
            parsed[key] = None
            continue
        parsed_date = _parse_iso_date(raw)
        if parsed_date is None:
            return False
        parsed[key] = parsed_date
    start, end = parsed["duration_start"], parsed["duration_end"]
    duration_fields_present = (start is not None, end is not None, duration_basis is not None)
    if len(set(duration_fields_present)) != 1:
        # A partial duration triple (e.g. only ``duration_end``/
        # ``duration_basis`` present but ``duration_start`` absent) is not a
        # shape `_resolve_free_cash_flow` can produce.
        return False
    if start is None or end is None:
        return True
    # A flow period whose claimed end precedes its claimed start is an
    # internally incompatible claim, never a genuine reported window.
    if start > end:
        return False
    # `_resolve_free_cash_flow` can only ever have copied
    # ``period_start``/``period_end`` from a `FundamentalValue` the annual or
    # TTM construction already proved falls inside this same inclusive span
    # (`MIN_ANNUAL_DAYS`-`MAX_ANNUAL_DAYS`); a stored triple outside it could
    # not have been genuinely generated under either basis.
    span_days = (end - start).days + 1
    return MIN_ANNUAL_DAYS <= span_days <= MAX_ANNUAL_DAYS


def _is_valid_under10_solvency(solvency: object, *, target_date: date) -> bool:
    """Whether ``solvency`` is internally consistent with itself, exactly.

    Every derived claim reachable from the recorded canonical operands is
    recomputed under the same precision-64/``ROUND_HALF_EVEN`` rules
    `under10.py` itself uses, and compared for *exact* canonical string
    equality -- never structural plausibility alone, a tolerance, or a
    float conversion. This is a pure internal-consistency proof over the
    already-persisted payload: it never re-reads evidence, queries the
    database, or rebuilds the assessment.

    ``near_term_debt``'s own two addends (``short_term_debt``/
    ``current_long_term_debt``) are not separately persisted anywhere in
    this payload, so its internal decomposition cannot be recomputed from
    stored fields alone; what *is* fully recomputed is every place its
    recorded value is actually used afterwards -- the ``debt_exceeds_cash``
    predicate and the frozen state/reason classification below -- which is
    exactly the exploit this closes (a forged ``near_term_debt`` no longer
    survives paired with an unchanged favorable state).
    """
    if not isinstance(solvency, dict):
        return False
    status = solvency.get("status")
    # A ``str`` type guard must run before the membership test below: `in`/
    # `not in` against a ``dict`` hashes its left operand, and a malformed
    # stored payload's status could be an unhashable ``list``/``dict``/``set``,
    # which would raise ``TypeError`` (an uncaught 500) instead of the
    # intended "unsupported" render.
    if not isinstance(status, str) or status not in UNDER10_SOLVENCY_LABELS:
        return False
    reasons = solvency.get("reasons")
    if not isinstance(reasons, list) or not all(_is_non_empty_str(item) for item in reasons):
        return False
    inputs = solvency.get("inputs")
    if not isinstance(inputs, dict):
        return False
    parsed_inputs: dict[str, Decimal | None] = {}
    for key in _UNDER10_SOLVENCY_INPUT_KEYS:
        if key not in inputs:
            return False
        value = inputs[key]
        if value is None:
            parsed_inputs[key] = None
            continue
        # Every present monetary/ratio input is the exact canonical
        # fixed-point form the generator emits for it; a raw ``None`` stays
        # a legitimate "withheld" input, but "NaN"/"Infinity"/an extreme
        # exponent/a bool/anything not in that exact lexical shape is never
        # trusted just because it happens to be a ``str``.
        places = _REPORTED_DECIMAL_PLACES if key == "current_ratio" else _MONETARY_DECIMAL_PLACES
        parsed_value = _parse_canonical_decimal(value, places=places)
        if parsed_value is None:
            return False
        parsed_inputs[key] = parsed_value
    cash = parsed_inputs["cash_and_equivalents"]
    near_term_debt = parsed_inputs["near_term_debt"]
    current_assets = parsed_inputs["current_assets"]
    liabilities = parsed_inputs["current_liabilities"]
    free_cash_flow = parsed_inputs["free_cash_flow"]
    # `_build_solvency` marks a zero/negative current liabilities value
    # unusable (turns it to ``None``) before it ever reaches `inputs`, so a
    # present value is always strictly positive; `_current_ratio` only ever
    # computes a ratio once *both* current assets and current liabilities
    # are themselves usable, so a present ratio without both a present
    # numerator and a present, positive liabilities denominator is not a
    # shape the generator can produce.
    if liabilities is not None and liabilities <= 0:
        return False
    if parsed_inputs["current_ratio"] is not None and (
        liabilities is None or current_assets is None
    ):
        return False
    # NB-5: the converse also holds -- `_current_ratio` computes a value
    # whenever *both* operands are usable (liabilities already proven
    # strictly positive above whenever present), regardless of any
    # unrelated debt/free-cash-flow input being withheld, so a null ratio
    # alongside both operands present is equally a contradiction.
    if (
        current_assets is not None
        and liabilities is not None
        and parsed_inputs["current_ratio"] is None
    ):
        return False
    if parsed_inputs["current_ratio"] is not None:
        # F1: the *exact* recomputed ``assets / liabilities`` quotient, not
        # merely structural plausibility -- never a float, never a
        # tolerance, and a negative recorded ratio can never match a
        # genuinely positive recomputed quotient.
        expected_ratio = _recompute_current_ratio_text(
            current_assets=current_assets, current_liabilities=liabilities
        )
        if expected_ratio is None or inputs["current_ratio"] != expected_ratio:
            return False
    periods = solvency.get("periods")
    if not _is_valid_under10_periods(periods):
        return False
    assert isinstance(periods, dict)  # narrowed by `_is_valid_under10_periods` above
    # A complete (favorable-or-adverse) solvency state is only ever reported
    # once every balance-sheet input and the shared instant/flow period
    # claims are usable (see `_build_solvency`'s `complete` gate and
    # `_resolve_instants`'s period-compatibility check); `insufficient_evidence`
    # is the only state architecturally allowed to expose a partial mix of
    # present and withheld inputs. A payload claiming a favorable/adverse
    # state while missing any of these is an internally inconsistent claim,
    # never a legitimate assessed state -- and, in the other direction,
    # `_build_solvency` only ever reaches `insufficient_evidence` when at
    # least one of the same five operands is unusable: a complete set
    # recorded alongside `insufficient_evidence` (NB-1) is equally a
    # contradiction, since a complete set always reaches
    # `_classify_solvency` instead. Genuine partial/mismatched/unusable
    # `insufficient_evidence` cases (missing any one of the five) remain
    # fully open, and its own reason vocabulary stays unconstrained.
    complete_input_keys = (
        "cash_and_equivalents",
        "near_term_debt",
        "current_assets",
        "current_liabilities",
        "free_cash_flow",
    )
    if status != SOLVENCY_INSUFFICIENT_EVIDENCE:
        if any(value is None for value in parsed_inputs.values()):
            return False
        if any(periods[key] is None for key in _UNDER10_PERIOD_KEYS):
            return False
    elif all(parsed_inputs[key] is not None for key in complete_input_keys):
        return False
    # F2: a shared instant date this reader will ever treat as backing a
    # *present* balance-sheet operand must be within `_validate_instant_fact`'s
    # own freshness window relative to this decision's own target date --
    # every one of the five instants sharing that date was validated
    # against the identical bound, so a present operand behind a
    # stale/future instant date is not a shape the generator can produce.
    # `insufficient_evidence` may still legitimately report a stale/future
    # shared instant date exactly when every one of these operands is
    # unavailable as a result.
    instant_date = _parse_iso_date(periods["instant_date"])
    if instant_date is not None and any(
        parsed_inputs[key] is not None
        for key in (
            "cash_and_equivalents",
            "current_assets",
            "current_liabilities",
            "near_term_debt",
        )
    ):
        if not _is_within_metric_age(claimed_date=instant_date, target_date=target_date):
            return False
    # F2: likewise for the flow window -- a stale/future ``duration_end``
    # may still preserve its assessed period metadata for audit, but can
    # never back a *usable* free cash flow value.
    duration_end = _parse_iso_date(periods["duration_end"])
    # C2: the converse also holds -- `_resolve_free_cash_flow` sets the
    # complete duration triple the moment any FCF candidate (usable or
    # not) is found, strictly *before* usability is even decided, so a
    # usable/present free cash flow value can never legitimately pair with
    # a nulled duration window. This holds independent of overall solvency
    # state -- including a genuine partial `insufficient_evidence` case
    # missing only unrelated debt, which still reports its own independent
    # runway diagnostic in full.
    if free_cash_flow is not None and duration_end is None:
        return False
    if duration_end is not None and free_cash_flow is not None:
        if not _is_within_metric_age(claimed_date=duration_end, target_date=target_date):
            return False
    if status != SOLVENCY_INSUFFICIENT_EVIDENCE:
        # F1: the recorded state and its exact ordered reason list are not
        # merely structurally plausible -- they are the exact frozen
        # first-match classification `_classify_solvency` would derive from
        # these same recorded operands, under the same precision-64
        # comparisons. `insufficient_evidence`'s own reason vocabulary spans
        # multiple modules and several dynamically concept-named strings,
        # and (being neither a favorable nor a specific adverse claim) stays
        # intentionally unrecomputed -- only a status that could itself
        # misrepresent a favorable-or-adverse claim is held to exact
        # recomputation.
        assert cash is not None
        assert near_term_debt is not None
        assert current_assets is not None
        assert liabilities is not None
        assert free_cash_flow is not None
        expected_status, expected_reasons = _recompute_solvency_classification(
            cash=cash,
            near_term_debt=near_term_debt,
            current_assets=current_assets,
            current_liabilities=liabilities,
            free_cash_flow=free_cash_flow,
        )
        if status != expected_status or reasons != expected_reasons:
            return False
    runway = solvency.get("runway")
    if not _is_valid_under10_runway(
        runway,
        free_cash_flow=free_cash_flow,
        cash=cash,
    ):
        return False
    assessed_fact_ids = solvency.get("assessed_fact_ids")
    assessed_assets = solvency.get("assessed_assets")
    if not _is_valid_assessed_fact_ids(assessed_fact_ids):
        return False
    if not _is_valid_assessed_assets(assessed_assets):
        return False
    assert isinstance(assessed_fact_ids, list)
    assert isinstance(assessed_assets, list)
    if bool(assessed_fact_ids) != bool(assessed_assets):
        return False
    if assessed_fact_ids:
        return True
    # Empty lineage is valid only when no evidence-derived claim survived:
    # an entirely insufficient, all-null result with a withheld runway.
    # Rejected/incompatible evidence remains valid because its nonempty
    # assessed lineage takes the branch above even when every input is null.
    return bool(
        status == SOLVENCY_INSUFFICIENT_EVIDENCE
        and all(value is None for value in parsed_inputs.values())
        and all(periods[key] is None for key in _UNDER10_PERIOD_KEYS)
        and isinstance(runway, dict)
        and runway.get("status") == RUNWAY_WITHHELD
        and runway.get("quarters") is None
    )


def _is_valid_under10_runway(
    runway: object,
    *,
    free_cash_flow: Decimal | None,
    cash: Decimal | None,
) -> bool:
    """Exact ``status``/``quarters``/``reason`` combinations `_build_runway` can produce.

    Cross-checked against the sibling solvency inputs it was actually
    derived from (already parsed by the caller), not re-read: ``computed``
    requires a negative, present free cash flow and a present cash figure of
    *either* sign (`_build_runway` applies no cash eligibility rule of its
    own -- a negative cash produces a negative quarters figure, e.g.
    ``"-0.0400"``, which is a legitimate reported value, not an error), and
    its reported quarters string must equal the *exact* recomputed
    ``4 * cash / abs(free_cash_flow)`` quotient (never merely sign-checked,
    never a float, never a tolerance) and never carries a reason.
    ``not_applicable_positive_fcf`` requires a present, non-negative free
    cash flow, and carries neither quarters nor a reason. ``withheld``
    requires free cash flow missing, or free cash flow negative with cash
    missing -- `_build_runway`'s own third branch (negative free cash flow,
    cash present, but the quotient itself non-finite) is a defensive
    precision-64 guard that is not reachable through any input magnitude
    this reader's own canonical fixed-point bound (`_UNDER10_DECIMAL_PRECISION`
    significant digits) admits: even the widest representable cash divided
    by the smallest representable nonzero free cash flow reaches an
    exponent orders of magnitude short of the ambient Decimal context's
    ``Emax``, confirmed empirically against the real generator, and any
    input extreme enough to approach it instead raises inside
    `_build_runway`'s own quantization before ever reaching a withheld
    return. So a `withheld` result while both free cash flow and cash are
    genuinely present is not a shape `_build_runway` can produce -- it
    always carries a reason, never a quarters figure.
    """
    if not isinstance(runway, dict):
        return False
    status = runway.get("status")
    if status not in (RUNWAY_COMPUTED, RUNWAY_NOT_APPLICABLE, RUNWAY_WITHHELD):
        return False
    quarters = runway.get("quarters")
    reason = runway.get("reason")
    fcf_non_negative = free_cash_flow is not None and free_cash_flow >= 0
    if status == RUNWAY_NOT_APPLICABLE:
        return fcf_non_negative and quarters is None and reason is None
    if fcf_non_negative:
        # A present, non-negative free cash flow can only ever produce
        # `not_applicable_positive_fcf` -- never `computed` (which requires
        # a *negative* free cash flow) and never `withheld` (a
        # non-negative free cash flow never reaches the cash/quotient
        # checks `_build_runway` applies before withholding).
        return False
    if status == RUNWAY_COMPUTED:
        if free_cash_flow is None or cash is None or reason is not None:
            return False
        # F1: the *exact* recomputed quotient, not merely a sign check --
        # never a float, never a tolerance. Correctly accepts a genuine
        # signed negative zero (a tiny negative cash rounding to
        # ``"-0.0000"``) because the recomputation itself preserves the
        # same sign bit, and correctly rejects an unrelated magnitude
        # forged onto an otherwise genuine payload.
        expected_quarters = _recompute_runway_quarters_text(
            cash=cash, free_cash_flow=free_cash_flow
        )
        return expected_quarters is not None and quarters == expected_quarters
    # status == RUNWAY_WITHHELD
    if quarters is not None or not _is_non_empty_str(reason):
        return False
    # Reached this branch with free cash flow already established negative
    # (the non-negative case returned above): `_build_runway` only ever
    # withholds from here when free cash flow is itself missing, or cash
    # is -- both present and finite is a `computed` shape, never withheld
    # (see the docstring above for why the one other branch that could
    # excuse this is not reachable at any admissible magnitude).
    return free_cash_flow is None or cash is None


def _liquidity_window_bounds(reason: str) -> tuple[int, int] | None:
    """Inclusive ``(minimum, maximum)`` session count a recognized reason's
    withheld branch can report, or ``None`` if that reason's branch never
    inspects a window at all (`sessions_used` itself absent/``None``).

    Derived directly from `median_dollar_volume`'s own code paths, not a
    business assumption: `insufficient_sessions` alone can report either an
    explicit zero-session empty history or a genuine too-short window of
    1..251 sessions (`window.height < sessions` is what produces this
    reason, so it can never reach 252); every other windowed reason
    requires a genuine 1..252-session window; `nonfinite_median` is only
    ever reached once a full 252-session window was already confirmed.
    """
    if reason == DOLLAR_VOLUME_INSUFFICIENT_SESSIONS:
        return (0, UNDER10_LIQUIDITY_SESSIONS - 1)
    if reason in _LIQUIDITY_FULL_WINDOW_REASONS:
        return (UNDER10_LIQUIDITY_SESSIONS, UNDER10_LIQUIDITY_SESSIONS)
    if reason in _LIQUIDITY_WINDOWED_REASONS:
        return (1, UNDER10_LIQUIDITY_SESSIONS)
    return None


def _is_valid_session_window(
    *,
    sessions_used: object,
    first_session: object,
    last_session: object,
    target_date: date,
    minimum: int,
    maximum: int,
    require_future_last_session: bool = False,
) -> bool:
    """A genuine ``[minimum, maximum]``-bounded session-count/date-descriptor
    shape, including the temporal bound on the reported last session. Never
    reads an exchange calendar or a price row: the inclusive-calendar-
    capacity check below is a necessary, not sufficient, arithmetic bound
    (``count`` genuinely distinct calendar dates cannot fit in fewer than
    ``count - 1`` days).

    ``require_future_last_session`` inverts that temporal bound for the one
    reason (`future_price_session`) whose entire purpose is to report a
    last session *after* the decision's own target date -- every other
    reason (including ``computed``) instead requires it on or before.
    """
    if not _is_strict_int(sessions_used) or not minimum <= sessions_used <= maximum:
        return False
    if sessions_used == 0:
        return first_session is None and last_session is None
    first_date = _parse_iso_date(first_session)
    last_date = _parse_iso_date(last_session)
    if first_date is None or last_date is None or first_date > last_date:
        return False
    if (last_date - first_date).days < sessions_used - 1:
        return False
    if require_future_last_session:
        return last_date > target_date
    return last_date <= target_date


def _is_valid_liquidity_window(
    *,
    reason: str,
    sessions_used: object,
    first_session: object,
    last_session: object,
    target_date: date,
) -> bool:
    """The exact session-count/date-descriptor shape a *recognized* withheld
    ``reason`` can produce."""
    bounds = _liquidity_window_bounds(reason)
    if bounds is None:
        return sessions_used is None and first_session is None and last_session is None
    minimum, maximum = bounds
    return _is_valid_session_window(
        sessions_used=sessions_used,
        first_session=first_session,
        last_session=last_session,
        target_date=target_date,
        minimum=minimum,
        maximum=maximum,
        require_future_last_session=reason == LIQUIDITY_FUTURE_PRICE_SESSION,
    )


def _is_valid_under10_liquidity(
    liquidity: object,
    *,
    target_date: date,
    expected_price_asset: _ExpectedPriceAssetReference | None,
) -> bool:
    if not isinstance(liquidity, dict):
        return False
    status = liquidity.get("status")
    if status not in (LIQUIDITY_COMPUTED, LIQUIDITY_WITHHELD):
        return False
    # `metric` is a fixed identity, constant across both `computed` and
    # `withheld` results (see `_build_liquidity`/`_withheld_liquidity`); an
    # unrecognized metric name is never a legitimate result under any status.
    if liquidity.get("metric") != UNDER10_LIQUIDITY_METRIC:
        return False
    value = liquidity.get("value")
    if status == LIQUIDITY_COMPUTED:
        if not _is_finite_number(value) or value < 0:
            return False
    elif value is not None:
        # A withheld liquidity result reporting a value would be an invalid
        # status combination.
        return False
    if liquidity.get("currency") != PRICE_BAND_CURRENCY:
        return False
    sessions_used = liquidity.get("sessions_used")
    first_session = liquidity.get("first_session")
    last_session = liquidity.get("last_session")
    if not _is_optional_str(first_session) or not _is_optional_str(last_session):
        return False
    if sessions_used is not None and not _is_strict_int(sessions_used):
        return False
    basis = liquidity.get("basis")
    if not isinstance(basis, dict):
        return False
    # `volume_basis` is never observed-copied (always the one fixed
    # constant, regardless of branch), so it stays strictly checked; the
    # other three fields may be *any* JSON scalar/container when withheld --
    # `_build_liquidity` faithfully preserves whatever the source asset's
    # metadata literally contained there (e.g. ``interval=7``,
    # ``adjustment=["splits"]``) rather than nulling out present-but-wrong
    # values, and this reader must not reject that honestly preserved
    # evidence, invent a type it never had, or ever treat it as compatible.
    if basis.get("volume_basis") != UNDER10_VOLUME_BASIS:
        return False
    for key in _UNDER10_OBSERVED_BASIS_KEYS:
        if key not in basis:
            return False
    price_asset = liquidity.get("price_asset")
    if status == LIQUIDITY_COMPUTED:
        # `_build_liquidity` only ever reaches `LIQUIDITY_COMPUTED` after
        # confirming the source asset's observed interval/adjustment/
        # return-definition metadata matches this exactly, resolving a
        # concrete, confirmed anchor, and reporting exactly
        # `UNDER10_LIQUIDITY_SESSIONS` observed sessions in chronological
        # order, no later than the decision's own target date -- never a
        # partial window, a mismatched basis, a missing anchor, or an
        # unverified adjustment passed off as compatible. A computed
        # result never carries a withholding reason.
        if basis != _UNDER10_COMPATIBLE_PRICE_BASIS or liquidity.get("reason") is not None:
            return False
        if not _is_bound_price_asset(price_asset, expected=expected_price_asset):
            return False
        if not _is_valid_session_window(
            sessions_used=sessions_used,
            first_session=first_session,
            last_session=last_session,
            target_date=target_date,
            minimum=UNDER10_LIQUIDITY_SESSIONS,
            maximum=UNDER10_LIQUIDITY_SESSIONS,
        ):
            return False
        # F2: `_build_liquidity` only reaches `LIQUIDITY_COMPUTED` once
        # `_price_staleness_reason` has already confirmed the last observed
        # session is neither stale nor in the future -- the window check
        # above only proves it is not *after* the target date, not that it
        # is recent enough. `last_session` is already a validated, non-null
        # ISO date here (`sessions_used == UNDER10_LIQUIDITY_SESSIONS > 0`
        # forces both dates present).
        computed_last_session = _parse_iso_date(last_session)
        assert computed_last_session is not None
        return _is_within_price_staleness(
            last_session=computed_last_session, target_date=target_date
        )
    # status == LIQUIDITY_WITHHELD
    reason = liquidity.get("reason")
    # A withheld result always names *why*: every generated withholding
    # reason is a non-empty, *recognized* string -- an unrecognized reason
    # can never establish or render completed evidence just because its
    # window descriptors happen to look plausible, and a missing/null/empty
    # reason is a malformed payload, not a legitimate withholding this
    # reader has simply not seen a label for yet (never rendered as
    # "Withheld - None" or a blank explanation).
    if (
        not isinstance(reason, str)
        or reason == ""
        or reason not in _RECOGNIZED_LIQUIDITY_WITHHELD_REASONS
    ):
        return False
    if not _is_valid_liquidity_window(
        reason=reason,
        sessions_used=sessions_used,
        first_session=first_session,
        last_session=last_session,
        target_date=target_date,
    ):
        return False
    # NB-4: staleness is resolved *before* `_build_liquidity` ever inspects
    # `median_dollar_volume`'s own status/reason, so every recognized
    # reason with a real reported window is held to the exact relationship
    # that placement implies -- not merely `stale_price_evidence` alone.
    if not _is_valid_liquidity_staleness(
        reason=reason, last_session=_parse_iso_date(last_session), target_date=target_date
    ):
        return False
    if reason == LIQUIDITY_PRICE_PROVENANCE_UNAVAILABLE:
        # The only withheld reason whose anchor is genuinely absent (no
        # price asset at all, or a mismatched one): a non-null anchor
        # paired with this reason is a contradiction the generator cannot
        # produce, and so is a basis reporting anything other than the
        # fixed empty shape -- nothing was observed for the correct asset
        # in this branch at all, so a compatible-looking (or any other
        # non-empty) string here is a forged claim, never genuine evidence.
        return price_asset is None and basis == _UNDER10_EMPTY_PRICE_BASIS
    # Every other recognized withheld reason (basis-incompatible, stale,
    # insufficient-sessions, missing-columns, etc.) is only ever reached
    # after the anchor was already confirmed; a missing or malformed
    # anchor paired with one of these reasons is likewise a contradiction,
    # and (C1) so is a genuinely well-formed anchor for a *different*
    # candidate's price asset.
    return _is_bound_price_asset(price_asset, expected=expected_price_asset)


#: The only two reasons `split_event_capability` can ever attach in this
#: policy version, enumerated positively (never a blacklist): every other
#: string is unrecognized, regardless of how plausible it looks.
_RECOGNIZED_SPLIT_REASONS = frozenset({CAPABILITY_PLAN_NOT_ENTITLED, CAPABILITY_NO_REVIEWED_SOURCE})


def _is_valid_under10_split(split: object) -> bool:
    """C4: exact recognized ``status``/``reason``/``provider``/``plan_recorded``
    relationships, not merely non-empty strings and a bare ``bool``.

    `split_event_capability` can never return anything but the one fixed
    ``status``/``capability``/``inference_prohibited`` triple this policy
    version supports, and its ``reason`` is a closed two-value vocabulary:
    `CAPABILITY_PLAN_NOT_ENTITLED` only for a provider that normalizes to
    Twelve Data *with* a recorded plan (only "basic" ever produces that
    reason, and "basic" is itself a recorded plan, so ``plan_recorded``
    must be ``True``) -- claiming that reason while also claiming no plan
    was recorded, or against a different provider, is a contradiction this
    reader must never render as genuine Twelve Data Basic wording.
    `CAPABILITY_NO_REVIEWED_SOURCE` is the generic catch-all for every
    other provider/plan combination and is not otherwise constrained here.
    """
    if not isinstance(split, dict):
        return False
    if split.get("status") != CAPABILITY_UNAVAILABLE:
        return False
    if split.get("capability") != SPLIT_EVENT_CAPABILITY:
        return False
    if split.get("inference_prohibited") is not True:
        return False
    reason = split.get("reason")
    # F1: a JSON list/dict `reason` is "not None" but unhashable -- membership
    # against the `frozenset` below would raise `TypeError` before this
    # function's own contract (a total, never-raising predicate) is upheld.
    # An `isinstance` guard, evaluated first, makes the frozenset test only
    # ever see a hashable string.
    if not isinstance(reason, str) or reason not in _RECOGNIZED_SPLIT_REASONS:
        return False
    provider = split.get("provider")
    if not isinstance(provider, str) or not provider:
        return False
    plan_recorded = split.get("plan_recorded")
    if not isinstance(plan_recorded, bool):
        return False
    if reason == CAPABILITY_PLAN_NOT_ENTITLED:
        if provider.strip().lower() != TWELVE_DATA_PROVIDER or not plan_recorded:
            return False
    return True


def _unsupported_under10_panel(analysis: StockAnalysis) -> dict[str, Any]:
    return {
        "state": "unsupported",
        "headline": UNDER10_UNSUPPORTED_MESSAGE,
        "decision_target_date": analysis.run.target_date,
        "decision_data_cutoff": analysis.run.data_cutoff,
    }


def _under10_evaluated_for(recorded: dict[str, Any], key: str) -> str | None:
    evaluated_for = recorded.get("evaluated_for")
    if not isinstance(evaluated_for, dict):
        return None
    value = evaluated_for.get(key)
    return value if isinstance(value, str) else None


def _decision_run_price_band(analysis: StockAnalysis) -> PriceBandAssessment | None:
    return classify_price_band(
        close=analysis.current_price,
        price_date=analysis.run.target_date,
        date_basis=DECISION_TARGET_DATE_BASIS,
        currency=analysis.listing.currency,
    )


def _under10_solvency_panel(solvency: dict[str, Any]) -> dict[str, Any]:
    status = solvency.get("status")
    label = UNDER10_SOLVENCY_LABELS.get(str(status))
    if label is None:
        return {
            "state": "withheld",
            "headline": UNDER10_UNSUPPORTED_MESSAGE,
            "reasons": [],
            "inputs": [],
            "runway": {"headline": UNDER10_UNSUPPORTED_MESSAGE},
        }
    withheld = status == "insufficient_evidence"
    reasons = solvency.get("reasons")
    inputs = solvency.get("inputs")
    return {
        "state": "withheld" if withheld else "assessed",
        "headline": (f"Withheld - {label.lower()}" if withheld else f"Assessed - {label.lower()}"),
        "reasons": [str(reason) for reason in reasons] if isinstance(reasons, list) else [],
        # Values are pre-rendered strings so an explicit "0.00000000" survives
        # template truthiness and default filters unchanged.
        "inputs": [
            {
                "label": label_text,
                "value": inputs.get(key),
                "present": isinstance(inputs.get(key), str),
            }
            for key, label_text in UNDER10_INPUT_LABELS
            if isinstance(inputs, dict)
        ],
        "periods": solvency.get("periods") if isinstance(solvency.get("periods"), dict) else {},
        "runway": _under10_runway_panel(solvency.get("runway")),
        "assessed_fact_count": (
            len(solvency["assessed_fact_ids"])
            if isinstance(solvency.get("assessed_fact_ids"), list)
            else 0
        ),
    }


def _under10_runway_panel(runway: object) -> dict[str, Any]:
    if not isinstance(runway, dict):
        return {"headline": UNDER10_UNSUPPORTED_MESSAGE, "quarters": None}
    status = runway.get("status")
    quarters = runway.get("quarters")
    if status == "computed" and isinstance(quarters, str):
        return {
            "headline": f"Cash runway - {quarters} quarters at the reported burn",
            "quarters": quarters,
        }
    if status == "not_applicable_positive_fcf":
        return {
            "headline": "Not applicable - FCF is non-negative.",
            "quarters": None,
        }
    reason = runway.get("reason")
    return {
        "headline": f"Withheld - {reason}" if isinstance(reason, str) else "Withheld.",
        "quarters": None,
    }


def _under10_liquidity_panel(liquidity: dict[str, Any]) -> dict[str, Any]:
    status = liquidity.get("status")
    value = liquidity.get("value")
    raw_basis = liquidity.get("basis")
    basis: dict[str, Any] = raw_basis if isinstance(raw_basis, dict) else {}
    computed = status == "computed" and isinstance(value, int | float)
    # Established requires the *complete* validated branch, not merely
    # three matching strings and a blacklist of two excluded reasons: a
    # confirmed, well-formed anchor; every compatible metadata field
    # (including currency); and a reason that is a *positively recognized*
    # post-basis-confirmation branch (`_LIQUIDITY_POST_BASIS_REASONS`) --
    # everything `_build_liquidity`/`median_dollar_volume` can still reach
    # once the anchor and metadata were already confirmed compatible,
    # never an unrecognized reason string. A missing anchor, a mismatched
    # anchor, or present-but-incompatible metadata (which can still
    # coincidentally repeat some of these exact strings, e.g. a
    # currency-only mismatch) must never read as established.
    reason = liquidity.get("reason")
    price_basis_established = (
        _is_valid_price_asset_reference(liquidity.get("price_asset"))
        and basis == _UNDER10_COMPATIBLE_PRICE_BASIS
        and liquidity.get("currency") == PRICE_BAND_CURRENCY
        and (status == LIQUIDITY_COMPUTED or reason in _LIQUIDITY_POST_BASIS_REASONS)
    )
    if price_basis_established:
        volume_basis_caveat = (
            "Split-only price basis is confirmed for this evidence. The provider's "
            "reported volume is not independently verified for splits, so this "
            "figure alone can never pass an activation gate."
        )
    else:
        volume_basis_caveat = (
            "Price/volume split basis was not established for this evidence, so "
            "liquidity is withheld."
        )
    return {
        "state": "assessed" if computed else "withheld",
        "headline": (
            "Assessed - median dollar volume over 252 observed sessions"
            if computed
            else f"Withheld - {reason}"
        ),
        "metric": liquidity.get("metric"),
        "value": value,
        "currency": liquidity.get("currency"),
        "sessions_used": liquidity.get("sessions_used"),
        "first_session": liquidity.get("first_session"),
        "last_session": liquidity.get("last_session"),
        "basis": basis,
        "volume_basis_caveat": volume_basis_caveat,
    }


def _under10_split_panel(split: dict[str, Any]) -> dict[str, Any]:
    reason = str(split.get("reason") or "")
    return {
        "state": "withheld",
        "headline": "Withheld - verified split evidence is unavailable.",
        "reason": reason,
        "explanation": UNDER10_SPLIT_EXPLANATIONS.get(
            reason,
            "Verified split and reverse-split evidence is unavailable for this provider.",
        ),
        "provider": split.get("provider"),
        "plan_recorded": split.get("plan_recorded"),
    }


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

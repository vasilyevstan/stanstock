"""Primary web adapter for the verified prospective research product."""

from __future__ import annotations

from collections import Counter
from typing import cast
from uuid import UUID

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from stanstock.core.models import JobRun
from stanstock.core.services import system_status
from stanstock.data.models import Listing, ProviderRecord
from stanstock.portfolio.models import TrackedSymbol
from stanstock.portfolio.watchlist import (
    TrackedSymbolValidationError,
    add_tracked_symbol,
)
from stanstock.research.models import AnalysisRun, Prediction, PredictionOutcome, StockAnalysis
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    MOMENTUM_METHOD_VERSION,
)
from stanstock.research.product_reader import (
    ProductCard,
    ProductHistoryRead,
    ProductRead,
    ProductVerificationSession,
    read_research_product,
    read_research_product_history,
)
from stanstock.research.product_study_evidence import read_registered_price_product_study
from stanstock.research.reporting import canonical_reportable_prediction_filter
from stanstock.web import views as legacy_views
from stanstock.web.forms import ResearchProductFilterForm, TrackedSymbolForm

_PERFORMANCE_RUN_BATCH_SIZE = 200


def _read(request: HttpRequest) -> ProductRead:
    result = read_research_product(user=cast(User, request.user))
    # The runtime context processor consumes this request-scoped result rather
    # than selecting or verifying the run a second time.
    request._stanstock_product_read = result  # type: ignore[attr-defined]
    return result


def _read_history(
    request: HttpRequest,
    *,
    verification: ProductVerificationSession | None = None,
) -> ProductHistoryRead:
    result = read_research_product_history(
        user=cast(User, request.user),
        verification=verification,
    )
    # Historical readers verify the newest run as part of the same complete
    # owner cohort. Reuse that active projection for the runtime banner rather
    # than verifying the current run a second time.
    request._stanstock_product_read = result.current  # type: ignore[attr-defined]
    return result


@login_required
def opportunities_page(request: HttpRequest) -> HttpResponse:
    if not settings.RESEARCH_PRODUCT_ENABLED:
        if _has_product_output():
            return archive_opportunities_page(request)
        return legacy_views.opportunities_page(request)
    product = _read(request)
    form = ResearchProductFilterForm(request.GET or None)
    cards = list(product.cards)
    if form.is_valid():
        cards = _filter_cards(cards, form.cleaned_data)
    elif request.GET:
        cards = []
    under_10 = [card for card in cards if card.target_under_10]
    standard = [card for card in cards if not card.target_under_10]
    promoted = [card for card in standard if card.current_promotion_eligible]
    return render(
        request,
        "web/product_opportunities.html",
        {
            "product": product,
            "filter_form": form,
            "cards": cards,
            "standard_cards": standard,
            "under_10_cards": under_10,
            "promoted_cards": promoted,
        },
    )


def _filter_cards(cards: list[ProductCard], values: dict[str, object]) -> list[ProductCard]:
    query = str(values.get("q") or "").strip().casefold()
    direction = str(values.get("direction") or "")
    suggestion = str(values.get("suggestion") or "")
    risk = str(values.get("risk") or "")
    price_band = str(values.get("price_band") or "")
    result = []
    for card in cards:
        if query and query not in (
            f"{card.listing.ticker} {card.listing.security.company.name}".casefold()
        ):
            continue
        if direction and card.direction != direction:
            continue
        if suggestion and card.suggestion != suggestion:
            continue
        if risk and card.relative_volatility_label != risk:
            continue
        if price_band == "under_10" and not card.target_under_10:
            continue
        if price_band == "at_least_10" and card.target_under_10:
            continue
        result.append(card)
    return result


@login_required
def stock_detail_page(request: HttpRequest, listing_id: UUID) -> HttpResponse:
    if not settings.RESEARCH_PRODUCT_ENABLED:
        if _has_product_output():
            return archive_stock_detail_page(request, listing_id)
        return legacy_views.stock_detail_page(request, listing_id)
    product = _read(request)
    card = next((item for item in product.cards if item.analysis.listing_id == listing_id), None)
    if card is None:
        if product.status == "integrity_failed":
            return render(
                request,
                "web/product_stock_detail.html",
                {"product": product, "card": None},
                status=503,
            )
        raise Http404("Stock is not in the verified active research cohort")
    return render(
        request,
        "web/product_stock_detail.html",
        {"product": product, "card": card},
    )


@login_required
def status_page(request: HttpRequest) -> HttpResponse:
    if not settings.RESEARCH_PRODUCT_ENABLED:
        if not _has_product_output():
            return legacy_views.status_page(request)
    product = _read(request)
    components = system_status()
    admission_counts = Counter(item.status for item in product.admissions)
    jobs = JobRun.objects.filter(
        Q(job_name__startswith="daily_research_v1")
        | Q(job_name__startswith="scheduled_refresh_research_v1")
        | Q(job_name="refresh_demo_research_v1")
    ).order_by("-started_at")[:10]
    return render(
        request,
        "web/product_status.html",
        {
            "product": product,
            "components": components,
            "system_ok": all(bool(component["ok"]) for component in components),
            "admission_counts": sorted(admission_counts.items()),
            "providers": ProviderRecord.objects.order_by("provider"),
            "recent_jobs": jobs,
            "scheduler": legacy_views._scheduler_status(),
        },
    )


@login_required
def prediction_history_page(request: HttpRequest) -> HttpResponse:
    if not settings.RESEARCH_PRODUCT_ENABLED:
        if _has_product_output():
            return archive_prediction_history_page(request)
        return legacy_views.prediction_history_page(request)
    history = _read_history(request)
    product = history.current
    cohort_page = Paginator(history.cohorts, 5).get_page(request.GET.get("page"))
    page_cards = [card for cohort in cohort_page.object_list for card in cohort.cards]
    decision_cards = page_cards
    advisory_rows = [
        (
            card,
            next(
                prediction
                for prediction in card.advisory_predictions
                if prediction.horizon == projection.horizon
            ),
            projection,
        )
        for card in page_cards
        for projection in card.projections
    ]
    return render(
        request,
        "web/product_history.html",
        {
            "product": product,
            "history": history,
            "cohort_page": cohort_page,
            "decision_cards": decision_cards,
            "advisory_rows": advisory_rows,
        },
    )


@login_required
def performance_page(request: HttpRequest) -> HttpResponse:
    if not settings.RESEARCH_PRODUCT_ENABLED:
        if _has_product_output():
            return archive_performance_page(request)
        return legacy_views.performance_page(request)
    verification = ProductVerificationSession()
    history = _read_history(request, verification=verification)
    product = history.current
    registered_study = read_registered_price_product_study(
        user=cast(User, request.user),
        verification=verification,
    )
    decision_groups: list[dict[str, object]] = []
    advisory_groups: list[dict[str, object]] = []
    if history.available:
        decision_groups, advisory_groups = _observed_performance_groups(history)
    return render(
        request,
        "web/product_performance.html",
        {
            "product": product,
            "history": history,
            "registered_study": registered_study,
            "decision_groups": decision_groups,
            "advisory_groups": advisory_groups,
            "has_matured": any(
                group["status"] == PredictionOutcome.Status.MATURED
                for group in (*decision_groups, *advisory_groups)
            ),
        },
    )


def _observed_performance_groups(
    history: ProductHistoryRead,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Aggregate all verified runs without an unbounded SQL ``IN`` clause."""

    decision_counts: Counter[str] = Counter()
    advisory_counts: Counter[tuple[str, str]] = Counter()
    run_ids = tuple(cohort.run.id for cohort in history.cohorts)
    for offset in range(0, len(run_ids), _PERFORMANCE_RUN_BATCH_SIZE):
        run_batch = run_ids[offset : offset + _PERFORMANCE_RUN_BATCH_SIZE]
        base = PredictionOutcome.objects.filter(
            canonical_reportable_prediction_filter("prediction__"),
            prediction__analysis__run_id__in=run_batch,
        )
        decision_query = (
            base.filter(
                prediction__method_version=MOMENTUM_METHOD_VERSION,
                prediction__evidence_role=Prediction.EvidenceRole.DECISION,
            )
            .values("status")
            .annotate(count=Count("prediction"))
            .order_by("status")
        )
        for row in decision_query.iterator(chunk_size=25):
            decision_counts[str(row["status"])] += int(row["count"])
        advisory_query = (
            base.filter(
                prediction__method_version=FHS_METHOD_VERSION,
                prediction__evidence_role=Prediction.EvidenceRole.ADVISORY,
            )
            .values("prediction__horizon", "status")
            .annotate(count=Count("prediction"))
            .order_by("prediction__horizon", "status")
        )
        for row in advisory_query.iterator(chunk_size=25):
            advisory_counts[(str(row["prediction__horizon"]), str(row["status"]))] += int(
                row["count"]
            )
    return (
        [{"status": status, "count": count} for status, count in sorted(decision_counts.items())],
        [
            {
                "prediction__horizon": horizon,
                "status": status,
                "count": count,
            }
            for (horizon, status), count in sorted(advisory_counts.items())
        ],
    )


@login_required
def my_list_page(request: HttpRequest) -> HttpResponse:
    if not settings.RESEARCH_PRODUCT_ENABLED:
        if not _has_product_output():
            return legacy_views.my_list_page(request)
    owner = cast(User, request.user)
    form = TrackedSymbolForm()
    invalid = False
    if request.method == "POST":
        form = TrackedSymbolForm(request.POST)
        invalid = not form.is_valid()
        if not invalid:
            symbol = form.cleaned_data["symbol"]
            try:
                preference, created = add_tracked_symbol(owner=owner, raw_symbol=symbol)
            except TrackedSymbolValidationError as exc:
                form.add_error("symbol", str(exc))
                invalid = True
            else:
                if created:
                    messages.success(
                        request,
                        f"{preference.symbol} added. It will be considered at the next intake.",
                    )
                else:
                    messages.info(request, f"{preference.symbol} is already in My list.")
                return redirect("my-list")
    product = _read(request)
    admission_by_symbol = {item.symbol: item for item in product.admissions}
    card_by_symbol = {card.listing.provider_symbol: card for card in product.cards}
    items = [
        {
            "preference": preference,
            "admission": admission_by_symbol.get(preference.symbol),
            "card": card_by_symbol.get(preference.symbol),
            "captured": preference.symbol in admission_by_symbol,
        }
        for preference in TrackedSymbol.objects.filter(owner=owner).order_by("symbol", "created_at")
    ]
    return render(
        request,
        "web/product_my_list.html",
        {
            "product": product,
            "form": form,
            "items": items,
        },
        status=400 if invalid else 200,
    )


def _has_product_output() -> bool:
    return AnalysisRun.objects.filter(config_version="research-product-v1").exists()


@login_required
def archive_opportunities_page(request: HttpRequest) -> HttpResponse:
    run = (
        AnalysisRun.objects.exclude(config_version="research-product-v1")
        .filter(status="complete")
        .select_related("universe_snapshot")
        .order_by("-target_date", "-generated_at", "-id")
        .first()
    )
    analyses = (
        StockAnalysis.objects.none()
        if run is None
        else StockAnalysis.objects.select_related("listing__security__company")
        .filter(run=run)
        .order_by("-overall_score", "listing__ticker")[:100]
    )
    return render(
        request,
        "web/product_archive_opportunities.html",
        {"archive_run": run, "analyses": analyses},
    )


@login_required
def archive_stock_detail_page(request: HttpRequest, listing_id: UUID) -> HttpResponse:
    listing = get_object_or_404(
        Listing.objects.select_related("security__company"),
        pk=listing_id,
    )
    analysis = (
        StockAnalysis.objects.select_related("run")
        .exclude(run__config_version="research-product-v1")
        .filter(listing=listing, run__status="complete")
        .order_by("-run__target_date", "-run__generated_at", "-id")
        .first()
    )
    if analysis is None:
        raise Http404("No archived analysis exists for this listing")
    predictions = (
        Prediction.objects.exclude(analysis__run__config_version="research-product-v1")
        .filter(listing=listing)
        .order_by("-generated_at", "horizon", "id")[:100]
    )
    return render(
        request,
        "web/product_archive_stock_detail.html",
        {"listing": listing, "analysis": analysis, "predictions": predictions},
    )


@login_required
def archive_prediction_history_page(request: HttpRequest) -> HttpResponse:
    predictions = (
        Prediction.objects.select_related("listing__security__company", "analysis__run", "outcome")
        .exclude(analysis__run__config_version="research-product-v1")
        .order_by("-generated_at", "listing__ticker", "id")[:200]
    )
    return render(
        request,
        "web/product_archive_history.html",
        {"predictions": predictions},
    )


@login_required
def archive_performance_page(request: HttpRequest) -> HttpResponse:
    groups = list(
        PredictionOutcome.objects.exclude(
            prediction__analysis__run__config_version="research-product-v1"
        )
        .values(
            "prediction__analysis__run__config_version",
            "prediction__method_version",
            "prediction__evidence_role",
            "prediction__horizon",
            "status",
        )
        .annotate(count=Count("prediction"))
        .order_by(
            "prediction__analysis__run__config_version",
            "prediction__method_version",
            "prediction__evidence_role",
            "prediction__horizon",
            "status",
        )
    )
    return render(
        request,
        "web/product_archive_performance.html",
        {"groups": groups},
    )

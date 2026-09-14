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
from django.http import Http404, HttpRequest, HttpResponse, QueryDict
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
    ProductAdmission,
    ProductCard,
    ProductHistoryRead,
    ProductRead,
    ProductVerificationSession,
    read_research_product,
    read_research_product_history,
)
from stanstock.research.product_study_evidence import (
    ProductStudyComparisonRow,
    ProductStudyPartitionView,
    ProductStudyScopeView,
    read_registered_price_product_study,
)
from stanstock.research.reporting import (
    canonical_reportable_prediction_filter,
    reportable_prediction_filter,
)
from stanstock.web import views as legacy_views
from stanstock.web.forms import (
    PRODUCT_HORIZON_CHOICES,
    ResearchProductFilterForm,
    TrackedSymbolForm,
)

_PERFORMANCE_RUN_BATCH_SIZE = 200
_OPPORTUNITIES_PAGE_SIZE = 20


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
    selected_horizon = "6m"
    active_filter_values: dict[str, object] = {}
    if form.is_valid():
        active_filter_values = dict(form.cleaned_data)
        selected_horizon = str(active_filter_values["horizon"] or "6m")
        cards = _filter_cards(cards, active_filter_values)
    elif request.GET:
        cards = []
    base_filter_values = {**active_filter_values, "price_band": ""}
    band_cards = (
        _filter_cards(list(product.cards), base_filter_values)
        if active_filter_values
        else list(product.cards)
    )
    page = Paginator(cards, _OPPORTUNITIES_PAGE_SIZE).get_page(request.GET.get("page"))
    overview_cards = [
        {
            "card": card,
            "projection": next(
                projection
                for projection in card.projections
                if projection.horizon == selected_horizon
            ),
        }
        for card in page.object_list
    ]
    admission_reasons = Counter(
        item.reason_code for item in product.admissions if item.status != "admitted"
    )
    return render(
        request,
        "web/product_opportunities.html",
        {
            "product": product,
            "filter_form": form,
            "opportunity_page": page,
            "overview_cards": overview_cards,
            "selected_horizon": selected_horizon,
            "selected_horizon_label": dict(PRODUCT_HORIZON_CHOICES)[selected_horizon],
            "band_links": (
                {
                    "label": "All",
                    "count": len(band_cards),
                    "active": not active_filter_values.get("price_band"),
                    "query": _product_querystring(
                        active_filter_values,
                        price_band=None,
                        page=None,
                    ),
                },
                {
                    "label": "$10 and above",
                    "count": sum(not card.target_under_10 for card in band_cards),
                    "active": active_filter_values.get("price_band") == "at_least_10",
                    "query": _product_querystring(
                        active_filter_values,
                        price_band="at_least_10",
                        page=None,
                    ),
                },
                {
                    "label": "Under $10",
                    "count": sum(card.target_under_10 for card in band_cards),
                    "active": active_filter_values.get("price_band") == "under_10",
                    "query": _product_querystring(
                        active_filter_values,
                        price_band="under_10",
                        page=None,
                    ),
                },
            ),
            "horizon_links": tuple(
                {
                    "value": value,
                    "label": label,
                    "active": selected_horizon == value,
                    "query": _product_querystring(
                        active_filter_values,
                        horizon=value,
                        page=None,
                    ),
                }
                for value, label in PRODUCT_HORIZON_CHOICES
            ),
            "clear_filters_query": "",
            "pagination_query": _product_querystring(active_filter_values, page=None),
            "advanced_filters_active": any(
                active_filter_values.get(field) for field in ("direction", "suggestion", "risk")
            ),
            "admission_reason_counts": sorted(admission_reasons.items()),
            "unavailable_admissions": tuple(
                item for item in product.admissions if item.status != "admitted"
            ),
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


def _product_querystring(
    values: dict[str, object],
    **updates: str | int | None,
) -> str:
    """Build links from validated product filters, not arbitrary GET input."""

    query = QueryDict("", mutable=True)
    for field in ResearchProductFilterForm.base_fields:
        value = values.get(field)
        if value:
            query[field] = str(value)
    for field, value in updates.items():
        if value is None:
            query.pop(field, None)
        else:
            query[field] = str(value)
    return query.urlencode()


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
    issuance_cards = [card for cohort in history.cohorts for card in cohort.cards]
    issuance_page = Paginator(issuance_cards, _OPPORTUNITIES_PAGE_SIZE).get_page(
        request.GET.get("page")
    )
    history_entries = [_history_entry(card) for card in issuance_page.object_list]
    return render(
        request,
        "web/product_history.html",
        {
            "product": product,
            "history": history,
            "issuance_page": issuance_page,
            # Preserve the verified, per-version card projection for callers
            # that inspect the request context. The template renders the
            # compact issuance adapter below.
            "decision_cards": list(issuance_page.object_list),
            "history_entries": history_entries,
        },
    )


def _history_entry(card: ProductCard) -> dict[str, object]:
    advisory_by_horizon = {
        prediction.horizon: prediction for prediction in card.advisory_predictions
    }
    advisory_entries = []
    for projection in card.projections:
        prediction = advisory_by_horizon[projection.horizon]
        non_evaluable = _is_non_evaluable_advisory(prediction)
        advisory_entries.append(
            {
                "prediction": prediction,
                "projection": projection,
                "non_evaluable": non_evaluable,
                "outcome_label": (
                    "Not evaluable — forecast withheld"
                    if non_evaluable
                    else _outcome_label(prediction)
                ),
            }
        )
    return {
        "card": card,
        "decision_outcome_label": _outcome_label(card.decision_prediction),
        "advisories": advisory_entries,
    }


def _outcome_label(prediction: Prediction) -> str:
    if hasattr(prediction, "outcome"):
        return prediction.outcome.get_status_display()
    return "Outcome pending"


def _is_non_evaluable_advisory(prediction: Prediction) -> bool:
    return (
        prediction.evidence_role == Prediction.EvidenceRole.ADVISORY
        and prediction.bear_return is None
        and prediction.base_return is None
        and prediction.bull_return is None
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
    observed_scope: dict[str, dict[str, int]] = {
        "decision": {},
        "advisory": {},
    }
    if history.available:
        decision_groups, advisory_groups, observed_scope = _observed_performance_groups(history)
    return render(
        request,
        "web/product_performance.html",
        {
            "product": product,
            "history": history,
            "registered_study": registered_study,
            "comparison_partitions": _comparison_partitions(registered_study.partitions),
            "decision_groups": decision_groups,
            "advisory_groups": advisory_groups,
            "observed_scope": observed_scope,
            "has_matured": any(
                group["status"] == PredictionOutcome.Status.MATURED
                for group in (*decision_groups, *advisory_groups)
            ),
        },
    )


def _comparison_partitions(
    partitions: tuple[ProductStudyPartitionView, ...],
) -> tuple[dict[str, object], ...]:
    """Keep each existing baseline visible once without changing study rows."""

    return tuple(
        {
            "partition": partition,
            "scopes": tuple(
                {
                    "scope": scope,
                    "baseline_summaries": _comparison_scope_summaries(scope),
                }
                for scope in partition.scopes
            ),
        }
        for partition in partitions
    )


def _comparison_scope_summaries(
    scope: ProductStudyScopeView,
) -> tuple[dict[str, object], ...]:
    """Project existing rows for concise display; detailed tables retain all rows."""

    by_baseline: dict[str, list[ProductStudyComparisonRow]] = {}
    for row in scope.rows:
        by_baseline.setdefault(row.baseline_model, []).append(row)

    summaries = []
    for rows in by_baseline.values():
        context_row = next(
            (row for row in rows if row.metric_name == "median_absolute_error"),
            rows[0],
        )
        assessments = tuple(dict.fromkeys(row.assessment for row in rows))
        unavailable_reasons = tuple(
            dict.fromkeys(
                row.unavailable_reason for row in rows if row.unavailable_reason is not None
            )
        )
        summaries.append(
            {
                "baseline_label": context_row.baseline_label,
                "context_row": context_row,
                "assessments": assessments,
                "unavailable_reasons": unavailable_reasons,
                "has_candidate_worse": "Candidate worse" in assessments,
            }
        )
    return tuple(summaries)


def _observed_performance_groups(
    history: ProductHistoryRead,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, int]],
]:
    """Aggregate all verified runs without an unbounded SQL ``IN`` clause."""

    decision_counts: Counter[str] = Counter()
    advisory_counts: Counter[tuple[str, str]] = Counter()
    scope = {
        "decision": {
            "not_eligible_for_observed": 0,
            "eligible_canonical": 0,
            "outcome_pending": 0,
            "matured": 0,
            "unresolved": 0,
            "corporate_event": 0,
        },
        "advisory": {
            "not_eligible_for_observed": 0,
            "eligible_canonical": 0,
            "outcome_pending": 0,
            "matured": 0,
            "unresolved": 0,
            "corporate_event": 0,
            "non_evaluable": 0,
        },
    }
    run_ids = tuple(cohort.run.id for cohort in history.cohorts)
    for offset in range(0, len(run_ids), _PERFORMANCE_RUN_BATCH_SIZE):
        run_batch = run_ids[offset : offset + _PERFORMANCE_RUN_BATCH_SIZE]
        prediction_base = Prediction.objects.filter(analysis__run_id__in=run_batch)
        decision_predictions = prediction_base.filter(
            method_version=MOMENTUM_METHOD_VERSION,
            evidence_role=Prediction.EvidenceRole.DECISION,
        )
        advisory_predictions = prediction_base.filter(
            method_version=FHS_METHOD_VERSION,
            evidence_role=Prediction.EvidenceRole.ADVISORY,
        )
        non_evaluable_advisory = Q(
            bear_return__isnull=True,
            base_return__isnull=True,
            bull_return__isnull=True,
        )
        evaluable_advisory = advisory_predictions.exclude(non_evaluable_advisory)
        for name, predictions in (
            ("decision", decision_predictions),
            ("advisory", evaluable_advisory),
        ):
            canonical = predictions.filter(canonical_reportable_prediction_filter())
            scope[name]["not_eligible_for_observed"] += predictions.exclude(
                reportable_prediction_filter()
            ).count()
            scope[name]["eligible_canonical"] += canonical.count()
            scope[name]["outcome_pending"] += canonical.filter(outcome__isnull=True).count()
        scope["advisory"]["non_evaluable"] += advisory_predictions.filter(
            non_evaluable_advisory
        ).count()

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
            base.exclude(
                prediction__bear_return__isnull=True,
                prediction__base_return__isnull=True,
                prediction__bull_return__isnull=True,
            )
            .filter(
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
        for name, outcomes in (
            (
                "decision",
                base.filter(
                    prediction__method_version=MOMENTUM_METHOD_VERSION,
                    prediction__evidence_role=Prediction.EvidenceRole.DECISION,
                ),
            ),
            (
                "advisory",
                base.exclude(
                    prediction__bear_return__isnull=True,
                    prediction__base_return__isnull=True,
                    prediction__bull_return__isnull=True,
                ).filter(
                    prediction__method_version=FHS_METHOD_VERSION,
                    prediction__evidence_role=Prediction.EvidenceRole.ADVISORY,
                ),
            ),
        ):
            for status, count in (
                outcomes.values("status")
                .annotate(count=Count("prediction"))
                .values_list("status", "count")
            ):
                scope[name][str(status)] += int(count)
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
        scope,
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
                        (
                            f"{preference.symbol} added. It will be checked at the next "
                            "scheduled refresh."
                        ),
                    )
                else:
                    messages.info(request, f"{preference.symbol} is already in My list.")
                return redirect("my-list")
    product = _read(request)
    admission_by_symbol = {item.symbol: item for item in product.admissions}
    card_by_symbol = {card.listing.provider_symbol: card for card in product.cards}
    items = [
        _my_list_item(
            preference=preference,
            admission=admission_by_symbol.get(preference.symbol),
            card=card_by_symbol.get(preference.symbol),
            product=product,
        )
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


def _my_list_item(
    *,
    preference: TrackedSymbol,
    admission: ProductAdmission | None,
    card: ProductCard | None,
    product: ProductRead,
) -> dict[str, object]:
    if not product.available:
        state = "Source failed" if product.status == "integrity_failed" else "Source unavailable"
        state_detail = product.message
    elif admission is None:
        state = "Pending"
        state_detail = "Checked at the next scheduled refresh."
    elif card is not None:
        state = "Ready"
        state_detail = "Research is available for this stock."
    elif admission.missing_closes:
        state = "More history needed"
        state_detail = admission.reason_code
    else:
        state = "Research unavailable"
        state_detail = admission.reason_code
    return {
        "preference": preference,
        "admission": admission,
        "card": card,
        "state": state,
        "state_detail": state_detail,
    }


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
            "prediction__evidence_grade",
            "prediction__source_mode",
            "prediction__price_provider",
            "prediction__issued_on_time",
            "status",
        )
        .annotate(count=Count("prediction"))
        .order_by(
            "prediction__analysis__run__config_version",
            "prediction__method_version",
            "prediction__evidence_role",
            "prediction__horizon",
            "prediction__evidence_grade",
            "prediction__source_mode",
            "prediction__price_provider",
            "prediction__issued_on_time",
            "status",
        )
    )
    return render(
        request,
        "web/product_archive_performance.html",
        {"groups": groups},
    )

from __future__ import annotations

import plistlib
import sys
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import polars as pl
import pytest
from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse
from django.utils import timezone

from stanstock.core.launchd import (
    LAUNCH_AGENT_LABEL,
    SCHEDULE_TIME_LABEL,
    launch_agent_paths,
    launch_agent_status,
)
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.etfs import sync_investable_spy_from_asset
from stanstock.data.models import (
    Company,
    DataAsset,
    LatestMarketData,
    Listing,
    Region,
    Security,
    UniverseSnapshot,
)
from stanstock.data.sec_config import load_sec_fundamentals_config
from stanstock.research.affordability import (
    UNDER_10_AVAILABLE_FOUNDATIONS,
    UNDER_10_RELEASED_SHADOW_DIAGNOSTICS,
    UNDER_10_SHADOW_DISCLOSURE,
    UNDER_10_UNRELEASED_ACTIVATION_CONTROLS,
)
from stanstock.research.models import (
    AnalysisRun,
    Prediction,
    PredictionOutcome,
    Recommendation,
    RiskClass,
    StockAnalysis,
)
from stanstock.research.under10 import under10_assessment_hash, under10_policy_hash
from stanstock.simulation.models import (
    SimulationDefinition,
    SimulationHolding,
    SimulationRun,
    SimulationTrade,
)
from stanstock.web.views import STOCK_DETAIL_PREDICTIONS_PER_PAGE

# `authenticated_client`, `scheduler_status`, and `persisted_analysis` are
# shared pytest fixtures defined in `tests/conftest.py` (used by this file
# and `test_web_under10_reader_matrix.py`); no import is required or
# possible for autouse/conftest-provided fixtures -- pytest resolves them by
# name alone.


@pytest.fixture(autouse=True)
def _preverified_under10_evidence_for_structural_web_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep this module focused on cheap schema/binding and presentation.

    Its historical Under-$10 fixtures intentionally hand-assemble
    ``StockAnalysis.data_quality`` and do not persist the immutable files and
    SEC rows needed for authoritative replay. Evidence replay itself is
    exercised with real ``analyze_listing`` output in
    ``test_research_under10_pipeline.py``.
    """
    monkeypatch.setattr(
        "stanstock.web.views.under10_assessment_matches_persisted_evidence",
        lambda **_kwargs: True,
    )


def _create_prediction(
    analysis: StockAnalysis,
    *,
    horizon: Prediction.Horizon,
    evidence_role: Prediction.EvidenceRole,
    model_version: str,
    bear_return: Decimal,
    base_return: Decimal,
    bull_return: Decimal,
) -> Prediction:
    decision = Prediction.objects.get(
        analysis=analysis,
        model_version="baseline-v1",
    )
    return Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=decision.generated_at,
        target_date=decision.target_date,
        horizon=horizon,
        evidence_role=evidence_role,
        evidence_grade=decision.evidence_grade,
        source_mode=decision.source_mode,
        price_provider=decision.price_provider,
        price_subject=decision.price_subject,
        price_at_prediction=decision.price_at_prediction,
        bear_return=bear_return,
        base_return=base_return,
        bull_return=bull_return,
        probability_positive=None,
        confidence=decision.confidence,
        confidence_status="deterministic_point_in_time",
        insufficiency_reason="",
        recommendation=decision.recommendation,
        overall_score=decision.overall_score,
        component_scores=decision.component_scores,
        model_version=model_version,
        method_version="us-sec-long-v2",
        config_hash=decision.config_hash,
        data_cutoff=decision.data_cutoff,
        source_assets=decision.source_assets,
        calculation={"evidence_role": evidence_role},
        code_revision=decision.code_revision,
    )


def _prediction_history_row(content: str, model_version: str) -> str:
    normalized_content = " ".join(content.split())
    for row in normalized_content.split("<article>")[1:]:
        row = row.split("</article>", 1)[0]
        if f'<span class="badge">{model_version}</span>' in row:
            return row
    raise AssertionError(f"Prediction history row not found for {model_version}")


def _advisory_report_groups(response) -> list[dict[str, object]]:
    return [
        group
        for section in response.context["advisory_report"]["sections"]
        for group in section["groups"]
    ]


def _persist_spy_etf(tmp_path) -> Listing:
    start = date(2025, 12, 1)
    dates = [start + timedelta(days=index) for index in range(260)]
    closes = [500.0 + index * 0.25 + (index % 7) * 0.1 for index in range(260)]
    store = AssetStore(tmp_path)
    stored = store.write_frame(
        "tests/web-spy.parquet",
        pl.DataFrame(
            {
                "date": dates,
                "close": closes,
                "volume": [10_000_000 + index for index in range(260)],
            }
        ),
    )
    observed_at = datetime(2026, 9, 5, 1, tzinfo=UTC)
    asset = register_asset(
        provider="twelve_data",
        kind="price_history",
        subject="SPY",
        stored=stored,
        retrieved_at=observed_at,
        available_at=observed_at,
        period_start=dates[0],
        period_end=dates[-1],
        metadata={
            "currency": "USD",
            "mic_code": "ARCX",
            "instrument_type": "ETF",
            "interval": "1day",
            "adjustment": "splits",
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )
    return sync_investable_spy_from_asset(
        asset=asset,
        target_date=dates[-1],
        store=store,
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "name",
    [
        "opportunities",
        "market",
        "predictions",
        "performance",
        "simulations",
        "methodology",
    ],
)
def test_research_pages_require_authentication(client, name: str) -> None:
    response = client.get(reverse(name))

    assert response.status_code == 302
    assert response.url.startswith(reverse("login"))


@pytest.mark.django_db
def test_opportunities_explain_when_no_completed_analysis_exists(
    authenticated_client,
) -> None:
    response = authenticated_client.get(reverse("opportunities"))

    assert response.status_code == 200
    assert response.context["price_band_groups"] == []
    content = response.content.decode()
    assert "No completed analysis run exists." in content
    assert "No persisted analysis in this price band" not in content


@pytest.mark.django_db
def test_opportunities_filter_and_stock_detail_render_persisted_analysis(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    opportunities = authenticated_client.get(
        reverse("opportunities"),
        {
            "region": "us",
            "recommendation": "buy",
            "price_band": "50_to_300",
            "country": "US",
            "exchange": "XNAS",
            "sector": "Technology",
            "min_score": "75",
            "min_confidence": "60",
            "q": "Alpha",
        },
    )

    assert opportunities.status_code == 200
    opportunity_content = opportunities.content.decode()
    assert "Synthetic research data." in opportunity_content
    assert "SYN-A" in opportunity_content
    assert "78.50/100" in opportunity_content
    assert "$50-$300" in opportunity_content
    assert "Latest close" in opportunity_content
    assert "Heuristic evidence score" in opportunity_content
    assert "Legacy 6-12 months" in opportunity_content
    assert "Legacy 3+ years" in opportunity_content

    excluded = authenticated_client.get(
        reverse("opportunities"),
        {"min_score": "90"},
    )
    assert "SYN-A" not in excluded.content.decode()

    invalid = authenticated_client.get(
        reverse("opportunities"),
        {"min_score": "not-a-number"},
    )
    assert "Correct the filters below." in invalid.content.decode()

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert detail.status_code == 200
    content = detail.content.decode()
    assert "Quality is above the configured threshold." in content
    assert "Prediction history" in content
    assert "heuristic evidence score" in content
    assert "Legacy 6-12 month scenario" in content
    assert "Legacy 3+ year scenario" in content
    assert "Reconstructed training evidence." not in content


@pytest.mark.django_db
def test_methodology_page_discloses_policy_and_uncertainty_boundaries(
    authenticated_client,
) -> None:
    response = authenticated_client.get(reverse("methodology"))

    assert response.status_code == 200
    content = " ".join(response.content.decode().split())
    assert "fixed affine or piecewise policy map" in content
    assert "Cutler/SMA style" in content
    assert "nominal central 60% analog-return range" in content
    assert "not a calibrated prediction, credible, or confidence interval" in content
    assert "GAAP accrual proxy" in content
    assert "separate frozen horizon-specific fade and multiple-reversion paths" in content
    assert "not literature-standard, optimized, causal, or statistically calibrated" in content
    assert "YAML/config policy is bound by the stored configuration hash" in content
    assert "code-defined transforms are identified by the stored code_revision" in content
    assert "Scheduled observed production automatically binds an exact clean commit SHA" in content
    assert (
        "Demo and direct research can record working-tree unless an exact committed revision "
        "is explicitly supplied"
    ) in content
    assert "other transforms are bound to the exact code revision" not in content


@pytest.mark.django_db
def test_opportunities_display_every_price_band_including_empty_bands(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    response = authenticated_client.get(reverse("opportunities"))

    assert response.status_code == 200
    groups = response.context["price_band_groups"]
    assert [(group["slug"], group["count"]) for group in groups] == [
        ("under_10", 0),
        ("10_to_50", 0),
        ("50_to_300", 1),
        ("300_plus", 0),
    ]
    content = response.content.decode()
    assert "Under $10 - speculative watchlist" in content
    assert "$10-$50" in content
    assert "$50-$300" in content
    assert "$300+" in content
    assert content.count("No persisted analysis in this price band") == 3


@pytest.mark.django_db
def test_spy_etf_has_a_separate_market_and_detail_path(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    tmp_path,
) -> None:
    with override_settings(DATA_DIR=tmp_path):
        spy = _persist_spy_etf(tmp_path)
        market = authenticated_client.get(reverse("market"))
        detail = authenticated_client.get(reverse("etf-detail", args=[spy.pk]))
        stock_path = authenticated_client.get(reverse("stock-detail", args=[spy.pk]))
        opportunities = authenticated_client.get(reverse("opportunities"))

    assert market.status_code == 200
    assert market.context["regions"][0]["listing_count"] == 1
    market_content = market.content.decode()
    assert "ETF core" in market_content
    assert "SPY" in market_content
    assert "no stock recommendation label" in market_content

    assert detail.status_code == 200
    detail_content = detail.content.decode()
    assert "Benchmark ETF, not a stock recommendation." in detail_content
    assert "Trailing price return" in detail_content
    assert "Annualized volatility" in detail_content
    assert "Maximum drawdown" in detail_content
    assert "Core benchmark ETF" in detail_content
    assert 'class="recommendation' not in detail_content

    assert stock_path.status_code == 302
    assert stock_path.url == reverse("etf-detail", args=[spy.pk])
    assert "SPY" not in opportunities.content.decode()
    assert not StockAnalysis.objects.filter(listing=spy).exists()
    assert not Prediction.objects.filter(listing=spy).exists()


@pytest.mark.django_db
def test_great_opportunity_is_highlighted_with_versioned_policy(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    persisted_analysis.overall_score = Decimal("85")
    persisted_analysis.confidence = Decimal("70")
    persisted_analysis.risk_class = RiskClass.LOW
    persisted_analysis.data_quality = {
        **persisted_analysis.data_quality,
        "analysis_mode": "price_only_baseline",
        "fundamentals_used": False,
    }
    persisted_analysis.save(
        update_fields=[
            "overall_score",
            "confidence",
            "risk_class",
            "data_quality",
        ]
    )

    opportunities = authenticated_client.get(reverse("opportunities"))
    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert "Strong short-term setup" in opportunities.content.decode()
    assert "great-opportunity-v2" in detail.content.decode()


@pytest.mark.django_db
def test_under_10_band_blocks_promotion_and_separates_long_horizon_controls(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    _create_prediction(
        persisted_analysis,
        horizon=Prediction.Horizon.THREE_YEAR,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        model_version="immutable-long-3y-v1",
        bear_return=Decimal("-0.11"),
        base_return=Decimal("0.41"),
        bull_return=Decimal("0.91"),
    )
    _create_prediction(
        persisted_analysis,
        horizon=Prediction.Horizon.FIVE_YEAR,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        model_version="immutable-long-5y-v1",
        bear_return=Decimal("-0.19"),
        base_return=Decimal("0.79"),
        bull_return=Decimal("1.59"),
    )
    _create_prediction(
        persisted_analysis,
        horizon=Prediction.Horizon.LONG,
        evidence_role=Prediction.EvidenceRole.DECISION,
        model_version="immutable-legacy-long-v1",
        bear_return=Decimal("-0.27"),
        base_return=Decimal("0.57"),
        bull_return=Decimal("1.17"),
    )
    persisted_analysis.overall_score = Decimal("85")
    persisted_analysis.confidence = Decimal("70")
    persisted_analysis.risk_class = RiskClass.LOW
    persisted_analysis.data_quality = {
        **persisted_analysis.data_quality,
        "analysis_mode": "price_only_baseline",
        "fundamentals_used": False,
    }
    persisted_analysis.save(
        update_fields=[
            "overall_score",
            "confidence",
            "risk_class",
            "data_quality",
        ]
    )
    persisted_analysis.forecast_scenarios = {
        **persisted_analysis.forecast_scenarios,
        "horizons": {
            **persisted_analysis.forecast_scenarios.get("horizons", {}),
            "3y": {"bear": -0.12, "base": 0.45, "bull": 0.92},
            "5y": {"bear": -0.20, "base": 0.80, "bull": 1.60},
        },
    }
    persisted_analysis.save(update_fields=["forecast_scenarios"])
    market_data = LatestMarketData.objects.get(listing=persisted_analysis.listing)
    market_data.close = Decimal("9.99")
    market_data.session_date = date(2026, 9, 5)
    market_data.save(update_fields=["close", "session_date"])

    opportunities = authenticated_client.get(
        reverse("opportunities"),
        {"price_band": "under_10"},
    )
    excluded_band = authenticated_client.get(
        reverse("opportunities"),
        {"price_band": "10_to_50"},
    )
    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))
    status = authenticated_client.get(reverse("status"))

    opportunity_content = opportunities.content.decode()
    normalized_opportunity_content = " ".join(opportunity_content.split())
    assert opportunities.status_code == 200
    assert "Under $10 - speculative watchlist" in opportunity_content
    assert "0% new allocation" in opportunity_content
    assert "9.99 USD" in opportunity_content
    assert "<strong>Long-horizon (3y/5y) forecast unavailable.</strong>" in opportunity_content
    assert "<p>Available foundations:</p>" in opportunity_content
    assert (
        "<p>Released shadow diagnostic capabilities &mdash; unactivated:</p>" in opportunity_content
    )
    assert "<p>Still-unreleased activation control:</p>" in opportunity_content
    assert normalized_opportunity_content.count("Forecast unavailable") == 2
    price_band_groups = opportunities.context["price_band_groups"]
    under_10_card = price_band_groups[0]["cards"][0]
    assert under_10_card["price_band"].price_date == date(2026, 9, 5)
    assert len(under_10_card["long_horizon_available_foundations"]) == 3
    assert len(under_10_card["long_horizon_unreleased_activation_controls"]) == 1
    assert len(opportunities.context["under_10_released_shadow_diagnostics"]) == 2
    assert "Strong short-term setup" not in opportunity_content
    assert persisted_analysis.listing.ticker not in excluded_band.content.decode()
    assert (
        "Point-in-time SEC facts with adverse-versus-missing branch behavior" in opportunity_content
    )
    assert "Shadow solvency/obligation assessment with negative-FCF cash runway" in (
        opportunity_content
    )
    assert UNDER_10_SHADOW_DISCLOSURE in normalized_opportunity_content
    assert "Joint Under-$10 review and candidate-specific eligibility remain" in opportunity_content

    detail_content = detail.content.decode()
    normalized_detail_content = " ".join(detail_content.split())
    assert detail.status_code == 200
    assert normalized_detail_content.count("<strong>Forecast unavailable</strong>") == 2
    assert normalized_detail_content.count("<p>Available foundations:</p>") == 1
    assert (
        normalized_detail_content.count(
            "<p>Released shadow diagnostic capabilities &mdash; unactivated:</p>"
        )
        == 1
    )
    assert normalized_detail_content.count("<p>Still-unreleased activation control:</p>") == 1
    assert "Released foundations are not candidate approvals." in detail_content
    assert (
        "Joint Under-$10 review and candidate-specific eligibility remain"
        in normalized_detail_content
    )
    for disclosure_item in (
        *UNDER_10_AVAILABLE_FOUNDATIONS,
        *UNDER_10_RELEASED_SHADOW_DIAGNOSTICS,
        *UNDER_10_UNRELEASED_ACTIVATION_CONTROLS,
    ):
        assert opportunity_content.count(disclosure_item) == 1
        assert detail_content.count(disclosure_item) == 1
    assert UNDER_10_SHADOW_DISCLOSURE in normalized_detail_content
    assert (
        "Point-in-time SEC facts with adverse-versus-missing branch behavior "
        "(released foundation; candidate qualification still required)" in detail_content
    )
    assert (
        "Long-v2 diluted-share/per-share continuity assessment with withholding, "
        "not post-period event verification (released foundation; candidate "
        "qualification still required)" in detail_content
    )
    assert (
        "Deterministic 3-year/5-year formula engine with missing-input withholding "
        "(released foundation; candidate qualification still required)" in detail_content
    )
    assert "Shadow solvency/obligation assessment with negative-FCF cash runway" in detail_content
    assert "Shadow 252-observed-session median dollar-volume diagnostic" in detail_content
    assert "Verified split and reverse-split event source" in detail_content
    assert "Dedicated solvency and cash-runway policy" not in detail_content
    assert "Versioned Under-$10-specific dollar-liquidity policy" not in detail_content
    assert "-12.0% / +45.0% / +92.0%" not in detail_content
    assert "-20.0% / +80.0% / +160.0%" not in detail_content
    assert normalized_detail_content.count("-11.0% / +41.0% / +91.0%") == 1
    assert normalized_detail_content.count("-19.0% / +79.0% / +159.0%") == 1
    under_10_context = (
        "Immutable advisory evidence; current activation context: Under-$10 "
        "long-horizon forecast remains unavailable; joint review and "
        "candidate-specific eligibility remain outstanding."
    )
    legacy_under_10_context = (
        "Immutable legacy long-horizon evidence; current activation context: "
        "Under-$10 long-horizon forecast remains unavailable; joint review and "
        "candidate-specific eligibility remain outstanding."
    )
    legacy_row = _prediction_history_row(detail_content, "immutable-legacy-long-v1")
    three_year_row = _prediction_history_row(detail_content, "immutable-long-3y-v1")
    five_year_row = _prediction_history_row(detail_content, "immutable-long-5y-v1")
    short_row = _prediction_history_row(detail_content, "baseline-v1")
    assert "Legacy 3+ years · Decision · BUY" in legacy_row
    assert "-27.0% / +57.0% / +117.0%" in legacy_row
    assert legacy_under_10_context in legacy_row
    assert under_10_context not in legacy_row
    assert "3 years · Advisory" in three_year_row
    assert "-11.0% / +41.0% / +91.0%" in three_year_row
    assert under_10_context in three_year_row
    assert legacy_under_10_context not in three_year_row
    assert "5 years · Advisory" in five_year_row
    assert "-19.0% / +79.0% / +159.0%" in five_year_row
    assert under_10_context in five_year_row
    assert legacy_under_10_context not in five_year_row
    assert "1-10 trading days · Decision · BUY" in short_row
    assert "current activation context" not in short_row
    assert normalized_detail_content.count(legacy_under_10_context) == 1
    assert normalized_detail_content.count(under_10_context) == 2
    status_content = status.content.decode()
    assert status.status_code == 200
    assert "Forecast unavailable" in status_content
    assert "-12.0% / +45.0% / +92.0%" not in status_content
    assert "-20.0% / +80.0% / +160.0%" not in status_content
    persisted_analysis.refresh_from_db()
    assert persisted_analysis.overall_score == Decimal("85")
    assert persisted_analysis.recommendation == Recommendation.BUY


@pytest.mark.django_db
def test_stock_detail_keeps_complete_prediction_history_and_current_context_labels(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    horizon_roles = (
        (Prediction.Horizon.SHORT, Prediction.EvidenceRole.DECISION),
        (Prediction.Horizon.MEDIUM, Prediction.EvidenceRole.DECISION),
        (Prediction.Horizon.LONG, Prediction.EvidenceRole.DECISION),
        (Prediction.Horizon.SIX_MONTH, Prediction.EvidenceRole.ADVISORY),
        (Prediction.Horizon.TWELVE_MONTH, Prediction.EvidenceRole.ADVISORY),
        (Prediction.Horizon.THREE_YEAR, Prediction.EvidenceRole.ADVISORY),
        (Prediction.Horizon.FIVE_YEAR, Prediction.EvidenceRole.ADVISORY),
    )
    prediction_rows: list[tuple[Prediction, str]] = []
    for horizon_index, (horizon, evidence_role) in enumerate(horizon_roles):
        for version_index in range(5):
            value_index = horizon_index * 5 + version_index
            offset = Decimal(value_index) / Decimal("1000")
            bear_return = Decimal("-0.90") + offset
            base_return = Decimal("-0.20") + offset
            bull_return = Decimal("0.50") + offset
            prediction = _create_prediction(
                persisted_analysis,
                horizon=horizon,
                evidence_role=evidence_role,
                model_version=f"history-{horizon}-{version_index}",
                bear_return=bear_return,
                base_return=base_return,
                bull_return=bull_return,
            )
            scenario_values = " / ".join(
                f"{value * Decimal(100):+.1f}%" for value in (bear_return, base_return, bull_return)
            )
            prediction_rows.append((prediction, scenario_values))

    assert len(prediction_rows) == 35
    assert len({prediction.model_version for prediction, _ in prediction_rows}) == 35
    assert len({scenario_values for _, scenario_values in prediction_rows}) == 35

    expected_predictions = list(
        Prediction.objects.filter(listing=persisted_analysis.listing).order_by(
            "-generated_at",
            "horizon",
            "pk",
        )
    )
    expected_order = [
        (prediction.pk, prediction.model_version) for prediction in expected_predictions
    ]
    expected_combinations = {
        (
            prediction.model_version,
            prediction.horizon,
            prediction.evidence_role,
            prediction.bear_return,
            prediction.base_return,
            prediction.bull_return,
        )
        for prediction in expected_predictions
    }
    expected_page_count = (
        len(expected_predictions) + STOCK_DETAIL_PREDICTIONS_PER_PAGE - 1
    ) // STOCK_DETAIL_PREDICTIONS_PER_PAGE
    detail_url = reverse("stock-detail", args=[persisted_analysis.listing_id])

    def traverse_prediction_pages():
        responses = []
        page_number = 1
        while True:
            response = authenticated_client.get(
                detail_url,
                {"prediction_page": page_number},
            )
            assert response.status_code == 200
            prediction_page = response.context["prediction_page"]
            assert prediction_page.number == page_number
            assert response.context["predictions"] is prediction_page.object_list
            assert len(prediction_page.object_list) <= STOCK_DETAIL_PREDICTIONS_PER_PAGE
            responses.append(response)
            if not prediction_page.has_next():
                return responses
            page_number = prediction_page.next_page_number()

    def collect_prediction_rows(responses):
        ordered_rows = []
        combinations = set()
        rendered_rows = {}
        for response in responses:
            content = response.content.decode()
            for prediction in response.context["predictions"]:
                ordered_rows.append((prediction.pk, prediction.model_version))
                combinations.add(
                    (
                        prediction.model_version,
                        prediction.horizon,
                        prediction.evidence_role,
                        prediction.bear_return,
                        prediction.base_return,
                        prediction.bull_return,
                    )
                )
                row = _prediction_history_row(content, prediction.model_version)
                rendered_rows[prediction.model_version] = row
                assert prediction.get_horizon_display() in row
                assert prediction.get_evidence_role_display() in row
                scenario_values = " / ".join(
                    f"{value * Decimal(100):+.1f}%"
                    for value in (
                        prediction.bear_return,
                        prediction.base_return,
                        prediction.bull_return,
                    )
                )
                assert scenario_values in row
        return ordered_rows, combinations, rendered_rows

    market_data = LatestMarketData.objects.get(listing=persisted_analysis.listing)
    market_data.close = Decimal("9.99")
    market_data.session_date = date(2026, 9, 5)
    market_data.save(update_fields=["close", "session_date"])

    blocked_pages = traverse_prediction_pages()

    assert len(blocked_pages) == expected_page_count == 2
    assert all(response.context["long_horizon_blocked"] is True for response in blocked_pages)
    first_blocked_page = blocked_pages[0].context["prediction_page"]
    last_blocked_page = blocked_pages[-1].context["prediction_page"]
    assert len(first_blocked_page.object_list) == STOCK_DETAIL_PREDICTIONS_PER_PAGE
    assert len(last_blocked_page.object_list) == (
        len(expected_predictions) - STOCK_DETAIL_PREDICTIONS_PER_PAGE
    )
    assert first_blocked_page.has_previous() is False
    assert first_blocked_page.has_next() is True
    assert last_blocked_page.has_previous() is True
    assert last_blocked_page.has_next() is False

    first_blocked_content = " ".join(blocked_pages[0].content.decode().split())
    last_blocked_content = " ".join(blocked_pages[-1].content.decode().split())
    assert "36 predictions total · Page 1 of 2" in first_blocked_content
    assert '<nav aria-label="Prediction history pages">' in first_blocked_content
    assert ">Previous</a>" not in first_blocked_content
    assert '<a class="text-link" href="?prediction_page=2">Next</a>' in first_blocked_content
    assert "36 predictions total · Page 2 of 2" in last_blocked_content
    assert '<nav aria-label="Prediction history pages">' in last_blocked_content
    assert '<a class="text-link" href="?prediction_page=1">Previous</a>' in last_blocked_content
    assert ">Next</a>" not in last_blocked_content

    invalid_page = authenticated_client.get(
        detail_url,
        {"prediction_page": "not-a-page"},
    )
    out_of_range_page = authenticated_client.get(
        detail_url,
        {"prediction_page": expected_page_count + 10},
    )
    assert invalid_page.status_code == 200
    assert invalid_page.context["prediction_page"].number == 1
    assert out_of_range_page.status_code == 200
    assert out_of_range_page.context["prediction_page"].number == expected_page_count

    page_queryset = first_blocked_page.object_list
    loaded_fields, defer_mode = page_queryset.query.deferred_loading
    assert defer_mode is False
    assert loaded_fields == {
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
    }
    assert page_queryset.query.select_related is False
    assert StockAnalysis._meta.db_table not in {
        join.table_name for join in page_queryset.query.alias_map.values()
    }

    blocked_order, blocked_combinations, blocked_rows = collect_prediction_rows(blocked_pages)
    assert blocked_order == expected_order
    assert len({prediction_id for prediction_id, _ in blocked_order}) == len(expected_predictions)
    assert len({model_version for _, model_version in blocked_order}) == len(expected_predictions)
    assert blocked_combinations == expected_combinations
    assert set(blocked_rows) == {prediction.model_version for prediction in expected_predictions}

    legacy_context_label = "Immutable legacy long-horizon evidence; current activation context:"
    advisory_context_label = "Immutable advisory evidence; current activation context:"
    for prediction in expected_predictions:
        row = blocked_rows[prediction.model_version]
        if prediction.horizon == Prediction.Horizon.LONG:
            assert legacy_context_label in row
            assert advisory_context_label not in row
        elif prediction.horizon in (
            Prediction.Horizon.THREE_YEAR,
            Prediction.Horizon.FIVE_YEAR,
        ):
            assert advisory_context_label in row
            assert legacy_context_label not in row
        else:
            assert "current activation context" not in row
    assert (
        sum(response.content.decode().count(legacy_context_label) for response in blocked_pages)
        == 5
    )
    assert (
        sum(response.content.decode().count(advisory_context_label) for response in blocked_pages)
        == 10
    )

    market_data.close = Decimal("25")
    market_data.save(update_fields=["close"])
    unblocked_pages = traverse_prediction_pages()

    assert len(unblocked_pages) == expected_page_count
    assert all(response.context["long_horizon_blocked"] is False for response in unblocked_pages)
    unblocked_order, unblocked_combinations, unblocked_rows = collect_prediction_rows(
        unblocked_pages
    )
    assert unblocked_order == expected_order
    assert unblocked_combinations == expected_combinations
    assert set(unblocked_rows) == set(blocked_rows)
    for response in unblocked_pages:
        unblocked_content = response.content.decode()
        assert "current activation context" not in unblocked_content
        for disclosure_item in (
            *UNDER_10_AVAILABLE_FOUNDATIONS,
            *UNDER_10_RELEASED_SHADOW_DIAGNOSTICS,
            *UNDER_10_UNRELEASED_ACTIVATION_CONTROLS,
        ):
            assert disclosure_item not in unblocked_content


@pytest.mark.django_db
def test_missing_current_usd_price_band_fails_closed(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    _create_prediction(
        persisted_analysis,
        horizon=Prediction.Horizon.THREE_YEAR,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        model_version="immutable-missing-price-3y-v1",
        bear_return=Decimal("-0.13"),
        base_return=Decimal("0.43"),
        bull_return=Decimal("0.93"),
    )
    persisted_analysis.overall_score = Decimal("85")
    persisted_analysis.confidence = Decimal("70")
    persisted_analysis.risk_class = RiskClass.LOW
    persisted_analysis.data_quality = {
        **persisted_analysis.data_quality,
        "analysis_mode": "price_only_baseline",
        "fundamentals_used": False,
    }
    persisted_analysis.save(
        update_fields=[
            "overall_score",
            "confidence",
            "risk_class",
            "data_quality",
        ]
    )
    LatestMarketData.objects.filter(listing=persisted_analysis.listing).delete()

    opportunities = authenticated_client.get(reverse("opportunities"))
    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert opportunities.status_code == 200
    groups = opportunities.context["price_band_groups"]
    unavailable_group = next(group for group in groups if group["slug"] == "unavailable")
    assert unavailable_group["count"] == 1
    assert unavailable_group["new_allocation_eligible"] is False
    card = unavailable_group["cards"][0]
    assert card["opportunity"].eligible is False
    assert card["long_horizon_blocked"] is True
    assert card["long_horizon_available_foundations"] == ()
    assert card["long_horizon_unreleased_activation_controls"] == ()
    content = opportunities.content.decode()
    assert "Strong short-term setup" not in content
    assert "Forecast unavailable" in content
    assert "No valid latest persisted USD close is available" in content

    assert detail.status_code == 200
    assert detail.context["long_horizon_available_foundations"] == ()
    assert detail.context["long_horizon_unreleased_activation_controls"] == ()
    detail_content = detail.content.decode()
    normalized_detail_content = " ".join(detail_content.split())
    assert "Forecast unavailable" in detail_content
    assert "No valid latest persisted USD close is available" in detail_content
    assert "Under-$10 long-horizon policy disclosure." not in detail_content
    assert "<p>Available foundations:</p>" not in detail_content
    assert "<p>Still-unreleased activation control:</p>" not in detail_content
    for disclosure_item in (
        *UNDER_10_AVAILABLE_FOUNDATIONS,
        *UNDER_10_RELEASED_SHADOW_DIAGNOSTICS,
        *UNDER_10_UNRELEASED_ACTIVATION_CONTROLS,
    ):
        assert disclosure_item not in detail_content
    assert normalized_detail_content.count("-13.0% / +43.0% / +93.0%") == 1
    missing_price_context = (
        "Immutable advisory evidence; current activation context: No valid latest "
        "persisted USD close is available to apply the guarded price-band policy."
    )
    assert normalized_detail_content.count(missing_price_context) == 1
    assert (
        "Immutable advisory evidence; current activation context: Under-$10" not in detail_content
    )


@pytest.mark.django_db
def test_non_usd_listing_marks_usd_band_not_applicable(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    persisted_analysis.overall_score = Decimal("85")
    persisted_analysis.confidence = Decimal("70")
    persisted_analysis.risk_class = RiskClass.LOW
    persisted_analysis.data_quality = {
        **persisted_analysis.data_quality,
        "analysis_mode": "price_only_baseline",
        "fundamentals_used": False,
    }
    persisted_analysis.save(
        update_fields=[
            "overall_score",
            "confidence",
            "risk_class",
            "data_quality",
        ]
    )
    listing = persisted_analysis.listing
    listing.currency = "EUR"
    listing.save(update_fields=["currency"])

    response = authenticated_client.get(reverse("opportunities"))

    assert response.status_code == 200
    groups = response.context["price_band_groups"]
    not_applicable = next(group for group in groups if group["slug"] == "not_applicable")
    assert not_applicable["count"] == 1
    assert not_applicable["new_allocation_eligible"] is True
    assert not_applicable["cards"][0]["opportunity"].eligible is True
    content = response.content.decode()
    assert "USD band not applicable" in content
    assert "Trades in EUR" in content
    assert "Price band unavailable" not in content
    assert "Strong short-term setup" in content


@pytest.mark.django_db
def test_price_only_analysis_discloses_model_and_return_limits(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    persisted_analysis.run.config_version = "renamed-price-baseline-v2"
    persisted_analysis.run.save(update_fields=["config_version"])
    persisted_analysis.data_quality = {
        **persisted_analysis.data_quality,
        "analysis_mode": "price_only_baseline",
        "return_definition": "split_adjusted_price_return",
        "dividends_included": False,
    }
    persisted_analysis.forecast_scenarios = {
        "schema_version": 1,
        "horizons": {
            "short": persisted_analysis.short_scenario,
            "medium": persisted_analysis.medium_scenario,
            "long": persisted_analysis.long_scenario,
            "6m": {
                "bear": -0.10,
                "base": 0.08,
                "bull": 0.24,
                "probability_positive": None,
                "confidence": 55,
                "confidence_status": "empirical_range_only",
                "insufficiency_reason": "Probability withheld: effective cohorts 4/8",
                "method": "conditional_empirical_price",
                "method_version": "us-price-medium-v1",
                "support": {
                    "effective_cohorts": 4,
                    "distinct_listings": 36,
                    "fallback_level": "stock_state",
                },
                "current_state": {
                    "relative_momentum": 0.07,
                    "drawdown": -0.12,
                    "volatility": 0.25,
                    "market_trend": 0.04,
                    "market_volatility": 0.16,
                },
                "return_basis": "split_adjusted_price_return",
            },
            "12m": {
                "bear": -0.18,
                "base": 0.14,
                "bull": 0.38,
                "probability_positive": None,
                "confidence": 48,
                "confidence_status": "empirical_range_only",
                "insufficiency_reason": (
                    "Probability withheld: walk-forward calibration insufficient"
                ),
                "method": "conditional_empirical_price",
                "method_version": "us-price-medium-v1",
                "support": {
                    "effective_cohorts": 3,
                    "distinct_listings": 41,
                    "fallback_level": "relative_momentum",
                },
                "current_state": {
                    "relative_momentum": 0.07,
                    "drawdown": -0.12,
                    "volatility": 0.25,
                    "market_trend": 0.04,
                    "market_volatility": 0.16,
                },
                "return_basis": "split_adjusted_price_return",
            },
        },
    }
    persisted_analysis.save(update_fields=["data_quality", "forecast_scenarios"])

    opportunities = authenticated_client.get(reverse("opportunities"))
    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert opportunities.status_code == 200
    opportunity_content = opportunities.content.decode()
    assert "US price-only baseline." in opportunity_content
    assert "The 6- and 12-month ranges are advisory" in opportunity_content
    assert "Three- and five-year ranges are separately" in opportunity_content
    assert detail.status_code == 200
    detail_content = detail.content.decode()
    assert "The recommendation remains short-horizon" in detail_content
    assert "Split-adjusted price return; dividends excluded." in detail_content
    assert "Reconstructed training evidence." in detail_content
    assert "6-month advisory forecast" in detail_content
    assert "12-month advisory forecast" in detail_content
    assert "Legacy 3+ year scenario" in detail_content
    assert "3-year advisory forecast" not in detail_content
    assert "-10.0% / +8.0% / +24.0%" in detail_content
    assert "-18.0% / +14.0% / +38.0%" in detail_content
    assert "4 non-overlapping cohorts" in detail_content
    assert "Relative momentum +7.0%" in detail_content
    assert "Split-adjusted price return · dividends excluded" in detail_content


@pytest.mark.django_db
def test_explicit_long_forecasts_render_method_support_and_annualized_values(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    horizons = {
        **persisted_analysis.forecast_scenarios.get("horizons", {}),
        "short": persisted_analysis.short_scenario,
        "medium": persisted_analysis.medium_scenario,
        "long": persisted_analysis.long_scenario,
    }
    for horizon, years, values in (
        ("3y", 3, (-0.12, 0.45, 0.92)),
        ("5y", 5, (-0.20, 0.80, 1.60)),
    ):
        horizons[horizon] = {
            "bear": values[0],
            "base": values[1],
            "bull": values[2],
            "probability_positive": None,
            "confidence": 64,
            "confidence_status": "deterministic_point_in_time",
            "insufficiency_reason": (
                "Positive-return probability is unavailable for deterministic long-v1"
            ),
            "method": "sec_per_share_growth_multiple_reversion",
            "method_version": "us-sec-long-v1",
            "metric_family": "fcf_per_share",
            "support": {
                "peer_count": 3,
                "sic_fallback_level": 4,
                "sic_prefix": "3571",
            },
            "annualized_return": {
                "bear": (1 + values[0]) ** (1 / years) - 1,
                "base": (1 + values[1]) ** (1 / years) - 1,
                "bull": (1 + values[2]) ** (1 / years) - 1,
            },
            "return_basis": "split_adjusted_price_return",
            "evidence_grade": "observed",
            "formula_inputs": {
                "current_multiple_raw": 18.2,
                "current_multiple_capped": 18.2,
                "historical_growth_capped": 0.09,
                "sustainable_growth": 0.07,
                "peer_growth": 0.08,
            },
            "split_basis": {
                "basis": "as_filed_diluted_shares_vs_split_adjusted_price",
                "verified_through": "2026-06-30",
                "post_period_exposure_days": 66,
                "maximum_exposure_days": 200,
                "continuity_tolerance": 0.15,
                "continuity_checks": [],
                "residual_risk": "unverified_post_period_split",
            },
        }
    persisted_analysis.forecast_scenarios = {
        "schema_version": 1,
        "horizons": horizons,
    }
    persisted_analysis.save(update_fields=["forecast_scenarios"])

    opportunities = authenticated_client.get(reverse("opportunities"))
    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert opportunities.status_code == 200
    opportunity_content = opportunities.content.decode()
    assert "Legacy 3+ years" not in opportunity_content
    assert "-12.0% / +45.0% / +92.0%" in opportunity_content

    assert detail.status_code == 200
    detail_content = detail.content.decode()
    assert "3-year advisory forecast" in detail_content
    assert "5-year advisory forecast" in detail_content
    assert "us-sec-long-v1" in detail_content
    assert "Fcf Per Share" in detail_content
    assert "3 SEC peers" in detail_content
    assert "SIC-4" in detail_content
    assert "Observed evidence" in detail_content
    assert "Annualized bear/base/bull" in detail_content
    assert "Current multiple 18.2x" in detail_content
    assert "historical growth +9.0%" in detail_content
    assert "Diluted-share basis verified through 2026-06-30" in detail_content
    assert "the following 66-day interval" in detail_content
    assert "has no independent split-event verification" in detail_content
    assert "Split-adjusted price return · dividends excluded" in detail_content
    assert "Positive-return probability is unavailable" in detail_content


@pytest.mark.django_db
def test_v2_withheld_forecast_renders_assessed_split_basis_not_verified(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """CT-1 template regression: a v2 withheld/assessed split_basis renders

    the assessed/unverified wording and must never render the verified-basis
    wording reserved for a metric that actually passed continuity.
    """
    horizons = {
        **persisted_analysis.forecast_scenarios.get("horizons", {}),
        "short": persisted_analysis.short_scenario,
        "medium": persisted_analysis.medium_scenario,
        "long": persisted_analysis.long_scenario,
    }
    for horizon, years in (("3y", 3), ("5y", 5)):
        horizons[horizon] = {
            "bear": None,
            "base": None,
            "bull": None,
            "probability_positive": None,
            "confidence": 0.0,
            "confidence_status": "insufficient_evidence",
            "insufficiency_reason": (
                "Adjacent annual diluted-share basis continuity is "
                "incompatible/unverified between 2023-12-31 and 2024-12-31: "
                "100.0% exceeds 15.0%"
            ),
            "method": "sec_per_share_growth_multiple_reversion",
            "method_version": "us-sec-long-v2",
            "metric_family": None,
            "years": years,
            "support": {},
            "formula_inputs": {},
            "annualized_return": {},
            "return_basis": "split_adjusted_price_return",
            "evidence_grade": "observed",
            "split_basis": {
                "basis": "as_filed_diluted_shares_vs_split_adjusted_price",
                "assessment_status": "incompatible_or_unverified",
                "assessed_through": "2025-12-31",
                "post_period_exposure_days": 58,
                "maximum_exposure_days": 200,
                "continuity_tolerance": 0.15,
                "continuity_checks": [
                    {
                        "check": "adjacent_annual_diluted_shares",
                        "previous_period_end": "2023-12-31",
                        "current_period_end": "2024-12-31",
                        "previous_shares": 10.0,
                        "current_shares": 20.0,
                        "relative_difference": 1.0,
                        "tolerance": 0.15,
                    }
                ],
            },
        }
    persisted_analysis.forecast_scenarios = {
        "schema_version": 1,
        "horizons": horizons,
    }
    persisted_analysis.save(update_fields=["forecast_scenarios"])

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert detail.status_code == 200
    detail_content = detail.content.decode()
    assert "3-year advisory forecast" in detail_content
    assert "5-year advisory forecast" in detail_content
    assert "us-sec-long-v2" in detail_content
    assert "Diluted-share basis is Incompatible Or Unverified" in detail_content
    assert "as of 2025-12-31" in detail_content
    assert "no split-adjusted" in detail_content
    assert "verification is claimed." in detail_content
    assert "Diluted-share basis verified through" not in detail_content
    assert (
        "Adjacent annual diluted-share basis continuity is incompatible/unverified"
        in detail_content
    )


@pytest.mark.django_db
@override_settings(DEMO_MODE=False)
def test_synthetic_provenance_banner_does_not_depend_on_demo_setting(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    response = authenticated_client.get(reverse("opportunities"))

    assert response.status_code == 200
    assert persisted_analysis.listing.ticker in response.content.decode()
    assert "Synthetic research data." in response.content.decode()


@pytest.mark.django_db
@override_settings(DEMO_MODE=True)
def test_provider_backed_run_suppresses_synthetic_banner(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    now = timezone.now()
    provider_asset = DataAsset.objects.create(
        provider="twelve_data",
        kind="price_history",
        subject=persisted_analysis.listing.ticker,
        relative_path="tests/live-syn-a.parquet",
        sha256="d" * 64,
        retrieved_at=now,
        available_at=now,
    )
    persisted_analysis.data_quality = {
        "source_assets": [
            {
                "id": str(provider_asset.pk),
                "provider": "twelve_data",
                "kind": "price_history",
                "subject": persisted_analysis.listing.ticker,
            }
        ]
    }
    persisted_analysis.save(update_fields=["data_quality"])
    market = LatestMarketData.objects.get(listing=persisted_analysis.listing)
    market.source_asset = provider_asset
    market.save(update_fields=["source_asset"])
    hidden_company = Company.objects.create(name="Synthetic Hidden", country="US")
    hidden_security = Security.objects.create(company=hidden_company)
    hidden_listing = Listing.objects.create(
        security=hidden_security,
        ticker="ZZHIDDEN",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    synthetic_asset = DataAsset.objects.create(
        provider="synthetic_demo",
        kind="price_history",
        subject=hidden_listing.ticker,
        relative_path="tests/hidden-synthetic.parquet",
        sha256="e" * 64,
        retrieved_at=now,
        available_at=now,
    )
    LatestMarketData.objects.create(
        listing=hidden_listing,
        observed_at=now,
        session_date=timezone.localdate(),
        close=Decimal("50"),
        source_asset=synthetic_asset,
    )

    status = authenticated_client.get(reverse("status"))
    market_page = authenticated_client.get(reverse("market"))

    assert status.status_code == 200
    assert "Twelve Data provider-backed" in status.content.decode()
    assert "Synthetic research data." not in status.content.decode()
    assert "Daily automation" in status.content.decode()
    assert "Scheduled" in status.content.decode()
    assert market_page.status_code == 200
    assert "Twelve Data provider-backed" in market_page.content.decode()
    assert "Twelve Data" in market_page.content.decode()
    assert "twelve_data" not in market_page.content.decode()
    assert "ZZHIDDEN" not in market_page.content.decode()


@pytest.mark.django_db
def test_status_page_renders_scheduled_time_from_the_launchd_constant(
    authenticated_client,
) -> None:
    response = authenticated_client.get(reverse("status"))

    assert response.status_code == 200
    assert f"Installed for {SCHEDULE_TIME_LABEL} (Tuesday-Saturday) local time" in (
        response.content.decode()
    )
    assert "Installed for 02:00" not in response.content.decode()


@pytest.mark.django_db
def test_status_page_flags_a_stale_installed_schedule_as_attention_required(
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.web.views.launch_agent_status",
        lambda: {
            "installed": True,
            "loaded": True,
            "timezone_matches": True,
            "expected_timezone": "America/New_York",
            "installed_schedule_label": "02:00 (Tuesday-Saturday)",
            "expected_schedule_label": SCHEDULE_TIME_LABEL,
            "schedule_matches": False,
        },
    )

    response = authenticated_client.get(reverse("status"))
    body = response.content.decode()

    assert response.status_code == 200
    assert "Installed for 02:00 (Tuesday-Saturday) local time" in body
    assert f"expected {SCHEDULE_TIME_LABEL} local time" in body
    assert "reinstall the LaunchAgent to update it" in body
    assert "Attention required" in body
    assert "Scheduled" not in body


@pytest.mark.django_db
def test_status_page_never_presents_a_missing_schedule_as_installed(
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stanstock.web.views.launch_agent_status",
        lambda: {
            "installed": True,
            "loaded": True,
            "timezone_matches": True,
            "expected_timezone": "America/New_York",
            "installed_schedule_label": None,
            "expected_schedule_label": SCHEDULE_TIME_LABEL,
            "schedule_matches": False,
        },
    )

    response = authenticated_client.get(reverse("status"))
    body = response.content.decode()

    assert response.status_code == 200
    assert "The installed schedule could not be read." in body
    assert f"Installed for {SCHEDULE_TIME_LABEL} local time" not in body
    assert "Attention required" in body
    assert "Scheduled" not in body


@pytest.mark.django_db
@pytest.mark.parametrize("extra_key", ["Month", "Day"])
def test_status_page_never_certifies_a_trigger_with_an_extra_calendar_key(
    authenticated_client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_key: str,
) -> None:
    """F4 regression: a Month=1/Day=1 (or any extra) key restricts when
    launchd actually fires; the status page must never present that as a
    normal healthy Tue-Sat schedule.
    """
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    plist_path.parent.mkdir(parents=True)
    triggers = [{"Weekday": weekday, "Hour": 3, "Minute": 30} for weekday in (2, 3, 4, 5, 6)]
    triggers[0] = {**triggers[0], extra_key: 1}
    with plist_path.open("wb") as handle:
        plistlib.dump(
            {
                "Label": LAUNCH_AGENT_LABEL,
                "StartCalendarInterval": triggers,
            },
            handle,
        )
    monkeypatch.setattr(
        "stanstock.web.views.launch_agent_status",
        lambda: launch_agent_status(home),
    )

    response = authenticated_client.get(reverse("status"))
    body = response.content.decode()

    assert response.status_code == 200
    assert "The installed schedule could not be read." in body
    assert "03:30 (Tuesday-Saturday)" not in body
    assert "Scheduled" not in body
    assert "Attention required" in body


@pytest.mark.django_db
def test_status_page_never_claims_tuesday_saturday_for_a_tuesday_only_trigger(
    authenticated_client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendered-card regression using the real (unmocked) plist reader."""
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    plist_path.parent.mkdir(parents=True)
    with plist_path.open("wb") as handle:
        plistlib.dump(
            {
                "Label": LAUNCH_AGENT_LABEL,
                "StartCalendarInterval": [{"Weekday": 2, "Hour": 3, "Minute": 30}],
            },
            handle,
        )
    monkeypatch.setattr(
        "stanstock.web.views.launch_agent_status",
        lambda: launch_agent_status(home),
    )

    response = authenticated_client.get(reverse("status"))
    body = response.content.decode()

    assert response.status_code == 200
    assert "Installed for 03:30 (Tuesday) local time" in body
    assert "Installed for 03:30 (Tuesday-Saturday)" not in body
    assert "Attention required" in body
    assert "Scheduled" not in body


@pytest.mark.django_db
def test_status_page_never_certifies_stale_triggers_from_fresh_metadata(
    authenticated_client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F1 regression: metadata claims 03:30 but triggers are still 02:00."""
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    plist_path.parent.mkdir(parents=True)
    with plist_path.open("wb") as handle:
        plistlib.dump(
            {
                "Label": LAUNCH_AGENT_LABEL,
                "StartCalendarInterval": [
                    {"Weekday": weekday, "Hour": 2, "Minute": 0} for weekday in (2, 3, 4, 5, 6)
                ],
                "StanStockSchedule": {
                    "timezone": "America/New_York",
                    "hour": 3,
                    "minute": 30,
                    "weekdays": [2, 3, 4, 5, 6],
                },
                "EnvironmentVariables": {"STANSTOCK_SCHEDULE_TIMEZONE": "America/New_York"},
            },
            handle,
        )
    monkeypatch.setattr(
        "stanstock.web.views.launch_agent_status",
        lambda: launch_agent_status(home),
    )

    response = authenticated_client.get(reverse("status"))
    body = response.content.decode()

    assert response.status_code == 200
    assert "Installed for 02:00 (Tuesday-Saturday) local time" in body
    assert "Scheduled" not in body
    assert "Attention required" in body


@pytest.mark.django_db
def test_status_page_flags_timezone_metadata_runtime_conflict(
    authenticated_client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F2 regression: StanStockSchedule.timezone disagrees with the runtime env var."""
    monkeypatch.setattr(
        "stanstock.core.launchd.detect_iana_timezone",
        lambda: "America/New_York",
    )
    home = tmp_path / "home"
    plist_path, _stdout_path, _stderr_path = launch_agent_paths(home)
    plist_path.parent.mkdir(parents=True)
    with plist_path.open("wb") as handle:
        plistlib.dump(
            {
                "Label": LAUNCH_AGENT_LABEL,
                "StartCalendarInterval": [
                    {"Weekday": weekday, "Hour": 3, "Minute": 30} for weekday in (2, 3, 4, 5, 6)
                ],
                "StanStockSchedule": {"timezone": "America/New_York"},
                "EnvironmentVariables": {"STANSTOCK_SCHEDULE_TIMEZONE": "America/Los_Angeles"},
            },
            handle,
        )
    monkeypatch.setattr(
        "stanstock.web.views.launch_agent_status",
        lambda: launch_agent_status(home),
    )

    response = authenticated_client.get(reverse("status"))
    body = response.content.decode()

    assert response.status_code == 200
    assert "The machine timezone changed or does not match the installed schedule" in body
    assert "Scheduled" not in body


@pytest.mark.django_db
def test_prediction_and_performance_pages_are_truthful_about_small_samples(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    predictions = authenticated_client.get(reverse("predictions"))
    performance = authenticated_client.get(reverse("performance"))

    assert predictions.status_code == 200
    prediction_content = predictions.content.decode()
    assert persisted_analysis.listing.ticker in prediction_content
    assert "Insufficient evidence" in prediction_content
    assert "Research-grade reconstruction" in prediction_content
    assert "Decision" in prediction_content
    assert "Legacy provider not proven" in prediction_content
    assert "Positive-return estimate" in prediction_content
    assert "Support/coverage heuristic" in prediction_content

    assert performance.status_code == 200
    performance_content = " ".join(performance.content.decode().split())
    assert "Insufficient sample" in performance_content
    assert "Withheld" in performance_content
    assert "30 canonical row-level prediction observations" in performance_content
    assert "does not establish statistical validity" in performance_content
    assert "meaningful evidence" not in performance_content


@pytest.mark.django_db
def test_performance_lede_states_next_market_session_open_and_canonical_reissue_copy(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """The disclosure text must not claim reportability requires generation
    on the target calendar date -- next-market-session-open/on-time is the
    actual rule -- and must state the fixed canonical-reissue copy."""
    response = authenticated_client.get(reverse("performance"))

    assert response.status_code == 200
    # Collapse whitespace so template line-wrapping cannot break a substring
    # check that spans a wrapped line boundary.
    content = " ".join(response.content.decode().split())
    assert "generated on their target date count" not in content
    assert "next market session opened" in content
    assert "earliest reportable issuance" in content
    assert "immutable prediction ledger" in content
    assert "does not replace or recount" in content


@pytest.mark.django_db
def test_recommendation_filter_excludes_advisory_predictions(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    decision = Prediction.objects.get(analysis=persisted_analysis)
    Prediction.objects.create(
        analysis=persisted_analysis,
        listing=persisted_analysis.listing,
        generated_at=decision.generated_at,
        target_date=decision.target_date,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        price_at_prediction=decision.price_at_prediction,
        bear_return=Decimal("-0.10"),
        base_return=Decimal("0.08"),
        bull_return=Decimal("0.20"),
        probability_positive=None,
        confidence=Decimal("40"),
        confidence_status="empirical_range_only",
        insufficiency_reason="Probability withheld",
        recommendation=Recommendation.BUY,
        overall_score=decision.overall_score,
        model_version="advisory-filter-v1",
        method_version="advisory-filter-v1",
        config_hash="f" * 64,
        data_cutoff=decision.data_cutoff,
        code_revision=decision.code_revision,
    )

    decisions = authenticated_client.get(
        reverse("predictions"),
        {"recommendation": Recommendation.BUY},
    )
    incompatible = authenticated_client.get(
        reverse("predictions"),
        {
            "recommendation": Recommendation.BUY,
            "evidence_role": Prediction.EvidenceRole.ADVISORY,
        },
    )
    all_predictions = authenticated_client.get(reverse("predictions"))

    assert decisions.status_code == 200
    assert decisions.context["result_count"] == 1
    assert decisions.context["prediction_cards"][0]["prediction"].evidence_role == "decision"
    assert incompatible.status_code == 200
    assert incompatible.context["result_count"] == 0
    assert "Analog range only — probability withheld" in all_predictions.content.decode()


@pytest.mark.django_db
def test_research_grade_outcomes_are_excluded_from_live_performance(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    prediction = Prediction.objects.get(analysis=persisted_analysis)
    PredictionOutcome.objects.create(
        prediction=prediction,
        evaluated_at=timezone.now(),
        evaluation_date=timezone.localdate(),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.10"),
        benchmark_return=Decimal("0.04"),
        success=True,
        resolution="Synthetic research outcome",
    )

    response = authenticated_client.get(reverse("performance"))

    assert response.status_code == 200
    content = response.content.decode()
    assert "Matured sample</small><strong>0</strong>" in content
    assert "Research evidence excluded." in content
    assert "matured synthetic or reconstructed" in content


@pytest.mark.django_db
def test_overnight_observed_prediction_is_included_when_marked_issued_on_time(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    snapshot = persisted_analysis.run.universe_snapshot
    snapshot.grade = UniverseSnapshot.Grade.OBSERVED
    snapshot.as_of_date = date(2026, 9, 8)
    snapshot.save(update_fields=["grade", "as_of_date"])
    generated_at = datetime(2026, 9, 9, 1, tzinfo=UTC)
    run = AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=generated_at,
        target_date=date(2026, 9, 8),
        issued_on_time=True,
        universe_snapshot=snapshot,
        config_version="us-price-baseline-v1",
        config_hash="d" * 64,
        code_revision="test-revision",
    )
    analysis = StockAnalysis.objects.create(
        run=run,
        listing=persisted_analysis.listing,
        current_price=Decimal("102"),
        overall_score=Decimal("70"),
        recommendation=Recommendation.HOLD,
        risk_score=Decimal("35"),
        risk_class=RiskClass.MEDIUM,
        confidence=Decimal("60"),
    )
    prediction = Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=generated_at,
        target_date=run.target_date,
        issued_on_time=True,
        horizon=Prediction.Horizon.SHORT,
        evidence_grade=UniverseSnapshot.Grade.OBSERVED,
        source_mode=Prediction.SourceMode.PROVIDER,
        price_provider="twelve_data",
        price_subject=analysis.listing.ticker,
        price_at_prediction=Decimal("102"),
        bear_return=Decimal("-0.03"),
        base_return=Decimal("0.02"),
        bull_return=Decimal("0.07"),
        probability_positive=None,
        confidence=Decimal("60"),
        confidence_status="heuristic",
        insufficiency_reason="Insufficient comparable observations",
        recommendation=Recommendation.HOLD,
        overall_score=Decimal("70"),
        model_version="overnight-observed-v1",
        method_version="us-price-baseline-v1",
        config_hash="d" * 64,
        data_cutoff=run.data_cutoff,
        code_revision="test-revision",
    )
    PredictionOutcome.objects.create(
        prediction=prediction,
        evaluated_at=datetime(2026, 10, 1, 12, tzinfo=UTC),
        evaluation_date=date(2026, 10, 1),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.05"),
        benchmark_return=Decimal("0.02"),
        success=True,
        resolution="Observed overnight-issued outcome",
    )
    late_prediction = Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=generated_at + timedelta(hours=12),
        target_date=run.target_date,
        issued_on_time=False,
        horizon=Prediction.Horizon.SHORT,
        evidence_grade=UniverseSnapshot.Grade.OBSERVED,
        source_mode=Prediction.SourceMode.PROVIDER,
        price_provider="twelve_data",
        price_subject=analysis.listing.ticker,
        price_at_prediction=Decimal("102"),
        bear_return=Decimal("-0.03"),
        base_return=Decimal("0.02"),
        bull_return=Decimal("0.07"),
        probability_positive=None,
        confidence=Decimal("60"),
        confidence_status="heuristic",
        insufficiency_reason="Insufficient comparable observations",
        recommendation=Recommendation.HOLD,
        overall_score=Decimal("70"),
        model_version="overnight-reissued-v2",
        method_version="us-price-baseline-v1",
        config_hash="d" * 64,
        data_cutoff=run.data_cutoff,
        code_revision="test-revision",
    )
    PredictionOutcome.objects.create(
        prediction=late_prediction,
        evaluated_at=datetime(2026, 10, 1, 13, tzinfo=UTC),
        evaluation_date=date(2026, 10, 1),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.05"),
        benchmark_return=Decimal("0.02"),
        success=True,
        resolution="Late reissued outcome",
    )
    advisory_prediction = Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=generated_at,
        target_date=run.target_date,
        issued_on_time=True,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        evidence_grade=UniverseSnapshot.Grade.OBSERVED,
        source_mode=Prediction.SourceMode.PROVIDER,
        price_provider="twelve_data",
        price_subject=analysis.listing.ticker,
        price_at_prediction=Decimal("102"),
        bear_return=Decimal("-0.10"),
        base_return=Decimal("0.08"),
        bull_return=Decimal("0.25"),
        probability_positive=None,
        confidence=Decimal("60"),
        confidence_status="experimental",
        insufficiency_reason="",
        recommendation=Recommendation.HOLD,
        overall_score=Decimal("70"),
        model_version="medium-price-v1",
        method_version="medium-price-v1",
        config_hash="e" * 64,
        data_cutoff=run.data_cutoff,
        calculation={"evidence_grade": "observed"},
        code_revision="test-revision",
    )
    PredictionOutcome.objects.create(
        prediction=advisory_prediction,
        evaluated_at=datetime(2027, 3, 10, 12, tzinfo=UTC),
        evaluation_date=date(2027, 3, 10),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.10"),
        benchmark_return=Decimal("0.06"),
        success=None,
        direction_correct=True,
        interval_covered=True,
        signed_error=Decimal("0.02"),
        resolution="Observed advisory outcome",
    )
    snapshot.grade = UniverseSnapshot.Grade.RESEARCH
    snapshot.save(update_fields=["grade"])

    response = authenticated_client.get(reverse("performance"))

    assert response.status_code == 200
    assert response.context["summary"]["sample_count"] == 1
    assert response.context["summary"]["research_matured_count"] == 1
    advisory_groups = _advisory_report_groups(response)
    assert len(advisory_groups) == 1
    assert advisory_groups[0]["candidate_cohort_count"] == 1
    assert advisory_groups[0]["base_sign_match"] is None
    assert advisory_groups[0]["publishable"] is False
    content = " ".join(response.content.decode().split())
    assert "Advisory evidence" in content
    assert "Overlap-aware advisory support" in content
    assert "Candidate target cohorts" in content
    assert "Recommendation success rate" in content
    assert "Directional accuracy" not in content
    assert "Base-case sign match" in content
    assert "Analog-range inclusion" in content
    assert '<th scope="col">Mean signed base-case error</th>' in content
    assert '<th scope="col">Mean signed error</th>' not in content
    assert "BUY succeeds when actual return is greater than 0" in content
    assert "AVOID when actual return is less than or equal to 0" in content
    assert "HOLD when actual return lies within the stored bear/bull range" in content
    assert "It is not advisory base-case sign match" in content


@pytest.mark.django_db
def test_performance_keeps_malformed_matured_advisory_evidence_visible_and_withheld(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    snapshot = persisted_analysis.run.universe_snapshot
    snapshot.grade = UniverseSnapshot.Grade.OBSERVED
    snapshot.save(update_fields=["grade"])
    run = persisted_analysis.run
    run.issued_on_time = True
    run.save(update_fields=["issued_on_time"])
    listing = persisted_analysis.listing
    generated_at = run.generated_at

    issued_prediction = Prediction.objects.create(
        analysis=persisted_analysis,
        listing=listing,
        generated_at=generated_at,
        target_date=run.target_date,
        issued_on_time=True,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        evidence_grade=UniverseSnapshot.Grade.OBSERVED,
        source_mode=Prediction.SourceMode.PROVIDER,
        price_provider="twelve_data",
        price_subject=listing.ticker,
        price_at_prediction=Decimal("101.25"),
        bear_return=Decimal("-0.10"),
        base_return=Decimal("0.08"),
        bull_return=Decimal("0.25"),
        probability_positive=None,
        confidence=Decimal("60"),
        confidence_status="experimental",
        insufficiency_reason="",
        recommendation=Recommendation.HOLD,
        overall_score=Decimal("70"),
        model_version="advisory-issued-v1",
        method_version="advisory-issued-v1",
        config_hash="c" * 64,
        data_cutoff=run.data_cutoff,
        code_revision="test-revision",
    )
    PredictionOutcome.objects.create(
        prediction=issued_prediction,
        evaluated_at=datetime(2027, 3, 10, 12, tzinfo=UTC),
        evaluation_date=date(2027, 3, 10),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.10"),
        benchmark_return=Decimal("0.06"),
        success=None,
        direction_correct=True,
        interval_covered=True,
        signed_error=Decimal("0.02"),
        resolution="Observed advisory outcome",
    )
    withheld_prediction = Prediction.objects.create(
        analysis=persisted_analysis,
        listing=listing,
        generated_at=generated_at,
        target_date=run.target_date,
        issued_on_time=True,
        horizon=Prediction.Horizon.TWELVE_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        evidence_grade=UniverseSnapshot.Grade.OBSERVED,
        source_mode=Prediction.SourceMode.PROVIDER,
        price_provider="twelve_data",
        price_subject=listing.ticker,
        price_at_prediction=Decimal("101.25"),
        bear_return=None,
        base_return=None,
        bull_return=None,
        probability_positive=None,
        confidence=Decimal("60"),
        confidence_status="experimental",
        insufficiency_reason="Withheld forecast",
        recommendation=Recommendation.HOLD,
        overall_score=Decimal("70"),
        model_version="advisory-withheld-v1",
        method_version="advisory-withheld-v1",
        config_hash="c" * 64,
        data_cutoff=run.data_cutoff,
        code_revision="test-revision",
    )
    # Simulates a historical malformed row: matured despite no issued scenario.
    PredictionOutcome.objects.create(
        prediction=withheld_prediction,
        evaluated_at=datetime(2027, 3, 10, 12, tzinfo=UTC),
        evaluation_date=date(2027, 3, 10),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.10"),
        benchmark_return=Decimal("0.06"),
        success=None,
        direction_correct=True,
        interval_covered=None,
        signed_error=Decimal("0.02"),
        resolution="Malformed legacy withheld-scenario outcome",
    )
    # Same-key canonical case: earliest is withheld/all-null (non-evaluable,
    # no outcome yet); a later reissue of the exact same key is evaluable and
    # matures. The reissue is not canonical, so it must not contribute a
    # sample to this cohort even though it is individually evaluable.
    samekey_withheld_earliest = Prediction.objects.create(
        analysis=persisted_analysis,
        listing=listing,
        generated_at=generated_at,
        target_date=run.target_date,
        issued_on_time=True,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        evidence_grade=UniverseSnapshot.Grade.OBSERVED,
        source_mode=Prediction.SourceMode.PROVIDER,
        price_provider="twelve_data",
        price_subject=listing.ticker,
        price_at_prediction=Decimal("101.25"),
        bear_return=None,
        base_return=None,
        bull_return=None,
        probability_positive=None,
        confidence=Decimal("60"),
        confidence_status="experimental",
        insufficiency_reason="Withheld forecast",
        recommendation=Recommendation.HOLD,
        overall_score=Decimal("70"),
        model_version="advisory-samekey-earliest",
        method_version="advisory-samekey-v1",
        config_hash="e" * 64,
        data_cutoff=run.data_cutoff,
        code_revision="test-revision",
    )
    PredictionOutcome.objects.create(
        prediction=samekey_withheld_earliest,
        evaluated_at=datetime(2027, 3, 10, 11, tzinfo=UTC),
        evaluation_date=date(2027, 3, 10),
        status=PredictionOutcome.Status.UNRESOLVED,
        resolution="All-null earliest advisory evidence remains unresolved",
    )
    samekey_reissue = Prediction.objects.create(
        analysis=persisted_analysis,
        listing=listing,
        generated_at=generated_at + timedelta(hours=1),
        target_date=samekey_withheld_earliest.target_date,
        issued_on_time=True,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        evidence_grade=UniverseSnapshot.Grade.OBSERVED,
        source_mode=Prediction.SourceMode.PROVIDER,
        price_provider="twelve_data",
        price_subject=listing.ticker,
        price_at_prediction=Decimal("101.25"),
        bear_return=Decimal("-0.09"),
        base_return=Decimal("0.07"),
        bull_return=Decimal("0.22"),
        probability_positive=None,
        confidence=Decimal("60"),
        confidence_status="experimental",
        insufficiency_reason="",
        recommendation=Recommendation.HOLD,
        overall_score=Decimal("70"),
        model_version="advisory-samekey-reissue",
        method_version="advisory-samekey-v1",
        config_hash="e" * 64,
        data_cutoff=run.data_cutoff,
        code_revision="test-revision",
    )
    PredictionOutcome.objects.create(
        prediction=samekey_reissue,
        evaluated_at=datetime(2027, 3, 10, 12, tzinfo=UTC),
        evaluation_date=date(2027, 3, 10),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.09"),
        benchmark_return=Decimal("0.06"),
        success=None,
        direction_correct=True,
        interval_covered=True,
        signed_error=Decimal("0.02"),
        resolution="Later reissued advisory outcome for a withheld-earliest key",
    )

    response = authenticated_client.get(reverse("performance"))

    assert response.status_code == 200
    advisory_report = response.context["advisory_report"]
    assert advisory_report["summary_count"] == 2
    advisory_groups = {group["horizon"]: group for group in _advisory_report_groups(response)}
    assert set(advisory_groups) == {
        Prediction.Horizon.SIX_MONTH.value,
        Prediction.Horizon.TWELVE_MONTH.value,
    }
    assert advisory_groups[Prediction.Horizon.SIX_MONTH.value]["candidate_cohort_count"] == 1
    assert advisory_groups[Prediction.Horizon.TWELVE_MONTH.value]["effective_cohort_count"] is None
    assert (
        "Malformed target-date evidence — metrics withheld"
        in advisory_groups[Prediction.Horizon.TWELVE_MONTH.value]["withheld_reasons"]
    )
    assert "advisory-samekey-v1" not in {
        group["method_version"] for group in _advisory_report_groups(response)
    }
    content = " ".join(response.content.decode().split())
    assert "Malformed target-date evidence — metrics withheld" in content
    assert (
        "Medium-horizon inclusion reports whether realized price returns fell inside "
        "the stored analog bear-to-bull ranges."
    ) in content
    assert "60%" not in content

    status_response = authenticated_client.get(reverse("status"))
    assert status_response.status_code == 200
    # Uncanonicalized, run-scoped count: correctly includes the same-key
    # reissue's own matured advisory outcome (2 = issued_prediction +
    # samekey_reissue; samekey_withheld_earliest remains unresolved).
    assert status_response.context["advisory_matured_count"] == 2


def _reportable_prediction(
    snapshot: UniverseSnapshot,
    listing: Listing,
    *,
    generated_at: datetime,
    target_date: date,
    model_version: str,
    method_version: str = "us-price-baseline-v2",
    config_hash: str = "2" * 64,
    price_provider: str = "twelve_data",
    horizon: str = Prediction.Horizon.SHORT,
    evidence_role: str = Prediction.EvidenceRole.DECISION,
) -> Prediction:
    """Focused helper: one observed/on-time/provider-backed Prediction (with
    its own run and analysis) per call. Used by the canonical-reporting
    tests below in place of near-duplicate fixture bodies."""
    run = AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=generated_at,
        target_date=target_date,
        issued_on_time=True,
        universe_snapshot=snapshot,
        config_version=method_version,
        config_hash=config_hash,
        code_revision="test-revision",
    )
    analysis = StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("100"),
        overall_score=Decimal("70"),
        recommendation=Recommendation.HOLD,
        risk_score=Decimal("35"),
        risk_class=RiskClass.MEDIUM,
        confidence=Decimal("60"),
    )
    return Prediction.objects.create(
        analysis=analysis,
        listing=listing,
        generated_at=generated_at,
        target_date=target_date,
        issued_on_time=True,
        horizon=horizon,
        evidence_role=evidence_role,
        evidence_grade=UniverseSnapshot.Grade.OBSERVED,
        source_mode=Prediction.SourceMode.PROVIDER,
        price_provider=price_provider,
        price_subject=listing.ticker,
        price_at_prediction=Decimal("100"),
        bear_return=Decimal("-0.03"),
        base_return=Decimal("0.02"),
        bull_return=Decimal("0.07"),
        probability_positive=None,
        confidence=Decimal("60"),
        confidence_status="heuristic",
        insufficiency_reason="",
        recommendation=Recommendation.HOLD,
        overall_score=Decimal("70"),
        model_version=model_version,
        method_version=method_version,
        config_hash=config_hash,
        data_cutoff=generated_at,
        code_revision="test-revision",
    )


def _matured_decision_outcome(
    prediction: Prediction,
    *,
    actual_return: Decimal = Decimal("0.05"),
    benchmark_return: Decimal | None = Decimal("0.02"),
    success: bool | None = True,
    evaluated_at: datetime = datetime(2026, 10, 1, 12, tzinfo=UTC),
    evaluation_date: date = date(2026, 10, 1),
    resolution: str = "Observed method cohort",
) -> PredictionOutcome:
    return PredictionOutcome.objects.create(
        prediction=prediction,
        evaluated_at=evaluated_at,
        evaluation_date=evaluation_date,
        status=PredictionOutcome.Status.MATURED,
        actual_return=actual_return,
        benchmark_return=benchmark_return,
        success=success,
        resolution=resolution,
    )


def _reportable_matured_cohort(
    snapshot: UniverseSnapshot,
    listing: Listing,
    *,
    method_version: str,
    config_hash: str,
    count: int,
    generated_at: datetime,
    target_date_start: date,
) -> list[Prediction]:
    """`count` genuinely distinct reportable+matured observations sharing one
    method/config/provider cohort. Each gets its own `target_date` (and thus
    its own canonical observation key), so the cohort's canonical sample
    size is exactly `count` rather than collapsing to one reissued
    observation distinguished only by `model_version`."""
    predictions = []
    for index in range(count):
        prediction = _reportable_prediction(
            snapshot,
            listing,
            generated_at=generated_at,
            target_date=target_date_start + timedelta(days=index),
            model_version=f"{method_version}-{index}",
            method_version=method_version,
            config_hash=config_hash,
        )
        _matured_decision_outcome(prediction)
        predictions.append(prediction)
    return predictions


@pytest.mark.django_db
def test_performance_never_pools_distinct_configuration_versions(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    snapshot = persisted_analysis.run.universe_snapshot
    snapshot.grade = UniverseSnapshot.Grade.OBSERVED
    snapshot.save(update_fields=["grade"])
    listing = persisted_analysis.listing
    for method_index, (version, digest) in enumerate(
        (
            ("us-price-baseline-v1", "1" * 64),
            ("us-price-baseline-v2", "2" * 64),
        )
    ):
        # Each of the 15 rows per cohort is a genuinely distinct market
        # observation (its own target_date), not 15 reissues of one
        # observation distinguished only by `model_version` -- the canonical
        # observation key excludes `model_version`, so same-key rows would
        # collapse to a single canonical sample and silently defeat this
        # insufficient-sample proof.
        _reportable_matured_cohort(
            snapshot,
            listing,
            method_version=version,
            config_hash=digest,
            count=15,
            generated_at=datetime(2026, 9, 8 + method_index, 1, tzinfo=UTC),
            target_date_start=date(2026, 1, 1) + timedelta(days=method_index * 100),
        )

    response = authenticated_client.get(reverse("performance"))

    assert response.status_code == 200
    assert response.context["summary"]["config_version"] == "us-price-baseline-v2"
    assert response.context["summary"]["sample_count"] == 15
    assert response.context["summary"]["sufficient_sample"] is False
    groups = list(response.context["groups"])
    assert len(groups) == 2
    assert {group["sample_count"] for group in groups} == {15}
    assert "Method versions remain separate." in response.content.decode()


@pytest.mark.django_db
def test_performance_reissues_cannot_inflate_sample_sufficiency_but_a_new_observation_can(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    snapshot = persisted_analysis.run.universe_snapshot
    snapshot.grade = UniverseSnapshot.Grade.OBSERVED
    snapshot.save(update_fields=["grade"])
    listing = persisted_analysis.listing
    method_version = "us-price-baseline-v2"
    config_hash = "2" * 64
    predictions = _reportable_matured_cohort(
        snapshot,
        listing,
        method_version=method_version,
        config_hash=config_hash,
        count=29,
        generated_at=datetime(2026, 9, 8, 1, tzinfo=UTC),
        target_date_start=date(2026, 1, 1),
    )
    last = predictions[-1]
    # Reissuing the 29th observation's exact key several times must never
    # raise the canonical sample count above 29.
    for reissue_index in range(3):
        reissue = _reportable_prediction(
            snapshot,
            listing,
            generated_at=last.generated_at + timedelta(hours=reissue_index + 1),
            target_date=last.target_date,
            model_version=f"reissue-{reissue_index}",
            method_version=method_version,
            config_hash=config_hash,
        )
        _matured_decision_outcome(reissue, actual_return=Decimal("0.99"))

    insufficient = authenticated_client.get(reverse("performance"))
    assert insufficient.context["summary"]["sample_count"] == 29
    assert insufficient.context["summary"]["sufficient_sample"] is False

    # A genuinely new 30th observation (its own target_date) is not a
    # reissue of any existing key, so it does cross the threshold.
    thirtieth = _reportable_prediction(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 1, tzinfo=UTC),
        target_date=date(2026, 1, 1) + timedelta(days=29),
        model_version="genuine-30th",
        method_version=method_version,
        config_hash=config_hash,
    )
    _matured_decision_outcome(thirtieth)

    sufficient = authenticated_client.get(reverse("performance"))
    assert sufficient.context["summary"]["sample_count"] == 30
    assert sufficient.context["summary"]["sufficient_sample"] is True


@pytest.mark.django_db
def test_performance_decision_and_advisory_groups_canonicalize_same_key_reissues(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    snapshot = persisted_analysis.run.universe_snapshot
    snapshot.grade = UniverseSnapshot.Grade.OBSERVED
    snapshot.save(update_fields=["grade"])
    listing = persisted_analysis.listing
    method_version = "us-price-baseline-v2"
    config_hash = "2" * 64

    decision_target = date(2026, 9, 8)
    decision_earliest = _reportable_prediction(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 1, tzinfo=UTC),
        target_date=decision_target,
        model_version="decision-earliest",
        method_version=method_version,
        config_hash=config_hash,
    )
    _matured_decision_outcome(decision_earliest, actual_return=Decimal("0.05"))
    decision_reissue = _reportable_prediction(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 2, tzinfo=UTC),
        target_date=decision_target,
        model_version="decision-reissue",
        method_version=method_version,
        config_hash=config_hash,
    )
    _matured_decision_outcome(decision_reissue, actual_return=Decimal("0.95"))

    advisory_target = date(2026, 9, 9)
    advisory_earliest = _reportable_prediction(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 9, 1, tzinfo=UTC),
        target_date=advisory_target,
        model_version="advisory-earliest",
        method_version=method_version,
        config_hash=config_hash,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
    )
    PredictionOutcome.objects.create(
        prediction=advisory_earliest,
        evaluated_at=datetime(2027, 3, 10, 12, tzinfo=UTC),
        evaluation_date=date(2027, 3, 10),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.10"),
        benchmark_return=Decimal("0.06"),
        success=None,
        direction_correct=True,
        interval_covered=False,
        signed_error=Decimal("0.08"),
        resolution="Observed advisory outcome",
    )
    advisory_reissue = _reportable_prediction(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 9, 2, tzinfo=UTC),
        target_date=advisory_target,
        model_version="advisory-reissue",
        method_version=method_version,
        config_hash=config_hash,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
    )
    PredictionOutcome.objects.create(
        prediction=advisory_reissue,
        evaluated_at=datetime(2027, 3, 10, 13, tzinfo=UTC),
        evaluation_date=date(2027, 3, 10),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.99"),
        benchmark_return=Decimal("0.06"),
        success=None,
        direction_correct=False,
        interval_covered=False,
        signed_error=Decimal("0.90"),
        resolution="Later reissued advisory outcome",
    )

    response = authenticated_client.get(reverse("performance"))

    assert response.status_code == 200
    decision_groups = {
        group["prediction__method_version"]: group for group in response.context["groups"]
    }
    decision_group = decision_groups[method_version]
    assert decision_group["sample_count"] == 1
    assert decision_group["mean_return"] == Decimal("0.0500")

    advisory_groups = {group["horizon"]: group for group in _advisory_report_groups(response)}
    advisory_group = advisory_groups[Prediction.Horizon.SIX_MONTH.value]
    assert advisory_group["candidate_cohort_count"] == 1
    assert response.context["advisory_report"]["summary_count"] == 1


@pytest.mark.django_db
def test_performance_long_only_advisory_language_stays_distinct(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    snapshot = persisted_analysis.run.universe_snapshot
    snapshot.grade = UniverseSnapshot.Grade.OBSERVED
    snapshot.save(update_fields=["grade"])
    prediction = _reportable_prediction(
        snapshot,
        persisted_analysis.listing,
        generated_at=datetime(2026, 9, 8, 1, tzinfo=UTC),
        target_date=date(2026, 9, 8),
        model_version="long-language-v1",
        method_version="long-language-v1",
        horizon=Prediction.Horizon.THREE_YEAR,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
    )
    PredictionOutcome.objects.create(
        prediction=prediction,
        evaluated_at=datetime(2029, 9, 8, 12, tzinfo=UTC),
        evaluation_date=date(2029, 9, 8),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.10"),
        success=None,
        direction_correct=True,
        interval_covered=False,
        signed_error=Decimal("0.08"),
        resolution="Synthetic long advisory outcome",
    )

    response = authenticated_client.get(reverse("performance"))

    assert response.status_code == 200
    content = " ".join(response.content.decode().split())
    medium_section, long_section = content.split(
        '<section aria-labelledby="advisory-long-title">', maxsplit=1
    )
    assert (
        "Medium-horizon inclusion reports whether realized price returns fell inside "
        "the stored analog bear-to-bull ranges. Metrics remain withheld until the "
        "overlap-aware support floors pass."
    ) in medium_section
    assert "60%" not in content
    assert "Scenario-envelope inclusion" in long_section
    assert "deterministic scenario cases" in long_section
    assert "coverage" not in long_section.lower()
    assert "calibration" not in long_section.lower()
    assert "test-revision" in long_section
    assert "Invalid code revision — metrics withheld" in long_section
    assert "recommendation success" not in long_section.lower()


@pytest.mark.django_db
def test_performance_unsupported_advisory_section_stays_explicit_and_withheld(
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disclosure = (
        "These groups have horizons outside 6m, 12m, 3y, and 5y. "
        "Their raw identities remain visible, but all metrics stay withheld "
        "because no support floors or inclusion semantics are defined for them."
    )
    monkeypatch.setattr(
        "stanstock.web.views.advisory_support_report",
        lambda: {
            "overflow": False,
            "summary_count": 1,
            "has_evidence": True,
            "sections": [
                {
                    "key": "unsupported",
                    "title": "Unsupported advisory horizons",
                    "disclosure": disclosure,
                    "inclusion_label": "Inclusion metric (withheld)",
                    "groups": [
                        {
                            "method_version": "unsupported-short-v1",
                            "config_hash": "c" * 64,
                            "price_provider": "synthetic_provider",
                            "evidence_grade": UniverseSnapshot.Grade.OBSERVED,
                            "horizon": Prediction.Horizon.SHORT,
                            "revision_label": "raw unsupported revision",
                            "candidate_cohort_count": 1,
                            "effective_cohort_count": 1,
                            "minimum_effective_cohorts": 0,
                            "minimum_listings_per_selected_cohort": 0,
                            "target_span_days": 0,
                            "minimum_target_span_days": 0,
                            "publishable": False,
                            "base_sign_match": None,
                            "inclusion_rate": None,
                            "mean_signed_base_error": None,
                            "status_label": "Metrics withheld",
                            "withheld_reasons": (
                                "Unsupported advisory horizon — metrics withheld",
                            ),
                        }
                    ],
                }
            ],
        },
    )

    response = authenticated_client.get(reverse("performance"))

    assert response.status_code == 200
    content = " ".join(response.content.decode().split())
    unsupported = content.split(
        '<section aria-labelledby="advisory-unsupported-title">', maxsplit=1
    )[1].split("</section>", maxsplit=1)[0]
    assert '<h3 id="advisory-unsupported-title">Unsupported advisory horizons</h3>' in unsupported
    assert disclosure in unsupported
    assert '<th scope="col">Inclusion metric (withheld)</th>' in unsupported
    assert "Horizon: short" in unsupported
    assert "Metrics withheld" in unsupported
    assert "Unsupported advisory horizon — metrics withheld" in unsupported
    for forbidden in (
        "analog",
        "nominal",
        "scenario-envelope",
        "deterministic scenario",
        "recommendation success",
    ):
        assert forbidden not in unsupported.lower()


@pytest.mark.django_db
def test_performance_publishable_medium_uses_qualified_nominal_disclosure(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def publishable_report() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "overflow": False,
            "summary_count": 1,
            "has_evidence": True,
            "sections": [
                {
                    "key": "medium",
                    "title": "Medium-horizon advisory support",
                    "disclosure": (
                        "The 6- and 12-month analog ranges have a nominal 60% "
                        "analog-range target. This is not a calibration claim or "
                        "a coverage guarantee."
                    ),
                    "inclusion_label": "Analog-range inclusion",
                    "groups": [
                        {
                            "method_version": "synthetic-medium-v1",
                            "config_hash": "c" * 64,
                            "price_provider": "synthetic_provider",
                            "evidence_grade": UniverseSnapshot.Grade.OBSERVED,
                            "horizon": Prediction.Horizon.SIX_MONTH,
                            "revision_label": "a1" * 20,
                            "candidate_cohort_count": 8,
                            "effective_cohort_count": 8,
                            "minimum_effective_cohorts": 8,
                            "minimum_listings_per_selected_cohort": 30,
                            "target_span_days": 1095,
                            "minimum_target_span_days": 1095,
                            "publishable": True,
                            "base_sign_match": Decimal("0.75"),
                            "inclusion_rate": Decimal("0.625"),
                            "mean_signed_base_error": Decimal("0.01"),
                            "status_label": "Metrics published",
                            "withheld_reasons": (),
                        }
                    ],
                },
                {
                    "key": "long",
                    "title": "Long-horizon advisory support",
                    "disclosure": (
                        "The 3- and 5-year bear, base, and bull values are deterministic "
                        "scenario cases. Inclusion reports whether the realized price "
                        "return fell inside that scenario envelope."
                    ),
                    "inclusion_label": "Scenario-envelope inclusion",
                    "groups": [],
                },
            ],
        }

    monkeypatch.setattr(
        "stanstock.web.views.advisory_support_report",
        publishable_report,
    )

    response = authenticated_client.get(reverse("performance"))

    assert response.status_code == 200
    assert calls == 1
    content = " ".join(response.content.decode().split())
    assert (
        "The 6- and 12-month analog ranges have a nominal 60% analog-range target. "
        "This is not a calibration claim or a coverage guarantee."
    ) in content
    assert "Metrics published" in content


@pytest.mark.django_db
def test_performance_advisory_overflow_notice_has_no_partial_rows_and_calls_once(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def overflow_report() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "overflow": True,
            "summary_count": None,
            "has_evidence": True,
            "sections": [],
        }

    monkeypatch.setattr(
        "stanstock.web.views.advisory_support_report",
        overflow_report,
    )

    response = authenticated_client.get(reverse("performance"))

    assert response.status_code == 200
    assert calls == 1
    content = response.content.decode()
    assert "Advisory support report withheld." in content
    assert "More than 50,000 grouped target-date summaries" in content
    assert "Exact evidence identity" not in content
    assert "Base-case sign match" not in content


@pytest.mark.django_db
def test_performance_canonicalizes_while_ledger_stays_per_version(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """Aggregate performance counts one canonical observation, but the
    immutable prediction ledger keeps showing every version."""
    snapshot = persisted_analysis.run.universe_snapshot
    snapshot.grade = UniverseSnapshot.Grade.OBSERVED
    snapshot.save(update_fields=["grade"])
    listing = persisted_analysis.listing
    method_version = "us-price-baseline-v2"
    config_hash = "2" * 64
    target = date(2026, 9, 8)

    earliest = _reportable_prediction(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 1, tzinfo=UTC),
        target_date=target,
        model_version="v-earliest",
        method_version=method_version,
        config_hash=config_hash,
    )
    _matured_decision_outcome(earliest, actual_return=Decimal("0.05"))
    reissue = _reportable_prediction(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 2, tzinfo=UTC),
        target_date=target,
        model_version="v-reissue",
        method_version=method_version,
        config_hash=config_hash,
    )
    _matured_decision_outcome(reissue, actual_return=Decimal("0.09"))

    performance = authenticated_client.get(reverse("performance"))
    assert performance.status_code == 200
    assert performance.context["summary"]["sample_count"] == 1
    # The matured reissue is a valid observed duplicate, not synthetic or
    # reconstructed evidence: it must not inflate research_matured_count.
    assert performance.context["summary"]["research_matured_count"] == 0

    # Only a genuinely non-reportable (research-grade) matured decision row
    # -- the fixture's own default-grade prediction -- should move this
    # counter, proving it is not structurally zero.
    research_grade_prediction = Prediction.objects.get(analysis=persisted_analysis)
    _matured_decision_outcome(research_grade_prediction, actual_return=Decimal("0.01"))

    performance_after = authenticated_client.get(reverse("performance"))
    assert performance_after.status_code == 200
    assert performance_after.context["summary"]["research_matured_count"] == 1

    ledger = authenticated_client.get(reverse("predictions"))
    assert ledger.status_code == 200
    ledger_versions = {
        card["prediction"].model_version for card in ledger.context["prediction_cards"]
    }
    assert {"v-earliest", "v-reissue"}.issubset(ledger_versions)
    assert Prediction.objects.filter(model_version="v-reissue", issued_on_time=True).exists()
    assert PredictionOutcome.objects.filter(
        prediction__model_version="v-reissue",
        status=PredictionOutcome.Status.MATURED,
    ).exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("earliest_status", "summary_field"),
    [
        (PredictionOutcome.Status.UNRESOLVED, "unresolved_count"),
        (PredictionOutcome.Status.CORPORATE_EVENT, "corporate_event_count"),
    ],
)
def test_performance_summary_keeps_earliest_state_over_later_matured_reissue(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    earliest_status: str,
    summary_field: str,
) -> None:
    """A same-key reissue that matures must not promote past an earlier
    unresolved/corporate-event canonical row: the summary still reports
    exactly one row in the matching state field and zero matured samples."""
    snapshot = persisted_analysis.run.universe_snapshot
    snapshot.grade = UniverseSnapshot.Grade.OBSERVED
    snapshot.save(update_fields=["grade"])
    listing = persisted_analysis.listing
    method_version = "us-price-baseline-v2"
    config_hash = "2" * 64
    target = date(2026, 9, 8)

    earliest = _reportable_prediction(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 1, tzinfo=UTC),
        target_date=target,
        model_version="earliest",
        method_version=method_version,
        config_hash=config_hash,
    )
    PredictionOutcome.objects.create(
        prediction=earliest,
        evaluated_at=datetime(2026, 10, 1, tzinfo=UTC),
        evaluation_date=date(2026, 10, 1),
        status=earliest_status,
        resolution="Earliest state",
    )
    reissue = _reportable_prediction(
        snapshot,
        listing,
        generated_at=datetime(2026, 9, 8, 2, tzinfo=UTC),
        target_date=target,
        model_version="reissue",
        method_version=method_version,
        config_hash=config_hash,
    )
    _matured_decision_outcome(reissue)

    response = authenticated_client.get(reverse("performance"))

    assert response.status_code == 200
    assert response.context["summary"]["sample_count"] == 0
    assert response.context["summary"][summary_field] == 1


@pytest.mark.django_db
def test_market_overview_uses_persisted_latest_market_data(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    response = authenticated_client.get(reverse("market"))

    assert response.status_code == 200
    content = response.content.decode()
    assert persisted_analysis.listing.ticker in content
    assert "Technology" in content
    assert "+1.2%" in content


@pytest.mark.django_db
def test_prediction_admin_is_view_only(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    user = get_user_model().objects.get(username="owner")
    user.is_staff = True
    user.is_superuser = True
    user.save(update_fields=["is_staff", "is_superuser"])
    prediction = Prediction.objects.get(analysis=persisted_analysis)

    change_page = authenticated_client.get(
        reverse("admin:research_prediction_change", args=[prediction.pk])
    )
    delete_page = authenticated_client.get(
        reverse("admin:research_prediction_delete", args=[prediction.pk])
    )

    assert change_page.status_code == 200
    assert 'name="_save"' not in change_page.content.decode()
    assert delete_page.status_code == 403


@pytest.mark.django_db
def test_simulation_detail_exposes_grade_metrics_and_trade_side(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    definition = SimulationDefinition.objects.create(
        name="Synthetic top five",
        mode=SimulationDefinition.Mode.BACKTEST,
        config={"top_n": 5},
    )
    run = SimulationRun.objects.create(
        definition=definition,
        universe_snapshot=persisted_analysis.run.universe_snapshot,
        status=SimulationRun.Status.COMPLETE,
        finished_at=timezone.now(),
        code_revision="test-revision",
        input_hash="d" * 64,
        result_asset_key="simulations/test/results.parquet",
        metrics={"cumulative_return": 0.12, "max_drawdown": -0.08},
    )
    SimulationHolding.objects.create(
        run=run,
        listing=persisted_analysis.listing,
        observation_date=timezone.localdate(),
        quantity=Decimal("10"),
        price=Decimal("101.25"),
        market_value=Decimal("1012.50"),
        weight=Decimal("0.50"),
    )
    SimulationTrade.objects.create(
        run=run,
        listing=persisted_analysis.listing,
        trade_date=timezone.localdate(),
        side=SimulationTrade.Side.BUY,
        quantity=Decimal("10"),
        price=Decimal("101.25"),
        gross_value=Decimal("1012.50"),
        costs=Decimal("1.50"),
    )

    response = authenticated_client.get(reverse("simulation-detail", args=[run.pk]))

    assert response.status_code == 200
    content = response.content.decode()
    assert "Synthetic top five" in content
    assert "Research-grade reconstruction" in content
    assert "Buy" in content


# ---------------------------------------------------------------------------
# Under-$10 shadow diagnostic panel (stock detail only).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _shadow_policy_hash() -> str:
    """The authoritative ``us-under10-shadow-v1`` policy hash, computed the
    same way the reader's own C3 check does (`_expected_under10_policy_hash`
    in `stanstock.web.views`), so a hand-built `_shadow_payload()` literal
    (never run through the real generator) carries a genuinely matching
    policy hash rather than a placeholder the checksum gate would now
    correctly reject on its own, before any of the semantic branch a given
    test actually means to exercise is ever reached."""
    return under10_policy_hash(load_sec_fundamentals_config())


def _shadow_payload(**overrides: object) -> dict:
    # `evaluated_for.target_date` must equal the persisted `AnalysisRun`'s
    # own `target_date` -- `persisted_analysis` (in `conftest.py`) always
    # uses "today" (`timezone.localdate()`), so this payload's dates are
    # derived from the same call rather than a fixed calendar date, or the
    # reader's own target-date cross-check would reject every one of these
    # otherwise-legitimate fixtures on any day but one.
    target_date = timezone.localdate()
    first_session = target_date - timedelta(days=364)
    # C1: `data_cutoff` must never be *after* `persisted_analysis.run.
    # data_cutoff` (itself set to `timezone.now()` at fixture creation,
    # strictly before this payload is built) -- midnight UTC of the same
    # `target_date` (TIME_ZONE is UTC) is the latest moment guaranteed to
    # be at or before "now" on that same calendar day, unlike a fixed
    # wall-clock hour that could fall after the fixture's own capture time.
    data_cutoff = datetime.combine(target_date, datetime.min.time(), tzinfo=UTC)
    # The solvency `periods` dates must likewise stay within the reader's
    # own freshness window relative to `target_date` (0..200 days for the
    # shared instant date, same for the flow window's own end) -- these
    # preserve the exact original offsets from the date this fixture used
    # before both became "today"-relative.
    instant_date = target_date - timedelta(days=61)
    duration_start = target_date - timedelta(days=425)
    duration_end = target_date - timedelta(days=61)
    payload = {
        "schema_version": 1,
        "policy_version": "us-under10-shadow-v1",
        "policy_hash": _shadow_policy_hash(),
        "assessment_hash": "1" * 64,
        "activated": False,
        "shadow_only": True,
        "activation_eligible": False,
        "code_revision": "test-revision",
        "evaluated_for": {
            "target_date": target_date.isoformat(),
            "data_cutoff": data_cutoff.isoformat(),
            "price_band": "under_10",
            "reference_close": "4.250000",
            "date_basis": "decision_target",
            "currency": "USD",
        },
        "solvency": {
            "status": "no_adverse_evidence_observed",
            "reasons": [],
            "inputs": {
                "cash_and_equivalents": "0.00000000",
                "near_term_debt": "0.00000000",
                "current_assets": "2000.00000000",
                "current_liabilities": "1000.00000000",
                "current_ratio": "2.0000",
                "free_cash_flow": "0.00000000",
            },
            "periods": {
                "instant_date": instant_date.isoformat(),
                "duration_start": duration_start.isoformat(),
                "duration_end": duration_end.isoformat(),
                "duration_basis": "annual",
            },
            "runway": {
                "status": "not_applicable_positive_fcf",
                "quarters": None,
                "reason": None,
            },
            "assessed_fact_ids": ["11111111-1111-4111-8111-111111111111"],
            "assessed_assets": [
                {
                    "id": "22222222-2222-4222-8222-222222222222",
                    "sha256": "3" * 64,
                }
            ],
        },
        "liquidity": {
            "status": "computed",
            "metric": "median_dollar_volume_252_sessions",
            "value": 0.0,
            "currency": "USD",
            "sessions_used": 252,
            "first_session": first_session.isoformat(),
            "last_session": target_date.isoformat(),
            "basis": {
                "interval": "1day",
                "adjustment": "splits",
                "return_definition": "split_adjusted_price_return",
                "volume_basis": "provider_reported_unverified_split_basis",
            },
            "price_asset": {"id": "44444444-4444-4444-8444-444444444444", "sha256": "5" * 64},
            "reason": None,
        },
        "split_verification": {
            "status": "unavailable",
            "reason": "provider_plan_not_entitled",
            "provider": "twelve_data",
            "plan_recorded": True,
            "capability": "corporate_actions_splits",
            "inference_prohibited": True,
        },
        "gates": {
            "solvency_obligation": False,
            "dollar_liquidity_252": False,
            "verified_split_evidence": False,
        },
        "blocking_reasons": ["provider_plan_not_entitled"],
        "new_allocation_percent": 0,
    }
    payload.update(overrides)
    return payload


def _make_under_ten(
    analysis: StockAnalysis,
    *,
    assessment: dict | None,
    recompute_hash: bool = True,
    align_code_revision: bool = True,
) -> None:
    """Persist ``assessment`` as the analysis's Under-$10 payload.

    ``recompute_hash=True`` (the default) keeps ``assessment_hash`` in sync
    with whatever content the caller built or mutated, so every existing
    corruption test here continues to exercise the *semantic* branch
    validators added for the root-hardening pass -- not merely the
    accidental-corruption checksum gate, which would otherwise reject a
    mutated payload for the wrong reason and silently stop testing what it
    claims to test. Pass ``recompute_hash=False`` only for a test that is
    deliberately proving the checksum gate itself (a stale/mismatched hash
    on an otherwise well-formed payload).

    C1: a genuinely persisted assessment's parent `data_quality["price_
    source"]["asset_id"]` plus the matching `data_quality["source_assets"]`
    entry (both id *and* checksum) are, by construction, the same
    `DataAsset` as the assessment's own recorded `liquidity.price_asset`
    (see `_is_valid_under10_payload`'s C1 note), and its parent
    `AnalysisRun.data_cutoff` is, by construction, the exact same
    `decision_time` the assessment's own `evaluated_for.data_cutoff`
    records -- so this fixture derives/aligns all three from ``assessment``
    itself, exactly as `_align_run_target_date` already aligns
    `target_date`, rather than leaving every one of this helper's call
    sites to reconstruct that binding by hand. Aligning `run.data_cutoff`
    to an *earlier* value (as every payload builder here always uses) is
    always safe against the DB's own `data_cutoff <= generated_at`
    ordering check. A test that deliberately wants a *mismatching* price
    asset (to exercise the C1 transplant guard) overrides
    ``analysis.data_quality["price_source"]``/``["source_assets"]``
    itself, after calling this helper.

    F2: unlike the fields above, `AnalysisRun.data_cutoff` can be freely
    realigned to the payload's own claim, but `StockAnalysis.listing_id`
    is an existing foreign key into an already-persisted `Listing` row --
    it cannot be reassigned to an arbitrary value without that row
    existing. So this fixture aligns in the other direction: the
    assessment's own `evaluated_for.listing_id` is overridden to match
    ``analysis``'s own already-persisted listing (with the checksum
    recomputed to match, when ``recompute_hash`` is set) rather than the
    other way around. A test that deliberately wants a *mismatching*
    listing id (to exercise the F2 whole-blob-transplant guard) overrides
    ``assessment["evaluated_for"]["listing_id"]`` again after this helper
    returns, or calls it with ``recompute_hash=False``.

    The payload's fixed ``code_revision`` is likewise generated from the
    same value persisted on its parent run. By default this fixture aligns
    that field to ``analysis.run.code_revision``; a corruption test for the
    revision binding passes ``align_code_revision=False``.
    """
    analysis.current_price = Decimal("4.250000")
    quality = dict(analysis.data_quality)
    if assessment is not None:
        evaluated_for = assessment.get("evaluated_for")
        if isinstance(evaluated_for, dict):
            assessment = {
                **assessment,
                "evaluated_for": {**evaluated_for, "listing_id": str(analysis.listing_id)},
            }
        if align_code_revision:
            assessment = {
                **assessment,
                "code_revision": analysis.run.code_revision,
            }
        if recompute_hash:
            assessment = {**assessment, "assessment_hash": under10_assessment_hash(assessment)}
        quality["under10_assessment"] = assessment
        liquidity = assessment.get("liquidity")
        price_asset = liquidity.get("price_asset") if isinstance(liquidity, dict) else None
        asset_id = price_asset.get("id") if isinstance(price_asset, dict) else None
        asset_sha256 = price_asset.get("sha256") if isinstance(price_asset, dict) else None
        if isinstance(asset_id, str):
            quality["price_source"] = {"asset_id": asset_id}
            if isinstance(asset_sha256, str):
                quality["source_assets"] = [{"id": asset_id, "sha256": asset_sha256}]
        evaluated_for = assessment.get("evaluated_for")
        raw_cutoff = evaluated_for.get("data_cutoff") if isinstance(evaluated_for, dict) else None
        if isinstance(raw_cutoff, str):
            try:
                data_cutoff = datetime.fromisoformat(raw_cutoff)
            except ValueError:
                data_cutoff = None
            if data_cutoff is not None:
                analysis.run.data_cutoff = data_cutoff
                analysis.run.save(update_fields=["data_cutoff"])
    analysis.data_quality = quality
    analysis.save(update_fields=["current_price", "data_quality"])
    market_data = LatestMarketData.objects.get(listing=analysis.listing)
    market_data.close = Decimal("4.25")
    market_data.save(update_fields=["close"])


def _insufficient_evidence_payload(*reasons: str) -> dict:
    """A legitimately shaped `insufficient_evidence` payload naming ``reasons``.

    `insufficient_evidence`'s own reason vocabulary is architecturally
    unbounded (it spans multiple modules and dynamically concept-named
    strings, and cannot itself misrepresent a favorable claim), so this is
    the correct, non-contradictory status/input combination for tests that
    need an arbitrary (long, or HTML-shaped) reason string to legitimately
    reach the "recorded" panel -- unlike `no_adverse_evidence_observed`/
    `elevated_obligation_risk`/`adverse_near_term_obligation`, which are now
    held to the generator's own closed reason vocabulary for that state.
    """
    payload = _shadow_payload()
    payload["solvency"] = {
        **payload["solvency"],
        "status": "insufficient_evidence",
        "reasons": list(reasons),
        "inputs": dict.fromkeys(payload["solvency"]["inputs"]),
        "runway": {
            "status": "withheld",
            "quarters": None,
            "reason": "cash_and_equivalents_missing",
        },
    }
    return payload


def _make_under_ten_raw(analysis: StockAnalysis, *, raw_value: object) -> None:
    """Like `_make_under_ten`, but the key is always set -- even to ``None``.

    `_make_under_ten(assessment=None)` deliberately never sets the key at
    all (the "absent" case). This helper constructs the distinct "key
    present but the stored value itself is malformed" case.
    """
    analysis.current_price = Decimal("4.250000")
    quality = dict(analysis.data_quality)
    quality["under10_assessment"] = raw_value
    analysis.data_quality = quality
    analysis.save(update_fields=["current_price", "data_quality"])
    market_data = LatestMarketData.objects.get(listing=analysis.listing)
    market_data.close = Decimal("4.25")
    market_data.save(update_fields=["close"])


@pytest.mark.django_db
def test_under_ten_panel_reports_assessed_state_and_renders_zero_explicitly(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    _make_under_ten(persisted_analysis, assessment=_shadow_payload())

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    content = " ".join(detail.content.decode().split())
    assert panel["state"] == "recorded"
    assert panel["solvency"]["state"] == "assessed"
    assert "Assessed - no adverse evidence observed" in content
    assert "Not applicable - FCF is non-negative." in content
    # Zero renders as an explicit value, never as a blank or a default.
    assert "0.00000000" in content
    assert "Assessed - median dollar volume over 252 observed sessions" in content
    assert "0.0 USD over 252 observed sessions" in content
    assert "Withheld - verified split evidence is unavailable." in content
    assert "recorded Twelve Data Basic plan is not entitled" in content
    assert "A different plan alone would not supply a reviewed split source" in content
    assert "New allocation remains 0%" in content
    # No badge, ranking, or scorecard is introduced by the panel.
    assert "opportunity-badge" not in detail.content.decode()


@pytest.mark.django_db
def test_under_ten_panel_reports_withheld_solvency_with_reasons(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    payload = _shadow_payload()
    payload["solvency"] = {
        **payload["solvency"],
        "status": "insufficient_evidence",
        "reasons": ["near_term_debt_components_missing", "stale_metric"],
        "inputs": {key: None for key in payload["solvency"]["inputs"]},
        "runway": {
            "status": "withheld",
            "quarters": None,
            "reason": "cash_and_equivalents_missing",
        },
    }
    payload["liquidity"] = {
        **payload["liquidity"],
        "status": "withheld",
        "value": None,
        "reason": "basis_incompatible",
        "sessions_used": None,
        "first_session": None,
        "last_session": None,
        "basis": {
            "interval": None,
            "adjustment": None,
            "return_definition": None,
            "volume_basis": "provider_reported_unverified_split_basis",
        },
    }
    payload["split_verification"] = {
        **payload["split_verification"],
        "reason": "no_reviewed_corporate_actions_source",
        "plan_recorded": False,
    }
    payload["blocking_reasons"] = ["no_reviewed_corporate_actions_source"]
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    content = " ".join(detail.content.decode().split())
    assert panel["solvency"]["state"] == "withheld"
    assert panel["liquidity"]["state"] == "withheld"
    assert "Withheld - insufficient evidence" in content
    assert "near_term_debt_components_missing" in content
    assert "stale_metric" in content
    assert "Withheld - basis_incompatible" in content
    assert "Withheld - cash_and_equivalents_missing" in content
    assert "No reviewed corporate-actions source is integrated" in content
    assert "0.00000000" not in content


@pytest.mark.django_db
def test_under_ten_panel_renders_the_elevated_obligation_risk_state(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """Regression: hardening the reader must not reject a genuinely valid, complete,
    non-favorable-but-not-worst-case state -- only the two states this file
    already covers (`no_adverse_evidence_observed`, `insufficient_evidence`).
    """
    payload = _shadow_payload()
    payload["solvency"] = {
        **payload["solvency"],
        "status": "elevated_obligation_risk",
        "reasons": ["near_term_debt_exceeds_cash"],
        "inputs": {
            **payload["solvency"]["inputs"],
            "near_term_debt": "2000.00000000",
        },
    }
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    content = " ".join(detail.content.decode().split())
    assert panel["state"] == "recorded"
    assert panel["solvency"]["state"] == "assessed"
    assert "Assessed - elevated obligation risk" in content
    assert "near_term_debt_exceeds_cash" in content


@pytest.mark.django_db
def test_under_ten_panel_renders_the_adverse_near_term_obligation_state(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    payload = _shadow_payload()
    payload["solvency"] = {
        **payload["solvency"],
        "status": "adverse_near_term_obligation",
        "reasons": [
            "near_term_debt_exceeds_cash",
            "negative_free_cash_flow",
            "cash_runway_below_minimum_quarters",
        ],
        "inputs": {
            **payload["solvency"]["inputs"],
            "near_term_debt": "2000.00000000",
            "free_cash_flow": "-100.00000000",
        },
        "runway": {
            "status": "computed",
            "quarters": "0.0000",
            "reason": None,
        },
    }
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    content = " ".join(detail.content.decode().split())
    assert panel["state"] == "recorded"
    assert panel["solvency"]["state"] == "assessed"
    assert "Assessed - adverse near-term obligation" in content
    assert "Cash runway - 0.0000 quarters at the reported burn" in content


@pytest.mark.django_db
def test_under_ten_panel_says_not_assessed_when_the_key_is_absent(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    _make_under_ten(persisted_analysis, assessment=None)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    content = detail.content.decode()
    assert panel["state"] == "not_assessed"
    assert "Not assessed for this analysis." in content
    assert "existing analyses were not backfilled" in content
    assert "Assessed -" not in content


@pytest.mark.django_db
def test_under_ten_panel_withholds_an_unreadable_payload(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    _make_under_ten(
        persisted_analysis,
        assessment=_shadow_payload(policy_version="us-under10-shadow-v9"),
    )

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    content = detail.content.decode()
    assert panel["state"] == "unsupported"
    assert "this build cannot read" in content
    assert "Assessed -" not in content
    assert "Not assessed for this analysis." not in content


# ---------------------------------------------------------------------------
# RI-3: the reader must not accept a malformed/incomplete/forged payload.
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_under_ten_panel_distinguishes_key_absent_from_key_present_but_null(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """A present ``None`` is "unsupported", never conflated with "not assessed"."""
    _make_under_ten_raw(persisted_analysis, raw_value=None)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    content = detail.content.decode()
    assert panel["state"] == "unsupported"
    assert "Not assessed for this analysis." not in content
    assert "this build cannot read" in content


@pytest.mark.django_db
@pytest.mark.parametrize("raw_value", [[], "under10_assessment", 4.25, True])
def test_under_ten_panel_rejects_a_present_non_dict_value(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    raw_value: object,
) -> None:
    _make_under_ten_raw(persisted_analysis, raw_value=raw_value)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert detail.context["under10_panel"]["state"] == "unsupported"


@pytest.mark.django_db
@pytest.mark.parametrize("forged_schema_version", [99, 0, -1, True, "1", 1.0, None])
def test_under_ten_panel_rejects_an_unrecognized_or_non_integer_schema_version(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    forged_schema_version: object,
) -> None:
    """Schema 99 (or any non-genuine-int) is withheld, not silently accepted.

    ``True`` is included because ``bool`` is an ``int`` subclass in Python
    (``True == 1``); a forged boolean schema version must still be rejected
    as the wrong *type*, not accepted because it compares equal to ``1``.
    """
    _make_under_ten(
        persisted_analysis,
        assessment=_shadow_payload(schema_version=forged_schema_version),
    )

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert detail.context["under10_panel"]["state"] == "unsupported"


@pytest.mark.django_db
def test_under_ten_panel_rejects_minimal_solvency_instead_of_rendering_favorable(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """An incomplete payload must never be able to render favorable evidence.

    Only ``status`` is present; every other required solvency field
    (``reasons``, ``inputs``, ``periods``, ``runway``, ``assessed_fact_ids``)
    is missing. The favorable-looking status alone must not be enough to
    render "Assessed - no adverse evidence observed".
    """
    payload = _shadow_payload()
    payload["solvency"] = {"status": "no_adverse_evidence_observed"}
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    content = detail.content.decode()
    assert panel["state"] == "unsupported"
    assert "Assessed - no adverse evidence observed" not in content
    assert "No adverse evidence observed" not in content


@pytest.mark.django_db
@pytest.mark.parametrize(
    "unhashable_status",
    [[], {}, ["no_adverse_evidence_observed"]],
    ids=["empty-list", "empty-dict", "single-item-list"],
)
def test_under_ten_panel_rejects_an_unhashable_solvency_status_without_raising(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    unhashable_status: object,
) -> None:
    """A malformed unhashable stored ``status`` must render ``unsupported``, never raise.

    `status in UNDER10_SOLVENCY_LABELS` (a ``dict``) hashes its left operand;
    an unhashable stored value (``list``/``dict``) would raise ``TypeError``
    without a ``str`` type guard ahead of the membership test, turning a
    malformed nested payload into an uncaught 500 instead of the intended
    read-only "unsupported" render. ``list``/``dict`` are used here (rather
    than e.g. a ``set``) because both are genuine JSON types that can
    actually round-trip through the persisted `data_quality` JSON column.
    """
    payload = _shadow_payload()
    payload["solvency"] = {**payload["solvency"], "status": unhashable_status}
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert detail.status_code == 200
    assert detail.context["under10_panel"]["state"] == "unsupported"


@pytest.mark.parametrize(
    "unhashable_status",
    [[], {}, set()],
    ids=["list", "dict", "set"],
)
def test_is_valid_under10_solvency_rejects_unhashable_status_values(
    unhashable_status: object,
) -> None:
    """Direct validator-function coverage, including a ``set`` (JSON cannot represent one).

    A ``set`` can never survive a JSON round trip (it would raise at
    persistence time, never at read time), so this exercises the reader's
    own defense directly rather than through an HTTP round trip that could
    never construct the row in the first place -- matching the existing
    NaN/Infinity liquidity precedent above.
    """
    from stanstock.web.views import _is_valid_under10_solvency

    payload = {**_shadow_payload()["solvency"], "status": unhashable_status}

    assert _is_valid_under10_solvency(payload, target_date=timezone.localdate()) is False


@pytest.mark.django_db
@pytest.mark.parametrize(
    "malformed_inputs",
    [
        {"cash_and_equivalents": True},
        {"near_term_debt": -5.0},
        {"current_assets": 2000.0},
    ],
)
def test_under_ten_panel_rejects_non_string_solvency_input_values(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    malformed_inputs: dict[str, object],
) -> None:
    """Every solvency input is a Decimal-compatible string or ``None``, never a raw number/bool."""
    payload = _shadow_payload()
    payload["solvency"] = {
        **payload["solvency"],
        "inputs": {**payload["solvency"]["inputs"], **malformed_inputs},
    }
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert detail.context["under10_panel"]["state"] == "unsupported"


@pytest.mark.django_db
@pytest.mark.parametrize(
    "malformed_liquidity",
    [
        {"value": True},
        {"value": -1_000_000.0},
        {"status": "withheld", "value": 500.0},
        {"status": "computed", "value": None},
        {"currency": "EUR"},
        {"sessions_used": True},
        {"sessions_used": -1},
        {"status": "withheld", "value": None, "reason": None},
        {"status": "withheld", "value": None, "reason": ""},
        {"status": "computed", "reason": "insufficient_sessions"},
    ],
)
def test_under_ten_panel_rejects_malformed_liquidity_combinations(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    malformed_liquidity: dict[str, object],
) -> None:
    """Booleans, negatives, and invalid status combinations never render as computed.

    NaN/Infinity liquidity values are covered separately, at the validator
    function, in `test_is_valid_under10_liquidity_rejects_nan_and_infinite_values`:
    SQLite's own `JSON_VALID` column constraint (and PostgreSQL's `json`/
    `jsonb` types) already refuse to persist a non-finite JSON number
    through a normal `.save()`, so this HTTP-level test cannot construct
    that row -- the reader's own defense is still exercised directly.
    """
    payload = _shadow_payload()
    payload["liquidity"] = {**payload["liquidity"], **malformed_liquidity}
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    assert panel["state"] == "unsupported"
    assert "median dollar volume over 252 observed sessions" not in detail.content.decode()


@pytest.mark.django_db
def test_under_ten_panel_never_renders_withheld_none_for_a_missing_liquidity_reason(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """A withheld liquidity result always names why; a null reason is withheld, not rendered.

    Every generated withholding carries a non-empty reason string, so a
    stored `status=withheld` with a null/absent `reason` is a malformed
    payload, not merely an incomplete but legitimate one -- it must never
    surface a bare "Withheld - None" (or similarly blank) label.
    """
    payload = _shadow_payload()
    payload["liquidity"] = {**payload["liquidity"], "status": "withheld", "value": None}
    del payload["liquidity"]["reason"]
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    content = detail.content.decode()
    assert panel["state"] == "unsupported"
    assert "Withheld - None" not in content


@pytest.mark.django_db
def test_under_ten_panel_withheld_liquidity_with_a_real_reason_still_renders(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """The tightened reason check does not regress a genuinely withheld result."""
    payload = _shadow_payload()
    payload["liquidity"] = {
        **payload["liquidity"],
        "status": "withheld",
        "value": None,
        "reason": "insufficient_sessions",
        "sessions_used": 0,
        "first_session": None,
        "last_session": None,
    }
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    content = detail.content.decode()
    assert panel["state"] == "recorded"
    assert panel["liquidity"]["state"] == "withheld"
    assert "Withheld - insufficient_sessions" in content


@pytest.mark.parametrize("nonfinite_value", [float("nan"), float("inf"), float("-inf")])
def test_is_valid_under10_liquidity_rejects_nan_and_infinite_values(nonfinite_value: float) -> None:
    """Direct validator-function coverage for NaN/Infinity liquidity values.

    A non-finite JSON number cannot actually reach a persisted
    `StockAnalysis` row (SQLite's `JSON_VALID` column constraint, and
    PostgreSQL's `json`/`jsonb` types, both refuse it), so this exercises
    the reader's own defense directly rather than through an HTTP round
    trip that could never construct the row in the first place.
    """
    from datetime import date

    from stanstock.web.views import _ExpectedPriceAssetReference, _is_valid_under10_liquidity

    full_payload = _shadow_payload()
    payload = full_payload["liquidity"]
    payload["value"] = nonfinite_value
    target_date = date.fromisoformat(full_payload["evaluated_for"]["target_date"])
    price_asset = payload.get("price_asset")
    expected_price_asset = (
        _ExpectedPriceAssetReference(id=price_asset["id"], sha256=price_asset["sha256"])
        if isinstance(price_asset, dict)
        else None
    )

    assert (
        _is_valid_under10_liquidity(
            payload, target_date=target_date, expected_price_asset=expected_price_asset
        )
        is False
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "malformed_runway",
    [
        {"status": "computed", "quarters": None},
        {"status": "withheld", "quarters": "1.0000"},
        {"status": "not_applicable_positive_fcf", "quarters": "1.0000"},
        {"status": "favorable-forged-status"},
    ],
)
def test_under_ten_panel_rejects_invalid_runway_status_combinations(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    malformed_runway: dict[str, object],
) -> None:
    payload = _shadow_payload()
    payload["solvency"] = {
        **payload["solvency"],
        "runway": {**payload["solvency"]["runway"], **malformed_runway},
    }
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert detail.context["under10_panel"]["state"] == "unsupported"


@pytest.mark.django_db
def test_under_ten_panel_rejects_a_non_boolean_plan_recorded(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    payload = _shadow_payload()
    payload["split_verification"] = {**payload["split_verification"], "plan_recorded": "true"}
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert detail.context["under10_panel"]["state"] == "unsupported"


@pytest.mark.django_db
def test_under_ten_panel_preserves_valid_zero_values(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """A genuine zero must survive validation and render explicitly, not as absent."""
    payload = _shadow_payload()
    payload["solvency"] = {
        **payload["solvency"],
        "inputs": {
            **payload["solvency"]["inputs"],
            "cash_and_equivalents": "0.00000000",
        },
    }
    payload["liquidity"] = {**payload["liquidity"], "value": 0.0}
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    assert panel["state"] == "recorded"
    assert panel["liquidity"]["state"] == "assessed"
    assert "0.00000000" in detail.content.decode()


@pytest.mark.django_db
def test_under_ten_panel_rejects_a_forged_allocation_percent(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """A noncanonical stored allocation mirror invalidates the payload."""
    _make_under_ten(
        persisted_analysis,
        assessment=_shadow_payload(new_allocation_percent=100),
    )

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    content = " ".join(detail.content.decode().split())
    assert panel["state"] == "unsupported"
    assert "New allocation remains 0%" in content
    assert "New allocation remains 100%" not in content


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("forged_field", "forged_value"),
    [("activated", True), ("activation_eligible", True)],
)
def test_under_ten_panel_rejects_a_forged_activation_flag(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    forged_field: str,
    forged_value: object,
) -> None:
    """A forged fixed policy flag is unsupported, never merely hidden."""
    _make_under_ten(
        persisted_analysis,
        assessment=_shadow_payload(**{forged_field: forged_value}),
    )

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    assert panel["state"] == "unsupported"


@pytest.mark.django_db
def test_under_ten_panel_activation_flags_are_authoritative_for_a_valid_payload(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """Future-safe: even a genuinely-generated payload's flags are never read from storage.

    The template does not currently render `activated`/`activation_eligible`
    at all, so this asserts the returned *context* directly: the values are
    correct today and stay correct if a future template starts rendering
    them, because the view never reads them from the stored payload in the
    first place.
    """
    _make_under_ten(persisted_analysis, assessment=_shadow_payload())

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    assert panel["state"] == "recorded"
    assert panel["activated"] is False
    assert panel["activation_eligible"] is False


@pytest.mark.django_db
def test_under_ten_panel_still_renders_a_valid_persisted_assessment_after_hardening(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """The stricter reader is not stricter than the actual generated contract."""
    _make_under_ten(persisted_analysis, assessment=_shadow_payload())

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert detail.status_code == 200
    panel = detail.context["under10_panel"]
    assert panel["state"] == "recorded"
    assert panel["solvency"]["state"] == "assessed"
    assert panel["liquidity"]["state"] == "assessed"


@pytest.mark.django_db
def test_under_ten_panel_detail_get_stays_query_bounded_for_malformed_payloads(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    django_assert_max_num_queries,
) -> None:
    """Rejecting a malformed payload must not add unbounded queries to the GET."""
    _make_under_ten(
        persisted_analysis,
        assessment=_shadow_payload(schema_version=99),
    )

    with django_assert_max_num_queries(60):
        detail = authenticated_client.get(
            reverse("stock-detail", args=[persisted_analysis.listing_id])
        )

    assert detail.status_code == 200
    assert detail.context["under10_panel"]["state"] == "unsupported"


@pytest.mark.django_db
def test_a_recorded_assessment_survives_a_later_price_band_change(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    _make_under_ten(persisted_analysis, assessment=_shadow_payload())
    market_data = LatestMarketData.objects.get(listing=persisted_analysis.listing)
    market_data.close = Decimal("42.00")
    market_data.save(update_fields=["close"])

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    assert detail.context["current_price_band"].slug == "10_to_50"
    assert panel["state"] == "recorded"
    assert panel["reference_close"] == "4.250000"


@pytest.mark.django_db
def test_a_later_under_ten_band_never_manufactures_an_old_assessment(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    # The decision-run close stays above $10 while the current market row
    # falls into the Under-$10 band.
    market_data = LatestMarketData.objects.get(listing=persisted_analysis.listing)
    market_data.close = Decimal("4.25")
    market_data.save(update_fields=["close"])

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    assert persisted_analysis.current_price == Decimal("101.25")
    assert detail.context["current_price_band"].slug == "under_10"
    assert panel["state"] == "not_assessed"
    assert "Not assessed for this analysis." in detail.content.decode()


@pytest.mark.django_db
def test_an_ordinary_priced_analysis_shows_no_under_ten_panel(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert detail.context["under10_panel"] is None
    assert "Under-$10 shadow diagnostics" not in detail.content.decode()


@pytest.mark.django_db
def test_detail_get_performs_no_assessment_write_or_provider_access(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stanstock.research.under10 as under10_module

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("stock detail must not compute an assessment on GET")

    monkeypatch.setattr(under10_module, "build_under10_assessment", _forbidden)
    _make_under_ten(persisted_analysis, assessment=_shadow_payload())
    before = StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert detail.status_code == 200
    assert StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality == before


@pytest.mark.django_db
def test_under_ten_panel_never_appears_on_the_opportunities_page(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    _make_under_ten(persisted_analysis, assessment=_shadow_payload())

    opportunities = authenticated_client.get(reverse("opportunities"))

    content = opportunities.content.decode()
    assert "Under-$10 shadow diagnostics" not in content
    assert "Assessed - no adverse evidence observed" not in content
    assert "median dollar volume over 252 observed sessions" not in content
    for card in opportunities.context["analysis_cards"]:
        assert "under10_panel" not in card


# ---------------------------------------------------------------------------
# F1: the numeric validator must be total (never raise) for any JSON scalar.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (10**400, False),
        (-(10**400), False),
        (float("nan"), False),
        (float("inf"), False),
        (float("-inf"), False),
        (True, False),
        (False, False),
        ("4250000.0", False),
        (None, False),
        ([], False),
        ({}, False),
        (0, True),
        (0.0, True),
        (-5.5, True),
        (4_250_000.0, True),
        (sys.float_info.max, True),
        (-sys.float_info.max, True),
    ],
)
def test_is_finite_number_never_raises_for_any_json_scalar(value: object, expected: bool) -> None:
    """`10**400` (and its negation) must not raise `OverflowError` inside `math.isfinite`."""
    from stanstock.web.views import _is_finite_number

    assert _is_finite_number(value) is expected


@pytest.mark.django_db
@pytest.mark.parametrize("oversized_value", [10**400, -(10**400)])
def test_under_ten_panel_rejects_an_oversized_stored_liquidity_value_without_raising(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    django_assert_max_num_queries,
    oversized_value: int,
) -> None:
    """A persisted `10**400` liquidity value must render `unsupported`, not crash."""
    payload = _shadow_payload()
    payload["liquidity"] = {**payload["liquidity"], "value": oversized_value}
    _make_under_ten(persisted_analysis, assessment=payload)
    before = StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality

    with django_assert_max_num_queries(60):
        detail = authenticated_client.get(
            reverse("stock-detail", args=[persisted_analysis.listing_id])
        )

    assert detail.status_code == 200
    panel = detail.context["under10_panel"]
    assert panel["state"] == "unsupported"
    content = " ".join(detail.content.decode().split())
    assert "New allocation remains 0%" in content
    assert str(oversized_value) not in content
    assert StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality == before


# ---------------------------------------------------------------------------
# F2: malformed/contradictory evidence must never render assessed/favorable.
# ---------------------------------------------------------------------------


def _assert_rejected(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    payload: dict,
    django_assert_max_num_queries,
) -> None:
    before = StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality
    with django_assert_max_num_queries(60):
        detail = authenticated_client.get(
            reverse("stock-detail", args=[persisted_analysis.listing_id])
        )
    assert detail.status_code == 200
    panel = detail.context["under10_panel"]
    assert panel["state"] == "unsupported"
    content = " ".join(detail.content.decode().split())
    assert "Assessed -" not in content
    assert "New allocation remains 0%" in content
    assert StockAnalysis.objects.get(pk=persisted_analysis.pk).data_quality == before


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("cash_and_equivalents", "NaN"),
        ("free_cash_flow", "Infinity"),
        ("free_cash_flow", "-Infinity"),
        ("current_ratio", "1e100000000"),
        ("near_term_debt", "-1e100000000"),
        ("current_assets", True),
    ],
)
def test_under_ten_panel_rejects_nonfinite_or_extreme_solvency_inputs(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    django_assert_max_num_queries,
    field: str,
    bad_value: object,
) -> None:
    payload = _shadow_payload()
    payload["solvency"] = {
        **payload["solvency"],
        "inputs": {**payload["solvency"]["inputs"], field: bad_value},
    }
    _make_under_ten(persisted_analysis, assessment=payload)
    _assert_rejected(
        authenticated_client, persisted_analysis, payload, django_assert_max_num_queries
    )


def test_is_valid_under10_solvency_rejects_a_raw_nan_input_value() -> None:
    """A raw (non-string) NaN can never be persisted (SQLite's `JSON_VALID` refuses it),

    so this exercises the reader's own defense directly, matching the
    existing NaN/Infinity liquidity precedent above.
    """
    from stanstock.web.views import _is_valid_under10_solvency

    payload = _shadow_payload()["solvency"]
    payload["inputs"] = {**payload["inputs"], "current_liabilities": float("nan")}

    assert _is_valid_under10_solvency(payload, target_date=timezone.localdate()) is False


@pytest.mark.django_db
def test_under_ten_panel_rejects_a_null_input_while_solvency_claims_no_adverse_evidence(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    django_assert_max_num_queries,
) -> None:
    """A favorable status requires the complete input set; a null cash is not complete."""
    payload = _shadow_payload()
    assert payload["solvency"]["status"] == "no_adverse_evidence_observed"
    payload["solvency"] = {
        **payload["solvency"],
        "inputs": {**payload["solvency"]["inputs"], "cash_and_equivalents": None},
    }
    _make_under_ten(persisted_analysis, assessment=payload)
    _assert_rejected(
        authenticated_client, persisted_analysis, payload, django_assert_max_num_queries
    )


@pytest.mark.django_db
def test_under_ten_panel_rejects_a_null_period_while_solvency_claims_no_adverse_evidence(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    django_assert_max_num_queries,
) -> None:
    """A favorable status requires the shared instant/flow period claims too."""
    payload = _shadow_payload()
    assert payload["solvency"]["status"] == "no_adverse_evidence_observed"
    payload["solvency"] = {
        **payload["solvency"],
        "periods": {**payload["solvency"]["periods"], "instant_date": None},
    }
    _make_under_ten(persisted_analysis, assessment=payload)
    _assert_rejected(
        authenticated_client, persisted_analysis, payload, django_assert_max_num_queries
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "malformed_periods",
    [
        {"instant_date": "not-a-date"},
        {"duration_start": "2025-13-45"},
        {"duration_basis": "quarterly"},
        {"duration_start": "2025-12-31", "duration_end": "2025-01-01"},
    ],
)
def test_under_ten_panel_rejects_malformed_period_claims(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    django_assert_max_num_queries,
    malformed_periods: dict[str, object],
) -> None:
    payload = _shadow_payload()
    payload["solvency"] = {
        **payload["solvency"],
        "periods": {**payload["solvency"]["periods"], **malformed_periods},
    }
    _make_under_ten(persisted_analysis, assessment=payload)
    _assert_rejected(
        authenticated_client, persisted_analysis, payload, django_assert_max_num_queries
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "malformed_liquidity",
    [
        {"sessions_used": 0},
        {"sessions_used": 251},
        {
            "basis": {
                "interval": "1day",
                "adjustment": "none",
                "return_definition": "split_adjusted_price_return",
                "volume_basis": "provider_reported_unverified_split_basis",
            }
        },
        {"metric": "average_dollar_volume_30_sessions"},
        {"first_session": "2026-03-02", "last_session": "2025-06-24"},
        {"first_session": "not-a-date"},
    ],
)
def test_under_ten_panel_rejects_malformed_computed_liquidity_claims(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    django_assert_max_num_queries,
    malformed_liquidity: dict[str, object],
) -> None:
    payload = _shadow_payload()
    assert payload["liquidity"]["status"] == "computed"
    payload["liquidity"] = {**payload["liquidity"], **malformed_liquidity}
    _make_under_ten(persisted_analysis, assessment=payload)
    _assert_rejected(
        authenticated_client, persisted_analysis, payload, django_assert_max_num_queries
    )


@pytest.mark.django_db
def test_under_ten_panel_rejects_an_unknown_liquidity_metric_even_when_withheld(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    django_assert_max_num_queries,
) -> None:
    payload = _shadow_payload()
    payload["liquidity"] = {
        "status": "withheld",
        "metric": "average_dollar_volume_30_sessions",
        "value": None,
        "currency": "USD",
        "sessions_used": None,
        "first_session": None,
        "last_session": None,
        "basis": {
            "interval": None,
            "adjustment": None,
            "return_definition": None,
            "volume_basis": "provider_reported_unverified_split_basis",
        },
        "reason": "price_provenance_unavailable",
    }
    _make_under_ten(persisted_analysis, assessment=payload)
    _assert_rejected(
        authenticated_client, persisted_analysis, payload, django_assert_max_num_queries
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "malformed_runway",
    [
        {"status": "computed", "quarters": "NaN", "reason": None},
        {"status": "computed", "quarters": "-1.0000", "reason": None},
        {"status": "computed", "quarters": 4.0, "reason": None},
        {"status": "computed", "quarters": "4.0000", "reason": "unexpected_reason"},
        {"status": "withheld", "quarters": "4.0000", "reason": "free_cash_flow_missing"},
    ],
)
def test_under_ten_panel_rejects_malformed_computed_runway_strings(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    django_assert_max_num_queries,
    malformed_runway: dict[str, object],
) -> None:
    payload = _shadow_payload()
    payload["solvency"] = {
        **payload["solvency"],
        "runway": {**payload["solvency"]["runway"], **malformed_runway},
    }
    _make_under_ten(persisted_analysis, assessment=payload)
    _assert_rejected(
        authenticated_client, persisted_analysis, payload, django_assert_max_num_queries
    )


@pytest.mark.django_db
def test_under_ten_panel_rejects_an_extreme_reference_close(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    django_assert_max_num_queries,
) -> None:
    payload = _shadow_payload()
    payload["evaluated_for"] = {**payload["evaluated_for"], "reference_close": "1e100000000"}
    _make_under_ten(persisted_analysis, assessment=payload)
    _assert_rejected(
        authenticated_client, persisted_analysis, payload, django_assert_max_num_queries
    )


# ---------------------------------------------------------------------------
# F3: the split-only price-basis "proven" claim must be conditional on what
# was actually validated compatible, never unconditional.
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_under_ten_liquidity_panel_confirms_price_basis_when_computed(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    payload = _shadow_payload()
    assert payload["liquidity"]["status"] == "computed"
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    content = " ".join(detail.content.decode().split())
    assert "Split-only price basis is confirmed for this evidence." in content
    assert "not for the provider's reported volume" not in content
    assert "Price/volume split basis was not established" not in content


@pytest.mark.django_db
def test_under_ten_liquidity_panel_confirms_price_basis_when_withheld_for_an_unrelated_reason(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """Withheld for insufficient sessions, but the price basis itself was still established."""
    payload = _shadow_payload()
    payload["liquidity"] = {
        **payload["liquidity"],
        "status": "withheld",
        "value": None,
        "sessions_used": 200,
        "reason": "insufficient_sessions",
    }
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    content = " ".join(detail.content.decode().split())
    assert "Split-only price basis is confirmed for this evidence." in content
    assert "Price/volume split basis was not established" not in content


@pytest.mark.django_db
def test_under_ten_liquidity_panel_denies_the_proof_claim_for_a_missing_anchor(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    payload = _shadow_payload()
    payload["liquidity"] = {
        **payload["liquidity"],
        "status": "withheld",
        "value": None,
        "sessions_used": None,
        "first_session": None,
        "last_session": None,
        "reason": "price_provenance_unavailable",
        "basis": {
            "interval": None,
            "adjustment": None,
            "return_definition": None,
            "volume_basis": "provider_reported_unverified_split_basis",
        },
        "price_asset": None,
    }
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    content = " ".join(detail.content.decode().split())
    assert "Price/volume split basis was not established for this evidence" in content
    assert "Split-only price basis is confirmed" not in content


@pytest.mark.django_db
def test_under_ten_liquidity_panel_denies_the_proof_claim_for_incompatible_present_metadata(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    """P3-1: the observed-but-incompatible basis is preserved (not nulled), and the
    proof claim must still be denied for it -- present-but-wrong is not proof.
    """
    payload = _shadow_payload()
    payload["liquidity"] = {
        **payload["liquidity"],
        "status": "withheld",
        "value": None,
        "sessions_used": None,
        "first_session": None,
        "last_session": None,
        "reason": "basis_incompatible",
        "basis": {
            "interval": "1week",
            "adjustment": "splits",
            "return_definition": "split_adjusted_price_return",
            "volume_basis": "provider_reported_unverified_split_basis",
        },
    }
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    panel = detail.context["under10_panel"]
    content = " ".join(detail.content.decode().split())
    # The observed (incompatible) metadata is still visibly preserved, not nulled.
    assert panel["liquidity"]["basis"]["interval"] == "1week"
    assert "Price/volume split basis was not established for this evidence" in content
    assert "Split-only price basis is confirmed" not in content


# ---------------------------------------------------------------------------
# F4: long/malicious stored text must never cause horizontal page overflow,
# and must remain autoescaped. Uses the repo's existing Playwright dependency
# (already declared, no new dependency) when its Chromium browser is
# installed in this environment; otherwise these regressions skip cleanly
# and only the CSS-contract assertion below still runs unconditionally.
# ---------------------------------------------------------------------------

_STANSTOCK_CSS_PATH = Path(__file__).resolve().parent.parent / "static" / "css" / "stanstock.css"
_LONG_UNBROKEN_REASON = "x" * 1000


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as playwright:
            return Path(playwright.chromium.executable_path).exists()
    except Exception:
        return False


def _viewport_overflow(html: str, *, viewport_width: int) -> tuple[int, int]:
    """``(scrollWidth, clientWidth)`` for ``html`` styled with the real project CSS."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": viewport_width, "height": 900})
            page.set_content(html)
            page.add_style_tag(path=str(_STANSTOCK_CSS_PATH))
            scroll_width = page.evaluate("document.documentElement.scrollWidth")
            client_width = page.evaluate("document.documentElement.clientWidth")
        finally:
            browser.close()
    return scroll_width, client_width


def test_under10_panel_css_rule_exists_and_contains_overflow_wrap() -> None:
    """Applied CSS contract, checked unconditionally (no browser required)."""
    css = _STANSTOCK_CSS_PATH.read_text()
    assert ".under10-panel" in css
    rule_start = css.index(".under10-panel")
    rule_end = css.index("}", rule_start)
    rule_body = css[rule_start:rule_end]
    assert "overflow-wrap: anywhere" in rule_body


@pytest.mark.django_db
@pytest.mark.skipif(not _chromium_available(), reason="Playwright's Chromium is not installed")
@pytest.mark.parametrize("viewport_width", [320, 375, 768, 1280])
def test_under_ten_panel_never_overflows_for_a_long_unbroken_stored_reason(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    viewport_width: int,
) -> None:
    payload = _insufficient_evidence_payload(_LONG_UNBROKEN_REASON)
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))
    assert detail.status_code == 200
    html = detail.content.decode()
    # Required: audit text is not hidden or truncated -- it still renders in full.
    assert _LONG_UNBROKEN_REASON in html

    scroll_width, client_width = _viewport_overflow(html, viewport_width=viewport_width)
    assert scroll_width <= client_width, (
        f"horizontal overflow at {viewport_width}px: "
        f"scrollWidth={scroll_width} clientWidth={client_width}"
    )


@pytest.mark.django_db
@pytest.mark.skipif(not _chromium_available(), reason="Playwright's Chromium is not installed")
@pytest.mark.parametrize("viewport_width", [320, 375, 768, 1280])
def test_under_ten_panel_ordinary_valid_payload_never_overflows(
    authenticated_client,
    persisted_analysis: StockAnalysis,
    viewport_width: int,
) -> None:
    """Baseline: the CSS containment fix must not disturb an ordinary rendered panel."""
    _make_under_ten(persisted_analysis, assessment=_shadow_payload())

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))
    assert detail.status_code == 200
    html = detail.content.decode()

    scroll_width, client_width = _viewport_overflow(html, viewport_width=viewport_width)
    assert scroll_width <= client_width


@pytest.mark.django_db
@pytest.mark.skipif(not _chromium_available(), reason="Playwright's Chromium is not installed")
def test_opportunities_page_ordinary_layout_never_overflows(authenticated_client) -> None:
    """Baseline: an unrelated ordinary page's layout is unaffected by the panel-scoped rule."""
    detail = authenticated_client.get(reverse("opportunities"))
    assert detail.status_code == 200
    html = detail.content.decode()

    scroll_width, client_width = _viewport_overflow(html, viewport_width=375)
    assert scroll_width <= client_width


@pytest.mark.django_db
def test_under_ten_panel_escapes_html_in_a_stored_reason(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    malicious = "<script>window.__stanstock_xss__=true</script>"
    payload = _insufficient_evidence_payload(malicious)
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert detail.status_code == 200
    html = detail.content.decode()
    assert malicious not in html
    assert "&lt;script&gt;" in html


@pytest.mark.django_db
@pytest.mark.skipif(not _chromium_available(), reason="Playwright's Chromium is not installed")
def test_under_ten_panel_never_executes_a_stored_script_in_the_dom(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    malicious = "<script>window.__stanstock_xss__=true</script>"
    payload = _insufficient_evidence_payload(malicious)
    _make_under_ten(persisted_analysis, assessment=payload)

    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))
    html = detail.content.decode()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html)
            executed = page.evaluate("window.__stanstock_xss__ === true")
            script_count = page.evaluate(
                "document.querySelectorAll('.under10-panel script').length"
            )
        finally:
            browser.close()

    assert executed is False
    assert script_count == 0

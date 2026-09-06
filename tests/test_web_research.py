from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse
from django.utils import timezone

from stanstock.data.models import (
    Company,
    DataAsset,
    LatestMarketData,
    Listing,
    Region,
    Security,
    Universe,
    UniverseMembership,
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
from stanstock.simulation.models import (
    SimulationDefinition,
    SimulationHolding,
    SimulationRun,
    SimulationTrade,
)


@pytest.fixture
def authenticated_client(client):
    user_model = get_user_model()
    user = user_model.objects.create_user(username="owner", password="correct-password")
    client.force_login(user)
    return client


@pytest.fixture
def persisted_analysis() -> StockAnalysis:
    company = Company.objects.create(
        name="Synthetic Alpha",
        country="US",
        sector="Technology",
    )
    security = Security.objects.create(company=company, name="Synthetic Alpha Common")
    listing = Listing.objects.create(
        security=security,
        ticker="SYN-A",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    universe = Universe.objects.create(
        slug="synthetic",
        name="Synthetic universe",
        config_version="demo-v1",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=timezone.localdate(),
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="a" * 64,
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    now = timezone.now()
    source_asset = DataAsset.objects.create(
        provider="synthetic_demo",
        kind="price_history",
        subject="SYN-A",
        relative_path="tests/syn-a.parquet",
        sha256="c" * 64,
        retrieved_at=now,
        available_at=now,
    )
    LatestMarketData.objects.create(
        listing=listing,
        observed_at=now,
        session_date=timezone.localdate(),
        close=Decimal("101.25"),
        previous_close=Decimal("100.00"),
        volume=1_000_000,
        source_asset=source_asset,
    )
    run = AnalysisRun.objects.create(
        generated_at=now,
        data_cutoff=now,
        target_date=timezone.localdate(),
        universe_snapshot=snapshot,
        config_version="rules-v1",
        config_hash="b" * 64,
        code_revision="test-revision",
    )
    analysis = StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("101.25"),
        daily_change=Decimal("0.012"),
        overall_score=Decimal("78.50"),
        recommendation=Recommendation.BUY,
        risk_score=Decimal("31.00"),
        risk_class=RiskClass.MEDIUM,
        confidence=Decimal("64.00"),
        short_scenario={"bear": -0.04, "base": 0.03, "bull": 0.09},
        medium_scenario={"bear": -0.16, "base": 0.12, "bull": 0.31},
        long_scenario={"bear": -0.25, "base": 0.34, "bull": 0.82},
        component_scores={"quality": 82, "momentum": 74},
        reasons=["Quality is above the configured threshold."],
        risks=["Volatility remains material."],
        data_quality={
            "source_assets": [
                {
                    "provider": "synthetic_demo",
                    "kind": "price_history",
                    "subject": "SYN-A",
                }
            ]
        },
    )
    Prediction.objects.create(
        analysis=analysis,
        listing=listing,
        generated_at=now,
        target_date=timezone.localdate(),
        horizon=Prediction.Horizon.SHORT,
        price_at_prediction=Decimal("101.25"),
        bear_return=Decimal("-0.04"),
        base_return=Decimal("0.03"),
        bull_return=Decimal("0.09"),
        probability_positive=None,
        confidence=Decimal("64"),
        confidence_status="heuristic",
        insufficiency_reason="Insufficient comparable observations",
        recommendation=Recommendation.BUY,
        overall_score=Decimal("78.5"),
        model_version="baseline-v1",
        config_hash="b" * 64,
        data_cutoff=now,
        code_revision="test-revision",
    )
    return analysis


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
def test_opportunities_filter_and_stock_detail_render_persisted_analysis(
    authenticated_client,
    persisted_analysis: StockAnalysis,
) -> None:
    opportunities = authenticated_client.get(
        reverse("opportunities"),
        {
            "region": "us",
            "recommendation": "buy",
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
    assert "great-opportunity-v1" in detail.content.decode()


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
    persisted_analysis.save(update_fields=["data_quality"])

    opportunities = authenticated_client.get(reverse("opportunities"))
    detail = authenticated_client.get(reverse("stock-detail", args=[persisted_analysis.listing_id]))

    assert opportunities.status_code == 200
    assert "US price-only baseline." in opportunities.content.decode()
    assert "Medium- and long-horizon scenarios are withheld" in opportunities.content.decode()
    assert detail.status_code == 200
    detail_content = detail.content.decode()
    assert "This recommendation does not use company fundamentals." in detail_content
    assert "Split-adjusted price return; dividends excluded." in detail_content


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

    assert performance.status_code == 200
    performance_content = performance.content.decode()
    assert "Insufficient sample" in performance_content
    assert "Withheld" in performance_content


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

    response = authenticated_client.get(reverse("performance"))

    assert response.status_code == 200
    assert response.context["summary"]["sample_count"] == 1
    assert response.context["summary"]["research_matured_count"] == 1


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

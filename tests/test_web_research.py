from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest
from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse
from django.utils import timezone

from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.etfs import sync_investable_spy_from_asset
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


@pytest.fixture(autouse=True)
def scheduler_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "stanstock.web.views.launch_agent_status",
        lambda: {
            "installed": True,
            "loaded": True,
            "timezone_matches": True,
            "expected_timezone": "America/New_York",
        },
    )


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
    assert "Legacy 6-12 month scenario" in content
    assert "Legacy 3+ year scenario" in content
    assert "Reconstructed training evidence." not in content


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
def test_under_10_band_blocks_promotion_and_discloses_missing_long_horizon_gates(
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
    assert opportunities.status_code == 200
    assert "Under $10 - speculative watchlist" in opportunity_content
    assert "0% new allocation" in opportunity_content
    assert "9.99 USD" in opportunity_content
    price_band_groups = opportunities.context["price_band_groups"]
    assert price_band_groups[0]["cards"][0]["price_band"].price_date == date(
        2026,
        9,
        5,
    )
    assert "Strong short-term setup" not in opportunity_content
    assert persisted_analysis.listing.ticker not in excluded_band.content.decode()

    detail_content = detail.content.decode()
    assert detail.status_code == 200
    assert "Forecast unavailable" in detail_content
    assert "Point-in-time SEC facts with adverse-versus-missing states" in detail_content
    assert "Verified split and reverse-split events" in detail_content
    status_content = status.content.decode()
    assert status.status_code == 200
    assert "Forecast unavailable" in status_content
    assert "-12.0% / +45.0% / +92.0%" not in status_content
    assert "-20.0% / +80.0% / +160.0%" not in status_content
    persisted_analysis.refresh_from_db()
    assert persisted_analysis.overall_score == Decimal("85")
    assert persisted_analysis.recommendation == Recommendation.BUY


@pytest.mark.django_db
def test_missing_current_usd_price_band_fails_closed(
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
    content = opportunities.content.decode()
    assert "Strong short-term setup" not in content
    assert "Forecast unavailable" in content
    assert "No valid latest persisted USD close is available" in content

    assert detail.status_code == 200
    detail_content = detail.content.decode()
    assert "Forecast unavailable" in detail_content
    assert "No valid latest persisted USD close is available" in detail_content


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

    assert performance.status_code == 200
    performance_content = performance.content.decode()
    assert "Insufficient sample" in performance_content
    assert "Withheld" in performance_content


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

    assert decisions.status_code == 200
    assert decisions.context["result_count"] == 1
    assert decisions.context["prediction_cards"][0]["prediction"].evidence_role == "decision"
    assert incompatible.status_code == 200
    assert incompatible.context["result_count"] == 0


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
    advisory_groups = response.context["advisory_groups"]
    assert len(advisory_groups) == 1
    assert advisory_groups[0]["sample_count"] == 1
    assert advisory_groups[0]["direction_accuracy"] is None
    assert advisory_groups[0]["direction_sample_count"] == 1
    assert "Advisory evidence" in response.content.decode()


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
        generated_at = datetime(2026, 9, 8 + method_index, 1, tzinfo=UTC)
        run = AnalysisRun.objects.create(
            generated_at=generated_at,
            data_cutoff=generated_at,
            target_date=date(2026, 9, 7 + method_index),
            issued_on_time=True,
            universe_snapshot=snapshot,
            config_version=version,
            config_hash=digest,
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
        for prediction_index in range(15):
            prediction = Prediction.objects.create(
                analysis=analysis,
                listing=listing,
                generated_at=generated_at,
                target_date=run.target_date,
                issued_on_time=True,
                horizon=Prediction.Horizon.SHORT,
                evidence_grade=UniverseSnapshot.Grade.OBSERVED,
                source_mode=Prediction.SourceMode.PROVIDER,
                price_provider="twelve_data",
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
                model_version=f"method-{method_index}-{prediction_index}",
                method_version=version,
                config_hash=digest,
                data_cutoff=generated_at,
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
                resolution="Observed method cohort",
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

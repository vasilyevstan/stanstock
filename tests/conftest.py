"""Shared pytest fixtures for the web/Under-$10 test suite.

Kept in `conftest.py` (rather than defined once and imported elsewhere) so
multiple test modules can request the same fixture by name without
triggering a false "redefinition" lint warning on the request-time
parameter that shares the fixture's name -- the standard pytest idiom for
sharing fixtures across files in the same directory.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone

from stanstock.core.launchd import SCHEDULE_TIME_LABEL
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
from stanstock.data.sec_config import load_sec_fundamentals_config
from stanstock.research.models import (
    AnalysisRun,
    Prediction,
    Recommendation,
    RiskClass,
    StockAnalysis,
)


@pytest.fixture(scope="module")
def sec_config():
    return load_sec_fundamentals_config()


@pytest.fixture
def authenticated_client(client):
    user_model = get_user_model()
    user = user_model.objects.create_user(username="owner", password="synthetic-test-only")
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
            "installed_schedule_label": f"{SCHEDULE_TIME_LABEL} (Tuesday-Saturday)",
            "expected_schedule_label": SCHEDULE_TIME_LABEL,
            "schedule_matches": True,
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

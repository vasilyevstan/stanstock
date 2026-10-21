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


@pytest.fixture
def c2_environment(tmp_path, monkeypatch, django_user_model, settings, request):
    """Only synthetic native-format bytes, disposable DB/store, and denied network."""
    import json
    from datetime import UTC, date, datetime, timedelta
    from types import SimpleNamespace
    from unittest.mock import Mock

    from exchange_calendars import get_calendar

    from stanstock.data.assets import AssetStore, asset_ref_for
    from stanstock.data.live_us import (
        _ensure_my_list_listing,
        _persist_catalog,
        _persist_price_series,
        load_us_universe_config,
    )
    from stanstock.data.management.config_loader import default_us_universe_config_path
    from stanstock.data.models import ProviderRecord
    from stanstock.data.provider_policy import PRIVATE_USAGE_SCOPE
    from stanstock.data.providers import twelve_data
    from stanstock.data.providers.contracts import StockCatalog
    from stanstock.research.decision_support_eligibility import (
        PRODUCT_EFFECTIVE_CONFIG_HASH,
        PRODUCT_VERSION,
    )
    from stanstock.research.forward_scorecard import declare_forward_protocol

    target = date(2026, 9, 11)
    now = datetime(2026, 9, 11, 21, tzinfo=UTC)
    clock = {"now": now}
    monkeypatch.setattr(timezone, "now", lambda: clock["now"])
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.RESEARCH_PRODUCT_VERSION = PRODUCT_VERSION
    settings.DEMO_MODE = False
    store = AssetStore(tmp_path / "c2-synthetic")
    settings.DATA_DIR = store.root
    owner = django_user_model.objects.create_user(username="c2-synthetic-owner", date_joined=now)
    record = ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=True,
        terms_url="https://twelvedata.com/terms",
        usage_scope=PRIVATE_USAGE_SCOPE,
        metadata={
            "internal_display_rights_confirmed": True,
            "plan": "grow",
            "daily_credit_limit": 800,
            "credits_per_minute": 8,
        },
    )
    path = default_us_universe_config_path()
    universe_config = load_us_universe_config(path)
    catalogs = []
    references = {}
    for exchange in universe_config.exchanges:
        symbols = universe_config.symbols if exchange == "NASDAQ" else ("VENUE",)
        rows = [
            {
                "symbol": symbol,
                "name": f"Synthetic C2 {symbol}",
                "currency": "USD",
                "exchange": exchange,
                "mic_code": "XNAS" if exchange == "NASDAQ" else "XNYS",
                "country": "United States",
                "type": "Common Stock" if symbol in ("AAPL", "NVDA") else "ETF",
            }
            for symbol in symbols
        ]
        raw = json.dumps({"status": "ok", "count": len(rows), "data": rows}).encode()
        parsed, count = twelve_data.parse_stock_catalog_references(
            raw,
            exchange=exchange,
            require_complete=True,
        )
        catalogs.append(
            _persist_catalog(
                store,
                StockCatalog(
                    provider="twelve_data",
                    exchange=exchange,
                    references=parsed,
                    count=count,
                    retrieved_at=now - timedelta(minutes=30),
                    source_url="https://synthetic.invalid/stocks",
                    raw_bytes=raw,
                ),
            )
        )
        references.update((ref.symbol, ref) for ref in parsed)
    calendar = get_calendar("XNYS")
    sessions = tuple(
        day.date() for day in calendar.sessions_in_range(target - timedelta(days=1300), target)
    )[-757:]
    assert len(sessions) == 757 and date(2024, 6, 3) in sessions
    options = {"mismatch": False, "ignored_none": False, "flat": False, "missing_volume": False}
    options.update(getattr(request, "param", {}))

    def series(symbol, *, mode="splits", start=None, retrieved_at=None, through=None):
        through = through or target
        observed_sessions = (
            sessions
            + tuple(
                day.date()
                for day in calendar.sessions_in_range(target + timedelta(days=1), through)
            )
            if through > target
            else sessions
        )
        values = []
        for index, session in enumerate(observed_sessions):
            if start is not None and session < start:
                continue
            close = (
                Decimal("100")
                if options["flat"] and symbol != "SPY"
                else Decimal("100")
                + Decimal(index) / (32 if symbol == "SPY" else 16)
                + Decimal(index % 8) / 64
            )
            if symbol != "SPY" and options.get("down"):
                close = Decimal("200") - Decimal(index) / 16 + Decimal(index % 8) / 64
            if symbol != "SPY" and options.get("tiny"):
                close = Decimal("100") + Decimal(index % 8) / Decimal(10**8)
            if symbol != "SPY":
                close *= Decimal(str(options.get("price_scale", 1)))
            volume = 1_000_000 * options.get("volume_scale", 1)
            if symbol == "NVDA" and date(2024, 6, 3) <= session <= date(2024, 6, 7):
                if mode == "none" and not options["ignored_none"]:
                    close *= 10
                else:
                    volume *= 10
            if mode == "none" and symbol == "AAPL" and session == target:
                if options["mismatch"]:
                    close += Decimal("0.00000000000000001")
                if options["missing_volume"]:
                    volume = None
            values.append(
                {
                    "datetime": session.isoformat(),
                    "open": str(close),
                    "high": str(close + 1),
                    "low": str(close - 1),
                    "close": str(close),
                    "volume": None if volume is None else str(volume),
                }
            )
        raw = json.dumps(
            {
                "status": "ok",
                "meta": {
                    "symbol": symbol,
                    "interval": "1day",
                    "currency": "USD",
                    "exchange": "NYSE Arca" if symbol == "SPY" else "NASDAQ",
                    "mic_code": "ARCX" if symbol == "SPY" else "XNAS",
                    "type": "ETF" if symbol == "SPY" else "Common Stock",
                    "exchange_timezone": "America/New_York",
                },
                "values": values,
            }
        ).encode()
        return twelve_data.parse_daily_price_series(
            raw,
            symbol=symbol,
            retrieved_at=retrieved_at or clock["now"],
            source_url="https://synthetic.invalid/time_series",
            start_date=start,
            end_date=through,
            adjustment=mode,
        )

    listings = {
        symbol: _ensure_my_list_listing(
            symbol=symbol, reference=references[symbol], target_date=target
        )
        for symbol in ("AAPL", "NVDA")
    }
    benchmark = _persist_price_series(
        store=store,
        series=series("SPY", retrieved_at=now - timedelta(minutes=10)),
        listing=None,
        resolved_mic_code="ARCX",
        catalog_assets=catalogs,
    )

    def fetch(symbol, **kwargs):
        assert kwargs["end_date"] == target
        assert kwargs["adjustment"] in ("none", "splits")
        if options.get("late_none") and symbol == "AAPL" and kwargs["adjustment"] == "none":
            clock["now"] = now + timedelta(minutes=1)
        return series(symbol, mode=kwargs["adjustment"], start=kwargs.get("start_date"))

    fetch_mock = Mock(side_effect=fetch)
    credentials = Mock(return_value="synthetic-c2-token-not-a-credential")
    monkeypatch.setattr(twelve_data, "fetch_daily_price_series", fetch_mock)
    monkeypatch.setattr(twelve_data, "resolve_api_key", credentials)
    monkeypatch.setattr(
        twelve_data, "fetch_stock_catalog", Mock(side_effect=AssertionError("network forbidden"))
    )
    monkeypatch.setattr(twelve_data, "fetch", Mock(side_effect=AssertionError("network forbidden")))
    clock["now"] = now - timedelta(days=1)
    declaration = declare_forward_protocol(user=owner, store=store, product_version=PRODUCT_VERSION)
    clock["now"] = now
    return SimpleNamespace(
        owner=owner,
        store=store,
        path=path,
        record=record,
        target=target,
        now=now,
        clock=clock,
        options=options,
        series=series,
        listings=listings,
        references=references,
        catalogs=tuple(catalogs),
        benchmark=benchmark,
        declaration=declaration,
        fetch=fetch_mock,
        credentials=credentials,
        version=PRODUCT_VERSION,
        config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
        asset_ref=asset_ref_for,
    )


@pytest.fixture
def c2_closed_batch(c2_environment):
    from datetime import timedelta

    from stanstock.core.models import JobRun
    from stanstock.data.liquidity_evidence import (
        acquire_liquidity_pairs,
        close_liquidity_batch,
        liquidity_job_identity,
    )
    from stanstock.data.research_product import (
        capture_authorized_product_intake,
        product_intake_payload,
    )
    from stanstock.data.research_product_jobs import product_job_name

    env = c2_environment
    intake = capture_authorized_product_intake(
        target_date=env.target,
        evidence_grade=env.options.get("grade", "research"),
        issuance_key="c2-fixture",
        owner=env.owner,
        core_config_path=env.path,
        policy_identity=env.config_hash,
        captured_at=env.now - timedelta(minutes=1),
        store=env.store,
        product_version=env.version,
    )
    job = JobRun.objects.create(
        job_name=product_job_name(
            "research_intake_v2",
            liquidity_job_identity(product_intake_payload(intake, store=env.store)),
        ),
        region="us",
        target_date=env.target,
        details={
            "intake_asset": env.asset_ref(intake.asset).to_json(),
            "forward_declaration": env.asset_ref(env.declaration.asset).to_json(),
        },
    )
    selected = {}
    entries = acquire_liquidity_pairs(
        intake=intake,
        job=job,
        listings=env.listings,
        references=env.references,
        split_assets=selected,
        request_key=lambda: "synthetic-c2-token",
        preflight=lambda credits: None,
        history_start=env.target - timedelta(days=366 * 7),
        target_date=env.target,
        store=env.store,
    )
    admissions = {
        symbol: {
            "status": "admitted" if symbol in selected else "identity_rejected",
            "reason": "" if symbol in selected else "catalog_listing_or_entitlement_not_supported",
            "listing_id": str(env.listings[symbol].id) if symbol in env.listings else None,
            "price_asset": env.asset_ref(selected[symbol]).to_json()
            if symbol in selected
            else None,
            "bootstrap_attempted": symbol in selected,
        }
        for symbol in intake.requested_symbols
    }
    job.details["liquidity_admissions"] = admissions
    job.save(update_fields=["details"])
    batch = close_liquidity_batch(
        intake=intake,
        job=job,
        entries=entries,
        catalog_assets=env.catalogs,
        benchmark_asset=env.benchmark,
        cutoff=env.now,
        store=env.store,
    )
    env.intake, env.job, env.batch, env.admissions, env.selected = (
        intake,
        job,
        batch,
        admissions,
        selected,
    )
    return env


@pytest.fixture
def c2_snapshot(c2_closed_batch):
    from stanstock.data.research_product import materialize_product_membership

    env = c2_closed_batch
    env.snapshot = materialize_product_membership(
        intake=env.intake,
        qualified_listing_ids=[row.id for row in env.listings.values()],
        candidate_states={row.id: "admitted" for row in env.listings.values()},
        admissions=env.admissions,
        catalog_assets=env.catalogs,
        benchmark_asset=env.benchmark,
        captured_at=env.batch.available_at,
        store=env.store,
        liquidity_batch=env.batch,
        forward_declaration=env.declaration.asset,
    )
    return env


@pytest.fixture
def c2_persisted(c2_snapshot):
    from stanstock.research.decision_support_eligibility import default_decision_support_config_path
    from stanstock.research.service import analyze_snapshot

    env = c2_snapshot
    env.results = analyze_snapshot(
        universe_snapshot=env.snapshot,
        decision_time=env.now,
        target_date=env.target,
        provider="twelve_data",
        benchmark_subject="SPY",
        config_path=default_decision_support_config_path(),
        store=env.store,
    )
    env.run = env.results[0].run
    return env

"""Targeted unit tests for `stanstock.core.refresh_verification`.

`tests/test_scheduled_refresh.py` proves the command-level integration
(persisted records flowing through verification end to end, including
retry/no-op zero-fetch and SEC-required paths). This module calls
`verify_scheduled_refresh` directly against a real, minimal set of
persisted evidence (built once per test via `refresh_fixtures` + the real
job entry points) and then mutates one fact at a time to prove each
fail-closed check actually fires -- rather than re-running the whole
command for every failure mode.
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl
import pytest
from django.conf import settings

import refresh_fixtures
from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun
from stanstock.core.refresh_verification import RefreshVerificationError, verify_scheduled_refresh
from stanstock.data import sec_ingestion
from stanstock.data.assets import AssetStore, asset_ref_for, register_asset
from stanstock.data.jobs import PreparedUsDailyJob, execute_us_daily_job
from stanstock.data.live_us import UsUniverseConfig
from stanstock.data.live_us import sync_investable_spy_from_asset as _real_sync_investable_spy
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
from stanstock.data.providers import sec, twelve_data
from stanstock.portfolio.jobs import execute_portfolio_snapshot_job
from stanstock.portfolio.models import Portfolio, PortfolioSnapshot, PortfolioSnapshotHolding
from stanstock.portfolio.service import (
    _corporate_action_suspected,
    calculate_portfolio_valuation,
    compute_snapshot_input_hash,
    upsert_holding,
)
from stanstock.research.jobs import execute_prediction_evaluation_job
from stanstock.research.long_forecast_config import (
    load_long_forecast_config,
    long_forecast_config_hash,
)
from stanstock.research.models import (
    AnalysisRun,
    Prediction,
    PredictionOutcome,
    Recommendation,
    RiskClass,
    StockAnalysis,
)
from stanstock.research.outcomes import evaluate_prediction

pytestmark = pytest.mark.django_db

TARGET_DATE = date(2026, 9, 4)
DECISION_TIME = datetime(2026, 9, 5, 6, tzinfo=UTC)
CODE_REVISION = "b" * 40


def _stage_entry(run: JobRun) -> dict[str, object]:
    return {
        "job_run_id": str(run.pk),
        "status": run.status,
        "attempt": run.attempt,
        "error": "",
    }


def _build_verified_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    symbols: tuple[str, ...] = ("AAA",),
    sec: bool = False,
    before_evaluation: Any = None,
) -> tuple[dict[str, dict[str, object]], UsUniverseConfig]:
    """Run the real market/evaluation/portfolio (and optional SEC) stages.

    Returns the `stages` mapping `verify_scheduled_refresh` expects,
    already satisfied end to end, so each test can mutate exactly one fact
    (a stage entry, a persisted row, an asset checksum, ...) before calling
    `verify_scheduled_refresh` and asserting the specific `reason_code` it
    raises.
    """
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", CODE_REVISION)
    config = refresh_fixtures.build_universe_config(symbols=symbols, minimum_eligible=1)
    refresh_fixtures.enable_twelve_data_provider()
    refresh_fixtures.set_twelve_data_api_key(monkeypatch)
    refresh_fixtures.patch_twelve_data_provider(
        monkeypatch,
        config,
        target_date=TARGET_DATE,
        retrieved_at=DECISION_TIME,
    )
    monkeypatch.setattr("stanstock.data.live_us.timezone.now", lambda: DECISION_TIME)

    stages: dict[str, dict[str, object]] = {}
    if sec:
        listing = refresh_fixtures.pre_create_stock_listing(symbols[0])
        sec_evidence = refresh_fixtures.build_sec_evidence(
            store=AssetStore(tmp_path),
            company=listing.security.company,
            available_before=DECISION_TIME,
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
        )
        sec_run = execute_target_job(
            job_name="sec_fundamentals",
            region="us",
            target_date=TARGET_DATE,
            task=lambda run: JobExecutionResult(
                details={
                    "mapping_asset_id": sec_evidence.mapping_asset_id,
                    "mapping_sha256": sec_evidence.mapping_sha256,
                    "cik_config_version": sec_evidence.cik_config_version,
                    "cik_config_hash": sec_evidence.cik_config_hash,
                    "config_version": sec_evidence.fundamentals_config_version,
                    "config_hash": sec_evidence.fundamentals_config_hash,
                    "asset_refs": [ref.to_json() for ref in sec_evidence.asset_refs],
                }
            ),
        )
        stages["sec_fundamentals"] = _stage_entry(sec_run)

    prepared = PreparedUsDailyJob(
        config=config,
        target_date=TARGET_DATE,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        decision_time=DECISION_TIME,
    )
    market_run = execute_us_daily_job(prepared, require_observed=True)
    stages["market"] = _stage_entry(market_run)

    if before_evaluation is not None:
        before_evaluation()

    evaluation_run = execute_prediction_evaluation_job(
        provider=twelve_data.PROVIDER,
        evaluation_date=TARGET_DATE,
        evaluation_time=DECISION_TIME,
        benchmark_subject=config.benchmark_symbol,
    )
    stages["evaluation"] = _stage_entry(evaluation_run)

    portfolio_run = execute_portfolio_snapshot_job(
        target_date=TARGET_DATE,
        require_session_date=True,
        require_all=True,
    )
    stages["portfolio_snapshots"] = _stage_entry(portfolio_run)

    return stages, config


def _create_lagging_prediction(
    *,
    target_date: date,
    price_provider: str = twelve_data.PROVIDER,
    generated_at: datetime | None = None,
) -> Prediction:
    """A standalone, already-matured `Prediction` unrelated to the main run.

    Built directly via the ORM (the same pattern `test_research_outcomes.py`
    uses) rather than through the real pipeline, so it belongs to its own
    `AnalysisRun`/`StockAnalysis` and is invisible to the main run-bound
    prediction checks -- it exists purely to prove the evaluation stage's
    independently re-derived candidate set picks up a genuinely mature,
    still-pending prediction from an earlier run, not just today's.

    `generated_at` defaults to midnight of `target_date` (always strictly
    before a same-or-later-day evaluation execution); pass an explicit
    value to test the `as_of=evaluation_time` pre-child boundary itself.
    """
    company = Company.objects.create(name=f"Lagging Co {uuid4().hex[:6]}", country="US")
    security = Security.objects.create(company=company)
    listing = Listing.objects.create(
        security=security,
        ticker=f"L{uuid4().hex[:6]}",
        provider_symbol=f"L{uuid4().hex[:6]}",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    universe = Universe.objects.create(
        slug=f"lagging-{uuid4().hex[:8]}", name="Lagging", config_version="1"
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=target_date,
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="a" * 64,
    )
    generated_at = generated_at or datetime.combine(target_date, datetime.min.time(), tzinfo=UTC)
    run = AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=generated_at,
        target_date=target_date,
        universe_snapshot=snapshot,
        config_version="lagging-v1",
        config_hash="b" * 64,
        code_revision="test",
    )
    analysis = StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("100"),
        overall_score=Decimal("70"),
        recommendation=Recommendation.HOLD,
        risk_score=Decimal("30"),
        risk_class=RiskClass.LOW,
        confidence=Decimal("60"),
    )
    return Prediction.objects.create(
        analysis=analysis,
        listing=listing,
        generated_at=generated_at,
        target_date=target_date,
        horizon=Prediction.Horizon.SHORT,
        evidence_role=Prediction.EvidenceRole.DECISION,
        data_cutoff=generated_at,
        price_provider=price_provider,
        price_subject=listing.provider_symbol,
        price_at_prediction=Decimal("100"),
        bear_return=Decimal("-0.10"),
        base_return=Decimal("0.02"),
        bull_return=Decimal("0.10"),
        probability_positive=None,
        confidence=Decimal("60"),
        confidence_status="heuristic",
        insufficiency_reason="",
        recommendation=Recommendation.HOLD,
        overall_score=Decimal("70"),
        component_scores={},
        model_version="lagging-short",
        config_hash="b" * 64,
        method_version="lagging-v1",
        code_revision="test",
        issued_on_time=True,
        evidence_grade=UniverseSnapshot.Grade.OBSERVED,
        source_mode="provider",
        source_assets=[{"provider": price_provider}],
    )


def _business_dates_after(start: date, count: int) -> list[date]:
    dates: list[date] = []
    current = start + timedelta(days=1)
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def _register_lagging_price_history(
    *,
    tmp_path: Path,
    subject: str,
    baseline_date: date,
    baseline_close: Decimal,
    available_at: datetime,
    session_count: int = 10,
) -> None:
    """Register real `price_history` evidence for a standalone lagging
    listing so its own evaluation can genuinely mature, rather than
    relying on a pre-seeded terminal outcome.
    """
    sessions = _business_dates_after(baseline_date, session_count)
    frame = pl.DataFrame(
        {
            "date": [baseline_date, *sessions],
            "close": [float(baseline_close), *[101.0 + index for index in range(session_count)]],
            "volume": [1_000_000] * (session_count + 1),
        }
    )
    store = AssetStore(tmp_path)
    stored = store.write_frame(f"lagging/{uuid4().hex}.parquet", frame)
    register_asset(
        provider=twelve_data.PROVIDER,
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )


def test_happy_path_verifies_and_is_path_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )

    assert result["status"] == "verified"
    assert result["eligible_count"] == 1
    assert result["sec"] == {"required": False}
    assert result["portfolio"]["active_portfolios"] == 0
    assert str(tmp_path) not in repr(result)


def test_wrong_child_target_date_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    # Simulate a child job run recorded against a different target date --
    # the fresh re-fetch must reject it rather than trust the cached label.
    wrong_target = execute_target_job(
        job_name="daily",
        region="us",
        target_date=date(2099, 1, 1),
        task=lambda run: JobExecutionResult(),
    )
    stages["market"]["job_run_id"] = str(wrong_target.pk)
    stages["market"]["status"] = wrong_target.status
    stages["market"]["attempt"] = wrong_target.attempt

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "stage_identity_mismatch"
    assert str(tmp_path) not in str(excinfo.value)


def test_missing_job_run_id_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    del stages["market"]["job_run_id"]

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "stage_job_run_id_missing"


def test_malformed_analysis_run_identity_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    market_run = JobRun.objects.get(pk=stages["market"]["job_run_id"])
    details = dict(market_run.details)
    details["analysis_run_id"] = str(uuid4())
    JobRun.objects.filter(pk=market_run.pk).update(details=details)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "analysis_run_missing"


def test_stale_latest_market_data_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    stale = LatestMarketData.objects.first()
    assert stale is not None
    LatestMarketData.objects.filter(pk=stale.pk).update(session_date=date(2020, 1, 1))

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "latest_market_data_stale_or_future"


def test_missing_latest_market_data_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    LatestMarketData.objects.all().delete()

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "latest_market_data_missing"


def test_corrupt_registered_asset_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    latest = LatestMarketData.objects.select_related("source_asset").first()
    assert latest is not None
    asset = latest.source_asset
    store = AssetStore(tmp_path)
    store.resolve(asset.relative_path).write_bytes(b"corrupted")

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "latest_market_data_asset_unreadable"
    assert str(tmp_path) not in str(excinfo.value)


# --- R5 F6: the upstream raw provider payload a normalized price asset
# derives from must itself be present, identified, and cutoff-safe --------


def test_corrupt_bound_raw_price_asset_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Corrupting the raw JSON payload a bound normalized price asset
    declares (`metadata["raw_asset_id"]`) fails closed even though the
    normalized parquet itself is untouched."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    latest = LatestMarketData.objects.select_related("source_asset").first()
    assert latest is not None
    raw_asset_id = uuid.UUID(str(latest.source_asset.metadata["raw_asset_id"]))
    raw_asset = DataAsset.objects.get(pk=raw_asset_id)
    store = AssetStore(tmp_path)
    store.resolve(raw_asset.relative_path).write_bytes(b"corrupted raw payload")

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "asset_integrity_failed"
    assert str(tmp_path) not in str(excinfo.value)


def test_unrelated_historical_raw_price_asset_corruption_does_not_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real but unrelated, corrupt historical `raw_price_history` asset
    (never declared by this target's own bound price asset) must not block
    verification -- the scoped manifest never touches it."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    latest = LatestMarketData.objects.select_related("source_asset").first()
    assert latest is not None
    store = AssetStore(tmp_path)
    written = store.write_bytes("raw/twelve_data/time_series/unrelated.json", b"unrelated raw")
    unrelated = DataAsset.objects.create(
        provider=twelve_data.PROVIDER,
        kind="raw_price_history",
        subject="UNRELATED",
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=latest.source_asset.retrieved_at - timedelta(days=400),
        available_at=latest.source_asset.retrieved_at - timedelta(days=400),
    )
    (tmp_path / unrelated.relative_path).write_bytes(b"now corrupted, but irrelevant")

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )
    assert result["status"] == "verified"


def test_sec_required_but_absent_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=True,
        )
    assert excinfo.value.reason_code == "stage_details_missing"


def test_sec_required_and_present_verifies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path, sec=True)

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=True,
    )
    assert result["sec"]["required"] is True
    assert "mapping_asset_id" in result["sec"]


# NOTE: SEC semantic fact/filing-evidence closure (`_require_sec_fact` in
# rev-1) is deliberately removed from this data-domain slice per
# `refresh-output-verification@rev-2` #6; it returns in the research-domain
# validator (Slice C). Until then, corrupting a `FundamentalFactEvidence`
# filing asset alone does not fail a data-stage-only verified refresh.


def test_portfolio_nonzero_requires_target_date_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    Portfolio.objects.create(
        owner=refresh_fixtures.create_portfolio_owner(),
        name="Test Portfolio",
        base_currency="USD",
    )
    # The real portfolio stage already ran (recorded as a zero-active skip);
    # a portfolio created afterwards must make verification fail closed
    # rather than silently accept the stale zero-active skip as evidence.
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "portfolio_stage_not_success"


def test_portfolio_nonzero_with_real_snapshot_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", CODE_REVISION)
    config = refresh_fixtures.build_universe_config(symbols=("AAA",), minimum_eligible=1)
    refresh_fixtures.enable_twelve_data_provider()
    refresh_fixtures.set_twelve_data_api_key(monkeypatch)
    refresh_fixtures.patch_twelve_data_provider(
        monkeypatch,
        config,
        target_date=TARGET_DATE,
        retrieved_at=DECISION_TIME,
    )
    monkeypatch.setattr("stanstock.data.live_us.timezone.now", lambda: DECISION_TIME)
    Portfolio.objects.create(
        owner=refresh_fixtures.create_portfolio_owner(),
        name="Test Portfolio",
        base_currency="USD",
    )

    prepared = PreparedUsDailyJob(
        config=config,
        target_date=TARGET_DATE,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        decision_time=DECISION_TIME,
    )
    market_run = execute_us_daily_job(prepared, require_observed=True)
    evaluation_run = execute_prediction_evaluation_job(
        provider=twelve_data.PROVIDER,
        evaluation_date=TARGET_DATE,
        evaluation_time=DECISION_TIME,
        benchmark_subject=config.benchmark_symbol,
    )
    portfolio_run = execute_portfolio_snapshot_job(
        target_date=TARGET_DATE,
        require_session_date=True,
        require_all=True,
    )
    stages = {
        "market": _stage_entry(market_run),
        "evaluation": _stage_entry(evaluation_run),
        "portfolio_snapshots": _stage_entry(portfolio_run),
    }

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )
    assert result["portfolio"]["active_portfolios"] == 1
    assert result["portfolio"]["snapshots_verified"] == 1


def test_no_op_retry_finds_same_evidence_with_zero_provider_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", CODE_REVISION)
    config = refresh_fixtures.build_universe_config(symbols=("AAA",), minimum_eligible=1)
    refresh_fixtures.enable_twelve_data_provider()
    refresh_fixtures.set_twelve_data_api_key(monkeypatch)
    catalog_calls, price_calls = refresh_fixtures.patch_twelve_data_provider(
        monkeypatch,
        config,
        target_date=TARGET_DATE,
        retrieved_at=DECISION_TIME,
    )
    monkeypatch.setattr("stanstock.data.live_us.timezone.now", lambda: DECISION_TIME)

    prepared = PreparedUsDailyJob(
        config=config,
        target_date=TARGET_DATE,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        decision_time=DECISION_TIME,
    )
    first_market = execute_us_daily_job(prepared, require_observed=True)
    evaluation_run = execute_prediction_evaluation_job(
        provider=twelve_data.PROVIDER,
        evaluation_date=TARGET_DATE,
        evaluation_time=DECISION_TIME,
        benchmark_subject=config.benchmark_symbol,
    )
    portfolio_run = execute_portfolio_snapshot_job(
        target_date=TARGET_DATE,
        require_session_date=True,
        require_all=True,
    )
    stages = {
        "market": _stage_entry(first_market),
        "evaluation": _stage_entry(evaluation_run),
        "portfolio_snapshots": _stage_entry(portfolio_run),
    }
    first_result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "SPY"]

    # Retry: `execute_us_daily_job` auto-skips-by-reference for an
    # already-succeeded target -- no provider boundary call is made -- and
    # verification must independently resolve the *same* underlying
    # analysis run and asset manifest, not merely accept the skip at face
    # value.
    second_market = execute_us_daily_job(prepared, require_observed=True)
    assert second_market.status == JobRun.Status.SKIPPED
    stages["market"] = _stage_entry(second_market)
    second_result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )

    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "SPY"]
    assert second_result["analysis_run_id"] == first_result["analysis_run_id"]
    assert second_result["asset_manifest"] == first_result["asset_manifest"]


def test_first_failure_then_repaired_local_evidence_recovers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    latest = LatestMarketData.objects.select_related("source_asset").first()
    assert latest is not None
    asset = latest.source_asset
    store = AssetStore(tmp_path)
    resolved_path = store.resolve(asset.relative_path)
    original_bytes = resolved_path.read_bytes()
    resolved_path.write_bytes(b"corrupted")

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "latest_market_data_asset_unreadable"

    # Repair the local output (no re-fetch, no re-run) and prove
    # verification recovers using the exact same child/output/asset
    # identity, never re-persisting a fresh, unrelated success shape.
    resolved_path.write_bytes(original_bytes)
    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )
    assert result["status"] == "verified"


def test_failure_summary_is_path_free(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    LatestMarketData.objects.all().delete()

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    failure_details = excinfo.value.to_failure_details()
    assert failure_details["status"] == "failed"
    assert failure_details["reason_code"] == "latest_market_data_missing"
    assert str(tmp_path) not in str(failure_details)


# --- F1: membership identity must resist a same-cardinality symbol swap ----


def test_membership_symbol_swap_same_cardinality_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path, symbols=("AAA", "BBB"))
    # Same cardinality (two symbols), different second symbol -- a naive
    # membership *count* check cannot distinguish this from the real
    # configuration.
    swapped_config = dataclasses.replace(config, symbols=("AAA", "CCC"))

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=swapped_config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "membership_symbol_missing"


# --- R5 F2: membership evidence must bind to an independent immutable
# physical artifact, not a self-authenticating mutable hash --------------


def test_membership_co_mutation_with_matching_hash_rewrite_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A same-cardinality membership tamper *plus* a matching config_hash
    rewrite cannot pass: the check now binds to an independent immutable
    evidence asset the attacker never touched."""
    from stanstock.data.live_us import universe_snapshot_evidence_payload
    from stanstock.data.refresh_validation import verify_membership_evidence

    stages, config = _build_verified_state(monkeypatch, tmp_path, symbols=("AAA", "BBB"))
    snapshot = UniverseSnapshot.objects.get(
        pk=JobRun.objects.get(pk=stages["market"]["job_run_id"]).details["snapshot_id"]
    )
    membership = UniverseMembership.objects.select_related("listing").get(
        snapshot=snapshot, listing__provider_symbol="BBB"
    )
    UniverseMembership.objects.filter(pk=membership.pk).update(
        eligible=False, exclusion_reason="tampered"
    )

    catalog_assets = list(
        DataAsset.objects.filter(
            pk__in=[
                uuid.UUID(value)
                for value in JobRun.objects.get(pk=stages["market"]["job_run_id"]).details[
                    "catalog_asset_ids"
                ]
            ]
        )
    )
    tampered_payload = universe_snapshot_evidence_payload(
        config=config,
        catalog_assets=catalog_assets,
        listings={
            m.listing.provider_symbol: m.listing
            for m in UniverseMembership.objects.filter(snapshot=snapshot).select_related("listing")
        },
        exclusion_reasons={"BBB": "tampered"},
    )
    from stanstock.data.management.config_loader import config_hash as recompute_hash

    UniverseSnapshot.objects.filter(pk=snapshot.pk).update(
        config_hash=recompute_hash(tampered_payload)
    )
    snapshot.refresh_from_db()

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_membership_evidence(
            snapshot,
            list(
                UniverseMembership.objects.filter(snapshot=snapshot).select_related(
                    "listing__security__company"
                )
            ),
            universe_config=config,
            catalog_assets=catalog_assets,
            cutoff=DECISION_TIME,
        )
    assert excinfo.value.reason_code == "membership_evidence_hash_mismatch"


def test_membership_evidence_asset_corrupted_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from stanstock.data.refresh_evidence import lookup_membership_evidence

    stages, config = _build_verified_state(monkeypatch, tmp_path)
    snapshot = UniverseSnapshot.objects.get(
        pk=JobRun.objects.get(pk=stages["market"]["job_run_id"]).details["snapshot_id"]
    )
    lookup = lookup_membership_evidence(snapshot)
    assert lookup.asset is not None
    evidence_asset = lookup.asset
    (tmp_path / evidence_asset.relative_path).write_bytes(b"corrupted")

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "membership_evidence_asset_corrupt"
    assert str(tmp_path) not in str(excinfo.value)


def test_unrelated_historical_membership_evidence_corruption_does_not_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from stanstock.data.live_us import UNIVERSE_MEMBERSHIP_EVIDENCE_KIND

    stages, config = _build_verified_state(monkeypatch, tmp_path)
    store = AssetStore(tmp_path)
    written = store.write_bytes(
        "universe/unrelated/membership-evidence.json", b'{"unrelated": true}'
    )
    unrelated = DataAsset.objects.create(
        provider="stanstock",
        kind=UNIVERSE_MEMBERSHIP_EVIDENCE_KIND,
        subject=str(uuid4()),
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=DECISION_TIME - timedelta(days=400),
        available_at=DECISION_TIME - timedelta(days=400),
    )
    (tmp_path / unrelated.relative_path).write_bytes(b"now corrupted, but irrelevant")

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )
    assert result["status"] == "verified"


# --- F2: LatestMarketData must bind to the exact evidence asset ------------


def test_substituted_market_data_asset_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    latest = LatestMarketData.objects.select_related("source_asset").first()
    assert latest is not None
    original = latest.source_asset
    store = AssetStore(tmp_path)
    written = store.write_bytes("substitute/decoy.bin", b"a real but unrelated asset")
    substitute = DataAsset.objects.create(
        provider=original.provider,
        kind=original.kind,
        subject=original.subject,
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=original.retrieved_at,
        available_at=original.available_at,
    )
    LatestMarketData.objects.filter(pk=latest.pk).update(source_asset=substitute)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "latest_market_data_unbound"


def test_stale_cutoff_admits_no_future_evidence_for_market_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    latest = LatestMarketData.objects.select_related("source_asset").first()
    assert latest is not None
    run = AnalysisRun.objects.get(target_date=TARGET_DATE)
    # `DataAsset` rows are immutable at the database level, so a "future
    # vintage" cannot be simulated by editing the asset's own timestamps in
    # place; moving the *run's* own cutoff earlier than real, unmodified
    # evidence exercises the exact same `available_at/retrieved_at <=
    # data_cutoff` comparison this check performs.
    earlier_cutoff = latest.source_asset.available_at - timedelta(days=1)
    AnalysisRun.objects.filter(pk=run.pk).update(data_cutoff=earlier_cutoff)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    # Catalog assets are now also cutoff-bound (F-7): since this fixture's
    # catalog and price evidence share the same retrieval instant, moving
    # the cutoff globally earlier trips the catalog check first. Both are
    # the same underlying invariant -- no evidence admitted after this
    # run's own cutoff can be used -- so this still proves it end to end.
    assert excinfo.value.reason_code == "catalog_asset_after_cutoff"


def test_substituted_spy_benchmark_asset_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    spy_listing = Listing.objects.get(ticker="SPY")
    spy_latest = LatestMarketData.objects.select_related("source_asset").get(listing=spy_listing)
    original = spy_latest.source_asset
    store = AssetStore(tmp_path)
    written = store.write_bytes("substitute/decoy-spy.bin", b"a real but unrelated spy asset")
    substitute = DataAsset.objects.create(
        provider=original.provider,
        kind=original.kind,
        subject=original.subject,
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=original.retrieved_at,
        available_at=original.available_at,
    )
    LatestMarketData.objects.filter(pk=spy_latest.pk).update(source_asset=substitute)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "latest_market_data_unbound"


# --- F3: SEC mapping asset must carry the exact production identity -------


def test_sec_mapping_wrong_provider_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path, sec=True)
    sec_run = JobRun.objects.get(pk=stages["sec_fundamentals"]["job_run_id"])
    # `DataAsset` rows are immutable -- simulate the child pointing at a
    # real, checksum-valid asset of the *wrong provider* (a Twelve Data
    # price asset) rather than mutating the genuine SEC mapping asset.
    wrong_provider_asset = LatestMarketData.objects.select_related("source_asset").first()
    assert wrong_provider_asset is not None
    details = dict(sec_run.details)
    details["mapping_asset_id"] = str(wrong_provider_asset.source_asset_id)
    details["mapping_sha256"] = wrong_provider_asset.source_asset.sha256
    JobRun.objects.filter(pk=sec_run.pk).update(details=details)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=True,
        )
    assert excinfo.value.reason_code == "sec_mapping_asset_identity_mismatch"


def test_sec_mapping_wrong_kind_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path, sec=True)
    sec_run = JobRun.objects.get(pk=stages["sec_fundamentals"]["job_run_id"])
    # Same provider (`sec`), but the real `companyfacts` asset instead of the
    # reviewed `sec_ticker_mapping` -- a plausible "child-copied a real SEC
    # asset of the wrong kind" substitution.
    from stanstock.data.models import FundamentalFact

    fact = FundamentalFact.objects.select_related("source_asset").first()
    assert fact is not None
    details = dict(sec_run.details)
    details["mapping_asset_id"] = str(fact.source_asset_id)
    details["mapping_sha256"] = fact.source_asset.sha256
    JobRun.objects.filter(pk=sec_run.pk).update(details=details)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=True,
        )
    assert excinfo.value.reason_code == "sec_mapping_asset_identity_mismatch"


def test_sec_mapping_not_reviewed_source_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same generic provider/kind/subject, different physically valid bytes.

    A checksum-valid `sec_ticker_mapping` asset that is *not* the reviewed
    CIK config's pinned `source_sha256` (e.g. a later, unreviewed refresh of
    the same SEC endpoint) must fail, not merely match on shape.
    """
    stages, config = _build_verified_state(monkeypatch, tmp_path, sec=True)
    sec_run = JobRun.objects.get(pk=stages["sec_fundamentals"]["job_run_id"])
    original = DataAsset.objects.get(pk=sec_run.details["mapping_asset_id"])
    store = AssetStore(tmp_path)
    written = store.write_bytes("sec/mapping-unreviewed.json", b'{"cik_lookup": {"other": true}}')
    substitute = DataAsset.objects.create(
        provider=original.provider,
        kind=original.kind,
        subject=original.subject,
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=original.retrieved_at,
        available_at=original.available_at,
    )
    details = dict(sec_run.details)
    details["mapping_asset_id"] = str(substitute.pk)
    details["mapping_sha256"] = substitute.sha256
    JobRun.objects.filter(pk=sec_run.pk).update(details=details)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=True,
        )
    assert excinfo.value.reason_code == "sec_mapping_asset_not_reviewed"


# --- F4: fail-closed asset payload validation ------------------------------


def _clone_prediction(template: Prediction, **overrides: Any) -> Prediction:
    """Create a brand-new `Prediction` row cloned from `template`'s fields.

    `Prediction.save`/`delete` is guarded by a database-level immutability
    trigger, so these adversarial fixtures never mutate an existing row --
    only insert an additional, independently-crafted one to prove per-row
    verification catches it among the otherwise-legitimate set.
    """
    fields = {
        name: getattr(template, name)
        for name in (
            "analysis_id",
            "listing_id",
            "generated_at",
            "target_date",
            "issued_on_time",
            "horizon",
            "evidence_role",
            "evidence_grade",
            "source_mode",
            "price_provider",
            "price_subject",
            "price_at_prediction",
            "bear_return",
            "base_return",
            "bull_return",
            "probability_positive",
            "confidence",
            "confidence_status",
            "insufficiency_reason",
            "recommendation",
            "overall_score",
            "component_scores",
            "model_version",
            "method_version",
            "config_hash",
            "data_cutoff",
            "source_assets",
            "calculation",
            "code_revision",
        )
    }
    fields.update(overrides)
    return Prediction.objects.create(**fields)


def _bump_prediction_count(stages: dict[str, dict[str, object]], delta: int) -> None:
    """Keep the market stage's recorded prediction count consistent with an
    intentionally-added rogue row, so the earlier aggregate-count guard does
    not mask the deeper per-row identity check this test targets -- mirrors
    the exact "same cardinality" attack shape used for the F1 regression."""
    market_run = JobRun.objects.get(pk=stages["market"]["job_run_id"])
    details = dict(market_run.details)
    details["predictions"] = details["predictions"] + delta
    JobRun.objects.filter(pk=market_run.pk).update(details=details)


def test_malformed_source_asset_id_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    analysis = StockAnalysis.objects.first()
    assert analysis is not None
    data_quality = dict(analysis.data_quality)
    source_assets = [dict(entry) for entry in data_quality["source_assets"]]
    source_assets[0]["id"] = "not-a-uuid"
    data_quality["source_assets"] = source_assets
    StockAnalysis.objects.filter(pk=analysis.pk).update(data_quality=data_quality)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "source_assets_entry_malformed"


def test_json_vs_row_checksum_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path, sec=True)
    sec_run = JobRun.objects.get(pk=stages["sec_fundamentals"]["job_run_id"])
    mapping_asset = DataAsset.objects.get(pk=sec_run.details["mapping_asset_id"])
    template = Prediction.objects.first()
    assert template is not None
    # A tampered *extra* entry for a real, checksum-valid asset that no
    # other row references -- `Prediction.source_assets` is only ever
    # inserted fresh (never updated in place), so an internally-consistent
    # tampered claim can only be introduced this way. It disagrees only
    # with the actual registered `DataAsset` row, not with any other JSON
    # copy, isolating the DB cross-check this test targets.
    tampered_entry = {
        "id": str(mapping_asset.pk),
        "provider": mapping_asset.provider,
        "kind": mapping_asset.kind,
        "subject": mapping_asset.subject,
        "sha256": "f" * 64,
    }
    _clone_prediction(
        template,
        model_version=f"{template.model_version}-rogue",
        source_assets=[*template.source_assets, tampered_entry],
    )
    _bump_prediction_count(stages, 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=True,
        )
    assert excinfo.value.reason_code == "source_assets_identity_mismatch"


# --- F5: predictions must be independently, exactly bound ------------------


def test_rogue_spy_prediction_transplant_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    template = Prediction.objects.filter(evidence_role=Prediction.EvidenceRole.DECISION).first()
    assert template is not None
    spy_listing = Listing.objects.get(ticker="SPY")
    _clone_prediction(
        template,
        listing_id=spy_listing.pk,
        model_version=f"{template.model_version}-rogue",
    )
    _bump_prediction_count(stages, 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "prediction_listing_transplanted"


def test_prediction_wrong_target_date_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    template = Prediction.objects.first()
    assert template is not None
    _clone_prediction(
        template,
        target_date=date(2020, 1, 1),
        model_version=f"{template.model_version}-rogue",
    )
    _bump_prediction_count(stages, 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "prediction_target_mismatch"


def test_research_grade_prediction_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    template = Prediction.objects.first()
    assert template is not None
    _clone_prediction(
        template,
        evidence_grade=UniverseSnapshot.Grade.RESEARCH,
        model_version=f"{template.model_version}-rogue",
    )
    _bump_prediction_count(stages, 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "prediction_not_observed"


def test_prediction_horizon_role_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    template = Prediction.objects.filter(evidence_role=Prediction.EvidenceRole.DECISION).first()
    assert template is not None
    # `Prediction.Horizon.MEDIUM` ("medium") is a legacy horizon the
    # database-level `prediction_horizon_role_valid` CHECK still permits for
    # decision-role rows, but no *current* decision issuance ever produces
    # it -- only `SHORT` is a live decision horizon. This proves the
    # verifier's stricter, current-issuance-only rule catches what the
    # broader legacy-compatible DB constraint alone would not.
    _clone_prediction(
        template,
        horizon=Prediction.Horizon.MEDIUM,
        model_version=f"{template.model_version}-rogue",
    )
    _bump_prediction_count(stages, 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "prediction_horizon_role_mismatch"


# --- F6: evaluation stage details must be independently re-derived ---------


def test_evaluation_fabricated_prediction_id_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    evaluation_run = JobRun.objects.get(pk=stages["evaluation"]["job_run_id"])
    details = dict(evaluation_run.details)
    details["eligible_predictions"] = 1
    details["evaluated_prediction_ids"] = [str(uuid4())]
    # Keep the structural `actions` claim internally consistent with the
    # fabricated count so the *identity* check (not the cheaper structural
    # one) is what actually fires here.
    details["actions"] = {"created": 1}
    JobRun.objects.filter(pk=evaluation_run.pk).update(details=details)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "evaluation_candidate_set_mismatch"


def test_evaluation_time_naive_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A naive (timezone-unaware) `evaluation_time` cannot be safely
    compared against timezone-aware `generated_at`/`evaluated_at` fields
    and must be rejected explicitly rather than surfacing as an
    unrelated datetime-comparison error deeper in verification.
    """
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    evaluation_run = JobRun.objects.get(pk=stages["evaluation"]["job_run_id"])
    details = dict(evaluation_run.details)
    details["evaluation_time"] = DECISION_TIME.replace(tzinfo=None).isoformat()
    JobRun.objects.filter(pk=evaluation_run.pk).update(details=details)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "evaluation_time_naive"


def test_evaluation_inflated_count_without_matching_ids_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    evaluation_run = JobRun.objects.get(pk=stages["evaluation"]["job_run_id"])
    details = dict(evaluation_run.details)
    # `evaluated_prediction_ids` legitimately stays empty (nothing matured
    # yet), but the eligible_predictions claim is inflated -- a fabricated
    # detail unsupported by its own companion identity list.
    details["eligible_predictions"] = 1
    JobRun.objects.filter(pk=evaluation_run.pk).update(details=details)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "evaluation_count_mismatch"


# --- Portfolio: exact snapshot binding, not an arbitrary latest row --------


def test_portfolio_stale_unrelated_snapshot_is_not_silently_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", CODE_REVISION)
    config = refresh_fixtures.build_universe_config(symbols=("AAA",), minimum_eligible=1)
    refresh_fixtures.enable_twelve_data_provider()
    refresh_fixtures.set_twelve_data_api_key(monkeypatch)
    refresh_fixtures.patch_twelve_data_provider(
        monkeypatch,
        config,
        target_date=TARGET_DATE,
        retrieved_at=DECISION_TIME,
    )
    monkeypatch.setattr("stanstock.data.live_us.timezone.now", lambda: DECISION_TIME)
    Portfolio.objects.create(
        owner=refresh_fixtures.create_portfolio_owner(),
        name="Test Portfolio",
        base_currency="USD",
    )

    prepared = PreparedUsDailyJob(
        config=config,
        target_date=TARGET_DATE,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        decision_time=DECISION_TIME,
    )
    market_run = execute_us_daily_job(prepared, require_observed=True)
    evaluation_run = execute_prediction_evaluation_job(
        provider=twelve_data.PROVIDER,
        evaluation_date=TARGET_DATE,
        evaluation_time=DECISION_TIME,
        benchmark_subject=config.benchmark_symbol,
    )
    portfolio_run = execute_portfolio_snapshot_job(
        target_date=TARGET_DATE,
        require_session_date=True,
        require_all=True,
    )
    # Tamper with the child's own recorded exact-snapshot reference so it
    # points at a fabricated identifier -- a stale/unrelated "latest
    # same-date snapshot" query could otherwise silently substitute a
    # different real snapshot row for the one the child actually produced.
    details = dict(portfolio_run.details)
    details["snapshot_ids"] = {key: str(uuid4()) for key in details["snapshot_ids"]}
    JobRun.objects.filter(pk=portfolio_run.pk).update(details=details)
    stages = {
        "market": _stage_entry(market_run),
        "evaluation": _stage_entry(evaluation_run),
        "portfolio_snapshots": _stage_entry(portfolio_run),
    }

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "portfolio_snapshot_missing"


# --- R3 F-1/F-3: evaluation binds to its own execution boundary, and a
# fabricated zero cannot hide a genuine unevaluated candidate ---------------


def test_lagging_evaluation_from_an_earlier_run_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A short prediction that already matured 11+ sessions before target
    (a missed-day/lagging-series catch-up scenario) still verifies.

    `PredictionOutcome.evaluation_date` is the maturity *observation* date,
    which can legitimately be well before the scheduled target. This
    prediction lives in an entirely separate, earlier `AnalysisRun` --
    invisible to the main run's own StockAnalysis/Prediction checks -- and
    already carries a genuine pre-existing terminal (`matured`) outcome
    from strictly before this evaluation execution, so the real
    `evaluate_predictions` job legitimately never re-touches it (the
    documented no-op/skip path); the verifier's independent re-derivation
    must still accept it rather than requiring `evaluation_date == target_date`.
    """
    lagging_target = TARGET_DATE - timedelta(days=30)
    injected: dict[str, Prediction] = {}

    def _inject() -> None:
        lagging = _create_lagging_prediction(target_date=lagging_target)
        injected["prediction"] = lagging
        PredictionOutcome.objects.create(
            prediction=lagging,
            evaluated_at=DECISION_TIME - timedelta(days=10),
            evaluation_date=lagging_target + timedelta(days=15),
            status=PredictionOutcome.Status.MATURED,
            actual_return=Decimal("0.02"),
            benchmark_return=Decimal("0.01"),
            success=True,
            direction_correct=True,
            interval_covered=True,
            resolution="matured before this evaluation execution",
            error=Decimal("0"),
            signed_error=Decimal("0"),
        )

    stages, config = _build_verified_state(monkeypatch, tmp_path, before_evaluation=_inject)

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )

    assert result["status"] == "verified"
    lagging = injected["prediction"]
    outcome = PredictionOutcome.objects.get(pk=lagging.pk)
    assert outcome.evaluation_date < TARGET_DATE


def test_evaluation_child_freshly_matures_a_genuinely_lagging_prediction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The *current* evaluation execution itself matures a genuinely
    pending prediction from an earlier run (not a pre-existing terminal
    no-op skip): the real `evaluate_predictions` call resolves a matured
    outcome with `evaluated_at == evaluation_time` and
    `evaluation_date < target_date`, and the verifier's independent
    candidate re-derivation must accept it.
    """
    lagging_target = TARGET_DATE - timedelta(days=30)
    injected: dict[str, Prediction] = {}

    def _inject() -> None:
        lagging = _create_lagging_prediction(target_date=lagging_target)
        injected["prediction"] = lagging
        assert not PredictionOutcome.objects.filter(prediction=lagging).exists()
        _register_lagging_price_history(
            tmp_path=tmp_path,
            subject=lagging.listing.provider_symbol,
            baseline_date=lagging_target,
            baseline_close=Decimal("100"),
            available_at=DECISION_TIME,
        )

    stages, config = _build_verified_state(monkeypatch, tmp_path, before_evaluation=_inject)

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )

    assert result["status"] == "verified"
    lagging = injected["prediction"]
    outcome = PredictionOutcome.objects.get(prediction=lagging)
    assert outcome.status == PredictionOutcome.Status.MATURED
    assert outcome.evaluated_at == DECISION_TIME
    assert outcome.evaluation_date < TARGET_DATE
    evaluation_run = JobRun.objects.get(pk=stages["evaluation"]["job_run_id"])
    assert str(lagging.pk) in evaluation_run.details["evaluated_prediction_ids"]


def test_stale_pre_child_outcome_falsely_claimed_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale, non-terminal outcome falsely named as newly evaluated fails.

    The prediction is genuinely mature and provider-matching (so it does
    belong to the re-derived candidate set), but its persisted outcome was
    never actually touched by *this* evaluation execution -- and no real
    price evidence exists for its listing at all, so an independent replay
    cannot reproduce the falsely-claimed "pending" resolution/metadata.
    """
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    evaluation_run = JobRun.objects.get(pk=stages["evaluation"]["job_run_id"])
    stale = _create_lagging_prediction(target_date=TARGET_DATE - timedelta(days=30))
    stale_time = DECISION_TIME - timedelta(days=5)
    PredictionOutcome.objects.create(
        prediction=stale,
        evaluated_at=stale_time,
        evaluation_date=stale.target_date,
        status=PredictionOutcome.Status.UNRESOLVED,
        resolution="pending",
    )
    details = dict(evaluation_run.details)
    details["eligible_predictions"] = 1
    details["evaluated_prediction_ids"] = [str(stale.pk)]
    details["actions"] = {"skipped": 1}
    details["outcome_statuses"] = {PredictionOutcome.Status.UNRESOLVED: 1}
    JobRun.objects.filter(pk=evaluation_run.pk).update(details=details)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "evaluation_outcome_replay_mismatch"


def test_unchanged_unresolved_same_target_retry_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The established rule against rewriting an unchanged unresolved
    outcome merely to create a fresh timestamp must not make a legitimate
    retry unverifiable. An earlier execution already left a genuinely
    unresolved outcome -- real `evaluate_prediction` output from only 3 of
    the 10 required sessions, not a fabricated fixture -- whose content an
    independent fresh replay still reproduces byte-for-byte, even though
    its own `evaluated_at` predates the recorded `evaluation_time`; the
    real evaluation job run this test drives then re-confirms it unchanged
    (the established `_outcome_matches` skip) rather than rewriting it.
    """
    lagging_target = TARGET_DATE - timedelta(days=30)
    prior_time = DECISION_TIME - timedelta(days=1)
    injected: dict[str, Prediction] = {}

    def _inject() -> None:
        lagging = _create_lagging_prediction(target_date=lagging_target)
        injected["prediction"] = lagging
        _register_lagging_price_history(
            tmp_path=tmp_path,
            subject=lagging.listing.provider_symbol,
            baseline_date=lagging_target,
            baseline_close=Decimal("100"),
            available_at=prior_time,
            session_count=3,
        )
        evaluate_prediction(
            lagging,
            provider=twelve_data.PROVIDER,
            evaluation_date=TARGET_DATE,
            evaluation_time=prior_time,
        )

    stages, config = _build_verified_state(monkeypatch, tmp_path, before_evaluation=_inject)

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )

    assert result["status"] == "verified"
    lagging = injected["prediction"]
    outcome = PredictionOutcome.objects.get(prediction=lagging)
    assert outcome.status == PredictionOutcome.Status.UNRESOLVED
    assert outcome.evaluated_at == prior_time


def test_evaluation_outcome_before_prediction_target_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A terminal outcome cannot observe maturity before its own
    prediction's issuance target date."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    evaluation_run = JobRun.objects.get(pk=stages["evaluation"]["job_run_id"])
    lagging_target = TARGET_DATE - timedelta(days=30)
    matured = _create_lagging_prediction(target_date=lagging_target)
    PredictionOutcome.objects.create(
        prediction=matured,
        evaluated_at=DECISION_TIME,
        evaluation_date=lagging_target - timedelta(days=1),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.01"),
        success=True,
        direction_correct=True,
        interval_covered=True,
        resolution="matured",
    )
    details = dict(evaluation_run.details)
    details["eligible_predictions"] += 1
    details["evaluated_prediction_ids"].append(str(matured.pk))
    details["actions"]["created"] = details["actions"].get("created", 0) + 1
    details["outcome_statuses"][PredictionOutcome.Status.MATURED] = (
        details["outcome_statuses"].get(PredictionOutcome.Status.MATURED, 0) + 1
    )
    JobRun.objects.filter(pk=evaluation_run.pk).update(details=details)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "evaluation_outcome_date_before_prediction_target"


def test_evaluation_outcome_maturity_date_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A freshly-touched terminal outcome whose observation date does not
    match its own prediction's independently re-derived required nth
    observed session must fail closed."""
    lagging_target = TARGET_DATE - timedelta(days=30)
    injected: dict[str, Prediction] = {}

    def _inject() -> None:
        lagging = _create_lagging_prediction(target_date=lagging_target)
        injected["prediction"] = lagging
        _register_lagging_price_history(
            tmp_path=tmp_path,
            subject=lagging.listing.provider_symbol,
            baseline_date=lagging_target,
            baseline_close=Decimal("100"),
            available_at=DECISION_TIME,
        )

    stages, config = _build_verified_state(monkeypatch, tmp_path, before_evaluation=_inject)
    lagging = injected["prediction"]
    real_outcome = PredictionOutcome.objects.get(prediction=lagging)
    # Fabricate a wrong observation date on the outcome the real evaluation
    # execution just produced -- the independently re-derived nth observed
    # session must catch this rather than trusting the persisted date.
    PredictionOutcome.objects.filter(pk=real_outcome.pk).update(
        evaluation_date=real_outcome.evaluation_date + timedelta(days=1)
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "evaluation_outcome_maturity_date_invalid"


def test_evaluation_fabricated_zero_with_remaining_candidate_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fabricated `eligible_predictions=0` cannot hide a genuine, matured,
    provider-matching prediction that the real child never actually saw."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    _create_lagging_prediction(target_date=TARGET_DATE - timedelta(days=30))
    evaluation_run = JobRun.objects.get(pk=stages["evaluation"]["job_run_id"])
    details = dict(evaluation_run.details)
    details["eligible_predictions"] = 0
    details["evaluated_prediction_ids"] = []
    details["actions"] = {}
    details["outcome_statuses"] = {}
    JobRun.objects.filter(pk=evaluation_run.pk).update(details=details)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "evaluation_candidate_set_mismatch"


def test_evaluation_candidate_generated_at_evaluation_time_boundary_is_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A mature, provider-matching prediction that existed by the exact
    recorded `evaluation_time` (inclusive) still makes a fabricated
    `eligible_predictions=0` fail -- the pre-child cutoff is `<=`, not `<`.
    """
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    _create_lagging_prediction(
        target_date=TARGET_DATE - timedelta(days=30), generated_at=DECISION_TIME
    )
    evaluation_run = JobRun.objects.get(pk=stages["evaluation"]["job_run_id"])
    details = dict(evaluation_run.details)
    details["eligible_predictions"] = 0
    details["evaluated_prediction_ids"] = []
    details["actions"] = {}
    details["outcome_statuses"] = {}
    JobRun.objects.filter(pk=evaluation_run.pk).update(details=details)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "evaluation_candidate_set_mismatch"


def test_prediction_generated_after_evaluation_time_does_not_retroactively_break_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A prediction created after the recorded `evaluation_time` (a race
    with verification, or a later reissue) must not retroactively count
    against an evaluation execution that could not possibly have seen it.
    """
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    _create_lagging_prediction(
        target_date=TARGET_DATE - timedelta(days=30),
        generated_at=DECISION_TIME + timedelta(minutes=1),
    )

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )
    assert result["status"] == "verified"


# --- R3 F-2: a StockAnalysis's own price source subject must bind to its
# own listing, never a run-wide or benchmark identity ----------------------


def test_stock_analysis_price_source_subject_transplant_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    analysis = StockAnalysis.objects.get(listing__ticker="AAA")
    spy_listing = Listing.objects.get(ticker="SPY")
    data_quality = dict(analysis.data_quality)
    price_source = dict(data_quality["price_source"])
    price_source["subject"] = spy_listing.provider_symbol
    data_quality["price_source"] = price_source
    StockAnalysis.objects.filter(pk=analysis.pk).update(data_quality=data_quality)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "stock_analysis_price_source_subject_mismatch"


# --- R5 F3: a mutable StockAnalysis's duplicated fields must be bound to
# the immutable Prediction ledger that actually carried them -----------------


def test_stock_analysis_current_price_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    analysis = StockAnalysis.objects.get(listing__ticker="AAA")
    StockAnalysis.objects.filter(pk=analysis.pk).update(
        current_price=analysis.current_price + Decimal("1.00")
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "stock_analysis_current_price_mismatch"


def test_stock_analysis_overall_score_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    analysis = StockAnalysis.objects.get(listing__ticker="AAA")
    StockAnalysis.objects.filter(pk=analysis.pk).update(
        overall_score=analysis.overall_score + Decimal("1.0000")
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "stock_analysis_overall_score_mismatch"


def test_stock_analysis_recommendation_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    analysis = StockAnalysis.objects.get(listing__ticker="AAA")
    rogue_recommendation = (
        Recommendation.AVOID
        if analysis.recommendation != Recommendation.AVOID
        else Recommendation.BUY
    )
    StockAnalysis.objects.filter(pk=analysis.pk).update(recommendation=rogue_recommendation)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "stock_analysis_recommendation_mismatch"


def test_stock_analysis_component_scores_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    analysis = StockAnalysis.objects.get(listing__ticker="AAA")
    rogue_scores = dict(analysis.component_scores)
    rogue_horizons = dict(rogue_scores["horizons"])
    a_key = next(iter(rogue_horizons))
    rogue_horizons[a_key] = float(rogue_horizons[a_key]) + 5.0
    rogue_scores["horizons"] = rogue_horizons
    StockAnalysis.objects.filter(pk=analysis.pk).update(component_scores=rogue_scores)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "stock_analysis_component_scores_mismatch"


def test_stock_analysis_scenario_value_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Independently mutating a `StockAnalysis`'s own `short_scenario` (with
    the immutable `Prediction` ledger left untouched) must fail closed --
    proving scenario values are bound to their own prediction, not merely
    replayed from the same mutable analysis row that carries them."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    analysis = StockAnalysis.objects.get(listing__ticker="AAA")
    forecast_scenarios = dict(analysis.forecast_scenarios)
    horizons = dict(forecast_scenarios["horizons"])
    rogue_scenario = dict(horizons["short"])
    rogue_scenario["confidence"] = float(rogue_scenario["confidence"]) + 5.0
    horizons["short"] = rogue_scenario
    forecast_scenarios["horizons"] = horizons
    StockAnalysis.objects.filter(pk=analysis.pk).update(forecast_scenarios=forecast_scenarios)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "stock_analysis_scenario_mismatch"


# --- R3 F-4: LatestMarketData must match its own bound asset's own rows ---


def test_latest_market_data_close_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    latest = LatestMarketData.objects.filter(session_date=TARGET_DATE).first()
    assert latest is not None
    # `LatestMarketData` is mutable current-state (unlike its immutable
    # source asset), so an in-place edit is the exact adversarial shape this
    # check exists to catch.
    LatestMarketData.objects.filter(pk=latest.pk).update(close=latest.close + 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "latest_market_data_close_mismatch"


def test_latest_market_data_volume_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    latest = LatestMarketData.objects.filter(session_date=TARGET_DATE).first()
    assert latest is not None
    LatestMarketData.objects.filter(pk=latest.pk).update(volume=latest.volume + 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "latest_market_data_volume_mismatch"


def test_latest_market_data_quantum_close_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A canonical exact-Decimal comparison, not a float/epsilon tolerance,
    must catch even the smallest persisted-precision close deviation."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    latest = LatestMarketData.objects.filter(session_date=TARGET_DATE).first()
    assert latest is not None
    close_places = LatestMarketData._meta.get_field("close").decimal_places
    smallest_unit = Decimal(1).scaleb(-close_places)
    LatestMarketData.objects.filter(pk=latest.pk).update(close=latest.close + smallest_unit)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "latest_market_data_close_mismatch"


def test_latest_market_data_volume_asymmetric_null_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`None` on one side and a real value on the other must fail closed
    without ever raising a bare `int(None)` `TypeError`."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    latest = LatestMarketData.objects.filter(session_date=TARGET_DATE).first()
    assert latest is not None
    assert latest.volume is not None
    LatestMarketData.objects.filter(pk=latest.pk).update(volume=None)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "latest_market_data_volume_mismatch"


# --- R3 F-5/F-8: prediction price binding and config-derived decision
# horizons (never a hard-coded set) -----------------------------------------


def test_prediction_synthetic_source_mode_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    template = Prediction.objects.filter(evidence_role=Prediction.EvidenceRole.DECISION).first()
    assert template is not None
    _clone_prediction(
        template,
        source_mode="synthetic",
        model_version=f"{template.model_version}-rogue",
    )
    _bump_prediction_count(stages, 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "prediction_source_mode_invalid"


def test_prediction_wrong_price_subject_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    template = Prediction.objects.filter(evidence_role=Prediction.EvidenceRole.DECISION).first()
    assert template is not None
    _clone_prediction(
        template,
        price_subject="NOT-THE-LISTING",
        model_version=f"{template.model_version}-rogue",
    )
    _bump_prediction_count(stages, 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "prediction_price_subject_mismatch"


def test_decision_horizon_derived_from_scoring_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A reviewed config requesting `medium` as a decision horizon is
    accepted -- proving decision horizons are derived from the loaded
    scoring config's own `supported_horizons`, never a hard-coded set."""
    from stanstock.data.management.config_loader import (
        default_us_scoring_config_path as get_default_scoring_path,
    )

    base_text = get_default_scoring_path().read_text(encoding="utf-8")
    modified_text = base_text.replace(
        "supported_horizons: [short]", "supported_horizons: [short, medium]"
    ).replace(
        "buy_max_bear_downside:\n    short: -0.08",
        "buy_max_bear_downside:\n    short: -0.08\n    medium: -0.15",
    )
    assert modified_text != base_text
    assert "medium: -0.15" in modified_text
    scoring_path = tmp_path / "scoring-with-medium.yml"
    scoring_path.write_text(modified_text, encoding="utf-8")
    monkeypatch.setattr(
        "stanstock.data.live_us.default_us_scoring_config_path", lambda: scoring_path
    )
    monkeypatch.setattr(
        "stanstock.core.refresh_verification.default_us_scoring_config_path", lambda: scoring_path
    )

    stages, config = _build_verified_state(monkeypatch, tmp_path)
    template = Prediction.objects.filter(evidence_role=Prediction.EvidenceRole.DECISION).first()
    assert template is not None
    medium_scenario = template.analysis.medium_scenario

    def _round(value: float | None, places: int) -> Decimal | None:
        return None if value is None else Decimal(str(round(value, places)))

    _clone_prediction(
        template,
        horizon=Prediction.Horizon.MEDIUM,
        # Stable, explicit, <=40-char literal: `model_version` is a
        # `CharField(max_length=40)` enforced by PostgreSQL (though not by
        # SQLite), so appending a suffix to the real, already
        # near-40-char production `model_version` can silently overflow
        # only on PostgreSQL, raising a `DataError` before this test's own
        # intended assertion ever runs.
        model_version="test-medium-decision-clone",
        bear_return=_round(medium_scenario["bear"], 4),
        base_return=_round(medium_scenario["base"], 4),
        bull_return=_round(medium_scenario["bull"], 4),
        probability_positive=_round(medium_scenario.get("probability_positive"), 4),
        confidence=_round(medium_scenario["confidence"], 2),
        confidence_status=medium_scenario["confidence_status"],
        insufficiency_reason=medium_scenario["insufficiency_reason"],
    )
    _bump_prediction_count(stages, 1)

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )
    assert result["status"] == "verified"


def test_advisory_long_config_hash_branch_is_checked_independently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A long-horizon (3y/5y) advisory prediction is bound to its own
    reviewed long-forecast config digest, never the main scoring digest or
    the medium-forecast digest."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    template = Prediction.objects.filter(evidence_role=Prediction.EvidenceRole.DECISION).first()
    assert template is not None
    _clone_prediction(
        template,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        horizon=Prediction.Horizon.THREE_YEAR,
        config_hash=template.config_hash,  # the *scoring* digest, not the long digest
        model_version=f"{template.model_version}-long-rogue",
    )
    _bump_prediction_count(stages, 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "prediction_advisory_config_mismatch"


def test_advisory_medium_method_version_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A same-config-hash medium-advisory clone with a wrong
    `method_version` fails independently of the config_hash check. The
    default fixture already issues genuine (withheld) 6m/12m advisory
    predictions, so cloning one of them preserves exact scenario/identity
    consistency and isolates the method_version defect."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    template = Prediction.objects.filter(
        evidence_role=Prediction.EvidenceRole.ADVISORY, horizon=Prediction.Horizon.SIX_MONTH
    ).first()
    assert template is not None
    _clone_prediction(
        template,
        method_version=f"{template.method_version}-rogue",
        # See `test_decision_horizon_derived_from_scoring_config`'s own
        # comment: a stable, explicit, <=40-char literal avoids a
        # PostgreSQL-only `DataError` from overflowing the real,
        # already near-40-char production `model_version`.
        model_version="test-medium-method-rogue-clone",
    )
    _bump_prediction_count(stages, 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "prediction_advisory_method_version_mismatch"


def test_advisory_long_method_version_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A same-config-hash long-advisory clone with a wrong `method_version`
    fails independently of the config_hash check. Long advisory is not
    issued by the default fixture, so this mirrors
    `test_advisory_long_config_hash_branch_is_checked_independently`'s
    clone-from-decision-template shape -- the per-row check raises before
    the analysis-field/completeness checks are ever reached."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    template = Prediction.objects.filter(evidence_role=Prediction.EvidenceRole.DECISION).first()
    assert template is not None
    long_config = load_long_forecast_config()
    long_digest = long_forecast_config_hash(long_config)
    _clone_prediction(
        template,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        horizon=Prediction.Horizon.THREE_YEAR,
        config_hash=long_digest,
        method_version=f"{long_config.version}-rogue",
        model_version=f"{template.model_version}-long-rogue",
    )
    _bump_prediction_count(stages, 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "prediction_advisory_method_version_mismatch"


def test_withheld_advisory_scenario_still_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A legitimate withheld medium-advisory scenario (present 6m/12m
    rows, `insufficiency_reason` populated, every return field `None`)
    still verifies with correct identity -- withholding is not the same
    as omitting the required prediction rows, and the default fixture's
    minimal history genuinely withholds this forecast."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    advisory_rows = list(
        Prediction.objects.filter(
            evidence_role=Prediction.EvidenceRole.ADVISORY,
            horizon__in=(Prediction.Horizon.SIX_MONTH, Prediction.Horizon.TWELVE_MONTH),
        )
    )
    assert len(advisory_rows) == 2
    assert all(row.bear_return is None and row.insufficiency_reason for row in advisory_rows)

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )
    assert result["status"] == "verified"


def test_omitted_configured_decision_horizon_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A reviewed scoring config requesting an additional decision horizon,
    where production's own issuance for one analysis genuinely omits that
    horizon's row, now fails closed *before* the run is ever persisted --
    proving exact per-analysis role/horizon completeness is enforced at
    write time by `research.service`'s own precomputed output plan
    (`refresh-output-verification` slice C1), not merely caught later by
    this module's downstream review of however many rows happen to exist.

    This is a strictly earlier version of the same safety property this
    test used to prove only via `verify_scheduled_refresh`'s legacy
    `stock_analysis_prediction_set_incomplete` check: the omission can no
    longer even reach a persisted, reviewable state at all, so the market
    stage job itself now fails first."""
    from stanstock.data.management.config_loader import (
        default_us_scoring_config_path as get_default_scoring_path,
    )

    base_text = get_default_scoring_path().read_text(encoding="utf-8")
    modified_text = base_text.replace(
        "supported_horizons: [short]", "supported_horizons: [short, medium]"
    ).replace(
        "buy_max_bear_downside:\n    short: -0.08",
        "buy_max_bear_downside:\n    short: -0.08\n    medium: -0.15",
    )
    assert modified_text != base_text
    scoring_path = tmp_path / "scoring-with-medium-omitted.yml"
    scoring_path.write_text(modified_text, encoding="utf-8")
    monkeypatch.setattr(
        "stanstock.data.live_us.default_us_scoring_config_path", lambda: scoring_path
    )
    monkeypatch.setattr(
        "stanstock.core.refresh_verification.default_us_scoring_config_path", lambda: scoring_path
    )

    import stanstock.research.service as research_service

    real_append_predictions = research_service.append_predictions

    def _drop_medium_horizon(*, supported_horizons: tuple[str, ...], **kwargs: Any) -> Any:
        return real_append_predictions(
            supported_horizons=tuple(h for h in supported_horizons if h != "medium"), **kwargs
        )

    monkeypatch.setattr(research_service, "append_predictions", _drop_medium_horizon)

    with pytest.raises(ValueError, match="does not match its precomputed output plan"):
        _build_verified_state(monkeypatch, tmp_path)


# --- R3 F-6: portfolio snapshots must be independently recomputed from
# their own live inputs, not merely id-bound --------------------------------


def _build_portfolio_with_holdings_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, Any], UsUniverseConfig, Portfolio]:
    """Real market+evaluation+portfolio stages with one non-empty holding.

    Shared by the happy-path test and the F8 per-field replay-mismatch
    regressions below so each mutation test only needs to corrupt one
    field of an otherwise-genuine snapshot/holding pair.
    """
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", CODE_REVISION)
    config = refresh_fixtures.build_universe_config(symbols=("AAA",), minimum_eligible=1)
    refresh_fixtures.enable_twelve_data_provider()
    refresh_fixtures.set_twelve_data_api_key(monkeypatch)
    refresh_fixtures.patch_twelve_data_provider(
        monkeypatch,
        config,
        target_date=TARGET_DATE,
        retrieved_at=DECISION_TIME,
    )
    monkeypatch.setattr("stanstock.data.live_us.timezone.now", lambda: DECISION_TIME)

    prepared = PreparedUsDailyJob(
        config=config,
        target_date=TARGET_DATE,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        decision_time=DECISION_TIME,
    )
    market_run = execute_us_daily_job(prepared, require_observed=True)

    aaa_listing = Listing.objects.get(ticker="AAA")
    portfolio = Portfolio.objects.create(
        owner=refresh_fixtures.create_portfolio_owner(),
        name="Held",
        base_currency="USD",
    )
    upsert_holding(
        portfolio=portfolio,
        listing=aaa_listing,
        quantity=Decimal("3"),
        average_cost=Decimal("50"),
    )

    evaluation_run = execute_prediction_evaluation_job(
        provider=twelve_data.PROVIDER,
        evaluation_date=TARGET_DATE,
        evaluation_time=DECISION_TIME,
        benchmark_subject=config.benchmark_symbol,
    )
    portfolio_run = execute_portfolio_snapshot_job(
        target_date=TARGET_DATE,
        require_session_date=True,
        require_all=True,
    )
    stages = {
        "market": _stage_entry(market_run),
        "evaluation": _stage_entry(evaluation_run),
        "portfolio_snapshots": _stage_entry(portfolio_run),
    }
    return stages, config, portfolio


def test_portfolio_snapshot_with_holdings_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config, _portfolio = _build_portfolio_with_holdings_state(monkeypatch, tmp_path)

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )
    assert result["status"] == "verified"
    assert result["portfolio"]["active_portfolios"] == 1


@pytest.mark.parametrize(
    "field, wrong_value",
    [
        ("total_value", Decimal("999999.000000")),
        ("securities_value", Decimal("999999.000000")),
        ("unrealized_gain", Decimal("999999.000000")),
        ("return_pct", Decimal("9.99999999")),
        ("oldest_price_date", TARGET_DATE - timedelta(days=999)),
        ("newest_price_date", TARGET_DATE - timedelta(days=999)),
        ("dividends_included", True),
        ("corporate_action_warnings", 7),
    ],
)
def test_portfolio_snapshot_derived_field_fabrication_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, wrong_value: object
) -> None:
    """A correct-input-hash snapshot with one derived field independently
    fabricated must fail closed -- an exact snapshot id is not sufficient,
    every field production actually emits must be replayed and compared.

    `PortfolioSnapshot`/`PortfolioSnapshotHolding` are DB-trigger immutable
    (no in-place `.update()` on an existing row), so the fabrication uses a
    second, never-snapshotted portfolio with the identical holding: its own
    real valuation supplies a genuinely correct `input_hash`, then exactly
    one derived field is substituted before the row is ever persisted."""
    stages, config, portfolio1 = _build_portfolio_with_holdings_state(monkeypatch, tmp_path)
    portfolio_run = JobRun.objects.get(pk=stages["portfolio_snapshots"]["job_run_id"])

    portfolio2 = Portfolio.objects.create(
        owner=refresh_fixtures.create_portfolio_owner(username="fabricated-sibling-owner"),
        name="Fabricated Sibling",
        base_currency="USD",
    )
    aaa_listing = Listing.objects.get(ticker="AAA")
    upsert_holding(
        portfolio=portfolio2,
        listing=aaa_listing,
        quantity=Decimal("3"),
        average_cost=Decimal("50"),
    )
    # Reloaded so `cash_balance` carries its exact persisted Decimal scale --
    # matching the verifier's own fresh re-query -- rather than the
    # in-memory zero-argument default's differing string representation.
    portfolio2 = Portfolio.objects.get(pk=portfolio2.pk)
    valuation2 = calculate_portfolio_valuation(portfolio2, expected_as_of_date=TARGET_DATE)
    assert valuation2.complete
    input_hash2 = compute_snapshot_input_hash(portfolio2, valuation2)
    fields = {
        "oldest_price_date": valuation2.oldest_price_date,
        "newest_price_date": valuation2.newest_price_date,
        "base_currency": portfolio2.base_currency,
        "cash_balance": valuation2.cash_balance,
        "securities_value": valuation2.securities_value,
        "total_value": valuation2.total_value,
        "cost_basis": valuation2.cost_basis,
        "unrealized_gain": valuation2.unrealized_gain,
        "return_pct": valuation2.return_pct,
        "return_definition": valuation2.return_definition,
        "dividends_included": valuation2.dividends_included,
        "corporate_action_warnings": valuation2.corporate_action_warnings,
    }
    fields[field] = wrong_value
    snapshot2 = PortfolioSnapshot.objects.create(
        portfolio=portfolio2,
        as_of_date=TARGET_DATE,
        input_hash=input_hash2,
        code_revision=CODE_REVISION,
        **fields,
    )

    details = dict(portfolio_run.details)
    details["portfolios"] = 2
    details["snapshot_ids"] = {
        **details["snapshot_ids"],
        str(portfolio2.pk): str(snapshot2.pk),
    }
    JobRun.objects.filter(pk=portfolio_run.pk).update(details=details)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "portfolio_snapshot_totals_mismatch"


@pytest.mark.parametrize(
    "field, wrong_value",
    [
        ("cost_basis", Decimal("999999.000000")),
        ("market_value", Decimal("999999.000000")),
        ("unrealized_gain", Decimal("999999.000000")),
        ("corporate_action_suspected", True),
    ],
)
def test_portfolio_snapshot_holding_field_fabrication_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, wrong_value: object
) -> None:
    """A correct-input-hash snapshot whose single non-empty holding has one
    derived field independently fabricated must fail closed. Uses the same
    never-snapshotted second-portfolio fabrication as the snapshot-level
    test above, since holdings are DB-trigger immutable too.

    `corporate_action_suspected` is included here as the "no genuine prior
    history" case: `portfolio2` has no earlier `PortfolioSnapshotHolding`,
    so the true value is `False`; the verifier's `before_recorded_at`
    boundary (this snapshot's own `recorded_at`) also excludes this very
    row from counting as its own "previous" reference, so a fabricated
    `True` is still caught. See
    `test_portfolio_snapshot_two_generation_*` below for the case where a
    genuine earlier generation exists.
    """
    stages, config, portfolio1 = _build_portfolio_with_holdings_state(monkeypatch, tmp_path)
    portfolio_run = JobRun.objects.get(pk=stages["portfolio_snapshots"]["job_run_id"])

    portfolio2 = Portfolio.objects.create(
        owner=refresh_fixtures.create_portfolio_owner(username="fabricated-sibling-owner"),
        name="Fabricated Sibling",
        base_currency="USD",
    )
    aaa_listing = Listing.objects.get(ticker="AAA")
    upsert_holding(
        portfolio=portfolio2,
        listing=aaa_listing,
        quantity=Decimal("3"),
        average_cost=Decimal("50"),
    )
    # Reloaded so `cash_balance` carries its exact persisted Decimal scale --
    # matching the verifier's own fresh re-query -- rather than the
    # in-memory zero-argument default's differing string representation.
    portfolio2 = Portfolio.objects.get(pk=portfolio2.pk)
    valuation2 = calculate_portfolio_valuation(portfolio2, expected_as_of_date=TARGET_DATE)
    assert valuation2.complete
    input_hash2 = compute_snapshot_input_hash(portfolio2, valuation2)
    position = valuation2.positions[0]
    assert position.market_data is not None
    holding_fields = {
        "quantity": position.holding.quantity,
        "average_cost": position.holding.average_cost,
        "price": position.market_data.close,
        "cost_basis": position.cost_basis,
        "market_value": position.market_value,
        "unrealized_gain": position.unrealized_gain,
        "corporate_action_suspected": False,
    }
    holding_fields[field] = wrong_value
    snapshot2 = PortfolioSnapshot.objects.create(
        portfolio=portfolio2,
        as_of_date=TARGET_DATE,
        input_hash=input_hash2,
        code_revision=CODE_REVISION,
        oldest_price_date=valuation2.oldest_price_date,
        newest_price_date=valuation2.newest_price_date,
        base_currency=portfolio2.base_currency,
        cash_balance=valuation2.cash_balance,
        securities_value=valuation2.securities_value,
        total_value=valuation2.total_value,
        cost_basis=valuation2.cost_basis,
        unrealized_gain=valuation2.unrealized_gain,
        return_pct=valuation2.return_pct,
        return_definition=valuation2.return_definition,
        dividends_included=valuation2.dividends_included,
        corporate_action_warnings=valuation2.corporate_action_warnings,
    )
    PortfolioSnapshotHolding.objects.create(
        snapshot=snapshot2,
        listing=position.holding.listing,
        source_asset=position.market_data.source_asset,
        source_session_date=position.market_data.session_date,
        **holding_fields,
    )

    details = dict(portfolio_run.details)
    details["portfolios"] = 2
    details["snapshot_ids"] = {
        **details["snapshot_ids"],
        str(portfolio2.pk): str(snapshot2.pk),
    }
    JobRun.objects.filter(pk=portfolio_run.pk).update(details=details)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "portfolio_snapshot_holding_mismatch"


def _create_prior_generation_holding(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recorded_at: datetime,
    portfolio: Portfolio,
    listing: Listing,
    price: Decimal,
    quantity: Decimal,
) -> None:
    """Persist one genuine earlier `PortfolioSnapshot`+holding generation.

    `recorded_at` is `auto_now_add`, so `timezone.now()` (the exact
    function Django's `DateTimeField.pre_save` calls) is monkeypatched only
    for the duration of these two inserts to pin a deterministic, strictly
    earlier wall-clock timestamp -- avoiding any flakiness from real
    execution-order timing.
    """
    prior_source_asset = LatestMarketData.objects.get(listing=listing).source_asset
    monkeypatch.setattr("django.utils.timezone.now", lambda: recorded_at)
    prior_snapshot = PortfolioSnapshot.objects.create(
        portfolio=portfolio,
        as_of_date=TARGET_DATE - timedelta(days=1),
        input_hash="0" * 64,
        code_revision=CODE_REVISION,
        oldest_price_date=TARGET_DATE - timedelta(days=1),
        newest_price_date=TARGET_DATE - timedelta(days=1),
        base_currency=portfolio.base_currency,
        cash_balance=Decimal("0.000000"),
        securities_value=quantity * price,
        total_value=quantity * price,
        cost_basis=quantity * price,
        unrealized_gain=Decimal("0.000000"),
        return_pct=Decimal("0.00000000"),
        return_definition="time_weighted",
        dividends_included=False,
        corporate_action_warnings=0,
    )
    PortfolioSnapshotHolding.objects.create(
        snapshot=prior_snapshot,
        listing=listing,
        source_asset=prior_source_asset,
        source_session_date=TARGET_DATE - timedelta(days=1),
        quantity=quantity,
        average_cost=price,
        price=price,
        cost_basis=quantity * price,
        market_value=quantity * price,
        unrealized_gain=Decimal("0.000000"),
        corporate_action_suspected=False,
    )


@pytest.mark.parametrize(
    "prior_price, claimed_suspected, expect_pass",
    [
        # Genuine two-generation split: a real, strictly earlier holding at
        # a price far outside the split-ratio band, correctly claimed and
        # accepted.
        (Decimal("300.000000"), True, True),
        # Same genuine prior generation, but the current snapshot's flag is
        # fabricated (falsely denies a real split) -- must fail closed.
        (Decimal("300.000000"), False, False),
        # A prior generation whose price is within the normal band (no real
        # split), but the current snapshot fabricates `True` anyway -- must
        # fail closed even though a genuine prior row now exists.
        (Decimal("100.000000"), True, False),
    ],
)
def test_portfolio_snapshot_two_generation_corporate_action_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prior_price: Decimal,
    claimed_suspected: bool,
    expect_pass: bool,
) -> None:
    """`_corporate_action_suspected` must reproduce the flag from real
    *prior* history, not from the very snapshot being verified.

    Builds a genuine earlier generation (a real, strictly-earlier
    `PortfolioSnapshot`+holding row) for a never-live-snapshotted sibling
    portfolio, then persists the (second, under-verification) generation
    with a claimed `corporate_action_suspected`/`corporate_action_warnings`
    value. Only the claim that matches what `_corporate_action_suspected`
    would genuinely derive from that real prior row must verify; every
    fabricated claim -- whether over- or under-claiming -- must fail
    closed on `portfolio_snapshot_holding_mismatch`.
    """
    stages, config, portfolio1 = _build_portfolio_with_holdings_state(monkeypatch, tmp_path)
    portfolio_run = JobRun.objects.get(pk=stages["portfolio_snapshots"]["job_run_id"])

    portfolio2 = Portfolio.objects.create(
        owner=refresh_fixtures.create_portfolio_owner(username="two-gen-sibling-owner"),
        name="Two Generation Sibling",
        base_currency="USD",
    )
    aaa_listing = Listing.objects.get(ticker="AAA")
    upsert_holding(
        portfolio=portfolio2,
        listing=aaa_listing,
        quantity=Decimal("3"),
        average_cost=Decimal("50"),
    )
    portfolio2 = Portfolio.objects.get(pk=portfolio2.pk)

    _create_prior_generation_holding(
        monkeypatch,
        recorded_at=datetime(2026, 8, 1, tzinfo=UTC),
        portfolio=portfolio2,
        listing=aaa_listing,
        price=prior_price,
        quantity=Decimal("3"),
    )

    # Restore real `timezone.now()` before computing/persisting the
    # generation under verification, so its `recorded_at` is strictly later
    # than the prior generation's pinned instant.
    monkeypatch.setattr("django.utils.timezone.now", lambda: datetime(2026, 8, 2, tzinfo=UTC))
    valuation2 = calculate_portfolio_valuation(portfolio2, expected_as_of_date=TARGET_DATE)
    assert valuation2.complete
    position = valuation2.positions[0]
    assert position.market_data is not None
    genuine_suspected = _corporate_action_suspected(position)
    expected_match = genuine_suspected == claimed_suspected
    assert expected_match == expect_pass, (
        f"fixture design error: genuine={genuine_suspected}, claimed={claimed_suspected}"
    )

    holding_fields = {
        "quantity": position.holding.quantity,
        "average_cost": position.holding.average_cost,
        "price": position.market_data.close,
        "cost_basis": position.cost_basis,
        "market_value": position.market_value,
        "unrealized_gain": position.unrealized_gain,
        "corporate_action_suspected": claimed_suspected,
    }
    input_hash2 = compute_snapshot_input_hash(portfolio2, valuation2)
    snapshot2 = PortfolioSnapshot.objects.create(
        portfolio=portfolio2,
        as_of_date=TARGET_DATE,
        input_hash=input_hash2,
        code_revision=CODE_REVISION,
        oldest_price_date=valuation2.oldest_price_date,
        newest_price_date=valuation2.newest_price_date,
        base_currency=portfolio2.base_currency,
        cash_balance=valuation2.cash_balance,
        securities_value=valuation2.securities_value,
        total_value=valuation2.total_value,
        cost_basis=valuation2.cost_basis,
        unrealized_gain=valuation2.unrealized_gain,
        return_pct=valuation2.return_pct,
        return_definition=valuation2.return_definition,
        dividends_included=valuation2.dividends_included,
        corporate_action_warnings=1 if claimed_suspected else 0,
    )
    PortfolioSnapshotHolding.objects.create(
        snapshot=snapshot2,
        listing=position.holding.listing,
        source_asset=position.market_data.source_asset,
        source_session_date=position.market_data.session_date,
        **holding_fields,
    )

    details = dict(portfolio_run.details)
    details["portfolios"] = 2
    details["snapshot_ids"] = {
        **details["snapshot_ids"],
        str(portfolio2.pk): str(snapshot2.pk),
    }
    JobRun.objects.filter(pk=portfolio_run.pk).update(details=details)

    if expect_pass:
        result = verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
        assert result["status"] == "verified"
    else:
        with pytest.raises(RefreshVerificationError) as excinfo:
            verify_scheduled_refresh(
                target_date=TARGET_DATE,
                universe_config=config,
                code_revision=CODE_REVISION,
                stages=stages,
                sec_required=False,
            )
        assert excinfo.value.reason_code in (
            "portfolio_snapshot_holding_mismatch",
            "portfolio_snapshot_totals_mismatch",
        )


def test_portfolio_snapshot_fabricated_input_hash_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", CODE_REVISION)
    config = refresh_fixtures.build_universe_config(symbols=("AAA",), minimum_eligible=1)
    refresh_fixtures.enable_twelve_data_provider()
    refresh_fixtures.set_twelve_data_api_key(monkeypatch)
    refresh_fixtures.patch_twelve_data_provider(
        monkeypatch,
        config,
        target_date=TARGET_DATE,
        retrieved_at=DECISION_TIME,
    )
    monkeypatch.setattr("stanstock.data.live_us.timezone.now", lambda: DECISION_TIME)
    portfolio = Portfolio.objects.create(
        owner=refresh_fixtures.create_portfolio_owner(),
        name="Fabricated",
        base_currency="USD",
    )

    prepared = PreparedUsDailyJob(
        config=config,
        target_date=TARGET_DATE,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        decision_time=DECISION_TIME,
    )
    market_run = execute_us_daily_job(prepared, require_observed=True)
    evaluation_run = execute_prediction_evaluation_job(
        provider=twelve_data.PROVIDER,
        evaluation_date=TARGET_DATE,
        evaluation_time=DECISION_TIME,
        benchmark_subject=config.benchmark_symbol,
    )
    portfolio_run = execute_portfolio_snapshot_job(
        target_date=TARGET_DATE,
        require_session_date=True,
        require_all=True,
    )

    # A same-portfolio, same-date snapshot that exists as a real row but was
    # never actually derived from this portfolio's own current holdings --
    # its `input_hash` cannot be reproduced by an independent recomputation.
    fabricated = PortfolioSnapshot.objects.create(
        portfolio=portfolio,
        as_of_date=TARGET_DATE,
        base_currency="USD",
        cash_balance=Decimal("0"),
        securities_value=Decimal("0"),
        total_value=Decimal("0"),
        cost_basis=Decimal("0"),
        unrealized_gain=Decimal("0"),
        input_hash="f" * 64,
        code_revision=CODE_REVISION,
    )
    details = dict(portfolio_run.details)
    details["snapshot_ids"] = {str(portfolio.pk): str(fabricated.pk)}
    JobRun.objects.filter(pk=portfolio_run.pk).update(details=details)
    stages = {
        "market": _stage_entry(market_run),
        "evaluation": _stage_entry(evaluation_run),
        "portfolio_snapshots": _stage_entry(portfolio_run),
    }

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "portfolio_snapshot_input_hash_mismatch"


# --- R3 F-7: catalog assets are cutoff-bound and target-scoped; unrelated
# historical assets must not block a target refresh -------------------------


def test_corrupt_catalog_asset_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    catalog_asset_id = uuid.UUID(
        JobRun.objects.get(pk=stages["market"]["job_run_id"]).details["catalog_asset_ids"][0]
    )
    asset_path = tmp_path / DataAsset.objects.get(pk=catalog_asset_id).relative_path
    asset_path.write_bytes(b"corrupted")

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "asset_integrity_failed"
    assert str(tmp_path) not in str(excinfo.value)


def test_unrelated_historical_catalog_asset_does_not_block_target_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    market_run = JobRun.objects.get(pk=stages["market"]["job_run_id"])
    real_catalog_asset = DataAsset.objects.get(
        pk=uuid.UUID(market_run.details["catalog_asset_ids"][0])
    )
    # A real, but wholly unrelated and *corrupt*, historical catalog asset
    # from a different date. It is never referenced by this target's
    # verified evidence, so the scoped manifest must not touch it -- proved
    # by verification still succeeding despite its corruption.
    store = AssetStore(tmp_path)
    written = store.write_bytes("catalog/unrelated-2020.json", b"irrelevant catalog snapshot")
    unrelated = DataAsset.objects.create(
        provider=real_catalog_asset.provider,
        kind=real_catalog_asset.kind,
        subject="unrelated-history",
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=real_catalog_asset.retrieved_at - timedelta(days=400),
        available_at=real_catalog_asset.retrieved_at - timedelta(days=400),
    )
    (tmp_path / unrelated.relative_path).write_bytes(b"now corrupted, but irrelevant")

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )
    assert result["status"] == "verified"


def test_analysis_committed_etf_sync_failure_recovers_with_zero_additional_fetches_and_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The analysis-committed/ETF-sync-failed recovery path must supply
    real, exact catalog asset identities -- not an empty tuple -- and
    verify end to end without any additional provider credits.

    Mirrors `test_data_live_us.py::
    test_etf_sync_failure_preserves_analysis_for_zero_credit_recovery` but
    carries the recovered run all the way through
    `verify_scheduled_refresh`, proving the benchmark-evidence-recovered
    `catalog_asset_ids` satisfy the same membership config_hash replay a
    fully first-try-successful run does.
    """
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", CODE_REVISION)
    config = refresh_fixtures.build_universe_config(symbols=("AAA",), minimum_eligible=1)
    refresh_fixtures.enable_twelve_data_provider()
    refresh_fixtures.set_twelve_data_api_key(monkeypatch)
    catalog_calls, price_calls = refresh_fixtures.patch_twelve_data_provider(
        monkeypatch,
        config,
        target_date=TARGET_DATE,
        retrieved_at=DECISION_TIME,
    )
    monkeypatch.setattr("stanstock.data.live_us.timezone.now", lambda: DECISION_TIME)

    def _fail_sync(**kwargs: object) -> Listing:
        raise ValueError("simulated ETF identity conflict")

    monkeypatch.setattr("stanstock.data.live_us.sync_investable_spy_from_asset", _fail_sync)
    prepared = PreparedUsDailyJob(
        config=config,
        target_date=TARGET_DATE,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        decision_time=DECISION_TIME,
    )
    with pytest.raises(ValueError, match="simulated ETF identity conflict"):
        execute_us_daily_job(prepared, require_observed=True)

    assert len(catalog_calls) == 1
    assert len(price_calls) == 2  # AAA + the SPY benchmark

    monkeypatch.setattr(
        "stanstock.data.live_us.sync_investable_spy_from_asset", _real_sync_investable_spy
    )
    market_run = execute_us_daily_job(prepared, require_observed=True)

    # Recovery must not spend any additional provider credits/calls.
    assert len(catalog_calls) == 1
    assert len(price_calls) == 2
    assert JobRun.objects.get(pk=market_run.pk).details["catalog_asset_ids"]

    evaluation_run = execute_prediction_evaluation_job(
        provider=twelve_data.PROVIDER,
        evaluation_date=TARGET_DATE,
        evaluation_time=DECISION_TIME,
        benchmark_subject=config.benchmark_symbol,
    )
    portfolio_run = execute_portfolio_snapshot_job(
        target_date=TARGET_DATE,
        require_session_date=True,
        require_all=True,
    )
    stages = {
        "market": _stage_entry(market_run),
        "evaluation": _stage_entry(evaluation_run),
        "portfolio_snapshots": _stage_entry(portfolio_run),
    }

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )
    assert result["status"] == "verified"


# --- R3 F-9: SEC mapping timing is bound to the SEC stage's own execution,
# decoupled from an older/recovered analysis run's cutoff -------------------


def test_sec_mapping_retrieved_after_older_analysis_cutoff_still_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path, sec=True)
    sec_run = JobRun.objects.get(pk=stages["sec_fundamentals"]["job_run_id"])
    run = AnalysisRun.objects.get(target_date=TARGET_DATE)
    # Simulate the SEC stage genuinely completing *after* this (older,
    # already-recovered) analysis run's own `data_cutoff` -- e.g. the SEC
    # stage was newly enabled and ran later in the same scheduled refresh.
    later_sec_finish = run.data_cutoff + timedelta(hours=3)
    JobRun.objects.filter(pk=sec_run.pk).update(finished_at=later_sec_finish)
    sec_run.refresh_from_db()
    later_retrieved_at = run.data_cutoff + timedelta(hours=2)
    assert run.data_cutoff < later_retrieved_at < sec_run.finished_at

    store = AssetStore(tmp_path)
    written = store.write_bytes("sec/late-mapping.json", b'{"cik_lookup": {}}')
    late_mapping = DataAsset.objects.create(
        provider=sec.PROVIDER,
        kind=sec_ingestion.MAPPING_KIND,
        subject="company_tickers_exchange",
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=later_retrieved_at,
        available_at=later_retrieved_at,
    )
    details = dict(sec_run.details)
    details["mapping_asset_id"] = str(late_mapping.pk)
    details["mapping_sha256"] = late_mapping.sha256
    late_mapping_ref = asset_ref_for(late_mapping)
    details["asset_refs"] = [
        late_mapping_ref.to_json() if ref["kind"] == sec_ingestion.MAPPING_KIND else ref
        for ref in details["asset_refs"]
    ]
    JobRun.objects.filter(pk=sec_run.pk).update(details=details)

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=True,
    )
    assert result["status"] == "verified"


# --- R5 F1: on-time issuance and source-asset cutoff timing must be
# independently recomputed, not trusted from stored flags -------------------


def test_analysis_run_forged_on_time_flag_after_next_open_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from exchange_calendars import get_calendar

    stages, config = _build_verified_state(monkeypatch, tmp_path)
    run = AnalysisRun.objects.get(target_date=TARGET_DATE)
    calendar = get_calendar("XNYS")
    next_session = calendar.next_session(TARGET_DATE.isoformat())
    next_open = calendar.session_open(next_session).to_pydatetime()
    # `issued_on_time=True` is forged/left unchanged; only the run's own
    # immutable `generated_at` moves to at-or-after the next session's
    # open, which the stored flag alone cannot detect.
    AnalysisRun.objects.filter(pk=run.pk).update(generated_at=next_open, issued_on_time=True)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "analysis_run_not_on_time"


def test_analysis_run_immediately_before_next_open_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from exchange_calendars import get_calendar

    stages, config = _build_verified_state(monkeypatch, tmp_path)
    run = AnalysisRun.objects.get(target_date=TARGET_DATE)
    calendar = get_calendar("XNYS")
    next_session = calendar.next_session(TARGET_DATE.isoformat())
    next_open = calendar.session_open(next_session).to_pydatetime()
    boundary = next_open - timedelta(seconds=1)
    AnalysisRun.objects.filter(pk=run.pk).update(
        generated_at=boundary, data_cutoff=boundary, issued_on_time=True
    )

    result = verify_scheduled_refresh(
        target_date=TARGET_DATE,
        universe_config=config,
        code_revision=CODE_REVISION,
        stages=stages,
        sec_required=False,
    )
    assert result["status"] == "verified"


def test_prediction_forged_on_time_flag_after_next_open_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from exchange_calendars import get_calendar

    stages, config = _build_verified_state(monkeypatch, tmp_path)
    template = Prediction.objects.filter(evidence_role=Prediction.EvidenceRole.DECISION).first()
    assert template is not None
    calendar = get_calendar("XNYS")
    next_session = calendar.next_session(TARGET_DATE.isoformat())
    next_open = calendar.session_open(next_session).to_pydatetime()
    _clone_prediction(
        template,
        model_version=f"{template.model_version}-rogue",
        generated_at=next_open,
        issued_on_time=True,
    )
    _bump_prediction_count(stages, 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "prediction_not_on_time"


def test_prediction_declared_source_asset_future_vintage_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A declared source asset admitted after its own row's cutoff fails,
    independent of the run's own (unaffected) `data_cutoff`."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    template = Prediction.objects.filter(evidence_role=Prediction.EvidenceRole.DECISION).first()
    assert template is not None
    store = AssetStore(tmp_path)
    written = store.write_bytes("future/rogue-asset.bin", b"a future-vintage asset")
    future_time = template.data_cutoff + timedelta(days=1)
    future_asset = DataAsset.objects.create(
        provider=twelve_data.PROVIDER,
        kind="price_history",
        subject="ROGUE",
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=future_time,
        available_at=future_time,
    )
    future_entry = {
        "id": str(future_asset.pk),
        "provider": future_asset.provider,
        "kind": future_asset.kind,
        "subject": future_asset.subject,
        "sha256": future_asset.sha256,
    }
    _clone_prediction(
        template,
        model_version=f"{template.model_version}-rogue",
        source_assets=[*template.source_assets, future_entry],
    )
    _bump_prediction_count(stages, 1)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_scheduled_refresh(
            target_date=TARGET_DATE,
            universe_config=config,
            code_revision=CODE_REVISION,
            stages=stages,
            sec_required=False,
        )
    assert excinfo.value.reason_code == "source_assets_entry_after_cutoff"

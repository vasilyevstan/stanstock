from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from io import BytesIO, StringIO
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import polars as pl
import pytest
from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection, connections
from django.urls import reverse

import refresh_fixtures
import test_data_sec_ingestion as sec_fixtures
from stanstock.core import refresh_verification as refresh_verification_module
from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.management.commands import scheduled_refresh
from stanstock.core.models import JobRun
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data import jobs as data_jobs_module
from stanstock.data import live_us as live_us_module
from stanstock.data import sec_jobs as sec_jobs_module
from stanstock.data.assets import AssetStore
from stanstock.data.jobs import PreparedUsDailyJob, execute_us_daily_job
from stanstock.data.live_us import UsUniverseConfig
from stanstock.data.management.commands import daily as daily_command
from stanstock.data.models import (
    DataAsset,
    LatestMarketData,
    ProviderRecord,
    Universe,
    UniverseSnapshot,
)
from stanstock.data.providers import sec as sec_provider
from stanstock.data.providers.contracts import FundamentalSourcePayload, PriceBar, PriceSeries
from stanstock.data.sec_config import load_sec_fundamentals_config
from stanstock.data.sec_evidence import MAPPING_SUBJECT
from stanstock.portfolio.models import (
    Portfolio,
    PortfolioHolding,
    PortfolioSnapshotHolding,
)
from stanstock.portfolio.refresh_validation import (
    PORTFOLIO_VERIFICATION_KIND,
    verify_portfolio_snapshot_stage,
)
from stanstock.research.jobs import execute_prediction_evaluation_job
from stanstock.research.models import AnalysisRun, Prediction, PredictionOutcome, StockAnalysis
from stanstock.research.refresh_evidence import ANALYSIS_OUTPUT_MANIFEST_KIND
from test_refresh_verification import _create_lagging_prediction, _register_lagging_price_history

pytestmark = pytest.mark.django_db

TARGET_DATE = date(2026, 9, 4)
DECISION_TIME = datetime(2026, 9, 5, 6, tzinfo=UTC)


def _prepared(target: date = TARGET_DATE) -> PreparedUsDailyJob:
    config = cast(
        UsUniverseConfig,
        SimpleNamespace(benchmark_symbol="SPY"),
    )
    return PreparedUsDailyJob(
        config=config,
        target_date=target,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        decision_time=DECISION_TIME,
    )


def _successful_child(job_name: str, region: str, target_date: date) -> JobRun:
    return execute_target_job(
        job_name=job_name,
        region=region,
        target_date=target_date,
        task=lambda run: JobExecutionResult(),
    )


def _real_prepared(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    symbols: tuple[str, ...] = ("AAA", "BBB"),
    minimum_eligible: int = 1,
    target: date = TARGET_DATE,
    decision_time: datetime = DECISION_TIME,
) -> tuple[PreparedUsDailyJob, UsUniverseConfig, tuple[list[str], list[str]]]:
    """Build a `PreparedUsDailyJob` backed by a genuinely-executable pipeline.

    Unlike `_prepared`, `config` here is a real `UsUniverseConfig` (not a
    bare `SimpleNamespace`) and the Twelve Data provider boundary is patched
    with synthetic fixtures, so calling the real, unmocked
    `execute_us_daily_job` produces genuine persisted evidence that
    `refresh_verification.verify_scheduled_refresh` can independently prove.
    """
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    config = refresh_fixtures.build_universe_config(
        symbols=symbols,
        minimum_eligible=minimum_eligible,
    )
    refresh_fixtures.enable_twelve_data_provider()
    refresh_fixtures.set_twelve_data_api_key(monkeypatch)
    calls = refresh_fixtures.patch_twelve_data_provider(
        monkeypatch,
        config,
        target_date=target,
        retrieved_at=decision_time,
    )
    # `execute_us_daily_job` always runs with `require_on_time=True`, whose
    # deadline check folds in the *real* wall clock (`timezone.now()`) --
    # not just the supplied `decision_time` -- so it must be pinned to a
    # moment inside the on-time window too (same technique
    # `test_data_live_us.py::test_automatic_run_uses_current_clock_for_issuance_deadline`
    # uses), or a real test run's actual current date would always appear
    # "late" for a fixed historical `target`/`decision_time` fixture.
    monkeypatch.setattr("stanstock.data.live_us.timezone.now", lambda: decision_time)
    prepared = PreparedUsDailyJob(
        config=config,
        target_date=target,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        decision_time=decision_time,
    )
    return prepared, config, calls


def test_scheduled_refresh_retry_and_no_op_render_persisted_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    authenticated_client,
) -> None:
    real_timezone_now = live_us_module.timezone.now
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(monkeypatch, tmp_path)
    clean_checks: list[Path] = []
    evaluation_attempts = 0
    evaluation_times: list[datetime] = []

    def research_identity_sets() -> dict[str, frozenset[UUID]]:
        return {
            "analysis_runs": frozenset(AnalysisRun.objects.values_list("id", flat=True)),
            "stock_analyses": frozenset(StockAnalysis.objects.values_list("id", flat=True)),
            "predictions": frozenset(Prediction.objects.values_list("id", flat=True)),
        }

    def asset_identities() -> frozenset[tuple[UUID, str]]:
        return frozenset(DataAsset.objects.values_list("id", "sha256"))

    def identity_manifest(
        identities: frozenset[tuple[UUID, str]],
    ) -> dict[str, object]:
        ordered = sorted((str(asset_id), sha256) for asset_id, sha256 in identities)
        canonical = "\n".join(f"{asset_id}:{sha256}" for asset_id, sha256 in ordered)
        return {
            "count": len(ordered),
            "hash": hashlib.sha256(canonical.encode()).hexdigest(),
        }

    def assert_scheduled_refresh_card(response: Any, run: JobRun) -> None:
        content = " ".join(response.content.decode().split())
        section_start = content.index('<section aria-labelledby="jobs-title">')
        section_end = content.index("</section>", section_start)
        job_section = content[section_start:section_end]
        cards = re.findall(r"<article>.*?</article>", job_section)
        matching = [
            card
            for card in cards
            if "<strong>Scheduled Refresh · US</strong>" in card
            and f"attempt {run.attempt}</small>" in card
        ]
        assert len(matching) == 1
        card = matching[0]
        assert f'class="badge badge-{run.status}"' in card
        assert f">{run.get_status_display()}</span>" in card

    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(
        scheduled_refresh,
        "prepare_us_daily_job",
        lambda **kwargs: prepared,
    )

    def clean_revision(root: Path) -> str:
        clean_checks.append(root)
        return "a" * 40

    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", clean_revision)
    # Market and portfolio stages are deliberately left unmocked here so the
    # real pipeline runs against the patched Twelve Data fixtures above,
    # producing genuine persisted evidence `verify_scheduled_refresh` can
    # independently re-derive -- not a hand-built `JobExecutionResult` that
    # merely has the right shape.

    def execute_evaluation(**kwargs: object) -> JobRun:
        nonlocal evaluation_attempts
        evaluation_attempts += 1
        evaluation_time = kwargs["evaluation_time"]
        assert isinstance(evaluation_time, datetime)
        evaluation_times.append(evaluation_time)
        if evaluation_attempts == 1:

            def task(run: JobRun) -> JobExecutionResult:
                raise ValueError("temporary evaluation failure")

            return execute_target_job(
                job_name="evaluate_predictions",
                region="us",
                target_date=prepared.target_date,
                task=task,
            )
        return execute_prediction_evaluation_job(**kwargs)

    monkeypatch.setattr(
        scheduled_refresh,
        "execute_prediction_evaluation_job",
        execute_evaluation,
    )

    with pytest.raises(CommandError, match="temporary evaluation failure"):
        call_command(
            "scheduled_refresh",
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )

    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    first_market_stage = first_parent.details["stages"]["market"]
    first_market_child = JobRun.objects.get(pk=first_market_stage["job_run_id"])
    first_market_identity = (
        first_market_child.pk,
        first_market_child.job_name,
        first_market_child.region,
        first_market_child.target_date,
        first_market_child.attempt,
        first_market_child.status,
    )
    assert first_market_identity == (
        first_market_child.pk,
        "daily",
        "us",
        prepared.target_date,
        1,
        JobRun.Status.SUCCESS,
    )
    first_analysis_run_id = UUID(first_market_child.details["analysis_run_id"])
    first_snapshot_id = UUID(first_market_child.details["snapshot_id"])
    first_analysis_run = AnalysisRun.objects.get(pk=first_analysis_run_id)
    first_snapshot = UniverseSnapshot.objects.get(pk=first_snapshot_id)
    assert first_analysis_run.universe_snapshot_id == first_snapshot.pk
    first_research_identities = research_identity_sets()
    assert first_research_identities["analysis_runs"] == frozenset({first_analysis_run_id})
    assert first_research_identities["stock_analyses"] == frozenset(
        StockAnalysis.objects.filter(run_id=first_analysis_run_id).values_list("id", flat=True)
    )
    assert first_research_identities["predictions"] == frozenset(
        Prediction.objects.filter(analysis__run_id=first_analysis_run_id).values_list(
            "id", flat=True
        )
    )
    first_market_asset_identities = frozenset(
        DataAsset.objects.filter(provider=live_us_module.PROVIDER).values_list("id", "sha256")
    )
    assert first_market_asset_identities
    first_analysis_manifest = DataAsset.objects.get(
        provider="stanstock",
        kind=ANALYSIS_OUTPUT_MANIFEST_KIND,
        subject=str(first_analysis_run_id),
    )
    first_analysis_manifest_identity = (
        first_analysis_manifest.pk,
        first_analysis_manifest.sha256,
    )
    first_asset_identities = asset_identities()
    assert first_analysis_manifest_identity in first_asset_identities
    first_expected_asset_manifest = identity_manifest(first_asset_identities)
    first_outcome_identities = frozenset(
        PredictionOutcome.objects.values_list("prediction_id", flat=True)
    )

    assert first_parent.status == JobRun.Status.FAILED
    assert first_parent.details["stages"]["market"]["status"] == JobRun.Status.SUCCESS
    assert first_parent.details["stages"]["evaluation"]["status"] == JobRun.Status.FAILED
    # Zero active portfolios is an explicit satisfied skip, not a failure.
    assert first_parent.details["stages"]["portfolio_snapshots"]["status"] == JobRun.Status.SKIPPED
    assert "verification" not in first_parent.details
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]
    first_status = authenticated_client.get(reverse("status"))
    assert first_status.status_code == 200
    assert first_parent.pk in {run.pk for run in first_status.context["recent_jobs"]}
    assert_scheduled_refresh_card(first_status, first_parent)

    def forbidden_market_access(*_args: object, **_kwargs: object) -> str:
        pytest.fail(
            "failed-parent recovery must not resolve credentials or cross a provider fetch boundary"
        )

    monkeypatch.setattr(live_us_module.twelve_data, "resolve_api_key", forbidden_market_access)
    monkeypatch.setattr(
        live_us_module.twelve_data,
        "fetch_stock_catalog",
        forbidden_market_access,
    )
    monkeypatch.setattr(
        live_us_module.twelve_data,
        "fetch_daily_price_series",
        forbidden_market_access,
    )
    second_attempt_time = prepared.decision_time + timedelta(minutes=1)
    monkeypatch.setattr(live_us_module.timezone, "now", lambda: second_attempt_time)
    call_command(
        "scheduled_refresh",
        config=tmp_path / "universe.yml",
        stdout=StringIO(),
    )

    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=2)
    assert second_parent.started_at > first_parent.started_at
    second_market_child = JobRun.objects.get(
        pk=second_parent.details["stages"]["market"]["job_run_id"]
    )
    assert second_parent.status == JobRun.Status.SUCCESS
    assert second_market_child.status == JobRun.Status.SKIPPED
    assert second_market_child.details == {
        "reason": "target_already_succeeded",
        "successful_run_id": str(first_market_child.pk),
    }
    assert second_parent.details["stages"]["market"] == {
        "job_run_id": str(second_market_child.pk),
        "status": JobRun.Status.SKIPPED,
        "attempt": second_market_child.attempt,
        "error": "",
    }
    assert second_parent.details["stages"]["evaluation"]["status"] == JobRun.Status.SUCCESS
    assert second_parent.details["stages"]["portfolio_snapshots"]["status"] == JobRun.Status.SKIPPED
    assert second_parent.details["verification"]["status"] == "verified"
    verification = second_parent.details["verification"]
    assert verification["snapshot_id"] == str(first_snapshot_id)
    assert verification["analysis_run_id"] == str(first_analysis_run_id)
    assert verification["stock_analysis_count"] == len(first_research_identities["stock_analyses"])
    assert verification["prediction_count"] == len(first_research_identities["predictions"])
    assert verification["asset_manifest"] == first_expected_asset_manifest
    assert verification["child_job_run_ids"]["market"] == str(first_market_child.pk)
    assert research_identity_sets() == first_research_identities
    assert asset_identities() == first_asset_identities
    assert (
        frozenset(
            DataAsset.objects.filter(provider=live_us_module.PROVIDER).values_list("id", "sha256")
        )
        == first_market_asset_identities
    )
    recovered_analysis_manifest = DataAsset.objects.get(
        provider="stanstock",
        kind=ANALYSIS_OUTPUT_MANIFEST_KIND,
        subject=str(first_analysis_run_id),
    )
    assert (
        recovered_analysis_manifest.pk,
        recovered_analysis_manifest.sha256,
    ) == first_analysis_manifest_identity
    second_evaluation_child = JobRun.objects.get(
        pk=second_parent.details["stages"]["evaluation"]["job_run_id"]
    )
    evaluated_prediction_ids = {
        UUID(raw_id) for raw_id in second_evaluation_child.details["evaluated_prediction_ids"]
    }
    recovered_outcome_identities = frozenset(
        PredictionOutcome.objects.values_list("prediction_id", flat=True)
    )
    assert first_outcome_identities <= recovered_outcome_identities
    assert (recovered_outcome_identities - first_outcome_identities) <= evaluated_prediction_ids
    # Retry/no-op: the market child is recovered by reference to its prior
    # success, never re-fetched -- zero additional provider calls/credits.
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]
    assert len(clean_checks) == 2
    assert len(evaluation_times) == 2
    # The global clock remains inside the market stage's on-time window while
    # advancing enough to give the recovered parent deterministic UI ordering.
    assert evaluation_times == [prepared.decision_time, second_attempt_time]
    assert os.environ["STANSTOCK_CODE_REVISION"] == "a" * 40

    recovered_status = authenticated_client.get(reverse("status"))
    assert recovered_status.status_code == 200
    assert second_parent.pk in {run.pk for run in recovered_status.context["recent_jobs"]}
    recovered_content = " ".join(recovered_status.content.decode().split())
    assert '<section aria-labelledby="jobs-title">' in recovered_content
    assert '<h2 id="jobs-title">Recent target-date jobs</h2>' in recovered_content
    assert_scheduled_refresh_card(recovered_status, second_parent)

    verified_identity = {
        "snapshot_id": second_parent.details["verification"]["snapshot_id"],
        "analysis_run_id": second_parent.details["verification"]["analysis_run_id"],
        "asset_manifest": second_parent.details["verification"]["asset_manifest"],
    }
    persisted_research_identities = research_identity_sets()
    persisted_asset_identities = asset_identities()
    persisted_outcome_identities = frozenset(
        PredictionOutcome.objects.values_list("prediction_id", flat=True)
    )
    persisted_job_ids = frozenset(JobRun.objects.values_list("id", flat=True))

    def forbidden_parent_no_op_work(*_args: object, **_kwargs: object) -> str:
        pytest.fail("successful parent no-op must not execute revisions, children, or verification")

    monkeypatch.setattr(
        scheduled_refresh,
        "clean_git_revision",
        forbidden_parent_no_op_work,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "execute_us_daily_job",
        forbidden_parent_no_op_work,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "execute_prediction_evaluation_job",
        forbidden_parent_no_op_work,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "execute_portfolio_snapshot_job",
        forbidden_parent_no_op_work,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "verify_scheduled_refresh",
        forbidden_parent_no_op_work,
    )
    # Restore the wall clock after the on-time historical fixture is complete,
    # so the UI's newest-job ordering is deterministic for the no-op attempt.
    monkeypatch.setattr(live_us_module.timezone, "now", real_timezone_now)
    call_command(
        "scheduled_refresh",
        config=tmp_path / "universe.yml",
        stdout=StringIO(),
    )

    no_op_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=3)
    assert no_op_parent.status == JobRun.Status.SKIPPED
    assert no_op_parent.details == {
        "reason": "target_already_succeeded",
        "successful_run_id": str(second_parent.pk),
    }
    assert research_identity_sets() == persisted_research_identities
    assert asset_identities() == persisted_asset_identities
    assert (
        frozenset(PredictionOutcome.objects.values_list("prediction_id", flat=True))
        == persisted_outcome_identities
    )
    assert frozenset(JobRun.objects.values_list("id", flat=True)) == (
        persisted_job_ids | {no_op_parent.pk}
    )
    second_parent.refresh_from_db()
    assert {
        "snapshot_id": second_parent.details["verification"]["snapshot_id"],
        "analysis_run_id": second_parent.details["verification"]["analysis_run_id"],
        "asset_manifest": second_parent.details["verification"]["asset_manifest"],
    } == verified_identity
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]
    assert len(clean_checks) == 2

    no_op_status = authenticated_client.get(reverse("status"))
    assert no_op_status.status_code == 200
    assert no_op_status.context["recent_jobs"][0].pk == no_op_parent.pk
    assert_scheduled_refresh_card(no_op_status, no_op_parent)


@pytest.mark.parametrize(
    "append_position",
    [False, True],
    ids=["unchanged-proof", "position-appended"],
)
def test_nonempty_portfolio_proof_survives_parent_retry_without_refetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    append_position: bool,
) -> None:
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(
        monkeypatch,
        tmp_path,
        symbols=("AAA",),
    )
    listing = refresh_fixtures.pre_create_stock_listing("AAA")
    portfolio = Portfolio.objects.create(
        owner=refresh_fixtures.create_portfolio_owner(username="command-proof-owner"),
        name="Command proof portfolio",
        base_currency="USD",
    )
    # The real market child creates the LatestMarketData row before the
    # portfolio child runs. Direct construction here avoids inventing a
    # pre-refresh mutable quote merely to satisfy the web-service add guard.
    PortfolioHolding.objects.create(
        portfolio=portfolio,
        listing=listing,
        quantity=Decimal("2"),
        average_cost=Decimal("50"),
    )
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(
        scheduled_refresh,
        "prepare_us_daily_job",
        lambda **kwargs: prepared,
    )
    original_revision = "a" * 40
    monkeypatch.setattr(
        scheduled_refresh,
        "clean_git_revision",
        lambda root: original_revision,
    )
    real_verify = scheduled_refresh.verify_scheduled_refresh
    verification_attempts = 0

    def fail_parent_once(**kwargs: Any) -> dict[str, Any]:
        nonlocal verification_attempts
        verification_attempts += 1
        if verification_attempts == 1:
            raise RefreshVerificationError(
                "forced_post_child_failure",
                "Synthetic parent verification failure",
            )
        return real_verify(**kwargs)

    monkeypatch.setattr(
        scheduled_refresh,
        "verify_scheduled_refresh",
        fail_parent_once,
    )
    with pytest.raises(CommandError, match="Synthetic parent verification failure"):
        call_command(
            "scheduled_refresh",
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )

    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    portfolio_child = JobRun.objects.get(
        pk=first_parent.details["stages"]["portfolio_snapshots"]["job_run_id"]
    )
    assert portfolio_child.status == JobRun.Status.SUCCESS
    proof_asset = DataAsset.objects.get(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(portfolio_child.pk),
    )
    assert DataAsset.objects.filter(kind=PORTFOLIO_VERIFICATION_KIND).count() == 1

    # Portfolio-proof acceptance is contract-versioned, not retry-HEAD
    # versioned. A later process revision cannot reject the recognized old
    # child proof after its complete immutable replay succeeds.
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", "b" * 40)
    portfolio_result = verify_portfolio_snapshot_stage(
        portfolio_child,
        target_date=TARGET_DATE,
    )

    def no_credentials(*args: object, **kwargs: object) -> str:
        pytest.fail("retry must not resolve provider credentials")

    monkeypatch.setattr(
        "stanstock.data.providers.twelve_data.resolve_api_key",
        no_credentials,
    )
    appended: PortfolioSnapshotHolding | None = None
    if append_position:
        snapshot_id = portfolio_child.details["snapshot_ids"][str(portfolio.pk)]
        spy_market = LatestMarketData.objects.get(listing__ticker="SPY")
        appended = PortfolioSnapshotHolding.objects.create(
            snapshot_id=snapshot_id,
            listing_id=spy_market.listing_id,
            source_asset_id=spy_market.source_asset_id,
            source_session_date=spy_market.session_date,
            quantity=Decimal("1"),
            average_cost=spy_market.close,
            price=spy_market.close,
            cost_basis=spy_market.close,
            market_value=spy_market.close,
            unrealized_gain=Decimal("0"),
            corporate_action_suspected=False,
        )
    captured_manifests: list[set[UUID]] = []
    real_manifest = refresh_verification_module._verify_asset_evidence

    def capture_manifest(asset_ids: set[UUID]) -> dict[str, Any]:
        captured_manifests.append(set(asset_ids))
        return real_manifest(asset_ids)

    monkeypatch.setattr(
        refresh_verification_module,
        "_verify_asset_evidence",
        capture_manifest,
    )
    if append_position:
        with pytest.raises(CommandError):
            call_command(
                "scheduled_refresh",
                config=tmp_path / "universe.yml",
                stdout=StringIO(),
            )
    else:
        call_command(
            "scheduled_refresh",
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )

    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=2)
    assert second_parent.status == (
        JobRun.Status.FAILED if append_position else JobRun.Status.SUCCESS
    )
    assert second_parent.details["stages"]["market"]["status"] == JobRun.Status.SKIPPED
    assert second_parent.details["stages"]["portfolio_snapshots"]["status"] == JobRun.Status.SKIPPED
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "SPY"]
    assert DataAsset.objects.filter(kind=PORTFOLIO_VERIFICATION_KIND).count() == 1
    if append_position:
        assert appended is not None
        assert PortfolioSnapshotHolding.objects.filter(pk=appended.pk).exists()
        assert captured_manifests == []
        return

    assert captured_manifests
    assert {ref.id for ref in portfolio_result.asset_refs} <= captured_manifests[-1]
    assert proof_asset.id in captured_manifests[-1]

    proof_path = AssetStore(tmp_path).resolve(proof_asset.relative_path)
    original = proof_path.read_bytes()
    proof_path.write_bytes(original + b"tampered")
    with pytest.raises(RefreshVerificationError):
        verify_portfolio_snapshot_stage(portfolio_child, target_date=TARGET_DATE)
    proof_path.write_bytes(original)
    verify_portfolio_snapshot_stage(portfolio_child, target_date=TARGET_DATE)


def test_scheduled_refresh_rejects_changed_machine_timezone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("STANSTOCK_SCHEDULE_TIMEZONE", "America/New_York")
    monkeypatch.setattr(
        scheduled_refresh,
        "detect_iana_timezone",
        lambda: "America/Los_Angeles",
    )

    with pytest.raises(CommandError, match="Reinstall the LaunchAgent"):
        call_command(
            "scheduled_refresh",
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )

    assert JobRun.objects.count() == 0


def test_automatic_daily_child_refuses_late_research_grade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = PreparedUsDailyJob(
        config=cast(UsUniverseConfig, SimpleNamespace(benchmark_symbol="SPY")),
        target_date=date(2026, 9, 4),
        snapshot_grade=UniverseSnapshot.Grade.RESEARCH,
        decision_time=datetime(2026, 9, 8, 15, tzinfo=UTC),
    )
    monkeypatch.setattr(
        "stanstock.data.jobs.run_us_daily",
        lambda **kwargs: pytest.fail("provider work must not start for a late run"),
    )

    with pytest.raises(ValueError, match="Automatic research-grade catch-up is forbidden"):
        execute_us_daily_job(prepared, require_observed=True)

    failed = JobRun.objects.get(job_name="daily")
    assert failed.status == JobRun.Status.FAILED


def test_enabled_sec_stage_runs_before_market(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    listing = refresh_fixtures.pre_create_stock_listing("AAA")
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(
        monkeypatch,
        tmp_path,
        symbols=("AAA",),
        minimum_eligible=1,
    )
    order: list[str] = []
    ProviderRecord.objects.create(provider="sec", enabled=True, status="ok")
    sec_evidence = refresh_fixtures.build_sec_evidence(
        store=AssetStore(tmp_path),
        company=listing.security.company,
        available_before=prepared.decision_time,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(
        scheduled_refresh,
        "prepare_us_daily_job",
        lambda **kwargs: prepared,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "clean_git_revision",
        lambda root: "a" * 40,
    )

    def execute_sec(**kwargs: object) -> JobRun:
        order.append("sec")

        def task(run: JobRun) -> JobExecutionResult:
            return JobExecutionResult(
                details={
                    "mapping_asset_id": sec_evidence.mapping_asset_id,
                    "mapping_sha256": sec_evidence.mapping_sha256,
                    "cik_config_version": sec_evidence.cik_config_version,
                    "cik_config_hash": sec_evidence.cik_config_hash,
                    "config_version": sec_evidence.fundamentals_config_version,
                    "config_hash": sec_evidence.fundamentals_config_hash,
                    "asset_refs": [ref.to_json() for ref in sec_evidence.asset_refs],
                }
            )

        return execute_target_job(
            job_name="sec_fundamentals",
            region="us",
            target_date=prepared.target_date,
            task=task,
        )

    def execute_market(*args: object, **kwargs: object) -> JobRun:
        order.append("market")
        # Delegate to the real, unmocked production entrypoint so the market
        # stage produces genuine persisted evidence -- ordering is captured
        # here, evidence realism is not sacrificed for it.
        return execute_us_daily_job(
            prepared,
            require_observed=True,
            long_forecast_requested=cast(bool, kwargs["long_forecast_requested"]),
            target_gate_reservation_id=cast(UUID, kwargs["target_gate_reservation_id"]),
        )

    monkeypatch.setattr(scheduled_refresh, "execute_sec_fundamentals_job", execute_sec)
    monkeypatch.setattr(scheduled_refresh, "execute_us_daily_job", execute_market)

    call_command(
        "scheduled_refresh",
        config=tmp_path / "universe.yml",
        stdout=StringIO(),
    )

    assert order == ["sec", "market"]
    parent = JobRun.objects.get(job_name="scheduled_refresh")
    assert parent.status == JobRun.Status.SUCCESS
    assert parent.details["stages"]["sec_fundamentals"]["status"] == JobRun.Status.SUCCESS
    assert parent.details["verification"]["status"] == "verified"
    assert parent.details["verification"]["sec"]["required"] is True
    assert parent.details["verification"]["sec"]["mapping_asset_id"] == (
        sec_evidence.mapping_asset_id
    )


def test_failed_sec_stage_blocks_market(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    ProviderRecord.objects.create(provider="sec", enabled=True, status="ok")
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(
        scheduled_refresh,
        "prepare_us_daily_job",
        lambda **kwargs: prepared,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "clean_git_revision",
        lambda root: "a" * 40,
    )

    def fail_sec(**kwargs: object) -> JobRun:
        def task(run: JobRun) -> JobExecutionResult:
            raise ValueError("SEC unavailable")

        return execute_target_job(
            job_name="sec_fundamentals",
            region="us",
            target_date=prepared.target_date,
            task=task,
        )

    monkeypatch.setattr(
        scheduled_refresh,
        "execute_sec_fundamentals_job",
        fail_sec,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "execute_us_daily_job",
        lambda *args, **kwargs: pytest.fail("market stage must remain blocked"),
    )

    with pytest.raises(CommandError, match="SEC unavailable"):
        call_command(
            "scheduled_refresh",
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )

    parent = JobRun.objects.get(job_name="scheduled_refresh")
    assert parent.details["stages"]["sec_fundamentals"]["status"] == JobRun.Status.FAILED
    assert "market" not in parent.details["stages"]


@pytest.mark.parametrize(
    ("scheduler_gate", "authoritative_gate"),
    [(False, True), (True, False)],
    ids=["scheduler-false-market-true", "scheduler-true-market-false"],
)
def test_scheduler_rejects_opposite_gate_success_inserted_before_market_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scheduler_gate: bool,
    authoritative_gate: bool,
) -> None:
    """The locked reservation rejects a success racing the scheduler's
    read-only proposal before SEC task entry or parent verification."""
    prepared = _prepared()
    ProviderRecord.objects.create(
        provider="sec",
        enabled=scheduler_gate,
        status="ok" if scheduler_gate else "disabled",
    )
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)

    def propose_then_insert_success(candidate: PreparedUsDailyJob) -> bool:
        assert candidate is prepared
        JobRun.objects.create(
            job_name="daily",
            region="us",
            target_date=candidate.target_date,
            attempt=1,
            status=JobRun.Status.SUCCESS,
            finished_at=candidate.decision_time,
            details={"long_forecast_requested": authoritative_gate},
        )
        return scheduler_gate

    monkeypatch.setattr(
        scheduled_refresh,
        "proposed_us_daily_target_gate",
        propose_then_insert_success,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "execute_sec_fundamentals_job",
        lambda **kwargs: pytest.fail("gate mismatch must block SEC task entry"),
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "execute_us_daily_job",
        lambda **kwargs: pytest.fail("gate mismatch must block market production"),
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "verify_scheduled_refresh",
        lambda **kwargs: pytest.fail("gate mismatch must block parent verification"),
    )

    with pytest.raises(CommandError, match="explicit long-forecast invocation gate conflicts"):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())

    parent = JobRun.objects.get(job_name="scheduled_refresh")
    authoritative = JobRun.objects.get(job_name="daily")
    assert parent.status == JobRun.Status.FAILED
    assert "verification" not in parent.details
    assert "evaluation" not in parent.details.get("stages", {})
    assert authoritative.status == JobRun.Status.SUCCESS
    assert authoritative.details["long_forecast_requested"] is authoritative_gate
    assert JobRun.objects.filter(job_name="daily").count() == 1
    assert "sec_fundamentals" not in parent.details.get("stages", {})


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL daily-target advisory-lock regression",
)
@pytest.mark.django_db(transaction=True)
def test_postgresql_opposite_daily_completion_wins_before_scheduler_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A false manual completion linearized after a true scheduler proposal
    is detected under the daily lock before SEC task/budget/fetch entry."""
    prepared = _prepared()
    universe = Universe.objects.create(
        slug="gate-race",
        name="Gate race",
        config_version="test-v1",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=prepared.target_date,
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="a" * 64,
    )
    ProviderRecord.objects.create(provider="sec", enabled=True, status="ok")
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)

    real_proposal = scheduled_refresh.proposed_us_daily_target_gate
    barrier = Barrier(2)

    def paused_true_proposal(candidate: PreparedUsDailyJob) -> bool:
        proposed = real_proposal(candidate)
        assert proposed is True
        barrier.wait(timeout=10)
        barrier.wait(timeout=10)
        return proposed

    monkeypatch.setattr(
        scheduled_refresh,
        "proposed_us_daily_target_gate",
        paused_true_proposal,
    )
    provider_entries: list[str] = []

    def no_sec_entry(**kwargs: object) -> JobRun:
        provider_entries.append("sec")
        pytest.fail("opposite-gate mismatch must block SEC task entry")

    monkeypatch.setattr(
        scheduled_refresh,
        "execute_sec_fundamentals_job",
        no_sec_entry,
    )
    monkeypatch.setattr(
        data_jobs_module,
        "run_us_daily",
        lambda **kwargs: live_us_module.LiveUsRunResult(
            snapshot=snapshot,
            analysis_run_id=uuid4(),
            analyses=1,
            predictions=1,
            eligible=1,
            excluded=0,
            price_assets=0,
            raw_assets=0,
            credits_used=0,
            benchmark_symbol="SPY",
            catalog_asset_ids=(),
        ),
    )

    def run_scheduler() -> Exception | None:
        connections.close_all()
        try:
            call_command(
                "scheduled_refresh",
                config=tmp_path / "universe.yml",
                stdout=StringIO(),
            )
        except CommandError as exc:
            return exc
        finally:
            connections.close_all()
        return None

    def complete_manual_false() -> JobRun:
        connections.close_all()
        try:
            barrier.wait(timeout=10)
            run = execute_us_daily_job(
                prepared,
                long_forecast_requested=False,
            )
            barrier.wait(timeout=10)
            return run
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as executor:
        scheduler_future = executor.submit(run_scheduler)
        manual_future = executor.submit(complete_manual_false)
        manual = manual_future.result(timeout=20)
        scheduler_error = scheduler_future.result(timeout=20)

    assert manual.status == JobRun.Status.SUCCESS
    assert manual.details["long_forecast_requested"] is False
    assert isinstance(scheduler_error, CommandError)
    assert "conflicts with the frozen market target gate" in str(scheduler_error)
    assert provider_entries == []
    parent = JobRun.objects.get(job_name="scheduled_refresh")
    assert parent.status == JobRun.Status.FAILED
    assert "sec_fundamentals" not in parent.details.get("stages", {})


def test_scheduler_retry_keeps_gate_reserved_before_sec_or_market(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    record = ProviderRecord.objects.create(provider="sec", enabled=True, status="ok")
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)
    real_reserve = scheduled_refresh.reserve_us_daily_target_gate
    reserved_gates: list[tuple[UUID | None, bool]] = []

    def reserve_then_fail(*args: object, **kwargs: object) -> object:
        gate = real_reserve(*args, **kwargs)
        reserved_gates.append((gate.reservation_id, gate.long_forecast_requested))
        raise ValueError("simulated failure after gate persistence")

    monkeypatch.setattr(
        scheduled_refresh,
        "reserve_us_daily_target_gate",
        reserve_then_fail,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "execute_sec_fundamentals_job",
        lambda **kwargs: pytest.fail("failure occurs before SEC stage entry"),
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "execute_us_daily_job",
        lambda *args, **kwargs: pytest.fail("market must not run after SEC setup failure"),
    )

    with pytest.raises(CommandError, match="after gate persistence"):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())

    reservation = JobRun.objects.get(job_name="daily")
    assert reservation.status == JobRun.Status.RUNNING
    assert reservation.details["target_gate_reservation"] is True
    assert reservation.details["long_forecast_requested"] is True

    record.enabled = False
    record.status = "disabled"
    record.save(update_fields=["enabled", "status"])
    monkeypatch.setattr(
        data_jobs_module,
        "_sample_current_long_forecast_requested",
        lambda: pytest.fail("retry must not resample mutable provider state"),
    )

    with pytest.raises(CommandError, match="after gate persistence"):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())

    reservation.refresh_from_db()
    assert reservation.status == JobRun.Status.RUNNING
    assert reservation.details["long_forecast_requested"] is True
    assert JobRun.objects.filter(job_name="daily").count() == 1
    assert {
        run.details["long_forecast_requested"] for run in JobRun.objects.filter(job_name="daily")
    } == {True}
    assert reserved_gates == [(reservation.pk, True), (reservation.pk, True)]


def test_manual_long_market_without_sec_child_fails_before_provider_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    listing = refresh_fixtures.pre_create_stock_listing("AAA")
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(
        monkeypatch,
        tmp_path,
        symbols=("AAA",),
        minimum_eligible=1,
    )
    ProviderRecord.objects.create(provider="sec", enabled=True, status="ok")
    refresh_fixtures.build_sec_evidence(
        store=AssetStore(tmp_path),
        company=listing.security.company,
        available_before=prepared.decision_time,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )
    manual = execute_us_daily_job(
        prepared,
        require_observed=True,
        long_forecast_requested=True,
    )
    assert manual.status == JobRun.Status.SUCCESS
    assert not JobRun.objects.filter(job_name="sec_fundamentals").exists()

    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)
    access_calls = 0

    def no_provider_access(*args: object, **kwargs: object) -> str:
        nonlocal access_calls
        access_calls += 1
        pytest.fail("committed market output must fail before provider access")

    monkeypatch.setattr(
        data_jobs_module,
        "_sample_current_long_forecast_requested",
        no_provider_access,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "execute_sec_fundamentals_job",
        no_provider_access,
    )
    monkeypatch.setattr(live_us_module.twelve_data, "resolve_api_key", no_provider_access)
    monkeypatch.setattr(live_us_module.twelve_data, "fetch_stock_catalog", no_provider_access)
    monkeypatch.setattr(
        live_us_module.twelve_data,
        "fetch_daily_price_series",
        no_provider_access,
    )
    monkeypatch.setattr(live_us_module.ProviderCreditBudget, "preflight", no_provider_access)
    monkeypatch.setattr(live_us_module.ProviderCreditBudget, "consume", no_provider_access)
    monkeypatch.setattr(sec_provider, "build_user_agent", no_provider_access)
    monkeypatch.setattr(sec_provider, "fetch_ticker_exchange_mapping", no_provider_access)
    monkeypatch.setattr(sec_provider, "fetch_submissions", no_provider_access)
    monkeypatch.setattr(sec_provider, "fetch_companyfacts", no_provider_access)

    with pytest.raises(CommandError, match="no same-target successful SEC child"):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())

    parent = JobRun.objects.get(job_name="scheduled_refresh")
    market_stage = parent.details["stages"]["market"]
    assert market_stage["status"] == JobRun.Status.SKIPPED
    assert market_stage["job_run_id"] != str(manual.pk)
    assert "sec_fundamentals" not in parent.details["stages"]
    assert access_calls == 0
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "SPY"]


def test_retry_recovers_successful_sec_child_even_if_provider_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    listing = refresh_fixtures.pre_create_stock_listing("AAA")
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(
        monkeypatch,
        tmp_path,
        symbols=("AAA",),
        minimum_eligible=1,
    )
    sec_evidence = refresh_fixtures.build_sec_evidence(
        store=AssetStore(tmp_path),
        company=listing.security.company,
        available_before=prepared.decision_time,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )
    prior_sec_success = execute_target_job(
        job_name="sec_fundamentals",
        region="us",
        target_date=prepared.target_date,
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
    ProviderRecord.objects.create(provider="sec", enabled=False, status="disabled")
    sec_calls = 0
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(
        scheduled_refresh,
        "prepare_us_daily_job",
        lambda **kwargs: prepared,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "clean_git_revision",
        lambda root: "a" * 40,
    )

    def recover_sec(**kwargs: object) -> JobRun:
        nonlocal sec_calls
        sec_calls += 1
        # A prior SUCCESS already exists for this exact target, so
        # `execute_target_job` must auto-skip by reference below without
        # ever invoking this task -- proving the disabled provider is never
        # actually contacted again for an already-succeeded target.
        return execute_target_job(
            job_name="sec_fundamentals",
            region="us",
            target_date=prepared.target_date,
            task=lambda run: pytest.fail(
                "SEC task must not re-run for an already-succeeded target"
            ),
        )

    monkeypatch.setattr(
        scheduled_refresh,
        "execute_sec_fundamentals_job",
        recover_sec,
    )

    call_command(
        "scheduled_refresh",
        config=tmp_path / "universe.yml",
        stdout=StringIO(),
    )

    assert sec_calls == 1
    parent = JobRun.objects.get(job_name="scheduled_refresh")
    assert parent.status == JobRun.Status.SUCCESS
    assert parent.details["stages"]["sec_fundamentals"]["status"] == JobRun.Status.SKIPPED
    assert parent.details["verification"]["status"] == "verified"
    assert parent.details["verification"]["sec"]["job_run_id"] == str(prior_sec_success.pk)


@pytest.mark.parametrize("entrypoint", ["execute", "command"])
@pytest.mark.parametrize(
    "original_gate",
    [False, True],
    ids=["disabled-to-enabled", "enabled-to-disabled"],
)
def test_daily_retry_after_post_analysis_etf_failure_keeps_original_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
    original_gate: bool,
) -> None:
    """Both canonical and manual-command retries recover the exact output
    under the pre-write gate without consulting any retry-time provider
    capability, credential, fetch, or quota boundary."""
    listing = refresh_fixtures.pre_create_stock_listing("AAA")
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(
        monkeypatch,
        tmp_path,
        symbols=("AAA",),
        minimum_eligible=1,
    )
    sec_record = ProviderRecord.objects.create(
        provider="sec",
        enabled=original_gate,
        status="ok" if original_gate else "disabled",
    )
    if original_gate:
        refresh_fixtures.build_sec_evidence(
            store=AssetStore(tmp_path),
            company=listing.security.company,
            available_before=prepared.decision_time,
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
        )
    monkeypatch.setattr(daily_command, "prepare_us_daily_job", lambda **kwargs: prepared)

    real_sync = live_us_module.sync_investable_spy_from_asset
    sync_calls = 0

    def fail_first_sync(**kwargs: object) -> object:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            raise ValueError("simulated post-analysis ETF projection failure")
        return real_sync(**kwargs)

    monkeypatch.setattr(live_us_module, "sync_investable_spy_from_asset", fail_first_sync)

    def invoke() -> JobRun | None:
        if entrypoint == "execute":
            return execute_us_daily_job(prepared)
        call_command(
            "daily",
            region="us",
            target_date=prepared.target_date.isoformat(),
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )
        return None

    expected_error = CommandError if entrypoint == "command" else ValueError
    with pytest.raises(expected_error, match="post-analysis ETF projection failure"):
        invoke()

    failed = JobRun.objects.get(job_name="daily", attempt=1)
    assert failed.status == JobRun.Status.FAILED
    assert failed.details["long_forecast_requested"] is original_gate
    analysis = AnalysisRun.objects.get(status="complete")
    manifest = DataAsset.objects.get(
        kind=ANALYSIS_OUTPUT_MANIFEST_KIND,
        subject=str(analysis.id),
    )
    prediction_ids = set(Prediction.objects.values_list("id", flat=True))

    sec_record.enabled = not original_gate
    sec_record.status = "ok" if sec_record.enabled else "disabled"
    sec_record.save(update_fields=["enabled", "status"])

    access_calls = 0

    def no_provider_access(*args: object, **kwargs: object) -> str:
        nonlocal access_calls
        access_calls += 1
        pytest.fail("retry must recover before provider enablement, credentials, fetch, or quota")

    monkeypatch.setattr(
        data_jobs_module,
        "_sample_current_long_forecast_requested",
        no_provider_access,
    )
    monkeypatch.setattr(live_us_module.twelve_data, "resolve_api_key", no_provider_access)
    monkeypatch.setattr(live_us_module.twelve_data, "fetch_stock_catalog", no_provider_access)
    monkeypatch.setattr(live_us_module.twelve_data, "fetch_daily_price_series", no_provider_access)
    monkeypatch.setattr(live_us_module.ProviderCreditBudget, "preflight", no_provider_access)
    monkeypatch.setattr(live_us_module.ProviderCreditBudget, "consume", no_provider_access)
    monkeypatch.setattr(sec_provider, "build_user_agent", no_provider_access)
    monkeypatch.setattr(sec_provider, "fetch_ticker_exchange_mapping", no_provider_access)
    monkeypatch.setattr(sec_provider, "fetch_submissions", no_provider_access)
    monkeypatch.setattr(sec_provider, "fetch_companyfacts", no_provider_access)

    result = invoke()
    succeeded = JobRun.objects.get(job_name="daily", attempt=2)
    if result is not None:
        assert result.pk == succeeded.pk
    assert succeeded.status == JobRun.Status.SUCCESS
    assert succeeded.details["long_forecast_requested"] is original_gate
    assert succeeded.details["analysis_run_id"] == str(analysis.id)
    assert AnalysisRun.objects.get(status="complete").id == analysis.id
    assert (
        DataAsset.objects.get(
            kind=ANALYSIS_OUTPUT_MANIFEST_KIND,
            subject=str(analysis.id),
        ).id
        == manifest.id
    )
    assert set(Prediction.objects.values_list("id", flat=True)) == prediction_ids
    assert {
        run.details["long_forecast_requested"] for run in JobRun.objects.filter(job_name="daily")
    } == {original_gate}
    assert sync_calls == 2
    assert access_calls == 0
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "SPY"]


def test_daily_retry_fails_closed_when_completed_output_has_no_gate_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(
        monkeypatch,
        tmp_path,
        symbols=("AAA",),
        minimum_eligible=1,
    )
    ProviderRecord.objects.create(provider="sec", enabled=False, status="disabled")
    monkeypatch.setattr(
        live_us_module,
        "sync_investable_spy_from_asset",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("ETF projection failure")),
    )
    with pytest.raises(ValueError, match="ETF projection failure"):
        execute_us_daily_job(prepared)

    failed = JobRun.objects.get(job_name="daily")
    JobRun.objects.filter(pk=failed.pk).update(details={})
    monkeypatch.setattr(
        data_jobs_module,
        "_sample_current_long_forecast_requested",
        lambda: pytest.fail(
            "completed output without gate evidence must not sample provider state"
        ),
    )

    with pytest.raises(ValueError, match="Completed US analysis output exists without"):
        execute_us_daily_job(prepared)

    assert JobRun.objects.filter(job_name="daily").count() == 1
    assert AnalysisRun.objects.filter(status="complete").count() == 1
    assert DataAsset.objects.filter(kind=ANALYSIS_OUTPUT_MANIFEST_KIND).count() == 1
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "SPY"]


def test_retry_uses_original_enabled_long_gate_without_provider_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The successful market child's pre-write SEC gate, not mutable retry
    state, keeps its committed long output required and recoverable."""
    listing = refresh_fixtures.pre_create_stock_listing("AAA")
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(
        monkeypatch,
        tmp_path,
        symbols=("AAA",),
        minimum_eligible=1,
    )
    sec_record = ProviderRecord.objects.create(provider="sec", enabled=True, status="ok")
    sec_evidence = refresh_fixtures.build_sec_evidence(
        store=AssetStore(tmp_path),
        company=listing.security.company,
        available_before=prepared.decision_time,
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)

    sec_task_calls = 0

    def execute_sec(**kwargs: object) -> JobRun:
        def task(run: JobRun) -> JobExecutionResult:
            nonlocal sec_task_calls
            sec_task_calls += 1
            return JobExecutionResult(
                details={
                    "mapping_asset_id": sec_evidence.mapping_asset_id,
                    "mapping_sha256": sec_evidence.mapping_sha256,
                    "cik_config_version": sec_evidence.cik_config_version,
                    "cik_config_hash": sec_evidence.cik_config_hash,
                    "config_version": sec_evidence.fundamentals_config_version,
                    "config_hash": sec_evidence.fundamentals_config_hash,
                    "asset_refs": [ref.to_json() for ref in sec_evidence.asset_refs],
                }
            )

        return execute_target_job(
            job_name="sec_fundamentals",
            region="us",
            target_date=prepared.target_date,
            task=task,
        )

    monkeypatch.setattr(scheduled_refresh, "execute_sec_fundamentals_job", execute_sec)

    evaluation_attempts = 0

    def flaky_evaluation(**kwargs: object) -> JobRun:
        nonlocal evaluation_attempts
        evaluation_attempts += 1
        if evaluation_attempts == 1:

            def task(run: JobRun) -> JobExecutionResult:
                raise ValueError("temporary evaluation failure")

            return execute_target_job(
                job_name="evaluate_predictions",
                region="us",
                target_date=prepared.target_date,
                task=task,
            )
        return execute_prediction_evaluation_job(**kwargs)

    monkeypatch.setattr(
        scheduled_refresh,
        "execute_prediction_evaluation_job",
        flaky_evaluation,
    )

    with pytest.raises(CommandError, match="temporary evaluation failure"):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())

    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    original_sec_id = first_parent.details["stages"]["sec_fundamentals"]["job_run_id"]
    original_market_id = first_parent.details["stages"]["market"]["job_run_id"]
    original_long_ids = set(
        Prediction.objects.filter(horizon__in=["3y", "5y"]).values_list("id", flat=True)
    )
    assert len(original_long_ids) == 2
    original_analysis_id = JobRun.objects.get(pk=original_market_id).details["analysis_run_id"]
    assert JobRun.objects.get(pk=original_market_id).details["long_forecast_requested"] is True
    assert sec_task_calls == 1

    sec_record.enabled = False
    sec_record.status = "disabled"
    sec_record.save(update_fields=["enabled", "status"])

    access_calls = 0

    def no_access(*args: object, **kwargs: object) -> str:
        nonlocal access_calls
        access_calls += 1
        pytest.fail("retry must not resolve credentials or fetch provider data")

    monkeypatch.setattr("stanstock.data.providers.twelve_data.resolve_api_key", no_access)
    monkeypatch.setattr(live_us_module.twelve_data, "fetch_stock_catalog", no_access)
    monkeypatch.setattr(live_us_module.twelve_data, "fetch_daily_price_series", no_access)
    monkeypatch.setattr(sec_provider, "build_user_agent", no_access)
    monkeypatch.setattr(sec_provider, "fetch_ticker_exchange_mapping", no_access)
    monkeypatch.setattr(sec_provider, "fetch_submissions", no_access)
    monkeypatch.setattr(sec_provider, "fetch_companyfacts", no_access)

    call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())

    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=2)
    sec_skip = JobRun.objects.get(
        pk=second_parent.details["stages"]["sec_fundamentals"]["job_run_id"]
    )
    market_skip = JobRun.objects.get(pk=second_parent.details["stages"]["market"]["job_run_id"])
    assert second_parent.status == JobRun.Status.SUCCESS
    assert sec_skip.details["successful_run_id"] == original_sec_id
    assert market_skip.details["successful_run_id"] == original_market_id
    assert second_parent.details["verification"]["analysis_run_id"] == original_analysis_id
    assert second_parent.details["verification"]["child_job_run_ids"]["sec_fundamentals"] == (
        original_sec_id
    )
    assert second_parent.details["verification"]["child_job_run_ids"]["market"] == (
        original_market_id
    )
    assert (
        set(Prediction.objects.filter(horizon__in=["3y", "5y"]).values_list("id", flat=True))
        == original_long_ids
    )
    assert sec_task_calls == 1
    assert access_calls == 0
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "SPY"]


def test_retry_uses_original_disabled_long_gate_without_provider_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Enabling SEC after a decision/medium-only market success cannot
    retroactively add an SEC child or make the verifier require long rows."""
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(
        monkeypatch,
        tmp_path,
        symbols=("AAA",),
        minimum_eligible=1,
    )
    sec_record = ProviderRecord.objects.create(provider="sec", enabled=False, status="disabled")
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)

    evaluation_attempts = 0

    def flaky_evaluation(**kwargs: object) -> JobRun:
        nonlocal evaluation_attempts
        evaluation_attempts += 1
        if evaluation_attempts == 1:

            def task(run: JobRun) -> JobExecutionResult:
                raise ValueError("temporary evaluation failure")

            return execute_target_job(
                job_name="evaluate_predictions",
                region="us",
                target_date=prepared.target_date,
                task=task,
            )
        return execute_prediction_evaluation_job(**kwargs)

    monkeypatch.setattr(
        scheduled_refresh,
        "execute_prediction_evaluation_job",
        flaky_evaluation,
    )

    with pytest.raises(CommandError, match="temporary evaluation failure"):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())

    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    original_market_id = first_parent.details["stages"]["market"]["job_run_id"]
    market_success = JobRun.objects.get(pk=original_market_id)
    assert market_success.details["long_forecast_requested"] is False
    assert not Prediction.objects.filter(horizon__in=["3y", "5y"]).exists()

    sec_record.enabled = True
    sec_record.status = "ok"
    sec_record.save(update_fields=["enabled", "status"])

    access_calls = 0

    def no_access(*args: object, **kwargs: object) -> str:
        nonlocal access_calls
        access_calls += 1
        pytest.fail("retry must not resolve credentials or fetch provider data")

    monkeypatch.setattr(
        scheduled_refresh,
        "execute_sec_fundamentals_job",
        no_access,
    )
    monkeypatch.setattr("stanstock.data.providers.twelve_data.resolve_api_key", no_access)
    monkeypatch.setattr(live_us_module.twelve_data, "fetch_stock_catalog", no_access)
    monkeypatch.setattr(live_us_module.twelve_data, "fetch_daily_price_series", no_access)
    monkeypatch.setattr(sec_provider, "build_user_agent", no_access)
    monkeypatch.setattr(sec_provider, "fetch_ticker_exchange_mapping", no_access)
    monkeypatch.setattr(sec_provider, "fetch_submissions", no_access)
    monkeypatch.setattr(sec_provider, "fetch_companyfacts", no_access)

    call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())

    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=2)
    market_skip = JobRun.objects.get(pk=second_parent.details["stages"]["market"]["job_run_id"])
    assert second_parent.status == JobRun.Status.SUCCESS
    assert "sec_fundamentals" not in second_parent.details["stages"]
    assert second_parent.details["verification"]["sec"] == {"required": False}
    assert market_skip.details["successful_run_id"] == original_market_id
    assert second_parent.details["verification"]["child_job_run_ids"]["market"] == (
        original_market_id
    )
    assert not Prediction.objects.filter(horizon__in=["3y", "5y"]).exists()
    assert access_calls == 0
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "SPY"]


def test_verification_failure_after_real_success_fails_closed_without_refetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """F7: a corrupted *local* output must fail the parent even though every
    child `JobRun` already recorded SUCCESS/SKIPPED -- proving the parent's
    own success genuinely depends on independently re-provable evidence, not
    merely on child status. The correcting retry must then recover without
    spending any additional provider credit.

    The parent `scheduled_refresh` job is itself an idempotent target job
    keyed by ``(job_name, region, target_date)``: once it records SUCCESS,
    a later invocation for the same target short-circuits before doing any
    work at all (by design -- this is the same recovery pattern children
    use). So corruption must be injected *while the parent's own attempt
    sequence is still open*: attempt 1 is forced to fail for an unrelated
    reason (a one-shot evaluation failure, same technique as
    `test_scheduled_refresh_retry_and_no_op_render_persisted_evidence`) after the
    market child has already really executed and persisted; the local
    output is corrupted before attempt 2, which recovers every child by
    reference (no refetch) yet must still fail on verification.
    """
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(monkeypatch, tmp_path)
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(
        scheduled_refresh,
        "prepare_us_daily_job",
        lambda **kwargs: prepared,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "clean_git_revision",
        lambda root: "a" * 40,
    )

    evaluation_attempts = 0

    def flaky_once_evaluation(**kwargs: object) -> JobRun:
        nonlocal evaluation_attempts
        evaluation_attempts += 1
        if evaluation_attempts == 1:

            def task(run: JobRun) -> JobExecutionResult:
                raise ValueError("temporary evaluation failure")

            return execute_target_job(
                job_name="evaluate_predictions",
                region="us",
                target_date=prepared.target_date,
                task=task,
            )
        return execute_prediction_evaluation_job(**kwargs)

    monkeypatch.setattr(
        scheduled_refresh,
        "execute_prediction_evaluation_job",
        flaky_once_evaluation,
    )

    with pytest.raises(CommandError, match="temporary evaluation failure"):
        call_command(
            "scheduled_refresh",
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )
    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    assert first_parent.status == JobRun.Status.FAILED
    assert first_parent.details["stages"]["market"]["status"] == JobRun.Status.SUCCESS
    assert "verification" not in first_parent.details
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]

    # Corrupt one persisted local output directly -- `LatestMarketData` is
    # not immutability-protected (it is meant to be overwritten with each
    # fresh session), so deleting a row here models genuinely losing local
    # proof (e.g. a partial write, a disk issue) without touching any
    # immutable evidence row. Its field values are captured first so the
    # "repair" step below can restore it from already-committed evidence
    # rather than re-fetching from the provider.
    corrupted = LatestMarketData.objects.get(listing__ticker="AAA")
    restore_fields = {
        field.name: getattr(corrupted, field.name)
        for field in LatestMarketData._meta.get_fields()
        if hasattr(field, "attname") and field.name != "id"
    }
    corrupted.delete()

    with pytest.raises(CommandError, match="Scheduled refresh failed"):
        call_command(
            "scheduled_refresh",
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )

    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=2)
    assert second_parent.status == JobRun.Status.FAILED
    # Every child stage still recovers by reference to its already-committed
    # success -- the corruption is caught by verification, not by re-running
    # (and possibly re-fetching) any child.
    assert second_parent.details["stages"]["market"]["status"] == JobRun.Status.SKIPPED
    assert second_parent.details["stages"]["evaluation"]["status"] == JobRun.Status.SUCCESS
    verification = second_parent.details["verification"]
    assert verification["status"] == "failed"
    assert verification["reason_code"] == "latest_market_data_missing"
    # No success-shaped partial verification block, and no local filesystem
    # path leaks into the persisted failure detail.
    assert "checks" not in verification
    assert not any(str(tmp_path) in str(value) for value in verification.values())
    # Zero additional provider fetches/credits were spent recovering (or
    # failing to recover) an already-committed child.
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]

    # Repairing the local output from the exact evidence already committed
    # (not re-fetching from the provider) lets the retry recover and reach a
    # genuinely verified success again.
    LatestMarketData.objects.create(**restore_fields)

    call_command(
        "scheduled_refresh",
        config=tmp_path / "universe.yml",
        stdout=StringIO(),
    )
    third_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=3)
    assert third_parent.status == JobRun.Status.SUCCESS
    assert third_parent.details["stages"]["market"]["status"] == JobRun.Status.SKIPPED
    assert third_parent.details["verification"]["status"] == "verified"
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]


def test_missing_bound_price_file_fails_path_free_and_recovers_without_refetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """F10: losing the *local file* behind an otherwise intact, correctly
    registered price asset (distinct from losing the `LatestMarketData` row
    itself) must fail the parent with a structured, path-free verification
    reason -- never a raw filesystem path in any error, detail, or log --
    and must recover on retry without any additional provider fetch.

    As in `test_verification_failure_after_real_success_fails_closed_without_refetch`,
    the parent job is itself idempotent by `(job_name, region, target_date)`,
    so corruption must be injected while its own attempt sequence is still
    open (a one-shot evaluation failure keeps attempt 1 from recording
    SUCCESS after the market child has already really executed).
    """
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(monkeypatch, tmp_path)
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)

    evaluation_attempts = 0

    def flaky_once_evaluation(**kwargs: object) -> JobRun:
        nonlocal evaluation_attempts
        evaluation_attempts += 1
        if evaluation_attempts == 1:

            def task(run: JobRun) -> JobExecutionResult:
                raise ValueError("temporary evaluation failure")

            return execute_target_job(
                job_name="evaluate_predictions",
                region="us",
                target_date=prepared.target_date,
                task=task,
            )
        return execute_prediction_evaluation_job(**kwargs)

    monkeypatch.setattr(
        scheduled_refresh, "execute_prediction_evaluation_job", flaky_once_evaluation
    )

    with pytest.raises(CommandError, match="temporary evaluation failure"):
        call_command(
            "scheduled_refresh",
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )
    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    assert first_parent.status == JobRun.Status.FAILED
    assert first_parent.details["stages"]["market"]["status"] == JobRun.Status.SUCCESS
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]

    bound = LatestMarketData.objects.select_related("source_asset").get(listing__ticker="AAA")
    asset_path = tmp_path / bound.source_asset.relative_path
    assert asset_path.exists()
    original_bytes = asset_path.read_bytes()
    asset_path.unlink()

    with pytest.raises(CommandError, match="Scheduled refresh failed"):
        call_command(
            "scheduled_refresh",
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )
    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=2)
    assert second_parent.status == JobRun.Status.FAILED
    assert second_parent.details["stages"]["market"]["status"] == JobRun.Status.SKIPPED
    verification = second_parent.details["verification"]
    assert verification["status"] == "failed"
    assert verification["reason_code"] == "latest_market_data_asset_unreadable"
    assert not any(str(tmp_path) in str(value) for value in verification.values())
    assert str(tmp_path) not in str(second_parent.details)
    # No additional provider credit was spent attempting (and failing) to
    # recover a child that already succeeded.
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]

    # Restoring the exact original bytes locally (no re-fetch) lets the
    # retry verify successfully again.
    asset_path.write_bytes(original_bytes)

    call_command(
        "scheduled_refresh",
        config=tmp_path / "universe.yml",
        stdout=StringIO(),
    )
    third_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=3)
    assert third_parent.status == JobRun.Status.SUCCESS
    assert third_parent.details["verification"]["status"] == "verified"
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]


def test_manifest_corruption_retry_is_credential_free_and_recovers_without_refetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A prior-success market child with a corrupt output manifest fails
    parent verification, then recovers from the same local bytes without
    resolving credentials or spending another provider credit."""
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(monkeypatch, tmp_path)
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)

    evaluation_attempts = 0

    def flaky_once_evaluation(**kwargs: object) -> JobRun:
        nonlocal evaluation_attempts
        evaluation_attempts += 1
        if evaluation_attempts == 1:

            def task(run: JobRun) -> JobExecutionResult:
                raise ValueError("temporary evaluation failure")

            return execute_target_job(
                job_name="evaluate_predictions",
                region="us",
                target_date=prepared.target_date,
                task=task,
            )
        return execute_prediction_evaluation_job(**kwargs)

    monkeypatch.setattr(
        scheduled_refresh,
        "execute_prediction_evaluation_job",
        flaky_once_evaluation,
    )

    with pytest.raises(CommandError, match="temporary evaluation failure"):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())
    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    assert first_parent.status == JobRun.Status.FAILED
    assert first_parent.details["stages"]["market"]["status"] == JobRun.Status.SUCCESS
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]

    manifest = DataAsset.objects.get(kind=ANALYSIS_OUTPUT_MANIFEST_KIND)
    manifest_path = AssetStore(tmp_path).resolve(manifest.relative_path)
    original_bytes = manifest_path.read_bytes()
    manifest_path.write_bytes(b"corrupt manifest bytes")

    credential_resolutions = 0

    def no_credentials(*args: object, **kwargs: object) -> str:
        nonlocal credential_resolutions
        credential_resolutions += 1
        pytest.fail("prior-success retry must not resolve provider credentials")

    monkeypatch.setattr(
        "stanstock.data.providers.twelve_data.resolve_api_key",
        no_credentials,
    )

    with pytest.raises(CommandError, match="Scheduled refresh failed"):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())

    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=2)
    assert second_parent.status == JobRun.Status.FAILED
    assert second_parent.details["stages"]["market"]["status"] == JobRun.Status.SKIPPED
    verification = second_parent.details["verification"]
    assert verification["status"] == "failed"
    assert verification["reason_code"] == "asset_corrupt"
    assert "checks" not in verification
    assert not any(str(tmp_path) in str(value) for value in verification.values())
    assert credential_resolutions == 0
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]

    manifest_path.write_bytes(original_bytes)
    call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())

    third_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=3)
    assert third_parent.status == JobRun.Status.SUCCESS
    assert third_parent.details["stages"]["market"]["status"] == JobRun.Status.SKIPPED
    assert third_parent.details["verification"]["status"] == "verified"
    assert credential_resolutions == 0
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]


def test_final_asset_integrity_os_error_fails_path_free_without_leaking_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An OS-level failure (permission error, I/O error) resolving/reading a
    registered asset's file during the final physical-integrity pass must
    never leak its own exception message (which embeds a resolved absolute
    path) into `JobRun.error`, `JobRun.details`, or any log line -- it must
    surface only as the structured, path-free `asset_integrity_failed`
    verification failure.

    As in `test_missing_bound_price_file_fails_path_free_and_recovers_without_refetch`,
    the parent job is itself idempotent by `(job_name, region, target_date)`,
    so the OS-level failure must be injected while attempt 1's own sequence
    is still open (a one-shot evaluation failure keeps attempt 1 from
    recording SUCCESS after the market child has already really executed).
    """
    prepared, _config, _calls = _real_prepared(monkeypatch, tmp_path)
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)

    sentinel = "/sentinel-should-never-leak/asset.bin"
    injected_at_attempt = 2
    evaluation_attempts = 0

    def flaky_once_evaluation(**kwargs: object) -> JobRun:
        nonlocal evaluation_attempts
        evaluation_attempts += 1
        if evaluation_attempts == 1:

            def task(run: JobRun) -> JobExecutionResult:
                raise ValueError("temporary evaluation failure")

            return execute_target_job(
                job_name="evaluate_predictions",
                region="us",
                target_date=prepared.target_date,
                task=task,
            )
        return execute_prediction_evaluation_job(**kwargs)

    monkeypatch.setattr(
        scheduled_refresh, "execute_prediction_evaluation_job", flaky_once_evaluation
    )

    with pytest.raises(CommandError, match="temporary evaluation failure"):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())
    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    assert first_parent.status == JobRun.Status.FAILED

    def boom_file_digest(source: object, algorithm: str) -> object:
        raise OSError(f"[Errno 5] Input/output error: {sentinel!r}")

    monkeypatch.setattr("stanstock.core.integrity.hashlib.file_digest", boom_file_digest)

    caplog.set_level("INFO")
    with pytest.raises(CommandError):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())

    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=injected_at_attempt)
    assert second_parent.status == JobRun.Status.FAILED
    assert sentinel not in second_parent.error
    assert sentinel not in str(second_parent.details)
    assert sentinel not in caplog.text
    verification = second_parent.details["verification"]
    assert verification["status"] == "failed"
    assert verification["reason_code"] == "asset_integrity_failed"


def test_recovered_spy_evidence_read_failure_fails_path_free_and_recovers_without_refetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A physical read/parse failure recovering the exact SPY price evidence
    asset -- via `_existing_completed_result`'s zero-fetch recovery path,
    after a real analysis has already committed but the initial ETF
    projection deliberately failed -- must never leak a path-bearing
    `OSError`/Polars exception into any command text, exception cause, log
    line, or child/parent `JobRun.error`/`details`. It must fail closed with
    a controlled, path-free error, then recover cleanly with zero
    additional provider fetch once the injected failure is removed.
    """
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(monkeypatch, tmp_path)
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)

    real_sync = live_us_module.sync_investable_spy_from_asset
    sync_attempts = 0

    def flaky_once_sync(**kwargs: object) -> object:
        nonlocal sync_attempts
        sync_attempts += 1
        if sync_attempts == 1:
            raise ValueError("temporary ETF projection failure")
        return real_sync(**kwargs)

    monkeypatch.setattr(live_us_module, "sync_investable_spy_from_asset", flaky_once_sync)

    # Attempt 1: analysis/predictions genuinely commit, but the initial SPY
    # projection is deliberately failed afterward -- the market stage fails
    # even though the analysis evidence is already durable.
    with pytest.raises(CommandError, match="temporary ETF projection failure"):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())
    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    assert first_parent.status == JobRun.Status.FAILED
    assert first_parent.details["stages"]["market"]["status"] == JobRun.Status.FAILED
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]

    benchmark_asset = DataAsset.objects.get(
        provider="twelve_data", kind="price_history", subject="SPY"
    )
    assert benchmark_asset.relative_path

    sentinel = "/sentinel-should-never-leak/spy-price-evidence.parquet"

    def boom_read_bytes(self: AssetStore, relative_path: str) -> object:
        raise OSError(f"[Errno 5] Input/output error: {sentinel!r}")

    # Attempt 2: `_existing_completed_result` recovers the already-committed
    # analysis and tries to re-project the exact SPY evidence asset, whose
    # physical read now fails with a path-bearing error.
    caplog.set_level("INFO")
    with monkeypatch.context() as read_patch:
        read_patch.setattr(AssetStore, "read_bytes", boom_read_bytes)
        with pytest.raises(CommandError) as excinfo:
            call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())
    assert sentinel not in str(excinfo.value)
    assert sentinel not in caplog.text
    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=2)
    assert second_parent.status == JobRun.Status.FAILED
    assert second_parent.details["stages"]["market"]["status"] == JobRun.Status.FAILED
    assert sentinel not in second_parent.error
    assert sentinel not in str(second_parent.details)
    # No additional provider credit was spent on the failed recovery
    # attempt -- `_existing_completed_result` returns before any fetch.
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]

    # Attempt 3: with the injected read failure removed, the same exact
    # recovered evidence asset now reads cleanly and the parent succeeds --
    # still with zero additional provider fetch.
    call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())
    third_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=3)
    assert third_parent.status == JobRun.Status.SUCCESS
    assert third_parent.details["stages"]["market"]["status"] == JobRun.Status.SUCCESS
    assert third_parent.details["verification"]["status"] == "verified"
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]


def test_recovered_spy_evidence_checksum_mismatch_fails_closed_before_any_mutation_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A physically altered SPY price file that still parses as a
    structurally valid Parquet frame (so no `OSError`/Polars parse error is
    raised) but whose bytes no longer match the registered
    `DataAsset.sha256` -- e.g. a wrong close projected into an otherwise
    well-formed file -- must fail the market stage *before* any SPY
    listing/`LatestMarketData` mutation, must never leak a filesystem path,
    must spend zero additional provider credit, and must recover cleanly
    (not remain poisoned) once the exact original bytes are restored.
    """
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(monkeypatch, tmp_path)
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)

    real_sync = live_us_module.sync_investable_spy_from_asset
    sync_attempts = 0

    def flaky_once_sync(**kwargs: object) -> object:
        nonlocal sync_attempts
        sync_attempts += 1
        if sync_attempts == 1:
            raise ValueError("temporary ETF projection failure")
        return real_sync(**kwargs)

    monkeypatch.setattr(live_us_module, "sync_investable_spy_from_asset", flaky_once_sync)

    # Attempt 1: analysis/predictions genuinely commit, but the initial SPY
    # projection is deliberately failed afterward -- no SPY listing or
    # `LatestMarketData` exists yet.
    with pytest.raises(CommandError, match="temporary ETF projection failure"):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())
    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    assert first_parent.status == JobRun.Status.FAILED
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]
    assert not LatestMarketData.objects.filter(listing__ticker="SPY").exists()

    benchmark_asset = DataAsset.objects.get(
        provider="twelve_data", kind="price_history", subject="SPY"
    )
    registered_sha256 = benchmark_asset.sha256
    asset_path = tmp_path / benchmark_asset.relative_path
    original_bytes = asset_path.read_bytes()

    # Tamper the physical bytes: still a structurally valid Parquet frame,
    # but its content (and therefore checksum) has changed -- e.g. the last
    # close has been altered.
    tampered_frame = pl.read_parquet(BytesIO(original_bytes)).with_columns(
        (pl.col("close") * 5.0).alias("close")
    )
    buffer = BytesIO()
    tampered_frame.write_parquet(buffer)
    tampered_bytes = buffer.getvalue()
    assert tampered_bytes != original_bytes
    asset_path.write_bytes(tampered_bytes)

    # Attempt 2: `_existing_completed_result` recovers the already-committed
    # analysis and tries to re-project the exact SPY evidence asset. The
    # checksum no longer matches, so the market stage must fail closed
    # *before* any SPY listing/`LatestMarketData` row is created.
    with pytest.raises(CommandError):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())
    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=2)
    assert second_parent.status == JobRun.Status.FAILED
    assert second_parent.details["stages"]["market"]["status"] == JobRun.Status.FAILED
    assert str(tmp_path) not in str(second_parent.error)
    assert str(tmp_path) not in str(second_parent.details)
    assert not LatestMarketData.objects.filter(listing__ticker="SPY").exists()
    # No additional provider credit was spent on the failed recovery
    # attempt -- `_existing_completed_result` returns before any fetch.
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]
    # The asset's own registered checksum was never mutated by the
    # tampering -- confirming the failure comes from a genuine mismatch
    # against durable evidence, not a fixture artifact.
    benchmark_asset.refresh_from_db()
    assert benchmark_asset.sha256 == registered_sha256

    # Restoring the exact original bytes (no re-fetch) lets the retry
    # recover cleanly rather than remaining poisoned by the tampered read.
    asset_path.write_bytes(original_bytes)

    call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())
    third_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=3)
    assert third_parent.status == JobRun.Status.SUCCESS
    assert third_parent.details["stages"]["market"]["status"] == JobRun.Status.SUCCESS
    assert third_parent.details["verification"]["status"] == "verified"
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]
    recovered = LatestMarketData.objects.get(listing__ticker="SPY")
    assert recovered.source_asset_id == benchmark_asset.id


def test_run_us_daily_default_store_construction_failure_fails_path_free_before_any_fetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`run_us_daily`'s own eager default-store construction -- the actual
    entry point reached on every scheduled retry, *before*
    `_existing_completed_result`'s zero-fetch recovery path -- must
    normalize a construction failure (an unwritable/misconfigured root)
    into a stable, path-free error. It must never leak a path into the
    command's `CommandError`, its cause, any log line, or child/parent
    `JobRun.error`/`details`, and it must fail *before* spending any
    provider credit recovering an already-completed run.

    As in `test_recovered_spy_evidence_read_failure_fails_path_free_and_recovers_without_refetch`,
    the parent job is itself idempotent by `(job_name, region, target_date)`,
    so the fault must be injected while attempt 1's own "daily" child
    sequence is still open (a one-shot SPY-projection failure keeps the
    "daily" child from recording SUCCESS after the real analysis has
    already committed).
    """
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(monkeypatch, tmp_path)
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)

    real_sync = live_us_module.sync_investable_spy_from_asset
    sync_attempts = 0

    def flaky_once_sync(**kwargs: object) -> object:
        nonlocal sync_attempts
        sync_attempts += 1
        if sync_attempts == 1:
            raise ValueError("temporary ETF projection failure")
        return real_sync(**kwargs)

    monkeypatch.setattr(live_us_module, "sync_investable_spy_from_asset", flaky_once_sync)

    # Attempt 1: analysis/predictions genuinely commit, but the initial SPY
    # projection is deliberately failed afterward -- the "daily" child
    # itself records FAILED even though its evidence is already durable.
    with pytest.raises(CommandError, match="temporary ETF projection failure"):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())
    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    assert first_parent.status == JobRun.Status.FAILED
    assert first_parent.details["stages"]["market"]["status"] == JobRun.Status.FAILED
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]

    sentinel = "/sentinel-should-never-leak/asset-store-root"

    def boom_init(self: AssetStore, root: Path | None = None) -> None:
        raise OSError(f"[Errno 13] Permission denied: {sentinel!r}")

    # Attempt 2: `run_us_daily`'s own default `AssetStore()` construction --
    # which happens *before* `_existing_completed_result`'s zero-fetch
    # recovery attempt -- now fails. No provider credit should be spent,
    # since the fault is hit before recovery is ever attempted.
    caplog.set_level("INFO")
    with monkeypatch.context() as store_patch:
        store_patch.setattr(AssetStore, "__init__", boom_init)
        with pytest.raises(CommandError) as excinfo:
            call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())
    assert sentinel not in str(excinfo.value)
    assert excinfo.value.__cause__ is not None
    assert sentinel not in str(excinfo.value.__cause__)
    assert sentinel not in caplog.text
    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=2)
    assert second_parent.status == JobRun.Status.FAILED
    assert second_parent.details["stages"]["market"]["status"] == JobRun.Status.FAILED
    assert sentinel not in second_parent.error
    assert sentinel not in str(second_parent.details)
    market_child = JobRun.objects.get(job_name="daily", region="us", attempt=2)
    assert market_child.status == JobRun.Status.FAILED
    assert sentinel not in market_child.error
    assert sentinel not in str(market_child.details)
    # No additional provider credit was spent -- the construction failure
    # happens before `_existing_completed_result` is ever reached.
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]

    # Attempt 3: with the fault removed, the same completed run recovers
    # cleanly -- still with zero additional provider fetch.
    call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())
    third_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=3)
    assert third_parent.status == JobRun.Status.SUCCESS
    assert third_parent.details["stages"]["market"]["status"] == JobRun.Status.SUCCESS
    assert third_parent.details["verification"]["status"] == "verified"
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]


def test_recovered_spy_evidence_malformed_non_target_row_fails_closed_with_zero_fetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A malformed row *elsewhere* in the exact, checksum-valid SPY price
    asset -- not the target-date row itself -- must fail the market stage
    closed, both on the run that first persists it and on every later
    zero-fetch recovery attempt, never silently filtered away to let the
    stage "succeed" over a quietly-reduced frame.

    `DataAsset` rows are immutable (enforced by a DB trigger), so this
    cannot be simulated by mutating an already-committed asset's bytes and
    checksum in place, unlike the separate physical-tampering/checksum-
    mismatch regressions. Instead, the malformed row is injected at the
    provider boundary so the very first persisted SPY asset is genuinely
    checksum-valid *and* malformed from the moment it is written -- exactly
    the "checksum-valid malformed evidence" scenario this fix defends
    against.
    """
    prepared, config, (catalog_calls, price_calls) = _real_prepared(monkeypatch, tmp_path)
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)

    real_fetch_prices = live_us_module.twelve_data.fetch_daily_price_series
    target_date = prepared.target_date

    def fetch_prices_with_malformed_benchmark(symbol: str, **kwargs: object) -> PriceSeries:
        if symbol != config.benchmark_symbol:
            return real_fetch_prices(symbol, **kwargs)
        price_calls.append(symbol)
        bars = (
            PriceBar(
                trade_date=live_us_module._years_before(target_date, 1),
                open=Decimal("400"),
                high=Decimal("402"),
                low=Decimal("398"),
                close=Decimal("400"),
                volume=1_000_000,
            ),
            PriceBar(
                trade_date=target_date - timedelta(days=1),
                open=Decimal("410"),
                high=Decimal("412"),
                low=Decimal("408"),
                close=Decimal("nan"),
                volume=1_100_000,
            ),
            PriceBar(
                trade_date=target_date,
                open=Decimal("415"),
                high=Decimal("418"),
                low=Decimal("413"),
                close=Decimal("417"),
                volume=1_200_000,
            ),
        )
        return PriceSeries(
            provider="twelve_data",
            symbol=symbol,
            currency="USD",
            bars=bars,
            retrieved_at=DECISION_TIME,
            source_url=f"https://api.twelvedata.com/time_series?symbol={symbol}",
            raw_bytes=f'{{"status":"ok","symbol":"{symbol}"}}'.encode(),
            exchange="NYSE ARCA",
            mic_code="ARCX",
            instrument_type="ETF",
            exchange_timezone="America/New_York",
            adjustment="splits",
        )

    monkeypatch.setattr(
        "stanstock.data.live_us.twelve_data.fetch_daily_price_series",
        fetch_prices_with_malformed_benchmark,
    )

    # Attempt 1: the real pipeline persists a genuinely checksum-valid SPY
    # asset -- but its middle row is malformed from the moment it is
    # written. The analysis/predictions commit, but the SPY projection at
    # the end of `run_us_daily` must itself fail closed on this malformed
    # evidence -- no `flaky_once` artifice is needed, since the malformed
    # data alone is enough to fail the stage.
    with pytest.raises(CommandError):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())
    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    assert first_parent.status == JobRun.Status.FAILED
    assert first_parent.details["stages"]["market"]["status"] == JobRun.Status.FAILED
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]
    assert not LatestMarketData.objects.filter(listing__ticker="SPY").exists()

    benchmark_asset = DataAsset.objects.get(
        provider="twelve_data", kind="price_history", subject="SPY"
    )
    persisted_bytes = (tmp_path / benchmark_asset.relative_path).read_bytes()
    assert hashlib.sha256(persisted_bytes).hexdigest() == benchmark_asset.sha256
    persisted_frame = pl.read_parquet(BytesIO(persisted_bytes))
    assert persisted_frame["close"].is_nan().any()

    # Attempt 2: `_existing_completed_result` recovers the already-
    # committed analysis and resolves the *same* checksum-valid, malformed
    # benchmark asset -- it must still fail closed, and it must not spend
    # any additional provider credit doing so.
    with pytest.raises(CommandError):
        call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())
    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=2)
    assert second_parent.status == JobRun.Status.FAILED
    assert second_parent.details["stages"]["market"]["status"] == JobRun.Status.FAILED
    assert not LatestMarketData.objects.filter(listing__ticker="SPY").exists()
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]


SEC_SENTINEL_ABS_PATH = "/definitely/not/a/real/path/sec-sentinel-evidence.json"


def _sec_command_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[PreparedUsDailyJob, list[str], list[str], list[str]]:
    """Wire a lightweight `scheduled_refresh` run whose SEC stage exercises
    the *real*, unmocked `run_sec_ingestion` pipeline.

    Market/evaluation/portfolio stages are never reached: a failed SEC
    stage aborts the parent job before them (`_run_stage`'s `failures=None`
    call immediately re-raises), so a bare `SimpleNamespace`-backed
    `_prepared()` config is enough -- there is no need to stand up the full
    real Twelve Data market pipeline just to prove this SEC-domain
    path-confidentiality boundary.
    """
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    prepared = _prepared()
    universe_config = sec_fixtures._universe()
    cik_config = sec_fixtures._cik_config()
    fundamentals_config = load_sec_fundamentals_config()
    monkeypatch.setattr(
        sec_jobs_module, "load_us_universe_config", lambda *a, **kw: universe_config
    )
    monkeypatch.setattr(sec_jobs_module, "load_sec_cik_config", lambda *a, **kw: cik_config)
    monkeypatch.setattr(
        sec_jobs_module, "load_sec_fundamentals_config", lambda *a, **kw: fundamentals_config
    )
    sec_fixtures._listing()
    ProviderRecord.objects.create(provider="sec", enabled=True, status="ok")

    mapping_calls: list[str] = []
    submissions_calls: list[str] = []
    companyfacts_calls: list[str] = []

    def fetch_mapping(**kwargs: object) -> FundamentalSourcePayload:
        mapping_calls.append(MAPPING_SUBJECT)
        return sec_fixtures._payload(
            MAPPING_SUBJECT,
            sec_fixtures.MAPPING_BYTES,
            "https://www.sec.gov/files/company_tickers_exchange.json",
        )

    def fetch_submissions(cik: str) -> FundamentalSourcePayload:
        submissions_calls.append(cik)
        return sec_fixtures._payload(
            "0000320193",
            sec_fixtures.SUBMISSIONS_BYTES,
            "https://data.sec.gov/submissions/CIK0000320193.json",
        )

    def fetch_companyfacts(cik: str) -> FundamentalSourcePayload:
        companyfacts_calls.append(cik)
        return sec_fixtures._payload(
            "0000320193",
            sec_fixtures._companyfacts_bytes(),
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        )

    monkeypatch.setattr(sec_provider, "fetch_ticker_exchange_mapping", fetch_mapping)
    monkeypatch.setattr(sec_provider, "fetch_submissions", fetch_submissions)
    monkeypatch.setattr(
        sec_provider,
        "fetch_submissions_history",
        lambda filename: sec_fixtures._payload(
            filename,
            sec_fixtures.HISTORY_BYTES,
            f"https://data.sec.gov/submissions/{filename}",
        ),
    )
    monkeypatch.setattr(sec_provider, "fetch_companyfacts", fetch_companyfacts)

    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(scheduled_refresh, "prepare_us_daily_job", lambda **kwargs: prepared)
    monkeypatch.setattr(scheduled_refresh, "clean_git_revision", lambda root: "a" * 40)
    monkeypatch.setattr(
        scheduled_refresh,
        "execute_us_daily_job",
        lambda *a, **kw: pytest.fail("market stage must remain blocked by a failed SEC stage"),
    )
    return prepared, mapping_calls, submissions_calls, companyfacts_calls


@pytest.mark.parametrize(
    "fault_kind,pre_register_mapping,expect_mapping_fetch",
    [
        ("constructor", False, False),
        ("resolve", False, True),
        ("read", True, False),
        ("checksum", True, False),
        ("hash", False, True),
        ("write", False, True),
    ],
)
def test_sec_stage_storage_faults_fail_closed_without_path_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    fault_kind: str,
    pre_register_mapping: bool,
    expect_mapping_fetch: bool,
) -> None:
    """Every SEC storage-boundary failure (default-store construction, path
    resolution, physical read, checksum/hash computation, physical write)
    must fail the SEC stage closed without leaking a resolved absolute or
    relative asset path into `CommandError`, its cause chain, logs, the SEC
    child or parent `JobRun.error`/`details`, or `ProviderRecord.last_error`
    -- and must spend zero downstream SEC company or Twelve Data/market
    provider calls doing so. Removing the fault must let a plain retry
    recover normally.
    """
    prepared, mapping_calls, submissions_calls, companyfacts_calls = _sec_command_ready(
        monkeypatch, tmp_path
    )
    if pre_register_mapping:
        sec_fixtures._mapping_asset(AssetStore(tmp_path))

    def broken_init(self: AssetStore, root: Path | None = None) -> None:
        raise OSError(f"[Errno 13] Permission denied: '{SEC_SENTINEL_ABS_PATH}'")

    def broken_resolve(self: AssetStore, relative_path: str) -> Path:
        raise ValueError(f"Asset path escapes STANSTOCK_DATA_DIR: {SEC_SENTINEL_ABS_PATH}")

    def broken_read_bytes(self: AssetStore, relative_path: str) -> bytes:
        raise OSError(f"[Errno 2] No such file or directory: '{SEC_SENTINEL_ABS_PATH}'")

    def unhashable_read_bytes(self: AssetStore, relative_path: str) -> bytes:
        return cast(bytes, "not-bytes")

    def broken_write_bytes(self: AssetStore, relative_path: str, payload: bytes) -> object:
        raise OSError(f"[Errno 28] No space left on device: '{SEC_SENTINEL_ABS_PATH}'")

    def fetch_mapping_unhashable_content(**kwargs: object) -> FundamentalSourcePayload:
        mapping_calls.append(MAPPING_SUBJECT)
        return sec_fixtures._payload(
            MAPPING_SUBJECT,
            cast(bytes, object()),
            f"https://www.sec.gov/files/{SEC_SENTINEL_ABS_PATH}",
        )

    with monkeypatch.context() as ctx:
        if fault_kind == "constructor":
            ctx.setattr(AssetStore, "__init__", broken_init)
        elif fault_kind == "resolve":
            ctx.setattr(AssetStore, "resolve", broken_resolve)
        elif fault_kind == "read":
            ctx.setattr(AssetStore, "read_bytes", broken_read_bytes)
        elif fault_kind == "checksum":
            ctx.setattr(AssetStore, "read_bytes", unhashable_read_bytes)
        elif fault_kind == "write":
            ctx.setattr(AssetStore, "write_bytes", broken_write_bytes)
        elif fault_kind == "hash":
            ctx.setattr(
                sec_provider, "fetch_ticker_exchange_mapping", fetch_mapping_unhashable_content
            )
        else:  # pragma: no cover - guards a typo in the parametrize table
            raise AssertionError(f"unhandled fault_kind {fault_kind!r}")

        caplog.set_level("INFO")
        with pytest.raises(CommandError) as excinfo:
            call_command("scheduled_refresh", config=tmp_path / "universe.yml", stdout=StringIO())

    assert SEC_SENTINEL_ABS_PATH not in str(excinfo.value)
    assert str(tmp_path) not in str(excinfo.value)
    cause = excinfo.value.__cause__
    assert cause is not None
    assert SEC_SENTINEL_ABS_PATH not in str(cause)
    assert str(tmp_path) not in str(cause)
    assert SEC_SENTINEL_ABS_PATH not in caplog.text
    assert str(tmp_path) not in caplog.text

    parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    assert parent.status == JobRun.Status.FAILED
    assert parent.details["stages"]["sec_fundamentals"]["status"] == JobRun.Status.FAILED
    assert "market" not in parent.details["stages"]
    assert SEC_SENTINEL_ABS_PATH not in str(parent.details)
    assert str(tmp_path) not in str(parent.details)
    assert SEC_SENTINEL_ABS_PATH not in parent.error
    assert str(tmp_path) not in parent.error

    sec_child = JobRun.objects.filter(job_name="sec_fundamentals").order_by("-attempt").first()
    assert sec_child is not None
    assert sec_child.status == JobRun.Status.FAILED
    assert SEC_SENTINEL_ABS_PATH not in sec_child.error
    assert str(tmp_path) not in sec_child.error
    assert SEC_SENTINEL_ABS_PATH not in str(sec_child.details)
    assert str(tmp_path) not in str(sec_child.details)

    record = ProviderRecord.objects.get(provider="sec")
    assert record.status == "error"
    assert SEC_SENTINEL_ABS_PATH not in record.last_error
    assert str(tmp_path) not in record.last_error

    # Zero downstream provider spend: no SEC company (submissions/history/
    # companyfacts) fetch was ever attempted, and the mapping fetch itself
    # was only reached for the fault kinds that occur *after* it (a fresh
    # fetch that then fails to persist), never for kinds that fail before
    # or during resolving an already-registered mapping asset.
    assert submissions_calls == []
    assert companyfacts_calls == []
    assert mapping_calls == ([MAPPING_SUBJECT] if expect_mapping_fetch else [])

    # Removing the fault (the `with monkeypatch.context()` above already
    # reverted it) lets a plain retry recover normally -- zero re-fetch of
    # anything already correctly persisted, and a genuine success for
    # anything that was not.
    retry = sec_jobs_module.execute_sec_fundamentals_job(target_date=prepared.target_date)
    assert retry.status == JobRun.Status.SUCCESS


def test_scheduled_refresh_outcome_tamper_retry_without_refetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A standalone, genuinely-matured `Prediction` (unrelated to the main
    run's own snapshot) is evaluated for real by the command's evaluation
    stage; tampering its persisted `PredictionOutcome` after that real
    success must fail the parent even though every child `JobRun` already
    recorded SUCCESS/SKIPPED, and a corrected retry must recover using the
    exact same completed children/assets with zero additional provider
    fetch/credits -- the outcome-verification analogue of
    `test_verification_failure_after_real_success_fails_closed_without_refetch`.
    """
    prepared, _config, (catalog_calls, price_calls) = _real_prepared(monkeypatch, tmp_path)
    monkeypatch.delenv("STANSTOCK_SCHEDULE_TIMEZONE", raising=False)
    monkeypatch.setattr(
        scheduled_refresh,
        "prepare_us_daily_job",
        lambda **kwargs: prepared,
    )
    monkeypatch.setattr(
        scheduled_refresh,
        "clean_git_revision",
        lambda root: "a" * 40,
    )

    # A standalone prediction, matured well before the main run's own
    # target date, evaluated by the SAME command-level evaluation stage
    # (which evaluates every eligible pending prediction system-wide, not
    # just ones tied to the current run's own snapshot).
    lagging_target = TARGET_DATE - timedelta(days=60)
    lagging = _create_lagging_prediction(target_date=lagging_target)
    _register_lagging_price_history(
        tmp_path=tmp_path,
        subject=lagging.price_subject,
        baseline_date=lagging_target,
        baseline_close=Decimal("100"),
        available_at=DECISION_TIME,
    )

    evaluation_attempts = 0

    def flaky_once_evaluation(**kwargs: object) -> JobRun:
        nonlocal evaluation_attempts
        evaluation_attempts += 1
        run = execute_prediction_evaluation_job(**kwargs)
        if evaluation_attempts == 1:
            # The child's own real work above already committed (including
            # genuinely maturing `lagging`); this failure only forces the
            # *parent's* attempt sequence to stay open, exactly like the
            # unrelated-failure technique `_real_prepared`'s callers use.
            raise ValueError("temporary post-evaluation failure")
        return run

    monkeypatch.setattr(
        scheduled_refresh,
        "execute_prediction_evaluation_job",
        flaky_once_evaluation,
    )

    with pytest.raises(CommandError, match="temporary post-evaluation failure"):
        call_command(
            "scheduled_refresh",
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )
    first_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=1)
    assert first_parent.status == JobRun.Status.FAILED
    assert first_parent.details["stages"]["market"]["status"] == JobRun.Status.SUCCESS
    assert first_parent.details["stages"]["evaluation"]["status"] == JobRun.Status.SUCCESS
    assert "verification" not in first_parent.details
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]

    lagging_outcome = PredictionOutcome.objects.get(prediction=lagging)
    assert lagging_outcome.status == PredictionOutcome.Status.MATURED
    original_resolution = lagging_outcome.resolution
    original_evaluated_at = lagging_outcome.evaluated_at

    # Tamper the already-persisted outcome directly (no re-evaluation, no
    # provider access) -- models a corrupted/forged local row the parent's
    # own verification must independently catch by replay.
    PredictionOutcome.objects.filter(prediction=lagging).update(
        resolution="a fabricated maturity claim"
    )

    with pytest.raises(CommandError, match="Scheduled refresh failed"):
        call_command(
            "scheduled_refresh",
            config=tmp_path / "universe.yml",
            stdout=StringIO(),
        )
    second_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=2)
    assert second_parent.status == JobRun.Status.FAILED
    # Every child stage recovers by reference to its already-committed
    # success -- no re-evaluation and no re-fetch is triggered by the
    # tamper; verification alone catches it.
    assert second_parent.details["stages"]["market"]["status"] == JobRun.Status.SKIPPED
    assert second_parent.details["stages"]["evaluation"]["status"] == JobRun.Status.SKIPPED
    verification = second_parent.details["verification"]
    assert verification["status"] == "failed"
    assert verification["reason_code"] == "evaluation_outcome_replay_mismatch"
    assert "checks" not in verification
    assert not any(str(tmp_path) in str(value) for value in verification.values())
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]
    assert evaluation_attempts == 2
    assert PredictionOutcome.objects.filter(prediction=lagging).count() == 1

    # Restore the exact original (synthetic) evidence -- not a re-fetch --
    # and confirm the retry recovers to a genuinely verified success.
    PredictionOutcome.objects.filter(prediction=lagging).update(resolution=original_resolution)

    call_command(
        "scheduled_refresh",
        config=tmp_path / "universe.yml",
        stdout=StringIO(),
    )
    third_parent = JobRun.objects.get(job_name="scheduled_refresh", attempt=3)
    assert third_parent.status == JobRun.Status.SUCCESS
    assert third_parent.details["stages"]["market"]["status"] == JobRun.Status.SKIPPED
    assert third_parent.details["stages"]["evaluation"]["status"] == JobRun.Status.SKIPPED
    assert third_parent.details["verification"]["status"] == "verified"
    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]
    assert evaluation_attempts == 3
    lagging_outcome.refresh_from_db()
    assert lagging_outcome.resolution == original_resolution
    assert lagging_outcome.evaluated_at == original_evaluated_at

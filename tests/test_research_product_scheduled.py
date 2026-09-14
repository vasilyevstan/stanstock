from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO
from typing import Any
from uuid import uuid4

import pytest
from django.conf import settings as django_settings
from django.core.management import call_command
from django.core.management.base import CommandError
from django.utils import timezone

from stanstock.core import research_product_refresh
from stanstock.core.jobs import JobExecutionResult, execute_target_job
from stanstock.core.models import JobRun
from stanstock.core.refresh_verification import replay_recorded_scheduled_refresh
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.live_us import _persist_price_series
from stanstock.data.models import DataAsset, ProviderRecord
from stanstock.data.provider_policy import BASIC_USAGE_SCOPE
from stanstock.data.providers.exceptions import ProviderError
from stanstock.data.research_product import (
    PRODUCT_INTAKE_KIND,
    PRODUCT_MEMBERSHIP_KIND,
)
from stanstock.data.research_product_jobs import (
    DAILY_RESEARCH_JOB,
    RESEARCH_INTAKE_JOB,
    SCHEDULED_RESEARCH_JOB,
    product_job_name,
)
from stanstock.portfolio.models import TrackedSymbol
from stanstock.portfolio.watchlist import (
    TrackedSymbolValidationError,
    add_tracked_symbol,
    verified_catalog_references_for_symbols,
)
from stanstock.research.jobs import execute_prediction_evaluation_job
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.product_pipeline import CALCULATION_ARTIFACT_KIND
from stanstock.research.refresh_evidence import ANALYSIS_OUTPUT_MANIFEST_KIND
from test_research_product_jobs import (
    NOW,
    TARGET,
    _series,
    make_product_environment,
)

pytestmark = pytest.mark.django_db
REVISION = "a" * 40


@pytest.fixture
def scheduled_environment(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    django_user_model: Any,
    settings: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    owner, store, path, resolve, fetch = make_product_environment(  # type: ignore[no-untyped-call]
        tmp_path,
        monkeypatch,
        django_user_model,
    )
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.OWNER_USERNAME = owner.username
    settings.DATA_DIR = store.root
    monkeypatch.setattr(
        research_product_refresh,
        "clean_git_revision",
        lambda _root: REVISION,
    )
    monkeypatch.setattr(
        "stanstock.research.product_pipeline.clean_git_revision",
        lambda _root: REVISION,
    )
    monkeypatch.setattr(
        "stanstock.core.management.commands.scheduled_refresh.execute_sec_fundamentals_job",
        lambda **_kwargs: pytest.fail("The price-only profile must not invoke SEC"),
    )
    return owner, store, path, resolve, fetch


def _identity(owner: Any) -> dict[str, str]:
    return research_product_refresh.scheduled_identity(owner)


def _parent(owner: Any, *, status: str = JobRun.Status.SUCCESS) -> JobRun:
    return JobRun.objects.get(
        job_name=product_job_name(SCHEDULED_RESEARCH_JOB, _identity(owner)),
        region="us",
        target_date=TARGET,
        status=status,
    )


def _run_command() -> str:
    output = StringIO()
    call_command("scheduled_refresh", stdout=output)
    return output.getvalue()


def test_native_profile_rejects_changed_machine_timezone(
    monkeypatch: pytest.MonkeyPatch,
    settings: Any,
) -> None:
    settings.RESEARCH_PRODUCT_ENABLED = True
    monkeypatch.setenv("STANSTOCK_SCHEDULE_TIMEZONE", "Europe/Tallinn")
    monkeypatch.setattr(
        "stanstock.core.management.commands.scheduled_refresh.detect_iana_timezone",
        lambda: "America/New_York",
    )

    def forbidden_refresh(**_kwargs: object) -> None:
        pytest.fail("Timezone drift must be rejected before native refresh dispatch")

    monkeypatch.setattr(
        research_product_refresh,
        "execute_scheduled_research_refresh",
        forbidden_refresh,
    )
    with pytest.raises(
        CommandError, match=r"Scheduled research refresh failed \(ValueError\)"
    ) as error:
        _run_command()
    assert isinstance(error.value.__cause__, ValueError)
    assert "Reinstall the LaunchAgent" in str(error.value.__cause__)
    assert not JobRun.objects.exists()


def test_native_profile_ignores_legacy_success_and_replays_exact_registered_output(
    scheduled_environment: tuple[Any, Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STANSTOCK_SCHEDULE_TIMEZONE", "Europe/Tallinn")
    monkeypatch.setattr(
        "stanstock.core.management.commands.scheduled_refresh.detect_iana_timezone",
        lambda: "Europe/Tallinn",
    )
    owner, store, _path, resolve, fetch = scheduled_environment
    JobRun.objects.create(
        job_name="scheduled_refresh",
        region="us",
        target_date=TARGET,
        attempt=1,
        status=JobRun.Status.SUCCESS,
        finished_at=NOW,
        details={"legacy": True},
    )

    output = _run_command()

    parent = _parent(owner)
    identity = _identity(owner)
    daily_name = product_job_name(DAILY_RESEARCH_JOB, identity)
    intake_name = product_job_name(RESEARCH_INTAKE_JOB, identity)
    daily = JobRun.objects.get(job_name=daily_name, status=JobRun.Status.SUCCESS)
    intake = JobRun.objects.get(job_name=intake_name, status=JobRun.Status.SUCCESS)
    run = AnalysisRun.objects.get(pk=parent.details["verification"]["analysis_run_id"])
    replayed = replay_recorded_scheduled_refresh(parent)

    assert parent.details["profile"] == "research_product_v1"
    assert parent.details["invocation_identity"] == identity
    assert set(parent.details["stages"]) == {
        "market",
        "evaluation",
        "portfolio_snapshots",
    }
    assert parent.details["stages"]["market"]["job_run_id"] == str(daily.pk)
    assert parent.details["verification"]["child_job_run_ids"]["research_intake"] == str(intake.pk)
    assert replayed.parent.pk == parent.pk
    assert replayed.analysis_run.pk == run.pk
    assert replayed.snapshot.pk == run.universe_snapshot_id
    assert len(replayed.catalog_assets) == 2
    assert run.issued_on_time is True
    assert run.code_revision == REVISION
    assert run.data_cutoff == NOW
    assert StockAnalysis.objects.filter(run=run).count() == 3
    assert Prediction.objects.filter(analysis__run=run).count() == 15
    assert parent.details["verification"]["stock_analysis_count"] == 3
    assert parent.details["verification"]["prediction_count"] == 15
    frequency_verification = parent.details["frequency_verification"]
    assert frequency_verification["status"] == "verified"
    assert frequency_verification["analysis_run_id"] == str(run.id)
    assert DataAsset.objects.filter(
        pk=frequency_verification["frequency_asset_id"],
        kind="research_product_frequency_evidence",
    ).exists()
    assert DataAsset.objects.filter(kind=PRODUCT_INTAKE_KIND).count() == 1
    assert DataAsset.objects.filter(kind=PRODUCT_MEMBERSHIP_KIND).count() == 1
    assert DataAsset.objects.filter(kind=CALCULATION_ARTIFACT_KIND).count() == 3
    assert DataAsset.objects.filter(kind=ANALYSIS_OUTPUT_MANIFEST_KIND).count() == 1
    assert DataAsset.objects.filter(kind="price_history", subject="SPY").count() == 1
    assert parent.details["verification"]["asset_manifest"]["count"] > 0
    assert str(store.root) not in output
    assert "CHEAP" not in output
    assert "details=" not in output
    assert "analyses=3 predictions=15" in output
    references = verified_catalog_references_for_symbols(symbols=("NEW",), store=store)
    assert references["NEW"].symbol == "NEW"
    preference, created = add_tracked_symbol(owner=owner, raw_symbol="NEW", store=store)
    assert created and preference.symbol == "NEW"
    assert Prediction.objects.filter(analysis__run=run).count() == 15
    resolve.assert_not_called()
    fetch.assert_not_called()
    parent.details["verification"]["analysis_run_id"] = str(uuid4())
    parent.save(update_fields=["details"])
    with pytest.raises(TrackedSymbolValidationError, match="unavailable or invalid"):
        verified_catalog_references_for_symbols(symbols=("NEW",), store=store)


def test_completed_new_parent_rechecks_frequency_sibling_before_skip_recovery(
    scheduled_environment: tuple[Any, Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tampered child cannot ride the successful-parent skip shortcut."""

    monkeypatch.setenv("STANSTOCK_SCHEDULE_TIMEZONE", "Europe/Tallinn")
    monkeypatch.setattr(
        "stanstock.core.management.commands.scheduled_refresh.detect_iana_timezone",
        lambda: "Europe/Tallinn",
    )
    owner, _store, _path, resolve, fetch = scheduled_environment
    _run_command()
    parent = _parent(owner)
    child = JobRun.objects.get(pk=parent.details["frequency_verification"]["child_job_run_id"])
    child.details["frequency_asset_sha256"] = "forged"
    child.save(update_fields=["details"])
    resolve.reset_mock()
    fetch.reset_mock()

    with pytest.raises(CommandError, match=r"Scheduled research refresh failed \(ValueError\)"):
        _run_command()

    # The integrity check is entirely local: no provider resolution or fetch
    # happens while refusing a corrupted completed parent.
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_new_parent_cannot_be_made_legacy_by_removing_frequency_bindings(
    scheduled_environment: tuple[Any, Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STANSTOCK_SCHEDULE_TIMEZONE", "Europe/Tallinn")
    monkeypatch.setattr(
        "stanstock.core.management.commands.scheduled_refresh.detect_iana_timezone",
        lambda: "Europe/Tallinn",
    )
    owner, _store, _path, resolve, fetch = scheduled_environment
    _run_command()
    parent = _parent(owner)
    parent.details.pop("frequency_verification")
    parent.details.pop("frequencies")
    parent.save(update_fields=["details"])
    resolve.reset_mock()
    fetch.reset_mock()

    with pytest.raises(CommandError, match=r"Scheduled research refresh failed \(ValueError\)"):
        _run_command()

    resolve.assert_not_called()
    fetch.assert_not_called()


def test_legacy_parent_without_frequency_child_skips_frequency_verification(
    scheduled_environment: tuple[Any, Any, Any, Any, Any],
) -> None:
    owner, store, _path, _resolve, _fetch = scheduled_environment
    parent = JobRun.objects.create(
        job_name=product_job_name(SCHEDULED_RESEARCH_JOB, _identity(owner)),
        region="us",
        target_date=TARGET,
        attempt=1,
        status=JobRun.Status.SUCCESS,
        finished_at=NOW,
        details={"profile": "research_product_v1", "stages": {}},
    )

    research_product_refresh._verify_recorded_frequency_stage(
        parent=parent,
        target_date=TARGET,
        owner_id=_identity(owner)["owner_id"],
        store=store,
    )


def test_manual_daily_product_is_research_only_and_cannot_occupy_scheduled_identity(
    scheduled_environment,
):
    _owner, _store, _path, resolve, fetch = scheduled_environment
    with pytest.raises(CommandError, match="reserved for automation"):
        call_command("daily", "--region", "us", "--issuance-key", "scheduled")
    assert JobRun.objects.count() == 0
    output = StringIO()
    call_command("daily", "--region", "us", stdout=output)
    manual_run = AnalysisRun.objects.get()
    assert manual_run.config_version == "research-product-v1"
    assert manual_run.issued_on_time is False
    assert manual_run.universe_snapshot.grade == "research"
    assert Prediction.objects.filter(analysis__run=manual_run, issued_on_time=True).count() == 0
    assert "research-product-v1" in output.getvalue()
    assert "CHEAP" not in output.getvalue()
    _run_command()
    observed = AnalysisRun.objects.get(issued_on_time=True)
    assert observed.pk != manual_run.pk
    assert observed.universe_snapshot_id != manual_run.universe_snapshot_id
    assert Prediction.objects.count() == 30
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_downstream_failure_still_runs_portfolio_then_recovers_old_output_dirty_and_late(
    scheduled_environment: tuple[Any, Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, store, _path, resolve, fetch = scheduled_environment
    real_evaluation = execute_prediction_evaluation_job
    attempts = 0

    def fail_evaluation_once(**kwargs: Any) -> JobRun:
        nonlocal attempts
        attempts += 1
        if attempts == 1:

            def fail(_run: JobRun) -> JobExecutionResult:
                raise ValueError("private synthetic provider diagnostic")

            return execute_target_job(
                job_name="evaluate_predictions",
                region="us",
                target_date=TARGET,
                task=fail,
            )
        return real_evaluation(**kwargs)

    monkeypatch.setattr(
        research_product_refresh,
        "execute_prediction_evaluation_job",
        fail_evaluation_once,
    )
    with pytest.raises(CommandError) as excinfo:
        _run_command()
    assert "private synthetic provider diagnostic" not in str(excinfo.value)

    first_parent = _parent(owner, status=JobRun.Status.FAILED)
    assert first_parent.details["stages"]["evaluation"]["status"] == JobRun.Status.FAILED
    assert first_parent.details["stages"]["portfolio_snapshots"]["status"] == JobRun.Status.SKIPPED
    run = AnalysisRun.objects.get()
    prediction_ids = set(Prediction.objects.values_list("pk", flat=True))
    asset_ids = set(DataAsset.objects.values_list("pk", flat=True))
    original_daily = JobRun.objects.get(
        job_name=product_job_name(DAILY_RESEARCH_JOB, _identity(owner)),
        status=JobRun.Status.SUCCESS,
    )

    TrackedSymbol.objects.filter(owner=owner).delete()
    TrackedSymbol.objects.create(owner=owner, symbol="NEW")
    ProviderRecord.objects.filter(provider="twelve_data").update(enabled=False)
    late_time = datetime(2026, 9, 14, 14, tzinfo=UTC)
    monkeypatch.setattr(timezone, "now", lambda: late_time)
    monkeypatch.setattr(
        research_product_refresh,
        "clean_git_revision",
        lambda _root: pytest.fail("Completed recovery must not inspect current cleanliness"),
    )

    output = _run_command()

    second_parent = _parent(owner)
    market_skip = JobRun.objects.get(pk=second_parent.details["stages"]["market"]["job_run_id"])
    assert market_skip.status == JobRun.Status.SKIPPED
    assert market_skip.details["successful_run_id"] == str(original_daily.pk)
    assert second_parent.details["verification"]["analysis_run_id"] == str(run.pk)
    assert second_parent.details["code_revision"] == REVISION
    assert set(Prediction.objects.values_list("pk", flat=True)) == prediction_ids
    assert set(DataAsset.objects.values_list("pk", flat=True)) == asset_ids
    assert "analyses=3 predictions=15" in output
    assert attempts == 2
    resolve.assert_not_called()
    fetch.assert_not_called()

    third_output = _run_command()
    no_op = JobRun.objects.filter(
        job_name=second_parent.job_name,
        target_date=TARGET,
        status=JobRun.Status.SKIPPED,
    ).latest("attempt")
    assert no_op.details["successful_run_id"] == str(second_parent.pk)
    assert "status=skipped" in third_output
    assert set(DataAsset.objects.values_list("pk", flat=True)) == asset_ids
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_parent_replay_rejects_forged_child_and_corrupt_calculation_bytes(
    scheduled_environment: tuple[Any, Any, Any, Any, Any],
) -> None:
    owner, store, _path, resolve, fetch = scheduled_environment
    _run_command()
    parent = _parent(owner)
    original_details = parent.details
    forged = {
        **original_details,
        "stages": {
            **original_details["stages"],
            "market": {
                **original_details["stages"]["market"],
                "job_run_id": str(uuid4()),
            },
        },
    }
    JobRun.objects.filter(pk=parent.pk).update(details=forged)

    with pytest.raises(RefreshVerificationError):
        replay_recorded_scheduled_refresh(parent)

    JobRun.objects.filter(pk=parent.pk).update(details=original_details)
    calculation = DataAsset.objects.filter(kind=CALCULATION_ARTIFACT_KIND).first()
    assert calculation is not None
    store.resolve(calculation.relative_path).write_bytes(b"synthetic corruption")
    with pytest.raises((RefreshVerificationError, ValueError)):
        replay_recorded_scheduled_refresh(parent)
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_scheduled_owner_never_falls_back_to_an_arbitrary_active_user(
    scheduled_environment: tuple[Any, Any, Any, Any, Any],
    django_user_model: Any,
) -> None:
    _owner, _store, _path, resolve, fetch = scheduled_environment
    django_user_model.objects.create_user(username="unrelated-active")
    django_settings.OWNER_USERNAME = "missing-configured-owner"

    with pytest.raises(CommandError):
        _run_command()

    assert not JobRun.objects.exists()
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_basic_policy_uses_licensed_owner_not_configured_username(
    scheduled_environment: tuple[Any, Any, Any, Any, Any],
) -> None:
    owner, _store, _path, resolve, fetch = scheduled_environment
    record = ProviderRecord.objects.get(provider="twelve_data")
    record.usage_scope = BASIC_USAGE_SCOPE
    record.metadata.update(
        plan="basic",
        licensed_user_id=str(owner.pk),
        personal_noncommercial_confirmed=True,
    )
    record.save(update_fields=["usage_scope", "metadata"])
    django_settings.OWNER_USERNAME = "not-the-licensed-user"

    assert research_product_refresh.resolve_scheduled_owner().pk == owner.pk
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_fresh_automatic_issuance_fails_closed_after_deadline(
    scheduled_environment: tuple[Any, Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _store, _path, resolve, fetch = scheduled_environment
    monkeypatch.setattr(
        timezone,
        "now",
        lambda: datetime(2026, 9, 14, 14, tzinfo=UTC),
    )
    monkeypatch.setattr(
        research_product_refresh,
        "clean_git_revision",
        lambda _root: pytest.fail("Late fresh work must fail before a Git check"),
    )

    with pytest.raises(CommandError):
        _run_command()

    parent = _parent(owner, status=JobRun.Status.FAILED)
    assert parent.details["stages"] == {}
    assert not DataAsset.objects.filter(kind=PRODUCT_INTAKE_KIND).exists()
    assert not JobRun.objects.filter(job_name__startswith=f"{DAILY_RESEARCH_JOB}:").exists()
    resolve.assert_not_called()
    fetch.assert_not_called()


def test_real_market_failure_blocks_both_downstream_children(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    django_user_model: Any,
    settings: Any,
) -> None:
    owner, store, _path, resolve, fetch = make_product_environment(  # type: ignore[no-untyped-call]
        tmp_path,
        monkeypatch,
        django_user_model,
        omit_benchmark=True,
    )
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.OWNER_USERNAME = owner.username
    settings.DATA_DIR = store.root
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    monkeypatch.setattr(
        research_product_refresh,
        "clean_git_revision",
        lambda _root: REVISION,
    )
    monkeypatch.setattr(
        "stanstock.research.product_pipeline.clean_git_revision",
        lambda _root: REVISION,
    )
    resolve.side_effect = None
    resolve.return_value = "synthetic-test-token"
    fetch.side_effect = ProviderError("private synthetic transport diagnostic")

    with pytest.raises(CommandError) as excinfo:
        _run_command()

    parent = _parent(owner, status=JobRun.Status.FAILED)
    assert "private synthetic transport diagnostic" not in str(excinfo.value)
    assert set(parent.details["stages"]) == {"market"}
    assert parent.details["stages"]["market"]["status"] == JobRun.Status.FAILED
    assert not JobRun.objects.filter(job_name="evaluate_predictions").exists()
    assert not JobRun.objects.filter(job_name="scheduled_portfolio_snapshots").exists()
    resolve.assert_called_once()
    fetch.assert_called_once()

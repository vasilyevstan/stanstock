from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.utils import timezone
from exchange_calendars import get_calendar

from stanstock.core import research_product_refresh
from stanstock.core.models import JobRun
from stanstock.core.revision import clean_git_revision
from stanstock.data.models import DataAsset
from stanstock.simulation import paper_jobs, paper_v2
from test_current_method_paper import _capture_prices
from test_current_method_paper_v2 import (
    _activate,
    _assert_no_generic_rows,
    _capture,
    _days,
    _files,
)
from test_current_method_paper_v2 import (
    environment as environment,
)

pytestmark = pytest.mark.django_db


def _all_capture_prices(activation, store, monkeypatch, session):
    # Retain all admitted decision sources for the normal market attempt too.
    # This changes synthetic fixture setup, NOT the activated selection.
    fake = copy.deepcopy(activation.document)
    fake["buys"] = [{"listing_id": row["listing_id"]} for row in fake["evidence"]]
    _capture_prices(paper_v2.PaperArtifact(activation.asset, fake), store, monkeypatch, session)


def test_not_due_and_outside_horizon_write_zero_and_never_check_fresh_revision(
    environment, monkeypatch
):
    owner, store, _run = environment
    activation = _activate(environment)
    before = (DataAsset.objects.count(), JobRun.objects.count(), _files(store))
    monkeypatch.setattr(
        paper_v2,
        "clean_git_revision",
        lambda _root: pytest.fail("Not-due paper work must not inspect fresh revision"),
    )
    for session, state in (
        (_days(activation)[0], "not_due"),
        (_days(activation)[-1] + timedelta(days=7), "outside_horizon"),
    ):
        result = paper_jobs.execute_paper_capture_job(owner=owner, session=session, store=store)
        assert result.state == state and result.child is None and result.capture is None
    assert (DataAsset.objects.count(), JobRun.objects.count(), _files(store)) == before


def test_unactivated_disposition_does_not_discover_a_later_activation(environment):
    owner, store, _run = environment
    with CaptureQueriesContext(connection) as queries:
        invocation = paper_jobs.prepare_paper_capture(
            owner=owner, session=timezone.now().date(), store=store
        )
    assert len(queries) == 1 and invocation.result.state == "unactivated"
    _activate(environment)  # Separate explicit action, not refresh auto-activation.
    before = (DataAsset.objects.count(), JobRun.objects.count(), _files(store))
    with CaptureQueriesContext(connection) as queries:
        invocation.recover()
        result = invocation.after_market(attempted=True)
    assert not queries
    assert result.state == "unactivated"
    assert (DataAsset.objects.count(), JobRun.objects.count(), _files(store)) == before


@pytest.mark.parametrize("status", ("success", "failed", "running"))
def test_after_deadline_recovery_preserves_exact_original_job_bytes_revision(
    environment, monkeypatch, status
):
    owner, store, _run = environment
    activation = _activate(environment)
    day = _days(activation)[0]
    first = _capture(environment, activation, monkeypatch, day)
    original = copy.deepcopy(first.child.details)
    origin_time = first.capture.document["recorded_at"]
    origin_revision = first.capture.document["execution_revision"]
    JobRun.objects.filter(pk=first.child.pk).update(
        status=status,
        finished_at=None if status == "running" else timezone.now(),
        error="" if status == "success" else "synthetic interrupted job finish",
    )
    before = (DataAsset.objects.count(), JobRun.objects.count(), _files(store))
    late = get_calendar("XNYS").session_open(get_calendar("XNYS").next_session(day)).to_pydatetime()
    monkeypatch.setattr(timezone, "now", lambda: late + timedelta(days=5))
    monkeypatch.setattr(
        paper_v2,
        "clean_git_revision",
        lambda _root: pytest.fail("Committed recovery cannot require current cleanliness"),
    )
    retry = paper_jobs.execute_paper_capture_job(owner=owner, session=day, store=store)
    assert retry.state == "recovered"
    assert retry.child.pk == first.child.pk
    assert retry.capture.asset.pk == first.capture.asset.pk
    assert retry.capture.document["recorded_at"] == origin_time
    assert retry.capture.document["execution_revision"] == origin_revision
    assert retry.child.details == original
    assert (DataAsset.objects.count(), JobRun.objects.count(), _files(store)) == before
    output = StringIO()
    call_command(
        "paper_portfolio",
        owner=owner.username,
        policy_version=paper_v2.PAPER_POLICY_VERSION,
        session=day,
        stdout=output,
    )
    assert json.loads(output.getvalue())["captured_count"] == 1
    assert (DataAsset.objects.count(), JobRun.objects.count(), _files(store)) == before


@pytest.mark.parametrize(
    "field",
    (
        "missing",
        "identity",
        "session",
        "measurement",
        "execution_revision",
        "origin_started_at",
        "origin_target",
    ),
)
def test_missing_or_inconsistent_atomic_origin_never_reconstructed(environment, monkeypatch, field):
    owner, store, _run = environment
    activation = _activate(environment)
    day = _days(activation)[0]
    result = _capture(environment, activation, monkeypatch, day)
    job = JobRun.objects.get(pk=result.child.pk)
    binding = copy.deepcopy(job.details["paper_v2_origin"])
    if field == "missing":
        job.details.pop("paper_v2_origin")
    elif field == "identity":
        binding["identity"]["source_identity"]["provider"] = "another_provider"
    elif field == "session":
        binding["session"] = _days(activation)[1].isoformat()
    elif field == "measurement":
        binding["measurement"]["id"] = binding["execution"]["id"]
    elif field == "execution_revision":
        binding["execution_revision"] = "b" * 40
    elif field == "origin_started_at":
        job.started_at = timezone.now() + timedelta(hours=1)
    elif field == "origin_target":
        job.target_date = _days(activation)[1]
    if field != "missing":
        job.details["paper_v2_origin"] = binding
    job.save(update_fields=["details", "started_at", "target_date"])
    before = (DataAsset.objects.count(), JobRun.objects.count(), _files(store))
    with pytest.raises(ValueError):
        paper_v2.read_current_method_paper(user=owner, store=store)
    retry = paper_jobs.execute_paper_capture_job(owner=owner, session=day, store=store)
    assert retry.state == "failed"
    assert (DataAsset.objects.count(), JobRun.objects.count(), _files(store)) == before


def test_successful_job_status_without_registered_assets_is_not_capture_proof(
    environment, monkeypatch
):
    owner, store, _run = environment
    activation = _activate(environment)
    day = _days(activation)[0]
    _capture_prices(activation, store, monkeypatch, day)
    JobRun.objects.create(
        job_name=paper_v2.capture_job_name(paper_v2.capture_identity(activation)),
        region="us",
        target_date=day,
        status="success",
        finished_at=timezone.now(),
        details={"diagnostic_only": True},
    )
    before = (JobRun.objects.count(), _files(store))
    result = paper_jobs.execute_paper_capture_job(owner=owner, session=day, store=store)
    assert result.state == "failed" and result.capture_count == 0
    assert (JobRun.objects.count(), _files(store)) == before
    assert not DataAsset.objects.filter(kind=paper_v2.EXECUTION_KIND).exists()
    assert not DataAsset.objects.filter(kind=paper_v2.MEASUREMENT_KIND).exists()


@pytest.mark.parametrize("phase", ("after_proof", "after_registration"))
def test_deadline_recheck_and_atomic_rollback_never_leave_capture_or_origin(
    environment, monkeypatch, phase
):
    owner, store, _run = environment
    activation = _activate(environment)
    day = _days(activation)[0]
    _capture_prices(activation, store, monkeypatch, day)
    before_files = _files(store)
    captured_at = timezone.now()
    deadline = (
        get_calendar("XNYS").session_open(get_calendar("XNYS").next_session(day)).to_pydatetime()
    )
    real_revision = paper_v2.clean_git_revision(Path("."))
    observed_registration = []
    if phase == "after_proof":

        def revision(_root):
            monkeypatch.setattr(timezone, "now", lambda: deadline)
            return real_revision

        monkeypatch.setattr(paper_v2, "clean_git_revision", revision)
    else:
        # Fixture clock, not a persistence/reader mock: the writer's real rows
        # exist in its uncommitted transaction when the final clock is sampled.
        def clock():
            present = DataAsset.objects.filter(kind=paper_v2.MEASUREMENT_KIND).exists()
            if present:
                observed_registration.append(True)
            return deadline if present else captured_at

        monkeypatch.setattr(timezone, "now", clock)
    result = paper_jobs.execute_paper_capture_job(owner=owner, session=day, store=store)
    assert result.state == "failed" and result.reason_code == "paper_v2_deadline_missed"
    assert result.child.status == "failed"
    assert "paper_v2_origin" not in result.child.details
    assert not DataAsset.objects.filter(kind=paper_v2.EXECUTION_KIND).exists()
    assert not DataAsset.objects.filter(kind=paper_v2.MEASUREMENT_KIND).exists()
    assert _files(store) == before_files
    assert bool(observed_registration) is (phase == "after_registration")


def test_actual_clean_revision_guard_refuses_fresh_uncommitted_publication(
    environment, monkeypatch, settings, tmp_path
):
    owner, store, _run = environment
    # A real non-checkout is not a committed execution revision. This negative
    # invokes the actual guard rather than returning a fixture revision.
    settings.BASE_DIR = tmp_path
    monkeypatch.setattr(paper_v2, "clean_git_revision", clean_git_revision)
    before = (DataAsset.objects.count(), _files(store))
    with pytest.raises(ValueError, match="Git revision validation failed"):
        paper_v2.activate_current_method_paper(user=owner, store=store)
    assert (DataAsset.objects.count(), _files(store)) == before


@pytest.mark.parametrize("when", ("close", "naive", "next_open", "late"))
def test_explicit_clock_edges_and_naive_publication_are_refused(when):
    session = datetime(2026, 9, 14, tzinfo=UTC).date()
    calendar = get_calendar("XNYS")
    close = calendar.session_close(session).to_pydatetime()
    deadline = calendar.session_open(calendar.next_session(session)).to_pydatetime()
    value = {
        "close": close,
        "naive": close.replace(tzinfo=None),
        "next_open": deadline,
        "late": deadline + timedelta(seconds=1),
    }[when]
    with pytest.raises(paper_v2.PaperV2Error, match="deadline_missed"):
        paper_v2._capture_time(session, value)


def test_future_source_not_admitted_and_missing_entry_never_late_backfilled(
    environment, monkeypatch
):
    owner, store, _run = environment
    activation = _activate(environment)
    day = _days(activation)[0]
    _capture_prices(activation, store, monkeypatch, day)
    source_clock = timezone.now()
    monkeypatch.setattr(timezone, "now", lambda: source_clock - timedelta(minutes=30))
    missing = paper_jobs.execute_paper_capture_job(owner=owner, session=day, store=store)
    assert missing.state == "failed"
    assert not DataAsset.objects.filter(kind=paper_v2.EXECUTION_KIND).exists()
    next_day = _days(activation)[1]
    _capture_prices(activation, store, monkeypatch, next_day)
    late_entry = paper_jobs.execute_paper_capture_job(owner=owner, session=day, store=store)
    assert late_entry.state == "failed"
    missing_entry = paper_jobs.execute_paper_capture_job(owner=owner, session=next_day, store=store)
    assert missing_entry.state == "failed"
    assert not DataAsset.objects.filter(kind=paper_v2.MEASUREMENT_KIND).exists()


def test_opted_refresh_fresh_capture_is_outside_parent_proof_and_count_only(
    environment, monkeypatch
):
    owner, store, _run = environment
    activation = _activate(environment)
    session = _days(activation)[0]
    _all_capture_prices(activation, store, monkeypatch, session)
    monkeypatch.setattr(research_product_refresh, "clean_git_revision", lambda _root: "a" * 40)
    output = StringIO()
    call_command("scheduled_refresh", capture_paper=True, stdout=output)
    text = output.getvalue()
    assert "paper_capture status=captured captures=1 child_status=success reason=" in text
    parent = JobRun.objects.get(
        job_name=research_product_refresh.product_job_name(
            research_product_refresh.SCHEDULED_RESEARCH_JOB,
            research_product_refresh.scheduled_identity(owner),
        ),
        target_date=session,
        status="success",
    )
    assert set(parent.details["stages"]) == {"market", "evaluation", "portfolio_snapshots"}
    assert "paper_capture" not in parent.details
    assert "paper_v2_origin" not in parent.details
    child = JobRun.objects.get(job_name__startswith="paper_v2:", target_date=session)
    state = paper_v2.read_state(user=owner, store=store)
    assert child.details["paper_v2_origin"] == paper_v2._origin_binding(
        activation, state.execution, state.captures[0]
    )
    assert DataAsset.objects.filter(
        pk=state.execution.asset.pk, kind=paper_v2.EXECUTION_KIND
    ).exists()
    assert DataAsset.objects.filter(
        pk=state.captures[0].asset.pk, kind=paper_v2.MEASUREMENT_KIND
    ).exists()
    for private in ("AAPL", "MSFT", "CHEAP", str(store.root), "http", "holdings", "details="):
        assert private not in text
    before = (DataAsset.objects.count(), _files(store))
    repeated = StringIO()
    call_command("scheduled_refresh", capture_paper=True, stdout=repeated)
    assert "status=skipped" in repeated.getvalue()
    assert "paper_capture status=recovered captures=1" in repeated.getvalue()
    assert JobRun.objects.filter(job_name__startswith="paper_v2:").count() == 1
    assert (DataAsset.objects.count(), _files(store)) == before
    _assert_no_generic_rows()


def test_optional_fresh_paper_failure_cannot_prevent_mandatory_parent_children(
    environment, monkeypatch
):
    owner, store, _run = environment
    activation = _activate(environment)
    session = _days(activation)[0]
    _all_capture_prices(activation, store, monkeypatch, session)
    revision = paper_v2.clean_git_revision(Path("."))
    deadline = (
        get_calendar("XNYS")
        .session_open(get_calendar("XNYS").next_session(session))
        .to_pydatetime()
    )
    monkeypatch.setattr(research_product_refresh, "clean_git_revision", lambda _root: "a" * 40)

    def paper_revision(_root):
        monkeypatch.setattr(timezone, "now", lambda: deadline)
        return revision

    monkeypatch.setattr(paper_v2, "clean_git_revision", paper_revision)
    output = StringIO()
    call_command("scheduled_refresh", capture_paper=True, stdout=output)
    assert "profile=research_product_v1 status=success" in output.getvalue()
    assert "paper_capture status=failed captures=0 child_status=failed" in output.getvalue()
    assert "reason=paper_v2_deadline_missed" in output.getvalue()
    parent = JobRun.objects.get(
        job_name=research_product_refresh.product_job_name(
            research_product_refresh.SCHEDULED_RESEARCH_JOB,
            research_product_refresh.scheduled_identity(owner),
        ),
        target_date=session,
        status="success",
    )
    assert parent.details["stages"]["evaluation"]["status"] == "success"
    assert parent.details["stages"]["portfolio_snapshots"]["status"] == "skipped"
    assert "paper_capture" not in parent.details
    assert not DataAsset.objects.filter(kind=paper_v2.MEASUREMENT_KIND).exists()


def test_expected_market_failure_still_captures_retained_paper_sources_without_fetch(
    environment, monkeypatch
):
    owner, store, _run = environment
    activation = _activate(environment)
    session = _days(activation)[0]
    _capture_prices(activation, store, monkeypatch, session)
    monkeypatch.setattr(research_product_refresh, "clean_git_revision", lambda _root: "a" * 40)
    from stanstock.data.provider_policy import ProviderConfigurationError
    from stanstock.data.providers import twelve_data

    def missing_parent_credentials():
        raise ProviderConfigurationError("private synthetic credential text")

    monkeypatch.setattr(twelve_data, "resolve_api_key", missing_parent_credentials)
    output = StringIO()
    with pytest.raises(CommandError) as failure:
        call_command("scheduled_refresh", capture_paper=True, stdout=output)
    assert "private synthetic credential text" not in str(failure.value)
    assert "paper_capture status=captured captures=1 child_status=success" in output.getvalue()
    assert "AAPL" not in output.getvalue() and str(store.root) not in output.getvalue()
    assert DataAsset.objects.filter(kind=paper_v2.EXECUTION_KIND).count() == 1
    assert DataAsset.objects.filter(kind=paper_v2.MEASUREMENT_KIND).count() == 1
    state = paper_v2.read_state(user=owner, store=store)
    paper_v2._verify_origin(activation, state.execution, state.captures[0])


def test_recovery_only_pre_market_before_parent_credentials_and_no_second_attempt(
    environment, monkeypatch
):
    owner, store, _run = environment
    activation = _activate(environment)
    session = _days(activation)[0]
    first = _capture(environment, activation, monkeypatch, session)
    before = (DataAsset.objects.count(), _files(store))
    # CHEAP intentionally has no session history. The ordinary market attempt
    # will ask for credentials; paper must already have recovered locally.
    monkeypatch.setattr(research_product_refresh, "clean_git_revision", lambda _root: "a" * 40)
    query_count_at_credentials = []
    from stanstock.data.providers import twelve_data

    with CaptureQueriesContext(connection) as queries:

        def unavailable_credentials(*_args, **_kwargs):
            query_count_at_credentials.append(len(queries))
            from stanstock.data.provider_policy import ProviderConfigurationError

            raise ProviderConfigurationError("private synthetic credential text")

        monkeypatch.setattr(twelve_data, "resolve_api_key", unavailable_credentials)
        with pytest.raises(research_product_refresh.ScheduledPaperRefreshError) as failure:
            research_product_refresh.execute_scheduled_research_refresh(
                store=store, capture_paper=True, enforce_rate_limit=False
            )
    paper_result = failure.value.paper_capture
    assert paper_result.state == "recovered" and paper_result.child.pk == first.child.pk
    assert paper_result.capture.asset.pk == first.capture.asset.pk
    assert query_count_at_credentials
    before_credentials = [query["sql"] for query in queries[: query_count_at_credentials[0]]]
    assert any(
        "paper_v2_origin" in sql or first.child.pk.hex in sql or str(first.child.pk) in sql
        for sql in before_credentials
    )
    assert DataAsset.objects.filter(kind=paper_v2.MEASUREMENT_KIND).count() == 1
    assert JobRun.objects.filter(job_name__startswith="paper_v2:").count() == 1
    # A failed ordinary market attempt may capture its intake, but may not
    # create another paper artifact or change the original bytes.
    assert all(_files(store)[path] == payload for path, payload in before[1].items())


def test_historical_parent_clock_cannot_make_a_late_fresh_capture_on_time(environment, monkeypatch):
    owner, store, _run = environment
    activation = _activate(environment)
    session = _days(activation)[0]
    _all_capture_prices(activation, store, monkeypatch, session)
    historical_parent_time = timezone.now()
    deadline = (
        get_calendar("XNYS")
        .session_open(get_calendar("XNYS").next_session(session))
        .to_pydatetime()
    )
    monkeypatch.setattr(timezone, "now", lambda: deadline)
    monkeypatch.setattr(research_product_refresh, "clean_git_revision", lambda _root: "a" * 40)
    with pytest.raises(research_product_refresh.ScheduledPaperRefreshError) as failure:
        research_product_refresh.execute_scheduled_research_refresh(
            decision_time=historical_parent_time,
            store=store,
            capture_paper=True,
            enforce_rate_limit=False,
        )
    assert failure.value.paper_capture.state == "withheld"
    assert failure.value.paper_capture.reason_code == "paper_v2_deadline_missed"
    assert not DataAsset.objects.filter(kind=paper_v2.EXECUTION_KIND).exists()
    assert not DataAsset.objects.filter(kind=paper_v2.MEASUREMENT_KIND).exists()


def test_real_disposable_database_concurrent_cli_refresh_and_activation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tests" / "test_current_method_paper_v2.py"),
            "concurrency",
            str(tmp_path),
        ],
        cwd=root,
        env={
            "DJANGO_SETTINGS_MODULE": "stanstock.settings.test",
            "PYTHONPATH": str(root / "src") + os.pathsep + str(root / "tests"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == {
        "activations": 1,
        "executions": 1,
        "captures": 1,
        "origins": 1,
        "distinct_clocks": True,
        "same_activation": True,
        "same_capture": True,
        "before_commit_neither": True,
        "after_commit_both": True,
        "lock_wait_deadline_refused": True,
        "no_generic_rows": True,
    }


def test_disabled_refresh_has_zero_paper_imports_queries_jobs_and_files_cold(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = """
import json, sys
from io import StringIO
from pathlib import Path
from django.conf import settings
settings.DATABASES["default"]["NAME"] = str(Path(sys.argv[1]) / "cold.sqlite3")
settings.DATA_DIR = Path(sys.argv[1]) / "assets"
import django
django.setup()
import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.test.utils import CaptureQueriesContext
from stanstock.core import research_product_refresh
from stanstock.core.models import JobRun
from stanstock.core.refresh_verification import replay_recorded_scheduled_refresh
from stanstock.data.live_us import _persist_price_series
from stanstock.data.models import DataAsset
from test_research_product_jobs import make_product_environment, _series
call_command("migrate", verbosity=0, stdout=StringIO())
patch = pytest.MonkeyPatch()
owner, store, path, resolve, fetch = make_product_environment(
    Path(sys.argv[1]), patch, get_user_model()
)
settings.RESEARCH_PRODUCT_ENABLED = True
settings.DEMO_MODE = False
settings.OWNER_USERNAME = owner.username
settings.DATA_DIR = store.root
_persist_price_series(store=store, series=_series("CHEAP"), listing=None)
patch.setattr(research_product_refresh, "clean_git_revision", lambda _root: "a" * 40)
patch.setattr("stanstock.research.product_pipeline.clean_git_revision", lambda _root: "a" * 40)
assert not any(name.startswith("stanstock.simulation.paper") for name in sys.modules)
with CaptureQueriesContext(connection) as queries:
    call_command("scheduled_refresh", stdout=StringIO())
parent = JobRun.objects.get(job_name__startswith="scheduled_refresh_research_v1", status="success")
replayed = replay_recorded_scheduled_refresh(parent)
settings.RESEARCH_PRODUCT_ENABLED = False
try:
    call_command("scheduled_refresh", capture_paper=True, stdout=StringIO())
except CommandError:
    pass
else:
    raise AssertionError("Legacy paper flag was accepted")
print(json.dumps({
    "paper_imports": [
        name for name in sys.modules if name.startswith("stanstock.simulation.paper")
    ],
    "paper_queries": sum("paper_portfolio_v2" in row["sql"] for row in queries),
    "paper_jobs": JobRun.objects.filter(job_name__startswith="paper_v2:").count(),
    "paper_assets": DataAsset.objects.filter(kind__startswith="paper_portfolio_v2").count(),
    "paper_files": (store.root / "simulations" / "paper-v2").exists(),
    "provider_calls": resolve.call_count + fetch.call_count,
    "parent_verified": replayed.parent.pk == parent.pk,
}))
patch.undo()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=root,
        env={
            "DJANGO_SETTINGS_MODULE": "stanstock.settings.test",
            "PYTHONPATH": str(root / "src") + os.pathsep + str(root / "tests"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == {
        "paper_imports": [],
        "paper_queries": 0,
        "paper_jobs": 0,
        "paper_assets": 0,
        "paper_files": False,
        "provider_calls": 0,
        "parent_verified": True,
    }

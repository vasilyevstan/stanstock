"""Focused tests for the SYNTHETIC-ONLY `refresh_demo` vertical-flow command.

`refresh_demo` calls `seed_demo`, resolves the research-grade synthetic
universe snapshot, resolves+validates `target_date` (defaulting to
`snapshot.as_of_date`, an actually-observed session in the synthetic
benchmark's price history), and wraps only `analyze_snapshot` in
`execute_target_job` so a target_date that already succeeded is
idempotently skipped.

These tests prove: (1) one run at the default target creates the expected
60 analyses / 180 predictions, (2) the resolved snapshot stays
research-grade (never observed), (3) rerunning the same (default) target
after a success creates no additional analyses/predictions -- it is
skipped, not re-executed, (4) a different, actually-observed prior session
runs again independently, and (5) both an after-as_of_date target and a
non-observed (weekend) target before as_of_date are rejected explicitly
rather than fabricating a session.

The synthetic snapshot is deterministically dated 2026-09-04 (a Friday,
NYSE trading session); 2026-09-03 (Thursday) is the actually-observed prior
session used for the "different valid target" tests, and 2026-08-29
(Saturday) is a real weekend date within the synthetic history's range that
was never generated as a session, used for the "reject non-observed date"
test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import override_settings

from stanstock.core.models import JobRun
from stanstock.data.models import UniverseSnapshot
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis

pytestmark = pytest.mark.django_db

SNAPSHOT_AS_OF_DATE = "2026-09-04"
PRIOR_OBSERVED_SESSION = "2026-09-03"
UNOBSERVED_WEEKEND_DATE = "2026-08-29"
AFTER_AS_OF_DATE = "2026-09-06"


def test_refresh_demo_creates_expected_analyses_and_predictions(tmp_path: Path) -> None:
    with override_settings(DATA_DIR=tmp_path):
        call_command("refresh_demo")

    assert AnalysisRun.objects.count() == 1
    assert StockAnalysis.objects.count() == 60
    assert Prediction.objects.count() == 180

    job_run = JobRun.objects.get()
    assert job_run.job_name == "refresh_demo"
    assert job_run.region == "all"
    assert job_run.status == JobRun.Status.SUCCESS
    assert job_run.target_date.isoformat() == SNAPSHOT_AS_OF_DATE
    assert job_run.details["analyses"] == 60
    assert job_run.details["predictions"] == 180
    assert job_run.details["provider"] == "synthetic_demo"
    assert job_run.details["benchmark_subject"] == "ZZBENCH01"


def test_refresh_demo_default_target_date_is_snapshot_as_of_date(tmp_path: Path) -> None:
    with override_settings(DATA_DIR=tmp_path):
        call_command("refresh_demo")

    snapshot = UniverseSnapshot.objects.get()
    assert snapshot.as_of_date.isoformat() == SNAPSHOT_AS_OF_DATE
    job_run = JobRun.objects.get()
    assert job_run.target_date == snapshot.as_of_date


def test_refresh_demo_snapshot_stays_research_grade(tmp_path: Path) -> None:
    with override_settings(DATA_DIR=tmp_path):
        call_command("refresh_demo")

    snapshot = UniverseSnapshot.objects.get()
    assert snapshot.grade == UniverseSnapshot.Grade.RESEARCH
    assert snapshot.grade != UniverseSnapshot.Grade.OBSERVED

    job_run = JobRun.objects.get()
    assert job_run.details["snapshot_grade"] == UniverseSnapshot.Grade.RESEARCH
    assert job_run.details["snapshot_id"] == str(snapshot.pk)


def test_refresh_demo_repeated_successful_target_is_skipped_without_new_rows(
    tmp_path: Path,
) -> None:
    with override_settings(DATA_DIR=tmp_path):
        call_command("refresh_demo")

        counts_first = (
            AnalysisRun.objects.count(),
            StockAnalysis.objects.count(),
            Prediction.objects.count(),
        )
        assert counts_first == (1, 60, 180)

        call_command("refresh_demo")

    counts_second = (
        AnalysisRun.objects.count(),
        StockAnalysis.objects.count(),
        Prediction.objects.count(),
    )
    assert counts_second == counts_first

    job_runs = list(JobRun.objects.order_by("started_at"))
    assert [run.status for run in job_runs] == [JobRun.Status.SUCCESS, JobRun.Status.SKIPPED]
    assert job_runs[1].details["reason"] == "target_already_succeeded"
    assert job_runs[1].details["successful_run_id"] == str(job_runs[0].pk)


def test_refresh_demo_different_observed_prior_session_runs_again(tmp_path: Path) -> None:
    with override_settings(DATA_DIR=tmp_path):
        call_command("refresh_demo")
        call_command("refresh_demo", "--target-date", PRIOR_OBSERVED_SESSION)

    # A different, actually-observed target_date is a different idempotency
    # key, so this is a second genuine run: another AnalysisRun/StockAnalysis
    # /Prediction batch, against the very same (still research-grade,
    # still-not-deleted) snapshot.
    assert AnalysisRun.objects.count() == 2
    assert StockAnalysis.objects.count() == 120
    assert Prediction.objects.count() == 360
    assert UniverseSnapshot.objects.count() == 1
    job_runs = list(JobRun.objects.order_by("started_at"))
    assert [run.status for run in job_runs] == [JobRun.Status.SUCCESS, JobRun.Status.SUCCESS]
    assert {run.target_date.isoformat() for run in job_runs} == {
        SNAPSHOT_AS_OF_DATE,
        PRIOR_OBSERVED_SESSION,
    }


def test_refresh_demo_rejects_target_after_snapshot_as_of_date(tmp_path: Path) -> None:
    with override_settings(DATA_DIR=tmp_path):
        with pytest.raises(CommandError, match="is after the synthetic snapshot's as_of_date"):
            call_command("refresh_demo", "--target-date", AFTER_AS_OF_DATE)

    # No job/analysis rows must be created for a rejected target.
    assert JobRun.objects.count() == 0
    assert AnalysisRun.objects.count() == 0


def test_refresh_demo_rejects_target_not_an_observed_benchmark_session(tmp_path: Path) -> None:
    with override_settings(DATA_DIR=tmp_path):
        with pytest.raises(CommandError, match="not an observed"):
            call_command("refresh_demo", "--target-date", UNOBSERVED_WEEKEND_DATE)

    # No job/analysis rows must be created for a rejected (fabricated) target.
    assert JobRun.objects.count() == 0
    assert AnalysisRun.objects.count() == 0

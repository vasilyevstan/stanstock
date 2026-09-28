"""Synthetic database integration of prospective evidence with the native product."""

import hashlib
import inspect
import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from django.utils import timezone
from exchange_calendars import get_calendar

import test_research_price_product_shadow_drift as adapter_tests
from stanstock.core.jobs import JobExecutionResult
from stanstock.core.models import JobRun
from stanstock.data import research_product_jobs
from stanstock.data.assets import asset_ref_for
from stanstock.data.live_us import _persist_price_series
from stanstock.data.models import DataAsset
from stanstock.data.providers import twelve_data
from stanstock.data.research_product import load_product_intake, product_membership_payload
from stanstock.data.research_product_jobs import execute_daily_research_job
from stanstock.research import shadow_jobs as jobs
from stanstock.research import shadow_study as study
from stanstock.research.models import Prediction
from stanstock.research.outcomes import evaluate_prediction
from test_research_product_jobs import NOW, TARGET
from test_research_product_scheduled import scheduled_environment  # noqa: F401

pytestmark = pytest.mark.django_db


@pytest.fixture
def activated_scheduled_environment(scheduled_environment, settings, monkeypatch):  # noqa: F811
    owner, store, path, resolve, fetch = scheduled_environment
    settings.DEMO_MODE = False
    settings.SHADOW_STUDY_ENABLED = True
    clock = [datetime(2026, 9, 11, 19, tzinfo=UTC)]
    monkeypatch.setattr(timezone, "now", lambda: clock[0])
    monkeypatch.setattr(study, "clean_git_revision", lambda root: "a" * 40)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", "a" * 40)
    activation = study.activate_shadow_study(owner=owner, store=store)
    clock[0] = NOW
    return owner, store, activation, clock, resolve, fetch, path


@pytest.fixture
def capture_environment(activated_scheduled_environment):
    owner, store, activation, clock, resolve, fetch, path = activated_scheduled_environment
    execute_daily_research_job(
        target_date=TARGET,
        owner=owner,
        issuance_key="scheduled",
        issued_on_time=True,
        store=store,
        core_config_path=path,
        enforce_rate_limit=False,
        derive_frequencies=False,
    )
    return owner, store, activation, clock, resolve, fetch


def test_committed_acceptance_recovers_publication_with_later_challenge(
    capture_environment,
    monkeypatch,
):
    from django.db.models.query import QuerySet

    owner, store, activation, clock, resolve, fetch = capture_environment
    jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    through, _assets = _future_prices(store, clock)
    _native_outcomes(clock, through)
    register = study._register_shadow_asset
    committed = []

    def crash_after_acceptance(**kwargs):
        asset = register(**kwargs)
        if kwargs["kind"] == study.KINDS[4] and kwargs["document"]["body"]["state"] == "accepted":
            committed.append(asset)
            raise RuntimeError("synthetic crash after first acceptance commit")
        return asset

    with monkeypatch.context() as patch:
        patch.setattr(study, "_register_shadow_asset", crash_after_acceptance)
        with pytest.raises(RuntimeError, match="synthetic crash"):
            jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=through, store=store)
    assert len(committed) == 1
    accepted_asset = committed[0]
    accepted_bytes = store.read_bytes(accepted_asset.relative_path)
    accepted_document = study._read_asset(accepted_asset, store=store)
    protocol = DataAsset.objects.get(kind=study.KINDS[0])
    assert (
        study._witness(
            accepted_asset,
            document=accepted_document,
            protocol=protocol,
            activation=activation,
        )
        is None
    )
    before_recovery = study.read_shadow_study(owner=owner, as_of=clock[0], store=store)
    assert before_recovery.case_counts["accepted"] == 0

    _future_prices(store, clock, terminal_change=1)
    clock[0] += timedelta(days=1)
    assessment_time = clock[0]
    original_update = QuerySet.update

    def finish_later(queryset, **kwargs):
        if queryset.model is JobRun and kwargs.get("status") == JobRun.Status.SUCCESS:
            clock[0] += timedelta(seconds=1)
            kwargs["finished_at"] = clock[0]
        return original_update(queryset, **kwargs)

    calls = (resolve.call_count, fetch.call_count)
    with monkeypatch.context() as patch:
        patch.setattr(QuerySet, "update", finish_later)
        result = jobs.execute_shadow_evaluation_job(
            owner=owner,
            evaluation_date=assessment_time.date(),
            store=store,
        )
    assert result.child.status == JobRun.Status.SUCCESS
    assert result.child.target_date != through
    assert asset_ref_for(accepted_asset).to_json() in result.child.details["assets"]
    assert store.read_bytes(accepted_asset.relative_path) == accepted_bytes
    assert (
        study._witness(
            accepted_asset,
            document=accepted_document,
            protocol=protocol,
            activation=activation,
        )
        == result.child.finished_at
    )
    challenge_asset = next(
        asset
        for asset in DataAsset.objects.filter(kind=study.KINDS[4])
        if study._read_asset(asset, store=store)["body"]["state"] == "challenge"
    )
    challenge = study._read_asset(challenge_asset, store=store)
    assert challenge["body"]["prior_accepted"] == asset_ref_for(accepted_asset).to_json()
    assert (
        study._witness(
            challenge_asset,
            document=challenge,
            protocol=protocol,
            activation=activation,
        )
        == result.child.finished_at
    )
    before_completion = study.read_shadow_study(owner=owner, as_of=assessment_time, store=store)
    assert before_completion.case_counts["accepted"] == 0
    assert before_completion.case_counts["accepted_with_challenge"] == 0
    visible = study.read_shadow_study(owner=owner, as_of=result.child.finished_at, store=store)
    assert visible.case_counts["accepted_with_challenge"] == 1
    assert visible.case_counts["accepted"] == 5
    first = study._read_asset(accepted_asset, store=store)["body"]
    assert first["scores"] == accepted_document["body"]["scores"]
    horizon = next(summary for summary in visible.summaries if summary.horizon == first["horizon"])
    accepted_scores = [
        document["body"]["scores"]
        for asset in DataAsset.objects.filter(kind=study.KINDS[4])
        if (document := study._read_asset(asset, store=store))["body"]["state"] == "accepted"
        and document["body"]["horizon"] == first["horizon"]
    ]
    assert horizon.per_anchor[0].arm_means == study._means(accepted_scores)
    retry = jobs.execute_shadow_evaluation_job(
        owner=owner,
        evaluation_date=assessment_time.date(),
        store=store,
    )
    assert retry.child.status == JobRun.Status.SKIPPED
    assert (resolve.call_count, fetch.call_count) == calls


def test_unpublished_capture_never_enters_a_completed_scan(
    capture_environment,
    monkeypatch,
):
    from django.db.models.query import QuerySet

    owner, store, _activation, clock, resolve, fetch = capture_environment
    original_update = QuerySet.update

    def crash_completion(queryset, **kwargs):
        if queryset.model is JobRun and kwargs.get("status") == JobRun.Status.SUCCESS:
            raise RuntimeError("synthetic capture completion crash")
        return original_update(queryset, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(QuerySet, "update", crash_completion)
        with pytest.raises(RuntimeError, match="synthetic capture"):
            jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    clock[0] += timedelta(seconds=1)
    scan_time = clock[0]
    scan = jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=TARGET, store=store)
    details = deepcopy(scan.child.details)
    assert details["captures"] == []
    assert details["assets"] == []
    historical = study.read_shadow_study(owner=owner, as_of=scan_time, store=store)
    clock[0] += timedelta(seconds=1)
    calls = (resolve.call_count, fetch.call_count)
    capture = jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    assert capture.state == "timely"
    retry = jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=TARGET, store=store)
    assert retry.child.status == JobRun.Status.SKIPPED
    scan.child.refresh_from_db()
    assert scan.child.details == details
    assert study.read_shadow_study(owner=owner, as_of=scan_time, store=store) == historical
    later = jobs.execute_shadow_evaluation_job(
        owner=owner,
        evaluation_date=clock[0].date(),
        store=store,
    )
    assert len(later.child.details["captures"]) == 1
    assert len(later.child.details["assets"]) == 6
    assert (resolve.call_count, fetch.call_count) == calls


@pytest.mark.parametrize("failure", ["acquisition_error", "no_data"])
def test_failed_scheduled_market_retains_known_missed_population(
    activated_scheduled_environment,
    monkeypatch,
    failure,
):
    from stanstock.core import research_product_refresh as refresh

    owner, store, _activation, clock, resolve, fetch, _path = activated_scheduled_environment
    acquire = research_product_jobs._acquire_product_membership

    def no_output(**kwargs):
        if failure == "acquisition_error":
            raise research_product_jobs.ProviderError("synthetic acquisition failure")
        result = acquire(**kwargs)
        return JobExecutionResult(status=JobRun.Status.NO_DATA, details=result.details)

    monkeypatch.setattr(research_product_jobs, "_acquire_product_membership", no_output)
    expected = "market stage failed" if failure == "acquisition_error" else "no complete output"
    with pytest.raises(ValueError, match=expected):
        refresh.execute_scheduled_research_refresh(store=store, enforce_rate_limit=False)
    intake = load_product_intake(
        owner_id=str(owner.pk),
        target_date=TARGET,
        issuance_key="scheduled",
        store=store,
    )
    assert intake is not None
    capture_child = JobRun.objects.get(job_name__startswith="shadow_study_capture_v1:")
    assert capture_child.status == JobRun.Status.NO_DATA
    assert not DataAsset.objects.filter(kind=study.KINDS[3]).exists()
    assert not Prediction.objects.exists()
    clock[0] += timedelta(days=4)
    monkeypatch.setattr(
        jobs, "project_shadow_drift", lambda *args, **kwargs: pytest.fail("backfill")
    )
    calls = (resolve.call_count, fetch.call_count)
    counts = (JobRun.objects.count(), DataAsset.objects.count())
    read = study.read_shadow_study(owner=owner, as_of=clock[0], store=store)
    assert read.status == "available"
    for summary in read.summaries:
        anchor = summary.per_anchor[0]
        assert anchor.target_date == TARGET
        assert anchor.intended_cases == len(intake.requested_symbols)
        assert anchor.missed == len(intake.requested_symbols)
        assert anchor.accepted == 0
        if failure == "no_data":
            snapshot = research_product_jobs._snapshot_for_intake(intake, store=store)
            membership = product_membership_payload(snapshot, store=store)
            assert anchor.withheld == sum(
                entry["status"] != "admitted" for entry in membership["admissions"].values()
            )
        else:
            assert anchor.withheld == len(intake.requested_symbols)
    assert (JobRun.objects.count(), DataAsset.objects.count()) == counts
    assert (resolve.call_count, fetch.call_count) == calls


@pytest.mark.parametrize("failure", ["typed_preparation", "no_capture"])
def test_scheduled_study_failure_is_visible_beside_healthy_core(
    activated_scheduled_environment,
    monkeypatch,
    client,
    failure,
):
    from django.urls import reverse

    from stanstock.core import research_product_refresh as refresh

    owner, store, _activation, clock, resolve, fetch, _path = activated_scheduled_environment
    project = jobs.project_shadow_drift
    register = study._register_shadow_asset

    def fail_projection(*args, **kwargs):
        if failure == "typed_preparation":
            raise ArithmeticError("synthetic preparation failure")
        return project(*args, **kwargs)

    def fail_registration(**kwargs):
        if failure == "no_capture" and kwargs["kind"] == study.KINDS[3]:
            raise study._error("shadow_asset_invalid")
        return register(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(jobs, "project_shadow_drift", fail_projection)
        patch.setattr(study, "_register_shadow_asset", fail_registration)
        result = refresh.execute_scheduled_research_refresh(store=store, enforce_rate_limit=False)
    assert result.parent.status == JobRun.Status.SUCCESS
    assert result.shadow_capture.state == "failed"
    assert DataAsset.objects.filter(kind=study.KINDS[3]).count() == (
        1 if failure == "typed_preparation" else 0
    )
    parent_details = deepcopy(result.parent.details)
    calls = (resolve.call_count, fetch.call_count)
    client.force_login(owner)
    response = client.get(reverse("status"))
    assert response.status_code == 200 and response.context["product"].available
    read = response.context["shadow_study"]
    assert read.status == "available" and read.capture_counts["failed"] == 1
    assert read.reason_code == (
        "shadow_preparation_failed" if failure == "typed_preparation" else "shadow_asset_invalid"
    )
    assert b"Research failure" in response.content
    assert b"Failed captures: 1" in response.content
    if failure == "no_capture":
        failed_at = clock[0]
        clock[0] += timedelta(seconds=1)
        recovered = refresh.execute_scheduled_research_refresh(
            store=store,
            enforce_rate_limit=False,
        )
        assert recovered.parent.status == JobRun.Status.SKIPPED
        assert recovered.shadow_capture.state == "timely"
        after = study.read_shadow_study(owner=owner, as_of=clock[0], store=store)
        assert after.capture_counts["failed"] == 0 and after.reason_code is None
        assert study.read_shadow_study(owner=owner, as_of=failed_at, store=store) == read
    result.parent.refresh_from_db()
    assert result.parent.details == parent_details
    assert (resolve.call_count, fetch.call_count) == calls


def test_capture_complete_original_intake_bytes_and_recovery(capture_environment, monkeypatch):
    owner, store, activation, clock, resolve, fetch = capture_environment
    original = tuple(Prediction.objects.values_list("pk", flat=True))
    before_calls = (resolve.call_count, fetch.call_count)
    result = jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    assert result.state == "timely"
    asset = DataAsset.objects.get(kind=study.KINDS[3])
    document = study._read_asset(asset, store=store)
    assert document["activation"] == asset_ref_for(activation).to_json()
    body = document["body"]
    assert body["population_count"] == len(body["members"])
    assert any(item["listing_id"] is None for item in body["members"])
    assert any(item["admission"] == "identity_rejected" for item in body["members"])
    assert len(body["forecasts"]) == 3
    for forecast in body["forecasts"]:
        adapter = jobs._adapter_document(forecast)
        assert len(adapter["projections"]) == 6
        assert adapter["path_count"] == 8192
        assert adapter["simulation_horizons"] == [
            ["6m", 126],
            ["12m", 252],
            ["3y", 756],
            ["5y", 1260],
        ]
        assert set(forecast["original_prediction_ids"]) == {"6m", "12m"}
    clock[0] += timedelta(days=3)
    monkeypatch.setattr(jobs, "project_shadow_drift", lambda *a, **kw: pytest.fail("rederived"))
    retry = jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    assert retry.state == "timely"
    assert retry.child.status == "skipped"
    assert DataAsset.objects.filter(kind=study.KINDS[3]).count() == 1
    assert tuple(Prediction.objects.values_list("pk", flat=True)) == original
    assert (resolve.call_count, fetch.call_count) == before_calls


def test_capture_default_off_has_no_io_or_job(settings, monkeypatch):
    settings.SHADOW_STUDY_ENABLED = False
    monkeypatch.setattr(study, "_context", lambda **kwargs: pytest.fail("study I/O"))
    result = jobs.execute_shadow_capture_job(owner=None, target_date=TARGET, store=None)
    assert result.state == "disabled" and result.child is None
    assert not JobRun.objects.exists()


def test_all_five_registered_schemas_are_closed(capture_environment):
    owner, store, _activation, _clock, _resolve, _fetch = capture_environment
    jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=TARGET, store=store)
    study.close_shadow_study(owner=owner, store=store)
    assert set(
        DataAsset.objects.filter(kind__in=study.KINDS).values_list("kind", flat=True)
    ) == set(study.KINDS)
    for kind in study.KINDS:
        asset = DataAsset.objects.filter(kind=kind).first()
        document = study._read_asset(asset, store=store)
        for key in document:
            changed = deepcopy(document)
            del changed[key]
            with pytest.raises(study.RefreshVerificationError):
                study._validate_document(changed, kind)
        for key in document["body"]:
            changed = deepcopy(document)
            del changed["body"][key]
            with pytest.raises(study.RefreshVerificationError):
                study._validate_document(changed, kind)
        for target in ("body", "envelope"):
            changed = deepcopy(document)
            (changed["body"] if target == "body" else changed)["unknown"] = None
            with pytest.raises(study.RefreshVerificationError):
                study._validate_document(changed, kind)


def test_adapter_copied_input_hash_cannot_authenticate_wrong_source_identity(capture_environment):
    owner, store, activation, _clock, _resolve, _fetch = capture_environment
    jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    original = DataAsset.objects.get(kind=study.KINDS[3])
    document = study._read_asset(original, store=store)
    forecast = document["body"]["forecasts"][0]
    adapter = json.loads(forecast["shadow_json"])
    adapter["stock_identity"]["asset_id"] = adapter["benchmark_identity"]["asset_id"]
    forecast["shadow_json"] = study._canonical_bytes(adapter).decode()
    forecast["shadow_sha256"] = hashlib.sha256(forecast["shadow_json"].encode()).hexdigest()
    payload = study._canonical_bytes(document)
    digest = hashlib.sha256(payload).hexdigest()
    stored = store.write_bytes(f"research/shadow/{original.kind}/{digest}.json", payload)
    forged = DataAsset.objects.create(
        provider=original.provider,
        kind=original.kind,
        subject=original.subject,
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        available_at=original.available_at,
        retrieved_at=original.retrieved_at,
        schema_version=original.schema_version,
        metadata=study._metadata(document, original.kind),
    )
    with pytest.raises(study.RefreshVerificationError, match="shadow_source_identity_invalid"):
        jobs._verify_capture(
            forged,
            owner_id=str(owner.pk),
            provider="twelve_data",
            protocol=DataAsset.objects.get(kind=study.KINDS[0]),
            activation=activation,
            store=store,
        )
    with pytest.raises(study.RefreshVerificationError, match="shadow_registry_ambiguous"):
        jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)


def test_late_first_capture_cannot_backfill_forecasts(capture_environment, monkeypatch):
    owner, store, _activation, clock, resolve, fetch = capture_environment
    clock[0] += timedelta(days=3)
    monkeypatch.setattr(
        jobs, "project_shadow_drift", lambda *a, **kw: pytest.fail("late derivation")
    )
    before = (resolve.call_count, fetch.call_count)
    result = jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    assert result.state == "missed"
    body = study._read_asset(DataAsset.objects.get(kind=study.KINDS[3]), store=store)["body"]
    assert body["population_count"] is not None
    assert all(item["shadow_json"] is None for item in body["forecasts"])
    assert (resolve.call_count, fetch.call_count) == before


def _future_prices(store, clock, *, terminal_change=0, basis_change=False, future_row=False):
    calendar = get_calendar("XNYS")
    sessions = calendar.sessions_window(TARGET, 253)[1:]
    through = sessions[-1].date()
    clock[0] = datetime.combine(through, datetime.min.time(), UTC) + timedelta(
        hours=22, seconds=terminal_change
    )
    subjects = set(Prediction.objects.values_list("price_subject", flat=True)) | {"SPY"}
    assets = []
    for subject in sorted(subjects):
        original = (
            DataAsset.objects.filter(
                provider="twelve_data",
                kind="price_history",
                subject=subject,
            )
            .order_by("available_at")
            .first()
        )
        raw = DataAsset.objects.get(pk=original.metadata["raw_asset_id"])
        document = json.loads(store.read_bytes(raw.relative_path))
        values = document["values"]
        baseline = next(row for row in values if row["datetime"] == TARGET.isoformat())
        close = float(baseline["close"])
        if basis_change:
            for field in ("open", "high", "low", "close"):
                baseline[field] = str(close * 2)
        for index, session in enumerate(sessions):
            value = close * (1 + (index + 1) / 1000)
            if terminal_change:
                value *= 1 + terminal_change / 100
            values.append(
                {
                    "datetime": session.date().isoformat(),
                    "open": str(value),
                    "high": str(value),
                    "low": str(value),
                    "close": str(value),
                    "volume": "1000000",
                }
            )
        if future_row:
            row = dict(values[-1])
            row["datetime"] = calendar.next_session(sessions[-1]).date().isoformat()
            values.append(row)
        series = twelve_data.parse_daily_price_series(
            json.dumps(document).encode(),
            symbol=subject,
            retrieved_at=clock[0],
            source_url="https://example.invalid/synthetic",
            end_date=through,
        )
        assets.append(_persist_price_series(store=store, series=series, listing=None))
    return through, assets


def _native_outcomes(clock, through):
    for prediction in Prediction.objects.filter(
        method_version="us-price-fhs-v1", horizon__in=study.HORIZONS
    ):
        evaluate_prediction(
            prediction,
            provider="twelve_data",
            evaluation_date=through,
            evaluation_time=clock[0],
            benchmark_subject="SPY",
        )


def test_first_acceptance_challenges_asof_and_frozen_contribution(capture_environment):
    owner, store, _activation, clock, resolve, fetch = capture_environment
    jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    before = (resolve.call_count, fetch.call_count)
    through, original_assets = _future_prices(store, clock)
    _native_outcomes(clock, through)
    for prediction in Prediction.objects.filter(
        method_version="us-price-fhs-v1",
        horizon__in=study.HORIZONS,
    ):
        outcome = prediction.outcome
        verification = jobs.verify_prediction_outcome(
            prediction,
            outcome,
            provider="twelve_data",
            benchmark_subject="SPY",
            evaluation_time=outcome.evaluated_at,
            parent_target_date=outcome.evaluation_date,
            frame_cache={},
        )
        jobs._price_evidence(
            prediction,
            cutoff=outcome.evaluated_at,
            through=outcome.evaluation_date,
            store=store,
            refs=jobs._sorted_refs([ref.to_json() for ref in verification.asset_refs]),
        )
    result = jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=through, store=store)
    assert result.state == "available"
    accepted_at = clock[0]
    rows = DataAsset.objects.filter(kind=study.KINDS[4])
    assert rows.count() == 6
    assert all(
        study._read_asset(asset, store=store)["body"]["state"] == "accepted" for asset in rows
    ), [
        (
            study._read_asset(asset, store=store)["body"]["qualification"],
            study._read_asset(asset, store=store)["body"]["reason_codes"],
        )
        for asset in rows
    ]
    original = study.read_shadow_study(owner=owner, as_of=accepted_at, store=store)
    assert original.status == "available"
    assert original.case_counts["accepted"] == 6
    assert original.case_counts["accepted_with_challenge"] == 0
    original_scores = original.summaries[0].per_anchor[0].arm_means
    assert original_scores is not None
    # Missed subsequent intended anchors must not disappear into an available-anchor mean.
    assert original.summaries[0].arm_means is None
    _future_prices(store, clock, terminal_change=1)
    evaluation_date = through + timedelta(days=1)
    clock[0] += timedelta(days=1)
    jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=evaluation_date, store=store)
    challenged = study.read_shadow_study(owner=owner, as_of=clock[0], store=store)
    assert challenged.status == "available"
    assert challenged.case_counts["accepted_with_challenge"] == 6
    assert challenged.summaries[0].per_anchor[0].arm_means == original_scores
    assert study.read_shadow_study(owner=owner, as_of=accepted_at, store=store) == original
    assert DataAsset.objects.filter(kind=study.KINDS[4]).count() == 12
    clock[0] += timedelta(days=1)
    jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=clock[0].date(), store=store)
    assert DataAsset.objects.filter(kind=study.KINDS[4]).count() == 12
    assert (resolve.call_count, fetch.call_count) == before
    # A real later vintage can challenge, but physical loss of frozen evidence fails closed.
    store.resolve(original_assets[0].relative_path).write_bytes(b"corrupt synthetic evidence")
    invalid = study.read_shadow_study(owner=owner, as_of=clock[0], store=store)
    assert invalid.status == "integrity_failed"
    assert all(summary.arm_means is None for summary in invalid.summaries)


def test_pending_evaluation_does_not_append_clock_only_duplicates(capture_environment):
    owner, store, _activation, clock, resolve, fetch = capture_environment
    jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    before = (resolve.call_count, fetch.call_count)
    jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=TARGET, store=store)
    count = DataAsset.objects.filter(kind=study.KINDS[4]).count()
    assert count == 6
    clock[0] += timedelta(days=4)
    jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=clock[0].date(), store=store)
    assert DataAsset.objects.filter(kind=study.KINDS[4]).count() == count
    assert (resolve.call_count, fetch.call_count) == before


def test_quarantined_vintage_can_reach_first_acceptance_without_rewriting_native_row(
    capture_environment,
):
    owner, store, _activation, clock, resolve, fetch = capture_environment
    jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    through, _assets = _future_prices(store, clock, terminal_change=1, basis_change=True)
    _native_outcomes(clock, through)
    native_before = {
        str(prediction.pk): jobs._native_snapshot(prediction.outcome)
        for prediction in Prediction.objects.filter(
            method_version="us-price-fhs-v1",
            horizon__in=study.HORIZONS,
        )
    }
    jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=through, store=store)
    assert (
        study.read_shadow_study(
            owner=owner,
            as_of=clock[0],
            store=store,
        ).case_counts["quarantined"]
        == 6
    )
    _future_prices(store, clock, terminal_change=2)
    clock[0] += timedelta(days=1)
    calls = (resolve.call_count, fetch.call_count)
    jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=clock[0].date(), store=store)
    accepted = study.read_shadow_study(owner=owner, as_of=clock[0], store=store)
    assert accepted.status == "available" and accepted.case_counts["accepted"] == 6
    assert DataAsset.objects.filter(kind=study.KINDS[4]).count() == 12
    assert {
        str(prediction.pk): jobs._native_snapshot(prediction.outcome)
        for prediction in Prediction.objects.filter(
            method_version="us-price-fhs-v1",
            horizon__in=study.HORIZONS,
        )
    } == native_before
    assert (resolve.call_count, fetch.call_count) == calls


def test_completed_evaluation_scan_is_bound_to_its_start_not_later_capture(
    capture_environment,
):
    owner, store, _activation, clock, resolve, fetch = capture_environment
    jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    first = jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=TARGET, store=store)
    details = dict(first.child.details)
    clock[0] += timedelta(days=4)
    future_target = study._calendar().next_session(TARGET).date()
    calls = (resolve.call_count, fetch.call_count)
    missed = jobs.execute_shadow_capture_job(owner=owner, target_date=future_target, store=store)
    assert missed.state == "missing_population"
    retry = jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=TARGET, store=store)
    assert retry.child.status == "skipped"
    first.child.refresh_from_db()
    assert first.child.details == details
    assert DataAsset.objects.filter(kind=study.KINDS[4]).count() == 6
    assert (resolve.call_count, fetch.call_count) == calls


@pytest.mark.parametrize("mode", ["compatible", "incompatible", "unverifiable"])
def test_each_later_vintage_challenge_retains_original_scores(capture_environment, mode):
    owner, store, _activation, clock, resolve, fetch = capture_environment
    jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    through, _assets = _future_prices(store, clock)
    _native_outcomes(clock, through)
    jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=through, store=store)
    first = study.read_shadow_study(owner=owner, as_of=clock[0], store=store)
    _through, assets = _future_prices(
        store,
        clock,
        terminal_change=1,
        basis_change=mode == "incompatible",
    )
    if mode == "unverifiable":
        for asset in assets:
            store.resolve(asset.relative_path).write_bytes(b"synthetic unavailable vintage")
    clock[0] += timedelta(days=1)
    calls = (resolve.call_count, fetch.call_count)
    jobs.execute_shadow_evaluation_job(owner=owner, evaluation_date=clock[0].date(), store=store)
    result = study.read_shadow_study(owner=owner, as_of=clock[0], store=store)
    assert result.status == "available"
    assert result.case_counts["accepted_with_challenge"] == 6
    for index in range(2):
        assert (
            result.summaries[index].per_anchor[0].arm_means
            == first.summaries[index].per_anchor[0].arm_means
        )
    qualifications = {
        study._read_asset(asset, store=store)["body"]["qualification"]
        for asset in DataAsset.objects.filter(kind=study.KINDS[4])
        if study._read_asset(asset, store=store)["body"]["state"] == "challenge"
    }
    assert qualifications == {
        f"later_{mode}_revision" if mode != "unverifiable" else "later_unverifiable_evidence"
    }
    assert (resolve.call_count, fetch.call_count) == calls


def test_postcommit_crash_recovery_cannot_inherit_on_time_source(capture_environment, monkeypatch):
    from django.db.models.query import QuerySet

    owner, store, _activation, clock, _resolve, _fetch = capture_environment
    original_update = QuerySet.update

    def fail_completion(queryset, **kwargs):
        if queryset.model is JobRun and kwargs.get("status") == JobRun.Status.SUCCESS:
            raise RuntimeError("synthetic crash after asset commit")
        return original_update(queryset, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(QuerySet, "update", fail_completion)
        with pytest.raises(RuntimeError, match="synthetic crash"):
            jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    asset = DataAsset.objects.get(kind=study.KINDS[3])
    original_bytes = store.read_bytes(asset.relative_path)
    clock[0] += timedelta(days=3)
    monkeypatch.setattr(jobs, "project_shadow_drift", lambda *a, **kw: pytest.fail("rederived"))
    result = jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    assert result.state == "late"
    assert store.read_bytes(asset.relative_path) == original_bytes
    assert DataAsset.objects.filter(kind=study.KINDS[3]).count() == 1


def test_deadline_crossing_during_calculation_is_not_timely(capture_environment, monkeypatch):
    owner, store, _activation, clock, _resolve, _fetch = capture_environment
    actual = jobs.project_shadow_drift
    deadline = (
        study._calendar().session_open(study._calendar().next_session(TARGET)).to_pydatetime()
    )

    def cross_deadline(*args, **kwargs):
        result = actual(*args, **kwargs)
        clock[0] = deadline
        return result

    monkeypatch.setattr(jobs, "project_shadow_drift", cross_deadline)
    result = jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    assert result.state == "late"


@pytest.mark.parametrize("boundary", ["lock_wait", "commit"])
def test_deadline_crossings_at_lock_and_commit(capture_environment, monkeypatch, boundary):
    from contextlib import contextmanager

    from stanstock.core import jobs as core_jobs

    owner, store, _activation, clock, _resolve, _fetch = capture_environment
    deadline = (
        study._calendar().session_open(study._calendar().next_session(TARGET)).to_pydatetime()
    )
    if boundary == "lock_wait":
        actual_lock = core_jobs.target_job_lock

        @contextmanager
        def delayed(**kwargs):
            with actual_lock(**kwargs):
                if kwargs["job_name"].startswith("shadow_study_capture_v1"):
                    clock[0] = deadline
                yield

        monkeypatch.setattr(core_jobs, "target_job_lock", delayed)
    else:
        actual_register = study._register_shadow_asset

        def delayed_commit(**kwargs):
            asset = actual_register(**kwargs)
            clock[0] = deadline
            return asset

        monkeypatch.setattr(study, "_register_shadow_asset", delayed_commit)
    result = jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    assert result.state == ("missed" if boundary == "lock_wait" else "late")


@pytest.mark.parametrize("unexpected", [False, True])
def test_optional_capture_failure_does_not_suppress_native_downstream(
    capture_environment,
    monkeypatch,
    unexpected,
):
    from stanstock.core import research_product_refresh as refresh

    owner, store, _activation, _clock, _resolve, _fetch = capture_environment
    original = refresh._run_stage
    attempted = []

    def record_stage(**kwargs):
        attempted.append(kwargs["stage_name"])
        return original(**kwargs)

    def fail(**kwargs):
        if unexpected:
            raise RuntimeError("synthetic unexpected capture defect")
        raise study._error("shadow_preparation_failed")

    monkeypatch.setattr(refresh, "_run_stage", record_stage)
    monkeypatch.setattr(jobs, "execute_shadow_capture_job", fail)
    if unexpected:
        with pytest.raises(RuntimeError, match="synthetic unexpected"):
            refresh.execute_scheduled_research_refresh(store=store, enforce_rate_limit=False)
    else:
        result = refresh.execute_scheduled_research_refresh(store=store, enforce_rate_limit=False)
        assert result.parent.status == "success"
        assert result.shadow_capture.state == "failed"
        assert set(result.parent.details["stages"]) == refresh.STAGE_NAMES
        assert not any("shadow" in key for key in result.parent.details)
    assert attempted == ["market", "evaluation", "portfolio_snapshots"]


def test_critical_flow_persistence_to_authorized_status_and_corruption(
    capture_environment,
    client,
    monkeypatch,
    settings,
):
    from django.urls import reverse

    owner, store, _activation, clock, _resolve, _fetch = capture_environment
    jobs.execute_shadow_capture_job(owner=owner, target_date=TARGET, store=store)
    client.force_login(owner)
    response = client.get(reverse("status"))
    assert response.status_code == 200
    assert response.context["shadow_study"].status == "available"
    assert b"Prospective three-arm study" in response.content
    count = DataAsset.objects.count()
    monkeypatch.setattr(
        jobs, "project_shadow_drift", lambda *a, **kw: pytest.fail("HTTP simulation")
    )
    asset = DataAsset.objects.get(kind=study.KINDS[3])
    store.resolve(asset.relative_path).write_bytes(b"synthetic corrupt capture")
    response = client.get(reverse("status"))
    assert response.context["shadow_study"].status == "integrity_failed"
    assert b"integrity failed" in response.content.lower()
    assert DataAsset.objects.count() == count
    settings.SHADOW_STUDY_ENABLED = False
    monkeypatch.setattr(study, "read_shadow_study", lambda **kw: pytest.fail("disabled I/O"))
    response = client.get(reverse("status"))
    assert response.context["shadow_study"] is None
    assert b"DISABLED" in response.content


@pytest.fixture(scope="module")
def exact_prospective_base(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(adapter_tests, "BASE_SHA", "8bae513b0b9a115a60fdc29474ab604787382a43")
        yield adapter_tests.exact_current_base.__wrapped__(tmp_path_factory)


@pytest.mark.parametrize("mode", ["scheduled", "recovery"])
def test_complete_default_off_native_differential_at_authorized_base(exact_prospective_base, mode):
    base = exact_prospective_base("base", mode).require("authorized base")
    candidate = exact_prospective_base("candidate", mode).require("source-only candidate")
    adapter_tests.frozen.assert_frozen_surface_unchanged(base, candidate, allow_additive=False)
    adapter_tests._require_complete_native_states(candidate)


def _optin_probe(workdir):
    """Run the same full native probe with only separately owned study additions."""
    from django.conf import settings
    from django.contrib.auth import get_user_model
    from django.utils import timezone

    import research_product_frozen_probability_probe as probe
    from stanstock.core import research_product_refresh as refresh
    from stanstock.research import shadow_study

    native = refresh.execute_scheduled_research_refresh
    observed = []

    def enabled(**kwargs):
        settings.SHADOW_STUDY_ENABLED = True
        settings.DEMO_MODE = False
        shadow_study.clean_git_revision = lambda root: probe.SYNTHETIC_REVISION
        owner = get_user_model().objects.get(username=probe.OWNER_USERNAME)
        previous_clock = timezone.now
        timezone.now = lambda: probe.NOW.replace(day=11, hour=19)
        try:
            shadow_study.activate_shadow_study(owner=owner, store=kwargs["store"])
        finally:
            timezone.now = previous_clock
        result = native(**kwargs)
        assert result.shadow_capture.state == "timely"
        assert result.shadow_evaluation.state == "available"
        observed.append(True)
        return result

    refresh.execute_scheduled_research_refresh = enabled
    payload = probe.run(mode="scheduled", cohort="full", workdir=workdir)
    assert observed == [True]
    return payload


def test_optin_only_adds_separate_study_evidence_no_native_payload_changes(
    exact_prospective_base,
    tmp_path,
):
    base = exact_prospective_base("base", "scheduled").require("authorized base")
    code = (
        "import json, sys, uuid\nfrom pathlib import Path\n"
        "import research_product_frozen_probability_probe as probe\n"
        "probe._install_boundary_guards()\n"
        "uuid.uuid4 = probe._DeterministicUuid('uuid4')\n"
        "import django\ndjango.setup()\n"
        + inspect.getsource(_optin_probe)
        + "\npayload = _optin_probe(Path(sys.argv[1]))\n"
        + "print(json.dumps(payload, sort_keys=True))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=tmp_path,
        env=adapter_tests._process_environment(adapter_tests.ROOT / "src"),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    candidate = json.loads(completed.stdout)
    before, after = adapter_tests.frozen._flatten(base), adapter_tests.frozen._flatten(candidate)
    # No key, original ref, byte digest, withholding or native field is stripped.
    assert set(before) <= set(after)
    assert {key: after[key] for key in before} == before
    added = set(after) - set(before)
    assert added
    for path in added:
        assert (len(path) >= 2 and path[0] == "assets" and path[1].startswith("shadow_study_")) or (
            len(path) >= 3
            and path[:2] == ("scheduled", "job_runs")
            and path[2].startswith("shadow_study_")
        )
    adapter_tests._require_complete_native_states(candidate)

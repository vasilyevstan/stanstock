from __future__ import annotations

import copy
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from threading import Barrier, Lock
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection, connections
from django.urls import reverse
from django.utils import timezone
from exchange_calendars import get_calendar

from interval_calibration_documents import build_interval_calibration_document
from stanstock.core.models import JobRun
from stanstock.data.assets import AssetStore, asset_ref_for
from stanstock.data.models import DataAsset
from stanstock.research import price_product_interval_shadow as shadow
from stanstock.research.models import Prediction
from stanstock.research.price_product_interval_calibration import canonical_sha256
from stanstock.research.price_product_interval_calibration_document import (
    fit_interval_calibration_document,
)
from stanstock.research.price_product_replay import LedgerTriplet, score_projection_metrics
from stanstock.research.product_study_evidence import register_price_product_study
from stanstock.research.reporting import _ADVISORY_FLOORS
from test_research_product_reader_web import live_product as live_product
from test_research_product_study_evidence import demo_product as demo_product

pytestmark = pytest.mark.django_db(transaction=True)

_FIXED_NOW = datetime(2026, 9, 13, 22, tzinfo=UTC)
_DECLARATION_KEYS = {
    "schema",
    "protocol",
    "protocol_sha256",
    "owner_id",
    "source_owner_id",
    "source_provider",
    "study",
    "study_full_report_sha256",
    "params",
    "params_sha256",
    "declared_at",
    "producer_job_id",
    "execution_revision",
    "calendar_dependency",
}
_METADATA_KEYS = {
    "owner_id",
    "protocol_version",
    "protocol_sha256",
    "declared_at",
    "params_sha256",
    "study_asset_id",
}


@pytest.fixture(autouse=True)
def remove_test_db_guards_before_transactional_flush(monkeypatch):
    """Permit Django's DELETE flush of the disposable transaction test DB."""

    def blocked_provider(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("provider access is forbidden in S3 tests")

    monkeypatch.setattr("httpx.Client.send", blocked_provider)
    monkeypatch.setattr("stanstock.data.providers.twelve_data.resolve_api_key", blocked_provider)
    yield
    if connection.vendor == "sqlite":
        with connection.cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
            for (trigger_name,) in cursor.fetchall():
                cursor.execute(f'DROP TRIGGER IF EXISTS "{trigger_name}"')


@pytest.fixture
def prepared_shadow(demo_product, monkeypatch):
    user, store, run = demo_product
    study = register_price_product_study(run=run, store=store)
    builder_document = build_interval_calibration_document()
    fitted = fit_interval_calibration_document(builder_document)
    assert fitted["params"]["horizons"]["6m"]["state"] == "usable"
    monkeypatch.setattr(shadow, "fit_interval_calibration_document", lambda _doc: fitted)
    monkeypatch.setattr(shadow, "_revision", lambda: "a" * 40)
    monkeypatch.setattr(timezone, "now", lambda: _FIXED_NOW)
    return user, store, run, study, builder_document, fitted


def _read_declaration(store: AssetStore, asset: DataAsset) -> dict[str, Any]:
    return json.loads(store.read_bytes(asset.relative_path))


def _copy_declaration(
    *,
    asset: DataAsset,
    document: dict[str, Any],
    store: AssetStore,
    metadata: dict[str, Any] | None = None,
    subject: str | None = None,
    available_at: datetime | None = None,
    retrieved_at: datetime | None = None,
) -> DataAsset:
    payload = shadow._bytes(document)
    stored = store.write_bytes(
        f"research/interval_shadow/test-copies/{hashlib.sha256(payload).hexdigest()}.json",
        payload,
    )
    copied = copy.copy(asset)
    copied.relative_path = stored.relative_path
    copied.sha256 = stored.sha256
    copied.metadata = metadata if metadata is not None else copy.deepcopy(asset.metadata)
    copied.subject = subject if subject is not None else asset.subject
    if available_at is not None:
        copied.available_at = available_at
    if retrieved_at is not None:
        copied.retrieved_at = retrieved_at
    return copied


def _verify_declaration(
    asset: DataAsset,
    *,
    document: dict[str, Any],
    user: Any,
    store: AssetStore,
    require_witness: bool = True,
) -> tuple[DataAsset, dict[str, Any]]:
    identity = shadow._identity(
        document["owner_id"], document["source_owner_id"], document["source_provider"]
    )
    return shadow._verify_declaration(
        asset,
        user=user,
        store=store,
        expected_identity=identity,
        require_witness=require_witness,
        expected_job_name=shadow._job_name(identity),
    )


def _make_scoring_records(*, negative_cohorts: int = 7) -> list[dict[str, Any]]:
    calendar = get_calendar("XNYS")
    anchors = calendar.sessions_window(calendar.date_to_session(date(2020, 1, 2)), 1500)
    records: list[dict[str, Any]] = []
    config = shadow.load_price_product_config()
    for cohort_index in range(8):
        target = anchors[cohort_index * 180].date()
        for listing in range(30):
            actual = Decimal("0.03")
            native = LedgerTriplet(Decimal("-0.01"), Decimal("0"), Decimal("0.01"))
            # Candidate has lower interval score on the requested number of
            # cohorts, with tie-free and deterministic synthetic returns.
            candidate = (
                LedgerTriplet(Decimal("-0.03"), Decimal("0"), Decimal("0.03"))
                if cohort_index < negative_cohorts
                else LedgerTriplet(Decimal("-0.01"), Decimal("0"), Decimal("0.01"))
            )
            if cohort_index >= negative_cohorts:
                actual = Decimal("0")
            row = {
                "id": uuid4(),
                "listing_id": f"synthetic-{listing}",
                "target_date": target,
                "generated_at": datetime.combine(target, datetime.min.time(), UTC),
                "reportable": True,
                "canonical": True,
                "outcome__status": "matured",
            }
            records.append(
                {
                    "row": row,
                    "target_date": target,
                    "evaluation_date": calendar.session_offset(
                        calendar.date_to_session(target), 126
                    ).date(),
                    "outcome_status": "matured",
                    "native": score_projection_metrics(
                        predicted_returns=native,
                        actual_return=actual,
                        config=config,
                    ),
                    "candidate": score_projection_metrics(
                        predicted_returns=candidate,
                        actual_return=actual,
                        config=config,
                    ),
                    "candidate_representable": True,
                }
            )
    return records


def _projection_row(
    *,
    target: date,
    generated_at: datetime,
    canonical: bool = True,
    listing_id: str | None = None,
) -> dict[str, Any]:
    calendar = get_calendar(
        "XNYS", start=target - timedelta(days=366), end=target + timedelta(days=366)
    )
    maturity = calendar.session_offset(calendar.date_to_session(target), 126).date()
    return {
        "id": uuid4(),
        "listing_id": listing_id or f"synthetic-{uuid4()}",
        "target_date": target,
        "generated_at": generated_at,
        "evidence_role": "advisory",
        "method_version": shadow.FHS_METHOD_VERSION,
        "config_hash": shadow.PRODUCT_EFFECTIVE_CONFIG_HASH,
        "price_provider": "synthetic_demo",
        "horizon": "6m",
        "bear_return": Decimal("-0.01"),
        "base_return": Decimal("0"),
        "bull_return": Decimal("0.01"),
        "reportable": True,
        "canonical": canonical,
        "outcome__status": "matured",
        "outcome__evaluation_date": maturity,
        "outcome__evaluated_at": _FIXED_NOW - timedelta(seconds=1),
        "outcome__actual_return": Decimal("0.03"),
    }


def _patch_report_projection(
    monkeypatch, rows, *, provider="synthetic_demo", current_time=_FIXED_NOW
):
    scope = SimpleNamespace(
        available=True,
        runs=(
            SimpleNamespace(run=SimpleNamespace(config_version=shadow.PRODUCT_VERSION, id=uuid4())),
        ),
    )
    monkeypatch.setattr(shadow, "read_research_product_performance", lambda **_kwargs: scope)
    monkeypatch.setattr(
        shadow, "iter_product_prediction_projection", lambda *_args, **_kwargs: iter(rows)
    )
    monkeypatch.setattr(
        shadow,
        "_authenticated_outcome",
        lambda row, _provider: SimpleNamespace(
            status="matured",
            evaluated_at=row["outcome__evaluated_at"],
            evaluation_date=row["outcome__evaluation_date"],
            actual_return=row["outcome__actual_return"],
        ),
    )
    monkeypatch.setattr(timezone, "now", lambda: current_time)
    return provider


def _report_keys() -> set[str]:
    return {
        "schema",
        "protocol_version",
        "protocol_sha256",
        "status",
        "verification_code",
        "declaration_attempts",
        "declaration",
        "evidence_label",
        "horizons",
        "primary_result",
        "companion_result",
        "s4_gate",
        "report_read_at",
        "disclosures",
    }


def test_protocol_pin_and_synthetic_declare_then_report(prepared_shadow):
    user, store, _run, study, _builder, fitted = prepared_shadow
    assert hashlib.sha256(shadow.forward_scorecard._bytes(shadow.SHADOW_PROTOCOL)).hexdigest() == (
        shadow.SHADOW_PROTOCOL_SHA256
    )
    assert _ADVISORY_FLOORS["6m"] == (8, 30, 1095)
    assert fitted["params"]["horizons"]["6m"]["state"] == "usable"

    declaration_asset = shadow.declare_interval_shadow(user=user, store=store)
    document = _read_declaration(store, declaration_asset)
    assert set(document) == _DECLARATION_KEYS
    assert set(declaration_asset.metadata) == _METADATA_KEYS
    assert document["study"] == asset_ref_for(study).to_json()
    assert document["params"] == fitted["params"]
    assert document["params_sha256"] == fitted["params"]["params_sha256"]
    assert document["execution_revision"] == "a" * 40
    assert document["calendar_dependency"] == fitted["source"]["calendar_dependency"]

    producer = JobRun.objects.get(pk=document["producer_job_id"])
    witness = shadow._witness(
        declaration_asset,
        identity=shadow._identity(
            document["owner_id"], document["source_owner_id"], document["source_provider"]
        ),
        job_name=producer.job_name,
    )
    assert witness == producer.finished_at
    assert (
        study.available_at <= producer.started_at <= datetime.fromisoformat(document["declared_at"])
    )
    assert producer.finished_at >= producer.started_at

    report = shadow.read_interval_shadow_report(user=user, store=store)
    assert set(report) == _report_keys()
    assert report["status"] == "available"
    assert report["evidence_label"] == "synthetic_demo_not_empirical"
    assert report["declaration"]["asset"] == asset_ref_for(declaration_asset).to_json()
    assert report["horizons"]["6m"]["state"] == "usable"
    assert report["horizons"]["12m"]["state"] == "withheld_native_fallback"
    assert report["horizons"]["12m"]["paired_evidence_count"] == 0
    assert report["s4_gate"]["state"] == "not_eligible_synthetic"


def test_declaration_leaves_routed_status_page_unchanged(prepared_shadow, settings, client):
    user, store, *_ = prepared_shadow
    settings.RESEARCH_PRODUCT_ENABLED = True
    client.force_login(user)

    before = client.get(reverse("status"))
    assert before.status_code == 200
    assert "web/product_status.html" in [template.name for template in before.templates]
    before_pks = [job.pk for job in before.context["recent_jobs"]]
    before_available = before.context["product"].available

    output = io.StringIO()
    call_command(
        "price_product_interval_shadow",
        owner=user.username,
        declare=True,
        stdout=output,
    )
    assert json.loads(output.getvalue())["status"] == "available"
    owner_id = _read_declaration(store, DataAsset.objects.get(kind=shadow.DECLARATION_KIND))[
        "owner_id"
    ]
    declarations = DataAsset.objects.filter(
        kind=shadow.DECLARATION_KIND,
        subject=f"{shadow.PROTOCOL_VERSION}:{owner_id}",
    )
    assert declarations.count() == 1
    successful_job = JobRun.objects.get(
        job_name__startswith=f"{shadow.DECLARE_JOB}:",
        status=JobRun.Status.SUCCESS,
        target_date=date.min,
    )
    assert (
        JobRun.objects.filter(
            job_name__startswith=f"{shadow.DECLARE_JOB}:",
            status=JobRun.Status.SUCCESS,
            target_date=date.min,
        ).count()
        == 1
    )
    declaration = declarations.get()
    declared_document = _read_declaration(store, declaration)
    assert str(successful_job.pk) == declared_document["producer_job_id"]
    assert successful_job.details["identity"] == shadow._identity(
        declared_document["owner_id"],
        declared_document["source_owner_id"],
        declared_document["source_provider"],
    )
    assert successful_job.details["assets"] == [asset_ref_for(declaration).to_json()]
    assert declaration.provider == "stanstock"
    assert declaration.metadata["owner_id"] == owner_id
    assert declaration.metadata["protocol_sha256"] == shadow.SHADOW_PROTOCOL_SHA256

    after = client.get(reverse("status"))
    assert after.status_code == 200
    assert "web/product_status.html" in [template.name for template in after.templates]
    assert [job.pk for job in after.context["recent_jobs"]] == before_pks
    assert after.context["product"].available == before_available
    assert all(
        not job.job_name.startswith("interval_calibration_shadow")
        for job in after.context["recent_jobs"]
    )
    assert b"interval_calibration_shadow" not in after.content


def test_verify_assets_rechecks_shadow_declaration_bytes(prepared_shadow):
    user, store, *_ = prepared_shadow
    asset = shadow.declare_interval_shadow(user=user, store=store)
    output = io.StringIO()
    call_command("verify_assets", stdout=output)
    assert output.getvalue().startswith("Verified ")

    store.resolve(asset.relative_path).write_bytes(b"tampered declaration")
    with pytest.raises(CommandError, match="checksum_mismatch"):
        call_command("verify_assets")


def test_declaration_retry_reuses_asset_and_provider_is_blocked(prepared_shadow, monkeypatch):
    user, store, *_ = prepared_shadow
    original = shadow.declare_interval_shadow(user=user, store=store)
    before_calls = JobRun.objects.filter(job_name__startswith=shadow.DECLARE_JOB).count()

    def blocked(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("provider access is forbidden")

    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("stanstock.data.providers.twelve_data.resolve_api_key", blocked)
    retry = shadow.declare_interval_shadow(user=user, store=store)
    assert retry.pk == original.pk
    assert (
        JobRun.objects.filter(
            job_name__startswith=shadow.DECLARE_JOB,
            status=JobRun.Status.SUCCESS,
        ).count()
        == 1
    )
    assert (
        JobRun.objects.filter(
            job_name__startswith=shadow.DECLARE_JOB,
            status=JobRun.Status.SKIPPED,
        ).count()
        == 1
    )
    assert (
        JobRun.objects.filter(
            job_name__startswith=shadow.DECLARE_JOB,
        ).count()
        == before_calls + 1
    )


def test_crash_after_append_only_asset_recovers_with_later_success_witness(
    prepared_shadow, monkeypatch
):
    user, store, *_ = prepared_shadow
    real_execute = shadow.execute_target_job

    def interrupt_after_publication(*, job_name, region, target_date, task, before_attempt):
        run = JobRun.objects.create(
            job_name=job_name,
            region=region,
            target_date=target_date,
            attempt=1,
        )
        task(run)
        JobRun.objects.filter(pk=run.pk).update(
            status=JobRun.Status.FAILED,
            finished_at=_FIXED_NOW,
            error="IntervalShadowError: interval_shadow_storage_failed",
        )
        raise shadow.IntervalShadowError("interval_shadow_storage_failed")

    monkeypatch.setattr(shadow, "execute_target_job", interrupt_after_publication)
    with pytest.raises(shadow.IntervalShadowError):
        shadow.declare_interval_shadow(user=user, store=store)
    asset = DataAsset.objects.get(kind=shadow.DECLARATION_KIND)
    document = _read_declaration(store, asset)
    failed_producer = JobRun.objects.get(pk=document["producer_job_id"])
    assert failed_producer.status == JobRun.Status.FAILED

    monkeypatch.setattr(timezone, "now", lambda: _FIXED_NOW + timedelta(seconds=1))
    monkeypatch.setattr(shadow, "execute_target_job", real_execute)
    recovered = shadow.declare_interval_shadow(user=user, store=store)
    assert recovered.pk == asset.pk
    success = JobRun.objects.get(
        job_name=failed_producer.job_name,
        status=JobRun.Status.SUCCESS,
    )
    witness = shadow._witness(
        recovered,
        identity=shadow._identity(
            document["owner_id"], document["source_owner_id"], document["source_provider"]
        ),
        job_name=failed_producer.job_name,
    )
    assert witness == success.finished_at
    assert witness > asset.available_at


def test_cli_declare_read_and_forbidden_injected_inputs(prepared_shadow):
    user, _store, *_ = prepared_shadow
    from stanstock.research.management.commands.price_product_interval_shadow import Command

    output = io.StringIO()
    call_command(
        "price_product_interval_shadow",
        owner=user.username,
        declare=True,
        stdout=output,
    )
    report = json.loads(output.getvalue())
    assert report["status"] == "available"
    parser = Command().create_parser("manage.py", "price_product_interval_shadow")
    for option in ("params_file", "study_file", "declared_at"):
        with pytest.raises((CommandError, TypeError)):
            call_command(
                "price_product_interval_shadow",
                owner=user.username,
                **{option: "forbidden"},
            )
        with pytest.raises((SystemExit, CommandError)):
            parser.parse_args(
                ["--owner", user.username, f"--{option.replace('_', '-')}", "forbidden"]
            )


def test_source_study_absence_is_failed_evidence_and_unauthorized_owner_has_no_job(
    demo_product,
):
    user, store, _run = demo_product
    with pytest.raises(shadow.IntervalShadowError) as absent:
        shadow.declare_interval_shadow(user=user, store=store)
    assert absent.value.code == "interval_shadow_source_study_absent"
    failed = JobRun.objects.get(job_name__startswith=shadow.DECLARE_JOB)
    assert failed.status == JobRun.Status.FAILED
    assert failed.error == "IntervalShadowError: interval_shadow_source_study_absent"
    before_unauthorized = JobRun.objects.filter(job_name__startswith=shadow.DECLARE_JOB).count()

    inactive = get_user_model().objects.create_user(username="inactive-shadow-owner")
    inactive.is_active = False
    inactive.save(update_fields=["is_active"])
    with pytest.raises(shadow.IntervalShadowError) as unauthorized:
        shadow.declare_interval_shadow(user=inactive, store=store)
    assert unauthorized.value.code == "interval_shadow_owner_unauthorized"
    assert (
        JobRun.objects.filter(job_name__startswith=shadow.DECLARE_JOB).count()
        == before_unauthorized
    )


def test_withheld_params_are_failed_evidence_without_asset(live_product, monkeypatch):
    owner, store, run = live_product
    study = register_price_product_study(run=run, store=store)
    source_document = shadow._read_registered_document(store=store, asset=study)
    real_fit = fit_interval_calibration_document(source_document)
    assert real_fit["params"]["horizons"]["6m"]["state"] != "usable"
    monkeypatch.setattr(shadow, "fit_interval_calibration_document", lambda _doc: real_fit)
    monkeypatch.setattr(shadow, "_revision", lambda: "b" * 40)
    monkeypatch.setattr(timezone, "now", lambda: _FIXED_NOW + timedelta(days=1))
    with pytest.raises(shadow.IntervalShadowError, match="interval_shadow_params_withheld:"):
        shadow.declare_interval_shadow(user=owner, store=store)
    job = JobRun.objects.get(job_name__startswith=shadow.DECLARE_JOB)
    assert job.status == JobRun.Status.FAILED
    assert job.error.startswith("IntervalShadowError: interval_shadow_params_withheld:")
    assert not DataAsset.objects.filter(kind=shadow.DECLARATION_KIND).exists()
    report = shadow.read_interval_shadow_report(user=owner, store=store)
    assert set(report) == _report_keys()
    assert report["status"] == "not_declared"
    assert report["declaration_attempts"]["latest_failure_code"].startswith(
        "interval_shadow_params_withheld:"
    )
    assert job.details == {}
    assert "offset" not in job.error


def test_t12_recompute_mismatch_is_fail_closed_native_fallback(prepared_shadow, monkeypatch):
    user, store, *_ = prepared_shadow
    shadow.declare_interval_shadow(user=user, store=store)
    original_adapter = shadow.fit_interval_calibration_document

    def changed_adapter(document):
        fitted = copy.deepcopy(original_adapter(document))
        params = fitted["params"]
        params["horizons"]["6m"]["offset"] = "0.123"
        unsigned = dict(params)
        unsigned.pop("params_sha256")
        params["params_sha256"] = canonical_sha256(unsigned)
        return fitted

    monkeypatch.setattr(shadow, "fit_interval_calibration_document", changed_adapter)
    report = shadow.read_interval_shadow_report(user=user, store=store)
    assert report["status"] == "integrity_failed"
    assert report["verification_code"] == "params_recompute_mismatch"
    assert report["horizons"]["6m"]["state"] == "withheld_native_fallback"
    assert report["horizons"]["6m"]["offset"] is None
    assert report["horizons"]["6m"]["paired_evidence_count"] == 0
    assert report["horizons"]["6m"]["native_interval_unchanged"] is True
    assert report["primary_result"] is None


def test_t13_valid_checksum_with_wrong_params_hash_fails_closed(prepared_shadow):
    user, store, *_ = prepared_shadow
    original_asset = shadow.declare_interval_shadow(user=user, store=store)
    document = _read_declaration(store, original_asset)
    document["params_sha256"] = "f" * 64
    document["params"]["params_sha256"] = "f" * 64
    document["declared_at"] = original_asset.available_at.isoformat()
    metadata = dict(original_asset.metadata)
    metadata["params_sha256"] = "f" * 64
    path = "research/interval_shadow/test-tampered/valid-checksum.json"
    stored = store.write_bytes(path, shadow._bytes(document))
    tampered = copy.copy(original_asset)
    tampered.relative_path = stored.relative_path
    tampered.sha256 = stored.sha256
    tampered.metadata = metadata
    with pytest.raises(shadow.IntervalShadowError) as mismatch:
        shadow._verify_declaration(
            tampered,
            user=user,
            store=store,
            expected_identity=shadow._identity(
                document["owner_id"], document["source_owner_id"], document["source_provider"]
            ),
            require_witness=False,
            expected_job_name=shadow._job_name(
                shadow._identity(
                    document["owner_id"],
                    document["source_owner_id"],
                    document["source_provider"],
                )
            ),
        )
    assert mismatch.value.code == "params_recompute_mismatch"


@pytest.mark.parametrize(
    "tamper,expected_code",
    [
        ("protocol", "interval_shadow_protocol_mismatch"),
        ("extra_key", "interval_shadow_declaration_invalid"),
        ("missing_key", "interval_shadow_declaration_invalid"),
        ("rehashed_params", "params_recompute_mismatch"),
    ],
)
def test_declaration_payload_tampering_fails_closed(prepared_shadow, tamper, expected_code):
    user, store, *_ = prepared_shadow
    asset = shadow.declare_interval_shadow(user=user, store=store)
    document = _read_declaration(store, asset)
    metadata = copy.deepcopy(asset.metadata)
    if tamper == "protocol":
        document["protocol"]["version"] = "attacker-version"
    elif tamper == "extra_key":
        document["unexpected"] = True
    elif tamper == "missing_key":
        document.pop("execution_revision")
    elif tamper == "rehashed_params":
        document["params"]["horizons"]["6m"]["offset"] = "0.777"
        unsigned = dict(document["params"])
        unsigned.pop("params_sha256")
        document["params_sha256"] = canonical_sha256(unsigned)
        document["params"]["params_sha256"] = document["params_sha256"]
        metadata["params_sha256"] = document["params_sha256"]

    copied = _copy_declaration(
        asset=asset,
        document=document,
        store=store,
        metadata=metadata,
    )
    with pytest.raises(shadow.IntervalShadowError) as rejected:
        _verify_declaration(
            copied, document=document, user=user, store=store, require_witness=False
        )
    assert rejected.value.code == expected_code


def test_changed_declaration_file_bytes_fail_checksum_verification(prepared_shadow):
    user, store, *_ = prepared_shadow
    asset = shadow.declare_interval_shadow(user=user, store=store)
    store.resolve(asset.relative_path).write_bytes(b"changed immutable bytes")
    report = shadow.read_interval_shadow_report(user=user, store=store)
    assert report["status"] == "integrity_failed"
    assert report["verification_code"] == "interval_shadow_declaration_invalid"


@pytest.mark.parametrize("negative_cohorts,expected", [(7, "met"), (6, "not_met")])
def test_primary_eight_fixed_cohorts_and_companion_arithmetic(negative_cohorts, expected):
    primary, companion = shadow._primary_and_companion(
        _make_scoring_records(negative_cohorts=negative_cohorts)
    )
    assert primary["state"] == expected
    assert primary["selected_cohort_count"] == 8
    assert primary["sign_count"] == negative_cohorts
    assert primary["sign_p_value"] == ("9/256" if negative_cohorts == 7 else "37/256")
    assert companion is not None
    assert companion["native"]["cluster_count"] == 8
    assert companion["calibrated"]["cluster_count"] == 8
    if negative_cohorts == 7:
        assert companion["native"]["mean"] == "0.125"
        assert companion["calibrated"]["mean"] == "1"
        assert companion["native"]["cohorts_below_0.6"] == 7
        assert companion["calibrated"]["cohorts_below_0.6"] == 0
        assert companion["met"] is True


def test_ties_count_against_sign_test_and_mean_gate_is_independent(monkeypatch):
    records = _make_scoring_records(negative_cohorts=7)
    primary, _companion = shadow._primary_and_companion(records)
    assert primary["sign_count"] == 7
    assert primary["state"] == "met"

    def seven_negative_but_nonnegative_mean(rows, *, horizon, cohort_date):
        assert horizon == "6m"
        assert cohort_date is not None
        value = (
            Decimal("-1") if rows[0]["target_date"] != records[-1]["target_date"] else Decimal("8")
        )
        return SimpleNamespace(
            paired_observation_count=len(rows),
            mean_difference_candidate_minus_baseline=value,
        )

    monkeypatch.setattr(shadow, "_compare", seven_negative_but_nonnegative_mean)
    primary, _companion = shadow._primary_and_companion(records)
    assert primary["sign_count"] == 7
    assert Decimal(primary["mean_difference"]) > 0
    assert primary["state"] == "not_met"


def test_primary_accrues_with_fewer_than_eight_cohorts():
    primary, companion = shadow._primary_and_companion(_make_scoring_records()[: 7 * 30])
    assert primary["state"] == "accruing"
    assert primary["selected_cohort_count"] == 7
    assert companion is None


def test_later_cohorts_do_not_change_fixed_primary_result():
    records = _make_scoring_records()
    before, _ = shadow._primary_and_companion(records)
    later = copy.deepcopy(records[:30])
    later_target = max(item["target_date"] for item in records) + timedelta(days=200)
    for item in later:
        item["target_date"] = later_target
        item["row"]["target_date"] = later_target
    after, _ = shadow._primary_and_companion(records + later)
    for key in ("state", "selected_cohort_count", "cohorts", "sign_count", "sign_p_value"):
        assert after[key] == before[key]


@pytest.mark.parametrize("pending_status", [None, "unresolved"])
def test_primary_keeps_pending_and_insufficient_cohorts_without_success_claim(pending_status):
    pending = _make_scoring_records()
    pending[0]["row"]["outcome__status"] = pending_status
    result, companion = shadow._primary_and_companion(pending)
    assert result["state"] == "pending_maturity"
    assert companion is None

    thin = _make_scoring_records()
    # Keep the first eight target cohorts, but remove a candidate score from
    # enough rows to fall below the per-cohort paired floor.
    thin[0]["candidate_representable"] = False
    result, companion = shadow._primary_and_companion(thin)
    assert result["state"] == "insufficient_support"
    assert result["cohorts"][0]["paired_scored_listings"] == 29
    assert companion is None


def test_primary_rejects_less_than_1095_days_of_cohort_span(monkeypatch):
    records = _make_scoring_records()
    calendar = get_calendar("XNYS")
    compact_targets = [
        calendar.sessions_window(calendar.date_to_session(date(2024, 1, 2)), 8)[index].date()
        for index in range(8)
    ]
    for index, target in enumerate(compact_targets):
        for record in records[index * 30 : (index + 1) * 30]:
            record["target_date"] = target
            record["row"]["target_date"] = target
    monkeypatch.setattr(
        shadow,
        "_select_non_overlapping_summaries",
        lambda summaries: sorted(summaries, key=lambda item: item["prediction__target_date"]),
    )
    primary, companion = shadow._primary_and_companion(records)
    assert primary["state"] == "insufficient_support"
    assert primary["span_days"] < 1095
    assert companion is None


def test_actual_delayed_maturity_crossing_next_selected_anchor_withholds_final_primary():
    records = _make_scoring_records()
    targets = sorted({item["target_date"] for item in records})
    next_anchor = targets[1]
    for item in records:
        if item["target_date"] == targets[0]:
            item["evaluation_date"] = next_anchor + timedelta(days=1)
    result, companion = shadow._primary_and_companion(records)
    assert result["state"] == "insufficient_support"
    assert result["selected_cohort_count"] == 8
    assert result["reason"] == "actual_maturity_windows_overlap_selected_cohorts"
    assert result["overlapping_cohort_count"] == 1
    assert result["sign_count"] is None
    assert companion is None


def test_within_prediction_cohort_comparison_weights_listings_not_actual_maturity_dates():
    target = date(2024, 1, 2)
    early_maturity = date(2024, 7, 2)
    late_maturity = date(2025, 1, 2)
    records = []
    for index in range(30):
        baseline = Decimal("0.1")
        candidate = Decimal("0.04") if index < 29 else Decimal("0.58")
        records.append(
            {
                "row": {"listing_id": f"unequal-maturity-{index}", "target_date": target},
                "target_date": target,
                "evaluation_date": early_maturity if index < 29 else late_maturity,
                "native": SimpleNamespace(interval_score=baseline, unavailable_reason=None),
                "candidate": SimpleNamespace(interval_score=candidate, unavailable_reason=None),
                "candidate_representable": True,
            }
        )

    date_grouped = shadow._compare(records, horizon="6m")
    prediction_cohort = shadow._compare(records, horizon="6m", cohort_date=target)
    assert date_grouped.mean_difference_candidate_minus_baseline == Decimal("0.21")
    assert prediction_cohort.paired_observation_count == 30
    assert prediction_cohort.mean_difference_candidate_minus_baseline == Decimal("-0.042")


def test_descriptive_comparison_uses_prediction_cohorts_with_unequal_maturity_group_sizes():
    calendar = get_calendar("XNYS")
    target_sessions = calendar.sessions_window(calendar.date_to_session(date(2020, 1, 2)), 1500)
    records = []
    for cohort_index in range(8):
        target = target_sessions[cohort_index * 180].date()
        maturity = calendar.session_offset(calendar.date_to_session(target), 126).date()
        late_maturity = calendar.session_offset(calendar.date_to_session(target), 127).date()
        for listing_index in range(30):
            candidate_score = Decimal("0.04") if listing_index < 29 else Decimal("0.58")
            evaluation_date = maturity if listing_index < 29 else late_maturity
            records.append(
                {
                    "row": {
                        "listing_id": f"descriptive-{cohort_index}-{listing_index}",
                        "target_date": target,
                    },
                    "target_date": target,
                    "evaluation_date": evaluation_date,
                    "native": SimpleNamespace(
                        interval_score=Decimal("0.1"),
                        interval_included=True,
                        unavailable_reason=None,
                    ),
                    "candidate": SimpleNamespace(
                        interval_score=candidate_score,
                        interval_included=True,
                        unavailable_reason=None,
                    ),
                    "candidate_representable": True,
                }
            )

    descriptive = shadow._descriptive(records, horizon="6m")
    assert descriptive["target_cohort_count"] == 8
    assert descriptive["paired_observation_count"] == 240
    assert descriptive["overlap_aware_support"] == 8
    assert descriptive["support_reason"] is None
    assert descriptive["native_average"] == "0.1"
    assert descriptive["calibrated_average"] == "0.058"
    assert descriptive["mean_difference_candidate_minus_baseline"] == "-0.042"


def test_candidate_scorer_rejection_is_excluded_from_all_paired_metrics(
    prepared_shadow, monkeypatch
):
    user, store, _run, _study, _builder, fitted = prepared_shadow
    params = copy.deepcopy(fitted["params"])
    params["horizons"]["6m"]["offset"] = "0.02"
    witness = _FIXED_NOW
    start = shadow.forward_scorecard.prospective_start_target(witness)
    calendar = get_calendar("XNYS", start=start, end=start + timedelta(days=6 * 366))
    first_session = calendar.date_to_session(start, direction="next")
    now = _FIXED_NOW + timedelta(days=6 * 366)
    rows = []
    for cohort_index in range(8):
        target = calendar.session_offset(first_session, cohort_index * 180).date()
        maturity = calendar.session_offset(calendar.date_to_session(target), 126).date()
        for listing_index in range(31):
            row = _projection_row(
                target=target,
                generated_at=witness + timedelta(seconds=1),
                listing_id=f"scorer-rejection-{cohort_index}-{listing_index}",
            )
            row["outcome__actual_return"] = Decimal("0")
            row["outcome__evaluation_date"] = maturity
            row["outcome__evaluated_at"] = now - timedelta(days=1)
            if listing_index == 0:
                row.update(
                    bear_return=Decimal("-0.01"),
                    base_return=Decimal("0"),
                    bull_return=Decimal("999999.9999"),
                )
            rows.append(row)
    _patch_report_projection(monkeypatch, rows, current_time=now)
    report = shadow._build_report(
        user=user,
        store=store,
        provider="synthetic_demo",
        params=params,
        declaration={
            "witnessed_at": witness.isoformat(),
            "prospective_start_target": start.isoformat(),
        },
        attempts={"failed": 0, "latest_failure_code": None},
    )
    six = report["horizons"]["6m"]
    assert six["denominators"]["matured"] == 248
    assert six["denominators"]["candidate_score_unavailable"] == 8
    assert six["denominators"]["candidate_unrepresentable"] == 0
    assert six["denominators"]["native_score_unavailable"] == 0
    assert six["paired_evidence_count"] == 240
    assert six["descriptive_all_matured"]["paired_observation_count"] == 240
    assert [item["paired_scored_listings"] for item in report["primary_result"]["cohorts"]] == [
        30
    ] * 8
    assert report["companion_result"]["native"]["cluster_means"] == ["1"] * 8
    assert report["companion_result"]["calibrated"]["cluster_means"] == ["1"] * 8


@pytest.mark.parametrize(
    "native,offset_text,expected",
    [
        (LedgerTriplet(Decimal("-0.99"), Decimal("0"), Decimal("1000")), "-1", (False, False)),
        (LedgerTriplet(Decimal("0.2"), Decimal("0.3"), Decimal("100")), "-1", (True, False)),
        (LedgerTriplet(Decimal("-0.9"), Decimal("0.1"), Decimal("0.2")), "-1", (False, True)),
        (LedgerTriplet(Decimal("0.2"), Decimal("0.3"), Decimal("0.5")), "-1", (True, True)),
    ],
)
def test_median_clamp_counts_classify_lower_upper_both_and_none(native, offset_text, expected):
    offset = Decimal(offset_text)
    adjusted = shadow.adjust_interval(native.lower, native.median, native.upper, offset)
    assert adjusted is not None
    candidate = LedgerTriplet(*adjusted)
    assert shadow._median_clamp_sides(native, candidate, offset) == expected


def test_report_clamp_denominators_count_actual_median_clamps(prepared_shadow, monkeypatch):
    user, store, _run, _study, _builder, fitted = prepared_shadow
    params = copy.deepcopy(fitted["params"])
    params["horizons"]["6m"]["offset"] = "-1"
    witness = _FIXED_NOW
    start = shadow.forward_scorecard.prospective_start_target(witness)
    target = get_calendar("XNYS").date_to_session(start, direction="next").date()
    triplets = (
        LedgerTriplet(Decimal("-0.99"), Decimal("0"), Decimal("1000")),
        LedgerTriplet(Decimal("0.2"), Decimal("0.3"), Decimal("100")),
        LedgerTriplet(Decimal("-0.9"), Decimal("0.1"), Decimal("0.2")),
        LedgerTriplet(Decimal("0.2"), Decimal("0.3"), Decimal("0.5")),
    )
    rows = []
    for index, triplet in enumerate(triplets):
        row = _projection_row(
            target=target,
            generated_at=witness + timedelta(seconds=1),
            listing_id=f"clamp-case-{index}",
        )
        row.update(
            bear_return=triplet.lower,
            base_return=triplet.median,
            bull_return=triplet.upper,
        )
        rows.append(row)
    _patch_report_projection(monkeypatch, rows)
    report = shadow._build_report(
        user=user,
        store=store,
        provider="synthetic_demo",
        params=params,
        declaration={
            "witnessed_at": witness.isoformat(),
            "prospective_start_target": start.isoformat(),
        },
        attempts={"failed": 0, "latest_failure_code": None},
    )
    denominators = report["horizons"]["6m"]["denominators"]
    assert denominators["clamped_lower"] == 2
    assert denominators["clamped_upper"] == 2


def test_outcome_authentication_and_future_evaluation_fail_closed(monkeypatch):
    row = {"id": uuid4()}
    prediction = SimpleNamespace(
        outcome=SimpleNamespace(evaluated_at=_FIXED_NOW, evaluation_date=date(2026, 9, 14))
    )
    monkeypatch.setattr(
        Prediction.objects,
        "select_related",
        lambda *_args: SimpleNamespace(get=lambda **_kw: prediction),
    )
    verified = []
    monkeypatch.setattr(
        shadow,
        "verify_prediction_outcome",
        lambda *args, **kwargs: verified.append((args, kwargs)),
    )
    monkeypatch.setattr(timezone, "now", lambda: _FIXED_NOW)
    assert shadow._authenticated_outcome(row, "synthetic_demo") is prediction.outcome
    assert len(verified) == 1

    prediction.outcome.evaluated_at = _FIXED_NOW + timedelta(seconds=1)
    with pytest.raises(shadow.IntervalShadowError) as future:
        shadow._authenticated_outcome(row, "synthetic_demo")
    assert future.value.code == "interval_shadow_outcome_unverified"


def test_prospective_scoring_adapter_counts_native_and_calibrated_pair(monkeypatch):
    fitted = fit_interval_calibration_document(build_interval_calibration_document())
    witness = datetime(2026, 9, 13, 22, tzinfo=UTC)
    start = shadow.forward_scorecard.prospective_start_target(witness)
    calendar = get_calendar("XNYS")
    target = calendar.date_to_session(start, direction="next").date()
    maturity = calendar.session_offset(calendar.date_to_session(target), 126).date()
    now = datetime(2030, 1, 1, tzinfo=UTC)
    row = {
        "id": uuid4(),
        "listing_id": "synthetic-prospective-listing",
        "target_date": target,
        "generated_at": witness + timedelta(seconds=1),
        "evidence_role": "advisory",
        "method_version": shadow.FHS_METHOD_VERSION,
        "config_hash": shadow.PRODUCT_EFFECTIVE_CONFIG_HASH,
        "price_provider": "synthetic_demo",
        "horizon": "6m",
        "bear_return": Decimal("-0.01"),
        "base_return": Decimal("0"),
        "bull_return": Decimal("0.01"),
        "reportable": True,
        "canonical": True,
        "outcome__status": "matured",
        "outcome__evaluation_date": maturity,
        "outcome__evaluated_at": now - timedelta(days=1),
        "outcome__actual_return": Decimal("0.03"),
    }
    scope = SimpleNamespace(
        available=True,
        runs=(
            SimpleNamespace(run=SimpleNamespace(config_version=shadow.PRODUCT_VERSION, id=uuid4())),
        ),
    )
    monkeypatch.setattr(shadow, "read_research_product_performance", lambda **_kwargs: scope)
    monkeypatch.setattr(
        shadow, "iter_product_prediction_projection", lambda *_a, **_kw: iter((row,))
    )
    monkeypatch.setattr(
        shadow,
        "_authenticated_outcome",
        lambda *_a, **_kw: SimpleNamespace(
            status="matured",
            evaluated_at=now - timedelta(days=1),
            evaluation_date=maturity,
            actual_return=Decimal("0.03"),
        ),
    )
    monkeypatch.setattr(timezone, "now", lambda: now)
    report = shadow._build_report(
        user=SimpleNamespace(pk="synthetic", is_authenticated=True),
        store=AssetStore.__new__(AssetStore),
        provider="synthetic_demo",
        params=fitted["params"],
        declaration={
            "witnessed_at": witness.isoformat(),
            "prospective_start_target": start.isoformat(),
        },
        attempts={"failed": 0, "latest_failure_code": None},
    )
    six = report["horizons"]["6m"]
    assert report["status"] == "available"
    assert six["denominators"]["eligible"] == 1
    assert six["denominators"]["matured"] == 1
    assert six["paired_evidence_count"] == 1
    assert six["descriptive_all_matured"]["paired_observation_count"] == 1
    assert six["descriptive_all_matured"]["native_average"] is None


def test_eligibility_boundaries_do_not_replace_canonical_rows_with_reissues(
    prepared_shadow, monkeypatch
):
    user, store, *_run, _study, _builder, fitted = prepared_shadow
    witness = _FIXED_NOW
    start = shadow.forward_scorecard.prospective_start_target(witness)
    calendar = get_calendar("XNYS")
    target = calendar.date_to_session(start, direction="next").date()
    before_target = calendar.session_offset(calendar.date_to_session(target), -1).date()
    reissued_listing = "same-listing-reissue-case"
    rows = [
        _projection_row(target=target, generated_at=witness + timedelta(seconds=1)),
        _projection_row(target=target, generated_at=witness),
        _projection_row(target=before_target, generated_at=witness + timedelta(seconds=1)),
        _projection_row(
            target=target,
            generated_at=witness - timedelta(seconds=1),
            listing_id=reissued_listing,
        ),
        _projection_row(
            target=target,
            generated_at=witness + timedelta(seconds=2),
            canonical=False,
            listing_id=reissued_listing,
        ),
        _projection_row(target=target, generated_at=witness + timedelta(seconds=1)),
        _projection_row(target=target, generated_at=witness + timedelta(seconds=1)),
    ]
    rows[-2]["config_hash"] = "0" * 64
    rows[-1]["price_provider"] = "another_synthetic_provider"
    non_evaluable = _projection_row(target=target, generated_at=witness + timedelta(seconds=1))
    non_evaluable["bear_return"] = None
    non_evaluable["base_return"] = None
    non_evaluable["bull_return"] = None
    rows.append(non_evaluable)
    _patch_report_projection(monkeypatch, rows)
    report = shadow._build_report(
        user=user,
        store=store,
        provider="synthetic_demo",
        params=fitted["params"],
        declaration={
            "witnessed_at": witness.isoformat(),
            "prospective_start_target": start.isoformat(),
        },
        attempts={"failed": 0, "latest_failure_code": None},
    )
    counts = report["horizons"]["6m"]["denominators"]
    assert counts["raw_rows"] == 8
    assert counts["non_evaluable"] == 1
    assert counts["config_or_provider_mismatch"] == 2
    assert counts["pre_declaration_excluded"] == 3
    assert counts["not_canonical"] == 1
    assert counts["excluded_reissue"] == 1
    assert counts["eligible"] == 1
    assert counts["matured"] == 1


def test_unrepresentable_candidate_is_counted_not_dropped(prepared_shadow, monkeypatch):
    user, store, _run, _study, _builder, fitted = prepared_shadow
    witness = _FIXED_NOW
    start = shadow.forward_scorecard.prospective_start_target(witness)
    target = get_calendar("XNYS").date_to_session(start, direction="next").date()
    _patch_report_projection(
        monkeypatch, [_projection_row(target=target, generated_at=witness + timedelta(seconds=1))]
    )
    monkeypatch.setattr(shadow, "adjust_interval", lambda *_args: None)
    report = shadow._build_report(
        user=user,
        store=store,
        provider="synthetic_demo",
        params=fitted["params"],
        declaration={
            "witnessed_at": witness.isoformat(),
            "prospective_start_target": start.isoformat(),
        },
        attempts={"failed": 0, "latest_failure_code": None},
    )
    six = report["horizons"]["6m"]
    assert six["denominators"]["matured"] == 1
    assert six["denominators"]["candidate_unrepresentable"] == 1
    assert six["paired_evidence_count"] == 0


@pytest.mark.parametrize("failure", ["verifier_raises", "future_evaluated_at"])
def test_outcome_integrity_failures_make_report_integrity_failed(
    prepared_shadow, monkeypatch, failure
):
    user, store, *_ = prepared_shadow
    shadow.declare_interval_shadow(user=user, store=store)
    witness = _FIXED_NOW
    start = shadow.forward_scorecard.prospective_start_target(witness)
    target = get_calendar("XNYS").date_to_session(start, direction="next").date()
    row = _projection_row(target=target, generated_at=witness + timedelta(seconds=1))
    _patch_report_projection(monkeypatch, [row])
    if failure == "verifier_raises":
        monkeypatch.setattr(
            shadow,
            "_authenticated_outcome",
            lambda *_args: (_ for _ in ()).throw(
                shadow.IntervalShadowError("interval_shadow_outcome_unverified")
            ),
        )
    else:
        monkeypatch.setattr(
            shadow,
            "_authenticated_outcome",
            lambda *_args: SimpleNamespace(
                status="matured",
                evaluated_at=_FIXED_NOW + timedelta(seconds=1),
                evaluation_date=row["outcome__evaluation_date"],
                actual_return=row["outcome__actual_return"],
            ),
        )
    report = shadow.read_interval_shadow_report(user=user, store=store)
    assert set(report) == _report_keys()
    assert report["status"] == "integrity_failed"
    assert report["verification_code"] == "interval_shadow_outcome_unverified"


def test_demo_primary_met_never_qualifies_s4_gate(monkeypatch):
    fitted = fit_interval_calibration_document(build_interval_calibration_document())
    _patch_report_projection(monkeypatch, [])
    monkeypatch.setattr(
        shadow,
        "_primary_and_companion",
        lambda _records: (
            {"state": "met"},
            {"met": True},
        ),
    )
    report = shadow._build_report(
        user=SimpleNamespace(pk="synthetic-owner", is_authenticated=True),
        store=AssetStore.__new__(AssetStore),
        provider="synthetic_demo",
        params=fitted["params"],
        declaration={
            "witnessed_at": _FIXED_NOW.isoformat(),
            "prospective_start_target": shadow.forward_scorecard.prospective_start_target(
                _FIXED_NOW
            ).isoformat(),
        },
        attempts={"failed": 0, "latest_failure_code": None},
    )
    assert report["primary_result"]["state"] == "met"
    assert report["companion_result"]["met"] is True
    assert report["s4_gate"]["state"] == "not_eligible_synthetic"


def test_job_witness_rejects_wrong_identity_and_early_finish(prepared_shadow):
    user, store, *_ = prepared_shadow
    asset = shadow.declare_interval_shadow(user=user, store=store)
    document = _read_declaration(store, asset)
    identity = shadow._identity(
        document["owner_id"], document["source_owner_id"], document["source_provider"]
    )
    job_name = shadow._job_name(identity)
    witness = shadow._witness(asset, identity=identity, job_name=job_name)
    assert witness is not None
    run = JobRun.objects.get(pk=document["producer_job_id"])
    run.details = {
        "identity": {**identity, "owner_id": "wrong"},
        "assets": [asset_ref_for(asset).to_json()],
    }
    run.save(update_fields=["details"])
    assert shadow._witness(asset, identity=identity, job_name=job_name) is None


@pytest.mark.parametrize(
    "attack,expected_code",
    [
        ("future_declared_at", "interval_shadow_clock_invalid"),
        ("document_asset_clock_mismatch", "interval_shadow_clock_invalid"),
        ("metadata_clock_mismatch", "interval_shadow_clock_invalid"),
        ("asset_clock_mismatch", "interval_shadow_clock_invalid"),
        ("study_available_after_declaration", "interval_shadow_owner_mismatch"),
    ],
)
def test_declaration_document_and_asset_clock_attacks(prepared_shadow, attack, expected_code):
    user, store, *_ = prepared_shadow
    asset = shadow.declare_interval_shadow(user=user, store=store)
    document = _read_declaration(store, asset)
    metadata = copy.deepcopy(asset.metadata)
    available_at = asset.available_at
    retrieved_at = asset.retrieved_at

    if attack == "future_declared_at":
        declared_at = _FIXED_NOW + timedelta(days=1)
        document["declared_at"] = declared_at.isoformat()
        metadata["declared_at"] = declared_at.isoformat()
        available_at = declared_at
        retrieved_at = declared_at
    elif attack == "document_asset_clock_mismatch":
        document["declared_at"] = (_FIXED_NOW + timedelta(seconds=1)).isoformat()
    elif attack == "metadata_clock_mismatch":
        metadata["declared_at"] = (_FIXED_NOW + timedelta(seconds=1)).isoformat()
    elif attack == "asset_clock_mismatch":
        available_at = _FIXED_NOW + timedelta(seconds=1)
    elif attack == "study_available_after_declaration":
        study = DataAsset.objects.get(pk=document["study"]["id"])
        declared_at = study.available_at - timedelta(seconds=1)
        document["declared_at"] = declared_at.isoformat()
        metadata["declared_at"] = declared_at.isoformat()
        available_at = declared_at
        retrieved_at = declared_at

    copied = _copy_declaration(
        asset=asset,
        document=document,
        store=store,
        metadata=metadata,
        available_at=available_at,
        retrieved_at=retrieved_at,
    )
    with pytest.raises(shadow.IntervalShadowError) as rejected:
        _verify_declaration(
            copied, document=document, user=user, store=store, require_witness=False
        )
    assert rejected.value.code == expected_code


def test_producer_clock_and_witness_attacks(prepared_shadow):
    user, store, *_ = prepared_shadow
    asset = shadow.declare_interval_shadow(user=user, store=store)
    document = _read_declaration(store, asset)
    producer = JobRun.objects.get(pk=document["producer_job_id"])

    JobRun.objects.filter(pk=producer.pk).update(
        started_at=asset.available_at + timedelta(seconds=1)
    )
    with pytest.raises(shadow.IntervalShadowError) as late_start:
        _verify_declaration(asset, document=document, user=user, store=store)
    assert late_start.value.code == "interval_shadow_clock_invalid"

    JobRun.objects.filter(pk=producer.pk).update(
        started_at=asset.available_at - timedelta(seconds=2),
        finished_at=asset.available_at - timedelta(seconds=1),
    )
    with pytest.raises(shadow.IntervalShadowError) as early_success:
        _verify_declaration(asset, document=document, user=user, store=store)
    assert early_success.value.code == "interval_shadow_publication_unverified"

    JobRun.objects.filter(pk=producer.pk).update(
        started_at=asset.available_at - timedelta(seconds=1),
        finished_at=_FIXED_NOW + timedelta(seconds=1),
    )
    with pytest.raises(shadow.IntervalShadowError) as future_witness:
        _verify_declaration(asset, document=document, user=user, store=store)
    assert future_witness.value.code == "interval_shadow_clock_invalid"


def test_backdated_unwitnessed_declaration_is_rejected(prepared_shadow, monkeypatch):
    user, store, *_ = prepared_shadow
    asset = shadow.declare_interval_shadow(user=user, store=store)
    document = _read_declaration(store, asset)
    backdated = _FIXED_NOW + timedelta(seconds=1)
    document["declared_at"] = backdated.isoformat()
    metadata = copy.deepcopy(asset.metadata)
    metadata["declared_at"] = backdated.isoformat()
    producer = JobRun.objects.get(pk=document["producer_job_id"])
    JobRun.objects.filter(pk=producer.pk).update(
        started_at=backdated - timedelta(seconds=1),
        finished_at=backdated - timedelta(milliseconds=1),
    )
    copied = _copy_declaration(
        asset=asset,
        document=document,
        store=store,
        metadata=metadata,
        available_at=backdated,
        retrieved_at=backdated,
    )
    monkeypatch.setattr(timezone, "now", lambda: _FIXED_NOW + timedelta(seconds=10))
    with pytest.raises(shadow.IntervalShadowError) as unwitnessed:
        _verify_declaration(copied, document=document, user=user, store=store)
    assert unwitnessed.value.code == "interval_shadow_publication_unverified"


def test_owner_isolation_and_copied_declaration_subject(prepared_shadow):
    user, store, *_ = prepared_shadow
    asset = shadow.declare_interval_shadow(user=user, store=store)
    other_user = get_user_model().objects.create_user(username="other-shadow-owner")
    other_report = shadow.read_interval_shadow_report(user=other_user, store=store)
    assert other_report["status"] == "not_declared"

    document = _read_declaration(store, asset)
    copied = _copy_declaration(
        asset=asset,
        document=document,
        store=store,
        subject=f"{shadow.PROTOCOL_VERSION}:{other_user.pk}",
    )
    with pytest.raises(shadow.IntervalShadowError) as wrong_owner:
        _verify_declaration(
            copied, document=document, user=user, store=store, require_witness=False
        )
    assert wrong_owner.value.code == "interval_shadow_owner_mismatch"


def test_other_owners_study_cannot_be_bound_to_declaration(prepared_shadow):
    user, store, _run, study, *_ = prepared_shadow
    asset = shadow.declare_interval_shadow(user=user, store=store)
    document = _read_declaration(store, asset)
    other_owner = get_user_model().objects.create_user(username="shadow-study-owner")
    other_metadata = copy.deepcopy(study.metadata)
    other_metadata["owner_id"] = str(other_owner.pk)
    other_subject = study.subject.replace(f":{study.metadata['owner_id']}:", f":{other_owner.pk}:")
    stored = store.write_bytes(
        f"research/interval_shadow/test-copies/other-study-{study.pk}.json",
        store.read_bytes(study.relative_path),
    )
    foreign_study = DataAsset.objects.create(
        provider=study.provider,
        kind=study.kind,
        subject=other_subject,
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        retrieved_at=study.retrieved_at,
        available_at=study.available_at,
        metadata=other_metadata,
    )
    document["study"] = asset_ref_for(foreign_study).to_json()
    document["study_full_report_sha256"] = other_metadata["full_report_sha256"]
    metadata = copy.deepcopy(asset.metadata)
    metadata["study_asset_id"] = str(foreign_study.pk)
    copied = _copy_declaration(
        asset=asset,
        document=document,
        store=store,
        metadata=metadata,
    )
    with pytest.raises(shadow.IntervalShadowError) as wrong_study:
        _verify_declaration(
            copied, document=document, user=user, store=store, require_witness=False
        )
    assert wrong_study.value.code == "interval_shadow_source_study_unverified"


def test_concurrent_declaration_has_one_asset_and_one_success(prepared_shadow):
    user, store, *_ = prepared_shadow
    barrier = Barrier(2)
    guard = Lock()
    results: list[tuple[str, bytes]] = []

    def publish() -> tuple[str, bytes]:
        connections.close_all()
        try:
            owner = get_user_model().objects.get(pk=user.pk)
            barrier.wait(timeout=10)
            asset = shadow.declare_interval_shadow(user=owner, store=store)
            value = (str(asset.pk), store.read_bytes(asset.relative_path))
            with guard:
                results.append(value)
            return value
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as pool:
        completed = [
            future.result(timeout=60) for future in [pool.submit(publish) for _ in range(2)]
        ]
    assert completed[0] == completed[1]
    assert DataAsset.objects.filter(kind=shadow.DECLARATION_KIND).count() == 1
    assert (
        JobRun.objects.filter(
            job_name__startswith=shadow.DECLARE_JOB,
            status=JobRun.Status.SUCCESS,
        ).count()
        == 1
    )


def test_failure_output_is_aggregate_and_does_not_expose_listing_or_paths(
    prepared_shadow, monkeypatch
):
    user, store, *_ = prepared_shadow
    asset = shadow.declare_interval_shadow(user=user, store=store)
    original = shadow._fit_params

    def drifted(document):
        fitted = copy.deepcopy(original(document))
        fitted["params"]["horizons"]["6m"]["offset"] = "99"
        return fitted

    monkeypatch.setattr(shadow, "_fit_params", drifted)
    report = shadow.read_interval_shadow_report(user=user, store=store)
    rendered = json.dumps(report)
    assert report["status"] == "integrity_failed"
    assert str(asset.relative_path) not in rendered
    assert "AAPL" not in rendered
    assert "Traceback" not in rendered

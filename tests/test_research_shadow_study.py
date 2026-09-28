"""Prospective protocol tests use isolated synthetic state, never installed data."""

import inspect
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from django.db import transaction
from django.utils import timezone

from stanstock.core.models import JobRun
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset
from stanstock.research import shadow_study as study

pytestmark = pytest.mark.django_db
REVISION = "a" * 40


@pytest.fixture
def study_environment(settings, tmp_path, django_user_model, monkeypatch):
    settings.DATA_DIR = tmp_path
    settings.SHADOW_STUDY_ENABLED = True
    settings.DEMO_MODE = True
    owner = django_user_model.objects.create_user(username="study-owner")
    clock = [datetime(2026, 9, 11, 19, tzinfo=UTC)]
    monkeypatch.setattr(timezone, "now", lambda: clock[0])
    monkeypatch.setattr(study, "clean_git_revision", lambda root: REVISION)
    return owner, AssetStore(tmp_path), clock


def test_protocol_pin_commit_epoch_and_independent_witness(study_environment):
    owner, store, clock = study_environment
    assert study._hash(study.PROTOCOL_BODY) == (
        "452f02fa105b528f4a9dc46825cb1712c8d2142a88d1c7a2bda5cf09d1efc907"
    )
    protocol = study.register_shadow_protocol(owner=owner, store=store)
    clock[0] += timedelta(seconds=1)
    activation = study.activate_shadow_study(owner=owner, store=store)
    document = study._read_asset(activation, store=store)
    assert protocol.available_at < study._parse_time(document["body"]["t0"])
    assert document["body"]["s0"] == "2026-09-11"
    assert len(activation.metadata) == 16
    witness = study._witness(
        activation,
        document=document,
        protocol=protocol,
        activation=None,
    )
    assert witness == clock[0]
    clock[0] += timedelta(days=10)
    assert study.activate_shadow_study(owner=owner, store=store).pk == activation.pk
    assert (
        study._witness(
            activation,
            document=document,
            protocol=protocol,
            activation=None,
        )
        == witness
    )
    assert JobRun.objects.filter(status="skipped").count() == 1


@pytest.mark.parametrize(
    ("instant", "expected"),
    [
        (datetime(2026, 9, 11, 19, 59, 59, tzinfo=UTC), date(2026, 9, 11)),
        (datetime(2026, 9, 11, 20, tzinfo=UTC), date(2026, 9, 14)),
        (datetime(2026, 11, 27, 18, tzinfo=UTC), date(2026, 11, 30)),
        (datetime(2026, 11, 27, 17, 59, tzinfo=UTC), date(2026, 11, 27)),
        (datetime(2026, 9, 7, 16, tzinfo=UTC), date(2026, 9, 8)),
        (datetime(2026, 3, 9, 20, tzinfo=UTC), date(2026, 3, 10)),
    ],
)
def test_strict_close_epoch_holiday_dst_and_early_close(instant, expected):
    assert study._epoch(instant)[0] == expected


def test_outer_transaction_cannot_release_registration_lock(study_environment):
    owner, store, _clock = study_environment
    with transaction.atomic():
        with pytest.raises(RuntimeError, match="durable atomic"):
            study.register_shadow_protocol(owner=owner, store=store)
    assert not DataAsset.objects.exists()
    assert not list(store.root.rglob("*.json"))


def test_flag_off_activation_does_not_create_evidence(study_environment, settings):
    owner, store, _clock = study_environment
    settings.SHADOW_STUDY_ENABLED = False
    with pytest.raises(RefreshVerificationError, match="shadow_activation_missing"):
        study.activate_shadow_study(owner=owner, store=store)
    assert not DataAsset.objects.exists()
    assert not JobRun.objects.exists()


def test_closed_epoch_does_not_reopen_when_disabled(study_environment, settings):
    owner, store, clock = study_environment
    activation = study.activate_shadow_study(owner=owner, store=store)
    settings.SHADOW_STUDY_ENABLED = False
    clock[0] += timedelta(days=4)
    closure = study.close_shadow_study(owner=owner, store=store)
    closed = study._read_asset(closure, store=store)["body"]["closed_at"]
    clock[0] += timedelta(days=4)
    assert study.close_shadow_study(owner=owner, store=store).pk == closure.pk
    assert study._read_asset(closure, store=store)["body"]["closed_at"] == closed
    settings.SHADOW_STUDY_ENABLED = True
    assert study.activate_shadow_study(owner=owner, store=store).pk == activation.pk


def test_later_closure_commit_crash_preserves_historical_read_and_exact_recovery(
    study_environment, monkeypatch
):
    from django.db.models.query import QuerySet

    owner, store, clock = study_environment
    activation = study.activate_shadow_study(owner=owner, store=store)
    clock[0] += timedelta(hours=2)
    historical_time = clock[0]
    historical = study.read_shadow_study(owner=owner, as_of=historical_time, store=store)
    assert historical.status == "available" and historical.phase == "collecting"
    clock[0] += timedelta(days=1)
    original_update = QuerySet.update

    def crash_completion(queryset, **kwargs):
        if queryset.model is JobRun and kwargs.get("status") == JobRun.Status.SUCCESS:
            raise RuntimeError("synthetic closure completion crash")
        return original_update(queryset, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(QuerySet, "update", crash_completion)
        with pytest.raises(RuntimeError, match="synthetic closure completion"):
            study.close_shadow_study(owner=owner, store=store)
    closure = DataAsset.objects.get(kind=study.KINDS[2])
    document = study._read_asset(closure, store=store)
    original_bytes = store.read_bytes(closure.relative_path)
    protocol = DataAsset.objects.get(kind=study.KINDS[0])
    assert closure.available_at == clock[0] > historical_time
    assert (
        study._witness(closure, document=document, protocol=protocol, activation=activation) is None
    )
    assert study.read_shadow_study(owner=owner, as_of=historical_time, store=store) == historical
    applicable = study.read_shadow_study(owner=owner, as_of=closure.available_at, store=store)
    assert applicable.status == "integrity_failed"
    assert applicable.reason_code == "shadow_publication_unverified"

    clock[0] += timedelta(days=1)
    recovered = study.close_shadow_study(owner=owner, store=store)
    assert recovered.pk == closure.pk
    assert DataAsset.objects.filter(kind=study.KINDS[2]).count() == 1
    assert store.read_bytes(recovered.relative_path) == original_bytes
    assert study._read_asset(recovered, store=store)["body"] == document["body"]
    witness = study._witness(recovered, document=document, protocol=protocol, activation=activation)
    assert witness == clock[0] > closure.available_at
    assert study.read_shadow_study(owner=owner, as_of=historical_time, store=store) == historical
    before_completion = study.read_shadow_study(
        owner=owner, as_of=witness - timedelta(microseconds=1), store=store
    )
    assert before_completion.status == "available"
    assert before_completion.phase == "collecting" and before_completion.closure is None
    closed = study.read_shadow_study(owner=owner, as_of=witness, store=store)
    assert closed.status == "available" and closed.phase == "closed"
    assert closed.closure is not None and closed.closure.id == closure.pk
    clock[0] += timedelta(days=1)
    assert study.close_shadow_study(owner=owner, store=store).pk == closure.pk
    assert (
        study._witness(closure, document=document, protocol=protocol, activation=activation)
        == witness
    )


@pytest.mark.parametrize("failure", ["corrupt", "ambiguous"])
def test_closure_validation_is_scoped_before_historical_lookup(study_environment, failure):
    owner, store, clock = study_environment
    study.activate_shadow_study(owner=owner, store=store)
    clock[0] += timedelta(hours=2)
    historical_time = clock[0]
    historical = study.read_shadow_study(owner=owner, as_of=historical_time, store=store)
    clock[0] += timedelta(days=1)
    closure = study.close_shadow_study(owner=owner, store=store)
    closed_time = clock[0]
    closed = study.read_shadow_study(owner=owner, as_of=closed_time, store=store)
    assert closed.status == "available" and closed.phase == "closed"
    clock[0] += timedelta(seconds=1)
    if failure == "corrupt":
        store.resolve(closure.relative_path).write_bytes(b"synthetic corrupt closure")
    else:
        document = study._read_asset(closure, store=store)
        document["recorded_at"] = study._time(clock[0])
        document["body"]["closed_at"] = study._time(clock[0])
        stored = store.write_bytes(
            f"research/shadow/{closure.kind}/{study._hash(document)}.json",
            study._canonical_bytes(document),
        )
        DataAsset.objects.create(
            provider=closure.provider,
            kind=closure.kind,
            subject=closure.subject,
            relative_path=stored.relative_path,
            sha256=stored.sha256,
            available_at=clock[0],
            retrieved_at=clock[0],
            schema_version=closure.schema_version,
            metadata=study._metadata(document, closure.kind),
        )
        assert study.read_shadow_study(owner=owner, as_of=closed_time, store=store) == closed
    assert study.read_shadow_study(owner=owner, as_of=historical_time, store=store) == historical
    applicable = study.read_shadow_study(owner=owner, as_of=clock[0], store=store)
    assert applicable.status == "integrity_failed"
    assert applicable.reason_code == (
        "shadow_asset_invalid" if failure == "corrupt" else "shadow_registry_ambiguous"
    )
    with pytest.raises(RefreshVerificationError, match=applicable.reason_code):
        study.close_shadow_study(owner=owner, store=store)


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "study_id",
        "owner_id",
        "source_provider",
        "protocol",
        "activation",
        "execution_revision",
        "recorded_at",
        "producer_job_id",
        "body",
    ],
)
def test_protocol_closed_envelope(study_environment, field):
    owner, store, _clock = study_environment
    protocol = study.register_shadow_protocol(owner=owner, store=store)
    document = study._read_asset(protocol, store=store)
    del document[field]
    with pytest.raises(RefreshVerificationError):
        study._validate_document(document, study.KINDS[0])
    document = study._read_asset(protocol, store=store)
    document["unexpected"] = None
    with pytest.raises(RefreshVerificationError):
        study._validate_document(document, study.KINDS[0])


def test_protocol_cannot_self_authenticate_mutated_definition(study_environment, monkeypatch):
    owner, store, _clock = study_environment
    changed = deepcopy(study.PROTOCOL_BODY)
    changed["native"]["path_count"] = 1
    monkeypatch.setattr(study, "PROTOCOL_BODY", changed)
    with pytest.raises(RefreshVerificationError, match="shadow_protocol_mismatch"):
        study.register_shadow_protocol(owner=owner, store=store)
    assert not DataAsset.objects.exists()


def test_current_owner_rechecked_even_for_existing_activation(study_environment):
    owner, store, _clock = study_environment
    study.activate_shadow_study(owner=owner, store=store)
    type(owner).objects.filter(pk=owner.pk).update(is_active=False)
    with pytest.raises(RefreshVerificationError, match="shadow_owner_unauthorized"):
        study.activate_shadow_study(owner=owner, store=store)


def test_asof_missing_activation_disabled_and_unauthorized_are_distinct(
    study_environment, settings
):
    owner, store, clock = study_environment
    assert (
        study.read_shadow_study(owner=owner, as_of=clock[0], store=store).status
        == "activation_missing"
    )
    assert not JobRun.objects.exists()
    assert not DataAsset.objects.exists()
    study.activate_shadow_study(owner=owner, store=store)
    assert (
        study.read_shadow_study(
            owner=owner, as_of=clock[0] - timedelta(seconds=1), store=store
        ).status
        == "activation_missing"
    )
    assert study.read_shadow_study(owner=None, as_of=clock[0], store=store).status == "unauthorized"
    settings.SHADOW_STUDY_ENABLED = False
    assert study.read_shadow_study(owner=None, as_of=clock[0], store=store).status == "disabled"


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "study_id",
        "owner_id",
        "source_provider",
        "protocol_id",
        "protocol_sha256",
        "activation_id",
        "activation_sha256",
        "execution_revision",
        "recorded_at",
        "producer_job_id",
        "identity_sha256",
        "target_date",
        "listing_id",
        "horizon",
        "evidence_key",
    ],
)
def test_all_metadata_fields_are_checked_against_payload(study_environment, field):
    owner, store, _clock = study_environment
    asset = study.activate_shadow_study(owner=owner, store=store)
    asset.metadata = {**asset.metadata, field: "invalid"}
    with pytest.raises(RefreshVerificationError, match="shadow_asset_invalid"):
        study._read_asset(asset, store=store)


def test_phase_anchors_permanent_and_never_rebased_on_missing_capture(study_environment):
    owner, store, _clock = study_environment
    asset = study.activate_shadow_study(owner=owner, store=store)
    document = study._read_asset(asset, store=store)
    calendar = study._calendar()
    for index in (0, 1, 125, 126, 251, 252):
        target = calendar.session_offset(date(2026, 9, 11), index).date()
        session = study._session(document, target)
        assert session["session_index"] == index
        assert session["primary_anchor"] == (index % 126 == 0)
        assert session["secondary_anchor"] == (index % 252 == 0)


def test_grouping_is_common_support_then_equal_anchor_not_pooled():
    def scores(value):
        return [
            {"arm_id": arm, "interval_score": str(value), "median_absolute_error": str(value)}
            for arm in study.ARMS
        ]

    first = study._means([scores(1), scores(1), scores(1)])
    second = study._means([scores(9)])
    anchors = [
        study.AnchorSummary(date(2026, 9, 11), 3, 3, 3, 0, 0, 0, 0, first, None),
        study.AnchorSummary(date(2027, 3, 15), 1, 1, 1, 0, 0, 0, 1, second, None),
    ]
    summary = study._summary("6m", anchors)
    assert summary.intended_cases == summary.paired == 4
    assert summary.challenged == 1
    assert summary.arm_means[0]["interval_score"] == Decimal(5)
    assert summary.candidate_minus_control[0]["interval_score"] == 0
    anchors.append(
        study.AnchorSummary(
            date(2027, 9, 14), None, 0, 0, 0, None, 0, 0, None, "shadow_population_missing"
        )
    )
    unavailable = study._summary("6m", anchors)
    assert unavailable.arm_means is None and unavailable.candidate_minus_control is None
    assert unavailable.intended_cases is None and unavailable.paired == 4


def test_missing_population_pending_retry_then_missed_is_not_zero(study_environment):
    from stanstock.research.shadow_jobs import execute_shadow_capture_job

    owner, store, clock = study_environment
    activation = study.activate_shadow_study(owner=owner, store=store)
    target = date(2026, 9, 11)
    clock[0] += timedelta(hours=3)
    pending = execute_shadow_capture_job(owner=owner, target_date=target, store=store)
    assert pending.state == "pending"
    assert not DataAsset.objects.filter(kind=study.KINDS[3]).exists()
    clock[0] += timedelta(days=3)
    missing = execute_shadow_capture_job(owner=owner, target_date=target, store=store)
    assert missing.state == "missing_population"
    asset = DataAsset.objects.get(kind=study.KINDS[3])
    body = study._read_asset(asset, store=store)["body"]
    assert body["population_count"] is None and body["members"] == []
    assert body["session_index"] == 0
    assert study._read_asset(activation, store=store)["body"]["s0"] == target.isoformat()


def test_activation_commit_crash_keeps_original_epoch_and_late_witness(
    study_environment, monkeypatch
):
    from django.db.models.query import QuerySet

    owner, store, clock = study_environment
    original = QuerySet.update

    def crash(queryset, **kwargs):
        if queryset.model is JobRun and kwargs.get("status") == "success":
            raise RuntimeError("synthetic completion crash")
        return original(queryset, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(QuerySet, "update", crash)
        with pytest.raises(RuntimeError, match="synthetic completion"):
            study.activate_shadow_study(owner=owner, store=store)
    asset = DataAsset.objects.get(kind=study.KINDS[1])
    document = study._read_asset(asset, store=store)
    original_body = deepcopy(document["body"])
    clock[0] += timedelta(hours=2)
    recovered = study.activate_shadow_study(owner=owner, store=store)
    assert recovered.pk == asset.pk
    assert study._read_asset(recovered, store=store)["body"] == original_body
    protocol = DataAsset.objects.get(kind=study.KINDS[0])
    assert (
        study._witness(
            asset,
            document=document,
            protocol=protocol,
            activation=None,
        )
        == clock[0]
    )
    assert clock[0] > study._parse_time(original_body["s0_scheduled_close"])


def _concurrent_activation_probe(workdir):
    from concurrent.futures import ThreadPoolExecutor
    from datetime import UTC, datetime, timedelta
    from itertools import count

    import django
    from django.conf import settings
    from django.contrib.auth import get_user_model
    from django.core.management import call_command
    from django.db import connections
    from django.utils import timezone

    settings.DATABASES["default"]["NAME"] = str(workdir / "isolated.sqlite3")
    settings.DATA_DIR = workdir / "assets"
    settings.DEMO_MODE = True
    settings.SHADOW_STUDY_ENABLED = True
    django.setup()
    from stanstock.core.models import JobRun
    from stanstock.data.assets import AssetStore
    from stanstock.data.models import DataAsset
    from stanstock.research import shadow_study as study

    call_command("migrate", verbosity=0)
    owner = get_user_model().objects.create_user(username="isolated-study")
    ticks = count()
    timezone.now = lambda: (
        datetime(2026, 9, 11, 19, tzinfo=UTC) + timedelta(microseconds=next(ticks))
    )
    study.clean_git_revision = lambda root: "a" * 40

    def activate(_index):
        try:
            store = AssetStore(settings.DATA_DIR)
            asset = study.activate_shadow_study(owner=owner, store=store)
            study._read_asset(asset, store=store)
            return asset.pk
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as executor:
        identities = list(executor.map(activate, range(2)))
    assert identities[0] == identities[1]
    assert DataAsset.objects.filter(kind=study.KINDS[1]).count() == 1
    assert JobRun.objects.filter(status="success").count() == 1
    assert JobRun.objects.filter(status="skipped").count() == 1


def test_sqlite_concurrent_registration_keeps_one_committed_epoch(tmp_path):
    import test_research_price_product_shadow_drift as adapter_tests

    code = (
        "import sys\nfrom pathlib import Path\n"
        + inspect.getsource(_concurrent_activation_probe)
        + "\n_concurrent_activation_probe(Path(sys.argv[1]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=tmp_path,
        env=adapter_tests._process_environment(adapter_tests.ROOT / "src"),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr

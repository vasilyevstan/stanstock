from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest
from django.urls import reverse
from django.utils import timezone

from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import DataAsset, ProviderRecord
from stanstock.data.provider_policy import BASIC_USAGE_SCOPE
from stanstock.data.research_product_demo import execute_demo_product_refresh
from stanstock.data.research_product_jobs import execute_daily_research_job
from stanstock.research.models import AnalysisRun, Prediction, PredictionOutcome, StockAnalysis
from stanstock.research.price_product_config import PRODUCT_VERSION
from stanstock.research.price_product_study import (
    serialize_price_product_study,
    study_price_product_run,
)
from stanstock.research.product_study_evidence import (
    STUDY_EVIDENCE_KIND,
    _register_price_product_study,
    read_registered_price_product_study,
    register_price_product_study,
)
from test_research_product_jobs import (
    TARGET,
    _persist_price_series,
    _series,
    make_product_environment,
)

pytestmark = pytest.mark.django_db


@pytest.fixture
def demo_product(settings, tmp_path, monkeypatch, django_user_model):
    monkeypatch.setattr(timezone, "now", lambda: datetime(2026, 9, 13, 17, tzinfo=UTC))
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = True
    settings.DATA_DIR = tmp_path
    blocked = Mock(
        side_effect=AssertionError(
            "Product-study evidence tests must not resolve credentials or use the network"
        )
    )
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr("stanstock.data.providers.twelve_data.resolve_api_key", blocked)
    store = AssetStore(tmp_path)
    execute_demo_product_refresh(store=store)
    viewer = django_user_model.objects.create_user(username="demo-study-viewer")
    run = AnalysisRun.objects.select_related("universe_snapshot").get(
        config_version=PRODUCT_VERSION
    )
    monkeypatch.setattr(timezone, "now", lambda: datetime(2026, 9, 13, 22, tzinfo=UTC))
    yield viewer, store, run
    blocked.assert_not_called()


@pytest.fixture
def live_product(settings, tmp_path, monkeypatch, django_user_model):
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = False
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", "a" * 40)
    monkeypatch.setattr(
        "stanstock.research.product_pipeline.clean_git_revision",
        lambda _path: "a" * 40,
    )
    owner, store, path, _resolve, _fetch = make_product_environment(
        tmp_path, monkeypatch, django_user_model
    )
    settings.DATA_DIR = store.root
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    job = execute_daily_research_job(
        target_date=TARGET,
        owner=owner,
        issued_on_time=False,
        store=store,
        core_config_path=path,
        enforce_rate_limit=False,
    )
    run = AnalysisRun.objects.select_related("universe_snapshot").get(
        pk=job.details["analysis_run_id"]
    )
    monkeypatch.setattr(timezone, "now", lambda: datetime(2026, 9, 13, 22, tzinfo=UTC))
    return owner, store, run


def _study_report(run: AnalysisRun, store: AssetStore, generated_at: datetime) -> dict[str, object]:
    return study_price_product_run(
        run=run,
        store=store,
        all_selected=True,
        report_generated_at=generated_at,
    )


def test_register_price_product_study_is_idempotent_without_other_db_mutations(
    demo_product, monkeypatch
) -> None:
    from stanstock.research import product_study_evidence as evidence

    _viewer, store, run = demo_product
    counts_before = {
        "runs": AnalysisRun.objects.count(),
        "analyses": StockAnalysis.objects.count(),
        "predictions": Prediction.objects.count(),
        "outcomes": PredictionOutcome.objects.count(),
        "assets": DataAsset.objects.count(),
    }

    monkeypatch.setenv("STANSTOCK_CODE_REVISION", "b" * 40)
    first = evidence.register_price_product_study(run=run, store=store)
    import json

    document = json.loads(store.read_bytes(first.relative_path))
    assert document["execution"]["code_revision"] == "b" * 40
    assert document["source_run"]["code_revision"] == run.code_revision
    assert run.code_revision != "b" * 40
    counts_after_first = {
        "runs": AnalysisRun.objects.count(),
        "analyses": StockAnalysis.objects.count(),
        "predictions": Prediction.objects.count(),
        "outcomes": PredictionOutcome.objects.count(),
        "assets": DataAsset.objects.count(),
    }
    monkeypatch.setattr(
        evidence,
        "study_price_product_run",
        Mock(side_effect=AssertionError("Registered evidence must recover without replay")),
    )
    second = evidence.register_price_product_study(run=run, store=store)
    with pytest.raises(TypeError):
        evidence.register_price_product_study(report={}, store=store)

    assert first.pk == second.pk
    assert counts_after_first == {
        **counts_before,
        "assets": counts_before["assets"] + 1,
    }
    assert counts_after_first == {
        "runs": AnalysisRun.objects.count(),
        "analyses": StockAnalysis.objects.count(),
        "predictions": Prediction.objects.count(),
        "outcomes": PredictionOutcome.objects.count(),
        "assets": DataAsset.objects.count(),
    }


def test_source_run_bound_studies_append_select_newest_and_recover_old_without_replay(
    live_product,
    django_user_model,
    monkeypatch,
) -> None:
    from stanstock.research import product_study_evidence as evidence

    owner, store, first_run = live_product
    second_run_at = datetime(2026, 9, 13, 20, tzinfo=UTC)
    monkeypatch.setattr(timezone, "now", lambda: second_run_at)
    second_job = execute_daily_research_job(
        target_date=TARGET,
        owner=owner,
        issuance_key="second-study-source-run",
        issued_on_time=False,
        store=store,
        enforce_rate_limit=False,
    )
    second_run = AnalysisRun.objects.select_related("universe_snapshot").get(
        pk=second_job.details["analysis_run_id"]
    )
    assert second_run.id != first_run.id

    first_registered_at = datetime(2026, 9, 13, 21, tzinfo=UTC)
    monkeypatch.setattr(timezone, "now", lambda: first_registered_at)
    first_asset = register_price_product_study(run=first_run, store=store)
    second_registered_at = datetime(2026, 9, 13, 22, tzinfo=UTC)
    monkeypatch.setattr(timezone, "now", lambda: second_registered_at)
    second_asset = register_price_product_study(run=second_run, store=store)

    assert first_asset.id != second_asset.id
    assert first_asset.subject != second_asset.subject
    assert first_asset.subject.endswith(str(first_run.id))
    assert second_asset.subject.endswith(str(second_run.id))
    assert DataAsset.objects.filter(kind=STUDY_EVIDENCE_KIND).count() == 2

    no_replay = Mock(side_effect=AssertionError("Old source-run retry must not recompute"))
    monkeypatch.setattr(evidence, "study_price_product_run", no_replay)
    old_retry = register_price_product_study(run=first_run, store=store)
    assert old_retry.id == first_asset.id
    no_replay.assert_not_called()

    result = read_registered_price_product_study(user=owner, store=store)
    assert result.status == "available"
    assert result.source_run_id == second_run.id
    assert result.source_generated_at == second_run.generated_at

    other = django_user_model.objects.create_user(username="other-study-history-owner")
    assert read_registered_price_product_study(user=other, store=store).status == "absent"

    store.resolve(second_asset.relative_path).write_bytes(b"corrupt newest study")
    corrupted = read_registered_price_product_study(user=owner, store=store)
    assert corrupted.status == "integrity_failed"
    assert corrupted.source_run_id is None


def test_register_price_product_study_requires_all_selected_and_exact_source_identity(
    demo_product,
) -> None:
    _viewer, store, run = demo_product
    listing_id = (
        StockAnalysis.objects.filter(run=run)
        .order_by("listing_id")
        .values_list("listing_id", flat=True)[0]
    )

    partial = study_price_product_run(
        run=run,
        store=store,
        listing_ids=(listing_id,),
        report_generated_at=datetime(2026, 9, 13, 18, tzinfo=UTC),
    )
    wrong_flag = serialize_price_product_study(
        _study_report(run, store, datetime(2026, 9, 13, 19, tzinfo=UTC))
    )
    wrong_flag["source_run"]["issued_on_time"] = True
    wrong_grade = serialize_price_product_study(
        _study_report(run, store, datetime(2026, 9, 13, 20, tzinfo=UTC))
    )
    wrong_grade["source_run"]["snapshot_grade"] = "observed"

    with pytest.raises(ValueError, match="all-selected source cohort"):
        _register_price_product_study(report=partial, store=store)
    with pytest.raises(ValueError, match="source run identity"):
        _register_price_product_study(report=wrong_flag, store=store)
    with pytest.raises(ValueError, match="source run identity"):
        _register_price_product_study(report=wrong_grade, store=store)


def test_registered_study_reader_fails_closed_on_corrupt_bytes(demo_product) -> None:
    viewer, store, run = demo_product
    asset = _register_price_product_study(
        report=_study_report(run, store, datetime(2026, 9, 13, 18, tzinfo=UTC)),
        store=store,
    )
    store.resolve(asset.relative_path).write_bytes(b"corrupt retrospective payload")

    result = read_registered_price_product_study(user=viewer, store=store)

    assert result.status == "integrity_failed"
    assert result.available is False
    assert result.verification_code


def test_registered_study_reader_fails_closed_on_forged_metadata_row(demo_product) -> None:
    viewer, store, run = demo_product
    asset = _register_price_product_study(
        report=_study_report(run, store, datetime(2026, 9, 13, 18, tzinfo=UTC)),
        store=store,
    )
    payload = store.read_bytes(asset.relative_path)
    stored = store.write_bytes(
        "research/studies/forged/duplicate-study.json",
        payload,
    )
    register_asset(
        provider="stanstock",
        kind=STUDY_EVIDENCE_KIND,
        subject=asset.subject,
        stored=stored,
        retrieved_at=asset.retrieved_at + timedelta(minutes=1),
        available_at=asset.available_at + timedelta(minutes=1),
        metadata={**asset.metadata, "owner_id": "forged-owner"},
    )

    result = read_registered_price_product_study(user=viewer, store=store)

    assert result.status == "integrity_failed"
    assert result.verification_code == "product_study_registry_ambiguous"


def test_registered_study_rendering_surfaces_real_worse_and_empty_scopes_in_demo_mode(
    demo_product,
    client,
) -> None:
    viewer, store, run = demo_product
    _register_price_product_study(
        report=_study_report(run, store, datetime(2026, 9, 13, 18, tzinfo=UTC)),
        store=store,
    )

    study = read_registered_price_product_study(user=viewer, store=store)
    worse_row = next(
        row
        for partition in study.partitions
        for scope in partition.scopes
        for row in scope.rows
        if row.assessment == "Candidate worse"
    )
    empty_scope = next(
        scope
        for partition in study.partitions
        for scope in partition.scopes
        if scope.insufficient_reason == "no_eligible_observations"
    )

    client.force_login(viewer)
    response = client.get(reverse("performance"))

    assert response.status_code == 200
    assert response.context["registered_study"].status == "available"
    content = response.content.decode()
    assert worse_row.metric_label in content
    assert worse_row.baseline_label in content
    assert "Candidate worse" in content
    assert empty_scope.horizon_label in content
    assert "No candidate actual observations yet." in content
    assert "No paired actual observations yet" in content


def test_live_registered_study_is_owner_bound_and_renders_without_get_time_replay(
    live_product,
    client,
    django_user_model,
    monkeypatch,
) -> None:
    owner, store, run = live_product
    assert run.universe_snapshot.grade == "research"
    assert run.issued_on_time is False
    report = _study_report(run, store, datetime(2026, 9, 13, 21, tzinfo=UTC))
    _register_price_product_study(report=report, store=store)

    other = django_user_model.objects.create_user(username="other-study-viewer")
    assert read_registered_price_product_study(user=other, store=store).status == "absent"

    study = read_registered_price_product_study(user=owner, store=store)
    empty_scope = next(
        scope
        for partition in study.partitions
        for scope in partition.scopes
        if scope.insufficient_reason == "no_eligible_observations"
    )
    no_paths = Mock(side_effect=AssertionError("GET must not run Monte Carlo paths"))
    no_calculation = Mock(side_effect=AssertionError("GET must not calculate a new study"))
    monkeypatch.setattr("stanstock.research.price_product.simulate_fhs_terminal_logs", no_paths)
    monkeypatch.setattr(
        "stanstock.research.product_pipeline.calculate_price_product", no_calculation
    )
    from stanstock.research.product_pipeline import (
        verify_price_product_output as real_product_verifier,
    )

    verifier = Mock(wraps=real_product_verifier)
    monkeypatch.setattr(
        "stanstock.research.product_reader.verify_price_product_output",
        verifier,
    )

    client.force_login(owner)
    response = client.get(reverse("performance"))

    assert response.status_code == 200
    assert verifier.call_count == 1
    assert response.context["registered_study"].status == "available"
    assert response.context["registered_study"].source_target_date == TARGET
    content = response.content.decode()
    assert "Registered retrospective evidence" in content
    assert "Current-universe/current-vintage historical comparisons" in content
    assert "source cohort dated" in content
    assert "Zero-log-drift Gaussian baseline" in content
    assert "Historical-log-drift Gaussian baseline" in content
    assert empty_scope.horizon_label in content
    assert "No candidate actual observations yet." in content
    assert "No paired actual observations yet" in content
    assert "The frozen protocol forbids tuning on the 2025+ final holdout." in content
    assert "survivorship and current-vintage bias remain visible limits" in content
    assert "recorded realized" in content
    assert "Width and inclusion are descriptive" in content
    no_paths.assert_not_called()
    no_calculation.assert_not_called()

    from stanstock.research.product_reader import read_research_product

    record = ProviderRecord.objects.get(provider="twelve_data")
    record.usage_scope = BASIC_USAGE_SCOPE
    record.metadata = {
        **record.metadata,
        "plan": "basic",
        "personal_noncommercial_confirmed": True,
        "licensed_user_id": str(other.pk),
    }
    record.save()
    owner.is_active = False
    owner.save(update_fields=["is_active"])
    assert read_research_product(user=owner, store=store).status == "unauthorized"
    assert read_registered_price_product_study(user=owner, store=store).status == "unauthorized"


def test_study_chronology_and_actual_registration_availability(
    demo_product, monkeypatch, settings
) -> None:
    viewer, store, run = demo_product
    report = serialize_price_product_study(
        _study_report(run, store, datetime(2026, 9, 13, 18, tzinfo=UTC))
    )
    for invalid_time in (
        run.generated_at - timedelta(seconds=1),
        timezone.now() + timedelta(seconds=1),
    ):
        with pytest.raises(ValueError, match="generation must follow"):
            _register_price_product_study(
                report={**report, "report_generated_at": invalid_time.isoformat()}, store=store
            )
    assert not DataAsset.objects.filter(kind=STUDY_EVIDENCE_KIND).exists()

    asset = _register_price_product_study(report=report, store=store)
    assert asset.available_at == asset.retrieved_at == timezone.now()
    assert asset.available_at > datetime(2026, 9, 13, 18, tzinfo=UTC)
    monkeypatch.setattr(timezone, "now", lambda: datetime(2026, 9, 13, 21, tzinfo=UTC))
    assert (
        read_registered_price_product_study(user=viewer, store=store).status == "integrity_failed"
    )
    settings.RESEARCH_PRODUCT_ENABLED = False
    assert read_registered_price_product_study(user=viewer, store=store).status == "disabled"


@pytest.mark.parametrize(
    ("metric", "delta", "label"),
    [
        ("interval_width", "-0.1", "Narrower interval"),
        ("interval_width", "0.1", "Wider interval"),
        ("interval_inclusion", "0.1", "Higher inclusion"),
        ("interval_inclusion", "-0.1", "Lower inclusion"),
        ("interval_score", "-0.1", "Candidate better"),
        ("median_absolute_error", "0.1", "Candidate worse"),
    ],
)
def test_width_and_inclusion_are_not_standalone_skill_verdicts(metric, delta, label):
    from stanstock.research.product_study_evidence import _assessment

    assert _assessment(metric_name=metric, delta=Decimal(delta)) == label

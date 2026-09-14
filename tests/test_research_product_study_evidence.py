from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from html.parser import HTMLParser
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
    ProductStudyComparisonRow,
    ProductStudyScopeView,
    _register_price_product_study,
    read_registered_price_product_study,
    register_price_product_study,
)
from stanstock.web.product_views import _comparison_scope_summaries
from test_research_product_jobs import (
    TARGET,
    _persist_price_series,
    _series,
    make_product_environment,
)

pytestmark = pytest.mark.django_db


class _ClosedDetailsTextParser(HTMLParser):
    """Collect text that is, or is not, nested in a closed native disclosure."""

    def __init__(self) -> None:
        super().__init__()
        self._details_open: list[bool] = []
        self.outside_closed_details: list[str] = []
        self.inside_closed_details: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag == "details":
            self._details_open.append(any(name == "open" for name, _value in attrs))

    def handle_endtag(self, tag: str) -> None:
        if tag == "details":
            self._details_open.pop()

    def handle_data(self, data: str) -> None:
        if not data.strip():
            return
        destination = (
            self.inside_closed_details
            if any(not is_open for is_open in self._details_open)
            else self.outside_closed_details
        )
        destination.append(data)


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
    monkeypatch.setattr(
        "httpx.Client.send",
        Mock(side_effect=AssertionError("Product-study evidence tests must not use the network")),
    )
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


def _convergence_aggregates(report: dict[str, object]) -> list[dict[str, object]]:
    aggregates = report.get("convergence_aggregates")
    assert isinstance(aggregates, list)
    for aggregate in aggregates:
        assert isinstance(aggregate, dict)
    return aggregates


def _set_convergence_counts(
    report: dict[str, object],
    *,
    observation_count: int,
    exceeded_count: int,
    unavailable_count: int,
) -> None:
    aggregates = _convergence_aggregates(report)
    for aggregate in aggregates:
        aggregate.update(
            observation_count=0,
            exceeded_count=0,
            unavailable_count=0,
            unavailable_reasons={},
        )
    aggregates[0].update(
        observation_count=observation_count,
        exceeded_count=exceeded_count,
        unavailable_count=unavailable_count,
        unavailable_reasons=(
            {} if unavailable_count == 0 else {"historical_anchor_data_missing": unavailable_count}
        ),
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


@pytest.mark.parametrize("different_subject", (False, True))
def test_registered_study_reader_fails_closed_on_forged_metadata_row(
    demo_product, different_subject
) -> None:
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
        subject=(
            asset.subject.rsplit(":", 1)[0] + ":00000000-0000-0000-0000-000000000001"
            if different_subject
            else asset.subject
        ),
        stored=stored,
        retrieved_at=asset.retrieved_at + timedelta(minutes=1),
        available_at=asset.available_at + timedelta(minutes=1),
        metadata={**asset.metadata, "owner_id": "forged-owner"},
    )

    result = read_registered_price_product_study(user=viewer, store=store)

    assert result.status == "integrity_failed"
    assert result.verification_code == "product_study_registry_ambiguous"


def test_registered_study_reader_surfaces_stored_convergence_counts(
    live_product,
    client,
    monkeypatch,
) -> None:
    viewer, store, run = live_product
    report = serialize_price_product_study(
        _study_report(run, store, datetime(2026, 9, 13, 18, tzinfo=UTC))
    )
    _set_convergence_counts(
        report,
        observation_count=5,
        exceeded_count=3,
        unavailable_count=2,
    )
    _register_price_product_study(report=report, store=store)

    study = read_registered_price_product_study(user=viewer, store=store)

    assert study.status == "available"
    assert study.convergence_summary is not None
    assert study.convergence_summary.production_paths == 8192
    assert study.convergence_summary.diagnostic_paths == 16384
    assert study.convergence_summary.observation_count == 5
    assert study.convergence_summary.available_quantile_comparison_count == 3
    assert study.convergence_summary.exceeded_count == 3
    assert study.convergence_summary.unavailable_count == 2
    assert study.convergence_summary.has_available_quantile_comparisons is True

    no_paths = Mock(side_effect=AssertionError("GET must not run Monte Carlo paths"))
    no_calculation = Mock(side_effect=AssertionError("GET must not calculate a new study"))
    monkeypatch.setattr("stanstock.research.price_product.simulate_fhs_terminal_logs", no_paths)
    monkeypatch.setattr(
        "stanstock.research.product_pipeline.calculate_price_product",
        no_calculation,
    )
    client.force_login(viewer)
    response = client.get(reverse("performance"))

    assert response.status_code == 200
    assert response.context["registered_study"].convergence_summary == study.convergence_summary
    content = " ".join(response.content.decode().split())
    assert "Historical anchor diagnostic" in content
    assert "<strong>Available:</strong> 3" in content
    assert "<strong>Exceeded:</strong> 3" in content
    assert "<strong>Unavailable:</strong> 2" in content
    assert (
        f"{study.convergence_summary.production_paths} production paths with "
        f"{study.convergence_summary.diagnostic_paths} diagnostic paths"
    ) in content
    assert (
        "Numerical precision checks are not market outcomes, calibration, or a skill claim."
        in content
    )
    assert "Exceedances remain disclosed and do not alter issuance." in content
    no_paths.assert_not_called()
    no_calculation.assert_not_called()


def test_registered_study_reader_marks_zero_available_convergence_as_not_a_pass(
    live_product,
    client,
) -> None:
    viewer, store, run = live_product
    report = serialize_price_product_study(
        _study_report(run, store, datetime(2026, 9, 13, 18, tzinfo=UTC))
    )
    _set_convergence_counts(
        report,
        observation_count=4,
        exceeded_count=0,
        unavailable_count=4,
    )
    _register_price_product_study(report=report, store=store)

    study = read_registered_price_product_study(user=viewer, store=store)

    assert study.status == "available"
    assert study.convergence_summary is not None
    assert study.convergence_summary.available_quantile_comparison_count == 0
    assert study.convergence_summary.exceeded_count == 0
    assert study.convergence_summary.unavailable_count == 4
    assert study.convergence_summary.has_available_quantile_comparisons is False

    client.force_login(viewer)
    response = client.get(reverse("performance"))

    assert response.status_code == 200
    content = " ".join(response.content.decode().split())
    assert "<strong>Available:</strong> 0" in content
    assert "<strong>Exceeded:</strong> 0" in content
    assert "<strong>Unavailable:</strong> 4" in content
    assert (
        "<strong>Comparison unavailable.</strong> Zero available comparisons is not a pass."
        in content
    )
    assert (
        "Numerical precision checks are not market outcomes, calibration, or a skill claim."
        in content
    )


@pytest.mark.parametrize(
    "invalid_kind",
    (
        "absent",
        "malformed",
        "duplicate",
        "boolean_count",
        "negative_count",
        "unavailable_exceeds_observation",
        "exceeded_exceeds_available",
    ),
)
def test_registered_study_reader_fails_closed_on_invalid_convergence_aggregates(
    live_product,
    invalid_kind,
) -> None:
    viewer, store, run = live_product
    report = serialize_price_product_study(
        _study_report(run, store, datetime(2026, 9, 13, 18, tzinfo=UTC))
    )
    if invalid_kind == "absent":
        report.pop("convergence_aggregates")
    elif invalid_kind == "malformed":
        report["convergence_aggregates"] = "not-a-list"
    else:
        aggregates = _convergence_aggregates(report)
        aggregate = aggregates[0]
        if invalid_kind == "duplicate":
            aggregates.append(aggregate.copy())
        elif invalid_kind == "boolean_count":
            aggregate["observation_count"] = True
        elif invalid_kind == "negative_count":
            aggregate["observation_count"] = -1
        elif invalid_kind == "unavailable_exceeds_observation":
            aggregate.update(
                observation_count=1,
                unavailable_count=2,
                exceeded_count=0,
                unavailable_reasons={"historical_anchor_data_missing": 2},
            )
        elif invalid_kind == "exceeded_exceeds_available":
            aggregate.update(
                observation_count=3,
                unavailable_count=1,
                exceeded_count=3,
                unavailable_reasons={"historical_anchor_data_missing": 1},
            )
        else:
            raise AssertionError(f"Unexpected invalid convergence aggregate case: {invalid_kind}")
    _register_price_product_study(report=report, store=store)

    result = read_registered_price_product_study(user=viewer, store=store)

    assert result.status == "integrity_failed"
    assert result.available is False
    assert result.convergence_summary is None
    assert result.verification_code == "product_study_evidence_invalid"


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
    assert "Comparison unavailable." in content
    assert "No Eligible Observations" in content
    expected_detail_rows = sum(
        len(scope.rows)
        for partition in study.partitions
        for scope in partition.scopes
    )
    assert expected_detail_rows == 168
    comparison_partitions = response.context["comparison_partitions"]
    scope_entries = [
        scope_entry
        for partition_entry in comparison_partitions
        for scope_entry in partition_entry["scopes"]
    ]
    assert all(len(scope_entry["baseline_summaries"]) <= 2 for scope_entry in scope_entries)
    assert response.content.count(b'class="comparison-detail-row"') == expected_detail_rows
    assert response.content.count(b'class="comparison-baseline-summary"') == sum(
        len(scope_entry["baseline_summaries"]) for scope_entry in scope_entries
    )
    expected_adverse_summaries = sum(
        summary["has_candidate_worse"]
        for scope_entry in scope_entries
        for summary in scope_entry["baseline_summaries"]
    )
    assert (
        response.content.count(b'class="field-error comparison-adverse"')
        == expected_adverse_summaries
    )
    assert b"<details open>" not in response.content
    assert response.content.index(b'id="numerical-sensitivity"') < response.content.index(
        b'id="historical-comparison"'
    )


def test_comparison_summary_retains_worse_state_when_median_error_is_better() -> None:
    """One median-error context row cannot suppress another metric's adverse state."""

    median_error = ProductStudyComparisonRow(
        baseline_model="synthetic_baseline",
        baseline_label="Synthetic baseline",
        metric_name="median_absolute_error",
        metric_label="Mean absolute error of the median forecast",
        paired_observation_count=7,
        paired_target_cohort_count=3,
        candidate_average=Decimal("0.10"),
        baseline_average=Decimal("0.20"),
        mean_difference_candidate_minus_baseline=Decimal("-0.10"),
        assessment="Candidate better",
        unavailable_reason=None,
    )
    worse_metric = ProductStudyComparisonRow(
        baseline_model="synthetic_baseline",
        baseline_label="Synthetic baseline",
        metric_name="interval_width",
        metric_label="Interval width",
        paired_observation_count=7,
        paired_target_cohort_count=3,
        candidate_average=Decimal("0.40"),
        baseline_average=Decimal("0.20"),
        mean_difference_candidate_minus_baseline=Decimal("0.20"),
        assessment="Candidate worse",
        unavailable_reason=None,
    )
    unavailable_metric = ProductStudyComparisonRow(
        baseline_model="synthetic_baseline",
        baseline_label="Synthetic baseline",
        metric_name="interval_inclusion",
        metric_label="Interval inclusion",
        paired_observation_count=0,
        paired_target_cohort_count=0,
        candidate_average=None,
        baseline_average=None,
        mean_difference_candidate_minus_baseline=None,
        assessment="Unavailable",
        unavailable_reason="historical_anchor_data_missing",
    )
    scope = ProductStudyScopeView(
        partition="synthetic_partition",
        partition_label="Synthetic partition",
        horizon="6m",
        horizon_label="6 months",
        target_cohort_count=3,
        distinct_listing_count=3,
        unavailable_count=1,
        insufficient_reason=None,
        rows=(median_error, worse_metric, unavailable_metric),
    )

    summaries = _comparison_scope_summaries(scope)

    assert len(summaries) == 1
    assert summaries[0]["context_row"] == median_error
    assert summaries[0]["assessments"] == (
        "Candidate better",
        "Candidate worse",
        "Unavailable",
    )
    assert summaries[0]["unavailable_reasons"] == ("historical_anchor_data_missing",)
    assert summaries[0]["has_candidate_worse"] is True


def test_registered_study_qualifications_are_not_hidden_in_a_closed_disclosure(
    demo_product,
    client,
) -> None:
    """Synthetic demo qualifications must remain visible beside comparisons."""

    viewer, store, run = demo_product
    _register_price_product_study(
        report=_study_report(run, store, datetime(2026, 9, 13, 18, tzinfo=UTC)),
        store=store,
    )
    study = read_registered_price_product_study(user=viewer, store=store)
    assert study.available
    assert study.disclosures

    client.force_login(viewer)
    response = client.get(reverse("performance"))

    assert response.status_code == 200
    parser = _ClosedDetailsTextParser()
    parser.feed(response.content.decode())
    visible_text = " ".join(parser.outside_closed_details)
    closed_details_text = " ".join(parser.inside_closed_details)
    assert 'class="compact-gate-list study-qualifications"' in response.content.decode()
    for disclosure in study.disclosures:
        assert disclosure in visible_text
        assert disclosure not in closed_details_text


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
    assert "Historical reconstruction comparison" in content
    assert "Current-universe/current-vintage evidence" in content
    assert "Studied cohort:" in content
    assert "Source run" in content
    assert "Better/worse refers to paired error" in content
    for partition in study.partitions:
        for scope in partition.scopes:
            for row in scope.rows:
                assert row.baseline_label in content
    assert empty_scope.horizon_label in content
    assert "Comparison unavailable." in content
    assert "No Eligible Observations" in content
    assert "The frozen protocol forbids tuning on the 2025+ final holdout." in content
    assert "survivorship" in content
    assert "current-vintage limits" in content
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

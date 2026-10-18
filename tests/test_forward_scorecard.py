from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django.utils import timezone
from exchange_calendars import get_calendar

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.models import DataAsset, ProviderRecord
from stanstock.data.provider_policy import ProviderConfigurationError
from stanstock.research import forward_scorecard as scorecard
from stanstock.research.models import Prediction
from stanstock.research.outcomes import evaluate_prediction
from stanstock.research.price_product_config import FHS_METHOD_VERSION, MOMENTUM_METHOD_VERSION
from stanstock.research.price_product_replay import _non_overlapping_target_count
from test_research_product_jobs import NOW
from test_research_product_reader_web import live_product as live_product
from test_research_product_reader_web import observed_product_history as observed_product_history

pytestmark = pytest.mark.django_db


def test_fixed_protocol_pin_and_declaration_do_not_read_losses(live_product, monkeypatch):
    owner, store, _run = live_product
    blocked = Mock(side_effect=AssertionError("declaration must precede every outcome read"))
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", "c" * 40)
    with monkeypatch.context() as patch:
        patch.setattr(scorecard, "iter_product_prediction_projection", blocked)
        declaration = scorecard.declare_forward_protocol(user=owner, store=store)
        assert scorecard.declare_forward_protocol(user=owner, store=store) == declaration
    assert hashlib.sha256(scorecard._bytes(scorecard.FORWARD_PROTOCOL)).hexdigest() == (
        "6fc2608de73d2de46c02cb2ee1308741e8882a0f5cb01f56cf9e230afa323524"
    )
    blocked.assert_not_called()
    assert declaration.document["execution_revision"] not in {"working-tree", "c" * 40}
    assert DataAsset.objects.filter(kind=scorecard.FORWARD_PROTOCOL_KIND).count() == 1
    payload = store.read_bytes(declaration.asset.relative_path)
    monkeypatch.setattr(timezone, "now", lambda: NOW + timedelta(minutes=8))
    retry = scorecard.declare_forward_protocol(user=owner, store=store)
    assert retry.asset.id == declaration.asset.id
    assert store.read_bytes(retry.asset.relative_path) == payload


def test_unmatured_native_scope_through_actual_command(live_product, monkeypatch):
    owner, store, run = live_product
    # Existing rows must not acquire a pre-issuance claim merely because
    # declaration and issuance have equal recorded clock precision.
    monkeypatch.setattr(timezone, "now", lambda: run.generated_at)
    with pytest.raises(ValueError, match="durable"):
        scorecard.read_forward_scorecard(user=owner, store=store)
    output = StringIO()
    call_command("forecast_scorecard", owner=owner.username, declare=True, stdout=output)
    report = json.loads(output.getvalue().splitlines()[-1])
    assert report["conclusion"] == "unmatured; skill not established"
    assert report["groups"]
    for group in report["groups"]:
        assert group["denominators"]["matured"] == 0
        assert "protocol declared before issuance" not in group["protocol_timing"]
        assert (
            group["protocol_timing"].get("protocol declared after issuance, before maturity", 0)
            == group["denominators"]["canonical_rows"]
        )
        if group["role"] == "advisory":
            assert set(group["metrics"].values()) == {None}
        assert group["hold_success"] is None
    assert sum(report["admission_denominators"].values()) > 0


def test_persisted_matured_native_outcomes_are_replayed_before_command_output(
    observed_product_history,
    monkeypatch,
):
    owner, store, original, reissue, current = observed_product_history
    evaluation_time = timezone.now()
    # This synthetic chronology declares after issuance but before native
    # maturity. It must never be described as pre-issuance preregistration.
    monkeypatch.setattr(timezone, "now", lambda: NOW + timedelta(minutes=6))
    declaration = scorecard.declare_forward_protocol(user=owner, store=store)
    monkeypatch.setattr(timezone, "now", lambda: evaluation_time)
    for run in (original, reissue):
        prediction = Prediction.objects.get(
            analysis__run=run,
            listing__provider_symbol="AAPL",
            method_version=FHS_METHOD_VERSION,
            horizon="6m",
        )
        result = evaluate_prediction(
            prediction,
            provider="twelve_data",
            evaluation_date=current.target_date,
            evaluation_time=evaluation_time,
            benchmark_subject="SPY",
            store=store,
        )
        assert result.outcome.status == "matured"
    real_verify = scorecard.verify_prediction_outcome
    calls = []

    def verify(*args, **kwargs):
        assert DataAsset.objects.filter(pk=declaration.asset.id).exists()
        calls.append(args[0].id)
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(scorecard, "verify_prediction_outcome", verify)
    output = StringIO()
    call_command("forecast_scorecard", owner=owner.username, stdout=output)
    report = json.loads(output.getvalue())
    advisory = next(
        group
        for group in report["groups"]
        if (group["role"] == "advisory" and group["horizon"] == "6m")
    )
    assert advisory["denominators"]["matured"] == 1
    assert advisory["denominators"]["excluded_reissue"] >= 1
    assert advisory["eligible_rows"] == 1
    assert advisory["protocol_timing"]["protocol declared after issuance, before maturity"] >= 1
    assert advisory["overlap_aware_support"] == 1
    assert set(advisory["metrics"].values()) == {None}  # existing floors, not relaxed
    assert len(calls) == 2  # canonical decision + advisory; never matured reissues
    prediction = Prediction.objects.get(pk=calls[-1])
    prediction.outcome.metadata["provider"] = "wrong"
    prediction.outcome.save(update_fields=["metadata"])
    with pytest.raises(CommandError, match="withheld"):
        call_command("forecast_scorecard", owner=owner.username, stdout=StringIO())


def test_declaration_corruption_or_revoked_rights_fail_instead_of_redeclaring(
    live_product,
):
    owner, store, _run = live_product
    declaration = scorecard.declare_forward_protocol(user=owner, store=store)
    record = ProviderRecord.objects.get(provider="twelve_data")
    record.metadata["internal_display_rights_confirmed"] = False
    record.save(update_fields=["metadata"])
    with pytest.raises(ProviderConfigurationError):
        scorecard.declare_forward_protocol(user=owner, store=store)
    record.metadata["internal_display_rights_confirmed"] = True
    record.save(update_fields=["metadata"])
    store.resolve(declaration.asset.relative_path).write_bytes(b"corrupt")
    with pytest.raises(RefreshVerificationError):
        scorecard.declare_forward_protocol(user=owner, store=store)
    assert DataAsset.objects.filter(kind=scorecard.FORWARD_PROTOCOL_KIND).count() == 1


def _adapter_rows():
    rows = []
    for cohort in range(8):
        anchor = (
            get_calendar("XNYS")
            .date_to_session(date(2020, 1, 2) + timedelta(days=200 * cohort), direction="next")
            .date()
        )
        for listing in range(30):
            rows.append(
                {
                    "id": len(rows),
                    "listing_id": listing,
                    "target_date": anchor,
                    "generated_at": datetime.combine(anchor, datetime.min.time(), UTC),
                    "reportable": True,
                    "canonical": True,
                    "outcome__status": "matured",
                    "outcome__evaluation_date": anchor + timedelta(days=180),
                    "bear_return": Decimal("-.1"),
                    "base_return": Decimal(".1"),
                    "bull_return": Decimal(".3"),
                    "recommendation": "buy",
                    "code_revision": "a" * 40,
                    "calculation__forecast__projections": [
                        {
                            "horizon": "6m",
                            "zero_drift_ledger_returns": {
                                "lower": "-.2",
                                "median": "0",
                                "upper": ".2",
                            },
                        }
                    ],
                }
            )
    return rows


def _stub_adapter_outcomes(rows, monkeypatch):
    outcomes = {
        row["id"]: SimpleNamespace(
            outcome=SimpleNamespace(
                evaluated_at=datetime(2025, 1, 1, tzinfo=UTC),
                evaluation_date=row["outcome__evaluation_date"],
                actual_return=Decimal(".15"),
                success=True,
            )
        )
        for row in rows
    }
    monkeypatch.setattr(timezone, "now", lambda: datetime(2026, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(
        scorecard.Prediction.objects,
        "select_related",
        lambda *_args: SimpleNamespace(get=lambda **kw: outcomes[kw["pk"]]),
    )
    monkeypatch.setattr(scorecard, "verify_prediction_outcome", lambda *_a, **_kw: None)
    return outcomes


def test_boundary_adapter_native_and_paired_support_are_independent(monkeypatch):
    """Loss math remains in incumbent helpers; only this boundary is synthetic."""
    rows = _adapter_rows()
    outcomes = _stub_adapter_outcomes(rows, monkeypatch)
    declaration = SimpleNamespace(declared_at=datetime(2019, 1, 1, tzinfo=UTC))
    key = ("us-price-fhs-v1", "config", "twelve_data", "observed", "advisory", "6m")
    group = scorecard._score_group(key, rows, declaration=declaration)
    assert group["support_reason"] is None
    assert group["metrics"]["median_absolute_error"] == Decimal(".05")
    assert group["metrics"]["signed_bias"] == Decimal(".05")
    assert group["overlap_aware_support"] == 8
    assert all(item["paired_observation_count"] == 240 for item in group["paired_zero_drift"])
    for row in rows[30:]:
        row["calculation__forecast__projections"] = []
    group = scorecard._score_group(key, rows, declaration=declaration)
    assert group["metrics"]["median_absolute_error"] == Decimal(".05")
    assert all(item["candidate_average"] is None for item in group["paired_zero_drift"])
    assert all(item["overlap_aware_support"] == 1 for item in group["paired_zero_drift"])
    decision_key = ("us-price-momentum-v1", *key[1:4], "decision", "6m")
    decision = scorecard._score_group(decision_key, rows, declaration=declaration)
    assert decision["decision_success"]["BUY"]["success_rate"] == 1
    assert decision["decision_success"]["AVOID"]["success_rate"] is None
    assert decision["hold_success"] is None
    # Unequal listing counts must not turn a large target cohort into more
    # independent market observations. Native support is still eight cohorts.
    for index in range(30):
        row = {
            **rows[index],
            "id": len(rows),
            "listing_id": 30 + index,
            "code_revision": "b" * 40,
        }
        outcomes[row["id"]] = SimpleNamespace(
            outcome=SimpleNamespace(
                evaluated_at=datetime(2025, 1, 1, tzinfo=UTC),
                evaluation_date=row["outcome__evaluation_date"],
                actual_return=Decimal(".95"),
                success=True,
            )
        )
        rows.append(row)
    weighted = scorecard._score_group(key, rows, declaration=declaration)
    assert weighted["metrics"]["median_absolute_error"] == Decimal(".1")
    assert weighted["metrics"]["signed_bias"] == Decimal(".1")
    assert weighted["overlap_aware_support"] == 8
    assert weighted["execution_revisions"] == ["a" * 40, "b" * 40]
    late = scorecard._score_group(
        key,
        rows,
        declaration=SimpleNamespace(
            declared_at=datetime(2025, 12, 31, tzinfo=UTC),
        ),
    )
    assert late["eligible_rows"] == late["overlap_aware_support"] == 0
    assert late["denominators"]["post_maturity_declaration_excluded"] == len(rows)
    assert set(late["metrics"].values()) == {None}


@pytest.mark.parametrize(
    ("role", "action"), (("advisory", "buy"), ("decision", "buy"), ("decision", "avoid"))
)
def test_support_floor_applies_to_each_selected_cohort(monkeypatch, role, action):
    rows = _adapter_rows()
    for row in rows:
        row["recommendation"] = action
    thin = [row for index, row in enumerate(rows) if index < 30 or row["listing_id"] == 0]
    for index, row in enumerate(thin[30:], start=30):
        row["listing_id"] = index
    _stub_adapter_outcomes(rows, monkeypatch)
    declaration = SimpleNamespace(declared_at=datetime(2019, 1, 1, tzinfo=UTC))
    method = FHS_METHOD_VERSION if role == "advisory" else MOMENTUM_METHOD_VERSION
    key = (method, "config", "twelve_data", "observed", role, "6m")
    group = scorecard._score_group(key, thin, declaration=declaration)
    assert group["eligible_target_cohorts"] == group["overlap_aware_support"] == 8
    assert group["support_reason"] == "insufficient_overlap_aware_support; skill not established"
    if role == "advisory":
        assert set(group["metrics"].values()) == {None}
        assert all(item["candidate_average"] is None for item in group["paired_zero_drift"])
    else:
        assert group["decision_success"][action.upper()]["success_rate"] is None
    control = scorecard._score_group(key, rows, declaration=declaration)
    assert control["support_reason"] is None
    if role == "advisory":
        assert control["metrics"]["median_absolute_error"] == Decimal(".05")
    else:
        assert control["decision_success"][action.upper()]["success_rate"] == 1


def test_paired_support_floor_does_not_pool_listings_between_cohorts(monkeypatch):
    rows = _adapter_rows()
    for row in rows[30:]:
        if row["listing_id"] != 0:
            row["calculation__forecast__projections"] = []
    _stub_adapter_outcomes(rows, monkeypatch)
    group = scorecard._score_group(
        (FHS_METHOD_VERSION, "config", "twelve_data", "observed", "advisory", "6m"),
        rows,
        declaration=SimpleNamespace(declared_at=datetime(2019, 1, 1, tzinfo=UTC)),
    )
    assert group["support_reason"] is None
    assert group["metrics"]["median_absolute_error"] == Decimal(".05")
    for comparison in group["paired_zero_drift"]:
        assert comparison["overlap_aware_support"] == 8
        assert comparison["candidate_average"] is comparison["baseline_average"] is None
        assert comparison["unavailable_reason"] == (
            "insufficient_overlap_aware_support; skill not established"
        )


def test_support_span_uses_selected_targets_not_maturity_ends():
    rows = [
        {
            "listing_id": listing,
            "target_date": date(2020, 1, 2) + timedelta(days=1094 * cohort),
            "outcome__evaluation_date": date(2020, 1, 2) + timedelta(days=1094 * cohort + 1093),
        }
        for cohort in range(3)
        for listing in range(30)
    ]
    # Three fully supported non-overlapping cohorts span only 2188 target days,
    # not the frozen three-year reporting floor of 2190.
    assert scorecard._support_result(rows, "3y") == (
        "insufficient_overlap_aware_support; skill not established",
        3,
    )


def test_support_count_uses_reporting_selection_at_shared_boundaries():
    calendar = get_calendar("XNYS")
    anchors = calendar.sessions_window(calendar.date_to_session(date(2020, 1, 2)), 883)
    metrics = scorecard.score_projection_metrics(
        predicted_returns=None, actual_return=None, config=scorecard.load_price_product_config()
    )
    observations = [
        scorecard.ReplayMetricObservation(
            listing_id="synthetic-support",
            anchor_date=anchor.date(),
            target_date=calendar.session_offset(anchor, 126).date(),
            horizon="6m",
            partition="final_holdout",
            model_name="fhs",
            metrics=metrics,
        )
        for anchor in anchors
    ]
    rows = [
        {
            "listing_id": listing,
            "target_date": observation.anchor_date,
            "outcome__evaluation_date": observation.target_date,
        }
        for observation in observations
        for listing in range(30)
    ]
    assert _non_overlapping_target_count(observations) == 8
    assert scorecard._support_result(rows, "6m") == (
        "insufficient_overlap_aware_support; skill not established",
        7,
    )


@pytest.mark.parametrize("status", (None, "unresolved", "matured"))
def test_null_advisory_is_only_non_evaluable(monkeypatch, django_assert_num_queries, status):
    row = {
        **_adapter_rows()[0],
        "bear_return": None,
        "base_return": None,
        "bull_return": None,
        "outcome__status": status,
    }
    blocked = Mock(side_effect=AssertionError("non-evaluable rows cannot read outcome evidence"))
    monkeypatch.setattr(scorecard, "verify_prediction_outcome", blocked)
    with django_assert_num_queries(0):
        group = scorecard._score_group(
            (FHS_METHOD_VERSION, "config", "twelve_data", "observed", "advisory", "6m"),
            [row],
            declaration=SimpleNamespace(declared_at=datetime(2019, 1, 1, tzinfo=UTC)),
        )
    assert group["denominators"]["raw_rows"] == group["denominators"]["non_evaluable"] == 1
    assert all(
        group["denominators"][name] == 0
        for name in ("canonical_rows", "pending", "matured", "unresolved", "corporate_event")
    )
    assert group["support_reason"] == "unmatured; skill not established"
    assert set(group["metrics"].values()) == {None}
    assert group["protocol_timing"] == {}
    blocked.assert_not_called()

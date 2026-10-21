from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest
from django.db import DatabaseError, transaction
from exchange_calendars import get_calendar

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.asof import AsOfData, raw_price_asset_for
from stanstock.data.assets import read_checksummed_bytes
from stanstock.data.live_us import _persist_price_series
from stanstock.data.models import DataAsset
from stanstock.research.decision_support_eligibility import default_decision_support_config_path
from stanstock.research.models import Prediction, PredictionOutcome
from stanstock.research.outcome_refresh_validation import verify_prediction_outcome
from stanstock.research.outcomes import evaluate_prediction
from stanstock.research.reporting import product_performance_groups
from stanstock.research.service import analyze_snapshot

pytestmark = pytest.mark.django_db


def _observed_maturity(env, monkeypatch, *, reissue=False):
    revision = "a" * 40
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", revision)
    monkeypatch.setattr(
        "stanstock.research.product_eligibility_pipeline.clean_git_revision", lambda _root: revision
    )
    kwargs = dict(
        universe_snapshot=env.snapshot,
        decision_time=env.now,
        target_date=env.target,
        issued_on_time=True,
        provider="twelve_data",
        benchmark_subject="SPY",
        config_path=default_decision_support_config_path(),
        store=env.store,
    )
    env.run = analyze_snapshot(**kwargs)[0].run
    if reissue:
        env.clock["now"] = env.now + timedelta(minutes=1)
        kwargs["decision_time"] = env.clock["now"]
        env.reissue = analyze_snapshot(**kwargs)[0].run
    calendar = get_calendar("XNYS")
    maturity = calendar.session_offset(calendar.date_to_session(env.target), 126)
    env.maturity = maturity.date()
    env.evaluated_at = calendar.session_close(maturity).to_pydatetime() + timedelta(hours=1)
    env.clock["now"] = env.evaluated_at
    for symbol in ("AAPL", "NVDA", "SPY"):
        _persist_price_series(
            store=env.store,
            series=env.series(symbol, through=env.maturity),
            listing=None,
            resolved_mic_code="ARCX" if symbol == "SPY" else None,
            catalog_assets=env.catalogs if symbol == "SPY" else None,
        )
    return env


def _evaluate(env, prediction):
    return evaluate_prediction(
        prediction,
        provider="twelve_data",
        evaluation_date=env.maturity,
        evaluation_time=env.evaluated_at,
        benchmark_subject="SPY",
        store=env.store,
    ).outcome


@pytest.mark.parametrize("c2_environment", [{"grade": "observed"}], indirect=True)
def test_v2_null_triplet_decision_matures_on_exact_126_split_sessions_and_spy(
    c2_snapshot, monkeypatch
):
    env = _observed_maturity(c2_snapshot, monkeypatch)
    row = Prediction.objects.get(
        analysis__run=env.run, price_subject="AAPL", evidence_role="decision"
    )
    assert row.bear_return is row.base_return is row.bull_return is None
    outcome = _evaluate(env, row)
    assert outcome.status == "matured" and outcome.evaluation_date == env.maturity
    assert outcome.actual_return > outcome.benchmark_return > 0 and outcome.success is True
    verify_prediction_outcome(
        row,
        outcome,
        provider="twelve_data",
        benchmark_subject="SPY",
        evaluation_time=env.evaluated_at,
        parent_target_date=env.maturity,
    )


@pytest.mark.parametrize(
    "c2_environment", [{"grade": "observed", "missing_volume": True}], indirect=True
)
def test_genuine_withheld_liquidity_hold_matures_but_success_stays_null_in_orm_and_bulk(
    c2_snapshot, monkeypatch
):
    env = _observed_maturity(c2_snapshot, monkeypatch)
    row = Prediction.objects.get(
        analysis__run=env.run, price_subject="AAPL", evidence_role="decision"
    )
    assert row.recommendation == "hold"
    outcome = _evaluate(env, row)
    assert outcome.status == "matured" and outcome.actual_return is not None
    assert outcome.success is None
    outcome.clean()
    with pytest.raises(DatabaseError, match="evidence role"), transaction.atomic():
        PredictionOutcome.objects.filter(pk=outcome.pk).update(success=True)


@pytest.mark.parametrize(
    "c2_environment", [{"grade": "observed", "down": True, "missing_volume": True}], indirect=True
)
def test_avoid_retains_negative_and_under_spy_maturity_success_without_liquidity(
    c2_snapshot, monkeypatch
):
    env = _observed_maturity(c2_snapshot, monkeypatch)
    row = Prediction.objects.get(
        analysis__run=env.run, price_subject="AAPL", evidence_role="decision"
    )
    assert row.recommendation == "avoid"
    outcome = _evaluate(env, row)
    assert outcome.status == "matured" and outcome.actual_return < 0
    assert outcome.actual_return < outcome.benchmark_return and outcome.success is True


@pytest.mark.parametrize("c2_environment", [{"grade": "observed"}], indirect=True)
def test_invalid_generic_none_maturity_is_integrity_failure_not_silent_fallback_or_unresolved(
    c2_snapshot, monkeypatch
):
    env = _observed_maturity(c2_snapshot, monkeypatch)
    env.clock["now"] += timedelta(seconds=1)
    poisoned = replace(env.series("AAPL", through=env.maturity), adjustment="none")
    _persist_price_series(store=env.store, series=poisoned, listing=None)
    row = Prediction.objects.get(
        analysis__run=env.run, price_subject="AAPL", evidence_role="decision"
    )
    with pytest.raises(RuntimeError, match="split-only"):
        evaluate_prediction(
            row,
            provider="twelve_data",
            evaluation_date=env.maturity,
            evaluation_time=env.clock["now"],
            benchmark_subject="SPY",
            store=env.store,
        )
    assert not PredictionOutcome.objects.filter(prediction=row).exists()


@pytest.mark.parametrize("c2_environment", [{"grade": "observed"}], indirect=True)
@pytest.mark.parametrize("boundary", ["producer", "verifier"])
def test_checksum_matching_malformed_raw_maturity_is_integrity_failure(
    c2_snapshot, monkeypatch, boundary
):
    env = _observed_maturity(c2_snapshot, monkeypatch)
    row = Prediction.objects.get(
        analysis__run=env.run, price_subject="AAPL", evidence_role="decision"
    )
    outcome = _evaluate(env, row) if boundary == "verifier" else None
    source = AsOfData(env.evaluated_at, env.store).latest_asset(
        provider="twelve_data", kind="price_history", subject="AAPL"
    )
    raw = raw_price_asset_for(source, cutoff=env.evaluated_at)
    payload = json.loads(read_checksummed_bytes(env.store, raw))
    payload["values"][0]["high"] = "0"
    env.evaluated_at += timedelta(seconds=1)
    env.clock["now"] = env.evaluated_at
    stored_raw = env.store.write_bytes(
        "synthetic/malformed-maturity-raw.json", json.dumps(payload).encode()
    )
    forged_raw = DataAsset.objects.create(
        provider=raw.provider,
        kind=raw.kind,
        subject=raw.subject,
        relative_path=stored_raw.relative_path,
        sha256=stored_raw.sha256,
        retrieved_at=env.evaluated_at,
        available_at=env.evaluated_at,
        period_start=raw.period_start,
        period_end=raw.period_end,
        metadata=raw.metadata,
    )
    stored_frame = env.store.write_bytes(
        "synthetic/malformed-maturity-normalized.parquet",
        read_checksummed_bytes(env.store, source),
    )
    DataAsset.objects.create(
        provider=source.provider,
        kind=source.kind,
        subject=source.subject,
        relative_path=stored_frame.relative_path,
        sha256=stored_frame.sha256,
        retrieved_at=env.evaluated_at,
        available_at=env.evaluated_at,
        period_start=source.period_start,
        period_end=source.period_end,
        metadata={
            **source.metadata,
            "raw_asset_id": str(forged_raw.id),
            "raw_sha256": forged_raw.sha256,
        },
    )
    if boundary == "producer":
        with pytest.raises(RuntimeError, match="split-only"):
            _evaluate(env, row)
        assert not PredictionOutcome.objects.filter(prediction=row).exists()
    else:
        with pytest.raises(RefreshVerificationError, match="split-only") as error:
            verify_prediction_outcome(
                row,
                outcome,
                provider="twelve_data",
                benchmark_subject="SPY",
                evaluation_time=env.evaluated_at,
                parent_target_date=env.maturity,
            )
        assert error.value.reason_code == "evaluation_outcome_maturity_unreadable"


@pytest.mark.parametrize("c2_environment", [{"grade": "observed"}], indirect=True)
def test_canonical_original_pending_cannot_be_replaced_by_matured_same_cohort_reissue(
    c2_snapshot, monkeypatch
):
    env = _observed_maturity(c2_snapshot, monkeypatch, reissue=True)
    reissue = Prediction.objects.get(
        analysis__run=env.reissue, price_subject="AAPL", evidence_role="decision"
    )
    assert _evaluate(env, reissue).status == "matured"
    groups, _advisory, scope = product_performance_groups([env.run.id, env.reissue.id])
    assert groups == []
    assert scope["decision"]["eligible_canonical"] == scope["decision"]["outcome_pending"] == 2
    original = Prediction.objects.get(
        analysis__run=env.run, price_subject="AAPL", evidence_role="decision"
    )
    _evaluate(env, original)
    groups, _advisory, scope = product_performance_groups([env.run.id, env.reissue.id])
    assert len(groups) == 1 and groups[0]["count"] == 1
    assert groups[0]["prediction__method_version"] == "us-relative-momentum-v2"
    assert groups[0]["prediction__config_hash"] == env.config_hash
    assert scope["decision"]["matured"] == 1 and scope["decision"]["outcome_pending"] == 1


def test_reporting_separates_shared_fhs_method_by_exact_config_provider_grade_role_horizon(
    monkeypatch,
):
    from stanstock.research import reporting
    from stanstock.research.decision_support_eligibility import (
        PRODUCT_EFFECTIVE_CONFIG_HASH as v2_hash,
    )
    from stanstock.research.price_product_config import PRODUCT_EFFECTIVE_CONFIG_HASH as v1_hash

    rows = [
        {
            "evidence_role": role,
            "method_version": method,
            "config_hash": config_hash,
            "price_provider": "twelve_data",
            "evidence_grade": "observed",
            "horizon": "6m",
            "reportable": True,
            "canonical": True,
            "outcome__status": "matured",
            "bear_return": Decimal("-0.1"),
            "base_return": Decimal("0.1"),
            "bull_return": Decimal("0.3"),
        }
        for role, method, config_hash in (
            ("decision", "us-relative-momentum-v1", v1_hash),
            ("decision", "us-relative-momentum-v2", v2_hash),
            ("advisory", "us-price-fhs-v1", v1_hash),
            ("advisory", "us-price-fhs-v1", v2_hash),
        )
    ]
    monkeypatch.setattr(reporting, "iter_product_prediction_projection", lambda _ids: iter(rows))
    decisions, forecasts, _scope = product_performance_groups([])
    assert len(decisions) == len(forecasts) == 2
    assert {row["prediction__config_hash"] for row in forecasts} == {v1_hash, v2_hash}
    assert all(row["count"] == 1 for row in decisions + forecasts)

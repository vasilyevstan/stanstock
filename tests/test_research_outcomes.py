from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal
from io import StringIO
from uuid import uuid4

import polars as pl
import pytest
from django.core.exceptions import ValidationError
from django.core.management import call_command
from django.db import IntegrityError, connection, transaction
from django.test import override_settings
from django.utils import timezone

import stanstock.research.outcomes as outcomes_module
from stanstock.data.asof import AsOfData
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import (
    Company,
    DataAsset,
    Listing,
    Region,
    Security,
    Universe,
    UniverseSnapshot,
)
from stanstock.research.jobs import eligible_pending_predictions
from stanstock.research.long_forecast_config import (
    LONG_V4_EFFECTIVE_CONFIG_HASH,
    LONG_V4_METHOD,
    LONG_V4_RESEARCH_STATUS,
    LONG_V4_VERSION,
)
from stanstock.research.long_forecasts_v4 import PROBABILITY_REASON, canonical_long_v4_price
from stanstock.research.models import (
    AnalysisRun,
    Prediction,
    PredictionOutcome,
    Recommendation,
    RiskClass,
    StockAnalysis,
)
from stanstock.research.outcomes import (
    HORIZON_SESSION_COUNTS,
    evaluate_prediction,
    evaluate_predictions,
    resolve_outcome,
)


@pytest.fixture(autouse=True)
def _isolate_outcome_tests_from_persisted_v4_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        outcomes_module,
        "_validate_long_v4_persisted_authority",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        outcomes_module,
        "_validate_long_v4_raw_sec_replay",
        lambda **_kwargs: None,
    )


def test_forecast_horizon_session_counts_keep_legacy_identity() -> None:
    assert HORIZON_SESSION_COUNTS == {
        Prediction.Horizon.SHORT.value: 10,
        Prediction.Horizon.SIX_MONTH.value: 126,
        Prediction.Horizon.TWELVE_MONTH.value: 252,
        Prediction.Horizon.THREE_YEAR.value: 756,
        Prediction.Horizon.FIVE_YEAR.value: 1260,
        Prediction.Horizon.MEDIUM.value: 252,
        Prediction.Horizon.LONG.value: 756,
    }


@pytest.mark.django_db
def test_short_prediction_matures_on_nth_observed_session_without_fabricating_weekends(
    tmp_path,
) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
        price_provider="synthetic",
    )
    target = prediction.target_date
    sessions = _business_dates_after(target, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store, listing.ticker, _evaluation_time(), sessions, [101 + i for i in range(10)]
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.MATURED
    assert result.outcome.evaluation_date == sessions[-1]
    assert sessions[-1] != target + timedelta(days=10)
    assert result.outcome.actual_return == Decimal("0.1")
    assert result.outcome.success is True
    assert result.outcome.metadata["horizon_sessions"] == 10


@pytest.mark.django_db
def test_insufficient_observed_sessions_create_unresolved_without_returns(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    sessions = _business_dates_after(prediction.target_date, 9)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store, listing.ticker, _evaluation_time(), sessions, [101 + i for i in range(9)]
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.UNRESOLVED
    assert "Insufficient observed sessions" in result.outcome.resolution
    assert result.outcome.actual_return is None
    assert result.outcome.benchmark_return is None
    assert result.outcome.success is None
    assert result.outcome.error is None


@pytest.mark.django_db
def test_unchanged_unresolved_outcome_is_not_rewritten(tmp_path) -> None:
    _, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
    )
    evaluation_date = prediction.target_date + timedelta(days=30)
    first_time = _evaluation_time() - timedelta(minutes=1)
    first = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=evaluation_date,
        evaluation_time=first_time,
        store=AssetStore(tmp_path),
    )

    second = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=evaluation_date,
        evaluation_time=_evaluation_time(),
        store=AssetStore(tmp_path),
    )

    second.outcome.refresh_from_db()
    assert first.action == "created"
    assert second.action == "skipped"
    assert second.outcome.evaluated_at == first_time


@pytest.mark.django_db
def test_recorded_price_provider_rejects_conflicting_override(tmp_path) -> None:
    _, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        price_provider="twelve_data",
    )

    with pytest.raises(ValueError, match="conflicts with requested provider"):
        evaluate_prediction(
            prediction,
            provider="synthetic",
            evaluation_date=prediction.target_date + timedelta(days=30),
            evaluation_time=_evaluation_time(),
            store=AssetStore(tmp_path),
        )

    assert not PredictionOutcome.objects.filter(prediction=prediction).exists()


@pytest.mark.django_db
def test_evaluator_uses_recorded_price_subject(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        price_provider="synthetic",
        price_subject=f"{listing.ticker}:US",
        recommendation=Recommendation.BUY,
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        prediction.price_subject,
        _evaluation_time(),
        sessions,
        [101 + index for index in range(10)],
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.MATURED
    assert result.outcome.metadata["subject"] == prediction.price_subject


@pytest.mark.django_db
def test_advisory_prediction_matures_without_decision_success(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        price_provider="synthetic",
        recommendation=Recommendation.BUY,
        bear=Decimal("-0.10"),
        base=Decimal("0.10"),
        bull=Decimal("0.30"),
    )
    sessions = _business_dates_after(prediction.target_date, 126)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        _evaluation_time(),
        sessions,
        [101 + index * (19 / 125) for index in range(126)],
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.MATURED
    assert result.outcome.actual_return == Decimal("0.2")
    assert result.outcome.success is None
    assert result.outcome.direction_correct is True
    assert result.outcome.interval_covered is True
    assert result.outcome.signed_error == Decimal("0.1")
    assert result.outcome.metadata["evidence_role"] == Prediction.EvidenceRole.ADVISORY


@pytest.mark.django_db
def test_withheld_advisory_forecast_stays_unresolved_without_price_lookup(tmp_path) -> None:
    _, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        price_provider="synthetic",
        recommendation=Recommendation.HOLD,
        bear=None,
        base=None,
        bull=None,
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=prediction.target_date + timedelta(days=200),
        evaluation_time=_evaluation_time(),
        store=AssetStore(tmp_path),
    )

    assert result.outcome.status == PredictionOutcome.Status.UNRESOLVED
    assert result.outcome.resolution == "Withheld forecast has no scenario to evaluate"
    assert result.outcome.actual_return is None
    assert result.outcome.benchmark_return is None
    assert result.outcome.success is None
    assert result.outcome.direction_correct is None
    assert result.outcome.interval_covered is None
    assert result.outcome.error is None
    assert result.outcome.signed_error is None


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        ("prediction_version", "identity_mismatch"),
        ("target_ticker", "target_identity_mismatch"),
        ("normalized_asset_id", "target_price_identity_mismatch"),
        ("valuation_source", "valuation_source_mismatch"),
        ("valuation_value", "valuation_value_invalid"),
        ("ledger_value", "ledger_value_mismatch"),
    ],
)
def test_long_v4_invalid_baseline_identity_is_exhaustive_and_precedes_all_price_io(
    mutation: str,
    expected_issue: str,
) -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(analysis)
    calculation = deepcopy(prediction.calculation)
    target_price = calculation["target_price"]

    if mutation == "prediction_version":
        calculation["prediction_version"] = "copied-from-another-prediction"
    elif mutation == "target_ticker":
        calculation["target"]["ticker"] = "COPIED"
    elif mutation == "normalized_asset_id":
        target_price["normalized_asset_id"] = str(uuid4())
        calculation["evidence_catalog"]["prices"] = [deepcopy(target_price)]
    elif mutation == "valuation_source":
        target_price["valuation_source"] = "ledger_price"
        calculation["evidence_catalog"]["prices"] = [deepcopy(target_price)]
    elif mutation == "valuation_value":
        target_price["valuation_value"] = "NaN"
        target_price["native_price"] = "NaN"
        calculation["evidence_catalog"]["prices"] = [deepcopy(target_price)]
    else:
        target_price["value"] = "100.000001"
        target_price["ledger_value"] = "100.000001"
        calculation["evidence_catalog"]["prices"] = [deepcopy(target_price)]
    prediction.calculation = calculation
    calls: list[tuple[str, date]] = []

    def forbidden_loader(subject: str, through_date: date) -> pl.DataFrame:
        calls.append((subject, through_date))
        raise AssertionError("invalid long-v4 identity must not load prices")

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date + timedelta(days=1200),
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject="BENCH",
        price_loader=forbidden_loader,
    )

    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.resolution == "Long-v4 valuation baseline could not be authenticated"
    assert resolved.metadata == {
        "provider": "twelve_data",
        "valuation_baseline": {
            "role": "calculation.target_price.valuation_value",
            "status": "authentication_failed",
            "issue_codes": [expected_issue],
        },
    }
    assert calls == []


@pytest.mark.django_db
def test_long_v4_invalid_baseline_issue_order_is_stable() -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(analysis)
    prediction.calculation = {}

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date,
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject=None,
        price_loader=lambda _subject, _through: (_ for _ in ()).throw(
            AssertionError("must not load")
        ),
    )

    assert resolved.metadata["valuation_baseline"]["issue_codes"] == [
        "identity_mismatch",
        "target_identity_mismatch",
        "target_price_identity_mismatch",
        "valuation_source_mismatch",
        "valuation_value_invalid",
        "ledger_value_mismatch",
    ]


@pytest.mark.django_db
def test_long_v4_cross_wired_selected_returns_fail_before_all_price_io() -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(analysis)
    calculation = deepcopy(prediction.calculation)
    calculation["selected_view"]["cumulative_returns"] = {
        "bear": -0.0500,
        "base": 0.2000,
        "bull": 0.4000,
    }
    prediction.calculation = calculation
    calls: list[tuple[str, date]] = []

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date + timedelta(days=1200),
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject="BENCH",
        price_loader=lambda subject, through_date: (
            calls.append((subject, through_date))
            or (_ for _ in ()).throw(AssertionError("cross-wired V4 returns must not load prices"))
        ),
    )

    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.resolution == "Long-v4 valuation baseline could not be authenticated"
    assert resolved.metadata["valuation_baseline"]["issue_codes"] == ["identity_mismatch"]
    assert calls == []


@pytest.mark.django_db
def test_long_v4_probability_and_confidence_semantics_are_immutable_identity() -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(analysis)
    prediction.probability_positive = Decimal("0.5000")
    prediction.confidence = Decimal("50.00")
    prediction.confidence_status = "calibrated"
    calculation = deepcopy(prediction.calculation)
    calculation["probability_semantics"] = {
        "status": "published",
        "value": 0.5,
        "reason": "",
    }
    calculation["confidence_semantics"] = {
        "status": "calibrated",
        "value": 50.0,
        "schema": "percent",
    }
    prediction.calculation = calculation
    calls: list[tuple[str, date]] = []

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date + timedelta(days=1200),
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject="BENCH",
        price_loader=lambda subject, through_date: (
            calls.append((subject, through_date))
            or (_ for _ in ()).throw(
                AssertionError("invalid V4 probability semantics must not load prices")
            )
        ),
    )

    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.resolution == "Long-v4 valuation baseline could not be authenticated"
    assert resolved.metadata["valuation_baseline"]["issue_codes"] == ["identity_mismatch"]
    assert calls == []


@pytest.mark.django_db
@pytest.mark.parametrize(
    "mutation",
    [
        "forged_extra",
        "duplicate",
        "reordered",
        "omitted",
        "malformed_uuid",
        "noncanonical_uuid",
        "missing_db_asset",
        "provider",
        "kind",
        "subject",
        "relative_path",
        "sha256",
        "retrieved_at",
        "available_at",
        "post_generated",
        "malformed_catalog",
    ],
)
def test_long_v4_complete_manifest_closure_fails_before_all_price_io(
    mutation: str,
) -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(analysis)
    calculation = deepcopy(prediction.calculation)
    prediction_manifest = deepcopy(prediction.source_assets)

    def replace_mapping_entry(entry: dict[str, str]) -> None:
        calculation["source_manifest"][0] = deepcopy(entry)
        prediction_manifest[0] = deepcopy(entry)

    if mutation == "forged_extra":
        forged = deepcopy(calculation["source_manifest"][0])
        forged["id"] = str(uuid4())
        calculation["source_manifest"].append(deepcopy(forged))
        prediction_manifest.append(deepcopy(forged))
    elif mutation == "duplicate":
        calculation["source_manifest"].append(deepcopy(calculation["source_manifest"][0]))
        prediction_manifest.append(deepcopy(prediction_manifest[0]))
    elif mutation == "reordered":
        calculation["source_manifest"][0], calculation["source_manifest"][1] = (
            calculation["source_manifest"][1],
            calculation["source_manifest"][0],
        )
        prediction_manifest[0], prediction_manifest[1] = (
            prediction_manifest[1],
            prediction_manifest[0],
        )
    elif mutation == "omitted":
        calculation["source_manifest"] = calculation["source_manifest"][1:]
        prediction_manifest = prediction_manifest[1:]
    elif mutation in {"malformed_uuid", "noncanonical_uuid", "missing_db_asset"}:
        replacement = {
            "malformed_uuid": "not-a-uuid",
            "noncanonical_uuid": str(uuid4()).upper(),
            "missing_db_asset": str(uuid4()),
        }[mutation]
        calculation["evidence_catalog"]["sec_mapping_authority"]["mapping_asset"]["id"] = (
            replacement
        )
        calculation_entry = deepcopy(calculation["source_manifest"][0])
        calculation_entry["id"] = replacement
        replace_mapping_entry(calculation_entry)
    elif mutation in {
        "provider",
        "kind",
        "subject",
        "relative_path",
        "sha256",
        "retrieved_at",
        "available_at",
    }:
        entry = deepcopy(calculation["source_manifest"][0])
        entry[mutation] = {
            "provider": "forged_provider",
            "kind": "forged_kind",
            "subject": "forged_subject",
            "relative_path": "forged/path.json",
            "sha256": "f" * 64,
            "retrieved_at": (prediction.generated_at - timedelta(seconds=1)).isoformat(),
            "available_at": (prediction.generated_at - timedelta(seconds=1)).isoformat(),
        }[mutation]
        replace_mapping_entry(entry)
    elif mutation == "post_generated":
        future = prediction.generated_at + timedelta(seconds=1)
        future_asset = DataAsset.objects.create(
            provider="sec",
            kind="sec_company_mapping",
            subject="company_tickers_exchange",
            relative_path=f"outcomes/{uuid4().hex}.json",
            sha256="8" * 64,
            retrieved_at=future,
            available_at=future,
        )
        future_payload = _asset_identity_payload(future_asset)
        calculation["evidence_catalog"]["sec_mapping_authority"]["mapping_asset"] = deepcopy(
            future_payload
        )
        replace_mapping_entry(future_payload)
    else:
        calculation["evidence_catalog"]["raw_fcf_authority"] = {}

    prediction.calculation = calculation
    prediction.source_assets = prediction_manifest
    calls: list[tuple[str, date]] = []

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date + timedelta(days=1200),
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject="BENCH",
        price_loader=lambda subject, through_date: (
            calls.append((subject, through_date))
            or (_ for _ in ()).throw(
                AssertionError("invalid complete manifest must fail before price I/O")
            )
        ),
    )

    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.resolution == "Long-v4 valuation baseline could not be authenticated"
    assert resolved.metadata == {
        "provider": "twelve_data",
        "valuation_baseline": {
            "role": "calculation.target_price.valuation_value",
            "status": "authentication_failed",
            "issue_codes": ["identity_mismatch"],
        },
    }
    assert calls == []


@pytest.mark.django_db
def test_long_v4_raw_sec_replay_failure_is_identity_mismatch_before_price_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(analysis)
    monkeypatch.setattr(
        outcomes_module,
        "_validate_long_v4_raw_sec_replay",
        lambda **_kwargs: (_ for _ in ()).throw(
            outcomes_module.LongV4PersistedAuthorityError(
                "Long-v4 raw SEC physical replay is incompatible"
            )
        ),
    )
    calls: list[tuple[str, date]] = []

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date + timedelta(days=1200),
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject="BENCH",
        price_loader=lambda subject, through_date: (
            calls.append((subject, through_date))
            or (_ for _ in ()).throw(AssertionError("failed SEC replay must precede price I/O"))
        ),
    )

    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.resolution == "Long-v4 valuation baseline could not be authenticated"
    assert resolved.metadata == {
        "provider": "twelve_data",
        "valuation_baseline": {
            "role": "calculation.target_price.valuation_value",
            "status": "authentication_failed",
            "issue_codes": ["identity_mismatch"],
        },
    }
    assert calls == []


@pytest.mark.django_db
def test_long_v4_peer_raw_asset_cannot_authenticate_target_price_closure() -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(analysis)
    available_at = prediction.data_cutoff
    peer_subject = "PEER:US"
    peer_raw = DataAsset.objects.create(
        provider="twelve_data",
        kind="raw_price_history",
        subject=peer_subject,
        relative_path=f"outcomes/{uuid4().hex}.json",
        sha256="3" * 64,
        retrieved_at=available_at,
        available_at=available_at,
    )
    peer_normalized = DataAsset.objects.create(
        provider="twelve_data",
        kind="price_history",
        subject=peer_subject,
        relative_path=f"outcomes/{uuid4().hex}.parquet",
        sha256="4" * 64,
        retrieved_at=available_at,
        available_at=available_at,
        metadata={
            "raw_asset_id": str(peer_raw.pk),
            "raw_sha256": peer_raw.sha256,
        },
    )
    peer_raw_payload = _asset_identity_payload(peer_raw)
    peer_normalized_payload = _asset_identity_payload(peer_normalized)
    calculation = deepcopy(prediction.calculation)
    target_price = calculation["target_price"]
    target_price["raw_asset_id"] = peer_raw_payload["id"]
    target_price["raw_asset_sha256"] = peer_raw_payload["sha256"]
    target_price["raw_asset"] = peer_raw_payload
    peer_price = {
        **deepcopy(target_price),
        "owner_listing_id": str(uuid4()),
        "listing_id": str(uuid4()),
        "subject": peer_subject,
        "normalized_asset_id": peer_normalized_payload["id"],
        "normalized_asset_sha256": peer_normalized_payload["sha256"],
        "normalized_asset": peer_normalized_payload,
        "raw_asset_id": peer_raw_payload["id"],
        "raw_asset_sha256": peer_raw_payload["sha256"],
        "raw_asset": peer_raw_payload,
    }
    calculation["evidence_catalog"]["prices"] = [deepcopy(target_price), peer_price]
    calculation["source_manifest"] = [
        deepcopy(calculation["source_manifest"][0]),
        deepcopy(calculation["source_manifest"][1]),
        peer_raw_payload,
        peer_normalized_payload,
    ]
    prediction.source_assets = deepcopy(calculation["source_manifest"])
    prediction.calculation = calculation
    calls: list[tuple[str, date]] = []

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date + timedelta(days=1200),
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject="BENCH",
        price_loader=lambda subject, through_date: (
            calls.append((subject, through_date))
            or (_ for _ in ()).throw(
                AssertionError("cross-subject raw closure must not load prices")
            )
        ),
    )

    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.resolution == "Long-v4 valuation baseline could not be authenticated"
    assert resolved.metadata["valuation_baseline"]["issue_codes"] == [
        "target_price_identity_mismatch"
    ]
    assert calls == []


@pytest.mark.django_db
def test_long_v4_complete_peer_price_pair_cannot_authenticate_target_listing() -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(analysis)
    peer_subject = "PEER:US"
    peer_raw = DataAsset.objects.create(
        provider="twelve_data",
        kind="raw_price_history",
        subject=peer_subject,
        relative_path=f"outcomes/{uuid4().hex}.json",
        sha256="6" * 64,
        retrieved_at=prediction.generated_at,
        available_at=prediction.generated_at,
    )
    peer_normalized = DataAsset.objects.create(
        provider="twelve_data",
        kind="price_history",
        subject=peer_subject,
        relative_path=f"outcomes/{uuid4().hex}.parquet",
        sha256="7" * 64,
        retrieved_at=prediction.generated_at,
        available_at=prediction.generated_at,
        metadata={
            "raw_asset_id": str(peer_raw.pk),
            "raw_sha256": peer_raw.sha256,
        },
    )
    raw_payload = _asset_identity_payload(peer_raw)
    normalized_payload = _asset_identity_payload(peer_normalized)
    calculation = deepcopy(prediction.calculation)
    target_price = calculation["target_price"]
    target_price.update(
        {
            "subject": peer_subject,
            "normalized_asset_id": normalized_payload["id"],
            "normalized_asset_sha256": normalized_payload["sha256"],
            "normalized_asset": normalized_payload,
            "raw_asset_id": raw_payload["id"],
            "raw_asset_sha256": raw_payload["sha256"],
            "raw_asset": raw_payload,
        }
    )
    calculation["price_subject"] = peer_subject
    calculation["evidence_catalog"]["prices"] = [deepcopy(target_price)]
    calculation["source_manifest"] = [
        deepcopy(calculation["source_manifest"][0]),
        normalized_payload,
        raw_payload,
    ]
    prediction.price_subject = peer_subject
    prediction.source_assets = deepcopy(calculation["source_manifest"])
    prediction.calculation = calculation
    calls: list[tuple[str, date]] = []

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date + timedelta(days=1200),
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject="BENCH",
        price_loader=lambda subject, through_date: (
            calls.append((subject, through_date))
            or (_ for _ in ()).throw(
                AssertionError("a peer price pair must not authenticate the target")
            )
        ),
    )

    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.resolution == "Long-v4 valuation baseline could not be authenticated"
    assert resolved.metadata["valuation_baseline"]["issue_codes"] == [
        "target_price_identity_mismatch"
    ]
    assert calls == []


@pytest.mark.django_db
def test_long_v4_persisted_normalized_to_raw_relation_is_authenticated() -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(analysis)
    unrelated_raw = DataAsset.objects.create(
        provider="twelve_data",
        kind="raw_price_history",
        subject=prediction.price_subject,
        relative_path=f"outcomes/{uuid4().hex}.json",
        sha256="5" * 64,
        retrieved_at=prediction.data_cutoff,
        available_at=prediction.data_cutoff,
    )
    unrelated_payload = _asset_identity_payload(unrelated_raw)
    calculation = deepcopy(prediction.calculation)
    target_price = calculation["target_price"]
    target_price["raw_asset_id"] = unrelated_payload["id"]
    target_price["raw_asset_sha256"] = unrelated_payload["sha256"]
    target_price["raw_asset"] = unrelated_payload
    calculation["evidence_catalog"]["prices"] = [deepcopy(target_price)]
    calculation["source_manifest"] = [
        deepcopy(calculation["source_manifest"][0]),
        deepcopy(calculation["source_manifest"][1]),
        unrelated_payload,
    ]
    prediction.source_assets = deepcopy(calculation["source_manifest"])
    prediction.calculation = calculation
    calls: list[tuple[str, date]] = []

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date + timedelta(days=1200),
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject="BENCH",
        price_loader=lambda subject, through_date: (
            calls.append((subject, through_date))
            or (_ for _ in ()).throw(
                AssertionError("unrelated persisted raw closure must not load prices")
            )
        ),
    )

    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.resolution == "Long-v4 valuation baseline could not be authenticated"
    assert resolved.metadata["valuation_baseline"]["issue_codes"] == [
        "target_price_identity_mismatch"
    ]
    assert calls == []


@pytest.mark.django_db
def test_long_v4_research_price_evidence_between_cutoff_and_issuance_matures(
    tmp_path,
) -> None:
    _, analysis = _analysis()
    data_cutoff = datetime(2026, 1, 2, 21, tzinfo=timezone.get_current_timezone())
    generated_at = datetime(2026, 1, 10, 21, tzinfo=timezone.get_current_timezone())
    source_available_at = datetime(2026, 1, 5, 12, tzinfo=timezone.get_current_timezone())
    analysis.run.data_cutoff = data_cutoff
    analysis.run.generated_at = generated_at
    analysis.run.save(update_fields=["data_cutoff", "generated_at"])
    prediction = _v4_prediction(
        analysis,
        source_available_at=source_available_at,
    )
    sessions = _business_dates_after(prediction.target_date, 756)
    evaluation_time = _v4_evaluation_time()
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        prediction.price_subject,
        evaluation_time,
        sessions,
        [120.0] * len(sessions),
        provider="twelve_data",
    )

    result = evaluate_prediction(
        prediction,
        provider="twelve_data",
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        store=store,
    )

    assert prediction.data_cutoff < source_available_at < prediction.generated_at
    assert prediction.generated_at == prediction.analysis.run.generated_at
    assert result.outcome.status == PredictionOutcome.Status.MATURED
    assert result.outcome.actual_return == Decimal("0.2000")


@pytest.mark.django_db
def test_long_v4_price_evidence_after_issuance_fails_before_all_price_io() -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(
        analysis,
        source_available_at=analysis.run.generated_at + timedelta(seconds=1),
    )
    calls: list[tuple[str, date]] = []

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date + timedelta(days=1200),
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject="BENCH",
        price_loader=lambda subject, through_date: (
            calls.append((subject, through_date))
            or (_ for _ in ()).throw(
                AssertionError("post-issuance evidence must fail before price I/O")
            )
        ),
    )

    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.resolution == "Long-v4 valuation baseline could not be authenticated"
    assert resolved.metadata["valuation_baseline"]["issue_codes"] == [
        "identity_mismatch",
        "target_price_identity_mismatch",
    ]
    assert calls == []


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("case", "expected_status", "expected_resolution"),
    [
        (
            "missing",
            PredictionOutcome.Status.UNRESOLVED,
            "Evaluation price history has no exact target-date close",
        ),
        (
            "equal",
            PredictionOutcome.Status.MATURED,
            "Matured after 756 observed sessions",
        ),
        (
            "mismatch",
            PredictionOutcome.Status.CORPORATE_EVENT,
            "Target-date price changed in the evaluation vintage",
        ),
    ],
)
def test_long_v4_requires_exact_target_date_close(
    case: str,
    expected_status: str,
    expected_resolution: str,
) -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(analysis, valuation_value=Decimal("100"))
    sessions = _business_dates_after(prediction.target_date, 756)
    dates = [prediction.target_date, *sessions]
    closes = [100.0, *([110.0] * len(sessions))]
    if case == "missing":
        dates[0] = prediction.target_date - timedelta(days=1)
    elif case == "mismatch":
        closes[0] = 99.0
    frame = pl.DataFrame({"date": dates, "close": closes})
    calls: list[str] = []

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=sessions[-1],
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject=None,
        price_loader=lambda subject, _through: calls.append(subject) or frame,
    )

    assert resolved.status == expected_status
    assert expected_resolution in resolved.resolution
    assert calls == [prediction.price_subject]
    baseline = resolved.metadata["valuation_baseline"]
    assert baseline["role"] == "calculation.target_price.valuation_value"
    assert baseline["valuation_value"] == "100"
    assert baseline["ledger_value"] == "100.000000"
    assert baseline["comparison_type"] == "Decimal(str(target_close)) == valuation_value"
    assert baseline["target_date"] == prediction.target_date.isoformat()
    assert baseline["return_quantum"] == "0.0001"
    assert baseline["rounding"] == "ROUND_HALF_EVEN"
    if case == "missing":
        assert baseline["status"] == "target_close_missing"
        assert baseline["target_close"] is None
    elif case == "equal":
        assert baseline["status"] == "authenticated"
        assert baseline["target_close"] == "100.0"
    else:
        assert baseline["status"] == "target_close_mismatch"
        assert baseline["target_close"] == "99.0"


@pytest.mark.django_db
@pytest.mark.parametrize(
    (
        "terminal_close",
        "bear",
        "base",
        "bull",
        "expected_actual",
        "expected_direction",
        "expected_error",
    ),
    [
        (
            100.005,
            Decimal("0.0000"),
            Decimal("0.0000"),
            Decimal("0.0000"),
            Decimal("0.0000"),
            True,
            Decimal("0.0000"),
        ),
        (
            100.015,
            Decimal("0.0000"),
            Decimal("0.0001"),
            Decimal("0.0002"),
            Decimal("0.0002"),
            True,
            Decimal("0.0001"),
        ),
        (
            99.995,
            Decimal("0.0000"),
            Decimal("0.0000"),
            Decimal("0.0000"),
            Decimal("-0.0000"),
            True,
            Decimal("-0.0000"),
        ),
        (
            99.985,
            Decimal("-0.0002"),
            Decimal("-0.0001"),
            Decimal("0.0000"),
            Decimal("-0.0002"),
            True,
            Decimal("-0.0001"),
        ),
    ],
)
def test_long_v4_decimal_return_drives_every_outcome_field(
    terminal_close: float,
    bear: Decimal,
    base: Decimal,
    bull: Decimal,
    expected_actual: Decimal,
    expected_direction: bool,
    expected_error: Decimal,
) -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(
        analysis,
        valuation_value=Decimal("100"),
        bear=bear,
        base=base,
        bull=bull,
    )
    sessions = _business_dates_after(prediction.target_date, 756)
    frame = pl.DataFrame(
        {
            "date": [prediction.target_date, *sessions],
            "close": [100.0, *([terminal_close] * len(sessions))],
        }
    )

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=sessions[-1],
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject=None,
        price_loader=lambda _subject, _through: frame,
    )

    assert resolved.status == PredictionOutcome.Status.MATURED
    assert resolved.actual_return == expected_actual
    assert resolved.success is None
    assert resolved.direction_correct is expected_direction
    assert resolved.interval_covered is True
    assert resolved.error == expected_error
    assert resolved.signed_error == expected_error


@pytest.mark.django_db
def test_long_v4_benchmark_behavior_matches_legacy_benchmark_path() -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(analysis, valuation_value=Decimal("100"))
    sessions = _business_dates_after(prediction.target_date, 756)
    stock_frame = pl.DataFrame(
        {
            "date": [prediction.target_date, *sessions],
            "close": [100.0, *([120.0] * len(sessions))],
        }
    )
    benchmark_frame = pl.DataFrame(
        {
            "date": [prediction.target_date - timedelta(days=1), sessions[-1]],
            "close": [200.0, 220.0],
        }
    )

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=sessions[-1],
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject="BENCH",
        price_loader=lambda subject, _through: (
            benchmark_frame if subject == "BENCH" else stock_frame
        ),
    )

    assert resolved.status == PredictionOutcome.Status.MATURED
    assert resolved.actual_return == Decimal("0.2000")
    assert resolved.benchmark_return == Decimal("0.1")
    assert "Benchmark return uses" in resolved.resolution


@pytest.mark.django_db
def test_long_v4_terminal_corporate_event_is_idempotently_skipped(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(analysis, valuation_value=Decimal("100"))
    sessions = _business_dates_after(prediction.target_date, 756)
    store = AssetStore(tmp_path)
    evaluation_time = _v4_evaluation_time()
    _register_price_asset(
        store,
        prediction.price_subject,
        evaluation_time,
        sessions,
        [110.0] * len(sessions),
        provider="twelve_data",
        baseline_close=Decimal("99"),
    )

    first = evaluate_prediction(
        prediction,
        provider="twelve_data",
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        store=store,
    )

    def forbidden_price_frame(*_args, **_kwargs) -> pl.DataFrame:
        raise AssertionError("terminal corporate event must not retry price I/O")

    monkeypatch.setattr(AsOfData, "price_frame", forbidden_price_frame)
    second = evaluate_prediction(
        prediction,
        provider="twelve_data",
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time + timedelta(days=1),
        store=store,
    )

    assert first.outcome.status == PredictionOutcome.Status.CORPORATE_EVENT
    assert first.outcome.resolution == "Target-date price changed in the evaluation vintage"
    assert second.action == "skipped"
    assert second.outcome.pk == first.outcome.pk
    assert second.outcome.metadata == first.outcome.metadata


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("guard", "expected_resolution"),
    [
        ("future", "Evaluation date is after actual evaluation time"),
        ("before", "Evaluation date is before prediction target date"),
        ("all_null", "Withheld forecast has no scenario to evaluate"),
    ],
)
def test_long_v4_preserves_date_and_all_null_guard_order(
    guard: str,
    expected_resolution: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, analysis = _analysis()
    prediction = _v4_prediction(
        analysis,
        bear=None if guard == "all_null" else Decimal("-0.1"),
        base=None if guard == "all_null" else Decimal("0.1"),
        bull=None if guard == "all_null" else Decimal("0.2"),
    )
    prediction.calculation = {}
    monkeypatch.setattr(
        outcomes_module,
        "_validate_long_v4_persisted_authority",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("date/all-null guards must precede DB authority")
        ),
    )
    monkeypatch.setattr(
        outcomes_module,
        "_validate_long_v4_raw_sec_replay",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("date/all-null guards must precede SEC replay")
        ),
    )
    evaluation_date = {
        "future": date(2031, 1, 1),
        "before": prediction.target_date - timedelta(days=1),
        "all_null": prediction.target_date,
    }[guard]

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=evaluation_date,
        evaluated_at=_v4_evaluation_time(),
        benchmark_subject="BENCH",
        price_loader=lambda _subject, _through: (_ for _ in ()).throw(
            AssertionError("guarded branch must not load")
        ),
    )

    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.resolution == expected_resolution


@pytest.mark.django_db
def test_withheld_decision_buy_still_matures_and_scores_from_actual_return(tmp_path) -> None:
    _, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        evidence_role=Prediction.EvidenceRole.DECISION,
        price_provider="synthetic",
        recommendation=Recommendation.BUY,
        bear=None,
        base=None,
        bull=None,
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store, prediction.listing.ticker, _evaluation_time(), sessions, [150] * 10
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.MATURED
    assert result.outcome.actual_return == Decimal("0.5")
    assert result.outcome.success is True
    assert result.outcome.error is None
    assert result.outcome.interval_covered is None


@pytest.mark.django_db
def test_withheld_decision_hold_stays_unresolved_without_bear_bull_range(tmp_path) -> None:
    _, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        evidence_role=Prediction.EvidenceRole.DECISION,
        price_provider="synthetic",
        recommendation=Recommendation.HOLD,
        bear=None,
        base=None,
        bull=None,
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store, prediction.listing.ticker, _evaluation_time(), sessions, [150] * 10
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.UNRESOLVED
    assert (
        result.outcome.resolution == "HOLD success requires non-null stored bear and bull returns"
    )
    assert result.outcome.success is None
    assert result.outcome.actual_return is None


@pytest.mark.django_db
def test_batch_evaluation_reuses_identical_price_frame(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing, analysis = _analysis()
    first = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
        version="cache-one",
    )
    second = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
        version="cache-two",
    )
    sessions = _business_dates_after(first.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        _evaluation_time(),
        sessions,
        [101 + index for index in range(10)],
    )
    original_price_frame = AsOfData.price_frame
    calls: list[tuple[str, str, date | None]] = []

    def counting_price_frame(
        self: AsOfData,
        *,
        provider: str,
        subject: str,
        through_date: date | None = None,
    ) -> pl.DataFrame:
        calls.append((provider, subject, through_date))
        return original_price_frame(
            self,
            provider=provider,
            subject=subject,
            through_date=through_date,
        )

    monkeypatch.setattr(AsOfData, "price_frame", counting_price_frame)

    results = evaluate_predictions(
        [first, second],
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert [result.outcome.status for result in results] == [
        PredictionOutcome.Status.MATURED,
        PredictionOutcome.Status.MATURED,
    ]
    assert calls == [("synthetic", listing.ticker, sessions[-1])]


@pytest.mark.django_db
def test_batch_provider_conflict_is_rejected_before_any_outcome_is_written(
    tmp_path,
) -> None:
    listing, analysis = _analysis()
    synthetic = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
        version="synthetic-source",
        price_provider="synthetic",
    )
    conflicting = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
        version="other-source",
        price_provider="twelve_data",
    )
    sessions = _business_dates_after(synthetic.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        _evaluation_time(),
        sessions,
        [101 + index for index in range(10)],
    )

    with pytest.raises(ValueError, match="conflicts with requested provider"):
        evaluate_predictions(
            [synthetic, conflicting],
            provider="synthetic",
            evaluation_date=sessions[-1],
            evaluation_time=_evaluation_time(),
            store=store,
        )

    assert PredictionOutcome.objects.count() == 0


@pytest.mark.django_db
def test_missing_data_asset_creates_unresolved_outcome(tmp_path) -> None:
    _, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=prediction.target_date + timedelta(days=30),
        evaluation_time=_evaluation_time(),
        store=AssetStore(tmp_path),
    )

    assert result.outcome.status == PredictionOutcome.Status.UNRESOLVED
    assert "Unable to load evaluation price history" in result.outcome.resolution
    assert result.outcome.actual_return is None
    assert result.outcome.success is None


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("evaluation_date", "expected_resolution"),
    [
        (date(2027, 1, 1), "Evaluation date is after actual evaluation time"),
        (date(2026, 1, 1), "Evaluation date is before prediction target date"),
    ],
)
def test_invalid_evaluation_cutoff_dates_are_unresolved(
    tmp_path, evaluation_date: date, expected_resolution: str
) -> None:
    _, analysis = _analysis()
    prediction = _prediction(analysis, horizon=Prediction.Horizon.SHORT)

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=evaluation_date,
        evaluation_time=_evaluation_time(),
        store=AssetStore(tmp_path),
    )

    assert result.outcome.status == PredictionOutcome.Status.UNRESOLVED
    assert result.outcome.resolution == expected_resolution
    assert result.outcome.actual_return is None


@pytest.mark.django_db
def test_duplicate_usable_session_dates_are_unresolved_not_counted(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        _evaluation_time(),
        sessions + [sessions[0]],
        [101 + i for i in range(10)] + [111],
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.UNRESOLVED
    assert result.outcome.resolution == "Duplicate usable price session dates in evaluation data"
    assert result.outcome.actual_return is None


@pytest.mark.django_db
def test_benchmark_return_uses_close_at_or_before_target_and_same_evaluation_session(
    tmp_path,
) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    evaluation_time = _evaluation_time()
    _register_price_asset(
        store, listing.ticker, evaluation_time, sessions, [101 + i for i in range(10)]
    )
    _register_price_asset(
        store,
        "BENCH",
        evaluation_time,
        [
            prediction.target_date - timedelta(days=1),
            sessions[-2],
            sessions[-1] + timedelta(days=3),
        ],
        [200, 220, 9999],
        baseline_date=None,
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        benchmark_subject="BENCH",
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.MATURED
    assert result.outcome.benchmark_return == Decimal("0.1")
    assert "Benchmark return uses" in result.outcome.resolution


@pytest.mark.django_db
def test_duplicate_benchmark_session_dates_do_not_block_stock_maturity(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    evaluation_time = _evaluation_time()
    _register_price_asset(
        store, listing.ticker, evaluation_time, sessions, [101 + i for i in range(10)]
    )
    _register_price_asset(
        store,
        "BENCH",
        evaluation_time,
        [prediction.target_date, sessions[-1], sessions[-1]],
        [200, 220, 221],
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        benchmark_subject="BENCH",
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.MATURED
    assert result.outcome.actual_return == Decimal("0.1")
    assert result.outcome.success is True
    assert result.outcome.benchmark_return is None
    assert (
        "Benchmark unavailable: Duplicate usable price session dates"
        in (result.outcome.metadata["benchmark_resolution"])
    )


@pytest.mark.django_db
def test_existing_matured_outcome_is_idempotently_skipped(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    existing = PredictionOutcome.objects.create(
        prediction=prediction,
        evaluated_at=_evaluation_time() - timedelta(days=1),
        evaluation_date=prediction.target_date + timedelta(days=20),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.1234"),
        success=True,
        resolution="Already matured",
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(store, listing.ticker, _evaluation_time(), sessions, [1_000_000] * 10)

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    existing.refresh_from_db()
    assert result.action == "skipped"
    assert existing.actual_return == Decimal("0.1234")
    assert existing.resolution == "Already matured"


@pytest.mark.django_db
def test_stale_prediction_instance_cannot_overwrite_terminal_outcome(tmp_path) -> None:
    listing, analysis = _analysis()
    stale_prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
    )
    with pytest.raises(PredictionOutcome.DoesNotExist):
        _ = stale_prediction.outcome

    PredictionOutcome.objects.create(
        prediction=Prediction.objects.get(pk=stale_prediction.pk),
        evaluated_at=_evaluation_time() - timedelta(days=1),
        evaluation_date=stale_prediction.target_date + timedelta(days=20),
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.1234"),
        success=True,
        resolution="Matured elsewhere",
    )

    result = evaluate_prediction(
        stale_prediction,
        provider="synthetic",
        evaluation_date=date(2027, 1, 1),
        evaluation_time=_evaluation_time(),
        store=AssetStore(tmp_path),
    )

    outcome = PredictionOutcome.objects.get(prediction=stale_prediction)
    assert result.action == "skipped"
    assert outcome.status == PredictionOutcome.Status.MATURED
    assert outcome.actual_return == Decimal("0.1234")


@pytest.mark.django_db
def test_adjusted_target_price_is_classified_as_corporate_event(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        _evaluation_time(),
        sessions,
        [50 + index for index in range(10)],
        baseline_close=Decimal("50"),
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.CORPORATE_EVENT
    assert result.outcome.actual_return is None
    assert result.outcome.success is None
    assert result.outcome.metadata["prediction_price"] == 100.0
    assert result.outcome.metadata["evaluation_vintage_target_close"] == 50.0


@pytest.mark.django_db
def test_evaluator_does_not_use_future_rows_in_eligible_asset(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        _evaluation_time(),
        sessions + [sessions[-1] + timedelta(days=1)],
        [101 + i for i in range(10)] + [9999],
    )

    result = evaluate_prediction(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluation_time=_evaluation_time(),
        store=store,
    )

    assert result.outcome.status == PredictionOutcome.Status.MATURED
    assert result.outcome.evaluation_date == sessions[-1]
    assert result.outcome.actual_return == Decimal("0.1")


@pytest.mark.django_db
def test_non_v4_outcome_keeps_prior_close_and_ledger_float_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    frame = pl.DataFrame(
        {
            "date": [prediction.target_date - timedelta(days=1), *sessions],
            "close": [100.0, *([110.0] * len(sessions))],
        }
    )
    monkeypatch.setattr(
        outcomes_module,
        "_validate_long_v4_raw_sec_replay",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-V4 outcomes must not replay raw SEC evidence")
        ),
    )

    resolved = resolve_outcome(
        prediction,
        provider="synthetic",
        evaluation_date=sessions[-1],
        evaluated_at=_evaluation_time(),
        benchmark_subject=None,
        price_loader=lambda _subject, _through: frame,
    )

    assert resolved.status == PredictionOutcome.Status.MATURED
    assert resolved.actual_return == Decimal("0.1")
    assert "valuation_baseline" not in resolved.metadata


@pytest.mark.django_db
def test_success_semantics_are_recommendation_specific(tmp_path) -> None:
    listing, analysis = _analysis()
    store = AssetStore(tmp_path)
    evaluation_time = _evaluation_time()
    sessions = _business_dates_after(analysis.run.target_date, 10)
    _register_price_asset(store, listing.ticker, evaluation_time, sessions, [98] * 10)
    buy = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY, version="buy"
    )
    avoid = _prediction(
        analysis,
        horizon=Prediction.Horizon.MEDIUM,
        recommendation=Recommendation.AVOID,
        version="avoid",
        target_date=analysis.run.target_date,
    )
    hold = _prediction(
        analysis,
        horizon=Prediction.Horizon.LONG,
        recommendation=Recommendation.HOLD,
        version="hold",
        target_date=analysis.run.target_date,
        bear=Decimal("-0.05"),
        bull=Decimal("0.05"),
    )
    _register_price_asset(
        store,
        listing.ticker,
        evaluation_time + timedelta(seconds=1),
        _business_dates_after(analysis.run.target_date, 756),
        [98] * 756,
    )

    assert (
        evaluate_prediction(
            buy,
            provider="synthetic",
            evaluation_date=sessions[-1],
            evaluation_time=evaluation_time,
            store=store,
        ).outcome.success
        is False
    )
    assert (
        evaluate_prediction(
            avoid,
            provider="synthetic",
            evaluation_date=_business_dates_after(analysis.run.target_date, 252)[-1],
            evaluation_time=evaluation_time + timedelta(seconds=1),
            store=store,
        ).outcome.success
        is True
    )
    assert (
        evaluate_prediction(
            hold,
            provider="synthetic",
            evaluation_date=_business_dates_after(analysis.run.target_date, 756)[-1],
            evaluation_time=datetime(2029, 1, 1, 12, tzinfo=timezone.get_current_timezone()),
            store=store,
        ).outcome.success
        is True
    )


@pytest.mark.django_db
def test_evaluate_command_all_pending_uses_tmp_asset_store(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
        price_provider="synthetic",
    )
    other_provider = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
        version="other-provider",
        price_provider="twelve_data",
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        timezone.now() - timedelta(minutes=1),
        sessions,
        [101 + i for i in range(10)],
    )

    with override_settings(DATA_DIR=tmp_path):
        call_command(
            "evaluate",
            all_pending=True,
            provider="synthetic",
            evaluation_date=sessions[-1].isoformat(),
            stdout=StringIO(),
        )

    assert (
        PredictionOutcome.objects.get(prediction=prediction).status
        == PredictionOutcome.Status.MATURED
    )
    assert not PredictionOutcome.objects.filter(prediction=other_provider).exists()


@pytest.mark.django_db
def test_evaluate_command_default_provider_is_synthetic_demo(tmp_path) -> None:
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, recommendation=Recommendation.BUY
    )
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        timezone.now() - timedelta(minutes=1),
        sessions,
        [101 + i for i in range(10)],
        provider="synthetic_demo",
    )

    with override_settings(DATA_DIR=tmp_path):
        call_command(
            "evaluate",
            str(prediction.pk),
            evaluation_date=sessions[-1].isoformat(),
            stdout=StringIO(),
        )

    assert (
        PredictionOutcome.objects.get(prediction=prediction).metadata["provider"]
        == "synthetic_demo"
    )


@pytest.mark.django_db
def test_prediction_outcome_constraints_enforce_matured_and_unresolved_contracts() -> None:
    _, analysis = _analysis()
    matured = _prediction(analysis, horizon=Prediction.Horizon.SHORT, version="bad-matured")
    unresolved = _prediction(analysis, horizon=Prediction.Horizon.MEDIUM, version="bad-unresolved")
    corporate_event = _prediction(
        analysis,
        horizon=Prediction.Horizon.LONG,
        version="bad-corporate-event",
    )

    with pytest.raises(IntegrityError), transaction.atomic():
        PredictionOutcome.objects.create(
            prediction=matured,
            evaluated_at=_evaluation_time(),
            evaluation_date=matured.target_date,
            status=PredictionOutcome.Status.MATURED,
            resolution="invalid",
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        PredictionOutcome.objects.create(
            prediction=unresolved,
            evaluated_at=_evaluation_time(),
            evaluation_date=unresolved.target_date,
            status=PredictionOutcome.Status.UNRESOLVED,
            actual_return=Decimal("0.01"),
            resolution="invalid",
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        PredictionOutcome.objects.create(
            prediction=corporate_event,
            evaluated_at=_evaluation_time(),
            evaluation_date=corporate_event.target_date,
            status=PredictionOutcome.Status.CORPORATE_EVENT,
            actual_return=Decimal("0.01"),
            resolution="invalid",
        )


@pytest.mark.django_db
def test_outcome_validation_separates_decision_and_advisory_success() -> None:
    _, analysis = _analysis()
    decision = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        version="decision-validation",
    )
    advisory = _prediction(
        analysis,
        horizon=Prediction.Horizon.TWELVE_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        version="advisory-validation",
    )

    decision_outcome = PredictionOutcome(
        prediction=decision,
        evaluated_at=_evaluation_time(),
        evaluation_date=decision.target_date,
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.01"),
        success=None,
        resolution="invalid decision",
    )
    advisory_outcome = PredictionOutcome(
        prediction=advisory,
        evaluated_at=_evaluation_time(),
        evaluation_date=advisory.target_date,
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.01"),
        success=True,
        resolution="invalid advisory",
    )

    with pytest.raises(ValidationError, match="decision outcomes require"):
        decision_outcome.full_clean()
    with pytest.raises(ValidationError, match="must not use decision success"):
        advisory_outcome.full_clean()
    with pytest.raises(IntegrityError), transaction.atomic():
        PredictionOutcome.objects.create(
            prediction=decision,
            evaluated_at=_evaluation_time(),
            evaluation_date=decision.target_date,
            status=PredictionOutcome.Status.MATURED,
            actual_return=Decimal("0.01"),
            success=None,
            resolution="invalid decision",
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        PredictionOutcome.objects.create(
            prediction=advisory,
            evaluated_at=_evaluation_time(),
            evaluation_date=advisory.target_date,
            status=PredictionOutcome.Status.MATURED,
            actual_return=Decimal("0.01"),
            success=True,
            resolution="invalid advisory",
        )


@pytest.mark.django_db
def test_prediction_horizon_and_evidence_role_must_match() -> None:
    _, analysis = _analysis()

    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.SIX_MONTH,
            evidence_role=Prediction.EvidenceRole.DECISION,
            version="invalid-canonical-decision",
        )


@pytest.mark.django_db
def test_outcome_role_integrity_triggers_are_installed() -> None:
    table_name = PredictionOutcome._meta.db_table
    with connection.cursor() as cursor:
        if connection.vendor == "sqlite":
            cursor.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'trigger' AND tbl_name = %s
                """,
                [table_name],
            )
            expected_triggers = {
                "research_predictionoutcome_role_insert",
                "research_predictionoutcome_role_update",
            }
        elif connection.vendor == "postgresql":
            cursor.execute(
                """
                SELECT trigger_name
                FROM information_schema.triggers
                WHERE event_object_schema = current_schema()
                  AND event_object_table = %s
                """,
                [table_name],
            )
            expected_triggers = {"research_predictionoutcome_role_guard"}
        else:
            pytest.skip(f"Unsupported database vendor: {connection.vendor}")

        trigger_names = {row[0] for row in cursor.fetchall()}

    assert expected_triggers <= trigger_names


@pytest.mark.django_db
def test_prediction_constraints_require_positive_price_and_valid_data_cutoff() -> None:
    _, analysis = _analysis()

    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.SHORT,
            version="bad-price",
            price_at_prediction=Decimal("0"),
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.MEDIUM,
            version="bad-cutoff",
            data_cutoff=analysis.run.generated_at + timedelta(seconds=1),
        )


@pytest.mark.django_db
def test_prediction_constraints_require_non_null_returns_above_negative_one() -> None:
    _, analysis = _analysis()

    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.SHORT,
            version="bad-bear-floor",
            bear=Decimal("-1.0001"),
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.MEDIUM,
            version="bad-base-floor",
            base=Decimal("-1.0001"),
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.LONG,
            version="bad-bull-floor",
            bull=Decimal("-1.0001"),
        )


@pytest.mark.django_db
def test_prediction_scenario_constraint_keeps_all_null_or_ordered_all_present_invariant() -> None:
    _, analysis = _analysis()

    all_null = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        version="all-null",
        bear=None,
        base=None,
        bull=None,
    )
    assert all_null.bear_return is None
    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.MEDIUM,
            version="partial-null",
            bear=Decimal("-0.10"),
            base=None,
            bull=Decimal("0.10"),
        )
    with pytest.raises(IntegrityError), transaction.atomic():
        _prediction(
            analysis,
            horizon=Prediction.Horizon.LONG,
            version="unordered",
            bear=Decimal("0.20"),
            base=Decimal("0.10"),
            bull=Decimal("0.30"),
        )


@pytest.mark.django_db
def test_pending_job_selection_is_provider_and_session_maturity_aware() -> None:
    _, analysis = _analysis()
    live_short = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        version="live-short",
        source_assets=[{"provider": "twelve_data"}],
    )
    _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        version="synthetic-short",
        source_assets=[{"provider": "synthetic_demo"}],
    )
    _prediction(
        analysis,
        horizon=Prediction.Horizon.MEDIUM,
        version="live-medium",
        source_assets=[{"provider": "twelve_data"}],
    )

    selected = eligible_pending_predictions(
        provider="twelve_data",
        evaluation_date=date(2026, 1, 16),
    )

    assert selected == [live_short]


@pytest.mark.django_db
def test_pending_job_selection_excludes_terminal_corporate_events() -> None:
    _, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        version="terminal-short",
        source_assets=[{"provider": "twelve_data"}],
    )
    PredictionOutcome.objects.create(
        prediction=prediction,
        evaluated_at=_evaluation_time(),
        evaluation_date=date(2026, 1, 16),
        status=PredictionOutcome.Status.CORPORATE_EVENT,
        resolution="Split basis requires review",
    )

    assert (
        eligible_pending_predictions(
            provider="twelve_data",
            evaluation_date=date(2026, 1, 16),
        )
        == []
    )


def _analysis() -> tuple[Listing, StockAnalysis]:
    company = Company.objects.create(name=f"Outcome Co {uuid4().hex[:6]}", country="US")
    security = Security.objects.create(company=company)
    listing = Listing.objects.create(
        security=security,
        ticker=f"O{uuid4().hex[:6]}",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    universe = Universe.objects.create(
        slug=f"outcome-{uuid4().hex[:8]}", name="Outcomes", config_version="1"
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=date(2026, 1, 2),
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="a" * 64,
    )
    generated_at = datetime(2026, 1, 2, 21, tzinfo=timezone.get_current_timezone())
    run = AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=generated_at,
        target_date=date(2026, 1, 2),
        universe_snapshot=snapshot,
        config_version="default-v1",
        config_hash="b" * 64,
        code_revision="test",
    )
    analysis = StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("100"),
        overall_score=Decimal("70"),
        recommendation=Recommendation.HOLD,
        risk_score=Decimal("30"),
        risk_class=RiskClass.LOW,
        confidence=Decimal("60"),
    )
    return listing, analysis


def _prediction(
    analysis: StockAnalysis,
    *,
    horizon: Prediction.Horizon,
    recommendation: Recommendation = Recommendation.HOLD,
    version: str | None = None,
    target_date: date | None = None,
    bear: Decimal | None = Decimal("-0.10"),
    base: Decimal | None = Decimal("0.02"),
    bull: Decimal | None = Decimal("0.10"),
    price_at_prediction: Decimal = Decimal("100"),
    data_cutoff: datetime | None = None,
    source_assets: list[dict[str, str]] | None = None,
    evidence_role: Prediction.EvidenceRole = Prediction.EvidenceRole.DECISION,
    evidence_grade: UniverseSnapshot.Grade = UniverseSnapshot.Grade.RESEARCH,
    price_provider: str = "",
    price_subject: str = "",
    method_version: str = "",
    config_hash: str = "b" * 64,
    calculation: dict[str, object] | None = None,
    probability_positive: Decimal | None = None,
    confidence: Decimal = Decimal("60"),
    confidence_status: str = "heuristic",
    insufficiency_reason: str = "",
) -> Prediction:
    return Prediction.objects.create(
        analysis=analysis,
        listing=analysis.listing,
        generated_at=analysis.run.generated_at,
        target_date=target_date or analysis.run.target_date,
        horizon=horizon,
        evidence_role=evidence_role,
        evidence_grade=evidence_grade,
        price_provider=price_provider,
        price_subject=price_subject,
        price_at_prediction=price_at_prediction,
        bear_return=bear,
        base_return=base,
        bull_return=bull,
        probability_positive=probability_positive,
        confidence=confidence,
        confidence_status=confidence_status,
        insufficiency_reason=insufficiency_reason,
        recommendation=recommendation,
        overall_score=Decimal("70"),
        component_scores={},
        model_version=version or f"outcome-{horizon.value}",
        method_version=method_version,
        config_hash=config_hash,
        data_cutoff=data_cutoff or analysis.run.generated_at,
        source_assets=source_assets or [],
        calculation=calculation or {},
        code_revision="test",
    )


def _v4_prediction(
    analysis: StockAnalysis,
    *,
    valuation_value: Decimal = Decimal("100"),
    bear: Decimal | None = Decimal("-0.1000"),
    base: Decimal | None = Decimal("0.1000"),
    bull: Decimal | None = Decimal("0.3000"),
    source_available_at: datetime | None = None,
) -> Prediction:
    listing = analysis.listing
    if not listing.provider_symbol:
        listing.provider_symbol = listing.ticker
        listing.save(update_fields=["provider_symbol"])
    ledger_value = canonical_long_v4_price(valuation_value)
    analysis.current_price = ledger_value
    analysis.save(update_fields=["current_price"])
    model_version = f"{LONG_V4_VERSION}-outcome-test"
    source_available_at = source_available_at or analysis.run.data_cutoff
    mapping_asset_row = DataAsset.objects.create(
        provider="sec",
        kind="sec_company_mapping",
        subject="company_tickers_exchange",
        relative_path=f"outcomes/{uuid4().hex}.json",
        sha256="0" * 64,
        retrieved_at=source_available_at,
        available_at=source_available_at,
    )
    raw_asset_row = DataAsset.objects.create(
        provider="twelve_data",
        kind="raw_price_history",
        subject=listing.provider_symbol,
        relative_path=f"outcomes/{uuid4().hex}.json",
        sha256="2" * 64,
        retrieved_at=source_available_at,
        available_at=source_available_at,
    )
    normalized_asset_row = DataAsset.objects.create(
        provider="twelve_data",
        kind="price_history",
        subject=listing.provider_symbol,
        relative_path=f"outcomes/{uuid4().hex}.parquet",
        sha256="1" * 64,
        retrieved_at=source_available_at,
        available_at=source_available_at,
        metadata={
            "raw_asset_id": str(raw_asset_row.pk),
            "raw_sha256": raw_asset_row.sha256,
        },
    )
    mapping_asset = _asset_identity_payload(mapping_asset_row)
    normalized_asset = _asset_identity_payload(normalized_asset_row)
    raw_asset = _asset_identity_payload(raw_asset_row)
    target = {
        "listing_id": str(listing.pk),
        "ticker": listing.ticker,
        "provider_symbol": listing.provider_symbol,
        "exchange_mic": listing.exchange_mic,
        "currency": listing.currency,
        "region": listing.region,
        "is_active": listing.is_active,
        "valid_from": listing.valid_from.isoformat() if listing.valid_from else None,
        "valid_to": listing.valid_to.isoformat() if listing.valid_to else None,
        "security_id": str(listing.security_id),
        "security_type": listing.security.security_type,
        "company_id": str(listing.security.company_id),
        "company_cik": listing.security.company.cik,
        "company_name": listing.security.company.name,
        "company_country": listing.security.company.country,
    }
    valuation_text = format(valuation_value, "f")
    ledger_text = format(ledger_value, ".6f")
    target_price = {
        "owner_listing_id": str(listing.pk),
        "listing_id": str(listing.pk),
        "provider": "twelve_data",
        "subject": listing.provider_symbol,
        "exchange_mic": listing.exchange_mic,
        "session_date": analysis.run.target_date.isoformat(),
        "value": ledger_text,
        "ledger_value": ledger_text,
        "valuation_value": valuation_text,
        "valuation_source": "normalized_parquet_target_close",
        "native_price": valuation_text,
        "currency": listing.currency,
        "native_currency": listing.currency,
        "applied_fx_rate": None,
        "fx_conversion": False,
        "normalized_asset_id": normalized_asset["id"],
        "normalized_asset_sha256": normalized_asset["sha256"],
        "raw_asset_id": raw_asset["id"],
        "raw_asset_sha256": raw_asset["sha256"],
        "normalized_asset": normalized_asset,
        "raw_asset": raw_asset,
    }
    source_assets = [mapping_asset, normalized_asset, raw_asset]
    calculation: dict[str, object] = {
        "schema_version": 2,
        "method": LONG_V4_METHOD,
        "method_version": LONG_V4_VERSION,
        "config_hash": LONG_V4_EFFECTIVE_CONFIG_HASH,
        "research_status": LONG_V4_RESEARCH_STATUS,
        "path_years": 5,
        "target_date": analysis.run.target_date.isoformat(),
        "return_basis": "split_adjusted_price_return",
        "dividends_included": False,
        "base_currency": "USD",
        "fx_conversion": False,
        "insufficiency_code": None,
        "insufficiency_reason": "",
        "probability_semantics": {
            "status": "unavailable",
            "value": None,
            "reason": PROBABILITY_REASON,
        },
        "confidence_semantics": {
            "status": "not_estimated_uncalibrated",
            "value": 0.0,
            "schema": "zero_is_unavailable_sentinel",
        },
        "target": target,
        "target_price": target_price,
        "evidence_catalog": {
            "sec_mapping_authority": {"mapping_asset": mapping_asset},
            "raw_fcf_authority": [],
            "facts": [],
            "classifications": [],
            "prices": [deepcopy(target_price)],
        },
        "source_manifest": source_assets,
        "forecast_horizon": Prediction.Horizon.THREE_YEAR,
        "years": 3,
        "selected_view": {
            "horizon": Prediction.Horizon.THREE_YEAR,
            "year": 3,
            "cumulative_returns": {
                "bear": float(bear) if bear is not None else None,
                "base": float(base) if base is not None else None,
                "bull": float(bull) if bull is not None else None,
            },
        },
        "prediction_version": model_version,
        "price_subject": listing.provider_symbol,
        "evidence_grade": UniverseSnapshot.Grade.RESEARCH,
    }
    return _prediction(
        analysis,
        horizon=Prediction.Horizon.THREE_YEAR,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        price_provider="twelve_data",
        price_subject=listing.provider_symbol,
        price_at_prediction=ledger_value,
        bear=bear,
        base=base,
        bull=bull,
        version=model_version,
        source_assets=source_assets,
        method_version=LONG_V4_VERSION,
        config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH,
        calculation=calculation,
        confidence=Decimal("0.00"),
        confidence_status="not_estimated_uncalibrated",
        insufficiency_reason=PROBABILITY_REASON,
        data_cutoff=analysis.run.data_cutoff,
    )


def _asset_identity_payload(asset: DataAsset) -> dict[str, str]:
    return {
        "id": str(asset.pk),
        "provider": asset.provider,
        "kind": asset.kind,
        "subject": asset.subject,
        "relative_path": asset.relative_path,
        "sha256": asset.sha256,
        "retrieved_at": asset.retrieved_at.isoformat(),
        "available_at": asset.available_at.isoformat(),
    }


def _business_dates_after(start: date, count: int) -> list[date]:
    dates: list[date] = []
    current = start + timedelta(days=1)
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def _evaluation_time() -> datetime:
    return datetime(2026, 12, 31, 12, tzinfo=timezone.get_current_timezone())


def _v4_evaluation_time() -> datetime:
    return datetime(2030, 12, 31, 12, tzinfo=timezone.get_current_timezone())


def _register_price_asset(
    store: AssetStore,
    subject: str,
    available_at: datetime,
    dates: list[date],
    closes: list[float],
    *,
    provider: str = "synthetic",
    baseline_date: date | None = date(2026, 1, 2),
    baseline_close: Decimal | float = Decimal("100"),
) -> None:
    if baseline_date is not None and baseline_date not in dates:
        dates = [baseline_date, *dates]
        closes = [float(baseline_close), *closes]
    frame = pl.DataFrame(
        {
            "date": dates,
            "close": closes,
            "volume": [1_000_000] * len(closes),
        }
    )
    stored = store.write_frame(f"outcomes/{uuid4().hex}.parquet", frame)
    register_asset(
        provider=provider,
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )

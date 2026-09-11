"""Tests for `stanstock.research.outcome_refresh_validation`.

`verify_prediction_outcome` replays every expected candidate's outcome
through the exact same pure `resolve_outcome` the writing producer uses, so
this file's own fabricated fixtures are only for *adversarial* tamper
scenarios; every valid-path scenario runs the real `evaluate_prediction`
producer first and then verifies its real persisted output, per the
project's stated preference for real-producer coverage over hand-built rows.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import polars as pl
import pytest
from django.conf import settings
from django.db import DatabaseError, transaction

from base_outcomes import (
    PURE_INSERTION_DEPENDENCY_PATHS,
    PureInsertionViolationError,
    _require_pure_insertion,
    base_outcomes_available,
    base_research_outcomes,
    require_pure_insertion_since_base,
)
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.asof import PriceFrameSchemaError
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import DataAsset
from stanstock.research.models import Prediction, PredictionOutcome, Recommendation
from stanstock.research.outcome_refresh_validation import (
    OUTCOME_FIELDS,
    _verifier_price_loader,
    _VerifierLoadError,
    _VisitedAssets,
    verify_prediction_outcome,
)
from stanstock.research.outcomes import (
    HORIZON_SESSION_COUNTS,
    evaluate_prediction,
    resolve_outcome,
)
from test_research_outcomes import (
    _analysis,
    _business_dates_after,
    _prediction,
    _register_price_asset,
)

pytestmark = pytest.mark.django_db

PROVIDER = "synthetic"
BENCHMARK_SUBJECT = "SPY"


def _concrete_field_names(model: type) -> set[str]:
    return {
        field.attname
        for field in model._meta.get_fields()
        if getattr(field, "concrete", False) and not field.many_to_many
    }


def test_outcome_fields_plus_evaluated_at_cover_every_concrete_field() -> None:
    """Fails closed if a migration adds/removes a `PredictionOutcome` field
    without updating `OUTCOME_FIELDS` or `_check_evaluated_at`'s coverage."""
    assert _concrete_field_names(PredictionOutcome) == set(OUTCOME_FIELDS) | {"evaluated_at"}
    assert len(OUTCOME_FIELDS) == len(set(OUTCOME_FIELDS))


# ---------------------------------------------------------------------------
# Real-producer setup helper
# ---------------------------------------------------------------------------


def _mature_case(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    horizon: Prediction.Horizon = Prediction.Horizon.SHORT,
    recommendation: Recommendation = Recommendation.BUY,
    evidence_role: Prediction.EvidenceRole = Prediction.EvidenceRole.DECISION,
    bear: Decimal | None = Decimal("-0.10"),
    base: Decimal | None = Decimal("0.02"),
    bull: Decimal | None = Decimal("0.10"),
    session_count: int = 10,
    closes: list[float] | None = None,
    provider: str = PROVIDER,
    with_benchmark: bool = False,
    benchmark_closes: list[float] | None = None,
) -> tuple[Prediction, datetime, date, AssetStore]:
    """Build+evaluate one real prediction; return it plus what the verifier needs."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=horizon,
        recommendation=recommendation,
        evidence_role=evidence_role,
        bear=bear,
        base=base,
        bull=bull,
        price_provider=provider,
        price_subject=listing.ticker,
    )
    sessions = _business_dates_after(prediction.target_date, session_count)
    store = AssetStore(tmp_path)
    evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    _register_price_asset(
        store,
        listing.ticker,
        evaluation_time,
        sessions,
        closes or [101.0 + index for index in range(session_count)],
        provider=provider,
    )
    benchmark_subject = ""
    if with_benchmark:
        benchmark_subject = BENCHMARK_SUBJECT
        _register_price_asset(
            store,
            benchmark_subject,
            evaluation_time,
            sessions,
            benchmark_closes or [100.0 + index * 0.5 for index in range(session_count)],
            provider=provider,
            baseline_date=prediction.target_date,
            baseline_close=Decimal("100"),
        )
    evaluate_prediction(
        prediction,
        provider=provider,
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        benchmark_subject=benchmark_subject or None,
        store=store,
    )
    return prediction, evaluation_time, sessions[-1], store


def _verify(
    prediction: Prediction, *, provider: str = PROVIDER, benchmark_subject: str = ""
) -> tuple[Prediction, PredictionOutcome, datetime, date]:
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    return outcome


# ---------------------------------------------------------------------------
# Happy-path: real producer output verifies cleanly, both terminal statuses
# ---------------------------------------------------------------------------


def test_matured_decision_outcome_verifies_and_returns_asset_ref(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction, evaluation_time, target_date, _store = _mature_case(tmp_path, monkeypatch)
    outcome = PredictionOutcome.objects.get(prediction=prediction)

    result = verify_prediction_outcome(
        prediction,
        outcome,
        provider=PROVIDER,
        benchmark_subject="",
        evaluation_time=evaluation_time,
        parent_target_date=target_date,
    )

    assert result.summary["status"] == PredictionOutcome.Status.MATURED
    assert len(result.asset_refs) == 1
    assert str(tmp_path) not in repr(result.summary)


def test_matured_with_benchmark_verifies_and_returns_two_asset_refs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction, evaluation_time, target_date, _store = _mature_case(
        tmp_path, monkeypatch, with_benchmark=True
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)

    result = verify_prediction_outcome(
        prediction,
        outcome,
        provider=PROVIDER,
        benchmark_subject=BENCHMARK_SUBJECT,
        evaluation_time=evaluation_time,
        parent_target_date=target_date,
    )

    assert result.summary["status"] == PredictionOutcome.Status.MATURED
    assert {ref.subject for ref in result.asset_refs} == {
        prediction.listing.ticker,
        BENCHMARK_SUBJECT,
    }


def test_unresolved_advisory_all_null_verifies(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        bear=None,
        base=None,
        bull=None,
        price_provider=PROVIDER,
        price_subject=listing.ticker,
    )
    evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    evaluate_prediction(
        prediction,
        provider=PROVIDER,
        evaluation_date=prediction.target_date,
        evaluation_time=evaluation_time,
        store=AssetStore(tmp_path),
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    assert outcome.status == PredictionOutcome.Status.UNRESOLVED

    result = verify_prediction_outcome(
        prediction,
        outcome,
        provider=PROVIDER,
        benchmark_subject="",
        evaluation_time=evaluation_time,
        parent_target_date=prediction.target_date,
    )
    assert result.summary["status"] == PredictionOutcome.Status.UNRESOLVED
    assert result.asset_refs == ()


def test_zero_read_date_guard_never_touches_asset_store(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`evaluation_date > evaluated_at.date()` short-circuits before any read."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    prediction = _prediction(analysis, horizon=Prediction.Horizon.SHORT, price_provider=PROVIDER)
    evaluation_time = datetime(2026, 1, 1, 12, tzinfo=UTC)
    evaluate_prediction(
        prediction,
        provider=PROVIDER,
        evaluation_date=prediction.target_date,
        evaluation_time=evaluation_time,
        store=AssetStore(tmp_path),
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    assert outcome.status == PredictionOutcome.Status.UNRESOLVED

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("no asset read should occur for a date-guard rejection")

    monkeypatch.setattr(DataAsset.objects, "filter", _boom)
    result = verify_prediction_outcome(
        prediction,
        outcome,
        provider=PROVIDER,
        benchmark_subject="",
        evaluation_time=evaluation_time,
        parent_target_date=prediction.target_date,
    )
    assert result.summary["status"] == PredictionOutcome.Status.UNRESOLVED
    assert result.asset_refs == ()


# ---------------------------------------------------------------------------
# Physical-integrity adversarial cases (subject evidence tamper)
# ---------------------------------------------------------------------------


def test_missing_price_file_fails_closed_path_free(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction, evaluation_time, target_date, store = _mature_case(tmp_path, monkeypatch)
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    asset = DataAsset.objects.get(provider=PROVIDER, subject=prediction.listing.ticker)
    store.resolve(asset.relative_path).unlink()

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=target_date,
        )
    assert excinfo.value.reason_code == "evaluation_outcome_maturity_unreadable"
    assert str(tmp_path) not in str(excinfo.value)


def test_checksum_mismatch_fails_closed_path_free(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction, evaluation_time, target_date, store = _mature_case(tmp_path, monkeypatch)
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    asset = DataAsset.objects.get(provider=PROVIDER, subject=prediction.listing.ticker)
    store.resolve(asset.relative_path).write_bytes(b"not the original checksummed payload")

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=target_date,
        )
    assert excinfo.value.reason_code == "evaluation_outcome_maturity_unreadable"
    assert str(tmp_path) not in str(excinfo.value)


def test_corrupt_but_checksum_matching_parquet_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later vintage that is not valid Parquet at all, but whose bytes do
    match their own registered checksum (a corruption the checksum check
    alone cannot catch), must still fail the physical Parquet parse when it
    becomes the asset the verifier's own cutoff selects as latest."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    prediction = _prediction(analysis, horizon=Prediction.Horizon.SHORT, price_provider=PROVIDER)
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    _register_price_asset(
        store,
        listing.ticker,
        evaluation_time - timedelta(seconds=1),
        sessions,
        [101.0 + i for i in range(10)],
        provider=PROVIDER,
    )
    evaluate_prediction(
        prediction,
        provider=PROVIDER,
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        store=store,
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    assert outcome.status == PredictionOutcome.Status.MATURED

    garbage = b"this is not a parquet file"
    stored = store.write_bytes(f"outcomes/{uuid4().hex}.parquet", garbage)
    register_asset(
        provider=PROVIDER,
        kind="price_history",
        subject=listing.ticker,
        stored=stored,
        retrieved_at=evaluation_time,
        available_at=evaluation_time,
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=sessions[-1],
        )
    assert excinfo.value.reason_code == "evaluation_outcome_maturity_unreadable"


def test_benchmark_evidence_tamper_fails_closed_independently_of_subject(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tampering only the benchmark's file must fail even though the
    prediction's own subject evidence is untouched."""
    prediction, evaluation_time, target_date, store = _mature_case(
        tmp_path, monkeypatch, with_benchmark=True
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    benchmark_asset = DataAsset.objects.get(provider=PROVIDER, subject=BENCHMARK_SUBJECT)
    store.resolve(benchmark_asset.relative_path).write_bytes(b"tampered benchmark bytes")

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=PROVIDER,
            benchmark_subject=BENCHMARK_SUBJECT,
            evaluation_time=evaluation_time,
            parent_target_date=target_date,
        )
    assert excinfo.value.reason_code == "evaluation_outcome_maturity_unreadable"


def test_later_vintage_registered_after_evaluation_time_is_excluded(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newer, differently-priced vintage retrieved after `evaluation_time`
    must not be picked up: the verifier's own `AsOfData` cutoff (identical
    to the producer's) must still select the same eligible asset."""
    prediction, evaluation_time, target_date, store = _mature_case(tmp_path, monkeypatch)
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    _register_price_asset(
        store,
        prediction.listing.ticker,
        evaluation_time + timedelta(days=5),
        _business_dates_after(prediction.target_date, 10),
        [1234.0 for _ in range(10)],
        provider=PROVIDER,
    )

    result = verify_prediction_outcome(
        prediction,
        outcome,
        provider=PROVIDER,
        benchmark_subject="",
        evaluation_time=evaluation_time,
        parent_target_date=target_date,
    )
    assert result.summary["status"] == PredictionOutcome.Status.MATURED


# ---------------------------------------------------------------------------
# Provider/subject resolution
# ---------------------------------------------------------------------------


def test_blank_recorded_provider_falls_back_to_requested_provider(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    prediction = _prediction(analysis, horizon=Prediction.Horizon.SHORT, price_provider="")
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    _register_price_asset(
        store,
        listing.ticker,
        evaluation_time,
        sessions,
        [101.0 + i for i in range(10)],
        provider=PROVIDER,
    )
    evaluate_prediction(
        prediction,
        provider=PROVIDER,
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        store=store,
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    assert outcome.status == PredictionOutcome.Status.MATURED

    result = verify_prediction_outcome(
        prediction,
        outcome,
        provider=PROVIDER,
        benchmark_subject="",
        evaluation_time=evaluation_time,
        parent_target_date=sessions[-1],
    )
    assert result.summary["status"] == PredictionOutcome.Status.MATURED


def test_provider_conflict_fails_closed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    prediction, evaluation_time, target_date, _store = _mature_case(tmp_path, monkeypatch)
    outcome = PredictionOutcome.objects.get(prediction=prediction)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider="a-different-provider",
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=target_date,
        )
    assert excinfo.value.reason_code == "evaluation_provider_mismatch"


# ---------------------------------------------------------------------------
# Persisted-row tamper: field/metadata/timestamp mismatches
# ---------------------------------------------------------------------------


def test_persisted_resolution_tamper_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction, evaluation_time, target_date, _store = _mature_case(tmp_path, monkeypatch)
    PredictionOutcome.objects.filter(prediction=prediction).update(resolution="a fabricated claim")
    outcome = PredictionOutcome.objects.get(prediction=prediction)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=target_date,
        )
    assert excinfo.value.reason_code == "evaluation_outcome_replay_mismatch"


def test_persisted_metadata_subject_tamper_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction, evaluation_time, target_date, _store = _mature_case(tmp_path, monkeypatch)
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    tampered_metadata = dict(outcome.metadata)
    tampered_metadata["subject"] = "SOME-OTHER-TICKER"
    PredictionOutcome.objects.filter(prediction=prediction).update(metadata=tampered_metadata)
    outcome.refresh_from_db()

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=target_date,
        )
    assert excinfo.value.reason_code == "evaluation_outcome_replay_mismatch"


def test_persisted_actual_return_rounding_tamper_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction, evaluation_time, target_date, _store = _mature_case(tmp_path, monkeypatch)
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    PredictionOutcome.objects.filter(prediction=prediction).update(
        actual_return=outcome.actual_return + Decimal("0.0001")
    )
    outcome.refresh_from_db()

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=target_date,
        )
    assert excinfo.value.reason_code == "evaluation_outcome_replay_mismatch"


def test_terminal_outcome_evaluated_at_mismatch_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction, evaluation_time, target_date, _store = _mature_case(tmp_path, monkeypatch)
    PredictionOutcome.objects.filter(prediction=prediction).update(
        evaluated_at=evaluation_time - timedelta(days=1)
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=target_date,
        )
    assert excinfo.value.reason_code == "evaluation_outcome_not_bound_to_execution"


def test_unresolved_outcome_future_evaluated_at_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    prediction = _prediction(analysis, horizon=Prediction.Horizon.SHORT, price_provider=PROVIDER)
    evaluation_time = datetime(2026, 1, 1, 12, tzinfo=UTC)
    evaluate_prediction(
        prediction,
        provider=PROVIDER,
        evaluation_date=prediction.target_date,
        evaluation_time=evaluation_time,
        store=AssetStore(tmp_path),
    )
    PredictionOutcome.objects.filter(prediction=prediction).update(
        evaluated_at=evaluation_time + timedelta(days=1)
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=prediction.target_date,
        )
    assert excinfo.value.reason_code == "evaluation_outcome_not_bound_to_execution"


def test_terminal_maturity_date_disagreement_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction, evaluation_time, target_date, _store = _mature_case(tmp_path, monkeypatch)
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    PredictionOutcome.objects.filter(prediction=prediction).update(
        evaluation_date=outcome.evaluation_date - timedelta(days=1)
    )
    outcome.refresh_from_db()

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=target_date,
        )
    assert excinfo.value.reason_code == "evaluation_outcome_maturity_date_invalid"


def test_withheld_advisory_forced_terminal_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        bear=None,
        base=None,
        bull=None,
        price_provider=PROVIDER,
    )
    evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    evaluate_prediction(
        prediction,
        provider=PROVIDER,
        evaluation_date=prediction.target_date,
        evaluation_time=evaluation_time,
        store=AssetStore(tmp_path),
    )
    PredictionOutcome.objects.filter(prediction=prediction).update(
        status=PredictionOutcome.Status.MATURED,
        actual_return=Decimal("0.01"),
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=prediction.target_date,
        )
    assert excinfo.value.reason_code == "evaluation_outcome_replay_mismatch"


def test_non_decision_matured_success_is_rejected_by_db_trigger(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evidence-role/`success` pairing is enforced at the DB layer for
    `matured` rows (both SQLite and PostgreSQL
    `research_predictionoutcome_role_*` triggers -- they fire `WHEN
    NEW.status = 'matured'`), so an advisory prediction's outcome can never
    even be persisted with a non-null `success` while `matured`. See
    `test_advisory_corporate_event_success_label_is_rejected_independently`
    for why the validator still enforces this itself rather than relying
    solely on these DB mechanisms."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        price_provider=PROVIDER,
    )
    sessions = _business_dates_after(prediction.target_date, 126)
    store = AssetStore(tmp_path)
    evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    _register_price_asset(
        store,
        listing.ticker,
        evaluation_time,
        sessions,
        [101.0 + i for i in range(126)],
        provider=PROVIDER,
    )
    evaluate_prediction(
        prediction,
        provider=PROVIDER,
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        store=store,
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    assert outcome.status == PredictionOutcome.Status.MATURED
    assert outcome.success is None

    with pytest.raises(DatabaseError, match="evidence role"):
        PredictionOutcome.objects.filter(prediction=prediction).update(success=True)


def test_advisory_corporate_event_success_label_is_rejected_independently(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The DB role triggers only fire `WHEN NEW.status = 'matured'`, so a
    `corporate_event`/`unresolved` terminal row falls entirely outside
    their `WHEN` clause. Separately, the `outcome_corporate_event_nulls`
    and `outcome_unresolved_nulls` CHECK constraints already forbid
    persisting a non-`matured` row with a non-null `success` at all (proven
    below), and the producer itself never constructs one -- so this exact
    combination cannot currently reach the database through any real
    write path. The validator's own independent invariant is deliberate
    defense in depth against a resolver defect that computed a wrong
    `success` value: it must reject a persisted row shaped this way on its
    own terms rather than relying on those other two mechanisms, so a
    fresh replay agreeing with the same wrong value could never
    self-authenticate it. The adversarial row below is therefore built
    in memory only (never saved), since the DB itself has no way to hold
    it."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        price_provider=PROVIDER,
    )
    sessions = _business_dates_after(prediction.target_date, 126)
    store = AssetStore(tmp_path)
    evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    _register_price_asset(
        store,
        listing.ticker,
        evaluation_time,
        sessions,
        [101.0 + i for i in range(126)],
        provider=PROVIDER,
    )
    evaluate_prediction(
        prediction,
        provider=PROVIDER,
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        store=store,
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    assert outcome.status == PredictionOutcome.Status.MATURED

    # Confirms the DB itself never lets this combination reach the table
    # through an ordinary write, regardless of which status is used.
    with pytest.raises(DatabaseError, match="outcome_corporate_event_nulls"), transaction.atomic():
        PredictionOutcome.objects.filter(prediction=prediction).update(
            status=PredictionOutcome.Status.CORPORATE_EVENT,
            success=True,
        )

    adversarial = PredictionOutcome(
        prediction=prediction,
        evaluated_at=outcome.evaluated_at,
        evaluation_date=outcome.evaluation_date,
        status=PredictionOutcome.Status.CORPORATE_EVENT,
        success=True,
        metadata={},
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            adversarial,
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=sessions[-1],
        )
    assert excinfo.value.reason_code == "evaluation_outcome_replay_mismatch"
    assert (
        str(excinfo.value)
        == "A non-decision prediction's persisted outcome carries a non-null success label"
    )


# ---------------------------------------------------------------------------
# Legacy horizons and independent reissues
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "horizon,evidence_role",
    [
        (Prediction.Horizon.MEDIUM, Prediction.EvidenceRole.DECISION),
        (Prediction.Horizon.LONG, Prediction.EvidenceRole.DECISION),
        (Prediction.Horizon.SIX_MONTH, Prediction.EvidenceRole.ADVISORY),
        (Prediction.Horizon.THREE_YEAR, Prediction.EvidenceRole.ADVISORY),
    ],
)
def test_legacy_and_current_horizons_all_verify(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    horizon: Prediction.Horizon,
    evidence_role: Prediction.EvidenceRole,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    prediction = _prediction(
        analysis, horizon=horizon, evidence_role=evidence_role, price_provider=PROVIDER
    )
    session_count = 5
    sessions = _business_dates_after(prediction.target_date, session_count)
    store = AssetStore(tmp_path)
    evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    _register_price_asset(
        store,
        listing.ticker,
        evaluation_time,
        sessions,
        [101.0 + i for i in range(session_count)],
        provider=PROVIDER,
    )
    evaluate_prediction(
        prediction,
        provider=PROVIDER,
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        store=store,
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)
    assert outcome.status == PredictionOutcome.Status.UNRESOLVED

    result = verify_prediction_outcome(
        prediction,
        outcome,
        provider=PROVIDER,
        benchmark_subject="",
        evaluation_time=evaluation_time,
        parent_target_date=sessions[-1],
    )
    assert result.summary["status"] == PredictionOutcome.Status.UNRESOLVED


def test_two_independent_reissues_verify_independently(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two immutable reissues for the same listing/target verify on their
    own evidence; tampering one must never affect the other."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    store = AssetStore(tmp_path)
    evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    sessions = _business_dates_after(analysis.run.target_date, 10)
    _register_price_asset(
        store,
        listing.ticker,
        evaluation_time,
        sessions,
        [101.0 + i for i in range(10)],
        provider=PROVIDER,
    )

    first = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, price_provider=PROVIDER, version="reissue-1"
    )
    second = _prediction(
        analysis, horizon=Prediction.Horizon.SHORT, price_provider=PROVIDER, version="reissue-2"
    )
    for reissue in (first, second):
        evaluate_prediction(
            reissue,
            provider=PROVIDER,
            evaluation_date=sessions[-1],
            evaluation_time=evaluation_time,
            store=store,
        )
    second_outcome = PredictionOutcome.objects.get(prediction=second)
    PredictionOutcome.objects.filter(prediction=first).update(resolution="tampered reissue")

    with pytest.raises(RefreshVerificationError):
        verify_prediction_outcome(
            first,
            PredictionOutcome.objects.get(prediction=first),
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=sessions[-1],
        )
    result = verify_prediction_outcome(
        second,
        second_outcome,
        provider=PROVIDER,
        benchmark_subject="",
        evaluation_time=evaluation_time,
        parent_target_date=sessions[-1],
    )
    assert result.summary["status"] == PredictionOutcome.Status.MATURED


# ---------------------------------------------------------------------------
# Raw-closure declaration
#
# `DataAsset` rows are immutable, so a raw-closure declaration must be baked
# into the *original* price asset's metadata at registration time rather
# than tampered in afterwards; each case below builds its own price asset
# directly (mirroring `_register_price_asset`) instead of using
# `_mature_case`, so the desired metadata can be supplied up front.
# ---------------------------------------------------------------------------


def _register_price_asset_with_metadata(
    store: AssetStore,
    subject: str,
    available_at: datetime,
    dates: list[date],
    closes: list[float],
    *,
    provider: str,
    metadata: dict[str, object],
) -> DataAsset:
    frame = pl.DataFrame({"date": dates, "close": closes, "volume": [1_000_000] * len(closes)})
    stored = store.write_frame(f"outcomes/{uuid4().hex}.parquet", frame)
    return register_asset(
        provider=provider,
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
        metadata=metadata,
    )


def test_raw_closure_undeclared_is_a_satisfied_skip(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction, evaluation_time, target_date, _store = _mature_case(tmp_path, monkeypatch)
    outcome = PredictionOutcome.objects.get(prediction=prediction)

    result = verify_prediction_outcome(
        prediction,
        outcome,
        provider=PROVIDER,
        benchmark_subject="",
        evaluation_time=evaluation_time,
        parent_target_date=target_date,
    )
    assert len(result.asset_refs) == 1


def test_raw_closure_partial_declaration_fails_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    prediction = _prediction(analysis, horizon=Prediction.Horizon.SHORT, price_provider=PROVIDER)
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    _register_price_asset_with_metadata(
        store,
        listing.ticker,
        evaluation_time,
        sessions,
        [101.0 + i for i in range(10)],
        provider=PROVIDER,
        metadata={"raw_asset_id": str(uuid4())},
    )
    evaluate_prediction(
        prediction,
        provider=PROVIDER,
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        store=store,
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=sessions[-1],
        )
    assert excinfo.value.reason_code == "price_asset_raw_link_missing"


def test_raw_closure_broken_link_fails_closed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    prediction = _prediction(analysis, horizon=Prediction.Horizon.SHORT, price_provider=PROVIDER)
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    _register_price_asset_with_metadata(
        store,
        listing.ticker,
        evaluation_time,
        sessions,
        [101.0 + i for i in range(10)],
        provider=PROVIDER,
        metadata={"raw_asset_id": str(uuid4()), "raw_sha256": "a" * 64},
    )
    evaluate_prediction(
        prediction,
        provider=PROVIDER,
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        store=store,
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_prediction_outcome(
            prediction,
            outcome,
            provider=PROVIDER,
            benchmark_subject="",
            evaluation_time=evaluation_time,
            parent_target_date=sessions[-1],
        )
    assert excinfo.value.reason_code == "price_asset_raw_asset_missing"


def test_raw_closure_declared_and_valid_adds_two_refs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    prediction = _prediction(analysis, horizon=Prediction.Horizon.SHORT, price_provider=PROVIDER)
    sessions = _business_dates_after(prediction.target_date, 10)
    store = AssetStore(tmp_path)
    evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    raw_bytes = b"raw provider payload"
    stored_raw = store.write_bytes(f"outcomes/{uuid4().hex}-raw.json", raw_bytes)
    raw_asset = register_asset(
        provider=PROVIDER,
        kind="raw_price_history",
        subject=listing.ticker,
        stored=stored_raw,
        retrieved_at=evaluation_time,
        available_at=evaluation_time,
    )
    asset = _register_price_asset_with_metadata(
        store,
        listing.ticker,
        evaluation_time,
        sessions,
        [101.0 + i for i in range(10)],
        provider=PROVIDER,
        metadata={"raw_asset_id": str(raw_asset.pk), "raw_sha256": raw_asset.sha256},
    )
    evaluate_prediction(
        prediction,
        provider=PROVIDER,
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        store=store,
    )
    outcome = PredictionOutcome.objects.get(prediction=prediction)

    result = verify_prediction_outcome(
        prediction,
        outcome,
        provider=PROVIDER,
        benchmark_subject="",
        evaluation_time=evaluation_time,
        parent_target_date=sessions[-1],
    )
    assert len(result.asset_refs) == 2
    assert {ref.id for ref in result.asset_refs} == {asset.id, raw_asset.id}


# ---------------------------------------------------------------------------
# Private verifier-loader unit coverage for defense-in-depth branches that
# are structurally unreachable through the public `verify_prediction_outcome`
# entrypoint (its own `AsOfData.latest_asset` call always requests
# `kind="price_history"` and a `through_date` already bounded by
# `resolve_outcome`'s own date guard, so a wrong-kind/late-cutoff asset can
# never actually reach `_verifier_price_loader` in practice).
# ---------------------------------------------------------------------------


def test_verifier_price_loader_rejects_through_date_after_cutoff(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    evaluation_time = datetime(2026, 1, 5, tzinfo=UTC)
    store = AssetStore(tmp_path)
    _register_price_asset(
        store,
        listing.ticker,
        evaluation_time,
        [date(2026, 1, 6)],
        [101.0],
        provider=PROVIDER,
        baseline_date=None,
    )
    with pytest.raises(_VerifierLoadError) as excinfo:
        _verifier_price_loader(
            provider=PROVIDER,
            subject=listing.ticker,
            through_date=date(2026, 1, 10),
            evaluation_time=evaluation_time,
            frame_cache={},
            visited=_VisitedAssets(),
        )
    assert excinfo.value.reason_code == "evaluation_outcome_maturity_unreadable"


# ---------------------------------------------------------------------------
# Deterministic literal vectors (contract H): call the pure `resolve_outcome`
# seam directly with fixed, hand-computed inputs/outputs for every major
# branch. These never touch `AssetStore`/`AsOfData` and always run (no
# skip), independent of the dynamic base-vs-head comparison below.
# ---------------------------------------------------------------------------

_TARGET = date(2026, 1, 2)
_EVALUATED_AT = datetime(2026, 6, 1, tzinfo=UTC)
_SUBJECT = "TEST"


def _session_frame(
    *,
    include_baseline: bool = True,
    baseline_close: float = 100.0,
    after_count: int = 10,
    final_close: float = 100.0,
    extra_rows: list[tuple[date, float]] | None = None,
) -> pl.DataFrame:
    """`after_count` sessions strictly after `_TARGET`, filler close 100.0
    except the last (sorted) row, which carries `final_close`."""
    dates: list[date] = [_TARGET] if include_baseline else []
    closes: list[float] = [baseline_close] if include_baseline else []
    for day in range(1, after_count):
        dates.append(_TARGET + timedelta(days=day))
        closes.append(100.0)
    if after_count:
        dates.append(_TARGET + timedelta(days=after_count))
        closes.append(final_close)
    for extra_date, extra_close in extra_rows or ():
        dates.append(extra_date)
        closes.append(extra_close)
    return pl.DataFrame({"date": dates, "close": closes, "volume": [1] * len(dates)})


def _loader(
    frames: dict[str, pl.DataFrame] | None = None,
    raises: dict[str, Exception] | None = None,
):
    frames = frames or {}
    raises = raises or {}

    def _load(subject: str, through_date: date) -> pl.DataFrame:
        del through_date
        if subject in raises:
            raise raises[subject]
        return frames[subject]

    return _load


def _literal_prediction(analysis, **overrides):
    overrides.setdefault("target_date", _TARGET)
    overrides.setdefault("price_subject", _SUBJECT)
    return _prediction(analysis, **overrides)


def test_after_evaluated_at_is_unresolved() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(analysis, horizon=Prediction.Horizon.SHORT)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_EVALUATED_AT.date() + timedelta(days=1),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(),
    )
    assert result.status == PredictionOutcome.Status.UNRESOLVED
    assert result.resolution == "Evaluation date is after actual evaluation time"


def test_before_target_date_is_unresolved() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(analysis, horizon=Prediction.Horizon.SHORT)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET - timedelta(days=1),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(),
    )
    assert result.status == PredictionOutcome.Status.UNRESOLVED
    assert result.resolution == "Evaluation date is before prediction target date"


def test_withheld_advisory_is_unresolved_before_any_read() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(
        analysis,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        bear=None,
        base=None,
        bull=None,
    )
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET,
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(),  # never called: empty dict would KeyError otherwise
    )
    assert result.status == PredictionOutcome.Status.UNRESOLVED
    assert result.resolution == "Withheld forecast has no scenario to evaluate"


@pytest.mark.parametrize(
    "exc",
    [
        DataAsset.DoesNotExist("no matching asset"),
        PriceFrameSchemaError("bad schema"),
        ValueError("boom"),
    ],
    ids=["does_not_exist", "schema_error", "value_error"],
)
def test_producer_domain_loader_exceptions_are_unresolved(exc: Exception) -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(analysis, horizon=Prediction.Horizon.SHORT)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=10),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(raises={_SUBJECT: exc}),
    )
    assert result.status == PredictionOutcome.Status.UNRESOLVED
    assert result.resolution.startswith("Unable to load evaluation price history:")


def test_duplicate_usable_session_dates_is_unresolved() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(analysis, horizon=Prediction.Horizon.SHORT)
    frame = _session_frame(
        extra_rows=[(_TARGET + timedelta(days=1), 101.0)]
    )  # day+1 already present -> duplicate
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=10),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(frames={_SUBJECT: frame}),
    )
    assert result.status == PredictionOutcome.Status.UNRESOLVED
    assert result.resolution == "Duplicate usable price session dates in evaluation data"


def test_insufficient_sessions_is_unresolved() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(analysis, horizon=Prediction.Horizon.SHORT)
    frame = _session_frame(after_count=3, final_close=101.0)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=3),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(frames={_SUBJECT: frame}),
    )
    assert result.status == PredictionOutcome.Status.UNRESOLVED
    assert result.resolution == (
        f"Insufficient observed sessions after target date: 3/10 through "
        f"{(_TARGET + timedelta(days=3)).isoformat()}"
    )


def test_non_positive_prediction_price_is_unresolved() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(analysis, horizon=Prediction.Horizon.SHORT)
    prediction.price_at_prediction = Decimal("0")  # in-memory only; never persisted
    frame = _session_frame(final_close=110.0)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=10),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(frames={_SUBJECT: frame}),
    )
    assert result.status == PredictionOutcome.Status.UNRESOLVED
    assert result.resolution == "Prediction price is not positive"


def test_missing_baseline_close_is_unresolved() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(analysis, horizon=Prediction.Horizon.SHORT)
    frame = _session_frame(include_baseline=False, final_close=110.0)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=10),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(frames={_SUBJECT: frame}),
    )
    assert result.status == PredictionOutcome.Status.UNRESOLVED
    assert result.resolution == (
        "Evaluation price history has no baseline close at or before target date"
    )


def test_target_date_price_changed_is_corporate_event() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(
        analysis, horizon=Prediction.Horizon.SHORT, price_at_prediction=Decimal("100")
    )
    frame = _session_frame(baseline_close=150.0, final_close=110.0)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=10),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(frames={_SUBJECT: frame}),
    )
    assert result.status == PredictionOutcome.Status.CORPORATE_EVENT
    assert result.resolution == "Target-date price changed in the evaluation vintage"


@pytest.mark.parametrize(
    "recommendation,final_close,expected_success",
    [
        (Recommendation.BUY, 110.0, True),
        (Recommendation.BUY, 90.0, False),
        (Recommendation.AVOID, 100.0, True),
        (Recommendation.AVOID, 110.0, False),
    ],
    ids=["buy_positive", "buy_negative", "avoid_nonpositive", "avoid_positive"],
)
def test_decision_success_labels(
    recommendation: Recommendation, final_close: float, expected_success: bool
) -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=recommendation,
        price_at_prediction=Decimal("100"),
    )
    frame = _session_frame(final_close=final_close)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=10),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(frames={_SUBJECT: frame}),
    )
    assert result.status == PredictionOutcome.Status.MATURED
    assert result.success is expected_success
    assert result.metadata == {
        "provider": PROVIDER,
        "subject": _SUBJECT,
        "horizon_sessions": HORIZON_SESSION_COUNTS[Prediction.Horizon.SHORT.value],
        "evidence_role": Prediction.EvidenceRole.DECISION,
        "evaluation_close": final_close,
        "benchmark_subject": "",
        "benchmark_resolution": "",
        "success_semantics": (
            "BUY succeeds when actual return is positive"
            if recommendation == Recommendation.BUY
            else "AVOID succeeds when actual return is non-positive"
        ),
    }


def test_hold_without_stored_interval_is_unresolved() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.HOLD,
        bear=None,
        base=None,
        bull=None,
        price_at_prediction=Decimal("100"),
    )
    frame = _session_frame(final_close=100.0)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=10),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(frames={_SUBJECT: frame}),
    )
    assert result.status == PredictionOutcome.Status.UNRESOLVED
    assert result.resolution == "HOLD success requires non-null stored bear and bull returns"


@pytest.mark.parametrize(
    "final_close,expected_success",
    [(105.0, True), (150.0, False)],
    ids=["hold_in_range", "hold_out_of_range"],
)
def test_hold_interval_success_labels(final_close: float, expected_success: bool) -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.HOLD,
        bear=Decimal("-0.10"),
        base=Decimal("0.02"),
        bull=Decimal("0.10"),
        price_at_prediction=Decimal("100"),
    )
    frame = _session_frame(final_close=final_close)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=10),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(frames={_SUBJECT: frame}),
    )
    assert result.status == PredictionOutcome.Status.MATURED
    assert result.success is expected_success


def test_advisory_matured_has_no_success_label_but_computes_error() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(
        analysis,
        horizon=Prediction.Horizon.SIX_MONTH,
        evidence_role=Prediction.EvidenceRole.ADVISORY,
        bear=Decimal("-0.10"),
        base=Decimal("0.02"),
        bull=Decimal("0.10"),
        price_at_prediction=Decimal("100"),
    )
    frame = _session_frame(after_count=126, final_close=110.0)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=126),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(frames={_SUBJECT: frame}),
    )
    assert result.status == PredictionOutcome.Status.MATURED
    assert result.success is None
    assert result.direction_correct is True
    assert result.error == Decimal("0.08")
    assert result.interval_covered is False


def test_benchmark_return_is_computed_when_resolvable() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
        price_at_prediction=Decimal("100"),
    )
    subject_frame = _session_frame(final_close=110.0)
    benchmark_frame = _session_frame(baseline_close=100.0, final_close=105.0)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=10),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=BENCHMARK_SUBJECT,
        price_loader=_loader(frames={_SUBJECT: subject_frame, BENCHMARK_SUBJECT: benchmark_frame}),
    )
    assert result.status == PredictionOutcome.Status.MATURED
    assert result.benchmark_return == Decimal("0.05")
    assert result.metadata["benchmark_subject"] == BENCHMARK_SUBJECT
    assert result.metadata["benchmark_resolution"] == (
        f"Benchmark return uses {_TARGET.isoformat()} to "
        f"{(_TARGET + timedelta(days=10)).isoformat()} closes"
    )


def test_benchmark_loader_failure_leaves_benchmark_return_null() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
        price_at_prediction=Decimal("100"),
    )
    subject_frame = _session_frame(final_close=110.0)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=10),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=BENCHMARK_SUBJECT,
        price_loader=_loader(
            frames={_SUBJECT: subject_frame},
            raises={BENCHMARK_SUBJECT: ValueError("no benchmark evidence")},
        ),
    )
    assert result.status == PredictionOutcome.Status.MATURED
    assert result.benchmark_return is None
    assert result.metadata["benchmark_resolution"] == (
        "Benchmark unavailable: no benchmark evidence"
    )


def test_benchmark_missing_target_close_leaves_benchmark_return_null() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
        price_at_prediction=Decimal("100"),
    )
    subject_frame = _session_frame(final_close=110.0)
    benchmark_frame = _session_frame(include_baseline=False, final_close=105.0)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=10),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=BENCHMARK_SUBJECT,
        price_loader=_loader(frames={_SUBJECT: subject_frame, BENCHMARK_SUBJECT: benchmark_frame}),
    )
    assert result.status == PredictionOutcome.Status.MATURED
    assert result.benchmark_return is None
    assert result.metadata["benchmark_resolution"] == (
        "Benchmark has no close at or before target date"
    )


def test_actual_return_rounds_to_four_decimal_places() -> None:
    _listing, analysis = _analysis()
    prediction = _literal_prediction(
        analysis,
        horizon=Prediction.Horizon.SHORT,
        recommendation=Recommendation.BUY,
        price_at_prediction=Decimal("100"),
    )
    frame = _session_frame(final_close=112.3456)
    result = resolve_outcome(
        prediction,
        provider=PROVIDER,
        evaluation_date=_TARGET + timedelta(days=10),
        evaluated_at=_EVALUATED_AT,
        benchmark_subject=None,
        price_loader=_loader(frames={_SUBJECT: frame}),
    )
    assert result.actual_return == Decimal("0.1235")
    assert result.error == result.signed_error == Decimal("0.1035")


# ---------------------------------------------------------------------------
# Dynamic base-vs-head differential (contract H): the SAME real-producer
# construction, once through base `research.outcomes` and once through
# head, must agree on every persisted field. Skips (does not fail) when the
# base object is genuinely unavailable; never falls back to comparing head
# against itself.
# ---------------------------------------------------------------------------


def _differential_scenarios() -> list[dict[str, object]]:
    return [
        {
            "id": "decision_buy_matured",
            "horizon": Prediction.Horizon.SHORT,
            "recommendation": Recommendation.BUY,
            "evidence_role": Prediction.EvidenceRole.DECISION,
            "final_close": 111.0,
            "with_benchmark": True,
        },
        {
            "id": "decision_hold_unresolved",
            "horizon": Prediction.Horizon.SHORT,
            "recommendation": Recommendation.HOLD,
            "evidence_role": Prediction.EvidenceRole.DECISION,
            "final_close": None,  # too few sessions -> unresolved
            "with_benchmark": False,
        },
        {
            "id": "advisory_matured",
            "horizon": Prediction.Horizon.SIX_MONTH,
            "recommendation": Recommendation.HOLD,
            "evidence_role": Prediction.EvidenceRole.ADVISORY,
            "final_close": 95.0,
            "with_benchmark": False,
        },
    ]


@pytest.mark.skipif(
    not base_outcomes_available(),
    reason="base revision is not in the local git object database",
)
@pytest.mark.parametrize(
    "scenario", _differential_scenarios(), ids=[s["id"] for s in _differential_scenarios()]
)
def test_base_and_head_evaluate_prediction_agree(
    tmp_path, monkeypatch: pytest.MonkeyPatch, scenario: dict[str, object]
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    listing, analysis = _analysis()
    store = AssetStore(tmp_path)
    evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    session_count = HORIZON_SESSION_COUNTS[scenario["horizon"].value]
    available_sessions = session_count if scenario["final_close"] is not None else 3
    closes = [100.0 + i for i in range(available_sessions)]
    if scenario["final_close"] is not None:
        closes[-1] = scenario["final_close"]
    sessions = _business_dates_after(analysis.run.target_date, available_sessions)
    _register_price_asset(store, listing.ticker, evaluation_time, sessions, closes)
    benchmark_subject = ""
    if scenario["with_benchmark"]:
        benchmark_subject = BENCHMARK_SUBJECT
        _register_price_asset(
            store,
            benchmark_subject,
            evaluation_time,
            sessions,
            [100.0 + i * 0.5 for i in range(available_sessions)],
            baseline_date=analysis.run.target_date,
        )

    def _make_prediction(version: str) -> Prediction:
        return _prediction(
            analysis,
            horizon=scenario["horizon"],
            recommendation=scenario["recommendation"],
            evidence_role=scenario["evidence_role"],
            price_provider=PROVIDER,
            version=version,
        )

    base_prediction = _make_prediction("differential-base")
    head_prediction = _make_prediction("differential-head")

    with base_research_outcomes() as base:
        base.outcomes.evaluate_prediction(
            base_prediction,
            provider=PROVIDER,
            evaluation_date=sessions[-1],
            evaluation_time=evaluation_time,
            benchmark_subject=benchmark_subject or None,
            store=store,
        )
    evaluate_prediction(
        head_prediction,
        provider=PROVIDER,
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        benchmark_subject=benchmark_subject or None,
        store=store,
    )

    base_outcome = PredictionOutcome.objects.get(prediction=base_prediction)
    head_outcome = PredictionOutcome.objects.get(prediction=head_prediction)
    for field in OUTCOME_FIELDS:
        if field == "prediction_id":
            continue
        assert getattr(base_outcome, field) == getattr(head_outcome, field), field
    assert base_outcome.evaluated_at == head_outcome.evaluated_at


@pytest.mark.skipif(
    not base_outcomes_available(),
    reason="base revision is not in the local git object database",
)
def test_guard_short_circuit_never_constructs_asset_store(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real critic-caught divergence: `AsOfData`/`AssetStore()` construction
    can create `STANSTOCK_DATA_DIR` on the filesystem, so it must happen
    lazily, only on the first *actual* price-frame request. Base's original
    `evaluate_prediction` checked its date/withheld-advisory guards before
    ever constructing `AsOfData`; the `resolve_outcome` extraction must
    preserve that ordering byte-for-byte rather than constructing `AsOfData`
    up front regardless of whether a guard will short-circuit first. This
    is proven here with a `STANSTOCK_DATA_DIR` that cannot be created (its
    parent path segment is a plain file): both base and head must persist
    the identical unresolved outcome without ever touching the filesystem,
    while a genuine non-guard path (below) still requires -- and gets -- a
    working loader against the same uncreated-until-needed directory."""
    blocker = tmp_path / "blocker-file"
    blocker.write_text("not a directory")
    uncreatable_data_dir = blocker / "unreachable"
    monkeypatch.setattr(settings, "DATA_DIR", uncreatable_data_dir)

    _listing, analysis = _analysis()
    evaluation_time = datetime(2026, 1, 1, 12, tzinfo=UTC)
    future_evaluation_date = date(2026, 6, 1)  # after evaluation_time: first guard

    def _make_prediction(version: str) -> Prediction:
        return _prediction(
            analysis,
            horizon=Prediction.Horizon.SHORT,
            price_provider=PROVIDER,
            version=version,
        )

    base_prediction = _make_prediction("guard-base")
    head_prediction = _make_prediction("guard-head")

    with base_research_outcomes() as base:
        base.outcomes.evaluate_prediction(
            base_prediction,
            provider=PROVIDER,
            evaluation_date=future_evaluation_date,
            evaluation_time=evaluation_time,
            store=None,
        )
    evaluate_prediction(
        head_prediction,
        provider=PROVIDER,
        evaluation_date=future_evaluation_date,
        evaluation_time=evaluation_time,
        store=None,
    )

    assert not uncreatable_data_dir.exists()
    base_outcome = PredictionOutcome.objects.get(prediction=base_prediction)
    head_outcome = PredictionOutcome.objects.get(prediction=head_prediction)
    assert head_outcome.status == PredictionOutcome.Status.UNRESOLVED
    assert head_outcome.resolution == "Evaluation date is after actual evaluation time"
    for field in OUTCOME_FIELDS:
        if field == "prediction_id":
            continue
        assert getattr(base_outcome, field) == getattr(head_outcome, field), field
    assert base_outcome.evaluated_at == head_outcome.evaluated_at

    # A genuine non-guard path still needs a real loader: pointing
    # `DATA_DIR` at a writable directory and letting a matured prediction
    # actually resolve through `store=None` proves the laziness above does
    # not silently break ordinary construction/usage.
    writable_data_dir = tmp_path / "writable"
    monkeypatch.setattr(settings, "DATA_DIR", writable_data_dir)
    matured_prediction = _make_prediction("guard-nonguard-path")
    mature_evaluation_time = datetime(2026, 12, 31, 12, tzinfo=UTC)
    session_count = HORIZON_SESSION_COUNTS[Prediction.Horizon.SHORT.value]
    sessions = _business_dates_after(analysis.run.target_date, session_count)
    closes = [100.0 + i for i in range(session_count)]
    _register_price_asset(
        AssetStore(writable_data_dir),
        matured_prediction.listing.ticker,
        mature_evaluation_time,
        sessions,
        closes,
    )

    evaluate_prediction(
        matured_prediction,
        provider=PROVIDER,
        evaluation_date=sessions[-1],
        evaluation_time=mature_evaluation_time,
        store=None,
    )

    matured_outcome = PredictionOutcome.objects.get(prediction=matured_prediction)
    assert matured_outcome.status == PredictionOutcome.Status.MATURED
    assert writable_data_dir.exists()


def test_dependency_modules_are_pure_insertions_since_base() -> None:
    """Mechanically proves the assumption `base_outcomes.py`'s docstring
    states in prose: every top-level statement the four dependency modules
    had in base -- compared by full AST semantics, not merely a name --
    still has a semantically identical statement in head in the same
    relative order, and every new head-only statement is an explicitly
    allowlisted, name-bound addition. Binding only `research.outcomes` from
    base -- while the *live* head versions of its four imports remain in
    `sys.modules` -- therefore reproduces exactly base's own behavior."""
    if not base_outcomes_available():
        pytest.skip("base revision is not in the local git object database")

    for relative_path in PURE_INSERTION_DEPENDENCY_PATHS:
        require_pure_insertion_since_base(relative_path)


def test_pure_insertion_proof_rejects_an_edit_inside_an_existing_function() -> None:
    """A line-based diff cannot see this: inserting a line *inside* an
    existing function's body, with no surrounding line deleted, is only
    ever reported as a bare `insert` opcode by `difflib.SequenceMatcher` --
    the exact gap that motivated replacing it with the AST-based proof
    above. This exercises that proof directly against tiny synthetic
    sources, independent of any real repository file."""
    base_source = "def helper(x):\n    return x + 1\n\n\nclass Thing:\n    value = 1\n"
    mutated_body_source = (
        "def helper(x):\n"
        "    x = x  # a change smuggled into the existing body\n"
        "    return x + 1\n"
        "\n"
        "\n"
        "class Thing:\n"
        "    value = 1\n"
    )
    pure_addition_source = base_source + "\n\ndef newly_added(y):\n    return y\n"

    with pytest.raises(PureInsertionViolationError, match="helper"):
        _require_pure_insertion(base_source, mutated_body_source, "synthetic")

    # A genuinely new top-level binding is only accepted when explicitly
    # allowlisted -- never for free.
    with pytest.raises(PureInsertionViolationError, match="newly_added"):
        _require_pure_insertion(base_source, pure_addition_source, "synthetic")
    _require_pure_insertion(
        base_source,
        pure_addition_source,
        "synthetic",
        allowed_additions=frozenset({"newly_added"}),
    )


def test_pure_insertion_proof_rejects_a_removed_binding() -> None:
    base_source = "def kept():\n    return 1\n\n\ndef removed():\n    return 2\n"
    head_source = "def kept():\n    return 1\n"

    with pytest.raises(PureInsertionViolationError, match="removed"):
        _require_pure_insertion(base_source, head_source, "synthetic")


def test_pure_insertion_proof_rejects_duplicate_top_level_bindings() -> None:
    duplicated_source = "def dup():\n    return 1\n\n\ndef dup():\n    return 2\n"
    other_source = "def dup():\n    return 1\n"

    with pytest.raises(PureInsertionViolationError, match="duplicate"):
        _require_pure_insertion(duplicated_source, other_source, "synthetic")


def test_pure_insertion_proof_rejects_a_changed_decorator() -> None:
    """A decorator argument change is not visible to a source-segment-only
    comparison unless the segment happens to include it; the full
    `ast.dump` comparison catches it unconditionally, with no special-case
    decorator handling required."""
    base_source = (
        "@dataclass(frozen=True)\nclass Thing:\n    value: int\n\n\ndef other():\n    return 1\n"
    )
    head_source = "@dataclass()\nclass Thing:\n    value: int\n\n\ndef other():\n    return 1\n"

    with pytest.raises(PureInsertionViolationError, match="Thing"):
        _require_pure_insertion(base_source, head_source, "synthetic")


def test_pure_insertion_proof_rejects_a_retargeted_import() -> None:
    """`from a import X` silently becoming `from b import X` keeps the same
    bound name `X`, so a name-only proof would miss it; the module/level
    fields inside the `ImportFrom` dump differ and must be caught."""
    base_source = "from a import X\n\n\ndef other():\n    return 1\n"
    head_source = "from b import X\n\n\ndef other():\n    return 1\n"

    with pytest.raises(PureInsertionViolationError, match="X"):
        _require_pure_insertion(base_source, head_source, "synthetic")


def test_pure_insertion_proof_rejects_a_new_unbound_top_level_statement() -> None:
    """An unbound top-level statement (here, a bare `if`) has no nameable
    binding to allowlist, so it can never be silently accepted as a "safe
    addition" the way a new function/class/import name can be -- it must
    already exist, unchanged, in base."""
    base_source = "def kept():\n    return 1\n"
    head_source = 'def kept():\n    return 1\n\n\nif True:\n    print("side effect")\n'

    with pytest.raises(PureInsertionViolationError, match="unbound top-level statement"):
        _require_pure_insertion(base_source, head_source, "synthetic")

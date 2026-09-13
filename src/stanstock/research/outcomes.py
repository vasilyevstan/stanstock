from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from typing import Any
from uuid import UUID

import polars as pl
from django.db import transaction
from django.utils import timezone

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.asof import AsOfData, PriceFrameSchemaError, raw_price_asset_for
from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset
from stanstock.research.long_forecast_config import (
    LONG_V4_EFFECTIVE_CONFIG_HASH,
    LONG_V4_METHOD,
    LONG_V4_RESEARCH_STATUS,
    LONG_V4_VERSION,
)
from stanstock.research.long_forecasts_v4 import (
    PROBABILITY_REASON,
    LongV4PersistedAuthorityError,
    _catalog_manifest_asset_ids,
    _listing_identity_payload,
    _validate_long_v4_persisted_authority,
    _validate_long_v4_raw_sec_replay,
    canonical_long_v4_price,
)
from stanstock.research.models import Prediction, PredictionOutcome, Recommendation

_LONG_V4_RETURN_QUANTUM = Decimal("0.0001")
_LONG_V4_BASELINE_ROLE = "calculation.target_price.valuation_value"
_LONG_V4_BASELINE_COMPARISON = "Decimal(str(target_close)) == valuation_value"
_LONG_V4_BASELINE_ISSUE_ORDER = (
    "identity_mismatch",
    "target_identity_mismatch",
    "target_price_identity_mismatch",
    "valuation_source_mismatch",
    "valuation_value_invalid",
    "ledger_value_mismatch",
)

HORIZON_SESSION_COUNTS: dict[str, int] = {
    Prediction.Horizon.SHORT.value: 10,
    Prediction.Horizon.SIX_MONTH.value: 126,
    Prediction.Horizon.TWELVE_MONTH.value: 252,
    Prediction.Horizon.THREE_YEAR.value: 756,
    Prediction.Horizon.FIVE_YEAR.value: 1260,
    Prediction.Horizon.MEDIUM.value: 252,
    Prediction.Horizon.LONG.value: 756,
}
PriceFrameCache = MutableMapping[tuple[str, str, date, datetime], pl.DataFrame]

#: `(subject, through_date) -> price frame` -- the only IO `resolve_outcome`
#: ever triggers. The producer's loader reads through `AsOfData` (and may
#: legitimately raise `DataAsset.DoesNotExist`/`PriceFrameSchemaError`/
#: `ValueError`, all replayed as a genuine unresolved outcome below); the
#: scheduled-refresh verifier's own loader independently re-derives and
#: checksum-proves the same subject's price evidence and must never raise
#: one of those same producer-domain exception types for a verifier-only
#: integrity failure (see `research.outcome_refresh_validation`).
PriceLoader = Callable[[str, date], pl.DataFrame]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    prediction: Prediction
    outcome: PredictionOutcome
    action: str
    resolution: str


@dataclass(frozen=True, slots=True)
class ResolvedOutcome:
    """One prediction's fully-resolved (but not yet persisted) outcome.

    Every field here is exactly the value `evaluate_prediction` would pass
    to `PredictionOutcome.objects.update_or_create` -- computed by the same
    pure `resolve_outcome` function the writing producer and the read-only
    verifier both call, so neither can ever compute a different answer from
    the same evidence.
    """

    status: str
    evaluation_date: date
    actual_return: Decimal | None
    benchmark_return: Decimal | None
    success: bool | None
    direction_correct: bool | None
    interval_covered: bool | None
    resolution: str
    error: Decimal | None
    signed_error: Decimal | None
    metadata: dict[str, Any]


class PriceSessionDataError(ValueError):
    """Raised when price observations cannot represent unique market sessions."""


@dataclass(frozen=True, slots=True)
class _LongV4Baseline:
    valuation_value: Decimal
    ledger_value: Decimal


def resolve_outcome(
    prediction: Prediction,
    *,
    provider: str,
    evaluation_date: date,
    evaluated_at: datetime,
    benchmark_subject: str | None,
    price_loader: PriceLoader,
    store: AssetStore | None = None,
) -> ResolvedOutcome:
    """Decide one prediction's outcome from already-resolved inputs.

    Pure with respect to persistence: performs no locking, no existing-row
    lookup, and no write. `provider` must already be the caller's own
    resolved/conflict-checked provider (`evaluate_prediction` calls
    `_resolve_price_provider` itself, unconditionally, before this
    function is ever reached -- including for an outcome this function
    will never be asked to resolve because it is already terminal -- so
    that check is never repeated here). Evaluation-price IO goes through
    `price_loader(subject, through_date)`. V4 additionally replays its
    assessed raw SEC closure after scalar/DB/manifest authentication. The
    date/withheld-scenario guards below intentionally run before either read.
    """
    if evaluation_date > evaluated_at.date():
        return _unresolved_outcome(
            evaluation_date=evaluation_date,
            resolution="Evaluation date is after actual evaluation time",
            metadata={"provider": provider, "evaluation_time": evaluated_at.isoformat()},
        )
    if evaluation_date < prediction.target_date:
        return _unresolved_outcome(
            evaluation_date=evaluation_date,
            resolution="Evaluation date is before prediction target date",
            metadata={"provider": provider, "target_date": prediction.target_date.isoformat()},
        )
    if (
        prediction.evidence_role == Prediction.EvidenceRole.ADVISORY
        and prediction.bear_return is None
        and prediction.base_return is None
        and prediction.bull_return is None
    ):
        return _unresolved_outcome(
            evaluation_date=evaluation_date,
            resolution="Withheld forecast has no scenario to evaluate",
            metadata={
                "provider": provider,
                "insufficiency_reason": prediction.insufficiency_reason,
            },
        )

    long_v4_baseline: _LongV4Baseline | None = None
    if prediction.method_version == LONG_V4_VERSION:
        long_v4_baseline, baseline_issues = _authenticate_long_v4_baseline(
            prediction,
            store=store,
        )
        if long_v4_baseline is None:
            return _unresolved_outcome(
                evaluation_date=evaluation_date,
                resolution="Long-v4 valuation baseline could not be authenticated",
                metadata={
                    "provider": provider,
                    "valuation_baseline": _long_v4_baseline_metadata(
                        status="authentication_failed",
                        issue_codes=baseline_issues,
                    ),
                },
            )

    subject = _price_subject(prediction)
    session_count = HORIZON_SESSION_COUNTS[prediction.horizon]

    try:
        price_frame = price_loader(subject, evaluation_date)
    except (DataAsset.DoesNotExist, PriceFrameSchemaError, ValueError) as exc:
        return _unresolved_outcome(
            evaluation_date=evaluation_date,
            resolution=f"Unable to load evaluation price history: {exc}",
            metadata={"provider": provider, "subject": subject},
        )

    try:
        session = _nth_observed_session(price_frame, prediction.target_date, session_count)
        observed = _observed_session_count(price_frame, prediction.target_date)
    except PriceSessionDataError as exc:
        return _unresolved_outcome(
            evaluation_date=evaluation_date,
            resolution=str(exc),
            metadata={"provider": provider, "subject": subject},
        )
    if session is None:
        return _unresolved_outcome(
            evaluation_date=evaluation_date,
            resolution=(
                f"Insufficient observed sessions after target date: "
                f"{observed}/{session_count} through {evaluation_date.isoformat()}"
            ),
            metadata={"provider": provider, "subject": subject, "observed_sessions": observed},
        )

    valuation_metadata: dict[str, Any] | None = None
    if long_v4_baseline is not None:
        evaluation_baseline = _close_on_date(price_frame, prediction.target_date)
        if evaluation_baseline is None:
            return _unresolved_outcome(
                evaluation_date=session.observation_date,
                resolution="Evaluation price history has no exact target-date close",
                metadata={
                    "provider": provider,
                    "subject": subject,
                    "valuation_baseline": _long_v4_baseline_metadata(
                        status="target_close_missing",
                        baseline=long_v4_baseline,
                        target_date=prediction.target_date,
                    ),
                },
            )
        evaluation_target_close = Decimal(str(evaluation_baseline.close))
        valuation_metadata = _long_v4_baseline_metadata(
            status="authenticated",
            baseline=long_v4_baseline,
            target_close=evaluation_target_close,
            target_date=evaluation_baseline.observation_date,
        )
        if evaluation_target_close != long_v4_baseline.valuation_value:
            valuation_metadata["status"] = "target_close_mismatch"
            return _corporate_event_outcome(
                evaluation_date=session.observation_date,
                resolution="Target-date price changed in the evaluation vintage",
                metadata={
                    "provider": provider,
                    "subject": subject,
                    "valuation_baseline": valuation_metadata,
                },
            )
        with localcontext() as context:
            context.prec = 64
            context.rounding = ROUND_HALF_EVEN
            raw_return = Decimal(str(session.close)) / long_v4_baseline.valuation_value - Decimal(1)
            actual_return: Decimal | float = raw_return.quantize(
                _LONG_V4_RETURN_QUANTUM,
                rounding=ROUND_HALF_EVEN,
            )
    else:
        price_at_prediction = float(prediction.price_at_prediction)
        if price_at_prediction <= 0:
            return _unresolved_outcome(
                evaluation_date=evaluation_date,
                resolution="Prediction price is not positive",
                metadata={"provider": provider, "subject": subject},
            )

        evaluation_baseline = _close_at_or_before(price_frame, prediction.target_date)
        if evaluation_baseline is None:
            return _unresolved_outcome(
                evaluation_date=session.observation_date,
                resolution=(
                    "Evaluation price history has no baseline close at or before target date"
                ),
                metadata={"provider": provider, "subject": subject},
            )
        if not _same_price_basis(evaluation_baseline.close, price_at_prediction):
            return _corporate_event_outcome(
                evaluation_date=session.observation_date,
                resolution="Target-date price changed in the evaluation vintage",
                metadata={
                    "provider": provider,
                    "subject": subject,
                    "prediction_price": price_at_prediction,
                    "evaluation_vintage_target_close": evaluation_baseline.close,
                    "evaluation_vintage_target_date": (
                        evaluation_baseline.observation_date.isoformat()
                    ),
                },
            )

        actual_return = session.close / price_at_prediction - 1.0
    success = (
        _success(prediction, actual_return)
        if (
            not isinstance(actual_return, Decimal)
            and prediction.evidence_role == Prediction.EvidenceRole.DECISION
        )
        else None
    )
    if prediction.evidence_role == Prediction.EvidenceRole.DECISION and success is None:
        return _unresolved_outcome(
            evaluation_date=session.observation_date,
            resolution="HOLD success requires non-null stored bear and bull returns",
            metadata={"provider": provider, "subject": subject},
        )

    benchmark_return = None
    benchmark_resolution = ""
    if benchmark_subject:
        benchmark_return, benchmark_resolution = _benchmark_return(
            price_loader=price_loader,
            subject=benchmark_subject,
            target_date=prediction.target_date,
            evaluation_date=session.observation_date,
        )
    error: Decimal | float | None = None
    direction_correct = None
    interval_covered = None
    if prediction.base_return is not None:
        if isinstance(actual_return, Decimal):
            error = actual_return - prediction.base_return
            direction_correct = _decimal_direction(actual_return) == _decimal_direction(
                prediction.base_return
            )
        else:
            error = actual_return - float(prediction.base_return)
            direction_correct = _direction(actual_return) == _direction(
                float(prediction.base_return)
            )
    if prediction.bear_return is not None and prediction.bull_return is not None:
        if isinstance(actual_return, Decimal):
            interval_covered = prediction.bear_return <= actual_return <= prediction.bull_return
        else:
            interval_covered = (
                float(prediction.bear_return) <= actual_return <= float(prediction.bull_return)
            )

    return ResolvedOutcome(
        status=PredictionOutcome.Status.MATURED,
        evaluation_date=session.observation_date,
        actual_return=(
            actual_return if isinstance(actual_return, Decimal) else _decimal(actual_return)
        ),
        benchmark_return=_optional_decimal(benchmark_return),
        success=success,
        direction_correct=direction_correct,
        interval_covered=interval_covered,
        resolution=_matured_resolution(prediction, session.observation_date, benchmark_resolution),
        error=error if isinstance(error, Decimal) else _optional_decimal(error),
        signed_error=error if isinstance(error, Decimal) else _optional_decimal(error),
        metadata={
            "provider": provider,
            "subject": subject,
            "horizon_sessions": session_count,
            "evidence_role": prediction.evidence_role,
            "evaluation_close": session.close,
            "benchmark_subject": benchmark_subject or "",
            "benchmark_resolution": benchmark_resolution,
            "success_semantics": _success_semantics(prediction),
            **(
                {"valuation_baseline": valuation_metadata} if valuation_metadata is not None else {}
            ),
        },
    )


def _authenticate_long_v4_baseline(
    prediction: Prediction,
    *,
    store: AssetStore | None,
) -> tuple[_LongV4Baseline | None, tuple[str, ...]]:
    calculation = prediction.calculation
    calculation_mapping = calculation if isinstance(calculation, Mapping) else {}
    target = calculation_mapping.get("target")
    target_mapping = target if isinstance(target, Mapping) else {}
    target_price = calculation_mapping.get("target_price")
    price_mapping = target_price if isinstance(target_price, Mapping) else {}

    identity_matches, assessed_owners = _long_v4_prediction_identity_matches(
        prediction,
        calculation_mapping,
    )
    identity_mismatch = not identity_matches
    target_identity_mismatch = not _long_v4_target_identity_matches(
        prediction,
        target_mapping,
    )
    target_price_identity_mismatch = not _long_v4_target_price_identity_matches(
        prediction,
        calculation_mapping=calculation_mapping,
        target_mapping=target_mapping,
        price_mapping=price_mapping,
    )
    valuation_source_mismatch = not (
        price_mapping.get("valuation_source") == "normalized_parquet_target_close"
        and price_mapping.get("native_price") == price_mapping.get("valuation_value")
        and calculation_mapping.get("return_basis") == "split_adjusted_price_return"
        and calculation_mapping.get("dividends_included") is False
        and calculation_mapping.get("base_currency") == "USD"
        and calculation_mapping.get("fx_conversion") is False
    )

    valuation_value = _canonical_positive_decimal_text(price_mapping.get("valuation_value"))
    valuation_value_invalid = valuation_value is None
    ledger_value = _canonical_six_place_decimal(price_mapping.get("ledger_value"))
    ledger_value_mismatch = (
        ledger_value is None
        or price_mapping.get("value") != price_mapping.get("ledger_value")
        or prediction.price_at_prediction != ledger_value
        or prediction.analysis.current_price != ledger_value
    )
    if valuation_value is not None and ledger_value is not None:
        try:
            with localcontext() as context:
                context.prec = 64
                context.rounding = ROUND_HALF_EVEN
                ledger_value_mismatch |= canonical_long_v4_price(valuation_value) != ledger_value
        except ValueError:
            ledger_value_mismatch = True

    issue_flags = {
        "identity_mismatch": identity_mismatch,
        "target_identity_mismatch": target_identity_mismatch,
        "target_price_identity_mismatch": target_price_identity_mismatch,
        "valuation_source_mismatch": valuation_source_mismatch,
        "valuation_value_invalid": valuation_value_invalid,
        "ledger_value_mismatch": ledger_value_mismatch,
    }
    issues = tuple(code for code in _LONG_V4_BASELINE_ISSUE_ORDER if issue_flags[code])
    if issues or valuation_value is None or ledger_value is None:
        return None, issues
    try:
        _validate_long_v4_raw_sec_replay(
            calculation=calculation_mapping,
            assessed_owners=assessed_owners,
            target_date=prediction.target_date,
            data_cutoff=prediction.data_cutoff,
            decision_time=prediction.generated_at,
            store=store,
        )
    except LongV4PersistedAuthorityError:
        return None, ("identity_mismatch",)
    return _LongV4Baseline(valuation_value, ledger_value), ()


def _long_v4_prediction_identity_matches(
    prediction: Prediction,
    calculation: Mapping[str, Any],
) -> tuple[bool, tuple[tuple[Any, Any], ...]]:
    analysis = prediction.analysis
    run = analysis.run
    years = 3 if prediction.horizon == Prediction.Horizon.THREE_YEAR else 5
    selected_view = calculation.get("selected_view")
    selected_mapping = selected_view if isinstance(selected_view, Mapping) else {}
    scalar_identity_matches = bool(
        calculation
        and prediction.evidence_role == Prediction.EvidenceRole.ADVISORY
        and prediction.evidence_grade == "research"
        and prediction.issued_on_time is False
        and prediction.price_provider == "twelve_data"
        and bool(prediction.price_subject)
        and prediction.method_version == LONG_V4_VERSION
        and prediction.config_hash == LONG_V4_EFFECTIVE_CONFIG_HASH
        and prediction.model_version.startswith(f"{LONG_V4_VERSION}-")
        and prediction.horizon in (Prediction.Horizon.THREE_YEAR, Prediction.Horizon.FIVE_YEAR)
        and prediction.analysis_id == analysis.pk
        and prediction.listing_id == analysis.listing_id
        and prediction.generated_at == run.generated_at
        and prediction.target_date == run.target_date
        and prediction.data_cutoff == run.data_cutoff
        and prediction.code_revision == run.code_revision
        and calculation.get("schema_version") == 2
        and calculation.get("method") == LONG_V4_METHOD
        and calculation.get("method_version") == prediction.method_version
        and calculation.get("config_hash") == prediction.config_hash
        and calculation.get("research_status") == LONG_V4_RESEARCH_STATUS
        and calculation.get("path_years") == 5
        and calculation.get("target_date") == prediction.target_date.isoformat()
        and calculation.get("forecast_horizon") == prediction.horizon
        and calculation.get("years") == years
        and calculation.get("prediction_version") == prediction.model_version
        and calculation.get("price_subject") == prediction.price_subject
        and calculation.get("evidence_grade") == prediction.evidence_grade
    )
    if not scalar_identity_matches:
        return False, ()
    manifests_match, assessed_owners = _long_v4_source_manifests_match(
        prediction,
        calculation,
    )
    identity_matches = bool(
        manifests_match
        and selected_mapping.get("horizon") == prediction.horizon
        and selected_mapping.get("year") == years
        and _long_v4_selected_returns_match(prediction, selected_mapping)
        and _long_v4_probability_confidence_matches(prediction, calculation)
    )
    return identity_matches, assessed_owners if identity_matches else ()


def _long_v4_selected_returns_match(
    prediction: Prediction,
    selected_view: Mapping[str, Any],
) -> bool:
    returns = selected_view.get("cumulative_returns")
    if not isinstance(returns, Mapping) or set(returns) != {"bear", "base", "bull"}:
        return False
    expected = {
        "bear": prediction.bear_return,
        "base": prediction.base_return,
        "bull": prediction.bull_return,
    }
    for name, stored in expected.items():
        valid, selected = _canonical_long_v4_return(returns.get(name))
        if not valid or selected != stored:
            return False
    return True


def _canonical_long_v4_return(value: object) -> tuple[bool, Decimal | None]:
    if value is None:
        return True, None
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return False, None
    try:
        parsed = Decimal(str(value))
        if not parsed.is_finite():
            return False, None
        with localcontext() as context:
            context.prec = 64
            context.rounding = ROUND_HALF_EVEN
            return True, parsed.quantize(
                _LONG_V4_RETURN_QUANTUM,
                rounding=ROUND_HALF_EVEN,
            )
    except (InvalidOperation, ValueError):
        return False, None


def _long_v4_probability_confidence_matches(
    prediction: Prediction,
    calculation: Mapping[str, Any],
) -> bool:
    scenario_values = (
        prediction.bear_return,
        prediction.base_return,
        prediction.bull_return,
    )
    complete = all(value is not None for value in scenario_values)
    withheld = all(value is None for value in scenario_values)
    if not complete and not withheld:
        return False
    calculation_reason = calculation.get("insufficiency_reason")
    insufficiency_code = calculation.get("insufficiency_code")
    if complete:
        if insufficiency_code is not None or calculation_reason != "":
            return False
        expected_status = "not_estimated_uncalibrated"
        expected_reason = PROBABILITY_REASON
    else:
        if not isinstance(insufficiency_code, str) or not insufficiency_code:
            return False
        if not isinstance(calculation_reason, str) or not calculation_reason:
            return False
        expected_status = "not_estimated_insufficient"
        expected_reason = calculation_reason
    return bool(
        prediction.probability_positive is None
        and prediction.confidence == Decimal("0.00")
        and prediction.confidence_status == expected_status
        and prediction.insufficiency_reason == expected_reason
        and calculation.get("probability_semantics")
        == {
            "status": "unavailable",
            "value": None,
            "reason": expected_reason,
        }
        and calculation.get("confidence_semantics")
        == {
            "status": expected_status,
            "value": 0.0,
            "schema": "zero_is_unavailable_sentinel",
        }
    )


def _long_v4_source_manifests_match(
    prediction: Prediction,
    calculation: Mapping[str, Any],
) -> tuple[bool, tuple[tuple[Any, Any], ...]]:
    try:
        assessed_owners = _validate_long_v4_persisted_authority(
            calculation=calculation,
            target_listing_id=prediction.listing_id,
            universe_snapshot_id=prediction.analysis.run.universe_snapshot_id,
            analysis_run_id=prediction.analysis.run_id,
            target_date=prediction.target_date,
            data_cutoff=prediction.data_cutoff,
            generated_at=prediction.generated_at,
            prediction_id=prediction.pk,
            prediction_horizon=prediction.horizon,
            prediction_model_version=prediction.model_version,
            prediction_scenario_values=(
                prediction.bear_return,
                prediction.base_return,
                prediction.bull_return,
            ),
        )
    except LongV4PersistedAuthorityError:
        return False, ()
    catalog = calculation.get("evidence_catalog")
    calculation_manifest = calculation.get("source_manifest")
    prediction_manifest = prediction.source_assets
    if (
        not isinstance(catalog, dict)
        or not isinstance(calculation_manifest, list)
        or not isinstance(prediction_manifest, list)
    ):
        return False, ()
    try:
        authoritative_ids = _catalog_manifest_asset_ids(catalog)
        authoritative_uuids = tuple(UUID(value) for value in authoritative_ids)
    except (TypeError, ValueError):
        return False, ()
    if len(authoritative_ids) != len(set(authoritative_ids)) or any(
        str(value) != raw for value, raw in zip(authoritative_uuids, authoritative_ids, strict=True)
    ):
        return False, ()

    def parsed_manifest(raw_manifest: list[object]) -> tuple[Mapping[str, Any], ...] | None:
        if len(raw_manifest) != len(authoritative_ids):
            return None
        entries: list[Mapping[str, Any]] = []
        manifest_ids: list[str] = []
        for raw_entry in raw_manifest:
            if not isinstance(raw_entry, Mapping):
                return None
            raw_id = raw_entry.get("id")
            if not isinstance(raw_id, str):
                return None
            try:
                parsed_id = UUID(raw_id)
            except (TypeError, ValueError):
                return None
            if str(parsed_id) != raw_id:
                return None
            entries.append(raw_entry)
            manifest_ids.append(raw_id)
        if len(manifest_ids) != len(set(manifest_ids)) or tuple(manifest_ids) != authoritative_ids:
            return None
        return tuple(entries)

    calculation_entries = parsed_manifest(calculation_manifest)
    prediction_entries = parsed_manifest(prediction_manifest)
    if calculation_entries is None or prediction_entries is None:
        return False, ()
    persisted = DataAsset.objects.in_bulk(authoritative_uuids)
    if set(persisted) != set(authoritative_uuids):
        return False, ()

    for asset_id, calculation_entry, prediction_entry in zip(
        authoritative_uuids,
        calculation_entries,
        prediction_entries,
        strict=True,
    ):
        asset = persisted[asset_id]
        expected = {
            "id": str(asset.pk),
            "provider": asset.provider,
            "kind": asset.kind,
            "subject": asset.subject,
            "relative_path": asset.relative_path,
            "sha256": asset.sha256,
            "retrieved_at": asset.retrieved_at.isoformat(),
            "available_at": asset.available_at.isoformat(),
        }
        if (
            any(calculation_entry.get(field) != value for field, value in expected.items())
            or any(prediction_entry.get(field) != value for field, value in expected.items())
            or asset.retrieved_at > prediction.generated_at
            or asset.available_at > prediction.generated_at
        ):
            return False, ()
    return True, assessed_owners


def _long_v4_target_identity_matches(
    prediction: Prediction,
    target: Mapping[str, Any],
) -> bool:
    listing = prediction.listing
    return dict(target) == _listing_identity_payload(listing)


def _long_v4_target_price_identity_matches(
    prediction: Prediction,
    *,
    calculation_mapping: Mapping[str, Any],
    target_mapping: Mapping[str, Any],
    price_mapping: Mapping[str, Any],
) -> bool:
    authoritative_subject = prediction.listing.provider_symbol
    normalized = price_mapping.get("normalized_asset")
    raw = price_mapping.get("raw_asset")
    normalized_mapping = normalized if isinstance(normalized, Mapping) else {}
    raw_mapping = raw if isinstance(raw, Mapping) else {}
    manifest = calculation_mapping.get("source_manifest")
    manifest_entries = manifest if isinstance(manifest, list) else []
    manifest_by_id = {
        item.get("id"): item
        for item in manifest_entries
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    catalog = calculation_mapping.get("evidence_catalog")
    prices = catalog.get("prices") if isinstance(catalog, Mapping) else None
    matching_catalog_prices = (
        [
            item
            for item in prices
            if isinstance(item, Mapping) and item.get("listing_id") == str(prediction.listing_id)
        ]
        if isinstance(prices, list)
        else []
    )
    return bool(
        price_mapping
        and isinstance(authoritative_subject, str)
        and bool(authoritative_subject)
        and prediction.price_subject == authoritative_subject
        and calculation_mapping.get("price_subject") == authoritative_subject
        and target_mapping.get("provider_symbol") == authoritative_subject
        and price_mapping.get("owner_listing_id") == str(prediction.listing_id)
        and price_mapping.get("listing_id") == str(prediction.listing_id)
        and price_mapping.get("listing_id") == target_mapping.get("listing_id")
        and price_mapping.get("provider") == prediction.price_provider
        and price_mapping.get("subject") == authoritative_subject
        and price_mapping.get("exchange_mic") == target_mapping.get("exchange_mic")
        and price_mapping.get("session_date") == prediction.target_date.isoformat()
        and price_mapping.get("currency") == target_mapping.get("currency")
        and price_mapping.get("native_currency") == target_mapping.get("currency")
        and price_mapping.get("fx_conversion") is False
        and price_mapping.get("applied_fx_rate") is None
        and price_mapping.get("normalized_asset_id") == normalized_mapping.get("id")
        and price_mapping.get("normalized_asset_sha256") == normalized_mapping.get("sha256")
        and price_mapping.get("raw_asset_id") == raw_mapping.get("id")
        and price_mapping.get("raw_asset_sha256") == raw_mapping.get("sha256")
        and normalized_mapping.get("provider") == prediction.price_provider
        and normalized_mapping.get("kind") == "price_history"
        and normalized_mapping.get("subject") == authoritative_subject
        and raw_mapping.get("provider") == prediction.price_provider
        and raw_mapping.get("kind") == "raw_price_history"
        and raw_mapping.get("subject") == authoritative_subject
        and normalized_mapping.get("id") in manifest_by_id
        and raw_mapping.get("id") in manifest_by_id
        and manifest_by_id.get(normalized_mapping.get("id")) == normalized_mapping
        and manifest_by_id.get(raw_mapping.get("id")) == raw_mapping
        and len(matching_catalog_prices) == 1
        and dict(matching_catalog_prices[0]) == dict(price_mapping)
        and _long_v4_persisted_price_closure_matches(
            prediction,
            normalized_mapping=normalized_mapping,
            raw_mapping=raw_mapping,
        )
    )


def _long_v4_persisted_price_closure_matches(
    prediction: Prediction,
    *,
    normalized_mapping: Mapping[str, Any],
    raw_mapping: Mapping[str, Any],
) -> bool:
    issuance_boundary = prediction.generated_at
    if issuance_boundary != prediction.analysis.run.generated_at:
        return False
    normalized_id = normalized_mapping.get("id")
    raw_id = raw_mapping.get("id")
    if not isinstance(normalized_id, str) or not isinstance(raw_id, str):
        return False
    try:
        normalized_uuid = UUID(normalized_id)
        raw_uuid = UUID(raw_id)
    except (TypeError, ValueError):
        return False
    if str(normalized_uuid) != normalized_id or str(raw_uuid) != raw_id:
        return False
    normalized_asset = DataAsset.objects.filter(pk=normalized_uuid).first()
    raw_asset = DataAsset.objects.filter(pk=raw_uuid).first()
    if normalized_asset is None or raw_asset is None:
        return False
    if (
        normalized_asset.available_at > issuance_boundary
        or normalized_asset.retrieved_at > issuance_boundary
        or not _long_v4_asset_identity_matches(normalized_mapping, normalized_asset)
        or not _long_v4_asset_identity_matches(raw_mapping, raw_asset)
    ):
        return False
    try:
        independently_resolved_raw = raw_price_asset_for(
            normalized_asset,
            cutoff=issuance_boundary,
        )
    except (RefreshVerificationError, TypeError, ValueError):
        return False
    return bool(
        independently_resolved_raw.pk == raw_asset.pk
        and raw_asset.provider == prediction.price_provider
        and raw_asset.kind == "raw_price_history"
        and raw_asset.subject == prediction.listing.provider_symbol
    )


def _long_v4_asset_identity_matches(
    payload: Mapping[str, Any],
    asset: DataAsset,
) -> bool:
    return (
        payload.get("id") == str(asset.pk)
        and payload.get("provider") == asset.provider
        and payload.get("kind") == asset.kind
        and payload.get("subject") == asset.subject
        and payload.get("sha256") == asset.sha256
    )


def _canonical_positive_decimal_text(value: object) -> Decimal | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    if not parsed.is_finite() or parsed <= 0 or value != format(parsed, "f"):
        return None
    return parsed


def _canonical_six_place_decimal(value: object) -> Decimal | None:
    parsed = _canonical_positive_decimal_text(value)
    if parsed is None or not isinstance(value, str) or value != format(parsed, ".6f"):
        return None
    return parsed


def _long_v4_baseline_metadata(
    *,
    status: str,
    baseline: _LongV4Baseline | None = None,
    issue_codes: tuple[str, ...] = (),
    target_close: Decimal | None = None,
    target_date: date | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "role": _LONG_V4_BASELINE_ROLE,
        "status": status,
    }
    if issue_codes:
        metadata["issue_codes"] = list(issue_codes)
    if baseline is not None:
        metadata.update(
            {
                "valuation_value": format(baseline.valuation_value, "f"),
                "ledger_value": format(baseline.ledger_value, ".6f"),
                "comparison_type": _LONG_V4_BASELINE_COMPARISON,
                "target_close": format(target_close, "f") if target_close is not None else None,
                "target_date": target_date.isoformat() if target_date is not None else None,
                "return_quantum": format(_LONG_V4_RETURN_QUANTUM, "f"),
                "rounding": "ROUND_HALF_EVEN",
            }
        )
    return metadata


def _price_subject(prediction: Prediction) -> str:
    return (
        prediction.price_subject or prediction.listing.provider_symbol or prediction.listing.ticker
    )


def _unresolved_outcome(
    *, evaluation_date: date, resolution: str, metadata: dict[str, Any]
) -> ResolvedOutcome:
    return ResolvedOutcome(
        status=PredictionOutcome.Status.UNRESOLVED,
        evaluation_date=evaluation_date,
        actual_return=None,
        benchmark_return=None,
        success=None,
        direction_correct=None,
        interval_covered=None,
        resolution=resolution[:120],
        error=None,
        signed_error=None,
        metadata=metadata,
    )


def _corporate_event_outcome(
    *, evaluation_date: date, resolution: str, metadata: dict[str, Any]
) -> ResolvedOutcome:
    return ResolvedOutcome(
        status=PredictionOutcome.Status.CORPORATE_EVENT,
        evaluation_date=evaluation_date,
        actual_return=None,
        benchmark_return=None,
        success=None,
        direction_correct=None,
        interval_covered=None,
        resolution=resolution[:120],
        error=None,
        signed_error=None,
        metadata=metadata,
    )


@transaction.atomic
def evaluate_prediction(
    prediction: Prediction,
    *,
    provider: str,
    evaluation_date: date,
    evaluation_time: datetime | None = None,
    benchmark_subject: str | None = None,
    store: AssetStore | None = None,
    frame_cache: PriceFrameCache | None = None,
) -> EvaluationResult:
    prediction = Prediction.objects.select_for_update().get(pk=prediction.pk)
    existing = PredictionOutcome.objects.select_for_update().filter(prediction=prediction).first()
    selected_provider = _resolve_price_provider(prediction, provider)
    if existing is not None and existing.status in _TERMINAL_OUTCOME_STATUSES:
        return EvaluationResult(
            prediction=prediction,
            outcome=existing,
            action="skipped",
            resolution="Existing terminal outcome left unchanged",
        )

    evaluated_at = evaluation_time or timezone.now()
    frame_cache = frame_cache if frame_cache is not None else {}
    # `AsOfData`/`AssetStore` construction can touch the filesystem (creating
    # `STANSTOCK_DATA_DIR` if `store` is not given), so it must not happen
    # before `resolve_outcome`'s own date/withheld-advisory guards have had a
    # chance to return an unresolved outcome without ever reading price data
    # -- matching the base revision's own ordering byte-for-byte. Memoized on
    # first actual frame request rather than constructed eagerly here.
    asof: AsOfData | None = None

    def _loader(subject: str, through_date: date) -> pl.DataFrame:
        nonlocal asof
        if asof is None:
            asof = AsOfData(evaluated_at, store)
        return _cached_price_frame(
            asof=asof,
            provider=selected_provider,
            subject=subject,
            through_date=through_date,
            evaluated_at=evaluated_at,
            cache=frame_cache,
        )

    resolved = resolve_outcome(
        prediction,
        provider=selected_provider,
        evaluation_date=evaluation_date,
        evaluated_at=evaluated_at,
        benchmark_subject=benchmark_subject,
        price_loader=_loader,
        store=store,
    )

    if resolved.status == PredictionOutcome.Status.UNRESOLVED:
        return _save_unresolved(
            prediction,
            existing=existing,
            evaluated_at=evaluated_at,
            evaluation_date=resolved.evaluation_date,
            resolution=resolved.resolution,
            metadata=resolved.metadata,
        )
    if resolved.status == PredictionOutcome.Status.CORPORATE_EVENT:
        return _save_corporate_event(
            prediction,
            existing=existing,
            evaluated_at=evaluated_at,
            evaluation_date=resolved.evaluation_date,
            resolution=resolved.resolution,
            metadata=resolved.metadata,
        )

    outcome, created = PredictionOutcome.objects.update_or_create(
        prediction=prediction,
        defaults={
            "evaluated_at": evaluated_at,
            "evaluation_date": resolved.evaluation_date,
            "status": PredictionOutcome.Status.MATURED,
            "actual_return": resolved.actual_return,
            "benchmark_return": resolved.benchmark_return,
            "success": resolved.success,
            "direction_correct": resolved.direction_correct,
            "interval_covered": resolved.interval_covered,
            "resolution": resolved.resolution,
            "error": resolved.error,
            "signed_error": resolved.signed_error,
            "metadata": resolved.metadata,
        },
    )
    return EvaluationResult(
        prediction=prediction,
        outcome=outcome,
        action="created" if created else "updated",
        resolution=outcome.resolution,
    )


def evaluate_predictions(
    predictions: list[Prediction],
    *,
    provider: str,
    evaluation_date: date,
    evaluation_time: datetime | None = None,
    benchmark_subject: str | None = None,
    store: AssetStore | None = None,
) -> list[EvaluationResult]:
    evaluated_at = evaluation_time or timezone.now()
    frame_cache: dict[tuple[str, str, date, datetime], pl.DataFrame] = {}
    for prediction in predictions:
        _resolve_price_provider(prediction, provider)
    return [
        evaluate_prediction(
            prediction,
            provider=provider,
            evaluation_date=evaluation_date,
            evaluation_time=evaluated_at,
            benchmark_subject=benchmark_subject,
            store=store,
            frame_cache=frame_cache,
        )
        for prediction in predictions
    ]


@dataclass(frozen=True, slots=True)
class PriceSession:
    observation_date: date
    close: float


def _nth_observed_session(
    frame: pl.DataFrame, target_date: date, session_count: int
) -> PriceSession | None:
    sessions = _usable_sessions(frame).filter(pl.col("date") > target_date).sort("date")
    if sessions.height < session_count:
        return None
    row = sessions.row(session_count - 1, named=True)
    return PriceSession(observation_date=row["date"], close=float(row["close"]))


def _observed_session_count(frame: pl.DataFrame, target_date: date) -> int:
    return _usable_sessions(frame).filter(pl.col("date") > target_date).height


def _close_at_or_before(frame: pl.DataFrame, target: date) -> PriceSession | None:
    sessions = _usable_sessions(frame).filter(pl.col("date") <= target).sort("date")
    if sessions.is_empty():
        return None
    row = sessions.row(-1, named=True)
    return PriceSession(observation_date=row["date"], close=float(row["close"]))


def _close_on_date(frame: pl.DataFrame, target: date) -> PriceSession | None:
    sessions = _usable_sessions(frame).filter(pl.col("date") == target)
    if sessions.is_empty():
        return None
    row = sessions.row(0, named=True)
    return PriceSession(observation_date=row["date"], close=float(row["close"]))


def _usable_sessions(frame: pl.DataFrame) -> pl.DataFrame:
    if "date" not in frame.columns or "close" not in frame.columns:
        return pl.DataFrame(
            {"date": [], "close": []}, schema={"date": pl.Date, "close": pl.Float64}
        )
    sessions = (
        frame.select(
            _date_expr(frame).alias("date"), pl.col("close").cast(pl.Float64, strict=False)
        )
        .filter(
            pl.col("date").is_not_null() & pl.col("close").is_not_null() & (pl.col("close") > 0)
        )
        .sort("date")
    )
    duplicate_count = sessions.filter(pl.col("date").is_duplicated()).height
    if duplicate_count:
        raise PriceSessionDataError("Duplicate usable price session dates in evaluation data")
    return sessions


def _date_expr(frame: pl.DataFrame) -> pl.Expr:
    dtype = frame.schema["date"]
    if dtype == pl.Date:
        return pl.col("date")
    if isinstance(dtype, pl.Datetime):
        return pl.col("date").dt.date()
    if dtype in (pl.Utf8, pl.String):
        return pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    return pl.col("date").cast(pl.Date, strict=False)


def _cached_price_frame(
    *,
    asof: AsOfData,
    provider: str,
    subject: str,
    through_date: date,
    evaluated_at: datetime,
    cache: PriceFrameCache,
) -> pl.DataFrame:
    key = (provider, subject, through_date, evaluated_at)
    cached = cache.get(key)
    if cached is not None:
        return cached
    frame = asof.price_frame(
        provider=provider,
        subject=subject,
        through_date=through_date,
    )
    cache[key] = frame
    return frame


def _benchmark_return(
    *,
    price_loader: PriceLoader,
    subject: str,
    target_date: date,
    evaluation_date: date,
) -> tuple[float | None, str]:
    try:
        frame = price_loader(subject, evaluation_date)
        target = _close_at_or_before(frame, target_date)
        evaluation = _close_at_or_before(frame, evaluation_date)
    except (
        DataAsset.DoesNotExist,
        PriceFrameSchemaError,
        ValueError,
        PriceSessionDataError,
    ) as exc:
        return None, f"Benchmark unavailable: {exc}"
    if target is None:
        return None, "Benchmark has no close at or before target date"
    if evaluation is None:
        return None, "Benchmark has no close at or before evaluation session"
    return evaluation.close / target.close - 1.0, (
        f"Benchmark return uses {target.observation_date.isoformat()} to "
        f"{evaluation.observation_date.isoformat()} closes"
    )


def _success(prediction: Prediction, actual_return: float) -> bool | None:
    if prediction.recommendation == Recommendation.BUY:
        return actual_return > 0
    if prediction.recommendation == Recommendation.AVOID:
        return actual_return <= 0
    bear = prediction.bear_return
    bull = prediction.bull_return
    if bear is None or bull is None:
        return None
    return float(bear) <= actual_return <= float(bull)


def _success_semantics(prediction: Prediction) -> str:
    if prediction.evidence_role == Prediction.EvidenceRole.ADVISORY:
        return "Advisory forecasts do not receive recommendation success labels"
    if prediction.recommendation == Recommendation.BUY:
        return "BUY succeeds when actual return is positive"
    if prediction.recommendation == Recommendation.AVOID:
        return "AVOID succeeds when actual return is non-positive"
    return "HOLD succeeds when actual return is within stored bear/bull range"


def _matured_resolution(
    prediction: Prediction,
    evaluation_date: date,
    benchmark_resolution: str,
) -> str:
    base = (
        f"Matured after {HORIZON_SESSION_COUNTS[prediction.horizon]} observed sessions; "
        f"evaluated on {evaluation_date.isoformat()}"
    )
    if benchmark_resolution:
        return f"{base}; {benchmark_resolution}"[:120]
    return base[:120]


def _save_unresolved(
    prediction: Prediction,
    *,
    existing: PredictionOutcome | None,
    evaluated_at: datetime,
    evaluation_date: date,
    resolution: str,
    metadata: dict[str, Any],
) -> EvaluationResult:
    if existing is not None and existing.status in _TERMINAL_OUTCOME_STATUSES:
        return EvaluationResult(prediction, existing, "skipped", existing.resolution)
    defaults = {
        "evaluated_at": evaluated_at,
        "evaluation_date": evaluation_date,
        "status": PredictionOutcome.Status.UNRESOLVED,
        "actual_return": None,
        "benchmark_return": None,
        "success": None,
        "direction_correct": None,
        "interval_covered": None,
        "resolution": resolution[:120],
        "error": None,
        "signed_error": None,
        "metadata": metadata,
    }
    if existing is not None and _outcome_matches(existing, defaults):
        return EvaluationResult(prediction, existing, "skipped", existing.resolution)
    outcome, created = PredictionOutcome.objects.update_or_create(
        prediction=prediction,
        defaults=defaults,
    )
    return EvaluationResult(
        prediction=prediction,
        outcome=outcome,
        action="created" if created else "updated",
        resolution=outcome.resolution,
    )


def _save_corporate_event(
    prediction: Prediction,
    *,
    existing: PredictionOutcome | None,
    evaluated_at: datetime,
    evaluation_date: date,
    resolution: str,
    metadata: dict[str, Any],
) -> EvaluationResult:
    if existing is not None and existing.status in _TERMINAL_OUTCOME_STATUSES:
        return EvaluationResult(prediction, existing, "skipped", existing.resolution)
    outcome, created = PredictionOutcome.objects.update_or_create(
        prediction=prediction,
        defaults={
            "evaluated_at": evaluated_at,
            "evaluation_date": evaluation_date,
            "status": PredictionOutcome.Status.CORPORATE_EVENT,
            "actual_return": None,
            "benchmark_return": None,
            "success": None,
            "direction_correct": None,
            "interval_covered": None,
            "resolution": resolution[:120],
            "error": None,
            "signed_error": None,
            "metadata": metadata,
        },
    )
    return EvaluationResult(
        prediction=prediction,
        outcome=outcome,
        action="created" if created else "updated",
        resolution=outcome.resolution,
    )


def _same_price_basis(evaluation_price: float, prediction_price: float) -> bool:
    tolerance = max(1e-6, abs(prediction_price) * 1e-6)
    return abs(evaluation_price - prediction_price) <= tolerance


def _resolve_price_provider(prediction: Prediction, requested_provider: str) -> str:
    recorded_provider = prediction.price_provider.strip()
    if recorded_provider and recorded_provider != requested_provider:
        raise ValueError(
            "Prediction price provider "
            f"{recorded_provider!r} conflicts with requested provider {requested_provider!r}"
        )
    return recorded_provider or requested_provider


def _direction(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _decimal_direction(value: Decimal) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _outcome_matches(
    outcome: PredictionOutcome,
    defaults: dict[str, Any],
) -> bool:
    comparable_fields = (
        "evaluation_date",
        "status",
        "actual_return",
        "benchmark_return",
        "success",
        "direction_correct",
        "interval_covered",
        "resolution",
        "error",
        "signed_error",
        "metadata",
    )
    return all(getattr(outcome, field) == defaults[field] for field in comparable_fields)


def _decimal(value: float) -> Decimal:
    return Decimal(str(round(value, 4)))


def _optional_decimal(value: float | None) -> Decimal | None:
    if value is None:
        return None
    return _decimal(value)


_TERMINAL_OUTCOME_STATUSES = {
    PredictionOutcome.Status.MATURED,
    PredictionOutcome.Status.CORPORATE_EVENT,
}

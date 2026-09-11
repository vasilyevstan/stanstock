"""Research-domain scheduled-refresh verification: per-prediction outcomes.

Proves one evaluated `Prediction`'s persisted `PredictionOutcome` row is
exactly what a fresh, independent replay of `research.outcomes.resolve_outcome`
would produce -- for every candidate the evaluation stage was expected to
touch, whether it is now terminal (`matured`/`corporate_event`), genuinely
still `unresolved`, or an all-null withheld advisory forecast. Never
short-circuits from the persisted row's own status: `core.refresh_verification`
calls `verify_prediction_outcome` once per expected candidate, regardless of
what that candidate's outcome claims to be.

This module never calls `evaluate_prediction` (the writing producer) or
mutates anything. It reuses exactly two producer-owned pieces from
`research.outcomes` -- the pure `resolve_outcome` decision function and the
`_resolve_price_provider` conflict-check `evaluate_prediction` itself applies
unconditionally before ever reaching `resolve_outcome` -- so this verifier can
never compute a different answer than the producer would from the same
evidence, and never re-implements provider/subject resolution or per-row
maturity-session arithmetic.

**Mandatory exception transport** (see `resolve_outcome`'s own docstring):
`resolve_outcome` only ever catches a fixed, narrow set of producer-domain
exceptions (`DataAsset.DoesNotExist`, `PriceFrameSchemaError`, plain
`ValueError`, `PriceSessionDataError`) and replays them as a genuine
unresolved/benchmark-unavailable outcome. This verifier's own `_price_loader`
must therefore never let a verifier-only integrity failure (wrong asset
`kind`, a cutoff violation, a missing file, a checksum mismatch, corrupt
Parquet, a broken raw-asset closure) surface as one of those same exception
types, or it would be silently absorbed into a plausible-looking but wrong
"unresolved" replay instead of failing this verification closed. Every such
failure is raised as `_VerifierLoadError` -- a `RuntimeError` subclass
`resolve_outcome` never catches -- and translated to a static, path-free
`RefreshVerificationError` only here, outside `resolve_outcome`.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from uuid import UUID

import polars as pl

from stanstock.core.verification_types import (
    AssetRef,
    RefreshVerificationError,
    StageVerificationResult,
)
from stanstock.data.asof import (
    AsOfData,
    PriceFrameChecksumMismatchError,
    PriceFrameSchemaError,
    raw_price_asset_for,
)
from stanstock.data.assets import asset_ref_for, open_asset_store, read_checksummed_bytes
from stanstock.data.models import DataAsset
from stanstock.research.models import Prediction, PredictionOutcome
from stanstock.research.outcomes import ResolvedOutcome, _resolve_price_provider, resolve_outcome
from stanstock.research.refresh_evidence import model_row_values

#: Every concrete `PredictionOutcome` field this verifier compares, save for
#: `evaluated_at` (checked separately, since its acceptable relationship to
#: `evaluation_time` differs by terminal/unresolved status -- see
#: `_check_evaluated_at`). `test_research_outcome_refresh_validation.py`'s
#: `test_outcome_fields_plus_evaluated_at_cover_every_concrete_field` fails
#: closed if a migration adds a concrete field neither list covers.
OUTCOME_FIELDS: tuple[str, ...] = (
    "prediction_id",
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

_TERMINAL_OUTCOME_STATUSES = frozenset(
    {PredictionOutcome.Status.MATURED, PredictionOutcome.Status.CORPORATE_EVENT}
)
_PRICE_HISTORY_KIND = "price_history"


class _VerifierLoadError(RuntimeError):
    """A verifier-only price-evidence integrity failure (never a ValueError).

    Carries only a reason code and a static, path-free message -- never the
    underlying exception's own text, which can embed an asset UUID or a
    filesystem path. Deliberately not a `ValueError` (unlike
    `RefreshVerificationError`) so `resolve_outcome`'s own producer-domain
    exception handling can never catch and silently absorb it.
    """

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(slots=True)
class _VisitedAssets:
    """Every distinct normalized price asset this call's loader actually read."""

    by_id: dict[UUID, DataAsset] = field(default_factory=dict)

    def record(self, asset: DataAsset) -> None:
        self.by_id[asset.id] = asset


def verify_prediction_outcome(
    prediction: Prediction,
    persisted: PredictionOutcome,
    *,
    provider: str,
    benchmark_subject: str,
    evaluation_time: datetime,
    parent_target_date: date,
    frame_cache: MutableMapping[tuple[str, str, date], pl.DataFrame] | None = None,
) -> StageVerificationResult:
    """Prove `persisted` is exactly what a fresh replay would produce.

    `provider` is the evaluation stage's own recorded provider (already
    proven to equal Twelve Data by the caller); this function independently
    re-derives the exact provider `evaluate_prediction` would have selected
    for `prediction` -- including its conflict hard-fail -- via the same
    `_resolve_price_provider` helper, unconditionally, before ever calling
    `resolve_outcome`, mirroring the producer's own unconditional ordering.
    """
    try:
        selected_provider = _resolve_price_provider(prediction, provider)
    except ValueError as exc:
        raise RefreshVerificationError("evaluation_provider_mismatch", str(exc)) from None

    visited = _VisitedAssets()
    frame_cache = frame_cache if frame_cache is not None else {}

    def _price_loader(subject: str, through_date: date) -> pl.DataFrame:
        return _verifier_price_loader(
            provider=selected_provider,
            subject=subject,
            through_date=through_date,
            evaluation_time=evaluation_time,
            frame_cache=frame_cache,
            visited=visited,
        )

    try:
        resolved = resolve_outcome(
            prediction,
            provider=selected_provider,
            evaluation_date=parent_target_date,
            evaluated_at=evaluation_time,
            benchmark_subject=benchmark_subject,
            price_loader=_price_loader,
        )
    except _VerifierLoadError as exc:
        raise RefreshVerificationError(exc.reason_code, str(exc)) from None

    _require_independent_invariants(
        prediction,
        persisted,
        parent_target_date=parent_target_date,
    )
    _require_replay_match(prediction, persisted, resolved)
    _check_evaluated_at(persisted, evaluation_time=evaluation_time)

    asset_refs: list[AssetRef] = []
    store = open_asset_store()
    for asset in visited.by_id.values():
        asset_refs.append(asset_ref_for(asset))
        raw_ref = _verify_raw_closure(asset, cutoff=evaluation_time, store=store)
        if raw_ref is not None:
            asset_refs.append(raw_ref)

    return StageVerificationResult(
        summary={"prediction_id": str(prediction.pk), "status": persisted.status},
        asset_refs=tuple(asset_refs),
    )


def _verifier_price_loader(
    *,
    provider: str,
    subject: str,
    through_date: date,
    evaluation_time: datetime,
    frame_cache: MutableMapping[tuple[str, str, date], pl.DataFrame],
    visited: _VisitedAssets,
) -> pl.DataFrame:
    cache_key = (provider, subject, through_date)
    cached = frame_cache.get(cache_key)
    if cached is not None:
        return cached
    asof = AsOfData(evaluation_time)
    # A genuine absence of any eligible asset is a real producer-domain
    # outcome (`DataAsset.DoesNotExist` is one of `resolve_outcome`'s own
    # handled exceptions) and must propagate unchanged, not become a
    # verifier-only failure.
    asset = asof.latest_asset(provider=provider, kind=_PRICE_HISTORY_KIND, subject=subject)
    try:
        read = asof.price_frame_for_asset_with_diagnostics(asset=asset, through_date=through_date)
    except PriceFrameSchemaError:
        # A genuine schema violation is one of `resolve_outcome`'s own
        # handled producer-domain exceptions (a `ValueError` subclass) and
        # must replay unresolved unchanged, not become a verifier failure.
        raise
    except ValueError:
        raise _VerifierLoadError(
            "evaluation_outcome_maturity_unreadable",
            "A maturity price asset failed independent kind/cutoff verification",
        ) from None
    except PriceFrameChecksumMismatchError:
        raise _VerifierLoadError(
            "evaluation_outcome_maturity_unreadable",
            "A maturity price asset's stored bytes do not match its registered checksum",
        ) from None
    except OSError:
        raise _VerifierLoadError(
            "evaluation_outcome_maturity_unreadable",
            "A maturity price asset's stored file could not be read",
        ) from None
    except pl.exceptions.PolarsError:
        raise _VerifierLoadError(
            "evaluation_outcome_maturity_unreadable",
            "A maturity price asset's stored bytes could not be parsed as Parquet",
        ) from None
    if read.invalid_session_date_rows:
        raise _VerifierLoadError(
            "evaluation_outcome_maturity_unreadable",
            "A maturity price asset has rows with an invalid session date",
        )
    visited.record(read.asset)
    frame_cache[cache_key] = read.frame
    return read.frame


def _verify_raw_closure(asset: DataAsset, *, cutoff: datetime, store: Any) -> AssetRef | None:
    """Prove `asset`'s declared raw-provider closure, if it declares one.

    A normalized price asset legitimately may or may not declare an
    upstream raw closure (`metadata["raw_asset_id"]`/`["raw_sha256"]`):
    unlike the data-domain `LatestMarketData` check (which always requires
    one), a maturity price asset used only for outcome evaluation is not
    required to carry this link. Both keys absent is the only legitimate
    "no closure declared" case; anything else -- including a partial
    declaration -- is delegated to the reused `raw_price_asset_for`, which
    already fails closed on exactly that (`price_asset_raw_link_missing`)
    as well as every other malformed/mismatched/after-cutoff case.
    """
    metadata = asset.metadata if isinstance(asset.metadata, dict) else {}
    if not metadata.get("raw_asset_id") and not metadata.get("raw_sha256"):
        return None
    raw_asset = raw_price_asset_for(asset, cutoff=cutoff)
    read_checksummed_bytes(store, raw_asset)
    return asset_ref_for(raw_asset)


def _require_independent_invariants(
    prediction: Prediction,
    persisted: PredictionOutcome,
    *,
    parent_target_date: date,
) -> None:
    """Cross-checks a byte-for-byte replay match alone cannot express.

    These bind the persisted row to facts outside `resolve_outcome`'s own
    scope: the parent job's own target date, the withheld-advisory
    non-terminal rule, and the evidence-role/`success` pairing. The
    `research_predictionoutcome_role_*` DB triggers only fire `WHEN
    NEW.status = 'matured'`, and the `outcome_corporate_event_nulls`/
    `outcome_unresolved_nulls` CHECK constraints separately forbid a
    non-`matured` row from carrying a non-null `success` at all, so no
    real write path can currently produce this combination. This check is
    still enforced independently and unconditionally here -- ahead of (and
    regardless of) the shared-resolver replay comparison -- as defense in
    depth: a resolver defect that computed a wrong `success` value would
    otherwise self-authenticate, because both the persisted row and a
    fresh replay would agree on the same wrong value.
    """
    if (
        persisted.success is not None
        and prediction.evidence_role != Prediction.EvidenceRole.DECISION
    ):
        raise RefreshVerificationError(
            "evaluation_outcome_replay_mismatch",
            "A non-decision prediction's persisted outcome carries a non-null success label",
        )
    is_withheld_advisory = (
        prediction.evidence_role == Prediction.EvidenceRole.ADVISORY
        and prediction.bear_return is None
        and prediction.base_return is None
        and prediction.bull_return is None
    )
    if is_withheld_advisory and persisted.status in _TERMINAL_OUTCOME_STATUSES:
        raise RefreshVerificationError(
            "evaluation_outcome_replay_mismatch",
            "A withheld all-null advisory forecast's outcome is terminal",
        )
    if persisted.evaluation_date > parent_target_date:
        raise RefreshVerificationError(
            "evaluation_outcome_date_after_target",
            "An evaluated prediction's outcome observation date is after the target date",
        )
    if persisted.status in _TERMINAL_OUTCOME_STATUSES:
        if persisted.evaluation_date <= prediction.target_date:
            raise RefreshVerificationError(
                "evaluation_outcome_date_before_prediction_target",
                "A terminal outcome's observation date does not come strictly after its own "
                "prediction's target date",
            )
    elif persisted.evaluation_date < prediction.target_date:
        raise RefreshVerificationError(
            "evaluation_outcome_date_before_prediction_target",
            "An evaluated prediction's outcome observation date is before its own "
            "prediction target date",
        )


def _require_replay_match(
    prediction: Prediction, persisted: PredictionOutcome, resolved: ResolvedOutcome
) -> None:
    expected_row = PredictionOutcome(
        prediction=prediction,
        evaluation_date=resolved.evaluation_date,
        status=resolved.status,
        actual_return=resolved.actual_return,
        benchmark_return=resolved.benchmark_return,
        success=resolved.success,
        direction_correct=resolved.direction_correct,
        interval_covered=resolved.interval_covered,
        resolution=resolved.resolution,
        error=resolved.error,
        signed_error=resolved.signed_error,
        metadata=resolved.metadata,
    )
    expected_values = model_row_values(expected_row, OUTCOME_FIELDS)
    actual_values = model_row_values(persisted, OUTCOME_FIELDS)
    if expected_values == actual_values:
        return
    if (
        persisted.status in _TERMINAL_OUTCOME_STATUSES
        and expected_values["evaluation_date"] != actual_values["evaluation_date"]
    ):
        raise RefreshVerificationError(
            "evaluation_outcome_maturity_date_invalid",
            "A terminal outcome's observation date does not match its own prediction's "
            "independently replayed required maturity session",
        )
    raise RefreshVerificationError(
        "evaluation_outcome_replay_mismatch",
        "A persisted outcome does not match an independent replay of its own evidence",
    )


def _check_evaluated_at(persisted: PredictionOutcome, *, evaluation_time: datetime) -> None:
    if persisted.status in _TERMINAL_OUTCOME_STATUSES:
        if persisted.evaluated_at != evaluation_time:
            raise RefreshVerificationError(
                "evaluation_outcome_not_bound_to_execution",
                "A terminal outcome was not produced by the recorded evaluation execution",
            )
        return
    if persisted.evaluated_at > evaluation_time:
        raise RefreshVerificationError(
            "evaluation_outcome_not_bound_to_execution",
            "An unresolved outcome's timestamp is after the recorded evaluation execution",
        )

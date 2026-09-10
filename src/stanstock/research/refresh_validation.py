"""Research-domain scheduled-refresh verification: the analysis-output manifest.

Proves an observed `AnalysisRun`'s complete, exact `StockAnalysis`/
`Prediction` output against its one immutable manifest asset, then proves
the run-level medium/long advisory issuance plan, the medium forecast
panel's declared source closure, and every long-forecast fact/
classification/filing-evidence binding named anywhere in a prediction's
`calculation` payload.

Deliberately **not** called from `core.refresh_verification` yet (Slice
C1). `core`'s existing `_verify_analysis_run`/`_verify_prediction_row`/
`_verify_prediction_matches_analysis`/`_iter_source_asset_entries`/
`_merge_asset_identity`/`_require_registered_asset_identity_matches`
remain untouched and still run as the current production safety net; the
exact minimal integration hook for retiring them in favor of this module
is documented in this slice's handoff report, not implemented here.

Unlike that legacy path, this module never infers whether medium/long
advisory issuance was "used" for a run from which prediction rows happen
to exist (a mutable-row inference the architecture explicitly rejects):
the caller supplies `medium` / `long` lane expectations derived from the
frozen scoring/forecast configuration and run gating policy, not from any
row content.
"""

from __future__ import annotations

import io
import uuid
from collections.abc import Callable, Mapping
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
from stanstock.data.assets import (
    asset_ref_for,
    open_asset_store,
    read_checksummed_bytes,
    resolve_asset_ref,
)
from stanstock.data.models import (
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    Listing,
)
from stanstock.data.provider_policy import TWELVE_DATA_PROVIDER
from stanstock.research.config import SCORE_HORIZONS
from stanstock.research.forecasting import FORECAST_SCENARIO_SCHEMA_VERSION, infer_price_source
from stanstock.research.long_forecasts import (
    classification_payload,
    fact_payload,
    fact_reference,
    referenced_evidence_fact_ids,
)
from stanstock.research.medium_forecasts import (
    PANEL_KIND,
    PANEL_PROVIDER,
    PANEL_SCHEMA,
    PANEL_SCHEMA_VERSION,
    asset_identity,
    calendar_sessions_through,
    dedupe_assets,
    hash_json,
)
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.provenance import DATA_MODE_PROVIDER, source_data_mode
from stanstock.research.refresh_evidence import (
    ANALYSIS_RUN_FIELDS,
    ANALYSIS_RUN_MODEL,
    LONG_HORIZONS,
    MEDIUM_HORIZONS,
    PREDICTION_FIELDS,
    PREDICTION_MODEL,
    STOCK_ANALYSIS_FIELDS,
    STOCK_ANALYSIS_MODEL,
    ManifestPayloadError,
    ManifestPlan,
    actual_output_plan,
    build_output_plan,
    decimal_from_float,
    lookup_manifest,
    model_row_values,
    optional_decimal_from_float,
    parse_manifest_envelope,
    row_digest,
)
from stanstock.research.timing import is_us_session_issuance_on_time

_MEDIUM_HORIZONS = MEDIUM_HORIZONS
_LONG_HORIZONS = LONG_HORIZONS
_ADVISORY_HORIZONS = _MEDIUM_HORIZONS | _LONG_HORIZONS
_PRICE_HISTORY_KIND = "price_history"


@dataclass(slots=True)
class _AssetRegistry:
    """Every distinct asset referenced anywhere in one verified run's output.

    Tracks both the claimed identity (`AssetRef`, as declared by whichever
    payload named it) and the resolved immutable `DataAsset` row, so a
    single final pass can checksum-read each distinct asset's physical
    bytes exactly once (finding 9) after every identity/cutoff check has
    already passed.
    """

    refs: dict[UUID, AssetRef] = field(default_factory=dict)
    rows: dict[UUID, DataAsset] = field(default_factory=dict)

    def add(self, ref: AssetRef, asset: DataAsset) -> None:
        existing = self.refs.get(ref.id)
        if existing is not None and existing != ref:
            raise RefreshVerificationError(
                "source_assets_entry_conflicting",
                "A referenced asset conflicts with an already-declared identity for the "
                "same asset id",
            )
        self.refs[ref.id] = ref
        self.rows[ref.id] = asset

    def resolve(self, ref: AssetRef, *, cutoff: datetime) -> DataAsset:
        asset = resolve_asset_ref(ref, cutoff=cutoff)
        self.add(ref, asset)
        return asset

    def sorted_refs(self) -> tuple[AssetRef, ...]:
        return tuple(self.refs[key] for key in sorted(self.refs, key=str))


@dataclass(frozen=True, slots=True)
class MediumLaneExpectation:
    """The reviewed medium-forecast configuration a panel must bind to.

    Every field here is read from the frozen `MediumForecastConfig` plus
    the reviewed `analyze_snapshot(provider=..., benchmark_subject=...)`
    call arguments -- never inferred from the panel asset's own
    self-reported metadata -- so the verifier can independently resolve
    the panel's exact expected calendar hash, source-asset identities, and
    metadata shape rather than merely checking that panel for internal
    self-consistency.
    """

    config_hash: str
    method_version: str
    calendar: str
    fixed_epoch: date
    provider: str
    benchmark_subject: str
    return_basis: str
    dividends_included: bool


@dataclass(frozen=True, slots=True)
class LongLaneExpectation:
    """The reviewed long-forecast configuration long predictions must bind to."""

    config_hash: str
    method_version: str


def _require_manifest_matches_current_rows(
    run: AnalysisRun,
    stock_analyses: list[StockAnalysis],
    predictions: list[Prediction],
    manifest_run_id: str,
    manifest_entries: Mapping[tuple[str, str], str],
) -> None:
    if manifest_run_id != str(run.id):
        raise RefreshVerificationError(
            "analysis_output_manifest_run_mismatch",
            "Analysis output manifest run_id does not match the verified analysis run",
        )
    current: dict[tuple[str, str], str] = {
        (ANALYSIS_RUN_MODEL, str(run.id)): row_digest(
            ANALYSIS_RUN_MODEL, model_row_values(run, ANALYSIS_RUN_FIELDS)
        )
    }
    for analysis in stock_analyses:
        current[(STOCK_ANALYSIS_MODEL, str(analysis.id))] = row_digest(
            STOCK_ANALYSIS_MODEL, model_row_values(analysis, STOCK_ANALYSIS_FIELDS)
        )
    for prediction in predictions:
        current[(PREDICTION_MODEL, str(prediction.id))] = row_digest(
            PREDICTION_MODEL, model_row_values(prediction, PREDICTION_FIELDS)
        )
    if dict(manifest_entries) != current:
        raise RefreshVerificationError(
            "analysis_output_manifest_diverged",
            "The analysis output manifest does not reproduce the exact current set of "
            "AnalysisRun/StockAnalysis/Prediction rows and their complete content",
        )


def _resolve_manifest(
    run: AnalysisRun,
) -> tuple[AssetRef, str, ManifestPlan, Mapping[tuple[str, str], str]]:
    lookup = lookup_manifest(run.id)
    if lookup.count == 0:
        raise RefreshVerificationError(
            "analysis_output_manifest_missing",
            "The verified analysis run has no immutable analysis output manifest asset",
        )
    if lookup.count > 1:
        raise RefreshVerificationError(
            "analysis_output_manifest_ambiguous",
            "The verified analysis run has more than one immutable analysis output manifest asset",
        )
    asset = lookup.asset
    assert asset is not None
    if asset.retrieved_at != run.generated_at or asset.available_at != run.generated_at:
        raise RefreshVerificationError(
            "analysis_output_manifest_identity_invalid",
            "The analysis output manifest asset's own timestamps do not match this run's own "
            "output time",
        )
    store = open_asset_store()
    payload = read_checksummed_bytes(store, asset)
    try:
        manifest_run_id, plan, entries = parse_manifest_envelope(payload)
    except ManifestPayloadError as exc:
        raise RefreshVerificationError(
            "analysis_output_manifest_malformed",
            "The analysis output manifest asset is not a valid manifest envelope",
        ) from exc
    entry_map = {(entry.model, entry.row_id): entry.digest for entry in entries}
    return (
        AssetRef(
            id=asset.id,
            provider=asset.provider,
            kind=asset.kind,
            subject=asset.subject,
            sha256=asset.sha256,
        ),
        manifest_run_id,
        plan,
        entry_map,
    )


def _require_prediction_horizon_multiset(
    stock_analyses: list[StockAnalysis],
    predictions: list[Prediction],
    *,
    decision_horizons: frozenset[str],
    medium: MediumLaneExpectation | None,
    long: LongLaneExpectation | None,
    eligible_listing_ids: set[UUID],
) -> dict[str, list[Prediction]]:
    for prediction in predictions:
        if prediction.listing_id != prediction.analysis.listing_id:
            raise RefreshVerificationError(
                "prediction_listing_mismatch",
                "A prediction's listing does not match its own analysis's listing",
            )
        if prediction.listing_id not in eligible_listing_ids:
            raise RefreshVerificationError(
                "prediction_listing_ineligible",
                "A prediction's listing is outside the verified eligible universe membership",
            )
    by_analysis: dict[str, list[Prediction]] = {}
    for prediction in predictions:
        by_analysis.setdefault(str(prediction.analysis_id), []).append(prediction)
    expected_medium = _MEDIUM_HORIZONS if medium is not None else frozenset()
    expected_long = _LONG_HORIZONS if long is not None else frozenset()
    for analysis in stock_analyses:
        rows = by_analysis.get(str(analysis.id), [])
        keys = [(p.evidence_role, p.horizon) for p in rows]
        if len(set(keys)) != len(keys):
            raise RefreshVerificationError(
                "prediction_role_horizon_duplicated",
                "A StockAnalysis carries more than one prediction for the same "
                "(evidence_role, horizon) pair",
            )
        decision = {p.horizon for p in rows if p.evidence_role == Prediction.EvidenceRole.DECISION}
        advisory = {p.horizon for p in rows if p.evidence_role == Prediction.EvidenceRole.ADVISORY}
        if decision != set(decision_horizons):
            raise RefreshVerificationError(
                "prediction_decision_multiset_mismatch",
                "A StockAnalysis does not carry exactly the configured decision horizon set",
            )
        if (advisory & _MEDIUM_HORIZONS) != set(expected_medium):
            raise RefreshVerificationError(
                "prediction_medium_multiset_mismatch",
                "A StockAnalysis does not carry exactly the planned medium advisory horizon set",
            )
        if (advisory & _LONG_HORIZONS) != set(expected_long):
            raise RefreshVerificationError(
                "prediction_long_multiset_mismatch",
                "A StockAnalysis does not carry exactly the planned long advisory horizon set",
            )
        if advisory - _ADVISORY_HORIZONS:
            raise RefreshVerificationError(
                "prediction_evidence_role_invalid",
                "A StockAnalysis carries an advisory prediction with an unsupported horizon",
            )
    return by_analysis


def _require_listing_set_matches(
    stock_analyses: list[StockAnalysis], *, eligible_listing_ids: set[UUID]
) -> None:
    listing_ids = [analysis.listing_id for analysis in stock_analyses]
    if len(set(listing_ids)) != len(listing_ids):
        raise RefreshVerificationError(
            "stock_analysis_listing_duplicated",
            "A verified analysis run carries more than one StockAnalysis for the same listing",
        )
    if set(listing_ids) != eligible_listing_ids:
        raise RefreshVerificationError(
            "stock_analysis_listing_mismatch",
            "Verified StockAnalysis listings do not match the exact eligible membership set",
        )


def _scenario_document_horizons(document: object) -> dict[str, Any]:
    """The canonical document's own `horizons` mapping, requiring the exact
    schema (`schema_version` match, `horizons` present as a mapping) shared
    by every scenario-document check in this module -- so a missing
    canonical entry fails closed instead of silently resolving to a legacy
    mirror field (the fail-open gap this replaces)."""
    if not isinstance(document, dict) or document.get("schema_version") != (
        FORECAST_SCENARIO_SCHEMA_VERSION
    ):
        raise RefreshVerificationError(
            "stock_analysis_scenario_document_invalid",
            "StockAnalysis forecast_scenarios is not a valid canonical scenario document",
        )
    horizons = document.get("horizons")
    if not isinstance(horizons, dict):
        raise RefreshVerificationError(
            "stock_analysis_scenario_document_invalid",
            "StockAnalysis forecast_scenarios has no canonical horizons mapping",
        )
    return horizons


def _canonical_scenario_for_horizon(document: object, horizon: str) -> dict[str, Any]:
    """The exact canonical scenario dict for `horizon`, with no legacy fallback."""
    horizons = _scenario_document_horizons(document)
    scenario = horizons.get(horizon)
    if not isinstance(scenario, dict):
        raise RefreshVerificationError(
            "stock_analysis_scenario_missing",
            f"StockAnalysis forecast_scenarios has no canonical entry for horizon {horizon!r}",
        )
    return scenario


_SCENARIO_RETURN_TRIPLET = ("bear", "base", "bull")
_SCENARIO_RETURN_FIELDS = (*_SCENARIO_RETURN_TRIPLET, "probability_positive")
_SCENARIO_REQUIRED_FIELDS = (
    *_SCENARIO_RETURN_FIELDS,
    "confidence",
    "confidence_status",
    "insufficiency_reason",
)


def _require_canonical_scenario_shape(scenario: object, *, horizon: str) -> None:
    """Exact shape/type proof for one canonical scenario dict.

    All seven common persisted fields (`bear`, `base`, `bull`,
    `probability_positive`, `confidence`, `confidence_status`,
    `insufficiency_reason`) must be explicitly present with a valid type --
    never defaulted via `.get()`, which would conflate a missing key with an
    explicit null/zero (finding 3). `bear`/`base`/`bull` are one atomic
    return-distribution triplet that must be uniformly null (fully withheld)
    or uniformly numeric; a partial-null mix is rejected. `probability_positive`
    is independently nullable *only when the triplet itself is numeric*
    (e.g. an empirically insufficient sample for a win-rate estimate, while
    the point-estimate return distribution itself is still evaluable) --
    a probability without any underlying return distribution is rejected.
    """
    if not isinstance(scenario, dict):
        raise RefreshVerificationError(
            "stock_analysis_scenario_shape_invalid",
            f"StockAnalysis forecast_scenarios horizon {horizon!r} is not an object",
        )
    for field_name in _SCENARIO_REQUIRED_FIELDS:
        if field_name not in scenario:
            raise RefreshVerificationError(
                "stock_analysis_scenario_shape_invalid",
                f"StockAnalysis forecast_scenarios horizon {horizon!r} is missing its own "
                f"required {field_name!r} field",
            )
    return_values = [scenario[field_name] for field_name in _SCENARIO_RETURN_FIELDS]
    for field_name, value in zip(_SCENARIO_RETURN_FIELDS, return_values, strict=True):
        if isinstance(value, bool) or not (value is None or isinstance(value, (int, float))):
            raise RefreshVerificationError(
                "stock_analysis_scenario_shape_invalid",
                f"StockAnalysis forecast_scenarios horizon {horizon!r} field {field_name!r} is "
                "not a number or null",
            )
    return_triplet = [scenario[field_name] for field_name in _SCENARIO_RETURN_TRIPLET]
    if any(value is None for value in return_triplet) and any(
        value is not None for value in return_triplet
    ):
        raise RefreshVerificationError(
            "stock_analysis_scenario_partial_null",
            f"StockAnalysis forecast_scenarios horizon {horizon!r} mixes null and non-null "
            "return fields; an explicit non-evaluable scenario must be complete",
        )
    if (
        all(value is None for value in return_triplet)
        and scenario["probability_positive"] is not None
    ):
        raise RefreshVerificationError(
            "stock_analysis_scenario_partial_null",
            f"StockAnalysis forecast_scenarios horizon {horizon!r} carries a probability "
            "estimate without any underlying return distribution",
        )
    confidence = scenario["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise RefreshVerificationError(
            "stock_analysis_scenario_shape_invalid",
            f"StockAnalysis forecast_scenarios horizon {horizon!r} confidence is not a number",
        )
    confidence_status = scenario["confidence_status"]
    if not isinstance(confidence_status, str) or not confidence_status:
        raise RefreshVerificationError(
            "stock_analysis_scenario_shape_invalid",
            f"StockAnalysis forecast_scenarios horizon {horizon!r} confidence_status is not a "
            "non-empty string",
        )
    insufficiency_reason = scenario["insufficiency_reason"]
    if not isinstance(insufficiency_reason, str):
        raise RefreshVerificationError(
            "stock_analysis_scenario_shape_invalid",
            f"StockAnalysis forecast_scenarios horizon {horizon!r} insufficiency_reason is "
            "not a string",
        )


def _require_scenario_document_shape(
    analysis: StockAnalysis, *, medium_active: bool, long_active: bool
) -> None:
    """Exact top-level shape and exact horizon-key set for one StockAnalysis's
    canonical `forecast_scenarios` document (finding 3).

    `short`/`medium`/`long` are *always* present -- `build_scenarios`
    unconditionally computes all three `SCORE_HORIZONS` (an unsupported one
    gets an explicit `unsupported_by_model` placeholder, never a persisted
    decision `Prediction` row), so their presence is independent of this
    run's own configured decision horizon subset. `6m`/`12m`/`3y`/`5y` are
    *only* present when the medium/long advisory lane is active for this
    run -- and, when active, always paired with their own planned advisory
    `Prediction` rows (`append_advisory_predictions`/
    `append_long_advisory_predictions` always create both of a lane's
    horizons together). An unknown, inactive-lane, or missing key is
    rejected by requiring exact set equality against this derived
    expectation rather than validating presence alone.
    """
    document = analysis.forecast_scenarios
    horizons = _scenario_document_horizons(document)
    expected_horizons = (
        set(SCORE_HORIZONS)
        | (_MEDIUM_HORIZONS if medium_active else set())
        | (_LONG_HORIZONS if long_active else set())
    )
    if set(horizons) != expected_horizons:
        raise RefreshVerificationError(
            "stock_analysis_scenario_horizon_set_mismatch",
            "StockAnalysis forecast_scenarios horizons does not exactly match the always-"
            "present score horizons plus this run's own active advisory lane horizons",
        )
    for horizon, scenario in horizons.items():
        _require_canonical_scenario_shape(scenario, horizon=horizon)


def _require_scenario_mirror_consistency(analysis: StockAnalysis) -> None:
    legacy_mirrors = {
        "short": analysis.short_scenario,
        "medium": analysis.medium_scenario,
        "long": analysis.long_scenario,
    }
    for horizon, legacy_value in legacy_mirrors.items():
        canonical = _canonical_scenario_for_horizon(analysis.forecast_scenarios, horizon)
        if canonical != legacy_value:
            raise RefreshVerificationError(
                "stock_analysis_scenario_mirror_mismatch",
                f"StockAnalysis {analysis.id} canonical forecast_scenarios does not agree "
                f"with its own legacy {horizon!r} mirror field",
            )


def _asset_ref_from_source_entry(entry: object, *, context: str) -> AssetRef:
    if not isinstance(entry, dict):
        raise RefreshVerificationError(
            "source_assets_entry_invalid", f"{context} source_assets entry is not an object"
        )
    identity = {
        field: entry.get(field) for field in ("id", "provider", "kind", "subject", "sha256")
    }
    try:
        return AssetRef.from_json(identity)
    except RefreshVerificationError as exc:
        raise RefreshVerificationError(
            "source_assets_entry_malformed",
            f"{context} source_assets entry identity is invalid",
        ) from exc


def _require_source_assets_bound(
    raw: object, *, context: str, cutoff: datetime, registry: _AssetRegistry
) -> list[AssetRef]:
    if not isinstance(raw, list) or not raw:
        raise RefreshVerificationError(
            "source_assets_payload_invalid",
            f"{context} source_assets must be a non-empty list for a provider-backed observed run",
        )
    refs: list[AssetRef] = []
    seen: set[UUID] = set()
    for entry in raw:
        ref = _asset_ref_from_source_entry(entry, context=context)
        if ref.id in seen:
            raise RefreshVerificationError(
                "source_assets_entry_duplicated",
                f"{context} source_assets declares the same asset id more than once",
            )
        seen.add(ref.id)
        registry.resolve(ref, cutoff=cutoff)
        refs.append(ref)
    return sorted(refs, key=lambda ref: str(ref.id))


def _require_stock_analysis_price_source_bound(
    analysis: StockAnalysis, *, raw_source_assets: list[dict[str, Any]], refs: list[AssetRef]
) -> None:
    """Bind a StockAnalysis's own declared price source to exactly one of
    its own declared source asset references, and to its own listing.

    Reuses the same pure `infer_price_source` filter the production
    forecasting path itself uses (by `kind == "price_history"` and subject
    membership), rather than inventing a divergent formula, so this check
    and the production behavior it is proving can never quietly diverge.
    """
    listing = analysis.listing
    provider, subject = infer_price_source(
        raw_source_assets,
        subjects=(listing.provider_symbol or listing.ticker, listing.ticker),
    )
    if not provider or not subject:
        raise RefreshVerificationError(
            "stock_analysis_price_source_unbound",
            f"StockAnalysis {analysis.id} source_assets do not resolve to exactly one "
            "price_history asset for its own listing",
        )
    matches = [ref for ref in refs if ref.provider == provider and ref.subject == subject]
    if len(matches) != 1 or matches[0].kind != "price_history":
        raise RefreshVerificationError(
            "stock_analysis_price_source_mismatch",
            f"StockAnalysis {analysis.id} inferred price source does not resolve to exactly "
            "one of its own declared source asset references",
        )
    price_source = analysis.data_quality.get("price_source")
    matched = matches[0]
    if (
        not isinstance(price_source, dict)
        or set(price_source) != {"asset_id", "provider", "subject"}
        or str(price_source.get("asset_id")) != str(matched.id)
        or price_source.get("provider") != matched.provider
        or price_source.get("subject") != matched.subject
    ):
        raise RefreshVerificationError(
            "stock_analysis_price_source_mismatch",
            f"StockAnalysis {analysis.id} recorded price_source does not match its own "
            "resolved price source asset reference",
        )


def _require_prediction_ground_truth(
    prediction: Prediction,
    *,
    target_date: Any,
    code_revision: str,
    scoring_config_version: str,
    scoring_config_hash: str,
    decision_horizons: frozenset[str],
    medium: MediumLaneExpectation | None,
    long: LongLaneExpectation | None,
    listing: Listing,
) -> None:
    if prediction.target_date != target_date:
        raise RefreshVerificationError(
            "prediction_target_mismatch", "A prediction targets a different date"
        )
    if not prediction.issued_on_time or not is_us_session_issuance_on_time(
        target_date=target_date, generated_at=prediction.generated_at
    ):
        raise RefreshVerificationError(
            "prediction_not_on_time", "A prediction was not issued on time"
        )
    if prediction.evidence_grade != prediction.analysis.run.universe_snapshot.grade:
        raise RefreshVerificationError(
            "prediction_evidence_grade_mismatch",
            "A prediction's evidence_grade does not match its own run's universe snapshot grade",
        )
    if prediction.code_revision != code_revision:
        raise RefreshVerificationError(
            "prediction_code_revision_mismatch",
            "A prediction's code revision does not match the verified refresh revision",
        )
    recomputed_mode = source_data_mode({"source_assets": prediction.source_assets})
    if recomputed_mode != prediction.source_mode or recomputed_mode != DATA_MODE_PROVIDER:
        raise RefreshVerificationError(
            "prediction_source_mode_invalid",
            "A prediction's source_mode is not a genuine provider-backed derivation of its "
            "own declared source_assets",
        )
    if prediction.price_provider != TWELVE_DATA_PROVIDER:
        raise RefreshVerificationError(
            "prediction_price_provider_mismatch",
            "A prediction's price provider does not match Twelve Data",
        )
    if prediction.price_subject != listing.provider_symbol:
        raise RefreshVerificationError(
            "prediction_price_subject_mismatch",
            "A prediction's price subject does not match its own listing's provider symbol",
        )
    horizon = prediction.horizon
    role = prediction.evidence_role
    if role == Prediction.EvidenceRole.DECISION:
        if horizon not in decision_horizons:
            raise RefreshVerificationError(
                "prediction_horizon_role_mismatch",
                "A decision prediction carries a horizon its role does not support",
            )
        if (
            prediction.config_hash != scoring_config_hash
            or prediction.method_version != scoring_config_version
        ):
            raise RefreshVerificationError(
                "prediction_decision_config_mismatch",
                "A decision prediction's configuration does not match the reviewed scoring "
                "configuration",
            )
    elif role == Prediction.EvidenceRole.ADVISORY:
        if horizon in _MEDIUM_HORIZONS:
            # `_require_prediction_horizon_multiset` already proved the medium lane is
            # active whenever a medium-horizon prediction survives to this point.
            assert medium is not None
            if prediction.config_hash != medium.config_hash:
                raise RefreshVerificationError(
                    "prediction_advisory_config_mismatch",
                    "A medium advisory prediction's configuration does not match the "
                    "reviewed medium forecast configuration",
                )
            if prediction.method_version != medium.method_version:
                raise RefreshVerificationError(
                    "prediction_advisory_method_version_mismatch",
                    "A medium advisory prediction's method_version does not match the "
                    "reviewed medium forecast configuration version",
                )
        elif horizon in _LONG_HORIZONS:
            # `_require_prediction_horizon_multiset` already proved the long lane is
            # active whenever a long-horizon prediction survives to this point.
            assert long is not None
            if prediction.config_hash != long.config_hash:
                raise RefreshVerificationError(
                    "prediction_advisory_config_mismatch",
                    "A long advisory prediction's configuration does not match the reviewed "
                    "long forecast configuration",
                )
            if prediction.method_version != long.method_version:
                raise RefreshVerificationError(
                    "prediction_advisory_method_version_mismatch",
                    "A long advisory prediction's method_version does not match the reviewed "
                    "long forecast configuration version",
                )
        else:
            raise RefreshVerificationError(
                "prediction_horizon_role_mismatch",
                "An advisory prediction carries a horizon its role does not support",
            )
    else:
        raise RefreshVerificationError(
            "prediction_evidence_role_invalid", "A prediction carries an unsupported evidence role"
        )


def _require_prediction_matches_analysis(prediction: Prediction, analysis: StockAnalysis) -> None:
    context = f"StockAnalysis {analysis.id}"
    if prediction.price_at_prediction != analysis.current_price:
        raise RefreshVerificationError(
            "stock_analysis_current_price_mismatch",
            f"{context} current_price does not match its own immutable prediction ledger",
        )
    if prediction.recommendation != analysis.recommendation:
        raise RefreshVerificationError(
            "stock_analysis_recommendation_mismatch",
            f"{context} recommendation does not match its own immutable prediction ledger",
        )
    if prediction.overall_score != analysis.overall_score:
        raise RefreshVerificationError(
            "stock_analysis_overall_score_mismatch",
            f"{context} overall_score does not match its own immutable prediction ledger",
        )
    if prediction.component_scores != analysis.component_scores:
        raise RefreshVerificationError(
            "stock_analysis_component_scores_mismatch",
            f"{context} component_scores does not match its own immutable prediction ledger",
        )
    scenario = _canonical_scenario_for_horizon(analysis.forecast_scenarios, prediction.horizon)
    _require_canonical_scenario_shape(scenario, horizon=prediction.horizon)
    expected = {
        "bear_return": optional_decimal_from_float(scenario["bear"], places=4),
        "base_return": optional_decimal_from_float(scenario["base"], places=4),
        "bull_return": optional_decimal_from_float(scenario["bull"], places=4),
        "probability_positive": optional_decimal_from_float(
            scenario["probability_positive"], places=4
        ),
        "confidence": decimal_from_float(scenario["confidence"], places=2),
    }
    for scenario_field, expected_value in expected.items():
        if getattr(prediction, scenario_field) != expected_value:
            raise RefreshVerificationError(
                "stock_analysis_scenario_mismatch",
                f"{context} scenario value for {scenario_field!r} at horizon "
                f"{prediction.horizon!r} does not match its own immutable prediction ledger",
            )
    if prediction.confidence_status != scenario["confidence_status"]:
        raise RefreshVerificationError(
            "stock_analysis_scenario_mismatch",
            f"{context} confidence_status at horizon {prediction.horizon!r} does not match "
            "its own immutable prediction ledger",
        )
    if prediction.insufficiency_reason != scenario["insufficiency_reason"]:
        raise RefreshVerificationError(
            "stock_analysis_scenario_mismatch",
            f"{context} insufficiency_reason at horizon {prediction.horizon!r} does not "
            "match its own immutable prediction ledger",
        )


def _require_decision_source_closure_exact(
    *, own_ids: set[UUID], analysis_ref_ids: set[UUID]
) -> None:
    """A decision prediction's own declared `source_assets` must equal its
    owning StockAnalysis's own declared `source_assets` exactly -- no
    arbitrary extra reference on either side."""
    if own_ids != analysis_ref_ids:
        raise RefreshVerificationError(
            "decision_prediction_source_assets_mismatch",
            "A decision prediction's own declared source_assets does not exactly match its "
            "StockAnalysis's own declared source assets",
        )


def _resolve_authoritative_medium_panel_asset(*, run: AnalysisRun) -> DataAsset:
    """Exactly one authoritative `medium_forecast_panel` asset may exist
    for this run's own subject (`str(run.id)`). A prediction's own
    `calculation["panel_asset_id"]` is never trusted to *pick* the panel --
    it is only ever compared against this independently, directly
    resolved row, so a duplicate same-content panel registered under an
    alternate id can never be silently accepted as interchangeable."""
    candidates = list(
        DataAsset.objects.filter(provider=PANEL_PROVIDER, kind=PANEL_KIND, subject=str(run.id))
    )
    if not candidates:
        raise RefreshVerificationError(
            "medium_panel_missing",
            "No medium forecast panel asset is registered for this run",
        )
    if len(candidates) > 1:
        raise RefreshVerificationError(
            "medium_panel_ambiguous",
            "More than one medium forecast panel asset is registered for this run's own subject",
        )
    return candidates[0]


_MEDIUM_PANEL_METADATA_LIST_FIELDS = frozenset({"source_assets"})
_MEDIUM_PANEL_METADATA_TEXT_PRESENCE_FIELDS = frozenset(
    {"calendar_library_version", "panel_library_version"}
)


def _require_medium_panel_metadata(
    panel_asset: DataAsset,
    *,
    medium: MediumLaneExpectation,
    run: AnalysisRun,
    scoring_config_version: str,
) -> Mapping[str, Any]:
    """Validate the panel's complete, exact metadata shape and every
    scalar value it can be checked against the reviewed configuration --
    including a `calendar_hash` independently recomputed from the frozen
    calendar and fixed epoch, never merely trusted from the panel's own
    self-reported value."""
    metadata = panel_asset.metadata
    if not isinstance(metadata, dict):
        raise RefreshVerificationError(
            "medium_panel_config_mismatch", "A medium forecast panel has no recorded metadata"
        )
    universe_snapshot = run.universe_snapshot
    try:
        calendar_sessions = calendar_sessions_through(
            calendar_name=medium.calendar,
            fixed_epoch=medium.fixed_epoch,
            target_date=run.target_date,
        )
    except ValueError as exc:
        raise RefreshVerificationError(
            "medium_panel_config_mismatch",
            "The reviewed medium forecast calendar cannot reach this run's own target date "
            "from its fixed epoch",
        ) from exc
    expected_calendar_hash = hash_json([session.isoformat() for session in calendar_sessions])
    expected_scalars: dict[str, object] = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "method_version": medium.method_version,
        "config_hash": medium.config_hash,
        "code_revision": run.code_revision,
        "calendar": medium.calendar,
        "fixed_epoch": medium.fixed_epoch.isoformat(),
        "calendar_hash": expected_calendar_hash,
        "target_date": run.target_date.isoformat(),
        "universe_snapshot_id": str(universe_snapshot.id),
        "universe_slug": universe_snapshot.universe.slug,
        "universe_config_hash": universe_snapshot.config_hash,
        "scoring_config_version": scoring_config_version,
        "scoring_config_hash": run.config_hash,
        "return_definition": medium.return_basis,
        "dividends_included": medium.dividends_included,
        "training_evidence_grade": "research",
        "current_universe_survivorship_bias": True,
        "usage_scope": "private_single_user_research",
        "content_sha256": panel_asset.sha256,
    }
    expected_keys = (
        set(expected_scalars)
        | _MEDIUM_PANEL_METADATA_TEXT_PRESENCE_FIELDS
        | _MEDIUM_PANEL_METADATA_LIST_FIELDS
        | {"row_count", "source_manifest_hash", "evidence_bundle_hash"}
    )
    if set(metadata) != expected_keys:
        raise RefreshVerificationError(
            "medium_panel_config_mismatch",
            "A medium forecast panel's metadata does not have the exact expected shape",
        )
    for expected_field, expected_value in expected_scalars.items():
        if metadata.get(expected_field) != expected_value:
            reason = (
                "medium_panel_content_checksum_mismatch"
                if expected_field == "content_sha256"
                else "medium_panel_config_mismatch"
            )
            raise RefreshVerificationError(
                reason,
                "A medium forecast panel does not bind the reviewed medium forecast "
                "configuration, scoring configuration, universe, code revision, or calendar",
            )
    for text_field in _MEDIUM_PANEL_METADATA_TEXT_PRESENCE_FIELDS:
        if not isinstance(metadata.get(text_field), str) or not metadata[text_field]:
            raise RefreshVerificationError(
                "medium_panel_config_mismatch",
                f"A medium forecast panel's {text_field} metadata is missing or blank",
            )
    row_count = metadata.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count <= 0:
        raise RefreshVerificationError(
            "medium_panel_config_mismatch",
            "A medium forecast panel's recorded row_count is not a positive integer",
        )
    return metadata


def _require_medium_panel_benchmark_asset(
    metadata: Mapping[str, Any], *, medium: MediumLaneExpectation, cutoff: datetime
) -> DataAsset:
    """Resolve the one benchmark price asset this panel's own declared
    source closure names, then independently prove it really is a
    cutoff-safe `price_history` asset for the reviewed benchmark identity
    -- the panel's self-reported closure is used only to *locate* the
    candidate id (an as-of read may have picked any cutoff-eligible
    vintage), never to establish its identity or eligibility."""
    source_assets = metadata.get("source_assets")
    if not isinstance(source_assets, list) or not source_assets:
        raise RefreshVerificationError(
            "medium_panel_source_assets_invalid",
            "A medium forecast panel has no recorded source asset closure",
        )
    matches: list[UUID] = []
    for entry in source_assets:
        if not isinstance(entry, dict):
            raise RefreshVerificationError(
                "medium_panel_source_assets_invalid",
                "A medium forecast panel source asset closure entry is not an object",
            )
        if (
            entry.get("provider") == medium.provider
            and entry.get("kind") == _PRICE_HISTORY_KIND
            and entry.get("subject") == medium.benchmark_subject
        ):
            matches.append(
                _parse_uuid(
                    entry.get("id"),
                    reason="medium_panel_source_assets_invalid",
                    message="A medium forecast panel benchmark reference id is not a valid "
                    "identifier",
                )
            )
    if len(matches) != 1:
        raise RefreshVerificationError(
            "medium_panel_benchmark_reference_invalid",
            "A medium forecast panel does not declare exactly one benchmark price asset "
            "reference matching the reviewed benchmark identity",
        )
    try:
        asset = DataAsset.objects.get(pk=matches[0])
    except DataAsset.DoesNotExist as exc:
        raise RefreshVerificationError(
            "medium_panel_benchmark_reference_invalid",
            "A medium forecast panel's declared benchmark asset does not resolve to a "
            "registered asset",
        ) from exc
    if (
        asset.provider != medium.provider
        or asset.kind != _PRICE_HISTORY_KIND
        or asset.subject != medium.benchmark_subject
    ):
        raise RefreshVerificationError(
            "medium_panel_benchmark_reference_invalid",
            "A medium forecast panel's declared benchmark asset does not match the reviewed "
            "benchmark identity",
        )
    _require_cutoff_safe(
        asset,
        cutoff=cutoff,
        reason="medium_panel_benchmark_not_cutoff_safe",
        message="A medium forecast panel's benchmark asset is not cutoff-safe for this run",
    )
    return asset


def _require_medium_panel_row_price_asset(
    price_asset_id: str, *, listing: Listing, medium: MediumLaneExpectation, cutoff: datetime
) -> DataAsset:
    price_asset_uuid = _parse_uuid(
        price_asset_id,
        reason="medium_panel_row_price_asset_malformed",
        message="A medium forecast panel row's price_asset_id is not a valid identifier",
    )
    try:
        asset = DataAsset.objects.get(pk=price_asset_uuid)
    except DataAsset.DoesNotExist as exc:
        raise RefreshVerificationError(
            "medium_panel_row_price_asset_missing",
            "A medium forecast panel row's price_asset_id does not resolve to a registered asset",
        ) from exc
    expected_subject = listing.provider_symbol or listing.ticker
    if (
        asset.provider != medium.provider
        or asset.kind != _PRICE_HISTORY_KIND
        or asset.subject != expected_subject
    ):
        raise RefreshVerificationError(
            "medium_panel_row_price_asset_mismatch",
            "A medium forecast panel row's price asset does not belong to its own row's listing",
        )
    _require_cutoff_safe(
        asset,
        cutoff=cutoff,
        reason="medium_panel_row_price_asset_not_cutoff_safe",
        message="A medium forecast panel row's price asset is not cutoff-safe for this run",
    )
    return asset


def _require_medium_panel_content(
    panel_asset: DataAsset,
    *,
    metadata: Mapping[str, Any],
    medium: MediumLaneExpectation,
    eligible_listings: Mapping[UUID, Listing],
    cutoff: datetime,
) -> None:
    """Physically checksum-read and parse the panel's own Parquet bytes,
    then prove its schema, row count, per-listing forecast-row coverage,
    and every declared `price_asset_id` against the run's own authoritative
    eligible membership -- never against the panel's own metadata -- and
    only then compare the source-manifest/evidence-bundle hashes against
    that independently derived closure."""
    store = open_asset_store()
    payload = read_checksummed_bytes(store, panel_asset)
    try:
        frame = pl.read_parquet(io.BytesIO(payload))
    except pl.exceptions.ComputeError:
        raise RefreshVerificationError(
            "medium_panel_content_unparseable",
            "A medium forecast panel's physical bytes could not be parsed as its own "
            "declared Parquet content",
        ) from None
    if dict(frame.schema) != PANEL_SCHEMA:
        raise RefreshVerificationError(
            "medium_panel_schema_mismatch",
            "A medium forecast panel's physical Parquet schema does not match the expected "
            "medium forecast panel schema",
        )
    if frame.height != metadata.get("row_count"):
        raise RefreshVerificationError(
            "medium_panel_row_count_mismatch",
            "A medium forecast panel's physical row count does not match its own recorded "
            "row_count metadata",
        )
    if frame.select(["horizon", "anchor_date", "listing_id", "is_forecast"]).is_duplicated().any():
        raise RefreshVerificationError(
            "medium_panel_duplicate_rows",
            "A medium forecast panel declares the same horizon/anchor/listing row more than once",
        )
    per_listing_price_asset: dict[str, str] = {}
    forecast_listing_horizons: dict[str, set[str]] = {}
    for row in frame.iter_rows(named=True):
        listing_id = row["listing_id"]
        price_asset_id = row["price_asset_id"]
        existing = per_listing_price_asset.get(listing_id)
        if existing is None:
            per_listing_price_asset[listing_id] = price_asset_id
        elif existing != price_asset_id:
            raise RefreshVerificationError(
                "medium_panel_row_price_asset_inconsistent",
                "A medium forecast panel declares more than one distinct price asset for the "
                "same listing",
            )
        if row["is_forecast"]:
            forecast_listing_horizons.setdefault(listing_id, set()).add(row["horizon"])
    eligible_id_strings = {str(listing_id) for listing_id in eligible_listings}
    if set(per_listing_price_asset) != eligible_id_strings:
        raise RefreshVerificationError(
            "medium_panel_listing_membership_mismatch",
            "A medium forecast panel's rows do not cover exactly the run's own reviewed "
            "eligible listing membership",
        )
    for listing_id in eligible_id_strings:
        if forecast_listing_horizons.get(listing_id) != set(_MEDIUM_HORIZONS):
            raise RefreshVerificationError(
                "medium_panel_forecast_rows_missing",
                "A medium forecast panel is missing a forecast-anchor row for every reviewed "
                "medium horizon for one of its own eligible listings",
            )
    benchmark_asset = _require_medium_panel_benchmark_asset(metadata, medium=medium, cutoff=cutoff)
    price_assets = [
        _require_medium_panel_row_price_asset(
            price_asset_id,
            listing=eligible_listings[uuid.UUID(listing_id)],
            medium=medium,
            cutoff=cutoff,
        )
        for listing_id, price_asset_id in per_listing_price_asset.items()
    ]
    closure = [benchmark_asset, *price_assets]
    recomputed_manifest_hash = hash_json(
        [asset_identity(asset) for asset in dedupe_assets(closure)]
    )
    if metadata.get("source_manifest_hash") != recomputed_manifest_hash:
        raise RefreshVerificationError(
            "medium_panel_source_manifest_hash_mismatch",
            "A medium forecast panel's source manifest hash does not match the source asset "
            "closure independently derived from the run's own eligible listings, reviewed "
            "benchmark identity, and physical panel rows",
        )
    recomputed_bundle_hash = hash_json(
        {
            "calendar_hash": metadata.get("calendar_hash"),
            "code_revision": metadata.get("code_revision"),
            "content_sha256": metadata.get("content_sha256"),
            "forecast_config_hash": metadata.get("config_hash"),
            "scoring_config_hash": metadata.get("scoring_config_hash"),
            "source_manifest_hash": recomputed_manifest_hash,
            "universe_config_hash": metadata.get("universe_config_hash"),
        }
    )
    if metadata.get("evidence_bundle_hash") != recomputed_bundle_hash:
        raise RefreshVerificationError(
            "medium_panel_evidence_bundle_hash_mismatch",
            "A medium forecast panel's evidence bundle hash does not match its own "
            "recomputed inputs",
        )


def _require_authoritative_medium_panel(
    *,
    run: AnalysisRun,
    medium: MediumLaneExpectation,
    scoring_config_version: str,
    eligible_listings: Mapping[UUID, Listing],
) -> DataAsset:
    """Resolve and fully prove this run's one authoritative medium
    forecast panel: identity, complete metadata shape/values (including an
    independently recomputed calendar hash), physical Parquet content, and
    an exact source closure derived from the run's own eligible listings
    and reviewed benchmark identity -- never from the panel's own
    self-reported source manifest."""
    panel_asset = _resolve_authoritative_medium_panel_asset(run=run)
    _require_cutoff_safe(
        panel_asset,
        cutoff=run.data_cutoff,
        reason="medium_panel_not_cutoff_safe",
        message="A medium forecast panel is not cutoff-safe for this run",
    )
    metadata = _require_medium_panel_metadata(
        panel_asset, medium=medium, run=run, scoring_config_version=scoring_config_version
    )
    _require_medium_panel_content(
        panel_asset,
        metadata=metadata,
        medium=medium,
        eligible_listings=eligible_listings,
        cutoff=run.data_cutoff,
    )
    return panel_asset


def _require_medium_source_closure_exact(
    *, own_ids: set[UUID], analysis_ref_ids: set[UUID], panel_asset_id: UUID
) -> None:
    """A medium advisory prediction's own declared `source_assets` must
    equal its owning StockAnalysis's own declared `source_assets` plus
    exactly the one bound panel asset -- no arbitrary extra reference."""
    if own_ids != analysis_ref_ids | {panel_asset_id}:
        raise RefreshVerificationError(
            "medium_prediction_source_assets_mismatch",
            "A medium advisory prediction's own declared source_assets is not exactly its "
            "StockAnalysis's own declared source assets plus its own bound panel asset",
        )


def _require_medium_panel_binding(
    prediction: Prediction,
    *,
    medium: MediumLaneExpectation,
    run: AnalysisRun,
    scoring_config_version: str,
    eligible_listings: Mapping[UUID, Listing],
    registry: _AssetRegistry,
    panel_cache: dict[UUID, DataAsset],
) -> UUID:
    calculation = prediction.calculation
    panel_asset_id = calculation.get("panel_asset_id")
    panel_sha256 = calculation.get("panel_sha256")
    if not isinstance(panel_asset_id, str) or not isinstance(panel_sha256, str):
        raise RefreshVerificationError(
            "medium_panel_reference_missing",
            "A medium advisory prediction has no recorded panel asset reference",
        )
    try:
        declared_id = uuid.UUID(panel_asset_id)
    except ValueError as exc:
        raise RefreshVerificationError(
            "medium_panel_reference_malformed",
            "A medium advisory prediction's panel asset reference is not a valid identifier",
        ) from exc
    authoritative = panel_cache.get(run.id)
    if authoritative is None:
        authoritative = _require_authoritative_medium_panel(
            run=run,
            medium=medium,
            scoring_config_version=scoring_config_version,
            eligible_listings=eligible_listings,
        )
        panel_cache[run.id] = authoritative
    if declared_id != authoritative.id or panel_sha256 != authoritative.sha256:
        raise RefreshVerificationError(
            "medium_panel_not_authoritative",
            "A medium advisory prediction's panel reference does not match this run's one "
            "authoritative registered panel asset",
        )
    ref = AssetRef(
        id=authoritative.id,
        provider=PANEL_PROVIDER,
        kind=PANEL_KIND,
        subject=str(run.id),
        sha256=authoritative.sha256,
    )
    declared = registry.refs.get(ref.id)
    if declared is None or declared != ref:
        raise RefreshVerificationError(
            "medium_panel_not_declared",
            "A medium advisory prediction's panel is not present exactly once in its own "
            "declared source_assets",
        )
    registry.resolve(ref, cutoff=prediction.data_cutoff)
    return ref.id


def _require_source_declared(
    asset_id: UUID, *, own_refs: set[UUID], reason: str, message: str
) -> None:
    if asset_id not in own_refs:
        raise RefreshVerificationError(reason, message)


def _parse_uuid(value: object, *, reason: str, message: str) -> UUID:
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(reason, message) from exc


def _require_cutoff_safe(asset: DataAsset, *, cutoff: datetime, reason: str, message: str) -> None:
    if asset.available_at > cutoff or asset.retrieved_at > cutoff:
        raise RefreshVerificationError(reason, message)


def _require_target_price_asset_bound(
    calculation: Mapping[str, Any],
    *,
    listing: Listing,
    own_refs: set[UUID],
    registry: _AssetRegistry,
) -> set[UUID]:
    """Bind `calculation["target_price_asset_id"]` to exactly one declared
    Twelve Data `price_history` reference for the prediction's own listing
    (never merely some price asset present elsewhere in the run)."""
    asset_id = _parse_uuid(
        calculation.get("target_price_asset_id"),
        reason="long_evidence_reference_malformed",
        message="A long forecast calculation has no valid target_price_asset_id",
    )
    _require_source_declared(
        asset_id,
        own_refs=own_refs,
        reason="long_evidence_target_price_asset_not_declared",
        message="A long forecast calculation's target price asset is not present in the "
        "prediction's own declared source_assets closure",
    )
    ref = registry.refs.get(asset_id)
    expected_subject = listing.provider_symbol or listing.ticker
    if (
        ref is None
        or ref.provider != TWELVE_DATA_PROVIDER
        or ref.kind != "price_history"
        or ref.subject != expected_subject
    ):
        raise RefreshVerificationError(
            "long_evidence_target_price_asset_mismatch",
            "A long forecast calculation's target price asset does not match its own "
            "prediction's listing and Twelve Data subject",
        )
    return {asset_id}


def _require_target_classification_bound(
    calculation: Mapping[str, Any],
    *,
    company_id: UUID,
    cutoff: datetime,
    own_refs: set[UUID],
    registry: _AssetRegistry,
) -> set[UUID]:
    node = calculation.get("target_classification")
    if node is None:
        return set()
    if not isinstance(node, dict):
        raise RefreshVerificationError(
            "long_evidence_reference_malformed",
            "A long forecast calculation's target_classification is not an object",
        )
    classification_id = _parse_uuid(
        node.get("id"),
        reason="long_evidence_reference_malformed",
        message="A long forecast target_classification has an invalid identifier",
    )
    classification = (
        CompanyClassificationObservation.objects.filter(pk=classification_id)
        .select_related("source_asset")
        .first()
    )
    if classification is None:
        raise RefreshVerificationError(
            "long_evidence_classification_missing",
            "A long forecast calculation names a target classification that could not be resolved",
        )
    if classification.available_at > cutoff:
        raise RefreshVerificationError(
            "long_evidence_classification_after_cutoff",
            "A long forecast target classification was admitted after its prediction's data cutoff",
        )
    if classification.company_id != company_id:
        raise RefreshVerificationError(
            "long_evidence_classification_company_mismatch",
            "A long forecast target classification does not belong to the prediction's own "
            "listing's company",
        )
    canonical = classification_payload(classification)
    if node != canonical:
        raise RefreshVerificationError(
            "long_evidence_classification_identity_mismatch",
            "A long forecast target classification reference does not exactly match its "
            "resolved classification's own identity",
        )
    asset = classification.source_asset
    _require_cutoff_safe(
        asset,
        cutoff=cutoff,
        reason="long_evidence_source_after_cutoff",
        message="A long forecast target classification's source asset was admitted after its "
        "prediction's data cutoff",
    )
    _require_source_declared(
        asset.id,
        own_refs=own_refs,
        reason="long_evidence_source_not_declared",
        message="A long forecast target classification's source asset is not present in the "
        "prediction's own declared source_assets closure",
    )
    registry.add(asset_ref_for(asset), asset)
    return {asset.id}


def _resolve_fact_and_filing(
    fact_id: UUID, *, cutoff: datetime
) -> tuple[FundamentalFact, FundamentalFactEvidence]:
    fact = FundamentalFact.objects.filter(pk=fact_id).select_related("source_asset").first()
    if fact is None:
        raise RefreshVerificationError(
            "long_evidence_fact_missing",
            "A long forecast calculation names a fact that could not be resolved",
        )
    if fact.available_at > cutoff:
        raise RefreshVerificationError(
            "long_evidence_fact_after_cutoff",
            "A long forecast fact was admitted after its prediction's data cutoff",
        )
    filing_evidence = (
        FundamentalFactEvidence.objects.filter(fact=fact, role=FundamentalFactEvidence.Role.FILING)
        .select_related("source_asset")
        .first()
    )
    if filing_evidence is None:
        raise RefreshVerificationError(
            "long_evidence_filing_missing",
            "A long forecast fact has no registered filing evidence link",
        )
    return fact, filing_evidence


def _bind_fact_closure(
    fact: FundamentalFact,
    filing_evidence: FundamentalFactEvidence,
    *,
    cutoff: datetime,
    own_refs: set[UUID],
    registry: _AssetRegistry,
) -> set[UUID]:
    fact_asset = fact.source_asset
    filing_asset = filing_evidence.source_asset
    _require_cutoff_safe(
        fact_asset,
        cutoff=cutoff,
        reason="long_evidence_source_after_cutoff",
        message="A long forecast fact's source asset was admitted after its prediction's data "
        "cutoff",
    )
    _require_cutoff_safe(
        filing_asset,
        cutoff=cutoff,
        reason="long_evidence_source_after_cutoff",
        message="A long forecast fact's filing evidence asset was admitted after its "
        "prediction's data cutoff",
    )
    _require_source_declared(
        fact_asset.id,
        own_refs=own_refs,
        reason="long_evidence_source_not_declared",
        message="A long forecast fact's source asset is not present in the prediction's own "
        "declared source_assets closure",
    )
    _require_source_declared(
        filing_asset.id,
        own_refs=own_refs,
        reason="long_evidence_source_not_declared",
        message="A long forecast fact's filing evidence asset is not present in the "
        "prediction's own declared source_assets closure",
    )
    registry.add(asset_ref_for(fact_asset), fact_asset)
    registry.add(asset_ref_for(filing_asset), filing_asset)
    return {fact_asset.id, filing_asset.id}


def _require_fact_identity(
    node: object,
    *,
    company_id: UUID,
    cutoff: datetime,
    canonical_fn: Callable[[FundamentalFact, dict[str, DataAsset]], dict[str, Any]],
    context: str,
) -> tuple[FundamentalFact, FundamentalFactEvidence]:
    """Resolve one fact-evidence node's id, require its resolved fact
    belongs to `company_id`, and require the node equals `canonical_fn(...)`
    exactly -- byte-for-byte, never a subset comparison (finding 6/7).

    `canonical_fn` distinguishes the two accepted shapes: `fact_payload`
    for a full `input_facts` entry, `fact_reference` for an abbreviated
    peer/`assessed_evidence` reference. Returns the resolved
    `(fact, filing_evidence)` pair; a caller that also needs closure/cutoff
    binding passes it to `_bind_fact_closure`, while `evidence_selection`'s
    own internal-consistency proof (which only needs the identity check
    itself, not a second closure binding of ids the generic `*_fact_id(s)`
    walk already binds) can use it directly.
    """
    if not isinstance(node, dict):
        raise RefreshVerificationError(
            "long_evidence_reference_malformed",
            f"A long forecast {context} entry is not an object",
        )
    fact_id = _parse_uuid(
        node.get("id"),
        reason="long_evidence_reference_malformed",
        message=f"A long forecast {context} entry has an invalid identifier",
    )
    fact, filing_evidence = _resolve_fact_and_filing(fact_id, cutoff=cutoff)
    if fact.company_id != company_id:
        raise RefreshVerificationError(
            "long_evidence_fact_company_mismatch",
            f"A long forecast {context} does not belong to the expected company",
        )
    canonical = canonical_fn(fact, {str(fact.pk): filing_evidence.source_asset})
    if set(node) != set(canonical) or node != canonical:
        raise RefreshVerificationError(
            "long_evidence_fact_identity_mismatch",
            f"A long forecast {context} does not exactly match its resolved fact's own identity",
        )
    return fact, filing_evidence


def _require_fact_node_bound(
    node: object,
    *,
    company_id: UUID,
    cutoff: datetime,
    own_refs: set[UUID],
    registry: _AssetRegistry,
    canonical_fn: Callable[[FundamentalFact, dict[str, DataAsset]], dict[str, Any]],
    context: str,
) -> set[UUID]:
    """One fact-evidence node with full closure/cutoff binding: a full
    `input_facts` entry (`canonical_fn=fact_payload`) or an abbreviated peer
    `fact_references` entry (`canonical_fn=fact_reference`). The identity
    proof is shared via `_require_fact_identity`; only the closure binding
    differs from `evidence_selection`'s own bare identity check."""
    fact, filing_evidence = _require_fact_identity(
        node, company_id=company_id, cutoff=cutoff, canonical_fn=canonical_fn, context=context
    )
    return _bind_fact_closure(
        fact, filing_evidence, cutoff=cutoff, own_refs=own_refs, registry=registry
    )


def _require_input_facts_bound(
    calculation: Mapping[str, Any],
    *,
    company_id: UUID,
    cutoff: datetime,
    own_refs: set[UUID],
    registry: _AssetRegistry,
) -> set[UUID]:
    entries = calculation.get("input_facts")
    if entries is None:
        return set()
    if not isinstance(entries, list):
        raise RefreshVerificationError(
            "long_evidence_reference_malformed",
            "A long forecast calculation's input_facts is not a list",
        )
    closure: set[UUID] = set()
    for entry in entries:
        closure |= _require_fact_node_bound(
            entry,
            company_id=company_id,
            cutoff=cutoff,
            own_refs=own_refs,
            registry=registry,
            canonical_fn=fact_payload,
            context="input fact",
        )
    return closure


def _require_peer_bound(
    node: object, *, cutoff: datetime, own_refs: set[UUID], registry: _AssetRegistry
) -> set[UUID]:
    """One `peer_set` entry: exact peer listing/ticker agreement, its own
    price asset provider/kind/subject, its own classification row (company
    and SIC agreement, full identity match), and every one of its own
    declared `fact_references` bound to that same peer's company."""
    if not isinstance(node, dict):
        raise RefreshVerificationError(
            "long_evidence_reference_malformed", "A long forecast peer_set entry is not an object"
        )
    listing_id = _parse_uuid(
        node.get("listing_id"),
        reason="long_evidence_reference_malformed",
        message="A long forecast peer entry has an invalid listing_id",
    )
    peer_listing = Listing.objects.select_related("security__company").filter(pk=listing_id).first()
    if peer_listing is None:
        raise RefreshVerificationError(
            "long_evidence_listing_missing",
            "A long forecast calculation names a peer listing that could not be resolved",
        )
    if node.get("ticker") != peer_listing.ticker:
        raise RefreshVerificationError(
            "long_evidence_peer_ticker_mismatch",
            "A long forecast peer entry's ticker does not match its resolved listing's own ticker",
        )
    peer_company_id = peer_listing.security.company_id

    price_asset_id = _parse_uuid(
        node.get("price_asset_id"),
        reason="long_evidence_reference_malformed",
        message="A long forecast peer entry has an invalid price_asset_id",
    )
    _require_source_declared(
        price_asset_id,
        own_refs=own_refs,
        reason="long_evidence_price_asset_not_declared",
        message="A long forecast peer entry's price asset is not present in the prediction's "
        "own declared source_assets closure",
    )
    price_ref = registry.refs.get(price_asset_id)
    expected_subject = peer_listing.provider_symbol or peer_listing.ticker
    if (
        price_ref is None
        or price_ref.provider != TWELVE_DATA_PROVIDER
        or price_ref.kind != "price_history"
        or price_ref.subject != expected_subject
    ):
        raise RefreshVerificationError(
            "long_evidence_price_asset_kind_mismatch",
            "A long forecast peer entry's price asset does not match that peer's own listing "
            "and Twelve Data subject",
        )
    closure: set[UUID] = {price_asset_id}

    classification_id_raw = node.get("classification_id")
    nested_classification = node.get("classification")
    classification_id = _parse_uuid(
        classification_id_raw,
        reason="long_evidence_reference_malformed",
        message="A long forecast peer entry has an invalid classification_id",
    )
    if not isinstance(nested_classification, dict) or str(nested_classification.get("id")) != str(
        classification_id
    ):
        raise RefreshVerificationError(
            "long_evidence_classification_id_mismatch",
            "A long forecast peer entry's classification_id does not match its own nested "
            "classification reference",
        )
    classification = (
        CompanyClassificationObservation.objects.filter(pk=classification_id)
        .select_related("source_asset")
        .first()
    )
    if classification is None:
        raise RefreshVerificationError(
            "long_evidence_classification_missing",
            "A long forecast peer entry names a classification that could not be resolved",
        )
    if classification.available_at > cutoff:
        raise RefreshVerificationError(
            "long_evidence_classification_after_cutoff",
            "A long forecast peer classification was admitted after its prediction's data cutoff",
        )
    if classification.company_id != peer_company_id:
        raise RefreshVerificationError(
            "long_evidence_classification_company_mismatch",
            "A long forecast peer classification does not belong to that peer's own company",
        )
    canonical_classification = classification_payload(classification)
    if nested_classification != canonical_classification:
        raise RefreshVerificationError(
            "long_evidence_classification_identity_mismatch",
            "A long forecast peer classification reference does not exactly match its "
            "resolved classification's own identity",
        )
    if node.get("sic") != canonical_classification["code"]:
        raise RefreshVerificationError(
            "long_evidence_peer_sic_mismatch",
            "A long forecast peer entry's sic does not match its resolved classification's own "
            "code",
        )
    classification_asset = classification.source_asset
    _require_cutoff_safe(
        classification_asset,
        cutoff=cutoff,
        reason="long_evidence_source_after_cutoff",
        message="A long forecast peer classification's source asset was admitted after its "
        "prediction's data cutoff",
    )
    _require_source_declared(
        classification_asset.id,
        own_refs=own_refs,
        reason="long_evidence_source_not_declared",
        message="A long forecast peer classification's source asset is not present in the "
        "prediction's own declared source_assets closure",
    )
    registry.add(asset_ref_for(classification_asset), classification_asset)
    closure.add(classification_asset.id)

    fact_references = node.get("fact_references")
    if fact_references is not None:
        if not isinstance(fact_references, list):
            raise RefreshVerificationError(
                "long_evidence_reference_malformed",
                "A long forecast peer entry's fact_references is not a list",
            )
        for entry in fact_references:
            closure |= _require_fact_node_bound(
                entry,
                company_id=peer_company_id,
                cutoff=cutoff,
                own_refs=own_refs,
                registry=registry,
                canonical_fn=fact_reference,
                context="peer fact reference",
            )
    return closure


def _require_peer_set_bound(
    calculation: Mapping[str, Any],
    *,
    cutoff: datetime,
    own_refs: set[UUID],
    registry: _AssetRegistry,
) -> set[UUID]:
    entries = calculation.get("peer_set")
    if entries is None:
        return set()
    if not isinstance(entries, list):
        raise RefreshVerificationError(
            "long_evidence_reference_malformed",
            "A long forecast calculation's peer_set is not a list",
        )
    closure: set[UUID] = set()
    for entry in entries:
        closure |= _require_peer_bound(entry, cutoff=cutoff, own_refs=own_refs, registry=registry)
    return closure


def _require_referenced_fact_closure(
    prediction: Prediction, *, own_refs: set[UUID], registry: _AssetRegistry
) -> set[UUID]:
    """Every raw fact id named anywhere under a `*_fact_id`/`*_fact_ids` key
    in `prediction.calculation` -- a closure/completeness list carrying no
    accompanying identity fields of its own (e.g. `manifest_evidence_fact_ids`,
    `assessed_evidence_fact_ids`, `selected_input_fact_ids`, plain `fact_ids`)
    -- must resolve to a real, cutoff-eligible `FundamentalFact` whose own
    registered source asset *and* registered filing evidence asset are
    already declared in this prediction's own `source_assets`. A
    same-content fact registered under an alternate id fails here because
    the literal cited id would then resolve to a different row (or none),
    not because content is compared directly.
    """
    fact_ids = referenced_evidence_fact_ids(prediction.calculation)
    if not fact_ids:
        return set()
    try:
        uuids = [uuid.UUID(value) for value in fact_ids]
    except ValueError as exc:
        raise RefreshVerificationError(
            "long_evidence_reference_malformed",
            "A long forecast calculation names a fact id that is not a valid identifier",
        ) from exc
    closure: set[UUID] = set()
    for fact_id in uuids:
        fact, filing_evidence = _resolve_fact_and_filing(fact_id, cutoff=prediction.data_cutoff)
        closure |= _bind_fact_closure(
            fact,
            filing_evidence,
            cutoff=prediction.data_cutoff,
            own_refs=own_refs,
            registry=registry,
        )
    return closure


def _require_fact_id_list(raw: object, *, field: str) -> tuple[str, ...]:
    """Strictly parse one `evidence_selection` fact-id-set field: a list of
    distinct, non-empty strings. Duplicates within a single field are
    rejected even though the field itself carries no accompanying identity
    (finding 2)."""
    if not isinstance(raw, list):
        raise RefreshVerificationError(
            "long_evidence_selection_malformed",
            f"A long forecast calculation's evidence_selection.{field} is not a list",
        )
    ids: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or not item:
            raise RefreshVerificationError(
                "long_evidence_selection_malformed",
                f"A long forecast calculation's evidence_selection.{field} has a non-string "
                "or empty entry",
            )
        if item in seen:
            raise RefreshVerificationError(
                "long_evidence_selection_duplicate",
                f"A long forecast calculation's evidence_selection.{field} declares the same "
                "fact id more than once",
            )
        seen.add(item)
        ids.append(item)
    return tuple(ids)


def _require_evidence_selection_bound(
    prediction: Prediction, *, target_company_id: UUID, cutoff: datetime
) -> None:
    """Prove `calculation['evidence_selection']`'s fact-id-set fields and its
    abbreviated `assessed_evidence` payload are exactly consistent with each
    other and with the target company (finding 2).

    - `selected_input_fact_ids` must equal exactly the ids named in the
      calculation's own `input_facts`;
    - `selected_input_fact_ids` and `assessed_evidence_fact_ids` must be
      disjoint;
    - `manifest_evidence_fact_ids` must equal exactly their union;
    - every `assessed_evidence` entry must be the abbreviated
      `fact_reference(...)` shape exactly, belong to the target company (never
      a peer's), and its ids must equal exactly `assessed_evidence_fact_ids`.

    Every id named here is separately resolved, cutoff-checked, and bound
    into the prediction's own declared `source_assets` closure by
    `_require_referenced_fact_closure`'s generic `*_fact_id(s)` walk; this
    function only proves the declared *sets themselves* -- and the abbreviated
    payload's exact content -- are internally consistent, not merely
    resolvable.
    """
    evidence_selection = prediction.calculation.get("evidence_selection")
    if evidence_selection is None:
        return
    if not isinstance(evidence_selection, dict):
        raise RefreshVerificationError(
            "long_evidence_selection_malformed",
            "A long forecast calculation's evidence_selection is not an object",
        )
    selected_ids = _require_fact_id_list(
        evidence_selection.get("selected_input_fact_ids"), field="selected_input_fact_ids"
    )
    assessed_ids = _require_fact_id_list(
        evidence_selection.get("assessed_evidence_fact_ids"), field="assessed_evidence_fact_ids"
    )
    manifest_ids = _require_fact_id_list(
        evidence_selection.get("manifest_evidence_fact_ids"), field="manifest_evidence_fact_ids"
    )

    selected_set = set(selected_ids)
    assessed_set = set(assessed_ids)
    if selected_set & assessed_set:
        raise RefreshVerificationError(
            "long_evidence_selection_overlap",
            "A long forecast calculation's selected and assessed evidence fact id sets are not "
            "disjoint",
        )
    if set(manifest_ids) != selected_set | assessed_set:
        raise RefreshVerificationError(
            "long_evidence_selection_union_mismatch",
            "A long forecast calculation's manifest_evidence_fact_ids is not exactly the union "
            "of its own selected and assessed evidence fact ids",
        )

    input_facts = prediction.calculation.get("input_facts")
    input_fact_ids: set[str] = set()
    if isinstance(input_facts, list):
        for entry in input_facts:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                input_fact_ids.add(entry["id"])
    if input_fact_ids != selected_set:
        raise RefreshVerificationError(
            "long_evidence_selection_selected_mismatch",
            "A long forecast calculation's selected_input_fact_ids does not exactly match its "
            "own input_facts",
        )

    assessed_evidence = evidence_selection.get("assessed_evidence")
    if not isinstance(assessed_evidence, list):
        raise RefreshVerificationError(
            "long_evidence_selection_malformed",
            "A long forecast calculation's evidence_selection.assessed_evidence is not a list",
        )
    seen_assessed_ids: set[str] = set()
    for entry in assessed_evidence:
        # Identity/company/shape proof is the same one `_require_fact_node_bound`
        # uses for a peer's `fact_references` entry; only the duplicate-id
        # check below is specific to this field's own set semantics.
        fact, _filing_evidence = _require_fact_identity(
            entry,
            company_id=target_company_id,
            cutoff=cutoff,
            canonical_fn=fact_reference,
            context="assessed_evidence entry",
        )
        fact_id_str = str(fact.pk)
        if fact_id_str in seen_assessed_ids:
            raise RefreshVerificationError(
                "long_evidence_selection_duplicate",
                "A long forecast calculation's assessed_evidence declares the same fact id "
                "more than once",
            )
        seen_assessed_ids.add(fact_id_str)
    if seen_assessed_ids != assessed_set:
        raise RefreshVerificationError(
            "long_evidence_selection_assessed_mismatch",
            "A long forecast calculation's assessed_evidence_fact_ids does not exactly match "
            "its own assessed_evidence entries",
        )


def _require_long_evidence_binding(
    prediction: Prediction, *, own_refs: set[UUID], registry: _AssetRegistry
) -> None:
    """Prove every fact/classification/peer/price reference named anywhere
    in a long forecast's calculation payload binds to a real, cutoff-
    eligible immutable row whose own declared source asset is a genuine
    member of *this prediction's own* declared `source_assets` closure --
    not merely present somewhere else in the verified run -- and that the
    prediction's own declared `source_assets` are *exactly* this closure,
    with no extra undeclared-but-otherwise-valid reference left over.

    Target price/classification/facts are bound to the prediction's own
    listing/company; each peer's price/classification/facts are bound to
    that peer's own listing/company, never the target's.
    """
    cutoff = prediction.data_cutoff
    calculation = prediction.calculation
    target_listing = prediction.analysis.listing
    target_company_id = target_listing.security.company_id

    _require_evidence_selection_bound(
        prediction, target_company_id=target_company_id, cutoff=cutoff
    )

    closure: set[UUID] = set()
    closure |= _require_target_price_asset_bound(
        calculation, listing=target_listing, own_refs=own_refs, registry=registry
    )
    closure |= _require_target_classification_bound(
        calculation,
        company_id=target_company_id,
        cutoff=cutoff,
        own_refs=own_refs,
        registry=registry,
    )
    closure |= _require_input_facts_bound(
        calculation,
        company_id=target_company_id,
        cutoff=cutoff,
        own_refs=own_refs,
        registry=registry,
    )
    closure |= _require_peer_set_bound(
        calculation, cutoff=cutoff, own_refs=own_refs, registry=registry
    )
    closure |= _require_referenced_fact_closure(prediction, own_refs=own_refs, registry=registry)

    if own_refs - closure:
        raise RefreshVerificationError(
            "long_evidence_source_assets_extra",
            "A long forecast prediction declares a source asset that is not part of its own "
            "calculation's exact evidence closure",
        )


def verify_analysis_output_manifest(
    *,
    run: AnalysisRun,
    eligible_listing_ids: set[UUID],
    listings_by_id: Mapping[UUID, Listing],
    code_revision: str,
    scoring_config_version: str,
    scoring_config_hash: str,
    decision_horizons: frozenset[str],
    medium: MediumLaneExpectation | None,
    long: LongLaneExpectation | None,
) -> StageVerificationResult:
    """Prove one observed `AnalysisRun`'s complete, immutable output.

    `run` is assumed already bound to its snapshot/target date/on-time
    status/code revision/scoring configuration by the caller (mirroring
    `core.refresh_verification._verify_analysis_run`, not reproduced here
    to avoid a duplicate, possibly-diverging binding check -- see this
    slice's handoff report for the deferred integration point).
    """
    manifest_ref, manifest_run_id, manifest_plan, manifest_entries = _resolve_manifest(run)

    stock_analyses = list(
        StockAnalysis.objects.filter(run=run).select_related("run__universe_snapshot")
    )
    predictions = list(
        Prediction.objects.filter(analysis__run=run).select_related(
            "analysis", "analysis__run__universe_snapshot"
        )
    )

    _require_manifest_matches_current_rows(
        run, stock_analyses, predictions, manifest_run_id, manifest_entries
    )
    _require_listing_set_matches(stock_analyses, eligible_listing_ids=eligible_listing_ids)
    by_analysis = _require_prediction_horizon_multiset(
        stock_analyses,
        predictions,
        decision_horizons=decision_horizons,
        medium=medium,
        long=long,
        eligible_listing_ids=eligible_listing_ids,
    )

    expected_plan = build_output_plan(
        eligible_listing_ids=eligible_listing_ids,
        decision_horizons=decision_horizons,
        medium_active=medium is not None,
        long_active=long is not None,
    )
    if manifest_plan != expected_plan:
        raise RefreshVerificationError(
            "analysis_output_manifest_plan_mismatch",
            "The analysis output manifest's own precomputed output plan does not match the "
            "reviewed eligible membership and decision/advisory lane expectations",
        )
    actual_plan = actual_output_plan(stock_analyses, predictions)
    if actual_plan != expected_plan:
        raise RefreshVerificationError(
            "analysis_output_plan_diverged",
            "The verified run's actual StockAnalysis/Prediction rows do not reproduce the "
            "exact reviewed output plan",
        )

    analyses_by_id = {analysis.id: analysis for analysis in stock_analyses}
    asset_registry = _AssetRegistry()
    medium_panel_cache: dict[UUID, DataAsset] = {}
    for analysis in stock_analyses:
        _require_scenario_document_shape(
            analysis, medium_active=medium is not None, long_active=long is not None
        )
        _require_scenario_mirror_consistency(analysis)
        data_quality = analysis.data_quality if isinstance(analysis.data_quality, dict) else {}
        raw_source_assets = data_quality.get("source_assets")
        analysis_refs = _require_source_assets_bound(
            raw_source_assets,
            context=f"StockAnalysis {analysis.id}",
            cutoff=run.data_cutoff,
            registry=asset_registry,
        )
        assert isinstance(raw_source_assets, list)
        _require_stock_analysis_price_source_bound(
            analysis, raw_source_assets=raw_source_assets, refs=analysis_refs
        )
        analysis_ref_ids = {ref.id for ref in analysis_refs}
        listing = listings_by_id.get(analysis.listing_id)
        if listing is None:
            raise RefreshVerificationError(
                "stock_analysis_listing_mismatch",
                "A StockAnalysis references a listing outside the verified eligible "
                "universe membership",
            )
        for prediction in by_analysis.get(str(analysis.id), []):
            _require_prediction_ground_truth(
                prediction,
                target_date=run.target_date,
                code_revision=code_revision,
                scoring_config_version=scoring_config_version,
                scoring_config_hash=scoring_config_hash,
                decision_horizons=decision_horizons,
                medium=medium,
                long=long,
                listing=listing,
            )
            _require_prediction_matches_analysis(prediction, analyses_by_id[prediction.analysis_id])
            prediction_refs = _require_source_assets_bound(
                prediction.source_assets,
                context=f"Prediction {prediction.id}",
                cutoff=prediction.data_cutoff,
                registry=asset_registry,
            )
            own_ids = {ref.id for ref in prediction_refs}
            if prediction.horizon in _MEDIUM_HORIZONS:
                assert medium is not None
                panel_asset_id = _require_medium_panel_binding(
                    prediction,
                    medium=medium,
                    run=run,
                    scoring_config_version=scoring_config_version,
                    eligible_listings=listings_by_id,
                    registry=asset_registry,
                    panel_cache=medium_panel_cache,
                )
                _require_medium_source_closure_exact(
                    own_ids=own_ids,
                    analysis_ref_ids=analysis_ref_ids,
                    panel_asset_id=panel_asset_id,
                )
            elif prediction.horizon in _LONG_HORIZONS:
                _require_long_evidence_binding(
                    prediction, own_refs=own_ids, registry=asset_registry
                )
            else:
                _require_decision_source_closure_exact(
                    own_ids=own_ids, analysis_ref_ids=analysis_ref_ids
                )

    store = open_asset_store()
    for asset_id in sorted(asset_registry.rows, key=str):
        read_checksummed_bytes(store, asset_registry.rows[asset_id])

    return StageVerificationResult(
        summary={
            "analysis_run_id": str(run.id),
            "stock_analysis_count": len(stock_analyses),
            "prediction_count": len(predictions),
            "manifest_entry_count": len(manifest_entries),
        },
        asset_refs=(manifest_ref, *asset_registry.sorted_refs()),
    )

"""Analysis-output-manifest contract leaf: envelope shape, field coverage,
canonical row digests, and the pre-write output plan.

Imports nothing from `research.service` or `research.refresh_validation`, so
both the writer (`research.service`) and the reader
(`research.refresh_validation`) can depend on it without a cycle. It does
import `research.models` (for field-coverage introspection and the fixed
`Prediction.Horizon`/`EvidenceRole` contract values) and
`data.models`/`data.refresh_evidence` (for `DataAsset` lookups and the
shared strict-JSON parser) -- all are safe, acyclic directions since none of
them imports back into this module or `research.service`.

Two immutable evidence manifests exist in this codebase: the universe
membership manifest (`data.refresh_evidence`) and this analysis-output
manifest, one per *observed* `AnalysisRun`. Both use the same envelope
idiom (`contract`/binding-id/payload) and the same duplicate-key-rejecting
strict JSON parser, but are otherwise independent: this manifest never
recomputes `AnalysisRun.config_hash` or any other mutable/derived field, it
only proves *exactly which rows exist and that none has changed since this
manifest was written* through a complete canonical digest per row -- never
a partial field-by-field re-derivation -- plus one additional, independent
proof: the exact *plan* (eligible listings and the exact prediction key
multiset they must carry) computed *before* the first `StockAnalysis`/
`Prediction` write, so a manifest can never simply echo back whatever rows
happened to be written.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from stanstock.data.models import DataAsset
from stanstock.data.refresh_evidence import strict_json_loads
from stanstock.research.models import Prediction, StockAnalysis

ANALYSIS_OUTPUT_MANIFEST_CONTRACT = "analysis-output-manifest@1"

#: `DataAsset.kind` for the immutable, checksummed manifest asset binding a
#: single observed `AnalysisRun` to the complete, exact set of
#: `StockAnalysis`/`Prediction` rows it produced.
ANALYSIS_OUTPUT_MANIFEST_KIND = "analysis_output_manifest"

#: Model labels used in every `ManifestEntry`. Deliberately explicit
#: (never derived from `Model.__name__` at read time) so a renamed Python
#: class can never silently change what an already-written manifest means.
ANALYSIS_RUN_MODEL = "AnalysisRun"
STOCK_ANALYSIS_MODEL = "StockAnalysis"
PREDICTION_MODEL = "Prediction"

#: Every concrete, persisted field each model carries, in the exact set a
#: manifest row digest must cover. `test_data_research_refresh_evidence.py`
#: fails closed if any of these three models gains a concrete field absent
#: from its own tuple here -- so a future migration cannot silently widen a
#: model without this manifest noticing the gap.
ANALYSIS_RUN_FIELDS: tuple[str, ...] = (
    "id",
    "generated_at",
    "data_cutoff",
    "target_date",
    "issued_on_time",
    "universe_snapshot_id",
    "config_version",
    "config_hash",
    "code_revision",
    "status",
)

STOCK_ANALYSIS_FIELDS: tuple[str, ...] = (
    "id",
    "run_id",
    "listing_id",
    "current_price",
    "daily_change",
    "overall_score",
    "recommendation",
    "risk_score",
    "risk_class",
    "confidence",
    "confidence_status",
    "component_scores",
    "forecast_scenarios",
    "short_scenario",
    "medium_scenario",
    "long_scenario",
    "reasons",
    "risks",
    "data_quality",
)

PREDICTION_FIELDS: tuple[str, ...] = (
    "id",
    "analysis_id",
    "listing_id",
    "generated_at",
    "target_date",
    "issued_on_time",
    "horizon",
    "evidence_role",
    "evidence_grade",
    "source_mode",
    "price_provider",
    "price_subject",
    "price_at_prediction",
    "bear_return",
    "base_return",
    "bull_return",
    "probability_positive",
    "confidence",
    "confidence_status",
    "insufficiency_reason",
    "recommendation",
    "overall_score",
    "component_scores",
    "model_version",
    "method_version",
    "config_hash",
    "data_cutoff",
    "source_assets",
    "calculation",
    "code_revision",
)

#: Field whitelist for each manifest-covered model label, keyed the same
#: way `ManifestEntry.model` is written/read.
MODEL_FIELDS: dict[str, tuple[str, ...]] = {
    ANALYSIS_RUN_MODEL: ANALYSIS_RUN_FIELDS,
    STOCK_ANALYSIS_MODEL: STOCK_ANALYSIS_FIELDS,
    PREDICTION_MODEL: PREDICTION_FIELDS,
}

_ENTRY_FIELDS = frozenset({"model", "row_id", "digest"})
_ENVELOPE_FIELDS = frozenset({"contract", "run_id", "plan", "entries"})
_SHA256_RE_LEN = 64


class ManifestPayloadError(ValueError):
    """A malformed analysis-output-manifest envelope or entry.

    A leaf-level `ValueError` subclass (not `core.verification_types`'s
    `RefreshVerificationError`) so this module never needs to import a
    `core` reason-coded type; the reader normalizes this into a
    `RefreshVerificationError` at its own boundary.
    """


def decimal_from_float(value: float, *, places: int) -> Decimal:
    """Round a raw float to `places` decimal places via its string form.

    Shared by `research.service` (persistence) and `research.refresh_validation`
    (verification) so both apply the exact same rounding rule; a leaf-level
    pure helper avoids `refresh_validation` needing to import `research.service`.
    """
    return Decimal(str(round(value, places)))


def optional_decimal_from_float(value: float | None, *, places: int) -> Decimal | None:
    if value is None:
        return None
    return decimal_from_float(value, places=places)


def _canonical_value(value: object) -> object:
    """Recursively normalize one field's value into a JSON-safe, stable form.

    `Decimal` and `UUID` always serialize as their exact string form (never
    a float, which could lose precision or reorder equal-but-differently
    -formatted values). `datetime` values must be timezone-aware (every
    `DateTimeField` in this project is, since `USE_TZ = True`); a naive
    datetime is a data-integrity bug this encoder refuses to paper over
    with an ambiguous local-time string. Every aware `datetime` is
    normalized to UTC before encoding: the same instant written with a
    non-UTC offset and later re-queried (a database driver may normalize
    the offset on read) must always encode identically, never diverge
    merely because of which offset happened to be attached in memory.
    `date` (never a `datetime`, since `datetime` is checked first) encodes
    as its plain ISO date string. `dict`/`list` are walked recursively --
    the persisted JSON fields (`component_scores`, `forecast_scenarios`,
    `source_assets`, `calculation`, ...) are already plain JSON-safe
    content, but are walked anyway in case a future field nests one of the
    special types above.
    """
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ManifestPayloadError(
                "Manifest field contains a naive datetime value; every DateTimeField in this "
                "project must be timezone-aware"
            )
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def canonical_row_json(model: str, values: dict[str, Any]) -> str:
    """The exact canonical JSON string a row's digest is computed from.

    Exposed (not only the digest) so a caller can hash-compare or log a
    diff without recomputing the encoding, while the digest itself stays
    the single value ever persisted or compared.
    """
    fields = MODEL_FIELDS.get(model)
    if fields is None:
        raise ManifestPayloadError(f"Unknown manifest model label: {model!r}")
    missing = [field for field in fields if field not in values]
    if missing:
        raise ManifestPayloadError(f"Row for {model!r} is missing required field(s)")
    payload = {
        "model": model,
        "fields": {field: _canonical_value(values[field]) for field in fields},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def row_digest(model: str, values: dict[str, Any]) -> str:
    """SHA-256 hex digest over every concrete persisted field of one row."""
    return hashlib.sha256(canonical_row_json(model, values).encode("utf-8")).hexdigest()


def _canonical_decimal(instance: Any, field: str, value: Decimal) -> Decimal:
    """Quantize `value` to its model field's own declared `decimal_places`.

    A `DecimalField`'s *database* representation is always quantized to
    `decimal_places` (Django's storage adapter enforces this on write), but
    an in-memory instance still holds whatever unquantized `Decimal` the
    caller originally assigned (e.g. `Decimal("63.7")` for a
    `decimal_places=2` field). Two otherwise-identical rows must never
    digest differently merely because one side read the value immediately
    after `.save()` (writer, same transaction) and the other re-queried it
    from the database (reader) -- so every `Decimal` is normalized to its
    field's authoritative quantization before canonical encoding, on both
    sides, regardless of which representation the caller happened to hold.
    """
    decimal_places = getattr(instance._meta.get_field(field), "decimal_places", None)
    if decimal_places is None:
        return value
    return value.quantize(Decimal(1).scaleb(-decimal_places))


def model_row_values(instance: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    """Every covered field's current stored value, keyed by attname.

    Shared by the writer (`research.service`, reading just-persisted ORM
    instances still inside their own transaction) and the reader
    (`research.refresh_validation`, reading freshly re-queried instances),
    so both sides compute a row's digest from exactly the same field
    extraction logic -- including the same `Decimal` quantization, so a
    writer-side unquantized in-memory value and a reader-side
    database-quantized value always encode identically.
    """
    values: dict[str, Any] = {}
    for field in fields:
        value = getattr(instance, field)
        if isinstance(value, Decimal):
            value = _canonical_decimal(instance, field, value)
        values[field] = value
    return values


def _validate_row_id_format(model: str, row_id: str) -> None:
    """Row-id shape is model-specific: `AnalysisRun`/`Prediction` use a
    `UUIDField` primary key; `StockAnalysis` uses Django's default
    integer `AutoField`. Reject anything else outright."""
    if model == STOCK_ANALYSIS_MODEL:
        if not row_id.isdigit() or str(int(row_id)) != row_id or int(row_id) < 1:
            raise ManifestPayloadError("Manifest entry row_id is not a valid positive integer")
        return
    try:
        UUID(row_id)
    except ValueError as exc:
        raise ManifestPayloadError("Manifest entry row_id is not a valid UUID") from exc


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One row's permanent identity and complete canonical content digest."""

    model: str
    row_id: str
    digest: str

    def to_json(self) -> dict[str, str]:
        return {"model": self.model, "row_id": self.row_id, "digest": self.digest}

    @classmethod
    def from_json(cls, raw: object) -> ManifestEntry:
        if not isinstance(raw, dict) or set(raw) != _ENTRY_FIELDS:
            raise ManifestPayloadError("Manifest entry payload shape is invalid")
        model, row_id, digest = raw["model"], raw["row_id"], raw["digest"]
        if not all(isinstance(v, str) and v for v in (model, row_id, digest)):
            raise ManifestPayloadError("Manifest entry has a blank or non-string field")
        if model not in MODEL_FIELDS:
            raise ManifestPayloadError("Manifest entry names an unsupported model")
        _validate_row_id_format(model, row_id)
        if len(digest) != _SHA256_RE_LEN or any(c not in "0123456789abcdef" for c in digest):
            raise ManifestPayloadError("Manifest entry digest is not 64 lowercase hex characters")
        return cls(model=model, row_id=row_id, digest=digest)


def sorted_entries(entries: list[ManifestEntry]) -> list[ManifestEntry]:
    """The one canonical entry order both writer and reader must use."""
    return sorted(entries, key=lambda entry: (entry.model, entry.row_id))


#: Fixed, contract-frozen decision-role/advisory-lane horizon values, read
#: directly off `Prediction`'s own `TextChoices` (the single source of
#: truth) rather than duplicated string literals.
_DECISION_ROLE = Prediction.EvidenceRole.DECISION.value
_ADVISORY_ROLE = Prediction.EvidenceRole.ADVISORY.value
MEDIUM_HORIZONS: frozenset[str] = frozenset(
    {Prediction.Horizon.SIX_MONTH.value, Prediction.Horizon.TWELVE_MONTH.value}
)
LONG_HORIZONS: frozenset[str] = frozenset(
    {Prediction.Horizon.THREE_YEAR.value, Prediction.Horizon.FIVE_YEAR.value}
)
_VALID_HORIZONS = frozenset(choice.value for choice in Prediction.Horizon)
_VALID_ROLES = frozenset({_DECISION_ROLE, _ADVISORY_ROLE})

_PLAN_PREDICTION_FIELDS = frozenset({"listing_id", "horizon", "evidence_role", "count"})
_PLAN_FIELDS = frozenset({"eligible_listing_ids", "predictions"})


@dataclass(frozen=True, slots=True)
class PlanPredictionKey:
    """One `(listing_id, horizon, evidence_role)` key this run must produce,
    with its expected row `count` (always 1 for any valid key; carried
    explicitly, never inferred, so a duplicate row for the same key cannot
    silently agree with an implicit "at least one" reading)."""

    listing_id: str
    horizon: str
    evidence_role: str
    count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "horizon": self.horizon,
            "evidence_role": self.evidence_role,
            "count": self.count,
        }

    @classmethod
    def from_json(cls, raw: object) -> PlanPredictionKey:
        if not isinstance(raw, dict) or set(raw) != _PLAN_PREDICTION_FIELDS:
            raise ManifestPayloadError("Manifest plan prediction entry shape is invalid")
        listing_id, horizon, evidence_role, count = (
            raw["listing_id"],
            raw["horizon"],
            raw["evidence_role"],
            raw["count"],
        )
        if not all(isinstance(v, str) and v for v in (listing_id, horizon, evidence_role)):
            raise ManifestPayloadError(
                "Manifest plan prediction entry has a blank or non-string field"
            )
        try:
            UUID(listing_id)
        except ValueError as exc:
            raise ManifestPayloadError(
                "Manifest plan prediction entry listing_id is not a valid UUID"
            ) from exc
        if evidence_role not in _VALID_ROLES:
            raise ManifestPayloadError(
                "Manifest plan prediction entry evidence_role is not recognized"
            )
        if horizon not in _VALID_HORIZONS:
            raise ManifestPayloadError("Manifest plan prediction entry horizon is not recognized")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ManifestPayloadError(
                "Manifest plan prediction entry count must be a positive integer"
            )
        return cls(listing_id=listing_id, horizon=horizon, evidence_role=evidence_role, count=count)


def sorted_plan_predictions(entries: Iterable[PlanPredictionKey]) -> list[PlanPredictionKey]:
    """The one canonical plan-prediction order both writer and reader use."""
    return sorted(entries, key=lambda entry: (entry.listing_id, entry.evidence_role, entry.horizon))


@dataclass(frozen=True, slots=True)
class ManifestPlan:
    """The exact, deterministic output plan for one observed run: which
    listings are eligible, and the exact `(listing_id, horizon,
    evidence_role)` prediction key multiset (with per-key counts) every
    correctly-behaving run must produce. Computed *before* the first
    `StockAnalysis`/`Prediction` write from planning inputs alone (eligible
    membership plus frozen scoring/lane gating), never from a row that has
    already been written.
    """

    eligible_listing_ids: tuple[str, ...]
    predictions: tuple[PlanPredictionKey, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "eligible_listing_ids": list(self.eligible_listing_ids),
            "predictions": [entry.to_json() for entry in sorted_plan_predictions(self.predictions)],
        }

    @classmethod
    def from_json(cls, raw: object) -> ManifestPlan:
        if not isinstance(raw, dict) or set(raw) != _PLAN_FIELDS:
            raise ManifestPayloadError("Manifest plan payload shape is invalid")
        raw_listing_ids = raw["eligible_listing_ids"]
        if not isinstance(raw_listing_ids, list):
            raise ManifestPayloadError("Manifest plan eligible_listing_ids is not a list")
        listing_ids: list[str] = []
        for value in raw_listing_ids:
            if not isinstance(value, str) or not value:
                raise ManifestPayloadError(
                    "Manifest plan eligible_listing_ids entry is blank or non-string"
                )
            try:
                UUID(value)
            except ValueError as exc:
                raise ManifestPayloadError(
                    "Manifest plan eligible_listing_ids entry is not a valid UUID"
                ) from exc
            listing_ids.append(value)
        if len(set(listing_ids)) != len(listing_ids):
            raise ManifestPayloadError("Manifest plan eligible_listing_ids contains a duplicate")
        if listing_ids != sorted(listing_ids):
            raise ManifestPayloadError(
                "Manifest plan eligible_listing_ids is not in canonical sorted order"
            )
        raw_predictions = raw["predictions"]
        if not isinstance(raw_predictions, list):
            raise ManifestPayloadError("Manifest plan predictions is not a list")
        predictions = [PlanPredictionKey.from_json(entry) for entry in raw_predictions]
        seen: set[tuple[str, str, str]] = set()
        for entry in predictions:
            key = (entry.listing_id, entry.horizon, entry.evidence_role)
            if key in seen:
                raise ManifestPayloadError("Manifest plan predictions contains a duplicate key")
            seen.add(key)
        if predictions != sorted_plan_predictions(predictions):
            raise ManifestPayloadError("Manifest plan predictions is not in canonical sorted order")
        return cls(
            eligible_listing_ids=tuple(listing_ids),
            predictions=tuple(predictions),
        )


def build_output_plan(
    *,
    eligible_listing_ids: set[UUID],
    decision_horizons: frozenset[str],
    medium_active: bool,
    long_active: bool,
) -> ManifestPlan:
    """The exact plan a correctly-behaving run must produce for the given
    eligible membership and lane gating -- called once by the writer before
    any write, and independently re-derived by the verifier from its own
    caller-supplied `eligible_listing_ids`/`decision_horizons`/lane
    expectations (never from the manifest's own recorded plan, and never
    from current database rows)."""
    listing_ids = sorted(str(listing_id) for listing_id in eligible_listing_ids)
    predictions: list[PlanPredictionKey] = []
    for listing_id in listing_ids:
        for horizon in sorted(decision_horizons):
            predictions.append(
                PlanPredictionKey(
                    listing_id=listing_id, horizon=horizon, evidence_role=_DECISION_ROLE, count=1
                )
            )
        if medium_active:
            for horizon in sorted(MEDIUM_HORIZONS):
                predictions.append(
                    PlanPredictionKey(
                        listing_id=listing_id,
                        horizon=horizon,
                        evidence_role=_ADVISORY_ROLE,
                        count=1,
                    )
                )
        if long_active:
            for horizon in sorted(LONG_HORIZONS):
                predictions.append(
                    PlanPredictionKey(
                        listing_id=listing_id,
                        horizon=horizon,
                        evidence_role=_ADVISORY_ROLE,
                        count=1,
                    )
                )
    return ManifestPlan(
        eligible_listing_ids=tuple(listing_ids),
        predictions=tuple(sorted_plan_predictions(predictions)),
    )


def actual_output_plan(
    stock_analyses: Iterable[StockAnalysis], predictions: Iterable[Prediction]
) -> ManifestPlan:
    """The plan-shaped multiset the *current* `StockAnalysis`/`Prediction`
    rows actually carry -- used both by the writer's post-write, pre-
    registration self-check (must equal the precomputed plan) and by the
    reader (must equal the manifest's own recorded plan)."""
    counts: dict[tuple[str, str, str], int] = {}
    for prediction in predictions:
        key = (str(prediction.listing_id), prediction.horizon, prediction.evidence_role)
        counts[key] = counts.get(key, 0) + 1
    plan_predictions = [
        PlanPredictionKey(listing_id=listing_id, horizon=horizon, evidence_role=role, count=count)
        for (listing_id, horizon, role), count in counts.items()
    ]
    listing_ids = sorted({str(analysis.listing_id) for analysis in stock_analyses})
    return ManifestPlan(
        eligible_listing_ids=tuple(listing_ids),
        predictions=tuple(sorted_plan_predictions(plan_predictions)),
    )


def build_manifest_envelope(
    *, run_id: UUID, plan: ManifestPlan, entries: list[ManifestEntry]
) -> dict[str, Any]:
    return {
        "contract": ANALYSIS_OUTPUT_MANIFEST_CONTRACT,
        "run_id": str(run_id),
        "plan": plan.to_json(),
        "entries": [entry.to_json() for entry in sorted_entries(entries)],
    }


def dumps_canonical_envelope(envelope: dict[str, Any]) -> bytes:
    return json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")


def parse_manifest_envelope(payload: bytes) -> tuple[str, ManifestPlan, list[ManifestEntry]]:
    """Strict-parse `payload`, returning `(run_id, plan, entries)`.

    Raises `ManifestPayloadError` (never a bare `json.JSONDecodeError` or
    unrelated exception) on invalid UTF-8/JSON, a duplicate top-level or
    entry key, an unexpected envelope shape, or a malformed entry or plan.

    After every structural check passes, the accepted envelope is
    reconstructed through the same canonical writer path
    (`build_manifest_envelope`/`dumps_canonical_envelope`) and its bytes are
    required to equal `payload` exactly (finding 5). Reordered object keys,
    added indentation/whitespace, and an alternate (but semantically
    equivalent) escaping all parse to the identical structure above yet
    produce different bytes than the canonical compact/sorted-key
    encoding -- and are rejected here even though nothing else in this
    function would otherwise notice them. A payload produced by the
    canonical writer itself always round-trips byte-for-byte and remains
    accepted.
    """
    try:
        envelope = strict_json_loads(payload)
    except ValueError as exc:
        raise ManifestPayloadError("Manifest payload is not valid JSON") from exc
    if not isinstance(envelope, dict) or set(envelope) != _ENVELOPE_FIELDS:
        raise ManifestPayloadError("Manifest envelope has an unexpected shape")
    if envelope.get("contract") != ANALYSIS_OUTPUT_MANIFEST_CONTRACT:
        raise ManifestPayloadError("Manifest envelope contract is not recognized")
    run_id = envelope.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ManifestPayloadError("Manifest envelope run_id is blank or missing")
    try:
        parsed_run_id = UUID(run_id)
    except ValueError as exc:
        raise ManifestPayloadError("Manifest envelope run_id is not a valid UUID") from exc
    plan = ManifestPlan.from_json(envelope.get("plan"))
    raw_entries = envelope.get("entries")
    if not isinstance(raw_entries, list):
        raise ManifestPayloadError("Manifest envelope entries is not a list")
    entries = [ManifestEntry.from_json(raw) for raw in raw_entries]
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry.model, entry.row_id)
        if key in seen:
            raise ManifestPayloadError("Manifest envelope contains a duplicate row entry")
        seen.add(key)
    if entries != sorted_entries(entries):
        raise ManifestPayloadError("Manifest envelope entries is not in canonical sorted order")
    reconstructed = dumps_canonical_envelope(
        build_manifest_envelope(run_id=parsed_run_id, plan=plan, entries=entries)
    )
    if reconstructed != payload:
        raise ManifestPayloadError("Manifest payload bytes are not canonically encoded")
    return run_id, plan, entries


@dataclass(frozen=True, slots=True)
class ManifestLookup:
    """Exact-one resolution outcome. `asset` is set only when exactly one
    manifest asset is recorded for the run; otherwise `count` distinguishes
    a legacy run with no manifest (0) from an ambiguous/corrupt one (>1)."""

    asset: DataAsset | None
    count: int


def manifest_assets_for_run(run_id: UUID) -> list[DataAsset]:
    return list(
        DataAsset.objects.filter(
            provider="stanstock",
            kind=ANALYSIS_OUTPUT_MANIFEST_KIND,
            subject=str(run_id),
        )
    )


def lookup_manifest(run_id: UUID) -> ManifestLookup:
    assets = manifest_assets_for_run(run_id)
    if len(assets) == 1:
        return ManifestLookup(asset=assets[0], count=1)
    return ManifestLookup(asset=None, count=len(assets))

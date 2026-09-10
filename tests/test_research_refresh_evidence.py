"""Tests for `research.refresh_evidence`: canonical row digests, field
coverage, envelope round-tripping, and exact-one manifest lookup.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from stanstock.data.models import DataAsset
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.refresh_evidence import (
    ANALYSIS_RUN_FIELDS,
    ANALYSIS_RUN_MODEL,
    LONG_HORIZONS,
    MEDIUM_HORIZONS,
    PREDICTION_FIELDS,
    PREDICTION_MODEL,
    STOCK_ANALYSIS_FIELDS,
    STOCK_ANALYSIS_MODEL,
    ManifestEntry,
    ManifestPayloadError,
    ManifestPlan,
    PlanPredictionKey,
    actual_output_plan,
    build_manifest_envelope,
    build_output_plan,
    canonical_row_json,
    dumps_canonical_envelope,
    lookup_manifest,
    parse_manifest_envelope,
    row_digest,
    sorted_entries,
)

pytestmark = pytest.mark.django_db


def _concrete_field_names(model: type) -> set[str]:
    return {
        field.attname
        for field in model._meta.get_fields()
        if getattr(field, "concrete", False) and not field.many_to_many
    }


@pytest.mark.parametrize(
    "model,fields",
    [
        (AnalysisRun, ANALYSIS_RUN_FIELDS),
        (StockAnalysis, STOCK_ANALYSIS_FIELDS),
        (Prediction, PREDICTION_FIELDS),
    ],
)
def test_manifest_field_whitelist_covers_every_concrete_model_field(
    model: type, fields: tuple[str, ...]
) -> None:
    """Fails closed if a migration adds/removes a concrete field on any of
    the three manifest-covered models without updating its whitelist here.
    """
    assert set(fields) == _concrete_field_names(model)
    assert len(fields) == len(set(fields)), "field whitelist must not repeat a field"


def test_row_digest_changes_when_any_covered_field_changes() -> None:
    base = {name: "x" for name in ANALYSIS_RUN_FIELDS}
    base["issued_on_time"] = True
    digest = row_digest(ANALYSIS_RUN_MODEL, base)
    for field in ANALYSIS_RUN_FIELDS:
        mutated = dict(base)
        mutated[field] = "different" if field != "issued_on_time" else False
        assert row_digest(ANALYSIS_RUN_MODEL, mutated) != digest, field


def test_row_digest_is_stable_across_equivalent_python_types() -> None:
    run_id = uuid.uuid4()
    values_a = {
        "id": run_id,
        "generated_at": datetime(2024, 1, 2, 9, 30, tzinfo=UTC),
        "data_cutoff": datetime(2024, 1, 2, 9, 0, tzinfo=UTC),
        "target_date": date(2024, 1, 2),
        "issued_on_time": True,
        "universe_snapshot_id": uuid.uuid4(),
        "config_version": "v1",
        "config_hash": "a" * 64,
        "code_revision": "deadbeef",
        "status": "complete",
    }
    values_b = dict(values_a)
    # Same identity, re-typed as `str` (as `.values()` and JSON round trips
    # can present them): must hash identically to the original typed dict.
    values_b["id"] = str(run_id)
    values_b["universe_snapshot_id"] = str(values_a["universe_snapshot_id"])
    assert row_digest(ANALYSIS_RUN_MODEL, values_a) == row_digest(ANALYSIS_RUN_MODEL, values_b)


def test_row_digest_distinguishes_decimal_precision() -> None:
    base = {name: "x" for name in STOCK_ANALYSIS_FIELDS}
    base["current_price"] = Decimal("10.500000")
    other = dict(base)
    other["current_price"] = Decimal("10.5")
    # Decimal("10.500000") != Decimal("10.5") as *stored strings*, matching
    # exact database precision rather than numeric equality.
    assert row_digest(STOCK_ANALYSIS_MODEL, base) != row_digest(STOCK_ANALYSIS_MODEL, other)


def test_canonical_row_json_rejects_unknown_model() -> None:
    with pytest.raises(ManifestPayloadError):
        canonical_row_json("NotAModel", {})


def test_canonical_row_json_rejects_incomplete_row() -> None:
    incomplete = {name: "x" for name in PREDICTION_FIELDS if name != "calculation"}
    with pytest.raises(ManifestPayloadError):
        canonical_row_json(PREDICTION_MODEL, incomplete)


def test_manifest_entry_round_trips() -> None:
    entry = ManifestEntry(model=ANALYSIS_RUN_MODEL, row_id=str(uuid.uuid4()), digest="0" * 64)
    assert ManifestEntry.from_json(entry.to_json()) == entry


def test_manifest_entry_row_id_format_is_model_specific() -> None:
    # StockAnalysis uses a plain positive-integer AutoField primary key.
    entry = ManifestEntry(model=STOCK_ANALYSIS_MODEL, row_id="7", digest="0" * 64)
    assert ManifestEntry.from_json(entry.to_json()) == entry
    for bad_row_id in ("0", "-1", "7.0", "007", str(uuid.uuid4()), "abc"):
        with pytest.raises(ManifestPayloadError):
            ManifestEntry.from_json(
                {"model": STOCK_ANALYSIS_MODEL, "row_id": bad_row_id, "digest": "0" * 64}
            )
    # AnalysisRun/Prediction use a UUIDField primary key: a plain integer
    # string must be rejected even though it would be valid for
    # StockAnalysis.
    for model in (ANALYSIS_RUN_MODEL, PREDICTION_MODEL):
        with pytest.raises(ManifestPayloadError):
            ManifestEntry.from_json({"model": model, "row_id": "7", "digest": "0" * 64})


@pytest.mark.parametrize(
    "raw",
    [
        {"model": ANALYSIS_RUN_MODEL, "row_id": "x"},  # missing digest
        {"model": ANALYSIS_RUN_MODEL, "row_id": "x", "digest": "0" * 64, "extra": "y"},
        {"model": "Bogus", "row_id": "x", "digest": "0" * 64},
        {"model": ANALYSIS_RUN_MODEL, "row_id": "", "digest": "0" * 64},
        {"model": ANALYSIS_RUN_MODEL, "row_id": "x", "digest": "Z" * 64},
        {"model": ANALYSIS_RUN_MODEL, "row_id": "x", "digest": "0" * 63},
        "not-a-dict",
    ],
)
def test_manifest_entry_rejects_malformed_payload(raw: object) -> None:
    with pytest.raises(ManifestPayloadError):
        ManifestEntry.from_json(raw)


def test_sorted_entries_is_deterministic_regardless_of_input_order() -> None:
    a = ManifestEntry(model=PREDICTION_MODEL, row_id="1", digest="0" * 64)
    b = ManifestEntry(model=ANALYSIS_RUN_MODEL, row_id="2", digest="0" * 64)
    c = ManifestEntry(model=STOCK_ANALYSIS_MODEL, row_id="3", digest="0" * 64)
    assert sorted_entries([a, b, c]) == sorted_entries([c, b, a]) == [b, a, c]


def test_envelope_round_trips_through_canonical_bytes() -> None:
    run_id = uuid.uuid4()
    listing_id = str(uuid.uuid4())
    entries = [
        ManifestEntry(model=ANALYSIS_RUN_MODEL, row_id=str(run_id), digest="a" * 64),
        ManifestEntry(model=STOCK_ANALYSIS_MODEL, row_id="1", digest="b" * 64),
    ]
    plan = ManifestPlan(
        eligible_listing_ids=(listing_id,),
        predictions=(
            PlanPredictionKey(
                listing_id=listing_id, horizon="short", evidence_role="decision", count=1
            ),
        ),
    )
    envelope = build_manifest_envelope(run_id=run_id, plan=plan, entries=entries)
    payload = dumps_canonical_envelope(envelope)
    parsed_run_id, parsed_plan, parsed_entries = parse_manifest_envelope(payload)
    assert parsed_run_id == str(run_id)
    assert parsed_plan == plan
    assert parsed_entries == sorted_entries(entries)


def test_parse_manifest_envelope_rejects_reordered_top_level_keys() -> None:
    """A byte-for-byte reordering of the envelope's own top-level object
    keys parses to the identical structure (`set`-based shape checks and
    per-field validation cannot see key order at all) but is not the
    canonical writer's own byte encoding, and must be rejected (finding 5)."""
    run_id = str(uuid.uuid4())
    canonical = {
        "contract": "analysis-output-manifest@1",
        "run_id": run_id,
        "plan": {"eligible_listing_ids": [], "predictions": []},
        "entries": [],
    }
    canonical_payload = dumps_canonical_envelope(canonical)
    parse_manifest_envelope(canonical_payload)  # sanity: the canonical form is accepted

    reordered_payload = json.dumps(
        {
            "entries": [],
            "run_id": run_id,
            "plan": {"predictions": [], "eligible_listing_ids": []},
            "contract": "analysis-output-manifest@1",
        },
        separators=(",", ":"),
    ).encode("utf-8")
    assert json.loads(reordered_payload) == json.loads(canonical_payload)
    assert reordered_payload != canonical_payload
    with pytest.raises(ManifestPayloadError):
        parse_manifest_envelope(reordered_payload)


def test_parse_manifest_envelope_rejects_added_whitespace() -> None:
    run_id = str(uuid.uuid4())
    canonical = {
        "contract": "analysis-output-manifest@1",
        "run_id": run_id,
        "plan": {"eligible_listing_ids": [], "predictions": []},
        "entries": [],
    }
    canonical_payload = dumps_canonical_envelope(canonical)
    indented_payload = json.dumps(canonical, sort_keys=True, indent=2).encode("utf-8")
    assert json.loads(indented_payload) == json.loads(canonical_payload)
    assert indented_payload != canonical_payload
    with pytest.raises(ManifestPayloadError):
        parse_manifest_envelope(indented_payload)


def test_parse_manifest_envelope_rejects_trailing_whitespace() -> None:
    run_id = str(uuid.uuid4())
    canonical = {
        "contract": "analysis-output-manifest@1",
        "run_id": run_id,
        "plan": {"eligible_listing_ids": [], "predictions": []},
        "entries": [],
    }
    canonical_payload = dumps_canonical_envelope(canonical)
    trailing_payload = canonical_payload + b"\n"
    with pytest.raises(ManifestPayloadError):
        parse_manifest_envelope(trailing_payload)


def test_parse_manifest_envelope_rejects_alternate_escaping() -> None:
    """A `run_id` string escaped character-by-character as `\\u00XX`
    sequences decodes to the identical Python string as the plain
    canonical encoding, but is not the canonical writer's own byte
    encoding (which never emits an escape for a plain ASCII hex/hyphen
    character)."""
    run_id = str(uuid.uuid4())
    canonical = {
        "contract": "analysis-output-manifest@1",
        "run_id": run_id,
        "plan": {"eligible_listing_ids": [], "predictions": []},
        "entries": [],
    }
    canonical_payload = dumps_canonical_envelope(canonical)
    escaped_run_id = "".join(f"\\u{ord(ch):04x}" for ch in run_id)
    escaped_payload = (
        b'{"contract":"analysis-output-manifest@1",'
        b'"entries":[],'
        b'"plan":{"eligible_listing_ids":[],"predictions":[]},'
        b'"run_id":"' + escaped_run_id.encode("ascii") + b'"}'
    )
    assert json.loads(escaped_payload) == json.loads(canonical_payload)
    assert escaped_payload != canonical_payload
    with pytest.raises(ManifestPayloadError):
        parse_manifest_envelope(escaped_payload)


def test_parse_manifest_envelope_rejects_duplicate_json_keys() -> None:
    payload = (
        b'{"contract":"analysis-output-manifest@1","contract":"tampered","run_id":"x",'
        b'"plan":{"eligible_listing_ids":[],"predictions":[]},"entries":[]}'
    )
    with pytest.raises(ManifestPayloadError):
        parse_manifest_envelope(payload)


def test_parse_manifest_envelope_rejects_wrong_contract() -> None:
    payload = dumps_canonical_envelope(
        {
            "contract": "something-else@1",
            "run_id": str(uuid.uuid4()),
            "plan": {"eligible_listing_ids": [], "predictions": []},
            "entries": [],
        }
    )
    with pytest.raises(ManifestPayloadError):
        parse_manifest_envelope(payload)


def test_parse_manifest_envelope_rejects_non_uuid_run_id() -> None:
    payload = dumps_canonical_envelope(
        {
            "contract": "analysis-output-manifest@1",
            "run_id": "not-a-uuid",
            "plan": {"eligible_listing_ids": [], "predictions": []},
            "entries": [],
        }
    )
    with pytest.raises(ManifestPayloadError):
        parse_manifest_envelope(payload)


@pytest.mark.parametrize(
    "envelope",
    [
        {
            "contract": "analysis-output-manifest@1",
            "run_id": "x",
            "plan": {"eligible_listing_ids": [], "predictions": []},
        },  # missing entries
        {
            "contract": "analysis-output-manifest@1",
            "run_id": "x",
            "plan": {"eligible_listing_ids": [], "predictions": []},
            "entries": [],
            "extra": 1,
        },
        {
            "contract": "analysis-output-manifest@1",
            "run_id": "",
            "plan": {"eligible_listing_ids": [], "predictions": []},
            "entries": [],
        },
        {
            "contract": "analysis-output-manifest@1",
            "run_id": "x",
            "plan": {"eligible_listing_ids": [], "predictions": []},
            "entries": "not-a-list",
        },
        {
            "contract": "analysis-output-manifest@1",
            "run_id": str(uuid.uuid4()),
            "entries": [],
        },  # missing plan
    ],
)
def test_parse_manifest_envelope_rejects_malformed_shape(envelope: dict[str, object]) -> None:
    with pytest.raises(ManifestPayloadError):
        parse_manifest_envelope(dumps_canonical_envelope(envelope))


def test_parse_manifest_envelope_rejects_duplicate_entry() -> None:
    row_id = str(uuid.uuid4())
    envelope = {
        "contract": "analysis-output-manifest@1",
        "run_id": str(uuid.uuid4()),
        "plan": {"eligible_listing_ids": [], "predictions": []},
        "entries": [
            {"model": ANALYSIS_RUN_MODEL, "row_id": row_id, "digest": "a" * 64},
            {"model": ANALYSIS_RUN_MODEL, "row_id": row_id, "digest": "b" * 64},
        ],
    }
    with pytest.raises(ManifestPayloadError):
        parse_manifest_envelope(dumps_canonical_envelope(envelope))


def test_parse_manifest_envelope_rejects_out_of_order_entries() -> None:
    """Entries in a valid-but-noncanonically-ordered sequence must be
    rejected outright, never silently normalized into canonical order --
    finding 1's closure for the row-entry list specifically."""
    first = ManifestEntry(model=ANALYSIS_RUN_MODEL, row_id=str(uuid.uuid4()), digest="a" * 64)
    second = ManifestEntry(model=STOCK_ANALYSIS_MODEL, row_id="1", digest="b" * 64)
    canonical = sorted_entries([first, second])
    assert canonical == [first, second]  # AnalysisRun sorts before StockAnalysis
    envelope = {
        "contract": "analysis-output-manifest@1",
        "run_id": str(uuid.uuid4()),
        "plan": {"eligible_listing_ids": [], "predictions": []},
        "entries": [entry.to_json() for entry in (second, first)],  # deliberately reversed
    }
    with pytest.raises(ManifestPayloadError):
        parse_manifest_envelope(dumps_canonical_envelope(envelope))
    # The canonical order itself must still be accepted.
    envelope["entries"] = [entry.to_json() for entry in canonical]
    parse_manifest_envelope(dumps_canonical_envelope(envelope))


def test_plan_prediction_key_round_trips_and_rejects_malformed() -> None:
    listing_id = str(uuid.uuid4())
    key = PlanPredictionKey(
        listing_id=listing_id, horizon="short", evidence_role="decision", count=1
    )
    assert PlanPredictionKey.from_json(key.to_json()) == key
    for raw in (
        {"listing_id": listing_id, "horizon": "short", "evidence_role": "decision"},  # no count
        {**key.to_json(), "listing_id": "not-a-uuid"},
        {**key.to_json(), "horizon": "bogus"},
        {**key.to_json(), "evidence_role": "bogus"},
        {**key.to_json(), "count": 0},
        {**key.to_json(), "count": True},
        {**key.to_json(), "count": "1"},
    ):
        with pytest.raises(ManifestPayloadError):
            PlanPredictionKey.from_json(raw)


def test_manifest_plan_rejects_duplicate_listing_id() -> None:
    listing_id = str(uuid.uuid4())
    with pytest.raises(ManifestPayloadError):
        ManifestPlan.from_json(
            {"eligible_listing_ids": [listing_id, listing_id], "predictions": []}
        )


def test_manifest_plan_rejects_duplicate_prediction_key() -> None:
    listing_id = str(uuid.uuid4())
    entry = {"listing_id": listing_id, "horizon": "short", "evidence_role": "decision", "count": 1}
    with pytest.raises(ManifestPayloadError):
        ManifestPlan.from_json(
            {"eligible_listing_ids": [listing_id], "predictions": [entry, entry]}
        )


def test_manifest_plan_requires_canonical_listing_id_order() -> None:
    """A byte-identical plan payload whose `eligible_listing_ids` is not
    already in canonical sorted order must be rejected outright, never
    silently normalized -- finding 1's noncanonical-bytes closure."""
    a, b = sorted((str(uuid.uuid4()), str(uuid.uuid4())))
    with pytest.raises(ManifestPayloadError):
        ManifestPlan.from_json({"eligible_listing_ids": [b, a], "predictions": []})
    # The canonical order itself must still be accepted.
    ManifestPlan.from_json({"eligible_listing_ids": [a, b], "predictions": []})


def test_manifest_plan_requires_canonical_prediction_order() -> None:
    a, b = sorted((str(uuid.uuid4()), str(uuid.uuid4())))
    entry_a = {"listing_id": a, "horizon": "short", "evidence_role": "decision", "count": 1}
    entry_b = {"listing_id": b, "horizon": "short", "evidence_role": "decision", "count": 1}
    with pytest.raises(ManifestPayloadError):
        ManifestPlan.from_json({"eligible_listing_ids": [a, b], "predictions": [entry_b, entry_a]})
    # The canonical order itself must still be accepted.
    ManifestPlan.from_json({"eligible_listing_ids": [a, b], "predictions": [entry_a, entry_b]})


def test_build_output_plan_covers_decision_medium_and_long_lanes() -> None:
    listing_id = uuid.uuid4()
    plan = build_output_plan(
        eligible_listing_ids={listing_id},
        decision_horizons=frozenset({"short"}),
        medium_active=True,
        long_active=True,
    )
    assert plan.eligible_listing_ids == (str(listing_id),)
    keys = {(p.horizon, p.evidence_role) for p in plan.predictions}
    assert ("short", "decision") in keys
    for horizon in MEDIUM_HORIZONS:
        assert (horizon, "advisory") in keys
    for horizon in LONG_HORIZONS:
        assert (horizon, "advisory") in keys
    assert len(plan.predictions) == 1 + len(MEDIUM_HORIZONS) + len(LONG_HORIZONS)


def test_build_output_plan_excludes_inactive_lanes() -> None:
    listing_id = uuid.uuid4()
    plan = build_output_plan(
        eligible_listing_ids={listing_id},
        decision_horizons=frozenset({"short"}),
        medium_active=False,
        long_active=False,
    )
    assert len(plan.predictions) == 1
    assert plan.predictions[0].evidence_role == "decision"


def test_actual_output_plan_counts_duplicate_prediction_keys() -> None:
    class _Row:
        def __init__(self, listing_id: object, horizon: str, evidence_role: str) -> None:
            self.listing_id = listing_id
            self.horizon = horizon
            self.evidence_role = evidence_role

    listing_id = uuid.uuid4()

    class _Analysis:
        def __init__(self, listing_id: object) -> None:
            self.listing_id = listing_id

    plan = actual_output_plan(
        [_Analysis(listing_id)],
        [
            _Row(listing_id, "short", "decision"),
            _Row(listing_id, "short", "decision"),
        ],
    )
    assert plan.predictions == (
        PlanPredictionKey(
            listing_id=str(listing_id), horizon="short", evidence_role="decision", count=2
        ),
    )


def test_lookup_manifest_distinguishes_missing_from_ambiguous() -> None:
    run_id = uuid.uuid4()
    lookup = lookup_manifest(run_id)
    assert lookup.asset is None
    assert lookup.count == 0

    def _register(path: str) -> DataAsset:
        return DataAsset.objects.create(
            provider="stanstock",
            kind="analysis_output_manifest",
            subject=str(run_id),
            relative_path=path,
            sha256="a" * 64,
            retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
            available_at=datetime(2024, 1, 1, tzinfo=UTC),
        )

    _register("research/analysis/one.json")
    lookup = lookup_manifest(run_id)
    assert lookup.asset is not None
    assert lookup.count == 1

    _register("research/analysis/two.json")
    lookup = lookup_manifest(run_id)
    assert lookup.asset is None
    assert lookup.count == 2

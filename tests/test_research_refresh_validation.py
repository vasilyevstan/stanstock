"""Focused tests for `research.refresh_validation.verify_analysis_output_manifest`
and the analysis-output-manifest writer it reads back
(`research.service._write_analysis_output_manifest`, wired into
`analyze_snapshot`, and its `data.live_us.run_us_daily` orphan-cleanup
extension).

Three layers, deliberately kept separate:

* Group 1 -- manifest *resolution* (missing/ambiguous/malformed/run-id
  mismatch): a bare `AnalysisRun` with no `StockAnalysis`/`Prediction` rows
  is enough, since every one of these failures happens before the reader
  ever needs to query them.
* Group 2 -- full real orchestration through `analyze_snapshot` (decision
  lane only, `provider="twelve_data"` to match production's own hard-coded
  price-provider expectation): happy path, then digest-divergence via a
  direct bulk `.update()`/insert against `StockAnalysis` (`StockAnalysis`
  carries no database trigger the way `DataAsset`/`FundamentalFact`/
  `FxRate`/`Prediction` do, so a bulk queryset `.update()` reaches its row
  unobstructed) and via a normal `Prediction.objects.create()` insert
  (`Prediction` *is* protected by a real `BEFORE UPDATE`/`BEFORE DELETE`
  database trigger -- see `research/migrations/0003_prediction_immutable.py`
  and `0005_reinstate_prediction_immutability.py` -- which blocks even a
  bulk queryset `.update()`/`.delete()` that bypasses `Prediction.save()`'s
  own Python-level immutability guard entirely; an *extra* row is
  therefore the only way to tamper a persisted run's `Prediction` set at
  all, which this manifest closes just as completely as a mutated row
  would be). Any such row-level tamper must trip the manifest's
  complete-row-digest check *before* any deeper semantic check ever runs,
  so this group cannot be used to exercise those deeper checks -- seeing
  a real row change is exactly what makes them unreachable in this group.
* Group 3 -- the deeper semantic checks (scenario-mirror consistency,
  medium panel binding, long fact/classification/filing binding, source
  asset identity/cutoff) exercised in isolation, against hand-built
  fixtures that are wired together in memory (never saved, when the
  function under test only reads attributes) or persisted only where the
  function itself performs a database lookup. This is the only way to
  reach an *internally inconsistent* row deliberately, since -- correctly
  -- the manifest itself can never be rewritten once persisted (it is
  registered as an immutable `DataAsset`) and a genuinely persisted
  `Prediction` row cannot be mutated at all once written, by trigger, not
  merely by convention.

Group 4 covers the writer's own transactional cleanup at the point closest
to its actual failure modes: constructor/write/register failure removing
only a file it just created, preserving a genuinely pre-existing one, and
never letting a cleanup-step fault itself replace the original error.
"""

from __future__ import annotations

import hashlib
import io
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import polars as pl
import pytest
from django.conf import settings
from django.db import DatabaseError, connection

from stanstock.core.verification_types import AssetRef, RefreshVerificationError
from stanstock.data.asof import AsOfData, raw_price_asset_for
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.management.config_loader import default_us_scoring_config_path
from stanstock.data.models import (
    Company,
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    Listing,
    ProviderRecord,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.data.provider_policy import TWELVE_DATA_PROVIDER
from stanstock.research import service
from stanstock.research.forecast_config import (
    load_medium_forecast_config,
    medium_forecast_config_hash,
)
from stanstock.research.forecasting import FORECAST_SCENARIO_SCHEMA_VERSION
from stanstock.research.long_forecasts import (
    classification_payload,
    fact_payload,
    fact_reference,
)
from stanstock.research.medium_forecasts import (
    PANEL_KIND,
    PANEL_PROVIDER,
    PANEL_SCHEMA,
    PANEL_SCHEMA_VERSION,
    MediumPanelPriceInput,
    asset_identity,
    calendar_sessions_through,
    dedupe_assets,
    hash_json,
    reconstruct_medium_forecast_panel,
    serialize_medium_forecast_panel,
)
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis
from stanstock.research.refresh_evidence import (
    ANALYSIS_OUTPUT_MANIFEST_KIND,
    ManifestPlan,
    build_manifest_envelope,
    build_output_plan,
    dumps_canonical_envelope,
)
from stanstock.research.refresh_validation import (
    _MEDIUM_HORIZONS,
    LongLaneExpectation,
    MediumLaneExpectation,
    _AssetRegistry,
    _require_authoritative_medium_panel,
    _require_canonical_scenario_shape,
    _require_decision_source_closure_exact,
    _require_long_evidence_binding,
    _require_medium_panel_binding,
    _require_medium_source_closure_exact,
    _require_prediction_ground_truth,
    _require_prediction_horizon_multiset,
    _require_scenario_document_shape,
    _require_scenario_mirror_consistency,
    _require_source_assets_bound,
    _require_stock_analysis_price_source_bound,
    verify_analysis_output_manifest,
)
from stanstock.research.service import (
    PersistedAnalysis,
    _analysis_output_manifest_entries,
    _write_analysis_output_manifest,
    analyze_listing,
    analyze_snapshot,
)
from test_long_forecasts import (
    DECISION_TIME,
    TARGET_DATE,
    _company_evidence,
    _small_peer_config,
    _write_price_asset,
)
from test_long_forecasts import (
    _listing as _long_listing,
)

pytestmark = pytest.mark.django_db

DECISION_HORIZONS = frozenset({"short"})


@pytest.fixture(autouse=True)
def _default_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`verify_analysis_output_manifest` resolves its manifest asset (and
    every other asset reference) through `open_asset_store()` -- the
    *default* `AssetStore`, never a store passed in by the caller -- so
    every test in this file must keep the default store's root pointed at
    its own `tmp_path`, exactly like the writer side's own `AssetStore
    (tmp_path)`."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _snapshot(*, slug: str, as_of_date=TARGET_DATE) -> UniverseSnapshot:
    universe = Universe.objects.create(slug=slug, name=slug, config_version="test-v1")
    return UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=as_of_date,
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="a" * 64,
    )


def _decision_only_results(tmp_path: Path, *, listing_count: int = 2) -> tuple[AssetStore, list]:
    """A real, fully persisted decision-only observed run (no medium/long
    advisory lane active): `provider="twelve_data"` throughout, matching the
    validator's own hard-coded price-provider expectation for a genuine
    scheduled-refresh run."""
    store = AssetStore(tmp_path)
    snapshot = _snapshot(slug=f"manifest-decision-{uuid4().hex[:8]}")
    listings = [_long_listing(f"MD{index}") for index in range(listing_count)]
    for listing in listings:
        UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
        _write_price_asset(store, listing, close=40.0 + listings.index(listing) * 5)
    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        issued_on_time=True,
        provider="twelve_data",
        store=store,
        config_path=default_us_scoring_config_path(),
    )
    return store, results


def _write_subject_price_asset(store: AssetStore, subject: str, *, close: float) -> DataAsset:
    """Same session/price generation `_write_price_asset` uses, but for an
    arbitrary `subject` string (e.g. a benchmark ticker rather than a
    `Listing`'s own ticker) so a medium-lane-active run can register its
    benchmark price series."""
    sessions: list = []
    cursor = TARGET_DATE
    while len(sessions) < 320:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor -= timedelta(days=1)
    sessions.reverse()
    frame = pl.DataFrame(
        {
            "date": sessions,
            "close": [close + index * 0.02 for index in range(len(sessions))],
            "volume": [2_000_000 + index for index in range(len(sessions))],
        },
        schema_overrides={"date": pl.Date, "close": pl.Float64, "volume": pl.Int64},
    )
    raw_stored = store.write_bytes(
        f"long-tests/{subject}-{uuid4().hex}-raw.json",
        f'{{"symbol":"{subject}","source":"synthetic-test"}}'.encode(),
    )
    raw_asset = register_asset(
        provider="twelve_data",
        kind="raw_price_history",
        subject=subject,
        stored=raw_stored,
        retrieved_at=DECISION_TIME,
        available_at=DECISION_TIME,
        period_start=sessions[0],
        period_end=sessions[-1],
    )
    stored = store.write_frame(f"long-tests/{subject}.parquet", frame)
    return register_asset(
        provider="twelve_data",
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=DECISION_TIME,
        available_at=DECISION_TIME,
        period_start=sessions[0],
        period_end=sessions[-1],
        metadata={
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
            "raw_asset_id": str(raw_asset.id),
            "raw_sha256": raw_asset.sha256,
        },
    )


def _medium_active_snapshot(
    tmp_path: Path, *, listing_count: int = 1
) -> tuple[AssetStore, UniverseSnapshot, list[Listing]]:
    """An observed snapshot with medium-lane-eligible listings and a
    registered benchmark series, ready for a real `analyze_snapshot(...,
    benchmark_subject=...)` call that activates the medium advisory lane
    (`default_us_scoring_config_path()`'s own `us-price-baseline-v2` is
    exactly the scoring version `config/forecasts/us-price-medium-v1.yml`
    enables)."""
    store = AssetStore(tmp_path)
    snapshot = _snapshot(slug=f"manifest-medium-{uuid4().hex[:8]}")
    listings = [_long_listing(f"MED{index}") for index in range(listing_count)]
    for index, listing in enumerate(listings):
        UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
        _write_subject_price_asset(store, listing.ticker, close=40.0 + index * 5)
    _write_subject_price_asset(store, "SPY", close=300.0)
    return store, snapshot, listings


def _verify(run: AnalysisRun, listings_by_id: dict, **overrides) -> object:
    kwargs = {
        "run": run,
        "eligible_listing_ids": set(listings_by_id),
        "listings_by_id": listings_by_id,
        "code_revision": run.code_revision,
        "scoring_config_version": run.config_version,
        "scoring_config_hash": run.config_hash,
        "decision_horizons": DECISION_HORIZONS,
        "medium": None,
        "long": None,
    }
    kwargs.update(overrides)
    return verify_analysis_output_manifest(**kwargs)


def _reason(exc_info: pytest.ExceptionInfo) -> str:
    assert isinstance(exc_info.value, RefreshVerificationError)
    return exc_info.value.reason_code


# ---------------------------------------------------------------------------
# Group 1: manifest resolution
# ---------------------------------------------------------------------------


def _bare_run(*, target_date=TARGET_DATE, generated_at: datetime = DECISION_TIME) -> AnalysisRun:
    snapshot = _snapshot(slug=f"manifest-bare-{uuid4().hex[:8]}", as_of_date=target_date)
    return AnalysisRun.objects.create(
        generated_at=generated_at,
        data_cutoff=generated_at,
        target_date=target_date,
        issued_on_time=True,
        universe_snapshot=snapshot,
        config_version="test-v1",
        config_hash="b" * 64,
        code_revision="rev-bare-test",
    )


def test_missing_manifest_fails_closed() -> None:
    run = _bare_run()
    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, {})
    assert _reason(exc_info) == "analysis_output_manifest_missing"


def test_ambiguous_manifest_fails_closed(tmp_path: Path) -> None:
    run = _bare_run()
    store = AssetStore(tmp_path)
    for index in range(2):
        stored = store.write_bytes(f"manifest-ambiguous/{index}.json", f"{{}}{index}".encode())
        register_asset(
            provider="stanstock",
            kind=ANALYSIS_OUTPUT_MANIFEST_KIND,
            subject=str(run.id),
            stored=stored,
            retrieved_at=run.generated_at,
            available_at=run.generated_at,
        )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, {})
    assert _reason(exc_info) == "analysis_output_manifest_ambiguous"


def test_malformed_manifest_envelope_fails_closed(tmp_path: Path) -> None:
    run = _bare_run()
    store = AssetStore(tmp_path)
    stored = store.write_bytes("manifest-malformed/manifest.json", b"not json at all")
    register_asset(
        provider="stanstock",
        kind=ANALYSIS_OUTPUT_MANIFEST_KIND,
        subject=str(run.id),
        stored=stored,
        retrieved_at=run.generated_at,
        available_at=run.generated_at,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, {})
    assert _reason(exc_info) == "analysis_output_manifest_malformed"


def test_manifest_run_id_mismatch_fails_closed(tmp_path: Path) -> None:
    run = _bare_run()
    store = AssetStore(tmp_path)
    envelope = build_manifest_envelope(
        run_id=uuid.uuid4(), plan=ManifestPlan(eligible_listing_ids=(), predictions=()), entries=[]
    )
    payload = dumps_canonical_envelope(envelope)
    stored = store.write_bytes("manifest-mismatch/manifest.json", payload)
    register_asset(
        provider="stanstock",
        kind=ANALYSIS_OUTPUT_MANIFEST_KIND,
        subject=str(run.id),
        stored=stored,
        retrieved_at=run.generated_at,
        available_at=run.generated_at,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, {})
    assert _reason(exc_info) == "analysis_output_manifest_run_mismatch"


def test_manifest_identity_timestamp_mismatch_fails_closed(tmp_path: Path) -> None:
    """The manifest asset's own `retrieved_at`/`available_at` must exactly
    equal this run's own `generated_at` -- proving the manifest was
    written at the run's own output time, never retrospectively attached
    to an unrelated run long after the fact."""
    run = _bare_run()
    store = AssetStore(tmp_path)
    envelope = build_manifest_envelope(
        run_id=run.id, plan=ManifestPlan(eligible_listing_ids=(), predictions=()), entries=[]
    )
    payload = dumps_canonical_envelope(envelope)
    stored = store.write_bytes("manifest-retro/manifest.json", payload)
    retrospective_time = run.generated_at + timedelta(days=1)
    register_asset(
        provider="stanstock",
        kind=ANALYSIS_OUTPUT_MANIFEST_KIND,
        subject=str(run.id),
        stored=stored,
        retrieved_at=retrospective_time,
        available_at=retrospective_time,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, {})
    assert _reason(exc_info) == "analysis_output_manifest_identity_invalid"


def test_manifest_recorded_plan_mismatch_with_matching_rows_fails_closed(tmp_path: Path) -> None:
    """A hand-tampered manifest whose *recorded* plan disagrees with the
    reviewed eligible membership/lane expectations must fail even when the
    actual persisted rows happen to still match the reviewed plan exactly
    -- proving the manifest's own stored plan is independently checked, not
    only the live rows."""
    store = AssetStore(tmp_path)
    run = _bare_run()
    listing = _long_listing("PLANMISMATCH")
    persisted = _bare_persisted_analysis(run=run, listing=listing)
    real_plan = build_output_plan(
        eligible_listing_ids={listing.id},
        decision_horizons=frozenset({"short"}),
        medium_active=False,
        long_active=False,
    )
    phantom_plan = build_output_plan(
        eligible_listing_ids={listing.id, uuid4()},
        decision_horizons=frozenset({"short"}),
        medium_active=False,
        long_active=False,
    )
    envelope = build_manifest_envelope(
        run_id=run.id,
        plan=phantom_plan,
        entries=_analysis_output_manifest_entries(run, [persisted]),
    )
    payload = dumps_canonical_envelope(envelope)
    stored = store.write_bytes(f"research/analysis/{run.id}/output-manifest.json", payload)
    register_asset(
        provider="stanstock",
        kind=ANALYSIS_OUTPUT_MANIFEST_KIND,
        subject=str(run.id),
        stored=stored,
        retrieved_at=run.generated_at,
        available_at=run.generated_at,
    )
    assert real_plan != phantom_plan

    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, {listing.id: listing}, decision_horizons=frozenset({"short"}))
    assert _reason(exc_info) == "analysis_output_manifest_plan_mismatch"


# ---------------------------------------------------------------------------
# Group 2: full real orchestration (decision-only)
# ---------------------------------------------------------------------------


def test_verify_analysis_output_manifest_succeeds_for_decision_only_run(tmp_path: Path) -> None:
    _store, results = _decision_only_results(tmp_path)
    run = results[0].run
    listings_by_id = {result.analysis.listing_id: result.analysis.listing for result in results}

    outcome = _verify(run, listings_by_id)

    assert outcome.summary["analysis_run_id"] == str(run.id)
    assert outcome.summary["stock_analysis_count"] == len(results)
    assert outcome.summary["prediction_count"] == len(results)  # one short decision row each
    manifest_ref = outcome.asset_refs[0]
    assert manifest_ref.kind == ANALYSIS_OUTPUT_MANIFEST_KIND


def test_v3_exact_source_run_round_trips_through_output_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot(slug=f"manifest-v3-{uuid4().hex[:8]}")
    listing = _long_listing("MV3")
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    listing_asset = _write_subject_price_asset(store, listing.ticker, close=45.0)
    benchmark_asset = _write_subject_price_asset(store, "SPY", close=300.0)
    revision = "1" * 40
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", revision)
    monkeypatch.setattr("stanstock.research.service.clean_git_revision", lambda _root: revision)

    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        issued_on_time=True,
        provider="twelve_data",
        benchmark_subject="SPY",
        store=store,
        config_path=(
            Path(__file__).resolve().parents[1] / "config/scoring/us-price-baseline-v3.yml"
        ),
        long_forecast_requested=False,
    )
    run = results[0].run
    listings_by_id = {listing.id: listing}

    outcome = _verify(run, listings_by_id)

    assert outcome.summary["stock_analysis_count"] == 1
    assert outcome.summary["prediction_count"] == 1
    assert {entry["id"] for entry in results[0].computation.source_assets} == {
        str(listing_asset.id),
        str(benchmark_asset.id),
    }


def test_stock_analysis_row_mutation_is_detected(tmp_path: Path) -> None:
    _store, results = _decision_only_results(tmp_path)
    run = results[0].run
    listings_by_id = {result.analysis.listing_id: result.analysis.listing for result in results}
    target = results[0].analysis
    StockAnalysis.objects.filter(pk=target.pk).update(overall_score=Decimal("1.00"))

    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, listings_by_id)
    assert _reason(exc_info) == "analysis_output_manifest_diverged"


def test_analysis_run_row_mutation_is_detected(tmp_path: Path) -> None:
    """`AnalysisRun` itself is a plain, unprotected model (no database
    trigger the way `Prediction`/`DataAsset` are); its own row must be
    covered by the manifest just as thoroughly as its children."""
    _store, results = _decision_only_results(tmp_path)
    run = results[0].run
    listings_by_id = {result.analysis.listing_id: result.analysis.listing for result in results}

    AnalysisRun.objects.filter(pk=run.pk).update(status="tampered")
    run.refresh_from_db()

    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, listings_by_id)
    assert _reason(exc_info) == "analysis_output_manifest_diverged"


def test_extra_prediction_row_is_detected(tmp_path: Path) -> None:
    _store, results = _decision_only_results(tmp_path)
    run = results[0].run
    listings_by_id = {result.analysis.listing_id: result.analysis.listing for result in results}
    original = results[0].predictions[0]

    Prediction.objects.create(
        analysis=original.analysis,
        listing=original.listing,
        generated_at=original.generated_at,
        target_date=original.target_date,
        issued_on_time=original.issued_on_time,
        horizon=original.horizon,
        evidence_role=original.evidence_role,
        evidence_grade=original.evidence_grade,
        source_mode=original.source_mode,
        price_provider=original.price_provider,
        price_subject=original.price_subject,
        price_at_prediction=original.price_at_prediction,
        confidence=original.confidence,
        confidence_status=original.confidence_status,
        recommendation=original.recommendation,
        overall_score=original.overall_score,
        model_version="extra-untracked-version",
        method_version=original.method_version,
        config_hash=original.config_hash,
        data_cutoff=original.data_cutoff,
        code_revision=original.code_revision,
    )

    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, listings_by_id)
    assert _reason(exc_info) == "analysis_output_manifest_diverged"


def test_stock_analysis_row_transplant_is_detected(tmp_path: Path) -> None:
    """A `StockAnalysis`'s `listing_id` bulk-updated to a different,
    uninvolved listing (no database trigger blocks it, unlike
    `Prediction`/`DataAsset`) must be detected even though the row count is
    unchanged."""
    _store, results = _decision_only_results(tmp_path)
    run = results[0].run
    listings_by_id = {result.analysis.listing_id: result.analysis.listing for result in results}
    outsider = _long_listing("MDX")
    target = results[0].analysis

    StockAnalysis.objects.filter(pk=target.pk).update(listing_id=outsider.id)

    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, listings_by_id)
    assert _reason(exc_info) == "analysis_output_manifest_diverged"


def test_eligible_listing_set_mismatch_fails_without_any_row_tamper(tmp_path: Path) -> None:
    """The eligible membership set is supplied by the caller (the verified
    universe snapshot's own membership), never inferred from whichever
    `StockAnalysis` rows happen to exist; passing an incomplete set must
    fail even though every row is untouched and the manifest matches
    exactly."""
    _store, results = _decision_only_results(tmp_path)
    run = results[0].run
    listings_by_id = {result.analysis.listing_id: result.analysis.listing for result in results}
    incomplete = dict(list(listings_by_id.items())[:-1])

    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, incomplete)
    assert _reason(exc_info) == "stock_analysis_listing_mismatch"


def test_decision_horizons_mismatch_fails_via_precomputed_plan(tmp_path: Path) -> None:
    """Reviewing a decision-only run against a wider decision horizon set
    than it actually produced must fail via the precomputed-plan mismatch,
    proving the plan is genuinely enforced rather than merely stored."""
    _store, results = _decision_only_results(tmp_path)
    run = results[0].run
    listings_by_id = {result.analysis.listing_id: result.analysis.listing for result in results}

    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, listings_by_id, decision_horizons=frozenset({"short", "6m"}))
    assert _reason(exc_info) == "prediction_decision_multiset_mismatch"


def test_prediction_bulk_update_is_blocked_by_database_trigger(tmp_path: Path) -> None:
    """`Prediction` rows are protected against tampering at the database
    level, not only by `Prediction.save()`'s own Python-level guard: a
    bulk queryset `.update()`, which bypasses `save()` entirely, must
    still be rejected by the `BEFORE UPDATE` trigger installed in
    `research/migrations/0003_prediction_immutable.py` (reinstated for
    SQLite table rebuilds in `0005_reinstate_prediction_immutability.py`).
    This is why the semantic `prediction_listing_mismatch` check (proven
    directly against `_require_prediction_horizon_multiset` above) can
    never actually be reached against a genuinely persisted, transplanted
    `Prediction` row in this test module: the transplant itself is refused
    before the verifier ever runs."""
    _store, results = _decision_only_results(tmp_path)
    outsider = _long_listing("MDY")
    target = results[0].predictions[0]

    with pytest.raises(DatabaseError, match="immutable"):
        Prediction.objects.filter(pk=target.pk).update(listing_id=outsider.id)


def test_physical_asset_corruption_is_detected(tmp_path: Path) -> None:
    """`resolve_asset_ref` only proves a matching database row exists; the
    physical bytes on disk for a source asset silently swapped after the
    row was registered (its `sha256` on the row left untouched) must still
    be caught by the final checksum-read pass over every accumulated
    referenced asset -- not only the manifest asset itself."""
    store, results = _decision_only_results(tmp_path)
    run = results[0].run
    listings_by_id = {result.analysis.listing_id: result.analysis.listing for result in results}
    data_quality = results[0].analysis.data_quality
    source_assets = data_quality["source_assets"]
    price_entry = next(entry for entry in source_assets if entry.get("kind") == "price_history")
    asset = DataAsset.objects.get(pk=price_entry["id"])

    store.resolve(asset.relative_path).write_bytes(
        b"corrupted-bytes-do-not-match-registered-checksum"
    )

    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, listings_by_id)
    assert _reason(exc_info) == "asset_corrupt"


def test_missing_physical_asset_is_detected(tmp_path: Path) -> None:
    store, results = _decision_only_results(tmp_path)
    run = results[0].run
    listings_by_id = {result.analysis.listing_id: result.analysis.listing for result in results}
    data_quality = results[0].analysis.data_quality
    source_assets = data_quality["source_assets"]
    price_entry = next(entry for entry in source_assets if entry.get("kind") == "price_history")
    asset = DataAsset.objects.get(pk=price_entry["id"])

    store.resolve(asset.relative_path).unlink()

    with pytest.raises(RefreshVerificationError) as exc_info:
        _verify(run, listings_by_id)
    assert _reason(exc_info) == "asset_unreadable"


def test_verify_analysis_output_manifest_succeeds_for_real_medium_producer_run(
    tmp_path: Path,
) -> None:
    """End-to-end: a real `analyze_snapshot(..., benchmark_subject="SPY")`
    call (not a hand-built verifier fixture) produces genuine medium
    advisory `Prediction` rows bound to one authoritative, physically
    written `medium_forecast_panel` `DataAsset` and a real SPY benchmark
    price asset; `verify_analysis_output_manifest`'s `medium=` branch must
    accept that real output as-is -- proving the producer -> persisted
    manifest/assets -> top-level verifier contract for the medium lane in
    one committed test, not merely its isolated sub-checks."""
    store, snapshot, listings = _medium_active_snapshot(tmp_path)

    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        issued_on_time=True,
        provider="twelve_data",
        benchmark_subject="SPY",
        store=store,
        config_path=default_us_scoring_config_path(),
    )
    run = results[0].run
    listings_by_id = {result.analysis.listing_id: result.analysis.listing for result in results}
    assert (
        Prediction.objects.filter(
            evidence_role=Prediction.EvidenceRole.ADVISORY, horizon__in=["6m", "12m"]
        ).count()
        == len(listings) * 2
    )

    outcome = _verify(run, listings_by_id, medium=_medium_lane_expectation())

    assert outcome.summary["analysis_run_id"] == str(run.id)
    assert any(ref.kind == PANEL_KIND for ref in outcome.asset_refs)
    panel = DataAsset.objects.get(kind=PANEL_KIND, subject=str(run.id))
    panel_source_ids = {
        UUID(str(entry["id"]))
        for entry in panel.metadata["source_assets"]
        if isinstance(entry, dict)
    }
    panel_raw_ids = {
        raw_price_asset_for(DataAsset.objects.get(pk=source_id), cutoff=run.data_cutoff).id
        for source_id in panel_source_ids
    }
    result_ids = [ref.id for ref in outcome.asset_refs]
    for source_id in panel_source_ids | panel_raw_ids:
        assert result_ids.count(source_id) == 1


def test_verify_analysis_output_manifest_succeeds_for_real_long_producer_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a real `analyze_snapshot(...)` call over genuine SEC
    fixtures (`test_long_forecasts`'s own `_company_evidence`/`_fact`
    helpers, extended here so every registered SEC source asset also has
    real physical bytes on disk -- the synthetic fixtures upstream only
    stamp a `sha256`, never write a file, which is fine for
    `build_long_forecasts`'s own arithmetic but would fail this
    verifier's physical checksum-read) produces genuine long advisory
    `Prediction` rows; `verify_analysis_output_manifest`'s `long=` branch
    must accept that real output as-is."""
    config = _small_peer_config()
    monkeypatch.setattr(
        "stanstock.research.service.load_long_forecast_config", lambda _path=None: config
    )
    monkeypatch.setattr(
        "stanstock.research.service.long_forecast_config_hash", lambda _config: "l" * 64
    )
    store = AssetStore(tmp_path)

    def _real_backed_asset(
        *,
        provider: str,
        kind: str,
        subject: str,
        retrieved_at: datetime,
        metadata: dict[str, object] | None = None,
    ) -> DataAsset:
        payload = f"{provider}:{kind}:{subject}:{retrieved_at.isoformat()}".encode()
        stored = store.write_bytes(f"long-e2e/{uuid4().hex}.bin", payload)
        return register_asset(
            provider=provider,
            kind=kind,
            subject=subject[:120],
            stored=stored,
            retrieved_at=retrieved_at,
            available_at=retrieved_at,
            metadata=metadata or {},
        )

    monkeypatch.setattr("test_long_forecasts._asset", _real_backed_asset)

    ProviderRecord.objects.create(provider="sec", enabled=True, status="ok")
    snapshot = _snapshot(slug=f"manifest-long-{uuid4().hex[:8]}")
    listings = [_long_listing(f"LNG{index}") for index in range(2)]
    for index, listing in enumerate(listings):
        UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
        _write_price_asset(store, listing, close=50.0 + index * 5)
        _company_evidence(
            listing,
            family="fcf_per_share",
            sic="3571",
            scale=1.0 + index * 0.1,
            create_price=False,
        )

    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        issued_on_time=True,
        provider="twelve_data",
        store=store,
        config_path=default_us_scoring_config_path(),
    )
    run = results[0].run
    listings_by_id = {result.analysis.listing_id: result.analysis.listing for result in results}
    assert (
        Prediction.objects.filter(
            evidence_role=Prediction.EvidenceRole.ADVISORY, horizon__in=["3y", "5y"]
        ).count()
        == len(listings) * 2
    )

    long_expectation = LongLaneExpectation(config_hash="l" * 64, method_version=config.version)
    outcome = _verify(run, listings_by_id, long=long_expectation)

    assert outcome.summary["analysis_run_id"] == str(run.id)


# ---------------------------------------------------------------------------
# Group 2b: isolated `_require_prediction_horizon_multiset` unit tests
# ---------------------------------------------------------------------------


def test_prediction_horizon_multiset_rejects_listing_mismatch() -> None:
    analysis = StockAnalysis(id=uuid4(), listing_id=uuid4())
    prediction = Prediction(
        id=uuid4(),
        analysis=analysis,
        listing_id=uuid4(),
        horizon="short",
        evidence_role=Prediction.EvidenceRole.DECISION,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_prediction_horizon_multiset(
            [analysis],
            [prediction],
            decision_horizons=DECISION_HORIZONS,
            medium=None,
            long=None,
            eligible_listing_ids={analysis.listing_id},
        )
    assert _reason(exc_info) == "prediction_listing_mismatch"


def test_prediction_horizon_multiset_rejects_ineligible_listing() -> None:
    analysis = StockAnalysis(id=uuid4(), listing_id=uuid4())
    prediction = Prediction(
        id=uuid4(),
        analysis=analysis,
        listing_id=analysis.listing_id,
        horizon="short",
        evidence_role=Prediction.EvidenceRole.DECISION,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_prediction_horizon_multiset(
            [analysis],
            [prediction],
            decision_horizons=DECISION_HORIZONS,
            medium=None,
            long=None,
            eligible_listing_ids=set(),
        )
    assert _reason(exc_info) == "prediction_listing_ineligible"


def test_prediction_horizon_multiset_rejects_duplicate_role_horizon() -> None:
    analysis = StockAnalysis(id=uuid4(), listing_id=uuid4())
    predictions = [
        Prediction(
            id=uuid4(),
            analysis=analysis,
            listing_id=analysis.listing_id,
            horizon="short",
            evidence_role=Prediction.EvidenceRole.DECISION,
        )
        for _ in range(2)
    ]
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_prediction_horizon_multiset(
            [analysis],
            predictions,
            decision_horizons=DECISION_HORIZONS,
            medium=None,
            long=None,
            eligible_listing_ids={analysis.listing_id},
        )
    assert _reason(exc_info) == "prediction_role_horizon_duplicated"


@pytest.mark.parametrize(
    ("advisory_horizon", "expected_reason"),
    [
        ("6m", "prediction_medium_multiset_mismatch"),
        ("3y", "prediction_long_multiset_mismatch"),
    ],
    ids=["medium", "long"],
)
def test_prediction_horizon_multiset_rejects_inactive_advisory_lane(
    advisory_horizon: str, expected_reason: str
) -> None:
    """An advisory prediction for a lane that was never planned as active
    (`medium`/`long` both `None`) must fail via the precise multiset
    mismatch here, before `_require_prediction_ground_truth`'s own
    per-prediction lane checks are ever reached."""
    analysis = StockAnalysis(id=uuid4(), listing_id=uuid4())
    predictions = [
        Prediction(
            id=uuid4(),
            analysis=analysis,
            listing_id=analysis.listing_id,
            horizon="short",
            evidence_role=Prediction.EvidenceRole.DECISION,
        ),
        Prediction(
            id=uuid4(),
            analysis=analysis,
            listing_id=analysis.listing_id,
            horizon=advisory_horizon,
            evidence_role=Prediction.EvidenceRole.ADVISORY,
        ),
    ]
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_prediction_horizon_multiset(
            [analysis],
            predictions,
            decision_horizons=DECISION_HORIZONS,
            medium=None,
            long=None,
            eligible_listing_ids={analysis.listing_id},
        )
    assert _reason(exc_info) == expected_reason


# ---------------------------------------------------------------------------
# `_require_prediction_ground_truth`: isolated single-field mutation
# coverage. It only reads plain Python attributes (no query of its own), so
# an in-memory, never-persisted `Prediction`/`StockAnalysis`/`AnalysisRun`/
# `UniverseSnapshot` chain reaches every branch directly, one field at a
# time, without the `Prediction` immutability trigger ever being involved.
# ---------------------------------------------------------------------------


def _ground_truth_listing() -> Listing:
    return _long_listing(f"GT{uuid4().hex[:6].upper()}")


def _ground_truth_prediction(
    listing: Listing,
    *,
    horizon: str = "short",
    evidence_role: str = Prediction.EvidenceRole.DECISION,
) -> Prediction:
    universe = Universe(slug="gt", name="gt", config_version="v1")
    snapshot = UniverseSnapshot(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="a" * 64,
    )
    run = AnalysisRun(universe_snapshot=snapshot, target_date=TARGET_DATE)
    analysis = StockAnalysis(id=uuid4(), listing=listing, run=run)
    return Prediction(
        id=uuid4(),
        analysis=analysis,
        listing=listing,
        target_date=TARGET_DATE,
        generated_at=DECISION_TIME,
        issued_on_time=True,
        evidence_grade=UniverseSnapshot.Grade.OBSERVED,
        code_revision="rev-1",
        source_mode="provider",
        source_assets=[{"provider": "twelve_data"}],
        price_provider=TWELVE_DATA_PROVIDER,
        price_subject=listing.provider_symbol,
        horizon=horizon,
        evidence_role=evidence_role,
        config_hash="cfg-hash",
        method_version="cfg-version",
    )


def _ground_truth_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "target_date": TARGET_DATE,
        "code_revision": "rev-1",
        "scoring_config_version": "cfg-version",
        "scoring_config_hash": "cfg-hash",
        "decision_horizons": DECISION_HORIZONS,
        "medium": None,
        "long": None,
    }
    kwargs.update(overrides)
    return kwargs


def test_prediction_ground_truth_succeeds_for_matching_decision_prediction() -> None:
    listing = _ground_truth_listing()
    prediction = _ground_truth_prediction(listing)
    _require_prediction_ground_truth(prediction, listing=listing, **_ground_truth_kwargs())


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (lambda p: setattr(p, "target_date", date(2026, 2, 28)), "prediction_target_mismatch"),
        (lambda p: setattr(p, "issued_on_time", False), "prediction_not_on_time"),
        (
            lambda p: setattr(p, "evidence_grade", UniverseSnapshot.Grade.RESEARCH),
            "prediction_evidence_grade_mismatch",
        ),
        (
            lambda p: setattr(p, "code_revision", "wrong-rev"),
            "prediction_code_revision_mismatch",
        ),
        (lambda p: setattr(p, "source_mode", "synthetic"), "prediction_source_mode_invalid"),
        (
            lambda p: setattr(p, "price_provider", "other_provider"),
            "prediction_price_provider_mismatch",
        ),
        (
            lambda p: setattr(p, "price_subject", "WRONGTICKER"),
            "prediction_price_subject_mismatch",
        ),
    ],
    ids=[
        "target_date",
        "issuance_timing",
        "evidence_grade",
        "code_revision",
        "source_mode",
        "price_provider",
        "price_subject",
    ],
)
def test_prediction_ground_truth_rejects_single_field_mutation(mutate, expected_reason) -> None:
    listing = _ground_truth_listing()
    prediction = _ground_truth_prediction(listing)
    mutate(prediction)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_prediction_ground_truth(prediction, listing=listing, **_ground_truth_kwargs())
    assert _reason(exc_info) == expected_reason


def _rgt_case_unsupported_decision_horizon() -> tuple[Prediction, Listing, dict[str, object]]:
    listing = _ground_truth_listing()
    prediction = _ground_truth_prediction(
        listing, horizon="6m", evidence_role=Prediction.EvidenceRole.DECISION
    )
    return prediction, listing, _ground_truth_kwargs()


def _rgt_case_decision_config_mismatch() -> tuple[Prediction, Listing, dict[str, object]]:
    listing = _ground_truth_listing()
    prediction = _ground_truth_prediction(listing)
    prediction.config_hash = "wrong-hash"
    return prediction, listing, _ground_truth_kwargs()


def _rgt_case_medium_config_mismatch() -> tuple[Prediction, Listing, dict[str, object]]:
    listing = _ground_truth_listing()
    medium = _medium_lane_expectation()
    prediction = _ground_truth_prediction(
        listing, horizon="6m", evidence_role=Prediction.EvidenceRole.ADVISORY
    )
    prediction.config_hash = "wrong-medium-hash"
    prediction.method_version = medium.method_version
    return prediction, listing, _ground_truth_kwargs(medium=medium)


def _rgt_case_medium_method_mismatch() -> tuple[Prediction, Listing, dict[str, object]]:
    listing = _ground_truth_listing()
    medium = _medium_lane_expectation()
    prediction = _ground_truth_prediction(
        listing, horizon="6m", evidence_role=Prediction.EvidenceRole.ADVISORY
    )
    prediction.config_hash = medium.config_hash
    prediction.method_version = "wrong-medium-version"
    return prediction, listing, _ground_truth_kwargs(medium=medium)


_GROUND_TRUTH_LONG_EXPECTATION = LongLaneExpectation(
    config_hash="long-cfg-hash", method_version="long-v1"
)


def _rgt_case_long_config_mismatch() -> tuple[Prediction, Listing, dict[str, object]]:
    listing = _ground_truth_listing()
    prediction = _ground_truth_prediction(
        listing, horizon="3y", evidence_role=Prediction.EvidenceRole.ADVISORY
    )
    prediction.config_hash = "wrong-long-hash"
    prediction.method_version = _GROUND_TRUTH_LONG_EXPECTATION.method_version
    return prediction, listing, _ground_truth_kwargs(long=_GROUND_TRUTH_LONG_EXPECTATION)


def _rgt_case_long_method_mismatch() -> tuple[Prediction, Listing, dict[str, object]]:
    listing = _ground_truth_listing()
    prediction = _ground_truth_prediction(
        listing, horizon="3y", evidence_role=Prediction.EvidenceRole.ADVISORY
    )
    prediction.config_hash = _GROUND_TRUTH_LONG_EXPECTATION.config_hash
    prediction.method_version = "wrong-long-version"
    return prediction, listing, _ground_truth_kwargs(long=_GROUND_TRUTH_LONG_EXPECTATION)


def _rgt_case_unsupported_advisory_horizon() -> tuple[Prediction, Listing, dict[str, object]]:
    listing = _ground_truth_listing()
    prediction = _ground_truth_prediction(
        listing, horizon="short", evidence_role=Prediction.EvidenceRole.ADVISORY
    )
    return prediction, listing, _ground_truth_kwargs()


def _rgt_case_invalid_evidence_role() -> tuple[Prediction, Listing, dict[str, object]]:
    listing = _ground_truth_listing()
    prediction = _ground_truth_prediction(listing)
    prediction.evidence_role = "bogus-role"
    return prediction, listing, _ground_truth_kwargs()


@pytest.mark.parametrize(
    ("build_case", "expected_reason"),
    [
        (_rgt_case_unsupported_decision_horizon, "prediction_horizon_role_mismatch"),
        (_rgt_case_decision_config_mismatch, "prediction_decision_config_mismatch"),
        (_rgt_case_medium_config_mismatch, "prediction_advisory_config_mismatch"),
        (_rgt_case_medium_method_mismatch, "prediction_advisory_method_version_mismatch"),
        (_rgt_case_long_config_mismatch, "prediction_advisory_config_mismatch"),
        (_rgt_case_long_method_mismatch, "prediction_advisory_method_version_mismatch"),
        (_rgt_case_unsupported_advisory_horizon, "prediction_horizon_role_mismatch"),
        (_rgt_case_invalid_evidence_role, "prediction_evidence_role_invalid"),
    ],
    ids=[
        "unsupported_decision_horizon",
        "decision_config_mismatch",
        "medium_config_mismatch",
        "medium_method_mismatch",
        "long_config_mismatch",
        "long_method_mismatch",
        "unsupported_advisory_horizon",
        "invalid_evidence_role",
    ],
)
def test_prediction_ground_truth_rejects_role_horizon_lane_mismatch(
    build_case, expected_reason: str
) -> None:
    prediction, listing, kwargs = build_case()
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_prediction_ground_truth(prediction, listing=listing, **kwargs)
    assert _reason(exc_info) == expected_reason


# ---------------------------------------------------------------------------
# Group 3: isolated deep-check unit tests
# ---------------------------------------------------------------------------


def _scenario(*, bear=-0.1, base=0.05, bull=0.2) -> dict:
    return {
        "bear": bear,
        "base": base,
        "bull": bull,
        "probability_positive": 0.6,
        "confidence": 50.0,
        "confidence_status": "heuristic",
        "insufficiency_reason": "",
    }


def _scenario_document(scenario: dict) -> dict:
    """`scenario_from_document` only consults the canonical
    `horizons` mapping when `schema_version` matches; without it, every
    lookup silently falls through to the legacy mirror argument and the
    mirror-consistency check can never observe a genuine disagreement."""
    return {
        "schema_version": FORECAST_SCENARIO_SCHEMA_VERSION,
        "horizons": {"short": scenario, "medium": scenario, "long": scenario},
    }


def test_scenario_mirror_consistency_passes_when_mirrors_agree() -> None:
    scenario = _scenario()
    analysis = StockAnalysis(
        forecast_scenarios=_scenario_document(scenario),
        short_scenario=scenario,
        medium_scenario=scenario,
        long_scenario=scenario,
    )
    _require_scenario_mirror_consistency(analysis)  # must not raise


def test_scenario_mirror_consistency_detects_disagreement() -> None:
    scenario = _scenario()
    mismatched_mirror = _scenario(base=0.99)
    analysis = StockAnalysis(
        forecast_scenarios=_scenario_document(scenario),
        short_scenario=mismatched_mirror,
        medium_scenario=scenario,
        long_scenario=scenario,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_scenario_mirror_consistency(analysis)
    assert _reason(exc_info) == "stock_analysis_scenario_mirror_mismatch"


def test_scenario_mirror_consistency_fails_closed_on_missing_canonical_horizon() -> None:
    """A canonical document missing one of the three required horizon keys
    must fail outright, never silently resolve that horizon from its
    legacy mirror field (the fail-open gap this check replaces)."""
    scenario = _scenario()
    document = _scenario_document(scenario)
    del document["horizons"]["long"]
    analysis = StockAnalysis(
        forecast_scenarios=document,
        short_scenario=scenario,
        medium_scenario=scenario,
        long_scenario=scenario,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_scenario_mirror_consistency(analysis)
    assert _reason(exc_info) == "stock_analysis_scenario_missing"


def test_scenario_mirror_consistency_fails_closed_on_missing_schema_version() -> None:
    scenario = _scenario()
    document = {"horizons": {"short": scenario, "medium": scenario, "long": scenario}}
    analysis = StockAnalysis(
        forecast_scenarios=document,
        short_scenario=scenario,
        medium_scenario=scenario,
        long_scenario=scenario,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_scenario_mirror_consistency(analysis)
    assert _reason(exc_info) == "stock_analysis_scenario_document_invalid"


def _full_scenario_document(*, medium_active: bool, long_active: bool) -> dict:
    scenario = _scenario()
    horizons = {"short": scenario, "medium": scenario, "long": scenario}
    if medium_active:
        horizons["6m"] = scenario
        horizons["12m"] = scenario
    if long_active:
        horizons["3y"] = scenario
        horizons["5y"] = scenario
    return {"schema_version": FORECAST_SCENARIO_SCHEMA_VERSION, "horizons": horizons}


def test_scenario_document_shape_succeeds_for_decision_only_run() -> None:
    analysis = StockAnalysis(
        forecast_scenarios=_full_scenario_document(medium_active=False, long_active=False)
    )
    _require_scenario_document_shape(analysis, medium_active=False, long_active=False)


def test_scenario_document_shape_succeeds_with_active_advisory_lanes() -> None:
    analysis = StockAnalysis(
        forecast_scenarios=_full_scenario_document(medium_active=True, long_active=True)
    )
    _require_scenario_document_shape(analysis, medium_active=True, long_active=True)


def test_scenario_document_shape_rejects_inactive_lane_horizon_key() -> None:
    """A `6m`/`12m` entry present despite the medium lane being inactive for
    this run must fail even though every field inside it is well-formed."""
    analysis = StockAnalysis(
        forecast_scenarios=_full_scenario_document(medium_active=True, long_active=False)
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_scenario_document_shape(analysis, medium_active=False, long_active=False)
    assert _reason(exc_info) == "stock_analysis_scenario_horizon_set_mismatch"


def test_scenario_document_shape_rejects_unknown_horizon_key() -> None:
    document = _full_scenario_document(medium_active=False, long_active=False)
    document["horizons"]["9m"] = _scenario()
    analysis = StockAnalysis(forecast_scenarios=document)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_scenario_document_shape(analysis, medium_active=False, long_active=False)
    assert _reason(exc_info) == "stock_analysis_scenario_horizon_set_mismatch"


def test_scenario_document_shape_rejects_missing_active_lane_horizon() -> None:
    document = _full_scenario_document(medium_active=True, long_active=False)
    del document["horizons"]["12m"]
    analysis = StockAnalysis(forecast_scenarios=document)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_scenario_document_shape(analysis, medium_active=True, long_active=False)
    assert _reason(exc_info) == "stock_analysis_scenario_horizon_set_mismatch"


@pytest.mark.parametrize(
    "field_name",
    [
        "bear",
        "base",
        "bull",
        "probability_positive",
        "confidence",
        "confidence_status",
        "insufficiency_reason",
    ],
)
def test_canonical_scenario_shape_rejects_missing_field(field_name: str) -> None:
    scenario = _scenario()
    del scenario[field_name]
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_canonical_scenario_shape(scenario, horizon="short")
    assert _reason(exc_info) == "stock_analysis_scenario_shape_invalid"


def test_canonical_scenario_shape_rejects_non_object() -> None:
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_canonical_scenario_shape(["not", "an", "object"], horizon="short")
    assert _reason(exc_info) == "stock_analysis_scenario_shape_invalid"


def test_canonical_scenario_shape_rejects_non_numeric_return_field() -> None:
    scenario = _scenario()
    scenario["bear"] = "not-a-number"
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_canonical_scenario_shape(scenario, horizon="short")
    assert _reason(exc_info) == "stock_analysis_scenario_shape_invalid"


def test_canonical_scenario_shape_rejects_boolean_return_field() -> None:
    """`bool` is a subclass of `int` in Python; it must not silently pass as
    a numeric return value."""
    scenario = _scenario()
    scenario["bear"] = True
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_canonical_scenario_shape(scenario, horizon="short")
    assert _reason(exc_info) == "stock_analysis_scenario_shape_invalid"


def test_canonical_scenario_shape_rejects_partial_null_return_triplet() -> None:
    scenario = _scenario()
    scenario["bear"] = None
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_canonical_scenario_shape(scenario, horizon="short")
    assert _reason(exc_info) == "stock_analysis_scenario_partial_null"


def test_canonical_scenario_shape_rejects_probability_without_return_distribution() -> None:
    scenario = _scenario()
    scenario["bear"] = None
    scenario["base"] = None
    scenario["bull"] = None
    # probability_positive left non-null: a probability with no underlying
    # return distribution at all.
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_canonical_scenario_shape(scenario, horizon="short")
    assert _reason(exc_info) == "stock_analysis_scenario_partial_null"


def test_canonical_scenario_shape_accepts_withheld_probability_with_valid_returns() -> None:
    """A legitimate production pattern: enough samples for a point-estimate
    return distribution but not enough for a reliable win-rate probability."""
    scenario = _scenario()
    scenario["probability_positive"] = None
    _require_canonical_scenario_shape(scenario, horizon="short")  # must not raise


def test_canonical_scenario_shape_accepts_fully_withheld_scenario() -> None:
    scenario = _scenario()
    scenario["bear"] = None
    scenario["base"] = None
    scenario["bull"] = None
    scenario["probability_positive"] = None
    _require_canonical_scenario_shape(scenario, horizon="short")  # must not raise


def test_canonical_scenario_shape_rejects_non_numeric_confidence() -> None:
    scenario = _scenario()
    scenario["confidence"] = "high"
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_canonical_scenario_shape(scenario, horizon="short")
    assert _reason(exc_info) == "stock_analysis_scenario_shape_invalid"


def test_canonical_scenario_shape_rejects_empty_confidence_status() -> None:
    scenario = _scenario()
    scenario["confidence_status"] = ""
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_canonical_scenario_shape(scenario, horizon="short")
    assert _reason(exc_info) == "stock_analysis_scenario_shape_invalid"


def test_canonical_scenario_shape_rejects_non_string_insufficiency_reason() -> None:
    scenario = _scenario()
    scenario["insufficiency_reason"] = None
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_canonical_scenario_shape(scenario, horizon="short")
    assert _reason(exc_info) == "stock_analysis_scenario_shape_invalid"


def _price_bound_listing() -> Listing:
    return _long_listing(f"PB{uuid4().hex[:6].upper()}")


def test_stock_analysis_price_source_bound_succeeds_for_matching_reference() -> None:
    listing = _price_bound_listing()
    asset = _price_source_asset(subject=listing.ticker, retrieved_at=DECISION_TIME)
    analysis = StockAnalysis(
        id=uuid4(),
        listing=listing,
        data_quality={
            "price_source": {
                "asset_id": str(asset.id),
                "provider": asset.provider,
                "subject": asset.subject,
            }
        },
    )
    raw = [_source_entry(asset)]
    refs = [
        AssetRef(
            id=asset.id,
            provider=asset.provider,
            kind=asset.kind,
            subject=asset.subject,
            sha256=asset.sha256,
        )
    ]
    _require_stock_analysis_price_source_bound(analysis, raw_source_assets=raw, refs=refs)


def test_stock_analysis_price_source_bound_rejects_missing_price_source() -> None:
    """Production always sets `data_quality["price_source"]` for any
    provider-backed observed run once a price asset is resolved; a missing
    or `None` value must fail closed rather than being treated as
    optional."""
    listing = _price_bound_listing()
    asset = _price_source_asset(subject=listing.ticker, retrieved_at=DECISION_TIME)
    for data_quality in ({}, {"price_source": None}):
        analysis = StockAnalysis(id=uuid4(), listing=listing, data_quality=data_quality)
        raw = [_source_entry(asset)]
        refs = [
            AssetRef(
                id=asset.id,
                provider=asset.provider,
                kind=asset.kind,
                subject=asset.subject,
                sha256=asset.sha256,
            )
        ]
        with pytest.raises(RefreshVerificationError) as exc_info:
            _require_stock_analysis_price_source_bound(analysis, raw_source_assets=raw, refs=refs)
        assert _reason(exc_info) == "stock_analysis_price_source_mismatch"


def test_stock_analysis_price_source_bound_rejects_extra_key() -> None:
    listing = _price_bound_listing()
    asset = _price_source_asset(subject=listing.ticker, retrieved_at=DECISION_TIME)
    analysis = StockAnalysis(
        id=uuid4(),
        listing=listing,
        data_quality={
            "price_source": {
                "asset_id": str(asset.id),
                "provider": asset.provider,
                "subject": asset.subject,
                "extra": "unexpected",
            }
        },
    )
    raw = [_source_entry(asset)]
    refs = [
        AssetRef(
            id=asset.id,
            provider=asset.provider,
            kind=asset.kind,
            subject=asset.subject,
            sha256=asset.sha256,
        )
    ]
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_stock_analysis_price_source_bound(analysis, raw_source_assets=raw, refs=refs)
    assert _reason(exc_info) == "stock_analysis_price_source_mismatch"


def test_stock_analysis_price_source_bound_rejects_no_price_history_asset() -> None:
    listing = _price_bound_listing()
    sec_asset = DataAsset.objects.create(
        provider="sec",
        kind="sec_submissions",
        subject=listing.ticker,
        relative_path=f"tests/manifest/{uuid4().hex}",
        sha256=uuid4().hex * 2,
        retrieved_at=DECISION_TIME,
        available_at=DECISION_TIME,
    )
    analysis = StockAnalysis(id=uuid4(), listing=listing, data_quality={})
    raw = [_source_entry(sec_asset)]
    refs = [
        AssetRef(
            id=sec_asset.id,
            provider=sec_asset.provider,
            kind=sec_asset.kind,
            subject=sec_asset.subject,
            sha256=sec_asset.sha256,
        )
    ]
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_stock_analysis_price_source_bound(analysis, raw_source_assets=raw, refs=refs)
    assert _reason(exc_info) == "stock_analysis_price_source_unbound"


def test_stock_analysis_price_source_bound_rejects_recorded_mismatch() -> None:
    """A `data_quality["price_source"]` recorded dict that disagrees with
    the analysis's own resolved price source reference must fail even
    though the resolved reference itself is unambiguous."""
    listing = _price_bound_listing()
    asset = _price_source_asset(subject=listing.ticker, retrieved_at=DECISION_TIME)
    analysis = StockAnalysis(
        id=uuid4(),
        listing=listing,
        data_quality={
            "price_source": {
                "asset_id": str(uuid4()),  # transplanted to an unrelated asset id
                "provider": asset.provider,
                "subject": asset.subject,
            }
        },
    )
    raw = [_source_entry(asset)]
    refs = [
        AssetRef(
            id=asset.id,
            provider=asset.provider,
            kind=asset.kind,
            subject=asset.subject,
            sha256=asset.sha256,
        )
    ]
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_stock_analysis_price_source_bound(analysis, raw_source_assets=raw, refs=refs)
    assert _reason(exc_info) == "stock_analysis_price_source_mismatch"


def _medium_lane_expectation(
    *, provider: str = "twelve_data", benchmark_subject: str = "SPY"
) -> MediumLaneExpectation:
    """The reviewed medium-forecast configuration for
    `config/forecasts/us-price-medium-v1.yml`, read the same way
    `analyze_snapshot` itself resolves and hashes it -- never a
    hand-typed literal that could silently drift from the real config."""
    medium_config = load_medium_forecast_config()
    return MediumLaneExpectation(
        config_hash=medium_forecast_config_hash(medium_config),
        method_version=medium_config.version,
        calendar=medium_config.calendar,
        fixed_epoch=medium_config.fixed_epoch,
        provider=provider,
        benchmark_subject=benchmark_subject,
        return_basis=medium_config.return_basis,
        dividends_included=medium_config.dividends_included,
        config=medium_config,
    )


def _price_source_asset(*, subject: str, retrieved_at: datetime) -> DataAsset:
    raw_asset = DataAsset.objects.create(
        provider="twelve_data",
        kind="raw_price_history",
        subject=subject,
        relative_path=f"tests/manifest/{uuid4().hex}",
        sha256=uuid4().hex * 2,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
    )
    return DataAsset.objects.create(
        provider="twelve_data",
        kind="price_history",
        subject=subject,
        relative_path=f"tests/manifest/{uuid4().hex}",
        sha256=uuid4().hex * 2,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
        metadata={
            "raw_asset_id": str(raw_asset.id),
            "raw_sha256": raw_asset.sha256,
        },
    )


def _panel_test_row(
    *, horizon: str, listing: Listing, price_asset: DataAsset, target_date
) -> dict[str, object]:
    """One schema-conforming, deliberately arbitrary forecast-anchor row.

    Structural failure tests use it only when their named check runs before
    semantic replay; positive/replay tests use the exact shared constructor.
    """
    return {
        "horizon": horizon,
        "anchor_date": target_date,
        "label_end_date": None,
        "is_forecast": True,
        "cohort_id": f"{horizon}:{target_date.isoformat()}",
        "listing_id": str(listing.pk),
        "ticker": listing.ticker,
        "price_asset_id": str(price_asset.pk),
        "relative_momentum": 0.01,
        "drawdown": -0.02,
        "volatility": 0.15,
        "market_trend": 0.01,
        "market_volatility": 0.12,
        "relative_momentum_bucket": 0,
        "drawdown_bucket": 0,
        "volatility_bucket": 0,
        "market_trend_bucket": 0,
        "market_volatility_bucket": 0,
        "close_vs_sma_50": 0.02,
        "close_vs_sma_200": 0.03,
        "downside_volatility": 0.05,
        "average_dollar_volume": 1_000_000.0,
        "forward_return": None,
        "benchmark_forward_return": None,
        "relative_forward_return": None,
        "eligible": True,
        "insufficiency_reason": "",
    }


def _panel_frame_bytes(rows: list[dict[str, object]]) -> bytes:
    frame = pl.DataFrame(rows, schema=PANEL_SCHEMA, orient="row")
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def _reconstruct_test_medium_frame(
    store: AssetStore,
    *,
    run: AnalysisRun,
    medium: MediumLaneExpectation,
    listings: list[Listing],
    price_assets: dict[uuid.UUID, DataAsset],
    benchmark_asset: DataAsset,
) -> pl.DataFrame:
    asof = AsOfData(run.data_cutoff, store)
    benchmark_frame = asof.price_frame_for_asset_with_diagnostics(
        asset=benchmark_asset,
        through_date=run.target_date,
    ).frame
    return reconstruct_medium_forecast_panel(
        benchmark_asset=benchmark_asset,
        benchmark_frame=benchmark_frame,
        listing_inputs=[
            MediumPanelPriceInput(
                listing=listing,
                asset=price_assets[listing.id],
                frame=asof.price_frame_for_asset_with_diagnostics(
                    asset=price_assets[listing.id],
                    through_date=run.target_date,
                ).frame,
            )
            for listing in listings
        ],
        target_date=run.target_date,
        config=medium.config,
    )


def _build_test_medium_panel(
    store: AssetStore,
    *,
    run: AnalysisRun,
    medium: MediumLaneExpectation,
    scoring_config_version: str,
    listings: list[Listing],
    price_assets: dict[uuid.UUID, DataAsset],
    benchmark_asset: DataAsset,
    payload: bytes | None = None,
    row_overrides: list[dict[str, object]] | None = None,
    metadata_overrides: dict | None = None,
    omit_metadata_fields: set[str] | None = None,
    retrieved_at: datetime | None = None,
) -> DataAsset:
    """A hand-built (never through the real `build_medium_forecast_panel`
    producer) medium forecast panel `DataAsset`, whose baseline content and
    metadata are otherwise exactly what the validator under test requires
    -- so each test can corrupt exactly one aspect and prove the validator
    (not merely the producer's own self-consistency) catches it."""
    if payload is None:
        rows = row_overrides
        if rows is None:
            frame = _reconstruct_test_medium_frame(
                store,
                run=run,
                medium=medium,
                listings=listings,
                price_assets=price_assets,
                benchmark_asset=benchmark_asset,
            )
            payload = serialize_medium_forecast_panel(frame)
        else:
            payload = _panel_frame_bytes(rows)
    stored = store.write_bytes(f"medium-tests/{uuid4().hex}.parquet", payload)
    sessions = calendar_sessions_through(
        calendar_name=medium.calendar, fixed_epoch=medium.fixed_epoch, target_date=run.target_date
    )
    calendar_hash = hash_json([session.isoformat() for session in sessions])
    closure = [benchmark_asset, *(price_assets[listing.id] for listing in listings)]
    source_manifest = [asset_identity(asset) for asset in dedupe_assets(closure)]
    source_manifest_hash = hash_json(source_manifest)
    universe_snapshot = run.universe_snapshot
    row_count = None
    try:
        row_count = pl.read_parquet(io.BytesIO(payload)).height
    except pl.exceptions.ComputeError:
        row_count = (
            len(row_overrides)
            if row_overrides is not None
            else len(listings) * len(_MEDIUM_HORIZONS)
        )
    metadata: dict[str, object] = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "method_version": medium.method_version,
        "config_hash": medium.config_hash,
        "code_revision": run.code_revision,
        "calendar": medium.calendar,
        "calendar_library_version": "test-calendar-lib",
        "panel_library_version": "test-panel-lib",
        "fixed_epoch": medium.fixed_epoch.isoformat(),
        "calendar_hash": calendar_hash,
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
        "row_count": row_count,
        "content_sha256": stored.sha256,
        "source_manifest_hash": source_manifest_hash,
        "source_assets": source_manifest,
    }
    metadata["evidence_bundle_hash"] = hash_json(
        {
            "calendar_hash": metadata["calendar_hash"],
            "code_revision": metadata["code_revision"],
            "content_sha256": metadata["content_sha256"],
            "forecast_config_hash": metadata["config_hash"],
            "scoring_config_hash": metadata["scoring_config_hash"],
            "source_manifest_hash": metadata["source_manifest_hash"],
            "universe_config_hash": metadata["universe_config_hash"],
        }
    )
    if metadata_overrides:
        metadata.update(metadata_overrides)
    for field_name in omit_metadata_fields or ():
        metadata.pop(field_name, None)
    effective_retrieved_at = retrieved_at or run.generated_at
    return register_asset(
        provider=PANEL_PROVIDER,
        kind=PANEL_KIND,
        subject=str(run.id),
        stored=stored,
        retrieved_at=effective_retrieved_at,
        available_at=effective_retrieved_at,
        metadata=metadata,
    )


def _medium_lane_fixture(
    tmp_path: Path, *, listing_count: int = 2
) -> tuple[AssetStore, AnalysisRun, list[Listing], dict[uuid.UUID, DataAsset], DataAsset]:
    """A bare observed `AnalysisRun` plus eligible listings, each with a
    real registered `twelve_data` `price_history` asset, and a benchmark
    `price_history` asset -- everything `_build_test_medium_panel` needs,
    without ever calling the real panel-building producer (so a test can
    hand-craft the exact panel content/metadata it wants to prove the
    validator rejects)."""
    store = AssetStore(tmp_path)
    run = _bare_run()
    listings = [_long_listing(f"MED{index}") for index in range(listing_count)]
    price_assets = {
        listing.id: _write_subject_price_asset(
            store,
            listing.ticker,
            close=40.0 + index * 5,
        )
        for index, listing in enumerate(listings)
    }
    benchmark_asset = _write_subject_price_asset(store, "SPY", close=300.0)
    return store, run, listings, price_assets, benchmark_asset


def _bare_medium_prediction_with_registry(
    *, run: AnalysisRun, panel: DataAsset, extra_source_assets: list[DataAsset] | None = None
) -> tuple[Prediction, _AssetRegistry]:
    """A prediction whose own `source_assets` already declares the panel (and
    any extra sources), pre-bound into a fresh registry exactly the way
    `verify_analysis_output_manifest`'s own per-prediction
    `_require_source_assets_bound` call would before ever calling
    `_require_medium_panel_binding`."""
    all_sources = [*(extra_source_assets or []), panel]
    entries = [_source_entry(asset) for asset in all_sources]
    prediction = Prediction(
        id=uuid4(),
        target_date=run.target_date,
        data_cutoff=run.data_cutoff,
        calculation={"panel_asset_id": str(panel.id), "panel_sha256": panel.sha256},
        source_assets=entries,
    )
    registry = _AssetRegistry()
    _require_source_assets_bound(
        entries, context="Prediction test", cutoff=run.data_cutoff, registry=registry
    )
    return prediction, registry


def _source_entry(asset: DataAsset) -> dict:
    return {
        "id": str(asset.id),
        "provider": asset.provider,
        "kind": asset.kind,
        "subject": asset.subject,
        "sha256": asset.sha256,
        "relative_path": asset.relative_path,  # deliberately present but must never be echoed
    }


def _bind_and_verify_medium_panel(
    *,
    run: AnalysisRun,
    medium: MediumLaneExpectation,
    listings: list[Listing],
    panel: DataAsset,
) -> UUID:
    prediction, registry = _bare_medium_prediction_with_registry(run=run, panel=panel)
    listings_by_id = {listing.id: listing for listing in listings}
    return _require_medium_panel_binding(
        prediction,
        medium=medium,
        run=run,
        scoring_config_version=run.config_version,
        eligible_listings=listings_by_id,
        registry=registry,
        panel_cache={},
    )


def test_medium_panel_binding_succeeds_for_matching_panel(tmp_path: Path) -> None:
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
    )
    panel_id = _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert panel_id == panel.id


def _semantically_tampered_medium_rows(
    frame: pl.DataFrame,
    *,
    mutation: str,
    target_date: date,
) -> list[dict[str, object]]:
    rows = frame.to_dicts()
    forecast = next(row for row in rows if bool(row["is_forecast"]))
    historical = next(
        row for row in rows if not bool(row["is_forecast"]) and row["forward_return"] is not None
    )
    if mutation == "forecast_after_target":
        forecast["anchor_date"] = target_date + timedelta(days=3)
        forecast["cohort_id"] = f"{forecast['horizon']}:{forecast['anchor_date'].isoformat()}"
    elif mutation == "historical_label_after_cutoff":
        historical["label_end_date"] = target_date + timedelta(days=3)
    elif mutation == "wrong_fixed_epoch_cohort":
        historical["cohort_id"] = f"{historical['horizon']}:wrong-fixed-epoch"
    elif mutation == "fabricated_forward_return":
        historical["forward_return"] = float(historical["forward_return"]) + 0.125
    elif mutation == "post_anchor_feature":
        forecast["relative_momentum"] = float(forecast["relative_momentum"]) + 0.25
    elif mutation == "wrong_eligibility_relation":
        forecast["eligible"] = False
        forecast["insufficiency_reason"] = "Insufficient medium forecast inputs: fabricated"
    else:  # pragma: no cover - guards the test table
        raise AssertionError(f"Unhandled panel mutation {mutation}")
    return (
        pl.DataFrame(rows, schema=PANEL_SCHEMA, orient="row")
        .sort("horizon", "anchor_date", "listing_id")
        .to_dicts()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "forecast_after_target",
        "historical_label_after_cutoff",
        "wrong_fixed_epoch_cohort",
        "fabricated_forward_return",
        "post_anchor_feature",
        "wrong_eligibility_relation",
    ],
)
def test_medium_panel_replay_rejects_semantic_tamper(
    tmp_path: Path,
    mutation: str,
) -> None:
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(
        tmp_path,
        listing_count=2,
    )
    medium = _medium_lane_expectation()
    expected = _reconstruct_test_medium_frame(
        store,
        run=run,
        medium=medium,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
    )
    rows = _semantically_tampered_medium_rows(
        expected,
        mutation=mutation,
        target_date=run.target_date,
    )
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        row_overrides=rows,
    )

    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(
            run=run,
            medium=medium,
            listings=listings,
            panel=panel,
        )

    assert _reason(exc_info) == "medium_panel_semantic_replay_mismatch"


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "reordered", "duplicate", "field_divergent"],
)
def test_medium_panel_requires_exact_declared_source_asset_list(
    tmp_path: Path,
    mutation: str,
) -> None:
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(
        tmp_path,
        listing_count=2,
    )
    medium = _medium_lane_expectation()
    expected_frame = _reconstruct_test_medium_frame(
        store,
        run=run,
        medium=medium,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
    )
    payload = serialize_medium_forecast_panel(expected_frame)
    canonical_sources = [
        asset_identity(asset)
        for asset in dedupe_assets(
            [benchmark_asset, *(price_assets[listing.id] for listing in listings)]
        )
    ]
    declared_sources = [dict(entry) for entry in canonical_sources]
    listing_entry_index = next(
        index
        for index, entry in enumerate(declared_sources)
        if entry["subject"] == listings[0].ticker
    )
    if mutation == "missing":
        declared_sources.pop(listing_entry_index)
    elif mutation == "extra":
        extra = _write_subject_price_asset(store, "EXTRA", close=25.0)
        declared_sources.append(asset_identity(extra))
    elif mutation == "reordered":
        declared_sources.reverse()
    elif mutation == "duplicate":
        declared_sources.append(dict(declared_sources[listing_entry_index]))
    elif mutation == "field_divergent":
        declared_sources[listing_entry_index]["relative_path"] = "forged/source.parquet"
    else:  # pragma: no cover - guards the test table
        raise AssertionError(f"Unhandled source-list mutation {mutation}")

    source_manifest_hash = hash_json(declared_sources)
    calendar_sessions = calendar_sessions_through(
        calendar_name=medium.calendar,
        fixed_epoch=medium.fixed_epoch,
        target_date=run.target_date,
    )
    calendar_hash = hash_json([session.isoformat() for session in calendar_sessions])
    content_sha256 = hashlib.sha256(payload).hexdigest()
    evidence_bundle_hash = hash_json(
        {
            "calendar_hash": calendar_hash,
            "code_revision": run.code_revision,
            "content_sha256": content_sha256,
            "forecast_config_hash": medium.config_hash,
            "scoring_config_hash": run.config_hash,
            "source_manifest_hash": source_manifest_hash,
            "universe_config_hash": run.universe_snapshot.config_hash,
        }
    )
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        payload=payload,
        metadata_overrides={
            "source_assets": declared_sources,
            "source_manifest_hash": source_manifest_hash,
            "evidence_bundle_hash": evidence_bundle_hash,
        },
    )

    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(
            run=run,
            medium=medium,
            listings=listings,
            panel=panel,
        )

    assert _reason(exc_info) == "medium_panel_source_assets_mismatch"


def test_medium_panel_binding_rejects_ambiguous_authoritative_panel(tmp_path: Path) -> None:
    """A duplicate same-content panel registered under a second, alternate
    id for the exact same run subject must be rejected outright -- never
    silently accepted as interchangeable with whichever one a prediction
    happens to reference."""
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
    )
    _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_ambiguous"


def test_medium_panel_binding_rejects_alternate_id_substitution(tmp_path: Path) -> None:
    """A prediction that names a real, resolvable, but *different* panel
    asset id than this run's own single authoritative panel must fail,
    even though the substituted panel is itself perfectly well-formed."""
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    authoritative = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
    )
    other_run = _bare_run(target_date=run.target_date)
    alternate = _build_test_medium_panel(
        store,
        run=other_run,
        medium=medium,
        scoring_config_version=other_run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
    )
    prediction, registry = _bare_medium_prediction_with_registry(run=run, panel=alternate)
    listings_by_id = {listing.id: listing for listing in listings}
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_medium_panel_binding(
            prediction,
            medium=medium,
            run=run,
            scoring_config_version=run.config_version,
            eligible_listings=listings_by_id,
            registry=registry,
            panel_cache={},
        )
    assert _reason(exc_info) == "medium_panel_not_authoritative"
    assert authoritative.id != alternate.id


def test_medium_panel_binding_rejects_non_parquet_content(tmp_path: Path) -> None:
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        payload=b"not parquet bytes at all",
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_content_unparseable"


@pytest.mark.parametrize(
    "override",
    [
        {"config_hash": "wrong" * 16},
        {"method_version": "wrong-version"},
        {"calendar": "WRONG"},
        {"fixed_epoch": "2011-01-03"},
        {"scoring_config_hash": "z" * 64},
        {"code_revision": "wrong-revision"},
        {"universe_snapshot_id": str(uuid.uuid4())},
        {"universe_slug": "wrong-universe-slug"},
        {"universe_config_hash": "q" * 64},
        {"scoring_config_version": "wrong-scoring-version"},
        {"return_definition": "wrong-basis"},
        {"dividends_included": True},
        {"content_sha256": "d" * 64},
        {"calendar_hash": "tampered-calendar-hash"},
    ],
)
def test_medium_panel_binding_rejects_config_mismatch(tmp_path: Path, override: dict) -> None:
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        metadata_overrides=override,
    )
    reason = (
        "medium_panel_content_checksum_mismatch"
        if "content_sha256" in override
        else "medium_panel_config_mismatch"
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == reason


@pytest.mark.parametrize(
    "missing_field",
    ["schema_version", "calendar_library_version", "panel_library_version", "row_count"],
)
def test_medium_panel_binding_rejects_incomplete_metadata(
    tmp_path: Path, missing_field: str
) -> None:
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        omit_metadata_fields={missing_field},
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_config_mismatch"


def test_medium_panel_binding_rejects_extra_metadata_field(tmp_path: Path) -> None:
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        metadata_overrides={"unexpected_extra_field": "surprise"},
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_config_mismatch"


def test_medium_panel_binding_rejects_wrong_row_count(tmp_path: Path) -> None:
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        metadata_overrides={"row_count": 999},
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_row_count_mismatch"


def test_medium_panel_binding_rejects_duplicate_rows(tmp_path: Path) -> None:
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    rows = [
        _panel_test_row(
            horizon=horizon,
            listing=listing,
            price_asset=price_assets[listing.id],
            target_date=run.target_date,
        )
        for listing in listings
        for horizon in sorted(_MEDIUM_HORIZONS)
    ]
    rows.append(dict(rows[0]))
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        row_overrides=rows,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_duplicate_rows"


def test_medium_panel_binding_rejects_missing_eligible_listing_row(tmp_path: Path) -> None:
    """A panel that omits every row for one of the run's own reviewed
    eligible listings must fail, not merely accept whatever subset of
    listings it happens to declare rows for."""
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(
        tmp_path, listing_count=2
    )
    medium = _medium_lane_expectation()
    kept_listing = listings[0]
    rows = [
        _panel_test_row(
            horizon=horizon,
            listing=kept_listing,
            price_asset=price_assets[kept_listing.id],
            target_date=run.target_date,
        )
        for horizon in sorted(_MEDIUM_HORIZONS)
    ]
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        row_overrides=rows,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_listing_membership_mismatch"


def test_medium_panel_binding_rejects_unexpected_extra_listing_row(tmp_path: Path) -> None:
    """A panel that declares a row for a listing outside the run's own
    reviewed eligible membership must fail."""
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    outsider = _long_listing("MEDOUT")
    outsider_price_asset = _price_source_asset(
        subject=outsider.ticker, retrieved_at=run.generated_at
    )
    rows = [
        _panel_test_row(
            horizon=horizon,
            listing=listing,
            price_asset=price_assets[listing.id],
            target_date=run.target_date,
        )
        for listing in listings
        for horizon in sorted(_MEDIUM_HORIZONS)
    ]
    rows.append(
        _panel_test_row(
            horizon=sorted(_MEDIUM_HORIZONS)[0],
            listing=outsider,
            price_asset=outsider_price_asset,
            target_date=run.target_date,
        )
    )
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        row_overrides=rows,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_listing_membership_mismatch"


def test_medium_panel_binding_rejects_missing_forecast_horizon_row(tmp_path: Path) -> None:
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(
        tmp_path, listing_count=1
    )
    medium = _medium_lane_expectation()
    listing = listings[0]
    rows = [
        _panel_test_row(
            horizon=sorted(_MEDIUM_HORIZONS)[0],
            listing=listing,
            price_asset=price_assets[listing.id],
            target_date=run.target_date,
        )
    ]
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        row_overrides=rows,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_forecast_rows_missing"


def test_medium_panel_binding_rejects_inconsistent_row_price_asset(tmp_path: Path) -> None:
    """The same listing must never be bound to more than one distinct
    price asset across its own rows in one panel."""
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(
        tmp_path, listing_count=1
    )
    medium = _medium_lane_expectation()
    listing = listings[0]
    alternate_price_asset = _price_source_asset(
        subject=listing.ticker, retrieved_at=run.generated_at
    )
    horizons = sorted(_MEDIUM_HORIZONS)
    rows = [
        _panel_test_row(
            horizon=horizons[0],
            listing=listing,
            price_asset=price_assets[listing.id],
            target_date=run.target_date,
        ),
        _panel_test_row(
            horizon=horizons[1],
            listing=listing,
            price_asset=alternate_price_asset,
            target_date=run.target_date,
        ),
    ]
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        row_overrides=rows,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_row_price_asset_inconsistent"


def test_medium_panel_binding_rejects_row_price_asset_for_wrong_listing(tmp_path: Path) -> None:
    """A panel row whose `price_asset_id` resolves to a real, registered
    `price_history` asset -- but one belonging to a different listing --
    must fail, not merely accept any resolvable asset id."""
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(
        tmp_path, listing_count=2
    )
    medium = _medium_lane_expectation()
    first, second = listings
    rows = [
        _panel_test_row(
            horizon=horizon,
            listing=first,
            price_asset=price_assets[second.id],
            target_date=run.target_date,
        )
        for horizon in sorted(_MEDIUM_HORIZONS)
    ] + [
        _panel_test_row(
            horizon=horizon,
            listing=second,
            price_asset=price_assets[second.id],
            target_date=run.target_date,
        )
        for horizon in sorted(_MEDIUM_HORIZONS)
    ]
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        row_overrides=rows,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_row_price_asset_mismatch"


def test_medium_panel_binding_rejects_omitted_source_asset(tmp_path: Path) -> None:
    """A panel that fails to declare one eligible listing's own price
    asset in its source closure must fail even though every row's own
    `price_asset_id` is otherwise perfectly correct -- the closure must be
    independently *derived*, not merely trusted from the panel's own
    declared list, but the declared benchmark entry is still required for
    lookup, so this corrupts the source manifest hash directly."""
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        metadata_overrides={"source_manifest_hash": "0" * 64},
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_source_manifest_hash_mismatch"


def test_medium_panel_binding_rejects_evidence_bundle_hash_tamper(tmp_path: Path) -> None:
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        metadata_overrides={"evidence_bundle_hash": "tampered" * 8},
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_evidence_bundle_hash_mismatch"


def test_medium_panel_binding_rejects_missing_benchmark_reference(tmp_path: Path) -> None:
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    rows = [
        _panel_test_row(
            horizon=horizon,
            listing=listing,
            price_asset=price_assets[listing.id],
            target_date=run.target_date,
        )
        for listing in listings
        for horizon in sorted(_MEDIUM_HORIZONS)
    ]
    payload = _panel_frame_bytes(rows)
    stored = store.write_bytes(f"medium-tests/{uuid4().hex}.parquet", payload)
    sessions = calendar_sessions_through(
        calendar_name=medium.calendar, fixed_epoch=medium.fixed_epoch, target_date=run.target_date
    )
    calendar_hash = hash_json([session.isoformat() for session in sessions])
    # Source closure omits the benchmark entirely.
    closure = [price_assets[listing.id] for listing in listings]
    source_manifest = [asset_identity(asset) for asset in dedupe_assets(closure)]
    source_manifest_hash = hash_json(source_manifest)
    universe_snapshot = run.universe_snapshot
    metadata: dict[str, object] = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "method_version": medium.method_version,
        "config_hash": medium.config_hash,
        "code_revision": run.code_revision,
        "calendar": medium.calendar,
        "calendar_library_version": "test-calendar-lib",
        "panel_library_version": "test-panel-lib",
        "fixed_epoch": medium.fixed_epoch.isoformat(),
        "calendar_hash": calendar_hash,
        "target_date": run.target_date.isoformat(),
        "universe_snapshot_id": str(universe_snapshot.id),
        "universe_slug": universe_snapshot.universe.slug,
        "universe_config_hash": universe_snapshot.config_hash,
        "scoring_config_version": run.config_version,
        "scoring_config_hash": run.config_hash,
        "return_definition": medium.return_basis,
        "dividends_included": medium.dividends_included,
        "training_evidence_grade": "research",
        "current_universe_survivorship_bias": True,
        "usage_scope": "private_single_user_research",
        "row_count": len(rows),
        "content_sha256": stored.sha256,
        "source_manifest_hash": source_manifest_hash,
        "source_assets": source_manifest,
    }
    metadata["evidence_bundle_hash"] = hash_json(
        {
            "calendar_hash": metadata["calendar_hash"],
            "code_revision": metadata["code_revision"],
            "content_sha256": metadata["content_sha256"],
            "forecast_config_hash": metadata["config_hash"],
            "scoring_config_hash": metadata["scoring_config_hash"],
            "source_manifest_hash": metadata["source_manifest_hash"],
            "universe_config_hash": metadata["universe_config_hash"],
        }
    )
    panel = register_asset(
        provider=PANEL_PROVIDER,
        kind=PANEL_KIND,
        subject=str(run.id),
        stored=stored,
        retrieved_at=run.generated_at,
        available_at=run.generated_at,
        metadata=metadata,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_benchmark_reference_invalid"


def test_medium_panel_binding_rejects_missing_reference() -> None:
    run = _bare_run()
    medium = _medium_lane_expectation()
    prediction = Prediction(
        id=uuid4(), target_date=run.target_date, data_cutoff=run.data_cutoff, calculation={}
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_medium_panel_binding(
            prediction,
            medium=medium,
            run=run,
            scoring_config_version=run.config_version,
            eligible_listings={},
            registry=_AssetRegistry(),
            panel_cache={},
        )
    assert _reason(exc_info) == "medium_panel_reference_missing"


def test_medium_panel_binding_rejects_undeclared_panel(tmp_path: Path) -> None:
    """A panel referenced by `calculation` but never actually present in
    this prediction's own declared `source_assets` (the closure the panel
    must appear in exactly once) must fail even if the panel asset itself
    is perfectly real and resolvable."""
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
    )
    prediction = Prediction(
        id=uuid4(),
        target_date=run.target_date,
        data_cutoff=run.data_cutoff,
        calculation={"panel_asset_id": str(panel.id), "panel_sha256": panel.sha256},
        source_assets=[],
    )
    listings_by_id = {listing.id: listing for listing in listings}
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_medium_panel_binding(
            prediction,
            medium=medium,
            run=run,
            scoring_config_version=run.config_version,
            eligible_listings=listings_by_id,
            registry=_AssetRegistry(),
            panel_cache={},
        )
    assert _reason(exc_info) == "medium_panel_not_declared"


def test_medium_panel_binding_rejects_late_panel(tmp_path: Path) -> None:
    """`_require_authoritative_medium_panel`'s own cutoff check on the panel
    asset itself, in isolation from the generic per-prediction
    `source_assets` cutoff check (which would otherwise always trip first
    in the full pipeline, since a medium prediction's own source_assets
    always includes its panel ref)."""
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(tmp_path)
    medium = _medium_lane_expectation()
    _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        retrieved_at=run.data_cutoff + timedelta(days=1),
    )
    listings_by_id = {listing.id: listing for listing in listings}
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_authoritative_medium_panel(
            run=run,
            medium=medium,
            scoring_config_version=run.config_version,
            eligible_listings=listings_by_id,
        )
    assert _reason(exc_info) == "medium_panel_not_cutoff_safe"


def test_medium_panel_binding_rejects_late_price_source_asset(tmp_path: Path) -> None:
    store, run, listings, _stale_price_assets, benchmark_asset = _medium_lane_fixture(
        tmp_path, listing_count=1
    )
    medium = _medium_lane_expectation()
    late_listing = listings[0]
    price_assets = {
        late_listing.id: _price_source_asset(
            subject=late_listing.ticker, retrieved_at=run.data_cutoff + timedelta(days=1)
        )
    }
    rows = [
        _panel_test_row(
            horizon=horizon,
            listing=late_listing,
            price_asset=price_assets[late_listing.id],
            target_date=run.target_date,
        )
        for horizon in sorted(_MEDIUM_HORIZONS)
    ]
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=medium,
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        row_overrides=rows,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(run=run, medium=medium, listings=listings, panel=panel)
    assert _reason(exc_info) == "medium_panel_row_price_asset_not_cutoff_safe"


def test_medium_panel_binding_requires_each_price_source_raw_upstream(tmp_path: Path) -> None:
    store, run, listings, price_assets, benchmark_asset = _medium_lane_fixture(
        tmp_path, listing_count=1
    )
    listing = listings[0]
    price_assets[listing.id] = DataAsset.objects.create(
        provider="twelve_data",
        kind="price_history",
        subject=listing.ticker,
        relative_path=f"tests/manifest/{uuid4().hex}",
        sha256=uuid4().hex * 2,
        retrieved_at=run.generated_at,
        available_at=run.generated_at,
        metadata={},
    )
    rows = [
        _panel_test_row(
            horizon=horizon,
            listing=listing,
            price_asset=price_assets[listing.id],
            target_date=run.target_date,
        )
        for horizon in sorted(_MEDIUM_HORIZONS)
    ]
    panel = _build_test_medium_panel(
        store,
        run=run,
        medium=_medium_lane_expectation(),
        scoring_config_version=run.config_version,
        listings=listings,
        price_assets=price_assets,
        benchmark_asset=benchmark_asset,
        row_overrides=rows,
    )

    with pytest.raises(RefreshVerificationError) as exc_info:
        _bind_and_verify_medium_panel(
            run=run,
            medium=_medium_lane_expectation(),
            listings=listings,
            panel=panel,
        )

    assert _reason(exc_info) == "price_asset_raw_link_missing"


def _fact_and_filing(
    *, company: Company, available_at: datetime, cik_subject: str
) -> tuple[FundamentalFact, DataAsset]:
    companyfacts = DataAsset.objects.create(
        provider="sec",
        kind="sec_companyfacts",
        subject=cik_subject,
        relative_path=f"tests/manifest/{uuid4().hex}",
        sha256=uuid4().hex * 2,
        retrieved_at=available_at,
        available_at=available_at,
    )
    filing = DataAsset.objects.create(
        provider="sec",
        kind="sec_submissions",
        subject=f"{cik_subject}-filing",
        relative_path=f"tests/manifest/{uuid4().hex}",
        sha256=uuid4().hex * 2,
        retrieved_at=available_at,
        available_at=available_at,
    )
    fact = FundamentalFact.objects.create(
        company=company,
        provider="sec",
        concept="revenue",
        source_concept="us-gaap:Revenues",
        value=Decimal("100"),
        unit="USD",
        currency="USD",
        period_end=TARGET_DATE,
        fiscal_year=TARGET_DATE.year,
        fiscal_period="FY",
        accession=f"acc-{uuid4().hex[:8]}",
        available_at=available_at,
        source_asset=companyfacts,
    )
    FundamentalFactEvidence.objects.create(
        fact=fact,
        role=FundamentalFactEvidence.Role.FILING,
        source_asset=filing,
    )
    fact.refresh_from_db()
    return fact, filing


def _company() -> Company:
    return Company.objects.create(name=f"Long Co {uuid4().hex[:6]}", country="US")


def _classification_and_asset(
    *, company: Company, available_at: datetime, subject: str, code: str = "3571"
) -> tuple[CompanyClassificationObservation, DataAsset]:
    source_asset = DataAsset.objects.create(
        provider="sec",
        kind="sec_submissions",
        subject=subject,
        relative_path=f"tests/manifest/{uuid4().hex}",
        sha256=uuid4().hex * 2,
        retrieved_at=available_at,
        available_at=available_at,
    )
    classification = CompanyClassificationObservation.objects.create(
        company=company,
        provider="sec",
        scheme="sec_sic",
        code=code,
        description="Test industry",
        observed_at=available_at,
        available_at=available_at,
        source_asset=source_asset,
    )
    return classification, source_asset


def _own_refs_and_registry(
    *, source_asset_rows: list[DataAsset], cutoff: datetime
) -> tuple[set[uuid.UUID], _AssetRegistry]:
    """A prediction's own declared source-assets closure, pre-bound into a
    fresh registry exactly the way `verify_analysis_output_manifest`'s own
    per-prediction `_require_source_assets_bound` call would before ever
    calling `_require_long_evidence_binding`."""
    registry = _AssetRegistry()
    entries = [_source_entry(asset) for asset in source_asset_rows]
    refs = _require_source_assets_bound(
        entries, context="Prediction test", cutoff=cutoff, registry=registry
    )
    return {ref.id for ref in refs}, registry


def _prediction_for(
    listing: Listing, *, calculation: dict, source_assets: list | None = None, cutoff: datetime
) -> Prediction:
    analysis = StockAnalysis(listing=listing)
    return Prediction(
        analysis=analysis,
        listing=listing,
        data_cutoff=cutoff,
        calculation=calculation,
        source_assets=source_assets or [],
    )


@dataclass
class _LongFixture:
    listing: Listing
    company: Company
    target_price_asset: DataAsset
    target_classification: CompanyClassificationObservation
    target_classification_asset: DataAsset
    input_fact: FundamentalFact
    input_filing_asset: DataAsset
    peer_listing: Listing
    peer_company: Company
    peer_price_asset: DataAsset
    peer_classification: CompanyClassificationObservation
    peer_classification_asset: DataAsset
    peer_fact: FundamentalFact
    peer_filing_asset: DataAsset


def _build_long_fixture(*, cutoff: datetime = DECISION_TIME) -> _LongFixture:
    listing = _long_listing(f"LNG{uuid4().hex[:6].upper()}")
    company = listing.security.company
    target_price_asset = _price_source_asset(subject=listing.ticker, retrieved_at=cutoff)
    target_classification, target_classification_asset = _classification_and_asset(
        company=company, available_at=cutoff, subject=f"{listing.ticker}-target-sic"
    )
    input_fact, input_filing_asset = _fact_and_filing(
        company=company, available_at=cutoff, cik_subject=f"{listing.ticker}-cik"
    )

    peer_listing = _long_listing(f"PEER{uuid4().hex[:6].upper()}")
    peer_company = peer_listing.security.company
    peer_price_asset = _price_source_asset(subject=peer_listing.ticker, retrieved_at=cutoff)
    peer_classification, peer_classification_asset = _classification_and_asset(
        company=peer_company, available_at=cutoff, subject=f"{peer_listing.ticker}-peer-sic"
    )
    peer_fact, peer_filing_asset = _fact_and_filing(
        company=peer_company, available_at=cutoff, cik_subject=f"{peer_listing.ticker}-cik"
    )
    return _LongFixture(
        listing=listing,
        company=company,
        target_price_asset=target_price_asset,
        target_classification=target_classification,
        target_classification_asset=target_classification_asset,
        input_fact=input_fact,
        input_filing_asset=input_filing_asset,
        peer_listing=peer_listing,
        peer_company=peer_company,
        peer_price_asset=peer_price_asset,
        peer_classification=peer_classification,
        peer_classification_asset=peer_classification_asset,
        peer_fact=peer_fact,
        peer_filing_asset=peer_filing_asset,
    )


def _calculation_for(fx: _LongFixture) -> dict:
    return {
        "target_price_asset_id": str(fx.target_price_asset.id),
        "target_classification": classification_payload(fx.target_classification),
        "input_facts": [
            fact_payload(fx.input_fact, {str(fx.input_fact.pk): fx.input_filing_asset})
        ],
        "peer_set": [
            {
                "listing_id": str(fx.peer_listing.id),
                "ticker": fx.peer_listing.ticker,
                "sic": fx.peer_classification.code,
                "classification_id": str(fx.peer_classification.id),
                "classification": classification_payload(fx.peer_classification),
                "price_asset_id": str(fx.peer_price_asset.id),
                "fact_references": [
                    fact_reference(fx.peer_fact, {str(fx.peer_fact.pk): fx.peer_filing_asset})
                ],
            }
        ],
    }


def _all_source_assets(fx: _LongFixture) -> list[DataAsset]:
    return [
        fx.target_price_asset,
        fx.target_classification_asset,
        fx.input_fact.source_asset,
        fx.input_filing_asset,
        fx.peer_price_asset,
        fx.peer_classification_asset,
        fx.peer_fact.source_asset,
        fx.peer_filing_asset,
    ]


def test_long_evidence_binding_succeeds_for_exact_closure() -> None:
    fx = _build_long_fixture()
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=_calculation_for(fx), cutoff=DECISION_TIME)
    _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    for asset in _all_source_assets(fx):
        assert asset.id in registry.refs


def test_long_evidence_binding_rejects_extra_undeclared_purpose_source() -> None:
    """An extra source asset declared in the prediction's own source_assets
    but never actually named anywhere in the calculation's evidence closure
    must fail exact-closure enforcement, even though every named reference
    itself resolves perfectly."""
    fx = _build_long_fixture()
    extra_asset = _price_source_asset(subject="UNRELATED", retrieved_at=DECISION_TIME)
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=[*_all_source_assets(fx), extra_asset], cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=_calculation_for(fx), cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_source_assets_extra"


def test_decision_source_closure_exact_succeeds_when_sets_match() -> None:
    shared_id = uuid4()
    _require_decision_source_closure_exact(own_ids={shared_id}, analysis_ref_ids={shared_id})


def test_decision_source_closure_exact_rejects_prediction_extra_source() -> None:
    """A decision prediction declaring a source asset its own StockAnalysis
    never declared must fail, even though that asset resolves perfectly on
    its own."""
    shared_id = uuid4()
    extra_id = uuid4()
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_decision_source_closure_exact(
            own_ids={shared_id, extra_id}, analysis_ref_ids={shared_id}
        )
    assert _reason(exc_info) == "decision_prediction_source_assets_mismatch"


def test_decision_source_closure_exact_rejects_analysis_extra_source() -> None:
    """A StockAnalysis declaring a source asset its own decision prediction
    never bound must fail just as closed as the reverse direction."""
    shared_id = uuid4()
    extra_id = uuid4()
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_decision_source_closure_exact(
            own_ids={shared_id}, analysis_ref_ids={shared_id, extra_id}
        )
    assert _reason(exc_info) == "decision_prediction_source_assets_mismatch"


def test_medium_source_closure_exact_succeeds_when_sets_match() -> None:
    shared_id = uuid4()
    panel_id = uuid4()
    _require_medium_source_closure_exact(
        own_ids={shared_id, panel_id}, analysis_ref_ids={shared_id}, panel_asset_id=panel_id
    )


def test_medium_source_closure_exact_rejects_extra_source() -> None:
    """A medium prediction declaring a source asset beyond its own
    StockAnalysis's declared sources and its own bound panel must fail."""
    shared_id = uuid4()
    panel_id = uuid4()
    extra_id = uuid4()
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_medium_source_closure_exact(
            own_ids={shared_id, panel_id, extra_id},
            analysis_ref_ids={shared_id},
            panel_asset_id=panel_id,
        )
    assert _reason(exc_info) == "medium_prediction_source_assets_mismatch"


def test_medium_source_closure_exact_rejects_missing_analysis_source() -> None:
    """A medium prediction missing one of its own StockAnalysis's declared
    sources must fail just as closed as declaring an extra one."""
    shared_id = uuid4()
    missing_id = uuid4()
    panel_id = uuid4()
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_medium_source_closure_exact(
            own_ids={shared_id, panel_id},
            analysis_ref_ids={shared_id, missing_id},
            panel_asset_id=panel_id,
        )
    assert _reason(exc_info) == "medium_prediction_source_assets_mismatch"


def test_long_evidence_binding_rejects_missing_target_price_asset_id() -> None:
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    del calculation["target_price_asset_id"]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_reference_malformed"


def test_long_evidence_binding_rejects_undeclared_target_price_asset() -> None:
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=[a for a in _all_source_assets(fx) if a != fx.target_price_asset],
        cutoff=DECISION_TIME,
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_target_price_asset_not_declared"


def test_long_evidence_binding_rejects_target_price_asset_substitution() -> None:
    """A target_price_asset_id that resolves to a real, declared price
    asset belonging to a *different* listing/subject must fail even though
    the asset itself is genuinely a price_history row."""
    fx = _build_long_fixture()
    other_price_asset = _price_source_asset(subject="OTHERX", retrieved_at=DECISION_TIME)
    calculation = _calculation_for(fx)
    calculation["target_price_asset_id"] = str(other_price_asset.id)
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=[*_all_source_assets(fx), other_price_asset], cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_target_price_asset_mismatch"


def test_long_evidence_binding_rejects_target_classification_transplant() -> None:
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    del calculation["target_classification"]["description"]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_classification_identity_mismatch"


def test_long_evidence_binding_rejects_target_classification_extra_field() -> None:
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    calculation["target_classification"]["unexpected"] = "value"
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_classification_identity_mismatch"


def test_long_evidence_binding_rejects_cross_company_target_classification() -> None:
    fx = _build_long_fixture()
    outsider_company = _company()
    cross_classification, cross_asset = _classification_and_asset(
        company=outsider_company, available_at=DECISION_TIME, subject="cross-sic"
    )
    calculation = _calculation_for(fx)
    calculation["target_classification"] = classification_payload(cross_classification)
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=[*_all_source_assets(fx), cross_asset], cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_classification_company_mismatch"


def test_long_evidence_binding_rejects_input_fact_missing_field() -> None:
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    del calculation["input_facts"][0]["concept"]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_fact_identity_mismatch"


def test_long_evidence_binding_rejects_input_fact_extra_field() -> None:
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    calculation["input_facts"][0]["unexpected"] = "value"
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_fact_identity_mismatch"


def test_long_evidence_binding_rejects_cross_company_input_fact() -> None:
    fx = _build_long_fixture()
    outsider_company = _company()
    outsider_fact, outsider_filing = _fact_and_filing(
        company=outsider_company, available_at=DECISION_TIME, cik_subject="outsider-cik"
    )
    calculation = _calculation_for(fx)
    calculation["input_facts"] = [
        fact_payload(outsider_fact, {str(outsider_fact.pk): outsider_filing})
    ]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=[
            *_all_source_assets(fx),
            outsider_fact.source_asset,
            outsider_filing,
        ],
        cutoff=DECISION_TIME,
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_fact_company_mismatch"


def test_long_evidence_binding_rejects_input_fact_missing_filing() -> None:
    fx = _build_long_fixture()
    unlinked_source = DataAsset.objects.create(
        provider="sec",
        kind="sec_companyfacts",
        subject="unlinked-cik",
        relative_path=f"tests/manifest/{uuid4().hex}",
        sha256=uuid4().hex * 2,
        retrieved_at=DECISION_TIME,
        available_at=DECISION_TIME,
    )
    unlinked_fact = FundamentalFact.objects.create(
        company=fx.company,
        provider="sec",
        concept="revenue",
        source_concept="us-gaap:Revenues",
        value=Decimal("100"),
        unit="USD",
        currency="USD",
        period_end=TARGET_DATE,
        fiscal_year=TARGET_DATE.year,
        fiscal_period="FY",
        accession=f"acc-{uuid4().hex[:8]}",
        available_at=DECISION_TIME,
        source_asset=unlinked_source,
    )
    calculation = _calculation_for(fx)
    calculation["input_facts"] = [
        {
            "id": str(unlinked_fact.id),
            "provider": unlinked_fact.provider,
            "concept": unlinked_fact.concept,
            "taxonomy": unlinked_fact.taxonomy,
            "source_concept": unlinked_fact.source_concept,
            "value": str(unlinked_fact.value),
            "unit": unlinked_fact.unit,
            "currency": unlinked_fact.currency,
            "period_type": unlinked_fact.period_type,
            "period_identity": unlinked_fact.period_identity,
            "period_start": None,
            "period_end": unlinked_fact.period_end.isoformat(),
            "fiscal_year": unlinked_fact.fiscal_year,
            "fiscal_period": unlinked_fact.fiscal_period,
            "frame": unlinked_fact.frame,
            "accession": unlinked_fact.accession,
            "filing_form": unlinked_fact.filing_form,
            "filing_date": None,
            "filed_at": None,
            "acceptance_at": None,
            "available_at": unlinked_fact.available_at.isoformat(),
            "availability_basis": unlinked_fact.availability_basis,
            "is_amendment": unlinked_fact.is_amendment,
            "source_revision": unlinked_fact.source_revision,
            "observation_hash": unlinked_fact.observation_hash,
            "quality_flags": unlinked_fact.quality_flags,
            "source_asset_id": str(unlinked_fact.source_asset_id),
            "filing_evidence_asset_id": str(uuid4()),
        }
    ]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=[*_all_source_assets(fx), unlinked_source],
        cutoff=DECISION_TIME,
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_filing_missing"


def test_long_evidence_binding_rejects_peer_ticker_mismatch() -> None:
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    calculation["peer_set"][0]["ticker"] = "WRONGTICKER"
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_peer_ticker_mismatch"


def test_long_evidence_binding_rejects_peer_price_asset_subject_mismatch() -> None:
    fx = _build_long_fixture()
    other_price_asset = _price_source_asset(subject="OTHERPEER", retrieved_at=DECISION_TIME)
    calculation = _calculation_for(fx)
    calculation["peer_set"][0]["price_asset_id"] = str(other_price_asset.id)
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=[*_all_source_assets(fx), other_price_asset], cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_price_asset_kind_mismatch"


def test_long_evidence_binding_rejects_peer_price_asset_undeclared() -> None:
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=[a for a in _all_source_assets(fx) if a != fx.peer_price_asset],
        cutoff=DECISION_TIME,
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_price_asset_not_declared"


def test_long_evidence_binding_rejects_peer_classification_id_inconsistency() -> None:
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    calculation["peer_set"][0]["classification_id"] = str(uuid4())
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_classification_id_mismatch"


def test_long_evidence_binding_rejects_cross_company_peer_classification() -> None:
    fx = _build_long_fixture()
    outsider_company = _company()
    cross_classification, cross_asset = _classification_and_asset(
        company=outsider_company, available_at=DECISION_TIME, subject="cross-peer-sic"
    )
    calculation = _calculation_for(fx)
    calculation["peer_set"][0]["classification_id"] = str(cross_classification.id)
    calculation["peer_set"][0]["classification"] = classification_payload(cross_classification)
    calculation["peer_set"][0]["sic"] = cross_classification.code
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=[*_all_source_assets(fx), cross_asset], cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_classification_company_mismatch"


def test_long_evidence_binding_rejects_cross_company_peer_fact() -> None:
    fx = _build_long_fixture()
    outsider_company = _company()
    outsider_fact, outsider_filing = _fact_and_filing(
        company=outsider_company, available_at=DECISION_TIME, cik_subject="outsider-peer-cik"
    )
    calculation = _calculation_for(fx)
    calculation["peer_set"][0]["fact_references"] = [
        fact_reference(outsider_fact, {str(outsider_fact.pk): outsider_filing})
    ]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=[
            *_all_source_assets(fx),
            outsider_fact.source_asset,
            outsider_filing,
        ],
        cutoff=DECISION_TIME,
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_fact_company_mismatch"


def test_long_evidence_binding_rejects_peer_fact_reference_transplant() -> None:
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    del calculation["peer_set"][0]["fact_references"][0]["accession"]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_fact_identity_mismatch"


def test_long_evidence_binding_rejects_peer_fact_reference_extra_field() -> None:
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    calculation["peer_set"][0]["fact_references"][0]["unexpected"] = "value"
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_fact_identity_mismatch"


def test_long_evidence_binding_proves_bare_fact_id_closure() -> None:
    """A bare fact id closure list (e.g. `manifest_evidence_fact_ids`,
    carrying no accompanying identity fields of its own) must still
    resolve to a real, cutoff-eligible fact whose source asset *and*
    filing evidence asset are declared in the prediction's own
    `source_assets` -- proving the calculation cannot silently cite
    evidence never actually declared."""
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    calculation["closure_diagnostics"] = {"manifest_evidence_fact_ids": [str(fx.input_fact.id)]}
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert fx.input_filing_asset.id in registry.refs


def test_long_evidence_binding_rejects_undeclared_bare_fact_id_closure() -> None:
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    outsider_fact, outsider_filing = _fact_and_filing(
        company=fx.company, available_at=DECISION_TIME, cik_subject="bare-undeclared-cik"
    )
    calculation["closure_diagnostics"] = {"manifest_evidence_fact_ids": [str(outsider_fact.id)]}
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_source_not_declared"


def test_long_evidence_binding_rejects_alternate_id_bare_fact_closure() -> None:
    """A same-content fact registered under an alternate id must fail:
    the literal cited id in the closure list resolves to nothing."""
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    calculation["closure_diagnostics"] = {"assessed_evidence_fact_ids": [str(uuid4())]}
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_fact_missing"


def test_long_evidence_binding_rejects_bare_fact_id_after_cutoff() -> None:
    fx = _build_long_fixture()
    late_fact, _late_filing = _fact_and_filing(
        company=fx.company,
        available_at=DECISION_TIME + timedelta(days=1),
        cik_subject="late-bare-cik",
    )
    calculation = _calculation_for(fx)
    calculation["closure_diagnostics"] = {"manifest_evidence_fact_ids": [str(late_fact.id)]}
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_fact_after_cutoff"


def _assessed_evidence_selection(
    fx: _LongFixture, assessed_fact: FundamentalFact, assessed_filing: DataAsset
) -> dict:
    """A valid `evidence_selection` payload: `fx.input_fact` selected,
    `assessed_fact` assessed-but-not-selected, both in the manifest union."""
    return {
        "selected_input_fact_ids": [str(fx.input_fact.id)],
        "assessed_evidence_fact_ids": [str(assessed_fact.id)],
        "assessed_evidence": [
            fact_reference(assessed_fact, {str(assessed_fact.pk): assessed_filing})
        ],
        "manifest_evidence_fact_ids": [str(fx.input_fact.id), str(assessed_fact.id)],
    }


def _long_fixture_with_assessed(
    *, cutoff: datetime = DECISION_TIME
) -> tuple[_LongFixture, FundamentalFact, DataAsset]:
    fx = _build_long_fixture(cutoff=cutoff)
    assessed_fact, assessed_filing = _fact_and_filing(
        company=fx.company, available_at=cutoff, cik_subject=f"{fx.listing.ticker}-assessed-cik"
    )
    return fx, assessed_fact, assessed_filing


def test_evidence_selection_binding_succeeds_for_exact_closure() -> None:
    fx, assessed_fact, assessed_filing = _long_fixture_with_assessed()
    calculation = _calculation_for(fx)
    calculation["evidence_selection"] = _assessed_evidence_selection(
        fx, assessed_fact, assessed_filing
    )
    source_assets = [
        *_all_source_assets(fx),
        assessed_fact.source_asset,
        assessed_filing,
    ]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=source_assets, cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert assessed_filing.id in registry.refs


def test_evidence_selection_binding_rejects_non_object() -> None:
    fx = _build_long_fixture()
    calculation = _calculation_for(fx)
    calculation["evidence_selection"] = ["not", "an", "object"]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=_all_source_assets(fx), cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_selection_malformed"


def test_evidence_selection_binding_rejects_duplicate_id_within_field() -> None:
    fx, assessed_fact, assessed_filing = _long_fixture_with_assessed()
    calculation = _calculation_for(fx)
    selection = _assessed_evidence_selection(fx, assessed_fact, assessed_filing)
    selection["selected_input_fact_ids"] = [str(fx.input_fact.id), str(fx.input_fact.id)]
    calculation["evidence_selection"] = selection
    source_assets = [*_all_source_assets(fx), assessed_fact.source_asset, assessed_filing]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=source_assets, cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_selection_duplicate"


def test_evidence_selection_binding_rejects_selected_assessed_overlap() -> None:
    fx, assessed_fact, assessed_filing = _long_fixture_with_assessed()
    calculation = _calculation_for(fx)
    selection = _assessed_evidence_selection(fx, assessed_fact, assessed_filing)
    # Same fact id appears in both the selected and assessed sets.
    selection["assessed_evidence_fact_ids"].append(str(fx.input_fact.id))
    calculation["evidence_selection"] = selection
    source_assets = [*_all_source_assets(fx), assessed_fact.source_asset, assessed_filing]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=source_assets, cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_selection_overlap"


def test_evidence_selection_binding_rejects_incomplete_manifest_union() -> None:
    fx, assessed_fact, assessed_filing = _long_fixture_with_assessed()
    calculation = _calculation_for(fx)
    selection = _assessed_evidence_selection(fx, assessed_fact, assessed_filing)
    selection["manifest_evidence_fact_ids"] = [str(fx.input_fact.id)]
    calculation["evidence_selection"] = selection
    source_assets = [*_all_source_assets(fx), assessed_fact.source_asset, assessed_filing]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=source_assets, cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_selection_union_mismatch"


def test_evidence_selection_binding_rejects_extra_manifest_id() -> None:
    fx, assessed_fact, assessed_filing = _long_fixture_with_assessed()
    calculation = _calculation_for(fx)
    selection = _assessed_evidence_selection(fx, assessed_fact, assessed_filing)
    selection["manifest_evidence_fact_ids"].append(str(uuid4()))
    calculation["evidence_selection"] = selection
    source_assets = [*_all_source_assets(fx), assessed_fact.source_asset, assessed_filing]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=source_assets, cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_selection_union_mismatch"


def test_evidence_selection_binding_rejects_selected_not_matching_input_facts() -> None:
    fx, assessed_fact, assessed_filing = _long_fixture_with_assessed()
    calculation = _calculation_for(fx)
    selection = _assessed_evidence_selection(fx, assessed_fact, assessed_filing)
    selection["selected_input_fact_ids"] = [str(uuid4())]
    selection["manifest_evidence_fact_ids"] = [
        selection["selected_input_fact_ids"][0],
        str(assessed_fact.id),
    ]
    calculation["evidence_selection"] = selection
    source_assets = [*_all_source_assets(fx), assessed_fact.source_asset, assessed_filing]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=source_assets, cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_selection_selected_mismatch"


def test_evidence_selection_binding_rejects_forged_assessed_evidence_field() -> None:
    """`assessed_evidence`'s abbreviated item must equal `fact_reference(...)`
    exactly; a payload that alters/removes a field must fail even though the
    fact id itself is correct."""
    fx, assessed_fact, assessed_filing = _long_fixture_with_assessed()
    calculation = _calculation_for(fx)
    selection = _assessed_evidence_selection(fx, assessed_fact, assessed_filing)
    selection["assessed_evidence"][0]["concept"] = "tampered-concept"
    calculation["evidence_selection"] = selection
    source_assets = [*_all_source_assets(fx), assessed_fact.source_asset, assessed_filing]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=source_assets, cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_fact_identity_mismatch"


def test_evidence_selection_binding_rejects_missing_assessed_evidence_field() -> None:
    """A `fact_payload`-shaped extra field is not the abbreviated shape --
    an extra key must also fail exact-equality."""
    fx, assessed_fact, assessed_filing = _long_fixture_with_assessed()
    calculation = _calculation_for(fx)
    selection = _assessed_evidence_selection(fx, assessed_fact, assessed_filing)
    selection["assessed_evidence"][0]["value"] = "999"
    calculation["evidence_selection"] = selection
    source_assets = [*_all_source_assets(fx), assessed_fact.source_asset, assessed_filing]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=source_assets, cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_fact_identity_mismatch"


def test_evidence_selection_binding_rejects_cross_company_assessed_evidence() -> None:
    """A peer's fact must not be usable as the target's assessed evidence."""
    fx, assessed_fact, assessed_filing = _long_fixture_with_assessed()
    calculation = _calculation_for(fx)
    peer_evidence = _assessed_evidence_selection(fx, fx.peer_fact, fx.peer_filing_asset)
    calculation["evidence_selection"] = peer_evidence
    source_assets = _all_source_assets(fx)
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=source_assets, cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_fact_company_mismatch"


def test_evidence_selection_binding_rejects_duplicate_assessed_evidence_entry() -> None:
    fx, assessed_fact, assessed_filing = _long_fixture_with_assessed()
    calculation = _calculation_for(fx)
    selection = _assessed_evidence_selection(fx, assessed_fact, assessed_filing)
    selection["assessed_evidence"].append(dict(selection["assessed_evidence"][0]))
    calculation["evidence_selection"] = selection
    source_assets = [*_all_source_assets(fx), assessed_fact.source_asset, assessed_filing]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=source_assets, cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_selection_duplicate"


def test_evidence_selection_binding_rejects_assessed_evidence_id_mismatch() -> None:
    """`assessed_evidence`'s own ids must equal `assessed_evidence_fact_ids`
    exactly -- an extra abbreviated entry not named in the id list fails."""
    fx, assessed_fact, assessed_filing = _long_fixture_with_assessed()
    extra_fact, extra_filing = _fact_and_filing(
        company=fx.company, available_at=DECISION_TIME, cik_subject=f"{fx.listing.ticker}-extra-cik"
    )
    calculation = _calculation_for(fx)
    selection = _assessed_evidence_selection(fx, assessed_fact, assessed_filing)
    selection["assessed_evidence"].append(
        fact_reference(extra_fact, {str(extra_fact.pk): extra_filing})
    )
    calculation["evidence_selection"] = selection
    source_assets = [
        *_all_source_assets(fx),
        assessed_fact.source_asset,
        assessed_filing,
        extra_fact.source_asset,
        extra_filing,
    ]
    own_refs, registry = _own_refs_and_registry(
        source_asset_rows=source_assets, cutoff=DECISION_TIME
    )
    prediction = _prediction_for(fx.listing, calculation=calculation, cutoff=DECISION_TIME)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_long_evidence_binding(prediction, own_refs=own_refs, registry=registry)
    assert _reason(exc_info) == "long_evidence_selection_assessed_mismatch"


def test_source_assets_bound_rejects_alternate_id_same_content_substitution() -> None:
    asset = _price_source_asset(subject="MDA", retrieved_at=DECISION_TIME)
    entry = _source_entry(asset)
    registry = _AssetRegistry()
    registry.refs[asset.id] = AssetRef(
        id=asset.id,
        provider=asset.provider,
        kind=asset.kind,
        subject=asset.subject,
        sha256="f" * 64,
    )
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_source_assets_bound(
            [entry], context="test", cutoff=DECISION_TIME, registry=registry
        )
    assert _reason(exc_info) == "source_assets_entry_conflicting"


def test_source_assets_bound_rejects_late_asset() -> None:
    asset = _price_source_asset(subject="MDA", retrieved_at=DECISION_TIME)
    entry = _source_entry(asset)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_source_assets_bound(
            [entry],
            context="test",
            cutoff=DECISION_TIME.replace(year=DECISION_TIME.year - 1),
            registry=_AssetRegistry(),
        )
    assert _reason(exc_info) == "asset_ref_after_cutoff"


def test_source_assets_bound_rejects_incomplete_entry() -> None:
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_source_assets_bound(
            [{"id": str(uuid4())}], context="test", cutoff=DECISION_TIME, registry=_AssetRegistry()
        )
    assert _reason(exc_info) == "source_assets_entry_malformed"


def test_source_assets_bound_rejects_empty_list() -> None:
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_source_assets_bound(
            [], context="test", cutoff=DECISION_TIME, registry=_AssetRegistry()
        )
    assert _reason(exc_info) == "source_assets_payload_invalid"


def test_source_assets_bound_rejects_none() -> None:
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_source_assets_bound(
            None, context="test", cutoff=DECISION_TIME, registry=_AssetRegistry()
        )
    assert _reason(exc_info) == "source_assets_payload_invalid"


def test_source_assets_bound_rejects_duplicate_entries() -> None:
    asset = _price_source_asset(subject="MDA", retrieved_at=DECISION_TIME)
    entry = _source_entry(asset)
    with pytest.raises(RefreshVerificationError) as exc_info:
        _require_source_assets_bound(
            [entry, dict(entry)], context="test", cutoff=DECISION_TIME, registry=_AssetRegistry()
        )
    assert _reason(exc_info) == "source_assets_entry_duplicated"


def test_source_assets_bound_returns_sorted_refs() -> None:
    assets = [_price_source_asset(subject=f"MDA{i}", retrieved_at=DECISION_TIME) for i in range(3)]
    entries = [_source_entry(asset) for asset in assets]
    refs = _require_source_assets_bound(
        entries, context="test", cutoff=DECISION_TIME, registry=_AssetRegistry()
    )
    assert [str(ref.id) for ref in refs] == sorted(str(ref.id) for ref in refs)
    assert {ref.id for ref in refs} == {asset.id for asset in assets}


# ---------------------------------------------------------------------------
# Group 4: writer transactional cleanup
# ---------------------------------------------------------------------------


def _bare_prediction_row(
    *, analysis: StockAnalysis, listing: Listing, run: AnalysisRun, model_version: str
) -> Prediction:
    return Prediction.objects.create(
        analysis=analysis,
        listing=listing,
        generated_at=run.generated_at,
        target_date=run.target_date,
        issued_on_time=True,
        horizon="short",
        evidence_role="decision",
        evidence_grade=run.universe_snapshot.grade,
        source_mode="synthetic",
        price_provider="twelve_data",
        price_subject=listing.provider_symbol or listing.ticker,
        price_at_prediction=Decimal("10.000000"),
        confidence=Decimal("50.00"),
        confidence_status="heuristic",
        recommendation="hold",
        overall_score=Decimal("50.00"),
        model_version=model_version,
        method_version="bare-v1",
        config_hash=run.config_hash,
        data_cutoff=run.data_cutoff,
        code_revision=run.code_revision,
    )


def _bare_persisted_analysis(*, run: AnalysisRun, listing: Listing) -> PersistedAnalysis:
    analysis = StockAnalysis.objects.create(
        run=run,
        listing=listing,
        current_price=Decimal("10.000000"),
        overall_score=Decimal("50.00"),
        recommendation="hold",
        risk_class="medium",
        confidence=Decimal("50.00"),
    )
    prediction = _bare_prediction_row(
        analysis=analysis, listing=listing, run=run, model_version="bare-decision"
    )
    return PersistedAnalysis(
        run=run, analysis=analysis, predictions=(prediction,), computation=None
    )


def test_write_analysis_output_manifest_removes_new_file_on_register_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AssetStore(tmp_path)
    run = _bare_run()
    listing = _long_listing("WBW1")
    persisted = _bare_persisted_analysis(run=run, listing=listing)

    real_create = DataAsset.objects.create

    def spy_create(*args, **kwargs):
        if kwargs.get("kind") == ANALYSIS_OUTPUT_MANIFEST_KIND:
            raise RuntimeError("boom register")
        return real_create(*args, **kwargs)

    monkeypatch.setattr(DataAsset.objects, "create", spy_create)

    plan = build_output_plan(
        eligible_listing_ids={listing.id},
        decision_horizons=frozenset({"short"}),
        medium_active=False,
        long_active=False,
    )
    with pytest.raises(RuntimeError, match="boom register"):
        _write_analysis_output_manifest(
            run=run, results=[persisted], plan=plan, store=store, retrieved_at=run.generated_at
        )

    assert not store.resolve(f"research/analysis/{run.id}/output-manifest.json").exists()
    assert not DataAsset.objects.filter(kind=ANALYSIS_OUTPUT_MANIFEST_KIND).exists()


def test_write_analysis_output_manifest_preserves_pre_existing_file_on_register_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AssetStore(tmp_path)
    run = _bare_run()
    listing = _long_listing("WBW2")
    persisted = _bare_persisted_analysis(run=run, listing=listing)

    envelope = build_manifest_envelope(
        run_id=run.id,
        plan=build_output_plan(
            eligible_listing_ids={listing.id},
            decision_horizons=frozenset({"short"}),
            medium_active=False,
            long_active=False,
        ),
        entries=_analysis_output_manifest_entries(run, [persisted]),
    )
    payload = dumps_canonical_envelope(envelope)
    relative_path = f"research/analysis/{run.id}/output-manifest.json"
    store.write_bytes(relative_path, payload)

    real_create = DataAsset.objects.create

    def spy_create(*args, **kwargs):
        if kwargs.get("kind") == ANALYSIS_OUTPUT_MANIFEST_KIND:
            raise RuntimeError("boom register 2")
        return real_create(*args, **kwargs)

    monkeypatch.setattr(DataAsset.objects, "create", spy_create)

    plan = build_output_plan(
        eligible_listing_ids={listing.id},
        decision_horizons=frozenset({"short"}),
        medium_active=False,
        long_active=False,
    )
    with pytest.raises(RuntimeError, match="boom register 2"):
        _write_analysis_output_manifest(
            run=run, results=[persisted], plan=plan, store=store, retrieved_at=run.generated_at
        )

    assert store.resolve(relative_path).exists()
    assert not DataAsset.objects.filter(kind=ANALYSIS_OUTPUT_MANIFEST_KIND).exists()


def test_write_analysis_output_manifest_cleanup_fault_does_not_mask_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AssetStore(tmp_path)
    run = _bare_run()
    listing = _long_listing("WBW3")
    persisted = _bare_persisted_analysis(run=run, listing=listing)

    def spy_create(*args, **kwargs):
        raise RuntimeError("boom register 3")

    monkeypatch.setattr(DataAsset.objects, "create", spy_create)

    def spy_unlink(self, *args, **kwargs):
        raise OSError("cleanup exploded")

    monkeypatch.setattr(Path, "unlink", spy_unlink)

    plan = build_output_plan(
        eligible_listing_ids={listing.id},
        decision_horizons=frozenset({"short"}),
        medium_active=False,
        long_active=False,
    )
    with pytest.raises(RuntimeError, match="boom register 3"):
        _write_analysis_output_manifest(
            run=run, results=[persisted], plan=plan, store=store, retrieved_at=run.generated_at
        )


# ---------------------------------------------------------------------------
# Group 5: real savepoint/atomic-exit fault injection through the actual
# observed `analyze_snapshot` orchestration -- not a monkeypatched
# `DataAsset.objects.create` raising an ordinary exception (Group 4 above),
# but Django's own `connection.savepoint_commit` genuinely releasing the
# nested savepoint and only then raising. This is the exact failure class
# that made a post-write `DataAsset.objects.filter(...).exists()` check
# unreliable (the connection is left in a doomed "needs rollback" state,
# so a further ORM query in the same `except` block would itself raise,
# masking the original error): both the medium-forecast-panel writer
# (`research.medium_forecasts.build_medium_forecast_panel`) and the
# manifest writer (`research.service._write_analysis_output_manifest`)
# must survive it with a full DB rollback and no orphaned physical file.
# ---------------------------------------------------------------------------


def test_medium_panel_savepoint_exit_failure_leaves_no_db_row_or_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, snapshot, listings = _medium_active_snapshot(tmp_path)

    armed = {"value": False}
    real_create = DataAsset.objects.create

    def spy_create(*args, **kwargs):
        created = real_create(*args, **kwargs)
        if kwargs.get("kind") == PANEL_KIND:
            armed["value"] = True
        return created

    monkeypatch.setattr(DataAsset.objects, "create", spy_create)

    real_savepoint_commit = connection.savepoint_commit

    def spy_savepoint_commit(sid: str) -> None:
        if armed["value"]:
            armed["value"] = False
            real_savepoint_commit(sid)
            raise RuntimeError("medium panel savepoint released but exit still raised")
        real_savepoint_commit(sid)

    monkeypatch.setattr(connection, "savepoint_commit", spy_savepoint_commit)

    with pytest.raises(RuntimeError, match="medium panel savepoint released but exit still raised"):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            issued_on_time=True,
            provider="twelve_data",
            benchmark_subject="SPY",
            store=store,
            config_path=default_us_scoring_config_path(),
        )

    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert not DataAsset.objects.filter(kind=PANEL_KIND).exists()
    assert not DataAsset.objects.filter(kind=ANALYSIS_OUTPUT_MANIFEST_KIND).exists()
    panel_files = list((tmp_path / "derived" / "forecast" / "medium").glob("**/*.parquet"))
    assert panel_files == []
    manifest_files = list((tmp_path / "research" / "analysis").glob("**/output-manifest.json"))
    assert manifest_files == []


def test_manifest_savepoint_exit_failure_leaves_no_db_row_or_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AssetStore(tmp_path)
    snapshot = _snapshot(slug=f"manifest-savepoint-{uuid4().hex[:8]}")
    listing = _long_listing("SPX1")
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    _write_price_asset(store, listing, close=40.0)

    armed = {"value": False}
    real_create = DataAsset.objects.create

    def spy_create(*args, **kwargs):
        created = real_create(*args, **kwargs)
        if kwargs.get("kind") == ANALYSIS_OUTPUT_MANIFEST_KIND:
            armed["value"] = True
        return created

    monkeypatch.setattr(DataAsset.objects, "create", spy_create)

    real_savepoint_commit = connection.savepoint_commit

    def spy_savepoint_commit(sid: str) -> None:
        if armed["value"]:
            armed["value"] = False
            real_savepoint_commit(sid)
            raise RuntimeError("manifest savepoint released but exit still raised")
        real_savepoint_commit(sid)

    monkeypatch.setattr(connection, "savepoint_commit", spy_savepoint_commit)

    with pytest.raises(RuntimeError, match="manifest savepoint released but exit still raised"):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            issued_on_time=True,
            provider="twelve_data",
            store=store,
            config_path=default_us_scoring_config_path(),
        )

    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert not DataAsset.objects.filter(kind=ANALYSIS_OUTPUT_MANIFEST_KIND).exists()
    manifest_files = list((tmp_path / "research" / "analysis").glob("**/output-manifest.json"))
    assert manifest_files == []


# ---------------------------------------------------------------------------
# Group 6: `analyze_listing`'s precomputed plan must derive its decision
# horizon set from the frozen, loaded `config.supported_horizons` -- never
# from `computation.data_quality["supported_horizons"]`, which is itself a
# mutable value the same computation call also uses to decide which
# decision predictions to persist (finding 4). If the two ever diverge --
# by a defect in computation, not by any legitimate production path -- the
# precomputed plan and the actually-persisted rows must disagree, and the
# existing `_finalize_observed_manifest` actual-vs-plan self-check must
# fail the whole write closed rather than silently accept whichever
# horizon set the (mutable) computation happened to produce.
# ---------------------------------------------------------------------------


def _two_decision_horizon_config_path(tmp_path: Path) -> Path:
    base_text = default_us_scoring_config_path().read_text(encoding="utf-8")
    modified_text = base_text.replace(
        "supported_horizons: [short]", "supported_horizons: [short, medium]"
    ).replace(
        "buy_max_bear_downside:\n    short: -0.08",
        "buy_max_bear_downside:\n    short: -0.08\n    medium: -0.15",
    )
    assert modified_text != base_text
    scoring_path = tmp_path / "scoring-two-decision-horizons.yml"
    scoring_path.write_text(modified_text, encoding="utf-8")
    return scoring_path


def test_analyze_listing_plan_ignores_tampered_computed_supported_horizons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewed config supports two decision horizons (`short`,
    `medium`); a defect that makes the computation's own
    `data_quality["supported_horizons"]` silently omit one must not let a
    one-horizon plan sail through -- the plan (built from frozen
    `config.supported_horizons`) still expects both, so it disagrees with
    the one-horizon actual result, and the whole run/analysis/prediction/
    manifest write rolls back."""
    store = AssetStore(tmp_path)
    snapshot = _snapshot(slug=f"single-listing-plan-{uuid4().hex[:8]}")
    listing = _long_listing("SLP1")
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    _write_price_asset(store, listing, close=40.0)
    scoring_path = _two_decision_horizon_config_path(tmp_path)

    real_compute = service._compute_listing_from_asof

    def spy_compute(**kwargs):
        computation = real_compute(**kwargs)
        computation.data_quality["supported_horizons"] = [
            horizon
            for horizon in computation.data_quality["supported_horizons"]
            if horizon != "medium"
        ]
        return computation

    monkeypatch.setattr(service, "_compute_listing_from_asof", spy_compute)

    with pytest.raises(ValueError, match="does not match its precomputed output plan"):
        analyze_listing(
            listing=listing,
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            issued_on_time=True,
            provider="twelve_data",
            store=store,
            config_path=scoring_path,
        )

    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert not DataAsset.objects.filter(kind=ANALYSIS_OUTPUT_MANIFEST_KIND).exists()
    manifest_files = list((tmp_path / "research" / "analysis").glob("**/output-manifest.json"))
    assert manifest_files == []


def test_analyze_listing_plan_matches_frozen_config_when_computation_agrees(
    tmp_path: Path,
) -> None:
    """Baseline: with no tampering, the same two-decision-horizon config
    persists both `short` and `medium` decision predictions for the single
    listing and writes exactly one manifest."""
    store = AssetStore(tmp_path)
    snapshot = _snapshot(slug=f"single-listing-plan-ok-{uuid4().hex[:8]}")
    listing = _long_listing("SLP2")
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    _write_price_asset(store, listing, close=40.0)
    scoring_path = _two_decision_horizon_config_path(tmp_path)

    persisted = analyze_listing(
        listing=listing,
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        issued_on_time=True,
        provider="twelve_data",
        store=store,
        config_path=scoring_path,
    )

    decision_horizons = {
        prediction.horizon
        for prediction in persisted.predictions
        if prediction.evidence_role == Prediction.EvidenceRole.DECISION
    }
    assert decision_horizons == {"short", "medium"}
    assert DataAsset.objects.filter(kind=ANALYSIS_OUTPUT_MANIFEST_KIND).count() == 1

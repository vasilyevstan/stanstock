"""Focused tests for `stanstock.data.refresh_validation` domain checks.

Builds on the real, minimal pipeline `test_refresh_verification._build_verified_state`
already assembles (market/evaluation/portfolio, optional SEC) rather than
re-implementing a second fixture harness, then exercises data-domain-only
behavior: the membership-evidence envelope, catalog `AssetRef` cross-binding,
ETF/membership exclusivity, `previous_close` proof, and SEC config binding.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stanstock.core.models import JobRun
from stanstock.core.verification_types import AssetRef, RefreshVerificationError
from stanstock.data.asof import verified_price_fields
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    read_checksummed_bytes,
    resolve_asset_ref,
)
from stanstock.data.models import DataAsset, UniverseSnapshot
from stanstock.data.refresh_evidence import lookup_membership_evidence
from stanstock.data.refresh_validation import (
    assert_no_etf_in_membership,
    require_bound_market_data,
    require_spy_listing,
    resolve_catalog_assets,
    verify_membership_evidence,
)
from stanstock.data.sec_refresh_validation import verify_sec_stage
from test_refresh_verification import DECISION_TIME, TARGET_DATE, _build_verified_state

pytestmark = pytest.mark.django_db


def _snapshot_and_market(stages: dict) -> tuple[UniverseSnapshot, JobRun]:
    market_run = JobRun.objects.get(pk=stages["market"]["job_run_id"])
    snapshot = UniverseSnapshot.objects.get(pk=market_run.details["snapshot_id"])
    return snapshot, market_run


def _evidence_asset_for(snapshot: UniverseSnapshot) -> DataAsset:
    lookup = lookup_membership_evidence(snapshot)
    assert lookup.asset is not None
    return lookup.asset


def test_membership_envelope_hash_separation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`DataAsset.sha256` hashes the whole envelope; `config_hash` hashes
    only the inner `hash_payload` -- the two must never be equal."""
    stages, _config = _build_verified_state(monkeypatch, tmp_path)
    snapshot, _market_run = _snapshot_and_market(stages)
    evidence_asset = _evidence_asset_for(snapshot)

    assert evidence_asset.sha256 != snapshot.config_hash
    payload = json.loads(AssetStore(tmp_path).read_bytes(evidence_asset.relative_path))
    assert payload["contract"] == "universe-membership-evidence@1"
    assert payload["snapshot_id"] == str(snapshot.id)
    assert isinstance(payload["hash_payload"], dict)
    assert isinstance(payload["catalog_refs"], list) and payload["catalog_refs"]


def test_legacy_snapshot_without_envelope_is_unverifiable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A snapshot created before the envelope existed (no membership
    evidence `DataAsset` at all) is named unverifiable, never synthesized."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    snapshot, market_run = _snapshot_and_market(stages)
    catalog_assets = list(
        DataAsset.objects.filter(
            pk__in=[uuid.UUID(v) for v in market_run.details["catalog_asset_ids"]]
        )
    )
    from stanstock.data.models import UniverseMembership

    legacy_snapshot = UniverseSnapshot.objects.create(
        universe=snapshot.universe,
        as_of_date=TARGET_DATE - timedelta(days=1),
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash=snapshot.config_hash,
    )
    for membership in UniverseMembership.objects.filter(snapshot=snapshot):
        UniverseMembership.objects.create(
            snapshot=legacy_snapshot,
            listing=membership.listing,
            eligible=membership.eligible,
            exclusion_reason=membership.exclusion_reason,
        )

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_membership_evidence(
            legacy_snapshot,
            list(
                UniverseMembership.objects.filter(snapshot=legacy_snapshot).select_related(
                    "listing"
                )
            ),
            universe_config=config,
            catalog_assets=catalog_assets,
            cutoff=DECISION_TIME,
        )
    assert excinfo.value.reason_code == "legacy_snapshot_unverifiable"


@pytest.mark.parametrize(
    "eligible,exclusion_reason",
    [(True, "should be blank"), (False, "")],
)
def test_membership_exclusion_reason_invariant_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, eligible: bool, exclusion_reason: str
) -> None:
    """Eligible => blank `exclusion_reason`; ineligible => non-blank. Either
    violation must fail closed rather than silently pass through."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    snapshot, market_run = _snapshot_and_market(stages)
    from stanstock.data.models import UniverseMembership

    memberships = list(
        UniverseMembership.objects.filter(snapshot=snapshot).select_related("listing")
    )
    UniverseMembership.objects.filter(pk=memberships[0].pk).update(
        eligible=eligible, exclusion_reason=exclusion_reason
    )
    memberships = list(
        UniverseMembership.objects.filter(snapshot=snapshot).select_related("listing")
    )
    catalog_assets = list(
        DataAsset.objects.filter(
            pk__in=[uuid.UUID(v) for v in market_run.details["catalog_asset_ids"]]
        )
    )
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_membership_evidence(
            snapshot,
            memberships,
            universe_config=config,
            catalog_assets=catalog_assets,
            cutoff=DECISION_TIME,
        )
    assert excinfo.value.reason_code == "membership_exclusion_reason_invalid"


def test_membership_catalog_ref_alternate_uuid_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A same-content catalog asset registered under a different UUID must
    fail the membership envelope's own `catalog_refs` cross-binding."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    snapshot, market_run = _snapshot_and_market(stages)
    catalog_assets = list(
        DataAsset.objects.filter(
            pk__in=[uuid.UUID(v) for v in market_run.details["catalog_asset_ids"]]
        )
    )
    original = catalog_assets[0]
    store = AssetStore(tmp_path)
    written = store.write_bytes("catalog/alt-copy.json", store.read_bytes(original.relative_path))
    alternate = DataAsset.objects.create(
        provider=original.provider,
        kind=original.kind,
        subject=original.subject,
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=original.retrieved_at,
        available_at=original.available_at,
    )
    swapped = [alternate, *catalog_assets[1:]]

    from stanstock.data.models import UniverseMembership

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_membership_evidence(
            snapshot,
            list(UniverseMembership.objects.filter(snapshot=snapshot).select_related("listing")),
            universe_config=config,
            catalog_assets=swapped,
            cutoff=DECISION_TIME,
        )
    assert excinfo.value.reason_code in {
        "snapshot_config_hash_mismatch",
        "membership_catalog_ref_mismatch",
    }


def test_market_catalog_ref_alternate_uuid_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    market_run = JobRun.objects.get(pk=stages["market"]["job_run_id"])
    details = dict(market_run.details)
    refs = list(details["catalog_refs"])
    tampered = dict(refs[0])
    tampered["id"] = str(uuid.uuid4())
    details["catalog_refs"] = [tampered, *refs[1:]]
    JobRun.objects.filter(pk=market_run.pk).update(details=details)
    market_run.refresh_from_db()

    with pytest.raises(RefreshVerificationError) as excinfo:
        resolve_catalog_assets(
            market_run.details,
            universe_config=config,
            cutoff=DECISION_TIME,
        )
    assert excinfo.value.reason_code == "catalog_ref_market_mismatch"


def test_membership_contains_investable_etf_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, _config = _build_verified_state(monkeypatch, tmp_path)
    snapshot, _market_run = _snapshot_and_market(stages)

    from stanstock.data.etfs import INVESTABLE_US_ETF_MIC, INVESTABLE_US_ETF_SYMBOL
    from stanstock.data.models import UniverseMembership

    spy_listing = require_spy_listing()
    assert spy_listing.ticker == INVESTABLE_US_ETF_SYMBOL
    assert spy_listing.exchange_mic == INVESTABLE_US_ETF_MIC
    UniverseMembership.objects.create(
        snapshot=snapshot, listing=spy_listing, eligible=True, exclusion_reason=""
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        assert_no_etf_in_membership(
            list(UniverseMembership.objects.filter(snapshot=snapshot).select_related("listing"))
        )
    assert excinfo.value.reason_code == "membership_contains_investable_etf"


def test_previous_close_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    from stanstock.data.models import LatestMarketData, Listing

    listing = Listing.objects.get(provider_symbol="AAA")
    latest = LatestMarketData.objects.get(listing=listing)
    LatestMarketData.objects.filter(pk=latest.pk).update(
        previous_close=(latest.previous_close or latest.close) + 1
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        require_bound_market_data(
            listing,
            target_date=TARGET_DATE,
            cutoff=DECISION_TIME + timedelta(days=1),
            expected_asset_id=latest.source_asset_id,
            expected_subject=listing.provider_symbol,
        )
    assert excinfo.value.reason_code == "latest_market_data_previous_close_mismatch"


@pytest.mark.parametrize("listing_symbol", ["AAA", "SPY"])
def test_observed_at_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, listing_symbol: str
) -> None:
    """`observed_at` must equal the bound asset's own `retrieved_at`, for
    both a plain stock listing and the SPY benchmark listing."""
    stages, _config = _build_verified_state(monkeypatch, tmp_path)
    from stanstock.data.models import LatestMarketData, Listing

    listing = Listing.objects.get(provider_symbol=listing_symbol)
    latest = LatestMarketData.objects.get(listing=listing)
    LatestMarketData.objects.filter(pk=latest.pk).update(
        observed_at=latest.observed_at + timedelta(seconds=1)
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        require_bound_market_data(
            listing,
            target_date=TARGET_DATE,
            cutoff=DECISION_TIME + timedelta(days=1),
            expected_asset_id=latest.source_asset_id,
            expected_subject=listing.provider_symbol,
        )
    assert excinfo.value.reason_code == "latest_market_data_observed_at_invalid"


@pytest.mark.parametrize("listing_symbol", ["AAA", "SPY"])
def test_observed_at_after_cutoff_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, listing_symbol: str
) -> None:
    """A matched-but-future `observed_at`/`retrieved_at` must still fail the
    claimed cutoff, for both a plain stock listing and SPY."""
    stages, _config = _build_verified_state(monkeypatch, tmp_path)
    from stanstock.data.models import LatestMarketData, Listing

    listing = Listing.objects.get(provider_symbol=listing_symbol)
    latest = LatestMarketData.objects.select_related("source_asset").get(listing=listing)
    original = latest.source_asset
    future = DECISION_TIME + timedelta(days=365)
    store = AssetStore(tmp_path)
    written = store.write_bytes(
        f"substitute/future-{listing_symbol}.bin", store.read_bytes(original.relative_path)
    )
    future_asset = DataAsset.objects.create(
        provider=original.provider,
        kind=original.kind,
        subject=original.subject,
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=future,
        available_at=future,
    )
    LatestMarketData.objects.filter(pk=latest.pk).update(
        observed_at=future, source_asset=future_asset
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        require_bound_market_data(
            listing,
            target_date=TARGET_DATE,
            cutoff=DECISION_TIME,
            expected_asset_id=future_asset.id,
            expected_subject=listing.provider_symbol,
        )
    assert excinfo.value.reason_code == "latest_market_data_observed_at_invalid"


def test_sec_fundamentals_config_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, _config = _build_verified_state(monkeypatch, tmp_path, sec=True)
    sec_run = JobRun.objects.get(pk=stages["sec_fundamentals"]["job_run_id"])
    details = dict(sec_run.details)
    details["config_hash"] = "0" * 64
    JobRun.objects.filter(pk=sec_run.pk).update(details=details)
    sec_run.refresh_from_db()

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_sec_stage(sec_run.details, sec_run=sec_run)
    assert excinfo.value.reason_code == "sec_fundamentals_config_mismatch"


def test_resolve_asset_ref_missing_and_after_cutoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, _config = _build_verified_state(monkeypatch, tmp_path)
    snapshot, _market_run = _snapshot_and_market(stages)
    evidence_asset = _evidence_asset_for(snapshot)
    real_ref = AssetRef(
        id=evidence_asset.id,
        provider=evidence_asset.provider,
        kind=evidence_asset.kind,
        subject=evidence_asset.subject,
        sha256=evidence_asset.sha256,
    )

    missing_ref = AssetRef(
        id=uuid.uuid4(),
        provider=real_ref.provider,
        kind=real_ref.kind,
        subject=real_ref.subject,
        sha256=real_ref.sha256,
    )
    with pytest.raises(RefreshVerificationError) as excinfo:
        resolve_asset_ref(missing_ref, cutoff=DECISION_TIME)
    assert excinfo.value.reason_code == "asset_ref_unresolved"

    with pytest.raises(RefreshVerificationError) as excinfo:
        resolve_asset_ref(real_ref, cutoff=evidence_asset.retrieved_at - timedelta(seconds=1))
    assert excinfo.value.reason_code == "asset_ref_after_cutoff"


def test_read_checksummed_bytes_unreadable_and_corrupt_are_path_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, _config = _build_verified_state(monkeypatch, tmp_path)
    snapshot, _market_run = _snapshot_and_market(stages)
    evidence_asset = _evidence_asset_for(snapshot)
    store = AssetStore(tmp_path)

    store.resolve(evidence_asset.relative_path).unlink()
    with pytest.raises(RefreshVerificationError) as excinfo:
        read_checksummed_bytes(store, evidence_asset)
    assert excinfo.value.reason_code == "asset_unreadable"
    assert str(tmp_path) not in str(excinfo.value)
    assert excinfo.value.__cause__ is None

    store.resolve(evidence_asset.relative_path).write_bytes(b"corrupted")
    with pytest.raises(RefreshVerificationError) as excinfo:
        read_checksummed_bytes(store, evidence_asset)
    assert excinfo.value.reason_code == "asset_corrupt"
    assert str(tmp_path) not in str(excinfo.value)


def test_read_checksummed_bytes_hash_failure_is_path_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hashing failure (not just an unreadable file) must also be
    normalized path-free -- `read_checksummed_bytes` must not merely wrap
    `store.read_bytes` errors and let a `hashlib` failure surface raw."""
    stages, _config = _build_verified_state(monkeypatch, tmp_path)
    snapshot, _market_run = _snapshot_and_market(stages)
    evidence_asset = _evidence_asset_for(snapshot)
    store = AssetStore(tmp_path)

    def unhashable_read_bytes(self: AssetStore, relative_path: str) -> bytes:
        # Intentionally not `bytes` -- `hashlib.sha256` raises `TypeError`
        # on a plain `str`, exercising the digest-computation failure branch
        # distinctly from the file-read failure branch above.
        return "not-bytes"  # type: ignore[return-value]

    monkeypatch.setattr(AssetStore, "read_bytes", unhashable_read_bytes)
    with pytest.raises(RefreshVerificationError) as excinfo:
        read_checksummed_bytes(store, evidence_asset)
    assert excinfo.value.reason_code == "asset_checksum_failed"
    assert str(tmp_path) not in str(excinfo.value)
    assert evidence_asset.relative_path not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_require_spy_listing_requires_supported_investable_etf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no pipeline-registered SPY at all, a same-ticker but
    non-conforming listing (wrong exchange) must not satisfy the check."""
    from stanstock.data.models import Company, Listing, Region, Security

    with pytest.raises(RefreshVerificationError) as excinfo:
        require_spy_listing()
    assert excinfo.value.reason_code == "spy_listing_unsupported"

    company = Company.objects.create(name="Wrong SPY", country="US")
    security = Security.objects.create(company=company)
    Listing.objects.create(
        security=security,
        ticker="SPY",
        provider_symbol="SPY",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        require_spy_listing()
    assert excinfo.value.reason_code == "spy_listing_unsupported"


# --- Correction 1: chained exceptions must never leak a resolved/relative
# filesystem path, even through a caller that logs `__cause__` -------------


def test_verified_price_fields_schema_error_is_path_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A price frame missing its `date` column raises a path-bearing
    `PriceFrameSchemaError` internally; `verified_price_fields` must sever
    that chain (`from None`) so neither the exception nor a logger that
    prints `__cause__`'s traceback ever exposes the sentinel path."""
    import logging

    from django.conf import settings
    from django.utils import timezone

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    store = AssetStore(tmp_path)
    stored = store.write_frame("prices/broken.parquet", pl.DataFrame({"close": [1.0]}))
    now = timezone.now()
    asset = DataAsset.objects.create(
        provider="twelve_data",
        kind="price_history",
        subject="AAA",
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        retrieved_at=now,
        available_at=now,
    )
    logger = logging.getLogger("test-path-free-verified-price-fields")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            verified_price_fields(asset, cutoff=now, target_date=date(2026, 9, 4), close_places=2)
        except RefreshVerificationError as exc:
            assert exc.reason_code == "latest_market_data_asset_unreadable"
            assert exc.__cause__ is None
            assert str(tmp_path) not in str(exc)
            logger.exception("verification failed")
        else:
            pytest.fail("expected RefreshVerificationError")
    assert caplog.text
    assert str(tmp_path) not in caplog.text


def _malformed_price_frame(kind: str, target: date) -> pl.DataFrame:
    """Build one deliberately non-conforming price frame for `kind`."""
    base: dict[str, Any] = {"date": [target], "close": [100.0], "volume": [10]}
    overrides: dict[str, pl.DataType] = {"date": pl.Date, "volume": pl.Int64}
    if kind == "missing_close":
        del base["close"]
    elif kind == "missing_volume":
        del base["volume"]
    elif kind == "close_wrong_dtype":
        base["close"] = ["100.0"]
        overrides["close"] = pl.Utf8
    elif kind == "close_nan":
        base["close"] = [float("nan")]
    elif kind == "close_inf":
        base["close"] = [float("inf")]
    elif kind == "close_non_positive":
        base["close"] = [-1.0]
    elif kind == "volume_fractional":
        base["volume"] = [10.5]
        overrides["volume"] = pl.Float64
    elif kind == "volume_bool":
        base["volume"] = [True]
        overrides["volume"] = pl.Boolean
    elif kind == "volume_negative":
        base["volume"] = [-5]
    elif kind == "duplicate_date":
        base = {"date": [target, target], "close": [100.0, 101.0], "volume": [10, 11]}
    elif kind == "previous_close_null":
        base = {
            "date": [target - timedelta(days=1), target],
            "close": [None, 100.0],
            "volume": [10, 10],
        }
    else:
        raise AssertionError(kind)
    return pl.DataFrame(base, schema_overrides=overrides)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_close",
        "missing_volume",
        "close_wrong_dtype",
        "close_nan",
        "close_inf",
        "close_non_positive",
        "volume_fractional",
        "volume_bool",
        "volume_negative",
        "duplicate_date",
        "previous_close_null",
    ],
)
def test_verified_price_fields_rejects_malformed_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    """`verified_price_fields` must validate schema/dtype/finiteness/sign
    before ever extracting `close`/`volume`, rather than crashing on
    extraction or silently accepting an out-of-contract value."""
    from django.conf import settings
    from django.utils import timezone

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    target = date(2026, 9, 4)
    store = AssetStore(tmp_path)
    stored = store.write_frame("prices/malformed.parquet", _malformed_price_frame(kind, target))

    now = timezone.now()
    asset = DataAsset.objects.create(
        provider="twelve_data",
        kind="price_history",
        subject="AAA",
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        retrieved_at=now,
        available_at=now,
    )
    with pytest.raises(RefreshVerificationError) as excinfo:
        verified_price_fields(asset, cutoff=now, target_date=target, close_places=2)
    assert excinfo.value.reason_code == "latest_market_data_asset_unreadable"
    assert excinfo.value.__cause__ is None


# --- Correction 3: an ambiguous (duplicate current-contract) evidence
# asset must not be misclassified as a legacy/unverifiable snapshot --------


def test_membership_evidence_ambiguous_is_distinct_from_legacy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    snapshot, _market_run = _snapshot_and_market(stages)
    from stanstock.data.models import UniverseMembership
    from stanstock.data.refresh_evidence import UNIVERSE_MEMBERSHIP_EVIDENCE_KIND

    original = _evidence_asset_for(snapshot)
    store = AssetStore(tmp_path)
    written = store.write_bytes(
        "universe/duplicate-evidence.json", store.read_bytes(original.relative_path)
    )
    DataAsset.objects.create(
        provider="stanstock",
        kind=UNIVERSE_MEMBERSHIP_EVIDENCE_KIND,
        subject=str(snapshot.id),
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=original.retrieved_at,
        available_at=original.available_at,
    )
    lookup = lookup_membership_evidence(snapshot)
    assert lookup.asset is None
    assert lookup.count == 2

    memberships = list(
        UniverseMembership.objects.filter(snapshot=snapshot).select_related("listing")
    )
    catalog_assets = _catalog_assets_for(stages)
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_membership_evidence(
            snapshot,
            memberships,
            universe_config=config,
            catalog_assets=catalog_assets,
            cutoff=DECISION_TIME,
        )
    assert excinfo.value.reason_code == "membership_evidence_ambiguous"


def test_membership_evidence_after_cutoff_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A membership evidence asset admitted after the claiming cutoff must
    fail closed -- a late envelope must never be treated as on-time."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    snapshot, _market_run = _snapshot_and_market(stages)
    from stanstock.data.models import UniverseMembership
    from stanstock.data.refresh_evidence import UNIVERSE_MEMBERSHIP_EVIDENCE_KIND

    original_evidence = _evidence_asset_for(snapshot)
    future_snapshot = UniverseSnapshot.objects.create(
        universe=snapshot.universe,
        as_of_date=snapshot.as_of_date - timedelta(days=1),
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash=snapshot.config_hash,
    )
    for membership in UniverseMembership.objects.filter(snapshot=snapshot):
        UniverseMembership.objects.create(
            snapshot=future_snapshot,
            listing=membership.listing,
            eligible=membership.eligible,
            exclusion_reason=membership.exclusion_reason,
        )
    future = DECISION_TIME + timedelta(days=365)
    store = AssetStore(tmp_path)
    written = store.write_bytes(
        "universe/future-evidence.json", store.read_bytes(original_evidence.relative_path)
    )
    DataAsset.objects.create(
        provider="stanstock",
        kind=UNIVERSE_MEMBERSHIP_EVIDENCE_KIND,
        subject=str(future_snapshot.id),
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=future,
        available_at=future,
    )

    memberships = list(
        UniverseMembership.objects.filter(snapshot=future_snapshot).select_related("listing")
    )
    catalog_assets = _catalog_assets_for(stages)
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_membership_evidence(
            future_snapshot,
            memberships,
            universe_config=config,
            catalog_assets=catalog_assets,
            cutoff=DECISION_TIME,
        )
    assert excinfo.value.reason_code == "membership_evidence_after_cutoff"


def test_verify_membership_evidence_normalizes_asset_store_constructor_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A store-level failure (an unwritable/misconfigured root) must not leak
    a path-bearing `OSError`/`ValueError` out of the membership validator."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    from stanstock.data.models import UniverseMembership

    snapshot, _market_run = _snapshot_and_market(stages)
    memberships = list(
        UniverseMembership.objects.filter(snapshot=snapshot).select_related("listing")
    )
    catalog_assets = _catalog_assets_for(stages)

    sentinel = tmp_path / "sentinel-unwritable-root"

    def fail_mkdir(*args: object, **kwargs: object) -> None:
        raise OSError(f"Permission denied: {sentinel}")

    monkeypatch.setattr("django.conf.settings.DATA_DIR", sentinel)
    monkeypatch.setattr(Path, "mkdir", fail_mkdir)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_membership_evidence(
            snapshot,
            memberships,
            universe_config=config,
            catalog_assets=catalog_assets,
            cutoff=DECISION_TIME,
        )
    assert excinfo.value.reason_code == "asset_store_unavailable"
    assert excinfo.value.__cause__ is None
    assert str(sentinel) not in str(excinfo.value)


def _fresh_snapshot(snapshot: UniverseSnapshot, *, day_offset: int) -> UniverseSnapshot:
    """Clone `snapshot`'s config_hash/memberships onto a fresh snapshot row,
    so a test can attach a deliberately malformed evidence asset without
    ever mutating an already-persisted (immutable) `DataAsset` row."""
    from stanstock.data.models import UniverseMembership

    fresh = UniverseSnapshot.objects.create(
        universe=snapshot.universe,
        as_of_date=snapshot.as_of_date - timedelta(days=day_offset),
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash=snapshot.config_hash,
    )
    for membership in UniverseMembership.objects.filter(snapshot=snapshot):
        UniverseMembership.objects.create(
            snapshot=fresh,
            listing=membership.listing,
            eligible=membership.eligible,
            exclusion_reason=membership.exclusion_reason,
        )
    return fresh


def _attach_evidence_bytes(snapshot: UniverseSnapshot, payload: bytes, relative_path: str) -> None:
    from stanstock.data.refresh_evidence import UNIVERSE_MEMBERSHIP_EVIDENCE_KIND

    written = AssetStore().write_bytes(relative_path, payload)
    DataAsset.objects.create(
        provider="stanstock",
        kind=UNIVERSE_MEMBERSHIP_EVIDENCE_KIND,
        subject=str(snapshot.id),
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=DECISION_TIME,
        available_at=DECISION_TIME,
    )


@pytest.mark.parametrize(
    "kind,expected_reason",
    [
        ("extra_key", "membership_evidence_envelope_invalid"),
        ("missing_key", "membership_evidence_envelope_invalid"),
        ("duplicate_json_key", "membership_evidence_asset_malformed"),
    ],
)
def test_membership_evidence_envelope_shape_violations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str, expected_reason: str
) -> None:
    """The envelope's top-level key set is exactly `contract, snapshot_id,
    hash_payload, catalog_refs`; an extra/missing key fails that exact-set
    check, and a duplicate top-level JSON key is rejected by the strict
    parser before the shape check ever runs."""
    stages, config = _build_verified_state(monkeypatch, tmp_path)
    snapshot, _market_run = _snapshot_and_market(stages)
    from stanstock.data.models import UniverseMembership

    fresh = _fresh_snapshot(snapshot, day_offset=90)
    original_evidence = _evidence_asset_for(snapshot)
    original_payload = AssetStore(tmp_path).read_bytes(original_evidence.relative_path).decode()
    retargeted = original_payload.replace(str(snapshot.id), str(fresh.id))
    if kind == "extra_key":
        text = retargeted[:-1] + ', "extra": 1}'
    elif kind == "missing_key":
        envelope = json.loads(retargeted)
        del envelope["catalog_refs"]
        text = json.dumps(envelope)
    else:
        text = retargeted[:-1] + ', "contract": "tampered"}'
    _attach_evidence_bytes(fresh, text.encode(), f"universe/{kind}.json")

    memberships = list(UniverseMembership.objects.filter(snapshot=fresh).select_related("listing"))
    catalog_assets = _catalog_assets_for(stages)
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_membership_evidence(
            fresh,
            memberships,
            universe_config=config,
            catalog_assets=catalog_assets,
            cutoff=DECISION_TIME + timedelta(days=365),
        )
    assert excinfo.value.reason_code == expected_reason


def _catalog_assets_for(stages: dict) -> list[DataAsset]:
    market_run = JobRun.objects.get(pk=stages["market"]["job_run_id"])
    return list(
        DataAsset.objects.filter(
            pk__in=[uuid.UUID(v) for v in market_run.details["catalog_asset_ids"]]
        )
    )


# --- Correction 4: the SEC data-stage's exact asset closure ---------------


def test_sec_stage_requires_asset_refs_and_unique_mapping_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, _config = _build_verified_state(monkeypatch, tmp_path, sec=True)
    sec_run = JobRun.objects.get(pk=stages["sec_fundamentals"]["job_run_id"])

    missing = dict(sec_run.details)
    del missing["asset_refs"]
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_sec_stage(missing, sec_run=sec_run)
    assert excinfo.value.reason_code == "sec_asset_refs_missing"

    malformed = dict(sec_run.details)
    malformed["asset_refs"] = [{"id": "not-a-uuid"}]
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_sec_stage(malformed, sec_run=sec_run)
    assert excinfo.value.reason_code == "sec_asset_ref_malformed"

    duplicated = dict(sec_run.details)
    duplicated["asset_refs"] = [*duplicated["asset_refs"], duplicated["asset_refs"][0]]
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_sec_stage(duplicated, sec_run=sec_run)
    assert excinfo.value.reason_code == "sec_asset_ref_duplicate"

    mapping_missing = dict(sec_run.details)
    mapping_missing["asset_refs"] = [
        ref for ref in mapping_missing["asset_refs"] if ref["kind"] != "sec_ticker_mapping"
    ]
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_sec_stage(mapping_missing, sec_run=sec_run)
    assert excinfo.value.reason_code == "sec_mapping_ref_not_unique"


def test_sec_stage_rejects_wrong_provider_or_kind_asset_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, _config = _build_verified_state(monkeypatch, tmp_path, sec=True)
    sec_run = JobRun.objects.get(pk=stages["sec_fundamentals"]["job_run_id"])
    mapping_asset = DataAsset.objects.get(pk=sec_run.details["mapping_asset_id"])

    store = AssetStore(tmp_path)
    written = store.write_bytes("sec/foreign.json", b'{"foreign": true}')
    foreign_asset = DataAsset.objects.create(
        provider="twelve_data",
        kind="sec_submissions",
        subject="0000000001",
        relative_path=written.relative_path,
        sha256=written.sha256,
        retrieved_at=mapping_asset.retrieved_at,
        available_at=mapping_asset.available_at,
    )
    tampered = dict(sec_run.details)
    tampered["asset_refs"] = [*tampered["asset_refs"], asset_ref_for(foreign_asset).to_json()]
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_sec_stage(tampered, sec_run=sec_run)
    assert excinfo.value.reason_code == "sec_asset_ref_identity_mismatch"


def test_sec_stage_returns_every_resolved_asset_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stages, _config = _build_verified_state(monkeypatch, tmp_path, sec=True)
    sec_run = JobRun.objects.get(pk=stages["sec_fundamentals"]["job_run_id"])

    result = verify_sec_stage(sec_run.details, sec_run=sec_run)
    assert len(result.asset_refs) == len(sec_run.details["asset_refs"]) == 3
    assert result.summary["asset_count"] == 3

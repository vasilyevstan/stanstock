from __future__ import annotations

import json

import pytest

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.asof import AsOfData
from stanstock.data.assets import read_checksummed_bytes
from stanstock.data.liquidity_evidence import (
    BATCH_SCHEMA,
    HISTORY_KIND,
    RAW_KIND,
    verify_liquidity_batch,
)
from stanstock.data.models import DataAsset, LatestMarketData

pytestmark = pytest.mark.django_db


def test_real_batch_reopens_raw_normalized_control_and_every_captured_candidate(c2_closed_batch):
    env = c2_closed_batch
    proof = verify_liquidity_batch(
        asset=env.batch, intake=env.intake, cutoff=env.now, store=env.store
    )
    assert env.batch.schema_version == BATCH_SCHEMA
    assert proof.document["witness"]["status"] == "matched"
    assert len(proof.document["entries"]) == len(env.intake.requested_symbols) == 100
    assert proof.assessments["AAPL"].status == "matched"
    assert proof.assessments["AAPL"].average_dollar_turnover_20d > 5_000_000
    assert DataAsset.objects.filter(kind=HISTORY_KIND).count() == 2
    assert DataAsset.objects.filter(kind=RAW_KIND).count() == 2
    assert len(env.fetch.call_args_list) == 4
    assert not LatestMarketData.objects.exists()
    none_calls = [call for call in env.fetch.call_args_list if call.kwargs["adjustment"] == "none"]
    assert [call.args[0] for call in none_calls] == ["NVDA", "AAPL"]
    assert none_calls[0].kwargs["start_date"].isoformat() == "2024-06-03"
    assert all(call.kwargs["outputsize"] == 5000 for call in none_calls)


def test_newest_none_cannot_shadow_generic_price_selection(c2_closed_batch):
    env = c2_closed_batch
    selected = AsOfData(env.now, env.store).latest_asset(
        provider="twelve_data", kind="price_history", subject="AAPL"
    )
    assert selected.id == env.selected["AAPL"].id
    assert selected.kind != HISTORY_KIND
    assert all(
        asset.kind in {"price_history", "raw_price_history", HISTORY_KIND, RAW_KIND}
        for asset in verify_liquidity_batch(
            asset=env.batch,
            intake=env.intake,
            cutoff=env.now,
            store=env.store,
        ).sources
    )


@pytest.mark.parametrize("kind", [RAW_KIND, HISTORY_KIND, "liquidity_run_evidence"])
def test_physical_corruption_is_integrity_failure_not_unavailable(c2_closed_batch, kind):
    env = c2_closed_batch
    source = DataAsset.objects.filter(kind=kind).first()
    env.store.resolve(source.relative_path).write_bytes(b"synthetic corruption")
    with pytest.raises(RefreshVerificationError):
        verify_liquidity_batch(asset=env.batch, intake=env.intake, cutoff=env.now, store=env.store)


def test_wrong_attempt_receipts_do_not_authenticate_an_otherwise_matching_batch(c2_closed_batch):
    env = c2_closed_batch
    env.job.details["liquidity_requests"][0]["mode"] = "none"
    env.job.save(update_fields=["details"])
    with pytest.raises(RefreshVerificationError):
        verify_liquidity_batch(asset=env.batch, intake=env.intake, cutoff=env.now, store=env.store)


def test_checksum_matching_malformed_closed_proof_is_not_a_hold(c2_closed_batch):
    env = c2_closed_batch
    document = json.loads(read_checksummed_bytes(env.store, env.batch))
    document["entries"][0]["assessment"] = {"status": "matched"}
    encoded = json.dumps(document).encode()
    stored = env.store.write_bytes("synthetic/forged-closed.json", encoded)
    forged = DataAsset.objects.create(
        provider=env.batch.provider,
        kind=env.batch.kind,
        subject=env.batch.subject,
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        schema_version=env.batch.schema_version,
        metadata=env.batch.metadata,
        retrieved_at=env.now,
        available_at=env.now,
    )
    with pytest.raises(RefreshVerificationError):
        verify_liquidity_batch(asset=forged, intake=env.intake, cutoff=env.now, store=env.store)


def test_normalized_ohlc_correspondence_is_checked_even_with_matching_registered_checksum(
    c2_closed_batch,
):
    import io

    import polars as pl

    from stanstock.data.liquidity_evidence import _read_none

    env = c2_closed_batch
    source = DataAsset.objects.get(kind=HISTORY_KIND, subject="AAPL")
    frame = pl.read_parquet(io.BytesIO(read_checksummed_bytes(env.store, source)))
    corrupted = frame.with_columns(pl.lit("1").alias("low"))
    stored = env.store.write_frame("synthetic/wrong-full-ohlc.parquet", corrupted)
    forged = DataAsset.objects.create(
        provider=source.provider,
        kind=source.kind,
        subject=source.subject,
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        metadata=source.metadata,
        schema_version=source.schema_version,
        period_start=source.period_start,
        period_end=source.period_end,
        available_at=env.now,
        retrieved_at=env.now,
    )
    with pytest.raises(ValueError, match="full public raw OHLC"):
        _read_none(
            asset=forged,
            listing=env.listings["AAPL"],
            intake=env.intake,
            job=env.job,
            start=source.period_start,
            target=env.target,
            publication=env.now,
            store=env.store,
        )


@pytest.mark.parametrize("dimension", ["start", "attempt", "region", "owner"])
def test_source_batch_independently_rejects_attempt_scope_dimensions(c2_closed_batch, dimension):
    env = c2_closed_batch
    if dimension == "start":
        env.job.details["liquidity_requests"][0]["start"] = "2020-01-01"
        env.job.save(update_fields=["details"])
    elif dimension == "owner":
        env.job.job_name = "research_intake_v2:incorrect-native-owner-identity"
        env.job.save(update_fields=["job_name"])
    else:
        setattr(env.job, dimension, 9 if dimension == "attempt" else "eu")
        env.job.save(update_fields=[dimension])
    with pytest.raises(RefreshVerificationError):
        verify_liquidity_batch(asset=env.batch, intake=env.intake, cutoff=env.now, store=env.store)


@pytest.mark.parametrize("c2_environment", [{"late_none": True}], indirect=True)
def test_late_raw_and_normalized_clocks_are_recorded_but_values_withheld_at_own_cutoff(
    c2_closed_batch,
):
    env = c2_closed_batch
    proof = verify_liquidity_batch(
        asset=env.batch,
        intake=env.intake,
        cutoff=env.clock["now"],
        store=env.store,
    )
    apple = next(row for row in proof.document["entries"] if row["symbol"] == "AAPL")
    assert apple["split_values"] is apple["none_values"] is None
    assert apple["split_raw_asset"] is not None and apple["none_raw_asset"] is not None
    assert len(apple["source_clocks"]) == 4
    assert "liquidity_cutoff_unsafe" in proof.assessments["AAPL"].reasons
    assert proof.assessments["AAPL"].average_dollar_turnover_20d is None
    assert proof.assessments["NVDA"].status == "matched"

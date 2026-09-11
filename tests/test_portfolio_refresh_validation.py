from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Context, Decimal, localcontext
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

import polars as pl
import pytest
from django import VERSION as DJANGO_VERSION
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import DatabaseError, IntegrityError, connection, connections, transaction
from django.db.backends.base.base import BaseDatabaseWrapper
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from stanstock.core.models import JobRun
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.assets import AssetStore, asset_ref_for
from stanstock.data.models import (
    Company,
    DataAsset,
    LatestMarketData,
    Listing,
    Region,
    Security,
)
from stanstock.data.provider_policy import PRIVATE_USAGE_SCOPE
from stanstock.portfolio import jobs as portfolio_jobs
from stanstock.portfolio import refresh_validation
from stanstock.portfolio.jobs import execute_portfolio_snapshot_job
from stanstock.portfolio.models import (
    Portfolio,
    PortfolioHolding,
    PortfolioSnapshot,
    PortfolioSnapshotHolding,
)
from stanstock.portfolio.refresh_validation import (
    BASE_DETAIL_KEYS,
    PORTFOLIO_SNAPSHOT_FIELDS,
    PORTFOLIO_SNAPSHOT_HOLDING_FIELDS,
    PORTFOLIO_SNAPSHOT_HOLDING_MODEL,
    PORTFOLIO_SNAPSHOT_MODEL,
    PORTFOLIO_VERIFICATION_KIND,
    PORTFOLIO_VERIFICATION_PROVIDER,
    PORTFOLIO_VERIFICATION_SCHEMA,
    VERIFIED_DETAIL_KEYS,
    PortfolioVerificationPayloadError,
    PortfolioVerificationProof,
    PositionProof,
    SnapshotProof,
    _matches_supported_persistence,
    _supported_persisted_decimals,
    attest_scheduled_portfolio_snapshots,
    canonical_json_bytes,
    canonical_row_bytes,
    model_row_values,
    parse_portfolio_verification_proof,
    report_sha256,
    row_digest,
    snapshot_holding_row_digest,
    snapshot_row_digest,
    verify_portfolio_snapshot_stage,
)
from stanstock.portfolio.service import (
    PortfolioSnapshotBatch,
    calculate_portfolio_valuation,
    compute_snapshot_input_hash,
    record_portfolio_snapshot,
    snapshot_all_portfolios,
    upsert_holding,
)

pytestmark = pytest.mark.django_db

TARGET = date(2026, 9, 4)
REVISION = "verification-rev-2"


def _wait_for_postgresql_lock(
    backend_pid: int,
    *,
    locktype: str,
    mode: str,
    granted: bool,
    relation: str | None = None,
    timeout: float = 10,
) -> None:
    """Wait until PostgreSQL reports one exact lock state for a test backend."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        query = """
            SELECT EXISTS (
                SELECT 1
                FROM pg_locks
                WHERE pid = %s
                  AND locktype = %s
                  AND mode = %s
                  AND granted = %s
        """
        params: list[object] = [backend_pid, locktype, mode, granted]
        if relation is not None:
            query += " AND relation = to_regclass(%s)"
            params.append(relation)
        query += ")"
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()
        if row is not None and row[0] is True:
            return
        time.sleep(0.02)
    pytest.fail(
        f"PostgreSQL backend {backend_pid} did not reach {locktype}/{mode}/granted={granted}"
    )


@dataclass(frozen=True)
class ProofState:
    portfolio: Portfolio
    listings: tuple[Listing, ...]
    normalized_assets: tuple[DataAsset, ...]
    raw_assets: tuple[DataAsset, ...]
    child: JobRun


def _owner(username: str = "proof-owner") -> Any:
    return get_user_model().objects.create_user(username=username)


def _add_priced_holding(
    *,
    tmp_path: Path,
    portfolio: Portfolio,
    ticker: str,
    currency: str,
    session_date: date,
    provider: str = "twelve_data",
    metadata_currency: str | None = None,
    include_raw: bool | None = None,
    partial_raw: bool = False,
    quantity: Decimal = Decimal("7.12345678"),
    average_cost: Decimal = Decimal("45.678901"),
    price: Decimal = Decimal("103.125000"),
    normalized_observed_at: datetime | None = None,
    raw_observed_at: datetime | None = None,
) -> tuple[Listing, DataAsset, DataAsset | None]:
    company = Company.objects.create(name=f"{ticker} Synthetic Co", country="US")
    security = Security.objects.create(company=company)
    listing = Listing.objects.create(
        security=security,
        ticker=ticker,
        provider_symbol=f"{ticker}:PROVIDER",
        exchange_mic="XNAS",
        currency=currency,
        region=Region.US,
    )
    store = AssetStore(tmp_path)
    observed_at = normalized_observed_at or timezone.now() - timedelta(minutes=2)
    raw_time = raw_observed_at or observed_at
    raw_asset = None
    metadata: dict[str, object] = {
        "return_definition": "split_adjusted_price_return",
        "dividends_included": False,
    }
    if metadata_currency is not None:
        metadata["currency"] = metadata_currency
    include_raw = (
        provider not in {"synthetic", "synthetic_demo"} if include_raw is None else include_raw
    )
    if include_raw:
        raw_payload = canonical_json_bytes({"synthetic_fixture": ticker})
        stored_raw = store.write_bytes(
            f"raw/{provider}/{ticker}-{uuid4()}.json",
            raw_payload,
        )
        raw_asset = DataAsset.objects.create(
            provider=provider,
            kind="raw_price_history",
            subject=listing.provider_symbol,
            relative_path=stored_raw.relative_path,
            sha256=stored_raw.sha256,
            retrieved_at=raw_time,
            available_at=raw_time,
            period_start=session_date,
            period_end=session_date,
        )
        metadata["raw_asset_id"] = str(raw_asset.pk)
        metadata["raw_sha256"] = raw_asset.sha256
    elif partial_raw:
        metadata["raw_asset_id"] = str(uuid4())
    previous_date = session_date - timedelta(days=1)
    frame = pl.DataFrame(
        {
            "date": [previous_date, session_date],
            "close": [float(price - Decimal("1")), float(price)],
            "volume": [900_000, 1_000_000],
        },
        schema_overrides={"date": pl.Date, "volume": pl.Int64},
    )
    stored = store.write_frame(
        f"price_history/{provider}/{ticker}-{uuid4()}.parquet",
        frame,
    )
    normalized = DataAsset.objects.create(
        provider=provider,
        kind="price_history",
        subject=listing.provider_symbol,
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        retrieved_at=observed_at,
        available_at=observed_at,
        period_start=previous_date,
        period_end=session_date,
        metadata=metadata,
    )
    LatestMarketData.objects.create(
        listing=listing,
        observed_at=observed_at,
        session_date=session_date,
        close=price,
        previous_close=price - Decimal("1"),
        volume=1_000_000,
        source_asset=normalized,
    )
    upsert_holding(
        portfolio=portfolio,
        listing=listing,
        quantity=quantity,
        average_cost=average_cost,
        notes=f"{ticker} PRIVATE NOTE",
    )
    return listing, normalized, raw_asset


def _build_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    currency: str = "USD",
    sessions: tuple[date, ...] = (TARGET,),
    provider: str = "twelve_data",
    metadata_currency: str | None = None,
    cash: Decimal = Decimal("12345.670001"),
) -> ProofState:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(f"owner-{uuid4()}"),
        name="PRIVATE PORTFOLIO NAME",
        description="PRIVATE PORTFOLIO DESCRIPTION",
        base_currency=currency,
        cash_balance=cash,
    )
    listings: list[Listing] = []
    normalized_assets: list[DataAsset] = []
    raw_assets: list[DataAsset] = []
    for index, session_date in enumerate(sessions):
        listing, normalized, raw = _add_priced_holding(
            tmp_path=tmp_path,
            portfolio=portfolio,
            ticker=f"T{index}{currency}",
            currency=currency,
            session_date=session_date,
            provider=provider,
            metadata_currency=(currency if metadata_currency is None else metadata_currency),
        )
        listings.append(listing)
        normalized_assets.append(normalized)
        if raw is not None:
            raw_assets.append(raw)
    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=True,
        require_all=True,
    )
    assert child.status == JobRun.Status.SUCCESS
    return ProofState(
        portfolio=portfolio,
        listings=tuple(listings),
        normalized_assets=tuple(normalized_assets),
        raw_assets=tuple(raw_assets),
        child=child,
    )


def _proof_bytes(state: ProofState, tmp_path: Path) -> bytes:
    proof_asset = DataAsset.objects.get(
        provider="stanstock",
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(state.child.pk),
    )
    return AssetStore(tmp_path).read_bytes(proof_asset.relative_path)


def _clone_child_with_proof(
    state: ProofState,
    tmp_path: Path,
    *,
    mutate: Any,
    asset_time: datetime | None = None,
) -> JobRun:
    """Build an independently registered semantic-tamper fixture.

    The original proof row is immutable, so semantic tampering is exercised
    on a new authoritative child rather than by weakening the DataAsset
    immutability guard in a test.
    """

    JobRun.objects.filter(pk=state.child.pk).update(status=JobRun.Status.FAILED)
    clone = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET,
        attempt=state.child.attempt + 1,
    )
    JobRun.objects.filter(pk=clone.pk).update(started_at=state.child.started_at)
    clone.refresh_from_db()
    raw = json.loads(_proof_bytes(state, tmp_path))
    raw["child_job_run_id"] = str(clone.pk)
    mutate(raw)
    payload = canonical_json_bytes(raw)
    stored = AssetStore(tmp_path).write_bytes(
        f"portfolio/verification/{clone.pk}.json",
        payload,
    )
    registration_time = asset_time or timezone.now()
    asset = DataAsset.objects.create(
        provider=PORTFOLIO_VERIFICATION_PROVIDER,
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(clone.pk),
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        retrieved_at=registration_time,
        available_at=registration_time,
        period_start=TARGET,
        period_end=TARGET,
        schema_version=PORTFOLIO_VERIFICATION_SCHEMA,
        metadata={"usage_scope": PRIVATE_USAGE_SCOPE},
    )
    details = {key: state.child.details[key] for key in BASE_DETAIL_KEYS}
    details["verification_asset_id"] = str(asset.pk)
    details["verification_sha256"] = asset.sha256
    JobRun.objects.filter(pk=clone.pk).update(
        status=JobRun.Status.SUCCESS,
        finished_at=max(timezone.now(), registration_time),
        details=details,
    )
    clone.refresh_from_db()
    return clone


def _batch_details(batch: PortfolioSnapshotBatch) -> dict[str, object]:
    return {
        "portfolios": batch.created + batch.unchanged,
        "snapshots_created": batch.created,
        "snapshots_unchanged": batch.unchanged,
        "snapshot_ids": dict(batch.snapshot_ids),
        "failures": list(batch.failures),
        "required_session_date": TARGET.isoformat(),
        "require_all": True,
        "reason": "",
    }


PersistenceBackend = Literal["postgresql", "sqlite"]


def _project_persisted_field(
    model: type[Any],
    field_name: str,
    value: Decimal,
    *,
    backend: PersistenceBackend,
) -> Decimal:
    field = model._meta.get_field(field_name)
    places = field.decimal_places
    assert places is not None
    quantum = Decimal(1).scaleb(-places)
    if backend == "sqlite":
        # Independent reproduction of Django 5.2's adapter, SQLite NUMERIC
        # affinity, and Col converter. Do not call the production projector.
        with closing(sqlite3.connect(":memory:")) as sqlite_database:
            with closing(
                sqlite_database.execute(
                    "SELECT CAST(? AS NUMERIC)",
                    (str(value),),
                )
            ) as cursor:
                sqlite_numeric = cursor.fetchone()[0]
        assert isinstance(sqlite_numeric, (int, float))
        intermediate = Context(prec=15).create_decimal_from_float(float(sqlite_numeric))
        return intermediate.quantize(quantum, context=field.context)
    with localcontext(field.context) as context:
        context.rounding = ROUND_HALF_UP
        return value.quantize(quantum, context=context)


def _persist_reused_snapshot(
    *,
    tmp_path: Path,
    ticker: str,
    quantity: Decimal,
    average_cost: Decimal,
    price: Decimal,
    cash: Decimal,
    backend: PersistenceBackend,
    invalid_field: str | None = None,
) -> tuple[Portfolio, PortfolioSnapshot, PortfolioSnapshotHolding]:
    """Persist one old snapshot as if either supported backend wrote it."""

    portfolio = Portfolio.objects.create(
        owner=_owner(f"reused-{ticker}-{uuid4()}"),
        name=f"Reused {ticker}",
        base_currency="USD",
        cash_balance=cash,
    )
    listing, source_asset, _raw = _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=portfolio,
        ticker=ticker,
        currency="USD",
        session_date=TARGET,
        metadata_currency="USD",
        quantity=quantity,
        average_cost=average_cost,
        price=price,
    )
    portfolio.refresh_from_db()
    valuation = calculate_portfolio_valuation(
        portfolio,
        expected_as_of_date=TARGET,
    )
    assert valuation.complete
    raw_cost = quantity * average_cost
    raw_market = quantity * price
    raw_gain = raw_market - raw_cost
    raw_return = raw_gain / raw_cost

    position_values = {
        "cost_basis": _project_persisted_field(
            PortfolioSnapshotHolding,
            "cost_basis",
            raw_cost,
            backend=backend,
        ),
        "market_value": _project_persisted_field(
            PortfolioSnapshotHolding,
            "market_value",
            raw_market,
            backend=backend,
        ),
        "unrealized_gain": _project_persisted_field(
            PortfolioSnapshotHolding,
            "unrealized_gain",
            raw_gain,
            backend=backend,
        ),
    }
    snapshot_values: dict[str, Decimal | None] = {
        "securities_value": _project_persisted_field(
            PortfolioSnapshot,
            "securities_value",
            raw_market,
            backend=backend,
        ),
        "total_value": _project_persisted_field(
            PortfolioSnapshot,
            "total_value",
            portfolio.cash_balance + raw_market,
            backend=backend,
        ),
        "cost_basis": _project_persisted_field(
            PortfolioSnapshot,
            "cost_basis",
            raw_cost,
            backend=backend,
        ),
        "unrealized_gain": _project_persisted_field(
            PortfolioSnapshot,
            "unrealized_gain",
            raw_gain,
            backend=backend,
        ),
        "return_pct": _project_persisted_field(
            PortfolioSnapshot,
            "return_pct",
            raw_return,
            backend=backend,
        ),
    }
    raw_fields: dict[str, tuple[type[Any], str, Decimal]] = {
        "position_cost_basis": (
            PortfolioSnapshotHolding,
            "cost_basis",
            raw_cost,
        ),
        "position_market_value": (
            PortfolioSnapshotHolding,
            "market_value",
            raw_market,
        ),
        "position_unrealized_gain": (
            PortfolioSnapshotHolding,
            "unrealized_gain",
            raw_gain,
        ),
        "snapshot_securities_value": (
            PortfolioSnapshot,
            "securities_value",
            raw_market,
        ),
        "snapshot_total_value": (
            PortfolioSnapshot,
            "total_value",
            portfolio.cash_balance + raw_market,
        ),
        "snapshot_cost_basis": (
            PortfolioSnapshot,
            "cost_basis",
            raw_cost,
        ),
        "snapshot_unrealized_gain": (
            PortfolioSnapshot,
            "unrealized_gain",
            raw_gain,
        ),
        "snapshot_return_pct": (
            PortfolioSnapshot,
            "return_pct",
            raw_return,
        ),
    }
    if invalid_field is not None:
        model, field_name, raw_value = raw_fields[invalid_field]
        places = model._meta.get_field(field_name).decimal_places
        assert places is not None
        quantum = Decimal(1).scaleb(-places)
        supported = {
            _project_persisted_field(
                model,
                field_name,
                raw_value,
                backend="sqlite",
            ),
            _project_persisted_field(
                model,
                field_name,
                raw_value,
                backend="postgresql",
            ),
        }
        invalid_value = max(supported) + quantum
        scope, key = invalid_field.split("_", 1)
        if scope == "position":
            position_values[key] = invalid_value
        else:
            snapshot_values[key] = invalid_value

    snapshot = PortfolioSnapshot.objects.create(
        portfolio=portfolio,
        as_of_date=TARGET,
        oldest_price_date=TARGET,
        newest_price_date=TARGET,
        base_currency="USD",
        cash_balance=portfolio.cash_balance,
        input_hash=compute_snapshot_input_hash(portfolio, valuation),
        code_revision=REVISION,
        return_definition="split_adjusted_price_return",
        dividends_included=False,
        corporate_action_warnings=0,
        **snapshot_values,
    )
    position = PortfolioSnapshotHolding.objects.create(
        snapshot=snapshot,
        listing=listing,
        source_asset=source_asset,
        source_session_date=TARGET,
        quantity=quantity,
        average_cost=average_cost,
        price=price,
        corporate_action_suspected=False,
        **position_values,
    )
    return portfolio, snapshot, position


def _preclaim_cash_proof_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    write_file: bool,
) -> tuple[UUID, DataAsset, Path, bytes]:
    """Create the exact proof bytes/path before the real child attempts them."""

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(f"path-claim-{uuid4()}"),
        name="Path claim",
        base_currency="USD",
        cash_balance=Decimal("10"),
    )
    first_batch = snapshot_all_portfolios(expected_as_of_date=TARGET)
    snapshot = PortfolioSnapshot.objects.get(pk=dict(first_batch.snapshot_ids)[str(portfolio.pk)])
    reused_batch = dataclasses.replace(
        first_batch,
        created=0,
        unchanged=1,
        _snapshot_actions=((str(portfolio.pk), str(snapshot.pk), "reused"),),
    )
    child_id = uuid4()
    proof = PortfolioVerificationProof(
        child_job_run_id=child_id,
        report_sha256=report_sha256(_batch_details(reused_batch)),
        snapshots=(
            SnapshotProof(
                classification="reused",
                input_listing_ids=(),
                portfolio_id=portfolio.pk,
                positions=(),
                snapshot_id=snapshot.pk,
                snapshot_sha256=snapshot_row_digest(snapshot),
            ),
        ),
        target_date=TARGET,
        validator_revision=REVISION,
    )
    payload = canonical_json_bytes(proof.to_json())
    relative_path = f"portfolio/verification/{child_id}.json"
    proof_path = AssetStore(tmp_path).resolve(relative_path)
    if write_file:
        AssetStore(tmp_path).write_bytes(relative_path, payload)
    now = timezone.now()
    claimant = DataAsset.objects.create(
        provider="conflicting-owner",
        kind="unrelated-proof",
        subject=str(uuid4()),
        relative_path=relative_path,
        sha256=hashlib.sha256(b"different claimant payload").hexdigest(),
        retrieved_at=now,
        available_at=now,
    )
    real_job_create = JobRun.objects.create

    def create_fixed_child(*args: object, **kwargs: object) -> JobRun:
        if kwargs.get("job_name") == "scheduled_portfolio_snapshots":
            kwargs["id"] = child_id
        return real_job_create(*args, **kwargs)

    monkeypatch.setattr(JobRun.objects, "create", create_fixed_child)
    return child_id, claimant, proof_path, payload


def _late_source_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    late_kind: str,
) -> tuple[
    Portfolio,
    Listing,
    DataAsset,
    DataAsset,
    PortfolioSnapshot,
    PortfolioSnapshotBatch,
    JobRun,
    datetime,
]:
    """Create a source timestamped after its snapshot but before proof time."""

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    base_time = timezone.now()
    snapshot_time = base_time + timedelta(minutes=1)
    late_time = snapshot_time + timedelta(minutes=1)
    proof_time = late_time + timedelta(minutes=1)
    portfolio = Portfolio.objects.create(
        owner=_owner(f"late-{late_kind}-{uuid4()}"),
        name=f"Late {late_kind}",
        base_currency="USD",
    )
    listing, normalized, raw = _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=portfolio,
        ticker=f"LATE{late_kind.upper()}",
        currency="USD",
        session_date=TARGET,
        metadata_currency="USD",
        normalized_observed_at=(late_time if late_kind == "normalized" else base_time),
        raw_observed_at=(late_time if late_kind == "raw" else base_time),
    )
    assert raw is not None
    monkeypatch.setattr(timezone, "now", lambda: base_time)
    child = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET,
    )
    monkeypatch.setattr(timezone, "now", lambda: snapshot_time)
    batch = snapshot_all_portfolios(expected_as_of_date=TARGET)
    snapshot = PortfolioSnapshot.objects.get(pk=dict(batch.snapshot_ids)[str(portfolio.pk)])
    monkeypatch.setattr(timezone, "now", lambda: proof_time)
    assert snapshot.recorded_at < late_time < timezone.now()
    return (
        portfolio,
        listing,
        normalized,
        raw,
        snapshot,
        batch,
        child,
        proof_time,
    )


def _register_unattested_late_source_proof(
    *,
    tmp_path: Path,
    listing: Listing,
    normalized: DataAsset,
    raw: DataAsset,
    snapshot: PortfolioSnapshot,
    batch: PortfolioSnapshotBatch,
    child: JobRun,
    proof_time: datetime,
) -> JobRun:
    """Register a self-consistent forged proof to exercise the parent gate."""

    report = _batch_details(batch)
    position = snapshot.positions.get()
    position_claim = PositionProof(
        listing_currency=listing.currency,
        listing_id=listing.pk,
        normalized_asset_ref=asset_ref_for(normalized),
        provider_subject=listing.provider_symbol,
        raw_asset_ref=asset_ref_for(raw),
        row_id=position.pk,
        row_sha256=snapshot_holding_row_digest(position),
        source_session_date=position.source_session_date,
    )
    proof = PortfolioVerificationProof(
        child_job_run_id=child.pk,
        report_sha256=report_sha256(report),
        snapshots=(
            SnapshotProof(
                classification="created",
                input_listing_ids=(listing.pk,),
                portfolio_id=snapshot.portfolio_id,
                positions=(position_claim,),
                snapshot_id=snapshot.pk,
                snapshot_sha256=snapshot_row_digest(snapshot),
            ),
        ),
        target_date=TARGET,
        validator_revision=REVISION,
    )
    stored = AssetStore(tmp_path).write_bytes(
        f"portfolio/verification/{child.pk}.json",
        canonical_json_bytes(proof.to_json()),
    )
    proof_asset = DataAsset.objects.create(
        provider=PORTFOLIO_VERIFICATION_PROVIDER,
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(child.pk),
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        retrieved_at=proof_time,
        available_at=proof_time,
        period_start=TARGET,
        period_end=TARGET,
        schema_version=PORTFOLIO_VERIFICATION_SCHEMA,
        metadata={"usage_scope": PRIVATE_USAGE_SCOPE},
    )
    details = {
        **report,
        "verification_asset_id": str(proof_asset.pk),
        "verification_sha256": proof_asset.sha256,
    }
    JobRun.objects.filter(pk=child.pk).update(
        status=JobRun.Status.SUCCESS,
        finished_at=proof_time + timedelta(seconds=1),
        details=details,
    )
    child.refresh_from_db()
    return child


def _append_valid_snapshot_position(
    *,
    state: ProofState,
    tmp_path: Path,
) -> PortfolioSnapshotHolding:
    listing, normalized, _raw = _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=state.portfolio,
        ticker=f"APPEND{uuid4().hex[:8].upper()}",
        currency=state.portfolio.base_currency,
        session_date=TARGET,
        metadata_currency=state.portfolio.base_currency,
        quantity=Decimal("3"),
        average_cost=Decimal("20"),
        price=Decimal("25"),
    )
    snapshot = PortfolioSnapshot.objects.get(
        pk=state.child.details["snapshot_ids"][str(state.portfolio.pk)]
    )
    return PortfolioSnapshotHolding.objects.create(
        snapshot=snapshot,
        listing=listing,
        source_asset=normalized,
        source_session_date=TARGET,
        quantity=Decimal("3"),
        average_cost=Decimal("20"),
        price=Decimal("25"),
        cost_basis=Decimal("60"),
        market_value=Decimal("75"),
        unrealized_gain=Decimal("15"),
        corporate_action_suspected=False,
    )


def _prior_snapshot_holding(
    *,
    portfolio: Portfolio,
    listing: Listing,
    source_asset: DataAsset,
    price: Decimal,
    quantity: Decimal,
    suspected: bool,
    snapshot_id: UUID | None = None,
    input_hash: str = "1" * 64,
) -> PortfolioSnapshotHolding:
    average_cost = Decimal("50.000000")
    cost = (quantity * average_cost).quantize(Decimal("0.000001"))
    market = (quantity * price).quantize(Decimal("0.000001"))
    gain = market - cost
    snapshot = PortfolioSnapshot.objects.create(
        id=snapshot_id or uuid4(),
        portfolio=portfolio,
        as_of_date=TARGET - timedelta(days=1),
        oldest_price_date=TARGET - timedelta(days=1),
        newest_price_date=TARGET - timedelta(days=1),
        base_currency=portfolio.base_currency,
        cash_balance=portfolio.cash_balance,
        securities_value=market,
        total_value=portfolio.cash_balance + market,
        cost_basis=cost,
        unrealized_gain=gain,
        return_pct=(gain / cost).quantize(Decimal("0.00000001")),
        input_hash=input_hash,
        code_revision="prior-revision",
        return_definition="split_adjusted_price_return",
        dividends_included=False,
        corporate_action_warnings=int(suspected),
    )
    return PortfolioSnapshotHolding.objects.create(
        snapshot=snapshot,
        listing=listing,
        source_asset=source_asset,
        source_session_date=TARGET - timedelta(days=1),
        quantity=quantity,
        average_cost=average_cost,
        price=price,
        cost_basis=cost,
        market_value=market,
        unrealized_gain=gain,
        corporate_action_suspected=suspected,
    )


def test_digest_field_contract_is_complete_and_every_field_changes_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _build_state(monkeypatch, tmp_path)
    snapshot = PortfolioSnapshot.objects.get(
        pk=next(iter(state.child.details["snapshot_ids"].values()))
    )
    position = snapshot.positions.get()

    assert PORTFOLIO_SNAPSHOT_FIELDS == tuple(
        sorted(field.attname for field in PortfolioSnapshot._meta.concrete_fields)
    )
    assert PORTFOLIO_SNAPSHOT_HOLDING_FIELDS == tuple(
        sorted(field.attname for field in PortfolioSnapshotHolding._meta.concrete_fields)
    )

    for model, label, fields, instance in (
        (
            PortfolioSnapshot,
            PORTFOLIO_SNAPSHOT_MODEL,
            PORTFOLIO_SNAPSHOT_FIELDS,
            snapshot,
        ),
        (
            PortfolioSnapshotHolding,
            PORTFOLIO_SNAPSHOT_HOLDING_MODEL,
            PORTFOLIO_SNAPSHOT_HOLDING_FIELDS,
            position,
        ),
    ):
        values = model_row_values(instance, fields)
        original = row_digest(label, values)
        for field_name in fields:
            changed = dict(values)
            value = changed[field_name]
            if isinstance(value, bool):
                changed[field_name] = not value
            elif isinstance(value, UUID):
                changed[field_name] = uuid4()
            elif isinstance(value, datetime):
                changed[field_name] = value + timedelta(seconds=1)
            elif isinstance(value, date):
                changed[field_name] = value + timedelta(days=1)
            elif isinstance(value, Decimal):
                changed[field_name] = value + Decimal("1")
            elif isinstance(value, int):
                changed[field_name] = value + 1
            elif value is None:
                concrete = next(
                    field for field in model._meta.concrete_fields if field.attname == field_name
                )
                changed[field_name] = (
                    Decimal("1") if concrete.get_internal_type() == "DecimalField" else TARGET
                )
            else:
                changed[field_name] = f"{value}-changed"
            assert row_digest(label, changed) != original, field_name


def test_row_canonicalization_uses_fixed_decimals_utc_and_rejects_naive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _build_state(monkeypatch, tmp_path)
    snapshot = state.portfolio.snapshots.get(as_of_date=TARGET)
    values = model_row_values(snapshot, PORTFOLIO_SNAPSHOT_FIELDS)
    values["cash_balance"] = Decimal("2")
    values["recorded_at"] = datetime(
        2026,
        9,
        4,
        5,
        tzinfo=timezone.get_fixed_timezone(60),
    )

    payload = canonical_row_bytes(PORTFOLIO_SNAPSHOT_MODEL, values)

    assert b'"cash_balance":"2.000000"' in payload
    assert b'"recorded_at":"2026-09-04T04:00:00+00:00"' in payload
    values["recorded_at"] = datetime(2026, 9, 4, 5)
    with pytest.raises(PortfolioVerificationPayloadError, match="naive"):
        canonical_row_bytes(PORTFOLIO_SNAPSHOT_MODEL, values)


def test_proof_is_canonical_private_and_strictly_parsed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _build_state(
        monkeypatch,
        tmp_path,
        sessions=(TARGET, TARGET - timedelta(days=3)),
    )
    payload = _proof_bytes(state, tmp_path)
    proof = parse_portfolio_verification_proof(payload)

    assert canonical_json_bytes(proof.to_json()) == payload
    assert proof.validator_revision == REVISION
    assert proof.snapshots[0].classification == "created"
    assert not payload.endswith(b"\n")
    for private_literal in (
        b"PRIVATE PORTFOLIO NAME",
        b"PRIVATE PORTFOLIO DESCRIPTION",
        b"PRIVATE NOTE",
        b"12345.670001",
        b"7.12345678",
        b"45.678901",
        b"103.125000",
    ):
        assert private_literal not in payload

    raw = json.loads(payload)
    invalid_payloads: list[bytes] = [
        payload + b"\n",
        canonical_json_bytes({**raw, "extra": "no"}),
        canonical_json_bytes({key: value for key, value in raw.items() if key != "target_date"}),
        canonical_json_bytes({**raw, "target_date": "not-a-date"}),
        canonical_json_bytes({**raw, "report_sha256": "A" * 64}),
        canonical_json_bytes({**raw, "snapshots": [raw["snapshots"][0], raw["snapshots"][0]]}),
    ]
    reversed_positions = json.loads(payload)
    reversed_positions["snapshots"][0]["positions"].reverse()
    invalid_payloads.append(canonical_json_bytes(reversed_positions))
    duplicate_key = b'{"child_job_run_id":"' + str(state.child.pk).encode() + b'",' + payload[1:]
    invalid_payloads.append(duplicate_key)
    for invalid in invalid_payloads:
        with pytest.raises(PortfolioVerificationPayloadError):
            parse_portfolio_verification_proof(invalid)


@pytest.mark.parametrize(
    ("require_session_date", "require_all", "proof_expected"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_only_exact_scheduled_flags_create_proof_and_detail_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    require_session_date: bool,
    require_all: bool,
    proof_expected: bool,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    Portfolio.objects.create(
        owner=_owner(f"flags-{require_session_date}-{require_all}"),
        name="Cash",
        base_currency="USD",
        cash_balance=Decimal("10"),
    )

    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=require_session_date,
        require_all=require_all,
    )

    expected_keys = VERIFIED_DETAIL_KEYS if proof_expected else BASE_DETAIL_KEYS
    assert set(child.details) == expected_keys
    assert (
        DataAsset.objects.filter(
            kind=PORTFOLIO_VERIFICATION_KIND,
            subject=str(child.pk),
        ).exists()
        is proof_expected
    )


def test_exact_zero_active_skip_is_proof_free_and_self_contained(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)

    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=True,
        require_all=True,
    )
    result = verify_portfolio_snapshot_stage(child, target_date=TARGET)

    assert child.status == JobRun.Status.SKIPPED
    assert set(child.details) == BASE_DETAIL_KEYS
    assert child.details == {
        "portfolios": 0,
        "snapshots_created": 0,
        "snapshots_unchanged": 0,
        "snapshot_ids": {},
        "failures": [],
        "required_session_date": TARGET.isoformat(),
        "require_all": True,
        "reason": "no_active_portfolios",
    }
    assert result.summary["active_portfolios"] == 0
    assert not DataAsset.objects.filter(kind=PORTFOLIO_VERIFICATION_KIND).exists()


def test_supported_persistence_midpoints_are_exact_not_a_tolerance() -> None:
    positive = Decimal("1.0000005")
    negative = Decimal("-1.0000005")
    assert _supported_persisted_decimals(
        PortfolioSnapshotHolding,
        "unrealized_gain",
        positive,
    ) == frozenset({Decimal("1.000000"), Decimal("1.000001")})
    assert _supported_persisted_decimals(
        PortfolioSnapshotHolding,
        "unrealized_gain",
        negative,
    ) == frozenset({Decimal("-1.000000"), Decimal("-1.000001")})
    for raw, outside in (
        (positive, Decimal("0.999999")),
        (positive, Decimal("1.000002")),
        (negative, Decimal("-1.000002")),
        (negative, Decimal("-0.999999")),
    ):
        assert not _matches_supported_persistence(
            PortfolioSnapshotHolding,
            "unrealized_gain",
            stored=outside,
            derived=raw,
        )
    assert _supported_persisted_decimals(
        PortfolioSnapshot,
        "return_pct",
        Decimal("0.000000005"),
    ) == frozenset({Decimal("0.00000000"), Decimal("0.00000001")})
    assert not _matches_supported_persistence(
        PortfolioSnapshot,
        "return_pct",
        stored=Decimal("0.00000002"),
        derived=Decimal("0.000000005"),
    )
    assert _matches_supported_persistence(
        PortfolioSnapshot,
        "return_pct",
        stored=None,
        derived=None,
    )
    high_significance = Decimal("1524156777.48819700000000")
    assert _supported_persisted_decimals(
        PortfolioSnapshotHolding,
        "market_value",
        high_significance,
    ) == frozenset(
        {
            Decimal("1524156777.488197"),
            Decimal("1524156777.488200"),
        }
    )
    assert not _matches_supported_persistence(
        PortfolioSnapshotHolding,
        "market_value",
        stored=Decimal("1524156777.488198"),
        derived=high_significance,
    )
    counterexample_a = Decimal("44076662.63860854879501")
    assert _supported_persisted_decimals(
        PortfolioSnapshotHolding,
        "market_value",
        counterexample_a,
    ) == frozenset(
        {
            Decimal("44076662.638608"),
            Decimal("44076662.638609"),
        }
    )
    counterexample_b = Decimal("2205113212.92995476730900")
    assert _supported_persisted_decimals(
        PortfolioSnapshotHolding,
        "market_value",
        counterexample_b,
    ) == frozenset(
        {
            Decimal("2205113212.929950"),
            Decimal("2205113212.929955"),
        }
    )
    assert not _matches_supported_persistence(
        PortfolioSnapshotHolding,
        "market_value",
        stored=Decimal("2205113212.929960"),
        derived=counterexample_b,
    )


def test_sqlite_projection_contract_pins_django_adapter_and_numeric_parser() -> None:
    # Importing the installed backend registers its Decimal adapter even when
    # this cross-backend test is currently running on PostgreSQL.
    from django.db.backends.sqlite3 import base as sqlite_backend

    assert DJANGO_VERSION[:2] == (5, 2)
    assert sqlite_backend.Database is sqlite3.dbapi2
    cases = (
        (
            Decimal("44076662.63860854879501"),
            "0x1.50473b51bdeccp+25",
            Decimal("44076662.638608"),
        ),
        (
            Decimal("2205113212.92995476730900"),
            "0x1.06deb6f9dc230p+31",
            Decimal("2205113212.929950"),
        ),
    )
    field = PortfolioSnapshotHolding._meta.get_field("market_value")
    quantum = Decimal("0.000001")
    for value, expected_binary64, expected_persisted in cases:
        assert sqlite3.adapt(value) == str(value)
        with closing(sqlite3.connect(":memory:")) as sqlite_database:
            with closing(
                sqlite_database.execute(
                    "SELECT CAST(? AS NUMERIC)",
                    (str(value),),
                )
            ) as cursor:
                sqlite_numeric = cursor.fetchone()[0]
        assert isinstance(sqlite_numeric, float)
        assert sqlite_numeric.hex() == expected_binary64
        converted = Context(prec=15).create_decimal_from_float(sqlite_numeric)
        assert converted.quantize(quantum, context=field.context) == expected_persisted


def test_supported_persistence_fails_closed_when_sqlite_projection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_sqlite_projection(*args: object, **kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("synthetic in-memory projection failure")

    monkeypatch.setattr(sqlite3, "connect", fail_sqlite_projection)
    assert (
        _supported_persisted_decimals(
            PortfolioSnapshotHolding,
            "market_value",
            Decimal("2205113212.92995476730900"),
        )
        == frozenset()
    )


@pytest.mark.parametrize(
    "invalid",
    [
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        Decimal("1e10000"),
        Decimal("999999999999999999.9999995"),
    ],
)
def test_supported_persistence_rejects_nonfinite_overflow_and_field_overflow(
    invalid: Decimal,
) -> None:
    assert (
        _supported_persisted_decimals(
            PortfolioSnapshotHolding,
            "market_value",
            invalid,
        )
        == frozenset()
    )
    assert not _matches_supported_persistence(
        PortfolioSnapshotHolding,
        "market_value",
        stored=Decimal("0"),
        derived=invalid,
    )


@pytest.mark.skipif(
    connection.vendor != "sqlite",
    reason="Real SQLite DecimalField persistence projection",
)
def test_sqlite_high_significance_writer_child_and_parent_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    quantity = Decimal("1234567891.00000000")
    price = Decimal("1.234567")
    raw_market = quantity * price
    assert raw_market == Decimal("1524156777.48819700000000")
    portfolio = Portfolio.objects.create(
        owner=_owner("sqlite-high-significance"),
        name="SQLite high significance",
        base_currency="USD",
        cash_balance=Decimal("3.000003"),
    )
    _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=portfolio,
        ticker="SQLITEHIGH",
        currency="USD",
        session_date=TARGET,
        metadata_currency="USD",
        quantity=quantity,
        average_cost=Decimal("1.000000"),
        price=price,
    )

    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=True,
        require_all=True,
    )
    result = verify_portfolio_snapshot_stage(child, target_date=TARGET)
    snapshot = PortfolioSnapshot.objects.get(portfolio=portfolio)
    position = snapshot.positions.get()

    assert child.status == JobRun.Status.SUCCESS
    assert result.summary["snapshots_verified"] == 1
    assert position.market_value == Decimal("1524156777.488200")
    assert snapshot.securities_value == Decimal("1524156777.488200")
    assert snapshot.total_value == Decimal("1524156780.488200")


@pytest.mark.skipif(
    connection.vendor != "sqlite",
    reason="Real SQLite NUMERIC-affinity DecimalField persistence",
)
def test_sqlite_numeric_affinity_counterexamples_persist_and_verify(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    cases = (
        (
            "SQLITEAFFINITYA",
            Decimal("78617.10825393"),
            Decimal("559.649757"),
            Decimal("560.649757"),
            Decimal("13.000007"),
            {
                "cost_basis": Decimal("43998045.530355"),
                "market_value": Decimal("44076662.638608"),
                "unrealized_gain": Decimal("78617.108254"),
                "total_value": Decimal("44076675.638616"),
                "return_pct": Decimal("0.00178683"),
            },
        ),
        (
            "SQLITEAFFINITYB",
            Decimal("660545.61919354"),
            Decimal("3337.320850"),
            Decimal("3338.320850"),
            Decimal("17.000009"),
            {
                "cost_basis": Decimal("2204452667.310760"),
                "market_value": Decimal("2205113212.929950"),
                "unrealized_gain": Decimal("660545.619194"),
                "total_value": Decimal("2205113229.929960"),
                "return_pct": Decimal("0.00029964"),
            },
        ),
    )
    portfolios: list[tuple[Portfolio, dict[str, Decimal]]] = []
    for ticker, quantity, average_cost, price, cash, expected in cases:
        portfolio = Portfolio.objects.create(
            owner=_owner(f"sqlite-affinity-{ticker.lower()}"),
            name=f"SQLite affinity {ticker}",
            base_currency="USD",
            cash_balance=cash,
        )
        _add_priced_holding(
            tmp_path=tmp_path,
            portfolio=portfolio,
            ticker=ticker,
            currency="USD",
            session_date=TARGET,
            metadata_currency="USD",
            quantity=quantity,
            average_cost=average_cost,
            price=price,
        )
        portfolios.append((portfolio, expected))

    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=True,
        require_all=True,
    )
    parent_result = verify_portfolio_snapshot_stage(child, target_date=TARGET)

    assert child.status == JobRun.Status.SUCCESS
    assert child.details["snapshots_created"] == 2
    assert parent_result.summary["snapshots_verified"] == 2
    for portfolio, expected in portfolios:
        snapshot = PortfolioSnapshot.objects.get(portfolio=portfolio)
        position = snapshot.positions.get()
        assert position.cost_basis == expected["cost_basis"]
        assert position.market_value == expected["market_value"]
        assert position.unrealized_gain == expected["unrealized_gain"]
        assert snapshot.securities_value == expected["market_value"]
        assert snapshot.cost_basis == expected["cost_basis"]
        assert snapshot.unrealized_gain == expected["unrealized_gain"]
        assert snapshot.total_value == expected["total_value"]
        assert snapshot.return_pct == expected["return_pct"]


def test_real_scheduled_child_and_parent_accept_native_half_scale_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise native Django/SQLite and PostgreSQL half-away writer results."""

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    specs = (
        (
            "POSITIVE",
            Decimal("1.00000050"),
            Decimal("1.000000"),
            Decimal("2.000000"),
            Decimal("0"),
        ),
        (
            "NEGATIVE",
            Decimal("1.00000050"),
            Decimal("6.000000"),
            Decimal("5.000000"),
            Decimal("1"),
        ),
        (
            "RETURNTIE",
            Decimal("1"),
            Decimal("200.000000"),
            Decimal("200.000001"),
            Decimal("0"),
        ),
    )
    portfolios: dict[str, Portfolio] = {}
    for ticker, quantity, average_cost, price, cash in specs:
        portfolio = Portfolio.objects.create(
            owner=_owner(f"native-{ticker.lower()}"),
            name=f"Native {ticker}",
            base_currency="USD",
            cash_balance=cash,
        )
        _add_priced_holding(
            tmp_path=tmp_path,
            portfolio=portfolio,
            ticker=ticker,
            currency="USD",
            session_date=TARGET,
            metadata_currency="USD",
            quantity=quantity,
            average_cost=average_cost,
            price=price,
        )
        portfolios[ticker] = portfolio

    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=True,
        require_all=True,
    )
    result = verify_portfolio_snapshot_stage(child, target_date=TARGET)

    assert child.status == JobRun.Status.SUCCESS
    assert child.details["snapshots_created"] == 3
    assert result.summary["snapshots_verified"] == 3
    assert connection.vendor in {"sqlite", "postgresql"}
    native_backend: PersistenceBackend = (
        "postgresql" if connection.vendor == "postgresql" else "sqlite"
    )
    positive = PortfolioSnapshot.objects.get(portfolio=portfolios["POSITIVE"])
    positive_position = positive.positions.get()
    assert positive_position.cost_basis == _project_persisted_field(
        PortfolioSnapshotHolding,
        "cost_basis",
        Decimal("1.0000005"),
        backend=native_backend,
    )
    assert positive_position.unrealized_gain == _project_persisted_field(
        PortfolioSnapshotHolding,
        "unrealized_gain",
        Decimal("1.0000005"),
        backend=native_backend,
    )
    assert positive.cost_basis == positive_position.cost_basis
    assert positive.unrealized_gain == positive_position.unrealized_gain

    negative = PortfolioSnapshot.objects.get(portfolio=portfolios["NEGATIVE"])
    negative_position = negative.positions.get()
    assert negative_position.market_value == _project_persisted_field(
        PortfolioSnapshotHolding,
        "market_value",
        Decimal("5.0000025"),
        backend=native_backend,
    )
    assert negative_position.unrealized_gain == _project_persisted_field(
        PortfolioSnapshotHolding,
        "unrealized_gain",
        Decimal("-1.0000005"),
        backend=native_backend,
    )
    assert negative.securities_value == negative_position.market_value
    assert negative.total_value == _project_persisted_field(
        PortfolioSnapshot,
        "total_value",
        Decimal("6.0000025"),
        backend=native_backend,
    )
    assert negative.unrealized_gain == negative_position.unrealized_gain

    return_tie = PortfolioSnapshot.objects.get(portfolio=portfolios["RETURNTIE"])
    assert return_tie.return_pct == _project_persisted_field(
        PortfolioSnapshot,
        "return_pct",
        Decimal("0.000000005"),
        backend=native_backend,
    )


@pytest.mark.parametrize(
    "backend",
    ["sqlite", "postgresql"],
    ids=["sqlite-django-projection", "postgresql-half-away"],
)
def test_reused_snapshots_accept_either_supported_persistence_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: PersistenceBackend,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    for ticker, quantity, average_cost, price, cash in (
        (
            "REUSEPOS",
            Decimal("1.00000050"),
            Decimal("1.000000"),
            Decimal("2.000000"),
            Decimal("0"),
        ),
        (
            "REUSENEG",
            Decimal("1.00000050"),
            Decimal("6.000000"),
            Decimal("5.000000"),
            Decimal("1"),
        ),
        (
            "REUSERET",
            Decimal("1"),
            Decimal("200.000000"),
            Decimal("200.000001"),
            Decimal("0"),
        ),
    ):
        _persist_reused_snapshot(
            tmp_path=tmp_path,
            ticker=ticker,
            quantity=quantity,
            average_cost=average_cost,
            price=price,
            cash=cash,
            backend=backend,
        )

    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=True,
        require_all=True,
    )
    result = verify_portfolio_snapshot_stage(child, target_date=TARGET)
    proof = parse_portfolio_verification_proof(
        AssetStore(tmp_path).read_bytes(
            DataAsset.objects.get(
                kind=PORTFOLIO_VERIFICATION_KIND,
                subject=str(child.pk),
            ).relative_path
        )
    )

    assert child.status == JobRun.Status.SUCCESS
    assert child.details["snapshots_created"] == 0
    assert child.details["snapshots_unchanged"] == 3
    assert result.summary["snapshots_verified"] == 3
    assert {snapshot.classification for snapshot in proof.snapshots} == {"reused"}


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="Cross-backend replay of a historical SQLite projection",
)
def test_postgresql_replays_exact_high_significance_sqlite_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    _portfolio, snapshot, position = _persist_reused_snapshot(
        tmp_path=tmp_path,
        ticker="PGREPLAYSQLITE",
        quantity=Decimal("1234567891.00000000"),
        average_cost=Decimal("1.000000"),
        price=Decimal("1.234567"),
        cash=Decimal("3.000003"),
        backend="sqlite",
    )

    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=True,
        require_all=True,
    )
    result = verify_portfolio_snapshot_stage(child, target_date=TARGET)

    position.refresh_from_db()
    snapshot.refresh_from_db()
    assert position.market_value == Decimal("1524156777.488200")
    assert snapshot.securities_value == Decimal("1524156777.488200")
    assert child.details["snapshots_unchanged"] == 1
    assert result.summary["snapshots_verified"] == 1


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="Cross-backend replay of SQLite NUMERIC-affinity counterexamples",
)
@pytest.mark.parametrize(
    ("ticker", "quantity", "average_cost", "price", "cash", "expected_market"),
    [
        (
            "PGREPLAYSQLITEA",
            Decimal("78617.10825393"),
            Decimal("559.649757"),
            Decimal("560.649757"),
            Decimal("13.000007"),
            Decimal("44076662.638608"),
        ),
        (
            "PGREPLAYSQLITEB",
            Decimal("660545.61919354"),
            Decimal("3337.320850"),
            Decimal("3338.320850"),
            Decimal("17.000009"),
            Decimal("2205113212.929950"),
        ),
    ],
)
def test_postgresql_replays_sqlite_numeric_affinity_counterexamples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ticker: str,
    quantity: Decimal,
    average_cost: Decimal,
    price: Decimal,
    cash: Decimal,
    expected_market: Decimal,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    _portfolio, snapshot, position = _persist_reused_snapshot(
        tmp_path=tmp_path,
        ticker=ticker,
        quantity=quantity,
        average_cost=average_cost,
        price=price,
        cash=cash,
        backend="sqlite",
    )

    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=True,
        require_all=True,
    )
    result = verify_portfolio_snapshot_stage(child, target_date=TARGET)

    position.refresh_from_db()
    snapshot.refresh_from_db()
    assert position.market_value == expected_market
    assert snapshot.securities_value == expected_market
    assert child.details["snapshots_unchanged"] == 1
    assert result.summary["snapshots_verified"] == 1


@pytest.mark.skipif(
    connection.vendor != "sqlite",
    reason="Cross-backend replay of a PostgreSQL numeric half-tie",
)
def test_sqlite_replays_exact_postgresql_half_tie_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    _portfolio, snapshot, position = _persist_reused_snapshot(
        tmp_path=tmp_path,
        ticker="SQLITEREPLAYPG",
        quantity=Decimal("1.00000050"),
        average_cost=Decimal("1.000000"),
        price=Decimal("2.000000"),
        cash=Decimal("0"),
        backend="postgresql",
    )

    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=True,
        require_all=True,
    )
    result = verify_portfolio_snapshot_stage(child, target_date=TARGET)

    position.refresh_from_db()
    snapshot.refresh_from_db()
    assert position.cost_basis == Decimal("1.000001")
    assert position.unrealized_gain == Decimal("1.000001")
    assert snapshot.cost_basis == Decimal("1.000001")
    assert snapshot.unrealized_gain == Decimal("1.000001")
    assert child.details["snapshots_unchanged"] == 1
    assert result.summary["snapshots_verified"] == 1


@pytest.mark.parametrize(
    ("invalid_field", "average_cost", "price", "cash", "reason_code"),
    [
        (
            "position_cost_basis",
            Decimal("1"),
            Decimal("2"),
            Decimal("0"),
            "portfolio_position_arithmetic_invalid",
        ),
        (
            "position_market_value",
            Decimal("6"),
            Decimal("5"),
            Decimal("1"),
            "portfolio_position_arithmetic_invalid",
        ),
        (
            "position_unrealized_gain",
            Decimal("1"),
            Decimal("2"),
            Decimal("0"),
            "portfolio_position_arithmetic_invalid",
        ),
        (
            "snapshot_securities_value",
            Decimal("6"),
            Decimal("5"),
            Decimal("1"),
            "portfolio_snapshot_arithmetic_invalid",
        ),
        (
            "snapshot_cost_basis",
            Decimal("1"),
            Decimal("2"),
            Decimal("0"),
            "portfolio_snapshot_arithmetic_invalid",
        ),
        (
            "snapshot_total_value",
            Decimal("6"),
            Decimal("5"),
            Decimal("1"),
            "portfolio_snapshot_arithmetic_invalid",
        ),
        (
            "snapshot_unrealized_gain",
            Decimal("1"),
            Decimal("2"),
            Decimal("0"),
            "portfolio_snapshot_arithmetic_invalid",
        ),
        (
            "snapshot_return_pct",
            Decimal("200.000000"),
            Decimal("200.000001"),
            Decimal("0"),
            "portfolio_snapshot_arithmetic_invalid",
        ),
    ],
)
def test_one_quantum_outside_supported_results_fails_scheduled_attestation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_field: str,
    average_cost: Decimal,
    price: Decimal,
    cash: Decimal,
    reason_code: str,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    _persist_reused_snapshot(
        tmp_path=tmp_path,
        ticker=f"BAD{invalid_field.upper()}",
        quantity=Decimal("1.00000050") if invalid_field != "snapshot_return_pct" else Decimal("1"),
        average_cost=average_cost,
        price=price,
        cash=cash,
        backend="sqlite",
        invalid_field=invalid_field,
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        execute_portfolio_snapshot_job(
            target_date=TARGET,
            require_session_date=True,
            require_all=True,
        )

    assert excinfo.value.reason_code == reason_code
    child = JobRun.objects.get(job_name="scheduled_portfolio_snapshots")
    assert child.status == JobRun.Status.FAILED
    assert not DataAsset.objects.filter(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(child.pk),
    ).exists()


@pytest.mark.parametrize("currency", ["USD", "EUR", "GBP"])
def test_native_currency_positions_attest_without_fx(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    currency: str,
) -> None:
    state = _build_state(monkeypatch, tmp_path, currency=currency)

    result = verify_portfolio_snapshot_stage(state.child, target_date=TARGET)
    proof = parse_portfolio_verification_proof(_proof_bytes(state, tmp_path))

    assert result.summary["snapshots_verified"] == 1
    assert proof.snapshots[0].positions[0].listing_currency == currency
    assert all("fx" not in ref.kind.lower() for ref in result.asset_refs)


def test_currency_mismatch_and_over_stale_source_fail_without_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Mismatched",
        base_currency="USD",
    )
    _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=portfolio,
        ticker="MISMATCH",
        currency="USD",
        session_date=TARGET,
        metadata_currency="EUR",
    )

    with pytest.raises(
        RefreshVerificationError,
        match="currency does not match",
    ):
        execute_portfolio_snapshot_job(
            target_date=TARGET,
            require_session_date=True,
            require_all=True,
        )
    assert not DataAsset.objects.filter(kind=PORTFOLIO_VERIFICATION_KIND).exists()

    Portfolio.objects.filter(pk=portfolio.pk).update(archived_at=timezone.now())
    stale = Portfolio.objects.create(
        owner=_owner("stale-owner"),
        name="Stale",
        base_currency="USD",
    )
    _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=stale,
        ticker="CURRENT",
        currency="USD",
        session_date=TARGET,
        metadata_currency="USD",
    )
    _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=stale,
        ticker="STALE",
        currency="USD",
        session_date=TARGET - timedelta(days=8),
        metadata_currency="USD",
    )
    with pytest.raises(ValueError, match="Scheduled portfolio snapshots failed"):
        execute_portfolio_snapshot_job(
            target_date=TARGET,
            require_session_date=True,
            require_all=True,
        )


@pytest.mark.parametrize("late_kind", ["normalized", "raw"])
def test_child_rejects_source_first_available_after_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    late_kind: str,
) -> None:
    (
        _portfolio,
        _listing,
        _normalized,
        _raw,
        _snapshot,
        batch,
        child,
        _proof_time,
    ) = _late_source_fixture(
        monkeypatch,
        tmp_path,
        late_kind=late_kind,
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        attest_scheduled_portfolio_snapshots(
            child_run=child,
            target_date=TARGET,
            report_details=_batch_details(batch),
            snapshot_actions=batch._snapshot_actions,
            validator_revision=REVISION,
        )

    assert excinfo.value.reason_code == "portfolio_source_asset_time_invalid"
    assert str(tmp_path) not in str(excinfo.value)
    assert not DataAsset.objects.filter(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(child.pk),
    ).exists()


@pytest.mark.parametrize("late_kind", ["normalized", "raw"])
def test_parent_rejects_source_first_available_after_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    late_kind: str,
) -> None:
    (
        _portfolio,
        listing,
        normalized,
        raw,
        snapshot,
        batch,
        child,
        proof_time,
    ) = _late_source_fixture(
        monkeypatch,
        tmp_path,
        late_kind=late_kind,
    )
    forged_child = _register_unattested_late_source_proof(
        tmp_path=tmp_path,
        listing=listing,
        normalized=normalized,
        raw=raw,
        snapshot=snapshot,
        batch=batch,
        child=child,
        proof_time=proof_time,
    )

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_portfolio_snapshot_stage(forged_child, target_date=TARGET)

    assert excinfo.value.reason_code == "portfolio_source_asset_time_invalid"
    assert str(tmp_path) not in str(excinfo.value)


@pytest.mark.parametrize("provider", ["synthetic", "synthetic_demo"])
def test_exact_synthetic_providers_may_omit_raw_evidence_and_currency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Legacy synthetic",
        base_currency="EUR",
    )
    _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=portfolio,
        ticker="SYNTH",
        currency="EUR",
        session_date=TARGET,
        provider=provider,
        metadata_currency=None,
    )

    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=True,
        require_all=True,
    )
    proof = parse_portfolio_verification_proof(
        AssetStore(tmp_path).read_bytes(
            DataAsset.objects.get(
                kind=PORTFOLIO_VERIFICATION_KIND,
                subject=str(child.pk),
            ).relative_path
        )
    )

    assert proof.snapshots[0].positions[0].raw_asset_ref is None
    verify_portfolio_snapshot_stage(child, target_date=TARGET)


@pytest.mark.parametrize(
    ("provider", "metadata_currency", "partial_raw"),
    [
        ("twelve_data", None, False),
        ("unknown_production", "USD", False),
        ("synthetic_demo", "USD", True),
    ],
)
def test_missing_currency_unknown_rawless_and_partial_raw_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    metadata_currency: str | None,
    partial_raw: bool,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Incomplete source",
        base_currency="USD",
    )
    _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=portfolio,
        ticker="INCOMPLETE",
        currency="USD",
        session_date=TARGET,
        provider=provider,
        metadata_currency=metadata_currency,
        include_raw=False,
        partial_raw=partial_raw,
    )

    with pytest.raises(RefreshVerificationError):
        execute_portfolio_snapshot_job(
            target_date=TARGET,
            require_session_date=True,
            require_all=True,
        )
    child = JobRun.objects.get(job_name="scheduled_portfolio_snapshots")
    assert child.status == JobRun.Status.FAILED
    assert not DataAsset.objects.filter(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(child.pk),
    ).exists()


def test_mixed_bounded_source_dates_and_cash_only_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _build_state(
        monkeypatch,
        tmp_path,
        sessions=(TARGET, TARGET - timedelta(days=7)),
    )
    cash = Portfolio.objects.create(
        owner=_owner("cash-only"),
        name="Cash only",
        base_currency="GBP",
        cash_balance=Decimal("200.000000"),
    )
    # The active set changed after the first completed child, so use the
    # immutable first proof only for the mixed-date assertion.
    proof = parse_portfolio_verification_proof(_proof_bytes(state, tmp_path))
    assert {item.source_session_date for item in proof.snapshots[0].positions} == {
        TARGET,
        TARGET - timedelta(days=7),
    }
    Portfolio.objects.filter(pk=state.portfolio.pk).update(archived_at=timezone.now())
    second = execute_portfolio_snapshot_job(
        target_date=TARGET + timedelta(days=1),
        require_session_date=True,
        require_all=True,
    )
    snapshot = PortfolioSnapshot.objects.get(pk=second.details["snapshot_ids"][str(cash.pk)])
    assert snapshot.oldest_price_date is None
    assert snapshot.newest_price_date is None
    assert snapshot.securities_value == 0
    assert snapshot.cost_basis == 0
    assert snapshot.unrealized_gain == 0
    assert snapshot.total_value == snapshot.cash_balance
    assert snapshot.return_pct is None
    assert snapshot.return_definition == "price_return"
    assert snapshot.dividends_included is False


def test_created_and_old_reused_revision_rules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Reusable cash",
        base_currency="EUR",
        cash_balance=Decimal("50"),
    )
    portfolio.refresh_from_db()
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", "old-revision")
    old_snapshot, created = record_portfolio_snapshot(
        portfolio,
        expected_as_of_date=TARGET,
    )
    assert created is True
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)

    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=True,
        require_all=True,
    )
    proof = parse_portfolio_verification_proof(
        AssetStore(tmp_path).read_bytes(
            DataAsset.objects.get(
                kind=PORTFOLIO_VERIFICATION_KIND,
                subject=str(child.pk),
            ).relative_path
        )
    )

    assert child.details["snapshots_created"] == 0
    assert child.details["snapshots_unchanged"] == 1
    assert proof.validator_revision == REVISION
    assert proof.snapshots[0].classification == "reused"
    assert old_snapshot.code_revision == "old-revision"
    verify_portfolio_snapshot_stage(child, target_date=TARGET)


def test_snapshot_batch_preserves_public_results_and_records_exact_private_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Batch action compatibility",
        base_currency="USD",
        cash_balance=Decimal("25"),
    )

    first = snapshot_all_portfolios(expected_as_of_date=TARGET)
    second = snapshot_all_portfolios(expected_as_of_date=TARGET)
    snapshot_id = dict(first.snapshot_ids)[str(portfolio.pk)]

    assert (first.created, first.unchanged, first.failures, first.snapshot_ids) == (
        1,
        0,
        (),
        ((str(portfolio.pk), snapshot_id),),
    )
    assert (second.created, second.unchanged, second.failures, second.snapshot_ids) == (
        0,
        1,
        (),
        first.snapshot_ids,
    )
    assert first._snapshot_actions == ((str(portfolio.pk), snapshot_id, "created"),)
    assert second._snapshot_actions == ((str(portfolio.pk), snapshot_id, "reused"),)


@pytest.mark.parametrize(
    ("snapshot_revision", "proof_expected"),
    [(REVISION, True), ("different-revision", False)],
)
def test_post_start_reused_snapshot_keeps_actual_action_and_requires_current_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    snapshot_revision: str,
    proof_expected: bool,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name=f"Post-start reuse {snapshot_revision}",
        base_currency="USD",
        cash_balance=Decimal("25"),
    )
    portfolio.refresh_from_db()
    child = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET,
    )
    monkeypatch.setattr(
        "stanstock.portfolio.service.code_revision",
        lambda: snapshot_revision,
    )
    snapshot, created = record_portfolio_snapshot(
        portfolio,
        expected_as_of_date=TARGET,
    )
    batch = snapshot_all_portfolios(expected_as_of_date=TARGET)

    assert created is True
    assert snapshot.recorded_at >= child.started_at
    assert batch.created == 0
    assert batch.unchanged == 1
    assert batch._snapshot_actions == ((str(portfolio.pk), str(snapshot.pk), "reused"),)
    if proof_expected:
        asset = attest_scheduled_portfolio_snapshots(
            child_run=child,
            target_date=TARGET,
            report_details=_batch_details(batch),
            snapshot_actions=batch._snapshot_actions,
            validator_revision=REVISION,
        )
        proof = parse_portfolio_verification_proof(
            AssetStore(tmp_path).read_bytes(asset.relative_path)
        )
        assert proof.snapshots[0].classification == "reused"
        assert proof.validator_revision == REVISION
    else:
        with pytest.raises(
            RefreshVerificationError,
            match="action timing or validator revision",
        ):
            attest_scheduled_portfolio_snapshots(
                child_run=child,
                target_date=TARGET,
                report_details=_batch_details(batch),
                snapshot_actions=batch._snapshot_actions,
                validator_revision=REVISION,
            )
        assert not DataAsset.objects.filter(
            kind=PORTFOLIO_VERIFICATION_KIND,
            subject=str(child.pk),
        ).exists()


@pytest.mark.parametrize(
    ("prior_price", "prior_quantity", "prior_suspected", "expected"),
    [
        (Decimal("300"), Decimal("2"), False, True),
        (Decimal("100"), Decimal("2"), False, False),
        (Decimal("100"), Decimal("2"), True, True),
        (Decimal("300"), Decimal("3"), True, False),
    ],
)
def test_immutable_prior_corporate_action_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    prior_price: Decimal,
    prior_quantity: Decimal,
    prior_suspected: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Corporate action replay",
        base_currency="USD",
    )
    listing, normalized, _raw = _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=portfolio,
        ticker="CA",
        currency="USD",
        session_date=TARGET,
        metadata_currency="USD",
        quantity=Decimal("2"),
        average_cost=Decimal("50"),
        price=Decimal("100"),
    )
    _prior_snapshot_holding(
        portfolio=portfolio,
        listing=listing,
        source_asset=normalized,
        price=prior_price,
        quantity=prior_quantity,
        suspected=prior_suspected,
    )

    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=True,
        require_all=True,
    )
    current = PortfolioSnapshotHolding.objects.get(
        snapshot_id=child.details["snapshot_ids"][str(portfolio.pk)]
    )

    assert current.corporate_action_suspected is expected
    assert current.snapshot.corporate_action_warnings == int(expected)
    verify_portfolio_snapshot_stage(child, target_date=TARGET)


def test_corporate_action_prior_tie_uses_existing_snapshot_id_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Corporate action tie",
        base_currency="USD",
    )
    listing, normalized, _raw = _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=portfolio,
        ticker="CATIE",
        currency="USD",
        session_date=TARGET,
        metadata_currency="USD",
        quantity=Decimal("2"),
        average_cost=Decimal("50"),
        price=Decimal("100"),
    )
    real_now = timezone.now
    tied_time = real_now() - timedelta(minutes=1)
    monkeypatch.setattr("django.utils.timezone.now", lambda: tied_time)
    _prior_snapshot_holding(
        portfolio=portfolio,
        listing=listing,
        source_asset=normalized,
        price=Decimal("300"),
        quantity=Decimal("2"),
        suspected=False,
        snapshot_id=UUID("00000000-0000-0000-0000-000000000001"),
        input_hash="1" * 64,
    )
    _prior_snapshot_holding(
        portfolio=portfolio,
        listing=listing,
        source_asset=normalized,
        price=Decimal("100"),
        quantity=Decimal("2"),
        suspected=False,
        snapshot_id=UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),
        input_hash="2" * 64,
    )
    monkeypatch.setattr("django.utils.timezone.now", real_now)

    child = execute_portfolio_snapshot_job(
        target_date=TARGET,
        require_session_date=True,
        require_all=True,
    )
    current = PortfolioSnapshotHolding.objects.get(
        snapshot_id=child.details["snapshot_ids"][str(portfolio.pk)]
    )

    assert current.corporate_action_suspected is False
    verify_portfolio_snapshot_stage(child, target_date=TARGET)


def test_wrong_arithmetic_at_birth_fails_before_proof_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Fabricated at birth",
        base_currency="USD",
        cash_balance=Decimal("10"),
    )
    portfolio.refresh_from_db()

    def fabricate(**kwargs: object) -> PortfolioSnapshotBatch:
        target = kwargs["expected_as_of_date"]
        assert isinstance(target, date)
        input_hash = hashlib.sha256(
            canonical_json_bytes(
                {
                    "portfolio_id": str(portfolio.pk),
                    "as_of_date": target.isoformat(),
                    "base_currency": portfolio.base_currency,
                    "cash_balance": str(portfolio.cash_balance),
                    "positions": [],
                }
            )
        ).hexdigest()
        snapshot = PortfolioSnapshot.objects.create(
            portfolio=portfolio,
            as_of_date=target,
            base_currency=portfolio.base_currency,
            cash_balance=portfolio.cash_balance,
            securities_value=Decimal("0"),
            total_value=Decimal("999"),
            cost_basis=Decimal("0"),
            unrealized_gain=Decimal("0"),
            return_pct=None,
            input_hash=input_hash,
            code_revision=REVISION,
            return_definition="price_return",
            dividends_included=False,
            corporate_action_warnings=0,
        )
        return PortfolioSnapshotBatch(
            created=1,
            unchanged=0,
            failures=(),
            snapshot_ids=((str(portfolio.pk), str(snapshot.pk)),),
            _snapshot_actions=((str(portfolio.pk), str(snapshot.pk), "created"),),
        )

    monkeypatch.setattr(portfolio_jobs, "snapshot_all_portfolios", fabricate)
    with pytest.raises(
        RefreshVerificationError,
        match="cash-only portfolio snapshot",
    ):
        execute_portfolio_snapshot_job(
            target_date=TARGET,
            require_session_date=True,
            require_all=True,
        )

    child = JobRun.objects.get(job_name="scheduled_portfolio_snapshots")
    assert child.status == JobRun.Status.FAILED
    assert not DataAsset.objects.filter(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(child.pk),
    ).exists()


@pytest.mark.parametrize("failure_mode", ["report", "state", "race", "classification"])
def test_wrong_report_or_raced_state_fails_child_without_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_mode: str,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Race",
        base_currency="USD",
        cash_balance=Decimal("10"),
    )
    real_snapshot_all = portfolio_jobs.snapshot_all_portfolios

    def changed_report(**kwargs: object) -> PortfolioSnapshotBatch:
        report = real_snapshot_all(**kwargs)
        if failure_mode == "report":
            return dataclasses.replace(
                report,
                snapshot_ids=((str(portfolio.pk), str(uuid4())),),
            )
        if failure_mode == "state":
            Portfolio.objects.filter(pk=portfolio.pk).update(cash_balance=Decimal("11"))
        if failure_mode == "classification":
            return dataclasses.replace(report, created=0, unchanged=1)
        return report

    monkeypatch.setattr(portfolio_jobs, "snapshot_all_portfolios", changed_report)
    if failure_mode == "race":
        real_final = refresh_validation._fresh_input_signatures

        def mutate_before_final(**kwargs: object) -> tuple[tuple[object, ...], ...]:
            Portfolio.objects.filter(pk=portfolio.pk).update(cash_balance=Decimal("12"))
            return real_final(**kwargs)

        monkeypatch.setattr(
            refresh_validation,
            "_fresh_input_signatures",
            mutate_before_final,
        )

    with pytest.raises((RefreshVerificationError, ValueError)):
        execute_portfolio_snapshot_job(
            target_date=TARGET,
            require_session_date=True,
            require_all=True,
        )

    child = JobRun.objects.get(job_name="scheduled_portfolio_snapshots")
    assert child.status == JobRun.Status.FAILED
    assert not DataAsset.objects.filter(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(child.pk),
    ).exists()
    assert not (tmp_path / f"portfolio/verification/{child.pk}.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "cash",
        "archive",
        "holding_quantity",
        "holding_delete",
        "listing_subject",
        "market_close",
        "new_portfolio",
    ],
)
def test_post_snapshot_mutable_input_changes_fail_child_without_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Mutable race",
        base_currency="USD",
        cash_balance=Decimal("10"),
    )
    listing, _normalized, _raw = _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=portfolio,
        ticker="MUTATE",
        currency="USD",
        session_date=TARGET,
        metadata_currency="USD",
    )
    real_snapshot_all = portfolio_jobs.snapshot_all_portfolios

    def mutate_after_snapshot(**kwargs: object) -> PortfolioSnapshotBatch:
        report = real_snapshot_all(**kwargs)
        if mutation == "cash":
            Portfolio.objects.filter(pk=portfolio.pk).update(cash_balance=Decimal("11"))
        elif mutation == "archive":
            Portfolio.objects.filter(pk=portfolio.pk).update(archived_at=timezone.now())
        elif mutation == "holding_quantity":
            PortfolioHolding.objects.filter(portfolio=portfolio).update(quantity=Decimal("9"))
        elif mutation == "holding_delete":
            PortfolioHolding.objects.filter(portfolio=portfolio).delete()
        elif mutation == "listing_subject":
            Listing.objects.filter(pk=listing.pk).update(provider_symbol="CHANGED:SUBJECT")
        elif mutation == "market_close":
            LatestMarketData.objects.filter(listing=listing).update(close=Decimal("104"))
        elif mutation == "new_portfolio":
            Portfolio.objects.create(
                owner=_owner("new-active-during-attestation"),
                name="New active",
                base_currency="USD",
            )
        return report

    monkeypatch.setattr(
        portfolio_jobs,
        "snapshot_all_portfolios",
        mutate_after_snapshot,
    )
    with pytest.raises((RefreshVerificationError, ValueError)):
        execute_portfolio_snapshot_job(
            target_date=TARGET,
            require_session_date=True,
            require_all=True,
        )

    child = JobRun.objects.get(job_name="scheduled_portfolio_snapshots")
    assert child.status == JobRun.Status.FAILED
    assert not DataAsset.objects.filter(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(child.pk),
    ).exists()


@pytest.mark.parametrize(
    "tamper",
    [
        "report",
        "snapshot_digest",
        "input_order",
        "source_date",
        "provider",
        "subject",
        "currency",
        "normalized_id",
        "raw",
    ],
)
def test_semantic_proof_tampering_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
) -> None:
    state = _build_state(
        monkeypatch,
        tmp_path,
        sessions=(TARGET, TARGET - timedelta(days=2)),
    )

    def mutate(raw: dict[str, object]) -> None:
        snapshot = raw["snapshots"][0]  # type: ignore[index]
        position = snapshot["positions"][0]
        if tamper == "report":
            raw["report_sha256"] = "0" * 64
        elif tamper == "snapshot_digest":
            snapshot["snapshot_sha256"] = "0" * 64
        elif tamper == "input_order":
            snapshot["input_listing_ids"].reverse()
        elif tamper == "source_date":
            position["source_session_date"] = (TARGET - timedelta(days=1)).isoformat()
        elif tamper == "provider":
            position["normalized_asset_ref"]["provider"] = "other-provider"
            position["raw_asset_ref"]["provider"] = "other-provider"
        elif tamper == "subject":
            position["provider_subject"] = "OTHER:SUBJECT"
            position["normalized_asset_ref"]["subject"] = "OTHER:SUBJECT"
            position["raw_asset_ref"]["subject"] = "OTHER:SUBJECT"
        elif tamper == "currency":
            position["listing_currency"] = "GBP"
        elif tamper == "normalized_id":
            position["normalized_asset_ref"]["id"] = str(uuid4())
        elif tamper == "raw":
            position["raw_asset_ref"] = None

    clone = _clone_child_with_proof(
        state,
        tmp_path,
        mutate=mutate,
    )
    with pytest.raises(RefreshVerificationError):
        verify_portfolio_snapshot_stage(clone, target_date=TARGET)


def test_proof_registration_time_outside_child_window_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _build_state(monkeypatch, tmp_path)
    clone = _clone_child_with_proof(
        state,
        tmp_path,
        mutate=lambda raw: None,
        asset_time=state.child.started_at - timedelta(seconds=1),
    )

    with pytest.raises(
        RefreshVerificationError,
        match="identity or timing",
    ):
        verify_portfolio_snapshot_stage(clone, target_date=TARGET)


def test_parent_retry_ignores_all_current_mutable_portfolio_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _build_state(monkeypatch, tmp_path)
    holding = PortfolioHolding.objects.get(portfolio=state.portfolio)
    Portfolio.objects.filter(pk=state.portfolio.pk).update(
        cash_balance=Decimal("999999"),
        archived_at=timezone.now(),
    )
    PortfolioHolding.objects.filter(pk=holding.pk).delete()
    Listing.objects.filter(pk=state.listings[0].pk).update(
        currency="GBP",
        provider_symbol="CHANGED",
        ticker="CHANGED",
        is_active=False,
    )
    LatestMarketData.objects.filter(listing_id=state.listings[0].pk).delete()
    Portfolio.objects.create(
        owner=_owner("later-owner"),
        name="Later active portfolio",
        base_currency="USD",
    )

    class NoCurrentManager:
        def __getattr__(self, name: str) -> Any:
            pytest.fail(f"parent verification queried current manager method {name}")

    monkeypatch.setattr(Portfolio, "objects", NoCurrentManager())
    monkeypatch.setattr(PortfolioHolding, "objects", NoCurrentManager())
    monkeypatch.setattr(Listing, "objects", NoCurrentManager())
    monkeypatch.setattr(LatestMarketData, "objects", NoCurrentManager())
    with CaptureQueriesContext(connection) as queries:
        result = verify_portfolio_snapshot_stage(state.child, target_date=TARGET)
    sql = "\n".join(query["sql"].lower() for query in queries.captured_queries)

    assert result.summary["snapshots_verified"] == 1
    for forbidden_table in (
        '"portfolio_portfolio"',
        '"portfolio_portfolioholding"',
        '"data_listing"',
        '"data_latestmarketdata"',
    ):
        assert forbidden_table not in sql


def test_parent_rejects_position_appended_after_attestation_without_provider_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _build_state(monkeypatch, tmp_path)
    appended = _append_valid_snapshot_position(state=state, tmp_path=tmp_path)

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("position-closure verification must not fetch provider data")

    monkeypatch.setattr(
        "stanstock.data.providers.twelve_data.fetch_daily_price_series",
        no_provider,
    )
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_portfolio_snapshot_stage(state.child, target_date=TARGET)

    assert excinfo.value.reason_code == "portfolio_verification_position_closure_mismatch"
    assert PortfolioSnapshotHolding.objects.filter(pk=appended.pk).exists()
    assert (
        DataAsset.objects.filter(
            kind=PORTFOLIO_VERIFICATION_KIND,
            subject=str(state.child.pk),
        ).count()
        == 1
    )


def test_parent_includes_proof_normalized_and_raw_and_calls_no_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _build_state(monkeypatch, tmp_path)

    def no_provider(*args: object, **kwargs: object) -> None:
        pytest.fail("portfolio proof verification must not fetch provider data")

    monkeypatch.setattr(
        "stanstock.data.providers.twelve_data.fetch_daily_price_series",
        no_provider,
    )
    result = verify_portfolio_snapshot_stage(state.child, target_date=TARGET)
    ids = {ref.id for ref in result.asset_refs}
    proof_asset = DataAsset.objects.get(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(state.child.pk),
    )

    assert ids == {
        proof_asset.id,
        state.normalized_assets[0].id,
        state.raw_assets[0].id,
    }


@pytest.mark.parametrize("asset_kind", ["normalized", "raw"])
def test_physical_source_tampering_fails_and_restore_recovers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    asset_kind: str,
) -> None:
    state = _build_state(monkeypatch, tmp_path)
    asset = state.normalized_assets[0] if asset_kind == "normalized" else state.raw_assets[0]
    path = AssetStore(tmp_path).resolve(asset.relative_path)
    original = path.read_bytes()
    path.write_bytes(original + b"tamper")

    with pytest.raises(RefreshVerificationError):
        verify_portfolio_snapshot_stage(state.child, target_date=TARGET)

    path.write_bytes(original)
    verify_portfolio_snapshot_stage(state.child, target_date=TARGET)


def test_registration_failure_cleans_only_attempt_file_and_preserves_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    Portfolio.objects.create(
        owner=_owner(),
        name="Registration cleanup",
        base_currency="USD",
    )
    child = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET,
    )
    batch = snapshot_all_portfolios(expected_as_of_date=TARGET)
    real_create = DataAsset.objects.create

    def fail_proof_registration(*args: object, **kwargs: object) -> DataAsset:
        if kwargs.get("kind") == PORTFOLIO_VERIFICATION_KIND:
            raise RuntimeError("sentinel registration failure")
        return real_create(*args, **kwargs)

    monkeypatch.setattr(DataAsset.objects, "create", fail_proof_registration)
    with pytest.raises(RuntimeError, match="sentinel registration failure"):
        attest_scheduled_portfolio_snapshots(
            child_run=child,
            target_date=TARGET,
            report_details=_batch_details(batch),
            snapshot_actions=batch._snapshot_actions,
            validator_revision=REVISION,
        )

    assert not (tmp_path / f"portfolio/verification/{child.pk}.json").exists()
    assert not DataAsset.objects.filter(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(child.pk),
    ).exists()


@pytest.mark.parametrize(
    "conflict_mode",
    ["precheck", "create-race"],
)
def test_scheduled_child_normalizes_claimed_proof_path_without_leaking_or_deleting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    conflict_mode: str,
) -> None:
    child_id, claimant, proof_path, expected_bytes = _preclaim_cash_proof_path(
        monkeypatch,
        tmp_path,
        write_file=False,
    )
    real_claimed = refresh_validation._proof_path_claimed
    claim_checks = 0

    if conflict_mode == "create-race":

        def miss_precheck_once(relative_path: str) -> bool:
            nonlocal claim_checks
            claim_checks += 1
            if claim_checks == 1:
                return False
            return real_claimed(relative_path)

        monkeypatch.setattr(
            refresh_validation,
            "_proof_path_claimed",
            miss_precheck_once,
        )

    caplog.set_level("ERROR")
    with pytest.raises(RefreshVerificationError) as excinfo:
        execute_portfolio_snapshot_job(
            target_date=TARGET,
            require_session_date=True,
            require_all=True,
        )

    assert excinfo.value.reason_code == "portfolio_verification_path_registered"
    assert str(excinfo.value) == ("The portfolio verification proof location is already registered")
    failed_child = JobRun.objects.get(pk=child_id)
    assert failed_child.status == JobRun.Status.FAILED
    assert DataAsset.objects.filter(pk=claimant.pk).exists()
    assert claimant.sha256 != hashlib.sha256(expected_bytes).hexdigest()
    assert not proof_path.exists()
    for leaked_path in (
        claimant.relative_path,
        str(proof_path),
        str(tmp_path),
    ):
        assert leaked_path not in str(excinfo.value)
        assert leaked_path not in failed_child.error
        assert leaked_path not in caplog.text
    if conflict_mode == "create-race":
        # A false precheck reaches the real unique constraint.  The nested
        # savepoint rolls back that failed INSERT before the second query
        # classifies the now-visible path claimant.
        assert claim_checks == 2


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL proof-path create-race coverage",
)
@pytest.mark.django_db(transaction=True)
def test_postgresql_concurrent_path_claim_wins_before_scheduled_child_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    Portfolio.objects.create(
        owner=_owner("concurrent-path-claim"),
        name="Concurrent path claim",
        base_currency="USD",
        cash_balance=Decimal("10"),
    )
    precheck_complete = threading.Event()
    claimant_committed = threading.Event()
    captured_paths: list[str] = []
    errors: list[BaseException] = []
    real_claimed = refresh_validation._proof_path_claimed

    def pause_after_clear_precheck(relative_path: str) -> bool:
        claimed = real_claimed(relative_path)
        if not captured_paths:
            assert claimed is False
            captured_paths.append(relative_path)
            precheck_complete.set()
            assert claimant_committed.wait(timeout=10)
        return claimed

    monkeypatch.setattr(
        refresh_validation,
        "_proof_path_claimed",
        pause_after_clear_precheck,
    )

    def run_child() -> None:
        connections.close_all()
        try:
            execute_portfolio_snapshot_job(
                target_date=TARGET,
                require_session_date=True,
                require_all=True,
            )
        except BaseException as exc:  # pragma: no cover - asserted in caller
            errors.append(exc)
        finally:
            connections.close_all()

    caplog.set_level("ERROR")
    child_thread = threading.Thread(target=run_child)
    child_thread.start()
    assert precheck_complete.wait(timeout=10)
    relative_path = captured_paths[0]
    now = timezone.now()
    claimant = DataAsset.objects.create(
        provider="concurrent-owner",
        kind="unrelated-proof",
        subject=str(uuid4()),
        relative_path=relative_path,
        sha256=hashlib.sha256(b"different concurrent claimant").hexdigest(),
        retrieved_at=now,
        available_at=now,
    )
    claimant_committed.set()
    child_thread.join(timeout=10)

    assert not child_thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RefreshVerificationError)
    assert errors[0].reason_code == "portfolio_verification_path_registered"
    failed_child = JobRun.objects.get(job_name="scheduled_portfolio_snapshots")
    proof_path = AssetStore(tmp_path).resolve(relative_path)
    assert failed_child.status == JobRun.Status.FAILED
    assert DataAsset.objects.filter(pk=claimant.pk).exists()
    assert not proof_path.exists()
    for leaked_path in (relative_path, str(proof_path), str(tmp_path)):
        assert leaked_path not in str(errors[0])
        assert leaked_path not in failed_child.error
        assert leaked_path not in caplog.text


def test_post_write_cleanup_failure_preserves_original_error_and_orphan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    Portfolio.objects.create(
        owner=_owner("cleanup-failure"),
        name="Cleanup failure",
        base_currency="USD",
        cash_balance=Decimal("10"),
    )
    child = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET,
    )
    batch = snapshot_all_portfolios(expected_as_of_date=TARGET)
    real_write = AssetStore.write_bytes

    def write_then_misreport(
        self: AssetStore,
        relative_path: str,
        payload: bytes,
    ) -> Any:
        stored = real_write(self, relative_path, payload)
        return dataclasses.replace(
            stored,
            byte_count=stored.byte_count + 1,
        )

    def fail_cleanup(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(AssetStore, "write_bytes", write_then_misreport)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(RefreshVerificationError) as excinfo:
        attest_scheduled_portfolio_snapshots(
            child_run=child,
            target_date=TARGET,
            report_details=_batch_details(batch),
            snapshot_actions=batch._snapshot_actions,
            validator_revision=REVISION,
        )

    proof_path = tmp_path / f"portfolio/verification/{child.pk}.json"
    assert excinfo.value.reason_code == "portfolio_verification_write_failed"
    assert str(excinfo.value) == "The portfolio verification proof could not be written"
    assert proof_path.exists()
    assert not DataAsset.objects.filter(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(child.pk),
    ).exists()


def test_outer_commit_failure_rolls_back_row_and_cleans_attempt_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    Portfolio.objects.create(
        owner=_owner("commit-failure"),
        name="Commit failure",
        base_currency="USD",
        cash_balance=Decimal("10"),
    )
    child = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET,
    )
    batch = snapshot_all_portfolios(expected_as_of_date=TARGET)
    original = DatabaseError("sentinel outer commit failure")
    real_create = DataAsset.objects.create
    real_savepoint_commit = connection.savepoint_commit
    armed = False
    proof_savepoint_commits = 0

    def arm_commit(*args: object, **kwargs: object) -> DataAsset:
        nonlocal armed
        asset = real_create(*args, **kwargs)
        if kwargs.get("kind") == PORTFOLIO_VERIFICATION_KIND:
            armed = True
        return asset

    def fail_outer_savepoint_commit(sid: str) -> None:
        nonlocal armed, proof_savepoint_commits
        if armed:
            proof_savepoint_commits += 1
            if proof_savepoint_commits == 2:
                armed = False
                raise original
        real_savepoint_commit(sid)

    monkeypatch.setattr(DataAsset.objects, "create", arm_commit)
    monkeypatch.setattr(connection, "savepoint_commit", fail_outer_savepoint_commit)
    with pytest.raises(DatabaseError) as excinfo:
        attest_scheduled_portfolio_snapshots(
            child_run=child,
            target_date=TARGET,
            report_details=_batch_details(batch),
            snapshot_actions=batch._snapshot_actions,
            validator_revision=REVISION,
        )

    assert excinfo.value is original
    assert not DataAsset.objects.filter(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(child.pk),
    ).exists()
    assert not (tmp_path / f"portfolio/verification/{child.pk}.json").exists()


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL blocked unique-path claimant coverage",
)
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize(
    "same_checksum",
    [False, True],
    ids=["different-checksum-removes-attempt-bytes", "same-checksum-adopts-attempt-bytes"],
)
def test_postgresql_cleanup_share_lock_reconciles_blocked_path_claimant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    same_checksum: bool,
) -> None:
    """Cleanup waits for a path claimant that was already blocked behind A.

    PostgreSQL proves both sides of the serialization: claimant B first holds
    ``ROW EXCLUSIVE`` and waits on A's uncommitted unique-path insertion.
    After A's forced rollback, B's insertion returns but its transaction stays
    open, and cleanup's ``SHARE`` table lock is observed waiting on B. Releasing
    B lets cleanup reconcile the now-visible claimant without deadlock.
    """

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    Portfolio.objects.create(
        owner=_owner(f"blocked-claimant-{same_checksum}"),
        name=f"Blocked claimant {same_checksum}",
        base_currency="USD",
        cash_balance=Decimal("10"),
    )

    bytes_written = threading.Event()
    claimant_insert_started = threading.Event()
    allow_attempt_rollback = threading.Event()
    claimant_inserted = threading.Event()
    allow_claimant_commit = threading.Event()
    claimant_committed = threading.Event()
    cleanup_started = threading.Event()
    cleanup_finished = threading.Event()
    captured: dict[str, str] = {}
    attempt_errors: list[BaseException] = []
    claimant_errors: list[BaseException] = []
    claimant_ids: list[UUID] = []
    claimant_backend_pids: list[int] = []
    cleanup_backend_pids: list[int] = []
    claimant_validated_sha256: list[str] = []
    original = DatabaseError("sentinel proof commit failure")
    real_write = AssetStore.write_bytes
    real_cleanup = refresh_validation._cleanup_attempt_file
    real_commit = BaseDatabaseWrapper._commit
    commit_failed = False
    attempt_thread: threading.Thread | None = None

    def write_then_release_claimant(
        self: AssetStore,
        relative_path: str,
        payload: bytes,
    ) -> Any:
        stored = real_write(self, relative_path, payload)
        if threading.current_thread() is attempt_thread and relative_path.startswith(
            "portfolio/verification/"
        ):
            captured["relative_path"] = relative_path
            captured["attempt_sha256"] = stored.sha256
            bytes_written.set()
            assert allow_attempt_rollback.wait(timeout=10)
        return stored

    def observe_real_cleanup(
        store: AssetStore,
        relative_path: str,
        *,
        attempt_sha256: str,
    ) -> None:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            row = cursor.fetchone()
        assert row is not None
        cleanup_backend_pids.append(int(row[0]))
        cleanup_started.set()
        try:
            real_cleanup(
                store,
                relative_path,
                attempt_sha256=attempt_sha256,
            )
        finally:
            cleanup_finished.set()

    def fail_attempt_commit(self: BaseDatabaseWrapper) -> None:
        nonlocal commit_failed
        if (
            threading.current_thread() is attempt_thread
            and bytes_written.is_set()
            and not commit_failed
        ):
            commit_failed = True
            raise original
        real_commit(self)

    monkeypatch.setattr(AssetStore, "write_bytes", write_then_release_claimant)
    monkeypatch.setattr(
        refresh_validation,
        "_cleanup_attempt_file",
        observe_real_cleanup,
    )
    monkeypatch.setattr(BaseDatabaseWrapper, "_commit", fail_attempt_commit)

    def run_attempt() -> None:
        connections.close_all()
        try:
            execute_portfolio_snapshot_job(
                target_date=TARGET,
                require_session_date=True,
                require_all=True,
            )
        except BaseException as exc:  # pragma: no cover - asserted in caller
            attempt_errors.append(exc)
        finally:
            connections.close_all()

    def run_claimant() -> None:
        connections.close_all()
        try:
            assert bytes_written.wait(timeout=10)
            relative_path = captured["relative_path"]
            attempt_sha256 = captured["attempt_sha256"]
            if same_checksum:
                proof_path = AssetStore(tmp_path).resolve(relative_path)
                stored = AssetStore(tmp_path).write_bytes(
                    relative_path,
                    proof_path.read_bytes(),
                )
                assert stored.sha256 == attempt_sha256
                claimant_validated_sha256.append(stored.sha256)
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                row = cursor.fetchone()
            assert row is not None
            claimant_backend_pids.append(int(row[0]))
            now = timezone.now()
            with transaction.atomic():
                claimant_insert_started.set()
                claimant = DataAsset.objects.create(
                    provider="concurrent-owner",
                    kind="unrelated-proof",
                    subject=str(uuid4()),
                    relative_path=relative_path,
                    sha256=(
                        attempt_sha256
                        if same_checksum
                        else hashlib.sha256(b"different claimant bytes").hexdigest()
                    ),
                    retrieved_at=now,
                    available_at=now,
                )
                claimant_ids.append(claimant.pk)
                claimant_inserted.set()
                assert allow_claimant_commit.wait(timeout=10)
            claimant_committed.set()
        except BaseException as exc:  # pragma: no cover - asserted in caller
            claimant_errors.append(exc)
        finally:
            connections.close_all()

    caplog.set_level("ERROR")
    attempt_thread = threading.Thread(target=run_attempt)
    claimant_thread = threading.Thread(target=run_claimant)
    attempt_thread.start()
    claimant_thread.start()
    try:
        assert bytes_written.wait(timeout=10)
        assert claimant_insert_started.wait(timeout=10)
        assert len(claimant_backend_pids) == 1
        claimant_pid = claimant_backend_pids[0]
        table_name = DataAsset._meta.db_table
        _wait_for_postgresql_lock(
            claimant_pid,
            locktype="relation",
            mode="RowExclusiveLock",
            granted=True,
            relation=table_name,
        )
        _wait_for_postgresql_lock(
            claimant_pid,
            locktype="transactionid",
            mode="ShareLock",
            granted=False,
        )

        # A now fails its outer commit and rolls back. B's INSERT can finish,
        # but B remains deliberately uncommitted while real cleanup starts.
        allow_attempt_rollback.set()
        assert claimant_inserted.wait(timeout=10)
        assert cleanup_started.wait(timeout=10)
        assert len(cleanup_backend_pids) == 1
        _wait_for_postgresql_lock(
            cleanup_backend_pids[0],
            locktype="relation",
            mode="ShareLock",
            granted=False,
            relation=table_name,
        )
        proof_path = AssetStore(tmp_path).resolve(captured["relative_path"])
        assert proof_path.exists()
        assert not cleanup_finished.wait(timeout=0.2)
        assert not claimant_committed.is_set()

        allow_claimant_commit.set()
    finally:
        allow_attempt_rollback.set()
        allow_claimant_commit.set()
        attempt_thread.join(timeout=20)
        claimant_thread.join(timeout=20)

    assert not attempt_thread.is_alive()
    assert not claimant_thread.is_alive()
    assert cleanup_started.is_set()
    assert cleanup_finished.is_set()
    assert claimant_committed.is_set()
    assert commit_failed
    assert claimant_errors == []
    assert len(attempt_errors) == 1
    assert attempt_errors[0] is original
    assert len(claimant_ids) == 1

    claimant = DataAsset.objects.get(pk=claimant_ids[0])
    failed_child = JobRun.objects.get(job_name="scheduled_portfolio_snapshots")
    relative_path = captured["relative_path"]
    proof_path = AssetStore(tmp_path).resolve(relative_path)
    assert failed_child.status == JobRun.Status.FAILED
    assert failed_child.error == "DatabaseError: sentinel proof commit failure"
    if same_checksum:
        assert claimant.sha256 == captured["attempt_sha256"]
        assert claimant_validated_sha256 == [claimant.sha256]
        assert hashlib.sha256(proof_path.read_bytes()).hexdigest() == claimant.sha256
    else:
        assert claimant_validated_sha256 == []
        assert claimant.sha256 != captured["attempt_sha256"]
        assert not proof_path.exists()
    for leaked_path in (relative_path, str(proof_path), str(tmp_path)):
        assert leaked_path not in str(attempt_errors[0])
        assert leaked_path not in failed_child.error
        assert leaked_path not in caplog.text


def test_unrelated_proof_registration_integrity_error_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    Portfolio.objects.create(
        owner=_owner("unrelated-integrity"),
        name="Unrelated integrity",
        base_currency="USD",
        cash_balance=Decimal("10"),
    )
    unrelated = IntegrityError("synthetic unrelated constraint failure")
    real_create = DataAsset.objects.create

    def fail_unrelated(*args: object, **kwargs: object) -> DataAsset:
        if kwargs.get("kind") == PORTFOLIO_VERIFICATION_KIND:
            raise unrelated
        return real_create(*args, **kwargs)

    monkeypatch.setattr(DataAsset.objects, "create", fail_unrelated)
    with pytest.raises(IntegrityError) as excinfo:
        execute_portfolio_snapshot_job(
            target_date=TARGET,
            require_session_date=True,
            require_all=True,
        )

    assert excinfo.value is unrelated
    child = JobRun.objects.get(job_name="scheduled_portfolio_snapshots")
    assert child.status == JobRun.Status.FAILED
    assert not DataAsset.objects.filter(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(child.pk),
    ).exists()
    assert not (tmp_path / f"portfolio/verification/{child.pk}.json").exists()


def test_identical_orphan_registers_and_different_preexisting_bytes_survive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    Portfolio.objects.create(
        owner=_owner(),
        name="Orphan registration",
        base_currency="USD",
    )
    child = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET,
    )
    batch = snapshot_all_portfolios(expected_as_of_date=TARGET)
    details = _batch_details(batch)
    real_write = AssetStore.write_bytes
    real_unlink = Path.unlink

    def leave_identical_orphan(
        self: AssetStore,
        relative_path: str,
        payload: bytes,
    ) -> Any:
        stored = real_write(self, relative_path, payload)
        if relative_path == f"portfolio/verification/{child.pk}.json":
            raise RuntimeError("leave identical orphan")
        return stored

    def fail_cleanup(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("cleanup unavailable")

    monkeypatch.setattr(AssetStore, "write_bytes", leave_identical_orphan)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(RuntimeError, match="leave identical orphan"):
        attest_scheduled_portfolio_snapshots(
            child_run=child,
            target_date=TARGET,
            report_details=details,
            snapshot_actions=batch._snapshot_actions,
            validator_revision=REVISION,
        )
    proof_path = tmp_path / f"portfolio/verification/{child.pk}.json"
    orphan_bytes = proof_path.read_bytes()

    monkeypatch.setattr(AssetStore, "write_bytes", real_write)
    monkeypatch.setattr(Path, "unlink", real_unlink)
    asset = attest_scheduled_portfolio_snapshots(
        child_run=child,
        target_date=TARGET,
        report_details=details,
        snapshot_actions=batch._snapshot_actions,
        validator_revision=REVISION,
    )
    assert proof_path.read_bytes() == orphan_bytes
    assert asset.relative_path == f"portfolio/verification/{child.pk}.json"

    other_child = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET + timedelta(days=1),
    )
    other_batch = snapshot_all_portfolios(expected_as_of_date=TARGET + timedelta(days=1))
    conflicting_path = tmp_path / f"portfolio/verification/{other_child.pk}.json"
    conflicting_path.parent.mkdir(parents=True, exist_ok=True)
    conflicting_path.write_bytes(b"different preexisting bytes")
    with pytest.raises(
        RefreshVerificationError,
        match="could not be written",
    ):
        attest_scheduled_portfolio_snapshots(
            child_run=other_child,
            target_date=TARGET + timedelta(days=1),
            report_details={
                **_batch_details(other_batch),
                "required_session_date": (TARGET + timedelta(days=1)).isoformat(),
            },
            snapshot_actions=other_batch._snapshot_actions,
            validator_revision=REVISION,
        )
    assert conflicting_path.read_bytes() == b"different preexisting bytes"


def test_detail_file_and_ambiguous_proof_tampering_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _build_state(monkeypatch, tmp_path)
    asset = DataAsset.objects.get(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(state.child.pk),
    )
    path = AssetStore(tmp_path).resolve(asset.relative_path)
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(RefreshVerificationError):
        verify_portfolio_snapshot_stage(state.child, target_date=TARGET)
    path.write_bytes(original)
    verify_portfolio_snapshot_stage(state.child, target_date=TARGET)

    details = dict(state.child.details)
    details["verification_sha256"] = "0" * 64
    JobRun.objects.filter(pk=state.child.pk).update(details=details)
    state.child.refresh_from_db()
    with pytest.raises(
        RefreshVerificationError,
        match="identity or timing",
    ):
        verify_portfolio_snapshot_stage(state.child, target_date=TARGET)
    details["verification_sha256"] = asset.sha256
    JobRun.objects.filter(pk=state.child.pk).update(details=details)
    state.child.refresh_from_db()

    duplicate_bytes = b'{"duplicate":"proof identity"}'
    stored = AssetStore(tmp_path).write_bytes(
        f"portfolio/verification/duplicate-{uuid4()}.json",
        duplicate_bytes,
    )
    DataAsset.objects.create(
        provider="stanstock",
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(state.child.pk),
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        retrieved_at=asset.retrieved_at,
        available_at=asset.available_at,
        period_start=TARGET,
        period_end=TARGET,
    )
    with pytest.raises(
        RefreshVerificationError,
        match="exactly one",
    ):
        verify_portfolio_snapshot_stage(state.child, target_date=TARGET)


def test_scheduled_failure_is_count_only_while_manual_failure_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    private_name = "PRIVATE BROKEN PORTFOLIO"
    private_ticker = "PRIVATE-TICKER"
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name=private_name,
        base_currency="USD",
    )
    company = Company.objects.create(name="Private Company", country="US")
    listing = Listing.objects.create(
        security=Security.objects.create(company=company),
        ticker=private_ticker,
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
        is_active=False,
    )
    PortfolioHolding.objects.create(
        portfolio=portfolio,
        listing=listing,
        quantity=Decimal("1"),
        average_cost=Decimal("2"),
    )

    with pytest.raises(
        ValueError,
        match=r"^Scheduled portfolio snapshots failed for 1 active portfolio\(s\)$",
    ):
        execute_portfolio_snapshot_job(
            target_date=TARGET,
            require_session_date=True,
            require_all=True,
        )
    scheduled = JobRun.objects.get(
        job_name="scheduled_portfolio_snapshots",
        target_date=TARGET,
    )
    assert private_name not in scheduled.error
    assert private_ticker not in scheduled.error
    assert str(tmp_path) not in scheduled.error

    with pytest.raises(ValueError, match=private_name):
        execute_portfolio_snapshot_job(
            target_date=TARGET + timedelta(days=1),
            require_session_date=False,
            require_all=True,
        )


def test_holding_lock_query_overrides_join_ordering() -> None:
    queryset = (
        PortfolioHolding.objects.select_for_update(of=("self",))
        .filter(portfolio_id__in=(uuid4(),))
        .order_by("pk")
    )
    with transaction.atomic():
        sql = str(queryset.query).upper()

    assert " JOIN " not in sql
    assert queryset.query.order_by == ("pk",)


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL row-lock interleaving coverage",
)
@pytest.mark.django_db(transaction=True)
def test_postgresql_existing_mutation_waits_until_after_final_observation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Postgres lock",
        base_currency="USD",
        cash_balance=Decimal("10"),
    )
    child = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET,
    )
    batch = snapshot_all_portfolios(expected_as_of_date=TARGET)
    details = _batch_details(batch)
    reached_final = threading.Event()
    release_final = threading.Event()
    mutation_done = threading.Event()
    errors: list[BaseException] = []
    real_final = refresh_validation._fresh_input_signatures

    def paused_final(**kwargs: object) -> tuple[tuple[object, ...], ...]:
        reached_final.set()
        assert release_final.wait(timeout=10)
        return real_final(**kwargs)

    monkeypatch.setattr(
        refresh_validation,
        "_fresh_input_signatures",
        paused_final,
    )

    def attest() -> None:
        connections.close_all()
        try:
            attest_scheduled_portfolio_snapshots(
                child_run=child,
                target_date=TARGET,
                report_details=details,
                snapshot_actions=batch._snapshot_actions,
                validator_revision=REVISION,
            )
        except BaseException as exc:  # pragma: no cover - asserted in caller
            errors.append(exc)
        finally:
            connections.close_all()

    def mutate() -> None:
        connections.close_all()
        try:
            with transaction.atomic():
                locked = Portfolio.objects.select_for_update().get(pk=portfolio.pk)
                locked.cash_balance = Decimal("11")
                locked.save(update_fields=["cash_balance", "updated_at"])
            mutation_done.set()
        except BaseException as exc:  # pragma: no cover - asserted in caller
            errors.append(exc)
        finally:
            connections.close_all()

    attester = threading.Thread(target=attest)
    attester.start()
    assert reached_final.wait(timeout=10)
    mutator = threading.Thread(target=mutate)
    mutator.start()
    assert not mutation_done.wait(timeout=0.25)
    release_final.set()
    attester.join(timeout=10)
    mutator.join(timeout=10)

    assert not attester.is_alive()
    assert not mutator.is_alive()
    assert errors == []
    assert mutation_done.is_set()
    assert (
        DataAsset.objects.filter(
            kind=PORTFOLIO_VERIFICATION_KIND,
            subject=str(child.pk),
        ).count()
        == 1
    )


@pytest.mark.parametrize(
    "mutation_target",
    ["portfolio_holding", "listing", "latest_market_data"],
)
@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL mutable-input row-lock interleaving coverage",
)
@pytest.mark.django_db(transaction=True)
def test_postgresql_mutable_input_mutation_waits_until_after_final_observation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation_target: str,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name=f"Postgres {mutation_target} lock",
        base_currency="USD",
    )
    listing, _normalized, _raw = _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=portfolio,
        ticker="PGLOCK",
        currency="USD",
        session_date=TARGET,
        metadata_currency="USD",
    )
    child = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET,
    )
    batch = snapshot_all_portfolios(expected_as_of_date=TARGET)
    details = _batch_details(batch)
    reached_final = threading.Event()
    release_final = threading.Event()
    mutation_done = threading.Event()
    errors: list[BaseException] = []
    real_final = refresh_validation._fresh_input_signatures

    def paused_final(**kwargs: object) -> tuple[tuple[object, ...], ...]:
        reached_final.set()
        assert release_final.wait(timeout=10)
        return real_final(**kwargs)

    monkeypatch.setattr(
        refresh_validation,
        "_fresh_input_signatures",
        paused_final,
    )

    def attest() -> None:
        connections.close_all()
        try:
            attest_scheduled_portfolio_snapshots(
                child_run=child,
                target_date=TARGET,
                report_details=details,
                snapshot_actions=batch._snapshot_actions,
                validator_revision=REVISION,
            )
        except BaseException as exc:  # pragma: no cover - asserted in caller
            errors.append(exc)
        finally:
            connections.close_all()

    def mutate() -> None:
        connections.close_all()
        try:
            with transaction.atomic():
                if mutation_target == "portfolio_holding":
                    row = PortfolioHolding.objects.select_for_update().get(
                        portfolio=portfolio,
                        listing=listing,
                    )
                    row.quantity = Decimal("9")
                    row.save(update_fields=["quantity", "updated_at"])
                elif mutation_target == "listing":
                    listing_row = Listing.objects.select_for_update().get(pk=listing.pk)
                    listing_row.provider_symbol = "PGLOCK:CHANGED"
                    listing_row.save(update_fields=["provider_symbol"])
                else:
                    market = LatestMarketData.objects.select_for_update().get(
                        listing=listing,
                    )
                    market.close = Decimal("104")
                    market.save(update_fields=["close"])
            mutation_done.set()
        except BaseException as exc:  # pragma: no cover - asserted in caller
            errors.append(exc)
        finally:
            connections.close_all()

    attester = threading.Thread(target=attest)
    attester.start()
    assert reached_final.wait(timeout=10)
    mutator = threading.Thread(target=mutate)
    mutator.start()
    assert not mutation_done.wait(timeout=0.25)
    release_final.set()
    attester.join(timeout=10)
    mutator.join(timeout=10)

    assert not attester.is_alive()
    assert not mutator.is_alive()
    assert errors == []
    assert mutation_done.is_set()
    assert (
        DataAsset.objects.filter(
            kind=PORTFOLIO_VERIFICATION_KIND,
            subject=str(child.pk),
        ).count()
        == 1
    )


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL portfolio-first lock-order coverage",
)
@pytest.mark.django_db(transaction=True)
def test_postgresql_portfolio_first_writer_and_attester_do_not_deadlock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Portfolio-first lock",
        base_currency="USD",
    )
    child = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET,
    )
    batch = snapshot_all_portfolios(expected_as_of_date=TARGET)
    details = _batch_details(batch)
    portfolio_locked = threading.Event()
    release_writer = threading.Event()
    attester_done = threading.Event()
    errors: list[BaseException] = []

    def portfolio_first_writer() -> None:
        connections.close_all()
        try:
            with transaction.atomic():
                Portfolio.objects.select_for_update().get(pk=portfolio.pk)
                Portfolio.objects.filter(pk=portfolio.pk).update(updated_at=timezone.now())
                portfolio_locked.set()
                assert release_writer.wait(timeout=10)
        except BaseException as exc:  # pragma: no cover - asserted in caller
            errors.append(exc)
        finally:
            connections.close_all()

    def attest() -> None:
        connections.close_all()
        try:
            attest_scheduled_portfolio_snapshots(
                child_run=child,
                target_date=TARGET,
                report_details=details,
                snapshot_actions=batch._snapshot_actions,
                validator_revision=REVISION,
            )
            attester_done.set()
        except BaseException as exc:  # pragma: no cover - asserted in caller
            errors.append(exc)
        finally:
            connections.close_all()

    writer = threading.Thread(target=portfolio_first_writer)
    writer.start()
    assert portfolio_locked.wait(timeout=10)
    attester = threading.Thread(target=attest)
    attester.start()
    assert not attester_done.wait(timeout=0.25)
    release_writer.set()
    writer.join(timeout=10)
    attester.join(timeout=10)

    assert not writer.is_alive()
    assert not attester.is_alive()
    assert errors == []
    assert attester_done.is_set()


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL phantom-insert final recheck coverage",
)
@pytest.mark.django_db(transaction=True)
def test_postgresql_insert_committed_before_final_recheck_is_detected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    Portfolio.objects.create(
        owner=_owner(),
        name="Initial active",
        base_currency="USD",
    )
    child = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET,
    )
    batch = snapshot_all_portfolios(expected_as_of_date=TARGET)
    details = _batch_details(batch)
    reached_final = threading.Event()
    insertion_done = threading.Event()
    errors: list[BaseException] = []
    real_final = refresh_validation._fresh_input_signatures

    def paused_final(**kwargs: object) -> tuple[tuple[object, ...], ...]:
        reached_final.set()
        assert insertion_done.wait(timeout=10)
        return real_final(**kwargs)

    monkeypatch.setattr(
        refresh_validation,
        "_fresh_input_signatures",
        paused_final,
    )

    def attest() -> None:
        connections.close_all()
        try:
            attest_scheduled_portfolio_snapshots(
                child_run=child,
                target_date=TARGET,
                report_details=details,
                snapshot_actions=batch._snapshot_actions,
                validator_revision=REVISION,
            )
        except BaseException as exc:  # pragma: no cover - asserted in caller
            errors.append(exc)
        finally:
            connections.close_all()

    thread = threading.Thread(target=attest)
    thread.start()
    assert reached_final.wait(timeout=10)
    Portfolio.objects.create(
        owner=_owner("inserted-before-final"),
        name="Inserted active",
        base_currency="USD",
    )
    insertion_done.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RefreshVerificationError)
    assert not DataAsset.objects.filter(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(child.pk),
    ).exists()


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL immutable-position phantom closure coverage",
)
@pytest.mark.django_db(transaction=True)
def test_postgresql_position_append_committed_before_final_observation_is_detected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Position closure",
        base_currency="USD",
    )
    _listing, _normalized, _raw = _add_priced_holding(
        tmp_path=tmp_path,
        portfolio=portfolio,
        ticker="PGPOSITION",
        currency="USD",
        session_date=TARGET,
        metadata_currency="USD",
    )
    extra_listing = Listing.objects.create(
        security=Security.objects.create(
            company=Company.objects.create(name="Position Phantom", country="US")
        ),
        ticker="PGPHANTOM",
        provider_symbol="PGPHANTOM:PROVIDER",
        exchange_mic="XNAS",
        currency="USD",
        region=Region.US,
    )
    child = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET,
    )
    batch = snapshot_all_portfolios(expected_as_of_date=TARGET)
    details = _batch_details(batch)
    snapshot = PortfolioSnapshot.objects.get(pk=dict(batch.snapshot_ids)[str(portfolio.pk)])
    existing = snapshot.positions.get()
    reached_closure = threading.Event()
    insertion_done = threading.Event()
    errors: list[BaseException] = []
    real_closure = refresh_validation._snapshot_position_identities

    def paused_closure(
        snapshot_ids: tuple[UUID, ...],
    ) -> tuple[tuple[UUID, int, UUID], ...]:
        reached_closure.set()
        assert insertion_done.wait(timeout=10)
        return real_closure(snapshot_ids)

    monkeypatch.setattr(
        refresh_validation,
        "_snapshot_position_identities",
        paused_closure,
    )

    def attest() -> None:
        connections.close_all()
        try:
            attest_scheduled_portfolio_snapshots(
                child_run=child,
                target_date=TARGET,
                report_details=details,
                snapshot_actions=batch._snapshot_actions,
                validator_revision=REVISION,
            )
        except BaseException as exc:  # pragma: no cover - asserted in caller
            errors.append(exc)
        finally:
            connections.close_all()

    thread = threading.Thread(target=attest)
    thread.start()
    assert reached_closure.wait(timeout=10)
    appended = PortfolioSnapshotHolding.objects.create(
        snapshot=snapshot,
        listing=extra_listing,
        source_asset_id=existing.source_asset_id,
        source_session_date=existing.source_session_date,
        quantity=Decimal("1"),
        average_cost=Decimal("10"),
        price=Decimal("12"),
        cost_basis=Decimal("10"),
        market_value=Decimal("12"),
        unrealized_gain=Decimal("2"),
        corporate_action_suspected=False,
    )
    insertion_done.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RefreshVerificationError)
    assert errors[0].reason_code == "portfolio_snapshot_position_closure_changed"
    assert PortfolioSnapshotHolding.objects.filter(pk=appended.pk).exists()
    assert not DataAsset.objects.filter(
        kind=PORTFOLIO_VERIFICATION_KIND,
        subject=str(child.pk),
    ).exists()


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL post-start concurrent snapshot reuse coverage",
)
@pytest.mark.django_db(transaction=True)
def test_postgresql_post_start_concurrent_snapshot_is_reused_with_current_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    portfolio = Portfolio.objects.create(
        owner=_owner(),
        name="Concurrent snapshot reuse",
        base_currency="USD",
        cash_balance=Decimal("10"),
    )
    portfolio.refresh_from_db()
    child_entered_service = threading.Event()
    manual_snapshot_done = threading.Event()
    proof_registered = threading.Event()
    release_child = threading.Event()
    errors: list[BaseException] = []
    results: list[JobRun] = []
    real_snapshot_all = portfolio_jobs.snapshot_all_portfolios
    real_attest = portfolio_jobs.attest_scheduled_portfolio_snapshots

    def paused_snapshot_all(**kwargs: object) -> PortfolioSnapshotBatch:
        child_entered_service.set()
        assert manual_snapshot_done.wait(timeout=10)
        return real_snapshot_all(**kwargs)

    def paused_after_proof(**kwargs: object) -> DataAsset:
        asset = real_attest(**kwargs)
        proof_registered.set()
        assert release_child.wait(timeout=10)
        return asset

    monkeypatch.setattr(
        portfolio_jobs,
        "snapshot_all_portfolios",
        paused_snapshot_all,
    )
    monkeypatch.setattr(
        portfolio_jobs,
        "attest_scheduled_portfolio_snapshots",
        paused_after_proof,
    )

    def run_child() -> None:
        connections.close_all()
        try:
            results.append(
                execute_portfolio_snapshot_job(
                    target_date=TARGET,
                    require_session_date=True,
                    require_all=True,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted in caller
            errors.append(exc)
        finally:
            connections.close_all()

    thread = threading.Thread(target=run_child)
    thread.start()
    assert child_entered_service.wait(timeout=10)
    running_child = JobRun.objects.get(
        job_name="scheduled_portfolio_snapshots",
        target_date=TARGET,
    )
    snapshot, created = record_portfolio_snapshot(
        portfolio,
        expected_as_of_date=TARGET,
    )
    assert created is True
    assert snapshot.recorded_at >= running_child.started_at
    assert snapshot.code_revision == REVISION
    manual_snapshot_done.set()
    assert proof_registered.wait(timeout=10)

    running_child.refresh_from_db()
    assert running_child.status == JobRun.Status.RUNNING
    assert (
        DataAsset.objects.filter(
            kind=PORTFOLIO_VERIFICATION_KIND,
            subject=str(running_child.pk),
        ).count()
        == 1
    )
    release_child.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert errors == []
    assert len(results) == 1
    child = results[0]
    assert child.status == JobRun.Status.SUCCESS
    assert child.details["snapshots_created"] == 0
    assert child.details["snapshots_unchanged"] == 1
    proof = parse_portfolio_verification_proof(
        _proof_bytes(
            ProofState(
                portfolio=portfolio,
                listings=(),
                normalized_assets=(),
                raw_assets=(),
                child=child,
            ),
            tmp_path,
        )
    )
    assert proof.snapshots[0].classification == "reused"
    assert proof.validator_revision == REVISION
    assert proof.snapshots[0].snapshot_id == snapshot.pk
    verify_portfolio_snapshot_stage(child, target_date=TARGET)


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="PostgreSQL post-observation insertion coverage",
)
@pytest.mark.django_db(transaction=True)
def test_postgresql_insert_after_final_observation_is_later_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", REVISION)
    Portfolio.objects.create(
        owner=_owner(),
        name="Linearized active",
        base_currency="USD",
    )
    child = JobRun.objects.create(
        job_name="scheduled_portfolio_snapshots",
        region="",
        target_date=TARGET,
    )
    batch = snapshot_all_portfolios(expected_as_of_date=TARGET)
    details = _batch_details(batch)
    final_observed = threading.Event()
    insertion_done = threading.Event()
    errors: list[BaseException] = []
    real_final = refresh_validation._fresh_input_signatures

    def pause_after_observation(
        **kwargs: object,
    ) -> tuple[tuple[object, ...], ...]:
        observed = real_final(**kwargs)
        final_observed.set()
        assert insertion_done.wait(timeout=10)
        return observed

    monkeypatch.setattr(
        refresh_validation,
        "_fresh_input_signatures",
        pause_after_observation,
    )

    def attest() -> None:
        connections.close_all()
        try:
            attest_scheduled_portfolio_snapshots(
                child_run=child,
                target_date=TARGET,
                report_details=details,
                snapshot_actions=batch._snapshot_actions,
                validator_revision=REVISION,
            )
        except BaseException as exc:  # pragma: no cover - asserted in caller
            errors.append(exc)
        finally:
            connections.close_all()

    thread = threading.Thread(target=attest)
    thread.start()
    assert final_observed.wait(timeout=10)
    Portfolio.objects.create(
        owner=_owner("inserted-after-final"),
        name="Later active",
        base_currency="USD",
    )
    insertion_done.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert errors == []
    assert (
        DataAsset.objects.filter(
            kind=PORTFOLIO_VERIFICATION_KIND,
            subject=str(child.pk),
        ).count()
        == 1
    )

"""Immutable scheduled-refresh evidence for portfolio snapshots.

The scheduled portfolio child writes one private, canonical proof after it
has replayed the snapshots against locked mutable inputs.  Parent refresh
verification later reads only that proof, immutable snapshot rows, and the
physical source assets named by those rows; it deliberately does not consult
current portfolio, holding, listing, or latest-market state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Context, Decimal, DecimalException, localcontext
from typing import Any
from uuid import UUID

from django.db import DatabaseError, IntegrityError, connection, transaction
from django.db.models import DecimalField
from django.utils import timezone

from stanstock.core.models import JobRun
from stanstock.core.verification_types import (
    AssetRef,
    RefreshVerificationError,
    StageVerificationResult,
)
from stanstock.data.asof import verified_price_fields
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    open_asset_store,
    read_checksummed_bytes,
)
from stanstock.data.models import DataAsset, LatestMarketData, Listing
from stanstock.data.provider_policy import PRIVATE_USAGE_SCOPE
from stanstock.portfolio.models import (
    Portfolio,
    PortfolioHolding,
    PortfolioSnapshot,
    PortfolioSnapshotHolding,
)
from stanstock.portfolio.service import (
    MAX_PORTFOLIO_PRICE_AGE_DAYS,
    SPLIT_WARNING_HIGH_RATIO,
    SPLIT_WARNING_LOW_RATIO,
)

PORTFOLIO_VERIFICATION_CONTRACT = "portfolio-snapshot-verification@1"
PORTFOLIO_VERIFICATION_KIND = "portfolio_snapshot_verification"
PORTFOLIO_VERIFICATION_PROVIDER = "stanstock"
PORTFOLIO_VERIFICATION_SCHEMA = "1"

PORTFOLIO_SNAPSHOT_MODEL = "PortfolioSnapshot"
PORTFOLIO_SNAPSHOT_HOLDING_MODEL = "PortfolioSnapshotHolding"

# These tuples are deliberately lexical and complete.  A focused field
# coverage test compares them to ``_meta.concrete_fields`` so a migration
# cannot silently leave a new private value outside the digest.
PORTFOLIO_SNAPSHOT_FIELDS: tuple[str, ...] = (
    "as_of_date",
    "base_currency",
    "cash_balance",
    "code_revision",
    "corporate_action_warnings",
    "cost_basis",
    "dividends_included",
    "id",
    "input_hash",
    "newest_price_date",
    "oldest_price_date",
    "portfolio_id",
    "recorded_at",
    "return_definition",
    "return_pct",
    "securities_value",
    "total_value",
    "unrealized_gain",
)
PORTFOLIO_SNAPSHOT_HOLDING_FIELDS: tuple[str, ...] = (
    "average_cost",
    "corporate_action_suspected",
    "cost_basis",
    "id",
    "listing_id",
    "market_value",
    "price",
    "quantity",
    "snapshot_id",
    "source_asset_id",
    "source_session_date",
    "unrealized_gain",
)

BASE_DETAIL_KEYS = frozenset(
    {
        "portfolios",
        "snapshots_created",
        "snapshots_unchanged",
        "snapshot_ids",
        "failures",
        "required_session_date",
        "require_all",
        "reason",
    }
)
VERIFIED_DETAIL_KEYS = BASE_DETAIL_KEYS | {
    "verification_asset_id",
    "verification_sha256",
}

_TOP_LEVEL_FIELDS = frozenset(
    {
        "child_job_run_id",
        "contract",
        "report_sha256",
        "snapshots",
        "target_date",
        "validator_revision",
    }
)
_SNAPSHOT_FIELDS = frozenset(
    {
        "classification",
        "input_listing_ids",
        "portfolio_id",
        "positions",
        "snapshot_id",
        "snapshot_sha256",
    }
)
_POSITION_FIELDS = frozenset(
    {
        "listing_currency",
        "listing_id",
        "normalized_asset_ref",
        "provider_subject",
        "raw_asset_ref",
        "row_id",
        "row_sha256",
        "source_session_date",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_CURRENCIES = frozenset(
    {
        Portfolio.Currency.USD,
        Portfolio.Currency.EUR,
        Portfolio.Currency.GBP,
    }
)


class PortfolioVerificationPayloadError(ValueError):
    """The portfolio proof bytes do not satisfy the frozen runtime contract."""


def _canonical_uuid(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PortfolioVerificationPayloadError(f"{label} is not a UUID string")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise PortfolioVerificationPayloadError(f"{label} is not a valid UUID") from exc
    if str(parsed) != value:
        raise PortfolioVerificationPayloadError(f"{label} is not canonically encoded")
    return value


def _canonical_date(value: object, *, label: str) -> date:
    if not isinstance(value, str):
        raise PortfolioVerificationPayloadError(f"{label} is not an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PortfolioVerificationPayloadError(f"{label} is not a valid ISO date") from exc
    if parsed.isoformat() != value:
        raise PortfolioVerificationPayloadError(f"{label} is not canonically encoded")
    return parsed


def _canonical_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PortfolioVerificationPayloadError(f"{label} is not lowercase SHA-256")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PortfolioVerificationPayloadError("Proof JSON contains a duplicate key")
        result[key] = value
    return result


def canonical_json_bytes(value: object) -> bytes:
    """Encode one strict compact/sorted JSON document with no trailing newline."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PortfolioVerificationPayloadError(
            "Proof content could not be canonically encoded"
        ) from exc


def _model_contract(model: str) -> tuple[type[Any], tuple[str, ...]]:
    if model == PORTFOLIO_SNAPSHOT_MODEL:
        return PortfolioSnapshot, PORTFOLIO_SNAPSHOT_FIELDS
    if model == PORTFOLIO_SNAPSHOT_HOLDING_MODEL:
        return PortfolioSnapshotHolding, PORTFOLIO_SNAPSHOT_HOLDING_FIELDS
    raise PortfolioVerificationPayloadError("Row digest names an unsupported model")


def _canonical_field_value(model: type[Any], attname: str, value: object) -> object:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        if timezone.is_naive(value):
            raise PortfolioVerificationPayloadError("Row digest contains a naive datetime")
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        field = next(
            (concrete for concrete in model._meta.concrete_fields if concrete.attname == attname),
            None,
        )
        places = getattr(field, "decimal_places", None)
        if places is None:
            raise PortfolioVerificationPayloadError("Decimal row value has no declared field scale")
        return format(value.quantize(Decimal(1).scaleb(-places)), f".{places}f")
    if value is None or isinstance(value, (bool, str, int)):
        return value
    # In particular, floats (including NaN/Inf) are never an accepted row
    # representation for financial evidence.
    raise PortfolioVerificationPayloadError("Row digest contains an unsupported field value")


def model_row_values(instance: Any, fields: tuple[str, ...]) -> dict[str, object]:
    return {attname: getattr(instance, attname) for attname in fields}


def canonical_row_bytes(model: str, values: dict[str, object]) -> bytes:
    model_type, fields = _model_contract(model)
    if set(values) != set(fields):
        raise PortfolioVerificationPayloadError("Row digest fields do not match the model contract")
    return canonical_json_bytes(
        {
            "fields": {
                attname: _canonical_field_value(
                    model_type,
                    attname,
                    values[attname],
                )
                for attname in fields
            },
            "model": model,
        }
    )


def row_digest(model: str, values: dict[str, object]) -> str:
    return hashlib.sha256(canonical_row_bytes(model, values)).hexdigest()


def snapshot_row_digest(snapshot: PortfolioSnapshot) -> str:
    return row_digest(
        PORTFOLIO_SNAPSHOT_MODEL,
        model_row_values(snapshot, PORTFOLIO_SNAPSHOT_FIELDS),
    )


def snapshot_holding_row_digest(position: PortfolioSnapshotHolding) -> str:
    return row_digest(
        PORTFOLIO_SNAPSHOT_HOLDING_MODEL,
        model_row_values(position, PORTFOLIO_SNAPSHOT_HOLDING_FIELDS),
    )


@dataclass(frozen=True, slots=True)
class PositionProof:
    listing_currency: str
    listing_id: UUID
    normalized_asset_ref: AssetRef
    provider_subject: str
    raw_asset_ref: AssetRef | None
    row_id: int
    row_sha256: str
    source_session_date: date

    def to_json(self) -> dict[str, object]:
        return {
            "listing_currency": self.listing_currency,
            "listing_id": str(self.listing_id),
            "normalized_asset_ref": self.normalized_asset_ref.to_json(),
            "provider_subject": self.provider_subject,
            "raw_asset_ref": (None if self.raw_asset_ref is None else self.raw_asset_ref.to_json()),
            "row_id": str(self.row_id),
            "row_sha256": self.row_sha256,
            "source_session_date": self.source_session_date.isoformat(),
        }

    @classmethod
    def from_json(cls, raw: object) -> PositionProof:
        if not isinstance(raw, dict) or set(raw) != _POSITION_FIELDS:
            raise PortfolioVerificationPayloadError("Position proof has an unexpected shape")
        currency = raw["listing_currency"]
        subject = raw["provider_subject"]
        row_id = raw["row_id"]
        if (
            not isinstance(currency, str)
            or currency not in _SUPPORTED_CURRENCIES
            or not isinstance(subject, str)
            or not subject
            or not isinstance(row_id, str)
            or not row_id.isdigit()
            or str(int(row_id)) != row_id
            or int(row_id) < 1
        ):
            raise PortfolioVerificationPayloadError(
                "Position proof contains an invalid scalar field"
            )
        try:
            normalized = AssetRef.from_json(raw["normalized_asset_ref"])
            raw_ref = (
                None if raw["raw_asset_ref"] is None else AssetRef.from_json(raw["raw_asset_ref"])
            )
        except RefreshVerificationError as exc:
            raise PortfolioVerificationPayloadError(
                "Position proof contains a malformed asset reference"
            ) from exc
        if (
            normalized.kind != "price_history"
            or normalized.subject != subject
            or (
                raw_ref is not None
                and (
                    raw_ref.kind != "raw_price_history"
                    or raw_ref.provider != normalized.provider
                    or raw_ref.subject != subject
                )
            )
        ):
            raise PortfolioVerificationPayloadError(
                "Position proof asset identities are inconsistent"
            )
        return cls(
            listing_currency=currency,
            listing_id=UUID(_canonical_uuid(raw["listing_id"], label="position listing_id")),
            normalized_asset_ref=normalized,
            provider_subject=subject,
            raw_asset_ref=raw_ref,
            row_id=int(row_id),
            row_sha256=_canonical_sha256(
                raw["row_sha256"],
                label="position row_sha256",
            ),
            source_session_date=_canonical_date(
                raw["source_session_date"],
                label="position source_session_date",
            ),
        )


@dataclass(frozen=True, slots=True)
class SnapshotProof:
    classification: str
    input_listing_ids: tuple[UUID, ...]
    portfolio_id: UUID
    positions: tuple[PositionProof, ...]
    snapshot_id: UUID
    snapshot_sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "input_listing_ids": [str(listing_id) for listing_id in self.input_listing_ids],
            "portfolio_id": str(self.portfolio_id),
            "positions": [position.to_json() for position in self.positions],
            "snapshot_id": str(self.snapshot_id),
            "snapshot_sha256": self.snapshot_sha256,
        }

    @classmethod
    def from_json(cls, raw: object) -> SnapshotProof:
        if not isinstance(raw, dict) or set(raw) != _SNAPSHOT_FIELDS:
            raise PortfolioVerificationPayloadError("Snapshot proof has an unexpected shape")
        classification = raw["classification"]
        raw_input_ids = raw["input_listing_ids"]
        raw_positions = raw["positions"]
        if classification not in {"created", "reused"}:
            raise PortfolioVerificationPayloadError("Snapshot proof classification is invalid")
        if not isinstance(raw_input_ids, list) or not isinstance(raw_positions, list):
            raise PortfolioVerificationPayloadError("Snapshot proof arrays are malformed")
        input_ids = tuple(
            UUID(_canonical_uuid(value, label="input listing id")) for value in raw_input_ids
        )
        if len(set(input_ids)) != len(input_ids):
            raise PortfolioVerificationPayloadError("Snapshot proof repeats an input listing id")
        positions = tuple(PositionProof.from_json(value) for value in raw_positions)
        if list(positions) != sorted(positions, key=lambda item: str(item.listing_id)):
            raise PortfolioVerificationPayloadError(
                "Snapshot proof positions are not canonically ordered"
            )
        listing_ids = [position.listing_id for position in positions]
        row_ids = [position.row_id for position in positions]
        if (
            len(set(listing_ids)) != len(listing_ids)
            or len(set(row_ids)) != len(row_ids)
            or set(input_ids) != set(listing_ids)
        ):
            raise PortfolioVerificationPayloadError(
                "Snapshot proof position identities are inconsistent"
            )
        return cls(
            classification=classification,
            input_listing_ids=input_ids,
            portfolio_id=UUID(_canonical_uuid(raw["portfolio_id"], label="portfolio_id")),
            positions=positions,
            snapshot_id=UUID(_canonical_uuid(raw["snapshot_id"], label="snapshot_id")),
            snapshot_sha256=_canonical_sha256(
                raw["snapshot_sha256"],
                label="snapshot_sha256",
            ),
        )


@dataclass(frozen=True, slots=True)
class PortfolioVerificationProof:
    child_job_run_id: UUID
    report_sha256: str
    snapshots: tuple[SnapshotProof, ...]
    target_date: date
    validator_revision: str

    def to_json(self) -> dict[str, object]:
        return {
            "child_job_run_id": str(self.child_job_run_id),
            "contract": PORTFOLIO_VERIFICATION_CONTRACT,
            "report_sha256": self.report_sha256,
            "snapshots": [snapshot.to_json() for snapshot in self.snapshots],
            "target_date": self.target_date.isoformat(),
            "validator_revision": self.validator_revision,
        }


def parse_portfolio_verification_proof(
    payload: bytes,
) -> PortfolioVerificationProof:
    """Strictly parse and byte-round-trip one canonical proof."""

    try:
        raw = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortfolioVerificationPayloadError(
            "Portfolio verification proof is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(raw, dict) or set(raw) != _TOP_LEVEL_FIELDS:
        raise PortfolioVerificationPayloadError(
            "Portfolio verification proof has an unexpected shape"
        )
    if raw["contract"] != PORTFOLIO_VERIFICATION_CONTRACT:
        raise PortfolioVerificationPayloadError(
            "Portfolio verification proof contract is not recognized"
        )
    revision = raw["validator_revision"]
    raw_snapshots = raw["snapshots"]
    if not isinstance(revision, str) or not revision or not isinstance(raw_snapshots, list):
        raise PortfolioVerificationPayloadError(
            "Portfolio verification proof scalar fields are malformed"
        )
    snapshots = tuple(SnapshotProof.from_json(value) for value in raw_snapshots)
    if list(snapshots) != sorted(snapshots, key=lambda item: str(item.portfolio_id)):
        raise PortfolioVerificationPayloadError(
            "Portfolio verification snapshots are not canonically ordered"
        )
    portfolio_ids = [snapshot.portfolio_id for snapshot in snapshots]
    snapshot_ids = [snapshot.snapshot_id for snapshot in snapshots]
    if len(set(portfolio_ids)) != len(portfolio_ids) or len(set(snapshot_ids)) != len(snapshot_ids):
        raise PortfolioVerificationPayloadError(
            "Portfolio verification proof repeats a snapshot identity"
        )
    proof = PortfolioVerificationProof(
        child_job_run_id=UUID(_canonical_uuid(raw["child_job_run_id"], label="child_job_run_id")),
        report_sha256=_canonical_sha256(
            raw["report_sha256"],
            label="report_sha256",
        ),
        snapshots=snapshots,
        target_date=_canonical_date(raw["target_date"], label="target_date"),
        validator_revision=revision,
    )
    if canonical_json_bytes(proof.to_json()) != payload:
        raise PortfolioVerificationPayloadError(
            "Portfolio verification proof is not canonically encoded"
        )
    return proof


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validated_report_details(
    details: dict[str, object],
    *,
    target_date: date,
    zero_active: bool = False,
) -> dict[str, object]:
    if set(details) != BASE_DETAIL_KEYS:
        raise RefreshVerificationError(
            "portfolio_report_shape_invalid",
            "Portfolio stage report details have an unexpected shape",
        )
    portfolios = details["portfolios"]
    created = details["snapshots_created"]
    unchanged = details["snapshots_unchanged"]
    snapshot_ids = details["snapshot_ids"]
    failures = details["failures"]
    if not all(_nonnegative_int(value) for value in (portfolios, created, unchanged)):
        raise RefreshVerificationError(
            "portfolio_report_counts_invalid",
            "Portfolio stage report counts are invalid",
        )
    assert isinstance(portfolios, int)
    assert isinstance(created, int)
    assert isinstance(unchanged, int)
    if not isinstance(snapshot_ids, dict) or not isinstance(failures, list):
        raise RefreshVerificationError(
            "portfolio_report_collections_invalid",
            "Portfolio stage report collections are invalid",
        )
    normalized_ids: dict[str, str] = {}
    try:
        for raw_portfolio_id, raw_snapshot_id in snapshot_ids.items():
            portfolio_id = _canonical_uuid(
                raw_portfolio_id,
                label="report portfolio id",
            )
            snapshot_id = _canonical_uuid(
                raw_snapshot_id,
                label="report snapshot id",
            )
            normalized_ids[portfolio_id] = snapshot_id
    except PortfolioVerificationPayloadError as exc:
        raise RefreshVerificationError(
            "portfolio_report_snapshot_ids_invalid",
            "Portfolio stage snapshot identities are malformed",
        ) from exc
    if (
        len(set(normalized_ids.values())) != len(normalized_ids)
        or any(not isinstance(value, str) for value in failures)
        or details["required_session_date"] != target_date.isoformat()
        or details["require_all"] is not True
    ):
        raise RefreshVerificationError(
            "portfolio_report_contract_invalid",
            "Portfolio stage report does not satisfy the scheduled contract",
        )
    if zero_active:
        valid = (
            portfolios == 0
            and created == 0
            and unchanged == 0
            and normalized_ids == {}
            and failures == []
            and details["reason"] == "no_active_portfolios"
        )
    else:
        valid = (
            portfolios > 0
            and portfolios == created + unchanged
            and len(normalized_ids) == portfolios
            and failures == []
            and details["reason"] == ""
        )
    if not valid:
        raise RefreshVerificationError(
            "portfolio_report_contract_invalid",
            "Portfolio stage report does not satisfy the scheduled contract",
        )
    return {
        **details,
        "snapshot_ids": normalized_ids,
    }


def report_sha256(details: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(details)).hexdigest()


def _validated_snapshot_actions(
    actions: tuple[tuple[str, str, str], ...],
    *,
    report: dict[str, object],
) -> dict[UUID, str]:
    """Bind private writer actions to the exact public snapshot map/counts."""

    if not isinstance(actions, tuple):
        raise RefreshVerificationError(
            "portfolio_snapshot_actions_invalid",
            "Snapshot writer actions are malformed",
        )
    report_ids = report["snapshot_ids"]
    assert isinstance(report_ids, dict)
    action_map: dict[UUID, str] = {}
    action_pairs: dict[str, str] = {}
    try:
        for item in actions:
            if not isinstance(item, tuple) or len(item) != 3:
                raise PortfolioVerificationPayloadError(
                    "Snapshot writer action has an unexpected shape"
                )
            raw_portfolio_id, raw_snapshot_id, action = item
            portfolio_id = _canonical_uuid(
                raw_portfolio_id,
                label="snapshot action portfolio id",
            )
            snapshot_id = _canonical_uuid(
                raw_snapshot_id,
                label="snapshot action snapshot id",
            )
            if action not in {"created", "reused"}:
                raise PortfolioVerificationPayloadError("Snapshot writer action is invalid")
            parsed_snapshot_id = UUID(snapshot_id)
            if portfolio_id in action_pairs or parsed_snapshot_id in action_map:
                raise PortfolioVerificationPayloadError(
                    "Snapshot writer action repeats an identity"
                )
            action_pairs[portfolio_id] = snapshot_id
            action_map[parsed_snapshot_id] = action
    except PortfolioVerificationPayloadError as exc:
        raise RefreshVerificationError(
            "portfolio_snapshot_actions_invalid",
            "Snapshot writer actions are malformed",
        ) from exc

    created = sum(action == "created" for action in action_map.values())
    reused = sum(action == "reused" for action in action_map.values())
    if (
        action_pairs != report_ids
        or created != report["snapshots_created"]
        or reused != report["snapshots_unchanged"]
    ):
        raise RefreshVerificationError(
            "portfolio_snapshot_actions_mismatch",
            "Snapshot writer actions do not match the child report",
        )
    return action_map


def _supported_persisted_decimals(
    model: type[Any],
    field_name: str,
    value: Decimal,
) -> frozenset[Decimal]:
    """Return the exact fixed-scale values written by supported databases.

    PostgreSQL ``numeric(p, s)`` rounds the exact decimal away from zero at a
    midpoint. SQLite is less direct: Django 5.2's
    ``DatabaseOperations.get_decimalfield_converter()`` converts the SQLite
    NUMERIC value through binary64, rounds that float to 15 significant
    decimal digits, and then quantizes with the model field's context. Django
    5.2 registers ``Decimal`` with sqlite3 using ``str``; applying NUMERIC
    affinity to that exact text through sqlite3 keeps SQLite's C parser in the
    projection instead of substituting Python's adjacent-float choice.

    Writer arithmetic intentionally remains unquantized until persistence, so
    historical evidence produced on either backend must remain replayable on
    the other. The returned values are an exact finite set, never a tolerance.
    Any value that cannot be represented by the declared field fails closed
    as an empty set.
    """

    field = model._meta.get_field(field_name)
    if not isinstance(field, DecimalField) or not isinstance(value, Decimal):
        return frozenset()
    places = field.decimal_places
    if places is None or not value.is_finite():
        return frozenset()
    quantum = Decimal(1).scaleb(-places)
    try:
        with localcontext(field.context) as postgres_context:
            postgres_context.rounding = ROUND_HALF_UP
            postgres_value = value.quantize(
                quantum,
                context=postgres_context,
            )

        with closing(sqlite3.connect(":memory:")) as sqlite_database:
            with closing(
                sqlite_database.execute(
                    "SELECT CAST(? AS NUMERIC)",
                    (str(value),),
                )
            ) as cursor:
                row = cursor.fetchone()
        if (
            row is None
            or len(row) != 1
            or isinstance(row[0], bool)
            or not isinstance(row[0], (float, int))
        ):
            return frozenset()
        sqlite_float = float(row[0])
        if not math.isfinite(sqlite_float):
            return frozenset()
        sqlite_intermediate = Context(prec=15).create_decimal_from_float(sqlite_float)
        sqlite_value = sqlite_intermediate.quantize(
            quantum,
            context=field.context,
        )
    except (sqlite3.Error, DecimalException, OverflowError, TypeError, ValueError):
        return frozenset()
    if not postgres_value.is_finite() or not sqlite_value.is_finite():
        return frozenset()
    return frozenset({postgres_value, sqlite_value})


def _matches_supported_persistence(
    model: type[Any],
    field_name: str,
    *,
    stored: Decimal | None,
    derived: Decimal | None,
) -> bool:
    if derived is None:
        return stored is None
    return (
        isinstance(stored, Decimal)
        and stored.is_finite()
        and stored
        in _supported_persisted_decimals(
            model,
            field_name,
            derived,
        )
    )


def _snapshot_input_hash(
    snapshot: PortfolioSnapshot,
    positions: list[PortfolioSnapshotHolding],
    input_listing_ids: tuple[UUID, ...],
) -> str:
    by_listing = {position.listing_id: position for position in positions}
    payload = {
        "portfolio_id": str(snapshot.portfolio_id),
        "as_of_date": snapshot.as_of_date.isoformat(),
        "base_currency": snapshot.base_currency,
        "cash_balance": str(snapshot.cash_balance),
        "positions": [
            {
                "listing_id": str(listing_id),
                "quantity": str(by_listing[listing_id].quantity),
                "average_cost": str(by_listing[listing_id].average_cost),
                "price": str(by_listing[listing_id].price),
                "session_date": by_listing[listing_id].source_session_date.isoformat(),
                "source_asset_id": str(by_listing[listing_id].source_asset_id),
            }
            for listing_id in input_listing_ids
        ],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _prior_corporate_action_suspected(
    snapshot: PortfolioSnapshot,
    position: PortfolioSnapshotHolding,
) -> bool:
    prior = (
        PortfolioSnapshotHolding.objects.filter(
            snapshot__portfolio_id=snapshot.portfolio_id,
            listing_id=position.listing_id,
            snapshot__recorded_at__lt=snapshot.recorded_at,
        )
        .order_by(
            "-snapshot__as_of_date",
            "-snapshot__recorded_at",
            "-snapshot_id",
        )
        .first()
    )
    if prior is None:
        return False
    if prior.corporate_action_suspected and prior.quantity == position.quantity:
        return True
    if prior.quantity != position.quantity:
        return False
    ratio = position.price / prior.price
    return ratio < SPLIT_WARNING_LOW_RATIO or ratio > SPLIT_WARNING_HIGH_RATIO


def _verify_snapshot_replay(
    snapshot: PortfolioSnapshot,
    positions: list[PortfolioSnapshotHolding],
    *,
    input_listing_ids: tuple[UUID, ...],
    source_assets: dict[UUID, DataAsset],
    target_date: date,
) -> None:
    if (
        snapshot.base_currency not in _SUPPORTED_CURRENCIES
        or snapshot.as_of_date != target_date
        or set(input_listing_ids) != {position.listing_id for position in positions}
        or len(input_listing_ids) != len(positions)
    ):
        raise RefreshVerificationError(
            "portfolio_snapshot_identity_mismatch",
            "An immutable portfolio snapshot does not match its proof identity",
        )

    if not positions:
        if not (
            snapshot.oldest_price_date is None
            and snapshot.newest_price_date is None
            and snapshot.securities_value == Decimal(0)
            and snapshot.cost_basis == Decimal(0)
            and snapshot.unrealized_gain == Decimal(0)
            and snapshot.total_value == snapshot.cash_balance
            and snapshot.return_pct is None
            and snapshot.return_definition == "price_return"
            and snapshot.dividends_included is False
            and snapshot.corporate_action_warnings == 0
        ):
            raise RefreshVerificationError(
                "portfolio_cash_only_snapshot_invalid",
                "A cash-only portfolio snapshot has invalid derived fields",
            )
    else:
        sessions = [position.source_session_date for position in positions]
        if (
            snapshot.newest_price_date != target_date
            or max(sessions) != target_date
            or snapshot.oldest_price_date != min(sessions)
            or any(
                session < target_date - timedelta(days=MAX_PORTFOLIO_PRICE_AGE_DAYS)
                or session > target_date
                for session in sessions
            )
        ):
            raise RefreshVerificationError(
                "portfolio_snapshot_session_invalid",
                "A portfolio snapshot source session is outside the scheduled bound",
            )

        raw_cost = Decimal(0)
        raw_securities = Decimal(0)
        return_definitions: set[str] = set()
        dividend_flags: list[bool] = []
        for position in positions:
            expected_cost = position.quantity * position.average_cost
            expected_market = position.quantity * position.price
            expected_gain = expected_market - expected_cost
            if (
                not _matches_supported_persistence(
                    PortfolioSnapshotHolding,
                    "cost_basis",
                    stored=position.cost_basis,
                    derived=expected_cost,
                )
                or not _matches_supported_persistence(
                    PortfolioSnapshotHolding,
                    "market_value",
                    stored=position.market_value,
                    derived=expected_market,
                )
                or not _matches_supported_persistence(
                    PortfolioSnapshotHolding,
                    "unrealized_gain",
                    stored=position.unrealized_gain,
                    derived=expected_gain,
                )
            ):
                raise RefreshVerificationError(
                    "portfolio_position_arithmetic_invalid",
                    "An immutable portfolio position has invalid arithmetic",
                )
            raw_cost += expected_cost
            raw_securities += expected_market
            asset = source_assets[position.source_asset_id]
            metadata = asset.metadata if isinstance(asset.metadata, dict) else {}
            return_definitions.add(str(metadata.get("return_definition") or "price_return"))
            dividend_flags.append(metadata.get("dividends_included") is True)

        raw_gain = raw_securities - raw_cost
        expected_return = raw_gain / raw_cost if raw_cost > 0 else None
        expected_definition = (
            next(iter(return_definitions)) if len(return_definitions) == 1 else "mixed_price_return"
        )
        if (
            not _matches_supported_persistence(
                PortfolioSnapshot,
                "securities_value",
                stored=snapshot.securities_value,
                derived=raw_securities,
            )
            or not _matches_supported_persistence(
                PortfolioSnapshot,
                "cost_basis",
                stored=snapshot.cost_basis,
                derived=raw_cost,
            )
            or not _matches_supported_persistence(
                PortfolioSnapshot,
                "total_value",
                stored=snapshot.total_value,
                derived=snapshot.cash_balance + raw_securities,
            )
            or not _matches_supported_persistence(
                PortfolioSnapshot,
                "unrealized_gain",
                stored=snapshot.unrealized_gain,
                derived=raw_gain,
            )
            or not _matches_supported_persistence(
                PortfolioSnapshot,
                "return_pct",
                stored=snapshot.return_pct,
                derived=expected_return,
            )
            or snapshot.return_definition != expected_definition
            or snapshot.dividends_included != (bool(dividend_flags) and all(dividend_flags))
        ):
            raise RefreshVerificationError(
                "portfolio_snapshot_arithmetic_invalid",
                "An immutable portfolio snapshot has invalid derived arithmetic",
            )

    if _snapshot_input_hash(snapshot, positions, input_listing_ids) != snapshot.input_hash:
        raise RefreshVerificationError(
            "portfolio_snapshot_input_hash_invalid",
            "An immutable portfolio snapshot does not reproduce its input hash",
        )
    expected_corporate_actions = 0
    for position in positions:
        expected = _prior_corporate_action_suspected(snapshot, position)
        if position.corporate_action_suspected is not expected:
            raise RefreshVerificationError(
                "portfolio_snapshot_corporate_action_invalid",
                "An immutable portfolio position has an invalid corporate-action flag",
            )
        expected_corporate_actions += int(expected)
    if snapshot.corporate_action_warnings != expected_corporate_actions:
        raise RefreshVerificationError(
            "portfolio_snapshot_corporate_action_invalid",
            "An immutable portfolio snapshot has an invalid corporate-action count",
        )


def _require_asset_time(asset: DataAsset, *, cutoff: datetime) -> None:
    if (
        timezone.is_naive(cutoff)
        or timezone.is_naive(asset.available_at)
        or timezone.is_naive(asset.retrieved_at)
        or asset.available_at > cutoff
        or asset.retrieved_at > cutoff
    ):
        raise RefreshVerificationError(
            "portfolio_source_asset_time_invalid",
            "A portfolio source asset is outside the snapshot availability boundary",
        )


def _resolve_source_asset_ref(ref: AssetRef, *, cutoff: datetime) -> DataAsset:
    asset = DataAsset.objects.filter(
        pk=ref.id,
        provider=ref.provider,
        kind=ref.kind,
        subject=ref.subject,
        sha256=ref.sha256,
    ).first()
    if asset is None:
        raise RefreshVerificationError(
            "asset_ref_unresolved",
            "A referenced asset could not be resolved by its exact identity and checksum",
        )
    _require_asset_time(asset, cutoff=cutoff)
    return asset


def _require_asset_currency(asset: DataAsset, currency: str) -> None:
    metadata = asset.metadata if isinstance(asset.metadata, dict) else {}
    raw_currency = metadata.get("currency")
    if raw_currency is not None and (
        not isinstance(raw_currency, str)
        or not raw_currency.strip()
        or raw_currency.upper() != currency
    ):
        raise RefreshVerificationError(
            "portfolio_source_currency_mismatch",
            "A portfolio source asset currency does not match the attested currency",
        )
    if asset.provider == "twelve_data" and raw_currency is None:
        raise RefreshVerificationError(
            "portfolio_source_currency_missing",
            "A Twelve Data portfolio source asset has no currency metadata",
        )


def _raw_ref_for_normalized(
    asset: DataAsset,
    *,
    cutoff: datetime,
    store: AssetStore,
) -> AssetRef | None:
    metadata = asset.metadata if isinstance(asset.metadata, dict) else {}
    raw_id = metadata.get("raw_asset_id")
    raw_sha = metadata.get("raw_sha256")
    if (raw_id is None) != (raw_sha is None):
        raise RefreshVerificationError(
            "portfolio_raw_asset_link_partial",
            "A portfolio source asset has a partial raw-evidence link",
        )
    if raw_id is None:
        if asset.provider not in {"synthetic", "synthetic_demo"}:
            raise RefreshVerificationError(
                "portfolio_raw_asset_link_missing",
                "A production portfolio source asset has no raw evidence",
            )
        return None
    try:
        raw_uuid = UUID(str(raw_id))
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "portfolio_raw_asset_link_invalid",
            "A portfolio source asset has a malformed raw-evidence link",
        ) from exc
    if not isinstance(raw_sha, str) or _SHA256_RE.fullmatch(raw_sha) is None:
        raise RefreshVerificationError(
            "portfolio_raw_asset_link_invalid",
            "A portfolio source asset has a malformed raw-evidence checksum",
        )
    raw_asset = DataAsset.objects.filter(
        pk=raw_uuid,
        provider=asset.provider,
        kind="raw_price_history",
        subject=asset.subject,
        sha256=raw_sha,
    ).first()
    if raw_asset is None:
        raise RefreshVerificationError(
            "portfolio_raw_asset_unresolved",
            "A portfolio raw source asset could not be resolved by complete identity",
        )
    _require_asset_time(raw_asset, cutoff=cutoff)
    read_checksummed_bytes(store, raw_asset)
    return asset_ref_for(raw_asset)


def _verify_position_source(
    claim: PositionProof,
    position: PortfolioSnapshotHolding,
    snapshot: PortfolioSnapshot,
    *,
    cutoff: datetime,
    store: AssetStore,
    locked_market: LatestMarketData | None = None,
) -> tuple[DataAsset, DataAsset | None]:
    if (
        claim.listing_id != position.listing_id
        or claim.row_id != position.pk
        or claim.row_sha256 != snapshot_holding_row_digest(position)
        or claim.source_session_date != position.source_session_date
        or claim.listing_currency != snapshot.base_currency
        or claim.provider_subject != claim.normalized_asset_ref.subject
    ):
        raise RefreshVerificationError(
            "portfolio_position_proof_mismatch",
            "A portfolio position does not match its immutable proof",
        )
    normalized = _resolve_source_asset_ref(
        claim.normalized_asset_ref,
        cutoff=cutoff,
    )
    if normalized.id != position.source_asset_id:
        raise RefreshVerificationError(
            "portfolio_position_source_mismatch",
            "A portfolio position names a different normalized source asset",
        )
    _require_asset_time(normalized, cutoff=cutoff)
    _require_asset_currency(normalized, claim.listing_currency)
    fields = verified_price_fields(
        normalized,
        cutoff=cutoff,
        target_date=position.source_session_date,
        close_places=PortfolioSnapshotHolding._meta.get_field("price").decimal_places,
    )
    if fields.close != position.price:
        raise RefreshVerificationError(
            "portfolio_position_source_price_mismatch",
            "A portfolio position price does not match its physical source asset",
        )
    raw_ref = _raw_ref_for_normalized(
        normalized,
        cutoff=cutoff,
        store=store,
    )
    if raw_ref != claim.raw_asset_ref:
        raise RefreshVerificationError(
            "portfolio_position_raw_proof_mismatch",
            "A portfolio position raw source does not match its immutable proof",
        )
    raw_asset = (
        None
        if raw_ref is None
        else _resolve_source_asset_ref(
            raw_ref,
            cutoff=cutoff,
        )
    )
    if locked_market is not None and (
        locked_market.source_asset_id != normalized.id
        or locked_market.session_date != position.source_session_date
        or locked_market.close != position.price
        or locked_market.observed_at != normalized.retrieved_at
        or locked_market.previous_close != fields.previous_close
        or locked_market.volume != fields.volume
    ):
        raise RefreshVerificationError(
            "portfolio_locked_market_source_mismatch",
            "Locked market data does not match its immutable physical source",
        )
    return normalized, raw_asset


def _input_order_for_portfolio(portfolio_id: UUID) -> tuple[UUID, ...]:
    # This intentionally mirrors ``calculate_portfolio_valuation``'s actual
    # ordering.  It is not a locking query; every row and listing it reads has
    # already been locked by a deterministic, join-free PK query.
    return tuple(
        PortfolioHolding.objects.filter(portfolio_id=portfolio_id)
        .order_by("listing__ticker")
        .values_list("listing_id", flat=True)
    )


def _locked_input_signatures(
    portfolios: list[Portfolio],
    holdings: list[PortfolioHolding],
    listings: list[Listing],
    market_rows: list[LatestMarketData],
) -> tuple[tuple[object, ...], ...]:
    return (
        tuple(
            (
                row.pk,
                row.archived_at,
                row.base_currency,
                row.cash_balance,
            )
            for row in portfolios
        ),
        tuple(
            (
                row.pk,
                row.portfolio_id,
                row.listing_id,
                row.quantity,
                row.average_cost,
            )
            for row in holdings
        ),
        tuple(
            (
                row.pk,
                row.is_active,
                row.currency,
                row.provider_symbol,
                row.ticker,
            )
            for row in listings
        ),
        tuple(
            (
                row.pk,
                row.observed_at,
                row.session_date,
                row.close,
                row.previous_close,
                row.volume,
                row.source_asset_id,
            )
            for row in market_rows
        ),
    )


def _fresh_input_signatures(
    *,
    portfolio_ids: tuple[UUID, ...],
    listing_ids: tuple[UUID, ...],
) -> tuple[tuple[object, ...], ...]:
    portfolios = list(Portfolio.objects.filter(archived_at__isnull=True).order_by("pk"))
    holdings = list(PortfolioHolding.objects.filter(portfolio_id__in=portfolio_ids).order_by("pk"))
    listings = list(Listing.objects.filter(pk__in=listing_ids).order_by("pk"))
    market_rows = list(LatestMarketData.objects.filter(listing_id__in=listing_ids).order_by("pk"))
    return _locked_input_signatures(
        portfolios,
        holdings,
        listings,
        market_rows,
    )


def _snapshot_position_identities(
    snapshot_ids: tuple[UUID, ...],
) -> tuple[tuple[UUID, int, UUID], ...]:
    """Read the complete persisted position closure for exact snapshots."""

    return tuple(
        PortfolioSnapshotHolding.objects.filter(snapshot_id__in=snapshot_ids)
        .order_by("snapshot_id", "pk")
        .values_list("snapshot_id", "pk", "listing_id")
    )


def _unlink_if_attempt_checksum(
    store: AssetStore,
    relative_path: str,
    *,
    attempt_sha256: str,
) -> None:
    """Unlink only bytes that still belong to the failed write attempt."""

    try:
        target = store.resolve(relative_path)
        if not target.exists():
            return
        # This second, immediately-before-unlink read is the filesystem
        # ownership guard. PostgreSQL cleanup holds the DataAsset table lock
        # across this read, and AssetStore refuses to replace different
        # existing bytes.
        if hashlib.sha256(target.read_bytes()).hexdigest() != attempt_sha256:
            return
        target.unlink(missing_ok=True)
    except (OSError, TypeError, ValueError):
        return


def _proof_path_claimed(relative_path: str) -> bool:
    return DataAsset.objects.filter(relative_path=relative_path).exists()


def _cleanup_attempt_file(
    store: AssetStore,
    relative_path: str,
    *,
    attempt_sha256: str,
) -> None:
    """Reconcile attempt-owned bytes with any claimant after rollback.

    A same-checksum claimant safely adopts the bytes. No claimant, or a
    different-checksum claimant, cannot adopt this attempt's bytes, so those
    bytes are removed if their checksum still matches. PostgreSQL's table
    lock waits for every already-in-flight ``DataAsset`` insert, including a
    direct writer that does not share the portfolio-child lock, before the
    claimant query and filesystem reconciliation. Any indeterminate
    database/filesystem state preserves the file. This does not claim to
    protect a wholly later direct insert that starts after cleanup's final
    claimant observation.
    """

    try:
        with transaction.atomic():
            if connection.vendor == "postgresql":
                table_name = connection.ops.quote_name(DataAsset._meta.db_table)
                with connection.cursor() as cursor:
                    cursor.execute(f"LOCK TABLE {table_name} IN SHARE MODE")
            # SQLite has no LOCK TABLE syntax. Its cleanup still runs in this
            # fresh atomic block and relies on SQLite's existing database
            # write serialization.
            claimant_sha256 = (
                DataAsset.objects.filter(relative_path=relative_path)
                .values_list("sha256", flat=True)
                .first()
            )
            if claimant_sha256 == attempt_sha256:
                return
            _unlink_if_attempt_checksum(
                store,
                relative_path,
                attempt_sha256=attempt_sha256,
            )
    except (DatabaseError, OSError, TypeError, ValueError):
        # Cleanup is best-effort and must never replace the original failure.
        # Indeterminate ownership preserves bytes rather than risking another
        # transaction's successfully registered proof.
        return


def _proof_relative_path(child_id: UUID) -> str:
    return f"portfolio/verification/{child_id}.json"


def attest_scheduled_portfolio_snapshots(
    *,
    child_run: JobRun,
    target_date: date,
    report_details: dict[str, object],
    snapshot_actions: tuple[tuple[str, str, str], ...],
    validator_revision: str,
) -> DataAsset:
    """Lock, replay, and attest one nonempty authoritative scheduled child."""

    report = _validated_report_details(
        report_details,
        target_date=target_date,
    )
    actions_by_snapshot = _validated_snapshot_actions(
        snapshot_actions,
        report=report,
    )
    if not isinstance(validator_revision, str) or not validator_revision:
        raise RefreshVerificationError(
            "portfolio_validator_revision_invalid",
            "The scheduled portfolio validator revision is unavailable",
        )

    store = open_asset_store()
    relative_path = _proof_relative_path(child_run.pk)
    attempt_created_file = False
    attempt_sha256: str | None = None
    try:
        with transaction.atomic():
            locked_child = (
                JobRun.objects.select_for_update(of=("self",))
                .filter(pk=child_run.pk)
                .order_by("pk")
                .first()
            )
            if (
                locked_child is None
                or locked_child.job_name != "scheduled_portfolio_snapshots"
                or locked_child.region != ""
                or locked_child.target_date != target_date
                or locked_child.status != JobRun.Status.RUNNING
                or locked_child.finished_at is not None
                or timezone.is_naive(locked_child.started_at)
            ):
                raise RefreshVerificationError(
                    "portfolio_child_identity_invalid",
                    "The portfolio attester child identity is invalid",
                )

            portfolios = list(
                Portfolio.objects.select_for_update(of=("self",))
                .filter(archived_at__isnull=True)
                .order_by("pk")
            )
            portfolio_ids = tuple(row.pk for row in portfolios)
            holdings = list(
                PortfolioHolding.objects.select_for_update(of=("self",))
                .filter(portfolio_id__in=portfolio_ids)
                .order_by("pk")
            )
            listing_ids = tuple(sorted({row.listing_id for row in holdings}, key=str))
            listings = list(
                Listing.objects.select_for_update(of=("self",))
                .filter(pk__in=listing_ids)
                .order_by("pk")
            )
            market_rows = list(
                LatestMarketData.objects.select_for_update(of=("self",))
                .filter(listing_id__in=listing_ids)
                .order_by("pk")
            )
            original_signatures = _locked_input_signatures(
                portfolios,
                holdings,
                listings,
                market_rows,
            )
            if len(portfolios) != report["portfolios"]:
                raise RefreshVerificationError(
                    "portfolio_active_set_changed",
                    "The active portfolio set changed before attestation",
                )

            report_ids = report["snapshot_ids"]
            assert isinstance(report_ids, dict)
            if set(report_ids) != {str(value) for value in portfolio_ids}:
                raise RefreshVerificationError(
                    "portfolio_report_active_set_mismatch",
                    "The snapshot report does not match the locked active set",
                )
            snapshot_ids = tuple(UUID(value) for value in report_ids.values())
            snapshots = list(
                PortfolioSnapshot.objects.filter(pk__in=snapshot_ids).order_by("portfolio_id")
            )
            if len(snapshots) != len(snapshot_ids):
                raise RefreshVerificationError(
                    "portfolio_report_snapshot_missing",
                    "A reported immutable portfolio snapshot is missing",
                )
            holdings_by_portfolio: dict[UUID, list[PortfolioHolding]] = {}
            for holding in holdings:
                holdings_by_portfolio.setdefault(
                    holding.portfolio_id,
                    [],
                ).append(holding)
            listing_by_id = {row.pk: row for row in listings}
            market_by_listing = {row.listing_id: row for row in market_rows}
            portfolio_by_id = {row.pk: row for row in portfolios}

            proof_snapshots: list[SnapshotProof] = []
            source_assets: dict[UUID, DataAsset] = {}
            for snapshot in snapshots:
                portfolio = portfolio_by_id.get(snapshot.portfolio_id)
                if portfolio is None or report_ids.get(str(snapshot.portfolio_id)) != str(
                    snapshot.pk
                ):
                    raise RefreshVerificationError(
                        "portfolio_report_snapshot_mismatch",
                        "A reported snapshot does not belong to the locked portfolio",
                    )
                classification = actions_by_snapshot[snapshot.pk]
                if (
                    timezone.is_naive(snapshot.recorded_at)
                    or (
                        classification == "created"
                        and snapshot.recorded_at < locked_child.started_at
                    )
                    or (
                        snapshot.recorded_at >= locked_child.started_at
                        and snapshot.code_revision != validator_revision
                    )
                ):
                    raise RefreshVerificationError(
                        "portfolio_snapshot_revision_invalid",
                        "A child snapshot has invalid action timing or validator revision",
                    )

                current_holdings = holdings_by_portfolio.get(snapshot.portfolio_id, [])
                positions = list(
                    PortfolioSnapshotHolding.objects.filter(snapshot=snapshot).order_by("pk")
                )
                if (
                    snapshot.base_currency != portfolio.base_currency
                    or snapshot.cash_balance != portfolio.cash_balance
                    or {row.listing_id for row in current_holdings}
                    != {row.listing_id for row in positions}
                ):
                    raise RefreshVerificationError(
                        "portfolio_locked_state_mismatch",
                        "A snapshot does not match the locked portfolio state",
                    )
                current_by_listing = {row.listing_id: row for row in current_holdings}
                input_order = _input_order_for_portfolio(snapshot.portfolio_id)
                position_proofs: list[PositionProof] = []
                for position in positions:
                    current = current_by_listing[position.listing_id]
                    listing = listing_by_id.get(position.listing_id)
                    market = market_by_listing.get(position.listing_id)
                    if (
                        listing is None
                        or market is None
                        or not listing.is_active
                        or listing.currency.upper() != snapshot.base_currency
                        or current.quantity != position.quantity
                        or current.average_cost != position.average_cost
                    ):
                        raise RefreshVerificationError(
                            "portfolio_locked_position_mismatch",
                            "A snapshot position does not match its locked inputs",
                        )
                    provider_subject = listing.provider_symbol or listing.ticker
                    normalized = DataAsset.objects.filter(pk=position.source_asset_id).first()
                    if (
                        normalized is None
                        or normalized.kind != "price_history"
                        or normalized.subject != provider_subject
                    ):
                        raise RefreshVerificationError(
                            "portfolio_source_identity_mismatch",
                            "A snapshot position has invalid normalized source identity",
                        )
                    # A valuation may only consume evidence that was knowable
                    # when its immutable snapshot was recorded. Proof
                    # registration is a later outer bound, never the
                    # point-in-time decision boundary.
                    source_cutoff = snapshot.recorded_at
                    _require_asset_time(normalized, cutoff=source_cutoff)
                    raw_ref = _raw_ref_for_normalized(
                        normalized,
                        cutoff=source_cutoff,
                        store=store,
                    )
                    claim = PositionProof(
                        listing_currency=listing.currency.upper(),
                        listing_id=listing.pk,
                        normalized_asset_ref=asset_ref_for(normalized),
                        provider_subject=provider_subject,
                        raw_asset_ref=raw_ref,
                        row_id=position.pk,
                        row_sha256=snapshot_holding_row_digest(position),
                        source_session_date=position.source_session_date,
                    )
                    _verify_position_source(
                        claim,
                        position,
                        snapshot,
                        cutoff=source_cutoff,
                        store=store,
                        locked_market=market,
                    )
                    source_assets[normalized.id] = normalized
                    position_proofs.append(claim)

                _verify_snapshot_replay(
                    snapshot,
                    positions,
                    input_listing_ids=input_order,
                    source_assets=source_assets,
                    target_date=target_date,
                )
                proof_snapshots.append(
                    SnapshotProof(
                        classification=classification,
                        input_listing_ids=input_order,
                        portfolio_id=snapshot.portfolio_id,
                        positions=tuple(
                            sorted(
                                position_proofs,
                                key=lambda item: str(item.listing_id),
                            )
                        ),
                        snapshot_id=snapshot.pk,
                        snapshot_sha256=snapshot_row_digest(snapshot),
                    )
                )

            final_signatures = _fresh_input_signatures(
                portfolio_ids=portfolio_ids,
                listing_ids=listing_ids,
            )
            final_orders = {
                portfolio_id: _input_order_for_portfolio(portfolio_id)
                for portfolio_id in portfolio_ids
            }
            expected_orders = {
                item.portfolio_id: item.input_listing_ids for item in proof_snapshots
            }
            if final_signatures != original_signatures or final_orders != expected_orders:
                raise RefreshVerificationError(
                    "portfolio_inputs_changed_during_attestation",
                    "Portfolio inputs changed during attestation",
                )

            proof = PortfolioVerificationProof(
                child_job_run_id=locked_child.pk,
                report_sha256=report_sha256(report),
                snapshots=tuple(
                    sorted(
                        proof_snapshots,
                        key=lambda item: str(item.portfolio_id),
                    )
                ),
                target_date=target_date,
                validator_revision=validator_revision,
            )
            payload = canonical_json_bytes(proof.to_json())
            payload_sha256 = hashlib.sha256(payload).hexdigest()
            attempt_sha256 = payload_sha256
            if _proof_path_claimed(relative_path):
                raise RefreshVerificationError(
                    "portfolio_verification_path_registered",
                    "The portfolio verification proof location is already registered",
                )
            try:
                resolved = store.resolve(relative_path)
                file_existed = resolved.exists()
            except (OSError, ValueError):
                raise RefreshVerificationError(
                    "portfolio_verification_path_unavailable",
                    "The portfolio verification path could not be checked",
                ) from None

            registration_time = timezone.now()
            if timezone.is_naive(registration_time):
                raise RefreshVerificationError(
                    "portfolio_registration_time_invalid",
                    "The proof registration clock is not timezone-aware",
                )
            if any(snapshot.recorded_at > registration_time for snapshot in snapshots) or any(
                asset.available_at > registration_time or asset.retrieved_at > registration_time
                for asset in source_assets.values()
            ):
                raise RefreshVerificationError(
                    "portfolio_verification_time_invalid",
                    "Portfolio evidence is later than proof registration",
                )
            if DataAsset.objects.filter(
                provider=PORTFOLIO_VERIFICATION_PROVIDER,
                kind=PORTFOLIO_VERIFICATION_KIND,
                subject=str(locked_child.pk),
            ).exists():
                raise RefreshVerificationError(
                    "portfolio_verification_already_registered",
                    "The child already has a portfolio verification asset",
                )
            expected_position_identities = {
                (
                    snapshot.snapshot_id,
                    position.row_id,
                    position.listing_id,
                )
                for snapshot in proof.snapshots
                for position in snapshot.positions
            }
            if set(_snapshot_position_identities(snapshot_ids)) != expected_position_identities:
                raise RefreshVerificationError(
                    "portfolio_snapshot_position_closure_changed",
                    "Snapshot positions changed before proof registration",
                )
            try:
                # The nested atomic block is a savepoint.  It restores the
                # surrounding transaction after an IntegrityError so the
                # conflicting path can be classified without querying inside
                # a broken transaction.
                with transaction.atomic():
                    asset = DataAsset.objects.create(
                        provider=PORTFOLIO_VERIFICATION_PROVIDER,
                        kind=PORTFOLIO_VERIFICATION_KIND,
                        subject=str(locked_child.pk),
                        relative_path=relative_path,
                        sha256=payload_sha256,
                        retrieved_at=registration_time,
                        available_at=registration_time,
                        period_start=target_date,
                        period_end=target_date,
                        schema_version=PORTFOLIO_VERIFICATION_SCHEMA,
                        metadata={"usage_scope": PRIVATE_USAGE_SCOPE},
                    )
            except IntegrityError:
                if _proof_path_claimed(relative_path):
                    raise RefreshVerificationError(
                        "portfolio_verification_path_registered",
                        "The portfolio verification proof location is already registered",
                    ) from None
                raise

            # The row is still private to this transaction. Creating it before
            # touching the deterministic path makes a concurrent claimant win
            # at the unique constraint without allowing this attempt to place
            # bytes beneath that claimant.
            try:
                stored = store.write_bytes(relative_path, payload)
            except (OSError, ValueError):
                raise RefreshVerificationError(
                    "portfolio_verification_write_failed",
                    "The portfolio verification proof could not be written",
                ) from None
            attempt_created_file = not file_existed
            if (
                stored.relative_path != asset.relative_path
                or stored.sha256 != asset.sha256
                or stored.byte_count != len(payload)
            ):
                raise RefreshVerificationError(
                    "portfolio_verification_write_failed",
                    "The portfolio verification proof could not be written",
                )
        return asset
    except Exception:
        if attempt_created_file and attempt_sha256 is not None:
            _cleanup_attempt_file(
                store,
                relative_path,
                attempt_sha256=attempt_sha256,
            )
        raise


def _resolve_proof_asset(
    child: JobRun,
    details: dict[str, object],
    *,
    target_date: date,
) -> tuple[DataAsset, PortfolioVerificationProof]:
    assets = list(
        DataAsset.objects.filter(
            provider=PORTFOLIO_VERIFICATION_PROVIDER,
            kind=PORTFOLIO_VERIFICATION_KIND,
            subject=str(child.pk),
        ).order_by("pk")
    )
    if len(assets) != 1:
        raise RefreshVerificationError(
            "portfolio_verification_asset_count_invalid",
            "The portfolio child does not have exactly one verification asset",
        )
    asset = assets[0]
    try:
        detail_id = UUID(
            _canonical_uuid(
                details["verification_asset_id"],
                label="verification_asset_id",
            )
        )
    except (KeyError, PortfolioVerificationPayloadError) as exc:
        raise RefreshVerificationError(
            "portfolio_verification_detail_id_invalid",
            "The portfolio child verification asset id is malformed",
        ) from exc
    detail_sha = details.get("verification_sha256")
    if (
        asset.id != detail_id
        or not isinstance(detail_sha, str)
        or asset.sha256 != detail_sha
        or _SHA256_RE.fullmatch(detail_sha) is None
        or asset.relative_path != _proof_relative_path(child.pk)
        or asset.schema_version != PORTFOLIO_VERIFICATION_SCHEMA
        or asset.period_start != target_date
        or asset.period_end != target_date
        or asset.metadata != {"usage_scope": PRIVATE_USAGE_SCOPE}
        or timezone.is_naive(asset.available_at)
        or timezone.is_naive(asset.retrieved_at)
        or asset.available_at != asset.retrieved_at
        or child.finished_at is None
        or asset.available_at < child.started_at
        or asset.available_at > child.finished_at
    ):
        raise RefreshVerificationError(
            "portfolio_verification_asset_identity_invalid",
            "The portfolio verification asset identity or timing is invalid",
        )
    payload = read_checksummed_bytes(open_asset_store(), asset)
    try:
        proof = parse_portfolio_verification_proof(payload)
    except PortfolioVerificationPayloadError as exc:
        raise RefreshVerificationError(
            "portfolio_verification_payload_invalid",
            "The portfolio verification proof payload is invalid",
        ) from exc
    return asset, proof


def verify_portfolio_snapshot_stage(
    child: JobRun,
    *,
    target_date: date,
) -> StageVerificationResult:
    """Verify the portfolio child from immutable proof and source evidence."""

    if (
        child.job_name != "scheduled_portfolio_snapshots"
        or child.region != ""
        or child.target_date != target_date
    ):
        raise RefreshVerificationError(
            "portfolio_child_identity_invalid",
            "The portfolio verification child identity is invalid",
        )
    details = child.details if isinstance(child.details, dict) else {}
    if child.status == JobRun.Status.SKIPPED:
        report = _validated_report_details(
            details,
            target_date=target_date,
            zero_active=True,
        )
        if DataAsset.objects.filter(
            provider=PORTFOLIO_VERIFICATION_PROVIDER,
            kind=PORTFOLIO_VERIFICATION_KIND,
            subject=str(child.pk),
        ).exists():
            raise RefreshVerificationError(
                "portfolio_zero_active_proof_present",
                "A zero-active portfolio skip unexpectedly has a proof asset",
            )
        assert report["portfolios"] == 0
        return StageVerificationResult(
            summary={
                "job_run_id": str(child.pk),
                "active_portfolios": 0,
                "skip_reason": "no_active_portfolios",
            },
            asset_refs=(),
        )
    if child.status != JobRun.Status.SUCCESS or set(details) != VERIFIED_DETAIL_KEYS:
        raise RefreshVerificationError(
            "portfolio_stage_contract_invalid",
            "The nonempty portfolio stage is not a verified scheduled success",
        )
    report = _validated_report_details(
        {key: details[key] for key in BASE_DETAIL_KEYS},
        target_date=target_date,
    )
    proof_asset, proof = _resolve_proof_asset(
        child,
        details,
        target_date=target_date,
    )
    if (
        proof.child_job_run_id != child.pk
        or proof.target_date != target_date
        or proof.report_sha256 != report_sha256(report)
        or len(proof.snapshots) != report["portfolios"]
    ):
        raise RefreshVerificationError(
            "portfolio_verification_report_mismatch",
            "The portfolio proof does not match its authoritative child report",
        )

    report_ids = report["snapshot_ids"]
    assert isinstance(report_ids, dict)
    if {str(item.portfolio_id): str(item.snapshot_id) for item in proof.snapshots} != report_ids:
        raise RefreshVerificationError(
            "portfolio_verification_snapshot_map_mismatch",
            "The portfolio proof snapshot map does not match the child report",
        )
    classifications = {
        "created": sum(item.classification == "created" for item in proof.snapshots),
        "reused": sum(item.classification == "reused" for item in proof.snapshots),
    }
    if (
        classifications["created"] != report["snapshots_created"]
        or classifications["reused"] != report["snapshots_unchanged"]
    ):
        raise RefreshVerificationError(
            "portfolio_verification_classification_mismatch",
            "The portfolio proof classifications do not match the child report",
        )

    snapshots = {
        row.pk: row
        for row in PortfolioSnapshot.objects.filter(
            pk__in=[item.snapshot_id for item in proof.snapshots]
        ).order_by("pk")
    }
    if len(snapshots) != len(proof.snapshots):
        raise RefreshVerificationError(
            "portfolio_verification_snapshot_missing",
            "An immutable snapshot named by the portfolio proof is missing",
        )
    persisted_positions = list(
        PortfolioSnapshotHolding.objects.filter(snapshot_id__in=snapshots).order_by(
            "snapshot_id", "pk"
        )
    )
    proof_position_identities = {
        (
            snapshot_proof.snapshot_id,
            claim.row_id,
            claim.listing_id,
        )
        for snapshot_proof in proof.snapshots
        for claim in snapshot_proof.positions
    }
    persisted_position_identities = {
        (
            position.snapshot_id,
            position.pk,
            position.listing_id,
        )
        for position in persisted_positions
    }
    if persisted_position_identities != proof_position_identities:
        raise RefreshVerificationError(
            "portfolio_verification_position_closure_mismatch",
            "Persisted snapshot positions do not exactly match the portfolio proof",
        )
    positions_by_snapshot: dict[UUID, list[PortfolioSnapshotHolding]] = {}
    for position in persisted_positions:
        positions_by_snapshot.setdefault(position.snapshot_id, []).append(position)

    store = open_asset_store()
    asset_refs: dict[UUID, AssetRef] = {proof_asset.id: asset_ref_for(proof_asset)}
    for snapshot_proof in proof.snapshots:
        snapshot = snapshots[snapshot_proof.snapshot_id]
        if (
            snapshot.portfolio_id != snapshot_proof.portfolio_id
            or snapshot_row_digest(snapshot) != snapshot_proof.snapshot_sha256
            or timezone.is_naive(snapshot.recorded_at)
            or snapshot.recorded_at > proof_asset.available_at
            or (
                snapshot_proof.classification == "created"
                and snapshot.recorded_at < child.started_at
            )
            or (
                snapshot.recorded_at >= child.started_at
                and snapshot.code_revision != proof.validator_revision
            )
        ):
            raise RefreshVerificationError(
                "portfolio_verification_snapshot_digest_mismatch",
                "An immutable snapshot does not match its portfolio proof",
            )
        positions = positions_by_snapshot.get(snapshot.pk, [])
        claims_by_row = {claim.row_id: claim for claim in snapshot_proof.positions}
        source_assets: dict[UUID, DataAsset] = {}
        for position in positions:
            claim = claims_by_row[position.pk]
            normalized, raw = _verify_position_source(
                claim,
                position,
                snapshot,
                cutoff=snapshot.recorded_at,
                store=store,
            )
            source_assets[normalized.id] = normalized
            asset_refs[normalized.id] = asset_ref_for(normalized)
            if raw is not None:
                asset_refs[raw.id] = asset_ref_for(raw)
        _verify_snapshot_replay(
            snapshot,
            positions,
            input_listing_ids=snapshot_proof.input_listing_ids,
            source_assets=source_assets,
            target_date=target_date,
        )

    return StageVerificationResult(
        summary={
            "job_run_id": str(child.pk),
            "active_portfolios": report["portfolios"],
            "snapshots_verified": len(proof.snapshots),
        },
        asset_refs=tuple(asset_refs[key] for key in sorted(asset_refs, key=str)),
    )

"""Durable pre-provider intake for the prospective price research product.

`TrackedSymbol` remains a mutable preference.  This module turns an approved
bounded set into a checksummed immutable intake asset before a caller may
resolve credentials or request history.  A retry is keyed by an explicit
issuance key, so it reuses its original capture even if the owner changes My
List afterwards; a changed same-target request must use a new issuance key.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import polars as pl
from django.contrib.auth import get_user_model
from django.db import transaction

from stanstock.data.asof import AsOfData
from stanstock.data.assets import AssetStore, asset_ref_for, read_checksummed_bytes, register_asset
from stanstock.data.live_us import load_us_universe_config
from stanstock.data.management.config_loader import (
    default_us_universe_config_path,
    load_yaml_mapping,
)
from stanstock.data.models import (
    DataAsset,
    Listing,
    ProviderRecord,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.data.provider_policy import TWELVE_DATA_PROVIDER, validate_provider_usage
from stanstock.data.providers import twelve_data
from stanstock.data.providers.contracts import StockReference
from stanstock.portfolio.models import TrackedSymbol
from stanstock.research.price_product_config import PRODUCT_EFFECTIVE_CONFIG_HASH, PRODUCT_VERSION

PRODUCT_INTAKE_KIND = "research_product_intake"
PRODUCT_INTAKE_CONTRACT = "research-product-intake@1"
PRODUCT_MEMBERSHIP_KIND = "research_product_membership"
PRODUCT_MEMBERSHIP_CONTRACT = "research-product-membership@1"
MAX_SAVED_NAMES = 20


def load_product_universe_policy() -> dict[str, Any]:
    core_path = default_us_universe_config_path()
    policy = load_yaml_mapping(core_path.with_name("us_research_product_v1.yaml"))
    expected = {
        "schema_version": 1,
        "config_version": "research-product-universe-v1",
        "core_universe": "us_liquid_starter_v1.yaml",
        "core_sha256": "5159829122cc7e53d15b7c55a5c9bbed2f23e9f2ad94f72a1dbb943c8fc89128",
        "maximum_core_names": 100,
        "maximum_saved_names": 20,
        "benchmark_symbol": "SPY",
        "required_closes": 757,
    }
    if (
        policy != expected
        or type(policy["schema_version"]) is not int
        or hashlib.sha256(core_path.read_bytes()).hexdigest() != policy["core_sha256"]
    ):
        raise ValueError("Research product universe policy or its frozen core has changed")
    return policy


@dataclass(frozen=True, slots=True)
class CapturedProductIntake:
    asset: DataAsset
    core_listing_ids: tuple[UUID, ...]
    saved_listing_ids: tuple[UUID, ...]
    candidate_listing_ids: tuple[UUID, ...]
    core_symbols: tuple[str, ...]
    saved_symbols: tuple[str, ...]

    @property
    def requested_symbols(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.core_symbols, *self.saved_symbols)))


def load_product_intake(
    *,
    target_date: date,
    owner_id: str,
    issuance_key: str,
    store: AssetStore,
) -> CapturedProductIntake | None:
    subject = _intake_subject(target_date, owner_id, issuance_key)
    assets = list(
        DataAsset.objects.filter(provider="stanstock", kind=PRODUCT_INTAKE_KIND, subject=subject)
    )
    if len(assets) > 1:
        raise ValueError("Product intake identity is ambiguous")
    return _captured_intake_from_asset(assets[0], store=store) if assets else None


def capture_authorized_product_intake(
    *,
    target_date: date,
    evidence_grade: str,
    issuance_key: str,
    owner: object,
    core_config_path: Path,
    policy_identity: str,
    captured_at: datetime,
    store: AssetStore,
) -> CapturedProductIntake:
    """Capture the real owner/catalog/provider-authorized product request.

    No credentials are resolved here.  This is the pre-provider boundary:
    Live preferences are read once under the target lock. Catalog resolution
    happens afterwards: an unknown or rejected symbol must not prevent
    capturing the request, or discard the otherwise valid core.
    """
    user_model = get_user_model()
    if not isinstance(owner, user_model):
        raise ValueError("Product intake owner is not an authenticated user")
    if not owner.is_active:
        raise ValueError("Product intake owner is inactive")
    existing = load_product_intake(
        target_date=target_date,
        owner_id=str(owner.pk),
        issuance_key=issuance_key,
        store=store,
    )
    if existing is not None:
        return existing
    record = ProviderRecord.objects.filter(provider=TWELVE_DATA_PROVIDER).first()
    if record is None:
        raise ValueError("Product intake provider authorization is missing")
    validate_provider_usage(record)
    if core_config_path.resolve() != default_us_universe_config_path().resolve():
        raise ValueError("Active research intake requires the reviewed curated core")
    universe_policy = load_product_universe_policy()
    config = load_us_universe_config(core_config_path)
    if len(config.symbols) > 100 or len(config.symbols) > config.maximum_symbols:
        raise ValueError("Configured product core exceeds its reviewed maximum")
    saved_symbols = tuple(
        TrackedSymbol.objects.filter(owner=owner)
        .order_by("symbol", "created_at")
        .values_list("symbol", flat=True)
    )
    if len(saved_symbols) > MAX_SAVED_NAMES:
        raise ValueError("Product intake owner has more than 20 saved symbols")
    entitlement = {
        "provider": record.provider,
        "usage_scope": record.usage_scope,
        "plan": str(record.metadata["plan"]).lower(),
        "owner_id": str(owner.pk),
        "licensed_user_id": record.metadata.get("licensed_user_id"),
        "internal_display_rights_confirmed": record.metadata.get(
            "internal_display_rights_confirmed"
        ),
        "personal_noncommercial_confirmed": record.metadata.get("personal_noncommercial_confirmed"),
    }
    entitlement_identity = hashlib.sha256(
        json.dumps(entitlement, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return capture_product_intake(
        target_date=target_date,
        evidence_grade=evidence_grade,
        issuance_key=issuance_key,
        owner_id=str(owner.pk),
        entitlement_identity=entitlement_identity,
        policy_identity=policy_identity,
        core_listings=(),
        saved_listings=(),
        captured_at=captured_at,
        store=store,
        core_symbols=config.symbols,
        saved_symbols=saved_symbols,
        source_provider=TWELVE_DATA_PROVIDER,
        core_config=config.raw,
        universe_policy=universe_policy,
        entitlement=entitlement,
    )


def capture_product_intake(
    *,
    target_date: date,
    evidence_grade: str,
    issuance_key: str,
    owner_id: str,
    entitlement_identity: str,
    policy_identity: str,
    core_listings: Iterable[Listing],
    saved_listings: Iterable[Listing],
    captured_at: datetime,
    store: AssetStore,
    core_symbols: tuple[str, ...] | None = None,
    saved_symbols: tuple[str, ...] | None = None,
    source_provider: str = "synthetic_demo",
    core_config: dict[str, object] | None = None,
    universe_policy: dict[str, object] | None = None,
    entitlement: dict[str, object] | None = None,
) -> CapturedProductIntake:
    """Capture a deduplicated 100-core-plus-20-saved request before provider work."""
    if not owner_id or not entitlement_identity or not policy_identity:
        raise ValueError("Product intake requires owner, entitlement, and policy identities")
    subject = _intake_subject(target_date, owner_id, issuance_key)
    existing = load_product_intake(
        target_date=target_date, owner_id=owner_id, issuance_key=issuance_key, store=store
    )
    if existing is not None:
        return existing
    if evidence_grade not in {"observed", "research"} or source_provider not in {
        TWELVE_DATA_PROVIDER,
        "synthetic_demo",
    }:
        raise ValueError("Product intake source or evidence grade is invalid")
    if source_provider == "synthetic_demo" and evidence_grade != "research":
        raise ValueError("Synthetic intake cannot claim observed evidence")
    core_rows, saved_rows = tuple(core_listings), tuple(saved_listings)
    core = _unique_ids(core_rows)
    saved = _unique_ids(saved_rows)
    requested_core = (
        tuple(row.provider_symbol for row in core_rows) if core_symbols is None else core_symbols
    )
    requested_saved = (
        tuple(row.provider_symbol for row in saved_rows) if saved_symbols is None else saved_symbols
    )
    if len(core) > 100 or len(requested_core) > 100:
        raise ValueError("Product intake permits at most 100 core listings")
    if len(saved) > MAX_SAVED_NAMES or len(requested_saved) > MAX_SAVED_NAMES:
        raise ValueError("Product intake permits at most 20 saved names")
    candidates = tuple(dict.fromkeys((*core, *saved)))
    payload = _intake_payload(
        target_date=target_date,
        evidence_grade=evidence_grade,
        issuance_key=issuance_key,
        owner_id=owner_id,
        entitlement_identity=entitlement_identity,
        policy_identity=policy_identity,
        core=core,
        saved=saved,
        candidates=candidates,
    )
    payload.update(
        {
            "core_symbols": list(dict.fromkeys(requested_core)),
            "saved_symbols": list(dict.fromkeys(requested_saved)),
            "source_provider": source_provider,
            "core_config": core_config,
            "universe_policy": universe_policy,
            "entitlement": entitlement,
        }
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    path = f"research/intake/{target_date.isoformat()}/{issuance_key}-{digest[:12]}.json"
    stored = store.write_bytes(path, encoded)
    try:
        with transaction.atomic():
            asset = register_asset(
                provider="stanstock",
                kind=PRODUCT_INTAKE_KIND,
                subject=subject,
                stored=stored,
                retrieved_at=captured_at,
                available_at=captured_at,
                metadata={"contract": PRODUCT_INTAKE_CONTRACT},
            )
    except Exception:
        if not DataAsset.objects.filter(relative_path=path).exists():
            store.resolve(path).unlink(missing_ok=True)
        raise
    return CapturedProductIntake(
        asset=asset,
        core_listing_ids=core,
        saved_listing_ids=saved,
        candidate_listing_ids=candidates,
        core_symbols=tuple(dict.fromkeys(requested_core)),
        saved_symbols=tuple(dict.fromkeys(requested_saved)),
    )


@transaction.atomic
def materialize_product_membership(
    *,
    intake: CapturedProductIntake,
    qualified_listing_ids: Iterable[UUID],
    candidate_states: dict[UUID, str],
    captured_at: datetime,
    store: AssetStore,
    admissions: dict[str, dict[str, Any]] | None = None,
    catalog_assets: tuple[DataAsset, ...] = (),
    benchmark_asset: DataAsset | None = None,
) -> UniverseSnapshot:
    """Materialize a new immutable qualified cohort from a captured intake.

    The snapshot uses a capture-specific universe slug.  This preserves the
    existing `(universe, date, grade)` uniqueness invariant while making a
    deliberate same-target intake/reissue a distinct immutable cohort.
    """
    qualified = tuple(dict.fromkeys(qualified_listing_ids))
    candidate_ids = set(intake.candidate_listing_ids)
    if admissions is not None:
        if set(admissions) != set(intake.requested_symbols):
            raise ValueError("Admission must account for every captured requested symbol")
        candidate_ids = {
            UUID(item["listing_id"])
            for item in admissions.values()
            if item.get("listing_id") is not None
        }
    if not set(qualified) <= candidate_ids:
        raise ValueError("Qualified membership is outside the captured intake")
    if set(candidate_states) != candidate_ids:
        raise ValueError("Every captured candidate requires an explicit admission state")
    if any(candidate_states[item] != "admitted" for item in qualified):
        raise ValueError("Only admitted candidates may enter product membership")
    if {item for item, state in candidate_states.items() if state == "admitted"} != set(qualified):
        raise ValueError("Product membership omitted an admitted candidate")
    payload = {
        "contract": PRODUCT_MEMBERSHIP_CONTRACT,
        "intake": asset_ref_for(intake.asset).to_json(),
        "qualified_listing_ids": sorted(str(item) for item in qualified),
        "candidate_states": {
            str(item): candidate_states[item] for item in sorted(candidate_states, key=str)
        },
        "admissions": admissions,
        "catalog_assets": [asset_ref_for(asset).to_json() for asset in catalog_assets],
        "benchmark_asset": (
            None if benchmark_asset is None else asset_ref_for(benchmark_asset).to_json()
        ),
        "decision_time": captured_at.isoformat(),
    }
    config_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    universe, _created = Universe.objects.get_or_create(
        slug=f"{PRODUCT_VERSION}-{intake.asset.sha256[:16]}",
        defaults={
            "name": "Captured research product cohort",
            "description": "Immutable research-product admission cohort.",
            "config_version": PRODUCT_VERSION,
        },
    )
    if universe.config_version != PRODUCT_VERSION:
        raise ValueError("Product cohort universe identity conflicts with existing evidence")
    snapshot, created = UniverseSnapshot.objects.get_or_create(
        universe=universe,
        as_of_date=_intake_target_date(intake.asset, store=store),
        grade=_intake_grade(intake.asset, store=store),
        defaults={"config_hash": config_hash},
    )
    if not created:
        if snapshot.config_hash != config_hash:
            raise ValueError("Product membership snapshot conflicts with captured admission")
        product_membership_payload(snapshot, store=store)
        return snapshot
    listings = {item.id: item for item in Listing.objects.filter(id__in=candidate_ids)}
    if set(listings) != candidate_ids:
        raise ValueError("Captured product listing is no longer registered")
    UniverseMembership.objects.bulk_create(
        [
            UniverseMembership(
                snapshot=snapshot,
                listing=listings[item],
                eligible=candidate_states[item] == "admitted",
                exclusion_reason=(
                    "" if candidate_states[item] == "admitted" else candidate_states[item]
                ),
            )
            for item in sorted(candidate_ids, key=str)
        ]
    )
    payload["snapshot_id"] = str(snapshot.id)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    stored = store.write_bytes(f"research/membership/{snapshot.id}.json", encoded)
    register_asset(
        provider="stanstock",
        kind=PRODUCT_MEMBERSHIP_KIND,
        subject=str(snapshot.id),
        stored=stored,
        retrieved_at=captured_at,
        available_at=captured_at,
        metadata={"contract": PRODUCT_MEMBERSHIP_CONTRACT},
    )
    return snapshot


def product_membership_payload(
    snapshot: UniverseSnapshot, *, store: AssetStore
) -> dict[str, object]:
    """Load the independently registered intake/membership authority."""
    assets = list(
        DataAsset.objects.filter(
            provider="stanstock", kind=PRODUCT_MEMBERSHIP_KIND, subject=str(snapshot.id)
        )
    )
    if len(assets) != 1:
        raise ValueError("Product membership evidence is missing or ambiguous")
    try:
        payload = json.loads(read_checksummed_bytes(store, assets[0]))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("Product membership evidence is malformed") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("contract") != PRODUCT_MEMBERSHIP_CONTRACT
        or payload.get("snapshot_id") != str(snapshot.id)
    ):
        raise ValueError("Product membership evidence has an invalid identity")
    return payload


def _intake_payload(
    *,
    target_date: date,
    evidence_grade: str,
    issuance_key: str,
    owner_id: str,
    entitlement_identity: str,
    policy_identity: str,
    core: tuple[UUID, ...],
    saved: tuple[UUID, ...],
    candidates: tuple[UUID, ...],
) -> dict[str, object]:
    return {
        "contract": PRODUCT_INTAKE_CONTRACT,
        "product_version": PRODUCT_VERSION,
        "target_date": target_date.isoformat(),
        "evidence_grade": evidence_grade,
        "issuance_key": issuance_key,
        "owner_id": owner_id,
        "entitlement_identity": entitlement_identity,
        "policy_identity": policy_identity,
        "core_listing_ids": [str(value) for value in core],
        "saved_listing_ids": [str(value) for value in saved],
        "candidate_listing_ids": [str(value) for value in candidates],
    }


def _captured_intake_from_asset(asset: DataAsset, *, store: AssetStore) -> CapturedProductIntake:
    try:
        payload = json.loads(read_checksummed_bytes(store, asset))
        if not isinstance(payload, dict) or payload.get("contract") != PRODUCT_INTAKE_CONTRACT:
            raise ValueError
        core = tuple(UUID(str(item)) for item in payload["core_listing_ids"])
        saved = tuple(UUID(str(item)) for item in payload["saved_listing_ids"])
        candidates = tuple(UUID(str(item)) for item in payload["candidate_listing_ids"])
        core_symbols = tuple(payload["core_symbols"])
        saved_symbols = tuple(payload["saved_symbols"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Captured product intake asset is malformed") from exc
    if (
        len(saved) > MAX_SAVED_NAMES
        or tuple(dict.fromkeys((*core, *saved))) != candidates
        or len(set(candidates)) != len(candidates)
        or len(core_symbols) > 100
        or len(saved_symbols) > MAX_SAVED_NAMES
        or any(
            not isinstance(symbol, str) or not symbol for symbol in (*core_symbols, *saved_symbols)
        )
        or len(set(core_symbols)) != len(core_symbols)
        or len(set(saved_symbols)) != len(saved_symbols)
        or payload.get("product_version") != PRODUCT_VERSION
        or payload.get("source_provider") not in {TWELVE_DATA_PROVIDER, "synthetic_demo"}
        or asset.subject
        != _intake_subject(
            date.fromisoformat(payload["target_date"]), payload["owner_id"], payload["issuance_key"]
        )
    ):
        raise ValueError("Captured product intake asset violates bounded membership")
    if payload["source_provider"] == TWELVE_DATA_PROVIDER:
        entitlement = payload.get("entitlement")
        core_config = load_us_universe_config(default_us_universe_config_path())
        if (
            not isinstance(entitlement, dict)
            or entitlement.get("provider") != TWELVE_DATA_PROVIDER
            or entitlement.get("owner_id") != payload["owner_id"]
            or hashlib.sha256(
                json.dumps(entitlement, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            != payload["entitlement_identity"]
            or payload.get("universe_policy") != load_product_universe_policy()
            or payload.get("core_config") != core_config.raw
            or core_symbols != core_config.symbols
        ):
            raise ValueError("Captured product entitlement or universe policy is invalid")
    return CapturedProductIntake(
        asset=asset,
        core_listing_ids=core,
        saved_listing_ids=saved,
        candidate_listing_ids=candidates,
        core_symbols=core_symbols,
        saved_symbols=saved_symbols,
    )


def _intake_target_date(asset: DataAsset, *, store: AssetStore) -> date:
    payload = json.loads(read_checksummed_bytes(store, asset))
    return date.fromisoformat(str(payload["target_date"]))


def _intake_grade(asset: DataAsset, *, store: AssetStore) -> str:
    payload = json.loads(read_checksummed_bytes(store, asset))
    value = str(payload["evidence_grade"])
    if value not in {UniverseSnapshot.Grade.RESEARCH, UniverseSnapshot.Grade.OBSERVED}:
        raise ValueError("Captured product intake has invalid evidence grade")
    return value


def _unique_ids(listings: Iterable[Listing]) -> tuple[UUID, ...]:
    values: list[UUID] = []
    seen: set[UUID] = set()
    for listing in listings:
        if not listing.pk:
            raise ValueError("Product intake cannot capture an unsaved listing")
        if listing.pk not in seen:
            seen.add(listing.pk)
            values.append(listing.pk)
    return tuple(values)


def _intake_subject(target_date: date, owner_id: str, issuance_key: str) -> str:
    if (
        not issuance_key
        or len(issuance_key) > 36
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in issuance_key
        )
    ):
        raise ValueError("Product intake requires a safe explicit issuance key")
    if not owner_id or len(owner_id) > 36 or ":" in owner_id:
        raise ValueError("Product intake owner identity is invalid")
    return f"{PRODUCT_VERSION}:{target_date.isoformat()}:{owner_id}:{issuance_key}"


def product_intake_payload(
    intake: CapturedProductIntake, *, store: AssetStore
) -> dict[str, object]:
    _captured_intake_from_asset(intake.asset, store=store)
    payload = json.loads(read_checksummed_bytes(store, intake.asset))
    if not isinstance(payload, dict):
        raise ValueError("Captured product intake is not an object")
    return payload


def verify_product_price_content(
    *,
    asset: DataAsset,
    raw: DataAsset,
    cutoff: datetime,
    target_date: date,
    store: AssetStore,
) -> None:
    """Bind normalized values to their exact raw response, not metadata alone."""
    frame = (
        AsOfData(cutoff, store)
        .price_frame_for_asset_with_diagnostics(asset=asset, through_date=target_date)
        .frame
    )
    if asset.provider == "synthetic_demo":
        payload = json.loads(read_checksummed_bytes(store, raw))
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "research-product-synthetic-prices@1"
            or payload.get("subject") != asset.subject
            or payload.get("currency") != asset.metadata.get("currency")
            or payload.get("adjustment") != asset.metadata.get("adjustment")
            or not isinstance(payload.get("volume_adjustment_compatible"), bool)
            or payload["volume_adjustment_compatible"]
            != asset.metadata.get("volume_adjustment_compatible")
            or not isinstance(payload.get("values"), list)
        ):
            raise ValueError("Synthetic price source has an invalid identity or shape")
        expected = []
        for row in payload["values"]:
            if (
                not isinstance(row, dict)
                or set(row) != {"date", "close", "volume"}
                or not isinstance(row["close"], float)
                or (
                    row["volume"] is not None
                    and (type(row["volume"]) is not int or row["volume"] < 0)
                )
            ):
                raise ValueError("Synthetic raw price row has an invalid shape")
            day = date.fromisoformat(row["date"])
            if day <= target_date:
                expected.append((day, row["close"], row["volume"]))
        if (
            frame.schema.get("close") != pl.Float64
            or frame.schema.get("volume") != pl.Int64
            or frame.select("date", "close", "volume").rows() != expected
        ):
            raise ValueError("Synthetic normalized prices diverge from their raw source")
        return
    parsed = twelve_data.parse_daily_price_series(
        read_checksummed_bytes(store, raw),
        symbol=asset.subject,
        retrieved_at=raw.retrieved_at,
        source_url=str(raw.metadata.get("source_url", "")),
        start_date=asset.period_start,
        end_date=min(target_date, asset.period_end) if asset.period_end else target_date,
        adjustment=str(raw.metadata.get("adjustment", "")),
    )
    metadata = asset.metadata
    if any(
        metadata.get(key) != value
        for key, value in {
            "currency": parsed.currency,
            "exchange": parsed.exchange,
            "mic_code": parsed.mic_code,
            "instrument_type": parsed.instrument_type,
            "adjustment": parsed.adjustment,
        }.items()
    ):
        raise ValueError("Normalized price identity diverges from its registered raw response")
    if (
        frame.schema.get("close") != pl.Float64
        or frame.schema.get("volume") != pl.Int64
        or frame.select("date", "close", "volume").rows()
        != [(bar.trade_date, float(bar.close), bar.volume) for bar in parsed.bars]
    ):
        raise ValueError("Normalized prices diverge from their registered raw response")


def product_catalog_references(
    *,
    catalog_assets: tuple[DataAsset, ...],
    symbols: Iterable[str],
    cutoff: datetime,
    store: AssetStore,
) -> tuple[StockReference, ...]:
    if not catalog_assets or len({asset.subject for asset in catalog_assets}) != len(
        catalog_assets
    ):
        raise ValueError("Product catalog bundle is missing or ambiguous")
    references: list[StockReference] = []
    requested = tuple(symbols)
    for asset in catalog_assets:
        if (
            asset.provider != TWELVE_DATA_PROVIDER
            or asset.kind != "stock_catalog"
            or asset.subject not in {"NASDAQ", "NYSE"}
            or asset.available_at > cutoff
            or asset.retrieved_at > cutoff
        ):
            raise ValueError("Product catalog has an invalid registered identity or cutoff")
        parsed, _count = twelve_data.parse_stock_catalog_references(
            read_checksummed_bytes(store, asset),
            exchange=asset.subject,
            required_symbols=requested,
            require_complete=True,
        )
        if any(reference.exchange != asset.subject for reference in parsed):
            raise ValueError("Product catalog row conflicts with its registered exchange")
        references.extend(parsed)
    return tuple(references)


def verify_product_listing_catalog(
    *,
    listing: Listing,
    stock_asset: DataAsset,
    benchmark_asset: DataAsset,
    catalog_assets: tuple[DataAsset, ...],
    cutoff: datetime,
    store: AssetStore,
) -> None:
    references = product_catalog_references(
        catalog_assets=catalog_assets,
        symbols=(listing.provider_symbol,),
        cutoff=cutoff,
        store=store,
    )
    if len(references) != 1:
        raise ValueError("Product stock does not have one exact official catalog identity")
    reference = references[0]
    supported_types = {
        "Common Stock": Security.SecurityType.COMMON_STOCK,
        "ADR": Security.SecurityType.ADR,
        "American Depositary Receipt": Security.SecurityType.ADR,
        "Depositary Receipt": Security.SecurityType.ADR,
    }
    supported_mics = {"NASDAQ": {"XNAS", "XNGS", "XNCM", "XNMS"}, "NYSE": {"XNYS"}}
    if (
        reference.country != "United States"
        or reference.currency != "USD"
        or reference.exchange not in supported_mics
        or reference.mic_code not in supported_mics[reference.exchange]
        or listing.region != "us"
        or listing.currency != reference.currency
        or listing.exchange_mic != reference.mic_code
        or listing.security.security_type != supported_types.get(reference.instrument_type)
        or stock_asset.subject != reference.symbol
        or any(
            stock_asset.metadata.get(key) != value
            for key, value in {
                "currency": reference.currency,
                "mic_code": reference.mic_code,
                "instrument_type": reference.instrument_type,
                "exchange": reference.exchange,
            }.items()
        )
    ):
        raise ValueError("Product stock identity diverges from its official catalog")
    benchmark = benchmark_asset.metadata
    if (
        benchmark_asset.subject != "SPY"
        or benchmark.get("currency") != "USD"
        or benchmark.get("instrument_type") != "ETF"
        or benchmark.get("mic_code") not in {None, "ARCX"}
        or (
            benchmark.get("mic_code") is None
            and (
                benchmark.get("resolved_mic_code") != "ARCX"
                or benchmark.get("mic_code_source") != "configured_spy_identity"
            )
        )
    ):
        raise ValueError("Product benchmark does not bind the reviewed SPY identity")


def verify_product_intake_membership(
    *,
    snapshot: UniverseSnapshot,
    payload: Mapping[str, Any],
    cutoff: datetime,
    store: AssetStore,
    provider: str,
) -> None:
    from stanstock.core.verification_types import AssetRef
    from stanstock.data.assets import resolve_asset_ref

    ref = AssetRef.from_json(payload["intake"])
    asset = resolve_asset_ref(ref, cutoff=cutoff)
    if asset.provider != "stanstock" or asset.kind != PRODUCT_INTAKE_KIND:
        raise ValueError("Product membership does not bind a registered intake")
    intake = _captured_intake_from_asset(asset, store=store)
    captured = product_intake_payload(intake, store=store)
    if (
        snapshot.universe.slug != f"{PRODUCT_VERSION}-{asset.sha256[:16]}"
        or snapshot.as_of_date.isoformat() != captured["target_date"]
        or snapshot.grade != captured["evidence_grade"]
        or captured["source_provider"] != provider
        or (
            provider == TWELVE_DATA_PROVIDER
            and captured["policy_identity"] != PRODUCT_EFFECTIVE_CONFIG_HASH
        )
    ):
        raise ValueError("Product snapshot identity diverges from captured intake")
    states = payload["candidate_states"]
    qualified = set(payload["qualified_listing_ids"])
    if (
        not isinstance(states, dict)
        or {identifier for identifier, state in states.items() if state == "admitted"} != qualified
    ):
        raise ValueError("Product qualified membership diverges from its admission states")
    if captured["source_provider"] == "synthetic_demo":
        if set(states) != {str(identifier) for identifier in intake.candidate_listing_ids}:
            raise ValueError("Synthetic membership diverges from its captured listing set")
        return
    admissions = payload["admissions"]
    if not isinstance(admissions, dict) or set(admissions) != set(intake.requested_symbols):
        raise ValueError("Product admissions do not account for the captured request")
    raw_catalogs = payload.get("catalog_assets")
    core_config = captured.get("core_config")
    if not isinstance(raw_catalogs, list) or not isinstance(core_config, dict):
        raise ValueError("Product membership has no captured catalog scope")
    catalog_refs = tuple(AssetRef.from_json(item) for item in raw_catalogs)
    if (
        len(catalog_refs) != len(core_config["exchanges"])
        or {ref.subject for ref in catalog_refs} != set(core_config["exchanges"])
        or any(
            ref.provider != TWELVE_DATA_PROVIDER or ref.kind != "stock_catalog"
            for ref in catalog_refs
        )
    ):
        raise ValueError("Product catalog bundle does not cover its captured exchange scope")
    derived_states: dict[str, str] = {}
    for symbol, entry in admissions.items():
        if not isinstance(entry, dict) or entry.get("status") not in {
            "admitted",
            "identity_rejected",
            "entitlement_rejected",
            "insufficient_history",
            "provider_failed",
            "evidence_invalid",
        }:
            raise ValueError("Product candidate admission has an invalid status")
        identifier = entry.get("listing_id")
        if identifier is None:
            if entry["status"] == "admitted" or entry.get("price_asset") is not None:
                raise ValueError("An admitted candidate has no permanent listing")
            continue
        listing = Listing.objects.filter(id=identifier, provider_symbol=symbol).first()
        if listing is None:
            raise ValueError("Product candidate does not resolve to its captured listing")
        if identifier in derived_states and derived_states[identifier] != entry["status"]:
            raise ValueError("Duplicate listing aliases have conflicting admission states")
        derived_states[identifier] = entry["status"]
    if derived_states != states:
        raise ValueError("Product membership states diverge from independently captured candidates")

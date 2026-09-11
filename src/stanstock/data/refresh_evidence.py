"""Membership-evidence contract leaf: envelope shape and exact-one lookup.

Imports nothing from `data.live_us` or any `research` module, so both the
writer (`live_us`) and the reader (`refresh_validation`) can depend on it
without a cycle -- `live_us` transitively imports `research.service`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from stanstock.core.verification_types import AssetRef
from stanstock.data.models import DataAsset, Listing, UniverseSnapshot

MEMBERSHIP_EVIDENCE_CONTRACT = "universe-membership-evidence@1"
#: `DataAsset.kind` for the immutable, checksummed physical payload backing
#: a `UniverseSnapshot.config_hash`.
UNIVERSE_MEMBERSHIP_EVIDENCE_KIND = "universe_membership_evidence"


class UniverseConfigLike(Protocol):
    """The subset of `UsUniverseConfig` the evidence contract needs."""

    @property
    def slug(self) -> str: ...

    @property
    def exchanges(self) -> tuple[str, ...]: ...

    @property
    def symbols(self) -> tuple[str, ...]: ...

    @property
    def raw(self) -> dict[str, Any]: ...


def universe_snapshot_evidence_payload(
    *,
    config: UniverseConfigLike,
    catalog_assets: list[DataAsset],
    listings: dict[str, Listing],
    exclusion_reasons: dict[str, str],
) -> dict[str, Any]:
    """The exact payload `UniverseSnapshot.config_hash` is derived from.

    Shared, byte-for-byte, by the snapshot writer (which hashes it to
    produce `config_hash`) and the immutable membership evidence asset
    written alongside a freshly-created snapshot, so a re-derivation of the
    hash from that asset's own physical bytes reproduces the same digest.
    """
    return {
        "config": config.raw,
        "catalog_assets": [
            {"sha256": asset.sha256, "subject": asset.subject} for asset in catalog_assets
        ],
        "members": [
            {
                "listing_id": str(listings[symbol].id),
                "symbol": symbol,
                "eligible": symbol not in exclusion_reasons,
                "exclusion_reason": exclusion_reasons.get(symbol, ""),
            }
            for symbol in config.symbols
        ],
    }


def build_membership_evidence_envelope(
    *, snapshot_id: UUID, hash_payload: dict[str, Any], catalog_assets: list[DataAsset]
) -> dict[str, Any]:
    """The envelope written around `hash_payload`. Its own checksum is
    deliberately never equal to `config_hash` (which hashes only
    `hash_payload`, unchanged), so neither value substitutes for the other.
    """
    return {
        "contract": MEMBERSHIP_EVIDENCE_CONTRACT,
        "snapshot_id": str(snapshot_id),
        "hash_payload": hash_payload,
        "catalog_refs": [
            AssetRef(
                id=asset.id,
                provider=asset.provider,
                kind=asset.kind,
                subject=asset.subject,
                sha256=asset.sha256,
            ).to_json()
            for asset in catalog_assets
        ],
    }


@dataclass(frozen=True, slots=True)
class MembershipEvidenceLookup:
    """Exact-one resolution outcome. `asset` is set only when exactly one
    evidence asset is recorded; otherwise `count` distinguishes a legacy
    snapshot (0) from an ambiguous/corrupt one (>1)."""

    asset: DataAsset | None
    count: int


def membership_evidence_assets_for_snapshot(snapshot: UniverseSnapshot) -> list[DataAsset]:
    return list(
        DataAsset.objects.filter(
            provider="stanstock",
            kind=UNIVERSE_MEMBERSHIP_EVIDENCE_KIND,
            subject=str(snapshot.id),
        )
    )


def lookup_membership_evidence(snapshot: UniverseSnapshot) -> MembershipEvidenceLookup:
    assets = membership_evidence_assets_for_snapshot(snapshot)
    if len(assets) == 1:
        return MembershipEvidenceLookup(asset=assets[0], count=1)
    return MembershipEvidenceLookup(asset=None, count=len(assets))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        seen[key] = value
    return seen


def strict_json_loads(payload: bytes) -> Any:
    """Parse `payload`, rejecting an object with a duplicate key instead of
    silently keeping only the last occurrence (the stdlib default)."""
    return json.loads(payload, object_pairs_hook=_reject_duplicate_keys)

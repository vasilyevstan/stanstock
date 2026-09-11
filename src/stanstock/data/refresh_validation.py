"""Data-domain scheduled-refresh verification: snapshot/market stages.

Called by `core.refresh_verification` after market/SEC children succeed.
Re-derives proof from persisted rows/checksummed bytes, never a child's
self-reported counts alone; imports nothing from `stanstock.research` or
`stanstock.data.live_us` (which transitively imports `research.service`),
so this module stays a safe dependency for any future research-side reader.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any
from uuid import UUID

from stanstock.core.verification_types import (
    AssetRef,
    RefreshVerificationError,
)
from stanstock.data.asof import raw_price_asset_for, verified_price_fields
from stanstock.data.assets import (
    asset_ref_for,
    open_asset_store,
    read_checksummed_bytes,
    verify_catalog_refs,
)
from stanstock.data.etfs import (
    INVESTABLE_US_ETF_MIC,
    INVESTABLE_US_ETF_SYMBOL,
    is_supported_investable_etf,
)
from stanstock.data.management.config_loader import config_hash as raw_config_hash
from stanstock.data.models import (
    DataAsset,
    LatestMarketData,
    Listing,
    Region,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.data.providers import twelve_data
from stanstock.data.refresh_evidence import (
    MEMBERSHIP_EVIDENCE_CONTRACT,
    UniverseConfigLike,
    lookup_membership_evidence,
    strict_json_loads,
    universe_snapshot_evidence_payload,
)

#: The membership-evidence envelope's exact top-level key set -- an
#: extra/missing key fails closed instead of silently passing through a
#: partial `dict.get` check.
_MEMBERSHIP_ENVELOPE_KEYS = frozenset({"contract", "snapshot_id", "hash_payload", "catalog_refs"})


def verify_snapshot(
    market_details: Mapping[str, Any], *, universe_config: UniverseConfigLike, target_date: date
) -> UniverseSnapshot:
    raw_id = market_details.get("snapshot_id")
    if not raw_id:
        raise RefreshVerificationError(
            "snapshot_identity_missing", "Market stage details have no snapshot_id"
        )
    try:
        snapshot_id = uuid.UUID(str(raw_id))
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "snapshot_identity_malformed", "Market stage snapshot_id is not a valid identifier"
        ) from exc
    snapshot = UniverseSnapshot.objects.select_related("universe").filter(pk=snapshot_id).first()
    if snapshot is None:
        raise RefreshVerificationError(
            "snapshot_missing", "Market stage snapshot could not be resolved"
        )
    if snapshot.universe.slug != universe_config.slug:
        raise RefreshVerificationError(
            "snapshot_universe_mismatch",
            "Resolved universe snapshot does not match the reviewed universe configuration",
        )
    if snapshot.as_of_date != target_date:
        raise RefreshVerificationError(
            "snapshot_target_mismatch", "Resolved universe snapshot targets a different date"
        )
    if snapshot.grade != UniverseSnapshot.Grade.OBSERVED:
        raise RefreshVerificationError(
            "snapshot_not_observed", "Resolved universe snapshot is not observed-grade"
        )
    return snapshot


def assert_no_etf_in_membership(memberships: list[UniverseMembership]) -> None:
    """The benchmark ETF (proved separately) may never double as a plain member."""
    for membership in memberships:
        if is_supported_investable_etf(membership.listing):
            raise RefreshVerificationError(
                "membership_contains_investable_etf",
                "Universe membership includes the benchmark ETF as an ordinary stock member",
            )


def resolve_catalog_assets(
    market_details: Mapping[str, Any], *, universe_config: UniverseConfigLike, cutoff: datetime
) -> list[DataAsset]:
    """Resolve the exact, ordered catalog assets a target's snapshot used."""
    raw_ids = market_details.get("catalog_asset_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise RefreshVerificationError(
            "catalog_asset_ids_missing",
            "Market stage details have no recorded catalog asset identities",
        )
    if len(raw_ids) != len(universe_config.exchanges):
        raise RefreshVerificationError(
            "catalog_asset_ids_count_mismatch",
            "Market stage catalog asset identities do not match the reviewed exchange list",
        )
    try:
        asset_ids = [uuid.UUID(str(value)) for value in raw_ids]
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError(
            "catalog_asset_id_malformed",
            "Market stage catalog_asset_ids contains an invalid identifier",
        ) from exc
    if len(set(asset_ids)) != len(asset_ids):
        raise RefreshVerificationError(
            "catalog_asset_id_duplicated",
            "Market stage catalog_asset_ids contains a duplicate identifier",
        )
    assets_by_id = {asset.id: asset for asset in DataAsset.objects.filter(pk__in=asset_ids)}
    if len(assets_by_id) != len(asset_ids):
        raise RefreshVerificationError(
            "catalog_asset_missing", "One or more recorded catalog assets could not be resolved"
        )
    ordered_assets = [assets_by_id[asset_id] for asset_id in asset_ids]
    for exchange, asset in zip(universe_config.exchanges, ordered_assets, strict=True):
        if asset.provider != twelve_data.PROVIDER or asset.kind != "stock_catalog":
            raise RefreshVerificationError(
                "catalog_asset_identity_mismatch",
                "A recorded catalog asset is not an eligible stock catalog",
            )
        if asset.subject != exchange:
            raise RefreshVerificationError(
                "catalog_asset_exchange_mismatch",
                "A recorded catalog asset does not match its reviewed exchange",
            )
        if asset.available_at > cutoff or asset.retrieved_at > cutoff:
            raise RefreshVerificationError(
                "catalog_asset_after_cutoff",
                "A recorded catalog asset was admitted after the verified run's data cutoff",
            )
    verify_catalog_refs(
        ordered_assets,
        market_details.get("catalog_refs"),
        reason_code="catalog_ref_market_mismatch",
    )
    return ordered_assets


def verify_membership_evidence(
    snapshot: UniverseSnapshot,
    memberships: list[UniverseMembership],
    *,
    universe_config: UniverseConfigLike,
    catalog_assets: list[DataAsset],
    cutoff: datetime,
) -> AssetRef:
    """Recompute `config_hash`, then bind it to the immutable evidence envelope
    (written once, at creation) so co-mutating `UniverseMembership` and
    `config_hash` together cannot self-authenticate."""
    if len(memberships) != len(universe_config.symbols):
        raise RefreshVerificationError(
            "membership_count_mismatch",
            "Universe membership count does not match the reviewed universe configuration",
        )
    memberships_by_symbol: dict[str, UniverseMembership] = {}
    for membership in memberships:
        symbol = membership.listing.provider_symbol
        if symbol in memberships_by_symbol:
            raise RefreshVerificationError(
                "membership_symbol_duplicated",
                "Universe membership references the same provider symbol more than once",
            )
        if membership.eligible and membership.exclusion_reason:
            raise RefreshVerificationError(
                "membership_exclusion_reason_invalid",
                "An eligible membership record has a non-blank exclusion reason",
            )
        if not membership.eligible and not membership.exclusion_reason:
            raise RefreshVerificationError(
                "membership_exclusion_reason_invalid",
                "An ineligible membership record has a blank exclusion reason",
            )
        memberships_by_symbol[symbol] = membership

    listings_by_symbol: dict[str, Listing] = {}
    exclusion_reasons: dict[str, str] = {}
    for symbol in universe_config.symbols:
        matched = memberships_by_symbol.get(symbol)
        if matched is None:
            raise RefreshVerificationError(
                "membership_symbol_missing",
                "A configured universe symbol has no persisted membership",
            )
        listings_by_symbol[symbol] = matched.listing
        if not matched.eligible:
            exclusion_reasons[symbol] = matched.exclusion_reason

    current_payload = universe_snapshot_evidence_payload(
        config=universe_config,
        catalog_assets=catalog_assets,
        listings=listings_by_symbol,
        exclusion_reasons=exclusion_reasons,
    )
    if raw_config_hash(current_payload) != snapshot.config_hash:
        raise RefreshVerificationError(
            "snapshot_config_hash_mismatch",
            "Recomputed universe snapshot evidence does not match its immutable config hash",
        )

    lookup = lookup_membership_evidence(snapshot)
    if lookup.count == 0:
        raise RefreshVerificationError(
            "legacy_snapshot_unverifiable",
            "Universe snapshot has no immutable membership evidence asset",
        )
    if lookup.count > 1:
        raise RefreshVerificationError(
            "membership_evidence_ambiguous",
            "Universe snapshot has more than one immutable membership evidence asset",
        )
    assert lookup.asset is not None
    evidence_asset = lookup.asset
    if evidence_asset.available_at > cutoff or evidence_asset.retrieved_at > cutoff:
        raise RefreshVerificationError(
            "membership_evidence_after_cutoff",
            "The snapshot's immutable membership evidence asset was admitted after the "
            "verified cutoff",
        )

    store = open_asset_store()
    try:
        payload_bytes = read_checksummed_bytes(store, evidence_asset)
    except RefreshVerificationError as exc:
        reason = (
            "membership_evidence_asset_unreadable"
            if exc.reason_code == "asset_unreadable"
            else "membership_evidence_asset_corrupt"
        )
        raise RefreshVerificationError(
            reason, "The snapshot's immutable membership evidence asset could not be verified"
        ) from exc
    try:
        envelope = strict_json_loads(payload_bytes)
    except ValueError as exc:
        raise RefreshVerificationError(
            "membership_evidence_asset_malformed",
            "The snapshot's immutable membership evidence asset is not valid JSON",
        ) from exc
    if (
        not isinstance(envelope, dict)
        or set(envelope) != _MEMBERSHIP_ENVELOPE_KEYS
        or envelope.get("contract") != MEMBERSHIP_EVIDENCE_CONTRACT
        or envelope.get("snapshot_id") != str(snapshot.id)
    ):
        raise RefreshVerificationError(
            "membership_evidence_envelope_invalid",
            "The snapshot's immutable membership evidence envelope has an unexpected shape",
        )
    hash_payload = envelope.get("hash_payload")
    if not isinstance(hash_payload, dict) or raw_config_hash(hash_payload) != snapshot.config_hash:
        raise RefreshVerificationError(
            "membership_evidence_hash_mismatch",
            "The snapshot's immutable membership evidence does not reproduce its own config hash",
        )
    if hash_payload != current_payload:
        raise RefreshVerificationError(
            "membership_evidence_diverged",
            "Current universe membership/catalog evidence no longer matches the immutable "
            "evidence recorded at snapshot creation",
        )
    verify_catalog_refs(
        catalog_assets, envelope.get("catalog_refs"), reason_code="membership_catalog_ref_mismatch"
    )
    return asset_ref_for(evidence_asset)


def require_spy_listing() -> Listing:
    listing = (
        Listing.objects.select_related("security__company")
        .filter(
            ticker=INVESTABLE_US_ETF_SYMBOL,
            provider_symbol=INVESTABLE_US_ETF_SYMBOL,
            exchange_mic=INVESTABLE_US_ETF_MIC,
            region=Region.US,
            is_active=True,
        )
        .first()
    )
    if listing is None or not is_supported_investable_etf(listing):
        raise RefreshVerificationError(
            "spy_listing_unsupported", "No supported investable SPY listing is registered"
        )
    return listing


def require_bound_market_data(
    listing: Listing,
    *,
    target_date: date,
    cutoff: datetime,
    expected_asset_id: UUID,
    expected_subject: str,
) -> tuple[LatestMarketData, DataAsset]:
    """Prove current market data reflects the exact asset the verified run used."""
    latest = LatestMarketData.objects.filter(listing=listing).select_related("source_asset").first()
    if latest is None:
        raise RefreshVerificationError(
            "latest_market_data_missing", f"No current market data exists for listing {listing.pk}"
        )
    if latest.session_date != target_date:
        raise RefreshVerificationError(
            "latest_market_data_stale_or_future",
            f"Current market data for listing {listing.pk} is not dated the scheduled target date",
        )
    if latest.source_asset_id != expected_asset_id:
        raise RefreshVerificationError(
            "latest_market_data_unbound",
            f"Current market data for listing {listing.pk} does not reference the exact price "
            "asset the verified run's own evidence names",
        )
    asset = latest.source_asset
    if (
        asset.provider != twelve_data.PROVIDER
        or asset.subject != expected_subject
        or asset.kind != "price_history"
    ):
        raise RefreshVerificationError(
            "latest_market_data_identity_mismatch",
            f"Current market data for listing {listing.pk} does not reference an "
            "eligible price asset",
        )
    if latest.observed_at != asset.retrieved_at or latest.observed_at > cutoff:
        raise RefreshVerificationError(
            "latest_market_data_observed_at_invalid",
            f"Current market data observed_at for listing {listing.pk} does not match its "
            "bound price asset's own retrieval, or was admitted after the verified cutoff",
        )
    close_places = LatestMarketData._meta.get_field("close").decimal_places
    fields = verified_price_fields(
        asset, cutoff=cutoff, target_date=target_date, close_places=close_places
    )
    if fields.close != latest.close:
        raise RefreshVerificationError(
            "latest_market_data_close_mismatch",
            f"Current market data close for listing {listing.pk} does not match its verified asset",
        )
    if latest.volume != fields.volume:
        raise RefreshVerificationError(
            "latest_market_data_volume_mismatch",
            f"Current market data volume for listing {listing.pk} does not match its "
            "verified asset",
        )
    if (latest.previous_close is None) != (fields.previous_close is None) or (
        latest.previous_close is not None and latest.previous_close != fields.previous_close
    ):
        raise RefreshVerificationError(
            "latest_market_data_previous_close_mismatch",
            f"Current market data previous_close for listing {listing.pk} does not match its "
            "verified asset",
        )
    raw_asset = raw_price_asset_for(asset, cutoff=cutoff)
    return latest, raw_asset

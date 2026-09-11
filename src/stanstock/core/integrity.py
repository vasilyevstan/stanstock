from __future__ import annotations

import hashlib
from dataclasses import dataclass

from django.db.models import QuerySet

from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset


@dataclass(frozen=True, slots=True)
class AssetIntegrityFailure:
    asset_id: str
    relative_path: str
    reason: str


@dataclass(frozen=True, slots=True)
class AssetIntegrityReport:
    checked: int
    failures: tuple[AssetIntegrityFailure, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def verify_registered_assets(
    *,
    store: AssetStore | None = None,
    assets: QuerySet[DataAsset] | None = None,
) -> AssetIntegrityReport:
    try:
        asset_store = store or AssetStore()
    except (OSError, ValueError):
        # The store itself (its configured root) could not be opened; no
        # asset was checked, so report one path-free, asset-free failure
        # rather than letting a path-bearing exception escape.
        return AssetIntegrityReport(
            checked=0,
            failures=(
                AssetIntegrityFailure(asset_id="", relative_path="", reason="store_unavailable"),
            ),
        )
    queryset = assets if assets is not None else DataAsset.objects.order_by("relative_path")
    checked = 0
    failures: list[AssetIntegrityFailure] = []

    for asset in queryset.iterator():
        checked += 1
        # `resolve`/`is_file`/`open`/hashing can all raise an `OSError` (a
        # missing parent directory, a permission failure, a broken symlink)
        # or `resolve`'s own confinement `ValueError`, and every one of
        # those exceptions' default messages embeds the resolved/absolute
        # path. None of that may ever propagate out of this function: it is
        # reported as an ordinary integrity failure instead, exactly like a
        # missing file or a checksum mismatch.
        try:
            path = asset_store.resolve(asset.relative_path)
            is_file = path.is_file()
        except (OSError, ValueError):
            failures.append(
                AssetIntegrityFailure(
                    asset_id=str(asset.pk),
                    relative_path=asset.relative_path,
                    reason="unreadable",
                )
            )
            continue
        if not is_file:
            failures.append(
                AssetIntegrityFailure(
                    asset_id=str(asset.pk),
                    relative_path=asset.relative_path,
                    reason="missing",
                )
            )
            continue
        try:
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
        except (OSError, ValueError):
            failures.append(
                AssetIntegrityFailure(
                    asset_id=str(asset.pk),
                    relative_path=asset.relative_path,
                    reason="unreadable",
                )
            )
            continue
        if digest != asset.sha256:
            failures.append(
                AssetIntegrityFailure(
                    asset_id=str(asset.pk),
                    relative_path=asset.relative_path,
                    reason="checksum_mismatch",
                )
            )

    return AssetIntegrityReport(checked=checked, failures=tuple(failures))

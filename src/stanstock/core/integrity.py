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
    asset_store = store or AssetStore()
    queryset = assets if assets is not None else DataAsset.objects.order_by("relative_path")
    checked = 0
    failures: list[AssetIntegrityFailure] = []

    for asset in queryset.iterator():
        checked += 1
        path = asset_store.resolve(asset.relative_path)
        if not path.is_file():
            failures.append(
                AssetIntegrityFailure(
                    asset_id=str(asset.pk),
                    relative_path=asset.relative_path,
                    reason="missing",
                )
            )
            continue
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != asset.sha256:
            failures.append(
                AssetIntegrityFailure(
                    asset_id=str(asset.pk),
                    relative_path=asset.relative_path,
                    reason="checksum_mismatch",
                )
            )

    return AssetIntegrityReport(checked=checked, failures=tuple(failures))

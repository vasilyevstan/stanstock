from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl
from django.conf import settings
from django.utils import timezone

from stanstock.data.models import DataAsset


@dataclass(frozen=True, slots=True)
class StoredAsset:
    relative_path: str
    sha256: str
    byte_count: int


class AssetConflictError(ValueError):
    """Raised when writing would silently overwrite a different existing asset.

    `DataAsset` rows treat `relative_path` as an immutable vintage's natural
    key, so the file it points at must never silently change underneath it.
    Writing the exact same bytes to an already-written path is a safe,
    idempotent no-op (this is what makes reruns of ``seed_demo`` and similar
    commands safe); writing *different* bytes to that same path is always a
    bug -- either a hash/path collision or a caller trying to mutate history
    -- and must fail loudly instead of corrupting a vintage other rows/files
    may already reference.
    """


class AssetStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or settings.DATA_DIR).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str) -> Path:
        target = (self.root / relative_path).resolve()
        if self.root not in target.parents:
            raise ValueError("Asset path escapes STANSTOCK_DATA_DIR")
        return target

    def write_bytes(self, relative_path: str, payload: bytes) -> StoredAsset:
        target = self.resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(payload).hexdigest()

        if target.exists():
            existing_digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if existing_digest != digest:
                raise AssetConflictError(
                    f"Refusing to overwrite existing asset at {relative_path!r}: "
                    f"on-disk sha256 {existing_digest} does not match the sha256 "
                    f"{digest} of the bytes being written. Same path with "
                    "identical bytes is treated as an idempotent no-op; same "
                    "path with different bytes is never allowed, since "
                    "relative_path is an immutable vintage's natural key."
                )
            # Identical bytes already on disk: idempotent no-op, do not
            # rewrite the file (and therefore do not touch its mtime/inode).
            return StoredAsset(relative_path, digest, len(payload))

        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp_file:
            temp_file.write(payload)
            temp_path = Path(temp_file.name)

        os.replace(temp_path, target)
        return StoredAsset(relative_path, digest, len(payload))

    def read_bytes(self, relative_path: str) -> bytes:
        return self.resolve(relative_path).read_bytes()

    def write_frame(self, relative_path: str, frame: pl.DataFrame) -> StoredAsset:
        target = self.resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            suffix=".parquet",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            frame.write_parquet(temp_path)
            payload = temp_path.read_bytes()
            return self.write_bytes(relative_path, payload)
        finally:
            temp_path.unlink(missing_ok=True)

    def read_frame(self, relative_path: str) -> pl.DataFrame:
        return pl.read_parquet(self.resolve(relative_path))


def register_asset(
    *,
    provider: str,
    kind: str,
    subject: str,
    stored: StoredAsset,
    retrieved_at: datetime | None = None,
    available_at: datetime | None = None,
    period_start: date | None = None,
    period_end: date | None = None,
    metadata: dict[str, object] | None = None,
) -> DataAsset:
    now = timezone.now()
    return DataAsset.objects.create(
        provider=provider,
        kind=kind,
        subject=subject,
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        retrieved_at=retrieved_at or now,
        available_at=available_at or now,
        period_start=period_start,
        period_end=period_end,
        metadata=metadata or {},
    )

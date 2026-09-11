"""Zero-domain-dependency leaf types for scheduled-refresh verification.

Never import Django, any `stanstock.data`/`research`/`portfolio` module,
Polars, or `Decimal`: this is the bottom of the verification dependency
graph every domain validator and the thin orchestrator build on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

_ASSET_REF_FIELDS = frozenset({"id", "provider", "kind", "subject", "sha256"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MALFORMED = "asset_ref_malformed"


class RefreshVerificationError(ValueError):
    """A path-free, reason-coded scheduled-refresh verification failure."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code

    def to_failure_details(self) -> dict[str, str]:
        return {"status": "failed", "reason_code": self.reason_code, "message": str(self)}


@dataclass(frozen=True, slots=True)
class AssetRef:
    """One immutable, checksummed asset identity claim (no timestamp)."""

    id: UUID
    provider: str
    kind: str
    subject: str
    sha256: str

    def to_json(self) -> dict[str, str]:
        return {
            "id": str(self.id),
            "provider": self.provider,
            "kind": self.kind,
            "subject": self.subject,
            "sha256": self.sha256,
        }

    @classmethod
    def from_json(cls, raw: object) -> AssetRef:
        if not isinstance(raw, dict) or set(raw) != _ASSET_REF_FIELDS:
            raise RefreshVerificationError(_MALFORMED, "Asset reference payload shape is invalid")
        id_value, provider, kind, subject, sha256 = (
            raw["id"],
            raw["provider"],
            raw["kind"],
            raw["subject"],
            raw["sha256"],
        )
        if not all(isinstance(v, str) and v for v in (id_value, provider, kind, subject, sha256)):
            raise RefreshVerificationError(
                _MALFORMED, "Asset reference has a blank or non-string field"
            )
        try:
            asset_id = UUID(id_value)
        except ValueError as exc:
            raise RefreshVerificationError(
                _MALFORMED, "Asset reference id is not a valid UUID"
            ) from exc
        if not _SHA256_RE.fullmatch(sha256):
            raise RefreshVerificationError(
                _MALFORMED, "Asset reference sha256 is not 64 hex characters"
            )
        return cls(id=asset_id, provider=provider, kind=kind, subject=subject, sha256=sha256)


@dataclass(frozen=True, slots=True)
class StageVerificationResult:
    """One domain validator's path-free summary plus the assets it proved."""

    summary: dict[str, object]
    asset_refs: tuple[AssetRef, ...]

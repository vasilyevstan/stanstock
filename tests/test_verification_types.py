from __future__ import annotations

from uuid import UUID

import pytest

from stanstock.core.verification_types import (
    AssetRef,
    RefreshVerificationError,
    StageVerificationResult,
)

VALID_ID = "8a2f9e2e-2222-4444-8888-000000000001"
VALID_SHA256 = "a" * 64


def _payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": VALID_ID,
        "provider": "twelve_data",
        "kind": "price_history",
        "subject": "AAA",
        "sha256": VALID_SHA256,
    }
    base.update(overrides)
    return base


def test_refresh_verification_error_emits_path_free_stable_shape() -> None:
    error = RefreshVerificationError("stage_not_success", "listing 7's stage did not succeed")
    assert isinstance(error, ValueError)
    assert error.reason_code == "stage_not_success"
    assert error.to_failure_details() == {
        "status": "failed",
        "reason_code": "stage_not_success",
        "message": "listing 7's stage did not succeed",
    }


def test_asset_ref_round_trips_and_is_frozen_hashable() -> None:
    ref = AssetRef.from_json(_payload())
    assert ref == AssetRef(
        id=UUID(VALID_ID),
        provider="twelve_data",
        kind="price_history",
        subject="AAA",
        sha256=VALID_SHA256,
    )
    assert ref.to_json() == _payload()
    with pytest.raises(AttributeError):
        ref.sha256 = "1" * 64  # type: ignore[misc]
    assert len({ref, AssetRef.from_json(_payload())}) == 1


@pytest.mark.parametrize(
    "raw",
    [
        "not-a-dict",
        {},
        {**_payload(), "extra": "unexpected"},
        _payload(provider=""),
        _payload(id="not-a-uuid"),
        _payload(sha256="a" * 63),
        _payload(sha256="g" * 64),
    ],
)
def test_asset_ref_from_json_rejects_malformed_payloads(raw: object) -> None:
    with pytest.raises(RefreshVerificationError) as excinfo:
        AssetRef.from_json(raw)
    assert excinfo.value.reason_code == "asset_ref_malformed"


def test_stage_verification_result_is_frozen_and_holds_ordered_refs() -> None:
    ref_a, ref_b = AssetRef.from_json(_payload()), AssetRef.from_json(_payload(subject="BBB"))
    result = StageVerificationResult(summary={"status": "verified"}, asset_refs=(ref_a, ref_b))
    assert result.asset_refs == (ref_a, ref_b)
    with pytest.raises(AttributeError):
        result.summary = {}  # type: ignore[misc]

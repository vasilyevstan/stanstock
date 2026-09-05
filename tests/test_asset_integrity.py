from __future__ import annotations

from pathlib import Path

import pytest
from django.utils import timezone

from stanstock.core.integrity import verify_registered_assets
from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset


@pytest.mark.django_db
def test_asset_integrity_reports_missing_and_mismatched_files(tmp_path: Path) -> None:
    store = AssetStore(tmp_path)
    now = timezone.now()
    correct = store.write_bytes("raw/correct.json", b'{"ok":true}')
    corrupt = store.write_bytes("raw/corrupt.json", b"original")

    DataAsset.objects.create(
        provider="synthetic",
        kind="raw",
        subject="correct",
        relative_path=correct.relative_path,
        sha256=correct.sha256,
        retrieved_at=now,
        available_at=now,
    )
    DataAsset.objects.create(
        provider="synthetic",
        kind="raw",
        subject="corrupt",
        relative_path=corrupt.relative_path,
        sha256=corrupt.sha256,
        retrieved_at=now,
        available_at=now,
    )
    DataAsset.objects.create(
        provider="synthetic",
        kind="raw",
        subject="missing",
        relative_path="raw/missing.json",
        sha256="0" * 64,
        retrieved_at=now,
        available_at=now,
    )
    store.resolve(corrupt.relative_path).write_bytes(b"changed")

    report = verify_registered_assets(store=store)

    assert report.checked == 3
    assert not report.ok
    assert {(failure.relative_path, failure.reason) for failure in report.failures} == {
        ("raw/corrupt.json", "checksum_mismatch"),
        ("raw/missing.json", "missing"),
    }

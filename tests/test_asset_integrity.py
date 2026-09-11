from __future__ import annotations

from pathlib import Path

import pytest
from django.utils import timezone

from stanstock.core.integrity import AssetIntegrityFailure, verify_registered_assets
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


@pytest.mark.django_db
def test_asset_integrity_reports_os_error_as_unreadable_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resolve`/`is_file`/`open`/hashing failures (a permission error, a
    confinement `ValueError`, a broken symlink) must be reported as an
    ordinary `unreadable` failure, never propagate a path-bearing exception
    out of this function."""
    store = AssetStore(tmp_path)
    now = timezone.now()
    stored = store.write_bytes("raw/ok.json", b'{"ok":true}')
    asset = DataAsset.objects.create(
        provider="synthetic",
        kind="raw",
        subject="ok",
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        retrieved_at=now,
        available_at=now,
    )
    sentinel = "/sentinel-should-never-leak/asset.bin"

    def boom_resolve(self: AssetStore, relative_path: str) -> Path:
        raise OSError(f"[Errno 13] Permission denied: {sentinel!r}")

    monkeypatch.setattr(AssetStore, "resolve", boom_resolve)

    report = verify_registered_assets(store=store)

    assert report.checked == 1
    assert report.failures == (
        AssetIntegrityFailure(
            asset_id=str(asset.pk), relative_path=stored.relative_path, reason="unreadable"
        ),
    )
    assert all(sentinel not in str(field) for field in report.failures)


@pytest.mark.django_db
def test_asset_integrity_reports_store_constructor_failure_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store-level failure (an unwritable/misconfigured root) must produce
    a path-free, asset-free failure entry rather than raising."""
    sentinel = tmp_path / "sentinel-unwritable-root"

    def boom_init(self: AssetStore, root: Path | None = None) -> None:
        raise OSError(f"Permission denied: {sentinel}")

    monkeypatch.setattr(AssetStore, "__init__", boom_init)

    report = verify_registered_assets()

    assert report.checked == 0
    assert report.failures == (
        AssetIntegrityFailure(asset_id="", relative_path="", reason="store_unavailable"),
    )
    assert str(sentinel) not in str(report.failures)

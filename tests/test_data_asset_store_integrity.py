"""Focused tests for `AssetStore`'s immutable-file invariant.

`relative_path` is the natural key `DataAsset` treats as an immutable
vintage, so the bytes on disk at a given path must never silently change.
Writing identical bytes to an already-written path must be a safe,
idempotent no-op; writing different bytes to that same path must fail
explicitly rather than silently overwriting history.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from stanstock.data.assets import AssetConflictError, AssetStore


def test_write_bytes_same_path_same_content_is_idempotent(tmp_path: Path) -> None:
    store = AssetStore(root=tmp_path)
    payload = b'{"value": 1}'

    first = store.write_bytes("raw/sample.json", payload)
    target = store.resolve("raw/sample.json")
    mtime_after_first_write = target.stat().st_mtime_ns

    second = store.write_bytes("raw/sample.json", payload)

    assert second == first
    assert target.read_bytes() == payload
    # Re-writing identical bytes must not even touch the file on disk.
    assert target.stat().st_mtime_ns == mtime_after_first_write


def test_write_bytes_same_path_different_content_fails_explicitly(tmp_path: Path) -> None:
    store = AssetStore(root=tmp_path)
    store.write_bytes("raw/sample.json", b'{"value": 1}')

    with pytest.raises(AssetConflictError, match="Refusing to overwrite"):
        store.write_bytes("raw/sample.json", b'{"value": 2}')

    # The original bytes must survive the failed write attempt untouched.
    assert store.resolve("raw/sample.json").read_bytes() == b'{"value": 1}'


def test_write_frame_same_path_different_content_fails_explicitly(tmp_path: Path) -> None:
    store = AssetStore(root=tmp_path)
    original = pl.DataFrame({"value": [1, 2, 3]})
    changed = pl.DataFrame({"value": [1, 2, 4]})

    store.write_frame("prices/sample.parquet", original)

    with pytest.raises(AssetConflictError, match="Refusing to overwrite"):
        store.write_frame("prices/sample.parquet", changed)

    # The original frame must survive the failed write attempt untouched.
    assert store.read_frame("prices/sample.parquet")["value"].to_list() == [1, 2, 3]


def test_write_frame_same_path_same_content_is_idempotent(tmp_path: Path) -> None:
    store = AssetStore(root=tmp_path)
    frame = pl.DataFrame({"value": [1, 2, 3]})

    first = store.write_frame("prices/sample.parquet", frame)
    second = store.write_frame("prices/sample.parquet", frame)

    assert second.sha256 == first.sha256
    assert store.read_frame("prices/sample.parquet")["value"].to_list() == [1, 2, 3]

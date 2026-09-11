"""Focused regressions for `stanstock.data.sec_refresh_validation`.

Drives the real `run_sec_ingestion` pipeline (not a synthetic evidence
stub) so the SEC-stage validator's exact per-CIK closure -- one
submissions asset, one companyfacts asset, and one history asset per
required filename -- is proven against genuine ingestion output,
including the honest no-op branch, which must still report the exact
history asset it consulted rather than silently omitting it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml
from django.conf import settings

import test_data_sec_ingestion as sec_fixtures
from stanstock.core.integrity import verify_registered_assets
from stanstock.core.models import JobRun
from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import DataAsset, SourceObservationEvent
from stanstock.data.providers import sec
from stanstock.data.sec_config import (
    SecCikConfig,
    load_sec_cik_config,
    load_sec_fundamentals_config,
)
from stanstock.data.sec_evidence import MAPPING_KIND, SUBMISSIONS_HISTORY_KIND, SUBMISSIONS_KIND
from stanstock.data.sec_ingestion import SecIngestionResult, SecRequestBudget, run_sec_ingestion
from stanstock.data.sec_refresh_validation import verify_sec_stage

pytestmark = pytest.mark.django_db

TARGET_DATE = sec_fixtures.RETRIEVED_AT.date()
REQUIRED_HISTORY_FILENAME = "CIK0000320193-submissions-001.json"


def _reviewed_cik_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SecCikConfig:
    cik_path = tmp_path / "cik.yml"
    cik_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "config_version": "test-cik-v1",
                "universe_config_version": "test-us-v1",
                "source_sha256": hashlib.sha256(sec_fixtures.MAPPING_BYTES).hexdigest(),
                "mappings": {
                    "AAPL": {
                        "cik": "0000320193",
                        "exchange": "Nasdaq",
                        "company_name": "Apple Inc.",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("stanstock.data.sec_config.default_sec_cik_mapping_path", lambda: cik_path)
    return load_sec_cik_config()


def _counting_provider(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counts = {"submissions": 0, "history": 0, "companyfacts": 0}

    def fetch_submissions(cik: str) -> object:
        counts["submissions"] += 1
        return sec_fixtures._payload(
            "0000320193", sec_fixtures.SUBMISSIONS_BYTES, "https://x/submissions"
        )

    def fetch_history(filename: str) -> object:
        counts["history"] += 1
        return sec_fixtures._payload(filename, sec_fixtures.HISTORY_BYTES, f"https://x/{filename}")

    def fetch_companyfacts(cik: str) -> object:
        counts["companyfacts"] += 1
        return sec_fixtures._payload(
            "0000320193", sec_fixtures._companyfacts_bytes(), "https://x/companyfacts"
        )

    monkeypatch.setattr(sec, "fetch_submissions", fetch_submissions)
    monkeypatch.setattr(sec, "fetch_submissions_history", fetch_history)
    monkeypatch.setattr(sec, "fetch_companyfacts", fetch_companyfacts)
    return counts


def _sec_details(result: SecIngestionResult, *, cik_config: SecCikConfig) -> dict[str, object]:
    fundamentals_config = load_sec_fundamentals_config()
    return {
        "provider": "sec",
        "mapping_asset_id": result.mapping_asset_id,
        "mapping_sha256": result.mapping_sha256,
        "config_version": fundamentals_config.config_version,
        "config_hash": fundamentals_config.config_hash,
        "cik_config_version": cik_config.config_version,
        "cik_config_hash": cik_config.config_hash,
        "asset_refs": [ref.to_json() for ref in result.asset_refs],
    }


def _job_run(sec_details: dict[str, object], *, attempt: int, finished_at: datetime) -> JobRun:
    return JobRun.objects.create(
        job_name="sec_fundamentals",
        region="us",
        target_date=TARGET_DATE,
        attempt=attempt,
        status=JobRun.Status.SUCCESS,
        finished_at=finished_at,
        details=sec_details,
    )


def _ingest(
    *, cik_config: SecCikConfig, store: AssetStore, budget: SecRequestBudget
) -> SecIngestionResult:
    return run_sec_ingestion(
        config=load_sec_fundamentals_config(),
        cik_config=cik_config,
        universe_config=sec_fixtures._universe(),
        target_date=TARGET_DATE,
        store=store,
        budget=budget,
    )


def test_verify_sec_stage_real_ingestion_initial_and_noop_include_history_asset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    sec_fixtures._listing()
    store = AssetStore(root=tmp_path)
    sec_fixtures._mapping_asset(store)
    cik_config = _reviewed_cik_config(tmp_path, monkeypatch)
    counts = _counting_provider(monkeypatch)
    budget = SecRequestBudget(requests_per_second=5, enforce_spacing=False, require_enabled=False)
    finished_at = sec_fixtures.RETRIEVED_AT + timedelta(hours=1)

    first = _ingest(cik_config=cik_config, store=store, budget=budget)
    assert counts == {"submissions": 1, "history": 1, "companyfacts": 1}
    first_details = _sec_details(first, cik_config=cik_config)
    first_run = _job_run(first_details, attempt=1, finished_at=finished_at)

    first_result = verify_sec_stage(first_details, sec_run=first_run)
    assert sorted(ref.kind for ref in first_result.asset_refs) == sorted(
        ["sec_ticker_mapping", "sec_submissions", "sec_submissions_history", "sec_companyfacts"]
    )

    # A second, honest no-op run (submissions is always re-checked; the
    # bytes are unchanged, so companyfacts/history are never re-fetched)
    # must still report the exact history asset consulted to prove
    # normalization was already complete -- not silently omit it.
    second = _ingest(cik_config=cik_config, store=store, budget=budget)
    assert counts == {"submissions": 2, "history": 1, "companyfacts": 1}
    second_details = _sec_details(second, cik_config=cik_config)
    first_run.delete()
    second_run = _job_run(second_details, attempt=2, finished_at=finished_at)

    second_result = verify_sec_stage(second_details, sec_run=second_run)
    assert sorted(ref.kind for ref in second_result.asset_refs) == sorted(
        ["sec_ticker_mapping", "sec_submissions", "sec_submissions_history", "sec_companyfacts"]
    )
    assert any(
        history_asset.metadata.get("filename") == REQUIRED_HISTORY_FILENAME
        for history_asset in DataAsset.objects.filter(kind=SUBMISSIONS_HISTORY_KIND)
    )


SECOND_HISTORY_FILENAME = "CIK0000320193-submissions-002.json"


def _two_filename_submissions_bytes() -> bytes:
    payload = json.loads(sec_fixtures.SUBMISSIONS_BYTES)
    payload["filings"]["files"] = [
        {"name": REQUIRED_HISTORY_FILENAME, "filingCount": 1},
        {"name": SECOND_HISTORY_FILENAME, "filingCount": 1},
    ]
    return json.dumps(payload, sort_keys=True).encode()


def _counting_provider_two_identical_history_filenames(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    counts = {"submissions": 0, "history": 0, "companyfacts": 0}
    submissions_bytes = _two_filename_submissions_bytes()

    def fetch_submissions(cik: str) -> object:
        counts["submissions"] += 1
        return sec_fixtures._payload("0000320193", submissions_bytes, "https://x/submissions")

    def fetch_history(filename: str) -> object:
        counts["history"] += 1
        # Both required filenames deliberately resolve to byte-identical
        # content at the same retrieval instant -- the exact collision this
        # regression exists to prevent from collapsing onto one asset.
        return sec_fixtures._payload(filename, sec_fixtures.HISTORY_BYTES, f"https://x/{filename}")

    def fetch_companyfacts(cik: str) -> object:
        counts["companyfacts"] += 1
        return sec_fixtures._payload(
            "0000320193", sec_fixtures._companyfacts_bytes(), "https://x/companyfacts"
        )

    monkeypatch.setattr(sec, "fetch_submissions", fetch_submissions)
    monkeypatch.setattr(sec, "fetch_submissions_history", fetch_history)
    monkeypatch.setattr(sec, "fetch_companyfacts", fetch_companyfacts)
    return counts


def test_distinct_history_filenames_with_identical_bytes_get_distinct_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two distinct, required history filenames whose responses happen to be
    byte-identical (same content, same retrieval instant) must never
    collapse onto one `DataAsset`/observation/ref: each filename is its own
    exact evidence identity, and an honest no-op retry must reuse both
    exact refs with zero additional history/companyfacts fetch.
    """
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    sec_fixtures._listing()
    store = AssetStore(root=tmp_path)
    sec_fixtures._mapping_asset(store)
    cik_config = _reviewed_cik_config(tmp_path, monkeypatch)
    counts = _counting_provider_two_identical_history_filenames(monkeypatch)
    budget = SecRequestBudget(requests_per_second=5, enforce_spacing=False, require_enabled=False)
    finished_at = sec_fixtures.RETRIEVED_AT + timedelta(hours=1)

    first = _ingest(cik_config=cik_config, store=store, budget=budget)
    assert counts == {"submissions": 1, "history": 2, "companyfacts": 1}

    history_assets = list(
        DataAsset.objects.filter(kind=SUBMISSIONS_HISTORY_KIND).order_by("relative_path")
    )
    assert len(history_assets) == 2
    assert len({asset.pk for asset in history_assets}) == 2
    assert len({asset.relative_path for asset in history_assets}) == 2
    assert {asset.metadata.get("filename") for asset in history_assets} == {
        REQUIRED_HISTORY_FILENAME,
        SECOND_HISTORY_FILENAME,
    }

    events = list(
        SourceObservationEvent.objects.filter(kind=SUBMISSIONS_HISTORY_KIND).order_by("subject")
    )
    assert [event.subject for event in events] == sorted(
        [f"0000320193:{REQUIRED_HISTORY_FILENAME}", f"0000320193:{SECOND_HISTORY_FILENAME}"]
    )

    first_details = _sec_details(first, cik_config=cik_config)
    first_ref_ids = [ref["id"] for ref in first_details["asset_refs"]]
    assert len(first_ref_ids) == len(set(first_ref_ids))
    first_run = _job_run(first_details, attempt=1, finished_at=finished_at)

    first_result = verify_sec_stage(first_details, sec_run=first_run)
    assert len(first_result.asset_refs) == len({ref.id for ref in first_result.asset_refs})

    # An honest no-op retry reuses both exact history refs (and
    # companyfacts) with zero additional history/companyfacts fetch.
    second = _ingest(cik_config=cik_config, store=store, budget=budget)
    assert counts == {"submissions": 2, "history": 2, "companyfacts": 1}
    second_details = _sec_details(second, cik_config=cik_config)
    assert second_details["asset_refs"] == first_details["asset_refs"]
    first_run.delete()
    second_run = _job_run(second_details, attempt=2, finished_at=finished_at)

    second_result = verify_sec_stage(second_details, sec_run=second_run)
    assert len(second_result.asset_refs) == len({ref.id for ref in second_result.asset_refs})


def _base_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, object], JobRun]:
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    sec_fixtures._listing()
    store = AssetStore(root=tmp_path)
    sec_fixtures._mapping_asset(store)
    cik_config = _reviewed_cik_config(tmp_path, monkeypatch)
    _counting_provider(monkeypatch)
    budget = SecRequestBudget(requests_per_second=5, enforce_spacing=False, require_enabled=False)
    result = _ingest(cik_config=cik_config, store=store, budget=budget)
    finished_at = sec_fixtures.RETRIEVED_AT + timedelta(hours=1)
    details = _sec_details(result, cik_config=cik_config)
    run = _job_run(details, attempt=1, finished_at=finished_at)
    return details, run


def test_verify_sec_stage_rejects_removed_duplicate_extra_and_swapped_refs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    details, run = _base_state(monkeypatch, tmp_path)

    missing_history = dict(details)
    missing_history["asset_refs"] = [
        ref for ref in details["asset_refs"] if ref["kind"] != SUBMISSIONS_HISTORY_KIND
    ]
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_sec_stage(missing_history, sec_run=run)
    assert excinfo.value.reason_code == "sec_history_ref_not_unique"

    duplicated = dict(details)
    history_ref = next(
        ref for ref in details["asset_refs"] if ref["kind"] == SUBMISSIONS_HISTORY_KIND
    )
    duplicated["asset_refs"] = [*details["asset_refs"], history_ref]
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_sec_stage(duplicated, sec_run=run)
    assert excinfo.value.reason_code == "sec_asset_ref_duplicate"

    unrelated_written = AssetStore(root=tmp_path).write_bytes(
        "sec/other-cik-history.json", b'{"unrelated": true}'
    )
    unrelated_asset = register_asset(
        provider=sec.PROVIDER,
        kind=SUBMISSIONS_HISTORY_KIND,
        subject="0000000002",
        stored=unrelated_written,
        retrieved_at=sec_fixtures.RETRIEVED_AT,
        available_at=sec_fixtures.RETRIEVED_AT,
        metadata={"filename": "other.json"},
    )
    extra = dict(details)
    extra["asset_refs"] = [
        *details["asset_refs"],
        {
            "id": str(unrelated_asset.pk),
            "provider": unrelated_asset.provider,
            "kind": unrelated_asset.kind,
            "subject": unrelated_asset.subject,
            "sha256": unrelated_asset.sha256,
        },
    ]
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_sec_stage(extra, sec_run=run)
    assert excinfo.value.reason_code == "sec_asset_ref_unexpected_cik"

    swapped_written = AssetStore(root=tmp_path).write_bytes(
        "sec/swapped-history.json", b'{"swapped": true}'
    )
    swapped_asset = register_asset(
        provider=sec.PROVIDER,
        kind=SUBMISSIONS_HISTORY_KIND,
        subject="0000320193",
        stored=swapped_written,
        retrieved_at=sec_fixtures.RETRIEVED_AT,
        available_at=sec_fixtures.RETRIEVED_AT,
        metadata={"filename": "unexpected-filename.json"},
    )
    swapped = dict(details)
    swapped["asset_refs"] = [
        {
            "id": str(swapped_asset.pk),
            "provider": swapped_asset.provider,
            "kind": swapped_asset.kind,
            "subject": swapped_asset.subject,
            "sha256": swapped_asset.sha256,
        }
        if ref["kind"] == SUBMISSIONS_HISTORY_KIND
        else ref
        for ref in details["asset_refs"]
    ]
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_sec_stage(swapped, sec_run=run)
    assert excinfo.value.reason_code == "sec_history_ref_unexpected"


def test_verify_sec_stage_corrupt_noop_history_asset_fails_final_integrity_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The validator only resolves the history asset's *identity*; a
    corrupted physical file must still be caught by the final physical
    pass over exactly the refs this stage's own closure names."""
    details, run = _base_state(monkeypatch, tmp_path)
    result = verify_sec_stage(details, sec_run=run)

    history_asset = DataAsset.objects.get(
        kind=SUBMISSIONS_HISTORY_KIND, metadata__filename=REQUIRED_HISTORY_FILENAME
    )
    AssetStore(root=tmp_path).resolve(history_asset.relative_path).write_bytes(b"corrupted")

    report = verify_registered_assets(
        assets=DataAsset.objects.filter(pk__in=[ref.id for ref in result.asset_refs])
    )
    assert not report.ok
    assert any(failure.asset_id == str(history_asset.pk) for failure in report.failures)


def test_verify_sec_stage_rejects_reordered_refs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The canonical sequence is mapping, then submissions/history/companyfacts
    per reviewed CIK in config order -- an otherwise-exact but reordered set
    of references must fail closed rather than silently pass a set check."""
    details, run = _base_state(monkeypatch, tmp_path)
    result = verify_sec_stage(details, sec_run=run)
    assert [ref.to_json() for ref in result.asset_refs] == details["asset_refs"]

    reordered = dict(details)
    refs = list(details["asset_refs"])
    mapping_index = next(i for i, ref in enumerate(refs) if ref["kind"] == MAPPING_KIND)
    submissions_index = next(i for i, ref in enumerate(refs) if ref["kind"] == SUBMISSIONS_KIND)
    refs[mapping_index], refs[submissions_index] = refs[submissions_index], refs[mapping_index]
    reordered["asset_refs"] = refs
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_sec_stage(reordered, sec_run=run)
    assert excinfo.value.reason_code == "sec_asset_ref_order_mismatch"


def _swap_submissions_content(
    details: dict[str, object], *, payload: bytes, tmp_path: Path, cik: str = "0000320193"
) -> dict[str, object]:
    store = AssetStore(root=tmp_path)
    written = store.write_bytes(f"sec/{cik}/submissions-swapped-{hash(payload)}.json", payload)
    asset = register_asset(
        provider=sec.PROVIDER,
        kind=SUBMISSIONS_KIND,
        subject=cik,
        stored=written,
        retrieved_at=sec_fixtures.RETRIEVED_AT,
        available_at=sec_fixtures.RETRIEVED_AT,
    )
    swapped_ref = {
        "id": str(asset.pk),
        "provider": asset.provider,
        "kind": asset.kind,
        "subject": asset.subject,
        "sha256": asset.sha256,
    }
    new_details = dict(details)
    assert isinstance(details["asset_refs"], list)
    new_details["asset_refs"] = [
        swapped_ref if ref["kind"] == SUBMISSIONS_KIND else ref for ref in details["asset_refs"]
    ]
    return new_details


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff\xfe\x00invalid",
        b"{not json",
        b'{"filings": {}, "filings": {"files": []}}',
        b'{"filings": {"files": "not-a-list"}}',
        b'{"filings": {"files": [{}]}}',
        b'{"filings": {"files": [{"name": ""}]}}',
        b'{"filings": {"files": [{"name": "a.json"}, {"name": "a.json"}]}}',
        b'{"filings": {"files": [{"name": "../escape.json"}]}}',
    ],
    ids=[
        "invalid_utf8",
        "invalid_json",
        "duplicate_json_key",
        "files_not_a_list",
        "blank_row",
        "blank_name",
        "duplicate_name",
        "unsafe_filename",
    ],
)
def test_verify_sec_stage_normalizes_malformed_submissions_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    payload: bytes,
) -> None:
    """A malformed submissions payload must fail as a stable, path-free
    `RefreshVerificationError` -- never a bare/provider-domain exception,
    and never chained (so no logger can print a leaked cause)."""
    import logging

    details, run = _base_state(monkeypatch, tmp_path)
    malformed = _swap_submissions_content(details, payload=payload, tmp_path=tmp_path)

    logger = logging.getLogger("test-path-free-sec-submissions")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            verify_sec_stage(malformed, sec_run=run)
        except RefreshVerificationError as exc:
            assert exc.reason_code == "sec_submissions_evidence_malformed"
            assert exc.__cause__ is None
            logger.exception("verification failed")
        else:
            pytest.fail("expected RefreshVerificationError")
    assert caplog.text
    assert str(tmp_path) not in caplog.text


def test_verify_sec_stage_valid_empty_files_list_is_not_malformed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An honest zero-history submissions payload is a valid, not malformed,
    shape -- it must fail (if at all) on the history-closure mismatch it
    creates here, never on the parser itself."""
    details, run = _base_state(monkeypatch, tmp_path)
    empty_history = _swap_submissions_content(
        details, payload=b'{"filings": {"files": []}}', tmp_path=tmp_path
    )
    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_sec_stage(empty_history, sec_run=run)
    assert excinfo.value.reason_code == "sec_history_ref_unexpected"


def test_verify_sec_stage_normalizes_asset_store_constructor_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A store-level failure (an unwritable/misconfigured root) must not leak
    a path-bearing `OSError`/`ValueError` out of the SEC-stage validator."""
    details, run = _base_state(monkeypatch, tmp_path)
    sentinel = tmp_path / "sentinel-unwritable-root"

    def fail_mkdir(*args: object, **kwargs: object) -> None:
        raise OSError(f"Permission denied: {sentinel}")

    monkeypatch.setattr(settings, "DATA_DIR", sentinel)
    monkeypatch.setattr(Path, "mkdir", fail_mkdir)

    with pytest.raises(RefreshVerificationError) as excinfo:
        verify_sec_stage(details, sec_run=run)
    assert excinfo.value.reason_code == "asset_store_unavailable"
    assert excinfo.value.__cause__ is None
    assert str(sentinel) not in str(excinfo.value)

"""Focused tests for the cycle-free `stanstock.data.sec_evidence` leaf."""

from __future__ import annotations

import inspect
import json

import pytest

from stanstock.data import providers, sec_ingestion, sec_refresh_validation
from stanstock.data.providers import sec as sec_provider
from stanstock.data.sec_evidence import (
    HISTORY_FILENAME_METADATA_KEY,
    MAPPING_SUBJECT,
    SecEvidencePayloadError,
    historical_submission_filenames,
    is_safe_history_filename,
)


def test_mapping_subject_is_not_hand_synced() -> None:
    """The provider, writer, and reader must import `MAPPING_SUBJECT` from
    the leaf rather than each redeclaring the same literal string."""
    for module in (sec_provider, sec_ingestion, sec_refresh_validation):
        source = inspect.getsource(module)
        assert f'"{MAPPING_SUBJECT}"' not in source
    assert providers.sec.MAPPING_SUBJECT is MAPPING_SUBJECT


def test_history_filename_metadata_key_is_imported_not_duplicated() -> None:
    """The writer and reader must read/write history metadata through the
    single imported `HISTORY_FILENAME_METADATA_KEY`, not a hand-typed key."""
    for module in (sec_ingestion, sec_refresh_validation):
        assert module.HISTORY_FILENAME_METADATA_KEY is HISTORY_FILENAME_METADATA_KEY
        source = inspect.getsource(module)
        assert 'metadata.get("filename")' not in source
        assert 'extra={"filename":' not in source


def test_valid_payload_preserves_source_order() -> None:
    payload = json.dumps(
        {
            "filings": {
                "files": [
                    {"name": "b-002.json"},
                    {"name": "a-001.json"},
                ]
            }
        }
    ).encode("utf-8")
    assert historical_submission_filenames(payload) == ("b-002.json", "a-001.json")


def test_valid_empty_files_list_returns_empty_tuple() -> None:
    payload = json.dumps({"filings": {"files": []}}).encode("utf-8")
    assert historical_submission_filenames(payload) == ()


@pytest.mark.parametrize("name", ["001.json", "a_b-1.2.json", "CIK0000320193-submissions.json"])
def test_is_safe_history_filename_accepts_plain_basenames(name: str) -> None:
    assert is_safe_history_filename(name)


@pytest.mark.parametrize(
    "name",
    ["../etc/passwd.json", "a/b.json", "a\\b.json", ".", "..", "\x00.json", "\x01name.json"],
)
def test_is_safe_history_filename_rejects_unsafe_names(name: str) -> None:
    assert not is_safe_history_filename(name)


def test_invalid_utf8_fails_closed() -> None:
    with pytest.raises(SecEvidencePayloadError):
        historical_submission_filenames(b"\xff\xfe\x00invalid")


def test_invalid_json_fails_closed() -> None:
    with pytest.raises(SecEvidencePayloadError):
        historical_submission_filenames(b"{not json")


def test_duplicate_json_key_fails_closed() -> None:
    with pytest.raises(SecEvidencePayloadError):
        historical_submission_filenames(b'{"filings": {}, "filings": {"files": []}}')


@pytest.mark.parametrize(
    "payload",
    [
        b"[]",
        b'{"filings": []}',
        b'{"filings": {}}',
        b'{"filings": {"files": "not-a-list"}}',
        b'{"filings": {"files": [1]}}',
        b'{"filings": {"files": [{"name": ""}]}}',
        b'{"filings": {"files": [{"name": "   "}]}}',
        b'{"filings": {"files": [{}]}}',
        b'{"filings": {"files": [{"name": "../escape.json"}]}}',
        b'{"filings": {"files": [{"name": "a.json"}, {"name": "a.json"}]}}',
    ],
)
def test_malformed_submissions_payload_fails_closed(payload: bytes) -> None:
    with pytest.raises(SecEvidencePayloadError):
        historical_submission_filenames(payload)

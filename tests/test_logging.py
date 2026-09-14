from __future__ import annotations

import json
import logging
import sys
from contextlib import ExitStack

import pytest

from stanstock.core.logging import JsonFormatter, suppress_http_client_request_logs


def test_json_formatter_escapes_messages_and_exceptions() -> None:
    try:
        raise ValueError('bad "value"\nnext line')
    except ValueError:
        exception_info = sys.exc_info()
        record = logging.LogRecord(
            name="stanstock.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=12,
            msg='request failed for "%s"',
            args=("sample",),
            exc_info=exception_info,
        )

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "ERROR"
    assert payload["logger"] == "stanstock.test"
    assert payload["message"] == 'request failed for "sample"'
    assert 'bad "value"' in payload["exception"]


def test_overlapping_request_log_scopes_restore_only_after_last_exit(caplog) -> None:
    loggers = (logging.getLogger("httpx"), logging.getLogger("httpcore"))
    original = tuple(logger.level for logger in loggers)
    try:
        loggers[0].setLevel(logging.INFO)
        loggers[1].setLevel(logging.DEBUG)
        with caplog.at_level(logging.DEBUG), ExitStack() as first, ExitStack() as second:
            first.enter_context(suppress_http_client_request_logs())
            second.enter_context(suppress_http_client_request_logs())
            first.close()
            assert all(logger.level == logging.WARNING for logger in loggers)
            loggers[0].info("synthetic private request")
            loggers[1].debug("synthetic private transport details")
            loggers[0].warning("Synthetic actionable transport warning")
        assert tuple(logger.level for logger in loggers) == (logging.INFO, logging.DEBUG)
        assert "synthetic private" not in caplog.text
        assert "Synthetic actionable transport warning" in caplog.text
    finally:
        for logger, level in zip(loggers, original, strict=True):
            logger.setLevel(level)


def test_request_log_scope_restores_levels_and_propagates_exception() -> None:
    loggers = (logging.getLogger("httpx"), logging.getLogger("httpcore"))
    original = tuple(logger.level for logger in loggers)
    with pytest.raises(ValueError, match="synthetic acquisition failure"):
        with suppress_http_client_request_logs():
            raise ValueError("synthetic acquisition failure")
    assert tuple(logger.level for logger in loggers) == original

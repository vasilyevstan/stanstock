from __future__ import annotations

import json
import logging
import sys

from stanstock.core.logging import JsonFormatter


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

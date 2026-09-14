from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from threading import Lock

_http_log_lock = Lock()
_http_log_scopes = 0
_http_log_levels: tuple[tuple[logging.Logger, int], ...] = ()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


@contextmanager
def suppress_http_client_request_logs() -> Iterator[None]:
    """Suppress httpx/httpcore request summaries within a bounded call scope."""
    global _http_log_scopes, _http_log_levels
    with _http_log_lock:
        if _http_log_scopes == 0:
            managed_loggers = (logging.getLogger("httpx"), logging.getLogger("httpcore"))
            _http_log_levels = tuple((logger, logger.level) for logger in managed_loggers)
            for logger in managed_loggers:
                if logger.getEffectiveLevel() < logging.WARNING:
                    logger.setLevel(logging.WARNING)
        _http_log_scopes += 1
    try:
        yield
    finally:
        with _http_log_lock:
            _http_log_scopes -= 1
            # Another overlapping acquisition must retain its private logging scope.
            if _http_log_scopes == 0:
                for logger, original_level in _http_log_levels:
                    logger.setLevel(original_level)
                _http_log_levels = ()

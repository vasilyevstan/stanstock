"""Shared HTTP fetch helper for provider clients.

Every provider client goes through :func:`fetch` so timeouts, redirects, and
the ``User-Agent`` header are applied consistently, and so transport-level
failures are translated into :class:`ProviderNetworkError` instead of an
uncaught ``httpx`` exception.

This module never retries, never solves CAPTCHAs or JavaScript challenges,
and never falls back to scraping HTML. Detecting a bot-verification
challenge is the caller's job (each provider's response shape differs); this
module only performs the request and returns the raw response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from stanstock.data.providers.exceptions import ProviderNetworkError

#: Conservative default: connect quickly, fail fast rather than hang a job.
DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


@dataclass(frozen=True, slots=True)
class HttpFetchResult:
    """The outcome of one HTTP GET, normalized for provider parsing code."""

    status_code: int
    headers: httpx.Headers
    content: bytes
    url: str

    @property
    def content_type(self) -> str:
        return str(self.headers.get("content-type", ""))


def fetch(
    url: str,
    *,
    user_agent: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    follow_redirects: bool = True,
) -> HttpFetchResult:
    """Perform a single HTTP GET with an explicit timeout and User-Agent.

    Raises :class:`ProviderNetworkError` for any transport-level failure
    (timeout, connection reset, DNS failure, TLS error). HTTP-level error
    status codes are returned normally so callers can classify them (a 403
    is meaningfully different from a network outage).
    """
    merged_headers = {"User-Agent": user_agent, **(headers or {})}
    try:
        with httpx.Client(timeout=timeout, follow_redirects=follow_redirects) as client:
            response = client.get(url, params=params, headers=merged_headers)
    except httpx.HTTPError as exc:
        raise ProviderNetworkError(f"{type(exc).__name__} calling {url}: {exc}") from exc
    return HttpFetchResult(
        status_code=response.status_code,
        headers=response.headers,
        content=response.content,
        url=str(response.url),
    )

"""Explicit exception hierarchy for provider clients.

Every provider client raises one of these instead of letting a low-level
network or parsing exception leak, so callers (management commands, jobs)
can distinguish causes without inspecting HTTP internals.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for all provider client failures."""


class ProviderConfigurationError(ProviderError):
    """Required configuration (for example an API key or compliant
    User-Agent) is missing or invalid. This is an environment/setup
    problem, not evidence that the provider itself is unusable."""


class ProviderNetworkError(ProviderError):
    """The HTTP request could not be completed (timeout, DNS, connection
    reset, TLS failure, and similar transport-level failures)."""


class ProviderBlockedError(ProviderError):
    """The provider responded, but with an access denial, a bot/browser
    verification challenge, or another signal that automated access is not
    permitted through this path. StanStock never attempts to solve or bypass
    such a challenge; this exception is the required stop signal instead."""


class ProviderQuotaError(ProviderError):
    """The provider's configured request or credit allowance is exhausted.

    This is distinct from an access-control block: callers may retry after
    the documented quota window resets, but must never hide the failure as a
    successful empty response.
    """


class ProviderResponseError(ProviderError):
    """The provider responded with HTTP 200 (or similar) but the payload was
    empty, malformed, or otherwise not usable (for example an unexpected
    content type, missing required fields, or a checksum mismatch)."""


class ProviderDataError(ProviderResponseError):
    """A valid provider response contained unusable data for one instrument.

    Callers may exclude that instrument when their methodology explicitly
    permits partial coverage. HTTP failures, non-JSON gateway responses, and
    malformed response envelopes remain plain :class:`ProviderResponseError`
    failures and must not be downgraded to a data-quality exclusion.
    """

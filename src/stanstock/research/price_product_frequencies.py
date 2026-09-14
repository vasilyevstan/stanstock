"""Pure fixed-model terminal-frequency classification for price projections.

These are shares of the existing deterministic FHS paths, not estimates of
real-world event probabilities.  The classifier deliberately operates on
terminal *log* returns, so the event boundaries have one stable representation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FREQUENCY_SCHEMA = "research-product-frequencies@1"
FREQUENCY_METHOD_VERSION = "us-price-fhs-frequencies-v1"
_METHOD_DOMAIN = "stanstock-research-product-frequency-method-v1"
LOSS_LOG_THRESHOLD = 0.0
FLAT_UPPER_LOG_THRESHOLD = math.log1p(0.20)
LARGE_LOSS_LOG_THRESHOLD = math.log1p(-0.20)
_BUCKET_ORDER = ("loss", "flat_to_20", "above_20")


@dataclass(frozen=True, slots=True)
class BucketCounts:
    """Three exhaustive counts plus the detail-only subset count."""

    loss: int
    flat_to_20: int
    above_20: int
    large_loss: int

    @property
    def total(self) -> int:
        return self.loss + self.flat_to_20 + self.above_20


@dataclass(frozen=True, slots=True)
class HorizonFrequencies:
    horizon: str
    sessions: int
    path_count: int
    counts: BucketCounts | None
    zero_drift_counts: BucketCounts | None
    insufficiency_reason: str | None

    @property
    def available(self) -> bool:
        return self.counts is not None and self.zero_drift_counts is not None


@dataclass(frozen=True, slots=True)
class DisplayShare:
    event: str
    count: int
    hamilton_percent: int
    label: str


def method_identity_sha256() -> str:
    """Hash the literal method and tie rules rather than a copied claim."""

    payload = {
        "domain": _METHOD_DOMAIN,
        "schema": FREQUENCY_SCHEMA,
        "method_version": FREQUENCY_METHOD_VERSION,
        "basis": "terminal_cumulative_split_adjusted_price_return_excluding_dividends",
        "classification_space": "terminal_log_return",
        "events": {
            "loss": "log_return < 0.0",
            "flat_to_20": f"0.0 <= log_return <= {FLAT_UPPER_LOG_THRESHOLD.hex()}",
            "above_20": f"log_return > {FLAT_UPPER_LOG_THRESHOLD.hex()}",
            "large_loss": f"log_return < {LARGE_LOSS_LOG_THRESHOLD.hex()}",
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def classify_terminal_log_returns(
    terminal_logs: npt.NDArray[np.float64],
    *,
    horizon: str,
    sessions: int,
    path_count: int,
    zero_drift_logs: npt.NDArray[np.float64],
    insufficiency_reason: str | None = None,
) -> HorizonFrequencies:
    """Classify every path or withhold the whole horizon.

    A non-finite value is an invalid projection, not a reason to filter a
    denominator.  The caller supplies the source projection reason for an
    already-withheld forecast.
    """

    if insufficiency_reason:
        return HorizonFrequencies(
            horizon=horizon,
            sessions=sessions,
            path_count=path_count,
            counts=None,
            zero_drift_counts=None,
            insufficiency_reason=insufficiency_reason,
        )
    if (
        path_count <= 0
        or terminal_logs.ndim != 1
        or terminal_logs.shape != zero_drift_logs.shape
        or terminal_logs.size != path_count
        or not np.all(np.isfinite(terminal_logs))
        or not np.all(np.isfinite(zero_drift_logs))
    ):
        return HorizonFrequencies(
            horizon=horizon,
            sessions=sessions,
            path_count=path_count,
            counts=None,
            zero_drift_counts=None,
            insufficiency_reason="frequency_terminal_logs_invalid",
        )
    return HorizonFrequencies(
        horizon=horizon,
        sessions=sessions,
        path_count=path_count,
        counts=_counts(terminal_logs),
        zero_drift_counts=_counts(zero_drift_logs),
        insufficiency_reason=None,
    )


def _counts(values: npt.NDArray[np.float64]) -> BucketCounts:
    loss = values < LOSS_LOG_THRESHOLD
    flat = (values >= LOSS_LOG_THRESHOLD) & (values <= FLAT_UPPER_LOG_THRESHOLD)
    above = values > FLAT_UPPER_LOG_THRESHOLD
    return BucketCounts(
        loss=int(np.count_nonzero(loss)),
        flat_to_20=int(np.count_nonzero(flat)),
        above_20=int(np.count_nonzero(above)),
        large_loss=int(np.count_nonzero(values < LARGE_LOSS_LOG_THRESHOLD)),
    )


def display_shares(counts: BucketCounts, *, path_count: int) -> tuple[DisplayShare, ...]:
    """Return Hamilton shares with honest labels for nonzero rounding tails."""

    if path_count <= 0 or counts.total != path_count or counts.large_loss > counts.loss:
        raise ValueError("Frequency counts do not describe one complete path set")
    raw = {
        "loss": counts.loss * 100 / path_count,
        "flat_to_20": counts.flat_to_20 * 100 / path_count,
        "above_20": counts.above_20 * 100 / path_count,
    }
    integers = {name: int(raw[name]) for name in _BUCKET_ORDER}
    remaining = 100 - sum(integers.values())
    for name in sorted(
        _BUCKET_ORDER,
        key=lambda item: (-(raw[item] - integers[item]), _BUCKET_ORDER.index(item)),
    )[:remaining]:
        integers[name] += 1
    source_counts = {
        "loss": counts.loss,
        "flat_to_20": counts.flat_to_20,
        "above_20": counts.above_20,
    }
    result = []
    for name in _BUCKET_ORDER:
        count = source_counts[name]
        integer = integers[name]
        label = (
            "0%"
            if count == 0
            else "100%"
            if count == path_count
            else "<1%"
            if integer == 0
            else ">99%"
            if integer == 100
            else f"{integer}%"
        )
        result.append(DisplayShare(name, count, integer, label))
    return tuple(result)

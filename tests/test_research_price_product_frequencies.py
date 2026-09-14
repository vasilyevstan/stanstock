from __future__ import annotations

import math

import numpy as np

from stanstock.research.price_product_frequencies import (
    BucketCounts,
    classify_terminal_log_returns,
    display_shares,
)


def test_log_thresholds_are_exhaustive_and_keep_zero_and_twenty_percent_in_middle() -> None:
    values = np.asarray(
        [
            math.log1p(-0.21),
            math.log1p(-0.20),
            -0.0001,
            0.0,
            math.log1p(0.20),
            math.nextafter(math.log1p(0.20), math.inf),
        ],
        dtype=np.float64,
    )
    result = classify_terminal_log_returns(
        values,
        horizon="6m",
        sessions=126,
        path_count=6,
        zero_drift_logs=values,
    )

    assert result.available
    assert result.counts == BucketCounts(loss=3, flat_to_20=2, above_20=1, large_loss=1)
    assert result.counts.total == 6


def test_nonfinite_terminal_withholds_the_complete_horizon_without_changing_denominator() -> None:
    values = np.asarray((0.0, np.nan), dtype=np.float64)
    result = classify_terminal_log_returns(
        values,
        horizon="12m",
        sessions=252,
        path_count=2,
        zero_drift_logs=values,
    )

    assert not result.available
    assert result.path_count == 2
    assert result.insufficiency_reason == "frequency_terminal_logs_invalid"


def test_hamilton_shares_are_exhaustive_but_nonzero_rounding_tails_are_honest() -> None:
    shares = display_shares(BucketCounts(1, 1, 8190, 0), path_count=8192)

    assert tuple(item.hamilton_percent for item in shares) == (0, 0, 100)
    assert tuple(item.label for item in shares) == ("<1%", "<1%", ">99%")
    assert sum(item.hamilton_percent for item in shares) == 100
    assert tuple(
        item.label for item in display_shares(BucketCounts(0, 0, 8192, 0), path_count=8192)
    ) == ("0%", "0%", "100%")

from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from stanstock.research.affordability import (
    PRICE_BAND_POLICY_VERSION,
    UNDER_10_ALLOCATION_REASON,
    UNDER_10_AVAILABLE_FOUNDATIONS,
    UNDER_10_RELEASED_SHADOW_DIAGNOSTICS,
    UNDER_10_SHADOW_DISCLOSURE,
    UNDER_10_UNRELEASED_ACTIVATION_CONTROLS,
    classify_price_band,
    price_band_choices,
)
from stanstock.research.opportunities import load_opportunity_policy


@pytest.mark.parametrize(
    ("close", "expected_slug"),
    [
        (Decimal("9.99"), "under_10"),
        (Decimal("10"), "10_to_50"),
        (Decimal("49.99"), "10_to_50"),
        (Decimal("50"), "50_to_300"),
        (Decimal("299.99"), "50_to_300"),
        (Decimal("300"), "300_plus"),
    ],
)
def test_price_band_boundaries_are_exact(
    close: Decimal,
    expected_slug: str,
) -> None:
    assessment = classify_price_band(
        close=close,
        price_date=date(2026, 9, 4),
        date_basis="market_session",
        currency="USD",
    )

    assert assessment is not None
    assert assessment.slug == expected_slug
    assert assessment.close == close
    assert assessment.price_date == date(2026, 9, 4)
    assert assessment.date_basis == "market_session"


@pytest.mark.parametrize(
    ("close", "currency"),
    [
        (Decimal("0"), "USD"),
        (Decimal("-1"), "USD"),
        (Decimal("NaN"), "USD"),
        (Decimal("9.99"), "EUR"),
    ],
)
def test_price_band_requires_a_valid_positive_usd_close(
    close: Decimal,
    currency: str,
) -> None:
    assert (
        classify_price_band(
            close=close,
            price_date=date(2026, 9, 4),
            date_basis="market_session",
            currency=currency,
        )
        is None
    )


def test_under_10_policy_is_neutral_but_not_newly_investable() -> None:
    assessment = classify_price_band(
        close=Decimal("9.99"),
        price_date=date(2026, 9, 4),
        date_basis="market_session",
        currency="usd",
    )

    assert PRICE_BAND_POLICY_VERSION == "us-price-bands-v1"
    assert assessment is not None
    assert assessment.label == "Under $10 - speculative watchlist"
    assert assessment.new_allocation_eligible is False
    assert assessment.blocks_long_horizon is True
    assert len(UNDER_10_AVAILABLE_FOUNDATIONS) == 3
    assert len(UNDER_10_RELEASED_SHADOW_DIAGNOSTICS) == 2
    assert len(UNDER_10_UNRELEASED_ACTIVATION_CONTROLS) == 1
    assert set(UNDER_10_AVAILABLE_FOUNDATIONS).isdisjoint(UNDER_10_UNRELEASED_ACTIVATION_CONTROLS)
    assert set(UNDER_10_AVAILABLE_FOUNDATIONS).isdisjoint(UNDER_10_RELEASED_SHADOW_DIAGNOSTICS)
    assert set(UNDER_10_RELEASED_SHADOW_DIAGNOSTICS).isdisjoint(
        UNDER_10_UNRELEASED_ACTIVATION_CONTROLS
    )
    assert all(
        "candidate qualification still required" in foundation
        for foundation in UNDER_10_AVAILABLE_FOUNDATIONS
    )
    assert UNDER_10_ALLOCATION_REASON == (
        "New allocation remains 0% pending joint Under-$10 review and "
        "candidate-specific eligibility."
    )
    assert price_band_choices() == [
        ("under_10", "Under $10 - speculative watchlist"),
        ("10_to_50", "$10-$50"),
        ("50_to_300", "$50-$300"),
        ("300_plus", "$300+"),
    ]


def test_released_shadow_diagnostics_are_named_and_unactivated() -> None:
    assert UNDER_10_RELEASED_SHADOW_DIAGNOSTICS == (
        "Shadow solvency/obligation assessment with negative-FCF cash runway",
        "Shadow 252-observed-session median dollar-volume diagnostic",
    )
    assert UNDER_10_UNRELEASED_ACTIVATION_CONTROLS == (
        "Verified split and reverse-split event source",
    )
    assert "not backfilled" in UNDER_10_SHADOW_DISCLOSURE
    assert "an absent assessment means not assessed" in UNDER_10_SHADOW_DISCLOSURE
    assert "not candidate approvals" in UNDER_10_SHADOW_DISCLOSURE
    assert "0% new allocation" in UNDER_10_SHADOW_DISCLOSURE
    assert "joint review remains required" in UNDER_10_SHADOW_DISCLOSURE
    # A released diagnostic is never described as an activation control, and
    # the remaining control is exactly the verified split source.
    assert not any(
        "solvency" in control.lower() or "liquidity" in control.lower()
        for control in UNDER_10_UNRELEASED_ACTIVATION_CONTROLS
    )


def test_opportunity_v2_is_prospective_and_v1_file_remains_immutable() -> None:
    policy = load_opportunity_policy()
    v1_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "opportunities"
        / "great-opportunity-v1.yml"
    )

    assert policy.version == "great-opportunity-v2"
    assert policy.excluded_new_allocation_price_bands == frozenset({"under_10"})
    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == (
        "9f67f95580af76d67dee40564f693ec58f81ba99db9221f93ad1c9b40af19612"
    )

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from stanstock.data.models import LatestMarketData, Listing

PRICE_BAND_POLICY_VERSION = "us-price-bands-v1"
PRICE_BAND_CURRENCY = "USD"
UNDER_10_BAND = "under_10"
MARKET_SESSION_DATE_BASIS = "market_session"
DECISION_TARGET_DATE_BASIS = "decision_target"
PRICE_DATE_BASES = frozenset(
    {
        MARKET_SESSION_DATE_BASIS,
        DECISION_TARGET_DATE_BASIS,
    }
)

UNDER_10_ALLOCATION_REASON = (
    "New allocation remains 0% pending joint Under-$10 review and candidate-specific eligibility."
)
PRICE_BAND_UNAVAILABLE_ALLOCATION_REASON = (
    "New allocation is withheld because no valid latest persisted USD close "
    "is available to apply the guarded price-band policy."
)
UNDER_10_AVAILABLE_FOUNDATIONS = (
    "Point-in-time SEC facts with adverse-versus-missing branch behavior "
    "(released foundation; candidate qualification still required)",
    "Long-v2 diluted-share/per-share continuity assessment with withholding, "
    "not post-period event verification (released foundation; candidate "
    "qualification still required)",
    "Deterministic 3-year/5-year formula engine with missing-input withholding "
    "(released foundation; candidate qualification still required)",
)

#: Released, but deliberately unactivated: these produce a recorded
#: diagnostic on newly created qualifying analyses and nothing else. They are
#: not candidate approvals and cannot pass an activation gate.
UNDER_10_RELEASED_SHADOW_DIAGNOSTICS = (
    "Shadow solvency/obligation assessment with negative-FCF cash runway",
    "Shadow 252-observed-session median dollar-volume diagnostic",
)

UNDER_10_UNRELEASED_ACTIVATION_CONTROLS = ("Verified split and reverse-split event source",)

#: Stated alongside the three capability groups so a released capability is
#: never read as historical coverage or as an activation approval.
UNDER_10_SHADOW_DISCLOSURE = (
    "These capabilities are recorded only on newly created qualifying "
    "analyses. Existing analyses were not backfilled; an absent assessment "
    "means not assessed. Diagnostics are not candidate approvals. Under-$10 "
    "policy remains 0% new allocation; verified split/reverse-split evidence "
    "is unavailable, and joint review remains required."
)


@dataclass(frozen=True, slots=True)
class PriceBandDefinition:
    slug: str
    label: str
    minimum: Decimal
    maximum: Decimal | None
    minimum_inclusive: bool
    new_allocation_eligible: bool = True
    description: str = ""

    def contains(self, price: Decimal) -> bool:
        above_minimum = price >= self.minimum if self.minimum_inclusive else price > self.minimum
        below_maximum = self.maximum is None or price < self.maximum
        return above_minimum and below_maximum

    @property
    def blocks_long_horizon(self) -> bool:
        return self.slug == UNDER_10_BAND


PRICE_BANDS = (
    PriceBandDefinition(
        slug=UNDER_10_BAND,
        label="Under $10 - speculative watchlist",
        minimum=Decimal("0"),
        maximum=Decimal("10"),
        minimum_inclusive=False,
        new_allocation_eligible=False,
        description="Research only; nominal price is not evidence of value.",
    ),
    PriceBandDefinition(
        slug="10_to_50",
        label="$10-$50",
        minimum=Decimal("10"),
        maximum=Decimal("50"),
        minimum_inclusive=True,
        description="Neutral affordability band.",
    ),
    PriceBandDefinition(
        slug="50_to_300",
        label="$50-$300",
        minimum=Decimal("50"),
        maximum=Decimal("300"),
        minimum_inclusive=True,
        description="Neutral affordability band.",
    ),
    PriceBandDefinition(
        slug="300_plus",
        label="$300+",
        minimum=Decimal("300"),
        maximum=None,
        minimum_inclusive=True,
        description="Execution and concentration context only; not a quality penalty.",
    ),
)
PRICE_BANDS_BY_SLUG = {definition.slug: definition for definition in PRICE_BANDS}


@dataclass(frozen=True, slots=True)
class PriceBandAssessment:
    definition: PriceBandDefinition
    close: Decimal
    price_date: date
    date_basis: str
    currency: str = PRICE_BAND_CURRENCY

    @property
    def slug(self) -> str:
        return self.definition.slug

    @property
    def label(self) -> str:
        return self.definition.label

    @property
    def new_allocation_eligible(self) -> bool:
        return self.definition.new_allocation_eligible

    @property
    def blocks_long_horizon(self) -> bool:
        return self.definition.blocks_long_horizon


def price_band_choices() -> list[tuple[str, str]]:
    return [(definition.slug, definition.label) for definition in PRICE_BANDS]


def classify_price_band(
    *,
    close: Decimal,
    price_date: date,
    date_basis: str,
    currency: str,
) -> PriceBandAssessment | None:
    if date_basis not in PRICE_DATE_BASES:
        raise ValueError(f"Unsupported price-band date basis: {date_basis}")
    if currency.upper() != PRICE_BAND_CURRENCY or not close.is_finite() or close <= 0:
        return None
    for definition in PRICE_BANDS:
        if definition.contains(close):
            return PriceBandAssessment(
                definition=definition,
                close=close,
                price_date=price_date,
                date_basis=date_basis,
            )
    return None


def latest_price_band(listing: Listing) -> PriceBandAssessment | None:
    try:
        market_data = listing.latest_market_data
    except LatestMarketData.DoesNotExist:
        return None
    return classify_price_band(
        close=market_data.close,
        price_date=market_data.session_date,
        date_basis=MARKET_SESSION_DATE_BASIS,
        currency=listing.currency,
    )

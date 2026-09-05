from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from stanstock.simulation.types import (
    MissingPriceError,
    MissingPricePolicy,
    UnresolvedObservation,
)


@dataclass
class CorporateEventResolution:
    action: str  # "retain_last_price", "liquidate_zero", "settle_cash", "raise_error"
    settlement_price: float | None = None
    observation: UnresolvedObservation | None = None
    details: str = ""


class CorporateEventHook(Protocol):
    def __call__(
        self,
        *,
        listing_id: str,
        symbol: str,
        current_date: date,
        last_known_date: date,
        last_known_price: float,
        quantity_held: float,
        event_type: str,
        policy: MissingPricePolicy,
    ) -> CorporateEventResolution: ...


def default_corporate_event_hook(
    *,
    listing_id: str,
    symbol: str,
    current_date: date,
    last_known_date: date,
    last_known_price: float,
    quantity_held: float,
    event_type: str,
    policy: MissingPricePolicy,
) -> CorporateEventResolution:
    """Standard hook handling missing prices and corporate events without silent dropping."""
    if policy == MissingPricePolicy.FAIL:
        raise MissingPriceError(
            f"Missing price for listing {listing_id} ({symbol}) on {current_date}. "
            f"Last known price: {last_known_price} on {last_known_date}."
        )

    if policy == MissingPricePolicy.DROP:
        obs = UnresolvedObservation(
            listing_id=listing_id,
            symbol=symbol,
            date=current_date,
            event_type=event_type,
            last_known_price=last_known_price,
            last_known_date=last_known_date,
            quantity_held=quantity_held,
            action_taken="liquidated_zero",
            details=f"Dropped and valued at zero per {policy.value} policy.",
        )
        return CorporateEventResolution(
            action="liquidate_zero",
            settlement_price=0.0,
            observation=obs,
            details=obs.details,
        )

    # For MARK_UNRESOLVED and CARRY_FORWARD:
    obs = UnresolvedObservation(
        listing_id=listing_id,
        symbol=symbol,
        date=current_date,
        event_type=event_type,
        last_known_price=last_known_price,
        last_known_date=last_known_date,
        quantity_held=quantity_held,
        action_taken="retained_at_last_known",
        details=(
            f"Price missing on {current_date}; retained last known price {last_known_price} "
            f"from {last_known_date}."
        ),
    )
    return CorporateEventResolution(
        action="retain_last_price",
        settlement_price=last_known_price,
        observation=obs,
        details=obs.details,
    )


class CashSettlementHook:
    """Hook for explicit corporate actions such as all-cash acquisitions or tender offers."""

    def __init__(
        self,
        settlements: dict[tuple[str, date], float],
        fallback_hook: CorporateEventHook = default_corporate_event_hook,
    ) -> None:
        self.settlements = settlements
        self.fallback_hook = fallback_hook

    def __call__(
        self,
        *,
        listing_id: str,
        symbol: str,
        current_date: date,
        last_known_date: date,
        last_known_price: float,
        quantity_held: float,
        event_type: str,
        policy: MissingPricePolicy,
    ) -> CorporateEventResolution:
        key = (listing_id, current_date)
        sym_key = (symbol, current_date)
        if key in self.settlements:
            settle_price = self.settlements[key]
        elif sym_key in self.settlements:
            settle_price = self.settlements[sym_key]
        else:
            settle_price = None

        if settle_price is not None:
            obs = UnresolvedObservation(
                listing_id=listing_id,
                symbol=symbol,
                date=current_date,
                event_type="cash_settlement",
                last_known_price=last_known_price,
                last_known_date=last_known_date,
                quantity_held=quantity_held,
                action_taken="settle_cash",
                details=f"Cash settlement at {settle_price} per share on {current_date}.",
            )
            return CorporateEventResolution(
                action="settle_cash",
                settlement_price=settle_price,
                observation=obs,
                details=obs.details,
            )

        return self.fallback_hook(
            listing_id=listing_id,
            symbol=symbol,
            current_date=current_date,
            last_known_date=last_known_date,
            last_known_price=last_known_price,
            quantity_held=quantity_held,
            event_type=event_type,
            policy=policy,
        )

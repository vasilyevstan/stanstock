from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from django import template

register = template.Library()

DISPLAY_LABELS = {
    "12m": "12 months",
    "3y": "3 years",
    "5y": "5 years",
    "6m": "6 months",
    "daily": "Daily market refresh",
    "europe": "Europe",
    "empirical_calibrated": "Probability gate passed — not calibrated",
    "empirical_range_only": "Analog range only — probability withheld",
    "filings_xbrl_org": "filings.xbrl.org",
    "heuristic": "Heuristic",
    "long": "3+ years (legacy)",
    "medium": "6-12 months (legacy)",
    "price_history": "Price history",
    "price_only_baseline": "Price-only baseline",
    "sec": "SEC EDGAR",
    "short": "1-10 trading days",
    "snapshot_portfolios": "Portfolio snapshots",
    "split_adjusted_price_return": "Split-adjusted price return",
    "stock_catalog": "Stock catalog",
    "synthetic_demo": "Synthetic demo",
    "twelve_data": "Twelve Data",
    "us": "US",
}


@register.filter
def percentage(value: object, digits: int = 1) -> str:
    if value is None or value == "":
        return "Unavailable"
    try:
        number = Decimal(str(value)) * Decimal(100)
    except (InvalidOperation, ValueError):
        return "Unavailable"
    return f"{number:+.{digits}f}%"


@register.filter
def price(value: object) -> str:
    if value is None or value == "":
        return "Unavailable"
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return "Unavailable"
    if not number.is_finite():
        return "Unavailable"

    digits = 2 if abs(number) >= 1 else 4
    rendered = f"{number:,.{digits}f}"
    if digits > 2:
        whole, fraction = rendered.split(".", maxsplit=1)
        fraction = fraction.rstrip("0")
        fraction = fraction.ljust(2, "0")
        rendered = f"{whole}.{fraction}"
    return rendered


@register.filter
def display_label(value: object) -> str:
    if value is None or value == "":
        return "Unavailable"
    text = str(value).strip()
    if not text:
        return "Unavailable"
    return DISPLAY_LABELS.get(text.lower(), text.replace("_", " ").replace("-", " ").title())


@register.filter
def label_list(value: object, separator: str = ", ") -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, str):
        return display_label(value)
    if not isinstance(value, Iterable) or isinstance(value, Mapping):
        return display_label(value)
    labels = [display_label(item) for item in value]
    return separator.join(labels) if labels else "Unavailable"


@register.filter
def scenario_range(value: object) -> str:
    if not isinstance(value, Mapping):
        return "Insufficient evidence"
    bear = _first(value, "bear", "bear_return")
    base = _first(value, "base", "base_return")
    bull = _first(value, "bull", "bull_return")
    if bear is None or base is None or bull is None:
        return "Insufficient evidence"
    return f"{percentage(bear)} / {percentage(base)} / {percentage(bull)}"


@register.filter
def mapping_items(value: object) -> list[tuple[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    return [(str(key), item) for key, item in value.items()]


def _first(value: Mapping[object, object], *keys: str) -> object | None:
    for key in keys:
        if key in value:
            return value[key]
    return None

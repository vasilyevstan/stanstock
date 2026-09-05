from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from django import template

register = template.Library()


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

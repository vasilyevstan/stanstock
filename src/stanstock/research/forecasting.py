from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

FORECAST_SCENARIO_SCHEMA_VERSION = 1
CANONICAL_FORECAST_HORIZONS = ("short", "6m", "12m", "3y", "5y")
LEGACY_FORECAST_HORIZONS = ("medium", "long")
FORECAST_HORIZONS = (*CANONICAL_FORECAST_HORIZONS, *LEGACY_FORECAST_HORIZONS)


def build_forecast_scenario_document(
    scenarios: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": FORECAST_SCENARIO_SCHEMA_VERSION,
        "horizons": {
            horizon: deepcopy(dict(scenarios[horizon]))
            for horizon in FORECAST_HORIZONS
            if horizon in scenarios
        },
    }


def scenario_from_document(
    document: object,
    horizon: str,
    *,
    legacy_fallbacks: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if (
        isinstance(document, dict)
        and document.get("schema_version") == FORECAST_SCENARIO_SCHEMA_VERSION
    ):
        horizons = document.get("horizons")
        if isinstance(horizons, dict):
            scenario = horizons.get(horizon)
            if isinstance(scenario, dict):
                return scenario
    if legacy_fallbacks is not None:
        fallback = legacy_fallbacks.get(horizon)
        if isinstance(fallback, dict):
            return fallback
    return {}


def infer_price_source(
    source_assets: Iterable[Mapping[str, object]],
    *,
    subjects: Iterable[str],
) -> tuple[str, str]:
    expected_subjects = {subject for subject in subjects if subject}
    sources = {
        (str(asset["provider"]), str(asset["subject"]))
        for asset in source_assets
        if asset.get("kind") == "price_history"
        and str(asset.get("subject") or "") in expected_subjects
        and isinstance(asset.get("provider"), str)
        and str(asset["provider"])
    }
    if len(sources) != 1:
        return "", ""
    return next(iter(sources))

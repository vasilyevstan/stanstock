from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from stanstock.research.models import AnalysisRun

DATA_MODE_PROVIDER = "provider"
DATA_MODE_SYNTHETIC = "synthetic"
DATA_MODE_UNKNOWN = "unknown"


def source_assets(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, dict):
        return ()
    raw_assets = value.get("source_assets")
    if not isinstance(raw_assets, list):
        return ()
    return tuple(asset for asset in raw_assets if isinstance(asset, dict))


def source_providers(value: object) -> frozenset[str]:
    return frozenset(
        str(provider) for asset in source_assets(value) if (provider := asset.get("provider"))
    )


def source_data_mode(value: object) -> str:
    providers = source_providers(value)
    if not providers:
        return DATA_MODE_UNKNOWN
    if any(provider.startswith("synthetic") for provider in providers):
        return DATA_MODE_SYNTHETIC
    return DATA_MODE_PROVIDER


def analysis_run_data_mode(run: AnalysisRun | None) -> str:
    if run is None:
        return DATA_MODE_UNKNOWN
    analysis = run.stocks.order_by("pk").first()
    return source_data_mode(analysis.data_quality) if analysis is not None else DATA_MODE_UNKNOWN


def analysis_run_source_providers(run: AnalysisRun | None) -> frozenset[str]:
    if run is None:
        return frozenset()
    analysis = run.stocks.order_by("pk").first()
    return source_providers(analysis.data_quality) if analysis is not None else frozenset()


def latest_serving_analysis_run() -> AnalysisRun | None:
    return (
        AnalysisRun.objects.filter(status="complete")
        .select_related("universe_snapshot__universe")
        .order_by("-generated_at")
        .first()
    )


def latest_provider_backed_analysis_run() -> AnalysisRun | None:
    runs = AnalysisRun.objects.filter(status="complete").order_by(
        "-target_date",
        "-generated_at",
    )[:100]
    for run in runs:
        if analysis_run_data_mode(run) == DATA_MODE_PROVIDER:
            return run
    return None


def data_mode_label(mode: str, providers: Iterable[str] = ()) -> str:
    normalized = sorted(set(providers))
    if mode == DATA_MODE_SYNTHETIC:
        return "Synthetic research data"
    if mode != DATA_MODE_PROVIDER:
        return "No serving analysis"
    if normalized == ["twelve_data"]:
        return "Twelve Data provider-backed"
    return "Provider-backed market data"

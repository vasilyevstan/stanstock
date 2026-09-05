from __future__ import annotations

from django.conf import settings
from django.http import HttpRequest

from stanstock.research.models import AnalysisRun, StockAnalysis


def stanstock_runtime(request: HttpRequest) -> dict[str, object]:
    synthetic_data = False
    if request.user.is_authenticated:
        latest_run_id = (
            AnalysisRun.objects.filter(status="complete")
            .order_by("-generated_at")
            .values_list("pk", flat=True)
            .first()
        )
        if latest_run_id is not None:
            qualities = StockAnalysis.objects.filter(run_id=latest_run_id).values_list(
                "data_quality",
                flat=True,
            )
            synthetic_data = any(_uses_synthetic_source(value) for value in qualities)
    return {
        "stanstock_demo_mode": settings.DEMO_MODE,
        "stanstock_synthetic_data": synthetic_data,
    }


def _uses_synthetic_source(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    assets = value.get("source_assets")
    if not isinstance(assets, list):
        return False
    return any(
        isinstance(asset, dict) and asset.get("provider") == "synthetic_demo" for asset in assets
    )

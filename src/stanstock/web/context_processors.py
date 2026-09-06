from __future__ import annotations

from django.conf import settings
from django.http import HttpRequest

from stanstock.research.provenance import (
    DATA_MODE_SYNTHETIC,
    analysis_run_data_mode,
    latest_serving_analysis_run,
)


def stanstock_runtime(request: HttpRequest) -> dict[str, object]:
    mode = "unknown"
    if request.user.is_authenticated:
        mode = analysis_run_data_mode(latest_serving_analysis_run())
    demo_fallback = mode == "unknown" and settings.DEMO_MODE
    synthetic_data = mode == DATA_MODE_SYNTHETIC
    return {
        "stanstock_demo_mode": demo_fallback,
        "stanstock_synthetic_data": synthetic_data,
        "stanstock_show_synthetic_banner": demo_fallback or synthetic_data,
    }

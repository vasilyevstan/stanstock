from __future__ import annotations

from django.conf import settings
from django.http import HttpRequest

from stanstock.research.provenance import (
    DATA_MODE_SYNTHETIC,
    analysis_run_data_mode,
    latest_serving_analysis_run,
)


def stanstock_runtime(request: HttpRequest) -> dict[str, object]:
    product_read = getattr(request, "_stanstock_product_read", None)
    if settings.RESEARCH_PRODUCT_ENABLED and product_read is not None:
        synthetic_data = (
            product_read.available
            and product_read.provider == "synthetic_demo"
            and product_read.owner_id == "synthetic-demo"
        )
        return {
            "stanstock_demo_mode": settings.DEMO_MODE and not product_read.available,
            "stanstock_synthetic_data": synthetic_data,
            "stanstock_show_synthetic_banner": settings.DEMO_MODE or synthetic_data,
            "stanstock_research_product_enabled": True,
        }
    mode = "unknown"
    if request.user.is_authenticated:
        mode = analysis_run_data_mode(latest_serving_analysis_run())
    demo_fallback = mode == "unknown" and settings.DEMO_MODE
    synthetic_data = mode == DATA_MODE_SYNTHETIC
    return {
        "stanstock_demo_mode": demo_fallback,
        "stanstock_synthetic_data": synthetic_data,
        "stanstock_show_synthetic_banner": demo_fallback or synthetic_data,
        "stanstock_research_product_enabled": settings.RESEARCH_PRODUCT_ENABLED,
    }

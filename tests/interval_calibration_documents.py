"""Shared synthetic serialized-study fixtures for interval calibration tests."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid5

from stanstock.research.price_product_config import FHS_METHOD_VERSION, load_price_product_config
from stanstock.research.price_product_replay import build_anchor_plan
from stanstock.research.price_product_study import _protocol_sessions
from stanstock.research.product_study_evidence import (
    _calendar_identity,
    _config_identity,
    _protocol_identity,
)
from test_research_price_product_calibration import (
    _aggregates,
    _anchor,
    _document,
    _listing,
    _model,
    _scored,
)

_COMPARATORS = ("zero_log_drift_gaussian", "historical_log_drift_gaussian")
_NAMESPACE = UUID("f0b52da9-f72c-4e16-99f5-b34ee0181a43")


def build_interval_calibration_document(
    *,
    listings_per_cluster: int = 30,
    development_6m_clusters: int = 7,
    development_12m_clusters: int = 3,
    include_validation: bool = True,
    include_holdout: bool = True,
    actual_return: str = "0.0300",
    lower: str = "-0.0100",
    median: str = "0.0000",
    upper: str = "0.0100",
    validation_actual_returns: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build a full-grid synthetic study anchored to the frozen calendar plan."""
    config = load_price_product_config()
    sessions = _protocol_sessions(config)
    plan = build_anchor_plan(sessions, config=config)
    selected: list[tuple[str, str, str, str]] = []
    for partition_plan in plan.partitions:
        candidates = list(partition_plan.anchors)
        if partition_plan.partition == "development" and partition_plan.horizon == "6m":
            candidates = candidates[:development_6m_clusters]
        elif partition_plan.partition == "development" and partition_plan.horizon == "12m":
            candidates = candidates[:development_12m_clusters]
        elif partition_plan.partition == "validation":
            if not include_validation or partition_plan.horizon != "6m":
                candidates = []
            else:
                candidates = candidates[:1]
        elif partition_plan.partition == "final_holdout":
            candidates = candidates[:1] if include_holdout else []
        else:
            candidates = []
        selected.extend(
            (
                partition_plan.partition,
                partition_plan.horizon,
                item.anchor_date.isoformat(),
                item.outcome_end_date.isoformat(),
            )
            for item in candidates
        )

    max_cluster = listings_per_cluster
    records: list[list[dict[str, Any]]] = [[] for _ in range(max_cluster)]
    for partition, horizon, anchor_date, outcome_end in selected:
        for listing_index, anchors in enumerate(records):
            source_actual = (
                validation_actual_returns[listing_index % len(validation_actual_returns)]
                if partition == "validation" and validation_actual_returns
                else actual_return
            )
            source_lower, source_median, source_upper = lower, median, upper
            metrics = _scored(source_actual, source_lower, source_median, source_upper)
            models = [
                _model(FHS_METHOD_VERSION, metrics),
                *(
                    _model(
                        comparator,
                        _scored(source_actual, "-0.0500", "0.0000", "0.0500"),
                    )
                    for comparator in _COMPARATORS
                ),
            ]
            anchors.append(_anchor(partition, horizon, anchor_date, outcome_end, models))
    listings = [
        _listing(str(uuid5(_NAMESPACE, f"synthetic-listing-{index}")), anchors)
        for index, anchors in enumerate(records)
    ]
    document = _document(listings, aggregates=_aggregates(listings))
    document["config_identity"] = _config_identity(config)
    document["protocol_identity"] = _protocol_identity(config)
    document["calendar_identity"] = _calendar_identity(config)
    document["scope"] = {
        "all_selected": True,
        "studied_listing_count": len(listings),
        "source_run_listing_count": len(listings),
    }
    return document

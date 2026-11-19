"""Validated study-document adapter for the pure interval-calibration core."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from stanstock.research.price_product import LedgerTriplet
from stanstock.research.price_product_calibration import (
    _parse_date,
    _parse_decimal,
    _require,
    _require_list,
    _require_mapping,
    _require_text,
    summarize_quantile_calibration,
)
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    load_price_product_config,
)
from stanstock.research.price_product_interval_calibration import (
    FIT_SCHEMA,
    CalibrationInputs,
    CalibrationObservation,
    canonical_json_bytes,
    fit_interval_calibration,
)
from stanstock.research.price_product_study import _protocol_sessions
from stanstock.research.product_study_evidence import (
    _calendar_identity,
    _config_identity,
    _protocol_identity,
)


def calibration_inputs_from_document(document: Mapping[str, Any]) -> CalibrationInputs:
    """Run every frozen S1 and identity check, then extract FHS development/validation rows."""
    summarize_quantile_calibration(document)
    return _extract_calibration_inputs(document)


def _extract_calibration_inputs(document: Mapping[str, Any]) -> CalibrationInputs:
    config = load_price_product_config()
    expected_identities = {
        "config_identity": _config_identity(config),
        "protocol_identity": _protocol_identity(config),
        "calendar_identity": _calendar_identity(config),
    }
    for key, expected in expected_identities.items():
        try:
            matches = canonical_json_bytes(document.get(key)) == canonical_json_bytes(expected)
        except ValueError:
            matches = False
        if not matches:
            raise ValueError(f"{key} does not match the frozen product identity")

    sessions = _protocol_sessions(config)
    rows: list[CalibrationObservation] = []
    for listing_index, raw_listing in enumerate(
        _require_list(_require(document, "listings", "document"), "listings")
    ):
        listing_where = f"listings[{listing_index}]"
        listing = _require_mapping(raw_listing, listing_where)
        listing_id = _require_text(
            _require(listing, "listing_id", listing_where), f"{listing_where}.listing_id"
        )
        for anchor_index, raw_anchor in enumerate(
            _require_list(_require(listing, "anchors", listing_where), f"{listing_where}.anchors")
        ):
            anchor_where = f"{listing_where}.anchors[{anchor_index}]"
            anchor = _require_mapping(raw_anchor, anchor_where)
            partition = _require_text(
                _require(anchor, "partition", anchor_where), f"{anchor_where}.partition"
            )
            horizon = _require_text(
                _require(anchor, "horizon", anchor_where), f"{anchor_where}.horizon"
            )
            anchor_date = _parse_date(
                _require(anchor, "anchor_date", anchor_where), f"{anchor_where}.anchor_date"
            )
            outcome_end = _parse_date(
                _require(anchor, "outcome_end_date", anchor_where),
                f"{anchor_where}.outcome_end_date",
            )
            for model_index, raw_model in enumerate(
                _require_list(_require(anchor, "models", anchor_where), f"{anchor_where}.models")
            ):
                model_where = f"{anchor_where}.models[{model_index}]"
                model = _require_mapping(raw_model, model_where)
                model_name = _require_text(
                    _require(model, "model_name", model_where), f"{model_where}.model_name"
                )
                if model_name != FHS_METHOD_VERSION:
                    continue
                if partition == "final_holdout":
                    continue
                if partition not in ("development", "validation"):
                    raise ValueError("study contains an unsupported anchor partition")
                metrics = _require_mapping(
                    _require(model, "metrics", model_where), f"{model_where}.metrics"
                )
                reason_raw = _require(metrics, "unavailable_reason", f"{model_where}.metrics")
                if reason_raw is None:
                    actual = _parse_decimal(
                        _require(metrics, "actual_return", f"{model_where}.metrics"),
                        f"{model_where}.metrics.actual_return",
                    )
                    predicted_mapping = _require_mapping(
                        _require(metrics, "predicted_returns", f"{model_where}.metrics"),
                        f"{model_where}.metrics.predicted_returns",
                    )
                    predicted = LedgerTriplet(
                        _parse_decimal(
                            _require(
                                predicted_mapping,
                                "lower",
                                f"{model_where}.metrics.predicted_returns",
                            ),
                            f"{model_where}.metrics.predicted_returns.lower",
                        ),
                        _parse_decimal(
                            _require(
                                predicted_mapping,
                                "median",
                                f"{model_where}.metrics.predicted_returns",
                            ),
                            f"{model_where}.metrics.predicted_returns.median",
                        ),
                        _parse_decimal(
                            _require(
                                predicted_mapping,
                                "upper",
                                f"{model_where}.metrics.predicted_returns",
                            ),
                            f"{model_where}.metrics.predicted_returns.upper",
                        ),
                    )
                    unavailable_reason = None
                else:
                    unavailable_reason = _require_text(
                        reason_raw, f"{model_where}.metrics.unavailable_reason"
                    )
                    actual = None
                    predicted = None
                rows.append(
                    CalibrationObservation(
                        partition=partition,
                        horizon=horizon,
                        listing_id=listing_id,
                        anchor_date=anchor_date,
                        outcome_end_date=outcome_end,
                        predicted=predicted,
                        actual_return=actual,
                        unavailable_reason=unavailable_reason,
                    )
                )
    return CalibrationInputs(
        config=config,
        config_identity=expected_identities["config_identity"],
        protocol_identity=expected_identities["protocol_identity"],
        calendar_identity=expected_identities["calendar_identity"],
        calendar_sessions=sessions,
        observations=tuple(rows),
    )


def fit_interval_calibration_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Fit the pure core and attach order-sensitive, caller-supplied provenance."""
    ledger = summarize_quantile_calibration(document)
    inputs = _extract_calibration_inputs(document)
    fit = fit_interval_calibration(inputs)
    try:
        calendar_version = version("exchange-calendars")
    except PackageNotFoundError:
        calendar_version = "unavailable"
    source = {
        "s1_ledger_input_sha256": ledger["input_sha256"],
        "source_run_id": ledger["source_run_id"],
        "source_evidence_label": ledger["source_evidence_label"],
        "source_execution_modes": ledger["source_execution_modes"],
        "source_snapshot_grade": ledger["source_snapshot_grade"],
        "source_issued_on_time": ledger["source_issued_on_time"],
        "scope": ledger["scope"],
        "source_disclosures": ledger["source_disclosures"],
        "evidence_label": ledger["evidence_label"],
        "attestation": "caller_supplied_file_unverified",
        "final_holdout_observations_excluded": sum(
            anchor["partition"] == "final_holdout" and row["model_name"] == FHS_METHOD_VERSION
            for listing in document.get("listings", [])
            if isinstance(listing, Mapping)
            for anchor in listing.get("anchors", [])
            if isinstance(anchor, Mapping)
            for row in anchor.get("models", [])
            if isinstance(row, Mapping)
        ),
        "calendar_dependency": {
            "name": "exchange-calendars",
            "version": calendar_version,
        },
    }
    return {
        "schema": FIT_SCHEMA,
        "params": fit["params"],
        "diagnostics": fit["diagnostics"],
        "source": source,
    }


def render_interval_calibration(fit_document: Mapping[str, Any], *, output_format: str) -> str:
    if output_format not in ("json", "text"):
        raise ValueError("output_format must be json or text")
    if fit_document.get("schema") != FIT_SCHEMA:
        raise ValueError("fit document schema is invalid")
    if output_format == "json":
        return json.dumps(fit_document, sort_keys=True, indent=2)
    params = fit_document.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("fit document parameters are malformed")
    horizons = params.get("horizons")
    if not isinstance(horizons, Mapping):
        raise ValueError("fit document horizons are malformed")
    lines = [
        "research-product interval calibration (research-only)",
        f"method={params['method']['method_version']}",
        "Not applied to native predictions; marginal guarantee is approximate; "
        "no conditional coverage claim; 3y/5y make no calibration claim.",
        (
            "S1 input hash is order-sensitive provenance, not identity: "
            f"{fit_document['source']['s1_ledger_input_sha256']}"
        ),
    ]
    for horizon in ("6m", "12m", "3y", "5y"):
        result = horizons[horizon]
        lines.append(
            f"{horizon}: state={result['state']} reason={result['reason']} "
            f"offset={result['offset']} clusters={result['clusters']}"
        )
    return "\n".join(lines)

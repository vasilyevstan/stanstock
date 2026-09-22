"""Pure, permanently unscored preparation for a fixed half-log-drift sensitivity.

Caller-supplied identities, checksums and calendars are unverified provenance,
not source authentication or observed issuance. The empirical protocol is not
frozen. No outcomes, calibration, skill, eligibility or adoption are established.
The declared future interval-score/median-MAE comparison is not calculated here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from datetime import date, datetime
from decimal import Decimal
from types import UnionType
from typing import Literal, get_args, get_origin, get_type_hints
from uuid import UUID

import numpy as np

from stanstock.research.price_product import (
    HorizonProjection,
    LedgerTriplet,
    PriceInputIdentity,
    PriceProductInput,
    RawTriplet,
    SimulationTerminals,
    SourceExecutionBinding,
    calculate_price_product,
    filter_historical_returns,
    projection_from_terminal_logs,
    simulate_fhs_terminal_logs,
)
from stanstock.research.price_product_config import (
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PriceProductConfig,
    price_product_config_hash,
)

ArmId = Literal["historical_fhs_control", "zero_log_drift_fhs_control", "fixed_half_drift_fhs"]
_ARMS: tuple[ArmId, ...] = (
    "historical_fhs_control",
    "zero_log_drift_fhs_control",
    "fixed_half_drift_fhs",
)
_HORIZONS = (("6m", 126), ("12m", 252), ("3y", 756), ("5y", 1260))
_ROLES = (
    ("6m", "primary"),
    ("12m", "secondary_diagnostic"),
    ("3y", "evaluation_unavailable"),
    ("5y", "evaluation_unavailable"),
)
_ASSUMPTIONS_DOCUMENT = (
    ("arm_ids", _ARMS),
    ("candidate_failure", "native_projection_reason_withholds_entire_candidate_horizon_only"),
    ("empirical_protocol_status", "not_frozen"),
    ("evidence_label", "research_only_unscored"),
    ("half_operation", "terminals.zero_drift[i] + H * (mu / 2.0)"),
    ("horizon_roles", _ROLES),
    ("inherited_withholding", "native_selected_horizon_reason_all_arms_no_rescue"),
    ("innovation_seed", "native_stanstock-research-product-seed-v1_first_128_sha256_bits"),
    ("native_config_hash", "55334183af29fc01b853e83f9bf75216f24564956925fb95912cb80a69420867"),
    ("native_method_version", "us-price-fhs-v1"),
    ("path_count", 8192),
    ("primary_guardrail_declared_only", "6m_median_mean_absolute_error"),
    ("primary_metric_declared_only", "6m_central_60_percent_interval_score"),
    ("projection", "native_linear_p20_p50_p80_with_native_zero_companion"),
    ("research_version", "shadow-fhs-drift-v1"),
    ("rounding", "native_independent_raw_return_4_price_6_decimal_half_even"),
    ("schema_version", "shadow-fhs-drift@1"),
    ("simulation", "native_PCG64_path_major_filter_coefficients_historical_mean_variance"),
    ("simulation_horizons", _HORIZONS),
    ("source_provenance_status", "caller_supplied_unverified"),
)
_ASSUMPTIONS_SHA256 = hashlib.sha256(
    json.dumps(
        dict(_ASSUMPTIONS_DOCUMENT),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
).hexdigest()
_DEPENDENCY_ERROR = "Shadow drift preparation native dependency mismatch"
_SCHEMA_ERROR = "Malformed shadow drift preparation schema"


@dataclass(frozen=True, slots=True)
class ShadowProjection:
    arm_id: ArmId
    horizon: Literal["6m", "12m"]
    sessions: Literal[126, 252]
    quantile_levels: tuple[float, float, float]
    central_model_mass: float
    raw_returns: RawTriplet | None
    ledger_returns: LedgerTriplet | None
    raw_prices: RawTriplet | None
    ledger_prices: LedgerTriplet | None
    insufficiency_reason: str | None


@dataclass(frozen=True, slots=True)
class ShadowDriftResult:
    schema_version: Literal["shadow-fhs-drift@1"]
    research_version: Literal["shadow-fhs-drift-v1"]
    assumptions_sha256: str
    native_method_version: Literal["us-price-fhs-v1"]
    native_config_hash: str
    input_hash: str
    listing_id: UUID
    target_date: date
    decision_time: datetime
    stock_identity: PriceInputIdentity
    benchmark_identity: PriceInputIdentity
    source_execution: SourceExecutionBinding
    evidence_label: Literal["research_only_unscored"]
    source_provenance_status: Literal["caller_supplied_unverified"]
    empirical_protocol_status: Literal["not_frozen"]
    innovation_seed: int
    path_count: Literal[8192]
    simulation_horizons: tuple[tuple[str, int], ...]
    horizon_roles: tuple[tuple[str, str], ...]
    mean_log_return: float | None
    half_log_drift: float | None
    native_forecast_insufficiency_reason: str | None
    projections: tuple[ShadowProjection, ...]


def project_shadow_drift(
    product_input: PriceProductInput, *, config: PriceProductConfig
) -> ShadowDriftResult:
    """Prepare three unscored arms; source admission remains entirely native."""
    if (
        type(config) is not PriceProductConfig
        or price_product_config_hash(config) != PRODUCT_EFFECTIVE_CONFIG_HASH
    ):
        raise ValueError("Shadow drift preparation requires the exact frozen price product config")
    if not all(
        isinstance(value, UUID)
        for value in (
            product_input.listing_id,
            product_input.stock.identity.asset_id,
            product_input.benchmark.identity.asset_id,
        )
    ):
        raise ValueError("Shadow drift preparation requires permanent UUID identities")
    native = calculate_price_product(
        product_input, config=config, effective_config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH
    )
    forecast = native.forecast
    if (
        forecast.method_version != "us-price-fhs-v1"
        or type(forecast.path_count) is not int
        or forecast.path_count != 8192
        or type(forecast.projections) is not tuple
        or len(forecast.projections) != 4
    ):
        raise ValueError(_DEPENDENCY_ERROR)
    for projection, (horizon, sessions) in zip(forecast.projections, _HORIZONS, strict=True):
        _encode(projection, HorizonProjection)
        if (projection.horizon, projection.sessions) != (horizon, sessions):
            raise ValueError(_DEPENDENCY_ERROR)
        _validate_projection(projection)
    selected = forecast.projections[:2]
    mean = forecast.mean_log_return
    half = None if mean is None else mean / 2.0
    candidates: tuple[HorizonProjection, ...] = selected
    if any(projection.insufficiency_reason is None for projection in selected):
        simulation = config.simulation
        filtered = filter_historical_returns(
            product_input.stock.closes,
            burn_in=simulation.filter_burn_in,
            variance_target_weight=simulation.variance_target_weight,
            variance_persistence=simulation.variance_persistence,
            innovation_weight=simulation.innovation_weight,
        )
        if mean is None or _encode(filtered.mean_log_return, float) != _encode(mean, float):
            raise ValueError(_DEPENDENCY_ERROR)
        terminals = simulate_fhs_terminal_logs(
            filtered,
            seed=forecast.seed,
            horizons=tuple(sessions for _name, sessions in _HORIZONS),
            path_count=8192,
            diagnostic_max_paths=simulation.diagnostic_max_paths,
            variance_target_weight=simulation.variance_target_weight,
            variance_persistence=simulation.variance_persistence,
            innovation_weight=simulation.innovation_weight,
        )
        _validate_terminals(terminals)
        candidate_rows: list[HorizonProjection] = []
        for index, projection in enumerate(forecast.projections):
            regenerated = projection_from_terminal_logs(
                horizon=projection.horizon,
                sessions=projection.sessions,
                terminal_logs=terminals.with_drift[index],
                zero_drift_logs=terminals.zero_drift[index],
                target_close=product_input.stock.closes[-1],
                quantiles=simulation.quantiles,
                quantile_method=simulation.quantile_method,
                return_places=config.rounding.return_decimal_places,
                price_places=config.rounding.price_decimal_places,
            )
            if _encode(regenerated, HorizonProjection) != _encode(projection, HorizonProjection):
                raise ValueError(_DEPENDENCY_ERROR)
            if index >= 2:
                continue
            if projection.insufficiency_reason is not None:
                candidate_rows.append(projection)
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                half_logs = terminals.zero_drift[index] + projection.sessions * (mean / 2.0)
            candidate_rows.append(
                projection_from_terminal_logs(
                    horizon=projection.horizon,
                    sessions=projection.sessions,
                    terminal_logs=half_logs,
                    zero_drift_logs=terminals.zero_drift[index],
                    target_close=product_input.stock.closes[-1],
                    quantiles=simulation.quantiles,
                    quantile_method=simulation.quantile_method,
                    return_places=config.rounding.return_decimal_places,
                    price_places=config.rounding.price_decimal_places,
                )
            )
        candidates = tuple(candidate_rows)
    result = ShadowDriftResult(
        schema_version="shadow-fhs-drift@1",
        research_version="shadow-fhs-drift-v1",
        assumptions_sha256=_ASSUMPTIONS_SHA256,
        native_method_version="us-price-fhs-v1",
        native_config_hash=native.effective_config_hash,
        input_hash=native.input_hash,
        listing_id=product_input.listing_id,
        target_date=product_input.target_date,
        decision_time=product_input.decision_time,
        stock_identity=product_input.stock.identity,
        benchmark_identity=product_input.benchmark.identity,
        source_execution=product_input.source_execution,
        evidence_label="research_only_unscored",
        source_provenance_status="caller_supplied_unverified",
        empirical_protocol_status="not_frozen",
        innovation_seed=forecast.seed,
        path_count=8192,
        simulation_horizons=_HORIZONS,
        horizon_roles=_ROLES,
        mean_log_return=mean,
        half_log_drift=half,
        native_forecast_insufficiency_reason=forecast.insufficiency_reason,
        projections=tuple(
            _copy_projection(arm, projection)
            for arm in _ARMS
            for projection in (candidates if arm == _ARMS[2] else selected)
        ),
    )
    _validate_result(result)
    return result


def serialize_shadow_drift(result: ShadowDriftResult) -> bytes:
    """Encode the complete unverified preparation, never an empirical verdict."""
    document = _encode(result, ShadowDriftResult)
    _validate_result(result)
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _copy_projection(arm: ArmId, projection: HorizonProjection) -> ShadowProjection:
    if projection.horizon not in ("6m", "12m") or projection.sessions not in (126, 252):
        raise ValueError(_DEPENDENCY_ERROR)
    zero = arm == "zero_log_drift_fhs_control"
    horizon: Literal["6m", "12m"] = "6m" if projection.horizon == "6m" else "12m"
    sessions: Literal[126, 252] = 126 if projection.sessions == 126 else 252
    return ShadowProjection(
        arm_id=arm,
        horizon=horizon,
        sessions=sessions,
        quantile_levels=projection.quantile_levels,
        central_model_mass=projection.central_model_mass,
        raw_returns=projection.zero_drift_raw_returns if zero else projection.raw_returns,
        ledger_returns=projection.zero_drift_ledger_returns if zero else projection.ledger_returns,
        raw_prices=projection.zero_drift_raw_prices if zero else projection.raw_prices,
        ledger_prices=projection.zero_drift_ledger_prices if zero else projection.ledger_prices,
        insufficiency_reason=projection.insufficiency_reason,
    )


def _validate_terminals(terminals: SimulationTerminals) -> None:
    if (
        type(terminals) is not SimulationTerminals
        or type(terminals.horizons) is not tuple
        or any(type(value) is not int for value in terminals.horizons)
        or terminals.horizons != (126, 252, 756, 1260)
        or type(terminals.path_count) is not int
        or terminals.path_count != 8192
    ):
        raise ValueError(_DEPENDENCY_ERROR)
    for arrays in (terminals.with_drift, terminals.zero_drift):
        if type(arrays) is not tuple or len(arrays) != 4:
            raise ValueError(_DEPENDENCY_ERROR)
        for array in arrays:
            if type(array) is not np.ndarray or array.dtype != np.float64 or array.shape != (8192,):
                raise ValueError(_DEPENDENCY_ERROR)


def _validate_projection(projection: ShadowProjection | HorizonProjection) -> None:
    if projection.quantile_levels != (0.2, 0.5, 0.8) or projection.central_model_mass != 0.6:
        raise ValueError(_SCHEMA_ERROR)
    triplets: tuple[RawTriplet | LedgerTriplet | None, ...] = (
        projection.raw_returns,
        projection.ledger_returns,
        projection.raw_prices,
        projection.ledger_prices,
    )
    if isinstance(projection, HorizonProjection):
        triplets += (
            projection.zero_drift_raw_returns,
            projection.zero_drift_ledger_returns,
            projection.zero_drift_raw_prices,
            projection.zero_drift_ledger_prices,
        )
    if projection.insufficiency_reason is not None:
        if not projection.insufficiency_reason or any(value is not None for value in triplets):
            raise ValueError(_SCHEMA_ERROR)
    elif any(value is None or not value.lower <= value.median <= value.upper for value in triplets):
        raise ValueError(_SCHEMA_ERROR)


def _validate_result(result: ShadowDriftResult) -> None:
    _encode(result, ShadowDriftResult)
    if (
        result.assumptions_sha256 != _ASSUMPTIONS_SHA256
        or result.native_config_hash != PRODUCT_EFFECTIVE_CONFIG_HASH
        or result.simulation_horizons != _HORIZONS
        or result.horizon_roles != _ROLES
        or not 0 <= result.innovation_seed < 2**128
        or tuple((p.arm_id, p.horizon, p.sessions) for p in result.projections)
        != tuple((arm, h, s) for arm in _ARMS for h, s in _HORIZONS[:2])
        or _encode(result.half_log_drift, float | None)
        != _encode(
            None if result.mean_log_return is None else result.mean_log_return / 2.0, float | None
        )
        or result.native_forecast_insufficiency_reason == ""
    ):
        raise ValueError(_SCHEMA_ERROR)
    for checksum in (
        result.input_hash,
        result.stock_identity.sha256,
        result.benchmark_identity.sha256,
    ):
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError(_SCHEMA_ERROR)
    for projection in result.projections:
        _validate_projection(projection)
    for index in range(2):
        control, zero, candidate = (result.projections[index + offset] for offset in (0, 2, 4))
        if zero.insufficiency_reason != control.insufficiency_reason or (
            control.insufficiency_reason is not None
            and candidate.insufficiency_reason != control.insufficiency_reason
        ):
            raise ValueError(_SCHEMA_ERROR)


def _encode(value: object, schema: object) -> object:
    """Schema-closed identity encoding shared by output and dependency checks."""
    origin, arguments = get_origin(schema), get_args(schema)
    if origin is UnionType:
        if value is None and type(None) in arguments:
            return None
        return _encode(value, next(item for item in arguments if item is not type(None)))
    if origin is Literal:
        if not any(type(value) is type(item) and value == item for item in arguments):
            raise ValueError(_SCHEMA_ERROR)
        return value
    if origin is tuple:
        if type(value) is not tuple:
            raise ValueError(_SCHEMA_ERROR)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return [_encode(item, arguments[0]) for item in value]
        if len(value) != len(arguments):
            raise ValueError(_SCHEMA_ERROR)
        return [_encode(item, kind) for item, kind in zip(value, arguments, strict=True)]
    if type(value) is not schema:
        raise ValueError(_SCHEMA_ERROR)
    if schema in (
        ShadowDriftResult,
        ShadowProjection,
        HorizonProjection,
        PriceInputIdentity,
        SourceExecutionBinding,
        RawTriplet,
        LedgerTriplet,
    ):
        annotations = get_type_hints(schema)
        return {
            field.name: _encode(getattr(value, field.name), annotations[field.name])
            for field in fields(schema)
        }
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(_SCHEMA_ERROR)
        return value.hex()
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(_SCHEMA_ERROR)
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise ValueError(_SCHEMA_ERROR)
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if type(value) in (str, int):
        return value
    raise ValueError(_SCHEMA_ERROR)

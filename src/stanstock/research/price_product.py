"""Pure prospective price-research operators.

The functions in this module perform no ORM, filesystem, clock, or provider
work. Callers must select immutable price assets through ``AsOfData`` and pass
the exact common exchange-session calendar used for the decision. Raw
floating-point calculations are retained separately from half-even ledger
rounding: returns are persisted to four decimal places and native prices to
six. A rounded price is therefore not recomputed from an already-rounded
return; both values are independently rounded from the same raw trajectory.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from typing import Literal
from uuid import UUID

import numpy as np
import numpy.typing as npt

from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PriceProductConfig,
    price_product_config_hash,
)

Direction = Literal["positive", "negative", "mixed"]
Suggestion = Literal["buy", "hold", "avoid"]
RelativeVolatilityLabel = Literal["low", "medium", "high", "very_high", "insufficient"]
SourceExecutionMode = Literal["provider", "synthetic_demo"]
EvidenceGrade = Literal["research", "observed"]

_SEED_DOMAIN = "stanstock-research-product-seed-v1"
_INPUT_HASH_DOMAIN = "stanstock-research-product-input-v1"
_MAX_LEDGER_RETURN = Decimal("999999.9999")
_MAX_LEDGER_PRICE = Decimal("99999999999999.999999")
_SIMULATION_CHUNK_PATHS = 256


class PriceProductInputError(ValueError):
    """A precise data insufficiency that must not be converted to a zero."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class PriceInputIdentity:
    asset_id: UUID
    provider: str
    subject: str
    sha256: str
    retrieved_at: datetime
    available_at: datetime


@dataclass(frozen=True, slots=True)
class PriceSeries:
    identity: PriceInputIdentity
    currency: str
    dates: tuple[date, ...]
    closes: tuple[float, ...]
    volumes: tuple[float | None, ...] | None = None
    volume_adjustment_compatible: bool = False


@dataclass(frozen=True, slots=True)
class SourceExecutionBinding:
    """Bind calculation mode to actual source identity and evidence grade.

    The future writer/reader must construct this from registered assets, not
    caller labels. Synthetic demo evidence is always research-grade and can
    never become an observed issuance.
    """

    mode: SourceExecutionMode
    evidence_grade: EvidenceGrade


@dataclass(frozen=True, slots=True)
class PriceProductInput:
    """Complete immutable inputs admitted for one calculation.

    ``decision_time`` is the recorded source-availability boundary, never a
    fresh wall-clock value chosen while replaying. For observed evidence the
    writer must set it to the logical data cutoff. A research-grade
    current-vintage reconstruction may admit assets retrieved later than the
    historical cutoff, but its writer must still cap every supplied price row
    at that historical cutoff and record actual generation time separately.
    The boundary and actual asset timestamps are hashed so an issuance can be
    reproduced without fabricating historical availability.
    """

    listing_id: UUID
    target_date: date
    decision_time: datetime
    calendar_sessions: tuple[date, ...]
    stock: PriceSeries
    benchmark: PriceSeries
    source_execution: SourceExecutionBinding
    source_eligible: bool = True
    source_ineligibility_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MomentumResult:
    stock_log_momentum: float
    stock_price_return: float
    benchmark_log_momentum: float
    relative_log_momentum: float
    direction: Direction


@dataclass(frozen=True, slots=True)
class FilteredReturns:
    mean_log_return: float
    population_variance: float
    terminal_variance: float
    standardized_returns: tuple[float, ...]
    residuals: tuple[float, ...]
    residual_center: float
    residual_scale: float


@dataclass(frozen=True, slots=True)
class RawTriplet:
    lower: float
    median: float
    upper: float


@dataclass(frozen=True, slots=True)
class LedgerTriplet:
    lower: Decimal
    median: Decimal
    upper: Decimal


@dataclass(frozen=True, slots=True)
class HorizonProjection:
    horizon: str
    sessions: int
    quantile_levels: tuple[float, float, float]
    central_model_mass: float
    raw_returns: RawTriplet | None
    ledger_returns: LedgerTriplet | None
    raw_prices: RawTriplet | None
    ledger_prices: LedgerTriplet | None
    zero_drift_raw_returns: RawTriplet | None
    zero_drift_ledger_returns: LedgerTriplet | None
    zero_drift_raw_prices: RawTriplet | None
    zero_drift_ledger_prices: LedgerTriplet | None
    insufficiency_reason: str | None


@dataclass(frozen=True, slots=True)
class FHSForecast:
    method_version: str
    path_count: int
    seed: int
    mean_log_return: float | None
    population_variance: float | None
    terminal_variance: float | None
    residual_count: int
    residual_mean: float | None
    residual_second_moment: float | None
    quantile_method: str
    projections: tuple[HorizonProjection, ...]
    insufficiency_reason: str | None


@dataclass(frozen=True, slots=True)
class RiskResult:
    annualized_volatility: float | None
    benchmark_annualized_volatility: float | None
    relative_volatility: float | None
    relative_volatility_label: RelativeVolatilityLabel
    maximum_drawdown: float
    average_dollar_turnover_20d: float | None
    insufficiency_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    raw_direction: Direction | None
    suggestion: Suggestion | None
    decision_horizon_sessions: int
    blocking_reasons: tuple[str, ...]
    allocation_restriction: str | None


@dataclass(frozen=True, slots=True)
class PriceProductResult:
    payload_schema: str
    product_version: str
    effective_config_hash: str
    input_hash: str
    stock_asset_id: UUID
    benchmark_asset_id: UUID
    source_execution: SourceExecutionBinding
    target_date: date
    momentum: MomentumResult | None
    momentum_insufficiency_reason: str | None
    risk: RiskResult
    recommendation: RecommendationResult
    forecast: FHSForecast
    overall_score: None = None
    risk_score: None = None
    confidence: None = None
    probability_positive: None = None
    confidence_status: Literal["not_estimated"] = "not_estimated"


@dataclass(frozen=True, slots=True)
class SimulationTerminals:
    """Unrounded terminal log returns for deterministic diagnostic replay."""

    horizons: tuple[int, ...]
    path_count: int
    with_drift: tuple[npt.NDArray[np.float64], ...]
    zero_drift: tuple[npt.NDArray[np.float64], ...]


def deterministic_seed(
    *,
    method_version: str,
    effective_config_hash: str,
    listing_id: UUID,
    target_date: date,
) -> int:
    """Return the first 128 SHA-256 bits, interpreted as a big-endian integer."""
    if "\n" in method_version or "\n" in effective_config_hash:
        raise ValueError("Seed identity text cannot contain newlines")
    canonical = (
        f"{_SEED_DOMAIN}\n"
        f"method_version={method_version}\n"
        f"effective_config_hash={effective_config_hash}\n"
        f"listing_uuid={listing_id}\n"
        f"target_date={target_date.isoformat()}\n"
    )
    return int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest()[:16], "big")


def prediction_model_version(*, method_version: str, issuance_id: UUID) -> str:
    """Derive a unique immutable issuance version without changing method identity."""
    if not method_version or len(method_version) > 31:
        raise ValueError("method_version must contain 1 to 31 characters")
    return f"{method_version}-{issuance_id.hex[:8]}"


def complete_input_hash(product_input: PriceProductInput) -> str:
    """Hash complete identities, values, volumes, and the supplied calendar."""

    def identity_payload(identity: PriceInputIdentity) -> dict[str, str]:
        return {
            "asset_id": str(identity.asset_id),
            "provider": identity.provider,
            "subject": identity.subject,
            "sha256": identity.sha256,
            "retrieved_at": identity.retrieved_at.isoformat(),
            "available_at": identity.available_at.isoformat(),
        }

    def series_payload(series: PriceSeries) -> dict[str, object]:
        return {
            "identity": identity_payload(series.identity),
            "currency": series.currency,
            "dates": [value.isoformat() for value in series.dates],
            "closes": [_canonical_float(value) for value in series.closes],
            "volumes": (
                None
                if series.volumes is None
                else [
                    None if value is None else _canonical_float(value) for value in series.volumes
                ]
            ),
            "volume_adjustment_compatible": series.volume_adjustment_compatible,
        }

    document = {
        "domain": _INPUT_HASH_DOMAIN,
        "listing_id": str(product_input.listing_id),
        "target_date": product_input.target_date.isoformat(),
        "decision_time": product_input.decision_time.isoformat(),
        "source_execution": {
            "mode": product_input.source_execution.mode,
            "evidence_grade": product_input.source_execution.evidence_grade,
        },
        "calendar_sessions": [value.isoformat() for value in product_input.calendar_sessions],
        "source_eligible": product_input.source_eligible,
        "source_ineligibility_reasons": list(product_input.source_ineligibility_reasons),
        "stock": series_payload(product_input.stock),
        "benchmark": series_payload(product_input.benchmark),
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calculate_momentum(
    stock_closes: tuple[float, ...],
    benchmark_closes: tuple[float, ...],
    *,
    lookback_sessions: int = 252,
    skip_sessions: int = 21,
) -> MomentumResult:
    """Calculate symmetric T-252 to T-21 stock and SPY log momentum."""
    expected = lookback_sessions + 1
    if len(stock_closes) < expected or len(benchmark_closes) < expected:
        raise PriceProductInputError(
            "momentum_history_insufficient",
            f"Momentum requires at least {expected} aligned closes",
        )
    start_index = len(stock_closes) - 1 - lookback_sessions
    end_index = len(stock_closes) - 1 - skip_sessions
    benchmark_start = len(benchmark_closes) - 1 - lookback_sessions
    benchmark_end = len(benchmark_closes) - 1 - skip_sessions
    stock_log = _safe_log_ratio(
        stock_closes[start_index],
        stock_closes[end_index],
        reason_code="stock_momentum_invalid",
    )
    benchmark_log = _safe_log_ratio(
        benchmark_closes[benchmark_start],
        benchmark_closes[benchmark_end],
        reason_code="benchmark_momentum_invalid",
    )
    relative = stock_log - benchmark_log
    with np.errstate(over="ignore", invalid="ignore"):
        stock_price_return = float(np.expm1(stock_log))
    if not all(
        math.isfinite(value) for value in (stock_log, benchmark_log, relative, stock_price_return)
    ):
        raise PriceProductInputError(
            "momentum_unrepresentable",
            "Momentum calculation produced a non-finite value",
        )
    if stock_log > 0 and relative > 0:
        direction: Direction = "positive"
    elif stock_log < 0 and relative < 0:
        direction = "negative"
    else:
        direction = "mixed"
    return MomentumResult(
        stock_log_momentum=stock_log,
        stock_price_return=stock_price_return,
        benchmark_log_momentum=benchmark_log,
        relative_log_momentum=relative,
        direction=direction,
    )


def filter_historical_returns(
    closes: tuple[float, ...],
    *,
    burn_in: int = 252,
    variance_target_weight: float = 0.01,
    variance_persistence: float = 0.94,
    innovation_weight: float = 0.05,
) -> FilteredReturns:
    """Apply the frozen variance-targeted filter to 756 daily log returns."""
    close_values = _finite_positive_array(closes, "filter_close_invalid")
    if close_values.size != 757:
        raise PriceProductInputError(
            "filter_history_length",
            "The historical filter requires exactly 757 closes",
        )
    returns = np.diff(np.log(close_values))
    if returns.size != 756 or not np.all(np.isfinite(returns)):
        raise PriceProductInputError(
            "filter_return_invalid",
            "Historical log returns are non-finite",
        )
    mean = float(np.mean(returns))
    centered = returns - mean
    variance = float(np.mean(centered * centered))
    if not math.isfinite(variance) or variance <= 0:
        raise PriceProductInputError(
            "filter_variance_degenerate",
            "Historical population variance must be finite and positive",
        )

    q = variance
    standardized = np.empty(returns.size, dtype=np.float64)
    for index, innovation in enumerate(centered):
        if not math.isfinite(q) or q <= 0:
            raise PriceProductInputError(
                "filter_variance_invalid",
                "Historical variance recursion became non-finite or non-positive",
            )
        standardized[index] = innovation / math.sqrt(q)
        q = (
            variance_target_weight * variance
            + variance_persistence * q
            + innovation_weight * innovation * innovation
        )
    if not math.isfinite(q) or q <= 0 or not np.all(np.isfinite(standardized)):
        raise PriceProductInputError(
            "filter_variance_invalid",
            "Historical variance recursion produced an invalid terminal state",
        )
    residual_population = standardized[burn_in:]
    if residual_population.size != 504:
        raise PriceProductInputError(
            "filter_residual_count",
            "The residual population must contain exactly 504 observations",
        )
    residual_center = float(np.mean(residual_population))
    deviations = residual_population - residual_center
    residual_scale = float(np.sqrt(np.mean(deviations * deviations)))
    if not math.isfinite(residual_scale) or residual_scale <= 0:
        raise PriceProductInputError(
            "filter_residual_degenerate",
            "Centered residual second moment must be finite and positive",
        )
    residuals = deviations / residual_scale
    if not np.all(np.isfinite(residuals)):
        raise PriceProductInputError(
            "filter_residual_invalid",
            "Centered and rescaled residuals are non-finite",
        )
    return FilteredReturns(
        mean_log_return=mean,
        population_variance=variance,
        terminal_variance=q,
        standardized_returns=tuple(float(value) for value in standardized),
        residuals=tuple(float(value) for value in residuals),
        residual_center=residual_center,
        residual_scale=residual_scale,
    )


def simulate_fhs_terminal_logs(
    filtered: FilteredReturns,
    *,
    seed: int,
    horizons: tuple[int, ...] = (126, 252, 756, 1260),
    path_count: int,
    diagnostic_max_paths: int = 16384,
    variance_target_weight: float = 0.01,
    variance_persistence: float = 0.94,
    innovation_weight: float = 0.05,
) -> SimulationTerminals:
    """Generate path-major PCG64 trajectories without retaining a path cube."""
    if type(path_count) is not int or path_count <= 0 or path_count > diagnostic_max_paths:
        raise ValueError(f"path_count must be in [1, {diagnostic_max_paths}] for diagnostics")
    if (
        not horizons
        or tuple(sorted(set(horizons))) != horizons
        or horizons[-1] > 1260
        or horizons[0] <= 0
    ):
        raise ValueError("horizons must be unique increasing sessions in [1, 1260]")
    residuals = np.asarray(filtered.residuals, dtype=np.float64)
    if residuals.shape != (504,) or not np.all(np.isfinite(residuals)):
        raise PriceProductInputError(
            "simulation_residual_invalid",
            "Simulation requires exactly 504 finite residuals",
        )
    if (
        not math.isfinite(filtered.mean_log_return)
        or not math.isfinite(filtered.population_variance)
        or filtered.population_variance <= 0
        or not math.isfinite(filtered.terminal_variance)
        or filtered.terminal_variance <= 0
    ):
        raise PriceProductInputError(
            "simulation_filter_state_invalid",
            "Simulation filter state must be finite and positive",
        )

    generator = np.random.Generator(np.random.PCG64(seed))
    terminal_arrays = [np.full(path_count, np.nan, dtype=np.float64) for _horizon in horizons]
    maximum_horizon = horizons[-1]
    valid_horizon_count = len(horizons)
    for path_start in range(0, path_count, _SIMULATION_CHUNK_PATHS):
        if valid_horizon_count == 0:
            break
        chunk_size = min(_SIMULATION_CHUNK_PATHS, path_count - path_start)
        # C-order rows consume the random stream path by path. Keeping this
        # chunk size fixed means a 16,384-path diagnostic has the exact first
        # 8,192 trajectories of the production run.
        sampled_indices = generator.integers(
            0,
            residuals.size,
            size=(chunk_size, maximum_horizon),
            dtype=np.int64,
        )
        q = np.full(chunk_size, filtered.terminal_variance, dtype=np.float64)
        cumulative = np.zeros(chunk_size, dtype=np.float64)
        horizon_index = 0
        with np.errstate(over="ignore", invalid="ignore"):
            for session_index in range(maximum_horizon):
                innovations = np.sqrt(q) * residuals[sampled_indices[:, session_index]]
                cumulative += filtered.mean_log_return + innovations
                next_q = (
                    variance_target_weight * filtered.population_variance
                    + variance_persistence * q
                    + innovation_weight * innovations * innovations
                )
                if not np.all(np.isfinite(cumulative)):
                    valid_horizon_count = min(valid_horizon_count, horizon_index)
                    break
                if session_index + 1 == horizons[horizon_index]:
                    terminal_arrays[horizon_index][path_start : path_start + chunk_size] = (
                        cumulative
                    )
                    horizon_index += 1
                    if horizon_index == valid_horizon_count:
                        break
                if not np.all(np.isfinite(next_q)) or np.any(next_q <= 0):
                    valid_horizon_count = min(valid_horizon_count, horizon_index)
                    break
                q = next_q
        if horizon_index < valid_horizon_count:
            valid_horizon_count = horizon_index
        if any(
            not np.all(np.isfinite(values[path_start : path_start + chunk_size]))
            for values in terminal_arrays[:valid_horizon_count]
        ):
            valid_horizon_count = 0
    zero_drift = tuple(
        (
            values - horizon * filtered.mean_log_return
            if index < valid_horizon_count
            else np.full(path_count, np.nan, dtype=np.float64)
        )
        for index, (values, horizon) in enumerate(zip(terminal_arrays, horizons, strict=True))
    )
    return SimulationTerminals(
        horizons=horizons,
        path_count=path_count,
        with_drift=tuple(terminal_arrays),
        zero_drift=zero_drift,
    )


def project_fhs(
    filtered: FilteredReturns,
    *,
    target_close: float,
    seed: int,
    config: PriceProductConfig,
) -> FHSForecast:
    """Produce the four production projections using exactly 8,192 paths."""
    simulation = config.simulation
    horizons = tuple(sessions for _name, sessions in simulation.horizons)
    try:
        terminals = simulate_fhs_terminal_logs(
            filtered,
            seed=seed,
            horizons=horizons,
            path_count=simulation.production_paths,
            diagnostic_max_paths=simulation.diagnostic_max_paths,
            variance_target_weight=simulation.variance_target_weight,
            variance_persistence=simulation.variance_persistence,
            innovation_weight=simulation.innovation_weight,
        )
    except PriceProductInputError as exc:
        return _withheld_forecast(config, seed=seed, reason=exc.reason_code)
    projections = tuple(
        (
            _projection_from_terminal_logs(
                horizon=name,
                sessions=sessions,
                terminal_logs=terminals.with_drift[index],
                zero_drift_logs=terminals.zero_drift[index],
                target_close=target_close,
                quantiles=simulation.quantiles,
                quantile_method=simulation.quantile_method,
                return_places=config.rounding.return_decimal_places,
                price_places=config.rounding.price_decimal_places,
            )
            if (
                np.all(np.isfinite(terminals.with_drift[index]))
                and np.all(np.isfinite(terminals.zero_drift[index]))
            )
            else _withheld_projection(
                name,
                sessions,
                simulation.quantiles,
                "simulation_nonfinite",
            )
        )
        for index, (name, sessions) in enumerate(simulation.horizons)
    )
    global_reason = (
        None
        if any(projection.insufficiency_reason is None for projection in projections)
        else "all_projection_horizons_withheld"
    )
    residual_values = np.asarray(filtered.residuals, dtype=np.float64)
    return FHSForecast(
        method_version=simulation.method_version,
        path_count=simulation.production_paths,
        seed=seed,
        mean_log_return=filtered.mean_log_return,
        population_variance=filtered.population_variance,
        terminal_variance=filtered.terminal_variance,
        residual_count=len(filtered.residuals),
        residual_mean=float(np.mean(residual_values)),
        residual_second_moment=float(np.mean(residual_values * residual_values)),
        quantile_method=simulation.quantile_method,
        projections=projections,
        insufficiency_reason=global_reason,
    )


def calculate_price_product(
    product_input: PriceProductInput,
    *,
    config: PriceProductConfig,
    effective_config_hash: str = PRODUCT_EFFECTIVE_CONFIG_HASH,
) -> PriceProductResult:
    """Calculate one listing's scoreless decision, risk, and FHS projections."""
    actual_hash = price_product_config_hash(config)
    if effective_config_hash != actual_hash:
        raise ValueError("The supplied effective config hash does not match the typed config")
    _validate_product_input(product_input, config=config)
    input_hash = complete_input_hash(product_input)
    seed = deterministic_seed(
        method_version=FHS_METHOD_VERSION,
        effective_config_hash=effective_config_hash,
        listing_id=product_input.listing_id,
        target_date=product_input.target_date,
    )

    momentum: MomentumResult | None
    momentum_reason: str | None = None
    try:
        momentum = calculate_momentum(
            product_input.stock.closes,
            product_input.benchmark.closes,
            lookback_sessions=config.momentum.lookback_sessions,
            skip_sessions=config.momentum.skip_sessions,
        )
    except PriceProductInputError as exc:
        momentum = None
        momentum_reason = exc.reason_code

    stock_filter: FilteredReturns | None
    stock_filter_reason: str | None = None
    try:
        stock_filter = filter_historical_returns(
            product_input.stock.closes,
            burn_in=config.simulation.filter_burn_in,
            variance_target_weight=config.simulation.variance_target_weight,
            variance_persistence=config.simulation.variance_persistence,
            innovation_weight=config.simulation.innovation_weight,
        )
    except PriceProductInputError as exc:
        stock_filter = None
        stock_filter_reason = exc.reason_code

    benchmark_filter: FilteredReturns | None
    benchmark_filter_reason: str | None = None
    try:
        benchmark_filter = filter_historical_returns(
            product_input.benchmark.closes,
            burn_in=config.simulation.filter_burn_in,
            variance_target_weight=config.simulation.variance_target_weight,
            variance_persistence=config.simulation.variance_persistence,
            innovation_weight=config.simulation.innovation_weight,
        )
    except PriceProductInputError as exc:
        benchmark_filter = None
        benchmark_filter_reason = exc.reason_code

    if stock_filter is None:
        forecast = _withheld_forecast(
            config,
            seed=seed,
            reason=stock_filter_reason or "stock_filter_unavailable",
        )
    else:
        forecast = project_fhs(
            stock_filter,
            target_close=product_input.stock.closes[-1],
            seed=seed,
            config=config,
        )
    risk = calculate_risk(
        product_input.stock,
        stock_filter=stock_filter,
        benchmark_filter=benchmark_filter,
        stock_filter_reason=stock_filter_reason,
        benchmark_filter_reason=benchmark_filter_reason,
        config=config,
    )
    recommendation = apply_recommendation_policy(
        momentum=momentum,
        momentum_insufficiency_reason=momentum_reason,
        risk=risk,
        target_close=product_input.stock.closes[-1],
        source_eligible=product_input.source_eligible,
        source_ineligibility_reasons=product_input.source_ineligibility_reasons,
        config=config,
    )
    return PriceProductResult(
        payload_schema=config.payload_schema,
        product_version=config.product_version,
        effective_config_hash=effective_config_hash,
        input_hash=input_hash,
        stock_asset_id=product_input.stock.identity.asset_id,
        benchmark_asset_id=product_input.benchmark.identity.asset_id,
        source_execution=product_input.source_execution,
        target_date=product_input.target_date,
        momentum=momentum,
        momentum_insufficiency_reason=momentum_reason,
        risk=risk,
        recommendation=recommendation,
        forecast=forecast,
    )


def calculate_risk(
    stock: PriceSeries,
    *,
    stock_filter: FilteredReturns | None,
    benchmark_filter: FilteredReturns | None,
    stock_filter_reason: str | None,
    benchmark_filter_reason: str | None,
    config: PriceProductConfig,
) -> RiskResult:
    reasons: list[str] = []
    annualized = (
        math.sqrt(config.risk.annualization_sessions * stock_filter.terminal_variance)
        if stock_filter is not None
        else None
    )
    benchmark_annualized = (
        math.sqrt(config.risk.annualization_sessions * benchmark_filter.terminal_variance)
        if benchmark_filter is not None
        else None
    )
    if stock_filter_reason is not None:
        reasons.append(f"stock_volatility:{stock_filter_reason}")
    if benchmark_filter_reason is not None:
        reasons.append(f"benchmark_volatility:{benchmark_filter_reason}")
    relative: float | None = None
    label: RelativeVolatilityLabel = "insufficient"
    if (
        annualized is not None
        and benchmark_annualized is not None
        and math.isfinite(annualized)
        and math.isfinite(benchmark_annualized)
        and benchmark_annualized > 0
    ):
        relative = annualized / benchmark_annualized
        if relative <= 1:
            label = "low"
        elif relative <= 2:
            label = "medium"
        elif relative <= 3:
            label = "high"
        else:
            label = "very_high"
    else:
        reasons.append("relative_volatility_unavailable")

    drawdown_closes = stock.closes[-(config.risk.drawdown_return_sessions + 1) :]
    maximum_drawdown = _maximum_drawdown(drawdown_closes)
    turnover: float | None = None
    if stock.volumes is None:
        reasons.append("dollar_turnover_volume_missing")
    elif not stock.volume_adjustment_compatible:
        reasons.append("dollar_turnover_adjustment_incompatible")
    else:
        window = config.risk.liquidity_sessions
        volume_window = stock.volumes[-window:]
        close_window = stock.closes[-window:]
        if len(volume_window) != window or any(value is None for value in volume_window):
            reasons.append("dollar_turnover_volume_missing")
        else:
            products = [
                close * volume
                for close, raw_volume in zip(close_window, volume_window, strict=True)
                if (volume := raw_volume) is not None
            ]
            turnover = math.fsum(products) / window
            if not math.isfinite(turnover):
                turnover = None
                reasons.append("dollar_turnover_nonfinite")
    return RiskResult(
        annualized_volatility=annualized,
        benchmark_annualized_volatility=benchmark_annualized,
        relative_volatility=relative,
        relative_volatility_label=label,
        maximum_drawdown=maximum_drawdown,
        average_dollar_turnover_20d=turnover,
        insufficiency_reasons=tuple(dict.fromkeys(reasons)),
    )


def apply_recommendation_policy(
    *,
    momentum: MomentumResult | None,
    momentum_insufficiency_reason: str | None,
    risk: RiskResult,
    target_close: float,
    source_eligible: bool,
    source_ineligibility_reasons: tuple[str, ...],
    config: PriceProductConfig,
) -> RecommendationResult:
    horizon = config.momentum.decision_horizon_sessions
    allocation_restriction = (
        "speculative_watch_0_percent_new_allocation"
        if target_close < config.risk.buy_minimum_target_close
        else None
    )
    if momentum is None:
        return RecommendationResult(
            raw_direction=None,
            suggestion=None,
            decision_horizon_sessions=horizon,
            blocking_reasons=(momentum_insufficiency_reason or "momentum_unavailable",),
            allocation_restriction=allocation_restriction,
        )
    if momentum.direction == "negative":
        return RecommendationResult(
            raw_direction="negative",
            suggestion="avoid",
            decision_horizon_sessions=horizon,
            blocking_reasons=(),
            allocation_restriction=allocation_restriction,
        )
    if momentum.direction == "mixed":
        return RecommendationResult(
            raw_direction="mixed",
            suggestion="hold",
            decision_horizon_sessions=horizon,
            blocking_reasons=("mixed_momentum_signal",),
            allocation_restriction=allocation_restriction,
        )

    blockers: list[str] = []
    if not source_eligible:
        blockers.extend(source_ineligibility_reasons or ("source_ineligible",))
    if risk.relative_volatility is None:
        blockers.append("relative_volatility_unavailable")
    elif risk.relative_volatility > config.risk.buy_max_relative_volatility:
        blockers.append("relative_volatility_above_buy_limit")
    if risk.maximum_drawdown < config.risk.buy_minimum_drawdown:
        blockers.append("drawdown_below_buy_limit")
    if risk.average_dollar_turnover_20d is None:
        blockers.append("dollar_turnover_unavailable")
    elif risk.average_dollar_turnover_20d < config.risk.buy_minimum_dollar_turnover:
        blockers.append("dollar_turnover_below_buy_minimum")
    if target_close < config.risk.buy_minimum_target_close:
        blockers.append("target_close_below_buy_minimum")
    return RecommendationResult(
        raw_direction="positive",
        suggestion="hold" if blockers else "buy",
        decision_horizon_sessions=horizon,
        blocking_reasons=tuple(dict.fromkeys(blockers)),
        allocation_restriction=allocation_restriction,
    )


def _validate_product_input(
    product_input: PriceProductInput, *, config: PriceProductConfig
) -> None:
    if product_input.target_date > product_input.decision_time.date():
        raise PriceProductInputError(
            "target_after_decision_time",
            "Target date cannot be after the source-availability boundary",
        )
    if len(product_input.calendar_sessions) != config.required_closes:
        raise PriceProductInputError(
            "calendar_session_count",
            f"Expected exactly {config.required_closes} calendar sessions",
        )
    if tuple(sorted(set(product_input.calendar_sessions))) != product_input.calendar_sessions:
        raise PriceProductInputError(
            "calendar_sessions_invalid",
            "Calendar sessions must be unique and strictly increasing",
        )
    if product_input.calendar_sessions[-1] != product_input.target_date:
        raise PriceProductInputError(
            "calendar_target_mismatch",
            "The final calendar session must equal the target date",
        )
    execution = product_input.source_execution
    if execution.mode not in ("provider", "synthetic_demo"):
        raise PriceProductInputError(
            "source_execution_mode_invalid",
            "Source execution mode must be provider or synthetic_demo",
        )
    if execution.evidence_grade not in ("research", "observed"):
        raise PriceProductInputError(
            "source_evidence_grade_invalid",
            "Source evidence grade must be research or observed",
        )
    if execution.mode == "synthetic_demo":
        if execution.evidence_grade != "research":
            raise PriceProductInputError(
                "synthetic_demo_observed_forbidden",
                "Synthetic demo inputs can only produce research-grade evidence",
            )
        expected_provider = "synthetic_demo"
    else:
        expected_provider = config.price_provider
    _validate_series(
        product_input.stock,
        role="stock",
        product_input=product_input,
        config=config,
        expected_provider=expected_provider,
    )
    _validate_series(
        product_input.benchmark,
        role="benchmark",
        product_input=product_input,
        config=config,
        expected_provider=expected_provider,
    )
    if product_input.stock.dates != product_input.benchmark.dates:
        raise PriceProductInputError(
            "stock_benchmark_dates_mismatch",
            "Stock and benchmark must use identical session dates",
        )
    if product_input.stock.dates != product_input.calendar_sessions:
        raise PriceProductInputError(
            "price_calendar_mismatch",
            "Price dates must exactly match the supplied exchange calendar",
        )
    if product_input.benchmark.identity.subject != config.benchmark_subject:
        raise PriceProductInputError(
            "benchmark_identity_mismatch",
            "Benchmark subject does not match the reviewed config",
        )
    if product_input.stock.identity.subject == config.benchmark_subject:
        raise PriceProductInputError(
            "stock_subject_is_benchmark",
            "Stock input must not use the configured benchmark subject",
        )


def _validate_series(
    series: PriceSeries,
    *,
    role: str,
    product_input: PriceProductInput,
    config: PriceProductConfig,
    expected_provider: str,
) -> None:
    if series.currency != config.currency:
        raise PriceProductInputError(
            f"{role}_currency_invalid",
            f"{role.title()} input must use {config.currency}",
        )
    if series.identity.provider != expected_provider:
        raise PriceProductInputError(
            f"{role}_provider_invalid",
            f"{role.title()} provider does not match the source execution mode",
        )
    if len(series.dates) != config.required_closes or len(series.closes) != config.required_closes:
        raise PriceProductInputError(
            f"{role}_history_length",
            f"{role.title()} input requires exactly {config.required_closes} closes",
        )
    if series.volumes is not None and len(series.volumes) != config.required_closes:
        raise PriceProductInputError(
            f"{role}_volume_length",
            f"{role.title()} volumes must align with every close",
        )
    if series.dates[-1] != product_input.target_date:
        raise PriceProductInputError(
            f"{role}_target_mismatch",
            f"{role.title()} final close must be on the target date",
        )
    _finite_positive_array(series.closes, f"{role}_close_invalid")
    if series.volumes is not None:
        for volume in series.volumes:
            if volume is None:
                continue
            if (
                isinstance(volume, bool)
                or not isinstance(volume, (int, float))
                or not math.isfinite(float(volume))
                or volume < 0
            ):
                raise PriceProductInputError(
                    f"{role}_volume_invalid",
                    f"{role.title()} volumes must be finite and non-negative",
                )
    identity = series.identity
    if not identity.subject or not _valid_sha256(identity.sha256):
        raise PriceProductInputError(
            f"{role}_asset_identity_invalid",
            f"{role.title()} immutable asset identity is incomplete",
        )
    if (
        identity.available_at > product_input.decision_time
        or identity.retrieved_at > product_input.decision_time
    ):
        raise PriceProductInputError(
            f"{role}_asset_after_decision",
            f"{role.title()} asset was not admitted by the source-availability boundary",
        )


def projection_from_terminal_logs(
    *,
    horizon: str,
    sessions: int,
    terminal_logs: npt.NDArray[np.float64],
    zero_drift_logs: npt.NDArray[np.float64],
    target_close: float,
    quantiles: tuple[float, float, float],
    quantile_method: Literal["linear"],
    return_places: int,
    price_places: int,
) -> HorizonProjection:
    """Project existing terminal paths without changing simulation semantics."""

    if not (np.all(np.isfinite(terminal_logs)) and np.all(np.isfinite(zero_drift_logs))):
        return _withheld_projection(horizon, sessions, quantiles, "simulation_nonfinite")
    return _projection_from_terminal_logs(
        horizon=horizon,
        sessions=sessions,
        terminal_logs=terminal_logs,
        zero_drift_logs=zero_drift_logs,
        target_close=target_close,
        quantiles=quantiles,
        quantile_method=quantile_method,
        return_places=return_places,
        price_places=price_places,
    )


def _projection_from_terminal_logs(
    *,
    horizon: str,
    sessions: int,
    terminal_logs: npt.NDArray[np.float64],
    zero_drift_logs: npt.NDArray[np.float64],
    target_close: float,
    quantiles: tuple[float, float, float],
    quantile_method: Literal["linear"],
    return_places: int,
    price_places: int,
) -> HorizonProjection:
    if (
        terminal_logs.ndim != 1
        or zero_drift_logs.shape != terminal_logs.shape
        or not np.all(np.isfinite(terminal_logs))
        or not np.all(np.isfinite(zero_drift_logs))
    ):
        return _withheld_projection(
            horizon, sessions, quantiles, "projection_terminal_logs_invalid"
        )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        returns = np.expm1(terminal_logs)
        zero_returns = np.expm1(zero_drift_logs)
    if not np.all(np.isfinite(returns)) or not np.all(np.isfinite(zero_returns)):
        return _withheld_projection(horizon, sessions, quantiles, "projection_return_nonfinite")
    raw_return = _raw_quantile_triplet(returns, quantiles, quantile_method)
    raw_zero_return = _raw_quantile_triplet(zero_returns, quantiles, quantile_method)
    raw_price = _price_triplet(target_close, raw_return)
    raw_zero_price = _price_triplet(target_close, raw_zero_return)
    if not all(
        _ordered_finite_triplet(value)
        for value in (raw_return, raw_zero_return, raw_price, raw_zero_price)
    ):
        return _withheld_projection(horizon, sessions, quantiles, "projection_triplet_invalid")
    if raw_return.lower <= -1 or raw_zero_return.lower <= -1:
        return _withheld_projection(
            horizon, sessions, quantiles, "projection_return_unrepresentable"
        )
    ledger_return = _round_triplet(raw_return, places=return_places, kind="return")
    ledger_zero_return = _round_triplet(raw_zero_return, places=return_places, kind="return")
    ledger_price = _round_triplet(raw_price, places=price_places, kind="price")
    ledger_zero_price = _round_triplet(raw_zero_price, places=price_places, kind="price")
    if any(
        value is None
        for value in (
            ledger_return,
            ledger_zero_return,
            ledger_price,
            ledger_zero_price,
        )
    ):
        return _withheld_projection(
            horizon, sessions, quantiles, "projection_database_unrepresentable"
        )
    assert ledger_return is not None
    assert ledger_zero_return is not None
    assert ledger_price is not None
    assert ledger_zero_price is not None
    if not all(
        _ordered_decimal_triplet(value)
        for value in (
            ledger_return,
            ledger_zero_return,
            ledger_price,
            ledger_zero_price,
        )
    ):
        return _withheld_projection(
            horizon, sessions, quantiles, "projection_rounded_triplet_unordered"
        )
    return HorizonProjection(
        horizon=horizon,
        sessions=sessions,
        quantile_levels=quantiles,
        central_model_mass=float(Decimal(str(quantiles[2])) - Decimal(str(quantiles[0]))),
        raw_returns=raw_return,
        ledger_returns=ledger_return,
        raw_prices=raw_price,
        ledger_prices=ledger_price,
        zero_drift_raw_returns=raw_zero_return,
        zero_drift_ledger_returns=ledger_zero_return,
        zero_drift_raw_prices=raw_zero_price,
        zero_drift_ledger_prices=ledger_zero_price,
        insufficiency_reason=None,
    )


def _withheld_forecast(config: PriceProductConfig, *, seed: int, reason: str) -> FHSForecast:
    return FHSForecast(
        method_version=config.simulation.method_version,
        path_count=config.simulation.production_paths,
        seed=seed,
        mean_log_return=None,
        population_variance=None,
        terminal_variance=None,
        residual_count=0,
        residual_mean=None,
        residual_second_moment=None,
        quantile_method=config.simulation.quantile_method,
        projections=tuple(
            _withheld_projection(name, sessions, config.simulation.quantiles, reason)
            for name, sessions in config.simulation.horizons
        ),
        insufficiency_reason=reason,
    )


def _withheld_projection(
    horizon: str,
    sessions: int,
    quantiles: tuple[float, float, float],
    reason: str,
) -> HorizonProjection:
    return HorizonProjection(
        horizon=horizon,
        sessions=sessions,
        quantile_levels=quantiles,
        central_model_mass=float(Decimal(str(quantiles[2])) - Decimal(str(quantiles[0]))),
        raw_returns=None,
        ledger_returns=None,
        raw_prices=None,
        ledger_prices=None,
        zero_drift_raw_returns=None,
        zero_drift_ledger_returns=None,
        zero_drift_raw_prices=None,
        zero_drift_ledger_prices=None,
        insufficiency_reason=reason,
    )


def _raw_quantile_triplet(
    values: npt.NDArray[np.float64],
    quantiles: tuple[float, float, float],
    method: Literal["linear"],
) -> RawTriplet:
    result = np.quantile(values, quantiles, method=method)
    return RawTriplet(*(float(value) for value in result))


def _price_triplet(target_close: float, returns: RawTriplet) -> RawTriplet:
    return RawTriplet(
        target_close * (1 + returns.lower),
        target_close * (1 + returns.median),
        target_close * (1 + returns.upper),
    )


def _round_triplet(
    triplet: RawTriplet, *, places: int, kind: Literal["return", "price"]
) -> LedgerTriplet | None:
    values = (
        _round_ledger(triplet.lower, places=places, kind=kind),
        _round_ledger(triplet.median, places=places, kind=kind),
        _round_ledger(triplet.upper, places=places, kind=kind),
    )
    if any(value is None for value in values):
        return None
    lower, median, upper = values
    assert lower is not None and median is not None and upper is not None
    return LedgerTriplet(lower, median, upper)


def _round_ledger(value: float, *, places: int, kind: Literal["return", "price"]) -> Decimal | None:
    if not math.isfinite(value):
        return None
    try:
        with localcontext() as context:
            context.prec = 64
            context.rounding = ROUND_HALF_EVEN
            rounded = Decimal(str(value)).quantize(
                Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN
            )
    except (InvalidOperation, ValueError):
        return None
    limit = _MAX_LEDGER_RETURN if kind == "return" else _MAX_LEDGER_PRICE
    if abs(rounded) > limit or (kind == "price" and rounded <= 0):
        return None
    return rounded


def _maximum_drawdown(closes: tuple[float, ...]) -> float:
    values = _finite_positive_array(closes, "drawdown_close_invalid")
    peaks = np.maximum.accumulate(values)
    drawdowns = values / peaks - 1.0
    result = float(np.min(drawdowns))
    if not math.isfinite(result):
        raise PriceProductInputError("drawdown_nonfinite", "Maximum drawdown is non-finite")
    return result


def _safe_log_ratio(start: float, end: float, *, reason_code: str) -> float:
    values = _finite_positive_array((start, end), reason_code)
    result = float(math.log(values[1] / values[0]))
    if not math.isfinite(result):
        raise PriceProductInputError(reason_code, "Log-return endpoint is non-finite")
    return result


def _finite_positive_array(values: tuple[float, ...], reason_code: str) -> npt.NDArray[np.float64]:
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise PriceProductInputError(
                reason_code, "Price inputs must be finite and strictly positive"
            )
    return np.asarray(values, dtype=np.float64)


def _canonical_float(value: float) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PriceProductInputError(
            "input_hash_nonfinite", "Canonical input values must be finite"
        )
    return float(value).hex()


def _ordered_finite_triplet(value: RawTriplet) -> bool:
    return (
        all(math.isfinite(item) for item in (value.lower, value.median, value.upper))
        and value.lower <= value.median <= value.upper
    )


def _ordered_decimal_triplet(value: LedgerTriplet) -> bool:
    return value.lower <= value.median <= value.upper


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)

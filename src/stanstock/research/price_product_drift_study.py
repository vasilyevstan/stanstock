"""Single-case, synthetic-only drift sensitivity; not a production forecast.

The noise-matched plug-in assumptions m=mu/2 and u2=v/(2*756) use variance
estimated from the same training window. They are neither a calibrated
posterior nor an optimal shrinkage rule. The motivating independent
normal-means construction ignores serial dependence. Simulated paths are not
independent market observations, and synthetic scores establish no skill.

All models retain the frozen historical filter and innovation recursion.
Only cumulative terminal drift changes: this says nothing about equivalence
of interim drawdown risk. Units are native USD cumulative split-adjusted
price returns, excluding dividends, FX, taxes and fees.

No source loading, persistence, clock, provider, or outcome-dependent
generation occurs here. Synthetic metadata is admission, not authentication
of provenance; the caller must actually construct synthetic prices.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from uuid import UUID

import numpy as np
import numpy.typing as npt

from stanstock.research.price_product import (
    HorizonProjection,
    PriceProductInput,
    RawTriplet,
    SourceExecutionBinding,
    calculate_price_product,
    complete_input_hash,
    deterministic_seed,
    filter_historical_returns,
    projection_from_terminal_logs,
    simulate_fhs_terminal_logs,
)
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PriceProductConfig,
    price_product_config_hash,
)
from stanstock.research.price_product_frequencies import (
    FLAT_UPPER_LOG_THRESHOLD,
    LOSS_LOG_THRESHOLD,
    BucketCounts,
    classify_terminal_log_returns,
)

STUDY_VERSION = "synthetic-fhs-drift-study-v1"
MODEL_IDS = (
    "historical_fhs_control",
    "zero_log_drift_fhs_control",
    "plugin_shrinkage",
    "plugin_shrinkage_uncertainty",
)
_PATH_COUNT = 8192
_ASSUMPTIONS_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "engine_config_hash": (
                "55334183af29fc01b853e83f9bf75216f24564956925fb95912cb80a69420867"
            ),
            "m": "mu/2",
            "mode": "plug_in_sensitivity",
            "parameter_draw": "PCG64.standard_normal(float64,8192)",
            "parameter_scope": "one_per_path_shared_across_horizons",
            "study_version": "synthetic-fhs-drift-study-v1",
            "u2": "v/(2*756)",
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
).hexdigest()


@dataclass(frozen=True, slots=True)
class SyntheticDriftHorizon:
    model_id: str
    horizon: str
    sessions: int
    insufficiency_reason: str | None
    # Exactly loss, flat_to_20, above_20; never a fourth large-loss category.
    event_counts: tuple[int, int, int] | None = None
    raw_returns: RawTriplet | None = None
    empirical_log_variance: float | None = None
    # Population moments over paths, not calibrated uncertainty estimates.
    theoretical_h2_u2: float | None = None
    sample_h2_theta_variance: float | None = None
    sample_2h_log_zero_theta_covariance: float | None = None

    @property
    def available(self) -> bool:
        return self.insufficiency_reason is None and self.event_counts is not None


@dataclass(frozen=True, slots=True)
class SyntheticDriftExperiment:
    study_version: str
    assumptions_sha256: str
    listing_id: UUID
    target_date: date
    input_hash: str
    innovation_seed: int
    parameter_seed: int
    m: float | None
    u2: float | None
    path_count: int
    # Model-major, then the frozen 6m/12m/3y/5y horizon order.
    horizons: tuple[SyntheticDriftHorizon, ...]


@dataclass(frozen=True, slots=True)
class SyntheticDriftScore:
    brier_score: float | None
    median_absolute_error: float | None
    interval_score: float | None
    insufficiency_reason: str | None


def project_synthetic_drift_experiment(
    product_input: PriceProductInput, *, config: PriceProductConfig
) -> SyntheticDriftExperiment:
    """Generate all four fixed models before any synthetic label is supplied."""
    if price_product_config_hash(config) != PRODUCT_EFFECTIVE_CONFIG_HASH:
        raise ValueError("Synthetic drift study requires the exact frozen price product config")
    if product_input.source_execution != SourceExecutionBinding("synthetic_demo", "research"):
        raise ValueError("Synthetic drift study requires synthetic_demo research inputs")
    if not all(
        isinstance(value, UUID)
        for value in (
            product_input.listing_id,
            product_input.stock.identity.asset_id,
            product_input.benchmark.identity.asset_id,
        )
    ):
        raise ValueError("Synthetic drift study requires permanent UUID identities")

    # Full native admission AND withholding, including the natural control
    # pair's price/ledger representability. Candidates cannot rescue it.
    native = calculate_price_product(product_input, config=config)
    input_hash = complete_input_hash(product_input)
    innovation_seed = deterministic_seed(
        method_version=FHS_METHOD_VERSION,
        effective_config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
        listing_id=product_input.listing_id,
        target_date=product_input.target_date,
    )
    parameter_material = "\n".join(
        (
            "stanstock-synthetic-drift-parameter-v1",
            _ASSUMPTIONS_SHA256,
            str(product_input.listing_id),
            product_input.target_date.isoformat(),
            input_hash,
        )
    ).encode("ascii")
    parameter_seed = int.from_bytes(hashlib.sha256(parameter_material).digest()[:16], "big")
    forecast = native.forecast
    m = None if forecast.mean_log_return is None else forecast.mean_log_return / 2.0
    u2 = (
        None
        if forecast.population_variance is None
        else forecast.population_variance / (2.0 * 756.0)
    )
    records: list[SyntheticDriftHorizon] = []
    if any(projection.insufficiency_reason is None for projection in forecast.projections):
        simulation = config.simulation
        filtered = filter_historical_returns(
            product_input.stock.closes,
            burn_in=simulation.filter_burn_in,
            variance_target_weight=simulation.variance_target_weight,
            variance_persistence=simulation.variance_persistence,
            innovation_weight=simulation.innovation_weight,
        )
        terminals = simulate_fhs_terminal_logs(
            filtered,
            seed=innovation_seed,
            horizons=tuple(sessions for _name, sessions in simulation.horizons),
            path_count=_PATH_COUNT,
            diagnostic_max_paths=simulation.diagnostic_max_paths,
            variance_target_weight=simulation.variance_target_weight,
            variance_persistence=simulation.variance_persistence,
            innovation_weight=simulation.innovation_weight,
        )
        assert m is not None and u2 is not None
        # Separate domain/PRNG. One unmodified draw per innovation path, shared
        # across horizons; no daily redraw or finite-sample normalization.
        z = np.random.Generator(np.random.PCG64(parameter_seed)).standard_normal(
            _PATH_COUNT, dtype=np.float64
        )
        theta = m + math.sqrt(u2) * z
        for index, native_projection in enumerate(forecast.projections):
            if native_projection.insufficiency_reason is not None:
                records.extend(_withheld_models(native_projection))
                continue
            historical = terminals.with_drift[index]
            zero = terminals.zero_drift[index]
            sessions = native_projection.sessions
            with np.errstate(over="ignore", invalid="ignore"):
                candidate_logs = (zero + sessions * m, zero + sessions * theta)
            for model_id, logs, companion in (
                (MODEL_IDS[0], historical, zero),
                (MODEL_IDS[1], zero, historical),
                (MODEL_IDS[2], candidate_logs[0], candidate_logs[0]),
                (MODEL_IDS[3], candidate_logs[1], candidate_logs[1]),
            ):
                records.append(
                    _summarize(
                        model_id=model_id,
                        native_projection=native_projection,
                        logs=logs,
                        companion=companion,
                        target_close=product_input.stock.closes[-1],
                        config=config,
                        zero=zero,
                        theta=theta if model_id == MODEL_IDS[3] else None,
                        u2=u2,
                    )
                )
    else:
        for projection in forecast.projections:
            records.extend(_withheld_models(projection))
    return SyntheticDriftExperiment(
        study_version=STUDY_VERSION,
        assumptions_sha256=_ASSUMPTIONS_SHA256,
        listing_id=product_input.listing_id,
        target_date=product_input.target_date,
        input_hash=input_hash,
        innovation_seed=innovation_seed,
        parameter_seed=parameter_seed,
        m=m,
        u2=u2,
        path_count=_PATH_COUNT,
        horizons=tuple(
            record for model in MODEL_IDS for record in records if record.model_id == model
        ),
    )


def _withheld_models(projection: HorizonProjection) -> tuple[SyntheticDriftHorizon, ...]:
    return tuple(
        SyntheticDriftHorizon(
            model, projection.horizon, projection.sessions, projection.insufficiency_reason
        )
        for model in MODEL_IDS
    )


def _summarize(
    *,
    model_id: str,
    native_projection: HorizonProjection,
    logs: npt.NDArray[np.float64],
    companion: npt.NDArray[np.float64],
    target_close: float,
    config: PriceProductConfig,
    zero: npt.NDArray[np.float64],
    theta: npt.NDArray[np.float64] | None,
    u2: float,
) -> SyntheticDriftHorizon:
    horizon, sessions = native_projection.horizon, native_projection.sessions
    projection = projection_from_terminal_logs(
        horizon=horizon,
        sessions=sessions,
        terminal_logs=logs,
        zero_drift_logs=companion,
        target_close=target_close,
        quantiles=config.simulation.quantiles,
        quantile_method=config.simulation.quantile_method,
        return_places=config.rounding.return_decimal_places,
        price_places=config.rounding.price_decimal_places,
    )
    frequencies = classify_terminal_log_returns(
        logs,
        horizon=horizon,
        sessions=sessions,
        path_count=_PATH_COUNT,
        zero_drift_logs=companion,
        insufficiency_reason=projection.insufficiency_reason,
    )
    if frequencies.insufficiency_reason is not None:
        return SyntheticDriftHorizon(model_id, horizon, sessions, frequencies.insufficiency_reason)
    counts: BucketCounts | None = frequencies.counts
    assert counts is not None and counts.total == _PATH_COUNT
    with np.errstate(over="ignore", invalid="ignore"):
        variance = float(np.var(logs, ddof=0))
        theory = None if theta is None else sessions**2 * u2
        sampled = None if theta is None else sessions**2 * float(np.var(theta, ddof=0))
        cross = (
            None
            if theta is None
            else 2 * sessions * float(np.mean((zero - np.mean(zero)) * (theta - np.mean(theta))))
        )
    if not all(
        math.isfinite(value) for value in (variance, theory, sampled, cross) if value is not None
    ):
        return SyntheticDriftHorizon(model_id, horizon, sessions, "simulation_nonfinite")
    return SyntheticDriftHorizon(
        model_id=model_id,
        horizon=horizon,
        sessions=sessions,
        insufficiency_reason=None,
        event_counts=(counts.loss, counts.flat_to_20, counts.above_20),
        raw_returns=projection.raw_returns,
        empirical_log_variance=variance,
        theoretical_h2_u2=theory,
        sample_h2_theta_variance=sampled,
        sample_2h_log_zero_theta_covariance=cross,
    )


def score_synthetic_drift_outcome(
    horizon_result: SyntheticDriftHorizon,
    *,
    synthetic_terminal_log_return: float | None,
) -> SyntheticDriftScore:
    """Score one explicitly hypothetical label, without changing its forecast.

    Primary: unnormalized three-class Brier (fixed denominator 8192).
    Secondary: raw median absolute error and central-60% interval score.
    A bad label withholds all scores; it never changes generation or seeds.
    """
    reason = horizon_result.insufficiency_reason
    y_log = synthetic_terminal_log_return
    # Normalize only the supplied scalar, never the simulation paths. Even an
    # integer too large for float64 is a bad label, not a forecast failure.
    if isinstance(y_log, (int, float)) and not isinstance(y_log, bool):
        try:
            y_log = float(y_log)
        except OverflowError:
            return SyntheticDriftScore(None, None, None, "synthetic_outcome_unrepresentable")
    if y_log is None:
        reason = "synthetic_outcome_missing"
    elif isinstance(y_log, bool) or not isinstance(y_log, (int, float)):
        reason = "synthetic_outcome_invalid"
    elif not math.isfinite(y_log):
        reason = "synthetic_outcome_nonfinite"
    else:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            y = float(np.expm1(y_log))
        if not math.isfinite(y) or y <= -1:
            reason = "synthetic_outcome_unrepresentable"
        elif reason is None:
            counts = horizon_result.event_counts
            quantiles = horizon_result.raw_returns
            if (
                counts is None
                or quantiles is None
                or len(counts) != 3
                or any(type(count) is not int or count < 0 for count in counts)
                or sum(counts) != _PATH_COUNT
            ):
                reason = "synthetic_forecast_unavailable"
            else:
                event = (
                    0
                    if y_log < LOSS_LOG_THRESHOLD
                    else 1
                    if y_log <= FLAT_UPPER_LOG_THRESHOLD
                    else 2
                )
                brier = sum(
                    (count / _PATH_COUNT - int(index == event)) ** 2
                    for index, count in enumerate(counts)
                )
                mae = abs(quantiles.median - y)
                interval = (
                    quantiles.upper
                    - quantiles.lower
                    + 5.0 * max(quantiles.lower - y, 0.0)
                    + 5.0 * max(y - quantiles.upper, 0.0)
                )
                if all(math.isfinite(value) for value in (brier, mae, interval)):
                    return SyntheticDriftScore(brier, mae, interval, None)
                reason = "synthetic_score_unrepresentable"
    return SyntheticDriftScore(None, None, None, reason)

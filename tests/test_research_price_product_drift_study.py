"""Synthetic analytical proofs and strict current-base frozen-engine evidence."""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import io
import json
import math
import os
import socket
import time
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
from exchange_calendars import get_calendar

import stanstock.research.price_product as engine
import stanstock.research.price_product_drift_study as study
import test_research_product_frozen_probability_contract as frozen
from stanstock.research.price_product import (
    PriceInputIdentity,
    PriceProductInput,
    PriceProductInputError,
    PriceSeries,
    RawTriplet,
    SourceExecutionBinding,
)
from stanstock.research.price_product_config import (
    load_price_product_config,
    price_product_config_hash,
)
from stanstock.research.price_product_frequencies import (
    FLAT_UPPER_LOG_THRESHOLD,
    classify_terminal_log_returns,
)

BASE_SHA = "7764dc2fa26357c88201dad2b4084beadc973d63"
ASSUMPTIONS_GOLDEN = "59f811e6ee4bcc3bfce3df4576cec1f68fc99ee2d811c8afef6e7424ba617df9"
CONFIG_GOLDEN = "55334183af29fc01b853e83f9bf75216f24564956925fb95912cb80a69420867"
INPUT_GOLDEN = "72a84d8ee9fdae2ee760490a26e819b7349767804206270eba50f9dd9c3bcd62"
INNOVATION_SEED = 60944437333149738280501899696230699917
PARAMETER_SEED = 167827220968017466954546915492007012406
MODELS = (
    "historical_fhs_control",
    "zero_log_drift_fhs_control",
    "plugin_shrinkage",
    "plugin_shrinkage_uncertainty",
)
HORIZONS = (("6m", 126), ("12m", 252), ("3y", 756), ("5y", 1260))


@pytest.fixture(scope="module")
def config():
    return load_price_product_config()


@pytest.fixture(scope="module")
def product_input():
    """Independent synthetic identity/calendar and exactly representable prices.

    Dyadic closes make the input hash portable, unlike a transcendental price
    generator. Identity checksums are fixture labels, not claims of assets.
    The benchmark is synthetic too; no provider prices are relabelled.
    """
    calendar = get_calendar("XNYS", start="2023-01-01", end="2026-09-11")
    sessions = tuple(session.date() for session in calendar.sessions_window("2026-09-11", -757))
    admitted = datetime(2026, 9, 11, 21, tzinfo=UTC)
    stock = PriceSeries(
        identity=PriceInputIdentity(
            UUID("11111111-1111-4111-8111-111111111111"),
            "synthetic_demo",
            "SYN-DRIFT",
            "a" * 64,
            admitted,
            admitted,
        ),
        currency="USD",
        dates=sessions,
        closes=tuple(100.0 + i / 128.0 + ((i * 17) % 29) / 32.0 for i in range(757)),
    )
    benchmark = PriceSeries(
        identity=PriceInputIdentity(
            UUID("22222222-2222-4222-8222-222222222222"),
            "synthetic_demo",
            "SPY",
            "b" * 64,
            admitted,
            admitted,
        ),
        currency="USD",
        dates=sessions,
        closes=tuple(200.0 + i / 256.0 + ((i * 11) % 23) / 64.0 for i in range(757)),
    )
    return PriceProductInput(
        listing_id=UUID("aaaaaaaa-1111-4222-8333-bbbbbbbbbbbb"),
        target_date=date(2026, 9, 11),
        decision_time=datetime(2026, 9, 11, 22, tzinfo=UTC),
        calendar_sessions=sessions,
        stock=stock,
        benchmark=benchmark,
        source_execution=SourceExecutionBinding("synthetic_demo", "research"),
    )


@pytest.fixture(scope="module")
def evidence(product_input, config):
    """Observe the actual private arrays without extending the public API."""
    captures = {}
    simulations = []
    summarize = study._summarize
    simulate = study.simulate_fhs_terminal_logs

    def capture_summary(**kwargs):
        captures[kwargs["model_id"], kwargs["native_projection"].horizon] = kwargs
        return summarize(**kwargs)

    def capture_simulation(filtered, **kwargs):
        result = simulate(filtered, **kwargs)
        simulations.append((filtered, kwargs, result))
        return result

    native = engine.calculate_price_product(product_input, config=config)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "_summarize", capture_summary)
        patch.setattr(study, "simulate_fhs_terminal_logs", capture_simulation)
        result = study.project_synthetic_drift_experiment(product_input, config=config)
    assert len(simulations) == 1
    filtered, arguments, terminals = simulations[0]
    reference = engine.simulate_fhs_terminal_logs(
        engine.filter_historical_returns(product_input.stock.closes),
        seed=INNOVATION_SEED,
        horizons=(126, 252, 756, 1260),
        path_count=8192,
    )
    return result, native, filtered, arguments, terminals, reference, captures


def test_independent_literal_assumptions_fixture_config_and_seeds(product_input, config, evidence):
    literal = {
        "engine_config_hash": CONFIG_GOLDEN,
        "m": "mu/2",
        "mode": "plug_in_sensitivity",
        "parameter_draw": "PCG64.standard_normal(float64,8192)",
        "parameter_scope": "one_per_path_shared_across_horizons",
        "study_version": "synthetic-fhs-drift-study-v1",
        "u2": "v/(2*756)",
    }
    digest = hashlib.sha256(
        json.dumps(
            literal, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    ).hexdigest()
    assert digest == ASSUMPTIONS_GOLDEN
    result = evidence[0]
    assert result.assumptions_sha256 == ASSUMPTIONS_GOLDEN
    assert result.study_version == "synthetic-fhs-drift-study-v1"
    assert study.MODEL_IDS == MODELS
    assert result.listing_id == UUID("aaaaaaaa-1111-4222-8333-bbbbbbbbbbbb")
    assert result.target_date == date(2026, 9, 11)
    assert product_input.calendar_sessions[0] == date(2023, 9, 6)
    assert len(product_input.calendar_sessions) == 757
    assert (
        hashlib.sha256(
            "\n".join(d.isoformat() for d in product_input.calendar_sessions).encode("ascii")
        ).hexdigest()
        == "e92a988d222691ae2d5fc8dabe77b7f3ed2fc01446cf699b721128a22de55673"
    )
    assert engine.complete_input_hash(product_input) == INPUT_GOLDEN == result.input_hash
    assert price_product_config_hash(config) == CONFIG_GOLDEN
    innovation_material = (
        "stanstock-research-product-seed-v1\n"
        "method_version=us-price-fhs-v1\n"
        f"effective_config_hash={CONFIG_GOLDEN}\n"
        "listing_uuid=aaaaaaaa-1111-4222-8333-bbbbbbbbbbbb\n"
        "target_date=2026-09-11\n"
    )
    expected = int.from_bytes(hashlib.sha256(innovation_material.encode()).digest()[:16], "big")
    assert result.innovation_seed == INNOVATION_SEED == expected
    assert result.parameter_seed == PARAMETER_SEED == _parameter_seed(INPUT_GOLDEN)
    assert result.parameter_seed != result.innovation_seed


def _parameter_seed(input_hash):
    material = "\n".join(
        (
            "stanstock-synthetic-drift-parameter-v1",
            ASSUMPTIONS_GOLDEN,
            "aaaaaaaa-1111-4222-8333-bbbbbbbbbbbb",
            "2026-09-11",
            input_hash,
        )
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def test_actual_controls_are_retained_bitwise_not_reconstructed_or_quantile_proxies(evidence):
    result, native, filtered, arguments, terminals, reference, captures = evidence
    assert result.path_count == arguments["path_count"] == terminals.path_count == 8192
    assert arguments["seed"] == INNOVATION_SEED == native.forecast.seed
    assert arguments["horizons"] == (126, 252, 756, 1260)
    assert (
        arguments["variance_target_weight"],
        arguments["variance_persistence"],
        arguments["innovation_weight"],
    ) == (0.01, 0.94, 0.05)
    assert native.forecast.mean_log_return == filtered.mean_log_return
    assert native.forecast.population_variance == filtered.population_variance
    assert native.forecast.terminal_variance == filtered.terminal_variance
    assert native.forecast.residual_count == len(filtered.residuals) == 504
    for i, (horizon, _sessions) in enumerate(HORIZONS):
        historical = captures[MODELS[0], horizon]["logs"]
        zero = captures[MODELS[1], horizon]["logs"]
        assert historical is terminals.with_drift[i]
        assert zero is terminals.zero_drift[i]
        assert historical.tobytes() == reference.with_drift[i].tobytes()
        assert zero.tobytes() == reference.zero_drift[i].tobytes()
        assert captures[MODELS[0], horizon]["companion"] is zero
        assert captures[MODELS[1], horizon]["companion"] is historical
        assert historical.dtype == zero.dtype == np.float64
        assert historical.shape == zero.shape == (8192,)
        rows = {r.model_id: r for r in result.horizons if r.horizon == horizon}
        assert rows[MODELS[0]].raw_returns == native.forecast.projections[i].raw_returns
        assert rows[MODELS[1]].raw_returns == native.forecast.projections[i].zero_drift_raw_returns


def test_parameter_stream_exact_unmodified_persistent_and_path_paired(evidence):
    result, _native, filtered, _arguments, terminals, _reference, captures = evidence
    returns_mean = filtered.mean_log_return
    assert result.m == returns_mean / 2.0
    assert result.u2 == filtered.population_variance / (2.0 * 756.0)
    z = np.random.Generator(np.random.PCG64(PARAMETER_SEED)).standard_normal(8192, dtype=np.float64)
    theta = result.m + math.sqrt(result.u2) * z
    assert abs(float(np.mean(z))) > 1e-5  # Must NOT recenter a finite sample.
    assert abs(float(np.var(z)) - 1.0) > 1e-5  # Nor rescale it.
    shared_theta = captures[MODELS[3], "6m"]["theta"]
    assert shared_theta.tobytes() == theta.tobytes()
    for i, (horizon, sessions) in enumerate(HORIZONS):
        zero = terminals.zero_drift[i]
        shrinkage = captures[MODELS[2], horizon]
        uncertain = captures[MODELS[3], horizon]
        assert uncertain["theta"] is shared_theta
        assert shrinkage["logs"].tobytes() == (zero + sessions * result.m).tobytes()
        assert uncertain["logs"].tobytes() == (zero + sessions * theta).tobytes()
        assert shrinkage["companion"] is shrinkage["logs"]
        assert uncertain["companion"] is uncertain["logs"]
        np.testing.assert_allclose(
            (uncertain["logs"] - zero) / sessions, theta, rtol=1e-11, atol=1e-18
        )
        assert not np.array_equal(uncertain["logs"], zero + sessions * theta[::-1])


def test_every_count_quantile_and_empirical_variance_comes_from_full_engine_paths(evidence):
    result, _native, _filtered, _arguments, _terminals, _reference, captures = evidence
    assert [(r.model_id, r.horizon, r.sessions) for r in result.horizons] == [
        (model, horizon, sessions) for model in MODELS for horizon, sessions in HORIZONS
    ]
    for record in result.horizons:
        logs = captures[record.model_id, record.horizon]["logs"]
        expected_counts = (
            int(np.count_nonzero(logs < 0.0)),
            int(np.count_nonzero((logs >= 0.0) & (logs <= FLAT_UPPER_LOG_THRESHOLD))),
            int(np.count_nonzero(logs > FLAT_UPPER_LOG_THRESHOLD)),
        )
        assert record.available and record.insufficiency_reason is None
        assert record.event_counts == expected_counts
        assert len(record.event_counts) == 3 and sum(record.event_counts) == 8192
        assert all(type(n) is int for n in record.event_counts)
        expected_quantiles = np.quantile(np.expm1(logs), (0.2, 0.5, 0.8), method="linear")
        assert record.raw_returns == RawTriplet(*expected_quantiles)
        assert record.empirical_log_variance == float(np.var(logs, ddof=0))
        # Nonlinear interpolation is detectably different on this fixture.
        wrong_quantiles = np.expm1(np.quantile(logs, (0.2, 0.5, 0.8), method="linear"))
        assert not np.array_equal(expected_quantiles, wrong_quantiles)


def test_sample_covariance_identity_is_not_theoretical_measured_increase(evidence):
    result, _native, _filtered, _arguments, terminals, _reference, captures = evidence
    for record in result.horizons:
        if record.model_id != MODELS[3]:
            assert record.theoretical_h2_u2 is None
            assert record.sample_h2_theta_variance is None
            assert record.sample_2h_log_zero_theta_covariance is None
            continue
        index = [h for h, _ in HORIZONS].index(record.horizon)
        zero = terminals.zero_drift[index]
        theta = captures[record.model_id, record.horizon]["theta"]
        h = record.sessions
        # Independently computed population covariance via np.cov.
        covariance = float(np.cov(zero, theta, ddof=0)[0, 1])
        sampled = h * h * float(np.var(theta, ddof=0))
        cross = 2 * h * covariance
        assert abs(cross) > 1e-8  # The finite-sample cross term is genuinely nonzero.
        assert record.theoretical_h2_u2 == h * h * result.u2
        assert record.sample_h2_theta_variance == sampled
        assert record.sample_2h_log_zero_theta_covariance == pytest.approx(cross, rel=1e-13)
        assert record.empirical_log_variance == pytest.approx(
            float(np.var(zero)) + sampled + cross, rel=2e-14, abs=1e-17
        )
        measured_increase = record.empirical_log_variance - float(np.var(zero))
        assert abs(measured_increase - record.theoretical_h2_u2) > 1e-8
        assert abs(sampled - record.theoretical_h2_u2) > 1e-8
        # Re-pairing leaves marginal theta variance unchanged, not the cross term.
        wrong_cross = 2 * h * float(np.cov(zero, theta[::-1], ddof=0)[0, 1])
        assert abs(cross - wrong_cross) > 1e-8


def test_same_shock_drift_quantile_identity_is_tolerance_aware(evidence):
    result, _native, filtered, _arguments, terminals, _reference, _captures = evidence
    for i, (horizon, sessions) in enumerate(HORIZONS):
        drift = sessions * filtered.mean_log_return
        np.testing.assert_allclose(
            terminals.zero_drift[i] + drift, terminals.with_drift[i], rtol=1e-14, atol=1e-16
        )
        rows = {r.model_id: r for r in result.horizons if r.horizon == horizon}
        for field in ("lower", "median", "upper"):
            assert 1 + getattr(rows[MODELS[0]].raw_returns, field) == pytest.approx(
                math.exp(drift) * (1 + getattr(rows[MODELS[1]].raw_returns, field)), rel=1e-14
            )


def test_zero_mean_log_return_does_not_force_zero_median_or_arithmetic_mean():
    logs = np.tile(np.asarray([-3.0, 1.0, 1.0, 1.0]), 2048)
    assert float(np.mean(logs)) == 0.0
    returns = np.expm1(logs)
    assert float(np.median(returns)) == pytest.approx(math.expm1(1.0))
    assert float(np.mean(returns)) > 0.0


def test_equal_cumulative_deterministic_drift_is_only_terminal_equivalence():
    shocks = np.asarray([[-0.125, 0.125], [0.25, -0.125]])
    constant = np.asarray([0.0, 0.0])
    fading = np.asarray([0.5, -0.5])
    assert float(np.sum(constant)) == float(np.sum(fading))
    first = np.cumsum(shocks + constant, axis=1)
    second = np.cumsum(shocks + fading, axis=1)
    np.testing.assert_array_equal(first[:, -1], second[:, -1])
    np.testing.assert_array_equal(np.expm1(first[:, -1]), np.expm1(second[:, -1]))

    def drawdown(log_paths):
        prices = np.exp(np.column_stack((np.zeros(2), log_paths)))
        return np.min(prices / np.maximum.accumulate(prices, axis=1) - 1, axis=1)

    assert not np.array_equal(drawdown(first), drawdown(second))


def _hand_forecast():
    return study.SyntheticDriftHorizon(
        MODELS[2], "6m", 126, None, (2048, 4096, 2048), RawTriplet(-0.1, 0.05, 0.3), 0.1
    )


@pytest.mark.parametrize(
    ("raw_outcome", "brier", "mae", "interval"),
    [
        (-0.2, 0.875, 0.25, 0.9),
        (-0.1, 0.875, 0.15, 0.4),
        (0.0, 0.375, 0.05, 0.4),
        (0.1, 0.375, 0.05, 0.4),
        (0.3, 0.875, 0.25, 0.4),
        (0.4, 0.875, 0.35, 0.9),
    ],
)
def test_hand_checked_unnormalized_three_class_brier_mae_interval(
    raw_outcome, brier, mae, interval
):
    score = study.score_synthetic_drift_outcome(
        _hand_forecast(), synthetic_terminal_log_return=math.log1p(raw_outcome)
    )
    assert score.insufficiency_reason is None
    assert score.brier_score == brier  # Not divided by 3; no -20% fourth category.
    assert score.median_absolute_error == pytest.approx(mae, abs=1e-15)
    assert score.interval_score == pytest.approx(interval, abs=1e-15)


@pytest.mark.parametrize(
    ("label", "event"),
    [
        (np.nextafter(0.0, -np.inf), 0),
        (-0.0, 1),
        (0.0, 1),
        (np.nextafter(0.0, np.inf), 1),
        (np.nextafter(FLAT_UPPER_LOG_THRESHOLD, -np.inf), 1),
        (FLAT_UPPER_LOG_THRESHOLD, 1),
        (np.nextafter(FLAT_UPPER_LOG_THRESHOLD, np.inf), 2),
    ],
)
def test_authoritative_log_event_thresholds_and_ties(label, event):
    # Use the classifier's authoritative log1p boundary, not log(1.2) or
    # a decimal approximation copied from an analytical derivation.
    logs = np.full(8192, label, dtype=np.float64)
    classified = classify_terminal_log_returns(
        logs, horizon="6m", sessions=126, path_count=8192, zero_drift_logs=logs
    )
    counts = classified.counts
    assert (counts.loss, counts.flat_to_20, counts.above_20)[event] == 8192
    record = replace(
        _hand_forecast(), event_counts=tuple(8192 if i == event else 0 for i in range(3))
    )
    score = study.score_synthetic_drift_outcome(record, synthetic_terminal_log_return=float(label))
    assert score.brier_score == 0.0


@pytest.mark.parametrize(
    ("label", "reason"),
    [
        (None, "synthetic_outcome_missing"),
        (float("nan"), "synthetic_outcome_nonfinite"),
        (float("inf"), "synthetic_outcome_nonfinite"),
        (-float("inf"), "synthetic_outcome_nonfinite"),
        (1000.0, "synthetic_outcome_unrepresentable"),
        (-1000.0, "synthetic_outcome_unrepresentable"),
        ("not-a-label", "synthetic_outcome_invalid"),
        (True, "synthetic_outcome_invalid"),
        (10**400, "synthetic_outcome_unrepresentable"),
        (709.0, "synthetic_score_unrepresentable"),
    ],
)
def test_bad_synthetic_labels_leave_forecast_intact_and_all_scores_null(label, reason, evidence):
    record = evidence[0].horizons[0]
    before = asdict(record)
    score = study.score_synthetic_drift_outcome(record, synthetic_terminal_log_return=label)
    assert score == study.SyntheticDriftScore(None, None, None, reason)
    assert asdict(record) == before


@pytest.mark.parametrize(
    ("mode", "grade"),
    [("provider", "research"), ("provider", "observed"), ("synthetic_demo", "observed")],
)
def test_rejects_non_synthetic_or_observed_before_calculation(
    mode, grade, product_input, config, monkeypatch
):
    def refused(*args, **kwargs):
        pytest.fail("Non-synthetic input reached native calculation")

    monkeypatch.setattr(study, "calculate_price_product", refused)
    altered = replace(product_input, source_execution=SourceExecutionBinding(mode, grade))
    with pytest.raises(ValueError, match="synthetic_demo research"):
        study.project_synthetic_drift_experiment(altered, config=config)


def test_exact_config_no_runtime_overrides(product_input, config):
    altered = replace(config, simulation=replace(config.simulation, production_paths=4096))
    with pytest.raises(ValueError, match="exact frozen"):
        study.project_synthetic_drift_experiment(product_input, config=altered)
    assert tuple(inspect.signature(study.project_synthetic_drift_experiment).parameters) == (
        "product_input",
        "config",
    )
    assert tuple(inspect.signature(study.score_synthetic_drift_outcome).parameters) == (
        "horizon_result",
        "synthetic_terminal_log_return",
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("provider", "stock_provider_invalid"),
        ("checksum", "stock_asset_identity_invalid"),
        ("subject", "benchmark_identity_mismatch"),
        ("currency", "stock_currency_invalid"),
        ("available", "stock_asset_after_decision"),
        ("retrieved", "stock_asset_after_decision"),
        ("future_target", "target_after_decision_time"),
        ("short_calendar", "calendar_session_count"),
        ("duplicate_calendar", "calendar_sessions_invalid"),
        ("calendar_anchor", "calendar_target_mismatch"),
        ("mismatched_dates", "price_calendar_mismatch"),
        ("future_row", "stock_target_mismatch"),
        ("bad_close", "stock_close_invalid"),
    ],
)
def test_full_native_admission_is_not_bypassed(mutation, reason, product_input, config):
    p = product_input
    identity = p.stock.identity
    future = p.decision_time + timedelta(seconds=1)
    changes = {
        "provider": lambda: replace(
            p, stock=replace(p.stock, identity=replace(identity, provider="twelve_data"))
        ),
        "checksum": lambda: replace(
            p, stock=replace(p.stock, identity=replace(identity, sha256="invalid"))
        ),
        "subject": lambda: replace(
            p,
            benchmark=replace(p.benchmark, identity=replace(p.benchmark.identity, subject="WRONG")),
        ),
        "currency": lambda: replace(p, stock=replace(p.stock, currency="EUR")),
        "available": lambda: replace(
            p, stock=replace(p.stock, identity=replace(identity, available_at=future))
        ),
        "retrieved": lambda: replace(
            p, stock=replace(p.stock, identity=replace(identity, retrieved_at=future))
        ),
        "future_target": lambda: replace(p, target_date=p.target_date + timedelta(days=1)),
        "short_calendar": lambda: replace(p, calendar_sessions=p.calendar_sessions[1:]),
        "duplicate_calendar": lambda: replace(
            p, calendar_sessions=(p.calendar_sessions[1], *p.calendar_sessions[1:])
        ),
        "calendar_anchor": lambda: replace(
            p, calendar_sessions=(*p.calendar_sessions[:-1], p.target_date + timedelta(days=1))
        ),
        "mismatched_dates": lambda: replace(
            p,
            stock=replace(
                p.stock, dates=(p.stock.dates[0] - timedelta(days=1), *p.stock.dates[1:])
            ),
            benchmark=replace(
                p.benchmark, dates=(p.stock.dates[0] - timedelta(days=1), *p.stock.dates[1:])
            ),
        ),
        "future_row": lambda: replace(
            p,
            stock=replace(p.stock, dates=(*p.stock.dates[:-1], p.target_date + timedelta(days=1))),
        ),
        "bad_close": lambda: replace(
            p, stock=replace(p.stock, closes=(float("nan"), *p.stock.closes[1:]))
        ),
    }
    with pytest.raises(PriceProductInputError) as error:
        study.project_synthetic_drift_experiment(changes[mutation](), config=config)
    assert error.value.reason_code == reason


def test_permanent_listing_and_asset_uuid_identity_required(product_input, config):
    for altered in (
        replace(product_input, listing_id="not-a-uuid"),
        replace(
            product_input,
            stock=replace(
                product_input.stock,
                identity=replace(product_input.stock.identity, asset_id="not-a-uuid"),
            ),
        ),
    ):
        with pytest.raises(ValueError, match="permanent UUID"):
            study.project_synthetic_drift_experiment(altered, config=config)


def test_genuine_filter_withholding_has_no_candidate_rescue(product_input, config, monkeypatch):
    flat = replace(product_input, stock=replace(product_input.stock, closes=(100.0,) * 757))
    native = engine.calculate_price_product(flat, config=config)
    assert native.forecast.insufficiency_reason == "filter_variance_degenerate"
    assert native.forecast.residual_count == 0

    def refused(*args, **kwargs):
        pytest.fail("An invalid native filter must not be rescued by another simulation")

    monkeypatch.setattr(study, "simulate_fhs_terminal_logs", refused)
    result = study.project_synthetic_drift_experiment(flat, config=config)
    assert result.m is result.u2 is None
    assert result.path_count == 8192 and len(result.horizons) == 16
    for record in result.horizons:
        assert record.insufficiency_reason == "filter_variance_degenerate"
        assert not record.available
        assert record.event_counts is record.raw_returns is record.empirical_log_variance is None
        assert record.theoretical_h2_u2 is record.sample_h2_theta_variance is None
        assert record.sample_2h_log_zero_theta_covariance is None
        score = study.score_synthetic_drift_outcome(record, synthetic_terminal_log_return=0.0)
        assert score == study.SyntheticDriftScore(None, None, None, "filter_variance_degenerate")


def test_genuine_native_horizon_withholding_cannot_be_rescued(product_input, config):
    # Moderate daily variation but enormous long-horizon extrapolation.
    # The native pair remains representable at 6m, not at 3y/5y.
    closes = tuple(
        float(x) for x in 100.0 * np.exp(np.arange(757) * 0.02 + np.sin(np.arange(757)) * 0.002)
    )
    altered = replace(product_input, stock=replace(product_input.stock, closes=closes))
    native = engine.calculate_price_product(altered, config=config)
    result = study.project_synthetic_drift_experiment(altered, config=config)
    reasons = {p.horizon: p.insufficiency_reason for p in native.forecast.projections}
    assert reasons["6m"] is None
    assert reasons["5y"] == "projection_database_unrepresentable"
    for record in result.horizons:
        if reasons[record.horizon] is not None:
            assert record.insufficiency_reason == reasons[record.horizon]
            assert record.event_counts is record.raw_returns is None


@pytest.mark.parametrize("model", MODELS[2:])
@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("nonfinite", "simulation_nonfinite"),
        ("overflow", "projection_return_nonfinite"),
        ("unrepresentable", "projection_database_unrepresentable"),
        ("variance_overflow", "simulation_nonfinite"),
        ("short", "frequency_terminal_logs_invalid"),
    ],
)
def test_candidate_failure_is_one_model_one_horizon_no_path_dropping(
    model, failure, reason, product_input, config, evidence, monkeypatch
):
    original = study._summarize

    def injected(**kwargs):
        if kwargs["model_id"] == model and kwargs["native_projection"].horizon == "12m":
            logs = kwargs["logs"].copy()
            if failure == "nonfinite":
                logs[0] = np.nan  # ONE invalid path cannot be silently dropped.
            elif failure == "overflow":
                logs[0] = 1000.0
            elif failure == "unrepresentable":
                logs[:] = 20.0
            elif failure == "variance_overflow":
                logs[0] = -1e308
            else:
                logs = logs[:-1]
            kwargs["logs"] = kwargs["companion"] = logs
        return original(**kwargs)

    monkeypatch.setattr(study, "_summarize", injected)
    # Reuse already-proven native/filter/simulation inputs for fault isolation.
    monkeypatch.setattr(study, "calculate_price_product", lambda *a, **k: evidence[1])
    monkeypatch.setattr(study, "simulate_fhs_terminal_logs", lambda *a, **k: evidence[4])
    result = study.project_synthetic_drift_experiment(product_input, config=config)
    for before, after in zip(evidence[0].horizons, result.horizons, strict=True):
        if (after.model_id, after.horizon) == (model, "12m"):
            assert after.insufficiency_reason == reason
            assert after.event_counts is after.raw_returns is after.empirical_log_variance is None
            assert not after.available
        else:
            assert after == before
            assert sum(after.event_counts) == result.path_count == 8192


def test_feature_hash_changes_parameter_stream_not_innovation_identity(
    product_input, config, evidence
):
    # A volume is hashed even when not used in the historical return filter.
    changed = replace(product_input, stock=replace(product_input.stock, volumes=(1234.0,) * 757))
    result = study.project_synthetic_drift_experiment(changed, config=config)
    assert result.input_hash != INPUT_GOLDEN
    assert result.parameter_seed == _parameter_seed(engine.complete_input_hash(changed))
    assert result.parameter_seed != PARAMETER_SEED
    assert result.innovation_seed == INNOVATION_SEED
    assert result.m == evidence[0].m and result.u2 == evidence[0].u2
    assert result.horizons[:12] == evidence[0].horizons[:12]
    assert result.horizons[12:] != evidence[0].horizons[12:]


def _assert_scalar_tree(value):
    if is_dataclass(value):
        assert value.__dataclass_params__.frozen
        for field in fields(value):
            _assert_scalar_tree(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            _assert_scalar_tree(item)
    else:
        assert value is None or type(value) in (str, int, float, bool, UUID, date)
        if type(value) is float:
            assert math.isfinite(value)


@pytest.mark.parametrize("counts", [(4096, 2048, 2047), (8192, 0, 0, 0), (-1, 1, 8192)])
def test_scorer_rejects_incomplete_or_non_three_class_forecasts(counts):
    score = study.score_synthetic_drift_outcome(
        replace(_hand_forecast(), event_counts=counts), synthetic_terminal_log_return=0.0
    )
    assert score == study.SyntheticDriftScore(None, None, None, "synthetic_forecast_unavailable")


def test_repeatable_no_mutation_scalar_only_and_feature_before_label(
    product_input, config, evidence, monkeypatch
):
    before = asdict(product_input)
    forecast = study.project_synthetic_drift_experiment(product_input, config=config)
    assert forecast == evidence[0]
    assert asdict(product_input) == before
    _assert_scalar_tree(forecast)
    with pytest.raises(FrozenInstanceError):
        forecast.m = 0.0

    def refused(*args, **kwargs):
        pytest.fail("Scoring must never invoke candidate generation")

    monkeypatch.setattr(study, "project_synthetic_drift_experiment", refused)
    for y in (-0.1, 0.0, 0.4):
        for row in forecast.horizons:
            score = study.score_synthetic_drift_outcome(
                row, synthetic_terminal_log_return=math.log1p(y)
            )
            assert score.insufficiency_reason is None
            _assert_scalar_tree(score)
    assert forecast == evidence[0] and asdict(product_input) == before


def test_functions_have_no_io_clock_orm_provider_or_credential_behavior(
    product_input, config, evidence
):
    # Static import closure of this module is deliberately narrow. The native
    # price operator is already pure; the config LOADER is never invoked here.
    tree = ast.parse(inspect.getsource(study))
    modules = [
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    ]
    assert set(modules) <= {
        "__future__",
        "hashlib",
        "json",
        "math",
        "dataclasses",
        "datetime",
        "uuid",
        "numpy",
        "numpy.typing",
        "stanstock.research.price_product",
        "stanstock.research.price_product_config",
        "stanstock.research.price_product_frequencies",
    }
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in ("now", "today", "utcnow", "objects")
        for node in ast.walk(tree)
    )

    def refused(*args, **kwargs):
        raise AssertionError("Synthetic functions attempted external state access")

    with pytest.MonkeyPatch.context() as patch:
        for obj, name in (
            (builtins, "open"),
            (io, "open"),
            (os, "getenv"),
            (socket.socket, "connect"),
            (socket, "create_connection"),
            (time, "time"),
            (time, "time_ns"),
        ):
            patch.setattr(obj, name, refused)
        patch.setattr(os, "environ", {})
        result = study.project_synthetic_drift_experiment(product_input, config=config)
        score = study.score_synthetic_drift_outcome(
            result.horizons[0], synthetic_terminal_log_return=0.0
        )
    assert result == evidence[0] and score.insufficiency_reason is None
    # No django_db marker: pytest-django also forbids ORM access throughout.


@pytest.fixture(scope="module")
def current_base_probe(tmp_path_factory):
    """Bind the EXISTING two-process machinery to this contract's exact base.

    Only the fixture adapter is new. No alternate harness, golden fallback,
    excluded fields, or successful skip if the required base is unavailable.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(frozen, "FROZEN_BASE_SHA", BASE_SHA)
        assert frozen._base_objects_available(), f"Required base objects unavailable: {BASE_SHA}"
        try:
            root = frozen.frozen_base_root.__wrapped__(tmp_path_factory)
        except pytest.skip.Exception:
            pytest.fail(f"Cannot execute required base objects: {BASE_SHA}")
        candidate = frozen.candidate_root.__wrapped__()
        assert candidate == Path(study.__file__).resolve().parents[2]
        yield frozen.probe.__wrapped__(tmp_path_factory, root, candidate)


def test_exact_current_base_complete_success_and_native_withheld_preservation(current_base_probe):
    base_run = current_base_probe("base", "daily_derived", "full")
    candidate_run = current_base_probe("candidate", "daily_derived", "full")
    assert base_run.succeeded, f"Exact-base probe failed (exit {base_run.returncode})"
    assert candidate_run.succeeded, f"Working-tree probe failed (exit {candidate_run.returncode})"
    base = base_run.require("exact base")
    candidate = candidate_run.require("working tree")
    frozen.assert_frozen_surface_unchanged(base, candidate, allow_additive=False)
    # Guard against vacuous equality: full successes, native withholding,
    # config identities, raw/ledger triplets and frequency nulls all exist.
    assert base["config"]["config_file_bytes_sha256"] == (
        "21dcfcb4a3560fe94a7e614bc8e659d6778312b21398cb249caf6e09df78b832"
    )
    assert base["config"]["recomputed_effective_config_hash"] == CONFIG_GOLDEN
    assert base["output"]["prediction_count"] == 20
    predictions = base["output"]["predictions"]
    for horizon, _sessions in HORIZONS:
        good = predictions[f"CHEAP|advisory|{horizon}|us-price-fhs-v1"]
        flat = predictions[f"FLAT|advisory|{horizon}|us-price-fhs-v1"]
        assert good["base_return"] is not None and good["insufficiency_reason"] == ""
        assert flat["base_return"] is None
        assert flat["insufficiency_reason"] == "filter_variance_degenerate"
        assert good["calculation"]["risk"]
        assert good["calculation"]["recommendation"]
        assert good["calculation"]["forecast"]["residual_count"] == 504
        assert flat["calculation"]["forecast"]["residual_count"] == 0
        for key in ("mean_log_return", "population_variance", "terminal_variance"):
            assert flat["calculation"]["forecast"][key] is None
        for projection in good["calculation"]["forecast"]["projections"]:
            for key in (
                "raw_returns",
                "ledger_returns",
                "raw_prices",
                "ledger_prices",
                "zero_drift_raw_returns",
                "zero_drift_ledger_returns",
                "zero_drift_raw_prices",
                "zero_drift_ledger_prices",
            ):
                assert projection[key] is not None
        for projection in flat["calculation"]["forecast"]["projections"]:
            assert projection["insufficiency_reason"] == "filter_variance_degenerate"
            assert projection["raw_returns"] is projection["zero_drift_raw_returns"] is None
    frequency_documents = [
        asset["document"]
        for key, asset in base["assets"].items()
        if key.startswith("research_product_frequency_evidence|")
    ]
    assert len(frequency_documents) == 1
    # The comparison above includes entire documents and stored-byte digests.
    # Walk solely to assert coverage; nothing is filtered from comparison.
    flat_payload = frozen._flatten(frequency_documents[0])
    assert any(path[-1:] == ("counts",) and value == "null" for path, value in flat_payload.items())
    assert any(
        path[-1:] == ("insufficiency_reason",) and value == '"filter_variance_degenerate"'
        for path, value in flat_payload.items()
    )
    assert any(path[-2:] == ("counts", "loss") for path in flat_payload)

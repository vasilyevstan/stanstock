"""Synthetic preparation proofs; no empirical forecast or adoption evidence."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

import stanstock.research.price_product as native
import stanstock.research.price_product_shadow_drift as shadow
import test_research_price_product_drift_study as old_drift_tests
import test_research_product_frozen_probability_contract as frozen
from stanstock.research.price_product import (
    LedgerTriplet,
    PriceProductInputError,
    RawTriplet,
    SourceExecutionBinding,
)
from stanstock.research.price_product_config import (
    default_price_product_config_path,
    load_price_product_config,
    price_product_config_hash,
)

BASE_SHA = "cf70f875b0fd0a6b8ab29ecd6cc2b3016f1abcb5"
CONFIG_HASH = "55334183af29fc01b853e83f9bf75216f24564956925fb95912cb80a69420867"
CONFIG_FILE_HASH = "21dcfcb4a3560fe94a7e614bc8e659d6778312b21398cb249caf6e09df78b832"
ASSUMPTIONS_HASH = "3dff36efe9fa8377cbbd2c0410db066b7d9688d5a2877d6d41b70a0e570aea6b"
INPUT_HASH = "72a84d8ee9fdae2ee760490a26e819b7349767804206270eba50f9dd9c3bcd62"
SEED = 60944437333149738280501899696230699917
ARMS = ("historical_fhs_control", "zero_log_drift_fhs_control", "fixed_half_drift_fhs")
HORIZONS = (("6m", 126), ("12m", 252), ("3y", 756), ("5y", 1260))
ROLES = (
    ("6m", "primary"),
    ("12m", "secondary_diagnostic"),
    ("3y", "evaluation_unavailable"),
    ("5y", "evaluation_unavailable"),
)
TRIPLETS = ("raw_returns", "ledger_returns", "raw_prices", "ledger_prices")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config():
    return load_price_product_config()


def _synthetic_input():
    return old_drift_tests.product_input.__wrapped__()


@pytest.fixture(scope="module")
def product_input():
    return _synthetic_input()


def _identity(value):
    """Independent test oracle, including bits/scale and every dataclass field."""
    if is_dataclass(value):
        return {field.name: _identity(getattr(value, field.name)) for field in fields(value)}
    if type(value) is tuple:
        return [_identity(item) for item in value]
    if type(value) is float:
        assert math.isfinite(value)
        return value.hex()
    if type(value) is Decimal:
        assert value.is_finite()
        return str(value)
    if type(value) in (date, datetime):
        return value.isoformat()
    if type(value) is UUID:
        return str(value)
    assert value is None or type(value) in (str, int, bool)
    return value


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


@pytest.fixture(scope="module")
def evidence(product_input, config):
    calls, projections = [], []
    simulate = shadow.simulate_fhs_terminal_logs
    project = shadow.projection_from_terminal_logs

    def capture_simulation(filtered, **kwargs):
        terminals = simulate(filtered, **kwargs)
        calls.append((filtered, kwargs, terminals))
        return terminals

    def capture_projection(**kwargs):
        result = project(**kwargs)
        projections.append((kwargs, result))
        return result

    actual = native.calculate_price_product(product_input, config=config)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(shadow, "simulate_fhs_terminal_logs", capture_simulation)
        patch.setattr(shadow, "projection_from_terminal_logs", capture_projection)
        result = shadow.project_shadow_drift(product_input, config=config)
    assert len(calls) == 1
    return result, actual, calls[0], projections


def test_literal_identity_pins_and_exact_api(config, product_input, evidence):
    document = {
        "arm_ids": ARMS,
        "candidate_failure": "native_projection_reason_withholds_entire_candidate_horizon_only",
        "empirical_protocol_status": "not_frozen",
        "evidence_label": "research_only_unscored",
        "half_operation": "terminals.zero_drift[i] + H * (mu / 2.0)",
        "horizon_roles": ROLES,
        "inherited_withholding": "native_selected_horizon_reason_all_arms_no_rescue",
        "innovation_seed": "native_stanstock-research-product-seed-v1_first_128_sha256_bits",
        "native_config_hash": CONFIG_HASH,
        "native_method_version": "us-price-fhs-v1",
        "path_count": 8192,
        "primary_guardrail_declared_only": "6m_median_mean_absolute_error",
        "primary_metric_declared_only": "6m_central_60_percent_interval_score",
        "projection": "native_linear_p20_p50_p80_with_native_zero_companion",
        "research_version": "shadow-fhs-drift-v1",
        "rounding": "native_independent_raw_return_4_price_6_decimal_half_even",
        "schema_version": "shadow-fhs-drift@1",
        "simulation": "native_PCG64_path_major_filter_coefficients_historical_mean_variance",
        "simulation_horizons": HORIZONS,
        "source_provenance_status": "caller_supplied_unverified",
    }
    result = evidence[0]
    assert hashlib.sha256(_json_bytes(document)).hexdigest() == ASSUMPTIONS_HASH
    assert result.assumptions_sha256 == ASSUMPTIONS_HASH
    assert result.native_config_hash == price_product_config_hash(config) == CONFIG_HASH
    assert hashlib.sha256(default_price_product_config_path().read_bytes()).hexdigest() == (
        CONFIG_FILE_HASH
    )
    assert native.complete_input_hash(product_input) == result.input_hash == INPUT_HASH
    seed_material = (
        "stanstock-research-product-seed-v1\n"
        "method_version=us-price-fhs-v1\n"
        f"effective_config_hash={CONFIG_HASH}\n"
        "listing_uuid=aaaaaaaa-1111-4222-8333-bbbbbbbbbbbb\n"
        "target_date=2026-09-11\n"
    )
    assert int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:16], "big") == SEED
    assert result.innovation_seed == SEED
    assert result.simulation_horizons == HORIZONS and result.horizon_roles == ROLES
    signature = inspect.signature(shadow.project_shadow_drift)
    assert tuple(signature.parameters) == ("product_input", "config")
    assert signature.parameters["config"].kind == inspect.Parameter.KEYWORD_ONLY
    assert tuple(inspect.signature(shadow.serialize_shadow_drift).parameters) == ("result",)
    assert result.evidence_label == "research_only_unscored"
    assert result.source_provenance_status == "caller_supplied_unverified"
    assert result.empirical_protocol_status == "not_frozen"


@pytest.mark.parametrize(
    "change",
    [
        lambda c: replace(c, simulation=replace(c.simulation, production_paths=4096)),
        lambda c: replace(c, risk=replace(c.risk, buy_minimum_target_close=9.0)),
        lambda c: replace(c, replay=replace(c.replay, comparators=("different",))),
        lambda c: replace(c, universe=replace(c.universe, maximum_saved_names=21)),
        lambda c: replace(c, momentum=replace(c.momentum, skip_sessions=20)),
        lambda c: replace(c, rounding=replace(c.rounding, price_decimal_places=5)),
        lambda c: replace(c, dividends_included=True),
    ],
)
def test_complete_config_admission(change, config, product_input, monkeypatch):
    monkeypatch.setattr(shadow, "calculate_price_product", _refuse)
    with pytest.raises(ValueError) as failure:
        shadow.project_shadow_drift(product_input, config=change(config))
    assert str(failure.value) == (
        "Shadow drift preparation requires the exact frozen price product config"
    )


def _refuse(*args, **kwargs):
    raise AssertionError("Forbidden dependency was invoked")


@pytest.mark.parametrize("field", ["listing", "stock", "benchmark"])
def test_uuid_objects_required(field, config, product_input):
    if field == "listing":
        changed = replace(product_input, listing_id=str(product_input.listing_id))
    else:
        series = getattr(product_input, field)
        changed = replace(
            product_input,
            **{
                field: replace(
                    series,
                    identity=replace(series.identity, asset_id=str(series.identity.asset_id)),
                )
            },
        )
    with pytest.raises(ValueError) as failure:
        shadow.project_shadow_drift(changed, config=config)
    assert str(failure.value) == "Shadow drift preparation requires permanent UUID identities"


@pytest.mark.parametrize(
    "change",
    [
        lambda p: replace(p, stock=replace(p.stock, currency="EUR")),
        lambda p: replace(
            p, stock=replace(p.stock, identity=replace(p.stock.identity, provider="wrong"))
        ),
        lambda p: replace(
            p,
            benchmark=replace(p.benchmark, identity=replace(p.benchmark.identity, subject="WRONG")),
        ),
        lambda p: replace(
            p, stock=replace(p.stock, identity=replace(p.stock.identity, sha256="bad"))
        ),
        lambda p: replace(
            p,
            stock=replace(
                p.stock,
                identity=replace(
                    p.stock.identity, available_at=p.decision_time + timedelta(seconds=1)
                ),
            ),
        ),
        lambda p: replace(
            p,
            benchmark=replace(
                p.benchmark,
                identity=replace(
                    p.benchmark.identity, retrieved_at=p.decision_time + timedelta(seconds=1)
                ),
            ),
        ),
        lambda p: replace(
            p,
            stock=replace(p.stock, dates=(*p.stock.dates[:-1], p.target_date + timedelta(days=1))),
        ),
        lambda p: replace(
            p,
            stock=replace(
                p.stock, dates=(p.stock.dates[0] - timedelta(days=1), *p.stock.dates[1:])
            ),
        ),
        lambda p: replace(p, stock=replace(p.stock, closes=(float("nan"), *p.stock.closes[1:]))),
        lambda p: replace(p, source_execution=SourceExecutionBinding("synthetic_demo", "observed")),
        lambda p: replace(p, calendar_sessions=p.calendar_sessions[1:]),
        lambda p: replace(p, target_date=p.target_date + timedelta(days=1)),
    ],
)
def test_native_rejection_type_code_and_wording_unchanged(change, config, product_input):
    changed = change(product_input)
    with pytest.raises(PriceProductInputError) as expected:
        native.calculate_price_product(changed, config=config)
    with pytest.raises(type(expected.value)) as actual:
        shadow.project_shadow_drift(changed, config=config)
    assert actual.value.reason_code == expected.value.reason_code
    assert str(actual.value) == str(expected.value)


def test_provider_label_does_not_authenticate_or_promote(config, product_input):
    provider = replace(
        product_input,
        stock=replace(
            product_input.stock,
            identity=replace(product_input.stock.identity, provider="twelve_data"),
        ),
        benchmark=replace(
            product_input.benchmark,
            identity=replace(product_input.benchmark.identity, provider="twelve_data"),
        ),
        source_execution=SourceExecutionBinding("provider", "observed"),
    )
    result = shadow.project_shadow_drift(provider, config=config)
    assert result.source_execution == provider.source_execution
    assert result.evidence_label == "research_only_unscored"
    assert result.source_provenance_status == "caller_supplied_unverified"
    assert result.input_hash != INPUT_HASH and result.innovation_seed == SEED
    with pytest.raises(ValueError, match="requires synthetic_demo research inputs"):
        old_drift_tests.study.project_synthetic_drift_experiment(provider, config=config)


@pytest.mark.parametrize("mutation", ["listing", "asset", "checksum", "close", "volume"])
def test_complete_input_identity_changes_not_authenticated(mutation, config, product_input):
    stock = product_input.stock
    if mutation == "listing":
        changed = replace(product_input, listing_id=UUID("bbbbbbbb-1111-4222-8333-bbbbbbbbbbbb"))
    elif mutation == "asset":
        changed = replace(
            product_input,
            stock=replace(
                stock,
                identity=replace(
                    stock.identity, asset_id=UUID("33333333-3333-4333-8333-333333333333")
                ),
            ),
        )
    elif mutation == "checksum":
        changed = replace(
            product_input, stock=replace(stock, identity=replace(stock.identity, sha256="c" * 64))
        )
    elif mutation == "close":
        changed = replace(
            product_input,
            stock=replace(
                stock, closes=(math.nextafter(stock.closes[0], math.inf), *stock.closes[1:])
            ),
        )
    else:
        changed = replace(product_input, stock=replace(stock, volumes=(1234.0,) * 757))
    result = shadow.project_shadow_drift(changed, config=config)
    assert result.input_hash == native.complete_input_hash(changed) != INPUT_HASH
    assert (result.innovation_seed != SEED) == (mutation == "listing")
    assert result.source_provenance_status == "caller_supplied_unverified"


def _independent_full_arrays(filtered, seed):
    """All paths at once, rather than the native chunked implementation."""
    indices = np.random.Generator(np.random.PCG64(seed)).integers(
        0, 504, size=(8192, 1260), dtype=np.int64
    )
    residuals = np.asarray(filtered.residuals, dtype=np.float64)
    variance = np.full(8192, filtered.terminal_variance)
    cumulative = np.zeros(8192, dtype=np.float64)
    historical, zero = [], []
    for step in range(1260):
        innovation = np.sqrt(variance) * residuals[indices[:, step]]
        cumulative += filtered.mean_log_return + innovation
        variance = (
            0.01 * filtered.population_variance + 0.94 * variance + 0.05 * innovation * innovation
        )
        if step + 1 in (126, 252, 756, 1260):
            historical.append(cumulative.copy())
            zero.append(cumulative - (step + 1) * filtered.mean_log_return)
    return historical, zero


def _assert_mapped_projection(row, projection, prefix=""):
    assert row.horizon == projection.horizon and row.sessions == projection.sessions
    assert _identity(row.quantile_levels) == _identity(projection.quantile_levels)
    assert row.central_model_mass.hex() == projection.central_model_mass.hex()
    assert row.insufficiency_reason == projection.insufficiency_reason
    for name in TRIPLETS:
        assert _identity(getattr(row, name)) == _identity(getattr(projection, prefix + name))


def test_full_array_oracle_all_controls_and_exact_candidate_operation(evidence, product_input):
    result, actual, (filtered, arguments, terminals), captures = evidence
    assert arguments == {
        "seed": SEED,
        "horizons": (126, 252, 756, 1260),
        "path_count": 8192,
        "diagnostic_max_paths": 16384,
        "variance_target_weight": 0.01,
        "variance_persistence": 0.94,
        "innovation_weight": 0.05,
    }
    historical, zero = _independent_full_arrays(filtered, SEED)
    assert len(captures) == 6
    assert [(p.arm_id, p.horizon, p.sessions) for p in result.projections] == [
        (arm, horizon, sessions) for arm in ARMS for horizon, sessions in HORIZONS[:2]
    ]
    for index in range(4):
        assert terminals.with_drift[index].tobytes() == historical[index].tobytes()
        assert terminals.zero_drift[index].tobytes() == zero[index].tobytes()
        recaptured = [
            args for args, _ in captures if args["terminal_logs"] is terminals.with_drift[index]
        ]
        assert len(recaptured) == 1
        assert recaptured[0]["zero_drift_logs"] is terminals.zero_drift[index]
    for index in range(2):
        projection = actual.forecast.projections[index]
        _assert_mapped_projection(result.projections[index], projection)
        _assert_mapped_projection(result.projections[index + 2], projection, "zero_drift_")
        args, projected = captures[2 * index + 1]
        assert (
            args["terminal_logs"].tobytes()
            == (zero[index] + projection.sessions * (filtered.mean_log_return / 2.0)).tobytes()
        )
        assert args["zero_drift_logs"] is terminals.zero_drift[index]
        _assert_mapped_projection(result.projections[index + 4], projected)
        for raw, ledger in (("raw_returns", "ledger_returns"), ("raw_prices", "ledger_prices")):
            triplet = getattr(projected, raw)
            assert triplet is not None and getattr(projected, ledger) is not None
        raw = projected.raw_returns
        assert projected.raw_prices == RawTriplet(
            *(
                product_input.stock.closes[-1] * (1 + getattr(raw, name))
                for name in ("lower", "median", "upper")
            )
        )
    assert result.mean_log_return.hex() == filtered.mean_log_return.hex()
    assert result.half_log_drift.hex() == (filtered.mean_log_return / 2.0).hex()


@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_positive_negative_and_exact_zero_mean(sign, product_input, config):
    closes = tuple((100.0 + (i % 2)) * (1.0 + sign * i / 8192.0) for i in range(757))
    changed = replace(product_input, stock=replace(product_input.stock, closes=closes))
    actual = native.calculate_price_product(changed, config=config)
    result = shadow.project_shadow_drift(changed, config=config)
    assert math.copysign(1, result.mean_log_return) == sign if sign else result.mean_log_return == 0
    assert result.mean_log_return.hex() == actual.forecast.mean_log_return.hex()
    assert result.half_log_drift.hex() == (result.mean_log_return / 2.0).hex()
    for index in range(2):
        _assert_mapped_projection(result.projections[index], actual.forecast.projections[index])
        _assert_mapped_projection(
            result.projections[index + 2], actual.forecast.projections[index], "zero_drift_"
        )
        if sign == 0:
            for name in TRIPLETS:
                assert _identity(getattr(result.projections[index + 4], name)) == _identity(
                    getattr(result.projections[index + 2], name)
                )


def test_genuine_flat_filter_withholding_no_regeneration(product_input, config, monkeypatch):
    flat = replace(product_input, stock=replace(product_input.stock, closes=(100.0,) * 757))
    monkeypatch.setattr(shadow, "simulate_fhs_terminal_logs", _refuse)
    result = shadow.project_shadow_drift(flat, config=config)
    assert result.native_forecast_insufficiency_reason == "filter_variance_degenerate"
    assert result.mean_log_return is result.half_log_drift is None
    assert len(result.projections) == 6
    for row in result.projections:
        assert row.insufficiency_reason == "filter_variance_degenerate"
        assert all(getattr(row, name) is None for name in TRIPLETS)
    assert shadow.serialize_shadow_drift(result) == _json_bytes(_identity(result))


@pytest.mark.parametrize("drift", [0.03, 0.04])
def test_genuine_horizon_representability_no_candidate_rescue(
    drift, product_input, config, monkeypatch
):
    closes = tuple(
        float(x) for x in 100.0 * np.exp(np.arange(757) * drift + np.sin(np.arange(757)) * 0.002)
    )
    changed = replace(product_input, stock=replace(product_input.stock, closes=closes))
    actual = native.calculate_price_product(changed, config=config)
    assert (
        actual.forecast.projections[1].insufficiency_reason == "projection_database_unrepresentable"
    )
    if drift == 0.04:
        assert actual.forecast.projections[0].insufficiency_reason is not None
        monkeypatch.setattr(shadow, "simulate_fhs_terminal_logs", _refuse)
    else:
        assert actual.forecast.projections[0].insufficiency_reason is None
    result = shadow.project_shadow_drift(changed, config=config)
    for row in result.projections:
        original = actual.forecast.projections[0 if row.horizon == "6m" else 1]
        assert row.insufficiency_reason == original.insufficiency_reason
        if row.insufficiency_reason:
            assert all(getattr(row, field) is None for field in TRIPLETS)


@pytest.mark.parametrize("failure", ["quotient", "turnover"])
def test_native_finite_input_arithmetic_exceptions_propagate(failure, product_input, config):
    if failure == "quotient":
        closes = list(product_input.stock.closes)
        closes[-253], closes[-22] = 1e300, 1e-300
        changed = replace(product_input, stock=replace(product_input.stock, closes=tuple(closes)))
        exception = ValueError
    else:
        changed = replace(
            product_input,
            stock=replace(
                product_input.stock, volumes=(1e305,) * 757, volume_adjustment_compatible=True
            ),
        )
        exception = OverflowError
    with pytest.raises(exception) as original:
        native.calculate_price_product(changed, config=config)
    with pytest.raises(type(original.value)) as result:
        shadow.project_shadow_drift(changed, config=config)
    assert str(result.value) == str(original.value)


def _reuse_actual_dependencies(monkeypatch, evidence):
    monkeypatch.setattr(shadow, "calculate_price_product", lambda *a, **k: evidence[1])
    monkeypatch.setattr(shadow, "simulate_fhs_terminal_logs", lambda *a, **k: evidence[2][2])


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("nonfinite", "simulation_nonfinite"),
        ("return", "projection_return_nonfinite"),
        ("price", "projection_triplet_invalid"),
        ("decimal", "projection_database_unrepresentable"),
        ("underflow", "projection_return_unrepresentable"),
    ],
)
def test_candidate_failure_entire_one_horizon_only(
    failure, reason, config, product_input, evidence, monkeypatch
):
    _reuse_actual_dependencies(monkeypatch, evidence)
    terminals = evidence[2][2]
    original = shadow.projection_from_terminal_logs
    injected = []

    def inject(**kwargs):
        if kwargs["horizon"] == "12m" and kwargs["terminal_logs"] is not terminals.with_drift[1]:
            logs = kwargs["terminal_logs"].copy()
            if failure == "nonfinite":
                logs[0] = np.nan
            elif failure == "return":
                logs[0] = 1000.0
            elif failure == "price":
                logs[:] = 10.0
                kwargs["target_close"] = 1e308
            elif failure == "decimal":
                logs[:] = 20.0
            else:
                logs[:] = -1000.0
            kwargs["terminal_logs"] = logs
            injected.append(logs)
        return original(**kwargs)

    monkeypatch.setattr(shadow, "projection_from_terminal_logs", inject)
    result = shadow.project_shadow_drift(product_input, config=config)
    assert len(injected) == 1 and injected[0].shape == (8192,)
    for before, after in zip(evidence[0].projections, result.projections, strict=True):
        if (after.arm_id, after.horizon) == (ARMS[2], "12m"):
            assert after.insufficiency_reason == reason
            assert all(getattr(after, name) is None for name in TRIPLETS)
        else:
            assert _identity(before) == _identity(after)


@pytest.mark.parametrize(
    "failure", ["horizons", "path_count", "dtype", "shape", "container", "missing", "array"]
)
def test_malformed_simulation_dependency_fails_explicitly(
    failure, config, product_input, evidence, monkeypatch
):
    _reuse_actual_dependencies(monkeypatch, evidence)
    terminals = evidence[2][2]
    changes = {
        "horizons": lambda: replace(terminals, horizons=(126, 252)),
        "path_count": lambda: replace(terminals, path_count=4096),
        "dtype": lambda: replace(
            terminals,
            with_drift=(terminals.with_drift[0].astype(np.float32), *terminals.with_drift[1:]),
        ),
        "shape": lambda: replace(
            terminals, zero_drift=(terminals.zero_drift[0][:-1], *terminals.zero_drift[1:])
        ),
        "container": lambda: replace(terminals, with_drift=list(terminals.with_drift)),
        "missing": lambda: replace(terminals, zero_drift=terminals.zero_drift[:-1]),
        "array": lambda: replace(terminals, zero_drift=((1.0,) * 8192, *terminals.zero_drift[1:])),
    }
    monkeypatch.setattr(shadow, "simulate_fhs_terminal_logs", lambda *a, **k: changes[failure]())
    with pytest.raises(ValueError, match="native dependency mismatch"):
        shadow.project_shadow_drift(product_input, config=config)


@pytest.mark.parametrize(
    "field",
    [
        "raw_returns",
        "ledger_returns",
        "raw_prices",
        "ledger_prices",
        "zero_drift_raw_returns",
        "zero_drift_ledger_returns",
        "zero_drift_raw_prices",
        "zero_drift_ledger_prices",
        "quantile_levels",
        "central_model_mass",
        "insufficiency_reason",
    ],
)
def test_complete_regenerated_payload_not_just_ledger_proxy(
    field, config, product_input, evidence, monkeypatch
):
    _reuse_actual_dependencies(monkeypatch, evidence)
    original = shadow.projection_from_terminal_logs
    terminals = evidence[2][2]

    def alter(**kwargs):
        result = original(**kwargs)
        if kwargs["terminal_logs"] is terminals.with_drift[0]:
            old = getattr(result, field)
            if "ledger" in field:
                changed = replace(old, median=old.median.quantize(Decimal("0.000000000")))
            elif "raw" in field:
                changed = replace(old, median=math.nextafter(old.median, math.inf))
            elif field == "quantile_levels":
                changed = (0.2, math.nextafter(0.5, 1.0), 0.8)
            elif field == "central_model_mass":
                changed = math.nextafter(0.6, 1.0)
            else:
                changed = "injected"
            return replace(result, **{field: changed})
        return result

    monkeypatch.setattr(shadow, "projection_from_terminal_logs", alter)
    with pytest.raises(ValueError, match="native dependency mismatch"):
        shadow.project_shadow_drift(product_input, config=config)


def _immutable_graph(value):
    if is_dataclass(value):
        assert value.__dataclass_params__.frozen
        assert hasattr(type(value), "__slots__") and not hasattr(value, "__dict__")
        for field in fields(value):
            _immutable_graph(getattr(value, field.name))
    elif type(value) is tuple:
        for item in value:
            _immutable_graph(item)
    else:
        assert value is None or type(value) in (str, int, float, Decimal, UUID, date, datetime)


def test_complete_encoding_immutable_graph_inputs_and_repeatability(
    config, product_input, evidence
):
    before = asdict(product_input)
    result = shadow.project_shadow_drift(product_input, config=config)
    _immutable_graph(result)
    for value, name, change in (
        (result, "mean_log_return", 0.0),
        (result.stock_identity, "subject", "changed"),
        (result.source_execution, "evidence_grade", "observed"),
        (result.projections[0], "sessions", 1),
        (result.projections[0].raw_returns, "median", 0.0),
        (result.projections[0].ledger_returns, "median", Decimal("0")),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, name, change)
    with pytest.raises(TypeError):
        result.projections[0] = result.projections[1]
    encoded = shadow.serialize_shadow_drift(result)
    assert encoded == _json_bytes(_identity(result))
    assert set(json.loads(encoded)) == {f.name for f in fields(shadow.ShadowDriftResult)}
    assert not encoded.endswith(b"\n") and encoded.isascii()
    assert encoded == shadow.serialize_shadow_drift(evidence[0])
    assert asdict(product_input) == before


def test_signed_zero_and_decimal_scale_are_identity(evidence):
    result = evidence[0]
    row = replace(
        result.projections[0],
        raw_returns=RawTriplet(-0.0, 0.0, 0.0),
        ledger_returns=LedgerTriplet(Decimal("-0.0000"), Decimal("0.00"), Decimal("0.000")),
    )
    changed = replace(result, projections=(row, *result.projections[1:]))
    document = json.loads(shadow.serialize_shadow_drift(changed))
    assert document["projections"][0]["raw_returns"]["lower"] == "-0x0.0p+0"
    assert document["projections"][0]["ledger_returns"] == {
        "lower": "-0.0000",
        "median": "0.00",
        "upper": "0.000",
    }
    other = replace(row, raw_returns=RawTriplet(0.0, 0.0, 0.0))
    assert shadow.serialize_shadow_drift(changed) != shadow.serialize_shadow_drift(
        replace(result, projections=(other, *result.projections[1:]))
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda r: asdict(r),
        lambda r: replace(r, schema_version="wrong"),
        lambda r: replace(r, path_count=True),
        lambda r: replace(r, path_count=4096),
        lambda r: replace(r, innovation_seed=-1),
        lambda r: replace(r, listing_id=str(r.listing_id)),
        lambda r: replace(r, projections=list(r.projections)),
        lambda r: replace(r, projections=r.projections[:-1]),
        lambda r: replace(r, simulation_horizons=(("6m", 126), ("12m", 252))),
        lambda r: replace(r, horizon_roles=tuple(reversed(r.horizon_roles))),
        lambda r: replace(r, mean_log_return=float("nan")),
        lambda r: replace(r, mean_log_return=float("inf")),
        lambda r: replace(r, mean_log_return=Decimal("0")),
        lambda r: replace(r, half_log_drift=None),
        lambda r: replace(r, assumptions_sha256="a" * 64),
        lambda r: replace(r, evidence_label="observed"),
        lambda r: replace(r, stock_identity=asdict(r.stock_identity)),
        lambda r: replace(r, decision_time=r.decision_time.replace(tzinfo=None)),
        lambda r: replace(
            r, projections=(replace(r.projections[0], arm_id="unknown"), *r.projections[1:])
        ),
        lambda r: replace(
            r,
            projections=(
                replace(r.projections[0], quantile_levels=[0.2, 0.5, 0.8]),
                *r.projections[1:],
            ),
        ),
        lambda r: replace(
            r,
            projections=(
                replace(r.projections[0], raw_returns=RawTriplet(0.0, float("nan"), 1.0)),
                *r.projections[1:],
            ),
        ),
        lambda r: replace(
            r,
            projections=(
                replace(
                    r.projections[0],
                    ledger_returns=LedgerTriplet(Decimal("0"), Decimal("NaN"), Decimal("1")),
                ),
                *r.projections[1:],
            ),
        ),
        lambda r: replace(
            r,
            projections=(
                replace(
                    r.projections[0],
                    ledger_prices=LedgerTriplet(
                        Decimal("0"), Decimal("Infinity"), Decimal("Infinity")
                    ),
                ),
                *r.projections[1:],
            ),
        ),
        lambda r: replace(
            r, projections=(replace(r.projections[0], raw_returns=None), *r.projections[1:])
        ),
        lambda r: replace(
            r, projections=(replace(r.projections[0], insufficiency_reason=""), *r.projections[1:])
        ),
    ],
)
def test_malformed_serialization_fails_closed(change, evidence):
    with pytest.raises(ValueError):
        shadow.serialize_shadow_drift(change(evidence[0]))


def _cold_purity():
    """Called only in its own fresh interpreter, never by a warmed fixture."""
    import builtins
    import io
    import socket
    import time
    from collections.abc import Mapping

    import django

    django.setup()
    from django.db.backends.utils import CursorWrapper

    p, c = _synthetic_input(), load_price_product_config()
    rng_before = np.random.get_state()

    class ForbiddenEnvironment(Mapping):
        __getitem__ = __iter__ = __len__ = _refuse

    def profile(frame, event, arg):
        if event == "c_call" and getattr(arg, "__name__", "") in ("now", "today", "utcnow"):
            raise AssertionError("Clock access during cold preparation")

    with pytest.MonkeyPatch.context() as patch:
        for obj, name in (
            (builtins, "open"),
            (io, "open"),
            (os, "open"),
            (os, "getenv"),
            (os, "listdir"),
            (os, "scandir"),
            (os, "stat"),
            (os, "lstat"),
            (os, "system"),
            (subprocess, "Popen"),
            (socket.socket, "connect"),
            (socket.socket, "connect_ex"),
            (socket, "create_connection"),
            (time, "time"),
            (time, "time_ns"),
            (time, "monotonic"),
            (time, "perf_counter"),
            (CursorWrapper, "execute"),
            (CursorWrapper, "executemany"),
        ):
            patch.setattr(obj, name, _refuse)
        patch.setattr(os, "environ", ForbiddenEnvironment())
        sys.setprofile(profile)
        try:
            cold = shadow.serialize_shadow_drift(shadow.project_shadow_drift(p, config=c))
        finally:
            sys.setprofile(None)
    assert cold == shadow.serialize_shadow_drift(shadow.project_shadow_drift(p, config=c))
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0] and rng_before[2:] == rng_after[2:]
    assert np.array_equal(rng_before[1], rng_after[1])
    return hashlib.sha256(cold).hexdigest()


def _process_environment(source):
    return {
        "PATH": os.defpath,
        "PYTHONPATH": os.pathsep.join((str(source), str(ROOT / "tests"))),
        "DJANGO_SETTINGS_MODULE": "stanstock.settings.test",
        "STANSTOCK_CODE_REVISION": "a" * 40,
    }


def test_cold_first_call_no_io_environment_clock_network_process_or_orm(tmp_path, evidence):
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "from test_research_price_product_shadow_drift import _cold_purity; "
            "print(_cold_purity())",
        ],
        cwd=tmp_path,
        env=_process_environment(ROOT / "src"),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert (
        process.stdout.strip()
        == hashlib.sha256(shadow.serialize_shadow_drift(evidence[0])).hexdigest()
    )


def test_interleaved_concurrent_independent_calls_no_global_state(product_input, config, evidence):
    flat = replace(product_input, stock=replace(product_input.stock, closes=(100.0,) * 757))
    before = np.random.get_state()
    inputs = (product_input, flat, product_input, flat)
    expected = {
        False: shadow.serialize_shadow_drift(evidence[0]),
        True: shadow.serialize_shadow_drift(shadow.project_shadow_drift(flat, config=config)),
    }

    def calculate(p):
        return shadow.serialize_shadow_drift(shadow.project_shadow_drift(p, config=config))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(calculate, inputs))
    assert results == tuple(expected[p is flat] for p in inputs)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    assert np.array_equal(before[1], after[1])


def test_pure_import_closure_no_production_consumers_and_bounded_worktree():
    tree = ast.parse(inspect.getsource(shadow))
    modules = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    }
    assert modules <= {
        "__future__",
        "hashlib",
        "json",
        "math",
        "dataclasses",
        "datetime",
        "decimal",
        "types",
        "typing",
        "uuid",
        "numpy",
        "stanstock.research.price_product",
        "stanstock.research.price_product_config",
    }
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in ("now", "today", "utcnow", "objects")
        for node in ast.walk(tree)
    )
    for path in (ROOT / "src").rglob("*.py"):
        if path.resolve() != Path(shadow.__file__).resolve():
            assert "price_product_shadow_drift" not in path.read_text()
    allowed = {
        "src/stanstock/research/price_product_shadow_drift.py",
        "tests/test_research_price_product_shadow_drift.py",
    }
    # Documentation is concurrently owned by the orchestrator, not this slice.
    diff = (
        frozen._git("diff", "--name-only", BASE_SHA, "--", ".", ":(exclude)docs")
        .stdout.decode()
        .splitlines()
    )
    assert set(diff) <= allowed
    untracked = (
        frozen._git("ls-files", "--others", "--exclude-standard").stdout.decode().splitlines()
    )
    assert set(untracked) <= allowed


def _retained_current_base_recovery(*, store):
    """Execute current native recovery over genuine, frequency-bound base state."""
    from django.conf import settings
    from django.contrib.auth import get_user_model

    import research_product_frozen_probability_probe as probe
    from stanstock.core import research_product_refresh as refresh
    from stanstock.core.models import JobRun
    from stanstock.data.management.config_loader import default_us_universe_config_path
    from stanstock.data.providers import twelve_data
    from stanstock.data.research_product_jobs import SCHEDULED_RESEARCH_JOB, product_job_name
    from stanstock.research.models import AnalysisRun, Prediction
    from stanstock.research.product_frequency_evidence import verify_registered_product_frequencies

    def blocked(*args, **kwargs):
        raise AssertionError("Recovery attempted provider or credential access")

    twelve_data.resolve_api_key = blocked
    twelve_data.fetch_daily_price_series = blocked
    twelve_data.fetch_stock_catalog = blocked
    owner = get_user_model().objects.get(username=probe.OWNER_USERNAME)
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.OWNER_USERNAME = owner.username
    settings.DATA_DIR = store.root
    probe._bind_synthetic_revision()
    config_path = default_us_universe_config_path()
    identity = refresh.scheduled_identity(owner)
    retained = JobRun.objects.get(
        job_name=product_job_name(SCHEDULED_RESEARCH_JOB, identity),
        region="us",
        target_date=probe.TARGET,
        status=JobRun.Status.SUCCESS,
    )
    before = probe._row(retained)
    assert {"frequencies", "frequency_verification"} <= set(retained.details)
    assert JobRun.objects.filter(job_name__startswith="research_product_frequency_v1").exists()
    before_assets = probe._document_assets(store)
    before_predictions = {str(row.pk): probe._row(row) for row in Prediction.objects.order_by("pk")}
    execution = refresh.execute_scheduled_research_refresh(
        core_config_path=config_path,
        store=store,
        enforce_rate_limit=False,
    )
    assert execution.parent.status == JobRun.Status.SKIPPED
    parent = JobRun.objects.get(pk=retained.pk)
    assert probe._row(parent) == before
    recomputed = refresh.verify_scheduled_research_refresh(
        target_date=probe.TARGET,
        owner=owner,
        code_revision=parent.details["code_revision"],
        stages=parent.details["stages"],
        store=store,
        core_config_path=config_path,
    )
    replayed = refresh.replay_recorded_research_product_refresh(parent, store=store)
    run = AnalysisRun.objects.get(pk=recomputed["analysis_run_id"])
    verify_registered_product_frequencies(run=run, store=store)
    assert probe._document_assets(store) == before_assets
    assert {
        str(row.pk): probe._row(row) for row in Prediction.objects.order_by("pk")
    } == before_predictions
    return {
        "scheduled_identity": probe._jsonable(identity),
        "retained_parent_before_recovery": before,
        "retained_predictions_before_recovery": before_predictions,
        "retained_assets_before_recovery": before_assets,
        "parent_job_run": probe._row(parent),
        "recovery_job_run": probe._row(JobRun.objects.get(pk=execution.parent.pk)),
        "job_runs": {
            f"{job.job_name}|{job.region}|{job.target_date}|{job.attempt}": probe._row(job)
            for job in JobRun.objects.order_by("job_name", "attempt")
        },
        "recorded_verification": probe._jsonable(parent.details["verification"]),
        "recomputed_verification": probe._jsonable(recomputed),
        "replayed_verification": probe._jsonable(replayed.verification),
        "replayed_parent_id": str(replayed.parent.pk),
        "replayed_snapshot_id": str(replayed.snapshot.pk),
        "replayed_analysis_run_id": str(replayed.analysis_run.pk),
        "replayed_catalog_asset_ids": sorted(str(asset.id) for asset in replayed.catalog_assets),
        "analysis_count": execution.analysis_count,
        "prediction_count": execution.prediction_count,
        "frequency_semantic_verification": "verified",
        "_run": run,
    }


def _old_study_payload(workdir):
    """Bounded fixed-identity construction of both unchanged research surfaces."""
    import json
    import math
    from dataclasses import asdict, replace
    from datetime import UTC, datetime

    from django.conf import settings
    from django.core.management import call_command

    import research_product_frozen_probability_probe as probe
    import test_research_price_product_drift_study as fixtures
    from stanstock.research.price_product_config import load_price_product_config
    from stanstock.research.price_product_drift_study import (
        project_synthetic_drift_experiment,
        score_synthetic_drift_outcome,
    )

    probe._configure_django(workdir)
    call_command("migrate", verbosity=0)
    probe._bind_natural_key_identities()
    probe._freeze_clock()
    config = load_price_product_config()
    product_input = fixtures.product_input.__wrapped__()
    drift_documents = {}
    for label, p in (
        ("successful", product_input),
        (
            "withheld",
            replace(product_input, stock=replace(product_input.stock, closes=(100.0,) * 757)),
        ),
    ):
        result = project_synthetic_drift_experiment(p, config=config)
        drift_documents[label] = {
            "experiment": probe._jsonable(asdict(result)),
            "hypothetical_scores": [
                {
                    "terminal_log_return": probe._jsonable(y),
                    "scores": [
                        probe._jsonable(
                            asdict(
                                score_synthetic_drift_outcome(row, synthetic_terminal_log_return=y)
                            )
                        )
                        for row in result.horizons
                    ],
                }
                for y in (math.log1p(-0.2), 0.0, math.log1p(0.4), None)
            ],
        }

    from stanstock.data import research_product_demo as demo
    from stanstock.data.assets import AssetStore
    from stanstock.research.models import AnalysisRun, Prediction
    from stanstock.research.price_product_study import (
        render_price_product_study,
        serialize_price_product_study,
        study_price_product_run,
    )

    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = True
    store = AssetStore(workdir / "assets")
    settings.DATA_DIR = store.root
    # Only synthetic fixture recipes change, before the genuine source writer.
    demo.DEMO_STOCKS = (
        demo.DEMO_STOCKS[0],
        demo.DemoSeries("ZZRPFLAT", "Synthetic Flat", 100.0, 0.0, 0.0),
    )
    job = demo.execute_demo_product_refresh(store=store)
    run = AnalysisRun.objects.select_related("universe_snapshot").get(
        pk=job.details["analysis_run_id"]
    )
    before_predictions = {str(row.pk): probe._row(row) for row in Prediction.objects.order_by("pk")}
    before_assets = probe._document_assets(store)
    report = study_price_product_run(
        run=run,
        store=store,
        all_selected=True,
        report_generated_at=datetime(2026, 9, 13, 18, tzinfo=UTC),
    )
    serialized = serialize_price_product_study(report, include_generated_at=True)
    assert before_assets == probe._document_assets(store)
    assert before_predictions == {
        str(row.pk): probe._row(row) for row in Prediction.objects.order_by("pk")
    }
    return {
        "config": probe._config_payload(),
        "synthetic_drift": drift_documents,
        "retrospective": serialized,
        "retrospective_json_bytes": render_price_product_study(
            report,
            output_format="json",
            include_generated_at=True,
        )
        .encode("utf-8")
        .hex(),
        "retrospective_text_bytes": render_price_product_study(
            report,
            output_format="text",
            include_generated_at=True,
        )
        .encode("utf-8")
        .hex(),
        "synthetic_drift_bytes": json.dumps(
            drift_documents,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        .encode("utf-8")
        .hex(),
        "source_predictions": before_predictions,
        "source_assets": before_assets,
    }


@pytest.fixture(scope="module")
def exact_current_base(tmp_path_factory):
    """One bounded adapter; old harness and all frequency bindings stay intact."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(frozen, "FROZEN_BASE_SHA", BASE_SHA)
        assert frozen._base_objects_available(), f"Required base objects unavailable: {BASE_SHA}"
        try:
            base = frozen.frozen_base_root.__wrapped__(tmp_path_factory)
        except pytest.skip.Exception:
            pytest.fail("Required exact-base export failed; preservation cannot be skipped")
    roots = {"base": base, "candidate": frozen.candidate_root.__wrapped__()}
    workspace = tmp_path_factory.mktemp("shadow-current-base")
    cache = {}

    def execute(revision, mode):
        key = (revision, mode)
        if key in cache:
            return cache[key]
        workdir = workspace / f"{revision}-{mode}"
        if mode == "recovery":
            source = execute("base", "scheduled")
            assert source.succeeded, source.failure_detail()
            shutil.copytree(source.workdir, workdir)
            (workdir / "payload.json").unlink()
        else:
            workdir.mkdir()
        if mode in ("daily_derived", "scheduled"):
            run = frozen._run_probe(
                source_root=roots[revision],
                mode=mode,
                cohort="full",
                workdir=workdir,
            )
        else:
            function = _retained_current_base_recovery if mode == "recovery" else _old_study_payload
            code = (
                "import json, os, sys, uuid\nfrom pathlib import Path\n"
                "import research_product_frozen_probability_probe as probe\n"
                "probe._install_boundary_guards()\n"
                "uuid.uuid4 = probe._DeterministicUuid('uuid4')\n"
                + inspect.getsource(function)
                + "\nworkdir = Path(sys.argv[1])\n"
            )
            if mode == "recovery":
                code += (
                    "probe._recover_scheduled = _retained_current_base_recovery\n"
                    "payload = probe.run(mode='scheduled_recovery', "
                    "cohort='full', workdir=workdir)\n"
                )
            else:
                assert mode == "old_studies"
                code += "payload = _old_study_payload(workdir)\n"
            code += (
                "(workdir / 'payload.json').write_text("
                "json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False))\n"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code, str(workdir)],
                cwd=workdir,
                env=_process_environment(roots[revision]),
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            output = workdir / "payload.json"
            run = frozen.ProbeRun(
                returncode=completed.returncode,
                payload=json.loads(output.read_text())
                if completed.returncode == 0 and output.is_file()
                else None,
                stderr=completed.stderr,
                workdir=workdir,
            )
        assert run.succeeded, (
            f"{revision}/{mode} failed (exit {run.returncode}): {run.failure_detail()}"
        )
        cache[key] = run
        return run

    return execute


def _require_complete_native_states(payload):
    assert payload["config"]["config_file_bytes_sha256"] == CONFIG_FILE_HASH
    assert payload["config"]["recomputed_effective_config_hash"] == CONFIG_HASH
    assert payload["output"]["prediction_count"] == 20
    for horizon, _sessions in HORIZONS:
        rows = payload["output"]["predictions"]
        good = rows[f"CHEAP|advisory|{horizon}|us-price-fhs-v1"]
        flat = rows[f"FLAT|advisory|{horizon}|us-price-fhs-v1"]
        assert good["base_return"] is not None and good["insufficiency_reason"] == ""
        assert flat["base_return"] is None
        assert flat["insufficiency_reason"] == "filter_variance_degenerate"
        assert good["calculation"]["forecast"]["residual_count"] == 504
        assert flat["calculation"]["forecast"]["residual_count"] == 0
        for projection in good["calculation"]["forecast"]["projections"]:
            for prefix in ("", "zero_drift_"):
                for field in TRIPLETS:
                    assert projection[prefix + field] is not None
        for projection in flat["calculation"]["forecast"]["projections"]:
            assert projection["insufficiency_reason"] == "filter_variance_degenerate"
            for prefix in ("", "zero_drift_"):
                for field in TRIPLETS:
                    assert projection[prefix + field] is None
    documents = [
        asset["document"]
        for key, asset in payload["assets"].items()
        if key.startswith("research_product_frequency_evidence|")
    ]
    assert len(documents) == 1
    complete = frozen._flatten(documents[0])
    assert any(path[-1:] == ("counts",) and value == "null" for path, value in complete.items())
    assert any(path[-2:] == ("counts", "loss") for path in complete)


@pytest.mark.parametrize("mode", ["daily_derived", "scheduled", "recovery"])
def test_exact_base_native_complete_payload_and_retained_recovery(exact_current_base, mode):
    before = exact_current_base("base", mode).require("current base")
    after = exact_current_base("candidate", mode).require("working tree")
    frozen.assert_frozen_surface_unchanged(before, after, allow_additive=False)
    _require_complete_native_states(before)
    if mode in ("scheduled", "recovery"):
        scheduled = before["scheduled"]
        assert scheduled["recorded_verification"] == scheduled["recomputed_verification"]
        assert scheduled["recorded_verification"] == scheduled["replayed_verification"]
        assert {"frequencies", "frequency_verification"} <= set(
            scheduled["parent_job_run"]["details"]
        )
    if mode == "recovery":
        source = exact_current_base("base", "scheduled").require("actual base-written state")
        assert before["output"] == source["output"]
        assert before["assets"] == source["assets"]
        assert scheduled["recovery_job_run"]["status"] == "skipped"
        assert scheduled["parent_job_run"] == source["scheduled"]["parent_job_run"]
        assert scheduled["retained_parent_before_recovery"] == scheduled["parent_job_run"]
        assert scheduled["frequency_semantic_verification"] == "verified"


def test_exact_base_complete_old_synthetic_and_retrospective_bytes(exact_current_base):
    base = exact_current_base("base", "old_studies").require("old studies at exact base")
    candidate = exact_current_base("candidate", "old_studies").require("old studies in worktree")
    frozen.assert_frozen_surface_unchanged(base, candidate, allow_additive=False)
    synthetic = base["synthetic_drift"]
    for name in ("successful", "withheld"):
        experiment = synthetic[name]["experiment"]
        assert len(experiment["horizons"]) == 16
        assert {row["model_id"] for row in experiment["horizons"]} == set(old_drift_tests.MODELS)
        for row in experiment["horizons"]:
            if name == "successful":
                assert row["insufficiency_reason"] is None and row["event_counts"] is not None
            else:
                assert row["insufficiency_reason"] == "filter_variance_degenerate"
                assert row["event_counts"] is row["raw_returns"] is None
        assert len(synthetic[name]["hypothetical_scores"]) == 4
        for row in synthetic[name]["hypothetical_scores"][0]["scores"]:
            if name == "successful":
                assert row["brier_score"] is not None and row["interval_score"] is not None
            else:
                assert row["insufficiency_reason"] == "filter_variance_degenerate"
    report = base["retrospective"]
    assert report["report_generated_at"] == datetime(2026, 9, 13, 18, tzinfo=UTC).isoformat()
    assert report["scope"]["studied_listing_count"] == 2
    assert {row["partition"] for row in report["projection_aggregates"]} == {
        "development",
        "validation",
        "final_holdout",
    }
    pairs = report["paired_model_comparisons"]
    assert {row["baseline_model"] for row in pairs} == {
        "historical_log_drift_gaussian",
        "zero_log_drift_gaussian",
    }
    assert any(row["paired_observation_count"] > 0 for row in pairs)
    assert any(
        row["unavailable_reason"] == "no_aligned_candidate_baseline_observations" for row in pairs
    )
    flat = frozen._flatten(report)
    assert any(value == '"filter_variance_degenerate"' for value in flat.values())
    assert report["convergence_aggregates"] and report["momentum_aggregates"]
    assert json.loads(bytes.fromhex(base["retrospective_json_bytes"])) == report
    assert (
        "report_generated_at=2026-09-13T18:00:00+00:00"
        in bytes.fromhex(base["retrospective_text_bytes"]).decode()
    )

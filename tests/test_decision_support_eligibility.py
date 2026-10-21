from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
from decimal import Decimal

import pytest

from stanstock.research.decision_support_eligibility import (
    LiquidityObservation,
    assess_liquidity,
    assess_mode_witness,
    default_decision_support_config_path,
    load_decision_support_config,
)


def _window(close: str = "20", volume: int | None = 300_000):
    return tuple(
        LiquidityObservation(date(2026, 9, 1) + timedelta(days=i), Decimal(close), volume)
        for i in range(20)
    )


def _assess(none=None, splits=None, **kwargs):
    values = _window()
    return assess_liquidity(
        required_sessions=tuple(row.session for row in values),
        none=values if none is None else none,
        splits=values if splits is None else splits,
        witness_status="matched",
        **kwargs,
    )


def test_exact_default_config_pins_and_immutable_used_values():
    payload = default_decision_support_config_path().read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "8021590dabbedde09a4c4508aa7203e244c78fed9eeb09ea860ac2f826c9cf89"
    )
    config = load_decision_support_config()
    assert config.effective_config_hash == (
        "16664e7fa98d94484f97da6c68a473e54dac5523c16f3bf5acfd1680f34559da"
    )
    with pytest.raises(FrozenInstanceError):
        config.forecast_config = None


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('"schema_version": 2', '"schema_version": true'),
        ('"schema_version": 2', '"schema_version": 2.0'),
        ('"sessions": 20', '"sessions": true'),
        ('"sessions": 20', '"sessions": 20.0'),
        ('"volume_minimum": 0', '"volume_minimum": false'),
        ('"volume_minimum": 0', '"volume_minimum": 0.0'),
        ('"volume_minimum": 0', '"volume_minimum": NaN'),
        ('"volume_minimum": 0', '"volume_minimum": Infinity'),
        ('"schema_version": 2', '"schema_version": 2, "unknown": 1'),
        ('"sessions": 20', '"sessions": 20, "sessions": 20'),
        ('"calendar": "XNYS"', '"calendar": "XNYS", "calendar": "XNYS"'),
        ('"sessions": 20', '"sessions": 20, "<<": {}'),
        ("55334183", "45334183"),
    ],
)
def test_strict_config_rejects_type_key_literal_and_base_drift(tmp_path, old, new):
    payload = default_decision_support_config_path().read_text()
    assert old in payload
    path = tmp_path / "config.yml"
    path.write_text(payload.replace(old, new))
    with pytest.raises(ValueError):
        load_decision_support_config(path)


@pytest.mark.parametrize("payload", ["schema_version: &v 2\ncopy: *v", '{"a": 1} trailing', "[]"])
def test_strict_config_rejects_yaml_and_non_documents(tmp_path, payload):
    path = tmp_path / "config.yml"
    path.write_text(payload)
    with pytest.raises(ValueError):
        load_decision_support_config(path)


def test_default_file_byte_drift_is_not_hidden_by_canonical_equivalence(tmp_path, monkeypatch):
    path = tmp_path / "config.yml"
    path.write_text(json.dumps(json.loads(default_decision_support_config_path().read_bytes())))
    assert load_decision_support_config(path).product_version == "research-product-v2"
    monkeypatch.setattr(
        "stanstock.research.decision_support_eligibility.default_decision_support_config_path",
        lambda: path,
    )
    with pytest.raises(ValueError, match="bytes changed"):
        load_decision_support_config()


def test_decimal_lsb_is_not_float_parity_and_lexical_equivalence_is_allowed():
    values = _window()
    near = replace(values[-1], close=Decimal("20.00000000000000001"))
    assert float(near.close) == float(values[-1].close)
    assert _assess(splits=values[:-1] + (near,)).reasons == ("liquidity_close_mismatch",)
    assert _assess(splits=_window("20.0000")).average_dollar_turnover_20d == 6_000_000


def test_real_zero_is_not_missing_or_float_volume():
    zero = _assess(none=_window(volume=0), splits=_window(volume=0))
    assert zero.status == "matched"
    assert zero.average_dollar_turnover_20d == 0.0
    assert _assess(none=_window(volume=None)).reasons == ("liquidity_volume_missing",)
    assert _assess(none=_window(volume=300_001)).reasons == ("liquidity_volume_mismatch",)
    for invalid in (True, 1.0, -1, 9223372036854775808):
        with pytest.raises(ValueError, match="bounded integers"):
            _assess(none=_window(volume=invalid))


@pytest.mark.parametrize("close", ["1e1000", "1e-1000", "1e308"])
def test_finite_extremes_are_explicitly_nonrepresentable(close):
    values = _window(close)
    assert _assess(none=values, splits=values).reasons == ("liquidity_nonrepresentable",)


@pytest.mark.parametrize("close", ["NaN", "Infinity", "0", "-1"])
def test_invalid_closes_are_not_admitted(close):
    with pytest.raises(ValueError, match="finite positive"):
        _assess(none=_window(close))


def test_exact_window_no_truncation_gaps_duplicates_or_extra_future_dates():
    values = _window()
    assert _assess(none=values[:-1]).reasons == ("liquidity_window_incomplete",)
    assert _assess(none=values + (replace(values[-1], session=date(2027, 1, 1)),)).reasons == (
        "liquidity_window_incomplete",
    )
    with pytest.raises(ValueError, match="unique"):
        _assess(none=values[:-1] + (values[0],))


def test_proxy_scale_invariance_and_fixed_volume_counterexample():
    original = _assess()
    transformed = _window("40", 150_000)
    assert _assess(none=transformed, splits=transformed) == original
    fixed_volume = _window("40")
    assert _assess(none=fixed_volume, splits=fixed_volume).average_dollar_turnover_20d == (
        2 * original.average_dollar_turnover_20d
    )


def test_fixed_control_distinguishes_ignored_none_and_missing_control():
    config = load_decision_support_config()
    sessions = config.witness_pre_sessions + config.witness_post_sessions
    splits = tuple(LiquidityObservation(s, Decimal("100"), 1000) for s in sessions)
    none = tuple(
        LiquidityObservation(
            s,
            Decimal("1000") if s in config.witness_pre_sessions else Decimal("100"),
            100 if s in config.witness_pre_sessions else 1000,
        )
        for s in sessions
    )
    assert assess_mode_witness(none=none, splits=splits, config=config) == "matched"
    assert assess_mode_witness(none=splits, splits=splits, config=config) == "failed"
    assert assess_mode_witness(none=none[1:], splits=splits, config=config) == "unavailable"
    result = assess_liquidity(
        required_sessions=tuple(row.session for row in _window()),
        none=_window(),
        splits=_window(),
        witness_status="failed",
    )
    assert result.reasons == ("liquidity_mode_witness_failed",)
    assert result.average_dollar_turnover_20d is None

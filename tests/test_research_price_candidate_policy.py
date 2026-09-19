"""Contract witnesses, not a real-data study or a repeat numerical audit."""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import io
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, getcontext, localcontext
from pathlib import Path
from typing import Any, Never, cast
from uuid import UUID

import numpy as np
import pytest
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

import stanstock.research.price_candidate_policy as policy
import stanstock.research.price_candidate_policy_study as driver
import stanstock.research.price_product as native
from stanstock.research.price_product import PriceProductInputError, PriceSeries, RiskResult
from stanstock.research.price_product_config import PriceProductConfig, load_price_product_config

CONFIG_SHA = "47280df24923b14b32c81bc42d324831c8f584650eefe7b12f549c201d8f09b6"
BASE_CONFIG_SHA = "55334183af29fc01b853e83f9bf75216f24564956925fb95912cb80a69420867"
CASE_IDS = (
    "continuation",
    "positive_pullback",
    "deep_reversal",
    "deterioration",
    "flat",
    "deep_missing_volume",
)


@pytest.fixture(scope="module")
def config() -> PriceProductConfig:
    return load_price_product_config()


@pytest.fixture(scope="module")
def candidates() -> tuple[policy.SyntheticCandidateInput, ...]:
    return driver._synthetic_candidates()


@pytest.fixture(scope="module")
def result(config: PriceProductConfig) -> driver.SyntheticCandidateStudy:
    return driver.run_synthetic_candidate_study(base_config=config)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _series_hash(series: PriceSeries) -> str:
    # Independent literal document; do not authenticate the producer with itself.
    document = {
        "schema": "candidate-synthetic-price-series@1",
        "provider": series.identity.provider,
        "subject": series.identity.subject,
        "currency": series.currency,
        "dates": [d.isoformat() for d in series.dates],
        "closes": [c.hex() for c in series.closes],
        "volumes": None
        if series.volumes is None
        else [None if v is None else v.hex() for v in series.volumes],
        "volume_adjustment_compatible": series.volume_adjustment_compatible,
    }
    return hashlib.sha256(_canonical(document)).hexdigest()


def _replace_series(
    case: policy.SyntheticCandidateInput, role: str = "stock", **changes: Any
) -> policy.SyntheticCandidateInput:
    series = replace(getattr(case.product_input, role), **changes)
    series = replace(series, identity=replace(series.identity, sha256=_series_hash(series)))
    return replace(case, product_input=replace(case.product_input, **{role: series}))


def _price(
    case: policy.SyntheticCandidateInput, index: int, value: float, role: str = "stock"
) -> policy.SyntheticCandidateInput:
    values = list(getattr(case.product_input, role).closes)
    values[index] = value
    return _replace_series(case, role, closes=tuple(values))


def _refuse(*args: object, **kwargs: object) -> Never:
    raise AssertionError("Forbidden work reached")


def test_independent_complete_45_key_config_pin() -> None:
    literal = {
        "schema": "opportunities-candidates-synthetic-config@1",
        "policy_version": "price-candidates-synthetic-v1",
        "contract_revision": "opportunities-candidates-synthetic@rev-2",
        "base_config_hash": BASE_CONFIG_SHA,
        "source_mode": "synthetic_demo",
        "evidence_grade": "research",
        "calendar": "XNYS",
        "calendar_start": "2023-01-01",
        "target_date": "2026-09-11",
        "required_closes": 757,
        "momentum_sessions": [252, 21],
        "recent_sessions": [5, 21],
        "drawdown_sessions": 252,
        "episode_lookback_sessions": 126,
        "episode_exclude_latest_sessions": 5,
        "trough_age_sessions": [5, 21],
        "episode_rule": "deepest_then_latest_trough_then_earliest_peak",
        "pullback_depth_bounds": ["0.10", "0.30"],
        "pullback_recovery_min": "0.05",
        "deep_depth_min": "0.30",
        "deep_recovery_min": "0.10",
        "recovery_rising_closes": 3,
        "exit_prior_low_sessions": 20,
        "exit_current_drawdown_max": "-0.10",
        "entry_relative_volatility_max": "2",
        "entry_maximum_drawdown_min": "-0.50",
        "entry_dollar_turnover_min": "5000000",
        "entry_target_close_min": "10",
        "price_arithmetic": "decimal_from_str_binary64_prec80_cross_products",
        "feature_decimal_places": 12,
        "rounding": "ROUND_HALF_EVEN",
        "arm_order": ["continuation", "positive_pullback", "deep_reversal"],
        "conflict_rule": "error_on_eligible_entry_and_matched_exit",
        "selection": "deferred",
        "future_maximum_per_list": 5,
        "outcome_horizons": [126, 252],
        "return_basis": "split_adjusted_price_return",
        "dividends_included": False,
        "exit_primary": "cash_minus_hold",
        "exit_secondary": "benchmark_minus_hold",
        "payoff_scope": "synthetic_terminal_algebra_only",
        "synthetic_cash_return": "0",
        "synthetic_differential_cost": "0",
        "synthetic_stock_terminal_returns": ["-0.20", "0", "0.20"],
        "synthetic_benchmark_terminal_return": "0.05",
    }
    assert len(literal) == 45
    assert hashlib.sha256(_canonical(literal)).hexdigest() == CONFIG_SHA
    assert json.loads(policy._canonical_bytes(asdict(policy._CONFIG))) == literal
    assert policy._CONFIG_SHA256 == CONFIG_SHA
    assert tuple(inspect.signature(policy.assess_synthetic_candidate).parameters) == (
        "candidate",
        "base_config",
    )
    assert tuple(inspect.signature(driver.run_synthetic_candidate_study).parameters) == (
        "base_config",
    )


def test_independent_driver_recipes_calendar_identities_and_full_hashes(
    candidates: tuple[policy.SyntheticCandidateInput, ...],
) -> None:
    anchors = (
        ((0, 10000), (504, 10000), (630, 12000), (735, 13000), (756, 13500)),
        (
            (0, 10000),
            (504, 10000),
            (630, 12000),
            (714, 15000),
            (735, 14000),
            (746, 12000),
            (756, 13500),
        ),
        (
            (0, 10000),
            (504, 15000),
            (630, 12000),
            (714, 12000),
            (735, 9500),
            (746, 7800),
            (756, 8800),
        ),
        ((0, 10000), (504, 14000), (630, 13000), (735, 11000), (755, 10000), (756, 9800)),
        (),
        (
            (0, 10000),
            (504, 15000),
            (630, 12000),
            (714, 12000),
            (735, 9500),
            (746, 7800),
            (756, 8800),
        ),
    )
    for n, (candidate, points) in enumerate(zip(candidates, anchors, strict=True), 1):
        p = candidate.product_input
        assert candidate.case_id == CASE_IDS[n - 1]
        assert candidate.security_type == "common_stock" and candidate.region == "us"
        assert p.listing_id == UUID(int=n)
        assert p.stock.identity.asset_id == UUID(int=100 + n)
        assert p.benchmark.identity.asset_id == UUID(int=200)
        assert p.stock.identity.subject == "SYN-" + CASE_IDS[n - 1].upper()
        assert p.benchmark.identity.subject == "SPY"
        assert len(p.calendar_sessions) == 757
        assert p.calendar_sessions[0] == date(2023, 9, 6)
        assert p.calendar_sessions[-1] == date(2026, 9, 11)
        assert (
            hashlib.sha256(
                "\n".join(d.isoformat() for d in p.calendar_sessions).encode("ascii")
            ).hexdigest()
            == "e92a988d222691ae2d5fc8dabe77b7f3ed2fc01446cf699b721128a22de55673"
        )
        assert p.decision_time == datetime(2026, 9, 11, 21, tzinfo=UTC)
        for series in (p.stock, p.benchmark):
            assert series.identity.provider == "synthetic_demo"
            assert series.identity.retrieved_at == datetime(2026, 9, 11, 20, 30, tzinfo=UTC)
            assert series.identity.available_at == series.identity.retrieved_at
            assert series.currency == "USD" and series.dates == p.calendar_sessions
            assert series.identity.sha256 == _series_hash(series)
        expected = []
        for i in range(757):
            if not points:
                expected.append(100.0)
                continue
            (a, ca), (b, cb) = next(
                (left, right)
                for left, right in zip(points, points[1:], strict=False)
                if left[0] <= i <= right[0]
            )
            cents = ca + ((cb - ca) * (i - a)) // (b - a) + (i % 2)
            expected.append(float(Decimal(cents).scaleb(-2)))
        assert p.stock.closes == tuple(expected)
        assert p.benchmark.closes == tuple(100.0 if i % 2 == 0 else 102.0 for i in range(757))
        assert p.stock.volumes == (None if n == 6 else (1_000_000.0,) * 757)
        assert p.stock.volume_adjustment_compatible is (n != 6)
    # A wider cached calendar cannot alter the bounded session identity.
    get_calendar("XNYS", start="2020-01-01", end="2027-12-31")
    assert driver._synthetic_candidates() == candidates


def test_all_six_cases_deep_independence_and_separate_drawdowns(
    result: driver.SyntheticCandidateStudy,
) -> None:
    assert tuple(c.case_id for c in result.cases) == CASE_IDS
    assert [tuple(a.entry_research_status for a in c.arms) for c in result.cases] == [
        ("eligible", "not_matched", "not_matched"),
        ("not_matched", "eligible", "not_matched"),
        ("not_matched", "not_matched", "eligible"),
        ("not_matched", "not_matched", "not_matched"),
        ("not_matched", "not_matched", "not_matched"),
        ("not_matched", "not_matched", "withheld"),
    ]
    assert [c.exit_review.status for c in result.cases] == [
        "not_matched",
        "not_matched",
        "not_matched",
        "matched",
        "not_matched",
        "not_matched",
    ]
    pullback, deep = result.cases[1:3]
    assert pullback.episode and deep.episode and deep.native_control and deep.features
    assert (pullback.episode.peak_index, pullback.episode.trough_index) == (714, 746)
    assert (deep.episode.peak_index, deep.episode.trough_index) == (631, 746)
    assert deep.native_control.recommendation.suggestion == "avoid"
    assert deep.native_control.risk.maximum_drawdown == -0.48
    assert deep.features.current_drawdown_252 == "-0.413333333333"
    assert deep.episode.depth == "0.350054162153"
    assert deep.episode.recovery == "0.128205128205"
    flat = result.cases[4]
    assert flat.episode is None and flat.entry_gates.status == "withheld"
    assert flat.native_control
    assert (
        "stock_volatility:filter_variance_degenerate"
        in flat.native_control.risk.insufficiency_reasons
    )


@pytest.mark.parametrize(
    ("index", "reason"),
    [
        (0, "recent_21_relative_not_positive"),
        (1, "recent_5_relative_not_positive"),
        (2, "recent_5_relative_not_positive"),
        (3, "recent_21_relative_not_negative"),
    ],
)
def test_public_benchmark_relative_only_failures(
    index: int,
    reason: str,
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    changed = _price(candidates[index], -1, 80.0 if index == 3 else 110.0, "benchmark")
    assessed = policy.assess_synthetic_candidate(changed, base_config=config)
    if index == 3:
        assert assessed.exit_review.reason == reason
    else:
        assert assessed.arms[index].pattern_reason == reason


def _momentum(direction: native.Direction = "positive") -> native.MomentumResult:
    # Constructed native scalar for predicate UNIT tests, not a price-history claim.
    return native.MomentumResult(0.1, 0.1, 0.0, 0.1, direction)


def _pattern_prices(trough: float = 80.0, target: float = 110.0) -> tuple[Decimal, ...]:
    p = [100.0] * 757
    p[746] = trough
    p[-6], p[-3], p[-2], p[-1] = target - 3, target - 2, target - 1, target
    return tuple(Decimal(str(x)) for x in p)


@pytest.mark.parametrize(
    ("arm", "trough_boundary", "toward", "matched"),
    [
        ("positive_pullback", 90.0, -math.inf, True),
        ("positive_pullback", 90.0, None, True),
        ("positive_pullback", 90.0, math.inf, False),
        ("positive_pullback", 70.0, -math.inf, False),
        ("positive_pullback", 70.0, None, True),
        ("positive_pullback", 70.0, math.inf, True),
        ("deep_reversal", 70.0, -math.inf, True),
        ("deep_reversal", 70.0, None, True),
        ("deep_reversal", 70.0, math.inf, False),
    ],
)
def test_exact_adjacent_cross_product_depth_boundaries(
    arm: str,
    trough_boundary: float,
    toward: float | None,
    matched: bool,
) -> None:
    trough = trough_boundary if toward is None else math.nextafter(trough_boundary, toward)
    p = _pattern_prices(trough)
    reason = policy._pattern_reason(
        arm,
        _momentum("negative" if arm == "deep_reversal" else "positive"),
        p,
        (Decimal(100),) * 757,
        (630, 746),
    )
    assert (reason == arm + "_pattern") is matched


@pytest.mark.parametrize(
    ("arm", "trough", "target"),
    [
        ("positive_pullback", 80.0, 84.0),
        ("deep_reversal", 60.0, 66.0),
    ],
)
@pytest.mark.parametrize("toward", [-math.inf, None, math.inf])
def test_exact_adjacent_recovery_boundaries_do_not_use_rounded_features(
    arm: str,
    trough: float,
    target: float,
    toward: float | None,
) -> None:
    value = target if toward is None else math.nextafter(target, toward)
    reason = policy._pattern_reason(
        arm,
        _momentum("negative" if arm == "deep_reversal" else "positive"),
        _pattern_prices(trough, value),
        (Decimal(100),) * 757,
        (630, 746),
    )
    assert (reason == arm + "_pattern") is (toward != -math.inf)
    with localcontext() as context:
        context.prec = 80
        rendered = policy._fixed12(Decimal(str(value)) / Decimal(str(trough)) - 1)
    assert rendered == ("0.050000000000" if arm == "positive_pullback" else "0.100000000000")


@pytest.mark.parametrize("arm", ["continuation", "positive_pullback", "deep_reversal"])
@pytest.mark.parametrize("toward", [-math.inf, None, math.inf])
def test_strict_relative_wealth_equality_and_adjacent_unit_boundaries(
    arm: str,
    toward: float | None,
) -> None:
    p = list(_pattern_prices(60.0 if arm == "deep_reversal" else 80.0))
    p[-6] = p[-22] = Decimal(100)
    b = [Decimal(100)] * 757
    b[-1] = Decimal(str(110.0 if toward is None else math.nextafter(110.0, toward)))
    reason = policy._pattern_reason(
        arm,
        _momentum("negative" if arm == "deep_reversal" else "positive"),
        tuple(p),
        tuple(b),
        (630, 746),
    )
    assert (reason == arm + "_pattern") is (toward == -math.inf)


@pytest.mark.parametrize("toward", [-math.inf, None, math.inf])
def test_three_rising_closes_strict_equality(toward: float | None) -> None:
    p = list(_pattern_prices())
    p[-2] = Decimal(str(110.0 if toward is None else math.nextafter(110.0, toward)))
    reason = policy._pattern_reason(
        "positive_pullback", _momentum(), tuple(p), (Decimal(100),) * 757, (630, 746)
    )
    assert (reason == "positive_pullback_pattern") is (toward == -math.inf)


def test_recovery_requires_recent_absolute_strength_and_first_failure_order() -> None:
    p = list(_pattern_prices())
    p[-6] = p[-1]
    p[-2] = p[-1]  # This also fails rising closes, but R5 is the earlier failure.
    reason = policy._pattern_reason(
        "positive_pullback", _momentum(), tuple(p), (Decimal(100),) * 757, (630, 746)
    )
    assert reason == "recent_5_not_positive"
    assert (
        policy._pattern_reason(
            "positive_pullback", _momentum(), tuple(p), (Decimal(100),) * 757, None
        )
        == "decline_episode_absent"
    )
    assert (
        policy._pattern_reason(
            "continuation", _momentum(), (Decimal(100),) * 757, (Decimal(100),) * 757, None
        )
        == "recent_21_not_positive"
    )


def test_features_are_relative_wealth_not_percentage_point_outperformance(
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    case = _price(candidates[0], -1, 110.0, "benchmark")
    r = policy.assess_synthetic_candidate(case, base_config=config)
    assert r.features
    p, b = case.product_input.stock.closes, case.product_input.benchmark.closes
    with localcontext() as context:
        context.prec = 80
        for k in (5, 21):
            stock = Decimal(str(p[-1])) / Decimal(str(p[-1 - k])) - 1
            bench = Decimal(str(b[-1])) / Decimal(str(b[-1 - k])) - 1
            relative = (stock + 1) / (bench + 1) - 1
            assert getattr(r.features, f"relative_return_{k}") == format(
                relative.quantize(Decimal("0.000000000001")), "f"
            )
            assert Decimal(getattr(r.features, f"relative_return_{k}")) != stock - bench
    assert r.features.prior_low_20 == str(Decimal(str(min(p[-21:-1]))))


def test_episode_strict_chronology_ties_and_no_recent_fallback() -> None:
    p = [Decimal(100)] * 757
    assert policy._episode_indices(tuple(p)) is None
    p[650] = Decimal(80)
    p[660:] = [Decimal(200)] * (757 - 660)
    p[710] = Decimal(160)
    assert policy._episode_indices(tuple(p)) == (660, 710)  # Equal depth: later trough.
    p[711] = Decimal(160)
    assert policy._episode_indices(tuple(p)) == (660, 711)  # Earliest maximum is still 660.
    p[700] = Decimal(100)
    assert policy._episode_indices(tuple(p)) == (660, 700)  # Deepest, not newest decline.
    reason = policy._pattern_reason(
        "positive_pullback",
        _momentum(),
        tuple(p),
        (Decimal(100),) * 757,
        policy._episode_indices(tuple(p)),
    )
    assert reason == "trough_age_outside_5_21"
    # The post-T-5 deepest lows are OUTSIDE the search, not a fabricated age<5 case.
    p = [Decimal(100)] * 757
    p[751] = Decimal(80)
    p[752:] = [Decimal(10)] * 5
    assert policy._episode_indices(tuple(p)) == (630, 751)


def test_running_scan_matches_literal_ordered_definition() -> None:
    # Deterministic integer fixtures exercise ties, not forecasting performance.
    rng = np.random.Generator(np.random.PCG64(42))
    for _ in range(8):
        prices = tuple(Decimal(int(x)) for x in rng.integers(50, 151, size=757))
        selected = None
        for j in range(631, 752):
            peak = max(range(630, j), key=lambda i: prices[i])  # earliest max
            if prices[j] >= prices[peak]:
                continue
            if selected is None:
                selected = peak, j
            else:
                p, q = selected
                if (prices[peak] - prices[j]) * prices[p] >= (prices[p] - prices[q]) * prices[peak]:
                    selected = peak, j
        assert policy._episode_indices(prices) == selected
        assert selected and 630 <= selected[0] < selected[1] <= 751


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (5, "positive_pullback_pattern"),
        (21, "positive_pullback_pattern"),
        (22, "trough_age_outside_5_21"),
    ],
)
def test_episode_age_inclusive_public_search_bounds(age: int, expected: str) -> None:
    p = [Decimal(100)] * 757
    p[756 - age] = Decimal(80)
    p[-3:] = [Decimal(108), Decimal(109), Decimal(110)]
    indices = policy._episode_indices(tuple(p))
    assert indices == (630, 756 - age)
    assert (
        policy._pattern_reason(
            "positive_pullback", _momentum(), tuple(p), (Decimal(100),) * 757, indices
        )
        == expected
    )


@pytest.mark.parametrize("toward", [-math.inf, None, math.inf])
def test_exit_current_drawdown_exact_adjacent_and_prior_low_excludes_today(
    toward: float | None,
) -> None:
    p = [Decimal(100)] * 757
    p[-1] = Decimal(str(90.0 if toward is None else math.nextafter(90.0, toward)))
    assert policy._exit_review(tuple(p), (Decimal(100),) * 757).status == (
        "not_matched" if toward == math.inf else "matched"
    )
    p[-2] = p[-1]
    assert policy._exit_review(tuple(p), (Decimal(100),) * 757).reason == "prior_20_low_not_broken"


@pytest.mark.parametrize("toward", [-math.inf, None, math.inf])
def test_exit_strict_relative_equality(toward: float | None) -> None:
    p, b = [Decimal(100)] * 757, [Decimal(100)] * 757
    p[-1] = Decimal(80)
    b[-1] = Decimal(str(80.0 if toward is None else math.nextafter(80.0, toward)))
    assert (policy._exit_review(tuple(p), tuple(b)).status == "matched") is (toward == math.inf)


def _risk() -> RiskResult:
    return RiskResult(0.2, 0.2, 1.0, "low", -0.2, 10_000_000.0, ())


@pytest.mark.parametrize(
    ("field", "boundary", "failed_toward", "reason"),
    [
        ("relative_volatility", 2.0, math.inf, "relative_volatility_above_buy_limit"),
        (
            "average_dollar_turnover_20d",
            5_000_000.0,
            -math.inf,
            "dollar_turnover_below_buy_minimum",
        ),
        ("maximum_drawdown", -0.5, -math.inf, "drawdown_below_buy_limit"),
        ("target_close", 10.0, -math.inf, "target_close_below_buy_minimum"),
    ],
)
@pytest.mark.parametrize("toward", [-math.inf, None, math.inf])
def test_native_scalar_entry_gate_exact_adjacent_boundaries(
    field: str,
    boundary: float,
    failed_toward: float,
    reason: str,
    toward: float | None,
) -> None:
    value = boundary if toward is None else math.nextafter(boundary, toward)
    changes: dict[str, Any] = {} if field == "target_close" else {field: value}
    risk = replace(_risk(), **changes)
    gate = policy._entry_gates(risk, value if field == "target_close" else 100.0)
    assert gate.status == ("blocked" if toward == failed_toward else "pass")
    assert gate.reason_codes == ((reason,) if toward == failed_toward else ())


def test_unavailable_gate_precedence_keeps_all_known_failures() -> None:
    risk = replace(
        _risk(), relative_volatility=None, maximum_drawdown=-0.6, average_dollar_turnover_20d=None
    )
    assert policy._entry_gates(risk, 9.0) == policy.EntryGates(
        "withheld",
        (
            "relative_volatility_unavailable",
            "drawdown_below_buy_limit",
            "dollar_turnover_unavailable",
            "target_close_below_buy_minimum",
        ),
    )


def test_deeper_decline_is_not_a_risk_exemption(
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    result = policy.assess_synthetic_candidate(
        _price(candidates[2], 504, 200.0), base_config=config
    )
    assert result.arms[2].pattern_status == "matched"
    assert result.arms[2].entry_research_status == "blocked"
    assert "drawdown_below_buy_limit" in result.entry_gates.reason_codes
    assert result.native_control and result.native_control.recommendation.suggestion == "avoid"


@pytest.mark.parametrize(
    ("volume", "compatible", "status", "native_reason"),
    [
        (None, False, "withheld", "dollar_turnover_volume_missing"),
        ((1_000_000.0,) * 757, False, "withheld", "dollar_turnover_adjustment_incompatible"),
        ((0.0,) * 757, True, "blocked", None),
        ((1_000_000.0,) * 756 + (None,), True, "withheld", "dollar_turnover_volume_missing"),
    ],
)
def test_missing_incompatible_and_verified_zero_turnover_are_distinct(
    volume: tuple[float | None, ...] | None,
    compatible: bool,
    status: str,
    native_reason: str | None,
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    case = _replace_series(candidates[2], volumes=volume, volume_adjustment_compatible=compatible)
    r = policy.assess_synthetic_candidate(case, base_config=config)
    assert r.arms[2].pattern_status == "matched" and r.arms[2].entry_research_status == status
    assert r.native_control
    if native_reason:
        assert r.native_control.risk.average_dollar_turnover_20d is None
        assert native_reason in r.native_control.risk.insufficiency_reasons
    else:
        assert r.native_control.risk.average_dollar_turnover_20d == 0.0
        assert r.entry_gates.reason_codes == ("dollar_turnover_below_buy_minimum",)
    assert r.exit_review.status == "not_matched"  # AVOID/missing volume are not exit evidence.


def test_exit_independent_of_under_ten_liquidity_and_entry_selection(
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    p = candidates[3].product_input
    case = _replace_series(
        candidates[3],
        closes=tuple(x / 16 for x in p.stock.closes),
        volumes=None,
        volume_adjustment_compatible=False,
    )
    r = policy.assess_synthetic_candidate(case, base_config=config)
    assert r.exit_review.status == "matched"
    assert all(a.entry_research_status != "eligible" for a in r.arms)
    assert r.entry_gates.reason_codes == (
        "dollar_turnover_unavailable",
        "target_close_below_buy_minimum",
    )


def test_overlap_is_one_listing_and_conflict_is_an_error(
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    case = _price(candidates[1], 735, 130.0)  # positive 21d return, same valid episode
    r = policy.assess_synthetic_candidate(case, base_config=config)
    assert [a.entry_research_status for a in r.arms[:2]] == ["eligible", "eligible"]
    assert r.listing_id == case.product_input.listing_id
    assert r.exit_review.status == "not_matched"
    with pytest.raises(PriceProductInputError) as error:
        policy._check_conflict(
            r.arms, policy.ExitReview("matched", "affirmative_deterioration_pattern")
        )
    assert error.value.reason_code == "candidate_policy_conflict"


def test_entry_exit_mutual_exclusion_for_many_price_predicates() -> None:
    rng = np.random.Generator(np.random.PCG64(81))
    for _ in range(24):
        p = tuple(Decimal(int(x)) for x in rng.integers(70, 121, size=757))
        b = (Decimal(100),) * 757
        for direction in ("positive", "negative", "mixed"):
            momentum = _momentum(direction)
            arms = policy._arms(
                momentum, p, b, policy._episode_indices(p), policy.EntryGates("pass", ())
            )
            exit_review = policy._exit_review(p, b)
            policy._check_conflict(arms, exit_review)
            assert not (
                exit_review.status == "matched"
                and any(a.entry_research_status == "eligible" for a in arms)
            )


@pytest.mark.parametrize("scale", [2.0, 0.0625])
def test_split_equivalent_scaling_and_fixed_volume_counterexample(
    scale: float,
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    case = candidates[2]
    p = case.product_input
    before = policy.assess_synthetic_candidate(case, base_config=config)
    scaled = _replace_series(
        case,
        closes=tuple(x * scale for x in p.stock.closes),
        volumes=(1_000_000.0 / scale,) * 757,
    )
    after = policy.assess_synthetic_candidate(scaled, base_config=config)
    assert [(a.pattern_status, a.pattern_reason) for a in before.arms] == [
        (a.pattern_status, a.pattern_reason) for a in after.arms
    ]
    assert before.native_control and after.native_control
    turnover = before.native_control.risk.average_dollar_turnover_20d
    assert turnover is not None
    assert after.native_control.risk.average_dollar_turnover_20d == turnover
    assert before.features and after.features
    for name in asdict(before.features):
        if name != "prior_low_20":
            assert getattr(before.features, name) == getattr(after.features, name)
    assert after.entry_gates.reason_codes == (
        () if scale == 2.0 else ("target_close_below_buy_minimum",)
    )
    fixed = policy.assess_synthetic_candidate(
        _replace_series(scaled, volumes=(1_000_000.0,) * 757), base_config=config
    )
    assert fixed.native_control
    assert fixed.native_control.risk.average_dollar_turnover_20d == pytest.approx(turnover * scale)


@pytest.mark.parametrize("kind", ["etf", "region", "source", "all"])
def test_global_withholding_follows_structural_validation_but_precedes_numeric_work(
    kind: str,
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = candidates[0]
    expected = []
    if kind in ("etf", "all"):
        c = replace(c, security_type="etf")
        expected.append("unsupported_security_type")
    if kind in ("region", "all"):
        c = replace(c, region="other")
        expected.append("unsupported_region")
    if kind in ("source", "all"):
        c = replace(c, product_input=replace(c.product_input, source_eligible=False))
        expected.append("source_ineligible")
    c = _price(_price(c, 504, 1e308), 735, 1e-308)
    monkeypatch.setattr(policy, "_native_control", _refuse)
    r = policy.assess_synthetic_candidate(c, base_config=config)
    assert r.status == "withheld" and r.root_reasons == tuple(expected)
    assert r.native_control is None and r.features is None and r.episode is None
    assert all(a.pattern_reason == "case_withheld" for a in r.arms)
    assert r.exit_review == policy.ExitReview("withheld", "case_withheld")


@pytest.mark.parametrize(
    ("mode", "grade"),
    [
        ("provider", "research"),
        ("provider", "observed"),
        ("synthetic_demo", "observed"),
    ],
)
def test_mode_grade_rejection(
    mode: str,
    grade: str,
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    c = candidates[0]
    p = replace(c.product_input, source_execution=native.SourceExecutionBinding(mode, grade))  # type: ignore[arg-type]
    with pytest.raises(PriceProductInputError, match="Synthetic research") as error:
        policy.assess_synthetic_candidate(replace(c, product_input=p), base_config=config)
    assert error.value.reason_code == "synthetic_input_required"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("closes", (True,) * 757, "synthetic_input_type_invalid"),
        ("closes", (100,) * 757, "synthetic_input_type_invalid"),
        ("closes", [100.0] * 757, "synthetic_input_type_invalid"),
        ("closes", (0.0,) * 757, "stock_close_invalid"),
        ("closes", (-1.0,) * 757, "stock_close_invalid"),
        ("closes", (float("inf"),) * 757, "stock_close_invalid"),
        ("closes", (float("nan"),) * 757, "stock_close_invalid"),
        ("closes", (100.0,) * 756, "stock_history_length"),
        ("currency", "EUR", "stock_currency_invalid"),
        ("volumes", (-1.0,) * 757, "stock_volume_invalid"),
        ("volumes", (float("inf"),) * 757, "stock_volume_invalid"),
        ("volumes", (True,) * 757, "synthetic_input_type_invalid"),
        ("volumes", (10.0,), "stock_volume_length"),
        ("volume_adjustment_compatible", 1, "synthetic_input_type_invalid"),
    ],
)
def test_series_type_native_admission_and_no_fingerprint_bypass(
    field: str,
    value: Any,
    reason: str,
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    c = candidates[0]
    p = replace(c.product_input, stock=replace(c.product_input.stock, **{field: value}))
    with pytest.raises(PriceProductInputError) as error:
        policy.assess_synthetic_candidate(replace(c, product_input=p), base_config=config)
    assert error.value.reason_code == reason


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("provider", "twelve_data", "synthetic_input_required"),
        ("asset_id", UUID(int=0), "synthetic_identity_invalid"),
        ("asset_id", "not-a-uuid", "synthetic_identity_invalid"),
        ("asset_id", UUID(int=200), "synthetic_identity_invalid"),
        ("subject", "REAL", "synthetic_identity_invalid"),
        ("sha256", "f" * 64, "synthetic_source_hash_mismatch"),
        ("retrieved_at", datetime(2026, 9, 11, 20, 30), "synthetic_time_invalid"),
        ("available_at", datetime(2026, 9, 11, 21, 0, 1, tzinfo=UTC), "stock_asset_after_decision"),
        ("retrieved_at", datetime(2026, 9, 11, 21, 0, 1, tzinfo=UTC), "stock_asset_after_decision"),
    ],
)
def test_source_identity_and_availability(
    field: str,
    value: Any,
    reason: str,
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    c = candidates[0]
    p = c.product_input
    altered = replace(p.stock, identity=replace(p.stock.identity, **{field: value}))
    with pytest.raises(PriceProductInputError) as error:
        policy.assess_synthetic_candidate(
            replace(c, product_input=replace(p, stock=altered)), base_config=config
        )
    assert error.value.reason_code == reason


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_calendar",
        "duplicate",
        "future",
        "future_target",
        "benchmark_dates",
        "decision",
        "naive_decision",
        "nil_listing",
        "case_id",
        "benchmark_subject",
        "config",
    ],
)
def test_context_calendar_config_and_temporal_rejection(
    fault: str,
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    c = candidates[0]
    p = c.product_input
    reason = ""
    if fault in ("wrong_calendar", "duplicate"):
        first = (
            p.calendar_sessions[0] - timedelta(days=1)
            if fault == "wrong_calendar"
            else p.calendar_sessions[1]
        )
        dates = (first, *p.calendar_sessions[1:])
        c = _replace_series(_replace_series(c, dates=dates), "benchmark", dates=dates)
        p = replace(c.product_input, calendar_sessions=dates)
        reason = (
            "synthetic_calendar_mismatch"
            if fault == "wrong_calendar"
            else "calendar_sessions_invalid"
        )
    elif fault == "future":
        p = replace(p, stock=replace(p.stock, dates=(*p.stock.dates[:-1], date(2026, 9, 12))))
        reason = "stock_target_mismatch"
    elif fault == "future_target":
        p = replace(p, target_date=date(2026, 9, 12))
        reason = "target_after_decision_time"
    elif fault == "benchmark_dates":
        p = replace(
            p, benchmark=replace(p.benchmark, dates=(date(2023, 9, 5), *p.benchmark.dates[1:]))
        )
        reason = "stock_benchmark_dates_mismatch"
    elif fault in ("decision", "naive_decision"):
        instant = (
            p.decision_time + timedelta(seconds=1)
            if fault == "decision"
            else p.decision_time.replace(tzinfo=None)
        )
        p = replace(p, decision_time=instant)
        reason = "synthetic_time_invalid"
    elif fault == "nil_listing":
        p = replace(p, listing_id=UUID(int=0))
        reason = "synthetic_identity_invalid"
    elif fault == "case_id":
        c = replace(c, case_id="../invalid")
        reason = "synthetic_input_type_invalid"
    elif fault == "benchmark_subject":
        p = replace(
            p,
            benchmark=replace(p.benchmark, identity=replace(p.benchmark.identity, subject="WRONG")),
        )
        reason = "benchmark_identity_mismatch"
    else:
        config = replace(config, currency="EUR")
        reason = "frozen_config_mismatch"
    with pytest.raises(PriceProductInputError) as error:
        policy.assess_synthetic_candidate(replace(c, product_input=p), base_config=config)
    assert error.value.reason_code == reason


@pytest.mark.parametrize("role", ["stock", "benchmark"])
def test_exact_zero_native_momentum_quotient_new_error_escapes(
    role: str,
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    c = _price(_price(candidates[0], 504, 1e308, role), 735, 1e-308, role)
    with pytest.raises(PriceProductInputError) as error:
        policy.assess_synthetic_candidate(c, base_config=config)
    assert error.value.reason_code == "synthetic_numeric_unrepresentable"
    assert str(error.value) == (
        "Synthetic candidate arithmetic cannot be represented by the frozen numeric contract."
    )


def test_only_native_risk_accumulation_overflow_translated(
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    c = _replace_series(candidates[4], volumes=(1e305,) * 757, volume_adjustment_compatible=True)
    products = [
        p * v for p, v in zip(c.product_input.stock.closes[-20:], (1e305,) * 20, strict=True)
    ]
    assert all(math.isfinite(x) for x in products)
    with pytest.raises(OverflowError):
        math.fsum(products)
    with pytest.raises(PriceProductInputError) as error:
        policy.assess_synthetic_candidate(c, base_config=config)
    assert error.value.reason_code == "synthetic_numeric_unrepresentable"


def test_fixed12_error_occurs_in_assessment_not_only_serializer(
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    c = _price(candidates[4], -1, 1e100)
    with pytest.raises(PriceProductInputError) as error:
        policy.assess_synthetic_candidate(c, base_config=config)
    assert error.value.reason_code == "synthetic_numeric_unrepresentable"


def test_fixed12_representability_half_even_and_rounding_carry_boundary() -> None:
    largest = "9" * 68 + "." + "9" * 12
    assert policy._fixed12(Decimal(largest)) == largest
    assert policy._fixed12(Decimal(largest + "4")) == largest
    assert policy._fixed12(Decimal("0.0000000000005")) == "0.000000000000"
    assert policy._fixed12(Decimal("0.0000000000015")) == "0.000000000002"
    with localcontext() as context:
        context.traps[InvalidOperation] = False
        with pytest.raises(PriceProductInputError) as error:
            policy._fixed12(Decimal(largest + "5"))
    assert error.value.reason_code == "synthetic_numeric_unrepresentable"


def test_small_nonzero_quotient_and_uniform_large_prices_not_arbitrarily_rejected(
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    small = _price(candidates[4], 504, 1e100)
    r = policy.assess_synthetic_candidate(small, base_config=config)
    assert r.native_control and r.native_control.momentum
    assert r.native_control.momentum.stock_log_momentum < -200
    large = _replace_series(candidates[4], closes=(1e100,) * 757)
    r = policy.assess_synthetic_candidate(large, base_config=config)
    assert r.native_control
    assert (
        "stock_volatility:filter_variance_degenerate" in r.native_control.risk.insufficiency_reasons
    )
    assert r.features and r.features.prior_low_20 == "1E+100"


@pytest.mark.parametrize(
    "code", ["momentum_unrepresentable", "stock_momentum_invalid", "benchmark_momentum_invalid"]
)
def test_native_momentum_reason_propagates_verbatim_exit_stays_independent(
    code: str,
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args: object, **kwargs: object) -> Never:
        raise PriceProductInputError(code, "Momentum calculation produced a non-finite value")

    monkeypatch.setattr(policy, "calculate_momentum", missing)
    r = policy.assess_synthetic_candidate(candidates[3], base_config=config)
    assert r.native_control and r.native_control.momentum is None
    assert r.native_control.momentum_reason == code
    assert r.native_control.recommendation.blocking_reasons == (code,)
    assert all(a.pattern_reason == "momentum_unavailable" for a in r.arms)
    assert all(a.entry_research_status == "withheld" for a in r.arms)
    assert r.exit_review.status == "matched"


def test_native_controlled_momentum_overflow_not_mislabeled(
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    c = _price(_price(candidates[0], 504, 1e-308), 735, 1e308)
    with pytest.warns(RuntimeWarning, match="overflow encountered in scalar divide"):
        r = policy.assess_synthetic_candidate(c, base_config=config)
    assert r.native_control and r.native_control.momentum is None
    assert r.native_control.momentum_reason == "stock_momentum_invalid"


def test_native_nonfinite_turnover_remains_withheld(
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    c = _replace_series(candidates[2], volumes=(1e308,) * 757)
    r = policy.assess_synthetic_candidate(c, base_config=config)
    assert r.native_control
    assert r.native_control.risk.average_dollar_turnover_20d is None
    assert "dollar_turnover_nonfinite" in r.native_control.risk.insufficiency_reasons
    assert r.arms[2].entry_research_status == "withheld"


def test_benchmark_filter_insufficiency_and_native_reasons_retained(
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    c = _replace_series(candidates[0], "benchmark", closes=(100.0,) * 757)
    r = policy.assess_synthetic_candidate(c, base_config=config)
    assert r.native_control
    assert (
        "benchmark_volatility:filter_variance_degenerate"
        in r.native_control.risk.insufficiency_reasons
    )
    assert r.native_control.risk.relative_volatility is None
    assert r.arms[0].pattern_status == "matched"
    assert r.arms[0].entry_research_status == "withheld"


@pytest.mark.parametrize(
    ("helper", "exception"),
    [
        ("calculate_momentum", ValueError),
        ("filter_historical_returns", OverflowError),
        ("calculate_risk", ValueError),
    ],
)
def test_unrelated_exceptions_not_blanket_translated(
    helper: str,
    exception: type[Exception],
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> Never:
        raise exception("unrelated native failure")

    monkeypatch.setattr(policy, helper, fail)
    with pytest.raises(exception, match="unrelated native failure") as error:
        policy.assess_synthetic_candidate(candidates[0], base_config=config)
    assert type(error.value) is exception


def test_complete_policy_hash_sensitivity_and_source_consistency(
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
) -> None:
    c = candidates[0]
    initial = policy.assess_synthetic_candidate(c, base_config=config)
    expected = {
        "domain": "candidate-synthetic-input@1",
        "complete_input_hash": native.complete_input_hash(c.product_input),
        "case_id": c.case_id,
        "security_type": "common_stock",
        "region": "us",
        "policy_version": "price-candidates-synthetic-v1",
        "config_sha256": CONFIG_SHA,
    }
    assert initial.input_hash == hashlib.sha256(_canonical(expected)).hexdigest()
    changes = (
        _price(c, 0, math.nextafter(c.product_input.stock.closes[0], math.inf)),
        _replace_series(c, volumes=(1_000_001.0,) * 757),
        _replace_series(c, volume_adjustment_compatible=False),
        _replace_series(
            c,
            identity=replace(
                c.product_input.stock.identity,
                retrieved_at=c.product_input.stock.identity.retrieved_at - timedelta(seconds=1),
            ),
        ),
        replace(c, case_id="another_case"),
        replace(c, security_type="adr"),
        replace(c, region="other"),
        replace(c, product_input=replace(c.product_input, listing_id=UUID(int=1000))),
    )
    for changed in changes:
        assessed = policy.assess_synthetic_candidate(changed, base_config=config)
        assert assessed.input_hash != initial.input_hash
    forged = replace(
        c,
        product_input=replace(
            c.product_input,
            stock=replace(
                c.product_input.stock, closes=(100.01, *c.product_input.stock.closes[1:])
            ),
        ),
    )
    with pytest.raises(PriceProductInputError) as error:
        policy.assess_synthetic_candidate(forged, base_config=config)
    assert error.value.reason_code == "synthetic_source_hash_mismatch"


def _execution() -> driver.SyntheticExecutionIdentity:
    # Declared synthetic serialization-unit identity, NOT an actual source attestation.
    return driver.SyntheticExecutionIdentity(
        None, "a" * 64, "b" * 64, "3.13.0", "2.5.2", "4.13.1", "synthetic-test-arm64"
    )


def test_payoffs_are_six_disconnected_ordered_hypothetical_rows(
    result: driver.SyntheticCandidateStudy,
) -> None:
    expected = [
        ("-0.200000000000", "0.200000000000", "0.250000000000", "0.200000000000", "0.000000000000"),
        ("0.000000000000", "0.000000000000", "0.050000000000", "0.000000000000", "0.000000000000"),
        (
            "0.200000000000",
            "-0.200000000000",
            "-0.150000000000",
            "0.000000000000",
            "0.200000000000",
        ),
    ]
    assert len(result.exit_payoff_examples) == 6
    for index, payoff in enumerate(result.exit_payoff_examples):
        assert payoff.horizon_sessions == (126 if index < 3 else 252)
        assert (
            payoff.stock_terminal_return,
            payoff.cash_minus_hold,
            payoff.benchmark_minus_hold,
            payoff.loss_avoided,
            payoff.foregone_upside,
        ) == expected[index % 3]
        assert payoff.cash_return == payoff.differential_cost == "0.000000000000"
        assert payoff.benchmark_terminal_return == "0.050000000000"
        assert set(asdict(payoff)) == {
            "horizon_sessions",
            "stock_terminal_return",
            "cash_return",
            "benchmark_terminal_return",
            "differential_cost",
            "cash_minus_hold",
            "benchmark_minus_hold",
            "loss_avoided",
            "foregone_upside",
        }


def test_canonical_complete_shape_hash_honesty_and_execution_binding(
    result: driver.SyntheticCandidateStudy,
) -> None:
    encoded = driver.serialize_synthetic_candidate_study(result, execution_identity=_execution())
    document = json.loads(encoded)
    assert _canonical(document) == encoded
    assert set(document) == {
        "schema",
        "contract_revision",
        "policy_version",
        "config_sha256",
        "base_config_hash",
        "execution_identity",
        "claim_status",
        "selection",
        "real_evaluation",
        "cases",
        "exit_payoff_examples",
        "report_sha256",
    }
    assert document["config_sha256"] == CONFIG_SHA
    assert document["base_config_hash"] == BASE_CONFIG_SHA
    assert document["claim_status"] == "synthetic_correctness_only"
    assert document["selection"] == {
        "status": "deferred",
        "future_limit": 5,
        "buy": None,
        "sell_review": None,
    }
    assert document["real_evaluation"] == {
        "status": "blocked",
        "reasons": [
            "real_data_adapter_not_authorized",
            "untouched_confirmation_not_established",
            "confirmation_protocol_not_frozen",
        ],
    }
    assert document["execution_identity"]["code_revision"] is None
    digest = document.pop("report_sha256")
    assert digest == hashlib.sha256(_canonical(document)).hexdigest()
    for case in document["cases"]:
        assert set(case) == {
            "policy_version",
            "config_sha256",
            "input_hash",
            "case_id",
            "listing_id",
            "target_date",
            "source_scope",
            "status",
            "root_reasons",
            "native_control",
            "features",
            "episode",
            "entry_gates",
            "arms",
            "exit_review",
        }
        assert case["source_scope"] == "synthetic_only"
        assert set(case["native_control"]) == {
            "momentum",
            "momentum_reason",
            "risk",
            "recommendation",
        }
        assert len(case["arms"]) == 3
    for field, value in (
        ("code_revision", "c" * 40),
        ("source_fingerprint_sha256", "c" * 64),
        ("dependency_lock_sha256", "c" * 64),
        ("python_version", "3.13.1"),
        ("numpy_version", "2.5.3"),
        ("exchange_calendars_version", "4.13.2"),
        ("platform_architecture", "synthetic-test-x86_64"),
    ):
        other = driver.serialize_synthetic_candidate_study(
            result, execution_identity=replace(_execution(), **{field: value})
        )
        assert other != encoded
    assert driver.SOURCE_MANIFEST_PATHS == (
        "config/scoring/research-product-v1.yml",
        "pyproject.toml",
        "src/stanstock/research/price_candidate_policy.py",
        "src/stanstock/research/price_candidate_policy_study.py",
        "src/stanstock/research/price_product.py",
        "src/stanstock/research/price_product_config.py",
        "uv.lock",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("code_revision", "working-tree"),
        ("source_fingerprint_sha256", "invalid"),
        ("platform_architecture", "/private/example"),
        ("python_version", ""),
    ],
)
def test_serializer_rejects_malformed_or_path_bearing_identity(
    field: str,
    value: str,
    result: driver.SyntheticCandidateStudy,
) -> None:
    with pytest.raises(ValueError):
        driver.serialize_synthetic_candidate_study(
            result, execution_identity=replace(_execution(), **{field: value})
        )


def test_serializer_cannot_emit_partial_cases_or_connected_payoff_substitutes(
    result: driver.SyntheticCandidateStudy,
) -> None:
    for changed in (
        replace(result, cases=result.cases[:-1]),
        replace(result, cases=tuple(reversed(result.cases))),
        replace(result, exit_payoff_examples=result.exit_payoff_examples[:-1]),
    ):
        with pytest.raises(ValueError):
            driver.serialize_synthetic_candidate_study(changed, execution_identity=_execution())


def _immutable_tree(value: Any) -> None:
    if is_dataclass(value):
        assert cast(Any, value).__dataclass_params__.frozen
        for field in fields(value):
            _immutable_tree(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            _immutable_tree(item)
    else:
        assert value is None or type(value) in (str, int, float, bool, UUID, date, datetime)


def test_repeat_interleave_no_mutation_or_numeric_state_change(
    candidates: tuple[policy.SyntheticCandidateInput, ...],
    config: PriceProductConfig,
    result: driver.SyntheticCandidateStudy,
) -> None:
    before = tuple(asdict(c) for c in candidates)
    decimal_before = getcontext().copy()
    numpy_before = np.geterr()
    _immutable_tree(result)
    with pytest.raises(FrozenInstanceError):
        result.cases[0].case_id = "changed"  # type: ignore[misc]
    for c, expected in zip(reversed(candidates), reversed(result.cases), strict=True):
        assert policy.assess_synthetic_candidate(c, base_config=config) == expected
    assert driver.run_synthetic_candidate_study(base_config=config) == result
    with localcontext() as context:
        context.prec = 9
        context.rounding = ROUND_HALF_EVEN
        assert driver.run_synthetic_candidate_study(base_config=config) == result
    assert tuple(asdict(c) for c in candidates) == before
    assert getcontext().prec == decimal_before.prec
    assert getcontext().rounding == decimal_before.rounding
    assert np.geterr() == numpy_before


def test_no_forbidden_work_in_assessment_driver_or_serializer(
    config: PriceProductConfig,
    result: driver.SyntheticCandidateStudy,
) -> None:
    for module in (policy, driver):
        tree = ast.parse(inspect.getsource(module))
        imports: list[str | None] = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imports.extend(alias.name for alias in n.names)
            elif isinstance(n, ast.ImportFrom):
                imports.append(n.module)
        assert not any(
            m
            and m.startswith(
                ("django", "stanstock.data", "os", "pathlib", "time", "subprocess", "socket")
            )
            for m in imports
        )
        assert not any(
            isinstance(n, ast.Attribute)
            and n.attr
            in (
                "now",
                "today",
                "utcnow",
                "objects",
                "read_bytes",
                "write_bytes",
                "environ",
            )
            for n in ast.walk(tree)
        )
    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (builtins, "open"),
            (io, "open"),
            (os, "getenv"),
            (subprocess, "run"),
            (time, "time"),
            (native, "calculate_price_product"),
            (native, "project_fhs"),
            (native, "simulate_fhs_terminal_logs"),
        ):
            patch.setattr(owner, name, _refuse)
        patch.setattr(os, "environ", {})
        actual = driver.run_synthetic_candidate_study(base_config=config)
        encoded = driver.serialize_synthetic_candidate_study(
            actual, execution_identity=_execution()
        )
    assert actual == result and json.loads(encoded)["claim_status"] == "synthetic_correctness_only"
    # No django_db marker: ORM access is also forbidden by pytest-django.


def test_fresh_import_and_invocation_preserve_full_native_success_and_withholding() -> None:
    # Fresh interpreter, existing native fixture, complete byte comparisons;
    # no new frozen-math harness or replacement of the historical probe baseline.
    script = """
import pickle
from dataclasses import asdict
import numpy as np
from decimal import getcontext
from test_research_price_product import _product_input
from stanstock.research.price_product import calculate_price_product
from stanstock.research.price_product_config import load_price_product_config
config = load_price_product_config()
cases = (_product_input(), _product_input(stock_closes=(100.0,) * 757))
before = tuple(calculate_price_product(c, config=config) for c in cases)
assert before[0].forecast.insufficiency_reason is None
assert before[1].forecast.insufficiency_reason == 'filter_variance_degenerate'
payloads = tuple(pickle.dumps(asdict(r)) for r in before)
config_bytes = pickle.dumps(config)
context = str(getcontext())
numeric = np.geterr()
from stanstock.research.price_candidate_policy_study import run_synthetic_candidate_study
run_synthetic_candidate_study(base_config=config)
after = tuple(calculate_price_product(c, config=config) for c in cases)
assert tuple(pickle.dumps(asdict(r)) for r in after) == payloads
assert pickle.dumps(config) == config_bytes
assert str(getcontext()) == context
assert np.geterr() == numeric
"""
    root = Path(__file__).resolve().parents[1]
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("STANSTOCK_", "DJANGO_", "PYTEST_"))
    }
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root / "tests")))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, "Fresh native preservation probe failed"

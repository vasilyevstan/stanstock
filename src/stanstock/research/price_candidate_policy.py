"""Fixed synthetic candidate hypotheses, NOT investment recommendations.

No provider provenance is authenticated by synthetic labels or checksums.
Only internally constructed synthetic fixtures are appropriate for this
research boundary. Native v1 arithmetic is a labelled control, never an entry
veto. No forecasting, FHS paths, source loading, clock or persistence occurs.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from typing import Literal
from uuid import UUID

from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.research.price_product import (
    FilteredReturns,
    MomentumResult,
    PriceInputIdentity,
    PriceProductInput,
    PriceProductInputError,
    PriceSeries,
    RecommendationResult,
    RiskResult,
    SourceExecutionBinding,
    _validate_product_input,
    apply_recommendation_policy,
    calculate_momentum,
    calculate_risk,
    complete_input_hash,
    filter_historical_returns,
)
from stanstock.research.price_product_config import (
    PRODUCT_EFFECTIVE_CONFIG_HASH,
    PriceProductConfig,
    price_product_config_hash,
)


@dataclass(frozen=True, slots=True)
class _CandidateConfig:
    schema: str = "opportunities-candidates-synthetic-config@1"
    policy_version: str = "price-candidates-synthetic-v1"
    contract_revision: str = "opportunities-candidates-synthetic@rev-2"
    base_config_hash: str = "55334183af29fc01b853e83f9bf75216f24564956925fb95912cb80a69420867"
    source_mode: str = "synthetic_demo"
    evidence_grade: str = "research"
    calendar: str = "XNYS"
    calendar_start: str = "2023-01-01"
    target_date: str = "2026-09-11"
    required_closes: int = 757
    momentum_sessions: tuple[int, int] = (252, 21)
    recent_sessions: tuple[int, int] = (5, 21)
    drawdown_sessions: int = 252
    episode_lookback_sessions: int = 126
    episode_exclude_latest_sessions: int = 5
    trough_age_sessions: tuple[int, int] = (5, 21)
    episode_rule: str = "deepest_then_latest_trough_then_earliest_peak"
    pullback_depth_bounds: tuple[str, str] = ("0.10", "0.30")
    pullback_recovery_min: str = "0.05"
    deep_depth_min: str = "0.30"
    deep_recovery_min: str = "0.10"
    recovery_rising_closes: int = 3
    exit_prior_low_sessions: int = 20
    exit_current_drawdown_max: str = "-0.10"
    entry_relative_volatility_max: str = "2"
    entry_maximum_drawdown_min: str = "-0.50"
    entry_dollar_turnover_min: str = "5000000"
    entry_target_close_min: str = "10"
    price_arithmetic: str = "decimal_from_str_binary64_prec80_cross_products"
    feature_decimal_places: int = 12
    rounding: str = "ROUND_HALF_EVEN"
    arm_order: tuple[str, str, str] = ("continuation", "positive_pullback", "deep_reversal")
    conflict_rule: str = "error_on_eligible_entry_and_matched_exit"
    selection: str = "deferred"
    future_maximum_per_list: int = 5
    outcome_horizons: tuple[int, int] = (126, 252)
    return_basis: str = "split_adjusted_price_return"
    dividends_included: bool = False
    exit_primary: str = "cash_minus_hold"
    exit_secondary: str = "benchmark_minus_hold"
    payoff_scope: str = "synthetic_terminal_algebra_only"
    synthetic_cash_return: str = "0"
    synthetic_differential_cost: str = "0"
    synthetic_stock_terminal_returns: tuple[str, str, str] = ("-0.20", "0", "0.20")
    synthetic_benchmark_terminal_return: str = "0.05"


_CONFIG = _CandidateConfig()


def _json_scalar(value: object) -> str:
    if isinstance(value, (date, UUID)):
        return value.isoformat() if isinstance(value, date) else str(value)
    raise TypeError("Unsupported synthetic document value")


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_scalar,
    ).encode("ascii")


def _sha256(document: object) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


_CONFIG_SHA256 = _sha256(asdict(_CONFIG))
_TARGET = date(2026, 9, 11)
_DECISION = datetime(2026, 9, 11, 21, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class SyntheticCandidateInput:
    case_id: str
    security_type: Literal["common_stock", "adr", "etf", "other"]
    region: Literal["us", "other"]
    product_input: PriceProductInput


@dataclass(frozen=True, slots=True)
class CandidateFeatures:
    stock_return_5: str
    benchmark_return_5: str
    relative_return_5: str
    stock_return_21: str
    benchmark_return_21: str
    relative_return_21: str
    current_drawdown_252: str
    prior_low_20: str


@dataclass(frozen=True, slots=True)
class DeclineEpisode:
    peak_index: int
    peak_date: date
    peak_close: str
    trough_index: int
    trough_date: date
    trough_close: str
    depth: str
    recovery: str
    age_sessions: int


@dataclass(frozen=True, slots=True)
class NativeControl:
    momentum: MomentumResult | None
    momentum_reason: str | None
    risk: RiskResult
    recommendation: RecommendationResult


@dataclass(frozen=True, slots=True)
class EntryGates:
    status: Literal["pass", "blocked", "withheld"]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EntryArm:
    arm_id: str
    pattern_status: Literal["matched", "not_matched", "withheld"]
    pattern_reason: str
    entry_research_status: Literal["eligible", "blocked", "withheld", "not_matched"]


@dataclass(frozen=True, slots=True)
class ExitReview:
    status: Literal["matched", "not_matched", "withheld"]
    reason: str


@dataclass(frozen=True, slots=True)
class SyntheticCandidateAssessment:
    policy_version: str
    config_sha256: str
    input_hash: str
    case_id: str
    listing_id: UUID
    target_date: date
    source_scope: Literal["synthetic_only"]
    status: Literal["evaluated", "withheld"]
    root_reasons: tuple[str, ...]
    native_control: NativeControl | None
    features: CandidateFeatures | None
    episode: DeclineEpisode | None
    entry_gates: EntryGates
    arms: tuple[EntryArm, ...]
    exit_review: ExitReview


def _calendar_sessions() -> tuple[date, ...]:
    calendar = get_calendar(_CONFIG.calendar, start=_CONFIG.calendar_start, end=_CONFIG.target_date)
    # A bounded calendar starts on its first SESSION (Jan 3), not Jan 1.
    # Clamp only to that first session to satisfy the library's range parser;
    # explicit interval bounds still determine the same inclusive session set
    # even if a caller previously cached a wider calendar.
    start = max(date.fromisoformat(_CONFIG.calendar_start), calendar.first_session.date())
    return tuple(
        session.date() for session in calendar.sessions_in_range(start, _CONFIG.target_date)
    )[-_CONFIG.required_closes :]


def _series_sha256(series: PriceSeries) -> str:
    """Full fixture consistency fingerprint, NOT registered-asset authority."""
    return _sha256(
        {
            "schema": "candidate-synthetic-price-series@1",
            "provider": series.identity.provider,
            "subject": series.identity.subject,
            "currency": series.currency,
            "dates": tuple(value.isoformat() for value in series.dates),
            "closes": tuple(value.hex() for value in series.closes),
            "volumes": None
            if series.volumes is None
            else tuple(None if value is None else value.hex() for value in series.volumes),
            "volume_adjustment_compatible": series.volume_adjustment_compatible,
        }
    )


def _validate_candidate(candidate: SyntheticCandidateInput, config: PriceProductConfig) -> None:
    if price_product_config_hash(config) != PRODUCT_EFFECTIVE_CONFIG_HASH:
        raise PriceProductInputError("frozen_config_mismatch", "Exact frozen base config required")
    if not isinstance(candidate, SyntheticCandidateInput) or not isinstance(
        candidate.product_input, PriceProductInput
    ):
        raise PriceProductInputError(
            "synthetic_input_type_invalid", "Typed synthetic input required"
        )
    p = candidate.product_input
    if (
        not isinstance(candidate.case_id, str)
        or re.fullmatch(r"[a-z][a-z0-9_]{0,47}", candidate.case_id) is None
        or candidate.security_type not in ("common_stock", "adr", "etf", "other")
        or candidate.region not in ("us", "other")
        or type(p.source_eligible) is not bool
        or not isinstance(p.source_ineligibility_reasons, tuple)
        or not all(isinstance(reason, str) for reason in p.source_ineligibility_reasons)
    ):
        raise PriceProductInputError("synthetic_input_type_invalid", "Invalid synthetic context")
    if p.source_execution != SourceExecutionBinding("synthetic_demo", "research"):
        raise PriceProductInputError("synthetic_input_required", "Synthetic research inputs only")
    if not isinstance(p.listing_id, UUID) or p.listing_id.int == 0:
        raise PriceProductInputError("synthetic_identity_invalid", "Non-nil listing UUID required")
    for series in (p.stock, p.benchmark):
        if (
            not isinstance(series, PriceSeries)
            or not isinstance(series.identity, PriceInputIdentity)
            or not isinstance(series.dates, tuple)
            or not all(type(value) is date for value in series.dates)
            or not isinstance(series.closes, tuple)
            or not all(type(value) is float for value in series.closes)
            or type(series.volume_adjustment_compatible) is not bool
            or not isinstance(series.currency, str)
            or not isinstance(series.identity.sha256, str)
            or not isinstance(series.identity.subject, str)
            or (
                series.volumes is not None
                and (
                    not isinstance(series.volumes, tuple)
                    or not all(value is None or type(value) is float for value in series.volumes)
                )
            )
        ):
            raise PriceProductInputError("synthetic_input_type_invalid", "Invalid synthetic series")
        if series.identity.provider != "synthetic_demo":
            raise PriceProductInputError("synthetic_input_required", "Synthetic providers required")
        if not isinstance(series.identity.asset_id, UUID) or series.identity.asset_id.int == 0:
            raise PriceProductInputError(
                "synthetic_identity_invalid", "Non-nil asset UUID required"
            )
        for instant in (series.identity.retrieved_at, series.identity.available_at):
            if not isinstance(instant, datetime) or instant.utcoffset() is None:
                raise PriceProductInputError(
                    "synthetic_time_invalid", "Aware source times required"
                )
    if (
        p.stock.identity.asset_id == p.benchmark.identity.asset_id
        or not p.stock.identity.subject.startswith("SYN-")
    ):
        raise PriceProductInputError(
            "synthetic_identity_invalid", "Invalid synthetic asset identity"
        )
    if (
        not isinstance(p.decision_time, datetime)
        or p.decision_time.utcoffset() is None
        or p.decision_time != _DECISION
    ):
        raise PriceProductInputError(
            "synthetic_time_invalid", "Exact synthetic decision instant required"
        )
    if (
        type(p.target_date) is not date
        or not isinstance(p.calendar_sessions, tuple)
        or not all(type(value) is date for value in p.calendar_sessions)
    ):
        raise PriceProductInputError(
            "synthetic_input_type_invalid", "Typed calendar dates required"
        )
    # Preserve native errors (including invalid prices/volumes, currency,
    # source availability, duplicate or future rows) rather than mapping them.
    _validate_product_input(p, config=config)
    if p.target_date != _TARGET or p.calendar_sessions != _calendar_sessions():
        raise PriceProductInputError(
            "synthetic_calendar_mismatch", "Exact bounded XNYS calendar required"
        )
    for series in (p.stock, p.benchmark):
        if series.identity.sha256 != _series_sha256(series):
            raise PriceProductInputError(
                "synthetic_source_hash_mismatch", "Synthetic series hash mismatch"
            )


def _numeric_error() -> PriceProductInputError:
    return PriceProductInputError(
        "synthetic_numeric_unrepresentable",
        "Synthetic candidate arithmetic cannot be represented by the frozen numeric contract.",
    )


def _fixed12(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_HALF_EVEN
        context.traps[InvalidOperation] = True
        try:
            rounded = value.quantize(Decimal("0.000000000001"))
        except InvalidOperation:
            raise _numeric_error() from None
    return format(rounded, "f")


def _filtered(
    series: PriceSeries, config: PriceProductConfig
) -> tuple[FilteredReturns | None, str | None]:
    try:
        return filter_historical_returns(
            series.closes,
            burn_in=config.simulation.filter_burn_in,
            variance_target_weight=config.simulation.variance_target_weight,
            variance_persistence=config.simulation.variance_persistence,
            innovation_weight=config.simulation.innovation_weight,
        ), None
    except PriceProductInputError as exc:
        return None, exc.reason_code


def _native_control(p: PriceProductInput, config: PriceProductConfig) -> NativeControl:
    # The new raised error must remain OUTSIDE native withholding handlers.
    for closes in (p.stock.closes, p.benchmark.closes):
        if closes[-22] / closes[-253] == 0.0:
            raise _numeric_error()
    momentum_reason = None
    try:
        momentum = calculate_momentum(
            p.stock.closes,
            p.benchmark.closes,
            lookback_sessions=config.momentum.lookback_sessions,
            skip_sessions=config.momentum.skip_sessions,
        )
    except PriceProductInputError as exc:
        momentum = None
        momentum_reason = exc.reason_code
    stock_filter, stock_reason = _filtered(p.stock, config)
    benchmark_filter, benchmark_reason = _filtered(p.benchmark, config)
    try:
        risk = calculate_risk(
            p.stock,
            stock_filter=stock_filter,
            benchmark_filter=benchmark_filter,
            stock_filter_reason=stock_reason,
            benchmark_filter_reason=benchmark_reason,
            config=config,
        )
    except OverflowError:
        raise _numeric_error() from None
    recommendation = apply_recommendation_policy(
        momentum=momentum,
        momentum_insufficiency_reason=momentum_reason,
        risk=risk,
        target_close=p.stock.closes[-1],
        source_eligible=p.source_eligible,
        source_ineligibility_reasons=p.source_ineligibility_reasons,
        config=config,
    )
    return NativeControl(momentum, momentum_reason, risk, recommendation)


def _entry_gates(risk: RiskResult, target_close: float) -> EntryGates:
    reasons = []
    unavailable = False
    if risk.relative_volatility is None:
        reasons.append("relative_volatility_unavailable")
        unavailable = True
    elif risk.relative_volatility > float(_CONFIG.entry_relative_volatility_max):
        reasons.append("relative_volatility_above_buy_limit")
    if risk.maximum_drawdown < float(_CONFIG.entry_maximum_drawdown_min):
        reasons.append("drawdown_below_buy_limit")
    if risk.average_dollar_turnover_20d is None:
        reasons.append("dollar_turnover_unavailable")
        unavailable = True
    elif risk.average_dollar_turnover_20d < float(_CONFIG.entry_dollar_turnover_min):
        reasons.append("dollar_turnover_below_buy_minimum")
    if target_close < float(_CONFIG.entry_target_close_min):
        reasons.append("target_close_below_buy_minimum")
    return EntryGates(
        "withheld" if unavailable else "blocked" if reasons else "pass", tuple(reasons)
    )


def _episode_indices(prices: tuple[Decimal, ...]) -> tuple[int, int] | None:
    """Running earliest maximum; deepest decline, then latest trough."""
    t = len(prices) - 1
    peak = t - _CONFIG.episode_lookback_sessions
    selected: tuple[int, int] | None = None
    for trough in range(peak + 1, t - _CONFIG.episode_exclude_latest_sessions + 1):
        if prices[trough - 1] > prices[peak]:
            peak = trough - 1
        if prices[trough] >= prices[peak]:
            continue
        if selected is not None:
            old_peak, old_trough = selected
            if (prices[peak] - prices[trough]) * prices[old_peak] < (
                prices[old_peak] - prices[old_trough]
            ) * prices[peak]:
                continue
        selected = peak, trough
    return selected


def _relative_positive(prices: tuple[Decimal, ...], benchmark: tuple[Decimal, ...], k: int) -> bool:
    return prices[-1] * benchmark[-1 - k] > benchmark[-1] * prices[-1 - k]


def _pattern_reason(
    arm: str,
    momentum: MomentumResult,
    prices: tuple[Decimal, ...],
    benchmark: tuple[Decimal, ...],
    episode: tuple[int, int] | None,
) -> str:
    positive = arm != "deep_reversal"
    if momentum.direction != ("positive" if positive else "negative"):
        return "long_trend_not_positive" if positive else "long_trend_not_negative"
    if arm == "continuation":
        if prices[-1] <= prices[-22]:
            return "recent_21_not_positive"
        if not _relative_positive(prices, benchmark, 21):
            return "recent_21_relative_not_positive"
        return "continuation_pattern"
    if episode is None:
        return "decline_episode_absent"
    peak, trough = episode
    age = len(prices) - 1 - trough
    if not _CONFIG.trough_age_sessions[0] <= age <= _CONFIG.trough_age_sessions[1]:
        return "trough_age_outside_5_21"
    depth_numerator = prices[peak] - prices[trough]
    if positive:
        lower, upper = map(Decimal, _CONFIG.pullback_depth_bounds)
        if not lower * prices[peak] <= depth_numerator <= upper * prices[peak]:
            return "pullback_depth_outside_10_30"
        recovery = Decimal(_CONFIG.pullback_recovery_min)
        recovery_reason = "pullback_recovery_below_5"
    else:
        if depth_numerator < Decimal(_CONFIG.deep_depth_min) * prices[peak]:
            return "deep_depth_below_30"
        recovery = Decimal(_CONFIG.deep_recovery_min)
        recovery_reason = "deep_recovery_below_10"
    if prices[-1] - prices[trough] < recovery * prices[trough]:
        return recovery_reason
    if prices[-1] <= prices[-6]:
        return "recent_5_not_positive"
    if not _relative_positive(prices, benchmark, 5):
        return "recent_5_relative_not_positive"
    if not prices[-3] < prices[-2] < prices[-1]:
        return "last_3_closes_not_rising"
    return f"{arm}_pattern"


def _arms(
    momentum: MomentumResult | None,
    prices: tuple[Decimal, ...],
    benchmark: tuple[Decimal, ...],
    episode: tuple[int, int] | None,
    gates: EntryGates,
) -> tuple[EntryArm, ...]:
    result = []
    for arm in _CONFIG.arm_order:
        if momentum is None:
            result.append(EntryArm(arm, "withheld", "momentum_unavailable", "withheld"))
            continue
        reason = _pattern_reason(arm, momentum, prices, benchmark, episode)
        matched = reason == f"{arm}_pattern"
        result.append(
            EntryArm(
                arm,
                "matched" if matched else "not_matched",
                reason,
                ("eligible" if gates.status == "pass" else gates.status)
                if matched
                else "not_matched",
            )
        )
    return tuple(result)


def _exit_review(prices: tuple[Decimal, ...], benchmark: tuple[Decimal, ...]) -> ExitReview:
    if prices[-1] >= prices[-22]:
        return ExitReview("not_matched", "recent_21_not_negative")
    if prices[-1] * benchmark[-22] >= benchmark[-1] * prices[-22]:
        return ExitReview("not_matched", "recent_21_relative_not_negative")
    if prices[-1] >= min(prices[-21:-1]):
        return ExitReview("not_matched", "prior_20_low_not_broken")
    maximum = max(prices[-253:])
    if prices[-1] > (1 + Decimal(_CONFIG.exit_current_drawdown_max)) * maximum:
        return ExitReview("not_matched", "current_drawdown_not_at_least_10")
    return ExitReview("matched", "affirmative_deterioration_pattern")


def _check_conflict(arms: tuple[EntryArm, ...], exit_review: ExitReview) -> None:
    if exit_review.status == "matched" and any(
        arm.entry_research_status == "eligible" for arm in arms
    ):
        raise PriceProductInputError(
            "candidate_policy_conflict", "Eligible entry conflicts with exit review"
        )


def assess_synthetic_candidate(
    candidate: SyntheticCandidateInput, *, base_config: PriceProductConfig
) -> SyntheticCandidateAssessment:
    """Assess separate research hypotheses; no ranking or production adoption."""
    _validate_candidate(candidate, base_config)
    p = candidate.product_input
    input_hash = _sha256(
        {
            "domain": "candidate-synthetic-input@1",
            "complete_input_hash": complete_input_hash(p),
            "case_id": candidate.case_id,
            "security_type": candidate.security_type,
            "region": candidate.region,
            "policy_version": _CONFIG.policy_version,
            "config_sha256": _CONFIG_SHA256,
        }
    )
    root_reasons = tuple(
        reason
        for condition, reason in (
            (candidate.security_type not in ("common_stock", "adr"), "unsupported_security_type"),
            (candidate.region != "us", "unsupported_region"),
            (not p.source_eligible, "source_ineligible"),
        )
        if condition
    )
    if root_reasons:
        return SyntheticCandidateAssessment(
            _CONFIG.policy_version,
            _CONFIG_SHA256,
            input_hash,
            candidate.case_id,
            p.listing_id,
            p.target_date,
            "synthetic_only",
            "withheld",
            root_reasons,
            None,
            None,
            None,
            EntryGates("withheld", ("case_withheld",)),
            tuple(
                EntryArm(arm, "withheld", "case_withheld", "withheld") for arm in _CONFIG.arm_order
            ),
            ExitReview("withheld", "case_withheld"),
        )
    native = _native_control(p, base_config)
    with localcontext() as context:
        context.prec = 80
        context.rounding = ROUND_HALF_EVEN
        prices = tuple(Decimal(str(value)) for value in p.stock.closes)
        benchmark = tuple(Decimal(str(value)) for value in p.benchmark.closes)
        recent: list[str] = []
        for k in _CONFIG.recent_sessions:
            recent.extend(
                (
                    _fixed12(prices[-1] / prices[-1 - k] - 1),
                    _fixed12(benchmark[-1] / benchmark[-1 - k] - 1),
                    _fixed12(
                        (prices[-1] * benchmark[-1 - k]) / (prices[-1 - k] * benchmark[-1]) - 1
                    ),
                )
            )
        features = CandidateFeatures(
            recent[0],
            recent[1],
            recent[2],
            recent[3],
            recent[4],
            recent[5],
            _fixed12(prices[-1] / max(prices[-253:]) - 1),
            str(min(prices[-21:-1])),
        )
        indices = _episode_indices(prices)
        episode = None
        if indices is not None:
            peak, trough = indices
            episode = DeclineEpisode(
                peak,
                p.stock.dates[peak],
                str(prices[peak]),
                trough,
                p.stock.dates[trough],
                str(prices[trough]),
                _fixed12(1 - prices[trough] / prices[peak]),
                _fixed12(prices[-1] / prices[trough] - 1),
                len(prices) - 1 - trough,
            )
        gates = _entry_gates(native.risk, p.stock.closes[-1])
        arms = _arms(native.momentum, prices, benchmark, indices, gates)
        exit_review = _exit_review(prices, benchmark)
        _check_conflict(arms, exit_review)
    return SyntheticCandidateAssessment(
        _CONFIG.policy_version,
        _CONFIG_SHA256,
        input_hash,
        candidate.case_id,
        p.listing_id,
        p.target_date,
        "synthetic_only",
        "evaluated",
        (),
        native,
        features,
        episode,
        gates,
        arms,
        exit_review,
    )

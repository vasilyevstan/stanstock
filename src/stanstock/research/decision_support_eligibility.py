"""The prospective C2 policy, with no persistence or provider dependencies.

This qualifies a closing-price × reported-volume activity proxy. It does not
certify a provider's volume units, alter the frozen forecast, or infer a split.
Physical source authority belongs to the caller, not to a copied assessment.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import ClassVar, Literal

from stanstock.research.price_product import (
    PriceProductInput,
    PriceProductResult,
    RiskResult,
    apply_recommendation_policy,
)
from stanstock.research.price_product_config import (
    PRODUCT_EFFECTIVE_CONFIG_HASH as _FORECAST_CONFIG_HASH,
)
from stanstock.research.price_product_config import (
    PriceProductConfig,
    load_price_product_config,
    price_product_config_hash,
)

PRODUCT_VERSION = "research-product-v2"
PRODUCT_PAYLOAD_SCHEMA = "research-product@2"
MOMENTUM_METHOD_VERSION = "us-relative-momentum-v2"
SOURCE_CONTRACT = "decision-support-liquidity@1"
WITNESS_CONTRACT = "twelve-data-none-witness@1"
LIQUIDITY_PURPOSE = "decision_liquidity_only"
PRODUCT_EFFECTIVE_CONFIG_HASH = "16664e7fa98d94484f97da6c68a473e54dac5523c16f3bf5acfd1680f34559da"
PRODUCT_CONFIG_FILE_SHA256 = "8021590dabbedde09a4c4508aa7203e244c78fed9eeb09ea860ac2f826c9cf89"
MAXIMUM_VOLUME = 9223372036854775807
FORECAST_CONFIG_HASH = _FORECAST_CONFIG_HASH

# Independently reviewed document, not a mapping loaded from the runtime file.
# Comparing canonical bytes distinguishes true/1/1.0 at every depth.
_REVIEWED_DOCUMENT = r"""
{
  "schema_version": 2,
  "product_version": "research-product-v2",
  "payload_schema": "research-product@2",
  "decision_method_version": "us-relative-momentum-v2",
  "base_product_version": "research-product-v1",
  "base_effective_config_hash": "55334183af29fc01b853e83f9bf75216f24564956925fb95912cb80a69420867",
  "liquidity": {
    "source_contract": "decision-support-liquidity@1",
    "purpose": "decision_liquidity_only",
    "calendar": "XNYS",
    "currency": "USD",
    "sessions": 20,
    "request_modes": ["none", "splits"],
    "admission": "same_batch_exact_decimal_close_and_integer_volume_every_session",
    "parity_tolerance": "0",
    "availability": "actual_retrieved_and_available_le_logical_cutoff",
    "volume_minimum": 0,
    "volume_maximum": 9223372036854775807,
    "zero_volume": "reported_zero_only_never_missing_substitute",
    "turnover": "mean_closing_price_times_provider_reported_volume",
    "arithmetic": "native_float_products_fsum_mean_finite_representable_only",
    "fallback": "withhold_liquidity_keep_independent_price_research"
  },
  "mode_witness": {
    "contract": "twelve-data-none-witness@1",
    "symbol": "NVDA",
    "currency": "USD",
    "exchange": "NASDAQ",
    "issuer_event_date": "2024-06-10",
    "issuer_split_new_shares_per_old_share": 10,
    "authority_url": "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-first-quarter-fiscal-2025",
    "request_start": "2024-06-03",
    "pre_sessions": [
      "2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07"
    ],
    "post_sessions": [
      "2024-06-10", "2024-06-11", "2024-06-12", "2024-06-13", "2024-06-14"
    ],
    "pre_rule": "none_close_gt_splits_close_and_10_times_none_volume_eq_splits_volume_all",
    "post_rule": "exact_decimal_close_and_integer_volume_parity_all",
    "control_volume": "strictly_positive_on_all_control_sessions",
    "cadence": "each_fresh_acquisition_batch_no_additional_control_requests",
    "failure_scope": "all_liquidity_in_that_batch",
    "maximum_outputsize": 5000,
    "maximum_none_requests_per_batch": 1
  }
}
"""

LIQUIDITY_REASONS = (
    "liquidity_mode_witness_unavailable",
    "liquidity_mode_witness_failed",
    "liquidity_pair_missing",
    "liquidity_window_incomplete",
    "liquidity_close_mismatch",
    "liquidity_volume_missing",
    "liquidity_volume_mismatch",
    "liquidity_cutoff_unsafe",
    "liquidity_quota_unavailable",
    "liquidity_provider_unavailable",
    "liquidity_provider_payload_invalid",
    "liquidity_nonrepresentable",
)
_SUPERSEDED_LIQUIDITY_REASONS = frozenset(
    (
        "dollar_turnover_volume_missing",
        "dollar_turnover_adjustment_incompatible",
        "dollar_turnover_nonfinite",
    )
)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("C2 JSON forbids duplicate keys")
        result[key] = value
    return result


def _nonfinite_constant(value: str) -> object:
    del value
    raise ValueError("C2 JSON forbids nonfinite constants")


def strict_json(payload: bytes) -> object:
    """Strict JSON for config and C2 evidence, including nested objects."""
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=_nonfinite_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("C2 JSON is malformed") from None


@dataclass(frozen=True, slots=True)
class DecisionSupportConfig:
    forecast_config: PriceProductConfig
    product_version: ClassVar[str] = PRODUCT_VERSION
    payload_schema: ClassVar[str] = PRODUCT_PAYLOAD_SCHEMA
    effective_config_hash: ClassVar[str] = PRODUCT_EFFECTIVE_CONFIG_HASH
    source_contract: ClassVar[str] = SOURCE_CONTRACT
    sessions: ClassVar[int] = 20
    witness_symbol: ClassVar[str] = "NVDA"
    witness_request_start: ClassVar[date] = date(2024, 6, 3)
    witness_pre_sessions: ClassVar[tuple[date, ...]] = tuple(
        date(2024, 6, day) for day in (3, 4, 5, 6, 7)
    )
    witness_post_sessions: ClassVar[tuple[date, ...]] = tuple(
        date(2024, 6, day) for day in (10, 11, 12, 13, 14)
    )


def default_decision_support_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config/scoring/research-product-v2.yml"


def load_decision_support_config(path: Path | None = None) -> DecisionSupportConfig:
    config_path = path if path is not None else default_decision_support_config_path()
    try:
        payload = config_path.read_bytes()
    except OSError:
        raise ValueError("research-product-v2 config is unreadable") from None
    document = strict_json(payload)
    reviewed = canonical_json(strict_json(_REVIEWED_DOCUMENT.encode("utf-8")))
    if (
        canonical_json(document) != reviewed
        or hashlib.sha256(reviewed).hexdigest() != PRODUCT_EFFECTIVE_CONFIG_HASH
    ):
        raise ValueError("research-product-v2 config does not match the reviewed exact document")
    if path is None and hashlib.sha256(payload).hexdigest() != PRODUCT_CONFIG_FILE_SHA256:
        raise ValueError("research-product-v2 config bytes changed")
    base = load_price_product_config()
    if price_product_config_hash(base) != FORECAST_CONFIG_HASH:
        raise ValueError("research-product-v2 frozen base config changed")
    return DecisionSupportConfig(forecast_config=base)


@dataclass(frozen=True, slots=True)
class LiquidityObservation:
    session: date
    close: Decimal
    volume: int | None


@dataclass(frozen=True, slots=True)
class LiquidityAssessment:
    status: Literal["matched", "unavailable"]
    reasons: tuple[str, ...]
    assessed_dates: tuple[date, ...]
    average_dollar_turnover_20d: float | None


def unavailable_liquidity(
    *reasons: str, assessed_dates: tuple[date, ...] = ()
) -> LiquidityAssessment:
    if not reasons or any(reason not in LIQUIDITY_REASONS for reason in reasons):
        raise ValueError("Unreviewed liquidity unavailability reason")
    ordered = tuple(reason for reason in LIQUIDITY_REASONS if reason in reasons)
    return LiquidityAssessment("unavailable", ordered, assessed_dates, None)


def _observations(
    values: tuple[LiquidityObservation, ...],
) -> dict[date, LiquidityObservation]:
    result: dict[date, LiquidityObservation] = {}
    for value in values:
        if type(value.session) is not date or value.session in result:
            raise ValueError("Liquidity observations require unique calendar dates")
        if not isinstance(value.close, Decimal) or not value.close.is_finite() or value.close <= 0:
            raise ValueError("Liquidity closes must be finite positive Decimals")
        if value.volume is not None and (
            type(value.volume) is not int or not 0 <= value.volume <= MAXIMUM_VOLUME
        ):
            raise ValueError("Liquidity volumes must be bounded integers or explicitly missing")
        result[value.session] = value
    return result


def assess_mode_witness(
    *,
    none: tuple[LiquidityObservation, ...] | None,
    splits: tuple[LiquidityObservation, ...] | None,
    config: DecisionSupportConfig,
) -> Literal["matched", "unavailable", "failed"]:
    if none is None or splits is None:
        return "unavailable"
    left, right = _observations(none), _observations(splits)
    required = config.witness_pre_sessions + config.witness_post_sessions
    if any(session not in left or session not in right for session in required):
        return "unavailable"
    for session in required:
        raw, adjusted = left[session], right[session]
        if raw.volume is None or adjusted.volume is None or raw.volume <= 0 or adjusted.volume <= 0:
            return "failed"
        if session in config.witness_pre_sessions:
            if not (raw.close > adjusted.close and 10 * raw.volume == adjusted.volume):
                return "failed"
        elif raw.close != adjusted.close or raw.volume != adjusted.volume:
            return "failed"
    return "matched"


def assess_liquidity(
    *,
    required_sessions: tuple[date, ...],
    none: tuple[LiquidityObservation, ...] | None,
    splits: tuple[LiquidityObservation, ...] | None,
    witness_status: Literal["matched", "unavailable", "failed"],
    unavailable_reasons: tuple[str, ...] = (),
) -> LiquidityAssessment:
    """The sole parity/proxy calculation; callers independently verify sources."""
    if (
        len(required_sessions) != 20
        or any(type(session) is not date for session in required_sessions)
        or tuple(sorted(set(required_sessions))) != required_sessions
    ):
        raise ValueError("Liquidity requires the independently derived ordered 20 sessions")
    reasons = list(unavailable_reasons)
    if witness_status not in ("matched", "unavailable", "failed"):
        raise ValueError("Invalid liquidity witness status")
    if witness_status != "matched":
        reasons.append(f"liquidity_mode_witness_{witness_status}")
    if none is None or splits is None:
        return unavailable_liquidity(*reasons, "liquidity_pair_missing")
    left, right = _observations(none), _observations(splits)
    if tuple(left) != required_sessions or tuple(right) != required_sessions:
        return unavailable_liquidity(*reasons, "liquidity_window_incomplete")
    for session in required_sessions:
        raw, adjusted = left[session], right[session]
        if raw.close != adjusted.close:
            reasons.append("liquidity_close_mismatch")
        if raw.volume is None or adjusted.volume is None:
            reasons.append("liquidity_volume_missing")
        elif raw.volume != adjusted.volume:
            reasons.append("liquidity_volume_mismatch")
    if reasons:
        return unavailable_liquidity(*reasons, assessed_dates=required_sessions)
    try:
        products: list[float] = []
        for value in none:
            close = float(value.close)
            if not math.isfinite(close) or close <= 0:
                raise OverflowError
            assert value.volume is not None  # independently checked above, zero is real zero
            product = close * value.volume
            if not math.isfinite(product) or (value.volume > 0 and product == 0):
                raise OverflowError
            products.append(product)
        turnover = math.fsum(products) / 20
        if not math.isfinite(turnover) or (turnover == 0 and any(value > 0 for value in products)):
            raise OverflowError
        canonical_json(turnover)
    except (OverflowError, ValueError):
        return unavailable_liquidity("liquidity_nonrepresentable", assessed_dates=required_sessions)
    return LiquidityAssessment("matched", (), required_sessions, turnover)


def apply_liquidity_to_risk(
    risk: RiskResult,
    liquidity: LiquidityAssessment,
) -> RiskResult:
    return replace(
        risk,
        average_dollar_turnover_20d=liquidity.average_dollar_turnover_20d,
        insufficiency_reasons=tuple(
            reason
            for reason in risk.insufficiency_reasons
            if reason not in _SUPERSEDED_LIQUIDITY_REASONS
        )
        + liquidity.reasons,
    )


def compose_decision_support(
    *,
    frozen_result: PriceProductResult,
    product_input: PriceProductInput,
    liquidity: LiquidityAssessment,
    input_hash: str,
    config: DecisionSupportConfig,
) -> PriceProductResult:
    """Replace only v2 liquidity; do not recalculate momentum, risk, or FHS."""
    if (
        frozen_result.product_version != "research-product-v1"
        or frozen_result.effective_config_hash != FORECAST_CONFIG_HASH
        or price_product_config_hash(config.forecast_config) != FORECAST_CONFIG_HASH
        or product_input.stock.volume_adjustment_compatible
        or product_input.benchmark.volume_adjustment_compatible
    ):
        raise ValueError("C2 composition requires the unchanged frozen forecast boundary")
    risk = apply_liquidity_to_risk(frozen_result.risk, liquidity)
    recommendation = apply_recommendation_policy(
        momentum=frozen_result.momentum,
        momentum_insufficiency_reason=frozen_result.momentum_insufficiency_reason,
        risk=risk,
        target_close=product_input.stock.closes[-1],
        source_eligible=product_input.source_eligible,
        source_ineligibility_reasons=product_input.source_ineligibility_reasons,
        config=config.forecast_config,
    )
    return replace(
        frozen_result,
        payload_schema=PRODUCT_PAYLOAD_SCHEMA,
        product_version=PRODUCT_VERSION,
        effective_config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
        input_hash=input_hash,
        risk=risk,
        recommendation=recommendation,
    )

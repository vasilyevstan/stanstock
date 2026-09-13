from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self, cast

import yaml

from stanstock.data.management.config_loader import (
    default_sec_cik_mapping_path,
    default_sec_fundamentals_config_path,
)
from stanstock.data.sec_config import (
    SecCikConfig,
    SecFundamentalsConfig,
    load_sec_cik_config,
    load_sec_fundamentals_config,
)

LONG_FORECAST_HORIZONS = ("3y", "5y")
LONG_METRIC_FAMILIES = ("fcf_per_share", "eps_per_share")
LONG_SCENARIOS = ("bear", "base", "bull")

#: Optional, explicitly default-off capability fields.
#:
#: A config that does not declare one of these keys parses to ``None`` and the
#: field is removed from the effective payload before hashing, so every
#: already-frozen configuration (``us-sec-long-v1``, ``us-sec-long-v2``)
#: keeps its exact historical effective hash when a later version adds a new
#: capability here. An explicit ``{enabled: false}`` is a *different*
#: configuration than an absent key and does change the hash.
LONG_OPTIONAL_CAPABILITY_FIELDS = (
    "adjacent_selected_annual_diluted_share_continuity",
    "newest_quarter_anchored_homogeneous_ttm_alias_selection",
    "joint_compatible_invested_capital_pair_selection",
    "proven_observation_correction_availability",
)

#: Optional non-capability fields that are also omitted from the effective
#: hash while absent, for the same freeze-preserving reason.
LONG_OPTIONAL_BINDING_FIELDS = (
    "fundamentals_config_version",
    "maximum_same_date_source_combinations",
)

#: The one reviewed value for `maximum_same_date_source_combinations`.
#:
#: It is a reviewed safety bound, not a tuning knob: a configuration that
#: enables joint invested-capital selection must declare exactly this number.
#: Any other value is rejected outright rather than silently accepted,
#: truncated, or defaulted, because a different ceiling would change which
#: balance-sheet dates are refused.
REVIEWED_MAX_SAME_DATE_SOURCE_COMBINATIONS = 256

LONG_V4_VERSION = "us-sec-long-v4"
LONG_V4_METHOD = "sec_entity_growth_dilution_multiple_reversion"
LONG_V4_RESEARCH_STATUS = "research_only_unactivated"
LONG_V4_SCORING_VERSION = "us-price-baseline-v2"
LONG_V4_FUNDAMENTALS_VERSION = "us-sec-fundamentals-v1"
LONG_V4_FUNDAMENTALS_CONFIG_FILE_SHA256 = (
    "829ed267eec62304804c9ef71f1816636389c2b0ad77423d8a534a00a5e1ae30"
)
LONG_V4_FUNDAMENTALS_CONFIG_HASH = (
    "7822a1faaae1c8028d71851337a7dbb7a65d9aeaa0f44478e3814a6468b62604"
)
LONG_V4_SEC_CIK_CONFIG_VERSION = "us-sec-cik-v1"
LONG_V4_SEC_CIK_CONFIG_FILE_SHA256 = (
    "3e65d924d77b3ea233806cdfb006ddb30a0b982fecafd9568cbf680e37ffeb07"
)
LONG_V4_SEC_CIK_CONFIG_HASH = "5443bb613e1b40545faa4f53794a869453c6f78317390389c3959e938ba99f65"
LONG_V4_SEC_MAPPING_SOURCE_SHA256 = (
    "ec43db74f82d1739cce6340f36b9695dcb51231fc38edd493215677627bb01cd"
)
LONG_V4_CONFIG_FILE_SHA256 = "840bda0d6b3122dd4c75b9b14ec32cf64a1b1dc49e921cf88a9256f413fe81d2"
LONG_V4_EFFECTIVE_CONFIG_HASH = "acaf8a3a8cd6975ef894a5fbce50a6a3ca3dc6dce0886f03dbf9416e285268fa"


class LongForecastConfigParseError(ValueError):
    """The configuration file could not be parsed as YAML.

    The PyYAML error is deliberately not carried forward. A parser error
    quotes the offending source line, so an operator pointing the loader at
    the wrong file (an ``.env``-shaped file, a credential fragment pasted
    into a config) would otherwise have that line echoed into stderr and
    logs. The path and the failure kind are enough to fix the file; the file
    contents are not reproduced.

    Semantic validation is unaffected: a well-formed YAML file that violates
    the long-forecast contract still raises a plain `ValueError` whose
    message names the offending key, never a configured value.
    """


@dataclass(frozen=True, slots=True)
class LongEligibilityConfig:
    maximum_metric_age_days: int
    minimum_share_consistency_periods: int
    share_consistency_relative_tolerance: float
    balance_sheet_date_tolerance_days: int
    fcf_failure_blocks_eps: bool
    unsupported_sic_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class LongMetricFamilyConfig:
    source_metric: str
    minimum_annual_periods: int
    minimum_current_value: float
    multiple_minimum: float
    multiple_maximum: float


@dataclass(frozen=True, slots=True)
class LongGrowthConfig:
    historical_recency_decay: float
    historical_minimum: float
    historical_maximum: float
    tax_rate_minimum: float
    tax_rate_maximum: float
    roic_minimum: float
    roic_maximum: float
    reinvestment_minimum: float
    reinvestment_maximum: float
    sustainable_minimum: float
    sustainable_maximum: float
    peer_minimum: float
    peer_maximum: float
    historical_weight: float
    sustainable_weight: float
    peer_weight: float
    initial_growth_minimum: float
    initial_growth_maximum: float
    terminal_growth: float


@dataclass(frozen=True, slots=True)
class LongPeerConfig:
    sic_prefix_levels: tuple[int, ...]
    minimum_peers: dict[int, int]


@dataclass(frozen=True, slots=True)
class LongHorizonConfig:
    years: int
    fade: tuple[float, ...]
    multiple_reversion: float


@dataclass(frozen=True, slots=True)
class LongScenarioConfig:
    growth_delta: float
    reinvestment_multiplier: float
    peer_multiple_multiplier: float


@dataclass(frozen=True, slots=True)
class LongForecastConfig:
    schema_version: int
    version: str
    enabled_scoring_versions: tuple[str, ...]
    price_provider: str
    fundamentals_provider: str
    return_basis: str
    dividends_included: bool
    probability_positive_enabled: bool
    eligibility: LongEligibilityConfig
    metric_families: dict[str, LongMetricFamilyConfig]
    growth: LongGrowthConfig
    peer: LongPeerConfig
    horizons: dict[str, LongHorizonConfig]
    scenarios: dict[str, LongScenarioConfig]
    adjacent_selected_annual_diluted_share_continuity: bool | None
    newest_quarter_anchored_homogeneous_ttm_alias_selection: bool | None
    joint_compatible_invested_capital_pair_selection: bool | None
    proven_observation_correction_availability: bool | None
    fundamentals_config_version: str | None
    maximum_same_date_source_combinations: int | None
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> Self:
        schema_version = _positive_int(mapping, "schema_version")
        if schema_version != 1:
            raise ValueError("Long forecast config schema_version must be 1")
        return_basis = _required_text(mapping, "return_basis")
        if return_basis != "split_adjusted_price_return":
            raise ValueError("Long forecasts require split_adjusted_price_return")
        dividends_included = mapping.get("dividends_included")
        if dividends_included is not False:
            raise ValueError("Long forecasts must exclude dividends")
        probability_enabled = mapping.get("probability_positive_enabled")
        if probability_enabled is not False:
            raise ValueError("Long forecast probability must remain disabled")

        eligibility_raw = _mapping(mapping, "eligibility")
        unsupported_ranges = tuple(
            _sic_range(value) for value in _list(eligibility_raw, "unsupported_sic_ranges")
        )
        eligibility = LongEligibilityConfig(
            maximum_metric_age_days=_positive_int(
                eligibility_raw,
                "maximum_metric_age_days",
            ),
            minimum_share_consistency_periods=_positive_int(
                eligibility_raw,
                "minimum_share_consistency_periods",
            ),
            share_consistency_relative_tolerance=_bounded_float(
                eligibility_raw,
                "share_consistency_relative_tolerance",
                minimum=0,
                maximum=0.15,
                minimum_inclusive=False,
            ),
            balance_sheet_date_tolerance_days=_nonnegative_int(
                eligibility_raw,
                "balance_sheet_date_tolerance_days",
            ),
            fcf_failure_blocks_eps=_required_bool(
                eligibility_raw,
                "fcf_failure_blocks_eps",
            ),
            unsupported_sic_ranges=unsupported_ranges,
        )

        metric_raw = _mapping(mapping, "metric_families")
        if set(metric_raw) != set(LONG_METRIC_FAMILIES):
            raise ValueError("metric_families must contain exactly fcf_per_share and eps_per_share")
        metric_families: dict[str, LongMetricFamilyConfig] = {}
        for family in LONG_METRIC_FAMILIES:
            raw_family = _mapping(metric_raw, family)
            minimum = _positive_float(raw_family, "multiple_minimum")
            maximum = _positive_float(raw_family, "multiple_maximum")
            if minimum >= maximum:
                raise ValueError(f"{family} multiple bounds must increase")
            metric_families[family] = LongMetricFamilyConfig(
                source_metric=_required_text(raw_family, "source_metric"),
                minimum_annual_periods=_integer_at_least(
                    raw_family,
                    "minimum_annual_periods",
                    minimum=2,
                ),
                minimum_current_value=_positive_float(
                    raw_family,
                    "minimum_current_value",
                ),
                multiple_minimum=minimum,
                multiple_maximum=maximum,
            )
        required_share_periods = max(
            family.minimum_annual_periods for family in metric_families.values()
        )
        if eligibility.minimum_share_consistency_periods < required_share_periods:
            raise ValueError(
                "minimum_share_consistency_periods must cover every selected annual period"
            )

        growth_raw = _mapping(mapping, "growth")
        growth = LongGrowthConfig(
            historical_recency_decay=_bounded_float(
                growth_raw,
                "historical_recency_decay",
                minimum=0,
                maximum=1,
                minimum_inclusive=False,
                maximum_inclusive=False,
            ),
            historical_minimum=_finite_float(growth_raw, "historical_minimum"),
            historical_maximum=_finite_float(growth_raw, "historical_maximum"),
            tax_rate_minimum=_finite_float(growth_raw, "tax_rate_minimum"),
            tax_rate_maximum=_finite_float(growth_raw, "tax_rate_maximum"),
            roic_minimum=_finite_float(growth_raw, "roic_minimum"),
            roic_maximum=_finite_float(growth_raw, "roic_maximum"),
            reinvestment_minimum=_finite_float(growth_raw, "reinvestment_minimum"),
            reinvestment_maximum=_finite_float(growth_raw, "reinvestment_maximum"),
            sustainable_minimum=_finite_float(growth_raw, "sustainable_minimum"),
            sustainable_maximum=_finite_float(growth_raw, "sustainable_maximum"),
            peer_minimum=_finite_float(growth_raw, "peer_minimum"),
            peer_maximum=_finite_float(growth_raw, "peer_maximum"),
            historical_weight=_positive_float(growth_raw, "historical_weight"),
            sustainable_weight=_positive_float(growth_raw, "sustainable_weight"),
            peer_weight=_positive_float(growth_raw, "peer_weight"),
            initial_growth_minimum=_finite_float(
                growth_raw,
                "initial_growth_minimum",
            ),
            initial_growth_maximum=_finite_float(
                growth_raw,
                "initial_growth_maximum",
            ),
            terminal_growth=_finite_float(growth_raw, "terminal_growth"),
        )
        for name, lower, upper in (
            ("historical", growth.historical_minimum, growth.historical_maximum),
            ("tax rate", growth.tax_rate_minimum, growth.tax_rate_maximum),
            ("ROIC", growth.roic_minimum, growth.roic_maximum),
            ("reinvestment", growth.reinvestment_minimum, growth.reinvestment_maximum),
            ("sustainable growth", growth.sustainable_minimum, growth.sustainable_maximum),
            ("peer growth", growth.peer_minimum, growth.peer_maximum),
            (
                "initial growth",
                growth.initial_growth_minimum,
                growth.initial_growth_maximum,
            ),
        ):
            if lower >= upper:
                raise ValueError(f"{name} bounds must increase")
        if not math.isclose(
            growth.historical_weight + growth.sustainable_weight + growth.peer_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Long forecast growth weights must sum to 1")

        peer_raw = _mapping(mapping, "peer")
        levels = tuple(
            _positive_int_value(value, label="sic_prefix_levels")
            for value in _list(peer_raw, "sic_prefix_levels")
        )
        if not levels or len(levels) != len(set(levels)) or any(level > 4 for level in levels):
            raise ValueError("sic_prefix_levels must contain unique values from 1 through 4")
        if tuple(sorted(levels, reverse=True)) != levels:
            raise ValueError("sic_prefix_levels must be ordered from most to least specific")
        floors_raw = _mapping(peer_raw, "minimum_peers")
        floors = {
            int(level): _positive_int_value(value, label=f"minimum_peers[{level}]")
            for level, value in floors_raw.items()
        }
        if set(floors) != set(levels):
            raise ValueError("minimum_peers must define every configured SIC prefix level")
        peer = LongPeerConfig(sic_prefix_levels=levels, minimum_peers=floors)

        horizons_raw = _mapping(mapping, "horizons")
        if set(horizons_raw) != set(LONG_FORECAST_HORIZONS):
            raise ValueError("horizons must contain exactly 3y and 5y")
        expected_years = {"3y": 3, "5y": 5}
        horizons: dict[str, LongHorizonConfig] = {}
        for horizon in LONG_FORECAST_HORIZONS:
            raw_horizon = _mapping(horizons_raw, horizon)
            years = _positive_int(raw_horizon, "years")
            if years != expected_years[horizon]:
                raise ValueError(f"{horizon} years must remain {expected_years[horizon]}")
            fade = tuple(
                _bounded_float_value(
                    value,
                    label=f"{horizon} fade",
                    minimum=0,
                    maximum=1,
                )
                for value in _list(raw_horizon, "fade")
            )
            if len(fade) != years:
                raise ValueError(f"{horizon} fade must contain one value per year")
            if any(left < right for left, right in zip(fade, fade[1:], strict=False)):
                raise ValueError(f"{horizon} fade must not increase")
            horizons[horizon] = LongHorizonConfig(
                years=years,
                fade=fade,
                multiple_reversion=_bounded_float(
                    raw_horizon,
                    "multiple_reversion",
                    minimum=0,
                    maximum=1,
                ),
            )

        scenarios_raw = _mapping(mapping, "scenarios")
        if set(scenarios_raw) != set(LONG_SCENARIOS):
            raise ValueError("scenarios must contain exactly bear, base, and bull")
        scenarios = {
            name: LongScenarioConfig(
                growth_delta=_finite_float(_mapping(scenarios_raw, name), "growth_delta"),
                reinvestment_multiplier=_positive_float(
                    _mapping(scenarios_raw, name),
                    "reinvestment_multiplier",
                ),
                peer_multiple_multiplier=_positive_float(
                    _mapping(scenarios_raw, name),
                    "peer_multiple_multiplier",
                ),
            )
            for name in LONG_SCENARIOS
        }
        if not (
            scenarios["bear"].growth_delta
            <= scenarios["base"].growth_delta
            <= scenarios["bull"].growth_delta
            and scenarios["bear"].reinvestment_multiplier
            <= scenarios["base"].reinvestment_multiplier
            <= scenarios["bull"].reinvestment_multiplier
            and scenarios["bear"].peer_multiple_multiplier
            <= scenarios["base"].peer_multiple_multiplier
            <= scenarios["bull"].peer_multiple_multiplier
        ):
            raise ValueError("Long forecast scenario assumptions must increase bear/base/bull")

        adjacent_selected_annual_diluted_share_continuity = _optional_capability_enabled(
            mapping,
            "adjacent_selected_annual_diluted_share_continuity",
        )
        newest_quarter_ttm_alias_selection = _optional_capability_enabled(
            mapping,
            "newest_quarter_anchored_homogeneous_ttm_alias_selection",
        )
        joint_invested_capital_pair_selection = _optional_capability_enabled(
            mapping,
            "joint_compatible_invested_capital_pair_selection",
        )
        proven_observation_correction_availability = _optional_capability_enabled(
            mapping,
            "proven_observation_correction_availability",
        )
        fundamentals_config_version = _optional_text(mapping, "fundamentals_config_version")
        maximum_same_date_source_combinations = _optional_positive_int(
            mapping,
            "maximum_same_date_source_combinations",
        )
        if (
            joint_invested_capital_pair_selection is True
            and maximum_same_date_source_combinations != REVIEWED_MAX_SAME_DATE_SOURCE_COMBINATIONS
        ):
            raise ValueError(
                "maximum_same_date_source_combinations must be declared as exactly "
                f"{REVIEWED_MAX_SAME_DATE_SOURCE_COMBINATIONS} when "
                "joint_compatible_invested_capital_pair_selection is enabled"
            )
        if (
            maximum_same_date_source_combinations is not None
            and joint_invested_capital_pair_selection is not True
        ):
            raise ValueError(
                "maximum_same_date_source_combinations only applies when "
                "joint_compatible_invested_capital_pair_selection is enabled"
            )

        return cls(
            schema_version=schema_version,
            version=_required_text(mapping, "version"),
            enabled_scoring_versions=_string_tuple(mapping, "enabled_scoring_versions"),
            price_provider=_required_text(mapping, "price_provider"),
            fundamentals_provider=_required_text(mapping, "fundamentals_provider"),
            return_basis=return_basis,
            dividends_included=dividends_included,
            probability_positive_enabled=probability_enabled,
            eligibility=eligibility,
            metric_families=metric_families,
            growth=growth,
            peer=peer,
            horizons=horizons,
            scenarios=scenarios,
            adjacent_selected_annual_diluted_share_continuity=(
                adjacent_selected_annual_diluted_share_continuity
            ),
            newest_quarter_anchored_homogeneous_ttm_alias_selection=(
                newest_quarter_ttm_alias_selection
            ),
            joint_compatible_invested_capital_pair_selection=(
                joint_invested_capital_pair_selection
            ),
            proven_observation_correction_availability=(proven_observation_correction_availability),
            fundamentals_config_version=fundamentals_config_version,
            maximum_same_date_source_combinations=maximum_same_date_source_combinations,
            raw=mapping,
        )


@dataclass(frozen=True, slots=True)
class LongForecastV4EligibilityConfig:
    maximum_metric_age_days: int
    annual_periods: int
    annual_duration_minimum_days: int
    annual_duration_maximum_days: int
    share_continuity_relative_tolerance: float
    reported_eps_relative_tolerance: float
    fcf_presence_blocks_net_income_fallback: bool


@dataclass(frozen=True, slots=True)
class LongForecastV4MetricFamilyConfig:
    source_metric: str
    multiple_minimum: float
    multiple_maximum: float


@dataclass(frozen=True, slots=True)
class LongForecastV4GrowthConfig:
    target_minimum: float
    target_maximum: float
    peer_minimum: float
    peer_maximum: float
    entity_minimum: float
    entity_maximum: float
    terminal_entity_growth: float
    target_weight: float
    peer_weight: float


@dataclass(frozen=True, slots=True)
class LongForecastV4PeerConfig:
    sic_prefix_levels: tuple[int, ...]
    minimum_cohort: dict[int, int]


@dataclass(frozen=True, slots=True)
class LongForecastV4PathConfig:
    fade: tuple[float, ...]
    multiple_reversion: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LongForecastV4ScenarioConfig:
    growth_delta: float
    dilution_multiplier: float
    peer_multiple_multiplier: float


@dataclass(frozen=True, slots=True)
class LongForecastV4ConfidenceConfig:
    success_status: str
    withheld_status: str
    numeric_value: float
    numeric_semantics: str


@dataclass(frozen=True, slots=True)
class LongForecastV4Config:
    """Strict schema-2 configuration for the research-only long-v4 lane.

    This is deliberately a separate type rather than an extension of
    :class:`LongForecastConfig`.  The legacy schema-1 parser and its effective
    hashes therefore cannot acquire placeholder tax, invested-capital, ROIC,
    reinvestment, or horizon-specific fields from v4.
    """

    schema_version: int
    version: str
    method: str
    method_version: str
    research_status: str
    enabled_scoring_versions: tuple[str, ...]
    price_provider: str
    fundamentals_provider: str
    fundamentals_config_version: str
    fundamentals_config_file_sha256: str
    fundamentals_config_hash: str
    sec_cik_config_version: str
    sec_cik_config_file_sha256: str
    sec_cik_config_hash: str
    sec_mapping_source_sha256: str
    return_basis: str
    dividends_included: bool
    base_currency: str
    fx_conversion: bool
    path_years: int
    probability_positive_enabled: bool
    eligibility: LongForecastV4EligibilityConfig
    metric_families: dict[str, LongForecastV4MetricFamilyConfig]
    growth: LongForecastV4GrowthConfig
    peer: LongForecastV4PeerConfig
    path: LongForecastV4PathConfig
    scenarios: dict[str, LongForecastV4ScenarioConfig]
    confidence: LongForecastV4ConfidenceConfig
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> Self:
        expected_top_level = {
            "schema_version",
            "version",
            "method",
            "method_version",
            "research_status",
            "enabled_scoring_versions",
            "price_provider",
            "fundamentals_provider",
            "fundamentals_config_version",
            "fundamentals_config_file_sha256",
            "fundamentals_config_hash",
            "sec_cik_config_version",
            "sec_cik_config_file_sha256",
            "sec_cik_config_hash",
            "sec_mapping_source_sha256",
            "return_basis",
            "dividends_included",
            "base_currency",
            "fx_conversion",
            "path_years",
            "probability_positive_enabled",
            "eligibility",
            "metric_families",
            "growth",
            "peer",
            "path",
            "scenarios",
            "confidence",
        }
        _require_exact_keys(mapping, expected_top_level, "long-v4")
        if _positive_int(mapping, "schema_version") != 2:
            raise ValueError("Long forecast v4 config schema_version must be 2")
        if _required_text(mapping, "version") != LONG_V4_VERSION:
            raise ValueError(f"Long forecast v4 version must be {LONG_V4_VERSION}")
        if _required_text(mapping, "method") != LONG_V4_METHOD:
            raise ValueError(f"Long forecast v4 method must be {LONG_V4_METHOD}")
        if _required_text(mapping, "method_version") != LONG_V4_VERSION:
            raise ValueError(f"Long forecast v4 method_version must be {LONG_V4_VERSION}")
        if _required_text(mapping, "research_status") != LONG_V4_RESEARCH_STATUS:
            raise ValueError(f"Long forecast v4 research_status must be {LONG_V4_RESEARCH_STATUS}")
        scoring_versions = _string_tuple(mapping, "enabled_scoring_versions")
        if scoring_versions != (LONG_V4_SCORING_VERSION,):
            raise ValueError(
                f"Long forecast v4 scoring identity must be exactly {LONG_V4_SCORING_VERSION}"
            )
        if _required_text(mapping, "price_provider") != "twelve_data":
            raise ValueError("Long forecast v4 price_provider must be twelve_data")
        if _required_text(mapping, "fundamentals_provider") != "sec":
            raise ValueError("Long forecast v4 fundamentals_provider must be sec")
        if _required_text(mapping, "fundamentals_config_version") != LONG_V4_FUNDAMENTALS_VERSION:
            raise ValueError(
                "Long forecast v4 fundamentals_config_version must be "
                f"{LONG_V4_FUNDAMENTALS_VERSION}"
            )
        if (
            _required_text(mapping, "fundamentals_config_file_sha256")
            != LONG_V4_FUNDAMENTALS_CONFIG_FILE_SHA256
        ):
            raise ValueError("Long forecast v4 SEC fundamentals config bytes changed")
        if _required_text(mapping, "fundamentals_config_hash") != LONG_V4_FUNDAMENTALS_CONFIG_HASH:
            raise ValueError("Long forecast v4 SEC fundamentals effective config changed")
        if _required_text(mapping, "sec_cik_config_version") != LONG_V4_SEC_CIK_CONFIG_VERSION:
            raise ValueError("Long forecast v4 SEC CIK config version changed")
        if (
            _required_text(mapping, "sec_cik_config_file_sha256")
            != LONG_V4_SEC_CIK_CONFIG_FILE_SHA256
        ):
            raise ValueError("Long forecast v4 SEC CIK config bytes changed")
        if _required_text(mapping, "sec_cik_config_hash") != LONG_V4_SEC_CIK_CONFIG_HASH:
            raise ValueError("Long forecast v4 SEC CIK effective config changed")
        if (
            _required_text(mapping, "sec_mapping_source_sha256")
            != LONG_V4_SEC_MAPPING_SOURCE_SHA256
        ):
            raise ValueError("Long forecast v4 SEC mapping source changed")
        if _required_text(mapping, "return_basis") != "split_adjusted_price_return":
            raise ValueError("Long forecast v4 requires split_adjusted_price_return")
        if mapping.get("dividends_included") is not False:
            raise ValueError("Long forecast v4 must exclude dividends")
        if _required_text(mapping, "base_currency") != "USD":
            raise ValueError("Long forecast v4 base_currency must be USD")
        if mapping.get("fx_conversion") is not False:
            raise ValueError("Long forecast v4 does not perform FX conversion")
        if _positive_int(mapping, "path_years") != 5:
            raise ValueError("Long forecast v4 path_years must be 5")
        if mapping.get("probability_positive_enabled") is not False:
            raise ValueError("Long forecast v4 probability must remain unavailable")

        eligibility_raw = _mapping(mapping, "eligibility")
        _require_exact_keys(
            eligibility_raw,
            {
                "maximum_metric_age_days",
                "annual_periods",
                "annual_duration_minimum_days",
                "annual_duration_maximum_days",
                "share_continuity_relative_tolerance",
                "reported_eps_relative_tolerance",
                "fcf_presence_blocks_net_income_fallback",
            },
            "long-v4 eligibility",
        )
        eligibility = LongForecastV4EligibilityConfig(
            maximum_metric_age_days=_positive_int(eligibility_raw, "maximum_metric_age_days"),
            annual_periods=_positive_int(eligibility_raw, "annual_periods"),
            annual_duration_minimum_days=_positive_int(
                eligibility_raw, "annual_duration_minimum_days"
            ),
            annual_duration_maximum_days=_positive_int(
                eligibility_raw, "annual_duration_maximum_days"
            ),
            share_continuity_relative_tolerance=_bounded_float(
                eligibility_raw,
                "share_continuity_relative_tolerance",
                minimum=0,
                maximum=0.15,
                minimum_inclusive=False,
            ),
            reported_eps_relative_tolerance=_bounded_float(
                eligibility_raw,
                "reported_eps_relative_tolerance",
                minimum=0,
                maximum=0.15,
                minimum_inclusive=False,
            ),
            fcf_presence_blocks_net_income_fallback=_required_bool(
                eligibility_raw, "fcf_presence_blocks_net_income_fallback"
            ),
        )
        if eligibility.annual_periods != 4:
            raise ValueError("Long forecast v4 annual_periods must be 4")
        if (
            eligibility.annual_duration_minimum_days != 350
            or eligibility.annual_duration_maximum_days != 380
        ):
            raise ValueError("Long forecast v4 annual duration bounds must be 350..380 days")

        metric_raw = _mapping(mapping, "metric_families")
        family_names = {"fcf_per_share", "net_income_per_share"}
        _require_exact_keys(metric_raw, family_names, "long-v4 metric_families")
        metric_families: dict[str, LongForecastV4MetricFamilyConfig] = {}
        expected_source = {
            "fcf_per_share": "free_cash_flow",
            "net_income_per_share": "net_income",
        }
        expected_bounds = {
            "fcf_per_share": (3.0, 60.0),
            "net_income_per_share": (5.0, 50.0),
        }
        for name in sorted(family_names):
            raw_family = _mapping(metric_raw, name)
            _require_exact_keys(
                raw_family,
                {"source_metric", "multiple_minimum", "multiple_maximum"},
                f"long-v4 {name}",
            )
            source_metric = _required_text(raw_family, "source_metric")
            minimum = _positive_float(raw_family, "multiple_minimum")
            maximum = _positive_float(raw_family, "multiple_maximum")
            if (
                source_metric != expected_source[name]
                or (minimum, maximum) != expected_bounds[name]
            ):
                raise ValueError(f"Long forecast v4 {name} identity or bounds changed")
            metric_families[name] = LongForecastV4MetricFamilyConfig(
                source_metric=source_metric,
                multiple_minimum=minimum,
                multiple_maximum=maximum,
            )

        growth_raw = _mapping(mapping, "growth")
        growth_fields = {
            "target_minimum",
            "target_maximum",
            "peer_minimum",
            "peer_maximum",
            "entity_minimum",
            "entity_maximum",
            "terminal_entity_growth",
            "target_weight",
            "peer_weight",
        }
        _require_exact_keys(growth_raw, growth_fields, "long-v4 growth")
        growth = LongForecastV4GrowthConfig(
            **{name: _finite_float(growth_raw, name) for name in growth_fields}
        )
        if (
            (growth.target_minimum, growth.target_maximum) != (-0.20, 0.25)
            or (growth.peer_minimum, growth.peer_maximum) != (-0.15, 0.25)
            or (growth.entity_minimum, growth.entity_maximum) != (-0.15, 0.25)
            or growth.terminal_entity_growth != 0.025
            or not math.isclose(growth.target_weight, 0.5)
            or not math.isclose(growth.peer_weight, 0.5)
        ):
            raise ValueError("Long forecast v4 growth policy constants changed")

        peer_raw = _mapping(mapping, "peer")
        _require_exact_keys(peer_raw, {"sic_prefix_levels", "minimum_cohort"}, "long-v4 peer")
        levels = tuple(
            _positive_int_value(value, label="sic_prefix_levels")
            for value in _list(peer_raw, "sic_prefix_levels")
        )
        floors_raw = _mapping(peer_raw, "minimum_cohort")
        floors = {
            int(level): _positive_int_value(value, label=f"minimum_cohort[{level}]")
            for level, value in floors_raw.items()
        }
        if levels != (4, 3, 2) or floors != {4: 3, 3: 5, 2: 8}:
            raise ValueError("Long forecast v4 peer lock policy must remain 4/3/2 -> 3/5/8")
        peer = LongForecastV4PeerConfig(levels, floors)

        path_raw = _mapping(mapping, "path")
        _require_exact_keys(path_raw, {"fade", "multiple_reversion"}, "long-v4 path")
        fade = tuple(
            _bounded_float_value(value, label="fade", minimum=0, maximum=1)
            for value in _list(path_raw, "fade")
        )
        reversion = tuple(
            _bounded_float_value(value, label="multiple_reversion", minimum=0, maximum=1)
            for value in _list(path_raw, "multiple_reversion")
        )
        if fade != (0.8, 0.6, 0.4, 0.2, 0.0):
            raise ValueError("Long forecast v4 fade path changed")
        if reversion != (0.14, 0.28, 0.42, 0.56, 0.70):
            raise ValueError("Long forecast v4 multiple-reversion path changed")
        path = LongForecastV4PathConfig(fade, reversion)

        scenarios_raw = _mapping(mapping, "scenarios")
        _require_exact_keys(scenarios_raw, set(LONG_SCENARIOS), "long-v4 scenarios")
        expected_scenarios = {
            "bear": (-0.04, 1.25, 0.80),
            "base": (0.00, 1.00, 1.00),
            "bull": (0.03, 0.75, 1.15),
        }
        scenarios: dict[str, LongForecastV4ScenarioConfig] = {}
        for name in LONG_SCENARIOS:
            raw_scenario = _mapping(scenarios_raw, name)
            _require_exact_keys(
                raw_scenario,
                {"growth_delta", "dilution_multiplier", "peer_multiple_multiplier"},
                f"long-v4 {name} scenario",
            )
            scenario = LongForecastV4ScenarioConfig(
                growth_delta=_finite_float(raw_scenario, "growth_delta"),
                dilution_multiplier=_positive_float(raw_scenario, "dilution_multiplier"),
                peer_multiple_multiplier=_positive_float(raw_scenario, "peer_multiple_multiplier"),
            )
            if (
                scenario.growth_delta,
                scenario.dilution_multiplier,
                scenario.peer_multiple_multiplier,
            ) != expected_scenarios[name]:
                raise ValueError(f"Long forecast v4 {name} scenario constants changed")
            scenarios[name] = scenario

        confidence_raw = _mapping(mapping, "confidence")
        _require_exact_keys(
            confidence_raw,
            {
                "success_status",
                "withheld_status",
                "numeric_value",
                "numeric_semantics",
            },
            "long-v4 confidence",
        )
        confidence = LongForecastV4ConfidenceConfig(
            success_status=_required_text(confidence_raw, "success_status"),
            withheld_status=_required_text(confidence_raw, "withheld_status"),
            numeric_value=_finite_float(confidence_raw, "numeric_value"),
            numeric_semantics=_required_text(confidence_raw, "numeric_semantics"),
        )
        if confidence != LongForecastV4ConfidenceConfig(
            success_status="not_estimated_uncalibrated",
            withheld_status="not_estimated_insufficient",
            numeric_value=0.0,
            numeric_semantics="zero_is_unavailable_sentinel",
        ):
            raise ValueError("Long forecast v4 confidence semantics changed")

        return cls(
            schema_version=2,
            version=LONG_V4_VERSION,
            method=LONG_V4_METHOD,
            method_version=LONG_V4_VERSION,
            research_status=LONG_V4_RESEARCH_STATUS,
            enabled_scoring_versions=scoring_versions,
            price_provider="twelve_data",
            fundamentals_provider="sec",
            fundamentals_config_version=LONG_V4_FUNDAMENTALS_VERSION,
            fundamentals_config_file_sha256=LONG_V4_FUNDAMENTALS_CONFIG_FILE_SHA256,
            fundamentals_config_hash=LONG_V4_FUNDAMENTALS_CONFIG_HASH,
            sec_cik_config_version=LONG_V4_SEC_CIK_CONFIG_VERSION,
            sec_cik_config_file_sha256=LONG_V4_SEC_CIK_CONFIG_FILE_SHA256,
            sec_cik_config_hash=LONG_V4_SEC_CIK_CONFIG_HASH,
            sec_mapping_source_sha256=LONG_V4_SEC_MAPPING_SOURCE_SHA256,
            return_basis="split_adjusted_price_return",
            dividends_included=False,
            base_currency="USD",
            fx_conversion=False,
            path_years=5,
            probability_positive_enabled=False,
            eligibility=eligibility,
            metric_families=metric_families,
            growth=growth,
            peer=peer,
            path=path,
            scenarios=scenarios,
            confidence=confidence,
            raw=mapping,
        )


def default_long_forecast_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "forecasts" / "us-sec-long-v2.yml"


def long_forecast_config_path(version: str) -> Path:
    """Resolve a pinned long forecast configuration file by version name."""
    return Path(__file__).resolve().parents[3] / "config" / "forecasts" / f"{version}.yml"


def load_long_forecast_config(path: Path | None = None) -> LongForecastConfig:
    config_path = path or default_long_forecast_config_path()
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        # ``from None`` suppresses the chained PyYAML error entirely, so the
        # offending source line cannot reach a traceback either.
        raise LongForecastConfigParseError(
            f"Long forecast config is not valid YAML: {config_path}"
        ) from None
    if not isinstance(data, dict):
        raise ValueError(f"Long forecast config must be a mapping: {config_path}")
    return LongForecastConfig.from_mapping(cast(dict[str, Any], data))


def long_forecast_config_hash(config: LongForecastConfig) -> str:
    effective = asdict(config)
    effective.pop("raw")
    for key in (*LONG_OPTIONAL_CAPABILITY_FIELDS, *LONG_OPTIONAL_BINDING_FIELDS):
        if effective.get(key) is None:
            effective.pop(key, None)
    payload = json.dumps(
        effective,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def long_forecast_v4_config_path() -> Path:
    """Return the only path from which the research-only v4 config may load."""
    return Path(__file__).resolve().parents[3] / "config" / "forecasts" / "us-sec-long-v4.yml"


def load_long_forecast_v4_config(path: Path) -> LongForecastV4Config:
    """Load v4 only from its exact repository-owned, byte-pinned path.

    V4 is not a default and must never be selected by a basename copied into
    another directory.  Requiring the canonical path and bytes also prevents
    a caller from presenting an untracked lookalike as the reviewed contract.
    """
    canonical = long_forecast_v4_config_path().resolve()
    if path.resolve() != canonical:
        raise ValueError(
            "us-sec-long-v4 requires the exact tracked config/forecasts/us-sec-long-v4.yml path"
        )
    try:
        raw_bytes = canonical.read_bytes()
        data = yaml.safe_load(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise LongForecastConfigParseError(
            "Long forecast v4 config could not be read as valid UTF-8 YAML"
        ) from None
    if not isinstance(data, dict):
        raise ValueError("Long forecast v4 config must be a mapping")
    if hashlib.sha256(raw_bytes).hexdigest() != LONG_V4_CONFIG_FILE_SHA256:
        raise ValueError("Long forecast v4 config bytes do not match the reviewed identity")
    config = LongForecastV4Config.from_mapping(cast(dict[str, Any], data))
    if long_forecast_v4_config_hash(config) != LONG_V4_EFFECTIVE_CONFIG_HASH:
        raise ValueError("Long forecast v4 effective config identity does not match the review")
    load_long_v4_sec_fundamentals_config(config)
    load_long_v4_sec_cik_config(config)
    return config


def long_forecast_v4_config_hash(config: LongForecastV4Config) -> str:
    effective = asdict(config)
    effective.pop("raw")
    payload = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_long_v4_sec_fundamentals_config(
    config: LongForecastV4Config,
) -> SecFundamentalsConfig:
    """Load and verify the one SEC normalization contract admitted by v4."""
    canonical = default_sec_fundamentals_config_path().resolve()
    try:
        raw_bytes = canonical.read_bytes()
    except OSError:
        raise LongForecastConfigParseError(
            "The canonical SEC fundamentals config could not be read"
        ) from None
    physical_hash = hashlib.sha256(raw_bytes).hexdigest()
    if (
        physical_hash != LONG_V4_FUNDAMENTALS_CONFIG_FILE_SHA256
        or config.fundamentals_config_file_sha256 != physical_hash
    ):
        raise ValueError("Long forecast v4 SEC fundamentals config bytes changed")
    sec_config = load_sec_fundamentals_config(canonical)
    if (
        sec_config.config_version != LONG_V4_FUNDAMENTALS_VERSION
        or config.fundamentals_config_version != sec_config.config_version
        or sec_config.config_hash != LONG_V4_FUNDAMENTALS_CONFIG_HASH
        or config.fundamentals_config_hash != sec_config.config_hash
    ):
        raise ValueError("Long forecast v4 SEC fundamentals effective config changed")
    return sec_config


def load_long_v4_sec_cik_config(config: LongForecastV4Config) -> SecCikConfig:
    """Load and physically bind the independently reviewed SEC CIK authority."""
    canonical = default_sec_cik_mapping_path().resolve()
    try:
        raw_bytes = canonical.read_bytes()
    except OSError:
        raise LongForecastConfigParseError(
            "The canonical SEC CIK config could not be read"
        ) from None
    physical_hash = hashlib.sha256(raw_bytes).hexdigest()
    if (
        physical_hash != LONG_V4_SEC_CIK_CONFIG_FILE_SHA256
        or config.sec_cik_config_file_sha256 != physical_hash
    ):
        raise ValueError("Long forecast v4 SEC CIK config bytes changed")
    cik_config = load_sec_cik_config(canonical)
    if (
        cik_config.config_version != LONG_V4_SEC_CIK_CONFIG_VERSION
        or config.sec_cik_config_version != cik_config.config_version
        or cik_config.config_hash != LONG_V4_SEC_CIK_CONFIG_HASH
        or config.sec_cik_config_hash != cik_config.config_hash
        or cik_config.source_sha256 != LONG_V4_SEC_MAPPING_SOURCE_SHA256
        or config.sec_mapping_source_sha256 != cik_config.source_sha256
    ):
        raise ValueError("Long forecast v4 SEC CIK authority changed")
    return cik_config


def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return cast(dict[str, Any], value)


def _require_exact_keys(mapping: dict[str, Any], expected: set[str], label: str) -> None:
    if set(mapping) != expected:
        raise ValueError(f"{label} keys do not match the reviewed schema")


def _list(mapping: dict[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    return value


def _required_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _string_tuple(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    values = _list(mapping, key)
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{key} contains a blank value")
    return normalized


def _required_bool(mapping: dict[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def _optional_capability_enabled(mapping: dict[str, Any], key: str) -> bool | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping with exactly an enabled boolean")
    if set(value) != {"enabled"}:
        raise ValueError(f"{key} must be a mapping with exactly an enabled boolean")
    enabled = value["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError(f"{key}.enabled must be boolean")
    return enabled


def _optional_text(mapping: dict[str, Any], key: str) -> str | None:
    """Return an optional pinned identifier, or ``None`` when absent.

    An absent key is a distinct configuration from a present one and is
    removed from the effective hash payload, so adding this field to a new
    version cannot change an already-frozen version's hash.
    """
    if key not in mapping:
        return None
    return _required_text(mapping, key)


def _optional_positive_int(mapping: dict[str, Any], key: str) -> int | None:
    """Return an optional positive integer, or ``None`` when absent.

    Absent is a distinct configuration from present and is removed from the
    effective hash payload, so adding this field to a new version cannot move
    an already-frozen version's hash.
    """
    if key not in mapping:
        return None
    return _positive_int(mapping, key)


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    return _positive_int_value(mapping.get(key), label=key)


def _nonnegative_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _integer_at_least(
    mapping: dict[str, Any],
    key: str,
    *,
    minimum: int,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} must be an integer of at least {minimum}")
    return value


def _positive_int_value(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_float(mapping: dict[str, Any], key: str) -> float:
    return _finite_float_value(mapping.get(key), label=key)


def _positive_float(mapping: dict[str, Any], key: str) -> float:
    value = _finite_float(mapping, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _bounded_float(
    mapping: dict[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    return _bounded_float_value(
        mapping.get(key),
        label=key,
        minimum=minimum,
        maximum=maximum,
        minimum_inclusive=minimum_inclusive,
        maximum_inclusive=maximum_inclusive,
    )


def _bounded_float_value(
    value: object,
    *,
    label: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    normalized = _finite_float_value(value, label=label)
    lower_valid = normalized >= minimum if minimum_inclusive else normalized > minimum
    upper_valid = normalized <= maximum if maximum_inclusive else normalized < maximum
    if not lower_valid or not upper_valid:
        left = "[" if minimum_inclusive else "("
        right = "]" if maximum_inclusive else ")"
        raise ValueError(f"{label} must be in {left}{minimum}, {maximum}{right}")
    return normalized


def _finite_float_value(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def _sic_range(value: object) -> tuple[int, int]:
    if not isinstance(value, str) or "-" not in value:
        raise ValueError("unsupported_sic_ranges entries must use NNNN-NNNN")
    raw_start, raw_end = value.split("-", maxsplit=1)
    if len(raw_start) != 4 or len(raw_end) != 4 or not raw_start.isdigit() or not raw_end.isdigit():
        raise ValueError("unsupported_sic_ranges entries must use NNNN-NNNN")
    start = int(raw_start)
    end = int(raw_end)
    if start > end:
        raise ValueError("unsupported_sic_ranges must increase")
    return start, end

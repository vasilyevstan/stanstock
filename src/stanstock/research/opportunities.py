from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml

from stanstock.research.models import Recommendation, StockAnalysis

POLICY_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "opportunities" / "great-opportunity-v1.yml"
)


@dataclass(frozen=True, slots=True)
class OpportunityModePolicy:
    label: str
    horizon: str
    min_score: Decimal
    min_confidence: Decimal
    allowed_risk_classes: frozenset[str]
    require_positive_base_case: bool
    require_fundamentals: bool


@dataclass(frozen=True, slots=True)
class OpportunityPolicy:
    version: str
    modes: dict[str, OpportunityModePolicy]


@dataclass(frozen=True, slots=True)
class OpportunityAssessment:
    eligible: bool
    label: str
    policy_version: str
    horizon: str
    criteria: dict[str, bool]


@lru_cache(maxsize=1)
def load_opportunity_policy() -> OpportunityPolicy:
    raw = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Opportunity policy must be a mapping: {POLICY_PATH}")
    modes_raw = raw.get("modes")
    if not isinstance(modes_raw, dict) or not modes_raw:
        raise ValueError("Opportunity policy requires a non-empty modes mapping")
    modes: dict[str, OpportunityModePolicy] = {}
    for mode, value in modes_raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Opportunity policy mode {mode!r} must be a mapping")
        horizon = str(value["horizon"])
        if horizon not in {"short", "medium", "long"}:
            raise ValueError(f"Unsupported opportunity horizon {horizon!r}")
        allowed = frozenset(str(item) for item in cast(list[object], value["allowed_risk_classes"]))
        if not allowed or not allowed <= {"low", "medium", "high", "very_high"}:
            raise ValueError(f"Invalid allowed risk classes for opportunity mode {mode!r}")
        modes[str(mode)] = OpportunityModePolicy(
            label=str(value["label"]),
            horizon=horizon,
            min_score=Decimal(str(value["min_score"])),
            min_confidence=Decimal(str(value["min_confidence"])),
            allowed_risk_classes=allowed,
            require_positive_base_case=bool(value["require_positive_base_case"]),
            require_fundamentals=bool(value["require_fundamentals"]),
        )
    return OpportunityPolicy(version=str(raw["version"]), modes=modes)


def assess_opportunity(analysis: StockAnalysis) -> OpportunityAssessment:
    policy = load_opportunity_policy()
    data_quality = analysis.data_quality if isinstance(analysis.data_quality, dict) else {}
    analysis_mode = str(data_quality.get("analysis_mode") or "full")
    mode_policy = policy.modes.get(analysis_mode, policy.modes.get("full"))
    if mode_policy is None:
        raise ValueError("Opportunity policy has no full-mode fallback")
    scenario = _scenario_for_horizon(analysis, mode_policy.horizon)
    base_case = _decimal_or_none(scenario.get("base")) if isinstance(scenario, dict) else None
    criteria = {
        "buy_recommendation": analysis.recommendation == Recommendation.BUY,
        "score": analysis.overall_score >= mode_policy.min_score,
        "confidence": analysis.confidence >= mode_policy.min_confidence,
        "risk": analysis.risk_class in mode_policy.allowed_risk_classes,
        "base_case": (
            base_case is not None and base_case > 0
            if mode_policy.require_positive_base_case
            else True
        ),
        "fundamentals": (
            data_quality.get("fundamentals_used") is True
            if mode_policy.require_fundamentals
            else True
        ),
    }
    return OpportunityAssessment(
        eligible=all(criteria.values()),
        label=mode_policy.label,
        policy_version=policy.version,
        horizon=mode_policy.horizon,
        criteria=criteria,
    )


def _scenario_for_horizon(analysis: StockAnalysis, horizon: str) -> dict[str, Any]:
    value = getattr(analysis, f"{horizon}_scenario")
    return value if isinstance(value, dict) else {}


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None

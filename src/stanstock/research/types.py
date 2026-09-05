from __future__ import annotations

from dataclasses import dataclass, field

ScoreComponent = str
Horizon = str


@dataclass(frozen=True, slots=True)
class ResearchValues:
    values: dict[str, float]
    missing: dict[str, str] = field(default_factory=dict)

    def present(self, name: str) -> bool:
        return name in self.values

    def get(self, name: str) -> float | None:
        return self.values.get(name)


@dataclass(frozen=True, slots=True)
class IndicatorResult(ResearchValues):
    observation_count: int = 0
    last_date: object | None = None


@dataclass(frozen=True, slots=True)
class FundamentalInputs:
    current: dict[str, float]
    previous: dict[str, float] = field(default_factory=dict)
    history: tuple[dict[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class ComponentScores:
    components: dict[str, float]
    factor_scores: dict[str, float]
    missing: dict[str, str]
    coverage: float


@dataclass(frozen=True, slots=True)
class AggregateScore:
    overall: float
    horizon_scores: dict[str, float]
    confidence: float
    confidence_status: str
    component_scores: ComponentScores
    missingness_penalty: float
    freshness_penalty: float


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    score: float | None
    risk_class: str
    insufficiency_reason: str = ""


@dataclass(frozen=True, slots=True)
class RecommendationDecision:
    recommendation: str
    gates: dict[str, bool]


@dataclass(frozen=True, slots=True)
class Scenario:
    bear: float | None
    base: float | None
    bull: float | None
    probability_positive: float | None
    confidence: float
    confidence_status: str
    insufficiency_reason: str
    method: str

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "bear": self.bear,
            "base": self.base,
            "bull": self.bull,
            "probability_positive": self.probability_positive,
            "confidence": self.confidence,
            "confidence_status": self.confidence_status,
            "insufficiency_reason": self.insufficiency_reason,
            "method": self.method,
        }

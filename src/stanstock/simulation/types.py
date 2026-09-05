from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl
import yaml


class SimulationMode(StrEnum):
    BACKTEST = "backtest"
    PORTFOLIO = "portfolio"


class SimulationGrade(StrEnum):
    RESEARCH = "research"
    OBSERVED = "observed"


class RebalanceFrequency(StrEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    NEVER = "never"


class ExecutionPriceBasis(StrEnum):
    NEXT_OPEN = "next_open"
    NEXT_CLOSE = "next_close"
    NEXT_ELIGIBLE = "next_eligible"


class MissingPricePolicy(StrEnum):
    CARRY_FORWARD = "carry_forward"
    MARK_UNRESOLVED = "mark_unresolved"
    DROP = "drop"
    FAIL = "fail"


class MissingPriceError(Exception):
    """Raised when a price observation is missing and policy is FAIL."""


class SimulationWorkflowError(ValueError):
    """Raised when a simulation workflow, input building, or validation fails."""


@dataclass
class SimulationConfig:
    name: str = "simulation"
    mode: SimulationMode = SimulationMode.BACKTEST
    grade: SimulationGrade = SimulationGrade.OBSERVED
    starting_capital: float = 100_000.0
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    execution_basis: ExecutionPriceBasis = ExecutionPriceBasis.NEXT_CLOSE
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    top_n: int | None = None
    selected_symbols: list[str] | None = None
    missing_price_policy: MissingPricePolicy = MissingPricePolicy.MARK_UNRESOLVED
    risk_free_rate: float = 0.0
    benchmark_symbol: str | None = None
    base_currency: str | None = None
    cash_buffer_bps: float = 0.0
    tie_breaker: str = "symbol_asc"
    description: str = ""
    custom_parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def simulation_kind(self) -> str:
        if self.mode == SimulationMode.BACKTEST:
            if self.grade == SimulationGrade.RESEARCH:
                return "research_backtest"
            return "observed_backtest"
        return "portfolio_simulation"

    VALID_TIE_BREAKERS = {
        "symbol_asc",
        "symbol_desc",
        "listing_id_asc",
        "listing_id_desc",
    }

    def validate(self) -> None:
        if self.starting_capital <= 0:
            raise ValueError(f"starting_capital must be positive, got {self.starting_capital}")
        if self.transaction_cost_bps < 0:
            raise ValueError(
                f"transaction_cost_bps must be non-negative, got {self.transaction_cost_bps}"
            )
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be non-negative, got {self.slippage_bps}")
        if self.cash_buffer_bps < 0 or self.cash_buffer_bps >= 10000:
            raise ValueError(f"cash_buffer_bps must be in [0, 10000), got {self.cash_buffer_bps}")
        if self.top_n is None and not self.selected_symbols:
            raise ValueError(
                "Either top_n or selected_symbols must be provided in SimulationConfig"
            )
        if self.top_n is not None and self.top_n <= 0:
            raise ValueError(f"top_n must be positive, got {self.top_n}")
        if self.tie_breaker not in self.VALID_TIE_BREAKERS:
            raise ValueError(
                f"Invalid tie_breaker '{self.tie_breaker}'. Must be one of "
                f"{sorted(self.VALID_TIE_BREAKERS)}"
            )
        if self.base_currency is not None and (
            len(self.base_currency) != 3 or not self.base_currency.isalpha()
        ):
            raise ValueError(
                f"base_currency must be a three-letter currency code, got {self.base_currency!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode.value,
            "grade": self.grade.value,
            "starting_capital": self.starting_capital,
            "rebalance_frequency": self.rebalance_frequency.value,
            "execution_basis": self.execution_basis.value,
            "transaction_cost_bps": self.transaction_cost_bps,
            "slippage_bps": self.slippage_bps,
            "top_n": self.top_n,
            "selected_symbols": self.selected_symbols,
            "missing_price_policy": self.missing_price_policy.value,
            "risk_free_rate": self.risk_free_rate,
            "benchmark_symbol": self.benchmark_symbol,
            "base_currency": self.base_currency,
            "cash_buffer_bps": self.cash_buffer_bps,
            "tie_breaker": self.tie_breaker,
            "description": self.description,
            "custom_parameters": self.custom_parameters,
            "simulation_kind": self.simulation_kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationConfig:
        config_data = dict(data)
        config_data.pop("simulation_kind", None)
        mode = SimulationMode(config_data.pop("mode", "backtest"))
        grade = SimulationGrade(config_data.pop("grade", "observed"))
        rebalance_frequency = RebalanceFrequency(config_data.pop("rebalance_frequency", "monthly"))
        execution_basis = ExecutionPriceBasis(config_data.pop("execution_basis", "next_close"))
        missing_price_policy = MissingPricePolicy(
            config_data.pop("missing_price_policy", "mark_unresolved")
        )

        return cls(
            mode=mode,
            grade=grade,
            rebalance_frequency=rebalance_frequency,
            execution_basis=execution_basis,
            missing_price_policy=missing_price_policy,
            **config_data,
        )

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, content_or_path: str | Path) -> SimulationConfig:
        if isinstance(content_or_path, Path) or (
            isinstance(content_or_path, str)
            and "\n" not in content_or_path
            and Path(content_or_path).exists()
        ):
            text = Path(content_or_path).read_text(encoding="utf-8")
        else:
            text = str(content_or_path)
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"YAML must deserialize to a dict, got {type(parsed)}")
        return cls.from_dict(parsed)


class FxAttributionStatus(StrEnum):
    """Whether an FX contribution figure may be reported for a run."""

    NOT_APPLICABLE = "not_applicable"
    EXACT = "exact"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class FxAttribution:
    """Split of a converted run's return into stock return and FX contribution.

    The split is produced by revaluing the *same* quantity path at each
    native currency's reference rate (its rate on the run's first simulated
    date) instead of at the dated rate: every cash movement is mirrored at
    the reference rate and every open position is revalued at it. Both
    tracks therefore start from identical capital, which makes
    ``local_currency_cumulative_return + contribution_return`` equal the
    reported cumulative return by construction rather than by approximation.

    When any part of that mirror cannot be computed exactly -- a cash
    settlement whose FX basis is not established, or a reference rate that is
    missing -- the status becomes ``unavailable`` and both figures stay
    ``None``. A partially-known split reported as a number would be fake
    precision, not a measurement.
    """

    status: FxAttributionStatus
    detail: str
    conversion_applied: bool = False
    native_currencies: tuple[str, ...] = ()
    max_carry_days_used: int | None = None
    local_currency_cumulative_return: float | None = None
    contribution_return: float | None = None

    @classmethod
    def not_applicable(cls) -> FxAttribution:
        return cls(
            status=FxAttributionStatus.NOT_APPLICABLE,
            detail="No FX conversion was applied; every value is already in the base currency.",
        )


@dataclass
class UnresolvedObservation:
    listing_id: str
    symbol: str
    date: dt.date
    event_type: str
    last_known_price: float
    last_known_date: dt.date
    quantity_held: float
    action_taken: str
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "symbol": self.symbol,
            "date": self.date.isoformat(),
            "event_type": self.event_type,
            "last_known_price": self.last_known_price,
            "last_known_date": self.last_known_date.isoformat(),
            "quantity_held": self.quantity_held,
            "action_taken": self.action_taken,
            "details": self.details,
        }


@dataclass
class TradeRecord:
    listing_id: str
    symbol: str
    trade_date: dt.date
    side: str
    quantity: float
    price: float
    gross_value: float
    costs: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "symbol": self.symbol,
            "trade_date": self.trade_date.isoformat(),
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "gross_value": self.gross_value,
            "costs": self.costs,
        }


@dataclass
class HoldingRecord:
    listing_id: str
    symbol: str
    observation_date: dt.date
    quantity: float
    price: float
    market_value: float
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "symbol": self.symbol,
            "observation_date": self.observation_date.isoformat(),
            "quantity": self.quantity,
            "price": self.price,
            "market_value": self.market_value,
            "weight": self.weight,
        }


@dataclass
class SimulationMetrics:
    cumulative_return: float
    cagr: float | None
    annualized_volatility: float
    sharpe_ratio: float | None
    max_drawdown: float
    turnover: float
    annualized_turnover: float
    positive_period_rate: float
    total_trades: int
    start_date: dt.date
    end_date: dt.date
    duration_days: int
    starting_capital: float
    ending_capital: float
    unresolved_count: int
    mode: str
    grade: str
    simulation_kind: str
    base_currency: str | None = None
    fx_conversion_applied: bool = False
    fx_native_currencies: list[str] | None = None
    fx_max_carry_days_used: int | None = None
    fx_local_currency_cumulative_return: float | None = None
    fx_contribution_return: float | None = None
    fx_attribution_status: str | None = None
    fx_attribution_detail: str | None = None
    benchmark_cumulative_return: float | None = None
    benchmark_cagr: float | None = None
    benchmark_annualized_volatility: float | None = None
    benchmark_sharpe: float | None = None
    benchmark_max_drawdown: float | None = None
    excess_return: float | None = None
    alpha: float | None = None
    beta: float | None = None
    tracking_error: float | None = None
    information_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_date"] = self.start_date.isoformat()
        data["end_date"] = self.end_date.isoformat()
        return data


@dataclass
class SimulationResult:
    daily_curves: pl.DataFrame
    holdings: pl.DataFrame
    trades: pl.DataFrame
    metrics: SimulationMetrics
    unresolved_observations: list[UnresolvedObservation]
    config: SimulationConfig
    input_hash: str

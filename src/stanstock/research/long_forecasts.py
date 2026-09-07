from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

from stanstock.data.asof import AsOfData
from stanstock.data.models import (
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    Listing,
)
from stanstock.data.sec_config import SecFundamentalsConfig, load_sec_fundamentals_config
from stanstock.data.sec_fundamentals import (
    FundamentalValue,
    SecFundamentalSeries,
    build_sec_fundamental_series,
)
from stanstock.research.long_forecast_config import (
    LONG_FORECAST_HORIZONS,
    LONG_SCENARIOS,
    LongForecastConfig,
    LongHorizonConfig,
    LongMetricFamilyConfig,
    LongScenarioConfig,
    long_forecast_config_hash,
)
from stanstock.research.types import Scenario

METHOD_NAME = "sec_per_share_growth_multiple_reversion"
MAX_SQL_IN_ITEMS = 500


@dataclass(frozen=True, slots=True)
class LongForecast:
    scenario: Scenario
    calculation: dict[str, Any]
    source_assets: tuple[DataAsset, ...]

    def scenario_payload(self) -> dict[str, Any]:
        return {
            **self.scenario.as_dict(),
            "method_version": self.calculation["method_version"],
            "metric_family": self.calculation.get("metric_family"),
            "support": self.calculation.get("support", {}),
            "annualized_return": self.calculation.get("annualized_returns", {}),
            "formula_inputs": self.calculation.get("formula_inputs", {}),
            "split_basis": self.calculation.get("split_basis", {}),
            "target_classification": self.calculation.get("target_classification"),
            "return_basis": self.calculation["return_basis"],
            "probability_status": self.calculation["probability_status"],
        }


@dataclass(frozen=True, slots=True)
class _PerSharePoint:
    value: float
    period_start: date
    period_end: date
    available_at: datetime
    fact_ids: tuple[str, ...]
    accessions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MetricEvidence:
    family: str
    current_per_share: float
    current_period_start: date
    current_period_end: date
    current_multiple_raw: float
    current_multiple: float
    annual_points: tuple[_PerSharePoint, ...]
    historical_growth_raw: float
    historical_growth: float
    growth_observations: tuple[dict[str, Any], ...]
    share_consistency: tuple[dict[str, Any], ...]
    fact_ids: tuple[str, ...]
    accessions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MetricAssessment:
    status: str
    reason: str
    evidence: _MetricEvidence | None
    fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _InvestedCapital:
    value: float
    period_end: date
    debt: float
    equity: float
    cash: float
    debt_method: str
    debt_components: tuple[str, ...]
    source_basis: tuple[tuple[str, str], ...]
    fact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SustainableGrowth:
    tax_rate_raw: float
    tax_rate: float
    nopat: float
    beginning_invested_capital: _InvestedCapital
    ending_invested_capital: _InvestedCapital
    average_invested_capital: float
    roic_raw: float
    roic: float
    reinvestment_raw: float
    reinvestment: float
    sustainable_growth: float
    fact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CompanyState:
    listing: Listing
    price: float
    price_asset: DataAsset
    sic: CompanyClassificationObservation | None
    metric: _MetricEvidence | None
    sustainable: _SustainableGrowth | None
    insufficiency_reason: str
    fact_map: dict[str, FundamentalFact]
    filing_assets: dict[str, DataAsset]
    failure_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _PeerSelection:
    prefix_length: int
    prefix: str
    peer_growth: float
    peer_multiple: float
    members: tuple[_CompanyState, ...]


def build_long_forecasts(
    *,
    listings: list[Listing],
    current_prices: dict[str, float],
    price_assets: dict[str, DataAsset],
    asof: AsOfData,
    data_cutoff: datetime,
    target_date: date,
    config: LongForecastConfig,
) -> dict[str, dict[str, LongForecast]]:
    sec_config = load_sec_fundamentals_config()
    company_ids = [listing.security.company_id for listing in listings]
    period_lookback = timedelta(
        days=(max(family.minimum_annual_periods for family in config.metric_families.values()) + 1)
        * 366
    )
    facts_by_company: dict[str, list[FundamentalFact]] = {}
    for fact in (
        asof.fundamental_facts_for_companies(
            company_ids=company_ids,
            concepts=[
                "operating_income",
                "pretax_income",
                "income_tax_expense",
                "net_income",
                "diluted_eps",
                "weighted_average_diluted_shares",
                "operating_cash_flow",
                "capital_expenditure",
                "cash_and_equivalents",
                "short_term_debt",
                "current_long_term_debt",
                "long_term_debt",
                "reported_long_term_debt",
                "equity",
            ],
            available_through=data_cutoff,
        )
        .filter(provider=config.fundamentals_provider)
        .filter(
            period_end__gte=target_date - period_lookback,
            period_end__lte=target_date,
        )
        .select_related("source_asset")
    ):
        facts_by_company.setdefault(str(fact.company_id), []).append(fact)
    classifications_by_company: dict[str, list[CompanyClassificationObservation]] = {}
    for observation in asof.company_classifications_for_companies(
        company_ids=company_ids,
        scheme="sec_sic",
        available_through=data_cutoff,
    ).filter(provider=config.fundamentals_provider):
        classifications_by_company.setdefault(str(observation.company_id), []).append(observation)
    filing_assets: dict[str, DataAsset] = {}
    states = {
        str(listing.pk): _company_state(
            listing=listing,
            price=current_prices[str(listing.pk)],
            price_asset=price_assets[str(listing.pk)],
            asof_decision_time=asof.decision_time,
            target_date=target_date,
            config=config,
            sec_config=sec_config,
            facts=facts_by_company.get(str(listing.security.company_id), []),
            classifications=classifications_by_company.get(
                str(listing.security.company_id),
                [],
            ),
            filing_assets=filing_assets,
        )
        for listing in listings
    }
    relevant_fact_ids = sorted(
        {fact_id for state in states.values() for fact_id in _state_fact_ids(state)}
    )
    for offset in range(0, len(relevant_fact_ids), MAX_SQL_IN_ITEMS):
        for evidence in FundamentalFactEvidence.objects.filter(
            fact_id__in=[
                UUID(value) for value in relevant_fact_ids[offset : offset + MAX_SQL_IN_ITEMS]
            ],
            role=FundamentalFactEvidence.Role.FILING,
        ).select_related("source_asset"):
            filing_assets[str(evidence.fact_id)] = evidence.source_asset
    return {
        listing_id: _forecasts_for_state(
            state=state,
            states=states,
            target_date=target_date,
            config=config,
            sec_config=sec_config,
        )
        for listing_id, state in states.items()
    }


def _company_state(
    *,
    listing: Listing,
    price: float,
    price_asset: DataAsset,
    asof_decision_time: datetime,
    target_date: date,
    config: LongForecastConfig,
    sec_config: SecFundamentalsConfig,
    facts: list[FundamentalFact],
    classifications: list[CompanyClassificationObservation],
    filing_assets: dict[str, DataAsset],
) -> _CompanyState:
    price_basis_reason = _price_basis_failure(
        listing=listing,
        price=price,
        price_asset=price_asset,
        asof_decision_time=asof_decision_time,
        config=config,
    )
    if price_basis_reason:
        return _CompanyState(
            listing=listing,
            price=price,
            price_asset=price_asset,
            sic=None,
            metric=None,
            sustainable=None,
            insufficiency_reason=price_basis_reason,
            fact_map={},
            filing_assets=filing_assets,
        )
    fact_map = {str(fact.pk): fact for fact in facts}
    series = build_sec_fundamental_series(facts, config=sec_config)
    sic = classifications[-1] if classifications else None
    if sic is None:
        return _CompanyState(
            listing=listing,
            price=price,
            price_asset=price_asset,
            sic=None,
            metric=None,
            sustainable=None,
            insufficiency_reason="No point-in-time SEC SIC classification is available",
            fact_map=fact_map,
            filing_assets=filing_assets,
        )
    normalized_sic = _normalized_sic(sic.code)
    if normalized_sic is None:
        return _CompanyState(
            listing=listing,
            price=price,
            price_asset=price_asset,
            sic=sic,
            metric=None,
            sustainable=None,
            insufficiency_reason=f"SEC SIC code {sic.code!r} is not a four-digit classification",
            fact_map=fact_map,
            filing_assets=filing_assets,
        )
    sic_number = int(normalized_sic)
    if any(start <= sic_number <= end for start, end in config.eligibility.unsupported_sic_ranges):
        return _CompanyState(
            listing=listing,
            price=price,
            price_asset=price_asset,
            sic=sic,
            metric=None,
            sustainable=None,
            insufficiency_reason=(
                f"SEC SIC {normalized_sic} is outside the supported long-v1 industries"
            ),
            fact_map=fact_map,
            filing_assets=filing_assets,
        )

    fcf = _metric_assessment(
        family="fcf_per_share",
        family_config=config.metric_families["fcf_per_share"],
        series=series,
        price=price,
        target_date=target_date,
        config=config,
    )
    eps = _metric_assessment(
        family="eps_per_share",
        family_config=config.metric_families["eps_per_share"],
        series=series,
        price=price,
        target_date=target_date,
        config=config,
    )
    if fcf.status == "eligible":
        metric = fcf.evidence
    elif fcf.status == "failed" and config.eligibility.fcf_failure_blocks_eps:
        return _CompanyState(
            listing=listing,
            price=price,
            price_asset=price_asset,
            sic=sic,
            metric=None,
            sustainable=None,
            insufficiency_reason=(
                f"FCF/share branch failed and cannot silently switch to EPS/share: {fcf.reason}"
            ),
            fact_map=fact_map,
            filing_assets=filing_assets,
            failure_fact_ids=fcf.fact_ids,
        )
    elif eps.status == "eligible":
        metric = eps.evidence
    else:
        reasons = [reason for reason in (fcf.reason, eps.reason) if reason]
        return _CompanyState(
            listing=listing,
            price=price,
            price_asset=price_asset,
            sic=sic,
            metric=None,
            sustainable=None,
            insufficiency_reason="; ".join(reasons) or "No supported per-share metric family",
            fact_map=fact_map,
            filing_assets=filing_assets,
            failure_fact_ids=_dedupe_text((*fcf.fact_ids, *eps.fact_ids)),
        )
    assert metric is not None
    sustainable, sustainable_reason = _sustainable_growth(
        series=series,
        metric=metric,
        config=config,
    )
    return _CompanyState(
        listing=listing,
        price=price,
        price_asset=price_asset,
        sic=sic,
        metric=metric,
        sustainable=sustainable,
        insufficiency_reason=sustainable_reason,
        fact_map=fact_map,
        filing_assets=filing_assets,
        failure_fact_ids=(
            () if sustainable is not None else _sustainable_candidate_fact_ids(series)
        ),
    )


def _metric_assessment(
    *,
    family: str,
    family_config: LongMetricFamilyConfig,
    series: SecFundamentalSeries,
    price: float,
    target_date: date,
    config: LongForecastConfig,
) -> _MetricAssessment:
    metric = series.ttm.get(family_config.source_metric)
    shares = series.ttm.get("weighted_average_diluted_shares")
    if metric is None or shares is None:
        status = "unavailable"
        if family == "fcf_per_share" and _has_fcf_evidence(series):
            status = "failed"
        return _MetricAssessment(
            status=status,
            reason=(
                f"{family} requires compatible TTM {family_config.source_metric} "
                "and weighted diluted shares"
            ),
            evidence=None,
            fact_ids=_value_fact_ids(
                (
                    metric,
                    shares,
                    series.ttm.get("operating_cash_flow"),
                    series.ttm.get("capital_expenditure"),
                    *series.annual.get(family_config.source_metric, ()),
                    *series.annual.get("operating_cash_flow", ()),
                    *series.annual.get("capital_expenditure", ()),
                    *series.quarters.get(family_config.source_metric, ()),
                    *series.quarters.get("operating_cash_flow", ()),
                    *series.quarters.get("capital_expenditure", ()),
                    *series.annual.get("weighted_average_diluted_shares", ()),
                )
            ),
        )
    if (
        metric.period_start != shares.period_start
        or metric.period_end != shares.period_end
        or shares.value <= 0
    ):
        return _MetricAssessment(
            status="failed",
            reason=f"{family} TTM metric and diluted-share periods are incompatible",
            evidence=None,
            fact_ids=_value_fact_ids((metric, shares)),
        )
    metric_age = (target_date - metric.period_end).days
    if metric_age < 0 or metric_age > config.eligibility.maximum_metric_age_days:
        return _MetricAssessment(
            status="failed",
            reason=(
                f"{family} latest TTM period is {metric_age} days from the forecast target; "
                f"maximum is {config.eligibility.maximum_metric_age_days}"
            ),
            evidence=None,
            fact_ids=_value_fact_ids((metric, shares)),
        )
    current_per_share = float(metric.value / shares.value)
    if current_per_share < family_config.minimum_current_value:
        return _MetricAssessment(
            status="failed",
            reason=(
                f"{family} current value {current_per_share:.6f} is below the positive "
                f"minimum {family_config.minimum_current_value:.6f}"
            ),
            evidence=None,
            fact_ids=_value_fact_ids((metric, shares)),
        )
    if not math.isfinite(price) or price <= 0:
        return _MetricAssessment(
            status="failed",
            reason=f"{family} requires a positive finite current price",
            evidence=None,
            fact_ids=_value_fact_ids((metric, shares)),
        )
    annual_points = _annual_per_share_points(
        metric_values=series.annual.get(family_config.source_metric, ()),
        share_values=series.annual.get("weighted_average_diluted_shares", ()),
    )
    contiguous = _contiguous_tail(annual_points)
    if len(contiguous) < family_config.minimum_annual_periods:
        return _MetricAssessment(
            status="failed",
            reason=(
                f"{family} has {len(contiguous)}/{family_config.minimum_annual_periods} "
                "positive contiguous annual per-share periods"
            ),
            evidence=None,
            fact_ids=_dedupe_text(
                (
                    *metric.source_fact_ids,
                    *shares.source_fact_ids,
                    *(fact_id for point in contiguous for fact_id in point.fact_ids),
                )
            ),
        )
    selected_points = contiguous[-family_config.minimum_annual_periods :]
    share_consistency, share_fact_ids, share_reason = _share_consistency(
        series=series,
        required_periods=tuple((point.period_start, point.period_end) for point in selected_points),
        current_shares=shares,
        minimum_periods=config.eligibility.minimum_share_consistency_periods,
        tolerance=config.eligibility.share_consistency_relative_tolerance,
    )
    if share_reason:
        return _MetricAssessment(
            status="failed",
            reason=share_reason,
            evidence=None,
            fact_ids=_dedupe_text(
                (
                    *metric.source_fact_ids,
                    *shares.source_fact_ids,
                    *(fact_id for point in selected_points for fact_id in point.fact_ids),
                    *share_fact_ids,
                )
            ),
        )
    growth_raw, growth_capped, growth_observations = _historical_growth(
        selected_points,
        config=config,
    )
    multiple_raw = price / current_per_share
    if multiple_raw < family_config.multiple_minimum:
        return _MetricAssessment(
            status="failed",
            reason=(
                f"{family} current multiple {multiple_raw:.3f} is below the supported "
                f"long-v1 minimum {family_config.multiple_minimum:.3f}"
            ),
            evidence=None,
            fact_ids=_dedupe_text(
                (
                    *metric.source_fact_ids,
                    *shares.source_fact_ids,
                    *(fact_id for point in selected_points for fact_id in point.fact_ids),
                    *share_fact_ids,
                )
            ),
        )
    multiple = _clamp(
        multiple_raw,
        family_config.multiple_minimum,
        family_config.multiple_maximum,
    )
    fact_ids = _dedupe_text(
        (
            *metric.source_fact_ids,
            *shares.source_fact_ids,
            *(fact_id for point in selected_points for fact_id in point.fact_ids),
            *share_fact_ids,
        )
    )
    accessions = _dedupe_text(
        (
            *metric.accessions,
            *shares.accessions,
            *(accession for point in selected_points for accession in point.accessions),
        )
    )
    return _MetricAssessment(
        status="eligible",
        reason="",
        evidence=_MetricEvidence(
            family=family,
            current_per_share=current_per_share,
            current_period_start=metric.period_start,
            current_period_end=metric.period_end,
            current_multiple_raw=multiple_raw,
            current_multiple=multiple,
            annual_points=selected_points,
            historical_growth_raw=growth_raw,
            historical_growth=growth_capped,
            growth_observations=growth_observations,
            share_consistency=share_consistency,
            fact_ids=fact_ids,
            accessions=accessions,
        ),
    )


def _has_fcf_evidence(series: SecFundamentalSeries) -> bool:
    concepts = ("free_cash_flow", "operating_cash_flow", "capital_expenditure")
    return any(
        series.ttm.get(concept) is not None
        or bool(series.annual.get(concept))
        or bool(series.quarters.get(concept))
        for concept in concepts
    )


def _annual_per_share_points(
    *,
    metric_values: tuple[FundamentalValue, ...],
    share_values: tuple[FundamentalValue, ...],
) -> tuple[_PerSharePoint, ...]:
    shares = {
        (value.period_start, value.period_end): value for value in share_values if value.value > 0
    }
    points: list[_PerSharePoint] = []
    for metric in metric_values:
        share = shares.get((metric.period_start, metric.period_end))
        if share is None:
            continue
        value = float(metric.value / share.value)
        if not math.isfinite(value) or value <= 0:
            continue
        points.append(
            _PerSharePoint(
                value=value,
                period_start=metric.period_start,
                period_end=metric.period_end,
                available_at=max(metric.available_at, share.available_at),
                fact_ids=_dedupe_text((*metric.source_fact_ids, *share.source_fact_ids)),
                accessions=_dedupe_text((*metric.accessions, *share.accessions)),
            )
        )
    return tuple(
        sorted(
            points,
            key=lambda point: (point.period_end, point.available_at),
        )
    )


def _value_fact_ids(
    values: tuple[FundamentalValue | None, ...],
) -> tuple[str, ...]:
    return _dedupe_text(
        [fact_id for value in values if value is not None for fact_id in value.source_fact_ids]
    )


def _contiguous_tail(points: tuple[_PerSharePoint, ...]) -> tuple[_PerSharePoint, ...]:
    if not points:
        return ()
    tail = [points[-1]]
    for point in reversed(points[:-1]):
        if (tail[0].period_start - point.period_end).days != 1:
            break
        tail.insert(0, point)
    return tuple(tail)


def _share_consistency(
    *,
    series: SecFundamentalSeries,
    required_periods: tuple[tuple[date, date], ...],
    current_shares: FundamentalValue,
    minimum_periods: int,
    tolerance: float,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], str]:
    net_income = {
        (value.period_start, value.period_end): value
        for value in series.annual.get("net_income", ())
    }
    shares = {
        (value.period_start, value.period_end): value
        for value in series.annual.get("weighted_average_diluted_shares", ())
    }
    reported_eps = {
        (value.period_start, value.period_end): value
        for value in series.annual.get("diluted_eps", ())
    }
    comparable = set(net_income) & set(shares) & set(reported_eps)
    missing_periods = [period for period in required_periods if period not in comparable]
    if len(required_periods) < minimum_periods or missing_periods:
        complete_count = len(required_periods) - len(missing_periods)
        return (
            (),
            (),
            f"Share basis has {complete_count}/{len(required_periods)} comparable diluted-EPS "
            "periods for the selected metric history",
        )
    checks: list[dict[str, Any]] = []
    fact_ids: list[str] = []
    for period in required_periods:
        income = net_income[period]
        share = shares[period]
        eps = reported_eps[period]
        if share.value <= 0:
            return (), (), "Share basis contains non-positive diluted shares"
        derived = float(income.value / share.value)
        reported = float(eps.value)
        denominator = max(abs(derived), abs(reported), 1e-12)
        relative_difference = abs(derived - reported) / denominator
        checks.append(
            {
                "check": "reported_diluted_eps",
                "period_start": period[0].isoformat(),
                "period_end": period[1].isoformat(),
                "derived_eps": derived,
                "reported_eps": reported,
                "relative_difference": relative_difference,
                "tolerance": tolerance,
            }
        )
        fact_ids.extend(
            (
                *income.source_fact_ids,
                *share.source_fact_ids,
                *eps.source_fact_ids,
            )
        )
        if relative_difference > tolerance:
            return (
                tuple(checks),
                _dedupe_text(fact_ids),
                f"Share basis differs from reported diluted EPS by "
                f"{relative_difference:.1%}, above {tolerance:.1%}",
            )
    latest_period = required_periods[-1]
    latest_annual_shares = shares[latest_period]
    if not (current_shares.period_start <= latest_period[1] <= current_shares.period_end):
        return (
            tuple(checks),
            _dedupe_text(fact_ids),
            "Latest selected annual diluted-share basis does not overlap the current TTM window",
        )
    annual_value = float(latest_annual_shares.value)
    ttm_value = float(current_shares.value)
    relative_difference = abs(ttm_value / annual_value - 1.0)
    checks.append(
        {
            "check": "ttm_to_latest_annual_diluted_shares",
            "annual_period_end": latest_period[1].isoformat(),
            "annual_shares": annual_value,
            "ttm_period_start": current_shares.period_start.isoformat(),
            "ttm_period_end": current_shares.period_end.isoformat(),
            "ttm_shares": ttm_value,
            "relative_difference": relative_difference,
            "tolerance": tolerance,
        }
    )
    fact_ids.extend((*latest_annual_shares.source_fact_ids, *current_shares.source_fact_ids))
    if relative_difference > tolerance:
        return (
            tuple(checks),
            _dedupe_text(fact_ids),
            "TTM diluted shares differ from the latest selected annual share basis by "
            f"{relative_difference:.1%}, above {tolerance:.1%}",
        )
    return tuple(checks), _dedupe_text(fact_ids), ""


def _historical_growth(
    points: tuple[_PerSharePoint, ...],
    *,
    config: LongForecastConfig,
) -> tuple[float, float, tuple[dict[str, Any], ...]]:
    observations: list[dict[str, Any]] = []
    capped_growth: list[float] = []
    for previous, current in zip(points, points[1:], strict=False):
        raw = current.value / previous.value - 1.0
        capped = _clamp(
            raw,
            config.growth.historical_minimum,
            config.growth.historical_maximum,
        )
        observations.append(
            {
                "period_end": current.period_end.isoformat(),
                "raw_growth": raw,
                "capped_growth": capped,
            }
        )
        capped_growth.append(capped)
    weights = [
        config.growth.historical_recency_decay ** (len(capped_growth) - index - 1)
        for index in range(len(capped_growth))
    ]
    raw_weighted = sum(
        observation["raw_growth"] * weight
        for observation, weight in zip(observations, weights, strict=True)
    ) / sum(weights)
    capped_weighted = sum(
        value * weight for value, weight in zip(capped_growth, weights, strict=True)
    ) / sum(weights)
    return (
        raw_weighted,
        _clamp(
            capped_weighted,
            config.growth.historical_minimum,
            config.growth.historical_maximum,
        ),
        tuple(observations),
    )


def _sustainable_growth(
    *,
    series: SecFundamentalSeries,
    metric: _MetricEvidence,
    config: LongForecastConfig,
) -> tuple[_SustainableGrowth | None, str]:
    required = {
        concept: series.ttm.get(concept)
        for concept in (
            "operating_income",
            "pretax_income",
            "income_tax_expense",
        )
    }
    missing = [concept for concept, value in required.items() if value is None]
    if missing:
        return None, "Sustainable growth requires TTM " + ", ".join(missing)
    operating = required["operating_income"]
    pretax = required["pretax_income"]
    tax = required["income_tax_expense"]
    assert operating is not None
    assert pretax is not None
    assert tax is not None
    expected_period = (metric.current_period_start, metric.current_period_end)
    if any(
        (value.period_start, value.period_end) != expected_period
        for value in (operating, pretax, tax)
    ):
        return None, "Sustainable-growth TTM inputs do not share the metric period"
    if pretax.value <= 0:
        return None, "Sustainable growth requires positive TTM pretax income"
    tax_rate_raw = float(tax.value / pretax.value)
    tax_rate = _clamp(
        tax_rate_raw,
        config.growth.tax_rate_minimum,
        config.growth.tax_rate_maximum,
    )
    nopat = float(operating.value) * (1.0 - tax_rate)
    if not math.isfinite(nopat) or nopat <= 0:
        return None, "Sustainable growth requires positive TTM NOPAT"
    beginning, beginning_reason = _invested_capital_near(
        series=series,
        target_date=metric.current_period_start - timedelta(days=1),
        tolerance_days=config.eligibility.balance_sheet_date_tolerance_days,
    )
    if beginning is None:
        return None, f"Beginning invested capital unavailable: {beginning_reason}"
    ending, ending_reason = _invested_capital_near(
        series=series,
        target_date=metric.current_period_end,
        tolerance_days=config.eligibility.balance_sheet_date_tolerance_days,
    )
    if ending is None:
        return None, f"Ending invested capital unavailable: {ending_reason}"
    if (
        beginning.debt_method != ending.debt_method
        or beginning.debt_components != ending.debt_components
        or beginning.source_basis != ending.source_basis
    ):
        return None, "Beginning/end invested-capital evidence uses incompatible source definitions"
    average = (beginning.value + ending.value) / 2.0
    if average <= 0:
        return None, "Average invested capital must be positive"
    roic_raw = nopat / average
    roic = _clamp(
        roic_raw,
        config.growth.roic_minimum,
        config.growth.roic_maximum,
    )
    reinvestment_raw = (ending.value - beginning.value) / nopat
    reinvestment = _clamp(
        reinvestment_raw,
        config.growth.reinvestment_minimum,
        config.growth.reinvestment_maximum,
    )
    sustainable = _clamp(
        roic * reinvestment,
        config.growth.sustainable_minimum,
        config.growth.sustainable_maximum,
    )
    return (
        _SustainableGrowth(
            tax_rate_raw=tax_rate_raw,
            tax_rate=tax_rate,
            nopat=nopat,
            beginning_invested_capital=beginning,
            ending_invested_capital=ending,
            average_invested_capital=average,
            roic_raw=roic_raw,
            roic=roic,
            reinvestment_raw=reinvestment_raw,
            reinvestment=reinvestment,
            sustainable_growth=sustainable,
            fact_ids=_dedupe_text(
                (
                    *operating.source_fact_ids,
                    *pretax.source_fact_ids,
                    *tax.source_fact_ids,
                    *beginning.fact_ids,
                    *ending.fact_ids,
                )
            ),
        ),
        "",
    )


def _sustainable_candidate_fact_ids(
    series: SecFundamentalSeries,
) -> tuple[str, ...]:
    value_ids = _value_fact_ids(
        tuple(
            series.ttm.get(concept)
            for concept in (
                "operating_income",
                "pretax_income",
                "income_tax_expense",
            )
        )
    )
    instant_ids = [
        str(fact.pk)
        for concept in (
            "equity",
            "cash_and_equivalents",
            "reported_long_term_debt",
            "long_term_debt",
            "current_long_term_debt",
            "short_term_debt",
        )
        for fact in series.instants.get(concept, ())
    ]
    return _dedupe_text((*value_ids, *instant_ids))


def _invested_capital_near(
    *,
    series: SecFundamentalSeries,
    target_date: date,
    tolerance_days: int,
) -> tuple[_InvestedCapital | None, str]:
    by_concept = {
        concept: {fact.period_end: fact for fact in facts}
        for concept, facts in series.instants.items()
    }
    dates = sorted(
        {
            fact.period_end
            for concept in (
                "equity",
                "cash_and_equivalents",
                "reported_long_term_debt",
                "long_term_debt",
                "current_long_term_debt",
                "short_term_debt",
            )
            for fact in series.instants.get(concept, ())
            if abs((fact.period_end - target_date).days) <= tolerance_days
        },
        key=lambda value: (abs((value - target_date).days), value),
    )
    for period_end in dates:
        equity = by_concept.get("equity", {}).get(period_end)
        cash = by_concept.get("cash_and_equivalents", {}).get(period_end)
        if equity is None or cash is None or equity.value <= 0 or cash.value < 0:
            continue
        debt_facts: list[FundamentalFact] = []
        reported = by_concept.get("reported_long_term_debt", {}).get(period_end)
        if reported is not None:
            debt_facts.append(reported)
            short_term = by_concept.get("short_term_debt", {}).get(period_end)
            if short_term is not None:
                debt_facts.append(short_term)
            debt_method = "reported_long_term_plus_short_term_borrowings"
        else:
            for concept in (
                "long_term_debt",
                "current_long_term_debt",
                "short_term_debt",
            ):
                fact = by_concept.get(concept, {}).get(period_end)
                if fact is not None:
                    debt_facts.append(fact)
            debt_method = "sum_non_overlapping_debt_components"
        if not debt_facts or any(fact.value < 0 for fact in debt_facts):
            continue
        debt = float(sum((fact.value for fact in debt_facts), Decimal("0")))
        equity_value = float(equity.value)
        cash_value = float(cash.value)
        invested = debt + equity_value - cash_value
        if not math.isfinite(invested) or invested <= 0:
            continue
        return (
            _InvestedCapital(
                value=invested,
                period_end=period_end,
                debt=debt,
                equity=equity_value,
                cash=cash_value,
                debt_method=debt_method,
                debt_components=tuple(fact.concept for fact in debt_facts),
                source_basis=(
                    ("equity", equity.source_concept),
                    ("cash_and_equivalents", cash.source_concept),
                    *((fact.concept, fact.source_concept) for fact in debt_facts),
                ),
                fact_ids=_dedupe_text(
                    (
                        str(equity.pk),
                        str(cash.pk),
                        *(str(fact.pk) for fact in debt_facts),
                    )
                ),
            ),
            "",
        )
    return None, f"no compatible balance sheet within {tolerance_days} days of {target_date}"


def _forecasts_for_state(
    *,
    state: _CompanyState,
    states: dict[str, _CompanyState],
    target_date: date,
    config: LongForecastConfig,
    sec_config: SecFundamentalsConfig,
) -> dict[str, LongForecast]:
    split_basis = (
        _split_basis_payload(
            metric=state.metric,
            target_date=target_date,
            config=config,
        )
        if state.metric is not None
        else None
    )
    if state.metric is None or state.sustainable is None or state.sic is None:
        return _missing_forecasts(
            reason=state.insufficiency_reason,
            config=config,
            source_assets=_state_source_assets(state),
            sec_config=sec_config,
            input_facts=_state_input_fact_payloads(state),
            target_classification=(
                _classification_payload(state.sic) if state.sic is not None else None
            ),
            target_price_asset_id=str(state.price_asset.pk),
            split_basis=split_basis,
        )
    peers = _select_peers(state=state, states=states, config=config)
    if peers is None:
        return _missing_forecasts(
            reason=(f"No same-family SEC SIC peer set met floors {config.peer.minimum_peers}"),
            config=config,
            source_assets=_state_source_assets(state),
            metric_family=state.metric.family,
            sec_config=sec_config,
            input_facts=_state_input_fact_payloads(state),
            target_classification=_classification_payload(state.sic),
            target_price_asset_id=str(state.price_asset.pk),
            split_basis=split_basis,
        )
    return {
        horizon: _forecast_horizon(
            state=state,
            peers=peers,
            horizon=horizon,
            horizon_config=config.horizons[horizon],
            target_date=target_date,
            config=config,
            sec_config=sec_config,
        )
        for horizon in LONG_FORECAST_HORIZONS
    }


def _select_peers(
    *,
    state: _CompanyState,
    states: dict[str, _CompanyState],
    config: LongForecastConfig,
) -> _PeerSelection | None:
    assert state.metric is not None
    assert state.sic is not None
    sic = _normalized_sic(state.sic.code)
    if sic is None:
        return None
    for level in config.peer.sic_prefix_levels:
        prefix = sic[:level]
        eligible = sorted(
            (
                candidate
                for candidate in states.values()
                if candidate.listing.security.company_id != state.listing.security.company_id
                and candidate.metric is not None
                and candidate.metric.family == state.metric.family
                and candidate.sic is not None
                and (candidate_sic := _normalized_sic(candidate.sic.code)) is not None
                and candidate_sic.startswith(prefix)
            ),
            key=lambda candidate: (
                candidate.listing.ticker,
                str(candidate.listing.pk),
            ),
        )
        candidates_by_company: dict[str, _CompanyState] = {}
        for candidate in eligible:
            candidates_by_company.setdefault(
                str(candidate.listing.security.company_id),
                candidate,
            )
        candidates = tuple(candidates_by_company.values())
        if len(candidates) < config.peer.minimum_peers[level]:
            continue
        return _PeerSelection(
            prefix_length=level,
            prefix=prefix,
            peer_growth=_clamp(
                statistics.median(
                    candidate.metric.historical_growth
                    for candidate in candidates
                    if candidate.metric is not None
                ),
                config.growth.peer_minimum,
                config.growth.peer_maximum,
            ),
            peer_multiple=statistics.median(
                candidate.metric.current_multiple
                for candidate in candidates
                if candidate.metric is not None
            ),
            members=candidates,
        )
    return None


def _forecast_horizon(
    *,
    state: _CompanyState,
    peers: _PeerSelection,
    horizon: str,
    horizon_config: LongHorizonConfig,
    target_date: date,
    config: LongForecastConfig,
    sec_config: SecFundamentalsConfig,
) -> LongForecast:
    assert state.metric is not None
    assert state.sustainable is not None
    assert state.sic is not None
    family_config = config.metric_families[state.metric.family]
    scenario_outputs = {
        name: _scenario_path(
            metric=state.metric,
            sustainable=state.sustainable,
            peer_growth=peers.peer_growth,
            peer_multiple=peers.peer_multiple,
            horizon=horizon_config,
            scenario=config.scenarios[name],
            family_config=family_config,
            config=config,
        )
        for name in LONG_SCENARIOS
    }
    returns = [scenario_outputs[name]["cumulative_return"] for name in LONG_SCENARIOS]
    if returns != sorted(returns):
        return _missing_forecast(
            horizon=horizon,
            reason="Frozen bear/base/bull assumptions did not produce ordered scenarios",
            config=config,
            source_assets=_forecast_source_assets(state=state, peers=peers),
            metric_family=state.metric.family,
            sec_config=sec_config,
            input_facts=_state_input_fact_payloads(state),
            target_classification=_classification_payload(state.sic),
            peer_set=[_peer_payload(member) for member in peers.members],
            target_price_asset_id=str(state.price_asset.pk),
            split_basis=_split_basis_payload(
                metric=state.metric,
                target_date=target_date,
                config=config,
            ),
        )
    share_consistency_periods = sum(
        check.get("check") == "reported_diluted_eps" for check in state.metric.share_consistency
    )
    confidence = min(
        80.0,
        45.0
        + len(state.metric.annual_points) * 4.0
        + share_consistency_periods * 3.0
        + min(len(peers.members), 10),
    )
    probability_reason = (
        "Positive-return probability is unavailable for deterministic long-v1 "
        "until qualifying prospective outcomes exist"
    )
    scenario = Scenario(
        bear=returns[0],
        base=returns[1],
        bull=returns[2],
        probability_positive=None,
        confidence=confidence,
        confidence_status="deterministic_point_in_time",
        insufficiency_reason=probability_reason,
        method=METHOD_NAME,
    )
    annualized = {name: scenario_outputs[name]["annualized_return"] for name in LONG_SCENARIOS}
    target_fact_ids = _dedupe_text((*state.metric.fact_ids, *state.sustainable.fact_ids))
    sic = state.sic
    calculation = {
        "schema_version": 1,
        "method": METHOD_NAME,
        "method_version": config.version,
        "config_hash": long_forecast_config_hash(config),
        "fundamentals_config_version": sec_config.config_version,
        "fundamentals_config_hash": sec_config.config_hash,
        "forecast_horizon": horizon,
        "years": horizon_config.years,
        "metric_family": state.metric.family,
        "target_price_asset_id": str(state.price_asset.pk),
        "return_basis": config.return_basis,
        "dividends_included": config.dividends_included,
        "probability_status": "withheld_unavailable",
        "support": {
            "annual_periods": len(state.metric.annual_points),
            "growth_observations": len(state.metric.growth_observations),
            "share_consistency_periods": share_consistency_periods,
            "peer_count": len(peers.members),
            "sic_fallback_level": peers.prefix_length,
            "sic_prefix": peers.prefix,
        },
        "formula_inputs": {
            "current_price": state.price,
            "current_per_share_metric": state.metric.current_per_share,
            "current_multiple_raw": state.metric.current_multiple_raw,
            "current_multiple_capped": state.metric.current_multiple,
            "historical_growth_raw": state.metric.historical_growth_raw,
            "historical_growth_capped": state.metric.historical_growth,
            "historical_growth_observations": state.metric.growth_observations,
            "share_consistency": state.metric.share_consistency,
            "tax_rate_raw": state.sustainable.tax_rate_raw,
            "tax_rate_capped": state.sustainable.tax_rate,
            "nopat": state.sustainable.nopat,
            "beginning_invested_capital": _invested_capital_payload(
                state.sustainable.beginning_invested_capital
            ),
            "ending_invested_capital": _invested_capital_payload(
                state.sustainable.ending_invested_capital
            ),
            "average_invested_capital": state.sustainable.average_invested_capital,
            "roic_raw": state.sustainable.roic_raw,
            "roic_capped": state.sustainable.roic,
            "reinvestment_raw": state.sustainable.reinvestment_raw,
            "reinvestment_capped": state.sustainable.reinvestment,
            "sustainable_growth": state.sustainable.sustainable_growth,
            "peer_growth": peers.peer_growth,
            "peer_multiple": peers.peer_multiple,
            "growth_weights": {
                "historical": config.growth.historical_weight,
                "sustainable": config.growth.sustainable_weight,
                "peer": config.growth.peer_weight,
            },
            "terminal_growth": config.growth.terminal_growth,
            "fade": list(horizon_config.fade),
            "multiple_reversion": horizon_config.multiple_reversion,
        },
        "split_basis": _split_basis_payload(
            metric=state.metric,
            target_date=target_date,
            config=config,
        ),
        "scenario_paths": scenario_outputs,
        "annualized_returns": annualized,
        "input_facts": [
            _fact_payload(state.fact_map[fact_id], state.filing_assets)
            for fact_id in target_fact_ids
            if fact_id in state.fact_map
        ],
        "target_classification": _classification_payload(sic),
        "peer_set": [_peer_payload(member) for member in peers.members],
        "contribution_detail": {
            name: output["growth_contributions"] for name, output in scenario_outputs.items()
        },
    }
    return LongForecast(
        scenario=scenario,
        calculation=calculation,
        source_assets=_forecast_source_assets(state=state, peers=peers),
    )


def _scenario_path(
    *,
    metric: _MetricEvidence,
    sustainable: _SustainableGrowth,
    peer_growth: float,
    peer_multiple: float,
    horizon: LongHorizonConfig,
    scenario: LongScenarioConfig,
    family_config: LongMetricFamilyConfig,
    config: LongForecastConfig,
) -> dict[str, Any]:
    historical = _clamp(
        metric.historical_growth + scenario.growth_delta,
        config.growth.historical_minimum,
        config.growth.historical_maximum,
    )
    reinvestment = _clamp(
        sustainable.reinvestment * scenario.reinvestment_multiplier,
        config.growth.reinvestment_minimum,
        config.growth.reinvestment_maximum,
    )
    sustainable_growth = _clamp(
        sustainable.roic * reinvestment,
        config.growth.sustainable_minimum,
        config.growth.sustainable_maximum,
    )
    adjusted_peer_growth = _clamp(
        peer_growth + scenario.growth_delta,
        config.growth.peer_minimum,
        config.growth.peer_maximum,
    )
    growth_contributions = {
        "historical": config.growth.historical_weight * historical,
        "sustainable": config.growth.sustainable_weight * sustainable_growth,
        "peer": config.growth.peer_weight * adjusted_peer_growth,
    }
    initial_growth = _clamp(
        sum(growth_contributions.values()),
        config.growth.initial_growth_minimum,
        config.growth.initial_growth_maximum,
    )
    annual_growth = [
        fade * initial_growth + (1.0 - fade) * config.growth.terminal_growth
        for fade in horizon.fade
    ]
    fundamental_growth = math.prod(1.0 + growth for growth in annual_growth)
    adjusted_peer_multiple = _clamp(
        peer_multiple * scenario.peer_multiple_multiplier,
        family_config.multiple_minimum,
        family_config.multiple_maximum,
    )
    terminal_multiple = math.exp(
        (1.0 - horizon.multiple_reversion) * math.log(metric.current_multiple)
        + horizon.multiple_reversion * math.log(adjusted_peer_multiple)
    )
    terminal_multiple = _clamp(
        terminal_multiple,
        family_config.multiple_minimum,
        family_config.multiple_maximum,
    )
    cumulative_return = max(
        -1.0,
        fundamental_growth * terminal_multiple / metric.current_multiple_raw - 1.0,
    )
    annualized_return = (1.0 + cumulative_return) ** (1.0 / horizon.years) - 1.0
    return {
        "historical_growth": historical,
        "reinvestment_rate": reinvestment,
        "sustainable_growth": sustainable_growth,
        "peer_growth": adjusted_peer_growth,
        "growth_contributions": growth_contributions,
        "initial_growth": initial_growth,
        "annual_growth": annual_growth,
        "fundamental_growth_factor": fundamental_growth,
        "current_multiple_return_denominator": metric.current_multiple_raw,
        "current_multiple_reversion_anchor": metric.current_multiple,
        "peer_multiple_adjusted": adjusted_peer_multiple,
        "terminal_multiple": terminal_multiple,
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
    }


def _missing_forecasts(
    *,
    reason: str,
    config: LongForecastConfig,
    source_assets: tuple[DataAsset, ...],
    sec_config: SecFundamentalsConfig,
    metric_family: str | None = None,
    input_facts: list[dict[str, Any]] | None = None,
    target_classification: dict[str, Any] | None = None,
    peer_set: list[dict[str, Any]] | None = None,
    target_price_asset_id: str | None = None,
    split_basis: dict[str, Any] | None = None,
) -> dict[str, LongForecast]:
    return {
        horizon: _missing_forecast(
            horizon=horizon,
            reason=reason,
            config=config,
            source_assets=source_assets,
            metric_family=metric_family,
            sec_config=sec_config,
            input_facts=input_facts,
            target_classification=target_classification,
            peer_set=peer_set,
            target_price_asset_id=target_price_asset_id,
            split_basis=split_basis,
        )
        for horizon in LONG_FORECAST_HORIZONS
    }


def _missing_forecast(
    *,
    horizon: str,
    reason: str,
    config: LongForecastConfig,
    source_assets: tuple[DataAsset, ...],
    metric_family: str | None,
    sec_config: SecFundamentalsConfig,
    input_facts: list[dict[str, Any]] | None = None,
    target_classification: dict[str, Any] | None = None,
    peer_set: list[dict[str, Any]] | None = None,
    target_price_asset_id: str | None = None,
    split_basis: dict[str, Any] | None = None,
) -> LongForecast:
    return LongForecast(
        scenario=Scenario(
            bear=None,
            base=None,
            bull=None,
            probability_positive=None,
            confidence=0.0,
            confidence_status="insufficient_evidence",
            insufficiency_reason=reason,
            method=METHOD_NAME,
        ),
        calculation={
            "schema_version": 1,
            "method": METHOD_NAME,
            "method_version": config.version,
            "config_hash": long_forecast_config_hash(config),
            "fundamentals_config_version": sec_config.config_version,
            "fundamentals_config_hash": sec_config.config_hash,
            "forecast_horizon": horizon,
            "years": config.horizons[horizon].years,
            "metric_family": metric_family,
            "target_price_asset_id": target_price_asset_id,
            "return_basis": config.return_basis,
            "dividends_included": config.dividends_included,
            "probability_status": "withheld_unavailable",
            "support": {},
            "formula_inputs": {},
            "split_basis": split_basis or {},
            "scenario_paths": {},
            "annualized_returns": {},
            "input_facts": input_facts or [],
            "target_classification": target_classification,
            "peer_set": peer_set or [],
            "insufficiency_reason": reason,
        },
        source_assets=source_assets,
    )


def _state_source_assets(state: _CompanyState) -> tuple[DataAsset, ...]:
    assets = [state.price_asset]
    if state.sic is not None:
        assets.append(state.sic.source_asset)
    assets.extend(
        _fact_source_assets(
            state.fact_map,
            state.filing_assets,
            _state_fact_ids(state),
        )
    )
    return _dedupe_assets(assets)


def _state_fact_ids(state: _CompanyState) -> tuple[str, ...]:
    fact_ids: tuple[str, ...] = ()
    if state.metric is not None:
        fact_ids = (*fact_ids, *state.metric.fact_ids)
    if state.sustainable is not None:
        fact_ids = (*fact_ids, *state.sustainable.fact_ids)
    fact_ids = (*fact_ids, *state.failure_fact_ids)
    return _dedupe_text(fact_ids)


def _state_input_fact_payloads(state: _CompanyState) -> list[dict[str, Any]]:
    return [
        _fact_payload(state.fact_map[fact_id], state.filing_assets)
        for fact_id in _state_fact_ids(state)
        if fact_id in state.fact_map
    ]


def _forecast_source_assets(
    *,
    state: _CompanyState,
    peers: _PeerSelection,
) -> tuple[DataAsset, ...]:
    assets = list(_state_source_assets(state))
    for peer in peers.members:
        assets.extend(_peer_source_assets(peer))
    return _dedupe_assets(assets)


def _peer_source_assets(state: _CompanyState) -> tuple[DataAsset, ...]:
    assert state.metric is not None
    assert state.sic is not None
    return _dedupe_assets(
        [
            state.price_asset,
            state.sic.source_asset,
            *_fact_source_assets(
                state.fact_map,
                state.filing_assets,
                state.metric.fact_ids,
            ),
        ]
    )


def _fact_source_assets(
    fact_map: dict[str, FundamentalFact],
    filing_assets: dict[str, DataAsset],
    fact_ids: tuple[str, ...],
) -> list[DataAsset]:
    assets: list[DataAsset] = []
    for fact_id in _dedupe_text(fact_ids):
        fact = fact_map.get(fact_id)
        if fact is None:
            continue
        assets.append(fact.source_asset)
        filing_asset = filing_assets.get(fact_id)
        if filing_asset is None:
            raise ValueError(f"SEC fact {fact.pk} has no visible filing evidence link")
        assets.append(filing_asset)
    return assets


def _peer_payload(state: _CompanyState) -> dict[str, Any]:
    assert state.metric is not None
    assert state.sic is not None
    return {
        "listing_id": str(state.listing.pk),
        "ticker": state.listing.ticker,
        "sic": _normalized_sic(state.sic.code),
        "classification_id": str(state.sic.pk),
        "classification": _classification_payload(state.sic),
        "metric_family": state.metric.family,
        "current_price": state.price,
        "historical_growth": state.metric.historical_growth,
        "current_multiple_raw": state.metric.current_multiple_raw,
        "current_multiple_capped": state.metric.current_multiple,
        "price_asset_id": str(state.price_asset.pk),
        "fact_references": [
            _fact_reference(state.fact_map[fact_id], state.filing_assets)
            for fact_id in state.metric.fact_ids
            if fact_id in state.fact_map
        ],
    }


def _fact_reference(
    fact: FundamentalFact,
    filing_assets: dict[str, DataAsset],
) -> dict[str, Any]:
    filing_asset = filing_assets.get(str(fact.pk))
    if filing_asset is None:
        raise ValueError(f"SEC fact {fact.pk} has no filing evidence link")
    return {
        "id": str(fact.pk),
        "concept": fact.concept,
        "accession": fact.accession,
        "available_at": fact.available_at.isoformat(),
        "source_revision": fact.source_revision,
        "source_asset_id": str(fact.source_asset_id),
        "filing_evidence_asset_id": str(filing_asset.pk),
    }


def _fact_payload(
    fact: FundamentalFact,
    filing_assets: dict[str, DataAsset],
) -> dict[str, Any]:
    filing_asset = filing_assets.get(str(fact.pk))
    if filing_asset is None:
        raise ValueError(f"SEC fact {fact.pk} has no filing evidence link")
    return {
        "id": str(fact.pk),
        "provider": fact.provider,
        "concept": fact.concept,
        "taxonomy": fact.taxonomy,
        "source_concept": fact.source_concept,
        "value": str(fact.value),
        "unit": fact.unit,
        "currency": fact.currency,
        "period_type": fact.period_type,
        "period_identity": fact.period_identity,
        "period_start": fact.period_start.isoformat() if fact.period_start else None,
        "period_end": fact.period_end.isoformat(),
        "fiscal_year": fact.fiscal_year,
        "fiscal_period": fact.fiscal_period,
        "frame": fact.frame,
        "accession": fact.accession,
        "filing_form": fact.filing_form,
        "filing_date": fact.filing_date.isoformat() if fact.filing_date else None,
        "filed_at": fact.filed_at.isoformat() if fact.filed_at else None,
        "acceptance_at": fact.acceptance_at.isoformat() if fact.acceptance_at else None,
        "available_at": fact.available_at.isoformat(),
        "availability_basis": fact.availability_basis,
        "is_amendment": fact.is_amendment,
        "source_revision": fact.source_revision,
        "observation_hash": fact.observation_hash,
        "quality_flags": fact.quality_flags,
        "source_asset_id": str(fact.source_asset_id),
        "filing_evidence_asset_id": str(filing_asset.pk),
    }


def _classification_payload(
    classification: CompanyClassificationObservation,
) -> dict[str, Any]:
    return {
        "id": str(classification.pk),
        "provider": classification.provider,
        "scheme": classification.scheme,
        "code": _normalized_sic(classification.code),
        "description": classification.description,
        "observed_at": classification.observed_at.isoformat(),
        "available_at": classification.available_at.isoformat(),
        "ingested_at": classification.ingested_at.isoformat(),
        "accession": classification.accession,
        "quality_flags": classification.quality_flags,
        "source_asset_id": str(classification.source_asset_id),
    }


def _invested_capital_payload(value: _InvestedCapital) -> dict[str, Any]:
    return {
        "value": value.value,
        "period_end": value.period_end.isoformat(),
        "debt": value.debt,
        "equity": value.equity,
        "cash": value.cash,
        "debt_method": value.debt_method,
        "debt_components": list(value.debt_components),
        "source_basis": [
            {
                "concept": concept,
                "source_concept": source_concept,
            }
            for concept, source_concept in value.source_basis
        ],
        "fact_ids": list(value.fact_ids),
    }


def _split_basis_payload(
    *,
    metric: _MetricEvidence,
    target_date: date,
    config: LongForecastConfig,
) -> dict[str, Any]:
    return {
        "basis": "as_filed_diluted_shares_vs_split_adjusted_price",
        "verified_through": metric.current_period_end.isoformat(),
        "post_period_exposure_days": (target_date - metric.current_period_end).days,
        "maximum_exposure_days": config.eligibility.maximum_metric_age_days,
        "continuity_tolerance": (config.eligibility.share_consistency_relative_tolerance),
        "continuity_checks": list(metric.share_consistency),
        "residual_risk": "unverified_post_period_split",
    }


def _normalized_sic(value: str) -> str | None:
    normalized = value.strip()
    if not normalized.isdigit() or len(normalized) > 4:
        return None
    return normalized.zfill(4)


def _price_basis_failure(
    *,
    listing: Listing,
    price: float,
    price_asset: DataAsset,
    asof_decision_time: datetime,
    config: LongForecastConfig,
) -> str:
    if not math.isfinite(price) or price <= 0:
        return "Long forecast requires a positive finite current price"
    if price_asset.provider != config.price_provider:
        return (
            f"Price asset provider {price_asset.provider!r} does not match "
            f"{config.price_provider!r}"
        )
    if price_asset.kind != "price_history":
        return "Long forecast price evidence must be a price_history asset"
    expected_subjects = {
        listing.ticker,
        listing.provider_symbol or listing.ticker,
    }
    if price_asset.subject not in expected_subjects:
        return (
            f"Price asset subject {price_asset.subject!r} does not match listing {listing.ticker!r}"
        )
    if (
        price_asset.available_at > asof_decision_time
        or price_asset.retrieved_at > asof_decision_time
    ):
        return "Price asset was not visible at the forecast decision time"
    return_definition = price_asset.metadata.get("return_definition")
    dividends_included = price_asset.metadata.get("dividends_included")
    if return_definition != config.return_basis:
        return f"Price asset does not prove required return basis {config.return_basis!r}"
    if dividends_included is not config.dividends_included:
        return "Price asset dividend basis is missing or incompatible"
    return ""


def _dedupe_text(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _dedupe_assets(assets: list[DataAsset]) -> tuple[DataAsset, ...]:
    return tuple({str(asset.pk): asset for asset in assets}.values())


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))

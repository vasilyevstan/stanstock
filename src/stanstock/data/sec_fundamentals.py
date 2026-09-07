from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal

from stanstock.data.models import FundamentalFact
from stanstock.data.sec_config import SecFundamentalsConfig

MIN_QUARTER_DAYS = 70
MAX_QUARTER_DAYS = 110
MIN_ANNUAL_DAYS = 350
MAX_ANNUAL_DAYS = 380

ADDITIVE_FLOW_CONCEPTS = frozenset(
    {
        "revenue",
        "operating_income",
        "pretax_income",
        "income_tax_expense",
        "net_income",
        "operating_cash_flow",
        "capital_expenditure",
        "depreciation_amortization",
        "interest_expense",
        "dividends_paid",
        "share_repurchases",
    }
)
WEIGHTED_AVERAGE_CONCEPTS = frozenset({"weighted_average_diluted_shares"})


@dataclass(frozen=True, slots=True)
class FundamentalValue:
    concept: str
    value: Decimal
    unit: str
    period_start: date
    period_end: date
    available_at: datetime
    source_fact_ids: tuple[str, ...]
    accessions: tuple[str, ...]
    source_concepts: tuple[str, ...]
    derivation: str

    @property
    def duration_days(self) -> int:
        return (self.period_end - self.period_start).days + 1


@dataclass(frozen=True, slots=True)
class SecFundamentalSeries:
    selected_facts: tuple[FundamentalFact, ...]
    annual: dict[str, tuple[FundamentalValue, ...]]
    quarters: dict[str, tuple[FundamentalValue, ...]]
    ttm: dict[str, FundamentalValue]
    instants: dict[str, tuple[FundamentalFact, ...]]
    latest_instants: dict[str, FundamentalFact]
    missing: dict[str, str]


def build_sec_fundamental_series(
    facts: Iterable[FundamentalFact],
    *,
    config: SecFundamentalsConfig,
) -> SecFundamentalSeries:
    selected = select_latest_fact_vintages(facts, config=config)
    annual = _annual_series(selected)
    quarters = _quarter_series(selected)
    _add_free_cash_flow_series(annual)
    _add_free_cash_flow_series(quarters)
    ttm = _ttm_series(quarters)
    instants = _instant_series(selected)
    latest_instants = {concept: values[-1] for concept, values in instants.items() if values}
    missing: dict[str, str] = {}
    for required in (
        "revenue",
        "net_income",
        "operating_cash_flow",
        "capital_expenditure",
        "weighted_average_diluted_shares",
    ):
        if required not in annual and required not in ttm:
            missing[required] = "No compatible annual or trailing-twelve-month SEC history"
    if "free_cash_flow" not in annual and "free_cash_flow" not in ttm:
        missing["free_cash_flow"] = (
            "Need compatible operating cash flow and capital expenditure periods"
        )
    return SecFundamentalSeries(
        selected_facts=selected,
        annual={key: tuple(values) for key, values in annual.items()},
        quarters={key: tuple(values) for key, values in quarters.items()},
        ttm=ttm,
        instants={key: tuple(values) for key, values in instants.items()},
        latest_instants=latest_instants,
        missing=missing,
    )


def select_latest_fact_vintages(
    facts: Iterable[FundamentalFact],
    *,
    config: SecFundamentalsConfig,
) -> tuple[FundamentalFact, ...]:
    priority = {
        (rule.canonical_concept, f"{taxonomy}:{source_concept}"): index
        for taxonomy in config.allowed_taxonomies
        for rule in config.concept_rules
        for index, source_concept in enumerate(rule.source_concepts)
    }
    selected: dict[tuple[str, str], FundamentalFact] = {}
    for fact in facts:
        if fact.provider != "sec" or (fact.concept, fact.source_concept) not in priority:
            continue
        key = (fact.concept, fact.period_identity)
        existing = selected.get(key)
        if existing is None or _fact_is_newer(
            candidate=fact,
            existing=existing,
            priority=priority,
        ):
            selected[key] = fact
    return tuple(
        sorted(
            selected.values(),
            key=lambda fact: (
                fact.concept,
                fact.period_end,
                fact.period_start or fact.period_end,
                fact.available_at,
            ),
        )
    )


def _fact_is_newer(
    *,
    candidate: FundamentalFact,
    existing: FundamentalFact,
    priority: dict[tuple[str, str], int],
) -> bool:
    same_revision_series = (
        candidate.source_concept == existing.source_concept
        and candidate.period_identity == existing.period_identity
        and candidate.accession == existing.accession
        and candidate.unit == existing.unit
    )
    if same_revision_series and candidate.source_revision != existing.source_revision:
        return candidate.source_revision > existing.source_revision
    return _fact_rank(candidate, priority) > _fact_rank(existing, priority)


def _fact_rank(
    fact: FundamentalFact,
    priority: dict[tuple[str, str], int],
) -> tuple[datetime, int, int, str]:
    source_priority = priority.get((fact.concept, fact.source_concept), 10_000)
    return (
        fact.available_at,
        fact.source_revision,
        -source_priority,
        fact.accession,
    )


def _annual_series(
    selected: tuple[FundamentalFact, ...],
) -> dict[str, list[FundamentalValue]]:
    annual: dict[str, list[FundamentalValue]] = {}
    for fact in selected:
        if fact.period_type != FundamentalFact.PeriodType.DURATION:
            continue
        value = _value_from_fact(fact)
        if value is None or not MIN_ANNUAL_DAYS <= value.duration_days <= MAX_ANNUAL_DAYS:
            continue
        annual.setdefault(fact.concept, []).append(value)
    for values in annual.values():
        values.sort(key=lambda value: (value.period_end, value.available_at))
    return annual


def _quarter_series(
    selected: tuple[FundamentalFact, ...],
) -> dict[str, list[FundamentalValue]]:
    direct: dict[tuple[str, date], FundamentalValue] = {}
    duration_facts: dict[tuple[str, str, str], list[FundamentalFact]] = {}
    for fact in selected:
        if fact.period_type != FundamentalFact.PeriodType.DURATION:
            continue
        value = _value_from_fact(fact)
        if value is None:
            continue
        if MIN_QUARTER_DAYS <= value.duration_days <= MAX_QUARTER_DAYS:
            direct[(fact.concept, fact.period_end)] = value
        duration_facts.setdefault(
            (fact.concept, fact.source_concept, fact.unit),
            [],
        ).append(fact)

    derived: dict[tuple[str, date], FundamentalValue] = {}
    for (concept, _source_concept, _unit), facts in duration_facts.items():
        if concept not in ADDITIVE_FLOW_CONCEPTS | WEIGHTED_AVERAGE_CONCEPTS:
            continue
        by_start: dict[date, list[FundamentalFact]] = {}
        for fact in facts:
            if fact.period_start is not None:
                by_start.setdefault(fact.period_start, []).append(fact)
        for same_start in by_start.values():
            same_start.sort(key=lambda fact: fact.period_end)
            for previous, current in zip(same_start, same_start[1:], strict=False):
                quarter_start = previous.period_end + timedelta(days=1)
                quarter_days = (current.period_end - quarter_start).days + 1
                if not MIN_QUARTER_DAYS <= quarter_days <= MAX_QUARTER_DAYS:
                    continue
                value = _subtract_ytd(
                    concept=concept,
                    previous=previous,
                    current=current,
                    quarter_start=quarter_start,
                )
                if value is not None:
                    derived[(concept, current.period_end)] = value

    combined = dict(direct)
    for key, derived_value in derived.items():
        direct_value = combined.get(key)
        if direct_value is None or derived_value.available_at > direct_value.available_at:
            combined[key] = derived_value
    quarters: dict[str, list[FundamentalValue]] = {}
    for (concept, _period_end), value in combined.items():
        quarters.setdefault(concept, []).append(value)
    for values in quarters.values():
        values.sort(key=lambda value: (value.period_end, value.available_at))
    return quarters


def _subtract_ytd(
    *,
    concept: str,
    previous: FundamentalFact,
    current: FundamentalFact,
    quarter_start: date,
) -> FundamentalValue | None:
    if previous.period_start is None or current.period_start is None:
        return None
    quarter_days = (current.period_end - quarter_start).days + 1
    if concept in ADDITIVE_FLOW_CONCEPTS:
        value = current.value - previous.value
        derivation = "ytd_difference"
    elif concept in WEIGHTED_AVERAGE_CONCEPTS:
        current_days = (current.period_end - current.period_start).days + 1
        previous_days = (previous.period_end - previous.period_start).days + 1
        weighted_total = current.value * current_days - previous.value * previous_days
        value = weighted_total / Decimal(quarter_days)
        if value <= 0:
            return None
        derivation = "weighted_ytd_difference"
    else:
        return None
    return FundamentalValue(
        concept=concept,
        value=value,
        unit=current.unit,
        period_start=quarter_start,
        period_end=current.period_end,
        available_at=max(previous.available_at, current.available_at),
        source_fact_ids=(str(previous.pk), str(current.pk)),
        accessions=tuple(dict.fromkeys((previous.accession, current.accession))),
        source_concepts=tuple(dict.fromkeys((previous.source_concept, current.source_concept))),
        derivation=derivation,
    )


def _ttm_series(
    quarters: dict[str, list[FundamentalValue]],
) -> dict[str, FundamentalValue]:
    result: dict[str, FundamentalValue] = {}
    for concept, values in quarters.items():
        if concept not in ADDITIVE_FLOW_CONCEPTS | WEIGHTED_AVERAGE_CONCEPTS:
            continue
        if len(values) < 4:
            continue
        window = values[-4:]
        if not _quarters_contiguous(window):
            continue
        if len({value.unit for value in window}) != 1:
            continue
        if len({value.source_concepts for value in window}) != 1:
            continue
        span_days = (window[-1].period_end - window[0].period_start).days + 1
        if not MIN_ANNUAL_DAYS <= span_days <= MAX_ANNUAL_DAYS:
            continue
        if concept in ADDITIVE_FLOW_CONCEPTS:
            value = sum((item.value for item in window), Decimal("0"))
            derivation = "sum_four_contiguous_quarters"
        else:
            total_days = sum(item.duration_days for item in window)
            value = sum(
                (item.value * item.duration_days for item in window), Decimal("0")
            ) / Decimal(total_days)
            derivation = "weighted_four_contiguous_quarters"
        result[concept] = FundamentalValue(
            concept=concept,
            value=value,
            unit=window[-1].unit,
            period_start=window[0].period_start,
            period_end=window[-1].period_end,
            available_at=max(item.available_at for item in window),
            source_fact_ids=tuple(
                dict.fromkeys(fact_id for item in window for fact_id in item.source_fact_ids)
            ),
            accessions=tuple(
                dict.fromkeys(accession for item in window for accession in item.accessions)
            ),
            source_concepts=window[-1].source_concepts,
            derivation=derivation,
        )
    _add_free_cash_flow_value(result)
    return result


def _quarters_contiguous(values: list[FundamentalValue]) -> bool:
    for previous, current in zip(values, values[1:], strict=False):
        gap = (current.period_start - previous.period_end).days - 1
        if gap != 0:
            return False
    return True


def _add_free_cash_flow_series(
    series: dict[str, list[FundamentalValue]],
) -> None:
    operating = {
        (value.period_start, value.period_end): value
        for value in series.get("operating_cash_flow", [])
    }
    capex = {
        (value.period_start, value.period_end): value
        for value in series.get("capital_expenditure", [])
    }
    derived: list[FundamentalValue] = []
    for period, operating_value in operating.items():
        capex_value = capex.get(period)
        if capex_value is None or capex_value.unit != operating_value.unit:
            continue
        derived.append(_free_cash_flow(operating_value, capex_value))
    if derived:
        series["free_cash_flow"] = sorted(
            derived,
            key=lambda value: (value.period_end, value.available_at),
        )


def _add_free_cash_flow_value(values: dict[str, FundamentalValue]) -> None:
    operating = values.get("operating_cash_flow")
    capex = values.get("capital_expenditure")
    if (
        operating is None
        or capex is None
        or operating.period_start != capex.period_start
        or operating.period_end != capex.period_end
        or operating.unit != capex.unit
    ):
        return
    values["free_cash_flow"] = _free_cash_flow(operating, capex)


def _free_cash_flow(
    operating: FundamentalValue,
    capex: FundamentalValue,
) -> FundamentalValue:
    return FundamentalValue(
        concept="free_cash_flow",
        value=operating.value - abs(capex.value),
        unit=operating.unit,
        period_start=operating.period_start,
        period_end=operating.period_end,
        available_at=max(operating.available_at, capex.available_at),
        source_fact_ids=tuple(dict.fromkeys((*operating.source_fact_ids, *capex.source_fact_ids))),
        accessions=tuple(dict.fromkeys((*operating.accessions, *capex.accessions))),
        source_concepts=tuple(dict.fromkeys((*operating.source_concepts, *capex.source_concepts))),
        derivation="operating_cash_flow_minus_absolute_capex",
    )


def _instant_series(
    selected: tuple[FundamentalFact, ...],
) -> dict[str, list[FundamentalFact]]:
    result: dict[str, list[FundamentalFact]] = {}
    for fact in selected:
        if fact.period_type != FundamentalFact.PeriodType.INSTANT:
            continue
        result.setdefault(fact.concept, []).append(fact)
    for facts in result.values():
        facts.sort(key=lambda fact: (fact.period_end, fact.available_at))
    return result


def _value_from_fact(fact: FundamentalFact) -> FundamentalValue | None:
    if fact.period_start is None:
        return None
    return FundamentalValue(
        concept=fact.concept,
        value=fact.value,
        unit=fact.unit,
        period_start=fact.period_start,
        period_end=fact.period_end,
        available_at=fact.available_at,
        source_fact_ids=(str(fact.pk),),
        accessions=(fact.accession,),
        source_concepts=(fact.source_concept,),
        derivation="reported",
    )

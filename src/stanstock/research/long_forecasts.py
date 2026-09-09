from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace
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
    TTM_SELECTION_LEGACY,
    TTM_SELECTION_NEWEST_QUARTER_ALIAS,
    FundamentalValue,
    NoncanonicalInstantIdentityError,
    ResolvedAvailability,
    SecFundamentalSeries,
    build_sec_fundamental_series,
    deferred_correction_payload,
    fact_selection_identity,
    partition_unproven_corrections,
    source_concept_priority,
)
from stanstock.research.long_forecast_config import (
    LONG_FORECAST_HORIZONS,
    LONG_SCENARIOS,
    REVIEWED_MAX_SAME_DATE_SOURCE_COMBINATIONS,
    LongForecastConfig,
    LongHorizonConfig,
    LongMetricFamilyConfig,
    LongScenarioConfig,
    long_forecast_config_hash,
)
from stanstock.research.types import Scenario

METHOD_NAME = "sec_per_share_growth_multiple_reversion"
MAX_SQL_IN_ITEMS = 500

JOINT_INVESTED_CAPITAL_POLICY = "joint_compatible_pair"
INDEPENDENT_INVESTED_CAPITAL_POLICY = "independent_nearest_compatible"

#: Canonical SEC concepts every long forecast version reads. Shared with the
#: read-only evidence audit so both see exactly the same evidence scope.
LONG_FORECAST_CONCEPTS = (
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
)


def ttm_selection_policy(config: LongForecastConfig) -> str:
    if config.newest_quarter_anchored_homogeneous_ttm_alias_selection is True:
        return TTM_SELECTION_NEWEST_QUARTER_ALIAS
    return TTM_SELECTION_LEGACY


def _invested_capital_policy(config: LongForecastConfig) -> str:
    if config.joint_compatible_invested_capital_pair_selection is True:
        return JOINT_INVESTED_CAPITAL_POLICY
    return INDEPENDENT_INVESTED_CAPITAL_POLICY


def invested_capital_alias_candidates_enabled(config: LongForecastConfig) -> bool:
    """Whether this config reads the same-date alias candidate surface.

    Only the joint pair search consumes it. Frozen `us-sec-long-v1`/
    `us-sec-long-v2` keep reading the collapsed `SecFundamentalSeries.instants`
    exactly as released.
    """
    return config.joint_compatible_invested_capital_pair_selection is True


def _evidence_selection_enabled(config: LongForecastConfig) -> bool:
    """Whether this config opts into the explicit evidence-selection payload.

    Frozen `us-sec-long-v1`/`us-sec-long-v2` configurations declare neither
    capability, so they never gain the extra payload key and their persisted
    calculation documents stay byte-for-byte unchanged.
    """
    return (
        config.newest_quarter_anchored_homogeneous_ttm_alias_selection is True
        or config.joint_compatible_invested_capital_pair_selection is True
    )


#: Correction policy that resolves a same-accession correction against the
#: observation that proves it, deferring one whose timing is unproven at the
#: requested cutoff.
CORRECTION_POLICY_PROVEN_OBSERVATION = "proven_observation_availability"

#: Frozen policy: read the recorded ``available_at`` and nothing else.
CORRECTION_POLICY_RECORDED_ONLY = "recorded_availability_only"


def same_date_combination_ceiling(config: LongForecastConfig) -> int:
    """The reviewed same-date combination ceiling this configuration declares.

    Only a configuration that enables joint invested-capital selection has
    one, and the loader accepts exactly the reviewed value, so this never
    returns a tuned or defaulted bound. A configuration without the joint
    search never reaches the enumeration that uses it.
    """
    ceiling = config.maximum_same_date_source_combinations
    if ceiling is None:
        raise ValueError(
            f"Long forecast config {config.version!r} does not declare "
            "maximum_same_date_source_combinations, so the same-date alias "
            "enumeration must not run"
        )
    return ceiling


def correction_availability_policy(config: LongForecastConfig) -> str:
    """The correction-timing policy this configuration selects.

    Driven by its own declared capability rather than inferred from the other
    two, so a future version can adopt one without silently acquiring this
    one. Frozen `us-sec-long-v1`/`us-sec-long-v2` declare no such key and
    therefore read recorded availability exactly as released.

    The forecast and the offline audit must answer this question with the
    same function. An audit that applied the prospective policy to a frozen
    version would report a selection that version would never make, which is
    worse than not auditing it at all.
    """
    if config.proven_observation_correction_availability is True:
        return CORRECTION_POLICY_PROVEN_OBSERVATION
    return CORRECTION_POLICY_RECORDED_ONLY


def resolve_corrections(
    facts: Sequence[FundamentalFact],
    *,
    data_cutoff: datetime,
    config: LongForecastConfig,
) -> tuple[list[FundamentalFact], tuple[ResolvedAvailability, ...]]:
    """Split visible facts into series inputs and deferred corrections.

    Only a configuration on `CORRECTION_POLICY_PROVEN_OBSERVATION` resolves
    anything. A frozen version gets its original list object back and an
    empty deferral tuple, so neither its selection nor its payloads can move.
    """
    if correction_availability_policy(config) != CORRECTION_POLICY_PROVEN_OBSERVATION:
        return list(facts), ()
    admitted, deferred = partition_unproven_corrections(facts, available_through=data_cutoff)
    if not deferred:
        return list(facts), ()
    return list(admitted), deferred


def _display_version_label(config: LongForecastConfig) -> str:
    """User-facing wording for reason strings, derived from behavior.

    Returns ``"long-v2"`` only when the adjacent annual diluted-share
    continuity capability is actually enabled; otherwise returns the
    original frozen ``"long-v1"`` wording. This keeps historical v1 reason
    strings byte-identical even though `config.version` now carries the
    full pinned identifier (e.g. ``"us-sec-long-v1"``). Method/config/model
    identity elsewhere continues to use the full `config.version`.
    """
    if config.adjacent_selected_annual_diluted_share_continuity is True:
        return "long-v2"
    return "long-v1"


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
    share_consistency: tuple[dict[str, Any], ...] = ()
    assessed_through: date | None = None


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
    available_at: datetime
    #: Declared alias priority for each entry of ``source_basis``. Populated
    #: only by the `us-sec-long-v3` same-date alias enumeration, where two
    #: candidates can otherwise differ only by generated fact UUIDs. It is
    #: never persisted; `source_basis` already names the aliases.
    source_priority: tuple[int, ...] = ()
    #: Stable observation identities of the participating facts, in the same
    #: order as ``fact_ids``. Ranking and ordering use these instead of the
    #: generated primary keys, so reassigning row UUIDs cannot change which
    #: snapshot a run selects. Never persisted.
    observation_identities: tuple[str, ...] = ()


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
    invested_capital_selection: dict[str, Any] | None = None


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
    failure_share_consistency: tuple[dict[str, Any], ...] = ()
    failure_assessed_through: date | None = None
    #: `us-sec-long-v3` only. When true, ``failure_fact_ids`` names candidate
    #: evidence this run read and *rejected* -- an unpaired or incompatible
    #: balance-sheet snapshot, an unusable metric branch -- so it is reported
    #: as assessed evidence and never as a selected, verified formula input.
    #: Frozen v1/v2 leave this false and keep their exact released
    #: classification, in which the same tuple stays inside ``input_facts``.
    assessed_failure_evidence: bool = False
    evidence_selection: dict[str, Any] | None = None
    #: Deduplicated closure of every fact the `us-sec-long-v3` evidence
    #: assessment referenced -- alias-selection lineage, TTM dependencies,
    #: every invested-capital candidate, every fact responsible for a refused
    #: combination space, and every rejected ``failure_fact_ids`` candidate --
    #: including facts that had no compatible partner or whose forecast was
    #: later withheld for peer insufficiency. Empty for frozen v1/v2.
    assessed_evidence_fact_ids: tuple[str, ...] = ()


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
    if (
        config.fundamentals_config_version is not None
        and config.fundamentals_config_version != sec_config.config_version
    ):
        raise ValueError(
            f"Long forecast config {config.version!r} binds SEC fundamentals "
            f"{config.fundamentals_config_version!r}, but the loaded fundamentals "
            f"configuration is {sec_config.config_version!r}"
        )
    company_ids = [listing.security.company_id for listing in listings]
    period_lookback = timedelta(
        days=(max(family.minimum_annual_periods for family in config.metric_families.values()) + 1)
        * 366
    )
    facts_by_company: dict[str, list[FundamentalFact]] = {}
    for fact in (
        asof.fundamental_facts_for_companies(
            company_ids=company_ids,
            concepts=list(LONG_FORECAST_CONCEPTS),
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
            data_cutoff=data_cutoff,
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
        {fact_id for state in states.values() for fact_id in _state_evidence_fact_ids(state)}
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
    data_cutoff: datetime,
    target_date: date,
    config: LongForecastConfig,
    sec_config: SecFundamentalsConfig,
    facts: list[FundamentalFact],
    classifications: list[CompanyClassificationObservation],
    filing_assets: dict[str, DataAsset],
) -> _CompanyState:
    """Resolve one listing's evidence and close its assessed-evidence manifest.

    Every fact the `us-sec-long-v3` assessment referenced -- selected or
    rejected -- is collected here so the forecast's immutable
    ``source_assets`` manifest and evidence payload can cover all of it. That
    closure includes the version's ``failure_fact_ids`` candidates, which
    `us-sec-long-v3` classifies as assessed rather than selected. Frozen
    v1/v2 carry no assessment payload, so the closure is empty and their
    manifests and ``input_facts`` are unchanged.

    `us-sec-long-v3` additionally resolves same-accession corrections against
    their *proven* availability. A correction persisted before
    `CORRECTION_AVAILABILITY_BASIS` existed carries the original acceptance
    timestamp, so the recorded cutoff alone would admit a restated value at a
    historical cutoff that predates the retrieval proving it. Such a
    correction is withheld from the series -- leaving the original
    observation, which *is* proven at that cutoff, in its place -- and
    recorded as assessed evidence with its own reason. Frozen v1/v2 keep
    reading exactly the facts they were handed.
    """
    admitted, deferred = resolve_corrections(
        facts,
        data_cutoff=data_cutoff,
        config=config,
    )
    state = replace(
        _resolve_company_state(
            listing=listing,
            price=price,
            price_asset=price_asset,
            asof_decision_time=asof_decision_time,
            target_date=target_date,
            config=config,
            sec_config=sec_config,
            facts=facts,
            series_facts=admitted,
            classifications=classifications,
            filing_assets=filing_assets,
        ),
        assessed_failure_evidence=_evidence_selection_enabled(config),
    )
    if state.evidence_selection is None:
        return state
    evidence_selection = {
        **state.evidence_selection,
        "assessed_evidence_role": ASSESSED_EVIDENCE_ROLE,
        "deferred_unproven_corrections": [
            deferred_correction_payload(entry, available_through=data_cutoff) for entry in deferred
        ],
    }
    candidates = _dedupe_text(
        (
            *_referenced_evidence_fact_ids(evidence_selection),
            *(state.failure_fact_ids if state.assessed_failure_evidence else ()),
        )
    )
    return replace(
        state,
        evidence_selection=evidence_selection,
        assessed_evidence_fact_ids=candidates,
    )


def _resolve_company_state(
    *,
    listing: Listing,
    price: float,
    price_asset: DataAsset,
    asof_decision_time: datetime,
    target_date: date,
    config: LongForecastConfig,
    sec_config: SecFundamentalsConfig,
    facts: list[FundamentalFact],
    series_facts: list[FundamentalFact],
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
    # ``fact_map`` deliberately covers *every* visible fact, including a
    # correction deferred for unproven timing: the manifest and the assessed
    # evidence payload must still be able to name and prove that row. Only
    # ``series_facts`` -- the proven-available subset -- may build values.
    fact_map = {str(fact.pk): fact for fact in facts}
    try:
        series = build_sec_fundamental_series(
            series_facts,
            config=sec_config,
            ttm_selection=ttm_selection_policy(config),
            alias_instant_candidates=(
                config.joint_compatible_invested_capital_pair_selection is True
            ),
        )
    except NoncanonicalInstantIdentityError as error:
        # Only the `us-sec-long-v3` alias boundary validates instant identity,
        # so this can never reach a frozen version. The whole listing is
        # withheld with the conflicting rows named; other listings in the same
        # run are unaffected.
        return _CompanyState(
            listing=listing,
            price=price,
            price_asset=price_asset,
            sic=None,
            metric=None,
            sustainable=None,
            insufficiency_reason=(
                "Balance-sheet evidence is unusable: "
                f"{error}. Same-date alias selection needs one canonical instant "
                "identity per balance-sheet date and never resolves a conflict by "
                "generated row identifier"
            ),
            fact_map=fact_map,
            filing_assets=filing_assets,
            evidence_selection=_boundary_failure_evidence_selection(
                config=config,
                status="noncanonical_instant_period_identity",
                reason=str(error),
                detail={"noncanonical_instant_facts": list(error.anomalies)},
            ),
        )
    evidence_selection = _evidence_selection_payload(series=series, config=config)
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
            evidence_selection=evidence_selection,
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
            evidence_selection=evidence_selection,
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
                f"SEC SIC {normalized_sic} is outside the supported "
                f"{_display_version_label(config)} industries"
            ),
            fact_map=fact_map,
            filing_assets=filing_assets,
            evidence_selection=evidence_selection,
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
            evidence_selection=evidence_selection,
            failure_fact_ids=fcf.fact_ids,
            failure_share_consistency=fcf.share_consistency,
            failure_assessed_through=fcf.assessed_through,
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
            evidence_selection=evidence_selection,
            failure_fact_ids=_dedupe_text((*fcf.fact_ids, *eps.fact_ids)),
            failure_share_consistency=(*fcf.share_consistency, *eps.share_consistency),
            failure_assessed_through=(
                fcf.assessed_through if fcf.assessed_through is not None else eps.assessed_through
            ),
        )
    assert metric is not None
    sustainable, sustainable_reason, invested_capital_assessment = _sustainable_growth(
        series=series,
        metric=metric,
        config=config,
        sec_config=sec_config,
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
        evidence_selection=_with_invested_capital_assessment(
            evidence_selection,
            invested_capital_assessment,
        ),
        failure_fact_ids=(
            () if sustainable is not None else _sustainable_candidate_fact_ids(series)
        ),
    )


def _with_invested_capital_assessment(
    evidence_selection: dict[str, Any] | None,
    assessment: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Attach an assessed invested-capital pair search to v3 provenance.

    A rejected search is recorded exactly like a successful one so a withheld
    forecast still shows the targets, candidates, and the disqualifying
    reason. Frozen v1/v2 carry no evidence-selection payload at all, so this
    returns ``None`` for them and their documents stay byte-identical.
    """
    if evidence_selection is None:
        return None
    return {**evidence_selection, "invested_capital_assessment": assessment}


def _evidence_selection_payload(
    *,
    series: SecFundamentalSeries,
    config: LongForecastConfig,
) -> dict[str, Any] | None:
    """Describe how this version selected its SEC evidence, or ``None``.

    Frozen `us-sec-long-v1`/`us-sec-long-v2` return ``None`` so their
    persisted calculation payloads are unchanged.

    ``ttm_dependencies`` records the facts every constructed TTM window
    actually depends on, including windows this forecast did not go on to
    use. Together with the alias-selection lineage and the invested-capital
    candidate list it makes the assessed-evidence manifest a closure rather
    than a sample.

    Schema 3 adds ``deferred_unproven_corrections`` (written by
    `_company_state`): same-accession corrections withheld because only a
    retrieval after the data cutoff proves their timing.
    """
    if not _evidence_selection_enabled(config):
        return None
    return {
        "schema_version": 3,
        "ttm_selection_policy": series.ttm_selection,
        "invested_capital_selection_policy": _invested_capital_policy(config),
        "bound_fundamentals_config_version": config.fundamentals_config_version,
        "ttm_alias_selection": {
            concept: series.ttm_alias_selection[concept]
            for concept in sorted(series.ttm_alias_selection)
        },
        "ttm_dependencies": {
            concept: {
                "period_start": value.period_start.isoformat(),
                "period_end": value.period_end.isoformat(),
                "unit": value.unit,
                "derivation": value.derivation,
                "source_concepts": list(value.source_concepts),
                "source_fact_ids": list(value.source_fact_ids),
            }
            for concept, value in sorted(series.ttm.items())
        },
        # Replaced with the real assessment once the invested-capital pair
        # search runs. ``None`` explicitly means "not assessed", never
        # "assessed and fine".
        "invested_capital_assessment": None,
    }


#: Label distinguishing candidate evidence that was merely *assessed* from the
#: selected formula inputs recorded under ``input_facts``. Assessed evidence
#: includes rejected, unpaired, and later-withheld candidates and is never a
#: claim that anything about it was confirmed.
ASSESSED_EVIDENCE_ROLE = "assessed_candidate_evidence"


def _boundary_failure_evidence_selection(
    *,
    config: LongForecastConfig,
    status: str,
    reason: str,
    detail: dict[str, Any],
) -> dict[str, Any] | None:
    """Evidence-selection payload for a refused `us-sec-long-v3` boundary.

    A boundary refusal (conflicting instant identities, or a same-date
    combination space above the reviewed ceiling) produces no series to
    describe, but it is still assessed evidence: the payload names the
    disqualifying rows and never claims a selection.
    """
    if not _evidence_selection_enabled(config):
        return None
    return {
        "schema_version": 3,
        "ttm_selection_policy": ttm_selection_policy(config),
        "invested_capital_selection_policy": _invested_capital_policy(config),
        "bound_fundamentals_config_version": config.fundamentals_config_version,
        "ttm_alias_selection": {},
        "ttm_dependencies": {},
        "invested_capital_assessment": {
            "schema_version": 1,
            "policy": _invested_capital_policy(config),
            "status": status,
            "assessment_status": "assessed_incompatible_or_unavailable",
            "rejection_reason": reason,
            "beginning_candidates": [],
            "ending_candidates": [],
            "compatible_pair_count": 0,
            "eligible_pair_count": 0,
            "selected_beginning_period_end": None,
            "selected_ending_period_end": None,
            "selected_debt_method": None,
            "selected_debt_components": [],
            "selected_source_basis": [],
            "selected_beginning_fact_ids": [],
            "selected_ending_fact_ids": [],
            **detail,
        },
    }


def _referenced_evidence_fact_ids(payload: Any) -> tuple[str, ...]:
    """Deduplicated closure of every fact id an assessment payload cites.

    Walking the payload -- rather than re-deriving a hand-maintained list --
    is what makes the manifest a closure: any current or future
    ``*_fact_id``/``*_fact_ids`` reference, in any nested candidate, lineage,
    dependency, or rejection record, is covered automatically.
    """
    found: list[str] = []

    def walk(node: Any, key: str | None) -> None:
        if isinstance(node, dict):
            for name, value in node.items():
                walk(value, name)
            return
        if isinstance(node, list):
            if key is not None and key.endswith("fact_ids"):
                found.extend(item for item in node if isinstance(item, str))
                return
            for item in node:
                walk(item, None)
            return
        if isinstance(node, str) and key is not None and key.endswith("fact_id"):
            found.append(node)

    walk(payload, None)
    return _dedupe_text(tuple(found))


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
        require_adjacent_selected_annual_diluted_share_continuity=(
            config.adjacent_selected_annual_diluted_share_continuity is True
        ),
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
            share_consistency=share_consistency,
            assessed_through=metric.period_end,
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
                f"{_display_version_label(config)} minimum {family_config.multiple_minimum:.3f}"
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
    require_adjacent_selected_annual_diluted_share_continuity: bool,
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
    if require_adjacent_selected_annual_diluted_share_continuity:
        for previous_period, current_period in zip(
            required_periods, required_periods[1:], strict=False
        ):
            previous_shares = shares[previous_period]
            current_period_shares = shares[current_period]
            previous_share_value = float(previous_shares.value)
            current_share_value = float(current_period_shares.value)
            relative_difference = abs(current_share_value / previous_share_value - 1.0)
            checks.append(
                {
                    "check": "adjacent_annual_diluted_shares",
                    "previous_period_end": previous_period[1].isoformat(),
                    "current_period_end": current_period[1].isoformat(),
                    "previous_shares": previous_share_value,
                    "current_shares": current_share_value,
                    "relative_difference": relative_difference,
                    "tolerance": tolerance,
                }
            )
            fact_ids.extend(
                (*previous_shares.source_fact_ids, *current_period_shares.source_fact_ids)
            )
            if relative_difference > tolerance:
                return (
                    tuple(checks),
                    _dedupe_text(fact_ids),
                    "Adjacent annual diluted-share basis continuity is incompatible/unverified "
                    f"between {previous_period[1].isoformat()} and "
                    f"{current_period[1].isoformat()}: {relative_difference:.1%} exceeds "
                    f"{tolerance:.1%}",
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
    sec_config: SecFundamentalsConfig,
) -> tuple[_SustainableGrowth | None, str, dict[str, Any] | None]:
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
        return None, "Sustainable growth requires TTM " + ", ".join(missing), None
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
        return None, "Sustainable-growth TTM inputs do not share the metric period", None
    if pretax.value <= 0:
        return None, "Sustainable growth requires positive TTM pretax income", None
    tax_rate_raw = float(tax.value / pretax.value)
    tax_rate = _clamp(
        tax_rate_raw,
        config.growth.tax_rate_minimum,
        config.growth.tax_rate_maximum,
    )
    nopat = float(operating.value) * (1.0 - tax_rate)
    if not math.isfinite(nopat) or nopat <= 0:
        return None, "Sustainable growth requires positive TTM NOPAT", None
    beginning_target = metric.current_period_start - timedelta(days=1)
    ending_target = metric.current_period_end
    tolerance_days = config.eligibility.balance_sheet_date_tolerance_days
    invested_capital_selection: dict[str, Any] | None = None
    assessment_payload: dict[str, Any] | None = None
    if config.joint_compatible_invested_capital_pair_selection is True:
        assessment = _joint_invested_capital_pair(
            series=series,
            beginning_target=beginning_target,
            ending_target=ending_target,
            tolerance_days=tolerance_days,
            priority=source_concept_priority(sec_config),
            maximum_combinations=same_date_combination_ceiling(config),
        )
        # The assessment is carried out of every branch below, including the
        # ones that reject the pair, so a withheld forecast still shows what
        # evidence was examined and why it was refused.
        assessment_payload = assessment.payload
        invested_capital_selection = assessment.selection
        if assessment.pair is None:
            return None, assessment.reason, assessment_payload
        beginning, ending = assessment.pair
    else:
        nearest_beginning, beginning_reason = _invested_capital_near(
            series=series,
            target_date=beginning_target,
            tolerance_days=tolerance_days,
        )
        if nearest_beginning is None:
            return None, f"Beginning invested capital unavailable: {beginning_reason}", None
        nearest_ending, ending_reason = _invested_capital_near(
            series=series,
            target_date=ending_target,
            tolerance_days=tolerance_days,
        )
        if nearest_ending is None:
            return None, f"Ending invested capital unavailable: {ending_reason}", None
        if not _invested_capital_compatible(nearest_beginning, nearest_ending):
            return (
                None,
                "Beginning/end invested-capital evidence uses incompatible source definitions",
                None,
            )
        beginning, ending = nearest_beginning, nearest_ending
    average = (beginning.value + ending.value) / 2.0
    if average <= 0:
        return None, "Average invested capital must be positive", assessment_payload
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
            invested_capital_selection=invested_capital_selection,
        ),
        "",
        assessment_payload,
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
    """Frozen long-v1/long-v2 selection: the nearest viable snapshot alone.

    Beginning and ending snapshots are chosen independently here, so a pair
    that turns out to use incompatible debt/source bases is withheld by the
    caller rather than searched around.
    """
    candidates = _invested_capital_candidates(
        series=series,
        target_date=target_date,
        tolerance_days=tolerance_days,
    )
    if not candidates:
        return None, f"no compatible balance sheet within {tolerance_days} days of {target_date}"
    return candidates[0], ""


def _invested_capital_candidates(
    *,
    series: SecFundamentalSeries,
    target_date: date,
    tolerance_days: int,
) -> tuple[_InvestedCapital, ...]:
    """Every viable snapshot within tolerance, nearest target date first.

    The ordering and the per-date construction rules are exactly the frozen
    ones; only the number of results returned differs from
    `_invested_capital_near`.
    """
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
    candidates: list[_InvestedCapital] = []
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
        candidates.append(
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
                available_at=max(fact.available_at for fact in (equity, cash, *debt_facts)),
            )
        )
    return tuple(candidates)


DEBT_COMPONENT_CONCEPTS = (
    "long_term_debt",
    "current_long_term_debt",
    "short_term_debt",
)

BALANCE_SHEET_CONCEPTS = (
    "equity",
    "cash_and_equivalents",
    "reported_long_term_debt",
    *DEBT_COMPONENT_CONCEPTS,
)

#: Hard ceiling on the same-date source-basis combinations `us-sec-long-v3`
#: will enumerate for one balance-sheet date, declared by the configuration
#: as `maximum_same_date_source_combinations`. The reviewed SEC fundamentals
#: configuration declares at most two aliases per balance-sheet concept, so a
#: real date stays far below it. Exceeding it is an explicit failure rather
#: than a silent truncation that would hide an available compatible pair, and
#: the loader accepts only the one reviewed value, so it is never tuned per
#: run.
MAX_SAME_DATE_SOURCE_COMBINATIONS = REVIEWED_MAX_SAME_DATE_SOURCE_COMBINATIONS


@dataclass(frozen=True, slots=True)
class _SameDateCombinationAxis:
    """One independent alias axis of a same-date source-basis space.

    An axis is a *factor* of the combination bound, never the product: the
    count that triggers the ceiling is derived by multiplying these, so every
    fact responsible for an oversized space is named without materializing a
    single combination. ``multiplier`` is what this axis contributes to that
    bound, which mirrors the debt rules exactly (an absent optional concept
    contributes no axis at all rather than a phantom option).
    """

    concept: str
    options: tuple[FundamentalFact, ...]
    multiplier: int

    def payload(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "option_count": len(self.options),
            "combination_multiplier": self.multiplier,
            "source_concepts": [fact.source_concept for fact in self.options],
            "fact_ids": [str(fact.pk) for fact in self.options],
        }


@dataclass(frozen=True, slots=True)
class _RejectedInvestedCapital:
    """One same-date source-basis combination refused by a value guard.

    It is assessed evidence and nothing more: the combination is never
    ranked, paired, or used in any arithmetic. Recording it keeps the
    configured aliases it consulted -- and therefore their Companyfacts and
    filing assets -- inside the immutable manifest, instead of vanishing
    because their values happened to be unusable.
    """

    period_end: date
    reason: str
    fact_ids: tuple[str, ...]
    source_basis: tuple[tuple[str, str], ...]


class SameDateSourceCombinationOverflow(ValueError):
    """One balance-sheet date declares more combinations than may be searched.

    The ceiling exists so a malformed or unexpectedly wide alias surface
    cannot turn one listing into an unbounded Cartesian product. It is
    evaluated from *counts*, before any product is materialized, so the
    refusal costs nothing even when the space is astronomically large.

    Neither truncating the space nor raising the ceiling is acceptable: both
    would silently decide which compatible pair the run is allowed to see.
    The listing is withheld with an explicit assessment instead, and the rest
    of the run continues.

    The refusal carries the per-axis aliases, options, and fact ids that
    produced the bound, so the disqualifying evidence is recorded rather than
    discarded: a reader can see exactly which filings made the space too wide
    without the run ever enumerating it.
    """

    def __init__(
        self,
        *,
        period_end: date,
        combination_count: int,
        ceiling: int,
        axes: tuple[_SameDateCombinationAxis, ...],
        assessed_candidates: tuple[_InvestedCapital, ...] = (),
        assessed_rejections: tuple[_RejectedInvestedCapital, ...] = (),
    ) -> None:
        self.period_end = period_end
        self.combination_count = combination_count
        self.ceiling = ceiling
        self.axes = axes
        #: Candidates already assessed at *earlier* dates on this side before
        #: the refusal. They are assessed evidence, never eligible: the whole
        #: side is withheld and no pair is selected from them.
        self.assessed_candidates = assessed_candidates
        #: Value-guard refusals already recorded at earlier dates on this
        #: side. They are assessed evidence too, and the refusal must not
        #: discard them.
        self.assessed_rejections = assessed_rejections
        super().__init__(
            f"Balance-sheet date {period_end.isoformat()} declares {combination_count} "
            f"same-date source-basis combinations, above the reviewed ceiling of "
            f"{ceiling}; refusing to guess a subset"
        )

    def responsible_fact_ids(self) -> tuple[str, ...]:
        """Every fact whose presence contributed to the refused bound."""
        return _dedupe_text(
            tuple(str(fact.pk) for axis in self.axes for fact in axis.options),
        )

    def detail(self) -> dict[str, Any]:
        return {
            "period_end": self.period_end.isoformat(),
            "combination_count": self.combination_count,
            "ceiling": self.ceiling,
            "axis_count": len(self.axes),
            "axes": [axis.payload() for axis in self.axes],
            "responsible_fact_ids": list(self.responsible_fact_ids()),
        }


def _alias_invested_capital_candidates(
    *,
    series: SecFundamentalSeries,
    target_date: date,
    tolerance_days: int,
    priority: dict[tuple[str, str], int],
    maximum_combinations: int,
) -> tuple[tuple[_InvestedCapital, ...], tuple[_RejectedInvestedCapital, ...]]:
    """Every permitted same-date source-basis snapshot, nearest date first.

    Returns the eligible candidates *and* the combinations a value guard
    refused. The second tuple never participates in pairing or ranking; it
    exists so refused evidence stays assessed and manifest-covered.

    The frozen path reads `series.instants`, which has already collapsed each
    canonical concept to one winning alias per period identity. A compatible
    beginning/end pair that exists only through a *non-winning* same-date
    equity, cash, or debt alias is therefore undiscoverable there. This
    `us-sec-long-v3`-only enumeration walks the retained
    ``(concept, alias, period identity)`` surface instead and emits one
    candidate per permitted combination.

    The debt rules are exactly the frozen ones: a reported long-term debt
    observation forbids the component basis at that date (so a component and
    its reported roll-up can never be counted twice), and otherwise the
    non-overlapping components present at the date are summed in their frozen
    order. Only the *alias* of each participating concept is enumerated;
    method, component set, and value guards are unchanged.
    """
    if not series.alias_instant_candidates:
        raise ValueError(
            "Same-date alias invested-capital candidates were requested, but the "
            "SEC fundamental series was built without the alias candidate surface"
        )
    by_concept_date: dict[str, dict[date, list[FundamentalFact]]] = {}
    for concept in BALANCE_SHEET_CONCEPTS:
        for alias in sorted(series.alias_instants.get(concept, {})):
            for fact in series.alias_instants[concept][alias]:
                by_concept_date.setdefault(concept, {}).setdefault(fact.period_end, []).append(fact)
    dates = sorted(
        {
            period_end
            for concept in BALANCE_SHEET_CONCEPTS
            for period_end in by_concept_date.get(concept, {})
            if abs((period_end - target_date).days) <= tolerance_days
        },
        key=lambda value: (abs((value - target_date).days), value),
    )
    candidates: list[_InvestedCapital] = []
    rejected: list[_RejectedInvestedCapital] = []
    for period_end in dates:
        axes = _same_date_combination_axes(by_concept_date, period_end)
        if axes is None:
            continue
        # Bound the space from the per-axis counts *before* materializing any
        # product, and keep those axes so a refusal can name every fact that
        # produced the bound instead of discarding the disqualifying evidence.
        combination_count = math.prod(axis.multiplier for axis in axes)
        if combination_count > maximum_combinations:
            raise SameDateSourceCombinationOverflow(
                period_end=period_end,
                combination_count=combination_count,
                ceiling=maximum_combinations,
                axes=axes,
                assessed_candidates=tuple(candidates),
                assessed_rejections=tuple(rejected),
            )
        equities = _alias_options(by_concept_date, "equity", period_end)
        cashes = _alias_options(by_concept_date, "cash_and_equivalents", period_end)
        debt_options = _debt_basis_options(by_concept_date, period_end)
        for equity in equities:
            for cash in cashes:
                for debt_method, debt_facts in debt_options:
                    candidate = _invested_capital_from_facts(
                        period_end=period_end,
                        equity=equity,
                        cash=cash,
                        debt_facts=debt_facts,
                        debt_method=debt_method,
                        priority=priority,
                    )
                    if isinstance(candidate, _RejectedInvestedCapital):
                        rejected.append(candidate)
                    else:
                        candidates.append(candidate)
    return (
        tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    abs((candidate.period_end - target_date).days),
                    candidate.period_end,
                    candidate.source_priority,
                    candidate.source_basis,
                    candidate.observation_identities,
                ),
            )
        ),
        _dedupe_rejections(rejected),
    )


def _alias_options(
    by_concept_date: dict[str, dict[date, list[FundamentalFact]]],
    concept: str,
    period_end: date,
) -> list[FundamentalFact]:
    return sorted(
        by_concept_date.get(concept, {}).get(period_end, []),
        key=lambda fact: (fact.source_concept, fact_selection_identity(fact)),
    )


def _same_date_combination_axes(
    by_concept_date: dict[str, dict[date, list[FundamentalFact]]],
    period_end: date,
) -> tuple[_SameDateCombinationAxis, ...] | None:
    """The independent alias axes at one balance-sheet date, or ``None``.

    ``None`` means the date cannot form any candidate at all (no equity, no
    cash, or no debt evidence) and is skipped exactly as before. Otherwise
    the returned axes are the complete factorization of that date's
    combination bound, so counting and refusing never require the product.
    """
    equities = _alias_options(by_concept_date, "equity", period_end)
    cashes = _alias_options(by_concept_date, "cash_and_equivalents", period_end)
    if not equities or not cashes:
        return None
    debt_axes = _debt_basis_axes(by_concept_date, period_end)
    if debt_axes is None:
        return None
    return (
        _combination_axis("equity", equities),
        _combination_axis("cash_and_equivalents", cashes),
        *debt_axes,
    )


def _combination_axis(
    concept: str,
    options: list[FundamentalFact],
) -> _SameDateCombinationAxis:
    return _SameDateCombinationAxis(
        concept=concept,
        options=tuple(options),
        multiplier=len(options),
    )


def _debt_basis_axes(
    by_concept_date: dict[str, dict[date, list[FundamentalFact]]],
    period_end: date,
) -> tuple[_SameDateCombinationAxis, ...] | None:
    """Factor the permitted debt bases at one date without building them.

    This mirrors `_debt_basis_options` exactly, axis by axis, so the ceiling
    is enforced on a number and the refusal can still name the aliases behind
    it. ``None`` means no debt basis exists at this date.
    """
    reported = _alias_options(by_concept_date, "reported_long_term_debt", period_end)
    short_term = _alias_options(by_concept_date, "short_term_debt", period_end)
    if reported:
        # Frozen rule: the reported roll-up excludes the component basis, and
        # an absent short-term alias leaves exactly one ("no short-term")
        # option rather than multiplying the space.
        axes = [_combination_axis("reported_long_term_debt", reported)]
        if short_term:
            axes.append(_combination_axis("short_term_debt", short_term))
        return tuple(axes)
    component_axes = [
        _combination_axis(concept, options)
        for concept in DEBT_COMPONENT_CONCEPTS
        if (options := _alias_options(by_concept_date, concept, period_end))
    ]
    return tuple(component_axes) if component_axes else None


def _debt_basis_options(
    by_concept_date: dict[str, dict[date, list[FundamentalFact]]],
    period_end: date,
) -> list[tuple[str, tuple[FundamentalFact, ...]]]:
    """Enumerate the permitted debt bases at one date, without double counting."""
    reported = _alias_options(by_concept_date, "reported_long_term_debt", period_end)
    short_term = _alias_options(by_concept_date, "short_term_debt", period_end)
    if reported:
        # Frozen rule: a reported long-term roll-up excludes the component
        # basis entirely at this date, so no component can be added twice.
        short_options: list[FundamentalFact | None] = list(short_term) if short_term else [None]
        return [
            (
                "reported_long_term_plus_short_term_borrowings",
                (reported_fact, *(() if short_fact is None else (short_fact,))),
            )
            for reported_fact in reported
            for short_fact in short_options
        ]
    component_options = [
        _alias_options(by_concept_date, concept, period_end) for concept in DEBT_COMPONENT_CONCEPTS
    ]
    present = [options for options in component_options if options]
    if not present:
        return []
    combinations: list[tuple[FundamentalFact, ...]] = [()]
    for options in present:
        combinations = [
            (*combination, option) for combination in combinations for option in options
        ]
    return [("sum_non_overlapping_debt_components", combination) for combination in combinations]


def _value_guard_rejection(
    *,
    period_end: date,
    reason: str,
    facts: tuple[FundamentalFact, ...],
) -> _RejectedInvestedCapital:
    """Build the one record every value-guard branch produces.

    All four guards -- nonpositive equity, negative cash, absent or negative
    debt, and nonfinite or nonpositive invested capital -- funnel through
    here, so there is exactly one provenance mechanism rather than one per
    branch. Only the reason differs.
    """
    return _RejectedInvestedCapital(
        period_end=period_end,
        reason=reason,
        fact_ids=_dedupe_text(tuple(str(fact.pk) for fact in facts)),
        source_basis=tuple(
            (fact.concept, fact.source_concept) for fact in facts if fact.source_concept
        ),
    )


def _invested_capital_from_facts(
    *,
    period_end: date,
    equity: FundamentalFact,
    cash: FundamentalFact,
    debt_facts: tuple[FundamentalFact, ...],
    debt_method: str,
    priority: dict[tuple[str, str], int],
) -> _InvestedCapital | _RejectedInvestedCapital:
    """Apply the frozen value guards to one explicit source-basis combination.

    The guards themselves are unchanged: a rejected combination never enters
    the arithmetic, the candidate list, or any ranking. What changed is that
    a rejection is now *returned* instead of discarded, so the configured
    aliases it consulted stay visible as assessed evidence and their
    Companyfacts and filing assets stay inside the manifest closure.
    """
    participating = (equity, cash, *debt_facts)
    if equity.value <= 0:
        return _value_guard_rejection(
            period_end=period_end,
            reason="Equity must be positive",
            facts=participating,
        )
    if cash.value < 0:
        return _value_guard_rejection(
            period_end=period_end,
            reason="Cash and equivalents must not be negative",
            facts=participating,
        )
    if not debt_facts or any(fact.value < 0 for fact in debt_facts):
        return _value_guard_rejection(
            period_end=period_end,
            reason="Debt basis must be present and non-negative",
            facts=participating,
        )
    debt = float(sum((fact.value for fact in debt_facts), Decimal("0")))
    equity_value = float(equity.value)
    cash_value = float(cash.value)
    invested = debt + equity_value - cash_value
    if not math.isfinite(invested) or invested <= 0:
        return _value_guard_rejection(
            period_end=period_end,
            reason="Invested capital must be finite and positive",
            facts=participating,
        )
    source_basis = (
        ("equity", equity.source_concept),
        ("cash_and_equivalents", cash.source_concept),
        *((fact.concept, fact.source_concept) for fact in debt_facts),
    )
    return _InvestedCapital(
        value=invested,
        period_end=period_end,
        debt=debt,
        equity=equity_value,
        cash=cash_value,
        debt_method=debt_method,
        debt_components=tuple(fact.concept for fact in debt_facts),
        source_basis=source_basis,
        fact_ids=_dedupe_text(
            (
                str(equity.pk),
                str(cash.pk),
                *(str(fact.pk) for fact in debt_facts),
            )
        ),
        available_at=max(fact.available_at for fact in (equity, cash, *debt_facts)),
        source_priority=tuple(
            priority.get((concept, source_concept), 10_000)
            for concept, source_concept in source_basis
        ),
        observation_identities=tuple(
            fact_selection_identity(fact) for fact in (equity, cash, *debt_facts)
        ),
    )


def _invested_capital_compatible(
    beginning: _InvestedCapital,
    ending: _InvestedCapital,
) -> bool:
    return (
        beginning.debt_method == ending.debt_method
        and beginning.debt_components == ending.debt_components
        and beginning.source_basis == ending.source_basis
    )


@dataclass(frozen=True, slots=True)
class _InvestedCapitalPairAssessment:
    """Complete, always-produced record of one joint pair search.

    A failed search is evidence too: it names the targets, every candidate
    snapshot with its facts and source basis, how many compatible pairs
    existed, and why the search was rejected. The payload is `us-sec-long-v3`
    only, so frozen v1/v2 documents are unaffected, and it never labels
    rejected evidence verified.
    """

    status: str
    reason: str
    payload: dict[str, Any]
    pair: tuple[_InvestedCapital, _InvestedCapital] | None
    selection: dict[str, Any] | None


def _joint_invested_capital_pair(
    *,
    series: SecFundamentalSeries,
    beginning_target: date,
    ending_target: date,
    tolerance_days: int,
    priority: dict[tuple[str, str], int],
    maximum_combinations: int,
) -> _InvestedCapitalPairAssessment:
    """Search every beginning/ending snapshot pair for a compatible one.

    Ranking uses only deterministic evidence criteria -- combined and
    per-side target-date distance, then eligible-evidence recency, then
    declared alias priority and stable date/fact-id tie-breaks. It never
    inspects the resulting ROIC, reinvestment, growth, forecast, or scenario
    favorability.

    Every outcome, including every failure, returns a complete assessment.
    A same-date combination space above the reviewed ceiling is one of those
    failures: it withholds this listing explicitly instead of propagating an
    exception that would abort the whole analysis, and it keeps both the
    facts responsible for the refused bound and whatever the other side had
    already assessed.
    """
    try:
        beginnings, beginning_rejections = _alias_invested_capital_candidates(
            maximum_combinations=maximum_combinations,
            series=series,
            target_date=beginning_target,
            tolerance_days=tolerance_days,
            priority=priority,
        )
    except SameDateSourceCombinationOverflow as error:
        return _combination_overflow_assessment(
            error,
            beginning_target=beginning_target,
            ending_target=ending_target,
            tolerance_days=tolerance_days,
            beginnings=error.assessed_candidates,
            endings=(),
            beginning_rejections=error.assessed_rejections,
            ending_rejections=(),
        )
    try:
        endings, ending_rejections = _alias_invested_capital_candidates(
            maximum_combinations=maximum_combinations,
            series=series,
            target_date=ending_target,
            tolerance_days=tolerance_days,
            priority=priority,
        )
    except SameDateSourceCombinationOverflow as error:
        return _combination_overflow_assessment(
            error,
            beginning_target=beginning_target,
            ending_target=ending_target,
            tolerance_days=tolerance_days,
            beginnings=beginnings,
            endings=error.assessed_candidates,
            beginning_rejections=beginning_rejections,
            ending_rejections=error.assessed_rejections,
        )
    payload = _pair_search_payload(
        beginning_target=beginning_target,
        ending_target=ending_target,
        tolerance_days=tolerance_days,
        beginnings=beginnings,
        endings=endings,
        beginning_rejections=beginning_rejections,
        ending_rejections=ending_rejections,
        pair_count=0,
    )
    pairs = [
        (beginning, ending)
        for beginning in beginnings
        for ending in endings
        if _invested_capital_compatible(beginning, ending)
    ]
    payload["compatible_pair_count"] = len(pairs)
    payload["eligible_pair_count"] = len(pairs)
    if not beginnings:
        return _rejected_pair_assessment(
            payload,
            status="missing_beginning_candidates",
            reason=(
                "Beginning invested capital unavailable: no compatible balance sheet "
                f"within {tolerance_days} days of {beginning_target}"
            ),
        )
    if not endings:
        return _rejected_pair_assessment(
            payload,
            status="missing_ending_candidates",
            reason=(
                "Ending invested capital unavailable: no compatible balance sheet "
                f"within {tolerance_days} days of {ending_target}"
            ),
        )
    if not pairs:
        return _rejected_pair_assessment(
            payload,
            status="no_compatible_pair",
            reason=(
                "No compatible beginning/end invested-capital pair within "
                f"{tolerance_days} days of {beginning_target} and {ending_target}: "
                "every candidate pair uses incompatible debt-method, debt-component, "
                "or source-concept bases"
            ),
        )
    beginning, ending = min(
        pairs,
        key=lambda pair: _invested_capital_pair_rank(
            pair,
            beginning_target=beginning_target,
            ending_target=ending_target,
        ),
    )
    selection: dict[str, Any] = {
        "policy": JOINT_INVESTED_CAPITAL_POLICY,
        "tolerance_days": tolerance_days,
        "beginning_target_date": beginning_target.isoformat(),
        "ending_target_date": ending_target.isoformat(),
        "beginning_candidate_period_ends": payload["beginning_candidate_period_ends"],
        "ending_candidate_period_ends": payload["ending_candidate_period_ends"],
        "eligible_pair_count": len(pairs),
        "selected_beginning_period_end": beginning.period_end.isoformat(),
        "selected_ending_period_end": ending.period_end.isoformat(),
        "selected_debt_method": beginning.debt_method,
        "selected_debt_components": list(beginning.debt_components),
        "selected_source_basis": [
            {"concept": concept, "source_concept": source_concept}
            for concept, source_concept in beginning.source_basis
        ],
        "selected_beginning_fact_ids": list(beginning.fact_ids),
        "selected_ending_fact_ids": list(ending.fact_ids),
    }
    payload.update(
        {
            "status": "selected_compatible_pair",
            "assessment_status": "selected_compatible_pair",
            "rejection_reason": "",
            "selected_beginning_period_end": beginning.period_end.isoformat(),
            "selected_ending_period_end": ending.period_end.isoformat(),
            "selected_debt_method": beginning.debt_method,
            "selected_debt_components": list(beginning.debt_components),
            "selected_source_basis": selection["selected_source_basis"],
            "selected_beginning_fact_ids": list(beginning.fact_ids),
            "selected_ending_fact_ids": list(ending.fact_ids),
        }
    )
    return _InvestedCapitalPairAssessment(
        status="selected_compatible_pair",
        reason="",
        payload=payload,
        pair=(beginning, ending),
        selection=selection,
    )


def _pair_search_payload(
    *,
    beginning_target: date,
    ending_target: date,
    tolerance_days: int,
    beginnings: tuple[_InvestedCapital, ...],
    endings: tuple[_InvestedCapital, ...],
    beginning_rejections: tuple[_RejectedInvestedCapital, ...] = (),
    ending_rejections: tuple[_RejectedInvestedCapital, ...] = (),
    pair_count: int,
) -> dict[str, Any]:
    """The complete, always-produced record of one invested-capital search.

    Schema 2 adds ``value_rejected_candidates``: combinations a value guard
    refused. They are assessed evidence, never candidates, and are listed
    here so the manifest closure covers the aliases they consulted.
    """
    return {
        "schema_version": 2,
        "policy": JOINT_INVESTED_CAPITAL_POLICY,
        "tolerance_days": tolerance_days,
        "beginning_target_date": beginning_target.isoformat(),
        "ending_target_date": ending_target.isoformat(),
        "beginning_candidates": [
            _invested_capital_candidate_payload(candidate) for candidate in beginnings
        ],
        "ending_candidates": [
            _invested_capital_candidate_payload(candidate) for candidate in endings
        ],
        "beginning_candidate_period_ends": [
            candidate.period_end.isoformat() for candidate in beginnings
        ],
        "ending_candidate_period_ends": [candidate.period_end.isoformat() for candidate in endings],
        "beginning_candidate_count": len(beginnings),
        "ending_candidate_count": len(endings),
        "compatible_pair_count": pair_count,
        "eligible_pair_count": pair_count,
        "value_rejected_candidates": [
            {
                "side": side,
                "period_end": rejection.period_end.isoformat(),
                "reason": rejection.reason,
                "source_basis": [
                    {"concept": concept, "source_concept": source_concept}
                    for concept, source_concept in rejection.source_basis
                ],
                "fact_ids": list(rejection.fact_ids),
            }
            for side, rejections in (
                ("beginning", beginning_rejections),
                ("ending", ending_rejections),
            )
            for rejection in rejections
        ],
    }


def _combination_overflow_assessment(
    error: SameDateSourceCombinationOverflow,
    *,
    beginning_target: date,
    ending_target: date,
    tolerance_days: int,
    beginnings: tuple[_InvestedCapital, ...],
    endings: tuple[_InvestedCapital, ...],
    beginning_rejections: tuple[_RejectedInvestedCapital, ...] = (),
    ending_rejections: tuple[_RejectedInvestedCapital, ...] = (),
) -> _InvestedCapitalPairAssessment:
    """Refuse the search while keeping every disqualifying fact visible.

    The refusal is precisely what makes these facts matter, so the axes that
    produced the bound, their aliases and fact ids, the resulting count, and
    the unchanged ceiling are all recorded. Candidates already assessed on
    the other side -- or at an earlier date on the refused side -- are
    retained as assessed evidence rather than dropped, as are any value-guard
    refusals recorded before the bound was hit. None of it is eligible: no
    pair is selected and no product is ever enumerated.
    """
    payload = _pair_search_payload(
        beginning_target=beginning_target,
        ending_target=ending_target,
        tolerance_days=tolerance_days,
        beginnings=beginnings,
        endings=endings,
        beginning_rejections=beginning_rejections,
        ending_rejections=ending_rejections,
        pair_count=0,
    )
    payload["same_date_combination_overflow"] = error.detail()
    return _rejected_pair_assessment(
        payload,
        status="same_date_combination_ceiling_exceeded",
        reason=str(error),
    )


def _rejected_pair_assessment(
    payload: dict[str, Any],
    *,
    status: str,
    reason: str,
) -> _InvestedCapitalPairAssessment:
    payload.update(
        {
            "status": status,
            # Rejected evidence is *assessed*, never verified: it disqualified
            # the forecast and must never read as a confirmed selection.
            "assessment_status": "assessed_incompatible_or_unavailable",
            "rejection_reason": reason,
            "selected_beginning_period_end": None,
            "selected_ending_period_end": None,
            "selected_debt_method": None,
            "selected_debt_components": [],
            "selected_source_basis": [],
            "selected_beginning_fact_ids": [],
            "selected_ending_fact_ids": [],
        }
    )
    return _InvestedCapitalPairAssessment(
        status=status,
        reason=reason,
        payload=payload,
        pair=None,
        selection=None,
    )


def _invested_capital_pair_rank(
    pair: tuple[_InvestedCapital, _InvestedCapital],
    *,
    beginning_target: date,
    ending_target: date,
) -> tuple[
    int,
    int,
    int,
    float,
    float,
    str,
    str,
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Rank a candidate pair on evidence alone, never on the resulting numbers.

    Same-date alias enumeration can produce two candidates that agree on
    date and availability, so declared alias priority and the canonical
    source basis are compared before anything else. The final terms are the
    participating observations' stable identities, never their generated
    fact UUIDs: reassigning row identifiers must not be able to change which
    pair a run selects.
    """
    beginning, ending = pair
    beginning_distance = abs((beginning.period_end - beginning_target).days)
    ending_distance = abs((ending.period_end - ending_target).days)
    return (
        beginning_distance + ending_distance,
        ending_distance,
        beginning_distance,
        -ending.available_at.timestamp(),
        -beginning.available_at.timestamp(),
        ending.period_end.isoformat(),
        beginning.period_end.isoformat(),
        ending.source_priority,
        beginning.source_priority,
        ending.source_basis,
        beginning.source_basis,
        ending.observation_identities,
        beginning.observation_identities,
    )


def audit_invested_capital_pairs(
    *,
    series: SecFundamentalSeries,
    beginning_target: date,
    ending_target: date,
    tolerance_days: int,
    priority: dict[tuple[str, str], int],
    maximum_combinations: int,
) -> dict[str, Any]:
    """Report invested-capital pair availability without producing a forecast.

    This is a read-only diagnostic over already-persisted evidence. It reuses
    the exact candidate enumeration, compatibility rule, and joint ranking
    used by the forecast path, so an operator sees what the engine would
    actually select, including whether the frozen independent nearest-date
    choices would have missed an available compatible pair.
    """
    legacy_beginnings = _invested_capital_candidates(
        series=series,
        target_date=beginning_target,
        tolerance_days=tolerance_days,
    )
    legacy_endings = _invested_capital_candidates(
        series=series,
        target_date=ending_target,
        tolerance_days=tolerance_days,
    )
    assessment = _joint_invested_capital_pair(
        series=series,
        beginning_target=beginning_target,
        ending_target=ending_target,
        tolerance_days=tolerance_days,
        priority=priority,
        maximum_combinations=maximum_combinations,
    )
    independent_compatible = bool(
        legacy_beginnings
        and legacy_endings
        and _invested_capital_compatible(legacy_beginnings[0], legacy_endings[0])
    )
    report = dict(assessment.payload)
    report.update(
        {
            "compatible_pair_available": assessment.pair is not None,
            "independent_nearest_pair_compatible": independent_compatible,
            "independent_nearest_beginning": (
                _invested_capital_candidate_payload(legacy_beginnings[0])
                if legacy_beginnings
                else None
            ),
            "independent_nearest_ending": (
                _invested_capital_candidate_payload(legacy_endings[0]) if legacy_endings else None
            ),
            "joint_selection_recovers_missed_pair": (
                assessment.pair is not None and not independent_compatible
            ),
            "selection": assessment.selection,
            "reason": assessment.reason,
        }
    )
    return report


def _invested_capital_candidate_payload(value: _InvestedCapital) -> dict[str, Any]:
    return {
        "period_end": value.period_end.isoformat(),
        "debt_method": value.debt_method,
        "debt_components": list(value.debt_components),
        "source_basis": [
            {"concept": concept, "source_concept": source_concept}
            for concept, source_concept in value.source_basis
        ],
        "available_at": value.available_at.isoformat(),
        "fact_ids": list(value.fact_ids),
    }


def _forecasts_for_state_split_basis(
    *,
    state: _CompanyState,
    target_date: date,
    config: LongForecastConfig,
) -> dict[str, Any] | None:
    if state.metric is not None:
        return _split_basis_payload(
            checks=state.metric.share_consistency,
            verified_through=state.metric.current_period_end,
            assessed_through=None,
            target_date=target_date,
            config=config,
        )
    if config.adjacent_selected_annual_diluted_share_continuity is not True:
        return None
    if not state.failure_share_consistency or state.failure_assessed_through is None:
        return None
    return _split_basis_payload(
        checks=state.failure_share_consistency,
        verified_through=None,
        assessed_through=state.failure_assessed_through,
        target_date=target_date,
        config=config,
    )


def _forecasts_for_state(
    *,
    state: _CompanyState,
    states: dict[str, _CompanyState],
    target_date: date,
    config: LongForecastConfig,
    sec_config: SecFundamentalsConfig,
) -> dict[str, LongForecast]:
    split_basis = _forecasts_for_state_split_basis(
        state=state,
        target_date=target_date,
        config=config,
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
            evidence_selection=_state_evidence_selection_payload(state),
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
            evidence_selection=_state_evidence_selection_payload(state),
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
                checks=state.metric.share_consistency,
                verified_through=state.metric.current_period_end,
                assessed_through=None,
                target_date=target_date,
                config=config,
            ),
            evidence_selection=_state_evidence_selection_payload(state),
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
        "Positive-return probability is unavailable for deterministic "
        f"{_display_version_label(config)} until qualifying prospective outcomes exist"
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
    calculation: dict[str, Any] = {
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
            checks=state.metric.share_consistency,
            verified_through=state.metric.current_period_end,
            assessed_through=None,
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
    if state.sustainable.invested_capital_selection is not None:
        calculation["formula_inputs"]["invested_capital_selection"] = (
            state.sustainable.invested_capital_selection
        )
    state_evidence_selection = _state_evidence_selection_payload(state)
    if state_evidence_selection is not None:
        calculation["evidence_selection"] = state_evidence_selection
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
    evidence_selection: dict[str, Any] | None = None,
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
            evidence_selection=evidence_selection,
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
    evidence_selection: dict[str, Any] | None = None,
) -> LongForecast:
    calculation: dict[str, Any] = {
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
    }
    if evidence_selection is not None:
        calculation["evidence_selection"] = evidence_selection
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
        calculation=calculation,
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
            _state_evidence_fact_ids(state),
        )
    )
    return _dedupe_assets(assets)


def _state_selected_fact_ids(state: _CompanyState) -> tuple[str, ...]:
    """Facts that became selected, verified formula inputs for this listing.

    "Selected" means the value actually entered this listing's metric,
    share-consistency, or sustainable-growth arithmetic. Under
    `us-sec-long-v3` (``assessed_failure_evidence``) a rejected candidate is
    therefore excluded here and reported as assessed evidence instead: a
    no-compatible-pair, missing-side, refused-boundary, or unusable-metric
    result selects nothing at all. Frozen v1/v2 keep their released
    classification, in which ``failure_fact_ids`` stays in ``input_facts``.
    """
    fact_ids: tuple[str, ...] = ()
    if state.metric is not None:
        fact_ids = (*fact_ids, *state.metric.fact_ids)
    if state.sustainable is not None:
        fact_ids = (*fact_ids, *state.sustainable.fact_ids)
    if not state.assessed_failure_evidence:
        fact_ids = (*fact_ids, *state.failure_fact_ids)
    return _dedupe_text(fact_ids)


def _state_assessed_fact_ids(state: _CompanyState) -> tuple[str, ...]:
    """Candidate facts this listing considered but did not select.

    Rejected invested-capital candidates, facts responsible for a refused
    combination space, unselected TTM alias lineages, dependencies of windows
    that lost, and every `us-sec-long-v3` ``failure_fact_ids`` candidate. A
    fact that did enter the arithmetic is never listed here, so the selected
    and assessed sets are disjoint by construction.
    """
    assessed = state.assessed_evidence_fact_ids
    if state.assessed_failure_evidence:
        assessed = (*assessed, *state.failure_fact_ids)
    selected = set(_state_selected_fact_ids(state))
    return tuple(fact_id for fact_id in _dedupe_text(assessed) if fact_id not in selected)


def _state_evidence_fact_ids(state: _CompanyState) -> tuple[str, ...]:
    """Selected inputs plus every assessed candidate this listing referenced.

    This union -- and nothing narrower -- is what the immutable source-asset
    manifest closes over. A rejected invested-capital candidate, a fact that
    forced a combination refusal, an unselected TTM alias lineage, and a
    dependency of a window that lost are all evidence the run actually read,
    so their companyfacts and filing assets must be provable from the
    persisted forecast even though none of them is a selected input.
    """
    return _dedupe_text((*_state_selected_fact_ids(state), *_state_assessed_fact_ids(state)))


def _state_evidence_selection_payload(state: _CompanyState) -> dict[str, Any] | None:
    """Split this listing's evidence into selected, assessed, and manifest sets.

    Three explicit, non-overlapping-by-construction lists replace any
    inference from ``input_facts`` membership:

    - ``selected_input_fact_ids`` -- exactly the facts described in
      ``input_facts``; each one entered the metric, share-consistency, or
      sustainable-growth arithmetic;
    - ``assessed_evidence_fact_ids``/``assessed_evidence`` -- every candidate
      that was read and considered but *not* selected: rejected
      invested-capital candidates, the facts responsible for a refused
      combination space, unselected alias lineages, and dependencies of
      windows that lost. An entry here is never a verified input;
    - ``manifest_evidence_fact_ids`` -- the union the immutable
      ``source_assets`` manifest closes over.

    Frozen v1/v2 carry no evidence-selection payload at all, so they gain
    none of these keys.
    """
    if state.evidence_selection is None:
        return None
    selected = _state_selected_fact_ids(state)
    assessed = _state_assessed_fact_ids(state)
    return {
        **state.evidence_selection,
        # (a) selected: entered the metric/share/sustainable arithmetic.
        "selected_input_fact_ids": [fact_id for fact_id in selected if fact_id in state.fact_map],
        # (b) assessed: read and considered, never selected.
        "assessed_evidence_fact_ids": list(assessed),
        "assessed_evidence": [
            _fact_reference(state.fact_map[fact_id], state.filing_assets)
            for fact_id in assessed
            if fact_id in state.fact_map
        ],
        # (c) the union the immutable source-asset manifest closes over.
        "manifest_evidence_fact_ids": list(_state_evidence_fact_ids(state)),
    }


def _state_input_fact_payloads(state: _CompanyState) -> list[dict[str, Any]]:
    return [
        _fact_payload(state.fact_map[fact_id], state.filing_assets)
        for fact_id in _state_selected_fact_ids(state)
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
    checks: tuple[dict[str, Any], ...],
    verified_through: date | None,
    assessed_through: date | None,
    target_date: date,
    config: LongForecastConfig,
) -> dict[str, Any]:
    if verified_through is not None:
        return {
            "basis": "as_filed_diluted_shares_vs_split_adjusted_price",
            "verified_through": verified_through.isoformat(),
            "post_period_exposure_days": (target_date - verified_through).days,
            "maximum_exposure_days": config.eligibility.maximum_metric_age_days,
            "continuity_tolerance": (config.eligibility.share_consistency_relative_tolerance),
            "continuity_checks": list(checks),
            "residual_risk": "unverified_post_period_split",
        }
    assert assessed_through is not None
    return {
        "basis": "as_filed_diluted_shares_vs_split_adjusted_price",
        "assessment_status": "incompatible_or_unverified",
        "assessed_through": assessed_through.isoformat(),
        "post_period_exposure_days": (target_date - assessed_through).days,
        "maximum_exposure_days": config.eligibility.maximum_metric_age_days,
        "continuity_tolerance": (config.eligibility.share_consistency_relative_tolerance),
        "continuity_checks": list(checks),
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


def _dedupe_rejections(
    rejections: list[_RejectedInvestedCapital],
) -> tuple[_RejectedInvestedCapital, ...]:
    """Collapse repeats deterministically, keeping the first reason seen.

    One unusable alias can be refused once per debt basis it was paired
    with, and the manifest only needs each distinct (date, basis, reason)
    once.
    """
    unique: dict[tuple[Any, ...], _RejectedInvestedCapital] = {}
    for rejection in rejections:
        key = (rejection.period_end, rejection.source_basis, rejection.reason)
        unique.setdefault(key, rejection)
    return tuple(
        unique[key] for key in sorted(unique, key=lambda item: (item[0], item[1], item[2]))
    )


def _dedupe_text(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _dedupe_assets(assets: list[DataAsset]) -> tuple[DataAsset, ...]:
    return tuple({str(asset.pk): asset for asset in assets}.values())


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))

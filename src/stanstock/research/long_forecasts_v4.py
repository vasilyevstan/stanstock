from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, cast
from uuid import UUID

import polars as pl

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.asof import (
    AsOfData,
    PriceFrameChecksumMismatchError,
    raw_price_asset_for,
)
from stanstock.data.assets import AssetStore, read_checksummed_bytes
from stanstock.data.models import (
    Company,
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    Listing,
    Region,
    Security,
    SourceObservationEvent,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.data.sec_config import SecCikConfig
from stanstock.data.sec_evidence import (
    COMPANYFACTS_KIND,
    HISTORY_FILENAME_METADATA_KEY,
    MAPPING_KIND,
    MAPPING_SUBJECT,
    SUBMISSIONS_HISTORY_KIND,
    SUBMISSIONS_KIND,
)
from stanstock.data.sec_fundamentals import (
    CORRECTION_AVAILABILITY_BASIS,
    CORRECTION_QUALITY_FLAG,
    REBOUND_QUALITY_FLAG,
    TTM_SELECTION_NEWEST_QUARTER_ALIAS,
    FundamentalValue,
    SecFundamentalSeries,
    build_sec_fundamental_series,
    deferred_correction_payload,
    partition_unproven_corrections,
    resolve_availability,
)
from stanstock.data.sec_ingestion import (
    SEC_EXCHANGE_MIC_RULE,
    ObservationEvidenceError,
    RawObservationLineage,
    SecCompanyfactsInspection,
    SecConfiguredObservation,
    SecDerivationError,
    SecFactDerivation,
    derive_sec_current_submissions,
    derive_sec_historical_submissions,
    inspect_sec_companyfacts,
    parse_sec_mapping,
    reconcile_sec_observation_lineages,
    validate_sec_observation_event,
    verify_sec_exchange_mic,
)
from stanstock.research.long_forecast_config import (
    LONG_FORECAST_HORIZONS,
    LONG_SCENARIOS,
    LONG_V4_EFFECTIVE_CONFIG_HASH,
    LONG_V4_FUNDAMENTALS_CONFIG_FILE_SHA256,
    LONG_V4_FUNDAMENTALS_CONFIG_HASH,
    LONG_V4_METHOD,
    LONG_V4_RESEARCH_STATUS,
    LONG_V4_SCORING_VERSION,
    LONG_V4_SEC_CIK_CONFIG_FILE_SHA256,
    LONG_V4_SEC_CIK_CONFIG_HASH,
    LONG_V4_SEC_CIK_CONFIG_VERSION,
    LONG_V4_SEC_MAPPING_SOURCE_SHA256,
    LONG_V4_VERSION,
    LongForecastV4Config,
    LongForecastV4MetricFamilyConfig,
    LongForecastV4PeerConfig,
    load_long_forecast_v4_config,
    load_long_v4_sec_cik_config,
    load_long_v4_sec_fundamentals_config,
    long_forecast_v4_config_hash,
    long_forecast_v4_config_path,
)
from stanstock.research.types import Scenario

PROBABILITY_REASON = (
    "Positive-return probability is unavailable for research-only us-sec-long-v4; "
    "activation requires separate cutoff-safe historical replay review"
)
CALCULATION_SCHEMA_VERSION = 2
EVIDENCE_CATALOG_SCHEMA_VERSION = 2
RETURN_IDENTITY_TOLERANCE = 1e-10
PRICE_QUANTUM = Decimal("0.000001")
V4_CONCEPTS = (
    "operating_cash_flow",
    "capital_expenditure",
    "net_income",
    "diluted_eps",
    "weighted_average_diluted_shares",
)
_FCF_SOURCE_CONCEPTS = frozenset(("operating_cash_flow", "capital_expenditure"))
_V4_METRIC_FAMILIES = frozenset(("fcf_per_share", "net_income_per_share"))
LONG_V4_SCORING_CONFIG_HASH = "43cc0ee0e29f79dec4ad8d8a91e43e7b3df1d368dbc385a116fcc6ff05f18e9b"
LONG_V4_PEER_POLICY = LongForecastV4PeerConfig(
    sic_prefix_levels=(4, 3, 2),
    minimum_cohort={4: 3, 3: 5, 2: 8},
)


@dataclass(frozen=True, slots=True)
class LongForecastV4:
    scenario: Scenario
    calculation: dict[str, Any]
    source_assets: tuple[DataAsset, ...]

    def scenario_payload(self) -> dict[str, Any]:
        return {
            **self.scenario.as_dict(),
            "schema_version": CALCULATION_SCHEMA_VERSION,
            "method_version": LONG_V4_VERSION,
            "research_status": LONG_V4_RESEARCH_STATUS,
            "metric_family": self.calculation.get("metric_family"),
            "support": self.calculation.get("support", {}),
            "annualized_return": self.calculation.get("selected_view", {}).get(
                "annualized_returns", {}
            ),
            "formula_inputs": self.calculation.get("formula_inputs", {}),
            "accounting_scope": self.calculation["accounting_scope"],
            "split_basis": self.calculation.get("split_basis", {}),
            "peer_set": self.calculation.get("peer_set", []),
            "return_basis": self.calculation["return_basis"],
            "probability_status": self.calculation["probability_semantics"]["status"],
            "confidence_semantics": self.calculation["confidence_semantics"],
            "insufficiency_code": self.calculation.get("insufficiency_code"),
        }


@dataclass(frozen=True, slots=True)
class _AnnualPoint:
    period_identity: str
    period_start: date
    period_end: date
    entity_value: float
    shares: float
    per_share: float
    entity_fact_ids: tuple[str, ...]
    share_fact_ids: tuple[str, ...]
    net_income_fact_ids: tuple[str, ...]
    eps_fact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MetricEvidence:
    family: str
    current_entity_metric: float
    current_shares: float
    current_per_share: float
    current_per_share_exact: Decimal
    current_price_valuation: float
    current_period_start: date
    current_period_end: date
    current_multiple_raw: float
    current_multiple_exact: Decimal
    current_multiple_capped: float
    annual_points: tuple[_AnnualPoint, ...]
    entity_growth_raw: tuple[float, ...]
    entity_growth_capped: tuple[float, ...]
    target_entity_growth: float
    share_changes_raw: tuple[float, ...]
    dilution_base: float
    share_checks: tuple[dict[str, Any], ...]
    selected_fact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _EntityAssessment:
    listing: Listing
    price: Decimal
    price_asset: DataAsset
    facts: tuple[FundamentalFact, ...]
    admitted_facts: tuple[FundamentalFact, ...]
    classification: CompanyClassificationObservation | None
    status: str
    insufficiency_code: str | None
    reason: str
    metric: _MetricEvidence | None
    selected_fact_ids: tuple[str, ...]
    assessed_fact_ids: tuple[str, ...]
    evidence_selection: dict[str, Any]
    split_basis: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CanonicalPrice:
    listing: Listing
    session_date: date
    valuation_value: Decimal
    ledger_value: Decimal
    normalized_asset: DataAsset
    raw_asset: DataAsset


@dataclass(frozen=True, slots=True)
class _SecMappingAuthority:
    asset: DataAsset
    cik_config: SecCikConfig
    cohort_rows: tuple[dict[str, Any], ...]

    @property
    def cik_by_listing_id(self) -> dict[str, str]:
        return {
            cast(str, item["listing_id"]): cast(str, item["authoritative_cik"])
            for item in self.cohort_rows
        }


@dataclass(frozen=True, slots=True)
class _SecRawAuthority:
    context_assets: dict[str, DataAsset]
    correction_events: dict[str, SourceObservationEvent | None]
    by_listing_id: dict[str, dict[str, Any]]
    latest_facts_by_listing_id: dict[str, tuple[FundamentalFact, ...]]


@dataclass(frozen=True, slots=True)
class _CompanyfactsVintage:
    source: DataAsset
    observed_at: datetime
    observation_event: SourceObservationEvent | None
    visible_events: tuple[SourceObservationEvent, ...]


@dataclass(frozen=True, slots=True)
class _ParsedCompanyfactsVintage:
    vintage: _CompanyfactsVintage
    context: DataAsset
    history_assets: tuple[DataAsset, ...]
    inspection: SecCompanyfactsInspection
    filing_assets: dict[UUID, DataAsset]


@dataclass(frozen=True, slots=True)
class _PeerLock:
    status: str
    target_classification: CompanyClassificationObservation
    examined_levels: tuple[dict[str, Any], ...]
    selected_level: int | None
    selected_prefix: str | None
    selected_floor: int | None
    members: tuple[Listing, ...]
    evidence_candidates: tuple[Listing, ...]

    @property
    def prefix_length(self) -> int | None:
        return self.selected_level

    @property
    def prefix(self) -> str | None:
        return self.selected_prefix

    @property
    def floor(self) -> int | None:
        return self.selected_floor

    def payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "target_classification_id": str(self.target_classification.pk),
            "target_sic": _normalized_sic(self.target_classification.code),
            "examined_levels": list(self.examined_levels),
            "selected_sic_prefix_level": self.selected_level,
            "selected_sic_prefix": self.selected_prefix,
            "selected_identity_floor": self.selected_floor,
            "selected_candidate_listing_ids": [str(listing.pk) for listing in self.members],
            "widening_after_assessment": False,
        }


def build_long_forecasts_v4(
    *,
    listings: list[Listing],
    current_prices: dict[str, float],
    price_assets: dict[str, DataAsset],
    asof: AsOfData,
    data_cutoff: datetime,
    target_date: date,
    config: LongForecastV4Config,
) -> dict[str, dict[str, LongForecastV4]]:
    """Build one frozen five-year result and expose exact year-3/year-5 views.

    The function reads provider-derived facts and classifications only through
    ``AsOfData``.  It never recursively forecasts peers: one non-recursive
    entity assessor is reused for the target and every member of the ex-ante
    locked cohort.
    """
    _validate_builder_contract(
        listings=listings,
        current_prices=current_prices,
        price_assets=price_assets,
        asof=asof,
        data_cutoff=data_cutoff,
        target_date=target_date,
        config=config,
    )
    ordered_listings = tuple(sorted(listings, key=lambda item: (item.ticker, str(item.pk))))
    cik_config = load_long_v4_sec_cik_config(config)
    mapping_authority = _resolve_sec_mapping_authority(
        listings=ordered_listings,
        config=config,
        cik_config=cik_config,
        asof=asof,
    )
    sec_config = load_long_v4_sec_fundamentals_config(config)
    canonical_prices = {
        str(listing.pk): _resolve_canonical_price(
            listing=listing,
            supplied_price=current_prices[str(listing.pk)],
            supplied_asset=price_assets[str(listing.pk)],
            asof=asof,
            target_date=target_date,
            expected_provider=config.price_provider,
        )
        for listing in ordered_listings
    }

    company_ids = tuple({listing.security.company_id for listing in ordered_listings})
    period_start = target_date - timedelta(days=6 * 366)
    source_facts = list(
        asof.fundamental_facts_for_companies(
            company_ids=company_ids,
            available_through=data_cutoff,
        )
        .filter(provider=config.fundamentals_provider)
        .select_related("source_asset")
    )
    facts_by_company: dict[str, list[FundamentalFact]] = {}
    for fact in source_facts:
        if fact.concept in V4_CONCEPTS and period_start <= fact.period_end <= target_date:
            facts_by_company.setdefault(str(fact.company_id), []).append(fact)

    classifications_by_company: dict[str, list[CompanyClassificationObservation]] = {}
    for observation in asof.company_classifications_for_companies(
        company_ids=company_ids,
        scheme="sec_sic",
        available_through=data_cutoff,
    ).filter(provider=config.fundamentals_provider):
        classifications_by_company.setdefault(str(observation.company_id), []).append(observation)

    filing_assets = _filing_assets_for(source_facts, decision_time=asof.decision_time)
    classification_by_listing = {
        str(listing.pk): _select_classification(
            classifications_by_company.get(str(listing.security.company_id), [])
        )
        for listing in ordered_listings
    }
    raw_authority = _validate_raw_sec_authority(
        listings=ordered_listings,
        facts=tuple(source_facts),
        filing_assets=filing_assets,
        classifications=classification_by_listing,
        config=sec_config,
        store=asof.store,
        data_cutoff=data_cutoff,
        decision_time=asof.decision_time,
        authoritative_ciks=mapping_authority.cik_by_listing_id,
        target_date=target_date,
    )
    latest_facts_by_company: dict[str, list[FundamentalFact]] = {}
    for listing in ordered_listings:
        latest_facts_by_company[str(listing.security.company_id)] = [
            fact
            for fact in raw_authority.latest_facts_by_listing_id[str(listing.pk)]
            if (fact.concept in V4_CONCEPTS and period_start <= fact.period_end <= target_date)
        ]
    listing_by_id = {str(listing.pk): listing for listing in ordered_listings}

    forecasts: dict[str, dict[str, LongForecastV4]] = {}
    for listing in ordered_listings:
        listing_id = str(listing.pk)
        price_identity = canonical_prices[listing_id]
        target = _assess_entity(
            listing=listing,
            price=price_identity.valuation_value,
            price_asset=price_identity.normalized_asset,
            facts=facts_by_company.get(str(listing.security.company_id), []),
            lineage_facts=latest_facts_by_company.get(
                str(listing.security.company_id),
                [],
            ),
            classification=classification_by_listing[listing_id],
            filing_assets=filing_assets,
            data_cutoff=data_cutoff,
            target_date=target_date,
            config=config,
            sec_config=sec_config,
            raw_fcf_authority=raw_authority.by_listing_id[listing_id],
        )
        if target.metric is None or target.classification is None:
            forecasts[listing_id] = _withheld_pair(
                target=target,
                peer_assessments=(),
                reason_code=target.insufficiency_code or "target_evidence_insufficient",
                reason=target.reason or "Target evidence is insufficient",
                target_date=target_date,
                config=config,
                filing_assets=filing_assets,
                peer_lock=None,
                classifications=classification_by_listing,
                cohort_prices=tuple(canonical_prices.values()),
                raw_authority=raw_authority,
                mapping_authority=mapping_authority,
                store=asof.store,
                decision_time=asof.decision_time,
            )
            continue

        peer_lock = _lock_peer_cohort(
            target_listing=target.listing,
            target_classification=target.classification,
            listings=tuple(listing_by_id.values()),
            classifications=classification_by_listing,
            authoritative_ciks=mapping_authority.cik_by_listing_id,
            peer_config=config.peer,
        )
        if peer_lock.status == "no_floor":
            forecasts[listing_id] = _withheld_pair(
                target=target,
                peer_assessments=(),
                reason_code="peer_identity_floor_unmet",
                reason="No cutoff-safe SIC identity cohort met the fixed 4/3/2 peer floors",
                target_date=target_date,
                config=config,
                filing_assets=filing_assets,
                peer_lock=peer_lock,
                classifications=classification_by_listing,
                cohort_prices=tuple(canonical_prices.values()),
                raw_authority=raw_authority,
                mapping_authority=mapping_authority,
                store=asof.store,
                decision_time=asof.decision_time,
            )
            continue

        peer_assessments = tuple(
            _assess_entity(
                listing=peer,
                price=canonical_prices[str(peer.pk)].valuation_value,
                price_asset=canonical_prices[str(peer.pk)].normalized_asset,
                facts=facts_by_company.get(str(peer.security.company_id), []),
                lineage_facts=latest_facts_by_company.get(
                    str(peer.security.company_id),
                    [],
                ),
                classification=classification_by_listing[str(peer.pk)],
                filing_assets=filing_assets,
                data_cutoff=data_cutoff,
                target_date=target_date,
                config=config,
                sec_config=sec_config,
                raw_fcf_authority=raw_authority.by_listing_id[str(peer.pk)],
            )
            for peer in peer_lock.members
        )
        admitted_peers = tuple(
            peer
            for peer in peer_assessments
            if peer.metric is not None and peer.metric.family == target.metric.family
        )
        assert peer_lock.floor is not None
        assert peer_lock.prefix_length is not None
        if len(admitted_peers) < peer_lock.floor:
            forecasts[listing_id] = _withheld_pair(
                target=target,
                peer_assessments=peer_assessments,
                reason_code="locked_peer_core_floor_unmet",
                reason=(
                    f"Locked SIC-{peer_lock.prefix_length} cohort admitted "
                    f"{len(admitted_peers)}/{peer_lock.floor} same-family peers; "
                    "the cohort is not widened after evidence assessment"
                ),
                target_date=target_date,
                config=config,
                filing_assets=filing_assets,
                peer_lock=peer_lock,
                classifications=classification_by_listing,
                cohort_prices=tuple(canonical_prices.values()),
                raw_authority=raw_authority,
                mapping_authority=mapping_authority,
                store=asof.store,
                decision_time=asof.decision_time,
            )
            continue
        forecasts[listing_id] = _successful_pair(
            target=target,
            locked=peer_lock,
            peer_assessments=peer_assessments,
            admitted_peers=admitted_peers,
            target_date=target_date,
            config=config,
            filing_assets=filing_assets,
            classifications=classification_by_listing,
            cohort_prices=tuple(canonical_prices.values()),
            raw_authority=raw_authority,
            mapping_authority=mapping_authority,
            store=asof.store,
            decision_time=asof.decision_time,
        )
    return forecasts


def _validate_builder_contract(
    *,
    listings: list[Listing],
    current_prices: dict[str, float],
    price_assets: dict[str, DataAsset],
    asof: AsOfData,
    data_cutoff: datetime,
    target_date: date,
    config: LongForecastV4Config,
) -> None:
    if config.version != LONG_V4_VERSION or config.method != LONG_V4_METHOD:
        raise ValueError("The long-v4 builder requires the exact schema-2 method identity")
    if long_forecast_v4_config_hash(config) != LONG_V4_EFFECTIVE_CONFIG_HASH:
        raise ValueError("The long-v4 builder requires the reviewed effective config hash")
    if config.peer != LONG_V4_PEER_POLICY:
        raise ValueError("The long-v4 builder requires the code-owned peer policy")
    if not listings:
        raise ValueError("Long-v4 requires a non-empty authoritative snapshot cohort")
    if data_cutoff > asof.decision_time:
        raise ValueError("Long-v4 data_cutoff cannot be after the as-of decision time")
    if target_date > data_cutoff.date():
        raise ValueError("Long-v4 target_date cannot be after data_cutoff")
    listing_ids = {str(listing.pk) for listing in listings}
    if set(current_prices) != listing_ids or set(price_assets) != listing_ids:
        raise ValueError("Long-v4 price inputs must exactly cover the snapshot cohort")
    if len(listing_ids) != len(listings):
        raise ValueError("Long-v4 snapshot cohort contains a duplicate listing identity")


def _resolve_sec_mapping_authority(
    *,
    listings: tuple[Listing, ...],
    config: LongForecastV4Config,
    cik_config: SecCikConfig,
    asof: AsOfData,
) -> _SecMappingAuthority:
    """Bind the cohort to one independently pinned SEC mapping snapshot."""
    try:
        if (
            config.sec_cik_config_version != cik_config.config_version
            or config.sec_cik_config_file_sha256 != LONG_V4_SEC_CIK_CONFIG_FILE_SHA256
            or config.sec_cik_config_hash != cik_config.config_hash
            or config.sec_mapping_source_sha256 != cik_config.source_sha256
            or cik_config.config_version != LONG_V4_SEC_CIK_CONFIG_VERSION
            or cik_config.config_hash != LONG_V4_SEC_CIK_CONFIG_HASH
            or cik_config.source_sha256 != LONG_V4_SEC_MAPPING_SOURCE_SHA256
            or cik_config.excluded
            or any(
                symbol != mapping.symbol
                or symbol != mapping.official_ticker
                or bool(mapping.reason)
                for symbol, mapping in cik_config.mappings.items()
            )
        ):
            raise ValueError("CIK config")
        candidates = list(
            DataAsset.objects.filter(
                provider="sec",
                kind=MAPPING_KIND,
                subject=MAPPING_SUBJECT,
                sha256=config.sec_mapping_source_sha256,
                available_at__lte=asof.decision_time,
                retrieved_at__lte=asof.decision_time,
            ).order_by("pk")
        )
        if len(candidates) != 1:
            raise ValueError("mapping asset")
        mapping_asset = candidates[0]
        rows = parse_sec_mapping(read_checksummed_bytes(asof.store, mapping_asset))
        rows_by_ticker: dict[str, list[Any]] = {}
        for row in rows:
            rows_by_ticker.setdefault(row.ticker, []).append(row)
        reviewed_rows: dict[str, Any] = {}
        for symbol, reviewed_mapping in cik_config.mappings.items():
            matches = rows_by_ticker.get(reviewed_mapping.official_ticker, [])
            if (
                len(matches) != 1
                or matches[0].ticker != reviewed_mapping.official_ticker
                or matches[0].cik != reviewed_mapping.cik
                or matches[0].exchange != reviewed_mapping.exchange
            ):
                raise ValueError("reviewed mapping row")
            verify_sec_exchange_mic(
                exchange=reviewed_mapping.exchange,
                mic=dict(SEC_EXCHANGE_MIC_RULE).get(reviewed_mapping.exchange),
            )
            reviewed_rows[symbol] = matches[0]

        cohort_rows: list[dict[str, Any]] = []
        for listing in listings:
            listing_mapping = cik_config.mappings.get(listing.ticker)
            raw_row = reviewed_rows.get(listing.ticker)
            company = listing.security.company
            if (
                listing_mapping is None
                or raw_row is None
                or listing.provider_symbol != listing.ticker
                or listing_mapping.symbol != listing.ticker
                or listing_mapping.official_ticker != listing.ticker
                or raw_row.ticker != listing.ticker
                or raw_row.cik != listing_mapping.cik
                or raw_row.exchange != listing_mapping.exchange
                or _strict_company_cik(company.cik) != listing_mapping.cik
            ):
                raise ValueError("cohort mapping")
            verify_sec_exchange_mic(
                exchange=listing_mapping.exchange,
                mic=listing.exchange_mic,
            )
            cohort_rows.append(
                {
                    "listing_id": str(listing.pk),
                    "company_id": str(company.pk),
                    "provider_symbol": listing.provider_symbol,
                    "listing_ticker": listing.ticker,
                    "listing_exchange_mic": listing.exchange_mic,
                    "authoritative_cik": listing_mapping.cik,
                    "config_symbol": listing_mapping.symbol,
                    "config_official_ticker": listing_mapping.official_ticker,
                    "config_exchange": listing_mapping.exchange,
                    "raw_ticker": raw_row.ticker,
                    "raw_cik": raw_row.cik,
                    "raw_exchange": raw_row.exchange,
                }
            )
        if len(cohort_rows) != len(listings):
            raise ValueError("cohort mapping cardinality")
        for identity_fields in (
            ("listing_id",),
            ("company_id",),
            ("authoritative_cik",),
            ("config_official_ticker",),
            ("raw_ticker", "raw_cik", "raw_exchange"),
        ):
            identities = {tuple(row[field] for field in identity_fields) for row in cohort_rows}
            if len(identities) != len(cohort_rows):
                raise ValueError("cohort mapping is not one-to-one")
    except (
        OSError,
        RefreshVerificationError,
        SecDerivationError,
        TypeError,
        ValueError,
    ):
        raise ValueError("Long-v4 SEC mapping authority is incompatible") from None
    return _SecMappingAuthority(
        asset=mapping_asset,
        cik_config=cik_config,
        cohort_rows=tuple(cohort_rows),
    )


def validate_long_v4_cohort_mapping_authority(
    *,
    listings: tuple[Listing, ...],
    config: LongForecastV4Config,
    asof: AsOfData,
) -> None:
    """Fail a service request before any price/fact read if issuer identity aliases."""
    ordered = tuple(sorted(listings, key=lambda item: (item.ticker, str(item.pk))))
    cik_config = load_long_v4_sec_cik_config(config)
    _resolve_sec_mapping_authority(
        listings=ordered,
        config=config,
        cik_config=cik_config,
        asof=asof,
    )


def _select_classification(
    observations: list[CompanyClassificationObservation],
) -> CompanyClassificationObservation | None:
    if not observations:
        return None
    latest_available = max(item.available_at for item in observations)
    latest = [item for item in observations if item.available_at == latest_available]
    if len(latest) != 1:
        return None
    return latest[0]


def _filing_assets_for(
    facts: list[FundamentalFact],
    *,
    decision_time: datetime,
) -> dict[str, DataAsset]:
    fact_ids = [fact.pk for fact in facts]
    result: dict[str, DataAsset] = {}
    for offset in range(0, len(fact_ids), 500):
        links = FundamentalFactEvidence.objects.filter(
            fact_id__in=fact_ids[offset : offset + 500],
            role=FundamentalFactEvidence.Role.FILING,
            source_asset__available_at__lte=decision_time,
            source_asset__retrieved_at__lte=decision_time,
        ).select_related("source_asset")
        for link in links:
            fact_id = str(link.fact_id)
            if fact_id in result:
                raise ValueError("Long-v4 SEC fact has ambiguous filing evidence")
            result[fact_id] = link.source_asset
    return result


def _resolve_canonical_price(
    *,
    listing: Listing,
    supplied_price: float | Decimal | str | None,
    supplied_asset: DataAsset | None,
    asof: AsOfData,
    target_date: date,
    expected_provider: str,
) -> _CanonicalPrice:
    """Resolve one exact listing close and its normalized/raw physical closure."""
    subject = listing.provider_symbol
    if (
        not isinstance(subject, str)
        or not subject
        or subject != subject.strip()
        or not listing.exchange_mic
        or not listing.currency
    ):
        raise ValueError("Long-v4 price listing identity is incomplete")
    candidates = list(
        DataAsset.objects.filter(
            provider=expected_provider,
            kind="price_history",
            subject=subject,
            available_at__lte=asof.decision_time,
            retrieved_at__lte=asof.decision_time,
        ).order_by("-available_at", "-retrieved_at", "pk")
    )
    if not candidates:
        raise ValueError("Long-v4 canonical normalized price asset is unavailable") from None
    maximum = (candidates[0].available_at, candidates[0].retrieved_at)
    maxima = [asset for asset in candidates if (asset.available_at, asset.retrieved_at) == maximum]
    if len(maxima) != 1:
        raise ValueError("Long-v4 canonical normalized price asset is ambiguous")
    normalized = maxima[0]
    if supplied_asset is not None and _asset_authority_tuple(supplied_asset) != (
        _asset_authority_tuple(normalized)
    ):
        raise ValueError("Long-v4 supplied price asset is not the canonical as-of asset")
    metadata = normalized.metadata if isinstance(normalized.metadata, dict) else {}
    resolved_mic = metadata.get("resolved_mic_code") or metadata.get("mic_code")
    if (
        normalized.provider != expected_provider
        or normalized.kind != "price_history"
        or normalized.subject != subject
        or metadata.get("currency") != listing.currency
        or resolved_mic != listing.exchange_mic
        or metadata.get("return_definition") != "split_adjusted_price_return"
        or metadata.get("dividends_included") is not False
        or normalized.available_at > asof.decision_time
        or normalized.retrieved_at > asof.decision_time
    ):
        raise ValueError("Long-v4 canonical normalized price identity is incompatible")
    try:
        read = asof.price_frame_for_asset_with_diagnostics(
            asset=normalized,
            through_date=target_date,
        )
        rows = read.frame.filter(pl.col("date") == target_date)
        if read.invalid_session_date_rows or rows.height != 1 or "close" not in rows.columns:
            raise ValueError("target row")
        raw_physical_value = rows.row(0, named=True)["close"]
        if (
            isinstance(raw_physical_value, bool)
            or not isinstance(raw_physical_value, float)
            or not math.isfinite(raw_physical_value)
            or raw_physical_value <= 0
        ):
            raise ValueError("target close")
        valuation_value = Decimal(str(raw_physical_value))
        if not valuation_value.is_finite() or valuation_value <= 0:
            raise ValueError("target close")
        ledger_value = canonical_long_v4_price(valuation_value)
        raw = raw_price_asset_for(normalized, cutoff=asof.decision_time)
        read_checksummed_bytes(asof.store, raw)
    except (
        InvalidOperation,
        OSError,
        PriceFrameChecksumMismatchError,
        RefreshVerificationError,
        TypeError,
        ValueError,
        pl.exceptions.PolarsError,
    ):
        raise ValueError("Long-v4 canonical price closure could not be verified") from None
    if supplied_price is not None and _exact_positive_decimal(supplied_price) != valuation_value:
        raise ValueError("Long-v4 supplied price does not match the canonical target-date close")
    return _CanonicalPrice(
        listing=listing,
        session_date=target_date,
        valuation_value=valuation_value,
        ledger_value=ledger_value,
        normalized_asset=normalized,
        raw_asset=raw,
    )


def canonical_long_v4_price(value: object) -> Decimal:
    """Return the one six-decimal price representation owned by long-v4."""
    if isinstance(value, bool):
        raise ValueError("Long-v4 price must be a finite positive number")
    try:
        numeric = float(cast(Any, value))
    except (OverflowError, TypeError, ValueError):
        raise ValueError("Long-v4 price must be a finite positive number") from None
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError("Long-v4 price must be a finite positive number")
    try:
        normalized = Decimal(str(round(numeric, 6))).quantize(PRICE_QUANTUM)
    except InvalidOperation:
        raise ValueError("Long-v4 price could not be represented at six decimals") from None
    if not normalized.is_finite() or normalized <= 0:
        raise ValueError("Long-v4 price must remain positive at six decimals")
    return normalized


def _exact_positive_decimal(value: object) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("Long-v4 valuation price must be a finite positive number")
    try:
        normalized = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("Long-v4 valuation price must be a finite positive number") from None
    if not normalized.is_finite() or normalized <= 0:
        raise ValueError("Long-v4 valuation price must be a finite positive number")
    return normalized


def _asset_authority_tuple(asset: DataAsset) -> tuple[Any, ...]:
    return (
        asset.pk,
        asset.provider,
        asset.kind,
        asset.subject,
        asset.relative_path,
        asset.sha256,
        asset.retrieved_at,
        asset.available_at,
        asset.period_start,
        asset.period_end,
        asset.schema_version,
        asset.metadata,
    )


def _validate_raw_sec_authority(
    *,
    listings: tuple[Listing, ...],
    facts: tuple[FundamentalFact, ...],
    filing_assets: dict[str, DataAsset],
    classifications: dict[str, CompanyClassificationObservation | None],
    config: Any,
    store: AssetStore,
    data_cutoff: datetime,
    decision_time: datetime,
    authoritative_ciks: dict[str, str],
    target_date: date,
) -> _SecRawAuthority:
    """Prove every prospective catalog row from its exact raw SEC bytes.

    All lower-level parser, checksum, and filesystem failures are normalized
    to one stable path-free surface.  A malformed persisted row is an
    operation-wide integrity failure; it is never converted into a
    forecast-shaped withholding result.
    """
    try:
        return _derive_raw_sec_authority(
            listings=listings,
            facts=facts,
            filing_assets=filing_assets,
            classifications=classifications,
            config=config,
            store=store,
            data_cutoff=data_cutoff,
            decision_time=decision_time,
            authoritative_ciks=authoritative_ciks,
            target_date=target_date,
        )
    except (
        DataAsset.DoesNotExist,
        InvalidOperation,
        OSError,
        ObservationEvidenceError,
        RefreshVerificationError,
        SecDerivationError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        raise ValueError("Long-v4 SEC raw evidence authority is incompatible") from None


def _derive_raw_sec_authority(
    *,
    listings: tuple[Listing, ...],
    facts: tuple[FundamentalFact, ...],
    filing_assets: dict[str, DataAsset],
    classifications: dict[str, CompanyClassificationObservation | None],
    config: Any,
    store: AssetStore,
    data_cutoff: datetime,
    decision_time: datetime,
    authoritative_ciks: dict[str, str],
    target_date: date,
) -> _SecRawAuthority:
    company_ids = {listing.security.company_id for listing in listings}
    resolved_companies = Company.objects.in_bulk(company_ids)
    if set(resolved_companies) != company_ids:
        raise ValueError("company authority")
    owners = {str(company_id): company for company_id, company in resolved_companies.items()}
    if any(
        listing.security.company_id not in resolved_companies
        or listing.security.company.cik != resolved_companies[listing.security.company_id].cik
        or authoritative_ciks.get(str(listing.pk))
        != resolved_companies[listing.security.company_id].cik
        for listing in listings
    ):
        raise ValueError("company authority changed")
    contexts: dict[str, DataAsset] = {}
    events: dict[str, SourceObservationEvent | None] = {}
    by_listing_id: dict[str, dict[str, Any]] = {}
    latest_facts_by_listing_id: dict[str, tuple[FundamentalFact, ...]] = {}
    facts_by_company: dict[str, list[FundamentalFact]] = {}
    for fact in facts:
        if str(fact.company_id) not in owners:
            raise ValueError("fact owner")
        facts_by_company.setdefault(str(fact.company_id), []).append(fact)

    window_start = target_date - timedelta(days=6 * 366)
    for listing in listings:
        listing_id = str(listing.pk)
        owner = owners[str(listing.security.company_id)]
        cik = authoritative_ciks[listing_id]
        owner_facts = facts_by_company.get(str(owner.pk), [])
        vintages = _companyfacts_vintage_history(
            cik=cik,
            decision_time=decision_time,
        )
        if not vintages:
            raise ValueError("companyfacts authority is unavailable")
        vintage_source_ids = {vintage.source.pk for vintage in vintages}
        if any(fact.source_asset_id not in vintage_source_ids for fact in owner_facts):
            raise ValueError("companyfacts fact source is outside the immutable history")
        raw_entries: list[dict[str, Any]] = []
        source_entries: list[dict[str, Any]] = []
        parsed_vintages: list[_ParsedCompanyfactsVintage] = []
        for vintage in vintages:
            source = vintage.source
            _validate_raw_asset_boundary(
                source,
                provider="sec",
                kind=COMPANYFACTS_KIND,
                subject=cik,
                decision_time=decision_time,
            )
            companyfacts_payload = read_checksummed_bytes(store, source)
            context = _companyfacts_context_asset(
                source=source,
                cik=cik,
                decision_time=decision_time,
            )
            current = derive_sec_current_submissions(
                read_checksummed_bytes(store, context),
                source_asset=context,
                expected_cik=cik,
            )
            history_assets = tuple(
                _history_context_asset(
                    cik=cik,
                    filename=filename,
                    decision_time=decision_time,
                )
                for filename in sorted(current.historical_filenames)
            )
            records = list(current.filings)
            for history in history_assets:
                filename = cast(str, history.metadata[HISTORY_FILENAME_METADATA_KEY])
                records.extend(
                    derive_sec_historical_submissions(
                        read_checksummed_bytes(store, history),
                        source_asset=history,
                        expected_cik=cik,
                        filename=filename,
                        allowed_filenames=current.historical_filenames,
                    )
                )
            inspection = inspect_sec_companyfacts(
                companyfacts_payload,
                source_asset=source,
                expected_cik=cik,
                filing_records=tuple(records),
                config=config,
            )
            derivation_filing_assets = {
                record.source_asset.pk: record.source_asset for record in records
            }
            if source.pk is not None:
                derivation_filing_assets[source.pk] = source
            parsed_vintages.append(
                _ParsedCompanyfactsVintage(
                    vintage=vintage,
                    context=context,
                    history_assets=history_assets,
                    inspection=inspection,
                    filing_assets=derivation_filing_assets,
                )
            )

        observation_states: dict[
            tuple[int, int],
            tuple[bool, datetime, str, SourceObservationEvent | None],
        ] = {}
        reconciled_lineages = reconcile_sec_observation_lineages(
            tuple(parsed.inspection for parsed in parsed_vintages)
        )
        lineage_by_derivation: dict[int, RawObservationLineage] = {}
        for parsed, lineages in zip(parsed_vintages, reconciled_lineages, strict=True):
            for observation, lineage in zip(
                parsed.inspection.configured_observations,
                lineages,
                strict=True,
            ):
                if observation.derivation is not None:
                    if lineage is None:
                        raise ValueError("derived SEC observation has no raw lineage")
                    lineage_by_derivation[id(observation.derivation)] = lineage
        source_has_future_correction: dict[int, bool] = {}
        source_has_new_cutoff_raw_evidence: dict[int, bool] = {}
        source_has_cutoff_non_fcf_rejection: dict[int, bool] = {}
        latest_cutoff_observations: dict[
            RawObservationLineage,
            SecConfiguredObservation,
        ] = {}
        latest_observation_states: dict[
            tuple[object, ...],
            tuple[
                tuple[object, ...],
                bool,
                datetime,
                str,
                SourceObservationEvent | None,
            ],
        ] = {}
        for parsed_index, (parsed, lineages) in enumerate(
            zip(parsed_vintages, reconciled_lineages, strict=True)
        ):
            source = parsed.vintage.source
            source_states: dict[
                tuple[object, ...],
                tuple[
                    tuple[object, ...],
                    bool,
                    datetime,
                    str,
                    SourceObservationEvent | None,
                ],
            ] = {}
            has_future_correction = False
            has_new_cutoff_raw_evidence = False
            has_cutoff_non_fcf_rejection = False
            cutoff_observations: list[tuple[RawObservationLineage, SecConfiguredObservation]] = []
            for index, (observation, lineage) in enumerate(
                zip(
                    parsed.inspection.configured_observations,
                    lineages,
                    strict=True,
                )
            ):
                identity = (
                    lineage
                    if lineage is not None
                    else _raw_configured_observation_identity(observation)
                )
                signature = _raw_configured_observation_signature(observation)
                previous_state = latest_observation_states.get(identity)
                changed_now = previous_state is not None and signature != previous_state[0]
                if previous_state is not None and not changed_now:
                    (
                        _previous_signature,
                        is_correction,
                        observation_available_at,
                        observation_availability_basis,
                        correction_event,
                    ) = previous_state
                else:
                    observation_available_at, observation_availability_basis = (
                        _raw_configured_observation_availability(
                            observation,
                            vintage=parsed.vintage,
                            is_correction=changed_now,
                        )
                    )
                    is_correction = changed_now
                    correction_event = parsed.vintage.observation_event if changed_now else None
                state = (
                    signature,
                    is_correction,
                    observation_available_at,
                    observation_availability_basis,
                    correction_event,
                )
                source_states[identity] = state
                observation_states[(parsed_index, index)] = (
                    is_correction,
                    observation_available_at,
                    observation_availability_basis,
                    correction_event,
                )
                if lineage is not None and observation_available_at <= data_cutoff:
                    cutoff_observations.append((lineage, observation))
                if is_correction and observation_available_at > data_cutoff:
                    has_future_correction = True
                if (
                    observation_available_at <= data_cutoff
                    and observation.status == "rejected"
                    and observation.concept not in _FCF_SOURCE_CONCEPTS
                    and observation.unit_supported is not False
                ):
                    has_cutoff_non_fcf_rejection = True
                is_new_observation_state = previous_state is None or changed_now
                if is_new_observation_state and _raw_fcf_observation_is_relevant(
                    observation,
                    window_start=window_start,
                    target_date=target_date,
                    data_cutoff=data_cutoff,
                    observation_available_at=observation_available_at,
                ):
                    has_new_cutoff_raw_evidence = True
            source_has_future_correction[parsed_index] = has_future_correction
            source_has_new_cutoff_raw_evidence[parsed_index] = has_new_cutoff_raw_evidence
            source_has_cutoff_non_fcf_rejection[parsed_index] = has_cutoff_non_fcf_rejection
            if not has_future_correction:
                latest_cutoff_observations.update(cutoff_observations)
            latest_observation_states.update(source_states)

        if any(
            source_has_cutoff_non_fcf_rejection[index] and not source_has_future_correction[index]
            for index, _parsed in enumerate(parsed_vintages)
        ):
            raise SecDerivationError("A configured non-FCF Companyfacts observation was malformed")

        fact_derivations: dict[str, SecFactDerivation] = {}
        fact_raw_signatures: dict[str, str] = {}
        fact_lineages: dict[str, RawObservationLineage] = {}
        fact_contexts: dict[str, DataAsset] = {}
        lineage_chains: dict[RawObservationLineage, list[FundamentalFact]] = {}
        for fact in owner_facts:
            filing = filing_assets.get(str(fact.pk))
            if filing is None:
                raise ValueError("filing evidence missing")
            matches: list[tuple[int, SecFactDerivation, RawObservationLineage, DataAsset]] = []
            for parsed_index, parsed in enumerate(parsed_vintages):
                source = parsed.vintage.source
                if source.pk != fact.source_asset_id:
                    continue
                for derivation in parsed.inspection.derivations:
                    lineage = lineage_by_derivation.get(id(derivation))
                    if lineage is not None and _fact_matches_derivation(
                        fact,
                        derivation=derivation,
                        filing=filing,
                    ):
                        matches.append(
                            (
                                parsed_index,
                                derivation,
                                lineage,
                                parsed.context,
                            )
                        )
            identities = {
                (lineage, derivation.raw_observation_signature)
                for _index, derivation, lineage, _context in matches
            }
            if len(identities) != 1:
                raise ValueError("fact raw derivation")
            eligible_matches = [
                match for match in matches if not source_has_future_correction[match[0]]
            ]
            if not eligible_matches:
                raise ValueError("post-cutoff correction source backs a cutoff-visible fact")
            _index, derivation, lineage, context = eligible_matches[0]
            fact_derivations[str(fact.pk)] = derivation
            fact_raw_signatures[str(fact.pk)] = derivation.raw_observation_signature
            fact_lineages[str(fact.pk)] = lineage
            fact_contexts[str(fact.pk)] = context
            lineage_chains.setdefault(lineage, []).append(fact)

        selected_facts: list[FundamentalFact] = []
        for chain in lineage_chains.values():
            ordered_chain = sorted(chain, key=lambda item: item.source_revision)
            if [item.source_revision for item in ordered_chain] != list(
                range(1, len(ordered_chain) + 1)
            ):
                raise ValueError("raw-observation lineage revision chain is ambiguous")
            lineage = fact_lineages[str(ordered_chain[-1].pk)]
            latest_observation = latest_cutoff_observations.get(lineage)
            if latest_observation is None:
                raise ValueError("raw-observation lineage has no cutoff-visible state")
            matching_latest = [
                fact
                for fact in ordered_chain
                if fact_raw_signatures[str(fact.pk)] == latest_observation.raw_observation_signature
            ]
            if matching_latest:
                selected_facts.append(max(matching_latest, key=lambda fact: fact.source_revision))
        latest_facts_by_listing_id[listing_id] = tuple(
            sorted(
                selected_facts,
                key=lambda fact: (
                    fact.concept,
                    fact.period_end,
                    fact.period_start or fact.period_end,
                    fact.source_revision,
                    fact.observation_hash,
                ),
            )
        )

        for fact in owner_facts:
            derivation = fact_derivations[str(fact.pk)]
            lineage = fact_lineages[str(fact.pk)]
            if fact.available_at > data_cutoff or fact.ingested_at > decision_time:
                raise ValueError("fact timing")
            event = _validate_fact_availability_authority(
                fact,
                derivation=derivation,
                lineage_chain=tuple(lineage_chains[lineage]),
                cik=cik,
                data_cutoff=data_cutoff,
                decision_time=decision_time,
            )
            contexts[str(fact.pk)] = fact_contexts[str(fact.pk)]
            events[str(fact.pk)] = event

        for parsed_index, parsed in enumerate(parsed_vintages):
            vintage = parsed.vintage
            source = vintage.source
            source_facts = [fact for fact in owner_facts if fact.source_asset_id == source.pk]

            include_source = bool(
                not source_has_future_correction[parsed_index]
                and (
                    source.available_at <= data_cutoff
                    or source.retrieved_at <= data_cutoff
                    or source_facts
                    or source_has_new_cutoff_raw_evidence[parsed_index]
                )
            )
            if not include_source:
                continue
            visible_events = tuple(
                event for event in vintage.visible_events if event.observed_at <= data_cutoff
            )
            if not visible_events and source.retrieved_at > data_cutoff and vintage.visible_events:
                # Research reconstruction may use a later retrieval to prove
                # an original/unchanged observation whose filing boundary is
                # cutoff-eligible. Retain the actual retrieval timestamp; it
                # must never be relabeled as a historical correction event.
                visible_events = (vintage.visible_events[0],)
            source_entries.append(
                {
                    "companyfacts_asset": _asset_payload(source),
                    "current_submissions_asset": _asset_payload(parsed.context),
                    "history_assets": [_asset_payload(asset) for asset in parsed.history_assets],
                    "latest_visible_observation_event": (
                        _observation_event_payload(visible_events[-1]) if visible_events else None
                    ),
                    "visible_observation_events": [
                        _observation_event_payload(event) for event in visible_events
                    ],
                }
            )
            for index, observation in enumerate(parsed.inspection.configured_observations):
                (
                    is_correction,
                    observation_available_at,
                    observation_availability_basis,
                    correction_event,
                ) = observation_states[(parsed_index, index)]
                if not _raw_fcf_observation_is_relevant(
                    observation,
                    window_start=window_start,
                    target_date=target_date,
                    data_cutoff=data_cutoff,
                    observation_available_at=observation_available_at,
                ):
                    continue
                raw_entries.append(
                    _raw_fcf_observation_payload(
                        observation,
                        source=source,
                        filing_assets=parsed.filing_assets,
                        owner_facts=owner_facts,
                        data_cutoff=data_cutoff,
                        decision_time=decision_time,
                        observation_available_at=observation_available_at,
                        observation_availability_basis=observation_availability_basis,
                        is_correction=is_correction,
                        correction_event=correction_event,
                    )
                )
        status = (
            "absent"
            if not raw_entries
            else (
                "present_complete"
                if all(item["status"] == "normalized" for item in raw_entries)
                else "present_normalization_incomplete"
            )
        )
        by_listing_id[listing_id] = {
            "owner_listing_id": listing_id,
            "company_id": str(owner.pk),
            "authoritative_cik": cik,
            "window_start": window_start.isoformat(),
            "window_end": target_date.isoformat(),
            "data_cutoff": data_cutoff.isoformat(),
            "decision_time": decision_time.isoformat(),
            "sources": source_entries,
            "derivations": raw_entries,
            "status": status,
        }

    for listing in listings:
        classification = classifications.get(str(listing.pk))
        if classification is None:
            continue
        owner = listing.security.company
        cik = _strict_company_cik(owner.cik)
        source = classification.source_asset
        _validate_raw_asset_boundary(
            source,
            provider="sec",
            kind=SUBMISSIONS_KIND,
            subject=cik,
            decision_time=decision_time,
        )
        evidence = derive_sec_current_submissions(
            read_checksummed_bytes(store, source),
            source_asset=source,
            expected_cik=cik,
        )
        if (
            classification.company_id != owner.pk
            or classification.provider != "sec"
            or classification.scheme != "sec_sic"
            or classification.code != evidence.sic
            or classification.description != evidence.sic_description
            or source.retrieved_at > classification.observed_at
            or classification.observed_at != classification.available_at
            or classification.available_at > data_cutoff
            or classification.ingested_at > decision_time
            or classification.quality_flags != ["current_snapshot_not_historical"]
        ):
            raise ValueError("classification raw derivation")
    return _SecRawAuthority(
        context_assets=contexts,
        correction_events=events,
        by_listing_id=by_listing_id,
        latest_facts_by_listing_id=latest_facts_by_listing_id,
    )


def _raw_fcf_observation_is_relevant(
    observation: SecConfiguredObservation,
    *,
    window_start: date,
    target_date: date,
    data_cutoff: datetime,
    observation_available_at: datetime | None = None,
) -> bool:
    if (
        observation.concept not in _FCF_SOURCE_CONCEPTS
        or observation.period_type != FundamentalFact.PeriodType.DURATION
    ):
        return False
    if observation.period_end is not None and not (
        window_start <= observation.period_end <= target_date
    ):
        return False
    availability = observation_available_at or observation.acceptance_at
    if availability is not None and availability > data_cutoff:
        return False
    if observation.status == "excluded" and observation.rejection_code == "filing_form_not_allowed":
        return False
    return True


def _raw_fcf_observation_payload(
    observation: SecConfiguredObservation,
    *,
    source: DataAsset,
    filing_assets: dict[UUID, DataAsset],
    owner_facts: list[FundamentalFact],
    data_cutoff: datetime,
    decision_time: datetime,
    observation_available_at: datetime,
    observation_availability_basis: str,
    is_correction: bool,
    correction_event: SourceObservationEvent | None,
) -> dict[str, Any]:
    derivation = observation.derivation
    filing = (
        filing_assets.get(observation.filing_source_asset_id)
        if observation.filing_source_asset_id is not None
        else None
    )
    matching_facts: list[FundamentalFact] = []
    if derivation is not None:
        if filing is None:
            raise ValueError("raw FCF filing context")
        matching_facts = [
            fact
            for fact in owner_facts
            if _raw_derivation_matches_normalized_fact(
                fact,
                derivation=derivation,
                filing=filing,
                data_cutoff=data_cutoff,
                decision_time=decision_time,
            )
        ]
    ordered_matches = sorted(
        matching_facts,
        key=lambda fact: (
            fact.available_at,
            fact.source_revision,
            str(fact.pk),
        ),
    )
    status = "normalized" if ordered_matches else "normalization_incomplete"
    issue = (
        None
        if status == "normalized"
        else observation.rejection_code or "normalized_fact_match_missing"
    )
    return {
        "concept": observation.concept,
        "taxonomy": observation.taxonomy,
        "source_concept": observation.source_concept,
        "value_decimal": (_decimal_text(derivation.value) if derivation is not None else None),
        "unit": observation.unit,
        "period_type": observation.period_type,
        "period_start": (
            observation.period_start.isoformat() if observation.period_start is not None else None
        ),
        "period_end": (
            observation.period_end.isoformat() if observation.period_end is not None else None
        ),
        "accession": observation.accession,
        "raw_filing_form": observation.raw_filing_form,
        "filing_form": observation.filing_form,
        "filing_date": (
            observation.filing_date.isoformat() if observation.filing_date is not None else None
        ),
        "acceptance_at": (
            observation.acceptance_at.isoformat() if observation.acceptance_at is not None else None
        ),
        "availability_basis": observation.filing_availability_basis,
        "raw_observation_available_at": observation_available_at.isoformat(),
        "raw_observation_availability_basis": observation_availability_basis,
        "is_same_accession_correction": is_correction,
        "correction_observation_event": (
            _observation_event_payload(correction_event) if correction_event is not None else None
        ),
        "observation_hash": derivation.observation_hash if derivation is not None else None,
        "companyfacts_asset_id": str(source.pk),
        "companyfacts_asset_sha256": source.sha256,
        "filing_asset_id": str(filing.pk) if filing is not None else None,
        "filing_asset_sha256": filing.sha256 if filing is not None else None,
        "matching_normalized_fact_ids": [str(fact.pk) for fact in ordered_matches],
        "normalization_issue": issue,
        "status": status,
    }


def _companyfacts_vintage_history(
    *,
    cik: str,
    decision_time: datetime,
) -> tuple[_CompanyfactsVintage, ...]:
    sources = list(
        DataAsset.objects.filter(
            provider="sec",
            kind=COMPANYFACTS_KIND,
            subject=cik,
            available_at__lte=decision_time,
            retrieved_at__lte=decision_time,
        ).order_by("retrieved_at", "available_at", "pk")
    )
    if not sources:
        return ()
    source_by_id = {source.pk: source for source in sources}
    events = tuple(
        SourceObservationEvent.objects.filter(
            provider="sec",
            kind=COMPANYFACTS_KIND,
            subject=cik,
            observed_at__lte=decision_time,
            recorded_at__lte=decision_time,
            source_asset__available_at__lte=decision_time,
            source_asset__retrieved_at__lte=decision_time,
        )
        .select_related("source_asset")
        .order_by("observed_at", "recorded_at", "pk")
    )
    events_by_source: dict[UUID, list[SourceObservationEvent]] = {}
    for event in events:
        validate_sec_observation_event(event)
        source = source_by_id.get(event.source_asset_id)
        if (
            source is None
            or event.source_asset_id != source.pk
            or event.source_asset.provider != source.provider
            or event.source_asset.kind != source.kind
            or event.source_asset.subject != source.subject
            or event.observed_at < source.retrieved_at
            or event.observed_at > event.recorded_at
        ):
            raise ValueError("companyfacts observation history")
        events_by_source.setdefault(event.source_asset_id, []).append(event)
    vintages = tuple(
        [
            _CompanyfactsVintage(
                source=source,
                observed_at=event.observed_at,
                observation_event=event,
                visible_events=(event,),
            )
            for event in events
            for source in (source_by_id[event.source_asset_id],)
        ]
        + [
            _CompanyfactsVintage(
                source=source,
                observed_at=source.retrieved_at,
                observation_event=None,
                visible_events=(),
            )
            for source in sources
            if source.pk not in events_by_source
        ]
    )
    ordered = tuple(
        sorted(
            vintages,
            key=lambda item: (
                item.observed_at,
                item.source.retrieved_at,
                item.source.available_at,
                str(item.source.pk),
            ),
        )
    )
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if (
            previous.observed_at == current.observed_at
            and previous.source.sha256 != current.source.sha256
        ):
            raise ValueError("ambiguous companyfacts observation history")
    return ordered


def _raw_configured_observation_identity(
    observation: SecConfiguredObservation,
) -> tuple[object, ...]:
    if observation.raw_observation_lineage is not None:
        return observation.raw_observation_lineage
    return (
        observation.taxonomy,
        observation.source_concept,
        observation.accession,
        observation.unit,
        observation.period_start,
        observation.period_end,
    )


def _raw_configured_observation_signature(
    observation: SecConfiguredObservation,
) -> tuple[object, ...]:
    derivation = observation.derivation
    return (
        observation.raw_observation_signature,
        observation.unit,
        observation.unit_supported,
        observation.status,
        observation.rejection_code,
        observation.raw_filing_form,
        observation.period_type,
        observation.period_start,
        observation.period_end,
        _decimal_text(derivation.value) if derivation is not None else None,
        derivation.fiscal_year if derivation is not None else None,
        derivation.fiscal_period if derivation is not None else None,
        derivation.frame if derivation is not None else None,
    )


def _raw_configured_observation_availability(
    observation: SecConfiguredObservation,
    *,
    vintage: _CompanyfactsVintage,
    is_correction: bool,
) -> tuple[datetime, str]:
    if not is_correction and observation.acceptance_at is not None:
        return (
            observation.acceptance_at,
            observation.filing_availability_basis or "reconciled_filing",
        )
    boundary = max(vintage.source.retrieved_at, vintage.observed_at)
    if observation.acceptance_at is not None:
        boundary = max(boundary, observation.acceptance_at)
    return (
        boundary,
        (
            "companyfacts_correction_observation_event"
            if is_correction and vintage.observation_event is not None
            else (
                "companyfacts_correction_asset_retrieval"
                if is_correction
                else "companyfacts_asset_retrieval"
            )
        ),
    )


def _companyfacts_context_asset(
    *,
    source: DataAsset,
    cik: str,
    decision_time: datetime,
) -> DataAsset:
    metadata = source.metadata if isinstance(source.metadata, dict) else {}
    raw_id = metadata.get("submissions_asset_id")
    raw_sha256 = metadata.get("submissions_sha256")
    if (
        not isinstance(raw_id, str)
        or not isinstance(raw_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", raw_sha256) is None
    ):
        raise ValueError("companyfacts context identity")
    context_id = UUID(raw_id)
    if str(context_id) != raw_id:
        raise ValueError("companyfacts context UUID")
    context = DataAsset.objects.filter(pk=context_id, sha256=raw_sha256).first()
    if context is None:
        raise ValueError("companyfacts context missing")
    _validate_raw_asset_boundary(
        context,
        provider="sec",
        kind=SUBMISSIONS_KIND,
        subject=cik,
        decision_time=decision_time,
    )
    return context


def _history_context_asset(
    *,
    cik: str,
    filename: str,
    decision_time: datetime,
) -> DataAsset:
    candidates = list(
        DataAsset.objects.filter(
            provider="sec",
            kind=SUBMISSIONS_HISTORY_KIND,
            subject=cik,
            metadata__filename=filename,
            available_at__lte=decision_time,
            retrieved_at__lte=decision_time,
        ).order_by("pk")
    )
    if len(candidates) != 1:
        raise ValueError("historical filing context is missing or ambiguous")
    asset = candidates[0]
    _validate_raw_asset_boundary(
        asset,
        provider="sec",
        kind=SUBMISSIONS_HISTORY_KIND,
        subject=cik,
        decision_time=decision_time,
    )
    return asset


def _raw_derivation_matches_normalized_fact(
    fact: FundamentalFact,
    *,
    derivation: SecFactDerivation,
    filing: DataAsset,
    data_cutoff: datetime,
    decision_time: datetime,
) -> bool:
    if (
        fact.available_at > data_cutoff
        or fact.ingested_at > decision_time
        or not _fact_matches_derivation(fact, derivation=derivation, filing=filing)
    ):
        return False
    return True


def _observation_event_payload(event: SourceObservationEvent) -> dict[str, Any]:
    return {
        "id": str(event.pk),
        "provider": event.provider,
        "kind": event.kind,
        "subject": event.subject,
        "source_asset_id": str(event.source_asset_id),
        "content_sha256": event.content_sha256,
        "observed_at": event.observed_at.isoformat(),
        "recorded_at": event.recorded_at.isoformat(),
    }


def _strict_company_cik(raw: object) -> str:
    if not isinstance(raw, str) or re.fullmatch(r"[0-9]{10}", raw) is None or raw == "0000000000":
        raise ValueError("company CIK")
    return raw


def _validate_raw_asset_boundary(
    asset: DataAsset,
    *,
    provider: str,
    kind: str,
    subject: str,
    decision_time: datetime,
) -> None:
    timestamps = (asset.available_at, asset.retrieved_at, decision_time)
    if (
        asset.pk is None
        or asset.provider != provider
        or asset.kind != kind
        or asset.subject != subject
        or any(value.tzinfo is None or value.utcoffset() is None for value in timestamps)
        or asset.available_at > asset.retrieved_at
        or asset.available_at > decision_time
        or asset.retrieved_at > decision_time
    ):
        raise ValueError("raw asset boundary")


def _fact_matches_derivation(
    fact: FundamentalFact,
    *,
    derivation: SecFactDerivation,
    filing: DataAsset,
) -> bool:
    return (
        fact.concept == derivation.concept
        and fact.taxonomy == derivation.taxonomy
        and fact.source_concept == derivation.source_concept
        and fact.value == derivation.value
        and fact.unit == derivation.unit
        and fact.currency == derivation.currency
        and fact.period_type == derivation.period_type
        and fact.period_identity == derivation.period_identity
        and fact.period_start == derivation.period_start
        and fact.period_end == derivation.period_end
        and fact.fiscal_year == derivation.fiscal_year
        and fact.fiscal_period == derivation.fiscal_period
        and fact.frame == derivation.frame
        and fact.accession == derivation.accession
        and fact.filing_form == derivation.filing_form
        and fact.filing_date == derivation.filing_date
        and fact.filed_at == derivation.acceptance_at
        and fact.acceptance_at == derivation.acceptance_at
        and fact.is_amendment is derivation.is_amendment
        and fact.observation_hash == derivation.observation_hash
        and filing.pk == derivation.filing_source_asset_id
    )


def _validate_fact_availability_authority(
    fact: FundamentalFact,
    *,
    derivation: SecFactDerivation,
    lineage_chain: tuple[FundamentalFact, ...],
    cik: str,
    data_cutoff: datetime,
    decision_time: datetime,
) -> SourceObservationEvent | None:
    base_flags = list(derivation.base_quality_flags)
    if fact.source_revision == 1:
        if (
            fact.available_at != derivation.acceptance_at
            or fact.availability_basis != derivation.filing_availability_basis
            or fact.quality_flags != base_flags
            or CORRECTION_QUALITY_FLAG in fact.quality_flags
            or REBOUND_QUALITY_FLAG in fact.quality_flags
        ):
            raise ValueError("original availability")
        return None
    if fact.source_revision < 1:
        raise ValueError("source revision")
    chain = sorted(
        (item for item in lineage_chain if item.source_revision <= fact.source_revision),
        key=lambda item: item.source_revision,
    )
    if (
        [item.source_revision for item in chain] != list(range(1, fact.source_revision + 1))
        or not chain
        or chain[-1].pk != fact.pk
    ):
        raise ValueError("correction predecessor chain")
    predecessor = chain[-2]
    predecessor_resolution = resolve_availability(chain[:-1])[str(predecessor.pk)]
    rebound_required = (
        predecessor.source_revision > 1
        and predecessor.availability_basis != CORRECTION_AVAILABILITY_BASIS
        and predecessor.observation_hash == fact.observation_hash
        and predecessor_resolution.proven_at is None
    )
    expected_flags = [
        *base_flags,
        CORRECTION_QUALITY_FLAG,
        *([REBOUND_QUALITY_FLAG] if rebound_required else []),
    ]
    if (
        fact.availability_basis != CORRECTION_AVAILABILITY_BASIS
        or fact.quality_flags != expected_flags
    ):
        raise ValueError("correction availability flags")
    candidates = list(
        SourceObservationEvent.objects.filter(
            provider="sec",
            kind=COMPANYFACTS_KIND,
            subject=cik,
            source_asset_id=fact.source_asset_id,
            content_sha256=fact.source_asset.sha256,
            observed_at__lte=data_cutoff,
            recorded_at__lte=decision_time,
        ).select_related("source_asset")
    )
    matching = [
        event
        for event in candidates
        if max(derivation.acceptance_at, event.observed_at) == fact.available_at
    ]
    if len(matching) != 1:
        raise ValueError("correction observation event")
    event = validate_sec_observation_event(matching[0])
    if (
        event.provider != "sec"
        or event.kind != COMPANYFACTS_KIND
        or event.subject != cik
        or event.source_asset_id != fact.source_asset_id
        or event.content_sha256 != fact.source_asset.sha256
        or event.source_asset.retrieved_at > event.observed_at
        or event.observed_at > data_cutoff
        or event.recorded_at > decision_time
        or event.observed_at > event.recorded_at
        or event.observed_at.tzinfo is None
        or event.observed_at.utcoffset() is None
        or event.recorded_at.tzinfo is None
        or event.recorded_at.utcoffset() is None
    ):
        raise ValueError("correction observation identity")
    return event


def _assess_entity(
    *,
    listing: Listing,
    price: Decimal,
    price_asset: DataAsset,
    facts: list[FundamentalFact],
    lineage_facts: list[FundamentalFact],
    classification: CompanyClassificationObservation | None,
    filing_assets: dict[str, DataAsset],
    data_cutoff: datetime,
    target_date: date,
    config: LongForecastV4Config,
    sec_config: Any,
    raw_fcf_authority: dict[str, Any],
) -> _EntityAssessment:
    """Assess one target or peer without ever selecting another peer set."""
    ordered_facts = tuple(
        sorted(
            facts,
            key=lambda fact: (
                fact.concept,
                fact.period_end,
                fact.period_start or fact.period_end,
                fact.available_at,
                fact.source_revision,
                fact.observation_hash,
            ),
        )
    )
    ordered_lineage_facts = tuple(
        sorted(
            lineage_facts,
            key=lambda fact: (
                fact.concept,
                fact.period_end,
                fact.period_start or fact.period_end,
                fact.available_at,
                fact.source_revision,
                fact.observation_hash,
            ),
        )
    )
    common_failure = _listing_failure(listing)
    if listing.security.security_type == Security.SecurityType.ADR:
        common_failure = (
            "adr_depositary_receipt_withheld",
            "Depositary receipts are withheld because us-sec-long-v4 does not infer ADS ratios",
        )
    if common_failure is not None:
        code, reason = common_failure
        return _failed_assessment(
            listing,
            price,
            price_asset,
            ordered_facts,
            classification,
            code,
            reason,
            raw_fcf_authority=raw_fcf_authority,
        )
    if classification is None or _normalized_sic(classification.code) is None:
        return _failed_assessment(
            listing,
            price,
            price_asset,
            ordered_facts,
            classification,
            "classification_unavailable_or_ambiguous",
            "One unambiguous cutoff-safe four-digit SEC SIC classification is required",
            raw_fcf_authority=raw_fcf_authority,
        )
    if classification.available_at > data_cutoff:
        return _failed_assessment(
            listing,
            price,
            price_asset,
            ordered_facts,
            classification,
            "classification_after_cutoff",
            "The selected SEC SIC classification was not available by data_cutoff",
            raw_fcf_authority=raw_fcf_authority,
        )
    if any(str(fact.pk) not in filing_assets for fact in ordered_facts):
        return _failed_assessment(
            listing,
            price,
            price_asset,
            ordered_facts,
            classification,
            "filing_evidence_incomplete",
            "One or more assessed SEC facts lacks cutoff-visible filing evidence",
            raw_fcf_authority=raw_fcf_authority,
        )

    admitted, deferred = partition_unproven_corrections(
        ordered_lineage_facts,
        available_through=data_cutoff,
    )
    raw_status = raw_fcf_authority.get("status")
    if raw_status not in {
        "absent",
        "present_complete",
        "present_normalization_incomplete",
    }:
        raise ValueError("Long-v4 raw FCF authority status is malformed")
    fcf_present = raw_status != "absent"
    evidence_selection = {
        "ttm_selection_policy": TTM_SELECTION_NEWEST_QUARTER_ALIAS,
        "ttm_alias_selection": {},
        "fcf_raw_source_evidence_present": fcf_present,
        "raw_fcf_authority": raw_fcf_authority,
        "metric_family_assessed": ("fcf_per_share" if fcf_present else "net_income_per_share"),
        "deferred_unproven_corrections": [
            deferred_correction_payload(item, available_through=data_cutoff) for item in deferred
        ],
    }
    if raw_status == "present_normalization_incomplete":
        return _EntityAssessment(
            listing=listing,
            price=price,
            price_asset=price_asset,
            facts=ordered_facts,
            admitted_facts=tuple(admitted),
            classification=classification,
            status="withheld",
            insufficiency_code="fcf_normalization_incomplete",
            reason=(
                "Cutoff/window-relevant raw FCF evidence exists but its normalized "
                "fact closure is incomplete; net-income fallback is blocked"
            ),
            metric=None,
            selected_fact_ids=(),
            assessed_fact_ids=tuple(str(fact.pk) for fact in ordered_facts),
            evidence_selection=evidence_selection,
            split_basis=_split_basis_document(),
        )
    series = build_sec_fundamental_series(
        admitted,
        config=sec_config,
        ttm_selection=TTM_SELECTION_NEWEST_QUARTER_ALIAS,
    )
    family = "fcf_per_share" if fcf_present else "net_income_per_share"
    metric, code, reason, split_basis = _assess_metric_family(
        family=family,
        series=series,
        admitted_facts=admitted,
        price=price,
        target_date=target_date,
        config=config,
    )
    evidence_selection["ttm_alias_selection"] = series.ttm_alias_selection
    if metric is None:
        if fcf_present:
            reason = f"FCF evidence is present, so net-income fallback is blocked: {reason}"
            code = f"fcf_{code}"
        return _EntityAssessment(
            listing=listing,
            price=price,
            price_asset=price_asset,
            facts=ordered_facts,
            admitted_facts=tuple(admitted),
            classification=classification,
            status="withheld",
            insufficiency_code=code,
            reason=_bounded_reason(reason),
            metric=None,
            selected_fact_ids=(),
            assessed_fact_ids=tuple(str(fact.pk) for fact in ordered_facts),
            evidence_selection=evidence_selection,
            split_basis=split_basis,
        )
    selected = set(metric.selected_fact_ids)
    return _EntityAssessment(
        listing=listing,
        price=price,
        price_asset=price_asset,
        facts=ordered_facts,
        admitted_facts=tuple(admitted),
        classification=classification,
        status="eligible",
        insufficiency_code=None,
        reason="",
        metric=metric,
        selected_fact_ids=metric.selected_fact_ids,
        assessed_fact_ids=tuple(
            str(fact.pk) for fact in ordered_facts if str(fact.pk) not in selected
        ),
        evidence_selection=evidence_selection,
        split_basis=split_basis,
    )


def _failed_assessment(
    listing: Listing,
    price: Decimal,
    price_asset: DataAsset,
    facts: tuple[FundamentalFact, ...],
    classification: CompanyClassificationObservation | None,
    code: str,
    reason: str,
    *,
    raw_fcf_authority: dict[str, Any],
) -> _EntityAssessment:
    raw_status = raw_fcf_authority.get("status")
    return _EntityAssessment(
        listing=listing,
        price=price,
        price_asset=price_asset,
        facts=facts,
        admitted_facts=(),
        classification=classification,
        status="withheld",
        insufficiency_code=code,
        reason=_bounded_reason(reason),
        metric=None,
        selected_fact_ids=(),
        assessed_fact_ids=tuple(str(fact.pk) for fact in facts),
        evidence_selection={
            "ttm_selection_policy": TTM_SELECTION_NEWEST_QUARTER_ALIAS,
            "ttm_alias_selection": {},
            "fcf_raw_source_evidence_present": (
                raw_status != "absent"
                if raw_status in {"absent", "present_complete", "present_normalization_incomplete"}
                else None
            ),
            "raw_fcf_authority": raw_fcf_authority,
            "metric_family_assessed": None,
            "deferred_unproven_corrections": [],
        },
        split_basis=_split_basis_document(),
    )


def _listing_failure(listing: Listing) -> tuple[str, str] | None:
    if not listing.is_active:
        return "listing_inactive", "us-sec-long-v4 requires an active listing"
    if listing.region != Region.US:
        return "listing_not_us", "us-sec-long-v4 requires a US listing"
    if listing.currency != "USD":
        return "listing_not_usd", "us-sec-long-v4 requires a USD listing and performs no FX"
    if listing.security.security_type != Security.SecurityType.COMMON_STOCK:
        return (
            "security_not_common_stock",
            "us-sec-long-v4 core evidence is available only for common stock",
        )
    return None


def _split_basis_document(
    *,
    assessment_status: str = "not_assessed",
    assessed_through: date | None = None,
    checks: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    copied_checks = [dict(check) for check in checks]
    relevant_ids: list[str] = []
    for check in copied_checks:
        raw_ids = check.get("relevant_fact_ids", [])
        if isinstance(raw_ids, list):
            relevant_ids.extend(str(value) for value in raw_ids)
    return {
        "basis": "reported_weighted_average_diluted_shares",
        "price_basis": "split_adjusted_close",
        "continuity_tolerance": 0.15,
        "assessment_status": assessment_status,
        "assessed_through": assessed_through.isoformat() if assessed_through else None,
        "checks": copied_checks,
        "relevant_fact_ids": list(_dedupe_text(relevant_ids)),
        "post_period_split_status": "unverified",
    }


def _basis_compatibility_check(
    *,
    check: str,
    outcome: str,
    assessed_through: date,
    compared_fields: tuple[str, ...],
    expected: dict[str, Any],
    actual: dict[str, Any],
    relevant_fact_ids: tuple[str, ...] | list[str],
    inclusive_tolerance: float | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "check": check,
        "outcome": outcome,
        "assessed_through": assessed_through.isoformat(),
        "compared_fields": list(compared_fields),
        "expected": expected,
        "actual": actual,
        "relevant_fact_ids": list(_dedupe_text(list(relevant_fact_ids))),
    }
    if inclusive_tolerance is not None:
        result["inclusive_tolerance"] = inclusive_tolerance
    result.update(details or {})
    return result


def _assess_metric_family(
    *,
    family: str,
    series: SecFundamentalSeries,
    admitted_facts: tuple[FundamentalFact, ...],
    price: Decimal,
    target_date: date,
    config: LongForecastV4Config,
) -> tuple[_MetricEvidence | None, str, str, dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    assessed_through: date | None = None

    def append_incompatible(
        *,
        check: str,
        through: date,
        compared_fields: tuple[str, ...],
        expected: dict[str, Any],
        actual: dict[str, Any],
        relevant_fact_ids: tuple[str, ...] | list[str],
        inclusive_tolerance: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        nonlocal assessed_through
        assessed_through = through
        checks.append(
            _basis_compatibility_check(
                check=check,
                outcome="incompatible",
                assessed_through=through,
                compared_fields=compared_fields,
                expected=expected,
                actual=actual,
                relevant_fact_ids=relevant_fact_ids,
                inclusive_tolerance=inclusive_tolerance,
                details=details,
            )
        )

    def failed(
        code: str,
        reason: str,
        *,
        incompatible: bool = False,
    ) -> tuple[None, str, str, dict[str, Any]]:
        status = (
            "incompatible_or_unverified"
            if incompatible
            else ("unverified" if checks else "not_assessed")
        )
        return (
            None,
            code,
            reason,
            _split_basis_document(
                assessment_status=status,
                assessed_through=assessed_through,
                checks=checks,
            ),
        )

    family_config = config.metric_families[family]
    metric = series.ttm.get(family_config.source_metric)
    shares = series.ttm.get("weighted_average_diluted_shares")
    if metric is None or shares is None:
        return failed(
            "ttm_missing_or_incompatible",
            (
                f"{family} requires a homogeneous newest-quarter TTM entity metric "
                "and matching weighted-average diluted shares"
            ),
        )
    if metric.period_start != shares.period_start or metric.period_end != shares.period_end:
        append_incompatible(
            check="ttm_entity_share_period_compatibility",
            through=max(metric.period_end, shares.period_end),
            compared_fields=("period_start", "period_end"),
            expected={
                "entity_period_start_equals_share_period_start": True,
                "entity_period_end_equals_share_period_end": True,
            },
            actual={
                "entity_period_start": metric.period_start.isoformat(),
                "entity_period_end": metric.period_end.isoformat(),
                "share_period_start": shares.period_start.isoformat(),
                "share_period_end": shares.period_end.isoformat(),
            },
            relevant_fact_ids=[*metric.source_fact_ids, *shares.source_fact_ids],
        )
        return failed(
            "ttm_period_mismatch",
            "TTM entity metric and diluted shares do not match",
            incompatible=True,
        )
    age = (target_date - metric.period_end).days
    if age < 0 or age > config.eligibility.maximum_metric_age_days:
        return failed(
            "ttm_stale_or_future",
            (
                f"The latest compatible TTM metric is {age} days from the target; "
                f"the allowed range is 0..{config.eligibility.maximum_metric_age_days}"
            ),
        )
    current_entity_decimal = metric.value
    current_shares_decimal = shares.value
    current_entity = _finite(current_entity_decimal)
    current_shares = _finite(current_shares_decimal)
    if current_entity is None or current_entity <= 0:
        return failed(
            "ttm_entity_nonpositive_or_nonfinite",
            "The latest compatible TTM entity metric must be finite and strictly positive",
        )
    if current_shares is None or current_shares <= 0:
        return failed(
            "ttm_diluted_shares_nonpositive_or_nonfinite",
            "The latest compatible TTM diluted shares must be finite and strictly positive",
        )
    try:
        current_per_share_exact = current_entity_decimal / current_shares_decimal
    except (InvalidOperation, ZeroDivisionError):
        current_per_share_exact = Decimal("NaN")
    if not current_per_share_exact.is_finite() or current_per_share_exact <= 0:
        return failed(
            "ttm_per_share_nonpositive_or_nonfinite",
            "The latest TTM per-share metric must be finite and strictly positive",
        )
    current_per_share = float(current_per_share_exact)
    if not math.isfinite(current_per_share) or current_per_share <= 0:
        return failed(
            "ttm_per_share_nonpositive_or_nonfinite",
            "The latest TTM per-share metric must be finite and strictly positive",
        )
    assessed_through = metric.period_end
    checks.append(
        _basis_compatibility_check(
            check="ttm_entity_per_share_basis",
            outcome="compatible",
            assessed_through=metric.period_end,
            compared_fields=(
                "period_start",
                "period_end",
                "entity_metric",
                "weighted_average_diluted_shares",
                "derived_per_share",
            ),
            expected={
                "same_period": True,
                "entity_metric_positive_finite": True,
                "weighted_average_diluted_shares_positive_finite": True,
                "derived_per_share_positive_finite": True,
            },
            actual={
                "period_start": metric.period_start.isoformat(),
                "period_end": metric.period_end.isoformat(),
                "entity_metric": current_entity,
                "weighted_average_diluted_shares": current_shares,
                "derived_per_share": current_per_share,
            },
            relevant_fact_ids=[*metric.source_fact_ids, *shares.source_fact_ids],
            details={
                "period_start": metric.period_start.isoformat(),
                "period_end": metric.period_end.isoformat(),
                "entity_metric": current_entity,
                "weighted_average_diluted_shares": current_shares,
                "derived_per_share": current_per_share,
            },
        )
    )

    period_identities = _latest_four_annual_identities(
        family=family,
        facts=admitted_facts,
        count=config.eligibility.annual_periods,
    )
    if len(period_identities) != config.eligibility.annual_periods:
        return failed(
            "annual_history_insufficient",
            (
                f"{family} has {len(period_identities)}/{config.eligibility.annual_periods} "
                "distinct annual period identities"
            ),
        )
    metric_values = _values_by_period(series.annual.get(family_config.source_metric, ()))
    share_values = _values_by_period(series.annual.get("weighted_average_diluted_shares", ()))
    income_values = _values_by_period(series.annual.get("net_income", ()))
    eps_values = _values_by_period(series.annual.get("diluted_eps", ()))

    points: list[_AnnualPoint] = []
    share_checks: list[dict[str, Any]] = []
    entity_signature: tuple[str, str, tuple[str, ...]] | None = None
    share_signature: tuple[str, str, tuple[str, ...]] | None = None
    selected_fact_ids: list[str] = [*metric.source_fact_ids, *shares.source_fact_ids]
    for period_identity, period_start, period_end in period_identities:
        duration = (period_end - period_start).days + 1
        key = (period_start, period_end)
        annual_metric = metric_values.get(key)
        annual_shares = share_values.get(key)
        annual_income = income_values.get(key)
        annual_eps = eps_values.get(key)
        if not (
            config.eligibility.annual_duration_minimum_days
            <= duration
            <= config.eligibility.annual_duration_maximum_days
        ):
            entity_concepts = (
                _FCF_SOURCE_CONCEPTS if family == "fcf_per_share" else frozenset(("net_income",))
            )
            entity_fact_ids = (
                annual_metric.source_fact_ids
                if annual_metric is not None
                else _annual_fact_ids(
                    admitted_facts,
                    concepts=entity_concepts,
                    period_start=period_start,
                    period_end=period_end,
                )
            )
            share_fact_ids = (
                annual_shares.source_fact_ids
                if annual_shares is not None
                else _annual_fact_ids(
                    admitted_facts,
                    concepts=frozenset(("weighted_average_diluted_shares",)),
                    period_start=period_start,
                    period_end=period_end,
                )
            )
            append_incompatible(
                check="annual_duration_compatibility",
                through=period_end,
                compared_fields=("duration_days", "period_start", "period_end"),
                expected={
                    "minimum_days": config.eligibility.annual_duration_minimum_days,
                    "maximum_days": config.eligibility.annual_duration_maximum_days,
                    "inclusive": True,
                },
                actual={
                    "duration_days": duration,
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                },
                relevant_fact_ids=[*entity_fact_ids, *share_fact_ids],
                details={"period_identity": period_identity},
            )
            return failed(
                "annual_duration_incompatible",
                (f"Latest annual period {period_identity} spans {duration} days, outside 350..380"),
                incompatible=True,
            )
        if (
            annual_metric is None
            or annual_shares is None
            or annual_income is None
            or annual_eps is None
        ):
            return failed(
                "annual_tuple_incomplete",
                (
                    f"Latest annual period {period_identity} lacks a matching entity metric, "
                    "diluted shares, net income, or reported diluted EPS"
                ),
            )
        entity_value = _finite(annual_metric.value)
        share_value = _finite(annual_shares.value)
        if entity_value is None or entity_value <= 0:
            return failed(
                "annual_entity_nonpositive_or_nonfinite",
                (
                    f"Latest annual period {period_identity} has a non-finite or "
                    "non-positive entity metric"
                ),
            )
        if share_value is None or share_value <= 0:
            return failed(
                "annual_diluted_shares_nonpositive_or_nonfinite",
                (
                    f"Latest annual period {period_identity} has non-finite or "
                    "non-positive diluted shares"
                ),
            )
        per_share = entity_value / share_value
        if not math.isfinite(per_share) or per_share <= 0:
            return failed(
                "annual_per_share_nonpositive_or_nonfinite",
                (
                    f"Latest annual period {period_identity} has a non-finite or "
                    "non-positive per-share value"
                ),
            )
        current_entity_signature = (
            annual_metric.unit,
            annual_metric.derivation,
            annual_metric.source_concepts,
        )
        current_share_signature = (
            annual_shares.unit,
            annual_shares.derivation,
            annual_shares.source_concepts,
        )
        if entity_signature is None:
            entity_signature = current_entity_signature
            share_signature = current_share_signature
        elif (
            current_entity_signature != entity_signature
            or current_share_signature != share_signature
        ):
            assert share_signature is not None
            baseline = points[0]
            append_incompatible(
                check="annual_entity_share_signature_compatibility",
                through=period_end,
                compared_fields=(
                    "entity_unit",
                    "entity_derivation",
                    "entity_source_concepts",
                    "share_unit",
                    "share_derivation",
                    "share_source_concepts",
                ),
                expected={
                    "entity_unit": entity_signature[0],
                    "entity_derivation": entity_signature[1],
                    "entity_source_concepts": list(entity_signature[2]),
                    "share_unit": share_signature[0],
                    "share_derivation": share_signature[1],
                    "share_source_concepts": list(share_signature[2]),
                },
                actual={
                    "entity_unit": current_entity_signature[0],
                    "entity_derivation": current_entity_signature[1],
                    "entity_source_concepts": list(current_entity_signature[2]),
                    "share_unit": current_share_signature[0],
                    "share_derivation": current_share_signature[1],
                    "share_source_concepts": list(current_share_signature[2]),
                },
                relevant_fact_ids=[
                    *baseline.entity_fact_ids,
                    *baseline.share_fact_ids,
                    *annual_metric.source_fact_ids,
                    *annual_shares.source_fact_ids,
                ],
            )
            return failed(
                "annual_alias_or_unit_incompatible",
                (
                    "The latest four annual tuples do not use compatible units, derivations, "
                    "and source concepts"
                ),
                incompatible=True,
            )
        if (
            annual_metric.unit != "USD"
            or annual_shares.unit != "shares"
            or annual_income.unit != "USD"
            or annual_eps.unit != "USD/shares"
        ):
            append_incompatible(
                check="annual_required_unit_compatibility",
                through=period_end,
                compared_fields=(
                    "entity_unit",
                    "share_unit",
                    "net_income_unit",
                    "diluted_eps_unit",
                ),
                expected={
                    "entity_unit": "USD",
                    "share_unit": "shares",
                    "net_income_unit": "USD",
                    "diluted_eps_unit": "USD/shares",
                },
                actual={
                    "entity_unit": annual_metric.unit,
                    "share_unit": annual_shares.unit,
                    "net_income_unit": annual_income.unit,
                    "diluted_eps_unit": annual_eps.unit,
                },
                relevant_fact_ids=[
                    *annual_metric.source_fact_ids,
                    *annual_shares.source_fact_ids,
                    *annual_income.source_fact_ids,
                    *annual_eps.source_fact_ids,
                ],
            )
            return failed(
                "annual_unit_incompatible",
                (
                    f"Latest annual period {period_identity} does not use USD, "
                    "shares, and USD/shares"
                ),
                incompatible=True,
            )
        income = _finite(annual_income.value)
        reported_eps = _finite(annual_eps.value)
        if income is None or reported_eps is None:
            append_incompatible(
                check="reported_diluted_eps_finiteness",
                through=period_end,
                compared_fields=("net_income", "reported_diluted_eps"),
                expected={"net_income_finite": True, "reported_diluted_eps_finite": True},
                actual={
                    "net_income": str(annual_income.value),
                    "net_income_finite": income is not None,
                    "reported_diluted_eps": str(annual_eps.value),
                    "reported_diluted_eps_finite": reported_eps is not None,
                },
                relevant_fact_ids=[
                    *annual_metric.source_fact_ids,
                    *annual_shares.source_fact_ids,
                    *annual_income.source_fact_ids,
                    *annual_eps.source_fact_ids,
                ],
            )
            return failed(
                "reported_eps_nonfinite",
                "Reported diluted-EPS reconciliation is non-finite",
                incompatible=True,
            )
        derived_eps = income / share_value
        if not math.isfinite(derived_eps):
            append_incompatible(
                check="reported_diluted_eps_finiteness",
                through=period_end,
                compared_fields=("derived_diluted_eps",),
                expected={"derived_diluted_eps_finite": True},
                actual={
                    "derived_diluted_eps": str(derived_eps),
                    "derived_diluted_eps_finite": False,
                },
                relevant_fact_ids=[
                    *annual_metric.source_fact_ids,
                    *annual_shares.source_fact_ids,
                    *annual_income.source_fact_ids,
                    *annual_eps.source_fact_ids,
                ],
            )
            return failed(
                "reported_eps_nonfinite",
                "Derived diluted-EPS reconciliation is non-finite",
                incompatible=True,
            )
        denominator = max(abs(derived_eps), abs(reported_eps))
        relative_difference = (
            0.0 if denominator == 0 else abs(derived_eps - reported_eps) / denominator
        )
        relevant_fact_ids = list(
            _dedupe_text(
                [
                    *annual_metric.source_fact_ids,
                    *annual_shares.source_fact_ids,
                    *annual_income.source_fact_ids,
                    *annual_eps.source_fact_ids,
                ]
            )
        )
        reconciliation_failed = _exceeds_inclusive(
            relative_difference,
            config.eligibility.reported_eps_relative_tolerance,
        )
        check = _basis_compatibility_check(
            check="reported_diluted_eps_reconciliation",
            outcome="incompatible" if reconciliation_failed else "compatible",
            assessed_through=period_end,
            compared_fields=("relative_difference",),
            expected={
                "relative_difference_maximum": (config.eligibility.reported_eps_relative_tolerance)
            },
            actual={"relative_difference": relative_difference},
            inclusive_tolerance=config.eligibility.reported_eps_relative_tolerance,
            relevant_fact_ids=relevant_fact_ids,
            details={
                "period_identity": period_identity,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "net_income": income,
                "weighted_average_diluted_shares": share_value,
                "derived_eps": derived_eps,
                "reported_diluted_eps": reported_eps,
                "relative_difference_denominator": denominator,
                "relative_difference": relative_difference,
            },
        )
        share_checks.append(check)
        checks.append(check)
        assessed_through = period_end
        if reconciliation_failed:
            return failed(
                "reported_eps_reconciliation_failed",
                (
                    f"Reported diluted EPS for {period_identity} differs from net income/shares "
                    f"by {relative_difference:.1%}, above 15%"
                ),
                incompatible=True,
            )
        selected_fact_ids.extend(
            (
                *annual_metric.source_fact_ids,
                *annual_shares.source_fact_ids,
                *annual_income.source_fact_ids,
                *annual_eps.source_fact_ids,
            )
        )
        points.append(
            _AnnualPoint(
                period_identity=period_identity,
                period_start=period_start,
                period_end=period_end,
                entity_value=entity_value,
                shares=share_value,
                per_share=per_share,
                entity_fact_ids=annual_metric.source_fact_ids,
                share_fact_ids=annual_shares.source_fact_ids,
                net_income_fact_ids=annual_income.source_fact_ids,
                eps_fact_ids=annual_eps.source_fact_ids,
            )
        )
    for previous, current in zip(points, points[1:], strict=False):
        if current.period_start != previous.period_end + timedelta(days=1):
            expected_start = previous.period_end + timedelta(days=1)
            append_incompatible(
                check="adjacent_annual_period_continuity",
                through=current.period_end,
                compared_fields=("current_period_start",),
                expected={"current_period_start": expected_start.isoformat()},
                actual={"current_period_start": current.period_start.isoformat()},
                relevant_fact_ids=[
                    *previous.entity_fact_ids,
                    *previous.share_fact_ids,
                    *current.entity_fact_ids,
                    *current.share_fact_ids,
                ],
                details={
                    "previous_period_identity": previous.period_identity,
                    "previous_period_end": previous.period_end.isoformat(),
                    "current_period_identity": current.period_identity,
                    "current_period_start": current.period_start.isoformat(),
                    "expected_current_period_start": expected_start.isoformat(),
                },
            )
            return failed(
                "annual_period_gap",
                (
                    f"Latest annual periods {previous.period_identity} and "
                    f"{current.period_identity} are not exactly adjacent"
                ),
                incompatible=True,
            )
        share_change = current.shares / previous.shares - 1.0
        share_incompatible = _exceeds_inclusive(
            abs(share_change),
            config.eligibility.share_continuity_relative_tolerance,
        )
        check = _basis_compatibility_check(
            check="adjacent_annual_diluted_share_continuity",
            outcome="incompatible" if share_incompatible else "compatible",
            assessed_through=current.period_end,
            compared_fields=("absolute_relative_change",),
            expected={
                "absolute_relative_change_maximum": (
                    config.eligibility.share_continuity_relative_tolerance
                )
            },
            actual={"absolute_relative_change": abs(share_change)},
            inclusive_tolerance=config.eligibility.share_continuity_relative_tolerance,
            relevant_fact_ids=[
                *previous.entity_fact_ids,
                *previous.share_fact_ids,
                *current.entity_fact_ids,
                *current.share_fact_ids,
            ],
            details={
                "previous_period_identity": previous.period_identity,
                "previous_shares": previous.shares,
                "current_period_identity": current.period_identity,
                "current_shares": current.shares,
                "raw_share_change": share_change,
                "absolute_relative_change": abs(share_change),
            },
        )
        share_checks.append(check)
        checks.append(check)
        assessed_through = current.period_end
        if share_incompatible:
            return failed(
                "annual_share_continuity_failed",
                (f"Adjacent annual diluted shares changed by {abs(share_change):.1%}, above 15%"),
                incompatible=True,
            )
    assert entity_signature is not None
    assert share_signature is not None
    if (
        metric.unit != entity_signature[0]
        or metric.source_concepts != entity_signature[2]
        or shares.unit != share_signature[0]
        or shares.source_concepts != share_signature[2]
    ):
        latest = points[-1]
        append_incompatible(
            check="annual_to_ttm_alias_unit_compatibility",
            through=max(latest.period_end, shares.period_end),
            compared_fields=(
                "entity_unit",
                "entity_source_concepts",
                "share_unit",
                "share_source_concepts",
            ),
            expected={
                "entity_unit": entity_signature[0],
                "entity_source_concepts": list(entity_signature[2]),
                "share_unit": share_signature[0],
                "share_source_concepts": list(share_signature[2]),
            },
            actual={
                "entity_unit": metric.unit,
                "entity_source_concepts": list(metric.source_concepts),
                "share_unit": shares.unit,
                "share_source_concepts": list(shares.source_concepts),
            },
            relevant_fact_ids=[
                *latest.entity_fact_ids,
                *latest.share_fact_ids,
                *metric.source_fact_ids,
                *shares.source_fact_ids,
            ],
        )
        return failed(
            "annual_to_ttm_alias_or_unit_incompatible",
            ("Latest annual and TTM entity/share evidence uses incompatible aliases or units"),
            incompatible=True,
        )
    if metric.derivation not in (
        entity_signature[1],
        "sum_four_contiguous_quarters",
    ) or shares.derivation not in (
        share_signature[1],
        "weighted_four_contiguous_quarters",
    ):
        latest = points[-1]
        append_incompatible(
            check="annual_to_ttm_derivation_compatibility",
            through=max(latest.period_end, shares.period_end),
            compared_fields=("entity_derivation", "share_derivation"),
            expected={
                "entity_derivations": [
                    entity_signature[1],
                    "sum_four_contiguous_quarters",
                ],
                "share_derivations": [
                    share_signature[1],
                    "weighted_four_contiguous_quarters",
                ],
            },
            actual={
                "entity_derivation": metric.derivation,
                "share_derivation": shares.derivation,
            },
            relevant_fact_ids=[
                *latest.entity_fact_ids,
                *latest.share_fact_ids,
                *metric.source_fact_ids,
                *shares.source_fact_ids,
            ],
        )
        return failed(
            "annual_to_ttm_derivation_incompatible",
            ("Latest annual and TTM entity/share derivations are incompatible"),
            incompatible=True,
        )
    latest = points[-1]
    if not (shares.period_start <= latest.period_end <= shares.period_end):
        append_incompatible(
            check="annual_to_ttm_period_overlap",
            through=max(latest.period_end, shares.period_end),
            compared_fields=(
                "latest_annual_period_end",
                "ttm_period_start",
                "ttm_period_end",
            ),
            expected={"latest_annual_period_end_within_ttm_period": True},
            actual={
                "latest_annual_period_end": latest.period_end.isoformat(),
                "ttm_period_start": shares.period_start.isoformat(),
                "ttm_period_end": shares.period_end.isoformat(),
            },
            relevant_fact_ids=[*latest.share_fact_ids, *shares.source_fact_ids],
        )
        return failed(
            "annual_to_ttm_period_incompatible",
            ("Latest annual diluted shares do not overlap the current TTM share period"),
            incompatible=True,
        )
    ttm_share_change = current_shares / latest.shares - 1.0
    ttm_share_incompatible = _exceeds_inclusive(
        abs(ttm_share_change),
        config.eligibility.share_continuity_relative_tolerance,
    )
    check = _basis_compatibility_check(
        check="ttm_to_latest_annual_diluted_share_continuity",
        outcome="incompatible" if ttm_share_incompatible else "compatible",
        assessed_through=shares.period_end,
        compared_fields=("absolute_relative_change",),
        expected={
            "absolute_relative_change_maximum": (
                config.eligibility.share_continuity_relative_tolerance
            )
        },
        actual={"absolute_relative_change": abs(ttm_share_change)},
        inclusive_tolerance=config.eligibility.share_continuity_relative_tolerance,
        relevant_fact_ids=[*latest.share_fact_ids, *shares.source_fact_ids],
        details={
            "annual_period_identity": latest.period_identity,
            "annual_shares": latest.shares,
            "ttm_period_start": shares.period_start.isoformat(),
            "ttm_period_end": shares.period_end.isoformat(),
            "ttm_shares": current_shares,
            "raw_share_change": ttm_share_change,
            "absolute_relative_change": abs(ttm_share_change),
        },
    )
    share_checks.append(check)
    checks.append(check)
    assessed_through = shares.period_end
    if ttm_share_incompatible:
        return failed(
            "annual_to_ttm_share_continuity_failed",
            (
                f"TTM diluted shares differ from the latest annual basis by "
                f"{abs(ttm_share_change):.1%}, above 15%"
            ),
            incompatible=True,
        )

    raw_growth = tuple(
        current.entity_value / previous.entity_value - 1.0
        for previous, current in zip(points, points[1:], strict=False)
    )
    capped_growth = tuple(
        _clamp(value, config.growth.target_minimum, config.growth.target_maximum)
        for value in raw_growth
    )
    target_growth = statistics.median(capped_growth)
    raw_share_changes = tuple(
        current.shares / previous.shares - 1.0
        for previous, current in zip(points, points[1:], strict=False)
    )
    if any(not math.isfinite(value) for value in (*raw_growth, *raw_share_changes)):
        return failed(
            "growth_or_dilution_nonfinite",
            "Annual entity growth or diluted-share change is non-finite",
        )
    dilution_base = _clamp(
        max(0.0, statistics.median(raw_share_changes)),
        0.0,
        0.15,
    )
    try:
        # Keep the eligibility gate and return denominator algebraically
        # invariant under a split.  Dividing entity value by shares first can
        # round a repeating Decimal quotient before price is divided by it,
        # making equivalent price/share transformations land on opposite
        # sides of the inclusive family floor.
        current_multiple_exact = price * current_shares_decimal / current_entity_decimal
    except (InvalidOperation, ZeroDivisionError):
        current_multiple_exact = Decimal("NaN")
    if not current_multiple_exact.is_finite() or current_multiple_exact <= 0:
        return failed(
            "current_multiple_nonpositive_or_nonfinite",
            "The current valuation multiple must be finite and strictly positive",
        )
    if current_multiple_exact < Decimal(str(family_config.multiple_minimum)):
        return failed(
            "current_multiple_below_family_minimum",
            (
                f"Raw current {family} multiple {current_multiple_exact:.3f} is below "
                f"{family_config.multiple_minimum:.3f}"
            ),
        )
    current_multiple_raw = float(current_multiple_exact)
    if not math.isfinite(current_multiple_raw):
        return failed(
            "current_multiple_nonpositive_or_nonfinite",
            "The current valuation multiple must be finite and strictly positive",
        )
    current_multiple_capped = _clamp(
        current_multiple_raw,
        family_config.multiple_minimum,
        family_config.multiple_maximum,
    )
    return (
        _MetricEvidence(
            family=family,
            current_entity_metric=current_entity,
            current_shares=current_shares,
            current_per_share=current_per_share,
            current_per_share_exact=current_per_share_exact,
            current_price_valuation=float(price),
            current_period_start=metric.period_start,
            current_period_end=metric.period_end,
            current_multiple_raw=current_multiple_raw,
            current_multiple_exact=current_multiple_exact,
            current_multiple_capped=current_multiple_capped,
            annual_points=tuple(points),
            entity_growth_raw=raw_growth,
            entity_growth_capped=capped_growth,
            target_entity_growth=target_growth,
            share_changes_raw=raw_share_changes,
            dilution_base=dilution_base,
            share_checks=tuple(share_checks),
            selected_fact_ids=_dedupe_text(selected_fact_ids),
        ),
        "",
        "",
        _split_basis_document(
            assessment_status="unverified",
            assessed_through=assessed_through,
            checks=checks,
        ),
    )


def _latest_four_annual_identities(
    *,
    family: str,
    facts: tuple[FundamentalFact, ...],
    count: int,
) -> tuple[tuple[str, date, date], ...]:
    concepts = _FCF_SOURCE_CONCEPTS if family == "fcf_per_share" else frozenset(("net_income",))
    identities: dict[str, tuple[str, date, date]] = {}
    for fact in facts:
        if (
            fact.concept not in concepts
            or fact.period_type != FundamentalFact.PeriodType.DURATION
            or fact.period_start is None
            or fact.fiscal_period != "FY"
        ):
            continue
        identity = fact.period_identity
        period = (identity, fact.period_start, fact.period_end)
        existing = identities.get(identity)
        if existing is not None and existing[1:] != period[1:]:
            # A persisted identity must not name two reporting intervals.
            return ()
        identities[identity] = period
    ordered = sorted(identities.values(), key=lambda item: (item[2], item[1], item[0]))
    return tuple(ordered[-count:])


def _annual_fact_ids(
    facts: tuple[FundamentalFact, ...],
    *,
    concepts: frozenset[str],
    period_start: date,
    period_end: date,
) -> tuple[str, ...]:
    """Return source-ordered, semantic-deduplicated IDs for one annual role."""
    identities: set[tuple[Any, ...]] = set()
    result: list[str] = []
    for fact in facts:
        if (
            fact.concept not in concepts
            or fact.period_start != period_start
            or fact.period_end != period_end
        ):
            continue
        semantic_identity = (
            fact.company_id,
            fact.concept,
            fact.taxonomy,
            fact.source_concept,
            fact.value,
            fact.unit,
            fact.period_identity,
            fact.accession,
            fact.observation_hash,
        )
        if semantic_identity in identities:
            continue
        identities.add(semantic_identity)
        result.append(str(fact.pk))
    return tuple(result)


def _values_by_period(
    values: tuple[FundamentalValue, ...],
) -> dict[tuple[date, date], FundamentalValue]:
    result: dict[tuple[date, date], FundamentalValue] = {}
    ambiguous: set[tuple[date, date]] = set()
    for value in values:
        key = (value.period_start, value.period_end)
        if key in ambiguous:
            continue
        if key in result:
            # Ambiguous period construction is not silently resolved.
            result.pop(key, None)
            ambiguous.add(key)
            continue
        result[key] = value
    return result


def _lock_peer_cohort(
    *,
    target_listing: Listing,
    target_classification: CompanyClassificationObservation,
    listings: tuple[Listing, ...],
    classifications: dict[str, CompanyClassificationObservation | None],
    authoritative_ciks: dict[str, str],
    peer_config: LongForecastV4PeerConfig,
) -> _PeerLock:
    target_sic = _normalized_sic(target_classification.code)
    assert target_sic is not None
    target_listing_id = str(target_listing.pk)
    target_cik = authoritative_ciks.get(target_listing_id)
    if (
        target_cik is None
        or set(authoritative_ciks) != {str(listing.pk) for listing in listings}
        or len(set(authoritative_ciks.values())) != len(authoritative_ciks)
    ):
        raise ValueError("Long-v4 authoritative peer identities are incomplete or ambiguous")
    examined_levels: list[dict[str, Any]] = []
    evidence_candidates: list[Listing] = []
    seen_evidence_ids: set[str] = set()
    for level in peer_config.sic_prefix_levels:
        prefix = target_sic[:level]
        identity_candidates = sorted(
            (
                listing
                for listing in listings
                if authoritative_ciks[str(listing.pk)] != target_cik
                and _listing_failure(listing) is None
                and (classification := classifications[str(listing.pk)]) is not None
                and (sic := _normalized_sic(classification.code)) is not None
                and sic.startswith(prefix)
            ),
            key=lambda listing: (listing.ticker, str(listing.pk)),
        )
        by_issuer: dict[str, Listing] = {}
        for candidate in identity_candidates:
            by_issuer.setdefault(authoritative_ciks[str(candidate.pk)], candidate)
        members = tuple(by_issuer.values())
        floor = peer_config.minimum_cohort[level]
        for candidate in members:
            listing_id = str(candidate.pk)
            if listing_id not in seen_evidence_ids:
                seen_evidence_ids.add(listing_id)
                evidence_candidates.append(candidate)
        candidate_payloads: list[dict[str, Any]] = []
        for candidate in members:
            classification = classifications[str(candidate.pk)]
            assert classification is not None
            candidate_payloads.append(
                {
                    "listing_id": str(candidate.pk),
                    "company_id": str(candidate.security.company_id),
                    "classification_id": str(classification.pk),
                    "sic": _normalized_sic(classification.code),
                }
            )
        examined_levels.append(
            {
                "sic_prefix_level": level,
                "sic_prefix": prefix,
                "identity_floor": floor,
                "candidate_count": len(members),
                "floor_met": len(members) >= floor,
                "candidates": candidate_payloads,
            }
        )
        if len(members) >= floor:
            return _PeerLock(
                status="locked",
                target_classification=target_classification,
                examined_levels=tuple(examined_levels),
                selected_level=level,
                selected_prefix=prefix,
                selected_floor=floor,
                members=members,
                evidence_candidates=tuple(evidence_candidates),
            )
    return _PeerLock(
        status="no_floor",
        target_classification=target_classification,
        examined_levels=tuple(examined_levels),
        selected_level=None,
        selected_prefix=None,
        selected_floor=None,
        members=(),
        evidence_candidates=tuple(evidence_candidates),
    )


def _successful_pair(
    *,
    target: _EntityAssessment,
    locked: _PeerLock,
    peer_assessments: tuple[_EntityAssessment, ...],
    admitted_peers: tuple[_EntityAssessment, ...],
    target_date: date,
    config: LongForecastV4Config,
    filing_assets: dict[str, DataAsset],
    classifications: dict[str, CompanyClassificationObservation | None],
    cohort_prices: tuple[_CanonicalPrice, ...],
    raw_authority: _SecRawAuthority,
    mapping_authority: _SecMappingAuthority,
    store: AssetStore,
    decision_time: datetime,
) -> dict[str, LongForecastV4]:
    assert target.metric is not None
    peer_growth = _clamp(
        statistics.median(
            peer.metric.target_entity_growth for peer in admitted_peers if peer.metric is not None
        ),
        config.growth.peer_minimum,
        config.growth.peer_maximum,
    )
    peer_multiple = statistics.median(
        peer.metric.current_multiple_capped for peer in admitted_peers if peer.metric is not None
    )
    paths = {
        name: _scenario_path(
            metric=target.metric,
            peer_growth=peer_growth,
            peer_multiple=peer_multiple,
            family_config=config.metric_families[target.metric.family],
            scenario_name=name,
            config=config,
        )
        for name in LONG_SCENARIOS
    }
    horizon_returns = {
        horizon: {
            name: paths[name]["years"][years - 1]["cumulative_price_return"]
            for name in LONG_SCENARIOS
        }
        for horizon, years in (("3y", 3), ("5y", 5))
    }
    annualized = {
        horizon: {
            name: (1.0 + value) ** (1.0 / years) - 1.0
            for name, value in horizon_returns[horizon].items()
        }
        for horizon, years in (("3y", 3), ("5y", 5))
    }
    if any(
        list(horizon_returns[horizon].values()) != sorted(horizon_returns[horizon].values())
        for horizon in LONG_FORECAST_HORIZONS
    ):
        return _withheld_pair(
            target=target,
            peer_assessments=peer_assessments,
            reason_code="scenario_ordering_failed",
            reason="Bear, base, and bull returns were not ordered for both exact views",
            target_date=target_date,
            config=config,
            filing_assets=filing_assets,
            peer_lock=locked,
            classifications=classifications,
            cohort_prices=cohort_prices,
            raw_authority=raw_authority,
            mapping_authority=mapping_authority,
            store=store,
            decision_time=decision_time,
        )

    catalog = _evidence_catalog(
        target=target,
        peer_lock=locked,
        peer_assessments=peer_assessments,
        selected_peer_ids={str(peer.listing.pk) for peer in admitted_peers},
        filing_assets=filing_assets,
        classifications=classifications,
        cohort_prices=cohort_prices,
        raw_authority=raw_authority,
        mapping_authority=mapping_authority,
    )
    assets = _source_assets_for_catalog(
        catalog,
        store=store,
        decision_time=decision_time,
    )
    target_price = _catalog_price_for(catalog, str(target.listing.pk))
    peer_set = [_peer_payload(peer) for peer in admitted_peers if peer.metric is not None]
    common = _base_calculation(
        target=target,
        peer_lock=locked,
        peer_assessments=peer_assessments,
        catalog=catalog,
        source_assets=assets,
        target_date=target_date,
        config=config,
        insufficiency_code=None,
        insufficiency_reason="",
    )
    common.update(
        {
            "metric_family": target.metric.family,
            "support": {
                "annual_periods": 4,
                "growth_observations": 3,
                "share_change_observations": 3,
                "locked_peer_count": len(locked.members),
                "admitted_peer_count": len(admitted_peers),
                "peer_count": len(admitted_peers),
                "locked_peer_floor": locked.floor,
                "sic_fallback_level": locked.prefix_length,
                "sic_prefix": locked.prefix,
            },
            "formula_inputs": {
                "current_price": float(target.price),
                "current_price_ledger": target_price["ledger_value"],
                "current_price_valuation": target_price["valuation_value"],
                "current_entity_metric": target.metric.current_entity_metric,
                "current_weighted_average_diluted_shares": target.metric.current_shares,
                "current_per_share_metric": target.metric.current_per_share,
                "current_per_share_exact": _decimal_text(target.metric.current_per_share_exact),
                "current_multiple_raw": target.metric.current_multiple_raw,
                "current_multiple_exact": _decimal_text(target.metric.current_multiple_exact),
                "current_multiple_capped_reversion_anchor": (target.metric.current_multiple_capped),
                "annual_entity_metrics": [
                    point.entity_value for point in target.metric.annual_points
                ],
                "annual_diluted_shares": [point.shares for point in target.metric.annual_points],
                "entity_growth_raw": list(target.metric.entity_growth_raw),
                "entity_growth_capped": list(target.metric.entity_growth_capped),
                "target_entity_growth": target.metric.target_entity_growth,
                "diluted_share_changes_raw": list(target.metric.share_changes_raw),
                "dilution_base": target.metric.dilution_base,
                "peer_entity_growth": peer_growth,
                "peer_multiple": peer_multiple,
                "target_weight": config.growth.target_weight,
                "peer_weight": config.growth.peer_weight,
                "terminal_entity_growth": config.growth.terminal_entity_growth,
                "fade": list(config.path.fade),
                "multiple_reversion": list(config.path.multiple_reversion),
            },
            "split_basis": target.split_basis,
            "scenario_paths": paths,
            "annualized_returns": annualized,
            "peer_set": peer_set,
            "locked_peer_candidates": [
                _peer_candidate_payload(
                    peer,
                    locked=locked,
                    selected=str(peer.listing.pk)
                    in {str(item.listing.pk) for item in admitted_peers},
                    target_family=target.metric.family,
                )
                for peer in peer_assessments
            ],
            "probability_semantics": {
                "status": "unavailable",
                "value": None,
                "reason": PROBABILITY_REASON,
            },
            "confidence_semantics": {
                "status": config.confidence.success_status,
                "value": config.confidence.numeric_value,
                "schema": config.confidence.numeric_semantics,
            },
        }
    )
    return {
        horizon: _view_forecast(
            common=common,
            horizon=horizon,
            returns=horizon_returns[horizon],
            annualized=annualized[horizon],
            source_assets=assets,
            config=config,
        )
        for horizon in LONG_FORECAST_HORIZONS
    }


def _scenario_path(
    *,
    metric: _MetricEvidence,
    peer_growth: float,
    peer_multiple: float,
    family_config: LongForecastV4MetricFamilyConfig,
    scenario_name: str,
    config: LongForecastV4Config,
) -> dict[str, Any]:
    scenario = config.scenarios[scenario_name]
    target_growth = _clamp(
        metric.target_entity_growth + scenario.growth_delta,
        config.growth.target_minimum,
        config.growth.target_maximum,
    )
    scenario_peer_growth = _clamp(
        peer_growth + scenario.growth_delta,
        config.growth.peer_minimum,
        config.growth.peer_maximum,
    )
    entity_growth = _clamp(
        config.growth.target_weight * target_growth
        + config.growth.peer_weight * scenario_peer_growth,
        config.growth.entity_minimum,
        config.growth.entity_maximum,
    )
    dilution = _clamp(
        metric.dilution_base * scenario.dilution_multiplier,
        0.0,
        0.15,
    )
    destination_multiple = _clamp(
        peer_multiple * scenario.peer_multiple_multiplier,
        family_config.multiple_minimum,
        family_config.multiple_maximum,
    )
    cumulative_entity_factor = 1.0
    cumulative_per_share_factor = 1.0
    years: list[dict[str, Any]] = []
    for index, (fade, reversion) in enumerate(
        zip(config.path.fade, config.path.multiple_reversion, strict=True),
        start=1,
    ):
        annual_entity_growth = (
            fade * entity_growth + (1.0 - fade) * config.growth.terminal_entity_growth
        )
        annual_per_share_growth = (1.0 + annual_entity_growth) / (1.0 + dilution) - 1.0
        cumulative_entity_factor *= 1.0 + annual_entity_growth
        cumulative_per_share_factor *= 1.0 + annual_per_share_growth
        projected_per_share = metric.current_per_share * cumulative_per_share_factor
        multiple = math.exp(
            (1.0 - reversion) * math.log(metric.current_multiple_capped)
            + reversion * math.log(destination_multiple)
        )
        projected_price = projected_per_share * multiple
        level_return_check = projected_price / metric.current_price_valuation - 1
        factor_return = cumulative_per_share_factor * multiple / metric.current_multiple_raw - 1
        if not math.isclose(
            level_return_check,
            factor_return,
            rel_tol=0,
            abs_tol=RETURN_IDENTITY_TOLERANCE,
        ):
            raise ValueError("Long-v4 level and factor return identities diverged")
        years.append(
            {
                "year": index,
                "fade": fade,
                "multiple_reversion": reversion,
                "entity_growth": annual_entity_growth,
                "dilution_rate": dilution,
                "per_share_growth": annual_per_share_growth,
                "cumulative_entity_factor": cumulative_entity_factor,
                "cumulative_per_share_factor": cumulative_per_share_factor,
                "projected_per_share_metric": projected_per_share,
                "multiple": multiple,
                "projected_price": projected_price,
                "cumulative_price_return": factor_return,
                # Persist the direct-multiple identity as the canonical return
                # for both views. The separately computed level identity above
                # remains a validation check, not a second rounded denominator.
                "return_from_level_identity": factor_return,
                "return_from_factor_identity": factor_return,
            }
        )
    return {
        "scenario": scenario_name,
        "target_entity_growth": target_growth,
        "peer_entity_growth": scenario_peer_growth,
        "blended_entity_growth": entity_growth,
        "dilution_base": metric.dilution_base,
        "dilution_multiplier": scenario.dilution_multiplier,
        "dilution_rate": dilution,
        "peer_multiple_raw_median": peer_multiple,
        "peer_multiple_multiplier": scenario.peer_multiple_multiplier,
        "peer_multiple_destination": destination_multiple,
        "current_multiple_return_denominator": metric.current_multiple_raw,
        "current_multiple_reversion_anchor": metric.current_multiple_capped,
        "years": years,
    }


def _withheld_pair(
    *,
    target: _EntityAssessment,
    peer_assessments: tuple[_EntityAssessment, ...],
    reason_code: str,
    reason: str,
    target_date: date,
    config: LongForecastV4Config,
    filing_assets: dict[str, DataAsset],
    peer_lock: _PeerLock | None,
    classifications: dict[str, CompanyClassificationObservation | None],
    cohort_prices: tuple[_CanonicalPrice, ...],
    raw_authority: _SecRawAuthority,
    mapping_authority: _SecMappingAuthority,
    store: AssetStore,
    decision_time: datetime,
) -> dict[str, LongForecastV4]:
    bounded_reason = _bounded_reason(reason)
    catalog = _evidence_catalog(
        target=target,
        peer_lock=peer_lock,
        peer_assessments=peer_assessments,
        selected_peer_ids=set(),
        filing_assets=filing_assets,
        classifications=classifications,
        cohort_prices=cohort_prices,
        raw_authority=raw_authority,
        mapping_authority=mapping_authority,
    )
    assets = _source_assets_for_catalog(
        catalog,
        store=store,
        decision_time=decision_time,
    )
    common = _base_calculation(
        target=target,
        peer_lock=peer_lock,
        peer_assessments=peer_assessments,
        catalog=catalog,
        source_assets=assets,
        target_date=target_date,
        config=config,
        insufficiency_code=reason_code,
        insufficiency_reason=bounded_reason,
    )
    common.update(
        {
            "metric_family": (target.metric.family if target.metric is not None else None),
            "support": {},
            "formula_inputs": {},
            "split_basis": target.split_basis,
            "scenario_paths": {},
            "annualized_returns": {},
            "peer_set": [],
            "probability_semantics": {
                "status": "unavailable",
                "value": None,
                "reason": bounded_reason,
            },
            "confidence_semantics": {
                "status": config.confidence.withheld_status,
                "value": config.confidence.numeric_value,
                "schema": config.confidence.numeric_semantics,
            },
        }
    )
    return {
        horizon: _view_forecast(
            common=common,
            horizon=horizon,
            returns={name: None for name in LONG_SCENARIOS},
            annualized={},
            source_assets=assets,
            config=config,
        )
        for horizon in LONG_FORECAST_HORIZONS
    }


def _base_calculation(
    *,
    target: _EntityAssessment,
    peer_lock: _PeerLock | None,
    peer_assessments: tuple[_EntityAssessment, ...],
    catalog: dict[str, Any],
    source_assets: tuple[DataAsset, ...],
    target_date: date,
    config: LongForecastV4Config,
    insufficiency_code: str | None,
    insufficiency_reason: str,
) -> dict[str, Any]:
    target_price = _catalog_price_for(catalog, str(target.listing.pk))
    return {
        "schema_version": CALCULATION_SCHEMA_VERSION,
        "method": LONG_V4_METHOD,
        "method_version": LONG_V4_VERSION,
        "config_hash": LONG_V4_EFFECTIVE_CONFIG_HASH,
        "fundamentals_config_version": config.fundamentals_config_version,
        "fundamentals_config_file_sha256": config.fundamentals_config_file_sha256,
        "fundamentals_config_hash": config.fundamentals_config_hash,
        "sec_cik_config_version": config.sec_cik_config_version,
        "sec_cik_config_file_sha256": config.sec_cik_config_file_sha256,
        "sec_cik_config_hash": config.sec_cik_config_hash,
        "sec_mapping_source_sha256": config.sec_mapping_source_sha256,
        "research_status": LONG_V4_RESEARCH_STATUS,
        "path_years": config.path_years,
        "target_date": target_date.isoformat(),
        "return_basis": config.return_basis,
        "dividends_included": False,
        "base_currency": config.base_currency,
        "fx_conversion": False,
        "target": _listing_identity_payload(target.listing),
        "target_price": target_price,
        "target_classification": (
            _classification_payload(target.classification)
            if target.classification is not None
            else None
        ),
        "accounting_scope": _accounting_scope(),
        "evidence_catalog": catalog,
        "selected_evidence_ids": catalog["selected_fact_ids"],
        "assessed_evidence_ids": catalog["assessed_fact_ids"],
        "locked_peer_candidates": (
            [
                _peer_candidate_payload(
                    peer,
                    locked=peer_lock,
                    selected=False,
                    target_family=(target.metric.family if target.metric is not None else None),
                )
                for peer in peer_assessments
            ]
            if peer_assessments
            else _unassessed_peer_candidates(peer_lock)
        ),
        "evidence_selection": {
            "target": target.evidence_selection,
            "locked_peers": {
                str(peer.listing.pk): peer.evidence_selection for peer in peer_assessments
            },
            "peer_lock": peer_lock.payload() if peer_lock is not None else None,
        },
        "source_manifest": [_asset_payload(asset) for asset in source_assets],
        "insufficiency_code": insufficiency_code,
        "insufficiency_reason": insufficiency_reason,
    }


def _view_forecast(
    *,
    common: dict[str, Any],
    horizon: str,
    returns: dict[str, float | None],
    annualized: dict[str, float],
    source_assets: tuple[DataAsset, ...],
    config: LongForecastV4Config,
) -> LongForecastV4:
    years = 3 if horizon == "3y" else 5
    reason = common["insufficiency_reason"] or PROBABILITY_REASON
    successful = not common["insufficiency_reason"]
    calculation = {
        **common,
        "forecast_horizon": horizon,
        "years": years,
        "selected_view": {
            "horizon": horizon,
            "year": years,
            "cumulative_returns": returns,
            "annualized_returns": annualized,
        },
    }
    return LongForecastV4(
        scenario=Scenario(
            bear=returns["bear"],
            base=returns["base"],
            bull=returns["bull"],
            probability_positive=None,
            confidence=config.confidence.numeric_value,
            confidence_status=(
                config.confidence.success_status
                if successful
                else config.confidence.withheld_status
            ),
            insufficiency_reason=reason,
            method=LONG_V4_METHOD,
        ),
        calculation=calculation,
        source_assets=source_assets,
    )


def validate_long_v4_forecast_pair(
    forecasts: dict[str, LongForecastV4],
    *,
    config_hash: str,
) -> None:
    """Validate pair shape and canonical shared five-year evidence pre-write."""
    if set(forecasts) != set(LONG_FORECAST_HORIZONS):
        raise ValueError("Long-v4 persistence requires exactly one 3y/5y pair")
    three = forecasts["3y"]
    five = forecasts["5y"]
    if config_hash != LONG_V4_EFFECTIVE_CONFIG_HASH:
        raise ValueError("Long-v4 persistence config hash is not the reviewed identity")
    if tuple(_asset_authority_tuple(asset) for asset in three.source_assets) != tuple(
        _asset_authority_tuple(asset) for asset in five.source_assets
    ):
        raise ValueError("Long-v4 pair source manifests differ")
    allowed_differences = {"forecast_horizon", "years", "selected_view"}
    shared_three = {
        key: value for key, value in three.calculation.items() if key not in allowed_differences
    }
    shared_five = {
        key: value for key, value in five.calculation.items() if key not in allowed_differences
    }
    if shared_three != shared_five:
        raise ValueError("Long-v4 pair does not share canonical evidence and path objects")
    for horizon, years in (("3y", 3), ("5y", 5)):
        forecast = forecasts[horizon]
        calculation = forecast.calculation
        if (
            calculation.get("schema_version") != CALCULATION_SCHEMA_VERSION
            or calculation.get("method") != LONG_V4_METHOD
            or calculation.get("method_version") != LONG_V4_VERSION
            or calculation.get("config_hash") != LONG_V4_EFFECTIVE_CONFIG_HASH
            or calculation.get("fundamentals_config_file_sha256")
            != LONG_V4_FUNDAMENTALS_CONFIG_FILE_SHA256
            or calculation.get("fundamentals_config_hash") != LONG_V4_FUNDAMENTALS_CONFIG_HASH
            or calculation.get("sec_cik_config_version") != LONG_V4_SEC_CIK_CONFIG_VERSION
            or calculation.get("sec_cik_config_file_sha256") != LONG_V4_SEC_CIK_CONFIG_FILE_SHA256
            or calculation.get("sec_cik_config_hash") != LONG_V4_SEC_CIK_CONFIG_HASH
            or calculation.get("sec_mapping_source_sha256") != LONG_V4_SEC_MAPPING_SOURCE_SHA256
            or calculation.get("research_status") != LONG_V4_RESEARCH_STATUS
            or calculation.get("forecast_horizon") != horizon
            or calculation.get("years") != years
            or calculation.get("path_years") != 5
        ):
            raise ValueError("Long-v4 calculation identity or horizon extraction is malformed")
        catalog = calculation.get("evidence_catalog")
        manifest = calculation.get("source_manifest")
        target_price = calculation.get("target_price")
        target_identity = calculation.get("target")
        split_basis = calculation.get("split_basis")
        if (
            not isinstance(catalog, dict)
            or catalog.get("schema_version") != EVIDENCE_CATALOG_SCHEMA_VERSION
            or not isinstance(manifest, list)
            or [item.get("id") for item in manifest if isinstance(item, dict)]
            != list(_catalog_manifest_asset_ids(catalog))
            or not isinstance(target_price, dict)
            or not isinstance(target_identity, dict)
            or target_price.get("listing_id") != target_identity.get("listing_id")
            or target_price.get("session_date") != calculation.get("target_date")
            or target_price.get("provider") != "twelve_data"
            or not isinstance(split_basis, dict)
            or "verified_through" in split_basis
            or split_basis.get("assessment_status")
            not in {"not_assessed", "unverified", "incompatible_or_unverified"}
        ):
            raise ValueError("Long-v4 evidence, price, or assessment document is malformed")
        _validate_catalog_assessment_contract(calculation)
        canonical_target_value = canonical_long_v4_price(target_price.get("value"))
        valuation_target_value = _exact_positive_decimal(target_price.get("valuation_value"))
        if (
            target_price.get("value") != format(canonical_target_value, ".6f")
            or target_price.get("ledger_value") != target_price.get("value")
            or canonical_long_v4_price(valuation_target_value) != canonical_target_value
            or target_price.get("valuation_value") != _decimal_text(valuation_target_value)
            or target_price.get("valuation_source") != "normalized_parquet_target_close"
            or target_price.get("native_price") != target_price.get("valuation_value")
        ):
            raise ValueError("Long-v4 canonical target price serialization is malformed")
        if target_price != _catalog_price_for(catalog, str(target_identity["listing_id"])):
            raise ValueError("Long-v4 target and catalog price identities differ")
        formula_inputs = calculation.get("formula_inputs")
        if isinstance(formula_inputs, dict) and formula_inputs:
            if (
                _exact_positive_decimal(formula_inputs.get("current_price"))
                != valuation_target_value
                or formula_inputs.get("current_price_ledger") != target_price.get("ledger_value")
                or formula_inputs.get("current_price_valuation")
                != target_price.get("valuation_value")
            ):
                raise ValueError("Long-v4 target and formula price values differ")
        view = calculation.get("selected_view")
        if (
            not isinstance(view, dict)
            or view.get("horizon") != horizon
            or view.get("year") != years
        ):
            raise ValueError("Long-v4 selected view is malformed")
        returns = view.get("cumulative_returns")
        if not isinstance(returns, dict) or set(returns) != set(LONG_SCENARIOS):
            raise ValueError("Long-v4 selected return view is malformed")
        scenario_values = {
            "bear": forecast.scenario.bear,
            "base": forecast.scenario.base,
            "bull": forecast.scenario.bull,
        }
        if returns != scenario_values:
            raise ValueError("Long-v4 selected return view and scenario disagree")
        annualized_views = calculation.get("annualized_returns")
        expected_annualized = (
            annualized_views.get(horizon, {}) if isinstance(annualized_views, dict) else None
        )
        if view.get("annualized_returns") != expected_annualized:
            raise ValueError("Long-v4 selected annualized return view is inconsistent")
        reason = (
            calculation.get("insufficiency_reason")
            or calculation["probability_semantics"]["reason"]
        )
        if forecast.scenario.insufficiency_reason != reason:
            raise ValueError("Long-v4 scenario and calculation reasons disagree")
        success = calculation.get("insufficiency_code") is None
        checks = split_basis.get("checks")
        relevant_fact_ids = split_basis.get("relevant_fact_ids")
        if (
            split_basis.get("basis") != "reported_weighted_average_diluted_shares"
            or split_basis.get("price_basis") != "split_adjusted_close"
            or split_basis.get("continuity_tolerance") != 0.15
            or split_basis.get("post_period_split_status") != "unverified"
            or not isinstance(checks, list)
            or not isinstance(relevant_fact_ids, list)
            or relevant_fact_ids
            != list(
                _dedupe_text(
                    [
                        str(fact_id)
                        for check in checks
                        if isinstance(check, dict)
                        for fact_id in check.get("relevant_fact_ids", [])
                    ]
                )
            )
            or (
                success
                and (
                    split_basis.get("assessment_status") != "unverified"
                    or not split_basis.get("assessed_through")
                )
            )
            or (
                split_basis.get("assessment_status") == "not_assessed"
                and (split_basis.get("assessed_through") is not None or checks or relevant_fact_ids)
            )
            or (
                calculation.get("insufficiency_code")
                in {
                    "fcf_reported_eps_reconciliation_failed",
                    "reported_eps_reconciliation_failed",
                    "fcf_annual_duration_incompatible",
                    "annual_duration_incompatible",
                    "fcf_annual_share_continuity_failed",
                    "annual_share_continuity_failed",
                    "fcf_annual_to_ttm_share_continuity_failed",
                    "annual_to_ttm_share_continuity_failed",
                }
                and (
                    split_basis.get("assessment_status") != "incompatible_or_unverified"
                    or not split_basis.get("assessed_through")
                    or not checks
                )
            )
        ):
            raise ValueError("Long-v4 split-basis assessment is malformed")
        expected_confidence = (
            "not_estimated_uncalibrated" if success else "not_estimated_insufficient"
        )
        if (
            forecast.scenario.confidence != 0
            or forecast.scenario.confidence_status != expected_confidence
            or forecast.scenario.probability_positive is not None
            or calculation.get("confidence_semantics")
            != {
                "status": expected_confidence,
                "value": 0.0,
                "schema": "zero_is_unavailable_sentinel",
            }
        ):
            raise ValueError("Long-v4 confidence/probability semantics are malformed")
    paths = three.calculation["scenario_paths"]
    if paths:
        if set(paths) != set(LONG_SCENARIOS):
            raise ValueError("Long-v4 scenario path set is incomplete")
        for name in LONG_SCENARIOS:
            years = paths[name].get("years")
            if not isinstance(years, list) or len(years) != 5:
                raise ValueError("Long-v4 must persist one complete five-year path")
        _validate_successful_path_math(three.calculation)
    else:
        for forecast in forecasts.values():
            if any(
                value is not None
                for value in (
                    forecast.scenario.bear,
                    forecast.scenario.base,
                    forecast.scenario.bull,
                )
            ):
                raise ValueError("Withheld long-v4 pair contains success-shaped returns")
            if forecast.calculation["annualized_returns"]:
                raise ValueError("Withheld long-v4 pair contains annualized values")


def _validate_catalog_assessment_contract(calculation: dict[str, Any]) -> None:
    catalog = calculation.get("evidence_catalog")
    if not isinstance(catalog, dict):
        raise ValueError("Long-v4 evidence catalog is malformed")
    facts = catalog.get("facts")
    selected = catalog.get("selected_fact_ids")
    assessed = catalog.get("assessed_fact_ids")
    target = calculation.get("target")
    target_id = target.get("listing_id") if isinstance(target, dict) else None
    locked_ids = catalog.get("locked_peer_listing_ids")
    cohort_ids = catalog.get("cohort_listing_ids")
    manifest = calculation.get("source_manifest")
    mapping_authority = catalog.get("sec_mapping_authority")
    raw_fcf_authority = catalog.get("raw_fcf_authority")
    if (
        not isinstance(facts, list)
        or not isinstance(selected, list)
        or not isinstance(assessed, list)
        or not isinstance(target_id, str)
        or not isinstance(locked_ids, list)
        or not isinstance(cohort_ids, list)
        or not isinstance(manifest, list)
        or not isinstance(mapping_authority, dict)
        or not isinstance(raw_fcf_authority, list)
        or any(
            not isinstance(value, str) for value in (*selected, *assessed, *locked_ids, *cohort_ids)
        )
        or len(selected) != len(set(selected))
        or len(assessed) != len(set(assessed))
        or set(selected) & set(assessed)
    ):
        raise ValueError("Long-v4 selected/assessed evidence partitions are malformed")
    fact_by_id: dict[str, dict[str, Any]] = {}
    allowed_owners = {target_id, *locked_ids}
    manifest_ids = [
        item.get("id")
        for item in manifest
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if len(manifest_ids) != len(manifest) or len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("Long-v4 source manifest contains duplicate or malformed identities")
    manifest_set = set(manifest_ids)
    raw_fact_owners = {
        item.get("id"): item.get("owner_listing_id")
        for item in facts
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and isinstance(item.get("owner_listing_id"), str)
    }
    mapping_asset = mapping_authority.get("mapping_asset")
    mapping_rows = mapping_authority.get("cohort_rows")
    if (
        mapping_authority.get("sec_cik_config_version") != LONG_V4_SEC_CIK_CONFIG_VERSION
        or mapping_authority.get("sec_cik_config_file_sha256") != LONG_V4_SEC_CIK_CONFIG_FILE_SHA256
        or mapping_authority.get("sec_cik_config_hash") != LONG_V4_SEC_CIK_CONFIG_HASH
        or mapping_authority.get("sec_mapping_source_sha256") != LONG_V4_SEC_MAPPING_SOURCE_SHA256
        or mapping_authority.get("exchange_to_mic_rule")
        != [{"exchange": exchange, "mic": mic} for exchange, mic in SEC_EXCHANGE_MIC_RULE]
        or not isinstance(mapping_asset, dict)
        or mapping_asset.get("id") not in manifest_set
        or mapping_asset.get("provider") != "sec"
        or mapping_asset.get("kind") != MAPPING_KIND
        or mapping_asset.get("subject") != MAPPING_SUBJECT
        or mapping_asset.get("sha256") != LONG_V4_SEC_MAPPING_SOURCE_SHA256
        or not isinstance(mapping_rows, list)
        or [row.get("listing_id") for row in mapping_rows if isinstance(row, dict)] != cohort_ids
    ):
        raise ValueError("Long-v4 SEC mapping authority is malformed")
    mapping_identity_values: dict[tuple[str, ...], set[tuple[object, ...]]] = {
        fields: set()
        for fields in (
            ("listing_id",),
            ("company_id",),
            ("authoritative_cik",),
            ("config_official_ticker",),
            ("raw_ticker", "raw_cik", "raw_exchange"),
        )
    }
    for row in mapping_rows:
        if (
            not isinstance(row, dict)
            or any(
                not isinstance(row.get(field), str) or not row[field]
                for field in (
                    "listing_id",
                    "company_id",
                    "provider_symbol",
                    "listing_ticker",
                    "listing_exchange_mic",
                    "authoritative_cik",
                    "config_symbol",
                    "config_official_ticker",
                    "config_exchange",
                    "raw_ticker",
                    "raw_cik",
                    "raw_exchange",
                )
            )
            or not (
                row["provider_symbol"]
                == row["listing_ticker"]
                == row["config_symbol"]
                == row["config_official_ticker"]
                == row["raw_ticker"]
            )
            or row["authoritative_cik"] != row["raw_cik"]
        ):
            raise ValueError("Long-v4 SEC cohort mapping row is malformed")
        try:
            verify_sec_exchange_mic(
                exchange=row["config_exchange"],
                mic=row["listing_exchange_mic"],
            )
        except SecDerivationError:
            raise ValueError("Long-v4 SEC cohort mapping exchange is malformed") from None
        if row["raw_exchange"] != row["config_exchange"]:
            raise ValueError("Long-v4 SEC cohort mapping exchange is malformed")
        for fields, identities in mapping_identity_values.items():
            identity = tuple(row[field] for field in fields)
            if identity in identities:
                raise ValueError("Long-v4 SEC cohort mapping is not one-to-one")
            identities.add(identity)

    raw_by_owner: dict[str, dict[str, Any]] = {}
    for authority in raw_fcf_authority:
        if not isinstance(authority, dict):
            raise ValueError("Long-v4 raw FCF authority is malformed")
        owner_id = authority.get("owner_listing_id")
        sources = authority.get("sources")
        derivations = authority.get("derivations")
        if (
            not isinstance(owner_id, str)
            or owner_id in raw_by_owner
            or owner_id not in allowed_owners
            or authority.get("status")
            not in {"absent", "present_complete", "present_normalization_incomplete"}
            or not isinstance(sources, list)
            or not isinstance(derivations, list)
        ):
            raise ValueError("Long-v4 raw FCF authority is malformed")
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("Long-v4 raw FCF source authority is malformed")
            source_assets = (
                source.get("companyfacts_asset"),
                source.get("current_submissions_asset"),
            )
            histories = source.get("history_assets")
            if (
                any(not isinstance(item, dict) for item in source_assets)
                or not isinstance(histories, list)
                or any(not isinstance(item, dict) for item in histories)
                or any(
                    item.get("id") not in manifest_set
                    for item in (*source_assets, *histories)
                    if isinstance(item, dict)
                )
            ):
                raise ValueError("Long-v4 raw FCF source closure is malformed")
        incomplete = False
        for derivation in derivations:
            matching_fact_ids = (
                derivation.get("matching_normalized_fact_ids")
                if isinstance(derivation, dict)
                else None
            )
            normalization_issue = (
                derivation.get("normalization_issue") if isinstance(derivation, dict) else None
            )
            if (
                not isinstance(derivation, dict)
                or derivation.get("concept") not in _FCF_SOURCE_CONCEPTS
                or derivation.get("period_type") != FundamentalFact.PeriodType.DURATION
                or derivation.get("status") not in {"normalized", "normalization_incomplete"}
                or not isinstance(matching_fact_ids, list)
                or any(
                    fact_id not in raw_fact_owners or raw_fact_owners[fact_id] != owner_id
                    for fact_id in matching_fact_ids
                )
                or (
                    derivation.get("status") == "normalized"
                    and (not matching_fact_ids or normalization_issue is not None)
                )
                or (
                    derivation.get("status") == "normalization_incomplete"
                    and (
                        matching_fact_ids
                        or not isinstance(normalization_issue, str)
                        or not normalization_issue
                    )
                )
            ):
                raise ValueError("Long-v4 raw FCF derivation authority is malformed")
            incomplete |= derivation["status"] == "normalization_incomplete"
        expected_status = (
            "absent"
            if not derivations
            else ("present_normalization_incomplete" if incomplete else "present_complete")
        )
        if authority["status"] != expected_status:
            raise ValueError("Long-v4 raw FCF status is inconsistent")
        raw_by_owner[owner_id] = authority
    for item in facts:
        if not isinstance(item, dict):
            raise ValueError("Long-v4 fact catalog entry is malformed")
        fact_id = item.get("id")
        owner_id = item.get("owner_listing_id")
        if (
            not isinstance(fact_id, str)
            or fact_id in fact_by_id
            or owner_id not in allowed_owners
            or item.get("selection_status")
            != ("selected_formula_input" if fact_id in selected else "assessed")
            or any(
                not isinstance(item.get(field), str) or item[field] not in manifest_set
                for field in (
                    "source_asset_id",
                    "filing_evidence_asset_id",
                    "submissions_context_asset_id",
                )
            )
        ):
            raise ValueError("Long-v4 fact catalog ownership or source closure is malformed")
        event = item.get("correction_observation_event")
        revision = item.get("source_revision")
        if (
            not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision < 1
            or (revision == 1 and event is not None)
            or (
                revision > 1
                and (
                    not isinstance(event, dict)
                    or set(event)
                    != {
                        "id",
                        "provider",
                        "kind",
                        "subject",
                        "source_asset_id",
                        "content_sha256",
                        "observed_at",
                        "recorded_at",
                    }
                    or event.get("provider") != "sec"
                    or event.get("kind") != COMPANYFACTS_KIND
                    or event.get("source_asset_id") != item.get("source_asset_id")
                )
            )
        ):
            raise ValueError("Long-v4 correction observation event is malformed")
        fact_by_id[fact_id] = item
    if set(fact_by_id) != set(selected) | set(assessed):
        raise ValueError("Long-v4 selected/assessed evidence does not exactly cover the catalog")

    split_basis = calculation.get("split_basis")
    _validate_split_basis_document(
        split_basis,
        owner_listing_id=target_id,
        fact_by_id=fact_by_id,
        reason_code=calculation.get("insufficiency_code"),
    )
    candidates = calculation.get("locked_peer_candidates")
    peer_set = calculation.get("peer_set")
    if not isinstance(candidates, list) or not isinstance(peer_set, list):
        raise ValueError("Long-v4 peer assessment documents are malformed")
    candidate_by_listing: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Long-v4 peer candidate is malformed")
        listing_id = candidate.get("listing_id")
        if (
            not isinstance(listing_id, str)
            or listing_id in candidate_by_listing
            or listing_id not in cohort_ids
        ):
            raise ValueError("Long-v4 peer candidate identity is malformed")
        split = candidate.get("split_basis")
        if candidate.get("status") == "identity_candidate_unassessed":
            if split is not None:
                raise ValueError("Long-v4 unassessed peer has a split assessment")
        else:
            _validate_split_basis_document(
                split,
                owner_listing_id=listing_id,
                fact_by_id=fact_by_id,
                reason_code=candidate.get("reason_code"),
            )
        candidate_by_listing[listing_id] = candidate
    assessed_owner_ids = {
        target_id,
        *(
            listing_id
            for listing_id, candidate in candidate_by_listing.items()
            if candidate.get("status") != "identity_candidate_unassessed"
        ),
    }
    expected_raw_owner_order = [
        listing_id for listing_id in cohort_ids if listing_id in assessed_owner_ids
    ]
    evidence_selection = calculation.get("evidence_selection")
    target_selection = (
        evidence_selection.get("target") if isinstance(evidence_selection, dict) else None
    )
    peer_selections = (
        evidence_selection.get("locked_peers") if isinstance(evidence_selection, dict) else None
    )
    if (
        list(raw_by_owner) != expected_raw_owner_order
        or not isinstance(target_selection, dict)
        or target_selection.get("raw_fcf_authority") != raw_by_owner.get(target_id)
        or target_selection.get("fcf_raw_source_evidence_present")
        is not (
            raw_by_owner.get(target_id, {}).get("status") != "absent"
            if target_id in raw_by_owner
            else None
        )
        or not isinstance(peer_selections, dict)
        or any(
            not isinstance(peer_selections.get(owner_id), dict)
            or peer_selections[owner_id].get("raw_fcf_authority") != raw_by_owner.get(owner_id)
            for owner_id in assessed_owner_ids - {target_id}
        )
    ):
        raise ValueError("Long-v4 raw FCF evidence selection is inconsistent")
    selected_peer_ids: list[str] = []
    for peer in peer_set:
        if not isinstance(peer, dict) or not isinstance(peer.get("listing_id"), str):
            raise ValueError("Long-v4 selected peer is malformed")
        listing_id = peer["listing_id"]
        candidate = candidate_by_listing.get(listing_id)
        if (
            candidate is None
            or candidate.get("selected_peer") is not True
            or peer.get("split_basis") != candidate.get("split_basis")
        ):
            raise ValueError("Long-v4 selected peer assessment copy is inconsistent")
        selected_peer_ids.append(listing_id)
    if len(selected_peer_ids) != len(set(selected_peer_ids)) or set(selected_peer_ids) != {
        listing_id
        for listing_id, candidate in candidate_by_listing.items()
        if candidate.get("selected_peer") is True
    }:
        raise ValueError("Long-v4 selected and assessed peer partitions are inconsistent")

    classifications = catalog.get("classifications")
    prices = catalog.get("prices")
    if not isinstance(classifications, list) or not isinstance(prices, list):
        raise ValueError("Long-v4 catalog source collections are malformed")
    for classification in classifications:
        if (
            not isinstance(classification, dict)
            or classification.get("owner_listing_id")
            not in set(catalog.get("cohort_listing_ids", []))
            or classification.get("source_asset_id") not in manifest_set
        ):
            raise ValueError("Long-v4 classification catalog closure is malformed")
    for price in prices:
        if not isinstance(price, dict):
            raise ValueError("Long-v4 price catalog entry is malformed")
        canonical = canonical_long_v4_price(price.get("value"))
        valuation = _exact_positive_decimal(price.get("valuation_value"))
        if (
            price.get("value") != format(canonical, ".6f")
            or price.get("ledger_value") != price.get("value")
            or canonical_long_v4_price(valuation) != canonical
            or price.get("valuation_value") != _decimal_text(valuation)
            or price.get("valuation_source") != "normalized_parquet_target_close"
            or price.get("native_price") != price.get("valuation_value")
            or price.get("normalized_asset_id") not in manifest_set
            or price.get("raw_asset_id") not in manifest_set
        ):
            raise ValueError("Long-v4 price catalog closure is malformed")


def _validate_split_basis_document(
    raw: object,
    *,
    owner_listing_id: str,
    fact_by_id: dict[str, dict[str, Any]],
    reason_code: object,
) -> None:
    if not isinstance(raw, dict) or "verified_through" in raw:
        raise ValueError("Long-v4 split-basis assessment is malformed")
    checks = raw.get("checks")
    relevant = raw.get("relevant_fact_ids")
    if not isinstance(checks, list) or not isinstance(relevant, list):
        raise ValueError("Long-v4 split-basis evidence is malformed")
    union: list[str] = []
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("relevant_fact_ids"), list):
            raise ValueError("Long-v4 split-basis check is malformed")
        ids = check["relevant_fact_ids"]
        if len(ids) != len(set(ids)) or any(not isinstance(value, str) for value in ids):
            raise ValueError("Long-v4 split-basis check has duplicate evidence")
        union.extend(ids)
    expected_union = list(_dedupe_text(union))
    if (
        relevant != expected_union
        or len(relevant) != len(set(relevant))
        or any(
            fact_id not in fact_by_id
            or fact_by_id[fact_id].get("owner_listing_id") != owner_listing_id
            for fact_id in relevant
        )
    ):
        raise ValueError("Long-v4 split-basis evidence ownership is malformed")
    # Every covered incompatibility must end with the check that actually
    # failed; earlier compatible checks remain in source order.
    if raw.get("assessment_status") == "incompatible_or_unverified":
        if not checks:
            raise ValueError("Long-v4 incompatible split basis has no failing check")
        failure = checks[-1]
        if (
            failure.get("outcome") != "incompatible"
            or failure.get("assessed_through") != raw.get("assessed_through")
            or not isinstance(failure.get("compared_fields"), list)
            or not failure["compared_fields"]
            or not isinstance(failure.get("expected"), dict)
            or not isinstance(failure.get("actual"), dict)
        ):
            raise ValueError("Long-v4 failing split-basis check is incomplete")
        normalized_code = reason_code.removeprefix("fcf_") if isinstance(reason_code, str) else None
        expected_checks = {
            "ttm_period_mismatch": "ttm_entity_share_period_compatibility",
            "annual_alias_or_unit_incompatible": ("annual_entity_share_signature_compatibility"),
            "annual_duration_incompatible": "annual_duration_compatibility",
            "annual_unit_incompatible": "annual_required_unit_compatibility",
            "reported_eps_nonfinite": "reported_diluted_eps_finiteness",
            "reported_eps_reconciliation_failed": "reported_diluted_eps_reconciliation",
            "annual_period_gap": "adjacent_annual_period_continuity",
            "annual_share_continuity_failed": ("adjacent_annual_diluted_share_continuity"),
            "annual_to_ttm_alias_or_unit_incompatible": ("annual_to_ttm_alias_unit_compatibility"),
            "annual_to_ttm_derivation_incompatible": ("annual_to_ttm_derivation_compatibility"),
            "annual_to_ttm_period_incompatible": "annual_to_ttm_period_overlap",
            "annual_to_ttm_share_continuity_failed": (
                "ttm_to_latest_annual_diluted_share_continuity"
            ),
        }
        expected_check = (
            expected_checks.get(normalized_code) if normalized_code is not None else None
        )
        if expected_check is not None and failure.get("check") != expected_check:
            raise ValueError("Long-v4 failing split-basis check does not match its reason")


def _validate_successful_path_math(calculation: dict[str, Any]) -> None:
    inputs = calculation.get("formula_inputs")
    paths = calculation.get("scenario_paths")
    peer_set = calculation.get("peer_set")
    if (
        not isinstance(inputs, dict)
        or not isinstance(paths, dict)
        or not isinstance(peer_set, list)
    ):
        raise ValueError("Long-v4 successful formula inputs are malformed")
    raw_growth = inputs.get("entity_growth_raw")
    capped_growth = inputs.get("entity_growth_capped")
    raw_share_changes = inputs.get("diluted_share_changes_raw")
    if (
        not isinstance(raw_growth, list)
        or len(raw_growth) != 3
        or not isinstance(capped_growth, list)
        or len(capped_growth) != 3
        or not isinstance(raw_share_changes, list)
        or len(raw_share_changes) != 3
    ):
        raise ValueError("Long-v4 growth/dilution observations are incomplete")
    annual_metrics = inputs.get("annual_entity_metrics")
    annual_shares = inputs.get("annual_diluted_shares")
    if (
        not isinstance(annual_metrics, list)
        or len(annual_metrics) != 4
        or not isinstance(annual_shares, list)
        or len(annual_shares) != 4
    ):
        raise ValueError("Long-v4 annual entity/share levels are incomplete")
    expected_raw_growth = [
        float(current) / float(previous) - 1.0
        for previous, current in zip(
            annual_metrics,
            annual_metrics[1:],
            strict=False,
        )
    ]
    expected_raw_share_changes = [
        float(current) / float(previous) - 1.0
        for previous, current in zip(
            annual_shares,
            annual_shares[1:],
            strict=False,
        )
    ]
    if not _payload_close(raw_growth, expected_raw_growth) or not _payload_close(
        raw_share_changes, expected_raw_share_changes
    ):
        raise ValueError("Long-v4 raw growth/dilution observations are inconsistent")
    expected_capped_growth = [_clamp(float(value), -0.20, 0.25) for value in raw_growth]
    target_growth = statistics.median(expected_capped_growth)
    dilution_base = _clamp(
        max(0.0, statistics.median(float(value) for value in raw_share_changes)),
        0.0,
        0.15,
    )
    peer_growth_values = [
        float(peer["entity_growth_estimate"])
        for peer in peer_set
        if isinstance(peer, dict) and "entity_growth_estimate" in peer
    ]
    peer_multiple_values = [
        float(peer["current_multiple_capped"])
        for peer in peer_set
        if isinstance(peer, dict) and "current_multiple_capped" in peer
    ]
    if (
        len(peer_growth_values) != len(peer_set)
        or len(peer_multiple_values) != len(peer_set)
        or not peer_set
    ):
        raise ValueError("Long-v4 selected peer formula inputs are incomplete")
    for peer in peer_set:
        assert isinstance(peer, dict)
        peer_raw_growth = peer.get("entity_growth_raw")
        if not isinstance(peer_raw_growth, list) or len(peer_raw_growth) != 3:
            raise ValueError("Long-v4 peer growth observations are incomplete")
        expected_peer_growth = statistics.median(
            _clamp(float(value), -0.20, 0.25) for value in peer_raw_growth
        )
        if not _payload_close(peer.get("entity_growth_estimate"), expected_peer_growth):
            raise ValueError("Long-v4 peer entity-growth estimate is inconsistent")
    peer_growth = _clamp(statistics.median(peer_growth_values), -0.15, 0.25)
    peer_multiple = statistics.median(peer_multiple_values)
    scalar_expectations = {
        "target_entity_growth": target_growth,
        "dilution_base": dilution_base,
        "peer_entity_growth": peer_growth,
        "peer_multiple": peer_multiple,
    }
    if not _payload_close(
        {key: inputs.get(key) for key in scalar_expectations},
        scalar_expectations,
    ) or not _payload_close(inputs.get("entity_growth_capped"), expected_capped_growth):
        raise ValueError("Long-v4 persisted growth/dilution medians are inconsistent")

    family = calculation.get("metric_family")
    if family == "fcf_per_share":
        multiple_minimum, multiple_maximum = 3.0, 60.0
    elif family == "net_income_per_share":
        multiple_minimum, multiple_maximum = 5.0, 50.0
    else:
        raise ValueError("Long-v4 successful metric family is invalid")
    current_per_share = float(inputs["current_per_share_metric"])
    current_multiple_raw = float(inputs["current_multiple_raw"])
    current_multiple_capped = float(inputs["current_multiple_capped_reversion_anchor"])
    current_price_valuation = _exact_positive_decimal(inputs.get("current_price_valuation"))
    current_price_ledger = canonical_long_v4_price(inputs.get("current_price_ledger"))
    current_entity_exact = _exact_positive_decimal(inputs.get("current_entity_metric"))
    current_shares_exact = _exact_positive_decimal(
        inputs.get("current_weighted_average_diluted_shares")
    )
    current_per_share_exact = _exact_positive_decimal(inputs.get("current_per_share_exact"))
    current_multiple_exact = _exact_positive_decimal(inputs.get("current_multiple_exact"))
    direct_multiple_exact = current_price_valuation * current_shares_exact / current_entity_exact
    if (
        inputs.get("current_price_valuation") != _decimal_text(current_price_valuation)
        or inputs.get("current_price_ledger") != format(current_price_ledger, ".6f")
        or canonical_long_v4_price(current_price_valuation) != current_price_ledger
        or not math.isclose(
            float(inputs["current_price"]),
            float(current_price_valuation),
            rel_tol=0,
            abs_tol=0,
        )
        or inputs.get("current_per_share_exact") != _decimal_text(current_per_share_exact)
        or inputs.get("current_multiple_exact") != _decimal_text(current_multiple_exact)
        or current_multiple_exact != direct_multiple_exact
        or current_multiple_exact < Decimal(str(multiple_minimum))
        or not math.isclose(
            current_per_share,
            float(current_per_share_exact),
            rel_tol=0,
            abs_tol=RETURN_IDENTITY_TOLERANCE,
        )
        or not math.isclose(
            current_multiple_raw,
            float(current_multiple_exact),
            rel_tol=0,
            abs_tol=RETURN_IDENTITY_TOLERANCE,
        )
        or not math.isclose(
            current_per_share,
            float(inputs["current_entity_metric"])
            / float(inputs["current_weighted_average_diluted_shares"]),
            rel_tol=0,
            abs_tol=RETURN_IDENTITY_TOLERANCE,
        )
        or not math.isclose(
            current_multiple_raw,
            float(direct_multiple_exact),
            rel_tol=0,
            abs_tol=RETURN_IDENTITY_TOLERANCE,
        )
        or not math.isclose(
            current_multiple_capped,
            _clamp(current_multiple_raw, multiple_minimum, multiple_maximum),
            rel_tol=0,
            abs_tol=RETURN_IDENTITY_TOLERANCE,
        )
    ):
        raise ValueError("Long-v4 current level/multiple identity is inconsistent")
    scenario_constants = {
        "bear": (-0.04, 1.25, 0.80),
        "base": (0.0, 1.0, 1.0),
        "bull": (0.03, 0.75, 1.15),
    }
    fade_path = (0.80, 0.60, 0.40, 0.20, 0.00)
    reversion_path = (0.14, 0.28, 0.42, 0.56, 0.70)
    expected_paths: dict[str, Any] = {}
    for name in LONG_SCENARIOS:
        delta, dilution_multiplier, peer_multiple_multiplier = scenario_constants[name]
        scenario_target_growth = _clamp(target_growth + delta, -0.20, 0.25)
        scenario_peer_growth = _clamp(peer_growth + delta, -0.15, 0.25)
        blended_growth = _clamp(
            0.5 * scenario_target_growth + 0.5 * scenario_peer_growth,
            -0.15,
            0.25,
        )
        dilution = _clamp(dilution_base * dilution_multiplier, 0.0, 0.15)
        destination_multiple = _clamp(
            peer_multiple * peer_multiple_multiplier,
            multiple_minimum,
            multiple_maximum,
        )
        cumulative_entity_factor = 1.0
        cumulative_per_share_factor = 1.0
        expected_years: list[dict[str, Any]] = []
        for year, (fade, reversion) in enumerate(
            zip(fade_path, reversion_path, strict=True),
            start=1,
        ):
            entity_growth = fade * blended_growth + (1.0 - fade) * 0.025
            per_share_growth = (1.0 + entity_growth) / (1.0 + dilution) - 1.0
            cumulative_entity_factor *= 1.0 + entity_growth
            cumulative_per_share_factor *= 1.0 + per_share_growth
            projected_per_share = current_per_share * cumulative_per_share_factor
            multiple = math.exp(
                (1.0 - reversion) * math.log(current_multiple_capped)
                + reversion * math.log(destination_multiple)
            )
            projected_price = projected_per_share * multiple
            cumulative_return = cumulative_per_share_factor * multiple / current_multiple_raw - 1.0
            level_return_check = projected_price / float(current_price_valuation) - 1.0
            if not math.isclose(
                level_return_check,
                cumulative_return,
                rel_tol=0,
                abs_tol=RETURN_IDENTITY_TOLERANCE,
            ):
                raise ValueError("Long-v4 replay level and direct-multiple identities diverged")
            expected_years.append(
                {
                    "year": year,
                    "fade": fade,
                    "multiple_reversion": reversion,
                    "entity_growth": entity_growth,
                    "dilution_rate": dilution,
                    "per_share_growth": per_share_growth,
                    "cumulative_entity_factor": cumulative_entity_factor,
                    "cumulative_per_share_factor": cumulative_per_share_factor,
                    "projected_per_share_metric": projected_per_share,
                    "multiple": multiple,
                    "projected_price": projected_price,
                    "cumulative_price_return": cumulative_return,
                    "return_from_level_identity": cumulative_return,
                    "return_from_factor_identity": cumulative_return,
                }
            )
        expected_paths[name] = {
            "scenario": name,
            "target_entity_growth": scenario_target_growth,
            "peer_entity_growth": scenario_peer_growth,
            "blended_entity_growth": blended_growth,
            "dilution_base": dilution_base,
            "dilution_multiplier": dilution_multiplier,
            "dilution_rate": dilution,
            "peer_multiple_raw_median": peer_multiple,
            "peer_multiple_multiplier": peer_multiple_multiplier,
            "peer_multiple_destination": destination_multiple,
            "current_multiple_return_denominator": current_multiple_raw,
            "current_multiple_reversion_anchor": current_multiple_capped,
            "years": expected_years,
        }
    if not _payload_close(paths, expected_paths):
        raise ValueError("Long-v4 five-year scenario path arithmetic is inconsistent")
    expected_returns = {
        horizon: {
            name: expected_paths[name]["years"][years - 1]["cumulative_price_return"]
            for name in LONG_SCENARIOS
        }
        for horizon, years in (("3y", 3), ("5y", 5))
    }
    expected_annualized = {
        horizon: {
            name: (1.0 + value) ** (1.0 / years) - 1.0
            for name, value in expected_returns[horizon].items()
        }
        for horizon, years in (("3y", 3), ("5y", 5))
    }
    if not _payload_close(calculation.get("annualized_returns"), expected_annualized):
        raise ValueError("Long-v4 annualized return views are inconsistent")


def _payload_close(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(
                float(actual),
                expected,
                rel_tol=0,
                abs_tol=RETURN_IDENTITY_TOLERANCE,
            )
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(_payload_close(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _payload_close(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    return bool(actual == expected)


class LongV4PersistedAuthorityError(ValueError):
    """Persisted database authority disagrees with a long-v4 calculation."""


_AssessedV4Owner = tuple[Listing, CompanyClassificationObservation | None]


def _validate_long_v4_persisted_authority(
    *,
    calculation: Mapping[str, Any],
    target_listing_id: UUID,
    universe_snapshot_id: UUID,
    analysis_run_id: UUID | None,
    target_date: date,
    data_cutoff: datetime,
    generated_at: datetime,
    prediction_id: UUID | None = None,
    prediction_horizon: str | None = None,
    prediction_model_version: str | None = None,
    prediction_scenario_values: tuple[Decimal | None, Decimal | None, Decimal | None] | None = None,
) -> tuple[_AssessedV4Owner, ...]:
    """Validate one V4 catalog from independently selected database authority.

    This boundary is intentionally DB-only. Physical bytes and model
    recalculation remain the responsibility of the existing producer replay.
    """
    from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis

    run_query = AnalysisRun.objects.filter(
        universe_snapshot_id=universe_snapshot_id,
        target_date=target_date,
        data_cutoff=data_cutoff,
        generated_at=generated_at,
    )
    if analysis_run_id is not None:
        run_query = run_query.filter(pk=analysis_run_id)
    runs = list(run_query.select_related("universe_snapshot")[:2])
    if len(runs) != 1:
        raise LongV4PersistedAuthorityError("Long-v4 analysis-run authority is ambiguous")
    run = runs[0]
    if (
        run.issued_on_time is not False
        or run.config_version != LONG_V4_SCORING_VERSION
        or run.config_hash != LONG_V4_SCORING_CONFIG_HASH
    ):
        raise LongV4PersistedAuthorityError(
            "Long-v4 parent analysis-run admission identity is incompatible"
        )
    snapshot = run.universe_snapshot
    if (
        snapshot.pk != universe_snapshot_id
        or snapshot.grade != UniverseSnapshot.Grade.RESEARCH
        or snapshot.as_of_date != target_date
    ):
        raise LongV4PersistedAuthorityError("Long-v4 snapshot authority is incompatible")

    memberships = list(
        UniverseMembership.objects.filter(
            snapshot_id=universe_snapshot_id,
            eligible=True,
        ).select_related("listing__security__company")
    )
    analyses = list(
        StockAnalysis.objects.filter(run_id=run.pk).select_related("listing__security__company")
    )
    ordered_memberships = tuple(
        sorted(memberships, key=lambda item: (item.listing.ticker, str(item.listing_id)))
    )
    ordered_analyses = tuple(
        sorted(analyses, key=lambda item: (item.listing.ticker, str(item.listing_id)))
    )
    membership_ids = tuple(str(item.listing_id) for item in ordered_memberships)
    analysis_ids = tuple(str(item.listing_id) for item in ordered_analyses)
    analysis_set_is_valid = (
        analysis_ids == membership_ids
        if analysis_run_id is not None or len(analysis_ids) == len(membership_ids)
        else (
            bool(analysis_ids)
            and set(analysis_ids).issubset(membership_ids)
            and str(target_listing_id) in analysis_ids
        )
    )
    if (
        not membership_ids
        or not analysis_set_is_valid
        or len(membership_ids) != len(set(membership_ids))
        or str(target_listing_id) not in membership_ids
    ):
        raise LongV4PersistedAuthorityError(
            "Long-v4 run, membership, and analysis closures disagree"
        )
    successful_prediction = bool(
        prediction_scenario_values is not None
        and all(value is not None for value in prediction_scenario_values)
    )
    if prediction_scenario_values is not None and not (
        successful_prediction or all(value is None for value in prediction_scenario_values)
    ):
        raise LongV4PersistedAuthorityError("Long-v4 prediction scenarios are partial")
    if successful_prediction and (
        calculation.get("metric_family") not in _V4_METRIC_FAMILIES
        or calculation.get("insufficiency_code") is not None
        or calculation.get("insufficiency_reason") != ""
    ):
        raise LongV4PersistedAuthorityError(
            "Long-v4 successful prediction has no valid metric family"
        )
    if prediction_id is not None:
        if prediction_horizon not in {"3y", "5y"} or not prediction_model_version:
            raise LongV4PersistedAuthorityError("Long-v4 prediction identity is incomplete")
        pair = list(
            Prediction.objects.filter(
                analysis__run_id=run.pk,
                listing_id=target_listing_id,
                method_version=LONG_V4_VERSION,
                model_version=prediction_model_version,
                horizon__in=("3y", "5y"),
            ).order_by("horizon")
        )
        if (
            len(pair) != 2
            or {item.horizon for item in pair} != {"3y", "5y"}
            or prediction_id not in {item.pk for item in pair}
        ):
            raise LongV4PersistedAuthorityError("Long-v4 persisted pair is incomplete")
        current = next(item for item in pair if item.pk == prediction_id)
        if (
            current.horizon != prediction_horizon
            or (
                current.bear_return,
                current.base_return,
                current.bull_return,
            )
            != prediction_scenario_values
            or current.calculation != dict(calculation)
        ):
            raise LongV4PersistedAuthorityError(
                "Long-v4 current prediction differs from immutable authority"
            )
        allowed_pair_differences = {"forecast_horizon", "years", "selected_view"}
        common_calculations = [
            {
                key: value
                for key, value in item.calculation.items()
                if key not in allowed_pair_differences
            }
            for item in pair
        ]
        if common_calculations[0] != common_calculations[1]:
            raise LongV4PersistedAuthorityError("Long-v4 persisted pair common authority differs")
        if successful_prediction and any(
            any(
                value is None
                for value in (
                    item.bear_return,
                    item.base_return,
                    item.bull_return,
                )
            )
            for item in pair
        ):
            raise LongV4PersistedAuthorityError("Long-v4 successful persisted pair is incomplete")
    listings = tuple(item.listing for item in ordered_memberships)
    listing_by_id = {str(listing.pk): listing for listing in listings}
    if len({listing.security.company_id for listing in listings}) != len(listings):
        raise LongV4PersistedAuthorityError("Long-v4 cohort issuer ownership is ambiguous")

    catalog = calculation.get("evidence_catalog")
    if (
        not isinstance(catalog, dict)
        or catalog.get("schema_version") != EVIDENCE_CATALOG_SCHEMA_VERSION
        or catalog.get("target_listing_id") != str(target_listing_id)
        or catalog.get("cohort_listing_ids") != list(membership_ids)
    ):
        raise LongV4PersistedAuthorityError("Long-v4 catalog cohort is incomplete")

    mapping = catalog.get("sec_mapping_authority")
    if not isinstance(mapping, dict):
        raise LongV4PersistedAuthorityError("Long-v4 mapping authority is malformed")
    mapping_assets = list(
        DataAsset.objects.filter(
            provider="sec",
            kind=MAPPING_KIND,
            subject=MAPPING_SUBJECT,
            sha256=LONG_V4_SEC_MAPPING_SOURCE_SHA256,
            available_at__lte=generated_at,
            retrieved_at__lte=generated_at,
        ).order_by("pk")
    )
    if len(mapping_assets) != 1:
        raise LongV4PersistedAuthorityError("Long-v4 pinned mapping asset is ambiguous")
    mapping_asset = mapping_assets[0]
    inverse_exchange_rule = {mic: exchange for exchange, mic in SEC_EXCHANGE_MIC_RULE}
    expected_mapping_rows = []
    for listing in listings:
        exchange = inverse_exchange_rule.get(listing.exchange_mic)
        company = listing.security.company
        if exchange is None or not company.cik or listing.provider_symbol != listing.ticker:
            raise LongV4PersistedAuthorityError("Long-v4 cohort mapping identity is invalid")
        expected_mapping_rows.append(
            {
                "listing_id": str(listing.pk),
                "company_id": str(company.pk),
                "provider_symbol": listing.provider_symbol,
                "listing_ticker": listing.ticker,
                "listing_exchange_mic": listing.exchange_mic,
                "authoritative_cik": company.cik,
                "config_symbol": listing.ticker,
                "config_official_ticker": listing.ticker,
                "config_exchange": exchange,
                "raw_ticker": listing.ticker,
                "raw_cik": company.cik,
                "raw_exchange": exchange,
            }
        )
    if mapping != {
        "sec_cik_config_version": LONG_V4_SEC_CIK_CONFIG_VERSION,
        "sec_cik_config_file_sha256": LONG_V4_SEC_CIK_CONFIG_FILE_SHA256,
        "sec_cik_config_hash": LONG_V4_SEC_CIK_CONFIG_HASH,
        "sec_mapping_source_sha256": LONG_V4_SEC_MAPPING_SOURCE_SHA256,
        "mapping_asset": _asset_payload(mapping_asset),
        "exchange_to_mic_rule": [
            {"exchange": exchange, "mic": mic} for exchange, mic in SEC_EXCHANGE_MIC_RULE
        ],
        "cohort_rows": expected_mapping_rows,
    }:
        raise LongV4PersistedAuthorityError("Long-v4 mapping authority does not match the DB")

    price_entries = catalog.get("prices")
    if not isinstance(price_entries, list) or [
        item.get("listing_id") for item in price_entries if isinstance(item, dict)
    ] != list(membership_ids):
        raise LongV4PersistedAuthorityError("Long-v4 cohort price closure is incomplete")
    for listing, price in zip(listings, price_entries, strict=True):
        if not isinstance(price, dict):
            raise LongV4PersistedAuthorityError("Long-v4 price entry is malformed")
        price_candidates = list(
            DataAsset.objects.filter(
                provider="twelve_data",
                kind="price_history",
                subject=listing.provider_symbol,
                available_at__lte=generated_at,
                retrieved_at__lte=generated_at,
            ).order_by("-available_at", "-retrieved_at", "pk")
        )
        if not price_candidates:
            raise LongV4PersistedAuthorityError("Long-v4 normalized price is unavailable")
        maximum = (
            price_candidates[0].available_at,
            price_candidates[0].retrieved_at,
        )
        maxima = [
            asset
            for asset in price_candidates
            if (asset.available_at, asset.retrieved_at) == maximum
        ]
        if len(maxima) != 1:
            raise LongV4PersistedAuthorityError("Long-v4 normalized price is ambiguous")
        normalized = maxima[0]
        try:
            raw = raw_price_asset_for(normalized, cutoff=generated_at)
        except (RefreshVerificationError, TypeError, ValueError) as exc:
            raise LongV4PersistedAuthorityError(
                "Long-v4 persisted normalized/raw relation is invalid"
            ) from exc
        metadata = normalized.metadata if isinstance(normalized.metadata, dict) else {}
        resolved_mic = metadata.get("resolved_mic_code") or metadata.get("mic_code")
        if (
            price.get("owner_listing_id") != str(listing.pk)
            or price.get("listing_id") != str(listing.pk)
            or price.get("provider") != "twelve_data"
            or price.get("subject") != listing.provider_symbol
            or price.get("exchange_mic") != listing.exchange_mic
            or price.get("session_date") != target_date.isoformat()
            or price.get("currency") != "USD"
            or price.get("native_currency") != "USD"
            or price.get("fx_conversion") is not False
            or price.get("applied_fx_rate") is not None
            or listing.currency != "USD"
            or resolved_mic != listing.exchange_mic
            or metadata.get("currency") != "USD"
            or metadata.get("return_definition") != "split_adjusted_price_return"
            or metadata.get("dividends_included") is not False
            or price.get("normalized_asset") != _asset_payload(normalized)
            or price.get("raw_asset") != _asset_payload(raw)
            or price.get("normalized_asset_id") != str(normalized.pk)
            or price.get("normalized_asset_sha256") != normalized.sha256
            or price.get("raw_asset_id") != str(raw.pk)
            or price.get("raw_asset_sha256") != raw.sha256
        ):
            raise LongV4PersistedAuthorityError("Long-v4 persisted price authority differs")
    target_prices = [
        item
        for item in price_entries
        if isinstance(item, dict) and item.get("listing_id") == str(target_listing_id)
    ]
    if len(target_prices) != 1 or calculation.get("target_price") != target_prices[0]:
        raise LongV4PersistedAuthorityError("Long-v4 target price copy differs")

    classifications_by_company: dict[str, list[CompanyClassificationObservation]] = {}
    classification_rows = (
        CompanyClassificationObservation.objects.filter(
            company_id__in=[listing.security.company_id for listing in listings],
            provider="sec",
            scheme="sec_sic",
            available_at__lte=data_cutoff,
            ingested_at__lte=generated_at,
            source_asset__available_at__lte=generated_at,
            source_asset__retrieved_at__lte=generated_at,
        )
        .select_related("source_asset")
        .order_by("company_id", "available_at", "pk")
    )
    for observation in classification_rows:
        classifications_by_company.setdefault(str(observation.company_id), []).append(observation)
    classifications = {
        str(listing.pk): _select_classification(
            classifications_by_company.get(str(listing.security.company_id), [])
        )
        for listing in listings
    }
    selection = calculation.get("evidence_selection")
    stored_trace = selection.get("peer_lock") if isinstance(selection, dict) else None
    peer_attempted = True if successful_prediction else calculation.get("metric_family") is not None
    expected_lock: _PeerLock | None = None
    if peer_attempted:
        target_classification = classifications[str(target_listing_id)]
        if target_classification is None or _normalized_sic(target_classification.code) is None:
            raise LongV4PersistedAuthorityError("Long-v4 target classification is unavailable")
        try:
            expected_lock = _lock_peer_cohort(
                target_listing=listing_by_id[str(target_listing_id)],
                target_classification=target_classification,
                listings=listings,
                classifications=classifications,
                authoritative_ciks={
                    str(listing.pk): listing.security.company.cik for listing in listings
                },
                peer_config=LONG_V4_PEER_POLICY,
            )
        except ValueError as exc:
            raise LongV4PersistedAuthorityError("Long-v4 peer authority is incompatible") from exc
        if stored_trace != expected_lock.payload():
            raise LongV4PersistedAuthorityError("Long-v4 peer-lock trace differs")
    elif stored_trace is not None:
        raise LongV4PersistedAuthorityError("Long-v4 unexpected peer-lock trace")

    expected_locked_ids = (
        [str(listing.pk) for listing in expected_lock.members]
        if expected_lock is not None and expected_lock.status == "locked"
        else []
    )
    if catalog.get("locked_peer_listing_ids") != expected_locked_ids:
        raise LongV4PersistedAuthorityError("Long-v4 locked peer identities differ")
    expected_assessed_ids = {
        str(target_listing_id),
        *expected_locked_ids,
    }
    raw_authorities = catalog.get("raw_fcf_authority")
    if not isinstance(raw_authorities, list) or [
        item.get("owner_listing_id") for item in raw_authorities if isinstance(item, dict)
    ] != [listing_id for listing_id in membership_ids if listing_id in expected_assessed_ids]:
        raise LongV4PersistedAuthorityError("Long-v4 assessed raw SEC owners differ")
    raw_by_owner = {
        item["owner_listing_id"]: item
        for item in raw_authorities
        if isinstance(item, dict) and isinstance(item.get("owner_listing_id"), str)
    }
    window_start = target_date - timedelta(days=6 * 366)
    for owner_id in expected_assessed_ids:
        authority = raw_by_owner.get(owner_id)
        listing = listing_by_id[owner_id]
        if (
            not isinstance(authority, dict)
            or authority.get("company_id") != str(listing.security.company_id)
            or authority.get("authoritative_cik") != listing.security.company.cik
            or authority.get("window_start") != window_start.isoformat()
            or authority.get("window_end") != target_date.isoformat()
            or authority.get("data_cutoff") != data_cutoff.isoformat()
            or authority.get("decision_time") != generated_at.isoformat()
        ):
            raise LongV4PersistedAuthorityError("Long-v4 raw SEC owner authority differs")
        sources = authority.get("sources")
        if not isinstance(sources, list) or not sources:
            raise LongV4PersistedAuthorityError("Long-v4 raw SEC source closure is empty")
        for source_bundle in sources:
            if not isinstance(source_bundle, dict):
                raise LongV4PersistedAuthorityError("Long-v4 raw SEC source is malformed")
            asset_payloads = [
                source_bundle.get("companyfacts_asset"),
                source_bundle.get("current_submissions_asset"),
                *(source_bundle.get("history_assets") or ()),
            ]
            if any(not isinstance(item, dict) for item in asset_payloads):
                raise LongV4PersistedAuthorityError("Long-v4 raw SEC assets are malformed")
            try:
                asset_ids = [UUID(item["id"]) for item in asset_payloads]
            except (KeyError, TypeError, ValueError) as exc:
                raise LongV4PersistedAuthorityError(
                    "Long-v4 raw SEC asset identity is malformed"
                ) from exc
            persisted_assets = DataAsset.objects.in_bulk(asset_ids)
            if set(persisted_assets) != set(asset_ids):
                raise LongV4PersistedAuthorityError("Long-v4 raw SEC asset is missing")
            for asset_id, payload in zip(asset_ids, asset_payloads, strict=True):
                asset = persisted_assets[asset_id]
                if (
                    payload != _asset_payload(asset)
                    or asset.available_at > generated_at
                    or asset.retrieved_at > generated_at
                ):
                    raise LongV4PersistedAuthorityError("Long-v4 raw SEC asset payload differs")
            raw_events = source_bundle.get("visible_observation_events")
            if not isinstance(raw_events, list):
                raise LongV4PersistedAuthorityError("Long-v4 raw SEC events are malformed")
            for raw_event in raw_events:
                if not isinstance(raw_event, dict):
                    raise LongV4PersistedAuthorityError("Long-v4 raw SEC event is malformed")
                try:
                    event_id = UUID(str(raw_event.get("id")))
                except (TypeError, ValueError) as exc:
                    raise LongV4PersistedAuthorityError(
                        "Long-v4 raw SEC event identity is malformed"
                    ) from exc
                loaded_event = (
                    SourceObservationEvent.objects.select_related("source_asset")
                    .filter(pk=event_id)
                    .first()
                )
                if (
                    loaded_event is None
                    or loaded_event.observed_at > generated_at
                    or loaded_event.recorded_at > generated_at
                    or raw_event != _observation_event_payload(loaded_event)
                ):
                    raise LongV4PersistedAuthorityError("Long-v4 raw SEC event payload differs")
            expected_latest_event = raw_events[-1] if raw_events else None
            if source_bundle.get("latest_visible_observation_event") != expected_latest_event:
                raise LongV4PersistedAuthorityError("Long-v4 latest raw SEC event differs")

    expected_company_ids = {
        listing_by_id[listing_id].security.company_id for listing_id in expected_assessed_ids
    }
    expected_facts = list(
        FundamentalFact.objects.filter(
            company_id__in=expected_company_ids,
            provider="sec",
            concept__in=V4_CONCEPTS,
            period_end__gte=window_start,
            period_end__lte=target_date,
            available_at__lte=data_cutoff,
            ingested_at__lte=generated_at,
            source_asset__available_at__lte=generated_at,
            source_asset__retrieved_at__lte=generated_at,
        ).select_related("source_asset")
    )
    filing_assets = _filing_assets_for(expected_facts, decision_time=generated_at)
    contexts: dict[str, DataAsset] = {}
    correction_events: dict[str, SourceObservationEvent | None] = {}
    owner_by_company = {
        listing.security.company_id: listing
        for listing in listings
        if str(listing.pk) in expected_assessed_ids
    }
    for fact in expected_facts:
        raw_metadata = fact.source_asset.metadata
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        context_id = metadata.get("submissions_asset_id")
        context_sha = metadata.get("submissions_sha256")
        try:
            context_uuid = UUID(str(context_id))
        except (TypeError, ValueError) as exc:
            raise LongV4PersistedAuthorityError(
                "Long-v4 fact submissions context is malformed"
            ) from exc
        context = DataAsset.objects.filter(pk=context_uuid, sha256=context_sha).first()
        if (
            context is None
            or context.available_at > generated_at
            or context.retrieved_at > generated_at
        ):
            raise LongV4PersistedAuthorityError("Long-v4 fact submissions context is unavailable")
        contexts[str(fact.pk)] = context
        if fact.source_revision == 1:
            correction_events[str(fact.pk)] = None
            continue
        correction_candidates = list(
            SourceObservationEvent.objects.filter(
                provider="sec",
                kind=COMPANYFACTS_KIND,
                subject=owner_by_company[fact.company_id].security.company.cik,
                source_asset_id=fact.source_asset_id,
                content_sha256=fact.source_asset.sha256,
                observed_at__lte=data_cutoff,
                recorded_at__lte=generated_at,
            ).select_related("source_asset")
        )
        matching = [
            event
            for event in correction_candidates
            if fact.acceptance_at is not None
            and max(fact.acceptance_at, event.observed_at) == fact.available_at
        ]
        if len(matching) != 1:
            raise LongV4PersistedAuthorityError("Long-v4 fact correction event is ambiguous")
        correction_events[str(fact.pk)] = matching[0]

    expected_fact_payloads = []
    selected = catalog.get("selected_fact_ids")
    assessed = catalog.get("assessed_fact_ids")
    if (
        not isinstance(selected, list)
        or not isinstance(assessed, list)
        or len(selected) != len(set(selected))
        or len(assessed) != len(set(assessed))
        or set(selected) & set(assessed)
    ):
        raise LongV4PersistedAuthorityError("Long-v4 fact partitions are malformed")
    expected_ids = {str(fact.pk) for fact in expected_facts}
    if set(selected) | set(assessed) != expected_ids:
        raise LongV4PersistedAuthorityError("Long-v4 fact closure is incomplete")
    fact_authority = _SecRawAuthority(
        context_assets=contexts,
        correction_events=correction_events,
        by_listing_id={},
        latest_facts_by_listing_id={},
    )
    for fact in expected_facts:
        owner = owner_by_company.get(fact.company_id)
        if owner is None:
            raise LongV4PersistedAuthorityError("Long-v4 fact owner is not assessed")
        expected_fact_payloads.append(
            {
                "owner_listing_id": str(owner.pk),
                "selection_status": (
                    "selected_formula_input" if str(fact.pk) in selected else "assessed"
                ),
                **_fact_payload(
                    fact,
                    filing_assets,
                    raw_authority=fact_authority,
                    owner=owner,
                ),
            }
        )
    expected_fact_payloads.sort(
        key=lambda item: (
            item["owner_listing_id"],
            item["concept"],
            item["period_end"],
            item["period_identity"],
            item["id"],
        )
    )
    if catalog.get("facts") != expected_fact_payloads:
        raise LongV4PersistedAuthorityError("Long-v4 persisted fact payloads differ")

    expected_classification_owners = (
        set(membership_ids) if expected_lock is not None else {str(target_listing_id)}
    )
    expected_classifications = []
    for listing_id in membership_ids:
        classification = classifications[listing_id]
        if listing_id in expected_classification_owners and classification is not None:
            expected_classifications.append(
                {
                    "owner_listing_id": listing_id,
                    **_classification_payload(
                        classification,
                        owner=listing_by_id[listing_id],
                    ),
                }
            )
    if catalog.get("classifications") != expected_classifications:
        raise LongV4PersistedAuthorityError("Long-v4 persisted classification payloads differ")

    peer_candidates = calculation.get("locked_peer_candidates")
    peer_set = calculation.get("peer_set")
    if not isinstance(peer_candidates, list) or not isinstance(peer_set, list):
        raise LongV4PersistedAuthorityError("Long-v4 peer documents are malformed")
    if expected_lock is None:
        if peer_candidates or peer_set:
            raise LongV4PersistedAuthorityError("Long-v4 unexpected peer assessment")
    elif expected_lock.status == "no_floor":
        if peer_candidates != _unassessed_peer_candidates(expected_lock) or peer_set:
            raise LongV4PersistedAuthorityError("Long-v4 no-floor peers differ")
    else:
        expected_candidate_ids = [str(listing.pk) for listing in expected_lock.members]
        if [item.get("listing_id") for item in peer_candidates if isinstance(item, dict)] != (
            expected_candidate_ids
        ):
            raise LongV4PersistedAuthorityError("Long-v4 assessed peer order differs")
        for candidate, peer_listing in zip(
            peer_candidates,
            expected_lock.members,
            strict=True,
        ):
            classification = classifications[str(peer_listing.pk)]
            if (
                not isinstance(candidate, dict)
                or classification is None
                or candidate.get("company_id") != str(peer_listing.security.company_id)
                or candidate.get("classification_id") != str(classification.pk)
                or candidate.get("sic") != _normalized_sic(classification.code)
            ):
                raise LongV4PersistedAuthorityError("Long-v4 assessed peer identity is cross-wired")
        selected_peer_ids = [
            item.get("listing_id")
            for item in peer_candidates
            if isinstance(item, dict) and item.get("selected_peer") is True
        ]
        if [item.get("listing_id") for item in peer_set if isinstance(item, dict)] != (
            selected_peer_ids
        ):
            raise LongV4PersistedAuthorityError("Long-v4 selected peer order differs")

    try:
        _validate_catalog_assessment_contract(dict(calculation))
    except (TypeError, ValueError) as exc:
        raise LongV4PersistedAuthorityError(
            "Long-v4 structural evidence authority is malformed"
        ) from exc
    return tuple(
        (listing_by_id[listing_id], classifications[listing_id])
        for listing_id in membership_ids
        if listing_id in expected_assessed_ids
    )


def _validate_long_v4_raw_sec_replay(
    *,
    calculation: Mapping[str, Any],
    assessed_owners: tuple[_AssessedV4Owner, ...],
    target_date: date,
    data_cutoff: datetime,
    decision_time: datetime,
    store: AssetStore | None = None,
) -> None:
    """Physically replay only the retained assessed-owner raw SEC closure."""
    try:
        config = load_long_forecast_v4_config(long_forecast_v4_config_path())
        if config.peer != LONG_V4_PEER_POLICY:
            raise ValueError("peer policy")
        sec_config = load_long_v4_sec_fundamentals_config(config)
        listings = tuple(owner[0] for owner in assessed_owners)
        classifications = {
            str(listing.pk): classification for listing, classification in assessed_owners
        }
        company_ids = tuple(listing.security.company_id for listing in listings)
        facts = tuple(
            FundamentalFact.objects.filter(
                company_id__in=company_ids,
                provider="sec",
                available_at__lte=data_cutoff,
                ingested_at__lte=decision_time,
                source_asset__available_at__lte=decision_time,
                source_asset__retrieved_at__lte=decision_time,
            )
            .select_related("source_asset")
            .order_by(
                "company_id",
                "concept",
                "period_end",
                "available_at",
                "source_revision",
                "observation_hash",
            )
        )
        filing_assets = _filing_assets_for(list(facts), decision_time=decision_time)
        raw_authority = _validate_raw_sec_authority(
            listings=listings,
            facts=facts,
            filing_assets=filing_assets,
            classifications=classifications,
            config=sec_config,
            store=store or AssetStore(),
            data_cutoff=data_cutoff,
            decision_time=decision_time,
            authoritative_ciks={
                str(listing.pk): listing.security.company.cik for listing in listings
            },
            target_date=target_date,
        )
        catalog = calculation.get("evidence_catalog")
        stored = catalog.get("raw_fcf_authority") if isinstance(catalog, Mapping) else None
        expected = [raw_authority.by_listing_id[str(listing.pk)] for listing in listings]
        if stored != expected:
            raise ValueError("raw SEC authority mismatch")
    except (
        DataAsset.DoesNotExist,
        InvalidOperation,
        OSError,
        ObservationEvidenceError,
        RefreshVerificationError,
        SecDerivationError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        raise LongV4PersistedAuthorityError(
            "Long-v4 raw SEC physical replay is incompatible"
        ) from None


def validate_long_v4_evidence_authority(
    *,
    calculation: dict[str, Any],
    source_assets: tuple[DataAsset, ...],
    target_listing: Listing,
    universe_snapshot: UniverseSnapshot,
    eligible_memberships: tuple[UniverseMembership, ...],
    target_date: date,
    data_cutoff: datetime,
    decision_time: datetime,
    store: AssetStore | None,
) -> tuple[tuple[DataAsset, ...], dict[str, Any]]:
    """Re-resolve the complete evidence closure independently before insert."""
    if (
        calculation.get("schema_version") != CALCULATION_SCHEMA_VERSION
        or calculation.get("target_date") != target_date.isoformat()
        or data_cutoff > decision_time
    ):
        raise ValueError("Long-v4 evidence authority boundaries are malformed")
    target_payload = calculation.get("target")
    if target_payload != _listing_identity_payload(target_listing):
        raise ValueError("Long-v4 target identity does not match its authoritative listing")
    if (
        universe_snapshot.pk is None
        or universe_snapshot.grade != "research"
        or universe_snapshot.as_of_date != target_date
    ):
        raise ValueError("Long-v4 locked snapshot authority is incompatible")
    _validate_long_v4_persisted_authority(
        calculation=calculation,
        target_listing_id=target_listing.pk,
        universe_snapshot_id=universe_snapshot.pk,
        analysis_run_id=None,
        target_date=target_date,
        data_cutoff=data_cutoff,
        generated_at=decision_time,
    )
    persisted_eligible_ids = tuple(
        str(value)
        for value in UniverseMembership.objects.filter(
            snapshot=universe_snapshot,
            eligible=True,
        )
        .order_by("listing_id")
        .values_list("listing_id", flat=True)
    )
    supplied_membership_ids = tuple(
        sorted(
            str(membership.listing_id)
            for membership in eligible_memberships
            if membership.snapshot_id == universe_snapshot.pk and membership.eligible
        )
    )
    if (
        not persisted_eligible_ids
        or supplied_membership_ids != persisted_eligible_ids
        or len(eligible_memberships) != len(persisted_eligible_ids)
    ):
        raise ValueError("Long-v4 complete eligible membership closure is missing")
    listings = {
        str(membership.listing_id): membership.listing for membership in eligible_memberships
    }
    if (
        set(listings) != set(persisted_eligible_ids)
        or str(target_listing.pk) not in listings
        or any(listing.security.company.pk is None for listing in listings.values())
    ):
        raise ValueError("Long-v4 locked listing/security/company closure is incomplete")
    ordered_listings = tuple(
        sorted(listings.values(), key=lambda item: (item.ticker, str(item.pk)))
    )
    expected_cohort_ids = [str(listing.pk) for listing in ordered_listings]
    catalog = calculation.get("evidence_catalog")
    if (
        not isinstance(catalog, dict)
        or catalog.get("schema_version") != EVIDENCE_CATALOG_SCHEMA_VERSION
        or catalog.get("target_listing_id") != str(target_listing.pk)
        or catalog.get("cohort_listing_ids") != expected_cohort_ids
    ):
        raise ValueError("Long-v4 evidence catalog does not match the complete cohort")

    reader = AsOfData(decision_time, store)
    replay_config = load_long_forecast_v4_config(long_forecast_v4_config_path())
    mapping_authority = _resolve_sec_mapping_authority(
        listings=ordered_listings,
        config=replay_config,
        cik_config=load_long_v4_sec_cik_config(replay_config),
        asof=reader,
    )
    if catalog.get("sec_mapping_authority") != _sec_mapping_authority_payload(mapping_authority):
        raise ValueError("Long-v4 SEC mapping authority does not match its replay")
    company_ids = tuple(listing.security.company_id for listing in ordered_listings)
    classifications_by_company: dict[str, list[CompanyClassificationObservation]] = {}
    for observation in reader.company_classifications_for_companies(
        company_ids=company_ids,
        scheme="sec_sic",
        available_through=data_cutoff,
    ).filter(provider="sec"):
        classifications_by_company.setdefault(str(observation.company_id), []).append(observation)
    classifications = {
        str(listing.pk): _select_classification(
            classifications_by_company.get(str(listing.security.company_id), [])
        )
        for listing in ordered_listings
    }
    selection = calculation.get("evidence_selection")
    stored_trace = selection.get("peer_lock") if isinstance(selection, dict) else None
    target_classification = classifications[str(target_listing.pk)]
    if stored_trace is not None:
        if target_classification is None or _normalized_sic(target_classification.code) is None:
            raise ValueError("Long-v4 target classification authority is unavailable")
        expected_lock = _lock_peer_cohort(
            target_listing=target_listing,
            target_classification=target_classification,
            listings=ordered_listings,
            classifications=classifications,
            authoritative_ciks=mapping_authority.cik_by_listing_id,
            peer_config=LONG_V4_PEER_POLICY,
        )
        if stored_trace != expected_lock.payload():
            raise ValueError("Long-v4 peer-lock trace does not match cutoff-safe authority")
        peer_set = calculation.get("peer_set")
        selected_peer_ids = (
            [item.get("listing_id") for item in peer_set if isinstance(item, dict)]
            if isinstance(peer_set, list)
            else []
        )
        if len(selected_peer_ids) != len(peer_set or ()) or not set(selected_peer_ids).issubset(
            {str(listing.pk) for listing in expected_lock.members}
        ):
            raise ValueError("Long-v4 selected peer is outside the authoritative lock")

    canonical_prices = {
        str(listing.pk): _resolve_canonical_price(
            listing=listing,
            supplied_price=None,
            supplied_asset=None,
            asof=reader,
            target_date=target_date,
            expected_provider="twelve_data",
        )
        for listing in ordered_listings
    }
    replay_prices = {
        listing_id: float(price.valuation_value) for listing_id, price in canonical_prices.items()
    }
    replay_price_assets = {
        listing_id: price.normalized_asset for listing_id, price in canonical_prices.items()
    }
    replay_pair = build_long_forecasts_v4(
        listings=list(ordered_listings),
        current_prices=replay_prices,
        price_assets=replay_price_assets,
        asof=reader,
        data_cutoff=data_cutoff,
        target_date=target_date,
        config=replay_config,
    )[str(target_listing.pk)]
    validate_long_v4_forecast_pair(
        replay_pair,
        config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH,
    )
    if replay_pair["3y"].calculation != calculation:
        raise ValueError("Long-v4 calculation does not match an authoritative evidence replay")
    canonical_assets = replay_pair["3y"].source_assets
    if len(source_assets) != len(canonical_assets) or any(
        _asset_authority_tuple(supplied) != _asset_authority_tuple(canonical)
        for supplied, canonical in zip(source_assets, canonical_assets, strict=True)
    ):
        raise ValueError("Long-v4 supplied source closure does not match the replay")
    expected_manifest = [_asset_payload(asset) for asset in canonical_assets]
    if calculation.get("source_manifest") != expected_manifest:
        raise ValueError("Long-v4 source manifest does not match the authoritative replay")
    if list(_catalog_manifest_asset_ids(replay_pair["3y"].calculation["evidence_catalog"])) != [
        str(asset.pk) for asset in canonical_assets
    ]:
        raise ValueError("Long-v4 source manifest order is not catalog-derived")
    try:
        for asset in canonical_assets:
            read_checksummed_bytes(reader.store, asset)
    except RefreshVerificationError:
        raise ValueError("Long-v4 evidence closure failed physical verification") from None
    target_price = _catalog_price_for(
        replay_pair["3y"].calculation["evidence_catalog"],
        str(target_listing.pk),
    )
    return canonical_assets, target_price


def _evidence_catalog(
    *,
    target: _EntityAssessment,
    peer_lock: _PeerLock | None,
    peer_assessments: tuple[_EntityAssessment, ...],
    selected_peer_ids: set[str],
    filing_assets: dict[str, DataAsset],
    classifications: dict[str, CompanyClassificationObservation | None],
    cohort_prices: tuple[_CanonicalPrice, ...],
    raw_authority: _SecRawAuthority,
    mapping_authority: _SecMappingAuthority,
) -> dict[str, Any]:
    states = (target, *peer_assessments)
    state_by_listing_id = {str(state.listing.pk): state for state in states}
    assessed_states = tuple(
        state_by_listing_id[str(price.listing.pk)]
        for price in cohort_prices
        if str(price.listing.pk) in state_by_listing_id
    )
    facts: list[dict[str, Any]] = []
    selected_ids: list[str] = list(target.selected_fact_ids)
    assessed_ids: list[str] = list(target.assessed_fact_ids)
    for state in peer_assessments:
        if str(state.listing.pk) in selected_peer_ids:
            selected_ids.extend(state.selected_fact_ids)
            assessed_ids.extend(state.assessed_fact_ids)
        else:
            assessed_ids.extend(str(fact.pk) for fact in state.facts)
    selected_set = set(selected_ids)
    for state in states:
        for fact in state.facts:
            fact_id = str(fact.pk)
            facts.append(
                {
                    "owner_listing_id": str(state.listing.pk),
                    "selection_status": (
                        "selected_formula_input" if fact_id in selected_set else "assessed"
                    ),
                    **_fact_payload(
                        fact,
                        filing_assets,
                        raw_authority=raw_authority,
                        owner=state.listing,
                    ),
                }
            )
    facts.sort(
        key=lambda item: (
            item["owner_listing_id"],
            item["concept"],
            item["period_end"],
            item["period_identity"],
            item["id"],
        )
    )
    if len({item["id"] for item in facts}) != len(facts):
        raise ValueError("Long-v4 evidence catalog contains a duplicate fact")
    classification_listings = (
        [price.listing for price in cohort_prices] if peer_lock is not None else [target.listing]
    )
    classification_entries: list[dict[str, Any]] = []
    seen_classification_owners: set[str] = set()
    for listing in classification_listings:
        listing_id = str(listing.pk)
        classification = classifications.get(listing_id)
        if listing_id in seen_classification_owners or classification is None:
            continue
        seen_classification_owners.add(listing_id)
        classification_entries.append(
            {
                "owner_listing_id": listing_id,
                **_classification_payload(classification, owner=listing),
            }
        )
    prices = [_canonical_price_payload(price) for price in cohort_prices]
    return {
        "schema_version": EVIDENCE_CATALOG_SCHEMA_VERSION,
        "sec_mapping_authority": _sec_mapping_authority_payload(mapping_authority),
        "raw_fcf_authority": [
            raw_authority.by_listing_id[str(state.listing.pk)] for state in assessed_states
        ],
        "facts": facts,
        "classifications": classification_entries,
        "prices": prices,
        "selected_fact_ids": list(_dedupe_text(selected_ids)),
        "assessed_fact_ids": [
            fact_id for fact_id in _dedupe_text(assessed_ids) if fact_id not in selected_set
        ],
        "target_listing_id": str(target.listing.pk),
        "locked_peer_listing_ids": (
            [str(listing.pk) for listing in peer_lock.members]
            if peer_lock is not None and peer_lock.status == "locked"
            else []
        ),
        "cohort_listing_ids": [str(price.listing.pk) for price in cohort_prices],
    }


def _sec_mapping_authority_payload(
    authority: _SecMappingAuthority,
) -> dict[str, Any]:
    return {
        "sec_cik_config_version": authority.cik_config.config_version,
        "sec_cik_config_file_sha256": LONG_V4_SEC_CIK_CONFIG_FILE_SHA256,
        "sec_cik_config_hash": authority.cik_config.config_hash,
        "sec_mapping_source_sha256": authority.cik_config.source_sha256,
        "mapping_asset": _asset_payload(authority.asset),
        "exchange_to_mic_rule": [
            {"exchange": exchange, "mic": mic} for exchange, mic in SEC_EXCHANGE_MIC_RULE
        ],
        "cohort_rows": [dict(row) for row in authority.cohort_rows],
    }


def _catalog_manifest_asset_ids(catalog: dict[str, Any]) -> tuple[str, ...]:
    """Traverse schema-2 evidence once in the sole canonical manifest order."""
    asset_ids: list[str] = []

    def append_id(raw: object) -> None:
        if not isinstance(raw, str) or not raw:
            raise ValueError("Long-v4 evidence catalog contains a malformed asset identity")
        asset_ids.append(raw)

    mapping = catalog.get("sec_mapping_authority")
    mapping_asset = mapping.get("mapping_asset") if isinstance(mapping, dict) else None
    if not isinstance(mapping_asset, dict):
        raise ValueError("Long-v4 SEC mapping authority is malformed")
    append_id(mapping_asset.get("id"))
    raw_authorities = catalog.get("raw_fcf_authority")
    if not isinstance(raw_authorities, list):
        raise ValueError("Long-v4 raw FCF authority is malformed")
    for authority in raw_authorities:
        sources = authority.get("sources") if isinstance(authority, dict) else None
        if not isinstance(sources, list):
            raise ValueError("Long-v4 raw FCF source authority is malformed")
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("Long-v4 raw FCF source authority is malformed")
            companyfacts = source.get("companyfacts_asset")
            submissions = source.get("current_submissions_asset")
            histories = source.get("history_assets")
            if (
                not isinstance(companyfacts, dict)
                or not isinstance(submissions, dict)
                or not isinstance(histories, list)
                or any(not isinstance(item, dict) for item in histories)
            ):
                raise ValueError("Long-v4 raw FCF source closure is malformed")
            append_id(companyfacts.get("id"))
            append_id(submissions.get("id"))
            for history in histories:
                append_id(history.get("id"))
    facts = catalog.get("facts")
    classifications = catalog.get("classifications")
    prices = catalog.get("prices")
    if (
        not isinstance(facts, list)
        or not isinstance(classifications, list)
        or not isinstance(prices, list)
    ):
        raise ValueError("Long-v4 evidence catalog source collections are malformed")
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("Long-v4 fact catalog entry is malformed")
        append_id(fact.get("source_asset_id"))
        append_id(fact.get("filing_evidence_asset_id"))
        append_id(fact.get("submissions_context_asset_id"))
    for classification in classifications:
        if not isinstance(classification, dict):
            raise ValueError("Long-v4 classification catalog entry is malformed")
        append_id(classification.get("source_asset_id"))
    for price in prices:
        normalized = price.get("normalized_asset") if isinstance(price, dict) else None
        raw = price.get("raw_asset") if isinstance(price, dict) else None
        if not isinstance(normalized, dict) or not isinstance(raw, dict):
            raise ValueError("Long-v4 price catalog entry is malformed")
        append_id(normalized.get("id"))
        append_id(raw.get("id"))
    return _dedupe_text(asset_ids)


def _source_assets_for_catalog(
    catalog: dict[str, Any],
    *,
    store: AssetStore,
    decision_time: datetime,
) -> tuple[DataAsset, ...]:
    ordered_ids = _catalog_manifest_asset_ids(catalog)
    try:
        uuid_ids = [UUID(value) for value in ordered_ids]
    except (TypeError, ValueError):
        raise ValueError("Long-v4 evidence catalog contains an invalid asset identity") from None
    assets = DataAsset.objects.in_bulk(uuid_ids)
    if set(assets) != set(uuid_ids):
        raise ValueError("Long-v4 evidence catalog source closure is incomplete")
    canonical = tuple(assets[value] for value in uuid_ids)
    if any(
        asset.available_at > decision_time or asset.retrieved_at > decision_time
        for asset in canonical
    ):
        raise ValueError("Long-v4 evidence source closure was not visible at decision time")
    try:
        for asset in canonical:
            read_checksummed_bytes(store, asset)
    except RefreshVerificationError:
        raise ValueError("Long-v4 evidence source closure failed physical verification") from None
    return canonical


def _peer_candidate_payload(
    peer: _EntityAssessment,
    *,
    locked: _PeerLock | None,
    selected: bool,
    target_family: str | None,
) -> dict[str, Any]:
    family_mismatch = (
        peer.metric is not None
        and target_family is not None
        and peer.metric.family != target_family
    )
    return {
        "listing_id": str(peer.listing.pk),
        "company_id": str(peer.listing.security.company_id),
        "ticker": peer.listing.ticker,
        "classification_id": (
            str(peer.classification.pk) if peer.classification is not None else None
        ),
        "sic": (
            _normalized_sic(peer.classification.code) if peer.classification is not None else None
        ),
        "locked_prefix_level": locked.prefix_length if locked is not None else None,
        "status": "excluded_metric_family" if family_mismatch else peer.status,
        "reason_code": (
            "peer_metric_family_mismatch" if family_mismatch else peer.insufficiency_code
        ),
        "reason": (
            f"Peer metric family {peer.metric.family} does not match target family {target_family}"
            if family_mismatch and peer.metric is not None
            else peer.reason
        ),
        "metric_family": peer.metric.family if peer.metric is not None else None,
        "selected_peer": selected,
        "split_basis": peer.split_basis,
    }


def _unassessed_peer_candidates(peer_lock: _PeerLock | None) -> list[dict[str, Any]]:
    if peer_lock is None:
        return []
    details: dict[str, dict[str, Any]] = {}
    for level in peer_lock.examined_levels:
        candidates = level.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if isinstance(candidate, dict) and isinstance(candidate.get("listing_id"), str):
                details.setdefault(candidate["listing_id"], candidate)
    return [
        {
            **details[str(listing.pk)],
            "ticker": listing.ticker,
            "locked_prefix_level": peer_lock.selected_level,
            "status": "identity_candidate_unassessed",
            "reason_code": "peer_identity_floor_unmet",
            "reason": "Identity candidate retained as no configured peer floor was met",
            "metric_family": None,
            "selected_peer": False,
            "split_basis": None,
        }
        for listing in peer_lock.evidence_candidates
    ]


def _peer_payload(peer: _EntityAssessment) -> dict[str, Any]:
    assert peer.metric is not None
    return {
        **_peer_candidate_payload(
            peer,
            locked=None,
            selected=True,
            target_family=peer.metric.family,
        ),
        "entity_growth_estimate": peer.metric.target_entity_growth,
        "entity_growth_raw": list(peer.metric.entity_growth_raw),
        "diluted_share_changes_raw": list(peer.metric.share_changes_raw),
        "current_multiple_raw": peer.metric.current_multiple_raw,
        "current_multiple_capped": peer.metric.current_multiple_capped,
        "current_price": float(peer.price),
        "native_price": _decimal_text(peer.price),
        "native_currency": peer.listing.currency,
        "fx_conversion": False,
        "applied_fx_rate": None,
        "price_asset_id": str(peer.price_asset.pk),
        "selected_fact_ids": list(peer.selected_fact_ids),
    }


def _listing_identity_payload(listing: Listing) -> dict[str, Any]:
    return {
        "listing_id": str(listing.pk),
        "ticker": listing.ticker,
        "provider_symbol": listing.provider_symbol,
        "exchange_mic": listing.exchange_mic,
        "currency": listing.currency,
        "region": listing.region,
        "is_active": listing.is_active,
        "valid_from": listing.valid_from.isoformat() if listing.valid_from else None,
        "valid_to": listing.valid_to.isoformat() if listing.valid_to else None,
        "security_id": str(listing.security_id),
        "security_type": listing.security.security_type,
        "company_id": str(listing.security.company_id),
        "company_cik": listing.security.company.cik,
        "company_name": listing.security.company.name,
        "company_country": listing.security.company.country,
    }


def _classification_payload(
    classification: CompanyClassificationObservation,
    *,
    owner: Listing | None = None,
) -> dict[str, Any]:
    if owner is not None:
        _validate_classification_provenance(classification, owner=owner)
    return {
        "id": str(classification.pk),
        "company_id": str(classification.company_id),
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


def _fact_payload(
    fact: FundamentalFact,
    filing_assets: dict[str, DataAsset],
    *,
    raw_authority: _SecRawAuthority,
    owner: Listing | None = None,
) -> dict[str, Any]:
    filing = filing_assets.get(str(fact.pk))
    context = raw_authority.context_assets.get(str(fact.pk))
    if filing is None or context is None:
        raise ValueError("A long-v4 SEC fact has no visible filing evidence link")
    event = raw_authority.correction_events.get(str(fact.pk))
    if owner is not None:
        _validate_fact_provenance(fact, filing=filing, owner=owner)
    return {
        "id": str(fact.pk),
        "company_id": str(fact.company_id),
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
        "ingested_at": fact.ingested_at.isoformat(),
        "availability_basis": fact.availability_basis,
        "is_amendment": fact.is_amendment,
        "source_revision": fact.source_revision,
        "observation_hash": fact.observation_hash,
        "quality_flags": fact.quality_flags,
        "source_asset_id": str(fact.source_asset_id),
        "filing_evidence_asset_id": str(filing.pk),
        "submissions_context_asset_id": str(context.pk),
        "correction_observation_event": (
            {
                "id": str(event.pk),
                "provider": event.provider,
                "kind": event.kind,
                "subject": event.subject,
                "source_asset_id": str(event.source_asset_id),
                "content_sha256": event.content_sha256,
                "observed_at": event.observed_at.isoformat(),
                "recorded_at": event.recorded_at.isoformat(),
            }
            if event is not None
            else None
        ),
        "filing_history_filename": (
            filing.metadata.get(HISTORY_FILENAME_METADATA_KEY)
            if filing.kind == SUBMISSIONS_HISTORY_KIND
            else None
        ),
    }


def _validate_fact_provenance(
    fact: FundamentalFact,
    *,
    filing: DataAsset,
    owner: Listing,
) -> None:
    cik = owner.security.company.cik
    source = fact.source_asset
    history_filename = (
        filing.metadata.get(HISTORY_FILENAME_METADATA_KEY)
        if isinstance(filing.metadata, dict)
        else None
    )
    if (
        not cik
        or fact.company_id != owner.security.company_id
        or fact.provider != "sec"
        or source.provider != "sec"
        or source.kind != COMPANYFACTS_KIND
        or source.subject != cik
        or filing.provider != "sec"
        or filing.kind not in {SUBMISSIONS_KIND, SUBMISSIONS_HISTORY_KIND}
        or filing.subject != cik
        or (
            filing.kind == SUBMISSIONS_HISTORY_KIND
            and (
                not isinstance(history_filename, str)
                or not history_filename
                or history_filename != history_filename.strip()
                or not history_filename.startswith(f"CIK{cik}-submissions-")
                or not history_filename.endswith(".json")
            )
        )
    ):
        raise ValueError("Long-v4 SEC fact provenance identity is incompatible")


def _validate_classification_provenance(
    classification: CompanyClassificationObservation,
    *,
    owner: Listing,
) -> None:
    source = classification.source_asset
    cik = owner.security.company.cik
    if (
        not cik
        or classification.company_id != owner.security.company_id
        or classification.provider != "sec"
        or classification.scheme != "sec_sic"
        or source.provider != "sec"
        or source.kind != SUBMISSIONS_KIND
        or source.subject != cik
    ):
        raise ValueError("Long-v4 SEC classification provenance identity is incompatible")


def _canonical_price_payload(price: _CanonicalPrice) -> dict[str, Any]:
    ledger_value = format(price.ledger_value, ".6f")
    valuation_value = _decimal_text(price.valuation_value)
    return {
        "owner_listing_id": str(price.listing.pk),
        "listing_id": str(price.listing.pk),
        "provider": price.normalized_asset.provider,
        "subject": price.normalized_asset.subject,
        "exchange_mic": price.listing.exchange_mic,
        "session_date": price.session_date.isoformat(),
        "value": ledger_value,
        "ledger_value": ledger_value,
        "valuation_value": valuation_value,
        "valuation_source": "normalized_parquet_target_close",
        "native_price": valuation_value,
        "currency": price.listing.currency,
        "native_currency": price.listing.currency,
        "applied_fx_rate": None,
        "fx_conversion": False,
        "normalized_asset_id": str(price.normalized_asset.pk),
        "normalized_asset_sha256": price.normalized_asset.sha256,
        "raw_asset_id": str(price.raw_asset.pk),
        "raw_asset_sha256": price.raw_asset.sha256,
        "normalized_asset": _asset_payload(price.normalized_asset),
        "raw_asset": _asset_payload(price.raw_asset),
    }


def _catalog_price_for(catalog: dict[str, Any], listing_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in catalog["prices"]
        if isinstance(item, dict) and item.get("listing_id") == listing_id
    ]
    if len(matches) != 1:
        raise ValueError("Long-v4 catalog does not contain one canonical target price")
    return dict(matches[0])


def _asset_payload(asset: DataAsset) -> dict[str, Any]:
    return {
        "id": str(asset.pk),
        "provider": asset.provider,
        "kind": asset.kind,
        "subject": asset.subject,
        "relative_path": asset.relative_path,
        "sha256": asset.sha256,
        "retrieved_at": asset.retrieved_at.isoformat(),
        "available_at": asset.available_at.isoformat(),
        "period_start": asset.period_start.isoformat() if asset.period_start else None,
        "period_end": asset.period_end.isoformat() if asset.period_end else None,
        "schema_version": asset.schema_version,
    }


def _accounting_scope() -> dict[str, Any]:
    return {
        "metric_scope": "reported_gaap_fcf_or_net_income",
        "free_cash_flow_definition": "operating_cash_flow_minus_absolute_capex",
        "invested_capital_used": False,
        "rd_capitalization_performed": False,
        "rd_classification": "unavailable_in_us-sec-fundamentals-v1",
        "missing_rd_treated_as_zero": False,
        "return_on_new_capital_inference": False,
        "weighted_diluted_shares_semantics": ("accounting_period_denominator_not_issuance_count"),
        "sic_semantics": "coarse_industry_classification",
        "causal_or_project_irr_claim": False,
    }


def _normalized_sic(value: str) -> str | None:
    normalized = value.strip()
    if not normalized.isdigit() or len(normalized) > 4:
        return None
    return normalized.zfill(4)


def _finite(value: Decimal) -> float | None:
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("Long-v4 decimal evidence must be finite")
    return format(value, "f")


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _exceeds_inclusive(value: float, limit: float) -> bool:
    return value > limit and not math.isclose(
        value,
        limit,
        rel_tol=0,
        abs_tol=RETURN_IDENTITY_TOLERANCE,
    )


def _dedupe_text(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _bounded_reason(reason: str) -> str:
    normalized = " ".join(reason.split()).strip()
    if not normalized:
        normalized = "Long-v4 evidence is insufficient"
    return normalized[:240]

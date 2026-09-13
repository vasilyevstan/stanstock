from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from time import sleep
from typing import TYPE_CHECKING
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from django.db import IntegrityError, transaction
from django.db.models import QuerySet
from django.utils import timezone

from stanstock.core.verification_types import AssetRef, RefreshVerificationError
from stanstock.data.assets import (
    AssetStore,
    asset_ref_for,
    open_asset_store,
    read_checksummed_bytes,
    register_asset,
)
from stanstock.data.fact_identity import build_observation_hash, build_period_identity
from stanstock.data.models import (
    OBSERVATION_INSTANT_CONSTRAINT,
    Company,
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    Listing,
    ProviderRecord,
    Region,
    Security,
    SourceObservationEvent,
)
from stanstock.data.providers import sec
from stanstock.data.providers.contracts import FundamentalSourcePayload
from stanstock.data.providers.exceptions import (
    ProviderConfigurationError,
    ProviderResponseError,
)
from stanstock.data.sec_config import (
    SecCikConfig,
    SecCikMapping,
    SecConceptRule,
    SecFundamentalsConfig,
)
from stanstock.data.sec_evidence import (
    COMPANYFACTS_KIND,
    HISTORY_FILENAME_METADATA_KEY,
    MAPPING_KIND,
    SUBMISSIONS_HISTORY_KIND,
    SUBMISSIONS_KIND,
    SecEvidencePayloadError,
    historical_submission_filenames,
    is_safe_history_filename,
)
from stanstock.data.sec_fundamentals import (
    CORRECTION_AVAILABILITY_BASIS,
    CORRECTION_QUALITY_FLAG,
    REBOUND_QUALITY_FLAG,
    resolve_availability,
)

if TYPE_CHECKING:
    from stanstock.data.live_us import UsUniverseConfig

PROVIDER = sec.PROVIDER
SIC_SCHEME = "sec_sic"
_COMPANYFACTS_VERIFICATIONS_KEY = "companyfacts_verifications"
_COMPANYFACTS_NORMALIZATION_VERSION = "sec-companyfacts-v2"
_NEW_YORK = ZoneInfo("America/New_York")
_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_SUBMISSIONS_REQUIRED_COLUMNS = (
    "accessionNumber",
    "filingDate",
    "form",
    "reportDate",
    "primaryDocument",
)
_SUBMISSIONS_OPTIONAL_COLUMNS = ("acceptanceDateTime",)
SEC_EXCHANGE_MIC_RULE = (("Nasdaq", "XNAS"), ("NYSE", "XNYS"))
RawObservationLineage = tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class SecMappingRow:
    cik: str
    company_name: str
    ticker: str
    exchange: str


@dataclass(frozen=True, slots=True)
class FilingRecord:
    accession: str
    filing_date: date
    acceptance_at: datetime
    acceptance_basis: str
    filing_form: str
    report_date: date | None
    primary_document: str
    source_asset: DataAsset


@dataclass(frozen=True, slots=True)
class SecSubmissionsEvidence:
    cik: str
    filings: tuple[FilingRecord, ...]
    historical_filenames: tuple[str, ...]
    sic: str
    sic_description: str


@dataclass(frozen=True, slots=True)
class SecFactDerivation:
    concept: str
    taxonomy: str
    source_concept: str
    value: Decimal
    unit: str
    currency: str
    period_type: str
    period_identity: str
    period_start: date | None
    period_end: date
    fiscal_year: int | None
    fiscal_period: str
    frame: str
    accession: str
    filing_form: str
    filing_date: date | None
    acceptance_at: datetime
    filing_availability_basis: str
    is_amendment: bool
    observation_hash: str
    base_quality_flags: tuple[str, ...]
    filing_source_asset_id: UUID
    raw_observation_lineage: RawObservationLineage
    raw_observation_signature: str
    raw_observation_match_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SecConfiguredObservation:
    """One configured Companyfacts alias occurrence before normalization.

    ``derive_sec_companyfacts`` remains the strict ingestion-writer API.
    Point-in-time readers that must distinguish a genuinely absent alias from
    an alias that could not be normalized use ``inspect_sec_companyfacts`` and
    consume these records.  Scope fields are best-effort only for rejected
    observations: ``None`` means the malformed source did not prove the field,
    not that the field was absent economically. ``raw_filing_form`` retains
    the Companyfacts cell byte-for-text, while ``filing_form`` uses the
    reconciled submissions form whenever authoritative filing evidence exists.
    """

    concept: str
    taxonomy: str
    source_concept: str
    unit: str | None
    unit_supported: bool | None
    status: str
    rejection_code: str | None
    period_type: str
    period_start: date | None
    period_end: date | None
    accession: str | None
    raw_filing_form: str | None
    filing_form: str | None
    filing_date: date | None
    acceptance_at: datetime | None
    filing_availability_basis: str | None
    filing_source_asset_id: UUID | None
    derivation: SecFactDerivation | None
    raw_observation_lineage: RawObservationLineage | None
    raw_observation_signature: str | None
    raw_observation_match_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SecCompanyfactsInspection:
    derivations: tuple[SecFactDerivation, ...]
    configured_observations: tuple[SecConfiguredObservation, ...]


@dataclass(frozen=True, slots=True)
class _PersistedLineageHead:
    fact: FundamentalFact
    raw_observation_signature: str


class SecDerivationError(ProviderResponseError):
    """Raw SEC evidence could not produce the shared canonical derivation."""


@dataclass(frozen=True, slots=True)
class MissingAccessionVerification:
    first_missing_at: datetime
    last_checked_at: datetime


@dataclass(frozen=True, slots=True)
class CompanyfactsVerification:
    checked_at: datetime
    asset_sha256: str
    filing_sources_hash: str
    config_hash: str
    normalization_version: str
    missing_accessions: dict[str, MissingAccessionVerification]


@dataclass(frozen=True, slots=True)
class SecCompanyIngestionResult:
    symbol: str
    cik: str
    raw_assets_created: int
    raw_assets_reused: int
    facts_created: int
    facts_reused: int
    classifications_created: int
    historical_submission_files: int
    companyfacts_fetched: bool
    asset_refs: tuple[AssetRef, ...]


@dataclass(frozen=True, slots=True)
class SecIngestionResult:
    mapping_asset_id: str
    mapping_sha256: str
    companies: tuple[SecCompanyIngestionResult, ...]
    asset_refs: tuple[AssetRef, ...]

    @property
    def facts_created(self) -> int:
        return sum(company.facts_created for company in self.companies)

    @property
    def facts_reused(self) -> int:
        return sum(company.facts_reused for company in self.companies)

    @property
    def raw_assets_created(self) -> int:
        return sum(company.raw_assets_created for company in self.companies)

    @property
    def raw_assets_reused(self) -> int:
        return sum(company.raw_assets_reused for company in self.companies)

    @property
    def companyfacts_fetched(self) -> int:
        return sum(company.companyfacts_fetched for company in self.companies)


class SecRequestBudget:
    """Coordinate SEC's per-user request-rate ceiling across local jobs."""

    def __init__(
        self,
        *,
        requests_per_second: int,
        enforce_spacing: bool = True,
        require_enabled: bool = True,
    ) -> None:
        if not 1 <= requests_per_second <= 10:
            raise ValueError("SEC requests_per_second must be between 1 and 10")
        self.requests_per_second = requests_per_second
        self.enforce_spacing = enforce_spacing
        self.require_enabled = require_enabled

    def consume(self) -> None:
        now = timezone.now()
        with transaction.atomic():
            record, _created = ProviderRecord.objects.select_for_update().get_or_create(
                provider=PROVIDER
            )
            if self.require_enabled and not record.enabled:
                raise ProviderConfigurationError(
                    "SEC is disabled. Run configure_sec --enable after a successful "
                    "bounded preflight."
                )
            metadata = dict(record.metadata)
            reserved_at = now
            raw_next = metadata.get("next_request_not_before")
            if isinstance(raw_next, str):
                try:
                    next_request = datetime.fromisoformat(raw_next)
                except ValueError as exc:
                    raise ProviderConfigurationError(
                        "SEC provider metadata contains an invalid "
                        "next_request_not_before timestamp"
                    ) from exc
                if next_request.tzinfo is None:
                    raise ProviderConfigurationError(
                        "SEC next_request_not_before must include a timezone"
                    )
                reserved_at = max(reserved_at, next_request)
            interval = 1.0 / self.requests_per_second
            metadata.update(
                {
                    "requests_per_second": self.requests_per_second,
                    "next_request_not_before": (
                        reserved_at + timedelta(seconds=interval)
                    ).isoformat(),
                }
            )
            record.metadata = metadata
            record.save(update_fields=["metadata"])
        if self.enforce_spacing:
            delay = (reserved_at - timezone.now()).total_seconds()
            if delay > 0:
                sleep(delay)


def fetch_sec_mapping_asset(
    *,
    config: SecFundamentalsConfig,
    store: AssetStore | None = None,
    budget: SecRequestBudget | None = None,
) -> tuple[DataAsset, bool]:
    store = store or open_asset_store()
    budget = budget or SecRequestBudget(requests_per_second=config.requests_per_second)
    budget.consume()
    payload = sec.fetch_ticker_exchange_mapping()
    return _persist_payload(
        store=store,
        payload=payload,
        kind=MAPPING_KIND,
        subject=payload.subject,
        metadata={
            "config_version": config.config_version,
            "config_hash": config.config_hash,
            "source_url": payload.source_url,
            **payload.metadata,
        },
    )


def parse_sec_mapping(payload: bytes) -> tuple[SecMappingRow, ...]:
    """Parse the SEC ticker map through the shared path-free error surface."""
    try:
        return _parse_sec_mapping(payload)
    except SecDerivationError:
        raise
    except ProviderResponseError as exc:
        raise SecDerivationError(str(exc)) from exc


def _parse_sec_mapping(payload: bytes) -> tuple[SecMappingRow, ...]:
    data = _load_json_object(payload, label="SEC ticker mapping")
    fields = data.get("fields")
    rows = data.get("data")
    if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
        raise ProviderResponseError("SEC ticker mapping fields were missing or invalid")
    if not isinstance(rows, list):
        raise ProviderResponseError("SEC ticker mapping rows were missing or invalid")
    indexes = {field: index for index, field in enumerate(fields)}
    required = {"cik", "name", "ticker", "exchange"}
    if not required.issubset(indexes):
        raise ProviderResponseError("SEC ticker mapping lacks required fields")
    parsed: list[SecMappingRow] = []
    for row_number, row in enumerate(rows):
        if not isinstance(row, list) or len(row) < len(fields):
            raise ProviderResponseError(f"SEC ticker mapping row {row_number} was malformed")
        raw_ticker = row[indexes["ticker"]]
        raw_name = row[indexes["name"]]
        raw_exchange = row[indexes["exchange"]]
        if not isinstance(raw_ticker, str) or not raw_ticker.strip():
            continue
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        parsed.append(
            SecMappingRow(
                cik=sec.format_cik(row[indexes["cik"]]),
                company_name=raw_name.strip(),
                ticker=raw_ticker.strip().upper(),
                exchange=raw_exchange.strip() if isinstance(raw_exchange, str) else "",
            )
        )
    return tuple(parsed)


def verify_sec_exchange_mic(*, exchange: object, mic: object) -> None:
    """Verify the exact SEC exchange spelling against StanStock's MIC.

    This intentionally owns only the two reviewed SEC spellings used by the
    canonical US CIK configuration. It is not a general exchange mapper.
    """
    expected = dict(SEC_EXCHANGE_MIC_RULE).get(exchange) if isinstance(exchange, str) else None
    if expected is None or not isinstance(mic, str) or mic != expected:
        raise SecDerivationError("SEC exchange and listing MIC do not match the reviewed rule")


def mapping_candidates_for_symbols(
    rows: tuple[SecMappingRow, ...],
    symbols: tuple[str, ...],
) -> dict[str, tuple[SecMappingRow, ...]]:
    by_ticker: dict[str, list[SecMappingRow]] = {}
    for row in rows:
        by_ticker.setdefault(row.ticker, []).append(row)
    return {
        symbol: tuple(
            sorted(
                by_ticker.get(symbol, []),
                key=lambda row: (row.exchange, row.cik, row.company_name),
            )
        )
        for symbol in symbols
    }


def run_sec_ingestion(
    *,
    config: SecFundamentalsConfig,
    cik_config: SecCikConfig,
    universe_config: UsUniverseConfig,
    target_date: date,
    symbols: tuple[str, ...] | None = None,
    store: AssetStore | None = None,
    budget: SecRequestBudget | None = None,
) -> SecIngestionResult:
    store = store or open_asset_store()
    budget = budget or SecRequestBudget(requests_per_second=config.requests_per_second)
    if cik_config.universe_config_version != universe_config.config_version:
        raise ValueError("SEC CIK mapping universe version does not match the active US universe")
    configured_symbols = set(cik_config.mappings) | set(cik_config.excluded)
    expected_symbols = set(universe_config.symbols)
    if configured_symbols != expected_symbols:
        missing = sorted(expected_symbols - configured_symbols)
        extra = sorted(configured_symbols - expected_symbols)
        raise ValueError(
            f"SEC CIK mapping does not exactly cover the US universe; "
            f"missing={missing}, extra={extra}"
        )
    mapping_asset = DataAsset.objects.filter(
        provider=PROVIDER,
        kind=MAPPING_KIND,
        sha256=cik_config.source_sha256,
    ).first()
    if mapping_asset is None:
        mapping_asset, _created = fetch_sec_mapping_asset(
            config=config,
            store=store,
            budget=budget,
        )
    _verify_cik_config(
        cik_config=cik_config,
        rows=parse_sec_mapping(read_checksummed_bytes(store, mapping_asset)),
    )
    selected_symbols = symbols or tuple(universe_config.symbols)
    unknown = sorted(set(selected_symbols) - expected_symbols)
    if unknown:
        raise ValueError(f"SEC ingestion requested symbols outside the US universe: {unknown}")
    selected = set(selected_symbols)
    companies: list[SecCompanyIngestionResult] = []
    # Iterate the reviewed CIK config's own mapping order (not the
    # universe's or a caller's `symbols` order) so this run's asset_refs
    # are a stable, canonical sequence the SEC-stage validator can
    # independently reconstruct and compare exactly.
    for symbol, mapping in cik_config.mappings.items():
        if symbol not in selected or symbol in cik_config.excluded:
            continue
        companies.append(
            _ingest_company(
                mapping=mapping,
                config=config,
                target_date=target_date,
                store=store,
                budget=budget,
            )
        )
    return SecIngestionResult(
        mapping_asset_id=str(mapping_asset.pk),
        mapping_sha256=mapping_asset.sha256,
        companies=tuple(companies),
        asset_refs=(
            asset_ref_for(mapping_asset),
            *(ref for company in companies for ref in company.asset_refs),
        ),
    )


def _verify_cik_config(
    *,
    cik_config: SecCikConfig,
    rows: tuple[SecMappingRow, ...],
) -> None:
    official = {(row.ticker, row.cik, row.exchange) for row in rows}
    for mapping in cik_config.mappings.values():
        expected_mic = dict(SEC_EXCHANGE_MIC_RULE).get(mapping.exchange)
        if expected_mic is None:
            raise ValueError(
                f"Reviewed SEC mapping for {mapping.symbol} uses unsupported "
                f"exchange {mapping.exchange!r}"
            )
        verify_sec_exchange_mic(exchange=mapping.exchange, mic=expected_mic)
        identity = (mapping.official_ticker, mapping.cik, mapping.exchange)
        if identity not in official:
            raise ValueError(
                f"Reviewed SEC mapping for {mapping.symbol} no longer appears in the "
                f"source snapshot: ticker={mapping.official_ticker}, "
                f"CIK={mapping.cik}, exchange={mapping.exchange!r}"
            )


def _ingest_company(
    *,
    mapping: SecCikMapping,
    config: SecFundamentalsConfig,
    target_date: date,
    store: AssetStore,
    budget: SecRequestBudget,
) -> SecCompanyIngestionResult:
    listing = _resolve_listing(mapping)
    company = listing.security.company
    if company.cik and company.cik != mapping.cik:
        raise ValueError(
            f"{mapping.symbol} already has CIK {company.cik}, not reviewed CIK {mapping.cik}"
        )
    if not company.cik:
        Company.objects.filter(pk=company.pk, cik="").update(cik=mapping.cik)
        company.refresh_from_db()

    budget.consume()
    submissions_payload = sec.fetch_submissions(mapping.cik)
    submissions_asset, submissions_created = _persist_payload(
        store=store,
        payload=submissions_payload,
        kind=SUBMISSIONS_KIND,
        subject=mapping.cik,
        metadata=_source_metadata(
            payload=submissions_payload,
            config=config,
            target_date=target_date,
            symbol=mapping.symbol,
        ),
    )
    submissions_evidence = derive_sec_current_submissions(
        submissions_payload.content,
        source_asset=submissions_asset,
        expected_cik=mapping.cik,
    )
    current_records = list(submissions_evidence.filings)
    raw_created = int(submissions_created)
    raw_reused = int(not submissions_created)
    history_filenames = submissions_evidence.historical_filenames
    history_assets = {
        filename: _latest_history_asset(cik=mapping.cik, filename=filename)
        for filename in history_filenames
    }
    initial_filing_sources_hash = (
        _filing_sources_hash(
            submissions_asset=submissions_asset,
            history_assets={
                filename: asset for filename, asset in history_assets.items() if asset is not None
            },
        )
        if all(asset is not None for asset in history_assets.values())
        else None
    )
    recovered_companyfacts = _latest_observed_companyfacts(mapping.cik)
    latest_companyfacts = (
        recovered_companyfacts.asset if recovered_companyfacts is not None else None
    )
    verification = _load_companyfacts_verification(cik=mapping.cik)
    verified_companyfacts = (
        verification
        if latest_companyfacts is not None
        and verification is not None
        and verification.asset_sha256 == latest_companyfacts.sha256
        else None
    )
    reconciliation_due = _reconciliation_due(
        cik=mapping.cik,
        last_observed_at=(
            recovered_companyfacts.observed_at if recovered_companyfacts is not None else None
        ),
        last_checked_at=(
            verified_companyfacts.checked_at if verified_companyfacts is not None else None
        ),
        days=config.submissions_reconciliation_days,
    )
    known_companyfacts_accessions = (
        _companyfacts_accessions(read_checksummed_bytes(store, latest_companyfacts))
        if latest_companyfacts is not None
        else set()
    )
    current_relevant_accessions = {
        record.accession for record in current_records if record.filing_form in config.allowed_forms
    }
    missing_checks = (
        verified_companyfacts.missing_accessions if verified_companyfacts is not None else {}
    )
    currently_missing_accessions = current_relevant_accessions - known_companyfacts_accessions
    check_time = timezone.now()
    retryable_missing_accessions = {
        accession
        for accession in currently_missing_accessions
        if _missing_accession_retry_due(
            verification=missing_checks.get(accession),
            checked_at=check_time,
            retry_days=config.companyfacts_lag_retry_days,
        )
    }
    fetch_companyfacts = (
        latest_companyfacts is None
        or submissions_created
        or reconciliation_due
        or bool(retryable_missing_accessions)
    )
    normalization_complete = (
        verified_companyfacts is not None
        and initial_filing_sources_hash is not None
        and verified_companyfacts.filing_sources_hash == initial_filing_sources_hash
        and verified_companyfacts.config_hash == config.config_hash
        and verified_companyfacts.normalization_version == _COMPANYFACTS_NORMALIZATION_VERSION
    )
    normalize_companyfacts = fetch_companyfacts or (
        latest_companyfacts is not None and not normalization_complete
    )

    filing_records = list(current_records)
    historical_count = 0
    resolved_history_assets: dict[str, DataAsset] = {}
    if normalize_companyfacts:
        for filename in history_filenames:
            history_asset = history_assets[filename]
            if history_asset is None or reconciliation_due:
                budget.consume()
                history_payload = sec.fetch_submissions_history(filename)
                history_asset, history_created = _persist_payload(
                    store=store,
                    payload=history_payload,
                    kind=SUBMISSIONS_HISTORY_KIND,
                    subject=mapping.cik,
                    metadata=_source_metadata(
                        payload=history_payload,
                        config=config,
                        target_date=target_date,
                        symbol=mapping.symbol,
                        extra={HISTORY_FILENAME_METADATA_KEY: filename},
                    ),
                    history_filename=filename,
                )
                raw_created += int(history_created)
                raw_reused += int(not history_created)
                history_content = history_payload.content
            else:
                raw_reused += 1
                history_content = read_checksummed_bytes(store, history_asset)
            resolved_history_assets[filename] = history_asset
            historical_count += 1
            filing_records.extend(
                derive_sec_historical_submissions(
                    history_content,
                    source_asset=history_asset,
                    expected_cik=mapping.cik,
                    filename=filename,
                    allowed_filenames=history_filenames,
                )
            )
        filing_sources_hash = _filing_sources_hash(
            submissions_asset=submissions_asset,
            history_assets=resolved_history_assets,
        )

    if fetch_companyfacts:
        budget.consume()
        companyfacts_payload = sec.fetch_companyfacts(mapping.cik)
        companyfacts_metadata = _source_metadata(
            payload=companyfacts_payload,
            config=config,
            target_date=target_date,
            symbol=mapping.symbol,
            extra={
                "submissions_asset_id": str(submissions_asset.pk),
                "submissions_sha256": submissions_asset.sha256,
                "reconciliation": reconciliation_due,
            },
        )
        _preflight_companyfacts_lineage(
            company=company,
            payload=companyfacts_payload,
            metadata=companyfacts_metadata,
            filing_records=tuple(filing_records),
            config=config,
            store=store,
        )
        companyfacts_asset, companyfacts_created = _persist_payload(
            store=store,
            payload=companyfacts_payload,
            kind=COMPANYFACTS_KIND,
            subject=mapping.cik,
            metadata=companyfacts_metadata,
        )
        raw_created += int(companyfacts_created)
        raw_reused += int(not companyfacts_created)
        companyfacts_content = companyfacts_payload.content
        # The retrieval that actually carried these bytes, which is *not*
        # `companyfacts_asset.retrieved_at` whenever the content deduplicated
        # onto an earlier asset (a restatement back to a previous value).
        companyfacts_observed_at = companyfacts_payload.retrieved_at
    elif normalize_companyfacts:
        if latest_companyfacts is None or recovered_companyfacts is None:
            raise RuntimeError("SEC companyfacts normalization state is inconsistent")
        companyfacts_asset = latest_companyfacts
        companyfacts_content = read_checksummed_bytes(store, companyfacts_asset)
        raw_reused += 1
        # Replaying an already-persisted asset is not a new observation, so
        # the boundary is the observation that committed this content --
        # which is *not* the asset's own `retrieved_at` when a later
        # retrieval deduplicated onto an earlier asset.
        companyfacts_observed_at = recovered_companyfacts.observed_at
        _assert_replay_is_recoverable(
            company=company,
            recovered=recovered_companyfacts,
            cik=mapping.cik,
        )

    if normalize_companyfacts:
        companyfacts_checked_at = (
            check_time
            if fetch_companyfacts
            else (
                verified_companyfacts.checked_at
                if verified_companyfacts is not None
                else companyfacts_asset.retrieved_at
            )
        )
        facts_created, facts_reused = _normalize_companyfacts(
            company=company,
            payload=companyfacts_content,
            source_asset=companyfacts_asset,
            observed_at=companyfacts_observed_at,
            filing_records=tuple(filing_records),
            config=config,
            store=store,
        )
        _record_companyfacts_verification(
            cik=mapping.cik,
            checked_at=companyfacts_checked_at,
            asset=companyfacts_asset,
            filing_sources_hash=filing_sources_hash,
            config_hash=config.config_hash,
            missing_accessions=_updated_missing_accession_checks(
                missing_accessions=(
                    current_relevant_accessions - _companyfacts_accessions(companyfacts_content)
                ),
                previous=missing_checks,
                checked_at=companyfacts_checked_at,
                companyfacts_fetched=fetch_companyfacts,
                companyfacts_asset=companyfacts_asset,
            ),
        )
    else:
        if latest_companyfacts is None:
            raise RuntimeError("SEC companyfacts refresh state is inconsistent")
        raw_reused += 1
        facts_created = 0
        facts_reused = 0
    classifications_created = _persist_sic_classification(
        company=company,
        evidence=submissions_evidence,
        observed_at=submissions_payload.retrieved_at,
        source_asset=submissions_asset,
    )
    companyfacts_used_asset = (
        companyfacts_asset
        if (fetch_companyfacts or normalize_companyfacts)
        else latest_companyfacts
    )
    if companyfacts_used_asset is None:
        raise RuntimeError("SEC companyfacts asset identity is inconsistent")
    # `resolved_history_assets` is populated only when this run actually
    # normalized companyfacts. In the honest no-op branch, the exact
    # history assets consulted to compute `initial_filing_sources_hash`
    # (which gated that no-op) are `history_assets` instead -- omitting
    # them here would silently under-report evidence for a run that did
    # read and rely on their content. That no-op path only runs once every
    # `history_assets` value is proven non-`None` (see
    # `initial_filing_sources_hash` above), so this lookup is safe.
    used_history_assets: list[DataAsset] = []
    for filename in sorted(history_filenames):
        history_asset = (resolved_history_assets if normalize_companyfacts else history_assets)[
            filename
        ]
        if history_asset is None:
            raise RuntimeError("SEC history asset identity is inconsistent")
        used_history_assets.append(history_asset)
    used_assets = [submissions_asset, *used_history_assets, companyfacts_used_asset]
    return SecCompanyIngestionResult(
        symbol=mapping.symbol,
        cik=mapping.cik,
        raw_assets_created=raw_created,
        raw_assets_reused=raw_reused,
        facts_created=facts_created,
        facts_reused=facts_reused,
        classifications_created=classifications_created,
        historical_submission_files=historical_count,
        companyfacts_fetched=fetch_companyfacts,
        asset_refs=tuple(asset_ref_for(asset) for asset in used_assets),
    )


def _latest_history_asset(*, cik: str, filename: str) -> DataAsset | None:
    assets = DataAsset.objects.filter(
        provider=PROVIDER,
        kind=SUBMISSIONS_HISTORY_KIND,
        subject=cik,
    ).order_by("-retrieved_at")
    for asset in assets:
        if asset.metadata.get(HISTORY_FILENAME_METADATA_KEY) == filename:
            return asset
    return None


@dataclass(frozen=True, slots=True)
class RecoveredCompanyfacts:
    """The exact companyfacts content a retry must replay, and its boundary.

    ``observed_at`` is the retrieval that committed this content, taken from
    the append-only `SourceObservationEvent`. It is deliberately *not*
    ``asset.retrieved_at``: raw assets are content-addressed, so a later
    response that repeats earlier bytes reuses the earlier asset row while
    the observation that actually carried it is much newer.
    """

    asset: DataAsset
    observed_at: datetime
    basis: str


#: The recovered pair came from an explicit observation event.
COMPANYFACTS_RECOVERY_OBSERVED = "observation_event"
#: No observation event exists (rows predate them), so the asset's own
#: retrieval is the only boundary available.
COMPANYFACTS_RECOVERY_LEGACY = "legacy_asset_retrieval"


def _latest_observed_companyfacts(cik: str) -> RecoveredCompanyfacts | None:
    """Recover the latest *observed* companyfacts asset, not the newest asset.

    Ordering persisted assets by ``retrieved_at`` answers the wrong question
    after a content reversion. If a value went 100 -> 101 -> 100, the newest
    observation carries the *older* asset (the 100 bytes already on file),
    while the 101 asset still has the newer ``retrieved_at``. A replay driven
    by asset retrieval would therefore substitute stale 101 content and
    append it as a fresh correction over the committed 100.

    Recovery is driven by the observation events instead, newest observation
    first. ``unique_source_observation_instant`` guarantees at most one event
    per ``(provider, kind, subject, observed_at)``, so "newest observation"
    is unambiguous by construction and recovery never has to order two
    conflicting payloads -- that conflict is refused at write time, before
    anything is normalized.
    """
    events = (
        SourceObservationEvent.objects.filter(
            provider=PROVIDER,
            kind=COMPANYFACTS_KIND,
            subject=cik,
        )
        .select_related("source_asset")
        .order_by("-observed_at")
    )
    newest = events.first()
    if newest is None:
        asset = (
            DataAsset.objects.filter(
                provider=PROVIDER,
                kind=COMPANYFACTS_KIND,
                subject=cik,
            )
            .order_by("-retrieved_at")
            .first()
        )
        if asset is None:
            return None
        return RecoveredCompanyfacts(
            asset=asset,
            observed_at=asset.retrieved_at,
            basis=COMPANYFACTS_RECOVERY_LEGACY,
        )
    _assert_event_matches_asset(newest)
    return RecoveredCompanyfacts(
        asset=newest.source_asset,
        observed_at=newest.observed_at,
        basis=COMPANYFACTS_RECOVERY_OBSERVED,
    )


def _assert_replay_is_recoverable(
    *,
    company: Company,
    recovered: RecoveredCompanyfacts,
    cik: str,
) -> None:
    """Refuse to replay when the committed content cannot be proven.

    A replay re-normalizes persisted bytes, so it must know *exactly* which
    bytes were committed last and when they were observed. Two situations
    make that unprovable, and both fail closed rather than guessing:

    1. **No observation evidence for a corrected chain.** A database upgraded
       from before observation events has assets but no events, so the only
       ordering available is `DataAsset.retrieved_at`. That is the wrong
       clock precisely when it matters: after a 100 -> 101 -> 100 reversion
       the superseded 101 asset still holds the newest retrieval, so a replay
       would re-append 101 as a brand-new, event-bound revision the provider
       never sent. Where the company has no correction chain at all there is
       nothing to mis-order, and recovery proceeds.

    2. **Content older than a committed correction.** If the recoverable
       observation predates a correction already on file, re-normalizing it
       would restore the superseded value as a new revision.

    Neither is a dead end: a fresh fetch is separately gated and establishes
    new evidence, after which recovery is proven again.
    """
    if recovered.basis == COMPANYFACTS_RECOVERY_LEGACY:
        corrected = (
            FundamentalFact.objects.filter(
                company=company,
                provider=PROVIDER,
                source_revision__gt=1,
            )
            .order_by("concept", "period_end", "-source_revision")
            .first()
        )
        if corrected is not None:
            raise ProviderResponseError(
                f"SEC companyfacts recovery for CIK {cik} is unproven: no observation "
                "event records which stored payload was committed last, and this "
                f"company already has a correction chain (e.g. {corrected.concept} "
                f"{corrected.period_end.isoformat()} at revision "
                f"{corrected.source_revision}). Ordering stored assets by retrieval "
                "would replay a superseded payload as a new correction, so the replay "
                "refuses. Re-run once a fresh Companyfacts retrieval is due, which "
                "records the observation this recovery needs."
            )
    newest_correction = (
        FundamentalFact.objects.filter(
            company=company,
            provider=PROVIDER,
            availability_basis=CORRECTION_AVAILABILITY_BASIS,
        )
        .order_by("-available_at")
        .first()
    )
    if newest_correction is None or newest_correction.available_at <= recovered.observed_at:
        return
    raise ProviderResponseError(
        f"SEC companyfacts recovery for CIK {cik} would replay content observed at "
        f"{recovered.observed_at.isoformat()}, which is older than the committed "
        f"correction for {newest_correction.concept} "
        f"{newest_correction.period_end.isoformat()} available at "
        f"{newest_correction.available_at.isoformat()}. Replaying it would append a "
        "correction the provider never sent."
    )


def _load_companyfacts_verification(*, cik: str) -> CompanyfactsVerification | None:
    metadata = (
        ProviderRecord.objects.filter(provider=PROVIDER).values_list("metadata", flat=True).first()
    )
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ProviderConfigurationError("SEC provider metadata must be an object")
    raw_verifications = metadata.get(_COMPANYFACTS_VERIFICATIONS_KEY)
    if raw_verifications is None:
        return None
    if not isinstance(raw_verifications, dict):
        raise ProviderConfigurationError(
            "SEC provider companyfacts_verifications metadata must be an object"
        )
    raw_verification = raw_verifications.get(cik)
    if raw_verification is None:
        return None
    if not isinstance(raw_verification, dict):
        raise ProviderConfigurationError(
            f"SEC companyfacts verification metadata for CIK {cik} must be an object"
        )
    raw_checked_at = raw_verification.get("checked_at")
    if not isinstance(raw_checked_at, str):
        raise ProviderConfigurationError(
            f"SEC companyfacts verification for CIK {cik} has no checked_at timestamp"
        )
    try:
        checked_at = datetime.fromisoformat(raw_checked_at)
    except ValueError as exc:
        raise ProviderConfigurationError(
            f"SEC companyfacts verification for CIK {cik} has an invalid checked_at timestamp"
        ) from exc
    if checked_at.tzinfo is None:
        raise ProviderConfigurationError(
            f"SEC companyfacts verification for CIK {cik} must include a timezone"
        )
    asset_sha256 = raw_verification.get("asset_sha256")
    if not isinstance(asset_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", asset_sha256):
        raise ProviderConfigurationError(
            f"SEC companyfacts verification for CIK {cik} has an invalid asset_sha256"
        )
    filing_sources_hash = raw_verification.get("filing_sources_hash", "")
    if not isinstance(filing_sources_hash, str) or (
        filing_sources_hash and not re.fullmatch(r"[0-9a-f]{64}", filing_sources_hash)
    ):
        raise ProviderConfigurationError(
            f"SEC companyfacts verification for CIK {cik} has an invalid filing_sources_hash"
        )
    config_hash = raw_verification.get("config_hash", "")
    if not isinstance(config_hash, str) or (
        config_hash and not re.fullmatch(r"[0-9a-f]{64}", config_hash)
    ):
        raise ProviderConfigurationError(
            f"SEC companyfacts verification for CIK {cik} has an invalid config_hash"
        )
    normalization_version = raw_verification.get("normalization_version", "")
    if not isinstance(normalization_version, str):
        raise ProviderConfigurationError(
            f"SEC companyfacts verification for CIK {cik} has an invalid normalization_version"
        )
    raw_missing_accessions = raw_verification.get("missing_accessions")
    missing_accessions: dict[str, MissingAccessionVerification] = {}
    if isinstance(raw_missing_accessions, list):
        if not all(
            isinstance(accession, str) and accession for accession in raw_missing_accessions
        ):
            raise ProviderConfigurationError(
                f"SEC companyfacts verification for CIK {cik} has invalid missing_accessions"
            )
        missing_accessions = {
            accession: MissingAccessionVerification(
                first_missing_at=checked_at,
                last_checked_at=checked_at,
            )
            for accession in raw_missing_accessions
        }
    elif isinstance(raw_missing_accessions, dict):
        for accession, raw_check in raw_missing_accessions.items():
            if not isinstance(accession, str) or not accession or not isinstance(raw_check, dict):
                raise ProviderConfigurationError(
                    f"SEC companyfacts verification for CIK {cik} has invalid missing_accessions"
                )
            first_missing_at = _parse_verification_timestamp(
                raw_check.get("first_missing_at"),
                label=f"first_missing_at for CIK {cik} accession {accession}",
            )
            last_checked_at = _parse_verification_timestamp(
                raw_check.get("last_checked_at"),
                label=f"last_checked_at for CIK {cik} accession {accession}",
            )
            if last_checked_at < first_missing_at:
                raise ProviderConfigurationError(
                    f"SEC companyfacts verification for CIK {cik} accession "
                    f"{accession} checks are out of order"
                )
            missing_accessions[accession] = MissingAccessionVerification(
                first_missing_at=first_missing_at,
                last_checked_at=last_checked_at,
            )
    else:
        raise ProviderConfigurationError(
            f"SEC companyfacts verification for CIK {cik} has invalid missing_accessions"
        )
    return CompanyfactsVerification(
        checked_at=checked_at,
        asset_sha256=asset_sha256,
        filing_sources_hash=filing_sources_hash,
        config_hash=config_hash,
        normalization_version=normalization_version,
        missing_accessions=missing_accessions,
    )


def _record_companyfacts_verification(
    *,
    cik: str,
    checked_at: datetime,
    asset: DataAsset,
    filing_sources_hash: str,
    config_hash: str,
    missing_accessions: dict[str, MissingAccessionVerification],
) -> None:
    if checked_at.tzinfo is None:
        raise ValueError("SEC companyfacts verification timestamp must include a timezone")
    with transaction.atomic():
        record, _created = ProviderRecord.objects.select_for_update().get_or_create(
            provider=PROVIDER
        )
        metadata = dict(record.metadata)
        raw_verifications = metadata.get(_COMPANYFACTS_VERIFICATIONS_KEY, {})
        if not isinstance(raw_verifications, dict):
            raise ProviderConfigurationError(
                "SEC provider companyfacts_verifications metadata must be an object"
            )
        verifications = dict(raw_verifications)
        verifications[cik] = {
            "checked_at": checked_at.astimezone(UTC).isoformat(),
            "asset_sha256": asset.sha256,
            "filing_sources_hash": filing_sources_hash,
            "config_hash": config_hash,
            "normalization_version": _COMPANYFACTS_NORMALIZATION_VERSION,
            "missing_accessions": {
                accession: {
                    "first_missing_at": verification.first_missing_at.astimezone(UTC).isoformat(),
                    "last_checked_at": verification.last_checked_at.astimezone(UTC).isoformat(),
                }
                for accession, verification in sorted(missing_accessions.items())
            },
        }
        metadata[_COMPANYFACTS_VERIFICATIONS_KEY] = verifications
        record.metadata = metadata
        record.save(update_fields=["metadata"])


def _parse_verification_timestamp(raw_value: object, *, label: str) -> datetime:
    if not isinstance(raw_value, str):
        raise ProviderConfigurationError(f"SEC companyfacts verification {label} is missing")
    try:
        parsed = datetime.fromisoformat(raw_value)
    except ValueError as exc:
        raise ProviderConfigurationError(
            f"SEC companyfacts verification {label} is invalid"
        ) from exc
    if parsed.tzinfo is None:
        raise ProviderConfigurationError(
            f"SEC companyfacts verification {label} must include a timezone"
        )
    return parsed


def _missing_accession_retry_due(
    *,
    verification: MissingAccessionVerification | None,
    checked_at: datetime,
    retry_days: int,
) -> bool:
    if verification is None:
        return True
    age_days = (checked_at.date() - verification.first_missing_at.date()).days
    return 0 < age_days <= retry_days and verification.last_checked_at.date() < checked_at.date()


def _updated_missing_accession_checks(
    *,
    missing_accessions: set[str],
    previous: dict[str, MissingAccessionVerification],
    checked_at: datetime,
    companyfacts_fetched: bool,
    companyfacts_asset: DataAsset,
) -> dict[str, MissingAccessionVerification]:
    result: dict[str, MissingAccessionVerification] = {}
    for accession in missing_accessions:
        existing = previous.get(accession)
        if companyfacts_fetched:
            result[accession] = MissingAccessionVerification(
                first_missing_at=(
                    existing.first_missing_at if existing is not None else checked_at
                ),
                last_checked_at=checked_at,
            )
        elif existing is not None:
            result[accession] = existing
        else:
            result[accession] = MissingAccessionVerification(
                first_missing_at=companyfacts_asset.retrieved_at,
                last_checked_at=companyfacts_asset.retrieved_at,
            )
    return result


def _reconciliation_due(
    *,
    cik: str,
    last_observed_at: datetime | None,
    last_checked_at: datetime | None,
    days: int,
) -> bool:
    """Whether this CIK's companyfacts are stale enough to re-request.

    Staleness is measured from the last *observation*, not from the stored
    asset's ``retrieved_at``. After a content reversion the committed asset
    is an older row that was observed again recently, so reading its
    retrieval clock would report months of staleness that did not happen and
    spend a request rebuilding evidence already on file.
    """
    if last_observed_at is None:
        return True
    reference_at = last_checked_at or last_observed_at
    age = (timezone.now().date() - reference_at.date()).days
    stagger_days = int(cik[-2:]) % 7
    return age >= days + stagger_days


def _companyfacts_accessions(payload: bytes) -> set[str]:
    data = _load_json_object(payload, label="SEC companyfacts")
    taxonomies = data.get("facts")
    if not isinstance(taxonomies, dict):
        return set()
    accessions: set[str] = set()
    for concepts in taxonomies.values():
        if not isinstance(concepts, dict):
            continue
        for concept in concepts.values():
            units = concept.get("units") if isinstance(concept, dict) else None
            if not isinstance(units, dict):
                continue
            for observations in units.values():
                if not isinstance(observations, list):
                    continue
                for observation in observations:
                    if not isinstance(observation, dict):
                        continue
                    accession = observation.get("accn")
                    if isinstance(accession, str) and accession:
                        accessions.add(accession)
    return accessions


def _filing_sources_hash(
    *,
    submissions_asset: DataAsset,
    history_assets: dict[str, DataAsset],
) -> str:
    payload = {
        "submissions_sha256": submissions_asset.sha256,
        "history": [
            {
                "filename": filename,
                "sha256": history_assets[filename].sha256,
            }
            for filename in sorted(history_assets)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_listing(mapping: SecCikMapping) -> Listing:
    listings = list(
        Listing.objects.select_related("security__company").filter(
            provider_symbol=mapping.symbol,
            region=Region.US,
            is_active=True,
            security__security_type__in=(
                Security.SecurityType.COMMON_STOCK,
                Security.SecurityType.ADR,
            ),
        )
    )
    if len(listings) != 1:
        raise ValueError(
            f"Expected exactly one active stock listing for {mapping.symbol}; found {len(listings)}"
        )
    return listings[0]


def _source_metadata(
    *,
    payload: FundamentalSourcePayload,
    config: SecFundamentalsConfig,
    target_date: date,
    symbol: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source_url": payload.source_url,
        "content_type": payload.content_type,
        "symbol": symbol,
        "target_date": target_date.isoformat(),
        "config_version": config.config_version,
        "config_hash": config.config_hash,
        **payload.metadata,
    }
    metadata.update(extra or {})
    return metadata


def _persist_payload(
    *,
    store: AssetStore,
    payload: FundamentalSourcePayload,
    kind: str,
    subject: str,
    metadata: dict[str, object],
    history_filename: str | None = None,
) -> tuple[DataAsset, bool]:
    """Persist `payload` as a content-addressed asset under `(kind, subject)`.

    `history_filename` narrows both the reuse lookup and the observation
    identity for `SUBMISSIONS_HISTORY_KIND`, whose `subject` is the CIK
    shared by every one of its distinct history filenames: without it, two
    different filenames with identical bytes would collapse onto the same
    asset row and the same observation instant. `DataAsset.subject` itself
    stays the plain CIK (the reader contract); only the reuse lookup and the
    observation event's `subject` gain the filename discriminator.
    """
    try:
        digest = hashlib.sha256(payload.content).hexdigest()
    except (TypeError, ValueError):
        raise RefreshVerificationError(
            "sec_evidence_digest_failed", "SEC evidence payload could not be checksummed"
        ) from None
    lookup: dict[str, object] = {
        "provider": PROVIDER,
        "kind": kind,
        "subject": subject,
        "sha256": digest,
    }
    if history_filename is not None:
        lookup[f"metadata__{HISTORY_FILENAME_METADATA_KEY}"] = history_filename
    observation_subject = (
        f"{subject}:{history_filename}" if history_filename is not None else subject
    )
    existing = DataAsset.objects.filter(**lookup).order_by("-retrieved_at").first()
    if existing is not None:
        # Content-addressed reuse: these exact bytes are already stored. The
        # asset row keeps its original `retrieved_at` (when the content was
        # *first* seen), so this retrieval is recorded as its own append-only
        # observation event instead.
        _record_observation_event(
            asset=existing,
            kind=kind,
            subject=observation_subject,
            digest=digest,
            observed_at=payload.retrieved_at,
        )
        return existing, False
    stamp = payload.retrieved_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    safe_subject = _SAFE_PATH_COMPONENT.sub("_", subject)
    discriminator = (
        f"-{_SAFE_PATH_COMPONENT.sub('_', history_filename)}"
        if history_filename is not None
        else ""
    )
    relative_path = f"raw/sec/{kind}/{safe_subject}/{stamp}-{digest[:12]}{discriminator}.json"
    try:
        stored = store.write_bytes(relative_path, payload.content)
    except (OSError, ValueError):
        raise RefreshVerificationError(
            "sec_evidence_write_failed",
            "SEC evidence payload could not be persisted to the asset store",
        ) from None
    try:
        with transaction.atomic():
            asset = register_asset(
                provider=PROVIDER,
                kind=kind,
                subject=subject,
                stored=stored,
                retrieved_at=payload.retrieved_at,
                available_at=payload.retrieved_at,
                metadata=metadata,
            )
            _record_observation_event(
                asset=asset,
                kind=kind,
                subject=observation_subject,
                digest=digest,
                observed_at=payload.retrieved_at,
            )
    except Exception:
        # Best-effort cleanup of the just-written file when registration
        # failed and no row ended up pointing at it. A raw `OSError` from
        # this cleanup unlink (e.g. a permissions fault) must never replace
        # -- and thereby leak a path via -- whatever exception is already
        # propagating from the `try` block above.
        try:
            if not DataAsset.objects.filter(relative_path=relative_path).exists():
                store.resolve(relative_path).unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
        raise
    return asset, True


class ObservationEvidenceError(ProviderResponseError):
    """An observation event and the asset it points at disagree.

    The event's ``content_sha256`` is the claim "these exact bytes were seen
    at this time". If it does not match the immutable asset's own checksum,
    the event certifies a boundary for content that asset never held, and
    every correction bound to it is unsound. There is no safe repair at read
    time, so this always fails explicitly.
    """


def _observation_instant_conflict(error: IntegrityError) -> bool:
    """Whether this error is the observation-instant uniqueness violation.

    Deliberately narrow. A blanket ``except IntegrityError`` would swallow a
    foreign-key violation, a not-null violation, or a check-constraint
    failure and then "recover" by returning whatever row happens to sit at
    that instant -- turning an unrelated database fault into a silent
    success.
    """
    cause = getattr(error, "__cause__", None)
    constraint = getattr(cause, "diag", None)
    if constraint is not None:
        # psycopg exposes the violated constraint by name.
        return getattr(constraint, "constraint_name", None) == OBSERVATION_INSTANT_CONSTRAINT
    # SQLite names the columns of the failed index in the message text.
    message = str(error)
    if "UNIQUE constraint failed" not in message:
        return False
    return all(
        f"{SourceObservationEvent._meta.db_table}.{column}" in message
        for column in ("provider", "kind", "subject", "observed_at")
    )


def validate_sec_observation_event(
    event: SourceObservationEvent,
) -> SourceObservationEvent:
    """Refuse an event whose digest does not match its own source asset.

    This deliberately validates only the event-to-asset relationship.  A
    caller that owns a particular provider/kind/subject boundary must also
    compare those identities with its independently resolved authority.
    """
    if event.content_sha256 != event.source_asset.sha256:
        raise ObservationEvidenceError(
            f"Observation event {event.pk} claims content {event.content_sha256[:12]} "
            f"at {event.observed_at.isoformat()}, but its source asset "
            f"{event.source_asset_id} holds {event.source_asset.sha256[:12]}. An "
            "observation may only certify the bytes its asset actually contains."
        )
    return event


def _assert_event_matches_asset(event: SourceObservationEvent) -> SourceObservationEvent:
    """Backward-compatible internal spelling for the public validator."""
    return validate_sec_observation_event(event)


def _record_observation_event(
    *,
    asset: DataAsset,
    kind: str,
    subject: str,
    digest: str,
    observed_at: datetime,
) -> SourceObservationEvent:
    """Append this retrieval as an immutable observation of exact bytes.

    One ``(provider, kind, subject, observed_at)`` names exactly one content.
    Recording the *same* retrieval again is idempotent, which is what makes a
    retried or replayed ingestion safe.

    Recording *different* content at that same timestamp is refused here,
    before anything is normalized. Two different payloads at one observation
    instant carry no evidence about which came later, and inferring an order
    from a local commit clock would let a superseded payload be replayed as
    the newest one (the A -> B -> A case, where the reversion to A reuses A's
    original event and leaves B looking newer). Refusing keeps the boundary
    honest; a genuinely later retrieval simply carries a later timestamp.

    The digest is also checked against the asset it is being bound to, both
    on insert and on an idempotent collision, so an event can never certify
    bytes its own asset does not hold.
    """
    if observed_at.tzinfo is None:
        raise ValueError("SEC observation events must carry a timezone-aware timestamp")
    if digest != asset.sha256:
        raise ObservationEvidenceError(
            f"Refusing to record an observation of content {digest[:12]} against asset "
            f"{asset.pk}, which holds {asset.sha256[:12]}. An observation may only "
            "certify the bytes its asset actually contains."
        )
    try:
        # The insert itself is the mutual exclusion. `unique_source_observation_instant`
        # covers (provider, kind, subject, observed_at) without the digest, so
        # exactly one writer can claim an instant no matter how many race for
        # it. A read-then-write check would let two concurrent transactions
        # both pass their read, and `select_for_update()` cannot help either:
        # there is no row yet to lock.
        #
        # The savepoint is what makes the surrounding transaction usable
        # again after the failed insert; without it the connection stays
        # broken and the recovery read below could not run.
        with transaction.atomic():
            return SourceObservationEvent.objects.create(
                provider=PROVIDER,
                kind=kind,
                subject=subject,
                content_sha256=digest,
                source_asset=asset,
                observed_at=observed_at,
            )
    except IntegrityError as error:
        if not _observation_instant_conflict(error):
            # Any other integrity fault is a real database problem, not a
            # racing writer. It must surface unchanged even when a row does
            # happen to exist at this instant.
            raise
        committed = (
            SourceObservationEvent.objects.select_related("source_asset")
            .filter(
                provider=PROVIDER,
                kind=kind,
                subject=subject,
                observed_at=observed_at,
            )
            .first()
        )
        if committed is None:
            # The uniqueness violation was reported but nothing is committed
            # at this instant. That is not a state this code can reason
            # about, so the original error stands.
            raise
    # Someone else claimed this instant. Whether that is a safe retry or a
    # genuine conflict is decided by the committed content, and it is decided
    # here -- before any fact is normalized.
    _assert_event_matches_asset(committed)
    if committed.content_sha256 != digest:
        raise ProviderResponseError(
            f"SEC {kind} for {subject} reports two different payloads observed at the "
            f"same instant {observed_at.isoformat()} "
            f"({committed.content_sha256[:12]} and {digest[:12]}). One observation "
            "timestamp names one content; there is no evidence of which is newer, "
            "so the run refuses rather than ordering them by a local clock."
        )
    return committed


def derive_sec_current_submissions(
    payload: bytes,
    *,
    source_asset: DataAsset,
    expected_cik: str,
) -> SecSubmissionsEvidence:
    """Derive current filing/SIC evidence from supplied bytes only.

    The function performs no ORM lookup, persistence, file read, or network
    access.  The caller supplies both the immutable source row and its bytes;
    their SEC/CIK identities are checked before any fields are derived.
    """
    try:
        return _derive_sec_current_submissions(
            payload,
            source_asset=source_asset,
            expected_cik=expected_cik,
        )
    except SecDerivationError:
        raise
    except (ProviderResponseError, SecEvidencePayloadError) as exc:
        raise SecDerivationError(str(exc)) from exc


def _derive_sec_current_submissions(
    payload: bytes,
    *,
    source_asset: DataAsset,
    expected_cik: str,
) -> SecSubmissionsEvidence:
    cik = _canonical_cik(expected_cik, label="expected CIK")
    _validate_sec_source_identity(
        source_asset,
        expected_cik=cik,
        expected_kind=SUBMISSIONS_KIND,
    )
    # This shared parser is the canonical strict owner of the source-ordered,
    # duplicate-free and path-safe history filename list.
    filenames = historical_submission_filenames(payload)
    data = _load_json_object(payload, label="SEC submissions")
    if _canonical_cik(data.get("cik"), label="SEC submissions CIK") != cik:
        raise ProviderResponseError("SEC submissions payload CIK does not match its source asset")
    filings = data.get("filings")
    if not isinstance(filings, dict):
        raise ProviderResponseError("SEC submissions filings was missing or malformed")
    recent = filings.get("recent")
    if not isinstance(recent, dict):
        raise ProviderResponseError("SEC submissions filings.recent was missing or malformed")
    records = _records_from_columns(recent, source_asset=source_asset)
    raw_sic = data.get("sic")
    sic = "" if raw_sic is None else str(raw_sic).strip()
    return SecSubmissionsEvidence(
        cik=cik,
        filings=records,
        historical_filenames=filenames,
        sic=sic,
        sic_description=_text(data.get("sicDescription")),
    )


def derive_sec_historical_submissions(
    payload: bytes,
    *,
    source_asset: DataAsset,
    expected_cik: str,
    filename: str,
    allowed_filenames: tuple[str, ...],
) -> tuple[FilingRecord, ...]:
    """Derive one context-authorized historical filing table without I/O."""
    try:
        return _derive_sec_historical_submissions(
            payload,
            source_asset=source_asset,
            expected_cik=expected_cik,
            filename=filename,
            allowed_filenames=allowed_filenames,
        )
    except SecDerivationError:
        raise
    except (ProviderResponseError, SecEvidencePayloadError) as exc:
        raise SecDerivationError(str(exc)) from exc


def _derive_sec_historical_submissions(
    payload: bytes,
    *,
    source_asset: DataAsset,
    expected_cik: str,
    filename: str,
    allowed_filenames: tuple[str, ...],
) -> tuple[FilingRecord, ...]:
    cik = _canonical_cik(expected_cik, label="expected CIK")
    normalized_filename = filename.strip() if isinstance(filename, str) else ""
    expected_prefix = f"CIK{cik}-submissions-"
    history_suffix = (
        normalized_filename[len(expected_prefix) : -len(".json")]
        if normalized_filename.startswith(expected_prefix) and normalized_filename.endswith(".json")
        else ""
    )
    allowed = tuple(allowed_filenames)
    allowed_are_strings = all(isinstance(item, str) for item in allowed)
    if (
        not normalized_filename
        or normalized_filename != filename
        or not is_safe_history_filename(normalized_filename)
        or not normalized_filename.startswith(expected_prefix)
        or not normalized_filename.endswith(".json")
        or not history_suffix
        or not allowed_are_strings
        or len(set(allowed)) != len(allowed)
        or any(
            item != item.strip() or not is_safe_history_filename(item)
            for item in allowed
            if isinstance(item, str)
        )
        or allowed.count(normalized_filename) != 1
    ):
        raise ProviderResponseError(
            "SEC historical submissions filename is not uniquely authorized by its context"
        )
    _validate_sec_source_identity(
        source_asset,
        expected_cik=cik,
        expected_kind=SUBMISSIONS_HISTORY_KIND,
        expected_filename=normalized_filename,
    )
    data = _load_json_object(payload, label="SEC historical submissions")
    if (
        "cik" in data
        and _canonical_cik(
            data.get("cik"),
            label="SEC historical submissions CIK",
        )
        != cik
    ):
        raise ProviderResponseError(
            "SEC historical submissions payload CIK does not match its source asset"
        )
    columns = data
    # Preserve the ingestion recovery compatibility path for an archived
    # response that carries the current-submissions envelope at a history
    # identity. The selected table is still parsed by the same strict column
    # owner below; a missing/non-object nested table is never treated as empty.
    if not all(field in columns for field in _SUBMISSIONS_REQUIRED_COLUMNS):
        filings = data.get("filings")
        recent = filings.get("recent") if isinstance(filings, dict) else None
        if not isinstance(recent, dict):
            raise ProviderResponseError(
                "SEC historical submissions columns were missing or malformed"
            )
        columns = recent
    return _records_from_columns(columns, source_asset=source_asset)


def _parse_current_submissions(
    payload: bytes,
    *,
    source_asset: DataAsset,
) -> tuple[FilingRecord, ...]:
    """Compatibility wrapper for pre-shared internal callers."""
    return derive_sec_current_submissions(
        payload,
        source_asset=source_asset,
        expected_cik=source_asset.subject,
    ).filings


def _parse_submission_rows(
    payload: bytes,
    *,
    source_asset: DataAsset,
) -> tuple[FilingRecord, ...]:
    """Compatibility parser retained for callers without context authority."""
    return _records_from_columns(
        _load_json_object(payload, label="SEC historical submissions"),
        source_asset=source_asset,
    )


def _canonical_cik(raw: object, *, label: str) -> str:
    if isinstance(raw, bool):
        raise ProviderResponseError(f"{label} is invalid")
    if isinstance(raw, int):
        text = str(raw)
    elif isinstance(raw, str):
        text = raw.strip()
    else:
        raise ProviderResponseError(f"{label} is invalid")
    if not text or len(text) > 10 or not text.isascii() or not text.isdigit():
        raise ProviderResponseError(f"{label} is invalid")
    return text.zfill(10)


def _validate_sec_source_identity(
    source_asset: DataAsset,
    *,
    expected_cik: str,
    expected_kind: str,
    expected_filename: str | None = None,
) -> None:
    metadata = source_asset.metadata if isinstance(source_asset.metadata, dict) else {}
    if (
        source_asset.pk is None
        or source_asset.provider != PROVIDER
        or source_asset.kind != expected_kind
        or source_asset.subject != expected_cik
        or (
            expected_filename is not None
            and metadata.get(HISTORY_FILENAME_METADATA_KEY) != expected_filename
        )
    ):
        raise ProviderResponseError("SEC source asset identity is incompatible")


def _records_from_columns(
    columns: dict[str, object],
    *,
    source_asset: DataAsset,
) -> tuple[FilingRecord, ...]:
    required: dict[str, list[object]] = {}
    for field in _SUBMISSIONS_REQUIRED_COLUMNS:
        values = columns.get(field)
        if not isinstance(values, list):
            raise ProviderResponseError(
                f"SEC submissions column {field!r} was missing or malformed"
            )
        required[field] = values
    accessions = required["accessionNumber"]
    row_count = len(accessions)
    if any(len(values) != row_count for values in required.values()):
        raise ProviderResponseError("SEC submissions required column lengths do not match")
    optional: dict[str, list[object]] = {}
    for field in _SUBMISSIONS_OPTIONAL_COLUMNS:
        if field not in columns:
            continue
        values = columns[field]
        if not isinstance(values, list) or len(values) != row_count:
            raise ProviderResponseError(
                f"SEC submissions optional column {field!r} has an invalid length or type"
            )
        optional[field] = values
    records: list[FilingRecord] = []
    for index, raw_accession in enumerate(accessions):
        accession = _required_cell_text(raw_accession, label="accessionNumber")
        filing_date = _required_cell_date(
            required["filingDate"][index],
            label="filingDate",
        )
        filing_form = _required_cell_text(required["form"][index], label="form").upper()
        report_date = _optional_cell_date(
            required["reportDate"][index],
            label="reportDate",
        )
        primary_document = _optional_cell_text(
            required["primaryDocument"][index],
            label="primaryDocument",
        )
        acceptance_at, acceptance_basis = _filing_acceptance(
            optional.get("acceptanceDateTime", [None] * row_count)[index],
            filing_date=filing_date,
        )
        records.append(
            FilingRecord(
                accession=accession,
                filing_date=filing_date,
                acceptance_at=acceptance_at,
                acceptance_basis=acceptance_basis,
                filing_form=filing_form,
                report_date=report_date,
                primary_document=primary_document,
                source_asset=source_asset,
            )
        )
    return tuple(_filing_index(records).values())


def _filing_index(records: list[FilingRecord]) -> dict[str, FilingRecord]:
    index: dict[str, FilingRecord] = {}
    for record in records:
        existing = index.get(record.accession)
        if existing is None:
            index[record.accession] = record
            continue
        index[record.accession] = _merge_filing_records(existing, record)
    return index


def _merge_filing_records(existing: FilingRecord, incoming: FilingRecord) -> FilingRecord:
    """Collapse one accession only when all independently supplied facts agree."""
    if (
        existing.filing_date != incoming.filing_date
        or existing.filing_form != incoming.filing_form
        or (
            existing.report_date is not None
            and incoming.report_date is not None
            and existing.report_date != incoming.report_date
        )
        or (
            existing.primary_document
            and incoming.primary_document
            and existing.primary_document != incoming.primary_document
        )
        or (
            existing.acceptance_basis == "acceptance_datetime"
            and incoming.acceptance_basis == "acceptance_datetime"
            and existing.acceptance_at != incoming.acceptance_at
        )
    ):
        raise ProviderResponseError(f"SEC submissions conflict for accession {incoming.accession}")
    exact = [
        item for item in (existing, incoming) if item.acceptance_basis == "acceptance_datetime"
    ]
    acceptance_owner = exact[0] if exact else existing
    # Exact acceptance and fuller metadata win first; the current-submissions
    # source then wins an otherwise exact current/history duplicate. The final
    # immutable-source ordering makes the result independent of input order.
    source_owner = min(
        (existing, incoming),
        key=lambda item: (
            -int(item.acceptance_basis == "acceptance_datetime"),
            -int(item.report_date is not None),
            -int(bool(item.primary_document)),
            0 if item.source_asset.kind == SUBMISSIONS_KIND else 1,
            item.source_asset.retrieved_at,
            item.source_asset.available_at,
            str(item.source_asset.pk),
        ),
    )
    return FilingRecord(
        accession=existing.accession,
        filing_date=existing.filing_date,
        acceptance_at=acceptance_owner.acceptance_at,
        acceptance_basis=acceptance_owner.acceptance_basis,
        filing_form=existing.filing_form,
        report_date=existing.report_date or incoming.report_date,
        primary_document=existing.primary_document or incoming.primary_document,
        source_asset=source_owner.source_asset,
    )


def _normalize_companyfacts(
    *,
    company: Company,
    payload: bytes,
    source_asset: DataAsset,
    observed_at: datetime,
    filing_records: tuple[FilingRecord, ...],
    config: SecFundamentalsConfig,
    store: AssetStore,
) -> tuple[int, int]:
    strict_inspection = _inspect_sec_companyfacts(
        payload,
        source_asset=source_asset,
        expected_cik=company.cik,
        filing_records=filing_records,
        config=config,
        tolerate_configured_observation_errors=False,
    )
    lineage_inspection = _inspect_sec_companyfacts(
        payload,
        source_asset=source_asset,
        expected_cik=company.cik,
        filing_records=filing_records,
        config=config,
        tolerate_configured_observation_errors=True,
    )
    derivations = strict_inspection.derivations
    predecessors = _latest_facts_by_raw_lineage(
        company=company,
        current_derivations=derivations,
        current_inspection=lineage_inspection,
        current_source=source_asset,
        current_payload=payload,
        current_observed_at=observed_at,
        filing_records=filing_records,
        config=config,
        store=store,
    )
    filing_assets = {
        record.source_asset.pk: record.source_asset
        for record in filing_records
        if record.source_asset.pk is not None
    }
    if source_asset.pk is not None:
        filing_assets[source_asset.pk] = source_asset
    created = 0
    reused = 0
    with transaction.atomic():
        for derivation, predecessor in zip(derivations, predecessors, strict=True):
            filing_source_asset = filing_assets.get(derivation.filing_source_asset_id)
            if filing_source_asset is None:
                raise ProviderResponseError(
                    "SEC fact derivation names an unavailable filing source asset"
                )
            _fact, was_created = _persist_fact_derivation(
                company=company,
                derivation=derivation,
                source_asset=source_asset,
                filing_source_asset=filing_source_asset,
                observed_at=observed_at,
                lineage_predecessor=predecessor,
            )
            created += int(was_created)
            reused += int(not was_created)
    return created, reused


def _preflight_companyfacts_lineage(
    *,
    company: Company,
    payload: FundamentalSourcePayload,
    metadata: dict[str, object],
    filing_records: tuple[FilingRecord, ...],
    config: SecFundamentalsConfig,
    store: AssetStore,
) -> None:
    """Reject ambiguous raw lineage before registering its asset or event."""
    try:
        digest = hashlib.sha256(payload.content).hexdigest()
    except (TypeError, ValueError):
        raise RefreshVerificationError(
            "sec_evidence_digest_failed", "SEC evidence payload could not be checksummed"
        ) from None
    prospective = DataAsset(
        id=uuid4(),
        provider=PROVIDER,
        kind=COMPANYFACTS_KIND,
        subject=company.cik,
        relative_path="preflight/companyfacts.json",
        sha256=digest,
        retrieved_at=payload.retrieved_at,
        available_at=payload.retrieved_at,
        metadata=metadata,
    )
    inspection = _inspect_sec_companyfacts(
        payload.content,
        source_asset=prospective,
        expected_cik=company.cik,
        filing_records=filing_records,
        config=config,
        tolerate_configured_observation_errors=True,
    )
    _inspect_sec_companyfacts(
        payload.content,
        source_asset=prospective,
        expected_cik=company.cik,
        filing_records=filing_records,
        config=config,
        tolerate_configured_observation_errors=False,
    )
    _latest_facts_by_raw_lineage(
        company=company,
        current_derivations=inspection.derivations,
        current_inspection=inspection,
        current_source=prospective,
        current_payload=payload.content,
        current_observed_at=payload.retrieved_at,
        filing_records=filing_records,
        config=config,
        store=store,
    )


def _latest_facts_by_raw_lineage(
    *,
    company: Company,
    current_derivations: tuple[SecFactDerivation, ...],
    current_inspection: SecCompanyfactsInspection,
    current_source: DataAsset,
    current_payload: bytes,
    current_observed_at: datetime,
    filing_records: tuple[FilingRecord, ...],
    config: SecFundamentalsConfig,
    store: AssetStore,
) -> tuple[_PersistedLineageHead | None, ...]:
    """Resolve persisted predecessors from their immutable raw source slots.

    Companyfacts has no provider-issued observation identifier below an
    accession. Exact complete observations are therefore matched first across
    immutable payloads, then changed observations are matched only when stable
    components establish a unique correspondence.

    A candidate that cannot be bound to exactly one occurrence is refused.
    Treating it as unrelated would make the incoming value look original and
    backdate it to filing acceptance.
    """
    pairs = {
        (derivation.source_concept, derivation.accession) for derivation in current_derivations
    }
    if not pairs:
        return ()
    existing = [
        fact
        for fact in FundamentalFact.objects.filter(
            company=company,
            provider=PROVIDER,
            source_concept__in={item[0] for item in pairs},
            accession__in={item[1] for item in pairs},
        ).select_related("source_asset")
        if (fact.source_concept, fact.accession) in pairs
    ]
    assets = {
        asset.pk: asset
        for asset in DataAsset.objects.filter(
            provider=PROVIDER,
            kind=COMPANYFACTS_KIND,
            subject=company.cik,
            retrieved_at__lte=current_observed_at,
        )
    }
    assets[current_source.pk] = current_source
    events = list(
        SourceObservationEvent.objects.filter(
            provider=PROVIDER,
            kind=COMPANYFACTS_KIND,
            subject=company.cik,
            observed_at__lte=current_observed_at,
        )
        .select_related("source_asset")
        .order_by("observed_at", "recorded_at", "pk")
    )
    history: list[tuple[datetime, DataAsset, bytes | None]] = []
    event_source_ids: set[UUID] = set()
    for event in events:
        _assert_event_matches_asset(event)
        event_source_ids.add(event.source_asset_id)
        history.append(
            (
                event.observed_at,
                event.source_asset,
                (
                    current_payload
                    if (
                        event.source_asset_id == current_source.pk
                        and event.observed_at == current_observed_at
                    )
                    else None
                ),
            )
        )
    for asset in assets.values():
        if asset.pk not in event_source_ids:
            history.append(
                (
                    asset.retrieved_at,
                    asset,
                    current_payload if asset.pk == current_source.pk else None,
                )
            )
    if not any(
        source.pk == current_source.pk and observed_at == current_observed_at
        for observed_at, source, _payload in history
    ):
        history.append((current_observed_at, current_source, current_payload))
    history.sort(key=lambda item: (item[0], item[1].retrieved_at, str(item[1].pk)))

    inspections: list[SecCompanyfactsInspection] = []
    current_history_index: int | None = None
    for history_index, (observed_at, source, supplied_payload) in enumerate(history):
        payload = (
            supplied_payload
            if supplied_payload is not None
            else read_checksummed_bytes(store, source)
        )
        inspection = (
            current_inspection
            if source.pk == current_source.pk
            and observed_at == current_observed_at
            and payload == current_payload
            else _inspect_sec_companyfacts(
                payload,
                source_asset=source,
                expected_cik=company.cik,
                filing_records=filing_records,
                config=config,
                tolerate_configured_observation_errors=True,
            )
        )
        inspections.append(inspection)
        if (
            source.pk == current_source.pk
            and observed_at == current_observed_at
            and payload == current_payload
        ):
            current_history_index = history_index
    if current_history_index is None:
        raise ProviderResponseError("Current SEC observation has no lineage history position")
    resolved_lineages = reconcile_sec_observation_lineages(tuple(inspections))

    facts_by_lineage: dict[
        RawObservationLineage,
        list[_PersistedLineageHead],
    ] = {}
    for fact in existing:
        matches: set[tuple[RawObservationLineage, str]] = set()
        for (_observed_at, source, _payload), inspection, lineages in zip(
            history,
            inspections,
            resolved_lineages,
            strict=True,
        ):
            if source.pk != fact.source_asset_id:
                continue
            for observation, lineage in zip(
                inspection.configured_observations,
                lineages,
                strict=True,
            ):
                if (
                    observation.derivation is not None
                    and lineage is not None
                    and _derivation_content_matches_fact(observation.derivation, fact)
                    and observation.raw_observation_signature is not None
                ):
                    matches.add((lineage, observation.raw_observation_signature))
        if len(matches) != 1:
            raise ProviderResponseError(
                "Persisted SEC fact could not be bound to one raw-observation lineage"
            )
        lineage, signature = matches.pop()
        facts_by_lineage.setdefault(lineage, []).append(
            _PersistedLineageHead(
                fact=fact,
                raw_observation_signature=signature,
            )
        )

    latest: dict[RawObservationLineage, _PersistedLineageHead] = {}
    for lineage, entries in facts_by_lineage.items():
        newest_revision = max(entry.fact.source_revision for entry in entries)
        heads = [entry for entry in entries if entry.fact.source_revision == newest_revision]
        if len(heads) != 1:
            raise ProviderResponseError(
                "Persisted SEC raw-observation lineage has an ambiguous latest revision"
            )
        latest[lineage] = heads[0]
    current_lineages = resolved_lineages[current_history_index]
    current_by_identity = {
        (
            observation.derivation.raw_observation_lineage,
            observation.derivation.observation_hash,
        ): lineage
        for observation, lineage in zip(
            current_inspection.configured_observations,
            current_lineages,
            strict=True,
        )
        if observation.derivation is not None and lineage is not None
    }
    previous_signatures: dict[RawObservationLineage, str] = {}
    for inspection, lineages in zip(
        inspections[:current_history_index],
        resolved_lineages[:current_history_index],
        strict=True,
    ):
        for observation, lineage in zip(
            inspection.configured_observations,
            lineages,
            strict=True,
        ):
            if lineage is not None and observation.raw_observation_signature is not None:
                previous_signatures[lineage] = observation.raw_observation_signature
    result: list[_PersistedLineageHead | None] = []
    for derivation in current_derivations:
        lineage = current_by_identity.get(
            (
                derivation.raw_observation_lineage,
                derivation.observation_hash,
            )
        )
        if lineage is None:
            raise ProviderResponseError("Current SEC derivation has no reconciled raw lineage")
        head = latest.get(lineage)
        if head is not None:
            current_boundary = max(
                derivation.acceptance_at,
                current_observed_at,
            )
            replays_persisted_current_observation = bool(
                head.fact.source_asset_id == current_source.pk
                and head.fact.available_at == current_boundary
                and head.fact.observation_hash == derivation.observation_hash
            )
            head = _PersistedLineageHead(
                fact=head.fact,
                raw_observation_signature=(
                    head.raw_observation_signature
                    if replays_persisted_current_observation
                    else previous_signatures.get(
                        lineage,
                        head.raw_observation_signature,
                    )
                ),
            )
        result.append(head)
    return tuple(result)


def _derivation_content_matches_fact(
    derivation: SecFactDerivation,
    fact: FundamentalFact,
) -> bool:
    """Compare complete normalized observation content, not vintage clocks.

    Filing acceptance can be enriched by a later submissions snapshot while
    the Companyfacts observation itself remains unchanged. Those availability
    fields are deliberately excluded; every field sourced from the raw
    observation, including period and unit, is compared.
    """
    return bool(
        derivation.concept == fact.concept
        and derivation.taxonomy == fact.taxonomy
        and derivation.source_concept == fact.source_concept
        and derivation.value == fact.value
        and derivation.unit == fact.unit
        and derivation.currency == fact.currency
        and derivation.period_type == fact.period_type
        and derivation.period_identity == fact.period_identity
        and derivation.period_start == fact.period_start
        and derivation.period_end == fact.period_end
        and derivation.fiscal_year == fact.fiscal_year
        and derivation.fiscal_period == fact.fiscal_period
        and derivation.frame == fact.frame
        and derivation.accession == fact.accession
        and derivation.filing_form == fact.filing_form
        and derivation.filing_date == fact.filing_date
        and derivation.is_amendment is fact.is_amendment
    )


def derive_sec_companyfacts(
    payload: bytes,
    *,
    source_asset: DataAsset,
    expected_cik: str,
    filing_records: tuple[FilingRecord, ...],
    config: SecFundamentalsConfig,
) -> tuple[SecFactDerivation, ...]:
    """Derive canonical SEC fact candidates from supplied evidence only.

    Revision numbering, correction timing, evidence-link writes, and every
    ORM operation remain the ingestion writer's responsibility.  This pure
    derivation is consequently reusable by a fail-closed reader that must
    prove a persisted row from the exact raw bytes.
    """
    try:
        return _inspect_sec_companyfacts(
            payload,
            source_asset=source_asset,
            expected_cik=expected_cik,
            filing_records=filing_records,
            config=config,
            tolerate_configured_observation_errors=False,
        ).derivations
    except SecDerivationError:
        raise
    except (ProviderResponseError, SecEvidencePayloadError) as exc:
        raise SecDerivationError(str(exc)) from exc


def inspect_sec_companyfacts(
    payload: bytes,
    *,
    source_asset: DataAsset,
    expected_cik: str,
    filing_records: tuple[FilingRecord, ...],
    config: SecFundamentalsConfig,
) -> SecCompanyfactsInspection:
    """Inspect configured aliases without turning normalization failure into absence.

    The ingestion writer intentionally remains strict through
    :func:`derive_sec_companyfacts`.  This companion API uses the same JSON
    parser, source-identity checks, filing reconciliation, and fact derivation
    owner, but records a configured observation that cannot be normalized.
    That lets a fail-closed research reader distinguish an absent source alias
    from present-but-unusable source evidence without implementing a second
    Companyfacts parser.
    """
    try:
        return _inspect_sec_companyfacts(
            payload,
            source_asset=source_asset,
            expected_cik=expected_cik,
            filing_records=filing_records,
            config=config,
            tolerate_configured_observation_errors=True,
        )
    except SecDerivationError:
        raise
    except (ProviderResponseError, SecEvidencePayloadError) as exc:
        raise SecDerivationError(str(exc)) from exc


def _inspect_sec_companyfacts(
    payload: bytes,
    *,
    source_asset: DataAsset,
    expected_cik: str,
    filing_records: tuple[FilingRecord, ...],
    config: SecFundamentalsConfig,
    tolerate_configured_observation_errors: bool,
) -> SecCompanyfactsInspection:
    cik = _canonical_cik(expected_cik, label="expected CIK")
    _validate_sec_source_identity(
        source_asset,
        expected_cik=cik,
        expected_kind=COMPANYFACTS_KIND,
    )
    data = _load_json_object(payload, label="SEC companyfacts")
    if _canonical_cik(data.get("cik"), label="SEC companyfacts CIK") != cik:
        raise ProviderResponseError("SEC companyfacts payload CIK does not match its source asset")
    taxonomies = data.get("facts")
    if not isinstance(taxonomies, dict):
        raise ProviderResponseError("SEC companyfacts payload has no facts mapping")
    for record in filing_records:
        if record.source_asset.kind not in {SUBMISSIONS_KIND, SUBMISSIONS_HISTORY_KIND}:
            raise ProviderResponseError("SEC filing record has an incompatible source kind")
        _validate_sec_source_identity(
            record.source_asset,
            expected_cik=cik,
            expected_kind=record.source_asset.kind,
        )
    if tolerate_configured_observation_errors:
        filing_index, ambiguous_accessions = _inspection_filing_index(filing_records)
    else:
        filing_index = _filing_index(list(filing_records))
        ambiguous_accessions = frozenset()
    source_rules = config.source_concept_rules
    derivations: list[SecFactDerivation] = []
    configured_observations: list[SecConfiguredObservation] = []
    lineage_counts: dict[tuple[str, str, str, str], int] = {}
    for taxonomy, concepts in taxonomies.items():
        if taxonomy not in config.allowed_taxonomies:
            continue
        if not isinstance(concepts, dict):
            raise ProviderResponseError("Configured SEC taxonomy payload was malformed")
        for source_name, concept_payload in concepts.items():
            rule = source_rules.get((taxonomy, source_name))
            if rule is None:
                continue
            if not isinstance(concept_payload, dict):
                if tolerate_configured_observation_errors:
                    configured_observations.append(
                        _rejected_configured_observation(
                            taxonomy=taxonomy,
                            source_name=source_name,
                            unit=None,
                            unit_supported=None,
                            observation=None,
                            filing_index=filing_index,
                            ambiguous_accessions=ambiguous_accessions,
                            rule=rule,
                            rejection_code="configured_concept_payload_malformed",
                        )
                    )
                    continue
                raise ProviderResponseError("Configured SEC concept payload was malformed")
            units = concept_payload.get("units")
            if not isinstance(units, dict):
                if tolerate_configured_observation_errors:
                    configured_observations.append(
                        _rejected_configured_observation(
                            taxonomy=taxonomy,
                            source_name=source_name,
                            unit=None,
                            unit_supported=None,
                            observation=None,
                            filing_index=filing_index,
                            ambiguous_accessions=ambiguous_accessions,
                            rule=rule,
                            rejection_code="configured_units_payload_malformed",
                        )
                    )
                    continue
                raise ProviderResponseError("Configured SEC concept units were malformed")
            for unit, observations in units.items():
                unit_supported = unit in rule.units
                if not isinstance(observations, list):
                    if tolerate_configured_observation_errors:
                        configured_observations.append(
                            _rejected_configured_observation(
                                taxonomy=taxonomy,
                                source_name=source_name,
                                unit=unit,
                                unit_supported=unit_supported,
                                observation=None,
                                filing_index=filing_index,
                                ambiguous_accessions=ambiguous_accessions,
                                rule=rule,
                                rejection_code="configured_observation_list_malformed",
                            )
                        )
                        continue
                    raise ProviderResponseError("Configured SEC observation list was malformed")
                for observation in observations:
                    if not isinstance(observation, dict):
                        if tolerate_configured_observation_errors:
                            configured_observations.append(
                                _rejected_configured_observation(
                                    taxonomy=taxonomy,
                                    source_name=source_name,
                                    unit=unit,
                                    unit_supported=unit_supported,
                                    observation=None,
                                    filing_index=filing_index,
                                    ambiguous_accessions=ambiguous_accessions,
                                    rule=rule,
                                    rejection_code="configured_observation_malformed",
                                )
                            )
                            continue
                        raise ProviderResponseError("Configured SEC observation was malformed")
                    accession = _best_effort_text(observation.get("accn"))
                    raw_lineage, raw_signature, raw_match_keys = _raw_observation_provenance(
                        taxonomy=taxonomy,
                        source_name=source_name,
                        unit=unit,
                        observation=observation,
                        lineage_counts=lineage_counts,
                    )
                    if not unit_supported and not tolerate_configured_observation_errors:
                        continue
                    if accession in ambiguous_accessions:
                        if not tolerate_configured_observation_errors:
                            raise ProviderResponseError(
                                f"SEC submissions conflict for accession {accession}"
                            )
                        configured_observations.append(
                            _rejected_configured_observation(
                                taxonomy=taxonomy,
                                source_name=source_name,
                                unit=unit,
                                unit_supported=unit_supported,
                                observation=observation,
                                filing_index=filing_index,
                                ambiguous_accessions=ambiguous_accessions,
                                rule=rule,
                                rejection_code="filing_availability_ambiguous",
                                raw_observation_lineage=raw_lineage,
                                raw_observation_signature=raw_signature,
                                raw_observation_match_keys=raw_match_keys,
                            )
                        )
                        continue
                    if not unit_supported:
                        configured_observations.append(
                            _rejected_configured_observation(
                                taxonomy=taxonomy,
                                source_name=source_name,
                                unit=unit,
                                unit_supported=False,
                                observation=observation,
                                filing_index=filing_index,
                                ambiguous_accessions=ambiguous_accessions,
                                rule=rule,
                                rejection_code="unsupported_unit",
                                raw_observation_lineage=raw_lineage,
                                raw_observation_signature=raw_signature,
                                raw_observation_match_keys=raw_match_keys,
                            )
                        )
                        continue
                    try:
                        derivation = _derive_sec_fact(
                            taxonomy=taxonomy,
                            source_name=source_name,
                            unit=unit,
                            observation=observation,
                            source_asset=source_asset,
                            filing_index=filing_index,
                            config=config,
                            rule=rule,
                            raw_observation_lineage=raw_lineage,
                            raw_observation_signature=raw_signature,
                            raw_observation_match_keys=raw_match_keys,
                        )
                    except ProviderResponseError:
                        if not tolerate_configured_observation_errors:
                            raise
                        configured_observations.append(
                            _rejected_configured_observation(
                                taxonomy=taxonomy,
                                source_name=source_name,
                                unit=unit,
                                unit_supported=True,
                                observation=observation,
                                filing_index=filing_index,
                                ambiguous_accessions=ambiguous_accessions,
                                rule=rule,
                                rejection_code="configured_fields_malformed",
                                raw_observation_lineage=raw_lineage,
                                raw_observation_signature=raw_signature,
                                raw_observation_match_keys=raw_match_keys,
                            )
                        )
                        continue
                    if derivation is not None:
                        derivations.append(derivation)
                        configured_observations.append(
                            _configured_observation_from_derivation(
                                derivation,
                                observation=observation,
                            )
                        )
                    elif tolerate_configured_observation_errors:
                        reconciled_filing = (
                            filing_index.get(accession) if accession is not None else None
                        )
                        configured_observations.append(
                            _rejected_configured_observation(
                                taxonomy=taxonomy,
                                source_name=source_name,
                                unit=unit,
                                unit_supported=True,
                                observation=observation,
                                filing_index=filing_index,
                                ambiguous_accessions=ambiguous_accessions,
                                rule=rule,
                                rejection_code="filing_form_not_allowed",
                                status=(
                                    "excluded" if reconciled_filing is not None else "rejected"
                                ),
                                raw_observation_lineage=raw_lineage,
                                raw_observation_signature=raw_signature,
                                raw_observation_match_keys=raw_match_keys,
                            )
                        )
    return SecCompanyfactsInspection(
        derivations=tuple(derivations),
        configured_observations=tuple(configured_observations),
    )


def _raw_observation_provenance(
    *,
    taxonomy: str,
    source_name: str,
    unit: str,
    observation: dict[str, object],
    lineage_counts: dict[tuple[str, str, str, str], int],
) -> tuple[RawObservationLineage | None, str, tuple[str, ...]]:
    """Name one raw source occurrence and hash all of its supplied content."""
    source_concept = f"{taxonomy}:{source_name}"
    accession = _best_effort_text(observation.get("accn"))
    lineage: RawObservationLineage | None = None
    canonical_payload: dict[str, object] = {
        "taxonomy": taxonomy,
        "source_concept": source_concept,
        "unit": unit,
        "observation": observation,
    }
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if accession is not None:
        owner = (taxonomy, source_concept, accession)
        signature_owner = (*owner, signature)
        occurrence = lineage_counts.get(signature_owner, 0)
        lineage_counts[signature_owner] = occurrence + 1
        lineage = (*owner, f"{signature}:{occurrence}")
    match_keys = tuple(
        _raw_observation_component_hash(
            canonical_payload,
            excluded_observation_fields=excluded_fields,
            exclude_unit=exclude_unit,
        )
        for excluded_fields, exclude_unit in (
            (frozenset(("val",)), False),
            (frozenset(), True),
            (frozenset(("start", "end", "frame")), True),
        )
    )
    return lineage, signature, match_keys


def _raw_observation_component_hash(
    payload: dict[str, object],
    *,
    excluded_observation_fields: frozenset[str],
    exclude_unit: bool,
) -> str:
    observation = payload["observation"]
    assert isinstance(observation, dict)
    reduced = {
        "taxonomy": payload["taxonomy"],
        "source_concept": payload["source_concept"],
        "observation": {
            key: value
            for key, value in observation.items()
            if key not in excluded_observation_fields
        },
    }
    if not exclude_unit:
        reduced["unit"] = payload["unit"]
    canonical = json.dumps(reduced, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _ResolvedRawObservation:
    lineage: RawObservationLineage
    signature: str
    match_keys: tuple[str, ...]


def reconcile_sec_observation_lineages(
    inspections: tuple[SecCompanyfactsInspection, ...],
) -> tuple[tuple[RawObservationLineage | None, ...], ...]:
    """Reconcile stable raw occurrences across ordered immutable snapshots.

    Exact signatures are matched first, including multiplicity. Remaining
    observations are paired only by mutually unique stable-component matches,
    or by a sole old/new remainder after all stronger matches have been
    removed. A many-to-many remainder with plausible matches is ambiguous and
    fails instead of using provider array position as evidence.
    """
    active: dict[
        tuple[str, str, str],
        list[_ResolvedRawObservation],
    ] = {}
    used: set[RawObservationLineage] = set()
    resolved_vintages: list[tuple[RawObservationLineage | None, ...]] = []

    for inspection in inspections:
        resolved: list[RawObservationLineage | None] = [
            None for _observation in inspection.configured_observations
        ]
        grouped: dict[tuple[str, str, str], list[int]] = {}
        for index, observation in enumerate(inspection.configured_observations):
            provisional = observation.raw_observation_lineage
            if provisional is None:
                continue
            grouped.setdefault(provisional[:3], []).append(index)

        next_active: dict[tuple[str, str, str], list[_ResolvedRawObservation]] = {}
        for group, indexes in grouped.items():
            previous = list(active.get(group, ()))
            unmatched_previous = set(range(len(previous)))
            unmatched_current = set(indexes)

            previous_by_signature: dict[str, list[int]] = {}
            for previous_index, item in enumerate(previous):
                previous_by_signature.setdefault(item.signature, []).append(previous_index)
            current_by_signature: dict[str, list[int]] = {}
            for current_index in indexes:
                signature = inspection.configured_observations[
                    current_index
                ].raw_observation_signature
                if signature is not None:
                    current_by_signature.setdefault(signature, []).append(current_index)
            for signature in sorted(set(previous_by_signature) & set(current_by_signature)):
                old_indexes = sorted(
                    previous_by_signature[signature],
                    key=lambda item: previous[item].lineage,
                )
                new_indexes = sorted(current_by_signature[signature])
                for old_index, current_index in zip(old_indexes, new_indexes, strict=False):
                    resolved[current_index] = previous[old_index].lineage
                    unmatched_previous.discard(old_index)
                    unmatched_current.discard(current_index)

            while unmatched_previous and unmatched_current:
                candidates_by_old = {
                    old_index: {
                        current_index
                        for current_index in unmatched_current
                        if _raw_observations_share_stable_components(
                            previous[old_index],
                            inspection.configured_observations[current_index],
                        )
                    }
                    for old_index in unmatched_previous
                }
                candidates_by_current = {
                    current_index: {
                        old_index
                        for old_index in unmatched_previous
                        if current_index in candidates_by_old[old_index]
                    }
                    for current_index in unmatched_current
                }
                unique_pairs = sorted(
                    (
                        old_index,
                        next(iter(candidates)),
                    )
                    for old_index, candidates in candidates_by_old.items()
                    if len(candidates) == 1
                    and len(candidates_by_current[next(iter(candidates))]) == 1
                )
                if not unique_pairs:
                    break
                for old_index, current_index in unique_pairs:
                    resolved[current_index] = previous[old_index].lineage
                    unmatched_previous.discard(old_index)
                    unmatched_current.discard(current_index)

            if len(unmatched_previous) == 1 and len(unmatched_current) == 1:
                old_index = next(iter(unmatched_previous))
                current_index = next(iter(unmatched_current))
                resolved[current_index] = previous[old_index].lineage
                unmatched_previous.clear()
                unmatched_current.clear()
            elif unmatched_previous and unmatched_current:
                raise SecDerivationError(
                    "SEC same-accession raw observations have ambiguous lineage"
                )

            for current_index in sorted(unmatched_current):
                observation = inspection.configured_observations[current_index]
                provisional = observation.raw_observation_lineage
                signature = observation.raw_observation_signature
                if provisional is None or signature is None:
                    continue
                candidate = provisional
                suffix = 1
                while candidate in used:
                    candidate = (*group, f"{signature}:{suffix}")
                    suffix += 1
                resolved[current_index] = candidate

            group_state: list[_ResolvedRawObservation] = []
            for current_index in indexes:
                observation = inspection.configured_observations[current_index]
                lineage = resolved[current_index]
                signature = observation.raw_observation_signature
                if lineage is None or signature is None:
                    raise SecDerivationError(
                        "SEC configured observation has no reconcilable raw lineage"
                    )
                used.add(lineage)
                group_state.append(
                    _ResolvedRawObservation(
                        lineage=lineage,
                        signature=signature,
                        match_keys=observation.raw_observation_match_keys,
                    )
                )
            next_active[group] = group_state
        active.update(next_active)
        resolved_vintages.append(tuple(resolved))
    return tuple(resolved_vintages)


def _raw_observations_share_stable_components(
    previous: _ResolvedRawObservation,
    current: SecConfiguredObservation,
) -> bool:
    return bool(
        previous.match_keys
        and current.raw_observation_match_keys
        and set(previous.match_keys).intersection(current.raw_observation_match_keys)
    )


def _inspection_filing_index(
    records: tuple[FilingRecord, ...],
) -> tuple[dict[str, FilingRecord], frozenset[str]]:
    """Build a deterministic index while retaining conflicting accessions."""
    index: dict[str, FilingRecord] = {}
    ambiguous: set[str] = set()
    for record in records:
        if record.accession in ambiguous:
            continue
        existing = index.get(record.accession)
        if existing is None:
            index[record.accession] = record
            continue
        try:
            index[record.accession] = _merge_filing_records(existing, record)
        except ProviderResponseError:
            index.pop(record.accession, None)
            ambiguous.add(record.accession)
    return index, frozenset(ambiguous)


def _configured_observation_from_derivation(
    derivation: SecFactDerivation,
    *,
    observation: dict[str, object],
) -> SecConfiguredObservation:
    return SecConfiguredObservation(
        concept=derivation.concept,
        taxonomy=derivation.taxonomy,
        source_concept=derivation.source_concept,
        unit=derivation.unit,
        unit_supported=True,
        status="derived",
        rejection_code=None,
        period_type=derivation.period_type,
        period_start=derivation.period_start,
        period_end=derivation.period_end,
        accession=derivation.accession,
        raw_filing_form=_raw_companyfacts_form(observation),
        filing_form=derivation.filing_form,
        filing_date=derivation.filing_date,
        acceptance_at=derivation.acceptance_at,
        filing_availability_basis=derivation.filing_availability_basis,
        filing_source_asset_id=derivation.filing_source_asset_id,
        derivation=derivation,
        raw_observation_lineage=derivation.raw_observation_lineage,
        raw_observation_signature=derivation.raw_observation_signature,
        raw_observation_match_keys=derivation.raw_observation_match_keys,
    )


def _rejected_configured_observation(
    *,
    taxonomy: str,
    source_name: str,
    unit: str | None,
    unit_supported: bool | None,
    observation: dict[str, object] | None,
    filing_index: dict[str, FilingRecord],
    ambiguous_accessions: frozenset[str],
    rule: SecConceptRule,
    rejection_code: str,
    status: str = "rejected",
    raw_observation_lineage: RawObservationLineage | None = None,
    raw_observation_signature: str | None = None,
    raw_observation_match_keys: tuple[str, ...] = (),
) -> SecConfiguredObservation:
    accession = _best_effort_text(observation.get("accn")) if observation is not None else None
    filing = (
        filing_index.get(accession)
        if accession is not None and accession not in ambiguous_accessions
        else None
    )
    period_start = _best_effort_date(observation.get("start")) if observation is not None else None
    period_end = _best_effort_date(observation.get("end")) if observation is not None else None
    raw_filing_form = _raw_companyfacts_form(observation)
    normalized_raw_form = _best_effort_text(raw_filing_form)
    filing_form = (
        filing.filing_form
        if filing is not None
        else (normalized_raw_form.upper() if normalized_raw_form is not None else None)
    )
    raw_filing_date = (
        _best_effort_date(observation.get("filed")) if observation is not None else None
    )
    filing_date = filing.filing_date if filing is not None else raw_filing_date
    acceptance_at = filing.acceptance_at if filing is not None else None
    availability_basis = filing.acceptance_basis if filing is not None else None
    filing_source_asset_id = filing.source_asset.pk if filing is not None else None
    if acceptance_at is None and filing_date is not None and accession not in ambiguous_accessions:
        acceptance_at = datetime.combine(
            filing_date + timedelta(days=1),
            time.min,
            tzinfo=_NEW_YORK,
        )
        availability_basis = "filed_date_next_day"
    return SecConfiguredObservation(
        concept=rule.canonical_concept,
        taxonomy=taxonomy,
        source_concept=f"{taxonomy}:{source_name}",
        unit=unit,
        unit_supported=unit_supported,
        status=status,
        rejection_code=rejection_code,
        period_type=rule.period_type,
        period_start=period_start,
        period_end=period_end,
        accession=accession,
        raw_filing_form=raw_filing_form,
        filing_form=filing_form,
        filing_date=filing_date,
        acceptance_at=acceptance_at,
        filing_availability_basis=availability_basis,
        filing_source_asset_id=filing_source_asset_id,
        derivation=None,
        raw_observation_lineage=raw_observation_lineage,
        raw_observation_signature=raw_observation_signature,
        raw_observation_match_keys=raw_observation_match_keys,
    )


def _raw_companyfacts_form(observation: dict[str, object] | None) -> str | None:
    """Return the provider cell unchanged for configured-observation evidence."""
    if observation is None:
        return None
    raw_form = observation.get("form")
    return raw_form if isinstance(raw_form, str) else None


def _best_effort_text(raw: object) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def _best_effort_date(raw: object) -> date | None:
    text_value = _best_effort_text(raw)
    if text_value is None:
        return None
    try:
        return date.fromisoformat(text_value)
    except ValueError:
        return None


def _derive_sec_fact(
    *,
    taxonomy: str,
    source_name: str,
    unit: str,
    observation: dict[str, object],
    source_asset: DataAsset,
    filing_index: dict[str, FilingRecord],
    config: SecFundamentalsConfig,
    rule: SecConceptRule,
    raw_observation_lineage: RawObservationLineage | None,
    raw_observation_signature: str,
    raw_observation_match_keys: tuple[str, ...],
) -> SecFactDerivation | None:
    accession = _required_cell_text(observation.get("accn"), label="Companyfacts accn")
    if raw_observation_lineage is None:
        raise ProviderResponseError("Configured SEC observation has no stable source lineage")
    period_end = _required_cell_date(observation.get("end"), label="Companyfacts end")
    filing = filing_index.get(accession)
    filing_form = _optional_cell_text(observation.get("form"), label="Companyfacts form").upper()
    if filing is not None:
        if filing_form and filing_form != filing.filing_form:
            raise ProviderResponseError(
                f"SEC Companyfacts form conflicts with submissions for accession {accession}"
            )
        filing_form = filing_form or filing.filing_form
    elif not filing_form:
        raise ProviderResponseError("SEC Companyfacts form was missing without submissions")
    companyfacts_filed = _optional_cell_date(
        observation.get("filed"),
        label="Companyfacts filed",
    )
    if filing is not None and (
        companyfacts_filed is not None and companyfacts_filed != filing.filing_date
    ):
        raise ProviderResponseError(
            f"SEC Companyfacts filed date conflicts with submissions for accession {accession}"
        )
    filing_date = filing.filing_date if filing is not None else companyfacts_filed
    if filing_date is None:
        raise ProviderResponseError("SEC Companyfacts filed date was missing without submissions")
    if filing is not None and filing.report_date is not None and period_end > filing.report_date:
        raise ProviderResponseError(
            f"SEC Companyfacts end is after submissions reportDate for accession {accession}"
        )
    if filing_form not in config.allowed_forms:
        return None
    period_start = _optional_cell_date(
        observation.get("start"),
        label="Companyfacts start",
    )
    period_type = _period_type(rule=rule, period_start=period_start)
    if period_type is None or (
        rule.period_type == FundamentalFact.PeriodType.DURATION and period_start is None
    ):
        raise ProviderResponseError("Configured SEC observation period was malformed")
    acceptance_at = filing.acceptance_at if filing is not None else None
    filing_source_asset = filing.source_asset if filing is not None else source_asset
    availability_basis = filing.acceptance_basis if filing is not None else "filed_date_next_day"
    if acceptance_at is None:
        acceptance_at = datetime.combine(
            filing_date + timedelta(days=1),
            time.min,
            tzinfo=_NEW_YORK,
        )
    value = _parse_decimal(observation.get("val"))
    if value is None:
        raise ProviderResponseError("Configured SEC observation value was malformed")
    fiscal_year = _optional_cell_int(observation.get("fy"), label="Companyfacts fy")
    fiscal_period = _optional_cell_text(
        observation.get("fp"),
        label="Companyfacts fp",
    ).upper()
    frame = _optional_cell_text(
        observation.get("frame"),
        label="Companyfacts frame",
    ).upper()
    period_identity = build_period_identity(
        period_type=period_type,
        period_start=period_start,
        period_end=period_end,
        fiscal_period=fiscal_period,
        frame=frame,
    )
    source_concept = f"{taxonomy}:{source_name}"
    currency = "USD" if unit.startswith("USD") else ""
    observation_hash = build_observation_hash(
        taxonomy=taxonomy,
        source_concept=source_concept,
        value=value,
        unit=unit,
        currency=currency,
        period_identity=period_identity,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        accession=accession,
        filing_form=filing_form,
        filing_date=filing_date,
        acceptance_at=acceptance_at,
        frame=frame,
    )
    quality_flags: list[str] = []
    if source_asset.retrieved_at > acceptance_at + timedelta(minutes=1):
        quality_flags.append("research_reconstruction")
    if availability_basis != "acceptance_datetime":
        quality_flags.append("availability_filed_date_fallback")
    if filing_form.endswith("/A"):
        quality_flags.append("amendment")
    if period_type == FundamentalFact.PeriodType.UNCLASSIFIED:
        quality_flags.append("unclassified_period")
    if rule.explanation_only:
        quality_flags.append("explanation_only")
    if filing_source_asset.pk is None:
        raise ProviderResponseError("SEC filing source asset has no persistent identity")
    return SecFactDerivation(
        concept=rule.canonical_concept,
        taxonomy=taxonomy,
        source_concept=source_concept,
        value=value,
        unit=unit,
        currency=currency,
        period_type=period_type,
        period_identity=period_identity,
        period_start=period_start,
        period_end=period_end,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        frame=frame,
        accession=accession,
        filing_form=filing_form,
        filing_date=filing_date,
        acceptance_at=acceptance_at,
        filing_availability_basis=availability_basis,
        is_amendment=filing_form.endswith("/A"),
        observation_hash=observation_hash,
        base_quality_flags=tuple(quality_flags),
        filing_source_asset_id=filing_source_asset.pk,
        raw_observation_lineage=raw_observation_lineage,
        raw_observation_signature=raw_observation_signature,
        raw_observation_match_keys=raw_observation_match_keys,
    )


def _persist_fact_derivation(
    *,
    company: Company,
    derivation: SecFactDerivation,
    source_asset: DataAsset,
    filing_source_asset: DataAsset,
    observed_at: datetime,
    lineage_predecessor: _PersistedLineageHead | None,
) -> tuple[FundamentalFact, bool]:
    identity = FundamentalFact.objects.filter(
        company=company,
        provider=PROVIDER,
        source_concept=derivation.source_concept,
        period_identity=derivation.period_identity,
        accession=derivation.accession,
        unit=derivation.unit,
    )
    latest = lineage_predecessor.fact if lineage_predecessor is not None else None
    if latest is None and identity.exists():
        raise ProviderResponseError(
            "Distinct SEC raw-observation lineages collide on one normalized fact identity"
        )
    rebinds_unproven_correction = False
    same_normalized_observation = bool(
        latest is not None
        and latest.observation_hash == derivation.observation_hash
        and latest.concept == derivation.concept
    )
    if same_normalized_observation and latest is not None:
        rebinds_unproven_correction = _needs_observation_rebinding(identity=identity, latest=latest)
        if (
            lineage_predecessor is not None
            and lineage_predecessor.raw_observation_signature
            == derivation.raw_observation_signature
            and not rebinds_unproven_correction
        ):
            FundamentalFactEvidence.objects.get_or_create(
                fact=latest,
                role=FundamentalFactEvidence.Role.FILING,
                defaults={"source_asset": filing_source_asset},
            )
            return latest, False
    quality_flags = list(derivation.base_quality_flags)
    # A later revision under an accession that already has one is a
    # correction: the provider restated this observation without filing a
    # new accession. `acceptance_at`/`filed_at` keep the original, unmodified
    # acceptance, but availability cannot: nothing before the retrieval that
    # first carried the corrected value proves that value existed, so
    # backdating it to acceptance would let a historical cutoff read a
    # correction the run could not have known.
    #
    # The boundary is this retrieval's own observation event, never
    # `source_asset.retrieved_at`. A restatement back to a previous value
    # (100 -> 101 -> 100) serves bytes that already exist, so the asset row
    # is reused and its `retrieved_at` still points at the *first* time that
    # content was seen. `max` keeps the boundary monotonic and satisfies
    # `fact_available_after_acceptance`.
    available_at = derivation.acceptance_at
    availability_basis = derivation.filing_availability_basis
    if latest is not None:
        available_at = max(derivation.acceptance_at, observed_at)
        availability_basis = CORRECTION_AVAILABILITY_BASIS
        quality_flags.append(CORRECTION_QUALITY_FLAG)
    if rebinds_unproven_correction:
        # Same economic value as the revision it follows -- the point is the
        # boundary, not the number -- so the provenance says explicitly why a
        # new vintage exists despite identical content.
        quality_flags.append(REBOUND_QUALITY_FLAG)
    next_revision = (latest.source_revision if latest is not None else 0) + 1
    revision_collision = identity.filter(source_revision=next_revision)
    if latest is not None:
        revision_collision = revision_collision.exclude(pk=latest.pk)
    if revision_collision.exists():
        raise ProviderResponseError(
            "SEC correction revision collides with a distinct normalized observation"
        )
    fact = FundamentalFact.objects.create(
        company=company,
        provider=PROVIDER,
        concept=derivation.concept,
        taxonomy=derivation.taxonomy,
        source_concept=derivation.source_concept,
        value=derivation.value,
        unit=derivation.unit,
        currency=derivation.currency,
        period_type=derivation.period_type,
        period_identity=derivation.period_identity,
        period_start=derivation.period_start,
        period_end=derivation.period_end,
        fiscal_year=derivation.fiscal_year,
        fiscal_period=derivation.fiscal_period,
        frame=derivation.frame,
        accession=derivation.accession,
        filing_form=derivation.filing_form,
        filing_date=derivation.filing_date,
        filed_at=derivation.acceptance_at,
        acceptance_at=derivation.acceptance_at,
        available_at=available_at,
        availability_basis=availability_basis,
        is_amendment=derivation.is_amendment,
        source_revision=next_revision,
        observation_hash=derivation.observation_hash,
        quality_flags=quality_flags,
        source_asset=source_asset,
    )
    FundamentalFactEvidence.objects.create(
        fact=fact,
        role=FundamentalFactEvidence.Role.FILING,
        source_asset=filing_source_asset,
    )
    return fact, True


def _needs_observation_rebinding(
    *,
    identity: QuerySet[FundamentalFact],
    latest: FundamentalFact,
) -> bool:
    """Whether an identical fresh observation must append a new vintage.

    The reuse shortcut ("same content, same row") is right almost always: a
    later retrieval of unchanged evidence is not a correction and must not
    inflate the ledger.

    It is wrong in exactly one case. When the newest revision is a *legacy*
    correction whose timing was never proven -- a reversion that deduplicated
    onto an earlier asset, or a revision from an asset retrieved before the
    one it supersedes -- returning it unchanged would waste the very proof
    that just arrived. The as-of readers would keep deferring that row and
    keep selecting the superseded value, even though a real retrieval has now
    confirmed the current content. So a new revision is appended, carrying
    the same economic value and observation hash but a genuine, observation-
    bound availability.

    Nothing is mutated or backdated: the unprovable row stays exactly as
    persisted, and the new vintage stands beside it.

    Both idempotent paths are preserved by the two short-circuits: an
    unchanged original (revision 1) and an already observation-bound
    correction both return `False`, so repeating either retrieval still
    reuses its row.
    """
    if latest.source_revision <= 1:
        return False
    if latest.availability_basis == CORRECTION_AVAILABILITY_BASIS:
        return False
    chain = list(identity.select_related("source_asset").order_by("source_revision"))
    return resolve_availability(chain)[str(latest.pk)].proven_at is None


def _period_type(*, rule: SecConceptRule, period_start: date | None) -> str | None:
    if rule.period_type == FundamentalFact.PeriodType.INSTANT:
        return FundamentalFact.PeriodType.INSTANT if period_start is None else None
    if rule.period_type == FundamentalFact.PeriodType.DURATION:
        return (
            FundamentalFact.PeriodType.DURATION
            if period_start is not None
            else FundamentalFact.PeriodType.UNCLASSIFIED
        )
    return FundamentalFact.PeriodType.UNCLASSIFIED


def _persist_sic_classification(
    *,
    company: Company,
    evidence: SecSubmissionsEvidence,
    observed_at: datetime,
    source_asset: DataAsset,
) -> int:
    code = evidence.sic
    if not code:
        return 0
    _classification, created = CompanyClassificationObservation.objects.get_or_create(
        company=company,
        provider=PROVIDER,
        scheme=SIC_SCHEME,
        code=code,
        source_asset=source_asset,
        defaults={
            "description": evidence.sic_description,
            "observed_at": observed_at,
            "available_at": observed_at,
            "quality_flags": ["current_snapshot_not_historical"],
        },
    )
    return int(created)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProviderResponseError(f"SEC evidence contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderResponseError(f"{label} JSON was malformed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProviderResponseError(f"{label} JSON was not an object")
    return parsed


def _required_cell_text(raw: object, *, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ProviderResponseError(f"SEC {label} cell was blank or invalid")
    return raw.strip()


def _optional_cell_text(raw: object, *, label: str) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ProviderResponseError(f"SEC {label} cell was invalid")
    return raw.strip()


def _required_cell_date(raw: object, *, label: str) -> date:
    text_value = _required_cell_text(raw, label=label)
    try:
        return date.fromisoformat(text_value)
    except ValueError as exc:
        raise ProviderResponseError(f"SEC {label} date {text_value!r} was invalid") from exc


def _optional_cell_date(raw: object, *, label: str) -> date | None:
    text_value = _optional_cell_text(raw, label=label)
    if not text_value:
        return None
    try:
        return date.fromisoformat(text_value)
    except ValueError as exc:
        raise ProviderResponseError(f"SEC {label} date {text_value!r} was invalid") from exc


def _filing_acceptance(raw: object, *, filing_date: date) -> tuple[datetime, str]:
    text_value = _optional_cell_text(raw, label="acceptanceDateTime")
    fallback = datetime.combine(
        filing_date + timedelta(days=1),
        time.min,
        tzinfo=_NEW_YORK,
    )
    if not text_value:
        return fallback, "filed_date_next_day"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text_value):
        try:
            acceptance_date = date.fromisoformat(text_value)
        except ValueError as exc:
            raise ProviderResponseError(
                f"SEC acceptanceDateTime {text_value!r} was invalid"
            ) from exc
        if acceptance_date != filing_date:
            raise ProviderResponseError(
                "SEC date-only acceptanceDateTime does not equal filingDate"
            )
        return fallback, "filed_date_next_day"
    if (
        re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?"
            r"(?:Z|[+-]\d{2}:\d{2})?",
            text_value,
        )
        is None
    ):
        raise ProviderResponseError(f"SEC acceptanceDateTime {text_value!r} was invalid")
    iso_value = f"{text_value[:-1]}+00:00" if text_value.endswith("Z") else text_value
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError as exc:
        raise ProviderResponseError(f"SEC acceptanceDateTime {text_value!r} was invalid") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_NEW_YORK)
    if parsed.astimezone(_NEW_YORK).date() != filing_date:
        raise ProviderResponseError(
            "SEC exact acceptanceDateTime New York date does not equal filingDate"
        )
    return parsed, "acceptance_datetime"


def _parse_decimal(raw: object) -> Decimal | None:
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return None
    if not value.is_finite():
        return None
    return value


def _optional_cell_int(raw: object, *, label: str) -> int | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ProviderResponseError(f"SEC {label} cell was invalid")
    if isinstance(raw, float) and not raw.is_integer():
        raise ProviderResponseError(f"SEC {label} cell was invalid")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ProviderResponseError(f"SEC {label} cell was invalid") from exc


def _text(raw: object) -> str:
    return raw.strip() if isinstance(raw, str) else ""

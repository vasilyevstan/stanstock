from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from time import sleep
from zoneinfo import ZoneInfo

from django.db import transaction
from django.utils import timezone

from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.fact_identity import build_observation_hash, build_period_identity
from stanstock.data.live_us import UsUniverseConfig
from stanstock.data.models import (
    Company,
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    Listing,
    ProviderRecord,
    Region,
    Security,
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

PROVIDER = sec.PROVIDER
MAPPING_KIND = "sec_ticker_mapping"
SUBMISSIONS_KIND = "sec_submissions"
SUBMISSIONS_HISTORY_KIND = "sec_submissions_history"
COMPANYFACTS_KIND = "sec_companyfacts"
SIC_SCHEME = "sec_sic"
_COMPANYFACTS_VERIFICATIONS_KEY = "companyfacts_verifications"
_COMPANYFACTS_NORMALIZATION_VERSION = "sec-companyfacts-v2"
_NEW_YORK = ZoneInfo("America/New_York")
_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class SecMappingRow:
    cik: str
    company_name: str
    ticker: str
    exchange: str


@dataclass(frozen=True, slots=True)
class FilingRecord:
    accession: str
    filing_date: date | None
    acceptance_at: datetime | None
    filing_form: str
    report_date: date | None
    primary_document: str
    source_asset: DataAsset


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


@dataclass(frozen=True, slots=True)
class SecIngestionResult:
    mapping_asset_id: str
    mapping_sha256: str
    companies: tuple[SecCompanyIngestionResult, ...]

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
    store = store or AssetStore()
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
    store = store or AssetStore()
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
        rows=parse_sec_mapping(store.read_bytes(mapping_asset.relative_path)),
    )
    selected_symbols = symbols or tuple(universe_config.symbols)
    unknown = sorted(set(selected_symbols) - expected_symbols)
    if unknown:
        raise ValueError(f"SEC ingestion requested symbols outside the US universe: {unknown}")
    companies: list[SecCompanyIngestionResult] = []
    for symbol in selected_symbols:
        if symbol in cik_config.excluded:
            continue
        companies.append(
            _ingest_company(
                mapping=cik_config.mappings[symbol],
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
    )


def _verify_cik_config(
    *,
    cik_config: SecCikConfig,
    rows: tuple[SecMappingRow, ...],
) -> None:
    official = {(row.ticker, row.cik, row.exchange) for row in rows}
    for mapping in cik_config.mappings.values():
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
    current_records = list(
        _parse_current_submissions(
            submissions_payload.content,
            source_asset=submissions_asset,
        )
    )
    raw_created = int(submissions_created)
    raw_reused = int(not submissions_created)
    history_filenames = _historical_submission_filenames(submissions_payload.content)
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
    latest_companyfacts = (
        DataAsset.objects.filter(
            provider=PROVIDER,
            kind=COMPANYFACTS_KIND,
            subject=mapping.cik,
        )
        .order_by("-retrieved_at")
        .first()
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
        latest_asset=latest_companyfacts,
        last_checked_at=(
            verified_companyfacts.checked_at if verified_companyfacts is not None else None
        ),
        days=config.submissions_reconciliation_days,
    )
    known_companyfacts_accessions = (
        _companyfacts_accessions(store.read_bytes(latest_companyfacts.relative_path))
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
                        extra={"filename": filename},
                    ),
                )
                raw_created += int(history_created)
                raw_reused += int(not history_created)
                history_content = history_payload.content
            else:
                raw_reused += 1
                history_content = store.read_bytes(history_asset.relative_path)
            resolved_history_assets[filename] = history_asset
            historical_count += 1
            filing_records.extend(
                _parse_submission_rows(
                    history_content,
                    source_asset=history_asset,
                )
            )
        filing_sources_hash = _filing_sources_hash(
            submissions_asset=submissions_asset,
            history_assets=resolved_history_assets,
        )
        filing_index = _filing_index(filing_records)

    if fetch_companyfacts:
        budget.consume()
        companyfacts_payload = sec.fetch_companyfacts(mapping.cik)
        companyfacts_asset, companyfacts_created = _persist_payload(
            store=store,
            payload=companyfacts_payload,
            kind=COMPANYFACTS_KIND,
            subject=mapping.cik,
            metadata=_source_metadata(
                payload=companyfacts_payload,
                config=config,
                target_date=target_date,
                symbol=mapping.symbol,
                extra={
                    "submissions_asset_id": str(submissions_asset.pk),
                    "submissions_sha256": submissions_asset.sha256,
                    "reconciliation": reconciliation_due,
                },
            ),
        )
        raw_created += int(companyfacts_created)
        raw_reused += int(not companyfacts_created)
        companyfacts_content = companyfacts_payload.content
    elif normalize_companyfacts:
        if latest_companyfacts is None:
            raise RuntimeError("SEC companyfacts normalization state is inconsistent")
        companyfacts_asset = latest_companyfacts
        companyfacts_content = store.read_bytes(companyfacts_asset.relative_path)
        raw_reused += 1

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
            filing_index=filing_index,
            config=config,
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
        submissions_payload=submissions_payload,
        source_asset=submissions_asset,
    )
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
    )


def _latest_history_asset(*, cik: str, filename: str) -> DataAsset | None:
    assets = DataAsset.objects.filter(
        provider=PROVIDER,
        kind=SUBMISSIONS_HISTORY_KIND,
        subject=cik,
    ).order_by("-retrieved_at")
    for asset in assets:
        if asset.metadata.get("filename") == filename:
            return asset
    return None


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
    latest_asset: DataAsset | None,
    last_checked_at: datetime | None,
    days: int,
) -> bool:
    if latest_asset is None:
        return True
    reference_at = last_checked_at or latest_asset.retrieved_at
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
) -> tuple[DataAsset, bool]:
    digest = hashlib.sha256(payload.content).hexdigest()
    existing = (
        DataAsset.objects.filter(
            provider=PROVIDER,
            kind=kind,
            subject=subject,
            sha256=digest,
        )
        .order_by("-retrieved_at")
        .first()
    )
    if existing is not None:
        return existing, False
    stamp = payload.retrieved_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    safe_subject = _SAFE_PATH_COMPONENT.sub("_", subject)
    relative_path = f"raw/sec/{kind}/{safe_subject}/{stamp}-{digest[:12]}.json"
    stored = store.write_bytes(relative_path, payload.content)
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
    except Exception:
        if not DataAsset.objects.filter(relative_path=relative_path).exists():
            store.resolve(relative_path).unlink(missing_ok=True)
        raise
    return asset, True


def _parse_current_submissions(
    payload: bytes,
    *,
    source_asset: DataAsset,
) -> tuple[FilingRecord, ...]:
    data = _load_json_object(payload, label="SEC submissions")
    filings = data.get("filings")
    recent = filings.get("recent") if isinstance(filings, dict) else None
    if not isinstance(recent, dict):
        return ()
    return _records_from_columns(recent, source_asset=source_asset)


def _historical_submission_filenames(payload: bytes) -> tuple[str, ...]:
    data = _load_json_object(payload, label="SEC submissions")
    filings = data.get("filings")
    files = filings.get("files") if isinstance(filings, dict) else None
    if not isinstance(files, list):
        return ()
    names: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return tuple(dict.fromkeys(names))


def _parse_submission_rows(
    payload: bytes,
    *,
    source_asset: DataAsset,
) -> tuple[FilingRecord, ...]:
    return _records_from_columns(
        _load_json_object(payload, label="SEC historical submissions"),
        source_asset=source_asset,
    )


def _records_from_columns(
    columns: dict[str, object],
    *,
    source_asset: DataAsset,
) -> tuple[FilingRecord, ...]:
    accessions = columns.get("accessionNumber")
    if not isinstance(accessions, list):
        return ()
    records: list[FilingRecord] = []
    for index, raw_accession in enumerate(accessions):
        if not isinstance(raw_accession, str) or not raw_accession.strip():
            continue
        records.append(
            FilingRecord(
                accession=raw_accession.strip(),
                filing_date=_parse_date(_column_value(columns, "filingDate", index)),
                acceptance_at=_parse_acceptance(
                    _column_value(columns, "acceptanceDateTime", index)
                ),
                filing_form=_text(_column_value(columns, "form", index)).upper(),
                report_date=_parse_date(_column_value(columns, "reportDate", index)),
                primary_document=_text(_column_value(columns, "primaryDocument", index)),
                source_asset=source_asset,
            )
        )
    return tuple(records)


def _filing_index(records: list[FilingRecord]) -> dict[str, FilingRecord]:
    index: dict[str, FilingRecord] = {}
    for record in records:
        existing = index.get(record.accession)
        if existing is None:
            index[record.accession] = record
            continue
        same_metadata = (
            existing.filing_date == record.filing_date
            and existing.acceptance_at == record.acceptance_at
            and existing.filing_form == record.filing_form
            and existing.report_date == record.report_date
            and existing.primary_document == record.primary_document
        )
        if same_metadata:
            index[record.accession] = min(
                (existing, record),
                key=lambda item: (
                    item.source_asset.retrieved_at,
                    str(item.source_asset.pk),
                ),
            )
            continue
        preferred = record if record.acceptance_at is not None else existing
        other = existing if preferred is record else record
        if (
            preferred.filing_date != other.filing_date
            or preferred.filing_form != other.filing_form
            or (
                preferred.acceptance_at is not None
                and other.acceptance_at is not None
                and preferred.acceptance_at != other.acceptance_at
            )
        ):
            raise ProviderResponseError(
                f"SEC submissions conflict for accession {record.accession}"
            )
        index[record.accession] = preferred
    return index


def _normalize_companyfacts(
    *,
    company: Company,
    payload: bytes,
    source_asset: DataAsset,
    filing_index: dict[str, FilingRecord],
    config: SecFundamentalsConfig,
) -> tuple[int, int]:
    data = _load_json_object(payload, label="SEC companyfacts")
    taxonomies = data.get("facts")
    if not isinstance(taxonomies, dict):
        raise ProviderResponseError("SEC companyfacts payload has no facts mapping")
    source_rules = config.source_concept_rules
    created = 0
    reused = 0
    for taxonomy, concepts in taxonomies.items():
        if taxonomy not in config.allowed_taxonomies or not isinstance(concepts, dict):
            continue
        for source_name, concept_payload in concepts.items():
            rule = source_rules.get((taxonomy, source_name))
            if rule is None or not isinstance(concept_payload, dict):
                continue
            units = concept_payload.get("units")
            if not isinstance(units, dict):
                continue
            for unit, observations in units.items():
                if unit not in rule.units or not isinstance(observations, list):
                    continue
                for observation in observations:
                    if not isinstance(observation, dict):
                        continue
                    normalized = _normalized_fact(
                        company=company,
                        taxonomy=taxonomy,
                        source_name=source_name,
                        unit=unit,
                        observation=observation,
                        source_asset=source_asset,
                        filing_index=filing_index,
                        config=config,
                        rule=rule,
                    )
                    if normalized is None:
                        continue
                    fact, was_created = normalized
                    created += int(was_created)
                    reused += int(not was_created)
                    del fact
    return created, reused


def _normalized_fact(
    *,
    company: Company,
    taxonomy: str,
    source_name: str,
    unit: str,
    observation: dict[str, object],
    source_asset: DataAsset,
    filing_index: dict[str, FilingRecord],
    config: SecFundamentalsConfig,
    rule: SecConceptRule,
) -> tuple[FundamentalFact, bool] | None:
    accession = _text(observation.get("accn"))
    period_end = _parse_date(observation.get("end"))
    if not accession or period_end is None:
        return None
    period_start = _parse_date(observation.get("start"))
    period_type = _period_type(rule=rule, period_start=period_start)
    if period_type is None:
        return None
    filing = filing_index.get(accession)
    filing_form = _text(observation.get("form")).upper()
    if filing is not None and not filing_form:
        filing_form = filing.filing_form
    if filing_form not in config.allowed_forms:
        return None
    filing_date = (
        filing.filing_date
        if filing is not None and filing.filing_date is not None
        else _parse_date(observation.get("filed"))
    )
    acceptance_at = filing.acceptance_at if filing is not None else None
    filing_source_asset = filing.source_asset if filing is not None else source_asset
    availability_basis = "acceptance_datetime"
    if acceptance_at is None:
        if filing_date is None:
            return None
        acceptance_at = datetime.combine(
            filing_date + timedelta(days=1),
            time.min,
            tzinfo=_NEW_YORK,
        )
        availability_basis = "filed_date_next_day"
    value = _parse_decimal(observation.get("val"))
    if value is None:
        return None
    fiscal_year = _parse_int(observation.get("fy"))
    fiscal_period = _text(observation.get("fp")).upper()
    frame = _text(observation.get("frame")).upper()
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
    identity = FundamentalFact.objects.filter(
        company=company,
        provider=PROVIDER,
        source_concept=source_concept,
        period_identity=period_identity,
        accession=accession,
        unit=unit,
    )
    latest = identity.order_by("-source_revision").first()
    if (
        latest is not None
        and latest.observation_hash == observation_hash
        and latest.concept == rule.canonical_concept
    ):
        FundamentalFactEvidence.objects.get_or_create(
            fact=latest,
            role=FundamentalFactEvidence.Role.FILING,
            defaults={"source_asset": filing_source_asset},
        )
        return latest, False
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
    with transaction.atomic():
        fact = FundamentalFact.objects.create(
            company=company,
            provider=PROVIDER,
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
            filed_at=acceptance_at,
            acceptance_at=acceptance_at,
            available_at=acceptance_at,
            availability_basis=availability_basis,
            is_amendment=filing_form.endswith("/A"),
            source_revision=(latest.source_revision if latest is not None else 0) + 1,
            observation_hash=observation_hash,
            quality_flags=quality_flags,
            source_asset=source_asset,
        )
        FundamentalFactEvidence.objects.create(
            fact=fact,
            role=FundamentalFactEvidence.Role.FILING,
            source_asset=filing_source_asset,
        )
    return fact, True


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
    submissions_payload: FundamentalSourcePayload,
    source_asset: DataAsset,
) -> int:
    data = _load_json_object(submissions_payload.content, label="SEC submissions")
    raw_code = data.get("sic")
    if raw_code is None:
        return 0
    code = str(raw_code).strip()
    if not code:
        return 0
    description = _text(data.get("sicDescription"))
    _classification, created = CompanyClassificationObservation.objects.get_or_create(
        company=company,
        provider=PROVIDER,
        scheme=SIC_SCHEME,
        code=code,
        source_asset=source_asset,
        defaults={
            "description": description,
            "observed_at": submissions_payload.retrieved_at,
            "available_at": submissions_payload.retrieved_at,
            "quality_flags": ["current_snapshot_not_historical"],
        },
    )
    return int(created)


def _load_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError(f"{label} JSON was malformed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProviderResponseError(f"{label} JSON was not an object")
    return parsed


def _column_value(columns: dict[str, object], field: str, index: int) -> object:
    values = columns.get(field)
    if not isinstance(values, list) or index >= len(values):
        return None
    return values[index]


def _parse_acceptance(raw: object) -> datetime | None:
    text_value = _text(raw)
    if not text_value:
        return None
    try:
        parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProviderResponseError(f"SEC acceptanceDateTime {text_value!r} was invalid") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_NEW_YORK)
    return parsed


def _parse_date(raw: object) -> date | None:
    text_value = _text(raw)
    if not text_value:
        return None
    try:
        return date.fromisoformat(text_value)
    except ValueError as exc:
        raise ProviderResponseError(f"SEC date {text_value!r} was invalid") from exc


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


def _parse_int(raw: object) -> int | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if not isinstance(raw, (int, float, str)):
        return None
    if isinstance(raw, float) and not raw.is_integer():
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _text(raw: object) -> str:
    return raw.strip() if isinstance(raw, str) else ""

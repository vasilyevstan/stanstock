from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from io import StringIO
from pathlib import Path
from typing import cast
from unittest.mock import patch
from uuid import UUID, uuid4

import polars as pl
import pytest
import yaml
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

import stanstock.research.long_forecast_config as long_config_module
import stanstock.research.long_forecasts_v4 as long_v4_module
import stanstock.research.outcomes as outcomes_module
import stanstock.research.service as service_module
from stanstock.data import sec_ingestion as sec_ingestion_module
from stanstock.data.asof import AsOfData
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.management.config_loader import (
    config_hash,
    default_us_scoring_config_path,
)
from stanstock.data.models import (
    Company,
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FundamentalFactEvidence,
    LatestMarketData,
    Listing,
    Region,
    Security,
    SourceObservationEvent,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.data.providers import sec
from stanstock.data.providers.contracts import FundamentalSourcePayload
from stanstock.data.sec_config import (
    SecCikConfig,
    SecCikMapping,
    load_sec_cik_config,
    load_sec_fundamentals_config,
)
from stanstock.data.sec_evidence import MAPPING_KIND, MAPPING_SUBJECT
from stanstock.data.sec_fundamentals import CORRECTION_AVAILABILITY_BASIS
from stanstock.data.sec_ingestion import (
    FilingRecord,
    SecDerivationError,
    derive_sec_companyfacts,
    derive_sec_current_submissions,
    derive_sec_historical_submissions,
    inspect_sec_companyfacts,
    verify_sec_exchange_mic,
)
from stanstock.research.long_forecast_config import (
    LONG_V4_CONFIG_FILE_SHA256,
    LONG_V4_EFFECTIVE_CONFIG_HASH,
    LongForecastV4Config,
    default_long_forecast_config_path,
    load_long_forecast_config,
    load_long_forecast_v4_config,
    load_long_v4_sec_cik_config,
    long_forecast_config_hash,
    long_forecast_v4_config_hash,
    long_forecast_v4_config_path,
)
from stanstock.research.long_forecasts_v4 import (
    LONG_V4_PEER_POLICY,
    LONG_V4_SCORING_CONFIG_HASH,
    PROBABILITY_REASON,
    LongForecastV4,
    build_long_forecasts_v4,
    canonical_long_v4_price,
    validate_long_v4_forecast_pair,
)
from stanstock.research.models import AnalysisRun, Prediction, PredictionOutcome, StockAnalysis
from stanstock.research.outcomes import evaluate_prediction, resolve_outcome
from stanstock.research.service import analyze_snapshot

TARGET_DATE = date(2026, 2, 27)
DECISION_TIME = datetime(2026, 3, 1, 12, tzinfo=UTC)
PRICE_AVAILABLE_AT = datetime(2026, 2, 27, 21, tzinfo=UTC)
V4_PATH = Path("config/forecasts/us-sec-long-v4.yml")

SOURCE_CONCEPTS = {
    "revenue": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "net_income": "us-gaap:NetIncomeLoss",
    "diluted_eps": "us-gaap:EarningsPerShareDiluted",
    "weighted_average_diluted_shares": ("us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"),
    "operating_cash_flow": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "capital_expenditure": "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
}
UNITS = {
    "diluted_eps": "USD/shares",
    "weighted_average_diluted_shares": "shares",
}
ANNUAL_PERIODS = (
    (date(2022, 1, 1), date(2022, 12, 31)),
    (date(2023, 1, 1), date(2023, 12, 31)),
    (date(2024, 1, 1), date(2024, 12, 31)),
    (date(2025, 1, 1), date(2025, 12, 31)),
)
QUARTERS = (
    (date(2025, 1, 1), date(2025, 3, 31), "Q1"),
    (date(2025, 4, 1), date(2025, 6, 30), "Q2"),
    (date(2025, 7, 1), date(2025, 9, 30), "Q3"),
    (date(2025, 10, 1), date(2025, 12, 31), "Q4"),
)
SOURCE_TAMPER_CASES = tuple(
    (role, mutation)
    for role in (
        "mapping",
        "normalized_price",
        "raw_price",
        "sec_source",
        "filing",
        "submissions_context",
        "classification",
    )
    for mutation in (
        "missing",
        "unreadable",
        "corrupt",
        "checksum",
        "wrong_provider",
        "wrong_kind",
        "wrong_subject",
        "late",
    )
) + (("normalized_price", "malformed"),)
_REAL_V4_MAPPING_RESOLVER = long_v4_module._resolve_sec_mapping_authority


def _annual_periods_with_duration(days: int) -> tuple[tuple[date, date], ...]:
    periods: list[tuple[date, date]] = []
    period_end = date(2025, 12, 31)
    for _index in range(4):
        period_start = period_end - timedelta(days=days - 1)
        periods.append((period_start, period_end))
        period_end = period_start - timedelta(days=1)
    return tuple(reversed(periods))


def _mapping_payload_for_config(config: SecCikConfig) -> bytes:
    return json.dumps(
        {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data": [
                [
                    int(mapping.cik),
                    mapping.company_name,
                    mapping.official_ticker,
                    mapping.exchange,
                ]
                for mapping in config.mappings.values()
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


@pytest.fixture(autouse=True)
def _synthetic_v4_mapping_authority(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep V4 tests synthetic while production pins stay literal and tested.

    The production loader is always executed first. Django-backed tests then
    replace only the four mapping identities with a deterministic temporary
    authority assembled from their synthetic listings, and the real mapping
    resolver still checksum-reads and parses the physical asset.
    """
    if request.node.get_closest_marker("django_db") is None:
        return

    production_loader = long_config_module.load_long_forecast_v4_config
    real_resolver = _REAL_V4_MAPPING_RESOLVER
    authorities: dict[str, tuple[SecCikConfig, bytes, str]] = {}

    def synthetic_authority() -> tuple[SecCikConfig, bytes, str]:
        mappings: dict[str, SecCikMapping] = {}
        for listing in (
            Listing.objects.select_related("security__company")
            .exclude(provider_symbol="")
            .order_by("ticker", "pk")
        ):
            exchange = "NYSE" if listing.exchange_mic == "XNYS" else "Nasdaq"
            mapping = SecCikMapping(
                symbol=listing.ticker,
                cik=listing.security.company.cik,
                official_ticker=listing.ticker,
                exchange=exchange,
                company_name=listing.security.company.name,
                reason="",
            )
            mappings[listing.ticker] = mapping
        raw = {
            "schema_version": 1,
            "config_version": "synthetic-sec-cik-v1",
            "universe_config_version": "synthetic-v4",
            "source_url": "https://example.invalid/synthetic-sec-mapping.json",
            "source_retrieved_at": PRICE_AVAILABLE_AT.isoformat(),
            "source_sha256": "",
            "mappings": {
                symbol: {
                    "cik": mapping.cik,
                    "official_ticker": mapping.official_ticker,
                    "exchange": mapping.exchange,
                    "company_name": mapping.company_name,
                }
                for symbol, mapping in mappings.items()
            },
            "excluded": {},
        }
        provisional = SecCikConfig(
            config_version="synthetic-sec-cik-v1",
            universe_config_version="synthetic-v4",
            source_sha256="",
            mappings=mappings,
            excluded={},
            raw=raw,
            config_hash="",
        )
        payload = _mapping_payload_for_config(provisional)
        source_sha = hashlib.sha256(payload).hexdigest()
        raw["source_sha256"] = source_sha
        cik_config = SecCikConfig(
            config_version="synthetic-sec-cik-v1",
            universe_config_version="synthetic-v4",
            source_sha256=source_sha,
            mappings=mappings,
            excluded={},
            raw=raw,
            config_hash=config_hash(raw),
        )
        file_bytes = yaml.safe_dump(raw, sort_keys=False).encode()
        file_sha = hashlib.sha256(file_bytes).hexdigest()
        authorities[source_sha] = (cik_config, payload, file_sha)
        return cik_config, payload, file_sha

    def load_v4(path: Path):
        production = production_loader(path)
        cik_config, _payload, file_sha = synthetic_authority()
        monkeypatch.setattr(
            long_v4_module,
            "LONG_V4_SEC_CIK_CONFIG_VERSION",
            cik_config.config_version,
        )
        monkeypatch.setattr(
            long_v4_module,
            "LONG_V4_SEC_CIK_CONFIG_FILE_SHA256",
            file_sha,
        )
        monkeypatch.setattr(
            long_v4_module,
            "LONG_V4_SEC_CIK_CONFIG_HASH",
            cik_config.config_hash,
        )
        monkeypatch.setattr(
            long_v4_module,
            "LONG_V4_SEC_MAPPING_SOURCE_SHA256",
            cik_config.source_sha256,
        )
        return replace(
            production,
            sec_cik_config_version=cik_config.config_version,
            sec_cik_config_file_sha256=file_sha,
            sec_cik_config_hash=cik_config.config_hash,
            sec_mapping_source_sha256=cik_config.source_sha256,
        )

    def load_cik(config):
        authority = authorities.get(config.sec_mapping_source_sha256)
        if authority is None:
            authority = synthetic_authority()
        return authority[0]

    def resolve_mapping(**kwargs):
        cik_config = kwargs["cik_config"]
        asof = kwargs["asof"]
        authority = authorities.get(cik_config.source_sha256)
        if authority is None:
            authority = synthetic_authority()
        _cik_config, payload, _file_sha = authority
        if not DataAsset.objects.filter(
            provider="sec",
            kind=MAPPING_KIND,
            subject=MAPPING_SUBJECT,
            sha256=cik_config.source_sha256,
            available_at__lte=asof.decision_time,
            retrieved_at__lte=asof.decision_time,
        ).exists():
            stored = asof.store.write_bytes(
                f"tests/long-v4/mapping/{cik_config.source_sha256}.json",
                payload,
            )
            register_asset(
                provider="sec",
                kind=MAPPING_KIND,
                subject=MAPPING_SUBJECT,
                stored=stored,
                retrieved_at=PRICE_AVAILABLE_AT,
                available_at=PRICE_AVAILABLE_AT,
            )
        return real_resolver(**kwargs)

    monkeypatch.setitem(globals(), "load_long_forecast_v4_config", load_v4)
    monkeypatch.setattr(long_v4_module, "load_long_forecast_v4_config", load_v4)
    monkeypatch.setattr(service_module, "load_long_forecast_v4_config", load_v4)
    monkeypatch.setattr(long_v4_module, "load_long_v4_sec_cik_config", load_cik)
    monkeypatch.setattr(long_v4_module, "_resolve_sec_mapping_authority", resolve_mapping)
    monkeypatch.setattr(
        long_v4_module,
        "long_forecast_v4_config_hash",
        lambda _config: LONG_V4_EFFECTIVE_CONFIG_HASH,
    )
    monkeypatch.setattr(
        service_module,
        "long_forecast_v4_config_hash",
        lambda _config: LONG_V4_EFFECTIVE_CONFIG_HASH,
    )


def test_v4_config_is_explicit_schema_2_and_legacy_defaults_are_frozen() -> None:
    import hashlib

    path = long_forecast_v4_config_path()
    config = load_long_forecast_v4_config(path)
    v1 = load_long_forecast_config(Path("config/forecasts/us-sec-long-v1.yml"))
    v2 = load_long_forecast_config(Path("config/forecasts/us-sec-long-v2.yml"))
    v3 = load_long_forecast_config(Path("config/forecasts/us-sec-long-v3.yml"))

    assert path == V4_PATH.resolve()
    assert config.schema_version == 2
    assert config.version == "us-sec-long-v4"
    assert config.peer == LONG_V4_PEER_POLICY
    assert LONG_V4_PEER_POLICY.sic_prefix_levels == (4, 3, 2)
    assert LONG_V4_PEER_POLICY.minimum_cohort == {4: 3, 3: 5, 2: 8}
    assert "per_share_minimum" not in config.raw["eligibility"]
    assert config.fundamentals_config_file_sha256 == (
        "829ed267eec62304804c9ef71f1816636389c2b0ad77423d8a534a00a5e1ae30"
    )
    assert config.fundamentals_config_hash == (
        "7822a1faaae1c8028d71851337a7dbb7a65d9aeaa0f44478e3814a6468b62604"
    )
    assert config.sec_cik_config_version == "us-sec-cik-v1"
    assert config.sec_cik_config_file_sha256 == (
        "3e65d924d77b3ea233806cdfb006ddb30a0b982fecafd9568cbf680e37ffeb07"
    )
    assert config.sec_cik_config_hash == (
        "5443bb613e1b40545faa4f53794a869453c6f78317390389c3959e938ba99f65"
    )
    assert config.sec_mapping_source_sha256 == (
        "ec43db74f82d1739cce6340f36b9695dcb51231fc38edd493215677627bb01cd"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == LONG_V4_CONFIG_FILE_SHA256
    assert long_forecast_v4_config_hash(config) == LONG_V4_EFFECTIVE_CONFIG_HASH
    assert LONG_V4_CONFIG_FILE_SHA256 == (
        "840bda0d6b3122dd4c75b9b14ec32cf64a1b1dc49e921cf88a9256f413fe81d2"
    )
    assert LONG_V4_EFFECTIVE_CONFIG_HASH == (
        "acaf8a3a8cd6975ef894a5fbce50a6a3ca3dc6dce0886f03dbf9416e285268fa"
    )
    cik_config = load_long_v4_sec_cik_config(config)
    assert cik_config.config_version == config.sec_cik_config_version
    assert cik_config.config_hash == config.sec_cik_config_hash
    assert cik_config.source_sha256 == config.sec_mapping_source_sha256
    assert cik_config.excluded == {}
    assert all(
        symbol == mapping.symbol == mapping.official_ticker and not mapping.reason
        for symbol, mapping in cik_config.mappings.items()
    )
    assert default_long_forecast_config_path().name == "us-sec-long-v2.yml"
    assert long_forecast_config_hash(v1) == (
        "ef0e0478aebf53ab605ff47df1d4732ad7cb4e7f3c5d6559b89674ad6a0adeb1"
    )
    assert long_forecast_config_hash(v2) == (
        "46a81d4bfe87d80ddcf2d62a7f05854eb36381027fb01bc40d637ef3294a5c36"
    )
    assert long_forecast_config_hash(v3) == (
        "073ac542195b0c67c8ab654aeec61266548ca2758a98ad432f91e48ee5154e57"
    )
    with pytest.raises(ValueError, match="schema_version must be 1"):
        load_long_forecast_config(V4_PATH)


def test_v4_loader_rejects_lookalikes_and_policy_drift(tmp_path: Path) -> None:
    lookalike = tmp_path / V4_PATH.name
    lookalike.write_bytes(V4_PATH.read_bytes())
    with pytest.raises(ValueError, match="exact tracked"):
        load_long_forecast_v4_config(lookalike)

    raw = deepcopy(load_long_forecast_v4_config(V4_PATH).raw)
    raw["scenarios"]["bear"]["dilution_multiplier"] = 1.5
    with pytest.raises(ValueError, match="scenario constants changed"):
        LongForecastV4Config.from_mapping(raw)

    raw = deepcopy(load_long_forecast_v4_config(V4_PATH).raw)
    raw["growth"]["tax_rate_minimum"] = 0.1
    with pytest.raises(ValueError, match="reviewed schema"):
        LongForecastV4Config.from_mapping(raw)


def test_v4_loader_rejects_same_version_sec_effective_config_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    sec_config = load_sec_fundamentals_config()
    monkeypatch.setattr(
        long_config_module,
        "load_sec_fundamentals_config",
        lambda _path: replace(sec_config, config_hash="f" * 64),
    )
    with pytest.raises(ValueError, match="SEC fundamentals effective config changed"):
        load_long_forecast_v4_config(V4_PATH)


@pytest.mark.parametrize("mutation", ["version", "effective_hash", "source_hash"])
def test_v4_loader_rejects_sec_cik_authority_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from stanstock.data.sec_config import load_sec_cik_config

    cik_config = load_sec_cik_config()
    changes = {
        "version": {"config_version": "us-sec-cik-v2"},
        "effective_hash": {"config_hash": "f" * 64},
        "source_hash": {"source_sha256": "f" * 64},
    }[mutation]
    monkeypatch.setattr(
        long_config_module,
        "load_sec_cik_config",
        lambda _path: replace(cik_config, **changes),
    )
    with pytest.raises(ValueError, match="SEC CIK authority changed"):
        load_long_forecast_v4_config(V4_PATH)


def _register_mapping_payload(
    store: AssetStore,
    *,
    payload: bytes,
    available_at: datetime = PRICE_AVAILABLE_AT,
    suffix: str = "",
) -> DataAsset:
    digest = hashlib.sha256(payload).hexdigest()
    stored = store.write_bytes(
        f"tests/long-v4/mapping/manual-{digest}{suffix}.json",
        payload,
    )
    return register_asset(
        provider="sec",
        kind=MAPPING_KIND,
        subject=MAPPING_SUBJECT,
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate_asset",
        "tampered",
        "late",
        "config",
        "cik_exclusion",
        "cik_reason",
        "cik_alias",
        "config_cik",
        "config_exchange",
        "duplicate_raw_row",
        "raw_cik",
        "raw_ticker",
        "raw_exchange",
        "listing_cik",
        "listing_ticker",
        "provider_symbol",
        "listing_mic",
    ],
)
def test_v4_sec_mapping_authority_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    store = AssetStore(tmp_path)
    listings, prices, assets = _cohort(store)
    ordered = tuple(sorted(listings, key=lambda item: (item.ticker, str(item.pk))))
    config = load_long_forecast_v4_config(V4_PATH)
    cik_config = long_v4_module.load_long_v4_sec_cik_config(config)
    payload = _mapping_payload_for_config(cik_config)

    if mutation == "config":
        config = replace(config, sec_cik_config_hash="f" * 64)
    elif mutation == "cik_exclusion":
        cik_config = replace(cik_config, excluded={"EXCLUDED": "synthetic"})
    elif mutation in {"cik_reason", "cik_alias", "config_cik", "config_exchange"}:
        mappings = dict(cik_config.mappings)
        symbol = next(iter(mappings))
        mapping = mappings[symbol]
        mappings[symbol] = replace(
            mapping,
            reason="synthetic alias" if mutation == "cik_reason" else "",
            official_ticker=(
                f"{mapping.official_ticker}.A"
                if mutation == "cik_alias"
                else mapping.official_ticker
            ),
            cik=("0000000001" if mutation == "config_cik" else mapping.cik),
            exchange=("NYSE" if mutation == "config_exchange" else mapping.exchange),
        )
        cik_config = replace(cik_config, mappings=mappings)
    elif mutation in {"duplicate_raw_row", "raw_cik", "raw_ticker", "raw_exchange"}:
        document = json.loads(payload)
        if mutation == "duplicate_raw_row":
            document["data"].append(list(document["data"][0]))
        elif mutation == "raw_cik":
            document["data"][0][0] += 1
        elif mutation == "raw_ticker":
            document["data"][0][2] = "WRONG"
        else:
            document["data"][0][3] = "NYSE"
        payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(payload).hexdigest()
        config = replace(config, sec_mapping_source_sha256=digest)
        cik_config = replace(cik_config, source_sha256=digest)
        monkeypatch.setattr(long_v4_module, "LONG_V4_SEC_MAPPING_SOURCE_SHA256", digest)

    if mutation != "missing":
        mapping_asset = _register_mapping_payload(
            store,
            payload=payload,
            available_at=(
                DECISION_TIME + timedelta(days=1) if mutation == "late" else PRICE_AVAILABLE_AT
            ),
        )
        if mutation == "duplicate_asset":
            _register_mapping_payload(store, payload=payload, suffix="-duplicate")
        elif mutation == "tampered":
            store.resolve(mapping_asset.relative_path).write_bytes(b"tampered")
    if mutation == "listing_cik":
        ordered[0].security.company.cik = "0000000001"
    elif mutation == "listing_ticker":
        ordered[0].ticker = f"{ordered[0].ticker}X"
    elif mutation == "provider_symbol":
        ordered[0].provider_symbol = f"{ordered[0].ticker}.ALT"
    elif mutation == "listing_mic":
        ordered[0].exchange_mic = "XNYS"

    with pytest.raises(ValueError, match="SEC mapping authority"):
        _REAL_V4_MAPPING_RESOLVER(
            listings=ordered,
            config=config,
            cik_config=cik_config,
            asof=AsOfData(DECISION_TIME, store),
        )

    # The failed authority gate precedes price, SEC-fact, and peer reads.
    assert prices
    assert assets


@pytest.mark.django_db
def test_v4_sec_mapping_accepts_exact_nasdaq_and_nyse_rows(tmp_path: Path) -> None:
    store = AssetStore(tmp_path)
    listings: list[Listing] = []
    prices: dict[str, float] = {}
    assets: dict[str, DataAsset] = {}
    for index, (ticker, mic) in enumerate(
        (("MAPT", "XNAS"), ("MAPP1", "XNYS"), ("MAPP2", "XNAS"), ("MAPP3", "XNAS"))
    ):
        listing = _listing(ticker, exchange_mic=mic)
        price, asset = _company_evidence(
            store,
            listing,
            sic="3571",
            scale=1.0 + index * 0.1,
        )
        listings.append(listing)
        prices[str(listing.pk)] = price
        assets[str(listing.pk)] = asset
    config = load_long_forecast_v4_config(V4_PATH)
    synthetic_cik = long_v4_module.load_long_v4_sec_cik_config(config)
    synthetic_config_path = tmp_path / "synthetic-sec-cik.yml"
    synthetic_config_path.write_text(
        yaml.safe_dump(synthetic_cik.raw, sort_keys=False),
        encoding="utf-8",
    )
    physically_loaded = load_sec_cik_config(synthetic_config_path)
    assert physically_loaded.config_hash == synthetic_cik.config_hash
    assert physically_loaded.source_sha256 == synthetic_cik.source_sha256

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=config,
    )[str(listings[0].pk)]
    rows = pair["3y"].calculation["evidence_catalog"]["sec_mapping_authority"]["cohort_rows"]
    assert {row["config_exchange"]: row["listing_exchange_mic"] for row in rows} == {
        "Nasdaq": "XNAS",
        "NYSE": "XNYS",
    }


def test_v3_audit_loader_rejects_v4_schema() -> None:
    with pytest.raises(CommandError, match="schema_version must be 1"):
        call_command(
            "audit_long_evidence",
            "--symbols",
            "SYNTHETIC",
            "--target-date",
            TARGET_DATE.isoformat(),
            "--available-through",
            DECISION_TIME.isoformat(),
            "--decision-time",
            DECISION_TIME.isoformat(),
            "--long-config",
            str(V4_PATH),
            stdout=StringIO(),
        )


@pytest.mark.django_db
def test_shared_sec_derivations_are_pure_and_match_v4_catalog(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listing = _listing("SHAREDSEC")
    _price, _price_asset = _company_evidence(store, listing, sic="3571")
    fact = (
        FundamentalFact.objects.filter(company=listing.security.company)
        .select_related("source_asset")
        .order_by("period_end", "concept")
        .first()
    )
    assert fact is not None
    source = fact.source_asset
    context = DataAsset.objects.get(pk=source.metadata["submissions_asset_id"])
    filing = (
        FundamentalFactEvidence.objects.select_related("source_asset")
        .get(
            fact=fact,
            role=FundamentalFactEvidence.Role.FILING,
        )
        .source_asset
    )
    before_counts = (
        DataAsset.objects.count(),
        FundamentalFact.objects.count(),
        FundamentalFactEvidence.objects.count(),
        CompanyClassificationObservation.objects.count(),
    )
    before_files = sorted(
        path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()
    )

    with CaptureQueriesContext(connection) as queries:
        current = derive_sec_current_submissions(
            store.read_bytes(context.relative_path),
            source_asset=context,
            expected_cik=listing.security.company.cik,
        )
        history_records = derive_sec_historical_submissions(
            store.read_bytes(filing.relative_path),
            source_asset=filing,
            expected_cik=listing.security.company.cik,
            filename=cast(str, filing.metadata["filename"]),
            allowed_filenames=current.historical_filenames,
        )
        derivations = derive_sec_companyfacts(
            store.read_bytes(source.relative_path),
            source_asset=source,
            expected_cik=listing.security.company.cik,
            filing_records=(*current.filings, *history_records),
            config=load_sec_fundamentals_config(),
        )

    assert len(queries) == 0
    matching = [
        item
        for item in derivations
        if item.observation_hash == fact.observation_hash
        and item.filing_source_asset_id == filing.pk
    ]
    assert len(matching) == 1
    derived = matching[0]
    assert (
        derived.concept,
        derived.taxonomy,
        derived.source_concept,
        derived.value,
        derived.unit,
        derived.period_identity,
        derived.accession,
    ) == (
        fact.concept,
        fact.taxonomy,
        fact.source_concept,
        fact.value,
        fact.unit,
        fact.period_identity,
        fact.accession,
    )
    assert before_counts == (
        DataAsset.objects.count(),
        FundamentalFact.objects.count(),
        FundamentalFactEvidence.objects.count(),
        CompanyClassificationObservation.objects.count(),
    )
    assert before_files == sorted(
        path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()
    )


@pytest.mark.django_db
def test_shared_sec_derivation_uses_conservative_filed_next_day_fallback(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    cik = "0000123456"
    submitted = datetime(2026, 2, 20, tzinfo=UTC)
    submissions = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject=cik,
        available_at=submitted,
        payload=json.dumps(
            {
                "cik": 123456,
                "sic": "3571",
                "sicDescription": "Synthetic",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000123456-26-000001"],
                        "filingDate": ["2026-02-15"],
                        "acceptanceDateTime": [""],
                        "form": ["10-Q"],
                        "reportDate": ["2025-12-31"],
                        "primaryDocument": ["fallback.htm"],
                    },
                    "files": [],
                },
            },
            sort_keys=True,
        ).encode(),
    )
    companyfacts = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=cik,
        available_at=submitted,
        payload=json.dumps(
            {
                "cik": 123456,
                "facts": {
                    "us-gaap": {
                        "NetIncomeLoss": {
                            "units": {
                                "USD": [
                                    {
                                        "start": "2025-10-01",
                                        "end": "2025-12-31",
                                        "val": "10",
                                        "accn": "0000123456-26-000001",
                                        "fy": 2025,
                                        "fp": "Q4",
                                        "form": "10-Q",
                                        "filed": "2026-02-15",
                                    }
                                ]
                            }
                        }
                    }
                },
            },
            sort_keys=True,
        ).encode(),
    )

    current = derive_sec_current_submissions(
        store.read_bytes(submissions.relative_path),
        source_asset=submissions,
        expected_cik=cik,
    )
    derived = derive_sec_companyfacts(
        store.read_bytes(companyfacts.relative_path),
        source_asset=companyfacts,
        expected_cik=cik,
        filing_records=current.filings,
        config=load_sec_fundamentals_config(),
    )

    assert len(derived) == 1
    assert derived[0].filing_availability_basis == "filed_date_next_day"
    assert derived[0].acceptance_at.isoformat() == "2026-02-16T00:00:00-05:00"
    assert "availability_filed_date_fallback" in derived[0].base_quality_flags


def _strict_filing_columns(
    *,
    acceptance: object = "2026-02-15T16:30:00-05:00",
) -> dict[str, list[object]]:
    return {
        "accessionNumber": ["0000123456-26-000001"],
        "filingDate": ["2026-02-15"],
        "form": ["10-K"],
        "reportDate": ["2025-12-31"],
        "primaryDocument": ["annual.htm"],
        "acceptanceDateTime": [acceptance],
    }


def _strict_current_payload(columns: dict[str, list[object]]) -> bytes:
    return json.dumps(
        {
            "cik": 123456,
            "sic": "3571",
            "sicDescription": "Synthetic",
            "filings": {"recent": columns, "files": []},
        },
        sort_keys=True,
    ).encode()


@pytest.mark.django_db
@pytest.mark.parametrize("shape", ["missing_recent", "nonobject_recent"])
def test_sec_current_submissions_requires_a_tabular_recent_object(
    tmp_path: Path,
    shape: str,
) -> None:
    store = AssetStore(tmp_path)
    document: dict[str, object] = {
        "cik": 123456,
        "filings": {"files": []},
    }
    if shape == "nonobject_recent":
        cast(dict[str, object], document["filings"])["recent"] = []
    asset = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=json.dumps(document).encode(),
    )
    with pytest.raises(SecDerivationError, match=r"filings\.recent"):
        derive_sec_current_submissions(
            store.read_bytes(asset.relative_path),
            source_asset=asset,
            expected_cik="0000123456",
        )


@pytest.mark.django_db
@pytest.mark.parametrize("source_kind", ["current", "history"])
@pytest.mark.parametrize(
    "mutation",
    ["missing_required", "misaligned_required", "misaligned_optional"],
)
def test_sec_submission_tables_require_exact_column_lengths(
    tmp_path: Path,
    source_kind: str,
    mutation: str,
) -> None:
    store = AssetStore(tmp_path)
    columns = _strict_filing_columns()
    if mutation == "missing_required":
        columns.pop("primaryDocument")
    elif mutation == "misaligned_required":
        columns["reportDate"].append("2024-12-31")
    else:
        columns["acceptanceDateTime"].append(None)
    cik = "0000123456"
    filename = f"CIK{cik}-submissions-001.json"
    if source_kind == "current":
        payload = _strict_current_payload(columns)
        asset = _asset(
            store,
            provider="sec",
            kind="sec_submissions",
            subject=cik,
            available_at=DECISION_TIME,
            payload=payload,
        )
        call = lambda: derive_sec_current_submissions(  # noqa: E731
            payload,
            source_asset=asset,
            expected_cik=cik,
        )
    else:
        payload = json.dumps(columns, sort_keys=True).encode()
        asset = _asset(
            store,
            provider="sec",
            kind="sec_submissions_history",
            subject=cik,
            available_at=DECISION_TIME,
            payload=payload,
            metadata={"filename": filename},
        )
        call = lambda: derive_sec_historical_submissions(  # noqa: E731
            payload,
            source_asset=asset,
            expected_cik=cik,
            filename=filename,
            allowed_filenames=(filename,),
        )
    with pytest.raises(SecDerivationError):
        call()


@pytest.mark.django_db
def test_sec_submission_tables_accept_only_structurally_complete_empty_tables(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    empty = {field: [] for field in _strict_filing_columns() if field != "acceptanceDateTime"}
    current_payload = _strict_current_payload(empty)
    current_asset = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=current_payload,
    )
    assert (
        derive_sec_current_submissions(
            current_payload,
            source_asset=current_asset,
            expected_cik="0000123456",
        ).filings
        == ()
    )
    filename = "CIK0000123456-submissions-001.json"
    empty["acceptanceDateTime"] = []
    history_payload = json.dumps(empty, sort_keys=True).encode()
    history_asset = _asset(
        store,
        provider="sec",
        kind="sec_submissions_history",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=history_payload,
        metadata={"filename": filename},
    )
    assert (
        derive_sec_historical_submissions(
            history_payload,
            source_asset=history_asset,
            expected_cik="0000123456",
            filename=filename,
            allowed_filenames=(filename,),
        )
        == ()
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("acceptance", "include_column", "expected_at", "expected_basis"),
    [
        (None, False, "2026-02-16T00:00:00-05:00", "filed_date_next_day"),
        ("", True, "2026-02-16T00:00:00-05:00", "filed_date_next_day"),
        (None, True, "2026-02-16T00:00:00-05:00", "filed_date_next_day"),
        ("2026-02-15", True, "2026-02-16T00:00:00-05:00", "filed_date_next_day"),
        ("2026-02-15T16:30:00", True, "2026-02-15T16:30:00-05:00", "acceptance_datetime"),
        (
            "2026-02-15T21:30:00+00:00",
            True,
            "2026-02-15T21:30:00+00:00",
            "acceptance_datetime",
        ),
    ],
)
def test_sec_acceptance_precision_and_conservative_fallback(
    tmp_path: Path,
    acceptance: object,
    include_column: bool,
    expected_at: str,
    expected_basis: str,
) -> None:
    store = AssetStore(tmp_path)
    columns = _strict_filing_columns(acceptance=acceptance)
    if not include_column:
        columns.pop("acceptanceDateTime")
    payload = _strict_current_payload(columns)
    asset = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=payload,
    )
    record = derive_sec_current_submissions(
        payload,
        source_asset=asset,
        expected_cik="0000123456",
    ).filings[0]
    assert record.acceptance_at.isoformat() == expected_at
    assert record.acceptance_basis == expected_basis


@pytest.mark.django_db
def test_sec_date_only_acceptance_must_equal_filing_date(tmp_path: Path) -> None:
    store = AssetStore(tmp_path)
    payload = _strict_current_payload(_strict_filing_columns(acceptance="2026-02-14"))
    asset = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=payload,
    )
    with pytest.raises(SecDerivationError, match="does not equal filingDate"):
        derive_sec_current_submissions(
            payload,
            source_asset=asset,
            expected_cik="0000123456",
        )


@pytest.mark.django_db
@pytest.mark.parametrize("source_kind", ["current", "history"])
@pytest.mark.parametrize(
    "acceptance",
    [
        "2026-02-14T23:59:59",
        "2026-02-16T00:00:00-05:00",
    ],
)
def test_sec_exact_acceptance_new_york_date_must_equal_filing_date(
    tmp_path: Path,
    source_kind: str,
    acceptance: str,
) -> None:
    store = AssetStore(tmp_path)
    cik = "0000123456"
    columns = _strict_filing_columns(acceptance=acceptance)
    filename = f"CIK{cik}-submissions-001.json"
    if source_kind == "current":
        payload = _strict_current_payload(columns)
        asset = _asset(
            store,
            provider="sec",
            kind="sec_submissions",
            subject=cik,
            available_at=DECISION_TIME,
            payload=payload,
        )
        derive = lambda: derive_sec_current_submissions(  # noqa: E731
            payload,
            source_asset=asset,
            expected_cik=cik,
        )
    else:
        payload = json.dumps(columns, sort_keys=True).encode()
        asset = _asset(
            store,
            provider="sec",
            kind="sec_submissions_history",
            subject=cik,
            available_at=DECISION_TIME,
            payload=payload,
            metadata={"filename": filename},
        )
        derive = lambda: derive_sec_historical_submissions(  # noqa: E731
            payload,
            source_asset=asset,
            expected_cik=cik,
            filename=filename,
            allowed_filenames=(filename,),
        )
    with pytest.raises(SecDerivationError, match="New York date does not equal filingDate"):
        derive()


@pytest.mark.django_db
@pytest.mark.parametrize("source_kind", ["current", "history"])
def test_sec_utc_acceptance_uses_its_new_york_calendar_date(
    tmp_path: Path,
    source_kind: str,
) -> None:
    store = AssetStore(tmp_path)
    cik = "0000123456"
    acceptance = "2026-02-16T04:30:00Z"
    columns = _strict_filing_columns(acceptance=acceptance)
    filename = f"CIK{cik}-submissions-001.json"
    if source_kind == "current":
        payload = _strict_current_payload(columns)
        asset = _asset(
            store,
            provider="sec",
            kind="sec_submissions",
            subject=cik,
            available_at=DECISION_TIME,
            payload=payload,
        )
        records = derive_sec_current_submissions(
            payload,
            source_asset=asset,
            expected_cik=cik,
        ).filings
    else:
        payload = json.dumps(columns, sort_keys=True).encode()
        asset = _asset(
            store,
            provider="sec",
            kind="sec_submissions_history",
            subject=cik,
            available_at=DECISION_TIME,
            payload=payload,
            metadata={"filename": filename},
        )
        records = derive_sec_historical_submissions(
            payload,
            source_asset=asset,
            expected_cik=cik,
            filename=filename,
            allowed_filenames=(filename,),
        )
    assert records[0].acceptance_at.isoformat() == "2026-02-16T04:30:00+00:00"
    assert records[0].acceptance_basis == "acceptance_datetime"


@pytest.mark.django_db
def test_sec_duplicate_accession_enriches_one_missing_acceptance_deterministically(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    columns = _strict_filing_columns(acceptance="")
    for field, values in tuple(columns.items()):
        values.append("2026-02-15T21:30:00+00:00" if field == "acceptanceDateTime" else values[0])
    payload = _strict_current_payload(columns)
    asset = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=payload,
    )
    records = derive_sec_current_submissions(
        payload,
        source_asset=asset,
        expected_cik="0000123456",
    ).filings
    assert len(records) == 1
    assert records[0].acceptance_basis == "acceptance_datetime"
    assert records[0].acceptance_at.isoformat() == "2026-02-15T21:30:00+00:00"


@pytest.mark.django_db
def test_sec_history_exact_acceptance_enriches_current_fallback(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    cik = "0000123456"
    current_payload = _strict_current_payload(_strict_filing_columns(acceptance=None))
    current_asset = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject=cik,
        available_at=DECISION_TIME,
        payload=current_payload,
    )
    filename = f"CIK{cik}-submissions-001.json"
    history_columns = _strict_filing_columns(acceptance="2026-02-15T21:30:00+00:00")
    history_payload = json.dumps(history_columns, sort_keys=True).encode()
    history_asset = _asset(
        store,
        provider="sec",
        kind="sec_submissions_history",
        subject=cik,
        available_at=DECISION_TIME,
        payload=history_payload,
        metadata={"filename": filename},
    )
    companyfacts_payload = json.dumps(
        {
            "cik": 123456,
            "facts": {
                "us-gaap": {
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2025-01-01",
                                    "end": "2025-12-31",
                                    "val": "10",
                                    "accn": "0000123456-26-000001",
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2026-02-15",
                                }
                            ]
                        }
                    }
                }
            },
        },
        sort_keys=True,
    ).encode()
    companyfacts = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=cik,
        available_at=DECISION_TIME,
        payload=companyfacts_payload,
    )
    current = derive_sec_current_submissions(
        current_payload,
        source_asset=current_asset,
        expected_cik=cik,
    )
    history = derive_sec_historical_submissions(
        history_payload,
        source_asset=history_asset,
        expected_cik=cik,
        filename=filename,
        allowed_filenames=(filename,),
    )
    derivation = derive_sec_companyfacts(
        companyfacts_payload,
        source_asset=companyfacts,
        expected_cik=cik,
        filing_records=(*current.filings, *history),
        config=load_sec_fundamentals_config(),
    )[0]
    assert derivation.acceptance_at.isoformat() == "2026-02-15T21:30:00+00:00"
    assert derivation.filing_availability_basis == "acceptance_datetime"
    assert derivation.filing_source_asset_id == history_asset.pk


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("field", "conflict"),
    [
        ("filingDate", "2026-02-16"),
        ("form", "10-K/A"),
        ("reportDate", "2025-12-30"),
        ("primaryDocument", "other.htm"),
        ("acceptanceDateTime", "2026-02-15T21:31:00+00:00"),
    ],
)
def test_sec_duplicate_accession_rejects_conflicting_nonblank_metadata(
    tmp_path: Path,
    field: str,
    conflict: object,
) -> None:
    store = AssetStore(tmp_path)
    columns = _strict_filing_columns()
    for name, values in tuple(columns.items()):
        values.append(conflict if name == field else values[0])
    if field == "filingDate":
        columns["acceptanceDateTime"][-1] = "2026-02-16T21:30:00+00:00"
    payload = _strict_current_payload(columns)
    asset = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=payload,
    )
    with pytest.raises(SecDerivationError, match="conflict"):
        derive_sec_current_submissions(
            payload,
            source_asset=asset,
            expected_cik="0000123456",
        )


@pytest.mark.django_db
def test_sec_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    store = AssetStore(tmp_path)
    payload = b'{"cik":123456,"cik":123456,"filings":{"recent":{},"files":[]}}'
    asset = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=payload,
    )
    with pytest.raises(SecDerivationError, match="Duplicate|duplicate"):
        derive_sec_current_submissions(
            payload,
            source_asset=asset,
            expected_cik="0000123456",
        )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("form", "10-Q"),
        ("filed", "2026-02-14"),
        ("end", "2026-01-01"),
    ],
)
def test_companyfacts_conflicts_with_submissions_are_unprovable(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    store = AssetStore(tmp_path)
    columns = _strict_filing_columns()
    submissions_payload = _strict_current_payload(columns)
    submissions = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=submissions_payload,
    )
    observation = {
        "start": "2025-01-01",
        "end": "2025-12-31",
        "val": "10",
        "accn": "0000123456-26-000001",
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "filed": "2026-02-15",
    }
    observation[field] = value
    companyfacts_payload = json.dumps(
        {
            "cik": 123456,
            "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [observation]}}}},
        },
        sort_keys=True,
    ).encode()
    companyfacts = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=companyfacts_payload,
    )
    records = derive_sec_current_submissions(
        submissions_payload,
        source_asset=submissions,
        expected_cik="0000123456",
    ).filings
    with pytest.raises(SecDerivationError, match="conflicts|after submissions"):
        derive_sec_companyfacts(
            companyfacts_payload,
            source_asset=companyfacts,
            expected_cik="0000123456",
            filing_records=records,
            config=load_sec_fundamentals_config(),
        )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "mutation",
    [
        "blank_accession",
        "missing_end",
        "missing_start",
        "bad_value",
        "bad_fiscal_year",
        "bad_fiscal_period",
        "bad_frame",
        "nonobject",
        "bad_units",
    ],
)
def test_malformed_configured_companyfacts_never_looks_absent(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = AssetStore(tmp_path)
    submissions_payload = _strict_current_payload(_strict_filing_columns())
    submissions = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=submissions_payload,
    )
    observation: object = {
        "start": "2025-01-01",
        "end": "2025-12-31",
        "val": "10",
        "accn": "0000123456-26-000001",
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "filed": "2026-02-15",
    }
    if mutation == "nonobject":
        observation = "malformed"
    else:
        assert isinstance(observation, dict)
        if mutation == "blank_accession":
            observation["accn"] = ""
        elif mutation == "missing_end":
            observation.pop("end")
        elif mutation == "missing_start":
            observation.pop("start")
        elif mutation == "bad_value":
            observation["val"] = "NaN"
        elif mutation == "bad_fiscal_year":
            observation["fy"] = "not-a-year"
        elif mutation == "bad_fiscal_period":
            observation["fp"] = []
        elif mutation == "bad_frame":
            observation["frame"] = {}
    units: object = {"USD": [observation]}
    if mutation == "bad_units":
        units = []
    companyfacts_payload = json.dumps(
        {
            "cik": 123456,
            "facts": {"us-gaap": {"NetIncomeLoss": {"units": units}}},
        },
        sort_keys=True,
    ).encode()
    companyfacts = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=companyfacts_payload,
    )
    records = derive_sec_current_submissions(
        submissions_payload,
        source_asset=submissions,
        expected_cik="0000123456",
    ).filings
    with pytest.raises(SecDerivationError):
        derive_sec_companyfacts(
            companyfacts_payload,
            source_asset=companyfacts,
            expected_cik="0000123456",
            filing_records=records,
            config=load_sec_fundamentals_config(),
        )


@pytest.mark.django_db
@pytest.mark.parametrize(
    (
        "raw_form",
        "submissions_form",
        "include_filing",
        "expected_status",
        "expected_issue",
        "expected_effective_form",
        "expected_relevant",
    ),
    [
        (
            "8-K",
            "10-K",
            True,
            "rejected",
            "configured_fields_malformed",
            "10-K",
            True,
        ),
        (
            "8-K",
            "8-K",
            True,
            "excluded",
            "filing_form_not_allowed",
            "8-K",
            False,
        ),
        (" 10-k ", "10-K", True, "derived", None, "10-K", True),
        (None, "10-K", True, "derived", None, "10-K", True),
        (
            "8-K",
            "10-K",
            False,
            "rejected",
            "filing_form_not_allowed",
            "8-K",
            True,
        ),
        (
            None,
            "10-K",
            False,
            "rejected",
            "configured_fields_malformed",
            None,
            True,
        ),
    ],
)
def test_configured_observation_preserves_raw_form_and_trusts_reconciled_form(
    tmp_path: Path,
    raw_form: str | None,
    submissions_form: str,
    include_filing: bool,
    expected_status: str,
    expected_issue: str | None,
    expected_effective_form: str | None,
    expected_relevant: bool,
) -> None:
    store = AssetStore(tmp_path)
    columns = _strict_filing_columns()
    columns["form"] = [submissions_form]
    submissions_payload = _strict_current_payload(columns)
    submissions = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=submissions_payload,
    )
    observation: dict[str, object] = {
        "start": "2025-01-01",
        "end": "2025-12-31",
        "val": "10",
        "accn": "0000123456-26-000001",
        "fy": 2025,
        "fp": "FY",
        "filed": "2026-02-15",
    }
    if raw_form is not None:
        observation["form"] = raw_form
    companyfacts_payload = json.dumps(
        {
            "cik": 123456,
            "facts": {
                "us-gaap": {
                    SOURCE_CONCEPTS["operating_cash_flow"].split(":", 1)[1]: {
                        "units": {"USD": [observation]}
                    }
                }
            },
        },
        sort_keys=True,
    ).encode()
    companyfacts = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=companyfacts_payload,
    )
    filings = derive_sec_current_submissions(
        submissions_payload,
        source_asset=submissions,
        expected_cik="0000123456",
    ).filings

    inspection = inspect_sec_companyfacts(
        companyfacts_payload,
        source_asset=companyfacts,
        expected_cik="0000123456",
        filing_records=filings if include_filing else (),
        config=load_sec_fundamentals_config(),
    )

    assert len(inspection.configured_observations) == 1
    configured = inspection.configured_observations[0]
    assert configured.raw_filing_form == raw_form
    assert configured.filing_form == expected_effective_form
    assert configured.status == expected_status
    assert configured.rejection_code == expected_issue
    assert (
        long_v4_module._raw_fcf_observation_is_relevant(
            configured,
            window_start=date(2020, 1, 1),
            target_date=TARGET_DATE,
            data_cutoff=DECISION_TIME,
        )
        is expected_relevant
    )


@pytest.mark.django_db
def test_sec_optional_provider_blanks_and_unknown_columns_are_preserved_as_missing(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    columns = _strict_filing_columns(acceptance=None)
    columns["reportDate"] = [""]
    columns["primaryDocument"] = [None]
    columns["providerExtension"] = [["ignored"]]
    payload = _strict_current_payload(columns)
    asset = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject="0000123456",
        available_at=DECISION_TIME,
        payload=payload,
    )
    record = derive_sec_current_submissions(
        payload,
        source_asset=asset,
        expected_cik="0000123456",
    ).filings[0]
    assert record.report_date is None
    assert record.primary_document == ""
    assert record.acceptance_basis == "filed_date_next_day"


def test_sec_exchange_to_mic_rule_is_exact() -> None:
    verify_sec_exchange_mic(exchange="Nasdaq", mic="XNAS")
    verify_sec_exchange_mic(exchange="NYSE", mic="XNYS")
    for exchange, mic in (("NASDAQ", "XNAS"), ("Nasdaq", "XNYS"), ("NYSE", "XNAS")):
        with pytest.raises(SecDerivationError, match="exchange"):
            verify_sec_exchange_mic(exchange=exchange, mic=mic)


@pytest.mark.django_db
def test_v4_one_path_math_evidence_and_accounting_contract(tmp_path: Path) -> None:
    store = AssetStore(tmp_path)
    listings, prices, assets = _cohort(store)

    forecasts = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )

    target_pair = forecasts[str(listings[0].pk)]
    validate_long_v4_forecast_pair(
        target_pair,
        config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH,
    )
    three = target_pair["3y"]
    five = target_pair["5y"]
    assert three.scenario.base is not None, three.scenario.insufficiency_reason
    assert five.scenario.base is not None, five.scenario.insufficiency_reason
    assert three.scenario.bear <= three.scenario.base <= three.scenario.bull
    assert five.scenario.bear <= five.scenario.base <= five.scenario.bull
    assert three.scenario.confidence == five.scenario.confidence == 0
    assert three.scenario.confidence_status == "not_estimated_uncalibrated"
    assert three.scenario.insufficiency_reason == PROBABILITY_REASON
    assert three.calculation["schema_version"] == 2
    for forecast in target_pair.values():
        assert forecast.calculation["config_hash"] == LONG_V4_EFFECTIVE_CONFIG_HASH
        assert forecast.calculation["fundamentals_config_file_sha256"] == (
            "829ed267eec62304804c9ef71f1816636389c2b0ad77423d8a534a00a5e1ae30"
        )
        assert forecast.calculation["fundamentals_config_hash"] == (
            "7822a1faaae1c8028d71851337a7dbb7a65d9aeaa0f44478e3814a6468b62604"
        )
    assert three.calculation["research_status"] == "research_only_unactivated"
    assert three.calculation["probability_semantics"]["value"] is None
    assert three.calculation["confidence_semantics"]["schema"] == ("zero_is_unavailable_sentinel")
    assert three.calculation["accounting_scope"] == {
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
    assert three.calculation["scenario_paths"] == five.calculation["scenario_paths"]
    assert three.calculation["evidence_catalog"] == five.calculation["evidence_catalog"]
    for scenario in ("bear", "base", "bull"):
        path = three.calculation["scenario_paths"][scenario]
        assert len(path["years"]) == 5
        assert (
            three.calculation["selected_view"]["cumulative_returns"][scenario]
            == path["years"][2]["cumulative_price_return"]
        )
        assert (
            five.calculation["selected_view"]["cumulative_returns"][scenario]
            == path["years"][4]["cumulative_price_return"]
        )
        for year in path["years"]:
            assert year["return_from_level_identity"] == pytest.approx(
                year["return_from_factor_identity"], abs=1e-10
            )
    inputs = three.calculation["formula_inputs"]
    raw_growth = inputs["entity_growth_raw"]
    assert inputs["target_entity_growth"] == pytest.approx(
        sorted(inputs["entity_growth_capped"])[1]
    )
    assert raw_growth == pytest.approx(
        [
            78 / 70 - 1,
            86 / 78 - 1,
            94 / 86 - 1,
        ]
    )
    assert len(three.calculation["peer_set"]) == 3
    assert all(item["selected_peer"] for item in three.calculation["locked_peer_candidates"])
    catalog = three.calculation["evidence_catalog"]
    assert catalog["schema_version"] == 2
    mapping_authority = catalog["sec_mapping_authority"]
    assert mapping_authority["cohort_rows"] == [
        next(row for row in mapping_authority["cohort_rows"] if row["listing_id"] == listing_id)
        for listing_id in catalog["cohort_listing_ids"]
    ]
    assert mapping_authority["exchange_to_mic_rule"] == [
        {"exchange": "Nasdaq", "mic": "XNAS"},
        {"exchange": "NYSE", "mic": "XNYS"},
    ]
    assert (
        three.calculation["source_manifest"][0]["id"] == (mapping_authority["mapping_asset"]["id"])
    )
    assert [item["owner_listing_id"] for item in catalog["raw_fcf_authority"]] == (
        catalog["cohort_listing_ids"]
    )
    assert all(item["status"] == "present_complete" for item in catalog["raw_fcf_authority"])
    assert len(catalog["facts"]) == len({item["id"] for item in catalog["facts"]})
    assert set(catalog["selected_fact_ids"]).isdisjoint(catalog["assessed_fact_ids"])
    assert set(catalog["selected_fact_ids"]) | set(catalog["assessed_fact_ids"]) == {
        item["id"] for item in catalog["facts"]
    }
    assert {item["id"] for item in three.calculation["source_manifest"]} == {
        str(asset.pk) for asset in three.source_assets
    }
    traversed_ids: list[str] = [catalog["sec_mapping_authority"]["mapping_asset"]["id"]]
    for authority in catalog["raw_fcf_authority"]:
        for source in authority["sources"]:
            traversed_ids.extend(
                (
                    source["companyfacts_asset"]["id"],
                    source["current_submissions_asset"]["id"],
                    *(item["id"] for item in source["history_assets"]),
                )
            )
    for fact in catalog["facts"]:
        assert fact["correction_observation_event"] is None
        assert fact["submissions_context_asset_id"]
        traversed_ids.extend(
            (
                fact["source_asset_id"],
                fact["filing_evidence_asset_id"],
                fact["submissions_context_asset_id"],
            )
        )
    traversed_ids.extend(item["source_asset_id"] for item in catalog["classifications"])
    for price_entry in catalog["prices"]:
        traversed_ids.extend(
            (
                price_entry["normalized_asset"]["id"],
                price_entry["raw_asset"]["id"],
            )
        )
    assert [item["id"] for item in three.calculation["source_manifest"]] == list(
        dict.fromkeys(traversed_ids)
    )
    for price_entry in catalog["prices"]:
        normalized = DataAsset.objects.get(pk=price_entry["normalized_asset"]["id"])
        assert price_entry["normalized_asset_id"] == price_entry["normalized_asset"]["id"]
        assert price_entry["normalized_asset_sha256"] == price_entry["normalized_asset"]["sha256"]
        assert price_entry["raw_asset_id"] == price_entry["raw_asset"]["id"]
        assert price_entry["raw_asset_sha256"] == price_entry["raw_asset"]["sha256"]
        assert normalized.metadata["raw_asset_id"] == price_entry["raw_asset"]["id"]
        assert normalized.metadata["raw_sha256"] == price_entry["raw_asset"]["sha256"]
        assert store.read_bytes(normalized.relative_path)
        assert store.read_bytes(
            DataAsset.objects.get(pk=price_entry["raw_asset"]["id"]).relative_path
        )
    assert three.calculation["split_basis"]["assessment_status"] == "unverified"
    assert three.calculation["split_basis"]["assessed_through"] == "2025-12-31"
    assert "verified_through" not in three.calculation["split_basis"]


@pytest.mark.django_db
def test_v4_buybacks_receive_no_credit_and_bear_dilution_caps_last(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    buyback_shares = (130.0, 120.0, 110.0, 100.0)
    target = _listing("BUYBACK")
    price, asset = _company_evidence(
        store,
        target,
        sic="3571",
        scale=1.0,
        annual_shares=buyback_shares,
    )
    peers: list[Listing] = []
    prices = {str(target.pk): price}
    assets = {str(target.pk): asset}
    for index in range(3):
        peer = _listing(f"BP{index}")
        peer_price, peer_asset = _company_evidence(
            store,
            peer,
            sic="3571",
            scale=1.1 + index * 0.1,
        )
        peers.append(peer)
        prices[str(peer.pk)] = peer_price
        assets[str(peer.pk)] = peer_asset
    pair = build_long_forecasts_v4(
        listings=[target, *peers],
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]
    assert pair["3y"].calculation["formula_inputs"]["dilution_base"] == 0
    assert all(
        path["dilution_rate"] == 0 for path in pair["3y"].calculation["scenario_paths"].values()
    )

    capped = deepcopy(pair)
    # The scenario calculator is proven indirectly from the persisted path:
    # a base exactly at the 15% policy ceiling remains capped after the
    # bear multiplier rather than calculating or persisting 18.75%.
    metric_inputs = capped["3y"].calculation["formula_inputs"]
    assert max(metric_inputs["diluted_share_changes_raw"]) < 0

    capped_target = _listing("DILCAP")
    capped_price, capped_asset = _company_evidence(
        store,
        capped_target,
        sic="3671",
        annual_shares=(100.0, 115.0, 132.25, 152.0875),
    )
    capped_listings = [capped_target]
    capped_prices = {str(capped_target.pk): capped_price}
    capped_assets = {str(capped_target.pk): capped_asset}
    for index in range(3):
        peer = _listing(f"DC{index}")
        peer_price, peer_asset = _company_evidence(
            store,
            peer,
            sic="3671",
            scale=1.1 + index * 0.1,
            annual_shares=(100.0, 115.0, 132.25, 152.0875),
        )
        capped_listings.append(peer)
        capped_prices[str(peer.pk)] = peer_price
        capped_assets[str(peer.pk)] = peer_asset
    capped_pair = build_long_forecasts_v4(
        listings=capped_listings,
        current_prices=capped_prices,
        price_assets=capped_assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(capped_target.pk)]
    assert capped_pair["3y"].scenario.base is not None
    capped_paths = capped_pair["3y"].calculation["scenario_paths"]
    assert capped_paths["base"]["dilution_base"] == pytest.approx(0.15)
    assert capped_paths["bear"]["dilution_multiplier"] == 1.25
    assert capped_paths["bear"]["dilution_rate"] == pytest.approx(0.15)
    assert capped_paths["bear"]["dilution_rate"] != pytest.approx(0.1875)


@pytest.mark.django_db
def test_v4_present_but_incompatible_fcf_blocks_net_income_fallback(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    target = _listing("FCFBLOCK")
    price, asset = _company_evidence(
        store,
        target,
        sic="3571",
        scale=1.0,
        omit_latest_capex=True,
    )
    listings = [target]
    prices = {str(target.pk): price}
    assets = {str(target.pk): asset}
    for index in range(3):
        peer = _listing(f"FB{index}")
        peer_price, peer_asset = _company_evidence(
            store,
            peer,
            sic="3571",
            scale=1.1 + index * 0.1,
        )
        listings.append(peer)
        prices[str(peer.pk)] = peer_price
        assets[str(peer.pk)] = peer_asset

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]

    assert pair["3y"].scenario.base is None
    assert pair["5y"].scenario.base is None
    assert pair["3y"].calculation["metric_family"] is None
    assert pair["3y"].calculation["insufficiency_code"].startswith("fcf_")
    assert "net-income fallback is blocked" in pair["3y"].scenario.insufficiency_reason
    assert pair["3y"].calculation["scenario_paths"] == {}
    assert pair["3y"].calculation["annualized_returns"] == {}


@pytest.mark.django_db
def test_v4_net_income_fallback_requires_genuinely_absent_fcf(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings: list[Listing] = []
    prices: dict[str, float] = {}
    assets: dict[str, DataAsset] = {}
    for index, ticker in enumerate(("NIT", "NIP1", "NIP2", "NIP3")):
        listing = _listing(ticker)
        price, asset = _company_evidence(
            store,
            listing,
            sic="3571",
            scale=1.0 + index * 0.1,
            include_fcf=False,
        )
        listings.append(listing)
        prices[str(listing.pk)] = price
        assets[str(listing.pk)] = asset
    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(listings[0].pk)]
    assert pair["3y"].scenario.base is not None
    assert pair["3y"].calculation["metric_family"] == "net_income_per_share"
    assert (
        pair["3y"].calculation["evidence_selection"]["target"]["fcf_raw_source_evidence_present"]
        is False
    )
    raw_authority = pair["3y"].calculation["evidence_selection"]["target"]["raw_fcf_authority"]
    assert raw_authority["status"] == "absent"
    assert raw_authority["derivations"] == []


@pytest.mark.django_db
@pytest.mark.parametrize("raw_concept", ["extra_ocf", "extra_capex"])
def test_v4_raw_fcf_one_leg_blocks_net_income_when_normalization_is_interrupted(
    tmp_path: Path,
    raw_concept: str,
) -> None:
    store = AssetStore(tmp_path)
    listings: list[Listing] = []
    prices: dict[str, float] = {}
    assets: dict[str, DataAsset] = {}
    for index, ticker in enumerate(("RAWNI", "RAWNIP1", "RAWNIP2", "RAWNIP3")):
        listing = _listing(ticker)
        price, asset = _company_evidence(
            store,
            listing,
            sic="3571",
            scale=1.0 + index * 0.1,
            include_fcf=False,
        )
        listings.append(listing)
        prices[str(listing.pk)] = price
        assets[str(listing.pk)] = asset
    _append_companyfacts_vintage(store, listings[0], mutation=raw_concept)

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(listings[0].pk)]
    selection = pair["3y"].calculation["evidence_selection"]["target"]
    authority = selection["raw_fcf_authority"]

    assert pair["3y"].scenario.base is None
    assert pair["3y"].calculation["insufficiency_code"] == "fcf_normalization_incomplete"
    assert pair["3y"].scenario.insufficiency_reason == (
        "Cutoff/window-relevant raw FCF evidence exists but its normalized fact "
        "closure is incomplete; net-income fallback is blocked"
    )
    assert selection["fcf_raw_source_evidence_present"] is True
    assert authority["status"] == "present_normalization_incomplete"
    incomplete = [
        item for item in authority["derivations"] if item["status"] == "normalization_incomplete"
    ]
    assert {item["concept"] for item in incomplete} == {
        "capital_expenditure" if raw_concept == "extra_capex" else "operating_cash_flow"
    }
    assert all(item["matching_normalized_fact_ids"] == [] for item in incomplete)
    validate_long_v4_forecast_pair(pair, config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH)


@pytest.mark.django_db
@pytest.mark.parametrize("raw_concept", ["operating_cash_flow", "capital_expenditure"])
def test_v4_rejected_raw_effective_form_conflict_cannot_become_false_absence(
    tmp_path: Path,
    raw_concept: str,
) -> None:
    store = AssetStore(tmp_path)
    listings: list[Listing] = []
    prices: dict[str, float] = {}
    assets: dict[str, DataAsset] = {}
    for index, ticker in enumerate(("FORMCON", "FORMCONP1", "FORMCONP2", "FORMCONP3")):
        listing = _listing(ticker)
        price, asset = _company_evidence(
            store,
            listing,
            sic="3571",
            scale=1.0 + index * 0.1,
            include_fcf=False,
        )
        listings.append(listing)
        prices[str(listing.pk)] = price
        assets[str(listing.pk)] = asset
    _append_companyfacts_vintage(
        store,
        listings[0],
        mutation=f"raw_form_conflict_{raw_concept}",
    )

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(listings[0].pk)]
    authority = pair["3y"].calculation["evidence_selection"]["target"]["raw_fcf_authority"]
    conflicting = [
        item
        for item in authority["derivations"]
        if item["raw_filing_form"] == "8-K"
        and item["filing_form"] in {"10-K", "10-Q"}
        and item["normalization_issue"] == "configured_fields_malformed"
    ]

    assert pair["3y"].scenario.base is None
    assert pair["3y"].calculation["insufficiency_code"] == "fcf_normalization_incomplete"
    assert authority["status"] == "present_normalization_incomplete"
    assert {item["concept"] for item in conflicting} == {raw_concept}
    assert all(item["status"] == "normalization_incomplete" for item in conflicting)


@pytest.mark.django_db
def test_v4_reconciled_unsupported_form_is_the_only_form_based_exclusion(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings: list[Listing] = []
    prices: dict[str, float] = {}
    assets: dict[str, DataAsset] = {}
    for index, ticker in enumerate(("FORMEX", "FORMEXP1", "FORMEXP2", "FORMEXP3")):
        listing = _listing(ticker)
        price, asset = _company_evidence(
            store,
            listing,
            sic="3571",
            scale=1.0 + index * 0.1,
            include_fcf=False,
        )
        listings.append(listing)
        prices[str(listing.pk)] = price
        assets[str(listing.pk)] = asset
    _append_reconciled_unsupported_fcf_vintage(store, listings[0])

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(listings[0].pk)]
    authority = pair["3y"].calculation["evidence_selection"]["target"]["raw_fcf_authority"]

    assert pair["3y"].scenario.base is not None
    assert pair["3y"].calculation["metric_family"] == "net_income_per_share"
    assert authority["status"] == "absent"
    assert authority["derivations"] == []


@pytest.mark.django_db
def test_v4_raw_fcf_normalization_gap_withholds_only_that_entity_pair(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    target = _listing("RAWGAP")
    target_price, target_asset = _company_evidence(
        store,
        target,
        sic="3571",
        include_fcf=False,
    )
    listings = [target]
    for index in range(3):
        peer = _listing(f"RAWGAPP{index}")
        _company_evidence(store, peer, sic="3571", scale=1.1 + index * 0.1)
        listings.append(peer)
    _append_companyfacts_vintage(store, target, mutation="extra_ocf")

    results = _run_v4(_snapshot(listings), store)
    target_result = next(item for item in results if item.analysis.listing_id == target.pk)
    target_predictions = list(
        Prediction.objects.filter(
            analysis=target_result.analysis,
            method_version="us-sec-long-v4",
        )
    )
    assert len(results) == 4
    assert len(target_predictions) == 2
    assert all(
        prediction.insufficiency_reason
        == (
            "Cutoff/window-relevant raw FCF evidence exists but its normalized "
            "fact closure is incomplete; net-income fallback is blocked"
        )
        and prediction.base_return is None
        for prediction in target_predictions
    )


@pytest.mark.django_db
@pytest.mark.parametrize("mutation", ["outside_window", "after_cutoff"])
def test_v4_raw_fcf_outside_the_admitted_window_does_not_block_genuine_absence(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = AssetStore(tmp_path)
    listings: list[Listing] = []
    prices: dict[str, float] = {}
    assets: dict[str, DataAsset] = {}
    for index, ticker in enumerate(("RAWOUT", "RAWOUTP1", "RAWOUTP2", "RAWOUTP3")):
        listing = _listing(ticker)
        price, asset = _company_evidence(
            store,
            listing,
            sic="3571",
            scale=1.0 + index * 0.1,
            include_fcf=False,
        )
        listings.append(listing)
        prices[str(listing.pk)] = price
        assets[str(listing.pk)] = asset
    _append_companyfacts_vintage(store, listings[0], mutation=mutation)

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(listings[0].pk)]
    authority = pair["3y"].calculation["evidence_selection"]["target"]["raw_fcf_authority"]
    assert pair["3y"].scenario.base is not None
    assert pair["3y"].calculation["metric_family"] == "net_income_per_share"
    assert authority["status"] == "absent"
    validate_long_v4_forecast_pair(pair, config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH)


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("mutation", "observed_at", "expected_status", "expected_source_count"),
    [
        ("unchanged", DECISION_TIME - timedelta(minutes=1), "present_complete", 1),
        ("changed_fcf", DECISION_TIME - timedelta(minutes=1), "present_complete", 1),
        (
            "changed_fcf",
            datetime(2026, 2, 26, 12, tzinfo=UTC),
            "present_normalization_incomplete",
            2,
        ),
    ],
)
def test_v4_raw_fcf_correction_uses_its_own_observation_boundary(
    tmp_path: Path,
    mutation: str,
    observed_at: datetime,
    expected_status: str,
    expected_source_count: int,
) -> None:
    store = AssetStore(tmp_path)
    listings, prices, assets = _cohort(store)
    changed_source = _append_companyfacts_vintage(
        store,
        listings[0],
        mutation=mutation,
        available_at=observed_at,
    )

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(listings[0].pk)]
    authority = pair["3y"].calculation["evidence_selection"]["target"]["raw_fcf_authority"]
    assert authority["status"] == expected_status
    assert len(authority["sources"]) == expected_source_count
    source_ids = {item["companyfacts_asset"]["id"] for item in authority["sources"]}
    assert (str(changed_source.pk) in source_ids) is (expected_source_count == 2)
    changed_entries = [
        item
        for item in authority["derivations"]
        if item["companyfacts_asset_id"] == str(changed_source.pk)
        and item["is_same_accession_correction"]
    ]
    if expected_status == "present_complete":
        assert pair["3y"].scenario.base is not None
        assert all(item["matching_normalized_fact_ids"] for item in authority["derivations"])
        assert changed_entries == []
    else:
        assert pair["3y"].scenario.base is None
        assert pair["3y"].calculation["insufficiency_code"] == ("fcf_normalization_incomplete")
        assert len(changed_entries) == 1
        assert changed_entries[0]["status"] == "normalization_incomplete"
        assert changed_entries[0]["raw_observation_available_at"] == observed_at.isoformat()
        assert changed_entries[0]["raw_observation_availability_basis"] == (
            "companyfacts_correction_asset_retrieval"
        )
    validate_long_v4_forecast_pair(pair, config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH)


@pytest.mark.django_db
def test_v4_latest_unsupported_fcf_unit_is_persisted_as_normalization_incomplete(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    unsupported = _append_companyfacts_vintage(
        store,
        listings[0],
        mutation="unsupported_fcf_unit",
        available_at=datetime(2026, 2, 26, 12, tzinfo=UTC),
    )

    results = _run_v4(_snapshot(listings), store)
    target = next(item for item in results if item.analysis.listing_id == listings[0].pk)
    predictions = list(
        Prediction.objects.filter(
            analysis=target.analysis,
            method_version="us-sec-long-v4",
        )
    )
    assert len(predictions) == 2
    assert all(
        prediction.base_return is None
        and prediction.calculation["insufficiency_code"] == "fcf_normalization_incomplete"
        and prediction.evidence_role == Prediction.EvidenceRole.ADVISORY
        for prediction in predictions
    )
    authority = predictions[0].calculation["evidence_selection"]["target"]["raw_fcf_authority"]
    assert authority["status"] == "present_normalization_incomplete"
    rejected = [
        item
        for item in authority["derivations"]
        if item["companyfacts_asset_id"] == str(unsupported.pk)
        and item["normalization_issue"] == "unsupported_unit"
    ]
    assert rejected
    assert {item["unit"] for item in rejected} == {"EUR"}
    assert all(item["matching_normalized_fact_ids"] == [] for item in rejected)


@pytest.mark.django_db
def test_v4_discovers_nonlatest_companyfacts_backing_visible_non_v4_fact(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings: list[Listing] = []
    prices: dict[str, float] = {}
    assets: dict[str, DataAsset] = {}
    for index, ticker in enumerate(("RAWNONV4", "RAWNONV4P1", "RAWNONV4P2", "RAWNONV4P3")):
        listing = _listing(ticker)
        price, asset = _company_evidence(
            store,
            listing,
            sic="3571",
            scale=1.0 + index * 0.1,
            include_fcf=False,
        )
        listings.append(listing)
        prices[str(listing.pk)] = price
        assets[str(listing.pk)] = asset
    hidden_source, latest_source = _append_nonlatest_non_v4_companyfacts_source(
        store,
        listings[0],
    )

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(listings[0].pk)]
    authority = pair["3y"].calculation["evidence_selection"]["target"]["raw_fcf_authority"]

    assert pair["3y"].scenario.base is None
    assert pair["3y"].calculation["insufficiency_code"] == "fcf_normalization_incomplete"
    assert authority["status"] == "present_normalization_incomplete"
    source_ids = [item["companyfacts_asset"]["id"] for item in authority["sources"]]
    assert str(hidden_source.pk) in source_ids
    assert str(latest_source.pk) not in source_ids
    assert str(hidden_source.pk) in {
        item["id"] for item in pair["3y"].calculation["source_manifest"]
    }
    assert any(
        item["companyfacts_asset_id"] == str(hidden_source.pk)
        and item["normalization_issue"] == "unsupported_unit"
        for item in authority["derivations"]
    )
    validate_long_v4_forecast_pair(pair, config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH)


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("mutation", "expected_concept", "expected_issue", "expected_unit"),
    [
        (
            "malformed_ocf",
            "operating_cash_flow",
            "configured_fields_malformed",
            "USD",
        ),
        (
            "unsupported_capex_unit",
            "capital_expenditure",
            "unsupported_unit",
            "EUR",
        ),
    ],
)
def test_v4_preserves_rejected_raw_fcf_before_later_no_fcf_vintage(
    tmp_path: Path,
    mutation: str,
    expected_concept: str,
    expected_issue: str,
    expected_unit: str,
) -> None:
    store = AssetStore(tmp_path)
    listings: list[Listing] = []
    prices: dict[str, float] = {}
    assets: dict[str, DataAsset] = {}
    for index, ticker in enumerate(("RAWREJECT", "RAWREJECTP1", "RAWREJECTP2", "RAWREJECTP3")):
        listing = _listing(ticker)
        price, asset = _company_evidence(
            store,
            listing,
            sic="3571",
            scale=1.0 + index * 0.1,
            include_fcf=False,
        )
        listings.append(listing)
        prices[str(listing.pk)] = price
        assets[str(listing.pk)] = asset
    rejected_source, latest_source = _append_rejected_fcf_then_no_fcf_vintage(
        store,
        listings[0],
        mutation=mutation,
    )

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(listings[0].pk)]
    authority = pair["3y"].calculation["evidence_selection"]["target"]["raw_fcf_authority"]
    rejected = [
        item
        for item in authority["derivations"]
        if item["companyfacts_asset_id"] == str(rejected_source.pk)
    ]
    source_ids = [item["companyfacts_asset"]["id"] for item in authority["sources"]]

    assert pair["3y"].scenario.base is None
    assert pair["3y"].calculation["metric_family"] is None
    assert pair["3y"].calculation["insufficiency_code"] == "fcf_normalization_incomplete"
    assert "net-income fallback is blocked" in pair["3y"].scenario.insufficiency_reason
    assert authority["status"] == "present_normalization_incomplete"
    assert source_ids[-2:] == [str(rejected_source.pk), str(latest_source.pk)]
    assert len(rejected) == 1
    assert rejected[0]["concept"] == expected_concept
    assert rejected[0]["unit"] == expected_unit
    assert rejected[0]["normalization_issue"] == expected_issue
    assert rejected[0]["status"] == "normalization_incomplete"
    assert rejected[0]["companyfacts_asset_sha256"] == rejected_source.sha256
    assert rejected[0]["matching_normalized_fact_ids"] == []
    validate_long_v4_forecast_pair(pair, config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH)


@pytest.mark.django_db
def test_v4_latest_adverse_period_is_not_replaced_by_older_favorable_tail(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    target = _listing("ADVERSE")
    price, asset = _company_evidence(
        store,
        target,
        sic="3571",
        scale=1.0,
        latest_fcf=-10.0,
        add_older_favorable_period=True,
    )
    listings = [target]
    prices = {str(target.pk): price}
    assets = {str(target.pk): asset}
    for index in range(3):
        peer = _listing(f"AP{index}")
        peer_price, peer_asset = _company_evidence(
            store,
            peer,
            sic="3571",
            scale=1.1 + index * 0.1,
        )
        listings.append(peer)
        prices[str(peer.pk)] = peer_price
        assets[str(peer.pk)] = peer_asset
    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]
    assert pair["3y"].scenario.base is None
    assert pair["3y"].calculation["insufficiency_code"] == (
        "fcf_annual_entity_nonpositive_or_nonfinite"
    )
    assert any(
        item["period_end"] == "2025-12-31"
        for item in pair["3y"].calculation["evidence_catalog"]["facts"]
    )


@pytest.mark.django_db
def test_v4_peer_cohort_locks_before_core_assessment_and_does_not_widen(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    target = _listing("LOCK")
    target_price, target_asset = _company_evidence(store, target, sic="3571")
    listings = [target]
    prices = {str(target.pk): target_price}
    assets = {str(target.pk): target_asset}
    # Three SIC-4 identities lock the narrow cohort, but one fails core
    # evidence. Additional valid SIC-3 names must not be used to rescue it.
    for index in range(3):
        peer = _listing(f"L4{index}")
        peer_price, peer_asset = _company_evidence(
            store,
            peer,
            sic="3571",
            omit_latest_capex=index == 0,
            scale=1.1 + index * 0.1,
        )
        listings.append(peer)
        prices[str(peer.pk)] = peer_price
        assets[str(peer.pk)] = peer_asset
    for index in range(3):
        peer = _listing(f"L3{index}")
        peer_price, peer_asset = _company_evidence(
            store,
            peer,
            sic=f"358{index}",
            scale=1.4 + index * 0.1,
        )
        listings.append(peer)
        prices[str(peer.pk)] = peer_price
        assets[str(peer.pk)] = peer_asset

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]

    assert pair["3y"].scenario.base is None
    assert pair["3y"].calculation["insufficiency_code"] == ("locked_peer_core_floor_unmet")
    lock = pair["3y"].calculation["evidence_selection"]["peer_lock"]
    assert lock["status"] == "locked"
    assert lock["selected_sic_prefix_level"] == 4
    assert len(lock["selected_candidate_listing_ids"]) == 3
    assert lock["widening_after_assessment"] is False
    assert all(
        not item["ticker"].startswith("L3")
        for item in pair["3y"].calculation["locked_peer_candidates"]
    )


@pytest.mark.django_db
def test_v4_duplicate_authoritative_issuer_identity_is_rejected(tmp_path: Path) -> None:
    store = AssetStore(tmp_path)
    target = _listing("DEDUP")
    target_price, target_asset = _company_evidence(store, target, sic="3571")
    peer_one = _listing("DPA")
    peer_one_price, peer_one_asset = _company_evidence(store, peer_one, sic="3571")
    peer_two = _listing("DPB")
    peer_two_price, peer_two_asset = _company_evidence(store, peer_two, sic="3571")
    duplicate_listing = Listing.objects.create(
        security=peer_one.security,
        ticker="DPA.B",
        provider_symbol="DPA.B",
        exchange_mic="XNYS",
        currency="USD",
        region=Region.US,
    )
    duplicate_price, duplicate_asset = _write_price_asset(
        store,
        duplicate_listing,
        close=55.0,
    )
    listings = [target, peer_one, duplicate_listing, peer_two]
    prices = {
        str(target.pk): target_price,
        str(peer_one.pk): peer_one_price,
        str(duplicate_listing.pk): duplicate_price,
        str(peer_two.pk): peer_two_price,
    }
    assets = {
        str(target.pk): target_asset,
        str(peer_one.pk): peer_one_asset,
        str(duplicate_listing.pk): duplicate_asset,
        str(peer_two.pk): peer_two_asset,
    }
    with pytest.raises(ValueError, match="SEC mapping authority"):
        build_long_forecasts_v4(
            listings=listings,
            current_prices=prices,
            price_assets=assets,
            asof=AsOfData(DECISION_TIME, store),
            data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
            target_date=TARGET_DATE,
            config=load_long_forecast_v4_config(V4_PATH),
        )


@pytest.mark.django_db
def test_v4_service_rejects_duplicate_authoritative_issuer_before_data_reads(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    first = _listing("DUPSVC")
    second = _listing("DUPALT")
    second.security.company.cik = first.security.company.cik
    second.security.company.save(update_fields=["cik"])
    second.ticker = first.ticker
    second.provider_symbol = first.provider_symbol
    second.save(update_fields=["ticker", "provider_symbol"])
    snapshot = _snapshot([first, second])

    with (
        patch.object(
            AsOfData,
            "price_frame_with_diagnostics",
            side_effect=AssertionError("price read must not occur"),
        ) as price_read,
        patch.object(
            AsOfData,
            "fundamental_facts",
            side_effect=AssertionError("fact read must not occur"),
        ) as fact_read,
        patch.object(
            long_v4_module,
            "_lock_peer_cohort",
            side_effect=AssertionError("peer lock must not occur"),
        ) as peer_lock,
        pytest.raises(ValueError, match="SEC mapping authority"),
    ):
        _run_v4(snapshot, store)

    price_read.assert_not_called()
    fact_read.assert_not_called()
    peer_lock.assert_not_called()
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
def test_v4_path_tampering_is_rejected_even_when_both_views_match(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    store = AssetStore(tmp_path)
    listings, prices, assets = _cohort(store)
    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(listings[0].pk)]
    altered: dict[str, object] = {}
    for horizon in ("3y", "5y"):
        calculation = deepcopy(pair[horizon].calculation)
        calculation["scenario_paths"]["base"]["years"][0]["entity_growth"] += 0.01
        altered[horizon] = replace(pair[horizon], calculation=calculation)
    with pytest.raises(ValueError, match="path arithmetic"):
        validate_long_v4_forecast_pair(  # type: ignore[arg-type]
            altered,
            config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH,
        )


@pytest.mark.django_db
def test_v4_adr_gets_one_explicit_withheld_pair_and_never_enters_peers(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    adr = _listing("ADR", security_type=Security.SecurityType.ADR)
    adr_price, adr_asset = _company_evidence(store, adr, sic="3571")
    listings = [adr]
    prices = {str(adr.pk): adr_price}
    assets = {str(adr.pk): adr_asset}
    for index in range(4):
        peer = _listing(f"C{index}")
        peer_price, peer_asset = _company_evidence(
            store,
            peer,
            sic="3571",
            scale=1.1 + index * 0.1,
        )
        listings.append(peer)
        prices[str(peer.pk)] = peer_price
        assets[str(peer.pk)] = peer_asset

    results = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )
    pair = results[str(adr.pk)]
    assert set(pair) == {"3y", "5y"}
    assert {forecast.calculation["insufficiency_code"] for forecast in pair.values()} == {
        "adr_depositary_receipt_withheld"
    }
    for peer in listings[1:]:
        peer_ids = {
            item["listing_id"]
            for item in results[str(peer.pk)]["3y"].calculation["locked_peer_candidates"]
        }
        assert str(adr.pk) not in peer_ids


@pytest.mark.django_db
def test_v4_scenarios_stress_growth_dilution_and_multiple_separately(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings, prices, assets = _cohort(store)
    calculation = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(listings[0].pk)]["3y"].calculation
    paths = calculation["scenario_paths"]
    assert paths["bear"]["target_entity_growth"] < paths["base"]["target_entity_growth"]
    assert paths["base"]["target_entity_growth"] < paths["bull"]["target_entity_growth"]
    assert paths["bear"]["dilution_rate"] >= paths["base"]["dilution_rate"]
    assert paths["base"]["dilution_rate"] >= paths["bull"]["dilution_rate"]
    assert (
        paths["bear"]["peer_multiple_destination"]
        < paths["base"]["peer_multiple_destination"]
        <= paths["bull"]["peer_multiple_destination"]
    )
    assert [paths[name]["peer_multiple_multiplier"] for name in ("bear", "base", "bull")] == [
        0.8,
        1.0,
        1.15,
    ]


@pytest.mark.django_db
def test_v4_unfavorable_but_valid_peer_growth_is_retained(tmp_path: Path) -> None:
    store = AssetStore(tmp_path)
    target = _listing("UNFAVT")
    target_price, target_asset = _company_evidence(store, target, sic="3571")
    listings = [target]
    prices = {str(target.pk): target_price}
    assets = {str(target.pk): target_asset}
    for index in range(3):
        peer = _listing(f"UNF{index}")
        peer_price, peer_asset = _company_evidence(
            store,
            peer,
            sic="3571",
            scale=1.1 + index * 0.1,
            annual_fcf_values=(
                (94.0, 86.0, 78.0, 70.0) if index == 0 else (70.0, 78.0, 86.0, 94.0)
            ),
        )
        listings.append(peer)
        prices[str(peer.pk)] = peer_price
        assets[str(peer.pk)] = peer_asset
    calculation = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]["3y"].calculation
    adverse = next(item for item in calculation["peer_set"] if item["ticker"] == "UNF0")
    assert adverse["entity_growth_estimate"] < 0
    assert adverse["selected_peer"] is True


@pytest.mark.django_db
def test_v4_accepts_tiny_positive_per_share_values_without_a_nominal_floor(
    tmp_path: Path,
) -> None:
    pair = _target_pair(
        AssetStore(tmp_path),
        ticker="TINY",
        target_options={
            "share_multiplier": 10_000.0,
            "price_multiplier": 1 / 10_000.0,
        },
        peer_options={
            "share_multiplier": 10_000.0,
            "price_multiplier": 1 / 10_000.0,
        },
    )

    assert pair["3y"].scenario.base is not None
    assert pair["3y"].calculation["formula_inputs"]["current_per_share_metric"] < 0.01
    assert pair["3y"].calculation["insufficiency_code"] is None


@pytest.mark.parametrize(
    "value",
    [
        True,
        None,
        "not-a-price",
        float("nan"),
        float("inf"),
        Decimal("0"),
        Decimal("-1"),
        Decimal("0.0000004"),
    ],
)
def test_v4_canonical_price_rejects_invalid_or_rounded_zero(value: object) -> None:
    with pytest.raises(ValueError):
        canonical_long_v4_price(value)


def test_v4_canonical_price_uses_service_float_rounding_boundary() -> None:
    assert Decimal("50.0001") / Decimal("200") == Decimal("0.2500005")
    canonical = canonical_long_v4_price(Decimal("50.0001") / Decimal("200"))
    assert canonical == Decimal("0.250001")
    assert format(canonical, ".6f") == "0.250001"


@pytest.mark.django_db
def test_v4_service_rejects_positive_price_that_rounds_to_zero_atomically(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store, target_close=0.0000004)

    with pytest.raises(ValueError, match="canonical price closure"):
        _run_v4(_snapshot(listings), store)

    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
def test_v4_eps_reconciliation_uses_the_exact_scale_free_denominator(
    tmp_path: Path,
) -> None:
    both_zero = _target_pair(
        AssetStore(tmp_path / "both-zero"),
        ticker="EPSZERO",
        target_options={
            "annual_income_values": (Decimal("0"),) * 4,
            "reported_eps_values": (Decimal("0"),) * 4,
        },
    )
    assert both_zero["3y"].scenario.base is not None
    zero_check = next(
        check
        for check in both_zero["3y"].calculation["split_basis"]["checks"]
        if check["check"] == "reported_diluted_eps_reconciliation"
    )
    assert zero_check["relative_difference_denominator"] == 0
    assert zero_check["relative_difference"] == 0

    one_sided = _target_pair(
        AssetStore(tmp_path / "one-sided"),
        ticker="EPSONE",
        target_options={"reported_eps_values": (Decimal("0"),) * 4},
    )
    assert one_sided["3y"].calculation["insufficiency_code"] == (
        "fcf_reported_eps_reconciliation_failed"
    )
    assert one_sided["3y"].calculation["split_basis"]["assessment_status"] == (
        "incompatible_or_unverified"
    )

    near_zero = _target_pair(
        AssetStore(tmp_path / "near-zero"),
        ticker="EPSNEAR",
        target_options={
            "share_multiplier": 1_000.0,
            "annual_income_values": (Decimal("0.00000001"),) * 4,
            "reported_eps_values": (Decimal("0"),) * 4,
        },
    )
    assert near_zero["3y"].calculation["insufficiency_code"] == (
        "fcf_reported_eps_reconciliation_failed"
    )
    near_check = next(
        check
        for check in near_zero["3y"].calculation["split_basis"]["checks"]
        if check["check"] == "reported_diluted_eps_reconciliation"
    )
    assert 0 < near_check["relative_difference_denominator"] < 1e-12
    assert near_check["relative_difference"] == 1

    boundary_shares = (100.0, 110.0, 120.0, 130.0)
    boundary_income = tuple(Decimal(str(value)) for value in boundary_shares)
    boundary = _target_pair(
        AssetStore(tmp_path / "boundary"),
        ticker="EPSBOUND",
        target_options={
            "annual_shares": boundary_shares,
            "annual_income_values": boundary_income,
            "reported_eps_values": (Decimal("0.85"),) * 4,
        },
    )
    assert boundary["3y"].scenario.base is not None
    over = _target_pair(
        AssetStore(tmp_path / "over"),
        ticker="EPSOVER",
        target_options={
            "annual_shares": boundary_shares,
            "annual_income_values": boundary_income,
            "reported_eps_values": (Decimal("0.849"),) * 4,
        },
    )
    assert over["3y"].calculation["insufficiency_code"] == (
        "fcf_reported_eps_reconciliation_failed"
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("ticker", "target_options", "reason_code", "failing_check"),
    [
        (
            "ANNSHARE",
            {"annual_shares": (100.0, 110.0, 130.0, 140.0)},
            "fcf_annual_share_continuity_failed",
            "adjacent_annual_diluted_share_continuity",
        ),
        (
            "TTMSHARE",
            {"ttm_share_multiplier": 1.16},
            "fcf_annual_to_ttm_share_continuity_failed",
            "ttm_to_latest_annual_diluted_share_continuity",
        ),
    ],
)
def test_v4_failed_share_assessment_keeps_prior_and_failing_evidence(
    tmp_path: Path,
    ticker: str,
    target_options: dict[str, object],
    reason_code: str,
    failing_check: str,
) -> None:
    pair = _target_pair(
        AssetStore(tmp_path),
        ticker=ticker,
        target_options=target_options,
    )
    calculation = pair["3y"].calculation
    assessment = calculation["split_basis"]
    assert calculation["insufficiency_code"] == reason_code
    for forecast in pair.values():
        assert forecast.calculation["config_hash"] == LONG_V4_EFFECTIVE_CONFIG_HASH
        assert forecast.calculation["fundamentals_config_file_sha256"] == (
            "829ed267eec62304804c9ef71f1816636389c2b0ad77423d8a534a00a5e1ae30"
        )
        assert forecast.calculation["fundamentals_config_hash"] == (
            "7822a1faaae1c8028d71851337a7dbb7a65d9aeaa0f44478e3814a6468b62604"
        )
    assert assessment["assessment_status"] == "incompatible_or_unverified"
    assert assessment["assessed_through"]
    assert assessment["post_period_split_status"] == "unverified"
    assert assessment["checks"][-1]["check"] == failing_check
    assert assessment["checks"][-1]["inclusive_tolerance"] == 0.15
    expected_ids = list(
        dict.fromkeys(
            fact_id for check in assessment["checks"] for fact_id in check["relevant_fact_ids"]
        )
    )
    assert assessment["relevant_fact_ids"] == expected_ids
    assert set(expected_ids).issubset(
        {item["id"] for item in calculation["evidence_catalog"]["facts"]}
    )
    assert "verified_through" not in assessment


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("mutation", "reason_code", "check_name"),
    [
        (
            "ttm_period",
            "fcf_ttm_period_mismatch",
            "ttm_entity_share_period_compatibility",
        ),
        (
            "annual_alias",
            "fcf_annual_alias_or_unit_incompatible",
            "annual_entity_share_signature_compatibility",
        ),
        (
            "annual_unit",
            "fcf_annual_unit_incompatible",
            "annual_required_unit_compatibility",
        ),
        (
            "annual_to_ttm_alias",
            "fcf_annual_to_ttm_alias_or_unit_incompatible",
            "annual_to_ttm_alias_unit_compatibility",
        ),
        (
            "annual_to_ttm_derivation",
            "fcf_annual_to_ttm_derivation_incompatible",
            "annual_to_ttm_derivation_compatibility",
        ),
        (
            "annual_to_ttm_period",
            "fcf_annual_to_ttm_period_incompatible",
            "annual_to_ttm_period_overlap",
        ),
    ],
)
def test_v4_target_incompatibilities_append_semantic_failing_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    reason_code: str,
    check_name: str,
) -> None:
    from dataclasses import replace

    store = AssetStore(tmp_path)
    target = _listing(f"CHK{mutation[:8]}")
    target_price, target_asset = _company_evidence(store, target, sic="3571")
    listings = [target]
    prices = {str(target.pk): target_price}
    assets = {str(target.pk): target_asset}
    for index in range(3):
        peer = _listing(f"CHP{mutation[:3]}{index}")
        price, asset = _company_evidence(store, peer, sic="3571", scale=1.1 + index * 0.1)
        listings.append(peer)
        prices[str(peer.pk)] = price
        assets[str(peer.pk)] = asset
    original_builder = long_v4_module.build_sec_fundamental_series

    def altered_series(facts, **kwargs):
        materialized = tuple(facts)
        result = original_builder(materialized, **kwargs)
        if not materialized or materialized[0].company_id != target.security.company_id:
            return result
        annual = {key: tuple(values) for key, values in result.annual.items()}
        ttm = dict(result.ttm)
        if mutation == "ttm_period":
            ttm["weighted_average_diluted_shares"] = replace(
                ttm["weighted_average_diluted_shares"],
                period_start=date(2025, 1, 2),
            )
        elif mutation == "annual_alias":
            values = list(annual["free_cash_flow"])
            values[1] = replace(values[1], source_concepts=("us-gaap:WrongAlias",))
            annual["free_cash_flow"] = tuple(values)
        elif mutation == "annual_unit":
            annual["free_cash_flow"] = tuple(
                replace(value, unit="EUR") for value in annual["free_cash_flow"]
            )
        elif mutation == "annual_to_ttm_alias":
            ttm["free_cash_flow"] = replace(
                ttm["free_cash_flow"],
                source_concepts=("us-gaap:WrongAlias",),
            )
        elif mutation == "annual_to_ttm_derivation":
            ttm["free_cash_flow"] = replace(
                ttm["free_cash_flow"],
                derivation="incompatible_derivation",
            )
        elif mutation == "annual_to_ttm_period":
            impossible_start = date(2026, 1, 1)
            ttm["free_cash_flow"] = replace(
                ttm["free_cash_flow"],
                period_start=impossible_start,
            )
            ttm["weighted_average_diluted_shares"] = replace(
                ttm["weighted_average_diluted_shares"],
                period_start=impossible_start,
            )
        return replace(result, annual=annual, ttm=ttm)

    monkeypatch.setattr(long_v4_module, "build_sec_fundamental_series", altered_series)
    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]
    assessment = pair["3y"].calculation["split_basis"]
    failing = assessment["checks"][-1]

    assert pair["3y"].calculation["insufficiency_code"] == reason_code
    assert assessment["assessment_status"] == "incompatible_or_unverified"
    assert assessment["assessed_through"] == failing["assessed_through"]
    assert failing["check"] == check_name
    assert failing["outcome"] == "incompatible"
    assert failing["compared_fields"]
    assert isinstance(failing["expected"], dict)
    assert isinstance(failing["actual"], dict)
    assert assessment["relevant_fact_ids"] == list(
        dict.fromkeys(
            fact_id for check in assessment["checks"] for fact_id in check["relevant_fact_ids"]
        )
    )
    validate_long_v4_forecast_pair(pair, config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH)


@pytest.mark.django_db
def test_v4_annual_gap_check_orders_baseline_then_failing_entity_and_shares(
    tmp_path: Path,
) -> None:
    gap_periods = (
        ANNUAL_PERIODS[0],
        (date(2023, 1, 2), ANNUAL_PERIODS[1][1]),
        ANNUAL_PERIODS[2],
        ANNUAL_PERIODS[3],
    )
    pair = _target_pair(
        AssetStore(tmp_path),
        ticker="ANNGAP",
        target_options={"annual_periods": gap_periods},
    )
    calculation = pair["3y"].calculation
    failing = calculation["split_basis"]["checks"][-1]
    catalog = {item["id"]: item for item in calculation["evidence_catalog"]["facts"]}

    assert calculation["insufficiency_code"] == "fcf_annual_period_gap"
    assert failing["check"] == "adjacent_annual_period_continuity"
    assert failing["outcome"] == "incompatible"
    assert [catalog[fact_id]["concept"] for fact_id in failing["relevant_fact_ids"]] == [
        "operating_cash_flow",
        "capital_expenditure",
        "weighted_average_diluted_shares",
        "operating_cash_flow",
        "capital_expenditure",
        "weighted_average_diluted_shares",
    ]


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("duration_days", "admitted"),
    [(349, False), (350, True), (380, True), (381, False)],
)
def test_v4_annual_duration_bounds_record_complete_target_assessment(
    tmp_path: Path,
    duration_days: int,
    admitted: bool,
) -> None:
    pair = _target_pair(
        AssetStore(tmp_path),
        ticker=f"DURT{duration_days}",
        target_options={"annual_periods": _annual_periods_with_duration(duration_days)},
        peer_prefix=f"DURTP{duration_days}",
    )
    calculation = pair["3y"].calculation
    if admitted:
        assert pair["3y"].scenario.base is not None
        assert calculation["insufficiency_code"] is None
        return

    assert pair["3y"].scenario.base is None
    assert calculation["insufficiency_code"] == "fcf_annual_duration_incompatible"
    assessment = calculation["split_basis"]
    failure = assessment["checks"][-1]
    assert assessment["assessment_status"] == "incompatible_or_unverified"
    assert assessment["assessed_through"] == failure["assessed_through"]
    assert failure == {
        "check": "annual_duration_compatibility",
        "outcome": "incompatible",
        "assessed_through": failure["assessed_through"],
        "compared_fields": ["duration_days", "period_start", "period_end"],
        "expected": {
            "minimum_days": 350,
            "maximum_days": 380,
            "inclusive": True,
        },
        "actual": {
            "duration_days": duration_days,
            "period_start": failure["actual"]["period_start"],
            "period_end": failure["actual"]["period_end"],
        },
        "relevant_fact_ids": failure["relevant_fact_ids"],
        "period_identity": failure["period_identity"],
    }
    concepts = {item["id"]: item["concept"] for item in calculation["evidence_catalog"]["facts"]}
    assert len(failure["relevant_fact_ids"]) == 3
    assert {concepts[fact_id] for fact_id in failure["relevant_fact_ids"]} == {
        "operating_cash_flow",
        "capital_expenditure",
        "weighted_average_diluted_shares",
    }
    validate_long_v4_forecast_pair(pair, config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH)


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("duration_days", "admitted"),
    [(349, False), (350, True), (380, True), (381, False)],
)
def test_v4_annual_duration_bounds_apply_to_locked_peers(
    tmp_path: Path,
    duration_days: int,
    admitted: bool,
) -> None:
    store = AssetStore(tmp_path)
    target = _listing(f"DURP{duration_days}T")
    target_price, target_asset = _company_evidence(store, target, sic="3571")
    listings = [target]
    prices = {str(target.pk): target_price}
    assets = {str(target.pk): target_asset}
    for index in range(3):
        peer = _listing(f"DURP{duration_days}{index}")
        price, asset = _company_evidence(
            store,
            peer,
            sic="3571",
            scale=1.1 + index * 0.1,
            annual_periods=_annual_periods_with_duration(duration_days),
        )
        listings.append(peer)
        prices[str(peer.pk)] = price
        assets[str(peer.pk)] = asset
    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]

    if admitted:
        assert pair["3y"].scenario.base is not None
    else:
        assert pair["3y"].calculation["insufficiency_code"] == ("locked_peer_core_floor_unmet")
        for candidate in pair["3y"].calculation["locked_peer_candidates"]:
            assert candidate["reason_code"] == "fcf_annual_duration_incompatible"
            failure = candidate["split_basis"]["checks"][-1]
            assert failure["check"] == "annual_duration_compatibility"
            assert failure["actual"]["duration_days"] == duration_days
            assert failure["relevant_fact_ids"]
    validate_long_v4_forecast_pair(pair, config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH)


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("peer_options", "reason_code"),
    [
        (
            {"reported_eps_values": (Decimal("0"),) * 4},
            "fcf_reported_eps_reconciliation_failed",
        ),
        (
            {"annual_shares": (100.0, 110.0, 130.0, 140.0)},
            "fcf_annual_share_continuity_failed",
        ),
    ],
)
def test_v4_locked_peer_failure_carries_complete_split_assessment(
    tmp_path: Path,
    peer_options: dict[str, object],
    reason_code: str,
) -> None:
    store = AssetStore(tmp_path)
    target = _listing(f"PEERFAIL{reason_code[-3:]}")
    target_price, target_asset = _company_evidence(store, target, sic="3571")
    listings = [target]
    prices = {str(target.pk): target_price}
    assets = {str(target.pk): target_asset}
    failing_peer_id = ""
    for index in range(3):
        peer = _listing(f"PFAIL{index}{reason_code[-2:]}")
        options = peer_options if index == 0 else {}
        price, asset = _company_evidence(
            store,
            peer,
            sic="3571",
            scale=1.1 + index * 0.1,
            **options,
        )
        if index == 0:
            failing_peer_id = str(peer.pk)
        listings.append(peer)
        prices[str(peer.pk)] = price
        assets[str(peer.pk)] = asset
    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]
    candidate = next(
        item
        for item in pair["3y"].calculation["locked_peer_candidates"]
        if item["listing_id"] == failing_peer_id
    )

    assert candidate["reason_code"] == reason_code
    assert candidate["split_basis"]["assessment_status"] == "incompatible_or_unverified"
    assert candidate["split_basis"]["checks"][-1]["outcome"] == "incompatible"
    validate_long_v4_forecast_pair(pair, config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH)


@pytest.mark.django_db
@pytest.mark.parametrize(
    "mutation",
    ["peer_id", "peer_fact", "peer_context_asset", "selected_split_copy"],
)
def test_v4_validator_rejects_tampered_peer_assessment_closure(
    tmp_path: Path,
    mutation: str,
) -> None:
    from dataclasses import replace

    store = AssetStore(tmp_path)
    listings, prices, assets = _cohort(store)
    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(listings[0].pk)]
    altered: dict[str, LongForecastV4] = {}
    for horizon, forecast in pair.items():
        calculation = deepcopy(forecast.calculation)
        candidates = calculation["locked_peer_candidates"]
        selected_candidate = next(item for item in candidates if item["selected_peer"])
        selected_peer = next(
            item
            for item in calculation["peer_set"]
            if item["listing_id"] == selected_candidate["listing_id"]
        )
        if mutation == "peer_id":
            selected_candidate["listing_id"] = str(uuid4())
        elif mutation == "peer_fact":
            target_fact = next(
                item
                for item in calculation["evidence_catalog"]["facts"]
                if item["owner_listing_id"] == str(listings[0].pk)
            )
            selected_candidate["split_basis"]["checks"][-1]["relevant_fact_ids"] = [
                target_fact["id"]
            ]
            selected_candidate["split_basis"]["relevant_fact_ids"] = [target_fact["id"]]
        elif mutation == "peer_context_asset":
            peer_fact = next(
                item
                for item in calculation["evidence_catalog"]["facts"]
                if item["owner_listing_id"] == selected_candidate["listing_id"]
            )
            peer_fact["submissions_context_asset_id"] = str(uuid4())
        else:
            selected_peer["split_basis"] = dict(selected_peer["split_basis"])
            selected_peer["split_basis"]["assessed_through"] = "1900-01-01"
        altered[horizon] = replace(forecast, calculation=calculation)

    with pytest.raises(ValueError):
        validate_long_v4_forecast_pair(
            altered,
            config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH,
        )


@pytest.mark.django_db
def test_v4_full_cohort_is_invariant_under_a_200_for_1_split(tmp_path: Path) -> None:
    base = _target_pair(
        AssetStore(tmp_path / "base"),
        ticker="SCALEA",
        target_options={},
        peer_prefix="SAP",
    )
    split = _target_pair(
        AssetStore(tmp_path / "split"),
        ticker="SCALEB",
        target_options={"share_multiplier": 200.0, "price_multiplier": 1 / 200.0},
        peer_options={"share_multiplier": 200.0, "price_multiplier": 1 / 200.0},
        peer_prefix="SBP",
    )
    base_calculation = base["3y"].calculation
    split_calculation = split["3y"].calculation

    for horizon in ("3y", "5y"):
        for name in ("bear", "base", "bull"):
            assert getattr(base[horizon].scenario, name) == pytest.approx(
                getattr(split[horizon].scenario, name),
                abs=1e-12,
            )
        assert base[horizon].scenario.confidence == split[horizon].scenario.confidence
        assert base[horizon].scenario.confidence_status == split[horizon].scenario.confidence_status
    assert base_calculation["metric_family"] == split_calculation["metric_family"]
    for key in (
        "entity_growth_raw",
        "entity_growth_capped",
        "target_entity_growth",
        "diluted_share_changes_raw",
        "dilution_base",
        "peer_entity_growth",
        "peer_multiple",
        "current_multiple_raw",
        "current_multiple_capped_reversion_anchor",
    ):
        assert base_calculation["formula_inputs"][key] == pytest.approx(
            split_calculation["formula_inputs"][key]
        )
    base_lock = base_calculation["evidence_selection"]["peer_lock"]
    split_lock = split_calculation["evidence_selection"]["peer_lock"]
    assert (
        base_lock["status"],
        base_lock["selected_sic_prefix_level"],
        base_lock["selected_sic_prefix"],
        base_lock["selected_identity_floor"],
        [level["candidate_count"] for level in base_lock["examined_levels"]],
    ) == (
        split_lock["status"],
        split_lock["selected_sic_prefix_level"],
        split_lock["selected_sic_prefix"],
        split_lock["selected_identity_floor"],
        [level["candidate_count"] for level in split_lock["examined_levels"]],
    )
    for scenario in ("bear", "base", "bull"):
        for base_year, split_year in zip(
            base_calculation["scenario_paths"][scenario]["years"],
            split_calculation["scenario_paths"][scenario]["years"],
            strict=True,
        ):
            for key in (
                "entity_growth",
                "dilution_rate",
                "per_share_growth",
                "cumulative_entity_factor",
                "cumulative_per_share_factor",
                "multiple",
                "cumulative_price_return",
                "return_from_level_identity",
                "return_from_factor_identity",
            ):
                assert base_year[key] == pytest.approx(split_year[key])
            assert base_year["projected_per_share_metric"] == pytest.approx(
                split_year["projected_per_share_metric"] * 200
            )
            assert base_year["projected_price"] == pytest.approx(
                split_year["projected_price"] * 200
            )


@pytest.mark.django_db
def test_v4_no_floor_trace_retains_every_examined_classification_asset(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    target = _listing("NOFLOOR")
    target_price, target_asset = _company_evidence(store, target, sic="3571")
    listings = [target]
    prices = {str(target.pk): target_price}
    assets = {str(target.pk): target_asset}
    for index, sic in enumerate(("3571", "3572")):
        peer = _listing(f"NF{index}")
        price, asset = _company_evidence(store, peer, sic=sic)
        listings.append(peer)
        prices[str(peer.pk)] = price
        assets[str(peer.pk)] = asset

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]
    calculation = pair["3y"].calculation
    validate_long_v4_forecast_pair(pair, config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH)
    trace = calculation["evidence_selection"]["peer_lock"]
    assert trace["status"] == "no_floor"
    assert [level["sic_prefix_level"] for level in trace["examined_levels"]] == [4, 3, 2]
    assert all(level["floor_met"] is False for level in trace["examined_levels"])
    assert trace["selected_sic_prefix_level"] is None
    assert trace["selected_candidate_listing_ids"] == []
    assert trace["widening_after_assessment"] is False
    retained_ids = {
        candidate["listing_id"]
        for level in trace["examined_levels"]
        for candidate in level["candidates"]
    }
    assert retained_ids == {str(listing.pk) for listing in listings[1:]}
    classification_entries = calculation["evidence_catalog"]["classifications"]
    assert {entry["owner_listing_id"] for entry in classification_entries} == {
        str(listing.pk) for listing in listings
    }
    classification_asset_ids = {entry["source_asset_id"] for entry in classification_entries}
    assert classification_asset_ids.issubset(
        {entry["id"] for entry in calculation["source_manifest"]}
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_fact",
        "accession",
        "value",
        "taxonomy",
        "source_concept",
        "unit",
        "period_start",
        "period_end",
        "fiscal_year",
        "fiscal_period",
        "frame",
        "form",
        "filed_date",
        "report_end",
        "malformed_value",
        "filing_missing_accession",
        "filing_conflicting_accession",
        "filing_time",
    ],
)
def test_v4_raw_sec_semantic_mismatch_aborts_atomically_without_paths(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = AssetStore(tmp_path)
    target = _listing(f"RAW{mutation[:8]}")
    _company_evidence(
        store,
        target,
        sic="3571",
        evidence_mutator=_raw_mutator(mutation),
    )
    peers = [_listing(f"R{mutation[:3]}P{index}") for index in range(3)]
    for index, peer in enumerate(peers):
        _company_evidence(store, peer, sic="3571", scale=1.1 + index * 0.1)
    snapshot = _snapshot([target, *peers])

    with pytest.raises(
        ValueError,
        match=r"^Long-v4 SEC raw evidence authority is incompatible$",
    ) as error:
        _run_v4(snapshot, store)

    assert str(tmp_path) not in str(error.value)
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize(
    "mutation",
    [
        "context_uuid_missing",
        "context_uuid_wrong",
        "context_hash_missing",
        "context_hash_wrong",
        "context_cik_wrong",
        "companyfacts_cik_wrong",
        "history_unsafe",
        "history_unlisted",
        "history_duplicate",
        "history_metadata_blank",
        "history_metadata_unsafe",
    ],
)
def test_v4_submissions_context_and_history_identity_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = AssetStore(tmp_path)
    target = _listing(f"CTX{mutation[:8]}")
    context_mutator = _context_metadata_mutator(mutation)
    history_mutator = _history_metadata_mutator(mutation)
    evidence_mutator = _context_evidence_mutator(mutation)
    _company_evidence(
        store,
        target,
        sic="3571",
        evidence_mutator=evidence_mutator,
        context_metadata_mutator=context_mutator,
        history_metadata_mutator=history_mutator,
    )
    peers = [_listing(f"C{mutation[:3]}P{index}") for index in range(3)]
    for index, peer in enumerate(peers):
        _company_evidence(store, peer, sic="3571", scale=1.1 + index * 0.1)
    snapshot = _snapshot([target, *peers])

    with pytest.raises(
        ValueError,
        match=r"^Long-v4 SEC raw evidence authority is incompatible$",
    ):
        _run_v4(snapshot, store)

    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize("mutation", ["sic", "description", "backdated"])
def test_v4_classification_raw_contradiction_aborts(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = AssetStore(tmp_path)
    target = _listing(f"CLS{mutation}")
    overrides: dict[str, object]
    if mutation == "sic":
        overrides = {"code": "9999"}
    elif mutation == "description":
        overrides = {"description": "Contradictory description"}
    else:
        overrides = {
            "observed_at": datetime(2026, 2, 19, tzinfo=UTC),
            "available_at": datetime(2026, 2, 20, tzinfo=UTC),
        }
    _company_evidence(
        store,
        target,
        sic="3571",
        classification_overrides=overrides,
    )
    peers = [_listing(f"CLSP{mutation[:2]}{index}") for index in range(3)]
    for index, peer in enumerate(peers):
        _company_evidence(store, peer, sic="3571", scale=1.1 + index * 0.1)

    with pytest.raises(
        ValueError,
        match=r"^Long-v4 SEC raw evidence authority is incompatible$",
    ):
        _run_v4(_snapshot([target, *peers]), store)

    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
def test_v4_missing_classification_remains_explicit_withholding(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    target = _listing("CLSMISSING")
    target_price, target_asset = _company_evidence(
        store,
        target,
        sic="3571",
        create_classification=False,
    )
    peers = [_listing(f"CLSMISS{index}") for index in range(3)]
    prices = {str(target.pk): target_price}
    assets = {str(target.pk): target_asset}
    for index, peer in enumerate(peers):
        price, asset = _company_evidence(store, peer, sic="3571", scale=1.1 + index * 0.1)
        prices[str(peer.pk)] = price
        assets[str(peer.pk)] = asset

    pair = build_long_forecasts_v4(
        listings=[target, *peers],
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]

    assert pair["3y"].calculation["insufficiency_code"] == (
        "classification_unavailable_or_ambiguous"
    )
    assert pair["3y"].scenario.base is None


@pytest.mark.django_db
def test_v4_valid_correction_is_bound_to_exact_observation_event(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings, prices, assets = _cohort(store)
    target = listings[0]
    correction = _append_fact_correction(store, target)

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]
    catalog_fact = next(
        item
        for item in pair["3y"].calculation["evidence_catalog"]["facts"]
        if item["id"] == str(correction.pk)
    )
    event = SourceObservationEvent.objects.get(
        source_asset_id=correction.source_asset_id,
        observed_at=correction.available_at,
    )

    assert catalog_fact["correction_observation_event"] == {
        "id": str(event.pk),
        "provider": "sec",
        "kind": "sec_companyfacts",
        "subject": target.security.company.cik,
        "source_asset_id": str(correction.source_asset_id),
        "content_sha256": correction.source_asset.sha256,
        "observed_at": event.observed_at.isoformat(),
        "recorded_at": event.recorded_at.isoformat(),
    }
    assert (
        catalog_fact["submissions_context_asset_id"]
        == (correction.source_asset.metadata["submissions_asset_id"])
    )


@pytest.mark.django_db
@pytest.mark.parametrize("mutation", ["period", "unit", "period_and_unit"])
def test_ingested_period_and_unit_correction_is_cutoff_safe_in_v4(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = AssetStore(tmp_path)
    listings, prices, assets = _cohort(store, include_fcf=False)
    target = listings[0]
    original = FundamentalFact.objects.get(
        company=target.security.company,
        concept="diluted_eps",
        fiscal_period="FY",
        period_end=date(2025, 12, 31),
    )
    original_source = original.source_asset
    historical_cutoff = datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC)
    correction_boundary = historical_cutoff + timedelta(days=3)
    issuance_boundary = correction_boundary + timedelta(days=1)

    before = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(issuance_boundary, store),
        data_cutoff=historical_cutoff,
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]

    document = json.loads(store.read_bytes(original_source.relative_path))
    source_name = original.source_concept.split(":", 1)[1]
    units = document["facts"]["us-gaap"][source_name]["units"]
    original_rows = units["USD/shares"]
    corrected_raw = next(row for row in original_rows if row["accn"] == original.accession)
    if mutation in {"period", "period_and_unit"}:
        corrected_raw["start"] = "2025-01-02"
    if mutation in {"unit", "period_and_unit"}:
        units["USD/shares"] = [row for row in original_rows if row["accn"] != original.accession]
        units["USD-per-shares"] = [corrected_raw]
    corrected_payload = json.dumps(document, sort_keys=True).encode()
    correction_source = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=target.security.company.cik,
        available_at=correction_boundary,
        payload=corrected_payload,
        metadata=dict(original_source.metadata),
    )
    with patch(
        "django.db.models.fields.timezone.now",
        return_value=correction_boundary,
    ):
        event = sec_ingestion_module._record_observation_event(
            asset=correction_source,
            kind="sec_companyfacts",
            subject=target.security.company.cik,
            digest=correction_source.sha256,
            observed_at=correction_boundary,
        )
    context = DataAsset.objects.get(pk=correction_source.metadata["submissions_asset_id"])
    current = derive_sec_current_submissions(
        store.read_bytes(context.relative_path),
        source_asset=context,
        expected_cik=target.security.company.cik,
    )
    filing_records = list(current.filings)
    for filename in current.historical_filenames:
        history = DataAsset.objects.get(
            provider="sec",
            kind="sec_submissions_history",
            subject=target.security.company.cik,
            metadata__filename=filename,
        )
        filing_records.extend(
            derive_sec_historical_submissions(
                store.read_bytes(history.relative_path),
                source_asset=history,
                expected_cik=target.security.company.cik,
                filename=filename,
                allowed_filenames=current.historical_filenames,
            )
        )

    with patch(
        "django.db.models.fields.timezone.now",
        return_value=correction_boundary,
    ):
        created, _reused = sec_ingestion_module._normalize_companyfacts(
            company=target.security.company,
            payload=corrected_payload,
            source_asset=correction_source,
            observed_at=correction_boundary,
            filing_records=tuple(filing_records),
            config=load_sec_fundamentals_config(),
            store=store,
        )
    correction = FundamentalFact.objects.get(
        company=target.security.company,
        concept="diluted_eps",
        accession=original.accession,
        source_revision=2,
    )

    assert created == 1
    assert (correction.period_identity != original.period_identity) is (
        mutation in {"period", "period_and_unit"}
    )
    assert correction.period_start == (
        date(2025, 1, 2) if mutation in {"period", "period_and_unit"} else original.period_start
    )
    assert correction.unit == (
        "USD-per-shares" if mutation in {"unit", "period_and_unit"} else original.unit
    )
    assert correction.available_at == correction_boundary
    assert correction.availability_basis == CORRECTION_AVAILABILITY_BASIS
    assert correction.source_asset_id == correction_source.pk
    assert event.source_asset_id == correction.source_asset_id
    assert event.observed_at == correction.available_at
    assert "same_accession_correction" in correction.quality_flags

    before_correction = list(
        AsOfData(issuance_boundary, store)
        .fundamental_facts(
            company_id=target.security.company_id,
            concepts=["diluted_eps"],
            available_through=historical_cutoff,
        )
        .filter(accession=original.accession)
    )
    after_correction = list(
        AsOfData(issuance_boundary, store)
        .fundamental_facts(
            company_id=target.security.company_id,
            concepts=["diluted_eps"],
            available_through=correction_boundary,
        )
        .filter(accession=original.accession)
    )
    assert [fact.pk for fact in before_correction] == [original.pk]
    assert {fact.pk for fact in after_correction} == {original.pk, correction.pk}

    historical_results = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(issuance_boundary, store),
        data_cutoff=historical_cutoff,
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )
    historical = historical_results[str(target.pk)]
    after = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(issuance_boundary, store),
        data_cutoff=correction_boundary,
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]
    assert {horizon: forecast.scenario_payload() for horizon, forecast in historical.items()} == {
        horizon: forecast.scenario_payload() for horizon, forecast in before.items()
    }
    assert (
        historical["3y"].calculation["evidence_catalog"]
        == (before["3y"].calculation["evidence_catalog"])
    )
    assert (
        historical["3y"].calculation["source_manifest"]
        == (before["3y"].calculation["source_manifest"])
    )
    assert str(original.pk) in before["3y"].calculation["evidence_catalog"]["selected_fact_ids"]
    assert str(original.pk) in historical["3y"].calculation["evidence_catalog"]["selected_fact_ids"]
    historical_fact_ids = {
        item["id"] for item in historical["3y"].calculation["evidence_catalog"]["facts"]
    }
    historical_payload = json.dumps(
        historical["3y"].calculation["evidence_catalog"],
        sort_keys=True,
    )
    historical_manifest_ids = {
        item["id"] for item in historical["3y"].calculation["source_manifest"]
    }
    after_catalog = after["3y"].calculation["evidence_catalog"]
    after_fact_ids = {item["id"] for item in after_catalog["facts"]}
    assert str(correction.pk) not in historical_fact_ids
    assert str(correction_source.pk) not in historical_payload
    assert correction_source.sha256 not in historical_payload
    assert str(event.pk) not in historical_payload
    assert str(correction_source.pk) not in historical_manifest_ids
    for forecast_pair in historical_results.values():
        for forecast in forecast_pair.values():
            complete_payload = json.dumps(forecast.calculation, sort_keys=True)
            assert str(correction_source.pk) not in complete_payload
            assert correction_source.sha256 not in complete_payload
            assert str(event.pk) not in complete_payload
    assert str(correction.pk) in after_fact_ids
    assert str(correction_source.pk) in {
        item["id"] for item in after["3y"].calculation["source_manifest"]
    }
    correction_payload = next(
        item for item in after_catalog["facts"] if item["id"] == str(correction.pk)
    )
    assert correction_payload["source_revision"] == 2
    assert correction_payload["correction_observation_event"]["id"] == str(event.pk)
    assert str(original.pk) not in after_catalog["selected_fact_ids"]
    assert after["3y"].calculation["insufficiency_code"] in {
        "annual_tuple_incomplete",
        "annual_unit_incompatible",
    }
    assert after["3y"].scenario.base is None
    assert str(correction.pk) in after_catalog["assessed_fact_ids"]

    with patch(
        "django.db.models.fields.timezone.now",
        return_value=issuance_boundary,
    ):
        duplicate_created, _duplicate_reused = sec_ingestion_module._normalize_companyfacts(
            company=target.security.company,
            payload=corrected_payload,
            source_asset=correction_source,
            observed_at=correction_boundary,
            filing_records=tuple(filing_records),
            config=load_sec_fundamentals_config(),
            store=store,
        )
    assert duplicate_created == 0
    assert (
        FundamentalFact.objects.filter(
            company=target.security.company,
            concept="diluted_eps",
            accession=original.accession,
        ).count()
        == 2
    )


@pytest.mark.django_db
def test_v4_raw_lineage_preserves_distinct_same_accession_observations(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings, prices, assets = _cohort(store, include_fcf=False)
    target = listings[0]
    original = FundamentalFact.objects.get(
        company=target.security.company,
        concept="net_income",
        fiscal_period="FY",
        period_end=date(2025, 12, 31),
    )
    source = original.source_asset
    document = json.loads(store.read_bytes(source.relative_path))
    source_name = original.source_concept.split(":", 1)[1]
    observations = document["facts"]["us-gaap"][source_name]["units"][original.unit]
    distinct = dict(next(row for row in observations if row["accn"] == original.accession))
    distinct["start"] = "2024-01-01"
    distinct["val"] = str(original.value + Decimal("7"))
    observations.append(distinct)
    observed_at = datetime(2026, 2, 26, 12, tzinfo=UTC)
    payload = json.dumps(document, sort_keys=True).encode()
    expanded_source = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=target.security.company.cik,
        available_at=observed_at,
        payload=payload,
        metadata=dict(source.metadata),
    )
    filing_records = _filing_records_for_companyfacts_source(
        store,
        target,
        expanded_source,
    )

    with patch("django.db.models.fields.timezone.now", return_value=observed_at):
        created, _reused = sec_ingestion_module._normalize_companyfacts(
            company=target.security.company,
            payload=payload,
            source_asset=expanded_source,
            observed_at=observed_at,
            filing_records=filing_records,
            config=load_sec_fundamentals_config(),
            store=store,
        )

    same_accession = list(
        FundamentalFact.objects.filter(
            company=target.security.company,
            concept="net_income",
            accession=original.accession,
        ).order_by("period_start")
    )
    derivations = derive_sec_companyfacts(
        payload,
        source_asset=expanded_source,
        expected_cik=target.security.company.cik,
        filing_records=filing_records,
        config=load_sec_fundamentals_config(),
    )
    same_accession_derivations = [
        item for item in derivations if item.accession == original.accession
    ]
    assert created == 1
    assert len(same_accession) == 2
    assert [fact.source_revision for fact in same_accession] == [1, 1]
    assert len({item.raw_observation_lineage for item in same_accession_derivations}) == 2

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]
    catalog_ids = {item["id"] for item in pair["3y"].calculation["evidence_catalog"]["facts"]}
    assert {str(fact.pk) for fact in same_accession}.issubset(catalog_ids)


@pytest.mark.django_db
def test_ingestion_reorders_exact_same_accession_rows_then_changes_only_one(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store, include_fcf=False)
    target = listings[0]
    original = FundamentalFact.objects.get(
        company=target.security.company,
        concept="net_income",
        fiscal_period="FY",
        period_end=date(2025, 12, 31),
    )
    source = original.source_asset
    document = json.loads(store.read_bytes(source.relative_path))
    source_name = original.source_concept.split(":", 1)[1]
    observations = document["facts"]["us-gaap"][source_name]["units"][original.unit]
    distinct = dict(next(row for row in observations if row["accn"] == original.accession))
    distinct["start"] = "2024-01-01"
    distinct["val"] = str(original.value + Decimal("7"))
    observations.append(distinct)

    expanded_at = datetime(2026, 2, 24, 12, tzinfo=UTC)
    expanded_payload = json.dumps(document, sort_keys=True).encode()
    expanded_source = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=target.security.company.cik,
        available_at=expanded_at,
        payload=expanded_payload,
        metadata=dict(source.metadata),
    )
    with patch("django.db.models.fields.timezone.now", return_value=expanded_at):
        created, _reused = sec_ingestion_module._normalize_companyfacts(
            company=target.security.company,
            payload=expanded_payload,
            source_asset=expanded_source,
            observed_at=expanded_at,
            filing_records=_filing_records_for_companyfacts_source(
                store,
                target,
                expanded_source,
            ),
            config=load_sec_fundamentals_config(),
            store=store,
        )
    assert created == 1

    reordered = deepcopy(document)
    reordered_observations = reordered["facts"]["us-gaap"][source_name]["units"][original.unit]
    reordered_observations.reverse()
    reordered_at = datetime(2026, 2, 25, 12, tzinfo=UTC)
    reordered_payload = json.dumps(reordered, sort_keys=True).encode()
    reordered_source = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=target.security.company.cik,
        available_at=reordered_at,
        payload=reordered_payload,
        metadata=dict(source.metadata),
    )
    fact_count = FundamentalFact.objects.count()
    event_count = SourceObservationEvent.objects.count()
    with patch("django.db.models.fields.timezone.now", return_value=reordered_at):
        reordered_created, _reordered_reused = sec_ingestion_module._normalize_companyfacts(
            company=target.security.company,
            payload=reordered_payload,
            source_asset=reordered_source,
            observed_at=reordered_at,
            filing_records=_filing_records_for_companyfacts_source(
                store,
                target,
                reordered_source,
            ),
            config=load_sec_fundamentals_config(),
            store=store,
        )

    same_accession = FundamentalFact.objects.filter(
        company=target.security.company,
        concept="net_income",
        accession=original.accession,
    )
    assert reordered_created == 0
    assert FundamentalFact.objects.count() == fact_count
    assert SourceObservationEvent.objects.count() == event_count
    assert same_accession.count() == 2
    assert set(same_accession.values_list("source_revision", flat=True)) == {1}
    assert not any("same_accession_correction" in fact.quality_flags for fact in same_accession)

    uniquely_changed = deepcopy(reordered)
    changed_observations = uniquely_changed["facts"]["us-gaap"][source_name]["units"][original.unit]
    changed_raw = next(row for row in changed_observations if row["start"] == "2024-01-01")
    changed_raw["val"] = str(Decimal(str(changed_raw["val"])) + Decimal("1"))
    changed_at = datetime(2026, 2, 26, 12, tzinfo=UTC)
    changed_payload = json.dumps(uniquely_changed, sort_keys=True).encode()
    changed_source = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=target.security.company.cik,
        available_at=changed_at,
        payload=changed_payload,
        metadata=dict(source.metadata),
    )
    with patch("django.db.models.fields.timezone.now", return_value=changed_at):
        event = sec_ingestion_module._record_observation_event(
            asset=changed_source,
            kind="sec_companyfacts",
            subject=target.security.company.cik,
            digest=changed_source.sha256,
            observed_at=changed_at,
        )
        changed_created, _changed_reused = sec_ingestion_module._normalize_companyfacts(
            company=target.security.company,
            payload=changed_payload,
            source_asset=changed_source,
            observed_at=changed_at,
            filing_records=_filing_records_for_companyfacts_source(
                store,
                target,
                changed_source,
            ),
            config=load_sec_fundamentals_config(),
            store=store,
        )

    assert changed_created == 1
    same_accession = FundamentalFact.objects.filter(
        company=target.security.company,
        concept="net_income",
        accession=original.accession,
    )
    assert same_accession.count() == 3
    unchanged_lineage = same_accession.filter(period_start=date(2025, 1, 1))
    changed_lineage = same_accession.filter(period_start=date(2024, 1, 1))
    assert list(unchanged_lineage.values_list("source_revision", flat=True)) == [1]
    assert list(
        changed_lineage.order_by("source_revision").values_list(
            "source_revision",
            flat=True,
        )
    ) == [1, 2]
    correction = changed_lineage.get(source_revision=2)
    assert correction.available_at == changed_at
    assert event.source_asset_id == correction.source_asset_id


@pytest.mark.django_db
def test_ambiguous_same_accession_lineage_fails_before_fact_or_event_write(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store, include_fcf=False)
    target = listings[0]
    original = FundamentalFact.objects.get(
        company=target.security.company,
        concept="diluted_eps",
        fiscal_period="FY",
        period_end=date(2025, 12, 31),
    )
    source = original.source_asset
    document = json.loads(store.read_bytes(source.relative_path))
    source_name = original.source_concept.split(":", 1)[1]
    observations = document["facts"]["us-gaap"][source_name]["units"][original.unit]
    second = dict(next(row for row in observations if row["accn"] == original.accession))
    second["start"] = "2024-01-01"
    observations.append(second)
    expanded_at = datetime(2026, 2, 24, 12, tzinfo=UTC)
    expanded_payload = json.dumps(document, sort_keys=True).encode()
    expanded_source = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=target.security.company.cik,
        available_at=expanded_at,
        payload=expanded_payload,
        metadata=dict(source.metadata),
    )
    filing_records = _filing_records_for_companyfacts_source(
        store,
        target,
        expanded_source,
    )
    with patch("django.db.models.fields.timezone.now", return_value=expanded_at):
        created, _reused = sec_ingestion_module._normalize_companyfacts(
            company=target.security.company,
            payload=expanded_payload,
            source_asset=expanded_source,
            observed_at=expanded_at,
            filing_records=filing_records,
            config=load_sec_fundamentals_config(),
            store=store,
        )
    assert created == 1

    ambiguous = deepcopy(document)
    old_rows = ambiguous["facts"]["us-gaap"][source_name]["units"].pop(original.unit)
    for index, row in enumerate(old_rows, start=1):
        row["start"] = f"202{index}-01-01"
    ambiguous["facts"]["us-gaap"][source_name]["units"]["USD-per-shares"] = old_rows
    ambiguous_at = datetime(2026, 2, 25, 12, tzinfo=UTC)
    ambiguous_bytes = json.dumps(ambiguous, sort_keys=True).encode()
    metadata = dict(source.metadata)
    payload = FundamentalSourcePayload(
        provider="sec",
        subject=target.security.company.cik,
        content=ambiguous_bytes,
        content_type="application/json",
        retrieved_at=ambiguous_at,
        source_url="https://example.invalid/companyfacts.json",
    )
    before = (
        FundamentalFact.objects.count(),
        SourceObservationEvent.objects.filter(kind="sec_companyfacts").count(),
        DataAsset.objects.filter(kind="sec_companyfacts").count(),
    )

    with pytest.raises(SecDerivationError, match="ambiguous lineage"):
        sec_ingestion_module._preflight_companyfacts_lineage(
            company=target.security.company,
            payload=payload,
            metadata=metadata,
            filing_records=filing_records,
            config=load_sec_fundamentals_config(),
            store=store,
        )

    assert (
        FundamentalFact.objects.count(),
        SourceObservationEvent.objects.filter(kind="sec_companyfacts").count(),
        DataAsset.objects.filter(kind="sec_companyfacts").count(),
    ) == before


@pytest.mark.django_db
def test_v4_retains_late_original_companyfacts_with_honest_retrieval_time(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    late_retrieval = datetime(2026, 3, 2, 12, tzinfo=UTC)
    decision_time = late_retrieval + timedelta(days=1)
    target = _listing("LATERESEARCH")
    target_price, target_asset = _company_evidence(
        store,
        target,
        sic="3571",
        companyfacts_at=late_retrieval,
    )
    listings = [target]
    prices = {str(target.pk): target_price}
    assets = {str(target.pk): target_asset}
    for index in range(3):
        peer = _listing(f"LATEP{index}")
        price, asset = _company_evidence(
            store,
            peer,
            sic="3571",
            scale=1.1 + index * 0.1,
        )
        listings.append(peer)
        prices[str(peer.pk)] = price
        assets[str(peer.pk)] = asset
    source = DataAsset.objects.get(
        provider="sec",
        kind="sec_companyfacts",
        subject=target.security.company.cik,
    )
    with patch("django.db.models.fields.timezone.now", return_value=late_retrieval):
        event = sec_ingestion_module._record_observation_event(
            asset=source,
            kind="sec_companyfacts",
            subject=target.security.company.cik,
            digest=source.sha256,
            observed_at=late_retrieval,
        )
    historical_cutoff = datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC)

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(decision_time, store),
        data_cutoff=historical_cutoff,
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]
    calculation = pair["3y"].calculation
    authority = calculation["evidence_selection"]["target"]["raw_fcf_authority"]
    source_entry = next(
        item for item in authority["sources"] if item["companyfacts_asset"]["id"] == str(source.pk)
    )

    assert source.retrieved_at > historical_cutoff
    assert source_entry["companyfacts_asset"]["retrieved_at"] == late_retrieval.isoformat()
    assert source_entry["latest_visible_observation_event"]["id"] == str(event.pk)
    assert source_entry["visible_observation_events"] == [
        source_entry["latest_visible_observation_event"]
    ]
    assert all(
        item["is_same_accession_correction"] is False
        and item["raw_observation_available_at"] <= historical_cutoff.isoformat()
        for item in authority["derivations"]
    )
    assert str(source.pk) in {item["id"] for item in calculation["source_manifest"]}


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("concept", "mutation", "expected_created"),
    [
        ("diluted_eps", "unsupported_unit", 0),
        ("weighted_average_diluted_shares", "unsupported_unit", 0),
        ("diluted_eps", "outside_window", 1),
    ],
)
def test_v4_latest_raw_lineage_state_supersedes_stale_normalized_fact(
    tmp_path: Path,
    concept: str,
    mutation: str,
    expected_created: int,
) -> None:
    store = AssetStore(tmp_path)
    listings, prices, assets = _cohort(store, include_fcf=False)
    target = listings[0]
    original = FundamentalFact.objects.get(
        company=target.security.company,
        concept=concept,
        fiscal_period="FY",
        period_end=date(2025, 12, 31),
    )
    source = original.source_asset
    document = json.loads(store.read_bytes(source.relative_path))
    source_name = original.source_concept.split(":", 1)[1]
    units = document["facts"]["us-gaap"][source_name]["units"]
    observations = units[original.unit]
    corrected_raw = next(row for row in observations if row["accn"] == original.accession)
    if mutation == "unsupported_unit":
        units[original.unit] = [row for row in observations if row["accn"] != original.accession]
        units["EUR"] = [corrected_raw]
    else:
        corrected_raw["start"] = "2018-01-01"
        corrected_raw["end"] = "2018-12-31"
    correction_boundary = datetime(2026, 2, 26, 12, tzinfo=UTC)
    payload = json.dumps(document, sort_keys=True).encode()
    correction_source = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=target.security.company.cik,
        available_at=correction_boundary,
        payload=payload,
        metadata=dict(source.metadata),
    )
    with patch("django.db.models.fields.timezone.now", return_value=correction_boundary):
        event = sec_ingestion_module._record_observation_event(
            asset=correction_source,
            kind="sec_companyfacts",
            subject=target.security.company.cik,
            digest=correction_source.sha256,
            observed_at=correction_boundary,
        )
        created, _reused = sec_ingestion_module._normalize_companyfacts(
            company=target.security.company,
            payload=payload,
            source_asset=correction_source,
            observed_at=correction_boundary,
            filing_records=_filing_records_for_companyfacts_source(
                store,
                target,
                correction_source,
            ),
            config=load_sec_fundamentals_config(),
            store=store,
        )

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]
    calculation = pair["3y"].calculation
    catalog = calculation["evidence_catalog"]
    authority = calculation["evidence_selection"]["target"]["raw_fcf_authority"]

    assert created == expected_created
    assert calculation["insufficiency_code"] == "annual_tuple_incomplete"
    assert pair["3y"].scenario.base is None
    assert str(original.pk) not in catalog["selected_fact_ids"]
    assert str(original.pk) in catalog["assessed_fact_ids"]
    assert str(correction_source.pk) in {
        entry["companyfacts_asset"]["id"] for entry in authority["sources"]
    }
    assert str(correction_source.pk) in {item["id"] for item in calculation["source_manifest"]}
    source_entry = next(
        entry
        for entry in authority["sources"]
        if entry["companyfacts_asset"]["id"] == str(correction_source.pk)
    )
    assert source_entry["latest_visible_observation_event"]["id"] == str(event.pk)
    if mutation == "unsupported_unit":
        assert not FundamentalFact.objects.filter(
            company=target.security.company,
            concept=concept,
            accession=original.accession,
            source_revision=2,
        ).exists()
    else:
        outside = FundamentalFact.objects.get(
            company=target.security.company,
            concept=concept,
            accession=original.accession,
            source_revision=2,
        )
        assert outside.period_end == date(2018, 12, 31)
        assert str(outside.pk) not in {item["id"] for item in catalog["facts"]}


@pytest.mark.django_db
def test_v4_distinct_unsupported_same_accession_does_not_supersede_valid_lineage(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings, prices, assets = _cohort(store, include_fcf=False)
    target = listings[0]
    original = FundamentalFact.objects.get(
        company=target.security.company,
        concept="diluted_eps",
        fiscal_period="FY",
        period_end=date(2025, 12, 31),
    )
    source = original.source_asset
    document = json.loads(store.read_bytes(source.relative_path))
    source_name = original.source_concept.split(":", 1)[1]
    units = document["facts"]["us-gaap"][source_name]["units"]
    distinct = dict(next(row for row in units[original.unit] if row["accn"] == original.accession))
    distinct["start"] = "2024-01-01"
    distinct["val"] = str(original.value + Decimal("7"))
    units["EUR"] = [distinct]
    observed_at = datetime(2026, 2, 26, 12, tzinfo=UTC)
    payload = json.dumps(document, sort_keys=True).encode()
    expanded_source = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=target.security.company.cik,
        available_at=observed_at,
        payload=payload,
        metadata=dict(source.metadata),
    )
    with patch("django.db.models.fields.timezone.now", return_value=observed_at):
        created, _reused = sec_ingestion_module._normalize_companyfacts(
            company=target.security.company,
            payload=payload,
            source_asset=expanded_source,
            observed_at=observed_at,
            filing_records=_filing_records_for_companyfacts_source(
                store,
                target,
                expanded_source,
            ),
            config=load_sec_fundamentals_config(),
            store=store,
        )

    pair = build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]
    catalog = pair["3y"].calculation["evidence_catalog"]

    assert created == 0
    assert pair["3y"].calculation["insufficiency_code"] is None
    assert pair["3y"].scenario.base is not None
    assert str(original.pk) in catalog["selected_fact_ids"]
    assert str(expanded_source.pk) in {
        item["id"] for item in pair["3y"].calculation["source_manifest"]
    }


@pytest.mark.django_db
@pytest.mark.parametrize(
    "mutation",
    [
        "event_missing",
        "event_wrong_provider",
        "event_wrong_kind",
        "event_wrong_subject",
        "event_wrong_asset",
        "event_wrong_digest",
        "event_after_cutoff",
        "event_recorded_late",
        "correction_basis",
        "correction_flag_missing",
        "rebound_flag_unjustified",
        "revision_gap",
        "changed_period_revision_one",
    ],
)
def test_v4_invalid_correction_event_chain_aborts_atomically(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    _append_fact_correction(store, listings[0], mutation=mutation)
    snapshot = _snapshot(listings)

    with pytest.raises(
        ValueError,
        match=r"^Long-v4 SEC raw evidence authority is incompatible$",
    ):
        _run_v4(snapshot, store)

    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"long_forecast_requested": None}, "long_forecast_requested=True"),
        ({"long_forecast_requested": False}, "long_forecast_requested=True"),
        ({"issued_on_time": None}, "literal issued_on_time=False"),
        ({"issued_on_time": True}, "literal issued_on_time=False"),
        ({"provider": "synthetic_demo"}, "provider='twelve_data'"),
    ],
)
def test_v4_invalid_requests_fail_before_any_output(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    listing = _listing("GATE")
    snapshot = _snapshot([listing])
    kwargs: dict[str, object] = {
        "universe_snapshot": snapshot,
        "decision_time": DECISION_TIME,
        "target_date": TARGET_DATE,
        "issued_on_time": False,
        "provider": "twelve_data",
        "store": AssetStore(tmp_path),
        "config_path": default_us_scoring_config_path(),
        "long_forecast_config_path": V4_PATH,
        "long_forecast_requested": True,
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        analyze_snapshot(**kwargs)  # type: ignore[arg-type]
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert DataAsset.objects.count() == 0


@pytest.mark.django_db
def test_v4_rejects_wrong_scoring_snapshot_and_listing_before_output(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listing = _listing("BOUNDARY")
    snapshot = _snapshot([listing])
    common = {
        "universe_snapshot": snapshot,
        "decision_time": DECISION_TIME,
        "target_date": TARGET_DATE,
        "issued_on_time": False,
        "provider": "twelve_data",
        "store": store,
        "long_forecast_config_path": V4_PATH,
        "long_forecast_requested": True,
    }
    with pytest.raises(ValueError, match="us-price-baseline-v2"):
        analyze_snapshot(
            **common,
            config_path=Path("config/scoring/default-v1.yml"),
        )
    snapshot.grade = UniverseSnapshot.Grade.OBSERVED
    snapshot.save(update_fields=["grade"])
    with pytest.raises(ValueError, match="research-grade"):
        analyze_snapshot(
            **common,
            config_path=default_us_scoring_config_path(),
        )
    snapshot.grade = UniverseSnapshot.Grade.RESEARCH
    snapshot.save(update_fields=["grade"])
    listing.currency = "EUR"
    listing.save(update_fields=["currency"])
    with pytest.raises(ValueError, match="active US/USD"):
        analyze_snapshot(
            **common,
            config_path=default_us_scoring_config_path(),
        )
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0
    assert DataAsset.objects.count() == 0


@pytest.mark.django_db
def test_v4_end_to_end_persists_pair_and_renders_research_only_language(
    tmp_path: Path,
    client,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    snapshot = _snapshot(listings)

    results = analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        issued_on_time=False,
        provider="twelve_data",
        store=store,
        config_path=default_us_scoring_config_path(),
        long_forecast_config_path=V4_PATH,
        long_forecast_requested=True,
    )
    target = next(item for item in results if item.analysis.listing_id == listings[0].pk)
    predictions = Prediction.objects.filter(
        analysis=target.analysis,
        method_version="us-sec-long-v4",
    ).order_by("horizon")
    assert predictions.count() == 2
    assert {prediction.horizon for prediction in predictions} == {"3y", "5y"}
    assert all(
        prediction.confidence == 0
        and prediction.probability_positive is None
        and prediction.confidence_status == "not_estimated_uncalibrated"
        and prediction.evidence_grade == UniverseSnapshot.Grade.RESEARCH
        and prediction.issued_on_time is False
        for prediction in predictions
    )
    target.analysis.refresh_from_db()
    canonical_price = predictions[0].calculation["target_price"]
    expected_price_source = {
        "listing_id": canonical_price["listing_id"],
        "provider": canonical_price["provider"],
        "subject": canonical_price["subject"],
        "exchange_mic": canonical_price["exchange_mic"],
        "session_date": canonical_price["session_date"],
        "value": canonical_price["value"],
        "ledger_value": canonical_price["ledger_value"],
        "valuation_value": canonical_price["valuation_value"],
        "valuation_source": canonical_price["valuation_source"],
        "native_price": canonical_price["native_price"],
        "currency": canonical_price["currency"],
        "normalized_asset_id": canonical_price["normalized_asset"]["id"],
        "normalized_asset_sha256": canonical_price["normalized_asset"]["sha256"],
        "raw_asset_id": canonical_price["raw_asset"]["id"],
        "raw_asset_sha256": canonical_price["raw_asset"]["sha256"],
        "asset_id": canonical_price["normalized_asset"]["id"],
        "asset_sha256": canonical_price["normalized_asset"]["sha256"],
    }
    assert target.analysis.current_price == Decimal(canonical_price["value"])
    assert target.analysis.data_quality["price_source"] == expected_price_source
    assert {
        (
            prediction.price_at_prediction,
            prediction.price_provider,
            prediction.price_subject,
            prediction.calculation["target_price"]["value"],
        )
        for prediction in predictions
    } == {
        (
            Decimal(canonical_price["value"]),
            canonical_price["provider"],
            canonical_price["subject"],
            canonical_price["value"],
        )
    }
    assert [item["id"] for item in predictions[0].source_assets] == [
        item["id"] for item in predictions[0].calculation["source_manifest"]
    ]
    assert target.analysis.three_year_forecast_scenario["schema_version"] == 2
    assert (
        target.analysis.three_year_forecast_scenario["split_basis"]["assessment_status"]
        == "unverified"
    )
    assert target.analysis.three_year_forecast_scenario["split_basis"]["assessed_through"]

    user_model = get_user_model()
    user = user_model.objects.create_user(username="v4-owner", password="synthetic-pass")
    client.force_login(user)
    detail = client.get(reverse("stock-detail", kwargs={"listing_id": target.analysis.listing_id}))
    ledger = client.get(reverse("predictions"))
    status = client.get(reverse("status"))
    opportunities = client.get(reverse("opportunities"))
    assert (
        detail.status_code == ledger.status_code == status.status_code == opportunities.status_code
    )
    assert status.status_code == 200
    detail_text = detail.content.decode()
    ledger_text = ledger.content.decode()
    status_text = status.content.decode()
    opportunities_text = opportunities.content.decode()
    rendered_scenario = detail.context["analysis"].three_year_forecast_scenario
    assert rendered_scenario["split_basis"]["assessed_through"]
    assert "Research-only / unactivated" in detail_text
    assert "Not estimated / unavailable" in detail_text
    assert "Reported GAAP" in detail_text
    assert "Diluted-share basis:" in detail_text
    assert "Unverified;" in detail_text
    assert "assessed through" in detail_text
    assert "Research-only / unactivated" in ledger_text
    assert "Diluted-share basis:" in ledger_text
    assert "assessed through" in ledger_text
    assert "0%" not in _v4_ledger_rows(ledger_text)
    status_cells = [
        text for text in _tag_texts(status_text, "td") if "Research-only / unactivated." in text
    ]
    opportunity_values = [
        text
        for text in _tag_texts(opportunities_text, "dd")
        if "Research-only / unactivated." in text
    ]
    assert len(status_cells) == len(opportunity_values) == len(listings) * 2
    for value in (*status_cells, *opportunity_values):
        assert "Research-only / unactivated." in value
        assert (
            "Confidence: Not estimated / unavailable. Positive-return probability: unavailable."
        ) in value
        assert "Forecast unavailable" not in value
        assert "Confidence: 0%" not in value

    analyses = list(
        StockAnalysis.objects.filter(run=target.analysis.run).order_by("listing__ticker")
    )
    valid = target.analysis
    mismatch_method = next(item for item in analyses if item.pk != valid.pk)
    mismatch_schema = next(
        item for item in analyses if item.pk not in {valid.pk, mismatch_method.pk}
    )
    blocked = next(
        item
        for item in analyses
        if item.pk not in {valid.pk, mismatch_method.pk, mismatch_schema.pk}
    )
    method_payload = deepcopy(mismatch_method.forecast_scenarios)
    schema_payload = deepcopy(mismatch_schema.forecast_scenarios)
    for horizon in ("3y", "5y"):
        method_payload["horizons"][horizon]["method_version"] = "legacy-long-method"
        schema_payload["horizons"][horizon]["schema_version"] = 1
    mismatch_method.forecast_scenarios = method_payload
    mismatch_method.save(update_fields=["forecast_scenarios"])
    mismatch_schema.forecast_scenarios = schema_payload
    mismatch_schema.save(update_fields=["forecast_scenarios"])
    LatestMarketData.objects.filter(listing=blocked.listing).update(close=Decimal("5.000000"))
    recommendations_before = dict(
        StockAnalysis.objects.filter(run=target.analysis.run).values_list(
            "listing_id",
            "recommendation",
        )
    )

    negative_status = client.get(reverse("status"))
    negative_opportunities = client.get(reverse("opportunities"))
    assert negative_status.status_code == negative_opportunities.status_code == 200
    highlighted_before = [
        card["analysis"].listing_id
        for card in negative_opportunities.context["great_opportunities"]
    ]
    repeated_opportunities = client.get(reverse("opportunities"))
    highlighted_after = [
        card["analysis"].listing_id
        for card in repeated_opportunities.context["great_opportunities"]
    ]
    assert highlighted_after == highlighted_before
    recommendations_after = dict(
        StockAnalysis.objects.filter(run=target.analysis.run).values_list(
            "listing_id",
            "recommendation",
        )
    )
    assert recommendations_after == recommendations_before
    valid_status_row = _element_containing(
        negative_status.content.decode(),
        "tr",
        valid.listing.ticker,
    )
    method_status_row = _element_containing(
        negative_status.content.decode(),
        "tr",
        mismatch_method.listing.ticker,
    )
    schema_status_row = _element_containing(
        negative_status.content.decode(),
        "tr",
        mismatch_schema.listing.ticker,
    )
    blocked_status_row = _element_containing(
        negative_status.content.decode(),
        "tr",
        blocked.listing.ticker,
    )
    assert valid_status_row.count("Research-only / unactivated.") == 2
    assert "Research-only / unactivated." not in method_status_row
    assert "Research-only / unactivated." not in schema_status_row
    assert "Forecast unavailable" in blocked_status_row
    assert "Research-only / unactivated." not in blocked_status_row

    opportunity_html = negative_opportunities.content.decode()
    valid_card = _element_containing(
        opportunity_html,
        "article",
        valid.listing.ticker,
        required_class="stock-band-card",
    )
    method_card = _element_containing(
        opportunity_html,
        "article",
        mismatch_method.listing.ticker,
        required_class="stock-band-card",
    )
    schema_card = _element_containing(
        opportunity_html,
        "article",
        mismatch_schema.listing.ticker,
        required_class="stock-band-card",
    )
    blocked_card = _element_containing(
        opportunity_html,
        "article",
        blocked.listing.ticker,
        required_class="stock-band-card",
    )
    assert valid_card.count("Research-only / unactivated.") == 2
    assert "Research-only / unactivated." not in method_card
    assert "Research-only / unactivated." not in schema_card
    assert "Forecast unavailable" in blocked_card
    assert "Research-only / unactivated." not in blocked_card
    assert "Confidence: 0%" not in negative_status.content.decode()
    assert "Confidence: 0%" not in opportunity_html


@pytest.mark.django_db
@pytest.mark.parametrize(
    "mutation",
    [
        "mapping_omitted",
        "mapping_corrupt",
        "non_target_price_removed",
        "raw_owner_removed",
        "fact_removed",
        "classification_removed",
        "duplicate_price_branch",
        "peer_trace_corrupt",
        "peer_order_reversed",
        "peer_owner_crosswired",
        "membership_mismatch",
        "run_snapshot_mismatch",
    ],
)
def test_v4_outcome_rejects_coherent_catalog_mutations_from_persisted_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    results = _run_v4(_snapshot(listings), store)
    target_result = next(item for item in results if item.analysis.listing_id == listings[0].pk)
    prediction = Prediction.objects.select_related(
        "analysis__run__universe_snapshot__universe",
        "listing__security__company",
    ).get(
        analysis=target_result.analysis,
        method_version="us-sec-long-v4",
        horizon=Prediction.Horizon.THREE_YEAR,
    )
    calculation = deepcopy(prediction.calculation)
    catalog = calculation["evidence_catalog"]
    target_id = str(prediction.listing_id)
    non_target_id = next(
        listing_id for listing_id in catalog["cohort_listing_ids"] if listing_id != target_id
    )

    if mutation == "mapping_omitted":
        catalog["sec_mapping_authority"]["cohort_rows"] = catalog["sec_mapping_authority"][
            "cohort_rows"
        ][:-1]
    elif mutation == "mapping_corrupt":
        catalog["sec_mapping_authority"]["cohort_rows"][0]["raw_cik"] = "0000000000"
    elif mutation == "non_target_price_removed":
        catalog["prices"] = [
            item for item in catalog["prices"] if item["listing_id"] != non_target_id
        ]
    elif mutation == "raw_owner_removed":
        catalog["raw_fcf_authority"] = [
            item
            for item in catalog["raw_fcf_authority"]
            if item["owner_listing_id"] != non_target_id
        ]
        calculation["evidence_selection"]["locked_peers"].pop(non_target_id, None)
    elif mutation == "fact_removed":
        removed = next(
            item for item in catalog["facts"] if item["owner_listing_id"] == non_target_id
        )
        fact_id = removed["id"]
        catalog["facts"] = [item for item in catalog["facts"] if item["id"] != fact_id]
        calculation = _remove_v4_fact_references(calculation, fact_id)
        catalog = calculation["evidence_catalog"]
    elif mutation == "classification_removed":
        catalog["classifications"] = [
            item for item in catalog["classifications"] if item["owner_listing_id"] != target_id
        ]
        calculation["target_classification"] = None
    elif mutation == "duplicate_price_branch":
        duplicate = next(item for item in catalog["prices"] if item["listing_id"] == non_target_id)
        catalog["prices"].append(deepcopy(duplicate))
    elif mutation == "peer_trace_corrupt":
        calculation["evidence_selection"]["peer_lock"]["examined_levels"][0]["candidate_count"] += 1
    elif mutation == "peer_order_reversed":
        calculation["locked_peer_candidates"].reverse()
        calculation["peer_set"].reverse()
    elif mutation == "peer_owner_crosswired":
        calculation["locked_peer_candidates"][0]["company_id"] = calculation["target"]["company_id"]
    elif mutation == "membership_mismatch":
        extra = _listing("AUTHX")
        UniverseMembership.objects.create(
            snapshot=prediction.analysis.run.universe_snapshot,
            listing=extra,
            eligible=True,
        )
    else:
        original_run = prediction.analysis.run
        replacement_snapshot = UniverseSnapshot.objects.create(
            universe=original_run.universe_snapshot.universe,
            as_of_date=original_run.target_date - timedelta(days=1),
            grade=UniverseSnapshot.Grade.RESEARCH,
            config_hash="9" * 64,
        )
        AnalysisRun.objects.filter(pk=original_run.pk).update(
            universe_snapshot=replacement_snapshot
        )

    if mutation not in {"membership_mismatch", "run_snapshot_mismatch"}:
        _rebuild_v4_manifest_from_catalog(calculation)
        prediction.source_assets = deepcopy(calculation["source_manifest"])
    prediction.calculation = calculation

    def forbidden_db_authority_io(*_args, **_kwargs):
        raise AssertionError("persisted authority validation must remain DB-only")

    monkeypatch.setattr(long_v4_module, "read_checksummed_bytes", forbidden_db_authority_io)
    monkeypatch.setattr(long_v4_module, "load_long_forecast_v4_config", forbidden_db_authority_io)
    monkeypatch.setattr(long_v4_module, "build_long_forecasts_v4", forbidden_db_authority_io)
    monkeypatch.setattr(AssetStore, "read_bytes", forbidden_db_authority_io)
    calls: list[tuple[str, date]] = []

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date + timedelta(days=1200),
        evaluated_at=datetime(2030, 12, 31, 12, tzinfo=UTC),
        benchmark_subject="BENCH",
        price_loader=lambda subject, through_date: (
            calls.append((subject, through_date))
            or (_ for _ in ()).throw(
                AssertionError("invalid persisted authority must fail before price I/O")
            )
        ),
    )

    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.resolution == "Long-v4 valuation baseline could not be authenticated"
    assert resolved.metadata == {
        "provider": "twelve_data",
        "valuation_baseline": {
            "role": "calculation.target_price.valuation_value",
            "status": "authentication_failed",
            "issue_codes": ["identity_mismatch"],
        },
    }
    assert calls == []


@pytest.mark.django_db
def test_v4_successful_prediction_rejects_null_metric_family_peer_erasure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_successful_v4_peer_erasure_rejected(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        metric_family=None,
    )


@pytest.mark.django_db
def test_v4_successful_prediction_rejects_unknown_metric_family_peer_erasure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_successful_v4_peer_erasure_rejected(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        metric_family="self_declared_family",
    )


@pytest.mark.django_db
def test_v4_successful_prediction_rejects_cross_pair_peer_state_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, _listings, prediction = _persisted_successful_v4_prediction(tmp_path)
    calculation = deepcopy(prediction.calculation)
    calculation["evidence_selection"]["peer_lock"]["selected_sic_prefix"] = "00"
    prediction.calculation = calculation
    _assert_v4_identity_failure_without_downstream_io(
        prediction,
        monkeypatch=monkeypatch,
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("issued_on_time", True, id="issued-on-time-true"),
        pytest.param("config_version", "wrong-scoring-version", id="wrong-config-version"),
        pytest.param("config_version", "", id="blank-config-version"),
        pytest.param("config_hash", "f" * 64, id="wrong-config-hash"),
        pytest.param("config_hash", "", id="blank-config-hash"),
    ],
)
def test_v4_outcome_rejects_mutated_parent_run_admission_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    _store, _listings, prediction = _persisted_successful_v4_prediction(tmp_path)
    AnalysisRun.objects.filter(pk=prediction.analysis.run_id).update(**{field: value})
    refreshed = Prediction.objects.select_related(
        "analysis__run__universe_snapshot",
        "listing__security__company",
    ).get(pk=prediction.pk)

    _assert_v4_identity_failure_without_downstream_io(
        refreshed,
        monkeypatch=monkeypatch,
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "mutation",
    [
        "source_bundle_omitted",
        "visible_event_omitted",
        "companyfacts_missing",
        "current_submissions_missing",
        "history_missing",
        "companyfacts_tampered",
        "current_submissions_tampered",
        "history_tampered",
    ],
)
def test_v4_outcome_physically_replays_complete_raw_sec_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    target = listings[0]
    if mutation == "source_bundle_omitted":
        _append_companyfacts_vintage(
            store,
            target,
            mutation="unchanged",
            available_at=datetime(2026, 2, 26, 12, tzinfo=UTC),
        )
    elif mutation == "visible_event_omitted":
        _append_fact_correction(store, target)
    results = _run_v4(_snapshot(listings), store)
    target_result = next(item for item in results if item.analysis.listing_id == target.pk)
    prediction = Prediction.objects.select_related(
        "analysis__run__universe_snapshot",
        "listing__security__company",
    ).get(
        analysis=target_result.analysis,
        method_version="us-sec-long-v4",
        horizon=Prediction.Horizon.THREE_YEAR,
    )
    calculation = deepcopy(prediction.calculation)
    catalog = calculation["evidence_catalog"]
    target_authority = next(
        item for item in catalog["raw_fcf_authority"] if item["owner_listing_id"] == str(target.pk)
    )

    if mutation == "source_bundle_omitted":
        assert len(target_authority["sources"]) >= 2
        removed_source = target_authority["sources"].pop()
        removed_id = removed_source["companyfacts_asset"]["id"]
        target_authority["derivations"] = [
            item
            for item in target_authority["derivations"]
            if item["companyfacts_asset_id"] != removed_id
        ]
        target_authority["status"] = _raw_authority_status(target_authority["derivations"])
        calculation["evidence_selection"]["target"]["raw_fcf_authority"] = deepcopy(
            target_authority
        )
        _rebuild_v4_manifest_from_catalog(calculation)
        prediction.source_assets = deepcopy(calculation["source_manifest"])
    elif mutation == "visible_event_omitted":
        source_bundle = next(
            item for item in target_authority["sources"] if item["visible_observation_events"]
        )
        source_bundle["visible_observation_events"] = []
        source_bundle["latest_visible_observation_event"] = None
        calculation["evidence_selection"]["target"]["raw_fcf_authority"] = deepcopy(
            target_authority
        )
    else:
        source_bundle = target_authority["sources"][0]
        role = mutation.removesuffix("_missing").removesuffix("_tampered")
        payload = {
            "companyfacts": source_bundle["companyfacts_asset"],
            "current_submissions": source_bundle["current_submissions_asset"],
            "history": source_bundle["history_assets"][0],
        }[role]
        asset = DataAsset.objects.get(pk=payload["id"])
        path = store.resolve(asset.relative_path)
        if mutation.endswith("_missing"):
            path.unlink()
        else:
            path.write_bytes(b"tampered")
    prediction.calculation = calculation

    def forbidden_non_sec_replay(*_args, **_kwargs):
        raise AssertionError("raw SEC replay must not rebuild forecasts or read prices")

    monkeypatch.setattr(long_v4_module, "build_long_forecasts_v4", forbidden_non_sec_replay)
    monkeypatch.setattr(long_v4_module, "_resolve_canonical_price", forbidden_non_sec_replay)
    monkeypatch.setattr(sec, "fetch_companyfacts", forbidden_non_sec_replay)
    monkeypatch.setattr(AssetStore, "read_frame", forbidden_non_sec_replay)
    calls: list[tuple[str, date]] = []

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date + timedelta(days=1200),
        evaluated_at=datetime(2030, 12, 31, 12, tzinfo=UTC),
        benchmark_subject="BENCH",
        price_loader=lambda subject, through_date: (
            calls.append((subject, through_date))
            or (_ for _ in ()).throw(AssertionError("failed raw SEC replay must precede price I/O"))
        ),
        store=store,
    )

    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.resolution == "Long-v4 valuation baseline could not be authenticated"
    assert resolved.metadata == {
        "provider": "twelve_data",
        "valuation_baseline": {
            "role": "calculation.target_price.valuation_value",
            "status": "authentication_failed",
            "issue_codes": ["identity_mismatch"],
        },
    }
    assert str(tmp_path) not in str(resolved.metadata)
    assert str(tmp_path) not in resolved.resolution
    assert calls == []


@pytest.mark.django_db
def test_v4_outcome_replays_normalization_incomplete_assessed_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AssetStore(tmp_path)
    listings, prices, assets = _cohort(store)
    extra_peer = _listing("V4P4")
    extra_price, extra_asset = _company_evidence(
        store,
        extra_peer,
        sic="3571",
        scale=1.4,
    )
    listings.append(extra_peer)
    prices[str(extra_peer.pk)] = extra_price
    assets[str(extra_peer.pk)] = extra_asset
    incomplete_peer = listings[-2]
    incomplete_source = _append_companyfacts_vintage(
        store,
        incomplete_peer,
        mutation="unsupported_fcf_unit",
        available_at=datetime(2026, 2, 26, 12, tzinfo=UTC),
    )
    results = _run_v4(_snapshot(listings), store)
    target_result = next(item for item in results if item.analysis.listing_id == listings[0].pk)
    prediction = Prediction.objects.select_related(
        "analysis__run__universe_snapshot",
        "listing__security__company",
    ).get(
        analysis=target_result.analysis,
        method_version="us-sec-long-v4",
        horizon=Prediction.Horizon.THREE_YEAR,
    )
    assert prediction.base_return is not None
    calculation = deepcopy(prediction.calculation)
    catalog = calculation["evidence_catalog"]
    authority = next(
        item
        for item in catalog["raw_fcf_authority"]
        if item["owner_listing_id"] == str(incomplete_peer.pk)
    )
    omitted = next(
        item
        for item in authority["derivations"]
        if item["companyfacts_asset_id"] == str(incomplete_source.pk)
        and item["normalization_issue"] == "unsupported_unit"
    )
    authority["derivations"].remove(omitted)
    authority["status"] = _raw_authority_status(authority["derivations"])
    calculation["evidence_selection"]["locked_peers"][str(incomplete_peer.pk)][
        "raw_fcf_authority"
    ] = deepcopy(authority)
    _rebuild_v4_manifest_from_catalog(calculation)
    prediction.calculation = calculation
    prediction.source_assets = deepcopy(calculation["source_manifest"])

    def forbidden_non_sec_replay(*_args, **_kwargs):
        raise AssertionError("raw SEC replay must not rebuild forecasts or read prices")

    monkeypatch.setattr(long_v4_module, "build_long_forecasts_v4", forbidden_non_sec_replay)
    monkeypatch.setattr(long_v4_module, "_resolve_canonical_price", forbidden_non_sec_replay)
    monkeypatch.setattr(sec, "fetch_companyfacts", forbidden_non_sec_replay)
    monkeypatch.setattr(AssetStore, "read_frame", forbidden_non_sec_replay)
    calls: list[tuple[str, date]] = []

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date + timedelta(days=1200),
        evaluated_at=datetime(2030, 12, 31, 12, tzinfo=UTC),
        benchmark_subject="BENCH",
        price_loader=lambda subject, through_date: (
            calls.append((subject, through_date))
            or (_ for _ in ()).throw(
                AssertionError("omitted raw SEC derivation must precede price I/O")
            )
        ),
        store=store,
    )

    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.metadata["valuation_baseline"]["issue_codes"] == ["identity_mismatch"]
    assert calls == []


@pytest.mark.django_db
def test_v4_outcome_runs_db_then_raw_replay_before_price_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    results = _run_v4(_snapshot(listings), store)
    target_result = next(item for item in results if item.analysis.listing_id == listings[0].pk)
    prediction = Prediction.objects.select_related(
        "analysis__run__universe_snapshot",
        "listing__security__company",
    ).get(
        analysis=target_result.analysis,
        method_version="us-sec-long-v4",
        horizon=Prediction.Horizon.THREE_YEAR,
    )
    sessions: list[date] = []
    cursor = prediction.target_date + timedelta(days=1)
    while len(sessions) < 756:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor += timedelta(days=1)
    valuation = float(prediction.calculation["target_price"]["valuation_value"])
    stock_frame = pl.DataFrame(
        {
            "date": [prediction.target_date, *sessions],
            "close": [valuation, *([valuation * 1.2] * len(sessions))],
        }
    )
    benchmark_frame = pl.DataFrame(
        {
            "date": [prediction.target_date, sessions[-1]],
            "close": [100.0, 110.0],
        }
    )

    def forbidden_non_sec_replay(*_args, **_kwargs):
        raise AssertionError("raw SEC replay must not rebuild forecasts or read prices")

    real_db_authority = outcomes_module._validate_long_v4_persisted_authority
    real_raw_replay = outcomes_module._validate_long_v4_raw_sec_replay
    sequence: list[str] = []
    assessed_owner_ids: list[tuple[str, ...]] = []

    def traced_db_authority(**kwargs):
        sequence.append("db")
        assessed = real_db_authority(**kwargs)
        assessed_owner_ids.append(tuple(str(owner[0].pk) for owner in assessed))
        return assessed

    def traced_raw_replay(**kwargs):
        sequence.append("raw")
        return real_raw_replay(**kwargs)

    monkeypatch.setattr(
        outcomes_module,
        "_validate_long_v4_persisted_authority",
        traced_db_authority,
    )
    monkeypatch.setattr(
        outcomes_module,
        "_validate_long_v4_raw_sec_replay",
        traced_raw_replay,
    )
    monkeypatch.setattr(long_v4_module, "build_long_forecasts_v4", forbidden_non_sec_replay)
    monkeypatch.setattr(long_v4_module, "_resolve_canonical_price", forbidden_non_sec_replay)
    monkeypatch.setattr(sec, "fetch_companyfacts", forbidden_non_sec_replay)
    monkeypatch.setattr(AssetStore, "read_frame", forbidden_non_sec_replay)

    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=sessions[-1],
        evaluated_at=datetime(2030, 12, 31, 12, tzinfo=UTC),
        benchmark_subject="BENCH",
        price_loader=lambda subject, _through_date: (
            sequence.append("benchmark" if subject == "BENCH" else "stock")
            or (benchmark_frame if subject == "BENCH" else stock_frame)
        ),
        store=store,
    )

    assert resolved.status == PredictionOutcome.Status.MATURED
    assert resolved.actual_return == Decimal("0.2000")
    assert resolved.benchmark_return == Decimal("0.1")
    assert sequence == ["db", "raw", "stock", "benchmark"]
    pair = list(
        Prediction.objects.filter(
            analysis=prediction.analysis,
            method_version="us-sec-long-v4",
        ).order_by("horizon")
    )
    assert len(pair) == 2
    assert {item.analysis.run_id for item in pair} == {prediction.analysis.run_id}
    assert prediction.analysis.run.issued_on_time is False
    assert prediction.analysis.run.config_version == "us-price-baseline-v2"
    assert prediction.analysis.run.config_hash == LONG_V4_SCORING_CONFIG_HASH
    assert assessed_owner_ids == [
        tuple(
            listing_id
            for listing_id in prediction.calculation["evidence_catalog"]["cohort_listing_ids"]
            if listing_id
            in {
                str(prediction.listing_id),
                *prediction.calculation["evidence_catalog"]["locked_peer_listing_ids"],
            }
        )
    ]


@pytest.mark.django_db
def test_v4_price_rounding_boundary_persists_through_service_and_templates(
    tmp_path: Path,
    client,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(
        store,
        split_multiplier=200.0,
        target_close=50.0001 / 200,
    )
    results = _run_v4(_snapshot(listings), store)
    target = next(item for item in results if item.analysis.listing_id == listings[0].pk)
    predictions = list(
        Prediction.objects.filter(
            analysis=target.analysis,
            method_version="us-sec-long-v4",
        ).order_by("horizon")
    )

    assert target.analysis.current_price == Decimal("0.250001")
    assert len(predictions) == 2
    assert {prediction.price_at_prediction for prediction in predictions} == {Decimal("0.250001")}
    assert {prediction.calculation["target_price"]["value"] for prediction in predictions} == {
        "0.250001"
    }
    assert {
        prediction.calculation["target_price"]["ledger_value"] for prediction in predictions
    } == {"0.250001"}
    assert {
        prediction.calculation["target_price"]["valuation_value"] for prediction in predictions
    } == {"0.2500005"}
    assert {
        prediction.calculation["target_price"]["native_price"] for prediction in predictions
    } == {"0.2500005"}
    assert {
        prediction.calculation["formula_inputs"]["current_price"] for prediction in predictions
    } == {0.2500005}
    assert {
        prediction.calculation["formula_inputs"]["current_price_ledger"]
        for prediction in predictions
    } == {"0.250001"}
    assert {
        prediction.calculation["formula_inputs"]["current_price_valuation"]
        for prediction in predictions
    } == {"0.2500005"}
    assert all(
        prediction.calculation["formula_inputs"]["current_per_share_exact"]
        and prediction.calculation["formula_inputs"]["current_multiple_exact"]
        for prediction in predictions
    )
    user = get_user_model().objects.create_user(username="v4-boundary")
    client.force_login(user)
    detail = client.get(reverse("stock-detail", kwargs={"listing_id": listings[0].pk}))
    ledger = client.get(reverse("predictions"))
    assert detail.status_code == 200
    assert ledger.status_code == 200
    assert "0.25" in detail.content.decode()
    assert "<table" in ledger.content.decode()


@pytest.mark.django_db
def test_v4_service_pair_carries_physical_baseline_into_outcome_evaluation(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(
        store,
        split_multiplier=200.0,
        target_close=50.0001 / 200,
    )
    results = _run_v4(_snapshot(listings), store)
    target = next(item for item in results if item.analysis.listing_id == listings[0].pk)
    prediction = Prediction.objects.get(
        analysis=target.analysis,
        method_version="us-sec-long-v4",
        horizon=Prediction.Horizon.THREE_YEAR,
    )
    target_price = prediction.calculation["target_price"]
    valuation_value = Decimal(target_price["valuation_value"])
    ledger_value = Decimal(target_price["ledger_value"])
    assert valuation_value == Decimal("0.2500005")
    assert ledger_value == Decimal("0.250001")

    sessions: list[date] = []
    cursor = TARGET_DATE + timedelta(days=1)
    while len(sessions) < 756:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor += timedelta(days=1)
    terminal_close = valuation_value * Decimal("1.00015")
    frame = pl.DataFrame(
        {
            "date": [TARGET_DATE, *sessions],
            "close": [float(valuation_value), *([float(terminal_close)] * len(sessions))],
            "volume": [1_000_000] * (len(sessions) + 1),
        }
    )
    stored = store.write_frame(f"long-v4/{prediction.pk}-evaluation.parquet", frame)
    evaluation_time = datetime(2030, 1, 1, 12, tzinfo=UTC)
    register_asset(
        provider="twelve_data",
        kind="price_history",
        subject=prediction.price_subject,
        stored=stored,
        retrieved_at=evaluation_time,
        available_at=evaluation_time,
        period_start=TARGET_DATE,
        period_end=sessions[-1],
    )

    result = evaluate_prediction(
        prediction,
        provider="twelve_data",
        evaluation_date=sessions[-1],
        evaluation_time=evaluation_time,
        store=store,
    )
    with localcontext() as context:
        context.prec = 64
        generic_ledger_return = (
            Decimal(str(float(terminal_close))) / ledger_value - Decimal(1)
        ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)

    assert result.outcome.status == PredictionOutcome.Status.MATURED, result.outcome.metadata
    assert result.outcome.actual_return == Decimal("0.0002")
    assert generic_ledger_return == Decimal("0.0001")
    assert result.outcome.actual_return != generic_ledger_return
    assert result.outcome.metadata["valuation_baseline"] == {
        "role": "calculation.target_price.valuation_value",
        "status": "authenticated",
        "valuation_value": "0.2500005",
        "ledger_value": "0.250001",
        "comparison_type": "Decimal(str(target_close)) == valuation_value",
        "target_close": "0.2500005",
        "target_date": TARGET_DATE.isoformat(),
        "return_quantum": "0.0001",
        "rounding": "ROUND_HALF_EVEN",
    }


@pytest.mark.django_db
def test_v4_exact_valuation_price_preserves_split_twins_and_rejects_ledger_substitution(
    tmp_path: Path,
) -> None:
    base = _target_pair(
        AssetStore(tmp_path / "base"),
        ticker="VALBASE",
        target_options={"close_override": 50.0001},
        peer_prefix="VALBP",
    )
    split = _target_pair(
        AssetStore(tmp_path / "split"),
        ticker="VALSPLIT",
        target_options={
            "share_multiplier": 200.0,
            "price_multiplier": 1 / 200,
            "close_override": 50.0001 / 200,
        },
        peer_options={"share_multiplier": 200.0, "price_multiplier": 1 / 200},
        peer_prefix="VALSP",
    )
    split_inputs = split["3y"].calculation["formula_inputs"]
    assert split["3y"].calculation["target_price"]["value"] == "0.250001"
    assert split["3y"].calculation["target_price"]["valuation_value"] == "0.2500005"
    assert split_inputs["current_price"] == 0.2500005
    assert split_inputs["current_price_ledger"] == "0.250001"
    assert split_inputs["current_price_valuation"] == "0.2500005"
    for horizon in ("3y", "5y"):
        for scenario in ("bear", "base", "bull"):
            assert getattr(base[horizon].scenario, scenario) == pytest.approx(
                getattr(split[horizon].scenario, scenario),
                abs=1e-10,
            )

    altered: dict[str, LongForecastV4] = {}
    for horizon, forecast in split.items():
        calculation = deepcopy(forecast.calculation)
        calculation["formula_inputs"]["current_price"] = 0.250001
        calculation["formula_inputs"]["current_price_valuation"] = "0.250001"
        altered[horizon] = replace(forecast, calculation=calculation)
    with pytest.raises(ValueError, match="formula price|arithmetic"):
        validate_long_v4_forecast_pair(
            altered,
            config_hash=LONG_V4_EFFECTIVE_CONFIG_HASH,
        )


@pytest.mark.django_db
def test_v4_exact_multiple_gate_rejects_a_value_ledger_rounds_up_to_three(
    tmp_path: Path,
) -> None:
    pair = _target_pair(
        AssetStore(tmp_path),
        ticker="EXACTGATE",
        target_options={
            "annual_shares": (80.0, 90.0, 100.0, 110.0),
            "close_override": 2.9999999,
        },
        peer_prefix="EXACTGATEP",
    )
    calculation = pair["3y"].calculation
    assert calculation["target_price"]["valuation_value"] == "2.9999999"
    assert calculation["target_price"]["ledger_value"] == "3.000000"
    assert calculation["insufficiency_code"] == "fcf_current_multiple_below_family_minimum"
    assert pair["3y"].scenario.base is None


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("family", "minimum", "shares", "price"),
    [
        ("fcf_per_share", Decimal("3"), 3.0, 1.0),
        (
            "net_income_per_share",
            Decimal("5"),
            6.0,
            float(Decimal("5") / Decimal("6")),
        ),
    ],
)
def test_v4_end_to_end_multiple_floor_is_split_invariant_with_repeating_per_share(
    tmp_path: Path,
    family: str,
    minimum: Decimal,
    shares: float,
    price: float,
) -> None:
    quarter_values = (Decimal("0.25"),) * 4

    def run_twin(label: str, split_multiplier: float) -> list[Prediction]:
        store = AssetStore(tmp_path / label)
        listings: list[Listing] = []
        target = _listing(f"{label}T")
        target_options: dict[str, object] = {
            "annual_shares": (shares,) * 4,
            "share_multiplier": split_multiplier,
            "close_override": price / split_multiplier,
            "include_fcf": family == "fcf_per_share",
            ("quarter_fcf_values" if family == "fcf_per_share" else "quarter_income_values"): (
                quarter_values
            ),
        }
        _company_evidence(store, target, sic="3571", **target_options)
        listings.append(target)
        for index in range(3):
            peer = _listing(f"{label}P{index}")
            peer_options = {
                "annual_shares": (shares,) * 4,
                "share_multiplier": split_multiplier,
                "price_multiplier": 1.0 / split_multiplier,
                "include_fcf": family == "fcf_per_share",
                (
                    "quarter_fcf_values" if family == "fcf_per_share" else "quarter_income_values"
                ): quarter_values,
            }
            _company_evidence(
                store,
                peer,
                sic="3571",
                scale=1.1 + index * 0.1,
                **peer_options,
            )
            listings.append(peer)
        results = _run_v4(_snapshot(listings), store)
        target_result = next(
            result for result in results if result.analysis.listing_id == target.pk
        )
        return list(
            Prediction.objects.filter(
                analysis=target_result.analysis,
                method_version="us-sec-long-v4",
            ).order_by("horizon")
        )

    base = run_twin(f"{family[:3]}BASE", 1.0)
    split = run_twin(f"{family[:3]}SPLIT", 200.0)
    assert len(base) == len(split) == 2
    assert {
        (
            prediction.horizon,
            prediction.evidence_role,
            prediction.evidence_grade,
            prediction.method_version,
            prediction.insufficiency_reason,
        )
        for prediction in base
    } == {
        (
            prediction.horizon,
            prediction.evidence_role,
            prediction.evidence_grade,
            prediction.method_version,
            prediction.insufficiency_reason,
        )
        for prediction in split
    }
    for base_prediction, split_prediction in zip(base, split, strict=True):
        assert base_prediction.base_return is not None
        assert split_prediction.base_return is not None
        assert base_prediction.calculation["metric_family"] == family
        assert split_prediction.calculation["metric_family"] == family
        base_inputs = base_prediction.calculation["formula_inputs"]
        split_inputs = split_prediction.calculation["formula_inputs"]
        assert Decimal(base_inputs["current_multiple_exact"]) >= minimum
        assert Decimal(split_inputs["current_multiple_exact"]) >= minimum
        assert Decimal(base_inputs["current_multiple_exact"]) == Decimal(
            split_inputs["current_multiple_exact"]
        )
        base_recomposed = Decimal(base_inputs["current_per_share_exact"]) * Decimal(str(shares))
        split_recomposed = Decimal(split_inputs["current_per_share_exact"]) * Decimal(
            str(shares * 200)
        )
        assert len(base_inputs["current_per_share_exact"].partition(".")[2]) >= 20
        assert len(split_inputs["current_per_share_exact"].partition(".")[2]) >= 20
        assert float(base_recomposed) == pytest.approx(1.0)
        assert float(split_recomposed) == pytest.approx(1.0)
        for scenario in ("bear", "base", "bull"):
            assert getattr(base_prediction, f"{scenario}_return") == getattr(
                split_prediction,
                f"{scenario}_return",
            )
            base_path = base_prediction.calculation["scenario_paths"][scenario]
            split_path = split_prediction.calculation["scenario_paths"][scenario]
            assert {key: value for key, value in base_path.items() if key != "years"} == {
                key: value for key, value in split_path.items() if key != "years"
            }
            assert [
                {
                    key: value
                    for key, value in year.items()
                    if key not in {"projected_per_share_metric", "projected_price"}
                }
                for year in base_path["years"]
            ] == [
                {
                    key: value
                    for key, value in year.items()
                    if key not in {"projected_per_share_metric", "projected_price"}
                }
                for year in split_path["years"]
            ]


@pytest.mark.django_db
def test_v4_applicable_withholding_renders_incompatible_assessment(
    tmp_path: Path,
    client,
) -> None:
    store = AssetStore(tmp_path)
    target = _listing("WITHHELD")
    target_price, target_asset = _company_evidence(
        store,
        target,
        sic="3571",
        annual_shares=(100.0, 110.0, 130.0, 140.0),
    )
    listings = [target]
    prices = {str(target.pk): target_price}
    assets = {str(target.pk): target_asset}
    for index in range(3):
        peer = _listing(f"WHP{index}")
        price, asset = _company_evidence(store, peer, sic="3571")
        listings.append(peer)
        prices[str(peer.pk)] = price
        assets[str(peer.pk)] = asset
    snapshot = _snapshot(listings)

    analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        issued_on_time=False,
        provider="twelve_data",
        store=store,
        config_path=default_us_scoring_config_path(),
        long_forecast_config_path=V4_PATH,
        long_forecast_requested=True,
    )
    prediction = Prediction.objects.get(
        listing=target,
        horizon=Prediction.Horizon.THREE_YEAR,
        method_version="us-sec-long-v4",
    )
    assert prediction.calculation["split_basis"]["assessment_status"] == (
        "incompatible_or_unverified"
    )
    assert prediction.confidence_status == "not_estimated_insufficient"
    user = get_user_model().objects.create_user(username="v4-withheld")
    client.force_login(user)
    detail = client.get(reverse("stock-detail", kwargs={"listing_id": target.pk}))
    ledger = client.get(reverse("predictions"))
    assert "Diluted-share basis:" in detail.content.decode()
    assert "Incompatible / unverified" in detail.content.decode()
    assert "assessed through" in detail.content.decode()
    assert "Incompatible / unverified" in ledger.content.decode()
    assert "0%" not in _v4_ledger_rows(ledger.content.decode())


@pytest.mark.django_db
def test_v4_second_insert_failure_rolls_back_run_analysis_and_predictions(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    snapshot = _snapshot(listings)
    original_create = Prediction.objects.create
    calls = 0

    def fail_second_v4(**kwargs):
        nonlocal calls
        if kwargs.get("method_version") == "us-sec-long-v4":
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic second insert failure")
        return original_create(**kwargs)

    with patch.object(Prediction.objects, "create", side_effect=fail_second_v4):
        with pytest.raises(RuntimeError, match="synthetic second insert failure"):
            analyze_snapshot(
                universe_snapshot=snapshot,
                decision_time=DECISION_TIME,
                target_date=TARGET_DATE,
                issued_on_time=False,
                provider="twelve_data",
                store=store,
                config_path=default_us_scoring_config_path(),
                long_forecast_config_path=V4_PATH,
                long_forecast_requested=True,
            )
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
def test_v4_source_manifest_tamper_fails_before_any_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    snapshot = _snapshot(listings)
    real_builder = service_module.build_long_forecasts_v4

    def tampered_builder(**kwargs):
        forecasts = real_builder(**kwargs)
        target_id = str(listings[0].pk)
        altered_pair = {}
        for horizon, forecast in forecasts[target_id].items():
            calculation = deepcopy(forecast.calculation)
            calculation["source_manifest"][0]["sha256"] = "f" * 64
            altered_pair[horizon] = replace(forecast, calculation=calculation)
        forecasts[target_id] = altered_pair
        return forecasts

    monkeypatch.setattr(service_module, "build_long_forecasts_v4", tampered_builder)
    with pytest.raises(ValueError, match="authoritative evidence replay"):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            issued_on_time=False,
            provider="twelve_data",
            store=store,
            config_path=default_us_scoring_config_path(),
            long_forecast_config_path=V4_PATH,
            long_forecast_requested=True,
        )
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize(
    "mutation",
    [
        "current_price",
        "data_quality_price",
        "catalog_price",
        "calculation_price",
        "physical_price_asset",
        "generated_kwargs",
        "generated_calculation",
    ],
)
def test_v4_price_identity_mutations_fail_before_either_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    target = listings[0]
    snapshot = _snapshot(listings)

    if mutation in {"current_price", "data_quality_price"}:
        original_create_analysis = service_module._create_stock_analysis

        def altered_analysis(*args, **kwargs):
            analysis = original_create_analysis(*args, **kwargs)
            if analysis.listing_id == target.pk:
                if mutation == "current_price":
                    StockAnalysis.objects.filter(pk=analysis.pk).update(
                        current_price=analysis.current_price + Decimal("1")
                    )
                else:
                    StockAnalysis.objects.filter(pk=analysis.pk).update(
                        data_quality={
                            **analysis.data_quality,
                            "price_source": {
                                **analysis.data_quality["price_source"],
                                "value": "999.000000",
                            },
                        }
                    )
            return analysis

        monkeypatch.setattr(service_module, "_create_stock_analysis", altered_analysis)
    elif mutation in {"catalog_price", "calculation_price", "physical_price_asset"}:
        real_builder = service_module.build_long_forecasts_v4

        def altered_builder(**kwargs):
            forecasts = real_builder(**kwargs)
            target_id = str(target.pk)
            if mutation == "physical_price_asset":
                asset_id = forecasts[target_id]["3y"].calculation["target_price"][
                    "normalized_asset"
                ]["id"]
                asset = DataAsset.objects.get(pk=asset_id)
                store.resolve(asset.relative_path).write_bytes(b"tampered parquet bytes")
                return forecasts
            altered_pair: dict[str, LongForecastV4] = {}
            for horizon, forecast in forecasts[target_id].items():
                calculation = deepcopy(forecast.calculation)
                if mutation == "catalog_price":
                    catalog_price = next(
                        item
                        for item in calculation["evidence_catalog"]["prices"]
                        if item["listing_id"] == target_id
                    )
                    catalog_price["value"] = "999.000000"
                else:
                    calculation["target_price"]["value"] = "999.000000"
                altered_pair[horizon] = replace(forecast, calculation=calculation)
            forecasts[target_id] = altered_pair
            return forecasts

        from dataclasses import replace

        monkeypatch.setattr(service_module, "build_long_forecasts_v4", altered_builder)
    else:
        original_kwargs = service_module._long_v4_prediction_kwargs

        def altered_kwargs(**kwargs):
            result = original_kwargs(**kwargs)
            if mutation == "generated_kwargs":
                result["price_at_prediction"] = result["price_at_prediction"] + Decimal("1")
            else:
                result["calculation"] = {
                    **result["calculation"],
                    "target_date": "1900-01-01",
                }
            return result

        monkeypatch.setattr(service_module, "_long_v4_prediction_kwargs", altered_kwargs)

    with pytest.raises(ValueError):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            issued_on_time=False,
            provider="twelve_data",
            store=store,
            config_path=default_us_scoring_config_path(),
            long_forecast_config_path=V4_PATH,
            long_forecast_requested=True,
        )
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
def test_v4_tied_latest_normalized_prices_fail_closed_and_atomically(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, assets = _cohort(store)
    target = listings[0]
    original = assets[str(target.pk)]
    frame = store.read_frame(original.relative_path)
    conflicting_close = float(frame["close"][-1]) + 1.0
    conflicting_frame = frame.with_columns(
        pl.when(pl.col("date") == TARGET_DATE)
        .then(pl.lit(conflicting_close))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    raw = _asset(
        store,
        provider="twelve_data",
        kind="raw_price_history",
        subject=target.provider_symbol,
        available_at=original.available_at,
        payload=b'{"synthetic":"conflicting tied close"}',
    )
    stored = store.write_frame(f"long-v4/{target.pk}-tied.parquet", conflicting_frame)
    tied = register_asset(
        provider="twelve_data",
        kind="price_history",
        subject=target.provider_symbol,
        stored=stored,
        retrieved_at=original.retrieved_at,
        available_at=original.available_at,
        period_start=original.period_start,
        period_end=original.period_end,
        metadata={
            **original.metadata,
            "raw_asset_id": str(raw.pk),
            "raw_sha256": raw.sha256,
        },
    )
    assert tied.sha256 != original.sha256
    assert conflicting_close != float(frame["close"][-1])

    with pytest.raises(
        ValueError,
        match=r"^Long-v4 canonical normalized price asset is ambiguous$",
    ) as error:
        _run_v4(_snapshot(listings), store)

    assert str(tmp_path) not in str(error.value)
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize(
    "mutation",
    ["omitted_member", "nonmember", "incomplete_catalog", "selected_peer_outside_lock"],
)
def test_v4_full_membership_and_peer_authority_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from dataclasses import replace

    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    outsider = _listing(f"OUT{mutation[:3]}")
    _company_evidence(store, outsider, sic="3571")
    snapshot = _snapshot(listings)
    real_builder = service_module.build_long_forecasts_v4

    def altered_builder(**kwargs):
        forecasts = real_builder(**kwargs)
        target_id = str(listings[0].pk)
        altered_pair: dict[str, LongForecastV4] = {}
        for horizon, forecast in forecasts[target_id].items():
            calculation = deepcopy(forecast.calculation)
            catalog = calculation["evidence_catalog"]
            if mutation == "omitted_member":
                catalog["cohort_listing_ids"].pop()
            elif mutation == "nonmember":
                catalog["cohort_listing_ids"][-1] = str(outsider.pk)
            elif mutation == "incomplete_catalog":
                catalog["prices"].pop()
                calculation["source_manifest"] = calculation["source_manifest"][:-2]
            else:
                calculation["peer_set"][0]["listing_id"] = str(outsider.pk)
            altered_pair[horizon] = replace(forecast, calculation=calculation)
        forecasts[target_id] = altered_pair
        return forecasts

    monkeypatch.setattr(service_module, "build_long_forecasts_v4", altered_builder)
    with pytest.raises(ValueError):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            issued_on_time=False,
            provider="twelve_data",
            store=store,
            config_path=default_us_scoring_config_path(),
            long_forecast_config_path=V4_PATH,
            long_forecast_requested=True,
        )
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize("closure_mutation", ["omitted_member", "nonmember"])
def test_v4_writer_rejects_incomplete_or_nonmember_locked_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    closure_mutation: str,
) -> None:
    from dataclasses import replace

    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    snapshot = _snapshot(listings)
    outsider = _listing(f"LOCKOUT{closure_mutation[:3]}")
    _company_evidence(store, outsider, sic="3571")
    outsider_snapshot = _snapshot([outsider])
    outsider_membership = UniverseMembership.objects.get(
        snapshot=outsider_snapshot,
        listing=outsider,
    )
    original_lock = service_module._lock_long_v4_prediction_authority

    def altered_lock(*args, **kwargs):
        analysis, authority = original_lock(*args, **kwargs)
        if closure_mutation == "omitted_member":
            authority = replace(
                authority,
                memberships=authority.memberships[:-1],
                listings=authority.listings[:-1],
            )
        else:
            authority = replace(
                authority,
                memberships=(*authority.memberships, outsider_membership),
                listings=(*authority.listings, outsider),
            )
        return analysis, authority

    monkeypatch.setattr(
        service_module,
        "_lock_long_v4_prediction_authority",
        altered_lock,
    )
    with pytest.raises(ValueError, match="eligible membership closure"):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            issued_on_time=False,
            provider="twelve_data",
            store=store,
            config_path=default_us_scoring_config_path(),
            long_forecast_config_path=V4_PATH,
            long_forecast_requested=True,
        )
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
def test_v4_writer_rejects_a_stale_classification_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    target = listings[0]
    snapshot = _snapshot(listings)
    real_builder = service_module.build_long_forecasts_v4
    inserted = False

    def altered_builder(**kwargs):
        nonlocal inserted
        forecasts = real_builder(**kwargs)
        if not inserted:
            inserted = True
            available_at = datetime(2026, 2, 25, tzinfo=UTC)
            source = _asset(
                store,
                provider="sec",
                kind="sec_submissions",
                subject=target.security.company.cik,
                available_at=available_at,
            )
            CompanyClassificationObservation.objects.create(
                company=target.security.company,
                provider="sec",
                scheme="sec_sic",
                code="9999",
                description="Later synthetic classification",
                observed_at=available_at,
                available_at=available_at,
                source_asset=source,
            )
        return forecasts

    monkeypatch.setattr(service_module, "build_long_forecasts_v4", altered_builder)
    with pytest.raises(
        ValueError,
        match="peer-lock trace|authoritative evidence replay|SEC raw evidence authority",
    ):
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            issued_on_time=False,
            provider="twelve_data",
            store=store,
            config_path=default_us_scoring_config_path(),
            long_forecast_config_path=V4_PATH,
            long_forecast_requested=True,
        )
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("asset_role", "mutation"),
    SOURCE_TAMPER_CASES,
)
def test_v4_physical_source_tampering_is_rejected_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    asset_role: str,
    mutation: str,
) -> None:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    target = listings[0]
    snapshot = _snapshot(listings)
    real_builder = service_module.build_long_forecasts_v4
    tampered = False

    def altered_builder(**kwargs):
        nonlocal tampered
        forecasts = real_builder(**kwargs)
        if tampered:
            return forecasts
        tampered = True
        catalog = forecasts[str(target.pk)]["3y"].calculation["evidence_catalog"]
        if asset_role == "mapping":
            asset_id = catalog["sec_mapping_authority"]["mapping_asset"]["id"]
        elif asset_role == "normalized_price":
            asset_id = catalog["prices"][0]["normalized_asset"]["id"]
        elif asset_role == "raw_price":
            asset_id = catalog["prices"][0]["raw_asset"]["id"]
        elif asset_role == "sec_source":
            asset_id = catalog["facts"][0]["source_asset_id"]
        elif asset_role == "filing":
            asset_id = catalog["facts"][0]["filing_evidence_asset_id"]
        elif asset_role == "submissions_context":
            asset_id = catalog["facts"][0]["submissions_context_asset_id"]
        else:
            asset_id = catalog["classifications"][0]["source_asset_id"]
        asset = next(
            item
            for item in forecasts[str(target.pk)]["3y"].source_assets
            if str(item.pk) == asset_id
        )
        if mutation == "missing":
            store.resolve(asset.relative_path).unlink()
        elif mutation == "unreadable":
            physical_path = store.resolve(asset.relative_path)
            physical_path.unlink()
            physical_path.mkdir()
        elif mutation == "malformed":
            payload = b"not a parquet document"
            store.resolve(asset.relative_path).write_bytes(payload)
            asset.sha256 = hashlib.sha256(payload).hexdigest()
        elif mutation == "corrupt":
            store.resolve(asset.relative_path).write_bytes(b"corrupt physical evidence")
        elif mutation == "checksum":
            asset.sha256 = "f" * 64
        elif mutation == "wrong_provider":
            asset.provider = "wrong-provider"
        elif mutation == "wrong_kind":
            asset.kind = "wrong-kind"
        elif mutation == "wrong_subject":
            asset.subject = "wrong-subject"
        elif mutation == "late":
            late = DECISION_TIME + timedelta(days=1)
            asset.available_at = late
            asset.retrieved_at = late
        return forecasts

    monkeypatch.setattr(service_module, "build_long_forecasts_v4", altered_builder)
    with pytest.raises(ValueError) as error:
        analyze_snapshot(
            universe_snapshot=snapshot,
            decision_time=DECISION_TIME,
            target_date=TARGET_DATE,
            issued_on_time=False,
            provider="twelve_data",
            store=store,
            config_path=default_us_scoring_config_path(),
            long_forecast_config_path=V4_PATH,
            long_forecast_requested=True,
        )
    assert str(tmp_path) not in str(error.value)
    assert AnalysisRun.objects.count() == 0
    assert StockAnalysis.objects.count() == 0
    assert Prediction.objects.count() == 0


def _append_fact_correction(
    store: AssetStore,
    listing: Listing,
    *,
    mutation: str | None = None,
) -> FundamentalFact:
    original = (
        FundamentalFact.objects.filter(
            company=listing.security.company,
            concept="net_income",
            fiscal_period="FY",
        )
        .select_related("source_asset")
        .order_by("period_end")
        .first()
    )
    assert original is not None
    original_source = original.source_asset
    document = json.loads(store.read_bytes(original_source.relative_path))
    source_name = original.source_concept.split(":", 1)[1]
    observations = document["facts"]["us-gaap"][source_name]["units"][original.unit]
    raw_observation = next(item for item in observations if item["accn"] == original.accession)
    raw_observation["val"] = str(original.value + Decimal("1"))
    if mutation == "changed_period_revision_one":
        raw_observation["start"] = (original.period_start + timedelta(days=1)).isoformat()
    observed_at = datetime(2026, 2, 25, 12, tzinfo=UTC)
    correction_source = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=listing.security.company.cik,
        available_at=observed_at,
        payload=json.dumps(document, sort_keys=True).encode(),
        metadata=dict(original_source.metadata),
    )
    context = DataAsset.objects.get(pk=correction_source.metadata["submissions_asset_id"])
    current = derive_sec_current_submissions(
        store.read_bytes(context.relative_path),
        source_asset=context,
        expected_cik=listing.security.company.cik,
    )
    filing_records = list(current.filings)
    for filename in current.historical_filenames:
        history = DataAsset.objects.get(
            provider="sec",
            kind="sec_submissions_history",
            subject=listing.security.company.cik,
            metadata__filename=filename,
        )
        filing_records.extend(
            derive_sec_historical_submissions(
                store.read_bytes(history.relative_path),
                source_asset=history,
                expected_cik=listing.security.company.cik,
                filename=filename,
                allowed_filenames=current.historical_filenames,
            )
        )
    derivations = derive_sec_companyfacts(
        store.read_bytes(correction_source.relative_path),
        source_asset=correction_source,
        expected_cik=listing.security.company.cik,
        filing_records=tuple(filing_records),
        config=load_sec_fundamentals_config(),
    )
    derived = next(
        item
        for item in derivations
        if item.source_concept == original.source_concept
        and item.accession == original.accession
        and item.value != original.value
        and (
            mutation == "changed_period_revision_one"
            or item.period_identity == original.period_identity
        )
    )
    event_observed_at = (
        datetime(2026, 3, 1, 12, tzinfo=UTC) if mutation == "event_after_cutoff" else observed_at
    )
    if mutation != "event_missing":
        event_kwargs: dict[str, object] = {
            "provider": "wrong" if mutation == "event_wrong_provider" else "sec",
            "kind": "wrong" if mutation == "event_wrong_kind" else "sec_companyfacts",
            "subject": (
                "0000000000" if mutation == "event_wrong_subject" else listing.security.company.cik
            ),
            "content_sha256": (
                "f" * 64 if mutation == "event_wrong_digest" else correction_source.sha256
            ),
            "source_asset": context if mutation == "event_wrong_asset" else correction_source,
            "observed_at": event_observed_at,
        }
        recorded_at = (
            DECISION_TIME + timedelta(days=1)
            if mutation == "event_recorded_late"
            else event_observed_at
        )
        with patch(
            "django.db.models.fields.timezone.now",
            return_value=recorded_at,
        ):
            SourceObservationEvent.objects.create(**event_kwargs)
    filing_asset = DataAsset.objects.get(pk=derived.filing_source_asset_id)
    legacy_backdated = mutation == "changed_period_revision_one"
    quality_flags = (
        list(derived.base_quality_flags)
        if legacy_backdated
        else [*derived.base_quality_flags, "same_accession_correction"]
    )
    if mutation == "correction_flag_missing":
        quality_flags.remove("same_accession_correction")
    if mutation == "rebound_flag_unjustified":
        quality_flags.append("reobserved_unproven_correction")
    with patch(
        "django.db.models.fields.timezone.now",
        return_value=observed_at,
    ):
        fact = FundamentalFact.objects.create(
            company=listing.security.company,
            provider="sec",
            concept=derived.concept,
            taxonomy=derived.taxonomy,
            source_concept=derived.source_concept,
            value=derived.value,
            unit=derived.unit,
            currency=derived.currency,
            period_type=derived.period_type,
            period_identity=derived.period_identity,
            period_start=derived.period_start,
            period_end=derived.period_end,
            fiscal_year=derived.fiscal_year,
            fiscal_period=derived.fiscal_period,
            frame=derived.frame,
            accession=derived.accession,
            filing_form=derived.filing_form,
            filing_date=derived.filing_date,
            filed_at=derived.acceptance_at,
            acceptance_at=derived.acceptance_at,
            available_at=derived.acceptance_at if legacy_backdated else observed_at,
            availability_basis=(
                derived.filing_availability_basis
                if mutation in {"correction_basis", "changed_period_revision_one"}
                else CORRECTION_AVAILABILITY_BASIS
            ),
            is_amendment=derived.is_amendment,
            source_revision=(1 if legacy_backdated else (3 if mutation == "revision_gap" else 2)),
            observation_hash=derived.observation_hash,
            quality_flags=quality_flags,
            source_asset=correction_source,
        )
    FundamentalFactEvidence.objects.create(
        fact=fact,
        role=FundamentalFactEvidence.Role.FILING,
        source_asset=filing_asset,
    )
    return fact


def _filing_records_for_companyfacts_source(
    store: AssetStore,
    listing: Listing,
    source: DataAsset,
) -> tuple[FilingRecord, ...]:
    context = DataAsset.objects.get(pk=source.metadata["submissions_asset_id"])
    current = derive_sec_current_submissions(
        store.read_bytes(context.relative_path),
        source_asset=context,
        expected_cik=listing.security.company.cik,
    )
    records = list(current.filings)
    for filename in current.historical_filenames:
        history = DataAsset.objects.get(
            provider="sec",
            kind="sec_submissions_history",
            subject=listing.security.company.cik,
            metadata__filename=filename,
        )
        records.extend(
            derive_sec_historical_submissions(
                store.read_bytes(history.relative_path),
                source_asset=history,
                expected_cik=listing.security.company.cik,
                filename=filename,
                allowed_filenames=current.historical_filenames,
            )
        )
    return tuple(records)


def _append_companyfacts_vintage(
    store: AssetStore,
    listing: Listing,
    *,
    mutation: str,
    available_at: datetime | None = None,
) -> DataAsset:
    source = (
        DataAsset.objects.filter(
            provider="sec",
            kind="sec_companyfacts",
            subject=listing.security.company.cik,
        )
        .order_by("-available_at", "-retrieved_at")
        .first()
    )
    assert source is not None
    document = json.loads(store.read_bytes(source.relative_path))
    taxonomy = cast(dict[str, object], cast(dict[str, object], document["facts"])["us-gaap"])
    if mutation == "unchanged":
        document["entityName"] = f"{document.get('entityName', '')} refreshed"
    elif mutation == "changed_fcf":
        concept = cast(
            dict[str, object],
            taxonomy[SOURCE_CONCEPTS["operating_cash_flow"].split(":", 1)[1]],
        )
        units = cast(dict[str, list[dict[str, object]]], concept["units"])
        units["USD"][0]["val"] = str(Decimal(str(units["USD"][0]["val"])) + Decimal("1"))
    elif mutation == "unsupported_fcf_unit":
        concept = cast(
            dict[str, object],
            taxonomy[SOURCE_CONCEPTS["operating_cash_flow"].split(":", 1)[1]],
        )
        units = cast(dict[str, list[dict[str, object]]], concept["units"])
        units["EUR"] = units.pop("USD")
    else:
        income = cast(
            dict[str, object],
            taxonomy[SOURCE_CONCEPTS["net_income"].split(":", 1)[1]],
        )
        income_units = cast(dict[str, list[dict[str, object]]], income["units"])
        observation = dict(income_units["USD"][-1])
        concept_name = (
            SOURCE_CONCEPTS["capital_expenditure"].split(":", 1)[1]
            if mutation in {"extra_capex", "raw_form_conflict_capital_expenditure"}
            else SOURCE_CONCEPTS["operating_cash_flow"].split(":", 1)[1]
        )
        if mutation.startswith("raw_form_conflict_"):
            observation["form"] = "8-K"
        if mutation == "outside_window":
            observation.update(
                {
                    "start": "2018-01-01",
                    "end": "2018-12-31",
                    "accn": f"{listing.ticker}-outside-window",
                    "filed": "2019-02-15",
                }
            )
        elif mutation == "after_cutoff":
            observation.update(
                {
                    "accn": f"{listing.ticker}-after-cutoff",
                    "filed": "2026-02-28",
                }
            )
        taxonomy[concept_name] = {"units": {"USD": [observation]}}
    return _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=listing.security.company.cik,
        available_at=available_at or DECISION_TIME - timedelta(minutes=1),
        payload=json.dumps(document, sort_keys=True).encode(),
        metadata=dict(source.metadata),
    )


def _append_reconciled_unsupported_fcf_vintage(
    store: AssetStore,
    listing: Listing,
) -> DataAsset:
    source = (
        DataAsset.objects.filter(
            provider="sec",
            kind="sec_companyfacts",
            subject=listing.security.company.cik,
        )
        .order_by("-available_at", "-retrieved_at")
        .first()
    )
    assert source is not None
    document = json.loads(store.read_bytes(source.relative_path))
    taxonomy = cast(dict[str, object], cast(dict[str, object], document["facts"])["us-gaap"])
    income = cast(
        dict[str, object],
        taxonomy[SOURCE_CONCEPTS["net_income"].split(":", 1)[1]],
    )
    income_units = cast(dict[str, list[dict[str, object]]], income["units"])
    observation = dict(income_units["USD"][-1])
    accession = f"{listing.ticker}-reconciled-8-k"
    observation.update(
        {
            "accn": accession,
            "form": "8-K",
            "filed": "2026-02-15",
        }
    )
    taxonomy[SOURCE_CONCEPTS["operating_cash_flow"].split(":", 1)[1]] = {
        "units": {"USD": [observation]}
    }

    prior_context = DataAsset.objects.get(pk=source.metadata["submissions_asset_id"])
    submissions_document = json.loads(store.read_bytes(prior_context.relative_path))
    recent = cast(
        dict[str, list[object]],
        cast(dict[str, object], submissions_document["filings"])["recent"],
    )
    additions: dict[str, object] = {
        "accessionNumber": accession,
        "filingDate": "2026-02-15",
        "acceptanceDateTime": "2026-02-15T16:30:00-05:00",
        "form": "8-K",
        "reportDate": observation["end"],
        "primaryDocument": "reconciled-8-k.htm",
    }
    for column, value in additions.items():
        recent[column].append(value)
    context = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject=listing.security.company.cik,
        available_at=DECISION_TIME - timedelta(minutes=2),
        payload=json.dumps(submissions_document, sort_keys=True).encode(),
    )
    return _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=listing.security.company.cik,
        available_at=DECISION_TIME - timedelta(minutes=1),
        payload=json.dumps(document, sort_keys=True).encode(),
        metadata={
            "submissions_asset_id": str(context.pk),
            "submissions_sha256": context.sha256,
        },
    )


def _append_nonlatest_non_v4_companyfacts_source(
    store: AssetStore,
    listing: Listing,
) -> tuple[DataAsset, DataAsset]:
    source = (
        DataAsset.objects.filter(
            provider="sec",
            kind="sec_companyfacts",
            subject=listing.security.company.cik,
        )
        .order_by("-available_at", "-retrieved_at", "-pk")
        .first()
    )
    assert source is not None
    original = json.loads(store.read_bytes(source.relative_path))
    middle_document = deepcopy(original)
    taxonomy = cast(
        dict[str, object],
        cast(dict[str, object], middle_document["facts"])["us-gaap"],
    )
    basis = (
        FundamentalFact.objects.filter(
            company=listing.security.company,
            concept="net_income",
            fiscal_period="FY",
        )
        .order_by("-period_end")
        .first()
    )
    assert basis is not None
    filing = FundamentalFactEvidence.objects.get(
        fact=basis,
        role=FundamentalFactEvidence.Role.FILING,
    ).source_asset
    observation = {
        "start": basis.period_start.isoformat() if basis.period_start else None,
        "end": basis.period_end.isoformat(),
        "val": "123",
        "accn": basis.accession,
        "fy": basis.fiscal_year,
        "fp": basis.fiscal_period,
        "form": basis.filing_form,
        "filed": basis.filing_date.isoformat() if basis.filing_date else None,
    }
    taxonomy[SOURCE_CONCEPTS["revenue"].split(":", 1)[1]] = {"units": {"USD": [observation]}}
    taxonomy[SOURCE_CONCEPTS["operating_cash_flow"].split(":", 1)[1]] = {
        "units": {"EUR": [dict(observation)]}
    }
    hidden_source = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=listing.security.company.cik,
        available_at=DECISION_TIME - timedelta(minutes=2),
        payload=json.dumps(middle_document, sort_keys=True).encode(),
        metadata=dict(source.metadata),
    )
    _fact(
        listing,
        hidden_source,
        filing,
        concept="revenue",
        value=Decimal("123"),
        start=cast(date, basis.period_start),
        end=basis.period_end,
        fiscal_period=basis.fiscal_period,
        available_at=cast(datetime, basis.acceptance_at),
        accession=basis.accession,
    )
    latest_source = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=listing.security.company.cik,
        available_at=DECISION_TIME - timedelta(minutes=1),
        payload=json.dumps(original, sort_keys=True).encode(),
        metadata=dict(source.metadata),
    )
    return hidden_source, latest_source


def _append_rejected_fcf_then_no_fcf_vintage(
    store: AssetStore,
    listing: Listing,
    *,
    mutation: str,
) -> tuple[DataAsset, DataAsset]:
    source = (
        DataAsset.objects.filter(
            provider="sec",
            kind="sec_companyfacts",
            subject=listing.security.company.cik,
        )
        .order_by("-available_at", "-retrieved_at", "-pk")
        .first()
    )
    assert source is not None
    original = json.loads(store.read_bytes(source.relative_path))
    rejected_document = deepcopy(original)
    taxonomy = cast(
        dict[str, object],
        cast(dict[str, object], rejected_document["facts"])["us-gaap"],
    )
    income = cast(
        dict[str, object],
        taxonomy[SOURCE_CONCEPTS["net_income"].split(":", 1)[1]],
    )
    income_units = cast(dict[str, list[dict[str, object]]], income["units"])
    observation = dict(income_units["USD"][-1])
    if mutation == "malformed_ocf":
        concept = SOURCE_CONCEPTS["operating_cash_flow"].split(":", 1)[1]
        observation["val"] = "not-a-decimal"
        units = {"USD": [observation]}
    elif mutation == "unsupported_capex_unit":
        concept = SOURCE_CONCEPTS["capital_expenditure"].split(":", 1)[1]
        units = {"EUR": [observation]}
    else:
        raise AssertionError(f"Unhandled rejected FCF mutation {mutation}")
    taxonomy[concept] = {"units": units}
    rejected_source = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=listing.security.company.cik,
        available_at=datetime(2026, 2, 25, 12, tzinfo=UTC),
        payload=json.dumps(rejected_document, sort_keys=True).encode(),
        metadata=dict(source.metadata),
    )
    no_fcf_document = deepcopy(original)
    no_fcf_document["entityName"] = f"{no_fcf_document.get('entityName', '')} refreshed"
    latest_source = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=listing.security.company.cik,
        available_at=datetime(2026, 2, 26, 12, tzinfo=UTC),
        payload=json.dumps(no_fcf_document, sort_keys=True).encode(),
        metadata=dict(source.metadata),
    )
    return rejected_source, latest_source


def _run_v4(snapshot: UniverseSnapshot, store: AssetStore):
    return analyze_snapshot(
        universe_snapshot=snapshot,
        decision_time=DECISION_TIME,
        target_date=TARGET_DATE,
        issued_on_time=False,
        provider="twelve_data",
        store=store,
        config_path=default_us_scoring_config_path(),
        long_forecast_config_path=V4_PATH,
        long_forecast_requested=True,
    )


def _raw_mutator(
    mutation: str,
) -> Callable[[dict[str, object], dict[str, object], dict[str, object]], None]:
    def mutate(
        submissions: dict[str, object],
        history: dict[str, object],
        companyfacts: dict[str, object],
    ) -> None:
        facts = cast(dict[str, object], companyfacts["facts"])
        taxonomy = cast(dict[str, object], facts["us-gaap"])
        source_name = next(iter(taxonomy))
        concept = cast(dict[str, object], taxonomy[source_name])
        units = cast(dict[str, object], concept["units"])
        unit = next(iter(units))
        observations = cast(list[dict[str, object]], units[unit])
        observation = observations[0]
        if mutation == "missing_fact":
            observations.pop(0)
        elif mutation == "accession":
            observation["accn"] = "0000000000-00-000000"
        elif mutation == "value":
            observation["val"] = "999999"
        elif mutation == "taxonomy":
            facts["dei"] = facts.pop("us-gaap")
        elif mutation == "source_concept":
            taxonomy[f"{source_name}Wrong"] = taxonomy.pop(source_name)
        elif mutation == "unit":
            units["EUR"] = units.pop(unit)
        elif mutation == "period_start":
            observation["start"] = "2022-01-02"
        elif mutation == "period_end":
            observation["end"] = "2022-12-30"
        elif mutation == "fiscal_year":
            observation["fy"] = 1900
        elif mutation == "fiscal_period":
            observation["fp"] = "Q1"
        elif mutation == "frame":
            observation["frame"] = "CY1900"
        elif mutation == "form":
            observation["form"] = "10-Q"
        elif mutation == "malformed_value":
            observation["val"] = "not-a-decimal"
        elif mutation == "report_end":
            source_history = cast(dict[str, list[object]], history)
            source_history["reportDate"][0] = "2021-12-31"
        elif mutation in {"filed_date", "filing_missing_accession", "filing_time"}:
            field = {
                "filed_date": "filingDate",
                "filing_missing_accession": "accessionNumber",
                "filing_time": "acceptanceDateTime",
            }[mutation]
            values = cast(dict[str, list[object]], history)[field]
            values[0] = {
                "filed_date": "2026-02-13",
                "filing_missing_accession": "0000000000-00-000000",
                "filing_time": "2026-02-15T00:00:00+00:00",
            }[mutation]
        elif mutation == "filing_conflicting_accession":
            filings = cast(dict[str, object], submissions["filings"])
            recent = cast(dict[str, list[object]], filings["recent"])
            source_history = cast(dict[str, list[object]], history)
            for field in (
                "accessionNumber",
                "filingDate",
                "acceptanceDateTime",
                "form",
                "reportDate",
                "primaryDocument",
            ):
                recent[field].append(source_history[field][0])
            recent["acceptanceDateTime"][-1] = "2026-02-16T00:00:00+00:00"
        else:
            raise AssertionError(f"Unhandled raw mutation {mutation}")

    return mutate


def _context_metadata_mutator(
    mutation: str,
) -> Callable[[dict[str, object]], None] | None:
    if mutation not in {
        "context_uuid_missing",
        "context_uuid_wrong",
        "context_hash_missing",
        "context_hash_wrong",
    }:
        return None

    def mutate(metadata: dict[str, object]) -> None:
        if mutation == "context_uuid_missing":
            metadata.pop("submissions_asset_id")
        elif mutation == "context_uuid_wrong":
            metadata["submissions_asset_id"] = str(uuid4())
        elif mutation == "context_hash_missing":
            metadata.pop("submissions_sha256")
        else:
            metadata["submissions_sha256"] = "f" * 64

    return mutate


def _history_metadata_mutator(
    mutation: str,
) -> Callable[[dict[str, object]], None] | None:
    if mutation not in {
        "history_unlisted",
        "history_metadata_blank",
        "history_metadata_unsafe",
    }:
        return None

    def mutate(metadata: dict[str, object]) -> None:
        filename = cast(str, metadata["filename"])
        if mutation == "history_unlisted":
            metadata["filename"] = filename.replace("-001.json", "-999.json")
        elif mutation == "history_metadata_blank":
            metadata["filename"] = ""
        else:
            metadata["filename"] = "../unsafe.json"

    return mutate


def _context_evidence_mutator(
    mutation: str,
) -> Callable[[dict[str, object], dict[str, object], dict[str, object]], None] | None:
    if mutation not in {
        "context_cik_wrong",
        "companyfacts_cik_wrong",
        "history_unsafe",
        "history_duplicate",
    }:
        return None

    def mutate(
        submissions: dict[str, object],
        _history: dict[str, object],
        companyfacts: dict[str, object],
    ) -> None:
        if mutation == "context_cik_wrong":
            submissions["cik"] = 9999999999
        elif mutation == "companyfacts_cik_wrong":
            companyfacts["cik"] = 9999999999
        else:
            filings = cast(dict[str, object], submissions["filings"])
            files = cast(list[dict[str, object]], filings["files"])
            if mutation == "history_unsafe":
                files[0]["name"] = "../unsafe.json"
            else:
                files.append(dict(files[0]))

    return mutate


def _v4_ledger_rows(content: str) -> str:
    return "\n".join(
        line for line in content.splitlines() if "us-sec-long-v4" in line or "Not estimated" in line
    )


def _tag_texts(document: str, tag: str) -> list[str]:
    blocks = re.findall(
        rf"<{tag}\b[^>]*>(.*?)</{tag}>",
        document,
        flags=re.DOTALL,
    )
    return [" ".join(re.sub(r"<[^>]+>", " ", block).split()) for block in blocks]


def _element_containing(
    document: str,
    tag: str,
    needle: str,
    *,
    required_class: str | None = None,
) -> str:
    opening = (
        rf"<{tag}\b(?=[^>]*class=\"[^\"]*\b{re.escape(required_class)}\b[^\"]*\")[^>]*>"
        if required_class is not None
        else rf"<{tag}\b[^>]*>"
    )
    matches = re.findall(
        rf"{opening}(.*?)</{tag}>",
        document,
        flags=re.DOTALL,
    )
    containing = [match for match in matches if needle in match]
    assert len(containing) == 1
    return containing[0]


def _rebuild_v4_manifest_from_catalog(calculation: dict[str, object]) -> None:
    catalog = cast(dict[str, object], calculation["evidence_catalog"])
    asset_ids = long_v4_module._catalog_manifest_asset_ids(catalog)
    uuids = tuple(UUID(value) for value in asset_ids)
    assets = DataAsset.objects.in_bulk(uuids)
    assert set(assets) == set(uuids)
    calculation["source_manifest"] = [
        long_v4_module._asset_payload(assets[asset_id]) for asset_id in uuids
    ]


def _remove_v4_fact_references(
    calculation: dict[str, object],
    fact_id: str,
) -> dict[str, object]:
    def remove(value: object) -> object:
        if isinstance(value, list):
            return [remove(item) for item in value if item != fact_id]
        if isinstance(value, dict):
            return {key: remove(item) for key, item in value.items()}
        return value

    return cast(dict[str, object], remove(calculation))


def _raw_authority_status(derivations: list[dict[str, object]]) -> str:
    if not derivations:
        return "absent"
    if any(item.get("status") == "normalization_incomplete" for item in derivations):
        return "present_normalization_incomplete"
    return "present_complete"


def _persisted_successful_v4_prediction(
    tmp_path: Path,
) -> tuple[AssetStore, list[Listing], Prediction]:
    store = AssetStore(tmp_path)
    listings, _prices, _assets = _cohort(store)
    results = _run_v4(_snapshot(listings), store)
    target_result = next(item for item in results if item.analysis.listing_id == listings[0].pk)
    prediction = Prediction.objects.select_related(
        "analysis__run__universe_snapshot",
        "listing__security__company",
    ).get(
        analysis=target_result.analysis,
        method_version="us-sec-long-v4",
        horizon=Prediction.Horizon.THREE_YEAR,
    )
    assert all(
        value is not None
        for value in (
            prediction.bear_return,
            prediction.base_return,
            prediction.bull_return,
        )
    )
    return store, listings, prediction


def _assert_successful_v4_peer_erasure_rejected(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_family: object,
) -> None:
    _store, _listings, prediction = _persisted_successful_v4_prediction(tmp_path)
    calculation = deepcopy(prediction.calculation)
    catalog = calculation["evidence_catalog"]
    target_id = str(prediction.listing_id)
    peer_fact_ids = [
        item["id"] for item in catalog["facts"] if item["owner_listing_id"] != target_id
    ]
    for fact_id in peer_fact_ids:
        calculation = _remove_v4_fact_references(calculation, fact_id)
    catalog = calculation["evidence_catalog"]
    calculation["metric_family"] = metric_family
    calculation["evidence_selection"]["peer_lock"] = None
    calculation["evidence_selection"]["locked_peers"] = {}
    calculation["locked_peer_candidates"] = []
    calculation["peer_set"] = []
    catalog["locked_peer_listing_ids"] = []
    catalog["facts"] = [item for item in catalog["facts"] if item["owner_listing_id"] == target_id]
    catalog["classifications"] = [
        item for item in catalog["classifications"] if item["owner_listing_id"] == target_id
    ]
    catalog["raw_fcf_authority"] = [
        item for item in catalog["raw_fcf_authority"] if item["owner_listing_id"] == target_id
    ]
    _rebuild_v4_manifest_from_catalog(calculation)
    prediction.calculation = calculation
    prediction.source_assets = deepcopy(calculation["source_manifest"])
    _assert_v4_identity_failure_without_downstream_io(
        prediction,
        monkeypatch=monkeypatch,
    )


def _assert_v4_identity_failure_without_downstream_io(
    prediction: Prediction,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_after_db_authority(*_args, **_kwargs):
        raise AssertionError(
            "invalid successful peer authority must precede config/store/raw/price IO"
        )

    monkeypatch.setattr(
        outcomes_module,
        "_validate_long_v4_raw_sec_replay",
        forbidden_after_db_authority,
    )
    monkeypatch.setattr(
        long_v4_module,
        "load_long_forecast_v4_config",
        forbidden_after_db_authority,
    )
    monkeypatch.setattr(AssetStore, "read_bytes", forbidden_after_db_authority)
    calls: list[tuple[str, date]] = []
    resolved = resolve_outcome(
        prediction,
        provider="twelve_data",
        evaluation_date=prediction.target_date + timedelta(days=1200),
        evaluated_at=datetime(2030, 12, 31, 12, tzinfo=UTC),
        benchmark_subject="BENCH",
        price_loader=lambda subject, through_date: (
            calls.append((subject, through_date))
            or (_ for _ in ()).throw(
                AssertionError("invalid successful peer authority must precede price IO")
            )
        ),
    )
    assert resolved.status == PredictionOutcome.Status.UNRESOLVED
    assert resolved.resolution == "Long-v4 valuation baseline could not be authenticated"
    assert resolved.metadata == {
        "provider": "twelve_data",
        "valuation_baseline": {
            "role": "calculation.target_price.valuation_value",
            "status": "authentication_failed",
            "issue_codes": ["identity_mismatch"],
        },
    }
    assert calls == []


def _listing(
    ticker: str,
    *,
    security_type: str = Security.SecurityType.COMMON_STOCK,
    exchange_mic: str = "XNAS",
) -> Listing:
    ticker = ticker.upper()
    cik = f"{int.from_bytes(hashlib.sha256(ticker.encode()).digest()[:4], 'big'):010d}"
    company = Company.objects.create(name=f"{ticker} Company", country="US", cik=cik)
    security = Security.objects.create(
        company=company,
        name=f"{ticker} Security",
        security_type=security_type,
    )
    return Listing.objects.create(
        security=security,
        ticker=ticker,
        provider_symbol=ticker,
        exchange_mic=exchange_mic,
        currency="USD",
        region=Region.US,
    )


def _snapshot(listings: list[Listing]) -> UniverseSnapshot:
    universe = Universe.objects.create(
        slug=f"v4-{uuid4().hex}",
        name="Synthetic v4 universe",
        config_version="synthetic-v4",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="4" * 64,
    )
    for listing in listings:
        UniverseMembership.objects.create(snapshot=snapshot, listing=listing)
    return snapshot


def _cohort(
    store: AssetStore,
    *,
    split_multiplier: float = 1.0,
    target_close: float | None = None,
    include_fcf: bool = True,
) -> tuple[list[Listing], dict[str, float], dict[str, DataAsset]]:
    listings: list[Listing] = []
    prices: dict[str, float] = {}
    assets: dict[str, DataAsset] = {}
    for index, ticker in enumerate(("V4T", "V4P1", "V4P2", "V4P3")):
        listing = _listing(ticker)
        price, asset = _company_evidence(
            store,
            listing,
            sic="3571",
            scale=1.0 + index * 0.1,
            share_multiplier=split_multiplier,
            price_multiplier=1.0 / split_multiplier,
            close_override=target_close if index == 0 else None,
            include_fcf=include_fcf,
        )
        listings.append(listing)
        prices[str(listing.pk)] = price
        assets[str(listing.pk)] = asset
    return listings, prices, assets


def _target_pair(
    store: AssetStore,
    *,
    ticker: str,
    target_options: dict[str, object],
    peer_options: dict[str, object] | None = None,
    peer_prefix: str = "TP",
) -> dict[str, LongForecastV4]:
    target = _listing(ticker)
    target_price, target_asset = _company_evidence(
        store,
        target,
        sic="3571",
        **target_options,
    )
    listings = [target]
    prices = {str(target.pk): target_price}
    assets = {str(target.pk): target_asset}
    for index in range(3):
        peer = _listing(f"{peer_prefix}{ticker}{index}")
        peer_price, peer_asset = _company_evidence(
            store,
            peer,
            sic="3571",
            scale=1.1 + index * 0.1,
            **(peer_options or {}),
        )
        listings.append(peer)
        prices[str(peer.pk)] = peer_price
        assets[str(peer.pk)] = peer_asset
    return build_long_forecasts_v4(
        listings=listings,
        current_prices=prices,
        price_assets=assets,
        asof=AsOfData(DECISION_TIME, store),
        data_cutoff=datetime.combine(TARGET_DATE, datetime.max.time(), tzinfo=UTC),
        target_date=TARGET_DATE,
        config=load_long_forecast_v4_config(V4_PATH),
    )[str(target.pk)]


def _company_evidence(
    store: AssetStore,
    listing: Listing,
    *,
    sic: str,
    scale: float = 1.0,
    annual_shares: tuple[float, float, float, float] = (100.0, 104.0, 108.0, 112.0),
    annual_periods: tuple[tuple[date, date], ...] = ANNUAL_PERIODS,
    omit_latest_capex: bool = False,
    latest_fcf: float | None = None,
    add_older_favorable_period: bool = False,
    include_fcf: bool = True,
    annual_fcf_values: tuple[float, float, float, float] | None = None,
    quarter_fcf_values: tuple[Decimal, Decimal, Decimal, Decimal] | None = None,
    quarter_income_values: tuple[Decimal, Decimal, Decimal, Decimal] | None = None,
    share_multiplier: float = 1.0,
    price_multiplier: float = 1.0,
    annual_income_values: tuple[Decimal, Decimal, Decimal, Decimal] | None = None,
    reported_eps_values: tuple[Decimal, Decimal, Decimal, Decimal] | None = None,
    ttm_share_multiplier: float = 1.0,
    evidence_mutator: (
        Callable[[dict[str, object], dict[str, object], dict[str, object]], None] | None
    ) = None,
    context_metadata_mutator: Callable[[dict[str, object]], None] | None = None,
    history_metadata_mutator: Callable[[dict[str, object]], None] | None = None,
    classification_overrides: dict[str, object] | None = None,
    create_classification: bool = True,
    omit_first_filing_link: bool = False,
    close_override: float | None = None,
    companyfacts_at: datetime | None = None,
) -> tuple[float, DataAsset]:
    fact_rows: list[dict[str, object]] = []

    def add_fact(
        *,
        concept: str,
        value: Decimal,
        start: date,
        end: date,
        fiscal_period: str,
        available_at: datetime,
        accession: str,
    ) -> None:
        fact_rows.append(
            {
                "concept": concept,
                "value": value.quantize(Decimal("0.00000001")),
                "start": start,
                "end": end,
                "fiscal_period": fiscal_period,
                "available_at": available_at,
                "accession": accession,
            }
        )

    periods = annual_periods
    shares_values = annual_shares
    if add_older_favorable_period:
        periods = ((date(2021, 1, 1), date(2021, 12, 31)), *periods)
        shares_values = (annual_shares[0], *annual_shares)
    for index, ((start, end), shares) in enumerate(
        zip(periods, shares_values, strict=True),
        start=0,
    ):
        available_at = datetime(end.year + 1, 2, 14, 17, tzinfo=UTC)
        base_index = index - (1 if add_older_favorable_period else 0)
        fcf = (
            annual_fcf_values[max(base_index, 0)]
            if annual_fcf_values is not None
            else 70 + max(base_index, 0) * 8
        ) * scale
        if end.year == 2025 and latest_fcf is not None:
            fcf = latest_fcf * scale
        capex = 10 * scale
        if include_fcf:
            add_fact(
                concept="operating_cash_flow",
                value=Decimal(str(fcf + capex)),
                start=start,
                end=end,
                fiscal_period="FY",
                available_at=available_at,
                accession=f"{listing.ticker}-ocf-{end.isoformat()}",
            )
            if not (omit_latest_capex and end.year == 2025):
                add_fact(
                    concept="capital_expenditure",
                    value=Decimal(str(capex)),
                    start=start,
                    end=end,
                    fiscal_period="FY",
                    available_at=available_at,
                    accession=f"{listing.ticker}-capex-{end.isoformat()}",
                )
        shares_decimal = Decimal(str(shares * scale * share_multiplier))
        income = (
            annual_income_values[max(base_index, 0)]
            if annual_income_values is not None
            else Decimal(str((55 + max(base_index, 0) * 5) * scale))
        )
        add_fact(
            concept="weighted_average_diluted_shares",
            value=shares_decimal,
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=available_at,
            accession=f"{listing.ticker}-shares-{end.isoformat()}",
        )
        add_fact(
            concept="net_income",
            value=income,
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=available_at,
            accession=f"{listing.ticker}-income-{end.isoformat()}",
        )
        add_fact(
            concept="diluted_eps",
            value=(
                reported_eps_values[max(base_index, 0)]
                if reported_eps_values is not None
                else income / shares_decimal
            ),
            start=start,
            end=end,
            fiscal_period="FY",
            available_at=available_at,
            accession=f"{listing.ticker}-eps-{end.isoformat()}",
        )
    latest_shares = Decimal(
        str(annual_shares[-1] * scale * share_multiplier * ttm_share_multiplier)
    )
    for index, (start, end, fiscal_period) in enumerate(QUARTERS, start=1):
        available_at = datetime(2026, 2, 9 + index, 17, tzinfo=UTC)
        quarter_fcf = (
            quarter_fcf_values[index - 1]
            if quarter_fcf_values is not None
            else Decimal(str((25 + index) * scale))
        )
        quarter_capex = Decimal(str(3 * scale))
        quarter_income = (
            quarter_income_values[index - 1]
            if quarter_income_values is not None
            else Decimal(str(16 * scale))
        )
        quarter_values = [
            ("weighted_average_diluted_shares", latest_shares),
            ("net_income", quarter_income),
        ]
        if include_fcf:
            quarter_values = [
                ("operating_cash_flow", quarter_fcf + quarter_capex),
                ("capital_expenditure", quarter_capex),
                *quarter_values,
            ]
        for concept, value in quarter_values:
            add_fact(
                concept=concept,
                value=value,
                start=start,
                end=end,
                fiscal_period=fiscal_period,
                available_at=available_at,
                accession=f"{listing.ticker}-{concept}-{fiscal_period}",
            )

    annual_rows = [row for row in fact_rows if row["fiscal_period"] == "FY"]
    current_rows = [row for row in fact_rows if row["fiscal_period"] != "FY"]
    cik = listing.security.company.cik
    history_filename = f"CIK{cik}-submissions-001.json"

    def filing_columns(rows: list[dict[str, object]]) -> dict[str, list[object]]:
        return {
            "accessionNumber": [row["accession"] for row in rows],
            "filingDate": [cast(datetime, row["available_at"]).date().isoformat() for row in rows],
            "acceptanceDateTime": [cast(datetime, row["available_at"]).isoformat() for row in rows],
            "form": ["10-K" if row["fiscal_period"] == "FY" else "10-Q" for row in rows],
            "reportDate": [cast(date, row["end"]).isoformat() for row in rows],
            "primaryDocument": [f"{row['accession']}.htm" for row in rows],
        }

    submissions_document: dict[str, object] = {
        "cik": int(cik),
        "name": listing.security.company.name,
        "sic": sic,
        "sicDescription": "Synthetic electronic equipment",
        "filings": {
            "recent": filing_columns(current_rows),
            "files": [{"name": history_filename, "filingCount": len(annual_rows)}],
        },
    }
    history_document: dict[str, object] = filing_columns(annual_rows)
    raw_facts: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {"us-gaap": {}}
    for row in fact_rows:
        concept = cast(str, row["concept"])
        source_name = SOURCE_CONCEPTS[concept].split(":", 1)[1]
        unit = UNITS.get(concept, "USD")
        concept_payload = raw_facts["us-gaap"].setdefault(source_name, {"units": {}})
        units = concept_payload["units"]
        units.setdefault(unit, []).append(
            {
                "start": cast(date, row["start"]).isoformat(),
                "end": cast(date, row["end"]).isoformat(),
                "val": str(row["value"]),
                "accn": row["accession"],
                "fy": cast(date, row["end"]).year,
                "fp": row["fiscal_period"],
                "form": "10-K" if row["fiscal_period"] == "FY" else "10-Q",
                "filed": cast(datetime, row["available_at"]).date().isoformat(),
            }
        )
    companyfacts_document: dict[str, object] = {
        "cik": int(cik),
        "entityName": listing.security.company.name,
        "facts": raw_facts,
    }
    if evidence_mutator is not None:
        evidence_mutator(submissions_document, history_document, companyfacts_document)
    submissions_at = datetime(2026, 2, 20, tzinfo=UTC)
    submissions = _asset(
        store,
        provider="sec",
        kind="sec_submissions",
        subject=cik,
        available_at=submissions_at,
        payload=json.dumps(submissions_document, sort_keys=True).encode(),
    )
    history_metadata: dict[str, object] = {"filename": history_filename}
    if history_metadata_mutator is not None:
        history_metadata_mutator(history_metadata)
    history = _asset(
        store,
        provider="sec",
        kind="sec_submissions_history",
        subject=cik,
        available_at=submissions_at,
        payload=json.dumps(history_document, sort_keys=True).encode(),
        metadata=history_metadata,
    )
    companyfacts_at = companyfacts_at or datetime(2026, 2, 20, tzinfo=UTC)
    context_metadata: dict[str, object] = {
        "submissions_asset_id": str(submissions.pk),
        "submissions_sha256": submissions.sha256,
    }
    if context_metadata_mutator is not None:
        context_metadata_mutator(context_metadata)
    companyfacts = _asset(
        store,
        provider="sec",
        kind="sec_companyfacts",
        subject=cik,
        available_at=companyfacts_at,
        payload=json.dumps(companyfacts_document, sort_keys=True).encode(),
        metadata=context_metadata,
    )
    classification_values: dict[str, object] = {
        "code": sic,
        "description": "Synthetic electronic equipment",
        "observed_at": submissions_at,
        "available_at": submissions_at,
        "quality_flags": ["current_snapshot_not_historical"],
        "source_asset": submissions,
    }
    classification_values.update(classification_overrides or {})
    if create_classification:
        with patch(
            "django.db.models.fields.timezone.now",
            return_value=submissions_at,
        ):
            CompanyClassificationObservation.objects.create(
                company=listing.security.company,
                provider="sec",
                scheme="sec_sic",
                **classification_values,
            )
    for row in fact_rows:
        _fact(
            listing,
            companyfacts,
            history if row["fiscal_period"] == "FY" else submissions,
            concept=cast(str, row["concept"]),
            value=cast(Decimal, row["value"]),
            start=cast(date, row["start"]),
            end=cast(date, row["end"]),
            fiscal_period=cast(str, row["fiscal_period"]),
            available_at=cast(datetime, row["available_at"]),
            accession=cast(str, row["accession"]),
            create_evidence=not omit_first_filing_link,
        )
    return _write_price_asset(
        store,
        listing,
        close=(
            close_override if close_override is not None else (50.0 + scale * 10) * price_multiplier
        ),
    )


def _fact(
    listing: Listing,
    companyfacts: DataAsset,
    filing: DataAsset,
    *,
    concept: str,
    value: Decimal,
    start: date,
    end: date,
    fiscal_period: str,
    available_at: datetime,
    accession: str,
    create_evidence: bool = True,
) -> FundamentalFact:
    with patch(
        "django.db.models.fields.timezone.now",
        return_value=available_at,
    ):
        fact = FundamentalFact.objects.create(
            company=listing.security.company,
            provider="sec",
            concept=concept,
            taxonomy="us-gaap",
            source_concept=SOURCE_CONCEPTS[concept],
            value=value,
            unit=UNITS.get(concept, "USD"),
            currency="" if concept == "weighted_average_diluted_shares" else "USD",
            period_type=FundamentalFact.PeriodType.DURATION,
            period_start=start,
            period_end=end,
            fiscal_year=end.year,
            fiscal_period=fiscal_period,
            accession=accession,
            filing_form="10-K" if fiscal_period == "FY" else "10-Q",
            filing_date=available_at.date(),
            filed_at=available_at,
            acceptance_at=available_at,
            available_at=available_at,
            availability_basis="acceptance_datetime",
            quality_flags=["research_reconstruction"],
            source_asset=companyfacts,
        )
    if create_evidence:
        FundamentalFactEvidence.objects.create(
            fact=fact,
            role=FundamentalFactEvidence.Role.FILING,
            source_asset=filing,
        )
    return fact


def _asset(
    store: AssetStore,
    *,
    provider: str,
    kind: str,
    subject: str,
    available_at: datetime,
    payload: bytes | None = None,
    metadata: dict[str, object] | None = None,
) -> DataAsset:
    asset_payload = payload or f"{provider}:{kind}:{subject}:{uuid4().hex}".encode()
    stored = store.write_bytes(
        f"tests/long-v4/evidence/{uuid4().hex}.json",
        asset_payload,
    )
    return register_asset(
        provider=provider,
        kind=kind,
        subject=subject,
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
        metadata=metadata,
    )


def _write_price_asset(
    store: AssetStore,
    listing: Listing,
    *,
    close: float,
) -> tuple[float, DataAsset]:
    sessions: list[date] = []
    cursor = TARGET_DATE
    while len(sessions) < 320:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor -= timedelta(days=1)
    sessions.reverse()
    step = close * 0.0001
    closes = [close - (len(sessions) - index - 1) * step for index in range(len(sessions))]
    frame = pl.DataFrame(
        {
            "date": sessions,
            "close": closes,
            "volume": [2_000_000 + index for index in range(len(sessions))],
        },
        schema_overrides={"date": pl.Date, "close": pl.Float64, "volume": pl.Int64},
    )
    raw_stored = store.write_bytes(
        f"long-v4/{listing.pk}.json",
        (
            f'{{"symbol":"{listing.provider_symbol}","date":"{TARGET_DATE.isoformat()}",'
            f'"close":"{closes[-1]:.6f}"}}'
        ).encode(),
    )
    raw_asset = register_asset(
        provider="twelve_data",
        kind="raw_price_history",
        subject=listing.provider_symbol,
        stored=raw_stored,
        retrieved_at=PRICE_AVAILABLE_AT,
        available_at=PRICE_AVAILABLE_AT,
        period_start=sessions[0],
        period_end=sessions[-1],
    )
    stored = store.write_frame(f"long-v4/{listing.pk}.parquet", frame)
    asset = register_asset(
        provider="twelve_data",
        kind="price_history",
        subject=listing.ticker,
        stored=stored,
        retrieved_at=PRICE_AVAILABLE_AT,
        available_at=PRICE_AVAILABLE_AT,
        period_start=sessions[0],
        period_end=sessions[-1],
        metadata={
            "currency": listing.currency,
            "mic_code": listing.exchange_mic,
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
            "raw_asset_id": str(raw_asset.pk),
            "raw_sha256": raw_asset.sha256,
        },
    )
    LatestMarketData.objects.create(
        listing=listing,
        observed_at=PRICE_AVAILABLE_AT,
        session_date=TARGET_DATE,
        close=Decimal(str(closes[-1])),
        previous_close=Decimal(str(closes[-2])),
        volume=2_000_000 + len(sessions) - 1,
        source_asset=asset,
    )
    return closes[-1], asset

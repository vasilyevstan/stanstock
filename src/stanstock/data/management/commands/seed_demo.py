"""Seed deterministic, entirely synthetic demo data for local development,
demos, and tests.

Creates about 60 obviously synthetic US/European listings (tickers prefixed
``ZZUS``/``ZZEU``, company names prefixed "Synthetic"), one ``research``
(never ``observed``) universe snapshot, 6+ years of business-day OHLCV
Parquet `DataAsset` history per listing (using each listing's real exchange
trading calendar via ``exchange_calendars``), a matching "latest quote" row
per listing, representative point-in-time fundamental facts (including one
deliberate amendment/restatement to exercise vintage handling), ECB-like
weekly FX vintages for EUR/USD and EUR/GBP, and one synthetic
equal-weighted benchmark price series.

No real provider data, real company identities, or real market data is used
anywhere in this command; every identifier is an obviously synthetic
placeholder (see ``config/universes/demo_us_europe_synthetic_v1.yaml`` and
``config/benchmarks/demo_synthetic_balanced_v1.yaml``). The universe
snapshot's ``grade`` is sourced from the universe config (``grade:
research``) rather than hardcoded, so it is unmistakably a research-grade
reconstruction rather than an ``observed`` live universe -- this dataset was
never actually observed trading at any point in time, it was generated.
Every synthetic `DataAsset`/`FundamentalFact` this command writes also
carries an explicit ``"synthetic": true`` marker (in `DataAsset.metadata`
and `FundamentalFact.quality_flags`) in addition to the already-synthetic
``provider="synthetic_demo"`` label, ``ZZ*`` tickers, and "Synthetic "
company-name prefix, so downstream consumers can never mistake this data
for a real provider payload by inspecting a single row in isolation.

Determinism: a fixed RNG seed and a fixed date range
(``START_DATE``..``END_DATE``) mean re-running this command regenerates
byte-identical content for anything not already present.

Idempotency without deletion: this command NEVER deletes or mutates a
previously written immutable source vintage. Every `DataAsset` is looked up
by its natural key (`relative_path`, which is unique) before being written;
if it already exists, the existing row and on-disk bytes are reused
untouched and no new row is created. Every `FundamentalFact` and `FxRate` is
only created alongside a `DataAsset` bundle that did not already exist
(the whole command runs inside one `transaction.atomic()` block, so a
prior run either fully committed -- meaning its facts/rates already exist
wherever its bundle asset exists -- or fully rolled back, so this
skip-if-exists check can never see a bundle asset without its facts).
`Company`/`Security`/`Listing` rows are looked up by natural key
(company name, ISIN, and ticker/exchange_mic/valid_from respectively) and
reused rather than duplicated. The `UniverseSnapshot` is looked up by its
natural key (`universe`, `as_of_date`, `grade`) and reused across reruns,
so its primary key -- and anything referencing it, such as
`research.AnalysisRun.universe_snapshot` -- is stable across reruns.
`UniverseMembership` rows are only created for listings not already linked
to that snapshot. The only rows this command ever updates in place are
`Universe` (a mutable config pointer, not a vintage) and
`LatestMarketData` (a mutable "current state" row -- exactly one per
listing by design -- not an immutable historical vintage).

A consequence of never mutating existing rows: passing a different
``--seed`` on a rerun does not retroactively regenerate already-seeded
listings/prices/facts/rates, because their `relative_path`/natural keys do
not depend on the seed value. To generate a materially different synthetic
dataset, clear the demo rows and `DATA_DIR` assets yourself first, then
rerun with the new seed.

Point-in-time delays are preserved throughout: every `DataAsset`/fact/rate
has an ``available_at`` set after its own economic period ends. Price rows
use end-of-day availability, fundamentals use "filed + a delay", and the
annual synthetic FX bundles conservatively expose their contained
observations only when that immutable bundle itself is available.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import polars as pl
from django.core.management.base import BaseCommand
from django.db import transaction
from exchange_calendars import get_calendar  # type: ignore[import-untyped]

from stanstock.data.assets import AssetStore, StoredAsset, register_asset
from stanstock.data.management.config_loader import (
    config_hash,
    default_benchmark_config_path,
    default_universe_config_path,
    load_yaml_mapping,
)
from stanstock.data.market_state import update_latest_market_data
from stanstock.data.models import (
    Company,
    DataAsset,
    FundamentalFact,
    FxRate,
    Listing,
    Region,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)

SEED = 20260905
START_DATE = date(2020, 1, 2)
END_DATE = date(2026, 9, 4)
PROVIDER = "synthetic_demo"

TICKER_PREFIX_US = "ZZUS"
TICKER_PREFIX_EU = "ZZEU"
COMPANY_NAME_PREFIX = "Synthetic "

US_COUNT = 40
EUROPE_COUNT = 20
US_MICS: tuple[tuple[str, str, str], ...] = (
    ("XNAS", "USD", "US"),
    ("XNYS", "USD", "US"),
)
EUROPE_MICS: tuple[tuple[str, str, str], ...] = (
    ("XLON", "GBP", "GB"),
    ("XPAR", "EUR", "FR"),
    ("XETR", "EUR", "DE"),
    ("XAMS", "EUR", "NL"),
    ("XMIL", "EUR", "IT"),
)
SECTOR_WORDS = (
    "Technology",
    "Industrials",
    "Healthcare",
    "Financials",
    "Consumer",
    "Materials",
    "Energy",
    "Utilities",
)
FUNDAMENTAL_CONCEPTS = ("Revenue", "NetIncomeLoss", "Assets", "StockholdersEquity")
FISCAL_YEARS = (2023, 2024, 2025)
FX_PAIRS: tuple[tuple[str, str], ...] = (("EUR", "USD"), ("EUR", "GBP"))

UTC = UTC
_CALENDAR_SESSION_CACHE: dict[str, list[date]] = {}


@dataclass(frozen=True, slots=True)
class ListingSpec:
    index: int
    ticker: str
    mic: str
    currency: str
    country: str
    region: str
    sector: str


def _build_listing_specs() -> list[ListingSpec]:
    specs: list[ListingSpec] = []
    for i in range(1, US_COUNT + 1):
        mic, currency, country = US_MICS[(i - 1) % len(US_MICS)]
        sector = SECTOR_WORDS[(i - 1) % len(SECTOR_WORDS)]
        specs.append(
            ListingSpec(
                index=i,
                ticker=f"{TICKER_PREFIX_US}{i:03d}",
                mic=mic,
                currency=currency,
                country=country,
                region=Region.US,
                sector=sector,
            )
        )
    for j in range(1, EUROPE_COUNT + 1):
        mic, currency, country = EUROPE_MICS[(j - 1) % len(EUROPE_MICS)]
        sector = SECTOR_WORDS[(j - 1) % len(SECTOR_WORDS)]
        specs.append(
            ListingSpec(
                index=US_COUNT + j,
                ticker=f"{TICKER_PREFIX_EU}{j:03d}",
                mic=mic,
                currency=currency,
                country=country,
                region=Region.EUROPE,
                sector=sector,
            )
        )
    return specs


def _company_name(spec: ListingSpec) -> str:
    return f"{COMPANY_NAME_PREFIX}{spec.sector} Holdings {spec.index:03d}"


def _synthetic_isin(index: int) -> str:
    checksum = (index * 7 + 3) % 10
    return f"ZZ{index:09d}{checksum}"


def _decimal(value: float, places: int) -> Decimal:
    return Decimal(str(round(value, places)))


def _sessions_for(mic: str) -> list[date]:
    cached = _CALENDAR_SESSION_CACHE.get(mic)
    if cached is not None:
        return cached
    calendar = get_calendar(mic)
    sessions = [
        timestamp.date()
        for timestamp in calendar.sessions_in_range(START_DATE.isoformat(), END_DATE.isoformat())
    ]
    _CALENDAR_SESSION_CACHE[mic] = sessions
    return sessions


def _generate_price_frame(sessions: list[date], seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(sessions)
    mu = float(rng.uniform(0.0001, 0.0006))
    sigma = float(rng.uniform(0.010, 0.022))
    log_returns = rng.normal(mu, sigma, size=n)
    start_price = float(rng.uniform(8.0, 220.0))
    close = start_price * np.exp(np.cumsum(log_returns))
    previous_close = np.empty(n)
    previous_close[0] = start_price
    previous_close[1:] = close[:-1]
    open_ = np.clip(previous_close * (1.0 + rng.normal(0.0, sigma * 0.2, size=n)), 0.01, None)
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, sigma * 0.3, size=n)))
    low = np.clip(
        np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, sigma * 0.3, size=n))), 0.01, None
    )
    volume = rng.integers(50_000, 5_000_000, size=n)
    return pl.DataFrame(
        {
            "date": sessions,
            "open": np.round(open_, 4),
            "high": np.round(high, 4),
            "low": np.round(low, 4),
            "close": np.round(close, 4),
            "volume": volume.astype(np.int64),
        }
    )


def _weekly_observation_dates(start: date, end: date) -> list[date]:
    offset = (4 - start.weekday()) % 7  # 4 == Friday
    current = start + timedelta(days=offset)
    dates: list[date] = []
    while current <= end:
        dates.append(current)
        current += timedelta(days=7)
    return dates


def _fx_anchor_date(start: date) -> date:
    """The last Friday on or before ``start``.

    The weekly series' own first Friday can fall *after* the first priced
    session, which would leave the opening day of a mixed-currency run with
    no rate observed on or before it -- and no way to price it without
    looking ahead. Anchoring one observation on or before the first session
    closes that gap using information that predates the whole dataset.
    """
    return start - timedelta(days=(start.weekday() - 4) % 7)


def _synthetic_metadata(**extra: object) -> dict[str, object]:
    """Metadata common to every synthetic `DataAsset` this command writes.

    Every value emitted by ``seed_demo`` is entirely synthetic (no real
    provider was contacted); ``synthetic: True`` makes that unmistakable
    even to a consumer inspecting a single `DataAsset` row in isolation,
    without needing to already know the ``synthetic_demo`` provider-name
    convention.
    """
    metadata: dict[str, object] = {"generator": "seed_demo", "synthetic": True}
    metadata.update(extra)
    return metadata


def _get_or_write_asset(
    *,
    relative_path: str,
    write: Callable[[], StoredAsset],
    provider: str,
    kind: str,
    subject: str,
    retrieved_at: datetime,
    available_at: datetime,
    period_start: date | None = None,
    period_end: date | None = None,
    metadata: dict[str, object] | None = None,
) -> tuple[DataAsset, bool]:
    """Return the existing `DataAsset` for ``relative_path``, or write+register a new one.

    Never overwrites or re-registers an already-present immutable vintage:
    on rerun, a previously written asset (identified by its unique
    ``relative_path``) is reused byte-for-byte and its existing row/PK is
    returned untouched. The returned ``bool`` is ``True`` only when a new
    asset was actually written+registered this call, so callers can skip
    building any dependent rows (facts/rates) that would already exist from
    the asset's original, fully-committed run.
    """
    existing = DataAsset.objects.filter(relative_path=relative_path).first()
    if existing is not None:
        return existing, False
    stored = write()
    asset = register_asset(
        provider=provider,
        kind=kind,
        subject=subject,
        stored=stored,
        retrieved_at=retrieved_at,
        available_at=available_at,
        period_start=period_start,
        period_end=period_end,
        metadata=metadata,
    )
    return asset, True


class Command(BaseCommand):
    help = "Seed deterministic synthetic demo data (listings, universe, prices, fundamentals, FX)."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--seed",
            type=int,
            default=SEED,
            help="Override the deterministic RNG seed (default keeps a fixed dataset)",
        )

    def handle(self, *args: object, **options: object) -> None:
        seed = int(str(options.get("seed") or SEED))
        store = AssetStore()
        universe_config = load_yaml_mapping(default_universe_config_path())
        benchmark_config = load_yaml_mapping(default_benchmark_config_path())

        with transaction.atomic():
            listings = self._create_listings()
            snapshot = self._create_universe_snapshot(universe_config, listings)

            fx_rates = self._build_fx_rates(store)
            if fx_rates:
                FxRate.objects.bulk_create(fx_rates)

            facts: list[FundamentalFact] = []
            for spec, listing in listings:
                sessions = _sessions_for(spec.mic)
                frame = _generate_price_frame(sessions, seed + spec.index)
                self._register_price_and_quote(store, listing, spec.ticker, frame)
                facts.extend(
                    self._build_fundamental_facts(
                        store=store,
                        company=listing.security.company,
                        ticker=spec.ticker,
                        index=spec.index,
                        currency=spec.currency,
                        seed=seed,
                    )
                )
            if facts:
                FundamentalFact.objects.bulk_create(facts)
            benchmark_subject = self._seed_benchmark(store, benchmark_config, seed)

        self.stdout.write(
            self.style.SUCCESS(
                f"Seeded {len(listings)} listings, universe snapshot {snapshot.pk} "
                f"(grade={snapshot.grade}), {len(fx_rates)} new FX observations, "
                f"{len(facts)} new fundamental facts, benchmark subject "
                f"{benchmark_subject!r}. Rows already present from a prior run "
                "were reused untouched."
            )
        )

    def _create_listings(self) -> list[tuple[ListingSpec, Listing]]:
        created: list[tuple[ListingSpec, Listing]] = []
        for spec in _build_listing_specs():
            listing = (
                Listing.objects.select_related("security__company")
                .filter(ticker=spec.ticker, exchange_mic=spec.mic, valid_from=START_DATE)
                .first()
            )
            if listing is None:
                company, _company_created = Company.objects.get_or_create(
                    name=_company_name(spec),
                    defaults={
                        "country": spec.country,
                        "sector": f"Synthetic {spec.sector}",
                        "industry": f"Synthetic {spec.sector} Services",
                    },
                )
                security, _security_created = Security.objects.get_or_create(
                    isin=_synthetic_isin(spec.index),
                    defaults={
                        "company": company,
                        "security_type": Security.SecurityType.COMMON_STOCK,
                        "name": company.name,
                    },
                )
                listing = Listing.objects.create(
                    security=security,
                    ticker=spec.ticker,
                    exchange_mic=spec.mic,
                    currency=spec.currency,
                    region=spec.region,
                    valid_from=START_DATE,
                    is_primary=True,
                    is_active=True,
                )
            created.append((spec, listing))
        return created

    def _create_universe_snapshot(
        self,
        universe_config: dict[str, Any],
        listings: list[tuple[ListingSpec, Listing]],
    ) -> UniverseSnapshot:
        universe, _created = Universe.objects.update_or_create(
            slug=str(universe_config["slug"]),
            defaults={
                "name": str(universe_config["name"]),
                "description": str(universe_config.get("description", "")),
                "config_version": str(universe_config["config_version"]),
            },
        )
        # Sourced from config (`grade: research`) rather than hardcoded, so this
        # is unmistakably a research-grade reconstruction, never `observed` --
        # this snapshot was generated, not observed live at any point in time.
        grade = str(universe_config.get("grade", UniverseSnapshot.Grade.RESEARCH))
        snapshot, _snapshot_created = UniverseSnapshot.objects.get_or_create(
            universe=universe,
            as_of_date=END_DATE,
            grade=grade,
            defaults={"config_hash": config_hash(universe_config)},
        )
        existing_listing_ids = set(
            UniverseMembership.objects.filter(snapshot=snapshot).values_list(
                "listing_id", flat=True
            )
        )
        new_memberships = [
            UniverseMembership(snapshot=snapshot, listing=listing, eligible=True)
            for _spec, listing in listings
            if listing.id not in existing_listing_ids
        ]
        if new_memberships:
            UniverseMembership.objects.bulk_create(new_memberships)
        return snapshot

    def _register_price_and_quote(
        self,
        store: AssetStore,
        listing: Listing,
        ticker: str,
        frame: pl.DataFrame,
    ) -> None:
        last_session: date = frame["date"][-1]
        retrieved_at = datetime.combine(last_session, time(21, 30), tzinfo=UTC)
        price_asset, _price_created = _get_or_write_asset(
            relative_path=f"price_history/{ticker}.parquet",
            write=lambda: store.write_frame(f"price_history/{ticker}.parquet", frame),
            provider=PROVIDER,
            kind="price_history",
            subject=ticker,
            retrieved_at=retrieved_at,
            available_at=retrieved_at,
            period_start=frame["date"][0],
            period_end=last_session,
            metadata=_synthetic_metadata(rows=frame.height, mic=listing.exchange_mic),
        )
        effective_frame = frame if _price_created else store.read_frame(price_asset.relative_path)

        observed_at = price_asset.retrieved_at + timedelta(minutes=5)
        last_close = float(effective_frame["close"][-1])
        previous_close = float(effective_frame["close"][-2]) if effective_frame.height > 1 else None
        last_volume = int(effective_frame["volume"][-1])

        def _write_quote() -> StoredAsset:
            quote_payload = json.dumps(
                {
                    "ticker": ticker,
                    "observed_at": observed_at.isoformat(),
                    "close": last_close,
                    "previous_close": previous_close,
                    "volume": last_volume,
                },
                sort_keys=True,
            ).encode("utf-8")
            return store.write_bytes(f"latest_quote/{ticker}.json", quote_payload)

        quote_asset, _quote_created = _get_or_write_asset(
            relative_path=f"latest_quote/{ticker}.json",
            write=_write_quote,
            provider=PROVIDER,
            kind="latest_quote",
            subject=ticker,
            retrieved_at=observed_at,
            available_at=observed_at,
            period_start=last_session,
            period_end=last_session,
            metadata=_synthetic_metadata(),
        )
        update_latest_market_data(
            listing=listing,
            observed_at=observed_at,
            session_date=last_session,
            close=_decimal(last_close, 4),
            previous_close=(_decimal(previous_close, 4) if previous_close is not None else None),
            volume=last_volume,
            source_asset=quote_asset,
        )
        del price_asset  # referenced only via its relative_path/on-disk bytes

    def _build_fundamental_facts(
        self,
        *,
        store: AssetStore,
        company: Company,
        ticker: str,
        index: int,
        currency: str,
        seed: int,
    ) -> list[FundamentalFact]:
        facts: list[FundamentalFact] = []
        rng = np.random.default_rng(seed + 90_000 + index)
        base_revenue = float(rng.uniform(50_000_000, 5_000_000_000))
        last_fiscal_year = FISCAL_YEARS[-1]
        for year_offset, fiscal_year in enumerate(FISCAL_YEARS):
            period_start = date(fiscal_year, 1, 1)
            period_end = date(fiscal_year, 12, 31)
            filed_at = datetime(fiscal_year + 1, 2, 15, 13, 0, tzinfo=UTC)
            available_at = filed_at + timedelta(hours=6)
            accession = f"0000000000-{(fiscal_year + 1) % 100:02d}-{index:06d}"
            growth = 1.0 + 0.03 * year_offset + float(rng.uniform(-0.02, 0.05))
            revenue = base_revenue * growth
            net_income = revenue * float(rng.uniform(0.04, 0.18))
            assets = revenue * float(rng.uniform(1.2, 3.0))
            equity = assets * float(rng.uniform(0.25, 0.55))
            values = {
                "Revenue": revenue,
                "NetIncomeLoss": net_income,
                "Assets": assets,
                "StockholdersEquity": equity,
            }
            asset, bundle_created = self._write_fundamentals_asset(
                store=store,
                ticker=ticker,
                fiscal_year=fiscal_year,
                accession=accession,
                filed_at=filed_at,
                available_at=available_at,
                values=values,
                suffix="",
            )
            # Facts are only ever created alongside a newly written bundle
            # asset: a reused (already-existing) bundle means its facts were
            # already committed by the run that first wrote it (the whole
            # command runs inside one `transaction.atomic()` block), so
            # rebuilding them here would violate `unique_fundamental_vintage`
            # and -- more importantly -- would risk mutating an immutable
            # vintage.
            if bundle_created:
                for concept in FUNDAMENTAL_CONCEPTS:
                    facts.append(
                        FundamentalFact(
                            company=company,
                            provider=PROVIDER,
                            concept=concept,
                            source_concept=f"us-gaap:{concept}",
                            value=_decimal(values[concept], 2),
                            unit=currency,
                            currency=currency,
                            period_start=period_start,
                            period_end=period_end,
                            fiscal_year=fiscal_year,
                            fiscal_period="FY",
                            accession=accession,
                            filed_at=filed_at,
                            available_at=available_at,
                            is_amendment=False,
                            quality_flags=["synthetic"],
                            source_asset=asset,
                        )
                    )
            if fiscal_year == last_fiscal_year:
                amendment_fact = self._build_amendment_fact(
                    store=store,
                    company=company,
                    ticker=ticker,
                    currency=currency,
                    period_start=period_start,
                    period_end=period_end,
                    fiscal_year=fiscal_year,
                    base_accession=accession,
                    original_filed_at=filed_at,
                    original_net_income=net_income,
                )
                if amendment_fact is not None:
                    facts.append(amendment_fact)
        return facts

    def _write_fundamentals_asset(
        self,
        *,
        store: AssetStore,
        ticker: str,
        fiscal_year: int,
        accession: str,
        filed_at: datetime,
        available_at: datetime,
        values: dict[str, float],
        suffix: str,
    ) -> tuple[DataAsset, bool]:
        relative_path = f"fundamentals/{ticker}/{fiscal_year}{suffix}.json"

        def _write() -> StoredAsset:
            payload = json.dumps(
                {
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "accession": accession,
                    "filed_at": filed_at.isoformat(),
                    "facts": values,
                },
                sort_keys=True,
            ).encode("utf-8")
            return store.write_bytes(relative_path, payload)

        return _get_or_write_asset(
            relative_path=relative_path,
            write=_write,
            provider=PROVIDER,
            kind="fundamentals",
            subject=f"{ticker}:{fiscal_year}{suffix}",
            retrieved_at=available_at,
            available_at=available_at,
            period_start=date(fiscal_year, 1, 1),
            period_end=date(fiscal_year, 12, 31),
            metadata=_synthetic_metadata(accession=accession),
        )

    def _build_amendment_fact(
        self,
        *,
        store: AssetStore,
        company: Company,
        ticker: str,
        currency: str,
        period_start: date,
        period_end: date,
        fiscal_year: int,
        base_accession: str,
        original_filed_at: datetime,
        original_net_income: float,
    ) -> FundamentalFact | None:
        amended_value = original_net_income * 1.02
        amend_filed_at = original_filed_at + timedelta(days=60)
        amend_available_at = amend_filed_at + timedelta(hours=6)
        amend_accession = f"{base_accession}-A"
        asset, created = self._write_fundamentals_asset(
            store=store,
            ticker=ticker,
            fiscal_year=fiscal_year,
            accession=amend_accession,
            filed_at=amend_filed_at,
            available_at=amend_available_at,
            values={"NetIncomeLoss": amended_value},
            suffix="-amendment",
        )
        if not created:
            return None
        return FundamentalFact(
            company=company,
            provider=PROVIDER,
            concept="NetIncomeLoss",
            source_concept="us-gaap:NetIncomeLoss",
            value=_decimal(amended_value, 2),
            unit=currency,
            currency=currency,
            period_start=period_start,
            period_end=period_end,
            fiscal_year=fiscal_year,
            fiscal_period="FY",
            accession=amend_accession,
            filed_at=amend_filed_at,
            available_at=amend_available_at,
            is_amendment=True,
            quality_flags=["synthetic", "restated"],
            source_asset=asset,
        )

    def _build_fx_rates(self, store: AssetStore) -> list[FxRate]:
        rates: list[FxRate] = []
        fridays = _weekly_observation_dates(START_DATE, END_DATE)
        anchor = _fx_anchor_date(START_DATE)
        for pair_index, (base, quote) in enumerate(FX_PAIRS):
            rng = np.random.default_rng(SEED + 70_000 + pair_index)
            start_value = 1.05 if quote == "USD" else 0.85
            log_returns = rng.normal(0.0, 0.006, size=len(fridays))
            values = start_value * np.exp(np.cumsum(log_returns))
            if anchor < fridays[0]:
                # Prepend the pre-history anchor at the series' own starting
                # level. It is deliberately *not* drawn from the RNG stream,
                # so every already-written weekly bundle keeps byte-identical
                # contents and reruns stay idempotent.
                pair_dates = [anchor, *fridays]
                pair_values = np.concatenate(([start_value], values))
            else:
                pair_dates = list(fridays)
                pair_values = values
            by_year: dict[int, list[int]] = {}
            for position, day in enumerate(pair_dates):
                by_year.setdefault(day.year, []).append(position)
            for year in sorted(by_year):
                positions = by_year[year]
                asset, bundle_created = self._write_fx_bundle_asset(
                    store=store,
                    base=base,
                    quote=quote,
                    year=year,
                    fridays=pair_dates,
                    positions=positions,
                    values=pair_values,
                )
                # As with fundamentals bundles: only build rate rows for a
                # newly written bundle; a reused bundle's rates were already
                # committed by the run that first wrote it.
                if not bundle_created:
                    continue
                for position in positions:
                    observation_date = pair_dates[position]
                    rates.append(
                        FxRate(
                            base_currency=base,
                            quote_currency=quote,
                            observation_date=observation_date,
                            value=_decimal(float(pair_values[position]), 6),
                            published_at=datetime.combine(
                                observation_date, time(14, 15), tzinfo=UTC
                            ),
                            available_at=asset.available_at,
                            source_asset=asset,
                        )
                    )
        return rates

    def _write_fx_bundle_asset(
        self,
        *,
        store: AssetStore,
        base: str,
        quote: str,
        year: int,
        fridays: list[date],
        positions: list[int],
        values: np.ndarray[Any, np.dtype[np.float64]],
    ) -> tuple[DataAsset, bool]:
        series_key = f"D.{quote}.{base}.SP00.A"
        relative_path = f"fx/{base}{quote}/{year}.csv"

        def _write() -> StoredAsset:
            lines = ["KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE"]
            for position in positions:
                observation_date = fridays[position]
                lines.append(
                    f"{series_key},D,{quote},{base},SP00,A,"
                    f"{observation_date.isoformat()},{values[position]:.6f}"
                )
            payload = ("\n".join(lines) + "\n").encode("utf-8")
            return store.write_bytes(relative_path, payload)

        last_day = fridays[positions[-1]]
        bundle_available_at = datetime.combine(last_day, time(16, 5), tzinfo=UTC)
        return _get_or_write_asset(
            relative_path=relative_path,
            write=_write,
            provider=PROVIDER,
            kind="fx_rates",
            subject=f"{base}{quote}",
            retrieved_at=bundle_available_at,
            available_at=bundle_available_at,
            period_start=fridays[positions[0]],
            period_end=last_day,
            metadata=_synthetic_metadata(series_key=series_key, rows=len(positions)),
        )

    def _seed_benchmark(
        self,
        store: AssetStore,
        benchmark_config: dict[str, Any],
        seed: int,
    ) -> str:
        subject = str(benchmark_config["benchmark_subject"])
        sessions = _sessions_for("XNYS")
        frame = _generate_price_frame(sessions, seed + 999_999)
        last_session: date = frame["date"][-1]
        retrieved_at = datetime.combine(last_session, time(21, 30), tzinfo=UTC)
        relative_path = f"price_history/{subject}.parquet"
        _get_or_write_asset(
            relative_path=relative_path,
            write=lambda: store.write_frame(relative_path, frame),
            provider=PROVIDER,
            kind="price_history",
            subject=subject,
            retrieved_at=retrieved_at,
            available_at=retrieved_at,
            period_start=frame["date"][0],
            period_end=last_session,
            metadata=_synthetic_metadata(
                rows=frame.height,
                role="benchmark",
                config_version=benchmark_config.get("config_version"),
            ),
        )
        return subject

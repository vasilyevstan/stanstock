from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from django.db import IntegrityError

from stanstock.data.assets import AssetStore
from stanstock.data.live_us import (
    PRIVATE_USAGE_SCOPE,
    ProviderCreditBudget,
    UsUniverseConfig,
    _ensure_listings,
    _persist_catalog,
    _persist_price_series,
    load_us_universe_config,
    resolve_us_target_date,
    run_us_daily,
)
from stanstock.data.management.config_loader import default_us_universe_config_path
from stanstock.data.models import (
    DataAsset,
    LatestMarketData,
    ProviderRecord,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.data.providers.contracts import (
    PriceBar,
    PriceSeries,
    StockCatalog,
    StockReference,
)
from stanstock.data.providers.exceptions import ProviderConfigurationError, ProviderQuotaError
from stanstock.research.models import AnalysisRun, Prediction, StockAnalysis

pytestmark = pytest.mark.django_db

TARGET_DATE = date(2026, 9, 4)
RETRIEVED_AT = datetime(2026, 9, 5, 1, tzinfo=UTC)


def _config(
    *,
    symbols: tuple[str, ...] = ("AAA", "BBB"),
    minimum_eligible: int = 1,
) -> UsUniverseConfig:
    raw: dict[str, Any] = {
        "schema_version": 1,
        "config_version": "test-v1",
        "slug": "test-us-live",
        "name": "Test US Live",
        "description": "Test-only private US universe.",
        "provider": "twelve_data",
        "country": "United States",
        "instrument_type": "Common Stock",
        "currency": "USD",
        "exchanges": ["NASDAQ"],
        "benchmark_symbol": "SPY",
        "benchmark_currency": "USD",
        "benchmark_type": "ETF",
        "history_years": 1,
        "minimum_history_sessions": 1,
        "price_adjustment": "splits",
        "minimum_eligible": minimum_eligible,
        "maximum_symbols": 10,
        "symbols": list(symbols),
    }
    return UsUniverseConfig(
        slug="test-us-live",
        name="Test US Live",
        description="Test-only private US universe.",
        config_version="test-v1",
        country="United States",
        instrument_type="Common Stock",
        currency="USD",
        exchanges=("NASDAQ",),
        benchmark_symbol="SPY",
        benchmark_currency="USD",
        benchmark_type="ETF",
        history_years=1,
        minimum_history_sessions=1,
        price_adjustment="splits",
        minimum_eligible=minimum_eligible,
        maximum_symbols=10,
        symbols=symbols,
        raw=raw,
    )


def _reference(symbol: str, *, access_plan: str | None = "Basic") -> StockReference:
    return StockReference(
        symbol=symbol,
        name=f"{symbol} Incorporated",
        currency="USD",
        exchange="NASDAQ",
        mic_code="XNAS",
        country="United States",
        instrument_type="Common Stock",
        figi_code=f"BBG-{symbol}",
        access_plan=access_plan,
    )


def _catalog(config: UsUniverseConfig) -> StockCatalog:
    return StockCatalog(
        provider="twelve_data",
        exchange="NASDAQ",
        references=tuple(_reference(symbol) for symbol in config.symbols),
        count=len(config.symbols),
        retrieved_at=RETRIEVED_AT,
        source_url="https://api.twelvedata.com/stocks?exchange=NASDAQ",
        raw_bytes=b'{"status":"ok","data":[]}',
    )


def _series(
    symbol: str,
    *,
    include_target: bool = True,
    instrument_type: str = "Common Stock",
) -> PriceSeries:
    bars = [
        PriceBar(
            trade_date=date(2025, 9, 4),
            open=Decimal("99"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=1_000_000,
        )
    ]
    if include_target:
        bars.append(
            PriceBar(
                trade_date=TARGET_DATE,
                open=Decimal("100"),
                high=Decimal("104"),
                low=Decimal("99"),
                close=Decimal("103"),
                volume=1_200_000,
            )
        )
    return PriceSeries(
        provider="twelve_data",
        symbol=symbol,
        currency="USD",
        bars=tuple(bars),
        retrieved_at=RETRIEVED_AT,
        source_url=f"https://api.twelvedata.com/time_series?symbol={symbol}",
        raw_bytes=f'{{"status":"ok","symbol":"{symbol}"}}'.encode(),
        exchange="NASDAQ" if instrument_type == "Common Stock" else "NYSE ARCA",
        mic_code="XNAS" if instrument_type == "Common Stock" else "ARCX",
        instrument_type=instrument_type,
        exchange_timezone="America/New_York",
        adjustment="splits",
    )


def _long_series(symbol: str, *, instrument_type: str) -> PriceSeries:
    dates: list[date] = []
    cursor = TARGET_DATE
    while len(dates) < 320:
        if cursor.weekday() < 5:
            dates.append(cursor)
        cursor -= timedelta(days=1)
    dates.reverse()
    bars = tuple(
        PriceBar(
            trade_date=trade_date,
            open=Decimal("100") + Decimal(index) / 10,
            high=Decimal("102") + Decimal(index) / 10,
            low=Decimal("99") + Decimal(index) / 10,
            close=Decimal("101") + Decimal(index) / 10,
            volume=1_000_000 + index,
        )
        for index, trade_date in enumerate(dates)
    )
    return PriceSeries(
        provider="twelve_data",
        symbol=symbol,
        currency="USD",
        bars=bars,
        retrieved_at=RETRIEVED_AT,
        source_url=f"https://api.twelvedata.com/time_series?symbol={symbol}",
        raw_bytes=f'{{"status":"ok","symbol":"{symbol}","rows":320}}'.encode(),
        exchange="NASDAQ" if instrument_type == "Common Stock" else "NYSE ARCA",
        mic_code="XNAS" if instrument_type == "Common Stock" else "ARCX",
        instrument_type=instrument_type,
        exchange_timezone="America/New_York",
        adjustment="splits",
    )


def _enable_provider(*, daily_limit: int = 800) -> ProviderRecord:
    return ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=True,
        terms_url="https://twelvedata.com/terms",
        usage_scope=PRIVATE_USAGE_SCOPE,
        status="ok",
        metadata={
            "daily_credit_limit": daily_limit,
            "credits_per_minute": 8,
            "internal_display_rights_confirmed": True,
            "plan": "grow",
        },
    )


def _patch_provider(
    monkeypatch: pytest.MonkeyPatch,
    config: UsUniverseConfig,
    *,
    missing_target: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    catalog_calls: list[str] = []
    price_calls: list[str] = []
    missing = missing_target or set()

    def fetch_catalog(**kwargs: object) -> StockCatalog:
        catalog_calls.append(str(kwargs["exchange"]))
        assert kwargs["api_key"] == "private-test-key"
        return _catalog(config)

    def fetch_prices(symbol: str, **kwargs: object) -> PriceSeries:
        price_calls.append(symbol)
        assert kwargs["api_key"] == "private-test-key"
        if symbol == config.benchmark_symbol:
            return _series(symbol, instrument_type="ETF")
        return _series(symbol, include_target=symbol not in missing)

    monkeypatch.setattr(
        "stanstock.data.live_us.twelve_data.fetch_stock_catalog",
        fetch_catalog,
    )
    monkeypatch.setattr(
        "stanstock.data.live_us.twelve_data.fetch_daily_price_series",
        fetch_prices,
    )
    return catalog_calls, price_calls


def _patch_analysis(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def analyze(**kwargs: object) -> list[SimpleNamespace]:
        calls.append(kwargs)
        snapshot = kwargs["universe_snapshot"]
        assert isinstance(snapshot, UniverseSnapshot)
        return [
            SimpleNamespace(predictions=(object(),))
            for _membership in snapshot.memberships.filter(eligible=True)
        ]

    monkeypatch.setattr("stanstock.data.live_us.analyze_snapshot", analyze)
    return calls


def test_committed_us_config_is_bounded_and_explicit() -> None:
    config = load_us_universe_config(default_us_universe_config_path())

    assert len(config.symbols) == 100
    assert len(set(config.symbols)) == 100
    assert config.maximum_symbols == 300
    assert config.minimum_eligible == 75
    assert config.minimum_history_sessions == 1260
    assert config.exchanges == ("NASDAQ", "NYSE")
    assert config.benchmark_symbol == "SPY"
    assert config.price_adjustment == "splits"
    assert "FISV" in config.symbols
    assert "MRSH" in config.symbols
    assert "FI" not in config.symbols
    assert "MMC" not in config.symbols


def test_config_rejects_duplicates_and_an_over_cap_symbol_list(tmp_path: Path) -> None:
    payload = _config().raw
    payload["symbols"] = ["AAA", "AAA"]
    duplicate_path = tmp_path / "duplicate.yaml"
    duplicate_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate symbols"):
        load_us_universe_config(duplicate_path)

    payload["symbols"] = ["AAA", "BBB"]
    payload["maximum_symbols"] = 1
    over_cap_path = tmp_path / "over-cap.yaml"
    over_cap_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="above configured maximum"):
        load_us_universe_config(over_cap_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provider", "other", "provider must be"),
        ("country", "Germany", "country must be"),
        ("instrument_type", "ETF", "instrument_type must be"),
        ("currency", "EUR", "currency must be"),
        ("exchanges", ["NASDAQ", "LSE"], "limited to NASDAQ and NYSE"),
        ("benchmark_currency", "EUR", "benchmark_currency must be"),
        ("benchmark_type", "Index", "benchmark_type must be"),
    ],
)
def test_config_rejects_values_outside_the_reviewed_us_scope(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _config().raw
    payload[field] = value
    config_path = tmp_path / f"{field}.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_us_universe_config(config_path)


@pytest.mark.parametrize(
    ("decision_time", "expected_date", "expected_grade"),
    [
        (
            datetime(2026, 9, 4, 19, 0, tzinfo=UTC),
            date(2026, 9, 3),
            UniverseSnapshot.Grade.RESEARCH,
        ),
        (
            datetime(2026, 9, 4, 21, 0, tzinfo=UTC),
            TARGET_DATE,
            UniverseSnapshot.Grade.OBSERVED,
        ),
        (
            datetime(2026, 9, 5, 12, 0, tzinfo=UTC),
            TARGET_DATE,
            UniverseSnapshot.Grade.OBSERVED,
        ),
        (
            datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
            TARGET_DATE,
            UniverseSnapshot.Grade.OBSERVED,
        ),
        (
            datetime(2026, 9, 9, 1, 0, tzinfo=UTC),
            date(2026, 9, 8),
            UniverseSnapshot.Grade.OBSERVED,
        ),
        (
            datetime(2026, 9, 8, 16, 0, tzinfo=UTC),
            TARGET_DATE,
            UniverseSnapshot.Grade.RESEARCH,
        ),
    ],
)
def test_target_date_uses_only_completed_us_sessions(
    decision_time: datetime,
    expected_date: date,
    expected_grade: str,
) -> None:
    target, grade = resolve_us_target_date(decision_time=decision_time)

    assert target == expected_date
    assert grade == expected_grade


def test_explicit_historical_target_is_research_grade() -> None:
    target, grade = resolve_us_target_date(
        decision_time=datetime(2026, 9, 5, 12, 0, tzinfo=UTC),
        explicit_target=date(2026, 9, 2),
    )

    assert target == date(2026, 9, 2)
    assert grade == UniverseSnapshot.Grade.RESEARCH


def test_target_date_rejects_non_sessions_and_unfinished_sessions() -> None:
    decision_time = datetime(2026, 9, 4, 19, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="not an XNYS session"):
        resolve_us_target_date(
            decision_time=decision_time,
            explicit_target=date(2026, 9, 5),
        )
    with pytest.raises(ValueError, match="not yet complete"):
        resolve_us_target_date(
            decision_time=decision_time,
            explicit_target=TARGET_DATE,
        )


def test_credit_budget_rejects_a_run_that_exceeds_local_daily_allowance() -> None:
    _enable_provider(daily_limit=5)
    record = ProviderRecord.objects.get(provider="twelve_data")
    record.metadata.update(
        {
            "credit_usage_date": datetime.now(tz=UTC).date().isoformat(),
            "credits_used_local": 4,
        }
    )
    record.save(update_fields=["metadata"])

    with pytest.raises(ProviderQuotaError, match="only 1 locally tracked credits remain"):
        ProviderCreditBudget(enforce_spacing=False).preflight(2)


def test_credit_budget_rejects_enabled_record_without_display_rights() -> None:
    ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=True,
        status="ok",
        metadata={"daily_credit_limit": 800, "credits_per_minute": 8},
    )

    with pytest.raises(ProviderConfigurationError, match="internal-display entitlement"):
        ProviderCreditBudget(enforce_spacing=False).preflight(1)


def test_run_us_daily_persists_vintages_snapshot_and_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    provider = _enable_provider()
    catalog_calls, price_calls = _patch_provider(monkeypatch, config)
    analysis_calls = _patch_analysis(monkeypatch)

    result = run_us_daily(
        config=config,
        target_date=TARGET_DATE,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        api_key="private-test-key",
        decision_time=RETRIEVED_AT,
        store=AssetStore(tmp_path),
        enforce_rate_limit=False,
    )

    assert catalog_calls == ["NASDAQ"]
    assert price_calls == ["AAA", "BBB", "SPY"]
    assert result.eligible == 2
    assert result.excluded == 0
    assert result.price_assets == 3
    assert result.raw_assets == 4
    assert result.credits_used == 4
    assert result.analyses == 2
    assert result.predictions == 2
    assert result.snapshot.grade == UniverseSnapshot.Grade.OBSERVED
    assert UniverseMembership.objects.filter(snapshot=result.snapshot, eligible=True).count() == 2
    assert LatestMarketData.objects.count() == 2
    assert DataAsset.objects.filter(kind="stock_catalog").count() == 1
    assert DataAsset.objects.filter(kind="raw_price_history").count() == 3
    assert DataAsset.objects.filter(kind="price_history").count() == 3
    assert all(
        (tmp_path / path).is_file()
        for path in DataAsset.objects.values_list("relative_path", flat=True)
    )
    assert all(
        asset.metadata["usage_scope"] == PRIVATE_USAGE_SCOPE for asset in DataAsset.objects.all()
    )
    assert analysis_calls[0]["provider"] == "twelve_data"
    assert analysis_calls[0]["benchmark_subject"] == "SPY"

    provider.refresh_from_db()
    assert provider.metadata["credits_used_local"] == 4
    assert provider.metadata["last_target_date"] == TARGET_DATE.isoformat()
    assert "private-test-key" not in str(provider.metadata)
    assert "private-test-key" not in " ".join(
        DataAsset.objects.values_list("relative_path", flat=True)
    )


def test_run_us_daily_reaches_the_real_analysis_and_prediction_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(symbols=("AAA",), minimum_eligible=1)
    _enable_provider()
    catalog_calls: list[None] = []
    price_calls: list[str] = []

    def fetch_catalog(**kwargs: object) -> StockCatalog:
        catalog_calls.append(None)
        return _catalog(config)

    def fetch_prices(symbol: str, **kwargs: object) -> PriceSeries:
        price_calls.append(symbol)
        return _long_series(
            symbol,
            instrument_type="ETF" if symbol == "SPY" else "Common Stock",
        )

    monkeypatch.setattr(
        "stanstock.data.live_us.twelve_data.fetch_stock_catalog",
        fetch_catalog,
    )
    monkeypatch.setattr(
        "stanstock.data.live_us.twelve_data.fetch_daily_price_series",
        fetch_prices,
    )
    monkeypatch.setattr(
        "stanstock.data.live_us.timezone.now",
        lambda: datetime(2026, 9, 9, 15, tzinfo=UTC),
    )

    result = run_us_daily(
        config=config,
        target_date=TARGET_DATE,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        api_key="private-test-key",
        decision_time=RETRIEVED_AT,
        store=AssetStore(tmp_path),
        enforce_rate_limit=False,
    )

    assert result.analyses == 1
    assert result.predictions == 1
    assert AnalysisRun.objects.count() == 1
    assert StockAnalysis.objects.count() == 1
    assert Prediction.objects.count() == 1
    analysis = StockAnalysis.objects.get()
    assert analysis.run.issued_on_time is True
    assert analysis.run.data_cutoff == analysis.run.generated_at
    assert analysis.data_quality["source_assets"][0]["provider"] == "twelve_data"
    prediction = Prediction.objects.get()
    assert prediction.horizon == Prediction.Horizon.SHORT
    assert prediction.issued_on_time is True
    assert all(
        datetime.fromisoformat(asset[field]) <= analysis.run.data_cutoff
        for asset in analysis.data_quality["source_assets"]
        for field in ("available_at", "retrieved_at")
    )

    repeated = run_us_daily(
        config=config,
        target_date=TARGET_DATE,
        snapshot_grade=UniverseSnapshot.Grade.RESEARCH,
        api_key=None,
        decision_time=datetime(2026, 9, 8, 15, tzinfo=UTC),
        store=AssetStore(tmp_path),
        enforce_rate_limit=False,
    )

    assert repeated.snapshot == result.snapshot
    assert repeated.snapshot.grade == UniverseSnapshot.Grade.OBSERVED
    assert repeated.credits_used == 0
    assert repeated.price_assets == 0
    assert repeated.raw_assets == 0
    assert catalog_calls == [None]
    assert price_calls == ["AAA", "SPY"]

    conflicting_snapshot = UniverseSnapshot.objects.create(
        universe=result.snapshot.universe,
        as_of_date=TARGET_DATE,
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="f" * 64,
    )
    AnalysisRun.objects.create(
        generated_at=datetime(2026, 9, 8, 15, tzinfo=UTC),
        data_cutoff=datetime(2026, 9, 4, 23, 59, tzinfo=UTC),
        target_date=TARGET_DATE,
        universe_snapshot=conflicting_snapshot,
        config_version=analysis.run.config_version,
        config_hash=analysis.run.config_hash,
        code_revision="conflict-test",
    )
    with pytest.raises(ValueError, match="Conflicting completed US analysis runs"):
        run_us_daily(
            config=config,
            target_date=TARGET_DATE,
            snapshot_grade=UniverseSnapshot.Grade.RESEARCH,
            api_key=None,
            decision_time=datetime(2026, 9, 8, 15, tzinfo=UTC),
            store=AssetStore(tmp_path),
            enforce_rate_limit=False,
        )


def test_missing_target_bar_creates_an_explicit_ineligible_membership(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(minimum_eligible=1)
    _enable_provider()
    _patch_provider(monkeypatch, config, missing_target={"BBB"})
    _patch_analysis(monkeypatch)

    result = run_us_daily(
        config=config,
        target_date=TARGET_DATE,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        api_key="private-test-key",
        decision_time=RETRIEVED_AT,
        store=AssetStore(tmp_path),
        enforce_rate_limit=False,
    )

    excluded = UniverseMembership.objects.get(snapshot=result.snapshot, listing__ticker="BBB")
    assert result.eligible == 1
    assert result.excluded == 1
    assert excluded.eligible is False
    assert excluded.exclusion_reason == f"No {TARGET_DATE.isoformat()} daily bar"
    assert not LatestMarketData.objects.filter(listing__ticker="BBB").exists()


def test_historical_price_vintage_cannot_move_latest_market_state_backward(
    tmp_path: Path,
) -> None:
    config = _config(symbols=("AAA",), minimum_eligible=1)
    listing = _ensure_listings(
        config,
        {"AAA": _reference("AAA")},
        target_date=TARGET_DATE,
    )["AAA"]
    store = AssetStore(tmp_path)
    current_series = _series("AAA")
    _persist_price_series(store=store, series=current_series, listing=listing)
    current = LatestMarketData.objects.get(listing=listing)
    original_asset_id = current.source_asset_id

    corrected_series = replace(
        current_series,
        bars=(
            *current_series.bars[:-1],
            replace(current_series.bars[-1], close=Decimal("104")),
        ),
        retrieved_at=RETRIEVED_AT + timedelta(hours=1),
        raw_bytes=b'{"status":"ok","symbol":"AAA","corrected":true}',
    )
    _persist_price_series(store=store, series=corrected_series, listing=listing)
    current.refresh_from_db()
    assert current.session_date == TARGET_DATE
    assert current.close == Decimal("104")
    corrected_asset_id = current.source_asset_id
    assert corrected_asset_id != original_asset_id

    historical_series = replace(
        current_series,
        bars=(
            PriceBar(
                trade_date=date(2025, 9, 4),
                open=Decimal("49"),
                high=Decimal("52"),
                low=Decimal("48"),
                close=Decimal("50"),
                volume=500_000,
            ),
        ),
        retrieved_at=RETRIEVED_AT + timedelta(days=1),
        raw_bytes=b'{"status":"ok","symbol":"AAA","historical":true}',
    )
    _persist_price_series(store=store, series=historical_series, listing=listing)

    current.refresh_from_db()
    assert current.session_date == TARGET_DATE
    assert current.close == Decimal("104")
    assert current.source_asset_id == corrected_asset_id


def test_short_history_is_ineligible_instead_of_becoming_a_live_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_config = _config(minimum_eligible=1)
    config = replace(
        base_config,
        minimum_history_sessions=100,
        raw={**base_config.raw, "minimum_history_sessions": 100},
    )
    _enable_provider()
    monkeypatch.setattr(
        "stanstock.data.live_us.twelve_data.fetch_stock_catalog",
        lambda **kwargs: _catalog(config),
    )

    def fetch_prices(symbol: str, **kwargs: object) -> PriceSeries:
        if symbol == "SPY":
            return _long_series(symbol, instrument_type="ETF")
        if symbol == "AAA":
            return _long_series(symbol, instrument_type="Common Stock")
        return _series(symbol)

    monkeypatch.setattr(
        "stanstock.data.live_us.twelve_data.fetch_daily_price_series",
        fetch_prices,
    )
    _patch_analysis(monkeypatch)

    result = run_us_daily(
        config=config,
        target_date=TARGET_DATE,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        api_key="private-test-key",
        decision_time=RETRIEVED_AT,
        store=AssetStore(tmp_path),
        enforce_rate_limit=False,
    )

    excluded = UniverseMembership.objects.get(snapshot=result.snapshot, listing__ticker="BBB")
    assert result.eligible == 1
    assert excluded.eligible is False
    assert excluded.exclusion_reason == "Only 2 history sessions; minimum is 100"


def test_catalog_plan_requirement_excludes_only_the_unavailable_symbol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(minimum_eligible=1)
    _enable_provider()
    catalog = StockCatalog(
        provider="twelve_data",
        exchange="NASDAQ",
        references=(
            _reference("AAA"),
            _reference("BBB", access_plan="Pro"),
        ),
        count=2,
        retrieved_at=RETRIEVED_AT,
        source_url="https://api.twelvedata.com/stocks?exchange=NASDAQ",
        raw_bytes=b'{"status":"ok","data":[]}',
    )
    price_calls: list[str] = []
    monkeypatch.setattr(
        "stanstock.data.live_us.twelve_data.fetch_stock_catalog",
        lambda **kwargs: catalog,
    )

    def fetch_prices(symbol: str, **kwargs: object) -> PriceSeries:
        price_calls.append(symbol)
        return _series(symbol, instrument_type="ETF" if symbol == "SPY" else "Common Stock")

    monkeypatch.setattr(
        "stanstock.data.live_us.twelve_data.fetch_daily_price_series",
        fetch_prices,
    )
    _patch_analysis(monkeypatch)

    result = run_us_daily(
        config=config,
        target_date=TARGET_DATE,
        snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
        api_key="private-test-key",
        decision_time=RETRIEVED_AT,
        store=AssetStore(tmp_path),
        enforce_rate_limit=False,
    )

    excluded = UniverseMembership.objects.get(snapshot=result.snapshot, listing__ticker="BBB")
    assert result.eligible == 1
    assert result.excluded == 1
    assert result.credits_used == 3
    assert price_calls == ["AAA", "SPY"]
    assert excluded.eligible is False
    assert "Requires Twelve Data plan Pro" in excluded.exclusion_reason


def test_analysis_failure_rolls_back_snapshot_but_keeps_immutable_source_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config()
    _enable_provider()
    _patch_provider(monkeypatch, config)

    def fail_analysis(**kwargs: object) -> list[object]:
        raise RuntimeError("analysis failed")

    monkeypatch.setattr("stanstock.data.live_us.analyze_snapshot", fail_analysis)

    with pytest.raises(RuntimeError, match="analysis failed"):
        run_us_daily(
            config=config,
            target_date=TARGET_DATE,
            snapshot_grade=UniverseSnapshot.Grade.OBSERVED,
            api_key="private-test-key",
            decision_time=RETRIEVED_AT,
            store=AssetStore(tmp_path),
            enforce_rate_limit=False,
        )

    assert Universe.objects.count() == 0
    assert UniverseSnapshot.objects.count() == 0
    assert UniverseMembership.objects.count() == 0
    assert DataAsset.objects.count() == 7
    assert LatestMarketData.objects.count() == 2


def test_duplicate_catalog_registration_does_not_delete_existing_asset_file(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    catalog = _catalog(_config())
    first = _persist_catalog(store, catalog)

    with pytest.raises(IntegrityError):
        _persist_catalog(store, catalog)

    assert store.resolve(first.relative_path).read_bytes() == catalog.raw_bytes
    assert DataAsset.objects.filter(pk=first.pk).exists()

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from io import StringIO
from math import sqrt

import numpy as np
import polars as pl
import pytest
from django.core.management import call_command
from django.test import override_settings

from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.etfs import (
    INVESTABLE_US_ETF_NAME,
    InvestableEtfEvidenceError,
    build_etf_overview,
    ensure_investable_spy_listing,
    sync_investable_spy_from_asset,
)
from stanstock.data.market_state import update_latest_market_data
from stanstock.data.models import Company, Listing, Region, Security

pytestmark = pytest.mark.django_db


def test_spy_listing_is_idempotent_and_explicitly_an_etf() -> None:
    first = ensure_investable_spy_listing(
        currency="USD",
        mic_code="ARCX",
        valid_from=date(2026, 9, 4),
    )
    second = ensure_investable_spy_listing(
        currency="usd",
        mic_code="arcx",
        valid_from=date(2026, 9, 5),
    )

    assert second.pk == first.pk
    assert first.ticker == "SPY"
    assert first.provider_symbol == "SPY"
    assert first.exchange_mic == "ARCX"
    assert first.security.security_type == Security.SecurityType.ETF
    assert first.security.company.name == INVESTABLE_US_ETF_NAME


def test_spy_listing_rejects_a_conflicting_stock_identity() -> None:
    company = Company.objects.create(name="Conflicting SPY", country="US")
    security = Security.objects.create(
        company=company,
        security_type=Security.SecurityType.COMMON_STOCK,
        name="Conflicting SPY Common",
    )
    Listing.objects.create(
        security=security,
        ticker="SPY",
        exchange_mic="ARCX",
        provider_symbol="SPY",
        currency="USD",
        region=Region.US,
    )

    with pytest.raises(ValueError, match="not identified as an exchange-traded fund"):
        ensure_investable_spy_listing(
            currency="USD",
            mic_code="ARCX",
            valid_from=date(2026, 9, 4),
        )


def test_spy_asset_sync_builds_price_only_etf_overview(tmp_path) -> None:
    start = date(2025, 9, 1)
    dates = [start + timedelta(days=index) for index in range(260)]
    closes = [Decimal("100") + Decimal(index) / Decimal("10") for index in range(260)]
    closes[180] = closes[179] * Decimal("0.80")
    frame = pl.DataFrame(
        {
            "date": dates,
            "close": [float(value) for value in closes],
            "volume": [1_000_000 + index for index in range(260)],
        }
    )
    store = AssetStore(tmp_path)
    stored = store.write_frame("tests/spy-etf.parquet", frame)
    retrieved_at = datetime(2026, 9, 5, 1, tzinfo=UTC)
    asset = register_asset(
        provider="twelve_data",
        kind="price_history",
        subject="SPY",
        stored=stored,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
        period_start=dates[0],
        period_end=dates[-1],
        metadata={
            "currency": "USD",
            "mic_code": "ARCX",
            "instrument_type": "ETF",
            "interval": "1day",
            "adjustment": "splits",
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )

    listing = sync_investable_spy_from_asset(
        asset=asset,
        target_date=dates[-1],
        store=store,
    )
    overview = build_etf_overview(listing, store=store)

    assert listing.latest_market_data.source_asset == asset
    assert listing.latest_market_data.close == closes[-1]
    assert overview.observation_count == 253
    assert overview.period_start == dates[-253]
    assert overview.period_end == dates[-1]
    expected_closes = np.asarray([float(value) for value in closes[-253:]])
    expected_returns = expected_closes[1:] / expected_closes[:-1] - 1.0
    expected_running_high = np.maximum.accumulate(expected_closes)
    assert overview.price_return == pytest.approx(expected_closes[-1] / expected_closes[0] - 1.0)
    assert overview.annualized_volatility == pytest.approx(
        np.std(expected_returns, ddof=1) * sqrt(252)
    )
    assert overview.max_drawdown == pytest.approx(
        np.min(expected_closes / expected_running_high - 1.0)
    )
    assert overview.benchmark_identity == "S&P 500 benchmark"
    assert overview.portfolio_role == "Core benchmark ETF"
    assert overview.dividends_included is False

    listing.delete()
    output = StringIO()
    with override_settings(DATA_DIR=tmp_path):
        call_command("sync_investable_etfs", stdout=output)
    restored = Listing.objects.select_related("security", "latest_market_data").get(
        provider_symbol="SPY"
    )
    assert restored.security.security_type == Security.SecurityType.ETF
    assert restored.latest_market_data.source_asset == asset
    assert "Synced SPY ETF" in output.getvalue()


def test_etf_overview_excludes_asset_rows_after_current_market_session(tmp_path) -> None:
    dates = [date(2026, 9, day) for day in range(1, 5)]
    frame = pl.DataFrame(
        {
            "date": dates,
            "close": [100.0, 110.0, 220.0, 55.0],
            "volume": [1_000_000] * 4,
        }
    )
    store = AssetStore(tmp_path)
    stored = store.write_frame("tests/spy-etf-cutoff.parquet", frame)
    retrieved_at = datetime(2026, 9, 5, 1, tzinfo=UTC)
    asset = register_asset(
        provider="twelve_data",
        kind="price_history",
        subject="SPY",
        stored=stored,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
        period_start=dates[0],
        period_end=dates[-1],
        metadata={
            "currency": "USD",
            "mic_code": "ARCX",
            "instrument_type": "ETF",
            "interval": "1day",
            "adjustment": "splits",
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )

    listing = sync_investable_spy_from_asset(
        asset=asset,
        target_date=dates[1],
        store=store,
    )
    overview = build_etf_overview(listing, store=store)

    assert overview.market_data.session_date == dates[1]
    assert overview.period_end == dates[1]
    assert overview.observation_count == 2
    assert overview.price_return == pytest.approx(0.10)
    assert overview.annualized_volatility is None
    assert overview.max_drawdown == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instrument_type", None),
        ("currency", None),
        ("mic_code", None),
        ("mic_code", "XNAS"),
        ("interval", None),
        ("interval", "1h"),
        ("adjustment", None),
        ("adjustment", "all"),
        ("return_definition", None),
        ("return_definition", "total_return"),
        ("dividends_included", None),
        ("dividends_included", True),
    ],
)
def test_spy_sync_rejects_unproven_or_incompatible_price_basis(
    tmp_path,
    field: str,
    value: object,
) -> None:
    target_date = date(2026, 9, 4)
    frame = pl.DataFrame(
        {
            "date": [date(2026, 9, 3), target_date],
            "close": [100.0, 101.0],
            "volume": [1_000_000, 1_100_000],
        }
    )
    store = AssetStore(tmp_path)
    stored = store.write_frame(f"tests/spy-etf-invalid-{field}-{value}.parquet", frame)
    metadata: dict[str, object] = {
        "currency": "USD",
        "mic_code": "ARCX",
        "instrument_type": "ETF",
        "interval": "1day",
        "adjustment": "splits",
        "return_definition": "split_adjusted_price_return",
        "dividends_included": False,
    }
    if value is None:
        metadata.pop(field)
    else:
        metadata[field] = value
    asset = register_asset(
        provider="twelve_data",
        kind="price_history",
        subject="SPY",
        stored=stored,
        retrieved_at=datetime(2026, 9, 5, 1, tzinfo=UTC),
        available_at=datetime(2026, 9, 5, 1, tzinfo=UTC),
        period_start=date(2026, 9, 3),
        period_end=target_date,
        metadata=metadata,
    )

    with pytest.raises(ValueError, match=field):
        sync_investable_spy_from_asset(
            asset=asset,
            target_date=target_date,
            store=store,
        )
    assert not Listing.objects.filter(provider_symbol="SPY").exists()


def _valid_spy_price_columns(target_date: date) -> dict[str, list[object]]:
    return {
        "date": [date(2026, 9, 3), target_date],
        "close": [100.0, 101.0],
        "volume": [1_000_000, 1_100_000],
    }


def _register_spy_price_asset(
    store: AssetStore,
    frame: pl.DataFrame,
    *,
    period_start: date,
    period_end: date,
    relative_path: str,
) -> object:
    stored = store.write_frame(relative_path, frame)
    retrieved_at = datetime(2026, 9, 5, 1, tzinfo=UTC)
    return register_asset(
        provider="twelve_data",
        kind="price_history",
        subject="SPY",
        stored=stored,
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
        period_start=period_start,
        period_end=period_end,
        metadata={
            "currency": "USD",
            "mic_code": "ARCX",
            "instrument_type": "ETF",
            "interval": "1day",
            "adjustment": "splits",
            "return_definition": "split_adjusted_price_return",
            "dividends_included": False,
        },
    )


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("date", ["2026-09-03", "2026-09-04"]),
        ("close", ["100.0", "101.0"]),
        ("volume", [True, False]),
        ("volume", [1_000_000.0, 1_100_000.5]),
        ("volume", ["1000000", "1100000"]),
        ("volume", [1_000_000, -5]),
    ],
    ids=[
        "date-as-string",
        "close-as-string",
        "volume-as-bool",
        "volume-fractional-float",
        "volume-as-string",
        "volume-negative",
    ],
)
def test_spy_sync_rejects_malformed_or_negative_price_schema(
    tmp_path,
    column: str,
    values: list[object],
) -> None:
    """Every case here would previously be *silently coerced* into
    valid-looking data by a lenient `strict=False` cast (a string date/close
    parses, a bool/fractional/string volume truncates or converts) rather
    than raising -- so a kill-switch reverting the strict dtype check (and
    the negative-volume check) must reproduce a *silent pass* here, not
    merely a differently-worded failure.
    """
    target_date = date(2026, 9, 4)
    columns = _valid_spy_price_columns(target_date)
    columns[column] = values
    frame = pl.DataFrame(columns)
    store = AssetStore(tmp_path)
    asset = _register_spy_price_asset(
        store,
        frame,
        period_start=date(2026, 9, 3),
        period_end=target_date,
        relative_path=f"tests/spy-schema-{column}-adversarial.parquet",
    )

    with pytest.raises(InvestableEtfEvidenceError, match="SPY price evidence could not be read"):
        sync_investable_spy_from_asset(
            asset=asset,
            target_date=target_date,
            store=store,
        )
    assert not Listing.objects.filter(provider_symbol="SPY").exists()


def test_spy_sync_accepts_a_genuine_nullable_volume_observation(tmp_path) -> None:
    target_date = date(2026, 9, 4)
    columns = _valid_spy_price_columns(target_date)
    columns["volume"] = [1_000_000, None]
    frame = pl.DataFrame(columns)
    assert frame.schema["volume"] == pl.Int64
    store = AssetStore(tmp_path)
    asset = _register_spy_price_asset(
        store,
        frame,
        period_start=date(2026, 9, 3),
        period_end=target_date,
        relative_path="tests/spy-schema-nullable-volume.parquet",
    )

    listing = sync_investable_spy_from_asset(
        asset=asset,
        target_date=target_date,
        store=store,
    )

    assert listing.latest_market_data.volume is None
    assert listing.latest_market_data.close == Decimal("101.0")


def test_spy_sync_fails_closed_on_checksum_mismatch_without_reopening_by_path(
    tmp_path,
) -> None:
    """A physically altered file whose content no longer matches the
    registered `DataAsset.sha256` must fail closed even when the altered
    bytes still parse as a structurally valid Parquet frame -- the fix must
    authenticate the exact bytes it parses, not merely re-open a path that
    happens to still exist.
    """
    target_date = date(2026, 9, 4)
    columns = _valid_spy_price_columns(target_date)
    frame = pl.DataFrame(columns)
    store = AssetStore(tmp_path)
    relative_path = "tests/spy-schema-checksum.parquet"
    asset = _register_spy_price_asset(
        store,
        frame,
        period_start=date(2026, 9, 3),
        period_end=target_date,
        relative_path=relative_path,
    )
    registered_sha256 = asset.sha256
    original_bytes = store.read_bytes(relative_path)

    tampered = frame.with_columns(pl.col("close") * 5.0)
    tampered_path = tmp_path / asset.relative_path
    tampered.write_parquet(tampered_path)
    assert tampered_path.read_bytes() != original_bytes

    with pytest.raises(InvestableEtfEvidenceError, match="SPY price evidence could not be read"):
        sync_investable_spy_from_asset(
            asset=asset,
            target_date=target_date,
            store=store,
        )
    assert not Listing.objects.filter(provider_symbol="SPY").exists()
    # Sanity: the asset's registered checksum was never mutated by the
    # tampering, confirming the mismatch is real, not a fixture artifact.
    asset.refresh_from_db()
    assert asset.sha256 == registered_sha256


def _three_session_price_columns(target_date: date) -> dict[str, list[object]]:
    """A genuinely well-formed 3-row frame spanning one session strictly
    *before*, exactly *at*, and one strictly *after* `target_date` -- so a
    malformed value can be injected at any of the three positions without
    disturbing the other two rows' validity.
    """
    return {
        "date": [
            target_date - timedelta(days=1),
            target_date,
            target_date + timedelta(days=1),
        ],
        "close": [100.0, 101.0, 102.0],
        "volume": [1_000_000, 1_100_000, 1_200_000],
    }


_MALFORMED_ROW_CASES: list[tuple[str, list[tuple[int, str, object]]]] = [
    ("null-date-before-target", [(0, "date", None)]),
    ("null-date-at-target", [(1, "date", None)]),
    ("null-date-after-target", [(2, "date", None)]),
    ("null-close-before-target", [(0, "close", None)]),
    ("null-close-at-target", [(1, "close", None)]),
    ("null-close-after-target", [(2, "close", None)]),
    ("nan-close-before-target", [(0, "close", float("nan"))]),
    ("nan-close-at-target", [(1, "close", float("nan"))]),
    ("positive-inf-close-after-target", [(2, "close", float("inf"))]),
    ("negative-inf-close-before-target", [(0, "close", float("-inf"))]),
    ("zero-close-at-target", [(1, "close", 0.0)]),
    ("negative-close-before-target", [(0, "close", -5.0)]),
    ("negative-volume-after-target", [(2, "volume", -100)]),
    (
        "negative-volume-with-invalid-close-at-target",
        [(1, "volume", -100), (1, "close", float("nan"))],
    ),
    (
        "negative-volume-with-null-date-before-target",
        [(0, "volume", -100), (0, "date", None)],
    ),
]


def _apply_row_overrides(
    columns: dict[str, list[object]],
    overrides: list[tuple[int, str, object]],
) -> dict[str, list[object]]:
    mutated = {key: list(values) for key, values in columns.items()}
    for row_index, column, value in overrides:
        mutated[column][row_index] = value
    return mutated


@pytest.mark.parametrize(
    ("overrides",),
    [(overrides,) for _case_id, overrides in _MALFORMED_ROW_CASES],
    ids=[case_id for case_id, _overrides in _MALFORMED_ROW_CASES],
)
def test_spy_sync_fails_closed_on_any_malformed_row_without_dropping_it(
    tmp_path,
    overrides: list[tuple[int, str, object]],
) -> None:
    """A single malformed row anywhere in an otherwise well-formed 3-session
    frame -- whether strictly before, exactly at, or strictly after the
    target date -- must fail the *entire* asset closed. It must never be
    silently filtered away: doing so would let a sync "bridge" across a
    discarded session (computing `previous_close` from the wrong prior row)
    or let an overview compute trailing metrics from a quietly-reduced
    frame.
    """
    target_date = date(2026, 9, 4)
    columns = _apply_row_overrides(_three_session_price_columns(target_date), overrides)
    frame = pl.DataFrame(columns)
    store = AssetStore(tmp_path)
    asset = _register_spy_price_asset(
        store,
        frame,
        period_start=date(2026, 9, 3),
        period_end=date(2026, 9, 5),
        relative_path="tests/spy-malformed-row.parquet",
    )

    with pytest.raises(InvestableEtfEvidenceError, match="SPY price evidence could not be read"):
        sync_investable_spy_from_asset(
            asset=asset,
            target_date=target_date,
            store=store,
        )
    assert not Listing.objects.filter(provider_symbol="SPY").exists()


def test_etf_overview_fails_closed_for_bound_malformed_evidence(tmp_path) -> None:
    """`build_etf_overview` must independently reject a malformed row in the
    asset its `LatestMarketData` is bound to -- even though the row that
    was actually used to derive the bound `close`/`session_date` (a real,
    valid target-date row) is itself fine, a malformed sibling row
    elsewhere in the same asset must still fail the whole read, since the
    overview's trailing-window computation reads the entire frame.
    """
    target_date = date(2026, 9, 4)
    columns = _apply_row_overrides(
        _three_session_price_columns(target_date),
        [(0, "close", float("nan"))],
    )
    frame = pl.DataFrame(columns)
    store = AssetStore(tmp_path)
    asset = _register_spy_price_asset(
        store,
        frame,
        period_start=date(2026, 9, 3),
        period_end=date(2026, 9, 5),
        relative_path="tests/spy-malformed-row-overview.parquet",
    )
    listing = ensure_investable_spy_listing(
        currency="USD",
        mic_code="ARCX",
        valid_from=target_date,
    )
    # Bind `LatestMarketData` directly (bypassing `sync_investable_spy_from_asset`,
    # which would itself already refuse this malformed asset) to prove
    # `build_etf_overview` independently validates the entire bound asset,
    # not just the specific target-date row it ultimately reports.
    update_latest_market_data(
        listing=listing,
        session_date=target_date,
        observed_at=datetime(2026, 9, 5, 1, tzinfo=UTC),
        close=Decimal("101.0"),
        previous_close=None,
        volume=1_100_000,
        source_asset=asset,
    )

    with pytest.raises(InvestableEtfEvidenceError, match="SPY price evidence could not be read"):
        build_etf_overview(listing, store=store)

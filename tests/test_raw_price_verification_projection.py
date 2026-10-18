from __future__ import annotations

import json
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
from exchange_calendars import get_calendar

from stanstock.data import research_product
from stanstock.data.assets import AssetStore
from stanstock.data.models import DataAsset
from stanstock.data.providers import twelve_data
from stanstock.data.providers.exceptions import ProviderDataError


def _verification_source(root: Path, *, fault: str | None = None, rows: int = 1758):
    target = datetime(2026, 9, 11, tzinfo=UTC).date()
    now = datetime(2026, 9, 12, tzinfo=UTC)
    calendar = get_calendar("XNYS")
    dates = [
        day.date() for day in calendar.sessions_window(calendar.date_to_session(target), -rows)
    ]
    assert len(dates) == rows
    prices = [64.0 + index / 64 for index in range(rows)]
    volumes = [None if index % 7 == 0 else 1_000_000 + index for index in range(rows)]
    values = [
        {
            "datetime": day.isoformat(),
            "open": f"{price:.6f}",
            "high": f"{price + 1:.6f}",
            "low": f"{price - 1:.6f}",
            "close": f"{price:.6f}",
            "volume": None if volume is None else str(volume),
        }
        for day, price, volume in zip(dates, prices, volumes, strict=True)
    ]
    metadata = {
        "symbol": "SYN",
        "interval": "1day",
        "currency": "USD",
        "exchange": "NYSE",
        "mic_code": "XNYS",
        "type": "Common Stock",
        "exchange_timezone": "America/New_York",
    }
    payload = {"meta": metadata, "values": values, "status": "ok"}
    if fault == "early_ohlc":
        values[0]["high"] = "NaN"
    elif fault == "late_ohlc":
        values[-1]["high"] = "1"
    elif fault == "late_volume":
        values[-1]["volume"] = "-1"
    elif fault == "duplicate":
        values[-1]["datetime"] = values[-2]["datetime"]
    elif fault == "future":
        values[-1]["datetime"] = (target + timedelta(days=1)).isoformat()
    elif fault == "before":
        values[0]["datetime"] = (dates[0] - timedelta(days=1)).isoformat()
    elif fault == "envelope":
        payload["meta"] = []
    if fault == "normalized":
        prices[-1] += 1
    store = AssetStore(root)
    stored_raw = store.write_bytes("raw/prices.json", json.dumps(payload).encode())
    stored_frame = store.write_frame(
        "normalized/prices.parquet",
        pl.DataFrame({"date": dates, "close": prices, "volume": volumes}),
    )
    raw = DataAsset(
        provider="twelve_data",
        kind="raw_price_history",
        subject="SYN",
        relative_path=stored_raw.relative_path,
        sha256=stored_raw.sha256,
        retrieved_at=now,
        available_at=now,
        metadata={
            "adjustment": "splits",
            "source_url": "https://example.invalid/synthetic-prices",
        },
    )
    asset = DataAsset(
        provider="twelve_data",
        kind="price_history",
        subject="SYN",
        relative_path=stored_frame.relative_path,
        sha256=stored_frame.sha256,
        retrieved_at=now,
        available_at=now,
        period_start=dates[0],
        period_end=target,
        metadata={
            "currency": "EUR" if fault == "identity" else "USD",
            "exchange": "NYSE",
            "mic_code": "XNYS",
            "instrument_type": "Common Stock",
            "adjustment": "splits",
        },
    )
    return {"asset": asset, "raw": raw, "cutoff": now, "target_date": target, "store": store}


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("network or credentials accessed"))
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(twelve_data, "fetch", forbidden)
    monkeypatch.setattr(twelve_data, "resolve_api_key", forbidden)
    yield
    forbidden.assert_not_called()


def test_raw_verification_validates_all_rows_without_public_records(tmp_path, monkeypatch):
    source = _verification_source(tmp_path)
    bars = Mock(wraps=twelve_data.PriceBar)
    series = Mock(wraps=twelve_data.PriceSeries)
    decimals = Mock(wraps=twelve_data._optional_decimal)
    monkeypatch.setattr(twelve_data, "PriceBar", bars)
    monkeypatch.setattr(twelve_data, "PriceSeries", series)
    monkeypatch.setattr(twelve_data, "_optional_decimal", decimals)
    public = twelve_data.parse_daily_price_series(
        source["store"].read_bytes(source["raw"].relative_path),
        symbol="SYN",
        retrieved_at=source["raw"].retrieved_at,
        source_url=source["raw"].metadata["source_url"],
        start_date=source["asset"].period_start,
        end_date=source["target_date"],
    )
    assert len(public.bars) == bars.call_count == 1758
    assert series.call_count == 1 and decimals.call_count == 1758 * 4
    bars.reset_mock()
    series.reset_mock()
    decimals.reset_mock()
    result = research_product._verify_product_price_content(**source)
    assert result.asset is source["asset"] and result.frame.height == 1758
    assert decimals.call_count == 1758 * 4
    assert bars.call_count == series.call_count == 0


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("early_ohlc", "non-finite 'high'"),
        ("late_ohlc", "high was below open/close"),
        ("late_volume", "negative volume"),
        ("duplicate", "duplicate date"),
        ("future", "after requested end"),
        ("before", "before requested start"),
        ("envelope", "meta"),
    ],
)
def test_raw_verification_rejects_long_history_faults(tmp_path, fault, reason):
    with pytest.raises(ProviderDataError, match=reason):
        research_product._verify_product_price_content(
            **_verification_source(tmp_path, fault=fault)
        )


@pytest.mark.parametrize("fault", ["identity", "normalized"])
def test_raw_verification_rejects_identity_or_normalized_value_mismatch(tmp_path, fault):
    with pytest.raises(ValueError, match="diverge"):
        research_product._verify_product_price_content(
            **_verification_source(tmp_path, fault=fault)
        )


def test_raw_identity_failure_precedes_close_conversion(tmp_path, monkeypatch):
    source = _verification_source(tmp_path, fault="identity")
    conversion = Mock(side_effect=AssertionError("close converted before identity validation"))
    monkeypatch.setattr(research_product, "float", conversion, raising=False)
    with pytest.raises(ValueError, match="identity diverges"):
        research_product._verify_product_price_content(**source)
    conversion.assert_not_called()

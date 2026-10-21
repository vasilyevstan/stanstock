from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, Inexact, InvalidOperation, localcontext
from unittest.mock import Mock

import httpx
import pytest

from stanstock.data.providers import twelve_data
from stanstock.data.providers.exceptions import (
    ProviderBlockedError,
    ProviderConfigurationError,
    ProviderDataError,
    ProviderQuotaError,
    ProviderResponseError,
)
from stanstock.data.providers.http import HttpFetchResult

VALID_SERIES = {
    "meta": {
        "symbol": "AAPL",
        "interval": "1day",
        "currency": "USD",
        "exchange_timezone": "America/New_York",
        "exchange": "NASDAQ",
        "mic_code": "XNGS",
        "type": "Common Stock",
    },
    "values": [
        {
            "datetime": "2026-09-03",
            "open": "324.87",
            "high": "330.81",
            "low": "324.10",
            "close": "328.21",
            "volume": "37225800",
        },
        {
            "datetime": "2026-09-04",
            "open": "328.31",
            "high": "328.93",
            "low": "317.86",
            "close": "319.97",
            "volume": "39551800",
        },
    ],
    "status": "ok",
}

VALID_CATALOG = {
    "data": [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "currency": "USD",
            "exchange": "NASDAQ",
            "mic_code": "XNGS",
            "country": "United States",
            "type": "Common Stock",
            "figi_code": "BBG000B9Y5X2",
            "access": {
                "global": "Basic",
                "plan": "Basic",
                "plan_business": "Basic",
            },
        }
    ],
    "count": 1,
    "status": "ok",
}


def _result(
    payload: object,
    *,
    status_code: int = 200,
    content_type: str = "application/json; charset=utf-8",
    url: str = twelve_data.TIME_SERIES_URL,
) -> HttpFetchResult:
    return HttpFetchResult(
        status_code=status_code,
        headers=httpx.Headers({"content-type": content_type}),
        content=json.dumps(payload).encode(),
        url=url,
    )


def _price_parser_cases():
    row = {
        "datetime": "2026-09-04",
        "open": "2",
        "high": "3",
        "low": "1",
        "close": "2",
        "volume": "10",
    }
    values = (None, "", "1", "2", "3", "0", "-1", "NaN", "Infinity", "bad")
    cases = [
        ([{**row, "open": opening, "high": high, "low": low}], None, None)
        for opening, high, low in itertools.product(values, repeat=3)
    ]
    for field in row:
        cases.extend(
            ([{**row, field: value}], None, None)
            for value in (
                None,
                "",
                " \t",
                True,
                False,
                0,
                1.5,
                "sNaN",
                "1E-999",
                "1E+999",
                " 2.000 ",
            )
        )
    cases.extend(
        [
            ([], None, None),
            ([None], None, None),
            ([row, row], None, None),
            ([row, {**row, "datetime": "2026-09-03"}], None, None),
            ([row], date(2026, 9, 5), None),
            ([row], None, date(2026, 9, 3)),
            ([{**row, "open": "0", "volume": "bad"}], None, None),
            ([{**row, "high": "1", "low": "3"}], None, None),
            ([{**row, "low": "bad", "close": "bad"}], None, None),
            ([{**row, "datetime": "bad", "open": "bad"}], None, None),
        ]
    )
    return cases


def _price_error_chain(error):
    causes = []
    while error is not None:
        causes.append((type(error).__name__, str(error)))
        error = error.__cause__
    return causes


def _price_parser_characterization() -> tuple[str, int, int]:
    outcomes = []
    accepted = rejected = 0
    for rows, start, end in _price_parser_cases():
        raw = json.dumps({**VALID_SERIES, "values": rows}, sort_keys=True).encode()
        outcome = {"raw": raw.hex(), "start": start, "end": end}
        try:
            parsed = twelve_data.parse_daily_price_series(
                raw,
                symbol="AAPL",
                retrieved_at=datetime(2026, 9, 5, tzinfo=UTC),
                source_url="https://example.invalid/synthetic-prices",
                start_date=start,
                end_date=end,
            )
        except ProviderDataError as error:
            outcome["errors"] = _price_error_chain(error)
            rejected += 1
        else:
            outcome["series"] = asdict(parsed)
            accepted += 1
        outcomes.append(outcome)
    payload = json.dumps(outcomes, default=str, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest(), accepted, rejected


def test_verification_projection_matches_frozen_public_parser_corpus() -> None:
    for rows, start, end in _price_parser_cases():
        raw = json.dumps({**VALID_SERIES, "values": rows}, sort_keys=True).encode()
        arguments = {"symbol": "AAPL", "start_date": start, "end_date": end}
        try:
            public = twelve_data.parse_daily_price_series(
                raw,
                **arguments,
                retrieved_at=datetime(2026, 9, 5, tzinfo=UTC),
                source_url="https://example.invalid/synthetic-prices",
            )
        except ProviderDataError as expected:
            with pytest.raises(ProviderDataError) as actual:
                twelve_data._parse_daily_price_projection(raw, **arguments)
            assert _price_error_chain(actual.value) == _price_error_chain(expected)
        else:
            projection = twelve_data._parse_daily_price_projection(raw, **arguments)
            assert projection.values == tuple(
                (bar.trade_date, bar.close, bar.volume) for bar in public.bars
            )
            for field in (
                "currency",
                "exchange",
                "mic_code",
                "instrument_type",
                "exchange_timezone",
                "adjustment",
            ):
                assert getattr(projection, field) == getattr(public, field)


@pytest.mark.parametrize("field", ["currency", "exchange", "mic_code", "type", "exchange_timezone"])
@pytest.mark.parametrize("value", [None, "", {"vendor": "synthetic"}, 42])
def test_verification_projection_preserves_all_optional_metadata(field, value) -> None:
    raw = json.dumps({**VALID_SERIES, "meta": {**VALID_SERIES["meta"], field: value}}).encode()
    public = twelve_data.parse_daily_price_series(
        raw,
        symbol="AAPL",
        retrieved_at=datetime(2026, 9, 5, tzinfo=UTC),
        source_url="https://example.invalid/synthetic-prices",
    )
    projection = twelve_data._parse_daily_price_projection(raw, symbol="AAPL")
    attribute = "instrument_type" if field == "type" else field
    expected = None if value is None else str(value).strip() or None
    assert getattr(public, attribute) == getattr(projection, attribute) == expected


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        {**VALID_SERIES["meta"], "symbol": None},
        {**VALID_SERIES["meta"], "symbol": "OTHER"},
        {**VALID_SERIES["meta"], "interval": "1min"},
    ],
)
def test_verification_projection_preserves_envelope_error_chains(metadata) -> None:
    raw = json.dumps({**VALID_SERIES, "meta": metadata}).encode()
    with pytest.raises(ProviderDataError) as public:
        twelve_data.parse_daily_price_series(
            raw,
            symbol="AAPL",
            retrieved_at=datetime(2026, 9, 5, tzinfo=UTC),
            source_url="https://example.invalid/synthetic-prices",
        )
    with pytest.raises(ProviderDataError) as projection:
        twelve_data._parse_daily_price_projection(raw, symbol="AAPL")
    assert _price_error_chain(projection.value) == _price_error_chain(public.value)


def test_price_parser_preserves_complete_base_payloads_and_exact_errors() -> None:
    # Complete series and exception chains captured before edits on unchanged 8f2daad.
    assert _price_parser_characterization() == (
        "6eaced43582adbf6f33f14af07daebe6ebfdcba8ed6d0188d1c928e51261ef8d",
        95,
        981,
    )


def test_optional_decimal_converts_nonblank_text_once() -> None:
    class Text:
        calls = 0

        def __str__(self):
            self.calls += 1
            return " 2.000 "

    value = Text()
    assert twelve_data._optional_decimal(value, field="close", symbol="SYN") == Decimal("2.000")
    assert value.calls == 1


@pytest.mark.parametrize("trap", [False, True])
def test_reused_row_preserves_parser_corpus_errors_and_decimal_state(trap) -> None:
    def outcome(raw, start, end, *, public):
        with localcontext() as context:
            context.prec = 2
            context.rounding = "ROUND_UP"
            context.traps[InvalidOperation] = trap
            context.clear_flags()
            context.flags[Inexact] = True
            try:
                if public:
                    series = twelve_data.parse_daily_price_series(
                        raw,
                        symbol="AAPL",
                        retrieved_at=datetime(2026, 9, 5, tzinfo=UTC),
                        source_url="https://example.invalid/prices",
                        start_date=start,
                        end_date=end,
                    )
                    values = tuple(
                        (bar.trade_date, bar.close.as_tuple(), bar.volume) for bar in series.bars
                    )
                else:
                    series = twelve_data._parse_daily_price_projection(
                        raw, symbol="AAPL", start_date=start, end_date=end
                    )
                    values = tuple(
                        (day, close.as_tuple(), volume) for day, close, volume in series.values
                    )
            except ProviderDataError as error:
                result = _price_error_chain(error)
            else:
                result = (values, series.currency, series.exchange, series.mic_code)
            return result, dict(context.flags), dict(context.traps), context.rounding

    for rows, start, end in _price_parser_cases():
        with twelve_data._reuse_admitted_price_rows():
            raw = json.dumps({**VALID_SERIES, "values": rows}).encode()
            expected = outcome(raw, start, end, public=True)
            # Successful cases are hit twice; failures can never become hits.
            assert outcome(raw, start, end, public=False) == expected
            assert outcome(raw, start, end, public=False) == expected
    with twelve_data._reuse_admitted_price_rows():
        # Earlier non-finite admission must not touch a later invalid decimal.
        raw = json.dumps(
            {
                **VALID_SERIES,
                "values": [{**VALID_SERIES["values"][0], "open": "NaN", "high": "bad"}],
            }
        ).encode()
        assert outcome(raw, None, None, public=False) == outcome(raw, None, None, public=True)
    assert twelve_data._ROW_ADMISSIONS.get() is None


def test_row_reuse_keeps_scalar_types_and_rechecks_window_and_duplicates() -> None:
    row = {**VALID_SERIES["values"][0], "volume": 1, "close": 328}
    raw = json.dumps({**VALID_SERIES, "values": [row]}).encode()
    with twelve_data._reuse_admitted_price_rows():
        first = twelve_data._parse_daily_price_projection(raw, symbol="AAPL")
        bad_volume = json.dumps({**VALID_SERIES, "values": [{**row, "volume": 1.0}]}).encode()
        with pytest.raises(ProviderDataError, match="invalid volume"):
            twelve_data._parse_daily_price_projection(bad_volume, symbol="AAPL")
        with pytest.raises(ProviderDataError, match="after requested end"):
            twelve_data._parse_daily_price_projection(raw, symbol="AAPL", end_date=date(2026, 9, 2))
        duplicate = json.dumps({**VALID_SERIES, "values": [row, row]}).encode()
        with pytest.raises(ProviderDataError, match="duplicate date"):
            twelve_data._parse_daily_price_projection(duplicate, symbol="AAPL")
        changed = json.dumps({**VALID_SERIES, "values": [{**row, "close": 328.0}]}).encode()
        second = twelve_data._parse_daily_price_projection(changed, symbol="AAPL")
        assert first.values[0][1].as_tuple() != second.values[0][1].as_tuple()
    assert twelve_data._ROW_ADMISSIONS.get() is None


def test_nonoverlapping_rows_are_readmitted_and_scope_is_bounded(monkeypatch) -> None:
    admission = Mock(wraps=twelve_data._required_decimal)
    monkeypatch.setattr(twelve_data, "_required_decimal", admission)
    with twelve_data._reuse_admitted_price_rows():
        for version in range(4):
            raw = json.dumps(
                {
                    **VALID_SERIES,
                    "values": [
                        {**row, "volume": str(100 + version)} for row in VALID_SERIES["values"]
                    ],
                }
            ).encode()
            for _ in range(2):
                twelve_data._parse_daily_price_projection(raw, symbol="AAPL")
            assert admission.call_count == (2 if version == 0 else 2 + 4 * version)
            resident = twelve_data._ROW_ADMISSIONS.get()["AAPL"]
            if version == 0:
                assert len(resident) == 2
            else:
                assert resident is None
    assert twelve_data._ROW_ADMISSIONS.get() is None
    with pytest.raises(RuntimeError, match="synthetic"), twelve_data._reuse_admitted_price_rows():
        twelve_data._parse_daily_price_projection(raw, symbol="AAPL")
        raise RuntimeError("synthetic")
    assert admission.call_count == 16
    assert twelve_data._ROW_ADMISSIONS.get() is None


@pytest.mark.parametrize(("cohorts", "retained"), [(3, 5), (26, 1758)])
def test_rolling_oldest_first_rows_reuse_common_history(monkeypatch, cohorts, retained) -> None:
    # Each older vintage adds a leading date, not a changed price level.
    # Values must be keyed by absolute dates, not each window's row offset.
    rows = [
        {
            "datetime": (date(2020, 1, 1) + timedelta(days=offset)).isoformat(),
            "open": str(64 + offset / 64),
            "high": str(65 + offset / 64),
            "low": str(63 + offset / 64),
            "close": str(64 + offset / 64),
            "volume": str(1_000_000 + offset),
        }
        for offset in range(retained + cohorts - 1)
    ]
    documents = [
        json.dumps({**VALID_SERIES, "values": rows[start : start + retained]}).encode()
        for start in reversed(range(cohorts))
    ]
    expected = [
        tuple(
            (bar.trade_date, bar.close, bar.volume)
            for bar in twelve_data.parse_daily_price_series(
                raw,
                symbol="AAPL",
                retrieved_at=datetime(2026, 9, 5, tzinfo=UTC),
                source_url="https://example.invalid/rolling",
            ).bars
        )
        for raw in documents
    ]
    calls = Counter()
    original_admission = twelve_data._required_decimal
    original_comparison = twelve_data._same_admitted_row

    def admit(*args, **kwargs):
        calls["admissions"] += 1
        return original_admission(*args, **kwargs)

    def compare(*args, **kwargs):
        matched = original_comparison(*args, **kwargs)
        calls["comparisons"] += 1
        calls["reuses"] += matched
        return matched

    monkeypatch.setattr(twelve_data, "_required_decimal", admit)
    monkeypatch.setattr(twelve_data, "_same_admitted_row", compare)
    residents = []
    with twelve_data._reuse_admitted_price_rows():
        for raw, values in zip(documents, expected, strict=True):
            parsed = twelve_data._parse_daily_price_projection(
                raw, symbol="AAPL", start_date=values[0][0], end_date=values[-1][0]
            )
            assert parsed.values == values
            resident = twelve_data._ROW_ADMISSIONS.get()["AAPL"]
            residents.append(None if resident is None else len(resident))
    print(f"ROLLING {cohorts=} {retained=} {dict(calls)} {residents=}")
    assert calls["admissions"] == retained + cohorts - 1
    assert calls["comparisons"] == calls["reuses"] == (cohorts - 1) * (retained - 1)
    assert residents == [retained] * cohorts
    assert twelve_data._ROW_ADMISSIONS.get() is None


def test_changed_leading_row_does_not_hide_later_exact_matches(monkeypatch) -> None:
    admission = Mock(wraps=twelve_data._required_decimal)
    monkeypatch.setattr(twelve_data, "_required_decimal", admission)
    with twelve_data._reuse_admitted_price_rows():
        twelve_data._parse_daily_price_projection(json.dumps(VALID_SERIES).encode(), symbol="AAPL")
        changed = {
            **VALID_SERIES,
            "values": [
                {**VALID_SERIES["values"][0], "volume": "123"},
                VALID_SERIES["values"][1],
            ],
        }
        result = twelve_data._parse_daily_price_projection(
            json.dumps(changed).encode(), symbol="AAPL"
        )
        assert result.values[0][2] == 123
        assert admission.call_count == 3
        assert len(twelve_data._ROW_ADMISSIONS.get()["AAPL"]) == 2


@pytest.mark.parametrize("fault", ["nonfinite", "volume", "duplicate", "future"])
def test_rolling_prefix_then_error_never_publishes_partial_rows(fault) -> None:
    prefix = {**VALID_SERIES["values"][0], "datetime": "2026-09-02"}
    last = dict(VALID_SERIES["values"][1])
    end = date(2026, 9, 4)
    if fault == "nonfinite":
        last.update(open="NaN", high="bad")
        reason = "non-finite 'open'"
    elif fault == "volume":
        last.update(open="0", volume="bad")
        reason = "invalid volume"
    elif fault == "duplicate":
        last["datetime"] = "2026-09-03"
        reason = "duplicate date"
    else:
        end = date(2026, 9, 3)
        reason = "after requested end"
    raw = json.dumps({**VALID_SERIES, "values": [prefix, VALID_SERIES["values"][0], last]}).encode()
    with localcontext() as context, twelve_data._reuse_admitted_price_rows():
        context.clear_flags()
        context.traps[InvalidOperation] = True
        twelve_data._parse_daily_price_projection(json.dumps(VALID_SERIES).encode(), symbol="AAPL")
        admitted = twelve_data._ROW_ADMISSIONS.get()["AAPL"]
        with pytest.raises(ProviderDataError, match=reason):
            twelve_data._parse_daily_price_projection(raw, symbol="AAPL", end_date=end)
        assert not context.flags[InvalidOperation]
        assert twelve_data._ROW_ADMISSIONS.get()["AAPL"] is admitted


def test_untrusted_payload_objects_keep_original_coercion_and_get_order() -> None:
    calls = []

    class Text:
        text = "328.21"

        def __str__(self):
            calls.append("str")
            return self.text

        def __eq__(self, other):
            pytest.fail("custom equality must not be used for admission reuse")

        def __hash__(self):
            pytest.fail("custom hashing must not be used for admission reuse")

    class Row(dict):
        def get(self, key, default=None):
            calls.append(key)
            return super().get(key, default)

    close = Text()
    row = Row({**VALID_SERIES["values"][0], "close": close})
    payload = {**VALID_SERIES, "values": [row]}
    with twelve_data._reuse_admitted_price_rows():
        for _ in range(2):
            calls.clear()
            twelve_data._parse_price_payload(
                payload,
                symbol="AAPL",
                start_date=None,
                end_date=None,
                adjustment="splits",
                projection=True,
            )
            assert calls == ["datetime", "open", "high", "low", "close", "str", "volume"]
            assert twelve_data._ROW_ADMISSIONS.get() == {}
        close.text = "NaN"
        calls.clear()
        with pytest.raises(ProviderDataError, match="non-finite 'close'"):
            twelve_data._parse_price_payload(
                payload,
                symbol="AAPL",
                start_date=None,
                end_date=None,
                adjustment="splits",
                projection=True,
            )
        assert calls == ["datetime", "open", "high", "low", "close", "str"]
        assert twelve_data._ROW_ADMISSIONS.get() == {}


def test_invalid_leading_row_does_not_publish_reuse_opt_out() -> None:
    raw = json.dumps(VALID_SERIES).encode()
    with twelve_data._reuse_admitted_price_rows():
        twelve_data._parse_daily_price_projection(raw, symbol="AAPL")
        admitted = twelve_data._ROW_ADMISSIONS.get()["AAPL"]
        invalid = json.dumps(
            {
                **VALID_SERIES,
                "values": [{**VALID_SERIES["values"][0], "volume": "-1"}],
            }
        ).encode()
        with pytest.raises(ProviderDataError, match="negative volume"):
            twelve_data._parse_daily_price_projection(invalid, symbol="AAPL")
        assert twelve_data._ROW_ADMISSIONS.get()["AAPL"] is admitted
        twelve_data._parse_daily_price_projection(raw, symbol="AAPL")
        assert twelve_data._ROW_ADMISSIONS.get()["AAPL"] is not None


def test_parses_split_adjusted_daily_series_and_uses_header_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_fetch(*args: object, **kwargs: object) -> HttpFetchResult:
        captured.update(kwargs)
        return _result(VALID_SERIES)

    monkeypatch.setattr(twelve_data, "fetch", fake_fetch)

    series = twelve_data.fetch_daily_price_series(
        "aapl",
        start_date=date(2026, 9, 3),
        end_date=date(2026, 9, 4),
        api_key="private-test-key",
    )

    assert series.provider == "twelve_data"
    assert series.symbol == "AAPL"
    assert series.currency == "USD"
    assert series.exchange == "NASDAQ"
    assert series.mic_code == "XNGS"
    assert series.instrument_type == "Common Stock"
    assert series.adjustment == "splits"
    assert [bar.trade_date for bar in series.bars] == [
        date(2026, 9, 3),
        date(2026, 9, 4),
    ]
    assert str(series.bars[-1].close) == "319.97"
    assert series.bars[-1].volume == 39_551_800
    assert captured["headers"] == {
        "Accept": "application/json",
        "Authorization": "apikey private-test-key",
    }
    assert captured["params"]["end_date"] == "2026-09-05"
    assert "private-test-key" not in series.source_url


def test_missing_or_demo_key_is_rejected_for_live_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(twelve_data.API_KEY_ENV, raising=False)
    monkeypatch.setattr(twelve_data, "read_twelve_data_api_key", lambda: None)

    with pytest.raises(ProviderConfigurationError, match="is required"):
        twelve_data.resolve_api_key()
    with pytest.raises(ProviderConfigurationError, match="demo key"):
        twelve_data.resolve_api_key("demo")

    assert twelve_data.resolve_api_key("demo", allow_demo=True) == "demo"


def test_api_key_resolution_uses_environment_before_keychain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(twelve_data.API_KEY_ENV, "environment-key")
    monkeypatch.setattr(
        twelve_data,
        "read_twelve_data_api_key",
        lambda: pytest.fail("keychain should not be read when the environment is set"),
    )

    assert twelve_data.resolve_api_key() == "environment-key"


def test_api_key_resolution_falls_back_to_macos_keychain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(twelve_data.API_KEY_ENV, raising=False)
    monkeypatch.setattr(twelve_data, "read_twelve_data_api_key", lambda: "keychain-key")

    assert twelve_data.resolve_api_key() == "keychain-key"


def test_noninteractive_api_key_resolution_never_reads_keychain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(twelve_data.API_KEY_ENV, raising=False)
    monkeypatch.setenv(twelve_data.DISABLE_KEYCHAIN_ENV, "1")
    monkeypatch.setattr(
        twelve_data,
        "read_twelve_data_api_key",
        lambda: pytest.fail("disabled keychain must not be read"),
    )

    with pytest.raises(ProviderConfigurationError, match="is required"):
        twelve_data.resolve_api_key()


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (
            {"status": "error", "code": 401, "message": "API key is invalid"},
            ProviderConfigurationError,
        ),
        (
            {"status": "error", "code": 429, "message": "API credits exhausted"},
            ProviderQuotaError,
        ),
        (
            {
                "status": "error",
                "code": 403,
                "message": "This symbol is not available with your plan",
            },
            ProviderBlockedError,
        ),
    ],
)
def test_classifies_documented_api_errors(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    error: type[Exception],
) -> None:
    monkeypatch.setattr(twelve_data, "fetch", lambda *args, **kwargs: _result(payload))

    with pytest.raises(error):
        twelve_data.fetch_daily_price_series("AAPL", api_key="test-key")


def test_rejects_malformed_or_inconsistent_daily_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        **VALID_SERIES,
        "values": [
            {
                "datetime": "2026-09-04",
                "open": "328.31",
                "high": "300.00",
                "low": "317.86",
                "close": "319.97",
                "volume": "39551800",
            }
        ],
    }
    monkeypatch.setattr(twelve_data, "fetch", lambda *args, **kwargs: _result(payload))

    with pytest.raises(ProviderDataError, match="high was below"):
        twelve_data.fetch_daily_price_series("AAPL", api_key="test-key")


def test_rejects_dates_outside_requested_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        twelve_data,
        "fetch",
        lambda *args, **kwargs: _result(VALID_SERIES),
    )

    with pytest.raises(ProviderResponseError, match="before requested start"):
        twelve_data.fetch_daily_price_series(
            "AAPL",
            start_date=date(2026, 9, 4),
            end_date=date(2026, 9, 4),
            api_key="test-key",
        )


def test_parses_us_stock_reference_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        twelve_data,
        "fetch",
        lambda *args, **kwargs: _result(VALID_CATALOG, url=twelve_data.STOCKS_URL),
    )

    catalog = twelve_data.fetch_stock_catalog(exchange="nasdaq", api_key="test-key")

    assert catalog.provider == "twelve_data"
    assert catalog.exchange == "NASDAQ"
    assert catalog.count == 1
    reference = catalog.references[0]
    assert reference.symbol == "AAPL"
    assert reference.name == "Apple Inc."
    assert reference.currency == "USD"
    assert reference.mic_code == "XNGS"
    assert reference.access_plan == "Basic"

    persisted_references, persisted_count = twelve_data.parse_stock_catalog_references(
        json.dumps(VALID_CATALOG).encode(),
        exchange="nasdaq",
    )
    assert persisted_references == catalog.references
    assert persisted_count == catalog.count


def test_catalog_filter_ignores_malformed_unrequested_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        **VALID_CATALOG,
        "data": [
            *VALID_CATALOG["data"],
            {
                "symbol": "UNUSED",
                "name": None,
                "currency": "USD",
                "exchange": "NASDAQ",
                "mic_code": "XNGS",
                "country": "United States",
                "type": "Common Stock",
            },
        ],
        "count": 2,
    }
    monkeypatch.setattr(
        twelve_data,
        "fetch",
        lambda *args, **kwargs: _result(payload, url=twelve_data.STOCKS_URL),
    )

    catalog = twelve_data.fetch_stock_catalog(
        exchange="nasdaq",
        required_symbols={"AAPL"},
        api_key="test-key",
    )

    assert [reference.symbol for reference in catalog.references] == ["AAPL"]
    assert catalog.count == 2


@pytest.mark.parametrize(
    "payload",
    [
        {**VALID_CATALOG, "count": 2},
        {key: value for key, value in VALID_CATALOG.items() if key != "count"},
    ],
)
def test_complete_persisted_catalog_requires_declared_exact_row_count(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ProviderResponseError, match="invalid count"):
        twelve_data.parse_stock_catalog_references(
            json.dumps(payload).encode(),
            exchange="NASDAQ",
            require_complete=True,
        )


def test_complete_catalog_filters_before_normalizing_unrelated_rows() -> None:
    payload = {
        **VALID_CATALOG,
        "data": [
            *VALID_CATALOG["data"],
            {
                "symbol": "UNRELATED",
                "name": None,
                "currency": "USD",
                "exchange": "NASDAQ",
                "mic_code": "XNGS",
                "country": "United States",
                "type": "Common Stock",
            },
        ],
        "count": 2,
    }

    references, count = twelve_data.parse_stock_catalog_references(
        json.dumps(payload).encode(),
        exchange="NASDAQ",
        required_symbols={"aapl"},
        require_complete=True,
    )

    assert [reference.symbol for reference in references] == ["AAPL"]
    assert count == 2


def test_complete_catalog_strictly_validates_matching_rows() -> None:
    payload = {
        **VALID_CATALOG,
        "data": [{**VALID_CATALOG["data"][0], "name": None}],
    }

    with pytest.raises(ProviderResponseError, match="had no usable 'name'"):
        twelve_data.parse_stock_catalog_references(
            json.dumps(payload).encode(),
            exchange="NASDAQ",
            required_symbols={"AAPL"},
            require_complete=True,
        )


def test_catalog_filter_still_rejects_malformed_required_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        **VALID_CATALOG,
        "data": [{**VALID_CATALOG["data"][0], "name": None}],
    }
    monkeypatch.setattr(
        twelve_data,
        "fetch",
        lambda *args, **kwargs: _result(payload, url=twelve_data.STOCKS_URL),
    )

    with pytest.raises(ProviderResponseError, match="had no usable 'name'"):
        twelve_data.fetch_stock_catalog(
            exchange="nasdaq",
            required_symbols={"AAPL"},
            api_key="test-key",
        )


def test_http_rate_limit_is_not_misreported_as_empty_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        twelve_data,
        "fetch",
        lambda *args, **kwargs: _result({}, status_code=429),
    )

    with pytest.raises(ProviderQuotaError, match="credit limit"):
        twelve_data.fetch_daily_price_series("AAPL", api_key="test-key")


def test_http_failure_is_not_downgraded_to_instrument_data_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        twelve_data,
        "fetch",
        lambda *args, **kwargs: _result({}, status_code=502),
    )

    with pytest.raises(ProviderResponseError, match="HTTP 502") as error:
        twelve_data.fetch_daily_price_series("AAPL", api_key="test-key")

    assert not isinstance(error.value, ProviderDataError)

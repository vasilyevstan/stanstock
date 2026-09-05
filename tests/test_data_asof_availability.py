from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from django.core.management import call_command

from stanstock.data.asof import AsOfData, PriceFrameSchemaError
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import DataAsset, FundamentalFact, FxRate, Listing

pytestmark = pytest.mark.django_db


def test_price_history_not_visible_before_its_available_at() -> None:
    call_command("seed_demo")
    asset = DataAsset.objects.get(
        provider="synthetic_demo", kind="price_history", subject="ZZUS001"
    )

    early = AsOfData(asset.available_at - timedelta(seconds=1))
    with pytest.raises(DataAsset.DoesNotExist):
        early.latest_asset(provider="synthetic_demo", kind="price_history", subject="ZZUS001")


def test_price_history_visible_at_its_available_at() -> None:
    call_command("seed_demo")
    asset = DataAsset.objects.get(
        provider="synthetic_demo", kind="price_history", subject="ZZUS001"
    )

    on_time = AsOfData(asset.available_at)
    frame = on_time.price_frame(provider="synthetic_demo", subject="ZZUS001")

    assert frame.height > 1600
    assert frame["date"].max() == asset.period_end


def test_fundamental_amendment_hidden_until_its_own_available_at() -> None:
    call_command("seed_demo")
    listing = Listing.objects.get(ticker="ZZUS001")
    company = listing.security.company
    original = FundamentalFact.objects.get(
        company=company, concept="NetIncomeLoss", fiscal_year=2025, is_amendment=False
    )
    amendment = FundamentalFact.objects.get(
        company=company, concept="NetIncomeLoss", fiscal_year=2025, is_amendment=True
    )
    assert amendment.available_at > original.available_at

    before = AsOfData(amendment.available_at - timedelta(seconds=1))
    facts_before = before.fundamental_facts(company_id=company.id, concepts=["NetIncomeLoss"])
    ids_before = {fact.pk for fact in facts_before}
    assert original.pk in ids_before
    assert amendment.pk not in ids_before

    after = AsOfData(amendment.available_at)
    facts_after = after.fundamental_facts(company_id=company.id, concepts=["NetIncomeLoss"])
    ids_after = {fact.pk for fact in facts_after}
    assert original.pk in ids_after
    assert amendment.pk in ids_after


def test_fundamental_facts_not_visible_before_filing_delay_elapses() -> None:
    call_command("seed_demo")
    listing = Listing.objects.get(ticker="ZZUS001")
    company = listing.security.company
    fact_2023 = FundamentalFact.objects.get(
        company=company, concept="Revenue", fiscal_year=2023, is_amendment=False
    )
    assert fact_2023.filed_at is not None

    before_filing = AsOfData(fact_2023.filed_at)
    facts = before_filing.fundamental_facts(company_id=company.id, concepts=["Revenue"])
    period_ends = {fact.period_end for fact in facts}
    # the 6-hour post-filing delay means the fact filed at exactly filed_at
    # (with available_at = filed_at + 6h) is not yet visible.
    assert fact_2023.period_end not in period_ends


def test_fx_rate_hidden_before_its_publication_convention_time() -> None:
    call_command("seed_demo")
    rate = (
        FxRate.objects.filter(base_currency="EUR", quote_currency="USD")
        .order_by("observation_date")
        .first()
    )
    assert rate is not None

    before = AsOfData(rate.available_at - timedelta(seconds=1))
    observation_dates_before = {
        r.observation_date for r in before.fx_rates(base_currency="EUR", quote_currency="USD")
    }
    assert rate.observation_date not in observation_dates_before

    at_publication = AsOfData(rate.available_at)
    observation_dates_after = {
        r.observation_date
        for r in at_publication.fx_rates(base_currency="EUR", quote_currency="USD")
    }
    assert rate.observation_date in observation_dates_after


def test_fx_rates_ordered_by_observation_date() -> None:
    call_command("seed_demo")
    last_rate = (
        FxRate.objects.filter(base_currency="EUR", quote_currency="USD")
        .order_by("-observation_date")
        .first()
    )
    assert last_rate is not None

    asof = AsOfData(last_rate.available_at)
    rates = list(asof.fx_rates(base_currency="EUR", quote_currency="USD"))

    observation_dates = [r.observation_date for r in rates]
    assert observation_dates == sorted(observation_dates)


def _register_price_asset(
    tmp_path: Path,
    *,
    subject: str,
    frame: pl.DataFrame,
    available_at: datetime,
) -> tuple[AssetStore, DataAsset]:
    """Write a price_history bundle directly, bypassing seed_demo.

    Lets tests construct the specific "eligible asset whose bytes still
    contain future-dated rows" scenario that `through_date` clipping guards
    against, independent of seed_demo's own (always-fully-available) bundle
    shape.
    """
    store = AssetStore(root=tmp_path)
    stored = store.write_frame(f"price_history/{subject}.parquet", frame)
    dates = frame["date"].to_list()
    asset = register_asset(
        provider="asof_hardening_test",
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
        period_start=min(dates),
        period_end=max(dates),
    )
    return store, asset


def test_price_frame_clips_future_dated_rows_even_from_an_eligible_asset(
    tmp_path: Path,
) -> None:
    """Regression test: asset-level eligibility alone must not leak future rows.

    The DataAsset is eligible at ``available_at`` (2024-01-03), but its
    Parquet bytes contain 10 days of rows through 2024-01-10 -- exactly the
    "eligible asset, future-dated frame contents" scenario the hardening
    guards against. Without physically clipping the frame, `price_frame`
    would return all 10 rows even though only 3 days had actually occurred
    by the decision time.
    """
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    frame = pl.DataFrame({"date": dates, "close": [float(i) for i in range(10)]})
    available_at = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)
    store, _asset = _register_price_asset(
        tmp_path, subject="FUTUREROWS", frame=frame, available_at=available_at
    )

    result = AsOfData(available_at, store=store).price_frame(
        provider="asof_hardening_test", subject="FUTUREROWS"
    )

    assert result.height == 3
    assert result["date"].max() == date(2024, 1, 3)
    assert set(result["date"].to_list()) == {date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)}


@pytest.mark.parametrize(
    ("through_date", "expected_dates"),
    [
        (date(2023, 12, 31), []),
        (date(2024, 1, 1), [date(2024, 1, 1)]),
        (date(2024, 1, 5), [date(2024, 1, 1 + i) for i in range(5)]),
        (date(2024, 1, 10), [date(2024, 1, 1 + i) for i in range(10)]),
    ],
)
def test_price_frame_through_date_never_returns_rows_past_cutoff(
    tmp_path: Path,
    through_date: date,
    expected_dates: list[date],
) -> None:
    """Property: for any cutoff, the result is exactly the rows with date <= cutoff."""
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    frame = pl.DataFrame({"date": dates, "close": [float(i) for i in range(10)]})
    available_at = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
    store, _asset = _register_price_asset(
        tmp_path, subject="PROPERTYCLIP", frame=frame, available_at=available_at
    )

    result = AsOfData(available_at, store=store).price_frame(
        provider="asof_hardening_test", subject="PROPERTYCLIP", through_date=through_date
    )

    assert result["date"].to_list() == expected_dates
    assert all(d <= through_date for d in result["date"].to_list())


def test_price_frame_rejects_cutoff_after_decision_date(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 10), date(2024, 1, 11)],
            "close": [10.0, 11.0],
        }
    )
    available_at = datetime(2024, 1, 10, 12, tzinfo=UTC)
    store, _asset = _register_price_asset(
        tmp_path,
        subject="FUTURECUTOFF",
        frame=frame,
        available_at=available_at,
    )

    with pytest.raises(ValueError, match="cannot be after the as-of decision date"):
        AsOfData(available_at, store=store).price_frame(
            provider="asof_hardening_test",
            subject="FUTURECUTOFF",
            through_date=date(2024, 1, 11),
        )


def test_price_frame_returns_dates_in_ascending_order(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 2)],
            "close": [3.0, 1.0, 2.0],
        }
    )
    available_at = datetime(2024, 1, 3, 12, tzinfo=UTC)
    store, _asset = _register_price_asset(
        tmp_path,
        subject="REVERSED",
        frame=frame,
        available_at=available_at,
    )

    result = AsOfData(available_at, store=store).price_frame(
        provider="asof_hardening_test",
        subject="REVERSED",
    )

    assert result["date"].to_list() == [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]


def test_price_frame_through_date_defaults_to_decision_time_date(tmp_path: Path) -> None:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    frame = pl.DataFrame({"date": dates, "close": [float(i) for i in range(10)]})
    available_at = datetime(2024, 1, 1, tzinfo=UTC)
    store, _asset = _register_price_asset(
        tmp_path, subject="DEFAULTCUTOFF", frame=frame, available_at=available_at
    )
    decision_time = datetime(2024, 1, 4, 9, 0, tzinfo=UTC)

    result = AsOfData(decision_time, store=store).price_frame(
        provider="asof_hardening_test", subject="DEFAULTCUTOFF"
    )

    assert result["date"].max() == date(2024, 1, 4)


def test_price_frame_missing_date_column_fails_explicitly(tmp_path: Path) -> None:
    frame = pl.DataFrame({"close": [1.0, 2.0]})
    available_at = datetime(2024, 1, 1, tzinfo=UTC)
    store = AssetStore(root=tmp_path)
    stored = store.write_frame("price_history/NODATECOL.parquet", frame)
    register_asset(
        provider="asof_hardening_test",
        kind="price_history",
        subject="NODATECOL",
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )

    with pytest.raises(PriceFrameSchemaError, match="no 'date' column"):
        AsOfData(available_at, store=store).price_frame(
            provider="asof_hardening_test", subject="NODATECOL"
        )


def test_price_frame_unsupported_date_dtype_fails_explicitly(tmp_path: Path) -> None:
    frame = pl.DataFrame({"date": [20240101, 20240102], "close": [1.0, 2.0]})
    available_at = datetime(2024, 1, 2, tzinfo=UTC)
    store = AssetStore(root=tmp_path)
    stored = store.write_frame("price_history/BADDTYPE.parquet", frame)
    register_asset(
        provider="asof_hardening_test",
        kind="price_history",
        subject="BADDTYPE",
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )

    with pytest.raises(PriceFrameSchemaError, match="unsupported dtype"):
        AsOfData(available_at, store=store).price_frame(
            provider="asof_hardening_test", subject="BADDTYPE"
        )


def test_price_frame_accepts_iso_string_date_column(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {"date": ["2024-01-01", "2024-01-02", "2024-01-03"], "close": [1.0, 2.0, 3.0]}
    )
    available_at = datetime(2024, 1, 1, tzinfo=UTC)
    store = AssetStore(root=tmp_path)
    stored = store.write_frame("price_history/ISOSTRINGS.parquet", frame)
    register_asset(
        provider="asof_hardening_test",
        kind="price_history",
        subject="ISOSTRINGS",
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )

    result = AsOfData(datetime(2024, 1, 2, tzinfo=UTC), store=store).price_frame(
        provider="asof_hardening_test", subject="ISOSTRINGS"
    )

    assert result.height == 2
    # The frame's on-disk `date` column is string-typed, but the returned
    # frame must normalize it to `pl.Date` so downstream consumers can
    # compare it against plain Python `date` values without re-parsing.
    assert result.schema["date"] == pl.Date
    assert result["date"].to_list() == [date(2024, 1, 1), date(2024, 1, 2)]


def test_price_frame_accepts_datetime_date_column(tmp_path: Path) -> None:
    timestamps = [
        datetime(2024, 1, 1, 16, 0, tzinfo=UTC),
        datetime(2024, 1, 2, 16, 0, tzinfo=UTC),
        datetime(2024, 1, 3, 16, 0, tzinfo=UTC),
    ]
    frame = pl.DataFrame({"date": timestamps, "close": [1.0, 2.0, 3.0]})
    available_at = datetime(2024, 1, 1, tzinfo=UTC)
    store = AssetStore(root=tmp_path)
    stored = store.write_frame("price_history/DATETIMECOL.parquet", frame)
    register_asset(
        provider="asof_hardening_test",
        kind="price_history",
        subject="DATETIMECOL",
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )

    result = AsOfData(datetime(2024, 1, 2, tzinfo=UTC), store=store).price_frame(
        provider="asof_hardening_test", subject="DATETIMECOL"
    )

    assert result.height == 2
    # The frame's on-disk `date` column is Datetime-typed, but the returned
    # frame must normalize it to `pl.Date` so downstream consumers can
    # compare it against plain Python `date` values without re-parsing.
    assert result.schema["date"] == pl.Date
    assert result["date"].to_list() == [date(2024, 1, 1), date(2024, 1, 2)]


def test_price_frame_unparseable_string_dates_fail_explicitly(tmp_path: Path) -> None:
    frame = pl.DataFrame({"date": ["not-a-date", "also-not"], "close": [1.0, 2.0]})
    available_at = datetime(2024, 1, 1, tzinfo=UTC)
    store = AssetStore(root=tmp_path)
    stored = store.write_frame("price_history/BADSTRINGS.parquet", frame)
    register_asset(
        provider="asof_hardening_test",
        kind="price_history",
        subject="BADSTRINGS",
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )

    with pytest.raises(PriceFrameSchemaError, match="could not be parsed"):
        AsOfData(available_at, store=store).price_frame(
            provider="asof_hardening_test", subject="BADSTRINGS"
        )

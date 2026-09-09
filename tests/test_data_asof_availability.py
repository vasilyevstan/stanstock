from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from django.core.management import call_command

from stanstock.data.asof import (
    AsOfData,
    PriceFrameChecksumMismatchError,
    PriceFrameRead,
    PriceFrameSchemaError,
)
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.models import (
    CompanyClassificationObservation,
    DataAsset,
    FundamentalFact,
    FxRate,
    Listing,
)

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


def test_company_classification_respects_source_asset_availability() -> None:
    call_command("seed_demo")
    listing = Listing.objects.get(ticker="ZZUS001")
    source_asset = DataAsset.objects.filter(
        provider="synthetic_demo",
        kind="fundamentals",
        subject__startswith="ZZUS001:",
    ).first()
    assert source_asset is not None
    observation = CompanyClassificationObservation.objects.create(
        company=listing.security.company,
        provider="sec",
        scheme="sec_sic",
        code="3571",
        observed_at=source_asset.available_at,
        available_at=source_asset.available_at,
        source_asset=source_asset,
    )

    before = AsOfData(observation.available_at - timedelta(seconds=1))
    assert (
        list(
            before.company_classifications(
                company_id=listing.security.company_id,
                scheme="sec_sic",
            )
        )
        == []
    )

    at_observation = AsOfData(observation.available_at)
    assert list(
        at_observation.company_classifications(
            company_id=listing.security.company_id,
            scheme="sec_sic",
        )
    ) == [observation]


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


def test_unfiltered_fx_rate_read_still_respects_availability() -> None:
    """A pair-agnostic read must not become an escape hatch around as-of rules.

    Cross-rate derivation cannot name the pivot pair up front, so `fx_rates`
    accepts no filters at all. That widened read must still exclude every
    vintage whose own availability -- or whose source asset's availability --
    is after the decision time.
    """
    call_command("seed_demo")
    first_rate = (
        FxRate.objects.filter(base_currency="EUR", quote_currency="USD")
        .order_by("available_at")
        .first()
    )
    assert first_rate is not None

    before = AsOfData(first_rate.available_at - timedelta(seconds=1))
    assert list(before.fx_rates()) == []

    at_publication = AsOfData(first_rate.available_at)
    visible = list(at_publication.fx_rates())
    assert visible
    assert {rate.quote_currency for rate in visible} == {"USD", "GBP"}
    assert all(rate.available_at <= first_rate.available_at for rate in visible)
    assert all(rate.source_asset.available_at <= first_rate.available_at for rate in visible)


def test_fx_rates_observation_window_never_widens_availability() -> None:
    call_command("seed_demo")
    first_rate = (
        FxRate.objects.filter(base_currency="EUR", quote_currency="USD")
        .order_by("available_at")
        .first()
    )
    assert first_rate is not None

    asof = AsOfData(first_rate.available_at)
    windowed = list(
        asof.fx_rates(
            observation_start=first_rate.observation_date,
            observation_end=first_rate.observation_date,
        )
    )
    assert {rate.observation_date for rate in windowed} == {first_rate.observation_date}
    assert all(rate.available_at <= first_rate.available_at for rate in windowed)


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
    dates = [value for value in frame["date"].to_list() if value is not None]
    asset = register_asset(
        provider="asof_hardening_test",
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
        period_start=min(dates) if dates else None,
        period_end=max(dates) if dates else None,
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


# ---------------------------------------------------------------------------
# `price_frame_with_diagnostics` -- null session-date counting.
#
# `_clip_to_through_date`'s `date <= through_date` filter silently drops a
# null-dated row exactly like it drops a future one: neither raises. Only a
# *future* row is a normal, valid clip; a null session identity could never
# be established at all, and `invalid_session_date_rows` exists so a caller
# can tell the two apart instead of reading a shorter-but-plausible history.
# ---------------------------------------------------------------------------


def _diagnostics_column(dates: list[date | None], *, date_kind: str) -> pl.Series:
    """Build a `date` column of the requested dtype, preserving ``None`` entries."""
    if date_kind == "date":
        return pl.Series("date", dates, dtype=pl.Date)
    if date_kind == "datetime":
        values = [
            datetime(value.year, value.month, value.day, 16, 0, tzinfo=UTC)
            if value is not None
            else None
            for value in dates
        ]
        return pl.Series("date", values, dtype=pl.Datetime("us", "UTC"))
    if date_kind == "utf8":
        values = [value.isoformat() if value is not None else None for value in dates]
        return pl.Series("date", values, dtype=pl.Utf8)
    raise ValueError(f"Unsupported date_kind: {date_kind!r}")


def _register_diagnostics_asset(
    tmp_path: Path,
    *,
    subject: str,
    dates: list[date | None],
    date_kind: str,
    available_at: datetime,
) -> AssetStore:
    frame = pl.DataFrame(
        {
            "date": _diagnostics_column(dates, date_kind=date_kind),
            "close": [float(index) for index in range(len(dates))],
        }
    )
    store = AssetStore(root=tmp_path)
    stored = store.write_frame(f"price_history/{subject}.parquet", frame)
    register_asset(
        provider="asof_diagnostics_test",
        kind="price_history",
        subject=subject,
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )
    return store


@pytest.mark.parametrize("date_kind", ["date", "datetime", "utf8"])
def test_price_frame_with_diagnostics_matches_price_frame_for_a_clean_frame(
    tmp_path: Path,
    date_kind: str,
) -> None:
    """No nulls, no future rows: the diagnostic frame is byte-identical to `price_frame`."""
    dates: list[date | None] = [date(2024, 1, 1) + timedelta(days=index) for index in range(5)]
    available_at = datetime(2024, 1, 5, tzinfo=UTC)
    store = _register_diagnostics_asset(
        tmp_path,
        subject=f"CLEAN-{date_kind}",
        dates=dates,
        date_kind=date_kind,
        available_at=available_at,
    )
    asof = AsOfData(available_at, store=store)

    direct = asof.price_frame(provider="asof_diagnostics_test", subject=f"CLEAN-{date_kind}")
    read = asof.price_frame_with_diagnostics(
        provider="asof_diagnostics_test", subject=f"CLEAN-{date_kind}"
    )

    assert read.frame.equals(direct)
    assert read.invalid_session_date_rows == 0
    assert read.frame.height == 5


@pytest.mark.parametrize("date_kind", ["date", "datetime", "utf8"])
def test_price_frame_with_diagnostics_counts_only_null_session_dates(
    tmp_path: Path,
    date_kind: str,
) -> None:
    """A null-dated row is counted; a merely-future row is clipped but not counted."""
    dates: list[date | None] = [
        date(2024, 1, 1),
        None,
        date(2024, 1, 3),
        None,
        date(2024, 1, 5),
        date(2024, 1, 20),  # after through_date: a normal clip, not an invalid row.
    ]
    available_at = datetime(2024, 1, 5, tzinfo=UTC)
    subject = f"NULLMIX-{date_kind}"
    store = _register_diagnostics_asset(
        tmp_path, subject=subject, dates=dates, date_kind=date_kind, available_at=available_at
    )
    asof = AsOfData(available_at, store=store)

    direct = asof.price_frame(provider="asof_diagnostics_test", subject=subject)
    read = asof.price_frame_with_diagnostics(provider="asof_diagnostics_test", subject=subject)

    assert read.invalid_session_date_rows == 2
    assert read.frame.equals(direct)
    assert set(read.frame["date"].to_list()) == {
        date(2024, 1, 1),
        date(2024, 1, 3),
        date(2024, 1, 5),
    }


@pytest.mark.parametrize("date_kind", ["date", "datetime", "utf8"])
def test_price_frame_with_diagnostics_counts_every_row_when_entirely_null(
    tmp_path: Path,
    date_kind: str,
) -> None:
    dates: list[date | None] = [None, None, None]
    available_at = datetime(2024, 1, 5, tzinfo=UTC)
    subject = f"ALLNULL-{date_kind}"
    store = _register_diagnostics_asset(
        tmp_path, subject=subject, dates=dates, date_kind=date_kind, available_at=available_at
    )
    asof = AsOfData(available_at, store=store)

    direct = asof.price_frame(provider="asof_diagnostics_test", subject=subject)
    read = asof.price_frame_with_diagnostics(provider="asof_diagnostics_test", subject=subject)

    assert read.invalid_session_date_rows == 3
    assert read.frame.height == 0
    assert read.frame.equals(direct)


@pytest.mark.parametrize("date_kind", ["date", "datetime", "utf8"])
def test_price_frame_with_diagnostics_future_only_rows_are_clipped_not_counted(
    tmp_path: Path,
    date_kind: str,
) -> None:
    dates: list[date | None] = [date(2024, 1, 10), date(2024, 1, 11)]
    available_at = datetime(2024, 1, 5, tzinfo=UTC)
    subject = f"FUTUREONLY-{date_kind}"
    store = _register_diagnostics_asset(
        tmp_path, subject=subject, dates=dates, date_kind=date_kind, available_at=available_at
    )
    asof = AsOfData(available_at, store=store)

    read = asof.price_frame_with_diagnostics(provider="asof_diagnostics_test", subject=subject)

    assert read.invalid_session_date_rows == 0
    assert read.frame.height == 0


def test_price_frame_with_diagnostics_missing_date_column_fails_explicitly(tmp_path: Path) -> None:
    frame = pl.DataFrame({"close": [1.0, 2.0]})
    available_at = datetime(2024, 1, 1, tzinfo=UTC)
    store = AssetStore(root=tmp_path)
    stored = store.write_frame("price_history/DIAGNODATECOL.parquet", frame)
    register_asset(
        provider="asof_diagnostics_test",
        kind="price_history",
        subject="DIAGNODATECOL",
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )
    asof = AsOfData(available_at, store=store)

    with pytest.raises(PriceFrameSchemaError, match="no 'date' column"):
        asof.price_frame(provider="asof_diagnostics_test", subject="DIAGNODATECOL")
    with pytest.raises(PriceFrameSchemaError, match="no 'date' column"):
        asof.price_frame_with_diagnostics(provider="asof_diagnostics_test", subject="DIAGNODATECOL")


def test_price_frame_with_diagnostics_unsupported_date_dtype_fails_explicitly(
    tmp_path: Path,
) -> None:
    frame = pl.DataFrame({"date": [20240101, 20240102], "close": [1.0, 2.0]})
    available_at = datetime(2024, 1, 2, tzinfo=UTC)
    store = AssetStore(root=tmp_path)
    stored = store.write_frame("price_history/DIAGBADDTYPE.parquet", frame)
    register_asset(
        provider="asof_diagnostics_test",
        kind="price_history",
        subject="DIAGBADDTYPE",
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )
    asof = AsOfData(available_at, store=store)

    with pytest.raises(PriceFrameSchemaError, match="unsupported dtype"):
        asof.price_frame(provider="asof_diagnostics_test", subject="DIAGBADDTYPE")
    with pytest.raises(PriceFrameSchemaError, match="unsupported dtype"):
        asof.price_frame_with_diagnostics(provider="asof_diagnostics_test", subject="DIAGBADDTYPE")


def test_price_frame_with_diagnostics_malformed_non_null_strings_fail_explicitly(
    tmp_path: Path,
) -> None:
    """A malformed *non-null* string still raises; `price_frame` does not start rejecting nulls."""
    frame = pl.DataFrame({"date": ["not-a-date", None, "2024-01-03"], "close": [1.0, 2.0, 3.0]})
    available_at = datetime(2024, 1, 3, tzinfo=UTC)
    store = AssetStore(root=tmp_path)
    stored = store.write_frame("price_history/DIAGBADSTRINGS.parquet", frame)
    register_asset(
        provider="asof_diagnostics_test",
        kind="price_history",
        subject="DIAGBADSTRINGS",
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )
    asof = AsOfData(available_at, store=store)

    with pytest.raises(PriceFrameSchemaError, match="could not be parsed"):
        asof.price_frame(provider="asof_diagnostics_test", subject="DIAGBADSTRINGS")
    with pytest.raises(PriceFrameSchemaError, match="could not be parsed"):
        asof.price_frame_with_diagnostics(
            provider="asof_diagnostics_test", subject="DIAGBADSTRINGS"
        )


def test_price_frame_with_diagnostics_returns_the_price_frame_read_dataclass(
    tmp_path: Path,
) -> None:
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(3)]
    available_at = datetime(2024, 1, 3, tzinfo=UTC)
    store = _register_diagnostics_asset(
        tmp_path, subject="DATACLASS", dates=dates, date_kind="date", available_at=available_at
    )

    read = AsOfData(available_at, store=store).price_frame_with_diagnostics(
        provider="asof_diagnostics_test", subject="DATACLASS"
    )

    assert isinstance(read, PriceFrameRead)
    assert isinstance(read.asset, DataAsset)
    assert read.asset.subject == "DATACLASS"
    assert isinstance(read.frame, pl.DataFrame)
    assert isinstance(read.invalid_session_date_rows, int)


def test_price_frame_with_diagnostics_selects_once_and_returns_that_exact_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hypothetical later vintage cannot separate frame bytes from provenance.

    The selector alternates between two eligible immutable assets. A second
    selection would therefore return ``newer``; the diagnostics read must make
    only one selection, read ``selected``'s path, and return ``selected``.
    """
    available_at = datetime(2024, 1, 3, tzinfo=UTC)
    store = AssetStore(root=tmp_path)
    selected_stored = store.write_frame(
        "price_history/SELECT-ONCE-a.parquet",
        pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "close": [1.0, 2.0, 3.0],
            }
        ),
    )
    selected = register_asset(
        provider="asof_diagnostics_test",
        kind="price_history",
        subject="SELECT-ONCE",
        stored=selected_stored,
        retrieved_at=available_at - timedelta(minutes=1),
        available_at=available_at - timedelta(minutes=1),
    )
    newer_stored = store.write_frame(
        "price_history/SELECT-ONCE-b.parquet",
        pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "close": [10.0, 20.0, 30.0],
            }
        ),
    )
    newer = register_asset(
        provider="asof_diagnostics_test",
        kind="price_history",
        subject="SELECT-ONCE",
        stored=newer_stored,
        retrieved_at=available_at,
        available_at=available_at,
    )
    asof = AsOfData(available_at, store=store)
    selections: list[str] = []

    def alternating_selector(*, provider: str, kind: str, subject: str) -> DataAsset:
        assert (provider, kind, subject) == (
            "asof_diagnostics_test",
            "price_history",
            "SELECT-ONCE",
        )
        selections.append(subject)
        return selected if len(selections) == 1 else newer

    monkeypatch.setattr(asof, "latest_asset", alternating_selector)

    read = asof.price_frame_with_diagnostics(
        provider="asof_diagnostics_test",
        subject="SELECT-ONCE",
    )

    assert selections == ["SELECT-ONCE"]
    assert read.asset == selected
    assert read.asset != newer
    assert read.frame["close"].to_list() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Explicit immutable-vintage replay.
# ---------------------------------------------------------------------------


def _explicit_price_asset(
    store: AssetStore,
    *,
    path: str,
    subject: str,
    frame: pl.DataFrame,
    available_at: datetime,
    retrieved_at: datetime | None = None,
    kind: str = "price_history",
) -> DataAsset:
    stored = store.write_frame(path, frame)
    return register_asset(
        provider="asof_explicit_asset_test",
        kind=kind,
        subject=subject,
        stored=stored,
        available_at=available_at,
        retrieved_at=retrieved_at or available_at,
    )


def test_explicit_price_asset_read_never_reselects_a_newer_vintage_and_clips_rows(
    tmp_path: Path,
    django_assert_num_queries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AssetStore(tmp_path)
    subject = "EXACT-VINTAGE"
    old = _explicit_price_asset(
        store,
        path="price_history/exact-vintage-a.parquet",
        subject=subject,
        frame=pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 5),
                ],
                "close": [1.0, 2.0, 500.0],
            }
        ),
        available_at=datetime(2024, 1, 2, 12, tzinfo=UTC),
    )
    newer = _explicit_price_asset(
        store,
        path="price_history/exact-vintage-b.parquet",
        subject=subject,
        frame=pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "close": [10.0, 20.0, 30.0],
            }
        ),
        available_at=datetime(2024, 1, 3, 12, tzinfo=UTC),
    )
    asof = AsOfData(datetime(2024, 1, 3, 18, tzinfo=UTC), store=store)

    assert (
        asof.latest_asset(
            provider="asof_explicit_asset_test",
            kind="price_history",
            subject=subject,
        )
        == newer
    )
    local_reads: list[str] = []
    original_read_bytes = store.read_bytes

    def recording_read_bytes(relative_path: str) -> bytes:
        local_reads.append(relative_path)
        return original_read_bytes(relative_path)

    def forbidden_read_frame(_relative_path: str) -> pl.DataFrame:
        raise AssertionError("verified exact-asset reads must parse the in-memory payload")

    monkeypatch.setattr(store, "read_bytes", recording_read_bytes)
    monkeypatch.setattr(store, "read_frame", forbidden_read_frame)
    with django_assert_num_queries(0):
        read = asof.price_frame_for_asset_with_diagnostics(
            asset=old,
            through_date=date(2024, 1, 3),
        )

    assert read.asset == old
    assert read.frame["date"].to_list() == [date(2024, 1, 1), date(2024, 1, 2)]
    assert read.frame["close"].to_list() == [1.0, 2.0]
    assert read.invalid_session_date_rows == 0
    assert local_reads == [old.relative_path]


def test_explicit_price_asset_read_rejects_checksum_mismatched_valid_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AssetStore(tmp_path)
    asset = _explicit_price_asset(
        store,
        path="price_history/explicit-checksum-mismatch.parquet",
        subject="EXPLICIT-CHECKSUM-MISMATCH",
        frame=pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "close": [1.0, 2.0],
            }
        ),
        available_at=datetime(2024, 1, 2, tzinfo=UTC),
    )
    target = store.resolve(asset.relative_path)
    pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "close": [1.0, 3.0],
        }
    ).write_parquet(target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() != asset.sha256

    local_reads: list[str] = []
    original_read_bytes = store.read_bytes

    def recording_read_bytes(relative_path: str) -> bytes:
        local_reads.append(relative_path)
        return original_read_bytes(relative_path)

    def forbidden_read_frame(_relative_path: str) -> pl.DataFrame:
        raise AssertionError("checksum verification must not reopen the asset path")

    monkeypatch.setattr(store, "read_bytes", recording_read_bytes)
    monkeypatch.setattr(store, "read_frame", forbidden_read_frame)

    with pytest.raises(PriceFrameChecksumMismatchError) as raised:
        AsOfData(
            datetime(2024, 1, 2, tzinfo=UTC),
            store=store,
        ).price_frame_for_asset_with_diagnostics(asset=asset)

    assert local_reads == [asset.relative_path]
    assert asset.relative_path not in str(raised.value)
    assert str(store.root) not in str(raised.value)


def test_explicit_price_asset_read_preserves_null_date_diagnostics(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    asset = _explicit_price_asset(
        store,
        path="price_history/explicit-null-date.parquet",
        subject="EXPLICIT-NULL",
        frame=pl.DataFrame(
            {
                "date": pl.Series(
                    "date",
                    [date(2024, 1, 2), None, date(2024, 1, 1), date(2024, 1, 8)],
                    dtype=pl.Date,
                ),
                "close": [2.0, 99.0, 1.0, 8.0],
            }
        ),
        available_at=datetime(2024, 1, 3, tzinfo=UTC),
    )

    read = AsOfData(
        datetime(2024, 1, 3, tzinfo=UTC), store=store
    ).price_frame_for_asset_with_diagnostics(asset=asset)

    assert read.invalid_session_date_rows == 1
    assert read.frame["date"].to_list() == [date(2024, 1, 1), date(2024, 1, 2)]


def test_explicit_price_asset_read_rejects_malformed_non_null_dates(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    asset = _explicit_price_asset(
        store,
        path="price_history/explicit-malformed-date.parquet",
        subject="EXPLICIT-MALFORMED",
        frame=pl.DataFrame(
            {
                "date": ["2024-01-01", None, "not-a-date"],
                "close": [1.0, 2.0, 3.0],
            }
        ),
        available_at=datetime(2024, 1, 3, tzinfo=UTC),
    )

    with pytest.raises(PriceFrameSchemaError, match="could not be parsed"):
        AsOfData(
            datetime(2024, 1, 3, tzinfo=UTC), store=store
        ).price_frame_for_asset_with_diagnostics(asset=asset)


def test_explicit_price_asset_read_rejects_wrong_kind_and_asof_violations(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    frame = pl.DataFrame({"date": [date(2024, 1, 1)], "close": [1.0]})
    wrong_kind = _explicit_price_asset(
        store,
        path="price_history/explicit-wrong-kind.parquet",
        subject="WRONG-KIND",
        frame=frame,
        available_at=datetime(2024, 1, 1, tzinfo=UTC),
        kind="raw_fundamentals",
    )
    late_available = _explicit_price_asset(
        store,
        path="price_history/explicit-late-available.parquet",
        subject="LATE-AVAILABLE",
        frame=frame,
        available_at=datetime(2024, 1, 3, tzinfo=UTC),
    )
    late_retrieved = _explicit_price_asset(
        store,
        path="price_history/explicit-late-retrieved.parquet",
        subject="LATE-RETRIEVED",
        frame=frame,
        available_at=datetime(2024, 1, 1, tzinfo=UTC),
        retrieved_at=datetime(2024, 1, 3, tzinfo=UTC),
    )
    valid = _explicit_price_asset(
        store,
        path="price_history/explicit-valid.parquet",
        subject="VALID",
        frame=frame,
        available_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    asof = AsOfData(datetime(2024, 1, 2, tzinfo=UTC), store=store)

    with pytest.raises(ValueError, match="expected 'price_history'"):
        asof.price_frame_for_asset_with_diagnostics(asset=wrong_kind)
    with pytest.raises(ValueError, match="available_at after"):
        asof.price_frame_for_asset_with_diagnostics(asset=late_available)
    with pytest.raises(ValueError, match="retrieved_at after"):
        asof.price_frame_for_asset_with_diagnostics(asset=late_retrieved)
    with pytest.raises(ValueError, match="through_date .* cannot be after"):
        asof.price_frame_for_asset_with_diagnostics(
            asset=valid,
            through_date=date(2024, 1, 3),
        )


def test_explicit_price_asset_read_fails_for_missing_or_escaping_files(
    tmp_path: Path,
) -> None:
    store = AssetStore(tmp_path)
    available_at = datetime(2024, 1, 2, tzinfo=UTC)
    missing = _explicit_price_asset(
        store,
        path="price_history/explicit-missing.parquet",
        subject="MISSING",
        frame=pl.DataFrame({"date": [date(2024, 1, 1)], "close": [1.0]}),
        available_at=available_at,
    )
    store.resolve(missing.relative_path).unlink()
    escaping = DataAsset.objects.create(
        provider="asof_explicit_asset_test",
        kind="price_history",
        subject="ESCAPING",
        relative_path="../escaping.parquet",
        sha256="f" * 64,
        available_at=available_at,
        retrieved_at=available_at,
    )
    asof = AsOfData(available_at, store=store)

    with pytest.raises(FileNotFoundError):
        asof.price_frame_for_asset_with_diagnostics(asset=missing)
    with pytest.raises(ValueError, match="escapes STANSTOCK_DATA_DIR"):
        asof.price_frame_for_asset_with_diagnostics(asset=escaping)

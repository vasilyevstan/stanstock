"""Point-in-time FX conversion: rate derivation, conversion, and attribution."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import Client
from django.urls import reverse

from stanstock.data.asof import AsOfData
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.fx import (
    DEFAULT_MAX_CARRY_DAYS,
    AmbiguousFxRateError,
    FxConverter,
    FxEvidenceGrade,
    MissingFxRateError,
    StaleFxRateError,
)
from stanstock.data.models import (
    Company,
    DataAsset,
    FxRate,
    Listing,
    Region,
    Security,
    Universe,
    UniverseMembership,
    UniverseSnapshot,
)
from stanstock.simulation.builders import build_price_panel, run_simulation_workflow
from stanstock.simulation.engine import AccountingEngine
from stanstock.simulation.fx import normalize_fx_frame
from stanstock.simulation.models import SimulationRun
from stanstock.simulation.types import (
    ExecutionPriceBasis,
    FxAttributionStatus,
    MissingPricePolicy,
    RebalanceFrequency,
    SimulationConfig,
    SimulationGrade,
    SimulationMode,
    SimulationWorkflowError,
)

DECISION_TIME = datetime(2026, 3, 2, 12, 0, tzinfo=UTC)

USD_ID = "11111111-1111-4111-8111-111111111111"
EUR_ID = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def fx_store(tmp_path: Path) -> AssetStore:
    return AssetStore(root=tmp_path / "fx_data")


def _fx_asset(store: AssetStore, *, name: str, available_at: datetime) -> DataAsset:
    stored = store.write_bytes(f"fx/{name}.csv", f"synthetic-{name}".encode())
    return register_asset(
        provider="ecb",
        kind="fx_rates",
        subject=name,
        stored=stored,
        retrieved_at=available_at,
        available_at=available_at,
    )


def _publish(
    asset: DataAsset,
    *,
    base: str,
    quote: str,
    observation: date,
    value: str,
    published_at: datetime | None = None,
    available_at: datetime | None = None,
) -> FxRate:
    resolved_published = published_at or datetime.combine(observation, time(14, 15), tzinfo=UTC)
    resolved_available = available_at or asset.available_at
    return FxRate.objects.create(
        base_currency=base,
        quote_currency=quote,
        observation_date=observation,
        value=Decimal(value),
        published_at=resolved_published,
        available_at=resolved_available,
        source_asset=asset,
    )


def _converter(
    store: AssetStore,
    *,
    decision_time: datetime = DECISION_TIME,
    grade: FxEvidenceGrade = FxEvidenceGrade.RESEARCH,
    max_carry_days: int = DEFAULT_MAX_CARRY_DAYS,
) -> FxConverter:
    return FxConverter(
        AsOfData(decision_time, store),
        evidence_grade=grade,
        max_carry_days=max_carry_days,
    )


# ---------------------------------------------------------------------------
# Rate derivation
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_direct_inverse_and_cross_paths_are_named_and_consistent(fx_store: AssetStore) -> None:
    asset = _fx_asset(
        fx_store, name="ecb-2026", available_at=datetime(2026, 2, 2, 16, 5, tzinfo=UTC)
    )
    observation = date(2026, 2, 2)
    _publish(asset, base="EUR", quote="USD", observation=observation, value="1.25")
    _publish(asset, base="EUR", quote="GBP", observation=observation, value="0.80")

    converter = _converter(fx_store)

    direct = converter.conversion(from_currency="EUR", to_currency="USD", value_date=observation)
    assert direct.path == "direct"
    assert direct.rate == pytest.approx(1.25)
    assert direct.carry_days == 0

    inverse = converter.conversion(from_currency="USD", to_currency="EUR", value_date=observation)
    assert inverse.path == "inverse"
    assert inverse.rate == pytest.approx(1.0 / 1.25)

    cross = converter.conversion(from_currency="GBP", to_currency="USD", value_date=observation)
    assert cross.path == "cross:EUR"
    # 1 GBP -> 1/0.80 EUR -> 1.25 EUR -> 1.5625 USD
    assert cross.rate == pytest.approx(1.5625)
    assert cross.source_asset_ids == (str(asset.id),)

    identity = converter.conversion(from_currency="USD", to_currency="USD", value_date=observation)
    assert identity.path == "identity"
    assert identity.rate == 1.0


@pytest.mark.django_db
def test_direct_quote_outranks_an_inconsistent_inverse_quote(fx_store: AssetStore) -> None:
    asset = _fx_asset(
        fx_store, name="both-ways", available_at=datetime(2026, 2, 2, 16, 5, tzinfo=UTC)
    )
    observation = date(2026, 2, 2)
    _publish(asset, base="EUR", quote="USD", observation=observation, value="1.25")
    _publish(asset, base="USD", quote="EUR", observation=observation, value="0.90")

    converter = _converter(fx_store)
    conversion = converter.conversion(
        from_currency="EUR", to_currency="USD", value_date=observation
    )

    assert conversion.path == "direct"
    assert conversion.rate == pytest.approx(1.25)


@pytest.mark.django_db
def test_disagreeing_pivots_fail_instead_of_picking_one(fx_store: AssetStore) -> None:
    asset = _fx_asset(
        fx_store, name="two-pivots", available_at=datetime(2026, 2, 2, 16, 5, tzinfo=UTC)
    )
    observation = date(2026, 2, 2)
    # GBP -> USD can be crossed through EUR or through CHF, and the two
    # implied cross rates disagree far beyond rounding noise.
    _publish(asset, base="EUR", quote="GBP", observation=observation, value="0.80")
    _publish(asset, base="EUR", quote="USD", observation=observation, value="1.25")
    _publish(asset, base="CHF", quote="GBP", observation=observation, value="0.80")
    _publish(asset, base="CHF", quote="USD", observation=observation, value="2.50")

    converter = _converter(fx_store)
    with pytest.raises(AmbiguousFxRateError, match="Ambiguous GBP/USD rate"):
        converter.conversion(from_currency="GBP", to_currency="USD", value_date=observation)


@pytest.mark.django_db
def test_weekend_carry_is_bounded_and_stale_rates_fail(fx_store: AssetStore) -> None:
    asset = _fx_asset(fx_store, name="friday", available_at=datetime(2026, 2, 6, 16, 5, tzinfo=UTC))
    friday = date(2026, 2, 6)
    _publish(asset, base="EUR", quote="USD", observation=friday, value="1.10")

    converter = _converter(fx_store, max_carry_days=7)

    monday = converter.conversion(
        from_currency="EUR", to_currency="USD", value_date=date(2026, 2, 9)
    )
    assert monday.observation_date == friday
    assert monday.carry_days == 3
    assert monday.rate == pytest.approx(1.10)

    thursday = converter.conversion(
        from_currency="EUR", to_currency="USD", value_date=date(2026, 2, 12)
    )
    assert thursday.carry_days == 6

    with pytest.raises(StaleFxRateError, match="exceeds the 7-day carry limit"):
        converter.conversion(from_currency="EUR", to_currency="USD", value_date=date(2026, 2, 14))


@pytest.mark.django_db
def test_observation_after_the_valued_date_is_never_used(fx_store: AssetStore) -> None:
    asset = _fx_asset(fx_store, name="later", available_at=datetime(2026, 2, 6, 16, 5, tzinfo=UTC))
    _publish(asset, base="EUR", quote="USD", observation=date(2026, 2, 3), value="1.10")
    _publish(asset, base="EUR", quote="USD", observation=date(2026, 2, 6), value="1.90")

    conversion = _converter(fx_store).conversion(
        from_currency="EUR", to_currency="USD", value_date=date(2026, 2, 4)
    )

    assert conversion.observation_date == date(2026, 2, 3)
    assert conversion.rate == pytest.approx(1.10)


@pytest.mark.django_db
def test_observed_evidence_refuses_a_later_retrieved_bundle(fx_store: AssetStore) -> None:
    """Later retrieval is a reconstruction, not something that was observed."""
    asset = _fx_asset(fx_store, name="bundle", available_at=datetime(2026, 2, 6, 16, 5, tzinfo=UTC))
    _publish(asset, base="EUR", quote="USD", observation=date(2026, 2, 3), value="1.10")

    research = _converter(fx_store, grade=FxEvidenceGrade.RESEARCH).conversion(
        from_currency="EUR", to_currency="USD", value_date=date(2026, 2, 4)
    )
    assert research.rate == pytest.approx(1.10)
    assert research.evidence_grade is FxEvidenceGrade.RESEARCH

    observed = _converter(fx_store, grade=FxEvidenceGrade.OBSERVED)
    with pytest.raises(MissingFxRateError, match="evidence grade observed"):
        observed.conversion(from_currency="EUR", to_currency="USD", value_date=date(2026, 2, 4))

    # Once the bundle had actually been held, observed evidence accepts it.
    assert observed.conversion(
        from_currency="EUR", to_currency="USD", value_date=date(2026, 2, 9)
    ).rate == pytest.approx(1.10)


@pytest.mark.django_db
def test_a_later_correction_cannot_change_an_earlier_valued_date(fx_store: AssetStore) -> None:
    """A February 9 correction must not rewrite how February 3 was priced."""
    original = _fx_asset(
        fx_store, name="orig", available_at=datetime(2026, 2, 3, 16, 5, tzinfo=UTC)
    )
    correction = _fx_asset(
        fx_store, name="corrected", available_at=datetime(2026, 2, 9, 16, 5, tzinfo=UTC)
    )
    observation = date(2026, 2, 3)
    _publish(original, base="EUR", quote="USD", observation=observation, value="1.10")
    _publish(
        correction,
        base="EUR",
        quote="USD",
        observation=observation,
        value="1.50",
        published_at=datetime(2026, 2, 9, 14, 15, tzinfo=UTC),
    )

    converter = _converter(fx_store)
    on_the_day = converter.conversion(
        from_currency="EUR", to_currency="USD", value_date=observation
    )
    assert on_the_day.rate == pytest.approx(1.10)
    assert on_the_day.availability_cutoff.date() == observation

    # The same correction does apply from the date it was published onward,
    # including to dates that carry the corrected observation forward.
    later = converter.conversion(
        from_currency="EUR", to_currency="USD", value_date=date(2026, 2, 9)
    )
    assert later.observation_date == observation
    assert later.rate == pytest.approx(1.50)


@pytest.mark.django_db
def test_late_publication_cannot_reach_back_to_an_earlier_execution(
    fx_store: AssetStore,
) -> None:
    """A quote published after the close of an earlier date is unusable then."""
    early = _fx_asset(fx_store, name="early", available_at=datetime(2026, 2, 2, 16, 5, tzinfo=UTC))
    late = _fx_asset(fx_store, name="late", available_at=datetime(2026, 2, 4, 16, 5, tzinfo=UTC))
    _publish(early, base="EUR", quote="USD", observation=date(2026, 2, 2), value="1.10")
    # Observed on the 3rd but only published after the 3rd had ended.
    _publish(
        late,
        base="EUR",
        quote="USD",
        observation=date(2026, 2, 3),
        value="1.80",
        published_at=datetime(2026, 2, 4, 9, 0, tzinfo=UTC),
    )

    converter = _converter(fx_store)
    third = converter.conversion(
        from_currency="EUR", to_currency="USD", value_date=date(2026, 2, 3)
    )
    assert third.observation_date == date(2026, 2, 2)
    assert third.rate == pytest.approx(1.10)

    fourth = converter.conversion(
        from_currency="EUR", to_currency="USD", value_date=date(2026, 2, 4)
    )
    assert fourth.observation_date == date(2026, 2, 3)
    assert fourth.rate == pytest.approx(1.80)


@pytest.mark.django_db
def test_rate_published_after_the_decision_boundary_is_invisible(fx_store: AssetStore) -> None:
    early = _fx_asset(fx_store, name="early", available_at=datetime(2026, 2, 3, 16, 5, tzinfo=UTC))
    late = _fx_asset(fx_store, name="late", available_at=datetime(2026, 2, 6, 16, 5, tzinfo=UTC))
    _publish(early, base="EUR", quote="USD", observation=date(2026, 2, 3), value="1.10")
    _publish(late, base="EUR", quote="USD", observation=date(2026, 2, 6), value="1.90")

    before_publication = _converter(fx_store, decision_time=datetime(2026, 2, 6, 15, 0, tzinfo=UTC))
    conversion = before_publication.conversion(
        from_currency="EUR", to_currency="USD", value_date=date(2026, 2, 6)
    )
    assert conversion.observation_date == date(2026, 2, 3)
    assert conversion.carry_days == 3

    after_publication = _converter(fx_store, decision_time=datetime(2026, 2, 6, 17, 0, tzinfo=UTC))
    assert after_publication.conversion(
        from_currency="EUR", to_currency="USD", value_date=date(2026, 2, 6)
    ).observation_date == date(2026, 2, 6)


@pytest.mark.django_db
def test_later_revision_applies_only_from_its_own_publication_onward(
    fx_store: AssetStore,
) -> None:
    original = _fx_asset(
        fx_store, name="original", available_at=datetime(2026, 2, 6, 16, 5, tzinfo=UTC)
    )
    revision = _fx_asset(
        fx_store, name="revision", available_at=datetime(2026, 2, 9, 16, 5, tzinfo=UTC)
    )
    observation = date(2026, 2, 6)
    _publish(original, base="EUR", quote="USD", observation=observation, value="1.10")
    _publish(
        revision,
        base="EUR",
        quote="USD",
        observation=observation,
        value="1.15",
        published_at=datetime(2026, 2, 9, 14, 15, tzinfo=UTC),
        available_at=revision.available_at,
    )

    # Before the revision existed at all, only the original is visible.
    before = _converter(fx_store, decision_time=datetime(2026, 2, 7, 12, 0, tzinfo=UTC))
    assert before.conversion(
        from_currency="EUR", to_currency="USD", value_date=observation
    ).rate == pytest.approx(1.10)

    after = _converter(fx_store, decision_time=datetime(2026, 2, 10, 12, 0, tzinfo=UTC))
    # The revision is now inside the decision boundary, but February 6 still
    # sees only what February 6 could see.
    assert after.conversion(
        from_currency="EUR", to_currency="USD", value_date=observation
    ).rate == pytest.approx(1.10)
    # A date after the revision's publication carries the corrected value.
    carried = after.conversion(from_currency="EUR", to_currency="USD", value_date=date(2026, 2, 10))
    assert carried.observation_date == observation
    assert carried.rate == pytest.approx(1.15)
    assert carried.carry_days == 4


@pytest.mark.django_db
def test_missing_pair_fails_explicitly(fx_store: AssetStore) -> None:
    asset = _fx_asset(
        fx_store, name="usd-only", available_at=datetime(2026, 2, 6, 16, 5, tzinfo=UTC)
    )
    _publish(asset, base="EUR", quote="USD", observation=date(2026, 2, 6), value="1.10")

    converter = _converter(fx_store)
    with pytest.raises(MissingFxRateError, match="No JPY/USD rate is derivable"):
        converter.conversion(from_currency="JPY", to_currency="USD", value_date=date(2026, 2, 6))


@pytest.mark.django_db
def test_conversion_frame_is_deterministic_and_engine_normalizable(
    fx_store: AssetStore,
) -> None:
    asset = _fx_asset(fx_store, name="frame", available_at=datetime(2026, 2, 6, 16, 5, tzinfo=UTC))
    _publish(asset, base="EUR", quote="USD", observation=date(2026, 2, 6), value="1.10")
    _publish(asset, base="EUR", quote="GBP", observation=date(2026, 2, 6), value="0.80")

    converter = _converter(fx_store)
    value_dates = [date(2026, 2, 9), date(2026, 2, 6)]
    frame = converter.conversion_frame(
        from_currencies=["USD", "EUR", "GBP"],
        to_currency="USD",
        value_dates=value_dates,
    )
    repeated = converter.conversion_frame(
        from_currencies=["GBP", "EUR", "USD"],
        to_currency="USD",
        value_dates=list(reversed(value_dates)),
    )

    assert frame.equals(repeated)
    assert frame.height == 6
    normalized = normalize_fx_frame(frame, base_currency="USD")
    assert set(normalized["from_currency"].to_list()) == {"USD", "EUR", "GBP"}
    assert normalized["value_date"].to_list() == sorted(normalized["value_date"].to_list())


def test_normalize_fx_frame_rejects_unusable_inputs() -> None:
    good = pl.DataFrame(
        {
            "value_date": [date(2026, 2, 6)],
            "from_currency": ["EUR"],
            "to_currency": ["USD"],
            "rate": [1.1],
        }
    )
    assert normalize_fx_frame(good, base_currency="USD").height == 1

    with pytest.raises(ValueError, match="quote into USD"):
        normalize_fx_frame(good, base_currency="EUR")

    duplicated = pl.concat([good, good])
    with pytest.raises(ValueError, match="Duplicate FX rates"):
        normalize_fx_frame(duplicated, base_currency="USD")

    negative = good.with_columns(pl.lit(-1.0).alias("rate"))
    with pytest.raises(ValueError, match="must all be positive"):
        normalize_fx_frame(negative, base_currency="USD")

    two_bases = pl.concat([good, good.with_columns(pl.lit("GBP").alias("to_currency"))])
    with pytest.raises(ValueError, match="one base currency"):
        normalize_fx_frame(two_bases, base_currency=None)

    with pytest.raises(ValueError, match="Missing"):
        normalize_fx_frame(good.drop("rate"), base_currency="USD")


# ---------------------------------------------------------------------------
# Engine attribution
# ---------------------------------------------------------------------------


def _two_currency_inputs(*, eur_usd: list[float]) -> tuple[pl.DataFrame, pl.DataFrame]:
    dates = [date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 4)]
    prices = pl.DataFrame(
        {
            "date": [d for d in dates for _ in range(2)],
            "listing_id": [USD_ID, EUR_ID] * 3,
            "close": [
                100.0,
                50.0 * eur_usd[0],
                100.0,
                50.0 * eur_usd[1],
                100.0,
                50.0 * eur_usd[2],
            ],
            "symbol": ["USDCO", "EURCO"] * 3,
            "currency": ["USD", "EUR"] * 3,
        }
    )
    fx_rates = pl.DataFrame(
        {
            "value_date": [d for d in dates for _ in range(2)],
            "from_currency": ["EUR", "USD"] * 3,
            "to_currency": ["USD"] * 6,
            "rate": [
                eur_usd[0],
                1.0,
                eur_usd[1],
                1.0,
                eur_usd[2],
                1.0,
            ],
            "carry_days": [0, 0, 1, 0, 0, 0],
        }
    )
    return prices, fx_rates


def _converted_config() -> SimulationConfig:
    return SimulationConfig(
        name="fx",
        mode=SimulationMode.PORTFOLIO,
        grade=SimulationGrade.RESEARCH,
        starting_capital=100_000.0,
        rebalance_frequency=RebalanceFrequency.NEVER,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        selected_symbols=[USD_ID, EUR_ID],
        base_currency="USD",
    )


def test_fx_attribution_splits_return_exactly() -> None:
    prices, fx_rates = _two_currency_inputs(eur_usd=[1.0, 1.0, 1.2])
    result = AccountingEngine(_converted_config()).run(prices=prices, fx_rates=fx_rates)

    metrics = result.metrics
    assert metrics.fx_conversion_applied is True
    assert metrics.fx_native_currencies == ["EUR", "USD"]
    assert metrics.fx_attribution_status == FxAttributionStatus.EXACT.value
    assert metrics.fx_max_carry_days_used == 1
    # Both holdings are flat in their own currency; the entire 10% comes from FX.
    assert metrics.cumulative_return == pytest.approx(0.10)
    assert metrics.fx_local_currency_cumulative_return == pytest.approx(0.0, abs=1e-12)
    assert metrics.fx_contribution_return == pytest.approx(0.10)
    assert metrics.fx_local_currency_cumulative_return + metrics.fx_contribution_return == (
        pytest.approx(metrics.cumulative_return)
    )


def test_fx_attribution_is_zero_when_rates_do_not_move() -> None:
    prices, fx_rates = _two_currency_inputs(eur_usd=[1.0, 1.0, 1.0])
    result = AccountingEngine(_converted_config()).run(prices=prices, fx_rates=fx_rates)

    assert result.metrics.fx_contribution_return == pytest.approx(0.0, abs=1e-12)
    assert result.metrics.fx_local_currency_cumulative_return == pytest.approx(0.0, abs=1e-12)


def test_single_currency_run_reports_no_fx_attribution() -> None:
    prices = pl.DataFrame(
        {
            "date": [date(2026, 2, 2), date(2026, 2, 3)],
            "listing_id": [USD_ID, USD_ID],
            "close": [100.0, 110.0],
            "symbol": ["USDCO", "USDCO"],
            "currency": ["USD", "USD"],
        }
    )
    config = _converted_config()
    config.selected_symbols = [USD_ID]
    result = AccountingEngine(config).run(prices=prices)

    assert result.metrics.fx_conversion_applied is False
    assert result.metrics.fx_attribution_status == FxAttributionStatus.NOT_APPLICABLE.value
    assert result.metrics.fx_contribution_return is None
    assert result.metrics.fx_native_currencies is None


def test_single_currency_input_hash_is_unchanged_by_fx_support() -> None:
    """Pinned pre-FX reproducibility identity for a no-conversion run.

    The literal below was produced by replaying the exact hashing algorithm
    that existed before FX conversion was added (config JSON, then the
    normalized price/signal/benchmark frames, then the calendar) against
    these inputs. It is pinned so a future change cannot silently invalidate
    the archived ``input_hash`` of every single-currency run ever persisted.
    """
    prices = pl.DataFrame(
        {
            "date": [date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 4)],
            "listing_id": [USD_ID] * 3,
            "close": [100.0, 105.0, 110.0],
            "open": [99.0, 104.0, 109.0],
            "symbol": ["USDCO"] * 3,
            "currency": ["USD"] * 3,
        }
    )
    config = SimulationConfig(
        name="pinned-single-currency",
        mode=SimulationMode.PORTFOLIO,
        grade=SimulationGrade.RESEARCH,
        starting_capital=100_000.0,
        rebalance_frequency=RebalanceFrequency.NEVER,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
        selected_symbols=[USD_ID],
        base_currency="USD",
    )

    result = AccountingEngine(config).run(prices=prices)

    assert result.input_hash == ("42ab6248de60f94b56ad29839e23227a642ff093923e14ba1a358c42105e28e2")


def test_native_currency_assignment_is_covered_by_the_input_hash() -> None:
    """Attribution reads native currency, so the hash must read it too."""
    prices, fx_rates = _two_currency_inputs(eur_usd=[1.0, 1.0, 1.0])
    baseline = AccountingEngine(_converted_config()).run(prices=prices, fx_rates=fx_rates)

    # Rates are all 1.0, so swapping which listing is EUR and which is USD
    # leaves every converted close identical while changing what the FX
    # attribution is measuring.
    swapped_prices = prices.with_columns(
        pl.when(pl.col("currency") == "EUR")
        .then(pl.lit("USD"))
        .otherwise(pl.lit("EUR"))
        .alias("currency")
    )
    swapped = AccountingEngine(_converted_config()).run(prices=swapped_prices, fx_rates=fx_rates)

    assert swapped_prices["close"].to_list() == prices["close"].to_list()
    assert swapped.input_hash != baseline.input_hash


def test_retained_native_prices_are_covered_by_the_input_hash() -> None:
    prices, fx_rates = _two_currency_inputs(eur_usd=[1.0, 1.0, 1.2])
    with_native = prices.with_columns((pl.col("close") / 1.0).alias("close_native"))
    baseline = AccountingEngine(_converted_config()).run(prices=with_native, fx_rates=fx_rates)
    altered = AccountingEngine(_converted_config()).run(
        prices=with_native.with_columns((pl.col("close_native") * 2.0).alias("close_native")),
        fx_rates=fx_rates,
    )
    assert altered.input_hash != baseline.input_hash


def test_engine_refuses_to_add_currencies_without_fx_rates() -> None:
    prices, _fx_rates = _two_currency_inputs(eur_usd=[1.0, 1.0, 1.2])
    with pytest.raises(ValueError, match="mixes native currencies"):
        AccountingEngine(_converted_config()).run(prices=prices)


def test_engine_refuses_a_panel_denominated_in_another_currency() -> None:
    prices = pl.DataFrame(
        {
            "date": [date(2026, 2, 2), date(2026, 2, 3)],
            "listing_id": [EUR_ID, EUR_ID],
            "close": [50.0, 55.0],
            "symbol": ["EURCO", "EURCO"],
            "currency": ["EUR", "EUR"],
        }
    )
    config = _converted_config()
    config.selected_symbols = [EUR_ID]
    with pytest.raises(ValueError, match="denominated in EUR"):
        AccountingEngine(config).run(prices=prices)


def test_closed_foreign_market_keeps_its_currency_exposure() -> None:
    """A EUR holiday must not freeze the euro at its last trading session."""
    prices, fx_rates = _two_currency_inputs(eur_usd=[1.0, 1.0, 1.2])
    # The euro listing does not trade on the final date; the euro itself does.
    closed = prices.filter(
        ~((pl.col("listing_id") == EUR_ID) & (pl.col("date") == date(2026, 2, 4)))
    )

    result = AccountingEngine(_converted_config()).run(prices=closed, fx_rates=fx_rates)

    # 500 USD shares at 100 plus 1000 EUR shares whose 50 EUR quote is now
    # worth 60 USD: 50,000 + 60,000 against 100,000 of capital.
    assert result.metrics.ending_capital == pytest.approx(110_000.0, rel=1e-9)
    assert result.metrics.cumulative_return == pytest.approx(0.10, rel=1e-9)
    terminal = result.holdings.filter(
        (pl.col("observation_date") == date(2026, 2, 4)) & (pl.col("listing_id") == EUR_ID)
    )
    assert terminal["price"].to_list() == pytest.approx([60.0])
    # The stock itself did not move in its own currency, so the entire gain
    # is still attributed to FX.
    assert result.metrics.fx_attribution_status == FxAttributionStatus.EXACT.value
    assert result.metrics.fx_local_currency_cumulative_return == pytest.approx(0.0, abs=1e-9)
    assert result.metrics.fx_contribution_return == pytest.approx(0.10, rel=1e-9)


def test_rebalance_sizes_a_closed_foreign_holding_at_todays_rate() -> None:
    """Pre-trade sizing must use the value the day actually reports.

    On the final date the euro market is shut while the euro itself moves
    20%. The closed holding stays untradable, but the capital it represents
    has changed, and the domestic leg must be rebalanced against that
    revalued total -- not against the conversion frozen at the euro market's
    last open session.
    """
    prices, fx_rates = _two_currency_inputs(eur_usd=[1.0, 1.0, 1.2])
    closed = prices.filter(
        ~((pl.col("listing_id") == EUR_ID) & (pl.col("date") == date(2026, 2, 4)))
    )
    config = _converted_config()
    config.rebalance_frequency = RebalanceFrequency.DAILY
    # Half the book stays in cash, so the rebalance has room to act on the
    # revalued total instead of being clipped by an empty cash balance.
    config.cash_buffer_bps = 5000.0

    result = AccountingEngine(config).run(prices=closed, fx_rates=fx_rates)

    final_curve = result.daily_curves.filter(pl.col("date") == date(2026, 2, 4)).to_dicts()[0]
    # Cash 50,000 + USD 250 shares at 100 + EUR 500 shares revalued at 60.
    assert final_curve["portfolio_value"] == pytest.approx(105_000.0, rel=1e-9)

    final_trades = result.trades.filter(pl.col("trade_date") == date(2026, 2, 4)).to_dicts()
    assert len(final_trades) == 1
    trade = final_trades[0]
    assert trade["listing_id"] == USD_ID
    assert trade["side"] == "buy"
    # Target dollars are half of the revalued 105,000, i.e. 52,500: the USD
    # leg goes from 250 to 525 shares. Sizing off the stale 100,000 total
    # would have bought only 250 more.
    assert trade["quantity"] == pytest.approx(275.0, rel=1e-9)
    assert trade["price"] == pytest.approx(100.0, rel=1e-9)

    final_usd = result.holdings.filter(
        (pl.col("observation_date") == date(2026, 2, 4)) & (pl.col("listing_id") == USD_ID)
    )
    assert final_usd["market_value"].to_list() == pytest.approx([52_500.0])
    assert final_usd["market_value"][0] == pytest.approx(
        0.5 * final_curve["portfolio_value"], rel=1e-9
    )


def test_calendar_date_beyond_fx_coverage_fails_the_run() -> None:
    """A date with no eligible rate must fail, not report a return."""
    prices, fx_rates = _two_currency_inputs(eur_usd=[1.0, 1.0, 1.2])
    calendar = [
        date(2026, 2, 2),
        date(2026, 2, 3),
        date(2026, 2, 4),
        date(2026, 2, 5),
    ]

    with pytest.raises(ValueError, match="EUR/USD on 2026-02-05"):
        AccountingEngine(_converted_config()).run(
            prices=prices, fx_rates=fx_rates, calendar=calendar
        )

    # The same calendar without the uncovered date still runs.
    covered = AccountingEngine(_converted_config()).run(
        prices=prices, fx_rates=fx_rates, calendar=calendar[:-1]
    )
    assert covered.metrics.end_date == date(2026, 2, 4)


def test_single_currency_run_accepts_a_calendar_without_fx_coverage() -> None:
    """The coverage gate must not touch a run that converts nothing."""
    prices = pl.DataFrame(
        {
            "date": [date(2026, 2, 2), date(2026, 2, 3)],
            "listing_id": [USD_ID, USD_ID],
            "close": [100.0, 110.0],
            "symbol": ["USDCO", "USDCO"],
            "currency": ["USD", "USD"],
        }
    )
    config = _converted_config()
    config.selected_symbols = [USD_ID]
    result = AccountingEngine(config).run(
        prices=prices, calendar=[date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 4)]
    )
    assert result.metrics.end_date == date(2026, 2, 4)


@pytest.mark.django_db
def test_converted_open_execution_is_rejected_for_post_open_publication(
    fx_store: AssetStore,
) -> None:
    """An opening trade cannot be settled at a rate published after the bell.

    The euro reference rate for a session is published in the afternoon, but
    FX availability is only resolved to end-of-day, so nothing distinguishes
    it from a rate that existed at the 09:30 open. Rather than convert an
    opening execution with information it could not have had, open-capable
    execution bases are refused outright for a converted run.
    """
    asset = _fx_asset(
        fx_store, name="post-open", available_at=datetime(2026, 2, 2, 16, 5, tzinfo=UTC)
    )
    for day, value in ((date(2026, 2, 2), "1.00"), (date(2026, 2, 3), "1.20")):
        _publish(
            asset,
            base="EUR",
            quote="USD",
            observation=day,
            value=value,
            # Published well after any equity market has opened.
            published_at=datetime.combine(day, time(16, 5), tzinfo=UTC),
            available_at=datetime.combine(day, time(16, 5), tzinfo=UTC),
        )

    converter = _converter(fx_store)
    fx_rates = converter.conversion_frame(
        from_currencies=["EUR", "USD"],
        to_currency="USD",
        value_dates=[date(2026, 2, 2), date(2026, 2, 3)],
    )
    market_open = datetime(2026, 2, 3, 9, 30, tzinfo=UTC)
    eur_row = fx_rates.filter(
        (pl.col("from_currency") == "EUR") & (pl.col("value_date") == date(2026, 2, 3))
    ).to_dicts()[0]
    # The rate that end-of-day resolution admits for this session did not
    # exist when the session opened.
    assert eur_row["rate_published_at"] > market_open
    assert eur_row["availability_cutoff"] > market_open

    prices = pl.DataFrame(
        {
            "date": [date(2026, 2, 2), date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 3)],
            "listing_id": [USD_ID, EUR_ID, USD_ID, EUR_ID],
            "close": [100.0, 50.0, 100.0, 60.0],
            "open": [100.0, 50.0, 100.0, 60.0],
            "symbol": ["USDCO", "EURCO", "USDCO", "EURCO"],
            "currency": ["USD", "EUR", "USD", "EUR"],
        }
    )

    for basis in (ExecutionPriceBasis.NEXT_OPEN, ExecutionPriceBasis.NEXT_ELIGIBLE):
        config = _converted_config()
        config.execution_basis = basis
        with pytest.raises(ValueError, match="may execute at a market open"):
            AccountingEngine(config).run(prices=prices, fx_rates=fx_rates)

    # Close-based execution is consistent with end-of-day FX resolution.
    config = _converted_config()
    config.execution_basis = ExecutionPriceBasis.NEXT_CLOSE
    assert AccountingEngine(config).run(prices=prices, fx_rates=fx_rates).metrics.total_trades > 0


def test_open_execution_remains_available_without_conversion() -> None:
    prices = pl.DataFrame(
        {
            "date": [date(2026, 2, 2), date(2026, 2, 3)],
            "listing_id": [USD_ID, USD_ID],
            "close": [100.0, 110.0],
            "open": [99.0, 109.0],
            "symbol": ["USDCO", "USDCO"],
            "currency": ["USD", "USD"],
        }
    )
    config = _converted_config()
    config.selected_symbols = [USD_ID]
    config.execution_basis = ExecutionPriceBasis.NEXT_OPEN

    result = AccountingEngine(config).run(prices=prices)

    assert result.trades["price"].to_list() == pytest.approx([99.0])


def test_single_currency_carry_is_unaffected_by_fx_support() -> None:
    prices = pl.DataFrame(
        {
            "date": [date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 4)],
            "listing_id": [USD_ID, USD_ID, EUR_ID],
            "close": [100.0, 100.0, 42.0],
            "symbol": ["USDCO", "USDCO", "OTHER"],
            "currency": ["USD", "USD", "USD"],
        }
    )
    config = _converted_config()
    config.selected_symbols = [USD_ID]
    result = AccountingEngine(config).run(prices=prices)

    terminal = result.holdings.filter(pl.col("observation_date") == date(2026, 2, 4))
    assert terminal["price"].to_list() == pytest.approx([100.0])


def test_fx_input_frame_changes_the_reproducibility_hash() -> None:
    prices, fx_rates = _two_currency_inputs(eur_usd=[1.0, 1.0, 1.2])
    baseline = AccountingEngine(_converted_config()).run(prices=prices, fx_rates=fx_rates)
    repeated = AccountingEngine(_converted_config()).run(prices=prices, fx_rates=fx_rates)
    assert baseline.input_hash == repeated.input_hash

    # A different FX vintage that happens to leave converted prices untouched
    # must still be a different, distinguishable input set.
    revised_rates = fx_rates.with_columns(
        pl.when(pl.col("from_currency") == "EUR")
        .then(pl.col("carry_days") + 1)
        .otherwise(pl.col("carry_days"))
        .alias("carry_days")
    )
    revised = AccountingEngine(_converted_config()).run(prices=prices, fx_rates=revised_rates)
    assert revised.input_hash != baseline.input_hash


def test_engine_rejects_fx_frame_quoting_a_different_base() -> None:
    prices, fx_rates = _two_currency_inputs(eur_usd=[1.0, 1.0, 1.2])
    config = _converted_config()
    config.base_currency = "GBP"
    with pytest.raises(ValueError, match="base currency is GBP"):
        AccountingEngine(config).run(prices=prices, fx_rates=fx_rates)


def test_engine_rejects_fx_frame_without_native_currency_column() -> None:
    prices, fx_rates = _two_currency_inputs(eur_usd=[1.0, 1.0, 1.2])
    with pytest.raises(ValueError, match="no 'currency' column"):
        AccountingEngine(_converted_config()).run(prices=prices.drop("currency"), fx_rates=fx_rates)


def test_cash_settlement_withholds_fx_attribution() -> None:
    prices, fx_rates = _two_currency_inputs(eur_usd=[1.0, 1.0, 1.2])
    # Remove the EUR listing's terminal observation so the missing-price
    # policy drives a settlement-style resolution on the final date.
    trimmed = prices.filter(
        ~((pl.col("listing_id") == EUR_ID) & (pl.col("date") == date(2026, 2, 4)))
    )
    config = _converted_config()
    config.missing_price_policy = MissingPricePolicy.DROP
    result = AccountingEngine(config).run(prices=trimmed, fx_rates=fx_rates)

    # A dropped holding leaves no unresolved FX basis, so attribution stays exact.
    assert result.metrics.fx_attribution_status == FxAttributionStatus.EXACT.value

    from stanstock.simulation.hooks import CashSettlementHook

    settled = AccountingEngine(
        _converted_config(),
        corporate_event_hook=CashSettlementHook({(EUR_ID, date(2026, 2, 4)): 55.0}),
    ).run(prices=trimmed, fx_rates=fx_rates)
    assert settled.metrics.fx_attribution_status == FxAttributionStatus.UNAVAILABLE.value
    assert settled.metrics.fx_contribution_return is None
    assert "cash-settled" in (settled.metrics.fx_attribution_detail or "")


# ---------------------------------------------------------------------------
# Workflow integration
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_currency_universe(
    fx_store: AssetStore,
) -> Iterator[tuple[UniverseSnapshot, dict[str, Listing], AssetStore]]:
    universe = Universe.objects.create(slug="mixed", name="Mixed currency", config_version="1.0")
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=date(2026, 2, 2),
        grade=UniverseSnapshot.Grade.RESEARCH,
        config_hash="mixed-hash",
    )
    sessions = [date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 4)]
    listings: dict[str, Listing] = {}
    specs = (
        ("USDCO", "USD", "XNAS", Region.US, [100.0, 100.0, 100.0]),
        ("EURCO", "EUR", "XETR", Region.EUROPE, [50.0, 50.0, 50.0]),
        ("GBPCO", "GBP", "XLON", Region.EUROPE, [40.0, 40.0, 40.0]),
    )
    for ticker, currency, mic, region, closes in specs:
        company = Company.objects.create(name=f"{ticker} Plc", country="US")
        security = Security.objects.create(company=company, name=f"{ticker} Common")
        listing = Listing.objects.create(
            security=security,
            ticker=ticker,
            exchange_mic=mic,
            currency=currency,
            region=region,
        )
        UniverseMembership.objects.create(snapshot=snapshot, listing=listing, eligible=True)
        listings[ticker] = listing
        frame = pl.DataFrame(
            {
                "date": sessions,
                "open": [value - 1.0 for value in closes],
                "close": closes,
            }
        )
        stored = fx_store.write_frame(f"prices/{ticker}.parquet", frame)
        available = datetime(2026, 2, 4, 22, 0, tzinfo=UTC)
        register_asset(
            provider="synthetic_demo",
            kind="price_history",
            subject=ticker,
            stored=stored,
            retrieved_at=available,
            available_at=available,
        )

    # ECB-style EUR-based quotes: USD is inverted, GBP is crossed through EUR.
    # The bundle becomes available the morning after the last priced session,
    # so a decision taken before then legitimately sees no FX at all.
    asset = _fx_asset(fx_store, name="ecb-feb", available_at=datetime(2026, 2, 5, 6, 0, tzinfo=UTC))
    for observation, eur_usd, eur_gbp in (
        (date(2026, 2, 2), "1.00", "0.80"),
        (date(2026, 2, 3), "1.00", "0.80"),
        (date(2026, 2, 4), "1.20", "0.80"),
    ):
        _publish(asset, base="EUR", quote="USD", observation=observation, value=eur_usd)
        _publish(asset, base="EUR", quote="GBP", observation=observation, value=eur_gbp)

    yield snapshot, listings, fx_store


def _add_listing(
    snapshot: UniverseSnapshot,
    store: AssetStore,
    *,
    ticker: str,
    currency: str,
    mic: str,
    sessions: list[date],
) -> Listing:
    company = Company.objects.create(name=f"{ticker} Plc", country="CH")
    security = Security.objects.create(company=company, name=f"{ticker} Common")
    listing = Listing.objects.create(
        security=security,
        ticker=ticker,
        exchange_mic=mic,
        currency=currency,
        region=Region.EUROPE,
    )
    UniverseMembership.objects.create(snapshot=snapshot, listing=listing, eligible=True)
    frame = pl.DataFrame(
        {
            "date": sessions,
            "open": [70.0] * len(sessions),
            "close": [70.0] * len(sessions),
        }
    )
    stored = store.write_frame(f"prices/{ticker}.parquet", frame)
    available = datetime(2026, 2, 26, 22, 0, tzinfo=UTC)
    register_asset(
        provider="synthetic_demo",
        kind="price_history",
        subject=ticker,
        stored=stored,
        retrieved_at=available,
        available_at=available,
    )
    return listing


@pytest.mark.django_db
def test_mixed_currency_panel_keeps_native_prices_and_converts(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, store = mixed_currency_universe

    panel = build_price_panel(
        snapshot=snapshot,
        start_date=date(2026, 2, 2),
        end_date=date(2026, 2, 4),
        provider="synthetic_demo",
        base_currency="USD",
        decision_time=DECISION_TIME,
        asset_store=store,
    )

    assert panel.base_currency == "USD"
    assert panel.native_currencies == ("EUR", "GBP", "USD")
    assert panel.conversion_applied is True
    assert panel.fx_rates is not None

    eur_rows = panel.prices.filter(pl.col("symbol") == "EURCO").sort("date")
    assert eur_rows["close_native"].to_list() == [50.0, 50.0, 50.0]
    assert eur_rows["close"].to_list() == pytest.approx([50.0, 50.0, 60.0])
    assert eur_rows["currency"].to_list() == ["EUR"] * 3
    assert set(eur_rows["base_currency"].to_list()) == {"USD"}
    assert eur_rows["fx_path"].to_list() == ["direct"] * 3

    gbp_rows = panel.prices.filter(pl.col("symbol") == "GBPCO").sort("date")
    assert gbp_rows["fx_path"].to_list() == ["cross:EUR"] * 3
    # 1 GBP = 1/0.80 EUR = 1.25 EUR, and 1 EUR = 1.20 USD on the final date.
    assert gbp_rows["close"].to_list() == pytest.approx([50.0, 50.0, 60.0])

    usd_rows = panel.prices.filter(pl.col("symbol") == "USDCO")
    assert usd_rows["fx_path"].to_list() == ["identity"] * 3
    assert usd_rows["close"].to_list() == [100.0, 100.0, 100.0]


@pytest.mark.django_db
def test_inverse_path_drives_a_eur_based_run(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, store = mixed_currency_universe

    panel = build_price_panel(
        snapshot=snapshot,
        start_date=date(2026, 2, 2),
        end_date=date(2026, 2, 4),
        provider="synthetic_demo",
        base_currency="EUR",
        decision_time=DECISION_TIME,
        asset_store=store,
    )

    usd_rows = panel.prices.filter(pl.col("symbol") == "USDCO").sort("date")
    assert usd_rows["fx_path"].to_list() == ["inverse"] * 3
    assert usd_rows["close"].to_list() == pytest.approx([100.0, 100.0, 100.0 / 1.2])
    gbp_rows = panel.prices.filter(pl.col("symbol") == "GBPCO").sort("date")
    assert gbp_rows["fx_path"].to_list() == ["inverse"] * 3


@pytest.mark.django_db
def test_missing_rate_fails_the_whole_build(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, _listings, store = mixed_currency_universe
    _add_listing(
        snapshot,
        store,
        ticker="CHFCO",
        currency="CHF",
        mic="XSWX",
        sessions=[date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 4)],
    )

    with pytest.raises(SimulationWorkflowError, match="No CHF/USD rate is derivable"):
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 2, 2),
            end_date=date(2026, 2, 4),
            provider="synthetic_demo",
            base_currency="USD",
            decision_time=DECISION_TIME,
            asset_store=store,
        )


@pytest.mark.django_db
def test_stale_rate_fails_the_whole_build(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, _listings, store = mixed_currency_universe
    # A session well past the newest FX observation must not be priced by
    # carrying a three-week-old rate onto it.
    _add_listing(
        snapshot,
        store,
        ticker="LATECO",
        currency="USD",
        mic="XNAS",
        sessions=[date(2026, 2, 2), date(2026, 2, 25)],
    )

    with pytest.raises(SimulationWorkflowError, match="carry limit"):
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 2, 2),
            end_date=date(2026, 2, 25),
            provider="synthetic_demo",
            base_currency="USD",
            decision_time=DECISION_TIME,
            asset_store=store,
        )


@pytest.mark.django_db
def test_rate_unavailable_at_the_decision_boundary_fails(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, _listings, store = mixed_currency_universe

    # Prices are available at this instant but the FX bundle is not, so the
    # run fails instead of quietly converting with a rate nobody had yet.
    with pytest.raises(SimulationWorkflowError, match="No EUR/USD rate is derivable"):
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 2, 2),
            end_date=date(2026, 2, 4),
            provider="synthetic_demo",
            base_currency="USD",
            decision_time=datetime(2026, 2, 5, 0, 0, tzinfo=UTC),
            asset_store=store,
        )


@pytest.mark.django_db
def test_benchmark_needs_an_explicit_currency_when_converting(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, store = mixed_currency_universe
    frame = pl.DataFrame(
        {
            "date": [date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 4)],
            "close": [200.0, 200.0, 200.0],
        }
    )
    stored = store.write_frame("prices/BENCH.parquet", frame)
    available = datetime(2026, 2, 4, 22, 0, tzinfo=UTC)
    register_asset(
        provider="synthetic_demo",
        kind="price_history",
        subject="BENCH",
        stored=stored,
        retrieved_at=available,
        available_at=available,
    )

    with pytest.raises(SimulationWorkflowError, match="needs an explicit benchmark currency"):
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 2, 2),
            end_date=date(2026, 2, 4),
            provider="synthetic_demo",
            base_currency="USD",
            benchmark_subject="BENCH",
            decision_time=DECISION_TIME,
            asset_store=store,
        )

    panel = build_price_panel(
        snapshot=snapshot,
        start_date=date(2026, 2, 2),
        end_date=date(2026, 2, 4),
        provider="synthetic_demo",
        base_currency="USD",
        benchmark_subject="BENCH",
        benchmark_currency="EUR",
        decision_time=DECISION_TIME,
        asset_store=store,
    )
    assert panel.benchmark is not None
    assert panel.benchmark["close_native"].to_list() == [200.0, 200.0, 200.0]
    assert panel.benchmark["close"].to_list() == pytest.approx([200.0, 200.0, 240.0])


@pytest.mark.django_db
def test_mixed_currency_workflow_persists_replayable_fx_inputs(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, store = mixed_currency_universe

    _definition, run, result = run_simulation_workflow(
        name="Mixed currency portfolio",
        mode="portfolio",
        snapshot=snapshot,
        start_date=date(2026, 2, 2),
        end_date=date(2026, 2, 4),
        starting_capital=90_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        selected_listing_ids=[listing.id for listing in listings.values()],
        base_currency="USD",
        provider="synthetic_demo",
        asset_store=store,
        decision_time=DECISION_TIME,
        code_revision="fx-test",
    )

    assert run.status == SimulationRun.Status.COMPLETE
    metrics = result.metrics
    assert metrics.base_currency == "USD"
    assert metrics.fx_conversion_applied is True
    assert metrics.fx_native_currencies == ["EUR", "GBP", "USD"]
    assert metrics.fx_attribution_status == FxAttributionStatus.EXACT.value
    # Two thirds of the book is non-USD and every stock is flat natively, so
    # the whole return is FX and the stock leg is exactly zero.
    assert metrics.fx_local_currency_cumulative_return == pytest.approx(0.0, abs=1e-9)
    assert metrics.fx_contribution_return == pytest.approx(metrics.cumulative_return)
    assert metrics.cumulative_return == pytest.approx(2.0 / 3.0 * 0.2, rel=1e-6)

    fx_info = run.metrics["input_assets"]["fx"]
    stored_fx = store.read_frame(fx_info["relative_path"])
    assert stored_fx.height == 9
    assert set(stored_fx["path"].to_list()) == {"identity", "direct", "cross:EUR"}
    assert set(stored_fx["to_currency"].to_list()) == {"USD"}

    stored_prices = store.read_frame(run.metrics["input_assets"]["prices"]["relative_path"])
    assert "close_native" in stored_prices.columns
    assert set(stored_prices["currency"].to_list()) == {"USD", "EUR", "GBP"}

    fx_asset = DataAsset.objects.get(kind="simulation_input_fx", subject=str(run.id))
    assert fx_asset.metadata["base_currency"] == "USD"
    assert fx_asset.metadata["native_currencies"] == ["EUR", "GBP", "USD"]


@pytest.mark.django_db
def test_benchmark_only_conversion_still_records_fx_provenance(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    """A USD-only book with a EUR benchmark converted something; say so."""
    snapshot, listings, store = mixed_currency_universe
    frame = pl.DataFrame(
        {
            "date": [date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 4)],
            "close": [200.0, 200.0, 200.0],
        }
    )
    stored = store.write_frame("prices/EUBENCH.parquet", frame)
    available = datetime(2026, 2, 4, 22, 0, tzinfo=UTC)
    register_asset(
        provider="synthetic_demo",
        kind="price_history",
        subject="EUBENCH",
        stored=stored,
        retrieved_at=available,
        available_at=available,
    )

    _definition, _run, result = run_simulation_workflow(
        name="USD book, EUR benchmark",
        mode="portfolio",
        snapshot=snapshot,
        start_date=date(2026, 2, 2),
        end_date=date(2026, 2, 4),
        starting_capital=50_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        selected_listing_ids=[listings["USDCO"].id],
        base_currency="USD",
        benchmark_subject="EUBENCH",
        benchmark_currency="EUR",
        provider="synthetic_demo",
        asset_store=store,
        decision_time=DECISION_TIME,
        code_revision="fx-test",
    )

    metrics = result.metrics
    assert metrics.fx_conversion_applied is True
    assert metrics.fx_native_currencies == ["EUR", "USD"]
    # The book itself holds no FX exposure, so its FX contribution is zero
    # while the converted benchmark still rises with the euro.
    assert metrics.fx_contribution_return == pytest.approx(0.0, abs=1e-12)
    assert metrics.benchmark_cumulative_return == pytest.approx(0.2, rel=1e-6)


@pytest.mark.django_db
def test_attribution_identity_survives_trading_and_costs(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    """Rebalancing at dated rates must not break the exact return split."""
    snapshot, listings, store = mixed_currency_universe

    _definition, _run, result = run_simulation_workflow(
        name="Traded mixed book",
        mode="portfolio",
        snapshot=snapshot,
        start_date=date(2026, 2, 2),
        end_date=date(2026, 2, 4),
        starting_capital=123_456.78,
        transaction_cost_bps=12.5,
        slippage_bps=7.5,
        selected_listing_ids=[listing.id for listing in listings.values()],
        base_currency="GBP",
        provider="synthetic_demo",
        asset_store=store,
        decision_time=DECISION_TIME,
        code_revision="fx-test",
    )

    metrics = result.metrics
    assert metrics.base_currency == "GBP"
    assert metrics.fx_attribution_status == FxAttributionStatus.EXACT.value
    assert metrics.fx_local_currency_cumulative_return is not None
    assert metrics.fx_contribution_return is not None
    assert (
        metrics.fx_local_currency_cumulative_return + metrics.fx_contribution_return
    ) == pytest.approx(metrics.cumulative_return, abs=1e-12)
    # Trading costs are charged on converted values, so the stock leg is the
    # negative friction rather than exactly zero on flat native prices.
    assert metrics.fx_local_currency_cumulative_return < 0.0


@pytest.mark.django_db
def test_selected_listings_conflicting_with_a_currency_restriction_are_rejected(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    """Naming holdings and then filtering them away must fail on the spot."""
    snapshot, listings, store = mixed_currency_universe

    with pytest.raises(SimulationWorkflowError, match="would exclude explicitly selected"):
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 2, 2),
            end_date=date(2026, 2, 4),
            listing_ids=[listing.id for listing in listings.values()],
            provider="synthetic_demo",
            base_currency="USD",
            restrict_native_currency="USD",
            decision_time=DECISION_TIME,
            asset_store=store,
        )

    # A restriction consistent with the selection is still accepted.
    panel = build_price_panel(
        snapshot=snapshot,
        start_date=date(2026, 2, 2),
        end_date=date(2026, 2, 4),
        listing_ids=[listings["USDCO"].id],
        provider="synthetic_demo",
        restrict_native_currency="USD",
        decision_time=DECISION_TIME,
        asset_store=store,
    )
    assert panel.native_currencies == ("USD",)
    assert panel.fx_rates is None


@pytest.mark.django_db
def test_benchmark_currency_without_a_benchmark_subject_is_rejected(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    """A stray benchmark currency must not trigger an identity conversion."""
    snapshot, listings, store = mixed_currency_universe

    with pytest.raises(SimulationWorkflowError, match="without a benchmark subject"):
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 2, 2),
            end_date=date(2026, 2, 4),
            listing_ids=[listings["USDCO"].id],
            provider="synthetic_demo",
            benchmark_currency="EUR",
            decision_time=DECISION_TIME,
            asset_store=store,
        )

    unaffected = build_price_panel(
        snapshot=snapshot,
        start_date=date(2026, 2, 2),
        end_date=date(2026, 2, 4),
        listing_ids=[listings["USDCO"].id],
        provider="synthetic_demo",
        decision_time=DECISION_TIME,
        asset_store=store,
    )
    assert unaffected.fx_rates is None
    assert unaffected.prices.columns == [
        "date",
        "listing_id",
        "close",
        "open",
        "symbol",
        "currency",
    ]


@pytest.mark.django_db
def test_carry_limit_is_bounded_and_zero_is_honored(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, store = mixed_currency_universe

    with pytest.raises(SimulationWorkflowError, match="must be between 0 and 7"):
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 2, 2),
            end_date=date(2026, 2, 4),
            provider="synthetic_demo",
            base_currency="USD",
            fx_max_carry_days=14,
            decision_time=DECISION_TIME,
            asset_store=store,
        )

    # Every panel date has its own observation, so a zero-carry run succeeds.
    strict = build_price_panel(
        snapshot=snapshot,
        start_date=date(2026, 2, 2),
        end_date=date(2026, 2, 4),
        provider="synthetic_demo",
        base_currency="USD",
        fx_max_carry_days=0,
        decision_time=DECISION_TIME,
        asset_store=store,
    )
    assert strict.fx_rates is not None
    assert set(strict.fx_rates["carry_days"].to_list()) == {0}

    # Adding a session the FX series never observed makes zero-carry fail.
    _add_listing(
        snapshot,
        store,
        ticker="SATCO",
        currency="USD",
        mic="XNAS",
        sessions=[date(2026, 2, 2), date(2026, 2, 7)],
    )
    with pytest.raises(SimulationWorkflowError, match="0-day carry limit"):
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 2, 2),
            end_date=date(2026, 2, 7),
            provider="synthetic_demo",
            base_currency="USD",
            fx_max_carry_days=0,
            decision_time=DECISION_TIME,
            asset_store=store,
        )


@pytest.mark.django_db
def test_persisted_fx_frame_carries_per_row_cutoff_provenance(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, store = mixed_currency_universe

    _definition, run, _result = run_simulation_workflow(
        name="Provenance",
        mode="portfolio",
        snapshot=snapshot,
        start_date=date(2026, 2, 2),
        end_date=date(2026, 2, 4),
        starting_capital=90_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        selected_listing_ids=[listing.id for listing in listings.values()],
        base_currency="USD",
        provider="synthetic_demo",
        asset_store=store,
        decision_time=DECISION_TIME,
        code_revision="fx-test",
    )

    stored_fx = store.read_frame(run.metrics["input_assets"]["fx"]["relative_path"])
    assert set(stored_fx["evidence_grade"].to_list()) == {"research"}
    for row in stored_fx.iter_rows(named=True):
        assert row["availability_cutoff"].date() == row["value_date"]
        if row["path"] == "identity":
            assert row["rate_published_at"] is None
            continue
        # Nothing published after the valued date may have been used.
        assert row["rate_published_at"] <= row["availability_cutoff"]
        assert row["observation_date"] <= row["value_date"]
        assert row["source_asset_ids"]


@pytest.mark.django_db
def test_observed_snapshot_refuses_a_later_retrieved_fx_bundle(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    """The snapshot's own grade decides whether later retrieval is allowed."""
    snapshot, listings, store = mixed_currency_universe

    observed_snapshot = UniverseSnapshot.objects.create(
        universe=snapshot.universe,
        as_of_date=snapshot.as_of_date,
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="mixed-hash-observed",
    )
    for listing in listings.values():
        UniverseMembership.objects.create(
            snapshot=observed_snapshot, listing=listing, eligible=True
        )

    # The research snapshot converts: its FX bundle was published on time and
    # only retrieved afterwards, which a labeled reconstruction may read.
    assert (
        build_price_panel(
            snapshot=snapshot,
            start_date=date(2026, 2, 2),
            end_date=date(2026, 2, 4),
            provider="synthetic_demo",
            base_currency="USD",
            decision_time=DECISION_TIME,
            asset_store=store,
        ).conversion_applied
        is True
    )

    with pytest.raises(SimulationWorkflowError, match="evidence grade observed"):
        build_price_panel(
            snapshot=observed_snapshot,
            start_date=date(2026, 2, 2),
            end_date=date(2026, 2, 4),
            provider="synthetic_demo",
            base_currency="USD",
            decision_time=DECISION_TIME,
            asset_store=store,
        )


@pytest.mark.django_db
def test_mixed_currency_workflow_is_reproducible(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, store = mixed_currency_universe
    listing_ids = [listing.id for listing in listings.values()]

    hashes = []
    for _index in range(2):
        _definition, run, _result = run_simulation_workflow(
            name="Repeat",
            mode="portfolio",
            snapshot=snapshot,
            start_date=date(2026, 2, 2),
            end_date=date(2026, 2, 4),
            starting_capital=90_000.0,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
            selected_listing_ids=listing_ids,
            base_currency="USD",
            provider="synthetic_demo",
            asset_store=store,
            decision_time=DECISION_TIME,
            code_revision="fx-test",
        )
        hashes.append(run.input_hash)

    assert hashes[0] == hashes[1]

    _definition, later_run, _result = run_simulation_workflow(
        name="Different base",
        mode="portfolio",
        snapshot=snapshot,
        start_date=date(2026, 2, 2),
        end_date=date(2026, 2, 4),
        starting_capital=90_000.0,
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        selected_listing_ids=listing_ids,
        base_currency="EUR",
        provider="synthetic_demo",
        asset_store=store,
        decision_time=DECISION_TIME,
        code_revision="fx-test",
    )
    assert later_run.input_hash != hashes[0]


@pytest.mark.django_db
def test_single_currency_workflow_persists_no_fx_asset(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, store = mixed_currency_universe

    _definition, run, result = run_simulation_workflow(
        name="USD only",
        mode="portfolio",
        snapshot=snapshot,
        start_date=date(2026, 2, 2),
        end_date=date(2026, 2, 4),
        starting_capital=50_000.0,
        selected_listing_ids=[listings["USDCO"].id],
        provider="synthetic_demo",
        asset_store=store,
        decision_time=DECISION_TIME,
        code_revision="fx-test",
    )

    assert result.metrics.base_currency == "USD"
    assert result.metrics.fx_conversion_applied is False
    assert "fx" not in run.metrics["input_assets"]
    stored_prices = store.read_frame(run.metrics["input_assets"]["prices"]["relative_path"])
    assert stored_prices.columns == ["date", "listing_id", "close", "open", "symbol", "currency"]


# ---------------------------------------------------------------------------
# Web and CLI surfaces
# ---------------------------------------------------------------------------


@pytest.fixture
def fx_client() -> Client:
    client = Client()
    user_model = get_user_model()
    user = user_model.objects.create_user(username="fxuser")
    client.force_login(user)
    return client


@pytest.mark.django_db
def test_web_form_requires_base_currency_for_mixed_selection(
    fx_client: Client,
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, _store = mixed_currency_universe

    response = fx_client.post(
        reverse("simulations"),
        {
            "name": "Mixed web run",
            "mode": "portfolio",
            "snapshot": str(snapshot.id),
            "start_date": "2026-02-02",
            "end_date": "2026-02-04",
            "starting_capital": "90000.00",
            "transaction_cost_bps": "0.00",
            "slippage_bps": "0.00",
            "fx_max_carry_days": "7",
            "selected_listings": [str(listing.id) for listing in listings.values()],
        },
    )

    assert response.status_code == 400
    assert "base_currency" in response.context["form"].errors
    assert b"multi-currency selection" in response.content


@pytest.mark.django_db
def test_web_mixed_currency_run_reports_fx_contribution(
    fx_client: Client,
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, store = mixed_currency_universe

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            "stanstock.simulation.builders.AssetStore",
            lambda *args, **kwargs: store,
        )
        patcher.setattr(
            "stanstock.simulation.service.AssetStore",
            lambda *args, **kwargs: store,
        )
        response = fx_client.post(
            reverse("simulations"),
            {
                "name": "Mixed web run",
                "mode": "portfolio",
                "snapshot": str(snapshot.id),
                "start_date": "2026-02-02",
                "end_date": "2026-02-04",
                "starting_capital": "90000.00",
                "transaction_cost_bps": "0.00",
                "slippage_bps": "0.00",
                "base_currency": "USD",
                "fx_max_carry_days": "7",
                "selected_listings": [str(listing.id) for listing in listings.values()],
            },
        )

    assert response.status_code == 302
    run = SimulationRun.objects.get()
    detail = fx_client.get(reverse("simulation-detail", args=[run.id]))
    assert detail.status_code == 200
    assert b"Converted into USD" in detail.content
    assert b"FX contribution" in detail.content
    assert b"base currency USD" in detail.content


@pytest.mark.django_db
def test_simulate_command_converts_and_reports(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot, listings, store = mixed_currency_universe

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            "stanstock.simulation.builders.AssetStore",
            lambda *args, **kwargs: store,
        )
        patcher.setattr(
            "stanstock.simulation.service.AssetStore",
            lambda *args, **kwargs: store,
        )
        call_command(
            "simulate",
            "--name",
            "CLI mixed",
            "--mode",
            "portfolio",
            "--snapshot",
            str(snapshot.id),
            "--start-date",
            "2026-02-02",
            "--end-date",
            "2026-02-04",
            "--cost-bps",
            "0",
            "--slippage-bps",
            "0",
            "--base-currency",
            "usd",
            "--listings",
            ",".join(str(listing.id) for listing in listings.values()),
        )

    output = capsys.readouterr().out
    assert "Converted EUR, GBP, USD into USD" in output
    assert "FX contribution" in output


@pytest.mark.django_db
def test_web_form_rejects_a_restriction_that_excludes_selected_listings(
    fx_client: Client,
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, _store = mixed_currency_universe

    response = fx_client.post(
        reverse("simulations"),
        {
            "name": "Conflicting restriction",
            "mode": "portfolio",
            "snapshot": str(snapshot.id),
            "start_date": "2026-02-02",
            "end_date": "2026-02-04",
            "starting_capital": "90000.00",
            "transaction_cost_bps": "0.00",
            "slippage_bps": "0.00",
            "base_currency": "USD",
            "restrict_native_currency": "USD",
            "fx_max_carry_days": "7",
            "selected_listings": [str(listing.id) for listing in listings.values()],
        },
    )

    assert response.status_code == 400
    errors = response.context["form"].errors
    assert "restrict_native_currency" in errors
    assert "EURCO (EUR)" in str(errors["restrict_native_currency"])


@pytest.mark.django_db
def test_web_form_rejects_benchmark_currency_without_a_subject(
    fx_client: Client,
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, _store = mixed_currency_universe

    response = fx_client.post(
        reverse("simulations"),
        {
            "name": "Stray benchmark currency",
            "mode": "portfolio",
            "snapshot": str(snapshot.id),
            "start_date": "2026-02-02",
            "end_date": "2026-02-04",
            "starting_capital": "90000.00",
            "transaction_cost_bps": "0.00",
            "slippage_bps": "0.00",
            "benchmark_currency": "EUR",
            "fx_max_carry_days": "7",
            "selected_listings": [str(listings["USDCO"].id)],
        },
    )

    assert response.status_code == 400
    assert "benchmark_currency" in response.context["form"].errors
    assert SimulationRun.objects.count() == 0


@pytest.mark.django_db
def test_web_form_rejects_a_carry_limit_above_the_reviewed_maximum(
    fx_client: Client,
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, _store = mixed_currency_universe

    response = fx_client.post(
        reverse("simulations"),
        {
            "name": "Loose carry",
            "mode": "portfolio",
            "snapshot": str(snapshot.id),
            "start_date": "2026-02-02",
            "end_date": "2026-02-04",
            "starting_capital": "90000.00",
            "transaction_cost_bps": "0.00",
            "slippage_bps": "0.00",
            "base_currency": "USD",
            "fx_max_carry_days": "30",
            "selected_listings": [str(listings["USDCO"].id)],
        },
    )

    assert response.status_code == 400
    assert "fx_max_carry_days" in response.context["form"].errors


@pytest.mark.django_db
def test_simulate_command_honors_a_zero_carry_limit(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    """Zero must mean zero, not fall through to the seven-day default."""
    snapshot, listings, store = mixed_currency_universe
    _gap_listing = _add_listing(
        snapshot,
        store,
        ticker="GAPCO",
        currency="USD",
        mic="XNAS",
        sessions=[date(2026, 2, 2), date(2026, 2, 6)],
    )

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            "stanstock.simulation.builders.AssetStore",
            lambda *args, **kwargs: store,
        )
        with pytest.raises(CommandError, match="0-day carry limit"):
            call_command(
                "simulate",
                "--name",
                "CLI zero carry",
                "--mode",
                "portfolio",
                "--snapshot",
                str(snapshot.id),
                "--start-date",
                "2026-02-02",
                "--end-date",
                "2026-02-06",
                "--base-currency",
                "usd",
                "--fx-max-carry-days",
                "0",
                "--listings",
                f"{listings['EURCO'].id},{_gap_listing.id}",
            )


@pytest.mark.django_db
def test_simulate_command_rejects_a_carry_limit_above_the_reviewed_maximum(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, _store = mixed_currency_universe

    with pytest.raises(CommandError, match="between 0 and 7"):
        call_command(
            "simulate",
            "--name",
            "CLI loose carry",
            "--mode",
            "portfolio",
            "--snapshot",
            str(snapshot.id),
            "--start-date",
            "2026-02-02",
            "--end-date",
            "2026-02-04",
            "--base-currency",
            "usd",
            "--fx-max-carry-days",
            "30",
            "--listings",
            str(listings["USDCO"].id),
        )


@pytest.mark.django_db
def test_simulate_command_rejects_conflicting_currency_restriction(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, store = mixed_currency_universe

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            "stanstock.simulation.builders.AssetStore",
            lambda *args, **kwargs: store,
        )
        with pytest.raises(CommandError, match="would exclude explicitly selected"):
            call_command(
                "simulate",
                "--name",
                "CLI conflicting restriction",
                "--mode",
                "portfolio",
                "--snapshot",
                str(snapshot.id),
                "--start-date",
                "2026-02-02",
                "--end-date",
                "2026-02-04",
                "--base-currency",
                "usd",
                "--restrict-native-currency",
                "usd",
                "--listings",
                ",".join(str(listing.id) for listing in listings.values()),
            )


@pytest.mark.django_db
def test_simulate_command_rejects_benchmark_currency_without_a_subject(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, _store = mixed_currency_universe

    with pytest.raises(CommandError, match="requires --benchmark-subject"):
        call_command(
            "simulate",
            "--name",
            "CLI stray benchmark currency",
            "--mode",
            "portfolio",
            "--snapshot",
            str(snapshot.id),
            "--start-date",
            "2026-02-02",
            "--end-date",
            "2026-02-04",
            "--benchmark-currency",
            "eur",
            "--listings",
            str(listings["USDCO"].id),
        )


@pytest.mark.django_db
def test_simulate_command_rejects_mixed_selection_without_a_base_currency(
    mixed_currency_universe: tuple[UniverseSnapshot, dict[str, Listing], AssetStore],
) -> None:
    snapshot, listings, store = mixed_currency_universe

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            "stanstock.simulation.builders.AssetStore",
            lambda *args, **kwargs: store,
        )
        with pytest.raises(CommandError, match="needs an explicit base currency"):
            call_command(
                "simulate",
                "--name",
                "CLI mixed unconverted",
                "--mode",
                "portfolio",
                "--snapshot",
                str(snapshot.id),
                "--start-date",
                "2026-02-02",
                "--end-date",
                "2026-02-04",
                "--listings",
                ",".join(str(listing.id) for listing in listings.values()),
            )


@pytest.mark.django_db
def test_seeded_demo_supports_a_mixed_currency_conversion() -> None:
    call_command("seed_demo")
    snapshot = UniverseSnapshot.objects.get()
    decision_time = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)

    panel = build_price_panel(
        snapshot=snapshot,
        start_date=date(2026, 8, 3),
        end_date=date(2026, 9, 4),
        provider="synthetic_demo",
        base_currency="USD",
        decision_time=decision_time,
    )

    assert panel.conversion_applied is True
    assert set(panel.native_currencies) == {"USD", "EUR", "GBP"}
    assert panel.fx_rates is not None
    # The synthetic bundles publish Friday observations only, so a mid-week
    # session is priced from the preceding Friday.
    assert panel.fx_rates["carry_days"].max() is not None
    assert int(panel.fx_rates["carry_days"].max()) <= 7  # type: ignore[arg-type]
    assert panel.prices.filter(pl.col("currency") == "GBP")["fx_path"].to_list()[0] == "cross:EUR"


@pytest.mark.django_db
def test_seeded_demo_converts_from_its_very_first_session() -> None:
    """The demo's first priced session must have a rate observed before it."""
    call_command("seed_demo")
    snapshot = UniverseSnapshot.objects.get()
    first_session = date(2020, 1, 2)

    anchor = (
        FxRate.objects.filter(base_currency="EUR", quote_currency="USD")
        .order_by("observation_date")
        .first()
    )
    assert anchor is not None
    assert anchor.observation_date <= first_session

    panel = build_price_panel(
        snapshot=snapshot,
        start_date=first_session,
        end_date=date(2020, 1, 31),
        provider="synthetic_demo",
        base_currency="USD",
        decision_time=datetime(2026, 9, 5, 12, 0, tzinfo=UTC),
    )

    assert panel.conversion_applied is True
    opening = panel.prices.filter(pl.col("date") == first_session)
    assert opening.height > 0
    assert set(opening["currency"].to_list()) >= {"EUR", "USD"}
    assert opening["fx_observation_date"].max() <= first_session
    assert int(panel.fx_rates["carry_days"].max()) <= 7  # type: ignore[union-attr,arg-type]

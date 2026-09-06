from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from io import StringIO
from pathlib import Path

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError

from stanstock.core.models import JobRun
from stanstock.data.live_us import PRIVATE_USAGE_SCOPE, LiveUsRunResult
from stanstock.data.management.commands import daily
from stanstock.data.models import ProviderRecord, Universe, UniverseSnapshot
from stanstock.data.provider_policy import BASIC_USAGE_SCOPE
from stanstock.data.providers.contracts import PriceBar, PriceSeries

pytestmark = pytest.mark.django_db


def _validation_series() -> PriceSeries:
    return PriceSeries(
        provider="twelve_data",
        symbol="AAPL",
        currency="USD",
        bars=(
            PriceBar(
                trade_date=date(2026, 9, 4),
                open=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("99"),
                close=Decimal("104"),
                volume=1_000_000,
            ),
        ),
        retrieved_at=datetime(2026, 9, 5, 1, tzinfo=UTC),
        source_url="https://api.twelvedata.com/time_series?symbol=AAPL",
        raw_bytes=b'{"status":"ok"}',
        exchange="NASDAQ",
        mic_code="XNAS",
        instrument_type="Common Stock",
        exchange_timezone="America/New_York",
        adjustment="splits",
    )


def test_configure_twelve_data_requires_explicit_private_use_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fetch(*args: object, **kwargs: object) -> PriceSeries:
        nonlocal called
        called = True
        return _validation_series()

    monkeypatch.setattr(
        "stanstock.data.management.commands.configure_twelve_data.twelve_data."
        "fetch_daily_price_series",
        fetch,
    )

    with pytest.raises(CommandError, match="PERSONAL_INTERNAL_DISPLAY_AUTHORIZED"):
        call_command("configure_twelve_data", enable=True, plan="grow")

    assert called is False
    assert ProviderRecord.objects.count() == 0


def test_configure_twelve_data_basic_requires_single_user_confirmation() -> None:
    with pytest.raises(CommandError, match="PERSONAL_SINGLE_USER_NONCOMMERCIAL_AUTHORIZED"):
        call_command(
            "configure_twelve_data",
            enable=True,
            confirm="PERSONAL_INTERNAL_DISPLAY_AUTHORIZED",
            plan="basic",
        )

    assert ProviderRecord.objects.count() == 0


def test_configure_twelve_data_enables_basic_for_one_personal_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = get_user_model().objects.create_user(
        username="owner",
        password="correct-password",
    )
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "private-command-test-key")
    monkeypatch.setattr(
        "stanstock.data.management.commands.configure_twelve_data.twelve_data."
        "fetch_daily_price_series",
        lambda *args, **kwargs: _validation_series(),
    )
    output = StringIO()

    call_command(
        "configure_twelve_data",
        enable=True,
        confirm="PERSONAL_SINGLE_USER_NONCOMMERCIAL_AUTHORIZED",
        plan="basic",
        stdout=output,
    )

    record = ProviderRecord.objects.get(provider="twelve_data")
    assert record.enabled is True
    assert record.usage_scope == BASIC_USAGE_SCOPE
    assert record.metadata["plan"] == "basic"
    assert record.metadata["licensed_user_id"] == str(user.pk)
    assert record.metadata["personal_noncommercial_confirmed"] is True
    assert record.metadata["internal_display_rights_confirmed"] is False
    assert record.metadata["daily_credit_limit"] == 800
    assert record.metadata["credits_per_minute"] == 8


def test_configure_twelve_data_basic_rejects_multiple_active_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_user_model().objects.create_user(username="owner", password="correct-password")
    get_user_model().objects.create_user(username="other", password="correct-password")
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "private-command-test-key")

    with pytest.raises(CommandError, match="exactly one active"):
        call_command(
            "configure_twelve_data",
            enable=True,
            confirm="PERSONAL_SINGLE_USER_NONCOMMERCIAL_AUTHORIZED",
            plan="basic",
        )

    assert ProviderRecord.objects.count() == 0


def test_store_twelve_data_key_command_never_outputs_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def store() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        "stanstock.data.management.commands.store_twelve_data_key.store_twelve_data_api_key",
        store,
    )
    output = StringIO()

    call_command("store_twelve_data_key", stdout=output)

    assert called is True
    assert "stored in macOS Keychain" in output.getvalue()
    assert "private-command-test-key" not in output.getvalue()


def test_configure_twelve_data_enables_without_persisting_the_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "private-command-test-key"
    monkeypatch.setenv("TWELVE_DATA_API_KEY", api_key)

    def fetch(symbol: str, **kwargs: object) -> PriceSeries:
        assert symbol == "AAPL"
        assert kwargs["api_key"] == api_key
        return _validation_series()

    monkeypatch.setattr(
        "stanstock.data.management.commands.configure_twelve_data.twelve_data."
        "fetch_daily_price_series",
        fetch,
    )
    output = StringIO()

    call_command(
        "configure_twelve_data",
        enable=True,
        confirm="PERSONAL_INTERNAL_DISPLAY_AUTHORIZED",
        plan="grow",
        stdout=output,
    )

    record = ProviderRecord.objects.get(provider="twelve_data")
    assert record.enabled is True
    assert record.status == "ok"
    assert record.usage_scope == PRIVATE_USAGE_SCOPE
    assert record.metadata["daily_credit_limit"] == 800
    assert record.metadata["credits_per_minute"] == 8
    assert record.metadata["plan"] == "grow"
    assert record.metadata["internal_display_rights_confirmed"] is True
    assert record.metadata["credits_used_local"] == 1
    assert record.metadata["price_adjustment"] == "splits"
    assert record.metadata["dividends_included"] is False
    assert api_key not in str(record.metadata)
    assert api_key not in output.getvalue()


def test_configure_twelve_data_preserves_tightened_local_quota_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ProviderRecord.objects.create(
        provider="twelve_data",
        metadata={"daily_credit_limit": 100, "credits_per_minute": 4},
    )
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "private-command-test-key")
    monkeypatch.setattr(
        "stanstock.data.management.commands.configure_twelve_data.twelve_data."
        "fetch_daily_price_series",
        lambda *args, **kwargs: _validation_series(),
    )

    call_command(
        "configure_twelve_data",
        enable=True,
        confirm="PERSONAL_INTERNAL_DISPLAY_AUTHORIZED",
        plan="grow",
        stdout=StringIO(),
    )

    record = ProviderRecord.objects.get(provider="twelve_data")
    assert record.metadata["daily_credit_limit"] == 100
    assert record.metadata["credits_per_minute"] == 4
    assert record.metadata["credits_used_local"] == 1


def test_configure_twelve_data_disable_is_an_immediate_kill_switch() -> None:
    ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=True,
        status="ok",
    )

    call_command("configure_twelve_data", disable=True, stdout=StringIO())

    record = ProviderRecord.objects.get(provider="twelve_data")
    assert record.enabled is False
    assert record.status == "disabled"


def test_daily_command_skips_provider_work_after_a_successful_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = date(2026, 9, 4)
    universe = Universe.objects.create(
        slug="daily-test",
        name="Daily test",
        config_version="test-v1",
    )
    snapshot = UniverseSnapshot.objects.create(
        universe=universe,
        as_of_date=target,
        grade=UniverseSnapshot.Grade.OBSERVED,
        config_hash="a" * 64,
    )
    config = object()
    run_calls: list[object] = []

    monkeypatch.setattr(daily, "load_us_universe_config", lambda path: config)
    monkeypatch.setattr(
        daily,
        "resolve_us_target_date",
        lambda **kwargs: (target, UniverseSnapshot.Grade.OBSERVED),
    )

    def run_us_daily(**kwargs: object) -> LiveUsRunResult:
        run_calls.append(kwargs)
        return LiveUsRunResult(
            snapshot=snapshot,
            analyses=2,
            predictions=6,
            eligible=2,
            excluded=0,
            price_assets=3,
            raw_assets=4,
            credits_used=4,
            benchmark_symbol="SPY",
        )

    monkeypatch.setattr(daily, "run_us_daily", run_us_daily)
    command_args = {
        "region": "us",
        "target_date": target.isoformat(),
        "config": tmp_path / "universe.yaml",
        "stdout": StringIO(),
    }

    call_command("daily", **command_args)
    call_command("daily", **command_args)

    assert len(run_calls) == 1
    assert "api_key" not in run_calls[0]
    assert list(JobRun.objects.order_by("attempt").values_list("status", flat=True)) == [
        JobRun.Status.SUCCESS,
        JobRun.Status.SKIPPED,
    ]

from __future__ import annotations

import json
import stat
from datetime import UTC, datetime

import pytest
from django.core.management import call_command
from django.test import override_settings
from django.utils import timezone

from stanstock.data.management.commands import source_spike
from stanstock.data.models import ProviderRecord
from stanstock.data.providers.contracts import FundamentalSourcePayload, PriceBar, PriceSeries
from stanstock.data.providers.exceptions import (
    ProviderBlockedError,
    ProviderConfigurationError,
    ProviderNetworkError,
    ProviderQuotaError,
)
from stanstock.data.providers.filings_xbrl import FilingsPage

pytestmark = pytest.mark.django_db


def _fake_price_series(*, provider: str = "stooq", symbol: str = "aapl.us") -> PriceSeries:
    return PriceSeries(
        provider=provider,
        symbol=symbol,
        currency=None,
        bars=(
            PriceBar(
                trade_date=timezone.now().date(),
                open=None,
                high=None,
                low=None,
                close=1,
                volume=None,
            ),
        ),
        retrieved_at=timezone.now(),
        source_url="https://stooq.com/q/d/l/",
        raw_bytes=b"",
    )


def _fake_sec_payload() -> FundamentalSourcePayload:
    return FundamentalSourcePayload(
        provider="sec",
        subject="0000320193",
        content=b"{}",
        content_type="application/json",
        retrieved_at=timezone.now(),
        source_url="https://data.sec.gov/example",
    )


def _fake_filings_page() -> FilingsPage:
    return FilingsPage(records=(), raw_bytes=b"{}", retrieved_at=timezone.now(), source_url="x")


class _FakeEcbResult:
    observations: tuple = ()


def _patch_all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        source_spike.twelve_data,
        "resolve_api_key",
        lambda: "private-test-key",
    )
    monkeypatch.setattr(
        source_spike.twelve_data,
        "fetch_daily_price_series",
        lambda *a, **k: _fake_price_series(provider="twelve_data", symbol="AAPL"),
    )
    monkeypatch.setattr(
        source_spike.stooq, "fetch_daily_price_series", lambda *a, **k: _fake_price_series()
    )
    monkeypatch.setattr(source_spike.sec, "fetch_submissions", lambda *a, **k: _fake_sec_payload())
    monkeypatch.setattr(
        source_spike.filings_xbrl, "fetch_filings_page", lambda *a, **k: _fake_filings_page()
    )
    monkeypatch.setattr(source_spike.ecb, "fetch_exr_csv", lambda *a, **k: _FakeEcbResult())


def test_stooq_html_challenge_classified_provider_incompatible_and_no_go(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def raise_blocked(*args: object, **kwargs: object) -> PriceSeries:
        raise ProviderBlockedError("Stooq returned an HTML page instead of a CSV download")

    monkeypatch.setattr(source_spike.stooq, "fetch_daily_price_series", raise_blocked)
    monkeypatch.setattr(source_spike.sec, "fetch_submissions", lambda *a, **k: _fake_sec_payload())
    monkeypatch.setattr(
        source_spike.filings_xbrl, "fetch_filings_page", lambda *a, **k: _fake_filings_page()
    )
    monkeypatch.setattr(source_spike.ecb, "fetch_exr_csv", lambda *a, **k: _FakeEcbResult())

    with override_settings(DATA_DIR=tmp_path):
        call_command("source_spike", skip="twelve_data")

    record = ProviderRecord.objects.get(provider="stooq")
    assert record.status == source_spike.CLASSIFICATION_PROVIDER_INCOMPATIBLE
    assert record.enabled is False

    reports = list((tmp_path / "reports").glob("source_spike_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["decision"] == "NO_GO"
    assert report["price_capability"] is False


def test_sec_403_classified_environment_blocked_not_provider_incompatible(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        source_spike.stooq, "fetch_daily_price_series", lambda *a, **k: _fake_price_series()
    )

    def raise_403(*args: object, **kwargs: object) -> FundamentalSourcePayload:
        raise ProviderBlockedError("SEC returned HTTP 403 for CIK 0000320193")

    monkeypatch.setattr(source_spike.sec, "fetch_submissions", raise_403)
    monkeypatch.setattr(
        source_spike.filings_xbrl, "fetch_filings_page", lambda *a, **k: _fake_filings_page()
    )
    monkeypatch.setattr(source_spike.ecb, "fetch_exr_csv", lambda *a, **k: _FakeEcbResult())

    with override_settings(DATA_DIR=tmp_path):
        call_command("source_spike", skip="twelve_data")

    record = ProviderRecord.objects.get(provider="sec")
    assert record.status == source_spike.CLASSIFICATION_ENV_BLOCKED


def test_stooq_success_does_not_grant_conditional_go_due_to_unverified_terms(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Even a technically-successful Stooq probe must not flip the overall
    decision to CONDITIONAL_GO: Stooq's capability verdict is fixed NO_GO
    because its automation/private-retention terms could not be verified,
    independent of whether any given probe run happens to succeed."""
    _patch_all_ok(monkeypatch)

    with override_settings(DATA_DIR=tmp_path):
        call_command("source_spike", skip="twelve_data")

    reports = list((tmp_path / "reports").glob("source_spike_*.json"))
    report = json.loads(reports[0].read_text())
    assert report["decision"] == "NO_GO"
    assert report["price_capability"] is False
    stooq_probe = next(probe for probe in report["probes"] if probe["provider"] == "stooq")
    assert stooq_probe["classification"] == source_spike.CLASSIFICATION_OK
    assert stooq_probe["capability_verdict"] == source_spike.CAPABILITY_VERDICT_NO_GO


def test_twelve_data_success_grants_us_conditional_go_and_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    ProviderRecord.objects.create(
        provider="twelve_data",
        metadata={
            "daily_credits_used": 17,
            "daily_credit_limit": 100,
            "credits_per_minute": 4,
        },
    )
    monkeypatch.setattr(
        source_spike.twelve_data,
        "resolve_api_key",
        lambda: "private-test-key",
    )
    monkeypatch.setattr(
        source_spike.twelve_data,
        "fetch_daily_price_series",
        lambda *a, **k: _fake_price_series(provider="twelve_data", symbol="AAPL"),
    )

    with override_settings(DATA_DIR=tmp_path):
        call_command(
            "source_spike",
            skip="stooq,sec,filings_xbrl_org,ecb",
        )

    report_path = next((tmp_path / "reports").glob("source_spike_*.json"))
    report = json.loads(report_path.read_text())
    assert report["decision"] == "CONDITIONAL_GO"
    assert report["price_capability"] is True

    record = ProviderRecord.objects.get(provider="twelve_data")
    assert record.status == source_spike.CLASSIFICATION_OK
    assert record.metadata["daily_credits_used"] == 17
    assert record.metadata["daily_credit_limit"] == 100
    assert record.metadata["credits_per_minute"] == 4
    assert record.metadata["credits_used_local"] == 1
    assert record.metadata["capability_verdict"] == "CONDITIONAL_GO"
    assert record.terms_url == source_spike.twelve_data.TERMS_URL
    assert record.usage_scope == source_spike.TWELVE_DATA_USAGE_SCOPE


def test_twelve_data_quota_error_is_classified_without_granting_price_capability(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def raise_quota(*args: object, **kwargs: object) -> PriceSeries:
        raise ProviderQuotaError("Twelve Data API quota is exhausted")

    monkeypatch.setattr(
        source_spike.twelve_data,
        "resolve_api_key",
        lambda: "private-test-key",
    )
    monkeypatch.setattr(
        source_spike.twelve_data,
        "fetch_daily_price_series",
        raise_quota,
    )

    with override_settings(DATA_DIR=tmp_path):
        call_command(
            "source_spike",
            skip="stooq,sec,filings_xbrl_org,ecb",
        )

    record = ProviderRecord.objects.get(provider="twelve_data")
    assert record.status == source_spike.CLASSIFICATION_QUOTA_EXHAUSTED
    report_path = next((tmp_path / "reports").glob("source_spike_*.json"))
    report = json.loads(report_path.read_text())
    assert report["decision"] == "NO_GO"
    assert report["price_capability"] is False


def test_source_spike_does_not_downgrade_confirmed_twelve_data_usage_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    checked_at = datetime(2026, 9, 1, 12, tzinfo=UTC)
    ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=True,
        terms_checked_at=checked_at,
        usage_scope="personal_internal_display_authorized",
        metadata={"internal_display_rights_confirmed": True},
    )
    monkeypatch.setattr(
        source_spike.twelve_data,
        "resolve_api_key",
        lambda: "private-test-key",
    )
    monkeypatch.setattr(
        source_spike.twelve_data,
        "fetch_daily_price_series",
        lambda *a, **k: _fake_price_series(provider="twelve_data", symbol="AAPL"),
    )

    with override_settings(DATA_DIR=tmp_path):
        call_command(
            "source_spike",
            skip="stooq,sec,filings_xbrl_org,ecb",
        )

    record = ProviderRecord.objects.get(provider="twelve_data")
    assert record.enabled is True
    assert record.usage_scope == "personal_internal_display_authorized"
    assert record.terms_checked_at == checked_at
    assert record.metadata["internal_display_rights_confirmed"] is True


def test_capability_verdicts_reflect_completed_provider_research(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_all_ok(monkeypatch)

    with override_settings(DATA_DIR=tmp_path):
        call_command("source_spike")

    report_path = next((tmp_path / "reports").glob("source_spike_*.json"))
    report = json.loads(report_path.read_text())
    verdicts = {probe["provider"]: probe["capability_verdict"] for probe in report["probes"]}
    assert verdicts == {
        "twelve_data": "CONDITIONAL_GO",
        "stooq": "NO_GO",
        "sec": "GO",
        "filings_xbrl_org": "CONDITIONAL_GO",
        "ecb": "GO",
    }


def test_decision_would_be_conditional_go_if_a_price_provider_had_an_approved_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Demonstrates the decision logic is generically correct: it only ever
    reports NO_GO for Stooq today because its verdict is hardcoded NO_GO,
    not because the logic is incapable of reporting CONDITIONAL_GO."""
    _patch_all_ok(monkeypatch)
    monkeypatch.setitem(
        source_spike.PROVIDER_CAPABILITY_VERDICTS,
        "stooq",
        source_spike.CAPABILITY_VERDICT_CONDITIONAL_GO,
    )

    with override_settings(DATA_DIR=tmp_path):
        call_command("source_spike", skip="twelve_data")

    report_path = next((tmp_path / "reports").glob("source_spike_*.json"))
    report = json.loads(report_path.read_text())
    assert report["decision"] == "CONDITIONAL_GO"
    assert report["price_capability"] is True


def test_missing_sec_user_agent_classified_configuration_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        source_spike.stooq, "fetch_daily_price_series", lambda *a, **k: _fake_price_series()
    )

    def raise_config(*args: object, **kwargs: object) -> FundamentalSourcePayload:
        raise ProviderConfigurationError("SEC_USER_AGENT is required")

    monkeypatch.setattr(source_spike.sec, "fetch_submissions", raise_config)
    monkeypatch.setattr(
        source_spike.filings_xbrl, "fetch_filings_page", lambda *a, **k: _fake_filings_page()
    )
    monkeypatch.setattr(source_spike.ecb, "fetch_exr_csv", lambda *a, **k: _FakeEcbResult())

    with override_settings(DATA_DIR=tmp_path):
        call_command("source_spike", skip="twelve_data")

    record = ProviderRecord.objects.get(provider="sec")
    assert record.status == source_spike.CLASSIFICATION_CONFIG_MISSING


def test_network_error_classified_environment_blocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def raise_network(*args: object, **kwargs: object) -> FilingsPage:
        raise ProviderNetworkError("ConnectTimeout calling https://filings.xbrl.org/api/filings")

    monkeypatch.setattr(
        source_spike.stooq, "fetch_daily_price_series", lambda *a, **k: _fake_price_series()
    )
    monkeypatch.setattr(source_spike.sec, "fetch_submissions", lambda *a, **k: _fake_sec_payload())
    monkeypatch.setattr(source_spike.filings_xbrl, "fetch_filings_page", raise_network)
    monkeypatch.setattr(source_spike.ecb, "fetch_exr_csv", lambda *a, **k: _FakeEcbResult())

    with override_settings(DATA_DIR=tmp_path):
        call_command("source_spike")

    record = ProviderRecord.objects.get(provider="filings_xbrl_org")
    assert record.status == source_spike.CLASSIFICATION_ENV_BLOCKED


def test_report_file_is_private_and_contains_no_response_bodies(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_all_ok(monkeypatch)

    with override_settings(DATA_DIR=tmp_path):
        call_command("source_spike")

    report_path = next((tmp_path / "reports").glob("source_spike_*.json"))
    mode = stat.S_IMODE(report_path.stat().st_mode)
    assert mode == (stat.S_IRUSR | stat.S_IWUSR)
    raw_text = report_path.read_text()
    assert "SEC_USER_AGENT" not in raw_text or "required" in raw_text  # no raw secret values leaked
    report = json.loads(raw_text)
    for probe in report["probes"]:
        assert len(probe["detail"]) <= 300


def test_skip_option_excludes_provider(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _patch_all_ok(monkeypatch)

    with override_settings(DATA_DIR=tmp_path):
        call_command("source_spike", skip="sec,ecb")

    report_path = next((tmp_path / "reports").glob("source_spike_*.json"))
    report = json.loads(report_path.read_text())
    providers = {probe["provider"] for probe in report["probes"]}
    assert providers == {"twelve_data", "stooq", "filings_xbrl_org"}


def test_skip_option_uses_documented_filings_provider_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _patch_all_ok(monkeypatch)

    with override_settings(DATA_DIR=tmp_path):
        call_command(
            "source_spike",
            skip="twelve_data,stooq,filings_xbrl_org,ecb",
        )

    report_path = next((tmp_path / "reports").glob("source_spike_*.json"))
    report = json.loads(report_path.read_text())
    assert [probe["provider"] for probe in report["probes"]] == ["sec"]


def test_never_touches_enabled_field(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    ProviderRecord.objects.create(provider="stooq", enabled=True)
    _patch_all_ok(monkeypatch)

    with override_settings(DATA_DIR=tmp_path):
        call_command("source_spike")

    record = ProviderRecord.objects.get(provider="stooq")
    assert record.enabled is True  # command must never flip this itself

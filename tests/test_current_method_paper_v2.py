"""Offline v2 acceptance and same-runtime, full-surface frozen-v1 differential."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
from datetime import date, timedelta
from decimal import Decimal, localcontext
from io import StringIO
from pathlib import Path

# Disposable subprocesses own migrated databases. Never open the runtime DB or
# disable immutable-evidence triggers to accommodate Django's transaction flush.
if __name__ == "__main__":
    os.environ["DJANGO_SETTINGS_MODULE"] = "stanstock.settings.test"
    if sys.argv[1] in {"v1", "replay-v1"}:
        import uuid
        from itertools import count

        _probe_uuid_sequence = count(1)

        def _probe_uuid():
            return uuid.UUID(int=next(_probe_uuid_sequence))

        # Before model defaults AND directly imported uuid4 issuance functions.
        uuid.uuid4 = _probe_uuid
    from django.conf import settings as probe_settings

    probe_settings.DATABASES["default"]["NAME"] = str(Path(sys.argv[2]) / "paper.sqlite3")
    probe_settings.DATA_DIR = Path(sys.argv[2]) / "assets"
    import django

    django.setup()

import polars as pl
import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import transaction
from django.utils import timezone
from exchange_calendars import get_calendar

from stanstock.core.models import JobRun
from stanstock.data.asof import PriceFrameChecksumMismatchError
from stanstock.data.assets import AssetStore, asset_ref_for, register_asset
from stanstock.data.models import DataAsset, Listing, ProviderRecord
from stanstock.portfolio.models import (
    Portfolio,
    PortfolioDeposit,
    PortfolioHolding,
    PortfolioPurchase,
)
from stanstock.research.forward_scorecard import declare_forward_protocol
from stanstock.simulation import paper

if __name__ != "__main__" or sys.argv[1] not in {"v1", "replay-v1"}:
    from stanstock.simulation import paper_jobs, paper_v2
from stanstock.simulation.models import SimulationDefinition, SimulationRun
from test_current_method_paper import _capture_prices, _paper_environment
from test_research_product_jobs import NOW

pytestmark = pytest.mark.django_db
BASE_SHA = "f8008c584c4847776c1489965a22c67e25d61fda"
V2_PIN = "e1839e1bdd25236de47dcaa421def82fdfb98e0ac46b7d0f7261cf74fee22a61"
V1_PIN = "25a6a2b5f6e0b37ab2f55d0fced023620d31614433fa16d0e8e3ea545318c835"
REVISION = "a" * 40  # deterministic synthetic execution binding, not native activation


def _environment(tmp_path, monkeypatch, django_user_model, settings, *, empty=False):
    owner, store, run = _paper_environment(
        tmp_path, monkeypatch, django_user_model, settings, empty=empty
    )
    settings.OWNER_USERNAME = owner.username
    revision = subprocess.run(
        ["git", "--no-pager", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    monkeypatch.setattr(paper_v2, "clean_git_revision", lambda _root: revision)
    declare_forward_protocol(user=owner, store=store)
    return owner, store, run


@pytest.fixture
def environment(tmp_path, monkeypatch, django_user_model, settings):
    return _environment(tmp_path, monkeypatch, django_user_model, settings)


def _activate(environment):
    owner, store, _run = environment
    return paper_v2.activate_current_method_paper(user=owner, store=store)


def _days(activation):
    return [date.fromisoformat(day) for day in activation.document["calendar"]]


def _capture(environment, activation, monkeypatch, day, **kwargs):
    owner, store, _run = environment
    _capture_prices(activation, store, monkeypatch, day, **kwargs)
    result = paper_jobs.execute_paper_capture_job(
        owner=owner, session=day, store=store, activation_asset=activation.asset
    )
    assert result.state == "captured", result.reason_code
    assert result.child.status == "success"
    return result


def _files(store):
    return {
        str(path.relative_to(store.root)): path.read_bytes()
        for path in store.root.rglob("*")
        if path.is_file()
    }


def _assert_no_generic_rows():
    for model in (
        Portfolio,
        PortfolioHolding,
        PortfolioPurchase,
        PortfolioDeposit,
        SimulationDefinition,
        SimulationRun,
    ):
        assert model.objects.count() == 0, model.__name__


def test_provider_error_facade_preserves_identity_hierarchy_and_sanitization():
    from stanstock.data import provider_policy
    from stanstock.data.providers import exceptions

    assert provider_policy.ProviderError is exceptions.ProviderError
    assert paper_v2.ProviderError is provider_policy.ProviderError
    assert paper_jobs.ProviderError is provider_policy.ProviderError
    for error_type in (
        exceptions.ProviderError,
        exceptions.ProviderConfigurationError,
        exceptions.ProviderNetworkError,
        exceptions.ProviderBlockedError,
        exceptions.ProviderQuotaError,
        exceptions.ProviderResponseError,
        exceptions.ProviderDataError,
    ):
        error = error_type("private synthetic provider error")
        assert isinstance(error, provider_policy.ProviderError)
        sanitized = paper_jobs._safe_error(error)
        assert sanitized.reason_code == "paper_v2_integrity_invalid"
        assert str(sanitized) == "Paper v2 evidence withheld (paper_v2_integrity_invalid)"


def test_v2_persistence_reader_command_terminal_survives_gap(environment, monkeypatch):
    """One real source -> persistence -> strict requery -> actual CLI flow."""
    owner, store, _run = environment
    assert len(paper_v2.PAPER_POLICY) == 28
    assert type(paper_v2.PAPER_POLICY["basis_reference_closes"]) is int
    assert hashlib.sha256(paper_v2._bytes(paper_v2.PAPER_POLICY)).hexdigest() == V2_PIN
    assert hashlib.sha256(paper._bytes(paper.PAPER_POLICY)).hexdigest() == V1_PIN
    output = StringIO()
    call_command(
        "paper_portfolio",
        owner=owner.username,
        policy_version=paper_v2.PAPER_POLICY_VERSION,
        activate=True,
        stdout=output,
    )
    first_report = json.loads(output.getvalue())
    assert first_report["terminal"]["status"] == "unmatured"
    assert first_report["terminal"]["financials"] is None
    assert first_report["path"]["status"] == "awaiting_inception"
    activation = paper_v2.read_activation(user=owner, store=store)
    days = _days(activation)
    entry = _capture(environment, activation, monkeypatch, days[0])
    frozen = {
        asset.id: (store.read_bytes(asset.relative_path), asset.metadata)
        for asset in DataAsset.objects.filter(
            kind__in=(paper_v2.ACTIVATION_KIND, paper_v2.EXECUTION_KIND, paper_v2.MEASUREMENT_KIND)
        )
    }
    _capture_prices(activation, store, monkeypatch, days[-1], growth=1.001)
    output = StringIO()
    call_command(
        "paper_portfolio",
        owner=owner.username,
        policy_version=paper_v2.PAPER_POLICY_VERSION,
        session=days[-1],
        stdout=output,
    )
    report = json.loads(output.getvalue())
    assert report == paper_v2.read_current_method_paper(user=owner, store=store)
    assert report["terminal"]["status"] == "available"
    assert report["path"]["status"] == "withheld"
    assert report["path"]["risk"] is None
    assert report["missing_sessions"] == [day.isoformat() for day in days[1:-1]]
    assert (report["calendar_size"], report["due_count"], report["captured_count"]) == (127, 127, 2)
    financials = report["terminal"]["financials"]
    with localcontext(paper_v2._DECIMAL_CONTEXT):
        a = Decimal(financials["account_value_usd"])
        s = Decimal(financials["full_spy_value_usd"])
        k = Decimal(financials["stock_or_cash_value_usd"])
        assert a == s * 7 / 10 + k
        assert a - s == k - s * 3 / 10 == Decimal(financials["account_minus_spy_usd"])
        assert a - 100000 == (
            Decimal(financials["fixed_quantity_price_pnl_usd"])
            - Decimal(financials["entry_costs_usd"])
            + Decimal(financials["native_entry_roundoff_usd"])
        )
    state = paper_v2.read_state(user=owner, store=store)
    assert state.captures[0].asset.id == entry.capture.asset.id
    assert len(state.execution.document["reference"]["dates"]) == 757
    for capture in state.captures:
        assert set(capture.document) == paper_v2.HEADER_FIELDS | {
            "activation",
            "execution",
            "origin_job_id",
            "session",
            "marks",
        }
        assert "reference" not in capture.document
        assert len(capture.document["marks"]) == 3
        origin = JobRun.objects.get(pk=capture.document["origin_job_id"])
        assert origin.details["paper_v2_origin"] == paper_v2._origin_binding(
            activation, state.execution, capture
        )
        assert len(origin.job_name) == 73
        assert origin.target_date == date.fromisoformat(capture.document["session"])
        for mark in capture.document["marks"]:
            assert mark["native_currency"] == "USD"
            assert type(mark["applied_fx_rate"]) is int and mark["applied_fx_rate"] == 1
            source = DataAsset.objects.get(pk=mark["source_asset"]["id"])
            raw = DataAsset.objects.get(pk=mark["raw_asset"]["id"])
            assert (
                source.sha256 == hashlib.sha256(store.read_bytes(source.relative_path)).hexdigest()
            )
            assert raw.sha256 == hashlib.sha256(store.read_bytes(raw.relative_path)).hexdigest()
    for asset_id, (payload, metadata) in frozen.items():
        asset = DataAsset.objects.get(pk=asset_id)
        assert (store.read_bytes(asset.relative_path), asset.metadata) == (payload, metadata)
    _assert_no_generic_rows()


@pytest.mark.parametrize("empty", (False, True))
def test_valid_empty_is_cash_not_invalid_or_incomplete_selection(
    tmp_path, monkeypatch, django_user_model, settings, empty
):
    env = _environment(tmp_path, monkeypatch, django_user_model, settings, empty=empty)
    owner, store, _run = env
    activation = _activate(env)
    result = _capture(env, activation, monkeypatch, _days(activation)[0])
    assert len(activation.document["buys"]) == (0 if empty else 2)
    state = paper_v2.read_state(user=owner, store=store)
    assert (state.execution.document["native_executions"]["stocks"] is None) is empty
    value = paper_v2._valuation(state.execution, result.capture, empty=empty)
    assert Decimal(value["account_pnl_usd"]) < 0
    assert Decimal(value["fixed_quantity_price_pnl_usd"]) == 0
    assert Decimal(value["entry_costs_usd"]) > 0
    assert value["effect_label"] == (
        "cash_allocation_effect" if empty else "stock_vs_same_dollar_SPY"
    )
    assert (value["stock_selection_excess_usd"] is None) is empty
    risk = paper_v2.read_current_method_paper(user=owner, store=store)["path"]["risk"]
    assert risk["account"]["max_drawdown"] < 0
    assert risk["account"]["annualized_volatility"] == 0.0
    assert risk["account"]["volatility_status"] == "native_zero_fewer_than_two_return_intervals"
    assert risk["account"]["sharpe_ratio"] is None
    _assert_no_generic_rows()


def test_terminal_due_awaiting_and_expired_states_have_null_financials(environment, monkeypatch):
    owner, store, _run = environment
    activation = _activate(environment)
    _capture(environment, activation, monkeypatch, _days(activation)[0])
    terminal = _days(activation)[-1]
    calendar = get_calendar("XNYS")
    close = calendar.session_close(terminal).to_pydatetime()
    for at, status in (
        (close - timedelta(seconds=1), "unmatured"),
        (close + timedelta(seconds=1), "awaiting_capture"),
        (calendar.session_open(calendar.next_session(terminal)).to_pydatetime(), "withheld"),
    ):
        monkeypatch.setattr(timezone, "now", lambda at=at: at)
        report = paper_v2.read_current_method_paper(user=owner, store=store)
        assert report["terminal"]["status"] == status
        assert report["terminal"]["financials"] is None
        assert report["path"]["risk"] is None


def test_full_127_prospective_captures_native_exact_accounting_and_risk(environment, monkeypatch):
    owner, store, _run = environment
    activation = _activate(environment)
    for index, day in enumerate(_days(activation)):
        _capture(environment, activation, monkeypatch, day, growth=1.001 if index % 2 else 0.999)
    report = paper_v2.read_current_method_paper(user=owner, store=store)
    assert report["terminal"]["status"] == report["path"]["status"] == "available"
    assert report["missing_sessions"] == []
    assert report["captured_count"] == 127
    risk = report["path"]["risk"]
    assert risk["capture_count"] == 127 and risk["return_intervals"] == 126
    assert risk["through_session"] == report["terminal_session"]
    state = paper_v2.read_state(user=owner, store=store)
    valuations = [paper_v2._valuation(state.execution, row, empty=False) for row in state.captures]
    assert len({row["cash_usd"] for row in valuations}) == 1
    assert len({row["entry_costs_usd"] for row in valuations}) == 1
    assert len({row["native_entry_roundoff_usd"] for row in valuations}) == 1
    for key, label in (("account_value_usd", "account"), ("full_spy_value_usd", "spy")):
        values = [float(Decimal(row[key])) for row in valuations]
        curve = pl.DataFrame(
            {
                "date": _days(activation),
                "portfolio_value": values,
                "daily_return": [0.0] + [values[i] / values[i - 1] - 1 for i in range(1, 127)],
            }
        )
        native = paper_v2.calculate_simulation_metrics(
            daily_curves=curve,
            trades=pl.DataFrame(),
            unresolved_observations=[],
            config=paper_v2._config(100000.0, ["SPY"]),
        )
        assert risk[label]["max_drawdown"] == native.max_drawdown
        assert risk[label]["annualized_volatility"] == native.annualized_volatility
        assert risk[label]["sharpe_ratio"] == native.sharpe_ratio
        assert not {"alpha", "beta", "tracking_error", "information_ratio"} & risk[label].keys()
    assert DataAsset.objects.filter(kind=paper_v2.MEASUREMENT_KIND).count() == 127
    _assert_no_generic_rows()


def test_v2_preview_unknown_version_and_repair_are_fail_closed(environment):
    owner, store, _run = environment
    before = (DataAsset.objects.count(), JobRun.objects.count(), _files(store))
    with pytest.raises(CommandError, match="withheld"):
        call_command(
            "paper_portfolio",
            owner=owner.username,
            policy_version=paper_v2.PAPER_POLICY_VERSION,
            preview=True,
            stdout=StringIO(),
        )
    for args in (("--policy-version", "unknown"), ("--repair",)):
        with pytest.raises(CommandError):
            call_command("paper_portfolio", *args, stdout=StringIO())
    for options in (
        {"policy_version": "unknown", "activate": True},
        {
            "policy_version": paper_v2.PAPER_POLICY_VERSION,
            "activate": True,
            "session": date(2026, 9, 14),
        },
    ):
        with pytest.raises(CommandError):
            call_command("paper_portfolio", owner=owner.username, stdout=StringIO(), **options)
    assert (DataAsset.objects.count(), JobRun.objects.count(), _files(store)) == before


def _price_variant(
    activation,
    store,
    monkeypatch,
    session,
    *,
    fault=None,
    short=False,
    false_bounds=False,
    metadata=None,
    same_clock=False,
):
    """Native raw + normalized immutable vintages; never update an asset row."""
    # Target a selected STOCK, not whichever random UUID sorts before SPY.
    listing_id = activation.document["buys"][0]["listing_id"]
    ref = paper_v2._references(activation)[listing_id]
    source = DataAsset.objects.filter(
        kind="price_history", subject=ref["subject"], period_end=session
    ).latest("available_at", "retrieved_at")
    raw_source = DataAsset.objects.get(pk=source.metadata["raw_asset_id"])
    payload = json.loads(store.read_bytes(raw_source.relative_path))
    frame = store.read_frame(source.relative_path)
    if short:
        payload["values"] = payload["values"][-2:]
        frame = frame.tail(2)
    if fault is not None:
        payload, frame = fault(payload, frame)
    now = timezone.now() + (timedelta(0) if same_clock else timedelta(seconds=10))
    stamp = len(_files(store))
    raw_stored = store.write_bytes(f"synthetic-variants/{stamp}.json", json.dumps(payload).encode())
    start = source.period_start if false_bounds else frame["date"][0]
    raw = register_asset(
        provider=source.provider,
        kind="raw_price_history",
        subject=source.subject,
        stored=raw_stored,
        retrieved_at=now,
        available_at=now,
        period_start=start,
        period_end=session,
        metadata=raw_source.metadata,
    )
    normalized = register_asset(
        provider=source.provider,
        kind="price_history",
        subject=source.subject,
        stored=store.write_frame(f"synthetic-variants/{stamp}.parquet", frame),
        retrieved_at=now,
        available_at=now,
        period_start=start,
        period_end=session,
        metadata={
            **source.metadata,
            "raw_asset_id": str(raw.id),
            "raw_sha256": raw.sha256,
            **(metadata or {}),
        },
    )
    monkeypatch.setattr(timezone, "now", lambda: now + timedelta(seconds=1))
    return listing_id, normalized, raw


def _rewrite_close(day, multiplier):
    def fault(payload, frame):
        for row in payload["values"]:
            if row["datetime"] == day.isoformat():
                row.update(
                    {
                        name: str(float(row[name]) * multiplier)
                        for name in ("open", "high", "low", "close")
                    }
                )
        frame = frame.with_columns(
            *[
                pl.when(pl.col("date") == day)
                .then(pl.col(name) * multiplier)
                .otherwise(pl.col(name))
                .alias(name)
                for name in ("open", "high", "low", "close")
            ]
        )
        return payload, frame

    return fault


@pytest.mark.parametrize(
    "case",
    (
        "two_prices",
        "reference_gap",
        "held_gap",
        "reference_conflict",
        "entry_conflict",
        "history_conflict",
        "recorded_overlap_conflict",
        "raw_normalized_conflict",
        "malformed_raw_with_valid_checksum",
    ),
)
def test_full_basis_not_endpoints_and_no_corruption_fallback(environment, monkeypatch, case):
    owner, store, _run = environment
    activation = _activate(environment)
    days = _days(activation)
    _capture(environment, activation, monkeypatch, days[0])
    if case == "recorded_overlap_conflict":
        _capture(environment, activation, monkeypatch, days[1])
    session = days[2]
    _capture_prices(activation, store, monkeypatch, session)
    ref_dates = paper_v2._reference_dates(activation)

    def sparse(payload, frame):
        keep = {ref_dates[0].isoformat(), session.isoformat()}
        payload["values"] = [row for row in payload["values"] if row["datetime"] in keep]
        return payload, frame.filter(pl.col("date").is_in([ref_dates[0], session]))

    def gap(day):
        def fault(payload, frame):
            payload["values"] = [
                row for row in payload["values"] if row["datetime"] != day.isoformat()
            ]
            return payload, frame.filter(pl.col("date") != day)

        return fault

    original = DataAsset.objects.get(
        pk=paper_v2._references(activation)[activation.document["buys"][0]["listing_id"]]["id"]
    )
    original_start = original.period_start
    faults = {
        "two_prices": sparse,
        "reference_gap": gap(ref_dates[10]),
        "held_gap": gap(days[1]),
        "reference_conflict": _rewrite_close(ref_dates[10], 0.5),
        "entry_conflict": _rewrite_close(days[0], 0.5),
        "history_conflict": _rewrite_close(original_start, 0.5),
        "recorded_overlap_conflict": _rewrite_close(days[1], 0.5),
        "raw_normalized_conflict": lambda payload, frame: (
            payload,
            frame.with_columns((pl.col("close") * 0.5).alias("close")),
        ),
        "malformed_raw_with_valid_checksum": lambda payload, frame: (
            {**payload, "status": "error", "message": "private synthetic parser text"},
            frame,
        ),
    }
    _listing_id, winner, _raw = _price_variant(
        activation, store, monkeypatch, session, fault=faults[case]
    )
    result = paper_jobs.execute_paper_capture_job(owner=owner, session=session, store=store)
    assert result.state == "failed"
    assert result.child.status == "failed"
    assert result.reason_code in paper_v2.SAFE_CODES
    assert not DataAsset.objects.filter(kind=paper_v2.MEASUREMENT_KIND, period_end=session).exists()
    assert DataAsset.objects.filter(kind=paper_v2.EXECUTION_KIND).count() == 1
    assert winner.period_end == session
    assert "private synthetic parser text" not in result.child.error
    if case == "malformed_raw_with_valid_checksum":
        assert result.reason_code == "paper_v2_source_unverified"
        assert result.child.error == (
            "PaperV2Error: Paper v2 evidence withheld (paper_v2_source_unverified)"
        )


@pytest.mark.parametrize("false_bounds", (False, True))
def test_legitimate_short_asset_is_ineligible_but_false_sufficient_bounds_fail(
    environment, monkeypatch, false_bounds
):
    owner, store, _run = environment
    activation = _activate(environment)
    day = _days(activation)[0]
    _capture_prices(activation, store, monkeypatch, day)
    listing_id, short, _raw = _price_variant(
        activation, store, monkeypatch, day, short=True, false_bounds=false_bounds
    )
    result = paper_jobs.execute_paper_capture_job(owner=owner, session=day, store=store)
    if false_bounds:
        assert result.state == "failed"
        assert not DataAsset.objects.filter(kind=paper_v2.EXECUTION_KIND).exists()
    else:
        assert result.state == "captured"
        used = next(
            row for row in result.capture.document["marks"] if row["listing_id"] == listing_id
        )
        assert used["source_asset"]["id"] != str(short.pk)


@pytest.mark.parametrize(
    "metadata",
    (
        {"currency": "EUR"},
        {"instrument_type": "ETF"},
        {"adjustment": "all"},
        {"dividends_included": True},
        {"mic_code": "XNYS"},
        {"return_definition": "total_return"},
    ),
)
def test_newest_eligible_wrong_identity_cannot_search_back(environment, monkeypatch, metadata):
    owner, store, _run = environment
    activation = _activate(environment)
    day = _days(activation)[0]
    _capture_prices(activation, store, monkeypatch, day)
    _price_variant(activation, store, monkeypatch, day, metadata=metadata)
    result = paper_jobs.execute_paper_capture_job(owner=owner, session=day, store=store)
    assert result.state == "failed"
    assert not DataAsset.objects.filter(kind=paper_v2.EXECUTION_KIND).exists()
    assert not DataAsset.objects.filter(kind=paper_v2.MEASUREMENT_KIND).exists()


def test_equal_timestamp_candidate_ambiguity_and_under_ten_entry_withhold(environment, monkeypatch):
    owner, store, _run = environment
    activation = _activate(environment)
    day = _days(activation)[0]
    _capture_prices(activation, store, monkeypatch, day)
    _price_variant(activation, store, monkeypatch, day, same_clock=True)
    result = paper_jobs.execute_paper_capture_job(owner=owner, session=day, store=store)
    assert result.state == "failed"
    assert not DataAsset.objects.filter(kind=paper_v2.EXECUTION_KIND).exists()
    # A newer eligible source with a genuine cheap entry is not removed or
    # reweighted; the entire selected basket remains withheld.
    _price_variant(activation, store, monkeypatch, day, fault=_rewrite_close(day, 0.01))
    result = paper_jobs.execute_paper_capture_job(owner=owner, session=day, store=store)
    assert result.state == "failed" and result.reason_code == "paper_v2_selection_invalid"
    assert not DataAsset.objects.filter(kind=paper_v2.EXECUTION_KIND).exists()


@pytest.mark.parametrize("kind", ("raw", "normalized", "paper"))
@pytest.mark.parametrize("fault", ("missing", "unreadable", "corrupt"))
def test_physical_corruption_between_requests_is_not_cached_or_missing(
    environment, monkeypatch, kind, fault
):
    owner, store, _run = environment
    activation = _activate(environment)
    result = _capture(environment, activation, monkeypatch, _days(activation)[0])
    assert paper_v2.read_current_method_paper(user=owner, store=store)["captured_count"] == 1
    mark = result.capture.document["marks"][0]
    asset = (
        result.capture.asset
        if kind == "paper"
        else DataAsset.objects.get(pk=mark["raw_asset" if kind == "raw" else "source_asset"]["id"])
    )
    path = store.resolve(asset.relative_path)
    if fault == "missing":
        path.unlink()
    elif fault == "unreadable":
        path.unlink()
        path.mkdir()  # actual IsADirectoryError, also when CI runs as root
    else:
        path.write_bytes(b"synthetic corrupt bytes")
    counts = (DataAsset.objects.count(), JobRun.objects.count())
    with pytest.raises((ValueError, OSError, KeyError, PriceFrameChecksumMismatchError)):
        paper_v2.read_current_method_paper(user=owner, store=store)
    retry = paper_jobs.execute_paper_capture_job(
        owner=owner, session=_days(activation)[0], store=store
    )
    assert retry.state == "failed"
    assert (DataAsset.objects.count(), JobRun.objects.count()) == counts


def _unpublished_activation_document(environment):
    owner, store, _run = environment
    declaration = paper_v2.read_forward_declaration(user=owner, store=store)
    recorded = timezone.now()
    revision = paper_v2.clean_git_revision(Path("."))
    return copy.deepcopy(
        {
            **paper_v2._header(str(owner.pk), recorded, revision),
            "policy": paper_v2.PAPER_POLICY,
            "protocol": asset_ref_for(declaration.asset).to_json(),
            "calendar": [day.isoformat() for day in paper_v2._sessions(recorded)],
            **paper_v2._native_selection(user=owner, store=store, require_current=True),
        }
    )


@pytest.mark.parametrize(
    "fault",
    (
        "extra",
        "policy_bool",
        "policy_string",
        "policy_float",
        "wrong_owner",
        "wrong_policy_hash",
        "wrong_revision",
        "wrong_listing",
        "wrong_provider",
        "wrong_method",
        "wrong_config",
        "incomplete_selection",
        "advisory_as_buy",
        "extra_nested",
    ),
)
def test_strict_activation_reader_rejects_plausible_checksum_forgeries(environment, fault):
    owner, store, _run = environment
    doc = _unpublished_activation_document(environment)
    if fault == "extra":
        doc["status"] = "available"
    elif fault.startswith("policy_"):
        doc["policy"]["basis_reference_closes"] = {
            "policy_bool": True,
            "policy_string": "757",
            "policy_float": 757.0,
        }[fault]
    elif fault == "wrong_owner":
        doc["owner_id"] = "999999"
    elif fault == "wrong_policy_hash":
        doc["policy_sha256"] = "b" * 64
    elif fault == "wrong_revision":
        doc["execution_revision"] = "b" * 40
    elif fault == "wrong_listing":
        doc["buys"][0]["listing_id"] = "00000000-0000-0000-0000-000000000001"
    elif fault == "wrong_provider":
        doc["evidence"][0]["source_assets"][0]["provider"] = "another_provider"
    elif fault == "wrong_method":
        doc["policy"]["source_method_version"] = "us-price-fhs-v1"
    elif fault == "wrong_config":
        doc["policy"]["product_config_hash"] = "b" * 64
    elif fault == "incomplete_selection":
        doc["evidence"].pop()
        doc["buys"] = []
    elif fault == "advisory_as_buy":
        doc["buys"][0]["prediction_id"] = doc["evidence"][-1]["prediction_id"]
    else:
        doc["evidence"][0]["source_assets"][0]["extra"] = "not a field"
    paper_v2._publish(
        kind=paper_v2.ACTIVATION_KIND,
        subject=paper_v2.activation_subject(str(owner.pk)),
        document=doc,
        store=store,
    )
    with pytest.raises((ValueError, KeyError, TypeError)):
        paper_v2.read_current_method_paper(user=owner, store=store)
    assert DataAsset.objects.filter(kind=paper_v2.ACTIVATION_KIND).count() == 1


def test_wrong_owner_revoked_rights_and_spy_security_guard(environment, django_user_model):
    owner, store, _run = environment
    activation = _activate(environment)
    stranger = django_user_model.objects.create_user(username="different-paper-owner")
    with pytest.raises(ValueError):
        paper_v2.read_current_method_paper(user=stranger, store=store)
    record = ProviderRecord.objects.get(provider="twelve_data")
    record.metadata["internal_display_rights_confirmed"] = False
    record.save(update_fields=["metadata"])
    with pytest.raises(ValueError):
        paper_v2.activate_current_method_paper(user=owner, store=store)
    record.metadata["internal_display_rights_confirmed"] = True
    record.save(update_fields=["metadata"])
    spy = Listing.objects.get(pk=activation.document["benchmark"]["listing_id"])
    spy.security.security_type = "common_stock"
    spy.security.save(update_fields=["security_type"])
    with pytest.raises(ValueError):
        paper_v2.read_current_method_paper(user=owner, store=store)


@pytest.mark.parametrize("kind", ("activation", "execution", "measurement"))
def test_wrong_provider_paper_registration_is_present_invalid_evidence(
    environment, monkeypatch, kind
):
    owner, store, _run = environment
    if kind == "activation":
        document = _unpublished_activation_document(environment)
        artifact_kind = paper_v2.ACTIVATION_KIND
        subject = paper_v2.activation_subject(str(owner.pk))
        period_start = paper_v2._day(document["calendar"][0])
        period_end = paper_v2._day(document["calendar"][-1])
    else:
        activation = _activate(environment)
        _capture(environment, activation, monkeypatch, _days(activation)[0])
        artifact_kind = {
            "execution": paper_v2.EXECUTION_KIND,
            "measurement": paper_v2.MEASUREMENT_KIND,
        }[kind]
        original = DataAsset.objects.get(kind=artifact_kind)
        document = json.loads(store.read_bytes(original.relative_path))
        subject = original.subject
        period_start, period_end = original.period_start, original.period_end
    encoded = paper_v2._bytes(document)
    digest = hashlib.sha256(encoded).hexdigest()
    recorded = paper_v2._timestamp(document["recorded_at"])
    wrong = register_asset(
        provider="untrusted_provider",
        kind=artifact_kind,
        subject=subject,
        stored=store.write_bytes(f"forged-paper/{digest}.json", encoded),
        retrieved_at=recorded,
        available_at=recorded,
        period_start=period_start,
        period_end=period_end,
        metadata={key: document[key] for key in paper_v2.HEADER_FIELDS},
    )
    assert store.read_bytes(wrong.relative_path) == encoded
    with pytest.raises(paper_v2.PaperV2Error) as failure:
        paper_v2.read_current_method_paper(user=owner, store=store)
    assert failure.value.reason_code == "paper_v2_integrity_invalid"


def _forged_entry(environment, activation, monkeypatch, *, kind, fault):
    """Adversarial setup, not a substitute for the production-writer E2E."""
    owner, store, _run = environment
    session = _days(activation)[0]
    _capture_prices(activation, store, monkeypatch, session)
    recorded = timezone.now()
    revision = paper_v2.clean_git_revision(Path("."))
    sources, known = paper_v2._source_context(activation, store)
    marks, histories = paper_v2._prove_marks(
        activation, sources=sources, known=known, session=session, cutoff=recorded, store=store
    )
    job = JobRun.objects.create(
        job_name=paper_v2.capture_job_name(paper_v2.capture_identity(activation)),
        region="us",
        target_date=session,
    )
    with transaction.atomic(durable=True):
        execution_doc = paper_v2._execution_document(
            activation,
            marks,
            histories,
            owner=str(owner.pk),
            recorded=recorded,
            revision=revision,
            job_id=str(job.pk),
        )
        if kind == "execution":
            fault(execution_doc)
        execution = paper_v2._publish(
            kind=paper_v2.EXECUTION_KIND,
            subject=str(activation.asset.pk),
            document=execution_doc,
            store=store,
        )
        captured_marks = copy.deepcopy(marks)
        for mark in captured_marks:
            mark["basis"]["recorded_overlap_sha256"] = hashlib.sha256(
                paper_v2._bytes([[session.isoformat(), mark["native_price"]]])
            ).hexdigest()
        measurement_doc = paper_v2._measurement_document(
            activation,
            execution,
            marks=captured_marks,
            session=session,
            owner=str(owner.pk),
            recorded=recorded,
            revision=revision,
            job_id=str(job.pk),
        )
        if kind == "measurement":
            fault(measurement_doc)
        capture = paper_v2._publish(
            kind=paper_v2.MEASUREMENT_KIND,
            subject=f"{activation.asset.pk}:{session.isoformat()}",
            document=measurement_doc,
            store=store,
        )
        JobRun.objects.filter(pk=job.pk).update(
            details={"paper_v2_origin": paper_v2._origin_binding(activation, execution, capture)},
            status="success",
            finished_at=timezone.now(),
        )
    return capture


@pytest.mark.parametrize("kind", ("execution", "measurement"))
@pytest.mark.parametrize(
    "case",
    (
        "extra",
        "fx_bool",
        "price_bool",
        "ref_extra",
        "basis_extra",
        "duplicate_mark",
        "origin_uuid",
        "future_publication",
    ),
)
def test_strict_execution_and_capture_nested_types_with_plausible_checksums(
    environment, monkeypatch, kind, case
):
    owner, store, _run = environment
    activation = _activate(environment)

    def fault(doc):
        if case == "extra":
            doc["status"] = "available"
        elif case == "fx_bool":
            doc["marks"][0]["applied_fx_rate"] = True
        elif case == "price_bool":
            doc["marks"][0]["native_price"] = True
        elif case == "ref_extra":
            doc["marks"][0]["raw_asset"]["extra"] = "not permitted"
        elif case == "basis_extra":
            doc["marks"][0]["basis"]["no_corporate_action"] = True
        elif case == "duplicate_mark":
            doc["marks"][1] = copy.deepcopy(doc["marks"][0])
        elif case == "origin_uuid":
            doc["origin_job_id"] = "00000000-0000-0000-0000-000000000001"
        else:
            doc["recorded_at"] = (timezone.now() + timedelta(days=1)).isoformat()

    _forged_entry(environment, activation, monkeypatch, kind=kind, fault=fault)
    before = (DataAsset.objects.count(), JobRun.objects.count(), _files(store))
    with pytest.raises((ValueError, TypeError, KeyError)):
        paper_v2.read_current_method_paper(user=owner, store=store)
    assert (DataAsset.objects.count(), JobRun.objects.count(), _files(store)) == before


@pytest.mark.parametrize(
    "case",
    (
        "config_bool",
        "receipt_extra",
        "quantity_bool",
        "roundoff_plug",
        "reference_missing",
        "reference_extra",
    ),
)
def test_native_receipts_and_frozen_reference_are_reproduced_not_hash_trusted(
    environment, monkeypatch, case
):
    owner, store, _run = environment
    activation = _activate(environment)

    def fault(doc):
        receipt = doc["native_executions"]["spy"]
        if case == "config_bool":
            receipt["config"]["cash_buffer_bps"] = False
            receipt["config_sha256"] = hashlib.sha256(
                paper_v2._bytes(receipt["config"])
            ).hexdigest()
        elif case == "receipt_extra":
            receipt["missing"] = []
        elif case == "quantity_bool":
            receipt["holdings"][0]["quantity"] = True
        elif case == "roundoff_plug":
            receipt["native_entry_roundoff"] = "1"
        elif case == "reference_missing":
            doc["reference"]["closes"][0]["closes"].pop()
        else:
            doc["reference"]["closes"][0]["extra"] = 1

    _forged_entry(environment, activation, monkeypatch, kind="execution", fault=fault)
    with pytest.raises((ValueError, KeyError, TypeError)):
        paper_v2.read_current_method_paper(user=owner, store=store)


@pytest.mark.parametrize("kind", ("activation", "execution", "measurement"))
def test_paper_registry_duplicate_ambiguity_is_integrity_failure(environment, monkeypatch, kind):
    owner, store, _run = environment
    activation = _activate(environment)
    result = _capture(environment, activation, monkeypatch, _days(activation)[0])
    state = paper_v2.read_state(user=owner, store=store)
    original = {
        "activation": activation,
        "execution": state.execution,
        "measurement": result.capture,
    }[kind]
    doc = {**original.document, "extra": "plausible checksum cannot disambiguate"}
    paper_v2._publish(
        kind=original.asset.kind, subject=original.asset.subject, document=doc, store=store
    )
    with pytest.raises(ValueError):
        paper_v2.read_current_method_paper(user=owner, store=store)


@pytest.mark.parametrize("value", (True, False, 0, -1, math.nan, math.inf, -math.inf, "20"))
def test_native_numeric_admission_never_defaults_or_clips(value):
    with pytest.raises(paper_v2.PaperV2Error):
        paper_v2._native(value, positive=True)


def test_independent_decimal_terminal_survives_float64_path_failure(environment, monkeypatch):
    owner, store, _run = environment
    activation = _activate(environment)
    entry = _capture(environment, activation, monkeypatch, _days(activation)[0])
    execution = paper_v2.read_state(user=owner, store=store).execution
    extreme = copy.deepcopy(entry.capture.document)
    for mark in extreme["marks"]:
        mark["native_price"] = 1e308
    capture = paper_v2.PaperArtifact(entry.capture.asset, extreme)
    value = paper_v2._valuation(execution, capture, empty=False)
    assert Decimal(value["account_value_usd"]).is_finite()
    assert Decimal(value["account_value_usd"]) > Decimal("1e308")
    assert (
        value["native_entry_roundoff_usd"]
        == paper_v2._valuation(execution, entry.capture, empty=False)["native_entry_roundoff_usd"]
    )
    with pytest.raises(paper_v2.PaperV2Error, match="native_insufficient"):
        paper_v2._path_risk(
            _days(activation)[:2],
            [paper_v2._valuation(execution, entry.capture, empty=False), value],
        )
    # The same frozen receipts also preserve cost-only loss and recovery that
    # remains below initial capital, rather than resetting the drawdown peak.
    below = copy.deepcopy(entry.capture.document)
    for mark in below["marks"]:
        mark["native_price"] *= 0.9999
    recovery = paper_v2._valuation(
        execution, paper_v2.PaperArtifact(entry.capture.asset, below), empty=False
    )
    risk = paper_v2._path_risk(
        _days(activation)[:3],
        [
            paper_v2._valuation(execution, entry.capture, empty=False),
            recovery,
            paper_v2._valuation(execution, entry.capture, empty=False),
        ],
    )
    assert risk["account"]["max_drawdown"] < float(Decimal(value["account_return"]))
    assert risk["account"]["max_drawdown"] < 0


@pytest.mark.parametrize("price", (1e20, 1e-300))
def test_native_unrepresentable_execution_does_not_create_dummy_holdings(environment, price):
    activation = _activate(environment)
    marks = [
        {"listing_id": listing_id, "native_price": price}
        for listing_id in paper_v2._references(activation)
    ]
    with pytest.raises(ValueError):
        paper_v2._receipt(activation, marks, spy=True)


def _run_probe(mode, workdir, source, *, timeout=180):
    root = Path(__file__).resolve().parents[1]
    workdir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode, str(workdir)],
        cwd=root,
        env={
            "DJANGO_SETTINGS_MODULE": "stanstock.settings.test",
            "PYTHONPATH": str(source) + os.pathsep + str(root / "tests"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
            "PAPER_PROBE_SOURCE": str(source),
        },
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


def test_full_v1_base_head_differential_and_retained_base_requery(tmp_path):
    root = Path(__file__).resolve().parents[1]
    available = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{BASE_SHA}^{{commit}}"],
        capture_output=True,
    )
    if available.returncode:
        pytest.skip(f"Frozen paper base {BASE_SHA} is absent; no differential proof was executed")
    archive = subprocess.run(
        ["git", "-C", str(root), "archive", "--format=tar", BASE_SHA],
        capture_output=True,
        check=True,
    )
    base_root = tmp_path / "base-tree"
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
        bundle.extractall(base_root, filter="data")
    base = _run_probe("v1", tmp_path / "base-run", base_root / "src")
    head = _run_probe("v1", tmp_path / "head-run", root / "src")
    assert base["surface"] == head["surface"], _first_difference(base["surface"], head["surface"])
    assert base["surface"]["gap"]["measurement"] is None
    assert base["surface"]["gap"]["status"] == "withheld"
    assert base["surface"]["successful"]["measurement"]["status"] == "available"
    assert base["surface"]["idempotent_activation"]
    assert base["surface"]["idempotent_measurement"]
    # No source allowlist or normalized/rounded payload comparison. BOTH full
    # transitive import manifests are retained separately from the exact output
    # comparison, including B's final-head code. A dependency change cannot be
    # hidden behind an archive of HEAD or a stripped summary comparison.
    for relative in (
        "stanstock/research/price_product.py",
        "stanstock/research/product_pipeline.py",
    ):
        assert relative in base["source_manifest"]
        assert (
            head["source_manifest"][relative]
            == hashlib.sha256((root / "src" / relative).read_bytes()).hexdigest()
        )
    for relative, digest in head["source_manifest"].items():
        assert digest == hashlib.sha256((root / "src" / relative).read_bytes()).hexdigest()
    retained = _run_probe("replay-v1", tmp_path / "base-run", root / "src")
    assert retained["retained"] == base["retained"]
    assert retained["assets_before"] == retained["assets_after"]
    assert retained["jobs_before"] == retained["jobs_after"]


def _import_manifest():
    source = Path(os.environ["PAPER_PROBE_SOURCE"]).resolve()
    manifest = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("stanstock.") and path and path.endswith(".py"):
            resolved = Path(path).resolve()
            assert resolved.is_relative_to(source), name
            manifest[str(resolved.relative_to(source))] = hashlib.sha256(
                resolved.read_bytes()
            ).hexdigest()
    return dict(sorted(manifest.items()))


def _first_difference(left, right, path=""):
    if type(left) is not type(right):
        return path, type(left).__name__, type(right).__name__
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path, sorted(left), sorted(right)
        for key in sorted(left):
            if left[key] != right[key]:
                return _first_difference(left[key], right[key], f"{path}/{key}")
    elif isinstance(left, list):
        if len(left) != len(right):
            return path, len(left), len(right)
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            if a != b:
                return _first_difference(a, b, f"{path}/{index}")
    elif isinstance(left, str):
        try:
            a, b = json.loads(left), json.loads(right)
        except (ValueError, TypeError):
            return path, left, right
        if isinstance(a, dict) and isinstance(b, dict):
            return _first_difference(a, b, f"{path}/JSON")
    return path, left, right


def _v1_probe(*, replay=False):
    global _probe_uuid_sequence
    from itertools import count
    from types import SimpleNamespace

    from django.conf import settings

    import stanstock.research.forward_scorecard as forward

    workdir = Path(sys.argv[2])
    patch = pytest.MonkeyPatch()
    patch.setattr(paper, "_get_git_revision", lambda: REVISION)
    patch.setattr(forward, "_git", lambda *_args: SimpleNamespace(stdout=REVISION))
    if not replay:
        call_command("migrate", verbosity=0, stdout=StringIO())
        _probe_uuid_sequence = count(1)
        owner, store, _run = _paper_environment(workdir, patch, get_user_model(), settings)
    else:
        settings.RESEARCH_PRODUCT_ENABLED = True
        settings.DEMO_MODE = False
        owner = get_user_model().objects.get(username="research-fixture-owner")
        store = AssetStore(settings.DATA_DIR)
        latest = DataAsset.objects.filter(kind=paper.MEASUREMENT_KIND).latest("available_at")
        patch.setattr(timezone, "now", lambda: latest.available_at)
    settings.OWNER_USERNAME = owner.username

    def cli(**options):
        output = StringIO()
        call_command("paper_portfolio", owner=owner.username, stdout=output, **options)
        return output.getvalue()

    def retained():
        before, jobs_before = DataAsset.objects.count(), JobRun.objects.count()
        report = paper.read_current_method_paper(user=owner, store=store)
        output = cli()
        assets = [
            {field.name: field.value_from_object(asset) for field in asset._meta.concrete_fields}
            for asset in DataAsset.objects.filter(
                kind__in=(paper.ACTIVATION_KIND, paper.EXECUTION_KIND, paper.MEASUREMENT_KIND)
            ).order_by("kind", "subject")
        ]
        return {
            "retained": {"report": report, "cli": output, "assets": assets},
            "assets_before": before,
            "assets_after": DataAsset.objects.count(),
            "jobs_before": jobs_before,
            "jobs_after": JobRun.objects.count(),
        }

    if replay:
        print(json.dumps({**retained(), "source_manifest": _import_manifest()}, default=str))
        patch.undo()
        return

    errors = []
    for operation in (
        lambda: paper.activate_current_method_paper(user=owner, store=store),
        lambda: cli(),
    ):
        try:
            operation()
        except (ValueError, CommandError) as exc:
            errors.append({"type": type(exc).__name__, "message": str(exc)})
        else:
            raise AssertionError("Missing declaration did not fail")
    declare_forward_protocol(user=owner, store=store)
    activation = paper.activate_current_method_paper(user=owner, store=store)
    preview = cli(preview=True)
    awaiting = cli()
    days = _days(activation)
    _capture_prices(activation, store, patch, days[0])
    first = paper.measure_current_method_paper(user=owner, store=store, session=days[0])
    entry_cli = cli(session=days[0])
    _capture_prices(activation, store, patch, days[1], growth=1.01)
    paper.measure_current_method_paper(user=owner, store=store, session=days[1])
    successful = paper.read_current_method_paper(user=owner, store=store)
    successful_cli = cli()
    patch.setattr(
        timezone,
        "now",
        lambda: get_calendar("XNYS").session_close(days[2]).to_pydatetime() + timedelta(hours=1),
    )
    try:
        paper.measure_current_method_paper(user=owner, store=store, session=days[2])
    except ValueError as exc:
        errors.append({"type": type(exc).__name__, "message": str(exc)})
    else:
        raise AssertionError("Missing prospective source did not fail")
    _capture_prices(activation, store, patch, days[3], growth=1.01)
    withheld = paper.measure_current_method_paper(user=owner, store=store, session=days[3])
    gap = paper.read_current_method_paper(user=owner, store=store)
    gap_cli = cli()
    retry_activation = paper.activate_current_method_paper(user=owner, store=store)
    retry_measurement = paper.measure_current_method_paper(user=owner, store=store, session=days[0])
    assets = []
    for asset in DataAsset.objects.filter(
        kind__in=(paper.ACTIVATION_KIND, paper.EXECUTION_KIND, paper.MEASUREMENT_KIND)
    ).order_by("kind", "subject"):
        assets.append(
            {
                "registered_fields": {
                    field.name: field.value_from_object(asset)
                    for field in asset._meta.concrete_fields
                },
                "bytes": store.read_bytes(asset.relative_path).decode(),
            }
        )
    result = {
        "surface": {
            "policy": paper.PAPER_POLICY,
            "policy_sha256": paper.PAPER_POLICY_SHA256,
            "assets": assets,
            "awaiting_cli": awaiting,
            "preview_cli": preview,
            "entry_cli": entry_cli,
            "successful_cli": successful_cli,
            "gap_cli": gap_cli,
            "successful": successful,
            "gap": gap,
            "withheld_measurement": withheld.document,
            "errors": errors,
            "idempotent_activation": retry_activation.asset.id == activation.asset.id,
            "idempotent_measurement": retry_measurement.asset.id == first.asset.id,
        },
        **retained(),
        "source_manifest": _import_manifest(),
    }
    print(json.dumps(result, default=str, allow_nan=False))
    patch.undo()


def _concurrency_probe():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from itertools import count

    from django.conf import settings
    from django.db import connections

    from stanstock.core import research_product_refresh
    from stanstock.core.jobs import target_job_lock
    from stanstock.research.forward_scorecard import PROTOCOL_EPOCH
    from test_paper_capture_jobs import _all_capture_prices

    call_command("migrate", verbosity=0, stdout=StringIO())
    patch = pytest.MonkeyPatch()
    env = _environment(Path(sys.argv[2]), patch, get_user_model(), settings, empty=True)
    owner, store, _run = env
    sequence = count(180)
    clock_lock = threading.Lock()
    base_clock = [NOW]
    before_commit = threading.Event()
    release_commit = threading.Event()
    observer_used = [False]

    def clock():
        with clock_lock:
            value = base_clock[0] + timedelta(seconds=next(sequence))
        if threading.current_thread().name == "paper-cli" and not observer_used[0]:
            if DataAsset.objects.filter(kind=paper_v2.MEASUREMENT_KIND).exists():
                job = JobRun.objects.filter(job_name__startswith="paper_v2:").first()
                if job is not None and "paper_v2_origin" not in job.details:
                    observer_used[0] = True
                    before_commit.set()
                    assert release_commit.wait(30)
        return value

    patch.setattr(timezone, "now", clock)
    attempts = []
    start = threading.Barrier(2)

    def activation_worker():
        connections.close_all()
        try:
            attempts.append(clock())
            start.wait(10)
            output = StringIO()
            call_command(
                "paper_portfolio",
                owner=owner.username,
                policy_version=paper_v2.PAPER_POLICY_VERSION,
                activate=True,
                stdout=output,
            )
            asset_id = json.loads(output.getvalue())["activation"]["id"]
            asset = DataAsset.objects.get(pk=asset_id)
            return asset_id, store.read_bytes(asset.relative_path)
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(activation_worker) for _ in range(2)]
        activations = [future.result(90) for future in futures]
    activation = paper_v2.read_activation(user=owner, store=store)
    day = _days(activation)[0]
    _all_capture_prices(activation, store, patch, day)
    base_clock[0] = timezone.now()
    patch.setattr(timezone, "now", clock)
    patch.setattr(research_product_refresh, "clean_git_revision", lambda _root: REVISION)

    def cli_worker():
        threading.current_thread().name = "paper-cli"
        connections.close_all()
        try:
            output = StringIO()
            call_command(
                "paper_portfolio",
                owner=owner.username,
                policy_version=paper_v2.PAPER_POLICY_VERSION,
                session=day,
                stdout=output,
            )
            return json.loads(output.getvalue())
        finally:
            connections.close_all()

    def refresh_worker():
        connections.close_all()
        try:
            return research_product_refresh.execute_scheduled_research_refresh(
                store=store, capture_paper=True, enforce_rate_limit=False
            ).paper_capture
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as pool:
        cli = pool.submit(cli_worker)
        assert before_commit.wait(60)
        refresh = pool.submit(refresh_worker)
        connections.close_all()
        neither = not DataAsset.objects.filter(
            kind__in=(paper_v2.EXECUTION_KIND, paper_v2.MEASUREMENT_KIND)
        ).exists() and all(
            "paper_v2_origin" not in row.details
            for row in JobRun.objects.filter(job_name__startswith="paper_v2:")
        )
        release_commit.set()
        cli_result, refresh_result = cli.result(90), refresh.result(90)
    state = paper_v2.read_state(user=owner, store=store)
    origin = paper_v2._verify_origin(activation, state.execution, state.captures[0])
    both = (
        DataAsset.objects.filter(pk=state.execution.asset.pk).exists()
        and DataAsset.objects.filter(pk=state.captures[0].asset.pk).exists()
        and bool(origin.details["paper_v2_origin"])
    )
    next_day = _days(activation)[1]
    _all_capture_prices(activation, store, patch, next_day)
    base_clock[0] = timezone.now()
    patch.setattr(timezone, "now", clock)
    identity = paper_v2.capture_identity(activation)
    target_key = f"{paper_v2.capture_job_name(identity)}:us:{next_day.isoformat()}"
    lock_path = (
        Path(settings.DATABASES["default"]["NAME"]).parent
        / ".stanstock-job-locks"
        / (hashlib.sha256(target_key.encode()).hexdigest() + ".lock")
    )

    def wait_worker():
        connections.close_all()
        try:
            return paper_jobs.execute_paper_capture_job(
                owner=get_user_model().objects.get(pk=owner.pk), session=next_day, store=store
            )
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with target_job_lock(
            job_name=paper_v2.activation_subject(str(owner.pk)),
            region="us",
            target_date=PROTOCOL_EPOCH,
        ):
            waiting = pool.submit(wait_worker)
            for _ in range(200):
                if lock_path.exists():
                    break
                time.sleep(0.05)
            assert lock_path.exists() and not waiting.done()
            base_clock[0] = (
                get_calendar("XNYS")
                .session_open(get_calendar("XNYS").next_session(next_day))
                .to_pydatetime()
            )
        late = waiting.result(60)
    _assert_no_generic_rows()
    print(
        json.dumps(
            {
                "activations": DataAsset.objects.filter(kind=paper_v2.ACTIVATION_KIND).count(),
                "executions": DataAsset.objects.filter(kind=paper_v2.EXECUTION_KIND).count(),
                "captures": DataAsset.objects.filter(kind=paper_v2.MEASUREMENT_KIND).count(),
                "origins": sum(
                    "paper_v2_origin" in row.details
                    for row in JobRun.objects.filter(job_name__startswith="paper_v2:")
                ),
                "distinct_clocks": len(set(attempts)) == 2,
                "same_activation": activations[0] == activations[1],
                "same_capture": (
                    refresh_result.state == "recovered"
                    and refresh_result.capture.asset.id == state.captures[0].asset.id
                    and cli_result["captured_count"] == 1
                ),
                "before_commit_neither": neither,
                "after_commit_both": both,
                "lock_wait_deadline_refused": (
                    late.state == "failed"
                    and late.reason_code == "paper_v2_deadline_missed"
                    and not DataAsset.objects.filter(
                        kind=paper_v2.MEASUREMENT_KIND, period_end=next_day
                    ).exists()
                ),
                "no_generic_rows": True,
            }
        )
    )
    patch.undo()


if __name__ == "__main__":
    if sys.argv[1] in {"v1", "replay-v1"}:
        _v1_probe(replay=sys.argv[1] == "replay-v1")
    else:
        _concurrency_probe()

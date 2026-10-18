from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from io import StringIO
from itertools import count
from pathlib import Path
from threading import Barrier, Lock

# The concurrency probe owns a disposable, fully migrated SQLite database.
# It never opens the repository/runtime database or disables immutability.
if __name__ == "__main__":
    os.environ["DJANGO_SETTINGS_MODULE"] = "stanstock.settings.test"
    from django.conf import settings as probe_settings

    probe_settings.DATABASES["default"]["NAME"] = str(Path(sys.argv[1]) / "paper.sqlite3")
    probe_settings.DATA_DIR = Path(sys.argv[1]) / "assets"
    import django

    django.setup()

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connections
from django.utils import timezone
from exchange_calendars import get_calendar

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.assets import AssetStore, register_asset
from stanstock.data.live_us import _persist_price_series
from stanstock.data.models import DataAsset, ProviderRecord
from stanstock.data.providers import twelve_data
from stanstock.data.research_product_jobs import execute_daily_research_job
from stanstock.portfolio.models import Portfolio, PortfolioHolding
from stanstock.research.forward_scorecard import declare_forward_protocol
from stanstock.research.models import AnalysisRun
from stanstock.research.price_product import SourceExecutionBinding
from stanstock.research.product_pipeline import issue_price_product_snapshot
from stanstock.simulation import paper
from stanstock.simulation.models import SimulationRun
from test_research_product_jobs import NOW, TARGET, _series, make_product_environment

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize(
    ("activated_at", "inception"),
    (
        (datetime(2026, 9, 14, 23, tzinfo=UTC), date(2026, 9, 15)),
        (datetime(2026, 9, 15, 1, tzinfo=UTC), date(2026, 9, 15)),
        (datetime(2026, 9, 14, 1, tzinfo=UTC), date(2026, 9, 14)),
        (datetime(2026, 11, 10, 2, tzinfo=UTC), date(2026, 11, 10)),
    ),
)
def test_paper_calendar_uses_exchange_local_activation_date(activated_at, inception):
    sessions = paper._sessions(activated_at)
    assert sessions[0] == inception
    assert len(sessions) == 127
    assert sessions[-1] == get_calendar("XNYS").session_offset(inception, 126).date()


def test_paper_calendar_rejects_naive_activation_time():
    with pytest.raises(ValueError, match="aware publication"):
        paper._sessions(datetime(2026, 9, 14, 21))


def _paper_environment(tmp_path, monkeypatch, django_user_model, settings, *, empty=False):
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = False
    owner, store, path, resolve, fetch = make_product_environment(
        tmp_path,
        monkeypatch,
        django_user_model,
    )
    settings.DATA_DIR = store.root
    _persist_price_series(store=store, series=_series("CHEAP"), listing=None)
    monkeypatch.setenv("STANSTOCK_CODE_REVISION", "a" * 40)
    monkeypatch.setattr(
        "stanstock.research.product_pipeline.clean_git_revision",
        lambda _p: "a" * 40,
    )
    if not empty:
        for symbol in ("AAPL", "MSFT"):
            series = _series(symbol)
            payload = json.loads(series.raw_bytes)
            for index, row in enumerate(payload["values"]):
                value = 20.0 * math.exp(0.0005 * index + 0.005 * math.sin(index / 11))
                row.update({name: str(value) for name in ("open", "high", "low", "close")})
            retrieved = NOW + timedelta(minutes=1)
            series = twelve_data.parse_daily_price_series(
                json.dumps(payload).encode(),
                symbol=symbol,
                retrieved_at=retrieved,
                source_url=series.source_url,
                end_date=TARGET,
            )
            original = _persist_price_series(store=store, series=series, listing=None)
            # Explicit synthetic compatibility proof, not a production-provider
            # waiver. Real Twelve Data ingestion still never asserts this flag.
            stored = store.write_bytes(
                f"synthetic-compatible/{symbol}.parquet",
                store.read_bytes(original.relative_path),
            )
            register_asset(
                provider=original.provider,
                kind=original.kind,
                subject=symbol,
                stored=stored,
                retrieved_at=retrieved + timedelta(seconds=1),
                available_at=retrieved + timedelta(seconds=1),
                period_start=original.period_start,
                period_end=original.period_end,
                metadata={**original.metadata, "volume_adjustment_compatible": True},
            )
    monkeypatch.setattr(timezone, "now", lambda: NOW + timedelta(minutes=2))
    job = execute_daily_research_job(
        target_date=TARGET,
        owner=owner,
        issued_on_time=True,
        store=store,
        core_config_path=path,
        enforce_rate_limit=False,
        derive_frequencies=False,
    )
    assert job.status == "success"
    run = AnalysisRun.objects.get(pk=job.details["analysis_run_id"])
    resolve.assert_not_called()
    fetch.assert_not_called()
    return owner, store, run


def _capture_prices(activation, store, monkeypatch, session, *, growth=1.0, conflict=False):
    now = get_calendar("XNYS").session_close(session).to_pydatetime() + timedelta(hours=1)
    monkeypatch.setattr(timezone, "now", lambda: now)
    for ref in paper._references(activation).values():
        source = DataAsset.objects.get(pk=ref["id"])
        raw = (
            DataAsset.objects.filter(
                provider=source.provider,
                kind="raw_price_history",
                subject=source.subject,
                period_end__lt=session,
                available_at__lte=now,
                retrieved_at__lte=now,
            )
            .order_by("-period_end", "-retrieved_at")
            .first()
        )
        assert raw is not None
        payload = json.loads(store.read_bytes(raw.relative_path))
        last = payload["values"][-1]
        final_close = float(last["close"])
        future = get_calendar("XNYS").sessions_in_range(
            get_calendar("XNYS").next_session(last["datetime"]),
            session,
        )
        if conflict:
            for row in payload["values"]:
                row.update(
                    {name: str(float(row[name]) / 2) for name in ("open", "high", "low", "close")}
                )
        for offset, day in enumerate(future, start=1):
            value = final_close * growth**offset
            payload["values"].append(
                {
                    "datetime": day.date().isoformat(),
                    **{name: str(value) for name in ("open", "high", "low", "close")},
                    "volume": "1000000",
                }
            )
        series = twelve_data.parse_daily_price_series(
            json.dumps(payload).encode(),
            symbol=source.subject,
            retrieved_at=now,
            source_url="https://example.invalid/synthetic-paper",
            end_date=session,
        )
        _persist_price_series(store=store, series=series, listing=None)


@pytest.mark.parametrize("empty", (False, True), ids=("all-native-buys", "valid-empty-cash"))
def test_native_activation_inception_measurement_reader_and_actual_command(
    tmp_path,
    monkeypatch,
    django_user_model,
    settings,
    empty,
):
    owner, store, run = _paper_environment(
        tmp_path,
        monkeypatch,
        django_user_model,
        settings,
        empty=empty,
    )
    assert hashlib.sha256(paper._bytes(paper.PAPER_POLICY)).hexdigest() == (
        "25a6a2b5f6e0b37ab2f55d0fced023620d31614433fa16d0e8e3ea545318c835"
    )
    with pytest.raises(ValueError, match="declaration"):
        paper.activate_current_method_paper(user=owner, store=store)
    declare_forward_protocol(user=owner, store=store)
    before = DataAsset.objects.count()
    preview = paper.activate_current_method_paper(user=owner, persist=False, store=store)
    assert preview.asset is None
    assert DataAsset.objects.count() == before
    activation = paper.activate_current_method_paper(user=owner, store=store)
    assert activation.document == preview.document
    assert len(activation.document["buys"]) == (0 if empty else 2)
    assert {row["symbol"] for row in activation.document["buys"]} == (
        set() if empty else {"AAPL", "MSFT"}
    )
    frozen = store.read_bytes(activation.asset.relative_path)
    monkeypatch.setattr(timezone, "now", lambda: NOW + timedelta(minutes=9))
    retry = paper.activate_current_method_paper(user=owner, store=store)
    assert retry.asset.id == activation.asset.id
    assert store.read_bytes(retry.asset.relative_path) == frozen
    reissue = issue_price_product_snapshot(
        universe_snapshot=run.universe_snapshot,
        decision_time=timezone.now(),
        target_date=TARGET,
        provider="twelve_data",
        benchmark_subject="SPY",
        source_execution=SourceExecutionBinding(mode="provider", evidence_grade="observed"),
        store=store,
        code_revision="a" * 40,
        issued_on_time=True,
    )[0].run
    selected_scopes = []
    real_projection = paper.iter_product_prediction_projection

    def record_projection(run_ids, **kwargs):
        selected_scopes.append(set(run_ids))
        return real_projection(run_ids, **kwargs)

    with monkeypatch.context() as proof:
        proof.setattr(paper, "iter_product_prediction_projection", record_projection)
        retry = paper.activate_current_method_paper(user=owner, store=store)
    assert selected_scopes == [{run.id}]
    assert reissue.id not in selected_scopes[0]
    assert retry.asset.id == activation.asset.id
    assert store.read_bytes(retry.asset.relative_path) == frozen
    calendar = [paper.date.fromisoformat(day) for day in activation.document["calendar"]]
    assert calendar[0] > NOW.date()
    assert len(calendar) == 127
    assert paper.read_current_method_paper(user=owner, store=store)["status"] == (
        "awaiting_inception"
    )
    _capture_prices(activation, store, monkeypatch, calendar[0])
    output = StringIO()
    call_command("paper_portfolio", owner=owner.username, session=calendar[0], stdout=output)
    report = json.loads(output.getvalue())
    assert report["status"] == "available"
    assert report["cash_sleeve_usd"] == (30000 if empty else 0)
    measurement = report["measurement"]
    assert measurement["portfolio_metrics"]["max_drawdown"] == pytest.approx(
        measurement["portfolio_metrics"]["cumulative_return"]
    )
    assert measurement["spy_only_baseline_metrics"]["cumulative_return"] == pytest.approx(
        1 / 1.0015 - 1
    )
    assert all(
        mark["native_currency"] == "USD"
        and mark["applied_fx_rate"] == 1
        and mark["native_price"] > 0
        for mark in measurement["marks"]
    )
    first_asset_id = report["latest_measurement"]["id"]
    call_command("paper_portfolio", owner=owner.username, session=calendar[0], stdout=StringIO())
    assert str(DataAsset.objects.get(kind=paper.MEASUREMENT_KIND).id) == first_asset_id
    execution = DataAsset.objects.get(kind=paper.EXECUTION_KIND)
    execution_bytes = store.read_bytes(execution.relative_path)
    _capture_prices(activation, store, monkeypatch, calendar[1], growth=1.1)
    second = paper.measure_current_method_paper(user=owner, session=calendar[1], store=store)
    assert second.document["status"] == "available"
    assert second.document["portfolio_metrics"]["start_date"] == calendar[0].isoformat()
    assert store.read_bytes(execution.relative_path) == execution_bytes
    assert paper.read_current_method_paper(user=owner, store=store)["measurement"] == (
        second.document
    )
    assert (
        SimulationRun.objects.count()
        == Portfolio.objects.count()
        == PortfolioHolding.objects.count()
        == 0
    )
    assert DataAsset.objects.filter(kind=paper.ACTIVATION_KIND).count() == 1
    assert DataAsset.objects.filter(kind=paper.EXECUTION_KIND).count() == 1
    assert DataAsset.objects.filter(kind=paper.MEASUREMENT_KIND).count() == 2


def test_unreadable_prospective_mark_is_withheld_without_publishing_execution(
    tmp_path,
    monkeypatch,
    django_user_model,
    settings,
):
    owner, store, _run = _paper_environment(
        tmp_path,
        monkeypatch,
        django_user_model,
        settings,
    )
    declare_forward_protocol(user=owner, store=store)
    activation = paper.activate_current_method_paper(user=owner, store=store)
    inception = paper.date.fromisoformat(activation.document["calendar"][0])
    _capture_prices(activation, store, monkeypatch, inception)
    source = DataAsset.objects.filter(kind="price_history", period_end=inception).first()
    assert source is not None
    real_read = AssetStore.read_bytes

    def unreadable(store, path):
        if path == source.relative_path:
            raise PermissionError("synthetic prospective mark is unreadable")
        return real_read(store, path)

    monkeypatch.setattr(AssetStore, "read_bytes", unreadable)
    with pytest.raises(CommandError, match="Paper policy withheld"):
        call_command(
            "paper_portfolio",
            owner=owner.username,
            session=inception,
            stdout=StringIO(),
        )
    assert not DataAsset.objects.filter(kind=paper.EXECUTION_KIND).exists()
    assert not DataAsset.objects.filter(kind=paper.MEASUREMENT_KIND).exists()
    assert DataAsset.objects.filter(kind=paper.ACTIVATION_KIND).count() == 1


def test_sunday_evening_activation_round_trips_and_executes_monday(
    tmp_path, monkeypatch, django_user_model, settings
):
    owner, store, _run = _paper_environment(
        tmp_path, monkeypatch, django_user_model, settings, empty=True
    )
    declare_forward_protocol(user=owner, store=store)
    monkeypatch.setattr(timezone, "now", lambda: datetime(2026, 9, 14, 1, tzinfo=UTC))
    activation = paper.activate_current_method_paper(user=owner, store=store)
    assert activation.document["calendar"][0] == "2026-09-14"
    assert paper.read_current_method_paper(user=owner, store=store)["status"] == (
        "awaiting_inception"
    )
    _capture_prices(activation, store, monkeypatch, date(2026, 9, 14))
    measurement = paper.measure_current_method_paper(
        user=owner, session=date(2026, 9, 14), store=store
    )
    assert measurement.document["status"] == "available"
    report = paper.read_current_method_paper(user=owner, store=store)
    assert report["status"] == "available"
    assert report["activation"]["id"] == str(activation.asset.id)
    assert report["measurement"] == measurement.document


def test_missing_or_late_marks_never_become_observed_backfill(
    tmp_path,
    monkeypatch,
    django_user_model,
    settings,
):
    owner, store, _run = _paper_environment(
        tmp_path,
        monkeypatch,
        django_user_model,
        settings,
        empty=True,
    )
    declare_forward_protocol(user=owner, store=store)
    activation = paper.activate_current_method_paper(user=owner, store=store)
    days = [paper.date.fromisoformat(day) for day in activation.document["calendar"]]
    _capture_prices(activation, store, monkeypatch, days[0])
    paper.measure_current_method_paper(user=owner, session=days[0], store=store)
    monkeypatch.setattr(
        timezone,
        "now",
        lambda: get_calendar("XNYS").session_close(days[1]).to_pydatetime() + timedelta(hours=1),
    )
    with pytest.raises(ValueError, match="missing"):
        paper.measure_current_method_paper(user=owner, session=days[1], store=store)
    _capture_prices(activation, store, monkeypatch, days[2], growth=1.01)
    with pytest.raises(ValueError, match="prospective"):
        paper.measure_current_method_paper(user=owner, session=days[1], store=store)
    result = paper.measure_current_method_paper(user=owner, session=days[2], store=store)
    assert result.document["status"] == "withheld"
    assert result.document["missing_sessions"] == [days[1].isoformat()]
    assert result.document["portfolio_value"] is result.document["portfolio_metrics"] is None
    report = paper.read_current_method_paper(user=owner, store=store)
    assert report["status"] == "withheld"
    assert report["measurement"] is None


def test_share_basis_conflict_and_revoked_rights_and_corrupt_winner_fail_closed(
    tmp_path,
    monkeypatch,
    django_user_model,
    settings,
):
    owner, store, _run = _paper_environment(
        tmp_path,
        monkeypatch,
        django_user_model,
        settings,
        empty=True,
    )
    declare_forward_protocol(user=owner, store=store)
    activation = paper.activate_current_method_paper(user=owner, store=store)
    day = paper.date.fromisoformat(activation.document["calendar"][0])
    _capture_prices(activation, store, monkeypatch, day, conflict=True)
    with pytest.raises(ValueError, match="share-basis"):
        paper.measure_current_method_paper(user=owner, session=day, store=store)
    assert not DataAsset.objects.filter(kind=paper.EXECUTION_KIND).exists()
    record = ProviderRecord.objects.get(provider="twelve_data")
    record.metadata["internal_display_rights_confirmed"] = False
    record.save(update_fields=["metadata"])
    with pytest.raises(CommandError, match="withheld"):
        call_command("paper_portfolio", owner=owner.username, activate=True, stdout=StringIO())
    record.metadata["internal_display_rights_confirmed"] = True
    record.save(update_fields=["metadata"])
    store.resolve(activation.asset.relative_path).write_bytes(b"corrupt")
    with pytest.raises(RefreshVerificationError):
        paper.activate_current_method_paper(user=owner, store=store)
    assert DataAsset.objects.filter(kind=paper.ACTIVATION_KIND).count() == 1


def test_one_durable_protocol_and_paper_activation_winner_at_distinct_clocks(tmp_path):
    root = Path(__file__).resolve().parents[1]
    probe = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=root,
        env={
            "DJANGO_SETTINGS_MODULE": "stanstock.settings.test",
            "PYTHONPATH": str(root / "src") + ":" + str(root / "tests"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "UV_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert json.loads(probe.stdout) == {
        "protocols": 1,
        "activations": 1,
        "distinct_attempt_clocks": True,
        "same_protocol_uuid_bytes": True,
        "same_activation_uuid_bytes": True,
    }


def _concurrency_probe():
    from django.conf import settings

    call_command("migrate", verbosity=0, stdout=StringIO())
    patch = pytest.MonkeyPatch()
    owner, store, _run = _paper_environment(
        Path(sys.argv[1]),
        patch,
        get_user_model(),
        settings,
        empty=True,
    )
    ticker = count(180)
    clock_lock = Lock()

    def clock():
        with clock_lock:
            return NOW + timedelta(seconds=next(ticker))

    patch.setattr(timezone, "now", clock)
    attempts = []

    def concurrent_publish(publish):
        start = Barrier(2)

        def run():
            connections.close_all()
            try:
                attempts.append(clock())
                start.wait(timeout=10)
                result = publish(get_user_model().objects.get(pk=owner.pk))
                asset = result.asset
                return str(asset.id), store.read_bytes(asset.relative_path)
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run) for _ in range(2)]
            return [future.result(timeout=60) for future in futures]

    protocols = concurrent_publish(lambda user: declare_forward_protocol(user=user, store=store))
    activations = concurrent_publish(
        lambda user: paper.activate_current_method_paper(user=user, store=store),
    )
    print(
        json.dumps(
            {
                "protocols": DataAsset.objects.filter(kind="forward_scorecard_protocol").count(),
                "activations": DataAsset.objects.filter(kind=paper.ACTIVATION_KIND).count(),
                "distinct_attempt_clocks": len(set(attempts)) == 4,
                "same_protocol_uuid_bytes": protocols[0] == protocols[1],
                "same_activation_uuid_bytes": activations[0] == activations[1],
            }
        )
    )
    patch.undo()


if __name__ == "__main__":
    _concurrency_probe()

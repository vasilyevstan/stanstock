"""Native source-to-screen work bounds; generated fixtures, no provider access."""

from __future__ import annotations

import json
import time
from collections import Counter
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from unittest.mock import Mock

import pytest
from django.db import connection
from django.urls import reverse
from django.utils import timezone
from exchange_calendars import get_calendar

from stanstock.core.verification_types import RefreshVerificationError
from stanstock.data.assets import AssetStore, asset_ref_for, register_asset
from stanstock.data.live_us import _persist_catalog, _persist_price_series, load_us_universe_config
from stanstock.data.management.config_loader import default_us_universe_config_path
from stanstock.data.models import Company, DataAsset, Listing, ProviderRecord, Security
from stanstock.data.provider_policy import PRIVATE_USAGE_SCOPE
from stanstock.data.providers import twelve_data
from stanstock.data.providers.contracts import StockCatalog
from stanstock.data.research_product import (
    capture_authorized_product_intake,
    materialize_product_membership,
)
from stanstock.portfolio.models import TrackedSymbol
from stanstock.research import (
    price_product,
    product_frequency_evidence,
    product_pipeline,
    product_reader,
)
from stanstock.research.models import Prediction
from stanstock.research.price_product import (
    SourceExecutionBinding,
    calculate_price_product,
    complete_input_hash,
    deterministic_seed,
)
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    PRODUCT_EFFECTIVE_CONFIG_HASH,
)
from stanstock.web import product_views

pytestmark = pytest.mark.django_db
TARGET = date(2026, 9, 11)
NOW = datetime(2026, 9, 12, 1, tzinfo=UTC)


def build_native_workload(
    tmp_path,
    monkeypatch,
    django_user_model,
    settings,
    *,
    cohorts=2,
    analyzed=2,
    observed=False,
    retained_closes=757,
    with_frequencies=False,
):
    """109 requests, explicit exclusions, full registered native product rows.

    Research-grade current-vintage reconstruction is intentional: each source
    bundle ends at its own historical target. It never claims on-time observed
    issuance. Only fixture SETUP
    substitutes the expensive path simulation, with a complete per-input hash.
    """
    settings.RESEARCH_PRODUCT_ENABLED = True
    settings.DEMO_MODE = False
    settings.DATA_DIR = tmp_path
    monkeypatch.setattr(timezone, "now", lambda: NOW)
    if observed:
        assert cohorts == 1  # never backdate generated/retrieved clocks for history
        monkeypatch.setenv("STANSTOCK_CODE_REVISION", "a" * 40)
        monkeypatch.setattr(product_pipeline, "clean_git_revision", lambda _path: "a" * 40)
    grade = "observed" if observed else "research"
    blocked = Mock(side_effect=AssertionError("offline synthetic workload only"))
    monkeypatch.setattr("httpx.Client.send", blocked)
    monkeypatch.setattr(twelve_data, "resolve_api_key", blocked)
    monkeypatch.setattr(twelve_data, "fetch_daily_price_series", blocked)
    monkeypatch.setattr(twelve_data, "fetch_stock_catalog", blocked)
    owner = django_user_model.objects.create_user(username="workload-owner")
    ProviderRecord.objects.create(
        provider="twelve_data",
        enabled=True,
        terms_url="https://example.invalid/terms",
        usage_scope=PRIVATE_USAGE_SCOPE,
        metadata={"plan": "grow", "internal_display_rights_confirmed": True},
    )
    store = AssetStore(tmp_path)
    core_path = default_us_universe_config_path()
    core = load_us_universe_config(core_path)
    saved = tuple(f"SAVED{i}" for i in range(9))
    symbols = (*core.symbols, *saved)
    for symbol in saved:
        TrackedSymbol.objects.create(owner=owner, symbol=symbol)
    listings = []
    for symbol in symbols:
        company = Company.objects.create(name=f"Synthetic {symbol}", country="US")
        listings.append(
            Listing.objects.create(
                security=Security.objects.create(company=company),
                ticker=symbol,
                provider_symbol=symbol,
                exchange_mic="XNAS",
                currency="USD",
                region="us",
            )
        )
    catalog_assets = []
    for exchange in ("NASDAQ", "NYSE"):
        rows = [
            {
                "symbol": symbol,
                "name": f"Synthetic {symbol}",
                "currency": "USD",
                "exchange": exchange,
                "mic_code": "XNAS" if exchange == "NASDAQ" else "XNYS",
                "country": "United States",
                "type": "Common Stock" if index < analyzed else "ETF",
            }
            for index, symbol in enumerate(symbols if exchange == "NASDAQ" else ("VENUE",))
        ]
        raw = json.dumps({"status": "ok", "count": len(rows), "data": rows}).encode()
        refs, count = twelve_data.parse_stock_catalog_references(
            raw, exchange=exchange, require_complete=True
        )
        _persist_catalog(
            store,
            StockCatalog(
                provider="twelve_data",
                exchange=exchange,
                references=refs,
                count=count,
                retrieved_at=NOW,
                source_url="https://example.invalid/catalog",
                raw_bytes=raw,
            ),
        )
        catalog_assets.append(DataAsset.objects.get(kind="stock_catalog", subject=exchange))
    calendar = get_calendar("XNYS")
    targets = [
        item.date()
        for item in calendar.sessions_window(calendar.date_to_session(TARGET), -(cohorts + 1))
    ][-cohorts:]
    sessions = [
        item.date()
        for item in calendar.sessions_window(
            calendar.date_to_session(TARGET), -(retained_closes + cohorts)
        )
    ]

    def persist_prices(target):
        prices = {}
        for index, symbol in enumerate((*symbols[:analyzed], "SPY")):
            # One real degenerate filter case is retained, not a null DTO.
            flat = index == analyzed - 1
            values = []
            for offset, session in enumerate(day for day in sessions if day <= target):
                close = 64.0 if flat else 64.0 + offset / 64 + (offset % 7) / 8
                values.append(
                    {
                        "datetime": session.isoformat(),
                        "open": str(close),
                        "high": str(close),
                        "low": str(close),
                        "close": str(close),
                        "volume": "1000000",
                    }
                )
            raw = json.dumps(
                {
                    "status": "ok",
                    "meta": {
                        "symbol": symbol,
                        "currency": "USD",
                        "interval": "1day",
                        "exchange": "NYSE ARCA" if symbol == "SPY" else "NASDAQ",
                        "mic_code": "ARCX" if symbol == "SPY" else "XNAS",
                        "type": "ETF" if symbol == "SPY" else "Common Stock",
                    },
                    "values": values,
                }
            ).encode()
            series = twelve_data.parse_daily_price_series(
                raw,
                symbol=symbol,
                retrieved_at=NOW,
                source_url="https://example.invalid/prices",
                end_date=target,
            )
            _persist_price_series(store=store, series=series, listing=None)
            prices[symbol] = DataAsset.objects.filter(kind="price_history", subject=symbol).latest(
                "period_end"
            )
        return prices

    template = None

    def setup_calculation(product_input, *, config, **_kwargs):
        nonlocal template
        if len(set(product_input.stock.closes)) == 1 or (
            with_frequencies and product_input.target_date == targets[-1]
        ):
            # An AVAILABLE child must replay genuine, per-seed projections.
            # Only older source-only cohorts use the setup template.
            return calculate_price_product(product_input, config=config)
        if template is None:
            template = calculate_price_product(product_input, config=config)
        return replace(
            template,
            input_hash=complete_input_hash(product_input),
            stock_asset_id=product_input.stock.identity.asset_id,
            benchmark_asset_id=product_input.benchmark.identity.asset_id,
            target_date=product_input.target_date,
            source_execution=product_input.source_execution,
            forecast=replace(
                template.forecast,
                seed=deterministic_seed(
                    method_version=FHS_METHOD_VERSION,
                    effective_config_hash=PRODUCT_EFFECTIVE_CONFIG_HASH,
                    listing_id=product_input.listing_id,
                    target_date=product_input.target_date,
                ),
            ),
        )

    runs = []
    with monkeypatch.context() as setup:
        setup.setattr(product_pipeline, "calculate_price_product", setup_calculation)
        for target in targets:
            prices = persist_prices(target)
            intake = capture_authorized_product_intake(
                target_date=target,
                evidence_grade=grade,
                issuance_key="workload",
                owner=owner,
                core_config_path=core_path,
                policy_identity=PRODUCT_EFFECTIVE_CONFIG_HASH,
                captured_at=NOW,
                store=store,
            )
            admissions = {
                listing.provider_symbol: {
                    "status": "admitted" if index < analyzed else "identity_rejected",
                    "reason": "" if index < analyzed else "unsupported_instrument_type",
                    "listing_id": str(listing.id),
                    "price_asset": (
                        asset_ref_for(prices[listing.provider_symbol]).to_json()
                        if index < analyzed
                        else None
                    ),
                    "bootstrap_attempted": False,
                }
                for index, listing in enumerate(listings)
            }
            snapshot = materialize_product_membership(
                intake=intake,
                qualified_listing_ids=[row.id for row in listings[:analyzed]],
                candidate_states={
                    row.id: "admitted" if index < analyzed else "identity_rejected"
                    for index, row in enumerate(listings)
                },
                captured_at=NOW,
                store=store,
                admissions=admissions,
                catalog_assets=tuple(catalog_assets),
                benchmark_asset=prices["SPY"],
            )
            issued = product_pipeline.issue_price_product_snapshot(
                universe_snapshot=snapshot,
                decision_time=NOW,
                target_date=target,
                provider="twelve_data",
                benchmark_subject="SPY",
                source_execution=SourceExecutionBinding(mode="provider", evidence_grade=grade),
                store=store,
                code_revision="a" * 40,
                issued_on_time=observed,
            )
            runs.append(issued[0].run)
    if with_frequencies:
        monkeypatch.setattr(timezone, "now", lambda: NOW + timedelta(minutes=5))
        product_frequency_evidence.register_product_frequencies(run=runs[-1], store=store)
    return owner, store, tuple(runs), blocked


@contextmanager
def measured_request(monkeypatch):
    with monkeypatch.context() as scoped:
        with _measured_request(scoped) as measurements:
            yield measurements


@contextmanager
def _measured_request(monkeypatch):
    timings = Counter()
    calls = Counter()
    phases = []

    def instrument(module, name, label):
        original = getattr(module, name)

        def measured(*args, **kwargs):
            started = time.perf_counter()
            calls[label] += 1
            if label == "verification":
                calls[f"{module.__name__.rsplit('.', 1)[-1]}_verification"] += 1
            if label == "raw_validation":
                calls["raw_rows"] += len(args[0])
            phases.append(label)
            try:
                return original(*args, **kwargs)
            finally:
                phases.pop()
                timings[label] += time.perf_counter() - started

        monkeypatch.setattr(module, name, measured)

    instrument(product_reader, "_select_authorized_candidates", "selection")
    instrument(product_reader, "verify_price_product_output", "verification")
    instrument(product_frequency_evidence, "verify_price_product_output", "verification")
    instrument(product_pipeline, "product_catalog_references", "catalog_proofs")
    instrument(product_pipeline, "_validate_price_asset_with_frame", "price_proofs")
    instrument(product_pipeline, "_required_sessions", "calendar_windows")
    instrument(price_product, "_input_series_payload", "hash_series")
    instrument(twelve_data, "_parse_bars", "raw_validation")
    instrument(product_frequency_evidence, "_validate_registered_asset", "child_proofs")
    instrument(product_reader, "_build_cards", "current_cards")
    instrument(product_reader, "read_registered_product_frequencies", "current_frequency")
    instrument(product_views, "_observed_performance_groups", "aggregation")
    instrument(product_views, "read_registered_price_product_study", "study_read")
    instrument(product_views, "render", "render")
    instrument(AssetStore, "read_bytes", "asset_reads")
    no_write = Mock(side_effect=AssertionError("GET must not write"))
    monkeypatch.setattr(AssetStore, "write_bytes", no_write)
    monkeypatch.setattr(AssetStore, "write_frame", no_write)
    monkeypatch.setattr(product_pipeline, "calculate_price_product", no_write)

    def queries(execute, sql, params, many, context):
        calls["queries"] += 1
        assert sql.lstrip().split()[0].upper() not in {"INSERT", "UPDATE", "DELETE"}
        if "aggregation" in phases:
            calls["aggregation_queries"] += 1
            assert "COUNT(" not in sql.upper()
        if (
            "research_product_calculation" in (params or ())
            and "data_dataasset" in sql
            and '"subject"' in sql
        ):
            calls["registry_enumerations"] += 1
        if 'AS "canonical"' in sql:
            calls["scalar_projections"] += 1
        return execute(sql, params, many, context)

    started = time.perf_counter()
    with connection.execute_wrapper(queries):
        yield timings, calls
    timings["total"] = time.perf_counter() - started


@pytest.mark.parametrize(
    ("cohorts", "analyzed", "retained_closes"),
    [(2, 2, 757), (3, 4, 757), (26, 105, 757), (26, 105, 1758)],
)
def test_native_performance_workload(
    tmp_path, monkeypatch, django_user_model, settings, client, cohorts, analyzed, retained_closes
):
    owner, store, runs, blocked = build_native_workload(
        tmp_path,
        monkeypatch,
        django_user_model,
        settings,
        cohorts=cohorts,
        analyzed=analyzed,
        retained_closes=retained_closes,
        with_frequencies=True,
    )
    assert Prediction.objects.filter(analysis__run_id__in=[run.id for run in runs]).count() == (
        cohorts * analyzed * 5
    )
    client.force_login(owner)
    for attempt in ("first", "repeat"):
        with measured_request(monkeypatch) as (timings, calls):
            response = client.get(reverse("performance"))
        assert response.status_code == 200
        assert response.context["history"].available
        assert len(response.context["history"].runs) == cohorts
        assert response.context["observed_scope"]["advisory"]["non_evaluable"] == cohorts * 4
        assert response.context["observed_scope"]["decision"]["not_eligible_for_observed"] == (
            cohorts * analyzed
        )
        assert calls["verification"] == calls["product_reader_verification"] == cohorts
        assert calls["product_frequency_evidence_verification"] == 0
        assert calls["registry_enumerations"] == cohorts
        assert calls["catalog_proofs"] == cohorts
        assert calls["price_proofs"] == calls["raw_validation"] == cohorts * (analyzed + 1)
        assert calls["calendar_windows"] == cohorts
        assert calls["hash_series"] == cohorts * (analyzed + 1)
        assert calls["raw_rows"] == (analyzed + 1) * sum(
            retained_closes + offset for offset in range(1, cohorts + 1)
        )
        assert calls["scalar_projections"] == 1
        assert calls["aggregation_queries"] == 1
        assert calls["current_cards"] == calls["current_frequency"] == calls["child_proofs"] == 1
        assert all(
            projection.frequency_status == "available"
            for card in response.context["product"].cards
            for projection in card.projections
        )
        assert (
            b"Registered terminal-path shares are available for this issuance." in response.content
        )
        assert len(response.context["product"].admissions) == 109
        assert response.context["product"].admitted_count == analyzed
        assert response.context["product"].rejected_count == 109 - analyzed
        print(
            f"\nPERFORMANCE {attempt=} {cohorts=} {analyzed=} {retained_closes=} "
            f"{dict(timings)} {dict(calls)}"
        )
    blocked.assert_not_called()


@pytest.mark.parametrize("fault", ["frequency_child", "source_price"])
def test_available_frequency_child_shares_only_exact_source_proof(
    tmp_path, monkeypatch, django_user_model, settings, client, fault
):
    owner, store, runs, blocked = build_native_workload(
        tmp_path,
        monkeypatch,
        django_user_model,
        settings,
        cohorts=2,
        analyzed=2,
        retained_closes=1758,
        with_frequencies=True,
    )
    child = DataAsset.objects.get(kind=product_frequency_evidence.FREQUENCY_EVIDENCE_KIND)
    child_read = Mock(wraps=product_frequency_evidence._validate_registered_asset)
    monkeypatch.setattr(product_frequency_evidence, "_validate_registered_asset", child_read)
    client.force_login(owner)
    with measured_request(monkeypatch) as (_timings, calls):
        first = client.get(reverse("performance"))
    assert first.status_code == 200 and first.context["history"].available
    assert len(first.context["history"].runs) == len(runs) == 2
    assert first.context["product"].cards[0].projections[0].frequency_status == "available"
    assert b"Registered terminal-path shares are available for this issuance." in first.content
    assert calls["verification"] == calls["registry_enumerations"] == 2
    assert child_read.call_count == 1

    if fault == "frequency_child":
        store.resolve(child.relative_path).write_bytes(b"corrupt registered child")
    else:
        price = DataAsset.objects.filter(
            kind="price_history",
            subject=first.context["product"].cards[0].listing.provider_symbol,
            period_end=TARGET,
        ).get()
        store.resolve(price.relative_path).write_bytes(b"corrupt current source")
    with measured_request(monkeypatch) as (_timings, second_calls):
        second = client.get(reverse("performance"))
    assert second.status_code == 200
    if fault == "frequency_child":
        assert second.context["history"].available
        assert (
            second.context["product"].cards[0].projections[0].frequency_status == "integrity_failed"
        )
        assert (
            b"Registered terminal-path shares are available for this issuance."
            not in second.content
        )
        assert child_read.call_count == 2
    else:
        assert second.context["history"].status == "integrity_failed"
        assert second.context["history"].verification_code == "asset_corrupt"
        assert second.context["history"].runs == ()
        assert second.context["decision_groups"] == []
    assert second_calls["verification"] == (2 if fault == "frequency_child" else 1)
    blocked.assert_not_called()


@pytest.mark.parametrize("fault", ["calculation", "price", "raw", "missing_price"])
def test_performance_scope_rechecks_older_bytes_between_requests(
    tmp_path, monkeypatch, django_user_model, settings, client, fault
):
    owner, store, runs, _blocked = build_native_workload(
        tmp_path, monkeypatch, django_user_model, settings, retained_closes=1758
    )
    client.force_login(owner)
    first = client.get(reverse("performance"))
    assert first.context["history"].available
    if fault == "calculation":
        old = DataAsset.objects.filter(
            kind="research_product_calculation", subject=str(runs[0].id)
        ).first()
    else:
        old = (
            DataAsset.objects.filter(
                kind="raw_price_history" if fault == "raw" else "price_history"
            )
            .order_by(
                "period_end",
                "subject",
                "id",
            )
            .first()
        )
    assert old is not None
    if fault == "missing_price":
        store.resolve(old.relative_path).unlink()
    else:
        store.resolve(old.relative_path).write_bytes(b"corrupt older evidence")
    second = client.get(reverse("performance"))
    assert second.context["history"].status == "integrity_failed"
    assert (
        second.context["history"].verification_code
        == {
            "calculation": "product_calculation_malformed",
            "price": "asset_corrupt",
            "raw": "asset_corrupt",
            "missing_price": "asset_unreadable",
        }[fault]
    )
    assert second.context["history"].runs == ()
    assert second.context["decision_groups"] == []


def test_native_observed_reissue_and_genuine_withholding_share_verified_render(
    tmp_path,
    monkeypatch,
    django_user_model,
    settings,
    client,
):
    owner, store, runs, blocked = build_native_workload(
        tmp_path,
        monkeypatch,
        django_user_model,
        settings,
        cohorts=1,
        observed=True,
    )
    monkeypatch.setattr(timezone, "now", lambda: NOW + timedelta(minutes=5))
    reissue = product_pipeline.issue_price_product_snapshot(
        universe_snapshot=runs[0].universe_snapshot,
        target_date=TARGET,
        decision_time=timezone.now(),
        provider="twelve_data",
        benchmark_subject="SPY",
        source_execution=SourceExecutionBinding(mode="provider", evidence_grade="observed"),
        store=store,
        code_revision="a" * 40,
        issued_on_time=True,
    )[0].run
    assert reissue.id != runs[0].id
    client.force_login(owner)
    with measured_request(monkeypatch) as (_timings, calls):
        response = client.get(reverse("performance"))
    assert response.status_code == 200
    assert len(response.context["history"].runs) == 2
    assert response.context["observed_scope"]["decision"]["eligible_canonical"] == 2
    assert response.context["observed_scope"]["decision"]["outcome_pending"] == 2
    assert response.context["observed_scope"]["advisory"]["eligible_canonical"] == 4
    assert response.context["observed_scope"]["advisory"]["non_evaluable"] == 8
    assert calls["verification"] == calls["registry_enumerations"] == 2
    assert calls["catalog_proofs"] == 2
    assert calls["price_proofs"] == 6  # same bytes/cutoff, different exact run identity
    assert calls["calendar_windows"] == 2
    assert calls["hash_series"] == 6
    assert calls["current_cards"] == calls["current_frequency"] == 1
    assert calls["aggregation_queries"] == 1
    blocked.assert_not_called()


def test_duplicate_same_bytes_calculation_registration_is_not_a_new_authority(
    tmp_path,
    monkeypatch,
    django_user_model,
    settings,
    client,
):
    owner, store, runs, _blocked = build_native_workload(
        tmp_path,
        monkeypatch,
        django_user_model,
        settings,
        cohorts=1,
    )
    original = DataAsset.objects.filter(
        kind="research_product_calculation",
        subject=str(runs[0].id),
    ).first()
    assert original is not None
    duplicate = register_asset(
        provider=original.provider,
        kind=original.kind,
        subject=original.subject,
        stored=store.write_bytes(
            "synthetic/alternate-calculation.json",
            store.read_bytes(
                original.relative_path,
            ),
        ),
        retrieved_at=original.retrieved_at,
        available_at=original.available_at,
        metadata=original.metadata,
    )
    assert duplicate.id != original.id and duplicate.sha256 == original.sha256
    client.force_login(owner)
    response = client.get(reverse("performance"))
    assert response.context["history"].status == "integrity_failed"
    assert response.context["decision_groups"] == []


def test_revoked_display_rights_suppress_the_entire_performance_projection(
    tmp_path,
    monkeypatch,
    django_user_model,
    settings,
    client,
):
    owner, _store, _runs, blocked = build_native_workload(
        tmp_path,
        monkeypatch,
        django_user_model,
        settings,
        cohorts=1,
    )
    record = ProviderRecord.objects.get(provider="twelve_data")
    record.metadata["internal_display_rights_confirmed"] = False
    record.save(update_fields=["metadata"])
    client.force_login(owner)
    response = client.get(reverse("performance"))
    assert response.context["history"].status == "unauthorized"
    assert response.context["history"].runs == ()
    assert response.context["advisory_groups"] == []
    blocked.assert_not_called()


def test_verification_session_separates_cutoff_target_and_request(
    tmp_path,
    monkeypatch,
    django_user_model,
    settings,
):
    _owner, store, runs, _blocked = build_native_workload(
        tmp_path,
        monkeypatch,
        django_user_model,
        settings,
        cohorts=1,
    )
    verifier = Mock(wraps=product_reader.verify_price_product_output)
    monkeypatch.setattr(product_reader, "verify_price_product_output", verifier)
    session = product_reader.ProductVerificationSession()
    session.verify(run=runs[0], store=store)
    session.verify(run=runs[0], store=store)
    assert verifier.call_count == 1
    changed = copy(runs[0])
    changed.data_cutoff += timedelta(seconds=1)
    with pytest.raises(ValueError, match="changed within the request"):
        session.verify(run=changed, store=store)
    changed = copy(runs[0])
    changed.target_date -= timedelta(days=1)
    with pytest.raises(ValueError, match="changed within the request"):
        session.verify(run=changed, store=store)
    product_reader.ProductVerificationSession().verify(run=runs[0], store=store)
    assert verifier.call_count == 2
    # A different session cannot borrow a previous cutoff's proof either.
    with pytest.raises(RefreshVerificationError):
        product_reader.ProductVerificationSession().verify(run=changed, store=store)
    assert verifier.call_count == 3
    changed = copy(runs[0])
    changed.config_hash = "0" * 64
    with pytest.raises(RefreshVerificationError) as invalid_config:
        product_reader.ProductVerificationSession().verify(run=changed, store=store)
    assert invalid_config.value.reason_code == "product_config_invalid"
    assert verifier.call_count == 4


def test_shared_registered_study_source_does_not_skip_the_child_proof(
    tmp_path,
    monkeypatch,
    django_user_model,
    settings,
    client,
):
    from stanstock.research import product_study_evidence
    from stanstock.research.price_product_study import study_price_product_run

    owner, store, runs, _blocked = build_native_workload(
        tmp_path,
        monkeypatch,
        django_user_model,
        settings,
        cohorts=1,
    )
    monkeypatch.setattr(timezone, "now", lambda: NOW + timedelta(minutes=10))
    report = study_price_product_run(
        run=runs[0],
        store=store,
        all_selected=True,
        report_generated_at=timezone.now(),
    )
    asset = product_study_evidence._register_price_product_study(report=report, store=store)
    child_read = Mock(wraps=product_study_evidence._read_registered_document)
    child_verify = Mock(wraps=product_study_evidence._verify_registered_asset)
    source_verify = Mock(wraps=product_reader.verify_price_product_output)
    monkeypatch.setattr(product_study_evidence, "_read_registered_document", child_read)
    monkeypatch.setattr(product_study_evidence, "_verify_registered_asset", child_verify)
    monkeypatch.setattr(product_reader, "verify_price_product_output", source_verify)
    no_replay = Mock(side_effect=AssertionError("GET must not numerically replay a study"))
    monkeypatch.setattr(product_pipeline, "calculate_price_product", no_replay)
    client.force_login(owner)
    response = client.get(reverse("performance"))
    assert response.context["registered_study"].status == "available"
    assert source_verify.call_count == child_read.call_count == child_verify.call_count == 1
    store.resolve(asset.relative_path).write_bytes(b"corrupt child proof")
    response = client.get(reverse("performance"))
    assert response.context["registered_study"].status == "integrity_failed"
    assert child_read.call_count == source_verify.call_count == 2
    no_replay.assert_not_called()

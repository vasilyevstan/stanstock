from __future__ import annotations

from dataclasses import asdict
from unittest.mock import Mock

import pytest

from stanstock.data.assets import read_checksummed_bytes
from stanstock.data.models import DataAsset
from stanstock.research.decision_support_eligibility import (
    default_decision_support_config_path,
    load_decision_support_config,
)
from stanstock.research.models import AnalysisRun, Prediction
from stanstock.research.price_product import calculate_price_product, complete_input_hash
from stanstock.research.price_product_frequencies import FREQUENCY_SCHEMA
from stanstock.research.product_eligibility_pipeline import (
    decision_support_run_assets,
    verify_decision_support_output,
)
from stanstock.research.product_frequency_evidence import (
    read_registered_product_frequencies,
    register_product_frequencies,
    verify_registered_product_frequencies,
)
from stanstock.research.service import analyze_snapshot

pytestmark = pytest.mark.django_db


def test_one_frozen_calculation_and_exact_one_v2_five_row_issuance(c2_snapshot, monkeypatch):
    env = c2_snapshot
    calculate = Mock(wraps=calculate_price_product)
    monkeypatch.setattr(
        "stanstock.research.product_eligibility_pipeline.calculate_price_product",
        calculate,
    )
    result = analyze_snapshot(
        universe_snapshot=env.snapshot,
        decision_time=env.now,
        target_date=env.target,
        issued_on_time=False,
        provider="twelve_data",
        benchmark_subject="SPY",
        config_path=default_decision_support_config_path(),
        store=env.store,
    )
    assert calculate.call_count == 2
    assert len(result) == 2
    assert AnalysisRun.objects.count() == 1
    assert Prediction.objects.count() == 10
    assert {row.run.config_version for row in result} == {"research-product-v2"}
    assert all(row.result.recommendation.suggestion == "buy" for row in result)
    for item in result:
        base = calculate_price_product(
            item.selection.product_input, config=load_decision_support_config().forecast_config
        )
        assert asdict(item.result.forecast) == asdict(base.forecast)
        assert item.result.momentum == base.momentum
        assert item.result.input_hash != complete_input_hash(item.selection.product_input)
        assert not item.selection.product_input.stock.volume_adjustment_compatible
        rows = list(
            Prediction.objects.filter(analysis__run=item.run, listing=item.analysis.listing)
        )
        assert len(rows) == 5
        assert sum(row.method_version == "us-relative-momentum-v2" for row in rows) == 1
        assert all(len(row.source_assets) == 4 for row in rows if row.evidence_role == "advisory")
        assert all(row.config_hash == env.config_hash and not row.issued_on_time for row in rows)
        assert all(
            row.confidence is row.overall_score is row.probability_positive is None for row in rows
        )
    run = AnalysisRun.objects.get()
    verify_decision_support_output(run=run, store=env.store, replay=True)
    assert DataAsset.objects.filter(kind="research_product_calculation").count() == 2


@pytest.mark.parametrize("c2_environment", [{"mismatch": True}], indirect=True)
def test_raw_decimal_lsb_mismatch_withholds_liquidity_without_changing_forecast(c2_snapshot):
    env = c2_snapshot
    results = analyze_snapshot(
        universe_snapshot=env.snapshot,
        decision_time=env.now,
        target_date=env.target,
        provider="twelve_data",
        benchmark_subject="SPY",
        config_path=default_decision_support_config_path(),
        store=env.store,
    )
    apple = next(item for item in results if item.analysis.listing.ticker == "AAPL")
    assert apple.result.recommendation.suggestion == "hold"
    assert apple.result.risk.average_dollar_turnover_20d is None
    assert "liquidity_close_mismatch" in apple.result.risk.insufficiency_reasons
    frozen = calculate_price_product(
        apple.selection.product_input, config=load_decision_support_config().forecast_config
    )
    assert apple.result.forecast == frozen.forecast
    verify_decision_support_output(run=apple.run, store=env.store, replay=True)


def test_v2_frequency_binding_retains_v1_numerical_schema_seed_and_split_closure(c2_snapshot):
    import json

    env = c2_snapshot
    result = analyze_snapshot(
        universe_snapshot=env.snapshot,
        decision_time=env.now,
        target_date=env.target,
        provider="twelve_data",
        benchmark_subject="SPY",
        config_path=default_decision_support_config_path(),
        store=env.store,
    )
    asset = register_product_frequencies(run=result[0].run, store=env.store)
    assert asset.schema_version == FREQUENCY_SCHEMA
    document = json.loads(read_checksummed_bytes(env.store, asset))
    assert document["contract"] == "research-product-frequency-evidence@2"
    assert document["product_version"] == env.version and document["config_hash"] == env.config_hash
    assert document["forecast_config_hash"] != env.config_hash
    assert len(document["rows"]) == 2
    for item in document["rows"]:
        for horizon in item["horizons"]:
            assert len(horizon["source_assets"]) == 4
            assert horizon["effective_config_hash"] == document["forecast_config_hash"]
            assert horizon["input_hash"] != horizon["eligibility_input_hash"]
    verify_registered_product_frequencies(run=result[0].run, store=env.store)
    assert (
        read_registered_product_frequencies(
            user=env.owner,
            run=result[0].run,
            decision_time=env.now,
            store=env.store,
        ).status
        == "available"
    )
    assert {
        row.id for row in decision_support_run_assets(run=result[0].run, store=env.store)
    } == set(DataAsset.objects.values_list("id", flat=True))


def test_manifest_failure_rolls_back_rows_and_cleans_only_new_output_paths_then_retry_succeeds(
    c2_snapshot, monkeypatch
):
    from stanstock.research import product_eligibility_pipeline as pipeline

    env = c2_snapshot
    before_assets = set(DataAsset.objects.values_list("id", flat=True))
    before_files = {
        path.relative_to(env.store.root) for path in env.store.root.rglob("*") if path.is_file()
    }
    original = pipeline.build_manifest_envelope
    monkeypatch.setattr(
        pipeline,
        "build_manifest_envelope",
        Mock(side_effect=ValueError("synthetic manifest failure")),
    )
    kwargs = dict(
        universe_snapshot=env.snapshot,
        decision_time=env.now,
        target_date=env.target,
        provider="twelve_data",
        benchmark_subject="SPY",
        config_path=default_decision_support_config_path(),
        store=env.store,
    )
    with pytest.raises(ValueError, match="synthetic manifest failure"):
        analyze_snapshot(**kwargs)
    assert not AnalysisRun.objects.exists() and not Prediction.objects.exists()
    assert set(DataAsset.objects.values_list("id", flat=True)) == before_assets
    assert {
        path.relative_to(env.store.root) for path in env.store.root.rglob("*") if path.is_file()
    } == before_files
    monkeypatch.setattr(pipeline, "build_manifest_envelope", original)
    result = analyze_snapshot(**kwargs)
    verify_decision_support_output(run=result[0].run, store=env.store, replay=True)


@pytest.mark.parametrize("c2_environment", [{"tiny": True}], indirect=True)
def test_genuine_near_zero_forecast_survives_decimal_db_roundtrip(c2_persisted):
    from decimal import Decimal

    env = c2_persisted
    forecasts = [item.result.forecast for item in env.results]
    assert all(item.insufficiency_reason is None for item in forecasts)
    assert any(
        value.is_signed() and value.is_zero()
        for forecast in forecasts
        for projection in forecast.projections
        for value in (
            projection.ledger_returns.lower,
            projection.ledger_returns.median,
            projection.ledger_returns.upper,
        )
    )
    rows = list(Prediction.objects.filter(evidence_role="advisory"))
    assert len(rows) == 8
    assert all(isinstance(row.bear_return, Decimal) and row.bear_return == 0 for row in rows)
    verify_decision_support_output(run=env.run, store=env.store, replay=True)

"""Migrated database proof; PostgreSQL races run on the existing CI service."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier

import pytest
from django.contrib.auth import get_user_model
from django.db import DatabaseError, connection, connections, transaction

from stanstock.core.models import JobRun
from stanstock.data.assets import read_checksummed_bytes
from stanstock.data.models import DataAsset, ProviderRecord
from stanstock.data.research_product_jobs import execute_daily_research_job
from stanstock.research.forward_scorecard import (
    FORWARD_PROTOCOL_KIND,
    declare_forward_protocol,
    read_forward_declaration,
)
from stanstock.research.models import AnalysisRun, Prediction, PredictionOutcome
from stanstock.research.product_eligibility_pipeline import verify_decision_support_output
from stanstock.research.product_frequency_evidence import (
    FREQUENCY_EVIDENCE_KIND,
    register_product_frequencies,
    verified_registered_frequency_asset,
)
from test_decision_support_eligibility_outcomes import _evaluate, _observed_maturity

pytestmark = pytest.mark.django_db


def _insert_without_model_or_orm_guards(row):
    """Use the backend's actual field adapters, but no model clean/save/bulk API."""
    fields = row._meta.local_concrete_fields
    columns = ", ".join(connection.ops.quote_name(field.column) for field in fields)
    parameters = [
        field.get_db_prep_save(getattr(row, field.attname), connection=connection)
        for field in fields
    ]
    placeholders = ", ".join(["%s"] * len(fields))
    table = connection.ops.quote_name(row._meta.db_table)
    with connection.cursor() as cursor:
        cursor.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", parameters)


def _copy_values(row):
    return {
        field.attname: getattr(row, field.attname)
        for field in row._meta.local_concrete_fields
        if not field.primary_key
    }


def _db_pk(row):
    return row._meta.pk.get_db_prep_value(row.pk, connection=connection)


def test_migrated_database_has_v2_parent_role_and_append_only_triggers():
    with connection.cursor() as cursor:
        if connection.vendor == "sqlite":
            cursor.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
            expected = {
                "research_prediction_prevent_update",
                "research_prediction_prevent_delete",
                "research_analysisrun_product_insert",
                "research_analysisrun_product_update",
                "research_stockanalysis_product_insert",
                "research_stockanalysis_product_update",
                "research_prediction_product_insert",
                "research_prediction_product_update",
                "research_predictionoutcome_role_insert",
                "research_predictionoutcome_role_update",
            }
        else:
            assert connection.vendor == "postgresql"
            cursor.execute(
                "SELECT trigger_name FROM information_schema.triggers "
                "WHERE event_object_schema = current_schema()"
            )
            expected = {
                "research_prediction_prevent_update",
                "research_prediction_prevent_delete",
                "research_analysisrun_product_guard",
                "research_stockanalysis_product_guard",
                "research_prediction_product_guard",
                "research_predictionoutcome_role_guard",
            }
        assert expected <= {row[0] for row in cursor.fetchall()}


def test_direct_sql_new_v2_parent_requires_exact_config_snapshot_target_and_grade(c2_persisted):
    original = c2_persisted.run
    values = _copy_values(original)
    for change in (
        {"config_hash": "f" * 64},
        {"target_date": original.target_date - timedelta(days=1)},
        {"issued_on_time": True},
    ):
        with pytest.raises(DatabaseError), transaction.atomic():
            _insert_without_model_or_orm_guards(AnalysisRun(**{**values, **change}))
    assert AnalysisRun.objects.count() == 1


def test_direct_sql_published_v2_parent_identity_and_prediction_chronology_are_guarded(
    c2_persisted,
):
    env = c2_persisted
    with (
        pytest.raises(DatabaseError),
        transaction.atomic(),
        connection.cursor() as cursor,
    ):
        cursor.execute(
            "UPDATE research_analysisrun SET generated_at = %s WHERE id = %s",
            [env.run.generated_at + timedelta(seconds=1), _db_pk(env.run)],
        )
    decisions = list(Prediction.objects.filter(evidence_role="decision").order_by("price_subject"))
    original, other = decisions
    values = _copy_values(original)
    for index, change in enumerate(
        (
            {"generated_at": original.generated_at + timedelta(seconds=1)},
            {"data_cutoff": original.data_cutoff - timedelta(seconds=1)},
            {"target_date": original.target_date - timedelta(days=1)},
            {"listing_id": other.listing_id, "price_subject": other.price_subject},
            {"method_version": "us-relative-momentum-v1"},
            {"config_hash": "f" * 64},
        )
    ):
        row = Prediction(
            **{
                **values,
                **change,
                "model_version": f"c2-direct-sql-invalid-parent-{index}",
            }
        )
        with pytest.raises(DatabaseError), transaction.atomic():
            _insert_without_model_or_orm_guards(row)
    assert Prediction.objects.count() == 10
    env.run.refresh_from_db()
    assert env.run.generated_at == env.now


def test_direct_sql_prediction_and_all_three_liquidity_kinds_remain_immutable_after_0010(
    c2_persisted,
):
    row = Prediction.objects.first()
    for sql in (
        "UPDATE research_prediction SET price_at_prediction = 23 WHERE id = %s",
        "DELETE FROM research_prediction WHERE id = %s",
    ):
        with (
            pytest.raises(DatabaseError, match="immutable"),
            transaction.atomic(),
            connection.cursor() as cursor,
        ):
            cursor.execute(sql, [_db_pk(row)])
    for kind in ("liquidity_price_raw", "liquidity_price_history", "liquidity_run_evidence"):
        asset = DataAsset.objects.filter(kind=kind).first()
        assert asset is not None
        for sql in (
            "UPDATE data_dataasset SET sha256 = 'forged' WHERE id = %s",
            "DELETE FROM data_dataasset WHERE id = %s",
        ):
            with (
                pytest.raises(DatabaseError, match="immutable"),
                transaction.atomic(),
                connection.cursor() as cursor,
            ):
                cursor.execute(sql, [_db_pk(asset)])
        asset.refresh_from_db()
        assert read_checksummed_bytes(c2_persisted.store, asset)
    assert Prediction.objects.count() == 10


def test_v2_declaration_refuses_an_ambient_transaction_before_any_publication(c2_environment):
    env = c2_environment
    assets = set(DataAsset.objects.values_list("pk", "sha256"))
    with transaction.atomic(), pytest.raises(RuntimeError, match="durable"):
        declare_forward_protocol(user=env.owner, store=env.store, product_version=env.version)
    assert set(DataAsset.objects.values_list("pk", "sha256")) == assets
    env.credentials.assert_not_called()
    env.fetch.assert_not_called()


@pytest.mark.parametrize(
    "c2_environment,role,invalid_success",
    [
        ({"grade": "observed"}, "decision", None),
        ({"grade": "observed", "price_scale": "0.05"}, "decision", True),
        ({"grade": "observed"}, "advisory", True),
    ],
    indirect=["c2_environment"],
)
def test_real_matured_v2_outcomes_enforce_role_on_direct_sql_insert_and_update(
    c2_snapshot, monkeypatch, role, invalid_success
):
    env = _observed_maturity(c2_snapshot, monkeypatch)
    rows = list(
        Prediction.objects.filter(analysis__run=env.run, evidence_role=role, horizon="6m").order_by(
            "price_subject"
        )
    )
    first = _evaluate(env, rows[0])
    assert first.status == "matured" and first.actual_return is not None
    assert first.success is not invalid_success
    invalid = PredictionOutcome(
        **{**_copy_values(first), "prediction_id": rows[1].pk, "success": invalid_success}
    )
    with pytest.raises(DatabaseError, match="evidence role"), transaction.atomic():
        _insert_without_model_or_orm_guards(invalid)
    assert PredictionOutcome.objects.count() == 1
    second = _evaluate(env, rows[1])
    assert second.status == "matured" and second.success is not invalid_success
    with (
        pytest.raises(DatabaseError, match="evidence role"),
        transaction.atomic(),
        connection.cursor() as cursor,
    ):
        cursor.execute(
            "UPDATE research_predictionoutcome SET success = %s WHERE prediction_id = %s",
            [invalid_success, _db_pk(second)],
        )
    second.refresh_from_db()
    assert second.success is not invalid_success
    assert PredictionOutcome.objects.count() == 2


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="Actual PostgreSQL independent-session acquisition/advisory-lock race",
)
@pytest.mark.django_db(transaction=True)
def test_postgresql_v2_native_race_publishes_one_parent_and_recovers_without_fetch(
    c2_environment, settings
):
    env = c2_environment
    start = Barrier(2)

    def issue(*, synchronize=True):
        connections.close_all()
        try:
            owner = get_user_model().objects.get(pk=env.owner.pk)
            if synchronize:
                start.wait(timeout=10)
            job = execute_daily_research_job(
                target_date=env.target,
                owner=owner,
                issuance_key="c2-postgresql-native-race",
                product_version=env.version,
                store=env.store,
                core_config_path=env.path,
                enforce_rate_limit=False,
                derive_frequencies=False,
            )
            return job.status, job.job_name
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(issue) for _ in range(2)]
        results = [future.result(timeout=60) for future in futures]
    assert {status for status, _name in results} == {"success", "skipped"}
    assert len({name for _status, name in results}) == 1
    assert JobRun.objects.filter(job_name=results[0][1], status=JobRun.Status.SUCCESS).count() == 1
    assert JobRun.objects.filter(job_name=results[0][1], status=JobRun.Status.SKIPPED).count() == 1
    assert AnalysisRun.objects.count() == 1 and Prediction.objects.count() == 10
    assert DataAsset.objects.filter(kind="liquidity_run_evidence").count() == 1
    assert env.fetch.call_count == 4 and env.credentials.call_count == 1
    verify_decision_support_output(run=AnalysisRun.objects.get(), store=env.store, replay=True)
    assets = set(DataAsset.objects.values_list("pk", "sha256"))
    credits = ProviderRecord.objects.get(pk=env.record.pk).metadata["credits_used_local"]
    assert credits == 4
    env.record.enabled = False
    env.record.save(update_fields=["enabled"])
    settings.RESEARCH_PRODUCT_ENABLED = False
    env.fetch.reset_mock()
    env.credentials.reset_mock()
    env.fetch.side_effect = AssertionError("PostgreSQL completed retry must not fetch")
    env.credentials.side_effect = AssertionError("PostgreSQL completed retry must not resolve keys")
    retry = issue(synchronize=False)
    assert retry[0] == "skipped"
    assert set(DataAsset.objects.values_list("pk", "sha256")) == assets
    assert ProviderRecord.objects.get(pk=env.record.pk).metadata["credits_used_local"] == credits
    env.fetch.assert_not_called()
    env.credentials.assert_not_called()


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="Actual PostgreSQL independent-session v2 frequency publication race",
)
@pytest.mark.django_db(transaction=True)
def test_postgresql_v2_frequency_publication_has_one_immutable_winner_and_no_fetch(c2_persisted):
    env = c2_persisted
    start = Barrier(2)
    env.fetch.reset_mock()
    env.credentials.reset_mock()
    env.fetch.side_effect = AssertionError("Derived publication must not fetch")
    env.credentials.side_effect = AssertionError("Derived publication must not resolve keys")

    def publish():
        connections.close_all()
        try:
            start.wait(timeout=10)
            run = AnalysisRun.objects.get(pk=env.run.pk)
            asset = register_product_frequencies(run=run, store=env.store)
            return asset.pk, read_checksummed_bytes(env.store, asset)
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish) for _ in range(2)]
        results = [future.result(timeout=60) for future in futures]
    assert results[0] == results[1]
    asset = DataAsset.objects.get(kind=FREQUENCY_EVIDENCE_KIND)
    assert asset.pk == results[0][0]
    assert asset.schema_version == "research-product-frequencies@1"
    document = json.loads(results[0][1])
    assert document["contract"] == "research-product-frequency-evidence@2"
    assert document["source_run"]["id"] == str(env.run.pk)
    assert verified_registered_frequency_asset(run=env.run, store=env.store) == asset
    retry = register_product_frequencies(run=env.run, store=env.store)
    assert retry.pk == asset.pk and read_checksummed_bytes(env.store, retry) == results[0][1]
    assert len(list(env.store.resolve(asset.relative_path).parent.glob("*.json"))) == 1
    env.fetch.assert_not_called()
    env.credentials.assert_not_called()


@pytest.mark.skipif(
    connection.vendor != "postgresql",
    reason="Actual PostgreSQL independent-session durable v2 declaration race",
)
@pytest.mark.django_db(transaction=True)
def test_postgresql_v2_declaration_has_one_durable_winner_without_acquisition(
    c2_environment, django_user_model
):
    env = c2_environment
    owner = django_user_model.objects.create_user(username="c2-postgresql-protocol-owner")
    start = Barrier(2)
    metadata = ProviderRecord.objects.get(pk=env.record.pk).metadata
    assert not AnalysisRun.objects.exists() and not Prediction.objects.exists()

    def declare():
        connections.close_all()
        try:
            user = get_user_model().objects.get(pk=owner.pk)
            start.wait(timeout=10)
            result = declare_forward_protocol(
                user=user, store=env.store, product_version=env.version
            )
            return result.asset.pk, read_checksummed_bytes(env.store, result.asset)
        finally:
            connections.close_all()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(declare) for _ in range(2)]
        results = [future.result(timeout=60) for future in futures]
    assert results[0] == results[1]
    assert (
        DataAsset.objects.filter(
            kind=FORWARD_PROTOCOL_KIND, metadata__owner_id=str(owner.pk)
        ).count()
        == 1
    )
    declaration = read_forward_declaration(user=owner, store=env.store, product_version=env.version)
    assert declaration.asset.pk == results[0][0]
    assert declaration.document["protocol"]["product_version"] == env.version
    assert declaration.document["source_intakes"] == []
    assert declaration.document["prospective_start_target"] > env.target.isoformat()
    retry = declare_forward_protocol(user=owner, store=env.store, product_version=env.version)
    assert retry.asset.pk == declaration.asset.pk
    assert read_checksummed_bytes(env.store, retry.asset) == results[0][1]
    assert ProviderRecord.objects.get(pk=env.record.pk).metadata == metadata
    assert not AnalysisRun.objects.exists() and not Prediction.objects.exists()
    env.credentials.assert_not_called()
    env.fetch.assert_not_called()

from __future__ import annotations

from importlib import import_module
from typing import Any

from django.db import migrations, models

_immutability = import_module(
    "stanstock.research.migrations.0005_reinstate_prediction_immutability"
)


def _scenario_document(analysis: Any) -> dict[str, object]:
    return {
        "schema_version": 1,
        "horizons": {
            "short": analysis.short_scenario,
            "medium": analysis.medium_scenario,
            "long": analysis.long_scenario,
        },
    }


def backfill_forecast_scenarios(apps: Any, schema_editor: Any) -> None:
    del schema_editor
    StockAnalysis = apps.get_model("research", "StockAnalysis")
    for analysis in StockAnalysis.objects.all().iterator(chunk_size=500):
        StockAnalysis.objects.filter(pk=analysis.pk).update(
            forecast_scenarios=_scenario_document(analysis),
        )


def _price_source(prediction: Any) -> tuple[str, str]:
    subjects = {
        prediction.listing.provider_symbol or prediction.listing.ticker,
        prediction.listing.ticker,
    }
    sources = {
        (str(asset["provider"]), str(asset["subject"]))
        for asset in prediction.source_assets
        if isinstance(asset, dict)
        and asset.get("kind") == "price_history"
        and str(asset.get("subject") or "") in subjects
        and isinstance(asset.get("provider"), str)
        and str(asset["provider"])
    }
    if len(sources) != 1:
        return "", ""
    return next(iter(sources))


def _legacy_scenario(prediction: Any) -> dict[str, object]:
    scenarios = {
        "short": prediction.analysis.short_scenario,
        "medium": prediction.analysis.medium_scenario,
        "long": prediction.analysis.long_scenario,
    }
    scenario = scenarios.get(prediction.horizon)
    return scenario if isinstance(scenario, dict) else {}


def _calculation_document(prediction: Any) -> dict[str, object]:
    scenario = _legacy_scenario(prediction)
    data_quality = prediction.analysis.data_quality
    snapshot = prediction.analysis.run.universe_snapshot
    return {
        "schema_version": 1,
        "method": scenario.get("method"),
        "method_version": prediction.analysis.run.config_version,
        "prediction_version": prediction.model_version,
        "config_hash": prediction.config_hash,
        "forecast_horizon": prediction.horizon,
        "score_group": prediction.horizon,
        "support": {
            "status": "legacy_recorded_values_only",
            "confidence": scenario.get("confidence"),
            "confidence_status": scenario.get("confidence_status"),
            "insufficiency_reason": scenario.get("insufficiency_reason"),
        },
        "formula_inputs": {
            "scenario": {
                "bear": scenario.get("bear"),
                "base": scenario.get("base"),
                "bull": scenario.get("bull"),
                "probability_positive": scenario.get("probability_positive"),
            },
            "overall_score": float(prediction.overall_score),
        },
        "contribution_detail": prediction.component_scores,
        "return_basis": (
            data_quality.get("return_definition")
            if isinstance(data_quality, dict)
            else None
        ),
        "dividends_included": (
            data_quality.get("dividends_included")
            if isinstance(data_quality, dict)
            else None
        ),
        "evidence_grade": snapshot.grade,
        "price_subject": _price_source(prediction)[1],
        "provenance": {
            "backfilled_from_legacy": True,
            "provenance_strengthened": False,
        },
    }


def _source_mode(prediction: Any) -> str:
    providers = {
        str(asset["provider"])
        for asset in prediction.source_assets
        if isinstance(asset, dict)
        and isinstance(asset.get("provider"), str)
        and str(asset["provider"])
    }
    if not providers:
        return "unknown"
    if any(provider.startswith("synthetic") for provider in providers):
        return "synthetic"
    return "provider"


def backfill_prediction_metadata(apps: Any, schema_editor: Any) -> None:
    del schema_editor
    Prediction = apps.get_model("research", "Prediction")
    queryset = Prediction.objects.select_related(
        "listing",
        "analysis__run__universe_snapshot",
    )
    for prediction in queryset.iterator(chunk_size=500):
        price_provider, price_subject = _price_source(prediction)
        Prediction.objects.filter(pk=prediction.pk).update(
            evidence_role="decision",
            evidence_grade=prediction.analysis.run.universe_snapshot.grade,
            source_mode=_source_mode(prediction),
            price_provider=price_provider,
            price_subject=price_subject,
            method_version=prediction.analysis.run.config_version,
            calculation=_calculation_document(prediction),
        )


def backfill_outcome_metrics(apps: Any, schema_editor: Any) -> None:
    del schema_editor
    PredictionOutcome = apps.get_model("research", "PredictionOutcome")
    queryset = PredictionOutcome.objects.select_related("prediction").filter(
        status="matured",
        actual_return__isnull=False,
    )
    for outcome in queryset.iterator(chunk_size=500):
        prediction = outcome.prediction
        direction_correct = None
        interval_covered = None
        signed_error = outcome.error
        if prediction.base_return is not None:
            direction_correct = (
                (outcome.actual_return > 0) - (outcome.actual_return < 0)
            ) == ((prediction.base_return > 0) - (prediction.base_return < 0))
        if prediction.bear_return is not None and prediction.bull_return is not None:
            interval_covered = (
                prediction.bear_return
                <= outcome.actual_return
                <= prediction.bull_return
            )
        PredictionOutcome.objects.filter(pk=outcome.pk).update(
            direction_correct=direction_correct,
            interval_covered=interval_covered,
            signed_error=signed_error,
        )


def clear_outcome_metrics(apps: Any, schema_editor: Any) -> None:
    del schema_editor
    PredictionOutcome = apps.get_model("research", "PredictionOutcome")
    PredictionOutcome.objects.filter(
        error__isnull=False,
    ).update(
        direction_correct=None,
        interval_covered=None,
        signed_error=None,
    )


def reject_unsafe_reverse(apps: Any, schema_editor: Any) -> None:
    del schema_editor
    Prediction = apps.get_model("research", "Prediction")
    if Prediction.objects.filter(
        models.Q(evidence_role="advisory")
        | models.Q(horizon__in=("6m", "12m", "3y", "5y"))
    ).exists():
        raise RuntimeError(
            "Cannot reverse explicit forecast horizons after advisory predictions exist"
        )
    StockAnalysis = apps.get_model("research", "StockAnalysis")
    for analysis in StockAnalysis.objects.only(
        "forecast_scenarios",
        "short_scenario",
        "medium_scenario",
        "long_scenario",
    ).iterator(chunk_size=500):
        document = analysis.forecast_scenarios
        if document == {}:
            continue
        if not isinstance(document, dict):
            raise RuntimeError("Cannot reverse an invalid forecast scenario document")
        horizons = document.get("horizons")
        if (
            document.get("schema_version") != 1
            or set(document) != {"schema_version", "horizons"}
            or not isinstance(horizons, dict)
        ):
            raise RuntimeError("Cannot reverse an invalid forecast scenario document")
        if set(horizons) - {"short", "medium", "long"}:
            raise RuntimeError(
                "Cannot reverse explicit forecast horizons after unrepresentable scenarios exist"
            )
        legacy = {
            "short": analysis.short_scenario,
            "medium": analysis.medium_scenario,
            "long": analysis.long_scenario,
        }
        if any(horizons.get(horizon) != value for horizon, value in legacy.items()):
            raise RuntimeError(
                "Cannot reverse after the unified scenario document diverges "
                "from legacy compatibility columns"
            )


def install_outcome_role_guards(apps: Any, schema_editor: Any) -> None:
    # SQLite table rebuilds discard triggers. Keep the trigger-presence regression
    # test in CI and reinstall these guards in every future outcome-table migration.
    del apps
    vendor = schema_editor.connection.vendor
    if vendor == "sqlite":
        schema_editor.execute(
            """
            CREATE TRIGGER research_predictionoutcome_role_insert
            BEFORE INSERT ON research_predictionoutcome
            FOR EACH ROW
            WHEN NEW.status = 'matured'
              AND (
                (
                  (SELECT evidence_role FROM research_prediction WHERE id = NEW.prediction_id)
                    = 'decision'
                  AND NEW.success IS NULL
                )
                OR
                (
                  (SELECT evidence_role FROM research_prediction WHERE id = NEW.prediction_id)
                    = 'advisory'
                  AND NEW.success IS NOT NULL
                )
              )
            BEGIN
              SELECT RAISE(
                ABORT,
                'matured outcome success does not match prediction evidence role'
              );
            END;
            """
        )
        schema_editor.execute(
            """
            CREATE TRIGGER research_predictionoutcome_role_update
            BEFORE UPDATE OF prediction_id, status, success ON research_predictionoutcome
            FOR EACH ROW
            WHEN NEW.status = 'matured'
              AND (
                (
                  (SELECT evidence_role FROM research_prediction WHERE id = NEW.prediction_id)
                    = 'decision'
                  AND NEW.success IS NULL
                )
                OR
                (
                  (SELECT evidence_role FROM research_prediction WHERE id = NEW.prediction_id)
                    = 'advisory'
                  AND NEW.success IS NOT NULL
                )
              )
            BEGIN
              SELECT RAISE(
                ABORT,
                'matured outcome success does not match prediction evidence role'
              );
            END;
            """
        )
        return
    if vendor == "postgresql":
        schema_editor.execute(
            """
            CREATE OR REPLACE FUNCTION stanstock_validate_prediction_outcome_role()
            RETURNS trigger AS $$
            DECLARE
              prediction_role text;
            BEGIN
              SELECT evidence_role
                INTO prediction_role
                FROM research_prediction
                WHERE id = NEW.prediction_id;
              IF NEW.status = 'matured'
                 AND (
                   (prediction_role = 'decision' AND NEW.success IS NULL)
                   OR (prediction_role = 'advisory' AND NEW.success IS NOT NULL)
                 )
              THEN
                RAISE EXCEPTION
                  'matured outcome success does not match prediction evidence role'
                  USING ERRCODE = '23514';
              END IF;
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
        schema_editor.execute(
            """
            CREATE TRIGGER research_predictionoutcome_role_guard
            BEFORE INSERT OR UPDATE OF prediction_id, status, success
            ON research_predictionoutcome
            FOR EACH ROW
            EXECUTE FUNCTION stanstock_validate_prediction_outcome_role();
            """
        )
        return
    raise RuntimeError(f"Unsupported database vendor for outcome role guard: {vendor}")


def remove_outcome_role_guards(apps: Any, schema_editor: Any) -> None:
    del apps
    vendor = schema_editor.connection.vendor
    if vendor == "sqlite":
        schema_editor.execute("DROP TRIGGER IF EXISTS research_predictionoutcome_role_insert")
        schema_editor.execute("DROP TRIGGER IF EXISTS research_predictionoutcome_role_update")
        return
    if vendor == "postgresql":
        schema_editor.execute(
            "DROP TRIGGER IF EXISTS research_predictionoutcome_role_guard "
            "ON research_predictionoutcome"
        )
        schema_editor.execute(
            "DROP FUNCTION IF EXISTS stanstock_validate_prediction_outcome_role()"
        )
        return
    raise RuntimeError(f"Unsupported database vendor for outcome role guard: {vendor}")


class Migration(migrations.Migration):
    dependencies = [
        ("research", "0007_analysis_run_issued_on_time"),
    ]

    operations = [
        migrations.AddField(
            model_name="stockanalysis",
            name="forecast_scenarios",
            field=models.JSONField(default=dict),
        ),
        migrations.RunPython(
            backfill_forecast_scenarios,
            migrations.RunPython.noop,
        ),
        migrations.RunPython(
            _immutability.unprotect_predictions,
            _immutability.protect_predictions,
        ),
        migrations.AlterField(
            model_name="prediction",
            name="horizon",
            field=models.CharField(
                choices=[
                    ("short", "1-10 trading days"),
                    ("6m", "6 months"),
                    ("12m", "12 months"),
                    ("3y", "3 years"),
                    ("5y", "5 years"),
                    ("medium", "Legacy 6-12 months"),
                    ("long", "Legacy 3+ years"),
                ],
                max_length=8,
            ),
        ),
        migrations.AddField(
            model_name="prediction",
            name="evidence_role",
            field=models.CharField(
                choices=[
                    ("decision", "Decision"),
                    ("advisory", "Advisory"),
                ],
                db_index=True,
                default="decision",
                max_length=12,
            ),
        ),
        migrations.AddField(
            model_name="prediction",
            name="evidence_grade",
            field=models.CharField(
                choices=[
                    ("research", "Research-grade reconstruction"),
                    ("observed", "Observed at run time"),
                ],
                db_index=True,
                default="research",
                max_length=12,
            ),
        ),
        migrations.AddField(
            model_name="prediction",
            name="source_mode",
            field=models.CharField(
                choices=[
                    ("provider", "Provider-backed"),
                    ("synthetic", "Synthetic"),
                    ("unknown", "Unknown"),
                ],
                db_index=True,
                default="unknown",
                max_length=12,
            ),
        ),
        migrations.AddField(
            model_name="prediction",
            name="price_provider",
            field=models.CharField(blank=True, db_index=True, max_length=40),
        ),
        migrations.AddField(
            model_name="prediction",
            name="price_subject",
            field=models.CharField(blank=True, max_length=128),
        ),
        migrations.AddField(
            model_name="prediction",
            name="calculation",
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name="prediction",
            name="method_version",
            field=models.CharField(blank=True, db_index=True, max_length=40),
        ),
        migrations.RunPython(
            backfill_prediction_metadata,
            migrations.RunPython.noop,
        ),
        migrations.AddConstraint(
            model_name="prediction",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(
                        evidence_role="decision",
                        horizon__in=("short", "medium", "long"),
                    )
                    | models.Q(
                        evidence_role="advisory",
                        horizon__in=("6m", "12m", "3y", "5y"),
                    )
                ),
                name="prediction_horizon_role_valid",
            ),
        ),
        migrations.RunPython(
            _immutability.protect_predictions,
            _immutability.unprotect_predictions,
        ),
        migrations.AddField(
            model_name="predictionoutcome",
            name="direction_correct",
            field=models.BooleanField(null=True),
        ),
        migrations.AddField(
            model_name="predictionoutcome",
            name="interval_covered",
            field=models.BooleanField(null=True),
        ),
        migrations.AddField(
            model_name="predictionoutcome",
            name="signed_error",
            field=models.DecimalField(
                decimal_places=4,
                max_digits=10,
                null=True,
            ),
        ),
        migrations.RunPython(
            backfill_outcome_metrics,
            clear_outcome_metrics,
        ),
        migrations.RemoveConstraint(
            model_name="predictionoutcome",
            name="outcome_matured_actual_success",
        ),
        migrations.RemoveConstraint(
            model_name="predictionoutcome",
            name="outcome_unresolved_nulls",
        ),
        migrations.RemoveConstraint(
            model_name="predictionoutcome",
            name="outcome_corporate_event_nulls",
        ),
        migrations.AddConstraint(
            model_name="predictionoutcome",
            constraint=models.CheckConstraint(
                condition=(
                    ~models.Q(status="matured")
                    | models.Q(actual_return__isnull=False)
                ),
                name="outcome_matured_actual",
            ),
        ),
        migrations.AddConstraint(
            model_name="predictionoutcome",
            constraint=models.CheckConstraint(
                condition=(
                    ~models.Q(status="unresolved")
                    | models.Q(
                        actual_return__isnull=True,
                        benchmark_return__isnull=True,
                        success__isnull=True,
                        direction_correct__isnull=True,
                        interval_covered__isnull=True,
                        error__isnull=True,
                        signed_error__isnull=True,
                    )
                ),
                name="outcome_unresolved_nulls",
            ),
        ),
        migrations.AddConstraint(
            model_name="predictionoutcome",
            constraint=models.CheckConstraint(
                condition=(
                    ~models.Q(status="corporate_event")
                    | models.Q(
                        actual_return__isnull=True,
                        benchmark_return__isnull=True,
                        success__isnull=True,
                        direction_correct__isnull=True,
                        interval_covered__isnull=True,
                        error__isnull=True,
                        signed_error__isnull=True,
                    )
                ),
                name="outcome_corporate_event_nulls",
            ),
        ),
        migrations.RunPython(
            install_outcome_role_guards,
            remove_outcome_role_guards,
        ),
        migrations.RunPython(
            migrations.RunPython.noop,
            reject_unsafe_reverse,
        ),
    ]

from __future__ import annotations

from datetime import datetime
from typing import Any, cast
from uuid import uuid4

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from stanstock.research.config import code_revision
from stanstock.research.models import Prediction, StockAnalysis
from stanstock.research.scenarios import _scenario
from stanstock.research.service import AnalysisComputation, append_predictions
from stanstock.research.types import (
    AggregateScore,
    ComponentScores,
    IndicatorResult,
    ResearchValues,
    Scenario,
)


class Command(BaseCommand):
    help = "Append a new immutable prediction version from an existing persisted analysis."

    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--analysis", required=True, help="StockAnalysis primary key")
        parser.add_argument(
            "--model-version", help="Optional model version; defaults to timestamped run version"
        )

    def handle(self, *args: object, **options: object) -> None:
        try:
            analysis_id = str(options["analysis"])
            analysis = StockAnalysis.objects.select_related("run", "listing").get(
                pk=analysis_id,
            )
        except (ValueError, StockAnalysis.DoesNotExist) as exc:
            raise CommandError(str(exc)) from exc
        generated_at = timezone.now()
        model_version = options.get("model_version") or _prediction_model_version(
            str(analysis.run.config_version), generated_at
        )
        computation = _computation_from_analysis(analysis)
        predictions = append_predictions(
            analysis=analysis,
            computation=computation,
            generated_at=generated_at,
            data_cutoff=analysis.run.data_cutoff,
            issued_on_time=False,
            supported_horizons=_supported_horizons(analysis),
            model_version=str(model_version),
            config_hash_value=analysis.run.config_hash,
            source_assets=computation.source_assets,
            code_revision_value=code_revision(),
        )
        self.stdout.write(
            self.style.SUCCESS(
                "Appended predictions: "
                + ", ".join(str(prediction.pk) for prediction in predictions)
            )
        )


def _supported_horizons(analysis: StockAnalysis) -> tuple[str, ...]:
    data_quality = analysis.data_quality if isinstance(analysis.data_quality, dict) else {}
    raw_horizons = data_quality.get("supported_horizons")
    if raw_horizons is None:
        return tuple(Prediction.Horizon.values)
    if not isinstance(raw_horizons, list) or not raw_horizons:
        raise CommandError("Analysis supported_horizons must be a non-empty list")
    invalid = [
        value
        for value in raw_horizons
        if not isinstance(value, str) or value not in Prediction.Horizon.values
    ]
    if invalid:
        raise CommandError(f"Analysis contains invalid supported horizons: {invalid}")
    return tuple(raw_horizons)


def _computation_from_analysis(analysis: StockAnalysis) -> AnalysisComputation:
    component_payload = (
        analysis.component_scores if isinstance(analysis.component_scores, dict) else {}
    )
    components = component_payload.get("components", {})
    component_mapping = components if isinstance(components, dict) else {}
    horizons = component_payload.get(
        "horizons",
        {
            "short": float(analysis.overall_score),
            "medium": float(analysis.overall_score),
            "long": float(analysis.overall_score),
        },
    )
    horizon_mapping = horizons if isinstance(horizons, dict) else {}
    data_quality = analysis.data_quality if isinstance(analysis.data_quality, dict) else {}
    component_scores = ComponentScores(
        components={str(key): float(value) for key, value in component_mapping.items()},
        factor_scores={},
        missing={},
        coverage=float(data_quality.get("coverage", 0.0)),
    )
    aggregate = AggregateScore(
        overall=float(analysis.overall_score),
        horizon_scores={str(key): float(value) for key, value in horizon_mapping.items()},
        confidence=float(analysis.confidence),
        confidence_status=analysis.confidence_status,
        component_scores=component_scores,
        missingness_penalty=1.0,
        freshness_penalty=1.0,
    )
    scenarios: dict[str, Scenario] = {
        Prediction.Horizon.SHORT.value: _scenario_from_mapping(analysis.short_scenario),
        Prediction.Horizon.MEDIUM.value: _scenario_from_mapping(analysis.medium_scenario),
        Prediction.Horizon.LONG.value: _scenario_from_mapping(analysis.long_scenario),
    }
    return AnalysisComputation(
        indicators=IndicatorResult(values={}),
        fundamentals=ResearchValues(values={}),
        aggregate=aggregate,
        scenarios=scenarios,
        risk_score=float(analysis.risk_score) if analysis.risk_score is not None else None,
        risk_class=analysis.risk_class,
        recommendation=analysis.recommendation,
        reasons=list(analysis.reasons),
        risks=list(analysis.risks),
        data_quality=cast(dict[str, Any], data_quality),
        source_assets=_source_assets_from_analysis(analysis),
        current_price=float(analysis.current_price),
        daily_change=float(analysis.daily_change) if analysis.daily_change is not None else None,
    )


def _scenario_from_mapping(raw: object) -> Scenario:
    if not isinstance(raw, dict):
        raise CommandError("Analysis scenario payload is not a mapping")
    required = ("bear", "base", "bull")
    missing_keys = [key for key in required if key not in raw]
    if missing_keys:
        raise CommandError(f"Analysis scenario payload is incomplete: {', '.join(missing_keys)}")
    null_keys = [key for key in required if raw[key] is None]
    if len(null_keys) == len(required):
        return Scenario(
            bear=None,
            base=None,
            bull=None,
            probability_positive=None,
            confidence=float(raw.get("confidence", 0.0)),
            confidence_status=str(raw.get("confidence_status", "heuristic")),
            insufficiency_reason=str(raw.get("insufficiency_reason", "")),
            method=str(raw.get("method", "persisted_analysis")),
        )
    if null_keys:
        raise CommandError(f"Analysis scenario payload is incomplete: {', '.join(null_keys)}")
    return _scenario(
        float(raw["bear"]),
        float(raw["base"]),
        float(raw["bull"]),
        None if raw.get("probability_positive") is None else float(raw["probability_positive"]),
        float(raw.get("confidence", 0.0)),
        str(raw.get("insufficiency_reason", "")),
        str(raw.get("method", "persisted_analysis")),
    )


def _source_assets_from_analysis(analysis: StockAnalysis) -> list[dict[str, Any]]:
    data_quality = analysis.data_quality if isinstance(analysis.data_quality, dict) else {}
    if "source_assets" not in data_quality:
        raise CommandError("Analysis provenance is missing source assets")
    raw_assets = data_quality.get("source_assets", [])
    if not isinstance(raw_assets, list):
        raise CommandError("Analysis provenance is not a source asset list")
    assets: list[dict[str, Any]] = []
    required = (
        "id",
        "provider",
        "kind",
        "subject",
        "relative_path",
        "sha256",
        "retrieved_at",
        "available_at",
    )
    for item in raw_assets:
        if not isinstance(item, dict):
            raise CommandError("Analysis provenance contains a non-mapping source asset")
        if not all(isinstance(item.get(key), str) for key in required):
            raise CommandError("Analysis provenance source asset is incomplete")
        normalized: dict[str, Any] = {key: str(item[key]) for key in required}
        if isinstance(item.get("return_definition"), str):
            normalized["return_definition"] = item["return_definition"]
        if isinstance(item.get("dividends_included"), bool):
            normalized["dividends_included"] = item["dividends_included"]
        assets.append(normalized)
    return assets


def _prediction_model_version(config_version: str, generated_at: datetime) -> str:
    del generated_at
    suffix = uuid4().hex[:12]
    return f"{config_version[:27]}-{suffix}"

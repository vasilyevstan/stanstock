from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models.base import ModelBase

from stanstock.data.models import Listing, UniverseSnapshot
from stanstock.research.forecasting import scenario_from_document
from stanstock.research.price_product_config import (
    FHS_METHOD_VERSION,
    MOMENTUM_METHOD_VERSION,
    PRODUCT_VERSION,
)


class Recommendation(models.TextChoices):
    BUY = "buy", "BUY"
    HOLD = "hold", "HOLD"
    AVOID = "avoid", "AVOID"


class RiskClass(models.TextChoices):
    LOW = "low", "LOW"
    MEDIUM = "medium", "MEDIUM"
    HIGH = "high", "HIGH"
    VERY_HIGH = "very_high", "VERY HIGH"
    INSUFFICIENT = "insufficient", "INSUFFICIENT EVIDENCE"


class AnalysisRun(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    generated_at = models.DateTimeField()
    data_cutoff = models.DateTimeField()
    target_date = models.DateField()
    issued_on_time = models.BooleanField(default=False)
    universe_snapshot = models.ForeignKey(UniverseSnapshot, on_delete=models.PROTECT)
    config_version = models.CharField(max_length=40)
    config_hash = models.CharField(max_length=64)
    code_revision = models.CharField(max_length=64)
    status = models.CharField(max_length=20, default="complete")

    class Meta:
        ordering = ["-generated_at"]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(data_cutoff__lte=models.F("generated_at")),
                name="analysis_cutoff_before_generated",
            )
        ]

    def __str__(self) -> str:
        return f"{self.target_date}:{self.config_version}:{self.status}"

    def clean(self) -> None:
        super().clean()
        if not self.pk:
            return
        original = (
            type(self)
            .objects.filter(pk=self.pk)
            .values(
                "config_version",
                "config_hash",
            )
            .first()
        )
        if not original or not self.stocks.exists():
            return
        original_touches_product = original["config_version"] == PRODUCT_VERSION
        current_touches_product = self.config_version == PRODUCT_VERSION
        if original_touches_product or current_touches_product:
            if (
                original["config_version"] != self.config_version
                or original["config_hash"] != self.config_hash
            ):
                raise ValidationError(
                    "Research-product analysis runs with stock analyses cannot "
                    "change config_version or config_hash."
                )


class StockAnalysis(models.Model):
    run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE, related_name="stocks")
    listing = models.ForeignKey(Listing, on_delete=models.PROTECT)
    current_price = models.DecimalField(max_digits=20, decimal_places=6)
    daily_change = models.DecimalField(max_digits=12, decimal_places=6, null=True)
    overall_score = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    recommendation = models.CharField(  # noqa: DJ001 - unavailable is SQL NULL by contract
        max_length=8,
        choices=Recommendation,
        null=True,
    )
    risk_score = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    risk_class = models.CharField(max_length=12, choices=RiskClass)
    confidence = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    confidence_status = models.CharField(max_length=32, default="heuristic")
    component_scores = models.JSONField(default=dict)
    forecast_scenarios = models.JSONField(default=dict)
    short_scenario = models.JSONField(default=dict)
    medium_scenario = models.JSONField(default=dict)
    long_scenario = models.JSONField(default=dict)
    reasons = models.JSONField(default=list)
    risks = models.JSONField(default=list)
    data_quality = models.JSONField(default=dict)

    class Meta:
        ordering = ["-overall_score"]
        constraints = [
            models.UniqueConstraint(
                fields=["run", "listing"],
                name="unique_analysis_listing",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(overall_score__isnull=True)
                    | models.Q(overall_score__gte=0, overall_score__lte=100)
                ),
                name="analysis_score_in_range",
            ),
            models.CheckConstraint(
                condition=models.Q(risk_score__gte=0, risk_score__lte=100),
                name="analysis_risk_in_range",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(confidence__isnull=True)
                    | models.Q(confidence__gte=0, confidence__lte=100)
                ),
                name="analysis_confidence_in_range",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(
                        overall_score__isnull=True,
                        confidence__isnull=True,
                        confidence_status="not_estimated",
                    )
                    | models.Q(
                        overall_score__isnull=False,
                        confidence__isnull=False,
                        recommendation__isnull=False,
                    )
                ),
                name="analysis_nullable_shape_valid",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.run_id}:{self.listing_id}:{self.overall_score}"

    def clean(self) -> None:
        super().clean()
        if self.pk:
            original = type(self).objects.filter(pk=self.pk).values("run_id").first()
            if original and original["run_id"] != self.run_id:
                original_parent = (
                    AnalysisRun.objects.filter(pk=original["run_id"])
                    .values(
                        "config_version",
                    )
                    .first()
                )
                current_parent = (
                    AnalysisRun.objects.filter(pk=self.run_id)
                    .values(
                        "config_version",
                    )
                    .first()
                )
                if (original_parent and original_parent["config_version"] == PRODUCT_VERSION) or (
                    current_parent and current_parent["config_version"] == PRODUCT_VERSION
                ):
                    raise ValidationError(
                        {
                            "run": (
                                "Persisted stock analyses cannot be reparented "
                                "into or out of research-product-v1."
                            )
                        }
                    )
        if not self.run_id:
            return
        run = (
            AnalysisRun.objects.filter(pk=self.run_id)
            .values(
                "config_version",
                "config_hash",
            )
            .first()
        )
        if run is None:
            return
        prospective = run["config_version"] == PRODUCT_VERSION
        if prospective:
            errors: dict[str, str] = {}
            if self.overall_score is not None:
                errors["overall_score"] = "Prospective price research does not estimate a score."
            if self.risk_score is not None:
                errors["risk_score"] = "Prospective price research does not aggregate risk."
            if self.confidence is not None:
                errors["confidence"] = "Prospective price research does not estimate confidence."
            if self.confidence_status != "not_estimated":
                errors["confidence_status"] = (
                    "Prospective price research confidence_status must be not_estimated."
                )
            if self.recommendation is None:
                quality = self.data_quality if isinstance(self.data_quality, dict) else {}
                if not quality.get("momentum_insufficiency_reason"):
                    errors["recommendation"] = (
                        "An unavailable prospective recommendation requires a momentum "
                        "insufficiency reason."
                    )
            if errors:
                raise ValidationError(errors)
            return
        legacy_nulls = {
            field: "Legacy analyses require this value."
            for field in ("overall_score", "confidence", "recommendation")
            if getattr(self, field) is None
        }
        if legacy_nulls:
            raise ValidationError(legacy_nulls)

    def scenario_for_horizon(self, horizon: str) -> dict[str, Any]:
        return scenario_from_document(
            self.forecast_scenarios,
            horizon,
            legacy_fallbacks={
                "short": self.short_scenario,
                "medium": self.medium_scenario,
                "long": self.long_scenario,
            },
        )

    @property
    def short_forecast_scenario(self) -> dict[str, Any]:
        return self.scenario_for_horizon("short")

    @property
    def legacy_medium_forecast_scenario(self) -> dict[str, Any]:
        return self.scenario_for_horizon("medium")

    @property
    def legacy_long_forecast_scenario(self) -> dict[str, Any]:
        return self.scenario_for_horizon("long")

    @property
    def six_month_forecast_scenario(self) -> dict[str, Any]:
        return self.scenario_for_horizon("6m")

    @property
    def twelve_month_forecast_scenario(self) -> dict[str, Any]:
        return self.scenario_for_horizon("12m")

    @property
    def three_year_forecast_scenario(self) -> dict[str, Any]:
        return self.scenario_for_horizon("3y")

    @property
    def five_year_forecast_scenario(self) -> dict[str, Any]:
        return self.scenario_for_horizon("5y")

    @property
    def has_explicit_medium_forecasts(self) -> bool:
        return self._has_canonical_forecast("6m") or self._has_canonical_forecast("12m")

    @property
    def has_explicit_long_forecasts(self) -> bool:
        return self._has_canonical_forecast("3y") or self._has_canonical_forecast("5y")

    @property
    def has_explicit_advisory_forecasts(self) -> bool:
        return self.has_explicit_medium_forecasts or self.has_explicit_long_forecasts

    def _has_canonical_forecast(self, horizon: str) -> bool:
        if not isinstance(self.forecast_scenarios, dict):
            return False
        horizons = self.forecast_scenarios.get("horizons")
        return isinstance(horizons, dict) and isinstance(horizons.get(horizon), dict)


class Prediction(models.Model):
    class Horizon(models.TextChoices):
        SHORT = "short", "1-10 trading days"
        SIX_MONTH = "6m", "6 months"
        TWELVE_MONTH = "12m", "12 months"
        THREE_YEAR = "3y", "3 years"
        FIVE_YEAR = "5y", "5 years"
        MEDIUM = "medium", "Legacy 6-12 months"
        LONG = "long", "Legacy 3+ years"

    class EvidenceRole(models.TextChoices):
        DECISION = "decision", "Decision"
        ADVISORY = "advisory", "Advisory"

    class SourceMode(models.TextChoices):
        PROVIDER = "provider", "Provider-backed"
        SYNTHETIC = "synthetic", "Synthetic"
        UNKNOWN = "unknown", "Unknown"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    analysis = models.ForeignKey(StockAnalysis, on_delete=models.PROTECT)
    listing = models.ForeignKey(Listing, on_delete=models.PROTECT)
    generated_at = models.DateTimeField()
    target_date = models.DateField()
    issued_on_time = models.BooleanField(default=False)
    horizon = models.CharField(max_length=8, choices=Horizon)
    evidence_role = models.CharField(
        max_length=12,
        choices=EvidenceRole,
        default=EvidenceRole.DECISION,
        db_index=True,
    )
    evidence_grade = models.CharField(
        max_length=12,
        choices=UniverseSnapshot.Grade,
        default=UniverseSnapshot.Grade.RESEARCH,
        db_index=True,
    )
    source_mode = models.CharField(
        max_length=12,
        choices=SourceMode,
        default=SourceMode.UNKNOWN,
        db_index=True,
    )
    price_provider = models.CharField(max_length=40, blank=True, db_index=True)
    price_subject = models.CharField(max_length=128, blank=True)
    price_at_prediction = models.DecimalField(max_digits=20, decimal_places=6)
    bear_return = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    base_return = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    bull_return = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    probability_positive = models.DecimalField(max_digits=6, decimal_places=4, null=True)
    confidence = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    confidence_status = models.CharField(max_length=32)
    insufficiency_reason = models.CharField(max_length=240, blank=True)
    recommendation = models.CharField(  # noqa: DJ001 - inapplicable/unavailable is SQL NULL
        max_length=8,
        choices=Recommendation,
        null=True,
    )
    overall_score = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    component_scores = models.JSONField(default=dict)
    model_version = models.CharField(max_length=40)
    method_version = models.CharField(max_length=40, blank=True, db_index=True)
    config_hash = models.CharField(max_length=64)
    data_cutoff = models.DateTimeField()
    source_assets = models.JSONField(default=list)
    calculation = models.JSONField(default=dict)
    code_revision = models.CharField(max_length=64)

    class Meta:
        ordering = ["-generated_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["listing", "target_date", "horizon", "model_version"],
                name="unique_prediction_version",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(overall_score__isnull=True)
                    | models.Q(overall_score__gte=0, overall_score__lte=100)
                ),
                name="prediction_score_in_range",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(confidence__isnull=True)
                    | models.Q(confidence__gte=0, confidence__lte=100)
                ),
                name="prediction_confidence_in_range",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(probability_positive__isnull=True)
                    | models.Q(probability_positive__gte=0, probability_positive__lte=1)
                ),
                name="prediction_probability_in_range",
            ),
            models.CheckConstraint(
                condition=models.Q(price_at_prediction__gt=0),
                name="prediction_price_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(data_cutoff__lte=models.F("generated_at")),
                name="prediction_cutoff_before_generated",
            ),
            models.CheckConstraint(
                condition=(
                    (models.Q(bear_return__isnull=True) | models.Q(bear_return__gte=-1.0))
                    & (models.Q(base_return__isnull=True) | models.Q(base_return__gte=-1.0))
                    & (models.Q(bull_return__isnull=True) | models.Q(bull_return__gte=-1.0))
                ),
                name="prediction_return_lower_bound",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(
                        bear_return__isnull=True,
                        base_return__isnull=True,
                        bull_return__isnull=True,
                    )
                    | models.Q(
                        bear_return__isnull=False,
                        base_return__isnull=False,
                        bull_return__isnull=False,
                        bear_return__lte=models.F("base_return"),
                        base_return__lte=models.F("bull_return"),
                    )
                ),
                name="prediction_scenarios_ordered",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(
                        evidence_role="decision",
                        horizon__in=("short", "medium", "long"),
                    )
                    | models.Q(
                        evidence_role="decision",
                        horizon="6m",
                        method_version=MOMENTUM_METHOD_VERSION,
                    )
                    | models.Q(
                        evidence_role="advisory",
                        horizon__in=("6m", "12m", "3y", "5y"),
                    )
                ),
                name="prediction_horizon_role_valid",
            ),
            models.CheckConstraint(
                condition=(
                    models.Q(
                        method_version=MOMENTUM_METHOD_VERSION,
                        evidence_role="decision",
                        horizon="6m",
                        overall_score__isnull=True,
                        confidence__isnull=True,
                        confidence_status="not_estimated",
                        probability_positive__isnull=True,
                        bear_return__isnull=True,
                        base_return__isnull=True,
                        bull_return__isnull=True,
                    )
                    & (
                        models.Q(
                            recommendation__isnull=False,
                            recommendation__in=("buy", "hold", "avoid"),
                        )
                        | (
                            models.Q(recommendation__isnull=True)
                            & ~models.Q(insufficiency_reason="")
                        )
                    )
                    | models.Q(
                        method_version=FHS_METHOD_VERSION,
                        evidence_role="advisory",
                        horizon__in=("6m", "12m", "3y", "5y"),
                        overall_score__isnull=True,
                        confidence__isnull=True,
                        confidence_status="not_estimated",
                        recommendation__isnull=True,
                        probability_positive__isnull=True,
                    )
                    | (
                        ~models.Q(
                            method_version__in=(
                                MOMENTUM_METHOD_VERSION,
                                FHS_METHOD_VERSION,
                            )
                        )
                        & models.Q(
                            overall_score__isnull=False,
                            confidence__isnull=False,
                            recommendation__isnull=False,
                        )
                    )
                ),
                name="prediction_prospective_shape_valid",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.listing_id}:{self.target_date}:{self.horizon}:{self.model_version}"

    def save(
        self,
        *,
        force_insert: bool | tuple[ModelBase, ...] = False,
        force_update: bool = False,
        using: str | None = None,
        update_fields: Iterable[str] | None = None,
    ) -> None:
        if self.pk and type(self).objects.filter(pk=self.pk).exists():
            raise ValidationError("Predictions are immutable; append a new version")
        super().save(
            force_insert=force_insert,
            force_update=force_update,
            using=using,
            update_fields=update_fields,
        )

    def clean(self) -> None:
        super().clean()
        if not self.analysis_id:
            return
        run = (
            StockAnalysis.objects.filter(pk=self.analysis_id)
            .values(
                "run__config_version",
                "run__config_hash",
            )
            .first()
        )
        if run is None:
            return
        run_version = run["run__config_version"]
        prospective_method = self.method_version in {
            MOMENTUM_METHOD_VERSION,
            FHS_METHOD_VERSION,
        }
        if run_version != PRODUCT_VERSION:
            if prospective_method:
                raise ValidationError(
                    {"method_version": "Prospective methods require research-product-v1."}
                )
            legacy_nulls = {
                field: "Legacy predictions require this value."
                for field in ("overall_score", "confidence", "recommendation")
                if getattr(self, field) is None
            }
            if legacy_nulls:
                raise ValidationError(legacy_nulls)
            return
        if not prospective_method:
            raise ValidationError(
                {"method_version": "research-product-v1 permits only its reviewed methods."}
            )
        if self.config_hash != run["run__config_hash"]:
            raise ValidationError(
                {
                    "config_hash": (
                        "Prospective predictions must match the owning AnalysisRun config_hash."
                    )
                }
            )
        if self.overall_score is not None or self.confidence is not None:
            raise ValidationError(
                "Prospective price research does not estimate score or confidence."
            )
        if self.confidence_status != "not_estimated" or self.probability_positive is not None:
            raise ValidationError("Prospective confidence and probability must remain unestimated.")
        if self.method_version == FHS_METHOD_VERSION and self.recommendation is not None:
            raise ValidationError(
                {"recommendation": "Advisory FHS projections have no recommendation."}
            )
        if (
            self.method_version == MOMENTUM_METHOD_VERSION
            and self.recommendation is None
            and not self.insufficiency_reason
        ):
            raise ValidationError(
                {"recommendation": "Unavailable momentum requires an insufficiency reason."}
            )

    def delete(
        self,
        using: Any | None = None,
        keep_parents: bool = False,
    ) -> tuple[int, dict[str, int]]:
        raise ValidationError("Predictions are immutable")


class PredictionOutcome(models.Model):
    class Status(models.TextChoices):
        MATURED = "matured", "Matured"
        UNRESOLVED = "unresolved", "Unresolved"
        CORPORATE_EVENT = "corporate_event", "Corporate event"

    prediction = models.OneToOneField(
        Prediction,
        on_delete=models.PROTECT,
        primary_key=True,
        related_name="outcome",
    )
    evaluated_at = models.DateTimeField()
    evaluation_date = models.DateField()
    status = models.CharField(max_length=20, choices=Status)
    actual_return = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    benchmark_return = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    success = models.BooleanField(null=True)
    direction_correct = models.BooleanField(null=True)
    interval_covered = models.BooleanField(null=True)
    resolution = models.CharField(max_length=120)
    error = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    signed_error = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=(~models.Q(status="matured") | models.Q(actual_return__isnull=False)),
                name="outcome_matured_actual",
            ),
            models.CheckConstraint(
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
            models.CheckConstraint(
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
        ]

    def __str__(self) -> str:
        return f"{self.prediction_id}:{self.status}:{self.evaluation_date}"

    def clean(self) -> None:
        super().clean()
        if self.status != self.Status.MATURED or not self.prediction_id:
            return
        if self.prediction.evidence_role == Prediction.EvidenceRole.DECISION:
            prospective_non_directional = (
                self.prediction.method_version == MOMENTUM_METHOD_VERSION
                and self.prediction.recommendation in (Recommendation.HOLD, None)
            )
            if prospective_non_directional and self.success is not None:
                raise ValidationError(
                    {"success": "Prospective HOLD/unavailable outcomes must not claim success."}
                )
            if not prospective_non_directional and self.success is None:
                message = (
                    "Matured directional decision outcomes require success."
                    if self.prediction.method_version == MOMENTUM_METHOD_VERSION
                    else "Matured decision outcomes require a success value."
                )
                raise ValidationError({"success": message})
        if (
            self.prediction.evidence_role == Prediction.EvidenceRole.ADVISORY
            and self.success is not None
        ):
            raise ValidationError(
                {"success": "Matured advisory outcomes must not use decision success."}
            )

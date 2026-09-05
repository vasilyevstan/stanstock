from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models.base import ModelBase

from stanstock.data.models import Listing, UniverseSnapshot


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


class StockAnalysis(models.Model):
    run = models.ForeignKey(AnalysisRun, on_delete=models.CASCADE, related_name="stocks")
    listing = models.ForeignKey(Listing, on_delete=models.PROTECT)
    current_price = models.DecimalField(max_digits=20, decimal_places=6)
    daily_change = models.DecimalField(max_digits=12, decimal_places=6, null=True)
    overall_score = models.DecimalField(max_digits=6, decimal_places=2)
    recommendation = models.CharField(max_length=8, choices=Recommendation)
    risk_score = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    risk_class = models.CharField(max_length=12, choices=RiskClass)
    confidence = models.DecimalField(max_digits=6, decimal_places=2)
    confidence_status = models.CharField(max_length=32, default="heuristic")
    component_scores = models.JSONField(default=dict)
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
                condition=models.Q(overall_score__gte=0, overall_score__lte=100),
                name="analysis_score_in_range",
            ),
            models.CheckConstraint(
                condition=models.Q(risk_score__gte=0, risk_score__lte=100),
                name="analysis_risk_in_range",
            ),
            models.CheckConstraint(
                condition=models.Q(confidence__gte=0, confidence__lte=100),
                name="analysis_confidence_in_range",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.run_id}:{self.listing_id}:{self.overall_score}"


class Prediction(models.Model):
    class Horizon(models.TextChoices):
        SHORT = "short", "1-10 trading days"
        MEDIUM = "medium", "6-12 months"
        LONG = "long", "3+ years"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    analysis = models.ForeignKey(StockAnalysis, on_delete=models.PROTECT)
    listing = models.ForeignKey(Listing, on_delete=models.PROTECT)
    generated_at = models.DateTimeField()
    target_date = models.DateField()
    horizon = models.CharField(max_length=8, choices=Horizon)
    price_at_prediction = models.DecimalField(max_digits=20, decimal_places=6)
    bear_return = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    base_return = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    bull_return = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    probability_positive = models.DecimalField(max_digits=6, decimal_places=4, null=True)
    confidence = models.DecimalField(max_digits=6, decimal_places=2)
    confidence_status = models.CharField(max_length=32)
    insufficiency_reason = models.CharField(max_length=240, blank=True)
    recommendation = models.CharField(max_length=8, choices=Recommendation)
    overall_score = models.DecimalField(max_digits=6, decimal_places=2)
    component_scores = models.JSONField(default=dict)
    model_version = models.CharField(max_length=40)
    config_hash = models.CharField(max_length=64)
    data_cutoff = models.DateTimeField()
    source_assets = models.JSONField(default=list)
    code_revision = models.CharField(max_length=64)

    class Meta:
        ordering = ["-generated_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["listing", "target_date", "horizon", "model_version"],
                name="unique_prediction_version",
            ),
            models.CheckConstraint(
                condition=models.Q(overall_score__gte=0, overall_score__lte=100),
                name="prediction_score_in_range",
            ),
            models.CheckConstraint(
                condition=models.Q(confidence__gte=0, confidence__lte=100),
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
    resolution = models.CharField(max_length=120)
    error = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=(
                    ~models.Q(status="matured")
                    | (models.Q(actual_return__isnull=False) & models.Q(success__isnull=False))
                ),
                name="outcome_matured_actual_success",
            ),
            models.CheckConstraint(
                condition=(
                    ~models.Q(status="unresolved")
                    | models.Q(
                        actual_return__isnull=True,
                        benchmark_return__isnull=True,
                        success__isnull=True,
                        error__isnull=True,
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
                        error__isnull=True,
                    )
                ),
                name="outcome_corporate_event_nulls",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.prediction_id}:{self.status}:{self.evaluation_date}"

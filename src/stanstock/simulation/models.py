from __future__ import annotations

import uuid

from django.db import models

from stanstock.data.models import Listing, UniverseSnapshot


class SimulationDefinition(models.Model):
    class Mode(models.TextChoices):
        BACKTEST = "backtest", "Backtest"
        PORTFOLIO = "portfolio", "Portfolio simulation"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=160)
    mode = models.CharField(max_length=12, choices=Mode)
    config = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.name} ({self.mode})"


class SimulationRun(models.Model):
    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        COMPLETE = "complete", "Complete"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    definition = models.ForeignKey(
        SimulationDefinition,
        on_delete=models.PROTECT,
        related_name="runs",
    )
    universe_snapshot = models.ForeignKey(UniverseSnapshot, on_delete=models.PROTECT)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=12, choices=Status, default=Status.RUNNING)
    code_revision = models.CharField(max_length=64)
    input_hash = models.CharField(max_length=64)
    result_asset_key = models.CharField(max_length=500, blank=True)
    metrics = models.JSONField(default=dict)
    error = models.TextField(blank=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=(
                    models.Q(status="running", finished_at__isnull=True)
                    | (~models.Q(status="running") & models.Q(finished_at__isnull=False))
                ),
                name="simulation_finished_at_matches_status",
            ),
            models.CheckConstraint(
                condition=(~models.Q(status="complete") | ~models.Q(result_asset_key="")),
                name="complete_simulation_has_result_asset",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.definition_id}:{self.started_at.isoformat()}:{self.status}"


class SimulationHolding(models.Model):
    run = models.ForeignKey(SimulationRun, on_delete=models.CASCADE, related_name="holdings")
    listing = models.ForeignKey(Listing, on_delete=models.PROTECT)
    observation_date = models.DateField()
    quantity = models.DecimalField(max_digits=24, decimal_places=8)
    price = models.DecimalField(max_digits=20, decimal_places=6)
    market_value = models.DecimalField(max_digits=24, decimal_places=6)
    weight = models.DecimalField(max_digits=12, decimal_places=8)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["run", "listing", "observation_date"],
                name="unique_simulation_holding",
            ),
            models.CheckConstraint(
                condition=models.Q(quantity__gte=0),
                name="simulation_holding_quantity_nonnegative",
            ),
            models.CheckConstraint(
                condition=models.Q(price__gte=0),
                name="simulation_holding_price_nonnegative",
            ),
            models.CheckConstraint(
                condition=models.Q(market_value__gte=0),
                name="simulation_holding_value_nonnegative",
            ),
            models.CheckConstraint(
                condition=models.Q(weight__gte=0, weight__lte=1),
                name="simulation_holding_weight_in_range",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.run_id}:{self.listing_id}:{self.observation_date}"


class SimulationTrade(models.Model):
    class Side(models.TextChoices):
        BUY = "buy", "Buy"
        SELL = "sell", "Sell"

    run = models.ForeignKey(SimulationRun, on_delete=models.CASCADE, related_name="trades")
    listing = models.ForeignKey(Listing, on_delete=models.PROTECT)
    trade_date = models.DateField()
    side = models.CharField(max_length=4, choices=Side)
    quantity = models.DecimalField(max_digits=24, decimal_places=8)
    price = models.DecimalField(max_digits=20, decimal_places=6)
    gross_value = models.DecimalField(max_digits=24, decimal_places=6)
    costs = models.DecimalField(max_digits=20, decimal_places=6, default=0)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=models.Q(quantity__gt=0),
                name="simulation_trade_quantity_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(price__gt=0),
                name="simulation_trade_price_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(gross_value__gt=0),
                name="simulation_trade_value_positive",
            ),
            models.CheckConstraint(
                condition=models.Q(costs__gte=0),
                name="simulation_trade_costs_nonnegative",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.run_id}:{self.listing_id}:{self.trade_date}:{self.side}"
